"""Dual-hand tracker: persistent IDs + left FSM + calibrated right mouse.

After cv2.flip selfie mirror, MediaPipe handedness is ignored. Roles are locked
by nearest-neighbor wrist matching across frames so close hands do not swap.
LEFT never drives mouse; RIGHT never writes text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.paths import resource_root
from src.vision.gesture_recognizer import GestureEvent, LeftHandGestureRecognizer, LeftGestureState
from src.vision.hand_calibrator import (
    HandCalibrator,
    HandProfile,
    default_profile,
    load_profile,
)
from src.vision.one_euro import OneEuroFilter2D
from src.vision.stroke_collector import StrokeCollector, is_index_writing_pose, live_hand_scale

REPO_ROOT = resource_root()
DEFAULT_MODEL = REPO_ROOT / "data" / "models" / "hand_landmarker.task"

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17

CLICK_FREEZE_S = 0.100
RECOG_FLASH_S = 0.500
LMB_PRESS_RATIO = 0.36
LMB_RELEASE_RATIO = 0.44
_GESTURE_STATES = {
    LeftGestureState.FIST,
    LeftGestureState.SWIPE_LEFT,
    LeftGestureState.SWIPE_RIGHT,
    LeftGestureState.V_SIGN,
    LeftGestureState.ENTER,
}

# Preview tags (OpenCV BGR): blue = writer, orange = mouse
COLOR_LEFT_BGR = (255, 140, 0)
COLOR_RIGHT_BGR = (0, 140, 255)

# Right-index status rings (BGR)
RING_HOVER = (80, 220, 80)  # green
RING_LMB = (40, 40, 240)  # red
RING_RMB = (0, 220, 255)  # yellow
RING_SCROLL = (255, 160, 40)  # blue


class MouseMode(str, Enum):
    HOVER = "HOVER"
    LMB = "LMB"
    RMB = "RMB"
    SCROLL = "SCROLL"


@dataclass
class DualHandCallbacks:
    on_mouse_move: Callable[[float, float], None] | None = None
    on_left_down: Callable[[], None] | None = None
    on_left_up: Callable[[], None] | None = None
    on_right_down: Callable[[], None] | None = None
    on_right_up: Callable[[], None] | None = None
    # Legacy aliases (instant click) — preferred path is down/up hysteresis.
    on_left_click: Callable[[], None] | None = None
    on_right_click: Callable[[], None] | None = None
    on_scroll: Callable[[int], None] | None = None
    # on_stroke(points[N,2], timestamps[N] | None)
    on_stroke: Callable[..., None] | None = None
    on_fist_tab: Callable[[], None] | None = None
    on_swipe_left: Callable[[], None] | None = None
    on_swipe_right: Callable[[], None] | None = None
    on_lang_switch: Callable[[], None] | None = None
    on_enter: Callable[[], None] | None = None


@dataclass
class DebugHUD:
    left_state: str = "IDLE"
    left_badge: str = "[STATE: IDLE]"
    left_badge_color: tuple[int, int, int] | None = None
    left_points: int = 0
    left_present: bool = False
    right_present: bool = False
    right_xy: tuple[float, float] | None = None
    os_lang: str = "EN"
    last_char: str = ""
    last_conf: float = 0.0
    recog_badge: str = ""
    recog_flash_until: float = 0.0
    trail: list[tuple[float, float]] = field(default_factory=list)
    mode_label: str = "Writing"
    calib_status: str = "Uncalibrated"
    mouse_mode: str = "HOVER"
    left_status: str = "READY"
    right_lmb: bool = False
    right_rmb: bool = False


@dataclass
class _RightState:
    tip_filter: OneEuroFilter2D | None = None
    pinch_left: bool = False
    pinch_right: bool = False
    last_scroll_y: float | None = None
    freeze_until: float = 0.0
    frozen_xy: tuple[float, float] | None = None
    scroll_lock_xy: tuple[float, float] | None = None
    mouse_mode: MouseMode = MouseMode.HOVER


@dataclass
class _LeftWriteState:
    state: LeftGestureState = LeftGestureState.IDLE


class DualHandTracker:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        camera_index: int = 0,
        mirror: bool = True,
        callbacks: DualHandCallbacks | None = None,
        show_preview: bool = False,
        writing_enabled: bool = True,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Hand landmarker model missing: {model_path}")
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.camera_index = camera_index
        self.mirror = bool(mirror)
        self.callbacks = callbacks or DualHandCallbacks()
        self.show_preview = show_preview
        self.writing_enabled = writing_enabled
        self.hud = DebugHUD()
        self.mode_label = "Writing"
        self._cap: cv2.VideoCapture | None = None
        self._right = _RightState()
        self._left = _LeftWriteState()
        self._gestures = LeftHandGestureRecognizer()
        self._strokes = StrokeCollector()
        self._left_wrist: np.ndarray | None = None
        self._right_wrist: np.ndarray | None = None
        self._frame_ts_ms = 0
        self._running = False

        self.calibrator = HandCalibrator()
        loaded = load_profile()
        if loaded is None:
            # Headless: seed defaults so mouse works; preview will force interactive calib.
            if not show_preview:
                self.apply_profile(default_profile())
                self.hud.calib_status = "Default profile (press C in preview to calibrate)"
            else:
                self.apply_profile(default_profile())
                self.hud.calib_status = "Needs calibration"
                self.calibrator.start()
        else:
            self.apply_profile(loaded)
            self.hud.calib_status = "Calibrated"

    def apply_profile(self, profile: HandProfile) -> None:
        self.profile = profile
        self._right.tip_filter = OneEuroFilter2D(
            min_cutoff=float(profile.one_euro_min_cutoff),
            beta=float(profile.one_euro_beta),
        )
        self._gestures.set_profile(profile)
        self._strokes.set_profile(profile)
        self.hud.calib_status = (
            f"Calibrated  palm={profile.palm_base_scale:.3f}  "
            f"pinchL={profile.pinch_threshold_lmb:.3f}  "
            f"1€={profile.one_euro_min_cutoff:.3f}"
        )

    def set_writing_enabled(self, enabled: bool) -> None:
        self.writing_enabled = bool(enabled)
        if not enabled:
            self._clear_stroke()
            if self._left.state == LeftGestureState.DRAWING:
                self._left.state = LeftGestureState.IDLE

    def set_os_lang(self, lang: str) -> None:
        self.hud.os_lang = str(lang).upper()

    def set_last_recognition(self, label: str, conf: float, *, flash: bool = True) -> None:
        self.hud.last_char = label
        self.hud.last_conf = float(conf)
        if flash and label:
            pct = 100.0 * float(conf)
            self.hud.recog_badge = f"[RECOGNIZED: '{label}' ({pct:.1f}%)]"
            self.hud.recog_flash_until = time.perf_counter() + RECOG_FLASH_S
        elif not flash:
            self.hud.recog_badge = ""
            self.hud.recog_flash_until = 0.0

    def open(self) -> None:
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        self._cap = cap
        self._running = True

    def close(self) -> None:
        self._release_mouse_buttons()
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self.show_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        self._landmarker.close()

    def spin_forever(self) -> None:
        self.open()
        try:
            while self._running:
                if not self.step():
                    break
        finally:
            self.close()

    def step(self) -> bool:
        if self._cap is None:
            return False
        ok, frame = self._cap.read()
        if not ok:
            return False
        # Mirror first, then MediaPipe on the mirrored frame.
        if self.mirror:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)
        now = time.perf_counter()

        left_lm, right_lm = self._match_hands(result)
        self.hud.left_present = left_lm is not None
        self.hud.right_present = right_lm is not None
        self.hud.mode_label = self.mode_label

        # Prefer any visible hand for calibration samples (open palm guide).
        calib_lm = right_lm if right_lm is not None else left_lm
        if self.calibrator.active:
            finished = self.calibrator.update(calib_lm)
            if finished is not None:
                self.apply_profile(finished)
            self._release_mouse_buttons()
            self.hud.right_xy = None
            self.hud.mouse_mode = "CALIB"
            self.hud.calib_status = self.calibrator.status_line
        else:
            # Keep failure message visible until a successful run.
            if self.calibrator._last_error:
                self.hud.calib_status = self.calibrator.status_line

            # RIGHT_HAND only → mouse. Never text.
            if right_lm is not None:
                self._handle_right(right_lm, now)
            else:
                self._release_mouse_buttons()
                self.hud.right_xy = None
                self._right.mouse_mode = MouseMode.HOVER

            # LEFT_HAND only → text/gestures. Never mouse.
            if left_lm is not None:
                self._handle_left(left_lm, now)
            else:
                self._gestures.update(None, now, writing_enabled=self.writing_enabled)
                self._clear_stroke()
                self._left.state = LeftGestureState.IDLE
                self.hud.left_badge = "[STATE: IDLE]"
                self.hud.left_badge_color = None

        self.hud.left_state = self._format_left_state()
        self.hud.left_status = self._left_hud_status()
        self.hud.left_points = self._strokes.point_count
        self.hud.trail = self._strokes.trail_abs
        self.hud.mouse_mode = self._right.mouse_mode.value
        self.hud.right_lmb = bool(self._right.pinch_left)
        self.hud.right_rmb = bool(self._right.pinch_right)
        if self.hud.recog_flash_until and time.perf_counter() >= self.hud.recog_flash_until:
            self.hud.recog_badge = ""
            self.hud.recog_flash_until = 0.0

        if self.show_preview:
            self._draw_preview(frame, left_lm, right_lm)
            if self.calibrator.active:
                self.calibrator.draw_overlay(frame)
            cv2.imshow("AirTouch", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                self._running = False
                return False
            if key in (ord("c"), ord("C")) and not self.calibrator.active:
                self._release_mouse_buttons()
                self.calibrator.start()
                self.hud.calib_status = self.calibrator.status_line
        return True

    def _match_hands(
        self, result: mp_vision.HandLandmarkerResult
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Lock LEFT/RIGHT IDs by nearest-neighbor wrist tracking (no x<0.5 flip)."""
        dets: list[tuple[np.ndarray, np.ndarray]] = []
        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
                dets.append((pts[WRIST, :2].copy(), pts))

        if not dets:
            self._left_wrist = None
            self._right_wrist = None
            return None, None

        prev_l = self._left_wrist
        prev_r = self._right_wrist
        left: np.ndarray | None = None
        right: np.ndarray | None = None

        def _d(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.linalg.norm(a - b))

        if len(dets) == 1:
            w, pts = dets[0]
            if prev_l is None and prev_r is None:
                if float(w[0]) < 0.5:
                    left = pts
                else:
                    right = pts
            elif prev_l is not None and prev_r is None:
                left = pts
            elif prev_r is not None and prev_l is None:
                right = pts
            else:
                assert prev_l is not None and prev_r is not None
                if _d(w, prev_l) <= _d(w, prev_r):
                    left = pts
                else:
                    right = pts
        else:
            w0, p0 = dets[0]
            w1, p1 = dets[1]
            if prev_l is not None and prev_r is not None:
                keep = _d(w0, prev_l) + _d(w1, prev_r)
                swap = _d(w0, prev_r) + _d(w1, prev_l)
                if keep <= swap:
                    left, right = p0, p1
                else:
                    left, right = p1, p0
            elif prev_l is not None:
                if _d(w0, prev_l) <= _d(w1, prev_l):
                    left, right = p0, p1
                else:
                    left, right = p1, p0
            elif prev_r is not None:
                if _d(w0, prev_r) <= _d(w1, prev_r):
                    right, left = p0, p1
                else:
                    right, left = p1, p0
            else:
                if float(w0[0]) <= float(w1[0]):
                    left, right = p0, p1
                else:
                    left, right = p1, p0

        self._left_wrist = left[WRIST, :2].copy() if left is not None else None
        self._right_wrist = right[WRIST, :2].copy() if right is not None else None
        return left, right

    def _dist(self, lm: np.ndarray, a: int, b: int) -> float:
        return float(np.linalg.norm(lm[a, :2] - lm[b, :2]))

    def _index_extended(self, lm: np.ndarray) -> bool:
        wrist = lm[WRIST, :2]
        return float(np.linalg.norm(lm[INDEX_TIP, :2] - wrist)) > float(
            np.linalg.norm(lm[INDEX_MCP, :2] - wrist)
        ) * 1.15

    def _middle_extended(self, lm: np.ndarray) -> bool:
        wrist = lm[WRIST, :2]
        return float(np.linalg.norm(lm[MIDDLE_TIP, :2] - wrist)) > float(
            np.linalg.norm(lm[MIDDLE_MCP, :2] - wrist)
        ) * 1.15

    def _clear_stroke(self) -> None:
        """Purge character stroke buffer so gestures never enter recognition."""
        self._strokes.purge()

    def _left_hud_status(self) -> str:
        now = time.perf_counter()
        if self.hud.recog_badge and now < self.hud.recog_flash_until:
            return "RECOGNIZED"
        if self._left.state in _GESTURE_STATES:
            return "GESTURE"
        if self._strokes.is_drawing:
            return "DRAWING"
        return "READY"

    def _format_left_state(self) -> str:
        st = self._left.state
        if st == LeftGestureState.DRAWING or self._strokes.is_drawing:
            return f"DRAWING ({self._strokes.point_count})"
        return st.value

    def _emit_stroke(self, points: np.ndarray, timestamps: np.ndarray) -> None:
        """Flip Y to model frame (y-up) and hand off to classifier callback."""
        model_pts = points.copy()
        model_pts[:, 1] = 1.0 - model_pts[:, 1]
        if self.callbacks.on_stroke:
            try:
                self.callbacks.on_stroke(model_pts, timestamps)
            except TypeError:
                self.callbacks.on_stroke(model_pts)

    def _dispatch_gesture(self, event: GestureEvent) -> None:
        cb = self.callbacks
        if event == GestureEvent.FIST_TAB and cb.on_fist_tab:
            cb.on_fist_tab()
        elif event == GestureEvent.SWIPE_LEFT and cb.on_swipe_left:
            cb.on_swipe_left()
        elif event == GestureEvent.SWIPE_RIGHT and cb.on_swipe_right:
            cb.on_swipe_right()
        elif event == GestureEvent.V_SIGN_LANG and cb.on_lang_switch:
            cb.on_lang_switch()
        elif event == GestureEvent.ENTER and cb.on_enter:
            cb.on_enter()

    def _map_index_to_screen(self, x: float, y: float) -> tuple[float, float]:
        """Map index tip through calibrated active margins [lo, hi] → primary screen [0,1]."""
        lo = float(self.profile.active_margin_lo)
        hi = float(self.profile.active_margin_hi)
        span = max(hi - lo, 1e-6)
        nx = (float(x) - lo) / span
        ny = (float(y) - lo) / span
        return (
            float(min(max(nx, 0.0), 1.0)),
            float(min(max(ny, 0.0), 1.0)),
        )

    def _fire_left_down(self) -> None:
        if self.callbacks.on_left_down:
            self.callbacks.on_left_down()
        elif self.callbacks.on_left_click:
            self.callbacks.on_left_click()

    def _fire_left_up(self) -> None:
        if self.callbacks.on_left_up:
            self.callbacks.on_left_up()

    def _fire_right_down(self) -> None:
        if self.callbacks.on_right_down:
            self.callbacks.on_right_down()
        elif self.callbacks.on_right_click:
            self.callbacks.on_right_click()

    def _fire_right_up(self) -> None:
        if self.callbacks.on_right_up:
            self.callbacks.on_right_up()

    def _release_mouse_buttons(self) -> None:
        st = self._right
        if st.pinch_left:
            st.pinch_left = False
            self._fire_left_up()
        if st.pinch_right:
            st.pinch_right = False
            self._fire_right_up()
        st.last_scroll_y = None
        st.scroll_lock_xy = None
        st.freeze_until = 0.0
        st.frozen_xy = None
        st.mouse_mode = MouseMode.HOVER

    def _handle_right(self, lm: np.ndarray, now: float) -> None:
        """RIGHT_HAND — calibrated margins, pinch hysteresis, click-freeze, two-finger scroll."""
        st = self._right
        if st.tip_filter is None:
            self.apply_profile(self.profile)

        raw_x, raw_y = float(lm[INDEX_TIP, 0]), float(lm[INDEX_TIP, 1])
        fx, fy = st.tip_filter(raw_x, raw_y, now)  # type: ignore[misc]
        mx, my = self._map_index_to_screen(fx, fy)

        hs = live_hand_scale(lm)
        d_lmb = self._dist(lm, THUMB_TIP, INDEX_TIP)
        d_rmb = self._dist(lm, THUMB_TIP, MIDDLE_TIP)
        d_im = self._dist(lm, INDEX_TIP, MIDDLE_TIP)
        lmb_ratio = d_lmb / hs
        rmb_ratio = d_rmb / hs
        thr_r = float(self.profile.pinch_threshold_rmb)
        scroll_align = 0.35 * hs

        # Two-finger vertical scroll (index + middle extended & aligned)
        scrolling = (
            self._index_extended(lm)
            and self._middle_extended(lm)
            and d_im < scroll_align
            and lmb_ratio > LMB_RELEASE_RATIO
            and d_rmb > thr_r * 1.25
        )

        if scrolling:
            # Release any held buttons; lock cursor translation.
            if st.pinch_left:
                st.pinch_left = False
                self._fire_left_up()
            if st.pinch_right:
                st.pinch_right = False
                self._fire_right_up()
            st.mouse_mode = MouseMode.SCROLL
            mid_y = 0.5 * (float(lm[INDEX_TIP, 1]) + float(lm[MIDDLE_TIP, 1]))
            if st.scroll_lock_xy is None:
                st.scroll_lock_xy = (mx, my)
            lock_xy = st.scroll_lock_xy
            self.hud.right_xy = lock_xy
            if self.callbacks.on_mouse_move and lock_xy is not None:
                self.callbacks.on_mouse_move(lock_xy[0], lock_xy[1])
            if st.last_scroll_y is not None:
                dy = mid_y - st.last_scroll_y
                if abs(dy) > 0.010 and self.callbacks.on_scroll:
                    # Screen Y down → scroll down (negative wheel on Windows feels natural inverted)
                    self.callbacks.on_scroll(-1 if dy > 0 else 1)
                    st.last_scroll_y = mid_y
            else:
                st.last_scroll_y = mid_y
            return

        st.last_scroll_y = None
        st.scroll_lock_xy = None

        # LMB: Thumb + Index — press < 0.36·scale, release > 0.44·scale
        if not st.pinch_left and lmb_ratio < LMB_PRESS_RATIO:
            st.pinch_left = True
            st.freeze_until = now + CLICK_FREEZE_S
            st.frozen_xy = (mx, my)
            st.mouse_mode = MouseMode.LMB
            self._fire_left_down()
        elif st.pinch_left and lmb_ratio > LMB_RELEASE_RATIO:
            st.pinch_left = False
            self._fire_left_up()

        # RMB: Thumb + Middle when index is free (not in LMB pinch)
        index_free = not st.pinch_left and lmb_ratio > LMB_RELEASE_RATIO
        if index_free and not st.pinch_right and d_rmb < thr_r:
            st.pinch_right = True
            st.freeze_until = now + CLICK_FREEZE_S
            st.frozen_xy = (mx, my)
            st.mouse_mode = MouseMode.RMB
            self._fire_right_down()
        elif st.pinch_right and (d_rmb > thr_r * 1.25 or rmb_ratio > LMB_RELEASE_RATIO):
            st.pinch_right = False
            self._fire_right_up()

        if st.pinch_left:
            st.mouse_mode = MouseMode.LMB
        elif st.pinch_right:
            st.mouse_mode = MouseMode.RMB
        else:
            st.mouse_mode = MouseMode.HOVER

        # Click-freeze: hold output cursor for 100ms after press
        if now < st.freeze_until and st.frozen_xy is not None:
            out_xy = st.frozen_xy
        else:
            out_xy = (mx, my)
            st.frozen_xy = None

        self.hud.right_xy = out_xy
        if self.callbacks.on_mouse_move:
            self.callbacks.on_mouse_move(out_xy[0], out_xy[1])

    def _handle_left(self, lm: np.ndarray, now: float) -> None:
        """LEFT_HAND — priority FIST → SWIPE → V-SIGN → ENTER → AIR-WRITING / IDLE."""
        buf = self._left
        idx_ext = is_index_writing_pose(lm)
        result = self._gestures.update(
            lm,
            now,
            writing_enabled=self.writing_enabled,
            index_extended=idx_ext,
            stroke_points=self._strokes.point_count,
        )
        buf.state = result.state
        self.hud.left_badge = result.badge
        self.hud.left_badge_color = result.flash_bgr

        if result.purge_stroke or result.event != GestureEvent.NONE:
            self._clear_stroke()

        if result.event != GestureEvent.NONE:
            self._dispatch_gesture(result.event)
            self.hud.left_status = "GESTURE"
            return

        completed = self._strokes.update(
            lm,
            now,
            index_extended=idx_ext,
            allow=bool(result.allow_writing),
        )
        if self._strokes.is_drawing:
            buf.state = LeftGestureState.DRAWING
            self.hud.left_badge = f"[STATE: AIR-WRITING ({self._strokes.point_count})]"

        if completed is not None:
            self._emit_stroke(completed.points, completed.timestamps)
            buf.state = LeftGestureState.IDLE

    def _draw_recog_badge(self, frame: np.ndarray, lm: np.ndarray) -> None:
        if not self.hud.recog_badge or time.perf_counter() >= self.hud.recog_flash_until:
            return
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self._hand_bbox_px(lm, w, h)
        badge = self.hud.recog_badge
        color = (80, 255, 180)
        bx, by = x1, max(y1 - 12, 22)
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        cv2.rectangle(frame, (bx - 4, by - th - 6), (bx + tw + 4, by + 6), color, -1, cv2.LINE_AA)
        cv2.putText(frame, badge, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (10, 10, 10), 2, cv2.LINE_AA)

    @staticmethod
    def _hand_bbox_px(lm: np.ndarray, w: int, h: int, pad: int = 18) -> tuple[int, int, int, int]:
        xs = (lm[:, 0] * w).astype(np.int32)
        ys = (lm[:, 1] * h).astype(np.int32)
        x1 = int(max(int(xs.min()) - pad, 0))
        y1 = int(max(int(ys.min()) - pad, 0))
        x2 = int(min(int(xs.max()) + pad, w - 1))
        y2 = int(min(int(ys.max()) + pad, h - 1))
        return x1, y1, x2, y2

    def _draw_hand_tag(
        self,
        frame: np.ndarray,
        lm: np.ndarray,
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self._hand_bbox_px(lm, w, h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        ty = max(y1 - 8, 18)
        cv2.putText(frame, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def _draw_left_state_badge(self, frame: np.ndarray, lm: np.ndarray) -> None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self._hand_bbox_px(lm, w, h)
        badge = self.hud.left_badge or "[STATE: IDLE]"
        color = self.hud.left_badge_color or (220, 220, 220)
        bx, by = x1, min(y2 + 22, h - 8)
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        if self.hud.left_badge_color is not None:
            cv2.rectangle(frame, (bx - 4, by - th - 6), (bx + tw + 4, by + 6), color, -1, cv2.LINE_AA)
            text_color = (20, 20, 20)
        else:
            text_color = color
        cv2.putText(frame, badge, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, badge, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA)

    def _draw_mouse_ring(self, frame: np.ndarray, lm: np.ndarray) -> None:
        h, w = frame.shape[:2]
        cx = int(lm[INDEX_TIP, 0] * w)
        cy = int(lm[INDEX_TIP, 1] * h)
        mode = self._right.mouse_mode
        if mode == MouseMode.LMB:
            color = RING_LMB
        elif mode == MouseMode.RMB:
            color = RING_RMB
        elif mode == MouseMode.SCROLL:
            color = RING_SCROLL
        else:
            color = RING_HOVER
        cv2.circle(frame, (cx, cy), 16, color, 3, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)

    def _draw_preview(
        self,
        frame: np.ndarray,
        left_lm: np.ndarray | None,
        right_lm: np.ndarray | None,
    ) -> None:
        # Neon trail (active + fade-out)
        self._strokes.draw_neon_trail(frame)

        if left_lm is not None:
            self._draw_hand_tag(
                frame,
                left_lm,
                f"[LEFT HAND - WRITER] [STATUS: {self.hud.left_status}]",
                COLOR_LEFT_BGR,
            )
            self._draw_left_state_badge(frame, left_lm)
            self._draw_recog_badge(frame, left_lm)
        if right_lm is not None:
            self._draw_hand_tag(frame, right_lm, "[RIGHT HAND - MOUSE]", COLOR_RIGHT_BGR)
            if not self.calibrator.active:
                self._draw_mouse_ring(frame, right_lm)

        rxy = self.hud.right_xy
        lmb = "ON" if self.hud.right_lmb else "OFF"
        rmb = "ON" if self.hud.right_rmb else "OFF"
        right_line = (
            f"[LMB: {lmb} | RMB: {rmb}]"
            + (f"  cursor {rxy[0]:.2f},{rxy[1]:.2f}" if self.hud.right_present and rxy else "  IDLE")
        )
        lines = [
            f"[Calibration: {self.hud.calib_status}]",
            f"[STATUS: {self.hud.left_status}]",
            f"[Left Hand: {self.hud.left_badge}]",
            f"[Active OS Language: {self.hud.os_lang}]",
            f"[Last Recognized: {self.hud.last_char or '-'}  conf={self.hud.last_conf:.2f}]",
            right_line,
            f"[Focus Mode: {self.hud.mode_label}]",
            "[C = recalibrate | STOP / Q = quit]",
        ]
        y0 = 26
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, y0 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (10, y0 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
