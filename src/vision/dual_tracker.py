"""Camera orchestrator with physically isolated mouse and writing pipelines."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.paths import resource_root
from src.vision.air_writing_controller import (
    AirWritingCallbacks,
    AirWritingController,
)
from src.vision.dual_hand_router import DualHandRouter
from src.vision.hand_calibrator import (
    HandCalibrator,
    HandProfile,
    default_profile,
    load_profile,
)
from src.vision.mouse_controller import MouseCallbacks, MouseController, MouseMode
from src.vision.threaded_camera import ThreadedCamera

REPO_ROOT = resource_root()
DEFAULT_MODEL = REPO_ROOT / "data" / "models" / "hand_landmarker.task"
INDEX_TIP = 8
RECOG_FLASH_S = 0.500

COLOR_LEFT_BGR = (255, 140, 0)
COLOR_RIGHT_BGR = (0, 140, 255)
RING_HOVER = (80, 220, 80)
RING_LMB = (40, 40, 240)
RING_RMB = (0, 220, 255)
RING_SCROLL = (255, 160, 40)


@dataclass
class DualHandCallbacks:
    on_mouse_move: Callable[[float, float], None] | None = None
    on_mouse_position: Callable[[], tuple[int, int]] | None = None
    on_screen_size: Callable[[], tuple[int, int]] | None = None
    on_left_down: Callable[[], None] | None = None
    on_left_up: Callable[[], None] | None = None
    on_right_down: Callable[[], None] | None = None
    on_right_up: Callable[[], None] | None = None
    on_left_click: Callable[[], None] | None = None
    on_right_click: Callable[[], None] | None = None
    on_scroll: Callable[[int], None] | None = None
    on_stroke: Callable[..., None] | None = None
    on_fist_tab: Callable[[], None] | None = None
    on_swipe_left: Callable[[], None] | None = None
    on_swipe_right: Callable[[], None] | None = None
    on_lang_switch: Callable[[], None] | None = None
    on_enter: Callable[[], None] | None = None
    on_space: Callable[[], None] | None = None
    on_backspace: Callable[[], None] | None = None
    on_tab: Callable[[], None] | None = None


@dataclass
class DebugHUD:
    left_state: str = "HOVER"
    left_badge: str = "[HOVER]"
    left_badge_color: tuple[int, int, int] | None = None
    left_charge: float = 0.0
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
    mouse_mode: str = "IDLE"
    left_status: str = "READY"
    right_lmb: bool = False
    right_rmb: bool = False


class DualHandTracker:
    """Capture MediaPipe frames and dispatch trusted role objects only."""

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
        # Strict role mapping requires selfie-view input. Keep the parameter
        # for API compatibility, but never permit an unmirrored dispatch frame.
        self.mirror = True
        self.callbacks = callbacks or DualHandCallbacks()
        self.show_preview = show_preview
        self.writing_enabled = bool(writing_enabled)
        self.hud = DebugHUD()
        self.mode_label = "Writing"
        self._cap: ThreadedCamera | None = None
        self._frame_ts_ms = 0
        self._running = False
        self._router = DualHandRouter(mouse_min_confidence=0.80)

        self.mouse_controller = MouseController(
            MouseCallbacks(
                move=self.callbacks.on_mouse_move,
                position=self.callbacks.on_mouse_position,
                screen_size=self.callbacks.on_screen_size,
                left_down=self.callbacks.on_left_down,
                left_up=self.callbacks.on_left_up,
                right_down=self.callbacks.on_right_down,
                right_up=self.callbacks.on_right_up,
                left_click=self.callbacks.on_left_click,
                right_click=self.callbacks.on_right_click,
                scroll=self.callbacks.on_scroll,
            )
        )
        self.air_writing_controller = AirWritingController(
            AirWritingCallbacks(
                stroke=self.callbacks.on_stroke,
                space=self.callbacks.on_space or self.callbacks.on_swipe_right,
                backspace=self.callbacks.on_backspace or self.callbacks.on_swipe_left,
                tab=self.callbacks.on_tab or self.callbacks.on_fist_tab,
                language_toggle=self.callbacks.on_lang_switch,
            )
        )
        self.air_writing_controller.set_writing_enabled(self.writing_enabled)

        self.calibrator = HandCalibrator()
        loaded = load_profile()
        if loaded is None:
            self.apply_profile(default_profile())
            if show_preview:
                self.hud.calib_status = "Needs calibration"
                self.calibrator.start()
            else:
                self.hud.calib_status = "Default profile"
        else:
            self.apply_profile(loaded)
            self.hud.calib_status = "Calibrated"

    def apply_profile(self, profile: HandProfile) -> None:
        self.profile = profile
        self.air_writing_controller.set_profile(profile)
        self.hud.calib_status = (
            f"Calibrated palm={profile.palm_base_scale:.3f}"
        )

    def set_writing_enabled(self, enabled: bool) -> None:
        self.writing_enabled = bool(enabled)
        self.air_writing_controller.set_writing_enabled(enabled)

    def set_os_lang(self, lang: str) -> None:
        self.hud.os_lang = str(lang).upper()

    def set_last_recognition(
        self, label: str, conf: float, *, flash: bool = True
    ) -> None:
        self.hud.last_char = label
        self.hud.last_conf = float(conf)
        if flash and label:
            self.hud.recog_badge = (
                f"[RECOGNIZED: '{label}' ({100.0 * float(conf):.1f}%)]"
            )
            self.hud.recog_flash_until = time.perf_counter() + RECOG_FLASH_S
        elif not flash:
            self.hud.recog_badge = ""
            self.hud.recog_flash_until = 0.0

    def open(self) -> None:
        cap = ThreadedCamera(
            self.camera_index,
            width=640,
            height=480,
            fps=60,
        )
        cap.start()
        self._cap = cap
        self._running = True

    def close(self) -> None:
        self.mouse_controller.suspend()
        self._running = False
        if self._cap is not None:
            self._cap.close()
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
        packet = self._cap.read_latest()
        if packet is None:
            time.sleep(0)
            return self._running
        frame = packet.image
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts_ms = max(self._frame_ts_ms + 1, packet.timestamp_ms)
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)
        now = time.perf_counter()

        phys_left_hand, phys_right_hand = self._router.route(result)
        writer_lm = (
            phys_left_hand.landmarks if phys_left_hand is not None else None
        )
        mouse_lm = (
            phys_right_hand.landmarks if phys_right_hand is not None else None
        )
        self.hud.left_present = phys_left_hand is not None
        self.hud.right_present = phys_right_hand is not None
        self.hud.mode_label = self.mode_label

        if self.calibrator.active:
            finished = self.calibrator.update(writer_lm)
            if finished is not None:
                self.apply_profile(finished)
            self.mouse_controller.suspend()
            self.hud.calib_status = self.calibrator.status_line
        else:
            # Hard null gate: update is never called without explicit mouse data.
            if phys_right_hand is not None and phys_right_hand.confidence >= 0.80:
                self.mouse_controller.update(phys_right_hand, now)
            else:
                self.mouse_controller.suspend()

            # Independent writer gate: mouse data cannot enter this controller.
            if phys_left_hand is not None:
                self.air_writing_controller.update(phys_left_hand, now)
            else:
                self.air_writing_controller.update_missing(now)

        self._sync_hud(now)
        if self.show_preview:
            self._draw_preview(frame, writer_lm, mouse_lm)
            if self.calibrator.active:
                self.calibrator.draw_overlay(frame)
            cv2.imshow("AirTouch", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                self._running = False
                return False
            if key in (ord("c"), ord("C")) and not self.calibrator.active:
                self.mouse_controller.suspend()
                self.calibrator.start()
                self.hud.calib_status = self.calibrator.status_line
        return True

    def _sync_hud(self, now: float) -> None:
        air = self.air_writing_controller
        mouse = self.mouse_controller
        self.hud.left_state = air.state.value
        self.hud.left_badge = air.badge
        self.hud.left_badge_color = air.badge_color
        self.hud.left_charge = air.charge_progress
        self.hud.left_points = air.point_count
        self.hud.trail = air.trail
        self.hud.left_status = air.status
        self.hud.right_xy = mouse.xy
        self.hud.mouse_mode = mouse.mode.value
        self.hud.right_lmb = mouse.lmb_active
        self.hud.right_rmb = mouse.rmb_active
        if self.hud.recog_flash_until and now >= self.hud.recog_flash_until:
            self.hud.recog_badge = ""
            self.hud.recog_flash_until = 0.0

    @staticmethod
    def _hand_bbox_px(
        lm: np.ndarray, width: int, height: int, pad: int = 18
    ) -> tuple[int, int, int, int]:
        xs = (lm[:, 0] * width).astype(np.int32)
        ys = (lm[:, 1] * height).astype(np.int32)
        return (
            int(max(int(xs.min()) - pad, 0)),
            int(max(int(ys.min()) - pad, 0)),
            int(min(int(xs.max()) + pad, width - 1)),
            int(min(int(ys.max()) + pad, height - 1)),
        )

    def _draw_hand_tag(
        self,
        frame: np.ndarray,
        lm: np.ndarray,
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self._hand_bbox_px(lm, width, height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        baseline = max(y1 - 8, 18)
        cv2.putText(
            frame, label, (x1, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.53,
            (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            frame, label, (x1, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.53,
            color, 2, cv2.LINE_AA
        )

    def _draw_writer_badges(self, frame: np.ndarray, lm: np.ndarray) -> None:
        height, width = frame.shape[:2]
        x1, _y1, _x2, y2 = self._hand_bbox_px(lm, width, height)
        badge = self.hud.left_badge
        color = self.hud.left_badge_color or (230, 230, 230)
        baseline = min(y2 + 22, height - 8)
        cv2.putText(
            frame, badge, (x1, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            frame, badge, (x1, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            color, 2, cv2.LINE_AA
        )
        if self.hud.recog_badge and time.perf_counter() < self.hud.recog_flash_until:
            cv2.putText(
                frame, self.hud.recog_badge, (x1, max(baseline - 24, 22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 180), 2, cv2.LINE_AA
            )
        if self.hud.left_charge > 0.0:
            x1, y1, x2, y2 = self._hand_bbox_px(lm, width, height, pad=8)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            radius = max(x2 - x1, y2 - y1) // 2 + 8
            cv2.circle(frame, center, radius, (70, 55, 85), 3, cv2.LINE_AA)
            cv2.ellipse(
                frame,
                center,
                (radius, radius),
                -90.0,
                0.0,
                360.0 * self.hud.left_charge,
                (220, 80, 200),
                4,
                cv2.LINE_AA,
            )

    def _draw_mouse_ring(self, frame: np.ndarray, lm: np.ndarray) -> None:
        height, width = frame.shape[:2]
        center = (int(lm[INDEX_TIP, 0] * width), int(lm[INDEX_TIP, 1] * height))
        colors = {
            MouseMode.LMB_PENDING: RING_LMB,
            MouseMode.LMB_DRAGGING: RING_LMB,
            MouseMode.RMB_CLICKED: RING_RMB,
            MouseMode.MOUSE_SCROLL: RING_SCROLL,
        }
        color = colors.get(self.mouse_controller.mode, RING_HOVER)
        cv2.circle(frame, center, 16, color, 3, cv2.LINE_AA)
        cv2.circle(frame, center, 4, color, -1, cv2.LINE_AA)

    def _draw_preview(
        self,
        frame: np.ndarray,
        writer_lm: np.ndarray | None,
        mouse_lm: np.ndarray | None,
    ) -> None:
        self.air_writing_controller.draw_trail(frame)
        if writer_lm is not None:
            self._draw_hand_tag(
                frame,
                writer_lm,
                "[LEFT HAND - WRITER]",
                COLOR_LEFT_BGR,
            )
            self._draw_writer_badges(frame, writer_lm)
        if mouse_lm is not None:
            self._draw_hand_tag(
                frame,
                mouse_lm,
                "[RIGHT HAND - MOUSE]",
                COLOR_RIGHT_BGR,
            )
            if not self.calibrator.active:
                self._draw_mouse_ring(frame, mouse_lm)

        lmb = "ON" if self.hud.right_lmb else "OFF"
        rmb = "ON" if self.hud.right_rmb else "OFF"
        lines = [
            f"[Calibration: {self.hud.calib_status}]",
            f"[Writer: {self.hud.left_badge} | {self.hud.left_points} pts]",
            f"[Mouse: {self.hud.mouse_mode} | LMB {lmb} | RMB {rmb}]",
            f"[Language: {self.hud.os_lang}]",
            f"[Last: {self.hud.last_char or '-'} {self.hud.last_conf:.2f}]",
            "[C = recalibrate | STOP / Q = quit]",
        ]
        for index, line in enumerate(lines):
            y = 26 + index * 24
            cv2.putText(
                frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (40, 40, 40), 3, cv2.LINE_AA
            )
            cv2.putText(
                frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (240, 240, 240), 1, cv2.LINE_AA
            )
