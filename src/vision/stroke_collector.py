"""In-air stroke collector: PEN_DOWN / buffer / PEN_UP + neon trail render."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from src.vision.hand_calibrator import HandProfile, default_profile, load_profile
from src.vision.one_euro import OneEuroFilter2D

# Neon green #00FF66 → OpenCV BGR
NEON_CORE = (102, 255, 0)
NEON_GLOW = (60, 180, 0)
NEON_SOFT = (40, 120, 0)

INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18
WRIST = 0
MIDDLE_MCP = 9

DWELL_S = 0.160
MIN_POINTS = 8
FADE_S = 0.450
DWELL_SPEED = 0.08


class PenState(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    FADING = "FADING"


def live_hand_scale(lm: np.ndarray) -> float:
    return max(float(np.linalg.norm(lm[MIDDLE_MCP, :2] - lm[WRIST, :2])), 1e-6)


def is_index_writing_pose(lm: np.ndarray) -> bool:
    """Index tip (8) higher than PIP (6); middle/ring/pinky relaxed (not raised)."""
    if lm is None or lm.shape[0] < 21:
        return False
    # Image Y grows downward — "higher" means smaller y.
    if float(lm[INDEX_TIP, 1]) >= float(lm[INDEX_PIP, 1]):
        return False
    for tip, pip in ((MIDDLE_TIP, MIDDLE_PIP), (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)):
        if float(lm[tip, 1]) < float(lm[pip, 1]) - 0.012:
            return False
    return True


def is_index_curled(lm: np.ndarray) -> bool:
    if lm is None or lm.shape[0] < 21:
        return True
    return float(lm[INDEX_TIP, 1]) >= float(lm[INDEX_PIP, 1])


@dataclass
class CompletedStroke:
    """Screen-space tip polyline ready for the ONNX preprocessor (y will be flipped by caller)."""

    points: np.ndarray  # (N, 2) float64 absolute normalized [0,1]
    timestamps: np.ndarray  # (N,) perf_counter seconds


@dataclass
class StrokeCollector:
    """Left-index tip trajectory FSM with noise rejection and neon preview trail."""

    profile: HandProfile = field(default_factory=lambda: load_profile() or default_profile())
    tip_filter: OneEuroFilter2D = field(default_factory=lambda: OneEuroFilter2D(1.2, 0.05))
    state: PenState = PenState.UP
    _abs: list[tuple[float, float, float]] = field(default_factory=list)
    _rel: list[tuple[float, float, float]] = field(default_factory=list)
    _anchor: tuple[float, float] | None = None
    _last_xy: tuple[float, float] | None = None
    _last_t: float | None = None
    _dwell_since: float | None = None
    _fade_pts: list[tuple[float, float]] = field(default_factory=list)
    _fade_until: float = 0.0
    dwell_speed: float = DWELL_SPEED

    def set_profile(self, profile: HandProfile) -> None:
        self.profile = profile

    def reset(self, *, keep_fade: bool = False) -> None:
        if keep_fade and len(self._abs) >= 2:
            self._start_fade(time.perf_counter())
        self._abs.clear()
        self._rel.clear()
        self._anchor = None
        self._dwell_since = None
        self.state = PenState.UP

    def purge(self) -> None:
        """Hard clear (gesture interruption) — no fade."""
        self._abs.clear()
        self._rel.clear()
        self._anchor = None
        self._dwell_since = None
        self._fade_pts.clear()
        self._fade_until = 0.0
        self.state = PenState.UP
        self.tip_filter.reset()

    @property
    def point_count(self) -> int:
        return len(self._abs)

    @property
    def is_drawing(self) -> bool:
        return self.state == PenState.DOWN

    @property
    def trail_abs(self) -> list[tuple[float, float]]:
        if self.state == PenState.FADING or (self._fade_pts and time.perf_counter() < self._fade_until):
            return list(self._fade_pts)
        return [(x, y) for x, y, _t in self._abs]

    def _hand_scale(self, lm: np.ndarray | None) -> float:
        if lm is not None and lm.shape[0] >= 21:
            return live_hand_scale(lm)
        return max(float(self.profile.palm_base_scale), 1e-4)

    def _min_diagonal(self, lm: np.ndarray | None) -> float:
        return 0.03 * self._hand_scale(lm)

    def _bbox_diagonal(self, pts: list[tuple[float, float, float]]) -> float:
        if len(pts) < 2:
            return 0.0
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))

    def _start_fade(self, now: float) -> None:
        self._fade_pts = [(x, y) for x, y, _t in self._abs]
        self._fade_until = now + FADE_S
        self.state = PenState.FADING

    def _emit_if_valid(self, now: float, lm: np.ndarray | None) -> CompletedStroke | None:
        n = len(self._abs)
        diag = self._bbox_diagonal(self._abs)
        valid = n >= MIN_POINTS and diag >= self._min_diagonal(lm)
        result: CompletedStroke | None = None
        if valid:
            pts = np.array([(x, y) for x, y, _t in self._abs], dtype=np.float64)
            ts = np.array([t for _x, _y, t in self._abs], dtype=np.float64)
            result = CompletedStroke(points=pts, timestamps=ts)
        self._start_fade(now)
        self._abs.clear()
        self._rel.clear()
        self._anchor = None
        self._dwell_since = None
        return result

    def update(
        self,
        lm: np.ndarray,
        now: float,
        *,
        allow: bool,
        index_extended: bool | None = None,
    ) -> CompletedStroke | None:
        """Advance collector. Returns a completed stroke once, then resets the buffer."""
        if self.state == PenState.FADING and now >= self._fade_until:
            self._fade_pts.clear()
            self.state = PenState.UP

        writing = is_index_writing_pose(lm) if lm is not None else False
        if index_extended is True:
            writing = True
        curled = is_index_curled(lm) if lm is not None else True
        tip_xy = (float(lm[INDEX_TIP, 0]), float(lm[INDEX_TIP, 1])) if lm is not None else (0.0, 0.0)

        if not allow:
            if self.state == PenState.DOWN:
                if curled:
                    out = self._emit_if_valid(now, lm)
                    self._last_xy = tip_xy
                    self._last_t = now
                    return out
                self._start_fade(now)
                self._abs.clear()
                self._rel.clear()
                self._anchor = None
                self._dwell_since = None
            self._last_xy = tip_xy
            self._last_t = now
            return None

        fx, fy = self.tip_filter(float(tip_xy[0]), float(tip_xy[1]), now)
        speed = 0.0
        if self._last_xy is not None and self._last_t is not None:
            dt = max(now - self._last_t, 1e-4)
            speed = float(np.hypot(fx - self._last_xy[0], fy - self._last_xy[1])) / dt
        self._last_xy = (fx, fy)
        self._last_t = now

        if self.state != PenState.DOWN:
            if writing:
                self.state = PenState.DOWN
                self._anchor = (fx, fy)
                self._abs = [(fx, fy, now)]
                self._rel = [(0.0, 0.0, now)]
                self._dwell_since = None
                self._fade_pts.clear()
            return None

        assert self._anchor is not None
        ax, ay = self._anchor
        self._abs.append((fx, fy, now))
        self._rel.append((fx - ax, fy - ay, now))
        if len(self._abs) > 500:
            self._abs = self._abs[-400:]
            self._rel = self._rel[-400:]

        if speed < self.dwell_speed:
            if self._dwell_since is None:
                self._dwell_since = now
        else:
            self._dwell_since = None

        dwell_done = self._dwell_since is not None and (now - self._dwell_since) >= DWELL_S
        if dwell_done or curled:
            return self._emit_if_valid(now, lm)
        return None

    def draw_neon_trail(self, frame: np.ndarray) -> None:
        """Antialiased neon green glow trail on the OpenCV preview frame."""
        h, w = frame.shape[:2]
        now = time.perf_counter()
        fading = bool(self._fade_pts) and now < self._fade_until
        pts_src = self._fade_pts if fading else [(x, y) for x, y, _t in self._abs]
        if len(pts_src) < 1:
            return

        alpha = 1.0
        if fading:
            alpha = max(0.0, (self._fade_until - now) / FADE_S)

        pts = np.array([(int(x * w), int(y * h)) for x, y in pts_src], dtype=np.int32)
        overlay = frame.copy()

        if len(pts) >= 2:
            thickness_glow = max(int(10 * alpha), 2)
            thickness_mid = max(int(6 * alpha), 2)
            thickness_core = max(int(2 + 2 * alpha), 1)
            cv2.polylines(overlay, [pts], False, NEON_SOFT, thickness_glow, cv2.LINE_AA)
            cv2.polylines(overlay, [pts], False, NEON_GLOW, thickness_mid, cv2.LINE_AA)
            cv2.polylines(overlay, [pts], False, NEON_CORE, thickness_core, cv2.LINE_AA)

        tip = tuple(int(v) for v in pts[-1])
        r_outer = max(int(14 * alpha), 3)
        r_inner = max(int(6 * alpha), 2)
        cv2.circle(overlay, tip, r_outer, NEON_SOFT, -1, cv2.LINE_AA)
        cv2.circle(overlay, tip, r_inner + 2, NEON_GLOW, -1, cv2.LINE_AA)
        cv2.circle(overlay, tip, r_inner, NEON_CORE, -1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.55 * alpha + 0.25, frame, 1.0 - (0.55 * alpha + 0.25), 0, frame)
