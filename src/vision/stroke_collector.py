"""Discrete air-writing strokes with an atomic 400 ms letter clutch."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from src.vision.bone_angles import (
    Finger,
    finger_angle_degrees,
    finger_curled,
    finger_extended,
    palm_scale,
)
from src.vision.hand_calibrator import HandProfile, default_profile, load_profile
from src.vision.one_euro import OneEuroFilter2D

NEON_CORE = (102, 255, 0)
NEON_GLOW = (60, 180, 0)
NEON_SOFT = (40, 120, 0)

INDEX_TIP = 8

COMMIT_DELAY_S = 0.400


class PenState(str, Enum):
    UP = "UP"
    DOWN = "DOWN"


def live_hand_scale(lm: np.ndarray) -> float:
    return palm_scale(lm)


def is_index_writing_pose(lm: np.ndarray) -> bool:
    """Stateless PEN_DOWN entry test; modal hysteresis lives in the controller."""
    if lm is None or lm.shape[0] < 21:
        return False
    return finger_angle_degrees(lm, Finger.INDEX) < 50.0


def is_pen_up_pose(lm: np.ndarray) -> bool:
    """Index + middle extended with ring and pinky curled."""
    if lm is None or lm.shape[0] < 21:
        return True
    return (
        finger_extended(lm, Finger.INDEX)
        and finger_extended(lm, Finger.MIDDLE)
        and finger_curled(lm, Finger.RING)
        and finger_curled(lm, Finger.PINKY)
    )


@dataclass(frozen=True)
class CompletedStroke:
    points: np.ndarray
    timestamps: np.ndarray
    strokes: tuple[np.ndarray, ...]


@dataclass
class StrokeCollector:
    """Collect separated strokes and clear ink atomically on commit."""

    profile: HandProfile = field(default_factory=lambda: load_profile() or default_profile())
    tip_filter: OneEuroFilter2D = field(
        default_factory=lambda: OneEuroFilter2D(1.2, 0.05)
    )
    state: PenState = PenState.UP
    current_stroke: list[tuple[float, float]] = field(default_factory=list)
    letter_strokes: list[list[tuple[float, float]]] = field(default_factory=list)
    letter_commit_timer: float | None = None
    _active_timestamps: list[float] = field(default_factory=list)
    _letter_timestamps: list[list[float]] = field(default_factory=list)

    def set_profile(self, profile: HandProfile) -> None:
        self.profile = profile

    @property
    def point_count(self) -> int:
        return len(self.current_stroke) + sum(
            len(stroke) for stroke in self.letter_strokes
        )

    @property
    def active_stroke(self) -> list[tuple[float, float]]:
        """Compatibility alias for the discrete ink-clutch contract."""
        return self.current_stroke

    @property
    def active_strokes(self) -> list[list[tuple[float, float]]]:
        strokes = [*self.letter_strokes]
        if self.current_stroke:
            strokes.append(self.current_stroke)
        return strokes

    @property
    def commit_timer(self) -> float | None:
        """Compatibility alias for the 400 ms letter deadline."""
        return self.letter_commit_timer

    @commit_timer.setter
    def commit_timer(self, value: float | None) -> None:
        self.letter_commit_timer = value

    @property
    def is_drawing(self) -> bool:
        return self.state == PenState.DOWN

    @property
    def trail_abs(self) -> list[tuple[float, float]]:
        return [
            point
            for stroke in [*self.letter_strokes, self.current_stroke]
            for point in stroke
        ]

    def _clear_canvas(self) -> None:
        self.current_stroke = []
        self.letter_strokes = []
        self._active_timestamps = []
        self._letter_timestamps = []
        self.letter_commit_timer = None
        self.state = PenState.UP

    def reset(self, *, keep_fade: bool = False) -> None:
        del keep_fade
        self._clear_canvas()
        self.tip_filter.reset()

    def purge(self) -> None:
        """Atomic command flush: active ink, completed ink, and timers vanish."""
        self._clear_canvas()
        self.tip_filter.reset()

    def _finish_active(self, now: float) -> None:
        if len(self.current_stroke) > 2:
            self.letter_strokes.append(self.current_stroke)
            self._letter_timestamps.append(self._active_timestamps)
        self.current_stroke = []
        self._active_timestamps = []
        self.state = PenState.UP
        if self.letter_strokes:
            self.letter_commit_timer = now + COMMIT_DELAY_S

    def _commit(self) -> CompletedStroke | None:
        strokes = tuple(
            np.asarray(stroke, dtype=np.float64) for stroke in self.letter_strokes
        )
        timestamp_groups = tuple(
            np.asarray(values, dtype=np.float64) for values in self._letter_timestamps
        )

        # Clear before returning so the preview cannot render stale ink while
        # asynchronous inference is being queued or executed.
        self._clear_canvas()
        self.tip_filter.reset()
        if not strokes:
            return None

        points = np.concatenate(strokes, axis=0)
        timestamps = np.concatenate(timestamp_groups, axis=0)
        return CompletedStroke(points, timestamps, strokes)

    def _filtered_tip(self, lm: np.ndarray, now: float) -> tuple[float, float]:
        return self.tip_filter(
            float(lm[INDEX_TIP, 0]),
            float(lm[INDEX_TIP, 1]),
            now,
        )

    def begin_stroke(self, lm: np.ndarray, now: float) -> None:
        self.state = PenState.DOWN
        self.current_stroke = []
        self._active_timestamps = []
        self.letter_commit_timer = None
        self.append_point(lm, now)

    def append_point(self, lm: np.ndarray, now: float) -> None:
        if self.state != PenState.DOWN:
            self.begin_stroke(lm, now)
            return
        self.current_stroke.append(self._filtered_tip(lm, now))
        self._active_timestamps.append(now)

    def end_stroke(self, now: float) -> None:
        if self.state == PenState.DOWN:
            self._finish_active(now)

    def commit_if_due(self, now: float) -> CompletedStroke | None:
        if (
            self.letter_commit_timer is not None
            and now >= self.letter_commit_timer
        ):
            return self._commit()
        return None

    def update(
        self,
        lm: np.ndarray,
        now: float,
        *,
        allow: bool,
        index_extended: bool | None = None,
    ) -> CompletedStroke | None:
        del index_extended
        if not allow:
            return None

        writing_pose = is_index_writing_pose(lm)
        pen_up_pose = is_pen_up_pose(lm)

        if writing_pose:
            if self.state != PenState.DOWN:
                self.begin_stroke(lm, now)
            else:
                self.letter_commit_timer = None
                self.append_point(lm, now)
            return None

        if self.state == PenState.DOWN:
            # Any departure from PEN_DOWN closes the stroke immediately. The
            # explicit PEN_UP pose and hand-loss path are therefore gap-safe.
            self._finish_active(now)
        elif (
            pen_up_pose
            and self.letter_strokes
            and self.letter_commit_timer is None
        ):
            self.letter_commit_timer = now + COMMIT_DELAY_S

        return self.commit_if_due(now)

    def update_missing(self, now: float) -> CompletedStroke | None:
        if self.state == PenState.DOWN:
            self._finish_active(now)
        elif self.letter_strokes and self.letter_commit_timer is None:
            self.letter_commit_timer = now + COMMIT_DELAY_S
        return self.commit_if_due(now)

    def draw_neon_trail(self, frame: np.ndarray) -> None:
        strokes = [*self.letter_strokes]
        if self.current_stroke:
            strokes.append(self.current_stroke)
        if not strokes:
            return

        height, width = frame.shape[:2]
        overlay = frame.copy()
        for stroke in strokes:
            points = np.asarray(
                [(int(x * width), int(y * height)) for x, y in stroke],
                dtype=np.int32,
            )
            if len(points) == 1:
                cv2.circle(overlay, tuple(points[0]), 6, NEON_CORE, -1, cv2.LINE_AA)
                continue
            cv2.polylines(overlay, [points], False, NEON_SOFT, 10, cv2.LINE_AA)
            cv2.polylines(overlay, [points], False, NEON_GLOW, 6, cv2.LINE_AA)
            cv2.polylines(overlay, [points], False, NEON_CORE, 3, cv2.LINE_AA)
            cv2.circle(overlay, tuple(points[-1]), 6, NEON_CORE, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
