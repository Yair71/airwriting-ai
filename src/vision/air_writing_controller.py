"""Exclusive PEN_DOWN/PEN_UP state machine for the physical left hand."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from src.vision.bone_angles import Finger, finger_angle_degrees
from src.vision.dual_hand_router import HandTrackId, RoutedHandData
from src.vision.gesture_recognizer import (
    GestureEvent,
    LeftGestureState,
    LeftHandGestureRecognizer,
)
from src.vision.hand_calibrator import HandProfile
from src.vision.stroke_collector import (
    CompletedStroke,
    StrokeCollector,
)

PEN_DOWN_ANGLE_DEG = 50.0
PEN_UP_ANGLE_DEG = 80.0
COMMAND_GUARD_S = 0.450


class LeftModalState(str, Enum):
    PEN_DOWN = "PEN_DOWN"
    PEN_UP = "PEN_UP"


@dataclass
class AirWritingCallbacks:
    stroke: Callable[..., None] | None = None
    space: Callable[[], None] | None = None
    backspace: Callable[[], None] | None = None
    tab: Callable[[], None] | None = None
    language_toggle: Callable[[], None] | None = None


class AirWritingController:
    """Own exactly one mutually exclusive writer mode at every instant."""

    def __init__(self, callbacks: AirWritingCallbacks | None = None) -> None:
        self.callbacks = callbacks or AirWritingCallbacks()
        self._gestures = LeftHandGestureRecognizer()
        self._strokes = StrokeCollector()
        self._writing_enabled = True
        self.modal_state = LeftModalState.PEN_UP
        self.state = LeftGestureState.HOVER
        self.badge = "[HOVER]"
        self.badge_color: tuple[int, int, int] | None = None
        self.charge_progress = 0.0
        self._last_stroke_activity = -1e9

    @property
    def point_count(self) -> int:
        return self._strokes.point_count

    @property
    def trail(self) -> list[tuple[float, float]]:
        return self._strokes.trail_abs

    @property
    def drawing(self) -> bool:
        return self.modal_state == LeftModalState.PEN_DOWN

    @property
    def status(self) -> str:
        if self.modal_state == LeftModalState.PEN_DOWN:
            return "DRAWING"
        if self.state in {
            LeftGestureState.SPACE,
            LeftGestureState.BACKSPACE,
            LeftGestureState.TAB,
            LeftGestureState.LANG_TOGGLE,
        }:
            return "GESTURE"
        return "READY"

    def set_profile(self, profile: HandProfile) -> None:
        self._gestures.set_profile(profile)
        self._strokes.set_profile(profile)

    def set_writing_enabled(self, enabled: bool) -> None:
        self._writing_enabled = bool(enabled)
        if not enabled:
            self.purge()

    def purge(self) -> None:
        self._strokes.purge()
        self._gestures.clear_motion()
        self.modal_state = LeftModalState.PEN_UP
        self.state = LeftGestureState.HOVER
        self.badge = "[HOVER]"
        self.badge_color = None
        self.charge_progress = 0.0
        self._last_stroke_activity = -1e9

    def draw_trail(self, frame: np.ndarray) -> None:
        self._strokes.draw_neon_trail(frame)

    def _emit_completed(self, completed: CompletedStroke | None) -> None:
        if completed is None:
            return
        self._emit_stroke(
            completed.points,
            completed.timestamps,
            completed.strokes,
        )
        self.state = LeftGestureState.HOVER
        self.badge = "[HOVER]"
        self.badge_color = None
        self.charge_progress = 0.0

    def _emit_stroke(
        self,
        points: np.ndarray,
        timestamps: np.ndarray,
        strokes: tuple[np.ndarray, ...],
    ) -> None:
        if not self.callbacks.stroke:
            return
        model_points = points.copy()
        model_points[:, 1] = 1.0 - model_points[:, 1]
        model_strokes = tuple(
            np.column_stack((stroke[:, 0], 1.0 - stroke[:, 1]))
            for stroke in strokes
        )
        self.callbacks.stroke(model_points, timestamps, model_strokes)

    def _dispatch(self, event: GestureEvent) -> None:
        callback: Callable[[], None] | None = None
        if event == GestureEvent.SPACE:
            callback = self.callbacks.space
        elif event == GestureEvent.BACKSPACE:
            callback = self.callbacks.backspace
        elif event == GestureEvent.TAB:
            callback = self.callbacks.tab
        elif event == GestureEvent.V_SIGN_LANG:
            callback = self.callbacks.language_toggle
        if callback:
            callback()

    def _enter_pen_down(self, lm: np.ndarray, now: float) -> None:
        self.modal_state = LeftModalState.PEN_DOWN
        self._gestures.clear_motion()
        self._strokes.begin_stroke(lm, now)
        self._last_stroke_activity = now
        self.state = LeftGestureState.WRITING
        self.badge = "[WRITING]"
        self.badge_color = None
        self.charge_progress = 0.0

    def _enter_pen_up(self, now: float) -> None:
        self._strokes.end_stroke(now)
        self.modal_state = LeftModalState.PEN_UP
        self._gestures.begin_pen_up()
        self._last_stroke_activity = now
        self.state = LeftGestureState.HOVER
        self.badge = "[HOVER]"
        self.badge_color = None
        self.charge_progress = 0.0

    def _update_pen_down(self, lm: np.ndarray, now: float) -> None:
        # This branch never invokes the gesture recognizer.
        theta_index = finger_angle_degrees(lm, Finger.INDEX)
        if theta_index > PEN_UP_ANGLE_DEG:
            self._enter_pen_up(now)
            return

        # Angles <=80° preserve PEN_DOWN; 50–80° is the hysteresis band.
        self._strokes.append_point(lm, now)
        self._last_stroke_activity = now
        self.state = LeftGestureState.WRITING
        self.badge = "[WRITING]"
        self.badge_color = None
        self.charge_progress = 0.0

    def _update_pen_up(self, lm: np.ndarray, now: float) -> None:
        theta_index = finger_angle_degrees(lm, Finger.INDEX)
        if (
            theta_index < PEN_DOWN_ANGLE_DEG
            and not self._gestures.in_cooldown(now)
            and self._writing_enabled
        ):
            self._enter_pen_down(lm, now)
            return

        self._emit_completed(self._strokes.commit_if_due(now))
        commands_blocked = (
            self._strokes.point_count > 0
            or now - self._last_stroke_activity < COMMAND_GUARD_S
        )
        if commands_blocked:
            self._gestures.clear_motion()
            self.state = LeftGestureState.HOVER
            self.badge = "[HOVER]"
            self.badge_color = None
            self.charge_progress = 0.0
            return

        result = self._gestures.update(
            lm,
            now,
            writing_enabled=self._writing_enabled,
            stroke_points=self._strokes.point_count,
            pen_down=False,
        )
        self.state = result.state
        self.badge = result.badge
        self.badge_color = result.flash_bgr
        self.charge_progress = result.charge_progress

        if result.event != GestureEvent.NONE:
            # Atomic ordering: erase all ink before emitting any OS command.
            self._strokes.purge()
            self._dispatch(result.event)
            return

    def update(self, hand: RoutedHandData, now: float) -> bool:
        if hand is None or hand.track_id != HandTrackId.LEFT_HAND_WRITER:
            return False
        if not self._writing_enabled:
            return True

        if self.modal_state == LeftModalState.PEN_DOWN:
            self._update_pen_down(hand.landmarks, now)
        else:
            self._update_pen_up(hand.landmarks, now)
        return True

    def update_missing(self, now: float) -> None:
        if self.modal_state == LeftModalState.PEN_DOWN:
            self._enter_pen_up(now)
        self._gestures.clear_motion()
        self._emit_completed(self._strokes.commit_if_due(now))
        self.state = LeftGestureState.HOVER
        self.badge = "[HOVER]"
        self.badge_color = None
        self.charge_progress = 0.0
