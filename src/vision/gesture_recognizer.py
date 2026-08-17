"""PEN_UP-only command recognizer for the exclusive left-hand FSM."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.vision.bone_angles import (
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    Finger,
    finger_angle_degrees,
    finger_spread_degrees,
)
from src.vision.hand_calibrator import HandProfile, default_profile, load_profile

PALM_CENTER = 9
SWIPE_WINDOW_S = 0.120
SWIPE_VELOCITY = 0.85
FIST_CURL_MIN_DEG = 85.0
FIST_HOLD_S = 0.250
FIST_STATIONARY_SPEED = 0.10
SWIPE_COOLDOWN_S = 0.550
FIST_COOLDOWN_S = 0.400
LANGUAGE_HOLD_S = 0.350
LANGUAGE_COOLDOWN_S = 0.600
V_EXTENDED_MAX_DEG = 45.0
V_CURLED_MIN_DEG = 75.0
V_SPREAD_MIN_DEG = 20.0
BADGE_FLASH_S = 0.450

FLASH_GREEN = (80, 220, 80)
FLASH_ORANGE = (0, 140, 255)
FLASH_BLUE = (255, 160, 40)
FLASH_PURPLE = (220, 80, 200)


class GestureEvent(str, Enum):
    NONE = "NONE"
    SPACE = "SPACE"
    BACKSPACE = "BACKSPACE"
    TAB = "TAB"
    V_SIGN_LANG = "V_SIGN_LANG"


class LeftGestureState(str, Enum):
    HOVER = "HOVER"
    WRITING = "WRITING"
    SPACE = "SPACE"
    BACKSPACE = "BACKSPACE"
    TAB = "TAB"
    LANG_TOGGLE = "LANG_TOGGLE"
    COOLDOWN = "COOLDOWN"
    IDLE = "HOVER"
    DRAWING = "WRITING"


@dataclass(frozen=True)
class GestureResult:
    event: GestureEvent
    state: LeftGestureState
    badge: str
    flash_bgr: tuple[int, int, int] | None
    purge_stroke: bool
    allow_writing: bool
    charge_progress: float = 0.0


class LeftHandGestureRecognizer:
    """Recognize commands only after the modal engine enters PEN_UP."""

    def __init__(self, profile: HandProfile | None = None) -> None:
        self.profile = profile or load_profile() or default_profile()
        self._palm_history: deque[tuple[float, float, float]] = deque(maxlen=32)
        self._fist_since: float | None = None
        self._language_since: float | None = None
        self._language_latched = False
        self._cooldown_until = 0.0
        self._swipe_cooldown_until = 0.0
        self._flash_until = 0.0
        self._flash_badge = "[HOVER]"
        self._flash_color: tuple[int, int, int] | None = None

    def set_profile(self, profile: HandProfile) -> None:
        self.profile = profile

    def reset(self) -> None:
        self.clear_motion()
        self._cooldown_until = 0.0
        self._swipe_cooldown_until = 0.0
        self._flash_until = 0.0
        self._flash_badge = "[HOVER]"
        self._flash_color = None

    def clear_motion(self) -> None:
        self._palm_history.clear()
        self._fist_since = None
        self._language_since = None
        self._language_latched = False

    def begin_pen_up(self) -> None:
        """Discard drawing motion before opening the command detector."""
        self.clear_motion()

    def in_cooldown(self, now: float) -> bool:
        return now < max(self._cooldown_until, self._swipe_cooldown_until)

    @staticmethod
    def _closed_fist(lm: np.ndarray) -> bool:
        return all(
            finger_angle_degrees(lm, finger) > FIST_CURL_MIN_DEG
            for finger in (Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY)
        )

    @staticmethod
    def _v_sign(lm: np.ndarray) -> bool:
        spread = finger_spread_degrees(
            lm,
            INDEX_MCP,
            INDEX_TIP,
            MIDDLE_MCP,
            MIDDLE_TIP,
        )
        return (
            finger_angle_degrees(lm, Finger.INDEX) < V_EXTENDED_MAX_DEG
            and finger_angle_degrees(lm, Finger.MIDDLE) < V_EXTENDED_MAX_DEG
            and finger_angle_degrees(lm, Finger.RING) > V_CURLED_MIN_DEG
            and finger_angle_degrees(lm, Finger.PINKY) > V_CURLED_MIN_DEG
            and spread > V_SPREAD_MIN_DEG
        )

    def _push_palm(self, lm: np.ndarray, now: float) -> None:
        self._palm_history.append(
            (float(lm[PALM_CENTER, 0]), float(lm[PALM_CENTER, 1]), now)
        )
        while (
            self._palm_history
            and now - self._palm_history[0][2] > SWIPE_WINDOW_S
        ):
            self._palm_history.popleft()

    def _palm_velocity(self) -> tuple[float, float]:
        if len(self._palm_history) < 2:
            return 0.0, 0.0
        first_x, first_y, first_t = self._palm_history[0]
        last_x, last_y, last_t = self._palm_history[-1]
        dt = max(last_t - first_t, 1e-6)
        return (last_x - first_x) / dt, (last_y - first_y) / dt

    def _swipe(self) -> GestureEvent:
        velocity_x, _velocity_y = self._palm_velocity()
        if velocity_x > SWIPE_VELOCITY:
            return GestureEvent.SPACE
        if velocity_x < -SWIPE_VELOCITY:
            return GestureEvent.BACKSPACE
        return GestureEvent.NONE

    def _fist_stationary(
        self,
        lm: np.ndarray,
        now: float,
        *,
        canvas_empty: bool,
    ) -> bool:
        velocity_x, velocity_y = self._palm_velocity()
        speed = float(np.hypot(velocity_x, velocity_y))
        if (
            not canvas_empty
            or not self._closed_fist(lm)
            or speed > FIST_STATIONARY_SPEED
        ):
            self._fist_since = None
            return False
        if self._fist_since is None:
            self._fist_since = now
            return False
        return now - self._fist_since >= FIST_HOLD_S

    def _visual(
        self, now: float, state: LeftGestureState
    ) -> tuple[str, tuple[int, int, int] | None]:
        if now < self._flash_until:
            return self._flash_badge, self._flash_color
        labels = {
            LeftGestureState.HOVER: "[HOVER]",
            LeftGestureState.WRITING: "[WRITING]",
            LeftGestureState.SPACE: "[SPACE]",
            LeftGestureState.BACKSPACE: "[BACKSPACE]",
            LeftGestureState.TAB: "[TAB]",
            LeftGestureState.LANG_TOGGLE: "[LANG_TOGGLE]",
            LeftGestureState.COOLDOWN: "[HOVER]",
        }
        return labels[state], None

    def _result(
        self,
        now: float,
        state: LeftGestureState,
        event: GestureEvent = GestureEvent.NONE,
        *,
        color: tuple[int, int, int] | None = None,
        purge: bool = False,
        charge_progress: float = 0.0,
    ) -> GestureResult:
        badge, visual_color = self._visual(now, state)
        if event != GestureEvent.NONE and color is not None:
            self._flash_badge = {
                GestureEvent.SPACE: "[SPACE]",
                GestureEvent.BACKSPACE: "[BACKSPACE]",
                GestureEvent.TAB: "[TAB]",
                GestureEvent.V_SIGN_LANG: "[LANG_TOGGLE]",
            }[event]
            self._flash_color = color
            self._flash_until = now + BADGE_FLASH_S
            badge = self._flash_badge
            visual_color = color
        return GestureResult(
            event=event,
            state=state,
            badge=badge,
            flash_bgr=visual_color,
            purge_stroke=purge,
            allow_writing=False,
            charge_progress=float(np.clip(charge_progress, 0.0, 1.0)),
        )

    def _trigger(
        self,
        now: float,
        event: GestureEvent,
        state: LeftGestureState,
        color: tuple[int, int, int],
        cooldown: float,
    ) -> GestureResult:
        if event in {GestureEvent.SPACE, GestureEvent.BACKSPACE}:
            self._swipe_cooldown_until = now + cooldown
        else:
            self._cooldown_until = now + cooldown
        self.clear_motion()
        return self._result(
            now,
            state,
            event,
            color=color,
            purge=True,
        )

    def update(
        self,
        lm: np.ndarray | None,
        now: float,
        writing_enabled: bool = True,
        index_extended: bool = False,
        stroke_points: int = 0,
        pen_down: bool = False,
    ) -> GestureResult:
        del writing_enabled, index_extended
        if pen_down:
            # Hard safety guard. No command state is updated in PEN_DOWN.
            return GestureResult(
                event=GestureEvent.NONE,
                state=LeftGestureState.WRITING,
                badge="[WRITING]",
                flash_bgr=None,
                purge_stroke=False,
                allow_writing=True,
                charge_progress=0.0,
            )
        if lm is None:
            self.clear_motion()
            return self._result(now, LeftGestureState.HOVER)
        if self.in_cooldown(now):
            self.clear_motion()
            return self._result(now, LeftGestureState.COOLDOWN)

        self._push_palm(lm, now)
        swipe = self._swipe()
        if swipe == GestureEvent.SPACE:
            return self._trigger(
                now,
                GestureEvent.SPACE,
                LeftGestureState.SPACE,
                FLASH_BLUE,
                SWIPE_COOLDOWN_S,
            )
        if swipe == GestureEvent.BACKSPACE:
            return self._trigger(
                now,
                GestureEvent.BACKSPACE,
                LeftGestureState.BACKSPACE,
                FLASH_ORANGE,
                SWIPE_COOLDOWN_S,
            )

        canvas_empty = stroke_points == 0
        if self._fist_stationary(lm, now, canvas_empty=canvas_empty):
            return self._trigger(
                now,
                GestureEvent.TAB,
                LeftGestureState.TAB,
                FLASH_GREEN,
                FIST_COOLDOWN_S,
            )

        if self._v_sign(lm):
            if self._language_since is None:
                self._language_since = now
            elif (
                not self._language_latched
                and now - self._language_since >= LANGUAGE_HOLD_S
            ):
                self._language_latched = True
                return self._trigger(
                    now,
                    GestureEvent.V_SIGN_LANG,
                    LeftGestureState.LANG_TOGGLE,
                    FLASH_PURPLE,
                    LANGUAGE_COOLDOWN_S,
                )
            progress = (now - self._language_since) / LANGUAGE_HOLD_S
            return self._result(
                now,
                LeftGestureState.LANG_TOGGLE,
                charge_progress=progress,
            )

        self._language_since = None
        self._language_latched = False
        if canvas_empty and self._closed_fist(lm):
            return self._result(now, LeftGestureState.TAB)
        return self._result(now, LeftGestureState.HOVER)
