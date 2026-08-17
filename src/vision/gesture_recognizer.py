"""Left-hand micro-gesture recognizer (fist / swipe / V-sign / enter hook).

Loads calibrated thresholds from configs/hand_profile.json.
Priority per frame: FIST → SWIPE → V-SIGN → ENTER → AIR-WRITING / IDLE.
Discrete gestures always purge writing trajectory buffers via GestureResult.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.vision.hand_calibrator import HandProfile, default_profile, load_profile

WRIST = 0
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17

SWIPE_WINDOW_S = 0.150
SWIPE_HORIZ_RATIO = 3.5
FIST_COOLDOWN_S = 0.350
SWIPE_COOLDOWN_S = 0.300
VSIGN_HOLD_S = 0.350
VSIGN_COOLDOWN_S = 0.600
ENTER_COOLDOWN_S = 0.350
BADGE_FLASH_S = 0.450
MOTION_HISTORY_S = 0.400

# OpenCV BGR flash colors for HUD badges
FLASH_GREEN = (80, 220, 80)
FLASH_ORANGE = (0, 140, 255)
FLASH_BLUE = (255, 160, 40)
FLASH_PURPLE = (220, 80, 200)
FLASH_CYAN = (255, 220, 80)


class GestureEvent(str, Enum):
    NONE = "NONE"
    FIST_TAB = "FIST_TAB"
    SWIPE_LEFT = "SWIPE_LEFT"
    SWIPE_RIGHT = "SWIPE_RIGHT"
    V_SIGN_LANG = "V_SIGN_LANG"
    ENTER = "ENTER"


class LeftGestureState(str, Enum):
    IDLE = "IDLE"
    FIST = "FIST"
    SWIPE_LEFT = "SWIPE_LEFT"
    SWIPE_RIGHT = "SWIPE_RIGHT"
    V_SIGN = "V_SIGN"
    ENTER = "ENTER"
    DRAWING = "DRAWING"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True)
class GestureResult:
    event: GestureEvent
    state: LeftGestureState
    badge: str
    flash_bgr: tuple[int, int, int] | None
    purge_stroke: bool
    allow_writing: bool


def _dist(lm: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(lm[a, :2] - lm[b, :2]))


class LeftHandGestureRecognizer:
    """Stateful left-hand discrete gestures with hysteresis + cooldowns."""

    def __init__(self, profile: HandProfile | None = None) -> None:
        self.profile = profile or load_profile() or default_profile()
        self._motion: deque[tuple[float, float, float]] = deque(maxlen=64)  # tip x,y,t
        self._wrist_motion: deque[tuple[float, float, float]] = deque(maxlen=64)
        self._fist_latched = False
        self._vsign_since: float | None = None
        self._vsign_fired = False
        self._cooldown_until = 0.0
        self._writing_lock_until = 0.0
        self._flash_until = 0.0
        self._flash_badge = "[STATE: IDLE]"
        self._flash_color: tuple[int, int, int] | None = None
        self._last_state = LeftGestureState.IDLE
        self.swipe_velocity_threshold = max(1.4, 9.0 * float(self.profile.palm_base_scale))
        self.swipe_disp_threshold = max(0.08, 0.55 * float(self.profile.palm_base_scale))

    def set_profile(self, profile: HandProfile) -> None:
        self.profile = profile
        self.swipe_velocity_threshold = max(1.4, 9.0 * float(profile.palm_base_scale))
        self.swipe_disp_threshold = max(0.08, 0.55 * float(profile.palm_base_scale))

    def reset(self) -> None:
        self._motion.clear()
        self._wrist_motion.clear()
        self._fist_latched = False
        self._vsign_since = None
        self._vsign_fired = False
        self._cooldown_until = 0.0
        self._writing_lock_until = 0.0
        self._flash_until = 0.0
        self._flash_badge = "[STATE: IDLE]"
        self._flash_color = None
        self._last_state = LeftGestureState.IDLE

    def clear_motion(self) -> None:
        self._motion.clear()
        self._wrist_motion.clear()

    def _set_flash(self, badge: str, color: tuple[int, int, int], now: float) -> None:
        self._flash_badge = badge
        self._flash_color = color
        self._flash_until = now + BADGE_FLASH_S

    def _badge_now(self, now: float, state: LeftGestureState) -> tuple[str, tuple[int, int, int] | None]:
        if now < self._flash_until:
            return self._flash_badge, self._flash_color
        mapping = {
            LeftGestureState.IDLE: ("[STATE: IDLE]", None),
            LeftGestureState.DRAWING: ("[STATE: AIR-WRITING]", None),
            LeftGestureState.FIST: ("[STATE: FIST -> TAB]", FLASH_GREEN),
            LeftGestureState.SWIPE_LEFT: ("[STATE: SWIPE LEFT -> BACKSPACE]", FLASH_ORANGE),
            LeftGestureState.SWIPE_RIGHT: ("[STATE: SWIPE RIGHT -> SPACE]", FLASH_BLUE),
            LeftGestureState.V_SIGN: ("[STATE: V-SIGN -> LANG TOGGLE]", FLASH_PURPLE),
            LeftGestureState.ENTER: ("[STATE: ENTER -> RETURN]", FLASH_CYAN),
            LeftGestureState.COOLDOWN: ("[STATE: COOLDOWN]", None),
        }
        return mapping.get(state, ("[STATE: IDLE]", None))

    def _is_fist(self, lm: np.ndarray) -> bool:
        palm = lm[MIDDLE_MCP, :2]
        thr = float(self.profile.fist_closed_threshold)
        for tip in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP):
            if float(np.linalg.norm(lm[tip, :2] - palm)) >= thr:
                return False
        return True

    def _is_vsign(self, lm: np.ndarray) -> bool:
        wrist = lm[WRIST, :2]
        palm = float(self.profile.palm_base_scale)

        def ext(tip: int, mcp: int) -> bool:
            return float(np.linalg.norm(lm[tip, :2] - wrist)) > float(np.linalg.norm(lm[mcp, :2] - wrist)) * 1.15

        def curl(tip: int, mcp: int) -> bool:
            # Curled toward palm center (MCP region)
            tip_to_palm = float(np.linalg.norm(lm[tip, :2] - lm[MIDDLE_MCP, :2]))
            mcp_to_palm = float(np.linalg.norm(lm[mcp, :2] - lm[MIDDLE_MCP, :2]))
            tip_to_wrist = float(np.linalg.norm(lm[tip, :2] - wrist))
            mcp_to_wrist = float(np.linalg.norm(lm[mcp, :2] - wrist))
            return tip_to_wrist < mcp_to_wrist * 1.05 or tip_to_palm < mcp_to_palm * 1.35

        spread = _dist(lm, INDEX_TIP, MIDDLE_TIP)
        return (
            ext(INDEX_TIP, INDEX_MCP)
            and ext(MIDDLE_TIP, MIDDLE_MCP)
            and curl(RING_TIP, RING_MCP)
            and curl(PINKY_TIP, PINKY_MCP)
            and spread > 0.20 * palm
        )

    def _push_motion(self, lm: np.ndarray, now: float) -> None:
        self._motion.append((float(lm[INDEX_TIP, 0]), float(lm[INDEX_TIP, 1]), now))
        self._wrist_motion.append((float(lm[WRIST, 0]), float(lm[WRIST, 1]), now))
        # Drop samples older than history window
        while self._motion and (now - self._motion[0][2]) > MOTION_HISTORY_S:
            self._motion.popleft()
        while self._wrist_motion and (now - self._wrist_motion[0][2]) > MOTION_HISTORY_S:
            self._wrist_motion.popleft()

    def _window(self, hist: deque[tuple[float, float, float]], now: float, span: float) -> list[tuple[float, float, float]]:
        return [(x, y, t) for x, y, t in hist if now - t <= span]

    def _detect_swipe(self, now: float) -> GestureEvent | None:
        """Rapid horizontal flick only: |dx| > 3.5*|dy| completed in < 150ms."""
        for hist in (self._motion, self._wrist_motion):
            pts = self._window(hist, now, SWIPE_WINDOW_S)
            if len(pts) < 3:
                continue
            x0, y0, t0 = pts[0]
            x1, y1, t1 = pts[-1]
            dx, dy = x1 - x0, y1 - y0
            dt = max(t1 - t0, 1e-4)
            if dt > SWIPE_WINDOW_S:
                continue
            if abs(dx) <= SWIPE_HORIZ_RATIO * abs(dy):
                continue
            vel = abs(dx) / dt
            if vel < self.swipe_velocity_threshold:
                continue
            if abs(dx) < self.swipe_disp_threshold:
                continue
            return GestureEvent.SWIPE_RIGHT if dx > 0 else GestureEvent.SWIPE_LEFT
        return None

    def _detect_enter_hook(self, now: float) -> bool:
        """Downward stroke then sharp left hook within ~350ms."""
        pts = list(self._motion)
        if len(pts) < 6:
            return False
        # Restrict to recent samples
        pts = [(x, y, t) for x, y, t in pts if now - t <= 0.35]
        if len(pts) < 6:
            return False

        # Find the sample with strongest downward velocity mid-window, then leftward finish
        best_i = None
        best_down = 0.0
        for i in range(1, len(pts) - 2):
            x0, y0, t0 = pts[i - 1]
            x1, y1, t1 = pts[i]
            dt = max(t1 - t0, 1e-4)
            dy = y1 - y0
            dx = x1 - x0
            if dy > 0 and dy > 1.4 * abs(dx):
                v = dy / dt
                if v > best_down:
                    best_down = v
                    best_i = i
        if best_i is None or best_down < self.swipe_velocity_threshold * 0.55:
            return False

        # After the down peak, require a left hook
        after = pts[best_i:]
        if len(after) < 3:
            return False
        xa, ya, ta = after[0]
        xb, yb, tb = after[-1]
        dx, dy = xb - xa, yb - ya
        if dx >= -self.swipe_disp_threshold * 0.65:
            return False
        if abs(dx) <= 1.2 * abs(dy):
            return False
        if (tb - ta) > 0.22:
            return False
        return True

    def update(
        self,
        lm: np.ndarray | None,
        now: float,
        writing_enabled: bool = True,
        index_extended: bool = False,
        stroke_points: int = 0,
    ) -> GestureResult:
        """Evaluate one frame. Priority: FIST → SWIPE → V-SIGN → ENTER → IDLE/DRAWING."""
        if lm is None:
            self._fist_latched = False
            self._vsign_since = None
            self._vsign_fired = False
            self.clear_motion()
            state = LeftGestureState.COOLDOWN if now < self._writing_lock_until else LeftGestureState.IDLE
            badge, color = self._badge_now(now, state)
            self._last_state = state
            return GestureResult(GestureEvent.NONE, state, badge, color, False, False)

        self._push_motion(lm, now)
        in_cooldown = now < self._cooldown_until
        writing_locked = now < self._writing_lock_until

        # --- A. FIST (highest priority) ---
        if self._is_fist(lm):
            state = LeftGestureState.FIST
            if writing_enabled and not in_cooldown and not self._fist_latched:
                self._fist_latched = True
                self._cooldown_until = now + FIST_COOLDOWN_S
                self._writing_lock_until = now + FIST_COOLDOWN_S
                self.clear_motion()
                badge = "[STATE: FIST -> TAB]"
                self._set_flash(badge, FLASH_GREEN, now)
                self._last_state = state
                return GestureResult(GestureEvent.FIST_TAB, state, badge, FLASH_GREEN, True, False)
            badge, color = self._badge_now(now, state)
            self._last_state = state
            return GestureResult(GestureEvent.NONE, state, badge, color, True, False)
        self._fist_latched = False

        if in_cooldown:
            state = LeftGestureState.COOLDOWN
            badge, color = self._badge_now(now, state)
            self._last_state = state
            # Still purge while cooling after a discrete gesture
            return GestureResult(GestureEvent.NONE, state, badge, color, True, False)

        # --- B. HORIZONTAL SWIPES (isolated from letter strokes) ---
        if writing_enabled and stroke_points < 8:
            swipe = self._detect_swipe(now)
            if swipe is not None:
                if swipe == GestureEvent.SWIPE_LEFT:
                    state = LeftGestureState.SWIPE_LEFT
                    badge = "[STATE: SWIPE LEFT -> BACKSPACE]"
                    color = FLASH_ORANGE
                else:
                    state = LeftGestureState.SWIPE_RIGHT
                    badge = "[STATE: SWIPE RIGHT -> SPACE]"
                    color = FLASH_BLUE
                self._cooldown_until = now + SWIPE_COOLDOWN_S
                self._writing_lock_until = now + SWIPE_COOLDOWN_S
                self.clear_motion()
                self._vsign_since = None
                self._set_flash(badge, color, now)
                self._last_state = state
                return GestureResult(swipe, state, badge, color, True, False)

        # --- C. V-SIGN / PEACE (language) ---
        if self._is_vsign(lm):
            state = LeftGestureState.V_SIGN
            if self._vsign_since is None:
                self._vsign_since = now
            elif (
                writing_enabled
                and not self._vsign_fired
                and (now - self._vsign_since) >= VSIGN_HOLD_S
            ):
                self._vsign_fired = True
                self._cooldown_until = now + VSIGN_COOLDOWN_S
                self._writing_lock_until = now + VSIGN_COOLDOWN_S
                self.clear_motion()
                badge = "[STATE: V-SIGN -> LANG TOGGLE]"
                self._set_flash(badge, FLASH_PURPLE, now)
                self._last_state = state
                return GestureResult(GestureEvent.V_SIGN_LANG, state, badge, FLASH_PURPLE, True, False)
            badge, color = self._badge_now(now, state)
            self._last_state = state
            return GestureResult(GestureEvent.NONE, state, badge, color, True, False)
        self._vsign_since = None
        self._vsign_fired = False

        # --- D. ENTER corner stroke (down then left hook) ---
        # Only when not mid-character (keeps hooks from eating letter strokes).
        if writing_enabled and stroke_points < 8 and self._detect_enter_hook(now):
            state = LeftGestureState.ENTER
            badge = "[STATE: ENTER -> RETURN]"
            self._cooldown_until = now + ENTER_COOLDOWN_S
            self._writing_lock_until = now + ENTER_COOLDOWN_S
            self.clear_motion()
            self._set_flash(badge, FLASH_CYAN, now)
            self._last_state = state
            return GestureResult(GestureEvent.ENTER, state, badge, FLASH_CYAN, True, False)

        # --- AIR-WRITING / IDLE ---
        if not writing_enabled or writing_locked:
            state = LeftGestureState.COOLDOWN if writing_locked else LeftGestureState.IDLE
            badge, color = self._badge_now(now, state)
            self._last_state = state
            return GestureResult(GestureEvent.NONE, state, badge, color, writing_locked, False)

        if index_extended:
            state = LeftGestureState.DRAWING
            badge, color = self._badge_now(now, state)
            self._last_state = state
            return GestureResult(GestureEvent.NONE, state, badge, color, False, True)

        state = LeftGestureState.IDLE
        badge, color = self._badge_now(now, state)
        self._last_state = state
        return GestureResult(GestureEvent.NONE, state, badge, color, False, False)
