"""Single-latch physical-right-hand mouse controller."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from src.vision.bone_angles import (
    INDEX_TIP,
    MIDDLE_TIP,
    THUMB_TIP,
    Finger,
    finger_curled,
    finger_extended,
    normalized_distance,
)
from src.vision.dual_hand_router import HandTrackId, RoutedHandData
from src.vision.one_euro import OneEuroFilter2D

MIN_CONFIDENCE = 0.80
MIN_CUTOFF = 1.0
BETA = 0.1
FILTER_BYPASS_SPEED_PX_S = 150.0
BOX_MIN = 0.15
BOX_MAX = 0.85
LMB_CONTACT = 0.22
LMB_RELEASE = 0.34
RMB_CONTACT = 0.20
RMB_RELEASE = 0.30
LMB_FREEZE_S = 0.180
DRAG_DISTANCE_PX = 20.0
POINTER_DEADZONE_PX = 2.5


class MouseMode(str, Enum):
    MOUSE_IDLE = "MOUSE_IDLE"
    MOUSE_HOVER = "MOUSE_HOVER"
    LMB_PENDING = "LMB_PENDING"
    LMB_DRAGGING = "LMB_DRAGGING"
    RMB_CLICKED = "RMB_CLICKED"
    MOUSE_SCROLL = "MOUSE_SCROLL"
    # Compatibility aliases for preview code.
    IDLE = "MOUSE_IDLE"
    HOVER = "MOUSE_HOVER"
    LMB = "LMB_PENDING"
    RMB = "RMB_CLICKED"
    SCROLL = "MOUSE_SCROLL"


@dataclass
class MouseCallbacks:
    move: Callable[[float, float], None] | None = None
    position: Callable[[], tuple[int, int]] | None = None
    screen_size: Callable[[], tuple[int, int]] | None = None
    left_down: Callable[[], None] | None = None
    left_up: Callable[[], None] | None = None
    right_down: Callable[[], None] | None = None
    right_up: Callable[[], None] | None = None
    left_click: Callable[[], None] | None = None
    right_click: Callable[[], None] | None = None
    scroll: Callable[[int], None] | None = None


@dataclass
class _MouseState:
    tip_filter: OneEuroFilter2D = field(
        default_factory=lambda: OneEuroFilter2D(MIN_CUTOFF, BETA)
    )
    mode: MouseMode = MouseMode.MOUSE_IDLE
    xy: tuple[float, float] | None = None
    contact_t: float = 0.0
    contact_hand_px: tuple[float, float] | None = None
    freeze_xy: tuple[float, float] | None = None
    last_raw_xy: tuple[float, float] | None = None
    last_raw_t: float | None = None
    last_stable_px: tuple[float, float] | None = None
    last_scroll_y: float | None = None
    scroll_lock_xy: tuple[float, float] | None = None


class MouseController:
    """One interaction state, one event transition, no frame-level spam."""

    def __init__(self, callbacks: MouseCallbacks | None = None) -> None:
        self.callbacks = callbacks or MouseCallbacks()
        self._state = _MouseState()
        self._mutex = threading.RLock()

    @property
    def xy(self) -> tuple[float, float] | None:
        return self._state.xy

    @property
    def mode(self) -> MouseMode:
        return self._state.mode

    @property
    def lmb_active(self) -> bool:
        return self._state.mode in {
            MouseMode.LMB_PENDING,
            MouseMode.LMB_DRAGGING,
        }

    @property
    def rmb_active(self) -> bool:
        return self._state.mode == MouseMode.RMB_CLICKED

    @staticmethod
    def _map_to_screen(x: float, y: float) -> tuple[float, float]:
        span = BOX_MAX - BOX_MIN
        return (
            float(np.clip((x - BOX_MIN) / span, 0.0, 1.0)),
            float(np.clip((y - BOX_MIN) / span, 0.0, 1.0)),
        )

    def _screen_size(self) -> tuple[int, int]:
        if self.callbacks.screen_size:
            width, height = self.callbacks.screen_size()
            return max(int(width), 1), max(int(height), 1)
        return 1920, 1080

    @staticmethod
    def _to_pixel(
        xy: tuple[float, float], width: int, height: int
    ) -> tuple[float, float]:
        return xy[0] * max(width - 1, 1), xy[1] * max(height - 1, 1)

    @staticmethod
    def _to_normalized(
        pixel: tuple[float, float], width: int, height: int
    ) -> tuple[float, float]:
        return (
            float(np.clip(pixel[0] / max(width - 1, 1), 0.0, 1.0)),
            float(np.clip(pixel[1] / max(height - 1, 1), 0.0, 1.0)),
        )

    def _move(self, xy: tuple[float, float]) -> None:
        self._state.xy = xy
        if self.callbacks.move:
            self.callbacks.move(*xy)

    def _left_down_once(self) -> None:
        if self.callbacks.left_down:
            self.callbacks.left_down()

    def _left_up_once(self) -> None:
        if self.callbacks.left_up:
            self.callbacks.left_up()

    def _left_click_once(self) -> None:
        if self.callbacks.left_click:
            self.callbacks.left_click()
        else:
            self._left_down_once()
            self._left_up_once()

    def _right_click_once(self) -> None:
        if self.callbacks.right_click:
            self.callbacks.right_click()
            return
        if self.callbacks.right_down:
            self.callbacks.right_down()
        if self.callbacks.right_up:
            self.callbacks.right_up()

    def _clear_interaction(self, mode: MouseMode = MouseMode.MOUSE_HOVER) -> None:
        self._state.mode = mode
        self._state.contact_t = 0.0
        self._state.contact_hand_px = None
        self._state.freeze_xy = None

    def suspend(self) -> None:
        """Release an active drag once, then discard every tracked coordinate."""
        with self._mutex:
            if self._state.mode == MouseMode.LMB_DRAGGING:
                self._left_up_once()
            self._clear_interaction(MouseMode.MOUSE_IDLE)
            self._state.xy = None
            self._state.last_raw_xy = None
            self._state.last_raw_t = None
            self._state.last_stable_px = None
            self._state.last_scroll_y = None
            self._state.scroll_lock_xy = None
            self._state.tip_filter.reset()

    def _filtered_pointer(
        self,
        lm: np.ndarray,
        now: float,
        width: int,
        height: int,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        raw_tip = (float(lm[INDEX_TIP, 0]), float(lm[INDEX_TIP, 1]))
        raw_xy = self._map_to_screen(*raw_tip)
        raw_speed = 0.0
        if self._state.last_raw_xy is not None and self._state.last_raw_t is not None:
            dt = max(now - self._state.last_raw_t, 1e-6)
            old_px = self._to_pixel(self._state.last_raw_xy, width, height)
            new_px = self._to_pixel(raw_xy, width, height)
            raw_speed = float(np.hypot(new_px[0] - old_px[0], new_px[1] - old_px[1])) / dt
        self._state.last_raw_xy = raw_xy
        self._state.last_raw_t = now

        if raw_speed > FILTER_BYPASS_SPEED_PX_S:
            self._state.tip_filter.reset()
            xy = raw_xy
        else:
            xy = self._map_to_screen(*self._state.tip_filter(*raw_tip, now))

        raw_pixel = self._to_pixel(xy, width, height)
        stable_pixel = raw_pixel
        if self._state.last_stable_px is not None:
            delta = float(
                np.hypot(
                    raw_pixel[0] - self._state.last_stable_px[0],
                    raw_pixel[1] - self._state.last_stable_px[1],
                )
            )
            if delta < POINTER_DEADZONE_PX:
                stable_pixel = self._state.last_stable_px
        self._state.last_stable_px = stable_pixel
        return self._to_normalized(stable_pixel, width, height), raw_pixel

    def _start_lmb_pending(
        self,
        hand_pixel: tuple[float, float],
        fallback_xy: tuple[float, float],
        now: float,
        width: int,
        height: int,
    ) -> None:
        if self.callbacks.position:
            cursor_pixel = self.callbacks.position()
            freeze_xy = self._to_normalized(cursor_pixel, width, height)
        else:
            freeze_xy = fallback_xy
        self._state.mode = MouseMode.LMB_PENDING
        self._state.contact_t = now
        self._state.contact_hand_px = hand_pixel
        self._state.freeze_xy = freeze_xy
        self._move(freeze_xy)

    def _update_lmb(
        self,
        distance: float,
        pointer_xy: tuple[float, float],
        hand_pixel: tuple[float, float],
        now: float,
    ) -> None:
        if self._state.mode == MouseMode.LMB_DRAGGING:
            if distance > LMB_RELEASE:
                self._left_up_once()
                self._clear_interaction()
            else:
                self._move(pointer_xy)
            return

        freeze_xy = self._state.freeze_xy or pointer_xy
        elapsed = now - self._state.contact_t
        if distance > LMB_RELEASE:
            self._move(freeze_xy)
            if elapsed <= LMB_FREEZE_S:
                self._left_click_once()
            self._clear_interaction()
            return

        movement = 0.0
        if self._state.contact_hand_px is not None:
            movement = float(
                np.hypot(
                    hand_pixel[0] - self._state.contact_hand_px[0],
                    hand_pixel[1] - self._state.contact_hand_px[1],
                )
            )
        if elapsed > LMB_FREEZE_S and movement > DRAG_DISTANCE_PX:
            self._state.mode = MouseMode.LMB_DRAGGING
            self._move(pointer_xy)
            self._left_down_once()
            return

        # Reassert the anchor while pending; no button event is emitted.
        self._move(freeze_xy)

    def _update_scroll(
        self,
        lm: np.ndarray,
        pointer_xy: tuple[float, float],
    ) -> None:
        self._state.mode = MouseMode.MOUSE_SCROLL
        middle_y = 0.5 * (
            float(lm[INDEX_TIP, 1]) + float(lm[MIDDLE_TIP, 1])
        )
        if self._state.scroll_lock_xy is None:
            self._state.scroll_lock_xy = pointer_xy
        self._move(self._state.scroll_lock_xy)
        if self._state.last_scroll_y is not None:
            delta_y = middle_y - self._state.last_scroll_y
            if abs(delta_y) > 0.010 and self.callbacks.scroll:
                self.callbacks.scroll(-1 if delta_y > 0 else 1)
                self._state.last_scroll_y = middle_y
        else:
            self._state.last_scroll_y = middle_y

    def update(self, hand: RoutedHandData, now: float) -> bool:
        if (
            hand is None
            or hand.track_id != HandTrackId.RIGHT_HAND_MOUSE
            or hand.confidence < MIN_CONFIDENCE
        ):
            return False

        with self._mutex:
            lm = hand.landmarks
            width, height = self._screen_size()
            pointer_xy, raw_hand_pixel = self._filtered_pointer(
                lm, now, width, height
            )
            index_distance = normalized_distance(lm, THUMB_TIP, INDEX_TIP)
            middle_distance = normalized_distance(lm, THUMB_TIP, MIDDLE_TIP)

            if self._state.mode in {
                MouseMode.LMB_PENDING,
                MouseMode.LMB_DRAGGING,
            }:
                self._update_lmb(
                    index_distance,
                    pointer_xy,
                    raw_hand_pixel,
                    now,
                )
                return True

            if self._state.mode == MouseMode.RMB_CLICKED:
                if middle_distance > RMB_RELEASE:
                    self._clear_interaction()
                return True

            if index_distance < LMB_CONTACT:
                self._start_lmb_pending(
                    raw_hand_pixel,
                    pointer_xy,
                    now,
                    width,
                    height,
                )
                return True

            index_available_for_rmb = (
                finger_extended(lm, Finger.INDEX)
                and index_distance > LMB_RELEASE
            )
            if middle_distance < RMB_CONTACT and index_available_for_rmb:
                self._right_click_once()
                self._state.mode = MouseMode.RMB_CLICKED
                self._state.xy = pointer_xy
                return True

            finger_gap = normalized_distance(lm, INDEX_TIP, MIDDLE_TIP)
            scrolling = (
                finger_extended(lm, Finger.INDEX)
                and finger_extended(lm, Finger.MIDDLE)
                and finger_curled(lm, Finger.RING)
                and finger_curled(lm, Finger.PINKY)
                and finger_gap < 0.35
                and index_distance > LMB_RELEASE
                and middle_distance > RMB_RELEASE
            )
            if scrolling:
                self._update_scroll(lm, pointer_xy)
                return True

            self._state.last_scroll_y = None
            self._state.scroll_lock_xy = None
            self._state.mode = MouseMode.MOUSE_HOVER
            pointing = (
                finger_extended(lm, Finger.INDEX)
                and finger_curled(lm, Finger.MIDDLE)
                and finger_curled(lm, Finger.RING)
                and finger_curled(lm, Finger.PINKY)
            )
            if pointing:
                self._move(pointer_xy)
            else:
                self._state.xy = None
            return True
