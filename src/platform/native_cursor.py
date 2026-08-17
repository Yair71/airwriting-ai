"""Low-overhead native cursor backend with a pynput fallback."""

from __future__ import annotations

import ctypes
import sys

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
WHEEL_DELTA = 120


class NativeCursor:
    def __init__(self) -> None:
        self._controller = None
        self._button = None
        if sys.platform == "win32":
            self._user32 = ctypes.windll.user32
            self._width = max(int(self._user32.GetSystemMetrics(0)), 1)
            self._height = max(int(self._user32.GetSystemMetrics(1)), 1)
        else:  # pragma: no cover - Windows is the primary runtime
            from pynput.mouse import Button, Controller

            self._user32 = None
            self._controller = Controller()
            self._button = Button
            self._width, self._height = self._fallback_screen_size()

    @staticmethod
    def _fallback_screen_size() -> tuple[int, int]:  # pragma: no cover
        try:
            import tkinter

            root = tkinter.Tk()
            root.withdraw()
            size = (root.winfo_screenwidth(), root.winfo_screenheight())
            root.destroy()
            return max(int(size[0]), 1), max(int(size[1]), 1)
        except Exception:
            return 1920, 1080

    def screen_size(self) -> tuple[int, int]:
        return self._width, self._height

    def position(self) -> tuple[int, int]:
        if self._user32 is not None:
            from ctypes import wintypes

            point = wintypes.POINT()
            if not self._user32.GetCursorPos(ctypes.byref(point)):
                raise ctypes.WinError()
            return int(point.x), int(point.y)
        x, y = self._controller.position  # pragma: no cover
        return int(x), int(y)

    def move_pointer(self, x: int, y: int) -> None:
        px = min(max(int(x), 0), self._width - 1)
        py = min(max(int(y), 0), self._height - 1)
        if self._user32 is not None:
            if not self._user32.SetCursorPos(px, py):
                raise ctypes.WinError()
        else:  # pragma: no cover
            self._controller.position = (px, py)

    def move_pointer_norm(self, x: float, y: float) -> None:
        nx = min(max(float(x), 0.0), 1.0)
        ny = min(max(float(y), 0.0), 1.0)
        self.move_pointer(
            round(nx * (self._width - 1)),
            round(ny * (self._height - 1)),
        )

    def _event(self, flag: int) -> None:
        if self._user32 is not None:
            self._user32.mouse_event(flag, 0, 0, 0, 0)

    def left_down(self) -> None:
        if self._controller is not None:  # pragma: no cover
            self._controller.press(self._button.left)
        else:
            self._event(MOUSEEVENTF_LEFTDOWN)

    def left_up(self) -> None:
        if self._controller is not None:  # pragma: no cover
            self._controller.release(self._button.left)
        else:
            self._event(MOUSEEVENTF_LEFTUP)

    def right_down(self) -> None:
        if self._controller is not None:  # pragma: no cover
            self._controller.press(self._button.right)
        else:
            self._event(MOUSEEVENTF_RIGHTDOWN)

    def right_up(self) -> None:
        if self._controller is not None:  # pragma: no cover
            self._controller.release(self._button.right)
        else:
            self._event(MOUSEEVENTF_RIGHTUP)

    def left_click(self) -> None:
        self.left_down()
        self.left_up()

    def right_click(self) -> None:
        self.right_down()
        self.right_up()

    def scroll(self, notches: int) -> None:
        if notches == 0:
            return
        if self._controller is not None:  # pragma: no cover
            self._controller.scroll(0, int(notches))
        else:
            data = ctypes.c_uint32(int(notches) * WHEEL_DELTA).value
            self._user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, data, 0)
