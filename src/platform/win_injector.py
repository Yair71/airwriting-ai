"""Windows HID injection via user32.SendInput (mouse + Unicode keyboard).

CPU: ~0.05–0.2 ms per event. Text injection gated by FocusDetector when attached.
"""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes

if sys.platform != "win32":  # pragma: no cover
    raise ImportError("win_injector is Windows-only")

user32 = ctypes.windll.user32

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_MENU = 0x12  # Alt
VK_LWIN = 0x5B
VK_SPACE = 0x20
WHEEL_DELTA = 120

ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class INPUT_UNION(ctypes.Union):
    _fields_ = (("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT))


class INPUT(ctypes.Structure):
    _fields_ = (("type", wintypes.DWORD), ("union", INPUT_UNION))


def _send(inputs: list[INPUT]) -> None:
    n = len(inputs)
    arr = (INPUT * n)(*inputs)
    sent = user32.SendInput(n, ctypes.byref(arr), ctypes.sizeof(INPUT))
    if sent != n:
        raise OSError(f"SendInput sent {sent}/{n} events")


def _mouse_input(flags: int, dx: int = 0, dy: int = 0, data: int = 0) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.union.mi = MOUSEINPUT(dx, dy, data, flags, 0, 0)
    return inp


def _key_vk(vk: int, up: bool = False) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki = KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP if up else 0, 0, 0)
    return inp


def _key_unicode(char: str, up: bool = False) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    flags = KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if up else 0)
    inp.union.ki = KEYBDINPUT(0, ord(char), flags, 0, 0)
    return inp


class WinInjector:
    """Direct OS mouse / keyboard injection with optional text-focus gating."""

    def __init__(self, require_text_focus: bool = True) -> None:
        self.require_text_focus = require_text_focus
        self.text_focused = False
        self._screen_w = user32.GetSystemMetrics(0)
        self._screen_h = user32.GetSystemMetrics(1)

    def set_text_focused(self, focused: bool) -> None:
        self.text_focused = bool(focused)

    def _allow_text(self) -> bool:
        return (not self.require_text_focus) or self.text_focused

    def refresh_metrics(self) -> None:
        self._screen_w = max(user32.GetSystemMetrics(0), 1)
        self._screen_h = max(user32.GetSystemMetrics(1), 1)

    def move_pointer_norm(self, x: float, y: float) -> None:
        """Normalized [0,1] → absolute mouse on primary virtual desktop."""
        self.refresh_metrics()
        x = min(max(float(x), 0.0), 1.0)
        y = min(max(float(y), 0.0), 1.0)
        ax = int(x * 65535.0)
        ay = int(y * 65535.0)
        _send([_mouse_input(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, ax, ay)])

    def left_down(self) -> None:
        _send([_mouse_input(MOUSEEVENTF_LEFTDOWN)])

    def left_up(self) -> None:
        _send([_mouse_input(MOUSEEVENTF_LEFTUP)])

    def right_down(self) -> None:
        _send([_mouse_input(MOUSEEVENTF_RIGHTDOWN)])

    def right_up(self) -> None:
        _send([_mouse_input(MOUSEEVENTF_RIGHTUP)])

    def left_click(self) -> None:
        """Instant LMB click (down + up). Prefer left_down/left_up for pinch hysteresis."""
        self.left_down()
        self.left_up()

    def right_click(self) -> None:
        """Instant RMB click (down + up). Prefer right_down/right_up for pinch hysteresis."""
        self.right_down()
        self.right_up()

    def scroll(self, notches: int) -> None:
        """Inject MOUSEEVENTF_WHEEL (notches * WHEEL_DELTA)."""
        if notches == 0:
            return
        _send([_mouse_input(MOUSEEVENTF_WHEEL, data=int(notches) * WHEEL_DELTA)])

    def tap_vk(self, vk: int) -> None:
        _send([_key_vk(vk, up=False), _key_vk(vk, up=True)])

    def backspace(self) -> None:
        if not self._allow_text():
            return
        self.tap_vk(VK_BACK)

    def enter(self) -> None:
        if not self._allow_text():
            return
        self.tap_vk(VK_RETURN)

    def tab(self) -> None:
        if not self._allow_text():
            return
        self.tap_vk(VK_TAB)

    def space(self) -> None:
        if not self._allow_text():
            return
        self.tap_vk(VK_SPACE)

    def type_text(self, text: str) -> None:
        """Inject Unicode at the focused caret via KEYEVENTF_UNICODE."""
        if not self._allow_text() or not text:
            return
        for ch in text:
            if ch == "\n":
                self.tap_vk(VK_RETURN)
            elif ch == "\t":
                self.tap_vk(VK_TAB)
            elif ch == "\b":
                self.tap_vk(VK_BACK)
            else:
                _send([_key_unicode(ch, up=False), _key_unicode(ch, up=True)])
            time.sleep(0.001)

    def inject_label(self, label: str) -> None:
        if label == "SPACE":
            self.space()
        elif label == "BACKSPACE":
            self.backspace()
        elif label == "ENTER":
            self.enter()
        elif label == "TAB":
            self.tab()
        else:
            self.type_text(label)

    def toggle_keyboard_layout(self, method: str = "alt_shift") -> None:
        """Cycle Windows input language for the foreground thread.

        method:
          - \"alt_shift\": Alt+Shift (default)
          - \"win_space\": Win+Space
        """
        if method == "win_space":
            _send(
                [
                    _key_vk(VK_LWIN, up=False),
                    _key_vk(VK_SPACE, up=False),
                    _key_vk(VK_SPACE, up=True),
                    _key_vk(VK_LWIN, up=True),
                ]
            )
            return
        _send(
            [
                _key_vk(VK_MENU, up=False),
                _key_vk(VK_SHIFT, up=False),
                _key_vk(VK_SHIFT, up=True),
                _key_vk(VK_MENU, up=True),
            ]
        )
