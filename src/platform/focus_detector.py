"""Detect whether the foreground window has an editable text focus / caret.

Uses GetGUIThreadInfo (GUI_CARETBLINKING, hwndCaret) and optional UIA Edit/Document.
CPU: ~0.1–0.5 ms per poll on Windows.
"""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass

if sys.platform != "win32":  # pragma: no cover
    raise ImportError("focus_detector is Windows-only")

user32 = ctypes.windll.user32

# When False, FocusDetector always reports text-focused Writing mode (no gating).
ENABLE_FOCUS_GATING = False

GUI_CARETBLINKING = 0x00000001
GUI_INMOVESIZE = 0x00000002
GUI_INMENUMODE = 0x00000004
GUI_SYSTEMMENUMODE = 0x00000008
GUI_POPUPMENUMODE = 0x00000010

# Common edit class names (fallback when UIA unavailable)
_EDIT_CLASSES = {
    "edit",
    "richedit",
    "richedit20a",
    "richedit20w",
    "richedit50w",
    "textfield",
    "chrome_renderwidgethosthwnd",  # Chromium content (best-effort)
    "notepad",
}


class RECT(ctypes.Structure):
    _fields_ = (
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    )


class GUITHREADINFO(ctypes.Structure):
    _fields_ = (
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hwndActive", wintypes.HWND),
        ("hwndFocus", wintypes.HWND),
        ("hwndCapture", wintypes.HWND),
        ("hwndMenuOwner", wintypes.HWND),
        ("hwndMoveSize", wintypes.HWND),
        ("hwndCaret", wintypes.HWND),
        ("rcCaret", RECT),
    )


@dataclass(frozen=True)
class FocusInfo:
    text_focused: bool
    hwnd_focus: int
    hwnd_caret: int
    caret_rect: tuple[int, int, int, int] | None  # screen L,T,R,B
    mode_label: str  # "Writing" | "Mouse Only"


class FocusDetector:
    def __init__(self, poll_hz: float = 20.0) -> None:
        self._min_interval = 1.0 / max(poll_hz, 1.0)
        self._last_t = 0.0
        self._cached = FocusInfo(False, 0, 0, None, "Mouse Only")
        self._uia = None
        try:
            import comtypes  # noqa: F401
            from comtypes.client import CreateObject

            # UIAutomationCore late bind
            self._uia = CreateObject("UIAutomationClient.CUIAutomation")
        except Exception:
            self._uia = None

    def poll(self, force: bool = False) -> FocusInfo:
        now = time.perf_counter()
        if not force and (now - self._last_t) < self._min_interval:
            return self._cached
        self._last_t = now
        if not ENABLE_FOCUS_GATING:
            # Bypass: never block left-hand writing / text injection.
            info = FocusInfo(
                text_focused=True,
                hwnd_focus=0,
                hwnd_caret=0,
                caret_rect=None,
                mode_label="Writing",
            )
            self._cached = info
            return info
        info = self._probe()
        self._cached = info
        return info

    def _probe(self) -> FocusInfo:
        fg = user32.GetForegroundWindow()
        if not fg:
            return FocusInfo(False, 0, 0, None, "Mouse Only")

        tid = user32.GetWindowThreadProcessId(fg, None)
        gti = GUITHREADINFO()
        gti.cbSize = ctypes.sizeof(GUITHREADINFO)
        ok = user32.GetGUIThreadInfo(tid, ctypes.byref(gti))
        hwnd_focus = int(gti.hwndFocus or 0) if ok else 0
        hwnd_caret = int(gti.hwndCaret or 0) if ok else 0
        caret_blink = bool(ok and (gti.flags & GUI_CARETBLINKING))
        caret_rect = None
        if ok and hwnd_caret:
            rc = gti.rcCaret
            # rcCaret is client coords of caret window — map to screen
            pt = wintypes.POINT(rc.left, rc.top)
            user32.ClientToScreen(gti.hwndCaret, ctypes.byref(pt))
            pt2 = wintypes.POINT(rc.right, rc.bottom)
            user32.ClientToScreen(gti.hwndCaret, ctypes.byref(pt2))
            caret_rect = (int(pt.x), int(pt.y), int(pt2.x), int(pt2.y))

        textish = False
        if caret_blink or hwnd_caret:
            textish = True
        if hwnd_focus and self._class_looks_editable(hwnd_focus):
            textish = True
        if not textish and hwnd_focus and self._uia_is_edit(hwnd_focus):
            textish = True

        return FocusInfo(
            text_focused=textish,
            hwnd_focus=hwnd_focus,
            hwnd_caret=hwnd_caret,
            caret_rect=caret_rect,
            mode_label="Writing" if textish else "Mouse Only",
        )

    def _class_looks_editable(self, hwnd: int) -> bool:
        buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, buf, 256)
        name = buf.value.lower()
        if name in _EDIT_CLASSES:
            return True
        if "edit" in name or "rich" in name:
            return True
        return False

    def _uia_is_edit(self, hwnd: int) -> bool:
        if self._uia is None:
            return False
        try:
            element = self._uia.ElementFromHandle(hwnd)
            if element is None:
                return False
            # ControlType: Edit=50004, Document=50030
            ct = int(element.CurrentControlType)
            if ct in {50004, 50030}:
                return True
            # Is keyboard focusable + has value pattern often means text
            return bool(element.CurrentIsKeyboardFocusable) and ct in {50004, 50030, 50000}
        except Exception:
            return False
