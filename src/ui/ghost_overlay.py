"""Native glassmorphism ghost-text overlay via Win32 UpdateLayeredWindow.

Per-pixel alpha (AC_SRC_ALPHA) — no color-key, no solid black slab.
Click-through, topmost, never activates / steals focus.
"""

from __future__ import annotations

import sys
import threading
import time
from ctypes import wintypes

if sys.platform != "win32":  # pragma: no cover
    raise ImportError("ghost_overlay is Windows-only")

import ctypes

from PIL import Image, ImageDraw, ImageFont

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

# --- Win32 constants ---
WS_POPUP = 0x80000000
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_TOPMOST = 0x00000008
WS_EX_NOACTIVATE = 0x08000000
WS_EX_FLAGS = WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST

HWND_TOPMOST = -1
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
SWP_HIDEWINDOW = 0x0080
SW_HIDE = 0
SW_SHOWNOACTIVATE = 4

ULW_ALPHA = 0x00000002
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01

BI_RGB = 0
DIB_RGB_COLORS = 0
SRCCOPY = 0x00CC0020

WM_DESTROY = 0x0002
WM_QUIT = 0x0012
CS_HREDRAW = 0x0002
CS_VREDRAW = 0x0001

MONITOR_DEFAULTTONEAREST = 2

# Cosmetics (RGBA)
PILL_RGBA = (15, 23, 42, 220)
BORDER_RGBA = (59, 130, 246, 180)
PREFIX_RGB = (255, 255, 255, 255)
GHOST_RGB = (148, 163, 184, 230)
HINT_BG = (14, 116, 144, 200)
HINT_FG = (56, 189, 248, 255)
HINT_LABEL = "[TAB / ✊]"

OFFSET_X = 18
OFFSET_Y = 24
IDLE_HIDE_S = 3.0
RADIUS = 12
PAD_X = 14
PAD_Y = 10
TICK_MS = 33


class POINT(ctypes.Structure):
    _fields_ = (("x", wintypes.LONG), ("y", wintypes.LONG))


class SIZE(ctypes.Structure):
    _fields_ = (("cx", wintypes.LONG), ("cy", wintypes.LONG))


class RECT(ctypes.Structure):
    _fields_ = (
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    )


class MONITORINFO(ctypes.Structure):
    _fields_ = (
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", RECT),
        ("rcWork", RECT),
        ("dwFlags", wintypes.DWORD),
    )


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = (
        ("BlendOp", ctypes.c_byte),
        ("BlendFlags", ctypes.c_byte),
        ("SourceConstantAlpha", ctypes.c_byte),
        ("AlphaFormat", ctypes.c_byte),
    )


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = (
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    )


class BITMAPINFO(ctypes.Structure):
    _fields_ = (("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3))


class MSG(ctypes.Structure):
    _fields_ = (
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", POINT),
    )


LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)


class WNDCLASSW(ctypes.Structure):
    _fields_ = (
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    )


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\SegoeUI-VF.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    if not text:
        return 0
    try:
        return int(draw.textlength(text, font=font))
    except AttributeError:
        box = draw.textbbox((0, 0), text, font=font)
        return int(box[2] - box[0])


def _rounded_rect(draw: ImageDraw.ImageDraw, box, radius: int, fill, outline=None, width: int = 1) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _rgba_to_premul_bgra(img: Image.Image) -> bytes:
    """Pillow RGBA → premultiplied BGRA bytes for UpdateLayeredWindow."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    px = img.tobytes("raw", "RGBA")
    # Premultiply and swap to BGRA
    out = bytearray(len(px))
    for i in range(0, len(px), 4):
        r, g, b, a = px[i], px[i + 1], px[i + 2], px[i + 3]
        if a == 255:
            out[i] = b
            out[i + 1] = g
            out[i + 2] = r
            out[i + 3] = a
        elif a == 0:
            out[i] = out[i + 1] = out[i + 2] = out[i + 3] = 0
        else:
            out[i] = (b * a + 127) // 255
            out[i + 1] = (g * a + 127) // 255
            out[i + 2] = (r * a + 127) // 255
            out[i + 3] = a
    return bytes(out)


def _work_area_for_point(x: int, y: int) -> tuple[int, int, int, int]:
    pt = POINT(x, y)
    mon = user32.MonitorFromPoint(pt, MONITOR_DEFAULTTONEAREST)
    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    if mon and user32.GetMonitorInfoW(mon, ctypes.byref(info)):
        r = info.rcWork
        return int(r.left), int(r.top), int(r.right), int(r.bottom)
    w = int(user32.GetSystemMetrics(0))
    h = int(user32.GetSystemMetrics(1))
    return 0, 0, w, h


def _clamp_pos(x: int, y: int, ww: int, hh: int) -> tuple[int, int]:
    left, top, right, bottom = _work_area_for_point(x, y)
    x = max(left + 4, min(x, right - ww - 4))
    y = max(top + 4, min(y, bottom - hh - 4))
    return x, y


class GhostOverlay:
    """Threaded layered HWND with glassmorphic ghost-text pill."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._active = ""
        self._ghost = ""
        self._mode = "Writing"
        self._visible = True
        self._caret: tuple[int, int, int, int] | None = None
        self._pos = (40.0, 40.0)
        self._last_activity = 0.0
        self._hwnd: int = 0
        self._class_atom = 0
        self._wndproc_ref = None  # keep callback alive
        self._font_prefix = _load_font(16, bold=True)
        self._font_ghost = _load_font(16, bold=False)
        self._font_hint = _load_font(11, bold=True)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ghost-overlay", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=4.0)

    def stop(self) -> None:
        self._stop.set()
        hwnd = self._hwnd
        if hwnd:
            try:
                user32.PostMessageW(hwnd, WM_QUIT, 0, 0)
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._ready.clear()
        self._hwnd = 0

    def update(
        self,
        active: str,
        suggestion: str | None,
        mode: str | None = None,
        caret_rect: tuple[int, int, int, int] | None = None,
    ) -> None:
        with self._lock:
            self._active = active or ""
            if suggestion and suggestion.startswith(self._active):
                self._ghost = suggestion[len(self._active) :]
            elif suggestion and not self._active:
                self._ghost = suggestion
            else:
                self._ghost = ""
            if mode is not None:
                self._mode = mode
            self._caret = caret_rect
            if self._active or self._ghost:
                self._last_activity = time.perf_counter()

    def update_text(self, prefix: str, suggestion: str | None) -> None:
        """Trie / runtime convenience alias."""
        self.update(prefix, suggestion)

    def set_visible(self, visible: bool) -> None:
        with self._lock:
            self._visible = bool(visible)

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self._mode = mode

    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        if msg == WM_DESTROY:
            user32.PostQuitMessage(0)
            return 0
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _create_window(self) -> int:
        hinst = kernel32.GetModuleHandleW(None)
        class_name = "AirTouchGhostOverlay"
        self._wndproc_ref = WNDPROC(self._wnd_proc)
        wc = WNDCLASSW()
        wc.style = CS_HREDRAW | CS_VREDRAW
        wc.lpfnWndProc = self._wndproc_ref
        wc.hInstance = hinst
        wc.hCursor = user32.LoadCursorW(None, 32512)  # IDC_ARROW
        wc.lpszClassName = class_name
        atom = user32.RegisterClassW(ctypes.byref(wc))
        if not atom:
            # Already registered in this process
            atom = 1
        self._class_atom = atom
        hwnd = user32.CreateWindowExW(
            WS_EX_FLAGS,
            class_name,
            "AirTouch Ghost",
            WS_POPUP,
            0,
            0,
            10,
            10,
            None,
            None,
            hinst,
            None,
        )
        if not hwnd:
            raise OSError("CreateWindowExW failed for ghost overlay")
        # Re-assert extended styles (some Windows builds strip on create)
        style = user32.GetWindowLongW(hwnd, -20)
        user32.SetWindowLongW(hwnd, -20, style | WS_EX_FLAGS)
        user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOACTIVATE | 0x0001 | 0x0002)  # NOSIZE|NOMOVE
        return int(hwnd)

    def _target_pos(self) -> tuple[int, int]:
        with self._lock:
            caret = self._caret
        if caret is not None:
            _l, _t, r, b = caret
            return int(r) + 10, int(b) + 6
        pt = POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return int(pt.x) + OFFSET_X, int(pt.y) + OFFSET_Y

    def _should_show(self) -> bool:
        with self._lock:
            visible = self._visible
            active = self._active
            ghost = self._ghost
            last = self._last_activity
        if not visible:
            return False
        if not (active or ghost):
            return False
        if last > 0.0 and (time.perf_counter() - last) > IDLE_HIDE_S:
            return False
        return True

    def _render_pill(self, active: str, ghost: str) -> Image.Image:
        hint = HINT_LABEL if ghost else ""
        # Measure
        tmp = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        d0 = ImageDraw.Draw(tmp)
        w_active = _text_width(d0, active, self._font_prefix)
        w_ghost = _text_width(d0, ghost, self._font_ghost)
        w_hint = _text_width(d0, hint, self._font_hint)
        hint_pad = 10 if hint else 0
        hint_w = (w_hint + 16) if hint else 0
        gap = 10 if hint else 0
        text_w = w_active + w_ghost
        width = max(text_w + hint_w + gap, 72) + PAD_X * 2
        height = 36 + PAD_Y
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        _rounded_rect(
            draw,
            (1, 1, width - 2, height - 2),
            RADIUS,
            fill=PILL_RGBA,
            outline=BORDER_RGBA,
            width=2,
        )
        baseline = height // 2
        x = PAD_X
        if active:
            draw.text((x, baseline), active, font=self._font_prefix, fill=PREFIX_RGB, anchor="lm")
            x += w_active
        if ghost:
            draw.text((x, baseline), ghost, font=self._font_ghost, fill=GHOST_RGB, anchor="lm")
            x += w_ghost
        if hint:
            hx1 = width - PAD_X - hint_w
            hy1 = (height - 22) // 2
            hx2 = width - PAD_X
            hy2 = hy1 + 22
            _rounded_rect(draw, (hx1, hy1, hx2, hy2), 8, fill=HINT_BG, outline=HINT_FG, width=1)
            draw.text(((hx1 + hx2) // 2, (hy1 + hy2) // 2), hint, font=self._font_hint, fill=HINT_FG, anchor="mm")
        return img

    def _present(self, hwnd: int, img: Image.Image, x: int, y: int) -> None:
        w, h = img.size
        bgra = _rgba_to_premul_bgra(img)

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h  # top-down
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB

        hdc_screen = user32.GetDC(None)
        hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
        bits = ctypes.c_void_p()
        hbmp = gdi32.CreateDIBSection(
            hdc_mem,
            ctypes.byref(bmi),
            DIB_RGB_COLORS,
            ctypes.byref(bits),
            None,
            0,
        )
        if not hbmp or not bits.value:
            gdi32.DeleteDC(hdc_mem)
            user32.ReleaseDC(None, hdc_screen)
            return

        ctypes.memmove(bits, bgra, len(bgra))
        old = gdi32.SelectObject(hdc_mem, hbmp)

        size = SIZE(w, h)
        pt_dst = POINT(x, y)
        pt_src = POINT(0, 0)
        blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)

        user32.UpdateLayeredWindow(
            hwnd,
            hdc_screen,
            ctypes.byref(pt_dst),
            ctypes.byref(size),
            hdc_mem,
            ctypes.byref(pt_src),
            0,
            ctypes.byref(blend),
            ULW_ALPHA,
        )

        gdi32.SelectObject(hdc_mem, old)
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(None, hdc_screen)

    def _run(self) -> None:
        hwnd = 0
        try:
            hwnd = self._create_window()
            self._hwnd = hwnd
            self._ready.set()
            msg = MSG()
            while not self._stop.is_set():
                # Drain messages without blocking the tick
                while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
                    if msg.message == WM_QUIT:
                        self._stop.set()
                        break
                    user32.TranslateMessage(ctypes.byref(msg))
                    user32.DispatchMessageW(ctypes.byref(msg))
                if self._stop.is_set():
                    break

                if not self._should_show():
                    user32.ShowWindow(hwnd, SW_HIDE)
                    time.sleep(TICK_MS / 1000.0)
                    continue

                with self._lock:
                    active = self._active
                    ghost = self._ghost

                img = self._render_pill(active, ghost)
                tx, ty = self._target_pos()
                cx, cy = self._pos
                self._pos = (cx + (tx - cx) * 0.38, cy + (ty - cy) * 0.38)
                x, y = _clamp_pos(int(self._pos[0]), int(self._pos[1]), img.size[0], img.size[1])
                self._present(hwnd, img, x, y)
                user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
                user32.SetWindowPos(
                    hwnd,
                    HWND_TOPMOST,
                    0,
                    0,
                    0,
                    0,
                    SWP_NOACTIVATE | 0x0001 | 0x0002 | SWP_SHOWWINDOW,
                )
                time.sleep(TICK_MS / 1000.0)
        except Exception:
            self._ready.set()
            raise
        finally:
            if hwnd:
                try:
                    user32.DestroyWindow(hwnd)
                except Exception:
                    pass
            self._hwnd = 0
