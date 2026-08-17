"""Active Windows keyboard layout (input locale) for language gating."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from enum import Enum

if sys.platform != "win32":  # pragma: no cover
    raise ImportError("keyboard_layout is Windows-only")

user32 = ctypes.windll.user32

LANG_EN = 0x0409
LANG_RU = 0x0419
LANG_HE = 0x040D


class InputLang(str, Enum):
    EN = "en"
    RU = "ru"
    HE = "he"
    OTHER = "other"


def get_foreground_layout_id() -> int:
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return int(user32.GetKeyboardLayout(0)) & 0xFFFF
    tid = user32.GetWindowThreadProcessId(hwnd, None)
    return int(user32.GetKeyboardLayout(tid)) & 0xFFFF


def layout_to_lang(lang_id: int | None = None) -> InputLang:
    lid = get_foreground_layout_id() if lang_id is None else int(lang_id) & 0xFFFF
    if lid == LANG_EN:
        return InputLang.EN
    if lid == LANG_RU:
        return InputLang.RU
    if lid == LANG_HE:
        return InputLang.HE
    # Primary language mask (low 10 bits sometimes; use full compare of primary)
    primary = lid & 0xFF
    if primary == 0x09:
        return InputLang.EN
    if primary == 0x19:
        return InputLang.RU
    if primary == 0x0D:
        return InputLang.HE
    return InputLang.OTHER


def script_of_char(ch: str) -> InputLang | None:
    if not ch or ch in {"SPACE", "BACKSPACE", "ENTER", "TAB"}:
        return None
    o = ord(ch[0])
    if ord("0") <= o <= ord("9") or ord("A") <= o <= ord("Z") or ord("a") <= o <= ord("z"):
        return InputLang.EN
    if 0x0400 <= o <= 0x04FF:
        return InputLang.RU
    if 0x0590 <= o <= 0x05FF:
        return InputLang.HE
    return InputLang.OTHER


def label_allowed_for_lang(label: str, lang: InputLang) -> bool:
    if label in {"SPACE", "BACKSPACE", "ENTER", "TAB"}:
        return True
    if len(label) == 1 and label.isdigit():
        return True
    script = script_of_char(label)
    if script is None:
        return True
    if lang == InputLang.OTHER:
        return True
    if lang == InputLang.EN:
        return script == InputLang.EN
    if lang == InputLang.RU:
        return script == InputLang.RU
    if lang == InputLang.HE:
        return script == InputLang.HE
    return True


def charset_mask(charset: list[str], lang: InputLang) -> list[bool]:
    return [label_allowed_for_lang(c, lang) for c in charset]
