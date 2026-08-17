"""Platform package."""

from src.platform.focus_detector import FocusDetector, FocusInfo
from src.platform.keyboard_layout import InputLang, layout_to_lang
from src.platform.win_injector import WinInjector

__all__ = ["FocusDetector", "FocusInfo", "InputLang", "WinInjector", "layout_to_lang"]

