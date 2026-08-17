"""AirTouch daemon: dual-hand FSM, focus gating, layout sync, glass ghost UI."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paths import resource_root

# Master switch: False => left hand never blocked by "Mouse Only" focus gating.
ENABLE_FOCUS_GATING = False


def _build_app(camera: int, preview: bool, mirror: bool):
    """Wire DualTracker → gestures/strokes → ONNX → trie → glass overlay → Win32 inject."""
    from src.autocompletion.trie_engine import GhostTextManager, TrieEngine
    from src.platform.focus_detector import ENABLE_FOCUS_GATING as FD_GATING
    from src.platform.focus_detector import FocusDetector
    from src.platform.keyboard_layout import InputLang, charset_mask, layout_to_lang
    from src.platform.win_injector import WinInjector
    from src.recognition.stroke_classifier import StrokeClassifier
    from src.ui.ghost_overlay import GhostOverlay
    from src.vision.dual_tracker import DualHandCallbacks, DualHandTracker

    focus_gating = bool(ENABLE_FOCUS_GATING and FD_GATING)

    root = resource_root()
    tries = {
        InputLang.EN: TrieEngine(),
        InputLang.RU: TrieEngine(),
        InputLang.HE: TrieEngine(),
    }
    dict_dir = root / "data" / "dictionaries"
    if (dict_dir / "en.txt").is_file():
        tries[InputLang.EN].load_file(dict_dir / "en.txt")
    if (dict_dir / "ru.txt").is_file():
        tries[InputLang.RU].load_file(dict_dir / "ru.txt")
    if (dict_dir / "he.txt").is_file():
        tries[InputLang.HE].load_file(dict_dir / "he.txt")

    lang = layout_to_lang()
    if lang == InputLang.OTHER:
        lang = InputLang.EN
    ghost = GhostTextManager(tries[lang])
    injector = WinInjector(require_text_focus=focus_gating)
    overlay = GhostOverlay()
    focus = FocusDetector(poll_hz=15.0)
    classifier = StrokeClassifier(
        onnx_path=root / "data" / "checkpoints" / "accurate_model.onnx",
        map_path=root / "configs" / "unistroke_map.json",
    )
    state = {"lang": lang, "preview": bool(preview)}

    def sync_overlay(focus_info=None) -> None:
        info = focus_info or focus.poll()
        mode = "Writing" if not focus_gating else info.mode_label
        # TrieEngine → glass GhostOverlay (update_text alias + full update with caret).
        overlay.update(
            ghost.active_prefix,
            ghost.suggestion,
            mode=mode,
            caret_rect=info.caret_rect,
        )

    def refresh_lang() -> None:
        new_lang = layout_to_lang()
        if new_lang == InputLang.OTHER:
            return
        if new_lang != state["lang"]:
            state["lang"] = new_lang
            ghost.trie = tries[new_lang]
            ghost._refresh()
        tracker.set_os_lang(state["lang"].value)
        sync_overlay()

    def gate_from_focus() -> None:
        info = focus.poll()
        if focus_gating:
            injector.set_text_focused(info.text_focused)
            tracker.set_writing_enabled(info.text_focused)
            tracker.mode_label = info.mode_label
            tracker.hud.mode_label = info.mode_label
        else:
            injector.set_text_focused(True)
            tracker.set_writing_enabled(True)
            tracker.mode_label = "Writing"
            tracker.hud.mode_label = "Writing"
        refresh_lang()
        sync_overlay(info)

    def on_stroke(points, timestamps=None) -> None:
        """StrokeCollector → ONNX (lang-masked) → SendInput + Trie → GhostOverlay."""
        if focus_gating and not injector.text_focused:
            return
        refresh_lang()
        hit = classifier.recognize(points, lang=state["lang"], timestamps=timestamps)
        if hit is None:
            ranked = classifier.predict(
                points,
                top_k=1,
                allowed=charset_mask(classifier.charset, state["lang"]),
                timestamps=timestamps,
            )
            label, conf = ranked[0]
            tracker.set_last_recognition(label, conf, flash=False)
            return
        label, conf = hit
        tracker.set_last_recognition(label, conf, flash=True)
        if label in {"SPACE", "BACKSPACE", "ENTER", "TAB"}:
            if label == "SPACE":
                injector.space()
                ghost.clear()
            elif label == "BACKSPACE":
                injector.backspace()
                ghost.pop_char()
            elif label == "ENTER":
                injector.enter()
                ghost.clear()
            elif label == "TAB":
                text = ghost.commit_tab()
                if text:
                    injector.type_text(text)
                else:
                    injector.tab()
                ghost.clear()
        else:
            injector.inject_label(label)
            ch = label
            if state["lang"] == InputLang.EN and len(ch) == 1 and ch.isalpha():
                ch = ch.lower()
            ghost.append_char(ch)
        sync_overlay()

    def on_fist_tab() -> None:
        """Fist → commit GhostText suffix + trailing space (or raw TAB)."""
        if focus_gating and not injector.text_focused:
            return
        tracker.set_last_recognition("TAB", 1.0)
        text = ghost.commit_tab()
        if text:
            injector.type_text(text)
        else:
            injector.tab()
        ghost.clear()
        sync_overlay()

    def on_swipe_left() -> None:
        if focus_gating and not injector.text_focused:
            return
        tracker.set_last_recognition("BACKSPACE", 1.0)
        injector.backspace()
        ghost.pop_char()
        sync_overlay()

    def on_swipe_right() -> None:
        if focus_gating and not injector.text_focused:
            return
        tracker.set_last_recognition("SPACE", 1.0)
        injector.space()
        ghost.clear()
        sync_overlay()

    def on_lang_switch() -> None:
        injector.toggle_keyboard_layout()
        refresh_lang()
        tracker.set_last_recognition("LANG", 1.0)
        sync_overlay()

    def on_enter() -> None:
        if focus_gating and not injector.text_focused:
            return
        tracker.set_last_recognition("ENTER", 1.0)
        injector.enter()
        ghost.clear()
        sync_overlay()

    callbacks = DualHandCallbacks(
        on_mouse_move=injector.move_pointer_norm,
        on_left_down=injector.left_down,
        on_left_up=injector.left_up,
        on_right_down=injector.right_down,
        on_right_up=injector.right_up,
        on_scroll=injector.scroll,
        on_stroke=on_stroke,
        on_fist_tab=on_fist_tab,
        on_swipe_left=on_swipe_left,
        on_swipe_right=on_swipe_right,
        on_lang_switch=on_lang_switch,
        on_enter=on_enter,
    )
    tracker = DualHandTracker(
        camera_index=camera,
        mirror=mirror,
        callbacks=callbacks,
        show_preview=preview,
        writing_enabled=True,
    )
    tracker.set_os_lang(state["lang"].value)
    if not focus_gating:
        injector.set_text_focused(True)
        tracker.set_writing_enabled(True)
        tracker.mode_label = "Writing"
        tracker.hud.mode_label = "Writing"

    _orig_step = tracker.step

    def step_with_gate() -> bool:
        gate_from_focus()
        return _orig_step()

    tracker.step = step_with_gate  # type: ignore[method-assign]

    def set_preview(enabled: bool) -> None:
        enabled = bool(enabled)
        state["preview"] = enabled
        tracker.show_preview = enabled
        if not enabled:
            try:
                import cv2

                cv2.destroyWindow("AirTouch")
            except Exception:
                try:
                    import cv2

                    cv2.destroyAllWindows()
                except Exception:
                    pass

    return {
        "tries": {k.value: v.word_count for k, v in tries.items()},
        "ghost": ghost,
        "injector": injector,
        "overlay": overlay,
        "classifier": classifier,
        "tracker": tracker,
        "focus": focus,
        "lang": state,
        "focus_gating": focus_gating,
        "set_preview": set_preview,
        "state": state,
    }


def run_tray(camera: int = 0, preview: bool = False, mirror: bool = True) -> int:
    try:
        import pystray
        from PIL import Image, ImageDraw
    except ImportError as exc:
        print("Install tray deps: pip install pystray pillow", file=sys.stderr)
        raise SystemExit(1) from exc

    app = _build_app(camera, preview, mirror)
    tracker = app["tracker"]
    overlay = app["overlay"]
    set_preview = app["set_preview"]
    state = app["state"]
    stop = threading.Event()

    def worker() -> None:
        overlay.start()
        try:
            tracker.open()
            while not stop.is_set():
                if not tracker.step():
                    break
        finally:
            tracker.close()
            overlay.stop()

    thread = threading.Thread(target=worker, name="airtouch-loop", daemon=True)

    def make_icon() -> Image.Image:
        img = Image.new("RGBA", (64, 64), (30, 41, 59, 255))
        draw = ImageDraw.Draw(img)
        draw.ellipse((8, 8, 56, 56), outline=(59, 130, 246, 255), width=3)
        draw.line((20, 40, 32, 20, 44, 40), fill=(248, 250, 252, 255), width=3)
        return img

    def on_quit(icon: pystray.Icon, _item) -> None:
        stop.set()
        tracker._running = False
        icon.stop()

    def on_toggle_preview(icon: pystray.Icon, _item) -> None:
        set_preview(not state["preview"])
        icon.update_menu()

    def preview_label(_item) -> str:
        return "Hide Preview" if state["preview"] else "Show Preview"

    menu = pystray.Menu(
        pystray.MenuItem(preview_label, on_toggle_preview),
        pystray.MenuItem("Quit AirTouch", on_quit),
    )
    icon = pystray.Icon("AirTouch", make_icon(), "AirTouch", menu)

    print("AirTouch starting (tray daemon)")
    print(f"  dictionaries : {app['tries']}")
    print(f"  classes      : {len(app['classifier'].charset)}")
    print(f"  camera       : {camera}  mirror={mirror} preview={preview}")
    print(f"  layout       : {app['lang']['lang'].value}")
    print("  pipeline     : DualTracker → Gestures/Strokes → ONNX → Trie → GhostOverlay")
    thread.start()
    icon.run()
    stop.set()
    tracker._running = False
    thread.join(timeout=2.0)
    return 0


def run_headless(camera: int = 0, preview: bool = True, mirror: bool = True) -> int:
    from src.ui.stop_panel import StopPanel

    app = _build_app(camera, preview, mirror)
    overlay = app["overlay"]
    tracker = app["tracker"]
    stop = {"flag": False}
    panel: StopPanel | None = None

    def request_stop() -> None:
        stop["flag"] = True
        tracker._running = False

    def _sig(_signum, _frame) -> None:
        request_stop()

    signal.signal(signal.SIGINT, _sig)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _sig)  # type: ignore[attr-defined]

    print("AirTouch preview — use the red STOP button (Ctrl+C is unreliable)", flush=True)
    print(f"  dictionaries : {app['tries']}", flush=True)
    print(f"  layout       : {app['lang']['lang'].value}", flush=True)
    print("  LEFT=gestures+strokes | RIGHT=mouse | GhostOverlay=glass TAB hint", flush=True)
    overlay.start()
    panel = StopPanel(on_stop=request_stop)
    panel.start()
    try:
        tracker.open()
        while tracker._running and not stop["flag"]:
            if not tracker.step():
                break
    except KeyboardInterrupt:
        request_stop()
    finally:
        request_stop()
        tracker.close()
        overlay.stop()
        if panel is not None:
            panel.stop()
        print("AirTouch stopped.", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AirTouch dual-hand air-writing daemon")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--preview", action="store_true", help="Show OpenCV debug window")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--no-tray", action="store_true", help="Run without system tray")
    args = parser.parse_args(argv)
    mirror = not args.no_mirror
    if args.no_tray or args.preview:
        return run_headless(args.camera, preview=True, mirror=mirror)
    return run_tray(args.camera, preview=False, mirror=mirror)


if __name__ == "__main__":
    raise SystemExit(main())
