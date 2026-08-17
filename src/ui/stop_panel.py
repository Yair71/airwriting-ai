"""Always-on-top STOP control panel (Ctrl+C is unreliable under OpenCV)."""

from __future__ import annotations

import threading
import tkinter as tk
from typing import Callable


class StopPanel:
    """Small red STOP button window that reliably ends the AirTouch loop."""

    def __init__(self, on_stop: Callable[[], None]) -> None:
        self._on_stop = on_stop
        self._thread: threading.Thread | None = None
        self._root: tk.Tk | None = None
        self._ready = threading.Event()
        self._stopped = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="airtouch-stop", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=3.0)

    def stop(self) -> None:
        root = self._root
        if root is not None:
            try:
                root.after(0, root.destroy)
            except Exception:
                pass
        self._ready.clear()

    def _fire(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            self._on_stop()
        finally:
            root = self._root
            if root is not None:
                try:
                    root.after(100, root.destroy)
                except Exception:
                    pass

    def _run(self) -> None:
        root = tk.Tk()
        self._root = root
        root.title("AirTouch")
        root.attributes("-topmost", True)
        root.resizable(False, False)
        root.configure(bg="#0F172A")
        # Keep a real window (not click-through) so the user can always click STOP.
        frame = tk.Frame(root, bg="#0F172A", padx=16, pady=12)
        frame.pack(fill="both", expand=True)
        tk.Label(
            frame,
            text="AirTouch is running",
            fg="#E2E8F0",
            bg="#0F172A",
            font=("Segoe UI", 11),
        ).pack(pady=(0, 8))
        btn = tk.Button(
            frame,
            text="STOP",
            command=self._fire,
            bg="#DC2626",
            fg="#FFFFFF",
            activebackground="#B91C1C",
            activeforeground="#FFFFFF",
            font=("Segoe UI", 16, "bold"),
            width=12,
            height=2,
            relief="flat",
            cursor="hand2",
        )
        btn.pack()
        tk.Label(
            frame,
            text="Click STOP  ·  or press Q in preview",
            fg="#94A3B8",
            bg="#0F172A",
            font=("Segoe UI", 9),
        ).pack(pady=(8, 0))
        root.protocol("WM_DELETE_WINDOW", self._fire)
        root.update_idletasks()
        # Park near top-right of primary screen
        w, h = root.winfo_reqwidth(), root.winfo_reqheight()
        sw = root.winfo_screenwidth()
        root.geometry(f"+{max(sw - w - 24, 24)}+24")
        self._ready.set()
        root.mainloop()
        self._root = None
