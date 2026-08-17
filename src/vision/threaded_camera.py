"""Latest-frame-only OpenCV capture running on a daemon thread."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    sequence: int
    timestamp_ms: int
    image: np.ndarray


class ThreadedCamera:
    """Continuously overwrite one atomic frame slot; never enqueue frames."""

    def __init__(
        self,
        camera_index: int = 0,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 60,
    ) -> None:
        self.camera_index = int(camera_index)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._slot: CameraFrame | None = None
        self._sequence = 0
        self._last_read_sequence = -1

    @property
    def is_opened(self) -> bool:
        return self._running and self._cap is not None and self._cap.isOpened()

    def start(self) -> None:
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open camera {self.camera_index}")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="airtouch-camera",
            daemon=True,
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        cap = self._cap
        if cap is None:
            return
        while self._running:
            if not cap.grab():
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            self._sequence += 1
            # Tuple/object assignment is atomic under CPython's GIL. The
            # capture thread never mutates a frame after publishing it.
            self._slot = CameraFrame(
                sequence=self._sequence,
                timestamp_ms=time.monotonic_ns() // 1_000_000,
                image=frame,
            )

    def read_latest(self) -> CameraFrame | None:
        """Return a newly published latest frame, or None without blocking."""
        packet = self._slot
        if packet is None or packet.sequence == self._last_read_sequence:
            return None
        self._last_read_sequence = packet.sequence
        return packet

    def close(self) -> None:
        self._running = False
        cap = self._cap
        self._cap = None
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)
        if cap is not None:
            cap.release()
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.2)
        self._slot = None
