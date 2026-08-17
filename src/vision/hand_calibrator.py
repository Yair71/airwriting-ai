"""Interactive open-palm hand calibration → configs/hand_profile.json.

Collects ~90 frames over a 3s countdown and derives pinch / fist / 1€ tunables.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.paths import repo_root, resource_root

WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

CALIB_SECONDS = 3.0
TARGET_FRAMES = 90
ACTIVE_MARGIN_LO = 0.15
ACTIVE_MARGIN_HI = 0.85


@dataclass
class HandProfile:
    palm_base_scale: float
    index_length: float
    thumb_reach: float
    pinch_threshold_lmb: float
    pinch_threshold_rmb: float
    fist_closed_threshold: float
    natural_jitter_std: float
    active_margin_lo: float = ACTIVE_MARGIN_LO
    active_margin_hi: float = ACTIVE_MARGIN_HI
    one_euro_min_cutoff: float = 0.04
    one_euro_beta: float = 0.7
    calibrated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HandProfile:
        base = default_profile()
        merged = {**asdict(base), **data}
        return cls(**{k: merged[k] for k in asdict(base)})


def default_profile() -> HandProfile:
    palm = 0.15
    return HandProfile(
        palm_base_scale=palm,
        index_length=0.08,
        thumb_reach=0.12,
        pinch_threshold_lmb=0.30 * palm,
        pinch_threshold_rmb=0.30 * palm,
        fist_closed_threshold=0.35 * palm,
        natural_jitter_std=0.0015,
        active_margin_lo=ACTIVE_MARGIN_LO,
        active_margin_hi=ACTIVE_MARGIN_HI,
        one_euro_min_cutoff=0.04,
        one_euro_beta=0.7,
        calibrated_at=0.0,
    )


def hand_profile_path() -> Path:
    """Writable profile next to the app / repo (not inside PyInstaller extract)."""
    return repo_root() / "configs" / "hand_profile.json"


def _bundled_profile_path() -> Path:
    return resource_root() / "configs" / "hand_profile.json"


def profile_exists() -> bool:
    return hand_profile_path().is_file() or _bundled_profile_path().is_file()


def load_profile() -> HandProfile | None:
    for path in (hand_profile_path(), _bundled_profile_path()):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            return HandProfile.from_dict(data)
        except (OSError, json.JSONDecodeError, TypeError, KeyError):
            continue
    return None


def save_profile(profile: HandProfile) -> Path:
    path = hand_profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    return path


def min_cutoff_from_jitter(std: float) -> float:
    """Map static open-palm noise → OneEuro min_cutoff (more noise → more smoothing)."""
    s = max(float(std), 1e-6)
    # Typical webcam landmark std ~0.0008–0.004 → cutoff ~0.02–0.12
    return float(np.clip(0.00012 / s, 0.015, 0.20))


def compute_profile_from_landmarks(samples: list[np.ndarray]) -> HandProfile:
    if len(samples) < 10:
        raise ValueError(f"Need ≥10 palm frames, got {len(samples)}")

    palms: list[float] = []
    indexes: list[float] = []
    thumbs: list[float] = []
    tip_xy: list[tuple[float, float]] = []

    for lm in samples:
        palms.append(float(np.linalg.norm(lm[MIDDLE_MCP, :3] - lm[WRIST, :3])))
        indexes.append(float(np.linalg.norm(lm[INDEX_TIP, :3] - lm[INDEX_MCP, :3])))
        thumbs.append(float(np.linalg.norm(lm[THUMB_TIP, :3] - lm[WRIST, :3])))
        tip_xy.append((float(lm[INDEX_TIP, 0]), float(lm[INDEX_TIP, 1])))

    palm = float(np.mean(palms))
    palm = max(palm, 1e-4)
    tip_arr = np.asarray(tip_xy, dtype=np.float64)
    jitter = float(np.mean(np.std(tip_arr, axis=0)))

    return HandProfile(
        palm_base_scale=palm,
        index_length=float(np.mean(indexes)),
        thumb_reach=float(np.mean(thumbs)),
        pinch_threshold_lmb=0.30 * palm,
        pinch_threshold_rmb=0.30 * palm,
        fist_closed_threshold=0.35 * palm,
        natural_jitter_std=jitter,
        active_margin_lo=ACTIVE_MARGIN_LO,
        active_margin_hi=ACTIVE_MARGIN_HI,
        one_euro_min_cutoff=min_cutoff_from_jitter(jitter),
        one_euro_beta=0.7,
        calibrated_at=time.time(),
    )


class CalibPhase(str, Enum):
    IDLE = "IDLE"
    COUNTDOWN = "COUNTDOWN"
    COLLECTING = "COLLECTING"
    DONE = "DONE"


@dataclass
class HandCalibrator:
    """3-second open-palm calibration with preview overlay."""

    phase: CalibPhase = CalibPhase.IDLE
    profile: HandProfile | None = None
    status_line: str = "Uncalibrated"
    _t0: float = 0.0
    _samples: list[np.ndarray] = field(default_factory=list)
    _last_error: str = ""

    def needs_start(self) -> bool:
        return load_profile() is None

    def start(self) -> None:
        self.phase = CalibPhase.COUNTDOWN
        self._t0 = time.perf_counter()
        self._samples.clear()
        self._last_error = ""
        self.status_line = "Calibration: hold open palm"
        self.profile = None

    @property
    def active(self) -> bool:
        return self.phase in {CalibPhase.COUNTDOWN, CalibPhase.COLLECTING}

    def update(self, landmarks: np.ndarray | None) -> HandProfile | None:
        """Advance calibration. Returns a finished HandProfile once when DONE."""
        if not self.active:
            return None

        now = time.perf_counter()
        elapsed = now - self._t0

        if self.phase == CalibPhase.COUNTDOWN:
            remaining = CALIB_SECONDS - elapsed
            if remaining > 0:
                self.status_line = f"Calibration: {int(np.ceil(remaining))}…"
                return None
            self.phase = CalibPhase.COLLECTING
            self._t0 = now
            self._samples.clear()
            self.status_line = "Calibration: collecting…"

        if self.phase == CalibPhase.COLLECTING:
            if landmarks is not None and landmarks.shape[0] >= 21:
                self._samples.append(landmarks.copy())
            n = len(self._samples)
            self.status_line = f"Calibration: {n}/{TARGET_FRAMES} frames"
            # Finish when we have enough frames or ~3s of collection elapsed
            if n >= TARGET_FRAMES or (now - self._t0) >= CALIB_SECONDS:
                try:
                    self.profile = compute_profile_from_landmarks(self._samples)
                    save_profile(self.profile)
                    self.phase = CalibPhase.DONE
                    self.status_line = "Calibrated"
                    return self.profile
                except ValueError as exc:
                    self._last_error = str(exc)
                    self.status_line = f"Calibration failed — press C ({exc})"
                    self.phase = CalibPhase.IDLE
                    return None
        return None

    def draw_overlay(self, frame: np.ndarray) -> None:
        if not self.active:
            return
        h, w = frame.shape[:2]
        cx, cy = w // 2, int(h * 0.55)
        guide_r = int(min(w, h) * 0.18)

        # Palm target guide
        color = (80, 200, 255) if self.phase != CalibPhase.DONE else (80, 220, 120)
        cv2.circle(frame, (cx, cy), guide_r, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 8, color, -1, cv2.LINE_AA)
        # Simple open-palm silhouette hints (five rays)
        for ang in (-50, -25, 0, 25, 50):
            rad = np.deg2rad(ang - 90)
            x2 = int(cx + guide_r * 0.85 * np.cos(rad))
            y2 = int(cy + guide_r * 0.85 * np.sin(rad))
            cv2.line(frame, (cx, cy - 10), (x2, y2), color, 2, cv2.LINE_AA)

        if self.phase == CalibPhase.COUNTDOWN:
            remaining = max(CALIB_SECONDS - (time.perf_counter() - self._t0), 0.0)
            digit = str(max(int(np.ceil(remaining)), 1))
            scale = 4.0
            thickness = 6
            (tw, th), _ = cv2.getTextSize(digit, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            cv2.putText(
                frame,
                digit,
                ((w - tw) // 2, (h + th) // 2 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (0, 0, 0),
                thickness + 4,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                digit,
                ((w - tw) // 2, (h + th) // 2 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (0, 220, 255),
                thickness,
                cv2.LINE_AA,
            )
            hint = "Hold OPEN PALM inside the guide"
            cv2.putText(frame, hint, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, hint, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        elif self.phase == CalibPhase.COLLECTING:
            n = len(self._samples)
            bar_w = int(w * 0.6)
            x0 = (w - bar_w) // 2
            y0 = h - 48
            frac = min(n / float(TARGET_FRAMES), 1.0)
            cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + 18), (40, 40, 40), -1)
            cv2.rectangle(frame, (x0, y0), (x0 + int(bar_w * frac), y0 + 18), (80, 200, 255), -1)
            msg = f"Collecting open palm… {n}/{TARGET_FRAMES}"
            cv2.putText(frame, msg, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, msg, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
