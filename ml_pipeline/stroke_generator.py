"""True unistroke centerline generator (cubic Bézier, not TTF outlines).

Data contract
-------------
Each glyph is one continuous pen path (centerline), 64 arc-length points.
Features float32 (63, 6) =
    [dx, dy, sin(theta), cos(theta), dtheta, norm_velocity]
NPZ: data/synthetic/stroke_dataset.npz
    features (N, 63, 6)  points (N, 64, 2)  labels (N,)  charset (C,)
N = 500_000  (~756 MB features + 256 MB points uncompressed)
CPU: ~0.05 ms / sample after templates are baked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ml_pipeline.font_sampler import REPO_ROOT, RESAMPLE_POINTS, resample_arc_length

FEATURE_STEPS = RESAMPLE_POINTS - 1
N_FEATURES = 6
DEFAULT_STROKE_NPZ = REPO_ROOT / "data" / "synthetic" / "stroke_dataset.npz"
DEFAULT_MAP = REPO_ROOT / "configs" / "unistroke_map.json"
TARGET_SAMPLES = 500_000

GESTURES = ("SPACE", "BACKSPACE", "ENTER", "TAB")


def _latin() -> tuple[str, ...]:
    return tuple(chr(c) for c in range(ord("A"), ord("Z") + 1))


def _digits() -> tuple[str, ...]:
    return tuple(chr(c) for c in range(ord("0"), ord("9") + 1))


def _cyrillic() -> tuple[str, ...]:
    return tuple(chr(c) for c in range(0x0410, 0x0430)) + ("Ё",)


def _hebrew() -> tuple[str, ...]:
    return tuple(chr(c) for c in range(0x05D0, 0x05EB))


def build_stroke_charset() -> tuple[str, ...]:
    return _digits() + _latin() + _cyrillic() + _hebrew() + GESTURES


def cubic_bezier(p0: np.ndarray, c1: np.ndarray, c2: np.ndarray, p1: np.ndarray, n: int = 16) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    mt = 1.0 - t
    return (
        (mt**3)[:, None] * p0
        + (3.0 * mt * mt * t)[:, None] * c1
        + (3.0 * mt * t * t)[:, None] * c2
        + (t**3)[:, None] * p1
    )


def waypoints_to_cubic_path(waypoints: np.ndarray, n_per_seg: int = 18) -> np.ndarray:
    """Catmull-Rom through waypoints, converted to cubic Bézier segments."""
    wp = np.asarray(waypoints, dtype=np.float64)
    if wp.ndim != 2 or wp.shape[1] != 2:
        raise ValueError(f"waypoints must be (M, 2), got {wp.shape}")
    if wp.shape[0] == 0:
        raise ValueError("empty waypoint list")
    if wp.shape[0] == 1:
        return np.repeat(wp, 8, axis=0)
    if wp.shape[0] == 2:
        c1 = wp[0] + (wp[1] - wp[0]) / 3.0
        c2 = wp[0] + 2.0 * (wp[1] - wp[0]) / 3.0
        return cubic_bezier(wp[0], c1, c2, wp[1], max(n_per_seg, 8))
    padded = np.vstack([wp[0], wp, wp[-1]])
    parts: list[np.ndarray] = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        curve = cubic_bezier(p1, c1, c2, p2, n_per_seg)
        parts.append(curve if not parts else curve[1:])
    return np.vstack(parts)


def ellipse(cx: float, cy: float, rx: float, ry: float, n: int = 24, start_deg: float = 90.0, sweep_deg: float = 360.0, ccw: bool = True) -> np.ndarray:
    a0 = np.deg2rad(start_deg)
    sign = 1.0 if ccw else -1.0
    ang = a0 + sign * np.deg2rad(np.linspace(0.0, sweep_deg, n, dtype=np.float64))
    return np.stack([cx + rx * np.cos(ang), cy + ry * np.sin(ang)], axis=1)


def W(*xy: float) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float64)
    if arr.size % 2:
        raise ValueError("waypoint list must be even (x,y pairs)")
    return arr.reshape(-1, 2)


def _glyph_waypoints() -> dict[str, np.ndarray]:
    """Hand-authored unistroke centerlines in a 0–100 box, y-up. Distinct topologies for lookalikes."""
    g: dict[str, np.ndarray] = {}

    # Digits — 0 is a narrow CCW oval from SW; O (latin) is a wide CW circle from north.
    g["0"] = ellipse(50, 50, 28, 44, n=28, start_deg=210, sweep_deg=360, ccw=True)
    g["1"] = W(38, 72, 50, 90, 50, 10, 38, 10, 62, 10)
    g["2"] = W(22, 72, 50, 92, 78, 72, 78, 58, 22, 12, 80, 12)
    g["3"] = W(24, 82, 72, 92, 78, 68, 48, 54, 78, 40, 72, 12, 24, 18)
    g["4"] = W(62, 10, 62, 90, 22, 38, 84, 38)
    g["5"] = W(76, 90, 28, 90, 26, 56, 70, 60, 78, 28, 40, 10, 22, 22)
    g["6"] = W(74, 84, 36, 70, 24, 40, 40, 12, 76, 22, 78, 48, 48, 58, 28, 44)
    g["7"] = W(20, 90, 82, 90, 40, 10)
    g["8"] = np.vstack(
        [
            ellipse(50, 70, 22, 20, n=16, start_deg=90, sweep_deg=360, ccw=False),
            ellipse(50, 28, 24, 22, n=16, start_deg=90, sweep_deg=360, ccw=True)[1:],
        ]
    )
    g["9"] = W(30, 28, 28, 70, 50, 92, 76, 74, 76, 40, 40, 12, 76, 18)

    # Latin. A crossbar is drawn right-to-left after the right leg.
    g["A"] = W(12, 8, 50, 94, 88, 8, 70, 8, 66, 40, 34, 40)
    g["B"] = W(22, 8, 22, 92, 62, 88, 70, 70, 58, 54, 22, 50, 64, 46, 74, 26, 62, 8, 22, 8)
    g["C"] = W(82, 78, 50, 94, 18, 70, 16, 30, 48, 8, 82, 22)
    g["D"] = W(22, 8, 22, 92, 58, 88, 80, 60, 80, 32, 58, 8, 22, 8)
    g["E"] = W(78, 90, 24, 90, 24, 50, 64, 50, 24, 50, 24, 10, 78, 10)
    g["F"] = W(24, 10, 24, 90, 78, 90, 24, 90, 24, 52, 66, 52)
    g["G"] = W(80, 76, 50, 94, 18, 68, 18, 30, 48, 8, 82, 24, 82, 48, 52, 48)
    g["H"] = W(20, 10, 20, 90, 20, 50, 80, 50, 80, 10, 80, 90)
    g["I"] = W(32, 90, 68, 90, 50, 90, 50, 10, 32, 10, 68, 10)
    g["J"] = W(28, 90, 72, 90, 58, 90, 58, 18, 40, 8, 22, 22)
    g["K"] = W(22, 90, 22, 10, 22, 48, 80, 90, 22, 48, 80, 10)
    g["L"] = W(24, 90, 24, 10, 80, 10)
    g["M"] = W(12, 10, 12, 90, 50, 40, 88, 90, 88, 10)
    g["N"] = W(22, 10, 22, 90, 78, 10, 78, 90)
    g["O"] = ellipse(50, 50, 36, 40, n=28, start_deg=90, sweep_deg=360, ccw=False)
    g["P"] = W(22, 10, 22, 90, 64, 86, 74, 68, 62, 50, 22, 48)
    g["Q"] = np.vstack([ellipse(50, 54, 34, 38, n=24, start_deg=90, sweep_deg=360, ccw=False), W(62, 28, 88, 6)])
    g["R"] = W(22, 10, 22, 90, 64, 86, 74, 68, 60, 50, 22, 48, 78, 10)
    g["S"] = W(76, 80, 48, 94, 22, 78, 28, 58, 72, 42, 78, 20, 48, 6, 22, 18)
    g["T"] = W(18, 90, 82, 90, 50, 90, 50, 10)
    g["U"] = W(20, 90, 20, 28, 36, 8, 64, 8, 80, 28, 80, 90)
    g["V"] = W(16, 90, 50, 8, 84, 90)
    g["W"] = W(10, 90, 28, 8, 50, 60, 72, 8, 90, 90)
    g["X"] = W(18, 90, 82, 10, 50, 50, 18, 10, 82, 90)
    g["Y"] = W(18, 90, 50, 48, 82, 90, 50, 48, 50, 8)
    g["Z"] = W(20, 90, 80, 90, 20, 10, 80, 10)

    # Cyrillic — different stroke order from Latin twins.
    g["А"] = W(50, 94, 14, 8, 36, 42, 64, 42, 86, 8)  # apex first, crossbar L→R
    g["Б"] = W(78, 90, 22, 90, 22, 10, 70, 14, 76, 40, 58, 52, 22, 48)
    g["В"] = W(22, 92, 22, 8, 66, 10, 76, 28, 62, 46, 22, 50, 66, 54, 74, 74, 60, 92, 22, 92)
    g["Г"] = W(78, 90, 22, 90, 22, 10)
    g["Д"] = W(22, 22, 50, 90, 78, 22, 22, 22, 16, 8, 84, 8, 78, 22)
    g["Е"] = W(24, 10, 24, 90, 78, 90, 24, 90, 24, 50, 66, 50, 24, 50, 24, 10, 78, 10)
    g["Ё"] = W(78, 90, 24, 90, 24, 50, 64, 50, 24, 50, 24, 10, 78, 10, 50, 10, 50, 96, 38, 100, 38, 92, 62, 92, 62, 100)
    g["Ж"] = W(18, 90, 50, 50, 18, 10, 50, 50, 82, 10, 50, 50, 82, 90)
    g["З"] = W(26, 84, 70, 94, 78, 70, 50, 56, 78, 40, 70, 12, 26, 16)
    g["И"] = W(24, 90, 24, 10, 76, 90, 76, 10)
    g["Й"] = W(24, 88, 24, 10, 76, 88, 76, 10, 76, 88, 50, 100, 24, 88)
    g["К"] = W(22, 10, 22, 90, 22, 50, 80, 90, 22, 50, 80, 10)
    g["Л"] = W(18, 12, 50, 90, 82, 12)
    g["М"] = W(14, 12, 20, 90, 50, 36, 80, 90, 86, 12)
    g["Н"] = W(22, 90, 22, 10, 22, 50, 78, 50, 78, 90, 78, 10)
    g["О"] = ellipse(50, 50, 34, 38, n=26, start_deg=0, sweep_deg=360, ccw=True)
    g["П"] = W(22, 10, 22, 90, 78, 90, 78, 10)
    g["Р"] = W(22, 8, 22, 92, 62, 90, 74, 70, 60, 50, 22, 48)
    g["С"] = W(78, 70, 40, 92, 16, 60, 18, 28, 48, 8, 80, 28)
    g["Т"] = W(16, 90, 84, 90, 50, 90, 50, 8)
    g["У"] = W(18, 90, 50, 36, 82, 90, 50, 36, 50, 8)
    g["Ф"] = np.vstack([W(50, 8, 50, 92), ellipse(50, 50, 30, 28, n=20, start_deg=90, sweep_deg=360, ccw=True)[1:]])
    g["Х"] = W(20, 88, 80, 12, 50, 50, 20, 12, 80, 88)
    g["Ц"] = W(22, 90, 22, 14, 78, 14, 78, 90, 78, 14, 92, 4)
    g["Ч"] = W(22, 90, 22, 52, 78, 52, 78, 90, 78, 10)
    g["Ш"] = W(18, 90, 18, 12, 50, 12, 50, 90, 50, 12, 82, 12, 82, 90)
    g["Щ"] = W(16, 90, 16, 14, 50, 14, 50, 90, 50, 14, 80, 14, 80, 90, 80, 14, 94, 4)
    g["Ъ"] = W(16, 90, 40, 90, 40, 12, 74, 16, 78, 42, 52, 50, 40, 46)
    g["Ы"] = W(24, 90, 24, 12, 58, 16, 62, 44, 28, 50, 24, 12, 82, 12, 82, 90)
    g["Ь"] = W(28, 90, 28, 12, 64, 16, 70, 42, 36, 52)
    g["Э"] = W(24, 78, 70, 94, 84, 50, 70, 8, 24, 22, 84, 50, 44, 50)
    g["Ю"] = np.vstack([W(18, 90, 18, 10, 18, 50, 42, 50), ellipse(68, 50, 24, 34, n=20, start_deg=180, sweep_deg=360, ccw=True)[1:]])
    g["Я"] = W(80, 10, 80, 90, 42, 86, 32, 68, 46, 50, 80, 50, 36, 10)

    # Hebrew — one continuous centerline each; finals have long descenders.
    g["א"] = W(18, 82, 50, 48, 82, 82, 50, 48, 50, 12)
    g["ב"] = W(78, 88, 22, 88, 22, 14, 82, 14)
    g["ג"] = W(22, 88, 78, 88, 40, 12)
    g["ד"] = W(22, 88, 78, 88, 78, 12)
    g["ה"] = W(22, 88, 78, 88, 78, 14, 78, 50, 36, 50, 36, 14)
    g["ו"] = W(50, 82, 50, 42)  # short
    g["ז"] = W(22, 88, 78, 88, 44, 12)
    g["ח"] = W(22, 88, 22, 14, 22, 88, 78, 88, 78, 14)
    g["ט"] = W(22, 16, 22, 88, 78, 88, 78, 16, 50, 16, 50, 52)
    g["י"] = W(48, 78, 56, 58)  # jot
    g["ך"] = W(70, 88, 70, 4)  # final kaf, long
    g["כ"] = W(74, 88, 28, 88, 28, 16, 74, 22)
    g["ל"] = W(22, 48, 22, 92, 80, 92, 80, 14)
    g["ם"] = W(24, 88, 76, 88, 76, 16, 24, 16, 24, 88)
    g["מ"] = W(24, 14, 24, 88, 76, 88, 76, 46, 40, 46)
    g["ן"] = W(50, 88, 50, 4)  # final nun, long
    g["נ"] = W(50, 88, 50, 22, 72, 14)
    g["ס"] = ellipse(50, 50, 30, 34, n=22, start_deg=270, sweep_deg=360, ccw=True)
    g["ע"] = W(18, 36, 50, 84, 82, 36, 50, 14)
    g["ף"] = W(28, 88, 72, 88, 72, 4)
    g["פ"] = W(28, 88, 74, 88, 74, 20, 40, 20, 40, 52)
    g["ץ"] = W(18, 84, 50, 50, 82, 84, 50, 6)
    g["צ"] = W(18, 84, 50, 50, 82, 84, 50, 22)
    g["ק"] = W(76, 92, 76, 14, 28, 14, 28, 52, 76, 52)
    g["ר"] = W(28, 88, 74, 88, 74, 14)
    g["ש"] = W(16, 14, 16, 88, 50, 46, 84, 88, 84, 14)
    g["ת"] = W(22, 88, 78, 88, 78, 14, 58, 32)

    # Gestures in the 5 cm box: direction is the identity.
    g["SPACE"] = W(8, 50, 92, 50)
    g["BACKSPACE"] = W(92, 50, 8, 50)
    g["ENTER"] = W(70, 88, 70, 22, 18, 22)
    g["TAB"] = W(12, 60, 48, 60, 48, 40, 88, 40)
    return g


def bake_templates(n_per_seg: int = 20) -> dict[str, np.ndarray]:
    baked: dict[str, np.ndarray] = {}
    for name, wp in _glyph_waypoints().items():
        path = waypoints_to_cubic_path(wp, n_per_seg=n_per_seg)
        if path.shape[0] < 8:
            raise RuntimeError(f"template {name!r} too short ({path.shape[0]} pts)")
        baked[name] = path
    charset = build_stroke_charset()
    missing = [c for c in charset if c not in baked]
    extra = [c for c in baked if c not in charset]
    if missing or extra:
        raise RuntimeError(f"template mismatch missing={missing} extra={extra}")
    return baked


def kinematic_noise(points: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (noisy_xy, speed_along_path) before arc-length resample.

    Speed is used for the 6th feature so velocity is not erased by resampling.
    """
    pts = np.asarray(points, dtype=np.float64).copy()
    pts -= pts.mean(axis=0, keepdims=True)
    extent = float(np.max(np.abs(pts)))
    if extent > 1e-12:
        pts /= extent

    ang = np.deg2rad(float(rng.uniform(-14.0, 14.0)))
    c, s = np.cos(ang), np.sin(ang)
    pts = pts @ np.array([[c, -s], [s, c]], dtype=np.float64).T
    shear = float(rng.uniform(-0.18, 0.18))
    pts[:, 0] = pts[:, 0] + shear * pts[:, 1]
    sx = float(rng.uniform(0.82, 1.18))
    sy = float(rng.uniform(0.82, 1.18))
    pts[:, 0] *= sx
    pts[:, 1] *= sy

    n = int(pts.shape[0])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total < 1e-12:
        speed = np.ones(n, dtype=np.float64)
        return pts, speed
    u = arc / total
    amp = float(rng.uniform(0.2, 0.65))
    phase = float(rng.uniform(0.0, 1.0))
    speed = np.clip(1.0 + amp * np.sin(2.0 * np.pi * (u + phase)), 0.28, 2.6)
    dt = np.diff(arc) / np.maximum(speed[:-1], 1e-8)
    time = np.concatenate([[0.0], np.cumsum(dt)])
    time = time / max(float(time[-1]), 1e-12)
    query = np.linspace(0.0, 1.0, n, dtype=np.float64)
    warped = np.stack([np.interp(query, time, pts[:, 0]), np.interp(query, time, pts[:, 1])], axis=1)
    speed_w = np.interp(query, time, speed)

    diag = float(np.linalg.norm(warped.max(0) - warped.min(0)))
    sigma = 0.018 * max(diag, 1e-8)
    corr = 0.9
    innov = sigma * np.sqrt(max(1.0 - corr * corr, 1e-8))
    noise = np.empty_like(warped)
    noise[0] = rng.normal(0.0, sigma, size=2)
    draws = rng.normal(0.0, innov, size=(n - 1, 2))
    for i in range(1, n):
        noise[i] = corr * noise[i - 1] + draws[i - 1]
    return warped + noise, speed_w


def _wrap_angle(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def stroke_features(points64: np.ndarray, speed64: np.ndarray) -> np.ndarray:
    """(64, 2) + (64,) speed -> (63, 6)."""
    pts = np.asarray(points64, dtype=np.float64)
    spd = np.asarray(speed64, dtype=np.float64)
    if pts.shape != (RESAMPLE_POINTS, 2) or spd.shape != (RESAMPLE_POINTS,):
        raise ValueError("expected 64 points and 64 speed samples")
    delta = np.diff(pts, axis=0)
    mag = np.linalg.norm(delta, axis=1)
    theta = np.arctan2(delta[:, 1], delta[:, 0])
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    zero = mag < 1e-12
    sin_t[zero] = 0.0
    cos_t[zero] = 1.0
    dtheta = np.zeros(FEATURE_STEPS, dtype=np.float64)
    dtheta[1:] = _wrap_angle(np.diff(theta))
    vel = 0.5 * (spd[:-1] + spd[1:])
    vmax = float(np.max(vel))
    if vmax > 1e-12:
        vel = vel / vmax
    else:
        vel = np.ones_like(vel)
    feats = np.stack([delta[:, 0], delta[:, 1], sin_t, cos_t, dtheta, vel], axis=1)
    return feats.astype(np.float32, copy=False)


def normalize_xy(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).copy()
    pts -= pts.mean(axis=0, keepdims=True)
    extent = float(np.max(np.abs(pts)))
    if extent < 1e-12:
        return pts
    return pts / extent


def render_sample(template: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    noisy, speed = kinematic_noise(template, rng)
    if noisy.shape[0] < 3:
        raise RuntimeError("kinematic noise produced a degenerate stroke")
    sm = noisy.copy()
    sm[1:-1] = (noisy[:-2] + noisy[1:-1] + noisy[2:]) / 3.0
    seg = np.linalg.norm(np.diff(sm, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total < 1e-12:
        pts64 = np.repeat(sm[:1], RESAMPLE_POINTS, axis=0)
        spd64 = np.ones(RESAMPLE_POINTS, dtype=np.float64)
    else:
        pts64 = resample_arc_length(sm, RESAMPLE_POINTS)
        targets = np.linspace(0.0, total, RESAMPLE_POINTS, dtype=np.float64)
        spd64 = np.interp(targets, arc, speed if speed.shape[0] == sm.shape[0] else np.interp(np.linspace(0, 1, sm.shape[0]), np.linspace(0, 1, speed.shape[0]), speed))
    pts64 = normalize_xy(pts64)
    feats = stroke_features(pts64, spd64)
    if not np.isfinite(feats).all():
        raise RuntimeError("non-finite features")
    return pts64.astype(np.float32), feats


def generate_stroke_dataset(
    out_path: Path = DEFAULT_STROKE_NPZ,
    n_samples: int = TARGET_SAMPLES,
    seed: int = 42,
    map_path: Path = DEFAULT_MAP,
) -> dict:
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    charset = build_stroke_charset()
    templates = bake_templates()
    n_classes = len(charset)
    per = n_samples // n_classes
    remainder = n_samples - per * n_classes
    rng = np.random.default_rng(seed)

    features = np.empty((n_samples, FEATURE_STEPS, N_FEATURES), dtype=np.float32)
    points = np.empty((n_samples, RESAMPLE_POINTS, 2), dtype=np.float32)
    labels = np.empty((n_samples,), dtype=np.int32)

    cursor = 0
    for class_id, name in enumerate(tqdm(charset, desc="glyphs")):
        tmpl = templates[name]
        count = per + (1 if class_id < remainder else 0)
        for _ in range(count):
            pts, feats = render_sample(tmpl, rng)
            points[cursor] = pts
            features[cursor] = feats
            labels[cursor] = class_id
            cursor += 1
    if cursor != n_samples:
        raise RuntimeError(f"wrote {cursor} samples, expected {n_samples}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, features=features, points=points, labels=labels, charset=np.asarray(list(charset)))
    write_stroke_map(map_path, charset)
    return {
        "n_samples": n_samples,
        "n_classes": n_classes,
        "out_path": out_path,
        "map_path": map_path,
        "bytes": out_path.stat().st_size,
    }


def write_stroke_map(path: Path, charset: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_classes": len(charset),
        "resample_points": RESAMPLE_POINTS,
        "sequence_length": FEATURE_STEPS,
        "n_features": N_FEATURES,
        "feature_order": ["dx", "dy", "sin_theta", "cos_theta", "dtheta", "norm_velocity"],
        "charset": list(charset),
        "char_to_id": {ch: i for i, ch in enumerate(charset)},
        "scripts": {
            "digits": list(_digits()),
            "latin": list(_latin()),
            "cyrillic": list(_cyrillic()),
            "hebrew": list(_hebrew()),
            "gestures": list(GESTURES),
        },
        "coordinate_frame": "centerline_y_up_centroid_normalized",
        "generator": "stroke_generator.cubic_catmull_rom",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_verification(npz_path: Path, plot_path: Path, n_show: int = 8, seed: int = 0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blob = np.load(npz_path, allow_pickle=False)
    points = blob["points"]
    labels = blob["labels"]
    charset = [str(x) for x in blob["charset"]]
    rng = np.random.default_rng(seed)
    unique = np.unique(labels)
    chosen = rng.choice(unique, size=min(n_show, unique.size), replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    for ax, lab in zip(axes, chosen):
        idx = rng.choice(np.flatnonzero(labels == lab))
        pts = points[int(idx)]
        ax.plot(pts[:, 0], pts[:, 1], "-k", lw=1.4)
        ax.scatter(pts[0, 0], pts[0, 1], c="g", s=22, zorder=3)
        ax.scatter(pts[-1, 0], pts[-1, 1], c="r", s=22, zorder=3)
        ax.set_title(charset[int(lab)], fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Centerline unistroke verification (64-pt)")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate 500k centerline unistroke samples.")
    parser.add_argument("--out", type=Path, default=DEFAULT_STROKE_NPZ)
    parser.add_argument("--n", type=int, default=TARGET_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", type=Path, default=REPO_ROOT / "data" / "synthetic" / "stroke_verification.png")
    args = parser.parse_args(argv)
    stats = generate_stroke_dataset(args.out, args.n, args.seed)
    print(f"samples   : {stats['n_samples']}")
    print(f"classes   : {stats['n_classes']}")
    print(f"npz       : {stats['out_path']}  ({int(stats['bytes'])/1e6:.1f} MB)")
    plot_verification(Path(stats["out_path"]), args.plot)
    print(f"plot      : {args.plot}")
    print("tensor    : features (N, 63, 6)  points (N, 64, 2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
