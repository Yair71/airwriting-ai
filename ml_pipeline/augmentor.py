"""Kinematic distortions applied to dense glyph polylines before resampling.

All transforms are shape-changing. Uniform translation/scale are omitted because
the preprocessor recenters and fits to [-1, 1].

CPU: ~0.02–0.08 ms per polyline (~200 vertices) on a desktop CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml_pipeline.font_sampler import dedupe_consecutive


@dataclass(frozen=True, slots=True)
class AugmentConfig:
    shear_range: float = 0.28
    rotation_deg: float = 18.0
    aniso_scale: float = 0.22
    perspective_k: float = 0.12
    tremor_sigma: float = 0.022
    tremor_corr: float = 0.88
    velocity_amp: float = 0.55
    incomplete_prob: float = 0.12
    incomplete_frac: tuple[float, float] = (0.04, 0.12)


class Augmentor:
    def __init__(self, config: AugmentConfig | None = None, rng: np.random.Generator | None = None) -> None:
        self.config = config or AugmentConfig()
        self.rng = rng or np.random.default_rng()

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return self.augment(points)

    def augment(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[0] < 3:
            return pts.copy()
        cfg = self.config
        rng = self.rng

        if rng.random() < cfg.incomplete_prob:
            frac = float(rng.uniform(*cfg.incomplete_frac))
            mode = int(rng.integers(0, 3))
            pts = incomplete_loop(pts, frac, mode)

        # Geometric warps are only valid in O(1) coordinates. Perspective
        # (x + k*x*y) in raw font units (~1e4) explodes X and collapses Y
        # after the later [-1, 1] normalize.
        pts = _unit_fit(_center(pts))
        shear = float(rng.uniform(-cfg.shear_range, cfg.shear_range))
        pts = shear_x(pts, shear)

        angle = np.deg2rad(float(rng.uniform(-cfg.rotation_deg, cfg.rotation_deg)))
        pts = rotate(pts, angle)

        sx = float(rng.uniform(1.0 - cfg.aniso_scale, 1.0 + cfg.aniso_scale))
        sy = float(rng.uniform(1.0 - cfg.aniso_scale, 1.0 + cfg.aniso_scale))
        pts = anisotropic_scale(pts, sx, sy)

        k = float(rng.uniform(-cfg.perspective_k, cfg.perspective_k))
        pts = perspective_skew(pts, k)

        amp = float(rng.uniform(0.15, cfg.velocity_amp))
        phase = float(rng.uniform(0.0, 1.0))
        pts = velocity_warp(pts, amp, phase)

        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        sigma = cfg.tremor_sigma * max(diag, 1e-8)
        pts = hand_tremor(pts, sigma, cfg.tremor_corr, rng)
        return dedupe_consecutive(pts)


def _center(points: np.ndarray) -> np.ndarray:
    return points - points.mean(axis=0, keepdims=True)


def _unit_fit(points: np.ndarray) -> np.ndarray:
    extent = float(np.max(np.abs(points)))
    if extent < 1e-12:
        return points
    return points / extent


def shear_x(points: np.ndarray, shear: float) -> np.ndarray:
    out = points.copy()
    out[:, 0] = points[:, 0] + shear * points[:, 1]
    return out


def rotate(points: np.ndarray, angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return points @ rot.T


def anisotropic_scale(points: np.ndarray, sx: float, sy: float) -> np.ndarray:
    out = points.copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out


def perspective_skew(points: np.ndarray, k: float) -> np.ndarray:
    """Mild bilinear warp: x' = x + k * x * y, after centering."""
    out = points.copy()
    out[:, 0] = points[:, 0] + k * points[:, 0] * points[:, 1]
    return out


def velocity_warp(points: np.ndarray, amp: float, phase: float) -> np.ndarray:
    """Re-sample uniformly in time under v(s) = 1 + amp * sin(2π (u + phase)).

    Slow regions receive more samples, so the later 3-point moving average
    smooths them more — a physically plausible speed-dependent tremor filter.
    """
    n = int(points.shape[0])
    if n < 4:
        return points
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total < 1e-12:
        return points
    u = arc / total
    speed = 1.0 + amp * np.sin(2.0 * np.pi * (u + phase))
    speed = np.clip(speed, 0.25, 2.75)
    dt = np.diff(arc) / np.maximum(speed[:-1], 1e-8)
    time = np.concatenate([[0.0], np.cumsum(dt)])
    t_end = float(time[-1])
    if t_end < 1e-12:
        return points
    time /= t_end
    query = np.linspace(0.0, 1.0, n, dtype=np.float64)
    x = np.interp(query, time, points[:, 0])
    y = np.interp(query, time, points[:, 1])
    return np.stack([x, y], axis=1)


def hand_tremor(points: np.ndarray, sigma: float, corr: float, rng: np.random.Generator) -> np.ndarray:
    """Correlated 2D Ornstein–Uhlenbeck jitter (human physiological tremor)."""
    n = int(points.shape[0])
    corr = float(np.clip(corr, 0.0, 0.999))
    innov = sigma * np.sqrt(max(1.0 - corr * corr, 1e-8))
    noise = np.empty((n, 2), dtype=np.float64)
    noise[0] = rng.normal(0.0, sigma, size=2)
    draws = rng.normal(0.0, innov, size=(n - 1, 2))
    for i in range(1, n):
        noise[i] = corr * noise[i - 1] + draws[i - 1]
    return points + noise


def incomplete_loop(points: np.ndarray, frac: float, mode: int) -> np.ndarray:
    """Drop a prefix, suffix, or interior window of the path (unfinished loops)."""
    n = int(points.shape[0])
    if n < 8:
        return points
    frac = float(np.clip(frac, 0.02, 0.45))
    drop = max(int(round(frac * n)), 1)
    drop = min(drop, n - 4)
    if mode == 0:
        return points[: n - drop]
    if mode == 1:
        return points[drop:]
    start = int(np.clip((n - drop) // 3, 1, n - drop - 1))
    return np.vstack([points[:start], points[start + drop :]])
