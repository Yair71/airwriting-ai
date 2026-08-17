"""Stroke kinematic preprocess + ONNX character inference.

Matches training contract: 64 arc-length points -> (1, 63, 6) features
[dx, dy, sin(theta), cos(theta), curvature_dtheta, normalized_velocity].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.paths import resource_root

REPO_ROOT = resource_root()
DEFAULT_ONNX = REPO_ROOT / "data" / "checkpoints" / "accurate_model.onnx"
DEFAULT_MAP = REPO_ROOT / "configs" / "unistroke_map.json"
RESAMPLE_POINTS = 64
FEATURE_STEPS = 63
N_FEATURES = 6
MIN_CONFIDENCE = 0.60


def resample_arc_length(points: np.ndarray, n: int = RESAMPLE_POINTS) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return np.zeros((n, 2), dtype=np.float64)
    if pts.shape[0] == 1:
        return np.repeat(pts, n, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total < 1e-12:
        return np.repeat(pts[:1], n, axis=0)
    targets = np.linspace(0.0, total, n, dtype=np.float64)
    x = np.interp(targets, arc, pts[:, 0])
    y = np.interp(targets, arc, pts[:, 1])
    return np.stack([x, y], axis=1)


def resample_arc_length_with_time(
    points: np.ndarray, timestamps: np.ndarray, n: int = RESAMPLE_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.float64)
    if pts.shape[0] == 0:
        return np.zeros((n, 2), dtype=np.float64), np.zeros(n, dtype=np.float64)
    if pts.shape[0] == 1:
        return np.repeat(pts, n, axis=0), np.repeat(ts[:1], n)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total < 1e-12:
        return np.repeat(pts[:1], n, axis=0), np.linspace(float(ts[0]), float(ts[-1]), n)
    targets = np.linspace(0.0, total, n, dtype=np.float64)
    x = np.interp(targets, arc, pts[:, 0])
    y = np.interp(targets, arc, pts[:, 1])
    t = np.interp(targets, arc, ts)
    return np.stack([x, y], axis=1), t


def _wrap_angle(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def center_and_unit_bbox(points: np.ndarray) -> np.ndarray:
    """Center to (0, 0) and scale uniformly by max(width, height) — no independent stretch."""
    pts = np.asarray(points, dtype=np.float64).copy()
    pts -= pts.mean(axis=0, keepdims=True)
    width = float(pts[:, 0].max() - pts[:, 0].min()) if pts.shape[0] else 0.0
    height = float(pts[:, 1].max() - pts[:, 1].min()) if pts.shape[0] else 0.0
    scale = max(width, height, 1e-12)
    pts /= scale
    return pts


def points_to_features6(points64: np.ndarray, speed64: np.ndarray | None = None) -> np.ndarray:
    """(64, 2) [+ optional speed] -> (63, 6)."""
    pts = np.asarray(points64, dtype=np.float64)
    if pts.shape != (RESAMPLE_POINTS, 2):
        raise ValueError(f"expected ({RESAMPLE_POINTS}, 2), got {pts.shape}")
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

    if speed64 is not None:
        spd = np.asarray(speed64, dtype=np.float64)
        if spd.shape != (RESAMPLE_POINTS,):
            raise ValueError("speed64 must have length 64")
        vel = 0.5 * (spd[:-1] + spd[1:])
    else:
        # Uniform resample → segment length as velocity proxy
        vel = mag.copy()
    vmax = float(np.max(vel))
    vel = vel / vmax if vmax > 1e-12 else np.ones_like(vel)

    feats = np.stack([delta[:, 0], delta[:, 1], sin_t, cos_t, dtheta, vel], axis=1)
    return feats.astype(np.float32, copy=False)


def preprocess_stroke(
    raw_points: np.ndarray,
    timestamps: np.ndarray | None = None,
) -> np.ndarray:
    """Raw (N,2) [+ optional times] -> (1, 63, 6) float32."""
    pts = np.asarray(raw_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected (N, 2) points, got {pts.shape}")
    if pts.shape[0] < 2:
        feats = np.zeros((1, FEATURE_STEPS, N_FEATURES), dtype=np.float32)
        feats[0, :, 3] = 1.0
        feats[0, :, 5] = 1.0
        return feats

    # Light 3-point smoothing
    if pts.shape[0] >= 3:
        sm = pts.copy()
        sm[1:-1] = (pts[:-2] + pts[1:-1] + pts[2:]) / 3.0
        pts = sm

    speed64: np.ndarray | None = None
    if timestamps is not None and len(timestamps) == pts.shape[0]:
        resampled, t64 = resample_arc_length_with_time(pts, timestamps, RESAMPLE_POINTS)
        dt = np.diff(t64)
        dt = np.maximum(dt, 1e-4)
        # Point speeds from neighboring segments; pad ends
        seg = np.linalg.norm(np.diff(resampled, axis=0), axis=1)
        mid = seg / dt
        speed64 = np.empty(RESAMPLE_POINTS, dtype=np.float64)
        speed64[0] = mid[0]
        speed64[-1] = mid[-1]
        speed64[1:-1] = 0.5 * (mid[:-1] + mid[1:])
    else:
        resampled = resample_arc_length(pts, RESAMPLE_POINTS)

    normalized = center_and_unit_bbox(resampled)
    feats = points_to_features6(normalized, speed64)
    return feats[None, ...]


class StrokeClassifier:
    def __init__(
        self,
        onnx_path: Path = DEFAULT_ONNX,
        map_path: Path = DEFAULT_MAP,
        providers: list[str] | None = None,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> None:
        onnx_path = Path(onnx_path)
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        payload = json.loads(Path(map_path).read_text(encoding="utf-8"))
        self.charset: list[str] = [str(c) for c in payload["charset"]]
        self.min_confidence = float(min_confidence)

    def predict(
        self,
        raw_points: np.ndarray,
        top_k: int = 3,
        allowed: list[bool] | None = None,
        timestamps: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        feats = preprocess_stroke(raw_points, timestamps=timestamps)
        logits = self.session.run(None, {self.input_name: feats})[0][0]
        logits = logits.astype(np.float64)
        if allowed is not None:
            if len(allowed) != logits.size:
                raise ValueError("allowed mask length must match n_classes")
            for i, ok in enumerate(allowed):
                if not ok:
                    logits[i] = -1e9
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= max(probs.sum(), 1e-12)
        k = min(top_k, probs.size)
        idx = np.argpartition(-probs, kth=k - 1)[:k]
        idx = idx[np.argsort(-probs[idx])]
        return [(self.charset[int(i)], float(probs[int(i)])) for i in idx]

    def predict_label(
        self,
        raw_points: np.ndarray,
        allowed: list[bool] | None = None,
        timestamps: np.ndarray | None = None,
    ) -> str:
        return self.predict(raw_points, top_k=1, allowed=allowed, timestamps=timestamps)[0][0]

    def recognize(
        self,
        raw_points: np.ndarray,
        lang: object | None = None,
        timestamps: np.ndarray | None = None,
    ) -> tuple[str, float] | None:
        """Masked Top-1; returns None if confidence ≤ min_confidence."""
        from src.platform.keyboard_layout import InputLang, charset_mask, layout_to_lang

        active = lang if lang is not None else layout_to_lang()
        if not isinstance(active, InputLang):
            active = layout_to_lang()
        if active == InputLang.OTHER:
            active = InputLang.EN
        mask = charset_mask(self.charset, active)
        label, conf = self.predict(raw_points, top_k=1, allowed=mask, timestamps=timestamps)[0]
        if conf <= self.min_confidence:
            return None
        return label, conf
