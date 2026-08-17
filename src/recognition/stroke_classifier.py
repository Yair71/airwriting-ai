"""Stroke kinematic preprocess + ONNX character inference.

Matches training contract: 64 arc-length points -> (1, 63, 6) features
[dx, dy, sin(theta), cos(theta), curvature_dtheta, normalized_velocity].
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import cv2
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
IMAGE_SIZE = 28
GLYPH_BOX = 20
STROKE_CANVAS_SIZE = 128
STROKE_CANVAS_MARGIN = 8


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
    """Match the trained sequence model: centroid origin and uniform max-abs fit."""
    pts = np.asarray(points, dtype=np.float64).copy()
    pts -= pts.mean(axis=0, keepdims=True)
    extent = float(np.max(np.abs(pts))) if pts.size else 0.0
    if extent > 1e-12:
        pts /= extent
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


def _resample_strokes(
    strokes: tuple[np.ndarray, ...] | list[np.ndarray],
    total_points: int = RESAMPLE_POINTS,
) -> tuple[np.ndarray, ...]:
    valid = [
        np.asarray(stroke, dtype=np.float64)
        for stroke in strokes
        if np.asarray(stroke).ndim == 2
        and np.asarray(stroke).shape[0] > 0
        and np.asarray(stroke).shape[1] == 2
    ]
    if not valid:
        return ()
    if len(valid) > total_points // 2:
        valid = valid[: total_points // 2]

    lengths = np.asarray(
        [
            float(np.linalg.norm(np.diff(stroke, axis=0), axis=1).sum())
            if len(stroke) > 1
            else 0.0
            for stroke in valid
        ],
        dtype=np.float64,
    )
    if float(lengths.sum()) <= 1e-12:
        allocations = np.full(len(valid), 2, dtype=np.int32)
    else:
        allocations = np.maximum(
            2,
            np.rint(lengths / lengths.sum() * total_points).astype(np.int32),
        )
    while int(allocations.sum()) > total_points:
        candidates = np.flatnonzero(allocations > 2)
        if candidates.size == 0:
            break
        allocations[int(candidates[np.argmax(allocations[candidates])])] -= 1
    while int(allocations.sum()) < total_points:
        allocations[int(np.argmax(lengths))] += 1

    return tuple(
        resample_arc_length(stroke, int(count))
        for stroke, count in zip(valid, allocations, strict=True)
    )


def preprocess_strokes_image(
    strokes: tuple[np.ndarray, ...] | list[np.ndarray],
) -> np.ndarray:
    """Render isolated 128px ink, normalize, then produce [1,1,28,28]."""
    sampled = _resample_strokes(strokes)
    if not sampled:
        return np.zeros((1, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    all_points = np.concatenate(sampled, axis=0)
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    width, height = maximum - minimum
    usable = STROKE_CANVAS_SIZE - 2 * STROKE_CANVAS_MARGIN
    scale = (usable - 1) / max(float(width), float(height), 1e-12)
    fitted_w = float(width) * scale
    fitted_h = float(height) * scale
    offset_x = (STROKE_CANVAS_SIZE - 1 - fitted_w) * 0.5
    offset_y = (STROKE_CANVAS_SIZE - 1 - fitted_h) * 0.5
    transformed: list[np.ndarray] = []
    for stroke in sampled:
        points = stroke.copy()
        points[:, 0] = (points[:, 0] - minimum[0]) * scale + offset_x
        points[:, 1] = (maximum[1] - points[:, 1]) * scale + offset_y
        transformed.append(points)

    canvas = np.zeros(
        (STROKE_CANVAS_SIZE, STROKE_CANVAS_SIZE),
        dtype=np.uint8,
    )
    thickness = 11
    for stroke in transformed:
        pixels = np.rint(stroke).astype(np.int32)
        if len(pixels) == 1:
            cv2.circle(canvas, tuple(pixels[0]), thickness // 2, 255, -1, cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [pixels], False, 255, thickness, cv2.LINE_AA)

    nonzero = cv2.findNonZero(canvas)
    if nonzero is None:
        return np.zeros((1, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    x, y, crop_width, crop_height = cv2.boundingRect(nonzero)
    crop = canvas[y : y + crop_height, x : x + crop_width]
    fit_scale = min(GLYPH_BOX / crop_width, GLYPH_BOX / crop_height)
    resized_width = max(1, int(round(crop_width * fit_scale)))
    resized_height = max(1, int(round(crop_height * fit_scale)))
    resized = cv2.resize(
        crop,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    paste_x = (IMAGE_SIZE - resized_width) // 2
    paste_y = (IMAGE_SIZE - resized_height) // 2
    image[
        paste_y : paste_y + resized_height,
        paste_x : paste_x + resized_width,
    ] = resized

    # Center of mass alignment, matching MNIST/EMNIST-style preprocessing.
    moments = cv2.moments(image, binaryImage=False)
    if moments["m00"] > 1e-9:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        shift_x = (IMAGE_SIZE - 1) * 0.5 - center_x
        shift_y = (IMAGE_SIZE - 1) * 0.5 - center_y
        matrix = np.asarray([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
        image = cv2.warpAffine(
            image,
            matrix,
            (IMAGE_SIZE, IMAGE_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return (image.astype(np.float32) / 255.0)[None, None, :, :]


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    confidence: float
    accepted: bool


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
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.input_shape = tuple(model_input.shape)
        self.expects_image = len(self.input_shape) == 4
        payload = json.loads(Path(map_path).read_text(encoding="utf-8"))
        self.charset: list[str] = [str(c) for c in payload["charset"]]
        self.min_confidence = float(min_confidence)

    def predict(
        self,
        raw_points: np.ndarray,
        top_k: int = 3,
        allowed: list[bool] | None = None,
        timestamps: np.ndarray | None = None,
        strokes: tuple[np.ndarray, ...] | None = None,
    ) -> list[tuple[str, float]]:
        if self.expects_image:
            image_strokes = strokes if strokes else (np.asarray(raw_points),)
            tensor = preprocess_strokes_image(image_strokes)
        else:
            tensor = preprocess_stroke(raw_points, timestamps=timestamps)
        output = self.session.run(None, {self.input_name: tensor})[0]
        logits = np.asarray(output, dtype=np.float64).reshape(-1)
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
        strokes: tuple[np.ndarray, ...] | None = None,
    ) -> str:
        return self.predict(
            raw_points,
            top_k=1,
            allowed=allowed,
            timestamps=timestamps,
            strokes=strokes,
        )[0][0]

    def classify(
        self,
        raw_points: np.ndarray,
        lang: object | None = None,
        timestamps: np.ndarray | None = None,
        strokes: tuple[np.ndarray, ...] | None = None,
    ) -> ClassificationResult:
        from src.platform.keyboard_layout import InputLang, charset_mask, layout_to_lang

        active = lang if lang is not None else layout_to_lang()
        if not isinstance(active, InputLang):
            active = layout_to_lang()
        if active == InputLang.OTHER:
            active = InputLang.EN
        mask = charset_mask(self.charset, active)
        label, confidence = self.predict(
            raw_points,
            top_k=1,
            allowed=mask,
            timestamps=timestamps,
            strokes=strokes,
        )[0]
        return ClassificationResult(
            label=label,
            confidence=confidence,
            accepted=confidence > self.min_confidence,
        )

    def recognize(
        self,
        raw_points: np.ndarray,
        lang: object | None = None,
        timestamps: np.ndarray | None = None,
        strokes: tuple[np.ndarray, ...] | None = None,
    ) -> tuple[str, float] | None:
        """Masked Top-1; returns None if confidence ≤ min_confidence."""
        result = self.classify(
            raw_points,
            lang=lang,
            timestamps=timestamps,
            strokes=strokes,
        )
        if not result.accepted:
            return None
        return result.label, result.confidence


class AsyncStrokeClassifier:
    """Single-worker ONNX queue that never blocks the camera thread."""

    def __init__(self, classifier: StrokeClassifier) -> None:
        self.classifier = classifier
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="airtouch-onnx",
        )
        self._closed = False

    def submit(
        self,
        raw_points: np.ndarray,
        *,
        lang: object | None,
        timestamps: np.ndarray | None,
        strokes: tuple[np.ndarray, ...] | None,
        on_complete: Callable[[ClassificationResult], None],
        on_error: Callable[[BaseException], None] | None = None,
    ) -> Future[ClassificationResult] | None:
        if self._closed:
            return None
        points_copy = np.asarray(raw_points, dtype=np.float64).copy()
        timestamps_copy = (
            np.asarray(timestamps, dtype=np.float64).copy()
            if timestamps is not None
            else None
        )
        strokes_copy = (
            tuple(np.asarray(stroke, dtype=np.float64).copy() for stroke in strokes)
            if strokes is not None
            else None
        )
        future = self._executor.submit(
            self.classifier.classify,
            points_copy,
            lang,
            timestamps_copy,
            strokes_copy,
        )

        def _done(done: Future[ClassificationResult]) -> None:
            try:
                on_complete(done.result())
            except BaseException as exc:
                if on_error is not None:
                    on_error(exc)

        future.add_done_callback(_done)
        return future

    def close(self, wait: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=False)
