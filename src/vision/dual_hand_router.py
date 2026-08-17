"""Fail-closed routing from MediaPipe detections to physical hand roles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class HandTrackId(str, Enum):
    LEFT_HAND_WRITER = "LEFT_HAND_WRITER"
    RIGHT_HAND_MOUSE = "RIGHT_HAND_MOUSE"


@dataclass(frozen=True)
class RoutedHandData:
    track_id: HandTrackId
    landmarks: np.ndarray
    centroid: np.ndarray
    wrist: np.ndarray
    confidence: float
    mediapipe_label: str


@dataclass(frozen=True)
class _Detection:
    landmarks: np.ndarray
    centroid: np.ndarray
    wrist: np.ndarray
    confidence: float
    label: str


class DualHandRouter:
    """Strict mirrored-camera role router with no single-hand fallback.

    MediaPipe label mapping after ``cv2.flip(frame, 1)``:
      ``Left``  -> physical right hand -> mouse
      ``Right`` -> physical left hand  -> writer

    A role is emitted only when label, confidence, and screen-side prior agree.
    Ambiguous detections are discarded; they are never reassigned to the other
    pipeline.
    """

    def __init__(
        self,
        mouse_min_confidence: float = 0.80,
        writer_min_confidence: float = 0.0,
        split_x: float = 0.50,
    ) -> None:
        self.mouse_min_confidence = float(mouse_min_confidence)
        self.writer_min_confidence = float(writer_min_confidence)
        self.split_x = float(split_x)

    def reset(self) -> None:
        """Stateless router compatibility hook."""

    @staticmethod
    def _classification_fields(item: Any) -> tuple[str, float]:
        label = getattr(item, "category_name", None)
        if not label:
            label = getattr(item, "label", None)
        if not label:
            label = getattr(item, "display_name", None)
        score = float(getattr(item, "score", 0.0) or 0.0)
        return str(label or ""), score

    @classmethod
    def _extract(cls, result: Any) -> list[_Detection]:
        # MediaPipe Tasks API.
        raw_landmarks = getattr(result, "hand_landmarks", None)
        raw_handedness = getattr(result, "handedness", None)

        # MediaPipe Solutions API (multi_hand_landmarks / multi_handedness).
        if raw_landmarks is None:
            raw_landmarks = getattr(result, "multi_hand_landmarks", None)
        if raw_handedness is None:
            raw_handedness = getattr(result, "multi_handedness", None)

        landmarks_list = list(raw_landmarks or [])
        handedness_list = list(raw_handedness or [])
        detections: list[_Detection] = []
        for index, hand_landmarks in enumerate(landmarks_list):
            landmarks = getattr(hand_landmarks, "landmark", hand_landmarks)
            points = np.asarray(
                [[lm.x, lm.y, lm.z] for lm in landmarks],
                dtype=np.float64,
            )
            if points.shape != (21, 3):
                continue

            classification: Any | None = None
            if index < len(handedness_list):
                handedness = handedness_list[index]
                candidates = getattr(handedness, "classification", handedness)
                if candidates:
                    classification = candidates[0]
            if classification is None:
                continue
            label, confidence = cls._classification_fields(classification)
            detections.append(
                _Detection(
                    landmarks=points,
                    centroid=points[:, :2].mean(axis=0),
                    wrist=points[0, :2].copy(),
                    confidence=confidence,
                    label=label,
                )
            )
        return detections

    @staticmethod
    def _public(detection: _Detection, track_id: HandTrackId) -> RoutedHandData:
        return RoutedHandData(
            track_id=track_id,
            landmarks=detection.landmarks,
            centroid=detection.centroid.copy(),
            wrist=detection.wrist.copy(),
            confidence=detection.confidence,
            mediapipe_label=detection.label,
        )

    def route(
        self, result: Any
    ) -> tuple[RoutedHandData | None, RoutedHandData | None]:
        writer_candidates: list[_Detection] = []
        mouse_candidates: list[_Detection] = []

        for detection in self._extract(result):
            x = float(detection.centroid[0])
            if (
                detection.label == "Left"
                and x >= self.split_x
                and detection.confidence >= self.mouse_min_confidence
            ):
                mouse_candidates.append(detection)
            elif (
                detection.label == "Right"
                and x < self.split_x
                and detection.confidence >= self.writer_min_confidence
            ):
                writer_candidates.append(detection)
            # Label/space conflict: discard. Never cross-route.

        writer_detection = (
            max(writer_candidates, key=lambda item: item.confidence)
            if writer_candidates
            else None
        )
        mouse_detection = (
            max(mouse_candidates, key=lambda item: item.confidence)
            if mouse_candidates
            else None
        )
        writer = (
            self._public(writer_detection, HandTrackId.LEFT_HAND_WRITER)
            if writer_detection is not None
            else None
        )
        mouse = (
            self._public(mouse_detection, HandTrackId.RIGHT_HAND_MOUSE)
            if mouse_detection is not None
            else None
        )
        return writer, mouse
