"""Rotation-independent 3D hand-pose geometry."""

from __future__ import annotations

from enum import Enum

import numpy as np

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

EXTENDED_MAX_DEG = 30.0
CURLED_MIN_DEG = 90.0


class Finger(str, Enum):
    THUMB = "THUMB"
    INDEX = "INDEX"
    MIDDLE = "MIDDLE"
    RING = "RING"
    PINKY = "PINKY"


_BONES = {
    Finger.THUMB: (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    Finger.INDEX: (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    Finger.MIDDLE: (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    Finger.RING: (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    Finger.PINKY: (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}


def _xyz(landmarks: np.ndarray, index: int) -> np.ndarray:
    return np.asarray(landmarks[index, :3], dtype=np.float64)


def vector_angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm <= 1e-9:
        return 180.0
    cosine = float(np.dot(a, b) / norm)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def finger_angle_degrees(landmarks: np.ndarray, finger: Finger) -> float:
    proximal, pip, dip, tip = _BONES[finger]
    v1 = _xyz(landmarks, pip) - _xyz(landmarks, proximal)
    v2 = _xyz(landmarks, tip) - _xyz(landmarks, dip)
    return vector_angle_degrees(v1, v2)


def finger_extended(landmarks: np.ndarray, finger: Finger) -> bool:
    return finger_angle_degrees(landmarks, finger) < EXTENDED_MAX_DEG


def finger_curled(landmarks: np.ndarray, finger: Finger) -> bool:
    return finger_angle_degrees(landmarks, finger) > CURLED_MIN_DEG


def palm_scale(landmarks: np.ndarray) -> float:
    return max(
        float(np.linalg.norm(_xyz(landmarks, MIDDLE_MCP) - _xyz(landmarks, WRIST))),
        1e-6,
    )


def normalized_distance(landmarks: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(_xyz(landmarks, a) - _xyz(landmarks, b))) / palm_scale(
        landmarks
    )


def finger_spread_degrees(
    landmarks: np.ndarray,
    first_mcp: int,
    first_tip: int,
    second_mcp: int,
    second_tip: int,
) -> float:
    first = _xyz(landmarks, first_tip) - _xyz(landmarks, first_mcp)
    second = _xyz(landmarks, second_tip) - _xyz(landmarks, second_mcp)
    return vector_angle_degrees(first, second)


def thumb_folded_over_fingers(landmarks: np.ndarray) -> bool:
    thumb_tip = _xyz(landmarks, THUMB_TIP)
    nearest_finger = min(
        float(np.linalg.norm(thumb_tip - _xyz(landmarks, joint)))
        for joint in (INDEX_PIP, MIDDLE_PIP, RING_PIP)
    )
    return nearest_finger / palm_scale(landmarks) < 0.80
