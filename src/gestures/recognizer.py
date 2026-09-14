"""Recognize static hand gestures from MediaPipe hand landmarks."""

from __future__ import annotations

from math import hypot
from typing import Any, Optional, Sequence


PLAY_PAUSE = "PLAY_PAUSE"
VOLUME_UP = "VOLUME_UP"
VOLUME_DOWN = "VOLUME_DOWN"
SEEK_FORWARD = "SEEK_FORWARD"
SEEK_BACKWARD = "SEEK_BACKWARD"

_LANDMARK_COUNT = 21
_FINGER_TIPS = (8, 12, 16, 20)
_FINGER_PIPS = (6, 10, 14, 18)
_THUMB_TIP_INDEX = 4
_THUMB_IP_INDEX = 3
_INDEX_TIP_INDEX = 8
_PINCH_THRESHOLD = 0.08


class GestureRecognizer:
    """Convert one hand's landmarks into a VLC command."""

    def __init__(self) -> None:
        pass

    def recognize(
        self,
        landmarks: Optional[Sequence[Any]],
        handedness: Optional[str] = None,
    ) -> Optional[str]:
        """Return a command for a supported gesture, or ``None``."""
        if landmarks is None or len(landmarks) < _LANDMARK_COUNT:
            return None

        if self._is_thumb_index_pinch(landmarks):
            return PLAY_PAUSE

        raised_fingers = [
            self._is_finger_extended(landmarks, tip, pip)
            for tip, pip in zip(_FINGER_TIPS, _FINGER_PIPS)
        ]

        index, middle, ring, pinky = raised_fingers
        if index and not middle and not ring and not pinky:
            return VOLUME_UP
        if index and middle and not ring and not pinky:
            return VOLUME_DOWN
        if index and middle and ring and not pinky:
            return SEEK_FORWARD
        if index and middle and ring and pinky and self._is_thumb_tucked(landmarks):
            return SEEK_BACKWARD
        return None

    @staticmethod
    def _is_finger_extended(
        landmarks: Sequence[Any],
        tip_index: int,
        pip_index: int,
    ) -> bool:
        tip = landmarks[tip_index]
        pip = landmarks[pip_index]
        return float(tip.y) < float(pip.y)

    @staticmethod
    def _is_thumb_index_pinch(landmarks: Sequence[Any]) -> bool:
        thumb_tip = landmarks[_THUMB_TIP_INDEX]
        index_tip = landmarks[_INDEX_TIP_INDEX]
        distance = hypot(
            float(thumb_tip.x) - float(index_tip.x),
            float(thumb_tip.y) - float(index_tip.y),
        )
        return distance <= _PINCH_THRESHOLD

    @staticmethod
    def _is_thumb_tucked(landmarks: Sequence[Any]) -> bool:
        wrist = landmarks[0]
        thumb_tip = landmarks[_THUMB_TIP_INDEX]
        thumb_ip = landmarks[_THUMB_IP_INDEX]
        thumb_distance = hypot(
            float(thumb_tip.x) - float(wrist.x),
            float(thumb_tip.y) - float(wrist.y),
        )
        thumb_base_distance = hypot(
            float(thumb_ip.x) - float(wrist.x),
            float(thumb_ip.y) - float(wrist.y),
        )
        return thumb_distance <= thumb_base_distance * 1.15