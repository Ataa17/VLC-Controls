"""Recognize static hand gestures from MediaPipe hand landmarks."""

from __future__ import annotations

from typing import Any, Optional, Sequence


PLAY_PAUSE = "PLAY_PAUSE"
VOLUME_UP = "VOLUME_UP"
VOLUME_DOWN = "VOLUME_DOWN"

_LANDMARK_COUNT = 21
_FINGER_TIPS = (8, 12, 16, 20)
_FINGER_PIPS = (6, 10, 14, 18)


class GestureRecognizer:
    """Convert one hand's landmarks into a VLC command."""

    def recognize(
        self,
        landmarks: Optional[Sequence[Any]],
        handedness: Optional[str] = None,
    ) -> Optional[str]:
        """Return a command for a supported gesture, or ``None``."""
        if landmarks is None or len(landmarks) < _LANDMARK_COUNT:
            return None

        raised_fingers = [
            self._is_finger_extended(landmarks, tip, pip)
            for tip, pip in zip(_FINGER_TIPS, _FINGER_PIPS)
        ]

        index, middle, ring, pinky = raised_fingers
        if index and middle and ring and pinky:
            return PLAY_PAUSE
        if index and not middle and not ring and not pinky:
            return VOLUME_UP
        if index and middle and not ring and not pinky:
            return VOLUME_DOWN
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