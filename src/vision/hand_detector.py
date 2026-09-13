"""Hand landmark detection using MediaPipe Hands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import cv2


@dataclass(frozen=True)
class HandDetection:
    """The landmarks and handedness detected in one video frame."""

    landmarks: Sequence[Any]
    handedness: Optional[str]


class HandDetector:
    """Detect one or more hands from OpenCV BGR frames."""

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
    ) -> None:
        try:
            import mediapipe as mp
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "MediaPipe is required for hand detection. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from error

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=0,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame) -> Optional[HandDetection]:
        """Detect the first hand in a BGR frame, if one is present."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None

        handedness = None
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        return HandDetection(
            landmarks=results.multi_hand_landmarks[0].landmark,
            handedness=handedness,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()