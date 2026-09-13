"""Hand landmark detection using MediaPipe Hands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
from typing import Any, Optional, Sequence
from urllib.request import urlopen

import cv2


_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


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
        model_path: Optional[str] = None,
    ) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "MediaPipe is required for hand detection. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from error

        resolved_model_path = self._get_model_path(model_path)
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(resolved_model_path),
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame) -> Optional[HandDetection]:
        """Detect the first hand in a BGR frame, if one is present."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb_frame,
        )
        results = self._landmarker.detect(image)

        if not results.hand_landmarks:
            return None

        handedness = None
        if results.handedness and results.handedness[0]:
            handedness = results.handedness[0][0].category_name

        return HandDetection(
            landmarks=results.hand_landmarks[0],
            handedness=handedness,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()

    @staticmethod
    def _get_model_path(model_path: Optional[str]) -> Path:
        path = Path(model_path or os.getenv("HAND_MODEL_PATH", "models/hand_landmarker.task"))
        if path.exists():
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urlopen(_MODEL_URL, timeout=30) as response:
                with tempfile.NamedTemporaryFile(
                    dir=path.parent,
                    delete=False,
                ) as temporary_file:
                    temporary_file.write(response.read())
                    temporary_path = Path(temporary_file.name)
            temporary_path.replace(path)
        except OSError as error:
            raise RuntimeError(
                f"Unable to download the hand model to {path}. "
                "Set HAND_MODEL_PATH to a local hand_landmarker.task file."
            ) from error
        return path

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()