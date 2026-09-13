"""Webcam capture for the VLC gesture controller."""

from __future__ import annotations

from typing import Optional

import cv2


class Camera:
	"""Manage a webcam and return captured frames to the application."""

	def __init__(
		self,
		camera_index: int = 0,
		width: Optional[int] = None,
		height: Optional[int] = None,
	) -> None:
		self.camera_index = camera_index
		self.width = width
		self.height = height
		self._capture: Optional[cv2.VideoCapture] = None

	def open(self) -> None:
		"""Open the configured webcam."""
		if self.is_open:
			return

		capture = cv2.VideoCapture(self.camera_index)
		if not capture.isOpened():
			capture.release()
			raise RuntimeError(
				f"Unable to open camera at index {self.camera_index}."
			)

		if self.width is not None:
			capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
		if self.height is not None:
			capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

		self._capture = capture

	@property
	def is_open(self) -> bool:
		"""Return whether the webcam is currently open."""
		return self._capture is not None and self._capture.isOpened()

	def read(self):
		"""Return the next frame, or ``None`` when capture fails."""
		if not self.is_open:
			raise RuntimeError("Camera is not open. Call open() before read().")

		success, frame = self._capture.read()
		return frame if success else None

	def release(self) -> None:
		"""Release the webcam if it is open."""
		if self._capture is not None:
			self._capture.release()
			self._capture = None

	def __enter__(self) -> "Camera":
		self.open()
		return self

	def __exit__(self, exc_type, exc_value, traceback) -> None:
		self.release()
