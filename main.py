"""Run the camera-to-VLC gesture control loop."""

from __future__ import annotations

import os
import time

import cv2

from src.gestures.recognizer import (
	PLAY_PAUSE,
	VOLUME_DOWN,
	VOLUME_UP,
	GestureRecognizer,
)
from src.vlc.controller import VLCConnectionError, VLCController
from src.vision.camera import Camera
from src.vision.hand_detector import HandDetector


GESTURE_COOLDOWN_SECONDS = 0.75


def _env_int(name: str, default: int) -> int:
	value = os.getenv(name)
	return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
	value = os.getenv(name)
	return default if value is None else float(value)


def main() -> None:
	camera = Camera(
		camera_index=_env_int("CAMERA_INDEX", 0),
		width=_env_int("CAMERA_WIDTH", 1280),
		height=_env_int("CAMERA_HEIGHT", 720),
	)
	detector = HandDetector()
	recognizer = GestureRecognizer()
	controller = VLCController(
		host=os.getenv("VLC_HOST", "localhost"),
		port=_env_int("VLC_PORT", 8080),
		password=os.getenv("VLC_PASSWORD", "admin"),
		volume_step=_env_int("VLC_VOLUME_STEP", 20),
		timeout=_env_float("VLC_TIMEOUT", 2.0),
	)
	command_handlers = {
		PLAY_PAUSE: controller.play_pause,
		VOLUME_UP: controller.volume_up,
		VOLUME_DOWN: controller.volume_down,
	}

	last_command = None
	last_command_time = 0.0

	try:
		with camera, detector:
			while True:
				frame = camera.read()
				if frame is None:
					print("Camera frame could not be read.")
					break

				display_frame = cv2.flip(frame, 1)
				detection = detector.detect(display_frame)
				command = None
				if detection is not None:
					command = recognizer.recognize(
						detection.landmarks,
						detection.handedness,
					)

				now = time.monotonic()
				cooldown_elapsed = now - last_command_time
				if (
					command is not None
					and command in command_handlers
					and (command != last_command or cooldown_elapsed >= GESTURE_COOLDOWN_SECONDS)
				):
					try:
						command_handlers[command]()
					except VLCConnectionError as error:
						print(error)
					else:
						last_command = command
						last_command_time = now

				label = command or "NO GESTURE"
				cv2.putText(
					display_frame,
					label,
					(20, 40),
					cv2.FONT_HERSHEY_SIMPLEX,
					1,
					(0, 255, 0),
					2,
				)
				cv2.imshow("VLC Gesture Controller", display_frame)

				key = cv2.waitKey(1) & 0xFF
				if key in (ord("q"), 27):
					break
	finally:
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
