"""Control VLC through its HTTP interface."""

from __future__ import annotations

from typing import Any, Optional

import requests


class VLCConnectionError(RuntimeError):
    """Raised when a VLC HTTP command cannot be delivered."""


class VLCController:
    """Send playback commands to a VLC HTTP interface."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        password: str = "admin",
        volume_step: int = 20,
        timeout: float = 2.0,
        session: Optional[Any] = None,
    ) -> None:
        self.url = f"http://{host}:{port}/requests/status.json"
        self.password = password
        self.volume_step = volume_step
        self.timeout = timeout
        self._session = session or requests.Session()

    def play_pause(self) -> None:
        """Toggle VLC playback."""
        self._send_command("pl_pause")

    def volume_up(self) -> None:
        """Increase VLC volume by the configured step."""
        self._send_command("volume", f"+{self.volume_step}")

    def volume_down(self) -> None:
        """Decrease VLC volume by the configured step."""
        self._send_command("volume", f"-{self.volume_step}")

    def seek_forward(self, seconds: int = 10) -> None:
        """Seek forward by the given number of seconds."""
        self._send_command("seek", f"+{seconds}")

    def seek_backward(self, seconds: int = 10) -> None:
        """Seek backward by the given number of seconds."""
        self._send_command("seek", f"-{seconds}")

    def next_track(self) -> None:
        """Switch VLC to the next track."""
        self._send_command("pl_next")

    def previous_track(self) -> None:
        """Switch VLC to the previous track."""
        self._send_command("pl_previous")

    def _send_command(self, command: str, value: Optional[str] = None) -> None:
        params = {"command": command}
        if value is not None:
            params["val"] = value

        try:
            response = self._session.get(
                self.url,
                params=params,
                auth=("", self.password),
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else "unknown"
            raise VLCConnectionError(
                f"Unable to send VLC command: {command} (HTTP {status})"
            ) from error
        except requests.RequestException as error:
            raise VLCConnectionError(
                f"Unable to send VLC command: {command}"
            ) from error