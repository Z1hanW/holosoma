"""Object-state synchronization subscriber for sim2sim inference."""

from __future__ import annotations

import json

import zmq
from loguru import logger


class ObjectStateSub:
    """Receive latest object state from simulator bridge via ZMQ."""

    def __init__(self, port: int = 5557) -> None:
        self.port: int = int(port)
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self._last_state: dict[str, object] | None = None
        self._enabled: bool = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{self.port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVTIMEO, 10)
        self._enabled = True
        logger.info(f"Object-state subscriber started, connecting to port {self.port}")

    def _drain_messages(self) -> None:
        if self.socket is None:
            return
        while True:
            try:
                message = self.socket.recv_string(zmq.NOBLOCK)
                payload = json.loads(message)
                if isinstance(payload, dict):
                    self._last_state = payload
            except zmq.Again:  # noqa: PERF203
                break
            except Exception as e:
                logger.debug(f"Dropping invalid object-state message: {e}")
                break

    def get_state(self) -> dict[str, object] | None:
        """Return the most recent object-state payload, if any."""
        self._drain_messages()
        return self._last_state

    def close(self) -> None:
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self._enabled = False
