"""ZMQ helpers for split sim2sim simulator control requests."""

from __future__ import annotations

import json

import zmq
from loguru import logger


class SimControlPush:
    """Send control messages such as reset requests to the simulator."""

    def __init__(self, port: int = 5559) -> None:
        self.port = int(port)
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.enabled = False

    def start(self) -> None:
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUSH)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.connect(f"tcp://localhost:{self.port}")
            self.enabled = True
            logger.info("Sim control publisher connected to port {}", self.port)
        except Exception as exc:
            logger.error("Failed to start sim control publisher: {}", exc)
            self.enabled = False

    def publish(self, payload: dict) -> None:
        if not self.enabled or self.socket is None:
            return
        try:
            self.socket.send_string(json.dumps(payload), zmq.NOBLOCK)
        except zmq.Again:
            pass
        except Exception as exc:
            logger.warning("Sim control publish failed: {}", exc)

    def request_reset(self, reason: str) -> None:
        self.publish({"action": "reset", "reason": str(reason)})

    def close(self) -> None:
        socket = self.socket
        context = self.context
        self.socket = None
        self.context = None
        if socket is not None:
            socket.close(0)
        if context is not None:
            context.term()
        self.enabled = False
