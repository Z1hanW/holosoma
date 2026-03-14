"""ZMQ helpers for split sim2sim perception observation publishing."""

from __future__ import annotations

import json

import zmq
from loguru import logger


class PerceptionObsPub:
    """Publish flattened perception observations for split sim2sim inference."""

    def __init__(self, port: int = 5558) -> None:
        self.port = int(port)
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.enabled = False

    def start(self) -> None:
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(f"tcp://*:{self.port}")
            self.enabled = True
            logger.info("Perception obs publisher started on port {}", self.port)
        except Exception as exc:
            logger.error("Failed to start perception obs publisher: {}", exc)
            self.enabled = False

    def publish(self, payload: dict) -> None:
        if not self.enabled or self.socket is None:
            return
        try:
            self.socket.send_string(json.dumps(payload), zmq.NOBLOCK)
        except zmq.Again:
            pass
        except Exception as exc:
            logger.warning("Perception obs publish failed: {}", exc)

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
        if self.context is not None:
            self.context.term()
        self.enabled = False
