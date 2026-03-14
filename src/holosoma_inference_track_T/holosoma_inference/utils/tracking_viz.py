"""Tracking-visualization publisher for sim2sim MuJoCo overlays."""

from __future__ import annotations

import json

import zmq
from loguru import logger


class TrackingVizPub:
    """Publish tracking-debug payloads via ZMQ PUB/SUB."""

    def __init__(self, port: int = 5560) -> None:
        self.port: int = int(port)
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.enabled: bool = False

    def start(self) -> None:
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(f"tcp://*:{self.port}")
            self.enabled = True
            logger.info(f"Tracking-viz publisher started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start tracking-viz publisher: {e}")
            self.enabled = False

    def publish(self, payload: dict[str, object]) -> None:
        if not self.enabled or self.socket is None:
            return
        try:
            self.socket.send_string(json.dumps(payload), zmq.NOBLOCK)
        except zmq.Again:
            pass
        except Exception as e:
            logger.debug(f"Tracking-viz publish failed: {e}")

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
        if self.context is not None:
            self.context.term()
        self.enabled = False

