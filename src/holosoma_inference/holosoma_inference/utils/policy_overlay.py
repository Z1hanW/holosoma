"""ZMQ helpers for publishing the live policy tracking target overlay."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger


class PolicyOverlayPub:
    """Publish the latest target skeleton/object pose from the policy process."""

    def __init__(self, port: int = 5663) -> None:
        self.port = int(port)
        self.context: Any | None = None
        self.socket: Any | None = None
        self._zmq: Any | None = None
        self.enabled = False

    def start(self) -> None:
        try:
            import zmq

            self._zmq = zmq
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.bind(f"tcp://*:{self.port}")
            self.enabled = True
            logger.info("Policy overlay publisher bound to port {}", self.port)
        except Exception as exc:
            logger.error("Failed to start policy overlay publisher: {}", exc)
            self._zmq = None
            self.enabled = False

    def publish(self, payload: dict) -> None:
        zmq = self._zmq
        if not self.enabled or self.socket is None or zmq is None:
            return
        try:
            self.socket.send_string(json.dumps(payload), zmq.NOBLOCK)
        except zmq.Again:
            pass
        except Exception as exc:
            logger.warning("Policy overlay publish failed: {}", exc)

    def close(self) -> None:
        socket = self.socket
        context = self.context
        self.socket = None
        self.context = None
        self._zmq = None
        if socket is not None:
            socket.close(0)
        if context is not None:
            context.term()
        self.enabled = False


class PolicyOverlaySub:
    """Subscribe to the latest target skeleton/object pose from the policy process."""

    def __init__(self, port: int = 5663) -> None:
        self.port = int(port)
        self.context: Any | None = None
        self.socket: Any | None = None
        self._zmq: Any | None = None
        self.last_payload: dict | None = None

    def start(self) -> None:
        import zmq

        self._zmq = zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://localhost:{self.port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVTIMEO, 10)
        logger.info("Policy overlay subscriber started, connecting to port {}", self.port)

    def _drain_messages(self) -> None:
        zmq = self._zmq
        if self.socket is None or zmq is None:
            return
        while True:
            try:
                self.last_payload = json.loads(self.socket.recv_string(zmq.NOBLOCK))
            except zmq.Again:
                break

    def get_payload(self) -> dict | None:
        self._drain_messages()
        return self.last_payload

    def close(self) -> None:
        socket = self.socket
        context = self.context
        self.socket = None
        self.context = None
        self._zmq = None
        if socket is not None:
            socket.close(0)
        if context is not None:
            context.term()
