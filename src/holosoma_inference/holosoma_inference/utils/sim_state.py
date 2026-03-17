"""ZMQ helpers for split sim2sim simulator state subscription."""

from __future__ import annotations

import json

import zmq
from loguru import logger


class SimStateSub:
    """Subscribe to simulator robot/object state published by split sim2sim."""

    def __init__(self, port: int = 5557) -> None:
        self.port = int(port)
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.last_state: dict | None = None

    def start(self) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://localhost:{self.port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVTIMEO, 10)
        logger.info("Sim state subscriber started, connecting to port {}", self.port)

    def _drain_messages(self) -> None:
        if self.socket is None:
            return
        while True:
            try:
                self.last_state = json.loads(self.socket.recv_string(zmq.NOBLOCK))
            except zmq.Again:  # noqa: PERF203
                break

    def get_state(self) -> dict | None:
        self._drain_messages()
        return self.last_state

    def close(self) -> None:
        socket = self.socket
        context = self.context
        self.socket = None
        self.context = None
        if socket is not None:
            socket.close(0)
        if context is not None:
            context.term()
