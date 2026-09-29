"""ZMQ helpers for split sim2sim simulator control requests."""

from __future__ import annotations

import json

import zmq
from loguru import logger


class SimControlPull:
    """Receive control messages such as reset requests from inference-side tools."""

    def __init__(self, port: int = 5559) -> None:
        self.port = int(port)
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.enabled = False

    def start(self) -> None:
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PULL)
            self.socket.bind(f"tcp://*:{self.port}")
            self.enabled = True
            logger.info("Sim control subscriber started on port {}", self.port)
        except Exception as exc:
            logger.error("Failed to start sim control subscriber: {}", exc)
            self.enabled = False

    def drain(self) -> list[dict]:
        if not self.enabled or self.socket is None:
            return []
        payloads: list[dict] = []
        while True:
            try:
                payloads.append(json.loads(self.socket.recv_string(zmq.NOBLOCK)))
            except zmq.Again:
                break
            except Exception as exc:
                logger.warning("Sim control receive failed: {}", exc)
                break
        return payloads

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
        if self.context is not None:
            self.context.term()
        self.enabled = False
