"""ZMQ helpers for split sim2sim policy control requests."""

from __future__ import annotations

import json

import zmq
from loguru import logger


class PolicyControlPush:
    """Send control messages such as start/stop requests to the policy."""

    def __init__(self, port: int = 5560) -> None:
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
            logger.info("Policy control publisher connected to port {}", self.port)
        except Exception as exc:
            logger.error("Failed to start policy control publisher: {}", exc)
            self.enabled = False

    def publish(self, payload: dict) -> None:
        if not self.enabled or self.socket is None:
            return
        try:
            self.socket.send_string(json.dumps(payload), zmq.NOBLOCK)
        except zmq.Again:
            pass
        except Exception as exc:
            logger.warning("Policy control publish failed: {}", exc)

    def request_action(self, action: str, **payload: object) -> None:
        message = {"action": str(action)}
        message.update(payload)
        self.publish(message)

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


class PolicyControlPull:
    """Receive policy control requests from split sim2sim viewer tools."""

    def __init__(self, port: int = 5560) -> None:
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
            logger.info("Policy control subscriber started on port {}", self.port)
        except Exception as exc:
            logger.error("Failed to start policy control subscriber: {}", exc)
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
                logger.warning("Policy control receive failed: {}", exc)
                break
        return payloads

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
        if self.context is not None:
            self.context.term()
        self.enabled = False
