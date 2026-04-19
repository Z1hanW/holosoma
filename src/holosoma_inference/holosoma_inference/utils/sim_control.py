"""ZMQ helpers for split sim2sim simulator control requests."""

from __future__ import annotations

import json
import time

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
        if not self.enabled or self.socket is None:
            return
        payload = json.dumps({"action": "reset", "reason": str(reason)})
        for _ in range(20):
            try:
                self.socket.send_string(payload, zmq.NOBLOCK)
                return
            except zmq.Again:
                time.sleep(0.01)
            except Exception as exc:
                logger.warning("Sim control reset publish failed: {}", exc)
                return
        logger.warning("Sim control reset publish dropped after retries")

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


class ManualRootCommandPub:
    """Publish root-frame sparse command overrides for split WBT rollouts."""

    def __init__(self, port: int = 5661) -> None:
        self.port = int(port)
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.enabled = False

    def start(self) -> None:
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.bind(f"tcp://*:{self.port}")
            self.enabled = True
            logger.info("Manual root command publisher bound to port {}", self.port)
        except Exception as exc:
            logger.error("Failed to start manual root command publisher: {}", exc)
            self.enabled = False

    def publish(self, *, enabled: bool, mode: str, command: list[float] | tuple[float, float, float]) -> None:
        if not self.enabled or self.socket is None:
            return
        payload = {
            "enabled": bool(enabled),
            "mode": str(mode),
            "command": [float(command[0]), float(command[1]), float(command[2])],
            "time": time.time(),
        }
        try:
            self.socket.send_string(json.dumps(payload), zmq.NOBLOCK)
        except zmq.Again:
            pass
        except Exception as exc:
            logger.warning("Manual root command publish failed: {}", exc)

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


class ManualRootCommandSub:
    """Subscribe to root-frame sparse command overrides for split WBT rollouts."""

    def __init__(self, port: int = 5661) -> None:
        self.port = int(port)
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.last_payload: dict | None = None

    def start(self) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://localhost:{self.port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVTIMEO, 10)
        logger.info("Manual root command subscriber started, connecting to port {}", self.port)

    def _drain_messages(self) -> None:
        if self.socket is None:
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
        if socket is not None:
            socket.close(0)
        if context is not None:
            context.term()
