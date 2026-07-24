"""ZMQ helpers for split sim2sim simulator control requests."""

from __future__ import annotations

import json
import time

import zmq
from loguru import logger

POLICY_CONTROL_ACTIONS = frozenset({"start", "stop", "init", "space"})


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

    def request_reset(self, reason: str, motion_init_mode: str | None = None) -> bool:
        if not self.enabled or self.socket is None:
            return False
        payload_dict = {"action": "reset", "reason": str(reason)}
        if motion_init_mode is not None:
            payload_dict["motion_init_mode"] = str(motion_init_mode)
        payload = json.dumps(payload_dict)
        for _ in range(20):
            try:
                self.socket.send_string(payload, zmq.NOBLOCK)
                return True
            except zmq.Again:
                time.sleep(0.01)
            except Exception as exc:
                logger.warning("Sim control reset publish failed: {}", exc)
                return False
        logger.warning("Sim control reset publish dropped after retries")
        return False

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

    def publish(
        self,
        *,
        enabled: bool,
        mode: str,
        command: list[float] | tuple[float, float, float],
        pickup_button: float | bool | None = None,
        drop_button: float | bool | None = None,
    ) -> None:
        if not self.enabled or self.socket is None:
            return
        payload = {
            "enabled": bool(enabled),
            "mode": str(mode),
            "command": [float(command[0]), float(command[1]), float(command[2])],
            "time": time.time(),
        }
        if pickup_button is not None:
            payload["pickup_button"] = 1.0 if bool(pickup_button) else 0.0
        if drop_button is not None:
            payload["drop_button"] = 1.0 if bool(drop_button) else 0.0
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


class PolicyControlPush:
    """Send policy lifecycle commands from the command web process."""

    def __init__(self, port: int = 5662) -> None:
        self.port = int(port)
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.enabled = False

    def start(self) -> None:
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUSH)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.SNDHWM, 16)
            self.socket.bind(f"tcp://*:{self.port}")
            self.enabled = True
            logger.info("Policy control publisher bound to port {}", self.port)
        except Exception as exc:
            logger.error("Failed to start policy control publisher: {}", exc)
            self.enabled = False

    def publish(self, action: str, *, source: str = "web") -> bool:
        if not self.enabled or self.socket is None:
            return False
        action = str(action).strip().lower()
        if action in {"]", "right_bracket"}:
            action = "start"
        elif action in {" ", "spacebar", "motion", "start_motion", "start_motion_clip"}:
            action = "space"
        if action not in POLICY_CONTROL_ACTIONS:
            raise ValueError(f"Unsupported policy control action: {action}")
        payload = json.dumps({"action": action, "source": str(source), "time": time.time()})
        for _ in range(5):
            try:
                self.socket.send_string(payload, zmq.NOBLOCK)
                return True
            except zmq.Again:
                time.sleep(0.01)
            except Exception as exc:
                logger.warning("Policy control publish failed: {}", exc)
                return False
        logger.warning("Policy control publish dropped after retries: {}", action)
        return False

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
    """Receive policy lifecycle commands in a policy process."""

    def __init__(self, port: int = 5662) -> None:
        self.port = int(port)
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None

    def start(self) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://localhost:{self.port}")
        logger.info("Policy control receiver started, connecting to port {}", self.port)

    def get_actions(self) -> list[str]:
        if self.socket is None:
            return []
        actions: list[str] = []
        while True:
            try:
                payload = json.loads(self.socket.recv_string(zmq.NOBLOCK))
            except zmq.Again:
                break
            except json.JSONDecodeError as exc:
                logger.warning("Policy control ignored invalid JSON: {}", exc)
                continue
            except Exception as exc:
                logger.warning("Policy control receive failed: {}", exc)
                break
            if not isinstance(payload, dict):
                continue
            action = str(payload.get("action", "")).strip().lower()
            if action in {"]", "right_bracket"}:
                action = "start"
            elif action in {" ", "spacebar", "motion", "start_motion", "start_motion_clip"}:
                action = "space"
            if action in POLICY_CONTROL_ACTIONS:
                actions.append(action)
            elif action:
                logger.warning("Policy control ignored unsupported action: {}", action)
        return actions

    def close(self) -> None:
        socket = self.socket
        context = self.context
        self.socket = None
        self.context = None
        if socket is not None:
            socket.close(0)
        if context is not None:
            context.term()
