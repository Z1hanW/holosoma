"""Helpers for split sim2sim perception observation subscription."""

from __future__ import annotations

import json
import os
from collections import deque
from multiprocessing import resource_tracker
from multiprocessing import shared_memory
from typing import Any

import numpy as np
from loguru import logger


class PerceptionObsSub:
    """Subscribe to simulator perception observations published by split sim2sim."""

    def __init__(self, port: int = 5558) -> None:
        self.port = int(port)
        self.context: Any | None = None
        self.socket: Any | None = None
        self._zmq: Any | None = None
        self.last_payload: dict | None = None
        self.payload_buffer: deque[dict] = deque(maxlen=512)

    def start(self) -> None:
        import zmq

        self._zmq = zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://localhost:{self.port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVTIMEO, 10)
        logger.info("Perception obs subscriber started, connecting to port {}", self.port)

    def _drain_messages(self) -> None:
        zmq = self._zmq
        if self.socket is None or zmq is None:
            return
        while True:
            try:
                self.last_payload = json.loads(self.socket.recv_string(zmq.NOBLOCK))
                if isinstance(self.last_payload, dict):
                    self.payload_buffer.append(self.last_payload)
            except zmq.Again:
                break

    def get_payload(self) -> dict | None:
        self._drain_messages()
        return self.last_payload

    def get_payload_at_or_before(self, sim_time_ms: float | int | None) -> dict | None:
        self._drain_messages()
        if sim_time_ms is None:
            return self.last_payload
        try:
            target_ms = float(sim_time_ms)
        except (TypeError, ValueError):
            return self.last_payload
        try:
            future_tolerance_ms = float(os.environ.get("HOLOSOMA_PERCEPTION_FUTURE_TOLERANCE_MS", "2") or "2")
        except ValueError:
            future_tolerance_ms = 2.0
        selected: dict | None = None
        for payload in reversed(self.payload_buffer):
            try:
                payload_ms = float(payload.get("sim_time_ms"))
            except (TypeError, ValueError):
                continue
            if payload_ms <= target_ms + max(future_tolerance_ms, 0.0):
                selected = payload
                break
        if selected is not None:
            return selected
        if self.last_payload is None:
            return None
        try:
            last_ms = float(self.last_payload.get("sim_time_ms"))
        except (TypeError, ValueError, AttributeError):
            return self.last_payload
        if last_ms <= target_ms + max(future_tolerance_ms, 0.0):
            return self.last_payload
        return None

    def reset(self) -> None:
        self.last_payload = None
        self.payload_buffer.clear()

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


class PerceptionObsShmSub:
    """Read flattened perception observations from legacy depth shared memory."""

    def __init__(self, name: str = "depth_img_shm") -> None:
        self.name = str(name)
        self.shm: shared_memory.SharedMemory | None = None
        self.array: np.ndarray | None = None
        self.dim: int | None = None

    def start(self) -> None:
        logger.info("Perception obs shared-memory subscriber configured: name={}", self.name)

    def _ensure_attached(self, expected_dim: int) -> bool:
        if self.shm is not None and self.array is not None and self.dim == int(expected_dim):
            return True

        self.close()
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)
        except FileNotFoundError:
            return False
        # Python's resource_tracker registers attached shared-memory segments even
        # for consumers. When the subscriber process exits, that can unlink the
        # producer-owned segment and break future attaches after a policy restart.
        try:
            resource_tracker.unregister(self.shm._name, "shared_memory")
        except Exception:
            pass

        expected_bytes = int(expected_dim) * np.dtype(np.float32).itemsize
        if len(self.shm.buf) < expected_bytes:
            logger.warning(
                "Perception obs shared memory '{}' is too small: {} bytes < {} bytes",
                self.name,
                len(self.shm.buf),
                expected_bytes,
            )
            self.close()
            return False

        self.dim = int(expected_dim)
        self.array = np.ndarray((self.dim,), dtype=np.float32, buffer=self.shm.buf)
        logger.info("Attached perception obs shared memory: name={} values={}", self.name, self.dim)
        return True

    def get_obs(self, expected_dim: int) -> np.ndarray | None:
        if not self._ensure_attached(int(expected_dim)):
            return None
        if self.array is None:
            return None
        return self.array.copy().reshape(1, int(expected_dim)).astype(np.float32, copy=False)

    def close(self) -> None:
        shm = self.shm
        self.shm = None
        self.array = None
        self.dim = None
        if shm is not None:
            shm.close()


class PerceptionObsShmMirror:
    """Publish the exact policy-consumed perception tensor for debugging/preview."""

    def __init__(self, name: str) -> None:
        self.name = str(name)
        self.shm: shared_memory.SharedMemory | None = None
        self.array: np.ndarray | None = None
        self.dim: int | None = None

    def _ensure_buffer(self, dim: int) -> None:
        if self.shm is not None and self.array is not None and self.dim == int(dim):
            return
        self.close(unlink=False)
        size = int(dim) * np.dtype(np.float32).itemsize
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=size)
            logger.info("Created policy perception mirror shared memory: name={} values={}", self.name, dim)
        except FileExistsError:
            existing = shared_memory.SharedMemory(name=self.name, create=False)
            if len(existing.buf) != size:
                existing.close()
                try:
                    stale = shared_memory.SharedMemory(name=self.name, create=False)
                    stale.unlink()
                    stale.close()
                except FileNotFoundError:
                    pass
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=size)
                logger.info("Recreated policy perception mirror shared memory: name={} values={}", self.name, dim)
            else:
                self.shm = existing
                logger.info("Connected to policy perception mirror shared memory: name={} values={}", self.name, dim)
        try:
            resource_tracker.unregister(self.shm._name, "shared_memory")
        except Exception:
            pass
        self.dim = int(dim)
        self.array = np.ndarray((self.dim,), dtype=np.float32, buffer=self.shm.buf)

    def publish(self, values: np.ndarray) -> None:
        obs = np.asarray(values, dtype=np.float32).reshape(-1)
        if obs.size <= 0:
            return
        self._ensure_buffer(int(obs.size))
        if self.array is not None:
            self.array[:] = obs

    def reset(self, fill_value: float = 0.0) -> None:
        if self.array is not None:
            self.array[:] = np.float32(fill_value)

    def close(self, *, unlink: bool = False) -> None:
        shm = self.shm
        self.shm = None
        self.array = None
        self.dim = None
        if shm is not None:
            if unlink:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            shm.close()
