"""Helpers for split sim2sim perception observation subscription."""

from __future__ import annotations

import json
from multiprocessing import shared_memory

import numpy as np
import zmq
from loguru import logger


class PerceptionObsSub:
    """Subscribe to simulator perception observations published by split sim2sim."""

    def __init__(self, port: int = 5558) -> None:
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
        logger.info("Perception obs subscriber started, connecting to port {}", self.port)

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
