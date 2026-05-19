"""Helpers for split sim2sim perception observation publishing."""

from __future__ import annotations

import json
from multiprocessing import shared_memory

import numpy as np
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


class PerceptionObsShmPub:
    """Publish flattened perception observations through a shared-memory buffer.

    The default name matches the legacy MuJoCo image-server depth transport
    (`depth_img_shm`).  The buffer stores contiguous float32 values; for the
    current depth policy this is equivalent to a `[1, 1, 58, 87]` image.
    """

    def __init__(self, name: str = "depth_img_shm") -> None:
        self.name = str(name)
        self.shm: shared_memory.SharedMemory | None = None
        self.array: np.ndarray | None = None
        self.dim: int | None = None
        self.enabled = False

    def start(self) -> None:
        self.enabled = True
        logger.info("Perception obs shared-memory publisher enabled: name={}", self.name)

    def _ensure_buffer(self, dim: int) -> None:
        if self.shm is not None and self.array is not None and self.dim == dim:
            return

        self.close(unlink=False)
        size = int(dim) * np.dtype(np.float32).itemsize
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=size)
            logger.info("Created perception obs shared memory: name={} values={}", self.name, dim)
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
                logger.info("Recreated perception obs shared memory: name={} values={}", self.name, dim)
            else:
                self.shm = existing
                logger.info("Connected to existing perception obs shared memory: name={} values={}", self.name, dim)

        self.dim = int(dim)
        self.array = np.ndarray((self.dim,), dtype=np.float32, buffer=self.shm.buf)

    def publish(self, values: list[float] | np.ndarray) -> None:
        if not self.enabled:
            return
        obs = np.asarray(values, dtype=np.float32).reshape(-1)
        if obs.size <= 0:
            return
        try:
            self._ensure_buffer(int(obs.size))
            if self.array is not None:
                self.array[:] = obs
        except Exception as exc:
            logger.warning("Perception obs shared-memory publish failed: {}", exc)

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
