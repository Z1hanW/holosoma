"""Helpers for split sim2sim perception observation publishing."""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import os
import secrets
import struct
import threading
import time
from multiprocessing import shared_memory

import numpy as np
import zmq
from loguru import logger


# Shared-memory protocol v1.  Keep these values in sync with
# holosoma_inference.utils.perception_obs.  The fixed, cache-line-aligned header
# lets readers reject legacy bare-float buffers before constructing an ndarray.
PERCEPTION_OBS_SHM_MAGIC = b"HSPOSHM\0"
PERCEPTION_OBS_SHM_VERSION = 1
PERCEPTION_OBS_SHM_HEADER_SIZE = 128
PERCEPTION_OBS_SHM_DTYPE_FLOAT32_LE = 1
PERCEPTION_OBS_SHM_FLAG_CONTRACT_SHA256 = 1
PERCEPTION_OBS_SHM_FLAG_EPISODE_GENERATION = 2
PERCEPTION_OBS_SHM_HEADER = struct.Struct("<8sIIIIQQQQqQQQ40s")
PERCEPTION_OBS_SHM_SEQUENCE_OFFSET = 40
PERCEPTION_OBS_SHM_PUBLISH_MONOTONIC_NS_OFFSET = 48
PERCEPTION_OBS_SHM_SIM_TIME_NS_OFFSET = 56
PERCEPTION_OBS_SHM_CHECKSUM_OFFSET = 72
PERCEPTION_OBS_SHM_AUTH_FIELD_OFFSET = 88
_ATOMIC_SEQ_CST = 5

if PERCEPTION_OBS_SHM_HEADER.size != PERCEPTION_OBS_SHM_HEADER_SIZE:  # pragma: no cover - import invariant
    raise RuntimeError("Invalid perception_obs shared-memory header size")

try:  # libatomic is part of the GCC runtime on supported Linux targets.
    _libatomic_path = ctypes.util.find_library("atomic")
    if not _libatomic_path:
        raise OSError("libatomic not found")
    _libatomic = ctypes.CDLL(_libatomic_path)
    _atomic_store_8 = _libatomic.__atomic_store_8
    _atomic_store_8.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
    _atomic_store_8.restype = None
    _atomic_exchange_8 = _libatomic.__atomic_exchange_8
    _atomic_exchange_8.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
    _atomic_exchange_8.restype = ctypes.c_uint64
    try:
        _atomic_thread_fence = _libatomic.atomic_thread_fence
        _atomic_thread_fence.argtypes = [ctypes.c_int]
        _atomic_thread_fence.restype = None
    except AttributeError:  # libatomic before LIBATOMIC_1.2
        _atomic_thread_fence = None
except (AttributeError, OSError):  # pragma: no cover - unsupported platform
    _atomic_store_8 = None
    _atomic_exchange_8 = None
    _atomic_thread_fence = None

_fence_word = ctypes.c_uint64(0)


def _require_atomic_runtime() -> None:
    if _atomic_store_8 is None or (_atomic_thread_fence is None and _atomic_exchange_8 is None):
        raise RuntimeError(
            "Protocol-v1 perception_obs shared memory requires libatomic for cross-process memory ordering"
        )


def _atomic_store_sequence(buffer, value: int) -> None:
    _require_atomic_runtime()
    address = ctypes.addressof(ctypes.c_char.from_buffer(buffer, PERCEPTION_OBS_SHM_SEQUENCE_OFFSET))
    _atomic_store_8(address, value, _ATOMIC_SEQ_CST)


def _full_memory_fence() -> None:
    _require_atomic_runtime()
    if _atomic_thread_fence is not None:
        _atomic_thread_fence(_ATOMIC_SEQ_CST)
    else:  # pragma: no cover - compatibility with older libatomic
        _atomic_exchange_8(ctypes.addressof(_fence_word), 0, _ATOMIC_SEQ_CST)


def _payload_checksum(payload: bytes | memoryview) -> int:
    """Return the protocol's deterministic 64-bit payload checksum."""
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _contract_digest_bytes(contract_sha256: str | None) -> bytes:
    if contract_sha256 is None:
        return b""
    if not isinstance(contract_sha256, str) or len(contract_sha256) != 64:
        raise ValueError("perception observation contract SHA-256 must be a 64-character hex string")
    try:
        digest = bytes.fromhex(contract_sha256)
    except ValueError as exc:
        raise ValueError("perception observation contract SHA-256 must contain only hexadecimal digits") from exc
    if len(digest) != 32:  # pragma: no cover - guarded by the string length
        raise ValueError("perception observation contract SHA-256 must decode to 32 bytes")
    return digest


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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
            socket = self.socket
            context = self.context
            self.socket = None
            self.context = None
            self.enabled = False
            if socket is not None:
                socket.close(0)
            if context is not None:
                context.term()
            raise RuntimeError(
                f"Failed to start perception obs publisher on port {self.port}."
            ) from exc

    def publish(self, payload: dict) -> None:
        if not self.enabled or self.socket is None:
            raise RuntimeError("Perception obs publisher is not enabled.")
        try:
            self.socket.send_string(json.dumps(payload), zmq.NOBLOCK)
        except zmq.Again as exc:
            raise RuntimeError("Perception obs publisher would drop a frame.") from exc
        except Exception as exc:
            raise RuntimeError("Perception obs publish failed.") from exc

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
        if self.context is not None:
            self.context.term()
        self.enabled = False


class PerceptionObsShmPub:
    """Publish finite flattened observations using protocol-v1 shared memory.

    A single writer owns each named segment.  Frames are committed with an
    odd/even sequence lock and a payload checksum, so readers never accept a
    partially copied frame.  Bare-float legacy segments are deliberately not
    reused.
    """

    def __init__(self, name: str = "depth_img_shm") -> None:
        _require_atomic_runtime()
        self.name = str(name)
        self.shm: shared_memory.SharedMemory | None = None
        self.array: np.ndarray | None = None
        self.dim: int | None = None
        self.generation: int | None = None
        self.enabled = False
        self._owns_segment = False
        self._sequence = 0
        self._contract_sha256: str | None = None
        self._carries_episode_generation: bool | None = None
        self._last_publish_monotonic_ns = 0
        self._publish_lock = threading.RLock()
        self._reset_on_first_buffer = os.environ.get(
            "HOLOSOMA_PERCEPTION_OBS_SHM_RESET_ON_START", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._did_reset_existing = False

    def start(self) -> None:
        self.enabled = True
        logger.info("Perception obs shared-memory publisher enabled: name={}", self.name)

    def _ensure_buffer(
        self,
        dim: int,
        contract_sha256: str | None,
        *,
        carries_episode_generation: bool,
        initial_episode_generation: int,
    ) -> None:
        contract_digest = _contract_digest_bytes(contract_sha256)
        normalized_contract = contract_digest.hex() if contract_digest else None
        if self.shm is not None and self.array is not None and self.dim == dim:
            if self._contract_sha256 != normalized_contract:
                raise RuntimeError(
                    "perception observation contract changed after shared-memory publication started"
                )
            if self._carries_episode_generation != bool(carries_episode_generation):
                raise RuntimeError(
                    "perception observation episode-generation authentication changed after "
                    "shared-memory publication started"
                )
            return

        if dim <= 0:
            raise ValueError(f"perception_obs dimension must be positive, got {dim}")

        # A dimension change publishes a new generation under the same name;
        # readers detect the unlink/recreate through both inode and generation.
        self.close(unlink=self._owns_segment)
        payload_bytes = int(dim) * np.dtype("<f4").itemsize
        size = PERCEPTION_OBS_SHM_HEADER_SIZE + payload_bytes
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=size)
        except FileExistsError:
            if not self._reset_on_first_buffer or self._did_reset_existing:
                raise RuntimeError(
                    f"perception_obs shared memory {self.name!r} already exists; "
                    "refusing to attach a second writer"
                )

            existing: shared_memory.SharedMemory | None = None
            try:
                existing = shared_memory.SharedMemory(name=self.name, create=False)
                existing_pid = 0
                if len(existing.buf) >= PERCEPTION_OBS_SHM_HEADER_SIZE:
                    try:
                        header = PERCEPTION_OBS_SHM_HEADER.unpack_from(existing.buf, 0)
                        if header[0] == PERCEPTION_OBS_SHM_MAGIC and header[1] == PERCEPTION_OBS_SHM_VERSION:
                            existing_pid = int(header[12])
                    except (ValueError, struct.error):
                        existing_pid = 0
                if _pid_is_alive(existing_pid):
                    raise RuntimeError(
                        f"perception_obs shared memory {self.name!r} is owned by live publisher "
                        f"pid={existing_pid}; refusing to replace it"
                    )
                existing.unlink()
            except FileNotFoundError:
                pass
            finally:
                if existing is not None:
                    existing.close()

            self._did_reset_existing = True
            try:
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=size)
            except FileExistsError as exc:
                raise RuntimeError(
                    f"perception_obs shared memory {self.name!r} was claimed by another publisher"
                ) from exc

        self.dim = int(dim)
        self.generation = secrets.randbits(64) or 1
        self._sequence = 0
        self._contract_sha256 = normalized_contract
        self._carries_episode_generation = bool(carries_episode_generation)
        self._last_publish_monotonic_ns = 0
        self._owns_segment = True
        assert self.shm is not None
        flags = 0
        if contract_digest:
            flags |= PERCEPTION_OBS_SHM_FLAG_CONTRACT_SHA256
        if carries_episode_generation:
            flags |= PERCEPTION_OBS_SHM_FLAG_EPISODE_GENERATION
        auth_field = (
            contract_digest.ljust(32, b"\0")
            + struct.pack("<Q", initial_episode_generation if carries_episode_generation else 0)
        )
        PERCEPTION_OBS_SHM_HEADER.pack_into(
            self.shm.buf,
            0,
            PERCEPTION_OBS_SHM_MAGIC,
            PERCEPTION_OBS_SHM_VERSION,
            PERCEPTION_OBS_SHM_HEADER_SIZE,
            PERCEPTION_OBS_SHM_DTYPE_FLOAT32_LE,
            flags,
            self.dim,
            self.generation,
            0,
            0,
            -1,
            payload_bytes,
            0,
            os.getpid(),
            auth_field,
        )
        self.array = np.ndarray(
            (self.dim,),
            dtype=np.dtype("<f4"),
            buffer=self.shm.buf,
            offset=PERCEPTION_OBS_SHM_HEADER_SIZE,
        )
        logger.info(
            "Created perception obs shared memory protocol v{}: name={} values={} generation={}",
            PERCEPTION_OBS_SHM_VERSION,
            self.name,
            self.dim,
            self.generation,
        )

    def publish(
        self,
        values: list[float] | np.ndarray,
        sim_time_ms: float | int | None = None,
        contract_sha256: str | None = None,
        episode_generation: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        with np.errstate(over="ignore", invalid="ignore"):
            obs = np.asarray(values, dtype=np.dtype("<f4")).reshape(-1)
        if obs.size <= 0:
            raise ValueError("perception_obs payload must not be empty")
        if not np.isfinite(obs).all():
            raise ValueError("perception_obs payload contains NaN or Inf")
        obs = np.ascontiguousarray(obs, dtype=np.dtype("<f4"))

        if sim_time_ms is None:
            sim_time_ns = -1
        else:
            sim_time_value = float(sim_time_ms)
            if not np.isfinite(sim_time_value) or sim_time_value < 0.0:
                raise ValueError(f"sim_time_ms must be finite and non-negative, got {sim_time_ms!r}")
            sim_time_ns = int(round(sim_time_value * 1_000_000.0))
            if sim_time_ns > (1 << 63) - 1:
                raise ValueError(f"sim_time_ms is out of protocol range: {sim_time_ms!r}")

        if episode_generation is None:
            normalized_episode_generation = 0
            carries_episode_generation = False
        else:
            if (
                isinstance(episode_generation, bool)
                or not isinstance(episode_generation, (int, np.integer))
                or int(episode_generation) < 0
                or int(episode_generation) > (1 << 63) - 1
            ):
                raise ValueError(
                    "episode_generation must be a non-negative integer within the protocol range, "
                    f"got {episode_generation!r}"
                )
            normalized_episode_generation = int(episode_generation)
            carries_episode_generation = True

        payload = obs.tobytes(order="C")
        checksum = _payload_checksum(payload)
        with self._publish_lock:
            self._ensure_buffer(
                int(obs.size),
                contract_sha256,
                carries_episode_generation=carries_episode_generation,
                initial_episode_generation=normalized_episode_generation,
            )
            if self.shm is None or self.array is None:
                raise RuntimeError("perception_obs shared-memory buffer was not initialized")
            if self._sequence >= (1 << 64) - 2:
                raise RuntimeError("perception_obs shared-memory sequence exhausted")

            odd_sequence = self._sequence + 1
            even_sequence = odd_sequence + 1
            publish_monotonic_ns = max(time.monotonic_ns(), self._last_publish_monotonic_ns + 1)

            # Sequence is written first and last.  A reader accepts a frame only if
            # both reads observe the same non-zero even value and the checksum
            # matches, covering both torn payloads and reordered/stale header data.
            _atomic_store_sequence(self.shm.buf, odd_sequence)
            _full_memory_fence()
            self.array[:] = obs
            struct.pack_into(
                "<QqQ",
                self.shm.buf,
                PERCEPTION_OBS_SHM_PUBLISH_MONOTONIC_NS_OFFSET,
                publish_monotonic_ns,
                sim_time_ns,
                int(obs.nbytes),
            )
            struct.pack_into("<Q", self.shm.buf, PERCEPTION_OBS_SHM_CHECKSUM_OFFSET, checksum)
            struct.pack_into(
                "<Q",
                self.shm.buf,
                PERCEPTION_OBS_SHM_AUTH_FIELD_OFFSET + 32,
                normalized_episode_generation,
            )
            _full_memory_fence()
            _atomic_store_sequence(self.shm.buf, even_sequence)
            self._sequence = even_sequence
            self._last_publish_monotonic_ns = publish_monotonic_ns

    def invalidate(self) -> None:
        """Atomically make the current frame unreadable until the next publish.

        Physical episode reset calls this before a new reset frame exists. The
        writer's private sequence remains monotonic, so the next frame cannot
        alias a subscriber-history sequence from the previous episode.
        """

        with self._publish_lock:
            if self.shm is None or self.array is None:
                return
            if self._sequence >= (1 << 64) - 2:
                raise RuntimeError("perception_obs shared-memory sequence exhausted")
            _atomic_store_sequence(self.shm.buf, self._sequence + 1)
            _full_memory_fence()
            struct.pack_into(
                "<QqQ",
                self.shm.buf,
                PERCEPTION_OBS_SHM_PUBLISH_MONOTONIC_NS_OFFSET,
                0,
                -1,
                int(self.array.nbytes),
            )
            struct.pack_into("<Q", self.shm.buf, PERCEPTION_OBS_SHM_CHECKSUM_OFFSET, 0)
            struct.pack_into(
                "<Q",
                self.shm.buf,
                PERCEPTION_OBS_SHM_AUTH_FIELD_OFFSET + 32,
                0,
            )
            _full_memory_fence()
            _atomic_store_sequence(self.shm.buf, 0)
            self._last_publish_monotonic_ns = 0

    def close(self, *, unlink: bool = False) -> None:
        with self._publish_lock:
            shm = self.shm
            owned = self._owns_segment
            self.shm = None
            self.array = None
            self.dim = None
            self.generation = None
            self._owns_segment = False
            self._sequence = 0
            self._contract_sha256 = None
            self._carries_episode_generation = None
            self._last_publish_monotonic_ns = 0
            if shm is not None:
                if unlink and owned:
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        pass
                shm.close()
