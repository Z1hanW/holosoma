"""Helpers for split sim2sim perception observation subscription."""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import mmap
import os
import struct
import time
from collections import deque
from multiprocessing import resource_tracker
from multiprocessing import shared_memory

import numpy as np
import zmq
from loguru import logger


# Shared-memory protocol v1.  Keep these values byte-for-byte compatible with
# holosoma.utils.perception_obs; inference intentionally has no dependency on
# the training/simulator package.
PERCEPTION_OBS_SHM_MAGIC = b"HSPOSHM\0"
PERCEPTION_OBS_SHM_VERSION = 1
PERCEPTION_OBS_SHM_HEADER_SIZE = 128
PERCEPTION_OBS_SHM_DTYPE_FLOAT32_LE = 1
PERCEPTION_OBS_SHM_FLAG_CONTRACT_SHA256 = 1
PERCEPTION_OBS_SHM_FLAG_EPISODE_GENERATION = 2
PERCEPTION_OBS_SHM_VALID_FLAGS = (
    PERCEPTION_OBS_SHM_FLAG_CONTRACT_SHA256 | PERCEPTION_OBS_SHM_FLAG_EPISODE_GENERATION
)
PERCEPTION_OBS_SHM_HEADER = struct.Struct("<8sIIIIQQQQqQQQ40s")
PERCEPTION_OBS_SHM_SEQUENCE_OFFSET = 40
_ATOMIC_ACQUIRE = 2
_ATOMIC_SEQ_CST = 5

if PERCEPTION_OBS_SHM_HEADER.size != PERCEPTION_OBS_SHM_HEADER_SIZE:  # pragma: no cover - import invariant
    raise RuntimeError("Invalid perception_obs shared-memory header size")

try:  # libatomic is part of the GCC runtime on supported Linux targets.
    _libatomic_path = ctypes.util.find_library("atomic")
    if not _libatomic_path:
        raise OSError("libatomic not found")
    _libatomic = ctypes.CDLL(_libatomic_path)
    _atomic_load_8 = _libatomic.__atomic_load_8
    _atomic_load_8.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _atomic_load_8.restype = ctypes.c_uint64
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
    _atomic_load_8 = None
    _atomic_exchange_8 = None
    _atomic_thread_fence = None

_fence_word = ctypes.c_uint64(0)


def _require_atomic_runtime() -> None:
    if _atomic_load_8 is None or (_atomic_thread_fence is None and _atomic_exchange_8 is None):
        raise RuntimeError(
            "Protocol-v1 perception_obs shared memory requires libatomic for cross-process memory ordering"
        )


def _atomic_load_sequence(buffer) -> int:
    _require_atomic_runtime()
    sequence_view = np.ndarray(
        (1,),
        dtype=np.dtype("<u8"),
        buffer=buffer,
        offset=PERCEPTION_OBS_SHM_SEQUENCE_OFFSET,
    )
    return int(_atomic_load_8(sequence_view.ctypes.data, _ATOMIC_ACQUIRE))


def _full_memory_fence() -> None:
    _require_atomic_runtime()
    if _atomic_thread_fence is not None:
        _atomic_thread_fence(_ATOMIC_SEQ_CST)
    else:  # pragma: no cover - compatibility with older libatomic
        _atomic_exchange_8(ctypes.addressof(_fence_word), 0, _ATOMIC_SEQ_CST)


def _payload_checksum(payload: bytes | memoryview) -> int:
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


def _normalize_expected_episode_generation(value: int | None) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        or int(value) > (1 << 63) - 1
    ):
        raise ValueError(
            "expected_episode_generation must be a non-negative integer within the protocol range, "
            f"got {value!r}"
        )
    return int(value)


class _ReadOnlySegment:
    """Read-only named-shm mapping that never registers consumer ownership.

    `multiprocessing.SharedMemory(create=False)` is registered as though the
    consumer owned the segment on Python <=3.12.  Its resource_tracker may then
    unlink the publisher's name when inference exits.  On POSIX, opening the
    same shm object directly and mmap'ing it read-only avoids that ownership
    bug.  The fallback unregisters immediately for non-POSIX runtimes.
    """

    def __init__(
        self,
        *,
        name: str,
        size: int,
        identity: tuple[int, int] | tuple[str, int],
        mapping: mmap.mmap | None = None,
        shm: shared_memory.SharedMemory | None = None,
    ) -> None:
        self.name = name
        self.size = int(size)
        self.identity = identity
        self._mapping = mapping
        self._shm = shm

    @property
    def buf(self):
        if self._mapping is not None:
            return self._mapping
        if self._shm is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("shared-memory segment is closed")
        return self._shm.buf

    def close(self) -> None:
        mapping = self._mapping
        shm = self._shm
        self._mapping = None
        self._shm = None
        if mapping is not None:
            mapping.close()
        if shm is not None:
            shm.close()


def _open_read_only_segment(name: str) -> _ReadOnlySegment | None:
    if bool(getattr(shared_memory, "_USE_POSIX", False)):
        internal_name = name if name.startswith("/") else f"/{name}"
        fd: int | None = None
        try:
            fd = shared_memory._posixshmem.shm_open(internal_name, os.O_RDONLY, mode=0o600)
            stat = os.fstat(fd)
            if stat.st_size <= 0:
                return None
            mapping = mmap.mmap(fd, stat.st_size, access=mmap.ACCESS_READ)
            return _ReadOnlySegment(
                name=name,
                size=stat.st_size,
                identity=(int(stat.st_dev), int(stat.st_ino)),
                mapping=mapping,
            )
        except FileNotFoundError:
            return None
        finally:
            if fd is not None:
                os.close(fd)

    try:  # pragma: no cover - split sim2sim currently runs on POSIX
        shm = shared_memory.SharedMemory(name=name, create=False)
    except FileNotFoundError:  # pragma: no cover - non-POSIX fallback
        return None
    try:  # pragma: no cover - non-POSIX fallback
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass
    return _ReadOnlySegment(
        name=name,
        size=len(shm.buf),
        identity=(str(shm.name), len(shm.buf)),
        shm=shm,
    )


class PerceptionObsSub:
    """Subscribe to simulator perception observations published by split sim2sim."""

    def __init__(self, port: int = 5558) -> None:
        self.port = int(port)
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.last_payload: dict | None = None
        self.payload_buffer: deque[dict] = deque(maxlen=512)

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
                if isinstance(self.last_payload, dict):
                    self.payload_buffer.append(self.last_payload)
            except zmq.Again:
                break

    def get_payload(self) -> dict | None:
        self._drain_messages()
        return self.last_payload

    def get_payload_at_or_before(
        self,
        sim_time_ms: float | int | None,
        *,
        expected_episode_generation: int | None = None,
    ) -> dict | None:
        self._drain_messages()
        target_ms: float | None = None
        if sim_time_ms is not None:
            try:
                target_ms = float(sim_time_ms)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(target_ms) or target_ms < 0.0:
                return None
        if expected_episode_generation is not None:
            if (
                isinstance(expected_episode_generation, bool)
                or not isinstance(expected_episode_generation, (int, np.integer))
                or int(expected_episode_generation) < 0
            ):
                return None
            expected_episode_generation = int(expected_episode_generation)
        selected: dict | None = None
        for payload in reversed(self.payload_buffer):
            if expected_episode_generation is not None:
                payload_generation = payload.get("episode_generation")
                if (
                    isinstance(payload_generation, bool)
                    or not isinstance(payload_generation, int)
                    or payload_generation != expected_episode_generation
                ):
                    continue
            if target_ms is not None:
                try:
                    payload_ms = float(payload.get("sim_time_ms"))
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(payload_ms) or payload_ms < 0.0 or payload_ms > target_ms:
                    continue
            selected = payload
            break
        # Returning the latest payload when every buffered frame is from the
        # future silently leaks simulator state across the control boundary.
        return selected

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
    """Read only complete, finite, fresh protocol-v1 perception frames."""

    def __init__(self, name: str = "depth_img_shm", max_age_s: float | None = None) -> None:
        _require_atomic_runtime()
        self.name = str(name)
        if not self.name:
            raise ValueError("perception_obs shared-memory name must not be empty")
        if max_age_s is None:
            max_age_s = float(os.environ.get("HOLOSOMA_PERCEPTION_OBS_SHM_MAX_AGE_S", "1.0"))
        if not np.isfinite(max_age_s) or float(max_age_s) <= 0.0:
            raise ValueError(f"perception_obs shared-memory max_age_s must be finite and positive, got {max_age_s!r}")
        retries_raw = os.environ.get("HOLOSOMA_PERCEPTION_OBS_SHM_READ_RETRIES", "5")
        try:
            self._read_retries = max(1, int(retries_raw))
        except ValueError as exc:
            raise ValueError(
                f"HOLOSOMA_PERCEPTION_OBS_SHM_READ_RETRIES must be an integer, got {retries_raw!r}"
            ) from exc
        self.max_age_s = float(max_age_s)
        self._max_age_ns = int(round(self.max_age_s * 1_000_000_000.0))
        self.shm: _ReadOnlySegment | None = None
        self.array: np.ndarray | None = None
        self.dim: int | None = None
        self.generation: int | None = None
        self.last_sequence: int | None = None
        self.last_publish_monotonic_ns: int | None = None
        self.last_sim_time_ms: float | None = None
        self.last_episode_generation: int | None = None
        self._frame_history: deque[
            tuple[int, int, float | None, int | None, np.ndarray]
        ] = deque(maxlen=512)
        self._warned: set[tuple[str, int | None]] = set()

    def start(self) -> None:
        logger.info(
            "Perception obs shared-memory subscriber configured: name={} protocol=v{} max_age_s={}",
            self.name,
            PERCEPTION_OBS_SHM_VERSION,
            self.max_age_s,
        )

    def _warn_once(self, reason: str, message: str, *args, generation: int | None = None) -> None:
        key = (reason, generation)
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(message, *args)

    def _validate_segment(
        self,
        segment: _ReadOnlySegment,
        expected_dim: int,
        expected_contract_sha256: str | None,
        expected_episode_generation: int | None,
    ) -> int | None:
        if segment.size < PERCEPTION_OBS_SHM_HEADER_SIZE:
            self._warn_once(
                "legacy",
                "Rejected legacy/truncated perception obs shared memory '{}': {} bytes < protocol header {} bytes",
                self.name,
                segment.size,
                PERCEPTION_OBS_SHM_HEADER_SIZE,
            )
            return None
        try:
            header = PERCEPTION_OBS_SHM_HEADER.unpack_from(segment.buf, 0)
        except (ValueError, struct.error):
            self._warn_once("header", "Rejected malformed perception obs shared-memory header '{}'", self.name)
            return None

        magic, version, header_size, dtype_code, flags = header[:5]
        dim, generation = int(header[5]), int(header[6])
        payload_bytes, publisher_pid = int(header[10]), int(header[12])
        contract_field = bytes(header[13])
        if magic != PERCEPTION_OBS_SHM_MAGIC:
            self._warn_once("magic", "Rejected legacy/unknown perception obs shared memory '{}'", self.name)
            return None
        if version != PERCEPTION_OBS_SHM_VERSION:
            self._warn_once(
                "version",
                "Rejected perception obs shared-memory version mismatch '{}': got {}, expected {}",
                self.name,
                version,
                PERCEPTION_OBS_SHM_VERSION,
            )
            return None
        if header_size != PERCEPTION_OBS_SHM_HEADER_SIZE or dtype_code != PERCEPTION_OBS_SHM_DTYPE_FLOAT32_LE:
            self._warn_once("layout", "Rejected incompatible perception obs shared-memory layout '{}'", self.name)
            return None
        if flags & ~PERCEPTION_OBS_SHM_VALID_FLAGS or generation == 0 or publisher_pid <= 0:
            self._warn_once("identity", "Rejected invalid perception obs shared-memory identity '{}'", self.name)
            return None
        has_contract = bool(flags & PERCEPTION_OBS_SHM_FLAG_CONTRACT_SHA256)
        has_episode_generation = bool(flags & PERCEPTION_OBS_SHM_FLAG_EPISODE_GENERATION)
        if has_contract:
            published_contract = contract_field[:32]
        else:
            if any(contract_field[:32]):
                self._warn_once("contract_layout", "Rejected unauthenticated perception contract bytes '{}'", self.name)
                return None
            published_contract = b""
        if not has_episode_generation and any(contract_field[32:]):
            self._warn_once(
                "episode_layout",
                "Rejected unauthenticated perception episode-generation bytes '{}'",
                self.name,
            )
            return None
        if expected_episode_generation is not None and not has_episode_generation:
            self._warn_once(
                "episode_missing",
                "Rejected perception obs shared memory without episode authentication '{}' generation={}",
                self.name,
                generation,
                generation=generation,
            )
            return None
        expected_contract = _contract_digest_bytes(expected_contract_sha256)
        if expected_contract and published_contract != expected_contract:
            self._warn_once(
                "contract",
                "Rejected perception obs shared-memory contract mismatch '{}' generation={}",
                self.name,
                generation,
                generation=generation,
            )
            return None
        if dim != int(expected_dim):
            self._warn_once(
                "dim",
                "Rejected perception obs shared-memory dimension mismatch '{}': got {}, expected {}",
                self.name,
                dim,
                expected_dim,
                generation=generation,
            )
            return None
        expected_payload_bytes = int(expected_dim) * np.dtype("<f4").itemsize
        expected_size = PERCEPTION_OBS_SHM_HEADER_SIZE + expected_payload_bytes
        if payload_bytes != expected_payload_bytes or segment.size != expected_size:
            self._warn_once(
                "size",
                "Rejected perception obs shared-memory size mismatch '{}': header payload={} total={}, expected payload={} total={}",
                self.name,
                payload_bytes,
                segment.size,
                expected_payload_bytes,
                expected_size,
                generation=generation,
            )
            return None
        return generation

    def _open_validated(
        self,
        expected_dim: int,
        expected_contract_sha256: str | None,
        expected_episode_generation: int | None,
    ) -> tuple[_ReadOnlySegment, int] | None:
        segment = _open_read_only_segment(self.name)
        if segment is None:
            return None
        generation = self._validate_segment(
            segment,
            expected_dim,
            expected_contract_sha256,
            expected_episode_generation,
        )
        if generation is None:
            segment.close()
            return None
        return segment, generation

    def _adopt(self, segment: _ReadOnlySegment, generation: int, expected_dim: int) -> None:
        old = self.shm
        self.array = None
        self.shm = segment
        self.dim = int(expected_dim)
        self.generation = int(generation)
        self._frame_history.clear()
        self.array = np.ndarray(
            (self.dim,),
            dtype=np.dtype("<f4"),
            buffer=segment.buf,
            offset=PERCEPTION_OBS_SHM_HEADER_SIZE,
        )
        if old is not None:
            old.close()
        logger.info(
            "Attached perception obs shared memory protocol v{}: name={} values={} generation={}",
            PERCEPTION_OBS_SHM_VERSION,
            self.name,
            self.dim,
            self.generation,
        )

    def _ensure_attached(
        self,
        expected_dim: int,
        expected_contract_sha256: str | None,
        expected_episode_generation: int | None,
    ) -> bool:
        opened = self._open_validated(
            expected_dim,
            expected_contract_sha256,
            expected_episode_generation,
        )
        if opened is None:
            # Once the publisher unlinks the name, the old mmap can remain
            # readable indefinitely.  Drop it immediately instead of serving
            # an apparently fresh frame from an orphaned mapping.
            self.close()
            return False
        segment, generation = opened
        if (
            self.shm is not None
            and self.dim == int(expected_dim)
            and self.generation == generation
            and self.shm.identity == segment.identity
        ):
            segment.close()
            return True
        self._adopt(segment, generation, expected_dim)
        return True

    def _read_consistent(
        self,
        expected_dim: int,
        expected_contract_sha256: str | None,
        expected_episode_generation: int | None,
    ) -> np.ndarray | None:
        segment = self.shm
        generation = self.generation
        if segment is None or generation is None:
            return None
        for _ in range(self._read_retries):
            sequence_before = _atomic_load_sequence(segment.buf)
            if sequence_before == 0 or sequence_before & 1:
                continue
            try:
                header = PERCEPTION_OBS_SHM_HEADER.unpack_from(segment.buf, 0)
            except (ValueError, struct.error):
                continue
            expected_payload_bytes = int(expected_dim) * np.dtype("<f4").itemsize
            expected_contract = _contract_digest_bytes(expected_contract_sha256)
            flags = int(header[4])
            contract_field = bytes(header[13])
            has_contract = bool(flags & PERCEPTION_OBS_SHM_FLAG_CONTRACT_SHA256)
            has_episode_generation = bool(flags & PERCEPTION_OBS_SHM_FLAG_EPISODE_GENERATION)
            published_contract = (
                contract_field[:32]
                if has_contract
                else b""
            )
            published_episode_generation = (
                int(struct.unpack("<Q", contract_field[32:])[0])
                if has_episode_generation
                else None
            )
            if (
                header[0] != PERCEPTION_OBS_SHM_MAGIC
                or int(header[1]) != PERCEPTION_OBS_SHM_VERSION
                or int(header[2]) != PERCEPTION_OBS_SHM_HEADER_SIZE
                or int(header[3]) != PERCEPTION_OBS_SHM_DTYPE_FLOAT32_LE
                or flags & ~PERCEPTION_OBS_SHM_VALID_FLAGS
                or int(header[5]) != int(expected_dim)
                or int(header[6]) != generation
                or int(header[7]) != sequence_before
                or int(header[10]) != expected_payload_bytes
                or int(header[12]) <= 0
                or (not has_contract and any(contract_field[:32]))
                or (not has_episode_generation and any(contract_field[32:]))
                or (
                    published_episode_generation is not None
                    and published_episode_generation > (1 << 63) - 1
                )
                or (expected_contract and published_contract != expected_contract)
                or (
                    expected_episode_generation is not None
                    and published_episode_generation != expected_episode_generation
                )
            ):
                self._warn_once(
                    "header_changed",
                    "Rejected inconsistent perception obs shared-memory header '{}' generation={}",
                    self.name,
                    generation,
                    generation=generation,
                )
                continue
            publish_monotonic_ns = int(header[8])
            sim_time_ns = int(header[9])
            checksum = int(header[11])
            payload = bytes(segment.buf[PERCEPTION_OBS_SHM_HEADER_SIZE : segment.size])
            _full_memory_fence()
            sequence_after = _atomic_load_sequence(segment.buf)
            if sequence_before != sequence_after or sequence_after & 1:
                continue
            if _payload_checksum(payload) != checksum:
                self._warn_once(
                    "checksum",
                    "Rejected torn/corrupt perception obs shared-memory frame '{}' generation={} sequence={}",
                    self.name,
                    generation,
                    sequence_after,
                    generation=generation,
                )
                return None
            obs = np.frombuffer(payload, dtype=np.dtype("<f4"), count=int(expected_dim)).copy()
            if obs.size != int(expected_dim) or not np.isfinite(obs).all():
                self._warn_once(
                    "nonfinite",
                    "Rejected non-finite perception obs shared-memory frame '{}' generation={} sequence={}",
                    self.name,
                    generation,
                    sequence_after,
                    generation=generation,
                )
                return None
            now_ns = time.monotonic_ns()
            age_ns = now_ns - publish_monotonic_ns
            if publish_monotonic_ns <= 0 or age_ns < 0 or age_ns > self._max_age_ns:
                self._warn_once(
                    "stale",
                    "Rejected stale perception obs shared-memory frame '{}' generation={} sequence={} age_s={:.6f} max_age_s={:.6f}",
                    self.name,
                    generation,
                    sequence_after,
                    age_ns / 1_000_000_000.0,
                    self.max_age_s,
                    generation=generation,
                )
                return None
            if sim_time_ns < -1:
                self._warn_once(
                    "sim_time",
                    "Rejected invalid perception obs sim_time in '{}' generation={} sequence={}",
                    self.name,
                    generation,
                    sequence_after,
                    generation=generation,
                )
                return None
            self.last_sequence = int(sequence_after)
            self.last_publish_monotonic_ns = publish_monotonic_ns
            self.last_sim_time_ms = None if sim_time_ns < 0 else sim_time_ns / 1_000_000.0
            self.last_episode_generation = published_episode_generation
            if not self._frame_history or self._frame_history[-1][0] != int(sequence_after):
                self._frame_history.append(
                    (
                        int(sequence_after),
                        publish_monotonic_ns,
                        self.last_sim_time_ms,
                        published_episode_generation,
                        obs.copy(),
                    )
                )
            return obs.reshape(1, int(expected_dim)).astype(np.float32, copy=False)
        return None

    def get_obs(
        self,
        expected_dim: int,
        expected_contract_sha256: str | None = None,
        expected_episode_generation: int | None = None,
    ) -> np.ndarray | None:
        expected_dim = int(expected_dim)
        if expected_dim <= 0:
            raise ValueError(f"expected_dim must be positive, got {expected_dim}")
        expected_episode_generation = _normalize_expected_episode_generation(
            expected_episode_generation
        )
        # A second name lookup after the copy establishes that the frame came
        # from the currently named generation, not an orphaned mapping that was
        # unlinked/replaced during the read.
        for _ in range(2):
            if not self._ensure_attached(
                expected_dim,
                expected_contract_sha256,
                expected_episode_generation,
            ):
                return None
            obs = self._read_consistent(
                expected_dim,
                expected_contract_sha256,
                expected_episode_generation,
            )
            if obs is None or self.shm is None or self.generation is None:
                return None
            confirmed = self._open_validated(
                expected_dim,
                expected_contract_sha256,
                expected_episode_generation,
            )
            if confirmed is None:
                self.close()
                return None
            segment, generation = confirmed
            if segment.identity == self.shm.identity and generation == self.generation:
                segment.close()
                return obs
            self._adopt(segment, generation, expected_dim)
        return None

    def get_obs_at_or_before(
        self,
        expected_dim: int,
        target_sim_time_ms: float | int,
        expected_contract_sha256: str | None = None,
        expected_episode_generation: int | None = None,
    ) -> np.ndarray | None:
        """Return the freshest cached frame that does not lead simulator state.

        The named segment contains only the newest publisher frame, so the
        subscriber retains validated frames locally.  This mirrors the ZMQ
        subscriber's temporal buffer and prevents a newly published t+1 depth
        frame from being paired with robot state from timestep t.
        """

        try:
            target_ms = float(target_sim_time_ms)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(target_ms) or target_ms < 0.0:
            return None
        expected_episode_generation = _normalize_expected_episode_generation(
            expected_episode_generation
        )
        # Cache the newest authenticated frame even if it belongs to a
        # different episode; selection below is exact on both episode and
        # simulator time. This preserves an already-buffered old-episode frame
        # while still making it impossible to substitute it for a new episode.
        if self.get_obs(int(expected_dim), expected_contract_sha256) is None:
            return None

        now_ns = time.monotonic_ns()
        for _, publish_ns, sim_time_ms, episode_generation, obs in reversed(self._frame_history):
            if (
                expected_episode_generation is not None
                and episode_generation != expected_episode_generation
            ):
                continue
            if sim_time_ms is None or sim_time_ms > target_ms:
                continue
            age_ns = now_ns - publish_ns
            if age_ns < 0 or age_ns > self._max_age_ns:
                continue
            return obs.reshape(1, int(expected_dim)).astype(np.float32, copy=True)
        return None

    def close(self) -> None:
        shm = self.shm
        self.shm = None
        self.array = None
        self.dim = None
        self.generation = None
        self.last_sequence = None
        self.last_publish_monotonic_ns = None
        self.last_sim_time_ms = None
        self.last_episode_generation = None
        self._frame_history.clear()
        if shm is not None:
            shm.close()
