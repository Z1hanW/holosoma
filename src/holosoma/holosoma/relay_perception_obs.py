"""Relay externally rendered perception_obs into a split sim2sim policy.

This is meant for debugging MuJoCo policy rollouts with perception produced by
another process, for example Isaac Sim.  The relay reads an already-preprocessed
flat `perception_obs` vector from a source ZMQ publisher, source shared-memory
buffer, or a .npy/.npz dump, then republishes it to the destination channel that
the MuJoCo-side policy subscribes to.

The policy input is expected to be the model-ready observation, typically the
58x87 normalized D435i depth image flattened to 5046 float32 values.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import resource_tracker
from multiprocessing import shared_memory
from pathlib import Path
from typing import Iterator

import numpy as np
import zmq
from loguru import logger


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _open_existing_shm(name: str, expected_dim: int) -> tuple[shared_memory.SharedMemory, np.ndarray] | None:
    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return None
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass
    expected_bytes = int(expected_dim) * np.dtype(np.float32).itemsize
    if len(shm.buf) < expected_bytes:
        logger.warning(
            "Source shared memory '{}' is too small: {} bytes < {} bytes",
            name,
            len(shm.buf),
            expected_bytes,
        )
        shm.close()
        return None
    array = np.ndarray((int(expected_dim),), dtype=np.float32, buffer=shm.buf)
    return shm, array


class Source:
    def next_obs(self) -> np.ndarray | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class ShmSource(Source):
    def __init__(self, name: str, expected_dim: int) -> None:
        self.name = str(name)
        self.expected_dim = int(expected_dim)
        self.shm: shared_memory.SharedMemory | None = None
        self.array: np.ndarray | None = None

    def _ensure_attached(self) -> bool:
        if self.shm is not None and self.array is not None:
            return True
        opened = _open_existing_shm(self.name, self.expected_dim)
        if opened is None:
            return False
        self.shm, self.array = opened
        logger.info("Attached source perception_obs shm: name={} values={}", self.name, self.expected_dim)
        return True

    def next_obs(self) -> np.ndarray | None:
        if not self._ensure_attached() or self.array is None:
            return None
        return self.array.copy().reshape(-1).astype(np.float32, copy=False)

    def close(self) -> None:
        if self.shm is not None:
            self.shm.close()
        self.shm = None
        self.array = None


class ZmqSource(Source):
    def __init__(self, port: int, expected_dim: int, key: str) -> None:
        self.port = int(port)
        self.expected_dim = int(expected_dim)
        self.key = str(key)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.connect(f"tcp://localhost:{self.port}")
        self.last_obs: np.ndarray | None = None
        logger.info("Subscribed source perception_obs ZMQ: tcp://localhost:{}", self.port)

    def next_obs(self) -> np.ndarray | None:
        while True:
            try:
                payload = json.loads(self.socket.recv_string(zmq.NOBLOCK))
            except zmq.Again:
                break
            except Exception as exc:
                logger.warning("Ignored invalid source ZMQ payload: {}", exc)
                continue
            values = payload.get(self.key) if isinstance(payload, dict) else None
            if values is None:
                logger.warning("Source ZMQ payload missing key '{}'", self.key)
                continue
            obs = np.asarray(values, dtype=np.float32).reshape(-1)
            if obs.size != self.expected_dim:
                logger.warning(
                    "Source ZMQ perception_obs dim mismatch: got {}, expected {}",
                    obs.size,
                    self.expected_dim,
                )
                continue
            self.last_obs = obs
        return None if self.last_obs is None else self.last_obs.copy()

    def close(self) -> None:
        self.socket.close(0)
        self.context.term()


class FileSource(Source):
    def __init__(self, path: Path, expected_dim: int, key: str, loop: bool) -> None:
        self.frames = list(self._load_frames(path, expected_dim=expected_dim, key=key))
        if not self.frames:
            raise RuntimeError(f"No perception_obs frames loaded from {path}")
        self.loop = bool(loop)
        self.index = 0
        logger.info("Loaded {} perception_obs frame(s) from {}", len(self.frames), path)

    @staticmethod
    def _load_array(path: Path, key: str) -> np.ndarray:
        if path.suffix.lower() == ".npz":
            with np.load(path) as data:
                if key in data.files:
                    return np.asarray(data[key], dtype=np.float32)
                if len(data.files) == 1:
                    return np.asarray(data[data.files[0]], dtype=np.float32)
                raise KeyError(f"{path} does not contain key '{key}'. Available keys: {data.files}")
        return np.asarray(np.load(path), dtype=np.float32)

    @classmethod
    def _load_frames(cls, path: Path, *, expected_dim: int, key: str) -> Iterator[np.ndarray]:
        paths: list[Path]
        if path.is_dir():
            paths = sorted([*path.glob("*.npy"), *path.glob("*.npz")])
        else:
            paths = [path]
        for item in paths:
            array = cls._load_array(item, key)
            if array.ndim <= 2:
                frames = array.reshape(1, -1)
            else:
                frames = array.reshape(array.shape[0], -1)
            for frame in frames:
                obs = np.asarray(frame, dtype=np.float32).reshape(-1)
                if obs.size != int(expected_dim):
                    raise ValueError(f"{item} frame has {obs.size} values, expected {expected_dim}")
                yield obs

    def next_obs(self) -> np.ndarray | None:
        if self.index >= len(self.frames):
            if not self.loop:
                return self.frames[-1].copy()
            self.index = 0
        obs = self.frames[self.index]
        self.index += 1
        return obs.copy()


class Publisher:
    def __init__(
        self,
        *,
        dest_port: int | None,
        dest_shm_name: str | None,
        expected_dim: int,
        initial_value: float,
    ) -> None:
        self.expected_dim = int(expected_dim)
        self.initial_value = float(initial_value)
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.dest_port = int(dest_port) if dest_port else None
        if self.dest_port:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.bind(f"tcp://*:{self.dest_port}")
            logger.info("Publishing destination perception_obs ZMQ on port {}", self.dest_port)

        self.dest_shm_name = str(dest_shm_name) if dest_shm_name else None
        self.shm: shared_memory.SharedMemory | None = None
        self.array: np.ndarray | None = None
        if self.dest_shm_name:
            self._ensure_dest_shm()

    def _ensure_dest_shm(self) -> None:
        if self.dest_shm_name is None:
            return
        if self.shm is not None and self.array is not None:
            return
        size = self.expected_dim * np.dtype(np.float32).itemsize
        try:
            self.shm = shared_memory.SharedMemory(name=self.dest_shm_name, create=True, size=size)
            logger.info("Created destination perception_obs shm: name={} values={}", self.dest_shm_name, self.expected_dim)
        except FileExistsError:
            existing = shared_memory.SharedMemory(name=self.dest_shm_name, create=False)
            if len(existing.buf) != size:
                existing.close()
                stale = shared_memory.SharedMemory(name=self.dest_shm_name, create=False)
                stale.unlink()
                stale.close()
                self.shm = shared_memory.SharedMemory(name=self.dest_shm_name, create=True, size=size)
                logger.info(
                    "Recreated destination perception_obs shm: name={} values={}",
                    self.dest_shm_name,
                    self.expected_dim,
                )
            else:
                self.shm = existing
                logger.info(
                    "Connected destination perception_obs shm: name={} values={}",
                    self.dest_shm_name,
                    self.expected_dim,
                )
        self.array = np.ndarray((self.expected_dim,), dtype=np.float32, buffer=self.shm.buf)
        self.array.fill(np.float32(self.initial_value))

    def publish(self, obs: np.ndarray, *, frame_idx: int) -> None:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs.size != self.expected_dim:
            raise ValueError(f"Destination perception_obs dim mismatch: got {obs.size}, expected {self.expected_dim}")
        if self.array is not None:
            self.array[:] = obs
        if self.socket is not None:
            payload = {
                "source": "relay_perception_obs",
                "frame_idx": int(frame_idx),
                "time": time.time(),
                "perception_obs": obs.tolist(),
            }
            try:
                self.socket.send_string(json.dumps(payload), zmq.NOBLOCK)
            except zmq.Again:
                pass

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(0)
        if self.context is not None:
            self.context.term()
        if self.shm is not None:
            self.shm.close()
        self.socket = None
        self.context = None
        self.shm = None
        self.array = None


def _build_source(args: argparse.Namespace) -> Source:
    sources = [bool(args.source_shm_name), bool(args.source_port), bool(args.source_file)]
    if sum(sources) != 1:
        raise SystemExit("Specify exactly one source: --source-shm-name, --source-port, or --source-file.")
    if args.source_shm_name:
        return ShmSource(args.source_shm_name, args.expected_dim)
    if args.source_port:
        return ZmqSource(args.source_port, args.expected_dim, args.source_key)
    return FileSource(Path(args.source_file).expanduser(), args.expected_dim, args.source_key, args.loop_file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-dim", type=_positive_int, default=58 * 87)
    parser.add_argument("--source-shm-name", default="")
    parser.add_argument("--source-port", type=int, default=0)
    parser.add_argument("--source-file", default="")
    parser.add_argument("--source-key", default="perception_obs")
    parser.add_argument("--dest-shm-name", default="depth_img_shm")
    parser.add_argument("--dest-port", type=int, default=0)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--loop-file", action="store_true")
    parser.add_argument("--require-fresh", action="store_true", help="Skip publishing repeated stale source frames.")
    parser.add_argument(
        "--initial-value",
        type=float,
        default=0.5,
        help="Initial value written to destination shm before the first source frame. For normalized D435 depth, 0.5 is far/empty.",
    )
    parser.add_argument("--stats-every", type=float, default=2.0)
    args = parser.parse_args()

    source = _build_source(args)
    publisher = Publisher(
        dest_port=args.dest_port if args.dest_port > 0 else None,
        dest_shm_name=args.dest_shm_name or None,
        expected_dim=args.expected_dim,
        initial_value=args.initial_value,
    )
    period = 1.0 / max(float(args.rate_hz), 1e-6)
    last_stats = time.monotonic()
    last_bytes: bytes | None = None
    frame_idx = 0
    published = 0

    try:
        while True:
            obs = source.next_obs()
            if obs is None:
                time.sleep(min(period, 0.05))
                continue
            obs = np.nan_to_num(obs.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
            obs_bytes = obs.tobytes()
            if args.require_fresh and obs_bytes == last_bytes:
                time.sleep(period)
                continue
            last_bytes = obs_bytes
            publisher.publish(obs, frame_idx=frame_idx)
            frame_idx += 1
            published += 1
            now = time.monotonic()
            if now - last_stats >= max(float(args.stats_every), 0.1):
                finite = obs[np.isfinite(obs)]
                if finite.size:
                    logger.info(
                        "Relayed {} frame(s); last dim={} min={:.4f} max={:.4f} mean={:.4f}",
                        published,
                        obs.size,
                        float(finite.min()),
                        float(finite.max()),
                        float(finite.mean()),
                    )
                else:
                    logger.info("Relayed {} frame(s); last dim={} all non-finite", published, obs.size)
                last_stats = now
            time.sleep(period)
    except KeyboardInterrupt:
        logger.info("Stopping perception_obs relay")
    finally:
        source.close()
        publisher.close()


if __name__ == "__main__":
    main()
