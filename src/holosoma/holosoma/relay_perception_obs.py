"""Relay externally rendered perception_obs into a split sim2sim policy.

This is meant for debugging MuJoCo policy rollouts with perception produced by
another process, for example Isaac Sim.  The relay reads an already-preprocessed
flat `perception_obs` vector from a source ZMQ publisher, source shared-memory
buffer, or a .npy/.npz dump, then republishes it to the destination channel that
the MuJoCo-side policy subscribes to.

The policy input is expected to be the model-ready observation, typically the
58x87 normalized D435i depth image flattened to 5046 float32 values.

Every source frame must also carry the episode_generation of the destination
run_sim state it was rendered from. An unrelated renderer-local counter is not
interchangeable; mismatched frames are intentionally rejected by the policy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import zmq
from loguru import logger

from holosoma.utils.perception_obs import PerceptionObsShmPub


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0 or parsed > (1 << 63) - 1:
        raise argparse.ArgumentTypeError("must be between 0 and 2^63-1")
    return parsed


def _frame_identity(
    obs: np.ndarray,
    *,
    sim_time_ms: float | None,
    episode_generation: int | None,
) -> tuple[bytes, float | None, int | None]:
    """Identity used by --require-fresh, including reset/session boundaries."""

    return (obs.tobytes(), sim_time_ms, episode_generation)


def _contract_sha256(value: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64:
        raise argparse.ArgumentTypeError("must be exactly 64 hexadecimal characters")
    try:
        digest = bytes.fromhex(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be exactly 64 hexadecimal characters") from exc
    if len(digest) != 32:
        raise argparse.ArgumentTypeError("must encode exactly 32 bytes")
    return normalized


class Source:
    last_sim_time_ms: float | None = None
    last_episode_generation: int | None = None
    contract_sha256: str

    def next_obs(self) -> np.ndarray | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class ShmSource(Source):
    def __init__(self, name: str, expected_dim: int, contract_sha256: str) -> None:
        self.name = str(name)
        self.expected_dim = int(expected_dim)
        self.contract_sha256 = _contract_sha256(contract_sha256)
        self.sub = None
        self._last_source_identity: tuple[int, int] | None = None

    def _ensure_attached(self) -> bool:
        if self.sub is not None:
            return True
        try:
            from holosoma_inference.utils.perception_obs import PerceptionObsShmSub
        except ImportError as exc:
            raise RuntimeError(
                "Protocol-v1 shared-memory relay input requires the holosoma_inference package"
            ) from exc
        self.sub = PerceptionObsShmSub(name=self.name)
        self.sub.start()
        logger.info(
            "Configured protocol-v1 source perception_obs shm: name={} values={}",
            self.name,
            self.expected_dim,
        )
        return True

    def next_obs(self) -> np.ndarray | None:
        if not self._ensure_attached() or self.sub is None:
            return None
        obs = self.sub.get_obs(self.expected_dim, self.contract_sha256)
        if obs is None:
            return None
        self.last_sim_time_ms = self.sub.last_sim_time_ms
        self.last_episode_generation = self.sub.last_episode_generation
        if self.last_episode_generation is None:
            logger.warning(
                "Source shared-memory perception frame has no episode identity; frame rejected"
            )
            return None
        source_generation = self.sub.generation
        source_sequence = self.sub.last_sequence
        if source_generation is None or source_sequence is None:
            return None
        source_identity = (int(source_generation), int(source_sequence))
        if source_identity == self._last_source_identity:
            return None
        self._last_source_identity = source_identity
        return np.asarray(obs, dtype=np.float32).reshape(-1).copy()

    def close(self) -> None:
        if self.sub is not None:
            self.sub.close()
        self.sub = None


class ZmqSource(Source):
    def __init__(self, port: int, expected_dim: int, key: str, contract_sha256: str) -> None:
        self.port = int(port)
        self.expected_dim = int(expected_dim)
        self.key = str(key)
        self.contract_sha256 = _contract_sha256(contract_sha256)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.connect(f"tcp://localhost:{self.port}")
        self.last_obs: np.ndarray | None = None
        self.last_sim_time_ms: float | None = None
        self.last_episode_generation: int | None = None
        logger.info("Subscribed source perception_obs ZMQ: tcp://localhost:{}", self.port)

    def next_obs(self) -> np.ndarray | None:
        newest_obs: np.ndarray | None = None
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
            published_contract = payload.get("perception_contract_sha256")
            if published_contract != self.contract_sha256:
                logger.warning(
                    "Source ZMQ perception contract mismatch: got {!r}, expected {!r}; frame rejected",
                    published_contract,
                    self.contract_sha256,
                )
                continue
            obs = np.asarray(values, dtype=np.float32).reshape(-1)
            if obs.size != self.expected_dim:
                logger.warning(
                    "Source ZMQ perception_obs dim mismatch: got {}, expected {}",
                    obs.size,
                    self.expected_dim,
                )
                continue
            if not np.isfinite(obs).all():
                logger.warning("Source ZMQ perception_obs contains NaN or Inf; frame rejected")
                continue
            raw_sim_time_ms = payload.get("sim_time_ms")
            if raw_sim_time_ms is None:
                sim_time_ms = None
            else:
                try:
                    sim_time_ms = float(raw_sim_time_ms)
                except (TypeError, ValueError):
                    logger.warning("Source ZMQ perception_obs has invalid sim_time_ms; frame rejected")
                    continue
                if not np.isfinite(sim_time_ms) or sim_time_ms < 0.0:
                    logger.warning("Source ZMQ perception_obs has invalid sim_time_ms; frame rejected")
                    continue
            episode_generation = payload.get("episode_generation")
            if (
                isinstance(episode_generation, bool)
                or not isinstance(episode_generation, int)
                or episode_generation < 0
                or episode_generation > (1 << 63) - 1
            ):
                logger.warning(
                    "Source ZMQ perception_obs has missing/invalid episode_generation; frame rejected"
                )
                continue
            self.last_obs = obs
            self.last_sim_time_ms = sim_time_ms
            self.last_episode_generation = int(episode_generation)
            newest_obs = obs
        # Never refresh an old source frame's destination wall-clock age merely
        # because the relay loop is still running after the source disconnects.
        return None if newest_obs is None else newest_obs.copy()

    def close(self) -> None:
        self.socket.close(0)
        self.context.term()


class FileSource(Source):
    def __init__(
        self,
        path: Path,
        expected_dim: int,
        key: str,
        loop: bool,
        contract_sha256: str,
        episode_generation: int,
    ) -> None:
        self.contract_sha256 = _contract_sha256(contract_sha256)
        self.last_episode_generation = _nonnegative_int(str(episode_generation))
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
                return None
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
        contract_sha256: str,
    ) -> None:
        self.expected_dim = int(expected_dim)
        self.initial_value = float(initial_value)
        self.contract_sha256 = _contract_sha256(contract_sha256)
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
        self.shm_pub: PerceptionObsShmPub | None = None
        if self.dest_shm_name:
            self.shm_pub = PerceptionObsShmPub(name=self.dest_shm_name)
            self.shm_pub.start()
            logger.info(
                "Publishing destination perception_obs protocol-v1 shm after the first "
                "episode-authenticated source frame: name={} values={}",
                self.dest_shm_name,
                self.expected_dim,
            )

    def publish(
        self,
        obs: np.ndarray,
        *,
        frame_idx: int,
        sim_time_ms: float | int | None = None,
        episode_generation: int,
    ) -> None:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs.size != self.expected_dim:
            raise ValueError(f"Destination perception_obs dim mismatch: got {obs.size}, expected {self.expected_dim}")
        if not np.isfinite(obs).all():
            raise ValueError("Destination perception_obs contains NaN or Inf")
        episode_generation = _nonnegative_int(str(episode_generation))
        if self.shm_pub is not None:
            self.shm_pub.publish(
                obs,
                sim_time_ms=sim_time_ms,
                contract_sha256=self.contract_sha256,
                episode_generation=episode_generation,
            )
        if self.socket is not None:
            payload = {
                "source": "relay_perception_obs",
                "frame_idx": int(frame_idx),
                "time": time.time(),
                "episode_generation": episode_generation,
                "perception_obs": obs.tolist(),
                "perception_contract_sha256": self.contract_sha256,
            }
            if sim_time_ms is not None:
                payload["sim_time_ms"] = float(sim_time_ms)
            try:
                self.socket.send_string(json.dumps(payload), zmq.NOBLOCK)
            except zmq.Again:
                pass

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(0)
        if self.context is not None:
            self.context.term()
        if self.shm_pub is not None:
            self.shm_pub.close(unlink=True)
        self.socket = None
        self.context = None
        self.shm_pub = None


def _build_source(args: argparse.Namespace) -> Source:
    sources = [bool(args.source_shm_name), bool(args.source_port), bool(args.source_file)]
    if sum(sources) != 1:
        raise SystemExit("Specify exactly one source: --source-shm-name, --source-port, or --source-file.")
    if args.source_shm_name:
        return ShmSource(args.source_shm_name, args.expected_dim, args.perception_contract_sha256)
    if args.source_port:
        return ZmqSource(
            args.source_port,
            args.expected_dim,
            args.source_key,
            args.perception_contract_sha256,
        )
    return FileSource(
        Path(args.source_file).expanduser(),
        args.expected_dim,
        args.source_key,
        args.loop_file,
        args.perception_contract_sha256,
        args.source_episode_generation,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-dim", type=_positive_int, default=58 * 87)
    parser.add_argument("--source-shm-name", default="")
    parser.add_argument("--source-port", type=int, default=0)
    parser.add_argument("--source-file", default="")
    parser.add_argument("--source-key", default="perception_obs")
    parser.add_argument(
        "--source-episode-generation",
        type=_nonnegative_int,
        default=None,
        help=(
            "Required only for file sources. It must match the live run_sim episode and is "
            "therefore safe for a single episode only; ZMQ/SHM sources carry it per frame."
        ),
    )
    parser.add_argument(
        "--perception-contract-sha256",
        type=_contract_sha256,
        required=True,
        help="Exact effective producer contract digest expected at the source and preserved at the destination.",
    )
    parser.add_argument("--dest-shm-name", default="depth_img_shm")
    parser.add_argument("--dest-port", type=int, default=0)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--loop-file", action="store_true")
    parser.add_argument("--require-fresh", action="store_true", help="Skip publishing repeated stale source frames.")
    parser.add_argument(
        "--initial-value",
        type=float,
        default=0.5,
        help=(
            "Deprecated compatibility option. No unauthenticated initial frame is published; "
            "the relay waits for the first episode-authenticated source frame."
        ),
    )
    parser.add_argument("--stats-every", type=float, default=2.0)
    args = parser.parse_args()

    if args.source_file and args.source_episode_generation is None:
        parser.error("--source-file requires --source-episode-generation for fail-closed pairing")
    if not args.source_file and args.source_episode_generation is not None:
        parser.error("--source-episode-generation is only valid with --source-file")

    source = _build_source(args)
    publisher = Publisher(
        dest_port=args.dest_port if args.dest_port > 0 else None,
        dest_shm_name=args.dest_shm_name or None,
        expected_dim=args.expected_dim,
        initial_value=args.initial_value,
        contract_sha256=args.perception_contract_sha256,
    )
    period = 1.0 / max(float(args.rate_hz), 1e-6)
    last_stats = time.monotonic()
    last_frame_identity: tuple[bytes, float | None, int | None] | None = None
    frame_idx = 0
    published = 0

    try:
        while True:
            obs = source.next_obs()
            if obs is None:
                time.sleep(min(period, 0.05))
                continue
            obs = obs.astype(np.float32, copy=False)
            if not np.isfinite(obs).all():
                logger.warning("Rejected non-finite source perception_obs frame")
                time.sleep(period)
                continue
            frame_identity = _frame_identity(
                obs,
                sim_time_ms=source.last_sim_time_ms,
                episode_generation=source.last_episode_generation,
            )
            if args.require_fresh and frame_identity == last_frame_identity:
                time.sleep(period)
                continue
            last_frame_identity = frame_identity
            if source.contract_sha256 != publisher.contract_sha256:
                raise RuntimeError(
                    "Relay source/destination perception contract changed unexpectedly: "
                    f"source={source.contract_sha256}, destination={publisher.contract_sha256}."
                )
            if source.last_episode_generation is None:
                raise RuntimeError(
                    "Relay source did not authenticate an episode generation; refusing to publish."
                )
            publisher.publish(
                obs,
                frame_idx=frame_idx,
                sim_time_ms=source.last_sim_time_ms,
                episode_generation=source.last_episode_generation,
            )
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
