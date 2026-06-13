from __future__ import annotations

import os
from pathlib import Path


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_OBJECT_MAP_NAME = "_clip_object_urdf_map.json"


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    return default


def rank_local_sharding_enabled() -> bool:
    root = os.environ.get("HOLOSOMA_RANK_LOCAL_MOTION_ROOT", "").strip()
    if not root:
        return False
    return _env_flag("HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED", default=True)


def current_rank_local_shard_dir() -> Path | None:
    if not rank_local_sharding_enabled():
        return None

    rank_raw = os.environ.get("LOCAL_RANK", "")
    if rank_raw == "":
        rank_raw = os.environ.get("RANK", "0")
    try:
        rank = int(rank_raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid rank for rank-local shard selection: {rank_raw!r}") from exc
    if rank < 0:
        raise RuntimeError(f"Invalid negative rank for rank-local shard selection: {rank}")

    root = Path(os.environ["HOLOSOMA_RANK_LOCAL_MOTION_ROOT"]).expanduser().resolve()
    shard_dir = root / f"rank_{rank}"
    if not shard_dir.is_dir():
        raise FileNotFoundError(
            f"Rank-local shard directory does not exist for rank {rank}: {shard_dir}. "
            "Check HOLOSOMA_RANK_LOCAL_MOTION_ROOT and NPROC/WORLD_SIZE."
        )
    return shard_dir


def resolve_rank_local_motion_path(path: str | os.PathLike[str]) -> str:
    shard_dir = current_rank_local_shard_dir()
    if shard_dir is None:
        return str(path)

    original = Path(path).expanduser()
    if original.is_dir():
        return str(shard_dir)
    return str(path)


def resolve_rank_local_object_map(path: str | os.PathLike[str]) -> str:
    shard_dir = current_rank_local_shard_dir()
    if shard_dir is None:
        return str(path)

    object_map = shard_dir / _OBJECT_MAP_NAME
    if not object_map.is_file():
        raise FileNotFoundError(
            f"Rank-local object map does not exist for shard {shard_dir}: {object_map}"
        )
    return str(object_map)
