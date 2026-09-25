from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path


def resolve_robot_usd_conversion_dir(
    asset_root: str | os.PathLike[str],
    local_rank: int,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve one rank-private robot URDF conversion directory."""

    if isinstance(local_rank, bool) or not isinstance(local_rank, int) or local_rank < 0:
        raise ValueError(f"local_rank must be a non-negative integer, got {local_rank!r}")

    environment = os.environ if environ is None else environ
    explicit_root = environment.get("HOLOSOMA_ROBOT_USD_CACHE_DIR", "").strip()
    if explicit_root:
        root_candidate = Path(explicit_root).expanduser()
        if not root_candidate.is_absolute():
            raise ValueError(
                "HOLOSOMA_ROBOT_USD_CACHE_DIR must be an absolute path when set, "
                f"got {explicit_root!r}"
            )
        cache_root = root_candidate.resolve()
        if cache_root == Path(cache_root.anchor):
            raise ValueError("HOLOSOMA_ROBOT_USD_CACHE_DIR must not be a filesystem root")
    else:
        cache_root = Path(asset_root).expanduser().resolve()

    return cache_root / f"converted_rank{local_rank}"
