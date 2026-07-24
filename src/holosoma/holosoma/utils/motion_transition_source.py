from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MOTION_TRANSITION_SOURCE_KEY = "motion_transition_source"
MOTION_TRANSITION_SOURCE_VERSION = 1
_SUPPORTED_SEMANTICS = {
    "single_clip_static",
    "global_multi_clip_runtime",
}
_OBJECT_MAP_NAMES = (
    "_clip_object_urdf_map.json",
    "clip_object_urdf_map.json",
)


def canonical_motion_transition_source(
    value: Any,
    *,
    active_clip_count: int,
    role: str = MOTION_TRANSITION_SOURCE_KEY,
) -> dict[str, Any]:
    """Validate the immutable source-bank transition lineage.

    This record is intentionally independent of rank-local/global objective
    clip counts.  It controls only the motion timeline implementation.
    """

    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    expected_keys = {"version", "source_clip_count", "source_semantics"}
    if set(value) != expected_keys:
        raise ValueError(
            f"{role} must contain exactly {sorted(expected_keys)}, got {sorted(map(str, value))}"
        )
    version = value.get("version")
    if type(version) is not int or version != MOTION_TRANSITION_SOURCE_VERSION:
        raise ValueError(
            f"{role}.version must be the integer {MOTION_TRANSITION_SOURCE_VERSION}, got {version!r}"
        )
    source_clip_count = value.get("source_clip_count")
    if type(source_clip_count) is not int or source_clip_count <= 0:
        raise ValueError(f"{role}.source_clip_count must be a positive integer")
    if type(active_clip_count) is not int or active_clip_count <= 0:
        raise ValueError(f"{role} active_clip_count must be a positive integer")
    if source_clip_count < active_clip_count:
        raise ValueError(
            f"{role}.source_clip_count={source_clip_count} is smaller than the active clip count "
            f"{active_clip_count}"
        )
    source_semantics = value.get("source_semantics")
    if source_semantics not in _SUPPORTED_SEMANTICS:
        raise ValueError(
            f"{role}.source_semantics must be one of {sorted(_SUPPORTED_SEMANTICS)}, "
            f"got {source_semantics!r}"
        )
    expected_semantics = (
        "single_clip_static" if source_clip_count == 1 else "global_multi_clip_runtime"
    )
    if source_semantics != expected_semantics:
        raise ValueError(
            f"{role} is internally inconsistent: source_clip_count={source_clip_count} requires "
            f"source_semantics={expected_semantics!r}, got {source_semantics!r}"
        )
    return {
        "version": MOTION_TRANSITION_SOURCE_VERSION,
        "source_clip_count": source_clip_count,
        "source_semantics": source_semantics,
    }


def resolve_motion_transition_source_for_motion_path(
    motion_path: str | Path,
    *,
    active_clip_count: int | None = None,
) -> dict[str, Any] | None:
    """Load explicit transition lineage from a directory object map.

    Legacy motion banks without this field return ``None`` so existing
    multi-clip/single-clip inference remains available.  A partially present
    or malformed explicit record always fails closed.
    """

    path = Path(motion_path).expanduser()
    if not path.is_dir():
        return None
    object_map: Path | None = None
    for name in _OBJECT_MAP_NAMES:
        candidate = path / name
        if candidate.is_file():
            object_map = candidate
            break
    if object_map is None:
        return None
    try:
        payload = json.loads(object_map.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse motion object map {object_map}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Motion object map must be a JSON object: {object_map}")
    if MOTION_TRANSITION_SOURCE_KEY not in payload:
        return None
    if active_clip_count is None:
        active_clip_count = len(
            [
                candidate
                for candidate in path.iterdir()
                if candidate.is_file() and candidate.suffix.lower() == ".npz"
            ]
        )
    return canonical_motion_transition_source(
        payload[MOTION_TRANSITION_SOURCE_KEY],
        active_clip_count=active_clip_count,
        role=f"motion object map {object_map} {MOTION_TRANSITION_SOURCE_KEY}",
    )
