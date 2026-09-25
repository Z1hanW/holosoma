"""Pure helpers shared by contact-window preflight and training runtime."""

from __future__ import annotations

import math
from collections.abc import Collection
from numbers import Integral, Real
from typing import Any

import numpy as np


CONTACT_INTERVAL_REGION_ALIASES = {
    "left_palm": "left_wrist",
    "right_palm": "right_wrist",
}
CONTACT_INTERVAL_PRIMARY_REGION_GROUPS = (
    ("left_wrist", "right_wrist"),
    (
        "left_elbow",
        "right_elbow",
        "left_wrist_roll",
        "right_wrist_roll",
        "left_wrist_pitch",
        "right_wrist_pitch",
        "torso",
    ),
)
CONTACT_INTERVAL_FALLBACK_FILES = {
    "left_wrist": "left_wrist_contact_interval_steps.npy",
    "right_wrist": "right_wrist_contact_interval_steps.npy",
    "left_elbow": "left_elbow_contact_interval_steps.npy",
    "right_elbow": "right_elbow_contact_interval_steps.npy",
    "left_wrist_roll": "left_wrist_roll_contact_interval_steps.npy",
    "right_wrist_roll": "right_wrist_roll_contact_interval_steps.npy",
    "left_wrist_pitch": "left_wrist_pitch_contact_interval_steps.npy",
    "right_wrist_pitch": "right_wrist_pitch_contact_interval_steps.npy",
    "torso": "torso_contact_interval_steps.npy",
}


def infer_contact_export_clip_id(directory_name: str) -> str:
    """Remove only the exporter-added numeric ordering prefix.

    Contact exports conventionally use names such as ``0034_box_10`` while
    some tools consume an already-normalized directory named ``box_10``.
    Splitting every name at its first underscore corrupts the latter into
    ``10`` (and similarly corrupts most AS clip identifiers).  A prefix is
    therefore structural only when it is a non-empty decimal index.
    """

    normalized = str(directory_name).strip()
    prefix, separator, suffix = normalized.partition("_")
    if separator and prefix.isdecimal() and suffix.strip():
        return suffix.strip()
    return normalized


def resolve_contact_export_clip_id(
    directory_name: str,
    active_clip_ids: Collection[str],
) -> str:
    """Resolve a directory against active clips without numeric-ID ambiguity.

    A real clip ID may itself begin with a decimal component.  Exact active
    names therefore take precedence over interpreting that component as an
    exporter ordering prefix.
    """

    normalized = str(directory_name).strip()
    if normalized in active_clip_ids:
        return normalized
    return infer_contact_export_clip_id(normalized)


def convert_contact_interval_timebase(
    interval: tuple[int, int],
    *,
    metadata: dict[str, Any] | None,
    motion_fps: float,
) -> tuple[int, int]:
    """Convert a half-open exported interval into active motion-step time."""

    if not metadata:
        return int(interval[0]), int(interval[1])
    raw_source_fps = metadata.get("contact_interval_fps", metadata.get("fps"))
    if raw_source_fps is None:
        return int(interval[0]), int(interval[1])
    if (
        isinstance(raw_source_fps, (bool, np.bool_))
        or not isinstance(raw_source_fps, Real)
        or isinstance(motion_fps, (bool, np.bool_))
        or not isinstance(motion_fps, Real)
    ):
        raise ValueError(
            f"Contact interval FPS metadata must be real numeric values: source={raw_source_fps!r}, "
            f"motion={motion_fps!r}."
        )
    try:
        source_fps = float(raw_source_fps)
        target_fps = float(motion_fps)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Contact interval FPS metadata must be numeric: source={raw_source_fps!r}, "
            f"motion={motion_fps!r}."
        ) from exc
    if (
        not math.isfinite(source_fps)
        or source_fps <= 0.0
        or not math.isfinite(target_fps)
        or target_fps <= 0.0
    ):
        raise ValueError(
            f"Contact interval FPS values must be finite and positive: source={source_fps}, "
            f"motion={target_fps}."
        )
    start_step, end_step = int(interval[0]), int(interval[1])
    if math.isclose(source_fps, target_fps, rel_tol=0.0, abs_tol=1.0e-9):
        return start_step, end_step

    scale = target_fps / source_fps
    converted_start = int(math.ceil(start_step * scale - 1.0e-9))
    converted_end = int(math.ceil(end_step * scale - 1.0e-9))
    if converted_end <= converted_start:
        raise ValueError(
            "Contact interval became empty after FPS conversion: "
            f"interval={interval}, source_fps={source_fps}, motion_fps={target_fps}, "
            f"converted={(converted_start, converted_end)}."
        )
    return converted_start, converted_end


def normalize_contact_interval_pair(raw_interval: Any) -> tuple[int, int] | None:
    if isinstance(raw_interval, (list, tuple)):
        if len(raw_interval) != 2 or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            for value in raw_interval
        ):
            return None
    try:
        values = np.asarray(raw_interval).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size != 2 or values.dtype.kind not in {"i", "u"}:
        return None
    start_step = int(values[0])
    end_step = int(values[1])
    if start_step < 0 or end_step <= start_step:
        return None
    return start_step, end_step


def select_primary_contact_interval(intervals_by_region: dict[str, Any]) -> tuple[int, int] | None:
    """Select the union of every recognized carry-contact region.

    Wrist-yaw links are not a reliable proxy for the complete hand/arm
    contact envelope: on some valid carries they touch only briefly at pickup
    or release while wrist-pitch/roll links remain in contact throughout.  A
    wrist-first early return therefore produced windows that did not overlap
    the object's lift at all.  Keep unknown-region fallback compatibility, but
    when the exporter provides recognized regions, bind their full union.
    """

    normalized: dict[str, tuple[int, int]] = {}
    for raw_region_name, raw_interval in intervals_by_region.items():
        region_name = str(raw_region_name).strip()
        region_name = CONTACT_INTERVAL_REGION_ALIASES.get(region_name, region_name)
        if not region_name:
            continue
        interval = normalize_contact_interval_pair(raw_interval)
        if interval is not None:
            normalized[region_name] = interval

    carry_intervals = [
        normalized[name]
        for region_group in CONTACT_INTERVAL_PRIMARY_REGION_GROUPS
        for name in region_group
        if name in normalized
    ]
    if carry_intervals:
        return (
            min(interval[0] for interval in carry_intervals),
            max(interval[1] for interval in carry_intervals),
        )

    if normalized:
        return (
            min(interval[0] for interval in normalized.values()),
            max(interval[1] for interval in normalized.values()),
        )
    return None
