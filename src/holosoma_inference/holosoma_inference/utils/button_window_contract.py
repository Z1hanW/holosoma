"""Digest-bound source-clock contract for contact-aware policy buttons."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np


EMBEDDED_BUTTON_WINDOW_CONTRACT_KEY = "embedded_button_window_contract"
EMBEDDED_BUTTON_WINDOW_CONTRACT_SHA256_KEY = (
    "embedded_button_window_contract_sha256"
)
EMBEDDED_BUTTON_WINDOW_CONTRACT_VERSION = 1

CONTACT_AWARE_BUTTON_WINDOW_MODES = frozenset(
    {"contact_interval", "kinematic_lift"}
)
KINEMATIC_LIFT_ALGORITHM = "object_root_rel_z_v1"
KINEMATIC_LIFT_HEIGHT_THRESHOLD = 0.10
KINEMATIC_LIFT_RATIO_THRESHOLD = 0.35
KINEMATIC_LIFT_CONSECUTIVE_STEPS = 5

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def validated_contact_aware_button_window_mode(
    motion_config: Mapping[str, object],
) -> str:
    """Return the exact serialized mode, preserving the legacy default."""

    raw_mode = motion_config.get(
        "contact_aware_button_window_mode",
        "contact_interval",
    )
    if not isinstance(raw_mode, str) or raw_mode not in CONTACT_AWARE_BUTTON_WINDOW_MODES:
        raise ValueError(
            "motion_config.contact_aware_button_window_mode must be exactly "
            f"'contact_interval' or 'kinematic_lift', got {raw_mode!r}."
        )
    return raw_mode


def _first_sustained_true_index(mask: np.ndarray, consecutive_steps: int) -> int | None:
    mask = np.asarray(mask, dtype=np.bool_).reshape(-1)
    consecutive_steps = max(1, int(consecutive_steps))
    if mask.size < consecutive_steps:
        return None
    candidates = np.convolve(
        mask.astype(np.int64, copy=False),
        np.ones((consecutive_steps,), dtype=np.int64),
        mode="valid",
    )
    indices = np.flatnonzero(candidates == consecutive_steps)
    return None if indices.size == 0 else int(indices[0])


def _first_sustained_true_index_from(
    mask: np.ndarray,
    consecutive_steps: int,
    *,
    start_idx: int,
) -> int | None:
    mask = np.asarray(mask, dtype=np.bool_).reshape(-1)
    start_idx = max(0, min(int(start_idx), int(mask.size)))
    relative = _first_sustained_true_index(mask[start_idx:], consecutive_steps)
    return None if relative is None else start_idx + relative


def kinematic_lift_window_from_rel_z_np(rel_z: np.ndarray) -> tuple[int, int]:
    """Resolve the canonical source-clock ``[t1, t2)`` kinematic lift window.

    All threshold arithmetic is deliberately float32, matching MotionLoader's
    training tensors.  The implementation is independent of contact sidecars:
    no release lead or contact-interval cap is permitted here.
    """

    values = np.asarray(rel_z, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(
            f"Kinematic button rel-z trace must be rank 1, got shape {values.shape}."
        )
    if values.size == 0:
        return 0, 0
    if not np.all(np.isfinite(values)):
        raise ValueError("Kinematic button rel-z trace must contain only finite values.")

    z_min = np.min(values).astype(np.float32)
    z_range = np.maximum(
        np.max(values).astype(np.float32) - z_min,
        np.float32(0.0),
    ).astype(np.float32)
    threshold = (
        z_min
        + np.maximum(
            np.float32(KINEMATIC_LIFT_HEIGHT_THRESHOLD),
            z_range * np.float32(KINEMATIC_LIFT_RATIO_THRESHOLD),
        ).astype(np.float32)
    ).astype(np.float32)

    lifted_mask = values >= threshold
    start = _first_sustained_true_index(
        lifted_mask,
        KINEMATIC_LIFT_CONSECUTIVE_STEPS,
    )
    if start is None:
        raise ValueError(
            "Kinematic button motion never reaches the lift threshold for "
            f"{KINEMATIC_LIFT_CONSECUTIVE_STEPS} consecutive frames."
        )

    end = _first_sustained_true_index_from(
        values < threshold,
        KINEMATIC_LIFT_CONSECUTIVE_STEPS,
        start_idx=min(int(start) + 1, int(values.size)),
    )
    if end is None:
        end = int(values.size)

    start = max(0, min(int(start), int(values.size)))
    end = max(start, min(int(end), int(values.size)))
    return start, end


def map_source_window_to_materialized_timeline(
    window: tuple[int, int],
    *,
    source_semantics: str,
    prepend_steps: int,
) -> tuple[int, int]:
    """Map a source-motion window through the authenticated runtime prepend."""

    start, end = int(window[0]), int(window[1])
    prepend_steps = int(prepend_steps)
    if source_semantics != "global_multi_clip_runtime" or prepend_steps <= 0:
        return start, end
    return (0 if start == 0 else start + prepend_steps, end + prepend_steps)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Button-window provenance must contain strict finite JSON values."
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def build_kinematic_button_window_contract(
    *,
    clip_id: str,
    source_motion_sha256: str,
    source_motion_size: int,
    source_frame_count: int,
    motion_fps: float,
    source_window: tuple[int, int],
    motion_transition_contract_sha256: str,
    source_semantics: str,
    effective_prepend_steps: int,
    effective_append_steps: int,
    materialized_window: tuple[int, int] | None = None,
) -> tuple[dict[str, object], str]:
    """Build the immutable integer window contract embedded by the patcher."""

    if materialized_window is None:
        if source_semantics == "single_clip_static" and (
            int(effective_prepend_steps) > 0 or int(effective_append_steps) > 0
        ):
            raise ValueError(
                "Static-splice button contracts require a window recomputed from the "
                "materialized object/root trace."
            )
        materialized_window = map_source_window_to_materialized_timeline(
            source_window,
            source_semantics=source_semantics,
            prepend_steps=effective_prepend_steps,
        )
    contract: dict[str, object] = {
        "version": EMBEDDED_BUTTON_WINDOW_CONTRACT_VERSION,
        "mode": "kinematic_lift",
        "algorithm": KINEMATIC_LIFT_ALGORITHM,
        "lift_height_threshold": KINEMATIC_LIFT_HEIGHT_THRESHOLD,
        "lift_ratio_threshold": KINEMATIC_LIFT_RATIO_THRESHOLD,
        "consecutive_steps": KINEMATIC_LIFT_CONSECUTIVE_STEPS,
        "clip_id": str(clip_id),
        "source_motion_sha256": str(source_motion_sha256),
        "source_motion_size": int(source_motion_size),
        "source_frame_count": int(source_frame_count),
        "motion_fps": float(motion_fps),
        "source_window": [int(source_window[0]), int(source_window[1])],
        "motion_transition_contract_sha256": str(
            motion_transition_contract_sha256
        ),
        "source_semantics": str(source_semantics),
        "effective_prepend_steps": int(effective_prepend_steps),
        "effective_append_steps": int(effective_append_steps),
        "materialized_window": [
            int(materialized_window[0]),
            int(materialized_window[1]),
        ],
    }
    _validate_contract(contract)
    return contract, _sha256(contract)


def _validate_window(
    value: object,
    *,
    path: str,
    maximum: int,
) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{path} must be a two-integer JSON list.")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in value):
        raise ValueError(f"{path} must be a two-integer JSON list.")
    start, end = int(value[0]), int(value[1])
    if not 0 <= start <= end <= maximum:
        raise ValueError(
            f"{path} must satisfy 0 <= start <= end <= {maximum}, got {value!r}."
        )
    return start, end


def _validate_contract(contract: Mapping[str, object]) -> dict[str, object]:
    expected_keys = {
        "version",
        "mode",
        "algorithm",
        "lift_height_threshold",
        "lift_ratio_threshold",
        "consecutive_steps",
        "clip_id",
        "source_motion_sha256",
        "source_motion_size",
        "source_frame_count",
        "motion_fps",
        "source_window",
        "motion_transition_contract_sha256",
        "source_semantics",
        "effective_prepend_steps",
        "effective_append_steps",
        "materialized_window",
    }
    if set(contract) != expected_keys:
        raise ValueError(
            "Embedded button-window contract keys are not canonical: "
            f"expected={sorted(expected_keys)}, actual={sorted(contract)}."
        )
    version = contract["version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, Integral)
        or int(version) != EMBEDDED_BUTTON_WINDOW_CONTRACT_VERSION
    ):
        raise ValueError("Unsupported embedded button-window contract version.")
    if contract["mode"] != "kinematic_lift":
        raise ValueError("Embedded button-window contract mode must be 'kinematic_lift'.")
    if contract["algorithm"] != KINEMATIC_LIFT_ALGORITHM:
        raise ValueError("Unsupported embedded button-window algorithm.")
    height_threshold = contract["lift_height_threshold"]
    if (
        isinstance(height_threshold, bool)
        or not isinstance(height_threshold, Real)
        or not math.isfinite(float(height_threshold))
        or float(height_threshold) != KINEMATIC_LIFT_HEIGHT_THRESHOLD
    ):
        raise ValueError("Embedded button-window lift-height threshold changed.")
    ratio_threshold = contract["lift_ratio_threshold"]
    if (
        isinstance(ratio_threshold, bool)
        or not isinstance(ratio_threshold, Real)
        or not math.isfinite(float(ratio_threshold))
        or float(ratio_threshold) != KINEMATIC_LIFT_RATIO_THRESHOLD
    ):
        raise ValueError("Embedded button-window lift-ratio threshold changed.")
    consecutive_steps = contract["consecutive_steps"]
    if (
        isinstance(consecutive_steps, bool)
        or not isinstance(consecutive_steps, Integral)
        or int(consecutive_steps) != KINEMATIC_LIFT_CONSECUTIVE_STEPS
    ):
        raise ValueError("Embedded button-window sustained-step contract changed.")

    clip_id = contract["clip_id"]
    if not isinstance(clip_id, str) or not clip_id or clip_id != clip_id.strip():
        raise ValueError("Embedded button-window clip_id must be a canonical string.")
    for key in ("source_motion_sha256", "motion_transition_contract_sha256"):
        value = contract[key]
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"Embedded button-window {key} must be lowercase SHA-256.")
    for key in ("source_motion_size", "source_frame_count"):
        value = contract[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
            raise ValueError(f"Embedded button-window {key} must be a positive integer.")
    fps = contract["motion_fps"]
    if isinstance(fps, bool) or not isinstance(fps, Real) or not math.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("Embedded button-window motion_fps must be finite and positive.")
    semantics = contract["source_semantics"]
    if not isinstance(semantics, str) or semantics not in {
        "global_multi_clip_runtime",
        "single_clip_static",
    }:
        raise ValueError("Embedded button-window source_semantics is unsupported.")
    prepend = contract["effective_prepend_steps"]
    append = contract["effective_append_steps"]
    for key, value in (
        ("effective_prepend_steps", prepend),
        ("effective_append_steps", append),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            raise ValueError(f"Embedded button-window {key} must be non-negative.")

    source_frames = int(contract["source_frame_count"])
    source_window = _validate_window(
        contract["source_window"],
        path="embedded_button_window_contract.source_window",
        maximum=source_frames,
    )
    materialized_maximum = source_frames + int(prepend) + int(append)
    materialized_window = _validate_window(
        contract["materialized_window"],
        path="embedded_button_window_contract.materialized_window",
        maximum=materialized_maximum,
    )
    if semantics == "global_multi_clip_runtime" or (
        int(prepend) == 0 and int(append) == 0
    ):
        expected_materialized = map_source_window_to_materialized_timeline(
            source_window,
            source_semantics=str(semantics),
            prepend_steps=int(prepend),
        )
        if materialized_window != expected_materialized:
            raise ValueError(
                "Embedded button-window materialized_window does not match its source timeline."
            )
    return dict(contract)


def embedded_button_window_contract_from_metadata(
    metadata: Mapping[str, object],
    *,
    required: bool = False,
) -> dict[str, object] | None:
    """Return and authenticate the embedded integer button-window contract."""

    raw_contract = metadata.get(EMBEDDED_BUTTON_WINDOW_CONTRACT_KEY)
    raw_digest = metadata.get(EMBEDDED_BUTTON_WINDOW_CONTRACT_SHA256_KEY)
    if raw_contract is None and raw_digest is None:
        if required:
            raise ValueError("ONNX metadata is missing its embedded button-window contract.")
        return None
    if raw_contract is None or raw_digest is None:
        raise ValueError(
            "Embedded button-window contract and SHA-256 metadata must appear together."
        )
    if not isinstance(raw_contract, Mapping):
        raise ValueError("Embedded button-window contract must be a JSON object.")
    if not isinstance(raw_digest, str) or _SHA256_RE.fullmatch(raw_digest) is None:
        raise ValueError("Embedded button-window contract digest must be lowercase SHA-256.")
    contract = _validate_contract(raw_contract)
    actual_digest = _sha256(contract)
    if actual_digest != raw_digest:
        raise ValueError(
            "Embedded button-window contract digest mismatch: "
            f"declared={raw_digest}, actual={actual_digest}."
        )
    return contract
