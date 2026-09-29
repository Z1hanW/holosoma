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
    {"contact_interval", "kinematic_lift", "peak_height"}
)
PEAK_HEIGHT_ALGORITHM = "object_world_z_peak_height_v1"
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
            f"'contact_interval', 'kinematic_lift' or 'peak_height', got {raw_mode!r}."
        )
    return raw_mode


def height_button_algorithm_contract(motion_config: Mapping[str, object]) -> dict[str, object]:
    mode = validated_contact_aware_button_window_mode(motion_config)
    if mode == "kinematic_lift":
        return {
            "mode": mode,
            "algorithm": KINEMATIC_LIFT_ALGORITHM,
            "lift_height_threshold": KINEMATIC_LIFT_HEIGHT_THRESHOLD,
            "lift_ratio_threshold": KINEMATIC_LIFT_RATIO_THRESHOLD,
            "consecutive_steps": KINEMATIC_LIFT_CONSECUTIVE_STEPS,
        }
    if mode != "peak_height":
        raise ValueError("Height-derived button contracts require peak_height or kinematic_lift.")
    alpha = motion_config.get("contact_aware_peak_height_alpha", 0.91)
    smoothing = motion_config.get("contact_aware_peak_height_smoothing_steps", 5)
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, Real)
        or not math.isfinite(float(alpha))
        or not 0 <= alpha <= 1
    ):
        raise ValueError("Peak-height alpha must be finite and in [0, 1].")
    if isinstance(smoothing, bool) or not isinstance(smoothing, Integral) or not 1 <= smoothing <= 4096:
        raise ValueError("Peak-height smoothing_steps must be an integer in [1, 4096].")
    return {
        "mode": mode,
        "algorithm": PEAK_HEIGHT_ALGORITHM,
        "peak_height_alpha": float(alpha),
        "smoothing_steps": int(smoothing),
        "consecutive_steps": KINEMATIC_LIFT_CONSECUTIVE_STEPS,
    }


def height_button_window_from_motion_np(
    object_height: np.ndarray,
    root_height: np.ndarray,
    motion_config: Mapping[str, object],
) -> tuple[int, int]:
    """Mirror the source-clock training rule; peak height never reads sidecars."""
    params = height_button_algorithm_contract(motion_config)
    if params["mode"] == "kinematic_lift":
        return kinematic_lift_window_from_rel_z_np(
            np.asarray(object_height, dtype=np.float32) - np.asarray(root_height, dtype=np.float32)
        )
    values = np.asarray(object_height, dtype=np.float32)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Peak-height object trace must be finite and rank 1.")
    if values.size == 0:
        return 0, 0
    smoothing = int(params["smoothing_steps"])
    padded = np.pad(values, (smoothing // 2, smoothing - 1 - smoothing // 2), mode="edge")
    height = np.convolve(padded, np.full(smoothing, np.float32(1.0 / smoothing)), mode="valid")
    threshold = np.float32(
        height.min() + np.float32(height.max() - height.min()) * np.float32(params["peak_height_alpha"])
    )
    high = height >= threshold
    start = _first_sustained_true_index(high, KINEMATIC_LIFT_CONSECUTIVE_STEPS)
    if start is None:
        # Retain the exact SW rule for short plateaus, including flat traces.
        indices = np.flatnonzero(high)
        start = int(indices[0]) if indices.size else int(np.argmax(height))
    end = _first_sustained_true_index_from(
        ~high,
        KINEMATIC_LIFT_CONSECUTIVE_STEPS,
        start_idx=min(int(np.argmax(height)) + 1, values.size),
    )
    return int(start), max(int(start), values.size if end is None else int(end))


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


def build_source_button_window_contract(
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
    motion_config: Mapping[str, object] | None = None,
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
        **height_button_algorithm_contract(
            {"contact_aware_button_window_mode": "kinematic_lift"}
            if motion_config is None
            else motion_config
        ),
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


# Preserve the historical builder API and byte-identical kinematic contracts.
build_kinematic_button_window_contract = build_source_button_window_contract


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
    mode = contract.get("mode")
    algorithm_config = {"contact_aware_button_window_mode": mode}
    if mode == "peak_height":
        algorithm_config.update(
            contact_aware_peak_height_alpha=contract.get("peak_height_alpha"),
            contact_aware_peak_height_smoothing_steps=contract.get("smoothing_steps"),
        )
    algorithm = height_button_algorithm_contract(algorithm_config)
    expected_keys = {
        "version",
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
    } | set(algorithm)
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
    if contract["algorithm"] != algorithm["algorithm"]:
        raise ValueError("Unsupported embedded button-window algorithm.")
    if mode == "kinematic_lift":
        for key in ("lift_height_threshold", "lift_ratio_threshold"):
            value = contract[key]
            if isinstance(value, bool) or not isinstance(value, Real) or value != algorithm[key]:
                raise ValueError(f"Embedded button-window {key} changed.")
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
