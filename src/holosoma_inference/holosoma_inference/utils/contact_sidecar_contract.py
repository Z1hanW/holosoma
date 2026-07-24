"""Digest-bound active contact-sidecar provenance for patched WBT artifacts.

The training provenance binds the complete contact bank.  A deployed policy,
however, should not need that complete bank merely to reproduce the button
window for one embedded motion.  This module verifies the complete bank once
while patching and serializes a strict, per-clip contract that inference can
consume without reopening mutable sidecar paths.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from holosoma_inference.utils.embedded_motion_timeline import (
    embedded_motion_timeline_contract_from_metadata,
    read_stable_regular_file_bytes,
)
from holosoma_inference.utils.policy_contract import PolicyContractError


EMBEDDED_CONTACT_SIDECAR_CONTRACT_KEY = "embedded_contact_sidecar_contract"
EMBEDDED_CONTACT_SIDECAR_CONTRACT_SHA256_KEY = (
    "embedded_contact_sidecar_contract_sha256"
)
EMBEDDED_CONTACT_SIDECAR_CONTRACT_VERSION = 1

CONTACT_WINDOW_OBSERVATION_TERMS = frozenset(
    {
        "sparse_target_root_trajectory_command_contact_aware",
        "drop_button",
        "pickup_button",
    }
)

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
CONTACT_INTERVAL_REGION_ALIASES = {
    "left_palm": "left_wrist",
    "right_palm": "right_wrist",
}

# These names intentionally mirror scripts/compute_training_provenance.py.
_REQUIRED_CONTACT_FILES = (
    "teacher_rollout_reference.npz",
    "left_wrist_contact_points.npy",
    "left_wrist_contact_point_counts.npy",
    "left_wrist_contact_interval_steps.npy",
    "right_wrist_contact_points.npy",
    "right_wrist_contact_point_counts.npy",
    "right_wrist_contact_interval_steps.npy",
)
_OPTIONAL_RUNTIME_CONTACT_FILES = (
    "contact_intervals.json",
    "metadata.json",
)
_RUNTIME_CONTACT_REGIONS = (
    "left_wrist",
    "right_wrist",
    "left_elbow",
    "right_elbow",
    "left_wrist_roll",
    "right_wrist_roll",
    "left_wrist_pitch",
    "right_wrist_pitch",
    "torso",
)
_RUNTIME_CONTACT_REGION_FILE_SUFFIXES = (
    "_contact_points.npy",
    "_contact_point_counts.npy",
    "_contact_interval_steps.npy",
    "_contact_active_mask.npy",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _canonical_json_bytes(value: Any, *, ensure_ascii: bool) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=ensure_ascii,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PolicyContractError(
            "Contact-sidecar provenance must contain strict finite JSON values."
        ) from exc


def _canonical_contract_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value, ensure_ascii=True)).hexdigest()


def _training_manifest_sha256(value: Mapping[str, Any]) -> str:
    # The training-side algorithm uses ensure_ascii=False.  Preserve that exact
    # byte contract so non-ASCII clip names cannot produce a false mismatch.
    return hashlib.sha256(_canonical_json_bytes(value, ensure_ascii=False)).hexdigest()


def _validate_sha256(value: Any, *, path: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        suffix = " or null" if optional else ""
        raise PolicyContractError(
            f"{path} must be 64 lowercase hexadecimal characters{suffix}."
        )
    return value


def _validate_positive_fps(value: Any, *, path: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise PolicyContractError(f"{path} must be a finite positive real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise PolicyContractError(f"{path} must be a finite positive real number.")
    return result


def normalize_contact_interval(raw_interval: Any) -> tuple[int, int] | None:
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
    # JSON floats and booleans must not become a different schedule through an
    # implicit int64 cast.  The exported half-open interval is exactly two
    # integer steps; extra values are schema errors, not ignorable metadata.
    if values.size != 2 or values.dtype.kind not in {"i", "u"}:
        return None
    start, end = int(values[0]), int(values[1])
    if start < 0 or end <= start:
        return None
    return start, end


def select_primary_contact_interval(
    intervals_by_region: Mapping[str, object],
) -> tuple[int, int] | None:
    """Mirror the training-side union of all recognized carry regions."""

    normalized: dict[str, tuple[int, int]] = {}
    for raw_name, raw_interval in intervals_by_region.items():
        stripped_name = str(raw_name).strip()
        name = CONTACT_INTERVAL_REGION_ALIASES.get(stripped_name, stripped_name)
        interval = normalize_contact_interval(raw_interval)
        if name and interval is not None:
            normalized[name] = interval

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


def infer_contact_export_clip_id(directory_name: str) -> str:
    normalized = str(directory_name).strip()
    prefix, separator, suffix = normalized.partition("_")
    if separator and prefix.isdecimal() and suffix.strip():
        return suffix.strip()
    return normalized


def resolve_contact_export_clip_id(directory_name: str, active_clip_ids: set[str]) -> str:
    normalized = str(directory_name).strip()
    if normalized in active_clip_ids:
        return normalized
    return infer_contact_export_clip_id(normalized)


def policy_uses_contact_window(metadata: Mapping[str, Any]) -> bool:
    experiment = metadata.get("experiment_config")
    if not isinstance(experiment, Mapping):
        return False
    algo = experiment.get("algo")
    config = algo.get("config") if isinstance(algo, Mapping) else None
    module_dict = config.get("module_dict") if isinstance(config, Mapping) else None
    actor = module_dict.get("actor") if isinstance(module_dict, Mapping) else None
    actor_groups = actor.get("input_dim") if isinstance(actor, Mapping) else None
    observation = experiment.get("observation")
    groups = observation.get("groups") if isinstance(observation, Mapping) else None
    if not isinstance(actor_groups, (list, tuple)) or not isinstance(groups, Mapping):
        return False
    for group_name in actor_groups:
        group = groups.get(group_name)
        terms = group.get("terms") if isinstance(group, Mapping) else None
        if isinstance(terms, Mapping) and CONTACT_WINDOW_OBSERVATION_TERMS.intersection(terms):
            return True
    return False


def policy_requires_contact_window(metadata: Mapping[str, Any]) -> bool:
    """Return the shared training/inference sidecar-consumer predicate."""

    if policy_uses_contact_window(metadata):
        return True
    experiment = metadata.get("experiment_config")
    command = experiment.get("command") if isinstance(experiment, Mapping) else None
    setup_terms = command.get("setup_terms") if isinstance(command, Mapping) else None
    motion_command = setup_terms.get("motion_command") if isinstance(setup_terms, Mapping) else None
    params = motion_command.get("params") if isinstance(motion_command, Mapping) else None
    motion_cfg = params.get("motion_config") if isinstance(params, Mapping) else None
    if not isinstance(motion_cfg, Mapping):
        return False
    flags: list[bool] = []
    for key in (
        "use_adaptive_timesteps_sampler",
        "uniform_t1_window_sampling_enabled",
    ):
        value = motion_cfg.get(key, False)
        if not isinstance(value, bool):
            raise PolicyContractError(f"motion_config.{key} must be boolean, got {value!r}.")
        flags.append(value)
    return any(flags)


def _directory_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat_result = path.stat()
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
    )


def _stable_file_record(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    payload = read_stable_regular_file_bytes(path, label=label)
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }, payload


def _contact_directory_map(clips_root: Path, motion_ids: set[str]) -> dict[str, Path]:
    root_identity = _directory_identity(clips_root)
    result: dict[str, list[Path]] = {}
    for directory in sorted(path for path in clips_root.iterdir() if path.is_dir()):
        clip_id = resolve_contact_export_clip_id(directory.name, motion_ids)
        if clip_id in motion_ids:
            result.setdefault(clip_id, []).append(directory)
    if _directory_identity(clips_root) != root_identity:
        raise RuntimeError(
            f"Contact-sidecar directory changed while it was enumerated: {clips_root}"
        )
    duplicates = {clip_id: paths for clip_id, paths in result.items() if len(paths) != 1}
    if duplicates:
        raise ValueError(
            "Contact export has duplicate directories for active clips: "
            + ", ".join(
                f"{clip_id}={paths}" for clip_id, paths in sorted(duplicates.items())
            )
        )
    missing = sorted(motion_ids.difference(result))
    if missing:
        raise ValueError(f"Contact export is missing active clips: {missing[:20]}")
    return {clip_id: paths[0] for clip_id, paths in result.items()}


def _verified_contact_manifest(
    *,
    motion_bank_dir: Path,
    contact_root: Path,
    active_clip_id: str,
) -> tuple[
    str,
    dict[str, dict[str, Any]],
    dict[str, bytes],
    Path,
    dict[str, Any],
]:
    """Recompute the exact v3 training digest and retain active runtime bytes."""

    motion_bank_dir = motion_bank_dir.expanduser().resolve()
    contact_root = contact_root.expanduser().resolve()
    clips_root = contact_root / "clips" if (contact_root / "clips").is_dir() else contact_root
    if not motion_bank_dir.is_dir():
        raise FileNotFoundError(f"Motion bank directory does not exist: {motion_bank_dir}")
    if not clips_root.is_dir():
        raise FileNotFoundError(f"Contact export root does not exist: {contact_root}")

    motion_root_identity = _directory_identity(motion_bank_dir)
    clips_root_identity = _directory_identity(clips_root)
    motion_ids = {path.stem for path in motion_bank_dir.glob("*.npz")}
    if not motion_ids:
        raise ValueError(f"Motion directory has no active clips: {motion_bank_dir}")
    if active_clip_id not in motion_ids:
        raise ValueError(
            f"Selected motion {active_clip_id!r} is not a member of motion bank {motion_bank_dir}."
        )
    bank_motion_path = motion_bank_dir / f"{active_clip_id}.npz"
    bank_motion_record, _ = _stable_file_record(
        bank_motion_path,
        label=f"Motion-bank member {active_clip_id}",
    )
    bank_motion_record = {"name": bank_motion_path.name, **bank_motion_record}
    directory_by_clip = _contact_directory_map(clips_root, motion_ids)

    rollout_candidates = (
        contact_root / "realmesh_rollout_manifest.json",
        contact_root.parent / "realmesh_rollout_manifest.json",
        contact_root.parent.parent / "realmesh_rollout_manifest.json",
    )
    rollout_manifests = [candidate for candidate in rollout_candidates if candidate.is_file()]
    unique_rollout_paths = {path.resolve() for path in rollout_manifests}
    if len(unique_rollout_paths) > 1:
        raise ValueError(
            "Contact export has ambiguous teacher rollout manifests: "
            f"{[str(path) for path in rollout_manifests]!r}"
        )
    rollout_manifest_record: dict[str, Any] | None = None
    if rollout_manifests:
        record, _ = _stable_file_record(
            rollout_manifests[0],
            label="Teacher rollout manifest",
        )
        rollout_manifest_record = record

    clips: dict[str, Any] = {}
    active_records: dict[str, dict[str, Any]] = {}
    active_payloads: dict[str, bytes] = {}
    for clip_id in sorted(motion_ids):
        clip_dir = directory_by_clip[clip_id]
        clip_dir_identity = _directory_identity(clip_dir)
        file_names = set(_REQUIRED_CONTACT_FILES)
        file_names.update(
            name
            for name in _OPTIONAL_RUNTIME_CONTACT_FILES
            if (clip_dir / name).is_file()
        )
        for region_name in _RUNTIME_CONTACT_REGIONS:
            for suffix in _RUNTIME_CONTACT_REGION_FILE_SUFFIXES:
                file_name = f"{region_name}{suffix}"
                if (clip_dir / file_name).is_file():
                    file_names.add(file_name)

        files: dict[str, Any] = {}
        for file_name in sorted(file_names):
            path = clip_dir / file_name
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing contact/rollout sidecar: {clip_id}/{file_name}"
                )
            record, payload = _stable_file_record(
                path,
                label=f"Contact sidecar {clip_id}/{file_name}",
            )
            files[file_name] = record
            if clip_id == active_clip_id:
                active_records[file_name] = record
                if file_name in _OPTIONAL_RUNTIME_CONTACT_FILES or file_name.endswith(
                    "_contact_interval_steps.npy"
                ):
                    active_payloads[file_name] = payload
        if _directory_identity(clip_dir) != clip_dir_identity:
            raise RuntimeError(
                f"Contact-sidecar directory changed while it was hashed: {clip_dir}"
            )
        clips[clip_id] = files

    final_motion_ids = {path.stem for path in motion_bank_dir.glob("*.npz")}
    if final_motion_ids != motion_ids or _directory_identity(motion_bank_dir) != motion_root_identity:
        raise RuntimeError(
            f"Motion bank changed while contact provenance was computed: {motion_bank_dir}"
        )
    if _directory_identity(clips_root) != clips_root_identity:
        raise RuntimeError(
            f"Contact export changed while its provenance was computed: {clips_root}"
        )
    digest = _training_manifest_sha256(
        {
            "version": 3,
            "clips": clips,
            "teacher_rollout_manifest": rollout_manifest_record,
        }
    )
    return (
        digest,
        active_records,
        active_payloads,
        directory_by_clip[active_clip_id],
        bank_motion_record,
    )


def _parse_active_runtime_payloads(
    *,
    active_clip_id: str,
    clip_dir: Path,
    payloads: Mapping[str, bytes],
) -> tuple[tuple[int, int], list[str], dict[str, object], str | None, float | None]:
    metadata: dict[str, object] = {}
    metadata_payload = payloads.get("metadata.json")
    if metadata_payload is not None:
        try:
            decoded_metadata = json.loads(metadata_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid contact metadata for active clip {active_clip_id!r}: {clip_dir / 'metadata.json'}"
            ) from exc
        if not isinstance(decoded_metadata, dict):
            raise ValueError("Active contact metadata must be a JSON object.")
        metadata = decoded_metadata
        declared_clip_id = str(metadata.get("clip_id") or "").strip()
        if declared_clip_id and declared_clip_id != active_clip_id:
            raise ValueError(
                "Active contact metadata clip_id does not match the selected motion: "
                f"declared={declared_clip_id!r}, selected={active_clip_id!r}."
            )

    intervals_by_region: dict[str, object] = {}
    source_files: list[str] = []
    json_payload = payloads.get("contact_intervals.json")
    if json_payload is not None:
        try:
            decoded_intervals = json.loads(json_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid contact_intervals.json for active clip {active_clip_id!r}."
            ) from exc
        if not isinstance(decoded_intervals, dict):
            raise ValueError("Active contact_intervals.json must be a JSON object.")
        intervals_by_region.update(decoded_intervals)
        source_files.append("contact_intervals.json")

    if not intervals_by_region:
        source_files.clear()
        for region_name, file_name in CONTACT_INTERVAL_FALLBACK_FILES.items():
            payload = payloads.get(file_name)
            if payload is None:
                continue
            try:
                loaded = np.load(io.BytesIO(payload), allow_pickle=False)
            except (OSError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid contact interval array for active clip: {file_name}."
                ) from exc
            if isinstance(loaded, np.lib.npyio.NpzFile):
                loaded.close()
                raise ValueError(f"Contact interval sidecar must be an NPY array: {file_name}.")
            intervals_by_region[region_name] = loaded
            source_files.append(file_name)

    interval = select_primary_contact_interval(intervals_by_region)
    if interval is None:
        raise ValueError(
            f"No valid contact interval exists for active clip {active_clip_id!r}."
        )

    fps_key: str | None = None
    raw_fps: object | None = None
    if "contact_interval_fps" in metadata:
        fps_key = "contact_interval_fps"
        raw_fps = metadata["contact_interval_fps"]
    elif "fps" in metadata:
        fps_key = "fps"
        raw_fps = metadata["fps"]
    source_fps = (
        None
        if fps_key is None or raw_fps is None
        else _validate_positive_fps(
            raw_fps,
            path=f"Contact metadata {fps_key}",
        )
    )
    return interval, sorted(source_files), metadata, fps_key, source_fps


def build_verified_contact_sidecar_contract(
    *,
    metadata: Mapping[str, Any],
    motion_path: Path,
    motion_bank_dir: Path,
    contact_root: Path,
    source_motion_sha256: str,
    source_motion_size: int,
    source_frame_count: int,
    motion_fps: float,
    verified_training_motion_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Verify the complete training bank and bind one active sidecar selection."""

    source_motion_sha256 = _validate_sha256(
        source_motion_sha256,
        path="source_motion_sha256",
    )
    if (
        isinstance(source_motion_size, bool)
        or not isinstance(source_motion_size, Integral)
        or int(source_motion_size) <= 0
    ):
        raise PolicyContractError("source_motion_size must be a positive integer.")
    source_motion_size = int(source_motion_size)
    if isinstance(source_frame_count, bool) or not isinstance(source_frame_count, Integral):
        raise PolicyContractError("source_frame_count must be a positive integer.")
    source_frame_count = int(source_frame_count)
    if source_frame_count <= 0:
        raise PolicyContractError("source_frame_count must be a positive integer.")
    motion_fps = _validate_positive_fps(motion_fps, path="motion_fps")

    training_provenance = metadata.get("training_provenance")
    if not isinstance(training_provenance, Mapping):
        raise PolicyContractError(
            "A digest-bound active contact sidecar requires training_provenance metadata."
        )
    declared_manifest_sha256 = _validate_sha256(
        training_provenance.get("contact_sidecar_manifest_sha256"),
        path="training_provenance.contact_sidecar_manifest_sha256",
    )
    declared_motion_manifest_sha256 = _validate_sha256(
        training_provenance.get("motion_shard_manifest_sha256"),
        path="training_provenance.motion_shard_manifest_sha256",
    )
    verified_training_motion_manifest_sha256 = _validate_sha256(
        verified_training_motion_manifest_sha256,
        path="verified_training_motion_manifest_sha256",
    )
    if verified_training_motion_manifest_sha256 != declared_motion_manifest_sha256:
        raise PolicyContractError(
            "Selected motion bank has not been verified against the training motion manifest."
        )

    active_clip_id = motion_path.stem
    (
        actual_manifest_sha256,
        active_records,
        payloads,
        clip_dir,
        bank_motion_record,
    ) = _verified_contact_manifest(
        motion_bank_dir=motion_bank_dir,
        contact_root=contact_root,
        active_clip_id=active_clip_id,
    )
    if actual_manifest_sha256 != declared_manifest_sha256:
        raise PolicyContractError(
            "Contact sidecar bank does not match the digest saved in training provenance: "
            f"declared={declared_manifest_sha256}, actual={actual_manifest_sha256}."
        )
    if (
        bank_motion_record["sha256"] != source_motion_sha256
        or int(bank_motion_record["size"]) != source_motion_size
    ):
        raise PolicyContractError(
            "Selected motion bytes do not match the same-named member of the motion bank that "
            "defined the training contact manifest: "
            f"selected_sha256={source_motion_sha256}, selected_size={source_motion_size}, "
            f"bank_record={bank_motion_record}."
        )

    interval, source_files, _clip_metadata, fps_key, source_fps = _parse_active_runtime_payloads(
        active_clip_id=active_clip_id,
        clip_dir=clip_dir,
        payloads=payloads,
    )
    experiment = metadata.get("experiment_config")
    command = experiment.get("command") if isinstance(experiment, Mapping) else None
    setup_terms = command.get("setup_terms") if isinstance(command, Mapping) else None
    motion_command = setup_terms.get("motion_command") if isinstance(setup_terms, Mapping) else None
    params = motion_command.get("params") if isinstance(motion_command, Mapping) else None
    motion_cfg = params.get("motion_config") if isinstance(params, Mapping) else None
    compensation = (
        motion_cfg.get("contact_interval_runtime_prepend_compensation", False)
        if isinstance(motion_cfg, Mapping)
        else False
    )
    if not isinstance(compensation, bool):
        raise PolicyContractError(
            "motion_config.contact_interval_runtime_prepend_compensation must be boolean."
        )
    transition_contract = embedded_motion_timeline_contract_from_metadata(metadata)
    transition_sha256 = (
        transition_contract["motion_transition_contract_sha256"]
        if transition_contract is not None
        else metadata.get("motion_transition_contract_sha256")
    )
    transition_sha256 = _validate_sha256(
        transition_sha256,
        path="motion_transition_contract_sha256",
        optional=True,
    )

    contract = {
        "version": EMBEDDED_CONTACT_SIDECAR_CONTRACT_VERSION,
        "binding": "training_contact_manifest_verified",
        "clip_id": active_clip_id,
        "source_motion_sha256": source_motion_sha256,
        "source_motion_size": source_motion_size,
        "source_frame_count": source_frame_count,
        "motion_bank_member": bank_motion_record,
        "training_contact_sidecar_manifest_sha256": declared_manifest_sha256,
        "training_motion_shard_manifest_sha256": declared_motion_manifest_sha256,
        "motion_transition_contract_sha256": transition_sha256,
        "contact_interval_runtime_prepend_compensation": compensation,
        "selected_raw_interval": [int(interval[0]), int(interval[1])],
        "interval_source_files": source_files,
        "metadata_file": "metadata.json" if "metadata.json" in active_records else None,
        "contact_interval_fps_key": fps_key,
        "contact_interval_fps": source_fps,
        "motion_fps": motion_fps,
        "active_files": active_records,
    }
    canonical = _validate_contract_fields(contract)
    return canonical, _canonical_contract_sha256(canonical)


def _validate_contract_fields(raw_contract: Any) -> dict[str, Any]:
    expected_keys = {
        "version",
        "binding",
        "clip_id",
        "source_motion_sha256",
        "source_motion_size",
        "source_frame_count",
        "motion_bank_member",
        "training_contact_sidecar_manifest_sha256",
        "training_motion_shard_manifest_sha256",
        "motion_transition_contract_sha256",
        "contact_interval_runtime_prepend_compensation",
        "selected_raw_interval",
        "interval_source_files",
        "metadata_file",
        "contact_interval_fps_key",
        "contact_interval_fps",
        "motion_fps",
        "active_files",
    }
    if not isinstance(raw_contract, Mapping) or set(raw_contract) != expected_keys:
        actual = set(raw_contract) if isinstance(raw_contract, Mapping) else set()
        raise PolicyContractError(
            "embedded_contact_sidecar_contract must contain exactly "
            f"{sorted(expected_keys)}; missing={sorted(expected_keys - actual)}, "
            f"unexpected={sorted(repr(key) for key in actual - expected_keys)}."
        )
    version = raw_contract["version"]
    if isinstance(version, bool) or not isinstance(version, Integral) or int(version) != 1:
        raise PolicyContractError("embedded_contact_sidecar_contract.version must equal integer 1.")
    if raw_contract["binding"] != "training_contact_manifest_verified":
        raise PolicyContractError(
            "embedded_contact_sidecar_contract.binding must equal "
            "'training_contact_manifest_verified'."
        )
    clip_id = raw_contract["clip_id"]
    if not isinstance(clip_id, str) or not clip_id.strip() or clip_id != clip_id.strip():
        raise PolicyContractError("embedded_contact_sidecar_contract.clip_id must be non-empty.")
    source_motion_sha256 = _validate_sha256(
        raw_contract["source_motion_sha256"],
        path="embedded_contact_sidecar_contract.source_motion_sha256",
    )
    source_motion_size = raw_contract["source_motion_size"]
    if (
        isinstance(source_motion_size, bool)
        or not isinstance(source_motion_size, Integral)
        or int(source_motion_size) <= 0
    ):
        raise PolicyContractError(
            "embedded_contact_sidecar_contract.source_motion_size must be a positive integer."
        )
    source_motion_size = int(source_motion_size)
    bank_member = raw_contract["motion_bank_member"]
    if not isinstance(bank_member, Mapping) or set(bank_member) != {"name", "sha256", "size"}:
        raise PolicyContractError(
            "embedded_contact_sidecar_contract.motion_bank_member is malformed."
        )
    bank_member_name = bank_member["name"]
    if bank_member_name != f"{clip_id}.npz":
        raise PolicyContractError(
            "embedded_contact_sidecar_contract.motion_bank_member has the wrong filename."
        )
    bank_member_sha256 = _validate_sha256(
        bank_member["sha256"],
        path="embedded_contact_sidecar_contract.motion_bank_member.sha256",
    )
    bank_member_size = bank_member["size"]
    if (
        isinstance(bank_member_size, bool)
        or not isinstance(bank_member_size, Integral)
        or int(bank_member_size) <= 0
    ):
        raise PolicyContractError(
            "embedded_contact_sidecar_contract.motion_bank_member.size must be positive."
        )
    bank_member_size = int(bank_member_size)
    if bank_member_sha256 != source_motion_sha256 or bank_member_size != source_motion_size:
        raise PolicyContractError(
            "Embedded motion-bank membership evidence does not match the selected motion."
        )
    manifest_sha256 = _validate_sha256(
        raw_contract["training_contact_sidecar_manifest_sha256"],
        path="embedded_contact_sidecar_contract.training_contact_sidecar_manifest_sha256",
    )
    motion_manifest_sha256 = _validate_sha256(
        raw_contract["training_motion_shard_manifest_sha256"],
        path="embedded_contact_sidecar_contract.training_motion_shard_manifest_sha256",
    )
    transition_sha256 = _validate_sha256(
        raw_contract["motion_transition_contract_sha256"],
        path="embedded_contact_sidecar_contract.motion_transition_contract_sha256",
        optional=True,
    )
    source_frame_count = raw_contract["source_frame_count"]
    if (
        isinstance(source_frame_count, bool)
        or not isinstance(source_frame_count, Integral)
        or int(source_frame_count) <= 0
    ):
        raise PolicyContractError(
            "embedded_contact_sidecar_contract.source_frame_count must be a positive integer."
        )
    compensation = raw_contract["contact_interval_runtime_prepend_compensation"]
    if not isinstance(compensation, bool):
        raise PolicyContractError(
            "embedded_contact_sidecar_contract contact compensation must be boolean."
        )
    interval = raw_contract["selected_raw_interval"]
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise PolicyContractError(
            "embedded_contact_sidecar_contract.selected_raw_interval must have two integers."
        )
    normalized_interval = normalize_contact_interval(interval)
    if normalized_interval is None or any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in interval
    ):
        raise PolicyContractError(
            "embedded_contact_sidecar_contract.selected_raw_interval is invalid."
        )

    active_files_raw = raw_contract["active_files"]
    if not isinstance(active_files_raw, Mapping) or not active_files_raw:
        raise PolicyContractError(
            "embedded_contact_sidecar_contract.active_files must be a non-empty mapping."
        )
    active_files: dict[str, dict[str, Any]] = {}
    for raw_name, raw_record in active_files_raw.items():
        if (
            not isinstance(raw_name, str)
            or not raw_name
            or Path(raw_name).name != raw_name
            or raw_name in {".", ".."}
        ):
            raise PolicyContractError("Contact-sidecar file names must be plain basenames.")
        if not isinstance(raw_record, Mapping) or set(raw_record) != {"sha256", "size"}:
            raise PolicyContractError(
                f"Contact-sidecar file record {raw_name!r} is malformed."
            )
        file_sha = _validate_sha256(
            raw_record["sha256"],
            path=f"embedded_contact_sidecar_contract.active_files[{raw_name!r}].sha256",
        )
        size = raw_record["size"]
        if isinstance(size, bool) or not isinstance(size, Integral) or int(size) <= 0:
            raise PolicyContractError(
                f"Contact-sidecar file {raw_name!r} must have a positive integer size."
            )
        active_files[raw_name] = {"sha256": file_sha, "size": int(size)}
    missing_required_files = set(_REQUIRED_CONTACT_FILES).difference(active_files)
    if missing_required_files:
        raise PolicyContractError(
            "Embedded active contact-sidecar records are missing training-manifest files: "
            f"{sorted(missing_required_files)}."
        )

    source_files_raw = raw_contract["interval_source_files"]
    if (
        not isinstance(source_files_raw, (list, tuple))
        or not source_files_raw
        or not all(isinstance(name, str) and name in active_files for name in source_files_raw)
        or list(source_files_raw) != sorted(set(source_files_raw))
    ):
        raise PolicyContractError(
            "embedded_contact_sidecar_contract.interval_source_files must be a sorted, unique, "
            "non-empty subset of active_files."
        )
    source_files = list(source_files_raw)
    allowed_fallback_files = set(CONTACT_INTERVAL_FALLBACK_FILES.values())
    if source_files != ["contact_intervals.json"] and not set(source_files).issubset(
        allowed_fallback_files
    ):
        raise PolicyContractError(
            "embedded_contact_sidecar_contract.interval_source_files must select either "
            "contact_intervals.json or known per-region interval NPY files."
        )
    metadata_file = raw_contract["metadata_file"]
    if metadata_file not in {None, "metadata.json"} or (
        metadata_file is not None and metadata_file not in active_files
    ) or (("metadata.json" in active_files) != (metadata_file == "metadata.json")):
        raise PolicyContractError("embedded_contact_sidecar_contract.metadata_file is invalid.")
    fps_key = raw_contract["contact_interval_fps_key"]
    fps_value = raw_contract["contact_interval_fps"]
    if fps_key not in {None, "contact_interval_fps", "fps"}:
        raise PolicyContractError(
            "embedded_contact_sidecar_contract.contact_interval_fps_key is invalid."
        )
    if fps_key is None:
        if fps_value is not None:
            raise PolicyContractError("A null contact FPS key requires a null contact FPS value.")
    else:
        if metadata_file is None:
            raise PolicyContractError("Contact FPS provenance requires metadata.json.")
        if fps_value is not None:
            fps_value = _validate_positive_fps(
                fps_value,
                path="embedded_contact_sidecar_contract.contact_interval_fps",
            )
    motion_fps = _validate_positive_fps(
        raw_contract["motion_fps"],
        path="embedded_contact_sidecar_contract.motion_fps",
    )
    return {
        "version": 1,
        "binding": "training_contact_manifest_verified",
        "clip_id": clip_id,
        "source_motion_sha256": source_motion_sha256,
        "source_motion_size": source_motion_size,
        "source_frame_count": int(source_frame_count),
        "motion_bank_member": {
            "name": bank_member_name,
            "sha256": bank_member_sha256,
            "size": bank_member_size,
        },
        "training_contact_sidecar_manifest_sha256": manifest_sha256,
        "training_motion_shard_manifest_sha256": motion_manifest_sha256,
        "motion_transition_contract_sha256": transition_sha256,
        "contact_interval_runtime_prepend_compensation": compensation,
        "selected_raw_interval": [normalized_interval[0], normalized_interval[1]],
        "interval_source_files": source_files,
        "metadata_file": metadata_file,
        "contact_interval_fps_key": fps_key,
        "contact_interval_fps": fps_value,
        "motion_fps": motion_fps,
        "active_files": active_files,
    }


def embedded_contact_sidecar_contract_from_metadata(
    metadata: Mapping[str, Any],
    *,
    required: bool = False,
) -> dict[str, Any] | None:
    raw_contract = metadata.get(EMBEDDED_CONTACT_SIDECAR_CONTRACT_KEY)
    declared_digest = metadata.get(EMBEDDED_CONTACT_SIDECAR_CONTRACT_SHA256_KEY)
    if raw_contract is None and declared_digest is None:
        if required:
            raise PolicyContractError(
                "Policy artifact is missing embedded_contact_sidecar_contract provenance."
            )
        return None
    if raw_contract is None or declared_digest is None:
        raise PolicyContractError(
            "Embedded contact-sidecar metadata must include both its contract and SHA-256."
        )
    contract = _validate_contract_fields(raw_contract)
    digest = _validate_sha256(
        declared_digest,
        path=EMBEDDED_CONTACT_SIDECAR_CONTRACT_SHA256_KEY,
    )
    computed = _canonical_contract_sha256(contract)
    if digest != computed:
        raise PolicyContractError(
            "Embedded contact-sidecar contract SHA-256 does not match its serialized contract: "
            f"declared={digest}, computed={computed}."
        )

    training_provenance = metadata.get("training_provenance")
    if not isinstance(training_provenance, Mapping):
        raise PolicyContractError(
            "Embedded contact-sidecar contract requires training_provenance metadata."
        )
    training_manifest_sha256 = _validate_sha256(
        training_provenance.get("contact_sidecar_manifest_sha256"),
        path="training_provenance.contact_sidecar_manifest_sha256",
    )
    if training_manifest_sha256 != contract["training_contact_sidecar_manifest_sha256"]:
        raise PolicyContractError(
            "Embedded contact-sidecar contract is bound to a different training contact manifest."
        )
    training_motion_manifest_sha256 = _validate_sha256(
        training_provenance.get("motion_shard_manifest_sha256"),
        path="training_provenance.motion_shard_manifest_sha256",
    )
    if training_motion_manifest_sha256 != contract["training_motion_shard_manifest_sha256"]:
        raise PolicyContractError(
            "Embedded contact-sidecar contract is bound to a different training motion manifest."
        )

    timeline_contract = embedded_motion_timeline_contract_from_metadata(metadata)
    if timeline_contract is not None:
        if contract["source_motion_sha256"] != timeline_contract["source_motion_sha256"]:
            raise PolicyContractError(
                "Embedded contact-sidecar contract is bound to a different source motion."
            )
        if contract["source_frame_count"] != timeline_contract["source_frame_count"]:
            raise PolicyContractError(
                "Embedded contact-sidecar contract source frame count contradicts the motion timeline."
            )
        if (
            contract["motion_transition_contract_sha256"]
            != timeline_contract["motion_transition_contract_sha256"]
        ):
            raise PolicyContractError(
                "Embedded contact-sidecar contract is bound to a different motion transition contract."
            )
    else:
        declared_transition_sha256 = _validate_sha256(
            metadata.get("motion_transition_contract_sha256"),
            path="motion_transition_contract_sha256",
            optional=True,
        )
        if declared_transition_sha256 != contract["motion_transition_contract_sha256"]:
            raise PolicyContractError(
                "Embedded contact-sidecar contract is bound to a different motion transition contract."
            )

    experiment = metadata.get("experiment_config")
    command = experiment.get("command") if isinstance(experiment, Mapping) else None
    setup_terms = command.get("setup_terms") if isinstance(command, Mapping) else None
    motion_command = setup_terms.get("motion_command") if isinstance(setup_terms, Mapping) else None
    params = motion_command.get("params") if isinstance(motion_command, Mapping) else None
    motion_cfg = params.get("motion_config") if isinstance(params, Mapping) else None
    if not isinstance(motion_cfg, Mapping):
        raise PolicyContractError(
            "Embedded contact-sidecar contract requires serialized motion_config metadata."
        )
    motion_file = motion_cfg.get("motion_file")
    motion_clip_name = motion_cfg.get("motion_clip_name")
    motion_clip_id = motion_cfg.get("motion_clip_id")
    if (
        not isinstance(motion_file, str)
        or Path(motion_file).suffix.lower() != ".npz"
        or Path(motion_file).stem != contract["clip_id"]
        or motion_clip_name != contract["clip_id"]
        or isinstance(motion_clip_id, bool)
        or motion_clip_id != 0
    ):
        raise PolicyContractError(
            "Embedded contact-sidecar clip contradicts the patched motion selection metadata."
        )
    compensation = (
        motion_cfg.get("contact_interval_runtime_prepend_compensation", False)
    )
    if not isinstance(compensation, bool):
        raise PolicyContractError(
            "motion_config.contact_interval_runtime_prepend_compensation must be boolean."
        )
    if compensation != contract["contact_interval_runtime_prepend_compensation"]:
        raise PolicyContractError(
            "Embedded contact-sidecar compensation contradicts the serialized motion config."
        )
    return contract
