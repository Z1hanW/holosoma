#!/usr/bin/env python3
"""Compute immutable training-input provenance before simulator startup."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import sys
import threading
from pathlib import Path
from typing import Any, Callable

import torch

from holosoma.utils.checkpoint_validation import load_verified_torch_checkpoint
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rank_local_shards import validate_rank_local_shard_manifest
from holosoma.utils.runtime_asset_manifest import build_urdf_asset_manifest
from holosoma.utils.training_provenance import (
    EXECUTION_RUNTIME_KEY,
    MOTION_GENERATOR_TEACHER_SHA256_KEY,
    PROVENANCE_VERSION,
    REQUIRE_MOTION_GENERATOR_TEACHER_MATCH_KEY,
    RUNTIME_ASSET_DIGEST_KEY,
    RUNTIME_ASSET_MANIFEST_KEY,
    RUNTIME_ASSET_PHASE_KEY,
    RUNTIME_ASSET_PHASE_PENDING,
    SEMANTIC_ENVIRONMENT_FIELDS,
    SEMANTIC_ENVIRONMENT_KEY,
    TEACHER_ENABLED_KEY,
    TRAINING_REGIME_DISTILLATION,
    TRAINING_REGIME_KEY,
    TRAINING_REGIME_PURE_RL,
    disabled_checkpoint_sha256,
    disabled_contact_sidecar_manifest_sha256,
    disabled_teacher_sha256,
    normalized_execution_bool_from_environ,
    normalized_execution_int_from_environ,
    pending_runtime_asset_manifest_sha256,
    semantic_environment_from_environ,
    validate_hierarchical_small_collectives_contract,
    validate_training_provenance,
)


REQUIRED_CONTACT_FILES = (
    "teacher_rollout_reference.npz",
    "left_wrist_contact_points.npy",
    "left_wrist_contact_point_counts.npy",
    "left_wrist_contact_interval_steps.npy",
    "right_wrist_contact_points.npy",
    "right_wrist_contact_point_counts.npy",
    "right_wrist_contact_interval_steps.npy",
)
OPTIONAL_RUNTIME_CONTACT_FILES = (
    "contact_intervals.json",
    "metadata.json",
    "teacher_rollout_reference.npz",
)
CONTACT_SIDECAR_MODES = ("full-sidecars", "runtime-intervals")
RUNTIME_CONTACT_REGIONS = (
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
RUNTIME_CONTACT_REGION_FILE_SUFFIXES = (
    "_contact_points.npy",
    "_contact_point_counts.npy",
    "_contact_interval_steps.npy",
    "_contact_active_mask.npy",
)
SOURCE_SNAPSHOT_ID_ENV = "HOLOSOMA_SOURCE_SNAPSHOT_ID"
SOURCE_MANIFEST_SHA256_ENV = "HOLOSOMA_SOURCE_MANIFEST_SHA256"
FORMAL_GIT_VERIFICATION_PATH_ENV = "HOLOSOMA_FORMAL_GIT_VERIFICATION_PATH"
PYTHON_RUNTIME_MANIFEST_SHA256_ENV = "HOLOSOMA_PYTHON_RUNTIME_MANIFEST_SHA256"
PYTHON_RUNTIME_SITEPACKAGES_ENV = "PYTHON_RUNTIME_SITEPACKAGES"
_SOURCE_SNAPSHOT_ID_RE = re.compile(r"^src-([0-9a-f]{64})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_ID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_DATA_ASSET_CACHE_VERSION = 1
_DATA_ASSET_CACHE_ROOT_ENV = "HOLOSOMA_DATA_PROVENANCE_CACHE_ROOT"
_DATA_ASSET_CACHE_THREAD_LOCKS: dict[str, threading.Lock] = {}
_DATA_ASSET_CACHE_THREAD_LOCKS_GUARD = threading.Lock()

_EXECUTION_BOOL_ENVS: tuple[tuple[str, bool], ...] = (
    ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", False),
    ("HOLOSOMA_GLOO_BARRIER", False),
    ("HOLOSOMA_GLOO_GRAD_REDUCE", False),
    ("HOLOSOMA_GLOO_SMALL_COLLECTIVES", False),
    ("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", False),
    ("HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES", False),
    ("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER", False),
    ("HOLOSOMA_RANK_VISIBLE_DEVICES", False),
    ("HOLOSOMA_CONTIGUOUS_MINIBATCHES", False),
    ("HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY", False),
    ("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", False),
    ("HOLOSOMA_DAGGER_SUPERVISED_ONLY", False),
    ("HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD", False),
    ("HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC", False),
)


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat_result = path.stat()
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
    )


def sha256_file(
    path: Path,
    *,
    identity_recorder: Callable[[Path], None] | None = None,
) -> str:
    unresolved_path = path.expanduser()
    if identity_recorder is not None:
        identity_recorder(unresolved_path)
    path = unresolved_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"provenance input does not exist or is not a regular file: {path}")
    before_identity = _file_identity(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after_identity = _file_identity(path)
    if before_identity != after_identity:
        raise RuntimeError(f"provenance input changed while being hashed: {path}")
    if identity_recorder is not None:
        identity_recorder(path)
    return digest.hexdigest()


def _hashed_file_record(
    path: Path,
    *,
    identity_recorder: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    identity = _file_identity(path)
    digest = sha256_file(path, identity_recorder=identity_recorder)
    if _file_identity(path) != identity:
        raise RuntimeError(f"provenance input changed while its record was being computed: {path}")
    return {"name": path.name, "size": identity[2], "sha256": digest}


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _disabled_runtime_component_sha256(component: str) -> str:
    """Return an unambiguous digest sentinel for a disabled runtime component."""

    return sha256_json({"version": 1, "component": component, "disabled": True})


def _normalized_bool_env(name: str, *, default: bool) -> bool:
    return normalized_execution_bool_from_environ(
        os.environ,
        name,
        default=default,
    )


def _normalized_nonnegative_int_env(name: str, *, default: int) -> int:
    return normalized_execution_int_from_environ(
        os.environ,
        name,
        default=default,
        minimum=0,
    )


def _normalized_python_hash_seed() -> str:
    raw_value = os.environ.get("PYTHONHASHSEED")
    if raw_value is None or not raw_value.strip():
        return "<unset>"
    value = raw_value.strip()
    if not value.isdecimal():
        raise ValueError(
            f"PYTHONHASHSEED must be an integer in [0, 4294967295], got {raw_value!r}"
        )
    parsed = int(value, 10)
    if not 0 <= parsed <= 4294967295:
        raise ValueError(
            f"PYTHONHASHSEED must be an integer in [0, 4294967295], got {raw_value!r}"
        )
    return str(parsed)


def _normalized_cublas_workspace_config() -> str:
    raw_value = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if raw_value is None or not raw_value.strip():
        return "<unset>"
    value = raw_value.strip()
    if value not in {":4096:8", ":16:8"}:
        raise ValueError(
            "CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8, "
            f"got {raw_value!r}"
        )
    return value


def _semantic_environment_metadata() -> dict[str, str | None]:
    """Capture a fixed set of raw training-semantic environment controls.

    Normalizing these values to booleans or numbers would collapse an unset
    variable into an explicit false/zero override.  That distinction is useful
    audit evidence and is intentionally part of the exact resume contract.
    """

    return semantic_environment_from_environ()


def _execution_runtime_metadata() -> dict[str, Any]:
    """Capture normalized launch controls that can change training numerics."""

    backend = os.environ.get("TORCH_DIST_BACKEND", "nccl").strip().lower()
    if backend not in {"nccl", "gloo"}:
        raise ValueError(
            f"TORCH_DIST_BACKEND must be nccl or gloo, got {os.environ.get('TORCH_DIST_BACKEND')!r}"
        )

    hierarchical_grad_reduce = _normalized_bool_env(
        "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE",
        default=False,
    )
    nccl_runtime_required = backend == "nccl" or hierarchical_grad_reduce
    raw_nccl_lib_sha256 = os.environ.get("NCCL_LIB_SHA256", "")
    if raw_nccl_lib_sha256 != raw_nccl_lib_sha256.strip():
        raise ValueError(
            "NCCL_LIB_SHA256 must not contain surrounding whitespace, "
            f"got {raw_nccl_lib_sha256!r}"
        )
    nccl_lib_sha256 = raw_nccl_lib_sha256
    if nccl_lib_sha256:
        if _SHA256_RE.fullmatch(nccl_lib_sha256) is None:
            raise ValueError("NCCL_LIB_SHA256 must be a 64-character lowercase SHA256 hex digest")
    elif nccl_runtime_required:
        raise ValueError(
            "NCCL_LIB_SHA256 is required when the default backend or hierarchical local gradient "
            "reduction uses NCCL, so the collective runtime is immutable"
        )
    else:
        nccl_lib_sha256 = _disabled_runtime_component_sha256("nccl_library")

    nproc = _normalized_nonnegative_int_env("NPROC", default=1)
    if nproc < 1:
        raise ValueError(f"NPROC must be a positive integer, got {os.environ.get('NPROC')!r}")
    nnodes = _normalized_nonnegative_int_env("NNODES", default=1)
    if nnodes < 1:
        raise ValueError(f"NNODES must be a positive integer, got {os.environ.get('NNODES')!r}")
    hierarchical_pg_timeout_sec = _normalized_nonnegative_int_env(
        "HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC",
        default=300,
    )
    if hierarchical_pg_timeout_sec < 1:
        raise ValueError(
            "HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC must be a positive integer, "
            f"got {os.environ.get('HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC')!r}"
        )

    metadata: dict[str, Any] = {
        "NCCL_LIB_SHA256": nccl_lib_sha256,
        "TORCH_DIST_BACKEND": backend,
        # Both values are read before training code runs.  Recording them is
        # necessary because Python hash ordering and cuBLAS workspace choice
        # can change otherwise identical resume numerics.
        "PYTHONHASHSEED": _normalized_python_hash_seed(),
        "CUBLAS_WORKSPACE_CONFIG": _normalized_cublas_workspace_config(),
        # Collective ordering and gradient averaging depend on both the
        # node-local and global process topology.  Keeping NPROC alone would
        # permit an exact-resume check to miss a changed global world size.
        # Rank-visible launch also changes each worker's CUDA visibility.
        "NPROC": nproc,
        "NNODES": nnodes,
        "HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC": hierarchical_pg_timeout_sec,
        "HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH": _normalized_nonnegative_int_env(
            "HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH",
            default=0,
        ),
        SEMANTIC_ENVIRONMENT_KEY: _semantic_environment_metadata(),
    }
    for name, default in _EXECUTION_BOOL_ENVS:
        metadata[name] = _normalized_bool_env(name, default=default)
    validate_hierarchical_small_collectives_contract(metadata)
    return metadata


def _optional_checkpoint_digest(path: Path | None, *, role: str) -> str:
    if path is None:
        return disabled_checkpoint_sha256(role)
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} checkpoint does not exist: {resolved}")
    return sha256_file(resolved)


def _teacher_contract_and_digest(checkpoint_path: Path) -> tuple[bool, bool, str]:
    # Derive semantics and identity from the same stable descriptor.  This
    # prevents a mutable teacher path from naming one payload during contract
    # inspection and another when provenance is hashed.
    checkpoint, checkpoint_sha256 = load_verified_torch_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("teacher checkpoint payload must be a dictionary")
    config = checkpoint.get("experiment_config")
    if not isinstance(config, dict):
        raise ValueError("teacher checkpoint has no experiment_config metadata")

    termination_terms = config.get("termination", {}).get("terms", {})
    if not isinstance(termination_terms, dict):
        termination_terms = {}
    teacher_motion_ends = any(
        str(name) == "motion_ends"
        or str(term.get("func", "")).rsplit(":", 1)[-1] == "motion_ends"
        for name, term in termination_terms.items()
        if isinstance(term, dict)
    )

    try:
        actor_inputs = config["algo"]["config"]["module_dict"]["actor"]["input_dim"]
        groups = config["observation"]["groups"]
    except (KeyError, TypeError) as exc:
        raise ValueError("teacher checkpoint lacks actor-input/observation metadata") from exc
    if isinstance(actor_inputs, str):
        actor_inputs = [actor_inputs]
    if not isinstance(actor_inputs, (list, tuple)) or not isinstance(groups, dict):
        raise ValueError("teacher checkpoint has invalid actor-input/observation metadata")

    teacher_uses_actions = False
    for group_name in actor_inputs:
        group = groups.get(str(group_name))
        if not isinstance(group, dict):
            continue
        terms = group.get("terms", {})
        if not isinstance(terms, dict):
            continue
        for term_name, term in terms.items():
            func = str(term.get("func", "")) if isinstance(term, dict) else ""
            if str(term_name) == "actions" or func.rsplit(":", 1)[-1] == "actions":
                teacher_uses_actions = True
                break
        if teacher_uses_actions:
            break
    return teacher_motion_ends, teacher_uses_actions, checkpoint_sha256


def _teacher_contract(checkpoint_path: Path) -> tuple[bool, bool]:
    """Compatibility wrapper used by focused contract tests."""

    teacher_motion_ends, teacher_uses_actions, _digest = _teacher_contract_and_digest(
        checkpoint_path
    )
    return teacher_motion_ends, teacher_uses_actions


def _resolve_local_asset(
    raw_path: str,
    *,
    base_dir: Path,
    role: str,
    identity_recorder: Callable[[Path], None] | None = None,
) -> Path:
    value = str(raw_path).strip()
    if not value:
        raise ValueError(f"{role} contains an empty asset path")
    if value.startswith(("http://", "https://", "package://", "file://")):
        raise ValueError(f"{role} uses an unsupported non-local asset URI: {value}")
    if value.startswith("holosoma/data"):
        path = Path(resolve_data_file_path(value)).expanduser()
    else:
        path = Path(value).expanduser()
    if not path.is_absolute() and not value.startswith("holosoma/data"):
        path = base_dir / path
    if identity_recorder is not None:
        identity_recorder(path)
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{role} asset does not exist: {path}")
    return path


def _object_asset_manifest(
    object_map: Path,
    *,
    active_motion_ids: set[str],
    identity_recorder: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    if identity_recorder is not None:
        identity_recorder(object_map)
    object_map_identity = _file_identity(object_map)
    try:
        payload = json.loads(object_map.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"failed to parse object map {object_map}: {exc}") from exc
    clips = payload.get("clips", payload) if isinstance(payload, dict) else None
    if not isinstance(clips, dict):
        raise ValueError(f"object map must contain a clips mapping: {object_map}")

    assets_by_clip: dict[str, Any] = {}
    missing_entries = sorted(active_motion_ids.difference(str(clip_id) for clip_id in clips))
    if missing_entries:
        raise ValueError(f"object map is missing active motion clips: {missing_entries[:20]}")

    for clip_id in sorted(active_motion_ids):
        raw_entry = clips[clip_id]
        entry = {"object_urdf_path": raw_entry} if isinstance(raw_entry, str) else raw_entry
        if not isinstance(entry, dict):
            raise ValueError(f"invalid object-map entry for clip {clip_id!r}")
        field_assets: dict[str, Any] = {}
        field = "object_urdf_path"
        raw_value = entry.get(field)
        if raw_value is not None and not isinstance(raw_value, str):
            raise ValueError(f"clip {clip_id!r} {field} must be a string")
        values = [raw_value.strip()] if isinstance(raw_value, str) and raw_value.strip() else []
        if not values:
            raise ValueError(f"active clip {clip_id!r} has no {field}")
        records: list[dict[str, Any]] = []
        for index, value in enumerate(values):
            urdf_path = _resolve_local_asset(
                value,
                base_dir=object_map.parent,
                role=f"clip {clip_id!r} {field}[{index}]",
                identity_recorder=identity_recorder,
            )
            if urdf_path.suffix.lower() != ".urdf":
                raise ValueError(f"clip {clip_id!r} object asset is not a URDF: {urdf_path}")
            record = build_urdf_asset_manifest(
                urdf_path,
                role=f"clip {clip_id!r} {field}[{index}]",
                asset_root=urdf_path.parent,
                reference=urdf_path.name,
                require_mesh=True,
                identity_recorder=identity_recorder,
            )
            records.append(record)
        field_assets[field] = records
        assets_by_clip[str(clip_id)] = field_assets
    result = {"version": 2, "clips": assets_by_clip}
    if _file_identity(object_map) != object_map_identity:
        raise RuntimeError(f"object map changed while its asset closure was being computed: {object_map}")
    return result


def _motion_manifest_digest(
    motion_dir: Path,
    object_map: Path,
    shard_manifest: Path | None,
    *,
    identity_recorder: Callable[[Path], None] | None = None,
) -> str:
    if identity_recorder is not None:
        identity_recorder(motion_dir)
        identity_recorder(object_map)
        if shard_manifest is not None:
            identity_recorder(shard_manifest)
    motion_dir = motion_dir.expanduser().resolve()
    object_map = object_map.expanduser().resolve()
    shard_manifest = shard_manifest.expanduser().resolve() if shard_manifest is not None else None
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"motion directory does not exist: {motion_dir}")
    motion_dir_identity = _file_identity(motion_dir)
    motion_files = sorted(motion_dir.glob("*.npz"))
    if not motion_files:
        raise ValueError(f"motion directory has no active .npz files: {motion_dir}")
    if not object_map.is_file():
        raise FileNotFoundError(f"object map does not exist: {object_map}")
    shard_manifest_sha256 = None
    if shard_manifest is not None:
        if not shard_manifest.is_file():
            raise FileNotFoundError(f"motion shard manifest does not exist: {shard_manifest}")
        # A scientific rank-local launch must bind provenance to a manifest
        # that closes over exact assignments and per-rank input bytes.  Merely
        # hashing a legacy assignment-only JSON would preserve the JSON but not
        # the local NPZ/object-map views actually consumed by workers.
        validate_rank_local_shard_manifest(
            shard_manifest,
            expected_clip_ids={path.stem for path in motion_files},
        )
        shard_manifest_sha256 = sha256_file(
            shard_manifest,
            identity_recorder=identity_recorder,
        )
    object_map_identity = _file_identity(object_map)
    object_map_sha256 = sha256_file(
        object_map,
        identity_recorder=identity_recorder,
    )
    object_asset_manifest = _object_asset_manifest(
        object_map,
        active_motion_ids={path.stem for path in motion_files},
        identity_recorder=identity_recorder,
    )
    if _file_identity(object_map) != object_map_identity:
        raise RuntimeError(f"object map changed while motion provenance was being computed: {object_map}")
    conceptual_manifest = {
        "version": 3,
        "clip_files": [
            _hashed_file_record(path, identity_recorder=identity_recorder)
            for path in motion_files
        ],
        "object_map_sha256": object_map_sha256,
        "object_asset_manifest": object_asset_manifest,
        "shard_assignment_manifest_sha256": shard_manifest_sha256,
    }
    final_motion_files = sorted(motion_dir.glob("*.npz"))
    if final_motion_files != motion_files or _file_identity(motion_dir) != motion_dir_identity:
        raise RuntimeError(
            f"motion directory changed while its provenance was being computed: {motion_dir}"
        )
    return sha256_json(conceptual_manifest)


def _source_bundle_digest(source_root: Path) -> str:
    source_root = source_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")

    def collect_candidates() -> set[Path]:
        candidates: set[Path] = set()
        package_root = source_root / "src" / "holosoma" / "holosoma"
        inference_root = source_root / "src" / "holosoma_inference"
        scripts_root = source_root / "scripts"
        defm_root = source_root / "submodules" / "defm"
        if package_root.is_dir():
            candidates.update(
                path for path in package_root.rglob("*.py") if "__pycache__" not in path.parts
            )
        if inference_root.is_dir():
            candidates.update(
                path for path in inference_root.rglob("*.py") if "__pycache__" not in path.parts
            )
        if scripts_root.is_dir():
            candidates.update(
                path
                for path in scripts_root.rglob("*")
                if path.is_file()
                and path.suffix in {".py", ".sh"}
                and "__pycache__" not in path.parts
            )
        if defm_root.is_dir():
            candidates.update(
                path
                for path in defm_root.rglob("*")
                if path.is_file()
                and path.suffix in {".py", ".yaml", ".yml"}
                and ".git" not in path.parts
                and "__pycache__" not in path.parts
            )
        for pattern in ("distill*.sh", "train*.sh", "batch*.sh"):
            candidates.update(path for path in source_root.glob(pattern) if path.is_file())
        return candidates

    candidates = collect_candidates()
    if not candidates:
        raise ValueError(f"no source files found for provenance under {source_root}")
    manifest = []
    for path in sorted(candidates, key=lambda value: value.relative_to(source_root).as_posix()):
        record = _hashed_file_record(path)
        manifest.append(
            {
                "path": path.relative_to(source_root).as_posix(),
                "size": record["size"],
                "sha256": record["sha256"],
            }
        )
    final_candidates = collect_candidates()
    if final_candidates != candidates:
        raise RuntimeError(
            f"source bundle changed while its provenance was being computed: {source_root}"
        )
    return sha256_json({"version": 1, "files": manifest})


def _source_snapshot_identity_from_env() -> dict[str, str]:
    snapshot_id = os.environ.get(SOURCE_SNAPSHOT_ID_ENV)
    manifest_sha256 = os.environ.get(SOURCE_MANIFEST_SHA256_ENV)
    if (snapshot_id is None) != (manifest_sha256 is None):
        raise ValueError(
            f"{SOURCE_SNAPSHOT_ID_ENV} and {SOURCE_MANIFEST_SHA256_ENV} must be set together"
        )
    if snapshot_id is None:
        return {}

    assert manifest_sha256 is not None
    snapshot_match = _SOURCE_SNAPSHOT_ID_RE.fullmatch(snapshot_id)
    if snapshot_match is None:
        raise ValueError(
            f"{SOURCE_SNAPSHOT_ID_ENV} must have format src-<64 lowercase SHA256 hex>, got {snapshot_id!r}"
        )
    if _SHA256_RE.fullmatch(manifest_sha256) is None:
        raise ValueError(
            f"{SOURCE_MANIFEST_SHA256_ENV} must be a 64-character lowercase SHA256 hex digest, "
            f"got {manifest_sha256!r}"
        )
    snapshot_digest = snapshot_match.group(1)
    if snapshot_digest != manifest_sha256:
        raise ValueError(
            f"{SOURCE_SNAPSHOT_ID_ENV} digest does not match {SOURCE_MANIFEST_SHA256_ENV}: "
            f"{snapshot_digest} != {manifest_sha256}"
        )
    return {
        "source_snapshot_id": snapshot_id,
        "source_manifest_sha256": manifest_sha256,
    }


def _formal_git_identity_from_env() -> dict[str, Any]:
    """Bind a previously fail-closed clean-checkout verification into provenance."""

    raw_path = os.environ.get(FORMAL_GIT_VERIFICATION_PATH_ENV, "").strip()
    if not raw_path:
        return {}
    path = Path(raw_path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            f"{FORMAL_GIT_VERIFICATION_PATH_ENV} must name one regular non-symlink JSON file"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("formal Git verification payload must be a JSON object")
    required_true = (
        "tracked_diff_clean",
        "untracked_clean",
        "legacy_unmapped_gitlinks_inactive_and_empty",
    )
    for key in required_true:
        if payload.get(key) is not True:
            raise ValueError(f"formal Git verification requires {key}=true")
    for key in ("commit_sha", "tree_sha", "fetched_ref_commit"):
        value = payload.get(key)
        if not isinstance(value, str) or _GIT_OBJECT_ID_RE.fullmatch(value) is None:
            raise ValueError(f"formal Git verification has malformed {key}: {value!r}")
    if payload["commit_sha"] != payload["fetched_ref_commit"]:
        raise ValueError("formal Git commit is not the commit fetched from the declared remote ref")
    for key in ("remote_url", "remote_ref"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"formal Git verification has empty {key}")
    declared_submodules = payload.get("declared_submodules")
    if not isinstance(declared_submodules, list):
        raise ValueError("formal Git verification declared_submodules must be a list")
    return {
        "source_distribution": "direct_remote_git_clean_checkout",
        "git_remote_url": payload["remote_url"],
        "git_remote_ref": payload["remote_ref"],
        "git_commit_sha": payload["commit_sha"],
        "git_tree_sha": payload["tree_sha"],
        "git_fetched_ref_commit": payload["fetched_ref_commit"],
        "git_declared_submodules": declared_submodules,
        "git_checkout_tracked_diff_clean": True,
        "git_checkout_untracked_clean": True,
        "git_legacy_unmapped_gitlinks_inactive_and_empty": True,
    }


def _environment_metadata() -> dict[str, Any]:
    distributions = {}
    for name in (
        "torch",
        "isaacsim",
        "isaaclab",
        "numpy",
        "omegaconf",
        "antlr4-python3-runtime",
        "PyYAML",
        "attrs",
    ):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "packages": distributions,
        EXECUTION_RUNTIME_KEY: _execution_runtime_metadata(),
    }
    runtime_manifest_sha256 = os.environ.get(PYTHON_RUNTIME_MANIFEST_SHA256_ENV, "").strip()
    runtime_sitepackages = os.environ.get(PYTHON_RUNTIME_SITEPACKAGES_ENV, "").strip()
    if bool(runtime_manifest_sha256) != bool(runtime_sitepackages):
        raise ValueError(
            f"{PYTHON_RUNTIME_SITEPACKAGES_ENV} and {PYTHON_RUNTIME_MANIFEST_SHA256_ENV} "
            "must be set together or both be disabled"
        )
    if runtime_manifest_sha256:
        if _SHA256_RE.fullmatch(runtime_manifest_sha256) is None:
            raise ValueError(
                f"{PYTHON_RUNTIME_MANIFEST_SHA256_ENV} must be a 64-character lowercase SHA256 hex digest"
            )
    else:
        runtime_manifest_sha256 = _disabled_runtime_component_sha256("python_runtime_overlay")
    metadata["python_runtime_manifest_sha256"] = runtime_manifest_sha256
    return metadata


def _contact_directory_map(clips_root: Path, motion_ids: set[str]) -> dict[str, Path]:
    result: dict[str, list[Path]] = {}
    for directory in sorted(path for path in clips_root.iterdir() if path.is_dir()):
        name = directory.name
        if name in motion_ids:
            clip_id = name
        elif "_" in name and name.split("_", 1)[0].isdigit():
            clip_id = name.split("_", 1)[1]
        else:
            continue
        if clip_id in motion_ids:
            result.setdefault(clip_id, []).append(directory)
    duplicates = {clip_id: paths for clip_id, paths in result.items() if len(paths) != 1}
    if duplicates:
        raise ValueError(
            "contact export has duplicate directories for active clips: "
            + ", ".join(f"{clip_id}={paths}" for clip_id, paths in sorted(duplicates.items()))
        )
    missing = sorted(motion_ids.difference(result))
    if missing:
        raise ValueError(f"contact export is missing active clips: {missing[:20]}")
    return {clip_id: paths[0] for clip_id, paths in result.items()}


def _contact_manifest_digest(
    motion_dir: Path,
    contact_root: Path,
    *,
    contact_sidecar_mode: str = "full-sidecars",
    identity_recorder: Callable[[Path], None] | None = None,
) -> str:
    if contact_sidecar_mode not in CONTACT_SIDECAR_MODES:
        raise ValueError(
            f"unsupported contact sidecar mode: {contact_sidecar_mode!r}; "
            f"expected one of {CONTACT_SIDECAR_MODES!r}"
        )
    if identity_recorder is not None:
        identity_recorder(motion_dir)
        identity_recorder(contact_root)
    motion_dir = motion_dir.expanduser().resolve()
    contact_root = contact_root.expanduser().resolve()
    clips_root = contact_root / "clips" if (contact_root / "clips").is_dir() else contact_root
    if not clips_root.is_dir():
        raise FileNotFoundError(f"contact export root does not exist: {contact_root}")
    if identity_recorder is not None:
        identity_recorder(clips_root)
    clips_root_identity = _file_identity(clips_root)
    motion_ids = {path.stem for path in motion_dir.glob("*.npz")}
    if not motion_ids:
        raise ValueError(f"motion directory has no active clips: {motion_dir}")
    directory_by_clip = _contact_directory_map(clips_root, motion_ids)

    rollout_manifest_candidates = (
        contact_root / "realmesh_rollout_manifest.json",
        contact_root.parent / "realmesh_rollout_manifest.json",
        contact_root.parent.parent / "realmesh_rollout_manifest.json",
    )
    rollout_manifests = [
        candidate for candidate in rollout_manifest_candidates if candidate.is_file()
    ]
    if len({path.resolve() for path in rollout_manifests}) > 1:
        raise ValueError(
            "contact export has ambiguous teacher rollout manifests: "
            f"{[str(path) for path in rollout_manifests]!r}"
        )
    rollout_manifest_record = None
    if rollout_manifests:
        rollout_manifest_path = rollout_manifests[0]
        record = _hashed_file_record(
            rollout_manifest_path,
            identity_recorder=identity_recorder,
        )
        rollout_manifest_record = {
            "sha256": record["sha256"],
            "size": record["size"],
        }

    clips: dict[str, Any] = {}
    for clip_id in sorted(motion_ids):
        clip_dir = directory_by_clip[clip_id]
        if identity_recorder is not None:
            identity_recorder(clip_dir)
        clip_dir_identity = _file_identity(clip_dir)
        files: dict[str, Any] = {}
        file_names = (
            {"contact_intervals.json"}
            if contact_sidecar_mode == "runtime-intervals"
            else set(REQUIRED_CONTACT_FILES)
        )
        file_names.update(name for name in OPTIONAL_RUNTIME_CONTACT_FILES if (clip_dir / name).is_file())
        for region_name in RUNTIME_CONTACT_REGIONS:
            for suffix in RUNTIME_CONTACT_REGION_FILE_SUFFIXES:
                file_name = f"{region_name}{suffix}"
                if (clip_dir / file_name).is_file():
                    file_names.add(file_name)
        for file_name in sorted(file_names):
            path = clip_dir / file_name
            if not path.is_file():
                raise FileNotFoundError(f"missing contact/rollout sidecar: {clip_id}/{file_name}")
            record = _hashed_file_record(path, identity_recorder=identity_recorder)
            files[file_name] = {"sha256": record["sha256"], "size": record["size"]}
        if _file_identity(clip_dir) != clip_dir_identity:
            raise RuntimeError(
                f"contact sidecar directory changed while being hashed: {clip_dir}"
            )
        clips[clip_id] = files
    final_motion_ids = {path.stem for path in motion_dir.glob("*.npz")}
    if final_motion_ids != motion_ids:
        raise RuntimeError(
            f"motion directory changed while contact provenance was being computed: {motion_dir}"
        )
    if _file_identity(clips_root) != clips_root_identity:
        raise RuntimeError(
            f"contact export changed while its provenance was being computed: {clips_root}"
        )
    return sha256_json(
        {
            "version": 4,
            "contact_sidecar_mode": contact_sidecar_mode,
            "clips": clips,
            "teacher_rollout_manifest": rollout_manifest_record,
        }
    )


def revalidate_data_asset_provenance(
    provenance: dict[str, Any],
    *,
    motion_dir: Path,
    object_map: Path,
    contact_root: Path | None,
    motion_shard_manifest: Path | None,
    identity_recorder: Callable[[Path], None] | None = None,
) -> dict[str, str]:
    """Recompute mutable training-data inputs and compare them fail closed.

    The launch script computes these digests before Tyro/config preflights.  A
    scientific worker calls this again immediately before simulator startup
    and environment construction so a changed NPZ, object map, transitive URDF
    asset, contact sidecar, shard manifest, or retargeted symlink is rejected.
    """

    validated = validate_training_provenance(provenance, require_finalized=True)
    contact_sidecar_mode = str(provenance.get("contact_sidecar_mode", "full-sidecars"))
    motion_dir = motion_dir.expanduser().resolve()
    object_map = object_map.expanduser().resolve()
    contact_root = contact_root.expanduser().resolve() if contact_root is not None else None
    motion_shard_manifest = (
        motion_shard_manifest.expanduser().resolve()
        if motion_shard_manifest is not None
        else None
    )

    actual = {
        "motion_shard_manifest_sha256": _motion_manifest_digest(
            motion_dir,
            object_map,
            motion_shard_manifest,
            identity_recorder=identity_recorder,
        ),
        "contact_sidecar_manifest_sha256": (
            _contact_manifest_digest(
                motion_dir,
                contact_root,
                contact_sidecar_mode=contact_sidecar_mode,
                identity_recorder=identity_recorder,
            )
            if contact_root is not None
            else disabled_contact_sidecar_manifest_sha256()
        ),
    }
    mismatches = [
        f"{key}: declared={validated[key]} actual={digest}"
        for key, digest in actual.items()
        if validated[key] != digest
    ]
    if mismatches:
        raise RuntimeError(
            "training data asset provenance changed after launcher preflight:\n  - "
            + "\n  - ".join(mismatches)
        )
    return actual


def _runtime_global_rank() -> int:
    rank_raw = os.environ.get("RANK", "").strip()
    if not rank_raw:
        raise RuntimeError(
            "rank-local shard provenance revalidation requires torchrun RANK"
        )
    try:
        rank = int(rank_raw)
    except ValueError as exc:
        raise RuntimeError(f"invalid torchrun RANK={rank_raw!r}") from exc
    if rank < 0:
        raise RuntimeError(f"invalid negative torchrun RANK={rank}")
    return rank


def _revalidate_selected_rank_local_shard(
    motion_shard_manifest: Path,
    *,
    identity_recorder: Callable[[Path], None] | None = None,
) -> dict[str, str]:
    """Verify the exact rank-local NPZ/map view, not only its manifest JSON."""

    if identity_recorder is not None:
        identity_recorder(motion_shard_manifest)
    runtime_world_size_raw = os.environ.get("WORLD_SIZE", "").strip()
    try:
        expected_world_size = int(runtime_world_size_raw) if runtime_world_size_raw else None
    except ValueError as exc:
        raise RuntimeError(f"invalid torchrun WORLD_SIZE={runtime_world_size_raw!r}") from exc
    manifest = validate_rank_local_shard_manifest(
        motion_shard_manifest,
        expected_world_size=expected_world_size,
    )
    rank = _runtime_global_rank()
    matching = [record for record in manifest["shards"] if int(record["rank"]) == rank]
    if len(matching) != 1:
        raise RuntimeError(
            f"rank-local shard manifest has no unique record for global rank {rank}: "
            f"{motion_shard_manifest}"
        )
    expected = matching[0]
    shard_dir = motion_shard_manifest.expanduser().resolve().parent / f"rank_{rank}"
    if identity_recorder is not None:
        identity_recorder(shard_dir)
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"rank-local shard directory does not exist: {shard_dir}")

    object_map = shard_dir / "_clip_object_urdf_map.json"
    if identity_recorder is not None:
        identity_recorder(object_map)
    actual_object_map_sha256 = sha256_file(
        object_map,
        identity_recorder=identity_recorder,
    )
    if actual_object_map_sha256 != expected["object_map_sha256"]:
        raise RuntimeError(
            "rank-local object map content digest mismatch before simulator startup: "
            f"rank={rank}, declared={expected['object_map_sha256']}, "
            f"actual={actual_object_map_sha256}, path={object_map}"
        )

    expected_npz_records = expected["npz_files"]
    expected_names = [record["name"] for record in expected_npz_records]
    actual_paths = sorted(shard_dir.glob("*.npz"), key=lambda path: path.name)
    actual_names = [path.name for path in actual_paths]
    if actual_names != expected_names:
        raise RuntimeError(
            "rank-local NPZ set changed before simulator startup: "
            f"rank={rank}, declared={expected_names}, actual={actual_names}"
        )
    actual_npz_records = [
        _hashed_file_record(path, identity_recorder=identity_recorder)
        for path in actual_paths
    ]
    if actual_npz_records != expected_npz_records:
        raise RuntimeError(
            "rank-local NPZ content changed before simulator startup: "
            f"rank={rank}, shard={shard_dir}"
        )
    if identity_recorder is not None:
        identity_recorder(shard_dir)
    return {
        "rank": str(rank),
        "object_map_sha256": actual_object_map_sha256,
        "npz_content_sha256": sha256_json(actual_npz_records),
    }


def _lexical_absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _stat_identity_list(stat_result: os.stat_result) -> list[int]:
    return list(_file_identity_from_stat(stat_result))


def _file_identity_from_stat(stat_result: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
    )


def _path_identity_record(path: Path) -> dict[str, Any]:
    lexical_path = _lexical_absolute_path(path)
    try:
        lstat_result = lexical_path.lstat()
    except FileNotFoundError:
        parent = lexical_path.parent
        try:
            parent_identity = _stat_identity_list(parent.stat())
        except FileNotFoundError:
            parent_identity = None
        return {
            "path": str(lexical_path),
            "lstat": None,
            "stat": None,
            "parent_stat": parent_identity,
        }
    try:
        stat_result = lexical_path.stat()
    except FileNotFoundError:
        stat_identity = None
    else:
        stat_identity = _stat_identity_list(stat_result)
    return {
        "path": str(lexical_path),
        "lstat": _stat_identity_list(lstat_result),
        "stat": stat_identity,
        "parent_stat": None,
    }


class _DataAssetIdentityCollector:
    """Collect and later recheck cheap identities for every hashed input."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def record(self, path: Path) -> None:
        record = _path_identity_record(path)
        key = record["path"]
        previous = self._records.get(key)
        if previous is not None and previous != record:
            raise RuntimeError(
                f"training data asset changed during provenance validation: {key}"
            )
        self._records[key] = record

    def snapshot(self) -> list[dict[str, Any]]:
        return [self._records[key] for key in sorted(self._records)]

    def verify_unchanged(self) -> None:
        _verify_identity_snapshot(self.snapshot())


def _verify_identity_snapshot(snapshot: Any) -> None:
    if not isinstance(snapshot, list) or not snapshot:
        raise ValueError("data provenance cache has no file identity snapshot")
    previous_path = ""
    for expected in snapshot:
        if not isinstance(expected, dict) or not isinstance(expected.get("path"), str):
            raise ValueError("data provenance cache has an invalid file identity record")
        path = expected["path"]
        if path <= previous_path:
            raise ValueError("data provenance cache file identities are not uniquely sorted")
        previous_path = path
        actual = _path_identity_record(Path(path))
        if actual != expected:
            raise RuntimeError(
                "cached training data asset identity changed: "
                f"{path}; cached={expected!r} actual={actual!r}"
            )


def _data_asset_cache_key(
    provenance: dict[str, Any],
    *,
    motion_dir: Path,
    object_map: Path,
    contact_root: Path | None,
    motion_shard_manifest: Path | None,
    node_id: str,
) -> str:
    validated = validate_training_provenance(provenance, require_finalized=True)
    identity = {
        "version": _DATA_ASSET_CACHE_VERSION,
        "node_id": node_id,
        "declared": {
            "motion_shard_manifest_sha256": validated[
                "motion_shard_manifest_sha256"
            ],
            "contact_sidecar_manifest_sha256": validated[
                "contact_sidecar_manifest_sha256"
            ],
        },
        "paths": {
            "motion_dir": str(_lexical_absolute_path(motion_dir)),
            "object_map": str(_lexical_absolute_path(object_map)),
            "contact_root": (
                str(_lexical_absolute_path(contact_root))
                if contact_root is not None
                else None
            ),
            "motion_shard_manifest": (
                str(_lexical_absolute_path(motion_shard_manifest))
                if motion_shard_manifest is not None
                else None
            ),
        },
    }
    return sha256_json(identity)


def _cache_record_digest(record: dict[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    return sha256_json(payload)


def _load_valid_data_asset_cache(
    cache_path: Path,
    *,
    cache_key: str,
    expected_digests: dict[str, str],
) -> dict[str, Any] | None:
    if not cache_path.is_file():
        return None
    try:
        record = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            return None
        if record.get("version") != _DATA_ASSET_CACHE_VERSION:
            return None
        if record.get("cache_key") != cache_key:
            return None
        if record.get("expected_digests") != expected_digests:
            return None
        if record.get("record_sha256") != _cache_record_digest(record):
            return None
        _verify_identity_snapshot(record.get("identities"))
    except (OSError, ValueError, RuntimeError, TypeError):
        return None
    return record


def _write_data_asset_cache(cache_path: Path, record: dict[str, Any]) -> None:
    record = dict(record)
    record["record_sha256"] = _cache_record_digest(record)
    temporary = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, cache_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _cache_thread_lock(cache_path: Path) -> threading.Lock:
    key = str(cache_path)
    with _DATA_ASSET_CACHE_THREAD_LOCKS_GUARD:
        return _DATA_ASSET_CACHE_THREAD_LOCKS.setdefault(key, threading.Lock())


def _revalidate_selected_rank_local_shard_cached(
    *,
    motion_shard_manifest: Path,
    declared_motion_digest: str,
    cache_root: Path,
    node_id: str,
) -> None:
    rank = _runtime_global_rank()
    expected = {
        "motion_shard_manifest_sha256": declared_motion_digest,
        "rank": str(rank),
    }
    cache_key = sha256_json(
        {
            "version": _DATA_ASSET_CACHE_VERSION,
            "kind": "selected_rank_local_shard",
            "node_id": node_id,
            "rank": rank,
            "manifest": str(_lexical_absolute_path(motion_shard_manifest)),
            "declared_motion_digest": declared_motion_digest,
        }
    )
    cache_path = cache_root / f"rank-{cache_key}.json"
    lock_path = cache_root / f"rank-{cache_key}.lock"
    with _cache_thread_lock(cache_path):
        with lock_path.open("a+", encoding="utf-8") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            cached = _load_valid_data_asset_cache(
                cache_path,
                cache_key=cache_key,
                expected_digests=expected,
            )
            if cached is not None:
                if cached.get("status") == "success":
                    return
                if cached.get("status") == "failure":
                    raise RuntimeError(
                        "cached rank-local shard provenance failure: "
                        + str(cached.get("error", "unknown failure"))
                    )
                raise RuntimeError("rank-local shard provenance cache has an invalid status")

            collector = _DataAssetIdentityCollector()
            collector.record(motion_shard_manifest)
            try:
                _revalidate_selected_rank_local_shard(
                    motion_shard_manifest,
                    identity_recorder=collector.record,
                )
                collector.verify_unchanged()
            except Exception as exc:
                try:
                    collector.verify_unchanged()
                except Exception:
                    raise exc
                _write_data_asset_cache(
                    cache_path,
                    {
                        "version": _DATA_ASSET_CACHE_VERSION,
                        "cache_key": cache_key,
                        "expected_digests": expected,
                        "status": "failure",
                        "error": f"{type(exc).__name__}: {exc}",
                        "identities": collector.snapshot(),
                    },
                )
                raise
            _write_data_asset_cache(
                cache_path,
                {
                    "version": _DATA_ASSET_CACHE_VERSION,
                    "cache_key": cache_key,
                    "expected_digests": expected,
                    "status": "success",
                    "identities": collector.snapshot(),
                },
            )


def revalidate_data_asset_provenance_cached(
    provenance: dict[str, Any],
    *,
    motion_dir: Path,
    object_map: Path,
    contact_root: Path | None,
    motion_shard_manifest: Path | None,
    source_root: Path | None = None,
    cache_root: Path | None = None,
    node_id: str | None = None,
) -> dict[str, str]:
    """Validate once per node, then reuse only under an unchanged stat closure."""

    validated = validate_training_provenance(provenance, require_finalized=True)
    source_actual: dict[str, str] = {}
    if source_root is not None:
        source_digest = _source_bundle_digest(source_root)
        if source_digest != validated["source_bundle_sha256"]:
            raise RuntimeError(
                "training source bundle changed after launcher preflight: "
                f"declared={validated['source_bundle_sha256']} actual={source_digest}"
            )
        source_actual["source_bundle_sha256"] = source_digest
    expected_digests = {
        "motion_shard_manifest_sha256": validated["motion_shard_manifest_sha256"],
        "contact_sidecar_manifest_sha256": validated[
            "contact_sidecar_manifest_sha256"
        ],
    }
    if node_id is None:
        node_id = socket.gethostname()
    if not node_id.strip():
        raise ValueError("data provenance cache node_id must not be empty")
    if cache_root is None:
        cache_root = Path(
            os.environ.get(
                _DATA_ASSET_CACHE_ROOT_ENV,
                f"/tmp/holosoma-data-provenance-{os.getuid()}",
            )
        )
    cache_root = cache_root.expanduser().resolve()
    cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(cache_root, 0o700)
    except PermissionError:
        # A deliberately shared node-local cache may be owned by an operator;
        # file-level integrity and identity checks still remain mandatory.
        pass

    cache_key = _data_asset_cache_key(
        validated,
        motion_dir=motion_dir,
        object_map=object_map,
        contact_root=contact_root,
        motion_shard_manifest=motion_shard_manifest,
        node_id=node_id,
    )
    cache_path = cache_root / f"{cache_key}.json"
    lock_path = cache_root / f"{cache_key}.lock"

    with _cache_thread_lock(cache_path):
        with lock_path.open("a+", encoding="utf-8") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            cached = _load_valid_data_asset_cache(
                cache_path,
                cache_key=cache_key,
                expected_digests=expected_digests,
            )
            if cached is not None:
                if cached.get("status") == "success":
                    actual = dict(expected_digests)
                if cached.get("status") == "failure":
                    raise RuntimeError(
                        "cached per-node training data provenance failure: "
                        + str(cached.get("error", "unknown failure"))
                    )
                if cached.get("status") not in {"success", "failure"}:
                    raise RuntimeError("data provenance cache has an invalid status")
            else:
                collector = _DataAssetIdentityCollector()
                for path in (motion_dir, object_map, contact_root, motion_shard_manifest):
                    if path is not None:
                        collector.record(path)
                try:
                    actual = revalidate_data_asset_provenance(
                        validated,
                        motion_dir=motion_dir,
                        object_map=object_map,
                        contact_root=contact_root,
                        motion_shard_manifest=motion_shard_manifest,
                        identity_recorder=collector.record,
                    )
                    collector.verify_unchanged()
                except Exception as exc:
                    try:
                        collector.verify_unchanged()
                    except Exception:
                        # Inputs are actively changing, so no stable negative
                        # cache may be published.  The original failure remains
                        # clearer.
                        raise exc
                    _write_data_asset_cache(
                        cache_path,
                        {
                            "version": _DATA_ASSET_CACHE_VERSION,
                            "cache_key": cache_key,
                            "expected_digests": expected_digests,
                            "status": "failure",
                            "error": f"{type(exc).__name__}: {exc}",
                            "identities": collector.snapshot(),
                        },
                    )
                    raise
                _write_data_asset_cache(
                    cache_path,
                    {
                        "version": _DATA_ASSET_CACHE_VERSION,
                        "cache_key": cache_key,
                        "expected_digests": expected_digests,
                        "status": "success",
                        "identities": collector.snapshot(),
                    },
                )
    if motion_shard_manifest is not None:
        _revalidate_selected_rank_local_shard_cached(
            motion_shard_manifest=motion_shard_manifest,
            declared_motion_digest=validated["motion_shard_manifest_sha256"],
            cache_root=cache_root,
            node_id=node_id,
        )
    return {**actual, **source_actual}


def compute_provenance(
    *,
    teacher_checkpoint: Path,
    motion_dir: Path,
    object_map: Path,
    contact_root: Path | None,
    contact_sidecar_mode: str = "full-sidecars",
    motion_shard_manifest: Path | None,
    student_motion_end_mode: str,
    contact_interval_runtime_prepend_compensation: bool,
    source_root: Path,
    policy_init_checkpoint: Path | None = None,
    stage4_init_checkpoint: Path | None = None,
    training_resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    teacher_checkpoint = teacher_checkpoint.expanduser().resolve()
    motion_dir = motion_dir.expanduser().resolve()
    object_map = object_map.expanduser().resolve()
    contact_root = contact_root.expanduser().resolve() if contact_root is not None else None
    if not teacher_checkpoint.is_file():
        raise FileNotFoundError(f"teacher checkpoint does not exist: {teacher_checkpoint}")
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"motion directory does not exist: {motion_dir}")

    teacher_motion_ends, teacher_uses_actions, teacher_sha256 = _teacher_contract_and_digest(
        teacher_checkpoint
    )
    if not teacher_uses_actions:
        raise ValueError(
            "teacher actor observation contract has no actions term; refusing an unverified first-frame history contract"
        )
    if student_motion_end_mode == "episodic" and not teacher_motion_ends:
        raise ValueError(
            "student uses episodic motion-end termination but teacher checkpoint metadata has no motion_ends term"
        )
    if student_motion_end_mode == "continuing" and teacher_motion_ends:
        print(
            "[WARN] STUDENT_MOTION_END_MODE=continuing is explicitly incompatible with the episodic teacher: "
            "clip rollover can leak previous actions and PPO GAE across motion boundaries.",
            file=sys.stderr,
        )

    provenance = {
        "version": PROVENANCE_VERSION,
        TRAINING_REGIME_KEY: TRAINING_REGIME_DISTILLATION,
        TEACHER_ENABLED_KEY: True,
        # Simulator assets are selected only after Tyro and runtime config
        # overrides.  train_agent replaces this domain-separated sentinel with
        # the content-closed effective asset digest before any cross-rank or
        # checkpoint preflight.
        RUNTIME_ASSET_PHASE_KEY: RUNTIME_ASSET_PHASE_PENDING,
        RUNTIME_ASSET_DIGEST_KEY: pending_runtime_asset_manifest_sha256(),
        RUNTIME_ASSET_MANIFEST_KEY: None,
        "teacher_sha256": teacher_sha256,
        "policy_init_enabled": policy_init_checkpoint is not None,
        "policy_init_sha256": _optional_checkpoint_digest(
            policy_init_checkpoint,
            role="policy_init",
        ),
        "stage4_init_enabled": stage4_init_checkpoint is not None,
        "stage4_init_sha256": _optional_checkpoint_digest(
            stage4_init_checkpoint,
            role="stage4_init",
        ),
        "training_resume_enabled": training_resume_checkpoint is not None,
        "training_resume_sha256": _optional_checkpoint_digest(
            training_resume_checkpoint,
            role="training_resume",
        ),
        "motion_shard_manifest_sha256": _motion_manifest_digest(
            motion_dir,
            object_map,
            motion_shard_manifest.expanduser().resolve() if motion_shard_manifest is not None else None,
        ),
        "contact_sidecar_manifest_sha256": (
            _contact_manifest_digest(
                motion_dir,
                contact_root,
                contact_sidecar_mode=contact_sidecar_mode,
            )
            if contact_root is not None
            else disabled_contact_sidecar_manifest_sha256()
        ),
        "contact_sidecar_mode": contact_sidecar_mode,
        "source_bundle_sha256": _source_bundle_digest(source_root),
        "teacher_motion_end_mode": "episodic" if teacher_motion_ends else "continuing",
        "teacher_uses_action_history": teacher_uses_actions,
        "student_motion_end_mode": student_motion_end_mode,
        "contact_interval_runtime_prepend_compensation": bool(
            contact_interval_runtime_prepend_compensation
        ),
        "environment": _environment_metadata(),
    }
    motion_generator_sha256 = os.environ.get(
        "HOLOSOMA_EXTERNAL_AS_MOTION_GENERATOR_TEACHER_SHA256",
        "",
    ).strip()
    if motion_generator_sha256:
        if _SHA256_RE.fullmatch(motion_generator_sha256) is None:
            raise ValueError(
                "HOLOSOMA_EXTERNAL_AS_MOTION_GENERATOR_TEACHER_SHA256 must be a lowercase SHA256 digest"
            )
        raw_require_match = os.environ.get("REQUIRE_MOTION_GENERATOR_TEACHER_MATCH")
        if raw_require_match is None or not raw_require_match.strip():
            raise ValueError(
                "REQUIRE_MOTION_GENERATOR_TEACHER_MATCH is required when motion-generator teacher identity is present"
            )
        provenance[MOTION_GENERATOR_TEACHER_SHA256_KEY] = motion_generator_sha256
        provenance[REQUIRE_MOTION_GENERATOR_TEACHER_MATCH_KEY] = _normalized_bool_env(
            "REQUIRE_MOTION_GENERATOR_TEACHER_MATCH",
            default=True,
        )
    provenance.update(_source_snapshot_identity_from_env())
    provenance.update(_formal_git_identity_from_env())
    return provenance


def compute_generalist_provenance(
    *,
    motion_dir: Path,
    object_map: Path,
    contact_root: Path | None,
    contact_sidecar_mode: str = "full-sidecars",
    motion_shard_manifest: Path | None,
    source_root: Path,
    contact_interval_runtime_prepend_compensation: bool = False,
    policy_init_checkpoint: Path | None = None,
    stage4_init_checkpoint: Path | None = None,
    training_resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Build content-closed provenance for teacher-free object RL training."""

    motion_dir = motion_dir.expanduser().resolve()
    object_map = object_map.expanduser().resolve()
    contact_root = contact_root.expanduser().resolve() if contact_root is not None else None
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"motion directory does not exist: {motion_dir}")

    provenance = {
        "version": PROVENANCE_VERSION,
        TRAINING_REGIME_KEY: TRAINING_REGIME_PURE_RL,
        TEACHER_ENABLED_KEY: False,
        # There is deliberately no synthetic teacher artifact.  The
        # domain-separated sentinel binds the teacher-free mode itself.
        "teacher_sha256": disabled_teacher_sha256(),
        RUNTIME_ASSET_PHASE_KEY: RUNTIME_ASSET_PHASE_PENDING,
        RUNTIME_ASSET_DIGEST_KEY: pending_runtime_asset_manifest_sha256(),
        RUNTIME_ASSET_MANIFEST_KEY: None,
        "policy_init_enabled": policy_init_checkpoint is not None,
        "policy_init_sha256": _optional_checkpoint_digest(
            policy_init_checkpoint,
            role="policy_init",
        ),
        "stage4_init_enabled": stage4_init_checkpoint is not None,
        "stage4_init_sha256": _optional_checkpoint_digest(
            stage4_init_checkpoint,
            role="stage4_init",
        ),
        "training_resume_enabled": training_resume_checkpoint is not None,
        "training_resume_sha256": _optional_checkpoint_digest(
            training_resume_checkpoint,
            role="training_resume",
        ),
        "motion_shard_manifest_sha256": _motion_manifest_digest(
            motion_dir,
            object_map,
            motion_shard_manifest.expanduser().resolve()
            if motion_shard_manifest is not None
            else None,
        ),
        "contact_sidecar_manifest_sha256": (
            _contact_manifest_digest(
                motion_dir,
                contact_root,
                contact_sidecar_mode=contact_sidecar_mode,
            )
            if contact_root is not None
            else disabled_contact_sidecar_manifest_sha256()
        ),
        "contact_sidecar_mode": contact_sidecar_mode,
        "contact_interval_runtime_prepend_compensation": bool(
            contact_interval_runtime_prepend_compensation
        ),
        "source_bundle_sha256": _source_bundle_digest(source_root),
        "environment": _environment_metadata(),
    }
    provenance.update(_source_snapshot_identity_from_env())
    provenance.update(_formal_git_identity_from_env())
    return validate_training_provenance(provenance)


def _revalidate_data_assets_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Revalidate launcher-bound motion/object/contact assets before simulator use."
    )
    parser.add_argument("--revalidate-data-assets", action="store_true", required=True)
    parser.add_argument("--motion-dir", required=True, type=Path)
    parser.add_argument("--object-map", required=True, type=Path)
    parser.add_argument("--contact-root", type=Path)
    parser.add_argument(
        "--contact-sidecar-mode",
        choices=CONTACT_SIDECAR_MODES,
        default="full-sidecars",
    )
    parser.add_argument("--motion-shard-manifest", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--node-id")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, dict):
            raise ValueError("finalized training provenance on stdin must be a JSON object")
        actual = revalidate_data_asset_provenance_cached(
            payload,
            motion_dir=args.motion_dir,
            object_map=args.object_map,
            contact_root=args.contact_root,
            motion_shard_manifest=args.motion_shard_manifest,
            source_root=args.source_root,
            cache_root=args.cache_root,
            node_id=args.node_id,
        )
    except Exception as exc:
        raise SystemExit(
            f"[ERROR] pre-simulator training data provenance revalidation failed: {exc}"
        ) from exc
    print(
        "[INFO] pre-simulator training data provenance verified "
        + json.dumps(actual, sort_keys=True, separators=(",", ":"))
    )


def main() -> None:
    if "--revalidate-data-assets" in sys.argv[1:]:
        _revalidate_data_assets_main(sys.argv[1:])
        return
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-regime",
        choices=(TRAINING_REGIME_DISTILLATION, TRAINING_REGIME_PURE_RL),
        default=TRAINING_REGIME_DISTILLATION,
    )
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--motion-dir", required=True, type=Path)
    parser.add_argument("--object-map", required=True, type=Path)
    parser.add_argument("--contact-root", type=Path)
    parser.add_argument(
        "--contact-sidecar-mode",
        choices=CONTACT_SIDECAR_MODES,
        default="full-sidecars",
    )
    parser.add_argument("--motion-shard-manifest", type=Path)
    parser.add_argument("--student-motion-end-mode", choices=("episodic", "continuing"))
    parser.add_argument(
        "--contact-interval-runtime-prepend-compensation",
        choices=("true", "false"),
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--policy-init-checkpoint", type=Path)
    parser.add_argument("--stage4-init-checkpoint", type=Path)
    parser.add_argument("--training-resume-checkpoint", type=Path)
    args = parser.parse_args()
    try:
        if args.training_regime == TRAINING_REGIME_DISTILLATION:
            if args.teacher_checkpoint is None:
                raise ValueError("distillation provenance requires --teacher-checkpoint")
            if args.student_motion_end_mode is None:
                raise ValueError("distillation provenance requires --student-motion-end-mode")
            if args.contact_interval_runtime_prepend_compensation is None:
                raise ValueError(
                    "distillation provenance requires "
                    "--contact-interval-runtime-prepend-compensation"
                )
            provenance = compute_provenance(
                teacher_checkpoint=args.teacher_checkpoint,
                motion_dir=args.motion_dir,
                object_map=args.object_map,
                contact_root=args.contact_root,
                contact_sidecar_mode=args.contact_sidecar_mode,
                motion_shard_manifest=args.motion_shard_manifest,
                student_motion_end_mode=args.student_motion_end_mode,
                contact_interval_runtime_prepend_compensation=(
                    args.contact_interval_runtime_prepend_compensation == "true"
                ),
                source_root=args.source_root,
                policy_init_checkpoint=args.policy_init_checkpoint,
                stage4_init_checkpoint=args.stage4_init_checkpoint,
                training_resume_checkpoint=args.training_resume_checkpoint,
            )
        else:
            if args.teacher_checkpoint is not None:
                raise ValueError("pure-RL provenance must not receive --teacher-checkpoint")
            if args.student_motion_end_mode is not None:
                raise ValueError("pure-RL provenance must not claim a student/teacher motion-end contract")
            provenance = compute_generalist_provenance(
                motion_dir=args.motion_dir,
                object_map=args.object_map,
                contact_root=args.contact_root,
                contact_sidecar_mode=args.contact_sidecar_mode,
                motion_shard_manifest=args.motion_shard_manifest,
                contact_interval_runtime_prepend_compensation=(
                    args.contact_interval_runtime_prepend_compensation == "true"
                    if args.contact_interval_runtime_prepend_compensation is not None
                    else False
                ),
                source_root=args.source_root,
                policy_init_checkpoint=args.policy_init_checkpoint,
                stage4_init_checkpoint=args.stage4_init_checkpoint,
                training_resume_checkpoint=args.training_resume_checkpoint,
            )
    except Exception as exc:
        raise SystemExit(f"[ERROR] training provenance preflight failed: {exc}") from exc
    print(json.dumps(provenance, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
