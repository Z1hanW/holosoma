"""Fail-closed validation for curriculum-correct training checkpoint resumes.

This module intentionally runs in a short-lived subprocess before the simulator
is imported.  It can therefore inspect a torch checkpoint without changing the
Isaac/torch import ordering of the training process.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import numbers
import os
import re
import sys
from pathlib import Path
from typing import Any

from holosoma.utils.checkpoint_validation import (
    load_verified_torch_checkpoint,
    validate_checkpoint_iterations,
    validate_finite_tree,
    validate_student_actor_contract,
)
from holosoma.utils.rng_checkpoint import (
    ALLOW_NONDETERMINISTIC_RNG_RESUME_ENV,
    validate_rng_checkpoint_state,
)
from holosoma.utils.training_provenance import (
    SEMANTIC_ENVIRONMENT_FIELDS,
    SEMANTIC_ENVIRONMENT_KEY,
    parse_training_provenance,
    validate_hierarchical_small_collectives_contract,
    validate_training_provenance,
)


ALLOW_RUNTIME_DRIFT_ENV = "HOLOSOMA_ALLOW_RUNTIME_DRIFT_ON_RESUME"
ALLOW_FIXED_BC_RESET_ENV = "HOLOSOMA_ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME"
ALLOW_LEGACY_ROLLOUT_RESUME_ENV = "HOLOSOMA_ALLOW_LEGACY_ROLLOUT_RESTART_RESUME"
ALLOW_LEGACY_UNPROVENANCED_RESUME_ENV = "HOLOSOMA_ALLOW_LEGACY_UNPROVENANCED_RESUME"
EXACT_ROLLOUT_RESUME_CONTRACT_VERSION = 2
EXACT_ROLLOUT_RESUME_CONTRACT_MODE = "canonical_reset_after_checkpoint"
RECOVERY_ROLLOUT_RESUME_CONTRACT_VERSION = 3
RECOVERY_ROLLOUT_RESUME_CONTRACT_MODE = "new_episode_on_resume"
_RUNTIME_TOP_LEVEL_FIELDS = (
    "python_runtime_manifest_sha256",
    "python",
    "torch",
    "torch_cuda",
)
_RUNTIME_PACKAGE_FIELDS = (
    "torch",
    "isaacsim",
    "isaaclab",
    "numpy",
    "omegaconf",
    "antlr4-python3-runtime",
    "PyYAML",
    "attrs",
)
_RUNTIME_EXECUTION_KEY = "execution_runtime"
_RUNTIME_SEMANTIC_ENVIRONMENT_KEY = SEMANTIC_ENVIRONMENT_KEY
_RUNTIME_EXECUTION_BOOL_FIELDS = (
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "HOLOSOMA_GLOO_BARRIER",
    "HOLOSOMA_GLOO_GRAD_REDUCE",
    "HOLOSOMA_GLOO_SMALL_COLLECTIVES",
    "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE",
    "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES",
    "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER",
    "HOLOSOMA_RANK_VISIBLE_DEVICES",
    "HOLOSOMA_CONTIGUOUS_MINIBATCHES",
    "HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP",
    "HOLOSOMA_DAGGER_SUPERVISED_ONLY",
    "HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD",
    "HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC",
)
_RUNTIME_EXECUTION_INT_FIELDS = (
    "HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH",
    "NPROC",
    "NNODES",
)
_RUNTIME_SEMANTIC_ENVIRONMENT_FIELDS = SEMANTIC_ENVIRONMENT_FIELDS
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _positive_finite_learning_rate(path: str, value: Any) -> float:
    """Parse one serialized PPO learning rate without accepting bool/string coercions."""

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{path}: expected a finite number > 0, got {value!r}.")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{path}: expected a finite number > 0, got {value!r}.")
    return parsed


def _canonicalize_ppo_learning_rate_bounds(manifest: dict[str, Any]) -> None:
    """Resolve serialized ``None`` PPO bounds exactly as PPO initialization does.

    A missing bound remains missing so this compatibility shim cannot conceal an
    incomplete schema.  Once either bound for an optimizer is serialized, its
    initial rate and both effective bounds are validated fail-closed.
    """

    try:
        algo_config = manifest["algo"]["config"]
    except (KeyError, TypeError):
        return
    if not isinstance(algo_config, dict):
        return

    for optimizer in ("actor", "critic"):
        initial_key = f"{optimizer}_learning_rate"
        minimum_key = f"min_{optimizer}_learning_rate"
        maximum_key = f"max_{optimizer}_learning_rate"
        if minimum_key not in algo_config and maximum_key not in algo_config:
            continue
        if initial_key not in algo_config:
            raise ValueError(
                f"algo.config.{initial_key}: missing while PPO learning-rate bounds are serialized."
            )

        initial = _positive_finite_learning_rate(
            f"algo.config.{initial_key}", algo_config[initial_key]
        )
        minimum_raw = algo_config.get(minimum_key)
        maximum_raw = algo_config.get(maximum_key)
        minimum = (
            min(initial, 1.0e-5)
            if minimum_raw is None
            else _positive_finite_learning_rate(
                f"algo.config.{minimum_key}", minimum_raw
            )
        )
        maximum = (
            max(initial, 1.0e-2)
            if maximum_raw is None
            else _positive_finite_learning_rate(
                f"algo.config.{maximum_key}", maximum_raw
            )
        )
        if not minimum <= initial <= maximum:
            raise ValueError(
                f"algo.config {optimizer} learning-rate bounds must satisfy "
                "minimum <= initial <= maximum, "
                f"got minimum={minimum}, initial={initial}, maximum={maximum}."
            )

        # Preserve structural mismatches: only canonicalize fields actually
        # serialized by this side of the resume comparison.
        if minimum_key in algo_config:
            algo_config[minimum_key] = minimum
        if maximum_key in algo_config:
            algo_config[maximum_key] = maximum


def _remove_path(value: dict[str, Any], path: tuple[str, ...]) -> None:
    current: Any = value
    for key in path[:-1]:
        if not isinstance(current, dict):
            return
        current = current.get(key)
    if isinstance(current, dict):
        current.pop(path[-1], None)


def _allow_runtime_drift_on_resume() -> bool:
    """Return the explicit runtime-drift escape hatch, rejecting ambiguous values."""

    raw_value = os.environ.get(ALLOW_RUNTIME_DRIFT_ENV)
    if raw_value is None or raw_value.strip() in ("", "0"):
        return False
    if raw_value.strip() == "1":
        return True
    raise ValueError(
        f"{ALLOW_RUNTIME_DRIFT_ENV} must be exactly 0 or 1; got {raw_value!r}. "
        "Runtime drift is allowed only by an explicit value of 1."
    )


def _allow_fixed_bc_reset_on_resume() -> bool:
    raw_value = os.environ.get(ALLOW_FIXED_BC_RESET_ENV)
    if raw_value is None or raw_value.strip() in ("", "0"):
        return False
    if raw_value.strip() == "1":
        return True
    raise ValueError(
        f"{ALLOW_FIXED_BC_RESET_ENV} must be exactly 0 or 1; got {raw_value!r}."
    )


def allow_legacy_unprovenanced_resume() -> bool:
    """Return the explicit non-scientific full-resume escape hatch."""

    raw_value = os.environ.get(ALLOW_LEGACY_UNPROVENANCED_RESUME_ENV)
    if raw_value is None or raw_value.strip() in ("", "0"):
        return False
    if raw_value.strip() == "1":
        return True
    raise ValueError(
        f"{ALLOW_LEGACY_UNPROVENANCED_RESUME_ENV} must be exactly 0 or 1; "
        f"got {raw_value!r}."
    )


def _allow_nondeterministic_rng_resume() -> bool:
    raw_value = os.environ.get(ALLOW_NONDETERMINISTIC_RNG_RESUME_ENV)
    if raw_value is None or raw_value.strip() in ("", "0"):
        return False
    if raw_value.strip() == "1":
        return True
    raise ValueError(
        f"{ALLOW_NONDETERMINISTIC_RNG_RESUME_ENV} must be exactly 0 or 1; got {raw_value!r}."
    )


def _core_runtime_identity(provenance: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Extract deterministic runtime fields that can change training numerics.

    Hostnames, paths, kernel/platform strings, and unrelated installed packages
    remain audit metadata but are deliberately outside the resume contract.
    """

    environment = provenance.get("environment")
    if not isinstance(environment, dict):
        return {}, ["environment: missing or not a JSON object"]

    identity: dict[str, Any] = {}
    problems: list[str] = []
    for field in _RUNTIME_TOP_LEVEL_FIELDS:
        if field not in environment:
            problems.append(f"environment.{field}: missing")
            continue
        value = environment[field]
        if not isinstance(value, str) or not value.strip():
            problems.append(f"environment.{field}: expected a non-empty string, got {value!r}")
            continue
        identity[field] = value

    runtime_manifest = identity.get("python_runtime_manifest_sha256")
    if runtime_manifest is not None and _SHA256_RE.fullmatch(runtime_manifest) is None:
        problems.append(
            "environment.python_runtime_manifest_sha256: expected a lowercase SHA256 digest, "
            f"got {runtime_manifest!r}"
        )

    packages = environment.get("packages")
    package_identity: dict[str, str] = {}
    if not isinstance(packages, dict):
        problems.append("environment.packages: missing or not a JSON object")
    else:
        for package in _RUNTIME_PACKAGE_FIELDS:
            if package not in packages:
                problems.append(f"environment.packages.{package}: missing")
                continue
            value = packages[package]
            if not isinstance(value, str) or not value.strip():
                problems.append(
                    f"environment.packages.{package}: expected a non-empty version string, got {value!r}"
                )
                continue
            package_identity[package] = value
    identity["packages"] = package_identity

    execution_runtime = environment.get(_RUNTIME_EXECUTION_KEY)
    if not isinstance(execution_runtime, dict):
        problems.append(f"environment.{_RUNTIME_EXECUTION_KEY}: missing or not a JSON object")
        identity[_RUNTIME_EXECUTION_KEY] = {}
        return identity, problems

    # Preserve the whole mapping in the comparison so newly added provenance
    # fields automatically become exact-resume fields.  Validate the known
    # numerical controls here to reject stringly typed booleans/integers.
    identity[_RUNTIME_EXECUTION_KEY] = dict(execution_runtime)
    backend = execution_runtime.get("TORCH_DIST_BACKEND")
    if backend not in {"nccl", "gloo"}:
        problems.append(
            f"environment.{_RUNTIME_EXECUTION_KEY}.TORCH_DIST_BACKEND: "
            f"expected 'nccl' or 'gloo', got {backend!r}"
        )
    nccl_digest = execution_runtime.get("NCCL_LIB_SHA256")
    if not isinstance(nccl_digest, str) or _SHA256_RE.fullmatch(nccl_digest) is None:
        problems.append(
            f"environment.{_RUNTIME_EXECUTION_KEY}.NCCL_LIB_SHA256: "
            f"expected a lowercase SHA256 digest, got {nccl_digest!r}"
        )
    python_hash_seed = execution_runtime.get("PYTHONHASHSEED")
    if not isinstance(python_hash_seed, str) or not (
        python_hash_seed == "<unset>"
        or (
            python_hash_seed.isdecimal()
            and 0 <= int(python_hash_seed, 10) <= 4294967295
            and str(int(python_hash_seed, 10)) == python_hash_seed
        )
    ):
        problems.append(
            f"environment.{_RUNTIME_EXECUTION_KEY}.PYTHONHASHSEED: expected '<unset>' or a "
            f"canonical integer string in [0, 4294967295], got {python_hash_seed!r}"
        )
    cublas_workspace = execution_runtime.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas_workspace not in {"<unset>", ":4096:8", ":16:8"}:
        problems.append(
            f"environment.{_RUNTIME_EXECUTION_KEY}.CUBLAS_WORKSPACE_CONFIG: expected '<unset>', "
            f"':4096:8', or ':16:8', got {cublas_workspace!r}"
        )
    for field in _RUNTIME_EXECUTION_BOOL_FIELDS:
        value = execution_runtime.get(field)
        if type(value) is not bool:
            problems.append(
                f"environment.{_RUNTIME_EXECUTION_KEY}.{field}: expected a boolean, got {value!r}"
            )
    try:
        validate_hierarchical_small_collectives_contract(execution_runtime)
    except ValueError as exc:
        problems.append(f"environment.{_RUNTIME_EXECUTION_KEY}: {exc}")
    for field in _RUNTIME_EXECUTION_INT_FIELDS:
        value = execution_runtime.get(field)
        minimum = 1 if field in {"NPROC", "NNODES"} else 0
        if type(value) is not int or value < minimum:
            problems.append(
                f"environment.{_RUNTIME_EXECUTION_KEY}.{field}: "
                f"expected an integer >= {minimum}, got {value!r}"
            )

    semantic_environment = execution_runtime.get(_RUNTIME_SEMANTIC_ENVIRONMENT_KEY)
    semantic_path = (
        f"environment.{_RUNTIME_EXECUTION_KEY}.{_RUNTIME_SEMANTIC_ENVIRONMENT_KEY}"
    )
    if not isinstance(semantic_environment, dict):
        problems.append(f"{semantic_path}: missing or not a JSON object")
    else:
        expected_keys = set(_RUNTIME_SEMANTIC_ENVIRONMENT_FIELDS)
        actual_keys = set(semantic_environment)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys.difference(actual_keys))
            unexpected = sorted(
                (repr(key) for key in actual_keys.difference(expected_keys))
            )
            problems.append(
                f"{semantic_path}: keys must exactly match the scientific schema; "
                f"missing={missing!r}, unexpected={unexpected!r}"
            )
        for field in _RUNTIME_SEMANTIC_ENVIRONMENT_FIELDS:
            if field not in semantic_environment:
                continue
            value = semantic_environment[field]
            if value is not None and not isinstance(value, str):
                problems.append(
                    f"{semantic_path}.{field}: expected a string or null, got {value!r}"
                )
            elif isinstance(value, str) and value != value.strip():
                problems.append(
                    f"{semantic_path}.{field}: expected a stripped canonical string, got {value!r}"
                )
    return identity, problems


def _validate_resume_runtime_identity(
    saved_provenance: dict[str, Any],
    current_provenance: dict[str, Any],
    *,
    parent_checkpoint_sha256: str,
) -> None:
    """Fail closed on numerical runtime drift unless explicitly overridden."""

    saved_identity, saved_problems = _core_runtime_identity(saved_provenance)
    current_identity, current_problems = _core_runtime_identity(current_provenance)
    differences = _diff(
        saved_identity,
        current_identity,
        "training_provenance.environment",
    )
    discrepancies = [
        *(f"checkpoint.{problem}" for problem in saved_problems),
        *(f"current.{problem}" for problem in current_problems),
        *differences,
    ]
    if not discrepancies:
        return

    preview = "\n  - ".join(discrepancies[:30])
    suffix = "" if len(discrepancies) <= 30 else f"\n  ... and {len(discrepancies) - 30} more discrepancy(s)"
    if not _allow_runtime_drift_on_resume():
        raise ValueError(
            "Training-resume core runtime identity is missing, invalid, or differs from the checkpoint. "
            "Refusing a numerically unverifiable resume. Set "
            f"{ALLOW_RUNTIME_DRIFT_ENV}=1 only when this lineage break is intentional:\n  - "
            + preview
            + suffix
        )

    # The current checkpoint provenance records the current environment and the
    # resume checkpoint digest.  Include both identities here as an explicit,
    # machine-readable lineage warning rather than silently accepting drift.
    print(
        f"[WARN] runtime_drift_on_resume_allowed override={ALLOW_RUNTIME_DRIFT_ENV}=1 "
        f"lineage_parent_checkpoint_sha256={parent_checkpoint_sha256} "
        f"checkpoint_runtime_identity={json.dumps(saved_identity, sort_keys=True, separators=(',', ':'))} "
        f"current_runtime_identity={json.dumps(current_identity, sort_keys=True, separators=(',', ':'))} "
        f"discrepancies={json.dumps(discrepancies, sort_keys=True, separators=(',', ':'))}",
        flush=True,
    )


def canonical_resume_manifest(
    config: dict[str, Any],
    *,
    teacher_identity_verified: bool = False,
) -> dict[str, Any]:
    """Build the training-semantic manifest; omit only approved resume overrides."""

    manifest = copy.deepcopy(config)
    _canonicalize_ppo_learning_rate_bounds(manifest)

    # Logging/deployment/debug controls do not alter the optimizer update or the
    # policy/environment semantics being resumed.
    manifest.pop("logger", None)
    manifest.pop("nightly", None)
    manifest.pop("eval_overrides", None)

    allowed_paths = (
        ("algo", "config", "num_learning_iterations"),
        ("algo", "config", "eval_callbacks"),
        # This controls whether *future* checkpoint publications interrupt the
        # live rollout.  The checkpoint's authenticated rollout-resume
        # contract below remains the authority for the boundary being loaded.
        ("algo", "config", "reset_rollout_at_checkpoint"),
        ("algo", "config", "distill", "schedule_name"),
        ("algo", "config", "distill", "schedule_notes"),
        ("training", "checkpoint"),
        ("training", "policy_init_checkpoint"),
        ("training", "project"),
        ("training", "name"),
        ("training", "headless"),
        ("training", "max_eval_steps"),
        ("training", "export_onnx"),
        ("training", "debug"),
        ("training", "enable_viser"),
        ("training", "viser_port"),
        ("training", "viser_env_id"),
        ("training", "viser_env_count"),
        ("training", "viser_multi_env_spacing"),
        ("training", "viser_update_hz"),
        ("training", "viser_sync_to_sim"),
        ("training", "viser_force_dt"),
        ("training", "viser_recenter"),
        ("training", "viser_global_frame_quat_wxyz"),
        ("training", "viser_show_scandots"),
        ("training", "viser_scandots_point_size"),
        ("training", "isaac_show_scandots"),
        ("training", "isaac_scandots_point_size"),
    )
    for path in allowed_paths:
        _remove_path(manifest, path)

    # A locally staged path and its W&B URI may only compare equal after the
    # checkpoint SHA256 has been verified through training_provenance.  A file
    # basename alone is never accepted as teacher identity.
    try:
        distill = manifest["algo"]["config"]["distill"]
    except (KeyError, TypeError):
        distill = None
    if teacher_identity_verified and isinstance(distill, dict):
        for key in ("policy_to_clone", "teacher_checkpoint"):
            distill.pop(key, None)

    # The compensation flag was added after legacy u7 checkpoints.  Its absent
    # serialized value is semantically False; this keeps old continuing-timebase
    # resumes stable while still rejecting a silent switch to True.
    try:
        motion_config = manifest["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]
    except (KeyError, TypeError):
        motion_config = None
    if isinstance(motion_config, dict):
        motion_config.setdefault("contact_interval_runtime_prepend_compensation", False)
        # Missing is the exact legacy button-label contract.  Materializing
        # this default on both sides keeps old resumes valid while making any
        # contact_interval -> kinematic_lift change a fail-closed semantic
        # mismatch.
        motion_config.setdefault(
            "contact_aware_button_window_mode",
            "contact_interval",
        )

    return manifest


def _diff(expected: Any, actual: Any, path: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: list[str] = []
        for key in sorted(
            set(expected) | set(actual),
            key=lambda item: (type(item).__name__, repr(item)),
        ):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in expected:
                differences.append(f"{child_path}: checkpoint=<missing> current={actual[key]!r}")
            elif key not in actual:
                differences.append(f"{child_path}: checkpoint={expected[key]!r} current=<missing>")
            else:
                differences.extend(_diff(expected[key], actual[key], child_path))
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: checkpoint_len={len(expected)} current_len={len(actual)}"]
        differences = []
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            differences.extend(_diff(left, right, f"{path}[{index}]"))
        return differences
    if expected != actual:
        return [f"{path}: checkpoint={expected!r} current={actual!r}"]
    return []


def validate_resume_payload_identity(
    checkpoint: dict[str, Any],
    current_config: dict[str, Any],
    *,
    current_provenance: dict[str, Any] | None,
    actual_resume_sha256: str,
) -> bool:
    """Validate config and provenance identity for an already authenticated payload.

    Returning ``True`` means teacher/input identity was proven by a finalized
    provenance pair.  The legacy both-missing case is accepted only through an
    explicit non-scientific escape hatch.
    """

    legacy_unprovenanced_resume = allow_legacy_unprovenanced_resume()
    if not isinstance(checkpoint, dict):
        raise ValueError("Training-resume checkpoint payload must be a mapping.")
    saved_config = checkpoint.get("experiment_config")
    if not isinstance(saved_config, dict):
        raise ValueError(
            "Training-resume checkpoint has no serialized experiment_config; refusing an unverifiable resume."
        )
    if not isinstance(current_config, dict):
        raise ValueError("Current training-resume config must be a mapping.")

    validated_current_provenance = None
    if current_provenance is not None:
        validated_current_provenance = validate_training_provenance(
            current_provenance,
            require_finalized=True,
        )
        if validated_current_provenance.get("training_resume_enabled") is not True:
            raise ValueError(
                "Current training provenance does not enable a full training resume."
            )

    raw_saved_provenance = checkpoint.get("training_provenance")
    teacher_identity_verified = (
        raw_saved_provenance is not None or validated_current_provenance is not None
    )
    if teacher_identity_verified:
        if raw_saved_provenance is None:
            raise ValueError(
                "Training-resume checkpoint has no training_provenance digests; "
                "refusing basename-only teacher identity."
            )
        if validated_current_provenance is None:
            raise ValueError(
                "Current launch has no training_provenance digests required by this "
                "training-resume checkpoint."
            )
        saved_provenance = validate_training_provenance(
            raw_saved_provenance,
            require_finalized=True,
        )
        _validate_resume_runtime_identity(
            saved_provenance,
            validated_current_provenance,
            parent_checkpoint_sha256=actual_resume_sha256,
        )

        # Policy-init and resume hashes are lineage edges rather than inputs
        # that are re-applied after the checkpoint boundary.
        resume_ignored_keys = {
            "environment",
            "policy_init_enabled",
            "policy_init_sha256",
            "training_resume_enabled",
            "training_resume_sha256",
        }
        saved_resume_provenance = {
            key: value
            for key, value in saved_provenance.items()
            if key not in resume_ignored_keys
        }
        current_resume_provenance = {
            key: value
            for key, value in validated_current_provenance.items()
            if key not in resume_ignored_keys
        }
        provenance_differences = _diff(
            saved_resume_provenance,
            current_resume_provenance,
            "training_provenance",
        )
        if provenance_differences:
            raise ValueError(
                "Training-resume input provenance mismatch:\n  - "
                + "\n  - ".join(provenance_differences)
            )
    else:
        if not legacy_unprovenanced_resume:
            raise ValueError(
                "Full training resume has neither current nor checkpoint training provenance. "
                "Scientific exact resume requires finalized provenance on both sides. Set "
                f"{ALLOW_LEGACY_UNPROVENANCED_RESUME_ENV}=1 only to enter an explicitly "
                "non-scientific legacy lineage."
            )
        print(
            "[WARN] legacy_unprovenanced_resume_allowed "
            f"override={ALLOW_LEGACY_UNPROVENANCED_RESUME_ENV}=1: source/assets/runtime "
            "identity is not scientifically authenticated.",
            flush=True,
        )

    saved_manifest = canonical_resume_manifest(
        saved_config,
        teacher_identity_verified=teacher_identity_verified,
    )
    current_manifest = canonical_resume_manifest(
        current_config,
        teacher_identity_verified=teacher_identity_verified,
    )
    differences = _diff(saved_manifest, current_manifest)
    if differences:
        preview = "\n  - ".join(differences[:30])
        suffix = (
            ""
            if len(differences) <= 30
            else f"\n  ... and {len(differences) - 30} more difference(s)"
        )
        raise ValueError(
            "Training-resume config mismatch outside the target/save/log/debug allowlist:\n  - "
            + preview
            + suffix
        )
    return teacher_identity_verified


def validate_resume_checkpoint(
    checkpoint_path: Path,
    current_config: dict[str, Any],
    *,
    world_size: int,
    allow_fresh_curriculum: bool,
    current_provenance: dict[str, Any] | None = None,
) -> str:
    # Parse every scientific override before touching checkpoint bytes so an
    # invalid ambient value fails in the short-lived preflight, not after the
    # simulator and distributed workers have started.
    allow_legacy_unprovenanced_resume()
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError(f"Resume world_size must be a positive integer, got {world_size!r}.")
    try:
        load_optimizer = current_config["algo"]["config"].get("load_optimizer", True)
    except (KeyError, TypeError) as exc:
        raise ValueError("Current config has no valid algo.config mapping.") from exc
    if load_optimizer is not True:
        raise ValueError(
            "Exact training resume requires algo.config.load_optimizer=true; resetting optimizer "
            "moments or adaptive learning rates is a warm start, not an exact continuation."
        )
    validated_current_provenance = None
    expected_resume_sha256 = None
    if current_provenance is not None:
        validated_current_provenance = validate_training_provenance(
            current_provenance,
            require_finalized=True,
        )
        if validated_current_provenance.get("training_resume_enabled") is not True:
            raise ValueError(
                "Current training provenance does not enable a full training resume."
            )
        expected_resume_sha256 = validated_current_provenance["training_resume_sha256"]
    checkpoint, actual_resume_sha256 = load_verified_torch_checkpoint(
        checkpoint_path,
        expected_sha256=expected_resume_sha256,
        map_location="cpu",
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Training-resume checkpoint payload must be a mapping.")
    try:
        saved_actor = checkpoint["experiment_config"]["algo"]["config"]["module_dict"]["actor"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Training-resume checkpoint has no valid student actor contract.") from exc
    try:
        current_actor = current_config["algo"]["config"]["module_dict"]["actor"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Current config has no valid student actor contract.") from exc
    validate_student_actor_contract(saved_actor)
    validate_student_actor_contract(current_actor)
    validate_resume_payload_identity(
        checkpoint,
        current_config,
        current_provenance=validated_current_provenance,
        actual_resume_sha256=actual_resume_sha256,
    )

    saved_iter, next_iter = validate_checkpoint_iterations(checkpoint)
    try:
        target_iteration = int(current_config["algo"]["config"]["num_learning_iterations"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Current config has no valid absolute num_learning_iterations target.") from exc
    if target_iteration <= next_iter:
        raise ValueError(
            f"Target learning iteration {target_iteration} must be greater than checkpoint next_iter {next_iter} "
            f"(saved iter {saved_iter})."
        )
    remaining_iterations = target_iteration - next_iter

    rollout_contract = checkpoint.get("rollout_resume_contract")
    if rollout_contract is None:
        raw_legacy_override = os.environ.get(ALLOW_LEGACY_ROLLOUT_RESUME_ENV)
        if raw_legacy_override is None or raw_legacy_override.strip() in ("", "0"):
            raise ValueError(
                "Checkpoint has no rollout_resume_contract, so its RNG state cannot establish equivalence "
                "to uninterrupted training across the forced resume reset. Set "
                f"{ALLOW_LEGACY_ROLLOUT_RESUME_ENV}=1 only to start an explicitly non-equivalent legacy "
                "restart lineage."
            )
        if raw_legacy_override.strip() != "1":
            raise ValueError(
                f"{ALLOW_LEGACY_ROLLOUT_RESUME_ENV} must be exactly 0 or 1; "
                f"got {raw_legacy_override!r}."
            )
        print(
            f"[WARN] legacy_rollout_restart_allowed override={ALLOW_LEGACY_ROLLOUT_RESUME_ENV}=1 "
            "checkpoint has no canonical rollout restart contract; continuation is not equivalent to "
            "an uninterrupted run.",
            flush=True,
        )
    else:
        try:
            algo_cfg = current_config["algo"]["config"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Current config has no PPO config for rollout-resume validation.") from exc
        distill_cfg = algo_cfg.get("distill", {})
        if not isinstance(distill_cfg, dict):
            distill_cfg = {}
        raw_checkpoint_reset = algo_cfg.get("reset_rollout_at_checkpoint", False)
        if type(raw_checkpoint_reset) is not bool:
            raise ValueError(
                "Current algo.config.reset_rollout_at_checkpoint must be an explicit boolean, "
                f"got {raw_checkpoint_reset!r}."
            )
        if raw_checkpoint_reset:
            expected_contract_version = EXACT_ROLLOUT_RESUME_CONTRACT_VERSION
            expected_contract_mode = EXACT_ROLLOUT_RESUME_CONTRACT_MODE
        else:
            expected_contract_version = RECOVERY_ROLLOUT_RESUME_CONTRACT_VERSION
            expected_contract_mode = RECOVERY_ROLLOUT_RESUME_CONTRACT_MODE
        expected_rollout_contract = {
            "version": expected_contract_version,
            "mode": expected_contract_mode,
            "next_iteration": next_iter,
            "save_interval": algo_cfg.get("save_interval"),
            "init_at_random_ep_len": bool(algo_cfg.get("init_at_random_ep_len", False)),
            "dagger_ignore_episode_initial_steps": int(
                distill_cfg.get("dagger_ignore_episode_initial_steps", 0)
            ),
            "reset_recurrent_hidden": True,
            "perception_state_mode": "checkpoint_stream_state_rebuild_derived_cache",
        }
        if rollout_contract != expected_rollout_contract:
            raise ValueError(
                "Training-resume rollout contract mismatch: "
                f"checkpoint={rollout_contract!r}, expected={expected_rollout_contract!r}."
            )
        rollout_resume_mode = expected_contract_mode

    states_by_rank = checkpoint.get("env_state_by_rank")
    legacy_state = checkpoint.get("env_state")
    if isinstance(states_by_rank, dict):
        noncanonical_keys = [key for key in states_by_rank if type(key) is not str]
        if noncanonical_keys:
            raise ValueError(
                "Training-resume checkpoint env_state_by_rank keys must be canonical decimal strings; "
                f"got non-string keys={noncanonical_keys!r}."
            )
        rank_keys = set(states_by_rank)
        expected_rank_keys = {str(rank) for rank in range(world_size)}
        if rank_keys != expected_rank_keys:
            raise ValueError(
                "Training-resume checkpoint world-size/rank-state mismatch: "
                f"checkpoint ranks={sorted(rank_keys)}, current ranks={sorted(expected_rank_keys)}."
            )
        empty_ranks = [str(rank) for rank, state in states_by_rank.items() if not isinstance(state, dict) or not state]
        if empty_ranks:
            raise ValueError(
                "Training-resume checkpoint has empty rank-local AS curriculum/sampler aggregate state "
                f"for ranks {empty_ranks}."
            )
        validate_finite_tree(states_by_rank, path="env_state_by_rank")
    elif states_by_rank is not None:
        raise ValueError("Training-resume checkpoint env_state_by_rank must be a mapping when present.")
    elif world_size == 1 and isinstance(legacy_state, dict) and legacy_state:
        validate_finite_tree(legacy_state, path="env_state")
    elif legacy_state is not None and world_size == 1:
        raise ValueError("Training-resume checkpoint env_state must be a non-empty mapping when present.")
    elif not allow_fresh_curriculum:
        raise ValueError(
            "Checkpoint has no saved rank-local AS curriculum/sampler aggregate state. "
            "This is not a curriculum-correct training resume. "
            "Set ALLOW_FRESH_CURRICULUM_RESUME=1 only if resetting adaptive sampler/failure/clip counters is intentional."
        )
    else:
        print(
            "[WARN] ALLOW_FRESH_CURRICULUM_RESUME=1: actor/critic/optimizers/normalizers will resume, but AS "
            "adaptive sampler, failure, and clip counters start fresh. This never restores "
            "per-environment physics/observation/action history.",
            flush=True,
        )

    rng_states = checkpoint.get("rng_state_by_rank")
    rng_stream_complete = False
    if isinstance(rng_states, dict):
        noncanonical_keys = [key for key in rng_states if type(key) is not str]
        expected_rank_keys = {str(rank) for rank in range(world_size)}
        if noncanonical_keys or set(rng_states) != expected_rank_keys:
            raise ValueError(
                "Training-resume checkpoint RNG world-size/rank-state mismatch: "
                f"noncanonical={noncanonical_keys!r}, checkpoint={sorted(map(str, rng_states))}, "
                f"current={sorted(expected_rank_keys)}."
            )
        for rank in range(world_size):
            validate_rng_checkpoint_state(
                rng_states[str(rank)],
                path=f"rng_state_by_rank[{rank}]",
                # The short-lived preflight intentionally does not initialize
                # CUDA generators. PPO.load performs device-count and generator
                # validation on the actual rank before mutating model state.
                validate_cuda_generators=False,
            )
        rng_stream_complete = True
    elif rng_states is not None:
        raise ValueError("Training-resume checkpoint rng_state_by_rank must be a mapping when present.")
    elif not _allow_nondeterministic_rng_resume():
        raise ValueError(
            "Checkpoint has no saved rank-local Python/NumPy/torch RNG state. This cannot continue the "
            "training stochastic streams. Set "
            f"{ALLOW_NONDETERMINISTIC_RNG_RESUME_ENV}=1 only to explicitly accept a non-deterministic "
            "legacy resume."
        )
    else:
        print(
            "[WARN] nondeterministic_rng_resume_allowed "
            f"override={ALLOW_NONDETERMINISTIC_RNG_RESUME_ENV}=1: checkpoint has no rng_state_by_rank; "
            "Python/NumPy/torch stochastic streams restart from the live post-construction state.",
            flush=True,
        )

    fixed_bc_states = checkpoint.get("fixed_bc_eval_by_rank")
    if fixed_bc_states is not None:
        try:
            if not isinstance(fixed_bc_states, dict):
                raise ValueError("Training-resume checkpoint fixed_bc_eval_by_rank must be a mapping when present.")
            noncanonical_keys = [key for key in fixed_bc_states if type(key) is not str]
            if noncanonical_keys:
                raise ValueError(
                    "Training-resume checkpoint fixed_bc_eval_by_rank keys must be canonical decimal strings; "
                    f"got non-string keys={noncanonical_keys!r}."
                )
            expected_rank_keys = {str(rank) for rank in range(world_size)}
            if set(fixed_bc_states) != expected_rank_keys:
                raise ValueError(
                    "Training-resume checkpoint fixed-BC world-size/rank-state mismatch: "
                    f"checkpoint ranks={sorted(fixed_bc_states)}, current ranks={sorted(expected_rank_keys)}."
                )
            malformed_ranks = [
                rank for rank, state in fixed_bc_states.items() if not isinstance(state, dict) or not state
            ]
            if malformed_ranks:
                raise ValueError(
                    "Training-resume checkpoint has malformed fixed-BC state for ranks "
                    f"{sorted(malformed_ranks)}."
                )
            validate_finite_tree(fixed_bc_states, path="fixed_bc_eval_by_rank")
        except ValueError as exc:
            if not _allow_fixed_bc_reset_on_resume():
                raise
            print(
                f"[WARN] fixed_bc_reset_on_resume_allowed override={ALLOW_FIXED_BC_RESET_ENV}=1 "
                f"reason={exc}",
                flush=True,
            )

    if rollout_contract is None:
        rollout_resume_mode = "legacy_new_episode_on_resume"
    print(
        "[INFO] training_resume_preflight_verified "
        f"checkpoint={checkpoint_path} iter={saved_iter} next_iter={next_iter} "
        f"target_iter={target_iteration} remaining_iterations={remaining_iterations} world_size={world_size} "
        f"resume_mode={'curriculum_and_rng_stream_complete' if rng_stream_complete else 'legacy_rng_missing'}"
        f"_not_bitwise_trajectory rollout_resume_mode={rollout_resume_mode}",
        flush=True,
    )
    return actual_resume_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--allow-fresh-curriculum", action="store_true")
    parser.add_argument("--current-provenance-json")
    args = parser.parse_args()
    current_config = json.load(sys.stdin)
    current_provenance = parse_training_provenance(args.current_provenance_json)
    validate_resume_checkpoint(
        args.checkpoint.expanduser().resolve(),
        current_config,
        world_size=args.world_size,
        allow_fresh_curriculum=args.allow_fresh_curriculum,
        current_provenance=current_provenance,
    )


if __name__ == "__main__":
    main()
