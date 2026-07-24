"""Strict training-input provenance shared by launch and checkpoint resume."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from typing import Any


ENV_NAME = "HOLOSOMA_TRAINING_PROVENANCE"
ALLOW_LEGACY_UNVERIFIED_TEACHER_LOAD_ENV = (
    "HOLOSOMA_ALLOW_LEGACY_UNVERIFIED_TEACHER_LOAD"
)
PROVENANCE_VERSION = 2
TRAINING_REGIME_KEY = "training_regime"
TRAINING_REGIME_DISTILLATION = "distillation"
TRAINING_REGIME_PURE_RL = "pure_rl"
TEACHER_ENABLED_KEY = "teacher_enabled"
MOTION_GENERATOR_TEACHER_SHA256_KEY = "motion_generator_teacher_sha256"
REQUIRE_MOTION_GENERATOR_TEACHER_MATCH_KEY = "require_motion_generator_teacher_match"
RUNTIME_ASSET_DIGEST_KEY = "runtime_asset_manifest_sha256"
RUNTIME_ASSET_MANIFEST_KEY = "runtime_asset_manifest"
RUNTIME_ASSET_PHASE_KEY = "runtime_asset_manifest_phase"
RUNTIME_ASSET_PHASE_PENDING = "pending"
RUNTIME_ASSET_PHASE_FINAL = "final"
RUNTIME_ASSET_MANIFEST_VERSION = 2
REQUIRED_DIGEST_KEYS = (
    "teacher_sha256",
    "policy_init_sha256",
    "training_resume_sha256",
    "motion_shard_manifest_sha256",
    "contact_sidecar_manifest_sha256",
    "source_bundle_sha256",
    RUNTIME_ASSET_DIGEST_KEY,
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_SNAPSHOT_ID_RE = re.compile(r"^src-([0-9a-f]{64})$")
_OPTIONAL_CHECKPOINT_ROLES = ("policy_init", "training_resume")
EXECUTION_RUNTIME_KEY = "execution_runtime"
SEMANTIC_ENVIRONMENT_KEY = "semantic_environment"

# These are normalized, process-wide launch controls already present in the
# provenance v2 execution_runtime mapping.  Unlike RANK/LOCAL_RANK/NODE_RANK,
# their values must be identical in every worker.  In particular, the
# rank-visible entrypoint is allowed to rewrite LOCAL_RANK/LOCAL_WORLD_SIZE and
# CUDA_VISIBLE_DEVICES, but it must not rewrite this contract.
_EXECUTION_RUNTIME_BOOL_DEFAULTS: tuple[tuple[str, bool], ...] = (
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
_EXECUTION_RUNTIME_INT_DEFAULTS: tuple[tuple[str, int, int], ...] = (
    ("HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC", 300, 1),
    ("HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH", 0, 0),
    ("NPROC", 1, 1),
    ("NNODES", 1, 1),
)

# Ambient controls are not represented by the Tyro config even though they
# change the student architecture, observations, reset distribution, data
# assignment, or optimizer numerics.  This fixed schema is shared by the
# provenance generator, live launch binding, and exact-resume validator.
SEMANTIC_ENVIRONMENT_FIELDS: tuple[str, ...] = (
    "HOLOSOMA_CAMERA_AUTOFIX_BACKWARD",
    "HOLOSOMA_CAMERA_BACKWARD_RATIO_THRESHOLD",
    "HOLOSOMA_CAMERA_DISABLE_OFFSETS",
    "HOLOSOMA_CAMERA_EXTRA_YAW_DEG",
    "HOLOSOMA_CAMERA_RANDOMIZE_PLACEMENT",
    "HOLOSOMA_CAMERA_STRICT_WARP",
    "HOLOSOMA_DEFM_FORWARD_BATCH_SIZE",
    "HOLOSOMA_DEFAULT_POSE_INIT",
    "HOLOSOMA_DISABLE_ACTIVE_OBS_GROUP_FILTER",
    "HOLOSOMA_DISABLE_AUTO_RESET",
    "HOLOSOMA_DISABLE_BAD_TRACKING_RESET",
    "HOLOSOMA_DISABLE_CLIP_END_RESET",
    "HOLOSOMA_DISABLE_MOTION_END_RESET",
    "HOLOSOMA_DISABLE_ONLINE_CONTACT_PRIOR",
    "HOLOSOMA_FAR_TRACKING_DISABLE_COMBINED_DEPTH_MESHES",
    "HOLOSOMA_FORCE_ROUND_ROBIN_CLIP_ASSIGNMENT",
    "HOLOSOMA_FORCE_ROUND_ROBIN_CLIP_START",
    "HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE",
    "HOLOSOMA_MUJOCO_RESET_NOISE",
    "HOLOSOMA_ONLINE_CONTACT_PRIOR",
    "HOLOSOMA_PERCEPTION_INCLUDE_ROBOT_MESH",
    "HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES",
    "HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE",
    "HOLOSOMA_PERCEPTION_SENSOR_OFFSET_DELTA",
    "HOLOSOMA_PERCEPTION_SENSOR_OFFSET_OVERRIDE",
    "HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS",
    "HOLOSOMA_RESET_TO_DEFAULT_POSE",
    "HOLOSOMA_STRICT_PERCEPTION_CAMERA_SOURCE",
    "HOLOSOMA_STRICT_PERCEPTION_OBJECT_MESHES",
    "HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE",
    "ISAAC_SCANDOTS_INCLUDE_MISSES",
    "ISAAC_SCANDOTS_USE_DEPTH_MASK",
)


def semantic_environment_from_environ(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    """Return the canonical raw training-semantic environment identity."""

    if environ is None:
        environ = os.environ
    return {
        name: (environ[name].strip() if name in environ else None)
        for name in SEMANTIC_ENVIRONMENT_FIELDS
    }


def validate_semantic_environment_binding(
    provenance: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    """Bind recorded ambient semantics to the process that will train.

    Launchers compute phase-one provenance before importing the simulator.  A
    later export must not silently make the live worker consume a different
    environment from the one recorded in that provenance.
    """

    environment = provenance.get("environment")
    execution_runtime = (
        environment.get(EXECUTION_RUNTIME_KEY)
        if isinstance(environment, Mapping)
        else None
    )
    recorded = (
        execution_runtime.get(SEMANTIC_ENVIRONMENT_KEY)
        if isinstance(execution_runtime, Mapping)
        else None
    )
    path = f"environment.{EXECUTION_RUNTIME_KEY}.{SEMANTIC_ENVIRONMENT_KEY}"
    if not isinstance(recorded, Mapping):
        raise ValueError(f"training provenance {path} must be a JSON object")
    expected_keys = set(SEMANTIC_ENVIRONMENT_FIELDS)
    actual_keys = set(recorded)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys.difference(actual_keys))
        unexpected = sorted(repr(key) for key in actual_keys.difference(expected_keys))
        raise ValueError(
            f"training provenance {path} keys must exactly match the scientific schema; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    normalized_recorded: dict[str, str | None] = {}
    for name in SEMANTIC_ENVIRONMENT_FIELDS:
        value = recorded[name]
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"training provenance {path}.{name} must be a string or null, got {value!r}"
            )
        if isinstance(value, str) and value != value.strip():
            raise ValueError(
                f"training provenance {path}.{name} must be a stripped canonical string, got {value!r}"
            )
        normalized_recorded[name] = value

    live = semantic_environment_from_environ(environ)
    differences = [
        f"{name}: recorded={normalized_recorded[name]!r} live={live[name]!r}"
        for name in SEMANTIC_ENVIRONMENT_FIELDS
        if normalized_recorded[name] != live[name]
    ]
    if differences:
        raise ValueError(
            "Training semantic environment changed after provenance generation: "
            + "; ".join(differences)
        )
    return normalized_recorded


def normalized_execution_bool_from_environ(
    environ: Mapping[str, str],
    name: str,
    *,
    default: bool,
) -> bool:
    raw_value = environ.get(name)
    if raw_value is None:
        return default
    if raw_value != raw_value.strip():
        raise ValueError(
            f"{name} must not contain surrounding whitespace, got {raw_value!r}"
        )
    if name == "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE":
        if raw_value == "1":
            return True
        if raw_value == "0":
            return False
        raise ValueError(
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE must be exactly 0 or 1 to match "
            f"PyTorch c10::utils::check_env semantics, got {raw_value!r}"
        )
    normalized = raw_value.lower()
    if name == "HOLOSOMA_CONTIGUOUS_MINIBATCHES" and normalized == "off":
        raise ValueError(
            "HOLOSOMA_CONTIGUOUS_MINIBATCHES does not accept 'off': its runtime consumer "
            "interprets that spelling as enabled; use 0 instead"
        )
    if name == "HOLOSOMA_RANK_VISIBLE_DEVICES" and normalized == "":
        raise ValueError(
            "HOLOSOMA_RANK_VISIBLE_DEVICES must not be explicitly empty because the "
            "rank-visible entrypoint interprets an empty value using its enabled default; use 0 or 1"
        )
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"", "0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw_value!r}")


def validate_hierarchical_small_collectives_contract(
    execution_runtime: Mapping[str, Any],
) -> None:
    """Reject a partial hierarchical-control-plane configuration.

    Hierarchical small collectives reuse the existing Gloo control plane and
    the node-leader topology established by hierarchical gradient reduction.
    The leader gradient device is intentionally independent: both the
    CPU/Gloo-leader and GPU/NCCL-leader gradient paths support this contract.
    """

    if execution_runtime.get("HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES") is not True:
        return
    if (
        execution_runtime.get("HOLOSOMA_GLOO_SMALL_COLLECTIVES") is not True
        or execution_runtime.get("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE") is not True
    ):
        raise ValueError(
            "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=1 requires "
            "HOLOSOMA_GLOO_SMALL_COLLECTIVES=1 and "
            "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1"
        )


def normalized_execution_int_from_environ(
    environ: Mapping[str, str],
    name: str,
    *,
    default: int,
    minimum: int,
) -> int:
    raw_value = environ.get(name)
    if raw_value is None:
        return default
    if re.fullmatch(r"[0-9]+", raw_value, flags=re.ASCII) is None:
        requirement = (
            "a base-10 non-negative integer"
            if minimum == 0
            else f"an ASCII base-10 integer >= {minimum}"
        )
        raise ValueError(
            f"{name} must be {requirement} without surrounding whitespace, got {raw_value!r}"
        )
    value = int(raw_value, 10)
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {raw_value!r}")
    return value


def _disabled_execution_runtime_component_sha256(component: str) -> str:
    payload = json.dumps(
        {"version": 1, "component": component, "disabled": True},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _execution_runtime_binding_from_environ(
    environ: Mapping[str, str],
) -> dict[str, Any]:
    """Normalize live, cross-rank execution controls using generator rules."""

    backend = environ.get("TORCH_DIST_BACKEND", "nccl").strip().lower()
    if backend not in {"nccl", "gloo"}:
        raise ValueError(
            "TORCH_DIST_BACKEND must be nccl or gloo, "
            f"got {environ.get('TORCH_DIST_BACKEND')!r}"
        )

    bool_values = {
        name: normalized_execution_bool_from_environ(
            environ,
            name,
            default=default,
        )
        for name, default in _EXECUTION_RUNTIME_BOOL_DEFAULTS
    }
    validate_hierarchical_small_collectives_contract(bool_values)
    raw_nccl_lib_sha256 = environ.get("NCCL_LIB_SHA256", "")
    if raw_nccl_lib_sha256 != raw_nccl_lib_sha256.strip():
        raise ValueError(
            "NCCL_LIB_SHA256 must not contain surrounding whitespace, "
            f"got {raw_nccl_lib_sha256!r}"
        )
    nccl_lib_sha256 = raw_nccl_lib_sha256
    if nccl_lib_sha256:
        if _SHA256_RE.fullmatch(nccl_lib_sha256) is None:
            raise ValueError(
                "NCCL_LIB_SHA256 must be a 64-character lowercase SHA256 hex digest"
            )
    elif backend == "nccl" or bool_values["HOLOSOMA_HIERARCHICAL_GRAD_REDUCE"]:
        raise ValueError(
            "NCCL_LIB_SHA256 is required when the default backend or hierarchical local "
            "gradient reduction uses NCCL, so the collective runtime is immutable"
        )
    else:
        nccl_lib_sha256 = _disabled_execution_runtime_component_sha256("nccl_library")

    binding: dict[str, Any] = {
        "NCCL_LIB_SHA256": nccl_lib_sha256,
        "TORCH_DIST_BACKEND": backend,
        **bool_values,
    }
    for name, default, minimum in _EXECUTION_RUNTIME_INT_DEFAULTS:
        binding[name] = normalized_execution_int_from_environ(
            environ,
            name,
            default=default,
            minimum=minimum,
        )
    return binding


def _validate_recorded_execution_runtime(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    environment = provenance.get("environment")
    recorded = (
        environment.get(EXECUTION_RUNTIME_KEY)
        if isinstance(environment, Mapping)
        else None
    )
    path = f"environment.{EXECUTION_RUNTIME_KEY}"
    if not isinstance(recorded, Mapping):
        raise ValueError(f"training provenance {path} must be a JSON object")
    recorded = dict(recorded)
    # Hierarchical small collectives and the explicit 16-minibatch canary label
    # were introduced additively, so absent fields unambiguously mean their
    # established disabled paths.  The subgroup timeout is operational
    # fail-closed behavior rather than a successful-step numerical input, so
    # legacy v2 payloads safely adopt the current bounded default.  Every other
    # missing or unexpected execution key remains fail-closed below.
    recorded.setdefault("HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES", False)
    recorded.setdefault("HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY", False)
    recorded.setdefault("HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC", 300)

    expected_keys = {
        "NCCL_LIB_SHA256",
        "TORCH_DIST_BACKEND",
        "PYTHONHASHSEED",
        "CUBLAS_WORKSPACE_CONFIG",
        SEMANTIC_ENVIRONMENT_KEY,
        *(name for name, _default in _EXECUTION_RUNTIME_BOOL_DEFAULTS),
        *(name for name, _default, _minimum in _EXECUTION_RUNTIME_INT_DEFAULTS),
    }
    actual_keys = set(recorded)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys.difference(actual_keys))
        unexpected = sorted(repr(key) for key in actual_keys.difference(expected_keys))
        raise ValueError(
            f"training provenance {path} keys must exactly match the current execution schema; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )

    normalized: dict[str, Any] = {}
    for name, _default in _EXECUTION_RUNTIME_BOOL_DEFAULTS:
        value = recorded.get(name)
        if type(value) is not bool:
            raise ValueError(
                f"training provenance {path}.{name} must be a boolean, got {value!r}"
            )
        normalized[name] = value
    validate_hierarchical_small_collectives_contract(normalized)
    for name, _default, minimum in _EXECUTION_RUNTIME_INT_DEFAULTS:
        value = recorded.get(name)
        if type(value) is not int or value < minimum:
            raise ValueError(
                f"training provenance {path}.{name} must be an integer >= {minimum}, "
                f"got {value!r}"
            )
        normalized[name] = value

    backend = recorded.get("TORCH_DIST_BACKEND")
    if backend not in {"nccl", "gloo"}:
        raise ValueError(
            f"training provenance {path}.TORCH_DIST_BACKEND must be 'nccl' or 'gloo', "
            f"got {backend!r}"
        )
    normalized["TORCH_DIST_BACKEND"] = backend
    nccl_lib_sha256 = recorded.get("NCCL_LIB_SHA256")
    if not isinstance(nccl_lib_sha256, str) or _SHA256_RE.fullmatch(nccl_lib_sha256) is None:
        raise ValueError(
            f"training provenance {path}.NCCL_LIB_SHA256 must be a lowercase SHA256 "
            f"hex digest, got {nccl_lib_sha256!r}"
        )
    normalized["NCCL_LIB_SHA256"] = nccl_lib_sha256

    python_hash_seed = recorded.get("PYTHONHASHSEED")
    if not isinstance(python_hash_seed, str) or not (
        python_hash_seed == "<unset>"
        or (
            python_hash_seed.isdecimal()
            and 0 <= int(python_hash_seed, 10) <= 4294967295
            and str(int(python_hash_seed, 10)) == python_hash_seed
        )
    ):
        raise ValueError(
            f"training provenance {path}.PYTHONHASHSEED must be '<unset>' or a canonical "
            f"integer string in [0, 4294967295], got {python_hash_seed!r}"
        )
    cublas_workspace = recorded.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas_workspace not in {"<unset>", ":4096:8", ":16:8"}:
        raise ValueError(
            f"training provenance {path}.CUBLAS_WORKSPACE_CONFIG must be '<unset>', "
            f"':4096:8', or ':16:8', got {cublas_workspace!r}"
        )
    normalized["PYTHONHASHSEED"] = python_hash_seed
    normalized["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace
    return normalized


def _strict_live_topology_int(
    environ: Mapping[str, str],
    name: str,
    *,
    default: int,
    minimum: int,
) -> int:
    raw_value = environ.get(name)
    if raw_value is None:
        return default
    if raw_value != raw_value.strip() or not raw_value.isascii() or not raw_value.isdecimal():
        raise ValueError(
            f"Live {name} must be a canonical base-10 integer >= {minimum}, got {raw_value!r}"
        )
    value = int(raw_value, 10)
    if value < minimum:
        raise ValueError(
            f"Live {name} must be a canonical base-10 integer >= {minimum}, got {raw_value!r}"
        )
    return value


def _validate_live_execution_topology(
    recorded: Mapping[str, Any],
    environ: Mapping[str, str],
) -> None:
    """Validate torchrun topology through rank-invariant relationships.

    Rank values themselves are intentionally absent from provenance.  These
    checks only prove that each rank-local identity is a valid member of the
    recorded NPROC x NNODES topology before simulator/environment creation.
    """

    nproc = int(recorded["NPROC"])
    nnodes = int(recorded["NNODES"])
    expected_world_size = nproc * nnodes
    world_size = _strict_live_topology_int(
        environ,
        "WORLD_SIZE",
        default=1,
        minimum=1,
    )
    if world_size != expected_world_size:
        raise ValueError(
            "Live distributed topology does not match training provenance: "
            f"WORLD_SIZE={world_size}, recorded NPROC={nproc}, recorded NNODES={nnodes}, "
            f"expected WORLD_SIZE={expected_world_size}"
        )
    global_rank = _strict_live_topology_int(
        environ,
        "RANK",
        default=0,
        minimum=0,
    )
    if global_rank >= world_size:
        raise ValueError(
            f"Live RANK must be in [0, WORLD_SIZE), got RANK={global_rank}, "
            f"WORLD_SIZE={world_size}"
        )

    rank_visible = bool(recorded["HOLOSOMA_RANK_VISIBLE_DEVICES"])
    original_local_rank_raw = environ.get("HOLOSOMA_ORIGINAL_LOCAL_RANK", "")
    original_local_world_raw = environ.get("HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE", "")
    if rank_visible and nproc > 1:
        if original_local_rank_raw == "" or original_local_world_raw == "":
            raise ValueError(
                "Rank-visible multi-process execution requires "
                "HOLOSOMA_ORIGINAL_LOCAL_RANK and HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE "
                "before simulator startup"
            )
        remapped_local_rank = _strict_live_topology_int(
            environ,
            "LOCAL_RANK",
            default=0,
            minimum=0,
        )
        remapped_local_world_size = _strict_live_topology_int(
            environ,
            "LOCAL_WORLD_SIZE",
            default=1,
            minimum=1,
        )
        if remapped_local_rank != 0 or remapped_local_world_size != 1:
            raise ValueError(
                "Rank-visible worker must expose one remapped local rank: "
                f"LOCAL_RANK={remapped_local_rank}, LOCAL_WORLD_SIZE={remapped_local_world_size}"
            )

    if rank_visible and original_local_rank_raw != "":
        effective_local_rank = _strict_live_topology_int(
            environ,
            "HOLOSOMA_ORIGINAL_LOCAL_RANK",
            default=0,
            minimum=0,
        )
    else:
        effective_local_rank = _strict_live_topology_int(
            environ,
            "LOCAL_RANK",
            default=0,
            minimum=0,
        )
    if rank_visible and original_local_world_raw != "":
        effective_local_world_size = _strict_live_topology_int(
            environ,
            "HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE",
            default=1,
            minimum=1,
        )
    else:
        effective_local_world_size = _strict_live_topology_int(
            environ,
            "LOCAL_WORLD_SIZE",
            default=1,
            minimum=1,
        )
    if effective_local_world_size != nproc:
        source = (
            "HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE"
            if rank_visible and original_local_world_raw != ""
            else "LOCAL_WORLD_SIZE"
        )
        raise ValueError(
            "Live node-local topology does not match training provenance: "
            f"{source}={effective_local_world_size}, recorded NPROC={nproc}"
        )
    if effective_local_rank >= nproc:
        raise ValueError(
            f"Live local rank must be in [0, NPROC), got local_rank={effective_local_rank}, "
            f"NPROC={nproc}"
        )
    expected_local_rank = global_rank % nproc
    if effective_local_rank != expected_local_rank:
        raise ValueError(
            "Live global/local ranks are inconsistent with recorded NPROC: "
            f"RANK={global_rank}, local_rank={effective_local_rank}, NPROC={nproc}, "
            f"expected_local_rank={expected_local_rank}"
        )
    if "NODE_RANK" in environ:
        node_rank = _strict_live_topology_int(
            environ,
            "NODE_RANK",
            default=0,
            minimum=0,
        )
        expected_node_rank = global_rank // nproc
        if node_rank >= nnodes or node_rank != expected_node_rank:
            raise ValueError(
                "Live NODE_RANK is inconsistent with global rank and recorded topology: "
                f"NODE_RANK={node_rank}, RANK={global_rank}, NPROC={nproc}, NNODES={nnodes}, "
                f"expected_NODE_RANK={expected_node_rank}"
            )


def validate_execution_runtime_binding(
    provenance: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Bind recorded process-wide execution controls to the live worker.

    ``PYTHONHASHSEED`` and ``CUBLAS_WORKSPACE_CONFIG`` are validated here as
    canonical recorded values, but their live/pre-interpreter identity remains
    the responsibility of train_agent's dedicated prestarted-runtime check.
    Rank identities and CUDA visibility are deliberately excluded from value
    equality because torchrun and the rank-visible wrapper make them
    rank-local; only their derived topology relationships are checked.
    """

    if environ is None:
        environ = os.environ
    normalized_recorded = _validate_recorded_execution_runtime(provenance)
    live = _execution_runtime_binding_from_environ(environ)
    differences = [
        f"{name}: recorded={normalized_recorded[name]!r} live={value!r}"
        for name, value in live.items()
        if normalized_recorded[name] != value
    ]
    if differences:
        raise ValueError(
            "Training execution runtime changed after provenance generation: "
            + "; ".join(differences)
        )
    _validate_live_execution_topology(normalized_recorded, environ)
    return normalized_recorded


def allow_legacy_unverified_teacher_load() -> bool:
    """Return the exact, explicitly non-scientific teacher-load hatch."""

    raw_value = os.environ.get(ALLOW_LEGACY_UNVERIFIED_TEACHER_LOAD_ENV)
    if raw_value is None or raw_value.strip() in ("", "0"):
        return False
    if raw_value.strip() == "1":
        return True
    raise ValueError(
        f"{ALLOW_LEGACY_UNVERIFIED_TEACHER_LOAD_ENV} must be exactly 0 or 1; "
        f"got {raw_value!r}."
    )


def canonical_runtime_asset_manifest_json(value: Any) -> str:
    """Return the only byte encoding used for runtime-asset identity.

    The full canonical manifest is embedded in finalized provenance, persisted
    beside the experiment config, and copied into checkpoints.  Keeping the
    encoder here lets the provenance validator independently verify that the
    declared digest actually describes the embedded manifest.
    """

    if not isinstance(value, dict):
        raise ValueError("runtime asset manifest must be a JSON object")
    if value.get("version") != RUNTIME_ASSET_MANIFEST_VERSION:
        raise ValueError(
            "unsupported runtime asset manifest version "
            f"{value.get('version')!r}; expected {RUNTIME_ASSET_MANIFEST_VERSION}"
        )
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"runtime asset manifest is not canonical JSON data: {exc}") from exc


def embedded_runtime_asset_manifest_sha256(value: Any) -> str:
    """Hash an embedded runtime-asset manifest using its canonical encoding."""

    return hashlib.sha256(canonical_runtime_asset_manifest_json(value).encode("utf-8")).hexdigest()


def pending_runtime_asset_manifest_sha256() -> str:
    """Return the only digest sentinel accepted during launch-time phase one."""

    payload = json.dumps(
        {
            "version": PROVENANCE_VERSION,
            "component": "runtime_asset_manifest",
            "phase": RUNTIME_ASSET_PHASE_PENDING,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def disabled_checkpoint_sha256(role: str) -> str:
    """Return the domain-separated digest that represents an absent checkpoint."""

    if role not in _OPTIONAL_CHECKPOINT_ROLES:
        raise ValueError(f"unsupported optional checkpoint role {role!r}")
    payload = json.dumps(
        {"version": 1, "role": role, "disabled": True},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def disabled_teacher_sha256() -> str:
    """Return the domain-separated identity for a run with no teacher.

    Pure-RL training still uses the common provenance/checkpoint schema, but
    it must not invent a teacher checkpoint merely to populate a digest field.
    The explicit mode plus this sentinel makes the absence itself immutable.
    """

    payload = json.dumps(
        {"version": 1, "role": "teacher", "disabled": True},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def disabled_contact_sidecar_manifest_sha256() -> str:
    """Return the launcher's canonical sentinel for disabled contact data."""

    payload = json.dumps(
        {"version": 1, "disabled": True},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checkpoint_lineage_enabled(provenance: dict[str, Any], role: str) -> bool:
    """Return an optional checkpoint's explicitly declared v2 mode."""

    provenance = validate_training_provenance(provenance)
    enabled_key = f"{role}_enabled"
    return provenance[enabled_key]


def validate_training_provenance(
    value: Any,
    *,
    require_finalized: bool = False,
) -> dict[str, Any]:
    """Validate either phase of a v2 provenance payload.

    Generic validation accepts the launch-time pending phase so the provenance
    generator itself can be tested.  Simulator/checkpoint consumers call this
    with ``require_finalized=True`` (directly or through the strict parsing
    helpers below).  Version 1 is intentionally rejected: it made no claim
    about the robot/perception/terrain bytes and therefore cannot support a
    scientific resume.
    """

    if not isinstance(value, dict):
        raise ValueError("training provenance must be a JSON object")
    provenance = dict(value)
    provenance_version = provenance.get("version")
    if type(provenance_version) is not int or provenance_version != PROVENANCE_VERSION:
        raise ValueError(
            f"unsupported training provenance version {provenance_version!r}; "
            f"expected {PROVENANCE_VERSION}"
        )
    environment = provenance.get("environment")
    if isinstance(environment, Mapping):
        execution_runtime = environment.get(EXECUTION_RUNTIME_KEY)
        if isinstance(execution_runtime, Mapping) and (
            "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES" not in execution_runtime
            or "HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY" not in execution_runtime
            or "HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC" not in execution_runtime
        ):
            normalized_execution_runtime = dict(execution_runtime)
            normalized_execution_runtime.setdefault(
                "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES",
                False,
            )
            normalized_execution_runtime.setdefault(
                "HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC",
                300,
            )
            normalized_execution_runtime.setdefault(
                "HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY",
                False,
            )
            normalized_environment = dict(environment)
            normalized_environment[EXECUTION_RUNTIME_KEY] = normalized_execution_runtime
            provenance["environment"] = normalized_environment
    for key in REQUIRED_DIGEST_KEYS:
        digest = provenance.get(key)
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"training provenance {key} must be a lowercase SHA256 hex digest")

    # ``training_regime`` and ``teacher_enabled`` were added compatibly to the
    # v2 payload after scientific pure-RL launches began sharing this schema.
    # Old v2 checkpoints could only have been produced by the distillation
    # generator, so normalize that legacy representation before comparison or
    # re-serialization.  New payloads must state both fields together.
    regime_present = TRAINING_REGIME_KEY in provenance
    teacher_enabled_present = TEACHER_ENABLED_KEY in provenance
    if regime_present != teacher_enabled_present:
        raise ValueError(
            f"training provenance {TRAINING_REGIME_KEY} and {TEACHER_ENABLED_KEY} "
            "must be present together"
        )
    if not regime_present:
        provenance[TRAINING_REGIME_KEY] = TRAINING_REGIME_DISTILLATION
        provenance[TEACHER_ENABLED_KEY] = True

    regime = provenance[TRAINING_REGIME_KEY]
    teacher_enabled = provenance[TEACHER_ENABLED_KEY]
    if regime not in {TRAINING_REGIME_DISTILLATION, TRAINING_REGIME_PURE_RL}:
        raise ValueError(
            f"training provenance {TRAINING_REGIME_KEY} must be "
            f"{TRAINING_REGIME_DISTILLATION!r} or {TRAINING_REGIME_PURE_RL!r}"
        )
    if not isinstance(teacher_enabled, bool):
        raise ValueError(f"training provenance {TEACHER_ENABLED_KEY} must be a boolean")
    # Older pure-RL v2 payloads predate the generalist launcher's explicit
    # contact-timebase binding; their live config could only use the legacy
    # false default.  Normalize that representation without allowing a
    # malformed non-boolean value to enter checkpoint/resume identity.
    provenance.setdefault("contact_interval_runtime_prepend_compensation", False)
    if not isinstance(
        provenance["contact_interval_runtime_prepend_compensation"], bool
    ):
        raise ValueError(
            "training provenance contact_interval_runtime_prepend_compensation must be a boolean"
        )
    disabled_teacher_digest = disabled_teacher_sha256()
    if regime == TRAINING_REGIME_DISTILLATION:
        if not teacher_enabled:
            raise ValueError(
                f"training provenance {TRAINING_REGIME_DISTILLATION} requires "
                f"{TEACHER_ENABLED_KEY}=True"
            )
        if provenance["teacher_sha256"] == disabled_teacher_digest:
            raise ValueError(
                "distillation training provenance cannot use the disabled teacher_sha256 sentinel"
            )
    else:
        if teacher_enabled:
            raise ValueError(
                f"training provenance {TRAINING_REGIME_PURE_RL} requires "
                f"{TEACHER_ENABLED_KEY}=False"
            )
        if provenance["teacher_sha256"] != disabled_teacher_digest:
            raise ValueError(
                "pure-RL training provenance requires the disabled teacher_sha256 sentinel"
            )
        forbidden_teacher_fields = sorted(
            key
            for key in ("teacher_motion_end_mode", "teacher_uses_action_history")
            if key in provenance
        )
        if forbidden_teacher_fields:
            raise ValueError(
                "pure-RL training provenance must not claim teacher semantics: "
                + ", ".join(forbidden_teacher_fields)
            )
    generator_present = MOTION_GENERATOR_TEACHER_SHA256_KEY in provenance
    generator_match_present = REQUIRE_MOTION_GENERATOR_TEACHER_MATCH_KEY in provenance
    if generator_present != generator_match_present:
        raise ValueError(
            "training provenance motion-generator teacher SHA and match mode must be present together"
        )
    if generator_present:
        generator_sha256 = provenance[MOTION_GENERATOR_TEACHER_SHA256_KEY]
        require_generator_match = provenance[REQUIRE_MOTION_GENERATOR_TEACHER_MATCH_KEY]
        if not isinstance(generator_sha256, str) or _SHA256_RE.fullmatch(generator_sha256) is None:
            raise ValueError(
                f"training provenance {MOTION_GENERATOR_TEACHER_SHA256_KEY} must be a lowercase SHA256 digest"
            )
        if not isinstance(require_generator_match, bool):
            raise ValueError(
                f"training provenance {REQUIRE_MOTION_GENERATOR_TEACHER_MATCH_KEY} must be a boolean"
            )
        if regime != TRAINING_REGIME_DISTILLATION:
            raise ValueError("pure-RL training provenance cannot claim a motion-generator teacher")
        if require_generator_match and provenance["teacher_sha256"] != generator_sha256:
            raise ValueError(
                "training provenance requires the distillation-label teacher to match the "
                "motion-generator teacher, but their SHA256 identities differ"
            )
    runtime_asset_phase = provenance.get(RUNTIME_ASSET_PHASE_KEY)
    if runtime_asset_phase not in {RUNTIME_ASSET_PHASE_PENDING, RUNTIME_ASSET_PHASE_FINAL}:
        raise ValueError(
            f"training provenance {RUNTIME_ASSET_PHASE_KEY} must be "
            f"{RUNTIME_ASSET_PHASE_PENDING!r} or {RUNTIME_ASSET_PHASE_FINAL!r}"
        )
    pending_digest = pending_runtime_asset_manifest_sha256()
    runtime_asset_digest = provenance[RUNTIME_ASSET_DIGEST_KEY]
    embedded_manifest = provenance.get(RUNTIME_ASSET_MANIFEST_KEY)
    if runtime_asset_phase == RUNTIME_ASSET_PHASE_PENDING:
        if runtime_asset_digest != pending_digest:
            raise ValueError(
                f"pending training provenance {RUNTIME_ASSET_DIGEST_KEY} must use the pending sentinel"
            )
        if require_finalized:
            raise ValueError(
                "training provenance runtime asset manifest is still pending; refusing simulator/checkpoint use"
            )
        if embedded_manifest is not None:
            raise ValueError(
                f"pending training provenance {RUNTIME_ASSET_MANIFEST_KEY} must be null or absent"
            )
    else:
        if runtime_asset_digest == pending_digest:
            raise ValueError(
                f"final training provenance {RUNTIME_ASSET_DIGEST_KEY} cannot use the pending sentinel"
            )
        if not isinstance(embedded_manifest, dict):
            raise ValueError(
                f"final training provenance {RUNTIME_ASSET_MANIFEST_KEY} must embed the canonical manifest"
            )
        embedded_digest = embedded_runtime_asset_manifest_sha256(embedded_manifest)
        if embedded_digest != runtime_asset_digest:
            raise ValueError(
                "final training provenance runtime asset manifest digest mismatch: "
                f"declared={runtime_asset_digest} embedded={embedded_digest}"
            )
    for role in _OPTIONAL_CHECKPOINT_ROLES:
        digest_key = f"{role}_sha256"
        enabled_key = f"{role}_enabled"
        if enabled_key not in provenance:
            raise ValueError(f"training provenance {enabled_key} is required by provenance v2")
        enabled = provenance[enabled_key]
        if not isinstance(enabled, bool):
            raise ValueError(f"training provenance {enabled_key} must be a boolean")
        disabled_digest = disabled_checkpoint_sha256(role)
        if enabled and provenance[digest_key] == disabled_digest:
            raise ValueError(
                f"training provenance {enabled_key}=True cannot use the disabled {digest_key} sentinel"
            )
        if not enabled and provenance[digest_key] != disabled_digest:
            raise ValueError(
                f"training provenance {enabled_key}=False requires the disabled {digest_key} sentinel"
            )
    snapshot_id_present = "source_snapshot_id" in provenance
    source_manifest_present = "source_manifest_sha256" in provenance
    if snapshot_id_present != source_manifest_present:
        raise ValueError(
            "training provenance source_snapshot_id and source_manifest_sha256 must be present together"
        )
    if snapshot_id_present:
        snapshot_id = provenance["source_snapshot_id"]
        source_manifest_sha256 = provenance["source_manifest_sha256"]
        snapshot_match = (
            _SOURCE_SNAPSHOT_ID_RE.fullmatch(snapshot_id) if isinstance(snapshot_id, str) else None
        )
        if snapshot_match is None:
            raise ValueError(
                "training provenance source_snapshot_id must have format "
                "src-<64 lowercase SHA256 hex>"
            )
        if (
            not isinstance(source_manifest_sha256, str)
            or _SHA256_RE.fullmatch(source_manifest_sha256) is None
        ):
            raise ValueError(
                "training provenance source_manifest_sha256 must be a lowercase SHA256 hex digest"
            )
        if snapshot_match.group(1) != source_manifest_sha256:
            raise ValueError(
                "training provenance source_snapshot_id digest must match source_manifest_sha256"
            )
    return provenance


def parse_training_provenance(
    raw: str | None,
    *,
    require_finalized: bool = True,
) -> dict[str, Any] | None:
    if raw is None or not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {ENV_NAME} JSON: {exc}") from exc
    return validate_training_provenance(value, require_finalized=require_finalized)


def training_provenance_from_env() -> dict[str, Any] | None:
    provenance = parse_training_provenance(os.environ.get(ENV_NAME), require_finalized=True)
    if provenance is not None:
        validate_execution_runtime_binding(provenance)
        validate_semantic_environment_binding(provenance)
    return provenance


def canonical_training_provenance_json(value: dict[str, Any]) -> str:
    return json.dumps(
        validate_training_provenance(value, require_finalized=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
