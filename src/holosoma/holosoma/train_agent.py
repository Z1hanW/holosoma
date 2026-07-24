from __future__ import annotations

import os

# This module has transitive top-level imports of torch.  Configure cuBLAS
# before any of them can initialize a CUDA handle; seeding() runs only after
# simulator/distributed startup and is too late to establish this contract.
_cublas_workspace_config = os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)
if _cublas_workspace_config not in {":4096:8", ":16:8"}:
    raise RuntimeError(
        "CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 before importing train_agent; "
        f"got {_cublas_workspace_config!r}."
    )
del _cublas_workspace_config

import dataclasses
import json
import logging
import re
import stat
import subprocess
import sys
import traceback
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, TypedDict, cast

import numpy as np
import tyro
from loguru import logger

from holosoma.agents.modules.logging_utils import collect_reward_wandb_metadata
from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.video import CartesianCameraConfig, FixedCameraConfig, SphericalCameraConfig, VideoConfig
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.observation import apply_observation_overrides
from holosoma.perception import apply_perception_overrides
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.atomic_output import emit_atomic_stdout_record
from holosoma.utils.common import rank_training_seed
from holosoma.utils.defm_runtime import set_defm_materialization_mode
from holosoma.utils.eval_utils import (
    init_sim_imports,
    load_checkpoint,
)
from holosoma.utils.experiment_paths import (
    get_experiment_dir,
    get_process_experiment_dir,
    get_timestamp,
    set_experiment_dir_override,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.policy_init_preflight import (
    required_policy_init_terminal_target_from_env,
)
from holosoma.utils.rotations import quat_apply, quat_from_euler_xyz, quat_rotate_inverse
from holosoma.utils.runtime_asset_manifest import (
    RUNTIME_ASSET_MANIFEST_FILENAME,
    finalize_runtime_asset_provenance,
    persist_runtime_asset_manifest,
)
from holosoma.utils.sim_utils import close_simulation_app
from holosoma.utils.training_provenance import (
    ALLOW_LEGACY_UNVERIFIED_TEACHER_LOAD_ENV,
    allow_legacy_unverified_teacher_load,
    canonical_training_provenance_json,
    checkpoint_lineage_enabled,
    disabled_contact_sidecar_manifest_sha256,
    training_provenance_from_env,
    validate_hierarchical_small_collectives_contract,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG


def _effective_runtime_config(config: ExperimentConfig) -> ExperimentConfig:
    """Apply every config rewrite that changes simulator/model runtime inputs."""

    config = apply_observation_overrides(config)
    config = apply_perception_overrides(config)
    return config


class TrainingContext:
    """Context manager for training lifecycle and resource management."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.simulation_app: Any | None = None
        self._policy_init_preflight_complete = False

    def __enter__(self):
        self.config = _effective_runtime_config(self.config)
        _validate_hierarchical_small_collectives_launch_contract()
        _current_rank_training_seed(self.config.training.seed)
        # A caller may hold this context after the launch-time preflights.
        # Re-hash immediately before importing/starting the simulator.
        finalized_provenance = finalize_runtime_asset_provenance(self.config)
        _validate_prestarted_runtime_provenance(finalized_provenance)
        _preflight_checkpoint_lineage_before_sim(self.config)
        _preflight_data_assets_before_sim()
        # ``TrainingContext`` is a public simulator-starting entrypoint, not
        # merely a wrapper around ``main``. Resolve and validate an actor
        # initializer here so an invalid/missing required terminal source can
        # never survive until Isaac is imported. The returned config replaces
        # a W&B URI with its verified local file, making later calls
        # idempotent without downloading the artifact again.
        self.config = _preflight_policy_init_before_sim(self.config)
        self._policy_init_preflight_complete = True
        # Initialize simulation app
        self.simulation_app = init_sim_imports(self.config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean shutdown using the utility function
        close_simulation_app(self.simulation_app)

    def train(self) -> None:
        """Train using this context's sim app."""
        train(self.config, training_context=self)


@contextmanager
def training_context(config: ExperimentConfig):
    """Context manager function for training."""
    with TrainingContext(config) as ctx:
        yield ctx


class MultGPUConfig(TypedDict):
    global_rank: int
    local_rank: int
    world_size: int


def _rank_training_seed(base_seed: int, *, world_size: int, global_rank: int) -> int:
    """Compatibility alias for the shared rank-seed contract."""

    return rank_training_seed(
        base_seed,
        world_size=world_size,
        global_rank=global_rank,
    )


def _current_rank_training_seed(base_seed: int) -> int:
    """Validate launcher topology and return this process's rank-local seed."""

    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        global_rank = int(os.environ.get("RANK", "0"))
    except ValueError as exc:
        raise ValueError(
            "WORLD_SIZE and RANK must be base-10 integers before simulator startup: "
            f"WORLD_SIZE={os.environ.get('WORLD_SIZE', '1')!r}, "
            f"RANK={os.environ.get('RANK', '0')!r}."
        ) from exc
    return _rank_training_seed(
        base_seed,
        world_size=world_size,
        global_rank=global_rank,
    )


def _validate_prestarted_runtime_provenance(provenance: dict[str, Any] | None) -> None:
    """Prove interpreter/cuBLAS settings match the launch-time provenance."""

    if provenance is None:
        return
    try:
        execution_runtime = provenance["environment"]["execution_runtime"]
        declared_hash_seed = execution_runtime["PYTHONHASHSEED"]
        declared_cublas = execution_runtime["CUBLAS_WORKSPACE_CONFIG"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "Scientific training provenance is missing pre-start PYTHONHASHSEED/"
            "CUBLAS_WORKSPACE_CONFIG identity."
        ) from exc

    raw_hash_seed = os.environ.get("PYTHONHASHSEED", "").strip()
    if not raw_hash_seed.isdecimal() or not 0 <= int(raw_hash_seed, 10) <= 4294967295:
        raise RuntimeError(
            "Scientific training requires PYTHONHASHSEED to be exported as an integer in "
            "[0, 4294967295] before Python starts."
        )
    actual_hash_seed = str(int(raw_hash_seed, 10))
    actual_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "").strip()
    if actual_cublas not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "Scientific training requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 "
            "before CUDA starts."
        )
    if declared_hash_seed != actual_hash_seed or declared_cublas != actual_cublas:
        raise RuntimeError(
            "Launch-time runtime provenance does not match this training process: "
            f"PYTHONHASHSEED declared={declared_hash_seed!r} actual={actual_hash_seed!r}; "
            f"CUBLAS_WORKSPACE_CONFIG declared={declared_cublas!r} actual={actual_cublas!r}."
        )


def _distributed_barrier(dist_module: Any, distributed_conf: MultGPUConfig | None) -> None:
    if distributed_conf is None or not dist_module.is_initialized():
        return
    try:
        dist_module.barrier(device_ids=[int(distributed_conf["local_rank"])])
    except TypeError:
        dist_module.barrier()


_LAUNCH_TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_SNAPSHOT_RE = re.compile(r"^src-[0-9a-f]{64}$")


def _emit_batch_worker_preflight_ready(
    *,
    dist_module: Any,
    distributed_conf: MultGPUConfig | None,
) -> bool:
    """Publish one launch-bound marker after the real worker is fully ready.

    This boundary is intentionally later than the lightweight pre-simulator
    provenance rendezvous.  Callers invoke it only after environment creation,
    algorithm setup/model synchronization and any full-resume or policy-init
    load.  The main training process group barrier proves every worker reached
    the same boundary before any marker is emitted.
    """

    launch_token = os.environ.get("HOLOSOMA_LAUNCH_TOKEN", "").strip()
    launch_epoch = os.environ.get("HOLOSOMA_LAUNCH_EPOCH", "").strip()
    if not launch_token and not launch_epoch:
        return False
    source_snapshot = os.environ.get("HOLOSOMA_SOURCE_SNAPSHOT_ID", "").strip()
    if not _LAUNCH_TOKEN_RE.fullmatch(launch_token):
        raise RuntimeError("HOLOSOMA_LAUNCH_TOKEN must be exactly 64 lowercase hexadecimal characters.")
    if not launch_epoch.isdecimal() or int(launch_epoch, 10) <= 0:
        raise RuntimeError("HOLOSOMA_LAUNCH_EPOCH must be a positive decimal Unix timestamp.")
    if not _SOURCE_SNAPSHOT_RE.fullmatch(source_snapshot):
        raise RuntimeError("Launch-bound worker readiness requires a valid HOLOSOMA_SOURCE_SNAPSHOT_ID.")

    global_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    original_local_rank_raw = os.environ.get("HOLOSOMA_ORIGINAL_LOCAL_RANK", "").strip()
    local_rank = int(original_local_rank_raw or os.environ.get("LOCAL_RANK", "0"))
    original_local_world_raw = os.environ.get("HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE", "").strip()
    local_world_size = int(original_local_world_raw or os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    nproc_raw = os.environ.get("NPROC", "").strip()
    nnodes_raw = os.environ.get("NNODES", "").strip()
    node_rank_raw = os.environ.get("NODE_RANK", "").strip()
    if not nproc_raw.isdecimal() or int(nproc_raw, 10) <= 0:
        raise RuntimeError("Launch-bound worker readiness requires a positive decimal NPROC.")
    if not nnodes_raw.isdecimal() or int(nnodes_raw, 10) <= 0:
        raise RuntimeError("Launch-bound worker readiness requires a positive decimal NNODES.")
    if not node_rank_raw.isdecimal():
        raise RuntimeError("Launch-bound worker readiness requires a non-negative decimal NODE_RANK.")
    nproc = int(nproc_raw, 10)
    nnodes = int(nnodes_raw, 10)
    node_rank = int(node_rank_raw, 10)
    if world_size < 1 or not 0 <= global_rank < world_size:
        raise RuntimeError(
            f"Invalid launch rank identity: global_rank={global_rank} world_size={world_size}."
        )
    if local_world_size < 1 or not 0 <= local_rank < local_world_size:
        raise RuntimeError(
            f"Invalid launch-local rank identity: local_rank={local_rank} local_world_size={local_world_size}."
        )
    if (
        world_size != nproc * nnodes
        or local_world_size != nproc
        or not 0 <= node_rank < nnodes
        or global_rank != node_rank * nproc + local_rank
    ):
        raise RuntimeError(
            "Launch-bound worker topology is inconsistent: "
            f"global_rank={global_rank}, local_rank={local_rank}, world_size={world_size}, "
            f"local_world_size={local_world_size}, NPROC={nproc}, NNODES={nnodes}, "
            f"NODE_RANK={node_rank}."
        )
    if distributed_conf is not None:
        if int(distributed_conf["global_rank"]) != global_rank or int(distributed_conf["world_size"]) != world_size:
            raise RuntimeError(
                "Main process-group identity differs from torchrun environment at worker-ready boundary: "
                f"distributed_conf={distributed_conf} RANK={global_rank} WORLD_SIZE={world_size}."
            )
        if not dist_module.is_initialized():
            raise RuntimeError(
                "Main training process group is not initialized at the launch-bound worker-ready boundary."
            )
    elif world_size != 1:
        raise RuntimeError(
            "A multi-worker launch cannot publish readiness without a distributed configuration/process group."
        )

    _distributed_barrier(dist_module, distributed_conf)
    emit_atomic_stdout_record(
        "[INFO] final_worker_preflight_verified "
        f"global_rank={global_rank} local_rank={local_rank} world_size={world_size} "
        f"source_snapshot={source_snapshot} launch_token={launch_token} launch_epoch={launch_epoch}"
    )
    return True


def _synchronize_experiment_identity(
    *,
    dist_module: Any,
    distributed_conf: MultGPUConfig | None,
    device: str,
    logger_config: Any,
    training_config: Any,
    task_name: str,
) -> tuple[str, Path]:
    """Create one fail-closed timestamp/log directory for every training rank."""

    if distributed_conf is None:
        timestamp = get_timestamp()
        experiment_dir = get_experiment_dir(
            logger_config,
            training_config,
            timestamp,
            task_name=task_name,
        )
    else:
        if not dist_module.is_initialized():
            raise RuntimeError("Distributed experiment identity requires an initialized default process group.")
        actual_rank = int(dist_module.get_rank())
        actual_world_size = int(dist_module.get_world_size())
        if actual_rank != int(distributed_conf["global_rank"]):
            raise RuntimeError(
                "Distributed rank changed after process-group initialization: "
                f"config={distributed_conf['global_rank']}, process_group={actual_rank}."
            )
        if actual_world_size != int(distributed_conf["world_size"]):
            raise RuntimeError(
                "Distributed world size changed after process-group initialization: "
                f"config={distributed_conf['world_size']}, process_group={actual_world_size}."
            )

        identity: list[tuple[str, str] | None] = [None]
        if actual_rank == 0:
            rank_zero_timestamp = get_timestamp()
            rank_zero_dir = get_experiment_dir(
                logger_config,
                training_config,
                rank_zero_timestamp,
                task_name=task_name,
            ).expanduser().resolve(strict=False)
            identity[0] = (rank_zero_timestamp, str(rank_zero_dir))

        backend = str(dist_module.get_backend()).strip().lower()
        import torch

        control_group = None
        use_gloo_control = backend == "nccl" and _bool_env(
            "HOLOSOMA_GLOO_SMALL_COLLECTIVES",
            default=False,
        )
        if use_gloo_control:
            # ``broadcast_object_list`` serializes through a small CPU tensor.
            # Sending that control-plane payload through NCCL needlessly couples
            # run-directory setup to every CUDA context while Isaac is still
            # starting.  The scientific launcher already requests Gloo for
            # small/control collectives, so honor that contract at the first
            # collective as well (before PPO has created its long-lived groups).
            dist_timeout_sec = int(os.getenv("TORCH_DIST_TIMEOUT_SEC", "600"))
            if dist_timeout_sec <= 0:
                raise ValueError(
                    f"TORCH_DIST_TIMEOUT_SEC must be positive, got {dist_timeout_sec}."
                )
            control_group = dist_module.new_group(
                backend="gloo",
                timeout=timedelta(seconds=dist_timeout_sec),
            )
            broadcast_device = torch.device("cpu")
            print(
                "[INFO] experiment_identity_collective backend=gloo "
                "reason=HOLOSOMA_GLOO_SMALL_COLLECTIVES",
                flush=True,
            )
        elif backend == "nccl":
            broadcast_device = torch.device(device)
            if broadcast_device.type != "cuda":
                raise RuntimeError(
                    "NCCL experiment-identity broadcast requires this rank's CUDA device, "
                    f"got {device!r}."
                )
        elif backend == "gloo":
            broadcast_device = torch.device("cpu")
        else:
            raise RuntimeError(
                "Experiment-identity broadcast supports the configured training backends "
                f"'nccl' and 'gloo', got {backend!r}."
            )
        try:
            if control_group is None:
                dist_module.broadcast_object_list(identity, src=0, device=broadcast_device)
            else:
                dist_module.broadcast_object_list(
                    identity,
                    src=0,
                    group=control_group,
                    device=broadcast_device,
                )
        finally:
            if control_group is not None:
                dist_module.destroy_process_group(control_group)
        received_identity = identity[0]
        if (
            not isinstance(received_identity, tuple)
            or len(received_identity) != 2
            or not all(isinstance(value, str) for value in received_identity)
        ):
            raise RuntimeError(f"Rank 0 broadcast an invalid experiment identity: {received_identity!r}.")
        timestamp, received_dir = received_identity
        experiment_dir = Path(received_dir)

    experiment_dir = set_experiment_dir_override(
        logger_config,
        training_config,
        timestamp=timestamp,
        experiment_dir=experiment_dir,
        task_name=task_name,
    )
    resolved_from_process = get_process_experiment_dir(
        logger_config,
        training_config,
        task_name=task_name,
        require_override=True,
    )
    if resolved_from_process != experiment_dir:
        raise RuntimeError(
            "Installed experiment identity did not round-trip through the process environment: "
            f"installed={experiment_dir}, resolved={resolved_from_process}."
        )
    return timestamp, experiment_dir


def _collect_object_bank_wandb_metadata() -> dict[str, int | str]:
    """Collect launcher-computed object-bank stats for W&B config/summary."""
    prefix = "HOLOSOMA_OBJECT_BANK_"
    raw_keys = {
        "TOTAL_MOTION_COUNT": "object_bank/total_motion_count",
        "TOTAL_UNIQUE_URDF_COUNT": "object_bank/total_unique_urdf_count",
        "BOX_MOTION_COUNT": "object_bank/box_motion_count",
        "BOX_UNIQUE_URDF_COUNT": "object_bank/box_unique_urdf_count",
        "OMOMO_MOTION_COUNT": "object_bank/omomo_motion_count",
        "OMOMO_UNIQUE_URDF_COUNT": "object_bank/omomo_unique_urdf_count",
        "MOTION_DIR": "object_bank/motion_dir",
        "OBJECT_MAP": "object_bank/object_map",
    }
    metadata: dict[str, int | str] = {}
    for env_suffix, metric_name in raw_keys.items():
        raw_value = os.environ.get(f"{prefix}{env_suffix}")
        if raw_value is None or raw_value == "":
            continue
        if env_suffix.endswith("_COUNT"):
            try:
                metadata[metric_name] = int(raw_value)
            except ValueError:
                logger.warning("Ignoring non-integer object-bank metadata {}={}", env_suffix, raw_value)
        else:
            metadata[metric_name] = raw_value
    return metadata


def _collect_env_count_wandb_metadata(
    *,
    requested_total_num_envs: int,
    effective_total_num_envs: int,
    per_rank_num_envs: int,
    world_size: int,
) -> dict[str, int | str]:
    """Collect environment-count metadata before W&B config hides distributed splitting."""

    metadata: dict[str, int | str] = {
        "training/num_envs_requested_total": int(requested_total_num_envs),
        "training/num_envs_effective_total": int(effective_total_num_envs),
        "training/num_envs_per_rank": int(per_rank_num_envs),
        "training/world_size": int(world_size),
    }

    int_env_keys = {
        "PER_GPU_ENVS": "launcher/per_gpu_envs",
        "TOTAL_NUM_ENVS": "launcher/total_num_envs",
        "NPROC": "launcher/nproc",
        "LOCAL_WORLD_SIZE": "launcher/local_world_size",
    }
    for env_key, metadata_key in int_env_keys.items():
        raw_value = os.environ.get(env_key)
        if raw_value is None or raw_value == "":
            continue
        try:
            metadata[metadata_key] = int(raw_value)
        except ValueError:
            logger.warning("Ignoring non-integer environment metadata {}={}", env_key, raw_value)

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        metadata["launcher/cuda_visible_devices"] = cuda_visible_devices

    return metadata


def _collect_training_provenance_wandb_metadata() -> dict[str, Any]:
    provenance = training_provenance_from_env()
    if provenance is None:
        return {}
    return {f"provenance/{key}": value for key, value in provenance.items()}


def _publish_wandb_startup_metadata(
    wandb_module: Any,
    *,
    config_metadata: dict[str, Any],
    summary_metadata: dict[str, Any] | None = None,
) -> None:
    """Publish run metadata without consuming a training-history step.

    W&B advances its internal history cursor when ``wandb.log`` commits a row,
    even when that call explicitly uses ``step=0``.  Startup metadata belongs
    in the immutable run config and summary, not in the iteration-indexed
    history.  Keeping this helper free of ``wandb.log`` preserves iteration 0
    for the first PPO metrics row.
    """

    if config_metadata:
        wandb_module.config.update(config_metadata, allow_val_change=True)
    if summary_metadata is None:
        summary_metadata = config_metadata
    for key, value in summary_metadata.items():
        wandb_module.run.summary[key] = value


def _wandb_init_failure_is_fatal(resume_mode: str | bool | None) -> bool:
    """A requested must-resume run must never silently continue without its lineage."""

    return isinstance(resume_mode, str) and resume_mode.strip().lower() == "must"


def _finish_wandb_run(wandb_module: Any, *, exit_code: int) -> None:
    """Finish the active run with its authoritative process outcome.

    Calling ``wandb.finish()`` without an exit code marks even a guard-triggered
    ``sys.exit(1)`` as a normally finished run.  Require every caller to state
    the outcome explicitly so the remote lifecycle cannot contradict the
    launcher/torchrun exit status.
    """

    if type(exit_code) is not int or exit_code < 0:
        raise ValueError(f"W&B exit_code must be a non-negative integer, got {exit_code!r}.")
    wandb_module.finish(exit_code=exit_code)


class WandbStartupOutcome(TypedDict):
    """Serializable rank-zero W&B startup result broadcast to every worker."""

    ok: bool
    run_path: str | None
    error_type: str | None
    error_message: str | None
    force_fatal: bool


_WANDB_RUN_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _validate_required_wandb_logger_mode(
    *,
    require_run: bool,
    wandb_enabled: bool,
    wandb_mode: str | None,
) -> None:
    """Reject logger settings that cannot satisfy a required cloud run."""

    if not require_run:
        return
    if not wandb_enabled:
        raise RuntimeError(
            "HOLOSOMA_REQUIRE_WANDB_RUN=1 requires logger.type='wandb'; "
            "the disabled logger cannot satisfy the scientific launch contract."
        )
    if wandb_mode != "online":
        raise RuntimeError(
            "HOLOSOMA_REQUIRE_WANDB_RUN=1 requires logger.mode='online'; "
            f"got {wandb_mode!r}."
        )


def _strict_active_wandb_run_path(
    wandb_module: Any,
    *,
    expected_entity: Any = None,
    expected_project: Any = None,
    expected_run_id: Any = None,
) -> str:
    """Return an active W&B path bound to every explicitly requested segment."""

    run = getattr(wandb_module, "run", None)
    if run is None:
        raise RuntimeError("wandb.init returned without creating an active run.")
    expected_segments = {
        "entity": expected_entity,
        "project": expected_project,
        "id": expected_run_id,
    }
    segments: list[str] = []
    for field_name in ("entity", "project", "id"):
        value = getattr(run, field_name, None)
        if not isinstance(value, str) or not _WANDB_RUN_PATH_SEGMENT_RE.fullmatch(value):
            raise RuntimeError(
                "The active W&B run has an invalid URL-path identity segment: "
                f"{field_name}={value!r}."
            )
        expected = expected_segments[field_name]
        if expected is not None:
            if not isinstance(expected, str) or not _WANDB_RUN_PATH_SEGMENT_RE.fullmatch(expected):
                raise RuntimeError(
                    "The requested W&B run has an invalid URL-path identity segment: "
                    f"{field_name}={expected!r}."
                )
            if value != expected:
                raise RuntimeError(
                    "The active W&B run identity does not match the requested launch identity: "
                    f"{field_name} requested={expected!r} active={value!r}."
                )
        segments.append(value)
    return "/".join(segments)


def _wandb_startup_error_outcome(
    wandb_module: Any,
    exc: BaseException,
) -> WandbStartupOutcome:
    """Convert a rank-zero failure to bounded data without raising pre-collective."""

    cleanup_error: BaseException | None = None
    if getattr(wandb_module, "run", None) is not None:
        try:
            _finish_wandb_run(wandb_module, exit_code=1)
        except BaseException as finish_exc:  # keep peers on the collective path
            cleanup_error = finish_exc
    message = " ".join(str(exc).split()) or type(exc).__name__
    if cleanup_error is not None:
        cleanup_message = " ".join(str(cleanup_error).split()) or type(cleanup_error).__name__
        message = f"{message}; partial-run cleanup failed: {type(cleanup_error).__name__}: {cleanup_message}"
    return {
        "ok": False,
        "run_path": None,
        "error_type": type(exc).__name__,
        "error_message": message[:4096],
        "force_fatal": not isinstance(exc, Exception),
    }


def _run_rank_zero_wandb_startup(
    wandb_module: Any,
    *,
    wandb_enabled: bool,
    wandb_mode: str | None,
    require_run: bool,
    wandb_kwargs: dict[str, Any] | None,
    publish_startup: Callable[[], None] | None,
) -> WandbStartupOutcome:
    """Run the complete rank-zero W&B startup transaction without escaping errors."""

    try:
        _validate_required_wandb_logger_mode(
            require_run=require_run,
            wandb_enabled=wandb_enabled,
            wandb_mode=wandb_mode,
        )
        if not wandb_enabled:
            return {
                "ok": True,
                "run_path": None,
                "error_type": None,
                "error_message": None,
                "force_fatal": False,
            }
        if wandb_kwargs is None or publish_startup is None:
            raise RuntimeError("Internal W&B startup contract is missing rank-zero inputs.")
        wandb_module.init(**wandb_kwargs)
        run_path = _strict_active_wandb_run_path(
            wandb_module,
            expected_entity=wandb_kwargs.get("entity"),
            expected_project=wandb_kwargs.get("project"),
            expected_run_id=wandb_kwargs.get("id"),
        )
        publish_startup()
        return {
            "ok": True,
            "run_path": run_path,
            "error_type": None,
            "error_message": None,
            "force_fatal": False,
        }
    except BaseException as exc:
        return _wandb_startup_error_outcome(wandb_module, exc)


def _validate_wandb_startup_outcome(value: Any) -> WandbStartupOutcome:
    """Validate the object-collective payload before any rank acts on it."""

    required_keys = {"ok", "run_path", "error_type", "error_message", "force_fatal"}
    if not isinstance(value, dict) or set(value) != required_keys:
        raise RuntimeError(f"Malformed rank-zero W&B startup outcome: {value!r}.")
    ok = value["ok"]
    run_path = value["run_path"]
    error_type = value["error_type"]
    error_message = value["error_message"]
    force_fatal = value["force_fatal"]
    if not isinstance(ok, bool) or not isinstance(force_fatal, bool):
        raise RuntimeError(f"Malformed rank-zero W&B startup outcome flags: {value!r}.")
    if run_path is not None and (
        not isinstance(run_path, str)
        or len(run_path.split("/")) != 3
        or any(not _WANDB_RUN_PATH_SEGMENT_RE.fullmatch(part) for part in run_path.split("/"))
    ):
        raise RuntimeError(f"Malformed rank-zero W&B run path: {run_path!r}.")
    if ok:
        if error_type is not None or error_message is not None or force_fatal:
            raise RuntimeError(f"Successful W&B startup outcome contains error state: {value!r}.")
    elif (
        run_path is not None
        or not isinstance(error_type, str)
        or not error_type
        or not isinstance(error_message, str)
        or not error_message
    ):
        raise RuntimeError(f"Failed W&B startup outcome is incomplete: {value!r}.")
    return cast("WandbStartupOutcome", value)


def _synchronize_wandb_startup_outcome(
    *,
    dist_module: Any,
    distributed_conf: MultGPUConfig | None,
    device: str,
    rank_zero_outcome: WandbStartupOutcome | None,
    local_require_run: bool,
    local_resume_must: bool,
) -> WandbStartupOutcome:
    """Broadcast startup plus policy and reject any divergent rank decision."""

    if distributed_conf is None:
        if not isinstance(local_require_run, bool) or not isinstance(local_resume_must, bool):
            raise TypeError("Local W&B startup policy flags must be booleans.")
        return _validate_wandb_startup_outcome(rank_zero_outcome)
    if not dist_module.is_initialized():
        raise RuntimeError("Distributed W&B startup synchronization requires an initialized process group.")
    rank = int(distributed_conf["global_rank"])
    if int(dist_module.get_rank()) != rank:
        raise RuntimeError("Distributed rank changed before W&B startup synchronization.")
    payload: list[dict[str, Any] | None] = [
        {
            "outcome": rank_zero_outcome,
            "require_run": local_require_run,
            "resume_must": local_resume_must,
        }
        if rank == 0
        else None
    ]
    backend = str(dist_module.get_backend()).strip().lower()

    import torch

    control_group = None
    if backend == "nccl" and _bool_env("HOLOSOMA_GLOO_SMALL_COLLECTIVES", default=False):
        dist_timeout_sec = int(os.getenv("TORCH_DIST_TIMEOUT_SEC", "600"))
        if dist_timeout_sec <= 0:
            raise ValueError(f"TORCH_DIST_TIMEOUT_SEC must be positive, got {dist_timeout_sec}.")
        control_group = dist_module.new_group(
            backend="gloo",
            timeout=timedelta(seconds=dist_timeout_sec),
        )
        broadcast_device = torch.device("cpu")
    elif backend == "nccl":
        broadcast_device = torch.device(device)
        if broadcast_device.type != "cuda":
            raise RuntimeError(
                "NCCL W&B startup broadcast requires this rank's CUDA device, "
                f"got {device!r}."
            )
    elif backend == "gloo":
        broadcast_device = torch.device("cpu")
    else:
        raise RuntimeError(
            "W&B startup outcome broadcast supports process-group backends 'nccl' and 'gloo', "
            f"got {backend!r}."
        )
    policy_mismatch = True
    try:
        if control_group is None:
            dist_module.broadcast_object_list(payload, src=0, device=broadcast_device)
        else:
            dist_module.broadcast_object_list(
                payload,
                src=0,
                group=control_group,
                device=broadcast_device,
            )
        envelope = payload[0]
        envelope_policy_valid = (
            isinstance(envelope, dict)
            and set(envelope) == {"outcome", "require_run", "resume_must"}
            and isinstance(envelope["require_run"], bool)
            and isinstance(envelope["resume_must"], bool)
        )
        local_policy_valid = isinstance(local_require_run, bool) and isinstance(local_resume_must, bool)
        local_policy_mismatch = (
            not envelope_policy_valid
            or not local_policy_valid
            or envelope["require_run"] != local_require_run
            or envelope["resume_must"] != local_resume_must
        )
        mismatch_tensor = torch.tensor(
            [1 if local_policy_mismatch else 0],
            dtype=torch.int32,
            device=broadcast_device,
        )
        all_reduce_kwargs: dict[str, Any] = {"op": dist_module.ReduceOp.MAX}
        if control_group is not None:
            all_reduce_kwargs["group"] = control_group
        dist_module.all_reduce(mismatch_tensor, **all_reduce_kwargs)
        policy_mismatch = bool(mismatch_tensor.item())
    finally:
        if control_group is not None:
            dist_module.destroy_process_group(control_group)
    if policy_mismatch:
        raise RuntimeError(
            "Distributed W&B startup policy differs across ranks or rank zero published "
            "a malformed policy envelope; refusing a divergent fatal/continue decision."
        )
    envelope = cast("dict[str, Any]", payload[0])
    return _validate_wandb_startup_outcome(envelope["outcome"])


def _resolve_wandb_startup_outcome(
    outcome: WandbStartupOutcome,
    *,
    require_run: bool,
    resume_mode: str | bool | None,
) -> str | None:
    """Return the shared run path or raise identically after synchronization."""

    outcome = _validate_wandb_startup_outcome(outcome)
    if outcome["ok"]:
        if require_run and outcome["run_path"] is None:
            raise RuntimeError(
                "HOLOSOMA_REQUIRE_WANDB_RUN=1 was set, but rank zero did not publish an active W&B run path."
            )
        return outcome["run_path"]
    fatal = require_run or _wandb_init_failure_is_fatal(resume_mode) or outcome["force_fatal"]
    if fatal:
        raise RuntimeError(
            "Rank-zero W&B startup failed; all ranks are aborting after the shared outcome collective: "
            f"{outcome['error_type']}: {outcome['error_message']}"
        )
    return None


def _per_rank_env_count(total_num_envs: int, world_size: int) -> int:
    """Return an exact per-rank environment count or fail before truncation."""

    total_num_envs = int(total_num_envs)
    world_size = int(world_size)
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if total_num_envs < world_size:
        raise ValueError(
            f"training.num_envs ({total_num_envs}) is too small for world size {world_size}. "
            "Increase num_envs or reduce the distributed world size."
        )
    if total_num_envs % world_size != 0:
        raise ValueError(
            f"training.num_envs ({total_num_envs}) must be divisible by world size {world_size}; "
            "floor division would silently run fewer environments than requested."
        )
    return total_num_envs // world_size


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}.")


def _validate_hierarchical_small_collectives_launch_contract() -> None:
    """Fail before simulator startup on an incomplete hierarchy contract."""

    validate_hierarchical_small_collectives_contract(
        {
            "HOLOSOMA_GLOO_SMALL_COLLECTIVES": _bool_env(
                "HOLOSOMA_GLOO_SMALL_COLLECTIVES"
            ),
            "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE": _bool_env(
                "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE"
            ),
            "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES": _bool_env(
                "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES"
            ),
        }
    )


def _canonicalize_fresh_curriculum_resume_env() -> bool:
    """Resolve the public launcher alias to the PPO runtime environment key."""

    canonical_name = "HOLOSOMA_ALLOW_FRESH_CURRICULUM_RESUME"
    launcher_name = "ALLOW_FRESH_CURRICULUM_RESUME"
    canonical_present = canonical_name in os.environ
    launcher_present = launcher_name in os.environ
    canonical_value = _bool_env(canonical_name) if canonical_present else None
    launcher_value = _bool_env(launcher_name) if launcher_present else None
    if canonical_value is not None and launcher_value is not None and canonical_value != launcher_value:
        raise ValueError(
            f"{canonical_name} and {launcher_name} disagree; refusing a resume whose preflight "
            "and PPO load would use different curriculum semantics."
        )
    allowed = bool(
        canonical_value
        if canonical_value is not None
        else launcher_value
        if launcher_value is not None
        else False
    )
    os.environ[canonical_name] = "1" if allowed else "0"
    return allowed


def _preflight_data_assets_before_sim() -> dict[str, Any] | None:
    """Revalidate launcher-hashed motion/object/contact bytes per node.

    The helper process owns the expensive hashing and a node-scoped locked
    cache.  Repeated calls in this worker, and calls from sibling local ranks,
    still recheck every cached inode/mtime/size identity but do not reread the
    full contact tree unless an input identity changed.
    """

    provenance = training_provenance_from_env()
    if provenance is None:
        return None

    motion_raw = os.environ.get("MOTION_DIR", "").strip()
    if not motion_raw:
        raise RuntimeError(
            "Scientific training provenance requires exported MOTION_DIR for pre-simulator revalidation."
        )
    object_spec_raw = os.environ.get("OBJECT_SPEC_PATH", "").strip()
    object_urdf_raw = os.environ.get("OBJECT_URDF", "").strip()
    object_raw = object_spec_raw or object_urdf_raw
    if not object_raw:
        raise RuntimeError(
            "Scientific training provenance requires exported OBJECT_SPEC_PATH or OBJECT_URDF "
            "for pre-simulator revalidation."
        )
    if object_spec_raw and object_urdf_raw:
        spec_path = Path(object_spec_raw).expanduser().resolve(strict=False)
        urdf_path = Path(object_urdf_raw).expanduser().resolve(strict=False)
        if spec_path != urdf_path:
            raise RuntimeError(
                "OBJECT_SPEC_PATH and OBJECT_URDF select different scientific object inputs: "
                f"{spec_path} != {urdf_path}."
            )

    contact_root: Path | None = None
    contact_enabled = (
        provenance["contact_sidecar_manifest_sha256"]
        != disabled_contact_sidecar_manifest_sha256()
    )
    if contact_enabled:
        contact_raw = os.environ.get("CONTACT_EXPORT_ROOT", "").strip()
        if not contact_raw:
            contact_raw = os.environ.get("AS_CONTACT_EXPORT_ROOT", "").strip()
        if not contact_raw:
            raise RuntimeError(
                "Training provenance declares contact sidecars, but neither CONTACT_EXPORT_ROOT "
                "nor AS_CONTACT_EXPORT_ROOT is exported for pre-simulator revalidation."
            )
        contact_root = Path(contact_raw)

    shard_manifest_raw = os.environ.get(
        "HOLOSOMA_MOTION_SHARD_MANIFEST",
        "",
    ).strip()
    if shard_manifest_raw:
        shard_manifest = Path(shard_manifest_raw)
    else:
        shard_root_raw = os.environ.get("HOLOSOMA_RANK_LOCAL_MOTION_ROOT", "").strip()
        shard_manifest = Path(shard_root_raw) / "manifest.json" if shard_root_raw else None

    script_path = (
        Path(__file__).resolve().parents[3] / "scripts" / "compute_training_provenance.py"
    )
    if not script_path.is_file():
        raise FileNotFoundError(
            "Scientific data provenance revalidation helper is missing from the source snapshot: "
            f"{script_path}"
        )
    source_root = Path(
        os.environ.get("HOLOSOMA_SOURCE_ROOT", str(script_path.parent.parent))
    ).expanduser()
    command = [
        sys.executable,
        str(script_path),
        "--revalidate-data-assets",
        "--motion-dir",
        motion_raw,
        "--object-map",
        object_raw,
        "--source-root",
        str(source_root),
    ]
    if contact_root is not None:
        command.extend(["--contact-root", str(contact_root)])
    if shard_manifest is not None:
        command.extend(["--motion-shard-manifest", str(shard_manifest)])
    cache_root_raw = os.environ.get("HOLOSOMA_DATA_PROVENANCE_CACHE_ROOT", "").strip()
    if cache_root_raw:
        command.extend(["--cache-root", cache_root_raw])

    completed = subprocess.run(
        command,
        input=canonical_training_provenance_json(provenance),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        detail = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or "unknown training data provenance revalidation failure"
        )
        raise RuntimeError(detail)
    return provenance


def _preflight_cross_rank_provenance_before_sim() -> dict[str, Any] | None:
    """Verify immutable training-input digests across torchrun ranks."""

    provenance = training_provenance_from_env()
    if provenance is None:
        return None
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    main_port = int(os.environ.get("MASTER_PORT", "29500"))
    default_provenance_port = main_port + 1 if main_port < 65535 else main_port - 1
    provenance_port = int(os.environ.get("HOLOSOMA_PROVENANCE_MASTER_PORT", str(default_provenance_port)))
    if not 1 <= provenance_port <= 65535:
        raise ValueError(f"HOLOSOMA_PROVENANCE_MASTER_PORT must be in [1, 65535], got {provenance_port}.")
    command = [
        sys.executable,
        "-m",
        "holosoma.utils.provenance_preflight",
        "--world-size",
        str(world_size),
        "--master-port",
        str(provenance_port),
    ]
    completed = subprocess.run(
        command,
        input=canonical_training_provenance_json(provenance),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        marker_prefix = "[INFO] cross_rank_training_provenance_verified "
        for output_line in completed.stdout.splitlines():
            if output_line.startswith(marker_prefix):
                # The helper process writes into a private capture pipe.  The
                # controller observes this parent process's shared stdout, so
                # preserve the one-write launch-record contract at this second
                # and operationally relevant boundary as well.
                emit_atomic_stdout_record(output_line)
            elif output_line:
                print(output_line, flush=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown provenance preflight failure"
        raise RuntimeError(detail)
    return provenance


def _training_consumes_teacher(tyro_config: ExperimentConfig) -> bool:
    """Mirror the PPO setup condition that actually constructs a teacher."""

    algo = getattr(tyro_config, "algo", None)
    algo_config = getattr(algo, "config", None)
    distill = getattr(algo_config, "distill", None)
    if distill is None:
        return False
    mode = str(getattr(distill, "mode", "mse")).strip().lower()
    if mode == "dagger":
        bc_loss_coef = (
            float(distill.bc_loss_coef)
            if distill.bc_loss_coef is not None
            else float(distill.loss_coef)
        )
        ppo_start_epoch = int(getattr(distill, "ppo_start_epoch", -1))
        dagger_end_epoch = int(getattr(distill, "dagger_end_epoch", -1))
        schedule_enabled = (
            ppo_start_epoch >= 0 and dagger_end_epoch > ppo_start_epoch
        )
        return (
            bc_loss_coef > 0.0
            or int(getattr(distill, "switch_to_rl_after", -1)) > 0
            or schedule_enabled
        )
    return bool(distill.enabled)


def _preflight_checkpoint_lineage_before_sim(tyro_config: ExperimentConfig) -> None:
    """Bind provenance checkpoint lineage to the effective training CLI mode."""

    provenance = training_provenance_from_env()
    if _training_consumes_teacher(tyro_config):
        legacy_unverified_teacher_load = allow_legacy_unverified_teacher_load()
        if provenance is None:
            if not legacy_unverified_teacher_load:
                raise ValueError(
                    "Scientific teacher loading requires finalized current training provenance "
                    "before simulator startup. Set "
                    f"{ALLOW_LEGACY_UNVERIFIED_TEACHER_LOAD_ENV}=1 only for an explicitly "
                    "non-scientific legacy teacher load."
                )
            print(
                "[WARN] legacy_unverified_teacher_load_allowed "
                f"override={ALLOW_LEGACY_UNVERIFIED_TEACHER_LOAD_ENV}=1: teacher checkpoint "
                "identity is not authenticated by current training provenance.",
                flush=True,
            )
        else:
            if provenance.get("teacher_enabled") is not True:
                raise ValueError(
                    "Training consumes a teacher but current training provenance disables it."
                )
            if bool(getattr(tyro_config.algo.config.distill, "use_multi_teacher", False)):
                raise ValueError(
                    "Scientific multi-teacher loading requires one authenticated digest per teacher; "
                    "the current provenance schema contains only teacher_sha256."
                )
    elif provenance is not None and provenance.get("teacher_enabled") is True:
        raise ValueError(
            "Current training provenance enables a teacher, but the effective training objective "
            "does not consume one."
        )
    if provenance is None:
        return
    configured = {
        "policy_init": tyro_config.training.policy_init_checkpoint is not None,
        "training_resume": tyro_config.training.checkpoint is not None,
    }
    if all(configured.values()):
        raise ValueError("--training.checkpoint and --training.policy-init-checkpoint are mutually exclusive.")
    for role, configured_enabled in configured.items():
        provenance_enabled = checkpoint_lineage_enabled(provenance, role)
        if provenance_enabled != configured_enabled:
            cli_option = (
                "--training.policy-init-checkpoint"
                if role == "policy_init"
                else "--training.checkpoint"
            )
            raise ValueError(
                f"Training provenance {role}_enabled={provenance_enabled} does not match "
                f"{cli_option} presence={configured_enabled}."
            )


def _preflight_training_resume_before_sim(tyro_config: ExperimentConfig) -> ExperimentConfig:
    """Validate a curriculum-correct training-resume contract before simulation."""

    checkpoint = tyro_config.training.checkpoint
    if checkpoint is None:
        return tyro_config
    checkpoint_path: Path
    if str(checkpoint).startswith("wandb://"):
        rank = int(os.environ.get("RANK", "0"))
        cache_root = Path(
            os.environ.get(
                "HOLOSOMA_RESUME_PREFLIGHT_CACHE",
                str(Path.home() / ".cache" / "holosoma" / "resume_preflight"),
            )
        )
        checkpoint_path = load_checkpoint(str(checkpoint), str(cache_root / f"rank_{rank}"))
    else:
        checkpoint_path = Path(str(checkpoint)).expanduser()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Training-resume checkpoint is not a readable local file: {checkpoint_path}")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    per_rank_num_envs = _per_rank_env_count(tyro_config.training.num_envs, world_size)
    effective_config = apply_observation_overrides(tyro_config)
    effective_config = apply_perception_overrides(effective_config)
    effective_config = dataclasses.replace(
        effective_config,
        training=dataclasses.replace(
            effective_config.training,
            num_envs=per_rank_num_envs,
            checkpoint=str(checkpoint_path),
        ),
    )

    command = [
        sys.executable,
        "-m",
        "holosoma.utils.resume_preflight",
        "--checkpoint",
        str(checkpoint_path),
        "--world-size",
        str(world_size),
    ]
    if _canonicalize_fresh_curriculum_resume_env():
        command.append("--allow-fresh-curriculum")
    current_provenance = training_provenance_from_env()
    if current_provenance is not None:
        command.extend(
            ["--current-provenance-json", canonical_training_provenance_json(current_provenance)]
        )
    completed = subprocess.run(
        command,
        input=json.dumps(effective_config.to_serializable_dict()),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown resume preflight failure"
        raise RuntimeError(detail)

    return dataclasses.replace(
        tyro_config,
        training=dataclasses.replace(tyro_config.training, checkpoint=str(checkpoint_path)),
    )


def _preflight_policy_init_before_sim(tyro_config: ExperimentConfig) -> ExperimentConfig:
    """Validate the actor semantic contract of a policy initializer before simulation."""

    checkpoint = tyro_config.training.policy_init_checkpoint
    required_terminal_target = required_policy_init_terminal_target_from_env()
    if checkpoint is None:
        if required_terminal_target is not None:
            raise ValueError(
                "A required policy-init terminal target was configured, but "
                "--training.policy-init-checkpoint is empty."
            )
        return tyro_config
    if tyro_config.training.checkpoint is not None:
        raise ValueError("--training.checkpoint and --training.policy-init-checkpoint are mutually exclusive.")

    checkpoint_path: Path
    if str(checkpoint).startswith("wandb://"):
        rank = int(os.environ.get("RANK", "0"))
        cache_root = Path(
            os.environ.get(
                "HOLOSOMA_POLICY_INIT_PREFLIGHT_CACHE",
                str(Path.home() / ".cache" / "holosoma" / "policy_init_preflight"),
            )
        )
        checkpoint_path = load_checkpoint(str(checkpoint), str(cache_root / f"rank_{rank}"))
    else:
        checkpoint_path = Path(str(checkpoint)).expanduser()
    # Preserve the lexical final component so the subprocess' O_NOFOLLOW open
    # can reject a symlink rather than resolving it before authentication.
    checkpoint_path = Path(os.path.abspath(os.fspath(checkpoint_path)))
    try:
        checkpoint_stat = os.lstat(checkpoint_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Policy-init checkpoint is not a readable local file: {checkpoint_path}"
        ) from exc
    if not stat.S_ISREG(checkpoint_stat.st_mode) or stat.S_ISLNK(checkpoint_stat.st_mode):
        raise ValueError(
            f"Policy-init checkpoint must be a non-symlink regular file: {checkpoint_path}"
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    per_rank_num_envs = _per_rank_env_count(
        tyro_config.training.num_envs,
        world_size,
    )
    effective_config = apply_observation_overrides(tyro_config)
    effective_config = apply_perception_overrides(effective_config)
    effective_config = dataclasses.replace(
        effective_config,
        training=dataclasses.replace(
            effective_config.training,
            num_envs=per_rank_num_envs,
            policy_init_checkpoint=str(checkpoint_path),
        ),
    )

    command = [
        sys.executable,
        "-m",
        "holosoma.utils.policy_init_preflight",
        "--checkpoint",
        str(checkpoint_path),
    ]
    current_provenance = training_provenance_from_env()
    if current_provenance is not None:
        command.extend(
            ["--current-provenance-json", canonical_training_provenance_json(current_provenance)]
        )
    if required_terminal_target is not None:
        command.extend(
            ["--require-terminal-target", str(required_terminal_target)]
        )
    completed = subprocess.run(
        command,
        input=json.dumps(effective_config.to_serializable_dict()),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown policy-init preflight failure"
        raise RuntimeError(detail)

    return dataclasses.replace(
        tyro_config,
        training=dataclasses.replace(
            tyro_config.training,
            policy_init_checkpoint=str(checkpoint_path),
        ),
    )


def configure_multi_gpu() -> MultGPUConfig | None:
    """Configure multi-gpu training and return configuration dictionary, or `None` if single-GPU training."""
    import torch

    gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
    is_distributed = gpu_world_size > 1

    if not is_distributed:
        return None

    gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
    gpu_global_rank = int(os.getenv("RANK", "0"))

    gpu_local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", str(gpu_world_size)))
    if gpu_local_rank >= gpu_local_world_size:
        raise ValueError(
            f"Local rank '{gpu_local_rank}' is greater than or equal to local world size '{gpu_local_world_size}'."
        )

    if gpu_global_rank >= gpu_world_size:
        raise ValueError(f"Global rank '{gpu_global_rank}' is greater than or equal to world size '{gpu_world_size}'.")

    dist_backend = os.getenv("TORCH_DIST_BACKEND", "nccl").strip().lower()
    if dist_backend not in ("nccl", "gloo"):
        raise ValueError(f"Unsupported TORCH_DIST_BACKEND={dist_backend!r}; expected 'nccl' or 'gloo'.")

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed CUDA training requested but CUDA is not available.")

    visible_gpu_count = torch.cuda.device_count()
    if gpu_local_rank >= visible_gpu_count:
        raise ValueError(
            f"Local rank '{gpu_local_rank}' is out of range for {visible_gpu_count} visible CUDA devices. "
            "Check CUDA_VISIBLE_DEVICES and --nproc_per_node."
        )

    torch.cuda.set_device(gpu_local_rank)
    dist_timeout_sec = int(os.getenv("TORCH_DIST_TIMEOUT_SEC", "600"))
    if dist_timeout_sec <= 0:
        raise ValueError(f"TORCH_DIST_TIMEOUT_SEC must be positive, got {dist_timeout_sec}.")
    import inspect

    init_kwargs: dict[str, Any] = {
        "backend": dist_backend,
        "rank": gpu_global_rank,
        "world_size": gpu_world_size,
        "timeout": timedelta(seconds=dist_timeout_sec),
    }
    if dist_backend == "nccl" and "device_id" in inspect.signature(torch.distributed.init_process_group).parameters:
        init_kwargs["device_id"] = torch.device(f"cuda:{gpu_local_rank}")
    torch.distributed.init_process_group(**init_kwargs)

    multi_gpu_config: MultGPUConfig = {
        "global_rank": gpu_global_rank,
        "local_rank": gpu_local_rank,
        "world_size": gpu_world_size,
    }
    logger.info(f"Running with multi-GPU parameters: {multi_gpu_config}")
    logger.info(
        "Distributed CUDA setup: global_rank={} local_rank={} local_world_size={} "
        "visible_gpu_count={} current_device={} cuda_visible_devices={}",
        gpu_global_rank,
        gpu_local_rank,
        gpu_local_world_size,
        visible_gpu_count,
        torch.cuda.current_device(),
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )
    logger.info("Distributed process group backend: {}", dist_backend)

    return multi_gpu_config


def get_device(config, distributed_conf: MultGPUConfig | None) -> str:
    import torch

    is_config_device_specified = hasattr(config, "device") and config.device is not None
    is_multi_gpu = distributed_conf is not None

    if is_config_device_specified:
        if is_multi_gpu and config.device != cast("dict", distributed_conf)["local_rank"]:
            raise ValueError(
                f"Device specified in config ({config.device}) \
                              does not match expected local rank {cast('dict', distributed_conf)['local_rank']}"
            )
        device = config.device
    elif is_multi_gpu:
        device = f"cuda:{cast('dict', distributed_conf)['local_rank']}"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    return device


def configure_logging(distributed_conf: MultGPUConfig | None = None, log_dir: Path | None = None):
    # Configure logging.
    from holosoma.utils.logging import LoguruLoggingBridge

    logger.remove()
    is_main_process = distributed_conf is None or distributed_conf["global_rank"] == 0

    # Install the console sink before touching the filesystem.  If a bad log
    # root cannot be created, the outer training exception handler must still
    # have a live sink and emit the actual PermissionError on every rank.
    if is_main_process:
        console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    else:
        console_log_level = "ERROR"
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    # logging to file (from all ranks)
    if log_dir is not None:
        fname = f"train_rank_{distributed_conf['global_rank']:02d}.log" if distributed_conf is not None else "train.log"
        log_path = log_dir / fname
        logger.add(str(log_path), level="DEBUG")
    logging.basicConfig(level=logging.DEBUG if is_main_process else logging.ERROR)
    logging.getLogger().addHandler(LoguruLoggingBridge())


def _zoom_out_video_config(config: VideoConfig, zoom: float) -> VideoConfig:
    if zoom <= 1.0:
        return config
    camera = config.camera
    if isinstance(camera, SphericalCameraConfig):
        camera = dataclasses.replace(camera, distance=float(camera.distance) * zoom)
    elif isinstance(camera, CartesianCameraConfig):
        camera = dataclasses.replace(camera, offset=[float(v) * zoom for v in camera.offset])
    elif isinstance(camera, FixedCameraConfig):
        position = np.array(camera.position, dtype=np.float32)
        target = np.array(camera.target, dtype=np.float32)
        position = target + (position - target) * zoom
        camera = dataclasses.replace(camera, position=position.tolist())
    return dataclasses.replace(config, camera=camera)


def _apply_debug_camera_tilt(env: Any, target_pitch_deg: float = -20.0) -> bool:
    import torch

    perception = getattr(env, "perception_manager", None)
    if perception is None or perception._camera_ray_dirs_base is None:
        return False

    try:
        record_env_id = int(getattr(env.simulator.video_config, "record_env_id", 0))
        idx = torch.tensor([record_env_id], device=perception.device)
        _, body_quat = perception._get_camera_body_pose(idx)
        ray_dirs_base = perception._camera_ray_dirs_base
        center_dir = ray_dirs_base.view(perception._camera_height, perception._camera_width, 3)[
            perception._camera_height // 2, perception._camera_width // 2
        ]
        forward_world = quat_apply(body_quat, center_dir.unsqueeze(0), w_last=True).squeeze(0)
        horiz = torch.sqrt(forward_world[0] ** 2 + forward_world[1] ** 2).clamp(min=1.0e-6)
        current_pitch = torch.atan2(forward_world[2], horiz)
        target_pitch = torch.deg2rad(torch.tensor(float(target_pitch_deg), device=perception.device))
        delta = target_pitch - current_pitch
        if torch.abs(delta).item() < 1.0e-3:
            return False
        delta_quat = quat_from_euler_xyz(torch.tensor(0.0, device=perception.device), delta, torch.tensor(0.0, device=perception.device))
        delta_quat = delta_quat.unsqueeze(0).expand(ray_dirs_base.shape[0], -1)
        perception._camera_ray_dirs_base = quat_rotate_inverse(delta_quat, ray_dirs_base, w_last=True)
        logger.info(
            f"Debug depth: auto-tilting camera rays by {float(torch.rad2deg(delta)):.2f} deg "
            f"(target pitch {target_pitch_deg:.1f} deg)."
        )
        return True
    except Exception as exc:
        logger.warning(f"Debug depth: failed to auto-tilt camera rays: {exc}")
        return False


def _run_debug_depth_video(env: Any, *, wandb_logging: bool) -> None:
    if not hasattr(env, "step_visualize_motion"):
        raise RuntimeError("Debug video requires an environment with step_visualize_motion().")
    if env.perception_manager is None or not env.perception_manager.enabled:
        raise RuntimeError("Debug video requires perception to be enabled.")
    if env.perception_manager.cfg.output_mode != "camera_depth":
        raise RuntimeError("Debug video requires perception output_mode=camera_depth.")

    video_recorder = env.simulator.video_recorder if hasattr(env, "simulator") else None
    debug_zoom = 2.5
    if video_recorder is not None and video_recorder.enabled:
        zoomed_config = _zoom_out_video_config(video_recorder.config, debug_zoom)
        video_recorder.config = zoomed_config
        env.simulator.video_config = zoomed_config
        video_recorder.start_recording(episode_id=0)
    else:
        logger.warning("Debug video: simulator video recorder not enabled; only depth video will be logged.")

    env.reset_all()
    _apply_debug_camera_tilt(env, target_pitch_deg=-20.0)

    record_env_id = int(getattr(env.simulator.video_config, "record_env_id", 0))
    frames: list[Any] = []
    done = False
    max_distance = float(env.perception_manager.cfg.max_distance)

    while not done:
        if hasattr(env.simulator, "sim"):
            env.simulator.sim.step()
        done = bool(env.step_visualize_motion(None))
        env.perception_manager.update()
        if video_recorder is not None and video_recorder.enabled:
            env.simulator.capture_video_frame(record_env_id)
        depth = env.perception_manager.get_camera_depth_map()[record_env_id].detach().cpu().numpy()
        if not frames:
            if np.allclose(depth, max_distance):
                logger.warning("Debug depth video: all rays hit max_distance; check camera pose/pitch/terrain.")
        frames.append(env._depth_to_rgb(depth))

    if not frames:
        logger.warning("Debug depth video: no frames captured.")
        if video_recorder is not None and video_recorder.enabled:
            video_recorder.stop_recording()
        return

    from holosoma.utils.video_utils import create_video  # noqa: PLC0415

    sim_config = env.simulator.simulator_config.sim
    control_frequency = sim_config.fps / sim_config.control_decimation
    display_fps = control_frequency * env.simulator.video_config.playback_rate
    save_dir = (
        Path(env.simulator.video_config.save_dir)
        if env.simulator.video_config.save_dir is not None
        else Path("/data/logs_new/videos")
    )
    create_video(
        video_frames=np.stack(frames, axis=0).astype(np.uint8),
        fps=display_fps,
        save_dir=save_dir,
        output_format=env.simulator.video_config.output_format,
        wandb_logging=wandb_logging,
        episode_id=0,
        wandb_key="Depth rollout (debug)",
    )
    if video_recorder is not None and video_recorder.enabled:
        video_recorder.stop_recording()


def _run_debug_motion_preview(env: Any, *, max_steps: int | None = None) -> None:
    if not hasattr(env, "step_visualize_motion"):
        raise RuntimeError("Debug motion preview requires an environment with step_visualize_motion().")
    env.reset_all()
    done = False
    step = 0
    while not done:
        done = bool(env.step_visualize_motion(None))
        step += 1
        if max_steps is not None and step >= max_steps:
            logger.info("Debug motion preview reached max_steps={} and will exit.", max_steps)
            break


def _run_debug_mode_by_perception(
    env: Any,
    *,
    wandb_logging: bool,
    max_steps: int | None = None,
) -> None:
    perception_mgr = getattr(env, "perception_manager", None)
    perception_enabled = bool(getattr(perception_mgr, "enabled", False))
    output_mode = str(getattr(getattr(perception_mgr, "cfg", None), "output_mode", "")).strip().lower()
    if perception_enabled and output_mode == "camera_depth":
        logger.info("Debug mode: perception output_mode=camera_depth, running depth debug rollout.")
        _run_debug_depth_video(env, wandb_logging=wandb_logging)
        return

    logger.info(
        "Debug mode: perception output_mode='{}' (enabled={}), running motion/viser preview.",
        output_mode or "none",
        perception_enabled,
    )
    _run_debug_motion_preview(env, max_steps=max_steps)


def _configure_defm_materialization_mode(config: ExperimentConfig) -> str:
    """Bind lazy DeFM construction to the checkpoint operation that follows."""

    has_resume = config.training.checkpoint is not None
    has_policy_init = config.training.policy_init_checkpoint is not None
    if has_resume and has_policy_init:
        raise ValueError(
            "--training.checkpoint and --training.policy-init-checkpoint are mutually exclusive."
        )
    mode = "full_resume" if has_resume else "policy_init" if has_policy_init else "fresh"
    # The serialized training config is authoritative. Do not permit an
    # ambient shell variable to silently choose a different initialization.
    return set_defm_materialization_mode(mode)


def train(tyro_config: ExperimentConfig, training_context: TrainingContext | None = None) -> None:
    """Train an agent with optional context for sim app management.

    Parameters
    ----------
    training_context : Optional[TrainingContext]
        Optional training context with pre-initialized sim app.
        If None, creates and manages sim app automatically.
    """

    # Direct API callers must hash and initialize exactly the same effective
    # config as the CLI entrypoint.  These rewrites are pure/idempotent and
    # must happen before every provenance or simulator boundary.
    tyro_config = _effective_runtime_config(tyro_config)
    _validate_hierarchical_small_collectives_launch_contract()
    if training_context is not None:
        if training_context.simulation_app is None:
            raise RuntimeError("TrainingContext must be entered before it is passed to train().")
        if not training_context._policy_init_preflight_complete:
            raise RuntimeError(
                "TrainingContext did not complete policy-init preflight before simulator startup."
            )
        context_config = _effective_runtime_config(training_context.config)
        if context_config != tyro_config:
            raise ValueError(
                "train() config differs from the effective config used to initialize TrainingContext."
            )
        tyro_config = context_config

    # Validate the complete distributed seed range before a direct API caller
    # can start Isaac or mutate any process RNG. ``main`` and TrainingContext
    # perform the same fail-closed check at their own simulator boundaries.
    _current_rank_training_seed(tyro_config.training.seed)
    _configure_defm_materialization_mode(tyro_config)

    # Narrow the hash-to-open window after checkpoint/cross-rank preflights and
    # before Isaac/Gym modules can open any mutable asset path.
    finalized_provenance = finalize_runtime_asset_provenance(tyro_config)
    _validate_prestarted_runtime_provenance(finalized_provenance)
    _preflight_checkpoint_lineage_before_sim(tyro_config)
    _preflight_data_assets_before_sim()

    if training_context is not None:
        # Use the context's pre-initialized sim app
        simulation_app = training_context.simulation_app
        auto_close = False  # Context will handle closing
    else:
        # ``train(config)`` is also a public simulator-starting entrypoint.
        # ``main`` already performs this preflight, but it installs the local
        # verified checkpoint path into the returned immutable config, so this
        # second fail-closed boundary cannot download a W&B artifact twice.
        tyro_config = _preflight_policy_init_before_sim(tyro_config)
        # Default behavior - create and manage sim app ourselves
        simulation_app = init_sim_imports(tyro_config)
        auto_close = True

    # These services are initialized inside the guarded block, but failures
    # can occur at any point after either one becomes active.  Keep explicit
    # outer state so ``finally`` can perform idempotent teardown on both the
    # success and exception paths.
    dist = None
    wandb = None
    is_distributed = False
    is_main_process = True
    wandb_enabled = False

    try:
        # have to import torch after isaacgym
        import torch  # noqa: F401
        import torch.distributed as dist
        import wandb

        from holosoma.agents.base_algo.base_algo import BaseAlgo
        from holosoma.utils.common import seeding

        # unresolved_conf = dataclasses.asdict(tyro_config)
        # import ipdb; ipdb.set_trace()

        # Initialize process group
        distributed_conf: MultGPUConfig | None = configure_multi_gpu()
        device: str = get_device(tyro_config, distributed_conf)
        is_distributed = distributed_conf is not None
        is_main_process = distributed_conf is None or distributed_conf["global_rank"] == 0

        # Configure logger
        logger_cfg = tyro_config.logger
        wandb_enabled = logger_cfg.type == "wandb"

        # Rank zero owns the run identity.  All ranks and the environment/simulator
        # reuse the exact timestamp and path published here.
        timestamp, experiment_dir = _synchronize_experiment_identity(
            dist_module=dist,
            distributed_conf=distributed_conf,
            device=device,
            logger_config=logger_cfg,
            training_config=tyro_config.training,
            task_name="locomotion",
        )

        # Configure logging with experiment directory
        configure_logging(distributed_conf=distributed_conf, log_dir=experiment_dir)

        wandb_run_path: str | None = None

        # Distribute environments across GPUs for proper multi-GPU training
        requested_total_num_envs = int(tyro_config.training.num_envs)
        world_size = int(distributed_conf["world_size"]) if distributed_conf is not None else 1
        if distributed_conf is not None:
            original_num_envs = requested_total_num_envs
            num_envs = _per_rank_env_count(original_num_envs, distributed_conf["world_size"])
            tyro_config = dataclasses.replace(
                tyro_config, training=dataclasses.replace(tyro_config.training, num_envs=num_envs)
            )
            logger.info(
                f"Distributed training: GPU {distributed_conf['global_rank']} will run {tyro_config.training.num_envs} "
                f"environments (total across all GPUs: {original_num_envs})"
            )
        effective_total_num_envs = int(tyro_config.training.num_envs) * world_size

        if tyro_config.training.debug and not tyro_config.training.headless:
            tyro_config = dataclasses.replace(
                tyro_config, training=dataclasses.replace(tyro_config.training, headless=True)
            )
            logger.info("Debug mode: forcing headless=True to avoid viewer-only issues.")

        experiment_save_dir = experiment_dir
        experiment_save_dir.mkdir(exist_ok=True, parents=True)

        config_path: Path | None = None
        runtime_asset_manifest_path: Path | None = None

        # Rank zero owns W&B, but no rank acts on its startup result until the
        # complete init/metadata/file-registration outcome reaches every peer.
        wandb_required = _bool_env("HOLOSOMA_REQUIRE_WANDB_RUN", default=False)
        wandb_mode = getattr(logger_cfg, "mode", None) if wandb_enabled else None
        wandb_resume_mode = getattr(logger_cfg, "resume", None) if wandb_enabled else None
        rank_zero_wandb_outcome: WandbStartupOutcome | None = None
        if is_main_process:
            local_config_persisted = False
            try:
                logger.info(f"Saving config file to {experiment_save_dir}")
                config_path = experiment_save_dir / CONFIG_NAME
                tyro_config.save_config(str(config_path))
                finalized_provenance = training_provenance_from_env()
                if finalized_provenance is not None:
                    runtime_asset_manifest_path = persist_runtime_asset_manifest(
                        experiment_save_dir / RUNTIME_ASSET_MANIFEST_FILENAME,
                        finalized_provenance,
                    )
                    logger.info(
                        "Persisted canonical runtime asset manifest to {}",
                        runtime_asset_manifest_path,
                    )
                local_config_persisted = True
                wandb_kwargs: dict[str, Any] | None = None
                publish_wandb_startup: Callable[[], None] | None = None
                if wandb_enabled:
                    from holosoma.config_types.logger import WandbLoggerConfig

                    if not isinstance(logger_cfg, WandbLoggerConfig):
                        raise TypeError("Logger config must be WandbLoggerConfig when type is wandb.")
                    wandb_cfg = logger_cfg
                    default_project = tyro_config.training.project or wandb_cfg.project or "default_project"
                    default_run_name = (
                        f"{timestamp}_{tyro_config.training.name or 'run'}_"
                        f"{wandb_cfg.group or 'default'}_{tyro_config.robot.asset.robot_type}"
                    )
                    wandb_dir = Path(wandb_cfg.dir or (experiment_dir / ".wandb"))
                    wandb_dir.mkdir(exist_ok=True, parents=True)
                    logger.info(f"Saving wandb logs to {wandb_dir}")

                    wandb_kwargs = {
                        "project": wandb_cfg.project or default_project,
                        "name": wandb_cfg.name or default_run_name,
                        "config": dataclasses.asdict(tyro_config),
                        "dir": str(wandb_dir),
                        "mode": wandb_cfg.mode,
                    }
                    if wandb_cfg.entity:
                        wandb_kwargs["entity"] = wandb_cfg.entity
                    if wandb_cfg.group:
                        wandb_kwargs["group"] = wandb_cfg.group
                    if wandb_cfg.id:
                        wandb_kwargs["id"] = wandb_cfg.id
                    if wandb_cfg.tags:
                        wandb_kwargs["tags"] = list(wandb_cfg.tags)
                    if wandb_cfg.resume is not None:
                        wandb_kwargs["resume"] = wandb_cfg.resume
                    wandb_kwargs["settings"] = wandb.Settings(
                        init_timeout=float(os.environ.get("WANDB_INIT_TIMEOUT", "60"))
                    )

                    def publish_wandb_startup() -> None:
                        env_count_metadata = _collect_env_count_wandb_metadata(
                            requested_total_num_envs=requested_total_num_envs,
                            effective_total_num_envs=effective_total_num_envs,
                            per_rank_num_envs=int(tyro_config.training.num_envs),
                            world_size=world_size,
                        )
                        _publish_wandb_startup_metadata(wandb, config_metadata=env_count_metadata)
                        logger.info("Logged environment-count metadata to W&B: {}", env_count_metadata)
                        object_bank_metadata = _collect_object_bank_wandb_metadata()
                        if object_bank_metadata:
                            _publish_wandb_startup_metadata(wandb, config_metadata=object_bank_metadata)
                            logger.info("Logged object-bank metadata to W&B: {}", object_bank_metadata)
                        provenance_metadata = _collect_training_provenance_wandb_metadata()
                        if provenance_metadata:
                            _publish_wandb_startup_metadata(wandb, config_metadata=provenance_metadata)
                            logger.info("Logged immutable training-input provenance to W&B: {}", provenance_metadata)
                        reward_config_metadata, reward_summary_metadata = collect_reward_wandb_metadata(
                            tyro_config.reward
                        )
                        if reward_config_metadata:
                            _publish_wandb_startup_metadata(
                                wandb,
                                config_metadata=reward_config_metadata,
                                summary_metadata=reward_summary_metadata,
                            )
                            logger.info("Logged grouped reward metadata to W&B.")
                        if config_path is not None:
                            wandb.save(str(config_path), base_path=experiment_save_dir)
                        if runtime_asset_manifest_path is not None:
                            wandb.save(str(runtime_asset_manifest_path), base_path=experiment_save_dir)

                rank_zero_wandb_outcome = _run_rank_zero_wandb_startup(
                    wandb,
                    wandb_enabled=wandb_enabled,
                    wandb_mode=wandb_mode,
                    require_run=wandb_required,
                    wandb_kwargs=wandb_kwargs,
                    publish_startup=publish_wandb_startup,
                )
            except BaseException as wandb_setup_exc:
                # Directory/config/Settings failures are rank-zero-only too;
                # serialize them instead of stranding peers at the next sync.
                rank_zero_wandb_outcome = _wandb_startup_error_outcome(wandb, wandb_setup_exc)
                if not local_config_persisted:
                    # Canonical local config/provenance persistence is required
                    # even for a direct optional-W&B run.
                    rank_zero_wandb_outcome["force_fatal"] = True

        shared_wandb_outcome = _synchronize_wandb_startup_outcome(
            dist_module=dist,
            distributed_conf=distributed_conf,
            device=device,
            rank_zero_outcome=rank_zero_wandb_outcome,
            local_require_run=wandb_required,
            local_resume_must=_wandb_init_failure_is_fatal(wandb_resume_mode),
        )
        wandb_run_path = _resolve_wandb_startup_outcome(
            shared_wandb_outcome,
            require_run=wandb_required,
            resume_mode=wandb_resume_mode,
        )
        if not shared_wandb_outcome["ok"] and is_main_process:
            logger.error(
                "W&B startup failed on rank zero ({}: {}); continuing without an active run because "
                "neither HOLOSOMA_REQUIRE_WANDB_RUN nor resume='must' is active.",
                shared_wandb_outcome["error_type"],
                shared_wandb_outcome["error_message"],
            )
        elif wandb_run_path is not None and is_main_process:
            logger.info("W&B startup outcome synchronized: run_path={}", wandb_run_path)

        _distributed_barrier(dist, distributed_conf)

        env_target = tyro_config.env_class

        # W&B/config setup can take long enough for mutable node-local assets to
        # drift.  Re-verify once more immediately before environment/simulator
        # instantiation.  The object loader additionally compares its frozen
        # normalized environment semantics with this embedded manifest.
        finalized_provenance = finalize_runtime_asset_provenance(tyro_config)
        _validate_prestarted_runtime_provenance(finalized_provenance)
        _preflight_data_assets_before_sim()

        # Establish the experiment RNG stream only after rank-0-only logger/W&B
        # setup and the matching all-rank barrier.  Logger initialization is
        # allowed to use Python/NumPy randomness internally; seeding before it
        # made rank 0's simulator stream depend on whether/how W&B initialized.
        # Keep this immediately before environment and algorithm construction so
        # a fresh run is a function of the configured seed and global rank, not
        # of auxiliary logging behavior.  A full checkpoint resume restores its
        # saved rank-local RNG state at the end of ``algo.load``.
        seed = _rank_training_seed(
            tyro_config.training.seed,
            world_size=world_size,
            global_rank=(
                int(distributed_conf["global_rank"])
                if distributed_conf is not None
                else 0
            ),
        )
        seeding(seed, torch_deterministic=tyro_config.training.torch_deterministic)

        tyro_env_config = get_tyro_env_config(tyro_config)
        env = get_class(env_target)(tyro_env_config, device=device)

        # For manager system, pre-process config AFTER env creation
        # (need managers to compute dims)
        observation_manager = getattr(env, "observation_manager", None)
        if observation_manager is None:
            raise RuntimeError(
                f"Manager environment {env_target} is missing observation_manager attribute. "
                "This should not happen if the environment is properly configured."
            )

        if tyro_config.training.debug:
            if is_main_process:
                max_debug_steps = tyro_config.training.max_eval_steps
                _run_debug_mode_by_perception(
                    env,
                    wandb_logging=wandb_enabled,
                    max_steps=max_debug_steps,
                )
            if is_main_process and wandb_enabled:
                logger.info("Shutting down wandb...")
                _finish_wandb_run(wandb, exit_code=0)
            if is_distributed:
                _distributed_barrier(dist, distributed_conf)
                logger.info("Shutting down distributed processes...")
                dist.destroy_process_group()
            return

        algo_class = get_class(tyro_config.algo._target_)
        algo: BaseAlgo = algo_class(
            device=device,
            env=env,
            config=tyro_config.algo.config,
            log_dir=experiment_save_dir,
            multi_gpu_cfg=distributed_conf,
        )
        algo.setup()

        algo.attach_checkpoint_metadata(tyro_config, wandb_run_path)
        if (
            tyro_config.training.checkpoint is not None
            and tyro_config.training.policy_init_checkpoint is not None
        ):
            raise ValueError("--training.checkpoint and --training.policy-init-checkpoint are mutually exclusive.")
        if tyro_config.training.checkpoint is not None:
            loaded_checkpoint = load_checkpoint(tyro_config.training.checkpoint, str(experiment_save_dir))
            tyro_config = dataclasses.replace(
                tyro_config, training=dataclasses.replace(tyro_config.training, checkpoint=str(loaded_checkpoint))
            )
            algo.load(loaded_checkpoint)
        elif tyro_config.training.policy_init_checkpoint is not None:
            loaded_checkpoint = load_checkpoint(tyro_config.training.policy_init_checkpoint, str(experiment_save_dir))
            tyro_config = dataclasses.replace(
                tyro_config,
                training=dataclasses.replace(
                    tyro_config.training,
                    policy_init_checkpoint=str(loaded_checkpoint),
                ),
            )
            algo.load_policy_init(loaded_checkpoint)

        # This is the controller's terminal startup handshake boundary.  It is
        # deliberately after real simulator/environment construction,
        # algo.setup (including model synchronization), and the authoritative
        # checkpoint load.  The helper performs one final main-process-group
        # barrier before every worker emits its unique launch-bound marker.
        _emit_batch_worker_preflight_ready(
            dist_module=dist,
            distributed_conf=distributed_conf,
        )

        # handle saving config
        algo.learn()

        # teardown wandb before SimApp closes ungracefully (IsaacLab)
        if is_main_process and wandb_enabled:
            logger.info("Shutting down wandb...")
            _finish_wandb_run(wandb, exit_code=0)

        # shutdown dist before SimApp closes ungracefully (IsaacLab)
        if is_distributed:
            logger.info("Shutting down distributed processes...")
            dist.destroy_process_group()
    except Exception as e:
        tb_str = traceback.format_exc()
        # Failures before ``configure_logging`` previously disappeared for
        # non-zero ranks, leaving torch-elastic with only ``exitcode=1``.  Keep
        # the structured logger, but always emit a flushed stderr traceback so
        # the first failed rank is diagnosable from the node log.
        print(f"Exception occurred during training: {e}\n{tb_str}", file=sys.stderr, flush=True)
        logger.error(f"Exception occurred during training: {e}\n{tb_str}")
        sys.exit(1)  # manually set exit code, not possible via isaacsim app.close()
    finally:
        if is_main_process and wandb_enabled and wandb is not None and getattr(wandb, "run", None) is not None:
            try:
                logger.info("Shutting down wandb from final cleanup...")
                _finish_wandb_run(
                    wandb,
                    exit_code=1 if sys.exc_info()[0] is not None else 0,
                )
            except Exception:
                logger.exception("W&B final cleanup failed.")
        if dist is not None and is_distributed:
            try:
                if dist.is_available() and dist.is_initialized():
                    logger.info("Shutting down distributed processes from final cleanup...")
                    dist.destroy_process_group()
            except Exception:
                logger.exception("Distributed final cleanup failed.")
        if auto_close:
            close_simulation_app(simulation_app)

    logger.info("Training shutdown complete.")


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    # Runtime asset provenance must describe the same effective config that the
    # environment will consume.  Finalize the launch-time pending sentinel
    # before every cross-rank/checkpoint preflight and before importing Isaac.
    tyro_cfg = _effective_runtime_config(tyro_cfg)
    _validate_hierarchical_small_collectives_launch_contract()
    _configure_defm_materialization_mode(tyro_cfg)
    launch_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _current_rank_training_seed(tyro_cfg.training.seed)
    finalized_provenance = finalize_runtime_asset_provenance(tyro_cfg)
    _validate_prestarted_runtime_provenance(finalized_provenance)
    if finalized_provenance is not None:
        print(
            "[INFO] runtime_asset_provenance_finalized "
            f"sha256={finalized_provenance['runtime_asset_manifest_sha256']}",
            flush=True,
        )
    _preflight_data_assets_before_sim()
    # torchrun publishes WORLD_SIZE before this process imports/initializes the
    # simulator. Validate the global environment count here so a bad launcher
    # contract cannot be rounded down after Isaac has already started.
    _per_rank_env_count(tyro_cfg.training.num_envs, launch_world_size)
    _preflight_cross_rank_provenance_before_sim()
    _preflight_checkpoint_lineage_before_sim(tyro_cfg)
    tyro_cfg = _preflight_policy_init_before_sim(tyro_cfg)
    tyro_cfg = _preflight_training_resume_before_sim(tyro_cfg)
    print(tyro_cfg.curriculum)
    train(tyro_cfg)


if __name__ == "__main__":
    main()
