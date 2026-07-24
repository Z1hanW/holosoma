from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import trimesh
import tyro
from loguru import logger

from holosoma.config_types.command import MotionConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.terrain import MeshType
from holosoma.observation import apply_observation_overrides
from holosoma.perception import apply_perception_overrides
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.defm_runtime import set_defm_checkpoint_restore_mode
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG


@dataclass(frozen=True)
class PhysicsRolloutInputs:
    motion_dir: str
    geometry_dir: str | None = None
    geometry_metadata: str | None = None
    num_rows: int = 1
    num_cols: int | None = None
    headless: bool = True
    num_envs: int = 1
    pair_terrain_with_motion: bool = True
    viser_port: int = 0
    viser_env_id: int = 0
    viser_update_hz: float = 30.0
    viser_recenter: bool = True


def _resolve_data_path(path: str) -> str:
    if path.startswith("@holosoma/"):
        return str(Path(get_holosoma_root()) / path[len("@holosoma/") :])
    return resolve_data_file_path(path)


def _resolve_motion_path(path: str) -> str:
    resolved = _resolve_data_path(path)
    if resolved.startswith("s3://"):
        return resolved
    motion_path = Path(resolved).expanduser()
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion path not found: {motion_path}")
    return str(motion_path)


def _resolve_geometry_inputs(
    inputs: PhysicsRolloutInputs,
    motion_path: str | None = None,
    motion_clip_name: str | None = None,
    motion_clip_id: int | None = None,
    pair_terrain_with_motion: bool | None = None,
) -> tuple[str | None, str | None, int | None, int | None, bool]:
    if inputs.geometry_dir is None:
        return None, None, None, None, False

    resolved = _resolve_data_path(inputs.geometry_dir)
    if resolved.startswith("s3://"):
        raise ValueError("Geometry must be a local path (OBJ directory or file).")
    geom_path = Path(resolved).expanduser()
    if geom_path.is_file() and geom_path.suffix.lower() == ".urdf":
        candidate = geom_path.with_suffix(".obj")
        if candidate.exists():
            geom_path = candidate

    def _create_single_tile_metadata(mesh_path: Path, clip_name: str) -> str:
        mesh = trimesh.load(str(mesh_path), process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Loaded geometry is not a trimesh: {type(mesh)}")

        bounds = mesh.bounds.astype(float)
        span = bounds[1] - bounds[0]
        stride = [float(span[0]), float(span[1]), 0.0]
        meta = {
            "tile_names": [clip_name],
            "tile_offsets": [[0.0, 0.0, 0.0]],
            "tile_stride": stride,
            "tile_rows": 1,
            "tile_cols": 1,
            "tile_max_z": [0.0],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(meta, handle)
            return handle.name

    if geom_path.is_dir():
        clip_name = motion_clip_name
        if clip_name is None and motion_path:
            motion_candidate = Path(motion_path).expanduser()
            if motion_candidate.is_file():
                clip_name = motion_candidate.stem
            elif motion_clip_id is not None and motion_candidate.is_dir():
                motion_files = sorted(motion_candidate.glob("*.npz"))
                if 0 <= int(motion_clip_id) < len(motion_files):
                    clip_name = motion_files[int(motion_clip_id)].stem
        if clip_name:
            matches = list(geom_path.glob(f"{clip_name}.obj")) + list(geom_path.glob(f"{clip_name}.OBJ"))
            if not matches:
                matches = [
                    path
                    for path in geom_path.glob("*.obj")
                    if path.stem.lower() == clip_name.lower()
                ] + [
                    path
                    for path in geom_path.glob("*.OBJ")
                    if path.stem.lower() == clip_name.lower()
                ]
            if matches:
                selected = matches[0]
                logger.info("Using geometry '{}' for motion clip '{}'.", selected.name, clip_name)
                num_rows = max(1, int(inputs.num_rows))
                num_cols = int(inputs.num_cols or 1)
                if pair_terrain_with_motion:
                    meta_path = _create_single_tile_metadata(selected, clip_name)
                    logger.info("Generated temporary geometry metadata at {}.", meta_path)
                    return str(selected), meta_path, num_rows, num_cols, True
                return str(selected), None, num_rows, num_cols, False
            logger.warning(
                "No geometry OBJ matching clip '{}' in {}; loading all OBJ tiles.",
                clip_name,
                geom_path,
            )
        obj_files = sorted(list(geom_path.glob("*.obj")) + list(geom_path.glob("*.OBJ")))
        if not obj_files:
            raise FileNotFoundError(f"No OBJ files found in geometry directory: {geom_path}")
        if inputs.geometry_metadata:
            logger.warning("Ignoring geometry_metadata for directory input: {}", inputs.geometry_metadata)
        num_rows = max(1, int(inputs.num_rows))
        num_cols = int(inputs.num_cols or len(obj_files))
        return str(geom_path), None, num_rows, num_cols, True

    if not geom_path.exists():
        raise FileNotFoundError(f"Geometry path not found: {geom_path}")
    if geom_path.suffix.lower() != ".obj":
        raise ValueError(f"Unsupported geometry file: {geom_path} (expected .obj)")

    meta_path = None
    if inputs.geometry_metadata:
        meta_resolved = _resolve_data_path(inputs.geometry_metadata)
        meta_candidate = Path(meta_resolved).expanduser()
        if not meta_candidate.exists():
            raise FileNotFoundError(f"Geometry metadata not found: {meta_candidate}")
        meta_path = str(meta_candidate)
    elif pair_terrain_with_motion:
        clip_name = motion_clip_name or geom_path.stem
        meta_path = _create_single_tile_metadata(geom_path, clip_name)
        logger.info("Generated temporary geometry metadata at {}.", meta_path)

    num_rows = max(1, int(inputs.num_rows))
    num_cols = int(inputs.num_cols or 1)
    return str(geom_path), meta_path, num_rows, num_cols, meta_path is not None


def _update_motion_config(
    config: ExperimentConfig,
    motion_path: str,
    pair_terrain: bool,
) -> ExperimentConfig:
    command_cfg = config.command
    if command_cfg is None:
        raise ValueError("Experiment config has no command manager; motion input is required.")

    term = command_cfg.setup_terms.get("motion_command")
    if term is None:
        raise ValueError("Command manager has no motion_command; use a WBT experiment config.")

    params = dict(term.params)
    motion_cfg = params.get("motion_config")
    if motion_cfg is None:
        raise ValueError("motion_command is missing motion_config.")
    if isinstance(motion_cfg, MotionConfig):
        motion_cfg = dataclasses.replace(
            motion_cfg,
            motion_file=motion_path,
            pair_terrain_with_motion=pair_terrain,
        )
    elif isinstance(motion_cfg, dict):
        updated = dict(motion_cfg)
        updated["motion_file"] = motion_path
        updated["pair_terrain_with_motion"] = pair_terrain
        motion_cfg = MotionConfig(**updated)
    else:
        raise ValueError(f"Unsupported motion_config type: {type(motion_cfg)}")

    params["motion_config"] = motion_cfg
    term = dataclasses.replace(term, params=params)
    setup_terms = dict(command_cfg.setup_terms)
    setup_terms["motion_command"] = term
    command_cfg = dataclasses.replace(command_cfg, setup_terms=setup_terms)
    return dataclasses.replace(config, command=command_cfg)


def _extract_motion_clip_hint(config: ExperimentConfig) -> tuple[str | None, int | None]:
    command_cfg = config.command
    if command_cfg is None:
        return None, None
    term = command_cfg.setup_terms.get("motion_command")
    if term is None:
        return None, None
    motion_cfg = term.params.get("motion_config")
    if isinstance(motion_cfg, MotionConfig):
        return motion_cfg.motion_clip_name, motion_cfg.motion_clip_id
    if isinstance(motion_cfg, dict):
        clip_name = motion_cfg.get("motion_clip_name")
        clip_id = motion_cfg.get("motion_clip_id")
        return clip_name, clip_id
    return None, None


def _normalize_motion_config_for_cli(config: ExperimentConfig) -> ExperimentConfig:
    """Coerce legacy dict payloads into MotionConfig so Tyro accepts typed overrides."""

    command_cfg = config.command
    if command_cfg is None:
        return config

    term = command_cfg.setup_terms.get("motion_command")
    if term is None:
        return config

    params = dict(term.params)
    motion_cfg = params.get("motion_config")
    if isinstance(motion_cfg, MotionConfig):
        return config
    if not isinstance(motion_cfg, dict):
        return config

    try:
        params["motion_config"] = MotionConfig(**motion_cfg)
    except Exception as exc:
        logger.warning(
            "Failed to coerce motion_config dict into MotionConfig for CLI overrides; keeping raw dict. Error: {}",
            exc,
        )
        return config

    term = dataclasses.replace(term, params=params)
    setup_terms = dict(command_cfg.setup_terms)
    setup_terms["motion_command"] = term
    command_cfg = dataclasses.replace(command_cfg, setup_terms=setup_terms)
    return dataclasses.replace(config, command=command_cfg)


def _update_terrain_config(
    config: ExperimentConfig,
    geometry_path: str | None,
    geometry_metadata: str | None,
    num_rows: int | None,
    num_cols: int | None,
) -> ExperimentConfig:
    term_cfg = config.terrain.terrain_term
    if geometry_path is None:
        term_cfg = dataclasses.replace(
            term_cfg,
            mesh_type=MeshType.PLANE,
            obj_file_path="",
            obj_metadata_path=None,
            num_rows=1,
            num_cols=1,
        )
    else:
        term_cfg = dataclasses.replace(
            term_cfg,
            mesh_type=MeshType.LOAD_OBJ,
            obj_file_path=geometry_path,
            obj_metadata_path=geometry_metadata,
            num_rows=int(num_rows or term_cfg.num_rows),
            num_cols=int(num_cols or term_cfg.num_cols),
        )
    terrain_cfg = dataclasses.replace(config.terrain, terrain_term=term_cfg)
    return dataclasses.replace(config, terrain=terrain_cfg)


def _update_training_config(
    config: ExperimentConfig,
    inputs: PhysicsRolloutInputs,
) -> ExperimentConfig:
    training_cfg = dataclasses.replace(
        config.training,
        headless=bool(inputs.headless),
        num_envs=int(inputs.num_envs),
        enable_viser=True,
        viser_port=int(inputs.viser_port),
        viser_env_id=int(inputs.viser_env_id),
        viser_update_hz=float(inputs.viser_update_hz),
        viser_recenter=bool(inputs.viser_recenter),
    )
    return dataclasses.replace(config, training=training_cfg)


def _update_algo_config(config: ExperimentConfig) -> ExperimentConfig:
    algo_cfg = config.algo.config
    if hasattr(algo_cfg, "load_optimizer") and getattr(algo_cfg, "load_optimizer"):
        logger.info("Disabling optimizer load for physics rollout.")
        algo_cfg = dataclasses.replace(algo_cfg, load_optimizer=False)
        algo = dataclasses.replace(config.algo, config=algo_cfg)
        return dataclasses.replace(config, algo=algo)
    return config


def _scrub_rollout_reference_reward_roots_for_eval(config: ExperimentConfig) -> ExperimentConfig:
    """Disable rollout-reference reward sidecars for lightweight video eval.

    Some training checkpoints carry reward params that point at node-local
    teacher rollout/contact export roots. Video rollout does not need those
    rewards, and missing sidecars should not prevent rendering the policy.
    This is opt-in through an environment variable so normal training/resume
    semantics stay unchanged.
    """

    raw = os.environ.get("HOLOSOMA_EVAL_DISABLE_ROLLOUT_REFERENCE_REWARDS", "")
    if raw.strip().lower() not in {"1", "true", "yes", "on"}:
        return config

    reward_cfg = getattr(config, "reward", None)
    terms = getattr(reward_cfg, "terms", None)
    if not isinstance(terms, dict):
        return config

    updated_terms = {}
    scrubbed_roots = 0
    dropped_terms = 0
    for name, term in terms.items():
        params = dict(getattr(term, "params", {}) or {})
        if "contact_export_root" in params:
            dropped_terms += 1
            continue
        if "rollout_reference_root" in params:
            params["rollout_reference_root"] = None
            term = dataclasses.replace(term, params=params)
            scrubbed_roots += 1
        updated_terms[name] = term

    if scrubbed_roots == 0 and dropped_terms == 0:
        return config
    logger.info(
        "Adjusted video-eval reward sidecars: scrubbed rollout_reference_root on {} term(s), dropped {} contact-export term(s).",
        scrubbed_roots,
        dropped_terms,
    )
    reward_cfg = dataclasses.replace(reward_cfg, terms=updated_terms)
    return dataclasses.replace(config, reward=reward_cfg)


def run_physics_rollout(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
) -> None:
    set_defm_checkpoint_restore_mode()
    env, device, simulation_app = setup_simulation_environment(tyro_config)
    try:
        eval_log_dir = get_experiment_dir(
            tyro_config.logger,
            tyro_config.training,
            get_timestamp(),
            task_name="eval",
        )
        eval_log_dir.mkdir(parents=True, exist_ok=True)
        tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

        checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
        checkpoint_path = str(checkpoint)

        algo_class = get_class(tyro_config.algo._target_)
        algo = algo_class(
            device=device,
            env=env,
            config=tyro_config.algo.config,
            log_dir=str(eval_log_dir),
            multi_gpu_cfg=None,
        )
        algo.attach_evaluation_metadata(
            saved_config,
            tyro_config,
            saved_wandb_path,
        )
        algo.setup()
        algo.load_evaluation(checkpoint_path)
        try:
            algo.evaluate_policy(max_eval_steps=tyro_config.training.max_eval_steps)
        finally:
            simulator = getattr(env, "simulator", None)
            video_recorder = getattr(simulator, "video_recorder", None)
            if video_recorder is not None:
                video_recorder.stop_recording()
    finally:
        if simulation_app:
            close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, remaining = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    inputs, remaining = tyro.cli(
        PhysicsRolloutInputs,
        args=remaining,
        return_unknown_args=True,
        description="Required physics rollout inputs.",
    )

    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = _normalize_motion_config_for_cli(saved_cfg.get_eval_config())
    eval_cfg_overrides, _ = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining,
        return_unknown_args=True,
        description="ExperimentConfig overrides.",
        config=TYRO_CONIFG,
    )

    motion_path = _resolve_motion_path(inputs.motion_dir)
    clip_name, clip_id = _extract_motion_clip_hint(eval_cfg_overrides)
    geom_path, geom_meta, num_rows, num_cols, pairing_supported = _resolve_geometry_inputs(
        inputs,
        motion_path=motion_path,
        motion_clip_name=clip_name,
        motion_clip_id=clip_id,
        pair_terrain_with_motion=inputs.pair_terrain_with_motion,
    )

    pair_terrain = bool(inputs.pair_terrain_with_motion)
    if geom_path is None:
        if pair_terrain:
            logger.warning("pair_terrain_with_motion disabled (geometry not provided).")
        pair_terrain = False
    elif pair_terrain and not pairing_supported:
        logger.warning("pair_terrain_with_motion requires OBJ tiles or metadata; disabling.")
        pair_terrain = False

    tyro_config = apply_observation_overrides(eval_cfg_overrides)
    tyro_config = apply_perception_overrides(tyro_config)
    # Keep checkpoint actor inputs immutable while still allowing the motion,
    # terrain, and visualization overrides applied below.
    from holosoma.eval_agent import (
        _bind_training_perception_reference_batch,
        _validate_eval_policy_contract,
    )

    _validate_eval_policy_contract(saved_cfg, tyro_config)
    tyro_config = _update_motion_config(tyro_config, motion_path, pair_terrain)
    tyro_config = _update_terrain_config(tyro_config, geom_path, geom_meta, num_rows, num_cols)
    tyro_config = _update_training_config(tyro_config, inputs)
    # A single-environment visual rollout must retain the training batch as
    # the normalization reference for camera-hole sampling.  The standard
    # eval_agent path already performs this binding; physics/Viser rollouts
    # must do the same after their final environment count is known.
    tyro_config = _bind_training_perception_reference_batch(saved_cfg, tyro_config)
    tyro_config = _update_algo_config(tyro_config)
    tyro_config = _scrub_rollout_reference_reward_roots_for_eval(tyro_config)

    run_physics_rollout(tyro_config, checkpoint_cfg, saved_cfg, saved_wandb_path)


if __name__ == "__main__":
    main()
