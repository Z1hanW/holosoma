from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import tyro
from loguru import logger

from holosoma.config_types.command import MotionConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.terrain import MeshType
from holosoma.observation import apply_observation_overrides
from holosoma.perception import apply_perception_overrides
from holosoma.utils.config_utils import CONFIG_NAME
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
    viser_port: int = 6060
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

    if geom_path.is_dir():
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


def run_physics_rollout(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
) -> None:
    env, device, simulation_app = setup_simulation_environment(tyro_config)

    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval")
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
    algo.setup()
    algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
    algo.load(checkpoint_path)
    algo.evaluate_policy(max_eval_steps=tyro_config.training.max_eval_steps)

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
    eval_cfg = saved_cfg.get_eval_config()
    eval_cfg_overrides, _ = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining,
        return_unknown_args=True,
        description="ExperimentConfig overrides.",
        config=TYRO_CONIFG,
    )

    motion_path = _resolve_motion_path(inputs.motion_dir)
    geom_path, geom_meta, num_rows, num_cols, pairing_supported = _resolve_geometry_inputs(inputs)

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
    tyro_config = _update_motion_config(tyro_config, motion_path, pair_terrain)
    tyro_config = _update_terrain_config(tyro_config, geom_path, geom_meta, num_rows, num_cols)
    tyro_config = _update_training_config(tyro_config, inputs)

    run_physics_rollout(tyro_config, checkpoint_cfg, saved_cfg, saved_wandb_path)


if __name__ == "__main__":
    main()
