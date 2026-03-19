#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

import holosoma.config_values.logger as logger_values
import holosoma.config_values.simulator as simulator_values
from holosoma.config_types.command import MotionConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg
from holosoma.observation import apply_observation_overrides
from holosoma.perception import apply_perception_overrides
from holosoma.utils.eval_utils import CheckpointConfig, init_eval_logging, load_saved_experiment_config
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment


DEFAULT_CHECKPOINT = "wandb://zihanw22/boxer/0z2aggr2/model_05000.pt"
DEFAULT_MOTION = (
    "/home/ubuntu/FAR/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/"
    "whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
)
DEFAULT_OBJECT_URDF = (
    "/home/ubuntu/FAR/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/"
    "whole_body_tracking/objects_largebox.urdf"
)


def _update_motion_config(config: ExperimentConfig, motion_path: str) -> ExperimentConfig:
    command_cfg = config.command
    term = command_cfg.setup_terms.get("motion_command")
    if term is None:
        raise ValueError("motion_command is missing from command manager config.")

    params = dict(term.params)
    motion_cfg = params.get("motion_config")
    if isinstance(motion_cfg, MotionConfig):
        motion_cfg = dataclasses.replace(motion_cfg, motion_file=motion_path, pair_terrain_with_motion=False)
    elif isinstance(motion_cfg, dict):
        updated = dict(motion_cfg)
        updated["motion_file"] = motion_path
        updated["pair_terrain_with_motion"] = False
        motion_cfg = MotionConfig(**updated)
    else:
        raise ValueError(f"Unsupported motion_config type: {type(motion_cfg)}")

    params["motion_config"] = motion_cfg
    updated_term = dataclasses.replace(term, params=params)
    updated_terms = dict(command_cfg.setup_terms)
    updated_terms["motion_command"] = updated_term
    return dataclasses.replace(config, command=dataclasses.replace(command_cfg, setup_terms=updated_terms))


def _maybe_update_term(
    terms: dict[str, RandomizationTermCfg],
    term_name: str,
    key: str,
    value: Any,
) -> dict[str, RandomizationTermCfg]:
    term = terms.get(term_name)
    if term is None:
        return terms
    params = dict(term.params)
    params[key] = value
    updated = dict(terms)
    updated[term_name] = dataclasses.replace(term, params=params)
    return updated


def _disable_randomization(config: ExperimentConfig) -> ExperimentConfig:
    rand_cfg = config.randomization
    setup_terms = dict(rand_cfg.setup_terms)
    reset_terms = dict(rand_cfg.reset_terms)
    step_terms = dict(rand_cfg.step_terms)

    for name in [
        "push_randomizer_state",
        "setup_action_delay_buffers",
        "randomize_robot_rigid_body_material_startup",
        "randomize_base_com_startup",
        "setup_dof_pos_bias",
        "setup_camera_raycast_randomization",
        "randomize_object_rigid_body_material_startup",
        "randomize_object_rigid_body_mass_startup",
        "randomize_object_rigid_body_inertia_startup",
    ]:
        setup_terms = _maybe_update_term(setup_terms, name, "enabled", False)

    setup_terms = _maybe_update_term(setup_terms, "actuator_randomizer_state", "enable_pd_gain", False)
    setup_terms = _maybe_update_term(setup_terms, "actuator_randomizer_state", "enable_rfi_lim", False)

    for name in ["randomize_push_schedule", "randomize_action_delay", "randomize_camera_raycast"]:
        reset_terms = _maybe_update_term(reset_terms, name, "enabled", False)
    reset_terms = _maybe_update_term(reset_terms, "randomize_dof_state", "randomize_dof_pos_bias", False)

    step_terms = _maybe_update_term(step_terms, "apply_pushes", "enabled", False)

    return dataclasses.replace(
        config,
        randomization=dataclasses.replace(
            rand_cfg,
            setup_terms=setup_terms,
            reset_terms=reset_terms,
            step_terms=step_terms,
            ignore_unsupported=True,
        ),
    )


def _build_eval_config(checkpoint: str, motion_path: str, source: str) -> ExperimentConfig:
    saved_cfg, _saved_wandb_path = load_saved_experiment_config(CheckpointConfig(checkpoint=checkpoint))
    cfg = saved_cfg.get_eval_config()
    cfg = apply_observation_overrides(cfg)
    cfg = apply_perception_overrides(cfg)
    cfg = _update_motion_config(cfg, motion_path)
    cfg = _disable_randomization(cfg)

    sim_cfg = dataclasses.replace(
        simulator_values.mujoco.config,
        robot_mjcf_filter=dataclasses.replace(
            simulator_values.mujoco.config.robot_mjcf_filter,
            enable=True,
            remove_lights=True,
            remove_ground=True,
        ),
        mujoco_backend=simulator_values.mujoco.config.mujoco_backend,
        sim=dataclasses.replace(
            simulator_values.mujoco.config.sim,
            max_episode_length_s=cfg.simulator.config.sim.max_episode_length_s,
        ),
    )

    robot_object_cfg = dataclasses.replace(
        cfg.robot.object,
        enabled=True,
        object_urdf_path=DEFAULT_OBJECT_URDF,
        mujoco_use_training_urdf_scene=True,
        mujoco_add_default_actuators=True,
        mujoco_copy_joint_defaults_from_robot_xml=True,
        mujoco_copy_tendons_from_robot_xml=True,
        mujoco_copy_collision_geoms_from_robot_xml=True,
        mujoco_copy_contact_pairs_from_robot_xml=True,
    )

    perception_cfg = dataclasses.replace(cfg.perception, camera_source=source)

    return dataclasses.replace(
        cfg,
        simulator=dataclasses.replace(simulator_values.mujoco, config=sim_cfg),
        robot=dataclasses.replace(cfg.robot, object=robot_object_cfg),
        perception=perception_cfg,
        training=dataclasses.replace(cfg.training, headless=True, num_envs=1, enable_viser=False),
        logger=logger_values.disabled,
    )


def _capture_depth_pair(config: ExperimentConfig) -> dict[str, np.ndarray]:
    env = None
    simulation_app = None
    try:
        env, _device, simulation_app = setup_simulation_environment(config)
        env.simulator.sim.step()
        env.step_visualize_motion(None)  # type: ignore[attr-defined]
        env.perception_manager.update()
        raw = env.perception_manager.get_camera_depth_map()[0].detach().cpu().numpy().astype(np.float32)
        obs = env.perception_manager.get_camera_depth_obs_map()[0].detach().cpu().numpy().astype(np.float32)
        return {"raw": raw, "obs": obs}
    finally:
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                logger.warning("Failed to close environment cleanly.", exc_info=True)
        close_simulation_app(simulation_app)


def _metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a - b
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "l1": float(np.mean(np.abs(diff))),
        "linf": float(np.max(np.abs(diff))),
    }


def main() -> None:
    init_eval_logging()
    parser = argparse.ArgumentParser(description="Compare MuJoCo rendered vs far_tracking_warp depth on the same frame.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--motion", default=DEFAULT_MOTION)
    parser.add_argument("--output-dir", default="/tmp/holosoma_depth_compare")
    parser.add_argument("--source-a", default="rendered")
    parser.add_argument("--source-b", default="far_tracking_warp")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    captures: dict[str, dict[str, np.ndarray]] = {}
    for source in [args.source_a, args.source_b]:
        logger.info("Capturing depth for source={} ...", source)
        cfg = _build_eval_config(args.checkpoint, args.motion, source)
        capture = _capture_depth_pair(cfg)
        captures[source] = capture
        np.save(output_dir / f"{source}_raw.npy", capture["raw"])
        np.save(output_dir / f"{source}_obs.npy", capture["obs"])

    summary = {
        "checkpoint": args.checkpoint,
        "motion": args.motion,
        "source_a": args.source_a,
        "source_b": args.source_b,
        "raw_metrics": _metrics(captures[args.source_a]["raw"], captures[args.source_b]["raw"]),
        "obs_metrics": _metrics(captures[args.source_a]["obs"], captures[args.source_b]["obs"]),
        "raw_shape_a": list(captures[args.source_a]["raw"].shape),
        "raw_shape_b": list(captures[args.source_b]["raw"].shape),
        "obs_shape_a": list(captures[args.source_a]["obs"].shape),
        "obs_shape_b": list(captures[args.source_b]["obs"].shape),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
