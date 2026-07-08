#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import tyro
from loguru import logger

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import CheckpointConfig, init_eval_logging, load_checkpoint, load_saved_experiment_config
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Current-repo IsaacSim physics rollout streamed to Viser.")
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint path or wandb:// checkpoint URI.")
    parser.add_argument("--port", type=int, default=2099, help="Viser server port.")
    parser.add_argument("--env-id", type=int, default=0, help="Environment index to stream.")
    parser.add_argument(
        "--sequence-envs",
        type=int,
        default=0,
        help="Minimum eval env count for sequence switching. Use at least one env per object URDF for object banks.",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until stopped.")
    parser.add_argument("--update-hz", type=float, default=30.0, help="Viser publish rate.")
    parser.add_argument("--red-points", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--red-point-size", type=float, default=0.035)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--randomize-tiles",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override eval_config terrain tile randomization. Defaults to the run's eval_overrides.",
    )
    parser.add_argument(
        "--xy-offset-range",
        type=float,
        default=None,
        help="Override eval_config terrain xy offset range. Defaults to the run's eval_overrides.",
    )
    parser.add_argument("--disable-randomization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Additional ExperimentConfig overrides after '--', using the same Tyro syntax as eval_agent.py.",
    )
    return parser.parse_args()


def _resolve_holosoma_path(path: str) -> Path:
    if path.startswith("@holosoma/"):
        return Path(get_holosoma_root()) / path[len("@holosoma/") :]
    return Path(resolve_data_file_path(path))


def _robot_urdf_path(config: ExperimentConfig) -> Path:
    asset_root = _resolve_holosoma_path(config.robot.asset.asset_root)
    return asset_root / config.robot.asset.urdf_file


def _disable_randomization(config: ExperimentConfig) -> ExperimentConfig:
    if config.randomization is None:
        return config

    def _disable_term(term: RandomizationTermCfg) -> RandomizationTermCfg:
        params = dict(term.params)
        if "enabled" in params:
            params["enabled"] = False
        return dataclasses.replace(term, params=params)

    setup_terms = {name: _disable_term(term) for name, term in config.randomization.setup_terms.items()}
    reset_terms = {
        name: term
        for name, term in config.randomization.reset_terms.items()
        if name
        not in {
            "push_randomizer_state",
            "randomize_push_schedule",
            "randomize_action_delay",
            "randomize_dof_state",
            "actuator_randomizer_state",
        }
    }
    step_terms = {
        name: term
        for name, term in config.randomization.step_terms.items()
        if name not in {"push_randomizer_state", "apply_pushes"}
    }
    randomization = RandomizationManagerCfg(
        setup_terms=setup_terms,
        reset_terms=reset_terms,
        step_terms=step_terms,
        ignore_unsupported=config.randomization.ignore_unsupported,
    )
    return dataclasses.replace(config, randomization=randomization)


def _make_eval_config(args: argparse.Namespace, saved_config: ExperimentConfig) -> ExperimentConfig:
    eval_config = saved_config.get_eval_config()
    randomize_tiles = (
        eval_config.terrain.terrain_term.spawn.randomize_tiles
        if args.randomize_tiles is None
        else bool(args.randomize_tiles)
    )
    xy_offset_range = (
        eval_config.terrain.terrain_term.spawn.xy_offset_range
        if args.xy_offset_range is None
        else float(args.xy_offset_range)
    )
    spawn = dataclasses.replace(
        eval_config.terrain.terrain_term.spawn,
        randomize_tiles=bool(randomize_tiles),
        xy_offset_range=float(xy_offset_range),
    )
    eval_config = dataclasses.replace(
        eval_config,
        terrain=dataclasses.replace(
            eval_config.terrain,
            terrain_term=dataclasses.replace(eval_config.terrain.terrain_term, spawn=spawn),
        ),
        training=dataclasses.replace(
            eval_config.training,
            headless=bool(args.headless),
            num_envs=max(int(args.env_id) + 1, int(args.sequence_envs), 1),
            max_eval_steps=None if args.max_steps <= 0 else int(args.max_steps),
            export_onnx=False,
        ),
    )
    if args.disable_randomization:
        eval_config = _disable_randomization(eval_config)

    overrides = list(args.overrides)
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]
    if overrides:
        eval_config = tyro.cli(
            ExperimentConfig,
            default=eval_config,
            args=overrides,
            description="ExperimentConfig overrides.",
            config=TYRO_CONIFG,
        )
    return eval_config


def _load_mesh_for_viser(mesh: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh or Scene, got {type(mesh)}")
    return mesh


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def _tensor_to_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy()


def _tensor_row_to_numpy(value: Any, env_id: int) -> np.ndarray:
    return _tensor_to_numpy(value[env_id])


def _resolve_existing_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    path_str = str(path).strip()
    if not path_str:
        return None
    try:
        resolved = _resolve_holosoma_path(path_str)
    except Exception:
        resolved = Path(path_str)
    return resolved if resolved.exists() else None


def _object_path_key(path: str | Path | None) -> str | None:
    resolved = _resolve_existing_path(path)
    if resolved is None:
        return None
    try:
        return str(resolved.resolve())
    except Exception:
        return str(resolved)


def _safe_scene_name(path: Path) -> str:
    stem = path.stem or "object"
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in stem)


def _object_urdf_path_for_env(env: Any, motion_state: Any | None, env_id: int) -> Path | None:
    env_object_paths = getattr(env.simulator, "_env_object_urdf_paths", None)
    if isinstance(env_object_paths, list) and 0 <= env_id < len(env_object_paths):
        path = _resolve_existing_path(env_object_paths[env_id])
        if path is not None:
            return path

    object_urdf_by_name = getattr(env.simulator, "_object_urdf_by_name", None)
    if isinstance(object_urdf_by_name, dict):
        for path_str in object_urdf_by_name.values():
            path = _resolve_existing_path(path_str)
            if path is not None:
                return path

    if motion_state is not None:
        try:
            motion_idx = int(motion_state.motion_ids[env_id].detach().cpu())
            clip_urdfs = getattr(motion_state.motion, "clip_object_urdf_paths", [])
            if 0 <= motion_idx < len(clip_urdfs):
                path = _resolve_existing_path(clip_urdfs[motion_idx])
                if path is not None:
                    return path
        except Exception:
            pass

    return None


def _object_urdf_path_for_clip(motion_state: Any | None, clip_idx: int | None) -> Path | None:
    if motion_state is None or clip_idx is None:
        return None
    clip_urdfs = getattr(getattr(motion_state, "motion", None), "clip_object_urdf_paths", None)
    if clip_urdfs is None or not (0 <= int(clip_idx) < len(clip_urdfs)):
        return None
    return _resolve_existing_path(clip_urdfs[int(clip_idx)])


def _clip_env_map(env: Any, motion_state: Any | None) -> dict[int, int]:
    if motion_state is None:
        return {}
    clip_urdfs = getattr(getattr(motion_state, "motion", None), "clip_object_urdf_paths", None)
    env_object_paths = getattr(env.simulator, "_env_object_urdf_paths", None)
    if clip_urdfs is None or not isinstance(env_object_paths, list):
        return {}

    env_by_object: dict[str, int] = {}
    for env_id, object_path in enumerate(env_object_paths[: env.num_envs]):
        key = _object_path_key(object_path)
        if key is not None and key not in env_by_object:
            env_by_object[key] = env_id

    mapping: dict[int, int] = {}
    for clip_idx, object_path in enumerate(clip_urdfs):
        key = _object_path_key(object_path)
        if key is not None and key in env_by_object:
            mapping[int(clip_idx)] = env_by_object[key]
    return mapping


def _force_motion_sequence(env: Any, motion_state: Any, clip_idx: int, env_id: int) -> dict[str, Any]:
    import torch

    clip_idx = int(clip_idx)
    env_id = int(env_id)
    env_ids = torch.tensor([env_id], device=env.device, dtype=torch.long)
    if not (0 <= clip_idx < int(motion_state.motion.num_motions)):
        raise ValueError(f"clip_idx {clip_idx} is out of range for {motion_state.motion.num_motions} clips")
    if not (0 <= env_id < int(env.num_envs)):
        raise ValueError(f"env_id {env_id} is out of range for {env.num_envs} envs")

    env.reset_envs_idx(env_ids)

    motion_state.motion_ids[env_ids] = clip_idx
    start_idx = motion_state.motion.motion_start_idx[clip_idx].to(device=env.device)
    end_idx = motion_state.motion.motion_end_idx[clip_idx].to(device=env.device)
    motion_state.time_steps[env_ids] = torch.minimum(start_idx, end_idx - 2).clamp_min(start_idx)
    motion_state._sync_env_origins_to_motion(env_ids)

    sim = env.simulator
    sim.dof_pos[env_ids] = motion_state.joint_pos[env_ids].clone()
    sim.dof_vel[env_ids] = motion_state.joint_vel[env_ids].clone()
    sim.robot_root_states[env_ids, :3] = motion_state.root_pos_w[env_ids].clone()
    sim.robot_root_states[env_ids, 3:7] = motion_state.root_quat_w[env_ids].clone()
    sim.robot_root_states[env_ids, 7:10] = motion_state.root_lin_vel_w[env_ids].clone()
    sim.robot_root_states[env_ids, 10:13] = motion_state.root_ang_vel_w[env_ids].clone()

    if getattr(motion_state.motion, "has_object", False):
        obj_pos = motion_state.object_pos_w[env_ids].clone()
        obj_quat = motion_state.object_quat_w[env_ids].clone()
        obj_lin_vel = motion_state.object_lin_vel_w[env_ids].clone()
        obj_states = torch.cat([obj_pos, obj_quat, obj_lin_vel, torch.zeros_like(obj_lin_vel)], dim=-1)
        motion_state._set_simulator_object_states(env_ids, obj_states)

    sim.set_actor_root_state_tensor_robots(env_ids, sim.robot_root_states)
    sim.set_dof_state_tensor_robots(env_ids, sim.dof_state)
    if hasattr(env, "_refresh_envs_after_reset"):
        env._refresh_envs_after_reset(env_ids)
    else:
        sim.refresh_sim_tensors()
        env._pre_compute_observations_callback()

    env.reset_buf[env_ids] = 0
    env.time_out_buf[env_ids] = False
    env.episode_length_buf[env_ids] = 0
    env._compute_observations()
    env._post_compute_observations_callback()
    env._clip_observations()
    return env.obs_buf_dict


def _spread_sequence_env_origins(env: Any) -> bool:
    import torch

    terrain_state = env.terrain_manager.get_state("locomotion_terrain")
    origins = getattr(terrain_state, "env_origins", None)
    if origins is None or int(origins.shape[0]) <= 1:
        return False

    terrain = getattr(terrain_state, "terrain", None)
    terrain_origins_np = getattr(terrain, "_env_origins", None)
    if terrain_origins_np is not None:
        terrain_origins = np.asarray(terrain_origins_np, dtype=np.float32).reshape(-1, 3)
        if terrain_origins.shape[0] >= env.num_envs:
            new_origins = torch.as_tensor(terrain_origins[: env.num_envs], device=origins.device, dtype=origins.dtype)
        else:
            new_origins = None
    else:
        new_origins = None

    if new_origins is None:
        cols = int(np.ceil(np.sqrt(env.num_envs)))
        spacing = float(
            max(
                getattr(getattr(terrain_state, "_cfg", None), "terrain_length", 8.0),
                getattr(getattr(terrain_state, "_cfg", None), "terrain_width", 8.0),
                4.0,
            )
        )
        base = origins[0].clone()
        new_origins = origins.clone()
        for env_id in range(env.num_envs):
            new_origins[env_id] = base
            new_origins[env_id, 0] += float(env_id % cols) * spacing
            new_origins[env_id, 1] += float(env_id // cols) * spacing

    if torch.allclose(origins, new_origins):
        return False

    origins[:] = new_origins
    scene_origins = getattr(getattr(env.simulator, "scene", None), "env_origins", None)
    if scene_origins is not None:
        scene_origins[:] = new_origins.to(device=scene_origins.device, dtype=scene_origins.dtype)
    simulator_origins = getattr(env.simulator, "env_origins", None)
    if simulator_origins is not None:
        simulator_origins[:] = new_origins.to(device=simulator_origins.device, dtype=simulator_origins.dtype)

    unique_origins = torch.unique(torch.round(new_origins[:, :2] * 10000.0) / 10000.0, dim=0).shape[0]
    logger.info(
        "Spread {} sequence env origins across {} unique terrain XY origins for collision isolation.",
        env.num_envs,
        unique_origins,
    )
    return True


def _simulator_object_state_wxyz(env: Any, motion_state: Any | None, env_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    if motion_state is not None and getattr(getattr(motion_state, "motion", None), "has_object", False):
        try:
            pos = _tensor_row_to_numpy(motion_state.simulator_object_pos_w, env_id).astype(np.float32)
            quat = _tensor_row_to_numpy(motion_state.simulator_object_quat_w, env_id).astype(np.float32)
            return pos, _xyzw_to_wxyz(quat)
        except Exception:
            pass

    try:
        import torch

        env_ids = torch.tensor([env_id], device=env.device, dtype=torch.long)
        states = env.simulator._get_object_states("object", env_ids)
        if states is not None and states.numel() > 0:
            state = _tensor_to_numpy(states[0])
            return state[:3].astype(np.float32), _xyzw_to_wxyz(state[3:7].astype(np.float32))
    except Exception:
        pass

    return None


def _reference_object_state_wxyz(motion_state: Any | None, env_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    if motion_state is None or not getattr(getattr(motion_state, "motion", None), "has_object", False):
        return None
    pos = _tensor_row_to_numpy(motion_state.object_pos_w, env_id).astype(np.float32)
    quat = _tensor_row_to_numpy(motion_state.object_quat_w, env_id).astype(np.float32)
    return pos, _xyzw_to_wxyz(quat)


def _reference_robot_state_wxyz(motion_state: Any | None, env_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    if motion_state is None:
        return None
    pos = _tensor_row_to_numpy(motion_state.root_pos_w, env_id).astype(np.float32)
    quat = _tensor_row_to_numpy(motion_state.root_quat_w, env_id).astype(np.float32)
    return pos, _xyzw_to_wxyz(quat)


def _reference_body_points(motion_state: Any | None, env_id: int) -> np.ndarray | None:
    if motion_state is None:
        return None
    try:
        points = _tensor_row_to_numpy(motion_state.body_pos_w, env_id).astype(np.float32)
    except Exception:
        return None
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return points[np.isfinite(points).all(axis=1)]


def _red_point_hits(env: Any, env_id: int) -> np.ndarray:
    """Return finite height-scan/raycast hit points for one env."""
    sensors = getattr(getattr(env.simulator, "scene", None), "sensors", {})
    sensor = sensors.get("height_scanner")
    if sensor is not None and hasattr(sensor, "data") and hasattr(sensor.data, "ray_hits_w"):
        points = sensor.data.ray_hits_w[env_id].detach().cpu().numpy()
    else:
        terrain_state = env.terrain_manager.get_state("locomotion_terrain")
        points = terrain_state._ray_hits_world_base[env_id].detach().cpu().numpy()

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return points[np.isfinite(points).all(axis=1)]


def main() -> None:
    args = _parse_args()
    init_eval_logging()
    logging.getLogger("trimesh").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)

    checkpoint_cfg = CheckpointConfig(checkpoint=args.checkpoint)
    saved_config, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_config = _make_eval_config(args, saved_config)

    env, device, simulation_app = setup_simulation_environment(eval_config)
    try:
        eval_log_dir = get_experiment_dir(eval_config.logger, eval_config.training, get_timestamp(), task_name="eval")
        eval_log_dir.mkdir(parents=True, exist_ok=True)
        eval_config.save_config(str(eval_log_dir / CONFIG_NAME))

        checkpoint = load_checkpoint(args.checkpoint, str(eval_log_dir))
        algo_class = get_class(eval_config.algo._target_)
        algo = algo_class(device=device, env=env, config=eval_config.algo.config, log_dir=str(eval_log_dir), multi_gpu_cfg=None)
        algo.setup()
        algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
        algo.load(str(checkpoint))

        import torch
        import viser
        from viser.extras import ViserUrdf

        server = viser.ViserServer(host="0.0.0.0", port=int(args.port), label="holosoma_current_physics")
        stream_env_id = {"value": int(args.env_id)}
        terrain_state = env.terrain_manager.get_state("locomotion_terrain")
        terrain_mesh = _load_mesh_for_viser(terrain_state.mesh)
        server.scene.add_mesh_simple(
            "/terrain",
            vertices=np.asarray(terrain_mesh.vertices, dtype=np.float32),
            faces=np.asarray(terrain_mesh.faces, dtype=np.int32),
            color=(95, 95, 95),
            opacity=0.78,
            side="double",
        )

        urdf_path = _robot_urdf_path(eval_config)
        logger.info("Viser using current-repo URDF: {}", urdf_path)
        robot_viser = ViserUrdf(server, urdf_path, root_node_name="/robot", load_meshes=True, load_collision_meshes=False)
        robot_root = robot_viser._visual_root_frame
        if robot_root is None:
            raise RuntimeError("ViserUrdf did not create a visual root frame.")

        viser_joint_names = list(robot_viser.get_actuated_joint_names())
        dof_name_to_idx = {name: i for i, name in enumerate(env.dof_names)}
        missing = [name for name in viser_joint_names if name not in dof_name_to_idx]
        if missing:
            raise RuntimeError(f"Viser joints missing from simulator DOFs: {missing}")
        viser_to_sim = torch.tensor([dof_name_to_idx[name] for name in viser_joint_names], device=device, dtype=torch.long)
        motion_state = env.command_manager.get_state("motion_command")
        if int(args.sequence_envs) > 1 and not eval_config.terrain.terrain_term.spawn.randomize_tiles:
            motion_terrain_origins = getattr(getattr(motion_state, "motion", None), "terrain_origins", None)
            if motion_terrain_origins is None:
                _spread_sequence_env_origins(env)

        motion_ref_viser = None
        motion_ref_root = None
        if motion_state is not None:
            motion_ref_viser = ViserUrdf(
                server,
                urdf_path,
                root_node_name="/motion_ref",
                mesh_color_override=(0.1, 0.45, 1.0, 0.32),
                load_meshes=True,
                load_collision_meshes=False,
            )
            motion_ref_root = motion_ref_viser._visual_root_frame
            logger.info("Viser streaming motion reference robot at /motion_ref")

        object_visual_cache: dict[str, dict[str, dict[str, Any]]] = {"actual": {}, "ref": {}}
        active_object_key: dict[str, str | None] = {"actual": None, "ref": None}

        def _activate_object_visual(kind: str, object_urdf_path: Path | None, color: tuple[float, float, float, float]):
            cache = object_visual_cache[kind]
            if object_urdf_path is None:
                if active_object_key[kind] is not None:
                    cache[active_object_key[kind]]["viser"].show_visual = False
                active_object_key[kind] = None
                return None

            key = _object_path_key(object_urdf_path)
            if key is None:
                return None
            if key not in cache:
                node_name = f"/object_{kind}/{_safe_scene_name(object_urdf_path)}"
                object_viser = ViserUrdf(
                    server,
                    object_urdf_path,
                    root_node_name=node_name,
                    mesh_color_override=color,
                    load_meshes=True,
                    load_collision_meshes=False,
                )
                object_viser.show_visual = False
                cache[key] = {"viser": object_viser, "root": object_viser._visual_root_frame, "node": node_name}
                logger.info("Viser loaded {} object mesh at {} from {}", kind, node_name, object_urdf_path)

            if active_object_key[kind] != key:
                if active_object_key[kind] is not None:
                    cache[active_object_key[kind]]["viser"].show_visual = False
                cache[key]["viser"].show_visual = True
                active_object_key[kind] = key
            return cache[key]["root"]

        logger.info("Viser listening on http://localhost:{}", args.port)
        logger.info(
            "Streaming env {} from {} eval envs; sequence_envs={} randomize_tiles={} xy_offset_range={} randomization_disabled={} red_points={}",
            args.env_id,
            env.num_envs,
            args.sequence_envs,
            eval_config.terrain.terrain_term.spawn.randomize_tiles,
            eval_config.terrain.terrain_term.spawn.xy_offset_range,
            args.disable_randomization,
            args.red_points,
        )

        algo._create_eval_callbacks()
        algo._pre_evaluate_policy()
        actor_state = algo._create_actor_state()
        algo.eval_policy = algo.get_inference_policy()

        obs_dict = env.reset_all()
        clip_to_env = _clip_env_map(env, motion_state)
        clip_names: list[str] = []
        sequence_options: list[str] = []
        option_to_clip: dict[str, int] = {}
        pending_clip_idx: dict[str, int | None] = {"idx": None}
        selected_clip_idx: dict[str, int | None] = {"idx": None}
        sequence_info = None

        if motion_state is not None and hasattr(motion_state, "motion_ids"):
            selected_clip_idx["idx"] = int(motion_state.motion_ids[stream_env_id["value"]].detach().cpu())

        raw_clip_names = getattr(getattr(motion_state, "motion", None), "clip_ids", []) if motion_state is not None else []
        for clip_idx, clip_name in enumerate(raw_clip_names):
            clip_path_name = Path(str(clip_name)).stem or str(clip_name)
            object_path = _object_urdf_path_for_clip(motion_state, clip_idx)
            object_suffix = f" | {object_path.stem}" if object_path is not None and object_path.stem not in clip_path_name else ""
            option = f"{clip_idx:03d} | {clip_path_name}{object_suffix}"
            clip_names.append(str(clip_name))
            sequence_options.append(option)
            option_to_clip[option] = int(clip_idx)

        if sequence_options and selected_clip_idx["idx"] is not None:
            initial_option = sequence_options[int(selected_clip_idx["idx"])]
            with server.gui.add_folder("Sequence"):
                sequence_dropdown = server.gui.add_dropdown(
                    "Clip",
                    options=tuple(sequence_options),
                    initial_value=initial_option,
                )
                sequence_info = server.gui.add_markdown("")

            @sequence_dropdown.on_update
            def _(_evt) -> None:
                pending_clip_idx["idx"] = option_to_clip[str(sequence_dropdown.value)]

            if getattr(getattr(motion_state, "motion", None), "has_object", False):
                clip_object_count = len(getattr(motion_state.motion, "clip_object_urdf_paths", []))
                if clip_object_count and len(clip_to_env) < clip_object_count:
                    logger.warning(
                        "Only {}/{} object clips have a matching eval env. "
                        "Use --sequence-envs >= number of object URDFs for fully correct object switching.",
                        len(clip_to_env),
                        clip_object_count,
                    )

        def _update_sequence_info() -> None:
            if sequence_info is None or selected_clip_idx["idx"] is None:
                return
            clip_idx = int(selected_clip_idx["idx"])
            env_id = int(stream_env_id["value"])
            clip_label = sequence_options[clip_idx] if 0 <= clip_idx < len(sequence_options) else str(clip_idx)
            actual_path = _object_urdf_path_for_env(env, motion_state, env_id)
            ref_path = _object_urdf_path_for_clip(motion_state, clip_idx)
            content = f"Clip: `{clip_label}` | env: `{env_id}` / `{env.num_envs}`"
            if actual_path is not None:
                content += f" | actual object: `{actual_path.stem}`"
            if ref_path is not None and _object_path_key(actual_path) != _object_path_key(ref_path):
                content += f" | ref object: `{ref_path.stem}`"
            sequence_info.content = content

        _update_sequence_info()

        red_points_handle = None
        motion_ref_points_handle = None
        if args.red_points:
            red_points = _red_point_hits(env, stream_env_id["value"])
            red_points_handle = server.scene.add_point_cloud(
                "/height_scan_red_points",
                red_points,
                colors=(255, 0, 0),
                point_size=float(args.red_point_size),
                point_shape="circle",
                point_shading="flat",
                precision="float32",
            )
        motion_ref_points = _reference_body_points(motion_state, stream_env_id["value"])
        if motion_ref_points is not None:
            motion_ref_points_handle = server.scene.add_point_cloud(
                "/motion_ref_body_points",
                motion_ref_points,
                colors=(0, 180, 255),
                point_size=0.035,
                point_shape="circle",
                point_shading="flat",
                precision="float32",
            )
        init_actions = torch.zeros(env.num_envs, algo.num_act, device=device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        critic_obs = torch.cat([actor_state["obs"][key] for key in algo.critic_obs_keys], dim=1)
        actor_state["obs"]["critic_obs"] = critic_obs
        actor_state = algo._pre_eval_env_step(actor_state)

        def _set_actor_obs(new_obs_dict: dict[str, Any]) -> None:
            actor_state["obs"] = new_obs_dict
            actor_state["actions"] = torch.zeros_like(actor_state["actions"])
            actor_state["obs"]["critic_obs"] = torch.cat([actor_state["obs"][key] for key in algo.critic_obs_keys], dim=1)

        def _apply_sequence(clip_idx: int, *, reason: str) -> None:
            if motion_state is None:
                return
            target_env_id = int(clip_to_env.get(int(clip_idx), stream_env_id["value"]))
            if int(clip_idx) not in clip_to_env and getattr(getattr(motion_state, "motion", None), "has_object", False):
                logger.warning(
                    "No eval env has the object URDF for clip {}; keeping env {}. Actual object mesh may not match reference.",
                    clip_idx,
                    target_env_id,
                )
            stream_env_id["value"] = target_env_id
            selected_clip_idx["idx"] = int(clip_idx)
            new_obs_dict = _force_motion_sequence(env, motion_state, int(clip_idx), target_env_id)
            _set_actor_obs(new_obs_dict)
            _update_sequence_info()
            clip_name = clip_names[int(clip_idx)] if 0 <= int(clip_idx) < len(clip_names) else str(clip_idx)
            logger.info("Applied sequence {} on env {} ({}) via {}", clip_idx, target_env_id, clip_name, reason)

        min_period = 0.0 if args.update_hz <= 0 else 1.0 / float(args.update_hz)
        last_publish = 0.0
        step = 0
        try:
            while args.max_steps <= 0 or step < args.max_steps:
                actor_state["step"] = step
                if pending_clip_idx["idx"] is not None:
                    clip_idx = int(pending_clip_idx["idx"])
                    pending_clip_idx["idx"] = None
                    _apply_sequence(clip_idx, reason="gui")
                elif motion_state is not None and selected_clip_idx["idx"] is not None:
                    env_id = int(stream_env_id["value"])
                    current_clip_idx = int(motion_state.motion_ids[env_id].detach().cpu())
                    if current_clip_idx != int(selected_clip_idx["idx"]):
                        _apply_sequence(int(selected_clip_idx["idx"]), reason="auto-reapply")
                actor_state = algo._pre_eval_env_step(actor_state)
                actor_state = algo.env_step(actor_state)
                actor_state = algo._post_eval_env_step(actor_state)

                now = time.monotonic()
                if now - last_publish >= min_period:
                    env.simulator.refresh_sim_tensors()
                    env_id = int(stream_env_id["value"])
                    current_clip_idx = None
                    if motion_state is not None and hasattr(motion_state, "motion_ids"):
                        current_clip_idx = int(motion_state.motion_ids[env_id].detach().cpu())
                    actual_object_path = _object_urdf_path_for_env(env, motion_state, env_id)
                    ref_object_path = _object_urdf_path_for_clip(motion_state, current_clip_idx) or actual_object_path
                    object_actual_root = _activate_object_visual(
                        "actual", actual_object_path, (1.0, 0.62, 0.12, 1.0)
                    )
                    object_ref_root = _activate_object_visual("ref", ref_object_path, (0.0, 0.85, 0.75, 0.35))

                    root_state = _tensor_row_to_numpy(env.simulator.robot_root_states, env_id)
                    dof_pos = _tensor_row_to_numpy(env.simulator.dof_pos[:, viser_to_sim], env_id)
                    robot_root.position = root_state[:3].astype(np.float32)
                    robot_root.wxyz = _xyzw_to_wxyz(root_state[3:7])
                    robot_viser.update_cfg(dof_pos.astype(np.float32))
                    if motion_ref_root is not None and motion_ref_viser is not None:
                        ref_robot_state = _reference_robot_state_wxyz(motion_state, env_id)
                        if ref_robot_state is not None:
                            ref_pos, ref_quat_wxyz = ref_robot_state
                            motion_ref_root.position = ref_pos
                            motion_ref_root.wxyz = ref_quat_wxyz
                            ref_dof_pos = _tensor_row_to_numpy(motion_state.joint_pos[:, viser_to_sim], env_id)
                            motion_ref_viser.update_cfg(ref_dof_pos.astype(np.float32))
                    if object_actual_root is not None:
                        object_state = _simulator_object_state_wxyz(env, motion_state, env_id)
                        if object_state is not None:
                            object_pos, object_quat_wxyz = object_state
                            object_actual_root.position = object_pos
                            object_actual_root.wxyz = object_quat_wxyz
                    if object_ref_root is not None:
                        object_ref_state = _reference_object_state_wxyz(motion_state, env_id)
                        if object_ref_state is not None:
                            object_ref_pos, object_ref_quat_wxyz = object_ref_state
                            object_ref_root.position = object_ref_pos
                            object_ref_root.wxyz = object_ref_quat_wxyz
                    if red_points_handle is not None:
                        red_points_handle.points = _red_point_hits(env, env_id)
                    if motion_ref_points_handle is not None:
                        motion_ref_points = _reference_body_points(motion_state, env_id)
                        if motion_ref_points is not None:
                            motion_ref_points_handle.points = motion_ref_points
                    _update_sequence_info()
                    last_publish = now

                if args.log_every > 0 and step % int(args.log_every) == 0:
                    env_id = int(stream_env_id["value"])
                    root_state = _tensor_row_to_numpy(env.simulator.robot_root_states, env_id)
                    msg = (
                        f"step={step} env={env_id} root_xyz=({root_state[0]:.3f}, "
                        f"{root_state[1]:.3f}, {root_state[2]:.3f})"
                    )
                    if motion_state is not None and hasattr(motion_state, "motion_ids"):
                        msg += f" clip={int(motion_state.motion_ids[env_id].detach().cpu())}"
                    if motion_state is not None and getattr(motion_state, "metrics", None):
                        metrics = motion_state.metrics
                        if "motion/error_ref_pos" in metrics:
                            err = float(metrics["motion/error_ref_pos"][env_id].detach().cpu())
                            msg += f" error_ref_pos={err:.4f}"
                        if "motion/error_body_pos" in metrics:
                            err = float(metrics["motion/error_body_pos"][env_id].detach().cpu())
                            msg += f" error_body_pos={err:.4f}"
                    logger.info(msg)
                step += 1
        finally:
            algo._post_evaluate_policy()
            logger.info("Viser rollout loop exited at step {}", step)
            server.stop()
    finally:
        if simulation_app:
            close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
