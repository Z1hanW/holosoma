from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from loguru import logger

# Ensure local packages are importable when running from source.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "holosoma"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

VISER_SRC = REPO_ROOT / "viser" / "src"
if VISER_SRC.exists() and str(VISER_SRC) not in sys.path:
    sys.path.insert(0, str(VISER_SRC))

import viser  # type: ignore[import-not-found]  # noqa: E402
from viser.extras import ViserUrdf  # type: ignore[import-not-found]  # noqa: E402

from holosoma.config_types.experiment import ExperimentConfig  # noqa: E402
from holosoma.config_types.robot import RobotConfig  # noqa: E402
from holosoma.perception import apply_perception_overrides  # noqa: E402
from holosoma.utils.config_utils import CONFIG_NAME  # noqa: E402
from holosoma.utils.eval_utils import (  # noqa: E402
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp  # noqa: E402
from holosoma.utils.helpers import get_class  # noqa: E402
from holosoma.utils.module_utils import get_holosoma_root  # noqa: E402
from holosoma.utils.path import resolve_data_file_path  # noqa: E402
from holosoma.utils.safe_torch_import import torch  # noqa: E402
from holosoma.utils.sim_utils import (  # noqa: E402
    close_simulation_app,
    setup_simulation_environment,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402


@dataclass(frozen=True)
class ViserLiveConfig:
    port: int = 6060
    env_index: int = 0
    update_interval: int = 1
    show_meshes: bool = True
    add_grid: bool = True
    grid_size: float = 10.0
    auto_reapply_clip: bool = True


def _resolve_data_path(path: str) -> str:
    if path.startswith("@holosoma/"):
        return str(Path(get_holosoma_root()) / path[len("@holosoma/") :])
    return resolve_data_file_path(path)


def _resolve_robot_urdf_path(robot_config: RobotConfig) -> str:
    asset_root = _resolve_data_path(robot_config.asset.asset_root)
    urdf_path = os.path.join(asset_root, robot_config.asset.urdf_file)
    return _resolve_data_path(urdf_path)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _current_clip_name(motion_cmd, env_index: int) -> str | None:
    try:
        clip_ids = motion_cmd.motion.clip_ids
        clip_idx = int(motion_cmd.clip_ids[env_index].item())
        if clip_idx < 0 or clip_idx >= len(clip_ids):
            return None
        return str(clip_ids[clip_idx])
    except Exception:
        return None


class ViserLiveViewer:
    def __init__(self, robot_config: RobotConfig, cfg: ViserLiveConfig) -> None:
        self.server = viser.ViserServer(port=cfg.port)
        self.robot_root = self.server.scene.add_frame("/robot", show_axes=False)
        self.object_root = self.server.scene.add_frame("/object", show_axes=False)

        urdf_path = _resolve_robot_urdf_path(robot_config)
        self.robot = ViserUrdf(self.server, urdf_or_path=urdf_path, root_node_name="/robot")
        self.robot.show_visual = cfg.show_meshes

        self.object = None
        if robot_config.object.object_urdf_path:
            object_path = _resolve_data_path(robot_config.object.object_urdf_path)
            self.object = ViserUrdf(self.server, urdf_or_path=object_path, root_node_name="/object")
            self.object.show_visual = cfg.show_meshes

        viser_joint_names = list(self.robot.get_actuated_joint_names())
        name_to_robot_idx = {name: idx for idx, name in enumerate(robot_config.dof_names)}
        missing = [name for name in viser_joint_names if name not in name_to_robot_idx]
        if missing:
            raise ValueError(f"Viser joints not found in robot config: {missing}")
        self.joint_order = [name_to_robot_idx[name] for name in viser_joint_names]

        if cfg.add_grid:
            self.server.scene.add_grid(
                "/grid",
                width=cfg.grid_size,
                height=cfg.grid_size,
                position=(0.0, 0.0, 0.0),
            )

        with self.server.gui.add_folder("Display"):
            show_meshes_cb = self.server.gui.add_checkbox("Show meshes", initial_value=cfg.show_meshes)

        @show_meshes_cb.on_update
        def _(_evt) -> None:
            visible = bool(show_meshes_cb.value)
            self.robot.show_visual = visible
            if self.object is not None:
                self.object.show_visual = visible

        logger.info(f"Viser server running on port {cfg.port}")

    def _get_object_state_wxyz(self, env, env_index: int) -> tuple[np.ndarray, np.ndarray] | None:
        if self.object is None:
            return None
        sim = env.simulator
        env_ids = torch.tensor([env_index], device=env.device, dtype=torch.long)
        states = None
        if hasattr(sim, "_get_object_states"):
            try:
                states = sim._get_object_states("object", env_ids)
            except Exception:
                states = None
        if states is None and hasattr(sim, "get_actor_states") and getattr(sim, "has_scene_objects", False):
            try:
                states = sim.get_actor_states(["object"], env_ids)
            except Exception:
                states = None
        if states is None or states.numel() == 0:
            return None
        state = states[0]
        pos = _to_numpy(state[0:3])
        quat_xyzw = _to_numpy(state[3:7])
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        return pos, quat_wxyz

    def update_from_env(self, env, env_index: int) -> None:
        root_state = _to_numpy(env.simulator.robot_root_states[env_index])
        dof_pos = _to_numpy(env.simulator.dof_pos[env_index]).reshape(-1)

        root_pos = root_state[:3]
        root_quat_xyzw = root_state[3:7]
        root_quat_wxyz = root_quat_xyzw[[3, 0, 1, 2]]

        self.robot_root.position = root_pos
        self.robot_root.wxyz = root_quat_wxyz
        self.robot.update_cfg(dof_pos[self.joint_order])

        obj_state = self._get_object_state_wxyz(env, env_index)
        if obj_state is not None:
            obj_pos, obj_quat_wxyz = obj_state
            self.object_root.position = obj_pos
            self.object_root.wxyz = obj_quat_wxyz


def _force_clip(env, motion_cmd, clip_idx: int, env_index: int) -> dict[str, torch.Tensor]:
    env_ids = torch.tensor([env_index], device=env.device, dtype=torch.long)
    env.reset_envs_idx(env_ids)

    motion_cmd.clip_ids[env_ids] = int(clip_idx)
    motion_cmd.time_steps[env_ids] = 0
    if motion_cmd.motion_cfg.align_motion_to_init_yaw:
        motion_cmd._update_motion_alignment(env_ids)

    root_pos = motion_cmd.body_pos_w[env_ids, 0].clone()
    root_rot = motion_cmd.body_quat_w[env_ids, 0].clone()
    root_lin_vel = motion_cmd.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = motion_cmd.body_ang_vel_w[env_ids, 0].clone()
    dof_pos = motion_cmd.joint_pos[env_ids].clone()
    dof_vel = motion_cmd.joint_vel[env_ids].clone()

    sim = env.simulator
    sim.dof_pos[env_ids] = dof_pos
    sim.dof_vel[env_ids] = dof_vel
    sim.robot_root_states[env_ids, :3] = root_pos
    sim.robot_root_states[env_ids, 3:7] = root_rot
    sim.robot_root_states[env_ids, 7:10] = root_lin_vel
    sim.robot_root_states[env_ids, 10:13] = root_ang_vel

    if hasattr(sim, "set_actor_root_state_tensor_robots"):
        sim.set_actor_root_state_tensor_robots(env_ids, sim.robot_root_states)
    else:
        sim.set_actor_root_state_tensor(env_ids, sim.all_root_states)

    if hasattr(sim, "set_dof_state_tensor_robots"):
        sim.set_dof_state_tensor_robots(env_ids, sim.dof_state)
    else:
        sim.set_dof_state_tensor(env_ids, sim.dof_state)

    if motion_cmd.motion.has_object:
        obj_pos = motion_cmd.object_pos_w[env_ids]
        obj_ori = motion_cmd.object_quat_w[env_ids]
        obj_lin_vel = motion_cmd.object_lin_vel_w[env_ids]
        obj_states = torch.cat([obj_pos, obj_ori, obj_lin_vel, torch.zeros_like(obj_lin_vel)], dim=-1)
        sim.set_actor_states([motion_cmd.object_name], env_ids, obj_states)

    if hasattr(sim, "scene") and hasattr(sim.scene, "write_data_to_sim"):
        sim.scene.write_data_to_sim()
    sim.refresh_sim_tensors()

    motion_cmd._update_future_target_poses()
    if hasattr(env, "_refresh_envs_after_reset"):
        env._refresh_envs_after_reset(env_ids)

    env.reset_buf[env_ids] = 0
    env.time_out_buf[env_ids] = 0
    env._compute_observations()
    env._post_compute_observations_callback()
    env._clip_observations()
    return env.obs_buf_dict


def _build_actor_obs(algo, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([obs_dict[k] for k in algo.actor_obs_keys], dim=1)


def run_eval_with_viser_clip(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
    viser_cfg: ViserLiveConfig,
) -> None:
    tyro_config = apply_perception_overrides(tyro_config)
    env, device, simulation_app = setup_simulation_environment(tyro_config)

    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    assert checkpoint_cfg.checkpoint is not None
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

    env.set_is_evaluating()
    obs_dict = env.reset_all()

    motion_cmd = env.command_manager.get_state("motion_command") if env.command_manager else None
    if motion_cmd is None or not hasattr(motion_cmd, "motion"):
        raise RuntimeError("motion_command is required for clip selection.")

    clip_names = list(motion_cmd.motion.clip_ids)
    if not clip_names:
        raise RuntimeError("No motion clips found in motion_command.")
    clip_name_to_idx = {name: idx for idx, name in enumerate(clip_names)}

    env_index = int(viser_cfg.env_index)
    if env_index < 0 or env_index >= getattr(env, "num_envs", 1):
        raise ValueError(f"env_index {env_index} is out of range for num_envs={getattr(env, 'num_envs', 1)}")

    viewer = ViserLiveViewer(tyro_config.robot, viser_cfg)
    pending_clip = {"name": None}

    with viewer.server.gui.add_folder("Motion"):
        clip_dropdown = viewer.server.gui.add_dropdown("Clip", options=tuple(clip_names), initial_value=clip_names[0])
        apply_btn = viewer.server.gui.add_button("Apply clip")
        clip_label = viewer.server.gui.add_markdown("")

    @clip_dropdown.on_update
    def _(_evt) -> None:
        pending_clip["name"] = str(clip_dropdown.value)

    @apply_btn.on_click
    def _(_evt) -> None:
        pending_clip["name"] = str(clip_dropdown.value)

    # Ensure initial clip matches the dropdown selection.
    obs_dict = _force_clip(env, motion_cmd, clip_name_to_idx[str(clip_dropdown.value)], env_index)

    policy = algo.get_inference_policy()
    step = 0

    while True:
        if pending_clip["name"]:
            name = pending_clip["name"]
            pending_clip["name"] = None
            if name in clip_name_to_idx:
                obs_dict = _force_clip(env, motion_cmd, clip_name_to_idx[name], env_index)

        actor_obs = _build_actor_obs(algo, obs_dict)
        actions = policy({"actor_obs": actor_obs})
        obs_dict, _, reset_buf, _ = env.step({"actions": actions})

        if step % max(1, int(viser_cfg.update_interval)) == 0:
            viewer.update_from_env(env, env_index)
            current_clip = _current_clip_name(motion_cmd, env_index) or "n/a"
            clip_label.content = f"Current clip: `{current_clip}`"

        if viser_cfg.auto_reapply_clip and bool(reset_buf[env_index].item()):
            name = str(clip_dropdown.value)
            if name in clip_name_to_idx:
                obs_dict = _force_clip(env, motion_cmd, clip_name_to_idx[name], env_index)

        step += 1

    if simulation_app:
        close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    eval_cfg_overrides, remaining_args = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        return_unknown_args=True,
        description="Overriding config on top of what's loaded.",
        config=TYRO_CONIFG,
    )
    viser_cfg = tyro.cli(
        ViserLiveConfig,
        args=remaining_args,
        description="Viser live evaluation configuration.",
    )
    run_eval_with_viser_clip(eval_cfg_overrides, checkpoint_cfg, saved_cfg, saved_wandb_path, viser_cfg)


if __name__ == "__main__":
    main()
