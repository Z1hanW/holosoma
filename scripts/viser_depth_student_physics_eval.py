#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from loguru import logger

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg
from holosoma.distill_depth_student import (
    _compute_depth,
    _get_actor_obs_group,
    get_actor_term_slices,
    select_student_lowdim_obs,
)
from holosoma.eval_depth_student_ablation import (
    _ablate_depth,
    _depth_args_from_checkpoint,
    _load_student,
    _maybe_build_teacher,
    _student_action,
)
from holosoma.train_agent import get_device
from holosoma.utils.eval_utils import init_eval_logging, init_sim_imports
from holosoma.utils.helpers import get_class
from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.sim_utils import close_simulation_app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "True-physics rollout evaluator for a depth-student checkpoint, streamed to Viser with "
            "the depth camera frustum drawn from the live RayCasterCamera pose."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Depth-student .pt checkpoint.")
    parser.add_argument("--teacher-checkpoint", default=None, help="Optional override for teacher/diagnostics.")
    parser.add_argument("--mode", choices=("normal", "zero", "shuffle", "teacher"), default="normal")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--port", type=int, default=2106)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until stopped.")
    parser.add_argument("--update-hz", type=float, default=30.0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-randomization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--red-points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Show height-scan red points as an explicit diagnostic. Defaults off because the depth student does "
            "not take heightmap/height_scan input."
        ),
    )
    parser.add_argument("--red-point-size", type=float, default=0.035)
    parser.add_argument(
        "--motion-ref",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Show the full target/reference G1 as an explicit diagnostic. Defaults off because the full ghost robot "
            "mesh is not a depth-student policy input."
        ),
    )
    parser.add_argument("--depth-hits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--depth-hit-size", type=float, default=0.018)
    parser.add_argument("--max-depth-hit-points", type=int, default=3500)
    parser.add_argument(
        "--frustum-scale",
        type=float,
        default=0.0,
        help="Viser frustum scale. <=0 scales the frustum image plane to the depth max range.",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--joystick",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Override the root_target_xy_yaw student input from a local gamepad/joystick.",
    )
    parser.add_argument("--joystick-device", type=int, default=0)
    parser.add_argument("--joystick-type", choices=("xbox",), default="xbox")
    parser.add_argument("--joystick-deadzone", type=float, default=0.08)
    parser.add_argument("--joystick-forward-scale", type=float, default=0.75)
    parser.add_argument("--joystick-lateral-scale", type=float, default=0.45)
    parser.add_argument("--joystick-yaw-scale", type=float, default=0.85)
    parser.add_argument(
        "--joystick-required",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail at startup if the requested joystick is not present. By default the evaluator keeps retrying.",
    )
    parser.add_argument(
        "--gui-command",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Expose root_target_xy_yaw controls in the Viser GUI and feed them to the student policy.",
    )
    parser.add_argument("--gui-command-x", type=float, default=0.0)
    parser.add_argument("--gui-command-y", type=float, default=0.0)
    parser.add_argument("--gui-command-yaw", type=float, default=0.0)
    parser.add_argument("--gui-command-x-range", type=float, default=1.5)
    parser.add_argument("--gui-command-y-range", type=float, default=1.0)
    parser.add_argument("--gui-command-yaw-range", type=float, default=1.6)
    parser.add_argument("--gui-command-step", type=float, default=0.05)
    return parser.parse_args()


def _resolve_holosoma_path(path: str) -> Path:
    if path.startswith("@holosoma/"):
        return Path(get_holosoma_root()) / path[len("@holosoma/") :]
    return Path(resolve_data_file_path(path))


def _robot_urdf_path(config: ExperimentConfig) -> Path:
    asset_root = _resolve_holosoma_path(config.robot.asset.asset_root)
    return asset_root / config.robot.asset.urdf_file


class _JoystickRootTarget:
    """Maps a local Xbox-style gamepad to root_target_xy_yaw policy input."""

    def __init__(self, args: argparse.Namespace):
        self.device_id = int(args.joystick_device)
        self.deadzone = float(args.joystick_deadzone)
        self.forward_scale = float(args.joystick_forward_scale)
        self.lateral_scale = float(args.joystick_lateral_scale)
        self.yaw_scale = float(args.joystick_yaw_scale)
        self.required = bool(args.joystick_required)
        self.joystick = None
        self.last_command = (0.0, 0.0, 0.0)
        self._next_retry = 0.0
        self._warned_missing = False

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame

        self.pygame = pygame
        self.pygame.init()
        try:
            self.pygame.display.init()
            if self.pygame.display.get_surface() is None:
                self.pygame.display.set_mode((1, 1))
        except Exception as exc:
            logger.debug("Pygame display init failed for joystick polling: {}", exc)
        self.pygame.joystick.init()
        self._try_open(required=self.required)

    def _try_open(self, required: bool = False) -> bool:
        if self.joystick is not None:
            return True
        now = time.monotonic()
        if not required and now < self._next_retry:
            return False
        self._next_retry = now + 2.0
        self.pygame.joystick.quit()
        self.pygame.joystick.init()
        count = self.pygame.joystick.get_count()
        if count <= self.device_id:
            msg = f"Joystick device {self.device_id} requested, but only {count} joystick(s) detected."
            if required:
                raise RuntimeError(msg)
            if not self._warned_missing:
                logger.warning("{} Root-target command remains zero; hotplug is retried while running.", msg)
                self._warned_missing = True
            return False

        self.joystick = self.pygame.joystick.Joystick(self.device_id)
        self.joystick.init()
        logger.info(
            "Joystick initialized: device={} name='{}' axes={} buttons={}",
            self.device_id,
            self.joystick.get_name(),
            self.joystick.get_numaxes(),
            self.joystick.get_numbuttons(),
        )
        return True

    def _axis(self, axis_id: int) -> float:
        if self.joystick is None or axis_id >= self.joystick.get_numaxes():
            return 0.0
        value = float(self.joystick.get_axis(axis_id))
        if abs(value) <= self.deadzone:
            return 0.0
        return math.copysign((abs(value) - self.deadzone) / max(1.0 - self.deadzone, 1.0e-6), value)

    def command_tensor(self, torch_module: Any, num_envs: int, device: Any, dtype: Any) -> Any:
        if not self._try_open(required=False):
            return torch_module.zeros((num_envs, 3), device=device, dtype=dtype)
        try:
            self.pygame.event.pump()
        except Exception:
            pass

        # Linux Xbox SDL mapping: left stick X/Y are axes 0/1, right stick X is axis 3.
        lx = self._axis(0)
        ly = -self._axis(1)
        rx = self._axis(3)
        command = torch_module.empty((num_envs, 3), device=device, dtype=dtype)
        command[:, 0] = self.forward_scale * ly
        command[:, 1] = -self.lateral_scale * lx
        command[:, 2] = -self.yaw_scale * rx
        self.last_command = (float(command[0, 0].item()), float(command[0, 1].item()), float(command[0, 2].item()))
        return command


class _GuiRootTarget:
    """Viser GUI controls for root_target_xy_yaw policy input."""

    def __init__(self, server: Any, args: argparse.Namespace):
        x_range = abs(float(args.gui_command_x_range))
        y_range = abs(float(args.gui_command_y_range))
        yaw_range = abs(float(args.gui_command_yaw_range))
        step = max(float(args.gui_command_step), 1.0e-4)
        with server.gui.add_folder("Command", order=-95.0):
            self.x = server.gui.add_slider(
                "Root target x (m)",
                min=-x_range,
                max=x_range,
                step=step,
                initial_value=float(np.clip(args.gui_command_x, -x_range, x_range)),
            )
            self.y = server.gui.add_slider(
                "Root target y (m)",
                min=-y_range,
                max=y_range,
                step=step,
                initial_value=float(np.clip(args.gui_command_y, -y_range, y_range)),
            )
            self.yaw = server.gui.add_slider(
                "Root target yaw (rad)",
                min=-yaw_range,
                max=yaw_range,
                step=step,
                initial_value=float(np.clip(args.gui_command_yaw, -yaw_range, yaw_range)),
            )
            zero_button = server.gui.add_button("Zero command")

        self.last_command = (float(self.x.value), float(self.y.value), float(self.yaw.value))

        @zero_button.on_click
        def _(_evt) -> None:
            self.x.value = 0.0
            self.y.value = 0.0
            self.yaw.value = 0.0

    def command_tensor(self, torch_module: Any, num_envs: int, device: Any, dtype: Any) -> Any:
        x = float(self.x.value)
        y = float(self.y.value)
        yaw = float(self.yaw.value)
        command = torch_module.empty((num_envs, 3), device=device, dtype=dtype)
        command[:, 0] = x
        command[:, 1] = y
        command[:, 2] = yaw
        self.last_command = (x, y, yaw)
        return command


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
    return dataclasses.replace(
        config,
        randomization=RandomizationManagerCfg(
            setup_terms=setup_terms,
            reset_terms=reset_terms,
            step_terms=step_terms,
            ignore_unsupported=config.randomization.ignore_unsupported,
        ),
    )


def _make_eval_config(checkpoint: dict[str, Any], args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig(**checkpoint["experiment_config"])
    ckpt_args = checkpoint.get("args", {})
    depth_cfg = config.simulator.config.depth_camera
    depth_cfg = dataclasses.replace(
        depth_cfg,
        enabled=True,
        min_range=float(ckpt_args.get("depth_min_range", depth_cfg.min_range)),
        max_range=float(ckpt_args.get("depth_max_range", depth_cfg.max_range)),
        width=int(ckpt_args.get("raw_depth_width", depth_cfg.width)),
        height=int(ckpt_args.get("raw_depth_height", depth_cfg.height)),
        horizontal_fov_deg=float(ckpt_args.get("depth_horizontal_fov_deg", depth_cfg.horizontal_fov_deg)),
        body_name=ckpt_args.get("depth_camera_body_name", depth_cfg.body_name),
        resize_mode=ckpt_args.get("depth_resize_mode", depth_cfg.resize_mode),
        randomize_placement=bool(ckpt_args.get("depth_camera_randomize_placement", depth_cfg.randomize_placement)),
        min_translation=list(ckpt_args.get("depth_camera_min_translation", depth_cfg.min_translation)),
        max_translation=list(ckpt_args.get("depth_camera_max_translation", depth_cfg.max_translation)),
        min_euler_rotation_deg=list(ckpt_args.get("depth_camera_min_rpy_deg", depth_cfg.min_euler_rotation_deg)),
        max_euler_rotation_deg=list(ckpt_args.get("depth_camera_max_rpy_deg", depth_cfg.max_euler_rotation_deg)),
        enable_self_occlusion=bool(ckpt_args.get("depth_camera_self_occlusion", depth_cfg.enable_self_occlusion)),
        latency_frame_min=int(ckpt_args.get("depth_latency_frame_min", depth_cfg.latency_frame_min)),
        latency_frame_max=int(ckpt_args.get("depth_latency_frame_max", depth_cfg.latency_frame_max)),
        buffer_len=int(ckpt_args.get("depth_buffer_len", depth_cfg.buffer_len)),
        enable_sensor_noise=bool(ckpt_args.get("depth_sensor_noise", depth_cfg.enable_sensor_noise)),
        pixel_std_dev_multiplier=float(
            ckpt_args.get("depth_pixel_std_dev_multiplier", depth_cfg.pixel_std_dev_multiplier)
        ),
        pixel_dropout_prob=float(ckpt_args.get("depth_pixel_dropout_prob", depth_cfg.pixel_dropout_prob)),
    )
    simulator_config = dataclasses.replace(config.simulator.config, depth_camera=depth_cfg)
    config = dataclasses.replace(
        config,
        simulator=dataclasses.replace(config.simulator, config=simulator_config),
        training=dataclasses.replace(
            config.training,
            headless=bool(args.headless),
            num_envs=max(int(args.num_envs), int(args.env_id) + 1, 1),
            export_onnx=False,
            seed=config.training.seed if args.seed is None else int(args.seed),
        ),
    )
    if args.disable_randomization:
        config = _disable_randomization(config)
    return config


def _load_mesh_for_viser(mesh: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh or Scene, got {type(mesh)}")
    return mesh


def _tensor_row_to_numpy(value: Any, env_id: int) -> np.ndarray:
    return value[env_id].detach().cpu().numpy()


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1.0e-12)
    w, x, y, z = q
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(matrix)))
        if idx == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
    q = np.asarray([w, x, y, z], dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1.0e-12)
    return q.astype(np.float32)


def _camera_world_wxyz_to_viser_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert IsaacLab camera world convention (+X forward, +Z up) to Viser/OpenCV."""
    try:
        import torch
        from isaaclab.utils.math import convert_camera_frame_orientation_convention

        q = torch.as_tensor(quat_wxyz, dtype=torch.float32).reshape(1, 4)
        q_ros = convert_camera_frame_orientation_convention(q, origin="world", target="ros")[0]
        return q_ros.detach().cpu().numpy().astype(np.float32)
    except Exception:
        rot_x_90 = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        rot_y_neg_90 = np.asarray([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        opengl_to_ros = np.diag([1.0, -1.0, -1.0])
        basis_change = rot_x_90 @ rot_y_neg_90 @ opengl_to_ros
        return _matrix_to_quat_wxyz(_quat_wxyz_to_matrix(quat_wxyz) @ basis_change)


def _vertical_fov_from_depth_cfg(depth_cfg: Any) -> tuple[float, float]:
    width = int(depth_cfg.width)
    height = int(depth_cfg.height)
    aspect = float(width) / max(float(height), 1.0)
    h_fov = math.radians(float(depth_cfg.horizontal_fov_deg))
    v_fov = 2.0 * math.atan(math.tan(h_fov / 2.0) / max(aspect, 1.0e-6))
    return v_fov, aspect


def _depth_image_rgb(sensor: Any, env_id: int, depth_cfg: Any) -> np.ndarray:
    output = getattr(sensor.data, "output", {})
    depth = output.get("distance_to_image_plane")
    if depth is None:
        return np.zeros((int(depth_cfg.height), int(depth_cfg.width), 3), dtype=np.uint8)
    depth_np = _tensor_row_to_numpy(depth, env_id)
    if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
        depth_np = depth_np[..., 0]
    depth_np = np.nan_to_num(depth_np, nan=float(depth_cfg.max_range), posinf=float(depth_cfg.max_range))
    alpha = (np.clip(depth_np, float(depth_cfg.min_range), float(depth_cfg.max_range)) - float(depth_cfg.min_range)) / max(
        float(depth_cfg.max_range) - float(depth_cfg.min_range), 1.0e-6
    )
    gray = np.asarray((1.0 - alpha) * 255.0, dtype=np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def _policy_depth_image_rgb(depth: Any | None, env_id: int) -> np.ndarray | None:
    if depth is None:
        return None
    depth_np = depth[env_id].detach().cpu().numpy()
    if depth_np.ndim == 3 and depth_np.shape[0] == 1:
        depth_np = depth_np[0]
    if depth_np.ndim != 2:
        return None
    alpha = np.clip(depth_np + 0.5, 0.0, 1.0)
    gray = np.asarray((1.0 - alpha) * 255.0, dtype=np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def _red_point_hits(env: Any, env_id: int) -> np.ndarray:
    sensors = getattr(getattr(env.simulator, "scene", None), "sensors", {})
    sensor = sensors.get("height_scanner")
    if sensor is not None and hasattr(sensor, "data") and hasattr(sensor.data, "ray_hits_w"):
        points = sensor.data.ray_hits_w[env_id].detach().cpu().numpy()
    else:
        terrain_state = env.terrain_manager.get_state("locomotion_terrain")
        points = terrain_state._ray_hits_world_base[env_id].detach().cpu().numpy()
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return points[np.isfinite(points).all(axis=1)]


def _depth_hit_points(env: Any, sensor: Any, env_id: int, max_points: int) -> np.ndarray:
    renderer = getattr(env, "_far_tracking_warp_depth_camera", None)
    points = getattr(renderer, "ray_hits_w", None)
    if points is None:
        points = getattr(sensor, "ray_hits_w", None)
    if points is None:
        return np.zeros((0, 3), dtype=np.float32)
    points_np = _tensor_row_to_numpy(points, env_id).astype(np.float32).reshape(-1, 3)
    points_np = points_np[np.isfinite(points_np).all(axis=1)]
    if max_points > 0 and points_np.shape[0] > max_points:
        stride = int(math.ceil(points_np.shape[0] / float(max_points)))
        points_np = points_np[::stride]
    return points_np


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


def _camera_sensor(env: Any, depth_cfg: Any) -> Any:
    sensors = getattr(getattr(env.simulator, "scene", None), "sensors", {})
    sensor = sensors.get(depth_cfg.sensor_name)
    if sensor is None:
        raise RuntimeError(f"Depth camera sensor '{depth_cfg.sensor_name}' is not available in the scene.")
    return sensor


def _update_depth_camera_visuals(
    env: Any,
    sensor: Any,
    env_id: int,
    depth_cfg: Any,
    frustum_handle: Any,
    camera_frame_handle: Any,
    depth_gui_handle: Any,
    depth_hits_handle: Any | None,
    args: argparse.Namespace,
    policy_depth: Any | None,
) -> None:
    data = sensor.data
    position = _tensor_row_to_numpy(data.pos_w, env_id).astype(np.float32)
    quat_world = _tensor_row_to_numpy(data.quat_w_world, env_id).astype(np.float32)
    quat_viser = _camera_world_wxyz_to_viser_wxyz(quat_world)
    image = _policy_depth_image_rgb(policy_depth, env_id)
    if image is None:
        image = _depth_image_rgb(sensor, env_id, depth_cfg)

    frustum_handle.position = position
    frustum_handle.wxyz = quat_viser
    frustum_handle.image = image
    depth_gui_handle.image = image
    camera_frame_handle.position = position
    camera_frame_handle.wxyz = quat_viser
    if depth_hits_handle is not None:
        depth_hits_handle.points = _depth_hit_points(env, sensor, env_id, int(args.max_depth_hit_points))


def main() -> None:
    args = _parse_args()
    init_eval_logging()
    logging.getLogger("trimesh").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)

    torch = None
    simulation_app = None
    try:
        import torch as _torch

        torch = _torch
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        config = _make_eval_config(checkpoint, args)
        simulation_app = init_sim_imports(config)
        device = get_device(config.training, distributed_conf=None)

        from holosoma.utils.common import seeding

        seeding(config.training.seed, torch_deterministic=config.training.torch_deterministic)
        env = get_class(config.env_class)(get_tyro_env_config(config), device=device)
        term_slices = get_actor_term_slices(env, "actor_obs")
        model, training_mode = _load_student(checkpoint, device)
        depth_args = _depth_args_from_checkpoint(checkpoint.get("args", {}))
        command_override_enabled = bool(args.gui_command or args.joystick)
        if command_override_enabled and getattr(depth_args, "student_command_mode", "root_xy_yaw") != "root_xy_yaw":
            raise RuntimeError("Command override requires a root_xy_yaw depth-student checkpoint.")
        joystick_root_target = _JoystickRootTarget(args) if args.joystick and not args.gui_command else None
        teacher_policy = _maybe_build_teacher(env, checkpoint, args, device)
        if args.mode == "teacher" and teacher_policy is None:
            raise RuntimeError("Mode 'teacher' requires a valid teacher checkpoint.")

        import viser
        from viser.extras import ViserUrdf

        server = viser.ViserServer(host="0.0.0.0", port=int(args.port), label="depth_student_physics_eval")
        gui_root_target = _GuiRootTarget(server, args) if args.gui_command else None
        depth_cfg = config.simulator.config.depth_camera
        v_fov, aspect = _vertical_fov_from_depth_cfg(depth_cfg)

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

        urdf_path = _robot_urdf_path(config)
        robot_viser = ViserUrdf(server, urdf_path, root_node_name="/robot", load_meshes=True, load_collision_meshes=False)
        robot_root = robot_viser._visual_root_frame
        if robot_root is None:
            raise RuntimeError("ViserUrdf did not create a robot visual root frame.")

        viser_joint_names = list(robot_viser.get_actuated_joint_names())
        dof_name_to_idx = {name: i for i, name in enumerate(env.dof_names)}
        missing = [name for name in viser_joint_names if name not in dof_name_to_idx]
        if missing:
            raise RuntimeError(f"Viser joints missing from simulator DOFs: {missing}")
        viser_to_sim = torch.tensor([dof_name_to_idx[name] for name in viser_joint_names], device=device, dtype=torch.long)

        motion_state = env.command_manager.get_state("motion_command")
        motion_ref_viser = None
        motion_ref_root = None
        motion_ref_points_handle = None
        if args.motion_ref and motion_state is not None:
            motion_ref_viser = ViserUrdf(
                server,
                urdf_path,
                root_node_name="/motion_ref",
                mesh_color_override=(0.1, 0.45, 1.0, 0.32),
                load_meshes=True,
                load_collision_meshes=False,
            )
            motion_ref_root = motion_ref_viser._visual_root_frame
            motion_ref_points = _reference_body_points(motion_state, int(args.env_id))
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

        red_points_handle = None
        if args.red_points:
            red_points_handle = server.scene.add_point_cloud(
                "/height_scan_red_points",
                np.zeros((0, 3), dtype=np.float32),
                colors=(255, 0, 0),
                point_size=float(args.red_point_size),
                point_shape="circle",
                point_shading="flat",
                precision="float32",
            )

        depth_sensor = _camera_sensor(env, depth_cfg)
        initial_image = np.zeros((int(depth_cfg.height), int(depth_cfg.width), 3), dtype=np.uint8)
        frustum_handle = server.scene.add_camera_frustum(
            "/depth_camera/frustum",
            fov=float(v_fov),
            aspect=float(aspect),
            scale=1.0,
            line_width=2.5,
            color=(0, 255, 255),
            image=initial_image,
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            variant="wireframe",
        )
        if float(args.frustum_scale) > 0.0:
            frustum_handle.scale = float(args.frustum_scale)
        else:
            frustum_handle.scale = float(depth_cfg.max_range) / max(
                float(frustum_handle.compute_canonical_frustum_size()[2]), 1.0e-6
            )
        camera_frame_handle = server.scene.add_frame(
            "/depth_camera/frame",
            axes_length=0.18,
            axes_radius=0.008,
            origin_radius=0.018,
        )
        depth_hits_handle = None
        if args.depth_hits:
            depth_hits_handle = server.scene.add_point_cloud(
                "/depth_camera/hits",
                np.zeros((0, 3), dtype=np.float32),
                colors=(0, 255, 180),
                point_size=float(args.depth_hit_size),
                point_shape="circle",
                point_shading="flat",
                precision="float32",
            )

        with server.gui.add_folder("Depth Camera", order=-100.0):
            depth_gui_handle = server.gui.add_image(
                initial_image,
                label="Live depth",
                format="jpeg",
                jpeg_quality=80,
            )
            server.gui.add_markdown(
                "`Depth Camera` is the student visual input. Height-scan/red-point diagnostics are hidden by default."
            )

        with server.gui.add_folder("Rollout", order=-90.0):
            status = server.gui.add_markdown("")

        logger.info("Viser listening on http://localhost:{}", args.port)
        logger.info(
            "Depth student eval: checkpoint={} mode={} envs={} stream_env={} depth_sensor={} raw={}x{} hfov={:.2f} max_range={:.2f}",
            args.checkpoint,
            args.mode,
            env.num_envs,
            args.env_id,
            depth_cfg.sensor_name,
            depth_cfg.width,
            depth_cfg.height,
            depth_cfg.horizontal_fov_deg,
            depth_cfg.max_range,
        )

        obs_dict = env.reset_all()
        reward_sum = torch.zeros(env.num_envs, device=device)
        episode_returns: list[float] = []
        reward_step_sum = 0.0
        done_sum = 0.0
        timeout_sum = 0.0
        non_timeout_done_sum = 0.0
        action_l1_sum = 0.0
        action_l1_count = 0
        min_period = 0.0 if args.update_hz <= 0 else 1.0 / float(args.update_hz)
        last_publish = 0.0
        step = 0

        with torch.no_grad():
            while args.max_steps <= 0 or step < int(args.max_steps):
                actor_obs = _get_actor_obs_group(obs_dict).to(device=device, dtype=torch.float)
                depth = _compute_depth(env, depth_args, device)
                if args.mode == "teacher":
                    assert teacher_policy is not None
                    actions = teacher_policy(obs_dict).to(device=device, dtype=torch.float)
                else:
                    lowdim_obs = select_student_lowdim_obs(env, actor_obs, term_slices, depth_args)
                    root_target_source = gui_root_target or joystick_root_target
                    if root_target_source is not None:
                        lowdim_obs = lowdim_obs.clone()
                        lowdim_obs[:, :3] = root_target_source.command_tensor(
                            torch,
                            env.num_envs,
                            lowdim_obs.device,
                            lowdim_obs.dtype,
                        )
                    depth_for_policy = _ablate_depth(depth, args.mode)
                    actions = _student_action(model, training_mode, lowdim_obs, depth_for_policy)
                if teacher_policy is not None:
                    teacher_actions = teacher_policy(obs_dict).to(device=device, dtype=torch.float)
                    action_l1_sum += float((actions - teacher_actions).abs().mean().item())
                    action_l1_count += 1

                obs_dict, rewards, dones, extras = env.step({"actions": actions})
                rewards = rewards.detach().float().view(-1)
                dones = dones.detach().bool().view(-1)
                timeouts = extras.get("time_outs")
                if timeouts is None:
                    timeouts = torch.zeros_like(dones)
                else:
                    timeouts = timeouts.detach().bool().view(-1)

                reward_step_sum += float(rewards.mean().item())
                reward_sum += rewards
                done_sum += float(dones.float().mean().item())
                timeout_sum += float((dones & timeouts).float().mean().item())
                non_timeout_done_sum += float((dones & ~timeouts).float().mean().item())
                if dones.any():
                    episode_returns.extend(float(v) for v in reward_sum[dones].detach().cpu())
                    reward_sum[dones] = 0.0

                now = time.monotonic()
                if now - last_publish >= min_period:
                    env.simulator.refresh_sim_tensors()
                    env_id = int(args.env_id)
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
                    if motion_ref_points_handle is not None:
                        motion_ref_points = _reference_body_points(motion_state, env_id)
                        if motion_ref_points is not None:
                            motion_ref_points_handle.points = motion_ref_points
                    if red_points_handle is not None:
                        red_points_handle.points = _red_point_hits(env, env_id)
                    _update_depth_camera_visuals(
                        env,
                        depth_sensor,
                        env_id,
                        depth_cfg,
                        frustum_handle,
                        camera_frame_handle,
                        depth_gui_handle,
                        depth_hits_handle,
                        args,
                        depth,
                    )

                    episode_return_mean = float(sum(episode_returns) / max(len(episode_returns), 1))
                    command_status = ""
                    root_target_source = gui_root_target or joystick_root_target
                    if root_target_source is not None:
                        cmd_x, cmd_y, cmd_yaw = root_target_source.last_command
                        command_source_name = "gui_root" if gui_root_target is not None else "joystick_root"
                        command_status = f" | {command_source_name}: `[{cmd_x:.2f}, {cmd_y:.2f}, {cmd_yaw:.2f}]`"
                    status.content = (
                        f"mode: `{args.mode}` | step: `{step}` | env: `{env_id}` | "
                        f"reward: `{reward_step_sum / max(step + 1, 1):.5f}` | "
                        f"episode_return: `{episode_return_mean:.3f}` | "
                        f"done: `{done_sum / max(step + 1, 1):.5f}` | "
                        f"non_timeout_done: `{non_timeout_done_sum / max(step + 1, 1):.5f}` | "
                        f"teacher_l1: `{action_l1_sum / max(action_l1_count, 1):.5f}`"
                        f"{command_status}"
                    )
                    last_publish = now

                if args.log_every > 0 and step % int(args.log_every) == 0:
                    episode_return_mean = float(sum(episode_returns) / max(len(episode_returns), 1))
                    logger.info(
                        "step={} mode={} reward={:.6f} episode_return={:.4f} done={:.5f} timeout={:.5f} "
                        "non_timeout_done={:.5f} teacher_l1={:.5f}",
                        step,
                        args.mode,
                        reward_step_sum / max(step + 1, 1),
                        episode_return_mean,
                        done_sum / max(step + 1, 1),
                        timeout_sum / max(step + 1, 1),
                        non_timeout_done_sum / max(step + 1, 1),
                        action_l1_sum / max(action_l1_count, 1),
                    )
                step += 1

        server.stop()
        logger.info("Depth student Viser eval finished at step {}", step)
    except Exception as exc:
        logger.error(f"Depth student Viser eval failed: {exc}")
        traceback.print_exc()
        raise
    finally:
        if simulation_app:
            close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
