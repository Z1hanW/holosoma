"""Whole body tracking observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.rotations import (
    get_euler_xyz,
    quat_rotate_inverse,
    quaternion_to_matrix,
    subtract_frame_transforms,
    wrap_to_pi,
    yaw_quat,
)
from holosoma.utils.torch_utils import get_axis_params, to_torch

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


#########################################################################################################
## terms same to managers/observation/terms/locomotion.py
#########################################################################################################
def _base_quat(env: WholeBodyTrackingManager) -> torch.Tensor:
    return env.base_quat


def gravity_vector(env: WholeBodyTrackingManager, up_axis_idx: int = 2) -> torch.Tensor:
    axis = to_torch(get_axis_params(-1.0, up_axis_idx), device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def base_forward_vector(env: WholeBodyTrackingManager) -> torch.Tensor:
    axis = to_torch([1.0, 0.0, 0.0], device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def get_base_lin_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    root_states = env.simulator.robot_root_states
    lin_vel_world = root_states[:, 7:10]
    return quat_rotate_inverse(_base_quat(env), lin_vel_world, w_last=True)


def get_base_ang_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    return quat_rotate_inverse(_base_quat(env), ang_vel_world, w_last=True)


def get_projected_gravity(env: WholeBodyTrackingManager) -> torch.Tensor:
    return quat_rotate_inverse(_base_quat(env), gravity_vector(env), w_last=True)


def base_lin_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Base linear velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_lin_vel()
    """
    return get_base_lin_vel(env)


def base_ang_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Base angular velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_ang_vel()
    """
    return get_base_ang_vel(env)


def projected_gravity(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Gravity vector projected into base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_projected_gravity()
    """
    return get_projected_gravity(env)


def dof_pos(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Joint positions relative to default positions.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_pos()
    """
    return env.simulator.dof_pos - env.default_dof_pos


def dof_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Joint velocities.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_vel()
    """
    return env.simulator.dof_vel


def actions(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Last actions taken by the policy.

    Returns:
        Tensor of shape [num_envs, num_actions]

    Equivalent to:
        env._get_obs_actions()
    """
    return env.action_manager.action


def height_scan(
    env: WholeBodyTrackingManager,
    sensor_name: str = "height_scanner",
    offset: float = 0.5,
    miss_value: float = 0.0,
    clip_min: float = -100.0,
    clip_max: float = 100.0,
) -> torch.Tensor:
    """Height scan from an IsaacLab RayCaster sensor.

    Matches IsaacLab's height_scan convention:
    sensor_world_z - terrain_hit_world_z - offset.

    RayCaster returns non-finite hit positions for rays that miss finite meshes.
    Loaded OBJ terrains are finite, so sanitize misses before empirical
    normalization sees them.
    """
    sensors = getattr(getattr(env.simulator, "scene", None), "sensors", {})
    if sensor_name not in sensors:
        raise RuntimeError(
            f"Height scanner sensor '{sensor_name}' is not available. "
            "Enable it with --simulator.config.height-scanner.enabled=True."
        )

    sensor = sensors[sensor_name]
    heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    heights = torch.nan_to_num(heights, nan=miss_value, posinf=miss_value, neginf=miss_value)
    return heights.clamp(min=clip_min, max=clip_max)


def depth_camera(
    env: WholeBodyTrackingManager,
    sensor_name: str = "depth_camera",
    output_key: str = "distance_to_image_plane",
    min_range: float = 0.3,
    max_range: float = 2.0,
    resize_height: int = 58,
    resize_width: int = 87,
    resize_mode: str | None = None,
    enable_self_occlusion: bool | None = None,
    enable_sensor_noise: bool | None = None,
    pixel_std_dev_multiplier: float | None = None,
    pixel_dropout_prob: float | None = None,
    latency_frame_min: int | None = None,
    latency_frame_max: int | None = None,
    buffer_len: int | None = None,
    flatten: bool = True,
) -> torch.Tensor:
    """Normalized ray-caster pinhole depth image for visual distillation.

    The source is either IsaacLab's terrain-only ``RayCasterCamera`` or the local
    far-tracking-style Warp renderer when dynamic robot self-occlusion is enabled.
    Preprocessing follows far-tracking's ZED path: clamp, resize, optional
    depth-proportional noise/dropout, normalize to roughly [-0.5, 0.5], then
    optional latency buffering.
    """
    depth_cfg = getattr(getattr(env.simulator, "simulator_config", None), "depth_camera", None)
    if depth_cfg is not None:
        resize_mode = depth_cfg.resize_mode if resize_mode is None else resize_mode
        enable_self_occlusion = depth_cfg.enable_self_occlusion if enable_self_occlusion is None else enable_self_occlusion
        enable_sensor_noise = depth_cfg.enable_sensor_noise if enable_sensor_noise is None else enable_sensor_noise
        pixel_std_dev_multiplier = (
            depth_cfg.pixel_std_dev_multiplier if pixel_std_dev_multiplier is None else pixel_std_dev_multiplier
        )
        pixel_dropout_prob = depth_cfg.pixel_dropout_prob if pixel_dropout_prob is None else pixel_dropout_prob
        latency_frame_min = depth_cfg.latency_frame_min if latency_frame_min is None else latency_frame_min
        latency_frame_max = depth_cfg.latency_frame_max if latency_frame_max is None else latency_frame_max
        buffer_len = depth_cfg.buffer_len if buffer_len is None else buffer_len
    resize_mode = resize_mode or "bicubic"
    enable_self_occlusion = bool(enable_self_occlusion)
    enable_sensor_noise = bool(enable_sensor_noise)
    pixel_std_dev_multiplier = 0.1 if pixel_std_dev_multiplier is None else float(pixel_std_dev_multiplier)
    pixel_dropout_prob = 0.05 if pixel_dropout_prob is None else float(pixel_dropout_prob)
    latency_frame_min = 0 if latency_frame_min is None else int(latency_frame_min)
    latency_frame_max = 0 if latency_frame_max is None else int(latency_frame_max)
    buffer_len = 1 if buffer_len is None else int(buffer_len)

    if enable_self_occlusion:
        if depth_cfg is None:
            raise RuntimeError("Depth camera self-occlusion requires simulator.config.depth_camera.")
        renderer = getattr(env, "_far_tracking_warp_depth_camera", None)
        if renderer is None:
            from holosoma.utils.warp_depth_camera import FarTrackingWarpDepthCamera

            renderer = FarTrackingWarpDepthCamera(env, depth_cfg)
            env._far_tracking_warp_depth_camera = renderer
            _sync_depth_camera_sensor_offsets(env, renderer, sensor_name)
        depth = renderer.capture()
    else:
        sensors = getattr(getattr(env.simulator, "scene", None), "sensors", {})
        if sensor_name not in sensors:
            raise RuntimeError(
                f"Depth camera sensor '{sensor_name}' is not available. "
                "Enable it with --simulator.config.depth-camera.enabled=True."
            )

        sensor = sensors[sensor_name]
        output = sensor.data.output
        if output_key not in output:
            raise RuntimeError(f"Depth camera output '{output_key}' is not available. Available outputs: {list(output)}")
        depth = output[output_key]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise RuntimeError(f"Expected depth image shape [num_envs, height, width], got {tuple(depth.shape)}")

    depth = torch.nan_to_num(depth, nan=max_range, posinf=max_range, neginf=max_range)
    depth = depth.clamp(min=min_range, max=max_range).unsqueeze(1)
    if resize_height > 0 and resize_width > 0 and tuple(depth.shape[-2:]) != (resize_height, resize_width):
        if resize_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            depth = F.interpolate(depth, size=(resize_height, resize_width), mode=resize_mode, align_corners=False)
        else:
            depth = F.interpolate(depth, size=(resize_height, resize_width), mode=resize_mode)
    depth = depth.clamp(min=min_range, max=max_range)
    if enable_sensor_noise:
        depth = depth + torch.randn_like(depth) * pixel_std_dev_multiplier * depth
        if pixel_dropout_prob > 0.0:
            dropout_mask = torch.rand_like(depth) < pixel_dropout_prob
            dropout_fill = torch.rand_like(depth) * (max_range - min_range) + min_range
            depth = torch.where(dropout_mask, dropout_fill, depth)
    depth = (depth - min_range) / max(max_range - min_range, 1.0e-6) - 0.5
    depth = _apply_depth_latency(
        env,
        depth,
        latency_frame_min=latency_frame_min,
        latency_frame_max=latency_frame_max,
        buffer_len=buffer_len,
    )
    return depth.flatten(start_dim=1) if flatten else depth


def _sync_depth_camera_sensor_offsets(env: WholeBodyTrackingManager, renderer: object, sensor_name: str) -> None:
    """Keep the IsaacLab camera pose/frustum aligned with the dynamic Warp renderer."""
    sensors = getattr(getattr(env.simulator, "scene", None), "sensors", {})
    sensor = sensors.get(sensor_name)
    if sensor is None or not hasattr(sensor, "_offset_pos") or not hasattr(sensor, "_offset_quat"):
        return
    local_position = getattr(renderer, "local_position", None)
    local_orientation = getattr(renderer, "local_orientation", None)
    if local_position is None or local_orientation is None:
        return
    sensor._offset_pos[:].copy_(local_position)
    sensor._offset_quat[:].copy_(local_orientation[:, [3, 0, 1, 2]])
    if hasattr(sensor, "reset"):
        sensor.reset()


def _apply_depth_latency(
    env: WholeBodyTrackingManager,
    depth: torch.Tensor,
    *,
    latency_frame_min: int,
    latency_frame_max: int,
    buffer_len: int,
) -> torch.Tensor:
    if latency_frame_max <= 0 and latency_frame_min <= 0:
        return depth
    latency_frame_min = max(0, latency_frame_min)
    latency_frame_max = max(latency_frame_min, latency_frame_max)
    if buffer_len <= latency_frame_max:
        raise RuntimeError(
            f"Depth latency buffer_len ({buffer_len}) must be greater than latency_frame_max ({latency_frame_max})."
        )

    depth_no_channel = depth[:, 0]
    shape_key = (depth_no_channel.shape, depth_no_channel.device, depth_no_channel.dtype, buffer_len)
    buffer = getattr(env, "_depth_camera_latency_buffer", None)
    if buffer is None or getattr(env, "_depth_camera_latency_shape_key", None) != shape_key:
        buffer = depth_no_channel.unsqueeze(1).repeat(1, buffer_len, 1, 1).clone()
        env._depth_camera_latency_buffer = buffer
        env._depth_camera_latency_shape_key = shape_key
    else:
        buffer[:, :-1].copy_(buffer[:, 1:].clone())
        buffer[:, -1].copy_(depth_no_channel)

    reset_mask = getattr(env, "episode_length_buf", None)
    if reset_mask is not None:
        reset_ids = (reset_mask <= 0).nonzero(as_tuple=False).flatten()
        if len(reset_ids) > 0:
            buffer[reset_ids] = depth_no_channel[reset_ids].unsqueeze(1).repeat(1, buffer_len, 1, 1)

    if latency_frame_min == latency_frame_max:
        latency = torch.full((depth.shape[0],), latency_frame_min, device=depth.device, dtype=torch.long)
    else:
        latency = torch.randint(
            latency_frame_min,
            latency_frame_max + 1,
            (depth.shape[0],),
            device=depth.device,
            dtype=torch.long,
        )
    env_ids = torch.arange(depth.shape[0], device=depth.device)
    selected = buffer[env_ids, buffer_len - 1 - latency]
    return selected.unsqueeze(1)


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


def motion_command(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.command


def root_target_xy_yaw(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Root target command: target root xy in robot-yaw frame plus target yaw error."""
    motion_command = _get_motion_command_and_assert_type(env)
    target_pos_w = motion_command.root_pos_w
    target_quat_w = motion_command.root_quat_w
    robot_pos_w = motion_command.robot_root_pos_w
    robot_quat_w = motion_command.robot_root_quat_w

    delta_pos_w = target_pos_w - robot_pos_w
    delta_pos_w = delta_pos_w.clone()
    delta_pos_w[:, 2] = 0.0
    delta_pos_b = quat_rotate_inverse(yaw_quat(robot_quat_w, w_last=True), delta_pos_w, w_last=True)

    _, _, target_yaw = get_euler_xyz(target_quat_w, w_last=True)
    _, _, robot_yaw = get_euler_xyz(robot_quat_w, w_last=True)
    yaw_error = wrap_to_pi(target_yaw - robot_yaw)
    return torch.cat([delta_pos_b[:, :2], yaw_error.unsqueeze(-1)], dim=-1)


def motion_ref_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    return pos.view(env.num_envs, -1)


def motion_ref_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)

    num_bodies = len(motion_command.motion_cfg.body_names_to_track)
    pos_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)

    num_bodies = len(motion_command.motion_cfg.body_names_to_track)
    _, ori_b = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
    )
    mat = quaternion_to_matrix(ori_b, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.simulator_object_pos_w,
        motion_command.simulator_object_quat_w,
    )
    return pos.view(env.num_envs, -1)


def obj_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.simulator_object_pos_w,
        motion_command.simulator_object_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_lin_vel_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    vel_b = quat_rotate_inverse(motion_command.robot_ref_quat_w, motion_command.simulator_object_lin_vel_w, w_last=True)
    return vel_b.view(env.num_envs, -1)


def obj_ref_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.object_pos_w,
        motion_command.object_quat_w,
    )
    return pos.view(env.num_envs, -1)


def obj_ref_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.object_pos_w,
        motion_command.object_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_ref_lin_vel_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    vel_b = quat_rotate_inverse(motion_command.robot_ref_quat_w, motion_command.object_lin_vel_w, w_last=True)
    return vel_b.view(env.num_envs, -1)


def _lookahead_time_steps(motion_command: MotionCommand, lookahead_steps: int) -> torch.Tensor:
    if lookahead_steps < 0:
        raise ValueError(f"lookahead_steps must be non-negative, got {lookahead_steps}")
    motion_ids = motion_command.motion_ids
    end_idx = motion_command.motion.motion_end_idx[motion_ids]
    return torch.minimum(motion_command.time_steps + int(lookahead_steps), end_idx - 1)


def obj_ref_pos_next_b(env: WholeBodyTrackingManager, lookahead_steps: int = 1) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    time_steps = _lookahead_time_steps(motion_command, lookahead_steps)
    pos_w = motion_command.motion.object_pos_w[time_steps] + env.simulator.scene.env_origins
    quat_w = motion_command.motion.object_quat_w[time_steps]
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        pos_w,
        quat_w,
    )
    return pos.view(env.num_envs, -1)


def obj_ref_ori_next_b(env: WholeBodyTrackingManager, lookahead_steps: int = 1) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    time_steps = _lookahead_time_steps(motion_command, lookahead_steps)
    pos_w = motion_command.motion.object_pos_w[time_steps] + env.simulator.scene.env_origins
    quat_w = motion_command.motion.object_quat_w[time_steps]
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        pos_w,
        quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_ref_lin_vel_next_b(env: WholeBodyTrackingManager, lookahead_steps: int = 1) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    time_steps = _lookahead_time_steps(motion_command, lookahead_steps)
    lin_vel_w = motion_command.motion.object_lin_vel_w[time_steps]
    vel_b = quat_rotate_inverse(motion_command.robot_ref_quat_w, lin_vel_w, w_last=True)
    return vel_b.view(env.num_envs, -1)
