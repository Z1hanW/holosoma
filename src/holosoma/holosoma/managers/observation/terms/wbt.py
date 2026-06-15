"""Whole body tracking observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.rotations import (
    calc_heading,
    calc_heading_quat_inv,
    normalize_angle,
    quat_apply,
    quat_rotate_inverse,
    quaternion_to_matrix,
    subtract_frame_transforms,
)
from holosoma.utils.torch_utils import get_axis_params, to_torch

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager

_DEG_TO_RAD = 0.017453292519943295


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


def _root_relative_xy_yaw_command(motion_command: MotionCommand) -> tuple[torch.Tensor, torch.Tensor]:
    rel_pos_w = motion_command.root_pos_w - motion_command.robot_root_pos_w
    heading_inv = calc_heading_quat_inv(motion_command.robot_root_quat_w, w_last=True)
    rel_pos_b = quat_apply(heading_inv, rel_pos_w, w_last=True)
    rel_xy = rel_pos_b[:, :2]

    target_heading = calc_heading(motion_command.root_quat_w)
    robot_heading = calc_heading(motion_command.robot_root_quat_w)
    rel_yaw = normalize_angle(target_heading - robot_heading).unsqueeze(1)
    return rel_xy, rel_yaw


def _contact_aware_segment_root_command(motion_command: MotionCommand) -> torch.Tensor:
    if not motion_command.motion.has_object:
        return torch.zeros((motion_command.num_envs, 3), device=motion_command.device, dtype=torch.float32)

    segment_steps = int(getattr(motion_command.motion_cfg, "contact_aware_sparse_root_segment_steps", 30))
    if segment_steps < 1:
        raise ValueError(f"contact_aware_sparse_root_segment_steps must be >= 1, got {segment_steps}")

    clip_ids = motion_command.clip_ids
    local_steps = motion_command.current_clip_local_steps
    clip_lengths = motion_command.current_clip_lengths
    carry_window_by_clip = motion_command._get_contact_aware_carry_window_by_clip()  # noqa: SLF001
    carry_start = carry_window_by_clip[clip_ids, 0]
    carry_end = carry_window_by_clip[clip_ids, 1]

    rel_steps = torch.clamp(local_steps - carry_start, min=0)
    segment_index = torch.div(rel_steps, segment_steps, rounding_mode="floor")
    segment_start = carry_start + segment_index * segment_steps
    segment_end = segment_start + segment_steps

    max_step = torch.clamp(clip_lengths - 1, min=0)
    safe_segment_start = torch.minimum(torch.clamp(segment_start, min=0), max_step)
    safe_segment_end = torch.minimum(torch.clamp(segment_end, min=0), max_step)
    start_motion_idx = motion_command._get_motion_indices_from_local_steps(safe_segment_start)  # noqa: SLF001
    end_motion_idx = motion_command._get_motion_indices_from_local_steps(safe_segment_end)  # noqa: SLF001

    root_pos_w = motion_command.motion.body_pos_w[:, 0]
    root_quat_w = motion_command.motion.body_quat_w[:, 0]
    start_pos_w = root_pos_w[start_motion_idx]
    end_pos_w = root_pos_w[end_motion_idx]
    start_quat_w = root_quat_w[start_motion_idx]
    end_quat_w = root_quat_w[end_motion_idx]

    heading_inv = calc_heading_quat_inv(start_quat_w, w_last=True)
    rel_pos_b = quat_apply(heading_inv, end_pos_w - start_pos_w, w_last=True)
    rel_xy = rel_pos_b[:, :2]
    rel_yaw = normalize_angle(calc_heading(end_quat_w) - calc_heading(start_quat_w)).unsqueeze(1)

    yaw_threshold_deg = float(
        getattr(motion_command.motion_cfg, "contact_aware_sparse_root_zero_yaw_threshold_deg", 0.0)
    )
    if yaw_threshold_deg > 0.0:
        yaw_threshold_rad = yaw_threshold_deg * _DEG_TO_RAD
        rel_yaw = torch.where(torch.abs(rel_yaw) <= yaw_threshold_rad, torch.zeros_like(rel_yaw), rel_yaw)

    command = torch.cat([rel_xy, rel_yaw], dim=-1)
    valid_endpoint = (segment_end < carry_end) & (segment_end < clip_lengths)
    active_mask = (local_steps >= carry_start) & (local_steps < carry_end) & valid_endpoint
    return torch.where(active_mask.unsqueeze(-1), command, torch.zeros_like(command))


def sparse_target_root_trajectory_command(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    rel_xy, rel_yaw = _root_relative_xy_yaw_command(motion_command)
    return torch.cat([rel_xy, rel_yaw], dim=-1)


def sparse_target_root_trajectory_command_contact_aware(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    base_command = sparse_target_root_trajectory_command(env)
    if not motion_command.motion.has_object:
        return base_command

    command_mode = (
        str(getattr(motion_command.motion_cfg, "contact_aware_sparse_root_command_mode", "tracking_error"))
        .strip()
        .lower()
        .replace("-", "_")
    )
    if command_mode in {"t1_aligned_segment", "segment", "segment_30", "segment_delta"}:
        return _contact_aware_segment_root_command(motion_command)
    if command_mode not in {"tracking_error", "tracking", "default", "robot_tracking_error"}:
        raise ValueError(f"Unsupported contact-aware sparse root command mode: {command_mode!r}")

    active_mask = motion_command.get_contact_aware_root_command_active_mask()
    return torch.where(active_mask.unsqueeze(-1), base_command, torch.zeros_like(base_command))


def drop_button(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    if not motion_command.motion.has_object:
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    return motion_command.get_contact_aware_drop_button().to(dtype=torch.float32).unsqueeze(-1)


def pickup_button(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    if not motion_command.motion.has_object:
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    return motion_command.get_contact_aware_pickup_button().to(dtype=torch.float32).unsqueeze(-1)


def clip_phase(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    clip_lengths = motion_command.current_clip_lengths.to(dtype=torch.float32)
    denom = torch.clamp(clip_lengths - 1.0, min=1.0)
    phase = motion_command.current_clip_local_steps.to(dtype=torch.float32) / denom
    return phase.unsqueeze(1)


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


def obj_target_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    if not motion_command.motion.has_object:
        return torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)

    pos_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.object_pos_w,
        motion_command.object_quat_w,
    )
    return pos_b.view(env.num_envs, -1)


def obj_target_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    if not motion_command.motion.has_object:
        return torch.zeros(env.num_envs, 6, device=env.device, dtype=torch.float32)

    _, ori_b = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.object_pos_w,
        motion_command.object_quat_w,
    )
    rot_mat = quaternion_to_matrix(ori_b, w_last=True)
    return rot_mat[..., :2].reshape(rot_mat.shape[0], -1)


def obj_size(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    if not motion_command.motion.has_object:
        return torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
    return motion_command.object_size.view(env.num_envs, -1)


def obj_lin_vel_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    unit_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    vel_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w.clone(),
        motion_command.robot_ref_quat_w.clone(),
        motion_command.simulator_object_lin_vel_w,
        unit_quat,
    )
    return vel_b.view(env.num_envs, -1)


def obj_ang_vel_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return quat_rotate_inverse(
        motion_command.robot_ref_quat_w,
        motion_command.simulator_object_ang_vel_w,
        w_last=True,
    )


def obj_target_pose_size_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    if not motion_command.motion.has_object:
        return torch.zeros(env.num_envs, 12, device=env.device, dtype=torch.float32)

    pos_b, ori_b = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.object_pos_w,
        motion_command.object_quat_w,
    )
    rot_mat = quaternion_to_matrix(ori_b, w_last=True)
    rot_6d = rot_mat[..., :2].reshape(rot_mat.shape[0], -1)
    return torch.cat([pos_b.view(env.num_envs, -1), rot_6d, motion_command.object_size.view(env.num_envs, -1)], dim=-1)


def obj_current_pose_size_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    if not motion_command.motion.has_object:
        return torch.zeros(env.num_envs, 12, device=env.device, dtype=torch.float32)

    pos_b, ori_b = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.simulator_object_pos_w,
        motion_command.simulator_object_quat_w,
    )
    rot_mat = quaternion_to_matrix(ori_b, w_last=True)
    rot_6d = rot_mat[..., :2].reshape(rot_mat.shape[0], -1)
    return torch.cat([pos_b.view(env.num_envs, -1), rot_6d, motion_command.object_size.view(env.num_envs, -1)], dim=-1)


def object_depth_map_b(
    env: WholeBodyTrackingManager,
    height: int = 17,
    width: int = 17,
    max_distance: float = 3.0,
    near: float = 0.15,
    hfov_deg: float = 89.5,
    vfov_deg: float = 58.6,
    sensor_offset: tuple[float, float, float] = (0.01, 0.01, 0.44),
    normalize: bool = True,
) -> torch.Tensor:
    """Analytic egocentric object depth raster for the public HOI student."""
    motion_command = _get_motion_command_and_assert_type(env)
    if height < 1 or width < 1:
        raise ValueError(f"object_depth_map_b height/width must be positive, got {height}x{width}")
    if max_distance <= near:
        raise ValueError(f"object_depth_map_b max_distance must be greater than near, got {max_distance} <= {near}")

    depth_map = torch.full((env.num_envs, height, width), float(max_distance), device=env.device, dtype=torch.float32)
    if not motion_command.motion.has_object:
        if normalize:
            depth_map = (depth_map - near) / (max_distance - near) - 0.5
        return depth_map.reshape(env.num_envs, -1)

    object_pos_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.simulator_object_pos_w,
        motion_command.simulator_object_quat_w,
    )
    offset = torch.tensor(sensor_offset, device=env.device, dtype=torch.float32).view(1, 3)
    object_pos_b = object_pos_b - offset
    object_size = torch.clamp(motion_command.object_size.to(dtype=torch.float32), min=0.01)

    forward = object_pos_b[:, 0].clamp(min=1.0e-6)
    lateral_angle = torch.atan2(object_pos_b[:, 1], forward)
    vertical_angle = torch.atan2(object_pos_b[:, 2], forward)
    hfov = hfov_deg * _DEG_TO_RAD
    vfov = vfov_deg * _DEG_TO_RAD

    h_axis = torch.linspace(-0.5 * hfov, 0.5 * hfov, width, device=env.device, dtype=torch.float32).view(
        1, 1, width
    )
    v_axis = torch.linspace(-0.5 * vfov, 0.5 * vfov, height, device=env.device, dtype=torch.float32).view(
        1, height, 1
    )

    half_lateral_extent = 0.5 * torch.linalg.norm(object_size[:, :2], dim=1)
    half_vertical_extent = 0.5 * object_size[:, 2]
    half_h_angle = torch.atan2(half_lateral_extent, forward).view(env.num_envs, 1, 1)
    half_v_angle = torch.atan2(half_vertical_extent, forward).view(env.num_envs, 1, 1)
    center_h = lateral_angle.view(env.num_envs, 1, 1)
    center_v = vertical_angle.view(env.num_envs, 1, 1)

    in_fov = (
        (object_pos_b[:, 0] > near)
        & (torch.abs(lateral_angle) <= 0.5 * hfov + half_h_angle.view(env.num_envs))
        & (torch.abs(vertical_angle) <= 0.5 * vfov + half_v_angle.view(env.num_envs))
    ).view(env.num_envs, 1, 1)
    mask = (torch.abs(h_axis - center_h) <= half_h_angle) & (torch.abs(v_axis - center_v) <= half_v_angle) & in_fov

    object_depth = torch.clamp(object_pos_b[:, 0] - 0.5 * object_size[:, 0], min=near, max=max_distance)
    depth_map = torch.where(mask, object_depth.view(env.num_envs, 1, 1), depth_map)
    if normalize:
        depth_map = (depth_map - near) / (max_distance - near) - 0.5
    return depth_map.reshape(env.num_envs, -1)
