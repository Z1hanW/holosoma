"""Whole body tracking observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.rotations import (
    calc_heading,
    calc_heading_quat_inv,
    get_euler_xyz,
    normalize_angle,
    quat_apply,
    quat_rotate_inverse,
    quaternion_to_matrix,
    subtract_frame_transforms,
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


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################


def torso_real(env: WholeBodyTrackingManager) -> torch.Tensor:
    """VideoMimic-style torso observation (current frame)."""
    return torch.cat(
        [
            base_ang_vel(env),
            projected_gravity(env),
            dof_pos(env),
            dof_vel(env),
            actions(env),
        ],
        dim=-1,
    )


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


def _root_relative_xy_yaw_command(motion_command: MotionCommand) -> tuple[torch.Tensor, torch.Tensor]:
    """Root-relative XY/yaw command in the robot root heading frame."""
    rel_pos_w = motion_command.root_pos_w - motion_command.robot_root_pos_w
    heading_inv = calc_heading_quat_inv(motion_command.robot_root_quat_w, w_last=True)
    rel_pos_b = quat_apply(heading_inv, rel_pos_w, w_last=True)
    rel_xy = rel_pos_b[:, :2]

    target_heading = calc_heading(motion_command.root_quat_w)
    robot_heading = calc_heading(motion_command.robot_root_quat_w)
    rel_yaw = normalize_angle(target_heading - robot_heading).unsqueeze(1)
    return rel_xy, rel_yaw


def _clip_final_object_goal_pose_size_w(motion_command: MotionCommand) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clip_ids = motion_command.clip_ids
    clip_offsets = motion_command.motion.clip_offsets[clip_ids]
    clip_lengths = motion_command.motion.clip_lengths[clip_ids]
    final_steps = torch.clamp(clip_lengths - 1, min=0)
    final_motion_idx = clip_offsets + final_steps

    goal_pos_w = motion_command.motion.object_pos_w[final_motion_idx]
    goal_quat_w = motion_command.motion.object_quat_w[final_motion_idx]
    if motion_command.motion_cfg.align_motion_to_init_yaw:
        goal_pos_w = motion_command._apply_motion_alignment_pos(goal_pos_w)  # noqa: SLF001
        goal_quat_w = motion_command._apply_motion_alignment_quat(goal_quat_w)  # noqa: SLF001
    else:
        goal_pos_w = goal_pos_w + motion_command._get_env_offsets()  # noqa: SLF001
    goal_size = motion_command.motion.object_size[final_motion_idx].view(motion_command.num_envs, -1)
    return goal_pos_w, goal_quat_w, goal_size


def _object_goal_pose_size_w(motion_command: MotionCommand) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not motion_command.motion.has_object:
        return None
    return _clip_final_object_goal_pose_size_w(motion_command)


def motion_command(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.command


def motion_future_target_poses(
    env: WholeBodyTrackingManager,
    num_future_steps: int | None = None,
    target_pose_type: str | None = None,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.get_future_target_poses(
        num_future_steps=num_future_steps,
        target_pose_type=target_pose_type,
    )


def torso_xy_rel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Local-frame XY position from robot torso to motion target torso."""
    motion_command = _get_motion_command_and_assert_type(env)
    if getattr(motion_command, "manual_control_enabled", False):
        manual_xy = getattr(motion_command, "manual_xy_rel", None)
        if manual_xy is not None:
            if manual_xy.device != motion_command.robot_ref_pos_w.device:
                manual_xy = manual_xy.to(motion_command.robot_ref_pos_w.device)
            return manual_xy
    rel_pos_w = motion_command.ref_pos_w - motion_command.robot_ref_pos_w
    heading_inv = calc_heading_quat_inv(motion_command.robot_ref_quat_w, w_last=True)
    rel_pos_b = quat_apply(heading_inv, rel_pos_w, w_last=True)
    return rel_pos_b[:, :2]


def torso_yaw_rel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Local-frame yaw from robot torso to motion target torso."""
    motion_command = _get_motion_command_and_assert_type(env)
    if getattr(motion_command, "manual_control_enabled", False):
        manual_yaw = getattr(motion_command, "manual_yaw_rel", None)
        if manual_yaw is not None:
            if manual_yaw.device != motion_command.robot_ref_pos_w.device:
                manual_yaw = manual_yaw.to(motion_command.robot_ref_pos_w.device)
            return manual_yaw
    target_heading = calc_heading(motion_command.ref_quat_w)
    robot_heading = calc_heading(motion_command.robot_ref_quat_w)
    heading_error = normalize_angle(target_heading - robot_heading)
    return heading_error.unsqueeze(1)


def sparse_target_root_trajectory_command(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Sparse root-trajectory command for locomotion-style distillation.

    Returns [rel_xy(2), rel_yaw(1)] in the robot root heading frame.
    """
    motion_command = _get_motion_command_and_assert_type(env)
    command_device = motion_command.robot_root_pos_w.device

    # Joystick/manual mode: use the operator command directly as sparse root command.
    if getattr(motion_command, "manual_control_enabled", False):
        manual_xy = getattr(motion_command, "manual_xy_rel", None)
        manual_yaw = getattr(motion_command, "manual_yaw_rel", None)
        if manual_xy is not None and manual_yaw is not None:
            if manual_xy.device != command_device:
                manual_xy = manual_xy.to(command_device)
            if manual_yaw.device != command_device:
                manual_yaw = manual_yaw.to(command_device)
            return torch.cat([manual_xy, manual_yaw], dim=-1)

    rel_xy, rel_yaw = _root_relative_xy_yaw_command(motion_command)
    return torch.cat([rel_xy, rel_yaw], dim=-1)


def clip_phase(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Normalized motion progress in current clip, in [0, 1]."""
    motion_command = _get_motion_command_and_assert_type(env)
    clip_lengths = motion_command.current_clip_lengths.to(dtype=torch.float32)
    denom = torch.clamp(clip_lengths - 1.0, min=1.0)
    phase = motion_command.time_steps.to(dtype=torch.float32) / denom
    return phase.unsqueeze(1)


def target_joints(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Target joint angles from motion data, relative to default pose."""
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.joint_pos - env.default_dof_pos


def target_root_roll(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Target root roll (motion data) in radians."""
    motion_command = _get_motion_command_and_assert_type(env)
    roll, _, _ = get_euler_xyz(motion_command.root_quat_w, w_last=True)
    return normalize_angle(roll).unsqueeze(1)


def target_root_pitch(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Target root pitch (motion data) in radians."""
    motion_command = _get_motion_command_and_assert_type(env)
    _, pitch, _ = get_euler_xyz(motion_command.root_quat_w, w_last=True)
    return normalize_angle(pitch).unsqueeze(1)


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
    """Target object info in robot-ref frame: [pos(3), rot6d(6), size(3)]."""
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
    size = motion_command.object_size.view(env.num_envs, -1)
    return torch.cat([pos_b.view(env.num_envs, -1), rot_6d, size], dim=-1)


def obj_current_pose_size_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Current object info in robot-ref frame: [pos(3), rot6d(6), size(3)].

    Uses simulator object pose (current state), not mocap target pose.
    Size stays sourced from active motion clip metadata.
    """
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
    size = motion_command.object_size.view(env.num_envs, -1)
    return torch.cat([pos_b.view(env.num_envs, -1), rot_6d, size], dim=-1)


def obj_goal_pos_size_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Final clip object target in robot-ref frame: [target_pos(3), size(3)].

    The target is extracted from each active motion clip's last frame.
    """
    motion_command = _get_motion_command_and_assert_type(env)
    goal = _object_goal_pose_size_w(motion_command)
    if goal is None:
        return torch.zeros(env.num_envs, 6, device=env.device, dtype=torch.float32)

    goal_pos_w, goal_quat_w, goal_size = goal
    goal_pos_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        goal_pos_w,
        goal_quat_w,
    )
    return torch.cat([goal_pos_b.view(env.num_envs, -1), goal_size], dim=-1)


def obj_goal_pose_size_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Final clip object target in robot-ref frame: [target_pos(3), target_rot6d(6), size(3)].

    The target pose is extracted from each active motion clip's last frame.
    """
    motion_command = _get_motion_command_and_assert_type(env)
    goal = _object_goal_pose_size_w(motion_command)
    if goal is None:
        return torch.zeros(env.num_envs, 12, device=env.device, dtype=torch.float32)

    goal_pos_w, goal_quat_w, goal_size = goal
    goal_pos_b, goal_quat_b = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        goal_pos_w,
        goal_quat_w,
    )
    goal_rot_mat_b = quaternion_to_matrix(goal_quat_b, w_last=True)
    goal_rot6d_b = goal_rot_mat_b[..., :2].reshape(goal_rot_mat_b.shape[0], -1)
    return torch.cat([goal_pos_b.view(env.num_envs, -1), goal_rot6d_b, goal_size], dim=-1)
