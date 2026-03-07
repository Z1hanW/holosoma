"""Whole body tracking observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.rotations import (
    calc_heading,
    calc_heading_quat_inv,
    get_euler_xyz,
    matrix_to_quaternion,
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


def _rot6d_to_quat_xyzw(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert rotation-6D representation to xyzw quaternion."""
    if rot6d.ndim != 2 or rot6d.shape[1] != 6:
        raise ValueError(f"Expected rot6d with shape [N, 6], got {tuple(rot6d.shape)}")

    x_raw = rot6d[:, 0:3]
    y_raw = rot6d[:, 3:6]

    x_axis = torch.nn.functional.normalize(x_raw, dim=-1, eps=1.0e-6)
    y_ortho = y_raw - torch.sum(x_axis * y_raw, dim=-1, keepdim=True) * x_axis
    y_axis = torch.nn.functional.normalize(y_ortho, dim=-1, eps=1.0e-6)

    z_axis = torch.cross(x_axis, y_axis, dim=-1)
    z_axis = torch.nn.functional.normalize(z_axis, dim=-1, eps=1.0e-6)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    y_axis = torch.nn.functional.normalize(y_axis, dim=-1, eps=1.0e-6)

    rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    quat_wxyz = matrix_to_quaternion(rot_mat)
    return quat_wxyz[:, [1, 2, 3, 0]]


def _manual_object_goal_pose_size_w(motion_command: MotionCommand) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not bool(getattr(motion_command, "manual_goal_enabled", False)):
        return None
    if not motion_command.motion.has_object:
        return None

    goal_pos_w = getattr(motion_command, "manual_goal_object_pos_w", None)
    goal_rot6d_w = getattr(motion_command, "manual_goal_object_rot6d_w", None)
    if not isinstance(goal_pos_w, torch.Tensor) or not isinstance(goal_rot6d_w, torch.Tensor):
        return None
    if goal_pos_w.ndim != 2 or goal_pos_w.shape[1] != 3:
        return None
    if goal_rot6d_w.ndim != 2 or goal_rot6d_w.shape[1] != 6:
        return None

    num_envs = int(motion_command.robot_ref_pos_w.shape[0])
    if goal_pos_w.shape[0] != num_envs or goal_rot6d_w.shape[0] != num_envs:
        return None

    device = motion_command.robot_ref_pos_w.device
    goal_pos_w = goal_pos_w.to(device=device, dtype=torch.float32)
    goal_rot6d_w = goal_rot6d_w.to(device=device, dtype=torch.float32)
    goal_quat_w = _rot6d_to_quat_xyzw(goal_rot6d_w)
    goal_size = motion_command.object_size.view(num_envs, -1)
    return goal_pos_w, goal_quat_w, goal_size


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
    manual_goal = _manual_object_goal_pose_size_w(motion_command)
    if manual_goal is not None:
        return manual_goal
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

    Returns [rel_xy(2), rel_yaw(1), target_vxy(2), target_wz(1)] in robot heading frame.
    """
    motion_command = _get_motion_command_and_assert_type(env)
    command_device = motion_command.robot_ref_pos_w.device

    # Sparse goal mode: object goal is externally specified; keep root command underdetermined.
    if bool(getattr(motion_command, "manual_goal_enabled", False)):
        return torch.zeros((env.num_envs, 6), device=command_device, dtype=torch.float32)

    # Joystick/manual mode: use the operator command directly as sparse root command.
    if getattr(motion_command, "manual_control_enabled", False):
        manual_xy = getattr(motion_command, "manual_xy_rel", None)
        manual_yaw = getattr(motion_command, "manual_yaw_rel", None)
        if manual_xy is not None and manual_yaw is not None:
            if manual_xy.device != command_device:
                manual_xy = manual_xy.to(command_device)
            if manual_yaw.device != command_device:
                manual_yaw = manual_yaw.to(command_device)
            # We expose the same desired motion in both pose- and velocity-like slots.
            return torch.cat([manual_xy, manual_yaw, manual_xy, manual_yaw], dim=-1)

    rel_pos_w = motion_command.ref_pos_w - motion_command.robot_ref_pos_w
    heading_inv = calc_heading_quat_inv(motion_command.robot_ref_quat_w, w_last=True)
    rel_pos_b = quat_apply(heading_inv, rel_pos_w, w_last=True)
    rel_xy = rel_pos_b[:, :2]

    target_heading = calc_heading(motion_command.ref_quat_w)
    robot_heading = calc_heading(motion_command.robot_ref_quat_w)
    rel_yaw = normalize_angle(target_heading - robot_heading).unsqueeze(1)

    target_lin_vel_b = quat_apply(heading_inv, motion_command.ref_lin_vel_w, w_last=True)
    target_vxy = target_lin_vel_b[:, :2]

    target_ang_vel_b = quat_rotate_inverse(
        motion_command.robot_ref_quat_w,
        motion_command.ref_ang_vel_w,
        w_last=True,
    )
    target_wz = target_ang_vel_b[:, 2:3]

    return torch.cat([rel_xy, rel_yaw, target_vxy, target_wz], dim=-1)


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
    """Final object goal in robot-ref frame: [goal_pos(3), size(3)].

    In sparse-goal mode, the goal comes from externally provided object target pose.
    Otherwise, the goal is extracted from each active motion clip's last frame.
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
    """Final object goal in robot-ref frame: [goal_pos(3), goal_rot6d(6), size(3)].

    In sparse-goal mode, goal pose is user-provided; otherwise it comes from the clip's final frame.
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
