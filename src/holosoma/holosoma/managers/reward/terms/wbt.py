"""Reward terms for Whole Body Tracking tasks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.rotations import calc_heading, normalize_angle, quat_error_magnitude

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


def _get_cached_name_subset_indexes(
    env: WholeBodyTrackingManager,
    *,
    cache_name: str,
    all_names: list[str],
    names: list[str] | tuple[str, ...] | None = None,
    pattern: str | None = None,
) -> torch.Tensor:
    cache = getattr(env, cache_name, None)
    if cache is None:
        cache = {}
        setattr(env, cache_name, cache)

    key = (tuple(names) if names is not None else None, pattern)
    cached = cache.get(key)
    if cached is not None:
        return cached

    if names is not None:
        missing = [name for name in names if name not in all_names]
        if missing:
            raise ValueError(f"Requested names {missing} are not available in {all_names}.")
        indexes = [all_names.index(name) for name in names]
    elif pattern:
        regex = re.compile(pattern)
        indexes = [idx for idx, name in enumerate(all_names) if regex.match(name)]
    else:
        indexes = list(range(len(all_names)))

    if not indexes:
        raise ValueError(
            f"No names matched names={list(names) if names is not None else None} "
            f"pattern={pattern!r} in {all_names}."
        )

    tensor = torch.tensor(indexes, dtype=torch.long, device=env.device)
    cache[key] = tensor
    return tensor


def _get_tracked_body_subset_indexes(
    env: WholeBodyTrackingManager,
    motion_command: MotionCommand,
    *,
    body_names: list[str] | tuple[str, ...] | None = None,
    body_name_pattern: str | None = None,
) -> torch.Tensor:
    return _get_cached_name_subset_indexes(
        env,
        cache_name="_wbt_reward_tracked_body_subset_cache",
        all_names=list(motion_command.motion_cfg.body_names_to_track),
        names=body_names,
        pattern=body_name_pattern,
    )


def _get_dof_subset_indexes(
    env: WholeBodyTrackingManager,
    *,
    dof_names: list[str] | tuple[str, ...] | None = None,
    dof_name_pattern: str | None = None,
) -> torch.Tensor:
    return _get_cached_name_subset_indexes(
        env,
        cache_name="_wbt_reward_dof_subset_cache",
        all_names=list(env.simulator.dof_names),
        names=dof_names,
        pattern=dof_name_pattern,
    )


def _get_sim_body_subset_indexes(
    env: WholeBodyTrackingManager,
    *,
    body_names: list[str] | tuple[str, ...] | None = None,
    body_name_pattern: str | None = None,
) -> torch.Tensor:
    return _get_cached_name_subset_indexes(
        env,
        cache_name="_wbt_reward_sim_body_subset_cache",
        all_names=list(env.simulator.body_names),  # type: ignore[attr-defined]
        names=body_names,
        pattern=body_name_pattern,
    )


#########################################################################################################
## terms same to managers/reward/terms/locomotion.py
#########################################################################################################


def penalty_action_rate(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Penalize changes in actions between steps.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    return torch.sum(torch.square(prev_actions - actions), dim=1)


def limits_dof_pos(env: WholeBodyTrackingManager, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Penalize joint positions too close to limits.

    Args:
        env: The environment instance
        soft_dof_pos_limit: Soft limit as fraction of hard limit

    Returns:
        Reward tensor [num_envs]
    """
    # Use soft limits as fraction of hard limits
    m = (env.simulator.hard_dof_pos_limits[:, 0] + env.simulator.hard_dof_pos_limits[:, 1]) / 2  # type: ignore[attr-defined]
    r = env.simulator.hard_dof_pos_limits[:, 1] - env.simulator.hard_dof_pos_limits[:, 0]  # type: ignore[attr-defined]
    lower_soft_limit = m - 0.5 * r * soft_dof_pos_limit
    upper_soft_limit = m + 0.5 * r * soft_dof_pos_limit

    out_of_limits = -(env.simulator.dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
    out_of_limits += (env.simulator.dof_pos - upper_soft_limit).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################

# ================================================================================================
# Robot Tracking Rewards
# ================================================================================================


def motion_global_ref_position_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    only_external: bool = False,
    only_clip_goal: bool = False,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.ref_pos_w - motion_command.robot_ref_pos_w), dim=-1)
    reward = torch.exp(-error / sigma**2)
    active_mask = _reward_episode_mask(
        motion_command,
        only_external=only_external,
        only_clip_goal=only_clip_goal,
    )
    return reward * active_mask.to(dtype=torch.float32)


def motion_global_ref_orientation_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    only_external: bool = False,
    only_clip_goal: bool = False,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.ref_quat_w, motion_command.robot_ref_quat_w) ** 2
    reward = torch.exp(-error / sigma**2)
    active_mask = _reward_episode_mask(
        motion_command,
        only_external=only_external,
        only_clip_goal=only_clip_goal,
    )
    return reward * active_mask.to(dtype=torch.float32)


def _reward_episode_mask(
    motion_command: MotionCommand,
    *,
    only_external: bool = False,
    only_clip_goal: bool = False,
) -> torch.Tensor:
    if only_external and only_clip_goal:
        raise ValueError("only_external and only_clip_goal cannot both be True.")
    if not only_external and not only_clip_goal:
        return torch.ones((motion_command.num_envs,), device=motion_command.device, dtype=torch.bool)
    if not motion_command.motion.has_object or not motion_command.manual_goal_enabled:
        return torch.zeros((motion_command.num_envs,), device=motion_command.device, dtype=torch.bool)

    external_mask = motion_command.get_sparse_goal_external_mask()
    if only_external:
        return external_mask
    return ~external_mask


def motion_relative_body_position_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
    only_external: bool = False,
    only_clip_goal: bool = False,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_pos_relative_w - motion_command.robot_body_pos_w), dim=-1)
    body_indexes = _get_tracked_body_subset_indexes(
        env,
        motion_command,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
    )
    error = error.index_select(1, body_indexes)
    reward = torch.exp(-error.mean(-1) / sigma**2)
    active_mask = _reward_episode_mask(
        motion_command,
        only_external=only_external,
        only_clip_goal=only_clip_goal,
    )
    return reward * active_mask.to(dtype=torch.float32)


def motion_relative_body_orientation_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
    only_external: bool = False,
    only_clip_goal: bool = False,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.body_quat_relative_w, motion_command.robot_body_quat_w) ** 2
    body_indexes = _get_tracked_body_subset_indexes(
        env,
        motion_command,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
    )
    error = error.index_select(1, body_indexes)
    reward = torch.exp(-error.mean(-1) / sigma**2)
    active_mask = _reward_episode_mask(
        motion_command,
        only_external=only_external,
        only_clip_goal=only_clip_goal,
    )
    return reward * active_mask.to(dtype=torch.float32)


def motion_global_body_lin_vel(
    env: WholeBodyTrackingManager,
    sigma: float,
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
    only_external: bool = False,
    only_clip_goal: bool = False,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_lin_vel_w - motion_command.robot_body_lin_vel_w), dim=-1)
    body_indexes = _get_tracked_body_subset_indexes(
        env,
        motion_command,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
    )
    error = error.index_select(1, body_indexes)
    reward = torch.exp(-error.mean(-1) / sigma**2)
    active_mask = _reward_episode_mask(
        motion_command,
        only_external=only_external,
        only_clip_goal=only_clip_goal,
    )
    return reward * active_mask.to(dtype=torch.float32)


def motion_global_body_ang_vel(
    env: WholeBodyTrackingManager,
    sigma: float,
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
    only_external: bool = False,
    only_clip_goal: bool = False,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_ang_vel_w - motion_command.robot_body_ang_vel_w), dim=-1)
    body_indexes = _get_tracked_body_subset_indexes(
        env,
        motion_command,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
    )
    error = error.index_select(1, body_indexes)
    reward = torch.exp(-error.mean(-1) / sigma**2)
    active_mask = _reward_episode_mask(
        motion_command,
        only_external=only_external,
        only_clip_goal=only_clip_goal,
    )
    return reward * active_mask.to(dtype=torch.float32)


def motion_joint_position_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    dof_names: list[str] | None = None,
    dof_name_pattern: str | None = None,
    only_external: bool = False,
    only_clip_goal: bool = False,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.square(motion_command.joint_pos - env.simulator.dof_pos)
    dof_indexes = _get_dof_subset_indexes(env, dof_names=dof_names, dof_name_pattern=dof_name_pattern)
    error = error.index_select(1, dof_indexes)
    reward = torch.exp(-error.mean(-1) / sigma**2)
    active_mask = _reward_episode_mask(
        motion_command,
        only_external=only_external,
        only_clip_goal=only_clip_goal,
    )
    return reward * active_mask.to(dtype=torch.float32)


def motion_joint_velocity_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    dof_names: list[str] | None = None,
    dof_name_pattern: str | None = None,
    only_external: bool = False,
    only_clip_goal: bool = False,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.square(motion_command.joint_vel - env.simulator.dof_vel)
    dof_indexes = _get_dof_subset_indexes(env, dof_names=dof_names, dof_name_pattern=dof_name_pattern)
    error = error.index_select(1, dof_indexes)
    reward = torch.exp(-error.mean(-1) / sigma**2)
    active_mask = _reward_episode_mask(
        motion_command,
        only_external=only_external,
        only_clip_goal=only_clip_goal,
    )
    return reward * active_mask.to(dtype=torch.float32)


# ================================================================================================
# Object Tracking Rewards
# ================================================================================================


def object_global_ref_position_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    only_external: bool = False,
    only_clip_goal: bool = False,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.object_pos_w - motion_command.simulator_object_pos_w), dim=-1)
    reward = torch.exp(-error / sigma**2)
    active_mask = _reward_episode_mask(
        motion_command,
        only_external=only_external,
        only_clip_goal=only_clip_goal,
    )
    return reward * active_mask.to(dtype=torch.float32)


def object_global_ref_orientation_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    only_external: bool = False,
    only_clip_goal: bool = False,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w) ** 2
    reward = torch.exp(-error / sigma**2)
    active_mask = _reward_episode_mask(
        motion_command,
        only_external=only_external,
        only_clip_goal=only_clip_goal,
    )
    return reward * active_mask.to(dtype=torch.float32)


def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    first_col = F.normalize(rot6d[..., 0:3], dim=-1)
    second_col_raw = rot6d[..., 3:6]
    second_col = F.normalize(
        second_col_raw - torch.sum(first_col * second_col_raw, dim=-1, keepdim=True) * first_col,
        dim=-1,
    )
    third_col = torch.cross(first_col, second_col, dim=-1)
    return torch.stack((first_col, second_col, third_col), dim=-1)


def _goal_episode_mask(motion_command: MotionCommand, *, only_external: bool) -> torch.Tensor:
    active_mask = _reward_episode_mask(motion_command, only_external=only_external)
    if motion_command.manual_goal_object_pos_w is None or motion_command.manual_goal_object_rot6d_w is None:
        return torch.zeros((motion_command.num_envs,), device=motion_command.device, dtype=torch.bool)
    return active_mask


def _picked_mask(motion_command: MotionCommand) -> torch.Tensor:
    if motion_command.pickup_anchor_set is None:
        return torch.zeros((motion_command.num_envs,), device=motion_command.device, dtype=torch.bool)
    return motion_command.pickup_anchor_set


def _manual_goal_heading(motion_command: MotionCommand) -> torch.Tensor:
    assert motion_command.manual_goal_object_rot6d_w is not None
    goal_rot_mat_w = _rot6d_to_matrix(motion_command.manual_goal_object_rot6d_w)
    return torch.atan2(goal_rot_mat_w[:, 1, 0], goal_rot_mat_w[:, 0, 0])


def _sparse_goal_errors(
    motion_command: MotionCommand,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert motion_command.manual_goal_object_pos_w is not None
    goal_pos_w = motion_command.manual_goal_object_pos_w
    goal_heading = _manual_goal_heading(motion_command)
    current_heading = calc_heading(motion_command.simulator_object_quat_w)

    xy_error = torch.norm(goal_pos_w[:, :2] - motion_command.simulator_object_pos_w[:, :2], dim=-1)
    yaw_error = torch.abs(normalize_angle(goal_heading - current_heading))
    z_error = torch.abs(goal_pos_w[:, 2] - motion_command.simulator_object_pos_w[:, 2])
    return xy_error, yaw_error, z_error


def _near_goal_mask(
    motion_command: MotionCommand,
    *,
    only_external: bool,
    picked_only: bool,
    xy_threshold: float,
    yaw_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    active_mask = _goal_episode_mask(motion_command, only_external=only_external)
    if picked_only:
        active_mask &= _picked_mask(motion_command)
    if not active_mask.any() or motion_command.manual_goal_object_pos_w is None:
        zeros = torch.zeros(motion_command.num_envs, device=motion_command.device, dtype=torch.float32)
        false_mask = torch.zeros(motion_command.num_envs, device=motion_command.device, dtype=torch.bool)
        return false_mask, zeros, zeros, zeros

    xy_error, yaw_error, z_error = _sparse_goal_errors(motion_command)
    near_goal = active_mask & (xy_error <= xy_threshold) & (yaw_error <= yaw_threshold)
    return near_goal, xy_error, yaw_error, z_error


def _sparse_goal_success_mask(
    motion_command: MotionCommand,
    *,
    only_external: bool,
    xy_threshold: float,
    yaw_threshold: float,
    z_threshold: float,
    lin_vel_threshold: float,
    ang_vel_threshold: float,
) -> torch.Tensor:
    active_mask = _goal_episode_mask(motion_command, only_external=only_external) & _picked_mask(motion_command)
    if not active_mask.any():
        return active_mask

    xy_error, yaw_error, z_error = _sparse_goal_errors(motion_command)
    lin_speed = torch.norm(motion_command.simulator_object_lin_vel_w, dim=-1)
    ang_speed = torch.norm(motion_command.simulator_object_ang_vel_w, dim=-1)

    return (
        active_mask
        & (xy_error <= xy_threshold)
        & (yaw_error <= yaw_threshold)
        & (z_error <= z_threshold)
        & (lin_speed <= lin_vel_threshold)
        & (ang_speed <= ang_vel_threshold)
    )


def sparse_goal_pickup_height_reward(
    env: WholeBodyTrackingManager,
    target_height_delta: float = 0.12,
    only_external: bool = True,
    stop_after_pick: bool = False,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    active_mask = _goal_episode_mask(motion_command, only_external=only_external)
    if stop_after_pick:
        active_mask &= ~_picked_mask(motion_command)
    if not active_mask.any() or motion_command.pickup_object_rel_z_baseline is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    current_rel_z = motion_command.simulator_object_pos_w[:, 2] - motion_command.robot_root_pos_w[:, 2]
    lifted = torch.clamp(current_rel_z - motion_command.pickup_object_rel_z_baseline, min=0.0)
    reward = torch.clamp(lifted / max(target_height_delta, 1.0e-6), min=0.0, max=1.0)
    return reward * active_mask.to(dtype=torch.float32)


def sparse_goal_object_xy_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    only_external: bool = True,
    picked_only: bool = True,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    active_mask = _goal_episode_mask(motion_command, only_external=only_external)
    if picked_only:
        active_mask &= _picked_mask(motion_command)
    if not active_mask.any() or motion_command.manual_goal_object_pos_w is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    error = torch.sum(
        torch.square(motion_command.manual_goal_object_pos_w[:, :2] - motion_command.simulator_object_pos_w[:, :2]),
        dim=-1,
    )
    reward = torch.exp(-error / sigma**2)
    return reward * active_mask.to(dtype=torch.float32)


def sparse_goal_object_yaw_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    only_external: bool = True,
    picked_only: bool = True,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    active_mask = _goal_episode_mask(motion_command, only_external=only_external)
    if picked_only:
        active_mask &= _picked_mask(motion_command)
    if not active_mask.any() or motion_command.manual_goal_object_rot6d_w is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    goal_heading = _manual_goal_heading(motion_command)
    current_heading = calc_heading(motion_command.simulator_object_quat_w)
    error = torch.square(normalize_angle(goal_heading - current_heading))
    reward = torch.exp(-error / sigma**2)
    return reward * active_mask.to(dtype=torch.float32)


def sparse_goal_object_z_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    only_external: bool = True,
    picked_only: bool = True,
    near_goal_xy_threshold: float = 0.25,
    near_goal_yaw_threshold: float = 0.70,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    near_goal, _, _, z_error = _near_goal_mask(
        motion_command,
        only_external=only_external,
        picked_only=picked_only,
        xy_threshold=near_goal_xy_threshold,
        yaw_threshold=near_goal_yaw_threshold,
    )
    if not near_goal.any():
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    reward = torch.exp(-torch.square(z_error) / sigma**2)
    return reward * near_goal.to(dtype=torch.float32)


def sparse_goal_object_pose_error_exp(
    env: WholeBodyTrackingManager,
    sigma_xy: float,
    sigma_yaw: float,
    sigma_z: float,
    only_external: bool = True,
    picked_only: bool = False,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    active_mask = _goal_episode_mask(motion_command, only_external=only_external)
    if picked_only:
        active_mask &= _picked_mask(motion_command)
    if not active_mask.any():
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    xy_error, yaw_error, z_error = _sparse_goal_errors(motion_command)
    pose_error = (
        torch.square(xy_error) / max(float(sigma_xy), 1.0e-6) ** 2
        + torch.square(yaw_error) / max(float(sigma_yaw), 1.0e-6) ** 2
        + torch.square(z_error) / max(float(sigma_z), 1.0e-6) ** 2
    )
    reward = torch.exp(-pose_error)
    return reward * active_mask.to(dtype=torch.float32)


def sparse_goal_hover_height_penalty(
    env: WholeBodyTrackingManager,
    only_external: bool = True,
    picked_only: bool = True,
    near_goal_xy_threshold: float = 0.20,
    near_goal_yaw_threshold: float = 0.60,
    target_height_margin: float = 0.10,
    height_scale: float = 0.12,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    near_goal, _, _, _ = _near_goal_mask(
        motion_command,
        only_external=only_external,
        picked_only=picked_only,
        xy_threshold=near_goal_xy_threshold,
        yaw_threshold=near_goal_yaw_threshold,
    )
    if not near_goal.any() or motion_command.manual_goal_object_pos_w is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    target_height = motion_command.manual_goal_object_pos_w[:, 2] + float(target_height_margin)
    height_excess = torch.clamp(motion_command.simulator_object_pos_w[:, 2] - target_height, min=0.0)
    penalty = torch.clamp(height_excess / max(float(height_scale), 1.0e-6), min=0.0, max=1.0)
    return penalty * near_goal.to(dtype=torch.float32)


def sparse_goal_success_bonus(
    env: WholeBodyTrackingManager,
    only_external: bool = True,
    xy_threshold: float = 0.10,
    yaw_threshold: float = 0.35,
    z_threshold: float = 0.06,
    lin_vel_threshold: float = 0.30,
    ang_vel_threshold: float = 1.50,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    success = _sparse_goal_success_mask(
        motion_command,
        only_external=only_external,
        xy_threshold=xy_threshold,
        yaw_threshold=yaw_threshold,
        z_threshold=z_threshold,
        lin_vel_threshold=lin_vel_threshold,
        ang_vel_threshold=ang_vel_threshold,
    )
    return success.to(dtype=torch.float32)


def body_contact_reward(
    env: WholeBodyTrackingManager,
    threshold: float = 1.0,
    force_scale: float = 25.0,
    reward_mode: str = "binary",
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
) -> torch.Tensor:
    body_indexes = _get_sim_body_subset_indexes(
        env,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
    )
    contact_forces = env.simulator.contact_forces_history[:, :, body_indexes]
    magnitudes = torch.norm(contact_forces, dim=-1)
    peak_force = torch.max(magnitudes, dim=1)[0]

    if reward_mode == "binary":
        reward = (peak_force > threshold).to(dtype=torch.float32)
    elif reward_mode == "linear":
        reward = torch.clamp((peak_force - threshold) / max(force_scale, 1.0e-6), min=0.0, max=1.0)
    elif reward_mode == "tanh":
        reward = torch.tanh(torch.clamp(peak_force - threshold, min=0.0) / max(force_scale, 1.0e-6))
    else:
        raise ValueError(f"Unsupported reward_mode '{reward_mode}'. Use one of: binary, linear, tanh.")

    return reward.mean(dim=1)


# ================================================================================================
# Undesired Contacts Rewards
# ================================================================================================


class UndesiredContacts(RewardTermBase):
    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        undesired_contacts_body_names = [
            body_name
            for body_name in self.env.simulator.body_names  # type: ignore[attr-defined]
            if re.match(cfg.params.get("undesired_contacts_body_names", ""), body_name)
        ]
        self.undesired_contacts_body_indexes = self._get_index_of_a_in_b(
            undesired_contacts_body_names,
            self.env.simulator.body_names,  # type: ignore[attr-defined]
            self.env.device,
        )
        self.threshold = cfg.params.get("threshold", 1.0)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        # (num_envs, history_length, num_bodies, 3)
        net_contact_forces = self.env.simulator.contact_forces_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.undesired_contacts_body_indexes], dim=-1), dim=1)[0]
            > self.threshold
        )
        return torch.sum(is_contact, dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
