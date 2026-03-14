"""Reward terms for Whole Body Tracking tasks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

import torch

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.rotations import quat_error_magnitude

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


def motion_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.ref_pos_w - motion_command.robot_ref_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def motion_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.ref_quat_w, motion_command.robot_ref_quat_w) ** 2
    return torch.exp(-error / sigma**2)


def motion_relative_body_position_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
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
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_relative_body_orientation_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
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
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_lin_vel(
    env: WholeBodyTrackingManager,
    sigma: float,
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
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
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_ang_vel(
    env: WholeBodyTrackingManager,
    sigma: float,
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
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
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_joint_position_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    dof_names: list[str] | None = None,
    dof_name_pattern: str | None = None,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.square(motion_command.joint_pos - env.simulator.dof_pos)
    dof_indexes = _get_dof_subset_indexes(env, dof_names=dof_names, dof_name_pattern=dof_name_pattern)
    error = error.index_select(1, dof_indexes)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_joint_velocity_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    dof_names: list[str] | None = None,
    dof_name_pattern: str | None = None,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.square(motion_command.joint_vel - env.simulator.dof_vel)
    dof_indexes = _get_dof_subset_indexes(env, dof_names=dof_names, dof_name_pattern=dof_name_pattern)
    error = error.index_select(1, dof_indexes)
    return torch.exp(-error.mean(-1) / sigma**2)


# ================================================================================================
# Object Tracking Rewards
# ================================================================================================


def object_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.object_pos_w - motion_command.simulator_object_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def object_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w) ** 2
    return torch.exp(-error / sigma**2)


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
