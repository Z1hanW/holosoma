"""Whole Body Tracking-specific termination terms."""

from __future__ import annotations

import os
from typing import Any, List

import torch.nn.functional as F

from holosoma.config_types.termination import TerminationTermCfg
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.terms.wbt import gravity_vector
from holosoma.managers.termination.base import TerminationTermBase
from holosoma.utils.rotations import (
    calc_heading,
    normalize_angle,
    quat_error_magnitude,
    quat_rotate_inverse,
)
from holosoma.utils.safe_torch_import import torch


#########################################################################################################
## Termination terms
#########################################################################################################
def motion_ends(env, **_) -> torch.Tensor:
    """Terminate if the motion ends."""
    if os.environ.get("HOLOSOMA_DISABLE_MOTION_END_RESET", "0").lower() in ("1", "true", "yes", "on"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    motion_command = env.command_manager.get_state("motion_command")
    return motion_command.motion_end_mask()


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
    if (
        not motion_command.motion.has_object
        or not motion_command.manual_goal_enabled
        or motion_command.manual_goal_object_pos_w is None
        or motion_command.manual_goal_object_rot6d_w is None
    ):
        return torch.zeros((motion_command.num_envs,), device=motion_command.device, dtype=torch.bool)
    if not only_external:
        return torch.ones((motion_command.num_envs,), device=motion_command.device, dtype=torch.bool)
    return motion_command.get_sparse_goal_external_mask()


def _picked_mask(motion_command: MotionCommand) -> torch.Tensor:
    if motion_command.pickup_anchor_set is None:
        return torch.zeros((motion_command.num_envs,), device=motion_command.device, dtype=torch.bool)
    return motion_command.pickup_anchor_set


def _manual_goal_heading(motion_command: MotionCommand) -> torch.Tensor:
    assert motion_command.manual_goal_object_rot6d_w is not None
    goal_rot_mat_w = _rot6d_to_matrix(motion_command.manual_goal_object_rot6d_w)
    return torch.atan2(goal_rot_mat_w[:, 1, 0], goal_rot_mat_w[:, 0, 0])


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

    assert motion_command.manual_goal_object_pos_w is not None
    goal_pos_w = motion_command.manual_goal_object_pos_w
    goal_heading = _manual_goal_heading(motion_command)
    current_heading = calc_heading(motion_command.simulator_object_quat_w)

    xy_error = torch.norm(goal_pos_w[:, :2] - motion_command.simulator_object_pos_w[:, :2], dim=-1)
    yaw_error = torch.abs(normalize_angle(goal_heading - current_heading))
    z_error = torch.abs(goal_pos_w[:, 2] - motion_command.simulator_object_pos_w[:, 2])
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


def motion_ends_if_clip_goal(env, only_clip_goal: bool = True, **_) -> torch.Tensor:
    """Terminate on clip end for clip-conditioned episodes, but keep external-goal episodes alive."""
    motion_command = env.command_manager.get_state("motion_command")
    end_mask = motion_ends(env)
    if not only_clip_goal:
        return end_mask
    return end_mask & (~motion_command.get_sparse_goal_external_mask())


def drop_task_base_height_below_threshold(env, min_height: float = 0.45) -> torch.Tensor:
    """Terminate when the robot base collapses below a fixed height."""
    return env.simulator.robot_root_states[:, 2] < min_height


class SparseGoalSuccess(TerminationTermBase):
    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.only_external = bool(cfg.params.get("only_external", True))
        self.xy_threshold = float(cfg.params.get("xy_threshold", 0.10))
        self.yaw_threshold = float(cfg.params.get("yaw_threshold", 0.35))
        self.z_threshold = float(cfg.params.get("z_threshold", 0.06))
        self.lin_vel_threshold = float(cfg.params.get("lin_vel_threshold", 0.30))
        self.ang_vel_threshold = float(cfg.params.get("ang_vel_threshold", 1.50))
        self.hold_steps = max(1, int(cfg.params.get("hold_steps", 10)))
        self._success_counter = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.env.device)

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        motion_command = self.env.command_manager.get_state("motion_command")
        success = _sparse_goal_success_mask(
            motion_command,
            only_external=self.only_external,
            xy_threshold=self.xy_threshold,
            yaw_threshold=self.yaw_threshold,
            z_threshold=self.z_threshold,
            lin_vel_threshold=self.lin_vel_threshold,
            ang_vel_threshold=self.ang_vel_threshold,
        )
        self._success_counter = torch.where(success, self._success_counter + 1, torch.zeros_like(self._success_counter))
        return self._success_counter >= self.hold_steps

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._success_counter.zero_()
        else:
            self._success_counter[env_ids] = 0


class SparseGoalDroppedAway(TerminationTermBase):
    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.only_external = bool(cfg.params.get("only_external", True))
        self.xy_fail_threshold = float(cfg.params.get("xy_fail_threshold", 0.35))
        self.release_height_margin = float(cfg.params.get("release_height_margin", 0.08))
        self.hold_steps = max(1, int(cfg.params.get("hold_steps", 2)))
        self._failure_counter = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.env.device)

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        motion_command = self.env.command_manager.get_state("motion_command")
        active_mask = _goal_episode_mask(motion_command, only_external=self.only_external) & _picked_mask(motion_command)
        if not active_mask.any() or motion_command.manual_goal_object_pos_w is None:
            self._failure_counter.zero_()
            return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)

        xy_error = torch.norm(
            motion_command.manual_goal_object_pos_w[:, :2] - motion_command.simulator_object_pos_w[:, :2],
            dim=-1,
        )
        dropped_low = motion_command.simulator_object_pos_w[:, 2] <= (
            motion_command.manual_goal_object_pos_w[:, 2] + self.release_height_margin
        )
        failed = active_mask & dropped_low & (xy_error > self.xy_fail_threshold)
        self._failure_counter = torch.where(failed, self._failure_counter + 1, torch.zeros_like(self._failure_counter))
        return self._failure_counter >= self.hold_steps

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._failure_counter.zero_()
        else:
            self._failure_counter[env_ids] = 0


class BadTracking(TerminationTermBase):
    """Terminate if the tracking is bad.

    - bad ref pos
    - bad ref ori
    - bad motion body pos
    if has object:
        - bad object pos
        - bad object ori

    When bad tracking is detected, the motion_commmand.AdaptiveTimestepsSampler will be updated.
    """

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        self.bad_ref_pos_threshold = cfg.params["bad_ref_pos_threshold"]
        self.bad_ref_ori_threshold = cfg.params["bad_ref_ori_threshold"]

        self.bad_motion_body_pos_body_names = cfg.params["bad_motion_body_pos_body_names"]

        # NOTE: body_names_to_track is shared with command_manager
        self.body_names_to_track = cfg.params["body_names_to_track"]
        self.bad_motion_body_pos_threshold = cfg.params["bad_motion_body_pos_threshold"]
        self.bad_motion_body_pos_body_indexes = self._get_index_of_a_in_b(
            self.bad_motion_body_pos_body_names, self.body_names_to_track, self.env.device
        )

        self.bad_object_pos_threshold = cfg.params["bad_object_pos_threshold"]
        self.bad_object_ori_threshold = cfg.params["bad_object_ori_threshold"]

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        motion_command = self.env.command_manager.get_state("motion_command")
        assert motion_command.motion_cfg.body_names_to_track == self.body_names_to_track, (
            "body_names_to_track in motion_command and termination.params are not the same"
            f"motion_command.motion_cfg.body_names_to_track: {motion_command.motion_cfg.body_names_to_track}"
            f"termination.params['body_names_to_track']: {self.body_names_to_track}"
        )

        # During evaluation, disable BadTracking-based termination entirely.
        if self.env.is_evaluating:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        bad_ref_pos = self.bad_ref_pos(motion_command)
        bad_ref_ori = self.bad_ref_ori(motion_command)
        bad_motion_body_pos = self.bad_motion_body_pos(motion_command)
        bad_tracking = bad_ref_pos | bad_ref_ori | bad_motion_body_pos

        if motion_command.motion.has_object:
            bad_object_pos = self.bad_object_pos(motion_command)
            bad_object_ori = self.bad_object_ori(motion_command)
            bad_tracking |= bad_object_pos | bad_object_ori

        if motion_command.use_adaptive_timesteps_sampler and torch.any(bad_tracking):
            failed_at_time_step = motion_command.time_steps[bad_tracking]
            motion_command.adaptive_timesteps_sampler.update_current_bin_failed_count(failed_at_time_step)

        return bad_tracking

    def bad_ref_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference position is too far from the robot's position."""
        return torch.norm(motion_command.ref_pos_w - motion_command.robot_ref_pos_w, dim=1) > self.bad_ref_pos_threshold

    def bad_ref_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference orientation is too far from the robot's orientation."""
        motion_projected_gravity_b = quat_rotate_inverse(
            motion_command.ref_quat_w, gravity_vector(self.env), w_last=True
        )
        robot_projected_gravity_b = quat_rotate_inverse(
            motion_command.robot_ref_quat_w, gravity_vector(self.env), w_last=True
        )
        return (
            torch.abs(motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]) > self.bad_ref_ori_threshold
        )

    def bad_motion_body_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the motion body position is too far from the robot's body position."""
        body_idx = self.bad_motion_body_pos_body_indexes
        error = torch.norm(
            motion_command.body_pos_relative_w[:, body_idx] - motion_command.robot_body_pos_w[:, body_idx], dim=-1
        )
        return torch.any(error > self.bad_motion_body_pos_threshold, dim=-1)

    def bad_object_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the object position is too far from the simulator's object position."""
        return (
            torch.norm(motion_command.object_pos_w - motion_command.simulator_object_pos_w, dim=-1)
            > self.bad_object_pos_threshold
        )

    def bad_object_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the object orientation is too far from the simulator's object orientation."""
        return (
            quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w)
            > self.bad_object_ori_threshold
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal state for specified environments."""

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
