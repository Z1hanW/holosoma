"""Whole Body Tracking-specific termination terms."""

from __future__ import annotations

import math
import os
from typing import Any, List

from holosoma.config_types.termination import TerminationTermCfg
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.terms.wbt import gravity_vector
from holosoma.managers.termination.base import TerminationTermBase
from holosoma.utils.rotations import (
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


def drop_task_base_height_below_threshold(env, min_height: float = 0.45) -> torch.Tensor:
    """Terminate when the robot base collapses below a fixed height."""
    return env.simulator.robot_root_states[:, 2] < min_height


class RobotFallenByTiltAfterIteration(TerminationTermBase):
    """Terminate on large base tilt, optionally only after DAgger ends."""

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.max_tilt_deg = float(cfg.params.get("max_tilt_deg", 60.0))
        if not (0.0 < self.max_tilt_deg < 180.0):
            raise ValueError(f"max_tilt_deg must be in (0, 180), got {self.max_tilt_deg}.")
        self.max_projected_gravity_xy = math.sin(math.radians(self.max_tilt_deg))
        self.hold_steps = max(1, int(cfg.params.get("hold_steps", 1)))
        self.apply_during_evaluation = bool(cfg.params.get("apply_during_evaluation", True))
        self.enable_after_iteration = self._resolve_enable_after_iteration(cfg)
        self._failure_counter = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.env.device)

    def _resolve_enable_after_iteration(self, cfg: TerminationTermCfg) -> int:
        explicit = cfg.params.get("enable_after_iteration")
        env_var_name = cfg.params.get("enable_after_iteration_env_var")
        if env_var_name is not None:
            env_var_name = str(env_var_name).strip()
            if env_var_name:
                env_value = os.environ.get(env_var_name)
                if env_value is not None and env_value.strip():
                    try:
                        return max(0, int(env_value))
                    except ValueError:
                        pass
        if explicit is not None:
            return max(0, int(explicit))
        return 0

    def _is_enabled(self) -> bool:
        if self.env.is_evaluating:
            return self.apply_during_evaluation
        motion_command = self.env.command_manager.get_state("motion_command")
        current_iteration = getattr(motion_command, "_training_iteration", 0)
        if current_iteration is None:
            current_iteration = 0
        return int(current_iteration) >= self.enable_after_iteration

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        if not self._is_enabled():
            self._failure_counter.zero_()
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        projected_gravity_b = quat_rotate_inverse(self.env.base_quat, gravity_vector(self.env), w_last=True)
        tilt_exceeded = torch.linalg.norm(projected_gravity_b[:, :2], dim=-1) > self.max_projected_gravity_xy
        self._failure_counter = torch.where(
            tilt_exceeded,
            self._failure_counter + 1,
            torch.zeros_like(self._failure_counter),
        )
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

        if os.environ.get("HOLOSOMA_DISABLE_BAD_TRACKING_RESET", "0").lower() in ("1", "true", "yes", "on"):
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

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


class BadTrackingZOnly(BadTracking):
    """BadTracking variant using z-axis-only position checks."""

    def bad_ref_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        z_err = torch.abs(motion_command.ref_pos_w[:, -1] - motion_command.robot_ref_pos_w[:, -1])
        return z_err > self.bad_ref_pos_threshold

    def bad_motion_body_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        body_idx = self.bad_motion_body_pos_body_indexes
        error = torch.abs(
            motion_command.body_pos_relative_w[:, body_idx, -1] - motion_command.robot_body_pos_w[:, body_idx, -1]
        )
        return torch.any(error > self.bad_motion_body_pos_threshold, dim=-1)
