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


class BodyGroupProximity(TerminationTermBase):
    """Terminate when any configured pair of robot bodies gets too close."""

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.min_distance = float(cfg.params.get("min_distance", 0.05))
        if not math.isfinite(self.min_distance) or self.min_distance <= 0.0:
            raise ValueError("BodyGroupProximity min_distance must be finite and positive.")
        simulator_body_names = list(self.env.simulator.body_names)
        self._group_pair_indices: list[tuple[torch.Tensor, torch.Tensor]] = []
        for group_idx, body_group in enumerate(
            cfg.params.get(
                "body_groups",
                [["left_foot_contact_point", "right_foot_contact_point"]],
            )
        ):
            if len(body_group) < 2:
                raise ValueError(
                    f"body_groups[{group_idx}] must contain at least two body names."
                )
            missing = [name for name in body_group if name not in simulator_body_names]
            if missing:
                raise ValueError(
                    f"BodyGroupProximity bodies are missing from the simulator: {missing}."
                )
            body_indices = torch.tensor(
                [simulator_body_names.index(name) for name in body_group],
                device=self.env.device,
                dtype=torch.long,
            )
            pair_i, pair_j = torch.triu_indices(
                len(body_group), len(body_group), offset=1, device=self.env.device
            )
            self._group_pair_indices.append((body_indices[pair_i], body_indices[pair_j]))

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        del kwargs
        body_pos_w = self.env.simulator._rigid_body_pos
        terminated = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        for body_i, body_j in self._group_pair_indices:
            pairwise_dist = torch.linalg.vector_norm(
                body_pos_w[:, body_i, :] - body_pos_w[:, body_j, :], dim=-1
            )
            terminated |= torch.any(pairwise_dist < self.min_distance, dim=-1)
        return terminated


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
        self._last_component_results: dict[str, torch.Tensor] = {}

    def _set_component_results(self, **components: torch.Tensor) -> None:
        self._last_component_results = components

    def _set_empty_component_results(self, env: Any) -> torch.Tensor:
        zeros = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._set_component_results(
            robot_ref_position=zeros,
            robot_ref_orientation=zeros,
            robot_body_position=zeros,
            object_position=zeros,
            object_orientation=zeros,
        )
        return zeros

    def get_last_component_results(self) -> dict[str, torch.Tensor]:
        """Return threshold-condition masks from the latest evaluation."""

        return self._last_component_results

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        motion_command = self.env.command_manager.get_state("motion_command")
        assert motion_command.motion_cfg.body_names_to_track == self.body_names_to_track, (
            "body_names_to_track in motion_command and termination.params are not the same"
            f"motion_command.motion_cfg.body_names_to_track: {motion_command.motion_cfg.body_names_to_track}"
            f"termination.params['body_names_to_track']: {self.body_names_to_track}"
        )

        if os.environ.get("HOLOSOMA_DISABLE_BAD_TRACKING_RESET", "0").lower() in ("1", "true", "yes", "on"):
            return self._set_empty_component_results(env)

        # During evaluation, disable BadTracking-based termination entirely.
        if self.env.is_evaluating:
            return self._set_empty_component_results(env)

        bad_ref_pos = self.bad_ref_pos(motion_command)
        bad_ref_ori = self.bad_ref_ori(motion_command)
        bad_motion_body_pos = self.bad_motion_body_pos(motion_command)
        bad_object_pos = torch.zeros_like(bad_ref_pos)
        bad_object_ori = torch.zeros_like(bad_ref_pos)

        if motion_command.motion.has_object:
            bad_object_pos = self.bad_object_pos(motion_command)
            bad_object_ori = self.bad_object_ori(motion_command)

        self._set_component_results(
            robot_ref_position=bad_ref_pos,
            robot_ref_orientation=bad_ref_ori,
            robot_body_position=bad_motion_body_pos,
            object_position=bad_object_pos,
            object_orientation=bad_object_ori,
        )

        return bad_ref_pos | bad_ref_ori | bad_motion_body_pos | bad_object_pos | bad_object_ori

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
    """Legacy z-only robot checks with a full-XYZ object-position check."""

    def bad_ref_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        z_err = torch.abs(motion_command.ref_pos_w[:, -1] - motion_command.robot_ref_pos_w[:, -1])
        return z_err > self.bad_ref_pos_threshold

    def bad_motion_body_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        body_idx = self.bad_motion_body_pos_body_indexes
        error = torch.abs(
            motion_command.body_pos_relative_w[:, body_idx, -1] - motion_command.robot_body_pos_w[:, body_idx, -1]
        )
        return torch.any(error > self.bad_motion_body_pos_threshold, dim=-1)


class BadTrackingAllPositionZOnly(BadTrackingZOnly):
    """Use z-axis-only errors for robot, tracked bodies, and object position."""

    def bad_object_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        z_err = torch.abs(
            motion_command.object_pos_w[:, -1]
            - motion_command.simulator_object_pos_w[:, -1]
        )
        return z_err > self.bad_object_pos_threshold


class HybridStage2BadTracking(BadTracking):
    """Reference failures before pickup; fall/drop safety after task activation."""

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.task_object_hold_pos_threshold = float(
            cfg.params.get("task_object_hold_pos_threshold", 0.35)
        )
        if self.task_object_hold_pos_threshold <= 0.0:
            raise ValueError("task_object_hold_pos_threshold must be positive.")

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        reference_failure = super().__call__(env, **kwargs)
        motion_command = self.env.command_manager.get_state("motion_command")
        if not bool(getattr(motion_command.motion_cfg, "hybrid_stage2_enabled", False)):
            raise RuntimeError("HybridStage2BadTracking requires hybrid_stage2_enabled=True.")
        task_active = motion_command.get_hybrid_stage2_task_active_mask()
        if self.env.is_evaluating:
            return reference_failure

        # Once task mode is active, planar/reference-pose deviation is the
        # objective rather than a failure. Preserve the gravity-based fall
        # check and terminate only when the carried object leaves its latched
        # robot-relative pose by a clearly unsafe distance.
        object_pos_b = quat_rotate_inverse(
            motion_command.robot_root_quat_w,
            motion_command.simulator_object_pos_w - motion_command.robot_root_pos_w,
            w_last=True,
        )
        object_hold_error = torch.linalg.vector_norm(
            object_pos_b - motion_command.pickup_anchor_object_pos_b,
            dim=-1,
        )
        task_ref_orientation_condition = self.bad_ref_ori(motion_command)
        task_object_hold_condition = object_hold_error > self.task_object_hold_pos_threshold
        task_failure = task_ref_orientation_condition | task_object_hold_condition
        reference_mask = ~task_active
        reference_components = self.get_last_component_results()
        self._set_component_results(
            **{
                name: reference_mask & component
                for name, component in reference_components.items()
            },
            task_ref_orientation=task_active & task_ref_orientation_condition,
            task_object_hold=task_active & task_object_hold_condition,
        )
        return torch.where(task_active, task_failure, reference_failure)


class HybridVelocityBadTracking(BadTracking):
    """Reference termination for tracking rows and task safety for task rows."""

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.task_object_hold_pos_threshold = float(
            cfg.params.get("task_object_hold_pos_threshold", 0.35)
        )
        self.task_max_tilt_deg = float(cfg.params.get("task_max_tilt_deg", 60.0))
        if self.task_object_hold_pos_threshold <= 0.0:
            raise ValueError("task_object_hold_pos_threshold must be positive.")
        if not (0.0 < self.task_max_tilt_deg < 90.0):
            raise ValueError("task_max_tilt_deg must be in (0, 90).")
        self.task_max_projected_gravity_xy = math.sin(
            math.radians(self.task_max_tilt_deg)
        )

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        reference_failure = super().__call__(env, **kwargs)
        motion_command = self.env.command_manager.get_state("motion_command")
        if not motion_command.hybrid_velocity_enabled():
            raise RuntimeError("HybridVelocityBadTracking requires hybrid_velocity_enabled=True.")
        if self.env.is_evaluating:
            return reference_failure

        task_mask = motion_command.get_hybrid_velocity_task_env_mask()
        task_active = motion_command.get_hybrid_velocity_task_active_mask()
        projected_gravity_b = quat_rotate_inverse(
            motion_command.robot_root_quat_w,
            gravity_vector(self.env),
            w_last=True,
        )
        fallen = (
            torch.linalg.vector_norm(projected_gravity_b[:, :2], dim=-1)
            > self.task_max_projected_gravity_xy
        )

        object_pos_b = quat_rotate_inverse(
            motion_command.robot_root_quat_w,
            motion_command.simulator_object_pos_w - motion_command.robot_root_pos_w,
            w_last=True,
        )
        object_hold_error = torch.linalg.vector_norm(
            object_pos_b - motion_command.pickup_anchor_object_pos_b,
            dim=-1,
        )
        dropped = task_active & (
            object_hold_error > self.task_object_hold_pos_threshold
        )
        task_failure = fallen | dropped
        reference_components = self.get_last_component_results()
        self._set_component_results(
            **{
                name: ~task_mask & component
                for name, component in reference_components.items()
            },
            task_fallen=task_mask & fallen,
            task_object_dropped=task_mask & dropped,
        )
        return torch.where(task_mask, task_failure, reference_failure)


class HMIBadTracking(BadTracking):
    """Strict full-XYZ tracking rows plus HMI generation-row safety gates."""

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.min_root_height = float(cfg.params.get("min_root_height", 0.45))
        self.gen_bad_ref_pos_z_threshold = float(
            cfg.params.get("gen_bad_ref_pos_z_threshold", 0.40)
        )
        self.gen_bad_ref_pos_xyz_threshold = float(
            cfg.params.get("gen_bad_ref_pos_xyz_threshold", 100.0)
        )
        self.gen_bad_object_pos_z_threshold = float(
            cfg.params.get("gen_bad_object_pos_z_threshold", 0.60)
        )
        self.gen_bad_object_ori_threshold = float(
            cfg.params.get("gen_bad_object_ori_threshold", 1.0)
        )
        thresholds = (
            self.min_root_height,
            self.gen_bad_ref_pos_z_threshold,
            self.gen_bad_ref_pos_xyz_threshold,
            self.gen_bad_object_pos_z_threshold,
            self.gen_bad_object_ori_threshold,
        )
        if any(value <= 0.0 for value in thresholds):
            raise ValueError("HMI generation safety thresholds must be positive.")

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        reference_failure = super().__call__(env, **kwargs)
        motion_command = self.env.command_manager.get_state("motion_command")
        if not motion_command.hmi_enabled():
            raise RuntimeError("HMIBadTracking requires motion_config.hmi.")
        if self.env.is_evaluating:
            return reference_failure

        track_mask = motion_command.get_hmi_track_env_mask()
        gen_mask = motion_command.get_hmi_gen_env_mask()
        reference_components = self.get_last_component_results()
        track_failure = reference_failure & track_mask

        low_root_height = motion_command.robot_root_pos_w[:, 2] < self.min_root_height
        gen_ref_pos_z = (
            torch.abs(
                motion_command.ref_pos_w[:, 2]
                - motion_command.robot_ref_pos_w[:, 2]
            )
            > self.gen_bad_ref_pos_z_threshold
        )
        gen_ref_pos_xyz = (
            torch.linalg.vector_norm(
                motion_command.ref_pos_w - motion_command.robot_ref_pos_w,
                dim=-1,
            )
            > self.gen_bad_ref_pos_xyz_threshold
        )
        gen_object_pos_z = (
            torch.abs(
                motion_command.object_pos_w[:, 2]
                - motion_command.simulator_object_pos_w[:, 2]
            )
            > self.gen_bad_object_pos_z_threshold
        )
        gen_object_ori = (
            quat_error_magnitude(
                motion_command.object_quat_w,
                motion_command.simulator_object_quat_w,
            )
            > self.gen_bad_object_ori_threshold
        )
        gen_failure = gen_mask & (
            low_root_height
            | gen_ref_pos_z
            | gen_ref_pos_xyz
            | gen_object_pos_z
            | gen_object_ori
        )
        self._set_component_results(
            **{
                name: track_mask & component
                for name, component in reference_components.items()
            },
            hmi_gen_low_root_height=gen_mask & low_root_height,
            hmi_gen_ref_position_z=gen_mask & gen_ref_pos_z,
            hmi_gen_ref_position_xyz=gen_mask & gen_ref_pos_xyz,
            hmi_gen_object_position_z=gen_mask & gen_object_pos_z,
            hmi_gen_object_orientation=gen_mask & gen_object_ori,
        )
        return track_failure | gen_failure
