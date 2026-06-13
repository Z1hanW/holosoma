"""Whole body tracking observation terms."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from holosoma.managers.observation.base import ObservationTermBase
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.rotations import (
    calc_heading,
    calc_heading_quat_inv,
    get_euler_xyz,
    normalize_angle,
    quat_apply,
    quat_apply_broadcast_left,
    quat_inverse,
    quat_mul_broadcast_left,
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


class ActionsHistory(ObservationTermBase):
    """Fixed-length raw action history stored as a single observation term."""

    def __init__(self, cfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        history_steps = int(getattr(cfg, "params", {}).get("history_steps", 1))
        if history_steps < 1:
            raise ValueError(f"history_steps must be >= 1, got {history_steps}")
        self.history_steps = history_steps
        self.action_dim = int(env.action_manager.total_action_dim)
        self.device = env.device
        self._history = torch.zeros(
            (env.num_envs, self.history_steps, self.action_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self._last_episode_length = torch.full(
            (env.num_envs,),
            fill_value=-1,
            device=self.device,
            dtype=torch.long,
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._history.zero_()
            self._last_episode_length.fill_(-1)
            return
        if env_ids.numel() == 0:
            return
        self._history[env_ids] = 0.0
        self._last_episode_length[env_ids] = -1

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        episode_length = env.episode_length_buf
        update_mask = episode_length != self._last_episode_length
        if torch.any(update_mask):
            self._history[update_mask] = torch.roll(self._history[update_mask], shifts=-1, dims=1)
            self._history[update_mask, -1, :] = env.action_manager.action[update_mask]
            self._last_episode_length[update_mask] = episode_length[update_mask]
        return self._history.reshape(env.num_envs, -1)


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


def _get_sim_body_subset_indexes(
    env: WholeBodyTrackingManager,
    *,
    body_names: list[str] | tuple[str, ...] | None = None,
    body_name_pattern: str | None = None,
) -> torch.Tensor:
    return _get_cached_name_subset_indexes(
        env,
        cache_name="_wbt_obs_sim_body_subset_cache",
        all_names=list(env.simulator.body_names),  # type: ignore[attr-defined]
        names=body_names,
        pattern=body_name_pattern,
    )


def _get_selected_sim_body_names(
    env: WholeBodyTrackingManager,
    selected_indexes: torch.Tensor,
) -> list[str]:
    all_names = list(env.simulator.body_names)  # type: ignore[attr-defined]
    return [all_names[int(idx)] for idx in selected_indexes.detach().cpu().tolist()]


def _get_object_contact_force_history(
    env: WholeBodyTrackingManager,
    *,
    body_names: list[str] | tuple[str, ...] | None = None,
    body_name_pattern: str | None = None,
) -> torch.Tensor:
    selected_indexes = _get_sim_body_subset_indexes(
        env,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
    )
    selected_names = _get_selected_sim_body_names(env, selected_indexes)
    motion_command = env.command_manager.get_state("motion_command")
    if isinstance(motion_command, MotionCommand):
        return motion_command.get_body_object_contact_force_history(selected_names)

    getter = getattr(env.simulator, "get_object_contact_force_history", None)
    if getter is None:
        raise RuntimeError(
            f"Simulator '{type(env.simulator).__name__}' does not expose box-filtered contact forces. "
            "Privileged box-contact critic observations require backend support for box-specific contacts."
        )

    return getter(selected_names)


def _current_contact_force_subset(
    env: WholeBodyTrackingManager,
    *,
    body_names: list[str] | tuple[str, ...] | None = None,
    body_name_pattern: str | None = None,
    object_only: bool = False,
    non_object_only: bool = False,
) -> torch.Tensor:
    if object_only and non_object_only:
        raise ValueError("object_only and non_object_only cannot both be True.")

    body_indexes = _get_sim_body_subset_indexes(
        env,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
    )
    current_contact_forces = env.simulator.contact_forces_history[:, 0, body_indexes, :]
    if not object_only and not non_object_only:
        return current_contact_forces

    object_contact_forces = _get_object_contact_force_history(
        env,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
    )[:, 0, :, :]
    if object_only:
        return object_contact_forces
    return current_contact_forces - object_contact_forces


def _reduce_contact_magnitudes(magnitudes: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "max":
        return torch.max(magnitudes, dim=1)[0]
    if reduction == "mean":
        return torch.mean(magnitudes, dim=1)
    if reduction == "sum":
        return torch.sum(magnitudes, dim=1)
    raise ValueError(f"Unsupported reduction '{reduction}'. Use one of: max, mean, sum.")


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


def _contact_aware_segment_root_command(motion_command: MotionCommand) -> torch.Tensor:
    """Non-overlap carry-window segment command in the segment-start root heading frame."""
    if not motion_command.motion.has_object:
        return torch.zeros((motion_command.num_envs, 3), device=motion_command.device, dtype=torch.float32)

    segment_steps = int(getattr(motion_command.motion_cfg, "contact_aware_sparse_root_segment_steps", 30))
    if segment_steps < 1:
        raise ValueError(f"contact_aware_sparse_root_segment_steps must be >= 1, got {segment_steps}")

    clip_ids = motion_command.clip_ids
    time_steps = motion_command.time_steps
    clip_lengths = motion_command.current_clip_lengths
    carry_window_by_clip = motion_command._get_contact_aware_carry_window_by_clip()  # noqa: SLF001
    carry_start = carry_window_by_clip[clip_ids, 0]
    carry_end = carry_window_by_clip[clip_ids, 1]

    rel_steps = torch.clamp(time_steps - carry_start, min=0)
    segment_index = torch.div(rel_steps, segment_steps, rounding_mode="floor")
    segment_start = carry_start + segment_index * segment_steps
    segment_end = segment_start + segment_steps

    max_step = torch.clamp(clip_lengths - 1, min=0)
    safe_segment_start = torch.minimum(torch.clamp(segment_start, min=0), max_step)
    safe_segment_end = torch.minimum(torch.clamp(segment_end, min=0), max_step)
    start_motion_idx = motion_command._get_motion_indices(safe_segment_start)  # noqa: SLF001
    end_motion_idx = motion_command._get_motion_indices(safe_segment_end)  # noqa: SLF001

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

    yaw_threshold_deg = float(getattr(motion_command.motion_cfg, "contact_aware_sparse_root_zero_yaw_threshold_deg", 0.0))
    if yaw_threshold_deg > 0.0:
        yaw_threshold_rad = yaw_threshold_deg * _DEG_TO_RAD
        rel_yaw = torch.where(torch.abs(rel_yaw) <= yaw_threshold_rad, torch.zeros_like(rel_yaw), rel_yaw)

    command = torch.cat([rel_xy, rel_yaw], dim=-1)
    valid_endpoint = (segment_end < carry_end) & (segment_end < clip_lengths)
    active_mask = (time_steps >= carry_start) & (time_steps < carry_end) & valid_endpoint
    return torch.where(active_mask.unsqueeze(-1), command, torch.zeros_like(command))


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


def _first_sustained_true_index(mask: torch.Tensor, consecutive_steps: int) -> int | None:
    """Return the earliest index where `mask` stays true for `consecutive_steps` frames."""
    if mask.numel() == 0:
        return None
    if consecutive_steps <= 1:
        true_indices = torch.nonzero(mask, as_tuple=False)
        if true_indices.numel() == 0:
            return None
        return int(true_indices[0, 0].item())

    run_length = 0
    for idx, flag in enumerate(mask.detach().cpu().tolist()):
        run_length = run_length + 1 if flag else 0
        if run_length >= consecutive_steps:
            return idx - consecutive_steps + 1
    return None


def _clip_pickup_goal_xy_yaw_root_heading(
    motion_command: MotionCommand,
    *,
    lift_height_threshold: float,
    lift_ratio_threshold: float,
    consecutive_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache per-clip final object goal in the pickup-time root-heading frame."""
    height_key = f"{lift_height_threshold:.4f}".replace(".", "p")
    ratio_key = f"{lift_ratio_threshold:.4f}".replace(".", "p")
    cache_name = (
        "_clip_pickup_goal_xy_yaw_root_heading_"
        f"h{height_key}_r{ratio_key}_c{consecutive_steps:d}"
    )
    cached = getattr(motion_command, cache_name, None)
    if cached is not None:
        return cached

    num_clips = motion_command.motion.num_clips
    goal_xy_yaw_by_clip = torch.zeros((num_clips, 3), device=motion_command.device, dtype=torch.float32)
    pickup_steps_by_clip = torch.zeros((num_clips,), device=motion_command.device, dtype=torch.long)

    if not motion_command.motion.has_object:
        cached = (goal_xy_yaw_by_clip, pickup_steps_by_clip)
        setattr(motion_command, cache_name, cached)
        return cached

    clip_offsets = motion_command.motion.clip_offsets
    clip_lengths = motion_command.motion.clip_lengths
    root_pos_w = motion_command.motion.body_pos_w[:, 0]
    root_quat_w = motion_command.motion.body_quat_w[:, 0]
    object_pos_w = motion_command.motion.object_pos_w
    object_quat_w = motion_command.motion.object_quat_w

    for clip_idx in range(num_clips):
        clip_start = int(clip_offsets[clip_idx].item())
        clip_length = int(clip_lengths[clip_idx].item())
        if clip_length <= 0:
            continue

        clip_end = clip_start + clip_length
        clip_root_pos_w = root_pos_w[clip_start:clip_end]
        clip_root_quat_w = root_quat_w[clip_start:clip_end]
        clip_object_pos_w = object_pos_w[clip_start:clip_end]
        clip_object_quat_w = object_quat_w[clip_start:clip_end]

        clip_heading_inv = calc_heading_quat_inv(clip_root_quat_w, w_last=True)
        clip_object_pos_heading = quat_apply(clip_heading_inv, clip_object_pos_w - clip_root_pos_w, w_last=True)
        clip_object_z = clip_object_pos_heading[:, 2]

        z_min = clip_object_z.min()
        z_range = torch.clamp(clip_object_z.max() - z_min, min=0.0)
        pickup_threshold = z_min + torch.maximum(
            z_min.new_tensor(float(lift_height_threshold)),
            z_range * float(lift_ratio_threshold),
        )

        lifted_mask = clip_object_z >= pickup_threshold
        pickup_step = _first_sustained_true_index(lifted_mask, consecutive_steps)
        if pickup_step is None:
            lifted_indices = torch.nonzero(lifted_mask, as_tuple=False)
            if lifted_indices.numel() > 0:
                pickup_step = int(lifted_indices[0, 0].item())
            else:
                pickup_step = int(torch.argmax(clip_object_z).item())
        pickup_steps_by_clip[clip_idx] = pickup_step

        anchor_root_pos_w = clip_root_pos_w[pickup_step : pickup_step + 1]
        anchor_root_quat_w = clip_root_quat_w[pickup_step : pickup_step + 1]
        goal_object_pos_w = clip_object_pos_w[-1:]
        goal_object_quat_w = clip_object_quat_w[-1:]

        anchor_heading_inv = calc_heading_quat_inv(anchor_root_quat_w, w_last=True)
        goal_pos_heading = quat_apply(anchor_heading_inv, goal_object_pos_w - anchor_root_pos_w, w_last=True)
        goal_heading = calc_heading(goal_object_quat_w)
        anchor_heading = calc_heading(anchor_root_quat_w)
        goal_xy_yaw_by_clip[clip_idx, :2] = goal_pos_heading[0, :2]
        goal_xy_yaw_by_clip[clip_idx, 2] = normalize_angle(goal_heading - anchor_heading)[0]

    cached = (goal_xy_yaw_by_clip, pickup_steps_by_clip)
    setattr(motion_command, cache_name, cached)
    return cached


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


def sparse_target_root_trajectory_command_contact_aware(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Sparse root command that is active only during the clip's carry phase."""
    motion_command = _get_motion_command_and_assert_type(env)
    base_command = sparse_target_root_trajectory_command(env)
    if getattr(motion_command, "manual_control_enabled", False) or not motion_command.motion.has_object:
        return base_command

    command_mode = (
        str(getattr(motion_command.motion_cfg, "contact_aware_sparse_root_command_mode", "tracking_error"))
        .strip()
        .lower()
        .replace("-", "_")
    )
    if command_mode in {"t1_aligned_segment", "segment", "segment_30"}:
        return _contact_aware_segment_root_command(motion_command)
    if command_mode not in {"tracking_error", "tracking", "default", "robot_tracking_error"}:
        raise ValueError(f"Unsupported contact-aware sparse root command mode: {command_mode!r}")

    active_mask = motion_command.get_contact_aware_root_command_active_mask()
    return torch.where(active_mask.unsqueeze(-1), base_command, torch.zeros_like(base_command))


def drop_button(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Binary drop button: 0 before carry-end t2, 1 from t2 to clip end."""
    motion_command = _get_motion_command_and_assert_type(env)
    manual_drop_button = getattr(motion_command, "manual_drop_button", None)
    if getattr(motion_command, "manual_drop_button_override_enabled", False) and manual_drop_button is not None:
        if manual_drop_button.device != env.device:
            manual_drop_button = manual_drop_button.to(env.device)
        return torch.clamp(manual_drop_button.to(dtype=torch.float32), 0.0, 1.0)
    if getattr(motion_command, "manual_control_enabled", False) or not motion_command.motion.has_object:
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    return motion_command.get_contact_aware_drop_button().to(dtype=torch.float32).unsqueeze(-1)


def pickup_button(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Binary pickup button: 1 before carry-start t1, 0 from t1 to clip end."""
    motion_command = _get_motion_command_and_assert_type(env)
    manual_pickup_button = getattr(motion_command, "manual_pickup_button", None)
    if getattr(motion_command, "manual_pickup_button_override_enabled", False) and manual_pickup_button is not None:
        if manual_pickup_button.device != env.device:
            manual_pickup_button = manual_pickup_button.to(env.device)
        return torch.clamp(manual_pickup_button.to(dtype=torch.float32), 0.0, 1.0)
    if getattr(motion_command, "manual_control_enabled", False) or not motion_command.motion.has_object:
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    return motion_command.get_contact_aware_pickup_button().to(dtype=torch.float32).unsqueeze(-1)


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

    ref_quat_inv = quat_inverse(motion_command.robot_ref_quat_w, w_last=True)
    pos_b = quat_apply_broadcast_left(
        ref_quat_inv,
        motion_command.robot_body_pos_w - motion_command.robot_ref_pos_w[:, None, :],
        w_last=True,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)

    ref_quat_inv = quat_inverse(motion_command.robot_ref_quat_w, w_last=True)
    ori_b = quat_mul_broadcast_left(
        ref_quat_inv,
        motion_command.robot_body_quat_w,
        w_last=True,
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
    """Target object position in robot-ref frame."""
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
    """Target object orientation in robot-ref frame as rot6d."""
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
    """Active object size from motion metadata."""
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


def obj_goal_xy_yaw_pick_root_heading(
    env: WholeBodyTrackingManager,
    lift_height_threshold: float = 0.10,
    lift_ratio_threshold: float = 0.35,
    consecutive_steps: int = 5,
) -> torch.Tensor:
    """Final clip object [dx, dy, dyaw] in the pickup-time root-heading frame.

    The pickup anchor is derived per clip from the earliest sustained object lift in the
    mocap reference, using the object height expressed in the clip's root-heading frame.
    """
    motion_command = _get_motion_command_and_assert_type(env)
    if not motion_command.motion.has_object:
        return torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)

    goal_xy_yaw_by_clip, _ = _clip_pickup_goal_xy_yaw_root_heading(
        motion_command,
        lift_height_threshold=lift_height_threshold,
        lift_ratio_threshold=lift_ratio_threshold,
        consecutive_steps=consecutive_steps,
    )
    return goal_xy_yaw_by_clip[motion_command.clip_ids]


def obj_goal_xy_pick_root_heading(
    env: WholeBodyTrackingManager,
    lift_height_threshold: float = 0.10,
    lift_ratio_threshold: float = 0.35,
    consecutive_steps: int = 5,
) -> torch.Tensor:
    """Final clip object [dx, dy] in the pickup-time root-heading frame."""
    motion_command = _get_motion_command_and_assert_type(env)
    if not motion_command.motion.has_object:
        return torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32)

    goal_xy_yaw_by_clip, _ = _clip_pickup_goal_xy_yaw_root_heading(
        motion_command,
        lift_height_threshold=lift_height_threshold,
        lift_ratio_threshold=lift_ratio_threshold,
        consecutive_steps=consecutive_steps,
    )
    return goal_xy_yaw_by_clip[motion_command.clip_ids, :2]


def obj_picked_flag(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Legacy mixed object-goal flag backed by the current pickup anchor state."""
    motion_command = _get_motion_command_and_assert_type(env)
    pickup_anchor_set = getattr(motion_command, "pickup_anchor_set", None)
    if pickup_anchor_set is None:
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    return pickup_anchor_set.to(device=env.device, dtype=torch.float32).unsqueeze(-1)


def _legacy_false_flag(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Compatibility flag for removed eval modes that are always inactive now."""
    return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)


def contact_prior_confidence(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, _, _, confidence, valid_mask = motion_command.get_contact_prior_targets()
    confidence = confidence * valid_mask.to(dtype=torch.float32)
    return confidence.unsqueeze(-1)


def contact_prior_region_occupancy(env: WholeBodyTrackingManager, region_name: str) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    occupancy, _, _, _, valid_mask = motion_command.get_contact_prior_region_targets(region_name)
    return occupancy * valid_mask


def contact_prior_region_force(env: WholeBodyTrackingManager, region_name: str) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, force, _, _, valid_mask = motion_command.get_contact_prior_region_targets(region_name)
    return force * valid_mask


def contact_prior_region_pos_obj(env: WholeBodyTrackingManager, region_name: str) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, _, position, _, valid_mask = motion_command.get_contact_prior_region_targets(region_name)
    return position * valid_mask


def body_contact_force_magnitude(
    env: WholeBodyTrackingManager,
    body_names: list[str] | tuple[str, ...] | None = None,
    body_name_pattern: str | None = None,
    object_only: bool = False,
    non_object_only: bool = False,
    reduction: str = "max",
) -> torch.Tensor:
    """Current-step contact force magnitude aggregated over the selected bodies."""
    contact_forces = _current_contact_force_subset(
        env,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
        object_only=object_only,
        non_object_only=non_object_only,
    )
    magnitudes = torch.norm(contact_forces, dim=-1)
    return _reduce_contact_magnitudes(magnitudes, reduction).unsqueeze(-1)


def body_contact_binary_flag(
    env: WholeBodyTrackingManager,
    threshold: float = 1.0,
    body_names: list[str] | tuple[str, ...] | None = None,
    body_name_pattern: str | None = None,
    object_only: bool = False,
    non_object_only: bool = False,
    reduction: str = "max",
) -> torch.Tensor:
    """Binary contact flag aggregated over the selected bodies."""
    magnitude = body_contact_force_magnitude(
        env,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
        object_only=object_only,
        non_object_only=non_object_only,
        reduction=reduction,
    )
    return (magnitude > threshold).to(dtype=torch.float32)
