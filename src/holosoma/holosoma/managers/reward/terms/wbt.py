"""Reward terms for Whole Body Tracking tasks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, List

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.command.terms.wbt import (
    MotionCommand,
    _CONTACT_PRIOR_REGION_FORCE_BODY_NAMES,
    _CONTACT_PRIOR_REGION_NAMES,
    _CONTACT_PRIOR_REGION_POSITION_BODY_NAMES,
    _normalize_contact_prior_region_name,
)
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.rotations import (
    quat_apply,
    quat_error_magnitude,
    quat_inverse,
    quat_mul,
    yaw_quat,
)

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


_CONTACT_EXPORT_LABEL_BY_REGION = {
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
    # Backward-compatible config aliases. New configs should use left_wrist/right_wrist.
    "left_palm": "left_wrist",
    "right_palm": "right_wrist",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_wrist_roll": "left_wrist_roll",
    "right_wrist_roll": "right_wrist_roll",
    "left_wrist_pitch": "left_wrist_pitch",
    "right_wrist_pitch": "right_wrist_pitch",
    "torso": "torso",
}


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
    selected_names = [env.simulator.body_names[int(idx)] for idx in selected_indexes.detach().cpu().tolist()]  # type: ignore[attr-defined]
    motion_command = env.command_manager.get_state("motion_command")
    if isinstance(motion_command, MotionCommand):
        return motion_command.get_body_object_contact_force_history(selected_names)

    getter = getattr(env.simulator, "get_object_contact_force_history", None)
    if getter is None:
        raise RuntimeError(
            f"Simulator '{type(env.simulator).__name__}' does not expose box-filtered contact forces. "
            "Object-only contact rewards/penalties require backend support for box-specific contacts."
        )

    return getter(selected_names)


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
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.ref_pos_w - motion_command.robot_ref_pos_w), dim=-1)
    reward = torch.exp(-error / sigma**2)
    return reward


def motion_global_ref_orientation_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.ref_quat_w, motion_command.robot_ref_quat_w) ** 2
    reward = torch.exp(-error / sigma**2)
    return reward


def _resolve_teacher_rollout_reference_root(raw_root: str) -> Path | None:
    root_str = raw_root.strip()
    if not root_str:
        return None
    root = Path(root_str).expanduser()
    if (root / "clips").is_dir():
        root = root / "clips"
    try:
        return root.resolve()
    except Exception:
        return root


def _infer_clip_id_from_dir_name(dir_name: str) -> str:
    if "_" not in dir_name:
        return dir_name.strip()
    return dir_name.split("_", 1)[1].strip()


def _gather_clip_timestep_values(values: torch.Tensor, clip_indices: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
    per_env_values = values.index_select(0, clip_indices)
    trailing_shape = tuple(per_env_values.shape[2:])
    gather_index = time_steps.view(per_env_values.shape[0], 1, *([1] * len(trailing_shape))).expand(
        per_env_values.shape[0], 1, *trailing_shape
    )
    return torch.gather(per_env_values, 1, gather_index).squeeze(1)


def _get_teacher_rollout_reference_bank(
    env: WholeBodyTrackingManager,
    motion_command: MotionCommand,
    *,
    rollout_reference_root: str,
) -> dict[str, torch.Tensor] | None:
    resolved_root = _resolve_teacher_rollout_reference_root(rollout_reference_root)
    if resolved_root is None:
        return None

    cache = getattr(env, "_teacher_rollout_reference_bank_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_teacher_rollout_reference_bank_cache", cache)

    cache_key = str(resolved_root)
    expected_clip_ids = tuple(str(clip_id) for clip_id in motion_command.motion.clip_ids)
    expected_body_names = tuple(str(name) for name in motion_command.motion_cfg.body_names_to_track)
    expected_ref_name = str(motion_command.motion_cfg.body_name_ref[0])

    cached_entry = cache.get(cache_key)
    if (
        cached_entry is not None
        and cached_entry.get("clip_ids") == expected_clip_ids
        and cached_entry.get("body_names") == expected_body_names
        and cached_entry.get("ref_name") == expected_ref_name
    ):
        return cached_entry.get("bank")

    if not resolved_root.is_dir():
        logger.warning(
            "Teacher rollout reference tracking disabled: rollout reference root '{}' does not exist.",
            resolved_root,
        )
        cache[cache_key] = {
            "clip_ids": expected_clip_ids,
            "body_names": expected_body_names,
            "ref_name": expected_ref_name,
            "bank": None,
        }
        return None

    num_clips = int(motion_command.motion.num_clips)
    num_bodies = len(expected_body_names)
    clip_name_to_index = {clip_name: idx for idx, clip_name in enumerate(motion_command.motion.clip_ids)}

    clip_payloads: dict[int, dict[str, np.ndarray]] = {}
    max_steps = 0
    has_any_object = False

    for clip_dir in sorted(resolved_root.iterdir()):
        if not clip_dir.is_dir():
            continue
        rollout_path = clip_dir / "teacher_rollout_reference.npz"
        if not rollout_path.is_file():
            continue
        try:
            with np.load(rollout_path, allow_pickle=False) as data:
                clip_id = ""
                if "clip_id" in data.files:
                    clip_id = str(np.asarray(data["clip_id"]).item()).strip()
                if not clip_id:
                    metadata_path = clip_dir / "metadata.json"
                    if metadata_path.is_file():
                        try:
                            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                            clip_id = str(metadata.get("clip_id", "")).strip()
                        except Exception:
                            clip_id = ""
                if not clip_id:
                    clip_id = _infer_clip_id_from_dir_name(clip_dir.name)
                if not clip_id or clip_id not in clip_name_to_index:
                    continue

                if "tracked_body_names" in data.files:
                    loaded_body_names = tuple(str(name) for name in np.asarray(data["tracked_body_names"]).tolist())
                    if loaded_body_names != expected_body_names:
                        logger.warning(
                            "Skipping rollout reference '{}' because tracked_body_names do not match training bodies.",
                            rollout_path,
                        )
                        continue
                if "ref_body_name" in data.files:
                    loaded_ref_name = str(np.asarray(data["ref_body_name"]).item())
                    if loaded_ref_name != expected_ref_name:
                        logger.warning(
                            "Skipping rollout reference '{}' because ref_body_name '{}' != '{}'.",
                            rollout_path,
                            loaded_ref_name,
                            expected_ref_name,
                        )
                        continue

                valid_steps = np.asarray(data["valid_steps"], dtype=np.bool_).reshape(-1)
                if valid_steps.size == 0:
                    continue
                body_pos_local = np.asarray(data["body_pos_local"], dtype=np.float32).reshape(valid_steps.size, num_bodies, 3)
                body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32).reshape(valid_steps.size, num_bodies, 4)
                body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float32).reshape(valid_steps.size, num_bodies, 3)
                body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float32).reshape(valid_steps.size, num_bodies, 3)
                ref_pos_local = np.asarray(data["ref_pos_local"], dtype=np.float32).reshape(valid_steps.size, 3)
                ref_quat_w = np.asarray(data["ref_quat_w"], dtype=np.float32).reshape(valid_steps.size, 4)
                ref_lin_vel_w = np.asarray(data["ref_lin_vel_w"], dtype=np.float32).reshape(valid_steps.size, 3)
                ref_ang_vel_w = np.asarray(data["ref_ang_vel_w"], dtype=np.float32).reshape(valid_steps.size, 3)
                root_pos_local = np.asarray(data["root_pos_local"], dtype=np.float32).reshape(valid_steps.size, 3)
                root_quat_w = np.asarray(data["root_quat_w"], dtype=np.float32).reshape(valid_steps.size, 4)
                root_lin_vel_w = np.asarray(data["root_lin_vel_w"], dtype=np.float32).reshape(valid_steps.size, 3)
                root_ang_vel_w = np.asarray(data["root_ang_vel_w"], dtype=np.float32).reshape(valid_steps.size, 3)

                payload: dict[str, np.ndarray] = {
                    "valid_steps": valid_steps,
                    "body_pos_local": body_pos_local,
                    "body_quat_w": body_quat_w,
                    "body_lin_vel_w": body_lin_vel_w,
                    "body_ang_vel_w": body_ang_vel_w,
                    "ref_pos_local": ref_pos_local,
                    "ref_quat_w": ref_quat_w,
                    "ref_lin_vel_w": ref_lin_vel_w,
                    "ref_ang_vel_w": ref_ang_vel_w,
                    "root_pos_local": root_pos_local,
                    "root_quat_w": root_quat_w,
                    "root_lin_vel_w": root_lin_vel_w,
                    "root_ang_vel_w": root_ang_vel_w,
                }
                if "object_pos_local" in data.files and "object_quat_w" in data.files:
                    payload["object_pos_local"] = np.asarray(data["object_pos_local"], dtype=np.float32).reshape(valid_steps.size, 3)
                    payload["object_quat_w"] = np.asarray(data["object_quat_w"], dtype=np.float32).reshape(valid_steps.size, 4)
                    payload["object_lin_vel_w"] = np.asarray(data["object_lin_vel_w"], dtype=np.float32).reshape(valid_steps.size, 3)
                    payload["object_ang_vel_w"] = np.asarray(data["object_ang_vel_w"], dtype=np.float32).reshape(valid_steps.size, 3)
                    has_any_object = True

                clip_payloads[int(clip_name_to_index[clip_id])] = payload
                max_steps = max(max_steps, int(valid_steps.size))
        except Exception as exc:
            logger.warning("Skipping invalid teacher rollout reference '{}': {}", rollout_path, exc)

    if max_steps <= 0:
        logger.warning(
            "Teacher rollout reference tracking disabled: no matching rollout references found in '{}'.",
            resolved_root,
        )
        cache[cache_key] = {
            "clip_ids": expected_clip_ids,
            "body_names": expected_body_names,
            "ref_name": expected_ref_name,
            "bank": None,
        }
        return None

    def _zeros(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(shape, device=env.device, dtype=torch.float32)

    bank: dict[str, torch.Tensor] = {
        "has_clip": torch.zeros((num_clips,), device=env.device, dtype=torch.bool),
        "lengths": torch.zeros((num_clips,), device=env.device, dtype=torch.long),
        "valid_steps": torch.zeros((num_clips, max_steps), device=env.device, dtype=torch.bool),
        "body_pos_local": _zeros((num_clips, max_steps, num_bodies, 3)),
        "body_quat_w": _zeros((num_clips, max_steps, num_bodies, 4)),
        "body_lin_vel_w": _zeros((num_clips, max_steps, num_bodies, 3)),
        "body_ang_vel_w": _zeros((num_clips, max_steps, num_bodies, 3)),
        "ref_pos_local": _zeros((num_clips, max_steps, 3)),
        "ref_quat_w": _zeros((num_clips, max_steps, 4)),
        "ref_lin_vel_w": _zeros((num_clips, max_steps, 3)),
        "ref_ang_vel_w": _zeros((num_clips, max_steps, 3)),
        "root_pos_local": _zeros((num_clips, max_steps, 3)),
        "root_quat_w": _zeros((num_clips, max_steps, 4)),
        "root_lin_vel_w": _zeros((num_clips, max_steps, 3)),
        "root_ang_vel_w": _zeros((num_clips, max_steps, 3)),
        "has_object": torch.zeros((num_clips,), device=env.device, dtype=torch.bool),
        "object_pos_local": _zeros((num_clips, max_steps, 3)),
        "object_quat_w": _zeros((num_clips, max_steps, 4)),
        "object_lin_vel_w": _zeros((num_clips, max_steps, 3)),
        "object_ang_vel_w": _zeros((num_clips, max_steps, 3)),
    }
    bank["body_quat_w"][..., 3] = 1.0
    bank["ref_quat_w"][..., 3] = 1.0
    bank["root_quat_w"][..., 3] = 1.0
    bank["object_quat_w"][..., 3] = 1.0

    for clip_index, payload in clip_payloads.items():
        step_count = int(payload["valid_steps"].shape[0])
        bank["has_clip"][clip_index] = True
        bank["lengths"][clip_index] = step_count
        bank["valid_steps"][clip_index, :step_count] = torch.as_tensor(
            payload["valid_steps"], device=env.device, dtype=torch.bool
        )
        for key in (
            "body_pos_local",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "ref_pos_local",
            "ref_quat_w",
            "ref_lin_vel_w",
            "ref_ang_vel_w",
            "root_pos_local",
            "root_quat_w",
            "root_lin_vel_w",
            "root_ang_vel_w",
        ):
            bank[key][clip_index, :step_count] = torch.as_tensor(payload[key], device=env.device, dtype=torch.float32)
        if "object_pos_local" in payload:
            bank["has_object"][clip_index] = True
            for key in ("object_pos_local", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w"):
                bank[key][clip_index, :step_count] = torch.as_tensor(payload[key], device=env.device, dtype=torch.float32)

    matched_clip_count = int(bank["has_clip"].sum().item())
    logger.info(
        "Teacher rollout reference tracking loaded {} clip(s) from '{}'. has_object={}",
        matched_clip_count,
        resolved_root,
        has_any_object,
    )
    cache[cache_key] = {
        "clip_ids": expected_clip_ids,
        "body_names": expected_body_names,
        "ref_name": expected_ref_name,
        "bank": bank,
    }
    return bank


def _sample_teacher_rollout_reference(
    env: WholeBodyTrackingManager,
    motion_command: MotionCommand,
    *,
    rollout_reference_root: str,
) -> dict[str, torch.Tensor] | None:
    bank = _get_teacher_rollout_reference_bank(
        env,
        motion_command,
        rollout_reference_root=rollout_reference_root,
    )
    if bank is None:
        return None

    clip_indices = motion_command.clip_ids.to(device=env.device, dtype=torch.long)
    lengths = bank["lengths"].index_select(0, clip_indices)
    has_clip = bank["has_clip"].index_select(0, clip_indices)
    raw_steps = motion_command.time_steps.to(device=env.device, dtype=torch.long)
    safe_steps = torch.minimum(raw_steps.clamp_min(0), (lengths - 1).clamp_min(0))
    sampled_valid = _gather_clip_timestep_values(bank["valid_steps"], clip_indices, safe_steps).to(dtype=torch.bool)
    valid_mask = has_clip & (lengths > 0) & (raw_steps >= 0) & (raw_steps < lengths) & sampled_valid

    env_offsets = motion_command._get_env_offsets().to(device=env.device, dtype=torch.float32)
    body_pos_local = _gather_clip_timestep_values(bank["body_pos_local"], clip_indices, safe_steps)
    ref_pos_local = _gather_clip_timestep_values(bank["ref_pos_local"], clip_indices, safe_steps)
    root_pos_local = _gather_clip_timestep_values(bank["root_pos_local"], clip_indices, safe_steps)
    has_object = bank["has_object"].index_select(0, clip_indices)

    sampled = {
        "valid_mask": valid_mask,
        "object_valid_mask": valid_mask & has_object,
        "body_pos_w": body_pos_local + env_offsets[:, None, :],
        "body_quat_w": _gather_clip_timestep_values(bank["body_quat_w"], clip_indices, safe_steps),
        "body_lin_vel_w": _gather_clip_timestep_values(bank["body_lin_vel_w"], clip_indices, safe_steps),
        "body_ang_vel_w": _gather_clip_timestep_values(bank["body_ang_vel_w"], clip_indices, safe_steps),
        "ref_pos_w": ref_pos_local + env_offsets,
        "ref_quat_w": _gather_clip_timestep_values(bank["ref_quat_w"], clip_indices, safe_steps),
        "ref_lin_vel_w": _gather_clip_timestep_values(bank["ref_lin_vel_w"], clip_indices, safe_steps),
        "ref_ang_vel_w": _gather_clip_timestep_values(bank["ref_ang_vel_w"], clip_indices, safe_steps),
        "root_pos_w": root_pos_local + env_offsets,
        "root_quat_w": _gather_clip_timestep_values(bank["root_quat_w"], clip_indices, safe_steps),
        "root_lin_vel_w": _gather_clip_timestep_values(bank["root_lin_vel_w"], clip_indices, safe_steps),
        "root_ang_vel_w": _gather_clip_timestep_values(bank["root_ang_vel_w"], clip_indices, safe_steps),
        "object_pos_w": _gather_clip_timestep_values(bank["object_pos_local"], clip_indices, safe_steps) + env_offsets,
        "object_quat_w": _gather_clip_timestep_values(bank["object_quat_w"], clip_indices, safe_steps),
        "object_lin_vel_w": _gather_clip_timestep_values(bank["object_lin_vel_w"], clip_indices, safe_steps),
        "object_ang_vel_w": _gather_clip_timestep_values(bank["object_ang_vel_w"], clip_indices, safe_steps),
    }
    return sampled


def _teacher_rollout_relative_body_targets(
    env: WholeBodyTrackingManager,
    motion_command: MotionCommand,
    sampled_reference: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    num_bodies = int(sampled_reference["body_pos_w"].shape[1])
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if episode_length_buf is None:
        episode_length_buf = torch.ones((motion_command.num_envs,), device=motion_command.device, dtype=torch.long)
    use_root = (episode_length_buf == 0).to(device=env.device, dtype=torch.float32).unsqueeze(1)

    ref_pos_w = sampled_reference["root_pos_w"] * use_root + sampled_reference["ref_pos_w"] * (1.0 - use_root)
    ref_quat_w = sampled_reference["root_quat_w"] * use_root + sampled_reference["ref_quat_w"] * (1.0 - use_root)
    robot_ref_pos_w = motion_command.robot_root_pos_w * use_root + motion_command.robot_ref_pos_w * (1.0 - use_root)
    robot_ref_quat_w = motion_command.robot_root_quat_w * use_root + motion_command.robot_ref_quat_w * (1.0 - use_root)

    ref_pos_w_repeat = ref_pos_w[:, None, :].repeat(1, num_bodies, 1)
    ref_quat_w_repeat = ref_quat_w[:, None, :].repeat(1, num_bodies, 1)
    robot_ref_pos_w_repeat = robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1)
    robot_ref_quat_w_repeat = robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1)

    delta_quat_w = yaw_quat(
        quat_mul(robot_ref_quat_w_repeat, quat_inverse(ref_quat_w_repeat, w_last=True), w_last=True),
        w_last=True,
    )
    relative_body_quat_w = quat_mul(delta_quat_w, sampled_reference["body_quat_w"], w_last=True)
    delta_pos_w_height = ref_pos_w_repeat - robot_ref_pos_w_repeat
    delta_pos_w_height[..., :2] = 0.0
    relative_body_pos_w = (
        robot_ref_pos_w_repeat
        + delta_pos_w_height
        + quat_apply(delta_quat_w, sampled_reference["body_pos_w"] - ref_pos_w_repeat, w_last=True)
    )
    return relative_body_pos_w, relative_body_quat_w


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
    reward = torch.exp(-error.mean(-1) / sigma**2)
    return reward


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
    reward = torch.exp(-error.mean(-1) / sigma**2)
    return reward


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
    reward = torch.exp(-error.mean(-1) / sigma**2)
    return reward


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
    reward = torch.exp(-error.mean(-1) / sigma**2)
    return reward


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
    reward = torch.exp(-error.mean(-1) / sigma**2)
    return reward


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
    reward = torch.exp(-error.mean(-1) / sigma**2)
    return reward


# ================================================================================================
# Object Tracking Rewards
# ================================================================================================


def object_global_ref_position_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.object_pos_w - motion_command.simulator_object_pos_w), dim=-1)
    reward = torch.exp(-error / sigma**2)
    return reward


def object_global_ref_orientation_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w) ** 2
    reward = torch.exp(-error / sigma**2)
    return reward


def teacher_rollout_global_ref_position_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    rollout_reference_root: str = "outputs/clips",
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    sampled_reference = _sample_teacher_rollout_reference(
        env,
        motion_command,
        rollout_reference_root=rollout_reference_root,
    )
    if sampled_reference is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    error = torch.sum(torch.square(sampled_reference["ref_pos_w"] - motion_command.robot_ref_pos_w), dim=-1)
    reward = torch.exp(-error / sigma**2)
    return reward * sampled_reference["valid_mask"].to(dtype=torch.float32)


def teacher_rollout_global_ref_orientation_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    rollout_reference_root: str = "outputs/clips",
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    sampled_reference = _sample_teacher_rollout_reference(
        env,
        motion_command,
        rollout_reference_root=rollout_reference_root,
    )
    if sampled_reference is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    error = quat_error_magnitude(sampled_reference["ref_quat_w"], motion_command.robot_ref_quat_w) ** 2
    reward = torch.exp(-error / sigma**2)
    return reward * sampled_reference["valid_mask"].to(dtype=torch.float32)


def teacher_rollout_relative_body_position_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    rollout_reference_root: str = "outputs/clips",
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    sampled_reference = _sample_teacher_rollout_reference(
        env,
        motion_command,
        rollout_reference_root=rollout_reference_root,
    )
    if sampled_reference is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    relative_body_pos_w, _ = _teacher_rollout_relative_body_targets(env, motion_command, sampled_reference)
    error = torch.sum(torch.square(relative_body_pos_w - motion_command.robot_body_pos_w), dim=-1)
    body_indexes = _get_tracked_body_subset_indexes(
        env,
        motion_command,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
    )
    error = error.index_select(1, body_indexes)
    reward = torch.exp(-error.mean(-1) / sigma**2)
    return reward * sampled_reference["valid_mask"].to(dtype=torch.float32)


def teacher_rollout_relative_body_orientation_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    rollout_reference_root: str = "outputs/clips",
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    sampled_reference = _sample_teacher_rollout_reference(
        env,
        motion_command,
        rollout_reference_root=rollout_reference_root,
    )
    if sampled_reference is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    _, relative_body_quat_w = _teacher_rollout_relative_body_targets(env, motion_command, sampled_reference)
    error = quat_error_magnitude(relative_body_quat_w, motion_command.robot_body_quat_w) ** 2
    body_indexes = _get_tracked_body_subset_indexes(
        env,
        motion_command,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
    )
    error = error.index_select(1, body_indexes)
    reward = torch.exp(-error.mean(-1) / sigma**2)
    return reward * sampled_reference["valid_mask"].to(dtype=torch.float32)


def teacher_rollout_global_body_lin_vel(
    env: WholeBodyTrackingManager,
    sigma: float,
    rollout_reference_root: str = "outputs/clips",
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    sampled_reference = _sample_teacher_rollout_reference(
        env,
        motion_command,
        rollout_reference_root=rollout_reference_root,
    )
    if sampled_reference is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    error = torch.sum(torch.square(sampled_reference["body_lin_vel_w"] - motion_command.robot_body_lin_vel_w), dim=-1)
    body_indexes = _get_tracked_body_subset_indexes(
        env,
        motion_command,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
    )
    error = error.index_select(1, body_indexes)
    reward = torch.exp(-error.mean(-1) / sigma**2)
    return reward * sampled_reference["valid_mask"].to(dtype=torch.float32)


def teacher_rollout_global_body_ang_vel(
    env: WholeBodyTrackingManager,
    sigma: float,
    rollout_reference_root: str = "outputs/clips",
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    sampled_reference = _sample_teacher_rollout_reference(
        env,
        motion_command,
        rollout_reference_root=rollout_reference_root,
    )
    if sampled_reference is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    error = torch.sum(torch.square(sampled_reference["body_ang_vel_w"] - motion_command.robot_body_ang_vel_w), dim=-1)
    body_indexes = _get_tracked_body_subset_indexes(
        env,
        motion_command,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
    )
    error = error.index_select(1, body_indexes)
    reward = torch.exp(-error.mean(-1) / sigma**2)
    return reward * sampled_reference["valid_mask"].to(dtype=torch.float32)


def teacher_rollout_object_global_ref_position_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    rollout_reference_root: str = "outputs/clips",
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    sampled_reference = _sample_teacher_rollout_reference(
        env,
        motion_command,
        rollout_reference_root=rollout_reference_root,
    )
    if sampled_reference is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    error = torch.sum(torch.square(sampled_reference["object_pos_w"] - motion_command.simulator_object_pos_w), dim=-1)
    reward = torch.exp(-error / sigma**2)
    return reward * sampled_reference["object_valid_mask"].to(dtype=torch.float32)


def teacher_rollout_object_global_ref_orientation_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    rollout_reference_root: str = "outputs/clips",
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    sampled_reference = _sample_teacher_rollout_reference(
        env,
        motion_command,
        rollout_reference_root=rollout_reference_root,
    )
    if sampled_reference is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    error = quat_error_magnitude(sampled_reference["object_quat_w"], motion_command.simulator_object_quat_w) ** 2
    reward = torch.exp(-error / sigma**2)
    return reward * sampled_reference["object_valid_mask"].to(dtype=torch.float32)


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


def body_object_contact_reward(
    env: WholeBodyTrackingManager,
    threshold: float = 1.0,
    force_scale: float = 25.0,
    reward_mode: str = "binary",
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
) -> torch.Tensor:
    object_contact_forces = _get_object_contact_force_history(
        env,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
    )
    magnitudes = torch.norm(object_contact_forces, dim=-1)
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


class OfflineContactPointGuidance(RewardTermBase):
    """Guide interaction with object-frame contact-point priors exported from teacher rollouts.

    The reward follows the structure:
      position_term * force_term
    averaged over all active end-effectors/regions for the current clip.

    Contact targets are loaded per clip from a teacher contact export directory, using
    region-specific point clouds such as ``left_wrist_contact_points.npy``.
    """

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.motion_command: MotionCommand | None = None
        self.position_sigma = float(cfg.params.get("position_sigma", 0.05))
        self.force_threshold = float(cfg.params.get("force_threshold", 1.7))
        self.force_sigma = float(cfg.params.get("force_sigma", 10.0))
        self.use_force_term = bool(cfg.params.get("use_force_term", True))
        self.force_gate_mode = str(cfg.params.get("force_gate_mode", "binary")).strip().lower()
        if self.force_gate_mode not in {"soft", "binary"}:
            raise ValueError(
                f"Unsupported force_gate_mode '{self.force_gate_mode}'. Expected one of: ['soft', 'binary']."
            )
        self.use_contact_schedule = bool(cfg.params.get("use_contact_schedule", False))
        self.contact_schedule_relax_steps = max(int(cfg.params.get("contact_schedule_relax_steps", 0)), 0)
        self.contact_schedule_missing_mode = str(cfg.params.get("contact_schedule_missing_mode", "always_on")).strip().lower()
        if self.contact_schedule_missing_mode not in {"always_on", "after_pickup", "inactive"}:
            raise ValueError(
                "Unsupported contact_schedule_missing_mode "
                f"'{self.contact_schedule_missing_mode}'. Expected one of: ['always_on', 'after_pickup', 'inactive']."
            )
        self.require_stable_contact = bool(cfg.params.get("require_stable_contact", True))
        self.min_target_points = max(int(cfg.params.get("min_target_points", 1)), 1)
        self.region_names = self._normalize_region_names(cfg.params.get("region_names"))
        self._region_force_body_names: dict[str, list[str]] = {}
        self._region_position_body_names: dict[str, list[str]] = {}
        self._region_position_body_indices: dict[str, torch.Tensor] = {}
        self._configure_region_measurements()

        self.contact_export_root = self._resolve_contact_export_root(str(cfg.params.get("contact_export_root", "")))
        self._clip_region_points: torch.Tensor | None = None
        self._clip_region_point_mask: torch.Tensor | None = None
        self._clip_region_has_targets: torch.Tensor | None = None
        self._clip_region_contact_intervals: torch.Tensor | None = None
        self._clip_region_has_contact_intervals: torch.Tensor | None = None
        self._clip_region_contact_schedule: torch.Tensor | None = None
        self._clip_region_contact_schedule_lengths: torch.Tensor | None = None
        self._clip_region_has_contact_schedule: torch.Tensor | None = None
        self._contact_bank_available = False
        self._contact_bank_initialized = False

    @staticmethod
    def _normalize_region_names(raw_region_names: object) -> list[str]:
        if raw_region_names is None:
            return list(_CONTACT_PRIOR_REGION_NAMES)
        if isinstance(raw_region_names, str):
            raw_names = [raw_region_names]
        else:
            raw_names = [str(name) for name in raw_region_names]
        region_names: list[str] = []
        seen_region_names: set[str] = set()
        for raw_name in raw_names:
            region_name = _normalize_contact_prior_region_name(raw_name)
            if region_name and region_name not in seen_region_names:
                region_names.append(region_name)
                seen_region_names.add(region_name)
        invalid = [name for name in region_names if name not in _CONTACT_PRIOR_REGION_NAMES]
        if invalid:
            raise ValueError(
                f"Unsupported region_names {invalid}. Expected subset of {list(_CONTACT_PRIOR_REGION_NAMES)}."
            )
        if not region_names:
            raise ValueError("region_names must not be empty.")
        return region_names

    @staticmethod
    def _resolve_contact_export_root(raw_root: str) -> Path | None:
        root_str = raw_root.strip()
        if not root_str:
            return None
        root = Path(root_str).expanduser()
        if (root / "clips").is_dir():
            root = root / "clips"
        try:
            return root.resolve()
        except Exception:
            return root

    def _configure_region_measurements(self) -> None:
        simulator_body_names = list(getattr(self.env.simulator, "body_names", []))
        body_name_to_index = {name: idx for idx, name in enumerate(simulator_body_names)}
        for region_name in self.region_names:
            force_names = [
                body_name
                for body_name in _CONTACT_PRIOR_REGION_FORCE_BODY_NAMES.get(region_name, ())
                if body_name in body_name_to_index
            ]
            position_names = [
                body_name
                for body_name in _CONTACT_PRIOR_REGION_POSITION_BODY_NAMES.get(region_name, ())
                if body_name in body_name_to_index
            ]
            self._region_force_body_names[region_name] = force_names
            self._region_position_body_names[region_name] = position_names
            self._region_position_body_indices[region_name] = torch.tensor(
                [body_name_to_index[name] for name in position_names],
                device=self.env.device,
                dtype=torch.long,
            )
            if not force_names or not position_names:
                logger.warning(
                    "OfflineContactPointGuidance region '{}' is partially unavailable. force_bodies={} position_bodies={}",
                    region_name,
                    force_names,
                    position_names,
                )

    def _ensure_motion_command_and_contact_bank(self) -> MotionCommand | None:
        if self.motion_command is None:
            if getattr(self.env, "command_manager", None) is None:
                return None
            try:
                motion_command = _get_motion_command_and_assert_type(self.env)
            except (AttributeError, AssertionError):
                return None
            self.motion_command = motion_command

        if not self._contact_bank_initialized:
            self._load_contact_bank()
            self._contact_bank_initialized = True

        return self.motion_command

    def _load_contact_bank(self) -> None:
        motion_command = self.motion_command
        if motion_command is None:
            raise RuntimeError("OfflineContactPointGuidance cannot load contact bank before motion_command is available.")

        num_clips = int(motion_command.motion.num_clips)
        num_regions = len(self.region_names)
        self._clip_region_points = torch.zeros((num_clips, num_regions, 1, 3), device=self.env.device, dtype=torch.float32)
        self._clip_region_point_mask = torch.zeros((num_clips, num_regions, 1), device=self.env.device, dtype=torch.bool)
        self._clip_region_has_targets = torch.zeros((num_clips, num_regions), device=self.env.device, dtype=torch.bool)
        self._clip_region_contact_intervals = torch.full(
            (num_clips, num_regions, 2), -1, device=self.env.device, dtype=torch.long
        )
        self._clip_region_has_contact_intervals = torch.zeros(
            (num_clips, num_regions), device=self.env.device, dtype=torch.bool
        )
        self._clip_region_contact_schedule = torch.zeros(
            (num_clips, num_regions, 1), device=self.env.device, dtype=torch.bool
        )
        self._clip_region_contact_schedule_lengths = torch.zeros(
            (num_clips, num_regions), device=self.env.device, dtype=torch.long
        )
        self._clip_region_has_contact_schedule = torch.zeros(
            (num_clips, num_regions), device=self.env.device, dtype=torch.bool
        )
        self._contact_bank_available = False

        if self.contact_export_root is None:
            logger.warning("OfflineContactPointGuidance disabled: contact_export_root is empty.")
            return
        if not self.contact_export_root.is_dir():
            logger.warning(
                "OfflineContactPointGuidance disabled: contact export root '{}' does not exist.",
                self.contact_export_root,
            )
            return

        clip_name_to_index = {clip_name: idx for idx, clip_name in enumerate(motion_command.motion.clip_ids)}
        clip_region_points: dict[tuple[int, int], np.ndarray] = {}
        clip_region_intervals: dict[tuple[int, int], tuple[int, int]] = {}
        clip_region_schedules: dict[tuple[int, int], np.ndarray] = {}
        max_points = 0
        max_schedule_steps = 0

        for clip_dir in sorted(self.contact_export_root.iterdir()):
            if not clip_dir.is_dir():
                continue
            metadata_path = clip_dir / "metadata.json"
            metadata: dict[str, object] = {}
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.warning("Skipping invalid contact metadata '{}': {}", metadata_path, exc)
                    continue

            clip_id = str(metadata.get("clip_id", "")).strip()
            if not clip_id:
                clip_id = self._infer_clip_id_from_dir_name(clip_dir.name)
            if not clip_id or clip_id not in clip_name_to_index:
                continue
            if self.require_stable_contact and metadata and not bool(metadata.get("stable_contact_success", False)):
                continue

            clip_index = clip_name_to_index[clip_id]
            for region_idx, region_name in enumerate(self.region_names):
                export_label = _CONTACT_EXPORT_LABEL_BY_REGION[region_name]
                points_path = clip_dir / f"{export_label}_contact_points.npy"
                if not points_path.is_file():
                    continue
                try:
                    points = np.asarray(np.load(points_path), dtype=np.float32).reshape(-1, 3)
                except Exception as exc:
                    logger.warning("Skipping invalid contact point cloud '{}': {}", points_path, exc)
                    continue
                if points.shape[0] < self.min_target_points:
                    continue
                clip_region_points[(clip_index, region_idx)] = points
                max_points = max(max_points, int(points.shape[0]))
                interval_path = clip_dir / f"{export_label}_contact_interval_steps.npy"
                if interval_path.is_file():
                    try:
                        interval_steps = np.asarray(np.load(interval_path), dtype=np.int64).reshape(-1)
                    except Exception as exc:
                        logger.warning("Skipping invalid contact interval '{}': {}", interval_path, exc)
                    else:
                        if interval_steps.size >= 2:
                            start_step = int(interval_steps[0])
                            end_step = int(interval_steps[1])
                            if start_step >= 0 and end_step > start_step:
                                clip_region_intervals[(clip_index, region_idx)] = (start_step, end_step)
                schedule_path = clip_dir / f"{export_label}_contact_active_mask.npy"
                if schedule_path.is_file():
                    try:
                        contact_schedule = np.asarray(np.load(schedule_path), dtype=np.bool_).reshape(-1)
                    except Exception as exc:
                        logger.warning("Skipping invalid contact schedule '{}': {}", schedule_path, exc)
                    else:
                        if contact_schedule.size > 0:
                            clip_region_schedules[(clip_index, region_idx)] = contact_schedule
                            max_schedule_steps = max(max_schedule_steps, int(contact_schedule.shape[0]))

        if max_points <= 0:
            logger.warning(
                "OfflineContactPointGuidance disabled: no matching region contact targets found in '{}'.",
                self.contact_export_root,
            )
            return

        points_tensor = torch.zeros(
            (num_clips, num_regions, max_points, 3),
            device=self.env.device,
            dtype=torch.float32,
        )
        point_mask = torch.zeros(
            (num_clips, num_regions, max_points),
            device=self.env.device,
            dtype=torch.bool,
        )
        has_targets = torch.zeros((num_clips, num_regions), device=self.env.device, dtype=torch.bool)
        for (clip_index, region_idx), points in clip_region_points.items():
            tensor = torch.as_tensor(points, device=self.env.device, dtype=torch.float32)
            point_count = int(tensor.shape[0])
            points_tensor[clip_index, region_idx, :point_count] = tensor
            point_mask[clip_index, region_idx, :point_count] = True
            has_targets[clip_index, region_idx] = True

        self._clip_region_points = points_tensor
        self._clip_region_point_mask = point_mask
        self._clip_region_has_targets = has_targets
        contact_intervals = torch.full((num_clips, num_regions, 2), -1, device=self.env.device, dtype=torch.long)
        has_intervals = torch.zeros((num_clips, num_regions), device=self.env.device, dtype=torch.bool)
        for key, interval in clip_region_intervals.items():
            clip_index, region_idx = key
            start_step, end_step = interval
            contact_intervals[clip_index, region_idx, 0] = int(start_step)
            contact_intervals[clip_index, region_idx, 1] = int(end_step)
            has_intervals[clip_index, region_idx] = True
        self._clip_region_contact_intervals = contact_intervals
        self._clip_region_has_contact_intervals = has_intervals
        if max_schedule_steps > 0:
            schedule_tensor = torch.zeros(
                (num_clips, num_regions, max_schedule_steps),
                device=self.env.device,
                dtype=torch.bool,
            )
            schedule_lengths = torch.zeros((num_clips, num_regions), device=self.env.device, dtype=torch.long)
            has_schedule = torch.zeros((num_clips, num_regions), device=self.env.device, dtype=torch.bool)
            for (clip_index, region_idx), schedule in clip_region_schedules.items():
                tensor = torch.as_tensor(schedule, device=self.env.device, dtype=torch.bool)
                step_count = int(tensor.shape[0])
                schedule_tensor[clip_index, region_idx, :step_count] = tensor
                schedule_lengths[clip_index, region_idx] = step_count
                has_schedule[clip_index, region_idx] = True
            self._clip_region_contact_schedule = schedule_tensor
            self._clip_region_contact_schedule_lengths = schedule_lengths
            self._clip_region_has_contact_schedule = has_schedule
        self._contact_bank_available = bool(has_targets.any().item())
        if self._contact_bank_available:
            matched_clip_count = int(has_targets.any(dim=1).sum().item())
            logger.info(
                "OfflineContactPointGuidance loaded {} clip(s) with targets from '{}'.",
                matched_clip_count,
                self.contact_export_root,
            )

    @staticmethod
    def _infer_clip_id_from_dir_name(dir_name: str) -> str:
        if "_" not in dir_name:
            return dir_name.strip()
        return dir_name.split("_", 1)[1].strip()

    def _compute_current_region_measurements(self) -> tuple[torch.Tensor, torch.Tensor]:
        num_envs = self.env.num_envs
        num_regions = len(self.region_names)
        current_force = torch.zeros((num_envs, num_regions), device=self.env.device, dtype=torch.float32)
        current_position = torch.zeros((num_envs, num_regions, 3), device=self.env.device, dtype=torch.float32)

        if not self.motion_command.motion.has_object:
            return current_force, current_position

        for region_idx, region_name in enumerate(self.region_names):
            force_body_names = self._region_force_body_names.get(region_name, [])
            if force_body_names:
                force_history = self.motion_command.get_body_object_contact_force_history(force_body_names)
                if force_history.shape[2] > 0:
                    per_body_force = torch.max(torch.norm(force_history, dim=-1), dim=1)[0]
                    current_force[:, region_idx] = torch.max(per_body_force, dim=1)[0]

            position_body_indices = self._region_position_body_indices.get(region_name)
            if position_body_indices is None or position_body_indices.numel() == 0:
                continue

            relative_positions = self.motion_command._body_positions_in_object_frame(position_body_indices)
            position_body_names = self._region_position_body_names.get(region_name, [])
            if position_body_names:
                position_force_history = self.motion_command.get_body_object_contact_force_history(position_body_names)
                if position_force_history.shape[2] == relative_positions.shape[1]:
                    position_force_weights = torch.max(torch.norm(position_force_history, dim=-1), dim=1)[0]
                else:
                    position_force_weights = torch.zeros(
                        (num_envs, relative_positions.shape[1]),
                        device=self.env.device,
                        dtype=torch.float32,
                    )
            else:
                position_force_weights = torch.zeros(
                    (num_envs, relative_positions.shape[1]),
                    device=self.env.device,
                    dtype=torch.float32,
                )

            uniform_weights = torch.full_like(
                position_force_weights,
                1.0 / float(max(relative_positions.shape[1], 1)),
            )
            weight_denom = position_force_weights.sum(dim=1, keepdim=True)
            normalized_weights = torch.where(
                weight_denom > 1.0e-6,
                position_force_weights / weight_denom.clamp_min(1.0e-6),
                uniform_weights,
            )
            current_position[:, region_idx] = torch.sum(relative_positions * normalized_weights.unsqueeze(-1), dim=1)

        return current_force, current_position

    def _reference_contact_schedule_mask(self, clip_indices: torch.Tensor) -> torch.Tensor:
        num_envs = int(clip_indices.shape[0])
        num_regions = len(self.region_names)
        schedule_active = torch.ones((num_envs, num_regions), device=self.env.device, dtype=torch.bool)
        if not self.use_contact_schedule:
            return schedule_active

        relax_steps = int(self.contact_schedule_relax_steps)

        interval_bank = self._clip_region_contact_intervals
        has_interval_bank = self._clip_region_has_contact_intervals
        if interval_bank is not None and has_interval_bank is not None and bool(has_interval_bank.any().item()):
            interval_steps = interval_bank.index_select(0, clip_indices)
            has_intervals = has_interval_bank.index_select(0, clip_indices)
            current_steps = self.motion_command.time_steps.to(device=self.env.device, dtype=torch.long).unsqueeze(-1)
            current_steps = current_steps.expand(-1, num_regions)
            start_steps = interval_steps[..., 0]
            end_steps = interval_steps[..., 1]
            if relax_steps > 0:
                start_steps = (start_steps - relax_steps).clamp_min(0)
                end_steps = end_steps + relax_steps
            interval_active = (current_steps >= start_steps) & (current_steps < end_steps)
            schedule_active = torch.where(has_intervals, interval_active, schedule_active)
            missing_schedule = ~has_intervals
        else:
            missing_schedule = torch.ones_like(schedule_active, dtype=torch.bool)

        has_schedule_bank = self._clip_region_has_contact_schedule
        schedule_bank = self._clip_region_contact_schedule
        schedule_lengths_bank = self._clip_region_contact_schedule_lengths
        if (
            has_schedule_bank is not None
            and schedule_bank is not None
            and schedule_lengths_bank is not None
            and bool(has_schedule_bank.any().item())
        ):
            has_schedule = has_schedule_bank.index_select(0, clip_indices)
            schedule_lengths = schedule_lengths_bank.index_select(0, clip_indices)
            current_steps = self.motion_command.time_steps.to(device=self.env.device, dtype=torch.long).unsqueeze(-1)
            current_steps = current_steps.expand(-1, num_regions)
            selected_schedule = schedule_bank.index_select(0, clip_indices)
            if relax_steps > 0:
                schedule_window_size = 2 * relax_steps + 1
                relaxed_schedule = F.max_pool1d(
                    selected_schedule.to(dtype=torch.float32).reshape(-1, 1, selected_schedule.shape[-1]),
                    kernel_size=schedule_window_size,
                    stride=1,
                    padding=relax_steps,
                ).reshape_as(selected_schedule) > 0.5
            else:
                relaxed_schedule = selected_schedule
            valid_schedule_steps = current_steps < schedule_lengths
            safe_schedule_steps = torch.minimum(current_steps, (schedule_lengths - 1).clamp_min(0))
            scheduled_values = torch.gather(relaxed_schedule, 2, safe_schedule_steps.unsqueeze(-1)).squeeze(-1)
            use_schedule = missing_schedule & has_schedule
            schedule_active = torch.where(use_schedule, valid_schedule_steps & scheduled_values, schedule_active)
            missing_schedule = missing_schedule & ~has_schedule

        if not missing_schedule.any():
            return schedule_active

        if self.contact_schedule_missing_mode == "always_on":
            return schedule_active
        if self.contact_schedule_missing_mode == "inactive":
            return torch.where(missing_schedule, torch.zeros_like(schedule_active), schedule_active)

        pickup_steps_getter = getattr(self.motion_command, "_get_clip_pickup_steps_by_clip", None)
        if callable(pickup_steps_getter):
            try:
                pickup_steps_by_clip = pickup_steps_getter().to(device=self.env.device, dtype=torch.long)
            except Exception:
                pickup_steps_by_clip = torch.zeros(
                    (int(self.motion_command.motion.num_clips),), device=self.env.device, dtype=torch.long
                )
        else:
            pickup_steps_by_clip = torch.zeros(
                (int(self.motion_command.motion.num_clips),), device=self.env.device, dtype=torch.long
            )
        current_steps = self.motion_command.time_steps.to(device=self.env.device, dtype=torch.long).unsqueeze(-1)
        current_steps = current_steps.expand(-1, num_regions)
        pickup_steps = pickup_steps_by_clip.index_select(0, clip_indices).unsqueeze(-1).expand(-1, num_regions)
        pickup_active = current_steps >= pickup_steps
        return torch.where(missing_schedule, pickup_active, schedule_active)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        motion_command = self._ensure_motion_command_and_contact_bank()
        if (
            motion_command is None
            or
            not self._contact_bank_available
            or self._clip_region_points is None
            or self._clip_region_point_mask is None
            or self._clip_region_has_targets is None
            or not motion_command.motion.has_object
        ):
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

        current_force, current_position = self._compute_current_region_measurements()
        clip_indices = motion_command.clip_ids.to(device=env.device, dtype=torch.long)
        target_points = self._clip_region_points.index_select(0, clip_indices)
        point_mask = self._clip_region_point_mask.index_select(0, clip_indices)
        active_regions = self._clip_region_has_targets.index_select(0, clip_indices)
        if self.use_contact_schedule:
            active_regions = active_regions & self._reference_contact_schedule_mask(clip_indices)
        if not active_regions.any():
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

        distances = torch.linalg.norm(target_points - current_position.unsqueeze(2), dim=-1)
        inf_fill = torch.full_like(distances, float("inf"))
        distances = torch.where(point_mask, distances, inf_fill)
        min_distance = distances.min(dim=-1).values

        position_term = torch.exp(-min_distance / max(self.position_sigma, 1.0e-6))
        if self.use_force_term:
            if self.force_gate_mode == "binary":
                force_term = (current_force >= self.force_threshold).to(dtype=torch.float32)
            else:
                force_term = torch.exp((current_force - self.force_threshold) / max(self.force_sigma, 1.0e-6)).clamp(
                    max=1.0
                )
            per_region_reward = position_term * force_term
        else:
            per_region_reward = position_term

        active_region_weights = active_regions.to(dtype=torch.float32)
        active_region_count = active_region_weights.sum(dim=1)
        reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        valid_env_mask = active_region_count > 0.0
        if valid_env_mask.any():
            reward[valid_env_mask] = (
                per_region_reward[valid_env_mask] * active_region_weights[valid_env_mask]
            ).sum(dim=1) / active_region_count[valid_env_mask].clamp_min(1.0)

        return reward


class _TerrainFootContactBase(RewardTermBase):
    """Compatibility shim for legacy terrain-contact reward terms stored in old checkpoints."""

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        self.contact_threshold = float(cfg.params.get("contact_threshold", 1.0))
        self.force_mode = str(cfg.params.get("force_mode", "binary")).lower()
        self.force_scale = float(cfg.params.get("force_scale", 25.0))
        # Legacy configs used ray_start_offset for explicit ray casts. We use a direct
        # terrain-height query instead, so this acts as the "good support" clearance.
        self.clearance_threshold = float(cfg.params.get("clearance_threshold", cfg.params.get("ray_start_offset", 0.25)))
        self.penalize_invalid = bool(cfg.params.get("penalize_invalid", True))

        self.contact_body_names = list(cfg.params.get("contact_body_names", []))
        self.query_body_names = list(cfg.params.get("query_body_names", []))
        if not self.contact_body_names:
            raise ValueError(f"{type(self).__name__} requires non-empty contact_body_names.")
        if not self.query_body_names:
            raise ValueError(f"{type(self).__name__} requires non-empty query_body_names.")

        self.contact_body_indexes = _get_sim_body_subset_indexes(env, body_names=self.contact_body_names)
        self.query_body_indexes = _get_sim_body_subset_indexes(env, body_names=self.query_body_names)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def _resolve_contact_signal(self) -> torch.Tensor:
        contact_forces = self.env.simulator.contact_forces_history[:, :, self.contact_body_indexes]
        magnitudes = torch.norm(contact_forces, dim=-1)
        peak_force = torch.max(magnitudes, dim=1)[0]

        if self.force_mode == "binary":
            return (peak_force > self.contact_threshold).to(dtype=torch.float32)
        if self.force_mode == "linear":
            return torch.clamp(
                (peak_force - self.contact_threshold) / max(self.force_scale, 1.0e-6),
                min=0.0,
                max=1.0,
            )
        if self.force_mode == "tanh":
            return torch.tanh(torch.clamp(peak_force - self.contact_threshold, min=0.0) / max(self.force_scale, 1.0e-6))
        raise ValueError(f"Unsupported force_mode '{self.force_mode}'. Use one of: binary, linear, tanh.")

    def _query_clearance(self) -> tuple[torch.Tensor, torch.Tensor]:
        terrain_state = self.env.terrain_manager.get_state("locomotion_terrain")
        query_positions = self.env.simulator._rigid_body_pos[:, self.query_body_indexes, :]
        flat_xy = query_positions[..., :2].reshape(-1, 2)
        terrain_heights = terrain_state.query_terrain_heights(flat_xy).reshape(query_positions.shape[0], query_positions.shape[1])
        clearance = query_positions[..., 2] - terrain_heights
        valid = torch.isfinite(clearance)
        return clearance, valid


class TerrainGreenFootContactReward(_TerrainFootContactBase):
    """Legacy reward: reward terrain contact only when the queried foot points are close to the ground."""

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        contact_signal = self._resolve_contact_signal()
        clearance, valid = self._query_clearance()
        is_green = (clearance <= self.clearance_threshold) & valid
        if self.contact_body_indexes.numel() == self.query_body_indexes.numel():
            per_body = contact_signal * is_green.to(dtype=torch.float32)
            return per_body.mean(dim=1)
        green_any = is_green.any(dim=1, keepdim=True).to(dtype=torch.float32)
        return (contact_signal * green_any).mean(dim=1)


class TerrainRedFootContactPenalty(_TerrainFootContactBase):
    """Legacy penalty: penalize terrain contact when the queried foot points are not near valid support."""

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        contact_signal = self._resolve_contact_signal()
        clearance, valid = self._query_clearance()
        invalid_support = clearance > self.clearance_threshold
        if self.penalize_invalid:
            invalid_support = invalid_support | (~valid)
        else:
            invalid_support = invalid_support & valid

        if self.contact_body_indexes.numel() == self.query_body_indexes.numel():
            per_body = contact_signal * invalid_support.to(dtype=torch.float32)
            return per_body.mean(dim=1)
        invalid_any = invalid_support.any(dim=1, keepdim=True).to(dtype=torch.float32)
        return (contact_signal * invalid_any).mean(dim=1)


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


class ObjectUndesiredContacts(RewardTermBase):
    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        self.threshold = cfg.params.get("threshold", 1.0)

        body_names = cfg.params.get("body_names")
        body_name_pattern = cfg.params.get("body_name_pattern")
        if body_name_pattern is None:
            body_name_pattern = cfg.params.get("undesired_contacts_body_names")

        all_body_names = list(self.env.simulator.body_names)  # type: ignore[attr-defined]
        if body_names is not None:
            missing = [name for name in body_names if name not in all_body_names]
            if missing:
                raise ValueError(f"Requested body names {missing} are not available in {all_body_names}.")
            self.body_names = list(body_names)
        elif body_name_pattern:
            regex = re.compile(body_name_pattern)
            self.body_names = [body_name for body_name in all_body_names if regex.match(body_name)]
        else:
            raise ValueError("ObjectUndesiredContacts requires either 'body_names' or 'body_name_pattern'.")

        if not self.body_names:
            raise ValueError("ObjectUndesiredContacts resolved an empty body-name set.")

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        object_contact_forces = _get_object_contact_force_history(env, body_names=self.body_names)
        is_contact = (torch.max(torch.norm(object_contact_forces, dim=-1), dim=1)[0] > self.threshold).to(
            dtype=torch.float32
        )
        return torch.sum(is_contact, dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass
