"""Reward terms for Whole Body Tracking tasks."""

from __future__ import annotations

import json
import os
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
    _convert_contact_interval_timebase,
    _normalize_contact_prior_region_name,
)
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.contact_intervals import (
    infer_contact_export_clip_id,
    resolve_contact_export_clip_id,
)
from holosoma.utils.rotations import (
    quat_apply_broadcast_left,
    quat_error_magnitude,
    quat_inverse,
    quat_mul,
    quat_mul_broadcast_left,
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


def _rollout_reference_uses_episodic_motion_end(env: WholeBodyTrackingManager) -> bool:
    """Whether the active environment terminates before sampling clip frame L-1."""

    if os.environ.get("HOLOSOMA_DISABLE_MOTION_END_RESET", "0").lower() in ("1", "true", "yes", "on"):
        return False
    manager = getattr(env, "termination_manager", None)
    term_names = tuple(str(name) for name in getattr(manager, "_term_names", ()))
    term_cfgs = tuple(getattr(manager, "_term_cfgs", ()))
    for index, term_name in enumerate(term_names):
        if index >= len(term_cfgs):
            # Lightweight test/dummy managers may expose only names.  The real
            # TerminationManager always carries aligned cfg entries.
            if term_name == "motion_ends":
                return True
            continue
        func = getattr(term_cfgs[index], "func", None)
        if isinstance(func, str):
            func_name = func.rsplit(":", 1)[-1]
        else:
            func_name = str(getattr(func, "__name__", ""))
        if func_name == "motion_ends":
            return True
    return False


def _required_rollout_reference_steps(clip_length: int, *, episodic_motion_end: bool) -> int:
    # BaseTask checks termination/reward before MotionCommand.step().  With
    # motion_end_mask >= L-2, episodic execution rewards indices 0..L-2;
    # continuing execution additionally rewards L-1 before clip rollover.
    return max(1, clip_length - 1) if episodic_motion_end else clip_length


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


def _get_sim_body_subset(
    env: WholeBodyTrackingManager,
    *,
    body_names: list[str] | tuple[str, ...] | None = None,
    body_name_pattern: str | None = None,
) -> tuple[tuple[str, ...], torch.Tensor]:
    """Resolve and cache simulator-body host names together with device indexes."""

    cache = getattr(env, "_wbt_reward_sim_body_subset_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_wbt_reward_sim_body_subset_cache", cache)

    key = (tuple(body_names) if body_names is not None else None, body_name_pattern)
    cached = cache.get(key)
    if cached is not None:
        return cached

    all_names = list(env.simulator.body_names)  # type: ignore[attr-defined]
    if body_names is not None:
        missing = [name for name in body_names if name not in all_names]
        if missing:
            raise ValueError(f"Requested names {missing} are not available in {all_names}.")
        indexes = [all_names.index(name) for name in body_names]
    elif body_name_pattern:
        regex = re.compile(body_name_pattern)
        indexes = [idx for idx, name in enumerate(all_names) if regex.match(name)]
    else:
        indexes = list(range(len(all_names)))

    if not indexes:
        raise ValueError(
            f"No names matched names={list(body_names) if body_names is not None else None} "
            f"pattern={body_name_pattern!r} in {all_names}."
        )

    selection = (
        tuple(all_names[index] for index in indexes),
        torch.tensor(indexes, dtype=torch.long, device=env.device),
    )
    cache[key] = selection
    return selection


def _get_sim_body_subset_indexes(
    env: WholeBodyTrackingManager,
    *,
    body_names: list[str] | tuple[str, ...] | None = None,
    body_name_pattern: str | None = None,
) -> torch.Tensor:
    _, selected_indexes = _get_sim_body_subset(
        env,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
    )
    return selected_indexes


def _get_object_contact_force_history(
    env: WholeBodyTrackingManager,
    *,
    body_names: list[str] | tuple[str, ...] | None = None,
    body_name_pattern: str | None = None,
) -> torch.Tensor:
    cached_names, _ = _get_sim_body_subset(
        env,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
    )
    # Preserve the previous downstream API (a fresh list on every call) while
    # avoiding the former device-to-host index transfer and synchronization.
    selected_names = list(cached_names)
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


def penalty_action_rate(
    env: WholeBodyTrackingManager,
    max_penalty: float | None = None,
) -> torch.Tensor:
    """Penalize changes in actions between steps.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    penalty = torch.sum(torch.square(prev_actions - actions), dim=1)
    if max_penalty is None or max_penalty <= 0.0:
        return penalty
    return torch.clamp(penalty, max=float(max_penalty))


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
    rollout_reference_root: str | None = None,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    if rollout_reference_root is None:
        reference_pos_w = motion_command.ref_pos_w
        valid_mask = None
    else:
        sampled_reference = _sample_rollout_reference(
            env,
            motion_command,
            rollout_reference_root=rollout_reference_root,
        )
        if sampled_reference is None:
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        reference_pos_w = sampled_reference["ref_pos_w"]
        valid_mask = sampled_reference["valid_mask"]

    error = torch.sum(torch.square(reference_pos_w - motion_command.robot_ref_pos_w), dim=-1)
    reward = torch.exp(-error / sigma**2)
    if valid_mask is not None:
        reward = reward * valid_mask.to(dtype=torch.float32)
    return reward


def motion_global_ref_orientation_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    rollout_reference_root: str | None = None,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    if rollout_reference_root is None:
        reference_quat_w = motion_command.ref_quat_w
        valid_mask = None
    else:
        sampled_reference = _sample_rollout_reference(
            env,
            motion_command,
            rollout_reference_root=rollout_reference_root,
        )
        if sampled_reference is None:
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        reference_quat_w = sampled_reference["ref_quat_w"]
        valid_mask = sampled_reference["valid_mask"]

    error = quat_error_magnitude(reference_quat_w, motion_command.robot_ref_quat_w) ** 2
    reward = torch.exp(-error / sigma**2)
    if valid_mask is not None:
        reward = reward * valid_mask.to(dtype=torch.float32)
    return reward


def _resolve_rollout_reference_root(raw_root: str) -> Path | None:
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
    return infer_contact_export_clip_id(dir_name)


def _gather_clip_timestep_values(values: torch.Tensor, clip_indices: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
    per_env_values = values.index_select(0, clip_indices)
    trailing_shape = tuple(per_env_values.shape[2:])
    gather_index = time_steps.view(per_env_values.shape[0], 1, *([1] * len(trailing_shape))).expand(
        per_env_values.shape[0], 1, *trailing_shape
    )
    return torch.gather(per_env_values, 1, gather_index).squeeze(1)


def _tensor_step_cache_signature(tensor: torch.Tensor) -> tuple[int, int, int, tuple[int, ...], str]:
    return (
        int(tensor.data_ptr()),
        int(getattr(tensor, "_version", 0)),
        int(tensor.numel()),
        tuple(int(dim) for dim in tensor.shape),
        str(tensor.device),
    )


def _rollout_reference_sample_cache_key(
    env: WholeBodyTrackingManager,
    motion_command: MotionCommand,
    *,
    rollout_reference_root: str,
) -> tuple[object, ...]:
    resolved_root = _resolve_rollout_reference_root(rollout_reference_root)
    root_key = "" if resolved_root is None else str(resolved_root)
    reward_compute_counter = getattr(env, "_reward_compute_counter", None)
    if reward_compute_counter is not None:
        return ("reward_compute", root_key, id(motion_command), int(reward_compute_counter))
    return (
        "tensor_signature",
        root_key,
        id(motion_command),
        _tensor_step_cache_signature(motion_command.clip_ids),
        _tensor_step_cache_signature(motion_command.time_steps),
    )


def _get_rollout_reference_bank(
    env: WholeBodyTrackingManager,
    motion_command: MotionCommand,
    *,
    rollout_reference_root: str,
) -> dict[str, torch.Tensor] | None:
    resolved_root = _resolve_rollout_reference_root(rollout_reference_root)
    if resolved_root is None:
        raise ValueError("rollout_reference_root must be a non-empty path when rollout-reference rewards are enabled.")

    cache = getattr(env, "_rollout_reference_bank_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_rollout_reference_bank_cache", cache)

    cache_key = str(resolved_root)
    episodic_motion_end = _rollout_reference_uses_episodic_motion_end(env)
    expected_clip_ids = tuple(str(clip_id) for clip_id in motion_command.motion.clip_ids)
    expected_body_names = tuple(str(name) for name in motion_command.motion_cfg.body_names_to_track)
    expected_ref_name = str(motion_command.motion_cfg.body_name_ref[0])

    cached_entry = cache.get(cache_key)
    if (
        cached_entry is not None
        and cached_entry.get("clip_ids") == expected_clip_ids
        and cached_entry.get("body_names") == expected_body_names
        and cached_entry.get("ref_name") == expected_ref_name
        and cached_entry.get("episodic_motion_end") == episodic_motion_end
    ):
        return cached_entry.get("bank")

    if not resolved_root.is_dir():
        raise FileNotFoundError(
            "Rollout-reference rewards are enabled, but the configured root does not exist or is not a directory: "
            f"'{resolved_root}'. Refusing to silently replace the configured rewards with zeros."
        )

    num_clips = int(motion_command.motion.num_clips)
    num_bodies = len(expected_body_names)
    clip_name_to_index = {clip_name: idx for idx, clip_name in enumerate(motion_command.motion.clip_ids)}

    clip_payloads: dict[int, dict[str, np.ndarray]] = {}
    clip_sources: dict[int, Path] = {}
    duplicate_clip_sources: dict[str, list[str]] = {}
    matching_load_errors: dict[str, str] = {}
    max_steps = 0
    has_any_object = False

    for clip_dir in sorted(resolved_root.iterdir()):
        if not clip_dir.is_dir():
            continue
        rollout_path = clip_dir / "teacher_rollout_reference.npz"
        if not rollout_path.is_file():
            continue
        loaded_clip_id = ""
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
                        except Exception as exc:
                            inferred_clip_id = resolve_contact_export_clip_id(
                                clip_dir.name,
                                clip_name_to_index,
                            )
                            if inferred_clip_id in clip_name_to_index:
                                loaded_clip_id = inferred_clip_id
                                raise ValueError(
                                    "invalid rollout-reference metadata for active clip "
                                    f"{inferred_clip_id!r}: {metadata_path}: {exc}"
                                ) from exc
                            clip_id = ""
                if not clip_id:
                    clip_id = resolve_contact_export_clip_id(clip_dir.name, clip_name_to_index)
                if not clip_id or clip_id not in clip_name_to_index:
                    continue
                loaded_clip_id = clip_id

                if "tracked_body_names" in data.files:
                    loaded_body_names = tuple(str(name) for name in np.asarray(data["tracked_body_names"]).tolist())
                    if loaded_body_names != expected_body_names:
                        raise ValueError(
                            f"tracked_body_names {loaded_body_names!r} do not match active training bodies "
                            f"{expected_body_names!r}"
                        )
                if "ref_body_name" in data.files:
                    loaded_ref_name = str(np.asarray(data["ref_body_name"]).item())
                    if loaded_ref_name != expected_ref_name:
                        raise ValueError(
                            f"ref_body_name {loaded_ref_name!r} does not match active reference body "
                            f"{expected_ref_name!r}"
                        )

                valid_steps = np.asarray(data["valid_steps"], dtype=np.bool_).reshape(-1)
                if valid_steps.size == 0:
                    raise ValueError("valid_steps is empty")
                if not bool(valid_steps.any()):
                    raise ValueError("valid_steps does not contain any usable reference frame")
                clip_index = int(clip_name_to_index[clip_id])
                clip_length = int(motion_command.motion.clip_lengths[clip_index].item())
                required_reference_steps = _required_rollout_reference_steps(
                    clip_length,
                    episodic_motion_end=episodic_motion_end,
                )
                motion_end_mode = "episodic" if episodic_motion_end else "continuing"
                if valid_steps.size < required_reference_steps:
                    raise ValueError(
                        f"rollout reference has {valid_steps.size} frames but {motion_end_mode} motion execution "
                        f"requires {required_reference_steps} reward-bearing frames (clip_length={clip_length})"
                    )
                if not bool(valid_steps[:required_reference_steps].all()):
                    first_invalid = int(np.flatnonzero(~valid_steps[:required_reference_steps])[0])
                    raise ValueError(
                        "rollout reference is missing a usable frame inside the reward-bearing motion range: "
                        f"first_invalid_step={first_invalid}, "
                        f"required_reference_steps={required_reference_steps}, "
                        f"clip_length={clip_length}, motion_end_mode={motion_end_mode}"
                    )
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

                if motion_command.motion.has_object and "object_pos_local" not in payload:
                    raise ValueError(
                        "active motion contains an object, but the rollout reference has no complete object state"
                    )

                non_finite_keys = [
                    key
                    for key, value in payload.items()
                    if np.issubdtype(value.dtype, np.floating) and not bool(np.isfinite(value).all())
                ]
                if non_finite_keys:
                    raise ValueError(f"non-finite values found in arrays {non_finite_keys}")

                for quat_key in ("body_quat_w", "ref_quat_w", "root_quat_w", "object_quat_w"):
                    quat_value = payload.get(quat_key)
                    if quat_value is None:
                        continue
                    valid_quat = quat_value[valid_steps]
                    quat_norm = np.linalg.norm(valid_quat, axis=-1)
                    if quat_norm.size == 0 or not bool(np.all(np.abs(quat_norm - 1.0) <= 1.0e-3)):
                        min_norm = float(quat_norm.min()) if quat_norm.size else float("nan")
                        max_norm = float(quat_norm.max()) if quat_norm.size else float("nan")
                        raise ValueError(
                            f"{quat_key} must contain unit quaternions on valid steps; "
                            f"norm range=[{min_norm}, {max_norm}]"
                        )

                if clip_index in clip_payloads:
                    duplicate_clip_sources.setdefault(
                        clip_id,
                        [str(clip_sources[clip_index])],
                    ).append(str(rollout_path))
                    continue
                clip_payloads[clip_index] = payload
                clip_sources[clip_index] = rollout_path
                max_steps = max(max_steps, int(valid_steps.size))
        except Exception as exc:
            if loaded_clip_id in clip_name_to_index:
                matching_load_errors[loaded_clip_id] = f"{rollout_path}: {exc}"
            logger.warning("Ignoring invalid teacher rollout reference '{}': {}", rollout_path, exc)

    if duplicate_clip_sources:
        raise RuntimeError(
            "Rollout-reference rewards found multiple directories for an active clip: "
            + "; ".join(
                f"{clip_id}={sources}"
                for clip_id, sources in sorted(duplicate_clip_sources.items())
            )
        )

    invalid_duplicate_sources = {
        clip_id: error
        for clip_id, error in matching_load_errors.items()
        if int(clip_name_to_index[clip_id]) in clip_payloads
    }
    if invalid_duplicate_sources:
        raise RuntimeError(
            "Rollout-reference rewards found both a valid and an invalid directory for an active clip: "
            + "; ".join(
                f"{clip_id}={error}"
                for clip_id, error in sorted(invalid_duplicate_sources.items())
            )
        )

    missing_clip_ids = [
        clip_id for clip_id, clip_index in clip_name_to_index.items() if int(clip_index) not in clip_payloads
    ]
    if missing_clip_ids:
        matching_error_details = [matching_load_errors[clip_id] for clip_id in missing_clip_ids if clip_id in matching_load_errors]
        detail_suffix = "" if not matching_error_details else " Invalid matching files: " + "; ".join(matching_error_details)
        raise RuntimeError(
            "Rollout-reference rewards require one valid teacher_rollout_reference.npz for every active clip. "
            f"Missing or invalid clip ids under '{resolved_root}': {missing_clip_ids}.{detail_suffix}"
        )
    if max_steps <= 0:
        raise RuntimeError(f"No usable rollout-reference frames were found under '{resolved_root}'.")

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
        "Rollout reference tracking loaded {} clip(s) from '{}'. has_object={}",
        matched_clip_count,
        resolved_root,
        has_any_object,
    )
    cache[cache_key] = {
        "clip_ids": expected_clip_ids,
        "body_names": expected_body_names,
        "ref_name": expected_ref_name,
        "episodic_motion_end": episodic_motion_end,
        "bank": bank,
    }
    return bank


def _sample_rollout_reference(
    env: WholeBodyTrackingManager,
    motion_command: MotionCommand,
    *,
    rollout_reference_root: str,
) -> dict[str, torch.Tensor] | None:
    sample_cache = getattr(env, "_rollout_reference_sample_cache", None)
    if sample_cache is None:
        sample_cache = {}
        setattr(env, "_rollout_reference_sample_cache", sample_cache)

    cache_key = _rollout_reference_sample_cache_key(env, motion_command, rollout_reference_root=rollout_reference_root)
    if cache_key in sample_cache:
        return sample_cache[cache_key]
    sample_cache.clear()

    bank = _get_rollout_reference_bank(
        env,
        motion_command,
        rollout_reference_root=rollout_reference_root,
    )
    if bank is None:
        sample_cache[cache_key] = None
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
    sample_cache[cache_key] = sampled
    return sampled


def _rollout_reference_relative_body_targets(
    env: WholeBodyTrackingManager,
    motion_command: MotionCommand,
    sampled_reference: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    reward_compute_counter = getattr(env, "_reward_compute_counter", None)
    cache_key = None
    if reward_compute_counter is not None:
        cache_key = (id(sampled_reference), id(motion_command), int(reward_compute_counter))
        cache = getattr(env, "_rollout_reference_relative_body_targets_cache", None)
        if cache is None:
            cache = {}
            setattr(env, "_rollout_reference_relative_body_targets_cache", cache)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    episode_length_buf = getattr(env, "episode_length_buf", None)
    if episode_length_buf is None:
        episode_length_buf = torch.ones((motion_command.num_envs,), device=motion_command.device, dtype=torch.long)
    use_root = (episode_length_buf == 0).to(device=env.device, dtype=torch.float32).unsqueeze(1)

    ref_pos_w = sampled_reference["root_pos_w"] * use_root + sampled_reference["ref_pos_w"] * (1.0 - use_root)
    ref_quat_w = sampled_reference["root_quat_w"] * use_root + sampled_reference["ref_quat_w"] * (1.0 - use_root)
    robot_ref_pos_w = motion_command.robot_root_pos_w * use_root + motion_command.robot_ref_pos_w * (1.0 - use_root)
    robot_ref_quat_w = motion_command.robot_root_quat_w * use_root + motion_command.robot_ref_quat_w * (1.0 - use_root)

    delta_quat_w = yaw_quat(
        quat_mul(robot_ref_quat_w, quat_inverse(ref_quat_w, w_last=True), w_last=True),
        w_last=True,
    )
    relative_body_quat_w = quat_mul_broadcast_left(delta_quat_w, sampled_reference["body_quat_w"], w_last=True)
    delta_pos_w_height = ref_pos_w - robot_ref_pos_w
    delta_pos_w_height[..., :2] = 0.0
    relative_body_pos_w = (
        robot_ref_pos_w[:, None, :]
        + delta_pos_w_height[:, None, :]
        + quat_apply_broadcast_left(
            delta_quat_w,
            sampled_reference["body_pos_w"] - ref_pos_w[:, None, :],
            w_last=True,
        )
    )
    if cache_key is not None:
        cache.clear()
        cache[cache_key] = (relative_body_pos_w, relative_body_quat_w)
    return relative_body_pos_w, relative_body_quat_w


def motion_relative_body_position_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
    rollout_reference_root: str | None = None,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    if rollout_reference_root is None:
        relative_body_pos_w = motion_command.body_pos_relative_w
        valid_mask = None
    else:
        sampled_reference = _sample_rollout_reference(
            env,
            motion_command,
            rollout_reference_root=rollout_reference_root,
        )
        if sampled_reference is None:
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        relative_body_pos_w, _ = _rollout_reference_relative_body_targets(env, motion_command, sampled_reference)
        valid_mask = sampled_reference["valid_mask"]

    error = torch.sum(torch.square(relative_body_pos_w - motion_command.robot_body_pos_w), dim=-1)
    body_indexes = _get_tracked_body_subset_indexes(
        env,
        motion_command,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
    )
    error = error.index_select(1, body_indexes)
    reward = torch.exp(-error.mean(-1) / sigma**2)
    if valid_mask is not None:
        reward = reward * valid_mask.to(dtype=torch.float32)
    return reward


def motion_relative_body_orientation_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
    rollout_reference_root: str | None = None,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    if rollout_reference_root is None:
        relative_body_quat_w = motion_command.body_quat_relative_w
        valid_mask = None
    else:
        sampled_reference = _sample_rollout_reference(
            env,
            motion_command,
            rollout_reference_root=rollout_reference_root,
        )
        if sampled_reference is None:
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        _, relative_body_quat_w = _rollout_reference_relative_body_targets(env, motion_command, sampled_reference)
        valid_mask = sampled_reference["valid_mask"]

    error = quat_error_magnitude(relative_body_quat_w, motion_command.robot_body_quat_w) ** 2
    body_indexes = _get_tracked_body_subset_indexes(
        env,
        motion_command,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
    )
    error = error.index_select(1, body_indexes)
    reward = torch.exp(-error.mean(-1) / sigma**2)
    if valid_mask is not None:
        reward = reward * valid_mask.to(dtype=torch.float32)
    return reward


def motion_global_body_lin_vel(
    env: WholeBodyTrackingManager,
    sigma: float,
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
    rollout_reference_root: str | None = None,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    if rollout_reference_root is None:
        body_lin_vel_w = motion_command.body_lin_vel_w
        valid_mask = None
    else:
        sampled_reference = _sample_rollout_reference(
            env,
            motion_command,
            rollout_reference_root=rollout_reference_root,
        )
        if sampled_reference is None:
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        body_lin_vel_w = sampled_reference["body_lin_vel_w"]
        valid_mask = sampled_reference["valid_mask"]

    error = torch.sum(torch.square(body_lin_vel_w - motion_command.robot_body_lin_vel_w), dim=-1)
    body_indexes = _get_tracked_body_subset_indexes(
        env,
        motion_command,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
    )
    error = error.index_select(1, body_indexes)
    reward = torch.exp(-error.mean(-1) / sigma**2)
    if valid_mask is not None:
        reward = reward * valid_mask.to(dtype=torch.float32)
    return reward


def motion_global_body_ang_vel(
    env: WholeBodyTrackingManager,
    sigma: float,
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
    rollout_reference_root: str | None = None,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    if rollout_reference_root is None:
        body_ang_vel_w = motion_command.body_ang_vel_w
        valid_mask = None
    else:
        sampled_reference = _sample_rollout_reference(
            env,
            motion_command,
            rollout_reference_root=rollout_reference_root,
        )
        if sampled_reference is None:
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        body_ang_vel_w = sampled_reference["body_ang_vel_w"]
        valid_mask = sampled_reference["valid_mask"]

    error = torch.sum(torch.square(body_ang_vel_w - motion_command.robot_body_ang_vel_w), dim=-1)
    body_indexes = _get_tracked_body_subset_indexes(
        env,
        motion_command,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
    )
    error = error.index_select(1, body_indexes)
    reward = torch.exp(-error.mean(-1) / sigma**2)
    if valid_mask is not None:
        reward = reward * valid_mask.to(dtype=torch.float32)
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
    rollout_reference_root: str | None = None,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    if rollout_reference_root is None:
        object_pos_w = motion_command.object_pos_w
        valid_mask = None
    else:
        sampled_reference = _sample_rollout_reference(
            env,
            motion_command,
            rollout_reference_root=rollout_reference_root,
        )
        if sampled_reference is None:
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        object_pos_w = sampled_reference["object_pos_w"]
        valid_mask = sampled_reference["object_valid_mask"]

    error = torch.sum(torch.square(object_pos_w - motion_command.simulator_object_pos_w), dim=-1)
    reward = torch.exp(-error / sigma**2)
    if valid_mask is not None:
        reward = reward * valid_mask.to(dtype=torch.float32)
    return reward


def object_global_ref_orientation_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    rollout_reference_root: str | None = None,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    if rollout_reference_root is None:
        object_quat_w = motion_command.object_quat_w
        valid_mask = None
    else:
        sampled_reference = _sample_rollout_reference(
            env,
            motion_command,
            rollout_reference_root=rollout_reference_root,
        )
        if sampled_reference is None:
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        object_quat_w = sampled_reference["object_quat_w"]
        valid_mask = sampled_reference["object_valid_mask"]

    error = quat_error_magnitude(object_quat_w, motion_command.simulator_object_quat_w) ** 2
    reward = torch.exp(-error / sigma**2)
    if valid_mask is not None:
        reward = reward * valid_mask.to(dtype=torch.float32)
    return reward


def teacher_rollout_global_ref_position_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    rollout_reference_root: str = "outputs/clips",
) -> torch.Tensor:
    return motion_global_ref_position_error_exp(env, sigma=sigma, rollout_reference_root=rollout_reference_root)


def teacher_rollout_global_ref_orientation_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    rollout_reference_root: str = "outputs/clips",
) -> torch.Tensor:
    return motion_global_ref_orientation_error_exp(env, sigma=sigma, rollout_reference_root=rollout_reference_root)


def teacher_rollout_relative_body_position_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    rollout_reference_root: str = "outputs/clips",
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
) -> torch.Tensor:
    return motion_relative_body_position_error_exp(
        env,
        sigma=sigma,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
        rollout_reference_root=rollout_reference_root,
    )


def teacher_rollout_relative_body_orientation_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    rollout_reference_root: str = "outputs/clips",
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
) -> torch.Tensor:
    return motion_relative_body_orientation_error_exp(
        env,
        sigma=sigma,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
        rollout_reference_root=rollout_reference_root,
    )


def teacher_rollout_global_body_lin_vel(
    env: WholeBodyTrackingManager,
    sigma: float,
    rollout_reference_root: str = "outputs/clips",
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
) -> torch.Tensor:
    return motion_global_body_lin_vel(
        env,
        sigma=sigma,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
        rollout_reference_root=rollout_reference_root,
    )


def teacher_rollout_global_body_ang_vel(
    env: WholeBodyTrackingManager,
    sigma: float,
    rollout_reference_root: str = "outputs/clips",
    body_names: list[str] | None = None,
    body_name_pattern: str | None = None,
) -> torch.Tensor:
    return motion_global_body_ang_vel(
        env,
        sigma=sigma,
        body_names=body_names,
        body_name_pattern=body_name_pattern,
        rollout_reference_root=rollout_reference_root,
    )


def teacher_rollout_object_global_ref_position_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    rollout_reference_root: str = "outputs/clips",
) -> torch.Tensor:
    return object_global_ref_position_error_exp(env, sigma=sigma, rollout_reference_root=rollout_reference_root)


def teacher_rollout_object_global_ref_orientation_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    rollout_reference_root: str = "outputs/clips",
) -> torch.Tensor:
    return object_global_ref_orientation_error_exp(env, sigma=sigma, rollout_reference_root=rollout_reference_root)


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
        self._measurement_body_names: list[str] = []
        self._measurement_body_indices = torch.empty(0, device=self.env.device, dtype=torch.long)
        self._region_force_measurement_indices: dict[str, torch.Tensor] = {}
        self._region_position_measurement_indices: dict[str, torch.Tensor] = {}
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
        # These banks are immutable after their one-time load.  Cache their
        # aggregate availability on the host so the reward hot path never
        # calls CUDA ``any().item()`` merely to rediscover static metadata.
        self._contact_bank_has_intervals = False
        self._contact_bank_has_schedule = False
        self._contact_bank_has_missing_schedule = True
        self._contact_bank_pickup_steps_by_clip: torch.Tensor | None = None
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

        measurement_body_names: list[str] = []
        measurement_body_name_to_index: dict[str, int] = {}
        for region_name in self.region_names:
            body_names = [
                *self._region_force_body_names.get(region_name, []),
                *self._region_position_body_names.get(region_name, []),
            ]
            for body_name in body_names:
                if body_name in measurement_body_name_to_index:
                    continue
                measurement_body_name_to_index[body_name] = len(measurement_body_names)
                measurement_body_names.append(body_name)

        self._measurement_body_names = measurement_body_names
        self._measurement_body_indices = torch.tensor(
            [body_name_to_index[name] for name in measurement_body_names],
            device=self.env.device,
            dtype=torch.long,
        )
        for region_name in self.region_names:
            force_indices = [
                measurement_body_name_to_index[name] for name in self._region_force_body_names.get(region_name, [])
            ]
            position_indices = [
                measurement_body_name_to_index[name] for name in self._region_position_body_names.get(region_name, [])
            ]
            self._region_force_measurement_indices[region_name] = torch.tensor(
                force_indices,
                device=self.env.device,
                dtype=torch.long,
            )
            self._region_position_measurement_indices[region_name] = torch.tensor(
                position_indices,
                device=self.env.device,
                dtype=torch.long,
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
        self._contact_bank_has_intervals = False
        self._contact_bank_has_schedule = False
        self._contact_bank_has_missing_schedule = True
        self._contact_bank_pickup_steps_by_clip = None
        self._contact_bank_available = False

        if self.contact_export_root is None:
            raise ValueError(
                "OfflineContactPointGuidance is enabled with non-zero reward weight, but contact_export_root is empty."
            )
        if not self.contact_export_root.is_dir():
            raise FileNotFoundError(
                "OfflineContactPointGuidance is enabled, but the configured contact export root does not exist or "
                f"is not a directory: '{self.contact_export_root}'."
            )

        clip_name_to_index = {clip_name: idx for idx, clip_name in enumerate(motion_command.motion.clip_ids)}
        clip_region_points: dict[tuple[int, int], np.ndarray] = {}
        clip_region_intervals: dict[tuple[int, int], tuple[int, int]] = {}
        clip_region_schedules: dict[tuple[int, int], np.ndarray] = {}
        clip_source_dirs: dict[int, Path] = {}
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
                    inferred_clip_id = resolve_contact_export_clip_id(
                        clip_dir.name,
                        clip_name_to_index,
                    )
                    if inferred_clip_id in clip_name_to_index:
                        raise RuntimeError(
                            "Invalid offline-contact metadata for an active clip: "
                            f"clip={inferred_clip_id!r}, path={metadata_path}: {exc}"
                        ) from exc
                    logger.warning("Skipping invalid contact metadata '{}': {}", metadata_path, exc)
                    continue
                if not isinstance(metadata, dict):
                    inferred_clip_id = resolve_contact_export_clip_id(
                        clip_dir.name,
                        clip_name_to_index,
                    )
                    if inferred_clip_id in clip_name_to_index:
                        raise RuntimeError(
                            "Offline-contact metadata for an active clip must be a JSON object: "
                            f"clip={inferred_clip_id!r}, path={metadata_path}."
                        )
                    logger.warning(
                        "Skipping non-object contact metadata for inactive directory '{}'.",
                        metadata_path,
                    )
                    continue

            clip_id = str(metadata.get("clip_id", "")).strip()
            if not clip_id:
                clip_id = resolve_contact_export_clip_id(clip_dir.name, clip_name_to_index)
            if not clip_id or clip_id not in clip_name_to_index:
                continue
            clip_index = clip_name_to_index[clip_id]
            previous_source = clip_source_dirs.get(clip_index)
            if previous_source is not None:
                raise RuntimeError(
                    "Multiple offline-contact directories resolve to the same active clip: "
                    f"clip={clip_id!r}, first={previous_source}, second={clip_dir}."
                )
            clip_source_dirs[clip_index] = clip_dir
            if self.require_stable_contact and metadata and not bool(metadata.get("stable_contact_success", False)):
                continue

            for region_idx, region_name in enumerate(self.region_names):
                export_label = _CONTACT_EXPORT_LABEL_BY_REGION[region_name]
                points_path = clip_dir / f"{export_label}_contact_points.npy"
                if not points_path.is_file():
                    continue
                try:
                    points = np.asarray(np.load(points_path), dtype=np.float32).reshape(-1, 3)
                except Exception as exc:
                    raise RuntimeError(f"Invalid contact point cloud '{points_path}': {exc}") from exc
                if not bool(np.isfinite(points).all()):
                    raise RuntimeError(f"Invalid contact point cloud '{points_path}': contains NaN or Inf values.")
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
                            converted_interval = _convert_contact_interval_timebase(
                                (int(interval_steps[0]), int(interval_steps[1])),
                                metadata=metadata,
                                motion_fps=float(motion_command.motion.fps),
                            )
                            compensation_enabled = bool(
                                getattr(
                                    getattr(motion_command, "motion_cfg", None),
                                    "contact_interval_runtime_prepend_compensation",
                                    False,
                                )
                            )
                            runtime_prepend_offset = (
                                int(getattr(motion_command, "_runtime_default_pose_prepend_steps", 0) or 0)
                                if compensation_enabled
                                and bool(getattr(motion_command, "_runtime_default_pose_prepend_enabled", False))
                                else 0
                            )
                            start_step = max(0, int(converted_interval[0]) - runtime_prepend_offset)
                            end_step = int(converted_interval[1]) - runtime_prepend_offset
                            if start_step >= 0 and end_step > start_step:
                                clip_lengths = getattr(motion_command.motion, "clip_lengths", None)
                                if clip_lengths is not None:
                                    clip_length = int(clip_lengths[clip_index].item())
                                    if start_step >= clip_length or end_step > clip_length:
                                        raise RuntimeError(
                                            "Contact schedule interval is outside the active motion-time range "
                                            "after runtime-prepend conversion: "
                                            f"clip={clip_id!r}, region={region_name!r}, "
                                            f"interval={(start_step, end_step)}, clip_length={clip_length}."
                                        )
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
            raise RuntimeError(
                "OfflineContactPointGuidance is enabled, but no matching valid region contact targets were found in "
                f"'{self.contact_export_root}'."
            )

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
        matched_clip_count = int(has_targets.any(dim=1).sum().item())
        require_complete_target_coverage = os.environ.get(
            "HOLOSOMA_REQUIRE_CONTACT_TARGET_COVERAGE",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        if require_complete_target_coverage and matched_clip_count != num_clips:
            missing_clip_ids = [
                str(motion_command.motion.clip_ids[index])
                for index in range(num_clips)
                if not bool(has_targets[index].any().item())
            ]
            raise RuntimeError(
                "Complete offline contact-target coverage is required, but valid targets were loaded for only "
                f"{matched_clip_count}/{num_clips} clips. missing_preview={missing_clip_ids[:20]}."
            )
        unmeasurable_position_regions = [
            region_name
            for region_idx, region_name in enumerate(self.region_names)
            if bool(has_targets[:, region_idx].any().item())
            and not self._region_position_body_names.get(region_name)
        ]
        if unmeasurable_position_regions:
            raise RuntimeError(
                "Contact targets are present for regions that have no position measurement bodies in the active "
                f"simulator: {unmeasurable_position_regions}."
            )
        if self.use_force_term:
            unmeasurable_force_regions = [
                region_name
                for region_idx, region_name in enumerate(self.region_names)
                if bool(has_targets[:, region_idx].any().item())
                and not self._region_force_body_names.get(region_name)
            ]
            if unmeasurable_force_regions:
                raise RuntimeError(
                    "Force-gated contact targets are present for regions that have no force measurement bodies in "
                    f"the active simulator: {unmeasurable_force_regions}."
                )
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
        self._contact_bank_has_intervals = bool(has_intervals.any().item())
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
        assert self._clip_region_has_contact_schedule is not None
        self._contact_bank_has_schedule = bool(self._clip_region_has_contact_schedule.any().item())
        schedule_coverage = has_intervals | self._clip_region_has_contact_schedule
        self._contact_bank_has_missing_schedule = bool((~schedule_coverage).any().item())
        if self.contact_schedule_missing_mode == "after_pickup" and self._contact_bank_has_missing_schedule:
            pickup_steps_getter = getattr(motion_command, "_get_clip_pickup_steps_by_clip", None)
            if callable(pickup_steps_getter):
                try:
                    self._contact_bank_pickup_steps_by_clip = pickup_steps_getter().to(
                        device=self.env.device,
                        dtype=torch.long,
                    )
                except Exception:
                    self._contact_bank_pickup_steps_by_clip = torch.zeros(
                        (num_clips,), device=self.env.device, dtype=torch.long
                    )
            else:
                self._contact_bank_pickup_steps_by_clip = torch.zeros(
                    (num_clips,), device=self.env.device, dtype=torch.long
                )
        self._contact_bank_available = bool(has_targets.any().item())
        if self._contact_bank_available:
            logger.info(
                "OfflineContactPointGuidance loaded {} clip(s) with targets from '{}'.",
                matched_clip_count,
                self.contact_export_root,
            )

    @staticmethod
    def _infer_clip_id_from_dir_name(dir_name: str) -> str:
        return infer_contact_export_clip_id(dir_name)

    def _compute_current_region_measurements(self) -> tuple[torch.Tensor, torch.Tensor]:
        num_envs = self.env.num_envs
        num_regions = len(self.region_names)
        current_force = torch.zeros((num_envs, num_regions), device=self.env.device, dtype=torch.float32)
        current_position = torch.zeros((num_envs, num_regions, 3), device=self.env.device, dtype=torch.float32)

        if not self.motion_command.motion.has_object:
            return current_force, current_position

        if self._measurement_body_names:
            all_force_history = self.motion_command.get_body_object_contact_force_history(self._measurement_body_names)
            all_force_magnitude = torch.amax(torch.linalg.norm(all_force_history, dim=-1), dim=1)
        else:
            all_force_magnitude = torch.zeros((num_envs, 0), device=self.env.device, dtype=torch.float32)

        if self._measurement_body_indices.numel() > 0:
            all_relative_positions = self.motion_command._body_positions_in_object_frame(self._measurement_body_indices)
        else:
            all_relative_positions = torch.zeros((num_envs, 0, 3), device=self.env.device, dtype=torch.float32)

        for region_idx, region_name in enumerate(self.region_names):
            force_measurement_indices = self._region_force_measurement_indices.get(region_name)
            if force_measurement_indices is not None and force_measurement_indices.numel() > 0:
                region_force = all_force_magnitude.index_select(1, force_measurement_indices)
                current_force[:, region_idx] = torch.amax(region_force, dim=1)

            position_measurement_indices = self._region_position_measurement_indices.get(region_name)
            if position_measurement_indices is None or position_measurement_indices.numel() == 0:
                continue

            relative_positions = all_relative_positions.index_select(1, position_measurement_indices)
            if relative_positions.shape[1] == 1:
                current_position[:, region_idx] = relative_positions[:, 0]
                continue

            position_force_weights = all_force_magnitude.index_select(1, position_measurement_indices)

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
        if interval_bank is not None and has_interval_bank is not None and self._contact_bank_has_intervals:
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
            and self._contact_bank_has_schedule
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

        if not self._contact_bank_has_missing_schedule:
            return schedule_active

        if self.contact_schedule_missing_mode == "always_on":
            return schedule_active
        if self.contact_schedule_missing_mode == "inactive":
            return torch.where(missing_schedule, torch.zeros_like(schedule_active), schedule_active)

        pickup_steps_by_clip = self._contact_bank_pickup_steps_by_clip
        if pickup_steps_by_clip is None:
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

    @staticmethod
    def _compute_guidance_reward_from_min_distance(
        *,
        min_distance: torch.Tensor,
        current_force: torch.Tensor,
        active_regions: torch.Tensor,
        position_sigma: float,
        use_force_term: bool,
        force_threshold: float,
        force_sigma: float,
        force_gate_mode: str,
        region_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if region_indices is not None:
            min_distance = min_distance.index_select(1, region_indices)
            current_force = current_force.index_select(1, region_indices)
            active_regions = active_regions.index_select(1, region_indices)

        position_term = torch.exp(-min_distance / max(float(position_sigma), 1.0e-6))
        if use_force_term:
            if force_gate_mode == "binary":
                force_term = (current_force >= force_threshold).to(dtype=torch.float32)
            elif force_gate_mode == "soft":
                force_term = torch.exp((current_force - force_threshold) / max(float(force_sigma), 1.0e-6)).clamp(
                    max=1.0
                )
            else:
                raise ValueError(f"Unsupported force_gate_mode '{force_gate_mode}'. Expected one of: ['soft', 'binary'].")
            per_region_reward = position_term * force_term
        else:
            per_region_reward = position_term

        active_region_weights = active_regions.to(dtype=torch.float32)
        active_region_count = active_region_weights.sum(dim=1)
        return (per_region_reward * active_region_weights).sum(dim=1) / active_region_count.clamp_min(1.0)

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

        distances = torch.linalg.norm(target_points - current_position.unsqueeze(2), dim=-1)
        inf_fill = torch.full_like(distances, float("inf"))
        distances = torch.where(point_mask, distances, inf_fill)
        min_distance = distances.min(dim=-1).values

        return self._compute_guidance_reward_from_min_distance(
            min_distance=min_distance,
            current_force=current_force,
            active_regions=active_regions,
            position_sigma=self.position_sigma,
            use_force_term=self.use_force_term,
            force_threshold=self.force_threshold,
            force_sigma=self.force_sigma,
            force_gate_mode=self.force_gate_mode,
        )


class FusedOfflineContactPointGuidance(OfflineContactPointGuidance):
    """Combine wrist target guidance and force-gated contact guidance in one pass."""

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        raw_params = dict(cfg.params)
        wrist_region_names = self._normalize_region_names(
            raw_params.get("wrist_region_names", ["left_wrist", "right_wrist"])
        )
        contact_region_names = self._normalize_region_names(
            raw_params.get("contact_region_names", raw_params.get("region_names", wrist_region_names))
        )

        union_region_names: list[str] = []
        seen_region_names: set[str] = set()
        for region_name in [*wrist_region_names, *contact_region_names]:
            if region_name in seen_region_names:
                continue
            union_region_names.append(region_name)
            seen_region_names.add(region_name)

        base_params = dict(raw_params)
        base_params["region_names"] = union_region_names
        base_cfg = RewardTermCfg(func=cfg.func, params=base_params, weight=cfg.weight, tags=cfg.tags)
        super().__init__(base_cfg, env)

        region_to_index = {region_name: idx for idx, region_name in enumerate(self.region_names)}
        self._wrist_region_indices = torch.tensor(
            [region_to_index[region_name] for region_name in wrist_region_names],
            device=self.env.device,
            dtype=torch.long,
        )
        self._contact_region_indices = torch.tensor(
            [region_to_index[region_name] for region_name in contact_region_names],
            device=self.env.device,
            dtype=torch.long,
        )

        self.wrist_weight = float(raw_params.get("wrist_weight", raw_params.get("target_weight", 0.0)))
        self.contact_weight = float(raw_params.get("contact_weight", 1.0))
        self.wrist_position_sigma = float(raw_params.get("wrist_position_sigma", raw_params.get("position_sigma", 0.05)))
        self.contact_position_sigma = float(
            raw_params.get("contact_position_sigma", raw_params.get("position_sigma", 0.05))
        )
        self.contact_force_threshold = float(
            raw_params.get("contact_force_threshold", raw_params.get("force_threshold", 1.7))
        )
        self.contact_force_sigma = float(raw_params.get("contact_force_sigma", raw_params.get("force_sigma", 10.0)))
        self.contact_use_force_term = bool(raw_params.get("contact_use_force_term", raw_params.get("use_force_term", True)))
        self.contact_force_gate_mode = str(
            raw_params.get("contact_force_gate_mode", raw_params.get("force_gate_mode", "binary"))
        ).strip().lower()
        if self.contact_force_gate_mode not in {"soft", "binary"}:
            raise ValueError(
                "Unsupported contact_force_gate_mode "
                f"'{self.contact_force_gate_mode}'. Expected one of: ['soft', 'binary']."
            )

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

        distances = torch.linalg.norm(target_points - current_position.unsqueeze(2), dim=-1)
        inf_fill = torch.full_like(distances, float("inf"))
        distances = torch.where(point_mask, distances, inf_fill)
        min_distance = distances.min(dim=-1).values

        reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        if self.wrist_weight != 0.0:
            wrist_reward = self._compute_guidance_reward_from_min_distance(
                min_distance=min_distance,
                current_force=current_force,
                active_regions=active_regions,
                position_sigma=self.wrist_position_sigma,
                use_force_term=False,
                force_threshold=self.contact_force_threshold,
                force_sigma=self.contact_force_sigma,
                force_gate_mode=self.contact_force_gate_mode,
                region_indices=self._wrist_region_indices,
            )
            reward = reward + self.wrist_weight * wrist_reward
        if self.contact_weight != 0.0:
            contact_reward = self._compute_guidance_reward_from_min_distance(
                min_distance=min_distance,
                current_force=current_force,
                active_regions=active_regions,
                position_sigma=self.contact_position_sigma,
                use_force_term=self.contact_use_force_term,
                force_threshold=self.contact_force_threshold,
                force_sigma=self.contact_force_sigma,
                force_gate_mode=self.contact_force_gate_mode,
                region_indices=self._contact_region_indices,
            )
            reward = reward + self.contact_weight * contact_reward

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
        all_body_names = list(self.env.simulator.body_names)  # type: ignore[attr-defined]
        body_names = cfg.params.get("body_names")
        body_name_selector = cfg.params.get("undesired_contacts_body_names", "")
        if body_names is None and isinstance(body_name_selector, (list, tuple)):
            body_names = body_name_selector

        if body_names is not None:
            missing = [name for name in body_names if name not in all_body_names]
            if missing:
                raise ValueError(f"Requested body names {missing} are not available in {all_body_names}.")
            undesired_contacts_body_names = list(body_names)
        else:
            undesired_contacts_body_names = [
                body_name
                for body_name in all_body_names
                if re.match(str(body_name_selector), body_name)
            ]
        required_selected_body_names = cfg.params.get("required_selected_body_names", ())
        missing_required_selected = [
            name for name in required_selected_body_names if name not in undesired_contacts_body_names
        ]
        if missing_required_selected:
            raise ValueError(
                f"UndesiredContacts expected selected body names {missing_required_selected}, "
                f"but selected {undesired_contacts_body_names} from simulator bodies {all_body_names}."
            )

        forbidden_sim_body_names = cfg.params.get("forbidden_sim_body_names", ())
        unexpected_present = [name for name in forbidden_sim_body_names if name in all_body_names]
        if unexpected_present:
            raise ValueError(
                f"UndesiredContacts found forbidden simulator body names {unexpected_present}. "
                "This usually means fixed joints were not collapsed, so parent-body contact penalties "
                f"will not catch those child-link contacts. Simulator bodies: {all_body_names}."
            )
        self.undesired_contacts_body_indexes = self._get_index_of_a_in_b(
            undesired_contacts_body_names,
            all_body_names,
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
