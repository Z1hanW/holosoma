from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import re
import sys
import zipfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, List

import numpy as np
import smart_open
import torch
import torch.nn.functional as F
from loguru import logger

from holosoma.config_types.command import (
    CleanNoisyClipCurriculumConfig,
    FixedClipGroupAssignmentConfig,
    HMIGoalPoseNoiseConfig,
    HMIMotionConfig,
    MotionConfig,
    NoiseToInitialPoseConfig,
)
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.base import CommandTermBase
from holosoma.managers.termination.manager import TerminationManager
from holosoma.utils.clip_sampling import build_prefix_mask, piecewise_constant_schedule_value, project_group_weights
from holosoma.utils.contact_intervals import (
    CONTACT_INTERVAL_FALLBACK_FILES as _ADAPTIVE_SAMPLING_CONTACT_INTERVAL_FALLBACK_FILES,
    convert_contact_interval_timebase as _convert_contact_interval_timebase,
    infer_contact_export_clip_id as _infer_contact_export_clip_id,
    normalize_contact_interval_pair as _normalize_contact_interval_pair,
    resolve_contact_export_clip_id as _resolve_contact_export_clip_id,
    select_primary_contact_interval as _select_primary_contact_interval,
)
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.motion_transition_source import (
    MOTION_TRANSITION_SOURCE_KEY,
    canonical_motion_transition_source,
    resolve_motion_transition_source_for_motion_path,
)
from holosoma.utils.rank_local_shards import current_rank_local_shard_metadata, resolve_rank_local_motion_path
from holosoma.utils.rotations import (
    calc_heading,
    calc_heading_quat_inv,
    get_euler_xyz,
    normalize_angle,
    quat_apply,
    quat_apply_broadcast_left,
    quat_conjugate,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inverse,
    quat_mul,
    quat_mul_broadcast_left,
    quat_normalize,
    quaternion_to_matrix,
    slerp,
    yaw_quat,
)
from holosoma.utils.simulator_config import SimulatorType

_RUNTIME_PICKUP_LIFT_HEIGHT_THRESHOLD = 0.10
_RUNTIME_PICKUP_CONSECUTIVE_STEPS = 5
_LIFT_DIAGNOSTIC_OBJECT_LIFT_HEIGHT_THRESHOLD = 0.08
_LIFT_DIAGNOSTIC_MIN_CONTACT_FORCE = 1.0
_LIFT_DIAGNOSTIC_FALSE_POSITIVE_MAX_WORLD_LIFT = 0.04
_LIFT_DIAGNOSTIC_FALSE_POSITIVE_MIN_ROOT_DROP = 0.05
_CLIP_PICKUP_LIFT_RATIO_THRESHOLD = 0.35
_STANDARD_MOTION_END_TERM_PATH = "holosoma.managers.termination.terms.wbt:motion_ends"
_ENABLED_ENV_FLAG_VALUES = frozenset({"1", "true", "yes", "on"})
MAX_MOTION_TRANSITION_STEPS = 4096
MOTION_TRANSITION_CONTRACT_VERSION = 1
_MOTION_TRANSITION_SOURCE_SEMANTICS = {
    "single_clip_static",
    "global_multi_clip_runtime",
}
_CURRICULUM_LIVE_TRACKING_ERROR_KEYS = frozenset(
    {
        "motion/error_ref_pos",
        "motion/error_ref_rot",
        "motion/error_ref_lin_vel",
        "motion/error_ref_ang_vel",
        "motion/error_body_pos",
        "motion/error_body_rot",
        "motion/error_body_lin_vel",
        "motion/error_body_ang_vel",
        "motion/error_joint_pos",
        "motion/error_joint_vel",
        "motion/error_object_ref_pos",
        "motion/error_object_ref_rot",
        "motion/error_object_ref_lin_vel",
    }
)


def build_fixed_hmi_track_mask(
    num_envs: int,
    track_ratio: float,
    partition_seed: int,
) -> torch.Tensor:
    """Build HMI's immutable track/generation partition on CPU.

    The dedicated generator makes the split reproducible without consuming the
    simulator/training RNG stream.  This follows Hybrid-Motion-Imitation's
    fixed-partition contract rather than resampling a task mode per episode.
    """

    if num_envs < 0:
        raise ValueError(f"num_envs must be non-negative, got {num_envs}.")
    if not 0.0 <= float(track_ratio) <= 1.0:
        raise ValueError(f"HMI track_ratio must be in [0, 1], got {track_ratio}.")
    num_track = max(0, min(num_envs, int(round(num_envs * float(track_ratio)))))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(partition_seed))
    shuffled_env_ids = torch.randperm(num_envs, generator=generator, device="cpu")
    track_mask = torch.zeros(num_envs, dtype=torch.bool, device="cpu")
    track_mask[shuffled_env_ids[:num_track]] = True
    return track_mask


def canonical_motion_transition_contract(contract: Any) -> dict[str, Any]:
    """Validate and canonicalize the effective training motion-transition contract."""

    if not isinstance(contract, dict):
        raise ValueError("motion_transition_contract must be a dictionary.")
    expected_keys = {"version", "control_dt_s", "source_semantics", "prepend", "append"}
    if set(contract) != expected_keys:
        raise ValueError(
            "motion_transition_contract must contain exactly "
            f"{sorted(expected_keys)}; got {sorted(map(str, contract))}."
        )

    version = contract["version"]
    if type(version) is not int or version != MOTION_TRANSITION_CONTRACT_VERSION:
        raise ValueError(
            "motion_transition_contract.version must equal integer "
            f"{MOTION_TRANSITION_CONTRACT_VERSION}, got {version!r}."
        )
    control_dt = contract["control_dt_s"]
    if (
        isinstance(control_dt, bool)
        or not isinstance(control_dt, numbers.Real)
        or not math.isfinite(float(control_dt))
        or float(control_dt) <= 0.0
    ):
        raise ValueError(
            "motion_transition_contract.control_dt_s must be a finite positive real number, "
            f"got {control_dt!r}."
        )
    source_semantics = contract["source_semantics"]
    if type(source_semantics) is not str or source_semantics not in _MOTION_TRANSITION_SOURCE_SEMANTICS:
        raise ValueError(
            "motion_transition_contract.source_semantics must be exactly "
            f"one of {sorted(_MOTION_TRANSITION_SOURCE_SEMANTICS)}, got {source_semantics!r}."
        )

    def canonical_phase(name: str, *, allowed_implementations: set[str]) -> dict[str, Any]:
        phase = contract[name]
        expected_phase_keys = {"implementation", "applied", "steps"}
        if not isinstance(phase, dict) or set(phase) != expected_phase_keys:
            actual = sorted(map(str, phase)) if isinstance(phase, dict) else type(phase).__name__
            raise ValueError(
                f"motion_transition_contract.{name} must contain exactly "
                f"{sorted(expected_phase_keys)}; got {actual}."
            )
        implementation = phase["implementation"]
        if type(implementation) is not str or implementation not in allowed_implementations:
            raise ValueError(
                f"motion_transition_contract.{name}.implementation must be one of "
                f"{sorted(allowed_implementations)}, got {implementation!r}."
            )
        applied = phase["applied"]
        if type(applied) is not bool:
            raise ValueError(
                f"motion_transition_contract.{name}.applied must be boolean, got {applied!r}."
            )
        steps = phase["steps"]
        if (
            type(steps) is not int
            or steps == 1
            or not 0 <= steps <= MAX_MOTION_TRANSITION_STEPS
        ):
            raise ValueError(
                f"motion_transition_contract.{name}.steps must be integer 0 (inactive) or in "
                f"[2, {MAX_MOTION_TRANSITION_STEPS}] (applied), got {steps!r}."
            )
        expected_applied = implementation != "none"
        if applied != expected_applied or applied != (steps > 0):
            raise ValueError(
                f"motion_transition_contract.{name} is internally inconsistent: "
                f"implementation={implementation!r}, applied={applied!r}, steps={steps}."
            )
        if implementation == "none" and steps != 0:
            raise ValueError(
                f"motion_transition_contract.{name}.steps must be zero when implementation='none'."
            )
        return {
            "implementation": implementation,
            "applied": applied,
            "steps": steps,
        }

    prepend_allowed = {"none", "static_splice"}
    if source_semantics == "global_multi_clip_runtime":
        prepend_allowed = {"none", "runtime_hold"}
    prepend = canonical_phase("prepend", allowed_implementations=prepend_allowed)
    append = canonical_phase("append", allowed_implementations={"none", "static_splice"})
    if source_semantics == "global_multi_clip_runtime" and append["applied"]:
        raise ValueError(
            "motion_transition_contract global_multi_clip_runtime semantics cannot apply an append transition."
        )

    return {
        "version": MOTION_TRANSITION_CONTRACT_VERSION,
        "control_dt_s": float(control_dt),
        "source_semantics": source_semantics,
        "prepend": prepend,
        "append": append,
    }


def motion_transition_contract_sha256(contract: Any) -> str:
    canonical = canonical_motion_transition_contract(contract)
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_CONTACT_PRIOR_REGION_BODY_NAMES = {
    "left_wrist": "left_wrist_yaw_link",
    "right_wrist": "right_wrist_yaw_link",
    "left_elbow": "left_elbow_link",
    "right_elbow": "right_elbow_link",
    "left_wrist_roll": "left_wrist_roll_link",
    "right_wrist_roll": "right_wrist_roll_link",
    "left_wrist_pitch": "left_wrist_pitch_link",
    "right_wrist_pitch": "right_wrist_pitch_link",
    "torso": "torso_link",
}
_CONTACT_PRIOR_REGION_NAMES = tuple(_CONTACT_PRIOR_REGION_BODY_NAMES)
_CONTACT_PRIOR_REGION_ALIASES = {
    "left_palm": "left_wrist",
    "right_palm": "right_wrist",
}
_CONTACT_PRIOR_REGION_FORCE_BODY_NAMES = {
    region_name: (body_name,) for region_name, body_name in _CONTACT_PRIOR_REGION_BODY_NAMES.items()
}
_CONTACT_PRIOR_REGION_POSITION_BODY_NAMES = _CONTACT_PRIOR_REGION_FORCE_BODY_NAMES
_CONTACT_PRIOR_PHASE_NAMES = ("before_pickup_anchor", "after_pickup_anchor")
_CONTACT_PRIOR_PHASE_COUNT = len(_CONTACT_PRIOR_PHASE_NAMES)
_CONTACT_PRIOR_FORCE_THRESHOLD = 1.0
_CONTACT_PRIOR_OBJECT_POS_ERROR_THRESHOLD = 0.20
_CONTACT_PRIOR_OBJECT_ROT_ERROR_THRESHOLD = 0.80
_CONTACT_PRIOR_BODY_POS_ERROR_THRESHOLD = 0.35
_CONTACT_PRIOR_CONFIDENCE_WARMUP_SAMPLES = 2048.0
_OBJECT_CONTACT_PROXY_DISTANCE_THRESHOLD = 0.08
_ADAPTIVE_SAMPLING_CONTACT_STAGE_PRE_EXTRA_STEPS = 10
_ADAPTIVE_SAMPLING_CONTACT_STAGE_RELEASE_LEAD_STEPS = 30
_CONTACT_WINDOW_OBSERVATION_FUNCTION_PATHS = frozenset(
    {
        "holosoma.managers.observation.terms.wbt.sparse_target_root_trajectory_command_contact_aware",
        "holosoma.managers.observation.terms.wbt.drop_button",
        "holosoma.managers.observation.terms.wbt.pickup_button",
    }
)


def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    first_col = F.normalize(rot6d[..., 0:3], dim=-1)
    second_col_raw = rot6d[..., 3:6]
    second_col = F.normalize(
        second_col_raw - torch.sum(first_col * second_col_raw, dim=-1, keepdim=True) * first_col,
        dim=-1,
    )
    third_col = torch.cross(first_col, second_col, dim=-1)
    return torch.stack((first_col, second_col, third_col), dim=-1)


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


def _first_sustained_true_index_from(mask: torch.Tensor, consecutive_steps: int, start_idx: int) -> int | None:
    """Return the earliest sustained-true index at or after `start_idx`."""
    if start_idx <= 0:
        return _first_sustained_true_index(mask, consecutive_steps)
    if start_idx >= int(mask.numel()):
        return None
    relative_idx = _first_sustained_true_index(mask[start_idx:], consecutive_steps)
    if relative_idx is None:
        return None
    return int(start_idx + relative_idx)


def _pickup_threshold_from_rel_z(
    rel_z: torch.Tensor,
    *,
    lift_height_threshold: float = _RUNTIME_PICKUP_LIFT_HEIGHT_THRESHOLD,
    lift_ratio_threshold: float = _CLIP_PICKUP_LIFT_RATIO_THRESHOLD,
) -> torch.Tensor:
    """Return the pickup threshold used to define the pickup-frame anchor.

    The threshold is relative-z based and matches the clip-time pickup detection rule:
    z_min + max(abs_height_threshold, lift_ratio_threshold * z_range).
    """
    if rel_z.numel() == 0:
        return rel_z.new_tensor(float(lift_height_threshold))

    z_min = rel_z.min()
    z_range = torch.clamp(rel_z.max() - z_min, min=0.0)
    return z_min + torch.maximum(
        z_min.new_tensor(float(lift_height_threshold)),
        z_range * float(lift_ratio_threshold),
    )


def _pickup_step_and_threshold_from_rel_z(
    rel_z: torch.Tensor,
    *,
    lift_height_threshold: float = _RUNTIME_PICKUP_LIFT_HEIGHT_THRESHOLD,
    lift_ratio_threshold: float = _CLIP_PICKUP_LIFT_RATIO_THRESHOLD,
    consecutive_steps: int = _RUNTIME_PICKUP_CONSECUTIVE_STEPS,
) -> tuple[int, torch.Tensor]:
    """Return the earliest sustained pickup step and its matching threshold."""
    pickup_threshold = _pickup_threshold_from_rel_z(
        rel_z,
        lift_height_threshold=lift_height_threshold,
        lift_ratio_threshold=lift_ratio_threshold,
    )
    if rel_z.numel() == 0:
        return 0, pickup_threshold

    lifted_mask = rel_z >= pickup_threshold
    pickup_step = _first_sustained_true_index(lifted_mask, consecutive_steps)
    if pickup_step is None:
        lifted_indices = torch.nonzero(lifted_mask, as_tuple=False)
        if lifted_indices.numel() > 0:
            pickup_step = int(lifted_indices[0, 0].item())
        else:
            pickup_step = int(torch.argmax(rel_z).item())
    return pickup_step, pickup_threshold


def _kinematic_lift_window_from_rel_z(
    rel_z: torch.Tensor,
    *,
    lift_height_threshold: float = _RUNTIME_PICKUP_LIFT_HEIGHT_THRESHOLD,
    lift_ratio_threshold: float = _CLIP_PICKUP_LIFT_RATIO_THRESHOLD,
    consecutive_steps: int = _RUNTIME_PICKUP_CONSECUTIVE_STEPS,
    require_sustained_lift: bool = False,
) -> tuple[int, int]:
    """Return the source-motion ``[start, end)`` object lift window.

    This primitive intentionally has no contact-sidecar input.  Automatic
    pickup/drop labels use it directly so collider contacts and root-command
    gating cannot silently change the label transition frames.
    """
    if rel_z.ndim != 1:
        raise ValueError(
            f"Kinematic button rel-z trace must be rank 1, got shape {tuple(rel_z.shape)}."
        )
    if rel_z.numel() == 0:
        return 0, 0
    if not bool(torch.isfinite(rel_z).all().item()):
        raise ValueError(
            "Kinematic button rel-z trace must contain only finite values."
        )

    pickup_threshold = _pickup_threshold_from_rel_z(
        rel_z,
        lift_height_threshold=lift_height_threshold,
        lift_ratio_threshold=lift_ratio_threshold,
    )
    lifted_mask = rel_z >= pickup_threshold
    pickup_step = _first_sustained_true_index(lifted_mask, consecutive_steps)
    if pickup_step is None:
        if require_sustained_lift:
            raise ValueError(
                "Kinematic button motion never reaches the lift threshold for "
                f"{int(consecutive_steps)} consecutive frames."
            )
        lifted_indices = torch.nonzero(lifted_mask, as_tuple=False)
        pickup_step = (
            int(lifted_indices[0, 0].item())
            if lifted_indices.numel() > 0
            else int(torch.argmax(rel_z).item())
        )

    total_steps = int(rel_z.shape[0])
    carry_start = max(0, min(int(pickup_step), total_steps))
    carry_end = total_steps

    lowered_mask = rel_z < pickup_threshold
    lowering_step = _first_sustained_true_index_from(
        lowered_mask,
        consecutive_steps,
        start_idx=min(carry_start + 1, total_steps),
    )
    if lowering_step is not None:
        carry_end = min(carry_end, int(lowering_step))

    carry_end = max(carry_start, min(carry_end, total_steps))
    return carry_start, carry_end


def _contact_aware_carry_window_from_rel_z(
    rel_z: torch.Tensor,
    *,
    contact_interval: tuple[int, int] | None = None,
    lift_height_threshold: float = _RUNTIME_PICKUP_LIFT_HEIGHT_THRESHOLD,
    lift_ratio_threshold: float = _CLIP_PICKUP_LIFT_RATIO_THRESHOLD,
    consecutive_steps: int = _RUNTIME_PICKUP_CONSECUTIVE_STEPS,
    release_lead_steps: int = _ADAPTIVE_SAMPLING_CONTACT_STAGE_RELEASE_LEAD_STEPS,
) -> tuple[int, int]:
    """Return the [start, end) carry window used by contact-aware root commands."""
    carry_start, carry_end = _kinematic_lift_window_from_rel_z(
        rel_z,
        lift_height_threshold=lift_height_threshold,
        lift_ratio_threshold=lift_ratio_threshold,
        consecutive_steps=consecutive_steps,
    )
    total_steps = int(rel_z.shape[0])

    normalized_interval = _normalize_contact_interval_pair(contact_interval) if contact_interval is not None else None
    if normalized_interval is not None:
        _, t2 = normalized_interval
        release_start = max(0, min(int(t2) - max(int(release_lead_steps), 0), total_steps))
        carry_end = min(carry_end, release_start)

    carry_end = max(carry_start, min(carry_end, total_steps))
    return carry_start, carry_end


def _smooth_1d_edge_padded(values: torch.Tensor, window_steps: int) -> torch.Tensor:
    """Return a centered moving average with edge padding, preserving length."""
    if values.numel() == 0:
        return values

    window_steps = max(1, int(window_steps))
    if window_steps <= 1:
        return values

    left_pad = window_steps // 2
    right_pad = window_steps - 1 - left_pad
    padded_parts = []
    if left_pad > 0:
        padded_parts.append(values[:1].expand(left_pad))
    padded_parts.append(values)
    if right_pad > 0:
        padded_parts.append(values[-1:].expand(right_pad))
    padded = torch.cat(padded_parts, dim=0)
    kernel = torch.full((1, 1, window_steps), 1.0 / float(window_steps), device=values.device, dtype=values.dtype)
    return F.conv1d(padded.view(1, 1, -1), kernel).view(-1)


def _contact_aware_carry_window_from_peak_height(
    object_height: torch.Tensor,
    *,
    peak_height_alpha: float = 0.91,
    smoothing_steps: int = 5,
    consecutive_steps: int = _RUNTIME_PICKUP_CONSECUTIVE_STEPS,
) -> tuple[int, int]:
    """Return the [start, end) carry window from the high plateau of object world height."""
    if object_height.numel() == 0:
        return 0, 0

    height = _smooth_1d_edge_padded(object_height, smoothing_steps)
    total_steps = int(height.shape[0])
    alpha = max(0.0, min(float(peak_height_alpha), 1.0))

    h_min = height.min()
    h_peak = height.max()
    threshold = h_min + torch.clamp(h_peak - h_min, min=0.0) * alpha
    high_mask = height >= threshold

    carry_start = _first_sustained_true_index(high_mask, consecutive_steps)
    if carry_start is None:
        high_indices = torch.nonzero(high_mask, as_tuple=False)
        if high_indices.numel() > 0:
            carry_start = int(high_indices[0, 0].item())
        else:
            carry_start = int(torch.argmax(height).item())
    carry_start = max(0, min(int(carry_start), total_steps))

    peak_step = int(torch.argmax(height).item())
    carry_end = _first_sustained_true_index_from(
        ~high_mask,
        consecutive_steps,
        start_idx=min(peak_step + 1, total_steps),
    )
    if carry_end is None:
        carry_end = total_steps
    carry_end = max(carry_start, min(int(carry_end), total_steps))
    return carry_start, carry_end


def _normalize_contact_prior_region_name(raw_name: str) -> str:
    name = str(raw_name).strip()
    if not name:
        return ""
    return _CONTACT_PRIOR_REGION_ALIASES.get(name, name)


def _compute_contact_stage_intervals(
    *,
    t1: int,
    t2: int,
    sample_end_step: float,
) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    total_end = max(min(float(t2), float(sample_end_step)), 0.0)
    if total_end <= 0.0:
        zero_intervals = [(0.0, 0.0)] * 5
        return zero_intervals, (0.0, max(float(sample_end_step), 0.0))

    left_anchor = min(max(float(t1 + _ADAPTIVE_SAMPLING_CONTACT_STAGE_PRE_EXTRA_STEPS), 0.0), total_end)
    right_anchor = min(
        max(float(t2 - _ADAPTIVE_SAMPLING_CONTACT_STAGE_RELEASE_LEAD_STEPS), 0.0),
        total_end,
    )
    right_anchor = max(right_anchor, left_anchor)

    middle_length = max(right_anchor - left_anchor, 0.0)
    middle_step = middle_length / 3.0
    stage_intervals = [
        (0.0, left_anchor),
        (left_anchor, left_anchor + middle_step),
        (left_anchor + middle_step, left_anchor + 2.0 * middle_step),
        (left_anchor + 2.0 * middle_step, right_anchor),
        (right_anchor, total_end),
    ]
    after_t2_interval = (total_end, max(float(sample_end_step), total_end))
    return stage_intervals, after_t2_interval


def _probability_mass_on_intervals(
    bin_probabilities: torch.Tensor,
    *,
    sample_end_step: float,
    intervals: list[tuple[float, float]],
) -> torch.Tensor:
    """Integrate a discrete probability vector over continuous intervals.

    This routine is part of diagnostic telemetry, but it runs on CUDA tensors
    during collection.  The previous scalar loop called ``.item()`` once per
    bin, turning a small reduction into thousands of device synchronizations
    across the motion bank.  Build the complete bin/interval overlap matrix on
    device and perform one matrix-vector reduction instead.
    """

    masses = torch.zeros((len(intervals),), device=bin_probabilities.device, dtype=torch.float32)
    if bin_probabilities.numel() == 0:
        return masses
    sample_end = float(sample_end_step)
    if sample_end <= 0.0:
        return masses

    num_bins = int(bin_probabilities.numel())
    bin_width = sample_end / float(max(num_bins, 1))
    if bin_width <= 0.0:
        return masses

    if not intervals:
        return masses
    interval_tensor = torch.as_tensor(
        intervals,
        device=bin_probabilities.device,
        dtype=torch.float32,
    )
    bin_starts = torch.arange(
        num_bins,
        device=bin_probabilities.device,
        dtype=torch.float32,
    ) * float(bin_width)
    bin_ends = bin_starts + float(bin_width)
    overlap = torch.minimum(bin_ends[:, None], interval_tensor[None, :, 1]) - torch.maximum(
        bin_starts[:, None], interval_tensor[None, :, 0]
    )
    overlap_weights = torch.clamp(overlap, min=0.0) / float(bin_width)
    return torch.matmul(bin_probabilities.to(dtype=torch.float32), overlap_weights)


#########################################################################################################
## MotionLoader and AdaptiveTimestepsSampler
#########################################################################################################
class MotionLoader:
    _PRECOMPUTED_ROOT_COMMAND_KEY = "policy_command_xy_yaw"
    _PRECOMPUTED_ROOT_COMMAND_PHASE_KEY = "policy_command_phase"
    _OBJECT_SIZE_KEYS = (
        "object_size",
        "box_size",
    )
    _OBJECT_SCALE_KEYS = ("object_scale", "box_scale")

    def __init__(
        self,
        motion_file: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
        motion_clip_id: int | None = None,
        motion_clip_name: str | None = None,
        object_size_scale: list[float] | None = None,
        allowed_object_categories: list[str] | None = None,
    ):
        self._robot_body_names = list(robot_body_names)
        self._robot_joint_names = list(robot_joint_names)
        self._object_size_scale = self._normalize_object_size_scale(object_size_scale)
        self._allowed_object_categories = self._normalize_allowed_object_categories(allowed_object_categories)

        # Resolve the motion file path using importlib.resources
        motion_file = resolve_data_file_path(motion_file)
        motion_file = resolve_rank_local_motion_path(motion_file)
        motion_path = Path(motion_file)

        # Read transition lineage from the effective (possibly rank-local)
        # object map before MotionLoader pins or filters clips.  A one-clip
        # derivative of a global bank must retain the source bank's runtime
        # timeline instead of being reclassified from its active clip count.
        self.motion_transition_source = resolve_motion_transition_source_for_motion_path(
            motion_path,
        )

        logger.info(f"Loading motion file: {motion_file}")
        self.clip_ids: list[str] = []
        self.clip_object_names: list[str] = []
        self.clip_object_urdf_paths: list[str] = []
        self.clip_offsets = torch.zeros(0, dtype=torch.long, device=device)
        self.clip_lengths = torch.zeros(0, dtype=torch.long, device=device)
        self.num_clips = 0
        self._precomputed_root_command: torch.Tensor | None = None
        self._precomputed_root_command_phase: torch.Tensor | None = None
        self.motion_clip_id = motion_clip_id
        self.motion_clip_name = motion_clip_name
        if motion_path.is_dir():
            body_names_in_motion_data, joint_names_in_motion_data = self._load_data_from_motion_npz_dir(
                motion_path,
                device,
                motion_clip_id=motion_clip_id,
                motion_clip_name=motion_clip_name,
            )
        elif motion_file.endswith((".h5", ".hdf5")):
            body_names_in_motion_data, joint_names_in_motion_data = self._load_data_from_motion_h5(
                motion_file,
                device,
                motion_clip_id=motion_clip_id,
                motion_clip_name=motion_clip_name,
            )
        else:
            body_names_in_motion_data, joint_names_in_motion_data = self._load_data_from_motion_npz(motion_file, device)
        body_indexes = self._get_index_of_a_in_b(robot_body_names, body_names_in_motion_data, device)
        joint_indexes = self._get_index_of_a_in_b(robot_joint_names, joint_names_in_motion_data, device)

        # All consumers operate in simulator/robot order.  Keeping the loaded
        # tensors in source order made every property access materialize the
        # entire motion bank through advanced indexing.  Canonicalize once at
        # load time instead; the identity indexes retained below also keep the
        # static transition write-back helpers internally consistent.
        self._canonicalize_robot_order(joint_indexes=joint_indexes, body_indexes=body_indexes)
        self.time_step_total = self._joint_pos.shape[0]
        self._apply_object_size_scale()

    def _canonicalize_robot_order(
        self,
        *,
        joint_indexes: torch.Tensor,
        body_indexes: torch.Tensor,
    ) -> None:
        """Store all robot motion tensors in canonical simulator order.

        Source clips may contain permuted or additional joints/bodies.  The
        public properties have always exposed only the configured robot in
        robot order, so dropping source-only columns does not change their
        contract.  Once canonicalized, both reads and transition splices use
        the same identity mapping rather than mixing source-order backing
        tensors with robot-order views.
        """

        self._joint_pos = self._joint_pos.index_select(1, joint_indexes).contiguous()
        self._joint_vel = self._joint_vel.index_select(1, joint_indexes).contiguous()
        self._body_pos_w = self._body_pos_w.index_select(1, body_indexes).contiguous()
        self._body_quat_w = self._body_quat_w.index_select(1, body_indexes).contiguous()
        self._body_lin_vel_w = self._body_lin_vel_w.index_select(1, body_indexes).contiguous()
        self._body_ang_vel_w = self._body_ang_vel_w.index_select(1, body_indexes).contiguous()

        self._joint_indexes = torch.arange(
            self._joint_pos.shape[1],
            device=self._joint_pos.device,
            dtype=torch.long,
        )
        self._body_indexes = torch.arange(
            self._body_pos_w.shape[1],
            device=self._body_pos_w.device,
            dtype=torch.long,
        )

    @staticmethod
    def _normalize_object_size_scale(raw_scale: list[float] | None) -> np.ndarray | None:
        if raw_scale is None:
            return None
        arr = np.asarray(raw_scale, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None
        if arr.size == 1:
            value = float(arr[0])
            normalized = np.array([value, value, value], dtype=np.float32)
        elif arr.size == 3:
            normalized = arr.astype(np.float32, copy=False)
        else:
            raise ValueError(
                "MotionConfig.object_size_scale must have length 1 or 3. "
                f"Got shape {arr.shape} from value {raw_scale!r}."
            )
        if not np.all(np.isfinite(normalized)) or np.any(normalized <= 0.0):
            raise ValueError(
                "MotionConfig.object_size_scale must contain finite positive scale factors. "
                f"Got {raw_scale!r}."
            )
        return normalized

    @staticmethod
    def _normalize_allowed_object_categories(raw_categories: list[str] | None) -> set[str]:
        if raw_categories is None:
            return set()
        aliases = {
            "boxes": "box",
            "cube": "box",
            "cubes": "box",
            "largebox": "box",
            "largeboxes": "box",
            "trash": "bin",
            "trashcan": "bin",
            "trashcans": "bin",
            "basket": "bin",
            "baskets": "bin",
            "bins": "bin",
            "barrels": "barrel",
            "sphere": "ball",
            "spheres": "ball",
            "balls": "ball",
        }
        normalized: set[str] = set()
        for value in raw_categories:
            category = str(value).strip().lower().replace("-", "_")
            if not category:
                continue
            normalized.add(aliases.get(category, category))
        return normalized

    @classmethod
    def _object_category_for_clip(cls, clip_id: str, clip_entry: dict[str, str] | None) -> str:
        parts = [clip_id]
        if clip_entry:
            for key in ("object_name", "object_urdf_path", "object_mesh_path", "object_category", "category", "object_type"):
                value = str(clip_entry.get(key, "")).strip()
                if value:
                    if key.endswith("_path"):
                        path = Path(value)
                        parts.extend([path.name, path.stem])
                    else:
                        parts.append(value)
        raw = " ".join(parts).lower().replace("-", "_")
        if "barrel" in raw:
            return "barrel"
        if "bin" in raw or "trash" in raw or "basket" in raw:
            return "bin"
        if "ball" in raw or "sphere" in raw:
            return "ball"
        if "box" in raw or "cube" in raw or "largebox" in raw:
            return "box"
        return "other"

    def _apply_object_size_scale(self) -> None:
        if self._object_size_scale is None or not hasattr(self, "_object_size"):
            return
        if self._object_size.numel() == 0:
            return
        scale = torch.tensor(self._object_size_scale, dtype=self._object_size.dtype, device=self._object_size.device)
        self._object_size = self._object_size * scale.view(1, 3)

    @classmethod
    def _normalize_object_size_array(cls, raw: np.ndarray, length: int, *, source: str) -> np.ndarray:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 0:
            scalar = float(arr)
            return np.full((length, 3), scalar, dtype=np.float32)

        if arr.ndim == 1:
            if arr.shape[0] == 1:
                return np.full((length, 3), float(arr[0]), dtype=np.float32)
            if arr.shape[0] == 3:
                return np.repeat(arr.reshape(1, 3), repeats=length, axis=0)
            if arr.shape[0] == length:
                return np.repeat(arr.reshape(length, 1), repeats=3, axis=1)

        if arr.ndim == 2:
            if arr.shape == (1, 3):
                return np.repeat(arr, repeats=length, axis=0)
            if arr.shape == (length, 1):
                return np.repeat(arr, repeats=3, axis=1)
            if arr.shape == (length, 3):
                return arr

        raise ValueError(
            f"Unsupported object-size shape {arr.shape} in {source}; "
            "expected scalar, (3,), (T,), (T,3), (1,3), or (T,1)."
        )

    @classmethod
    def _extract_object_size_np(cls, data: Any, length: int, *, source: str) -> np.ndarray:
        for key in cls._OBJECT_SIZE_KEYS:
            if key in data:
                raw = np.asarray(data[key])
                if raw.dtype.kind != "f":
                    raise ValueError(
                        f"Motion field {key} in {source} must use a real floating dtype, got {raw.dtype}."
                    )
                normalized = cls._normalize_object_size_array(raw, length, source=f"{source}:{key}")
                if not np.all(np.isfinite(normalized)) or np.any(normalized <= 0.0):
                    raise ValueError(
                        f"Motion field {key} in {source} must contain finite positive physical extents."
                    )
                return normalized
        scale_keys = [key for key in cls._OBJECT_SCALE_KEYS if key in data]
        if scale_keys:
            raise ValueError(
                f"Motion file {source} provides mesh scale field(s) {scale_keys} but no physical "
                "object_size/box_size extents; scale and size are not interchangeable."
            )
        return np.ones((length, 3), dtype=np.float32)

    @classmethod
    def _extract_precomputed_root_command_np(
        cls,
        data: Any,
        length: int,
        *,
        source: str,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        command_present = cls._PRECOMPUTED_ROOT_COMMAND_KEY in data
        phase_present = cls._PRECOMPUTED_ROOT_COMMAND_PHASE_KEY in data
        if command_present != phase_present:
            raise ValueError(
                f"Motion file {source} must contain both {cls._PRECOMPUTED_ROOT_COMMAND_KEY!r} "
                f"and {cls._PRECOMPUTED_ROOT_COMMAND_PHASE_KEY!r}."
            )
        if not command_present:
            return None

        command = np.asarray(data[cls._PRECOMPUTED_ROOT_COMMAND_KEY])
        phase = np.asarray(data[cls._PRECOMPUTED_ROOT_COMMAND_PHASE_KEY])
        if command.dtype.kind != "f" or command.shape != (length, 3):
            raise ValueError(
                f"Motion field {cls._PRECOMPUTED_ROOT_COMMAND_KEY} in {source} must be a "
                f"floating array with shape ({length}, 3), got {command.dtype} {command.shape}."
            )
        if phase.dtype.kind not in "iu" or phase.shape != (length,):
            raise ValueError(
                f"Motion field {cls._PRECOMPUTED_ROOT_COMMAND_PHASE_KEY} in {source} must be an "
                f"integer array with shape ({length},), got {phase.dtype} {phase.shape}."
            )
        if not np.all(np.isfinite(command)):
            raise ValueError(f"Precomputed root command in {source} contains non-finite values.")

        phase_i64 = phase.astype(np.int64, copy=False)
        if not np.all(np.isin(phase_i64, (0, 1, 2))):
            invalid = np.unique(phase_i64[~np.isin(phase_i64, (0, 1, 2))]).tolist()
            raise ValueError(f"Precomputed root command phase in {source} contains invalid values: {invalid}")

        zero_phase = phase_i64 == 0
        forward_phase = phase_i64 == 1
        yaw_phase = phase_i64 == 2
        if np.any(command[:, 1] != 0.0):
            raise ValueError(f"Precomputed turn-then-forward command in {source} must keep dy exactly zero.")
        if np.any((command[:, 0] != 0.0) & (command[:, 2] != 0.0)):
            raise ValueError(f"Precomputed turn-then-forward command in {source} couples dx and dyaw.")
        if np.any(command[zero_phase] != 0.0):
            raise ValueError(f"Zero-phase precomputed command rows in {source} must be zero.")
        if np.any(command[forward_phase, 0] <= 0.0) or np.any(command[forward_phase, 2] != 0.0):
            raise ValueError(f"Forward-phase precomputed command rows in {source} are inconsistent.")
        if np.any(command[yaw_phase, 0] != 0.0) or np.any(command[yaw_phase, 2] == 0.0):
            raise ValueError(f"Yaw-phase precomputed command rows in {source} are inconsistent.")
        if np.any(command[:, 0] < 0.0) or np.any(command[:, 0] > 10.0):
            raise ValueError(f"Precomputed forward command in {source} must lie in [0, 10] metres.")
        if np.any(np.abs(command[:, 2]) > math.pi):
            raise ValueError(f"Precomputed yaw command in {source} must lie in [-pi, pi].")
        return command.astype(np.float32, copy=False), phase_i64.astype(np.uint8, copy=False)

    @staticmethod
    def _scalar_str(value: Any) -> str:
        arr = np.asarray(value)
        if arr.size == 0:
            return ""
        if arr.shape == ():
            item = arr.item()
        else:
            item = arr.reshape(-1)[0]
            if hasattr(item, "item"):
                item = item.item()
        return str(item).strip()

    @classmethod
    def _load_clip_object_metadata_map(cls, motion_dir: Path) -> dict[str, dict[str, str]]:
        candidate_files = (
            motion_dir / "_clip_object_urdf_map.json",
            motion_dir / "clip_object_urdf_map.json",
        )
        for path in candidate_files:
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to parse clip-object metadata map '{}': {}", path, exc)
                return {}

            if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
                payload = payload["clips"]
            if not isinstance(payload, dict):
                logger.warning("Invalid clip-object metadata map format in '{}': expected dict.", path)
                return {}

            normalized: dict[str, dict[str, str]] = {}
            for clip_id, entry in payload.items():
                if not isinstance(clip_id, str):
                    continue
                if isinstance(entry, str):
                    normalized[clip_id] = {"object_name": "", "object_urdf_path": entry.strip()}
                elif isinstance(entry, dict):
                    obj_name = str(entry.get("object_name", "")).strip()
                    obj_urdf = str(entry.get("object_urdf_path", "")).strip()
                    normalized[clip_id] = {"object_name": obj_name, "object_urdf_path": obj_urdf}
            logger.info("Loaded clip-object metadata map '{}' ({} entries).", path, len(normalized))
            return normalized
        return {}

    @classmethod
    def _extract_object_clip_metadata(
        cls,
        *,
        data: Any,
        clip_id: str,
        clip_map: dict[str, dict[str, str]] | None = None,
        base_dir: Path,
    ) -> tuple[str, str]:
        object_name = cls._scalar_str(data["object_name"]) if "object_name" in data else ""
        object_urdf_path = cls._scalar_str(data["object_urdf_path"]) if "object_urdf_path" in data else ""

        if clip_map is not None and clip_id in clip_map:
            mapped = clip_map[clip_id]
            mapped_name = mapped.get("object_name", "").strip()
            mapped_urdf = mapped.get("object_urdf_path", "").strip()
            if mapped_name:
                object_name = mapped_name
            if mapped_urdf:
                object_urdf_path = mapped_urdf

        if object_urdf_path:
            object_urdf_path = cls._resolve_motion_object_urdf_path(object_urdf_path, base_dir=base_dir)
        if not object_name and object_urdf_path:
            object_name = Path(object_urdf_path).stem
        if not object_name:
            object_name = "object"
        return object_name, object_urdf_path

    @staticmethod
    def _resolve_motion_object_urdf_path(raw_path: str, *, base_dir: Path) -> str:
        path_str = str(raw_path).strip()
        if not path_str:
            return ""
        candidate = Path(path_str)
        if not candidate.is_absolute() and not path_str.startswith("holosoma/data"):
            candidate = (base_dir / path_str).resolve()
            return str(candidate)
        return str(Path(resolve_data_file_path(path_str)).resolve())

    @staticmethod
    def _format_motion_file_issues(issues: list[tuple[Path, str]], limit: int = 5) -> str:
        if not issues:
            return ""
        sample = "; ".join(f"{path}: {reason}" for path, reason in issues[:limit])
        remaining = len(issues) - min(len(issues), limit)
        if remaining > 0:
            sample = f"{sample}; +{remaining} more"
        return sample

    @staticmethod
    def _filter_valid_npz_archives(files: list[Path]) -> tuple[list[Path], list[tuple[Path, str]]]:
        valid_files: list[Path] = []
        invalid_files: list[tuple[Path, str]] = []
        for path in files:
            try:
                if zipfile.is_zipfile(path):
                    valid_files.append(path)
                else:
                    invalid_files.append((path, "not a valid .npz zip archive"))
            except OSError as exc:
                invalid_files.append((path, f"{type(exc).__name__}: {exc}"))
        return valid_files, invalid_files

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)

    def _resolve_body_subset_indexes(
        self,
        body_names_clip: list[str],
        *,
        source: str,
    ) -> tuple[list[str], np.ndarray]:
        """Select body indexes in clip order according to configured robot body names.

        This makes multi-clip loading robust when clips include extra scene bodies whose
        count varies per clip.
        """
        name_to_idx: dict[str, int] = {}
        duplicates: list[str] = []
        for idx, name in enumerate(body_names_clip):
            if name in name_to_idx:
                duplicates.append(name)
                continue
            name_to_idx[name] = idx
        if duplicates:
            dup_sorted = sorted(set(duplicates))
            raise ValueError(f"Duplicate body names in {source}: {dup_sorted}")

        missing = [name for name in self._robot_body_names if name not in name_to_idx]
        if missing:
            raise ValueError(f"Missing robot body names in {source}: {missing}")

        body_indexes = np.array([name_to_idx[name] for name in self._robot_body_names], dtype=np.int64)
        return list(self._robot_body_names), body_indexes

    def _set_clip_metadata(
        self,
        clip_ids: list[str],
        offsets: np.ndarray,
        lengths: np.ndarray,
        device: str,
        clip_object_names: list[str] | None = None,
        clip_object_urdf_paths: list[str] | None = None,
    ) -> None:
        self.clip_ids = clip_ids
        if clip_object_names is None:
            clip_object_names = [""] * len(clip_ids)
        if clip_object_urdf_paths is None:
            clip_object_urdf_paths = [""] * len(clip_ids)
        if len(clip_object_names) != len(clip_ids):
            raise ValueError("clip_object_names length must match clip_ids length")
        if len(clip_object_urdf_paths) != len(clip_ids):
            raise ValueError("clip_object_urdf_paths length must match clip_ids length")
        self.clip_object_names = clip_object_names
        self.clip_object_urdf_paths = clip_object_urdf_paths
        self.clip_offsets = torch.tensor(offsets, dtype=torch.long, device=device)
        self.clip_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.num_clips = len(clip_ids)

    def _load_data_from_motion_npz(self, motion_file: str, device: str) -> tuple[list[str], list[str]]:
        clip_id = Path(motion_file).stem
        clip_object_names: list[str] | None = None
        clip_object_urdfs: list[str] | None = None
        try:
            with smart_open.open(motion_file, "rb") as f, np.load(f, allow_pickle=False) as data:
                self.fps = data["fps"]

                body_names = data["body_names"].tolist()
                joint_names = data["joint_names"].tolist()

                # The first 7 joints_pos are [xyz, wxyz] of the pelvis, omit them from the joint_pos
                # The first 6 joints_vel are [vel_xyz, vel_wxyz] of the pelvis, omit them from the joint_vel
                # We'll use the pelvis position and quaternion from body_pos_w[:, 0] and body_quat_w[:, 0] directly.
                self._joint_pos = torch.tensor(data["joint_pos"][:, 7:], dtype=torch.float32, device=device)
                self._joint_vel = torch.tensor(data["joint_vel"][:, 6:], dtype=torch.float32, device=device)
                assert len(joint_names) == self._joint_pos.shape[1], "Joint names in motion data does not match"

                self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
                assert len(body_names) == self._body_pos_w.shape[1], "Body names in motion data does not match"

                # NOTE: wxyz after loading from npz
                body_quat_w_wxyz = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)  # This is wxyz
                self._body_quat_w = body_quat_w_wxyz[:, :, [1, 2, 3, 0]]  # Change to xyzw

                self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
                self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)

                precomputed_command = self._extract_precomputed_root_command_np(
                    data,
                    int(self._joint_pos.shape[0]),
                    source=motion_file,
                )
                if precomputed_command is not None:
                    self._precomputed_root_command = torch.tensor(
                        precomputed_command[0], dtype=torch.float32, device=device
                    )
                    self._precomputed_root_command_phase = torch.tensor(
                        precomputed_command[1], dtype=torch.uint8, device=device
                    )

                # add object pos and quat
                self.has_object = "object_pos_w" in data
                if self.has_object:
                    length = int(self._joint_pos.shape[0])
                    # NOTE: wxyz after loading from npz
                    self._object_pos_w = torch.tensor(data["object_pos_w"], dtype=torch.float32, device=device)
                    object_quat_w = torch.tensor(data["object_quat_w"], dtype=torch.float32, device=device)
                    self._object_quat_w = object_quat_w[:, [1, 2, 3, 0]]  # Change to xyzw
                    self._object_lin_vel_w = torch.tensor(data["object_lin_vel_w"], dtype=torch.float32, device=device)
                    object_size = self._extract_object_size_np(data, length, source=motion_file)
                    self._object_size = torch.tensor(object_size, dtype=torch.float32, device=device)
                    obj_name, obj_urdf = self._extract_object_clip_metadata(
                        data=data,
                        clip_id=clip_id,
                        clip_map=None,
                        base_dir=Path(motion_file).parent,
                    )
                    clip_object_names = [obj_name]
                    clip_object_urdfs = [obj_urdf]
                else:
                    self._object_pos_w = torch.zeros(0, 3, device=device)
                    self._object_quat_w = torch.zeros(0, 4, device=device)
                    self._object_lin_vel_w = torch.zeros(0, 3, device=device)
                    self._object_size = torch.zeros(0, 3, device=device)
        except (AssertionError, KeyError, zipfile.BadZipFile, EOFError, OSError, ValueError) as exc:
            raise zipfile.BadZipFile(f"Failed to load motion npz '{motion_file}': {exc}") from exc
        length = int(self._joint_pos.shape[0])
        self._set_clip_metadata(
            [clip_id],
            np.array([0]),
            np.array([length]),
            device,
            clip_object_names=clip_object_names,
            clip_object_urdf_paths=clip_object_urdfs,
        )
        return body_names, joint_names

    def _load_data_from_motion_npz_dir(
        self,
        motion_dir: Path,
        device: str,
        motion_clip_id: int | None,
        motion_clip_name: str | None,
    ) -> tuple[list[str], list[str]]:
        clip_object_map = self._load_clip_object_metadata_map(motion_dir)
        files = sorted(motion_dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found in motion directory: {motion_dir}")

        if self._allowed_object_categories:
            files = [
                file_path
                for file_path in files
                if self._object_category_for_clip(file_path.stem, clip_object_map.get(file_path.stem, {}))
                in self._allowed_object_categories
            ]
            if not files:
                raise ValueError(
                    "No motion clips matched allowed object categories "
                    f"{sorted(self._allowed_object_categories)} in {motion_dir}"
                )
            logger.info(
                "Filtered motion directory '{}' by allowed object categories {} -> {} clips.",
                motion_dir,
                sorted(self._allowed_object_categories),
                len(files),
            )

        if motion_clip_name is not None:
            matches = [path for path in files if path.stem == motion_clip_name]
            if not matches:
                raise ValueError(f"Clip name '{motion_clip_name}' not found in {motion_dir}")
            files = matches
        elif motion_clip_id is not None:
            clip_idx = int(motion_clip_id)
            if clip_idx < 0 or clip_idx >= len(files):
                raise IndexError(f"Clip index {clip_idx} out of range for {motion_dir}")
            files = [files[clip_idx]]

        files, invalid_archives = self._filter_valid_npz_archives(files)
        if invalid_archives:
            issue_summary = self._format_motion_file_issues(invalid_archives)
            if not files:
                raise zipfile.BadZipFile(
                    f"No valid motion clips remain in {motion_dir}. Invalid clips: {issue_summary}"
                )
            logger.warning(
                "Skipping {} invalid motion clips in '{}'. Examples: {}",
                len(invalid_archives),
                motion_dir,
                issue_summary,
            )

        if len(files) == 1:
            body_names, joint_names = self._load_data_from_motion_npz(str(files[0]), device)
            clip_entry = clip_object_map.get(files[0].stem, {})
            if self.has_object and clip_entry:
                mapped_name = str(clip_entry.get("object_name", "")).strip()
                mapped_urdf = str(clip_entry.get("object_urdf_path", "")).strip()
                if mapped_urdf:
                    mapped_urdf = self._resolve_motion_object_urdf_path(mapped_urdf, base_dir=files[0].parent)
                if self.clip_object_names:
                    if mapped_name:
                        self.clip_object_names[0] = mapped_name
                if self.clip_object_urdf_paths:
                    if mapped_urdf:
                        self.clip_object_urdf_paths[0] = mapped_urdf
            return body_names, joint_names

        required_keys = (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "joint_names",
            "body_names",
            "fps",
        )
        object_keys = ("object_pos_w", "object_quat_w", "object_lin_vel_w")

        joint_names: list[str] = []
        body_names: list[str] = []
        fps_ref: float | None = None
        has_object: bool | None = None

        clip_ids: list[str] = []
        offsets: list[int] = []
        lengths: list[int] = []
        offset = 0

        joint_pos_list: list[np.ndarray] = []
        joint_vel_list: list[np.ndarray] = []
        body_pos_list: list[np.ndarray] = []
        body_quat_list: list[np.ndarray] = []
        body_lin_vel_list: list[np.ndarray] = []
        body_ang_vel_list: list[np.ndarray] = []
        object_pos_list: list[np.ndarray] = []
        object_quat_list: list[np.ndarray] = []
        object_lin_vel_list: list[np.ndarray] = []
        object_size_list: list[np.ndarray] = []
        precomputed_command_list: list[np.ndarray] = []
        precomputed_phase_list: list[np.ndarray] = []
        clips_without_precomputed_command: list[str] = []

        clip_object_names: list[str] = []
        clip_object_urdfs: list[str] = []
        late_load_failures: list[tuple[Path, str]] = []

        for file_path in files:
            try:
                data_file = np.load(file_path, allow_pickle=False)
            except (zipfile.BadZipFile, EOFError, OSError, ValueError) as exc:
                late_load_failures.append((file_path, f"{type(exc).__name__}: {exc}"))
                continue

            with data_file as data:
                try:
                    missing = [key for key in required_keys if key not in data]
                    if missing:
                        raise KeyError(f"Missing keys in {file_path}: {missing}")

                    clip_has_object = "object_pos_w" in data
                    if clip_has_object:
                        for key in object_keys:
                            if key not in data:
                                raise KeyError(f"Missing object key '{key}' in {file_path}")
                    if has_object is None:
                        has_object = clip_has_object
                    elif has_object != clip_has_object:
                        raise ValueError("Object fields are inconsistent across clips.")

                    joint_names_clip = self._decode_h5_strings(np.asarray(data["joint_names"]))
                    body_names_clip_raw = self._decode_h5_strings(np.asarray(data["body_names"]))
                    body_names_clip, body_indexes_clip = self._resolve_body_subset_indexes(
                        body_names_clip_raw,
                        source=str(file_path),
                    )
                    if not joint_names:
                        joint_names = joint_names_clip
                    elif joint_names_clip != joint_names:
                        raise ValueError(f"Joint names mismatch in {file_path}")
                    if not body_names:
                        body_names = body_names_clip
                    elif body_names_clip != body_names:
                        raise ValueError(f"Body names mismatch in {file_path}")

                    fps_arr = np.array(data["fps"]).reshape(-1)
                    fps = float(fps_arr[0]) if fps_arr.size > 0 else 30.0
                    if fps_ref is None:
                        fps_ref = fps
                    elif abs(fps_ref - fps) > 1e-6:
                        raise ValueError(f"FPS mismatch in {file_path}: {fps} != {fps_ref}")

                    joint_pos = np.asarray(data["joint_pos"])
                    length = int(joint_pos.shape[0])
                    precomputed_command = self._extract_precomputed_root_command_np(
                        data,
                        length,
                        source=str(file_path),
                    )
                    if precomputed_command is None:
                        clips_without_precomputed_command.append(file_path.stem)
                    else:
                        precomputed_command_list.append(precomputed_command[0])
                        precomputed_phase_list.append(precomputed_command[1])

                    clip_ids.append(file_path.stem)
                    offsets.append(offset)
                    lengths.append(length)
                    offset += length

                    joint_pos_list.append(joint_pos)
                    joint_vel_list.append(np.asarray(data["joint_vel"]))
                    body_pos = np.asarray(data["body_pos_w"])
                    body_quat = np.asarray(data["body_quat_w"])
                    body_lin_vel = np.asarray(data["body_lin_vel_w"])
                    body_ang_vel = np.asarray(data["body_ang_vel_w"])
                    expected_bodies = len(body_names_clip_raw)
                    for key, arr in (
                        ("body_pos_w", body_pos),
                        ("body_quat_w", body_quat),
                        ("body_lin_vel_w", body_lin_vel),
                        ("body_ang_vel_w", body_ang_vel),
                    ):
                        if arr.shape[1] != expected_bodies:
                            raise ValueError(
                                f"{key} body dimension mismatch in {file_path}: "
                                f"{arr.shape[1]} != {expected_bodies}"
                            )

                    body_pos_list.append(body_pos[:, body_indexes_clip])
                    body_quat_list.append(body_quat[:, body_indexes_clip])
                    body_lin_vel_list.append(body_lin_vel[:, body_indexes_clip])
                    body_ang_vel_list.append(body_ang_vel[:, body_indexes_clip])

                    if clip_has_object:
                        object_pos_list.append(np.asarray(data["object_pos_w"]))
                        object_quat_list.append(np.asarray(data["object_quat_w"]))
                        object_lin_vel_list.append(np.asarray(data["object_lin_vel_w"]))
                        object_size_list.append(
                            self._extract_object_size_np(data, length, source=str(file_path))
                        )
                        obj_name, obj_urdf = self._extract_object_clip_metadata(
                            data=data,
                            clip_id=file_path.stem,
                            clip_map=clip_object_map,
                            base_dir=file_path.parent,
                        )
                        clip_object_names.append(obj_name)
                        clip_object_urdfs.append(obj_urdf)
                    else:
                        clip_object_names.append("")
                        clip_object_urdfs.append("")
                except (AssertionError, KeyError, ValueError) as exc:
                    late_load_failures.append((file_path, f"{type(exc).__name__}: {exc}"))
                    continue

        if late_load_failures:
            issue_summary = self._format_motion_file_issues(late_load_failures)
            raise zipfile.BadZipFile(
                f"Failed to load {len(late_load_failures)} motion clip(s) from {motion_dir}; "
                "refusing to silently change the training distribution. "
                f"Examples: {issue_summary}"
            )

        if precomputed_command_list and clips_without_precomputed_command:
            sample = clips_without_precomputed_command[:5]
            raise ValueError(
                "Precomputed root command coverage is partial across the motion directory; "
                f"missing={sample}, missing_count={len(clips_without_precomputed_command)}."
            )

        self.fps = float(fps_ref) if fps_ref is not None else 30.0
        self._set_clip_metadata(
            clip_ids,
            np.array(offsets),
            np.array(lengths),
            device,
            clip_object_names=clip_object_names,
            clip_object_urdf_paths=clip_object_urdfs,
        )

        joint_pos = np.concatenate(joint_pos_list, axis=0)
        joint_vel = np.concatenate(joint_vel_list, axis=0)
        body_pos_w = np.concatenate(body_pos_list, axis=0)
        body_quat_w = np.concatenate(body_quat_list, axis=0)
        body_lin_vel_w = np.concatenate(body_lin_vel_list, axis=0)
        body_ang_vel_w = np.concatenate(body_ang_vel_list, axis=0)

        self._joint_pos = torch.tensor(joint_pos[:, 7:], dtype=torch.float32, device=device)
        self._joint_vel = torch.tensor(joint_vel[:, 6:], dtype=torch.float32, device=device)
        assert len(joint_names) == self._joint_pos.shape[1], "Joint names in motion data does not match"

        self._body_pos_w = torch.tensor(body_pos_w, dtype=torch.float32, device=device)
        assert len(body_names) == self._body_pos_w.shape[1], "Body names in motion data does not match"

        body_quat_w_wxyz = torch.tensor(body_quat_w, dtype=torch.float32, device=device)
        self._body_quat_w = body_quat_w_wxyz[:, :, [1, 2, 3, 0]]

        self._body_lin_vel_w = torch.tensor(body_lin_vel_w, dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(body_ang_vel_w, dtype=torch.float32, device=device)
        if precomputed_command_list:
            self._precomputed_root_command = torch.tensor(
                np.concatenate(precomputed_command_list, axis=0),
                dtype=torch.float32,
                device=device,
            )
            self._precomputed_root_command_phase = torch.tensor(
                np.concatenate(precomputed_phase_list, axis=0),
                dtype=torch.uint8,
                device=device,
            )

        self.has_object = bool(has_object)
        if self.has_object:
            object_pos_w = np.concatenate(object_pos_list, axis=0)
            object_quat_w = np.concatenate(object_quat_list, axis=0)
            object_lin_vel_w = np.concatenate(object_lin_vel_list, axis=0)
            object_size = np.concatenate(object_size_list, axis=0)

            self._object_pos_w = torch.tensor(object_pos_w, dtype=torch.float32, device=device)
            object_quat_w = torch.tensor(object_quat_w, dtype=torch.float32, device=device)
            self._object_quat_w = object_quat_w[:, [1, 2, 3, 0]]
            self._object_lin_vel_w = torch.tensor(object_lin_vel_w, dtype=torch.float32, device=device)
            self._object_size = torch.tensor(object_size, dtype=torch.float32, device=device)
        else:
            self._object_pos_w = torch.zeros(0, 3, device=device)
            self._object_quat_w = torch.zeros(0, 4, device=device)
            self._object_lin_vel_w = torch.zeros(0, 3, device=device)
            self._object_size = torch.zeros(0, 3, device=device)

        return body_names, joint_names

    @staticmethod
    def _decode_h5_strings(values: np.ndarray) -> list[str]:
        decoded: list[str] = []
        for item in values:
            if isinstance(item, (bytes, np.bytes_)):
                decoded.append(item.decode("utf-8"))
            else:
                decoded.append(str(item))
        return decoded

    @staticmethod
    def _finite_diff(data: np.ndarray, fps: float) -> np.ndarray:
        if data.shape[0] == 1:
            return np.zeros_like(data)
        vel = (data[1:] - data[:-1]) * fps
        return np.concatenate([vel, vel[-1:]], axis=0)

    @staticmethod
    def _quat_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
        out = q.copy()
        out[..., :3] *= -1.0
        return out

    @staticmethod
    def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ax, ay, az, aw = np.split(a, 4, axis=-1)
        bx, by, bz, bw = np.split(b, 4, axis=-1)
        x = aw * bx + ax * bw + ay * bz - az * by
        y = aw * by - ax * bz + ay * bw + az * bx
        z = aw * bz + ax * by - ay * bx + az * bw
        w = aw * bw - ax * bx - ay * by - az * bz
        return np.concatenate([x, y, z, w], axis=-1)

    @staticmethod
    def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        qvec = q[..., :3]
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        return v + 2.0 * (q[..., 3:4] * uv + uuv)

    @staticmethod
    def _angular_velocity_xyzw(quats: np.ndarray, fps: float) -> np.ndarray:
        if quats.shape[0] == 1:
            return np.zeros(quats.shape[:-1] + (3,), dtype=quats.dtype)
        q0 = quats[:-1]
        q1 = quats[1:]
        dq = MotionLoader._quat_mul_xyzw(q1, MotionLoader._quat_conjugate_xyzw(q0))
        dq = dq / np.linalg.norm(dq, axis=-1, keepdims=True)
        w = np.clip(dq[..., 3], -1.0, 1.0)
        v = dq[..., :3]
        sin_half = np.linalg.norm(v, axis=-1)
        angle = 2.0 * np.arctan2(sin_half, w)
        small = sin_half < 1e-8
        axis = np.zeros_like(v)
        axis[~small] = v[~small] / sin_half[~small][..., None]
        omega = axis * (angle[..., None] * fps)
        omega[small] = 2.0 * v[small] * fps
        return np.concatenate([omega, omega[-1:]], axis=0)

    @staticmethod
    def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
        return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)

    @staticmethod
    def _infer_link_frame(link_names: list[str], link_pos: np.ndarray, root_pos: np.ndarray) -> str:
        for pelvis_name in ("pelvis", "pelvis_link"):
            if pelvis_name in link_names:
                idx = link_names.index(pelvis_name)
                diff = np.linalg.norm(link_pos[:, idx] - root_pos, axis=-1)
                if np.median(diff) < 1e-3:
                    return "world"
                return "local"
        return "world"

    @staticmethod
    def _normalize_link_name(name: str) -> str:
        if name.endswith(".STL") or name.endswith(".stl"):
            return name[:-4]
        return name

    def _get_h5_attr_or_dataset(self, h5f: Any, name: str) -> np.ndarray | None:
        if name in h5f.attrs:
            return np.asarray(h5f.attrs[name])
        if f"/{name}" in h5f.attrs:
            return np.asarray(h5f.attrs[f"/{name}"])
        if name in h5f:
            return np.asarray(h5f[name])
        if f"/{name}" in h5f:
            return np.asarray(h5f[f"/{name}"])
        return None

    def _resolve_h5_clip_metadata_values(
        self,
        h5f: Any,
        *,
        clip_ids: list[str],
        selected_clip_indices: list[int],
        field_names: tuple[str, ...],
    ) -> list[str]:
        containers = []
        if "clips" in h5f:
            containers.append(h5f["clips"])
        if "meta" in h5f:
            containers.append(h5f["meta"])
        containers.append(h5f)

        raw_values = None
        for container in containers:
            for field_name in field_names:
                raw_values = self._get_h5_attr_or_dataset(container, field_name)
                if raw_values is not None:
                    break
            if raw_values is not None:
                break

        if raw_values is not None:
            arr = np.asarray(raw_values)
            if arr.shape == ():
                return [self._scalar_str(arr)] * len(selected_clip_indices)
            flat = arr.reshape(-1)
            if flat.shape[0] >= max(selected_clip_indices, default=0) + 1:
                return [self._scalar_str(flat[idx]) for idx in selected_clip_indices]

        clips_group = h5f["clips"] if "clips" in h5f else None
        if clips_group is not None:
            nested_values: list[str] = []
            for clip_idx in selected_clip_indices:
                clip_id = clip_ids[clip_idx]
                clip_group = clips_group.get(clip_id, None)
                if clip_group is None:
                    return []
                clip_value = None
                for field_name in field_names:
                    clip_value = self._get_h5_attr_or_dataset(clip_group, field_name)
                    if clip_value is not None:
                        break
                if clip_value is None:
                    return []
                nested_values.append(self._scalar_str(clip_value))
            return nested_values

        return []

    def _resolve_h5_clip_object_metadata(
        self,
        h5f: Any,
        *,
        motion_file: str,
        clip_ids: list[str],
        selected_clip_indices: list[int],
    ) -> tuple[list[str], list[str]]:
        raw_object_names = self._resolve_h5_clip_metadata_values(
            h5f,
            clip_ids=clip_ids,
            selected_clip_indices=selected_clip_indices,
            field_names=("object_name", "object_names"),
        )
        raw_object_urdfs = self._resolve_h5_clip_metadata_values(
            h5f,
            clip_ids=clip_ids,
            selected_clip_indices=selected_clip_indices,
            field_names=("object_urdf_path", "object_urdf_paths"),
        )

        base_dir = Path(motion_file).parent
        clip_object_names: list[str] = []
        clip_object_urdfs: list[str] = []
        for local_idx, clip_idx in enumerate(selected_clip_indices):
            clip_name = raw_object_names[local_idx].strip() if local_idx < len(raw_object_names) else ""
            clip_urdf = raw_object_urdfs[local_idx].strip() if local_idx < len(raw_object_urdfs) else ""
            if clip_urdf:
                clip_urdf = self._resolve_motion_object_urdf_path(clip_urdf, base_dir=base_dir)
            if not clip_name and clip_urdf:
                clip_name = Path(clip_urdf).stem
            if not clip_name:
                clip_name = "object"
            clip_object_names.append(clip_name)
            clip_object_urdfs.append(clip_urdf)
        return clip_object_names, clip_object_urdfs

    def _load_data_from_motion_h5(
        self,
        motion_file: str,
        device: str,
        motion_clip_id: int | None,
        motion_clip_name: str | None,
    ) -> tuple[list[str], list[str]]:
        try:
            import h5py  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("h5py is required to load HDF5 motion files.") from exc

        with h5py.File(motion_file, "r") as h5f:
            if "meta" not in h5f or "data" not in h5f:
                return self._load_data_from_motion_h5_videomimic(h5f, motion_file, device)

            meta = h5f["meta"]
            data = h5f["data"]

            joint_names = self._decode_h5_strings(np.asarray(meta["joint_names"]))
            body_names = self._decode_h5_strings(np.asarray(meta["body_names"]))

            clips = h5f["clips"] if "clips" in h5f else None
            clip_ids: list[str] = []
            offsets = None
            lengths = None
            clip_fps = None
            selected_clip_idx: int | None = None
            selected_clip_indices: list[int] = [0]
            if clips is not None:
                clip_ids = self._decode_h5_strings(np.asarray(clips["clip_ids"]))
                offsets = np.asarray(clips["offsets"], dtype=np.int64)
                lengths = np.asarray(clips["lengths"], dtype=np.int64)
                if "clip_fps" in clips:
                    clip_fps = np.asarray(clips["clip_fps"], dtype=np.float32)

            load_all = motion_clip_id is None and motion_clip_name is None
            if clips is None:
                if not load_all:
                    raise ValueError("motion_clip_id/name provided but HDF5 motion file has no /clips group.")
                start = 0
                length = int(data["joint_pos"].shape[0])
                fps_val = np.asarray(meta["fps"])
                clip_id = Path(motion_file).stem
                clip_ids = [clip_id]
                selected_clip_indices = [0]
            elif load_all:
                start = 0
                length = int(data["joint_pos"].shape[0])
                fps_val = np.asarray(meta["fps"])
                if clip_fps is not None:
                    if not np.allclose(clip_fps, float(np.array(fps_val).reshape(-1)[0])):
                            raise ValueError("clip_fps must be consistent across clips for multi-clip loading.")
                assert offsets is not None and lengths is not None
                selected_clip_indices = list(range(len(clip_ids)))
            else:
                if motion_clip_name is not None:
                    if motion_clip_name not in clip_ids:
                        raise ValueError(f"Clip name '{motion_clip_name}' not found in HDF5 motion file.")
                    clip_idx = clip_ids.index(motion_clip_name)
                else:
                    clip_idx = int(motion_clip_id)

                assert offsets is not None and lengths is not None
                if clip_idx < 0 or clip_idx >= len(lengths):
                    raise IndexError(f"Clip index {clip_idx} out of range for HDF5 motion file.")
                selected_clip_idx = clip_idx
                selected_clip_indices = [clip_idx]
                start = int(offsets[clip_idx])
                length = int(lengths[clip_idx])
                fps_val = clip_fps[clip_idx] if clip_fps is not None else np.asarray(meta["fps"])

            clip_object_names, clip_object_urdfs = self._resolve_h5_clip_object_metadata(
                h5f,
                motion_file=motion_file,
                clip_ids=clip_ids,
                selected_clip_indices=selected_clip_indices,
            )
            if clips is None:
                self._set_clip_metadata(
                    clip_ids,
                    np.array([0]),
                    np.array([length]),
                    device,
                    clip_object_names=clip_object_names,
                    clip_object_urdf_paths=clip_object_urdfs,
                )
            elif load_all:
                assert offsets is not None and lengths is not None
                self._set_clip_metadata(
                    clip_ids,
                    offsets,
                    lengths,
                    device,
                    clip_object_names=clip_object_names,
                    clip_object_urdf_paths=clip_object_urdfs,
                )
            else:
                self._set_clip_metadata(
                    [clip_ids[selected_clip_idx]],
                    np.array([0]),
                    np.array([length]),
                    device,
                    clip_object_names=clip_object_names,
                    clip_object_urdf_paths=clip_object_urdfs,
                )

            fps_arr = np.array(fps_val).reshape(-1)
            self.fps = float(fps_arr[0]) if fps_arr.size > 0 else 30.0

            end = start + length
            joint_pos = np.asarray(data["joint_pos"][start:end])
            joint_vel = np.asarray(data["joint_vel"][start:end])
            body_pos_w = np.asarray(data["body_pos_w"][start:end])
            body_quat_w = np.asarray(data["body_quat_w"][start:end])
            body_lin_vel_w = np.asarray(data["body_lin_vel_w"][start:end])
            body_ang_vel_w = np.asarray(data["body_ang_vel_w"][start:end])

            self._joint_pos = torch.tensor(joint_pos[:, 7:], dtype=torch.float32, device=device)
            self._joint_vel = torch.tensor(joint_vel[:, 6:], dtype=torch.float32, device=device)
            assert len(joint_names) == self._joint_pos.shape[1], "Joint names in motion data does not match"

            self._body_pos_w = torch.tensor(body_pos_w, dtype=torch.float32, device=device)
            assert len(body_names) == self._body_pos_w.shape[1], "Body names in motion data does not match"

            body_quat_w_wxyz = torch.tensor(body_quat_w, dtype=torch.float32, device=device)
            self._body_quat_w = body_quat_w_wxyz[:, :, [1, 2, 3, 0]]

            self._body_lin_vel_w = torch.tensor(body_lin_vel_w, dtype=torch.float32, device=device)
            self._body_ang_vel_w = torch.tensor(body_ang_vel_w, dtype=torch.float32, device=device)

            self.has_object = "object_pos_w" in data
            if self.has_object:
                object_pos_w = np.asarray(data["object_pos_w"][start:end])
                object_quat_w = np.asarray(data["object_quat_w"][start:end])
                object_lin_vel_w = np.asarray(data["object_lin_vel_w"][start:end])
                object_size = None
                for key in self._OBJECT_SIZE_KEYS:
                    if key not in data:
                        continue
                    raw_size = np.asarray(data[key])
                    if raw_size.dtype.kind != "f":
                        raise ValueError(
                            f"Motion field {key} in {motion_file} must use a real floating dtype, "
                            f"got {raw_size.dtype}."
                        )
                    # Support clip-wise object size annotations: shape (num_clips, 3) or (num_clips,).
                    if (
                        clips is not None
                        and lengths is not None
                        and raw_size.ndim in (1, 2)
                        and raw_size.shape[0] == len(lengths)
                    ):
                        if selected_clip_idx is not None:
                            raw_size = raw_size[selected_clip_idx]
                            object_size = self._normalize_object_size_array(
                                raw_size, length, source=f"{motion_file}:{key}"
                            )
                            break
                        if load_all:
                            per_clip_sizes = []
                            for clip_i, clip_len in enumerate(lengths):
                                clip_size = self._normalize_object_size_array(
                                    raw_size[clip_i], int(clip_len), source=f"{motion_file}:{key}"
                                )
                                per_clip_sizes.append(clip_size)
                            object_size = np.concatenate(per_clip_sizes, axis=0)
                            break
                    # Most common format stores size per frame for the full bank.
                    if raw_size.ndim >= 1 and raw_size.shape[0] >= end:
                        raw_size = raw_size[start:end]
                    object_size = self._normalize_object_size_array(
                        raw_size, length, source=f"{motion_file}:{key}"
                    )
                    break
                if object_size is None:
                    scale_keys = [key for key in self._OBJECT_SCALE_KEYS if key in data]
                    if scale_keys:
                        raise ValueError(
                            f"Motion file {motion_file} provides mesh scale field(s) {scale_keys} but no physical "
                            "object_size/box_size extents; scale and size are not interchangeable."
                        )
                if object_size is None:
                    object_size = np.ones((length, 3), dtype=np.float32)
                if not np.all(np.isfinite(object_size)) or np.any(object_size <= 0.0):
                    raise ValueError(
                        f"Motion object_size in {motion_file} must contain finite positive physical extents."
                    )

                self._object_pos_w = torch.tensor(object_pos_w, dtype=torch.float32, device=device)
                object_quat_w = torch.tensor(object_quat_w, dtype=torch.float32, device=device)
                self._object_quat_w = object_quat_w[:, [1, 2, 3, 0]]
                self._object_lin_vel_w = torch.tensor(object_lin_vel_w, dtype=torch.float32, device=device)
                self._object_size = torch.tensor(object_size, dtype=torch.float32, device=device)
            else:
                self._object_pos_w = torch.zeros(0, 3, device=device)
                self._object_quat_w = torch.zeros(0, 4, device=device)
                self._object_lin_vel_w = torch.zeros(0, 3, device=device)
                self._object_size = torch.zeros(0, 3, device=device)

        return body_names, joint_names

    def _load_data_from_motion_h5_videomimic(
        self,
        h5f: Any,
        motion_file: str,
        device: str,
    ) -> tuple[list[str], list[str]]:
        required = ("root_pos", "root_quat", "joints", "link_pos", "link_quat")
        missing = [key for key in required if key not in h5f]
        if missing:
            raise KeyError(f"Missing keys in VideoMimic HDF5 file: {missing}")

        root_pos = np.asarray(h5f["root_pos"], dtype=np.float32)
        root_quat_xyzw = np.asarray(h5f["root_quat"], dtype=np.float32)
        joints = np.asarray(h5f["joints"], dtype=np.float32)
        link_pos = np.asarray(h5f["link_pos"], dtype=np.float32)
        link_quat_xyzw = np.asarray(h5f["link_quat"], dtype=np.float32)

        joint_names_raw = self._get_h5_attr_or_dataset(h5f, "joint_names")
        link_names_raw = self._get_h5_attr_or_dataset(h5f, "link_names")
        if joint_names_raw is None or link_names_raw is None:
            raise ValueError("VideoMimic HDF5 file must provide joint_names and link_names.")
        joint_names = self._decode_h5_strings(np.asarray(joint_names_raw))
        link_names = self._decode_h5_strings(np.asarray(link_names_raw))
        link_names = [self._normalize_link_name(name) for name in link_names]

        fps_raw = self._get_h5_attr_or_dataset(h5f, "fps")
        fps_arr = np.array(fps_raw).reshape(-1) if fps_raw is not None else np.array([30.0], dtype=np.float32)
        self.fps = float(fps_arr[0]) if fps_arr.size > 0 else 30.0

        num_frames = int(root_pos.shape[0])
        if joints.shape[0] != num_frames:
            raise ValueError("VideoMimic HDF5 joint length does not match root_pos length.")

        if self._robot_joint_names:
            missing_joints = [name for name in self._robot_joint_names if name not in joint_names]
            if missing_joints:
                zeros = np.zeros((num_frames, len(missing_joints)), dtype=joints.dtype)
                joints = np.concatenate([joints, zeros], axis=1)
                joint_names.extend(missing_joints)
                logger.warning("Missing joints in VideoMimic HDF5, padded with zeros: {}", missing_joints)

        # VideoMimic uses link_pos/link_quat in the env/world frame. Keep them as-is.
        link_pos_w = link_pos
        link_quat_w = link_quat_xyzw

        body_names = list(self._robot_body_names)
        num_bodies = len(body_names)
        body_pos_w = np.broadcast_to(root_pos[:, None, :], (num_frames, num_bodies, 3)).copy()
        body_quat_w = np.broadcast_to(root_quat_xyzw[:, None, :], (num_frames, num_bodies, 4)).copy()

        link_name_map = {name: i for i, name in enumerate(link_names)}
        for body_idx, body_name in enumerate(body_names):
            link_idx = link_name_map.get(body_name)
            if link_idx is None:
                continue
            body_pos_w[:, body_idx] = link_pos_w[:, link_idx]
            body_quat_w[:, body_idx] = link_quat_w[:, link_idx]

        body_lin_vel_w = self._finite_diff(body_pos_w, self.fps)
        body_ang_vel_w = self._angular_velocity_xyzw(body_quat_w, self.fps)

        root_lin_vel = self._finite_diff(root_pos, self.fps)
        root_ang_vel = self._angular_velocity_xyzw(root_quat_xyzw, self.fps)
        dof_vel = self._finite_diff(joints, self.fps)

        joint_pos = np.concatenate([root_pos, self._xyzw_to_wxyz(root_quat_xyzw), joints], axis=-1)
        joint_vel = np.concatenate([root_lin_vel, root_ang_vel, dof_vel], axis=-1)

        self._joint_pos = torch.tensor(joint_pos[:, 7:], dtype=torch.float32, device=device)
        self._joint_vel = torch.tensor(joint_vel[:, 6:], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(body_pos_w, dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(body_quat_w, dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(body_lin_vel_w, dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(body_ang_vel_w, dtype=torch.float32, device=device)

        self.has_object = False
        self._object_pos_w = torch.zeros(0, 3, device=device)
        self._object_quat_w = torch.zeros(0, 4, device=device)
        self._object_lin_vel_w = torch.zeros(0, 3, device=device)
        self._object_size = torch.zeros(0, 3, device=device)

        clip_id = Path(motion_file).stem
        self._set_clip_metadata([clip_id], np.array([0]), np.array([num_frames]), device)
        return body_names, joint_names

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._object_pos_w[:]

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._object_quat_w[:]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._object_lin_vel_w[:]

    @property
    def object_size(self) -> torch.Tensor:
        return self._object_size[:]

    @property
    def has_precomputed_root_command(self) -> bool:
        return self._precomputed_root_command is not None

    @property
    def precomputed_root_command(self) -> torch.Tensor:
        if self._precomputed_root_command is None:
            raise RuntimeError("The loaded motion bank does not contain a precomputed root command.")
        return self._precomputed_root_command

    @property
    def precomputed_root_command_phase(self) -> torch.Tensor:
        if self._precomputed_root_command_phase is None:
            raise RuntimeError("The loaded motion bank does not contain a precomputed root command phase.")
        return self._precomputed_root_command_phase

    def extend_with_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> MotionLoader:
        """Merge interpolated segments with motion data, mutating this MotionLoader."""
        concat_targets = [
            ("joint_pos", "_joint_pos"),
            ("joint_vel", "_joint_vel"),
            ("body_pos", "_body_pos_w"),
            ("body_quat", "_body_quat_w"),
            ("body_lin_vel", "_body_lin_vel_w"),
            ("body_ang_vel", "_body_ang_vel_w"),
        ]
        if self.has_object:
            concat_targets.extend(
                [
                    ("object_pos", "_object_pos_w"),
                    ("object_quat", "_object_quat_w"),
                    ("object_lin_vel", "_object_lin_vel_w"),
                    ("object_size", "_object_size"),
                ]
            )

        for seg_key, attr_name in concat_targets:
            existing = getattr(self, attr_name)
            tensors = (segments[seg_key], existing) if prepend else (existing, segments[seg_key])
            setattr(self, attr_name, torch.cat(tensors, dim=0))

        if self._precomputed_root_command is not None:
            segment_length = int(segments["joint_pos"].shape[0])
            command_padding = torch.zeros(
                (segment_length, 3),
                dtype=self._precomputed_root_command.dtype,
                device=self._precomputed_root_command.device,
            )
            phase_padding = torch.zeros(
                (segment_length,),
                dtype=self.precomputed_root_command_phase.dtype,
                device=self.precomputed_root_command_phase.device,
            )
            command_tensors = (
                (command_padding, self._precomputed_root_command)
                if prepend
                else (self._precomputed_root_command, command_padding)
            )
            phase_tensors = (
                (phase_padding, self._precomputed_root_command_phase)
                if prepend
                else (self._precomputed_root_command_phase, phase_padding)
            )
            self._precomputed_root_command = torch.cat(command_tensors, dim=0)
            self._precomputed_root_command_phase = torch.cat(phase_tensors, dim=0)

        self.time_step_total = self._joint_pos.shape[0]
        if self.num_clips == 1:
            device = self.clip_lengths.device if self.clip_lengths.numel() > 0 else self._joint_pos.device
            self.clip_lengths = torch.tensor([self.time_step_total], dtype=torch.long, device=device)
        return self


class AdaptiveTimestepsSampler:
    """Prioritizes training on motion segments where the robot fails most often."""

    def __init__(
        self,
        motion_time_step_total: int | None,
        device: str,
        env_fps: int,
        clip_lengths: torch.Tensor | None = None,
        valid_start_counts: torch.Tensor | None = None,
        adaptive_kernel_size: int = 1,
        adaptive_lambda: float = 0.8,
        adaptive_uniform_ratio: float = 0.1,
        adaptive_alpha: float = 0.001,
    ):
        self.device = device
        if isinstance(env_fps, bool) or not isinstance(env_fps, numbers.Integral) or int(env_fps) <= 0:
            raise ValueError(f"env_fps must be a positive integer, got {env_fps!r}.")
        self.env_fps = int(env_fps)

        if clip_lengths is not None:
            raw_clip_lengths = torch.as_tensor(clip_lengths, device=self.device)
            if (
                raw_clip_lengths.dtype == torch.bool
                or raw_clip_lengths.is_floating_point()
                or raw_clip_lengths.is_complex()
            ):
                raise ValueError(f"clip_lengths must use an integer dtype, got {raw_clip_lengths.dtype}.")
            clip_lengths = raw_clip_lengths.to(dtype=torch.long).reshape(-1)
            if clip_lengths.numel() == 0:
                raise ValueError("clip_lengths must contain at least one clip.")
            if bool((clip_lengths < 1).any().item()):
                raise ValueError("clip_lengths must be positive for every clip.")
            self.clip_lengths = clip_lengths
        else:
            if motion_time_step_total is None:
                raise ValueError("motion_time_step_total must be provided when clip_lengths is None.")
            if (
                isinstance(motion_time_step_total, bool)
                or not isinstance(motion_time_step_total, numbers.Integral)
                or int(motion_time_step_total) <= 0
            ):
                raise ValueError(
                    "motion_time_step_total must be a positive integer when clip_lengths is absent, "
                    f"got {motion_time_step_total!r}."
                )
            total_steps = int(motion_time_step_total)
            self.clip_lengths = torch.tensor([total_steps], dtype=torch.long, device=self.device)

        self.num_clips = int(self.clip_lengths.numel())
        self.motion_time_step_total = int(self.clip_lengths.max().item())
        if valid_start_counts is None:
            self.valid_start_counts = self.clip_lengths.clone()
        else:
            raw_valid_start_counts = torch.as_tensor(valid_start_counts, device=self.device)
            if (
                raw_valid_start_counts.dtype == torch.bool
                or raw_valid_start_counts.is_floating_point()
                or raw_valid_start_counts.is_complex()
            ):
                raise ValueError(
                    f"valid_start_counts must use an integer dtype, got {raw_valid_start_counts.dtype}."
                )
            valid_start_counts = raw_valid_start_counts.to(dtype=torch.long).reshape(-1)
            if valid_start_counts.numel() != self.num_clips:
                raise ValueError(
                    "valid_start_counts must have one entry per clip: "
                    f"expected {self.num_clips}, got {valid_start_counts.numel()}."
                )
            if torch.any(valid_start_counts < 1) or torch.any(valid_start_counts > self.clip_lengths):
                raise ValueError("valid_start_counts must stay within [1, clip_length] for every clip.")
            self.valid_start_counts = valid_start_counts
        if (
            isinstance(adaptive_kernel_size, bool)
            or not isinstance(adaptive_kernel_size, numbers.Integral)
            or int(adaptive_kernel_size) <= 0
        ):
            raise ValueError(
                f"adaptive_kernel_size must be a positive integer, got {adaptive_kernel_size!r}."
            )
        self.adaptive_kernel_size = int(adaptive_kernel_size)
        if isinstance(adaptive_lambda, bool) or not isinstance(adaptive_lambda, numbers.Real):
            raise ValueError(f"adaptive_lambda must be a real number, got {adaptive_lambda!r}.")
        self.adaptive_lambda = float(adaptive_lambda)
        if not np.isfinite(self.adaptive_lambda) or not 0.0 <= self.adaptive_lambda <= 1.0:
            raise ValueError(f"adaptive_lambda must be finite and within [0, 1], got {adaptive_lambda!r}.")
        if isinstance(adaptive_uniform_ratio, bool) or not isinstance(adaptive_uniform_ratio, numbers.Real):
            raise ValueError(
                f"adaptive_uniform_ratio must be a real number, got {adaptive_uniform_ratio!r}."
            )
        self.adaptive_uniform_ratio = float(adaptive_uniform_ratio)
        if not 0.0 <= self.adaptive_uniform_ratio <= 1.0:
            raise ValueError(
                "adaptive_uniform_ratio must be within [0, 1], "
                f"got {self.adaptive_uniform_ratio}."
            )
        if isinstance(adaptive_alpha, bool) or not isinstance(adaptive_alpha, numbers.Real):
            raise ValueError(f"adaptive_alpha must be a real number, got {adaptive_alpha!r}.")
        self.adaptive_alpha = float(adaptive_alpha)
        if not np.isfinite(self.adaptive_alpha) or not 0.0 <= self.adaptive_alpha <= 1.0:
            raise ValueError(f"adaptive_alpha must be finite and within [0, 1], got {adaptive_alpha!r}.")

        # A bin covers at most one second of *valid reset starts*.  Ceiling division is
        # important here: an exact multiple of env_fps must not create an empty tail bin.
        fps = max(self.env_fps, 1)
        self.num_bins_per_clip = torch.clamp((self.valid_start_counts + fps - 1) // fps, min=1)
        self.max_num_bins = int(self.num_bins_per_clip.max().item())
        self.num_bins = self.max_num_bins

        # BeyondMimic-style non-causal decay kernel.
        self.kernel = torch.tensor(
            [self.adaptive_lambda**i for i in range(self.adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()

        # Sampling geometry is immutable.  Keep its padded representation on the
        # sampler device so reset-time sampling can operate on the whole reset
        # batch without reading one clip id / window boundary back to Python at a
        # time.  Invalid padded entries are always masked before they contribute.
        self.max_valid_start_count = int(self.valid_start_counts.max().item())
        bin_axis = torch.arange(self.max_num_bins, dtype=torch.long, device=self.device)
        step_axis = torch.arange(self.max_valid_start_count, dtype=torch.long, device=self.device)
        self._step_axis = step_axis
        self._valid_bin_mask = bin_axis.unsqueeze(0) < self.num_bins_per_clip.unsqueeze(1)
        self._valid_step_mask = step_axis.unsqueeze(0) < self.valid_start_counts.unsqueeze(1)

        step_bin_indices = torch.div(
            step_axis.unsqueeze(0) * self.num_bins_per_clip.unsqueeze(1),
            self.valid_start_counts.unsqueeze(1),
            rounding_mode="floor",
        )
        self._step_bin_indices = torch.minimum(
            step_bin_indices,
            self.num_bins_per_clip.unsqueeze(1) - 1,
        )
        self._step_counts_per_bin = torch.zeros(
            (self.num_clips, self.max_num_bins),
            dtype=torch.float32,
            device=self.device,
        )
        self._step_counts_per_bin.scatter_add_(
            1,
            self._step_bin_indices,
            self._valid_step_mask.to(dtype=torch.float32),
        )

        kernel_offsets = torch.arange(
            self.adaptive_kernel_size,
            dtype=torch.long,
            device=self.device,
        )
        kernel_bin_indices = bin_axis.view(1, -1, 1) + kernel_offsets.view(1, 1, -1)
        self._kernel_bin_indices = torch.minimum(
            kernel_bin_indices,
            (self.num_bins_per_clip - 1).view(-1, 1, 1),
        )

        # key data: failure counts
        self.init_buffers()
        # metrics
        self.metrics: dict[str, torch.Tensor] = {}

    def init_buffers(self):
        shape = (self.num_clips, self.max_num_bins)
        self.current_bin_failed_count = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.bin_failed_count = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.current_bin_exposure_count = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.bin_exposure_count = torch.zeros(shape, dtype=torch.float32, device=self.device)

    def _resolve_clip_ids(self, clip_ids: torch.Tensor | None, count: int) -> torch.Tensor:
        if clip_ids is None:
            if self.num_clips != 1:
                raise ValueError("clip_ids must be provided for multi-clip adaptive timestep sampling.")
            return torch.zeros((count,), dtype=torch.long, device=self.device)
        raw_clip_ids = torch.as_tensor(clip_ids, device=self.device)
        if (
            raw_clip_ids.dtype == torch.bool
            or raw_clip_ids.is_floating_point()
            or raw_clip_ids.is_complex()
        ):
            raise ValueError(f"clip_ids must use an integer dtype, got {raw_clip_ids.dtype}.")
        clip_ids = raw_clip_ids.to(dtype=torch.long).reshape(-1)
        if clip_ids.numel() != count:
            raise ValueError(f"Expected {count} clip ids, got {clip_ids.numel()}.")
        if bool(((clip_ids < 0) | (clip_ids >= self.num_clips)).any().item()):
            raise ValueError(f"clip_ids must be in [0, {self.num_clips}).")
        return clip_ids

    def _coerce_time_steps(
        self,
        time_steps: torch.Tensor,
        *,
        trusted: bool,
    ) -> torch.Tensor:
        raw_time_steps = torch.as_tensor(time_steps, device=self.device)
        if raw_time_steps.dtype == torch.bool or raw_time_steps.is_complex():
            raise ValueError(f"time_steps must use a real numeric dtype, got {raw_time_steps.dtype}.")
        converted = raw_time_steps.to(dtype=torch.float32).reshape(-1)
        if not trusted and not bool(torch.isfinite(converted).all().item()):
            raise ValueError("time_steps must be finite.")
        return converted

    def _sampling_probabilities_for_clip(self, clip_id: int) -> torch.Tensor:
        valid_bins = int(self.num_bins_per_clip[clip_id].item())
        failed = self.bin_failed_count[clip_id, :valid_bins]
        exposure = self.bin_exposure_count[clip_id, :valid_bins]
        if not bool(torch.isfinite(failed).all().item()) or not bool(torch.isfinite(exposure).all().item()):
            raise RuntimeError("Adaptive timestep sampler contains non-finite failure/exposure state.")
        if bool((failed < 0.0).any().item()) or bool((exposure < 0.0).any().item()):
            raise RuntimeError("Adaptive timestep sampler failure/exposure state must be non-negative.")
        tolerance = 1.0e-6 * torch.maximum(torch.ones_like(exposure), exposure)
        if bool((failed > exposure + tolerance).any().item()):
            raise RuntimeError(
                "Adaptive timestep sampler failure state exceeds exposure state; refusing to change "
                "the sampling distribution silently."
            )
        failure_rate = torch.where(
            exposure > 1.0e-12,
            failed / exposure.clamp_min(1.0e-12),
            torch.zeros_like(failed),
        )
        adaptive_scores = F.pad(
            failure_rate.unsqueeze(0).unsqueeze(0),
            (0, self.adaptive_kernel_size - 1),
            mode="replicate",
        )
        adaptive_scores = F.conv1d(adaptive_scores, self.kernel.view(1, 1, -1)).view(-1)
        score_sum = adaptive_scores.sum()
        uniform = torch.full_like(adaptive_scores, 1.0 / float(valid_bins))
        adaptive = torch.where(
            score_sum > 1.0e-12,
            adaptive_scores / score_sum.clamp_min(1.0e-12),
            uniform,
        )
        ratio = self.adaptive_uniform_ratio
        return (1.0 - ratio) * adaptive + ratio * uniform

    def _bin_indices_for_clip(self, clip_id: int) -> torch.Tensor:
        """Map every valid discrete reset timestep to its adaptive bin."""
        valid_start_count = int(self.valid_start_counts[clip_id].item())
        num_bins = int(self.num_bins_per_clip[clip_id].item())
        steps = torch.arange(valid_start_count, dtype=torch.long, device=self.device)
        bin_ids = torch.div(steps * num_bins, valid_start_count, rounding_mode="floor")
        return torch.clamp(bin_ids, max=num_bins - 1)

    @staticmethod
    def _validated_window_reweight_args(
        window_density_boost: float,
        window_target_probability: float | None,
    ) -> tuple[float, float | None]:
        if isinstance(window_density_boost, bool) or not isinstance(
            window_density_boost, numbers.Real
        ):
            raise ValueError(
                "window_density_boost must be a finite real number >= 1, "
                f"got {window_density_boost!r}."
            )
        parsed_density_boost = float(window_density_boost)
        if not np.isfinite(parsed_density_boost) or parsed_density_boost < 1.0:
            raise ValueError(
                "window_density_boost must be a finite real number >= 1, "
                f"got {window_density_boost!r}."
            )

        parsed_target_probability: float | None = None
        if window_target_probability is not None:
            if isinstance(window_target_probability, bool) or not isinstance(
                window_target_probability, numbers.Real
            ):
                raise ValueError(
                    "window_target_probability must be a finite probability in [0, 1], "
                    f"got {window_target_probability!r}."
                )
            parsed_target_probability = float(window_target_probability)
            if (
                not np.isfinite(parsed_target_probability)
                or not 0.0 <= parsed_target_probability <= 1.0
            ):
                raise ValueError(
                    "window_target_probability must be a finite probability in [0, 1], "
                    f"got {window_target_probability!r}."
                )
        return parsed_density_boost, parsed_target_probability

    def timestep_probabilities_for_clip(
        self,
        clip_id: int,
        *,
        exclude_zero: bool = False,
        window: tuple[int, int] | None = None,
        window_density_boost: float = 1.0,
        window_target_probability: float | None = None,
    ) -> torch.Tensor:
        """Return the normalized probability of every valid discrete reset timestep.

        Adaptive failure mass is first spread uniformly over the discrete timesteps in
        each bin.  An optional contact window then reweights that density, so contact
        bias and failure prioritization compose instead of one silently replacing the
        other.
        """
        if clip_id < 0 or clip_id >= self.num_clips:
            raise IndexError(f"clip_id must be in [0, {self.num_clips}), got {clip_id}.")

        parsed_density_boost, parsed_target_probability = self._validated_window_reweight_args(
            window_density_boost,
            window_target_probability,
        )

        bin_ids = self._bin_indices_for_clip(clip_id)
        bin_probabilities = self._sampling_probabilities_for_clip(clip_id)
        bin_counts = torch.bincount(bin_ids, minlength=bin_probabilities.numel()).to(dtype=torch.float32)
        probabilities = bin_probabilities[bin_ids] / bin_counts[bin_ids].clamp_min(1.0)

        if exclude_zero and probabilities.numel() > 1:
            probabilities[0] = 0.0

        if window is not None and probabilities.numel() > 0:
            lo = max(0, int(window[0]))
            hi = min(int(window[1]), probabilities.numel() - 1)
            if hi >= lo:
                window_mask = torch.zeros_like(probabilities, dtype=torch.bool)
                window_mask[lo : hi + 1] = True
                base_window_mass = probabilities[window_mask].sum()
                base_outside_mass = probabilities[~window_mask].sum()
                if parsed_target_probability is not None:
                    window_mass = float(base_window_mass.item())
                    outside_mass = float(base_outside_mass.item())
                    if window_mass > 0.0 and outside_mass > 0.0:
                        probabilities[window_mask] *= parsed_target_probability / window_mass
                        probabilities[~window_mask] *= (1.0 - parsed_target_probability) / outside_mass
                    elif window_mass > 0.0:
                        # The requested target is infeasible when every supported
                        # timestep lies in the window. Preserve the only available
                        # support and report an effective window mass of one.
                        probabilities[window_mask] /= window_mass
                    elif outside_mass > 0.0:
                        # Symmetrically, a target above zero cannot create support
                        # inside an empty window. Preserve the outside distribution.
                        probabilities[~window_mask] /= outside_mass
                else:
                    probabilities[window_mask] *= parsed_density_boost

        total = probabilities.sum()
        if not torch.isfinite(total) or float(total.item()) <= 0.0:
            raise RuntimeError(
                "Adaptive timestep probability construction produced a non-finite or zero-mass distribution."
            )
        return probabilities / total

    def timestep_probabilities_for_samples(
        self,
        clip_ids: torch.Tensor,
        *,
        exclude_zero: bool = False,
        windows: torch.Tensor | None = None,
        window_valid: torch.Tensor | None = None,
        window_density_boost: float = 1.0,
        window_target_probability: float | None = None,
        _trusted_inputs: bool = False,
    ) -> torch.Tensor:
        """Build padded timestep distributions for a complete sample batch on-device.

        This is mathematically equivalent to calling
        :meth:`timestep_probabilities_for_clip` for every sample, but it avoids
        per-clip Python iteration and CUDA scalar reads.  Columns at or beyond a
        sample's valid-start count have exactly zero probability.
        """
        if _trusted_inputs:
            clip_ids = torch.as_tensor(clip_ids, dtype=torch.long, device=self.device).reshape(-1)
        else:
            clip_ids = self._resolve_clip_ids(clip_ids, int(clip_ids.numel()))
        num_samples = clip_ids.numel()
        if num_samples == 0:
            return torch.zeros(
                (0, self.max_valid_start_count),
                dtype=torch.float32,
                device=self.device,
            )

        parsed_density_boost, parsed_target_probability = self._validated_window_reweight_args(
            window_density_boost,
            window_target_probability,
        )

        if windows is not None:
            windows = torch.as_tensor(windows, dtype=torch.long, device=self.device).reshape(-1, 2)
            if windows.shape[0] != num_samples:
                raise ValueError("windows must have shape [num_samples, 2].")
        if window_valid is not None:
            window_valid = torch.as_tensor(window_valid, dtype=torch.bool, device=self.device).reshape(-1)
            if window_valid.numel() != num_samples:
                raise ValueError("window_valid must have one entry per sample.")

        # Preserve the existing one-window-per-clip API contract, but validate it
        # with one batched device predicate instead of reading every unique clip
        # and its first window into Python.  Mixed validity for one clip is also
        # rejected because accepting it would make the distribution order-dependent.
        window_contract_valid = torch.ones((), dtype=torch.bool, device=self.device)
        if not _trusted_inputs and windows is not None and num_samples > 1:
            active_windows = (
                torch.ones((num_samples,), dtype=torch.bool, device=self.device)
                if window_valid is None
                else window_valid
            )
            order = torch.argsort(clip_ids)
            sorted_clip_ids = clip_ids[order]
            sorted_windows = windows[order]
            sorted_active = active_windows[order]
            same_clip = sorted_clip_ids[1:] == sorted_clip_ids[:-1]
            validity_mismatch = same_clip & (sorted_active[1:] != sorted_active[:-1])
            window_mismatch = (
                same_clip
                & sorted_active[1:]
                & sorted_active[:-1]
                & torch.any(sorted_windows[1:] != sorted_windows[:-1], dim=1)
            )
            window_contract_valid = ~(validity_mismatch | window_mismatch).any()

        valid_bin_mask = self._valid_bin_mask[clip_ids]
        failed = self.bin_failed_count[clip_ids]
        exposure = self.bin_exposure_count[clip_ids]
        if not _trusted_inputs:
            if not bool(window_contract_valid.item()):
                raise ValueError("All samples for one clip must use the same contact window and validity.")
            if not bool(((torch.isfinite(failed) & torch.isfinite(exposure)) | ~valid_bin_mask).all().item()):
                raise RuntimeError("Adaptive timestep sampler contains non-finite failure/exposure state.")
            if bool((((failed < 0.0) | (exposure < 0.0)) & valid_bin_mask).any().item()):
                raise RuntimeError("Adaptive timestep sampler failure/exposure state must be non-negative.")
            tolerance = 1.0e-6 * torch.maximum(torch.ones_like(exposure), exposure)
            if bool(((failed > exposure + tolerance) & valid_bin_mask).any().item()):
                raise RuntimeError(
                    "Adaptive timestep sampler failure state exceeds exposure state; refusing to change "
                    "the sampling distribution silently."
                )

        failure_rate = torch.where(
            valid_bin_mask & (exposure > 1.0e-12),
            failed / exposure.clamp_min(1.0e-12),
            torch.zeros_like(failed),
        )
        kernel_indices = self._kernel_bin_indices[clip_ids]
        kernel_values = torch.gather(
            failure_rate.unsqueeze(1).expand(-1, self.max_num_bins, -1),
            2,
            kernel_indices,
        )
        adaptive_scores = (kernel_values * self.kernel.view(1, 1, -1)).sum(dim=2)
        adaptive_scores = torch.where(valid_bin_mask, adaptive_scores, torch.zeros_like(adaptive_scores))
        score_sum = adaptive_scores.sum(dim=1, keepdim=True)
        uniform = valid_bin_mask.to(dtype=torch.float32) / self.num_bins_per_clip[clip_ids].to(
            dtype=torch.float32
        ).unsqueeze(1)
        adaptive = torch.where(
            score_sum > 1.0e-12,
            adaptive_scores / score_sum.clamp_min(1.0e-12),
            uniform,
        )
        bin_probabilities = (
            (1.0 - self.adaptive_uniform_ratio) * adaptive
            + self.adaptive_uniform_ratio * uniform
        )

        step_bin_indices = self._step_bin_indices[clip_ids]
        step_bin_counts = torch.gather(
            self._step_counts_per_bin[clip_ids],
            1,
            step_bin_indices,
        )
        probabilities = torch.gather(bin_probabilities, 1, step_bin_indices)
        probabilities = probabilities / step_bin_counts.clamp_min(1.0)
        valid_step_mask = self._valid_step_mask[clip_ids]
        probabilities = torch.where(valid_step_mask, probabilities, torch.zeros_like(probabilities))

        if exclude_zero:
            exclude_zero_mask = (
                self.valid_start_counts[clip_ids] > 1
            ).unsqueeze(1) & (self._step_axis.unsqueeze(0) == 0)
            probabilities = torch.where(exclude_zero_mask, torch.zeros_like(probabilities), probabilities)

        if windows is not None:
            active_windows = (
                torch.ones((num_samples,), dtype=torch.bool, device=self.device)
                if window_valid is None
                else window_valid
            )
            max_step = self.valid_start_counts[clip_ids] - 1
            lo = torch.clamp(windows[:, 0], min=0)
            hi = torch.minimum(windows[:, 1], max_step)
            step_axis = self._step_axis.unsqueeze(0)
            window_mask = (
                active_windows.unsqueeze(1)
                & (hi >= lo).unsqueeze(1)
                & (step_axis >= lo.unsqueeze(1))
                & (step_axis <= hi.unsqueeze(1))
                & valid_step_mask
            )
            if parsed_target_probability is None:
                probabilities = torch.where(
                    window_mask,
                    probabilities * parsed_density_boost,
                    probabilities,
                )
            else:
                window_mass = torch.where(
                    window_mask,
                    probabilities,
                    torch.zeros_like(probabilities),
                ).sum(dim=1, keepdim=True)
                outside_mass = torch.where(
                    window_mask,
                    torch.zeros_like(probabilities),
                    probabilities,
                ).sum(dim=1, keepdim=True)
                has_window_mass = window_mass > 0.0
                has_outside_mass = outside_mass > 0.0
                both_supported = has_window_mass & has_outside_mass
                window_scale = torch.where(
                    both_supported,
                    parsed_target_probability / window_mass.clamp_min(1.0e-12),
                    torch.where(
                        has_window_mass,
                        1.0 / window_mass.clamp_min(1.0e-12),
                        torch.ones_like(window_mass),
                    ),
                )
                outside_scale = torch.where(
                    both_supported,
                    (1.0 - parsed_target_probability) / outside_mass.clamp_min(1.0e-12),
                    torch.where(
                        has_outside_mass,
                        1.0 / outside_mass.clamp_min(1.0e-12),
                        torch.ones_like(outside_mass),
                    ),
                )
                probabilities = torch.where(
                    window_mask,
                    probabilities * window_scale,
                    probabilities * outside_scale,
                )

        total = probabilities.sum(dim=1, keepdim=True)
        if not _trusted_inputs and bool((~torch.isfinite(total) | (total <= 0.0)).any().item()):
            raise RuntimeError(
                "Adaptive timestep probability construction produced a non-finite or zero-mass distribution."
            )
        return probabilities / total

    def _counts_by_bin(
        self,
        time_steps: torch.Tensor,
        clip_ids: torch.Tensor | None = None,
        *,
        weights: torch.Tensor | None = None,
        _trusted_clip_ids: bool = False,
    ) -> torch.Tensor:
        time_steps = self._coerce_time_steps(time_steps, trusted=_trusted_clip_ids)
        if time_steps.numel() == 0:
            return torch.zeros_like(self.current_bin_exposure_count)
        if _trusted_clip_ids:
            if clip_ids is None:
                clip_ids = torch.zeros((time_steps.numel(),), dtype=torch.long, device=self.device)
            else:
                clip_ids = torch.as_tensor(clip_ids, dtype=torch.long, device=self.device).reshape(-1)
                if clip_ids.numel() != time_steps.numel():
                    raise ValueError(f"Expected {time_steps.numel()} clip ids, got {clip_ids.numel()}.")
        else:
            clip_ids = self._resolve_clip_ids(clip_ids, time_steps.numel())
        valid_start_counts = self.valid_start_counts[clip_ids]
        num_bins = self.num_bins_per_clip[clip_ids]
        steps = torch.floor(time_steps).to(dtype=torch.long)
        steps = torch.clamp(steps, min=0)
        steps = torch.minimum(steps, valid_start_counts - 1)
        bin_ids = torch.div(
            steps * num_bins,
            valid_start_counts,
            rounding_mode="floor",
        )
        bin_ids = torch.clamp(bin_ids, min=0)
        bin_ids = torch.minimum(bin_ids, num_bins - 1)
        flat_ids = clip_ids * self.max_num_bins + bin_ids
        if weights is not None:
            weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device).reshape(-1)
            if weights.numel() != time_steps.numel():
                raise ValueError(f"Expected {time_steps.numel()} count weights, got {weights.numel()}.")
        counts = torch.bincount(
            flat_ids,
            weights=weights,
            minlength=self.num_clips * self.max_num_bins,
        ).to(dtype=torch.float32)
        return counts.view(self.num_clips, self.max_num_bins)

    def update_current_bin_exposure_count(
        self,
        time_steps: torch.Tensor,
        clip_ids: torch.Tensor | None = None,
        *,
        observed: torch.Tensor | None = None,
        _trusted_clip_ids: bool = False,
    ) -> None:
        """Accumulate state/action visits before the corresponding timestep is advanced or reset."""
        time_steps = self._coerce_time_steps(time_steps, trusted=_trusted_clip_ids)
        weights = None
        if observed is not None:
            observed = torch.as_tensor(observed, dtype=torch.bool, device=self.device).reshape(-1)
            if observed.numel() != time_steps.numel():
                raise ValueError(f"Expected {time_steps.numel()} observation flags, got {observed.numel()}.")
            weights = observed.to(dtype=torch.float32)
        self.current_bin_exposure_count.add_(
            self._counts_by_bin(
                time_steps,
                clip_ids,
                weights=weights,
                _trusted_clip_ids=_trusted_clip_ids,
            )
        )

    def update_current_bin_outcome_count(
        self,
        time_steps: torch.Tensor,
        *,
        clip_ids: torch.Tensor | None = None,
        failed: torch.Tensor | None = None,
        observed: torch.Tensor | None = None,
        _trusted_clip_ids: bool = False,
    ) -> None:
        """Accumulate reset-time exposure and the subset caused by genuine failure."""
        time_steps = self._coerce_time_steps(time_steps, trusted=_trusted_clip_ids)
        if time_steps.numel() == 0:
            return
        if _trusted_clip_ids:
            if clip_ids is None:
                resolved_clip_ids = torch.zeros(time_steps.numel(), dtype=torch.long, device=self.device)
            else:
                resolved_clip_ids = torch.as_tensor(clip_ids, dtype=torch.long, device=self.device).reshape(-1)
                if resolved_clip_ids.numel() != time_steps.numel():
                    raise ValueError(f"Expected {time_steps.numel()} clip ids, got {resolved_clip_ids.numel()}.")
        else:
            resolved_clip_ids = self._resolve_clip_ids(clip_ids, time_steps.numel())
        if failed is None:
            failed = torch.ones(time_steps.numel(), dtype=torch.bool, device=self.device)
        else:
            failed = torch.as_tensor(failed, dtype=torch.bool, device=self.device).reshape(-1)
            if failed.numel() != time_steps.numel():
                raise ValueError(f"Expected {time_steps.numel()} failure flags, got {failed.numel()}.")
        if observed is None:
            observed = torch.ones(time_steps.numel(), dtype=torch.bool, device=self.device)
        else:
            observed = torch.as_tensor(observed, dtype=torch.bool, device=self.device).reshape(-1)
            if observed.numel() != time_steps.numel():
                raise ValueError(f"Expected {time_steps.numel()} observation flags, got {observed.numel()}.")
        observation_weights = observed.to(dtype=torch.float32)
        self.current_bin_exposure_count.add_(
            self._counts_by_bin(
                time_steps,
                resolved_clip_ids,
                weights=observation_weights,
                _trusted_clip_ids=True,
            )
        )
        self.current_bin_failed_count.add_(
            self._counts_by_bin(
                time_steps,
                resolved_clip_ids,
                weights=(failed & observed).to(dtype=torch.float32),
                _trusted_clip_ids=True,
            )
        )

    def update_current_bin_failed_count(self, failed_at_time_step: torch.Tensor, clip_ids: torch.Tensor | None = None):
        """Compatibility wrapper: every supplied failure is also one observed exposure."""
        self.update_current_bin_outcome_count(
            failed_at_time_step,
            clip_ids=clip_ids,
        )

    def update_bin_failed_count(self):
        """Update failure and exposure EMAs after all reset/visit events for this environment step."""
        self.bin_failed_count = (self.adaptive_alpha * self.current_bin_failed_count) + (
            1 - self.adaptive_alpha
        ) * self.bin_failed_count
        self.bin_exposure_count = (self.adaptive_alpha * self.current_bin_exposure_count) + (
            1 - self.adaptive_alpha
        ) * self.bin_exposure_count
        self.current_bin_failed_count.zero_()
        self.current_bin_exposure_count.zero_()

    @property
    def sampling_probabilities(self) -> torch.Tensor:
        if self.num_clips != 1:
            raise RuntimeError("sampling_probabilities is only defined for single-clip adaptive timestep sampling.")
        return self._sampling_probabilities_for_clip(0)

    def sample(self, clip_ids_or_num_samples: torch.Tensor | int) -> torch.Tensor:
        """Compatibility API returning phases sampled on the valid-start grid."""
        if isinstance(clip_ids_or_num_samples, int):
            clip_ids = self._resolve_clip_ids(None, clip_ids_or_num_samples)
        else:
            clip_ids = self._resolve_clip_ids(clip_ids_or_num_samples, int(clip_ids_or_num_samples.numel()))

        if clip_ids.numel() == 0:
            return torch.zeros((0,), dtype=torch.float32, device=self.device)
        sampled_steps = self.sample_time_steps(clip_ids)
        valid_start_counts = self.valid_start_counts[clip_ids].to(dtype=torch.float32)
        phase_offsets = sampled_steps.to(dtype=torch.float32) + torch.rand(clip_ids.numel(), device=self.device)
        return phase_offsets / valid_start_counts

    def sample_time_steps(
        self,
        clip_ids: torch.Tensor,
        *,
        exclude_zero: bool = False,
        windows: torch.Tensor | None = None,
        window_valid: torch.Tensor | None = None,
        window_density_boost: float = 1.0,
        window_target_probability: float | None = None,
    ) -> torch.Tensor:
        """Sample valid discrete reset timesteps, optionally with contact-window bias."""
        sampled_steps, _ = self._sample_time_steps_with_probabilities(
            clip_ids,
            exclude_zero=exclude_zero,
            windows=windows,
            window_valid=window_valid,
            window_density_boost=window_density_boost,
            window_target_probability=window_target_probability,
        )
        return sampled_steps

    def _sample_time_steps_with_probabilities(
        self,
        clip_ids: torch.Tensor,
        *,
        exclude_zero: bool = False,
        windows: torch.Tensor | None = None,
        window_valid: torch.Tensor | None = None,
        window_density_boost: float = 1.0,
        window_target_probability: float | None = None,
        _trusted_inputs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a reset batch and retain its already-validated probability rows."""
        probabilities = self.timestep_probabilities_for_samples(
            clip_ids,
            exclude_zero=exclude_zero,
            windows=windows,
            window_valid=window_valid,
            window_density_boost=window_density_boost,
            window_target_probability=window_target_probability,
            _trusted_inputs=_trusted_inputs,
        )
        if probabilities.shape[0] == 0:
            sampled_steps = torch.zeros((0,), dtype=torch.long, device=self.device)
        else:
            sampled_steps = torch.multinomial(probabilities, 1, replacement=True).squeeze(1)
        return sampled_steps, probabilities

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 3,
            # These payloads also serve as in-memory canonical-boundary
            # snapshots.  ``cpu()`` aliases CPU-resident live buffers, so each
            # tensor must own its storage before reset/init mutates the source.
            "clip_lengths": self.clip_lengths.detach().to("cpu").clone(),
            "valid_start_counts": self.valid_start_counts.detach().to("cpu").clone(),
            "num_bins_per_clip": self.num_bins_per_clip.detach().to("cpu").clone(),
            "env_fps": int(self.env_fps),
            "adaptive_kernel_size": int(self.adaptive_kernel_size),
            "adaptive_lambda": float(self.adaptive_lambda),
            "adaptive_uniform_ratio": float(self.adaptive_uniform_ratio),
            "adaptive_alpha": float(self.adaptive_alpha),
            "current_bin_failed_count": self.current_bin_failed_count.detach().to("cpu").clone(),
            "bin_failed_count": self.bin_failed_count.detach().to("cpu").clone(),
            "current_bin_exposure_count": self.current_bin_exposure_count.detach().to("cpu").clone(),
            "bin_exposure_count": self.bin_exposure_count.detach().to("cpu").clone(),
        }

    @staticmethod
    def _checkpoint_integer_scalar(value: Any, *, path: str, minimum: int | None = None) -> int:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"Adaptive sampler checkpoint {path} must be one integer scalar.")
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f"Adaptive sampler checkpoint {path} must be an integer, got {value!r}.")
        parsed = int(value)
        if minimum is not None and parsed < minimum:
            raise ValueError(
                f"Adaptive sampler checkpoint {path} must be >= {minimum}, got {parsed}."
            )
        return parsed

    @staticmethod
    def _checkpoint_real_scalar(value: Any, *, path: str) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"Adaptive sampler checkpoint {path} must be one real scalar.")
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"Adaptive sampler checkpoint {path} must be a real number, got {value!r}.")
        parsed = float(value)
        if not np.isfinite(parsed):
            raise ValueError(f"Adaptive sampler checkpoint {path} must be finite, got {value!r}.")
        return parsed

    def _checkpoint_integer_tensor(self, value: Any, *, path: str) -> torch.Tensor:
        if value is None:
            raise ValueError(f"Adaptive sampler checkpoint is missing {path}.")
        restored = torch.as_tensor(value, device=self.device)
        if restored.dtype == torch.bool or restored.is_floating_point() or restored.is_complex():
            raise ValueError(
                f"Adaptive sampler checkpoint {path} must use an integer dtype, got {restored.dtype}."
            )
        return restored.to(dtype=torch.long).reshape(-1)

    def _prepare_state_dict(self, state: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Validate the complete sampler payload without mutating live buffers."""

        if not isinstance(state, dict):
            raise ValueError("Adaptive timestep sampler checkpoint state must be a dictionary.")
        version = self._checkpoint_integer_scalar(state.get("version", 0), path="version")
        if version not in (1, 2, 3):
            raise ValueError(f"Unsupported adaptive timestep sampler checkpoint version: {state.get('version')!r}.")
        geometry = {
            "clip_lengths": self.clip_lengths,
            "valid_start_counts": self.valid_start_counts,
            "num_bins_per_clip": self.num_bins_per_clip,
        }
        for key, expected in geometry.items():
            restored = self._checkpoint_integer_tensor(state.get(key), path=key)
            if not torch.equal(restored, expected):
                raise ValueError(
                    f"Adaptive sampler checkpoint {key} does not match the current motion bank: "
                    f"checkpoint={restored.detach().cpu().tolist()}, current={expected.detach().cpu().tolist()}."
                )
        checkpoint_env_fps = self._checkpoint_integer_scalar(
            state.get("env_fps", -1),
            path="env_fps",
            minimum=1,
        )
        if checkpoint_env_fps != int(self.env_fps):
            raise ValueError(
                f"Adaptive sampler checkpoint env_fps={state.get('env_fps')} does not match current {self.env_fps}."
            )

        runtime_hyperparameters: dict[str, int | float] = {
            "adaptive_kernel_size": int(self.adaptive_kernel_size),
            "adaptive_lambda": float(self.adaptive_lambda),
            "adaptive_uniform_ratio": float(self.adaptive_uniform_ratio),
            "adaptive_alpha": float(self.adaptive_alpha),
        }
        if version >= 3:
            checkpoint_hyperparameters: dict[str, int | float] = {
                "adaptive_kernel_size": self._checkpoint_integer_scalar(
                    state.get("adaptive_kernel_size"),
                    path="adaptive_kernel_size",
                    minimum=1,
                ),
                "adaptive_lambda": self._checkpoint_real_scalar(
                    state.get("adaptive_lambda"), path="adaptive_lambda"
                ),
                "adaptive_uniform_ratio": self._checkpoint_real_scalar(
                    state.get("adaptive_uniform_ratio"), path="adaptive_uniform_ratio"
                ),
                "adaptive_alpha": self._checkpoint_real_scalar(
                    state.get("adaptive_alpha"), path="adaptive_alpha"
                ),
            }
            hyperparameter_mismatches = [
                f"{key}: checkpoint={checkpoint_hyperparameters[key]!r}, current={current!r}"
                for key, current in runtime_hyperparameters.items()
                if checkpoint_hyperparameters[key] != current
            ]
            if hyperparameter_mismatches:
                raise ValueError(
                    "Adaptive sampler checkpoint hyperparameters do not match the current sampler: "
                    + "; ".join(hyperparameter_mismatches)
                )
        else:
            # Versions 1/2 predate serialized sampler hyperparameters.  The
            # production MotionCommand constructor used these exact defaults,
            # so legacy states remain safe only with that historical runtime.
            legacy_defaults: dict[str, int | float] = {
                "adaptive_kernel_size": 1,
                "adaptive_lambda": 0.8,
                "adaptive_uniform_ratio": 0.1,
                "adaptive_alpha": 0.001,
            }
            legacy_mismatches = [
                f"{key}: legacy_default={legacy_defaults[key]!r}, current={current!r}"
                for key, current in runtime_hyperparameters.items()
                if legacy_defaults[key] != current
            ]
            if legacy_mismatches:
                raise ValueError(
                    f"Adaptive sampler checkpoint version {version} does not encode sampler "
                    "hyperparameters and can only be restored with the historical production defaults: "
                    + "; ".join(legacy_mismatches)
                )

        restored_buffers: dict[str, torch.Tensor] = {}
        required_buffer_names = ["current_bin_failed_count", "bin_failed_count"]
        if version >= 2:
            required_buffer_names.extend(("current_bin_exposure_count", "bin_exposure_count"))
        valid_mask = (
            torch.arange(self.max_num_bins, device=self.device).unsqueeze(0)
            < self.num_bins_per_clip.unsqueeze(1)
        )
        for key in required_buffer_names:
            target = getattr(self, key)
            raw_value = state.get(key)
            if raw_value is None:
                raise ValueError(f"Adaptive sampler checkpoint is missing {key}.")
            restored = torch.as_tensor(raw_value, device=self.device)
            if restored.shape != target.shape:
                raise ValueError(
                    f"Adaptive sampler checkpoint {key} shape {tuple(restored.shape)} does not match "
                    f"current {tuple(target.shape)}."
                )
            if restored.dtype != target.dtype:
                raise ValueError(
                    f"Adaptive sampler checkpoint {key} dtype {restored.dtype} does not match "
                    f"current {target.dtype}."
                )
            if not bool(torch.isfinite(restored).all().item()):
                raise ValueError(f"Adaptive sampler checkpoint {key} contains NaN or infinity.")
            if bool((restored < 0.0).any().item()):
                raise ValueError(f"Adaptive sampler checkpoint {key} must be non-negative.")
            if bool((restored[~valid_mask] != 0.0).any().item()):
                raise ValueError(
                    f"Adaptive sampler checkpoint {key} has nonzero values in padded invalid bins."
                )
            restored_buffers[key] = restored.detach().clone()

        if version >= 2:
            for failed_key, exposure_key in (
                ("current_bin_failed_count", "current_bin_exposure_count"),
                ("bin_failed_count", "bin_exposure_count"),
            ):
                failed = restored_buffers[failed_key]
                exposure = restored_buffers[exposure_key]
                tolerance = 1.0e-6 * torch.maximum(torch.ones_like(exposure), exposure)
                if bool((failed > exposure + tolerance).any().item()):
                    raise ValueError(
                        f"Adaptive sampler checkpoint {failed_key} exceeds {exposure_key}; "
                        "failure events must be a subset of observed exposures."
                    )
        else:
            # Version 1 stored raw failure EMA only.  Unit exposure on valid
            # bins preserves its relative adaptive priorities while allowing
            # the new failure-rate sampler to resume safely.
            restored_buffers["current_bin_exposure_count"] = restored_buffers[
                "current_bin_failed_count"
            ].clone()
            legacy_exposure = torch.zeros_like(self.bin_exposure_count)
            for clip_id in range(self.num_clips):
                valid_bins = int(self.num_bins_per_clip[clip_id].item())
                # v1 stored a failure-count EMA rather than a rate denominator.
                # A clip-wise constant at least as large as every failure bin
                # preserves the old relative priorities while satisfying the
                # v2+ invariant that failures are a subset of exposures.
                max_failure = restored_buffers["bin_failed_count"][
                    clip_id, :valid_bins
                ].max()
                legacy_exposure[clip_id, :valid_bins] = torch.clamp(
                    max_failure,
                    min=1.0,
                )
            restored_buffers["bin_exposure_count"] = legacy_exposure
        return restored_buffers

    def validate_state_dict(self, state: dict[str, Any]) -> None:
        """Prove that ``load_state_dict`` can apply the payload atomically."""

        self._prepare_state_dict(state)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore adaptive statistics after complete fail-closed validation."""

        restored_buffers = self._prepare_state_dict(state)
        for key, restored in restored_buffers.items():
            getattr(self, key).copy_(restored)

    def get_stats(self, probability_overrides: list[torch.Tensor] | None = None):
        # Metrics
        entropies: list[torch.Tensor] = []
        top1_probs: list[torch.Tensor] = []
        top1_bins: list[torch.Tensor] = []
        for clip_id in range(self.num_clips):
            prob = (
                self._sampling_probabilities_for_clip(clip_id)
                if probability_overrides is None
                else probability_overrides[clip_id]
            )
            if prob.numel() <= 1:
                entropies.append(torch.zeros((), device=self.device, dtype=torch.float32))
                top1_probs.append(torch.ones((), device=self.device, dtype=torch.float32))
                top1_bins.append(torch.zeros((), device=self.device, dtype=torch.float32))
                continue
            H = -(prob * (prob + 1e-12).log()).sum()
            H_norm = H / np.log(float(prob.numel()))
            pmax, imax = prob.max(dim=0)
            entropies.append(H_norm.to(dtype=torch.float32))
            top1_probs.append(pmax.to(dtype=torch.float32))
            top1_bins.append(imax.to(dtype=torch.float32) / float(prob.numel()))

        self.metrics["sampling_entropy"] = torch.stack(entropies).mean()
        self.metrics["sampling_top1_prob"] = torch.stack(top1_probs).mean()
        self.metrics["sampling_top1_bin"] = torch.stack(top1_bins).mean()


#########################################################################################################
## Helper functions
#########################################################################################################
FAKE_BODY_NAME_ALIASES: dict[str, str] = {
    # Fake foot contact bodies are authored in the URDF purely for height computation.
    # They do not exist in the motion-capture dataset, so we alias them back to the
    # closest real body when indexing into motion data. These are not actually used in training.
    "left_foot_contact_point": "left_ankle_roll_link",
    "right_foot_contact_point": "right_ankle_roll_link",
}


def get_filtered_body_names(body_list: List[str], pattern: str) -> List[str]:
    return [body_name for body_name in body_list if re.match(pattern, body_name)]


class MotionCommand(CommandTermBase):
    def __init__(self, cfg: Any, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        self._env = env
        # self.motion_cfg: MotionConfig = cfg.params["motion_config"]
        # TODO(jchen):temporary fix for motion_config being a dict after tyro.cli
        if isinstance(cfg.params["motion_config"], MotionConfig):
            self.motion_cfg = cfg.params["motion_config"]
        else:
            self.motion_cfg = MotionConfig(**cfg.params["motion_config"])
        self.init_pose_cfg: NoiseToInitialPoseConfig = self.motion_cfg.noise_to_initial_pose
        self._clip_terrain_offsets: torch.Tensor | None = None
        self._clip_terrain_offsets_by_row: torch.Tensor | None = None
        self._terrain_row_ids: torch.Tensor | None = None
        self._terrain_row_stride: float = 0.0
        self._terrain_row_count: int = 0
        self._forced_clip_idx: int | None = None
        self._forced_reset_timestep: int | None = None
        self.manual_control_enabled = False
        self.manual_xy_rel: torch.Tensor | None = None
        self.manual_yaw_rel: torch.Tensor | None = None
        self.manual_pickup_button_override_enabled = False
        self.manual_pickup_button: torch.Tensor | None = None
        self.manual_drop_button_override_enabled = False
        self.manual_drop_button: torch.Tensor | None = None
        self._manual_forward_after_lift_enabled = False
        self._manual_forward_after_lift_command_m = 0.0
        self._manual_forward_after_lift_rel_z_delta_m = 0.0
        self._manual_forward_after_lift_consecutive_steps = 0
        self._manual_forward_after_lift_preserve_native_contact_buttons = False
        self._manual_forward_after_lift_preserve_native_pickup_button = False
        self._manual_forward_after_lift_preserve_native_drop_button = False
        self._manual_forward_after_lift_baseline_object_z: torch.Tensor | None = None
        self._manual_forward_after_lift_consecutive_count: torch.Tensor | None = None
        self._manual_forward_after_lift_triggered: torch.Tensor | None = None
        self._manual_forward_after_lift_trigger_episode_step: torch.Tensor | None = None
        self._manual_forward_after_lift_command_semantics = (
            "legacy_constant_robot_heading_frame"
        )
        self._manual_forward_heading_lock_enabled = False
        self._manual_forward_heading_lock_command_m = 0.0
        self._manual_forward_heading_lock_active: torch.Tensor | None = None
        self._manual_forward_heading_lock_origin_xy_w: torch.Tensor | None = None
        self._manual_forward_heading_lock_yaw_w: torch.Tensor | None = None
        self.manual_object_reset_enabled = False
        self.manual_object_reset_pos_offset_w: torch.Tensor | None = None
        self.manual_object_reset_rpy_offset: torch.Tensor | None = None
        self._training_iteration: int | None = None
        self._training_total_iterations: int | None = None
        self._clean_noisy_clip_curriculum_cfg: CleanNoisyClipCurriculumConfig | None = None
        self._clean_noisy_clip_curriculum_enabled = False
        self._clean_clip_mask: torch.Tensor | None = None
        self._noisy_clip_mask: torch.Tensor | None = None
        self._fixed_clip_group_assignment_cfg: FixedClipGroupAssignmentConfig | None = None
        self._fixed_clip_group_env_mask: torch.Tensor | None = None
        self._fixed_clip_group_clip_mask: torch.Tensor | None = None
        self._fixed_clip_complement_clip_mask: torch.Tensor | None = None
        self.hybrid_stage2_task_env_mask: torch.Tensor | None = None
        self.hybrid_velocity_task_env_mask: torch.Tensor | None = None
        self._hybrid_velocity_task_priority: torch.Tensor | None = None
        self.hmi_cfg: HMIMotionConfig | None = self.motion_cfg.hmi
        self.hmi_track_env_mask: torch.Tensor | None = None
        self.hmi_gen_env_mask: torch.Tensor | None = None
        self.hmi_exact_goal_object_pos_w: torch.Tensor | None = None
        self.hmi_exact_goal_object_quat_w: torch.Tensor | None = None
        self.hmi_goal_object_pos_w: torch.Tensor | None = None
        self.hmi_goal_object_quat_w: torch.Tensor | None = None
        self.hmi_goal_version: torch.Tensor | None = None
        self.hmi_goal_reached: torch.Tensor | None = None
        self.hmi_goal_noise_scale: torch.Tensor | None = None
        self.hmi_goal_success_ema: torch.Tensor | None = None
        self.hmi_goal_success_ema_initialized = False
        self.hmi_goal_success_sum: torch.Tensor | None = None
        self.hmi_goal_success_count: torch.Tensor | None = None
        self.hmi_last_curriculum_update_iteration = 0
        self.pickup_anchor_set: torch.Tensor | None = None
        self.pickup_anchor_root_pos_w: torch.Tensor | None = None
        self.pickup_anchor_root_quat_w: torch.Tensor | None = None
        self.pickup_anchor_object_pos_b: torch.Tensor | None = None
        self.pickup_anchor_object_quat_b: torch.Tensor | None = None
        self.pickup_object_rel_z_baseline: torch.Tensor | None = None
        self.hybrid_velocity_object_z_baseline: torch.Tensor | None = None
        self.pickup_consecutive_counter: torch.Tensor | None = None
        self._multi_object_enabled = False
        self._sim_object_names: list[str] = []
        self._clip_object_ids: torch.Tensor | None = None
        self._object_indices_matrix: torch.Tensor | None = None
        self._fixed_clip_ids: torch.Tensor | None = None
        self.object_name: str = "object"
        self.object_indices_in_simulator: torch.Tensor | None = None
        # One authoritative active-object snapshot is shared by every WBT
        # consumer during a control step.  In IsaacSim, repeatedly indexing
        # AllRootStatesProxy otherwise performs CUDA-to-host index decoding for
        # each position/quaternion/velocity property access.
        self._simulator_object_state_snapshot: torch.Tensor | None = None
        self._simulator_object_state_snapshot_ready = False
        self._debug_representative_clip_ids: torch.Tensor | None = None
        self._contact_prior_active = False
        self._contact_prior_available = False
        self._contact_prior_force_body_names_by_region: dict[str, list[str]] = {}
        self._contact_prior_position_body_names_by_region: dict[str, list[str]] = {}
        self._contact_prior_position_body_indices_by_region: dict[str, torch.Tensor] = {}
        self._contact_prior_total_count: torch.Tensor | None = None
        self._contact_prior_contact_sum: torch.Tensor | None = None
        self._contact_prior_force_mean: torch.Tensor | None = None
        self._contact_prior_force_count: torch.Tensor | None = None
        self._contact_prior_position_mean: torch.Tensor | None = None
        self._contact_prior_position_count: torch.Tensor | None = None
        self._object_contact_body_indices_cache: dict[tuple[str, ...], torch.Tensor] = {}
        self._adaptive_sampling_contact_interval_root: Path | None = None
        self._adaptive_sampling_contact_window_by_clip: torch.Tensor | None = None
        self._adaptive_sampling_contact_window_valid_by_clip: torch.Tensor | None = None
        # Python-side copy of the static contact bank.  Hot-path diagnostics
        # use this to avoid per-clip CUDA ``.item()`` calls for indices and
        # interval endpoints.
        self._adaptive_sampling_contact_intervals_by_clip: dict[int, tuple[int, int]] = {}
        # ``device`` is assigned by setup(), after CommandManager constructs
        # every term.  Keep constructor state device-independent and publish
        # device scalars only once setup has bound the environment device.
        self._uniform_t1_window_last_reset_available_frac = 0.0
        self._uniform_t1_window_last_reset_sample_frac = 0.0
        self._uniform_t1_window_last_reset_expected_sample_frac = 0.0
        self._uniform_t1_window_last_reset_sample_frac_valid = 0.0
        self._uniform_t1_window_last_reset_expected_sample_frac_valid = 0.0
        self._uniform_t1_window_last_reset_mean_window_len = 0.0
        self._rank_local_shard_metadata: dict[str, Any] | None = None
        self._rank_local_inverse_cover_weights: torch.Tensor | None = None
        self.distributed_loss_weight = 1.0
        self._static_default_pose_prepend_steps = 0
        self._static_default_pose_append_steps = 0
        self._runtime_default_pose_prepend_enabled = False
        self._runtime_default_pose_prepend_steps = 0
        self._runtime_default_pose_prepend_active: torch.Tensor | None = None
        self._runtime_default_pose_prepend_step: torch.Tensor | None = None
        self._runtime_default_pose_prepend_defaults: dict[str, torch.Tensor] = {}

    def _effective_initial_pose_noise_config(self) -> NoiseToInitialPoseConfig:
        """Return the final-writer reset randomization configured for this run."""

        randomization_manager = getattr(self._env, "randomization_manager", None)
        get_state = getattr(randomization_manager, "get_state", None)
        state = (
            get_state("motion_relative_reset_randomizer_state")
            if callable(get_state)
            else None
        )
        if state is None:
            return self.init_pose_cfg
        noise_config = getattr(state, "noise_config", None)
        if not isinstance(noise_config, NoiseToInitialPoseConfig):
            raise TypeError(
                "motion_relative_reset_randomizer_state must expose a "
                "NoiseToInitialPoseConfig as noise_config."
            )
        return noise_config

    def _initialize_uniform_t1_window_metric_state(self) -> None:
        zeros = torch.zeros((6,), device=self.device, dtype=torch.float32).unbind()
        (
            self._uniform_t1_window_last_reset_available_frac,
            self._uniform_t1_window_last_reset_sample_frac,
            self._uniform_t1_window_last_reset_expected_sample_frac,
            self._uniform_t1_window_last_reset_sample_frac_valid,
            self._uniform_t1_window_last_reset_expected_sample_frac_valid,
            self._uniform_t1_window_last_reset_mean_window_len,
        ) = zeros

    def set_forced_clip(self, clip_idx: int | None) -> None:
        """Force a specific clip index for resets (None clears the override)."""
        if clip_idx is None:
            self._forced_clip_idx = None
            return
        if clip_idx < 0 or clip_idx >= self.motion.num_clips:
            raise ValueError(f"clip_idx {clip_idx} out of range for {self.motion.num_clips} clips.")
        self._forced_clip_idx = int(clip_idx)

    def set_forced_reset_timestep(self, timestep: int | None) -> None:
        """Force an exact motion timestep on reset, primarily for controlled evaluation."""
        if timestep is None:
            self._forced_reset_timestep = None
            return
        if isinstance(timestep, bool) or not isinstance(timestep, numbers.Integral):
            raise TypeError(f"timestep must be an integer or None, got {timestep!r}.")
        if int(timestep) < 0:
            raise ValueError(f"timestep must be non-negative, got {timestep}.")
        self._forced_reset_timestep = int(timestep)

    def set_fixed_clip_ids_for_envs(self, env_ids, clip_ids) -> None:
        """Pin specific envs to specific clips for subsequent resets."""
        env_ids_t = self._ensure_index_tensor(env_ids)
        clip_ids_t = torch.as_tensor(clip_ids, device=self.device, dtype=torch.long).reshape(-1)
        if env_ids_t.numel() != clip_ids_t.numel():
            raise ValueError(
                f"env_ids and clip_ids must have the same length, got {env_ids_t.numel()} and {clip_ids_t.numel()}."
            )
        if clip_ids_t.numel() == 0:
            return
        if torch.any(clip_ids_t < 0) or torch.any(clip_ids_t >= int(self.motion.num_clips)):
            raise ValueError(f"clip_ids must be within [0, {self.motion.num_clips - 1}].")
        if self._fixed_clip_ids is None or int(self._fixed_clip_ids.numel()) != int(self.num_envs):
            if hasattr(self, "clip_ids") and isinstance(self.clip_ids, torch.Tensor) and int(self.clip_ids.numel()) == int(self.num_envs):
                self._fixed_clip_ids = self.clip_ids.clone()
            else:
                self._fixed_clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._fixed_clip_ids[env_ids_t] = clip_ids_t

    def set_training_iteration(self, iteration: int, *, total_iterations: int | None = None) -> None:
        """Expose the current PPO iteration so command curriculum can follow the training schedule exactly."""
        self._training_iteration = int(iteration)
        self._training_total_iterations = None if total_iterations is None else int(total_iterations)
        self._update_hmi_goal_noise_curriculum(int(iteration))
        self._refresh_current_clip_sampling_weights()

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Serialize motion-bank adaptive curriculum state for exact resume."""
        state: dict[str, Any] = {
            "version": 3,
            "clip_ids": list(self.motion.clip_ids),
            "valid_start_counts": self._valid_start_counts()
            .to(dtype=torch.long)
            .detach()
            .to("cpu")
            .clone(),
            "clip_weighting_strategy": str(self.clip_weighting_strategy),
            "training_iteration": self._training_iteration,
            "distributed_loss_weight": float(self.distributed_loss_weight),
            "contact_prior": self._get_contact_prior_checkpoint_state(),
        }
        hmi_contract = self._hmi_checkpoint_contract()
        if hmi_contract is not None:
            state["hmi_contract"] = hmi_contract
            state["hmi_state"] = self._hmi_checkpoint_state()
        if self.adaptive_timesteps_sampler is not None:
            state["adaptive_timesteps_sampler"] = self.adaptive_timesteps_sampler.state_dict()
        for key in ("_clip_success_counts", "_clip_total_counts", "_raw_clip_sampling_weights"):
            value = getattr(self, key, None)
            if isinstance(value, torch.Tensor):
                state[key.removeprefix("_")] = value.detach().to("cpu").clone()
        return state

    def _hmi_checkpoint_contract(self) -> dict[str, Any] | None:
        """Return the immutable HMI semantics that must match on exact resume."""

        cfg = getattr(self, "hmi_cfg", None)
        if cfg is None:
            return None
        goal_noise = cfg.object_goal_noise
        return {
            "version": 1,
            "upstream_repository": str(cfg.upstream_repository),
            "upstream_commit": str(cfg.upstream_commit),
            "interface": str(cfg.actor_interface_semantics),
            "track_ratio": float(cfg.track_ratio),
            "env_partition_seed": int(cfg.env_partition_seed),
            "gen_start_at_timestep_zero_prob": (
                None
                if cfg.gen_start_at_timestep_zero_prob is None
                else float(cfg.gen_start_at_timestep_zero_prob)
            ),
            "object_goal_noise": {
                "pos_std_xyz": [float(value) for value in goal_noise.pos_std_xyz],
                "pos_clip_xyz": [float(value) for value in goal_noise.pos_clip_xyz],
                "rpy_std": [float(value) for value in goal_noise.rpy_std],
                "rpy_clip": [float(value) for value in goal_noise.rpy_clip],
            },
            "gen_step_zero_root_pos_std_xyz": [
                float(value) for value in cfg.gen_step_zero_root_pos_std_xyz
            ],
            "gen_step_zero_root_pos_clip_xyz": [
                float(value) for value in cfg.gen_step_zero_root_pos_clip_xyz
            ],
            "gen_step_zero_root_rpy_std": [
                float(value) for value in cfg.gen_step_zero_root_rpy_std
            ],
            "gen_step_zero_root_rpy_clip": [
                float(value) for value in cfg.gen_step_zero_root_rpy_clip
            ],
            "goal_noise_curriculum": {
                "initial_scale": float(cfg.goal_noise_initial_scale),
                "min_scale": float(cfg.goal_noise_min_scale),
                "max_scale": float(cfg.goal_noise_max_scale),
                "scale_step": float(cfg.goal_noise_scale_step),
                "success_threshold_up": float(cfg.goal_noise_success_threshold_up),
                "success_threshold_down": float(cfg.goal_noise_success_threshold_down),
                "update_interval": int(cfg.goal_noise_update_interval),
                "ema_alpha": float(cfg.goal_noise_ema_alpha),
            },
        }

    def _hmi_checkpoint_state(self) -> dict[str, Any]:
        tensors = {
            "goal_noise_scale": self.hmi_goal_noise_scale,
            "goal_success_ema": self.hmi_goal_success_ema,
            "goal_success_sum": self.hmi_goal_success_sum,
            "goal_success_count": self.hmi_goal_success_count,
        }
        missing = [name for name, value in tensors.items() if value is None]
        if missing:
            raise RuntimeError(f"HMI checkpoint state requested before buffer initialization: {missing}.")
        return {
            "version": 1,
            **{
                name: value.detach().to("cpu").clone()
                for name, value in tensors.items()
                if value is not None
            },
            "goal_success_ema_initialized": bool(self.hmi_goal_success_ema_initialized),
            "last_curriculum_update_iteration": int(
                self.hmi_last_curriculum_update_iteration
            ),
        }

    @staticmethod
    def _contact_prior_checkpoint_tensor_names() -> tuple[str, ...]:
        return (
            "total_count",
            "contact_sum",
            "force_mean",
            "force_count",
            "position_mean",
            "position_count",
        )

    def _contact_prior_checkpoint_targets(self) -> dict[str, torch.Tensor | None]:
        return {
            "total_count": getattr(self, "_contact_prior_total_count", None),
            "contact_sum": getattr(self, "_contact_prior_contact_sum", None),
            "force_mean": getattr(self, "_contact_prior_force_mean", None),
            "force_count": getattr(self, "_contact_prior_force_count", None),
            "position_mean": getattr(self, "_contact_prior_position_mean", None),
            "position_count": getattr(self, "_contact_prior_position_count", None),
        }

    def _contact_prior_checkpoint_schema(self) -> dict[str, Any]:
        region_names = list(_CONTACT_PRIOR_REGION_NAMES)
        force_names_by_region = getattr(self, "_contact_prior_force_body_names_by_region", {})
        position_names_by_region = getattr(self, "_contact_prior_position_body_names_by_region", {})
        return {
            "clip_ids": list(self.motion.clip_ids),
            "phase_names": list(_CONTACT_PRIOR_PHASE_NAMES),
            "region_names": region_names,
            "force_body_names_by_region": {
                region_name: list(force_names_by_region.get(region_name, [])) for region_name in region_names
            },
            "position_body_names_by_region": {
                region_name: list(position_names_by_region.get(region_name, [])) for region_name in region_names
            },
        }

    def _validate_contact_prior_checkpoint_schema(self, schema: Any) -> None:
        expected_schema = self._contact_prior_checkpoint_schema()
        expected_keys = set(expected_schema)
        if not isinstance(schema, dict) or set(schema) != expected_keys:
            actual_keys = sorted(map(str, schema)) if isinstance(schema, dict) else type(schema).__name__
            raise ValueError(
                "Motion command checkpoint contact_prior.schema must contain exactly "
                f"{sorted(expected_keys)}; got {actual_keys}."
            )
        for axis_name in ("clip_ids", "phase_names", "region_names"):
            axis_values = schema[axis_name]
            if not isinstance(axis_values, list) or not all(type(value) is str for value in axis_values):
                raise ValueError(
                    f"Motion command checkpoint contact_prior.schema.{axis_name} must be an ordered list of strings."
                )
        checkpoint_region_names = schema["region_names"]
        for mapping_name in ("force_body_names_by_region", "position_body_names_by_region"):
            mapping = schema[mapping_name]
            if not isinstance(mapping, dict) or set(mapping) != set(checkpoint_region_names):
                raise ValueError(
                    f"Motion command checkpoint contact_prior.schema.{mapping_name} must map every declared "
                    "region exactly once."
                )
            for region_name, body_names in mapping.items():
                if type(region_name) is not str or not isinstance(body_names, list) or not all(
                    type(body_name) is str for body_name in body_names
                ):
                    raise ValueError(
                        f"Motion command checkpoint contact_prior.schema.{mapping_name} values must be "
                        "ordered lists of body-name strings."
                    )
        if schema != expected_schema:
            raise ValueError(
                "Motion command checkpoint contact-prior clip/phase/region schema differs from the current runtime: "
                f"checkpoint={schema!r}, current={expected_schema!r}."
            )

    def _contact_prior_runtime_flags(self) -> tuple[bool, bool]:
        active = getattr(self, "_contact_prior_active", False)
        available = getattr(self, "_contact_prior_available", False)
        if type(active) is not bool or type(available) is not bool:
            raise RuntimeError(
                "Motion command contact-prior active/available flags must be booleans before checkpointing."
            )
        if available and not active:
            raise RuntimeError("Motion command contact prior cannot be available while it is inactive.")
        return active, available

    def _contact_prior_expected_tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        num_clips = len(self.motion.clip_ids)
        num_regions = len(_CONTACT_PRIOR_REGION_NAMES)
        prefix = (num_clips, _CONTACT_PRIOR_PHASE_COUNT)
        return {
            "total_count": prefix,
            "contact_sum": (*prefix, num_regions),
            "force_mean": (*prefix, num_regions),
            "force_count": (*prefix, num_regions),
            "position_mean": (*prefix, num_regions, 3),
            "position_count": (*prefix, num_regions),
        }

    def _get_contact_prior_checkpoint_state(self) -> dict[str, Any]:
        """Snapshot all online contact-prior statistics without aliasing live CPU buffers."""

        active, available = self._contact_prior_runtime_flags()
        targets = self._contact_prior_checkpoint_targets()
        expected_shapes = self._contact_prior_expected_tensor_shapes()
        tensors: dict[str, torch.Tensor] = {}
        for name in self._contact_prior_checkpoint_tensor_names():
            target = targets[name]
            if not isinstance(target, torch.Tensor):
                raise RuntimeError(
                    f"Motion command contact-prior tensor {name!r} is not initialized; "
                    "checkpointing before init_buffers() cannot provide exact resume."
                )
            if tuple(target.shape) != expected_shapes[name] or target.dtype != torch.float32:
                raise RuntimeError(
                    f"Motion command live contact-prior tensor {name!r} has incompatible "
                    f"shape/dtype {tuple(target.shape)}/{target.dtype}; expected "
                    f"{expected_shapes[name]}/torch.float32."
                )
            tensors[name] = target.detach().to("cpu").clone()
        # Validate the live snapshot through the same path used for loaded data.
        self._validate_contact_prior_tensor_invariants(tensors)
        return {
            "active": active,
            "available": available,
            "schema": self._contact_prior_checkpoint_schema(),
            **tensors,
        }

    def _validate_contact_prior_tensor_invariants(self, tensors: dict[str, torch.Tensor]) -> None:
        count_names = ("total_count", "contact_sum", "force_count", "position_count")
        for name, value in tensors.items():
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"Motion command checkpoint contact_prior.{name} contains NaN or infinity.")
        for name in count_names:
            value = tensors[name]
            if bool((value < 0.0).any().item()):
                raise ValueError(f"Motion command checkpoint contact_prior.{name} must be non-negative.")
            if bool((value != torch.floor(value)).any().item()):
                raise ValueError(f"Motion command checkpoint contact_prior.{name} must contain integer counts.")

        total_count = tensors["total_count"].unsqueeze(-1)
        contact_sum = tensors["contact_sum"]
        if bool((contact_sum > total_count).any().item()):
            raise ValueError(
                "Motion command checkpoint contact_prior.contact_sum cannot exceed total_count."
            )
        for name in ("force_count", "position_count"):
            if not torch.equal(tensors[name], contact_sum):
                raise ValueError(
                    f"Motion command checkpoint contact_prior.{name} must equal contact_sum; "
                    "each mean sample must correspond to one observed contact."
                )

        force_mean = tensors["force_mean"]
        if bool((force_mean < 0.0).any().item()):
            raise ValueError("Motion command checkpoint contact_prior.force_mean must be non-negative.")
        no_force_samples = tensors["force_count"] == 0.0
        if bool((force_mean[no_force_samples] != 0.0).any().item()):
            raise ValueError(
                "Motion command checkpoint contact_prior.force_mean must be zero where force_count is zero."
            )
        no_position_samples = tensors["position_count"] == 0.0
        if bool((tensors["position_mean"][no_position_samples] != 0.0).any().item()):
            raise ValueError(
                "Motion command checkpoint contact_prior.position_mean must be zero where position_count is zero."
            )

    def _prepare_contact_prior_checkpoint_state(self, state: Any) -> dict[str, torch.Tensor]:
        if not isinstance(state, dict):
            raise ValueError("Motion command checkpoint contact_prior must be a dictionary.")
        expected_keys = {
            "active",
            "available",
            "schema",
            *self._contact_prior_checkpoint_tensor_names(),
        }
        if set(state) != expected_keys:
            missing = sorted(map(str, expected_keys - set(state)))
            unexpected = sorted(map(str, set(state) - expected_keys))
            raise ValueError(
                "Motion command checkpoint contact_prior keys differ from the exact-resume schema: "
                f"missing={missing}, unexpected={unexpected}."
            )

        checkpoint_active = state["active"]
        checkpoint_available = state["available"]
        if type(checkpoint_active) is not bool or type(checkpoint_available) is not bool:
            raise ValueError("Motion command checkpoint contact_prior active/available must be booleans.")
        if checkpoint_available and not checkpoint_active:
            raise ValueError("Motion command checkpoint contact_prior cannot be available while inactive.")
        current_active, current_available = self._contact_prior_runtime_flags()
        if (checkpoint_active, checkpoint_available) != (current_active, current_available):
            raise ValueError(
                "Motion command checkpoint contact-prior activation differs from the current runtime: "
                f"checkpoint=(active={checkpoint_active}, available={checkpoint_available}), "
                f"current=(active={current_active}, available={current_available})."
            )

        self._validate_contact_prior_checkpoint_schema(state["schema"])

        targets = self._contact_prior_checkpoint_targets()
        expected_shapes = self._contact_prior_expected_tensor_shapes()
        restored_tensors: dict[str, torch.Tensor] = {}
        for name in self._contact_prior_checkpoint_tensor_names():
            target = targets[name]
            if not isinstance(target, torch.Tensor):
                raise ValueError(
                    f"Current motion command contact-prior tensor {name!r} is not initialized; exact resume is impossible."
                )
            raw_value = state[name]
            if not isinstance(raw_value, torch.Tensor):
                raise ValueError(f"Motion command checkpoint contact_prior.{name} must be a tensor.")
            restored = raw_value.detach().to(device=self.device)
            if tuple(target.shape) != expected_shapes[name] or target.dtype != torch.float32:
                raise ValueError(
                    f"Current motion command contact-prior tensor {name!r} has incompatible "
                    f"shape/dtype {tuple(target.shape)}/{target.dtype}."
                )
            if tuple(restored.shape) != expected_shapes[name]:
                raise ValueError(
                    f"Motion command checkpoint contact_prior.{name} shape {tuple(restored.shape)} does not match "
                    f"current schema {expected_shapes[name]}."
                )
            if restored.dtype != target.dtype:
                raise ValueError(
                    f"Motion command checkpoint contact_prior.{name} dtype {restored.dtype} does not match "
                    f"current {target.dtype}."
                )
            restored_tensors[name] = restored.clone()
        self._validate_contact_prior_tensor_invariants(restored_tensors)
        return restored_tensors

    def _prepare_hmi_checkpoint_state(self, state: Any) -> dict[str, Any] | None:
        if not self.hmi_enabled():
            if state is not None:
                raise ValueError("Non-HMI runtime cannot restore HMI curriculum state.")
            return None
        if not isinstance(state, dict):
            raise ValueError("HMI exact resume requires motion_command.hmi_state.")
        expected_keys = {
            "version",
            "goal_noise_scale",
            "goal_success_ema",
            "goal_success_sum",
            "goal_success_count",
            "goal_success_ema_initialized",
            "last_curriculum_update_iteration",
        }
        if set(state) != expected_keys:
            raise ValueError(
                "Motion command HMI state keys differ from the exact-resume schema: "
                f"missing={sorted(expected_keys - set(state))}, "
                f"unexpected={sorted(set(state) - expected_keys)}."
            )
        version = AdaptiveTimestepsSampler._checkpoint_integer_scalar(
            state["version"], path="motion_command.hmi_state.version", minimum=1
        )
        if version != 1:
            raise ValueError(f"Unsupported HMI checkpoint-state version {version}.")
        assert self.hmi_cfg is not None

        def finite_scalar(name: str, *, nonnegative: bool = True) -> torch.Tensor:
            value = torch.as_tensor(state[name], device=self.device)
            if value.numel() != 1 or not value.is_floating_point():
                raise ValueError(f"HMI checkpoint {name} must be one floating scalar.")
            value = value.reshape(()).to(dtype=torch.float32)
            if not bool(torch.isfinite(value).item()):
                raise ValueError(f"HMI checkpoint {name} must be finite.")
            if nonnegative and float(value.item()) < 0.0:
                raise ValueError(f"HMI checkpoint {name} must be non-negative.")
            return value.clone()

        scale = finite_scalar("goal_noise_scale")
        ema = finite_scalar("goal_success_ema")
        success_sum = finite_scalar("goal_success_sum")
        count_raw = torch.as_tensor(state["goal_success_count"], device=self.device)
        if count_raw.numel() != 1 or count_raw.dtype == torch.bool or count_raw.is_floating_point():
            raise ValueError("HMI checkpoint goal_success_count must be one integer scalar.")
        success_count = count_raw.reshape(()).to(dtype=torch.long).clone()
        if int(success_count.item()) < 0:
            raise ValueError("HMI checkpoint goal_success_count must be non-negative.")
        if float(success_sum.item()) > float(success_count.item()):
            raise ValueError("HMI checkpoint success sum cannot exceed its count.")
        if not (
            float(self.hmi_cfg.goal_noise_min_scale)
            <= float(scale.item())
            <= float(self.hmi_cfg.goal_noise_max_scale)
        ):
            raise ValueError("HMI checkpoint goal-noise scale is outside configured bounds.")
        if not 0.0 <= float(ema.item()) <= 1.0:
            raise ValueError("HMI checkpoint success EMA must be in [0, 1].")
        ema_initialized = state["goal_success_ema_initialized"]
        if type(ema_initialized) is not bool:
            raise ValueError("HMI checkpoint EMA-initialized flag must be boolean.")
        if not ema_initialized and float(ema.item()) != 0.0:
            raise ValueError("Uninitialized HMI success EMA must be zero.")
        last_update = AdaptiveTimestepsSampler._checkpoint_integer_scalar(
            state["last_curriculum_update_iteration"],
            path="motion_command.hmi_state.last_curriculum_update_iteration",
            minimum=0,
        )
        return {
            "goal_noise_scale": scale,
            "goal_success_ema": ema,
            "goal_success_sum": success_sum,
            "goal_success_count": success_count,
            "goal_success_ema_initialized": ema_initialized,
            "last_curriculum_update_iteration": last_update,
        }

    def _process_checkpoint_state(
        self,
        state: dict[str, Any] | None,
        *,
        validate_only: bool,
    ) -> None:
        """Validate, then optionally commit, rank-local curriculum state."""
        if not state:
            return
        if not isinstance(state, dict):
            raise ValueError("Motion command checkpoint state must be a dictionary.")
        version = AdaptiveTimestepsSampler._checkpoint_integer_scalar(
            state.get("version", 0),
            path="motion_command.version",
        )
        if version not in (1, 2, 3):
            raise ValueError(f"Unsupported motion command checkpoint version: {state.get('version')!r}.")
        checkpoint_hmi_contract = state.get("hmi_contract")
        current_hmi_contract = self._hmi_checkpoint_contract()
        if checkpoint_hmi_contract != current_hmi_contract:
            raise ValueError(
                "Motion command checkpoint HMI contract differs from the current runtime: "
                f"checkpoint={checkpoint_hmi_contract!r}, current={current_hmi_contract!r}. "
                "Stage-1 to Stage-2 is a policy initialization, not an exact resume."
            )
        restored_hmi_state = self._prepare_hmi_checkpoint_state(state.get("hmi_state"))
        current_contact_prior_active, _ = self._contact_prior_runtime_flags()
        if version < 3 and current_contact_prior_active:
            raise ValueError(
                f"Motion command checkpoint version {version} predates online contact-prior state, "
                "but the current contact prior is active; exact resume is impossible."
            )
        raw_checkpoint_clip_ids = state.get("clip_ids", [])
        if not isinstance(raw_checkpoint_clip_ids, list) or not all(
            type(value) is str for value in raw_checkpoint_clip_ids
        ):
            raise ValueError("Motion command checkpoint clip_ids must be an ordered list of strings.")
        checkpoint_clip_ids = list(raw_checkpoint_clip_ids)
        if checkpoint_clip_ids != list(self.motion.clip_ids):
            raise ValueError(
                "Motion command checkpoint belongs to a different clip shard/order: "
                f"checkpoint={checkpoint_clip_ids}, current={list(self.motion.clip_ids)}."
            )
        if self.adaptive_timesteps_sampler is not None:
            checkpoint_valid_starts = self.adaptive_timesteps_sampler._checkpoint_integer_tensor(
                state.get("valid_start_counts"),
                path="motion_command.valid_start_counts",
            )
        else:
            raw_valid_starts = state.get("valid_start_counts")
            if raw_valid_starts is None:
                raise ValueError("Motion command checkpoint is missing valid_start_counts.")
            checkpoint_valid_starts = torch.as_tensor(raw_valid_starts, device=self.device)
            if (
                checkpoint_valid_starts.dtype == torch.bool
                or checkpoint_valid_starts.is_floating_point()
                or checkpoint_valid_starts.is_complex()
            ):
                raise ValueError(
                    "Motion command checkpoint valid_start_counts must use an integer dtype, "
                    f"got {checkpoint_valid_starts.dtype}."
                )
            checkpoint_valid_starts = checkpoint_valid_starts.to(dtype=torch.long).reshape(-1)
        current_valid_starts = self._valid_start_counts().to(dtype=torch.long)
        if not torch.equal(checkpoint_valid_starts, current_valid_starts):
            raise ValueError(
                "Motion command checkpoint valid reset geometry differs from the current configuration: "
                f"checkpoint={checkpoint_valid_starts.detach().cpu().tolist()}, "
                f"current={current_valid_starts.detach().cpu().tolist()}."
            )
        checkpoint_strategy = state.get("clip_weighting_strategy")
        if type(checkpoint_strategy) is not str:
            raise ValueError("Motion command checkpoint clip_weighting_strategy must be a string.")
        if checkpoint_strategy != str(self.clip_weighting_strategy):
            raise ValueError(
                "Motion command checkpoint clip weighting strategy differs from the current configuration: "
                f"checkpoint={checkpoint_strategy!r}, current={self.clip_weighting_strategy!r}."
            )
        raw_loss_weight = state.get("distributed_loss_weight", self.distributed_loss_weight)
        if isinstance(raw_loss_weight, bool) or not isinstance(raw_loss_weight, numbers.Real):
            raise ValueError(
                "Motion command checkpoint distributed_loss_weight must be a real number, "
                f"got {raw_loss_weight!r}."
            )
        checkpoint_loss_weight = float(raw_loss_weight)
        if not np.isfinite(checkpoint_loss_weight) or checkpoint_loss_weight <= 0.0:
            raise ValueError(
                "Motion command checkpoint distributed_loss_weight must be finite and positive, "
                f"got {checkpoint_loss_weight}."
            )
        if not np.isclose(checkpoint_loss_weight, self.distributed_loss_weight, rtol=1.0e-6, atol=1.0e-8):
            raise ValueError(
                "Motion command checkpoint rank-local loss weight differs from the current shard: "
                f"checkpoint={checkpoint_loss_weight}, current={self.distributed_loss_weight}."
            )

        sampler_state = state.get("adaptive_timesteps_sampler")
        if sampler_state is not None:
            if self.adaptive_timesteps_sampler is None:
                raise ValueError("Checkpoint contains adaptive timestep state, but adaptive sampling is disabled.")
            self.adaptive_timesteps_sampler.validate_state_dict(sampler_state)
        elif self.adaptive_timesteps_sampler is not None:
            raise ValueError(
                "Checkpoint is missing adaptive_timesteps_sampler state while adaptive sampling is enabled. "
                "A partial curriculum restore would change the resumed sampling distribution."
            )

        tensor_targets = {
            "clip_success_counts": self._clip_success_counts,
            "clip_total_counts": self._clip_total_counts,
            "raw_clip_sampling_weights": self._raw_clip_sampling_weights,
        }
        restored_targets: dict[str, torch.Tensor] = {}
        for key, target in tensor_targets.items():
            restored_value = state.get(key)
            if restored_value is None:
                if target is not None:
                    raise ValueError(
                        f"Checkpoint is missing enabled curriculum tensor {key!r}; refusing a partial restore."
                    )
                continue
            if target is None:
                raise ValueError(f"Checkpoint contains {key}, but it is not enabled in the current configuration.")
            restored = torch.as_tensor(restored_value, device=self.device)
            if restored.shape != target.shape:
                raise ValueError(
                    f"Motion command checkpoint {key} shape {tuple(restored.shape)} does not match "
                    f"current {tuple(target.shape)}."
                )
            if restored.dtype != target.dtype:
                raise ValueError(
                    f"Motion command checkpoint {key} dtype {restored.dtype} does not match "
                    f"current {target.dtype}."
                )
            if not bool(torch.isfinite(restored).all().item()):
                raise ValueError(f"Motion command checkpoint {key} contains NaN or infinity.")
            if bool((restored < 0.0).any().item()):
                raise ValueError(f"Motion command checkpoint {key} must be non-negative.")
            restored_targets[key] = restored.detach().clone()

        success = restored_targets.get("clip_success_counts")
        total = restored_targets.get("clip_total_counts")
        if success is not None and total is not None and bool((success > total).any().item()):
            raise ValueError(
                "Motion command checkpoint clip_success_counts cannot exceed clip_total_counts."
            )
        raw_weights = restored_targets.get("raw_clip_sampling_weights")
        if raw_weights is not None:
            weight_sum = float(raw_weights.sum().item())
            if weight_sum <= 0.0 or not np.isclose(weight_sum, 1.0, rtol=1.0e-5, atol=1.0e-6):
                raise ValueError(
                    "Motion command checkpoint raw_clip_sampling_weights must have positive unit sum, "
                    f"got {weight_sum}."
                )

        training_iteration = state.get("training_iteration")
        if training_iteration is not None:
            training_iteration = AdaptiveTimestepsSampler._checkpoint_integer_scalar(
                training_iteration,
                path="motion_command.training_iteration",
                minimum=0,
            )

        restored_contact_prior: dict[str, torch.Tensor] = {}
        if version >= 3:
            restored_contact_prior = self._prepare_contact_prior_checkpoint_state(state.get("contact_prior"))

        # Every component is now known to be compatible.  Apply only after the
        # complete payload has passed validation so a caught exception cannot
        # leave a partially restored curriculum.
        if validate_only:
            return
        if sampler_state is not None:
            self.adaptive_timesteps_sampler.load_state_dict(sampler_state)
        for key, restored in restored_targets.items():
            tensor_targets[key].copy_(restored)
        if training_iteration is not None:
            self._training_iteration = training_iteration
        if restored_hmi_state is not None:
            hmi_targets = {
                "goal_noise_scale": self.hmi_goal_noise_scale,
                "goal_success_ema": self.hmi_goal_success_ema,
                "goal_success_sum": self.hmi_goal_success_sum,
                "goal_success_count": self.hmi_goal_success_count,
            }
            for key, target in hmi_targets.items():
                if target is None:
                    raise RuntimeError(f"HMI runtime target {key} is not initialized.")
                target.copy_(restored_hmi_state[key])
            self.hmi_goal_success_ema_initialized = restored_hmi_state[
                "goal_success_ema_initialized"
            ]
            self.hmi_last_curriculum_update_iteration = restored_hmi_state[
                "last_curriculum_update_iteration"
            ]
        self._refresh_current_clip_sampling_weights()
        # Commit the six mutually constrained contact-prior buffers last.  All
        # potentially fallible validation and curriculum refresh work has
        # completed, so a rejected load cannot partially replace this group.
        contact_prior_targets = self._contact_prior_checkpoint_targets()
        for key, restored in restored_contact_prior.items():
            target = contact_prior_targets[key]
            assert isinstance(target, torch.Tensor)
            target.copy_(restored)

    def validate_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        """Validate curriculum state without changing the live command."""

        self._process_checkpoint_state(state, validate_only=True)

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        """Restore curriculum state after strict motion-bank identity validation."""

        self._process_checkpoint_state(state, validate_only=False)

    def _validate_motion_control_timebase(self) -> None:
        """Require one motion frame per environment control step.

        ``MotionCommand.step`` advances the discrete motion clock exactly once
        per control step.  A mismatched motion FPS would therefore replay the
        reference at the wrong physical speed and put AS/contact/prepend
        coordinates on different timebases.
        """

        def positive_finite_scalar(value: Any, *, name: str) -> float:
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(f"{name} must contain exactly one scalar, got shape {tuple(value.shape)}.")
                value = value.detach().item()
            elif isinstance(value, np.ndarray):
                if value.size != 1:
                    raise ValueError(f"{name} must contain exactly one scalar, got shape {value.shape}.")
                value = value.reshape(-1)[0]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
                raise ValueError(f"{name} must be a finite positive real scalar, got {value!r}.")
            parsed = float(value)
            if not np.isfinite(parsed) or parsed <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {parsed!r}.")
            return parsed

        motion_fps = positive_finite_scalar(self.motion.fps, name="motion.fps")
        control_dt = positive_finite_scalar(self._env.dt, name="env.dt")
        control_fps = 1.0 / control_dt
        if not np.isfinite(control_fps):
            raise ValueError(
                f"Environment control frequency 1/env.dt must be finite, got env.dt={control_dt!r}."
            )
        if not np.isclose(motion_fps, control_fps, rtol=1.0e-6, atol=1.0e-6):
            raise ValueError(
                "Motion FPS must match the environment control frequency because MotionCommand advances "
                "one motion frame per control step: "
                f"motion.fps={motion_fps}, env.dt={control_dt}, control_fps={control_fps}. "
                "Resample the motion bank or configure simulator fps/control_decimation to match."
            )

    def _build_hybrid_stage2_task_env_mask(self) -> torch.Tensor:
        """Build a deterministic task assignment, stratified by fixed clip when available."""

        enabled = bool(getattr(self.motion_cfg, "hybrid_stage2_enabled", False))
        mask = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        if not enabled:
            return mask

        fraction = float(getattr(self.motion_cfg, "hybrid_stage2_task_env_fraction", 0.5))

        def assign_evenly(env_ids: torch.Tensor) -> int:
            group_size = int(env_ids.numel())
            group_task_count = min(max(int(round(group_size * fraction)), 0), group_size)
            if group_task_count == 0:
                return 0
            if group_task_count == group_size:
                mask[env_ids] = True
                return group_task_count
            row_ids = torch.arange(group_size, device=self.device, dtype=torch.long)
            selected = ((row_ids + 1) * group_task_count // group_size) > (
                row_ids * group_task_count // group_size
            )
            mask[env_ids[selected]] = True
            return group_task_count

        # With object banks, envs are pinned to compatible clips. Stratifying
        # here prevents the alternating 50/50 assignment from becoming
        # perfectly correlated with a two-clip round-robin (one whole clip in
        # task mode and the other in tracking mode).
        fixed_clip_ids = getattr(self, "_fixed_clip_ids", None)
        task_count = 0
        if isinstance(fixed_clip_ids, torch.Tensor) and fixed_clip_ids.shape == (self.num_envs,):
            for clip_id in torch.unique(fixed_clip_ids, sorted=True):
                env_ids = torch.nonzero(fixed_clip_ids == clip_id, as_tuple=False).squeeze(-1)
                task_count += assign_evenly(env_ids)
        else:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            task_count = assign_evenly(env_ids)

        logger.info(
            "Enabled hybrid stage-2 assignment: task_envs={}/{} ({:.3f}), "
            "tracking_envs={}, forward_command_m={:.3f}, stratified_by_fixed_clip={}.",
            task_count,
            self.num_envs,
            task_count / float(self.num_envs),
            self.num_envs - task_count,
            float(getattr(self.motion_cfg, "hybrid_stage2_forward_command_m", 0.15)),
            isinstance(fixed_clip_ids, torch.Tensor)
            and fixed_clip_ids.shape == (self.num_envs,),
        )
        return mask

    def hybrid_velocity_enabled(self) -> bool:
        return bool(getattr(self.motion_cfg, "hybrid_velocity_enabled", False))

    def _validate_hybrid_velocity_config(self) -> None:
        if not self.hybrid_velocity_enabled():
            return
        if bool(getattr(self.motion_cfg, "hybrid_stage2_enabled", False)):
            raise ValueError("hybrid_velocity and hybrid_stage2 are mutually exclusive.")
        if self.pure_rl_policy_command_after_lift_enabled():
            raise ValueError(
                "hybrid_velocity and pure_rl_policy_command_after_lift are mutually exclusive."
            )
        if not self.motion.has_object:
            raise ValueError("hybrid_velocity requires motion clips with object trajectories.")

        command_frame = str(
            getattr(self.motion_cfg, "hybrid_velocity_command_frame", "heading")
        ).strip().lower()
        if command_frame not in {"heading", "world"}:
            raise ValueError(
                "hybrid_velocity_command_frame must be 'heading' or 'world', "
                f"got {command_frame!r}."
            )

        start_fraction = float(self.motion_cfg.hybrid_velocity_task_env_fraction_start)
        end_fraction = float(self.motion_cfg.hybrid_velocity_task_env_fraction_end)
        start_iter = int(self.motion_cfg.hybrid_velocity_task_env_fraction_start_iter)
        end_iter = int(self.motion_cfg.hybrid_velocity_task_env_fraction_end_iter)
        if end_fraction < start_fraction:
            raise ValueError(
                "hybrid_velocity task fraction must be monotonic: "
                f"start={start_fraction}, end={end_fraction}."
            )
        if end_fraction != start_fraction and end_iter <= start_iter:
            raise ValueError(
                "hybrid_velocity task-fraction ramp requires end_iter > start_iter: "
                f"start_iter={start_iter}, end_iter={end_iter}."
            )

    def _current_hybrid_velocity_task_env_fraction(self) -> float:
        if not self.hybrid_velocity_enabled():
            return 0.0
        return self._scheduled_reset_prob(
            float(self.motion_cfg.hybrid_velocity_task_env_fraction_start),
            end_value=float(self.motion_cfg.hybrid_velocity_task_env_fraction_end),
            start_iter=int(self.motion_cfg.hybrid_velocity_task_env_fraction_start_iter),
            end_iter=int(self.motion_cfg.hybrid_velocity_task_env_fraction_end_iter),
        )

    def _configure_hybrid_velocity_task_assignment(self) -> None:
        """Create a monotonic, RNG-free task priority within every fixed clip."""

        self._validate_hybrid_velocity_config()
        if not self.hybrid_velocity_enabled():
            if self.hybrid_velocity_task_env_mask is not None:
                self.hybrid_velocity_task_env_mask.zero_()
            return

        priority = torch.ones((self.num_envs,), device=self.device, dtype=torch.float32)
        fixed_clip_ids = getattr(self, "_fixed_clip_ids", None)
        if isinstance(fixed_clip_ids, torch.Tensor) and fixed_clip_ids.shape == (self.num_envs,):
            groups = [
                torch.nonzero(fixed_clip_ids == clip_id, as_tuple=False).squeeze(-1)
                for clip_id in torch.unique(fixed_clip_ids, sorted=True)
            ]
        else:
            groups = [torch.arange(self.num_envs, device=self.device, dtype=torch.long)]

        # Golden-ratio ordering gives each monotonic prefix an evenly spread
        # subset without consuming or perturbing the training RNG stream.
        golden_ratio_conjugate = 0.6180339887498949
        for env_ids in groups:
            group_size = int(env_ids.numel())
            if group_size == 0:
                continue
            row_ids = torch.arange(group_size, device=self.device, dtype=torch.float64)
            keys = torch.remainder((row_ids + 1.0) * golden_ratio_conjugate, 1.0)
            order = torch.argsort(keys, stable=True)
            ranks = torch.empty_like(order)
            ranks[order] = torch.arange(group_size, device=self.device, dtype=torch.long)
            priority[env_ids] = (
                (ranks.to(dtype=torch.float32) + 0.5) / float(group_size)
            )

        self._hybrid_velocity_task_priority = priority
        self._refresh_hybrid_velocity_task_env_mask()
        active_count = int(self.get_hybrid_velocity_task_env_mask().sum().item())
        logger.info(
            "Enabled hybrid velocity assignment: active_task_envs={}/{} ({:.3f}), "
            "fraction_schedule={:.3f}->{:.3f} over iterations {}->{}, "
            "forward_command_mps={:.3f}, lift_height_m={:.3f}, "
            "mode_changes_only_on_reset=True, stratified_by_fixed_clip={}.",
            active_count,
            self.num_envs,
            active_count / float(self.num_envs),
            float(self.motion_cfg.hybrid_velocity_task_env_fraction_start),
            float(self.motion_cfg.hybrid_velocity_task_env_fraction_end),
            int(self.motion_cfg.hybrid_velocity_task_env_fraction_start_iter),
            int(self.motion_cfg.hybrid_velocity_task_env_fraction_end_iter),
            float(self.motion_cfg.hybrid_velocity_forward_command_mps),
            float(self.motion_cfg.hybrid_velocity_lift_height_m),
            isinstance(fixed_clip_ids, torch.Tensor)
            and fixed_clip_ids.shape == (self.num_envs,),
        )

    def _refresh_hybrid_velocity_task_env_mask(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Apply the current curriculum only to rows beginning a new episode."""

        if not self.hybrid_velocity_enabled():
            return
        if self.hybrid_velocity_task_env_mask is None or self._hybrid_velocity_task_priority is None:
            raise RuntimeError("hybrid_velocity task assignment was not configured.")
        selected_env_ids = self._ensure_index_tensor(env_ids)
        fraction = self._current_hybrid_velocity_task_env_fraction()
        desired = self._hybrid_velocity_task_priority[selected_env_ids] <= fraction
        self.hybrid_velocity_task_env_mask[selected_env_ids] = desired

    def get_hybrid_velocity_task_env_mask(self) -> torch.Tensor:
        mask = self.hybrid_velocity_task_env_mask
        if mask is None:
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        return mask

    def get_hybrid_velocity_task_active_mask(self) -> torch.Tensor:
        task_mask = self.get_hybrid_velocity_task_env_mask()
        if self.pickup_anchor_set is None:
            return torch.zeros_like(task_mask)
        return task_mask & self.pickup_anchor_set

    def get_hybrid_velocity_tracking_reward_mask(self) -> torch.Tensor:
        return ~self.get_hybrid_velocity_task_env_mask()

    def get_hybrid_velocity_command(self) -> torch.Tensor:
        """Return the unified ``[vx, vy, yaw_rate]`` actor/critic command."""

        if not self.hybrid_velocity_enabled():
            raise RuntimeError("hybrid velocity command requires hybrid_velocity_enabled=True.")

        command_frame = str(
            getattr(self.motion_cfg, "hybrid_velocity_command_frame", "heading")
        ).strip().lower()
        if getattr(self, "manual_control_enabled", False):
            expected_semantics = (
                "robot_heading_velocity_mps"
                if command_frame == "heading"
                else "world_velocity_mps"
            )
            actual_semantics = getattr(
                self,
                "_manual_forward_after_lift_command_semantics",
                "legacy_constant_robot_heading_frame",
            )
            if actual_semantics != expected_semantics:
                raise RuntimeError(
                    "Manual hybrid-velocity evaluation command semantics mismatch: "
                    f"actual={actual_semantics!r}, expected={expected_semantics!r}."
                )
            if self.manual_xy_rel is None or self.manual_yaw_rel is None:
                raise RuntimeError("Manual hybrid-velocity command tensors are not initialized.")
            return torch.cat((self.manual_xy_rel, self.manual_yaw_rel), dim=-1)
        if command_frame == "heading":
            heading_inv = calc_heading_quat_inv(self.robot_root_quat_w, w_last=True)
            reference_velocity_xy = quat_apply(
                heading_inv,
                self.root_lin_vel_w,
                w_last=True,
            )[:, :2]
        elif command_frame == "world":
            reference_velocity_xy = self.root_lin_vel_w[:, :2]
        else:
            raise RuntimeError(
                "hybrid_velocity_command_frame was not validated: "
                f"{command_frame!r}."
            )
        tracking_command = torch.stack(
            (
                reference_velocity_xy[:, 0],
                reference_velocity_xy[:, 1],
                self.root_ang_vel_w[:, 2],
            ),
            dim=-1,
        )

        task_command = torch.zeros_like(tracking_command)
        task_active = self.get_hybrid_velocity_task_active_mask()
        task_command[:, 0] = torch.where(
            task_active,
            torch.full_like(
                task_command[:, 0],
                float(self.motion_cfg.hybrid_velocity_forward_command_mps),
            ),
            torch.zeros_like(task_command[:, 0]),
        )
        task_mask = self.get_hybrid_velocity_task_env_mask()
        return torch.where(task_mask.unsqueeze(-1), task_command, tracking_command)

    def pure_rl_policy_command_after_lift_enabled(self) -> bool:
        return bool(
            getattr(
                self.motion_cfg,
                "pure_rl_policy_command_after_lift_enabled",
                False,
            )
        )

    def precomputed_turn_then_forward_enabled(self) -> bool:
        mode = str(
            getattr(
                self.motion_cfg,
                "contact_aware_sparse_root_command_mode",
                "tracking_error",
            )
        ).strip().lower().replace("-", "_")
        return mode == "precomputed_turn_then_forward"

    def _validate_precomputed_turn_then_forward_config(self) -> None:
        if not self.precomputed_turn_then_forward_enabled():
            return
        conflicting_modes = {
            "pure_rl_policy_command_after_lift": self.pure_rl_policy_command_after_lift_enabled(),
            "hybrid_stage2": bool(getattr(self.motion_cfg, "hybrid_stage2_enabled", False)),
            "hybrid_velocity": self.hybrid_velocity_enabled(),
        }
        enabled_conflicts = [name for name, enabled in conflicting_modes.items() if enabled]
        if enabled_conflicts:
            raise ValueError(
                "precomputed_turn_then_forward is an exclusive actor-command mode; "
                f"disable {enabled_conflicts}."
            )
        if not self.motion.has_object:
            raise ValueError("precomputed_turn_then_forward requires object motion clips.")
        if not self.motion.has_precomputed_root_command:
            raise ValueError(
                "precomputed_turn_then_forward requires every selected motion NPZ to contain "
                "policy_command_xy_yaw and policy_command_phase."
            )
        command = self.motion.precomputed_root_command
        phase = self.motion.precomputed_root_command_phase
        if command.shape != (self.motion.time_step_total, 3) or phase.shape != (
            self.motion.time_step_total,
        ):
            raise ValueError(
                "Precomputed command tensors do not align with the loaded motion timeline: "
                f"command={tuple(command.shape)}, phase={tuple(phase.shape)}, "
                f"motion_steps={self.motion.time_step_total}."
            )
        phase_counts = torch.bincount(phase.to(dtype=torch.long), minlength=3)
        logger.info(
            "Enabled immutable precomputed turn-then-forward actor command: "
            "zero_frames={}, forward_frames={}, yaw_frames={}, dy_always_zero=True, "
            "dx_dyaw_overlap=False, runtime_pickup_latch=True, rewards_and_terminations_unchanged=True.",
            int(phase_counts[0].item()),
            int(phase_counts[1].item()),
            int(phase_counts[2].item()),
        )

    def get_precomputed_turn_then_forward_command(self) -> torch.Tensor:
        if not self.precomputed_turn_then_forward_enabled():
            raise RuntimeError(
                "Precomputed turn-then-forward command requested while its command mode is disabled."
            )
        motion_indices = self._get_motion_indices(self.time_steps)
        command = self.motion.precomputed_root_command.index_select(0, motion_indices)
        if self.pickup_anchor_set is None:
            return torch.zeros_like(command)
        return torch.where(
            self.pickup_anchor_set.unsqueeze(-1),
            command,
            torch.zeros_like(command),
        )

    def get_precomputed_turn_then_forward_phase(self) -> torch.Tensor:
        if not self.precomputed_turn_then_forward_enabled():
            raise RuntimeError(
                "Precomputed turn-then-forward phase requested while its command mode is disabled."
            )
        motion_indices = self._get_motion_indices(self.time_steps)
        phase = self.motion.precomputed_root_command_phase.index_select(0, motion_indices)
        if self.pickup_anchor_set is None:
            return torch.zeros_like(phase)
        return torch.where(self.pickup_anchor_set, phase, torch.zeros_like(phase))

    def get_pure_rl_policy_command_after_lift_active_mask(self) -> torch.Tensor:
        if not self.pure_rl_policy_command_after_lift_enabled():
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        if self.pickup_anchor_set is None:
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        return self.pickup_anchor_set

    def get_hybrid_stage2_task_env_mask(self) -> torch.Tensor:
        mask = self.hybrid_stage2_task_env_mask
        if mask is None:
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        return mask

    def get_hybrid_stage2_task_active_mask(self) -> torch.Tensor:
        task_mask = self.get_hybrid_stage2_task_env_mask()
        if self.pickup_anchor_set is None:
            return torch.zeros_like(task_mask)
        return task_mask & self.pickup_anchor_set

    def get_hybrid_stage2_tracking_reward_mask(self) -> torch.Tensor:
        """Keep imitation for tracking envs and for task envs before pickup."""

        return ~self.get_hybrid_stage2_task_active_mask()

    def hmi_enabled(self) -> bool:
        return getattr(self, "hmi_cfg", None) is not None

    def _configure_hmi_partition(self) -> None:
        """Create HMI's fixed track/gen environment identity."""

        self.hmi_cfg = self.motion_cfg.hmi
        if self.hmi_cfg is None:
            self.hmi_track_env_mask = torch.ones(
                (self.num_envs,), device=self.device, dtype=torch.bool
            )
            self.hmi_gen_env_mask = torch.zeros_like(self.hmi_track_env_mask)
            return
        if not self.motion.has_object:
            raise ValueError("HMI object training requires motion clips with object trajectories.")
        incompatible = {
            "hybrid_stage2": bool(getattr(self.motion_cfg, "hybrid_stage2_enabled", False)),
            "hybrid_velocity": self.hybrid_velocity_enabled(),
            "pure_rl_policy_command_after_lift": self.pure_rl_policy_command_after_lift_enabled(),
        }
        enabled_incompatible = [name for name, enabled in incompatible.items() if enabled]
        if enabled_incompatible:
            raise ValueError(
                "motion_config.hmi is mutually exclusive with "
                + ", ".join(enabled_incompatible)
                + "."
            )
        track_mask = build_fixed_hmi_track_mask(
            self.num_envs,
            float(self.hmi_cfg.track_ratio),
            int(self.hmi_cfg.env_partition_seed),
        )
        self.hmi_track_env_mask = track_mask.to(device=self.device)
        self.hmi_gen_env_mask = ~self.hmi_track_env_mask
        logger.info(
            "Enabled HMI fixed partition: track_envs={}/{} ({:.3f}), gen_envs={}, seed={}, "
            "actor_reference_leakage=False.",
            int(self.hmi_track_env_mask.sum().item()),
            self.num_envs,
            float(self.hmi_track_env_mask.float().mean().item()),
            int(self.hmi_gen_env_mask.sum().item()),
            int(self.hmi_cfg.env_partition_seed),
        )

    def get_hmi_track_env_mask(self) -> torch.Tensor:
        if not self.hmi_enabled() or self.hmi_track_env_mask is None:
            raise RuntimeError("HMI track mask requested without motion_config.hmi.")
        return self.hmi_track_env_mask

    def get_hmi_gen_env_mask(self) -> torch.Tensor:
        if not self.hmi_enabled() or self.hmi_gen_env_mask is None:
            raise RuntimeError("HMI generation mask requested without motion_config.hmi.")
        return self.hmi_gen_env_mask

    def _hmi_goal_noise_axis_scales(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hmi_goal_noise_scale is None:
            raise RuntimeError("HMI goal-noise scale requested before buffer initialization.")
        scale = self.hmi_goal_noise_scale.to(device=self.device, dtype=torch.float32)
        one = torch.ones((), device=self.device, dtype=torch.float32)
        return torch.stack((scale, scale, one)), torch.stack((one, one, scale))

    def _record_hmi_completed_goal_outcomes(self, env_ids: torch.Tensor) -> None:
        """Accumulate one binary outcome for each completed generation episode."""

        if not self.hmi_enabled():
            return
        if (
            self.hmi_goal_reached is None
            or self.hmi_goal_success_sum is None
            or self.hmi_goal_success_count is None
        ):
            raise RuntimeError("HMI success buffers are not initialized.")
        previous_action = self._base_reset_has_previous_action_mask(env_ids)
        completed_gen = self.get_hmi_gen_env_mask()[env_ids] & previous_action
        if torch.any(completed_gen):
            completed_ids = env_ids[completed_gen]
            self.hmi_goal_success_sum.add_(
                self.hmi_goal_reached[completed_ids].to(dtype=torch.float32).sum()
            )
            self.hmi_goal_success_count.add_(completed_ids.numel())
        self.hmi_goal_reached[env_ids] = False

    def mark_hmi_goal_reached(self, env_ids: torch.Tensor) -> None:
        if not self.hmi_enabled() or self.hmi_goal_reached is None:
            raise RuntimeError("HMI goal success reported without initialized HMI buffers.")
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        self.hmi_goal_reached[ids] = True

    def _reapply_hmi_gen_goal_noise(self) -> None:
        if not self.hmi_enabled():
            return
        gen_env_ids = torch.where(self.get_hmi_gen_env_mask())[0]
        if gen_env_ids.numel() == 0:
            return
        if (
            self.hmi_exact_goal_object_pos_w is None
            or self.hmi_exact_goal_object_quat_w is None
            or self.hmi_goal_object_pos_w is None
            or self.hmi_goal_object_quat_w is None
            or self.hmi_goal_version is None
        ):
            raise RuntimeError("HMI goal buffers are not initialized.")
        assert self.hmi_cfg is not None
        pos_noise, quat_noise = self._sample_hmi_goal_noise(
            int(gen_env_ids.numel()), self.hmi_cfg.object_goal_noise
        )
        self.hmi_goal_object_pos_w[gen_env_ids] = (
            self.hmi_exact_goal_object_pos_w[gen_env_ids] + pos_noise
        )
        self.hmi_goal_object_quat_w[gen_env_ids] = quat_normalize(
            quat_mul(
                quat_noise,
                self.hmi_exact_goal_object_quat_w[gen_env_ids],
                w_last=True,
            )
        )
        self.hmi_goal_version[gen_env_ids] += 1
        if self.hmi_goal_reached is not None:
            self.hmi_goal_reached[gen_env_ids] = False

    def _update_hmi_goal_noise_curriculum(self, iteration: int) -> None:
        """Update the upstream HMI goal curriculum at a PPO boundary.

        Sufficient statistics are reduced globally before the shared objective
        changes, so all ranks keep the same goal distribution.
        """

        if not self.hmi_enabled() or iteration <= 0:
            return
        assert self.hmi_cfg is not None
        interval = int(self.hmi_cfg.goal_noise_update_interval)
        if (
            iteration % interval != 0
            or iteration <= int(self.hmi_last_curriculum_update_iteration)
        ):
            return
        if (
            self.hmi_goal_success_sum is None
            or self.hmi_goal_success_count is None
            or self.hmi_goal_success_ema is None
            or self.hmi_goal_noise_scale is None
        ):
            raise RuntimeError("HMI curriculum buffers are not initialized.")

        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        collective_device = self.device
        if distributed and str(torch.distributed.get_backend()).lower() != "nccl":
            collective_device = torch.device("cpu")
        sufficient_stats = torch.tensor(
            [
                float(self.hmi_goal_success_sum.item()),
                float(self.hmi_goal_success_count.item()),
            ],
            device=collective_device,
            dtype=torch.float64,
        )
        if distributed:
            torch.distributed.all_reduce(sufficient_stats, op=torch.distributed.ReduceOp.SUM)
        success_sum = sufficient_stats[0]
        success_count = sufficient_stats[1]
        if float(success_count.item()) > 0.0:
            batch_success = success_sum / success_count
            alpha = float(self.hmi_cfg.goal_noise_ema_alpha)
            if self.hmi_goal_success_ema_initialized:
                self.hmi_goal_success_ema.copy_(
                    (1.0 - alpha) * self.hmi_goal_success_ema
                    + alpha * batch_success.to(dtype=torch.float32)
                )
            else:
                self.hmi_goal_success_ema.copy_(batch_success.to(dtype=torch.float32))
                self.hmi_goal_success_ema_initialized = True

            old_scale = self.hmi_goal_noise_scale.clone()
            ema = float(self.hmi_goal_success_ema.item())
            if ema > float(self.hmi_cfg.goal_noise_success_threshold_up):
                self.hmi_goal_noise_scale.add_(float(self.hmi_cfg.goal_noise_scale_step))
            elif ema < float(self.hmi_cfg.goal_noise_success_threshold_down):
                self.hmi_goal_noise_scale.sub_(float(self.hmi_cfg.goal_noise_scale_step))
            self.hmi_goal_noise_scale.clamp_(
                min=float(self.hmi_cfg.goal_noise_min_scale),
                max=float(self.hmi_cfg.goal_noise_max_scale),
            )
            if not torch.equal(old_scale, self.hmi_goal_noise_scale):
                self._reapply_hmi_gen_goal_noise()

        self.hmi_goal_success_sum.zero_()
        self.hmi_goal_success_count.zero_()
        self.hmi_last_curriculum_update_iteration = int(iteration)

    def _sample_hmi_goal_noise(
        self,
        count: int,
        cfg: HMIGoalPoseNoiseConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_axis_scale, rpy_axis_scale = self._hmi_goal_noise_axis_scales()

        def clipped(
            std_values: list[float],
            clip_values: list[float],
            axis_scale: torch.Tensor,
        ) -> torch.Tensor:
            std = (
                torch.tensor(std_values, device=self.device, dtype=torch.float32)
                * axis_scale
            ).unsqueeze(0)
            clip = (
                torch.tensor(clip_values, device=self.device, dtype=torch.float32)
                * axis_scale
            ).unsqueeze(0)
            values = torch.randn((count, 3), device=self.device, dtype=torch.float32) * std
            return torch.clamp(values, min=-clip, max=clip)

        pos_noise = clipped(cfg.pos_std_xyz, cfg.pos_clip_xyz, pos_axis_scale)
        rpy_noise = clipped(cfg.rpy_std, cfg.rpy_clip, rpy_axis_scale)
        quat_noise = quat_from_euler_xyz(
            rpy_noise[:, 0], rpy_noise[:, 1], rpy_noise[:, 2]
        )
        return pos_noise, quat_normalize(quat_noise)

    def _refresh_hmi_terminal_object_goals(self, env_ids: torch.Tensor) -> None:
        """Bind each reset row to its active clip's terminal object pose."""

        if not self.hmi_enabled():
            return
        if (
            self.hmi_exact_goal_object_pos_w is None
            or self.hmi_exact_goal_object_quat_w is None
            or self.hmi_goal_object_pos_w is None
            or self.hmi_goal_object_quat_w is None
            or self.hmi_goal_version is None
        ):
            raise RuntimeError("HMI goal buffers are not initialized.")
        assert self.hmi_cfg is not None
        clip_ids = self.clip_ids[env_ids]
        final_indices = (
            self.motion.clip_offsets[clip_ids]
            + self.motion.clip_lengths[clip_ids]
            - 1
        )
        exact_pos = self.motion.object_pos_w[final_indices]
        exact_quat = self.motion.object_quat_w[final_indices]
        if self.motion_cfg.align_motion_to_init_yaw:
            exact_pos = self._apply_motion_alignment_pos(exact_pos, env_ids)
            exact_quat = self._apply_motion_alignment_quat(exact_quat, env_ids)
        else:
            exact_pos = exact_pos + self._get_env_offsets(env_ids)

        self.hmi_exact_goal_object_pos_w[env_ids] = exact_pos
        self.hmi_exact_goal_object_quat_w[env_ids] = exact_quat
        self.hmi_goal_object_pos_w[env_ids] = exact_pos
        self.hmi_goal_object_quat_w[env_ids] = exact_quat

        gen_env_ids = env_ids[self.get_hmi_gen_env_mask()[env_ids]]
        if gen_env_ids.numel() > 0:
            pos_noise, quat_noise = self._sample_hmi_goal_noise(
                int(gen_env_ids.numel()), self.hmi_cfg.object_goal_noise
            )
            self.hmi_goal_object_pos_w[gen_env_ids] = (
                self.hmi_exact_goal_object_pos_w[gen_env_ids] + pos_noise
            )
            self.hmi_goal_object_quat_w[gen_env_ids] = quat_normalize(
                quat_mul(
                    quat_noise,
                    self.hmi_exact_goal_object_quat_w[gen_env_ids],
                    w_last=True,
                )
            )
        self.hmi_goal_version[env_ids] += 1

    def get_hmi_object_goal_command(self) -> torch.Tensor:
        """Return terminal object ``[x, y, yaw]`` in the current robot heading frame."""

        if not self.hmi_enabled() or self.hmi_goal_object_pos_w is None or self.hmi_goal_object_quat_w is None:
            raise RuntimeError("HMI goal command requested before HMI goal initialization.")
        heading_inv = calc_heading_quat_inv(self.robot_ref_quat_w, w_last=True)
        goal_delta_w = self.hmi_goal_object_pos_w - self.robot_ref_pos_w
        goal_delta_heading = quat_apply(heading_inv, goal_delta_w, w_last=True)
        goal_yaw = normalize_angle(
            calc_heading(self.hmi_goal_object_quat_w)
            - calc_heading(self.robot_ref_quat_w)
        )
        return torch.cat((goal_delta_heading[:, :2], goal_yaw.unsqueeze(-1)), dim=-1)

    def _apply_hmi_step_zero_root_noise(
        self,
        env_ids: torch.Tensor,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        target_root_pos: torch.Tensor,
        target_root_rot: torch.Tensor,
    ) -> None:
        """Override gen rows at clip frame zero with HMI's clipped Gaussian noise."""

        if not self.hmi_enabled():
            return
        assert self.hmi_cfg is not None
        local_mask = self.get_hmi_gen_env_mask()[env_ids] & (self.time_steps[env_ids] == 0)
        count = int(local_mask.sum().item())
        if count == 0:
            return

        def clipped(std_values: list[float], clip_values: list[float]) -> torch.Tensor:
            std = torch.tensor(std_values, device=self.device, dtype=torch.float32).unsqueeze(0)
            clip = torch.tensor(clip_values, device=self.device, dtype=torch.float32).unsqueeze(0)
            values = torch.randn((count, 3), device=self.device, dtype=torch.float32) * std
            return torch.clamp(values, min=-clip, max=clip)

        pos_noise = clipped(
            self.hmi_cfg.gen_step_zero_root_pos_std_xyz,
            self.hmi_cfg.gen_step_zero_root_pos_clip_xyz,
        )
        rpy_noise = clipped(
            self.hmi_cfg.gen_step_zero_root_rpy_std,
            self.hmi_cfg.gen_step_zero_root_rpy_clip,
        )
        quat_noise = quat_from_euler_xyz(
            rpy_noise[:, 0], rpy_noise[:, 1], rpy_noise[:, 2]
        )
        target_root_pos[local_mask] = root_pos[local_mask] + pos_noise
        target_root_rot[local_mask] = quat_normalize(
            quat_mul(quat_noise, root_rot[local_mask], w_last=True)
        )

    def setup(self) -> None:
        self._validate_reset_sampling_curriculum_config(self.motion_cfg)
        self.num_envs = self._env.num_envs
        self.device = self._env.device
        self._initialize_uniform_t1_window_metric_state()
        self.manual_control_enabled = False
        self.manual_xy_rel = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.manual_yaw_rel = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        self.manual_pickup_button_override_enabled = False
        self.manual_pickup_button = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        self.manual_drop_button_override_enabled = False
        self.manual_drop_button = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        self._manual_forward_after_lift_enabled = False
        self._manual_forward_after_lift_command_m = 0.0
        self._manual_forward_after_lift_rel_z_delta_m = 0.0
        self._manual_forward_after_lift_consecutive_steps = 0
        self._manual_forward_after_lift_preserve_native_contact_buttons = False
        self._manual_forward_after_lift_preserve_native_pickup_button = False
        self._manual_forward_after_lift_preserve_native_drop_button = False
        self._manual_forward_after_lift_baseline_object_z = None
        self._manual_forward_after_lift_consecutive_count = None
        self._manual_forward_after_lift_triggered = None
        self._manual_forward_after_lift_trigger_episode_step = None
        self._manual_forward_heading_lock_enabled = False
        self._manual_forward_heading_lock_command_m = 0.0
        self._manual_forward_heading_lock_active = None
        self._manual_forward_heading_lock_origin_xy_w = None
        self._manual_forward_heading_lock_yaw_w = None
        self.manual_object_reset_enabled = False
        self.manual_object_reset_pos_offset_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.manual_object_reset_rpy_offset = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._training_iteration = 0
        self._training_total_iterations = None
        self._clean_noisy_clip_curriculum_cfg = self.motion_cfg.clean_noisy_clip_curriculum
        self._clean_noisy_clip_curriculum_enabled = bool(self._clean_noisy_clip_curriculum_cfg.enabled)
        self._clean_clip_mask = None
        self._noisy_clip_mask = None
        self._fixed_clip_group_assignment_cfg = self.motion_cfg.fixed_clip_group_assignment
        self._fixed_clip_group_env_mask = None
        self._fixed_clip_group_clip_mask = None
        self._fixed_clip_complement_clip_mask = None
        self.hybrid_stage2_task_env_mask = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self.hybrid_velocity_task_env_mask = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self._hybrid_velocity_task_priority = torch.ones(
            (self.num_envs,), device=self.device, dtype=torch.float32
        )
        self.pickup_anchor_set = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.pickup_anchor_root_pos_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.pickup_anchor_root_quat_w = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self.pickup_anchor_root_quat_w[:, 3] = 1.0
        self.pickup_anchor_object_pos_b = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )
        self.pickup_anchor_object_quat_b = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=torch.float32
        )
        self.pickup_anchor_object_quat_b[:, 3] = 1.0
        self.pickup_object_rel_z_baseline = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.hybrid_velocity_object_z_baseline = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.float32
        )
        self.pickup_consecutive_counter = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)

        init_state = self._env.robot_config.init_state
        reset_to_default_pose_env = os.environ.get("HOLOSOMA_RESET_TO_DEFAULT_POSE")
        if reset_to_default_pose_env is None:
            reset_to_default_pose_env = os.environ.get("HOLOSOMA_DEFAULT_POSE_INIT", "0")
        self._reset_to_default_pose = reset_to_default_pose_env.lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        disable_clip_end_reset_env = os.environ.get("HOLOSOMA_DISABLE_CLIP_END_RESET", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        disable_auto_reset_env = os.environ.get("HOLOSOMA_DISABLE_AUTO_RESET", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._disable_clip_end_reset = bool(disable_clip_end_reset_env or disable_auto_reset_env)
        if self._reset_to_default_pose:
            start_probs = [float(self.motion_cfg.start_at_timestep_zero_prob)]
            if self.motion_cfg.start_at_timestep_zero_prob_end is not None:
                start_probs.append(float(self.motion_cfg.start_at_timestep_zero_prob_end))
            if any(prob < 0.999 for prob in start_probs):
                logger.warning(
                    "reset_to_default_pose=True applies to every reset, including non-zero motion starts. "
                    "This can make random clip starts much harder than runtime prepend alone."
                )
        self._init_root_pos = torch.tensor(init_state.pos, dtype=torch.float32, device=self.device)
        self._init_root_rot = torch.tensor(init_state.rot, dtype=torch.float32, device=self.device)
        self._init_root_lin_vel = torch.tensor(init_state.lin_vel, dtype=torch.float32, device=self.device)
        self._init_root_ang_vel = torch.tensor(init_state.ang_vel, dtype=torch.float32, device=self.device)
        init_root_quat = torch.tensor(init_state.rot, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, _, init_yaw = get_euler_xyz(init_root_quat, w_last=True)
        self._init_root_yaw = init_yaw.squeeze(0)

        robot_body_names = self._env.simulator._body_list  # type: ignore[attr-defined]
        robot_body_names_alias = [FAKE_BODY_NAME_ALIASES.get(bn, bn) for bn in robot_body_names]

        robot_joint_names = self._env.simulator.dof_names  # type: ignore[attr-defined]

        # 1. load motion data
        self.motion: MotionLoader = MotionLoader(
            self.motion_cfg.motion_file,
            robot_body_names_alias,
            robot_joint_names,
            device=self.device,
            motion_clip_id=self.motion_cfg.motion_clip_id,
            motion_clip_name=self.motion_cfg.motion_clip_name,
            object_size_scale=self.motion_cfg.object_size_scale,
            allowed_object_categories=self.motion_cfg.allowed_object_categories,
        )
        self._validate_precomputed_turn_then_forward_config()
        self._validate_kinematic_button_motion_object()
        self._validate_motion_control_timebase()
        # Rank-local AS shards may intentionally contain only one clip even
        # though they are a partition (or duplicate-balanced cover) of a
        # global multi-clip bank.  Load the shard metadata before choosing the
        # default-pose transition implementation: using the local clip count
        # alone would splice the transition into the motion tensors and change
        # the motion clock, reward horizon, and sidecar alignment relative to
        # an unsharded launch.
        self._rank_local_shard_metadata = current_rank_local_shard_metadata()
        self.multi_clip = self.motion.num_clips > 1
        if self.multi_clip:
            logger.info("Multi-clip motion bank detected ({} clips).", self.motion.num_clips)
        elif self._uses_global_multi_clip_transition_semantics():
            transition_source = self._explicit_motion_transition_source()
            assert transition_source is not None
            logger.info(
                "Motion view contains {} active clip(s) from a {}-clip transition source; "
                "preserving global multi-clip default-pose transition semantics.",
                self.motion.num_clips,
                int(transition_source["source_clip_count"]),
            )

        self._configure_motion_terrain_pairs()

        # Store body and joint indexes for interpolation
        self._body_indexes_in_motion = self.motion._body_indexes
        self._joint_indexes_in_motion = self.motion._joint_indexes

        # Maybe prepend interpolated transition from default pose
        self._maybe_add_default_pose_transition(prepend=True)

        # Maybe append interpolated transition back to default pose
        self._maybe_add_default_pose_transition(prepend=False)

        # 2. get the indexes of the root link and the tracked links
        self.ref_body_index = robot_body_names.index(self.motion_cfg.body_name_ref[0])  # int
        self.tracked_body_indexes = self._get_index_of_a_in_b(
            self.motion_cfg.body_names_to_track, robot_body_names, self.device
        )

        # 3. get the name of the object, or indices of the object
        if self.motion.has_object:
            simulator_type = self._env.simulator.get_simulator_type()
            assert simulator_type in {SimulatorType.ISAACSIM, SimulatorType.MUJOCO}, (
                f"Object carry motions currently support IsaacSim or MuJoCo, got {simulator_type}."
            )
            self._configure_simulator_object_mapping()
            self._configure_fixed_env_clip_assignment()
            self._configure_debug_representative_clips()
        else:
            self.object_indices_in_simulator = None
        # Fixed env-to-clip assignment is not known until object mapping is
        # configured. Build the hybrid split only now so every clip receives
        # the requested tracking/task fraction independently.
        self.hybrid_stage2_task_env_mask = self._build_hybrid_stage2_task_env_mask()
        self._configure_hybrid_velocity_task_assignment()
        self._configure_hmi_partition()
        if self.pure_rl_policy_command_after_lift_enabled():
            if bool(getattr(self.motion_cfg, "hybrid_stage2_enabled", False)):
                raise ValueError(
                    "pure_rl_policy_command_after_lift and hybrid_stage2 are mutually exclusive."
                )
            logger.info(
                "Enabled pure-RL policy-input command override for all {} environments: "
                "pre_lift_actor_command=[0,0,0], post_lift_actor_command="
                "[{:.3f},0,0], rewards_and_terminations_unchanged=True, "
                "reference_tracking_offset_in_actor_command=False.",
                self.num_envs,
                float(
                    getattr(
                        self.motion_cfg,
                        "pure_rl_policy_forward_command_m",
                        0.5,
                    )
                ),
            )
        self._configure_runtime_default_pose_prepend()
        self._configure_contact_prior_regions()
        # The valid reset range depends on future-target lookahead and must be
        # finalized before constructing the adaptive sampler.
        self._configure_target_pose_settings()

        # 4. get the adaptive timesteps sampler
        self.use_adaptive_timesteps_sampler = self.motion_cfg.use_adaptive_timesteps_sampler
        if self.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler = AdaptiveTimestepsSampler(
                self.motion.time_step_total,
                self.device,
                int(1 / (self._env.dt)),
                clip_lengths=self.motion.clip_lengths,
                valid_start_counts=self._valid_start_counts().to(dtype=torch.long),
            )
            if self.multi_clip:
                logger.info(
                    "Per-clip adaptive timestep sampling enabled for multi-clip motion bank ({} clips).",
                    self.motion.num_clips,
                )
        else:
            self.adaptive_timesteps_sampler = None
        self._configure_adaptive_sampling_contact_interval_bank()

        # 5. clip sampling configuration
        self.clip_weighting_strategy = self.motion_cfg.clip_weighting_strategy
        self.min_weight_factor = self.motion_cfg.min_weight_factor
        self.max_weight_factor = self.motion_cfg.max_weight_factor
        self._clip_sampling_weights: torch.Tensor | None = None
        self._raw_clip_sampling_weights: torch.Tensor | None = None
        self._base_clip_weights: torch.Tensor | None = None
        self._clip_success_counts: torch.Tensor | None = None
        self._clip_total_counts: torch.Tensor | None = None
        self._configure_rank_local_clip_weighting()
        self._validate_fixed_env_clip_sampling_distribution()

        # 6. metrics
        self.metrics: dict[str, torch.Tensor] = {}

        self._init_clip_sampling()
        self.init_buffers()

        # 7. visualization markers for isaacsim
        if self._env.viewer and self._env.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
            self._setup_visualization_markers_for_isaacsim()

    def _configure_rank_local_clip_weighting(self) -> None:
        """Correct duplicated shard clips and expose the matching DDP loss scale.

        A clip present on ``cover_count`` ranks receives local base weight
        ``1 / cover_count``.  Because DDP gives every rank equal weight, the
        companion rank loss scale is ``world_size * local_mass / global_count``.
        Together these make every global clip contribute exactly ``1/global_count``.
        """
        if self._rank_local_shard_metadata is None:
            self._rank_local_shard_metadata = current_rank_local_shard_metadata()
        self._rank_local_inverse_cover_weights = None
        self.distributed_loss_weight = 1.0
        metadata = self._rank_local_shard_metadata
        if metadata is None:
            return
        if str(self.clip_weighting_strategy) != "uniform_clip":
            raise ValueError(
                "Rank-local global clip correction currently requires clip_weighting_strategy='uniform_clip'; "
                f"got {self.clip_weighting_strategy!r}. Other strategies require globally synchronized weights."
            )

        cover_counts = metadata.get("clip_cover_counts")
        global_clip_count = int(metadata.get("global_clip_count", 0) or 0)
        world_size = int(metadata.get("world_size", 0) or 0)
        if not isinstance(cover_counts, dict) or global_clip_count <= 0 or world_size <= 0:
            raise ValueError(
                "Rank-local shard metadata predates global clip weighting correction. "
                "Regenerate shards with scripts/prepare_as_rank_shards.py before training or inference."
            )

        metadata_clip_ids = set(str(clip_id) for clip_id in cover_counts)
        current_clip_ids = set(self.motion.clip_ids)
        if metadata_clip_ids != current_clip_ids:
            raise ValueError(
                "Rank-local shard correction metadata does not match the loaded motion clips: "
                f"metadata_only={sorted(metadata_clip_ids - current_clip_ids)}, "
                f"motion_only={sorted(current_clip_ids - metadata_clip_ids)}."
            )
        counts = torch.tensor(
            [int(cover_counts[clip_id]) for clip_id in self.motion.clip_ids],
            device=self.device,
            dtype=torch.float32,
        )
        if torch.any(counts <= 0):
            raise ValueError("Rank-local shard clip_cover_counts must all be positive.")
        self._rank_local_inverse_cover_weights = counts.reciprocal()
        local_mass = float(self._rank_local_inverse_cover_weights.sum().item())
        expected_loss_weight = float(world_size) * local_mass / float(global_clip_count)
        stored_loss_weight = float(metadata.get("distributed_loss_weight", expected_loss_weight))
        if not np.isclose(stored_loss_weight, expected_loss_weight, rtol=1.0e-6, atol=1.0e-8):
            raise ValueError(
                "Rank-local shard distributed_loss_weight is inconsistent with clip_cover_counts: "
                f"stored={stored_loss_weight}, expected={expected_loss_weight}."
            )
        self.distributed_loss_weight = expected_loss_weight
        logger.info(
            "Enabled rank-local global clip correction for {} local / {} global clips; DDP loss weight={:.6f}.",
            self.motion.num_clips,
            global_clip_count,
            self.distributed_loss_weight,
        )

    def _validate_fixed_env_clip_sampling_distribution(self) -> None:
        """Fail when a fixed object layout changes the requested clip objective.

        A per-environment object URDF cannot be changed at episode reset, so
        object-bearing clips are pinned to compatible environments.  The
        resulting empirical clip frequencies are therefore the actual training
        distribution; ``_init_clip_sampling`` cannot repair them with a sampler.
        Require that finite layout to represent the requested uniform-clip
        (and rank-local inverse-cover) distribution exactly.
        """

        fixed_clip_ids = self._fixed_clip_ids
        if fixed_clip_ids is None or int(self.motion.num_clips) <= 1:
            return
        if bool(getattr(self._env, "is_evaluating", False)):
            return

        strategy = str(self.clip_weighting_strategy)
        if strategy != "uniform_clip":
            raise ValueError(
                "Fixed env-to-clip assignment cannot honor clip_weighting_strategy="
                f"{strategy!r}: fixed object assets bypass clip resampling. Use "
                "clip_weighting_strategy='uniform_clip' or a simulator layout whose "
                "object asset can be changed safely at reset."
            )

        counts = torch.bincount(
            fixed_clip_ids.to(device=self.device, dtype=torch.long),
            minlength=int(self.motion.num_clips),
        ).to(device=self.device, dtype=torch.float64)
        if counts.numel() != int(self.motion.num_clips) or torch.any(counts <= 0):
            raise ValueError(
                "Fixed env-to-clip assignment leaves one or more motion clips unrepresented; "
                f"num_envs={int(fixed_clip_ids.numel())}, num_clips={int(self.motion.num_clips)}, "
                f"counts={counts.detach().cpu().tolist()}. Increase per-rank environments or "
                "change the rank-local shard topology."
            )

        target_weights = self._rank_local_inverse_cover_weights
        if target_weights is None:
            target_weights = torch.ones_like(counts)
        else:
            target_weights = target_weights.to(device=self.device, dtype=torch.float64)
        if (
            target_weights.numel() != counts.numel()
            or not torch.isfinite(target_weights).all()
            or torch.any(target_weights <= 0)
        ):
            raise ValueError("Fixed clip assignment received invalid rank-local target weights.")

        actual_probabilities = counts / counts.sum()
        target_probabilities = target_weights / target_weights.sum()
        if not torch.allclose(actual_probabilities, target_probabilities, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "Fixed env-to-clip assignment cannot exactly represent the scientific clip "
                "distribution on this rank. The per-rank environment count/object cycle must "
                "be compatible with the local clip weights; otherwise DDP optimizes a topology-"
                "dependent objective. "
                f"counts={counts.detach().cpu().tolist()}, "
                f"target_probabilities={target_probabilities.detach().cpu().tolist()}."
            )

    def _uses_global_multi_clip_transition_semantics(self) -> bool:
        """Keep sharded launches transition-equivalent to the global bank."""

        transition_source = self._explicit_motion_transition_source()
        if transition_source is not None:
            return transition_source["source_semantics"] == "global_multi_clip_runtime"
        if self.multi_clip:
            return True
        metadata = self._rank_local_shard_metadata
        if metadata is None:
            return False
        try:
            global_clip_count = int(metadata.get("global_clip_count", 0) or 0)
        except (TypeError, ValueError):
            return False
        return global_clip_count > 1

    def _explicit_motion_transition_source(self) -> dict[str, Any] | None:
        """Resolve and cross-check explicit timeline lineage, if present.

        New rank shards bind the same record in their object-map root and in
        ``rank_local_shard``.  Either both copies agree or scientific startup
        fails; legacy artifacts with neither copy retain the old inference.
        """

        loader_source = getattr(self.motion, "motion_transition_source", None)
        metadata = self._rank_local_shard_metadata
        rank_source = (
            metadata.get(MOTION_TRANSITION_SOURCE_KEY)
            if isinstance(metadata, dict)
            else None
        )
        if loader_source is None and rank_source is None:
            return None
        if isinstance(metadata, dict) and (loader_source is None or rank_source is None):
            raise ValueError(
                "Rank-local transition provenance is only partially present; regenerate the "
                "content-addressed shards."
            )
        active_clip_count = int(self.motion.num_clips)
        normalized_loader = canonical_motion_transition_source(
            loader_source,
            active_clip_count=active_clip_count,
            role=f"MotionLoader.{MOTION_TRANSITION_SOURCE_KEY}",
        )
        if rank_source is not None:
            normalized_rank = canonical_motion_transition_source(
                rank_source,
                active_clip_count=active_clip_count,
                role=f"rank_local_shard.{MOTION_TRANSITION_SOURCE_KEY}",
            )
            if normalized_rank != normalized_loader:
                raise ValueError(
                    "MotionLoader and rank-local shard transition provenance disagree: "
                    f"loader={normalized_loader}, rank={normalized_rank}."
                )
        return normalized_loader

    @staticmethod
    def _normalize_path_key(path: str) -> str:
        if not path:
            return ""
        try:
            return str(Path(resolve_data_file_path(path)).resolve())
        except Exception:
            return path

    def _resolve_sim_object_name(
        self,
        *,
        clip_id: str,
        clip_object_name: str,
        clip_object_urdf: str,
        sim_names: list[str],
        sim_name_by_urdf: dict[str, str],
        sim_name_by_stem: dict[str, str],
    ) -> str:
        normalized_urdf = self._normalize_path_key(clip_object_urdf)
        if normalized_urdf and normalized_urdf in sim_name_by_urdf:
            return sim_name_by_urdf[normalized_urdf]

        if normalized_urdf:
            stem = Path(normalized_urdf).stem.lower()
            if stem in sim_name_by_stem:
                return sim_name_by_stem[stem]

        key = clip_object_name.strip().lower()
        if key:
            if key in sim_name_by_stem:
                return sim_name_by_stem[key]
            for name in sim_names:
                name_lc = name.lower()
                if key == name_lc or name_lc.endswith(f"_{key}") or name_lc.endswith(key):
                    return name

        available_urdfs = sorted(sim_name_by_urdf.keys())
        raise RuntimeError(
            "Failed to resolve simulator object for clip "
            f"'{clip_id}' (object_name='{clip_object_name}', object_urdf='{clip_object_urdf}'). "
            f"Available simulator objects: {sim_names}. "
            f"Available simulator URDFs: {available_urdfs}."
        )

    def _configure_simulator_object_mapping(self) -> None:
        sim = self._env.simulator
        rigid_objects = getattr(getattr(sim, "scene", None), "rigid_objects", {})

        object_urdf_by_name_raw = getattr(sim, "_object_urdf_by_name", {})
        object_urdf_by_name: dict[str, str] = (
            dict(object_urdf_by_name_raw) if isinstance(object_urdf_by_name_raw, dict) else {}
        )
        sim_object_names: list[str] = [name for name in object_urdf_by_name.keys() if name != "usd_scene_objects"]
        if not sim_object_names and hasattr(rigid_objects, "keys"):
            sim_object_names = [name for name in rigid_objects.keys() if name != "usd_scene_objects"]
        if not sim_object_names:
            sim_object_names = ["object"]

        self._sim_object_names = list(dict.fromkeys(sim_object_names))
        self._clip_object_ids = torch.zeros(self.motion.num_clips, dtype=torch.long, device=self.device)

        if len(self._sim_object_names) == 1:
            self.object_name = self._sim_object_names[0]
            self.object_indices_in_simulator = sim.get_actor_indices(self.object_name, env_ids=None)
            self._multi_object_enabled = False
            self._object_indices_matrix = None
            env_object_urdf_paths = getattr(sim, "_env_object_urdf_paths", None)
            if isinstance(env_object_urdf_paths, list) and env_object_urdf_paths:
                unique_env_object_count = len({self._normalize_path_key(path) for path in env_object_urdf_paths if path})
                logger.info(
                    "Using single simulator object slot '{}' with {} env-specific object assignment(s) across {} clips.",
                    self.object_name,
                    unique_env_object_count,
                    self.motion.num_clips,
                )
            else:
                logger.info("Using single object '{}' for all {} clips.", self.object_name, self.motion.num_clips)
            return

        sim_name_by_urdf: dict[str, str] = {}
        sim_name_by_stem: dict[str, str] = {}
        for name in self._sim_object_names:
            sim_name_by_stem[name.lower()] = name
            urdf_path = object_urdf_by_name.get(name, "")
            normalized = self._normalize_path_key(urdf_path)
            if normalized:
                sim_name_by_urdf[normalized] = name
                sim_name_by_stem[Path(normalized).stem.lower()] = name
        clip_object_names = self.motion.clip_object_names
        clip_object_urdfs = self.motion.clip_object_urdf_paths
        if len(clip_object_names) != self.motion.num_clips:
            clip_object_names = [""] * self.motion.num_clips
        if len(clip_object_urdfs) != self.motion.num_clips:
            clip_object_urdfs = [""] * self.motion.num_clips

        clip_object_ids: list[int] = []
        for clip_idx, clip_id in enumerate(self.motion.clip_ids):
            resolved_name = self._resolve_sim_object_name(
                clip_id=clip_id,
                clip_object_name=clip_object_names[clip_idx],
                clip_object_urdf=clip_object_urdfs[clip_idx],
                sim_names=self._sim_object_names,
                sim_name_by_urdf=sim_name_by_urdf,
                sim_name_by_stem=sim_name_by_stem,
            )
            clip_object_ids.append(self._sim_object_names.index(resolved_name))

        self._clip_object_ids = torch.tensor(clip_object_ids, dtype=torch.long, device=self.device)
        object_indices = [sim.get_actor_indices(name, env_ids=None) for name in self._sim_object_names]
        self._object_indices_matrix = torch.stack(object_indices, dim=0)
        self.object_name = self._sim_object_names[0]
        self.object_indices_in_simulator = self._object_indices_matrix[0]
        self._multi_object_enabled = True
        logger.info(
            "Configured multi-object mapping: {} simulator objects for {} clips.",
            len(self._sim_object_names),
            self.motion.num_clips,
        )

    def _configure_debug_representative_clips(self) -> None:
        self._debug_representative_clip_ids = None
        debug_mode = bool(getattr(self._env.training_config, "debug", False))
        toy_mode = bool(getattr(self._env.training_config, "toy_mode", False))
        if not (debug_mode or toy_mode):
            return
        if not self.multi_clip or not self.motion.has_object:
            return

        clip_object_names = self.motion.clip_object_names
        clip_object_urdfs = self.motion.clip_object_urdf_paths
        representative_ids: list[int] = []
        seen_keys: set[str] = set()
        for clip_idx in range(self.motion.num_clips):
            obj_name = clip_object_names[clip_idx].strip() if clip_idx < len(clip_object_names) else ""
            obj_urdf = clip_object_urdfs[clip_idx].strip() if clip_idx < len(clip_object_urdfs) else ""
            normalized_urdf = self._normalize_path_key(obj_urdf)
            if normalized_urdf:
                key = f"urdf::{normalized_urdf}"
            elif obj_name:
                key = f"name::{obj_name.lower()}"
            else:
                key = "unknown"

            if key in seen_keys:
                continue
            seen_keys.add(key)
            representative_ids.append(clip_idx)

        if not representative_ids:
            representative_ids = [0]

        self._debug_representative_clip_ids = torch.tensor(representative_ids, dtype=torch.long, device=self.device)
        logger.info(
            "Debug/Toy mode: using {} representative clips (one per URDF/object key) over {} total clips.",
            len(representative_ids),
            self.motion.num_clips,
        )

    def _configure_fixed_env_clip_assignment(self) -> None:
        self._fixed_clip_ids = None
        self._fixed_clip_group_env_mask = None
        self._fixed_clip_group_clip_mask = None
        self._fixed_clip_complement_clip_mask = None
        force_round_robin = os.environ.get("HOLOSOMA_FORCE_ROUND_ROBIN_CLIP_ASSIGNMENT", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if force_round_robin:
            if self.motion.num_clips <= 0:
                raise RuntimeError("Round-robin clip assignment requested but motion bank is empty.")
            clip_start = int(os.environ.get("HOLOSOMA_FORCE_ROUND_ROBIN_CLIP_START", "0") or "0")
            fixed_clip_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long) + clip_start
            fixed_clip_ids = torch.remainder(fixed_clip_ids, int(self.motion.num_clips))
            self._fixed_clip_ids = fixed_clip_ids
            if hasattr(self, "clip_ids") and isinstance(self.clip_ids, torch.Tensor) and self.clip_ids.numel() == fixed_clip_ids.numel():
                self.clip_ids[:] = fixed_clip_ids
            logger.info(
                "Configured forced round-robin env-to-clip assignment across {} envs and {} clips (start={}).",
                self.num_envs,
                self.motion.num_clips,
                clip_start,
            )
            return

        if self._configure_fixed_clip_group_assignment():
            return

        if not self.motion.has_object:
            return

        env_object_urdf_paths = getattr(self._env.simulator, "_env_object_urdf_paths", None)
        if not isinstance(env_object_urdf_paths, list) or not env_object_urdf_paths:
            require_single_slot_objects = os.environ.get(
                "HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS", ""
            ).strip().lower() in {"1", "true", "yes", "on"}
            if require_single_slot_objects:
                raise RuntimeError(
                    "HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS=1 requires simulator._env_object_urdf_paths "
                    "so AS envs can be pinned to matching object clips."
                )
            return
        if len(env_object_urdf_paths) != self.num_envs:
            raise RuntimeError(
                "Fixed env-to-clip assignment requires one simulator object URDF per env. "
                f"Got {len(env_object_urdf_paths)} entries for {self.num_envs} envs."
            )

        clip_object_urdfs = self.motion.clip_object_urdf_paths
        if len(clip_object_urdfs) != self.motion.num_clips:
            raise RuntimeError(
                "Fixed env-to-clip assignment requires clip object URDF metadata for every clip. "
                f"Motion bank exposed {len(clip_object_urdfs)} URDF entries for {self.motion.num_clips} clips."
            )

        clip_ids_by_urdf: dict[str, list[int]] = {}
        missing_clip_urdf_ids: list[str] = []
        for clip_idx, clip_urdf in enumerate(clip_object_urdfs):
            normalized_urdf = self._normalize_path_key(clip_urdf)
            if not normalized_urdf:
                missing_clip_urdf_ids.append(self.motion.clip_ids[clip_idx])
                continue
            clip_ids_by_urdf.setdefault(normalized_urdf, []).append(clip_idx)

        if missing_clip_urdf_ids:
            raise RuntimeError(
                "Fixed env-to-clip assignment requires object URDF metadata on every clip. "
                f"Missing clip metadata for {len(missing_clip_urdf_ids)} clip(s): {missing_clip_urdf_ids[:8]}"
            )

        fixed_clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        seen_counts_by_urdf: dict[str, int] = {}
        unmatched_env_ids: list[int] = []
        for env_id, env_object_urdf in enumerate(env_object_urdf_paths):
            normalized_urdf = self._normalize_path_key(env_object_urdf)
            clip_candidates = clip_ids_by_urdf.get(normalized_urdf)
            if not clip_candidates:
                unmatched_env_ids.append(env_id)
                continue
            seen_count = seen_counts_by_urdf.get(normalized_urdf, 0)
            fixed_clip_ids[env_id] = int(clip_candidates[seen_count % len(clip_candidates)])
            seen_counts_by_urdf[normalized_urdf] = seen_count + 1

        if unmatched_env_ids:
            sample_env_ids = unmatched_env_ids[:8]
            sample_urdfs = [env_object_urdf_paths[idx] for idx in sample_env_ids]
            raise RuntimeError(
                "Fixed env-to-clip assignment requires every env object URDF to appear in the motion bank. "
                f"Unmatched env count={len(unmatched_env_ids)} sample env ids={sample_env_ids} "
                f"sample urdfs={sample_urdfs}"
            )

        mismatched_envs: list[tuple[int, str, str]] = []
        fixed_clip_ids_cpu = fixed_clip_ids.detach().to(device="cpu").tolist()
        for env_id, clip_idx in enumerate(fixed_clip_ids_cpu):
            env_key = self._normalize_path_key(env_object_urdf_paths[env_id])
            clip_key = self._normalize_path_key(clip_object_urdfs[int(clip_idx)])
            if env_key != clip_key:
                mismatched_envs.append((env_id, env_key, clip_key))
                if len(mismatched_envs) >= 8:
                    break
        if mismatched_envs:
            raise RuntimeError(
                "Fixed env-to-clip assignment produced object/clip URDF mismatches. "
                f"Samples: {mismatched_envs}"
            )

        self._fixed_clip_ids = fixed_clip_ids
        assigned_unique_clip_count = int(torch.unique(fixed_clip_ids).numel())
        clip_groups_with_multiple_clips = sum(1 for clip_ids in clip_ids_by_urdf.values() if len(clip_ids) > 1)
        if clip_groups_with_multiple_clips > 0:
            logger.info(
                "Configured fixed env-to-clip assignment across {} envs using {} URDF groups and {} active clips. "
                "URDF groups with multiple clips are assigned round-robin across envs.",
                self.num_envs,
                len(clip_ids_by_urdf),
                assigned_unique_clip_count,
            )
        else:
            logger.info(
                "Configured fixed env-to-clip assignment across {} envs and {} active clips.",
                self.num_envs,
                assigned_unique_clip_count,
            )

    def _configure_fixed_clip_group_assignment(self) -> bool:
        cfg = self._fixed_clip_group_assignment_cfg
        if cfg is None or not cfg.enabled:
            return False
        if not self.multi_clip:
            logger.warning("fixed_clip_group_assignment is enabled but the loaded motion is not a multi-clip bank.")
            return False

        env_object_urdf_paths = getattr(self._env.simulator, "_env_object_urdf_paths", None)
        if isinstance(env_object_urdf_paths, list) and env_object_urdf_paths:
            logger.warning(
                "fixed_clip_group_assignment is enabled but simulator has per-env object URDF assignment. "
                "Using fixed env-to-clip assignment instead to avoid object/clip mismatches."
            )
            return False

        group_env_fraction = float(cfg.group_env_fraction)
        if group_env_fraction < 0.0 or group_env_fraction > 1.0:
            raise ValueError(
                "fixed_clip_group_assignment.group_env_fraction must stay in [0, 1], "
                f"got {cfg.group_env_fraction}."
            )

        group_clip_mask = build_prefix_mask(self.motion.clip_ids, cfg.group_clip_name_prefixes).to(device=self.device)
        complement_clip_mask = ~group_clip_mask
        if not torch.any(group_clip_mask):
            raise ValueError(
                "fixed_clip_group_assignment is enabled but no clips matched prefixes "
                f"{cfg.group_clip_name_prefixes}."
            )
        if not torch.any(complement_clip_mask):
            raise ValueError(
                "fixed_clip_group_assignment is enabled but all clips matched prefixes "
                f"{cfg.group_clip_name_prefixes}; complement group is empty."
            )

        group_env_count = int(round(float(self.num_envs) * group_env_fraction))
        group_env_count = max(0, min(int(self.num_envs), group_env_count))
        group_env_mask = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        if group_env_count > 0:
            spread_ids = torch.floor(
                (torch.arange(group_env_count, device=self.device, dtype=torch.float32) + 0.5)
                * float(self.num_envs)
                / float(group_env_count)
            ).to(dtype=torch.long)
            spread_ids = torch.clamp(spread_ids, min=0, max=int(self.num_envs) - 1)
            group_env_mask[spread_ids] = True

        self._fixed_clip_group_env_mask = group_env_mask
        self._fixed_clip_group_clip_mask = group_clip_mask
        self._fixed_clip_complement_clip_mask = complement_clip_mask
        logger.info(
            "Enabled fixed clip-group assignment: {} / {} envs sample only group clips matching prefixes {}; "
            "{} group clips, {} complement clips.",
            int(group_env_mask.sum().item()),
            self.num_envs,
            list(cfg.group_clip_name_prefixes),
            int(group_clip_mask.sum().item()),
            int(complement_clip_mask.sum().item()),
        )
        return True

    def _configure_contact_prior_regions(self) -> None:
        self._contact_prior_active = False
        self._contact_prior_available = False
        self._contact_prior_force_body_names_by_region = {region: [] for region in _CONTACT_PRIOR_REGION_NAMES}
        self._contact_prior_position_body_names_by_region = {region: [] for region in _CONTACT_PRIOR_REGION_NAMES}
        self._contact_prior_position_body_indices_by_region = {
            region: torch.zeros((0,), device=self.device, dtype=torch.long) for region in _CONTACT_PRIOR_REGION_NAMES
        }
        if not self.motion.has_object:
            return
        self._contact_prior_active = bool(self._should_enable_online_contact_prior())
        if not self._contact_prior_active:
            logger.info("Online contact prior disabled: no contact-prior observation term is configured.")
            return

        getter = getattr(self._env.simulator, "get_object_contact_force_history", None)
        if getter is None:
            logger.warning("Online contact prior disabled: simulator does not expose object-only contact force history.")
            return

        all_body_names = list(self._env.simulator.body_names)  # type: ignore[attr-defined]
        body_name_to_index = {name: idx for idx, name in enumerate(all_body_names)}
        self._contact_prior_available = True

        for region_name in _CONTACT_PRIOR_REGION_NAMES:
            force_names = [
                body_name
                for body_name in _CONTACT_PRIOR_REGION_FORCE_BODY_NAMES[region_name]
                if body_name in body_name_to_index
            ]
            position_names = [
                body_name
                for body_name in _CONTACT_PRIOR_REGION_POSITION_BODY_NAMES[region_name]
                if body_name in body_name_to_index
            ]
            self._contact_prior_force_body_names_by_region[region_name] = force_names
            self._contact_prior_position_body_names_by_region[region_name] = position_names
            position_indices = [body_name_to_index[body_name] for body_name in position_names]
            self._contact_prior_position_body_indices_by_region[region_name] = torch.tensor(
                position_indices,
                dtype=torch.long,
                device=self.device,
            )
            if not force_names or not position_names:
                logger.warning(
                    "Contact prior region '{}' is partially unavailable. force_bodies={} position_bodies={}",
                    region_name,
                    force_names,
                    position_names,
                )

    def _should_enable_online_contact_prior(self) -> bool:
        override = os.environ.get("HOLOSOMA_ONLINE_CONTACT_PRIOR", "").strip().lower()
        if override in ("1", "true", "yes", "on"):
            return True
        if override in ("0", "false", "no", "off"):
            return False

        disable = os.environ.get("HOLOSOMA_DISABLE_ONLINE_CONTACT_PRIOR", "").strip().lower()
        if disable in ("1", "true", "yes", "on"):
            return False

        observation_manager = getattr(self._env, "observation_manager", None)
        cfg = getattr(observation_manager, "cfg", None)
        groups = getattr(cfg, "groups", {}) or {}
        for group_cfg in groups.values():
            terms = getattr(group_cfg, "terms", {}) or {}
            for term_cfg in terms.values():
                func = getattr(term_cfg, "func", "")
                func_name = str(func)
                if not func_name or func_name == "None":
                    func_name = getattr(func, "__name__", "")
                if "contact_prior" in func_name:
                    return True
        return False

    def _has_contact_window_observation_consumer(self) -> bool:
        """Return whether a configured observation term consumes exported contact windows.

        Contact-aware command/button observations are independent of adaptive reset
        sampling.  In particular, evaluation commonly disables the sampler while it
        must retain the observation semantics saved by training.
        """

        observation_manager = getattr(getattr(self, "_env", None), "observation_manager", None)
        cfg = getattr(observation_manager, "cfg", None)
        groups = getattr(cfg, "groups", {}) or {}
        for group_cfg in groups.values():
            terms = getattr(group_cfg, "terms", {}) or {}
            for term_cfg in terms.values():
                func = getattr(term_cfg, "func", "")
                if isinstance(func, str):
                    func_path = func.replace(":", ".")
                else:
                    func_path = f"{getattr(func, '__module__', '')}.{getattr(func, '__name__', '')}"
                if func_path in _CONTACT_WINDOW_OBSERVATION_FUNCTION_PATHS:
                    return True
        return False

    def _get_active_object_indices(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if self.object_indices_in_simulator is None:
            raise RuntimeError(
                "Simulator object indices are not configured. "
                "Use motion clips with object data and enable robot object assets, "
                "or switch to a non-object experiment."
            )
        env_ids_tensor = self._ensure_index_tensor(env_ids)
        if not self._multi_object_enabled or self._clip_object_ids is None or self._object_indices_matrix is None:
            if env_ids is None:
                return self.object_indices_in_simulator
            return self.object_indices_in_simulator[env_ids_tensor]
        active_object_ids = self._clip_object_ids[self.clip_ids[env_ids_tensor]]
        return self._object_indices_matrix[active_object_ids, env_ids_tensor]

    def _read_active_simulator_object_states(
        self,
        env_ids: torch.Tensor,
        *,
        all_envs: bool = False,
    ) -> torch.Tensor:
        """Read each environment's active object exactly once from the backend."""

        simulator = self._env.simulator
        simulator_type_getter = getattr(simulator, "get_simulator_type", None)
        simulator_type = simulator_type_getter() if callable(simulator_type_getter) else None

        # AllRootStatesProxy must reverse-decode flat CUDA indices.  A
        # single-object IsaacSim WBT task already knows both the object name and
        # env rows, so route directly through the state adapter instead.
        state_adapter = getattr(simulator, "_state_adapter", None)
        direct_getter = getattr(state_adapter, "get_object_states", None)
        if (
            not getattr(self, "_multi_object_enabled", False)
            and simulator_type == SimulatorType.ISAACSIM
            and callable(direct_getter)
        ):
            states = direct_getter(self.object_name, env_ids)
        else:
            active_indices = (
                self._get_active_object_indices()
                if all_envs
                else self._get_active_object_indices(env_ids)
            )
            states = simulator.all_root_states[active_indices, :13]

        if states.ndim != 2 or states.shape != (env_ids.numel(), 13):
            raise RuntimeError(
                "Simulator active-object state read must return shape "
                f"({env_ids.numel()}, 13), got {tuple(states.shape)}."
            )
        return states

    def refresh_simulator_object_state_snapshot(self, env_ids: torch.Tensor | None = None) -> None:
        """Refresh the shared active-object snapshot after simulator tensor refresh.

        ``env_ids=None`` is the normal once-per-control-step refresh.  Reset
        synchronization passes a subset, preserving survivor rows while making
        the freshly written/reset rows authoritative immediately.
        """

        if not self.motion.has_object:
            return
        full_refresh = env_ids is None
        env_ids_tensor = self._ensure_index_tensor(env_ids)
        if env_ids_tensor.numel() == 0:
            return
        states = self._read_active_simulator_object_states(env_ids_tensor, all_envs=full_refresh)
        snapshot = getattr(self, "_simulator_object_state_snapshot", None)
        if snapshot is None or snapshot.shape != (self.num_envs, 13):
            snapshot = torch.empty(
                (self.num_envs, 13),
                device=self.device,
                dtype=states.dtype,
            )
            self._simulator_object_state_snapshot = snapshot
            self._simulator_object_state_snapshot_ready = False
        if env_ids is None:
            snapshot.copy_(states)
            self._simulator_object_state_snapshot_ready = True
        else:
            snapshot[env_ids_tensor] = states

    def _update_simulator_object_state_snapshot(
        self,
        env_ids: torch.Tensor,
        active_states: torch.Tensor,
    ) -> None:
        """Update reset/write rows without an unnecessary backend round trip."""

        snapshot = getattr(self, "_simulator_object_state_snapshot", None)
        if snapshot is None or snapshot.shape != (self.num_envs, 13):
            snapshot = torch.empty(
                (self.num_envs, 13),
                device=self.device,
                dtype=active_states.dtype,
            )
            self._simulator_object_state_snapshot = snapshot
            self._simulator_object_state_snapshot_ready = False
        snapshot[env_ids] = active_states[:, :13].to(device=snapshot.device, dtype=snapshot.dtype)

    @property
    def simulator_object_state_snapshot(self) -> torch.Tensor:
        """Return the current active-object state shared by WBT consumers."""

        if not self.motion.has_object:
            states = torch.zeros((self.num_envs, 13), device=self.device, dtype=torch.float32)
            states[:, 6] = 1.0
            return states
        if not getattr(self, "_simulator_object_state_snapshot_ready", False):
            self.refresh_simulator_object_state_snapshot()
        snapshot = getattr(self, "_simulator_object_state_snapshot", None)
        if snapshot is None:
            raise RuntimeError("Simulator object snapshot refresh did not initialize its buffer.")
        return snapshot

    def _set_simulator_object_states(self, env_ids: torch.Tensor, active_states: torch.Tensor) -> None:
        env_ids = self._ensure_index_tensor(env_ids)
        if active_states.ndim != 2 or active_states.shape != (env_ids.numel(), 13):
            raise ValueError(
                "Active simulator object states must have shape "
                f"({env_ids.numel()}, 13), got {tuple(active_states.shape)}."
            )
        if not self._multi_object_enabled or self._clip_object_ids is None:
            self._env.simulator.set_actor_states([self.object_name], env_ids, active_states)
            self._update_simulator_object_state_snapshot(env_ids, active_states)
            return

        active_object_ids = self._clip_object_ids[self.clip_ids[env_ids]]
        all_states: list[torch.Tensor] = []
        for object_id, _ in enumerate(self._sim_object_names):
            states = torch.zeros((env_ids.numel(), 13), device=self.device, dtype=torch.float32)
            states[:, 2] = -100.0 - 5.0 * float(object_id)
            states[:, 6] = 1.0
            active_mask = active_object_ids == object_id
            if active_mask.any():
                states[active_mask] = active_states[active_mask]
            all_states.append(states)
        stacked_states = torch.cat(all_states, dim=0)
        self._env.simulator.set_actor_states(self._sim_object_names, env_ids, stacked_states)
        self._update_simulator_object_state_snapshot(env_ids, active_states)

    def _apply_manual_object_reset_overrides(
        self,
        obj_pos_w: torch.Tensor,
        obj_quat_w: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.manual_object_reset_enabled:
            return obj_pos_w, obj_quat_w
        if self.manual_object_reset_pos_offset_w is not None:
            obj_pos_w = obj_pos_w + self.manual_object_reset_pos_offset_w[env_ids]
        if self.manual_object_reset_rpy_offset is not None:
            rpy = self.manual_object_reset_rpy_offset[env_ids]
            delta_quat = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
            obj_quat_w = quat_mul(delta_quat, obj_quat_w, w_last=True)
        return obj_pos_w, obj_quat_w

    @staticmethod
    def contact_prior_region_names() -> tuple[str, ...]:
        return _CONTACT_PRIOR_REGION_NAMES

    def _current_contact_prior_phase_ids(self) -> torch.Tensor:
        if self.pickup_anchor_set is None:
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        return self.pickup_anchor_set.to(dtype=torch.long)

    def _object_contact_force_history_by_names(self, body_names: list[str]) -> torch.Tensor:
        return self.get_body_object_contact_force_history(body_names)

    def _object_contact_body_indices_by_names(self, body_names: list[str]) -> torch.Tensor:
        key = tuple(body_names)
        cached = self._object_contact_body_indices_cache.get(key)
        if cached is not None:
            return cached

        simulator_body_names = list(getattr(self._env.simulator, "body_names", []))
        missing = [name for name in body_names if name not in simulator_body_names]
        if missing:
            raise ValueError(f"Requested object-contact bodies {missing} are not available in simulator bodies.")

        indices = torch.tensor(
            [simulator_body_names.index(name) for name in body_names],
            device=self.device,
            dtype=torch.long,
        )
        self._object_contact_body_indices_cache[key] = indices
        return indices

    def _object_contact_proximity_mask_by_indices(
        self,
        body_indices: torch.Tensor,
        *,
        distance_threshold: float = _OBJECT_CONTACT_PROXY_DISTANCE_THRESHOLD,
    ) -> torch.Tensor:
        if body_indices.numel() == 0 or not self.motion.has_object:
            return torch.zeros((self.num_envs, body_indices.numel()), device=self.device, dtype=torch.bool)

        half_extents = 0.5 * torch.clamp(self.object_size, min=1.0e-4)
        body_pos_obj = self._body_positions_in_object_frame(body_indices)
        signed_outside = torch.abs(body_pos_obj) - half_extents.unsqueeze(1)
        outside = torch.clamp(signed_outside, min=0.0)
        outside_dist = torch.linalg.norm(outside, dim=-1)
        return outside_dist <= float(distance_threshold)

    def _proxy_body_object_contact_force_history(self, body_names: list[str]) -> torch.Tensor:
        if not body_names:
            return torch.zeros((self.num_envs, 1, 0, 3), device=self.device, dtype=torch.float32)

        raw_history = getattr(self._env.simulator, "contact_forces_history", None)
        if raw_history is None:
            return torch.zeros((self.num_envs, 1, len(body_names), 3), device=self.device, dtype=torch.float32)

        body_indices = self._object_contact_body_indices_by_names(body_names)
        body_force_history = raw_history[:, :, body_indices, :].to(dtype=torch.float32)
        proximity_mask = self._object_contact_proximity_mask_by_indices(body_indices).to(dtype=body_force_history.dtype)
        return body_force_history * proximity_mask.unsqueeze(1).unsqueeze(-1)

    def get_body_object_contact_force_history(self, body_names: list[str]) -> torch.Tensor:
        if not body_names:
            return torch.zeros((self.num_envs, 1, 0, 3), device=self.device, dtype=torch.float32)

        proxy_history = self._proxy_body_object_contact_force_history(body_names)

        getter = getattr(self._env.simulator, "get_object_contact_force_history", None)
        if getter is None:
            return proxy_history

        try:
            filtered_history = getter(body_names).to(dtype=torch.float32)
        except Exception:
            return proxy_history

        if filtered_history.shape[1] != proxy_history.shape[1]:
            if filtered_history.shape[1] == 1:
                filtered_history = filtered_history.expand(-1, proxy_history.shape[1], -1, -1)
            elif proxy_history.shape[1] == 1:
                proxy_history = proxy_history.expand(-1, filtered_history.shape[1], -1, -1)
            else:
                history_len = min(filtered_history.shape[1], proxy_history.shape[1])
                filtered_history = filtered_history[:, :history_len]
                proxy_history = proxy_history[:, :history_len]

        filtered_norm = torch.linalg.norm(filtered_history, dim=-1)
        proxy_norm = torch.linalg.norm(proxy_history, dim=-1)
        use_proxy = proxy_norm > filtered_norm
        return torch.where(use_proxy.unsqueeze(-1), proxy_history, filtered_history)

    def _body_positions_in_object_frame(self, body_indices: torch.Tensor) -> torch.Tensor:
        if body_indices.numel() == 0:
            return torch.zeros((self.num_envs, 0, 3), device=self.device, dtype=torch.float32)
        body_pos_w = self._env.simulator._rigid_body_pos[:, body_indices, :]
        object_pos_w = self.simulator_object_pos_w[:, None, :]
        object_quat_inv = quat_inverse(self.simulator_object_quat_w, w_last=True)[:, None, :].expand(
            -1, body_indices.numel(), -1
        )
        return quat_apply(object_quat_inv, body_pos_w - object_pos_w, w_last=True)

    def get_current_contact_prior_region_measurements(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_regions = len(_CONTACT_PRIOR_REGION_NAMES)
        current_force = torch.zeros((self.num_envs, num_regions), device=self.device, dtype=torch.float32)
        current_contact = torch.zeros((self.num_envs, num_regions), device=self.device, dtype=torch.bool)
        current_position = torch.zeros((self.num_envs, num_regions, 3), device=self.device, dtype=torch.float32)
        if not self.motion.has_object or not self._contact_prior_available:
            return current_force, current_contact, current_position

        for region_idx, region_name in enumerate(_CONTACT_PRIOR_REGION_NAMES):
            force_body_names = self._contact_prior_force_body_names_by_region.get(region_name, [])
            if force_body_names:
                force_history = self._object_contact_force_history_by_names(force_body_names)
                per_body_force = torch.max(torch.norm(force_history, dim=-1), dim=1)[0]
                region_force = torch.max(per_body_force, dim=1)[0]
                current_force[:, region_idx] = region_force
                current_contact[:, region_idx] = region_force > _CONTACT_PRIOR_FORCE_THRESHOLD

            position_body_indices = self._contact_prior_position_body_indices_by_region.get(region_name)
            if position_body_indices is None or position_body_indices.numel() == 0:
                continue

            position_body_names = self._contact_prior_position_body_names_by_region.get(region_name, [])
            relative_positions = self._body_positions_in_object_frame(position_body_indices)
            if position_body_names:
                position_force_history = self._object_contact_force_history_by_names(position_body_names)
                position_force_weights = torch.max(torch.norm(position_force_history, dim=-1), dim=1)[0]
            else:
                position_force_weights = torch.zeros(
                    (self.num_envs, position_body_indices.numel()),
                    device=self.device,
                    dtype=torch.float32,
                )

            uniform_weights = torch.full_like(position_force_weights, 1.0 / float(position_body_indices.numel()))
            weight_denom = position_force_weights.sum(dim=1, keepdim=True)
            normalized_weights = torch.where(
                weight_denom > 1.0e-6,
                position_force_weights / weight_denom.clamp_min(1.0e-6),
                uniform_weights,
            )
            current_position[:, region_idx] = torch.sum(relative_positions * normalized_weights.unsqueeze(-1), dim=1)

        return current_force, current_contact, current_position

    def _default_pose_reset_targets(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dof_pos = self._env.default_dof_pos[env_ids].clone()
        dof_vel = torch.zeros_like(dof_pos)

        root_pos = self._motion_body_pos_w(env_ids)[:, 0].clone()
        root_pos[:, 2] = self._init_root_pos[2]

        init_root_quat = self._init_root_rot.unsqueeze(0).expand(env_ids.numel(), -1)
        init_roll, init_pitch, _ = get_euler_xyz(init_root_quat, w_last=True)
        _, _, motion_yaw = get_euler_xyz(self._motion_body_quat_w(env_ids)[:, 0], w_last=True)
        root_rot = quat_from_euler_xyz(init_roll, init_pitch, motion_yaw)

        root_lin_vel = self._init_root_lin_vel.unsqueeze(0).expand(env_ids.numel(), -1).clone()
        root_ang_vel = self._init_root_ang_vel.unsqueeze(0).expand(env_ids.numel(), -1).clone()
        return dof_pos, dof_vel, root_pos, root_rot, root_lin_vel, root_ang_vel

    def _adaptive_failure_mask_for_env_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return genuine failure terms, excluding successful motion completion."""
        manager = self._env.termination_manager
        device = self.time_steps.device
        failed = torch.zeros(env_ids.numel(), device=device, dtype=torch.bool)
        found_failure_term = False
        terms = getattr(getattr(manager, "cfg", None), "terms", {})
        items = terms.items() if hasattr(terms, "items") else ()
        get_last_term_result = getattr(manager, "get_last_term_result", lambda _name: None)
        for term_name, term_cfg in items:
            if bool(getattr(term_cfg, "is_timeout", False)) or str(term_name) == "motion_ends":
                continue
            result = get_last_term_result(str(term_name))
            if result is None:
                continue
            failed |= result[env_ids].to(device=device, dtype=torch.bool)
            found_failure_term = True
        if found_failure_term:
            return failed

        # Compatibility fallback for lightweight/custom termination managers.
        failed = manager.terminated[env_ids].to(device=device, dtype=torch.bool).clone()
        motion_ends = get_last_term_result("motion_ends")
        if motion_ends is not None:
            failed &= ~motion_ends[env_ids].to(device=device, dtype=torch.bool)
        return failed

    def _motion_completed_mask_for_env_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        device = self.time_steps.device
        completed = self.motion_end_mask()[env_ids].to(device=device, dtype=torch.bool)
        get_last_term_result = getattr(self._env.termination_manager, "get_last_term_result", lambda _name: None)
        motion_ends = get_last_term_result("motion_ends")
        if motion_ends is not None:
            completed |= motion_ends[env_ids].to(device=device, dtype=torch.bool)
        return completed

    def _base_reset_has_previous_action_mask(self, env_ids: torch.Tensor) -> torch.Tensor:
        episode_lengths = self._env.episode_length_buf[env_ids]
        pending_lengths = getattr(self._env, "_pending_episode_lengths", None)
        if not isinstance(pending_lengths, torch.Tensor):
            return torch.zeros(env_ids.numel(), device=self.time_steps.device, dtype=torch.bool)
        return (episode_lengths == 0) & (pending_lengths[env_ids] > 0)

    def _update_adaptive_timestep_failure_stats_before_resample(self, env_ids: torch.Tensor) -> None:
        """Record the old state/action visit and reset outcome before clip/time replacement."""
        if not self.use_adaptive_timesteps_sampler or bool(getattr(self._env, "is_evaluating", False)):
            return
        if self.hmi_enabled():
            env_ids = env_ids[self.get_hmi_track_env_mask()[env_ids]]
            if env_ids.numel() == 0:
                return
        previous_action = self._base_reset_has_previous_action_mask(env_ids)

        # BaseTask has already zeroed episode_length_buf, but clip_ids/time_steps
        # still describe the state on which the terminating action was taken.
        failed = self._adaptive_failure_mask_for_env_ids(env_ids)
        self.adaptive_timesteps_sampler.update_current_bin_outcome_count(
            self.time_steps[env_ids],
            clip_ids=self.clip_ids[env_ids],
            failed=failed,
            observed=previous_action,
            _trusted_clip_ids=True,
        )

    def _record_adaptive_timestep_exposure_before_advance(self) -> None:
        """Record visits for envs that were not reset before MotionCommand.step()."""
        if not self.use_adaptive_timesteps_sampler or bool(getattr(self._env, "is_evaluating", False)):
            return
        visited = self._env.episode_length_buf > 0
        if self.hmi_enabled():
            visited &= self.get_hmi_track_env_mask()
        self.adaptive_timesteps_sampler.update_current_bin_exposure_count(
            self.time_steps,
            clip_ids=self.clip_ids,
            observed=visited,
            _trusted_clip_ids=True,
        )

    def reset(self, env_ids: torch.Tensor | None) -> None:
        """called per reset_idx, reset timesteps and robot/object poses."""
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        debug_tile_layout = os.environ.get("HOLOSOMA_DEBUG_TILE_LAYOUT", "0").lower() in ("1", "true", "yes", "on")
        use_fixed_tile_layout = (
            debug_tile_layout
            and self.multi_clip
            and self._fixed_clip_ids is None
            and self.motion_cfg.pair_terrain_with_motion
            and self._terrain_row_ids is not None
            and self._terrain_row_count > 0
            and self.motion.num_clips > 0
        )

        self._update_adaptive_timestep_failure_stats_before_resample(env_ids)
        self._record_hmi_completed_goal_outcomes(env_ids)

        if use_fixed_tile_layout:
            row_count = max(1, int(self._terrain_row_count))
            tile_capacity = row_count * int(self.motion.num_clips)
            tile_ids = torch.remainder(env_ids, tile_capacity)
            self.clip_ids[env_ids] = torch.div(tile_ids, row_count, rounding_mode="floor")
            self._terrain_row_ids[env_ids] = torch.remainder(tile_ids, row_count)
        else:
            if self._forced_clip_idx is not None:
                self.clip_ids[env_ids] = int(self._forced_clip_idx)
            elif self._fixed_clip_ids is not None:
                self.clip_ids[env_ids] = self._fixed_clip_ids[env_ids]
            elif self._debug_representative_clip_ids is not None and self._debug_representative_clip_ids.numel() > 0:
                reps = self._debug_representative_clip_ids
                self.clip_ids[env_ids] = reps[env_ids % reps.numel()]
            elif self.multi_clip:
                self._update_clip_success_stats(env_ids)
                if self._env.is_evaluating:
                    self.clip_ids[env_ids] = 0
                elif self._fixed_clip_group_env_mask is not None:
                    self.clip_ids[env_ids] = self._sample_fixed_clip_group_ids(env_ids)
                else:
                    if self._clip_sampling_weights is None:
                        self.clip_ids[env_ids] = torch.randint(
                            0, self.motion.num_clips, (env_ids.numel(),), device=self.device
                        )
                    else:
                        self.clip_ids[env_ids] = torch.multinomial(
                            self._clip_sampling_weights, env_ids.numel(), replacement=True
                        )
            else:
                self.clip_ids[env_ids] = 0

            if self._terrain_row_ids is not None:
                if self._env.is_evaluating or self._terrain_row_count <= 1:
                    self._terrain_row_ids[env_ids] = 0
                else:
                    self._terrain_row_ids[env_ids] = torch.randint(
                        0, self._terrain_row_count, (env_ids.numel(),), device=self.device
                    )

        # Task/tracking identity is sampled only at an episode boundary.  A
        # curriculum update can therefore never change an in-flight row's
        # command, reward, critic mask, or termination contract.
        self._refresh_hybrid_velocity_task_env_mask(env_ids)

        # 0. Sample the time steps.  start_at_timestep_zero_prob is applied below
        # as an explicit delta-at-zero mixture, so the base reset distribution
        # samples nonzero timesteps whenever the clip has one.
        clip_lengths = self._current_clip_lengths(env_ids)
        start_margin = self._min_start_margin_steps()
        valid_starts = torch.clamp(clip_lengths - start_margin, min=1)
        adaptive_reset_probabilities: torch.Tensor | None = None
        if self._forced_reset_timestep is not None:
            forced_timestep = torch.full_like(valid_starts, self._forced_reset_timestep)
            max_valid = torch.clamp(clip_lengths - 2, min=0)
            if torch.any(forced_timestep > max_valid):
                invalid_lengths = clip_lengths[forced_timestep > max_valid].detach().cpu().tolist()
                raise ValueError(
                    "Forced reset timestep must be at most clip_length - 2 for every selected clip: "
                    f"timestep={self._forced_reset_timestep}, invalid_clip_lengths={invalid_lengths}."
                )
            self.time_steps[env_ids] = forced_timestep
        elif self._env.is_evaluating:
            self.time_steps[env_ids] = 0
        elif self.use_adaptive_timesteps_sampler:
            clip_ids = self.clip_ids[env_ids]
            windows = None
            window_valid = None
            if self._uniform_t1_window_sampling_active():
                window_valid, lo, hi, _, _, _ = self._uniform_t1_window_bounds(clip_ids, valid_starts)
                windows = torch.stack((lo, hi), dim=-1)
            sampling_kwargs = {
                "exclude_zero": True,
                "windows": windows,
                "window_valid": window_valid,
                "window_density_boost": float(self.motion_cfg.uniform_t1_window_density_boost),
                "window_target_probability": self._uniform_t1_window_conditional_target_probability(),
            }
            sampled_steps, probabilities = (
                self.adaptive_timesteps_sampler._sample_time_steps_with_probabilities(
                    clip_ids,
                    **sampling_kwargs,
                    _trusted_inputs=True,
                )
            )
            if self._uniform_t1_window_sampling_active():
                adaptive_reset_probabilities = probabilities
            self.time_steps[env_ids] = sampled_steps
        elif self._uniform_t1_window_sampling_active():
            self.time_steps[env_ids] = self._sample_uniform_t1_window_time_steps(env_ids, valid_starts)
        else:
            nonzero_counts = torch.clamp(valid_starts - 1, min=1)
            sampled = (torch.rand(env_ids.numel(), device=self.device) * nonzero_counts).long() + 1
            self.time_steps[env_ids] = torch.where(valid_starts > 1, sampled, torch.zeros_like(sampled))

        # HMI keeps the adaptive reset distribution only for tracking rows.
        # Generation rows sample a uniform nonzero reference frame before the
        # explicit frame-zero mixture below, matching the upstream contract.
        if self.hmi_enabled() and self.use_adaptive_timesteps_sampler:
            gen_local_mask = self.get_hmi_gen_env_mask()[env_ids]
            if torch.any(gen_local_mask):
                gen_valid_starts = valid_starts[gen_local_mask]
                nonzero_counts = torch.clamp(gen_valid_starts - 1, min=1)
                sampled = (
                    torch.rand(
                        int(gen_local_mask.sum().item()), device=self.device
                    )
                    * nonzero_counts
                ).long() + 1
                self.time_steps[env_ids[gen_local_mask]] = torch.where(
                    gen_valid_starts > 1,
                    sampled,
                    torch.zeros_like(sampled),
                )

        # Handle start_at_timestep_zero_prob.
        base_prob = self._current_start_at_timestep_zero_prob()
        hmi_gen_zero_prob = (
            None
            if not self.hmi_enabled() or self.hmi_cfg is None
            else self.hmi_cfg.gen_start_at_timestep_zero_prob
        )
        if self._forced_reset_timestep is None and (
            base_prob > 0.0
            or (hmi_gen_zero_prob is not None and hmi_gen_zero_prob > 0.0)
        ):
            probs = torch.full((env_ids.numel(),), base_prob, device=self.device, dtype=torch.float32)
            if hmi_gen_zero_prob is not None:
                gen_local_mask = self.get_hmi_gen_env_mask()[env_ids]
                probs[gen_local_mask] = float(hmi_gen_zero_prob)
            probs = torch.clamp(probs, 0.0, 1.0)
            subset = self.time_steps[env_ids]
            rand_vals = torch.rand_like(subset, dtype=torch.float32)
            subset = torch.where(rand_vals < probs, torch.zeros_like(subset), subset)
            self.time_steps[env_ids] = subset

        # If the motion is at the last timestep, set it to the second last timestep;
        # Otherwise, update_tasks_callback will advance the timestep to the next timestep -> out of bounds error.
        max_valid = torch.clamp(clip_lengths - 2, min=0)
        self.time_steps[env_ids] = torch.minimum(self.time_steps[env_ids], max_valid)
        if bool(self.motion_cfg.uniform_t1_window_sampling_enabled):
            self._record_uniform_t1_window_reset_metrics(
                env_ids,
                valid_starts,
                adaptive_reset_probabilities=adaptive_reset_probabilities,
            )

        if self.motion_cfg.align_motion_to_init_yaw:
            self._update_motion_alignment(env_ids)
        self._refresh_hmi_terminal_object_goals(env_ids)
        self._clear_runtime_default_pose_prepend(env_ids)

        # 1. Get the reference root/body poses
        root_pos = self._motion_body_pos_w(env_ids)[:, 0].clone()
        root_rot = self._motion_body_quat_w(env_ids)[:, 0].clone()  # xyzw
        root_lin_vel = self._motion_body_lin_vel_w(env_ids)[:, 0].clone()
        root_ang_vel = self._motion_body_ang_vel_w(env_ids)[:, 0].clone()

        dof_pos = self._motion_joint_pos(env_ids).clone()
        dof_vel = self._motion_joint_vel(env_ids).clone()
        runtime_prepend_mask = self._runtime_default_pose_prepend_reset_mask(env_ids)

        if self._reset_to_default_pose:
            dof_pos, dof_vel, root_pos, root_rot, root_lin_vel, root_ang_vel = self._default_pose_reset_targets(env_ids)
        elif self._runtime_default_pose_prepend_enabled:
            # Reset batches are sparse.  Compute their deterministic default
            # targets once and select on device instead of synchronizing the
            # CUDA mask to Python and compacting it through boolean indexing.
            prepend_targets = self._default_pose_reset_targets(env_ids)
            scalar_mask = runtime_prepend_mask[:, None]
            dof_pos = torch.where(scalar_mask, prepend_targets[0], dof_pos)
            dof_vel = torch.where(scalar_mask, prepend_targets[1], dof_vel)
            root_pos = torch.where(scalar_mask, prepend_targets[2], root_pos)
            root_rot = torch.where(scalar_mask, prepend_targets[3], root_rot)
            root_lin_vel = torch.where(scalar_mask, prepend_targets[4], root_lin_vel)
            root_ang_vel = torch.where(scalar_mask, prepend_targets[5], root_ang_vel)

        soft_joint_pos_limits = self._env.simulator.dof_pos_limits  # type: ignore[attr-defined]  # (num_dofs, 2)
        mujoco_reset_noise_enabled = os.environ.get("HOLOSOMA_MUJOCO_RESET_NOISE", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        disable_reset_noise = (
            self._env.simulator.get_simulator_type() == SimulatorType.MUJOCO and not mujoco_reset_noise_enabled
        )
        if disable_reset_noise:
            target_dof_pos = torch.clip(dof_pos, soft_joint_pos_limits[:, 0], soft_joint_pos_limits[:, 1])
            target_dof_vel = dof_vel
            target_root_pos = root_pos
            target_root_rot = root_rot
            target_root_lin_vel = root_lin_vel
            target_root_ang_vel = root_ang_vel
        else:
            reset_noise_scale = torch.ones((env_ids.numel(), 1), device=self.device, dtype=torch.float32)
            reset_noise_scale_3 = reset_noise_scale.expand(-1, 3)

            init_pose_cfg = self._effective_initial_pose_noise_config()
            dof_pos_noise = init_pose_cfg.dof_pos * init_pose_cfg.overall_noise_scale
            dof_vel_noise = init_pose_cfg.dof_vel * init_pose_cfg.overall_noise_scale
            root_pos_noise = (
                torch.tensor(
                    init_pose_cfg.root_pos,
                    device=self.device,
                )
                * init_pose_cfg.overall_noise_scale
            )
            root_rot_noise_rpy = (
                torch.tensor(
                    init_pose_cfg.root_rot,
                    device=self.device,
                )
                * init_pose_cfg.overall_noise_scale
            )
            root_vel_noise = (
                torch.tensor(
                    init_pose_cfg.root_lin_vel,
                    device=self.device,
                )
                * init_pose_cfg.overall_noise_scale
            )
            root_ang_vel_noise_rpy = (
                torch.tensor(
                    init_pose_cfg.root_ang_vel,
                    device=self.device,
                )
                * init_pose_cfg.overall_noise_scale
            )

            target_dof_pos = dof_pos + (
                torch.rand(dof_pos.shape, device=self.device) - 0.5
            ) * 2 * dof_pos_noise * reset_noise_scale
            target_dof_pos = torch.clip(target_dof_pos, soft_joint_pos_limits[:, 0], soft_joint_pos_limits[:, 1])
            target_dof_vel = dof_vel + (
                torch.rand(dof_vel.shape, device=self.device) - 0.5
            ) * 2 * dof_vel_noise * reset_noise_scale

            target_root_pos = root_pos + (
                torch.rand(root_pos.shape, device=self.device) - 0.5
            ) * 2 * root_pos_noise.unsqueeze(0) * reset_noise_scale_3

            rand_sample_rpy = (
                (torch.rand((len(env_ids), 3), device=self.device) - 0.5)
                * 2
                * root_rot_noise_rpy.unsqueeze(0)
                * reset_noise_scale_3
            )
            orientations_delta = quat_from_euler_xyz(
                rand_sample_rpy[:, 0], rand_sample_rpy[:, 1], rand_sample_rpy[:, 2]
            )
            target_root_rot = quat_mul(orientations_delta, root_rot, w_last=True)

            target_root_lin_vel = root_lin_vel + (
                torch.rand(root_lin_vel.shape, device=self.device) - 0.5
            ) * 2 * root_vel_noise.unsqueeze(0) * reset_noise_scale_3

            target_root_ang_vel = root_ang_vel + (
                torch.rand(root_ang_vel.shape, device=self.device) - 0.5
            ) * 2 * root_ang_vel_noise_rpy.unsqueeze(0) * reset_noise_scale_3

        self._apply_hmi_step_zero_root_noise(
            env_ids,
            root_pos,
            root_rot,
            target_root_pos,
            target_root_rot,
        )

        # 3. Set the robot states in simulator
        self._env.simulator.dof_pos[env_ids] = target_dof_pos
        self._env.simulator.dof_vel[env_ids] = target_dof_vel

        self._env.simulator.robot_root_states[env_ids, :3] = target_root_pos
        self._env.simulator.robot_root_states[env_ids, 3:7] = target_root_rot
        self._env.simulator.robot_root_states[env_ids, 7:10] = target_root_lin_vel
        self._env.simulator.robot_root_states[env_ids, 10:13] = target_root_ang_vel
        self._reset_pickup_anchor_state(env_ids, root_pos_w=target_root_pos, root_quat_w=target_root_rot)

        # 4. Set the object states in simulator
        if self.motion.has_object:
            obj_pos = self._motion_object_pos_w(env_ids)
            obj_ori = self._motion_object_quat_w(env_ids)
            obj_lin_vel = self._motion_object_lin_vel_w(env_ids)

            if disable_reset_noise:
                target_obj_pos = obj_pos
            else:
                obj_pos_noise = torch.tensor(
                    [init_pose_cfg.object_pos],
                    device=self.device,
                )
                obj_pos_noise = obj_pos_noise * init_pose_cfg.overall_noise_scale
                target_obj_pos = obj_pos + (
                    (torch.rand(obj_pos.shape, device=self.device) - 0.5)
                    * 2
                    * obj_pos_noise
                    * reset_noise_scale_3
                )
            target_obj_pos, target_obj_ori = self._apply_manual_object_reset_overrides(
                target_obj_pos, obj_ori, env_ids
            )

            object_states = torch.cat(
                [target_obj_pos, target_obj_ori, obj_lin_vel, torch.zeros_like(obj_lin_vel)], dim=-1
            )  # (num_envs, 13), xyzw
            # 4.3 set active object states; inactive objects are parked away for multi-URDF banks.
            self._set_simulator_object_states(env_ids, object_states)
            self._reset_pickup_anchor_state(
                env_ids,
                root_pos_w=target_root_pos,
                root_quat_w=target_root_rot,
                object_pos_w=target_obj_pos,
                object_quat_w=target_obj_ori,
            )

        if (
            self._runtime_default_pose_prepend_enabled
            and self._runtime_default_pose_prepend_active is not None
            and self._runtime_default_pose_prepend_step is not None
        ):
            # `_clear_runtime_default_pose_prepend` already zeroed the selected
            # clocks, so assigning the device mask directly is equivalent to
            # activating its compacted IDs without a host-visible branch.
            self._runtime_default_pose_prepend_active[env_ids] = runtime_prepend_mask

        self._update_future_target_poses()

    def _has_standard_episodic_motion_end_contract(self) -> bool:
        """Return whether the standard termination manager owns clip completion.

        This deliberately accepts only the exact built-in manager, term name,
        configuration, and resolved function.  Custom managers and wrappers
        retain the rollover fallback because their call/reset ordering cannot
        be proven here.
        """

        manager = getattr(self._env, "termination_manager", None)
        if manager is None or manager.__class__ is not TerminationManager:
            return False

        terms = getattr(getattr(manager, "cfg", None), "terms", None)
        if not isinstance(terms, dict):
            return False
        term_cfg = terms.get("motion_ends")
        if term_cfg is None:
            return False
        if getattr(term_cfg, "func", None) != _STANDARD_MOTION_END_TERM_PATH:
            return False
        if getattr(term_cfg, "is_timeout", None) is not False:
            return False

        term_names = getattr(manager, "_term_names", None)
        term_funcs = getattr(manager, "_term_funcs", None)
        term_instances = getattr(manager, "_term_instances", None)
        if not isinstance(term_names, list) or term_names.count("motion_ends") != 1:
            return False
        if not isinstance(term_funcs, dict) or not isinstance(term_instances, dict):
            return False
        if "motion_ends" in term_instances:
            return False

        # The standard term imports MotionCommand, so importing it at module
        # scope would create a cycle.  A real TerminationManager has already
        # resolved the configured term; absence from sys.modules fails closed.
        standard_module = sys.modules.get("holosoma.managers.termination.terms.wbt")
        standard_motion_ends = getattr(standard_module, "motion_ends", None)
        return standard_motion_ends is not None and term_funcs.get("motion_ends") is standard_motion_ends

    def _termination_owns_clip_rollover(self) -> bool:
        """Fail closed unless episodic motion-end reset is currently active."""

        if bool(getattr(self, "_disable_clip_end_reset", True)):
            return False
        if not self._has_standard_episodic_motion_end_contract():
            return False
        if os.environ.get("HOLOSOMA_DISABLE_AUTO_RESET", "0").lower() in _ENABLED_ENV_FLAG_VALUES:
            return False
        return (
            os.environ.get("HOLOSOMA_DISABLE_MOTION_END_RESET", "0").lower() not in _ENABLED_ENV_FLAG_VALUES
        )

    def _handle_clip_rollover(self) -> None:
        """Apply the continuous-clip fallback when termination does not own it."""

        if self.hmi_enabled():
            # HMI holds the terminal reference goal through the remainder of
            # the 10-second task episode; reaching the motion end is not an
            # episode termination and must not resample the command.
            current_clip_lengths = self._current_clip_lengths()
            ended_env_ids = torch.where(self.time_steps >= current_clip_lengths)[0]
            if ended_env_ids.numel() > 0:
                self.time_steps[ended_env_ids] = torch.clamp(
                    current_clip_lengths[ended_env_ids] - 1, min=0
                )
            return

        if self._termination_owns_clip_rollover():
            # BaseTask evaluated motion_ends before resetting and entering this
            # command step.  The standard term fires at clip_length - 2, so no
            # surviving row can reach clip_length after this step's +1 advance.
            return

        current_clip_lengths = self._current_clip_lengths()
        ended_env_ids = torch.where(self.time_steps >= current_clip_lengths)[0]
        if ended_env_ids.numel() == 0:
            return
        if self._disable_clip_end_reset:
            self.time_steps[ended_env_ids] = torch.clamp(current_clip_lengths[ended_env_ids] - 1, min=0)
            return

        self.reset(ended_env_ids)
        sim = self._env.simulator
        sim.set_actor_root_state_tensor_robots(ended_env_ids, sim.robot_root_states)
        sim.set_dof_state_tensor_robots(ended_env_ids)
        sim.refresh_sim_tensors()

    def step(self) -> None:
        """called in _update_tasks_callback of the environment. (after compute_reward, before compute_observations)"""
        timing = getattr(self._env, "step_timing", None)
        if not getattr(timing, "enabled", False):
            timing = None

        # BaseTask resets terminated envs before this hook. Their old visits
        # were recorded inside reset(); only still-active envs are exposed here.
        self._record_adaptive_timestep_exposure_before_advance()

        # 0. update time steps, all motion joint/body poses are updated automatically with the time steps.
        with (timing.record("post/tasks/motion/time_advance") if timing is not None else nullcontext()):
            advance_mask = torch.ones_like(self.time_steps, dtype=torch.bool)
            if (
                self._runtime_default_pose_prepend_enabled
                and self._runtime_default_pose_prepend_active is not None
                and self._runtime_default_pose_prepend_step is not None
            ):
                active_mask = self._runtime_default_pose_prepend_active
                advance_mask = advance_mask & ~active_mask
                last_step_mask = active_mask & (
                    self._runtime_default_pose_prepend_step >= (self._runtime_default_pose_prepend_steps - 1)
                )
                keep_warmup_mask = active_mask & ~last_step_mask
                self._runtime_default_pose_prepend_step.add_(
                    keep_warmup_mask.to(dtype=self._runtime_default_pose_prepend_step.dtype)
                )
                self._runtime_default_pose_prepend_active.logical_and_(~last_step_mask)

            # Handle freeze_at_timestep_zero_prob: for envs at timestep 0, randomly decide whether to advance
            freeze_prob = self._current_freeze_at_timestep_zero_prob()
            if freeze_prob > 0.0:
                zero_mask = self.time_steps == 0
                if zero_mask.any():
                    rand_vals = torch.rand(self.num_envs, device=self.device)
                    freeze_mask = (rand_vals < freeze_prob) & zero_mask
                    advance_mask = advance_mask & ~freeze_mask

            self.time_steps += advance_mask.long()

        # Match BeyondMimic-style clip rollover: once a clip ends, reset only the
        # motion/object state for those envs instead of terminating the episode.
        with (timing.record("post/tasks/motion/clip_rollover") if timing is not None else nullcontext()):
            self._handle_clip_rollover()

        # Evaluation-only two-phase sparse-root command.  The object snapshot
        # was refreshed after physics and before this command hook, so a newly
        # triggered value is visible in the observation computed immediately
        # after this method returns.  Training never enters this branch.
        self._update_manual_forward_after_lift()
        self._update_manual_forward_heading_lock()

        # 1. update body_pos_relative_w and body_quat_relative_w
        # definition of body_pos/quat_relative_w:
        # If I take this motion data and adapt it to where my robot currently is
        # (accounting for position(x, y) offset and yaw difference of a reference body),
        # what should each body part's target pose be?

        ## 1.0 get the reference body poses

        # Issue (This is a isaacgym only issue.):
        # ------------------------------------------------------------
        # In isaacgym, immediately after reset (self._env.episode_length_buf == 0), calling
        # simulator.set_actor_root_state_tensor and simulator.set_dof_state_tensor will reset
        # the robot_root_pos_w and robot_root_quat_w successfully.
        # However, the robot_body_pos_w and robot_body_quat_w are not updated successfully,
        # (since kinematic forward has not been applied yet).
        # Therefore, using robot_ref_pos_w and robot_ref_quat_w as reference body poses is not resetted correctly.

        # Solution:
        # ------------------------------------------------------------
        # if episode_length_buf == 0, use robot_root_pos_w and robot_root_quat_w as reference body.
        # else, use configured reference body as reference body.
        with (timing.record("post/tasks/motion/relative_body_pose") if timing is not None else nullcontext()):
            use_root = (self._env.episode_length_buf == 0).unsqueeze(1).float()

            ref_pos_w = self.root_pos_w * use_root + self.ref_pos_w * (1 - use_root)
            ref_quat_w = self.root_quat_w * use_root + self.ref_quat_w * (1 - use_root)
            robot_ref_pos_w = self.robot_root_pos_w * use_root + self.robot_ref_pos_w * (1 - use_root)
            robot_ref_quat_w = self.robot_root_quat_w * use_root + self.robot_ref_quat_w * (1 - use_root)

            ## 1.1 compute the relative body poses
            delta_quat_w = yaw_quat(
                quat_mul(robot_ref_quat_w, quat_inverse(ref_quat_w, w_last=True), w_last=True),
                w_last=True,
            )
            ### 1.1.1 body_quat_relative_w
            self.body_quat_relative_w = quat_mul_broadcast_left(delta_quat_w, self.body_quat_w, w_last=True)
            ### 1.1.2 body_pos_relative_w
            delta_pos_w_height = ref_pos_w - robot_ref_pos_w
            delta_pos_w_height[..., :2] = 0.0  # adjusting for height differences
            self.body_pos_relative_w = (
                robot_ref_pos_w[:, None, :]
                + delta_pos_w_height[:, None, :]
                + quat_apply_broadcast_left(delta_quat_w, self.body_pos_w - ref_pos_w[:, None, :], w_last=True)
            )

        ### 1.3 update the adaptive timesteps sampler
        with (timing.record("post/tasks/motion/adaptive_sampler") if timing is not None else nullcontext()):
            if self.use_adaptive_timesteps_sampler:
                self.adaptive_timesteps_sampler.update_bin_failed_count()

        with (timing.record("post/tasks/motion/future_targets") if timing is not None else nullcontext()):
            self._update_future_target_poses()
        with (timing.record("post/tasks/motion/pickup_anchor") if timing is not None else nullcontext()):
            self._update_pickup_anchor_state()
        with (timing.record("post/tasks/motion/contact_prior") if timing is not None else nullcontext()):
            self._update_contact_prior_state()

    def configure_manual_forward_after_lift(
        self,
        *,
        command_m: float,
        rel_z_delta_m: float,
        consecutive_steps: int,
        preserve_native_contact_buttons: bool = False,
        preserve_native_pickup_button: bool = False,
        preserve_native_drop_button: bool = False,
        heading_lock: bool = False,
        command_semantics: str = "legacy_constant_robot_heading_frame",
    ) -> None:
        """Hold a zero manual root command, then switch after a stable lift.

        This is intentionally configured at runtime by the single-environment
        evaluation recorder.  It does not alter the training configuration or
        add to the reference root command: manual mode replaces the actor's
        sparse-root command with zero until the object has remained above its
        initial world-z by ``rel_z_delta_m`` for ``consecutive_steps`` control
        steps.  By default the replacement remains the deployment-faithful
        constant ``[command_m, 0, 0]``.  The optional ``heading_lock`` mode is
        a simulator diagnostic: it latches world heading at the transition
        and recomputes the same actor slots every control step.  It must not
        be presented as policy-only behavior or used for deployment-faithful
        evaluation.
        """

        command = float(command_m)
        lift_delta = float(rel_z_delta_m)
        stable_steps = int(consecutive_steps)
        semantics = str(command_semantics).strip().lower()
        allowed_semantics = {
            "legacy_constant_robot_heading_frame",
            "robot_heading_velocity_mps",
            "world_velocity_mps",
            "world_root_error_m",
        }
        if semantics not in allowed_semantics:
            raise ValueError(
                "command_semantics must identify the checkpoint's actor command "
                f"contract, got {command_semantics!r}."
            )
        if type(heading_lock) is not bool:
            raise ValueError(f"heading_lock must be a bool, got {heading_lock!r}.")
        if type(preserve_native_contact_buttons) is not bool:
            raise ValueError(
                "preserve_native_contact_buttons must be a bool, got "
                f"{preserve_native_contact_buttons!r}."
            )
        if type(preserve_native_pickup_button) is not bool:
            raise ValueError(
                "preserve_native_pickup_button must be a bool, got "
                f"{preserve_native_pickup_button!r}."
            )
        if type(preserve_native_drop_button) is not bool:
            raise ValueError(
                "preserve_native_drop_button must be a bool, got "
                f"{preserve_native_drop_button!r}."
            )
        if heading_lock and semantics != "legacy_constant_robot_heading_frame":
            raise ValueError(
                "heading_lock is defined only for the legacy heading-frame "
                "relative-pose command."
            )
        if not np.isfinite(command):
            raise ValueError(f"command_m must be finite, got {command_m!r}.")
        if not np.isfinite(lift_delta) or lift_delta <= 0.0:
            raise ValueError(f"rel_z_delta_m must be finite and positive, got {rel_z_delta_m!r}.")
        if isinstance(consecutive_steps, bool) or stable_steps < 0 or stable_steps != consecutive_steps:
            raise ValueError(f"consecutive_steps must be a non-negative integer, got {consecutive_steps!r}.")
        if not self.motion.has_object:
            raise RuntimeError("manual forward-after-lift requires an object motion.")
        if self.manual_xy_rel is None or self.manual_yaw_rel is None or self.manual_drop_button is None:
            raise RuntimeError("MotionCommand manual-control tensors are not initialized.")

        current_object_z = self.simulator_object_state_snapshot[:, 2].detach().clone()
        if current_object_z.shape != (self.num_envs,):
            raise RuntimeError(
                "Active-object z snapshot must contain exactly one value per environment, "
                f"got {tuple(current_object_z.shape)}."
            )

        self.manual_control_enabled = True
        self.manual_xy_rel.zero_()
        self.manual_yaw_rel.zero_()
        self.manual_pickup_button_override_enabled = False
        preserve_pickup = (
            preserve_native_contact_buttons or preserve_native_pickup_button
        )
        preserve_drop = preserve_native_contact_buttons or preserve_native_drop_button
        self.manual_drop_button_override_enabled = not preserve_drop
        if self.manual_drop_button_override_enabled:
            self.manual_drop_button.zero_()

        self._manual_forward_after_lift_enabled = True
        self._manual_forward_after_lift_command_m = command
        self._manual_forward_after_lift_rel_z_delta_m = lift_delta
        self._manual_forward_after_lift_consecutive_steps = stable_steps
        self._manual_forward_after_lift_preserve_native_contact_buttons = (
            preserve_native_contact_buttons
        )
        self._manual_forward_after_lift_preserve_native_pickup_button = preserve_pickup
        self._manual_forward_after_lift_preserve_native_drop_button = preserve_drop
        self._manual_forward_after_lift_command_semantics = semantics
        self._manual_forward_after_lift_baseline_object_z = current_object_z
        self._manual_forward_after_lift_consecutive_count = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=torch.long,
        )
        self._manual_forward_after_lift_triggered = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=torch.bool,
        )
        self._manual_forward_after_lift_trigger_episode_step = torch.full(
            (self.num_envs,),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        if heading_lock:
            self._prepare_manual_forward_heading_lock(command)
        else:
            # Deployment-faithful mode: reproduce the externally supplied
            # constant body-frame command without simulator-state feedback.
            self._manual_forward_heading_lock_enabled = False
            self._manual_forward_heading_lock_command_m = command
            self._manual_forward_heading_lock_active = None
            self._manual_forward_heading_lock_origin_xy_w = None
            self._manual_forward_heading_lock_yaw_w = None

    def configure_manual_heading_locked_forward(self, *, command_m: float) -> None:
        """Enable an immediate world-heading-locked manual forward command.

        ``command_m`` preserves the existing external command and actor
        interfaces.  It specifies a constant lookahead along the robot's world
        heading at configuration time.  The actor still receives exactly
        ``[dx, dy, dyaw]`` in its current heading frame.
        """

        command = float(command_m)
        if not np.isfinite(command):
            raise ValueError(f"command_m must be finite, got {command_m!r}.")
        if self.manual_xy_rel is None or self.manual_yaw_rel is None:
            raise RuntimeError("MotionCommand manual-control tensors are not initialized.")

        self.manual_control_enabled = True
        self.manual_xy_rel.zero_()
        self.manual_yaw_rel.zero_()
        self._prepare_manual_forward_heading_lock(command)
        active = self._manual_forward_heading_lock_active
        if active is None:
            raise RuntimeError("Manual forward heading-lock state was not initialized.")
        active.fill_(True)
        self._capture_manual_forward_heading_lock(active)
        self._update_manual_forward_heading_lock()

    def _prepare_manual_forward_heading_lock(self, command_m: float) -> None:
        self._manual_forward_heading_lock_enabled = True
        self._manual_forward_heading_lock_command_m = float(command_m)
        self._manual_forward_heading_lock_active = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=torch.bool,
        )
        self._manual_forward_heading_lock_origin_xy_w = torch.zeros(
            (self.num_envs, 2),
            device=self.device,
            dtype=torch.float32,
        )
        self._manual_forward_heading_lock_yaw_w = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=torch.float32,
        )

    def _capture_manual_forward_heading_lock(self, env_mask: torch.Tensor) -> None:
        active = self._manual_forward_heading_lock_active
        origin_xy_w = self._manual_forward_heading_lock_origin_xy_w
        anchor_yaw_w = self._manual_forward_heading_lock_yaw_w
        if active is None or origin_xy_w is None or anchor_yaw_w is None:
            raise RuntimeError("Manual forward heading-lock state is incomplete.")
        if env_mask.shape != (self.num_envs,) or env_mask.dtype != torch.bool:
            raise RuntimeError(
                "Manual forward heading-lock activation mask must be bool[num_envs], "
                f"got dtype={env_mask.dtype}, shape={tuple(env_mask.shape)}."
            )
        if not torch.any(env_mask):
            return
        origin_xy_w[env_mask] = self.robot_root_pos_w[env_mask, :2].detach()
        anchor_yaw_w[env_mask] = calc_heading(self.robot_root_quat_w[env_mask]).detach()

    def _update_manual_forward_heading_lock(self) -> None:
        if not self._manual_forward_heading_lock_enabled:
            return
        active = self._manual_forward_heading_lock_active
        anchor_yaw_w = self._manual_forward_heading_lock_yaw_w
        if active is None or anchor_yaw_w is None:
            raise RuntimeError("Manual forward heading-lock state is incomplete.")
        if self.manual_xy_rel is None or self.manual_yaw_rel is None:
            raise RuntimeError("MotionCommand manual-control tensors are not initialized.")
        if not torch.any(active):
            return

        current_yaw_w = calc_heading(self.robot_root_quat_w)
        heading_error = normalize_angle(anchor_yaw_w - current_yaw_w)
        command = self._manual_forward_heading_lock_command_m

        # Express a constant world-heading lookahead in the robot's current
        # heading frame.  A yaw disturbance therefore produces both a lateral
        # translation component and a restoring relative-yaw command while the
        # actor input shape/order stays unchanged.
        self.manual_xy_rel[active, 0] = command * torch.cos(heading_error[active])
        self.manual_xy_rel[active, 1] = command * torch.sin(heading_error[active])
        self.manual_yaw_rel[active, 0] = heading_error[active]

    def _update_manual_forward_after_lift(self) -> None:
        if not self._manual_forward_after_lift_enabled:
            return
        baseline = self._manual_forward_after_lift_baseline_object_z
        counter = self._manual_forward_after_lift_consecutive_count
        triggered = self._manual_forward_after_lift_triggered
        trigger_step = self._manual_forward_after_lift_trigger_episode_step
        if baseline is None or counter is None or triggered is None or trigger_step is None:
            raise RuntimeError("manual forward-after-lift state is incomplete.")
        heading_lock_active = self._manual_forward_heading_lock_active
        if self._manual_forward_heading_lock_enabled and heading_lock_active is None:
            raise RuntimeError("manual forward-after-lift heading-lock state is incomplete.")
        if self.manual_xy_rel is None or self.manual_yaw_rel is None:
            raise RuntimeError("MotionCommand manual-control tensors are not initialized.")

        object_z = self.simulator_object_state_snapshot[:, 2]
        above_threshold = (object_z - baseline) >= self._manual_forward_after_lift_rel_z_delta_m
        waiting = ~triggered
        counter.copy_(torch.where(waiting & above_threshold, counter + 1, torch.zeros_like(counter)))
        if self._manual_forward_after_lift_consecutive_steps == 0:
            # Zero disables debounce, but it must not bypass the lift threshold:
            # trigger on the first threshold-qualified control step.
            newly_triggered = waiting & above_threshold
        else:
            newly_triggered = waiting & (counter >= self._manual_forward_after_lift_consecutive_steps)
        if torch.any(newly_triggered):
            if self._manual_forward_heading_lock_enabled:
                if heading_lock_active is None:
                    raise RuntimeError("manual forward-after-lift heading-lock state is incomplete.")
                heading_lock_active.logical_or_(newly_triggered)
                self._capture_manual_forward_heading_lock(newly_triggered)
            else:
                self.manual_xy_rel[newly_triggered, 0] = self._manual_forward_after_lift_command_m
                self.manual_xy_rel[newly_triggered, 1] = 0.0
                self.manual_yaw_rel[newly_triggered, 0] = 0.0
            triggered.logical_or_(newly_triggered)
            trigger_step[newly_triggered] = self._env.episode_length_buf[newly_triggered].to(dtype=torch.long)
            if self._manual_forward_heading_lock_enabled:
                self._update_manual_forward_heading_lock()

    def get_manual_forward_after_lift_status(self, env_id: int = 0) -> dict[str, Any] | None:
        """Return a compact audit record for the evaluation recorder."""

        if not self._manual_forward_after_lift_enabled:
            return None
        if env_id < 0 or env_id >= self.num_envs:
            raise IndexError(f"env_id {env_id} is outside [0, {self.num_envs}).")
        baseline = self._manual_forward_after_lift_baseline_object_z
        counter = self._manual_forward_after_lift_consecutive_count
        triggered = self._manual_forward_after_lift_triggered
        trigger_step = self._manual_forward_after_lift_trigger_episode_step
        if baseline is None or counter is None or triggered is None or trigger_step is None:
            raise RuntimeError("manual forward-after-lift state is incomplete.")
        object_z = self.simulator_object_state_snapshot[env_id, 2]
        heading_lock_status = self.get_manual_forward_heading_lock_status(env_id)
        return {
            "phase": "forward" if bool(triggered[env_id].item()) else "pickup_zero",
            "command_semantics": (
                "world_heading_locked_lookahead"
                if self._manual_forward_heading_lock_enabled
                else self._manual_forward_after_lift_command_semantics
            ),
            "configured_forward_command_m": float(self._manual_forward_after_lift_command_m),
            "active_forward_command_m": float(self.manual_xy_rel[env_id, 0].item()),
            "rel_z_delta_m": float(object_z.item() - baseline[env_id].item()),
            "trigger_rel_z_delta_m": float(self._manual_forward_after_lift_rel_z_delta_m),
            "consecutive_count": int(counter[env_id].item()),
            "required_consecutive_steps": int(self._manual_forward_after_lift_consecutive_steps),
            "preserve_native_contact_buttons": bool(
                self._manual_forward_after_lift_preserve_native_contact_buttons
            ),
            "preserve_native_pickup_button": bool(
                self._manual_forward_after_lift_preserve_native_pickup_button
            ),
            "preserve_native_drop_button": bool(
                self._manual_forward_after_lift_preserve_native_drop_button
            ),
            "triggered": bool(triggered[env_id].item()),
            "trigger_episode_step": int(trigger_step[env_id].item()),
            "heading_lock": heading_lock_status,
        }

    def get_manual_forward_heading_lock_status(self, env_id: int = 0) -> dict[str, Any] | None:
        """Return the external command and effective actor command for audit."""

        if not self._manual_forward_heading_lock_enabled:
            return None
        if env_id < 0 or env_id >= self.num_envs:
            raise IndexError(f"env_id {env_id} is outside [0, {self.num_envs}).")
        active = self._manual_forward_heading_lock_active
        origin_xy_w = self._manual_forward_heading_lock_origin_xy_w
        anchor_yaw_w = self._manual_forward_heading_lock_yaw_w
        if active is None or origin_xy_w is None or anchor_yaw_w is None:
            raise RuntimeError("Manual forward heading-lock state is incomplete.")

        is_active = bool(active[env_id].item())
        status: dict[str, Any] = {
            "semantics": "world_heading_locked_lookahead",
            "active": is_active,
            "configured_forward_command_m": float(self._manual_forward_heading_lock_command_m),
            "actor_command_xy_m": self.manual_xy_rel[env_id].detach().cpu().tolist(),
            "actor_command_yaw_rad": float(self.manual_yaw_rel[env_id, 0].item()),
        }
        if not is_active:
            return status

        current_xy_w = self.robot_root_pos_w[env_id, :2]
        anchor_yaw = anchor_yaw_w[env_id]
        current_yaw = calc_heading(self.robot_root_quat_w[env_id : env_id + 1])[0]
        displacement = current_xy_w - origin_xy_w[env_id]
        forward_w = torch.stack((torch.cos(anchor_yaw), torch.sin(anchor_yaw)))
        left_w = torch.stack((-torch.sin(anchor_yaw), torch.cos(anchor_yaw)))
        status.update(
            {
                "anchor_yaw_w_rad": float(anchor_yaw.item()),
                "heading_error_rad": float(normalize_angle(anchor_yaw - current_yaw).item()),
                "along_track_displacement_m": float(torch.dot(displacement, forward_w).item()),
                "cross_track_error_m": float(torch.dot(displacement, left_w).item()),
            }
        )
        return status

    def _current_clip_lengths(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        clip_ids = self.clip_ids if env_ids is None else self.clip_ids[env_ids]
        return self.motion.clip_lengths[clip_ids]

    def _get_motion_indices(self, steps: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if self.motion.num_clips <= 1:
            return steps
        clip_ids = self.clip_ids if env_ids is None else self.clip_ids[env_ids]
        offsets = self.motion.clip_offsets[clip_ids]
        if steps.ndim > offsets.ndim:
            offsets = offsets.view(-1, *([1] * (steps.ndim - 1)))
        return offsets + steps

    def _clear_runtime_default_pose_prepend(self, env_ids: torch.Tensor) -> None:
        if (
            not self._runtime_default_pose_prepend_enabled
            or self._runtime_default_pose_prepend_active is None
            or self._runtime_default_pose_prepend_step is None
        ):
            return
        self._runtime_default_pose_prepend_active[env_ids] = False
        self._runtime_default_pose_prepend_step[env_ids] = 0

    def _runtime_default_pose_prepend_reset_mask(self, env_ids: torch.Tensor) -> torch.Tensor:
        if not self._runtime_default_pose_prepend_enabled:
            return torch.zeros((env_ids.numel(),), device=self.device, dtype=torch.bool)
        return self.time_steps[env_ids] == 0

    def _activate_runtime_default_pose_prepend(self, env_ids: torch.Tensor) -> None:
        if (
            env_ids.numel() == 0
            or not self._runtime_default_pose_prepend_enabled
            or self._runtime_default_pose_prepend_active is None
            or self._runtime_default_pose_prepend_step is None
        ):
            return
        self._runtime_default_pose_prepend_active[env_ids] = True
        self._runtime_default_pose_prepend_step[env_ids] = 0

    def get_runtime_default_pose_prepend_mask(self) -> torch.Tensor:
        if self._runtime_default_pose_prepend_active is None:
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        return self._runtime_default_pose_prepend_active

    def _runtime_default_pose_prepend_alpha(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self._runtime_default_pose_prepend_step is not None
        step = self._runtime_default_pose_prepend_step
        if env_ids is not None:
            step = step[env_ids]
        alpha = step.to(dtype=torch.float32)
        return alpha / float(self._runtime_default_pose_prepend_steps)

    def _blend_runtime_default_pose_prepend_lerp(
        self,
        current: torch.Tensor,
        key: str,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._runtime_default_pose_prepend_enabled:
            return current
        defaults = self._runtime_default_pose_prepend_defaults.get(key)
        if defaults is None:
            return current
        assert self._runtime_default_pose_prepend_active is not None
        clip_ids = self.clip_ids if env_ids is None else self.clip_ids[env_ids]
        active = (
            self._runtime_default_pose_prepend_active
            if env_ids is None
            else self._runtime_default_pose_prepend_active[env_ids]
        )
        alpha = self._runtime_default_pose_prepend_alpha(env_ids)
        alpha_view = alpha.view(-1, *([1] * (current.ndim - 1)))
        active_view = active.view(-1, *([1] * (current.ndim - 1)))
        starts = defaults[clip_ids]
        blended = starts + alpha_view * (current - starts)
        return torch.where(active_view, blended, current)

    def _blend_runtime_default_pose_prepend_quat(
        self,
        current: torch.Tensor,
        key: str,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._runtime_default_pose_prepend_enabled:
            return current
        defaults = self._runtime_default_pose_prepend_defaults.get(key)
        if defaults is None:
            return current
        assert self._runtime_default_pose_prepend_active is not None
        clip_ids = self.clip_ids if env_ids is None else self.clip_ids[env_ids]
        active = (
            self._runtime_default_pose_prepend_active
            if env_ids is None
            else self._runtime_default_pose_prepend_active[env_ids]
        )
        start = defaults[clip_ids]
        alpha = self._runtime_default_pose_prepend_alpha(env_ids)

        if current.ndim == 2:
            blended = slerp(start, current, alpha.unsqueeze(-1))
        elif current.ndim == 3:
            alpha_flat = alpha.unsqueeze(1).expand(-1, start.shape[1]).reshape(-1, 1)
            blended = slerp(
                start.reshape(-1, 4),
                current.reshape(-1, 4),
                alpha_flat,
            ).view_as(start)
        else:
            raise ValueError(f"Unsupported quaternion tensor rank {current.ndim}.")

        active_view = active.view(-1, *([1] * (current.ndim - 1)))
        return torch.where(active_view, blended, current)

    def _raw_motion_joint_pos(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        steps = self.time_steps if env_ids is None else self.time_steps[env_ids]
        motion_idx = self._get_motion_indices(steps, env_ids)
        joint_pos = self.motion.joint_pos[motion_idx]
        return self._blend_runtime_default_pose_prepend_lerp(joint_pos, "joint_pos", env_ids)

    def _raw_motion_joint_vel(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        steps = self.time_steps if env_ids is None else self.time_steps[env_ids]
        motion_idx = self._get_motion_indices(steps, env_ids)
        joint_vel = self.motion.joint_vel[motion_idx]
        return self._blend_runtime_default_pose_prepend_lerp(joint_vel, "joint_vel", env_ids)

    def _raw_motion_body_pos_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        steps = self.time_steps if env_ids is None else self.time_steps[env_ids]
        motion_idx = self._get_motion_indices(steps, env_ids)
        body_pos = self.motion.body_pos_w[motion_idx]
        return self._blend_runtime_default_pose_prepend_lerp(body_pos, "body_pos", env_ids)

    def _raw_motion_body_quat_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        steps = self.time_steps if env_ids is None else self.time_steps[env_ids]
        motion_idx = self._get_motion_indices(steps, env_ids)
        body_quat = self.motion.body_quat_w[motion_idx]
        return self._blend_runtime_default_pose_prepend_quat(body_quat, "body_quat", env_ids)

    def _raw_motion_body_lin_vel_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        steps = self.time_steps if env_ids is None else self.time_steps[env_ids]
        motion_idx = self._get_motion_indices(steps, env_ids)
        body_lin_vel = self.motion.body_lin_vel_w[motion_idx]
        return self._blend_runtime_default_pose_prepend_lerp(body_lin_vel, "body_lin_vel", env_ids)

    def _raw_motion_body_ang_vel_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        steps = self.time_steps if env_ids is None else self.time_steps[env_ids]
        motion_idx = self._get_motion_indices(steps, env_ids)
        body_ang_vel = self.motion.body_ang_vel_w[motion_idx]
        return self._blend_runtime_default_pose_prepend_lerp(body_ang_vel, "body_ang_vel", env_ids)

    def _raw_motion_object_pos_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if not self.motion.has_object:
            count = self.num_envs if env_ids is None else env_ids.numel()
            return torch.zeros(count, 3, device=self.device, dtype=torch.float32)
        steps = self.time_steps if env_ids is None else self.time_steps[env_ids]
        motion_idx = self._get_motion_indices(steps, env_ids)
        object_pos = self.motion.object_pos_w[motion_idx]
        return self._blend_runtime_default_pose_prepend_lerp(object_pos, "object_pos", env_ids)

    def _raw_motion_object_quat_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if not self.motion.has_object:
            count = self.num_envs if env_ids is None else env_ids.numel()
            quat = torch.zeros(count, 4, device=self.device, dtype=torch.float32)
            quat[:, 3] = 1.0
            return quat
        steps = self.time_steps if env_ids is None else self.time_steps[env_ids]
        motion_idx = self._get_motion_indices(steps, env_ids)
        object_quat = self.motion.object_quat_w[motion_idx]
        return self._blend_runtime_default_pose_prepend_quat(object_quat, "object_quat", env_ids)

    def _raw_motion_object_lin_vel_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if not self.motion.has_object:
            count = self.num_envs if env_ids is None else env_ids.numel()
            return torch.zeros(count, 3, device=self.device, dtype=torch.float32)
        steps = self.time_steps if env_ids is None else self.time_steps[env_ids]
        motion_idx = self._get_motion_indices(steps, env_ids)
        object_lin_vel = self.motion.object_lin_vel_w[motion_idx]
        return self._blend_runtime_default_pose_prepend_lerp(object_lin_vel, "object_lin_vel", env_ids)

    @property
    def current_clip_lengths(self) -> torch.Tensor:
        return self._current_clip_lengths()

    def motion_end_mask(self) -> torch.Tensor:
        clip_lengths = self._current_clip_lengths()
        return self.time_steps >= (clip_lengths - 2)

    def _min_start_margin_steps(self) -> int:
        """Ensure enough frames for stepping + future target poses."""
        return max(2, int(self.num_future_steps))

    def _valid_start_counts(self) -> torch.Tensor:
        margin = self._min_start_margin_steps()
        valid = self.motion.clip_lengths - margin
        valid = torch.clamp(valid, min=1)
        return valid.to(dtype=torch.float32)

    def _configure_clean_noisy_clip_curriculum(self) -> None:
        if not self.multi_clip:
            self._clean_noisy_clip_curriculum_enabled = False
            self._clean_clip_mask = None
            self._noisy_clip_mask = None
            return

        cfg = self._clean_noisy_clip_curriculum_cfg
        if cfg is None or not cfg.enabled:
            self._clean_noisy_clip_curriculum_enabled = False
            self._clean_clip_mask = None
            self._noisy_clip_mask = None
            return

        clean_mask = build_prefix_mask(self.motion.clip_ids, cfg.clean_clip_name_prefixes).to(device=self.device)
        noisy_mask = ~clean_mask
        if not torch.any(clean_mask):
            logger.warning(
                "clean_noisy_clip_curriculum is enabled but no clips matched clean prefixes {}. Disabling it.",
                cfg.clean_clip_name_prefixes,
            )
            self._clean_noisy_clip_curriculum_enabled = False
            self._clean_clip_mask = None
            self._noisy_clip_mask = None
            return
        if not torch.any(noisy_mask):
            logger.warning(
                "clean_noisy_clip_curriculum is enabled but all clips matched clean prefixes {}. Disabling it.",
                cfg.clean_clip_name_prefixes,
            )
            self._clean_noisy_clip_curriculum_enabled = False
            self._clean_clip_mask = None
            self._noisy_clip_mask = None
            return

        if len(cfg.stage_start_iterations) != len(cfg.clean_group_probabilities):
            raise ValueError(
                "clean_noisy_clip_curriculum.stage_start_iterations and clean_group_probabilities "
                f"must have the same length, got {len(cfg.stage_start_iterations)} and "
                f"{len(cfg.clean_group_probabilities)}."
            )
        if not cfg.stage_start_iterations:
            raise ValueError("clean_noisy_clip_curriculum requires at least one schedule stage.")
        if any(value < 0.0 or value > 1.0 for value in cfg.clean_group_probabilities):
            raise ValueError(
                "clean_noisy_clip_curriculum.clean_group_probabilities must stay in [0, 1], "
                f"got {cfg.clean_group_probabilities}."
            )

        self._clean_clip_mask = clean_mask
        self._noisy_clip_mask = noisy_mask
        logger.info(
            "Enabled clean/noisy clip curriculum: {} clean clips, {} noisy clips, stages={} probs={}.",
            int(clean_mask.sum().item()),
            int(noisy_mask.sum().item()),
            list(cfg.stage_start_iterations),
            [float(value) for value in cfg.clean_group_probabilities],
        )

    def _current_clean_group_probability(self) -> float | None:
        cfg = self._clean_noisy_clip_curriculum_cfg
        if not self._clean_noisy_clip_curriculum_enabled or cfg is None:
            return None
        return piecewise_constant_schedule_value(
            self._training_iteration,
            cfg.stage_start_iterations,
            cfg.clean_group_probabilities,
        )

    def _refresh_current_clip_sampling_weights(self) -> None:
        if not self.multi_clip:
            return

        if self._raw_clip_sampling_weights is None:
            return

        weights = self._raw_clip_sampling_weights
        if self._clean_noisy_clip_curriculum_enabled and self._clean_clip_mask is not None:
            clean_prob = self._current_clean_group_probability()
            if clean_prob is not None:
                weights = project_group_weights(
                    weights,
                    clean_mask=self._clean_clip_mask,
                    clean_group_probability=clean_prob,
                )
        total = torch.sum(weights)
        if torch.isfinite(total) and total.item() > 0.0:
            self._clip_sampling_weights = weights / total
        else:
            self._clip_sampling_weights = None

    def get_clean_noisy_clip_curriculum_log_state(self) -> dict[str, float]:
        """Return scalar clean/noisy curriculum metrics for training logs."""
        clean_prob = self._current_clean_group_probability()
        if clean_prob is None or self._clean_clip_mask is None or self._clip_sampling_weights is None:
            return {}
        clean_weight = float(self._clip_sampling_weights[self._clean_clip_mask].sum().item())
        return {
            "clean_clip_target_prob": float(clean_prob),
            "clean_clip_sample_weight": clean_weight,
            "noisy_clip_sample_weight": max(0.0, 1.0 - clean_weight),
        }

    def _init_clip_sampling(self) -> None:
        if not self.multi_clip:
            return
        if self._fixed_clip_ids is not None:
            self._clean_noisy_clip_curriculum_enabled = False
            self._clean_clip_mask = None
            self._noisy_clip_mask = None
            logger.info(
                "Fixed env-to-clip assignment is active; bypassing clip-level weighting curricula. "
                "Only within-clip timestep curriculum remains enabled."
            )
            return
        if self._fixed_clip_group_env_mask is not None:
            self._clean_noisy_clip_curriculum_enabled = False
            self._clean_clip_mask = None
            self._noisy_clip_mask = None
            logger.info(
                "Fixed clip-group assignment is active; bypassing clean/noisy clip-level weighting curricula. "
                "Clip weighting strategy '{}' is still applied within each group.",
                self.clip_weighting_strategy,
            )
        else:
            self._configure_clean_noisy_clip_curriculum()
        strategy = self.clip_weighting_strategy
        if strategy == "uniform_step":
            weights = self._valid_start_counts()
        elif strategy in ("uniform_clip", "success_rate_adaptive"):
            weights = torch.ones(self.motion.num_clips, device=self.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown clip_weighting_strategy '{strategy}'.")

        if self._rank_local_inverse_cover_weights is not None:
            weights = weights * self._rank_local_inverse_cover_weights

        weights = weights / weights.sum()
        self._raw_clip_sampling_weights = weights

        if strategy == "success_rate_adaptive":
            self._base_clip_weights = weights.clone()
            self._clip_success_counts = torch.zeros(self.motion.num_clips, device=self.device)
            self._clip_total_counts = torch.zeros(self.motion.num_clips, device=self.device)
        self._refresh_current_clip_sampling_weights()

    def _sample_clip_ids_from_mask(self, clip_mask: torch.Tensor, num_samples: int) -> torch.Tensor:
        if num_samples <= 0:
            return torch.empty((0,), device=self.device, dtype=torch.long)
        clip_indices = torch.nonzero(clip_mask, as_tuple=False).flatten().to(device=self.device, dtype=torch.long)
        if clip_indices.numel() == 0:
            raise RuntimeError("Cannot sample fixed clip group because its clip mask is empty.")
        if self._clip_sampling_weights is None:
            sampled_local_ids = torch.randint(0, clip_indices.numel(), (num_samples,), device=self.device)
            return clip_indices[sampled_local_ids]

        group_weights = self._clip_sampling_weights[clip_indices]
        total = torch.sum(group_weights)
        if not torch.isfinite(total) or total.item() <= 0.0:
            sampled_local_ids = torch.randint(0, clip_indices.numel(), (num_samples,), device=self.device)
            return clip_indices[sampled_local_ids]
        sampled_local_ids = torch.multinomial(group_weights / total, num_samples, replacement=True)
        return clip_indices[sampled_local_ids]

    def _sample_fixed_clip_group_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        if (
            self._fixed_clip_group_env_mask is None
            or self._fixed_clip_group_clip_mask is None
            or self._fixed_clip_complement_clip_mask is None
        ):
            raise RuntimeError("Fixed clip-group sampling requested before fixed clip-group masks were configured.")

        group_env_mask = self._fixed_clip_group_env_mask[env_ids]
        sampled_clip_ids = torch.empty((env_ids.numel(),), device=self.device, dtype=torch.long)
        if torch.any(group_env_mask):
            group_count = int(group_env_mask.sum().item())
            sampled_clip_ids[group_env_mask] = self._sample_clip_ids_from_mask(
                self._fixed_clip_group_clip_mask,
                group_count,
            )
        complement_env_mask = ~group_env_mask
        if torch.any(complement_env_mask):
            complement_count = int(complement_env_mask.sum().item())
            sampled_clip_ids[complement_env_mask] = self._sample_clip_ids_from_mask(
                self._fixed_clip_complement_clip_mask,
                complement_count,
            )
        return sampled_clip_ids

    def _update_clip_success_stats(self, env_ids: torch.Tensor) -> None:
        if not self.multi_clip or self.clip_weighting_strategy != "success_rate_adaptive":
            return
        if self._env.is_evaluating:
            return
        if self._clip_success_counts is None or self._clip_total_counts is None:
            return
        if env_ids.numel() == 0:
            return

        episode_lengths = self._env.episode_length_buf[env_ids]
        base_reset = self._base_reset_has_previous_action_mask(env_ids)
        completed = self._motion_completed_mask_for_env_ids(env_ids)
        # Direct MotionCommand.reset() during clip rollover keeps a positive
        # episode length; BaseTask resets have already zeroed it and are
        # identified through _pending_episode_lengths.
        clip_rollover = (episode_lengths > 0) & completed
        valid_mask = base_reset | clip_rollover
        if not torch.any(valid_mask):
            return

        valid_env_ids = env_ids[valid_mask]
        clip_ids = self.clip_ids[valid_env_ids]
        failed = self._adaptive_failure_mask_for_env_ids(env_ids)[valid_mask]
        successes = (completed[valid_mask] & ~failed).to(dtype=torch.float32)

        ones = torch.ones_like(successes)
        self._clip_total_counts.index_add_(0, clip_ids, ones)
        self._clip_success_counts.index_add_(0, clip_ids, successes)
        self._refresh_adaptive_clip_weights()

    def _refresh_adaptive_clip_weights(self) -> None:
        if self.clip_weighting_strategy != "success_rate_adaptive":
            return
        if self._clip_total_counts is None or self._clip_success_counts is None:
            return
        if self._base_clip_weights is None:
            return

        total = self._clip_total_counts
        success = self._clip_success_counts
        valid_mask = total > 0

        inv_success = torch.ones_like(total)
        if torch.any(valid_mask):
            success_rates = torch.zeros_like(total)
            success_rates[valid_mask] = success[valid_mask] / total[valid_mask]
            inv_success[valid_mask] = 1.0 / (success_rates[valid_mask] + 0.05)
            mean_inv = inv_success[valid_mask].mean()
            if mean_inv > 1e-6:
                inv_success = inv_success / mean_inv

        factors = torch.clamp(inv_success, self.min_weight_factor, self.max_weight_factor)
        weights = self._base_clip_weights * factors
        if weights.sum() > 1e-9:
            self._raw_clip_sampling_weights = weights / weights.sum()
        else:
            self._raw_clip_sampling_weights = self._base_clip_weights.clone()
        self._refresh_current_clip_sampling_weights()

    @staticmethod
    def _clamp01(value: float) -> float:
        if not np.isfinite(value):
            raise ValueError(f"Probability value must be finite, got {value!r}.")
        return float(max(0.0, min(1.0, value)))

    @staticmethod
    def _validate_reset_sampling_curriculum_config(motion_cfg: Any) -> None:
        """Fail setup when reset/T1 curriculum values would be ignored or sanitized."""

        def probability(name: str, value: Any, *, optional: bool = False) -> float | None:
            if value is None and optional:
                return None
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(f"{name} must be a real probability, got {value!r}.")
            parsed = float(value)
            if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1], got {value!r}.")
            return parsed

        def schedule(prefix: str) -> None:
            probability(prefix, getattr(motion_cfg, prefix))
            end_name = f"{prefix}_end"
            start_iter_name = f"{prefix}_start_iter"
            end_iter_name = f"{prefix}_end_iter"
            end_value = getattr(motion_cfg, end_name)
            start_iter = getattr(motion_cfg, start_iter_name)
            end_iter = getattr(motion_cfg, end_iter_name)
            provided = (end_value is not None, start_iter is not None, end_iter is not None)
            if any(provided) and not all(provided):
                raise ValueError(
                    f"{prefix} schedule must set {end_name}, {start_iter_name}, and {end_iter_name} together; "
                    f"got end={end_value!r}, start_iter={start_iter!r}, end_iter={end_iter!r}."
                )
            if not all(provided):
                return
            probability(end_name, end_value)
            for name, value in ((start_iter_name, start_iter), (end_iter_name, end_iter)):
                if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 0:
                    raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
            if int(end_iter) < int(start_iter):
                raise ValueError(
                    f"{end_iter_name} must be >= {start_iter_name}, got {end_iter} < {start_iter}."
                )

        schedule("start_at_timestep_zero_prob")
        schedule("freeze_at_timestep_zero_prob")

        enabled = getattr(motion_cfg, "uniform_t1_window_sampling_enabled")
        if type(enabled) is not bool:
            raise ValueError(
                "uniform_t1_window_sampling_enabled must be a boolean, "
                f"got {enabled!r}."
            )
        half_width = getattr(motion_cfg, "uniform_t1_window_half_width_steps")
        if (
            isinstance(half_width, bool)
            or not isinstance(half_width, numbers.Integral)
            or int(half_width) < 0
        ):
            raise ValueError(
                "uniform_t1_window_half_width_steps must be a non-negative integer, "
                f"got {half_width!r}."
            )
        density_boost = getattr(motion_cfg, "uniform_t1_window_density_boost")
        if isinstance(density_boost, bool) or not isinstance(density_boost, numbers.Real):
            raise ValueError(
                "uniform_t1_window_density_boost must be a real number, "
                f"got {density_boost!r}."
            )
        density_boost = float(density_boost)
        if not np.isfinite(density_boost) or density_boost < 1.0:
            raise ValueError(
                "uniform_t1_window_density_boost must be finite and >= 1, "
                f"got {density_boost!r}."
            )
        target = probability(
            "uniform_t1_window_target_sample_frac",
            getattr(motion_cfg, "uniform_t1_window_target_sample_frac"),
            optional=True,
        )
        if not enabled and (target is not None or density_boost != 1.0):
            raise ValueError(
                "uniform_t1_window target/density settings would be ignored because "
                "uniform_t1_window_sampling_enabled=False."
            )
        if enabled and target is not None:
            zero_start_probabilities = [float(motion_cfg.start_at_timestep_zero_prob)]
            if motion_cfg.start_at_timestep_zero_prob_end is not None:
                zero_start_probabilities.append(
                    float(motion_cfg.start_at_timestep_zero_prob_end)
                )
            minimum_nonzero_mass = 1.0 - max(zero_start_probabilities)
            if target > minimum_nonzero_mass + 1.0e-12:
                raise ValueError(
                    "uniform_t1_window_target_sample_frac cannot be realized by the configured "
                    "start-at-zero mixture: the T1 window contains only nonzero reset timesteps, "
                    f"target={target}, minimum_nonzero_reset_mass={minimum_nonzero_mass}. "
                    "Lower the target or the maximum start_at_timestep_zero_prob (including its "
                    "scheduled end value)."
                )

    def _iteration_curriculum_progress(self, start_iter: int | None, end_iter: int | None) -> float | None:
        if start_iter is None or end_iter is None or self._training_iteration is None:
            return None
        if self._training_iteration < start_iter:
            return 0.0
        if end_iter <= start_iter:
            return 1.0
        return min(max(float(self._training_iteration - start_iter) / float(end_iter - start_iter), 0.0), 1.0)

    def _scheduled_reset_prob(
        self,
        start_value: float,
        *,
        end_value: float | None,
        start_iter: int | None,
        end_iter: int | None,
    ) -> float:
        start_value = self._clamp01(float(start_value))
        if end_value is None or start_iter is None or end_iter is None:
            return start_value

        end_value = self._clamp01(float(end_value))
        if self._env.is_evaluating:
            return end_value

        alpha = self._iteration_curriculum_progress(start_iter, end_iter)
        if alpha is None:
            return start_value
        return self._clamp01(start_value + (end_value - start_value) * alpha)

    def _current_start_at_timestep_zero_prob(self) -> float:
        return self._scheduled_reset_prob(
            float(self.motion_cfg.start_at_timestep_zero_prob),
            end_value=self.motion_cfg.start_at_timestep_zero_prob_end,
            start_iter=self.motion_cfg.start_at_timestep_zero_prob_start_iter,
            end_iter=self.motion_cfg.start_at_timestep_zero_prob_end_iter,
        )

    def _uniform_t1_window_sampling_active(self) -> bool:
        return bool(self.motion_cfg.uniform_t1_window_sampling_enabled)

    def _uniform_t1_window_conditional_target_probability(self) -> float | None:
        """Convert an overall target mass into the nonzero branch target mass."""
        target_frac = self.motion_cfg.uniform_t1_window_target_sample_frac
        if target_frac is None:
            return None
        nonzero_prob = max(0.0, 1.0 - self._current_start_at_timestep_zero_prob())
        target_frac = self._clamp01(float(target_frac))
        if target_frac > nonzero_prob + 1.0e-8:
            raise RuntimeError(
                "uniform_t1_window_target_sample_frac exceeds the live nonzero reset mass: "
                f"target={target_frac}, nonzero_reset_mass={nonzero_prob}. Refusing to silently "
                "clip the requested scientific reset distribution."
            )
        if nonzero_prob <= 1.0e-12:
            return 0.0
        return self._clamp01(target_frac / nonzero_prob)

    def _uniform_t1_window_bounds(
        self,
        clip_ids: torch.Tensor,
        valid_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_samples = clip_ids.numel()
        device = self.device
        max_step = torch.clamp(valid_starts.to(device=device, dtype=torch.long) - 1, min=0)
        zeros = torch.zeros((num_samples,), device=device, dtype=torch.long)

        if (
            self._adaptive_sampling_contact_window_by_clip is None
            or self._adaptive_sampling_contact_window_valid_by_clip is None
            or num_samples == 0
        ):
            return torch.zeros((num_samples,), device=device, dtype=torch.bool), zeros, zeros, zeros, zeros, max_step

        clip_ids = clip_ids.to(device=device, dtype=torch.long)
        contact_valid = self._adaptive_sampling_contact_window_valid_by_clip[clip_ids]
        t1 = self._adaptive_sampling_contact_window_by_clip[clip_ids, 0]
        half_width = max(0, int(self.motion_cfg.uniform_t1_window_half_width_steps))

        lo = torch.clamp(t1 - half_width, min=1)
        hi = torch.minimum(t1 + half_width, max_step)
        window_valid = contact_valid & (max_step >= 1) & (hi >= lo)
        window_len = torch.where(window_valid, hi - lo + 1, zeros)
        total_nonzero_len = max_step
        outside_len = torch.clamp(total_nonzero_len - window_len, min=0)
        return window_valid, lo, hi, window_len, outside_len, total_nonzero_len

    def _uniform_t1_window_probability(self, window_len: torch.Tensor, outside_len: torch.Tensor) -> torch.Tensor:
        conditional_target = self._uniform_t1_window_conditional_target_probability()
        if conditional_target is not None:
            target_prob = torch.full_like(
                window_len,
                conditional_target,
                dtype=torch.float32,
            )
            target_prob = torch.where(window_len > 0, target_prob, torch.zeros_like(target_prob))
            window_covers_all_nonzero_steps = (window_len > 0) & (outside_len <= 0)
            return torch.where(window_covers_all_nonzero_steps, torch.ones_like(target_prob), target_prob)

        boost = max(1.0, float(self.motion_cfg.uniform_t1_window_density_boost))
        window_score = window_len.to(dtype=torch.float32) * boost
        outside_score = outside_len.to(dtype=torch.float32)
        denom = window_score + outside_score
        return torch.where(denom > 0.0, window_score / denom, torch.zeros_like(denom))

    def _sample_uniform_t1_window_time_steps(self, env_ids: torch.Tensor, valid_starts: torch.Tensor) -> torch.Tensor:
        num_samples = env_ids.numel()
        device = self.device
        zeros = torch.zeros((num_samples,), device=device, dtype=torch.long)
        if num_samples == 0:
            return zeros

        clip_ids = self.clip_ids[env_ids]
        window_valid, lo, hi, window_len, outside_len, total_nonzero_len = self._uniform_t1_window_bounds(
            clip_ids,
            valid_starts,
        )

        fallback_count = torch.clamp(total_nonzero_len, min=1)
        fallback_sample = (torch.rand(num_samples, device=device) * fallback_count.to(dtype=torch.float32)).long() + 1
        fallback_sample = torch.where(total_nonzero_len > 0, fallback_sample, zeros)

        p_window = self._uniform_t1_window_probability(window_len, outside_len)
        choose_window = window_valid & (torch.rand(num_samples, device=device) < p_window)

        window_count = torch.clamp(window_len, min=1)
        window_offset = (torch.rand(num_samples, device=device) * window_count.to(dtype=torch.float32)).long()
        window_sample = lo + window_offset

        outside_count = torch.clamp(outside_len, min=1)
        outside_offset = (torch.rand(num_samples, device=device) * outside_count.to(dtype=torch.float32)).long()
        before_len = torch.clamp(lo - 1, min=0)
        outside_sample = torch.where(
            outside_offset < before_len,
            outside_offset + 1,
            hi + 1 + (outside_offset - before_len),
        )
        outside_sample = torch.minimum(outside_sample, total_nonzero_len)

        weighted_sample = torch.where(choose_window, window_sample, outside_sample)
        return torch.where(window_valid, weighted_sample, fallback_sample)

    def _effective_adaptive_timestep_probabilities_for_clip(self, clip_idx: int) -> torch.Tensor:
        """Return the reset distribution after contact bias and zero-start mixing."""
        if self.adaptive_timesteps_sampler is None:
            raise RuntimeError("Adaptive timestep probabilities requested while the sampler is disabled.")
        valid_start_count = int(self.adaptive_timesteps_sampler.valid_start_counts[clip_idx].item())
        window: tuple[int, int] | None = None
        if self._uniform_t1_window_sampling_active():
            clip_ids = torch.tensor([clip_idx], device=self.device, dtype=torch.long)
            valid_starts = torch.tensor([valid_start_count], device=self.device, dtype=torch.long)
            window_valid, lo, hi, _, _, _ = self._uniform_t1_window_bounds(clip_ids, valid_starts)
            if bool(window_valid[0].item()):
                window = (int(lo[0].item()), int(hi[0].item()))

        probabilities = self.adaptive_timesteps_sampler.timestep_probabilities_for_clip(
            clip_idx,
            exclude_zero=True,
            window=window,
            window_density_boost=float(self.motion_cfg.uniform_t1_window_density_boost),
            window_target_probability=self._uniform_t1_window_conditional_target_probability(),
        )
        if self._env.is_evaluating:
            probabilities.zero_()
            probabilities[0] = 1.0
            return probabilities
        zero_prob = self._current_start_at_timestep_zero_prob()
        probabilities *= 1.0 - zero_prob
        probabilities[0] += zero_prob
        return probabilities

    def _effective_adaptive_probability_views(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Build timestep and bin probability views once for telemetry."""

        timestep_probabilities_by_clip: list[torch.Tensor] = []
        probabilities_by_clip: list[torch.Tensor] = []
        for clip_idx in range(self.motion.num_clips):
            timestep_probabilities = self._effective_adaptive_timestep_probabilities_for_clip(clip_idx)
            timestep_probabilities_by_clip.append(timestep_probabilities)
            bin_ids = self.adaptive_timesteps_sampler._bin_indices_for_clip(clip_idx)
            bin_probabilities = torch.zeros(
                int(self.adaptive_timesteps_sampler.num_bins_per_clip[clip_idx].item()),
                device=self.device,
                dtype=torch.float32,
            )
            bin_probabilities.index_add_(0, bin_ids, timestep_probabilities)
            probabilities_by_clip.append(bin_probabilities)
        return timestep_probabilities_by_clip, probabilities_by_clip

    def _effective_adaptive_bin_probabilities(self) -> list[torch.Tensor]:
        _, probabilities_by_clip = self._effective_adaptive_probability_views()
        return probabilities_by_clip

    def _record_uniform_t1_window_reset_metrics(
        self,
        env_ids: torch.Tensor,
        valid_starts: torch.Tensor,
        *,
        adaptive_reset_probabilities: torch.Tensor | None = None,
    ) -> None:
        if not bool(self.motion_cfg.uniform_t1_window_sampling_enabled) or env_ids.numel() == 0:
            return

        clip_ids = self.clip_ids[env_ids]
        window_valid, lo, hi, window_len, outside_len, _ = self._uniform_t1_window_bounds(clip_ids, valid_starts)
        sampled_steps = self.time_steps[env_ids]
        sampled_window = window_valid & (sampled_steps >= lo) & (sampled_steps <= hi)
        nonzero_prob = max(0.0, 1.0 - self._current_start_at_timestep_zero_prob())
        if self.use_adaptive_timesteps_sampler:
            if self._env.is_evaluating:
                expected_window_prob = torch.zeros_like(window_len, dtype=torch.float32)
            else:
                probabilities = adaptive_reset_probabilities
                if probabilities is None:
                    windows = torch.stack((lo, hi), dim=-1)
                    probabilities = self.adaptive_timesteps_sampler.timestep_probabilities_for_samples(
                        clip_ids,
                        exclude_zero=True,
                        windows=windows,
                        window_valid=window_valid,
                        window_density_boost=float(self.motion_cfg.uniform_t1_window_density_boost),
                        window_target_probability=self._uniform_t1_window_conditional_target_probability(),
                        _trusted_inputs=True,
                    )
                if probabilities.shape != (env_ids.numel(), self.adaptive_timesteps_sampler.max_valid_start_count):
                    raise RuntimeError(
                        "Adaptive reset telemetry received probability rows with incompatible geometry."
                    )
                step_axis = self.adaptive_timesteps_sampler._step_axis.unsqueeze(0)
                telemetry_window_mask = (
                    window_valid.unsqueeze(1)
                    & (step_axis >= lo.unsqueeze(1))
                    & (step_axis <= hi.unsqueeze(1))
                )
                expected_window_prob = torch.where(
                    telemetry_window_mask,
                    probabilities,
                    torch.zeros_like(probabilities),
                ).sum(dim=1) * nonzero_prob
        else:
            p_window = self._uniform_t1_window_probability(window_len, outside_len)
            expected_window_prob = p_window * nonzero_prob

        window_valid_float = window_valid.to(dtype=torch.float32)
        valid_count = window_valid_float.sum().clamp_min(1.0)
        telemetry_values = torch.stack(
            (
                window_valid_float.mean(),
                sampled_window.to(dtype=torch.float32).mean(),
                expected_window_prob.mean(),
                (sampled_window.to(dtype=torch.float32) * window_valid_float).sum() / valid_count,
                (expected_window_prob * window_valid_float).sum() / valid_count,
                (window_len.to(dtype=torch.float32) * window_valid_float).sum() / valid_count,
            )
        )
        (
            self._uniform_t1_window_last_reset_available_frac,
            self._uniform_t1_window_last_reset_sample_frac,
            self._uniform_t1_window_last_reset_expected_sample_frac,
            self._uniform_t1_window_last_reset_sample_frac_valid,
            self._uniform_t1_window_last_reset_expected_sample_frac_valid,
            self._uniform_t1_window_last_reset_mean_window_len,
        ) = telemetry_values

    def _current_freeze_at_timestep_zero_prob(self) -> float:
        return self._scheduled_reset_prob(
            float(self.motion_cfg.freeze_at_timestep_zero_prob),
            end_value=self.motion_cfg.freeze_at_timestep_zero_prob_end,
            start_iter=self.motion_cfg.freeze_at_timestep_zero_prob_start_iter,
            end_iter=self.motion_cfg.freeze_at_timestep_zero_prob_end_iter,
        )

    def _get_clip_pickup_stats_by_clip(self) -> tuple[torch.Tensor, torch.Tensor]:
        cache_name = (
            "_clip_pickup_stats_by_clip_"
            f"h{_RUNTIME_PICKUP_LIFT_HEIGHT_THRESHOLD:.4f}_"
            f"r{_CLIP_PICKUP_LIFT_RATIO_THRESHOLD:.4f}_"
            f"c{_RUNTIME_PICKUP_CONSECUTIVE_STEPS:d}"
        ).replace(".", "p")
        cached = getattr(self, cache_name, None)
        if cached is not None:
            return cached

        pickup_steps_by_clip = torch.zeros((self.motion.num_clips,), device=self.device, dtype=torch.long)
        pickup_thresholds_by_clip = torch.zeros((self.motion.num_clips,), device=self.device, dtype=torch.float32)
        if not self.motion.has_object:
            cached = (pickup_steps_by_clip, pickup_thresholds_by_clip)
            setattr(self, cache_name, cached)
            return cached

        clip_offsets = self.motion.clip_offsets
        clip_lengths = self.motion.clip_lengths
        root_pos_w = self.motion.body_pos_w[:, 0]
        object_pos_w = self.motion.object_pos_w

        for clip_idx in range(self.motion.num_clips):
            clip_start = int(clip_offsets[clip_idx].item())
            clip_length = int(clip_lengths[clip_idx].item())
            if clip_length <= 0:
                continue

            clip_end = clip_start + clip_length
            clip_rel_z = object_pos_w[clip_start:clip_end, 2] - root_pos_w[clip_start:clip_end, 2]
            pickup_step, pickup_threshold = _pickup_step_and_threshold_from_rel_z(
                clip_rel_z,
                lift_height_threshold=_RUNTIME_PICKUP_LIFT_HEIGHT_THRESHOLD,
                lift_ratio_threshold=_CLIP_PICKUP_LIFT_RATIO_THRESHOLD,
                consecutive_steps=_RUNTIME_PICKUP_CONSECUTIVE_STEPS,
            )
            pickup_steps_by_clip[clip_idx] = pickup_step
            pickup_thresholds_by_clip[clip_idx] = pickup_threshold

        cached = (pickup_steps_by_clip, pickup_thresholds_by_clip)
        setattr(self, cache_name, cached)
        return cached

    def _get_clip_pickup_steps_by_clip(self) -> torch.Tensor:
        return self._get_clip_pickup_stats_by_clip()[0]

    def _get_clip_pickup_thresholds_by_clip(self) -> torch.Tensor:
        return self._get_clip_pickup_stats_by_clip()[1]

    def _get_contact_aware_carry_window_by_clip(self) -> torch.Tensor:
        carry_window_mode = (
            str(getattr(self.motion_cfg, "contact_aware_carry_window_mode", "rel_z")).strip().lower().replace("-", "_")
        )
        peak_height_alpha = float(getattr(self.motion_cfg, "contact_aware_peak_height_alpha", 0.91))
        peak_height_smoothing_steps = int(getattr(self.motion_cfg, "contact_aware_peak_height_smoothing_steps", 5))
        cache_name = (
            "_contact_aware_carry_window_by_clip_"
            f"{carry_window_mode}_"
            f"h{_RUNTIME_PICKUP_LIFT_HEIGHT_THRESHOLD:.4f}_"
            f"r{_CLIP_PICKUP_LIFT_RATIO_THRESHOLD:.4f}_"
            f"peak{peak_height_alpha:.4f}_"
            f"smooth{peak_height_smoothing_steps:d}_"
            f"c{_RUNTIME_PICKUP_CONSECUTIVE_STEPS:d}_"
            f"release{_ADAPTIVE_SAMPLING_CONTACT_STAGE_RELEASE_LEAD_STEPS:d}"
        ).replace(".", "p")
        cached = getattr(self, cache_name, None)
        if cached is not None:
            return cached

        if carry_window_mode not in {"rel_z", "peak_height"}:
            raise ValueError(
                "Unsupported contact_aware_carry_window_mode="
                f"'{getattr(self.motion_cfg, 'contact_aware_carry_window_mode', None)}'. "
                "Expected 'rel_z' or 'peak_height'."
            )

        carry_window_by_clip = torch.zeros((self.motion.num_clips, 2), device=self.device, dtype=torch.long)
        carry_window_by_clip[:, 1] = torch.clamp(self.motion.clip_lengths, min=0)
        if not self.motion.has_object:
            setattr(self, cache_name, carry_window_by_clip)
            return carry_window_by_clip

        clip_offsets = self.motion.clip_offsets
        clip_lengths = self.motion.clip_lengths
        root_pos_w = self.motion.body_pos_w[:, 0]
        object_pos_w = self.motion.object_pos_w

        for clip_idx in range(self.motion.num_clips):
            clip_start = int(clip_offsets[clip_idx].item())
            clip_length = int(clip_lengths[clip_idx].item())
            if clip_length <= 0:
                continue

            clip_end = clip_start + clip_length
            clip_rel_z = object_pos_w[clip_start:clip_end, 2] - root_pos_w[clip_start:clip_end, 2]

            contact_interval = None
            if (
                self._adaptive_sampling_contact_window_by_clip is not None
                and self._adaptive_sampling_contact_window_valid_by_clip is not None
                and bool(self._adaptive_sampling_contact_window_valid_by_clip[clip_idx].item())
            ):
                contact_interval = (
                    int(self._adaptive_sampling_contact_window_by_clip[clip_idx, 0].item()),
                    int(self._adaptive_sampling_contact_window_by_clip[clip_idx, 1].item()),
                )

            if carry_window_mode == "peak_height":
                carry_start, carry_end = _contact_aware_carry_window_from_peak_height(
                    object_pos_w[clip_start:clip_end, 2],
                    peak_height_alpha=peak_height_alpha,
                    smoothing_steps=peak_height_smoothing_steps,
                    consecutive_steps=_RUNTIME_PICKUP_CONSECUTIVE_STEPS,
                )
            else:
                carry_start, carry_end = _contact_aware_carry_window_from_rel_z(
                    clip_rel_z,
                    contact_interval=contact_interval,
                    lift_height_threshold=_RUNTIME_PICKUP_LIFT_HEIGHT_THRESHOLD,
                    lift_ratio_threshold=_CLIP_PICKUP_LIFT_RATIO_THRESHOLD,
                    consecutive_steps=_RUNTIME_PICKUP_CONSECUTIVE_STEPS,
                    release_lead_steps=_ADAPTIVE_SAMPLING_CONTACT_STAGE_RELEASE_LEAD_STEPS,
                )
            carry_window_by_clip[clip_idx, 0] = carry_start
            carry_window_by_clip[clip_idx, 1] = carry_end

        setattr(self, cache_name, carry_window_by_clip)
        return carry_window_by_clip

    def get_contact_aware_root_command_active_mask(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        env_ids_t = self._ensure_index_tensor(env_ids)
        if not self.motion.has_object:
            return torch.ones((env_ids_t.numel(),), device=self.device, dtype=torch.bool)

        clip_ids = self.clip_ids[env_ids_t]
        time_steps = self.time_steps[env_ids_t]
        carry_window_by_clip = self._get_contact_aware_carry_window_by_clip()
        carry_start = carry_window_by_clip[clip_ids, 0]
        carry_end = carry_window_by_clip[clip_ids, 1]
        return (time_steps >= carry_start) & (time_steps < carry_end)

    def _get_contact_aware_button_window_by_clip(self) -> torch.Tensor:
        """Return the configured source-clock pickup/drop transition window."""
        self._validate_kinematic_button_motion_object()
        button_window_mode = getattr(
            getattr(self, "motion_cfg", None),
            "contact_aware_button_window_mode",
            "contact_interval",
        )
        if button_window_mode not in {"contact_interval", "kinematic_lift"}:
            raise ValueError(
                "Unsupported contact_aware_button_window_mode="
                f"{button_window_mode!r}. Expected 'contact_interval' or "
                "'kinematic_lift'."
            )

        if button_window_mode == "kinematic_lift":
            cache_name = "_contact_aware_button_window_by_clip_kinematic_lift_v1"
            cached = getattr(self, cache_name, None)
            if cached is not None:
                return cached

            result = torch.zeros(
                (self.motion.num_clips, 2),
                device=self.device,
                dtype=torch.long,
            )
            result[:, 1] = torch.clamp(self.motion.clip_lengths, min=0)
            if self.motion.has_object:
                clip_offsets = self.motion.clip_offsets
                clip_lengths = self.motion.clip_lengths
                root_pos_w = self.motion.body_pos_w[:, 0]
                object_pos_w = self.motion.object_pos_w
                for clip_idx in range(self.motion.num_clips):
                    clip_start = int(clip_offsets[clip_idx].item())
                    clip_length = int(clip_lengths[clip_idx].item())
                    if clip_length <= 0:
                        continue
                    clip_end = clip_start + clip_length
                    rel_z = (
                        object_pos_w[clip_start:clip_end, 2]
                        - root_pos_w[clip_start:clip_end, 2]
                    )
                    lift_start, lift_end = _kinematic_lift_window_from_rel_z(
                        rel_z,
                        lift_height_threshold=_RUNTIME_PICKUP_LIFT_HEIGHT_THRESHOLD,
                        lift_ratio_threshold=_CLIP_PICKUP_LIFT_RATIO_THRESHOLD,
                        consecutive_steps=_RUNTIME_PICKUP_CONSECUTIVE_STEPS,
                        require_sustained_lift=True,
                    )
                    result[clip_idx, 0] = lift_start
                    result[clip_idx, 1] = lift_end
            setattr(self, cache_name, result)
            return result

        # Legacy mode: exported contact t1/t2 override the configured root
        # carry window per valid clip.  This remains the default so old
        # checkpoints keep their serialized behavior.
        fallback = self._get_contact_aware_carry_window_by_clip()
        contact_windows = getattr(self, "_adaptive_sampling_contact_window_by_clip", None)
        contact_valid = getattr(self, "_adaptive_sampling_contact_window_valid_by_clip", None)
        if contact_windows is None or contact_valid is None or not torch.any(contact_valid):
            return fallback
        result = fallback.clone()
        result[contact_valid] = contact_windows[contact_valid]
        return result

    def _validate_kinematic_button_motion_object(self) -> None:
        """Fail before kinematic button training can degrade to constant zeros."""

        button_window_mode = getattr(
            getattr(self, "motion_cfg", None),
            "contact_aware_button_window_mode",
            "contact_interval",
        )
        if button_window_mode == "kinematic_lift" and not bool(
            getattr(getattr(self, "motion", None), "has_object", False)
        ):
            raise ValueError(
                "contact_aware_button_window_mode='kinematic_lift' requires a motion "
                "with an object trajectory; pickup/drop labels cannot be constant-zero fallbacks."
            )

    def get_contact_aware_drop_button(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        self._validate_kinematic_button_motion_object()
        env_ids_t = self._ensure_index_tensor(env_ids)
        if not self.motion.has_object:
            return torch.zeros((env_ids_t.numel(),), device=self.device, dtype=torch.bool)

        clip_ids = self.clip_ids[env_ids_t]
        time_steps = self.time_steps[env_ids_t]
        carry_window_by_clip = self._get_contact_aware_button_window_by_clip()
        carry_end = carry_window_by_clip[clip_ids, 1]
        return time_steps >= carry_end

    def get_contact_aware_pickup_button(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        self._validate_kinematic_button_motion_object()
        env_ids_t = self._ensure_index_tensor(env_ids)
        if not self.motion.has_object:
            return torch.zeros((env_ids_t.numel(),), device=self.device, dtype=torch.bool)

        clip_ids = self.clip_ids[env_ids_t]
        time_steps = self.time_steps[env_ids_t]
        carry_window_by_clip = self._get_contact_aware_button_window_by_clip()
        carry_start = carry_window_by_clip[clip_ids, 0]
        return time_steps < carry_start

    def _reset_pickup_anchor_state(
        self,
        env_ids: torch.Tensor,
        *,
        root_pos_w: torch.Tensor | None = None,
        root_quat_w: torch.Tensor | None = None,
        object_pos_w: torch.Tensor | None = None,
        object_quat_w: torch.Tensor | None = None,
    ) -> None:
        if (
            self.pickup_anchor_set is None
            or self.pickup_anchor_root_pos_w is None
            or self.pickup_anchor_root_quat_w is None
            or self.pickup_object_rel_z_baseline is None
            or self.pickup_consecutive_counter is None
        ):
            return

        self.pickup_anchor_set[env_ids] = False
        self.pickup_consecutive_counter[env_ids] = 0
        self.pickup_anchor_root_pos_w[env_ids] = 0.0
        self.pickup_anchor_root_quat_w[env_ids] = 0.0
        self.pickup_anchor_root_quat_w[env_ids, 3] = 1.0
        self.pickup_object_rel_z_baseline[env_ids] = 0.0
        hybrid_object_z_baseline = getattr(self, "hybrid_velocity_object_z_baseline", None)
        if hybrid_object_z_baseline is not None:
            hybrid_object_z_baseline[env_ids] = 0.0
        anchor_object_pos_b = getattr(self, "pickup_anchor_object_pos_b", None)
        anchor_object_quat_b = getattr(self, "pickup_anchor_object_quat_b", None)
        if anchor_object_pos_b is not None:
            anchor_object_pos_b[env_ids] = 0.0
        if anchor_object_quat_b is not None:
            anchor_object_quat_b[env_ids] = 0.0
            anchor_object_quat_b[env_ids, 3] = 1.0

        if root_pos_w is None or root_quat_w is None or object_pos_w is None:
            return
        self.pickup_anchor_root_pos_w[env_ids] = root_pos_w
        self.pickup_anchor_root_quat_w[env_ids] = root_quat_w
        self.pickup_object_rel_z_baseline[env_ids] = object_pos_w[:, 2] - root_pos_w[:, 2]
        if hybrid_object_z_baseline is not None:
            hybrid_object_z_baseline[env_ids] = object_pos_w[:, 2]

        # If reset starts after the clip's pickup phase, treat the object as already
        # picked at reset time.
        clip_pickup_steps = self._get_clip_pickup_steps_by_clip()[self.clip_ids[env_ids]]
        already_picked_mask = self.time_steps[env_ids] >= clip_pickup_steps
        # Keep the reset path device-only too.  The anchor poses above already
        # contain the supplied reset root for every selected row, so only the
        # latch and counter differ for clips that start after pickup.
        self.pickup_anchor_set[env_ids] = already_picked_mask
        self.pickup_consecutive_counter[env_ids] = already_picked_mask.to(
            dtype=self.pickup_consecutive_counter.dtype
        ) * _RUNTIME_PICKUP_CONSECUTIVE_STEPS
        if hybrid_object_z_baseline is not None and self.hybrid_velocity_enabled():
            lift_height = float(self.motion_cfg.hybrid_velocity_lift_height_m)
            hybrid_object_z_baseline[env_ids] = torch.where(
                already_picked_mask,
                object_pos_w[:, 2] - lift_height,
                hybrid_object_z_baseline[env_ids],
            )
        if (
            anchor_object_pos_b is not None
            and anchor_object_quat_b is not None
            and object_quat_w is not None
        ):
            root_quat_inv = quat_inverse(root_quat_w, w_last=True)
            candidate_object_pos_b = quat_apply(
                root_quat_inv,
                object_pos_w - root_pos_w,
                w_last=True,
            )
            candidate_object_quat_b = quat_mul(
                root_quat_inv,
                object_quat_w,
                w_last=True,
            )
            anchor_object_pos_b[env_ids] = torch.where(
                already_picked_mask.unsqueeze(-1),
                candidate_object_pos_b,
                anchor_object_pos_b[env_ids],
            )
            anchor_object_quat_b[env_ids] = torch.where(
                already_picked_mask.unsqueeze(-1),
                candidate_object_quat_b,
                anchor_object_quat_b[env_ids],
            )

    def _update_pickup_anchor_state(self) -> None:
        if (
            not self.motion.has_object
            or self.pickup_anchor_set is None
            or self.pickup_anchor_root_pos_w is None
            or self.pickup_anchor_root_quat_w is None
            or self.pickup_object_rel_z_baseline is None
            or self.pickup_consecutive_counter is None
        ):
            return

        current_rel_z = self.simulator_object_pos_w[:, 2] - self.robot_root_pos_w[:, 2]
        # Reuse the same clip-derived pickup threshold that defined the fixed pickup-frame
        # command so runtime anchor latching stays in the same coordinate frame.
        clip_pickup_thresholds = self._get_clip_pickup_thresholds_by_clip()[self.clip_ids]
        lifted = current_rel_z >= clip_pickup_thresholds
        self.pickup_consecutive_counter.copy_(
            torch.where(
                lifted,
                self.pickup_consecutive_counter + 1,
                torch.zeros_like(self.pickup_consecutive_counter),
            )
        )
        newly_picked = (~self.pickup_anchor_set) & (
            self.pickup_consecutive_counter >= _RUNTIME_PICKUP_CONSECUTIVE_STEPS
        )
        update_pos_mask = newly_picked.unsqueeze(-1)
        self.pickup_anchor_root_pos_w.copy_(
            torch.where(update_pos_mask, self.robot_root_pos_w, self.pickup_anchor_root_pos_w)
        )
        self.pickup_anchor_root_quat_w.copy_(
            torch.where(update_pos_mask, self.robot_root_quat_w, self.pickup_anchor_root_quat_w)
        )
        anchor_object_pos_b = getattr(self, "pickup_anchor_object_pos_b", None)
        anchor_object_quat_b = getattr(self, "pickup_anchor_object_quat_b", None)
        if anchor_object_pos_b is not None and anchor_object_quat_b is not None:
            root_quat_inv = quat_inverse(self.robot_root_quat_w, w_last=True)
            candidate_object_pos_b = quat_apply(
                root_quat_inv,
                self.simulator_object_pos_w - self.robot_root_pos_w,
                w_last=True,
            )
            candidate_object_quat_b = quat_mul(
                root_quat_inv,
                self.simulator_object_quat_w,
                w_last=True,
            )
            update_object_mask = newly_picked.unsqueeze(-1)
            anchor_object_pos_b.copy_(
                torch.where(update_object_mask, candidate_object_pos_b, anchor_object_pos_b)
            )
            anchor_object_quat_b.copy_(
                torch.where(update_object_mask, candidate_object_quat_b, anchor_object_quat_b)
            )
        self.pickup_anchor_set.logical_or_(newly_picked)

    def _update_contact_prior_state(self) -> None:
        if (
            not self.motion.has_object
            or not self._contact_prior_available
            or self._contact_prior_total_count is None
            or self._contact_prior_contact_sum is None
            or self._contact_prior_force_mean is None
            or self._contact_prior_force_count is None
            or self._contact_prior_position_mean is None
            or self._contact_prior_position_count is None
        ):
            return

        source_mask = self._env.episode_length_buf > 1
        if not torch.any(source_mask):
            return

        body_pos_error = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(dim=-1)
        object_pos_error = torch.norm(self.object_pos_w - self.simulator_object_pos_w, dim=-1)
        object_rot_error = quat_error_magnitude(self.object_quat_w, self.simulator_object_quat_w)
        stable_mask = (
            source_mask
            & (body_pos_error <= _CONTACT_PRIOR_BODY_POS_ERROR_THRESHOLD)
            & (object_pos_error <= _CONTACT_PRIOR_OBJECT_POS_ERROR_THRESHOLD)
            & (object_rot_error <= _CONTACT_PRIOR_OBJECT_ROT_ERROR_THRESHOLD)
        )
        if not torch.any(stable_mask):
            return

        current_force, current_contact, current_position = self.get_current_contact_prior_region_measurements()
        stable_clip_ids = self.clip_ids[stable_mask]
        stable_phase_ids = self._current_contact_prior_phase_ids()[stable_mask]
        stable_contact = current_contact[stable_mask]
        stable_force = current_force[stable_mask]
        stable_position = current_position[stable_mask]
        clip_phase_pairs = torch.unique(torch.stack((stable_clip_ids, stable_phase_ids), dim=1), dim=0)

        for clip_id, phase_id in clip_phase_pairs.tolist():
            pair_mask = (stable_clip_ids == clip_id) & (stable_phase_ids == phase_id)
            if not torch.any(pair_mask):
                continue

            pair_contact = stable_contact[pair_mask]
            pair_force = stable_force[pair_mask]
            pair_position = stable_position[pair_mask]
            pair_count = float(pair_mask.sum().item())
            self._contact_prior_total_count[clip_id, phase_id] += pair_count
            self._contact_prior_contact_sum[clip_id, phase_id] += pair_contact.to(dtype=torch.float32).sum(dim=0)

            for region_idx in range(len(_CONTACT_PRIOR_REGION_NAMES)):
                region_contact_mask = pair_contact[:, region_idx]
                region_contact_count = float(region_contact_mask.to(dtype=torch.float32).sum().item())
                if region_contact_count <= 0.0:
                    continue

                batch_force_mean = pair_force[region_contact_mask, region_idx].mean()
                prev_force_count = self._contact_prior_force_count[clip_id, phase_id, region_idx]
                new_force_count = prev_force_count + region_contact_count
                self._contact_prior_force_mean[clip_id, phase_id, region_idx] = (
                    self._contact_prior_force_mean[clip_id, phase_id, region_idx] * prev_force_count
                    + batch_force_mean * region_contact_count
                ) / new_force_count.clamp_min(1.0)
                self._contact_prior_force_count[clip_id, phase_id, region_idx] = new_force_count

                batch_position_mean = pair_position[region_contact_mask, region_idx].mean(dim=0)
                prev_position_count = self._contact_prior_position_count[clip_id, phase_id, region_idx]
                new_position_count = prev_position_count + region_contact_count
                self._contact_prior_position_mean[clip_id, phase_id, region_idx] = (
                    self._contact_prior_position_mean[clip_id, phase_id, region_idx] * prev_position_count
                    + batch_position_mean * region_contact_count
                ) / new_position_count.clamp_min(1.0)
                self._contact_prior_position_count[clip_id, phase_id, region_idx] = new_position_count

    def get_contact_prior_targets(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_regions = len(_CONTACT_PRIOR_REGION_NAMES)
        occupancy = torch.zeros((self.num_envs, num_regions), device=self.device, dtype=torch.float32)
        force = torch.zeros((self.num_envs, num_regions), device=self.device, dtype=torch.float32)
        position = torch.zeros((self.num_envs, num_regions, 3), device=self.device, dtype=torch.float32)
        confidence = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        valid_mask = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        if (
            self._contact_prior_total_count is None
            or self._contact_prior_contact_sum is None
            or self._contact_prior_force_mean is None
            or self._contact_prior_position_mean is None
        ):
            return occupancy, force, position, confidence, valid_mask

        phase_ids = self._current_contact_prior_phase_ids()
        total_count = self._contact_prior_total_count[self.clip_ids, phase_ids]
        observed_contact_count = self._contact_prior_contact_sum[self.clip_ids, phase_ids].sum(dim=-1)
        # A prior should only be considered valid after we have actually observed at least one
        # supported body-object contact for this clip/phase. Otherwise confidence can rise from
        # stable co-tracking samples while all contact targets remain identically zero.
        valid_mask = observed_contact_count > 0.0
        if torch.any(valid_mask):
            occupancy[valid_mask] = self._contact_prior_contact_sum[self.clip_ids[valid_mask], phase_ids[valid_mask]] / (
                total_count[valid_mask].unsqueeze(-1).clamp_min(1.0)
            )
            force[valid_mask] = self._contact_prior_force_mean[self.clip_ids[valid_mask], phase_ids[valid_mask]]
            position[valid_mask] = self._contact_prior_position_mean[self.clip_ids[valid_mask], phase_ids[valid_mask]]
            confidence[valid_mask] = torch.clamp(
                total_count[valid_mask] / float(_CONTACT_PRIOR_CONFIDENCE_WARMUP_SAMPLES),
                min=0.0,
                max=1.0,
            )
        return occupancy, force, position, confidence, valid_mask

    def get_contact_prior_region_targets(
        self,
        region_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        region_name = _normalize_contact_prior_region_name(region_name)
        if region_name not in _CONTACT_PRIOR_REGION_NAMES:
            raise ValueError(f"Unknown contact prior region '{region_name}'.")
        region_idx = _CONTACT_PRIOR_REGION_NAMES.index(region_name)
        occupancy, force, position, confidence, valid_mask = self.get_contact_prior_targets()
        return (
            occupancy[:, region_idx].unsqueeze(-1),
            force[:, region_idx].unsqueeze(-1),
            position[:, region_idx, :],
            confidence.unsqueeze(-1),
            valid_mask.unsqueeze(-1).to(dtype=torch.float32),
        )

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    def _get_env_offsets(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        terrain_state = None
        terrain_manager = getattr(self._env, "terrain_manager", None)
        if terrain_manager is not None:
            try:
                terrain_state = terrain_manager.get_state("locomotion_terrain")
            except Exception:
                terrain_state = None
        base = getattr(terrain_state, "env_origins", None)
        if base is None:
            base = self._env.simulator.scene.env_origins
        if self._clip_terrain_offsets is None or not hasattr(self, "clip_ids"):
            return base if env_ids is None else base[env_ids]

        clip_ids = self.clip_ids if env_ids is None else self.clip_ids[env_ids]
        clip_offsets = self._clip_terrain_offsets[clip_ids]
        if self._terrain_row_ids is not None:
            row_ids = self._terrain_row_ids if env_ids is None else self._terrain_row_ids[env_ids]
            if self._clip_terrain_offsets_by_row is not None:
                clip_offsets = self._clip_terrain_offsets_by_row[row_ids, clip_ids]
            elif self._terrain_row_stride > 0.0:
                row_offsets = torch.zeros_like(clip_offsets)
                row_offsets[:, 1] = row_ids.to(row_offsets.dtype) * self._terrain_row_stride
                clip_offsets = clip_offsets + row_offsets

        if self.motion_cfg.pair_terrain_with_motion:
            return clip_offsets

        if base.device != clip_offsets.device:
            base = base.to(clip_offsets.device)
        return base + clip_offsets

    #########################################################################################
    ## Robot from motion data
    #########################################################################################
    @property
    def joint_pos(self) -> torch.Tensor:
        return self._motion_joint_pos()

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._motion_joint_vel()

    def _motion_joint_pos(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self._raw_motion_joint_pos(env_ids)

    def _motion_joint_vel(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self._raw_motion_joint_vel(env_ids)

    def _motion_body_pos_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        pos = self._raw_motion_body_pos_w(env_ids)[:, self.tracked_body_indexes]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_pos(pos, env_ids)
        return pos + self._get_env_offsets(env_ids)[:, None, :]

    def _motion_body_quat_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        quat = self._raw_motion_body_quat_w(env_ids)[:, self.tracked_body_indexes]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_quat(quat, env_ids)
        return quat

    def _motion_body_lin_vel_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        vel = self._raw_motion_body_lin_vel_w(env_ids)[:, self.tracked_body_indexes]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_vec(vel, env_ids)
        return vel

    def _motion_body_ang_vel_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        vel = self._raw_motion_body_ang_vel_w(env_ids)[:, self.tracked_body_indexes]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_vec(vel, env_ids)
        return vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._motion_body_pos_w()

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._motion_body_quat_w()

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._motion_body_lin_vel_w()

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._motion_body_ang_vel_w()

    @property
    def ref_pos_w(self) -> torch.Tensor:
        pos = self._raw_motion_body_pos_w()[:, self.ref_body_index]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_pos(pos)
        return pos + self._get_env_offsets()

    @property
    def ref_quat_w(self) -> torch.Tensor:
        quat = self._raw_motion_body_quat_w()[:, self.ref_body_index]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_quat(quat)
        return quat

    @property
    def root_pos_w(self) -> torch.Tensor:
        pos = self._raw_motion_body_pos_w()[:, 0]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_pos(pos)
        return pos + self._get_env_offsets()

    @property
    def root_quat_w(self) -> torch.Tensor:
        quat = self._raw_motion_body_quat_w()[:, 0]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_quat(quat)
        return quat

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        vel = self._raw_motion_body_lin_vel_w()[:, 0]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_vec(vel)
        return vel

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        vel = self._raw_motion_body_ang_vel_w()[:, 0]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_vec(vel)
        return vel

    @property
    def ref_lin_vel_w(self) -> torch.Tensor:
        vel = self._raw_motion_body_lin_vel_w()[:, self.ref_body_index]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_vec(vel)
        return vel

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        vel = self._raw_motion_body_ang_vel_w()[:, self.ref_body_index]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_vec(vel)
        return vel

    #########################################################################################
    ## Robot from simulator
    #########################################################################################
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self._env.simulator.dof_pos  # (num_envs, num_dofs)

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self._env.simulator.dof_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self.tracked_body_indexes, :]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self.tracked_body_indexes, :]  # xyzw

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self.tracked_body_indexes, :]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self.tracked_body_indexes, :]

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, :3]  # type: ignore[attr-defined]

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 3:7]  # type: ignore[attr-defined]

    @property
    def robot_root_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 7:10]  # type: ignore[attr-defined]

    @property
    def robot_root_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 10:13]  # type: ignore[attr-defined]

    @property
    def robot_ref_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self.ref_body_index, :]

    @property
    def robot_ref_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self.ref_body_index, :]  # xyzw

    @property
    def robot_ref_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self.ref_body_index, :]

    @property
    def robot_ref_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self.ref_body_index, :]

    #########################################################################################
    ## Object from motion data
    #########################################################################################
    def _motion_object_pos_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        count = self.num_envs if env_ids is None else env_ids.numel()
        if not self.motion.has_object:
            return torch.zeros(count, 3, device=self.device, dtype=torch.float32)
        pos = self._raw_motion_object_pos_w(env_ids)
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_pos(pos, env_ids)
        return pos + self._get_env_offsets(env_ids)

    def _motion_object_quat_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        count = self.num_envs if env_ids is None else env_ids.numel()
        if not self.motion.has_object:
            quat = torch.zeros(count, 4, device=self.device, dtype=torch.float32)
            quat[:, 3] = 1.0
            return quat
        quat = self._raw_motion_object_quat_w(env_ids)
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_quat(quat, env_ids)
        return quat

    def _motion_object_lin_vel_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        count = self.num_envs if env_ids is None else env_ids.numel()
        if not self.motion.has_object:
            return torch.zeros(count, 3, device=self.device, dtype=torch.float32)
        vel = self._raw_motion_object_lin_vel_w(env_ids)
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_vec(vel, env_ids)
        return vel

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._motion_object_pos_w()

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._motion_object_quat_w()

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._motion_object_lin_vel_w()

    @property
    def object_size(self) -> torch.Tensor:
        if not self.motion.has_object:
            return torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        motion_idx = self._get_motion_indices(self.time_steps)
        return self.motion.object_size[motion_idx]

    #########################################################################################
    ## Object from simulator
    #########################################################################################
    @property
    def simulator_object_pos_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            return torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        return self.simulator_object_state_snapshot[:, :3]

    @property
    def simulator_object_quat_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            quat = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
            quat[:, 3] = 1.0
            return quat
        return self.simulator_object_state_snapshot[:, 3:7]

    @property
    def simulator_object_lin_vel_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            return torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        return self.simulator_object_state_snapshot[:, 7:10]

    @property
    def simulator_object_ang_vel_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            return torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        return self.simulator_object_state_snapshot[:, 10:13]

    #########################################################################################
    ## Methods that does not fit into setup/step/reset pattern
    #########################################################################################

    def init_buffers(self):
        if self.motion.has_object:
            self._simulator_object_state_snapshot = torch.empty(
                (self.num_envs, 13),
                device=self.device,
                dtype=torch.float32,
            )
            self._simulator_object_state_snapshot_ready = False
        else:
            self._simulator_object_state_snapshot = None
            self._simulator_object_state_snapshot_ready = True
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if self._fixed_clip_ids is not None and int(self._fixed_clip_ids.numel()) == int(self.num_envs):
            self.clip_ids[:] = self._fixed_clip_ids
        if self._terrain_row_ids is not None:
            self._terrain_row_ids.zero_()
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 3, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 4, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w[:, :, 0] = 1.0
        self._align_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._align_quat[:, 3] = 1.0
        self._align_pos = torch.zeros(self.num_envs, 3, device=self.device)
        if self.hmi_enabled():
            self.hmi_exact_goal_object_pos_w = torch.zeros(
                (self.num_envs, 3), device=self.device, dtype=torch.float32
            )
            self.hmi_exact_goal_object_quat_w = torch.zeros(
                (self.num_envs, 4), device=self.device, dtype=torch.float32
            )
            self.hmi_exact_goal_object_quat_w[:, 3] = 1.0
            self.hmi_goal_object_pos_w = torch.zeros_like(
                self.hmi_exact_goal_object_pos_w
            )
            self.hmi_goal_object_quat_w = self.hmi_exact_goal_object_quat_w.clone()
            self.hmi_goal_version = torch.zeros(
                (self.num_envs,), device=self.device, dtype=torch.long
            )
            self.hmi_goal_reached = torch.zeros(
                (self.num_envs,), device=self.device, dtype=torch.bool
            )
            assert self.hmi_cfg is not None
            self.hmi_goal_noise_scale = torch.tensor(
                float(self.hmi_cfg.goal_noise_initial_scale),
                device=self.device,
                dtype=torch.float32,
            )
            self.hmi_goal_success_ema = torch.zeros(
                (), device=self.device, dtype=torch.float32
            )
            self.hmi_goal_success_ema_initialized = False
            self.hmi_goal_success_sum = torch.zeros(
                (), device=self.device, dtype=torch.float32
            )
            self.hmi_goal_success_count = torch.zeros(
                (), device=self.device, dtype=torch.long
            )
            self.hmi_last_curriculum_update_iteration = 0
        else:
            self.hmi_exact_goal_object_pos_w = None
            self.hmi_exact_goal_object_quat_w = None
            self.hmi_goal_object_pos_w = None
            self.hmi_goal_object_quat_w = None
            self.hmi_goal_version = None
            self.hmi_goal_reached = None
            self.hmi_goal_noise_scale = None
            self.hmi_goal_success_ema = None
            self.hmi_goal_success_ema_initialized = False
            self.hmi_goal_success_sum = None
            self.hmi_goal_success_count = None
            self.hmi_last_curriculum_update_iteration = 0
        num_regions = len(_CONTACT_PRIOR_REGION_NAMES)
        self._contact_prior_total_count = torch.zeros(
            (self.motion.num_clips, _CONTACT_PRIOR_PHASE_COUNT),
            device=self.device,
            dtype=torch.float32,
        )
        self._contact_prior_contact_sum = torch.zeros(
            (self.motion.num_clips, _CONTACT_PRIOR_PHASE_COUNT, num_regions),
            device=self.device,
            dtype=torch.float32,
        )
        self._contact_prior_force_mean = torch.zeros(
            (self.motion.num_clips, _CONTACT_PRIOR_PHASE_COUNT, num_regions),
            device=self.device,
            dtype=torch.float32,
        )
        self._contact_prior_force_count = torch.zeros(
            (self.motion.num_clips, _CONTACT_PRIOR_PHASE_COUNT, num_regions),
            device=self.device,
            dtype=torch.float32,
        )
        self._contact_prior_position_mean = torch.zeros(
            (self.motion.num_clips, _CONTACT_PRIOR_PHASE_COUNT, num_regions, 3),
            device=self.device,
            dtype=torch.float32,
        )
        self._contact_prior_position_count = torch.zeros(
            (self.motion.num_clips, _CONTACT_PRIOR_PHASE_COUNT, num_regions),
            device=self.device,
            dtype=torch.float32,
        )

        if self.num_future_steps > 0 and self.target_pose_type is not None:
            self.future_target_poses = torch.zeros(
                self.num_envs,
                self.num_future_steps * self.num_obs_per_target_pose,
                device=self.device,
            )

        if self.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler.init_buffers()

        if self._clip_success_counts is not None:
            self._clip_success_counts.zero_()
        if self._clip_total_counts is not None:
            self._clip_total_counts.zero_()
        if self.clip_weighting_strategy == "success_rate_adaptive" and self._base_clip_weights is not None:
            self._raw_clip_sampling_weights = self._base_clip_weights.clone()
        self._refresh_current_clip_sampling_weights()
        if self.pickup_anchor_set is not None:
            self.pickup_anchor_set.zero_()
        if self.pickup_anchor_root_pos_w is not None:
            self.pickup_anchor_root_pos_w.zero_()
        if self.pickup_anchor_root_quat_w is not None:
            self.pickup_anchor_root_quat_w.zero_()
            self.pickup_anchor_root_quat_w[:, 3] = 1.0
        if self.pickup_anchor_object_pos_b is not None:
            self.pickup_anchor_object_pos_b.zero_()
        if self.pickup_anchor_object_quat_b is not None:
            self.pickup_anchor_object_quat_b.zero_()
            self.pickup_anchor_object_quat_b[:, 3] = 1.0
        if self.pickup_object_rel_z_baseline is not None:
            self.pickup_object_rel_z_baseline.zero_()
        if self.hybrid_velocity_object_z_baseline is not None:
            self.hybrid_velocity_object_z_baseline.zero_()
        if self.pickup_consecutive_counter is not None:
            self.pickup_consecutive_counter.zero_()
        if self._runtime_default_pose_prepend_active is not None:
            self._runtime_default_pose_prepend_active.zero_()
        if self._runtime_default_pose_prepend_step is not None:
            self._runtime_default_pose_prepend_step.zero_()

    def _update_motion_alignment(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        clip_ids = self.clip_ids[env_ids]
        clip_offsets = self.motion.clip_offsets[clip_ids]
        motion_root_quat = self.motion.body_quat_w[clip_offsets, 0]
        _, _, motion_yaw = get_euler_xyz(motion_root_quat, w_last=True)

        yaw_delta = self._init_root_yaw - motion_yaw
        zeros = torch.zeros_like(yaw_delta)
        align_quat = quat_from_euler_xyz(zeros, zeros, yaw_delta)
        self._align_quat[env_ids] = align_quat

        motion_root_pos = self.motion.body_pos_w[clip_offsets, 0]
        env_offsets = self._get_env_offsets(env_ids)
        desired_root_pos = env_offsets + self._init_root_pos
        aligned_root_pos = quat_apply(align_quat, motion_root_pos, w_last=True)
        self._align_pos[env_ids] = desired_root_pos - aligned_root_pos

    def _apply_motion_alignment_pos(
        self,
        pos: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        align_quat = self._align_quat if env_ids is None else self._align_quat[env_ids]
        align_pos = self._align_pos if env_ids is None else self._align_pos[env_ids]
        if pos.ndim == 3:
            align_quat = align_quat[:, None, :].expand(-1, pos.shape[1], -1)
            align_pos = align_pos[:, None, :]
        return quat_apply(align_quat, pos, w_last=True) + align_pos

    def _apply_motion_alignment_vec(
        self,
        vec: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        align_quat = self._align_quat if env_ids is None else self._align_quat[env_ids]
        if vec.ndim == 3:
            align_quat = align_quat[:, None, :].expand(-1, vec.shape[1], -1)
        return quat_apply(align_quat, vec, w_last=True)

    def _apply_motion_alignment_quat(
        self,
        quat: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        align_quat = self._align_quat if env_ids is None else self._align_quat[env_ids]
        if quat.ndim == 3:
            align_quat = align_quat[:, None, :].expand(-1, quat.shape[1], -1)
        return quat_mul(align_quat, quat, w_last=True)

    def _resolve_adaptive_sampling_contact_interval_root(self) -> Path | None:
        raw_root = getattr(self.motion_cfg, "adaptive_sampling_contact_interval_root", None)
        if raw_root is None:
            return None
        root_str = str(raw_root).strip()
        if not root_str:
            return None
        try:
            resolved = Path(resolve_data_file_path(root_str)).resolve()
        except Exception as exc:
            raise FileNotFoundError(
                f"Failed to resolve configured adaptive-sampling contact interval root '{root_str}': {exc}"
            ) from exc
        if not resolved.is_dir():
            raise FileNotFoundError(
                "Configured adaptive-sampling contact interval root does not exist or is not a directory: "
                f"'{resolved}'."
            )
        return resolved

    @staticmethod
    def _infer_clip_id_from_contact_export_dir_name(dir_name: str) -> str:
        return _infer_contact_export_clip_id(dir_name)

    def _load_adaptive_sampling_contact_window_from_dir(
        self,
        clip_dir: Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[int, int] | None:
        intervals_by_region: dict[str, Any] = {}
        contact_intervals_path = clip_dir / "contact_intervals.json"
        if contact_intervals_path.is_file():
            try:
                payload = json.loads(contact_intervals_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Skipping invalid adaptive contact intervals '{}': {}", contact_intervals_path, exc)
            else:
                if isinstance(payload, dict):
                    intervals_by_region.update(payload)

        if not intervals_by_region:
            for region_name, file_name in _ADAPTIVE_SAMPLING_CONTACT_INTERVAL_FALLBACK_FILES.items():
                interval_path = clip_dir / file_name
                if not interval_path.is_file():
                    continue
                try:
                    intervals_by_region[region_name] = np.load(
                        interval_path,
                        allow_pickle=False,
                    )
                except Exception as exc:
                    logger.warning("Skipping invalid adaptive contact interval '{}': {}", interval_path, exc)

        interval = _select_primary_contact_interval(intervals_by_region)
        if interval is None:
            return None

        interval = _convert_contact_interval_timebase(
            interval,
            metadata=metadata,
            motion_fps=float(getattr(getattr(self, "motion", None), "fps", 1.0)),
        )

        # Contact exports index physical rollout steps.  A multi-clip runtime
        # prepend holds motion time at zero for the warmup, so convert the
        # exported interval back to the motion-time coordinates used by AS,
        # button observations, and random-start sampling.  Single-clip
        # prepends are spliced into the motion itself and therefore need no
        # conversion.
        compensation_enabled = bool(
            getattr(self.motion_cfg, "contact_interval_runtime_prepend_compensation", False)
        )
        runtime_prepend_offset = (
            int(self._runtime_default_pose_prepend_steps)
            if compensation_enabled and bool(self._runtime_default_pose_prepend_enabled)
            else 0
        )
        if runtime_prepend_offset <= 0:
            return interval
        start_step = max(0, int(interval[0]) - runtime_prepend_offset)
        end_step = int(interval[1]) - runtime_prepend_offset
        if end_step <= start_step:
            return None
        return start_step, end_step

    def _configure_adaptive_sampling_contact_interval_bank(self) -> None:
        self._adaptive_sampling_contact_interval_root = None
        self._adaptive_sampling_contact_intervals_by_clip = {}
        self._adaptive_sampling_contact_window_by_clip = torch.full(
            (self.motion.num_clips, 2),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        self._adaptive_sampling_contact_window_valid_by_clip = torch.zeros(
            (self.motion.num_clips,),
            device=self.device,
            dtype=torch.bool,
        )

        configured_root = str(
            getattr(self.motion_cfg, "adaptive_sampling_contact_interval_root", None) or ""
        ).strip()
        observation_consumer = self._has_contact_window_observation_consumer()
        needs_contact_windows = (
            self.use_adaptive_timesteps_sampler
            or bool(self.motion_cfg.uniform_t1_window_sampling_enabled)
            or bool(configured_root and observation_consumer)
        )
        if not needs_contact_windows or not self.motion.has_object:
            return

        contact_root = self._resolve_adaptive_sampling_contact_interval_root()
        if contact_root is None:
            if bool(self.motion_cfg.uniform_t1_window_sampling_enabled):
                raise ValueError(
                    "uniform_t1_window_sampling_enabled=True requires a non-empty "
                    "adaptive_sampling_contact_interval_root."
                )
            return

        clip_name_to_index = {clip_name: idx for idx, clip_name in enumerate(self.motion.clip_ids)}
        loaded_count = 0
        clip_source_dirs: dict[int, Path] = {}
        for clip_dir in sorted(contact_root.iterdir()):
            if not clip_dir.is_dir():
                continue
            metadata_path = clip_dir / "metadata.json"
            metadata: dict[str, Any] = {}
            clip_id = ""
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    inferred_clip_id = _resolve_contact_export_clip_id(
                        clip_dir.name,
                        clip_name_to_index,
                    )
                    if inferred_clip_id in clip_name_to_index:
                        raise ValueError(
                            "Adaptive contact metadata is invalid for an active clip: "
                            f"clip={inferred_clip_id!r}, path={metadata_path}: {exc}"
                        ) from exc
                    logger.warning("Skipping invalid adaptive contact metadata '{}': {}", metadata_path, exc)
                    continue
                if not isinstance(metadata, dict):
                    inferred_clip_id = _resolve_contact_export_clip_id(
                        clip_dir.name,
                        clip_name_to_index,
                    )
                    if inferred_clip_id in clip_name_to_index:
                        raise ValueError(
                            "Adaptive contact metadata for an active clip must be a JSON object: "
                            f"clip={inferred_clip_id!r}, path='{metadata_path}'."
                        )
                    logger.warning(
                        "Skipping non-object adaptive contact metadata for inactive directory '{}'.",
                        metadata_path,
                    )
                    continue
                clip_id = str(metadata.get("clip_id", "")).strip()
            if not clip_id:
                clip_id = _resolve_contact_export_clip_id(clip_dir.name, clip_name_to_index)
            clip_index = clip_name_to_index.get(clip_id)
            if clip_index is None:
                continue
            previous_source = clip_source_dirs.get(clip_index)
            if previous_source is not None:
                raise RuntimeError(
                    "Multiple adaptive contact directories resolve to the same active clip: "
                    f"clip={clip_id!r}, first={previous_source}, second={clip_dir}."
                )
            clip_source_dirs[clip_index] = clip_dir
            interval = self._load_adaptive_sampling_contact_window_from_dir(
                clip_dir,
                metadata=metadata,
            )
            if interval is None:
                continue
            clip_length = int(self.motion.clip_lengths[clip_index].item())
            if interval[0] < 0 or interval[0] >= clip_length or interval[1] <= interval[0] or interval[1] > clip_length:
                raise ValueError(
                    "Adaptive contact interval is outside the active motion-time range after runtime-prepend "
                    f"conversion: clip={clip_id!r}, interval={interval}, clip_length={clip_length}."
                )
            self._adaptive_sampling_contact_window_by_clip[clip_index, 0] = int(interval[0])
            self._adaptive_sampling_contact_window_by_clip[clip_index, 1] = int(interval[1])
            self._adaptive_sampling_contact_window_valid_by_clip[clip_index] = True
            self._adaptive_sampling_contact_intervals_by_clip[clip_index] = (
                int(interval[0]),
                int(interval[1]),
            )
            loaded_count += 1

        require_complete_coverage = os.environ.get(
            "HOLOSOMA_REQUIRE_CONTACT_INTERVAL_COVERAGE",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        # A configured export root is part of the policy input contract when a
        # contact-aware observation consumes these windows.  Falling back to a
        # kinematic window for only the missing clips (or for the entire bank)
        # would change the meaning of a saved actor input across train/eval or
        # checkpoint resume.  Keep the no-root legacy fallback, and keep partial
        # banks available for sampler/metrics-only users, but make an explicitly
        # configured observation bank complete by construction.
        require_complete_coverage = require_complete_coverage or bool(
            configured_root and observation_consumer
        )
        if require_complete_coverage and loaded_count != self.motion.num_clips:
            missing_clip_ids = [
                str(self.motion.clip_ids[index])
                for index in range(self.motion.num_clips)
                if not bool(self._adaptive_sampling_contact_window_valid_by_clip[index].item())
            ]
            raise RuntimeError(
                "Complete adaptive contact-interval coverage is required, but valid motion-time windows were "
                f"loaded for only {loaded_count}/{self.motion.num_clips} clips. "
                f"missing_preview={missing_clip_ids[:20]}."
            )

        if loaded_count > 0:
            self._adaptive_sampling_contact_interval_root = contact_root
            logger.info(
                "Loaded adaptive-sampling contact windows for {} / {} clip(s) from '{}'.",
                loaded_count,
                self.motion.num_clips,
                contact_root,
            )
        else:
            message = f"No matching adaptive-sampling contact windows were found in '{contact_root}'."
            if bool(self.motion_cfg.uniform_t1_window_sampling_enabled):
                raise RuntimeError(
                    message
                    + " uniform_t1_window_sampling_enabled=True cannot be honored without matching windows."
                )
            logger.warning("{} Contact-window stage metrics will be skipped.", message)

    def _current_adaptive_sampling_clip_weights(self) -> torch.Tensor:
        if self.motion.num_clips <= 1:
            return torch.ones((1,), device=self.device, dtype=torch.float32)

        if self._clip_sampling_weights is not None:
            return self._clip_sampling_weights.to(device=self.device, dtype=torch.float32)

        if self._fixed_clip_ids is not None and self._fixed_clip_ids.numel() > 0:
            counts = torch.bincount(self._fixed_clip_ids, minlength=self.motion.num_clips).to(
                device=self.device, dtype=torch.float32
            )
            total = counts.sum()
            if float(total.item()) > 0.0:
                return counts / total

        return torch.full(
            (self.motion.num_clips,),
            1.0 / float(max(self.motion.num_clips, 1)),
            device=self.device,
            dtype=torch.float32,
        )

    def _compute_adaptive_sampling_contact_stage_metrics(
        self,
        timestep_probability_overrides: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.adaptive_timesteps_sampler is None:
            return {}
        if self._adaptive_sampling_contact_window_by_clip is None or self._adaptive_sampling_contact_window_valid_by_clip is None:
            return {}
        valid_clip_indices = tuple(self._adaptive_sampling_contact_intervals_by_clip)
        if not valid_clip_indices:
            return {}

        valid_mask = self._adaptive_sampling_contact_window_valid_by_clip
        clip_weights = self._current_adaptive_sampling_clip_weights()
        valid_clip_index_tensor = torch.as_tensor(
            valid_clip_indices,
            device=self.device,
            dtype=torch.long,
        )
        valid_clip_prob_mass = clip_weights[valid_clip_index_tensor].sum()
        if float(valid_clip_prob_mass.item()) <= 0.0:
            return {}

        stage_prob_masses = torch.zeros((5,), device=self.device, dtype=torch.float32)
        after_t2_prob_mass = torch.zeros((), device=self.device, dtype=torch.float32)
        for clip_idx in valid_clip_indices:
            clip_weight = clip_weights[clip_idx].to(dtype=torch.float32)
            sampling_probabilities = (
                self._effective_adaptive_timestep_probabilities_for_clip(clip_idx)
                if timestep_probability_overrides is None
                else timestep_probability_overrides[clip_idx]
            )
            sample_end_step = float(sampling_probabilities.numel())
            t1, t2 = self._adaptive_sampling_contact_intervals_by_clip[clip_idx]
            stage_intervals, after_t2_interval = _compute_contact_stage_intervals(
                t1=t1,
                t2=t2,
                sample_end_step=sample_end_step,
            )
            stage_masses = _probability_mass_on_intervals(
                sampling_probabilities,
                sample_end_step=sample_end_step,
                intervals=stage_intervals,
            )
            after_t2_mass = _probability_mass_on_intervals(
                sampling_probabilities,
                sample_end_step=sample_end_step,
                intervals=[after_t2_interval],
            )[0]
            stage_prob_masses += clip_weight * stage_masses
            after_t2_prob_mass += clip_weight * after_t2_mass

        tracked_prob_mass = stage_prob_masses.sum()
        missing_clip_prob_mass = torch.clamp(
            torch.tensor(1.0, device=self.device, dtype=torch.float32) - valid_clip_prob_mass,
            min=0.0,
        )
        return {
            "contact_interval_valid_clip_fraction": valid_mask.to(dtype=torch.float32).mean(),
            "contact_interval_valid_clip_prob_mass": valid_clip_prob_mass,
            "contact_interval_missing_clip_prob_mass": missing_clip_prob_mass,
            "contact_interval_after_t2_prob_mass": after_t2_prob_mass,
            "contact_interval_tracked_prob_mass": tracked_prob_mass,
            "contact_interval_stage_0_to_t1p10_prob_mass": stage_prob_masses[0],
            "contact_interval_stage_mid1_prob_mass": stage_prob_masses[1],
            "contact_interval_stage_mid2_prob_mass": stage_prob_masses[2],
            "contact_interval_stage_mid3_prob_mass": stage_prob_masses[3],
            "contact_interval_stage_t2m30_to_t2_prob_mass": stage_prob_masses[4],
        }

    def _clip_start_object_pos_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            return torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        start_steps = torch.zeros_like(self.time_steps)
        motion_idx = self._get_motion_indices(start_steps)
        pos = self.motion.object_pos_w[motion_idx]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_pos(pos)
        return pos + self._get_env_offsets()

    def _clip_start_root_pos_w(self) -> torch.Tensor:
        start_steps = torch.zeros_like(self.time_steps)
        motion_idx = self._get_motion_indices(start_steps)
        pos = self.motion.body_pos_w[motion_idx, 0]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_pos(pos)
        return pos + self._get_env_offsets()

    def _object_contact_force_magnitude_for_bodies(self, body_names: tuple[str, ...]) -> torch.Tensor:
        zeros = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        simulator_body_names = set(getattr(self._env.simulator, "body_names", []) or [])
        selected_names = [name for name in body_names if name in simulator_body_names]
        if not selected_names:
            return zeros

        try:
            force_history = self.get_body_object_contact_force_history(selected_names)
        except Exception:
            return zeros
        if not isinstance(force_history, torch.Tensor) or force_history.ndim != 4 or force_history.shape[2] == 0:
            return zeros
        current_forces = force_history[:, 0].to(device=self.device, dtype=torch.float32)
        return torch.amax(torch.linalg.norm(current_forces, dim=-1), dim=1)

    def _update_empty_lift_diagnostic_metrics(self, zeros: torch.Tensor) -> None:
        for key in (
            "lift/object_z",
            "lift/ref_object_z",
            "lift/root_z",
            "lift/object_minus_root_z",
            "lift/object_z_delta_from_clip_start",
            "lift/ref_object_z_delta_from_clip_start",
            "lift/root_z_delta_from_clip_start",
            "lift/object_minus_root_z_delta_from_clip_start",
            "lift/left_wrist_object_force",
            "lift/right_wrist_object_force",
            "lift/left_hand_object_force",
            "lift/right_hand_object_force",
            "lift/max_hand_object_force",
            "lift/hand_contact_frac",
            "lift/world_lift_frac",
            "lift/world_lift_with_hand_contact_frac",
            "lift/relative_lift_false_positive_frac",
            "lift/object_contact_getter_available",
            "lift/hand_contact_body_available",
        ):
            self.metrics[key] = zeros

    def _update_lift_diagnostic_metrics(self) -> None:
        sim_object_pos = self.simulator_object_pos_w
        ref_object_pos = self.object_pos_w
        robot_root_pos = self.robot_root_pos_w
        clip_start_object_pos = self._clip_start_object_pos_w()
        clip_start_root_pos = self._clip_start_root_pos_w()

        object_z = sim_object_pos[:, 2]
        ref_object_z = ref_object_pos[:, 2]
        root_z = robot_root_pos[:, 2]
        object_minus_root_z = object_z - root_z
        object_z_delta = object_z - clip_start_object_pos[:, 2]
        ref_object_z_delta = ref_object_z - clip_start_object_pos[:, 2]
        root_z_delta = root_z - clip_start_root_pos[:, 2]
        object_minus_root_z_delta = object_minus_root_z - (
            clip_start_object_pos[:, 2] - clip_start_root_pos[:, 2]
        )

        left_wrist_force = self._object_contact_force_magnitude_for_bodies(("left_wrist_yaw_link",))
        right_wrist_force = self._object_contact_force_magnitude_for_bodies(("right_wrist_yaw_link",))
        left_hand_force = self._object_contact_force_magnitude_for_bodies(("left_wrist_yaw_link", "left_rubber_hand"))
        right_hand_force = self._object_contact_force_magnitude_for_bodies(
            ("right_wrist_yaw_link", "right_rubber_hand")
        )
        max_hand_force = torch.maximum(left_hand_force, right_hand_force)

        world_lift = object_z_delta > _LIFT_DIAGNOSTIC_OBJECT_LIFT_HEIGHT_THRESHOLD
        hand_contact = max_hand_force > _LIFT_DIAGNOSTIC_MIN_CONTACT_FORCE
        relative_lift = object_minus_root_z_delta > _LIFT_DIAGNOSTIC_OBJECT_LIFT_HEIGHT_THRESHOLD
        false_positive = (
            relative_lift
            & (object_z_delta <= _LIFT_DIAGNOSTIC_FALSE_POSITIVE_MAX_WORLD_LIFT)
            & (root_z_delta < -_LIFT_DIAGNOSTIC_FALSE_POSITIVE_MIN_ROOT_DROP)
        )

        simulator_body_names = set(getattr(self._env.simulator, "body_names", []) or [])
        hand_body_available = any(
            name in simulator_body_names
            for name in ("left_wrist_yaw_link", "right_wrist_yaw_link", "left_rubber_hand", "right_rubber_hand")
        )
        hand_body_available_tensor = torch.full_like(object_z, 1.0 if hand_body_available else 0.0)
        getter_available_tensor = torch.full_like(
            object_z,
            1.0 if getattr(self._env.simulator, "get_object_contact_force_history", None) is not None else 0.0,
        )

        # These three values can be views into simulator/motion storage.  The
        # logging meter retains references until the PPO iteration boundary;
        # snapshot them so later simulator updates cannot rewrite earlier
        # diagnostic samples in place.
        self.metrics["lift/object_z"] = object_z.clone()
        self.metrics["lift/ref_object_z"] = ref_object_z.clone()
        self.metrics["lift/root_z"] = root_z.clone()
        self.metrics["lift/object_minus_root_z"] = object_minus_root_z
        self.metrics["lift/object_z_delta_from_clip_start"] = object_z_delta
        self.metrics["lift/ref_object_z_delta_from_clip_start"] = ref_object_z_delta
        self.metrics["lift/root_z_delta_from_clip_start"] = root_z_delta
        self.metrics["lift/object_minus_root_z_delta_from_clip_start"] = object_minus_root_z_delta
        self.metrics["lift/left_wrist_object_force"] = left_wrist_force
        self.metrics["lift/right_wrist_object_force"] = right_wrist_force
        self.metrics["lift/left_hand_object_force"] = left_hand_force
        self.metrics["lift/right_hand_object_force"] = right_hand_force
        self.metrics["lift/max_hand_object_force"] = max_hand_force
        self.metrics["lift/hand_contact_frac"] = hand_contact.to(dtype=torch.float32)
        self.metrics["lift/world_lift_frac"] = world_lift.to(dtype=torch.float32)
        self.metrics["lift/world_lift_with_hand_contact_frac"] = (world_lift & hand_contact).to(dtype=torch.float32)
        self.metrics["lift/relative_lift_false_positive_frac"] = false_positive.to(dtype=torch.float32)
        self.metrics["lift/object_contact_getter_available"] = getter_available_tensor
        self.metrics["lift/hand_contact_body_available"] = hand_body_available_tensor

    @staticmethod
    def supported_live_metric_keys() -> frozenset[str]:
        """Metrics that an enabled curriculum may request at every environment step."""

        return _CURRICULUM_LIVE_TRACKING_ERROR_KEYS

    def _update_tracking_error_metrics(self, metric_keys: frozenset[str] | None = None) -> None:
        """Refresh selected tracking errors using the same expressions as a full metrics update."""

        requested = _CURRICULUM_LIVE_TRACKING_ERROR_KEYS if metric_keys is None else metric_keys

        if "motion/error_ref_pos" in requested:
            self.metrics["motion/error_ref_pos"] = torch.norm(self.ref_pos_w - self.robot_ref_pos_w, dim=-1)
        if "motion/error_ref_rot" in requested:
            self.metrics["motion/error_ref_rot"] = quat_error_magnitude(self.ref_quat_w, self.robot_ref_quat_w)
        if "motion/error_ref_lin_vel" in requested:
            self.metrics["motion/error_ref_lin_vel"] = torch.norm(
                self.ref_lin_vel_w - self.robot_ref_lin_vel_w,
                dim=-1,
            )
        if "motion/error_ref_ang_vel" in requested:
            self.metrics["motion/error_ref_ang_vel"] = torch.norm(
                self.ref_ang_vel_w - self.robot_ref_ang_vel_w,
                dim=-1,
            )

        if "motion/error_body_pos" in requested:
            self.metrics["motion/error_body_pos"] = torch.norm(
                self.body_pos_relative_w - self.robot_body_pos_w,
                dim=-1,
            ).mean(dim=-1)
        if "motion/error_body_rot" in requested:
            self.metrics["motion/error_body_rot"] = quat_error_magnitude(
                self.body_quat_relative_w,
                self.robot_body_quat_w,
            ).mean(dim=-1)
        if "motion/error_body_lin_vel" in requested:
            self.metrics["motion/error_body_lin_vel"] = torch.norm(
                self.body_lin_vel_w - self.robot_body_lin_vel_w,
                dim=-1,
            ).mean(dim=-1)
        if "motion/error_body_ang_vel" in requested:
            self.metrics["motion/error_body_ang_vel"] = torch.norm(
                self.body_ang_vel_w - self.robot_body_ang_vel_w,
                dim=-1,
            ).mean(dim=-1)

        if "motion/error_joint_pos" in requested:
            self.metrics["motion/error_joint_pos"] = torch.norm(
                self.joint_pos - self.robot_joint_pos,
                dim=-1,
            )
        if "motion/error_joint_vel" in requested:
            self.metrics["motion/error_joint_vel"] = torch.norm(
                self.joint_vel - self.robot_joint_vel,
                dim=-1,
            )

        object_metric_keys = requested.intersection(
            {
                "motion/error_object_ref_pos",
                "motion/error_object_ref_rot",
                "motion/error_object_ref_lin_vel",
            }
        )
        if not object_metric_keys:
            return
        if self.motion.has_object:
            if "motion/error_object_ref_pos" in object_metric_keys:
                self.metrics["motion/error_object_ref_pos"] = torch.norm(
                    self.object_pos_w - self.simulator_object_pos_w,
                    dim=-1,
                )
            if "motion/error_object_ref_rot" in object_metric_keys:
                self.metrics["motion/error_object_ref_rot"] = quat_error_magnitude(
                    self.object_quat_w,
                    self.simulator_object_quat_w,
                )
            if "motion/error_object_ref_lin_vel" in object_metric_keys:
                self.metrics["motion/error_object_ref_lin_vel"] = torch.norm(
                    self.object_lin_vel_w - self.simulator_object_lin_vel_w,
                    dim=-1,
                )
            return

        # Allocate on every call.  Logging meters retain tensor references, so
        # mutating a cached zero buffer would rewrite samples from earlier steps.
        zeros = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        for metric_key in object_metric_keys:
            self.metrics[metric_key] = zeros

    def update_live_metrics(self, metric_keys) -> None:
        """Refresh only curriculum-consumed tracking metrics before reset handling."""

        requested = frozenset(str(metric_key) for metric_key in metric_keys)
        unsupported = requested - _CURRICULUM_LIVE_TRACKING_ERROR_KEYS
        if unsupported:
            raise ValueError(
                "Unsupported live motion metric key(s): "
                f"{sorted(unsupported)}. Supported keys are "
                f"{sorted(_CURRICULUM_LIVE_TRACKING_ERROR_KEYS)}."
            )
        self._update_tracking_error_metrics(requested)

    def update_metrics(self):
        """Update full tracking and diagnostic telemetry after an environment action."""

        self._update_tracking_error_metrics()

        # Lift/contact metrics are diagnostics only.  Keep them out of the
        # per-step curriculum path because they query motion starts and contact
        # history repeatedly.
        if self.motion.has_object:
            self._update_lift_diagnostic_metrics()
        else:
            zeros = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            self._update_empty_lift_diagnostic_metrics(zeros)

        if self.hybrid_velocity_enabled():
            task_mask = self.get_hybrid_velocity_task_env_mask()
            task_active = self.get_hybrid_velocity_task_active_mask()
            task_count = task_mask.to(dtype=torch.float32).sum().clamp_min(1.0)
            lifted_fraction_of_task = (
                task_active.to(dtype=torch.float32).sum() / task_count
            )
            self.metrics["hybrid_velocity/task_env_fraction_active"] = task_mask.to(
                dtype=torch.float32
            )
            self.metrics["hybrid_velocity/task_env_fraction_target"] = torch.full(
                (self.num_envs,),
                self._current_hybrid_velocity_task_env_fraction(),
                device=self.device,
                dtype=torch.float32,
            )
            self.metrics["hybrid_velocity/lifted_fraction_of_task"] = (
                lifted_fraction_of_task.expand(self.num_envs)
            )

        if self.hmi_enabled():
            track_mask = self.get_hmi_track_env_mask()
            gen_mask = self.get_hmi_gen_env_mask()
            self.metrics["hmi/track_env"] = track_mask.to(dtype=torch.float32)
            self.metrics["hmi/gen_env"] = gen_mask.to(dtype=torch.float32)
            if self.hmi_goal_object_pos_w is not None:
                goal_pos_error = torch.linalg.vector_norm(
                    self.hmi_goal_object_pos_w - self.simulator_object_pos_w,
                    dim=-1,
                )
                self.metrics["hmi/object_goal_pos_error"] = goal_pos_error
            if self.hmi_goal_object_quat_w is not None:
                self.metrics["hmi/object_goal_ori_error"] = quat_error_magnitude(
                    self.hmi_goal_object_quat_w,
                    self.simulator_object_quat_w,
                )
            if self.hmi_goal_noise_scale is not None:
                self.metrics["hmi/goal_noise_scale"] = self.hmi_goal_noise_scale.expand(
                    self.num_envs
                )
            if self.hmi_goal_success_ema is not None:
                self.metrics["hmi/goal_success_ema"] = self.hmi_goal_success_ema.expand(
                    self.num_envs
                )

        if self.precomputed_turn_then_forward_enabled():
            phase = self.get_precomputed_turn_then_forward_phase()
            command = self.get_precomputed_turn_then_forward_command()
            self.metrics["precomputed_command/zero_phase"] = (phase == 0).to(
                dtype=torch.float32
            )
            self.metrics["precomputed_command/forward_phase"] = (phase == 1).to(
                dtype=torch.float32
            )
            self.metrics["precomputed_command/yaw_phase"] = (phase == 2).to(
                dtype=torch.float32
            )
            self.metrics["precomputed_command/forward_value_m"] = command[:, 0]
            self.metrics["precomputed_command/abs_yaw_value_rad"] = command[:, 2].abs()

        self.metrics["motion/reset_start_at_timestep_zero_prob"] = torch.full(
            (self.num_envs,),
            float(self._current_start_at_timestep_zero_prob()),
            device=self.device,
            dtype=torch.float32,
        )
        if bool(self.motion_cfg.uniform_t1_window_sampling_enabled):
            uniform_t1_stats = {
                "enabled": 1.0 if self._uniform_t1_window_sampling_active() else 0.0,
                "density_boost": float(self.motion_cfg.uniform_t1_window_density_boost),
                "target_sample_frac": (
                    -1.0
                    if self.motion_cfg.uniform_t1_window_target_sample_frac is None
                    else float(self.motion_cfg.uniform_t1_window_target_sample_frac)
                ),
                "half_width_steps": float(self.motion_cfg.uniform_t1_window_half_width_steps),
                "last_reset_available_frac": self._uniform_t1_window_last_reset_available_frac,
                "last_reset_sample_frac": self._uniform_t1_window_last_reset_sample_frac,
                "last_reset_expected_sample_frac": self._uniform_t1_window_last_reset_expected_sample_frac,
                "last_reset_sample_frac_valid": self._uniform_t1_window_last_reset_sample_frac_valid,
                "last_reset_expected_sample_frac_valid": self._uniform_t1_window_last_reset_expected_sample_frac_valid,
                "last_reset_mean_window_len": self._uniform_t1_window_last_reset_mean_window_len,
            }
            for metric_name, metric_value in uniform_t1_stats.items():
                if isinstance(metric_value, torch.Tensor):
                    self.metrics[f"motion/reset_uniform_t1_window_{metric_name}"] = metric_value.to(
                        device=self.device,
                        dtype=torch.float32,
                    ).expand(self.num_envs)
                else:
                    self.metrics[f"motion/reset_uniform_t1_window_{metric_name}"] = torch.full(
                        (self.num_envs,),
                        float(metric_value),
                        device=self.device,
                        dtype=torch.float32,
                    )
        self.metrics["motion/reset_freeze_at_timestep_zero_prob"] = torch.full(
            (self.num_envs,),
            float(self._current_freeze_at_timestep_zero_prob()),
            device=self.device,
            dtype=torch.float32,
        )
        clean_prob = self._current_clean_group_probability()
        if clean_prob is not None and self._clean_clip_mask is not None and self._clip_sampling_weights is not None:
            clean_weight = float(self._clip_sampling_weights[self._clean_clip_mask].sum().item())
            self.metrics["motion/clean_clip_target_prob"] = torch.full(
                (self.num_envs,),
                float(clean_prob),
                device=self.device,
                dtype=torch.float32,
            )
            self.metrics["motion/clean_clip_sample_weight"] = torch.full(
                (self.num_envs,),
                clean_weight,
                device=self.device,
                dtype=torch.float32,
            )
            self.metrics["motion/noisy_clip_sample_weight"] = torch.full(
                (self.num_envs,),
                max(0.0, 1.0 - clean_weight),
                device=self.device,
                dtype=torch.float32,
            )

        if self.use_adaptive_timesteps_sampler:
            (
                timestep_probabilities_by_clip,
                bin_probabilities_by_clip,
            ) = self._effective_adaptive_probability_views()
            self.adaptive_timesteps_sampler.get_stats(bin_probabilities_by_clip)
            self.metrics["motion/adaptive_timesteps_sampler_entropy"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_entropy"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_top1_prob"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_top1_prob"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_top1_bin"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_top1_bin"
            ]
            for metric_name, metric_value in self._compute_adaptive_sampling_contact_stage_metrics(
                timestep_probabilities_by_clip
            ).items():
                self.metrics[f"motion/adaptive_timesteps_sampler_{metric_name}"] = metric_value

    #########################################################################################
    ## Internal helpers
    #########################################################################################
    def _configure_motion_terrain_pairs(self) -> None:
        self._clip_terrain_offsets = None
        self._clip_terrain_offsets_by_row = None
        self._terrain_row_ids = None
        self._terrain_row_stride = 0.0
        self._terrain_row_count = 0
        if not self.motion_cfg.pair_terrain_with_motion:
            return

        terrain_state = self._env.terrain_manager.get_state("locomotion_terrain")
        terrain = getattr(terrain_state, "terrain", None)
        tile_names = getattr(terrain, "obj_tile_names", []) if terrain is not None else []
        tile_offsets = getattr(terrain, "obj_tile_offsets", None) if terrain is not None else None
        tile_stride = getattr(terrain, "obj_tile_stride", None) if terrain is not None else None
        tile_rows = int(getattr(terrain, "obj_tile_rows", 0) or 0) if terrain is not None else 0

        if tile_names and tile_offsets is not None and tile_stride is not None and tile_rows > 0:
            if len(set(tile_names)) != len(tile_names):
                raise ValueError("Duplicate OBJ tile names detected; stems must be unique for pairing.")

            tile_offsets = np.asarray(tile_offsets, dtype=np.float32)
            if tile_offsets.shape[0] != len(tile_names):
                raise ValueError("OBJ tile offsets length does not match tile names.")
            stride = np.asarray(tile_stride, dtype=np.float32).reshape(-1)
            if stride.size < 2:
                raise ValueError("OBJ tile stride must provide at least X/Y spacing.")

            name_to_idx = {name: idx for idx, name in enumerate(tile_names)}
            missing = [clip_id for clip_id in self.motion.clip_ids if clip_id not in name_to_idx]
            if missing:
                raise ValueError(f"Missing terrain OBJ for clips: {missing}")

            clip_offsets = np.stack([tile_offsets[name_to_idx[clip_id]] for clip_id in self.motion.clip_ids], axis=0)
            row_offsets = np.repeat(clip_offsets[None, :, :], repeats=max(1, tile_rows), axis=0)
            if row_offsets.shape[0] > 1:
                row_offsets[:, :, 1] += np.arange(row_offsets.shape[0], dtype=np.float32)[:, None] * float(stride[1])
            self._clip_terrain_offsets = torch.tensor(clip_offsets, device=self.device, dtype=torch.float32)
            self._clip_terrain_offsets_by_row = torch.tensor(row_offsets, device=self.device, dtype=torch.float32)
            self._terrain_row_stride = float(stride[1])
            self._terrain_row_count = max(1, tile_rows)
            self._terrain_row_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

            capacity = self._terrain_row_count * len(tile_names)
            if self.num_envs > capacity:
                raise ValueError(
                    "pair_terrain_with_motion requires terrain slots >= envs per rank "
                    f"(got num_envs={self.num_envs}, rows={self._terrain_row_count}, "
                    f"cols={len(tile_names)}, capacity={capacity}). "
                    "Increase terrain num_rows or reduce num_envs."
                )

            unused = [name for name in tile_names if name not in self.motion.clip_ids]
            if unused:
                logger.warning("Unused terrain OBJ tiles (no matching motion clip): {}", unused)

            logger.info("Motion/terrain pairing enabled for {} clips.", len(self.motion.clip_ids))
            return

        origin_grid = None
        if terrain is not None and hasattr(terrain, "env_origin_grid"):
            origin_grid = getattr(terrain, "env_origin_grid")
        elif terrain is not None and hasattr(terrain, "_env_origins"):
            origin_grid = getattr(terrain, "_env_origins")

        if origin_grid is None:
            raise ValueError(
                "pair_terrain_with_motion requires terrain tile metadata or a terrain origin grid. "
                "For OBJ pairing, set --terrain.terrain-term.obj-file-path to named OBJ tiles."
            )

        origin_grid_np = np.asarray(origin_grid, dtype=np.float32)
        if origin_grid_np.ndim != 3 or origin_grid_np.shape[2] < 3:
            raise ValueError(
                "Terrain origin grid must have shape (rows, cols, 3) to pair motion clips with terrain columns."
            )
        if origin_grid_np.shape[2] > 3:
            origin_grid_np = origin_grid_np[:, :, :3]

        num_rows, num_cols, _ = origin_grid_np.shape
        num_clips = len(self.motion.clip_ids)
        if num_cols < num_clips:
            raise ValueError(
                "pair_terrain_with_motion requires terrain columns >= motion clips "
                f"(got num_cols={num_cols}, num_clips={num_clips})."
            )

        clip_offsets_by_row = origin_grid_np[:, :num_clips, :]
        self._clip_terrain_offsets = torch.tensor(clip_offsets_by_row[0], device=self.device, dtype=torch.float32)
        self._clip_terrain_offsets_by_row = torch.tensor(clip_offsets_by_row, device=self.device, dtype=torch.float32)
        self._terrain_row_count = max(1, num_rows)
        self._terrain_row_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        capacity = self._terrain_row_count * num_clips
        if self.num_envs > capacity:
            raise ValueError(
                "pair_terrain_with_motion requires terrain slots >= envs per rank "
                f"(got num_envs={self.num_envs}, rows={self._terrain_row_count}, "
                f"clip_paired_cols={num_clips}, capacity={capacity}). "
                "Increase terrain num_rows or reduce num_envs."
            )

        if num_cols > num_clips:
            logger.warning(
                "Terrain has more columns ({}) than motion clips ({}); extra columns are unused for pairing.",
                num_cols,
                num_clips,
            )

        logger.info(
            "Motion/terrain pairing enabled for {} clips using terrain origin-grid column order.",
            len(self.motion.clip_ids),
        )

    def _configure_target_pose_settings(self) -> None:
        self.num_future_steps = int(self.motion_cfg.num_future_steps)
        self.target_pose_type = self.motion_cfg.target_pose_type
        self.num_obs_per_target_pose = 0
        self.future_target_poses: torch.Tensor | None = None

        if self.num_future_steps <= 0:
            return
        if self.target_pose_type is None:
            raise ValueError("target_pose_type must be set when num_future_steps > 0.")

        include_time = self._target_pose_includes_time(self.target_pose_type)
        num_bodies = len(self.motion_cfg.body_names_to_track)
        self.num_obs_per_target_pose = num_bodies * 18 + (1 if include_time else 0)

    def _target_pose_includes_time(self, target_pose_type: str) -> bool:
        if target_pose_type == "max-coords-future-rel":
            return False
        if target_pose_type == "max-coords-future-rel-with-time":
            return True
        raise ValueError(f"Unknown target_pose_type '{target_pose_type}'.")

    def _update_future_target_poses(self) -> None:
        if self.num_future_steps <= 0 or self.target_pose_type is None:
            return
        if self.future_target_poses is None:
            return
        self.future_target_poses[:] = self._compute_future_target_poses(
            num_future_steps=self.num_future_steps,
            target_pose_type=self.target_pose_type,
        )

    def _compute_future_target_poses(self, num_future_steps: int, target_pose_type: str) -> torch.Tensor:
        include_time = self._target_pose_includes_time(target_pose_type)

        time_offsets = torch.arange(1, num_future_steps + 1, device=self.device, dtype=torch.long)
        future_steps = self.time_steps.unsqueeze(1) + time_offsets.unsqueeze(0)
        max_steps = self._current_clip_lengths().unsqueeze(1) - 1
        future_steps = torch.minimum(future_steps, max_steps)

        times = (future_steps - self.time_steps.unsqueeze(1)).to(dtype=torch.float32) * self._env.dt
        future_steps_global = self._get_motion_indices(future_steps)

        target_body_pos = (
            self.motion.body_pos_w[future_steps_global][:, :, self.tracked_body_indexes]
            + self._get_env_offsets()[:, None, None, :]
        )
        target_body_rot = self.motion.body_quat_w[future_steps_global][:, :, self.tracked_body_indexes]

        reference_body_pos = target_body_pos.roll(shifts=1, dims=1)
        reference_body_pos[:, 0] = self.body_pos_w
        reference_body_rot = target_body_rot.roll(shifts=1, dims=1)
        reference_body_rot[:, 0] = self.body_quat_w

        reference_root_pos = reference_body_pos[:, :, 0, :]
        reference_root_rot = reference_body_rot[:, :, 0, :]

        heading_quat = yaw_quat(reference_root_rot, w_last=True)
        heading_inv = quat_inverse(heading_quat, w_last=True)
        heading_inv = heading_inv.unsqueeze(2).expand(-1, -1, target_body_pos.shape[2], -1)

        target_rel_body_pos = target_body_pos - reference_body_pos
        target_body_pos_rel_root = target_body_pos - reference_root_pos.unsqueeze(2)

        flat_heading_inv = heading_inv.reshape(-1, 4)
        flat_rel_body_pos = target_rel_body_pos.reshape(-1, 3)
        flat_body_pos = target_body_pos_rel_root.reshape(-1, 3)

        flat_rel_body_pos = quat_apply(flat_heading_inv, flat_rel_body_pos, w_last=True)
        flat_body_pos = quat_apply(flat_heading_inv, flat_body_pos, w_last=True)

        rel_body_pos = flat_rel_body_pos.reshape(
            self.num_envs, num_future_steps, target_body_pos.shape[2] * 3
        )
        body_pos = flat_body_pos.reshape(
            self.num_envs, num_future_steps, target_body_pos.shape[2] * 3
        )

        rel_body_rot = quat_mul(
            quat_conjugate(reference_body_rot, w_last=True),
            target_body_rot,
            w_last=True,
        )
        body_rot = quat_mul(heading_inv, target_body_rot, w_last=True)

        rel_body_rot_mat = quaternion_to_matrix(rel_body_rot.reshape(-1, 4), w_last=True)
        body_rot_mat = quaternion_to_matrix(body_rot.reshape(-1, 4), w_last=True)

        rel_body_rot_obs = rel_body_rot_mat[..., :2].reshape(
            self.num_envs, num_future_steps, target_body_pos.shape[2] * 6
        )
        body_rot_obs = body_rot_mat[..., :2].reshape(
            self.num_envs, num_future_steps, target_body_pos.shape[2] * 6
        )

        obs = torch.cat((rel_body_pos, body_pos, rel_body_rot_obs, body_rot_obs), dim=-1)

        if include_time:
            obs = torch.cat((obs, times.unsqueeze(-1)), dim=-1)

        return obs.reshape(self.num_envs, -1)

    def get_future_target_poses(
        self, *, num_future_steps: int | None = None, target_pose_type: str | None = None
    ) -> torch.Tensor:
        if num_future_steps is None and target_pose_type is None:
            if self.future_target_poses is None:
                return torch.zeros(self.num_envs, 0, device=self.device)
            return self.future_target_poses

        resolved_steps = self.num_future_steps if num_future_steps is None else num_future_steps
        resolved_type = self.target_pose_type if target_pose_type is None else target_pose_type
        if resolved_steps <= 0 or resolved_type is None:
            return torch.zeros(self.num_envs, 0, device=self.device)
        return self._compute_future_target_poses(resolved_steps, resolved_type)

    def _maybe_add_default_pose_transition(self, *, prepend: bool) -> None:
        """Shared path for optionally inserting default-pose interpolation before/after the clip."""
        applied_steps_attr = (
            "_static_default_pose_prepend_steps"
            if prepend
            else "_static_default_pose_append_steps"
        )
        setattr(self, applied_steps_attr, 0)
        if self._uses_global_multi_clip_transition_semantics():
            if prepend:
                logger.warning(
                    "Skipping in-place default pose transitions for global multi-clip motion semantics."
                )
            return
        enabled = self.motion_cfg.enable_default_pose_prepend if prepend else self.motion_cfg.enable_default_pose_append
        if not enabled:
            return

        duration = (
            self.motion_cfg.default_pose_prepend_duration_s
            if prepend
            else self.motion_cfg.default_pose_append_duration_s
        )
        if duration <= 0.0:
            return

        num_steps = round(duration / self._env.dt)
        if num_steps > MAX_MOTION_TRANSITION_STEPS:
            raise ValueError(
                "Default-pose {} transition requires {} steps, exceeding the deployment-safe maximum {}. "
                "Reduce the duration or increase the control timestep.".format(
                    "prepend" if prepend else "append",
                    num_steps,
                    MAX_MOTION_TRANSITION_STEPS,
                )
            )
        if num_steps <= 1:
            logger.warning(
                "Default pose {} duration {}s is too short for dt {}; skipping augmentation.",
                "prepend" if prepend else "append",
                duration,
                self._env.dt,
            )
            return

        default_state = self._build_default_pose_state(use_motion_end=not prepend)

        action = "prepend" if prepend else "append"
        log_str = f"{action} {num_steps} interpolated frames ({duration}s) from default pose to motion"
        try:
            self._add_transition_to_motion(default_state, num_steps, prepend=prepend)
            setattr(self, applied_steps_attr, int(num_steps))
            logger.info(log_str)
        except Exception as exc:
            logger.error(f"Failed to {action} default pose transition: {exc}")
            raise RuntimeError(
                f"Critical error during motion interpolation setup: {exc}\n"
                "This indicates a mismatch in tensor dimensions during interpolation. "
                "Please check that the motion file and robot configuration are compatible."
            ) from exc

    def _configure_runtime_default_pose_prepend(self) -> None:
        self._runtime_default_pose_prepend_enabled = False
        self._runtime_default_pose_prepend_steps = 0
        self._runtime_default_pose_prepend_defaults = {}
        self._runtime_default_pose_prepend_active = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._runtime_default_pose_prepend_step = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)

        if not self._uses_global_multi_clip_transition_semantics() or not self.motion_cfg.enable_default_pose_prepend:
            return

        duration = self.motion_cfg.default_pose_prepend_duration_s
        if duration <= 0.0:
            return

        if self._env.simulator.get_simulator_type() != SimulatorType.ISAACSIM:
            logger.warning("Runtime default-pose prepend only supports IsaacSim; disabling multi-clip prepend.")
            return

        num_steps = round(duration / self._env.dt)
        if num_steps > MAX_MOTION_TRANSITION_STEPS:
            raise ValueError(
                "Runtime default-pose prepend requires {} steps, exceeding the deployment-safe maximum {}. "
                "Reduce the duration or increase the control timestep.".format(
                    num_steps,
                    MAX_MOTION_TRANSITION_STEPS,
                )
            )
        if num_steps <= 1:
            logger.warning(
                "Runtime default pose prepend duration {}s is too short for dt {}; disabling multi-clip prepend.",
                duration,
                self._env.dt,
            )
            return

        default_states = [
            self._build_default_pose_state_robot_order(int(motion_idx.item()))
            for motion_idx in self.motion.clip_offsets
        ]
        if not default_states:
            return

        self._runtime_default_pose_prepend_defaults = {
            "joint_pos": torch.stack([state["joint_pos"] for state in default_states], dim=0),
            "joint_vel": torch.stack([state["joint_vel"] for state in default_states], dim=0),
            "body_pos": torch.stack([state["body_pos"] for state in default_states], dim=0),
            "body_quat": torch.stack([state["body_quat"] for state in default_states], dim=0),
            "body_lin_vel": torch.stack([state["body_lin_vel"] for state in default_states], dim=0),
            "body_ang_vel": torch.stack([state["body_ang_vel"] for state in default_states], dim=0),
            "object_pos": torch.stack([state["object_pos"] for state in default_states], dim=0),
            "object_quat": torch.stack([state["object_quat"] for state in default_states], dim=0),
            "object_lin_vel": torch.stack([state["object_lin_vel"] for state in default_states], dim=0),
        }
        self._runtime_default_pose_prepend_enabled = True
        self._runtime_default_pose_prepend_steps = num_steps
        logger.info(
            "Using runtime default-pose prepend for multi-clip motion bank ({} clips, {} frames, {}s).",
            self.motion.num_clips,
            num_steps,
            duration,
        )

    def get_motion_transition_contract(self) -> dict[str, Any]:
        """Return the exact transition sequence that shaped this training environment."""

        global_multi_clip = self._uses_global_multi_clip_transition_semantics()
        if global_multi_clip:
            prepend_steps = (
                int(self._runtime_default_pose_prepend_steps)
                if bool(self._runtime_default_pose_prepend_enabled)
                else 0
            )
            prepend_implementation = "runtime_hold" if prepend_steps > 0 else "none"
            append_steps = 0
            append_implementation = "none"
            source_semantics = "global_multi_clip_runtime"
        else:
            prepend_steps = int(getattr(self, "_static_default_pose_prepend_steps", 0) or 0)
            append_steps = int(getattr(self, "_static_default_pose_append_steps", 0) or 0)
            prepend_implementation = "static_splice" if prepend_steps > 0 else "none"
            append_implementation = "static_splice" if append_steps > 0 else "none"
            source_semantics = "single_clip_static"

        contract = {
            "version": MOTION_TRANSITION_CONTRACT_VERSION,
            "control_dt_s": float(self._env.dt),
            "source_semantics": source_semantics,
            "prepend": {
                "implementation": prepend_implementation,
                "applied": prepend_steps > 0,
                "steps": prepend_steps,
            },
            "append": {
                "implementation": append_implementation,
                "applied": append_steps > 0,
                "steps": append_steps,
            },
        }
        return canonical_motion_transition_contract(contract)

    def _build_default_pose_state_robot_order(self, motion_idx: int) -> dict[str, torch.Tensor]:
        """Build the robot default standing pose anchored to a specific motion frame."""
        init_state = self._env.robot_config.init_state
        joint_pos = self._env.default_dof_pos_base.squeeze(0).to(self.device)
        joint_vel = torch.zeros_like(joint_pos)

        init_root_quat = torch.tensor(init_state.rot, dtype=torch.float32, device=self.device).unsqueeze(0)
        init_roll, init_pitch, _ = get_euler_xyz(init_root_quat, w_last=True)

        motion_root_pos = self.motion.body_pos_w[motion_idx, 0].to(self.device)
        motion_root_quat = self.motion.body_quat_w[motion_idx, 0].to(self.device).unsqueeze(0)
        _, _, motion_yaw = get_euler_xyz(motion_root_quat, w_last=True)

        default_root_pos = torch.tensor(
            [motion_root_pos[0], motion_root_pos[1], init_state.pos[2]],
            dtype=torch.float32,
            device=self.device,
        )
        default_root_quat = quat_from_euler_xyz(
            init_roll.squeeze(0),
            init_pitch.squeeze(0),
            motion_yaw.squeeze(0),
        )
        default_root_lin_vel = torch.tensor(init_state.lin_vel, dtype=torch.float32, device=self.device)
        default_root_ang_vel = torch.tensor(init_state.ang_vel, dtype=torch.float32, device=self.device)

        body_states = self._capture_body_states(
            joint_pos,
            joint_vel,
            default_root_pos,
            default_root_quat,
            default_root_lin_vel,
            default_root_ang_vel,
        )

        if self.motion.has_object:
            object_pos = self.motion.object_pos_w[motion_idx].to(self.device)
            object_quat = self.motion.object_quat_w[motion_idx].to(self.device)
            object_lin_vel = self.motion.object_lin_vel_w[motion_idx].to(self.device)
            object_size = self.motion.object_size[motion_idx].to(self.device)
        else:
            object_pos = torch.zeros(3, device=self.device, dtype=torch.float32)
            object_quat = torch.zeros(4, device=self.device, dtype=torch.float32)
            object_quat[3] = 1.0
            object_lin_vel = torch.zeros(3, device=self.device, dtype=torch.float32)
            object_size = torch.zeros(3, device=self.device, dtype=torch.float32)

        return {
            "joint_pos": joint_pos.clone(),
            "joint_vel": joint_vel,
            "root_pos": default_root_pos,
            "root_quat": default_root_quat,
            "root_lin_vel": default_root_lin_vel,
            "root_ang_vel": default_root_ang_vel,
            "body_pos": body_states["pos"],
            "body_quat": body_states["quat"],
            "body_lin_vel": body_states["lin_vel"],
            "body_ang_vel": body_states["ang_vel"],
            "object_pos": object_pos,
            "object_quat": object_quat,
            "object_lin_vel": object_lin_vel,
            "object_size": object_size,
        }

    def _build_default_pose_state(self, use_motion_end: bool = False) -> dict[str, torch.Tensor]:
        """Build the state dict representing the robot's default standing pose.

        By default, anchor root pos/yaw to the motion start; when use_motion_end is True, anchor to motion end.
        """
        motion_idx = -1 if use_motion_end else 0
        default_state = self._build_default_pose_state_robot_order(motion_idx)

        return {
            "joint_pos": default_state["joint_pos"].clone(),
            "joint_vel": default_state["joint_vel"],
            "root_pos": default_state["root_pos"],
            "root_quat": default_state["root_quat"],
            "root_lin_vel": default_state["root_lin_vel"],
            "root_ang_vel": default_state["root_ang_vel"],
            "body_pos": self._map_robot_bodies_to_motion_order(default_state["body_pos"]),
            "body_quat": self._map_robot_bodies_to_motion_order(default_state["body_quat"]),
            "body_lin_vel": self._map_robot_bodies_to_motion_order(default_state["body_lin_vel"]),
            "body_ang_vel": self._map_robot_bodies_to_motion_order(default_state["body_ang_vel"]),
            "object_pos": default_state["object_pos"],
            "object_quat": default_state["object_quat"],
            "object_lin_vel": default_state["object_lin_vel"],
            "object_size": default_state["object_size"],
        }

    def _add_transition_to_motion(self, default_state: dict[str, torch.Tensor], num_steps: int, prepend: bool) -> None:
        """Add interpolated frames either before or after the motion data."""
        assert self._body_indexes_in_motion is not None
        assert self._joint_indexes_in_motion is not None

        if num_steps <= 0:
            return

        device = self.device
        dtype = self.motion._joint_pos.dtype

        default_motion_state = self._default_motion_state(default_state, dtype=dtype, device=device)
        motion_state = self._motion_state(0 if prepend else -1, dtype=dtype, device=device)

        start_state = default_motion_state if prepend else motion_state
        target_state = motion_state if prepend else default_motion_state
        drop_first, drop_last = (False, True) if prepend else (True, False)

        self._build_and_apply_transition(
            start_state=start_state,
            target_state=target_state,
            num_steps=num_steps,
            prepend=prepend,
            drop_first=drop_first,
            drop_last=drop_last,
            dtype=dtype,
            device=device,
        )

    def _slerp_quat_sequence(self, start: torch.Tensor, end: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
        """Spherically interpolate quaternions across multiple time steps."""
        if alphas.numel() == 0:
            return start.new_zeros((0,) + start.shape)

        num_steps = alphas.shape[0]
        start_expand = start.unsqueeze(0).expand(num_steps, -1, -1)
        end_expand = end.unsqueeze(0).expand(num_steps, -1, -1)
        alpha_flat = alphas.repeat_interleave(start.shape[0]).unsqueeze(-1)
        blended = slerp(
            start_expand.reshape(-1, 4),
            end_expand.reshape(-1, 4),
            alpha_flat,
        )
        return blended.view(num_steps, start.shape[0], 4)

    def _capture_body_states(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Capture body states by temporarily setting the robot state in the simulator."""
        simulator = self._env.simulator
        assert simulator.get_simulator_type() == SimulatorType.ISAACSIM, (
            "Default-pose interpolation only supports IsaacSim; IsaacGym write_state_updates does not run FK."
        )
        env_id = 0
        env_origin = self._get_env_offsets()[env_id]

        root_backup = simulator.robot_root_states[env_id].clone()
        dof_pos_backup = simulator.dof_pos[env_id].clone()
        dof_vel_backup = simulator.dof_vel[env_id].clone()

        try:
            simulator.robot_root_states[env_id, :3] = root_pos + env_origin
            simulator.robot_root_states[env_id, 3:7] = root_quat
            simulator.robot_root_states[env_id, 7:10] = root_lin_vel
            simulator.robot_root_states[env_id, 10:13] = root_ang_vel
            simulator.dof_pos[env_id] = joint_pos
            simulator.dof_vel[env_id] = joint_vel

            simulator.set_actor_root_state_tensor_robots()
            simulator.set_dof_state_tensor_robots()
            simulator.write_state_updates()
            simulator.refresh_sim_tensors()

            body_pos = (simulator._rigid_body_pos[env_id] - env_origin).clone()
            body_quat = simulator._rigid_body_rot[env_id].clone()
            body_lin_vel = simulator._rigid_body_vel[env_id].clone()
            body_ang_vel = simulator._rigid_body_ang_vel[env_id].clone()
        finally:
            simulator.robot_root_states[env_id] = root_backup
            simulator.dof_pos[env_id] = dof_pos_backup
            simulator.dof_vel[env_id] = dof_vel_backup
            simulator.set_actor_root_state_tensor_robots()
            simulator.set_dof_state_tensor_robots()
            simulator.write_state_updates()
            simulator.refresh_sim_tensors()

        return {
            "pos": body_pos,
            "quat": body_quat,
            "lin_vel": body_lin_vel,
            "ang_vel": body_ang_vel,
        }

    def _map_robot_bodies_to_motion_order(self, robot_tensor: torch.Tensor) -> torch.Tensor:
        """Map robot body tensor to motion data order using body indexes."""
        assert self._body_indexes_in_motion is not None
        num_motion_bodies = self.motion._body_pos_w.shape[1]
        motion_shape = (num_motion_bodies,) + robot_tensor.shape[1:]
        motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        motion_tensor[self._body_indexes_in_motion] = robot_tensor
        return motion_tensor

    def _map_robot_joints_to_motion_order(
        self, robot_tensor: torch.Tensor, num_motion_joints: int | None = None
    ) -> torch.Tensor:
        """Map robot joint tensor to motion data order using joint indexes."""
        assert self._joint_indexes_in_motion is not None
        if num_motion_joints is None:
            num_motion_joints = self.motion._joint_pos.shape[1]
        motion_shape = robot_tensor.shape[:-1] + (num_motion_joints,)
        motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        motion_tensor[..., self._joint_indexes_in_motion] = robot_tensor
        return motion_tensor


    def _motion_state(self, idx: int, dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
        """Slice motion tensors at a given index into a state dict."""
        state = {
            "joint_pos": self.motion._joint_pos[idx].to(device=device, dtype=dtype),
            "joint_vel": self.motion._joint_vel[idx].to(device=device, dtype=dtype),
            "body_pos": self.motion._body_pos_w[idx].to(device=device, dtype=dtype),
            "body_quat": self.motion._body_quat_w[idx].to(device=device, dtype=dtype),
            "body_lin_vel": self.motion._body_lin_vel_w[idx].to(device=device, dtype=dtype),
            "body_ang_vel": self.motion._body_ang_vel_w[idx].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = self.motion._object_pos_w[idx].to(device=device, dtype=dtype)
            state["object_quat"] = self.motion._object_quat_w[idx].to(device=device, dtype=dtype)
            state["object_lin_vel"] = self.motion._object_lin_vel_w[idx].to(device=device, dtype=dtype)
            state["object_size"] = self.motion._object_size[idx].to(device=device, dtype=dtype)
        return state

    def _default_motion_state(
        self, default_state: dict[str, torch.Tensor], dtype: torch.dtype, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Map default robot-state tensors into motion order for interpolation."""
        state = {
            "joint_pos": self._map_robot_joints_to_motion_order(
                default_state["joint_pos"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_pos.shape[1],
            ),
            "joint_vel": self._map_robot_joints_to_motion_order(
                default_state["joint_vel"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_vel.shape[1],
            ),
            "body_pos": default_state["body_pos"].to(device=device, dtype=dtype),
            "body_quat": default_state["body_quat"].to(device=device, dtype=dtype),
            "body_lin_vel": default_state["body_lin_vel"].to(device=device, dtype=dtype),
            "body_ang_vel": default_state["body_ang_vel"].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = default_state["object_pos"].to(device=device, dtype=dtype)
            state["object_quat"] = default_state["object_quat"].to(device=device, dtype=dtype)
            state["object_lin_vel"] = default_state["object_lin_vel"].to(device=device, dtype=dtype)
            state["object_size"] = default_state["object_size"].to(device=device, dtype=dtype)
        return state

    def _build_transition_segments(
        self,
        start: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        alphas: torch.Tensor,
        alphas_joint: torch.Tensor,
        alphas_body: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Linearly/spherically interpolate between start and target states."""

        def _lerp(a: torch.Tensor, b: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
            return a.unsqueeze(0) + view * (b - a).unsqueeze(0)

        segments = {
            "joint_pos": _lerp(start["joint_pos"], target["joint_pos"], alphas_joint),
            "joint_vel": _lerp(start["joint_vel"], target["joint_vel"], alphas_joint),
            "body_pos": _lerp(start["body_pos"], target["body_pos"], alphas_body),
            "body_lin_vel": _lerp(start["body_lin_vel"], target["body_lin_vel"], alphas_body),
            "body_ang_vel": _lerp(start["body_ang_vel"], target["body_ang_vel"], alphas_body),
            "body_quat": self._slerp_quat_sequence(start["body_quat"], target["body_quat"], alphas),
        }

        if self.motion.has_object:
            segments["object_pos"] = _lerp(start["object_pos"], target["object_pos"], alphas_joint)
            segments["object_lin_vel"] = _lerp(start["object_lin_vel"], target["object_lin_vel"], alphas_joint)
            segments["object_quat"] = self._slerp_quat_sequence(
                start["object_quat"].unsqueeze(0), target["object_quat"].unsqueeze(0), alphas
            ).squeeze(1)
            segments["object_size"] = _lerp(start["object_size"], target["object_size"], alphas_joint)

        return segments

    def _apply_transition_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> None:
        """Splice interpolated segments into motion data, either prepending or appending."""
        self.motion = self.motion.extend_with_segments(segments, prepend=prepend)

    def _build_and_apply_transition(
        self,
        start_state: dict[str, torch.Tensor],
        target_state: dict[str, torch.Tensor],
        num_steps: int,
        prepend: bool,
        drop_first: bool,
        drop_last: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Shared interpolation path for prepend/append transitions."""
        if num_steps <= 0:
            return

        alphas = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device, dtype=dtype)
        if drop_first:
            alphas = alphas[1:]
        if drop_last:
            alphas = alphas[:-1]
        if alphas.numel() == 0:
            return

        alphas_joint = alphas.view(num_steps, 1)
        alphas_body = alphas.view(num_steps, 1, 1)

        segments = self._build_transition_segments(start_state, target_state, alphas, alphas_joint, alphas_body)
        self._apply_transition_segments(segments, prepend=prepend)

    def _setup_visualization_markers_for_isaacsim(self):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG, RAY_CASTER_MARKER_CFG

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/real_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        real_robot_visualizer = VisualizationMarkers(visualization_markers_cfg)

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/motion_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        motion_robot_visualizer = VisualizationMarkers(visualization_markers_cfg)
        self.visualization_markers = {
            "real_robot": real_robot_visualizer,
            "motion_robot": motion_robot_visualizer,
        }

        for body_names in self.motion_cfg.body_names_to_track:
            visualization_markers_cfg = RAY_CASTER_MARKER_CFG.replace(
                prim_path=f"/Visuals/Command/motion_robot_body/motion_{body_names}",
            )
            visualization_markers_cfg.markers["hit"].radius = 0.03
            visualization_markers_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
            self.visualization_markers[f"motion_{body_names}"] = VisualizationMarkers(visualization_markers_cfg)

        if self.motion.has_object:
            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/real_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            real_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/motion_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            motion_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            self.visualization_markers["real_object"] = real_object_visualizer
            self.visualization_markers["motion_object"] = motion_object_visualizer

    def _ensure_index_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
