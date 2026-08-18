from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import numbers
import os
import random
import stat
import tempfile
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from rich.console import Console
from torch import nn
from torch.nn.parameter import UninitializedParameter
from torch.distributions import Normal, kl_divergence
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.agents.modules.augmentation_utils import SymmetryUtils
from holosoma.agents.modules.data_utils import RolloutStorage
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.agents.modules.modules import DeFMEncoder
from holosoma.agents.modules.module_utils import (
    setup_ppo_actor_module,
    setup_ppo_critic_module,
)
from holosoma.config_types.algo import LayerConfig, ModuleConfig, PPOConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.managers.perception.manager import PerceptionManager
from holosoma.utils.checkpoint_validation import (
    FIXED_BC_EVAL_ALLOCATION_SCHEME,
    FIXED_BC_EVAL_ALLOCATION_VERSION,
    TERMINAL_FIXED_BC_EVAL_STATE_KEY,
    TERMINAL_FIXED_BC_EVAL_STATE_SHA256_KEY,
    TERMINAL_FIXED_BC_EVAL_STATE_VERSION,
    checkpoint_saved_run_target,
    fixed_bc_dataset_sha256,
    fixed_bc_global_dataset_sha256,
    load_verified_torch_checkpoint,
    require_mapping,
    terminal_fixed_bc_eval_state_sha256,
    validate_checkpoint_iterations,
    validate_checkpoint_terminal_fixed_bc_eval,
    validate_finite_tree,
    validate_module_state_compatibility,
    validate_optimizer_compatibility,
    validate_optimizer_state,
    validate_terminal_fixed_bc_eval_artifact_payload,
    validate_terminal_fixed_bc_eval_state,
)
from holosoma.utils.helpers import instantiate
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    export_policy_as_onnx,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_urdf_text_from_robot_config,
    validate_exported_policy_onnx,
)
from holosoma.utils.normalization import EmpiricalNormalization
from holosoma.utils.policy_init_preflight import (
    ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV,
    allow_legacy_unverified_policy_load,
    required_policy_init_terminal_target_from_env,
    validate_policy_init_payload_identity,
    validate_policy_init_terminal_source_payload,
    validate_stage4_init_payload_identity,
)
from holosoma.utils.rng_checkpoint import (
    ALLOW_NONDETERMINISTIC_RNG_RESUME_ENV,
    capture_rng_checkpoint_state,
    restore_rng_checkpoint_state,
    validate_rng_checkpoint_state,
)
from holosoma.utils.resume_preflight import (
    ALLOW_LEGACY_UNPROVENANCED_RESUME_ENV,
    allow_legacy_unprovenanced_resume,
    validate_resume_payload_identity,
)
from holosoma.utils.step_timing import StepTiming, compact_timing_summary, env_int
from holosoma.utils.training_provenance import (
    ALLOW_LEGACY_UNVERIFIED_TEACHER_LOAD_ENV,
    allow_legacy_unverified_teacher_load,
    training_provenance_from_env,
    validate_training_provenance,
)

console = Console()
_UNSET = object()
_FIXED_BC_EVAL_ALLOCATION_VERSION = FIXED_BC_EVAL_ALLOCATION_VERSION
_FIXED_BC_EVAL_ALLOCATION_SCHEME = FIXED_BC_EVAL_ALLOCATION_SCHEME
_FIXED_BC_GUARD_STATE_VERSION = 1
_TERMINAL_FIXED_BC_EVAL_STATE_VERSION = TERMINAL_FIXED_BC_EVAL_STATE_VERSION
_TERMINAL_FIXED_BC_EVAL_STATE_KEY = TERMINAL_FIXED_BC_EVAL_STATE_KEY
_TERMINAL_FIXED_BC_EVAL_STATE_SHA256_KEY = TERMINAL_FIXED_BC_EVAL_STATE_SHA256_KEY
_EXACT_ROLLOUT_RESUME_CONTRACT_VERSION = 2
_EXACT_ROLLOUT_RESUME_CONTRACT_MODE = "canonical_reset_after_checkpoint"
_RECOVERY_ROLLOUT_RESUME_CONTRACT_VERSION = 3
_RECOVERY_ROLLOUT_RESUME_CONTRACT_MODE = "new_episode_on_resume"
_ALLOW_LEGACY_ROLLOUT_RESUME_ENV = "HOLOSOMA_ALLOW_LEGACY_ROLLOUT_RESTART_RESUME"
_EXPECTED_EVALUATION_CHECKPOINT_SHA256_ENV = (
    "HOLOSOMA_EXPECTED_EVALUATION_CHECKPOINT_SHA256"
)
_ALLOW_AUTHENTICATED_LEGACY_EVALUATION_MOTION_CONTRACT_ENV = (
    "HOLOSOMA_EVAL_ALLOW_AUTHENTICATED_LEGACY_MOTION_CONTRACT"
)
_ACTOR_PERCEPTION_GEOMETRY_SUPPORT_KEY = "actor_perception_training_geometry_support"
_DAGGER_REPLAY_STATE_KEY = "dagger_replay_by_rank"
_DAGGER_REPLAY_STATE_VERSION = 1
_PRECOMPUTED_COMMAND_CONTRACT_KEY = (
    "precomputed_turn_then_forward_deployment_contract"
)
_PRECOMPUTED_COMMAND_CONTRACT_SHA256_KEY = (
    "precomputed_turn_then_forward_deployment_contract_sha256"
)
_ROLLING_REFERENCE_DELTA_CONTRACT_KEY = (
    "rolling_reference_delta_deployment_contract"
)
_ROLLING_REFERENCE_DELTA_CONTRACT_SHA256_KEY = (
    "rolling_reference_delta_deployment_contract_sha256"
)


def _precomputed_turn_then_forward_deployment_contract(
    *,
    zero_root_command_when_drop_active: bool = False,
) -> tuple[dict[str, Any], str]:
    if not isinstance(zero_root_command_when_drop_active, bool):
        raise ValueError(
            "zero_root_command_when_drop_active must be a boolean, "
            f"got {zero_root_command_when_drop_active!r}."
        )
    contract: dict[str, Any] = {
        "version": 1,
        "mode": "precomputed_turn_then_forward",
        "adapter": (
            "holosoma_inference.policies.wbt.WholeBodyTrackingPolicy"
        ),
        "command_field": "policy_command_xy_yaw",
        "phase_field": "policy_command_phase",
        "command_layout": ["dx_m", "dy_m", "dyaw_rad"],
        "pre_pickup_command": [0.0, 0.0, 0.0],
        "pickup_latch": {
            "algorithm": "object_root_rel_z_v1",
            "lift_height_threshold_m": 0.1,
            "lift_ratio_threshold": 0.35,
            "consecutive_physics_steps": 5,
            "sticky_until_episode_reset": True,
            "requires_live_sim_object_state": True,
        },
        "phase_contract": {
            "dy_always_zero": True,
            "dx_dyaw_overlap": False,
            "zero_phase": 0,
            "forward_phase": 1,
            "yaw_phase": 2,
        },
        "transition_padding": "zero_command_zero_phase",
        "external_override": "disabled_by_default_explicit_diagnostic_only",
    }
    if zero_root_command_when_drop_active:
        contract["version"] = 2
        contract["drop_exclusivity"] = {
            "drop_field": "drop_button",
            "active_threshold": 0.5,
            "active_root_command": [0.0, 0.0, 0.0],
            "applied_after_external_overrides": True,
        }
    payload = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return contract, hashlib.sha256(payload).hexdigest()


def _rolling_reference_delta_deployment_contract(
    *,
    lookahead_motion_frames: int,
    zero_yaw_threshold_deg: float,
    zero_root_command_when_drop_active: bool = False,
) -> tuple[dict[str, Any], str]:
    if (
        isinstance(lookahead_motion_frames, (bool, np.bool_))
        or not isinstance(lookahead_motion_frames, numbers.Integral)
        or int(lookahead_motion_frames) < 1
    ):
        raise ValueError(
            "lookahead_motion_frames must be a positive integer, "
            f"got {lookahead_motion_frames!r}."
        )
    if (
        isinstance(zero_yaw_threshold_deg, (bool, np.bool_))
        or not isinstance(zero_yaw_threshold_deg, numbers.Real)
        or not math.isfinite(float(zero_yaw_threshold_deg))
        or not 0.0 <= float(zero_yaw_threshold_deg) <= 180.0
    ):
        raise ValueError(
            "zero_yaw_threshold_deg must be a finite real in [0, 180], "
            f"got {zero_yaw_threshold_deg!r}."
        )
    if not isinstance(zero_root_command_when_drop_active, bool):
        raise ValueError(
            "zero_root_command_when_drop_active must be a boolean, "
            f"got {zero_root_command_when_drop_active!r}."
        )
    contract: dict[str, Any] = {
        "version": 1,
        "mode": "rolling_reference_delta",
        "adapter": "holosoma_inference.policies.wbt.WholeBodyTrackingPolicy",
        "command_layout": ["dx_m", "dy_m", "dyaw_rad"],
        "lookahead_motion_frames": int(lookahead_motion_frames),
        "zero_yaw_threshold_deg": float(zero_yaw_threshold_deg),
        "update_rule": "recompute_every_policy_step",
        "delta_source": "reference_frame_t_to_reference_frame_t_plus_lookahead",
        "coordinate_frame": "reference_root_heading_at_frame_t",
        "robot_state_feedback_used": False,
        "active_window": "carry_t1_inclusive_to_t2_exclusive",
        "invalid_endpoint": "zero_if_t_plus_lookahead_reaches_t2_or_clip_end",
        "external_override": "disabled_by_default_explicit_diagnostic_only",
    }
    if zero_root_command_when_drop_active:
        contract["version"] = 2
        contract["drop_exclusivity"] = {
            "drop_field": "drop_button",
            "active_threshold": 0.5,
            "active_root_command": [0.0, 0.0, 0.0],
            "applied_after_external_overrides": True,
        }
    payload = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return contract, hashlib.sha256(payload).hexdigest()


class _DaggerReplayBuffer:
    """Bounded, rank-local reservoir for deterministic DAgger rehearsal.

    The buffer owns an independent CPU generator.  Both reservoir replacement
    and replay sampling consume only that generator, so simulator, policy, and
    fixed-evaluation RNG streams remain untouched.  Its authenticated state is
    sufficient to continue the exact insertion/sample sequence after a full
    resume.
    """

    _FLOAT_DTYPE = torch.float32
    _REQUIRED_STATE_KEYS = {
        "version",
        "capacity",
        "actor_obs_dim",
        "actor_perception_dim",
        "action_dim",
        "base_seed",
        "effective_seed",
        "rank",
        "size",
        "seen_valid_count",
        "sample_draw_count",
        "capture_start_iteration",
        "fixed_bc_global_dataset_digest",
        "actor_obs_raw",
        "actor_perception",
        "teacher_actions",
        "mask",
        "generator_state",
        "sha256",
    }

    def __init__(
        self,
        *,
        capacity: int,
        actor_obs_dim: int,
        actor_perception_dim: int,
        action_dim: int,
        base_seed: int,
        rank: int,
    ) -> None:
        integer_fields = {
            "capacity": capacity,
            "actor_obs_dim": actor_obs_dim,
            "actor_perception_dim": actor_perception_dim,
            "action_dim": action_dim,
            "base_seed": base_seed,
            "rank": rank,
        }
        for name, value in integer_fields.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, numbers.Integral
            ):
                raise ValueError(f"DAgger replay {name} must be an integer, got {value!r}.")
        if int(capacity) <= 0:
            raise ValueError(f"DAgger replay capacity must be positive, got {capacity}.")
        if int(actor_obs_dim) <= 0 or int(action_dim) <= 0:
            raise ValueError(
                "DAgger replay actor/action widths must be positive, got "
                f"actor_obs_dim={actor_obs_dim}, action_dim={action_dim}."
            )
        if int(actor_perception_dim) < 0 or int(base_seed) < 0 or int(rank) < 0:
            raise ValueError(
                "DAgger replay perception width, seed, and rank must be non-negative; "
                f"got perception={actor_perception_dim}, seed={base_seed}, rank={rank}."
            )

        self.capacity = int(capacity)
        self.actor_obs_dim = int(actor_obs_dim)
        self.actor_perception_dim = int(actor_perception_dim)
        self.action_dim = int(action_dim)
        self.base_seed = int(base_seed)
        self.rank = int(rank)
        seed_material = f"holosoma-dagger-replay-v1:{self.base_seed}:{self.rank}".encode(
            "utf-8"
        )
        self.effective_seed = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8], "big"
        ) & ((1 << 63) - 1)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.effective_seed)

        self.actor_obs_raw = torch.empty(
            (self.capacity, self.actor_obs_dim), dtype=self._FLOAT_DTYPE
        )
        self.actor_perception = torch.empty(
            (self.capacity, self.actor_perception_dim), dtype=self._FLOAT_DTYPE
        )
        self.teacher_actions = torch.empty(
            (self.capacity, self.action_dim), dtype=self._FLOAT_DTYPE
        )
        self.mask = torch.empty((self.capacity, 1), dtype=torch.bool)
        self.size = 0
        self.seen_valid_count = 0
        self.sample_draw_count = 0
        self.capture_start_iteration: int | None = None
        self.fixed_bc_global_dataset_digest: str | None = None

    @staticmethod
    def _validate_sha256(value: Any, *, path: str, allow_none: bool) -> str | None:
        if value is None and allow_none:
            return None
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"{path} must be a lowercase SHA-256 digest.")
        return value

    @staticmethod
    def _strict_nonnegative_int(value: Any, *, path: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, numbers.Integral
        ):
            raise ValueError(f"{path} must be a non-negative integer, got {value!r}.")
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"{path} must be non-negative, got {parsed}.")
        return parsed

    @classmethod
    def _validate_dense_tensor(
        cls,
        value: Any,
        *,
        path: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        finite: bool,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{path} must be a tensor.")
        if value.layout != torch.strided or tuple(value.shape) != shape:
            raise ValueError(
                f"{path} must be a dense tensor with shape {shape}, got "
                f"layout={value.layout}, shape={tuple(value.shape)}."
            )
        if value.dtype != dtype:
            raise ValueError(f"{path} dtype must be {dtype}, got {value.dtype}.")
        if finite and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{path} contains NaN or infinity.")
        return value.detach().cpu().contiguous().clone()

    def bind_fixed_dataset(self, *, iteration: int, global_digest: str) -> None:
        iteration = self._strict_nonnegative_int(
            iteration, path="DAgger replay capture_start_iteration"
        )
        global_digest = self._validate_sha256(
            global_digest,
            path="DAgger replay fixed_bc_global_dataset_digest",
            allow_none=False,
        )
        assert global_digest is not None
        if self.capture_start_iteration is None:
            if self.size != 0 or self.seen_valid_count != 0:
                raise RuntimeError(
                    "Cannot bind a fixed-BC boundary after DAgger replay collection started."
                )
            self.capture_start_iteration = iteration
            self.fixed_bc_global_dataset_digest = global_digest
            return
        if self.fixed_bc_global_dataset_digest != global_digest:
            raise RuntimeError(
                "DAgger replay fixed-BC boundary digest changed after collection began."
            )

    def insert(
        self,
        *,
        actor_obs_raw: torch.Tensor,
        actor_perception: torch.Tensor | None,
        teacher_actions: torch.Tensor,
        mask: torch.Tensor,
        finite_prevalidated: bool = False,
    ) -> int:
        if self.capture_start_iteration is None or self.fixed_bc_global_dataset_digest is None:
            raise RuntimeError(
                "DAgger replay collection requires an authenticated complete fixed-BC boundary."
            )
        if not isinstance(actor_obs_raw, torch.Tensor) or actor_obs_raw.ndim != 2:
            raise ValueError("DAgger replay actor_obs_raw must be a rank-2 tensor.")
        rows = int(actor_obs_raw.shape[0])
        input_tensors: dict[str, torch.Tensor] = {
            "actor_obs_raw": actor_obs_raw,
            "teacher_actions": teacher_actions,
            "mask": mask,
        }
        if self.actor_perception_dim == 0:
            if actor_perception is not None and (
                not isinstance(actor_perception, torch.Tensor)
                or tuple(actor_perception.shape) != (rows, 0)
            ):
                raise ValueError(
                    "DAgger replay received actor perception for a policy without perception."
                )
            actor_perception = torch.empty(
                (rows, 0), device=actor_obs_raw.device, dtype=self._FLOAT_DTYPE
            )
        else:
            if actor_perception is None:
                raise ValueError(
                    "DAgger replay requires actor perception for a perception policy."
                )
            input_tensors["actor_perception"] = actor_perception
        assert actor_perception is not None
        input_tensors["actor_perception"] = actor_perception

        expected_shapes = {
            "actor_obs_raw": (rows, self.actor_obs_dim),
            "actor_perception": (rows, self.actor_perception_dim),
            "teacher_actions": (rows, self.action_dim),
            "mask": (rows, 1),
        }
        input_device = actor_obs_raw.device
        finite_checks: list[torch.Tensor] = []
        for key, expected_shape in expected_shapes.items():
            value = input_tensors[key]
            expected_dtype = torch.bool if key == "mask" else self._FLOAT_DTYPE
            if (
                not isinstance(value, torch.Tensor)
                or value.layout != torch.strided
                or tuple(value.shape) != expected_shape
                or value.dtype != expected_dtype
                or value.device != input_device
            ):
                raise ValueError(
                    f"DAgger replay {key} must have shape={expected_shape}, "
                    f"dtype={expected_dtype}, device={input_device}."
                )
            if key != "mask":
                finite_checks.append(torch.isfinite(value).all())
        if type(finite_prevalidated) is not bool:
            raise ValueError("DAgger replay finite_prevalidated must be a Python bool.")
        if (
            not finite_prevalidated
            and finite_checks
            and not bool(torch.stack(finite_checks).all().item())
        ):
            raise ValueError("DAgger replay insertion tensors contain NaN or infinity.")

        valid_indices = torch.nonzero(mask.view(-1), as_tuple=False).view(-1)
        valid_count = int(valid_indices.numel())
        if valid_count == 0:
            return 0
        # A float64 uniform draw has 53 bits of integer resolution.  Formal
        # runs are many orders of magnitude below this bound; fail rather than
        # silently bias reservoir replacement if that contract is exceeded.
        final_seen_count = self.seen_valid_count + valid_count
        if final_seen_count > 2**53:
            raise OverflowError(
                "DAgger replay reservoir exceeded the exact float64 sampling range."
            )

        fill_count = min(self.capacity - self.size, valid_count)
        final_source_by_destination: dict[int, int] = {
            self.size + offset: offset for offset in range(fill_count)
        }
        remaining_count = valid_count - fill_count
        if remaining_count > 0:
            # Batched Algorithm R.  Each row i draws j_i uniformly from
            # [0, seen_before+i].  When multiple rows target one slot, only the
            # chronologically last replacement survives, exactly as in the
            # sequential reservoir update, while host work remains bounded by
            # the small number of successful replacements.
            seen_after_each_row = torch.arange(
                self.seen_valid_count + fill_count + 1,
                final_seen_count + 1,
                dtype=torch.float64,
            )
            uniform = torch.rand(
                (remaining_count,),
                generator=self.generator,
                dtype=torch.float64,
            )
            destinations = torch.floor(uniform * seen_after_each_row).to(torch.int64)
            replacement_positions = torch.nonzero(
                destinations < self.capacity,
                as_tuple=False,
            ).view(-1)
            for position in replacement_positions.tolist():
                destination = int(destinations[position].item())
                final_source_by_destination[destination] = fill_count + position

        self.size = min(self.capacity, self.size + valid_count)
        self.seen_valid_count = final_seen_count
        if not final_source_by_destination:
            return 0
        destination_indices = torch.tensor(
            sorted(final_source_by_destination), dtype=torch.int64
        )
        valid_positions = torch.tensor(
            [final_source_by_destination[int(index)] for index in destination_indices],
            device=input_device,
            dtype=torch.int64,
        )
        selected_raw_indices = valid_indices.index_select(0, valid_positions)
        actor_obs_cpu = actor_obs_raw.index_select(
            0, selected_raw_indices
        ).detach().cpu()
        actor_perception_cpu = actor_perception.index_select(
            0, selected_raw_indices
        ).detach().cpu()
        teacher_actions_cpu = teacher_actions.index_select(
            0, selected_raw_indices
        ).detach().cpu()
        mask_cpu = mask.index_select(0, selected_raw_indices).detach().cpu()
        self.actor_obs_raw.index_copy_(0, destination_indices, actor_obs_cpu)
        self.actor_perception.index_copy_(
            0, destination_indices, actor_perception_cpu
        )
        self.teacher_actions.index_copy_(
            0, destination_indices, teacher_actions_cpu
        )
        self.mask.index_copy_(0, destination_indices, mask_cpu)
        return len(final_source_by_destination)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        batch_size = self._strict_nonnegative_int(
            batch_size, path="DAgger replay batch_size"
        )
        if batch_size <= 0:
            raise ValueError("DAgger replay batch_size must be positive.")
        if self.size == 0:
            return {
                "actor_obs_raw": self.actor_obs_raw[:0].clone(),
                "actor_perception": self.actor_perception[:0].clone(),
                "teacher_actions": self.teacher_actions[:0].clone(),
                "mask": self.mask[:0].clone(),
            }
        indices = torch.randint(
            0,
            self.size,
            (batch_size,),
            generator=self.generator,
            dtype=torch.int64,
        )
        self.sample_draw_count += batch_size
        return {
            "actor_obs_raw": self.actor_obs_raw[indices].clone(),
            "actor_perception": self.actor_perception[indices].clone(),
            "teacher_actions": self.teacher_actions[indices].clone(),
            "mask": self.mask[indices].clone(),
        }

    @classmethod
    def _state_sha256(cls, state_without_digest: Mapping[str, Any]) -> str:
        tensor_keys = (
            "actor_obs_raw",
            "actor_perception",
            "teacher_actions",
            "mask",
            "generator_state",
        )
        metadata = {
            key: value
            for key, value in state_without_digest.items()
            if key not in tensor_keys
        }
        hasher = hashlib.sha256()
        hasher.update(
            json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        for key in tensor_keys:
            value = state_without_digest[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"DAgger replay digest field {key!r} must be a tensor.")
            tensor = value.detach().cpu().contiguous()
            hasher.update(key.encode("utf-8"))
            hasher.update(str(tensor.dtype).encode("ascii"))
            hasher.update(json.dumps(list(tensor.shape)).encode("ascii"))
            hasher.update(tensor.numpy().tobytes(order="C"))
        return hasher.hexdigest()

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "version": _DAGGER_REPLAY_STATE_VERSION,
            "capacity": self.capacity,
            "actor_obs_dim": self.actor_obs_dim,
            "actor_perception_dim": self.actor_perception_dim,
            "action_dim": self.action_dim,
            "base_seed": self.base_seed,
            "effective_seed": self.effective_seed,
            "rank": self.rank,
            "size": self.size,
            "seen_valid_count": self.seen_valid_count,
            "sample_draw_count": self.sample_draw_count,
            "capture_start_iteration": self.capture_start_iteration,
            "fixed_bc_global_dataset_digest": self.fixed_bc_global_dataset_digest,
            "actor_obs_raw": self.actor_obs_raw[: self.size].clone(),
            "actor_perception": self.actor_perception[: self.size].clone(),
            "teacher_actions": self.teacher_actions[: self.size].clone(),
            "mask": self.mask[: self.size].clone(),
            "generator_state": self.generator.get_state().detach().cpu().clone(),
        }
        state["sha256"] = self._state_sha256(state)
        return state

    def load_state_dict(self, state: Any) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("DAgger replay checkpoint state must be a mapping.")
        if set(state) != self._REQUIRED_STATE_KEYS:
            raise ValueError(
                "DAgger replay checkpoint schema mismatch: "
                f"missing={sorted(self._REQUIRED_STATE_KEYS - set(state))}, "
                f"extra={sorted(set(state) - self._REQUIRED_STATE_KEYS)}."
            )
        expected_metadata = {
            "version": _DAGGER_REPLAY_STATE_VERSION,
            "capacity": self.capacity,
            "actor_obs_dim": self.actor_obs_dim,
            "actor_perception_dim": self.actor_perception_dim,
            "action_dim": self.action_dim,
            "base_seed": self.base_seed,
            "effective_seed": self.effective_seed,
            "rank": self.rank,
        }
        mismatches = [
            f"{key}={state.get(key)!r} != {expected!r}"
            for key, expected in expected_metadata.items()
            if type(state.get(key)) is not type(expected) or state.get(key) != expected
        ]
        if mismatches:
            raise ValueError(
                "DAgger replay checkpoint metadata disagrees with runtime: "
                + "; ".join(mismatches)
            )
        size = self._strict_nonnegative_int(state["size"], path="DAgger replay size")
        seen_valid_count = self._strict_nonnegative_int(
            state["seen_valid_count"], path="DAgger replay seen_valid_count"
        )
        sample_draw_count = self._strict_nonnegative_int(
            state["sample_draw_count"], path="DAgger replay sample_draw_count"
        )
        if size > self.capacity or seen_valid_count < size:
            raise ValueError(
                "DAgger replay checkpoint counters are inconsistent: "
                f"size={size}, capacity={self.capacity}, seen={seen_valid_count}."
            )
        capture_start_iteration = state["capture_start_iteration"]
        if capture_start_iteration is not None:
            capture_start_iteration = self._strict_nonnegative_int(
                capture_start_iteration,
                path="DAgger replay capture_start_iteration",
            )
        fixed_digest = self._validate_sha256(
            state["fixed_bc_global_dataset_digest"],
            path="DAgger replay fixed_bc_global_dataset_digest",
            allow_none=True,
        )
        if (capture_start_iteration is None) != (fixed_digest is None):
            raise ValueError(
                "DAgger replay capture iteration and fixed-BC digest must be present together."
            )
        if size > 0 and capture_start_iteration is None:
            raise ValueError(
                "Non-empty DAgger replay checkpoint has no authenticated capture boundary."
            )

        tensors = {
            "actor_obs_raw": self._validate_dense_tensor(
                state["actor_obs_raw"],
                path="DAgger replay actor_obs_raw",
                shape=(size, self.actor_obs_dim),
                dtype=self._FLOAT_DTYPE,
                finite=True,
            ),
            "actor_perception": self._validate_dense_tensor(
                state["actor_perception"],
                path="DAgger replay actor_perception",
                shape=(size, self.actor_perception_dim),
                dtype=self._FLOAT_DTYPE,
                finite=True,
            ),
            "teacher_actions": self._validate_dense_tensor(
                state["teacher_actions"],
                path="DAgger replay teacher_actions",
                shape=(size, self.action_dim),
                dtype=self._FLOAT_DTYPE,
                finite=True,
            ),
            "mask": self._validate_dense_tensor(
                state["mask"],
                path="DAgger replay mask",
                shape=(size, 1),
                dtype=torch.bool,
                finite=False,
            ),
        }
        if size > 0 and not bool(tensors["mask"].all().item()):
            raise ValueError(
                "DAgger replay stores only valid teacher-labelled rows; checkpoint mask contains false."
            )
        raw_generator_state = state["generator_state"]
        if not isinstance(raw_generator_state, torch.Tensor) or raw_generator_state.ndim != 1:
            raise ValueError("DAgger replay generator_state must be a rank-1 tensor.")
        generator_state = self._validate_dense_tensor(
            raw_generator_state,
            path="DAgger replay generator_state",
            shape=(int(raw_generator_state.numel()),),
            dtype=torch.uint8,
            finite=False,
        )
        declared_digest = self._validate_sha256(
            state["sha256"], path="DAgger replay sha256", allow_none=False
        )
        state_without_digest = {
            key: value for key, value in state.items() if key != "sha256"
        }
        actual_digest = self._state_sha256(state_without_digest)
        if declared_digest != actual_digest:
            raise ValueError(
                "DAgger replay checkpoint SHA-256 mismatch: "
                f"declared={declared_digest}, actual={actual_digest}."
            )
        generator = torch.Generator(device="cpu")
        try:
            generator.set_state(generator_state)
        except Exception as exc:
            raise ValueError("DAgger replay generator_state is invalid.") from exc

        self.actor_obs_raw[:size].copy_(tensors["actor_obs_raw"])
        self.actor_perception[:size].copy_(tensors["actor_perception"])
        self.teacher_actions[:size].copy_(tensors["teacher_actions"])
        self.mask[:size].copy_(tensors["mask"])
        self.size = size
        self.seen_valid_count = seen_valid_count
        self.sample_draw_count = sample_draw_count
        self.capture_start_iteration = capture_start_iteration
        self.fixed_bc_global_dataset_digest = fixed_digest
        self.generator = generator


class Minibatch(TypedDict):
    """A minibatch of data for training a PPO agent."""

    actor_obs: torch.Tensor
    """The observation of the actor.

    Shape: (mini_batch_size, actor_obs_dim), dtype: torch.float32
    """

    critic_obs: torch.Tensor
    """The observation of the critic.

    Shape: (mini_batch_size, critic_obs_dim), dtype: torch.float32
    """

    actions: torch.Tensor
    """The actions taken by the agent.

    Shape: (mini_batch_size, num_act), dtype: torch.float32
    """

    rewards: torch.Tensor
    """The rewards received from the environment.

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    dones: torch.Tensor
    """Whether each episode is done after taking the action.

    Shape: (mini_batch_size, 1), dtype: torch.bool
    """

    values: torch.Tensor
    """The value estimates from the critic.

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    returns: torch.Tensor
    """The computed (unnormalized) returns for each step.

    The returns are computed following Generalized Advantage Estimation (GAE).

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    advantages: torch.Tensor
    """The computed (normalized) advantages for each step.

    The advantages are computed following Generalized Advantage Estimation (GAE).

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    actions_log_prob: torch.Tensor
    """The log probabilities of the actions.

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    action_mean: torch.Tensor
    """The mean of the action distribution (assuming Gaussian distribution).

    Shape: (mini_batch_size, num_act), dtype: torch.float32
    """

    action_sigma: torch.Tensor
    """The standard deviation of the action distribution (assuming Gaussian distribution).

    Shape: (mini_batch_size, num_act), dtype: torch.float32
    """



class PPO(BaseAlgo):
    config: PPOConfig

    @staticmethod
    def _configure_collection_extras_contract(env: BaseTask) -> None:
        """Request transition-local, sparse episode extras for PPO rollouts."""

        extras_contract_setter = getattr(env, "set_collection_extras_contract", None)
        if callable(extras_contract_setter):
            extras_contract_setter(dense_episode_stats=False)
        else:
            # Compatibility for lightweight/fake environments used by external
            # integrations: the BaseTask implementation exposes the setter.
            setattr(env, "_dense_episode_stats_each_step", False)

    @staticmethod
    def _has_timeout_final_observations(infos: Mapping[str, Any]) -> bool:
        """Return whether this transition carries timeout bootstrap observations.

        ``BaseTask`` publishes ``final_observations`` only on transitions with at
        least one timeout.  Checking that sparse host-side key avoids reducing the
        device ``time_outs`` tensor merely to decide whether bootstrap work exists.
        """

        if "final_observations" not in infos:
            return False
        final_observations = infos["final_observations"]
        return isinstance(final_observations, Mapping) and bool(final_observations)

    def __init__(self, env: BaseTask, config: PPOConfig, log_dir, device="cpu", multi_gpu_cfg: dict | None = None):
        super().__init__(env, config, device, multi_gpu_cfg)
        self._configure_collection_extras_contract(env)
        self.log_dir = log_dir
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.logging_helper = LoggingHelper(
            self.writer,
            self.log_dir,
            device=self.device,
            num_envs=self.env.num_envs,
            num_steps_per_env=self.config.num_steps_per_env,
            num_learning_iterations=self.config.num_learning_iterations,
            is_main_process=self.is_main_process,
            num_gpus=self.gpu_world_size,
        )
        self.algo_timing = StepTiming.from_env(device=self.device)
        self._step_timing_interval = max(1, env_int("HOLOSOMA_STEP_TIMING_INTERVAL", default=1))
        self._last_algo_step_timing: dict[str, dict[str, float]] = {}
        self._last_env_step_timing: dict[str, dict[str, float]] = {}
        # ``train_agent_rank_visible`` deliberately exposes one CUDA device to
        # each worker and rewrites LOCAL_RANK/LOCAL_WORLD_SIZE to 0/1.  Those
        # rewritten values are correct for selecting ``cuda:0`` inside the
        # process, but not for describing the torchrun topology.  Hierarchical
        # collectives must use the preserved physical node topology.
        self.gpu_local_world_size = self._topology_local_world_size()
        self.gpu_topology_local_rank = self._topology_local_rank()
        self._hierarchical_grad_reduce_ready = False
        self._hierarchical_grad_reduce_available = False
        self._hierarchical_grad_reduce_cpu_leader = False
        self._hierarchical_local_group = None
        self._hierarchical_local_barrier_group = None
        self._hierarchical_leader_group = None
        self._hierarchical_leader_gloo_group = None
        self._hierarchical_local_leader_rank = 0
        self._hierarchical_is_leader_rank = False
        # Freeze the control-plane topology choice before any process group is
        # created.  Reading a mutable ambient value at each collective could
        # otherwise send different ranks through incompatible group orders.
        self._hierarchical_small_collectives = self._strict_environment_bool(
            "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES",
            default=False,
        )
        self._hierarchical_pg_timeout_seconds = (
            self._hierarchical_pg_timeout_seconds_from_env()
        )
        self._gloo_grad_reduce_ready = False
        self._gloo_grad_reduce_group = None
        self._gloo_barrier_ready = False
        self._gloo_barrier_group = None
        if self.algo_timing.enabled and self.is_main_process:
            logger.info(
                "Step timing enabled (sync_cuda={}, interval={})",
                self.algo_timing.sync_cuda,
                self._step_timing_interval,
            )

        self._init_config()

        self.current_learning_iteration = 0
        self.eval_callbacks: list[RLEvalCallback] = []
        reset_at_boundary = getattr(
            self.env,
            "reset_all_at_checkpoint_boundary",
            self.env.reset_all,
        )
        _ = reset_at_boundary()

    def _init_config(self) -> None:
        self.algo_obs_dim_dict = self.env.observation_manager.get_obs_dims()

        # Observation manager system - history is defined per-module in module_dict
        assert self.env.observation_manager is not None
        self.algo_history_length_dict = {
            "actor_obs": self.env.observation_manager.cfg.groups["actor_obs"].history_length,
            "critic_obs": self.env.observation_manager.cfg.groups["critic_obs"].history_length,
        }

        self.num_act = self.env.robot_config.actions_dim

        schedule = str(self.config.schedule)
        if schedule not in {"adaptive", "fixed"}:
            raise ValueError(f"PPO schedule must be 'adaptive' or 'fixed', got {schedule!r}.")
        if self.config.desired_kl is not None:
            desired_kl = float(self.config.desired_kl)
            if not math.isfinite(desired_kl) or desired_kl <= 0.0:
                raise ValueError(f"PPO desired_kl must be finite and > 0 when set, got {self.config.desired_kl!r}.")
        elif schedule == "adaptive":
            raise ValueError("PPO schedule='adaptive' requires a positive desired_kl.")
        self.max_grad_norm = self._validate_max_grad_norm(self.config.max_grad_norm)
        (
            self._entropy_coef_start,
            self._entropy_coef_end,
            self._entropy_coef_decay_start_iteration,
            self._entropy_coef_decay_end_iteration,
        ) = self._validate_entropy_coefficient_schedule()
        self.reset_rollout_at_checkpoint = (
            self._reset_rollout_at_checkpoint_enabled()
        )
        self._validate_scientific_fail_closed_environment()
        self._validate_distributed_penalty_curriculum_contract()

        (
            self.actor_learning_rate,
            self.min_actor_learning_rate,
            self.max_actor_learning_rate,
        ) = self._resolve_learning_rate_bounds(
            "actor",
            self.config.actor_learning_rate,
            self.config.min_actor_learning_rate,
            self.config.max_actor_learning_rate,
        )
        (
            self.critic_learning_rate,
            self.min_critic_learning_rate,
            self.max_critic_learning_rate,
        ) = self._resolve_learning_rate_bounds(
            "critic",
            self.config.critic_learning_rate,
            self.config.min_critic_learning_rate,
            self.config.max_critic_learning_rate,
        )

        # Observation related Config
        self.use_symmetry = self.config.use_symmetry
        self._init_obs_keys()
        self._init_obs_slices()
        self._setup_obs_normalizers()
        self.distill_enabled = False
        self.distill_mode = "mse"
        self.dagger_enabled = False
        self.distill_loss_coef = 0.0
        self.bc_loss_coef = 0.0
        self._configured_bc_loss_coef = 0.0
        self.clip_teacher_actions = False
        self.clip_actions_threshold = 0.0
        self.take_teacher_actions = False
        self.teacher_action_mix_ratio = 0.0
        self._configured_teacher_action_mix_ratio = 0.0
        self.teacher_action_mix_ratio_start: float | None = None
        self.teacher_action_mix_ratio_end: float | None = None
        self.teacher_action_mix_ratio_end_iteration = -1
        self.use_teacher_action_mix_schedule = False
        self.switch_to_rl_after = -1
        self.use_multi_teacher = False
        self.multi_teacher_select_obs_var = "teacher_checkpoint_index"
        self.ppo_start_epoch = -1
        self.dagger_end_epoch = -1
        self.ppo_target_coeff = 0.9
        self.ppo_start_coeff = 0.0
        self.ppo_start_noise_std: float | None = None
        self.ppo_start_noise_std_until_coeff = 0.1
        self._ppo_start_noise_std_cap_announced = False
        self.dagger_loss_coef = 10.0
        self.use_ppo_dagger_schedule = False
        self.ppo_coeff = 1.0
        self.distill_loss_fn = F.mse_loss
        self.dagger_ignore_zero_teacher_actions = True
        self.dagger_ignore_episode_initial_steps = 0
        self.dagger_replay_enabled = False
        self.dagger_replay_capacity = 0
        self.dagger_replay_batch_size = 0
        self.dagger_replay_fraction = 0.0
        self.dagger_replay_seed = 0
        self._dagger_replay_buffer: _DaggerReplayBuffer | None = None
        self._dagger_replay_fixed_eval_ready_at_rollout_start = False
        self._dagger_replay_fixed_boundary_verified_runtime = False
        self.dagger_match_std = False
        self.strict_teacher_load = True
        self.teacher_perception_obs_key = ""
        self.teacher_actor = None
        self.teacher_actors: list[nn.Module] = []
        self.teacher_actor_obs_normalizers: dict[str, nn.Module] = {}
        self.teacher_actor_obs_normalizers_list: list[dict[str, nn.Module]] = []
        self.fixed_bc_eval_num_samples = 0
        self.fixed_bc_eval_log_interval = 1
        self.fixed_bc_guard_enabled = False
        self.fixed_bc_guard_reference_end_epoch = 600
        self.fixed_bc_guard_max_reference_ratio = 2.0
        self.fixed_bc_guard_absolute_max_mu_mse = 0.160
        self.fixed_bc_guard_start_epoch = -1
        self.fixed_bc_guard_consecutive_evals = 3
        self._fixed_bc_guard_config_fingerprint = ""
        self._fixed_bc_guard_state: dict[str, Any] = {}
        self._fixed_bc_eval_ready = False
        self._fixed_bc_eval_size = 0
        self._fixed_bc_eval_actor_obs_parts: list[torch.Tensor] = []
        self._fixed_bc_eval_teacher_actions_parts: list[torch.Tensor] = []
        self._fixed_bc_eval_actor_perception_parts: list[torch.Tensor] = []
        self._fixed_bc_eval_dataset: dict[str, torch.Tensor] = {}
        self._terminal_fixed_bc_eval_state: dict[str, Any] | None = None

    @staticmethod
    def _resolve_learning_rate_bounds(
        name: str,
        initial: float,
        minimum: float | None,
        maximum: float | None,
    ) -> tuple[float, float, float]:
        initial = float(initial)
        minimum = min(initial, 1.0e-5) if minimum is None else float(minimum)
        maximum = max(initial, 1.0e-2) if maximum is None else float(maximum)
        values = {"initial": initial, "minimum": minimum, "maximum": maximum}
        invalid = {key: value for key, value in values.items() if not math.isfinite(value) or value <= 0.0}
        if invalid:
            raise ValueError(f"PPO {name} learning rates must be finite and > 0, got {invalid}.")
        if not minimum <= initial <= maximum:
            raise ValueError(
                f"PPO {name} learning-rate bounds must satisfy minimum <= initial <= maximum, "
                f"got minimum={minimum}, initial={initial}, maximum={maximum}."
            )
        return initial, minimum, maximum

    @staticmethod
    def _validate_max_grad_norm(value: Any) -> float:
        if isinstance(value, bool):
            raise ValueError(f"PPO max_grad_norm must be finite and > 0, got {value!r}.")
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"PPO max_grad_norm must be finite and > 0, got {value!r}.") from exc
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise ValueError(f"PPO max_grad_norm must be finite and > 0, got {value!r}.")
        return parsed

    def _validate_entropy_coefficient_schedule(
        self,
    ) -> tuple[float, float, int, int]:
        """Validate and freeze the absolute-iteration entropy schedule."""

        start = self.config.entropy_coef
        end = self.config.entropy_coef_end
        for name, value in (("entropy_coef", start), ("entropy_coef_end", end)):
            if value is None and name == "entropy_coef_end":
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
                raise ValueError(f"PPO {name} must be a finite non-negative real, got {value!r}.")
            parsed = float(value)
            if not math.isfinite(parsed) or parsed < 0.0:
                raise ValueError(f"PPO {name} must be a finite non-negative real, got {value!r}.")

        start_value = float(start)
        if end is None:
            return start_value, start_value, 0, 0

        raw_start_iteration = self.config.entropy_coef_decay_start_iteration
        raw_end_iteration = self.config.entropy_coef_decay_end_iteration
        for name, value in (
            ("entropy_coef_decay_start_iteration", raw_start_iteration),
            ("entropy_coef_decay_end_iteration", raw_end_iteration),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
                raise ValueError(f"PPO {name} must be a non-negative integer, got {value!r}.")
            if int(value) < 0:
                raise ValueError(f"PPO {name} must be a non-negative integer, got {value!r}.")

        start_iteration = int(raw_start_iteration)
        end_iteration = int(raw_end_iteration)
        if end_iteration <= start_iteration:
            raise ValueError(
                "PPO entropy decay requires entropy_coef_decay_end_iteration > "
                "entropy_coef_decay_start_iteration, got "
                f"{end_iteration} <= {start_iteration}."
            )
        return start_value, float(end), start_iteration, end_iteration

    def _operational_entropy_coefficient(self, iteration: int | None = None) -> float:
        """Return the configured coefficient at an absolute learning iteration."""

        iteration = self.current_learning_iteration if iteration is None else int(iteration)
        if iteration <= self._entropy_coef_decay_start_iteration:
            value = self._entropy_coef_start
        elif iteration >= self._entropy_coef_decay_end_iteration:
            value = self._entropy_coef_end
        else:
            fraction = (
                (iteration - self._entropy_coef_decay_start_iteration)
                / (
                    self._entropy_coef_decay_end_iteration
                    - self._entropy_coef_decay_start_iteration
                )
            )
            value = self._entropy_coef_start + fraction * (
                self._entropy_coef_end - self._entropy_coef_start
            )
        return self._operational_float32_loss_weight(value)

    @staticmethod
    def _strict_config_int(name: str, value: Any) -> int:
        """Parse a scientific count/iteration without bool or truncation."""

        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f"distill.{name} must be an integer, got {value!r}.")
        return int(value)

    @staticmethod
    def _strict_config_real(name: str, value: Any) -> float:
        """Parse a finite scientific scalar without bool/string coercion."""

        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"distill.{name} must be a finite real number, got {value!r}.")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"distill.{name} must be finite, got {value!r}.")
        return parsed

    @staticmethod
    def _strict_positive_ppo_int(name: str, value: Any) -> int:
        """Parse a positive PPO count without accepting bools or truncation."""

        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, numbers.Integral)
            or int(value) <= 0
        ):
            raise ValueError(f"PPO {name} must be a positive integer, got {value!r}.")
        return int(value)

    @staticmethod
    def _strict_positive_integer_equivalent(name: str, value: Any) -> int:
        """Parse an integer-valued positive runtime scalar without truncation."""

        parsed: int | None = None
        if (
            not isinstance(value, (bool, np.bool_))
            and isinstance(value, numbers.Real)
        ):
            try:
                value_float = float(value)
            except (TypeError, ValueError, OverflowError):
                value_float = float("nan")
            if (
                math.isfinite(value_float)
                and value_float > 0.0
                and value_float.is_integer()
            ):
                parsed = int(value)
        if parsed is None:
            raise ValueError(
                f"{name} must be a finite positive integer-equivalent value, got {value!r}."
            )
        return parsed

    @staticmethod
    def _canonical_rollout_episode_age_capacity(
        episode_horizon: int,
        block_transitions: int,
        *,
        init_at_random_ep_len: bool,
    ) -> int:
        """Return the largest episode-age support inside one canonical block.

        ``reset_all()`` executes one dummy environment transition before PPO
        clears its DAgger age buffer.  Without random episode-length
        initialization, the first rollout episode therefore has only ``H - 1``
        transitions left (except ``H == 1``, where the dummy transition already
        timed out and returned the reset episode).  Later episodes have the full
        horizon.  A checkpoint reset starts this same pattern again.

        When random episode-length initialization is enabled, PPO overwrites the
        post-dummy episode length with a value in ``[0, H)``.  The maximum
        realizable support used by this impossibility check is then the ordinary
        ``min(H, T)`` (the zero draw realizes it).
        """

        episode_horizon = int(episode_horizon)
        block_transitions = max(0, int(block_transitions))
        if episode_horizon <= 0 or block_transitions <= 0:
            return 0
        if init_at_random_ep_len:
            return min(episode_horizon, block_transitions)

        first_episode_transitions = (
            episode_horizon if episode_horizon == 1 else episode_horizon - 1
        )
        first_capacity = min(block_transitions, first_episode_transitions)
        remaining_transitions = max(
            0,
            block_transitions - first_episode_transitions,
        )
        later_capacity = min(episode_horizon, remaining_transitions)
        return max(first_capacity, later_capacity)

    @staticmethod
    def _operational_float32_loss_weight(value: float) -> float:
        """Materialize a Python loss scalar exactly as the float32 actor graph does."""

        value = float(value)
        if abs(value) > float(np.finfo(np.float32).max):
            return math.copysign(float("inf"), value)
        return float(np.float32(value))

    @classmethod
    def _operational_ppo_coefficient(cls, value: float) -> float:
        """Return the exact PPO blend coefficient seen by the float32 actor graph.

        Rollout action selection, teacher-control validation, and the actor loss
        must agree on whether PPO is present.  Keeping a positive Python
        subnormal here while tensor multiplication rounds it to zero makes the
        rollout stochastic/on-policy contract describe an objective that is not
        actually optimized.
        """

        clamped = max(0.0, min(1.0, float(value)))
        return cls._operational_float32_loss_weight(clamped)

    @staticmethod
    def _strict_config_bool(name: str, value: Any) -> bool:
        """Reject truthy strings/integers that invert a typed config switch."""

        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"distill.{name} must be a boolean, got {value!r}.")
        return bool(value)

    @staticmethod
    def _strict_environment_bool(name: str, *, default: bool) -> bool:
        raw_value = os.environ.get(name)
        if raw_value is None:
            return default
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"", "0", "false", "no", "off"}:
            return False
        raise ValueError(f"{name} must be an explicit boolean value, got {raw_value!r}.")

    @staticmethod
    def _strict_environment_nonnegative_int(name: str, *, default: int) -> int:
        raw_value = os.environ.get(name)
        if raw_value is None or raw_value.strip() == "":
            return int(default)
        normalized = raw_value.strip()
        if not normalized.isascii() or not normalized.isdecimal():
            raise ValueError(
                f"{name} must be a base-10 non-negative integer, got {raw_value!r}."
            )
        parsed = int(normalized, 10)
        if parsed < 0:
            raise ValueError(
                f"{name} must be a base-10 non-negative integer, got {raw_value!r}."
            )
        return parsed

    @classmethod
    def _hierarchical_pg_timeout_seconds_from_env(cls) -> int:
        seconds = cls._strict_environment_nonnegative_int(
            "HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC",
            default=300,
        )
        if seconds <= 0 or seconds > 2_147_483_647:
            raise ValueError(
                "HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC must be a base-10 integer "
                f"in [1, 2147483647], got {seconds!r}."
            )
        return seconds

    def _hierarchical_pg_timeout(self) -> timedelta:
        seconds = getattr(self, "_hierarchical_pg_timeout_seconds", None)
        if seconds is None:
            seconds = self._hierarchical_pg_timeout_seconds_from_env()
        return timedelta(seconds=int(seconds))

    @staticmethod
    def _validate_scientific_fail_closed_environment() -> None:
        """Reject legacy switches that can conceal an invalid optimization.

        Production launchers already prohibit these variables.  Keeping the
        same guard inside PPO protects direct Python entry points and future
        launchers from silently disabling scientific integrity checks.
        """

        false_values = {"", "0", "false", "no", "off"}
        true_values = {"1", "true", "yes", "on"}
        for name in (
            "HOLOSOMA_SKIP_GRAD_FINITE_CHECK",
            "HOLOSOMA_SKIP_LOSS_FINITE_CHECK",
            "HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION",
            "HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC",
        ):
            raw_value = os.environ.get(name, "").strip().lower()
            if raw_value in false_values:
                continue
            if raw_value in true_values:
                raise RuntimeError(
                    f"{name} cannot be enabled: PPO scientific finite checks and loss accounting "
                    "are mandatory."
                )
            raise ValueError(
                f"{name} must be an explicit boolean value, got {os.environ.get(name)!r}."
            )

    def _reset_step_timing(self) -> None:
        if not self.algo_timing.enabled:
            return
        self.algo_timing.reset()
        env_timing = getattr(self.env, "step_timing", None)
        if env_timing is not None:
            env_timing.reset()

    def _capture_step_timing(self) -> None:
        if not self.algo_timing.enabled:
            return
        self._last_algo_step_timing = self.algo_timing.snapshot(reset=False)
        env_timing = getattr(self.env, "step_timing", None)
        if env_timing is None:
            self._last_env_step_timing = {}
        else:
            self._last_env_step_timing = env_timing.snapshot(reset=False)

    def _emit_step_timing_summary(self, it: int) -> None:
        if not self.algo_timing.enabled or not self.is_main_process:
            return
        if it % self._step_timing_interval != 0:
            return
        algo_order = (
            "iteration/rollout",
            "iteration/training_step",
            "rollout/env_step",
            "rollout/teacher_actions",
            "rollout/actor_forward",
            "rollout/critic_forward",
            "rollout/returns",
            "training/update_algo_step",
        )
        env_order = (
            "env_step_total",
            "physics",
            "physics/apply_force",
            "physics/simulate_step",
            "physics/sim/write_data_to_sim",
            "physics/sim/write_robot_to_sim",
            "physics/sim/robot/apply_actuator_model",
            "physics/sim/robot/set_dof_forces",
            "physics/sim/write_nonrobot_to_sim",
            "physics/sim/step",
            "physics/sim/scene_update",
            "physics/sim/update_dof_refs",
            "post/perception",
            "post/reward",
            "post/reward/term/offline_contact_guidance",
            "post/log_update",
            "post/log_update/update_log_dict",
            "post/log_update/motion_metrics",
            "post/tasks",
            "post/tasks/command_manager",
            "post/tasks/motion/contact_prior",
            "post/tasks/motion/future_targets",
            "post/tasks/motion/relative_body_pose",
            "post/observations",
            "post/reset_envs",
            "pre_physics",
        )
        logger.info(
            "StepTiming iter={} algo {}",
            it,
            compact_timing_summary(self._last_algo_step_timing, algo_order, max_extra=4),
        )
        logger.info(
            "StepTiming iter={} env {}",
            it,
            compact_timing_summary(self._last_env_step_timing, env_order, max_extra=4),
        )

    def _add_step_timing_logs(self, extra_log_dicts: dict[str, dict[str, float]]) -> None:
        if not self.algo_timing.enabled:
            return
        timing_logs = extra_log_dicts.setdefault("Timing", {})
        selected = (
            ("algo", self._last_algo_step_timing, "iteration/rollout"),
            ("algo", self._last_algo_step_timing, "iteration/training_step"),
            ("algo", self._last_algo_step_timing, "rollout/env_step"),
            ("algo", self._last_algo_step_timing, "rollout/teacher_actions"),
            ("algo", self._last_algo_step_timing, "rollout/actor_forward"),
            ("algo", self._last_algo_step_timing, "rollout/critic_forward"),
            ("algo", self._last_algo_step_timing, "rollout/returns"),
            ("algo", self._last_algo_step_timing, "training/update_algo_step"),
            ("env", self._last_env_step_timing, "env_step_total"),
            ("env", self._last_env_step_timing, "physics"),
            ("env", self._last_env_step_timing, "post/perception"),
            ("env", self._last_env_step_timing, "post/reward"),
            ("env", self._last_env_step_timing, "post/log_update"),
            ("env", self._last_env_step_timing, "post/log_update/update_log_dict"),
            ("env", self._last_env_step_timing, "post/log_update/motion_metrics"),
            ("env", self._last_env_step_timing, "post/tasks"),
            ("env", self._last_env_step_timing, "post/tasks/command_manager"),
            ("env", self._last_env_step_timing, "post/tasks/motion/contact_prior"),
            ("env", self._last_env_step_timing, "post/tasks/motion/future_targets"),
            ("env", self._last_env_step_timing, "post/tasks/motion/relative_body_pose"),
            ("env", self._last_env_step_timing, "post/observations"),
            ("env", self._last_env_step_timing, "post/reset_envs"),
        )
        for prefix, snapshot, name in selected:
            stats = snapshot.get(name)
            if stats is None:
                continue
            safe_name = name.replace("/", "_")
            timing_logs[f"{prefix}_{safe_name}_sum_ms"] = float(stats.get("sum_ms", 0.0))
            timing_logs[f"{prefix}_{safe_name}_mean_ms"] = float(stats.get("mean_ms", 0.0))

    def _build_obs_slices(self, keys: list[str]) -> dict[str, slice]:
        seen: set[str] = set()
        duplicates: list[str] = []
        for key in keys:
            if key in seen and key not in duplicates:
                duplicates.append(key)
            seen.add(key)
        if duplicates:
            # The slice table is keyed by group name.  Repeating a group used
            # to overwrite its first slice, after which normalization and
            # ONNX export read the last occurrence twice even though rollout
            # concatenation still contained both occurrences.
            raise ValueError(
                "Policy observation input groups must be unique; "
                f"duplicates={duplicates}, ordered_inputs={list(keys)}."
            )
        slices: dict[str, slice] = {}
        start = 0
        for key in keys:
            dim = self.algo_obs_dim_dict[key]
            slices[key] = slice(start, start + dim)
            start += dim
        return slices

    def _init_obs_slices(self) -> None:
        self.actor_obs_slices = self._build_obs_slices(self.actor_obs_keys)
        self.critic_obs_slices = self._build_obs_slices(self.critic_obs_keys)
        self.teacher_obs_keys = list(self.actor_obs_keys)
        self.teacher_obs_slices = dict(self.actor_obs_slices)
        self.teacher_obs_dim = self._get_obs_dim(self.teacher_obs_keys)

    def _setup_obs_normalizers(self) -> None:
        self.actor_obs_normalizers = self._build_group_normalizers(self.actor_obs_keys, self.config.normalize_actor_obs)
        self.critic_obs_normalizers = self._build_group_normalizers(self.critic_obs_keys, self.config.normalize_critic_obs)

    def _build_group_normalizers(
        self,
        keys: list[str],
        enabled: bool,
        *,
        eps: float | None = None,
        until: int | None | object = _UNSET,
    ) -> dict[str, nn.Module]:
        normalizers: dict[str, nn.Module] = {}
        normalizer_eps = self.config.obs_normalizer_eps if eps is None else float(eps)
        normalizer_until = self.config.obs_normalizer_until if until is _UNSET else until
        for key in keys:
            dim = self.algo_obs_dim_dict[key]
            if enabled:
                normalizers[key] = EmpiricalNormalization(
                    shape=(dim,),
                    device=self.device,
                    eps=normalizer_eps,
                    until=normalizer_until,
                )
            else:
                normalizers[key] = nn.Identity()
        return normalizers

    def _apply_obs_normalizer(self, normalizer: nn.Module, obs: torch.Tensor, update: bool) -> torch.Tensor:
        if isinstance(normalizer, EmpiricalNormalization):
            return normalizer(
                obs,
                update=update,
                sample_weight=self._get_distributed_loss_weight() if update else 1.0,
            )
        return normalizer(obs)

    def _normalize_concat_obs(
        self,
        obs: torch.Tensor,
        keys: list[str],
        slices: dict[str, slice],
        normalizers: dict[str, nn.Module],
        *,
        update: bool,
    ) -> torch.Tensor:
        parts = []
        for key in keys:
            part = obs[..., slices[key]]
            part = self._apply_obs_normalizer(normalizers[key], part, update)
            parts.append(part)
        return torch.cat(parts, dim=-1)

    def _normalize_actor_obs(self, obs: torch.Tensor, *, update: bool) -> torch.Tensor:
        if not self.config.normalize_actor_obs:
            return obs
        return self._normalize_concat_obs(
            obs, self.actor_obs_keys, self.actor_obs_slices, self.actor_obs_normalizers, update=update
        )

    def _normalize_critic_obs(self, obs: torch.Tensor, *, update: bool) -> torch.Tensor:
        if not self.config.normalize_critic_obs:
            return obs
        return self._normalize_concat_obs(
            obs, self.critic_obs_keys, self.critic_obs_slices, self.critic_obs_normalizers, update=update
        )

    def _normalize_teacher_actor_obs(
        self, obs: torch.Tensor, normalizers: dict[str, nn.Module] | None = None
    ) -> torch.Tensor:
        if not self.distill_enabled:
            return obs
        if obs.shape[-1] != self.teacher_obs_dim:
            raise ValueError(
                f"Teacher obs dim mismatch: expected {self.teacher_obs_dim}, got {obs.shape[-1]}"
            )
        if normalizers is None:
            normalizers = self.teacher_actor_obs_normalizers
        if all(not isinstance(normalizer, EmpiricalNormalization) for normalizer in normalizers.values()):
            return obs
        return self._normalize_concat_obs(
            obs,
            self.teacher_obs_keys,
            self.teacher_obs_slices,
            normalizers,
            update=False,
        )

    def _project_actor_std_constraints(
        self,
        std: torch.Tensor,
        *,
        actor: nn.Module | None = None,
    ) -> torch.Tensor:
        """Project finite Gaussian std entries while preserving NaN/Inf."""
        actor = self.actor if actor is None else actor
        finite_mask = torch.isfinite(std)
        projected = torch.clamp(std, min=1e-6)
        min_noise_std = getattr(actor, "min_noise_std", None)
        min_mean_noise_std = getattr(actor, "min_mean_noise_std", None)
        if min_noise_std:
            projected = torch.clamp(projected, min=min_noise_std)
        max_noise_std = getattr(actor, "max_noise_std", None)
        if max_noise_std is not None:
            # Apply the component-wise cap before evaluating the mean floor.
            # Otherwise one oversized component can hide an under-sized mean,
            # only to be capped after the floor check and leave the effective
            # distribution below ``min_mean_noise_std``.
            projected = torch.clamp(projected, max=float(max_noise_std))
        if min_noise_std is None and min_mean_noise_std:
            current_mean = projected.mean()
            if max_noise_std is not None:
                alpha = (min_mean_noise_std - current_mean) / (max_noise_std - current_mean)
                alpha = torch.clamp(alpha, min=0.0, max=1.0)
                raised = projected + alpha * (max_noise_std - projected)
            else:
                raised = projected * (min_mean_noise_std / current_mean)
            needs_mean_floor = finite_mask.all() & (current_mean < min_mean_noise_std)
            projected = torch.where(needs_mean_floor, raised, projected)
        if max_noise_std is not None:
            projected = torch.clamp(projected, max=float(max_noise_std))
        return torch.where(finite_mask, projected, std)

    def _get_actor_std_for_loss(self, actor: nn.Module) -> torch.Tensor:
        return self._project_actor_std_constraints(actor.std, actor=actor)

    @staticmethod
    def _std_matching_loss_per_sample(
        student_sigma: torch.Tensor,
        teacher_sigma: torch.Tensor,
    ) -> torch.Tensor:
        return (student_sigma - teacher_sigma).pow(2).mean(dim=-1)

    def _init_obs_keys(self):
        self.actor_obs_keys = self.config.module_dict.actor.input_dim
        self.critic_obs_keys = self.config.module_dict.critic.input_dim
        self.actor_perception_key = self.config.module_dict.actor.layer_config.perception_input_name or ""
        self.critic_perception_key = self.config.module_dict.critic.layer_config.perception_input_name or ""
        if self.actor_perception_key and self.actor_perception_key not in self.algo_obs_dim_dict:
            raise ValueError(f"Actor perception key '{self.actor_perception_key}' not found in observation manager.")
        if self.critic_perception_key and self.critic_perception_key not in self.algo_obs_dim_dict:
            raise ValueError(f"Critic perception key '{self.critic_perception_key}' not found in observation manager.")

    def _configure_active_observation_groups(self) -> None:
        observation_manager = getattr(self.env, "observation_manager", None)
        if observation_manager is None or not hasattr(observation_manager, "set_active_groups"):
            return

        disabled = os.environ.get("HOLOSOMA_DISABLE_ACTIVE_OBS_GROUP_FILTER", "").strip().lower()
        if disabled in ("1", "true", "yes", "on"):
            observation_manager.set_active_groups(None)
            return

        required: list[str] = []

        def add_group(group_name: str | None) -> None:
            if not group_name:
                return
            if group_name not in required:
                required.append(group_name)

        def add_groups(group_names: list[str] | tuple[str, ...] | None) -> None:
            if not group_names:
                return
            for group_name in group_names:
                add_group(group_name)

        add_groups(self.actor_obs_keys)
        critic_observations_required = bool(
            getattr(self, "_evaluation_only", False)
            or self._critic_optimizer_objective_enabled()
        )
        if critic_observations_required:
            add_groups(self.critic_obs_keys)
        add_group(self.actor_perception_key)
        if critic_observations_required:
            add_group(self.critic_perception_key)
        # Teacher-policy evaluation is selected only after the student checkpoint
        # and its authenticated provenance have been loaded.  At construction
        # time distillation is deliberately disabled for inference, so the normal
        # rollout predicate cannot discover this late-bound dependency.  Keep the
        # teacher groups active explicitly in this evaluation mode; otherwise the
        # first evaluation step receives only the student groups and cannot build
        # the teacher input.
        teacher_policy_evaluation = bool(
            getattr(self, "_evaluation_only", False)
            and getattr(self, "_evaluation_policy_mode", "checkpoint_actor")
            in {
                "distill_label_teacher",
                "distill_label_teacher_bc_target",
            }
        )
        teacher_observations_required = (
            teacher_policy_evaluation
            or self._teacher_observations_required_for_current_rollout()
        )
        if teacher_observations_required:
            add_groups(self.teacher_obs_keys)
            add_group(self.teacher_perception_obs_key)
            teacher_candidates = (
                list(getattr(self, "teacher_actors", ()) or ())
                if self.use_multi_teacher
                else (
                    [self.teacher_actor]
                    if getattr(self, "teacher_actor", None) is not None
                    else []
                )
            )
            for teacher_actor in teacher_candidates:
                # A strict checkpoint may supply its own compatible side-input
                # group even when teacher_perception_obs_key was not explicitly
                # repeated in the student config.  Keep the actual loaded actor's
                # dependency active rather than discovering the omission during
                # the first rollout.
                add_group(str(getattr(teacher_actor, "perception_input_name", "") or ""))
            if self.use_multi_teacher:
                add_group(self.multi_teacher_select_obs_var)

        previous_groups = getattr(observation_manager, "active_group_names", None)
        observation_manager.set_active_groups(required)
        if self.is_main_process and tuple(required) != previous_groups:
            total = len(getattr(observation_manager.cfg, "groups", {}))
            logger.info("Active PPO observation groups: {} / {} {}", len(required), total, required)

    def _validate_hierarchical_collective_setup_agreement(self) -> bool:
        """Prove all-rank control/topology agreement before creating subgroups.

        ``new_group`` is an ordered world-level operation even when most ranks
        are not members of the resulting group.  A rank-local configuration or
        topology error must therefore be discovered through the already
        initialized default process group, before any rank conditionally enters
        subgroup creation.
        """

        requested = self._hierarchical_small_collectives_enabled()
        if requested and getattr(self, "_evaluation_only", False) and not self.is_multi_gpu:
            # A single-rank evaluator has no distributed control plane to
            # accelerate.  It may legitimately inherit the training
            # provenance environment, so do not require a multi-node topology
            # for a path that cannot be entered.
            return False
        if (
            not self.is_multi_gpu
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return requested

        backend = str(torch.distributed.get_backend()).strip().lower()
        if backend == "nccl":
            agreement_device = torch.device(self.device)
            if agreement_device.type != "cuda":
                raise RuntimeError(
                    "NCCL hierarchical-collective setup agreement requires a CUDA device, "
                    f"got {self.device!r}."
                )
        elif backend == "gloo":
            agreement_device = torch.device("cpu")
        else:
            raise RuntimeError(
                "Hierarchical-collective setup agreement supports only NCCL or Gloo "
                f"default process groups, got {backend!r}."
            )

        local_record = torch.tensor(
            [
                int(requested),
                int(self._gloo_small_collectives_enabled()),
                int(self._hierarchical_grad_reduce_enabled()),
                int(self._hierarchical_grad_reduce_cpu_leader_enabled()),
                int(self._hierarchical_pg_timeout().total_seconds()),
                int(self.gpu_world_size),
                int(self.gpu_global_rank),
                int(self.gpu_local_world_size),
                int(self.gpu_topology_local_rank),
            ],
            dtype=torch.int64,
            device=agreement_device,
        )
        actual_world_size = int(torch.distributed.get_world_size())
        gathered = [torch.empty_like(local_record) for _ in range(actual_world_size)]
        if agreement_device.type == "cuda":
            torch.cuda.synchronize(agreement_device)
        torch.distributed.all_gather(gathered, local_record)
        if agreement_device.type == "cuda":
            torch.cuda.synchronize(agreement_device)
        records = [tuple(int(value) for value in item.cpu().tolist()) for item in gathered]

        problems: list[str] = []
        reference_config = records[0][:5]
        for slot, record in enumerate(records):
            if record[:5] != reference_config:
                problems.append(
                    f"rank {slot} collective config={record[:5]} "
                    f"differs from rank 0={reference_config}"
                )
            if record[5] != actual_world_size:
                problems.append(
                    f"rank {slot} reports world_size={record[5]}, actual={actual_world_size}"
                )
            if record[6] != slot:
                problems.append(f"slot {slot} reports global_rank={record[6]}")

        (
            requested_on_all,
            gloo_small_on_all,
            hierarchical_grad_on_all,
            _cpu_leader,
            _hierarchical_timeout,
        ) = (
            reference_config
        )
        local_world_size = records[0][7]
        if requested_on_all or hierarchical_grad_on_all:
            local_world_sizes = {record[7] for record in records}
            if len(local_world_sizes) != 1:
                problems.append(
                    "ranks disagree on preserved local_world_size: "
                    + repr(sorted(local_world_sizes))
                )
                local_world_size = 0
            else:
                local_world_size = next(iter(local_world_sizes))
            if local_world_size <= 0 or actual_world_size % local_world_size != 0:
                problems.append(
                    "preserved topology must have a positive local_world_size dividing world_size: "
                    f"world_size={actual_world_size}, local_world_size={local_world_size}"
                )
            else:
                for slot, record in enumerate(records):
                    expected_local_rank = slot % local_world_size
                    if record[8] != expected_local_rank:
                        problems.append(
                            f"rank {slot} reports topology_local_rank={record[8]}, "
                            f"expected={expected_local_rank}"
                        )

        if requested_on_all:
            if not gloo_small_on_all or not hierarchical_grad_on_all:
                problems.append(
                    "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=1 requires "
                    "HOLOSOMA_GLOO_SMALL_COLLECTIVES=1 and "
                    "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1"
                )
            if local_world_size <= 1 or actual_world_size <= local_world_size:
                problems.append(
                    "hierarchical small collectives require multiple ranks per node and "
                    "multiple nodes"
                )

        if problems:
            raise RuntimeError(
                "Hierarchical collective setup agreement failed before subgroup creation: "
                + "; ".join(problems)
            )
        return bool(requested_on_all)

    def setup(self):
        logger.info("Setting up PPO")
        # Process-group creation is itself a world-order collective.  Build the
        # shared local/leader groups at one unconditional all-rank setup boundary
        # instead of lazily from a later, potentially conditional, verdict.
        hierarchical_small_collectives = self._validate_hierarchical_collective_setup_agreement()
        if hierarchical_small_collectives:
            if not self._gloo_small_collectives_enabled():
                raise RuntimeError(
                    "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=1 requires "
                    "HOLOSOMA_GLOO_SMALL_COLLECTIVES=1."
                )
            if not self._hierarchical_grad_reduce_enabled():
                raise RuntimeError(
                    "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=1 requires "
                    "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1 so the local topology groups "
                    "have one deterministic owner and creation order."
                )
            if not self._setup_hierarchical_grad_reduce_groups():
                raise RuntimeError(
                    "Hierarchical small collectives were requested but the distributed "
                    "topology cannot provide node-local Gloo and leader Gloo groups."
                )
            if (
                self._hierarchical_local_barrier_group is None
                or self._hierarchical_leader_gloo_group is None
            ):
                raise RuntimeError(
                    "Hierarchical small collectives were requested but their Gloo groups "
                    "were not created on every rank."
                )
        self._setup_models_and_optimizer()
        debug_heartbeat = os.environ.get("HOLOSOMA_DEBUG_HEARTBEAT", "").lower() not in ("", "0", "false", "no")
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} PPO.setup models/optimizers ready", self.gpu_global_rank)
        self._configure_active_observation_groups()
        if getattr(self, "_evaluation_only", False):
            logger.info(
                "PPO evaluation-only setup: skipped teacher construction, optimizers, and rollout storage."
            )
            return
        logger.info("Setting up Storage")
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} PPO.setup storage begin", self.gpu_global_rank)
        self._setup_storage()
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} PPO.setup storage finished", self.gpu_global_rank)

        # Log curriculum synchronization status for multi-GPU training
        if self.is_multi_gpu:
            if self.has_curricula_enabled():
                logger.info(f"Multi-GPU curriculum synchronization enabled across {self.gpu_world_size} GPUs")

    def _setup_models_and_optimizer(self):
        debug_heartbeat = os.environ.get("HOLOSOMA_DEBUG_HEARTBEAT", "").lower() not in ("", "0", "false", "no")
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} setup actor/critic begin", self.gpu_global_rank)
        self.actor = setup_ppo_actor_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=self.config.module_dict.actor,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std,
            device=self.device,
            history_length=self.algo_history_length_dict,
        )
        self.critic = setup_ppo_critic_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=self.config.module_dict.critic,
            device=self.device,
            history_length=self.algo_history_length_dict,
        )
        defm_restore_mode = os.environ.get(
            "HOLOSOMA_DEFM_MATERIALIZATION_MODE",
            "fresh",
        ).strip().lower()
        if defm_restore_mode not in {"fresh", "policy_init", "stage4_init", "full_resume"}:
            raise ValueError(
                "HOLOSOMA_DEFM_MATERIALIZATION_MODE must be fresh, policy_init, "
                "stage4_init, or full_resume."
            )
        self._materialize_lazy_model_modules(
            self.actor,
            checkpoint_restore=defm_restore_mode in {"policy_init", "stage4_init", "full_resume"},
        )
        self._materialize_lazy_model_modules(
            self.critic,
            checkpoint_restore=defm_restore_mode in {"stage4_init", "full_resume"},
        )
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} setup actor/critic finished", self.gpu_global_rank)
        self._sanitize_actor_std()
        self.use_time_gru = bool(
            getattr(self.actor, "perception_time_gru", None) is not None
            or getattr(self.critic, "perception_time_gru", None) is not None
        )

        if debug_heartbeat:
            logger.info("Heartbeat: rank {} setup distillation begin", self.gpu_global_rank)
        self._setup_distillation()
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} setup distillation finished", self.gpu_global_rank)

        if not getattr(self, "_evaluation_only", False):
            self._validate_training_objective_configuration()

        if self.use_symmetry and not getattr(self, "_evaluation_only", False):
            self.symmetry_utils = SymmetryUtils(self.env)

        # Synchronize model weights across GPUs after initialization
        if self.is_multi_gpu:
            if debug_heartbeat:
                logger.info("Heartbeat: rank {} model weight sync begin", self.gpu_global_rank)
            self._synchronize_model_weights()
            if debug_heartbeat:
                logger.info("Heartbeat: rank {} model weight sync finished", self.gpu_global_rank)

        if getattr(self, "_evaluation_only", False):
            self._assert_model_parameters_finite(
                phase="evaluation-only model setup",
                trainable_only=False,
            )
            return

        if debug_heartbeat:
            logger.info("Heartbeat: rank {} optimizer setup begin", self.gpu_global_rank)
        self.actor_optimizer = instantiate(
            self.config.actor_optimizer, params=self.actor.parameters(), lr=self.actor_learning_rate
        )
        self._validate_optimizer_parameter_coverage(
            self.actor,
            self.actor_optimizer,
            role="actor",
        )
        self.critic_optimizer = instantiate(
            self.config.critic_optimizer, params=self.critic.parameters(), lr=self.critic_learning_rate
        )
        self._validate_optimizer_parameter_coverage(
            self.critic,
            self.critic_optimizer,
            role="critic",
        )
        self._assert_model_parameters_finite(
            phase="model setup",
            trainable_only=False,
        )
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} optimizer setup finished", self.gpu_global_rank)

    def _materialize_lazy_model_modules(
        self,
        *models: nn.Module,
        checkpoint_restore: bool = False,
    ) -> None:
        """Materialize setup-sensitive modules without advancing training RNG.

        Lazy parameters must exist before model synchronization, optimizer
        construction, and checkpoint schema validation.  External model
        factories may consume Python, NumPy, torch CPU, or torch CUDA RNG, so
        the complete process-global RNG state is restored even when setup
        fails.  The initialized parameters remain in the modules; only the
        stochastic streams are rolled back to the pre-materialization state.
        """

        materializers = []
        seen_module_ids: set[int] = set()
        for model in models:
            for module in tuple(model.modules()):
                module_id = id(module)
                if module_id in seen_module_ids:
                    continue
                seen_module_ids.add(module_id)
                materialize = (
                    getattr(module, "materialize_for_checkpoint_restore", None)
                    if checkpoint_restore
                    else None
                )
                if not callable(materialize):
                    materialize = getattr(module, "materialize_for_setup", None)
                if callable(materialize):
                    materializers.append(materialize)

        if not materializers:
            return

        rng_state = capture_rng_checkpoint_state()
        try:
            for materialize in materializers:
                materialize(self.device)
        finally:
            restore_rng_checkpoint_state(
                rng_state,
                path="pre_lazy_model_materialization_rng_state",
            )

    @staticmethod
    def _validate_optimizer_parameter_coverage(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        role: str,
    ) -> None:
        """Fail setup if a trainable parameter is absent from its optimizer."""

        optimizer_parameter_ids = {
            id(parameter)
            for parameter_group in optimizer.param_groups
            for parameter in parameter_group["params"]
        }
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) not in optimizer_parameter_ids
        ]
        if missing:
            preview = ", ".join(missing[:8])
            suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
            raise RuntimeError(
                f"The {role} optimizer does not cover trainable model parameters: {preview}{suffix}. "
                "Materialize all lazy modules before constructing the optimizer."
            )

    def _assert_model_parameters_finite(
        self,
        *,
        phase: str,
        trainable_only: bool,
        include_optimizer_state: bool = False,
    ) -> None:
        """Fail all ranks together if live model numerical state is non-finite.

        This boundary runs after setup and once after a complete optimizer
        iteration, rather than on every rollout step or minibatch.  It closes
        the gap where actor output sanitization could otherwise turn a NaN
        parameter into finite zero actions and zero gradients indefinitely.
        At iteration boundaries it also scans optimizer floating tensors: an
        infinite Adam second moment can leave parameters finite while making a
        newly published checkpoint scientifically unusable.
        """

        named_parameters = [
            (f"{role}.{name}", parameter)
            for role, model in (("actor", self.actor), ("critic", self.critic))
            for name, parameter in model.named_parameters()
            if not isinstance(parameter, UninitializedParameter)
            and (parameter.requires_grad or not trainable_only)
        ]
        # Buffers (for example BatchNorm running statistics) can change the
        # policy even though they are not optimized parameters.  Scan them at
        # every iteration boundary as well as during setup/load validation.
        named_buffers = [
            (f"{role}.{name}", buffer)
            for role, model in (("actor", self.actor), ("critic", self.critic))
            for name, buffer in model.named_buffers()
            if buffer.is_floating_point() or buffer.is_complex()
        ]
        named_optimizer_state: list[tuple[str, torch.Tensor]] = []
        non_finite_optimizer_scalars: list[str] = []
        if include_optimizer_state:
            for role, optimizer in (
                ("actor_optimizer", getattr(self, "actor_optimizer", None)),
                ("critic_optimizer", getattr(self, "critic_optimizer", None)),
            ):
                if optimizer is None:
                    continue
                stack: list[tuple[str, Any]] = [(role, optimizer.state_dict())]
                while stack:
                    state_path, value = stack.pop()
                    if isinstance(value, torch.Tensor):
                        if value.is_floating_point() or value.is_complex():
                            named_optimizer_state.append((state_path, value))
                    elif isinstance(value, dict):
                        stack.extend(
                            (f"{state_path}.{key}", child)
                            for key, child in value.items()
                        )
                    elif isinstance(value, (list, tuple)):
                        stack.extend(
                            (f"{state_path}[{index}]", child)
                            for index, child in enumerate(value)
                        )
                    elif isinstance(value, numbers.Number) and not isinstance(
                        value,
                        (bool, numbers.Integral),
                    ):
                        try:
                            scalar_finite = math.isfinite(value)
                        except TypeError:
                            scalar_finite = math.isfinite(float(value.real)) and math.isfinite(float(value.imag))
                        if not scalar_finite:
                            non_finite_optimizer_scalars.append(state_path)

        named_floating_state = [*named_parameters, *named_buffers, *named_optimizer_state]
        finite_checks = [
            torch.isfinite(value).all().to(device=self.device)
            for _, value in named_floating_state
        ]
        if finite_checks:
            local_bad = ~torch.stack(finite_checks).all()
        else:
            local_bad = torch.zeros((), device=self.device, dtype=torch.bool)
        if non_finite_optimizer_scalars:
            local_bad = local_bad | torch.ones((), device=self.device, dtype=torch.bool)

        global_bad = local_bad.to(dtype=torch.int32)
        if (
            getattr(self, "is_multi_gpu", False)
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            global_bad = self._all_reduce_small_tensor(
                global_bad,
                op=torch.distributed.ReduceOp.MAX,
            )
        if not bool(global_bad.item()):
            return

        local_bad_names = []
        if bool(local_bad.item()):
            local_bad_names = [
                name for name, value in named_floating_state if not bool(torch.isfinite(value).all().item())
            ]
            local_bad_names.extend(non_finite_optimizer_scalars)
        if local_bad_names:
            preview = ", ".join(local_bad_names[:8])
            suffix = "" if len(local_bad_names) <= 8 else f", ... ({len(local_bad_names)} total)"
            detail = f"this rank has: {preview}{suffix}"
        else:
            detail = "another rank reported non-finite trainable state"
        raise FloatingPointError(
            f"Non-finite model/optimizer state detected after {phase}; {detail}. "
            "Refusing to continue with a corrupt policy state."
        )

    def _ppo_can_contribute(self) -> bool:
        if not getattr(self, "dagger_enabled", True):
            return True
        if self.use_ppo_dagger_schedule:
            return bool(
                self._operational_ppo_coefficient(self.ppo_start_coeff) > 0.0
                or self._operational_ppo_coefficient(self.ppo_target_coeff) > 0.0
            )
        # Any positive PPO weight requires a valid on-policy action/log-prob
        # pair.  Do not round a coefficient close to one into "pure BC": the
        # loss below uses the exact positive weight ``1 - bc_loss_coef``.
        return self._configured_bc_loss_coef < 1.0 or self.switch_to_rl_after > 0

    def _validate_training_objective_configuration(self) -> None:
        forbidden_training_overrides = (
            "HOLOSOMA_DISABLE_ACTIVE_OBS_GROUP_FILTER",
            "HOLOSOMA_DISABLE_AUTO_RESET",
            "HOLOSOMA_DISABLE_CLIP_END_RESET",
            "HOLOSOMA_DISABLE_MOTION_END_RESET",
            "HOLOSOMA_DISABLE_BAD_TRACKING_RESET",
        )
        for name in forbidden_training_overrides:
            if self._strict_environment_bool(name, default=False):
                raise RuntimeError(
                    f"{name} cannot be enabled for scientific PPO training: it bypasses the configured "
                    "observation or episode/reset contract. Express the intended experiment in the typed "
                    "observation/termination configuration instead of an ambient debug override."
                )
        if not self._strict_environment_bool(
            "HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE",
            default=True,
        ):
            raise RuntimeError(
                "HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE cannot be disabled for scientific PPO training: "
                "custom multi-stream rollout/backward work must be quiescent before gradients are packed."
            )

        if self.use_symmetry and (self.actor_perception_key or self.critic_perception_key):
            raise ValueError(
                "PPO symmetry augmentation is not supported with external actor/critic perception inputs: "
                "the corresponding perception transform is undefined. Disable symmetry or include a fully "
                "specified perception transformation before training."
            )

        if self.use_time_gru:
            actor_gru = getattr(self.actor, "perception_time_gru", None)
            critic_gru = getattr(self.critic, "perception_time_gru", None)
            if actor_gru is None or critic_gru is None:
                raise ValueError("time_gru PPO requires recurrent encoders for both actor and critic.")
            if self.distill_enabled or (self.distill_mode == "dagger" and self.dagger_enabled):
                raise ValueError(
                    "Distillation is not supported in time_gru mode; refusing before rollout instead of "
                    "failing at the first recurrent optimizer update."
                )

        ppo_start_noise_std = getattr(self, "ppo_start_noise_std", None)
        if ppo_start_noise_std is not None:
            configured_cap = float(ppo_start_noise_std)
            required_floors = [1e-6]
            for attr_name in ("min_noise_std", "min_mean_noise_std"):
                value = getattr(self.actor, attr_name, None)
                if value is not None:
                    required_floors.append(float(value))
            required_floor = max(required_floors)
            if configured_cap < required_floor:
                raise ValueError(
                    "distill.ppo_start_noise_std conflicts with the actor's policy-noise floor: "
                    f"cap={configured_cap}, required_floor={required_floor}."
                )

        supervised_only = self._strict_environment_bool(
            "HOLOSOMA_DAGGER_SUPERVISED_ONLY",
            default=False,
        )
        actor_only_step = self._strict_environment_bool(
            "HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP",
            default=False,
        )
        stream_supervised_actor_backward = self._strict_environment_bool(
            "HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD",
            default=False,
        )
        supervised_actor_microbatch_size = self._strict_environment_nonnegative_int(
            "HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH",
            default=0,
        )
        # Freeze semantics are resolved once at setup.  Re-reading ambient
        # environment state inside every minibatch could change the objective
        # after validation and provenance capture.
        self._supervised_dagger_only = supervised_only
        self._supervised_actor_only_step = actor_only_step
        self._configured_supervised_actor_stream_backward = stream_supervised_actor_backward
        self._configured_supervised_actor_microbatch_size = supervised_actor_microbatch_size
        self._supervised_actor_stream_backward = bool(
            supervised_only and stream_supervised_actor_backward
        )
        self._supervised_actor_microbatch_size = (
            supervised_actor_microbatch_size if supervised_only else 0
        )
        if actor_only_step != supervised_only:
            if actor_only_step:
                raise ValueError(
                    "HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP is only valid when "
                    "HOLOSOMA_DAGGER_SUPERVISED_ONLY=1; standalone actor-only optimization would "
                    "silently freeze the critic in a different objective."
                )
            raise ValueError(
                "HOLOSOMA_DAGGER_SUPERVISED_ONLY=1 requires "
                "HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP=1: pure supervised action-BC has no "
                "critic objective, so recording an ordinary actor/critic step would misstate the "
                "effective optimizer and resume contract."
            )
        if (
            not supervised_only
            and (stream_supervised_actor_backward or supervised_actor_microbatch_size > 0)
        ):
            logger.warning(
                "Supervised-only actor microbatch settings were requested while "
                "HOLOSOMA_DAGGER_SUPERVISED_ONLY is disabled; preserving the requested values for "
                "runtime provenance but canonicalizing the effective microbatch/stream settings to "
                "0/false."
            )
        if (
            supervised_only
            and stream_supervised_actor_backward
            and supervised_actor_microbatch_size <= 0
        ):
            raise ValueError(
                "HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD=1 requires a positive "
                "HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH; without microbatching the stream option "
                "has no effect."
            )
        if (
            supervised_only
            and self._actor_uses_flow_matching()
            and (
                stream_supervised_actor_backward
                or supervised_actor_microbatch_size > 0
            )
        ):
            raise ValueError(
                "Supervised-only Flow actor microbatch/stream-backward training is not implemented. "
                "The Flow loss currently samples and evaluates one full actor batch, so accepting "
                "HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH>0 or "
                "HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD=1 would silently ignore the requested "
                "memory/backward contract. Use the default whole-batch Flow path (microbatch=0, "
                "stream-backward=0), or implement an explicitly equivalent chunked Flow objective."
            )
        if supervised_only:
            conflicts = []
            if not self.dagger_enabled:
                conflicts.append("DAgger is not enabled")
            if self.use_symmetry:
                conflicts.append("use_symmetry (not implemented by the supervised-only path)")
            if self.use_ppo_dagger_schedule and self._ppo_can_contribute():
                conflicts.append(
                    "ppo_start_epoch/dagger_end_epoch schedule with an operational PPO contribution"
                )
            if self.switch_to_rl_after > 0:
                conflicts.append("switch_to_rl_after")
            if self._configured_bc_loss_coef < 1.0:
                conflicts.append(f"bc_loss_coef={self._configured_bc_loss_coef}")
            if self.dagger_match_std:
                conflicts.append("dagger_match_std (not implemented by the supervised-only path)")
            if conflicts:
                raise ValueError(
                    "HOLOSOMA_DAGGER_SUPERVISED_ONLY requires pure action-BC training; conflicting settings: "
                    + ", ".join(conflicts)
                    + "."
                )

        objective_config = getattr(self, "config", None)
        critic_coefficients = {
            "value_loss_coef": getattr(objective_config, "value_loss_coef", 1.0),
            "symmetry_critic_coef": getattr(objective_config, "symmetry_critic_coef", 0.0),
        }
        for coefficient_name, coefficient in critic_coefficients.items():
            if (
                isinstance(coefficient, bool)
                or not isinstance(coefficient, numbers.Real)
                or not math.isfinite(float(coefficient))
                or float(coefficient) < 0.0
            ):
                raise ValueError(
                    f"PPO {coefficient_name} must be finite and non-negative, got {coefficient!r}."
                )
        if not self.use_symmetry and float(critic_coefficients["symmetry_critic_coef"]) != 0.0:
            raise ValueError(
                "PPO symmetry_critic_coef is non-zero while use_symmetry=False; the configured critic "
                "objective would be silently ignored."
            )

        ppo_can_contribute = self._ppo_can_contribute()
        if ppo_can_contribute and float(critic_coefficients["value_loss_coef"]) <= 0.0:
            raise ValueError(
                "PPO can contribute to the actor objective, so value_loss_coef must be > 0: GAE and "
                "timeout bootstrapping require a critic trained against returns. A frozen random/stale "
                "critic is not a scientifically valid PPO baseline."
            )
        safe_frozen_batch_norm_ids = {
            id(module)
            for encoder in self.actor.modules()
            if isinstance(encoder, DeFMEncoder)
            and encoder.freeze_backbone
            and encoder.backbone is not None
            for module in encoder.backbone.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
            and module.track_running_stats
            and module.running_mean is not None
            and module.running_var is not None
        }
        unsafe_batch_norm_names = [
            name or "<root>"
            for name, module in self.actor.named_modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
            and id(module) not in safe_frozen_batch_norm_ids
        ]
        if unsafe_batch_norm_names and (
            ppo_can_contribute or getattr(self, "is_multi_gpu", False)
        ):
            preview = ", ".join(unsafe_batch_norm_names[:8])
            suffix = (
                ""
                if len(unsafe_batch_norm_names) <= 8
                else f", ... ({len(unsafe_batch_norm_names)} total)"
            )
            if ppo_can_contribute:
                reason = (
                    "rollout batches and update minibatches use different batch statistics, and "
                    "running-stat mutation changes the policy outside optimizer steps"
                )
            else:
                reason = (
                    "custom multi-GPU gradient reduction synchronizes parameters but not BatchNorm "
                    "running buffers, so the published rank-zero student would represent only one shard"
                )
            raise ValueError(
                f"Actor BatchNorm is not supported for this training objective: {reason}. "
                f"Unsafe modules: {preview}{suffix}. Freeze the DeFM backbone in eval mode or use a "
                "normalization layer without running batch statistics."
            )
        if not ppo_can_contribute:
            return
        actor_impl = getattr(getattr(self.actor, "actor_module", None), "module", None)
        inference_noise_std = float(getattr(actor_impl, "inference_noise_std", 0.0) or 0.0)
        if self._actor_uses_flow_matching() and inference_noise_std > 0.0:
            raise ValueError(
                "flow_inference_noise_std must be 0 when PPO contributes: stochastic policy means make "
                "the stored/recomputed likelihood ratio invalid."
            )
        dropout_modules = [module for module in self.actor.modules() if isinstance(module, nn.modules.dropout._DropoutNd)]
        active_dropout = [module for module in dropout_modules if float(module.p) > 0.0]
        if active_dropout:
            raise ValueError(
                "Actor dropout must be disabled when PPO contributes: rollout and update likelihoods cannot "
                "be recomputed from the same policy realization."
            )

    def _build_teacher_actor_config(self, obs_keys: list[str], base_actor_cfg: ModuleConfig | None = None):
        actor_cfg = base_actor_cfg or self.config.module_dict.actor
        if list(actor_cfg.input_dim) == list(obs_keys):
            return actor_cfg
        layer_cfg = actor_cfg.layer_config
        excluded_inputs = set()
        if layer_cfg.encoder_input_name:
            excluded_inputs.add(layer_cfg.encoder_input_name)
        if layer_cfg.encoder_obs_token_name:
            excluded_inputs.add(layer_cfg.encoder_obs_token_name)
        if layer_cfg.perception_input_name:
            excluded_inputs.add(layer_cfg.perception_input_name)
        module_inputs = tuple(name for name in obs_keys if name not in excluded_inputs)
        layer_cfg = dataclasses.replace(layer_cfg, module_input_name=module_inputs)
        if layer_cfg.encoder_input_name and layer_cfg.encoder_input_name not in obs_keys:
            layer_cfg = dataclasses.replace(layer_cfg, encoder_input_name="")
        if layer_cfg.encoder_obs_token_name and layer_cfg.encoder_obs_token_name not in obs_keys:
            layer_cfg = dataclasses.replace(layer_cfg, encoder_obs_token_name=None)
        removed_perception_input = False
        if layer_cfg.perception_input_name:
            resolved_perception_key = ""
            if self.teacher_perception_obs_key:
                resolved_perception_key = self.teacher_perception_obs_key
            elif layer_cfg.perception_input_name in self.algo_obs_dim_dict:
                resolved_perception_key = layer_cfg.perception_input_name
            if resolved_perception_key:
                layer_cfg = dataclasses.replace(layer_cfg, perception_input_name=resolved_perception_key)
            else:
                layer_cfg = dataclasses.replace(layer_cfg, perception_input_name="")
                removed_perception_input = True

        actor_type = actor_cfg.type
        # In strict mode we do not auto-fallback teacher architecture on obs mismatch.
        if self.strict_teacher_load and actor_type == "MLPPerceptionEncoder" and not layer_cfg.perception_input_name:
            raise ValueError(
                "Teacher checkpoint expects perception input, but current teacher_obs_keys remove it. "
                "Set matching teacher_obs_keys (e.g. legacy group) or disable strict_teacher_load explicitly."
            )
        # Backward-compatible fallback for non-strict mode only.
        if (not self.strict_teacher_load) and actor_type == "MLPPerceptionEncoder" and not layer_cfg.perception_input_name:
            actor_type = "MLP"
        if removed_perception_input and layer_cfg.extra_input_to_hidden:
            layer_cfg = dataclasses.replace(layer_cfg, extra_input_to_hidden=False)

        return dataclasses.replace(actor_cfg, type=actor_type, input_dim=list(obs_keys), layer_config=layer_cfg)

    def _extract_teacher_actor_config(self, teacher_state: dict) -> ModuleConfig | None:
        exp_cfg = teacher_state.get("experiment_config")
        if not isinstance(exp_cfg, dict):
            return None
        try:
            actor_cfg_raw = exp_cfg["algo"]["config"]["module_dict"]["actor"]
        except (KeyError, TypeError):
            return None
        if not isinstance(actor_cfg_raw, dict):
            return None
        layer_cfg_raw = actor_cfg_raw.get("layer_config")
        if not isinstance(layer_cfg_raw, dict):
            return None
        layer_kwargs = dict(layer_cfg_raw)
        module_input_name = layer_kwargs.get("module_input_name")
        if isinstance(module_input_name, list):
            layer_kwargs["module_input_name"] = tuple(module_input_name)
        try:
            layer_cfg = LayerConfig(**layer_kwargs)
            actor_cfg = ModuleConfig(
                type=str(actor_cfg_raw.get("type", "MLP")),
                input_dim=list(actor_cfg_raw.get("input_dim", [])),
                output_dim=list(actor_cfg_raw.get("output_dim", [])),
                layer_config=layer_cfg,
                min_noise_std=actor_cfg_raw.get("min_noise_std"),
                min_mean_noise_std=actor_cfg_raw.get("min_mean_noise_std"),
                max_noise_std=actor_cfg_raw.get("max_noise_std"),
            )
            return actor_cfg
        except Exception as exc:
            if getattr(self, "strict_teacher_load", False):
                raise ValueError(
                    "Teacher checkpoint actor configuration is invalid under strict_teacher_load."
                ) from exc
            logger.warning(
                "Failed to parse teacher actor config from checkpoint; falling back to runtime config. {}",
                exc,
            )
            return None

    def _teacher_normalization_config(self, teacher_state: dict) -> tuple[bool, float, int | None]:
        """Resolve actor-observation normalization from the teacher checkpoint.

        Older checkpoints may not contain serialized experiment metadata.  In
        that case the runtime setting remains the only backward-compatible
        fallback, but checkpoints that do carry metadata must be interpreted
        according to the policy that produced them rather than the student.
        """
        strict = bool(getattr(self, "strict_teacher_load", False))
        enabled = bool(getattr(self.config, "normalize_actor_obs", False))
        eps = float(getattr(self.config, "obs_normalizer_eps", 1e-2))
        until = getattr(self.config, "obs_normalizer_until", None)
        exp_cfg = teacher_state.get("experiment_config")
        algo_cfg = None
        if isinstance(exp_cfg, dict):
            try:
                algo_cfg = exp_cfg["algo"]["config"]
            except (KeyError, TypeError):
                algo_cfg = None
            if isinstance(algo_cfg, dict):
                value = algo_cfg.get("normalize_actor_obs")
                if strict and not isinstance(value, bool):
                    raise ValueError(
                        "Strict teacher checkpoint must declare a boolean normalize_actor_obs contract."
                    )
                if isinstance(value, bool):
                    enabled = value
                raw_eps = algo_cfg.get("obs_normalizer_eps")
                if strict and enabled and raw_eps is None:
                    raise ValueError(
                        "Strict normalized teacher checkpoint must declare obs_normalizer_eps."
                    )
                if raw_eps is not None:
                    eps = float(raw_eps)
                if "obs_normalizer_until" in algo_cfg:
                    raw_until = algo_cfg["obs_normalizer_until"]
                    if raw_until is None:
                        until = None
                    elif type(raw_until) is not int or raw_until < 0:
                        raise ValueError(
                            "Teacher checkpoint obs_normalizer_until must be a non-negative "
                            f"integer or None, got {raw_until!r}."
                        )
                    else:
                        until = raw_until
        if strict and not isinstance(algo_cfg, dict):
            raise ValueError(
                "Strict teacher checkpoint is missing algo.config normalization metadata."
            )
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"Teacher obs_normalizer_eps must be finite and positive, got {eps}.")
        if until is not None and (type(until) is not int or until < 0):
            raise ValueError(
                f"Teacher obs_normalizer_until must be a non-negative integer or None, got {until!r}."
            )
        return enabled, eps, until

    def _teacher_normalization_enabled(self, teacher_state: dict) -> bool:
        """Backward-compatible convenience wrapper used by tests/callers."""
        return self._teacher_normalization_config(teacher_state)[0]

    def _load_teacher_normalizer_states(
        self,
        teacher_normalizers: dict[str, nn.Module],
        actor_norm_state: dict,
        checkpoint_obs_keys: list[str] | None,
        *,
        require_state: bool = False,
    ) -> None:
        """Load normalizers, including positional compatibility-group aliases."""
        runtime_keys = list(teacher_normalizers)
        if checkpoint_obs_keys is not None:
            # Strict teacher compatibility maps checkpoint actor inputs to
            # runtime aliases by position.  Prefer that declared mapping even
            # when both sides happen to reuse the same key set in a different
            # order; same-name lookup would then attach statistics to the
            # wrong feature slice.
            source_keys = list(checkpoint_obs_keys)
            missing_source_keys = [key for key in source_keys if key not in actor_norm_state]
            if missing_source_keys:
                raise ValueError(
                    "Teacher checkpoint normalizer state is missing checkpoint actor input keys: "
                    f"{missing_source_keys}."
                )
            if len(source_keys) != len(runtime_keys):
                raise ValueError(
                    "Cannot map teacher checkpoint normalizers to runtime observation aliases: "
                    f"checkpoint_keys={source_keys}, runtime_keys={runtime_keys}."
                )
            key_pairs = list(zip(source_keys, runtime_keys))
        elif all(key in actor_norm_state for key in runtime_keys):
            key_pairs = [(key, key) for key in runtime_keys]
        else:
            source_keys = list(actor_norm_state)
            if len(source_keys) != len(runtime_keys):
                raise ValueError(
                    "Cannot map teacher checkpoint normalizers to runtime observation aliases without "
                    "checkpoint actor input order: "
                    f"checkpoint_keys={source_keys}, runtime_keys={runtime_keys}."
                )
            key_pairs = list(zip(source_keys, runtime_keys))

        prepared_states: list[tuple[str, str, nn.Module, dict[str, Any]]] = []
        for source_key, runtime_key in key_pairs:
            state = actor_norm_state[source_key]
            if state is None:
                if require_state:
                    raise ValueError(
                        "Teacher checkpoint enabled observation normalization but contains an empty state for "
                        f"checkpoint_key={source_key}, runtime_key={runtime_key}."
                    )
                continue
            if not isinstance(state, dict):
                raise ValueError(
                    "Teacher observation normalizer state must be a mapping: "
                    f"checkpoint_key={source_key}, runtime_key={runtime_key}."
                )
            normalizer = teacher_normalizers[runtime_key]
            self._validate_normalizer_module_state(
                normalizer,
                state,
                path=f"teacher.actor_obs_normalizer_state[{source_key!r}]",
            )
            prepared_states.append((source_key, runtime_key, normalizer, state))

        # Apply only after every alias/state has passed validation, so one bad
        # later group cannot leave earlier teacher normalizers partially loaded.
        for source_key, runtime_key, normalizer, state in prepared_states:
            try:
                normalizer.load_state_dict(state, strict=True)
            except RuntimeError as exc:
                raise ValueError(
                    "Teacher observation normalizer is incompatible with its runtime alias: "
                    f"checkpoint_key={source_key}, runtime_key={runtime_key}."
                ) from exc

    def _validate_teacher_checkpoint_runtime_config(
        self,
        teacher_state: dict,
        *,
        obs_keys: list[str],
        teacher_actor_cfg: ModuleConfig,
    ) -> None:
        if not self.strict_teacher_load:
            return

        exp_cfg = teacher_state.get("experiment_config")
        if not isinstance(exp_cfg, dict):
            raise ValueError(
                "Teacher checkpoint has no verifiable experiment_config under strict_teacher_load."
            )

        checkpoint_observation_cfg = exp_cfg.get("observation")
        checkpoint_groups = (
            checkpoint_observation_cfg.get("groups", {})
            if isinstance(checkpoint_observation_cfg, Mapping)
            else {}
        )
        runtime_observation_cfg = getattr(getattr(self.env, "observation_manager", None), "cfg", None)
        runtime_groups = (
            runtime_observation_cfg.get("groups", {})
            if isinstance(runtime_observation_cfg, Mapping)
            else getattr(runtime_observation_cfg, "groups", {})
            if runtime_observation_cfg is not None
            else {}
        )

        try:
            checkpoint_actor_cfg = exp_cfg["algo"]["config"]["module_dict"]["actor"]
        except (KeyError, TypeError):
            checkpoint_actor_cfg = {}
        checkpoint_actor_inputs = (
            checkpoint_actor_cfg.get("input_dim") if isinstance(checkpoint_actor_cfg, dict) else None
        )
        if isinstance(checkpoint_actor_inputs, tuple):
            checkpoint_actor_inputs = list(checkpoint_actor_inputs)
        if checkpoint_actor_inputs:
            if len(checkpoint_actor_inputs) != len(obs_keys):
                raise ValueError(
                    "Teacher checkpoint/runtime actor input group count mismatch under strict_teacher_load: "
                    f"checkpoint={checkpoint_actor_inputs}, runtime={obs_keys}."
                )
            group_pairs = list(zip(checkpoint_actor_inputs, obs_keys))
        else:
            group_pairs = [(key, key) for key in obs_keys]

        def canonical(value):
            if dataclasses.is_dataclass(value):
                value = dataclasses.asdict(value)
            if isinstance(value, Mapping):
                return {str(key): canonical(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [canonical(item) for item in value]
            if torch.is_tensor(value):
                return canonical(value.detach().cpu().tolist())
            return value

        perception_term_suffix_to_role = {
            ":perception_obs": "student",
            ":teacher_perception_obs": "teacher",
            ":critic_perception_obs": "critic",
        }

        def perception_term_role(func: Any) -> str | None:
            normalized = str(func or "").strip()
            for suffix, role in perception_term_suffix_to_role.items():
                if normalized.endswith(suffix):
                    return role
            return None

        def group_perception_role(group: Any) -> str | None:
            terms = group.get("terms", {}) if isinstance(group, Mapping) else getattr(group, "terms", {})
            roles: list[str] = []
            if isinstance(terms, Mapping):
                for term in terms.values():
                    func = term.get("func") if isinstance(term, Mapping) else getattr(term, "func", None)
                    role = perception_term_role(func)
                    if role is not None:
                        roles.append(role)
            if len(roles) > 1:
                raise ValueError(
                    "Teacher runtime observation group contains multiple perception-manager terms."
                )
            return roles[0] if roles else None

        mismatches: list[str] = []
        checkpoint_clip_observations = (
            checkpoint_observation_cfg.get("clip_observations")
            if isinstance(checkpoint_observation_cfg, Mapping)
            else None
        )
        runtime_clip_observations = (
            runtime_observation_cfg.get("clip_observations")
            if isinstance(runtime_observation_cfg, Mapping)
            else getattr(runtime_observation_cfg, "clip_observations", None)
        )
        for source, value in (
            ("checkpoint", checkpoint_clip_observations),
            ("runtime", runtime_clip_observations),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                mismatches.append(
                    f"teacher {source} observation.clip_observations must be finite and positive, "
                    f"got {value!r}"
                )
        if (
            isinstance(checkpoint_clip_observations, numbers.Real)
            and not isinstance(checkpoint_clip_observations, bool)
            and isinstance(runtime_clip_observations, numbers.Real)
            and not isinstance(runtime_clip_observations, bool)
            and float(checkpoint_clip_observations) != float(runtime_clip_observations)
        ):
            mismatches.append(
                "teacher observation.clip_observations mismatch: "
                f"checkpoint={checkpoint_clip_observations!r} runtime={runtime_clip_observations!r}"
            )
        matched_groups: list[str] = []
        for checkpoint_key, runtime_key in group_pairs:
            checkpoint_group = checkpoint_groups.get(checkpoint_key) if isinstance(checkpoint_groups, dict) else None
            if not isinstance(checkpoint_group, dict):
                mismatches.append(f"{checkpoint_key}->{runtime_key}: missing checkpoint observation group metadata")
                continue

            runtime_group = runtime_groups.get(runtime_key) if isinstance(runtime_groups, dict) else None
            if runtime_group is None:
                mismatches.append(f"{checkpoint_key}->{runtime_key}: missing runtime observation group")
                continue

            checkpoint_history = checkpoint_group.get("history_length")
            runtime_history = getattr(runtime_group, "history_length", None)
            if checkpoint_history != runtime_history:
                mismatches.append(
                    f"{checkpoint_key}->{runtime_key}: history_length "
                    f"checkpoint={checkpoint_history} runtime={runtime_history}"
                )
            for group_field in ("concatenate", "enable_noise"):
                checkpoint_value = checkpoint_group.get(group_field)
                runtime_value = getattr(runtime_group, group_field, None)
                if canonical(checkpoint_value) != canonical(runtime_value):
                    mismatches.append(
                        f"{checkpoint_key}->{runtime_key}.{group_field}: "
                        f"checkpoint={checkpoint_value!r} runtime={runtime_value!r}"
                    )

            checkpoint_terms = checkpoint_group.get("terms", {})
            checkpoint_term_names = list(checkpoint_terms.keys()) if isinstance(checkpoint_terms, dict) else []
            runtime_terms = getattr(runtime_group, "terms", {})
            runtime_term_names = list(runtime_terms.keys()) if isinstance(runtime_terms, dict) else []
            if checkpoint_term_names != runtime_term_names:
                mismatches.append(
                    f"{checkpoint_key}->{runtime_key}: terms "
                    f"checkpoint={checkpoint_term_names} runtime={runtime_term_names}"
                )
            elif isinstance(checkpoint_terms, dict) and isinstance(runtime_terms, dict):
                for term_name in checkpoint_term_names:
                    checkpoint_term = checkpoint_terms[term_name]
                    runtime_term = runtime_terms[term_name]
                    for field_name in ("func", "params", "scale", "noise", "clip"):
                        checkpoint_value = (
                            checkpoint_term.get(field_name)
                            if isinstance(checkpoint_term, dict)
                            else getattr(checkpoint_term, field_name, None)
                        )
                        runtime_value = getattr(runtime_term, field_name, None)
                        if (
                            field_name == "func"
                            and perception_term_role(checkpoint_value) is not None
                            and perception_term_role(runtime_value) is not None
                        ):
                            # A teacher input may intentionally remap the
                            # checkpoint's student perception group onto a
                            # teacher- or critic-specific manager.  The full
                            # manager/encoder contract is checked below.
                            continue
                        if canonical(checkpoint_value) != canonical(runtime_value):
                            mismatches.append(
                                f"{checkpoint_key}->{runtime_key}.{term_name}.{field_name}: "
                                f"checkpoint={checkpoint_value!r} runtime={runtime_value!r}"
                            )
            matched_groups.append(f"{checkpoint_key}->{runtime_key}")

        checkpoint_layer_cfg = (
            checkpoint_actor_cfg.get("layer_config", {}) if isinstance(checkpoint_actor_cfg, dict) else {}
        )
        checkpoint_perception_key = ""
        if isinstance(checkpoint_layer_cfg, dict):
            checkpoint_perception_key = str(checkpoint_layer_cfg.get("perception_input_name", "") or "")
        runtime_perception_key = str(teacher_actor_cfg.layer_config.perception_input_name or "")
        if bool(checkpoint_perception_key) != bool(runtime_perception_key):
            mismatches.append(
                "teacher perception input presence mismatch: "
                f"checkpoint={checkpoint_perception_key or '<none>'} "
                f"runtime={runtime_perception_key or '<none>'}"
            )
        elif runtime_perception_key:
            perception_actor_targets = [
                runtime_key
                for checkpoint_key, runtime_key in group_pairs
                if checkpoint_key == checkpoint_perception_key
            ]
            if perception_actor_targets and perception_actor_targets != [runtime_perception_key]:
                mismatches.append(
                    "teacher perception input conflicts with its positional checkpoint/runtime group mapping: "
                    f"checkpoint={checkpoint_perception_key!r}, runtime={runtime_perception_key!r}, "
                    f"actor_targets={perception_actor_targets!r}"
                )
            checkpoint_perception_group = (
                checkpoint_groups.get(checkpoint_perception_key)
                if isinstance(checkpoint_groups, Mapping)
                else None
            )
            runtime_perception_group = (
                runtime_groups.get(runtime_perception_key)
                if isinstance(runtime_groups, Mapping)
                else None
            )
            # Perception is a policy side input and is intentionally absent
            # from ModuleConfig.input_dim.  Audit its observation group
            # independently instead of requiring it in the positional actor
            # input mapping above.
            if (checkpoint_perception_key, runtime_perception_key) not in group_pairs:
                if not isinstance(checkpoint_perception_group, Mapping):
                    mismatches.append(
                        f"{checkpoint_perception_key}->{runtime_perception_key}: "
                        "missing checkpoint perception observation group metadata"
                    )
                elif runtime_perception_group is None:
                    mismatches.append(
                        f"{checkpoint_perception_key}->{runtime_perception_key}: "
                        "missing runtime perception observation group"
                    )
                else:
                    runtime_group_value = lambda name: (  # noqa: E731 - local config accessor
                        runtime_perception_group.get(name)
                        if isinstance(runtime_perception_group, Mapping)
                        else getattr(runtime_perception_group, name, None)
                    )
                    checkpoint_history = checkpoint_perception_group.get("history_length")
                    runtime_history = runtime_group_value("history_length")
                    if checkpoint_history != runtime_history:
                        mismatches.append(
                            f"{checkpoint_perception_key}->{runtime_perception_key}: history_length "
                            f"checkpoint={checkpoint_history} runtime={runtime_history}"
                        )
                    for group_field in ("concatenate", "enable_noise"):
                        checkpoint_value = checkpoint_perception_group.get(group_field)
                        runtime_value = runtime_group_value(group_field)
                        if canonical(checkpoint_value) != canonical(runtime_value):
                            mismatches.append(
                                f"{checkpoint_perception_key}->{runtime_perception_key}.{group_field}: "
                                f"checkpoint={checkpoint_value!r} runtime={runtime_value!r}"
                            )
                    checkpoint_terms = checkpoint_perception_group.get("terms", {})
                    runtime_terms = runtime_group_value("terms")
                    checkpoint_term_names = (
                        list(checkpoint_terms) if isinstance(checkpoint_terms, Mapping) else []
                    )
                    runtime_term_names = list(runtime_terms) if isinstance(runtime_terms, Mapping) else []
                    if checkpoint_term_names != runtime_term_names:
                        mismatches.append(
                            f"{checkpoint_perception_key}->{runtime_perception_key}: terms "
                            f"checkpoint={checkpoint_term_names} runtime={runtime_term_names}"
                        )
                    elif isinstance(checkpoint_terms, Mapping) and isinstance(runtime_terms, Mapping):
                        for term_name in checkpoint_term_names:
                            checkpoint_term = checkpoint_terms[term_name]
                            runtime_term = runtime_terms[term_name]
                            for field_name in ("func", "params", "scale", "noise", "clip"):
                                checkpoint_value = (
                                    checkpoint_term.get(field_name)
                                    if isinstance(checkpoint_term, Mapping)
                                    else getattr(checkpoint_term, field_name, None)
                                )
                                runtime_value = (
                                    runtime_term.get(field_name)
                                    if isinstance(runtime_term, Mapping)
                                    else getattr(runtime_term, field_name, None)
                                )
                                if (
                                    field_name == "func"
                                    and perception_term_role(checkpoint_value) is not None
                                    and perception_term_role(runtime_value) is not None
                                ):
                                    continue
                                if canonical(checkpoint_value) != canonical(runtime_value):
                                    mismatches.append(
                                        f"{checkpoint_perception_key}->{runtime_perception_key}."
                                        f"{term_name}.{field_name}: checkpoint={checkpoint_value!r} "
                                        f"runtime={runtime_value!r}"
                                    )
                    matched_groups.append(
                        f"{checkpoint_perception_key}->{runtime_perception_key}"
                    )
            try:
                checkpoint_manager_role = group_perception_role(checkpoint_perception_group)
                runtime_manager_role = group_perception_role(runtime_perception_group)
            except ValueError as exc:
                mismatches.append(str(exc))
                checkpoint_manager_role = None
                runtime_manager_role = None
            if checkpoint_manager_role is None or runtime_manager_role is None:
                mismatches.append(
                    "teacher perception input must resolve to one audited perception manager term: "
                    f"checkpoint_role={checkpoint_manager_role!r}, runtime_role={runtime_manager_role!r}"
                )
            else:
                manager_attr = {
                    "student": "perception_manager",
                    "teacher": "teacher_perception_manager",
                    "critic": "critic_perception_manager",
                }[runtime_manager_role]
                runtime_manager = getattr(self.env, manager_attr, None)
                runtime_perception_cfg = getattr(runtime_manager, "cfg", None)
                if runtime_perception_cfg is None:
                    mismatches.append(
                        f"runtime {runtime_manager_role} perception group has no active {manager_attr}."
                    )
                else:
                    output_mode = str(getattr(runtime_perception_cfg, "output_mode", "") or "")
                    if output_mode == "camera_depth":
                        input_height = int(
                            getattr(runtime_perception_cfg, "camera_height", None)
                            or getattr(runtime_perception_cfg, "grid_size", 0)
                        )
                        input_width = int(
                            getattr(runtime_perception_cfg, "camera_width", None)
                            or getattr(runtime_perception_cfg, "grid_size", 0)
                        )
                        if bool(getattr(runtime_perception_cfg, "camera_warp_preprocess", False)):
                            resize = getattr(runtime_perception_cfg, "camera_warp_resize", None)
                            if resize is not None:
                                input_height, input_width = (int(resize[0]), int(resize[1]))
                            else:
                                input_height = max(
                                    1,
                                    input_height
                                    - int(getattr(runtime_perception_cfg, "camera_warp_crop_top", 0))
                                    - int(getattr(runtime_perception_cfg, "camera_warp_crop_bottom", 0)),
                                )
                                input_width = max(
                                    1,
                                    input_width
                                    - int(getattr(runtime_perception_cfg, "camera_warp_crop_left", 0))
                                    - int(getattr(runtime_perception_cfg, "camera_warp_crop_right", 0)),
                                )
                    elif output_mode == "heightmap":
                        input_height = input_width = int(getattr(runtime_perception_cfg, "grid_size", 0))
                    else:
                        input_height = input_width = None

                    layer_manager_fields = (
                        ("perception_encoder_type", "encoder_type"),
                        ("perception_output_dim", "encoder_output_dim"),
                        ("perception_freeze_backbone", "encoder_freeze_backbone"),
                        ("perception_target_size", "encoder_target_size"),
                        ("perception_patch_size", "encoder_patch_size"),
                    )
                    for layer_field, manager_field in layer_manager_fields:
                        checkpoint_value = checkpoint_layer_cfg.get(layer_field)
                        runtime_value = getattr(runtime_perception_cfg, manager_field, None)
                        if canonical(checkpoint_value) != canonical(runtime_value):
                            mismatches.append(
                                f"teacher {layer_field}/{manager_field} mismatch: "
                                f"checkpoint={checkpoint_value!r} runtime={runtime_value!r}"
                            )
                    for layer_field, runtime_value in (
                        ("perception_input_height", input_height),
                        ("perception_input_width", input_width),
                    ):
                        checkpoint_value = checkpoint_layer_cfg.get(layer_field)
                        if canonical(checkpoint_value) != canonical(runtime_value):
                            mismatches.append(
                                f"teacher {layer_field} mismatch: "
                                f"checkpoint={checkpoint_value!r} runtime={runtime_value!r}"
                            )

                    runtime_encoder_type = str(
                        getattr(runtime_perception_cfg, "encoder_type", "") or ""
                    ).strip().lower()
                    if runtime_encoder_type.startswith("defm_") and (
                        output_mode != "camera_depth"
                        or bool(getattr(runtime_perception_cfg, "camera_warp_normalize", False))
                    ):
                        mismatches.append(
                            "teacher DeFM perception requires camera_depth metric meters with "
                            "camera_warp_normalize=False."
                        )

                    checkpoint_perception_cfg = (
                        exp_cfg.get("perception")
                        if checkpoint_manager_role == "student"
                        else None
                    )
                    if not isinstance(checkpoint_perception_cfg, Mapping):
                        mismatches.append(
                            "teacher checkpoint does not embed the resolved perception-manager config "
                            f"for checkpoint role {checkpoint_manager_role!r}."
                        )
                    else:
                        runtime_manager_semantics = (
                            dataclasses.asdict(runtime_perception_cfg)
                            if dataclasses.is_dataclass(runtime_perception_cfg)
                            else dict(vars(runtime_perception_cfg))
                            if hasattr(runtime_perception_cfg, "__dict__")
                            else runtime_perception_cfg
                        )
                        ignored_manager_fields = {
                            "inject_into_policy_modules",
                            "inject_into_critic_modules",
                        }
                        checkpoint_sensor_semantics = {
                            key: value
                            for key, value in checkpoint_perception_cfg.items()
                            if key not in ignored_manager_fields and not str(key).startswith("encoder_")
                        }
                        runtime_sensor_semantics = {
                            key: value
                            for key, value in runtime_manager_semantics.items()
                            if key not in ignored_manager_fields and not str(key).startswith("encoder_")
                        }
                        if canonical(checkpoint_sensor_semantics) != canonical(runtime_sensor_semantics):
                            mismatches.append(
                                "teacher checkpoint/runtime perception-manager sensor semantics differ."
                            )

        checkpoint_robot = exp_cfg.get("robot")
        runtime_robot = getattr(self.env, "robot_config", None)
        if not isinstance(checkpoint_robot, dict) or runtime_robot is None:
            mismatches.append("teacher robot/action contract metadata is missing")
        else:
            for field_name in (
                "actions_dim",
                "dof_names",
                "dof_effort_limit_list",
                "init_state",
                "control",
            ):
                checkpoint_value = checkpoint_robot.get(field_name)
                runtime_value = (
                    runtime_robot.get(field_name)
                    if isinstance(runtime_robot, dict)
                    else getattr(runtime_robot, field_name, None)
                )
                if canonical(checkpoint_value) != canonical(runtime_value):
                    mismatches.append(
                        f"teacher robot.{field_name}: "
                        f"checkpoint={checkpoint_value!r} runtime={runtime_value!r}"
                    )

        checkpoint_action = exp_cfg.get("action")
        runtime_action_cfg = getattr(getattr(self.env, "action_manager", None), "cfg", None)
        checkpoint_action_terms = checkpoint_action.get("terms") if isinstance(checkpoint_action, dict) else None
        runtime_action_terms = (
            runtime_action_cfg.get("terms")
            if isinstance(runtime_action_cfg, dict)
            else getattr(runtime_action_cfg, "terms", None)
        )
        if checkpoint_action_terms is None or runtime_action_terms is None:
            mismatches.append("teacher action.terms contract metadata is missing")
        elif canonical(checkpoint_action_terms) != canonical(runtime_action_terms):
            mismatches.append(
                "teacher action.terms: "
                f"checkpoint={checkpoint_action_terms!r} runtime={runtime_action_terms!r}"
            )

        if mismatches:
            details = "; ".join(mismatches)
            raise ValueError(
                "Teacher checkpoint/runtime observation config mismatch under strict_teacher_load. "
                f"{details}"
            )

        if matched_groups:
            logger.info(
                "Teacher checkpoint observation config matches runtime for groups: {}",
                ", ".join(matched_groups),
            )

    def _load_teacher_actor(
        self, ckpt_path: str, obs_keys: list[str] | None = None
    ) -> tuple[nn.Module, dict[str, nn.Module]]:
        """Build one teacher atomically without consuming training RNG.

        Teacher resolution, safe deserialization, module initialization, and
        the distributed outcome envelope are setup side effects.  Restoring
        the pre-load state makes a local path and an authenticated remote URI
        with identical bytes stochastic-equivalent for subsequent training.
        """

        rng_state: dict[str, Any] | None = None
        result: tuple[nn.Module, dict[str, nn.Module]] | None = None
        local_error: Exception | None = None
        try:
            rng_state = capture_rng_checkpoint_state()
            result = self._load_teacher_actor_impl(ckpt_path, obs_keys=obs_keys)
        except Exception as exc:
            local_error = exc
        try:
            self._synchronize_distributed_operation_error(
                local_error,
                operation="Teacher checkpoint resolution/construction",
            )
        finally:
            if rng_state is not None:
                restore_rng_checkpoint_state(
                    rng_state,
                    path="pre_teacher_checkpoint_load_rng_state",
                )
        assert result is not None
        return result

    def _load_teacher_actor_impl(
        self, ckpt_path: str, obs_keys: list[str] | None = None
    ) -> tuple[nn.Module, dict[str, nn.Module]]:
        debug_heartbeat = os.environ.get("HOLOSOMA_DEBUG_HEARTBEAT_VERBOSE", "").lower() not in (
            "",
            "0",
            "false",
            "no",
        )
        legacy_unverified_teacher_load = allow_legacy_unverified_teacher_load()
        attached_training_provenance = getattr(self, "_training_provenance", None)
        if attached_training_provenance is None:
            training_provenance = training_provenance_from_env()
        elif not isinstance(attached_training_provenance, dict):
            raise ValueError(
                "Attached teacher training provenance must be a mapping when present; "
                "the legacy override cannot bypass malformed provenance."
            )
        else:
            training_provenance = attached_training_provenance
        expected_teacher_sha256 = None
        if isinstance(training_provenance, dict):
            if (
                training_provenance.get("teacher_enabled") is True
                and not isinstance(training_provenance.get("teacher_sha256"), str)
            ):
                raise ValueError(
                    "Attached training provenance enables teacher loading but has no authenticated "
                    "teacher_sha256."
                )
            training_provenance = validate_training_provenance(
                training_provenance,
                require_finalized=True,
            )
            if training_provenance.get("teacher_enabled") is not True:
                raise ValueError(
                    "Teacher loading was requested while attached training provenance disables the teacher."
                )
            if getattr(self, "use_multi_teacher", False):
                raise ValueError(
                    "Scientific multi-teacher loading requires one authenticated digest per teacher; "
                    "the current provenance schema contains only teacher_sha256."
                )
            expected_teacher_sha256 = training_provenance.get("teacher_sha256")
            if not isinstance(expected_teacher_sha256, str):
                raise ValueError(
                    "Attached training provenance enables teacher loading but has no authenticated "
                    "teacher_sha256."
                )
        elif not legacy_unverified_teacher_load:
            raise ValueError(
                "Scientific teacher loading requires finalized current training provenance with "
                "an authenticated teacher_sha256. Set "
                f"{ALLOW_LEGACY_UNVERIFIED_TEACHER_LOAD_ENV}=1 only for an explicitly "
                "non-scientific legacy teacher load."
            )
        else:
            logger.warning(
                "{}=1: allowing teacher loading without authenticated current provenance.",
                ALLOW_LEGACY_UNVERIFIED_TEACHER_LOAD_ENV,
            )
        if ckpt_path.startswith("wandb://"):
            from holosoma.utils.eval_utils import load_checkpoint  # noqa: PLC0415

            teacher_cache_dir = self.log_dir / ".teacher_ckpt_cache" / f"rank_{self.gpu_global_rank}"
            ckpt_path = str(load_checkpoint(ckpt_path, str(teacher_cache_dir)))

        if debug_heartbeat:
            logger.info("Heartbeat: rank {} teacher torch.load begin {}", self.gpu_global_rank, ckpt_path)
        teacher_state, _ = load_verified_torch_checkpoint(
            ckpt_path,
            expected_sha256=expected_teacher_sha256,
            map_location="cpu",
        )
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} teacher torch.load finished", self.gpu_global_rank)
        if not isinstance(teacher_state, dict):
            raise ValueError("Teacher checkpoint payload must be a mapping.")
        teacher_obs_keys = obs_keys if obs_keys is not None else self.actor_obs_keys
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} teacher config build begin", self.gpu_global_rank)
        teacher_actor_base_cfg = self._extract_teacher_actor_config(teacher_state)
        if self.strict_teacher_load and teacher_actor_base_cfg is None:
            raise ValueError(
                "Teacher checkpoint actor configuration could not be parsed under strict_teacher_load; "
                "refusing to substitute the student runtime actor configuration."
            )
        teacher_actor_cfg = self._build_teacher_actor_config(teacher_obs_keys, base_actor_cfg=teacher_actor_base_cfg)
        self._validate_teacher_checkpoint_runtime_config(
            teacher_state,
            obs_keys=teacher_obs_keys,
            teacher_actor_cfg=teacher_actor_cfg,
        )
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} teacher actor module build begin", self.gpu_global_rank)
        teacher_actor = setup_ppo_actor_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=teacher_actor_cfg,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std,
            device=self.device,
            history_length=self.algo_history_length_dict,
        )
        # A DeFM teacher checkpoint contains backbone keys after its first
        # forward.  Construct the same schema before compatibility validation
        # and strict loading; the outer teacher-load boundary also preserves
        # the complete training RNG state.
        self._materialize_lazy_model_modules(
            teacher_actor,
            checkpoint_restore=True,
        )
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} teacher state_dict load begin", self.gpu_global_rank)
        teacher_actor_state = require_mapping(teacher_state, "actor_model_state_dict")
        validate_finite_tree(teacher_actor_state, path="teacher.actor_model_state_dict")
        reference_teacher_state = teacher_actor.state_dict()
        strict_compatible = True
        try:
            validate_module_state_compatibility(
                teacher_actor_state,
                reference_state=reference_teacher_state,
                path="teacher.actor_model_state_dict",
            )
        except (RuntimeError, ValueError) as compatibility_error:
            if self.strict_teacher_load:
                raise ValueError(
                    "Teacher checkpoint actor state is incompatible with the strict runtime actor."
                ) from compatibility_error
            allow_non_strict = False
            if hasattr(teacher_actor, "actor_module"):
                allow_non_strict = getattr(teacher_actor.actor_module.module, "supports_extra_input", False)
            if not allow_non_strict:
                raise ValueError(
                    "Teacher checkpoint actor state is incompatible and this actor does not support "
                    "the explicit extra-input non-strict compatibility path."
                ) from compatibility_error
            shared_keys = set(teacher_actor_state) & set(reference_teacher_state)
            if not shared_keys:
                raise ValueError(
                    "Teacher checkpoint has no parameters in common with the runtime actor."
                ) from compatibility_error
            validate_module_state_compatibility(
                {key: teacher_actor_state[key] for key in shared_keys},
                reference_state={key: reference_teacher_state[key] for key in shared_keys},
                path="teacher.actor_model_state_dict.shared",
            )
            strict_compatible = False
            logger.warning(
                "Teacher actor uses the explicit non-strict extra-input compatibility path; "
                "all shared tensors were validated before loading."
            )
        self._validate_checkpoint_actor_std(
            teacher_actor_state,
            path="teacher.actor_model_state_dict",
            actor=teacher_actor,
        )
        teacher_actor.load_state_dict(teacher_actor_state, strict=strict_compatible)
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} teacher state_dict load finished", self.gpu_global_rank)
        teacher_actor.eval()
        for param in teacher_actor.parameters():
            if isinstance(param, UninitializedParameter):
                continue
            param.requires_grad_(False)

        if debug_heartbeat:
            logger.info("Heartbeat: rank {} teacher normalizers build begin", self.gpu_global_rank)
        teacher_normalization_enabled, teacher_normalizer_eps, teacher_normalizer_until = (
            self._teacher_normalization_config(teacher_state)
        )
        teacher_normalizers = self._build_group_normalizers(
            teacher_obs_keys,
            teacher_normalization_enabled,
            eps=teacher_normalizer_eps,
            until=teacher_normalizer_until,
        )
        actor_norm_state = teacher_state.get("actor_obs_normalizer_state")
        if teacher_normalization_enabled and not isinstance(actor_norm_state, dict):
            raise ValueError(
                "Teacher checkpoint metadata enables normalize_actor_obs, but actor_obs_normalizer_state is "
                "missing. Refusing to run the teacher with uninitialized normalization statistics."
            )
        if isinstance(actor_norm_state, dict):
            checkpoint_obs_keys = (
                list(teacher_actor_base_cfg.input_dim)
                if teacher_actor_base_cfg is not None
                else None
            )
            self._load_teacher_normalizer_states(
                teacher_normalizers,
                actor_norm_state,
                checkpoint_obs_keys,
                require_state=teacher_normalization_enabled,
            )
        for normalizer in teacher_normalizers.values():
            normalizer.eval()
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} teacher load finished", self.gpu_global_rank)
        return teacher_actor, teacher_normalizers

    def _new_dagger_replay_buffer(self) -> _DaggerReplayBuffer:
        perception_dim = (
            int(self.algo_obs_dim_dict[self.actor_perception_key])
            if self.actor_perception_key
            else 0
        )
        return _DaggerReplayBuffer(
            capacity=self.dagger_replay_capacity,
            actor_obs_dim=self._get_obs_dim(self.actor_obs_keys),
            actor_perception_dim=perception_dim,
            action_dim=int(self.num_act),
            base_seed=self.dagger_replay_seed,
            rank=int(getattr(self, "gpu_global_rank", 0) or 0),
        )

    def _reset_dagger_replay_state(self) -> None:
        self._dagger_replay_buffer = (
            self._new_dagger_replay_buffer()
            if bool(getattr(self, "dagger_replay_enabled", False))
            else None
        )
        self._dagger_replay_fixed_eval_ready_at_rollout_start = False
        self._dagger_replay_fixed_boundary_verified_runtime = False

    def _setup_distillation(self) -> None:
        if getattr(self, "_evaluation_only", False):
            # A deployed student policy is fully defined by its actor and
            # actor normalizers.  Reconstructing the training teacher during
            # evaluation is both semantically irrelevant and can make a valid
            # student checkpoint depend on a stale node-local teacher path.
            self.distill_enabled = False
            self.dagger_enabled = False
            self.teacher_perception_obs_key = ""
            self.teacher_actor = None
            self.teacher_actors = []
            self.teacher_actor_obs_normalizers = {}
            self.teacher_actor_obs_normalizers_list = []
            return

        distill_cfg = self.config.distill
        self.distill_mode = str(getattr(distill_cfg, "mode", "mse")).strip().lower()
        if self.distill_mode not in ("mse", "dagger"):
            raise ValueError(f"distill.mode must be 'mse' or 'dagger', got {self.distill_mode!r}.")
        distill_config_enabled = self._strict_config_bool(
            "enabled",
            getattr(distill_cfg, "enabled", False),
        )
        if self.distill_mode == "dagger" and not distill_config_enabled:
            raise ValueError(
                "distill.mode='dagger' requires distill.enabled=True; mode selection must not silently "
                "override the documented distillation enable switch."
            )
        policy_to_clone = getattr(distill_cfg, "policy_to_clone", None)
        teacher_checkpoint_alias = getattr(distill_cfg, "teacher_checkpoint", None)
        if policy_to_clone and teacher_checkpoint_alias:
            aliases_match = (
                isinstance(policy_to_clone, str)
                and policy_to_clone == teacher_checkpoint_alias
            )
            if not aliases_match:
                raise ValueError(
                    "distill.policy_to_clone and distill.teacher_checkpoint identify different teacher "
                    "sources. Configure exactly one source (or the same single path in both aliases)."
                )
        self.distill_enabled = False
        self.dagger_enabled = False
        # ``setup`` is normally called once, but keep configuration parsing
        # transactional: a retry must not inherit a prior mix schedule.
        self.teacher_action_mix_ratio_start = None
        self.teacher_action_mix_ratio_end = None
        self.use_teacher_action_mix_schedule = False
        self.distill_loss_coef = float(distill_cfg.loss_coef)
        if not math.isfinite(self.distill_loss_coef) or self.distill_loss_coef < 0.0:
            raise ValueError(f"distill.loss_coef must be finite and non-negative, got {self.distill_loss_coef}.")
        operational_distill_loss_coef = self._operational_float32_loss_weight(
            self.distill_loss_coef
        )
        if (
            self.distill_mode == "mse"
            and distill_config_enabled
            and self.distill_loss_coef > 0.0
            and (
                not math.isfinite(operational_distill_loss_coef)
                or operational_distill_loss_coef <= 0.0
            )
        ):
            raise ValueError(
                "distill.loss_coef must remain finite and positive when materialized in the "
                f"float32 actor loss graph; got Python={self.distill_loss_coef}, "
                f"float32={operational_distill_loss_coef}."
            )
        if self.distill_mode == "mse":
            # Keep setup validation and the tensor graph on the identical
            # materialized coefficient, rather than retaining a Python scalar
            # whose float32 multiplication has different semantics.
            self.distill_loss_coef = operational_distill_loss_coef
        raw_bc_loss_coef = distill_cfg.bc_loss_coef
        if (
            self.distill_mode == "mse"
            and distill_config_enabled
            and raw_bc_loss_coef is not None
        ):
            raise ValueError(
                "distill.bc_loss_coef is DAgger-only and is not consumed by distill.mode='mse'. "
                "Configure the legacy MSE objective with distill.loss_coef only."
            )
        self.bc_loss_coef = (
            float(raw_bc_loss_coef)
            if raw_bc_loss_coef is not None
            else float(distill_cfg.loss_coef)
        )
        if not math.isfinite(self.bc_loss_coef):
            raise ValueError(f"distill.bc_loss_coef must be finite, got {self.bc_loss_coef}.")
        if self.distill_mode == "dagger" and not (0.0 <= self.bc_loss_coef <= 1.0):
            raise ValueError(f"distill.bc_loss_coef must be in [0.0, 1.0], got {self.bc_loss_coef}.")
        if (
            self.distill_mode == "dagger"
            and self.bc_loss_coef > 0.0
            and self._operational_float32_loss_weight(self.bc_loss_coef) <= 0.0
        ):
            raise ValueError(
                "distill.bc_loss_coef is positive as a Python scalar but rounds to zero in the "
                f"float32 actor loss graph: {self.bc_loss_coef}. Use a representable BC weight."
            )
        if (
            self.distill_mode == "dagger"
            and raw_bc_loss_coef is not None
            and self.distill_loss_coef != 1.0
            and self.distill_loss_coef != self.bc_loss_coef
        ):
            raise ValueError(
                "distill.loss_coef and distill.bc_loss_coef conflict in DAgger mode: bc_loss_coef "
                "takes precedence while the non-default loss_coef would be silently ignored. Configure "
                "bc_loss_coef only, or keep equal values for legacy compatibility."
            )
        self._configured_bc_loss_coef = self.bc_loss_coef
        self.clip_teacher_actions = self._strict_config_bool(
            "clip_teacher_actions",
            distill_cfg.clip_teacher_actions,
        )
        self.clip_actions_threshold = float(distill_cfg.clip_actions_threshold)
        if not math.isfinite(self.clip_actions_threshold) or self.clip_actions_threshold <= 0.0:
            raise ValueError(
                "distill.clip_actions_threshold must be finite and > 0.0, "
                f"got {self.clip_actions_threshold}."
            )
        if not self.clip_teacher_actions and self.clip_actions_threshold != 100.0:
            raise ValueError(
                "distill.clip_actions_threshold is only consumed when clip_teacher_actions=True; "
                "restore its default or enable clipping."
            )
        self.take_teacher_actions = self._strict_config_bool(
            "take_teacher_actions",
            distill_cfg.take_teacher_actions,
        )
        self.teacher_use_stochastic_actions = self._strict_config_bool(
            "teacher_use_stochastic_actions",
            getattr(distill_cfg, "teacher_use_stochastic_actions", False),
        )
        self.teacher_action_mix_ratio = float(getattr(distill_cfg, "teacher_action_mix_ratio", 0.0))
        if not math.isfinite(self.teacher_action_mix_ratio) or not (
            0.0 <= self.teacher_action_mix_ratio <= 1.0
        ):
            raise ValueError(
                f"distill.teacher_action_mix_ratio must be in [0.0, 1.0], got {self.teacher_action_mix_ratio}."
            )
        # Retain the unscheduled value: enabling a schedule below changes the
        # live ratio to its start value, but validation must still detect a
        # separately configured static ratio that would otherwise be ignored.
        self._configured_teacher_action_mix_ratio = self.teacher_action_mix_ratio
        teacher_action_mix_ratio_start = getattr(distill_cfg, "teacher_action_mix_ratio_start", None)
        teacher_action_mix_ratio_end = getattr(distill_cfg, "teacher_action_mix_ratio_end", None)
        self.teacher_action_mix_ratio_end_iteration = self._strict_config_int(
            "teacher_action_mix_ratio_end_iteration",
            getattr(distill_cfg, "teacher_action_mix_ratio_end_iteration", -1),
        )
        if (teacher_action_mix_ratio_start is None) != (teacher_action_mix_ratio_end is None):
            raise ValueError(
                "distill.teacher_action_mix_ratio_start and distill.teacher_action_mix_ratio_end must be set together."
            )
        if teacher_action_mix_ratio_start is not None and teacher_action_mix_ratio_end is not None:
            self.teacher_action_mix_ratio_start = float(teacher_action_mix_ratio_start)
            self.teacher_action_mix_ratio_end = float(teacher_action_mix_ratio_end)
            if not math.isfinite(self.teacher_action_mix_ratio_start) or not (
                0.0 <= self.teacher_action_mix_ratio_start <= 1.0
            ):
                raise ValueError(
                    "distill.teacher_action_mix_ratio_start must be in [0.0, 1.0], "
                    f"got {self.teacher_action_mix_ratio_start}."
                )
            if not math.isfinite(self.teacher_action_mix_ratio_end) or not (
                0.0 <= self.teacher_action_mix_ratio_end <= 1.0
            ):
                raise ValueError(
                    f"distill.teacher_action_mix_ratio_end must be in [0.0, 1.0], got {self.teacher_action_mix_ratio_end}."
                )
            if self.teacher_action_mix_ratio_end_iteration <= 0:
                raise ValueError(
                    "distill.teacher_action_mix_ratio_end_iteration must be > 0 when teacher-action mix scheduling is enabled."
                )
            if self.teacher_action_mix_ratio_start == self.teacher_action_mix_ratio_end:
                raise ValueError(
                    "distill teacher-action mix schedule start and end must differ; configure a constant "
                    "behavior with teacher_action_mix_ratio instead."
                )
            self.use_teacher_action_mix_schedule = True
            self.teacher_action_mix_ratio = self.teacher_action_mix_ratio_start
        elif self.teacher_action_mix_ratio_end_iteration != -1:
            raise ValueError(
                "distill.teacher_action_mix_ratio_end_iteration is only valid when both "
                "teacher_action_mix_ratio_start and teacher_action_mix_ratio_end are set."
            )
        self.switch_to_rl_after = self._strict_config_int(
            "switch_to_rl_after",
            distill_cfg.switch_to_rl_after,
        )
        if self.switch_to_rl_after < -1:
            raise ValueError(
                "distill.switch_to_rl_after must be -1/0 to disable or a positive iteration, "
                f"got {self.switch_to_rl_after}."
            )
        if self.switch_to_rl_after > 0 and self.bc_loss_coef <= 0.0:
            raise ValueError(
                "distill.switch_to_rl_after requires a positive pre-switch bc_loss_coef; "
                "otherwise the requested BC-to-RL transition has no BC phase."
            )
        self.use_multi_teacher = self._strict_config_bool(
            "use_multi_teacher",
            distill_cfg.use_multi_teacher,
        )
        self.multi_teacher_select_obs_var = str(distill_cfg.multi_teacher_select_obs_var)
        self.ppo_start_epoch = self._strict_config_int(
            "ppo_start_epoch",
            getattr(distill_cfg, "ppo_start_epoch", -1),
        )
        self.dagger_end_epoch = self._strict_config_int(
            "dagger_end_epoch",
            getattr(distill_cfg, "dagger_end_epoch", -1),
        )
        self.ppo_target_coeff = float(getattr(distill_cfg, "ppo_target_coeff", 0.9))
        if not math.isfinite(self.ppo_target_coeff) or not (0.0 <= self.ppo_target_coeff <= 1.0):
            raise ValueError(f"distill.ppo_target_coeff must be in [0.0, 1.0], got {self.ppo_target_coeff}.")
        self.ppo_start_coeff = float(getattr(distill_cfg, "ppo_start_coeff", 0.0))
        if not math.isfinite(self.ppo_start_coeff) or not (0.0 <= self.ppo_start_coeff <= 1.0):
            raise ValueError(f"distill.ppo_start_coeff must be in [0.0, 1.0], got {self.ppo_start_coeff}.")
        if self.ppo_start_coeff > self.ppo_target_coeff:
            raise ValueError(
                "distill.ppo_start_coeff must be <= distill.ppo_target_coeff, "
                f"got {self.ppo_start_coeff} > {self.ppo_target_coeff}."
            )
        raw_start_noise_std = getattr(distill_cfg, "ppo_start_noise_std", None)
        self.ppo_start_noise_std = None if raw_start_noise_std is None else float(raw_start_noise_std)
        if self.ppo_start_noise_std is not None and (
            not math.isfinite(self.ppo_start_noise_std) or self.ppo_start_noise_std <= 0.0
        ):
            raise ValueError(
                "distill.ppo_start_noise_std must be finite and > 0.0 when set, "
                f"got {self.ppo_start_noise_std}."
            )
        self.ppo_start_noise_std_until_coeff = float(
            getattr(distill_cfg, "ppo_start_noise_std_until_coeff", 0.1)
        )
        if not math.isfinite(self.ppo_start_noise_std_until_coeff) or not (
            0.0 <= self.ppo_start_noise_std_until_coeff <= 1.0
        ):
            raise ValueError(
                "distill.ppo_start_noise_std_until_coeff must be in [0.0, 1.0], "
                f"got {self.ppo_start_noise_std_until_coeff}."
            )
        self.ppo_schedule_step_epochs = self._strict_config_int(
            "ppo_schedule_step_epochs",
            getattr(distill_cfg, "ppo_schedule_step_epochs", 0),
        )
        if self.ppo_schedule_step_epochs < 0:
            raise ValueError(
                "distill.ppo_schedule_step_epochs must be >= 0, "
                f"got {self.ppo_schedule_step_epochs}."
            )
        self.dagger_loss_coef = float(getattr(distill_cfg, "dagger_loss_coef", 10.0))
        if not math.isfinite(self.dagger_loss_coef) or self.dagger_loss_coef < 0.0:
            raise ValueError(
                "distill.dagger_loss_coef must be finite and non-negative, "
                f"got {self.dagger_loss_coef}."
            )
        operational_dagger_loss_coef = self._operational_float32_loss_weight(
            self.dagger_loss_coef
        )
        if self.dagger_loss_coef > 0.0 and (
            not math.isfinite(operational_dagger_loss_coef)
            or operational_dagger_loss_coef <= 0.0
        ):
            raise ValueError(
                "distill.dagger_loss_coef must remain finite and positive when materialized in the "
                f"float32 actor loss graph; got Python={self.dagger_loss_coef}, "
                f"float32={operational_dagger_loss_coef}."
            )
        schedule_disabled = self.ppo_start_epoch == -1 and self.dagger_end_epoch == -1
        schedule_valid = self.ppo_start_epoch >= 0 and self.dagger_end_epoch > self.ppo_start_epoch
        if not schedule_disabled and not schedule_valid:
            raise ValueError(
                "distill PPO/DAgger schedule must either set both ppo_start_epoch and dagger_end_epoch to -1, "
                "or satisfy 0 <= ppo_start_epoch < dagger_end_epoch; got "
                f"{self.ppo_start_epoch}, {self.dagger_end_epoch}."
            )
        self.use_ppo_dagger_schedule = schedule_valid
        if self.use_ppo_dagger_schedule:
            configured_ppo_coefficients = (
                ("ppo_start_coeff", self.ppo_start_coeff),
                ("ppo_target_coeff", self.ppo_target_coeff),
            )
            for coefficient_name, coefficient in configured_ppo_coefficients:
                operational_coefficient = self._operational_ppo_coefficient(coefficient)
                if coefficient > 0.0 and operational_coefficient <= 0.0:
                    raise ValueError(
                        f"distill.{coefficient_name} is positive as a Python scalar but rounds to zero "
                        "in the float32 PPO actor loss graph: "
                        f"Python={coefficient}, float32={operational_coefficient}. Use a representable "
                        "PPO coefficient or configure the endpoint as exact zero."
                    )

            # A zero start followed by a very small positive target can have
            # representable endpoints while its first positive interpolated
            # tier still rounds to zero.  Since the schedule is monotone, proving
            # its smallest positive discrete tier proves every later PPO tier.
            minimum_positive_scheduled_coefficient: float | None = None
            if self.ppo_start_coeff > 0.0:
                minimum_positive_scheduled_coefficient = self.ppo_start_coeff
            elif self.ppo_target_coeff > 0.0:
                total_epochs = self.dagger_end_epoch - self.ppo_start_epoch
                if self.ppo_schedule_step_epochs > 0:
                    total_steps = max(
                        1,
                        (total_epochs + self.ppo_schedule_step_epochs - 1)
                        // self.ppo_schedule_step_epochs,
                    )
                    minimum_positive_scheduled_coefficient = (
                        self.ppo_target_coeff / float(total_steps)
                    )
                else:
                    minimum_positive_scheduled_coefficient = (
                        self.ppo_target_coeff / float(total_epochs)
                    )
            if minimum_positive_scheduled_coefficient is not None:
                operational_minimum = self._operational_ppo_coefficient(
                    minimum_positive_scheduled_coefficient
                )
                if operational_minimum <= 0.0:
                    raise ValueError(
                        "The scheduled PPO/DAgger curriculum has a positive Python PPO tier that rounds "
                        "to zero in the float32 actor loss graph: "
                        f"minimum_positive_python={minimum_positive_scheduled_coefficient}, "
                        f"float32={operational_minimum}. Increase the PPO coefficient or shorten the "
                        "schedule so every declared positive PPO tier is operational."
                    )
        if (
            self.use_ppo_dagger_schedule
            and self.ppo_start_epoch == 0
            and self.ppo_start_coeff == 1.0
        ):
            raise ValueError(
                "A scheduled PPO/DAgger curriculum with ppo_start_epoch=0 and "
                "ppo_start_coeff=1.0 has no BC phase. Because ppo_start_coeff must be <= "
                "ppo_target_coeff, the PPO coefficient remains 1.0 for every iteration, so "
                "the nominal DAgger run is actually pure PPO and silently ignores the teacher. "
                "Use ppo_start_coeff < 1.0, delay ppo_start_epoch, or disable distillation for "
                "an intentional pure-PPO run."
            )
        if self.use_ppo_dagger_schedule and self.dagger_loss_coef <= 0.0:
            raise ValueError(
                "A scheduled PPO/DAgger curriculum requires distill.dagger_loss_coef > 0. "
                "With a zero BC weight, pre-PPO iterations can have no actor objective and the "
                "declared DAgger phase is silently absent."
            )
        if self.use_ppo_dagger_schedule:
            initial_ppo_coeff = self._compute_ppo_dagger_coeff_for_epoch(0)
            initial_operational_bc_weight = self._operational_float32_loss_weight(
                self.dagger_loss_coef
                * (1.0 - max(0.0, min(1.0, float(initial_ppo_coeff))))
            )
            if initial_operational_bc_weight <= 0.0:
                raise ValueError(
                    "The scheduled PPO/DAgger curriculum has no operational float32 BC phase: "
                    f"iteration-0 ppo_coeff={initial_ppo_coeff}, "
                    f"dagger_loss_coef={self.dagger_loss_coef}, "
                    f"float32_bc_weight={initial_operational_bc_weight}. Increase the BC weight or "
                    "delay PPO so teacher supervision is representable in the actor loss graph."
                )
            # The BC blend decreases monotonically as PPO ramps up.  Checking
            # iteration zero alone is insufficient: a tiny but representable
            # dagger_loss_coef can silently round later positive BC tiers to
            # zero before the declared DAgger phase ends.  Validate the final
            # positive tier (or the persistent target tier when target < 1),
            # which is the smallest mathematical BC weight in the schedule.
            final_positive_bc_epoch = (
                self.dagger_end_epoch
                if self.ppo_target_coeff < 1.0
                else self.dagger_end_epoch - 1
            )
            final_positive_ppo_coeff = self._compute_ppo_dagger_coeff_for_epoch(
                final_positive_bc_epoch
            )
            final_positive_bc_weight = self.dagger_loss_coef * (
                1.0
                - max(0.0, min(1.0, float(final_positive_ppo_coeff)))
            )
            if (
                final_positive_bc_weight > 0.0
                and self._operational_float32_loss_weight(
                    final_positive_bc_weight
                )
                <= 0.0
            ):
                raise ValueError(
                    "The scheduled PPO/DAgger curriculum declares a positive future BC tier that rounds "
                    "to zero in the float32 actor loss graph: "
                    f"epoch={final_positive_bc_epoch}, PPO={final_positive_ppo_coeff}, "
                    f"Python BC weight={final_positive_bc_weight}. Increase dagger_loss_coef, reduce "
                    "ppo_target_coeff, or shorten/coarsen the PPO ramp so every declared positive BC "
                    "tier remains operational."
                )
        if self.use_ppo_dagger_schedule and self.bc_loss_coef != 1.0:
            raise ValueError(
                "Scheduled PPO/DAgger ignores distill.bc_loss_coef; its BC scale is controlled by "
                "distill.dagger_loss_coef * (1 - ppo_coeff). Keep bc_loss_coef=1.0 as the neutral "
                "schedule sentinel instead of configuring a silently ignored value."
            )
        if self.use_ppo_dagger_schedule and self.switch_to_rl_after > 0:
            raise ValueError(
                "distill.switch_to_rl_after is mutually exclusive with the ppo_start_epoch/dagger_end_epoch schedule."
            )
        if not self.use_ppo_dagger_schedule and self.ppo_start_noise_std is not None:
            raise ValueError(
                "distill.ppo_start_noise_std requires an enabled ppo_start_epoch/dagger_end_epoch schedule."
            )
        if not self.use_ppo_dagger_schedule and self.ppo_schedule_step_epochs > 0:
            raise ValueError(
                "distill.ppo_schedule_step_epochs requires an enabled ppo_start_epoch/dagger_end_epoch schedule."
            )
        if (
            self.use_ppo_dagger_schedule
            and self.ppo_start_noise_std is not None
            and self.ppo_schedule_step_epochs == 0
            and self.ppo_start_coeff > self.ppo_start_noise_std_until_coeff
        ):
            raise ValueError(
                "A linear PPO/DAgger schedule starts above ppo_start_noise_std_until_coeff, so the "
                "configured PPO-start noise cap would never apply. Raise the threshold to at least "
                "ppo_start_coeff or remove ppo_start_noise_std."
            )
        self.ppo_coeff = 0.0 if self.use_ppo_dagger_schedule else 1.0
        loss_type = str(getattr(distill_cfg, "distill_loss_type", "mse")).strip().lower()
        if self.distill_mode == "mse" and loss_type != "mse":
            raise ValueError(
                "distill.mode='mse' requires distill.distill_loss_type='mse'; otherwise the run is "
                "labeled as MSE while optimizing a different loss."
            )
        if loss_type == "mse":
            self.distill_loss_fn = F.mse_loss
        elif loss_type == "huber":
            self.distill_loss_fn = F.huber_loss
        else:
            raise ValueError(f"Unknown distill_loss_type: {loss_type}")
        self.dagger_ignore_zero_teacher_actions = self._strict_config_bool(
            "dagger_ignore_zero_teacher_actions",
            getattr(distill_cfg, "dagger_ignore_zero_teacher_actions", True),
        )
        self.dagger_ignore_episode_initial_steps = self._strict_config_int(
            "dagger_ignore_episode_initial_steps",
            getattr(distill_cfg, "dagger_ignore_episode_initial_steps", 0),
        )
        if self.dagger_ignore_episode_initial_steps < 0:
            raise ValueError(
                "distill.dagger_ignore_episode_initial_steps must be >= 0, "
                f"got {self.dagger_ignore_episode_initial_steps}."
            )
        if self.distill_mode == "dagger" and self.dagger_ignore_episode_initial_steps > 0:
            raw_episode_horizon = getattr(getattr(self, "env", None), "max_episode_length", None)
            try:
                episode_horizon = self._strict_positive_integer_equivalent(
                    "env.max_episode_length",
                    raw_episode_horizon,
                )
            except ValueError as exc:
                raise ValueError(
                    "env.max_episode_length must be a finite positive integer-equivalent horizon when "
                    "distill.dagger_ignore_episode_initial_steps is enabled; "
                    f"got {raw_episode_horizon!r}."
                ) from exc
            save_interval = self._strict_positive_ppo_int(
                "save_interval",
                getattr(self.config, "save_interval", None),
            )
            rollout_steps = self._strict_positive_ppo_int(
                "num_steps_per_env",
                getattr(self.config, "num_steps_per_env", None),
            )
            checkpoint_block_capacity = save_interval * rollout_steps
            effective_age_capacity = self._canonical_rollout_episode_age_capacity(
                episode_horizon,
                checkpoint_block_capacity,
                init_at_random_ep_len=bool(
                    getattr(self.config, "init_at_random_ep_len", False)
                ),
            )
            if self.dagger_ignore_episode_initial_steps >= effective_age_capacity:
                raise ValueError(
                    "distill.dagger_ignore_episode_initial_steps must be smaller than both the "
                    f"episode horizon ({episode_horizon}) and one canonical checkpoint block "
                    f"(save_interval={save_interval} * num_steps_per_env={rollout_steps} = "
                    f"{checkpoint_block_capacity}), got {self.dagger_ignore_episode_initial_steps}. "
                    f"Accounting for reset_all()'s dummy transition, episode age can cover at most "
                    f"0..{effective_age_capacity - 1} before a timeout or canonical checkpoint reset, "
                    "so this configuration would "
                    "permanently empty the BC mask, remove all teacher supervision, and freeze the "
                    "student actor whenever BC is its only signal."
                )
        self.dagger_replay_enabled = self._strict_config_bool(
            "dagger_replay_enabled",
            getattr(distill_cfg, "dagger_replay_enabled", False),
        )
        self.dagger_replay_capacity = self._strict_config_int(
            "dagger_replay_capacity",
            getattr(distill_cfg, "dagger_replay_capacity", 512),
        )
        self.dagger_replay_batch_size = self._strict_config_int(
            "dagger_replay_batch_size",
            getattr(distill_cfg, "dagger_replay_batch_size", 512),
        )
        self.dagger_replay_fraction = self._strict_config_real(
            "dagger_replay_fraction",
            getattr(distill_cfg, "dagger_replay_fraction", 0.5),
        )
        self.dagger_replay_seed = self._strict_config_int(
            "dagger_replay_seed",
            getattr(distill_cfg, "dagger_replay_seed", 0),
        )
        if self.dagger_replay_capacity <= 0:
            raise ValueError(
                "distill.dagger_replay_capacity must be > 0, "
                f"got {self.dagger_replay_capacity}."
            )
        if self.dagger_replay_batch_size <= 0:
            raise ValueError(
                "distill.dagger_replay_batch_size must be > 0, "
                f"got {self.dagger_replay_batch_size}."
            )
        if not 0.0 < self.dagger_replay_fraction < 1.0:
            raise ValueError(
                "distill.dagger_replay_fraction must be strictly between 0 and 1, "
                f"got {self.dagger_replay_fraction}."
            )
        if self.dagger_replay_seed < 0:
            raise ValueError(
                "distill.dagger_replay_seed must be non-negative, "
                f"got {self.dagger_replay_seed}."
            )
        if self.dagger_replay_enabled:
            if self.distill_mode != "dagger" or not distill_config_enabled:
                raise ValueError(
                    "distill.dagger_replay_enabled requires enabled DAgger distillation."
                )
            if self.switch_to_rl_after > 0:
                raise ValueError(
                    "DAgger replay is a pure-BC Stage1 objective and cannot be combined with "
                    "switch_to_rl_after."
                )
            if self.use_ppo_dagger_schedule:
                future_ppo_coefficients = (
                    self._operational_ppo_coefficient(self.ppo_start_coeff),
                    self._operational_ppo_coefficient(self.ppo_target_coeff),
                )
                if any(value > 0.0 for value in future_ppo_coefficients):
                    raise ValueError(
                        "DAgger replay requires operational PPO to remain exactly zero for the "
                        "entire target; replay samples are off-policy and must never enter a PPO "
                        "likelihood-ratio objective."
                    )
            elif self._configured_bc_loss_coef != 1.0:
                raise ValueError(
                    "DAgger replay without a PPO/DAgger schedule requires bc_loss_coef=1.0 so "
                    "the entire target remains pure BC."
                )
        self.dagger_match_std = self._strict_config_bool(
            "dagger_match_std",
            getattr(distill_cfg, "dagger_match_std", False),
        )
        if self.dagger_replay_enabled and self.dagger_match_std:
            raise ValueError(
                "DAgger replay currently rehearses authenticated teacher actions only and cannot "
                "be combined with dagger_match_std; teacher std is intentionally absent from the "
                "replay schema."
            )
        self.strict_teacher_load = self._strict_config_bool(
            "strict_teacher_load",
            getattr(distill_cfg, "strict_teacher_load", True),
        )
        self._configured_fixed_bc_eval_num_samples = self._strict_config_int(
            "fixed_bc_eval_num_samples",
            getattr(distill_cfg, "fixed_bc_eval_num_samples", 0),
        )
        if self._configured_fixed_bc_eval_num_samples < 0:
            raise ValueError(
                "distill.fixed_bc_eval_num_samples must be >= 0, "
                f"got {self._configured_fixed_bc_eval_num_samples}."
            )
        # This diagnostic is defined only for DAgger labels.  The dataclass
        # retains a non-zero DAgger default, so make the effective runtime
        # budget explicit instead of letting pure PPO / legacy MSE checkpoints
        # carry a meaningless fixed-BC stratum.
        self.fixed_bc_eval_num_samples = (
            self._configured_fixed_bc_eval_num_samples
            if self.distill_mode == "dagger"
            else 0
        )
        self.fixed_bc_eval_log_interval = self._strict_config_int(
            "fixed_bc_eval_log_interval",
            getattr(distill_cfg, "fixed_bc_eval_log_interval", 1),
        )
        if self.fixed_bc_eval_log_interval <= 0:
            raise ValueError(
                "distill.fixed_bc_eval_log_interval must be > 0, "
                f"got {self.fixed_bc_eval_log_interval}."
            )
        if self.fixed_bc_eval_num_samples == 0 and self.fixed_bc_eval_log_interval != 1:
            raise ValueError(
                "distill.fixed_bc_eval_log_interval is only consumed when the effective DAgger "
                "fixed_bc_eval_num_samples budget is positive."
            )
        self.fixed_bc_guard_enabled = self._strict_config_bool(
            "fixed_bc_guard_enabled",
            getattr(distill_cfg, "fixed_bc_guard_enabled", False),
        )
        self.fixed_bc_guard_reference_end_epoch = self._strict_config_int(
            "fixed_bc_guard_reference_end_epoch",
            getattr(distill_cfg, "fixed_bc_guard_reference_end_epoch", 600),
        )
        self.fixed_bc_guard_max_reference_ratio = self._strict_config_real(
            "fixed_bc_guard_max_reference_ratio",
            getattr(distill_cfg, "fixed_bc_guard_max_reference_ratio", 2.0),
        )
        self.fixed_bc_guard_absolute_max_mu_mse = self._strict_config_real(
            "fixed_bc_guard_absolute_max_mu_mse",
            getattr(distill_cfg, "fixed_bc_guard_absolute_max_mu_mse", 0.160),
        )
        self.fixed_bc_guard_start_epoch = self._strict_config_int(
            "fixed_bc_guard_start_epoch",
            getattr(distill_cfg, "fixed_bc_guard_start_epoch", -1),
        )
        self.fixed_bc_guard_consecutive_evals = self._strict_config_int(
            "fixed_bc_guard_consecutive_evals",
            getattr(distill_cfg, "fixed_bc_guard_consecutive_evals", 3),
        )
        for field_name, value in (
            (
                "fixed_bc_guard_max_reference_ratio",
                self.fixed_bc_guard_max_reference_ratio,
            ),
            (
                "fixed_bc_guard_absolute_max_mu_mse",
                self.fixed_bc_guard_absolute_max_mu_mse,
            ),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"distill.{field_name} must be finite and > 0, got {value}."
                )
        if self.fixed_bc_guard_max_reference_ratio < 1.0:
            raise ValueError(
                "distill.fixed_bc_guard_max_reference_ratio must be >= 1.0 so the "
                "post-reference ceiling cannot be tighter than the observed reference minimum."
            )
        if self.fixed_bc_guard_reference_end_epoch < 0:
            raise ValueError(
                "distill.fixed_bc_guard_reference_end_epoch must be >= 0, "
                f"got {self.fixed_bc_guard_reference_end_epoch}."
            )
        if self.fixed_bc_guard_consecutive_evals <= 0:
            raise ValueError(
                "distill.fixed_bc_guard_consecutive_evals must be > 0, "
                f"got {self.fixed_bc_guard_consecutive_evals}."
            )
        if self.fixed_bc_guard_enabled:
            if self.distill_mode != "dagger" or not distill_config_enabled:
                raise ValueError(
                    "distill.fixed_bc_guard_enabled requires enabled DAgger distillation."
                )
            if self.fixed_bc_eval_num_samples <= 0:
                raise ValueError(
                    "distill.fixed_bc_guard_enabled requires fixed_bc_eval_num_samples > 0."
                )
            if self.fixed_bc_eval_log_interval <= 0:
                raise ValueError(
                    "distill.fixed_bc_guard_enabled requires fixed_bc_eval_log_interval > 0."
                )
            if not self.use_ppo_dagger_schedule:
                raise ValueError(
                    "distill.fixed_bc_guard_enabled requires a valid PPO/DAgger schedule."
                )
            if self.fixed_bc_guard_start_epoch < self.fixed_bc_guard_reference_end_epoch:
                raise ValueError(
                    "distill.fixed_bc_guard_start_epoch must be >= "
                    "fixed_bc_guard_reference_end_epoch; got "
                    f"{self.fixed_bc_guard_start_epoch} < "
                    f"{self.fixed_bc_guard_reference_end_epoch}."
                )
            reference_ppo_coeff = self._compute_ppo_dagger_coeff_for_epoch(
                self.fixed_bc_guard_reference_end_epoch
            )
            if reference_ppo_coeff != 0.0:
                raise ValueError(
                    "distill fixed-BC guard reference period must remain pure BC through "
                    "fixed_bc_guard_reference_end_epoch; the configured PPO/DAgger schedule "
                    f"has ppo_coeff={reference_ppo_coeff} at that iteration."
                )
            if self.fixed_bc_guard_start_epoch < self.dagger_end_epoch:
                raise ValueError(
                    "distill.fixed_bc_guard_start_epoch must be >= dagger_end_epoch so "
                    "post-reference exceedances are evaluated only after the configured PPO "
                    "ramp completes."
                )
            if self.fixed_bc_guard_reference_end_epoch % self.fixed_bc_eval_log_interval != 0:
                raise ValueError(
                    "distill.fixed_bc_guard_reference_end_epoch must coincide with a fixed-BC "
                    "evaluation iteration."
                )
            if self.fixed_bc_guard_start_epoch % self.fixed_bc_eval_log_interval != 0:
                raise ValueError(
                    "distill.fixed_bc_guard_start_epoch must coincide with a fixed-BC "
                    "evaluation iteration."
                )
            expected_reference_evals = (
                self.fixed_bc_guard_reference_end_epoch
                // self.fixed_bc_eval_log_interval
                + 1
            )
            if expected_reference_evals < 3:
                raise ValueError(
                    "distill fixed-BC guard reference period must contain at least three "
                    "expected evaluations; got "
                    f"{expected_reference_evals}."
                )
            run_end_iteration = getattr(self.config, "num_learning_iterations", None)
            if run_end_iteration is not None:
                run_end_iteration = self._strict_config_int(
                    "num_learning_iterations",
                    run_end_iteration,
                )
                if self.fixed_bc_guard_reference_end_epoch >= run_end_iteration:
                    raise ValueError(
                        "distill.fixed_bc_guard_reference_end_epoch must be below "
                        "algo.config.num_learning_iterations."
                    )
                if self.fixed_bc_guard_start_epoch >= run_end_iteration:
                    raise ValueError(
                        "distill.fixed_bc_guard_start_epoch must be below "
                        "algo.config.num_learning_iterations."
                    )
                final_possible_trip_iteration = self.fixed_bc_guard_start_epoch + (
                    self.fixed_bc_guard_consecutive_evals - 1
                ) * self.fixed_bc_eval_log_interval
                if final_possible_trip_iteration >= run_end_iteration:
                    raise ValueError(
                        "distill fixed-BC guard must have enough scheduled evaluations to "
                        "reach fixed_bc_guard_consecutive_evals before "
                        "algo.config.num_learning_iterations; last required evaluation="
                        f"{final_possible_trip_iteration}, run end={run_end_iteration}."
                    )
        elif self.fixed_bc_guard_start_epoch != -1:
            raise ValueError(
                "Disabled fixed-BC guard requires distill.fixed_bc_guard_start_epoch=-1, "
                f"got {self.fixed_bc_guard_start_epoch}."
            )
        if self.dagger_replay_enabled:
            if self.fixed_bc_eval_num_samples <= 0 or not self.fixed_bc_guard_enabled:
                raise ValueError(
                    "DAgger replay requires an enabled, non-empty fixed-BC guard. Replay collection "
                    "starts only on a later rollout after that immutable dataset is complete, so the "
                    "training rehearsal set can never contaminate the evaluation gate."
                )
        self._fixed_bc_guard_config_fingerprint = (
            self._fixed_bc_guard_runtime_config_fingerprint()
        )
        self._fixed_bc_guard_state = self._new_fixed_bc_guard_state()
        self._validate_teacher_rollout_action_config()
        teacher_checkpoint = policy_to_clone or teacher_checkpoint_alias
        if self.distill_mode == "dagger":
            if not teacher_checkpoint:
                raise ValueError("DAgger mode requires policy_to_clone or teacher_checkpoint.")
            if self.use_multi_teacher:
                if not isinstance(policy_to_clone, list):
                    raise ValueError(
                        "use_multi_teacher=True requires policy_to_clone to be a real list of at least "
                        "two teacher checkpoint paths."
                    )
                if len(policy_to_clone) < 2:
                    raise ValueError(
                        "use_multi_teacher=True requires at least two teacher checkpoint paths; "
                        f"got {len(policy_to_clone)}."
                    )
                invalid_teacher_paths = [
                    index
                    for index, path in enumerate(policy_to_clone)
                    if not isinstance(path, str) or not path.strip()
                ]
                if invalid_teacher_paths:
                    raise ValueError(
                        "Multi-teacher policy_to_clone entries must be non-empty strings; invalid "
                        f"indices={invalid_teacher_paths}."
                    )
                normalized_teacher_paths = [path.strip() for path in policy_to_clone]
                if len(set(normalized_teacher_paths)) != len(normalized_teacher_paths):
                    raise ValueError(
                        "Multi-teacher policy_to_clone contains duplicate checkpoint paths; selector "
                        "classes must identify distinct teacher sources."
                    )
                if not self.multi_teacher_select_obs_var.strip():
                    raise ValueError("multi_teacher_select_obs_var must be a non-empty observation key.")
                selector_dim = self.algo_obs_dim_dict.get(self.multi_teacher_select_obs_var)
                if selector_dim != 1:
                    raise ValueError(
                        "Multi-teacher selector observation must exist and contain exactly one scalar per "
                        f"environment: key={self.multi_teacher_select_obs_var!r}, dimension={selector_dim!r}."
                    )
            if self.bc_loss_coef <= 0.0 and self.switch_to_rl_after <= 0 and not self.use_ppo_dagger_schedule:
                raise ValueError(
                    "DAgger is enabled but has no active or future BC objective: bc_loss_coef=0, "
                    "switch_to_rl_after is disabled, and no PPO/DAgger schedule is configured. This "
                    "would silently ignore the teacher and run pure PPO. Configure pure PPO explicitly "
                    "with distill.enabled=False and distill.mode='mse', or enable a BC objective."
                )
        elif not distill_config_enabled:
            return
        elif self.distill_loss_coef <= 0.0:
            raise ValueError(
                "Legacy MSE distillation is enabled but distill.loss_coef=0, so teacher loading and "
                "teacher-only observations would be computed while the optimization objective remains "
                "pure PPO. Disable distillation explicitly or configure a positive loss_coef."
            )

        teacher_perception_obs_key = getattr(distill_cfg, "teacher_perception_obs_key", None)
        self.teacher_perception_obs_key = str(teacher_perception_obs_key).strip() if teacher_perception_obs_key else ""
        if self.teacher_perception_obs_key and self.teacher_perception_obs_key not in self.algo_obs_dim_dict:
            raise ValueError(
                "Distillation teacher_perception_obs_key not found in observation manager: "
                f"{self.teacher_perception_obs_key}"
            )

        self.teacher_actor = None
        self.teacher_actors = []
        self.teacher_actor_obs_normalizers = {}
        self.teacher_actor_obs_normalizers_list = []

        teacher_obs_keys = distill_cfg.teacher_obs_keys or self.actor_obs_keys
        if isinstance(teacher_obs_keys, str):
            cleaned = teacher_obs_keys.strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                cleaned = cleaned[1:-1]
            teacher_obs_keys = [
                item.strip().strip("'").strip('"')
                for item in cleaned.split(",")
                if item.strip()
            ]
        if not teacher_obs_keys:
            raise ValueError("Distillation teacher_obs_keys is empty.")
        missing_keys = [key for key in teacher_obs_keys if key not in self.algo_obs_dim_dict]
        if missing_keys:
            raise ValueError(f"Teacher obs keys not found in observation manager: {missing_keys}")
        self.teacher_obs_keys = list(teacher_obs_keys)
        self.teacher_obs_slices = self._build_obs_slices(self.teacher_obs_keys)
        self.teacher_obs_dim = self._get_obs_dim(self.teacher_obs_keys)

        if self.distill_mode == "dagger":
            teacher_paths = teacher_checkpoint if isinstance(teacher_checkpoint, list) else [teacher_checkpoint]
            if self.use_multi_teacher:
                if not teacher_paths:
                    raise ValueError("use_multi_teacher=True requires a non-empty policy_to_clone list.")
            elif len(teacher_paths) != 1:
                raise ValueError("Multiple teacher checkpoints provided but use_multi_teacher is False.")

            for path in teacher_paths:
                teacher_actor, teacher_normalizers = self._load_teacher_actor(path, obs_keys=self.teacher_obs_keys)
                if self.use_multi_teacher:
                    self.teacher_actors.append(teacher_actor)
                    self.teacher_actor_obs_normalizers_list.append(teacher_normalizers)
                else:
                    self.teacher_actor = teacher_actor
                    self.teacher_actor_obs_normalizers = teacher_normalizers

            self._validate_loaded_teacher_inference_contract()

            if self.bc_loss_coef > 0.0 or self.switch_to_rl_after > 0 or self.use_ppo_dagger_schedule:
                self.distill_enabled = True
                self.dagger_enabled = True
                self._refresh_distillation_iteration_state(self.current_learning_iteration)
                self._reset_dagger_replay_state()
            return

        if not distill_config_enabled:
            return
        if not teacher_checkpoint:
            raise ValueError("Teacher checkpoint is required for distillation.")
        if isinstance(teacher_checkpoint, list):
            raise ValueError("Single-teacher mode expects a single teacher checkpoint.")

        self.teacher_actor, self.teacher_actor_obs_normalizers = self._load_teacher_actor(
            teacher_checkpoint, obs_keys=self.teacher_obs_keys
        )
        self._validate_loaded_teacher_inference_contract()
        if distill_config_enabled:
            self.distill_enabled = True

    @staticmethod
    def _requested_evaluation_policy_mode() -> str:
        raw_mode = os.environ.get("HOLOSOMA_EVAL_POLICY", "checkpoint_actor").strip().lower()
        aliases = {
            "": "checkpoint_actor",
            "checkpoint_actor": "checkpoint_actor",
            # Backward-compatible spelling for evaluating the checkpoint
            # actor.  This may be a student or a generator teacher depending
            # on which exact checkpoint was supplied to the entrypoint.
            "student": "checkpoint_actor",
            "distill_label_teacher": "distill_label_teacher",
            "distill_label_teacher_bc_target": "distill_label_teacher_bc_target",
        }
        if raw_mode == "teacher":
            raise ValueError(
                "HOLOSOMA_EVAL_POLICY='teacher' is ambiguous: use 'checkpoint_actor' "
                "for the exact supplied checkpoint (including a motion-generator teacher), "
                "or 'distill_label_teacher' for the label policy authenticated by a student checkpoint."
            )
        if raw_mode not in aliases:
            raise ValueError(
                "HOLOSOMA_EVAL_POLICY must be exactly 'checkpoint_actor', 'student', or "
                "'distill_label_teacher', or 'distill_label_teacher_bc_target', "
                f"got {raw_mode!r}."
            )
        return aliases[raw_mode]

    def _prepare_selected_evaluation_policy(self) -> None:
        """Load the authenticated distillation-label teacher only when requested.

        ``checkpoint_actor`` evaluates exactly the checkpoint supplied to the
        evaluation entrypoint; it can therefore represent either a student or
        a separately authenticated motion-generator teacher.  The distinct
        ``distill_label_teacher`` mode resolves only the label policy recorded
        by a loaded student checkpoint's provenance.  Keeping the roles
        explicit prevents a label policy from being presented as the teacher
        that generated the input motion.
        """

        mode = self._requested_evaluation_policy_mode()
        self._evaluation_policy_mode = mode
        if mode == "checkpoint_actor":
            return
        if not bool(getattr(self, "_evaluation_only", False)):
            raise RuntimeError(
                "Distillation-label teacher evaluation is valid only for evaluation-only policy loads."
            )

        source_config = getattr(self, "_experiment_config", None)
        source_algo = getattr(source_config, "algo", None)
        source_algo_config = getattr(source_algo, "config", None)
        distill_cfg = getattr(source_algo_config, "distill", None)
        if distill_cfg is None or not bool(getattr(distill_cfg, "enabled", False)):
            raise ValueError(
                "Distillation-label teacher evaluation requires a student checkpoint whose source config enables distillation."
            )
        if bool(getattr(distill_cfg, "use_multi_teacher", False)):
            raise ValueError(
                "Distillation-label teacher evaluation currently requires exactly one authenticated teacher; "
                "multi-teacher selector semantics are not implicit."
            )
        teacher_checkpoint = getattr(distill_cfg, "policy_to_clone", None) or getattr(
            distill_cfg,
            "teacher_checkpoint",
            None,
        )
        if not isinstance(teacher_checkpoint, str) or not teacher_checkpoint.strip():
            raise ValueError(
                "Distillation-label teacher evaluation source config has no single teacher checkpoint path."
            )

        source_provenance = getattr(self, "_training_provenance", None)
        if not isinstance(source_provenance, dict) or source_provenance.get("teacher_enabled") is not True:
            raise ValueError(
                "Distillation-label teacher evaluation requires the loaded student checkpoint's finalized teacher provenance."
            )

        teacher_obs_keys = getattr(distill_cfg, "teacher_obs_keys", None) or self.actor_obs_keys
        if isinstance(teacher_obs_keys, str):
            cleaned = teacher_obs_keys.strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                cleaned = cleaned[1:-1]
            teacher_obs_keys = [
                item.strip().strip("'").strip('"')
                for item in cleaned.split(",")
                if item.strip()
            ]
        if not isinstance(teacher_obs_keys, (list, tuple)) or not teacher_obs_keys:
            raise ValueError(
                "Distillation-label teacher evaluation source config has no usable teacher observation groups."
            )
        teacher_obs_keys = [str(key) for key in teacher_obs_keys]
        missing_keys = [key for key in teacher_obs_keys if key not in self.algo_obs_dim_dict]
        if missing_keys:
            raise ValueError(
                "Distillation-label teacher evaluation observation groups are absent from the authenticated runtime: "
                f"{missing_keys}."
            )

        self.distill_mode = "dagger"
        self.distill_enabled = True
        self.dagger_enabled = False
        self.use_multi_teacher = False
        self.strict_teacher_load = self._strict_config_bool(
            "strict_teacher_load",
            getattr(distill_cfg, "strict_teacher_load", True),
        )
        self.teacher_use_stochastic_actions = False
        source_clips_teacher_actions = self._strict_config_bool(
            "clip_teacher_actions",
            getattr(distill_cfg, "clip_teacher_actions", False),
        )
        source_clip_actions_threshold = float(
            getattr(distill_cfg, "clip_actions_threshold", 100.0)
        )
        if source_clips_teacher_actions and (
            not math.isfinite(source_clip_actions_threshold)
            or source_clip_actions_threshold <= 0.0
        ):
            raise ValueError(
                "Distillation-label teacher evaluation source config has an "
                "invalid clip_actions_threshold: "
                f"{source_clip_actions_threshold}."
            )
        self._evaluation_teacher_action_clip_enabled = bool(
            mode == "distill_label_teacher_bc_target"
            and source_clips_teacher_actions
        )
        self._evaluation_teacher_action_clip_threshold = (
            source_clip_actions_threshold
            if self._evaluation_teacher_action_clip_enabled
            else None
        )
        teacher_perception_obs_key = getattr(distill_cfg, "teacher_perception_obs_key", None)
        self.teacher_perception_obs_key = (
            str(teacher_perception_obs_key).strip() if teacher_perception_obs_key else ""
        )
        self.teacher_obs_keys = teacher_obs_keys
        self.teacher_obs_slices = self._build_obs_slices(self.teacher_obs_keys)
        self.teacher_obs_dim = self._get_obs_dim(self.teacher_obs_keys)
        self.teacher_actors = []
        self.teacher_actor_obs_normalizers_list = []
        self.teacher_actor, self.teacher_actor_obs_normalizers = self._load_teacher_actor(
            teacher_checkpoint.strip(),
            obs_keys=self.teacher_obs_keys,
        )
        self._validate_loaded_teacher_inference_contract()
        self.teacher_actor.eval()
        for normalizer in self.teacher_actor_obs_normalizers.values():
            normalizer.eval()
        # PPO configured the observation manager before load_evaluation() knew
        # that the authenticated teacher had been requested.  Recompute the
        # active set now, before evaluate_policy() resets the environment and
        # obtains its first observation dictionary.
        self._configure_active_observation_groups()
        logger.info(
            "Evaluation policy selected authenticated distillation-label teacher "
            "SHA256={} with obs groups {} and action semantics {}.",
            source_provenance.get("teacher_sha256"),
            self.teacher_obs_keys,
            (
                f"source BC target clamp [-{source_clip_actions_threshold:g},"
                f"{source_clip_actions_threshold:g}]"
                if self._evaluation_teacher_action_clip_enabled
                else "raw deterministic teacher action"
            ),
        )

    def _get_obs_dim(self, obs_keys: list[str]) -> int:
        """Compute total observation dimension for given observation keys."""
        obs_dim = 0
        for obs_key in obs_keys:
            key_dim = self.algo_obs_dim_dict[obs_key]
            assert isinstance(key_dim, int), f"Observation dimension for {obs_key} is not an integer: {key_dim}"
            # Note: algo_obs_dim_dict from observation_manager.get_obs_dims() already includes history
            obs_dim += key_dim
        return obs_dim

    @staticmethod
    def _clip_teacher_actions_preserving_non_finite(
        teacher_actions: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        """Clip only finite teacher values so clamp cannot conceal corruption."""

        finite_mask = torch.isfinite(teacher_actions)
        clipped = torch.clamp(teacher_actions, -threshold, threshold)
        return torch.where(finite_mask, clipped, teacher_actions)

    def _maybe_capture_fixed_bc_eval_samples(
        self,
        *,
        actor_obs_raw: torch.Tensor,
        actor_perception_obs: torch.Tensor | None,
        teacher_actions: torch.Tensor | None,
        teacher_bc_mask: torch.Tensor | None,
    ) -> None:
        if self.fixed_bc_eval_num_samples <= 0:
            return
        if not self.dagger_enabled or teacher_actions is None or self._fixed_bc_eval_ready:
            return
        if self.actor_perception_key and actor_perception_obs is None:
            return

        eval_teacher_actions = teacher_actions
        if getattr(self, "clip_teacher_actions", False):
            eval_teacher_actions = self._clip_teacher_actions_preserving_non_finite(
                teacher_actions,
                self.clip_actions_threshold,
            )

        valid_mask = torch.ones((teacher_actions.shape[0],), device=teacher_actions.device, dtype=torch.bool)
        if teacher_bc_mask is not None:
            valid_mask &= teacher_bc_mask.view(-1).to(dtype=torch.bool)
        if self.dagger_ignore_zero_teacher_actions:
            valid_mask &= ~torch.all(teacher_actions == 0.0, dim=-1)
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            return

        # Rank-local AS gives every worker a different clip shard.  Capturing
        # only global rank zero therefore measures one clip, not the student
        # objective.  Every rank keeps a bounded local stratum; evaluation
        # aggregates those strata with the same rank weights as training.
        local_target = self._fixed_bc_eval_local_target()
        if local_target == 0:
            self._fixed_bc_eval_ready = True
            return
        remaining = local_target - self._fixed_bc_eval_size
        if remaining <= 0:
            return
        selected = valid_indices[:remaining]
        self._fixed_bc_eval_actor_obs_parts.append(actor_obs_raw[selected].detach().cpu().clone())
        self._fixed_bc_eval_teacher_actions_parts.append(
            eval_teacher_actions[selected].detach().cpu().clone()
        )
        if self.actor_perception_key:
            assert actor_perception_obs is not None
            self._fixed_bc_eval_actor_perception_parts.append(actor_perception_obs[selected].detach().cpu().clone())
        self._fixed_bc_eval_size += int(selected.numel())

        if self._fixed_bc_eval_size < local_target:
            return

        self._fixed_bc_eval_dataset = {
            "actor_obs_raw": torch.cat(self._fixed_bc_eval_actor_obs_parts, dim=0)[:local_target],
            "teacher_actions": torch.cat(self._fixed_bc_eval_teacher_actions_parts, dim=0)[:local_target],
        }
        if self.actor_perception_key:
            self._fixed_bc_eval_dataset["actor_perception"] = torch.cat(
                self._fixed_bc_eval_actor_perception_parts, dim=0
            )[:local_target]
        self._fixed_bc_eval_actor_obs_parts.clear()
        self._fixed_bc_eval_teacher_actions_parts.clear()
        self._fixed_bc_eval_actor_perception_parts.clear()
        self._fixed_bc_eval_ready = True

    def _fixed_bc_eval_capture_pending(self) -> bool:
        """Whether teacher labels are still needed to finish the fixed stratum."""
        return bool(
            getattr(self, "dagger_enabled", False)
            and int(getattr(self, "fixed_bc_eval_num_samples", 0) or 0) > 0
            and self._fixed_bc_eval_local_target() > 0
            and not bool(getattr(self, "_fixed_bc_eval_ready", False))
        )

    def _fixed_bc_eval_local_target(self) -> int:
        """Return this rank's share of the exact global fixed-BC budget."""
        global_budget = max(0, int(getattr(self, "fixed_bc_eval_num_samples", 0) or 0))
        world_size = int(getattr(self, "gpu_world_size", 1) or 1)
        rank = int(getattr(self, "gpu_global_rank", 0) or 0)
        if world_size < 1:
            raise ValueError(f"fixed BC evaluation world size must be positive, got {world_size}.")
        if rank < 0 or rank >= world_size:
            raise ValueError(
                "fixed BC evaluation rank is outside the active world: "
                f"rank={rank}, world_size={world_size}."
            )
        quotient, remainder = divmod(global_budget, world_size)
        return quotient + int(rank < remainder)

    def _fixed_bc_guard_runtime_config(self) -> dict[str, Any]:
        return {
            "state_version": _FIXED_BC_GUARD_STATE_VERSION,
            "enabled": bool(getattr(self, "fixed_bc_guard_enabled", False)),
            "reference_end_epoch": int(
                getattr(self, "fixed_bc_guard_reference_end_epoch", 600)
            ),
            "max_reference_ratio": float(
                getattr(self, "fixed_bc_guard_max_reference_ratio", 2.0)
            ),
            "absolute_max_mu_mse": float(
                getattr(self, "fixed_bc_guard_absolute_max_mu_mse", 0.160)
            ),
            "start_epoch": int(getattr(self, "fixed_bc_guard_start_epoch", -1)),
            "consecutive_evals": int(
                getattr(self, "fixed_bc_guard_consecutive_evals", 3)
            ),
            "fixed_bc_eval_num_samples": int(
                getattr(self, "fixed_bc_eval_num_samples", 0)
            ),
            "fixed_bc_eval_log_interval": int(
                getattr(self, "fixed_bc_eval_log_interval", 1)
            ),
            "world_size": int(getattr(self, "gpu_world_size", 1) or 1),
            "distill_mode": str(getattr(self, "distill_mode", "mse")),
            "ppo_start_epoch": int(getattr(self, "ppo_start_epoch", -1)),
            "dagger_end_epoch": int(getattr(self, "dagger_end_epoch", -1)),
            "ppo_start_coeff": float(getattr(self, "ppo_start_coeff", 0.0)),
            "ppo_target_coeff": float(getattr(self, "ppo_target_coeff", 0.9)),
            "ppo_schedule_step_epochs": int(
                getattr(self, "ppo_schedule_step_epochs", 0)
            ),
        }

    def _fixed_bc_guard_runtime_config_fingerprint(self) -> str:
        payload = json.dumps(
            self._fixed_bc_guard_runtime_config(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _new_fixed_bc_guard_state(self) -> dict[str, Any]:
        return {
            "version": _FIXED_BC_GUARD_STATE_VERSION,
            "config_fingerprint": str(
                getattr(self, "_fixed_bc_guard_config_fingerprint", "")
            ),
            "local_dataset_digest_by_rank": None,
            "global_dataset_digest": None,
            "reference_min_mu_mse": None,
            "reference_min_iteration": None,
            "reference_eval_count": 0,
            "threshold_mu_mse": None,
            "consecutive_exceedances": 0,
            "last_eval_iteration": None,
            "last_mu_mse": None,
            "tripped": False,
            "trip_iteration": None,
            "trip_mu_mse": None,
        }

    def _fixed_bc_guard_digest_dataset(
        self,
        dataset: Mapping[str, Any],
        *,
        expected_rows: int,
        context: str,
    ) -> str:
        required = {"actor_obs_raw", "teacher_actions"}
        if getattr(self, "actor_perception_key", ""):
            required.add("actor_perception")
        expected_widths = {
            "actor_obs_raw": self._get_obs_dim(self.actor_obs_keys),
            "teacher_actions": int(self.num_act),
        }
        if getattr(self, "actor_perception_key", ""):
            expected_widths["actor_perception"] = int(
                self.algo_obs_dim_dict[self.actor_perception_key]
            )
        return fixed_bc_dataset_sha256(
            dataset,
            expected_rows=expected_rows,
            required_tensor_keys=required,
            expected_widths=expected_widths,
            expected_dtype=self._fixed_bc_checkpoint_dtype(),
            context=context,
        )

    @staticmethod
    def _fixed_bc_guard_global_dataset_digest(
        digest_by_rank: Mapping[str, str],
        *,
        global_sample_budget: int,
        world_size: int,
    ) -> str:
        return fixed_bc_global_dataset_sha256(
            digest_by_rank,
            global_sample_budget=global_sample_budget,
            world_size=world_size,
        )

    def _fixed_bc_guard_live_dataset_digests(self) -> tuple[dict[str, str], str]:
        rank = int(getattr(self, "gpu_global_rank", 0) or 0)
        world_size = int(getattr(self, "gpu_world_size", 1) or 1)
        local_target = 0
        local_error: Exception | None = None
        local_digest: str | None = None
        try:
            local_target = self._fixed_bc_eval_local_target()
            ready = bool(
                getattr(self, "_fixed_bc_eval_ready", False) or local_target == 0
            )
            if not ready:
                raise RuntimeError(
                    "Fixed-BC guard expected a complete frozen dataset on every allocated rank."
                )
            local_size = int(getattr(self, "_fixed_bc_eval_size", 0) or 0)
            if local_size != local_target:
                raise RuntimeError(
                    "Fixed-BC guard rank-local sample allocation drifted: "
                    f"rank={rank}, expected={local_target}, actual={local_size}."
                )
            local_digest = self._fixed_bc_guard_digest_dataset(
                getattr(self, "_fixed_bc_eval_dataset", {}),
                expected_rows=local_target,
                context=f"live fixed BC dataset rank {rank}",
            )
        except Exception as exc:
            local_error = exc
        self._synchronize_training_phase_error(
            local_error,
            operation="fixed BC guard dataset digest",
        )
        assert local_digest is not None

        distributed = (
            getattr(self, "is_multi_gpu", False)
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if not distributed:
            digest_by_rank = {str(rank): local_digest}
        else:
            group = self._setup_gloo_barrier_group()
            group_error = (
                RuntimeError(
                    "Distributed fixed-BC guard digest validation requires an all-rank Gloo group."
                )
                if group is None
                else None
            )
            self._synchronize_training_phase_error(
                group_error,
                operation="fixed BC guard dataset-digest group validation",
            )
            assert group is not None
            group_world_size = torch.distributed.get_world_size(group=group)
            group_rank = torch.distributed.get_rank(group=group)
            topology_error: Exception | None = None
            if group_world_size != world_size or group_rank != rank:
                topology_error = RuntimeError(
                    "Fixed-BC guard Gloo topology disagrees with the PPO topology: "
                    f"group={group_rank}/{group_world_size}, PPO={rank}/{world_size}."
                )
            self._synchronize_training_phase_error(
                topology_error,
                operation="fixed BC guard dataset-digest topology validation",
            )
            gathered: list[dict[str, Any] | None] = [None] * group_world_size
            torch.distributed.all_gather_object(
                gathered,
                {"rank": rank, "local_target": local_target, "digest": local_digest},
                group=group,
            )
            digest_by_rank = {}
            quotient, remainder = divmod(
                int(self.fixed_bc_eval_num_samples), world_size
            )
            problems = []
            for expected_rank, result in enumerate(gathered):
                expected_target = quotient + int(expected_rank < remainder)
                if not isinstance(result, dict) or result.get("rank") != expected_rank:
                    problems.append(f"slot={expected_rank}: malformed digest envelope")
                    continue
                if result.get("local_target") != expected_target:
                    problems.append(
                        f"rank={expected_rank}: local_target={result.get('local_target')!r} "
                        f"!= {expected_target}"
                    )
                digest_value = result.get("digest")
                if (
                    not isinstance(digest_value, str)
                    or len(digest_value) != 64
                    or any(char not in "0123456789abcdef" for char in digest_value)
                ):
                    problems.append(
                        f"rank={expected_rank}: malformed dataset SHA-256"
                    )
                else:
                    digest_by_rank[str(expected_rank)] = digest_value
            if problems:
                raise RuntimeError(
                    "Distributed fixed-BC guard dataset digests are malformed: "
                    + "; ".join(problems)
                )
        global_digest = self._fixed_bc_guard_global_dataset_digest(
            digest_by_rank,
            global_sample_budget=int(self.fixed_bc_eval_num_samples),
            world_size=world_size,
        )
        return digest_by_rank, global_digest

    def _fixed_bc_guard_expected_weighted_sample_count(self) -> float:
        local_error: Exception | None = None
        local_weight = 0.0
        try:
            local_weight = (
                float(self._fixed_bc_eval_local_target())
                * float(self._get_distributed_loss_weight())
            )
        except Exception as exc:
            local_error = exc
        self._synchronize_training_phase_error(
            local_error,
            operation="fixed BC guard weighted-count validation",
        )
        local = torch.tensor(local_weight, device=self.device, dtype=torch.float64)
        if getattr(self, "is_multi_gpu", False):
            local = self._all_reduce_small_tensor(
                local,
                op=torch.distributed.ReduceOp.SUM,
            )
        result = float(local.item())
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(
                "Fixed-BC guard expected weighted sample count must be finite and positive, "
                f"got {result}."
            )
        return result

    def _update_fixed_bc_guard(
        self,
        *,
        current_iteration: int,
        metrics: Mapping[str, Any],
    ) -> bool:
        """Update the guard without making its digest/collective protocol stochastic."""
        if not bool(getattr(self, "fixed_bc_guard_enabled", False)):
            return False
        if current_iteration % int(self.fixed_bc_eval_log_interval) != 0:
            off_interval_error = (
                RuntimeError(
                    "Fixed-BC guard received metrics outside the configured evaluation interval."
                )
                if metrics
                else None
            )
            self._synchronize_training_phase_error(
                off_interval_error,
                operation="fixed BC guard off-interval metric validation",
            )
            return False

        rng_state = None
        capture_error: Exception | None = None
        try:
            rng_state = capture_rng_checkpoint_state()
        except Exception as exc:
            capture_error = exc
        self._synchronize_training_phase_error(
            capture_error,
            operation="fixed BC guard RNG-state capture",
        )
        assert rng_state is not None

        result = False
        update_error: Exception | None = None
        try:
            result = self._update_fixed_bc_guard_impl(
                current_iteration=current_iteration,
                metrics=metrics,
            )
        except Exception as exc:
            update_error = exc

        restore_error: Exception | None = None
        try:
            restore_rng_checkpoint_state(
                rng_state,
                path="pre_fixed_bc_guard_update_rng_state",
            )
        except Exception as exc:
            restore_error = exc
        self._synchronize_training_phase_error(
            restore_error,
            operation="fixed BC guard RNG-state restoration",
        )
        self._synchronize_training_phase_error(
            update_error,
            operation="fixed BC guard update outcome",
        )
        return result

    def _update_fixed_bc_guard_impl(
        self,
        *,
        current_iteration: int,
        metrics: Mapping[str, Any],
    ) -> bool:
        """Validate one fixed-set measurement and update the fail-closed state."""

        interval = int(self.fixed_bc_eval_log_interval)
        if current_iteration % interval != 0:
            if metrics:
                raise RuntimeError(
                    "Fixed-BC guard received metrics outside the configured evaluation interval."
                )
            return False

        # No rank may advance into the weighted-count collective while a peer
        # rejected its rank-local metric envelope.  Even though the metrics are
        # normally produced by all-reduces, explicit synchronization here is
        # what makes corruption and mocked/alternate callers fail closed rather
        # than split the collective sequence.
        parsed: dict[str, float] = {}
        expected_budget = int(self.fixed_bc_eval_num_samples)
        metric_error: Exception | None = None
        try:
            required_metrics = {
                "fixed_bc_mu_mse",
                "fixed_bc_num_samples",
                "fixed_bc_weighted_num_samples",
                "fixed_bc_rank_strata",
            }
            missing = required_metrics - set(metrics)
            if missing:
                raise RuntimeError(
                    "Fixed-BC guard evaluation is missing required metrics: "
                    f"{sorted(missing)}."
                )
            for key in sorted(required_metrics):
                raw_value = metrics[key]
                if isinstance(raw_value, bool) or not isinstance(raw_value, numbers.Real):
                    raise ValueError(
                        f"Fixed-BC guard metric {key} must be a real scalar, got {raw_value!r}."
                    )
                value = float(raw_value)
                if not math.isfinite(value):
                    raise ValueError(
                        f"Fixed-BC guard metric {key} is non-finite: {value}."
                    )
                parsed[key] = value
            if parsed["fixed_bc_mu_mse"] < 0.0:
                raise ValueError("Fixed-BC guard mean-action MSE cannot be negative.")
            if parsed["fixed_bc_num_samples"] != float(expected_budget):
                raise ValueError(
                    "Fixed-BC guard sample budget mismatch: "
                    f"expected={expected_budget}, actual={parsed['fixed_bc_num_samples']}."
                )
            expected_strata = min(
                expected_budget,
                int(getattr(self, "gpu_world_size", 1) or 1),
            )
            if parsed["fixed_bc_rank_strata"] != float(expected_strata):
                raise ValueError(
                    "Fixed-BC guard rank-strata mismatch: "
                    f"expected={expected_strata}, actual={parsed['fixed_bc_rank_strata']}."
                )
        except Exception as exc:
            metric_error = exc
        self._synchronize_training_phase_error(
            metric_error,
            operation="fixed BC guard metric-envelope validation",
        )

        expected_weighted_count = self._fixed_bc_guard_expected_weighted_sample_count()
        state: dict[str, Any] | None = None
        pre_digest_error: Exception | None = None
        try:
            if not math.isclose(
                parsed["fixed_bc_weighted_num_samples"],
                expected_weighted_count,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "Fixed-BC guard weighted sample count mismatch: "
                    f"expected={expected_weighted_count}, "
                    f"actual={parsed['fixed_bc_weighted_num_samples']}."
                )
            live_state = self._fixed_bc_guard_state
            if not isinstance(live_state, dict):
                raise RuntimeError("Fixed-BC guard runtime state must be a dictionary.")
            state = live_state
            expected_previous = current_iteration - interval if current_iteration > 0 else None
            if state.get("last_eval_iteration") != expected_previous:
                raise RuntimeError(
                    "Fixed-BC guard evaluation timeline is discontinuous: "
                    f"iteration={current_iteration}, previous={state.get('last_eval_iteration')!r}, "
                    f"expected_previous={expected_previous!r}."
                )
        except Exception as exc:
            pre_digest_error = exc
        self._synchronize_training_phase_error(
            pre_digest_error,
            operation="fixed BC guard timeline validation",
        )
        assert state is not None

        digest_by_rank, global_digest = self._fixed_bc_guard_live_dataset_digests()
        semantic_error: Exception | None = None
        try:
            # Before applying the next measurement, prove that the complete
            # current state is exactly the state implied by all evaluations
            # preceding current_iteration.  This prevents coordinated or
            # rank-local corruption from silently re-basing reference minima,
            # counters, thresholds, or dataset authentication.
            self._validate_fixed_bc_guard_checkpoint_state(
                state,
                next_iteration=current_iteration,
                digest_by_rank=digest_by_rank,
                global_digest=global_digest,
            )
        except Exception as exc:
            semantic_error = exc
        self._synchronize_training_phase_error(
            semantic_error,
            operation="fixed BC guard pre-transition semantic validation",
        )

        transition_error: Exception | None = None
        try:
            if type(state.get("tripped")) is not bool:
                raise RuntimeError("Fixed-BC guard runtime trip flag must be boolean.")
            if state["tripped"]:
                raise RuntimeError(
                    "Fixed-BC guard runtime state is already tripped; training must not continue."
                )
            stored_by_rank = state.get("local_dataset_digest_by_rank")
            stored_global = state.get("global_dataset_digest")
            if stored_by_rank is None and stored_global is None:
                if current_iteration != 0:
                    raise RuntimeError(
                        "Fixed-BC guard dataset digests may be initialized only at the first "
                        "evaluation; refusing to re-baseline a later or drifted dataset."
                    )
                state["local_dataset_digest_by_rank"] = digest_by_rank
                state["global_dataset_digest"] = global_digest
            elif stored_by_rank != digest_by_rank or stored_global != global_digest:
                raise RuntimeError(
                    "Fixed-BC guard frozen dataset content digest drifted during training."
                )

            mu_mse = parsed["fixed_bc_mu_mse"]
            reference_end = int(self.fixed_bc_guard_reference_end_epoch)
            if current_iteration <= reference_end:
                previous_min = state.get("reference_min_mu_mse")
                if previous_min is None or mu_mse < float(previous_min):
                    state["reference_min_mu_mse"] = mu_mse
                    state["reference_min_iteration"] = current_iteration
                state["reference_eval_count"] = int(state["reference_eval_count"]) + 1
                expected_count = current_iteration // interval + 1
                if state["reference_eval_count"] != expected_count:
                    raise RuntimeError(
                        "Fixed-BC guard reference evaluation count is discontinuous: "
                        f"actual={state['reference_eval_count']}, expected={expected_count}."
                    )
                if current_iteration == reference_end:
                    reference_min = state["reference_min_mu_mse"]
                    if reference_min is None:
                        raise RuntimeError("Fixed-BC guard reference minimum was not established.")
                    state["threshold_mu_mse"] = min(
                        float(reference_min)
                        * float(self.fixed_bc_guard_max_reference_ratio),
                        float(self.fixed_bc_guard_absolute_max_mu_mse),
                    )
            else:
                if state.get("threshold_mu_mse") is None:
                    raise RuntimeError(
                        "Fixed-BC guard reference threshold was not frozen at the configured boundary."
                    )
                if current_iteration >= int(self.fixed_bc_guard_start_epoch):
                    if mu_mse > float(state["threshold_mu_mse"]):
                        state["consecutive_exceedances"] = (
                            int(state["consecutive_exceedances"]) + 1
                        )
                    else:
                        state["consecutive_exceedances"] = 0

            state["last_eval_iteration"] = current_iteration
            state["last_mu_mse"] = mu_mse
            if int(state["consecutive_exceedances"]) >= int(
                self.fixed_bc_guard_consecutive_evals
            ):
                state["tripped"] = True
                state["trip_iteration"] = current_iteration
                state["trip_mu_mse"] = mu_mse
        except Exception as exc:
            transition_error = exc
        self._synchronize_training_phase_error(
            transition_error,
            operation="fixed BC guard state transition",
        )

        # The state digest covers the counter, threshold, dataset digests, and
        # trip flag.  Require exact all-rank equality before any worker is
        # allowed to enter the diagnostic-checkpoint barrier.
        self._require_all_rank_fixed_bc_guard_state_match(state)
        # Branch only from the state whose canonical digest reached consensus.
        return state.get("tripped") is True

    def _fixed_bc_guard_log_metrics(self) -> dict[str, float]:
        if not bool(getattr(self, "fixed_bc_guard_enabled", False)):
            return {}
        state = self._fixed_bc_guard_state
        reference_min = state.get("reference_min_mu_mse")
        last_mu_mse = state.get("last_mu_mse")
        if reference_min is None or last_mu_mse is None:
            return {}
        effective_threshold = state.get("threshold_mu_mse")
        if effective_threshold is None:
            effective_threshold = min(
                float(reference_min)
                * float(self.fixed_bc_guard_max_reference_ratio),
                float(self.fixed_bc_guard_absolute_max_mu_mse),
            )
        return {
            "fixed_bc_guard_reference_min_mu_mse": float(reference_min),
            "fixed_bc_guard_effective_threshold_mu_mse": float(
                effective_threshold
            ),
            "fixed_bc_guard_consecutive_exceedances": float(
                state["consecutive_exceedances"]
            ),
            "fixed_bc_guard_last_mu_mse": float(last_mu_mse),
        }

    def _publish_fixed_bc_guard_trip_diagnostic(self, *, next_iteration: int) -> str:
        diagnostic_path = os.path.join(
            self.log_dir,
            f"diagnostic_fixed_bc_guard_{next_iteration:05d}.pt",
        )
        self._distributed_barrier()
        self._save_checkpoint_with_distributed_outcome(
            diagnostic_path,
            next_iteration=next_iteration,
            allow_tripped_fixed_bc_guard=True,
        )
        self._distributed_barrier()
        if self._is_node_local_main_process():
            guard_state = self._fixed_bc_guard_state
            print(
                "HOLOSOMA_FIXED_BC_GUARD_TRIPPED "
                f"completed_iteration={next_iteration - 1} "
                f"eval_iteration={guard_state['trip_iteration']} "
                f"fixed_bc_mu_mse={guard_state['trip_mu_mse']} "
                f"threshold={guard_state['threshold_mu_mse']} "
                f"consecutive={guard_state['consecutive_exceedances']} "
                f"checkpoint={diagnostic_path}",
                flush=True,
            )
        return diagnostic_path

    def _fixed_bc_guard_trip_error(
        self,
        *,
        diagnostic_path: str,
        logging_error: Exception | None = None,
    ) -> RuntimeError:
        message = (
            "Fixed-BC scientific guard tripped after "
            f"{self.fixed_bc_guard_consecutive_evals} consecutive evaluations "
            "above its frozen reference threshold; diagnostic checkpoint="
            f"{diagnostic_path}."
        )
        if logging_error is not None:
            message += (
                " Terminal metric logging also failed after the diagnostic was safely published: "
                f"{type(logging_error).__name__}: {logging_error}"
            )
        return RuntimeError(message)

    def _abort_for_fixed_bc_guard_trip(self, *, next_iteration: int) -> None:
        diagnostic_path = self._publish_fixed_bc_guard_trip_diagnostic(
            next_iteration=next_iteration
        )
        raise self._fixed_bc_guard_trip_error(diagnostic_path=diagnostic_path)

    @staticmethod
    def _terminal_fixed_bc_eval_state_sha256(state: Mapping[str, Any]) -> str:
        return terminal_fixed_bc_eval_state_sha256(state)

    @classmethod
    def _validate_terminal_fixed_bc_eval_state(
        cls,
        state: Any,
        *,
        expected_completed_iteration: int | None = None,
    ) -> dict[str, Any]:
        """Validate the immutable final observation without touching guard state."""

        return validate_terminal_fixed_bc_eval_state(
            state,
            expected_completed_iteration=expected_completed_iteration,
        )

    def _build_terminal_fixed_bc_eval_state(
        self,
        *,
        completed_iteration: int,
        metrics: Mapping[str, Any],
        scheduled_evaluation: bool,
        global_dataset_digest: str,
        expected_weighted_num_samples: float,
    ) -> dict[str, Any]:
        required_metrics = {
            "fixed_bc_mu_mse",
            "fixed_bc_num_samples",
            "fixed_bc_weighted_num_samples",
            "fixed_bc_rank_strata",
        }
        missing = required_metrics - set(metrics)
        if missing:
            raise RuntimeError(
                "Final fixed-BC observation is missing required metrics: "
                f"{sorted(missing)}."
            )

        if type(scheduled_evaluation) is not bool:
            raise ValueError(
                "Final fixed-BC scheduled_evaluation must be a boolean."
            )

        def strict_real(key: str, value: Any) -> float:
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(
                    f"Final fixed-BC metric {key} must be a real scalar."
                )
            parsed = float(value)
            if not math.isfinite(parsed) or parsed < 0.0:
                raise ValueError(
                    f"Final fixed-BC metric {key} must be finite and non-negative."
                )
            return parsed

        def strict_count(key: str) -> int:
            value = metrics[key]
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(f"Final fixed-BC metric {key} must be a real scalar.")
            parsed = float(value)
            if not math.isfinite(parsed) or not parsed.is_integer():
                raise ValueError(f"Final fixed-BC metric {key} must be a finite integer value.")
            return int(parsed)

        mu_mse = strict_real("fixed_bc_mu_mse", metrics["fixed_bc_mu_mse"])
        weighted_num_samples = strict_real(
            "fixed_bc_weighted_num_samples",
            metrics["fixed_bc_weighted_num_samples"],
        )
        expected_weighted_num_samples = strict_real(
            "fixed_bc_expected_weighted_num_samples",
            expected_weighted_num_samples,
        )

        guard_enabled = bool(getattr(self, "fixed_bc_guard_enabled", False))
        guard_state_digest: str | None = None
        threshold: float | None = None
        within_threshold: bool | None = None
        if guard_enabled:
            guard_state = getattr(self, "_fixed_bc_guard_state", None)
            if not isinstance(guard_state, Mapping):
                raise RuntimeError(
                    "Final fixed-BC observation requires a valid live guard state."
                )
            guard_state_digest = self._terminal_fixed_bc_eval_state_sha256(
                guard_state
            )
            raw_threshold = guard_state.get("threshold_mu_mse")
            if (
                isinstance(raw_threshold, bool)
                or not isinstance(raw_threshold, numbers.Real)
                or not math.isfinite(float(raw_threshold))
                or float(raw_threshold) < 0.0
            ):
                raise RuntimeError(
                    "Final fixed-BC observation requires the scientific guard's "
                    "frozen finite threshold."
                )
            threshold = float(raw_threshold)
            within_threshold = mu_mse <= threshold

        state = {
            "version": _TERMINAL_FIXED_BC_EVAL_STATE_VERSION,
            "terminal_observation": True,
            "completed_iteration": int(completed_iteration),
            "next_iteration": int(completed_iteration) + 1,
            "run_target_iteration": int(self.config.num_learning_iterations),
            "scheduled_evaluation": bool(scheduled_evaluation),
            "guard_enabled": guard_enabled,
            "guard_applied": bool(
                scheduled_evaluation
                and guard_enabled
            ),
            "fixed_bc_eval_log_interval": int(self.fixed_bc_eval_log_interval),
            "fixed_bc_eval_num_samples": int(self.fixed_bc_eval_num_samples),
            "world_size": int(getattr(self, "gpu_world_size", 1) or 1),
            "fixed_bc_global_dataset_sha256": str(global_dataset_digest),
            "fixed_bc_guard_config_sha256": str(
                self._fixed_bc_guard_config_fingerprint
            ),
            "fixed_bc_guard_state_sha256": guard_state_digest,
            "fixed_bc_guard_threshold_mu_mse": threshold,
            "fixed_bc_terminal_within_threshold": within_threshold,
            "fixed_bc_mu_mse": mu_mse,
            "fixed_bc_num_samples": strict_count("fixed_bc_num_samples"),
            "fixed_bc_weighted_num_samples": weighted_num_samples,
            "fixed_bc_expected_weighted_num_samples": (
                expected_weighted_num_samples
            ),
            "fixed_bc_rank_strata": strict_count("fixed_bc_rank_strata"),
        }
        return self._validate_terminal_fixed_bc_eval_state(
            state,
            expected_completed_iteration=completed_iteration,
        )

    def _build_terminal_fixed_bc_eval_state_preserving_rng(
        self,
        *,
        completed_iteration: int,
        metrics: Mapping[str, Any],
        scheduled_evaluation: bool,
    ) -> dict[str, Any]:
        """Authenticate the terminal proof without advancing training RNG."""

        rng_state = None
        capture_error: Exception | None = None
        try:
            rng_state = capture_rng_checkpoint_state()
        except Exception as exc:
            capture_error = exc
        self._synchronize_training_phase_error(
            capture_error,
            operation="terminal fixed BC proof RNG-state capture",
        )
        assert rng_state is not None

        terminal_state: dict[str, Any] | None = None
        proof_verdict_error: Exception | None = None
        restore_error: Exception | None = None
        try:
            proof_error: Exception | None = None
            try:
                _, global_dataset_digest = (
                    self._fixed_bc_guard_live_dataset_digests()
                )
                expected_weighted_num_samples = (
                    self._fixed_bc_guard_expected_weighted_sample_count()
                )
                terminal_state = self._build_terminal_fixed_bc_eval_state(
                    completed_iteration=completed_iteration,
                    metrics=metrics,
                    scheduled_evaluation=scheduled_evaluation,
                    global_dataset_digest=global_dataset_digest,
                    expected_weighted_num_samples=expected_weighted_num_samples,
                )
                # Prove every rank built byte-identical canonical state.  This
                # verifier is observational and does not mutate the guard.
                self._require_all_rank_fixed_bc_guard_state_match(terminal_state)
            except Exception as exc:
                proof_error = exc
            try:
                self._synchronize_training_phase_error(
                    proof_error,
                    operation="terminal fixed BC proof construction",
                )
            except Exception as exc:
                proof_verdict_error = exc
        finally:
            try:
                restore_rng_checkpoint_state(
                    rng_state,
                    path="pre_terminal_fixed_bc_proof_rng_state",
                )
            except Exception as exc:
                restore_error = exc

        self._synchronize_training_phase_error(
            restore_error,
            operation="terminal fixed BC proof RNG-state restoration",
        )
        if proof_verdict_error is not None:
            raise proof_verdict_error
        assert terminal_state is not None
        return terminal_state

    def _terminal_fixed_bc_eval_artifact_metadata(
        self,
        *,
        completed_iteration: int,
    ) -> dict[str, Any]:
        state = getattr(self, "_terminal_fixed_bc_eval_state", None)
        if state is None:
            return {}
        if state.get("completed_iteration") != int(completed_iteration):
            return {}
        canonical = self._validate_terminal_fixed_bc_eval_state(
            state,
            expected_completed_iteration=completed_iteration,
        )
        return {
            _TERMINAL_FIXED_BC_EVAL_STATE_KEY: canonical,
            _TERMINAL_FIXED_BC_EVAL_STATE_SHA256_KEY: self._terminal_fixed_bc_eval_state_sha256(
                canonical
            ),
        }

    @classmethod
    def _validate_checkpoint_terminal_fixed_bc_eval(
        cls,
        loaded_dict: Mapping[str, Any],
        *,
        expected_completed_iteration: int,
    ) -> dict[str, Any] | None:
        return validate_checkpoint_terminal_fixed_bc_eval(
            loaded_dict,
            expected_completed_iteration=expected_completed_iteration,
        )

    @staticmethod
    def _checkpoint_saved_run_target(loaded_dict: Mapping[str, Any]) -> int:
        return checkpoint_saved_run_target(loaded_dict)

    def _validate_terminal_fixed_bc_eval_artifact_payload(
        self,
        loaded_dict: Mapping[str, Any],
        *,
        expected_completed_iteration: int,
        compare_runtime_guard_config: bool = False,
    ) -> dict[str, Any] | None:
        terminal_state = validate_checkpoint_terminal_fixed_bc_eval(
            loaded_dict,
            expected_completed_iteration=expected_completed_iteration,
        )
        if terminal_state is None:
            return None
        required_tensor_keys = {"actor_obs_raw", "teacher_actions"}
        expected_widths = {
            "actor_obs_raw": self._get_obs_dim(self.actor_obs_keys),
            "teacher_actions": int(self.num_act),
        }
        if getattr(self, "actor_perception_key", ""):
            required_tensor_keys.add("actor_perception")
            expected_widths["actor_perception"] = int(
                self.algo_obs_dim_dict[self.actor_perception_key]
            )
        return validate_terminal_fixed_bc_eval_artifact_payload(
            loaded_dict,
            expected_completed_iteration=expected_completed_iteration,
            required_tensor_keys=required_tensor_keys,
            expected_widths=expected_widths,
            expected_dtype=self._fixed_bc_checkpoint_dtype(),
            expected_guard_config_sha256=(
                str(getattr(self, "_fixed_bc_guard_config_fingerprint", ""))
                if compare_runtime_guard_config
                else None
            ),
        )

    @torch.no_grad()
    def _get_fixed_bc_eval_metrics(
        self,
        current_iteration: int,
        *,
        terminal_observation: bool = False,
    ) -> dict[str, float]:
        """Evaluate the frozen BC stratum without perturbing training RNG.

        Every rank enters this method because readiness and sufficient
        statistics are distributed collectives.  Flow actors are allowed to
        consume randomness in pure-BC configurations.  A fixed input stratum
        is not a fixed diagnostic if those draws depend on the live training
        RNG, so every evaluation also starts from one stable rank-specific RNG
        seed.  Treat the complete diagnostic as observational and restore the
        caller's Python, NumPy, CPU-torch, and CUDA-torch streams even on an
        exception.
        """

        rng_state = None
        capture_error: Exception | None = None
        try:
            rng_state = capture_rng_checkpoint_state()
        except Exception as exc:
            capture_error = exc
        self._synchronize_training_phase_error(
            capture_error,
            operation="fixed BC evaluation RNG-state capture",
        )
        assert rng_state is not None

        metrics: dict[str, float] | None = None
        seed_verdict_error: Exception | None = None
        evaluation_error: Exception | None = None
        restore_error: Exception | None = None
        try:
            seed_error: Exception | None = None
            try:
                # Keep different rank strata decorrelated while making a given
                # rank's flow/dropout-independent inference draw identical across
                # iterations and exact resumes.  ``torch.manual_seed`` seeds both
                # CPU and CUDA generators; the complete caller state is restored
                # below.
                diagnostic_seed = (
                    0x484F4C4F + int(getattr(self, "gpu_global_rank", 0))
                ) % (2**32)
                random.seed(diagnostic_seed)
                np.random.seed(diagnostic_seed)
                torch.manual_seed(diagnostic_seed)
            except Exception as exc:
                seed_error = exc
            try:
                self._synchronize_training_phase_error(
                    seed_error,
                    operation="fixed BC evaluation deterministic seeding",
                )
            except Exception as exc:
                # A rank may have mutated one or more RNG streams before a
                # local seed failure, and a peer's seed failure is reported by
                # this verdict.  Defer propagation until the caller boundary
                # has been restored on every surviving rank.
                seed_verdict_error = exc

            if seed_verdict_error is None:
                try:
                    metrics = self._get_fixed_bc_eval_metrics_impl(
                        current_iteration,
                        terminal_observation=terminal_observation,
                    )
                except Exception as exc:
                    evaluation_error = exc
        finally:
            try:
                restore_rng_checkpoint_state(
                    rng_state,
                    path="pre_fixed_bc_eval_metrics_rng_state",
                )
            except Exception as exc:
                restore_error = exc

        self._synchronize_training_phase_error(
            restore_error,
            operation="fixed BC evaluation RNG-state restoration",
        )
        if seed_verdict_error is not None:
            raise seed_verdict_error
        self._synchronize_training_phase_error(
            evaluation_error,
            operation="fixed BC evaluation outcome",
        )
        assert metrics is not None
        return metrics

    def _get_fixed_bc_eval_metrics_impl(
        self,
        current_iteration: int,
        *,
        terminal_observation: bool = False,
    ) -> dict[str, float]:
        if self.fixed_bc_eval_num_samples <= 0 or self.fixed_bc_eval_log_interval <= 0:
            return {}
        scheduled_evaluation = (
            current_iteration % self.fixed_bc_eval_log_interval == 0
        )
        if not scheduled_evaluation and not terminal_observation:
            return {}

        local_target = 0
        allocation_error: Exception | None = None
        try:
            local_target = self._fixed_bc_eval_local_target()
        except Exception as exc:
            allocation_error = exc
        self._synchronize_training_phase_error(
            allocation_error,
            operation="fixed BC evaluation allocation validation",
        )
        local_ready = bool(self._fixed_bc_eval_ready or local_target == 0)
        if getattr(self, "is_multi_gpu", False):
            ready = torch.tensor(int(local_ready), device=self.device, dtype=torch.int32)
            ready = self._all_reduce_small_tensor(ready, op=torch.distributed.ReduceOp.MIN)
            if not bool(ready.item()):
                if terminal_observation:
                    raise RuntimeError(
                        "Final fixed-BC observation is unavailable because at least one "
                        "rank has not frozen its exact evaluation stratum."
                    )
                return {}
        elif not local_ready:
            if terminal_observation:
                raise RuntimeError(
                    "Final fixed-BC observation is unavailable because its exact "
                    "evaluation stratum has not been frozen."
                )
            return {}

        squared_error_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        element_count = torch.zeros((), device=self.device, dtype=torch.float64)
        raw_sample_count = torch.zeros((), device=self.device, dtype=torch.float64)
        rank_weight = 1.0
        local_eval_error: Exception | None = None
        try:
            if getattr(self, "is_multi_gpu", False):
                rank_weight = float(self._get_distributed_loss_weight())
            if local_target > 0:
                local_rows = int(self._fixed_bc_eval_dataset["teacher_actions"].shape[0])
                if local_rows != local_target:
                    raise ValueError(
                        "Fixed BC evaluation rank-local dataset does not match its allocation: "
                        f"rank={getattr(self, 'gpu_global_rank', 0)}, expected={local_target}, actual={local_rows}."
                    )
                actor_training = self.actor.training
                normalizer_training = {
                    key: normalizer.training
                    for key, normalizer in self.actor_obs_normalizers.items()
                    if hasattr(normalizer, "training")
                }
                try:
                    self.actor.eval()
                    for normalizer in self.actor_obs_normalizers.values():
                        if hasattr(normalizer, "eval"):
                            normalizer.eval()

                    actor_obs_raw = self._fixed_bc_eval_dataset["actor_obs_raw"].to(self.device)
                    actor_obs = self._normalize_actor_obs(actor_obs_raw, update=False)
                    policy_state = {"actor_obs": actor_obs}
                    if self.actor_perception_key and "actor_perception" in self._fixed_bc_eval_dataset:
                        policy_state[self.actor_perception_key] = self._fixed_bc_eval_dataset[
                            "actor_perception"
                        ].to(self.device)
                    student_actions = self.actor.act_inference(policy_state)
                    teacher_actions = self._fixed_bc_eval_dataset["teacher_actions"].to(self.device)
                    action_error = student_actions - teacher_actions
                finally:
                    self.actor.train(actor_training)
                    for key, normalizer in self.actor_obs_normalizers.items():
                        if hasattr(normalizer, "train") and key in normalizer_training:
                            normalizer.train(normalizer_training[key])

                squared_error_sum = action_error.to(dtype=torch.float64).pow(2).sum()
                element_count = squared_error_sum.new_tensor(float(action_error.numel()))
                raw_sample_count = squared_error_sum.new_tensor(float(teacher_actions.shape[0]))
        except Exception as exc:
            local_eval_error = exc

        # No rank may enter the statistics all-reduce while a peer failed in
        # local actor inference, dataset validation, or mode restoration.
        self._synchronize_training_phase_error(
            local_eval_error,
            operation="fixed BC evaluation local inference",
        )
        weighted_sample_count = raw_sample_count
        rank_strata = raw_sample_count.new_tensor(float(local_target > 0))
        if getattr(self, "is_multi_gpu", False):
            stats = torch.stack(
                [
                    squared_error_sum * rank_weight,
                    element_count * rank_weight,
                    raw_sample_count,
                    raw_sample_count * rank_weight,
                    rank_strata,
                ]
            )
            stats = self._all_reduce_small_tensor(stats, op=torch.distributed.ReduceOp.SUM)
            (
                squared_error_sum,
                element_count,
                raw_sample_count,
                weighted_sample_count,
                rank_strata,
            ) = stats.unbind()
        if float(element_count.item()) <= 0.0:
            raise ValueError("Fixed BC evaluation has no globally weighted action elements.")
        expected_sample_count = int(self.fixed_bc_eval_num_samples)
        actual_sample_count = int(raw_sample_count.item())
        if actual_sample_count != expected_sample_count:
            raise ValueError(
                "Fixed BC evaluation did not realize its exact global sample budget: "
                f"expected={expected_sample_count}, actual={actual_sample_count}."
            )

        return {
            "fixed_bc_mu_mse": float((squared_error_sum / element_count).item()),
            "fixed_bc_num_samples": float(raw_sample_count.item()),
            "fixed_bc_weighted_num_samples": float(weighted_sample_count.item()),
            "fixed_bc_rank_strata": float(rank_strata.item()),
        }

    def _get_zero_input(self):
        """
        Create a dummy (all-zero) input for the actor.

        During training, we cannot use the logic in `self.get_example_obs()`, since it resets environments mid-rollout.
        """
        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        return torch.zeros(1, actor_obs_dim, device=self.device)

    def _get_zero_perception_input(self) -> torch.Tensor | None:
        if not self.actor_perception_key:
            return None
        perception_dim = self.algo_obs_dim_dict[self.actor_perception_key]
        return torch.zeros(1, perception_dim, device=self.device)

    def _setup_storage(self):
        self.storage = RolloutStorage(self.env.num_envs, self.config.num_steps_per_env, device=self.device)
        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        print(f"Registering key: actor_obs with shape: {actor_obs_dim}")
        self.storage.register("actor_obs", shape=(actor_obs_dim,), dtype=torch.float)
        if self.use_symmetry or bool(getattr(self, "dagger_replay_enabled", False)):
            self.storage.register("actor_obs_raw", shape=(actor_obs_dim,), dtype=torch.float)

        critic_obs_dim = self._get_obs_dim(self.critic_obs_keys)
        print(f"Registering key: critic_obs with shape: {critic_obs_dim}")
        self.storage.register("critic_obs", shape=(critic_obs_dim,), dtype=torch.float)
        if self.use_symmetry:
            self.storage.register("critic_obs_raw", shape=(critic_obs_dim,), dtype=torch.float)

        # Register others based on Minibatch structure
        minibatch_keys = [
            ("actions", (self.num_act,), torch.float),
            ("rewards", (1,), torch.float),
            ("dones", (1,), torch.bool),
            ("values", (1,), torch.float),
            ("returns", (1,), torch.float),
            ("advantages", (1,), torch.float),
            ("actions_log_prob", (1,), torch.float),
            ("action_mean", (self.num_act,), torch.float),
            ("action_sigma", (self.num_act,), torch.float),
        ]
        for key, shape, dtype in minibatch_keys:
            self.storage.register(
                key,
                shape=shape,
                dtype=dtype,
                required_on_add=key not in {"returns", "advantages"},
            )
        if self.distill_enabled:
            self.storage.register("teacher_actions", shape=(self.num_act,), dtype=torch.float)
        if self.dagger_enabled:
            if self.use_multi_teacher:
                self.storage.register("teacher_indices", shape=(1,), dtype=torch.long)
            if (
                self.dagger_ignore_episode_initial_steps > 0
                or self._motion_command_supports_runtime_default_pose_prepend_mask()
            ):
                self.storage.register("teacher_bc_mask", shape=(1,), dtype=torch.bool)
        perception_keys = {key for key in [self.actor_perception_key] if key}
        if self._critic_optimizer_objective_enabled() and self.critic_perception_key:
            perception_keys.add(self.critic_perception_key)
        for key in perception_keys:
            self.storage.register(key, shape=(self.algo_obs_dim_dict[key],), dtype=torch.float)
        if self.use_time_gru:
            self.storage.register(
                "actor_gru_hidden",
                shape=(self.actor.perception_time_gru.hidden_dim,),
                dtype=torch.float,
            )
            self.storage.register(
                "critic_gru_hidden",
                shape=(self.critic.perception_time_gru.hidden_dim,),
                dtype=torch.float,
            )

    def _add_rollout_storage_transition(self, storage_values: dict[str, torch.Tensor]) -> None:
        """Store exactly the transition fields declared by the active schema.

        Rollout construction intentionally computes a superset of optional
        values (raw symmetry observations, teacher labels, recurrent state).
        ``RolloutStorage.add`` itself is strict, so select its declared
        transition fields here and let a missing registered value fail before
        any row is mutated.
        """

        required_keys = self.storage.required_on_add_keys
        missing_keys = required_keys.difference(storage_values)
        if missing_keys:
            raise KeyError(
                "PPO rollout transition is missing fields required by storage: "
                f"{sorted(missing_keys)}"
            )
        self.storage.add(
            **{key: storage_values[key] for key in required_keys}
        )

    def _eval_mode(self):
        self.actor.eval()
        self.critic.eval()
        for normalizer in self.actor_obs_normalizers.values():
            normalizer.eval()
        for normalizer in self.critic_obs_normalizers.values():
            normalizer.eval()
        if self.teacher_actor is not None:
            self.teacher_actor.eval()
            for normalizer in self.teacher_actor_obs_normalizers.values():
                normalizer.eval()
        if self.teacher_actors:
            for teacher_actor, normalizers in zip(self.teacher_actors, self.teacher_actor_obs_normalizers_list):
                teacher_actor.eval()
                for normalizer in normalizers.values():
                    normalizer.eval()

    def _train_mode(self):
        self.actor.train()
        for normalizer in self.actor_obs_normalizers.values():
            normalizer.train()
        if self._critic_optimizer_objective_enabled():
            self.critic.train()
            for normalizer in self.critic_obs_normalizers.values():
                normalizer.train()
        else:
            # A frozen critic must be observationally frozen too.  Leaving it
            # in train mode would consume dropout RNG and mutate BatchNorm or
            # empirical-normalizer buffers even though no critic optimizer
            # step is allowed.
            self.critic.eval()
            for normalizer in self.critic_obs_normalizers.values():
                normalizer.eval()
        if self.teacher_actor is not None:
            self.teacher_actor.eval()
            for normalizer in self.teacher_actor_obs_normalizers.values():
                normalizer.eval()
        if self.teacher_actors:
            for teacher_actor, normalizers in zip(self.teacher_actors, self.teacher_actor_obs_normalizers_list):
                teacher_actor.eval()
                for normalizer in normalizers.values():
                    normalizer.eval()

    def _distributed_barrier(self) -> None:
        if not self.is_multi_gpu or not torch.distributed.is_initialized():
            return
        if os.environ.get("HOLOSOMA_GLOO_BARRIER", "").lower() in ("1", "true", "yes", "on"):
            gloo_group = self._setup_gloo_barrier_group()
            if gloo_group is not None:
                torch.distributed.barrier(group=gloo_group)
                return
        try:
            torch.distributed.barrier(device_ids=[int(self.gpu_local_rank)])
        except TypeError:
            torch.distributed.barrier()

    def _save_checkpoint_with_distributed_outcome(
        self,
        path: str,
        *,
        next_iteration: int,
        allow_tripped_fixed_bc_guard: bool = False,
        upload: bool = True,
    ) -> None:
        """Keep the complete save/outcome protocol outside training RNG."""

        checkpoint_rng_state = None
        capture_error: Exception | None = None
        try:
            checkpoint_rng_state = capture_rng_checkpoint_state()
        except Exception as exc:
            capture_error = exc
        self._synchronize_training_phase_error(
            capture_error,
            operation="distributed checkpoint RNG-state capture",
        )
        assert checkpoint_rng_state is not None

        save_error: Exception | None = None
        try:
            self._save_checkpoint_with_distributed_outcome_impl(
                path,
                next_iteration=next_iteration,
                allow_tripped_fixed_bc_guard=allow_tripped_fixed_bc_guard,
                upload=upload,
            )
        except Exception as exc:
            save_error = exc

        restore_error: Exception | None = None
        try:
            restore_rng_checkpoint_state(
                checkpoint_rng_state,
                path="pre_distributed_checkpoint_protocol_rng_state",
            )
        except Exception as exc:
            restore_error = exc
        self._synchronize_training_phase_error(
            restore_error,
            operation="distributed checkpoint RNG-state restoration",
        )
        self._synchronize_training_phase_error(
            save_error,
            operation="distributed checkpoint protocol outcome",
        )

    def _save_checkpoint_with_distributed_outcome_impl(
        self,
        path: str,
        *,
        next_iteration: int,
        allow_tripped_fixed_bc_guard: bool = False,
        upload: bool = True,
    ) -> None:
        """Publish a checkpoint and make every rank's failure visible to all ranks."""

        if not self.is_multi_gpu or not torch.distributed.is_initialized():
            if allow_tripped_fixed_bc_guard:
                if upload:
                    self.save(
                        path,
                        next_iteration=next_iteration,
                        allow_tripped_fixed_bc_guard=True,
                    )
                else:
                    self.save(
                        path,
                        next_iteration=next_iteration,
                        allow_tripped_fixed_bc_guard=True,
                        upload=False,
                    )
            else:
                if upload:
                    self.save(path, next_iteration=next_iteration)
                else:
                    self.save(path, next_iteration=next_iteration, upload=False)
            return

        local_error: Exception | None = None
        try:
            # save() first gathers all-rank environment/fixed-BC state, then
            # only rank zero serializes and uploads the artifact.  Catch that
            # rank-zero I/O phase as well as post-collective failures on any
            # peer, so no worker can leave the others at the next barrier.
            if allow_tripped_fixed_bc_guard:
                if upload:
                    self.save(
                        path,
                        next_iteration=next_iteration,
                        allow_tripped_fixed_bc_guard=True,
                    )
                else:
                    self.save(
                        path,
                        next_iteration=next_iteration,
                        allow_tripped_fixed_bc_guard=True,
                        upload=False,
                    )
            else:
                if upload:
                    self.save(path, next_iteration=next_iteration)
                else:
                    self.save(path, next_iteration=next_iteration, upload=False)
        except Exception as exc:
            local_error = exc

        gloo_group = self._setup_gloo_barrier_group()
        group_error = (
            RuntimeError("Distributed checkpoint outcome requires an all-rank Gloo process group.")
            if gloo_group is None
            else None
        )
        self._synchronize_training_phase_error(
            group_error,
            operation="distributed checkpoint outcome group validation",
        )
        assert gloo_group is not None
        group_world_size = torch.distributed.get_world_size(group=gloo_group)
        group_rank = torch.distributed.get_rank(group=gloo_group)
        expected_world_size = int(getattr(self, "gpu_world_size", 1) or 1)
        expected_rank = int(getattr(self, "gpu_global_rank", 0) or 0)
        topology_error = None
        if group_world_size != expected_world_size or group_rank != expected_rank:
            topology_error = RuntimeError(
                "Distributed checkpoint outcome Gloo topology disagrees with PPO topology: "
                f"group={group_rank}/{group_world_size}, "
                f"PPO={expected_rank}/{expected_world_size}."
            )
        self._synchronize_training_phase_error(
            topology_error,
            operation="distributed checkpoint outcome topology validation",
        )
        local_result = {
            "rank": group_rank,
            "error": (
                None
                if local_error is None
                else f"{type(local_error).__name__}: {local_error}"
            ),
        }
        gathered_results: list[dict[str, Any] | None] = [None] * group_world_size
        torch.distributed.all_gather_object(
            gathered_results,
            local_result,
            group=gloo_group,
        )
        envelope_problems = []
        failures = []
        for expected_rank, result in enumerate(gathered_results):
            if not isinstance(result, dict) or result.get("rank") != expected_rank:
                envelope_problems.append(
                    f"slot={expected_rank}: malformed payload/reported_rank="
                    f"{result.get('rank') if isinstance(result, dict) else None}"
                )
            elif result.get("error") is not None:
                failures.append(f"rank={expected_rank}: {result['error']}")
        if envelope_problems:
            raise RuntimeError(
                "Distributed checkpoint outcome returned malformed all-rank envelopes: "
                + "; ".join(envelope_problems)
            )
        if failures:
            failure = RuntimeError(
                f"Distributed checkpoint publication failed for {path}: "
                + "; ".join(failures)
            )
            if local_error is not None:
                raise failure from local_error
            raise failure

    def _sync_iteration_boundary(self, label: str) -> None:
        if os.environ.get("HOLOSOMA_SYNC_EACH_ITERATION", "").lower() not in ("1", "true", "yes", "on"):
            return
        if not self.is_multi_gpu or not torch.distributed.is_initialized():
            return
        self._debug_training_phase("iteration_boundary_sync_begin", label=label)
        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self._distributed_barrier()
        self._debug_training_phase("iteration_boundary_sync_finished", label=label)

    def _gloo_small_collectives_enabled(self) -> bool:
        return os.environ.get("HOLOSOMA_GLOO_SMALL_COLLECTIVES", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _hierarchical_small_collectives_enabled(self) -> bool:
        if hasattr(self, "_hierarchical_small_collectives"):
            return bool(self._hierarchical_small_collectives)
        # Legacy/test objects that bypass ``__init__`` still receive strict
        # parsing, while fully initialized runs use the frozen setup value.
        return self._strict_environment_bool(
            "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES",
            default=False,
        )

    @staticmethod
    def _hierarchical_small_control_dtype_supported(tensor: torch.Tensor) -> bool:
        # Keep bool on the established flat path until every required Gloo
        # ReduceOp has an explicit backend-compatibility test.  Complex tensors
        # are likewise not integral even though ``is_floating_point`` is false.
        # Production verdicts use int32.  int64 is also covered by the real
        # multi-process Gloo regression.  Keep narrower integer types on the
        # established flat path until their SUM overflow/backend semantics are
        # explicitly required and tested.
        return tensor.dtype in (torch.int32, torch.int64)

    def _hierarchical_all_reduce_small_cpu_tensor(
        self,
        cpu_tensor: torch.Tensor,
        *,
        op: torch.distributed.ReduceOp,
    ) -> torch.Tensor:
        """Reduce a non-floating control tensor through local Gloo leaders.

        Every rank in a node executes local reduce -> local broadcast. Exactly
        one leader per node executes the inter-node all-reduce between those two
        local collectives. Once the first reduce begins there is deliberately
        no fallback path: switching a subset of ranks to the flat group would
        violate collective order and can deadlock both process groups.
        """

        if cpu_tensor.device.type != "cpu":
            raise ValueError("Hierarchical small collectives require a CPU staging tensor.")
        if not self._hierarchical_small_control_dtype_supported(cpu_tensor):
            raise TypeError(
                "Hierarchical small collectives are restricted to supported integral control tensors."
            )
        if op not in (
            torch.distributed.ReduceOp.SUM,
            torch.distributed.ReduceOp.MIN,
            torch.distributed.ReduceOp.MAX,
        ):
            raise ValueError(
                "Hierarchical small collectives support only SUM, MIN, and MAX reductions."
            )
        if not getattr(self, "_hierarchical_grad_reduce_ready", False) or not getattr(
            self,
            "_hierarchical_grad_reduce_available",
            False,
        ):
            raise RuntimeError(
                "Hierarchical small collective groups were not initialized at PPO.setup()."
            )
        local_group = getattr(self, "_hierarchical_local_barrier_group", None)
        leader_group = getattr(self, "_hierarchical_leader_gloo_group", None)
        if local_group is None or leader_group is None:
            raise RuntimeError(
                "Hierarchical small collectives require both node-local and leader Gloo groups."
            )

        torch.distributed.reduce(
            cpu_tensor,
            dst=self._hierarchical_local_leader_rank,
            op=op,
            group=local_group,
        )
        if self._hierarchical_is_leader_rank:
            torch.distributed.all_reduce(
                cpu_tensor,
                op=op,
                group=leader_group,
            )
        # Broadcast is the local rendezvous after the leader all-reduce; an
        # additional local barrier would add latency without changing order.
        torch.distributed.broadcast(
            cpu_tensor,
            src=self._hierarchical_local_leader_rank,
            group=local_group,
        )
        return cpu_tensor

    def _all_reduce_small_tensor(
        self,
        tensor: torch.Tensor,
        *,
        op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM,
    ) -> torch.Tensor:
        if self._gloo_small_collectives_enabled():
            cpu_tensor = tensor.detach().cpu()
            hierarchical_integral_control = (
                self._hierarchical_small_collectives_enabled()
                and self._hierarchical_small_control_dtype_supported(tensor)
            )
            if hierarchical_integral_control:
                cpu_tensor = self._hierarchical_all_reduce_small_cpu_tensor(
                    cpu_tensor,
                    op=op,
                )
                return cpu_tensor.to(device=tensor.device, dtype=tensor.dtype)
            gloo_group = self._setup_gloo_barrier_group()
            if gloo_group is not None:
                torch.distributed.all_reduce(cpu_tensor, op=op, group=gloo_group)
                return cpu_tensor.to(device=tensor.device, dtype=tensor.dtype)
        # With the NCCL default group, make the small collective a real device
        # boundary.  ``async_op=False`` only makes enqueueing synchronous with
        # the host; without an explicit CUDA wait, one rank can enter the next
        # collective while another still has producer/collective work pending.
        # This mirrors the verified flat-gradient NCCL boundary below.
        if tensor.device.type == "cuda":
            torch.cuda.synchronize(tensor.device)
        torch.distributed.all_reduce(tensor, op=op)
        if tensor.device.type == "cuda":
            torch.cuda.synchronize(tensor.device)
        return tensor

    def _broadcast_tensor(self, tensor: torch.Tensor, *, src: int = 0) -> None:
        if self._gloo_small_collectives_enabled():
            gloo_group = self._setup_gloo_barrier_group()
            if gloo_group is not None:
                cpu_tensor = tensor.detach().cpu()
                torch.distributed.broadcast(cpu_tensor, src=src, group=gloo_group)
                tensor.detach().copy_(cpu_tensor.to(device=tensor.device, dtype=tensor.dtype))
                return
        torch.distributed.broadcast(tensor, src=src)

    def _synchronize_curriculum_metrics(self):
        if not self._curriculum_state_sync_enabled():
            return
        env = self._unwrap_env()
        command_manager = getattr(env, "command_manager", None)
        motion_command = command_manager.get_state("motion_command") if command_manager is not None else None
        # Rank-local AS has heterogeneous tensor shapes and synchronizes only
        # duplicate clip rows through an object collective.  Never let that
        # pickle path fall back to the NCCL default group: besides staging all
        # bytes through the current GPU, NCCL object collectives are fragile on
        # large multi-node worlds.  Every rank with a rank-local shard enters
        # this branch, so Gloo group creation has one consistent global order.
        if self.gpu_world_size > 1 and getattr(motion_command, "_rank_local_shard_metadata", None) is not None:
            gloo_group = self._setup_gloo_barrier_group()
            if gloo_group is None:
                raise RuntimeError("Rank-local AS synchronization requires an all-rank Gloo process group.")
            env.synchronize_curriculum_state(
                device="cpu",
                world_size=self.gpu_world_size,
                process_group=gloo_group,
            )
            return
        if self._gloo_small_collectives_enabled():
            gloo_group = self._setup_gloo_barrier_group()
            if gloo_group is not None:
                env.synchronize_curriculum_state(
                    device="cpu",
                    world_size=self.gpu_world_size,
                    process_group=gloo_group,
                )
                return
        env.synchronize_curriculum_state(device=self.device, world_size=self.gpu_world_size)

    def _curriculum_state_sync_enabled(self) -> bool:
        env = self._unwrap_env()
        return self.has_curricula_enabled() or bool(getattr(env, "curriculum_state_sync_enabled", False))

    def _curriculum_state_required_for_resume(self) -> bool:
        """Whether omitting environment state would change the resumed objective.

        Rank-local AS shards must not average sampler tensors across ranks, but
        every rank still has sampler/exposure/failure state that must be restored.
        Keep that checkpoint requirement separate from synchronization policy.
        """
        env = self._unwrap_env()
        if self.has_curricula_enabled() or bool(getattr(env, "curriculum_state_checkpoint_required", False)):
            return True
        command_manager = getattr(env, "command_manager", None)
        if command_manager is None:
            return False
        motion_command = command_manager.get_state("motion_command")
        if motion_command is None:
            return False
        if getattr(motion_command, "adaptive_timesteps_sampler", None) is not None:
            return True
        return any(
            isinstance(getattr(motion_command, name, None), torch.Tensor)
            for name in ("_clip_success_counts", "_clip_total_counts")
        )

    def _reset_recurrent_rollout_state(self) -> None:
        """Make an all-environment reset equivalent to a fresh process actor."""

        models = [getattr(self, "actor", None), getattr(self, "critic", None)]
        teacher_actor = getattr(self, "teacher_actor", None)
        if teacher_actor is not None:
            models.append(teacher_actor)
        models.extend(getattr(self, "teacher_actors", ()) or ())
        for model in models:
            recurrent_encoder = getattr(model, "perception_time_gru", None)
            if recurrent_encoder is not None:
                recurrent_encoder.hidden = None

    def _reset_rollout_stream_at_canonical_boundary(
        self,
        *,
        current_iteration: int,
        total_iterations: int,
    ) -> dict[str, torch.Tensor]:
        """Start one reproducible rollout stream from checkpointed RNG/state.

        A simulator snapshot, observation history, DAgger episode age, and
        recurrent hidden state are not part of the compact training
        checkpoint.  Consequently a resume cannot honestly continue the
        pre-checkpoint episode.  New checkpoints instead define a canonical
        all-environment reset boundary.  Uninterrupted training performs this
        same reset immediately after publishing every non-final checkpoint,
        so its next rollout consumes the same RNG and curriculum state as a
        process restarted from that checkpoint.
        """

        self._sync_training_curriculum_state(
            current_iteration=current_iteration,
            total_iterations=total_iterations,
        )
        if self._curriculum_state_sync_enabled():
            self._synchronize_curriculum_metrics()
        self._reset_recurrent_rollout_state()
        logger.info(
            "PPO canonical rollout reset starting for iteration {}.",
            current_iteration,
        )
        reset_at_boundary = getattr(
            self.env,
            "reset_all_at_checkpoint_boundary",
            self.env.reset_all,
        )
        obs_dict = reset_at_boundary()
        logger.info(
            "PPO canonical rollout reset finished with obs keys: {}.",
            sorted(obs_dict.keys()),
        )

        self._dagger_episode_step_buf = (
            torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long)
            if getattr(self, "dagger_ignore_episode_initial_steps", 0) > 0
            else None
        )
        if bool(getattr(self.config, "init_at_random_ep_len", False)):
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )
        for obs_key in obs_dict:
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
        return obs_dict

    def learn(self):
        logger.info("Entering PPO.learn at iteration {}.", self.current_learning_iteration)
        run_end_iteration = int(self.config.num_learning_iterations)
        if self.current_learning_iteration > run_end_iteration:
            raise ValueError(
                "Checkpoint next iteration exceeds algo.config.num_learning_iterations: "
                f"next_iter={self.current_learning_iteration}, target={run_end_iteration}."
            )
        if self.current_learning_iteration == run_end_iteration:
            logger.info(
                "PPO target iteration {} is already complete; learn() is a side-effect-free no-op.",
                run_end_iteration,
            )
            return
        # A terminal observation authenticates only the target that produced
        # it.  Clear it before a same-process target extension so an
        # intermediate checkpoint can never inherit stale final metadata.
        self._terminal_fixed_bc_eval_state = None
        if self._should_export_onnx():
            # Reject an unrepresentable final policy before changing module
            # mode, resetting the environment, or collecting any rollout.
            # This applies equally to pure BC and PPO-containing objectives.
            self._validate_actor_onnx_compatibility(training_preflight=True)
        self._validate_future_dagger_bc_mask_signal(
            start_iteration=self.current_learning_iteration,
            end_iteration=run_end_iteration,
        )
        self._train_mode()
        self._prepare_rollout_objective_for_iteration(
            self.current_learning_iteration
        )
        obs_dict = self._reset_rollout_stream_at_canonical_boundary(
            current_iteration=self.current_learning_iteration,
            total_iterations=run_end_iteration,
        )
        rollout_objective_prepared_iteration: int | None = self.current_learning_iteration
        curriculum_synchronized_iteration: int | None = self.current_learning_iteration
        self._reset_step_timing()
        logger.info("PPO.learn initial obs transfer to {} finished.", self.device)

        debug_heartbeat = os.environ.get("HOLOSOMA_DEBUG_HEARTBEAT", "").lower() not in ("", "0", "false", "no")
        start_iteration = self.current_learning_iteration
        last_saved_iteration: int | None = None
        final_onnx_sha256: str | None = None
        fixed_bc_eval_configured = bool(
            getattr(self, "dagger_enabled", False)
            and int(getattr(self, "fixed_bc_eval_num_samples", 0) or 0) > 0
            and int(getattr(self, "fixed_bc_eval_log_interval", 0) or 0) > 0
        )
        for it in range(
            self.current_learning_iteration,
            run_end_iteration,
        ):
            self.current_learning_iteration = it
            if it > start_iteration:
                self._sync_iteration_boundary("iteration_start")
            self._reset_step_timing()
            if debug_heartbeat:
                logger.info("Heartbeat: iter {} starting teacher/curriculum updates", it)
            # The coefficient determines whether rollout actions are sampled or
            # deterministic, so it must be derived before collecting this
            # iteration's transitions (and not first changed during update).
            if it != rollout_objective_prepared_iteration:
                self._prepare_rollout_objective_for_iteration(it)
            rollout_objective_prepared_iteration = it
            self._sync_training_curriculum_state(
                current_iteration=it,
                total_iterations=run_end_iteration,
            )
            if debug_heartbeat:
                logger.info("Heartbeat: iter {} finished teacher/curriculum state update", it)

            # Synchronize curriculum metrics across GPUs before rollout
            # The first rollout's curriculum state was already synchronized
            # immediately before reset_all().  WBT reset_all() restores that
            # exact state after its zero-action warm-up step, so repeating the
            # rank-local object collective here is redundant.  Later
            # iterations still synchronize the AS updates collected by the
            # preceding rollout.
            if (
                it != curriculum_synchronized_iteration
                and self._curriculum_state_sync_enabled()
            ):
                if debug_heartbeat:
                    logger.info("Heartbeat: iter {} starting curriculum metric sync", it)
                self._synchronize_curriculum_metrics()
                if debug_heartbeat:
                    logger.info("Heartbeat: iter {} finished curriculum metric sync", it)
            curriculum_synchronized_iteration = None

            if debug_heartbeat:
                logger.info("Heartbeat: iter {} starting rollout", it)
            if self.algo_timing.enabled:
                with self.algo_timing.record("iteration/rollout"):
                    with self.logging_helper.record_collection_time():
                        obs_dict = self._rollout_step(obs_dict)
            else:
                with self.logging_helper.record_collection_time():
                    obs_dict = self._rollout_step(obs_dict)
            if debug_heartbeat:
                logger.info("Heartbeat: iter {} finished rollout", it)

            if debug_heartbeat:
                logger.info("Heartbeat: iter {} starting training_step", it)
            if self.algo_timing.enabled:
                with self.algo_timing.record("iteration/training_step"):
                    with self.logging_helper.record_learn_time():
                        loss_dict = self._training_step()
            else:
                with self.logging_helper.record_learn_time():
                    loss_dict = self._training_step()
            if debug_heartbeat:
                logger.info("Heartbeat: iter {} finished training_step", it)
            self._capture_step_timing()
            self._emit_step_timing_summary(it)

            if self.is_multi_gpu:
                metrics_group = self._setup_gloo_barrier_group()
                if metrics_group is None:
                    raise RuntimeError("Distributed metric synchronization requires an all-rank Gloo process group.")
                loss_dict = self.logging_helper.synchronize_distributed_metrics(
                    loss_dict,
                    loss_weight=self._get_distributed_loss_weight(),
                    process_group=metrics_group,
                )

            # Fixed-BC evaluation is a distributed, rank-shard-weighted
            # diagnostic.  On scheduled or forced-terminal observations every
            # rank must enter its readiness/statistics collectives even though
            # only rank zero emits logger records.  Off-cadence iterations have
            # no evaluation state transition and enter none of that protocol.
            fixed_bc_eval_metrics = {}
            terminal_fixed_bc_requested = bool(
                fixed_bc_eval_configured and it == run_end_iteration - 1
            )
            scheduled_fixed_bc_evaluation = bool(
                fixed_bc_eval_configured
                and it % int(self.fixed_bc_eval_log_interval) == 0
            )
            fixed_bc_evaluation_due = bool(
                scheduled_fixed_bc_evaluation or terminal_fixed_bc_requested
            )
            if fixed_bc_evaluation_due:
                fixed_bc_eval_metrics = self._get_fixed_bc_eval_metrics(
                    current_iteration=it,
                    terminal_observation=terminal_fixed_bc_requested,
                )
            # The direct guard method retains its fail-closed off-interval
            # validation for external callers, but learn() constructs the
            # cadence locally and has no metrics to validate between scheduled
            # observations.  Avoid an otherwise unconditional all-rank verdict
            # on every off-grid iteration.
            fixed_bc_guard_tripped = False
            if scheduled_fixed_bc_evaluation:
                fixed_bc_guard_tripped = self._update_fixed_bc_guard(
                    current_iteration=it,
                    metrics=fixed_bc_eval_metrics,
                )
            if fixed_bc_eval_metrics:
                fixed_bc_eval_metrics = dict(fixed_bc_eval_metrics)
                fixed_bc_eval_metrics.update(self._fixed_bc_guard_log_metrics())

            next_iteration = it + 1
            if fixed_bc_guard_tripped:
                # Publish the non-resumable diagnostic before terminal logging
                # so a W&B/TensorBoard failure cannot erase the primary failure
                # artifact.  Then make a best effort to expose the final MSE,
                # threshold, and counter=patience in W&B before all ranks exit.
                diagnostic_path = self._publish_fixed_bc_guard_trip_diagnostic(
                    next_iteration=next_iteration
                )
                terminal_logging_error: Exception | None = None
                try:
                    self._post_epoch_logging_preserving_rng(
                        it,
                        loss_dict,
                        fixed_bc_eval_metrics=fixed_bc_eval_metrics,
                    )
                except Exception as exc:
                    terminal_logging_error = exc
                guard_error = self._fixed_bc_guard_trip_error(
                    diagnostic_path=diagnostic_path,
                    logging_error=terminal_logging_error,
                )
                if terminal_logging_error is not None:
                    raise guard_error from terminal_logging_error
                raise guard_error

            terminal_fixed_bc_state: dict[str, Any] | None = None
            if terminal_fixed_bc_requested:
                terminal_fixed_bc_state = (
                    self._build_terminal_fixed_bc_eval_state_preserving_rng(
                        completed_iteration=it,
                        metrics=fixed_bc_eval_metrics,
                        scheduled_evaluation=scheduled_fixed_bc_evaluation,
                    )
                )
                self._terminal_fixed_bc_eval_state = terminal_fixed_bc_state
                fixed_bc_eval_metrics.update(
                    {
                        "fixed_bc_final_mu_mse": terminal_fixed_bc_state[
                            "fixed_bc_mu_mse"
                        ],
                        "fixed_bc_terminal_observation": 1.0,
                        "fixed_bc_scheduled_evaluation": float(
                            terminal_fixed_bc_state["scheduled_evaluation"]
                        ),
                        "fixed_bc_guard_applied": float(
                            terminal_fixed_bc_state["guard_applied"]
                        ),
                    }
                )
            # Every rank enters the logging outcome verdict.  Only rank zero
            # performs I/O, but its W&B/TensorBoard/rich failure must be made
            # visible before peers can enter checkpoint or guard barriers.
            self._post_epoch_logging_preserving_rng(
                it,
                loss_dict,
                fixed_bc_eval_metrics=fixed_bc_eval_metrics,
            )

            should_save_checkpoint = next_iteration % self.config.save_interval == 0

            if should_save_checkpoint:
                self._distributed_barrier()
                # save() contains an all-rank environment-state collective;
                # every rank must participate even though only rank zero writes.
                checkpoint_path = os.path.join(
                    self.log_dir,
                    f"model_{next_iteration:05d}.pt",
                )
                onnx_path = os.path.join(self.log_dir, f"model_{next_iteration:05d}.onnx")
                if self._should_export_onnx():
                    pair_onnx_sha256 = (
                        self._save_policy_artifact_pair_with_distributed_outcome(
                            checkpoint_path=checkpoint_path,
                            onnx_path=onnx_path,
                            next_iteration=next_iteration,
                        )
                    )
                    if next_iteration == run_end_iteration:
                        final_onnx_sha256 = pair_onnx_sha256
                else:
                    self._save_checkpoint_with_distributed_outcome(
                        checkpoint_path,
                        next_iteration=next_iteration,
                    )
                self._distributed_barrier()
                last_saved_iteration = next_iteration
                if (
                    next_iteration < run_end_iteration
                    and self._reset_rollout_at_checkpoint_enabled()
                ):
                    # The checkpoint's RNG and pending curriculum state were
                    # captured immediately before this reset.  Performing the
                    # same boundary in the uninterrupted process makes a
                    # restart from that artifact stochastic-equivalent.
                    self.current_learning_iteration = next_iteration
                    self._prepare_rollout_objective_for_iteration(next_iteration)
                    obs_dict = self._reset_rollout_stream_at_canonical_boundary(
                        current_iteration=next_iteration,
                        total_iterations=run_end_iteration,
                    )
                    rollout_objective_prepared_iteration = next_iteration
                    curriculum_synchronized_iteration = next_iteration
                elif next_iteration < run_end_iteration and self.is_main_process:
                    logger.info(
                        "Checkpoint {} published without resetting the live rollout; "
                        "subsequent training continues the current episodes.",
                        next_iteration,
                    )

            # Emit one authoritative, machine-readable progress marker per
            # node.  Global rank zero is insufficient for a node-by-node
            # launcher monitor, while logging from all workers is needlessly
            # noisy.  The rank-visible launcher rewrites LOCAL_RANK to zero in
            # every process, so use its preserved topology rank when present.
            if self._is_node_local_main_process():
                print(f"HOLOSOMA_PROGRESS completed_iteration={next_iteration}", flush=True)

        self.current_learning_iteration = run_end_iteration
        # The interval checkpoint above is already the authoritative final
        # artifact when the target is divisible by save_interval.  Rewriting
        # that same path here creates a truncation window and duplicate W&B
        # registration without adding state, so only publish a missing final.
        if last_saved_iteration != run_end_iteration:
            self._distributed_barrier()
            checkpoint_path = os.path.join(
                self.log_dir,
                f"model_{run_end_iteration:05d}.pt",
            )
            onnx_path = os.path.join(
                self.log_dir,
                f"model_{run_end_iteration:05d}.onnx",
            )
            if self._should_export_onnx():
                final_onnx_sha256 = (
                    self._save_policy_artifact_pair_with_distributed_outcome(
                        checkpoint_path=checkpoint_path,
                        onnx_path=onnx_path,
                        next_iteration=run_end_iteration,
                    )
                )
            else:
                self._save_checkpoint_with_distributed_outcome(
                    checkpoint_path,
                    next_iteration=run_end_iteration,
                )
            self._distributed_barrier()
        final_onnx_path = os.path.join(
            self.log_dir,
            f"model_{run_end_iteration:05d}.onnx",
        )
        if self._should_export_onnx() and final_onnx_sha256 is None:
            raise RuntimeError(
                "Final ONNX export was requested but no authenticated artifact digest was produced."
            )
        if self._is_node_local_main_process():
            terminal_fixed_bc_state = self._terminal_fixed_bc_eval_state
            if terminal_fixed_bc_state is not None:
                # Emit terminal success only after the final checkpoint and
                # every requested deployment artifact have completed.  This
                # prevents a launcher from accepting a metric whose policy
                # artifact subsequently failed publication.
                terminal_within_threshold = terminal_fixed_bc_state[
                    "fixed_bc_terminal_within_threshold"
                ]
                terminal_state_sha256 = (
                    self._terminal_fixed_bc_eval_state_sha256(
                        terminal_fixed_bc_state
                    )
                )
                print(
                    "HOLOSOMA_FINAL_FIXED_BC_EVAL "
                    f"completed_iteration={terminal_fixed_bc_state['completed_iteration']} "
                    f"scheduled_evaluation={int(terminal_fixed_bc_state['scheduled_evaluation'])} "
                    f"guard_applied={int(terminal_fixed_bc_state['guard_applied'])} "
                    "within_threshold="
                    f"{('disabled' if terminal_within_threshold is None else int(terminal_within_threshold))} "
                    f"fixed_bc_mu_mse={terminal_fixed_bc_state['fixed_bc_mu_mse']:.17g} "
                    f"fixed_bc_num_samples={terminal_fixed_bc_state['fixed_bc_num_samples']} "
                    f"terminal_state_sha256={terminal_state_sha256}",
                    flush=True,
                )
            onnx_fields = (
                f" onnx={final_onnx_path} onnx_sha256={final_onnx_sha256}"
                if final_onnx_sha256 is not None
                else ""
            )
            print(
                "HOLOSOMA_RUN_COMPLETE "
                f"target_iteration={run_end_iteration} "
                f"checkpoint={os.path.join(self.log_dir, f'model_{run_end_iteration:05d}.pt')}"
                f"{onnx_fields}",
                flush=True,
            )

    @staticmethod
    def _config_field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    def _actor_motion_config(self) -> Any:
        experiment_config = getattr(self, "_experiment_config", None)
        command_config = self._config_field(experiment_config, "command")
        setup_terms = self._config_field(command_config, "setup_terms")
        motion_term = self._config_field(setup_terms, "motion_command")
        motion_params = self._config_field(motion_term, "params")
        return self._config_field(motion_params, "motion_config")

    def _actor_sparse_root_command_mode(self) -> str:
        motion_config = self._actor_motion_config()
        sparse_mode_raw = self._config_field(
            motion_config,
            "contact_aware_sparse_root_command_mode",
            "tracking_error",
        )
        if not isinstance(sparse_mode_raw, str):
            raise ValueError(
                "contact_aware_sparse_root_command_mode must be a string when ONNX export "
                f"is required; got {sparse_mode_raw!r}."
            )
        sparse_mode = sparse_mode_raw.strip().lower().replace("-", "_")
        if sparse_mode in {"tracking", "default", "robot_tracking_error"}:
            return "tracking_error"
        return sparse_mode

    def _actor_zero_root_command_when_drop_active(self) -> bool:
        motion_config = self._actor_motion_config()
        value = self._config_field(
            motion_config,
            "zero_root_command_when_drop_active",
            False,
        )
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(
                "zero_root_command_when_drop_active must be a boolean when ONNX "
                f"export is required; got {value!r}."
            )
        return bool(value)

    def _actor_rolling_reference_delta_parameters(self) -> tuple[int, float]:
        motion_config = self._actor_motion_config()
        lookahead = self._config_field(
            motion_config,
            "contact_aware_sparse_root_segment_steps",
            30,
        )
        zero_yaw_threshold_deg = self._config_field(
            motion_config,
            "contact_aware_sparse_root_zero_yaw_threshold_deg",
            0.0,
        )
        contract, _ = _rolling_reference_delta_deployment_contract(
            lookahead_motion_frames=lookahead,
            zero_yaw_threshold_deg=zero_yaw_threshold_deg,
            zero_root_command_when_drop_active=(
                self._actor_zero_root_command_when_drop_active()
            ),
        )
        return (
            int(contract["lookahead_motion_frames"]),
            float(contract["zero_yaw_threshold_deg"]),
        )

    def _checkpoint_metadata(self, iteration: int | None = None) -> dict[str, Any]:
        metadata = super()._checkpoint_metadata(iteration=iteration)
        if self._actor_sparse_root_command_mode() == "precomputed_turn_then_forward":
            contract, contract_sha256 = (
                _precomputed_turn_then_forward_deployment_contract(
                    zero_root_command_when_drop_active=(
                        self._actor_zero_root_command_when_drop_active()
                    )
                )
            )
            metadata[_PRECOMPUTED_COMMAND_CONTRACT_KEY] = contract
            metadata[_PRECOMPUTED_COMMAND_CONTRACT_SHA256_KEY] = contract_sha256
        elif self._actor_sparse_root_command_mode() == "rolling_reference_delta":
            lookahead, zero_yaw_threshold_deg = (
                self._actor_rolling_reference_delta_parameters()
            )
            contract, contract_sha256 = (
                _rolling_reference_delta_deployment_contract(
                    lookahead_motion_frames=lookahead,
                    zero_yaw_threshold_deg=zero_yaw_threshold_deg,
                    zero_root_command_when_drop_active=(
                        self._actor_zero_root_command_when_drop_active()
                    ),
                )
            )
            metadata[_ROLLING_REFERENCE_DELTA_CONTRACT_KEY] = contract
            metadata[_ROLLING_REFERENCE_DELTA_CONTRACT_SHA256_KEY] = (
                contract_sha256
            )
        return metadata

    def _should_export_onnx(self) -> bool:
        if getattr(self, "_experiment_config", None) is None:
            return True
        return bool(getattr(self._experiment_config.training, "export_onnx", True))

    def _validate_actor_onnx_compatibility(
        self,
        *,
        training_preflight: bool = False,
    ) -> None:
        """Fail before work when the Python inference policy cannot be exported faithfully."""

        forbidden_skip_uploads = [
            name
            for name in (
                "HOLOSOMA_SKIP_WANDB_FILE_UPLOAD",
                "HOLOSOMA_SKIP_WANDB_CHECKPOINT_UPLOAD",
            )
            if os.environ.get(name, "").strip().lower()
            in {"1", "true", "yes", "on"}
        ]
        if forbidden_skip_uploads:
            raise ValueError(
                "ONNX-required training cannot disable checkpoint publication via "
                f"{forbidden_skip_uploads}."
            )

        if bool(getattr(self, "use_time_gru", False)):
            if training_preflight:
                raise ValueError(
                    "time_gru training cannot require ONNX export: recurrent hidden state is not represented as "
                    "explicit ONNX input/output, so the final artifact would fail after the full training run. "
                    "Set --training.export-onnx False before starting recurrent training."
                )
            raise ValueError(
                "time_gru policy export is unsupported: recurrent hidden state is process-local Python "
                "state rather than an explicit ONNX input/output. Export would not preserve reset or "
                "batch semantics."
            )

        actor_impl = getattr(
            getattr(getattr(self, "actor", None), "actor_module", None),
            "module",
            None,
        )
        inference_noise_std = float(
            getattr(actor_impl, "inference_noise_std", 0.0) or 0.0
        )
        if self._actor_uses_flow_matching() and inference_noise_std > 0.0:
            if training_preflight:
                raise ValueError(
                    "Flow training cannot require ONNX export while flow_inference_noise_std is positive: "
                    "export uses a deterministic zero-noise initial condition, so the final artifact would "
                    "represent a different policy from Python stochastic inference. Set "
                    "flow_inference_noise_std=0 or --training.export-onnx False before training."
                )
            raise ValueError(
                "ONNX export requires flow_inference_noise_std=0: export uses a deterministic zero-noise "
                "initial condition and would otherwise serialize a different policy from Python inference."
            )

        # The ONNX graph exports only the actor network.  Contact-aware sparse
        # root commands are reconstructed by ``holosoma_inference`` from the
        # authenticated experiment metadata, so requiring an artifact also
        # requires that the deployment runtime implement the configured
        # command semantics.  ``t1_aligned_segment`` is currently training-only;
        # without this preflight a long run can finish successfully and produce
        # an artifact that the strict inference contract must reject.
        sparse_mode = self._actor_sparse_root_command_mode()
        if sparse_mode not in {
            "tracking_error",
            "rolling_reference_delta",
            "precomputed_turn_then_forward",
        }:
            if training_preflight:
                raise ValueError(
                    "Training cannot require ONNX export with "
                    f"contact_aware_sparse_root_command_mode={sparse_mode!r}: "
                    "holosoma_inference currently implements only tracking_error/default, "
                    "rolling_reference_delta, and precomputed_turn_then_forward commands. "
                    "Set --training.export-onnx False before training or implement and validate exact "
                    "deployment parity for this command mode."
                )
            raise ValueError(
                "ONNX deployment currently supports only tracking_error/default, "
                "rolling_reference_delta, and precomputed_turn_then_forward "
                "contact-aware sparse-root commands; "
                f"configured contact_aware_sparse_root_command_mode={sparse_mode!r}."
            )
        if sparse_mode == "precomputed_turn_then_forward":
            command_manager = getattr(self.env, "command_manager", None)
            motion_command = (
                command_manager.get_state("motion_command")
                if command_manager is not None
                else None
            )
            if (
                motion_command is None
                or not bool(motion_command.precomputed_turn_then_forward_enabled())
                or not bool(motion_command.motion.has_object)
                or not bool(motion_command.motion.has_precomputed_root_command)
            ):
                raise ValueError(
                    "precomputed_turn_then_forward ONNX export requires the live command manager "
                    "to expose object motion plus complete precomputed command/phase fields."
                )
            _precomputed_turn_then_forward_deployment_contract(
                zero_root_command_when_drop_active=(
                    self._actor_zero_root_command_when_drop_active()
                )
            )
        elif sparse_mode == "rolling_reference_delta":
            command_manager = getattr(self.env, "command_manager", None)
            motion_command = (
                command_manager.get_state("motion_command")
                if command_manager is not None
                else None
            )
            if (
                motion_command is None
                or not bool(motion_command.motion.has_object)
            ):
                raise ValueError(
                    "rolling_reference_delta ONNX export requires the live command "
                    "manager to expose object motion on the authenticated timeline."
                )
            lookahead, zero_yaw_threshold_deg = (
                self._actor_rolling_reference_delta_parameters()
            )
            _rolling_reference_delta_deployment_contract(
                lookahead_motion_frames=lookahead,
                zero_yaw_threshold_deg=zero_yaw_threshold_deg,
                zero_root_command_when_drop_active=(
                    self._actor_zero_root_command_when_drop_active()
                ),
            )

    @staticmethod
    def _stable_onnx_sha256(onnx_file_path: str) -> str:
        """Hash one regular, non-empty ONNX file without accepting replacement races."""

        path = Path(onnx_file_path)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"ONNX artifact is not a regular file: {path}")
            if before.st_size <= 0:
                raise ValueError(f"ONNX artifact is empty: {path}")
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 4 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)

        identity = lambda value: (  # noqa: E731 - compact immutable stat tuple
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_size),
            int(value.st_mtime_ns),
        )
        if identity(before) != identity(after):
            raise RuntimeError(f"ONNX artifact changed while it was being hashed: {path}")
        path_after = path.stat(follow_symlinks=False)
        if identity(after) != identity(path_after) or not stat.S_ISREG(path_after.st_mode):
            raise RuntimeError(f"ONNX artifact path was replaced while it was being hashed: {path}")
        return digest.hexdigest()

    @staticmethod
    def _remove_failed_onnx_artifact(onnx_file_path: str) -> None:
        path = Path(onnx_file_path)
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            logger.exception("Failed to remove incomplete ONNX artifact {}.", path)

    def _export_onnx_checkpoint(
        self,
        onnx_file_path: str,
        *,
        iteration: int | None = None,
        required: bool = False,
        upload: bool = True,
    ) -> str | None:
        if not self._should_export_onnx():
            return None
        rng_state = capture_rng_checkpoint_state()
        try:
            # A stale artifact at the same path must never satisfy a later
            # failed export.  Completion is authenticated against bytes made
            # by this exact call.
            self._remove_failed_onnx_artifact(onnx_file_path)
            try:
                self.export(
                    onnx_file_path=onnx_file_path,
                    iteration=iteration,
                    upload=upload,
                )
                digest = self._stable_onnx_sha256(onnx_file_path)
                logger.info("Authenticated ONNX artifact {} sha256={}", onnx_file_path, digest)
                return digest
            except Exception:
                self._remove_failed_onnx_artifact(onnx_file_path)
                if required:
                    raise
                logger.exception(
                    "ONNX export failed for {}; continuing after saving the non-final .pt checkpoint.",
                    onnx_file_path,
                )
                return None
        finally:
            # Export/tracing and artifact upload are deployment side effects;
            # they must not perturb the stochastic stream seen by the next
            # training iteration.  Restoration failure remains fatal.
            restore_rng_checkpoint_state(rng_state, path="pre_onnx_export_rng_state")

    def _export_final_onnx_with_distributed_outcome(
        self,
        onnx_file_path: str,
        *,
        iteration: int | None = None,
        upload: bool = True,
    ) -> str | None:
        """Export final ONNX on rank zero and publish one all-rank verdict."""

        protocol_rng_state = capture_rng_checkpoint_state()
        try:
            local_error: Exception | None = None
            local_digest: str | None = None
            if self.is_main_process:
                try:
                    local_digest = self._export_onnx_checkpoint(
                        onnx_file_path,
                        iteration=iteration,
                        required=True,
                        upload=upload,
                    )
                except Exception as exc:
                    local_error = exc

            if not self.is_multi_gpu or not torch.distributed.is_initialized():
                if local_error is not None:
                    raise local_error
                return local_digest

            group = self._setup_gloo_barrier_group()
            if group is None:
                raise RuntimeError("Distributed final ONNX outcome requires an all-rank Gloo group.")
            world_size = torch.distributed.get_world_size(group=group)
            rank = torch.distributed.get_rank(group=group)
            local_result = {
                "rank": rank,
                "error": (
                    None
                    if local_error is None
                    else f"{type(local_error).__name__}: {local_error}"
                ),
                "sha256": local_digest,
            }
            gathered: list[dict[str, Any] | None] = [None] * world_size
            torch.distributed.all_gather_object(gathered, local_result, group=group)

            envelope_problems: list[str] = []
            failures: list[str] = []
            for expected_rank, result in enumerate(gathered):
                if not isinstance(result, dict) or result.get("rank") != expected_rank:
                    envelope_problems.append(f"slot={expected_rank}: malformed result={result!r}")
                    continue
                if result.get("error") is not None:
                    failures.append(f"rank={expected_rank}: {result['error']}")
                if expected_rank != 0 and result.get("sha256") is not None:
                    envelope_problems.append(
                        f"rank={expected_rank}: non-main rank reported an ONNX digest"
                    )
            if envelope_problems:
                raise RuntimeError(
                    "Final ONNX export returned malformed all-rank envelopes: "
                    + "; ".join(envelope_problems)
                )
            if failures:
                synchronized_error = RuntimeError(
                    "Final ONNX export failed before run completion:\n  - "
                    + "\n  - ".join(failures)
                )
                if local_error is not None:
                    raise synchronized_error from local_error
                raise synchronized_error
            rank_zero_result = gathered[0]
            assert isinstance(rank_zero_result, dict)
            digest = rank_zero_result.get("sha256")
            if digest is not None and (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise RuntimeError(f"Final ONNX export returned an invalid SHA-256 digest: {digest!r}")
            return digest
        finally:
            restore_rng_checkpoint_state(
                protocol_rng_state,
                path="pre_final_onnx_outcome_protocol_rng_state",
            )

    def _write_policy_artifact_pair_manifest(
        self,
        *,
        checkpoint_path: str,
        onnx_path: str,
        next_iteration: int,
        onnx_sha256: str,
    ) -> str:
        """Authenticate one same-iteration PT/ONNX pair before publication."""

        import onnx

        checkpoint, checkpoint_sha256 = load_verified_torch_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )
        if not isinstance(checkpoint, Mapping):
            raise TypeError("Published checkpoint payload must be a mapping.")
        completed_iteration = int(next_iteration) - 1
        if checkpoint.get("iter") != completed_iteration or checkpoint.get(
            "next_iter"
        ) != int(next_iteration):
            raise RuntimeError(
                "PT checkpoint iteration does not match the artifact-pair boundary: "
                f"iter={checkpoint.get('iter')!r}, next_iter={checkpoint.get('next_iter')!r}, "
                f"expected={completed_iteration}/{next_iteration}."
            )
        actual_onnx_sha256 = self._stable_onnx_sha256(onnx_path)
        if actual_onnx_sha256 != onnx_sha256:
            raise RuntimeError(
                "ONNX bytes changed between parity validation and pair publication."
            )

        model = onnx.load(onnx_path)
        metadata: dict[str, Any] = {}
        for item in model.metadata_props:
            if not item.key or item.key in metadata:
                raise RuntimeError(
                    "ONNX artifact-pair validation found an empty or duplicate metadata key."
                )
            metadata[item.key] = json.loads(
                item.value,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {value!r}")
                ),
            )

        def strict_json_payload(value: Any, *, label: str) -> bytes:
            try:
                return json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"PT/ONNX pair field {label!r} is not strict finite JSON."
                ) from exc

        def require_same_json_field(field_name: str) -> bytes | None:
            checkpoint_present = field_name in checkpoint
            onnx_present = field_name in metadata
            if checkpoint_present != onnx_present:
                raise RuntimeError(
                    "PT and ONNX artifact metadata presence differs for "
                    f"{field_name!r}."
                )
            if not checkpoint_present:
                return None
            checkpoint_payload = strict_json_payload(
                checkpoint[field_name],
                label=f"pt.{field_name}",
            )
            onnx_payload = strict_json_payload(
                metadata[field_name],
                label=f"onnx.{field_name}",
            )
            if checkpoint_payload != onnx_payload:
                raise RuntimeError(
                    f"PT and ONNX artifacts serialize different {field_name!r} contracts."
                )
            return onnx_payload

        bound_metadata_payloads = {
            field_name: payload
            for field_name in (
                "experiment_config",
                "wandb_run_path",
                "training_provenance",
                "source_checkpoint_sha256",
                "motion_transition_contract",
                "motion_transition_contract_sha256",
                _PRECOMPUTED_COMMAND_CONTRACT_KEY,
                _PRECOMPUTED_COMMAND_CONTRACT_SHA256_KEY,
                _ROLLING_REFERENCE_DELTA_CONTRACT_KEY,
                _ROLLING_REFERENCE_DELTA_CONTRACT_SHA256_KEY,
            )
            if (payload := require_same_json_field(field_name)) is not None
        }
        if metadata.get("iteration") != completed_iteration:
            raise RuntimeError(
                "ONNX completed iteration does not match its PT checkpoint: "
                f"onnx={metadata.get('iteration')!r}, pt={completed_iteration}."
            )
        validation = metadata.get("onnx_validation_contract")
        if (
            not isinstance(validation, Mapping)
            or validation.get("version") != 1
            or validation.get("checker") != "onnx.checker.check_model"
            or validation.get("runtime") != "onnxruntime_cpu"
            or validation.get("pytorch_vs_ort") is not True
            or validation.get("completed_iteration") != completed_iteration
            or validation.get("actor_graph_semantics")
            != "raw_actor_observation_plus_authenticated_external_observation_adapter"
            or validation.get("precomputed_command_contract_sha256")
            != metadata.get(_PRECOMPUTED_COMMAND_CONTRACT_SHA256_KEY)
            or validation.get("rolling_reference_delta_contract_sha256")
            != metadata.get(_ROLLING_REFERENCE_DELTA_CONTRACT_SHA256_KEY)
        ):
            raise RuntimeError(
                "ONNX artifact is missing its same-iteration checker/ORT/parity contract."
            )
        for field_name in ("rtol", "atol", "max_abs_error", "max_rel_error"):
            field_value = validation.get(field_name)
            if (
                isinstance(field_value, bool)
                or not isinstance(field_value, numbers.Real)
                or not math.isfinite(float(field_value))
                or float(field_value) < 0.0
            ):
                raise RuntimeError(
                    "ONNX parity metadata contains an invalid non-negative finite field: "
                    f"{field_name}={field_value!r}."
                )
        probe_rows = validation.get("probe_rows")
        if (
            isinstance(probe_rows, bool)
            or not isinstance(probe_rows, numbers.Integral)
            or int(probe_rows) <= 0
        ):
            raise RuntimeError(
                f"ONNX parity metadata has invalid probe_rows={probe_rows!r}."
            )
        pt_command_digest = checkpoint.get(
            _PRECOMPUTED_COMMAND_CONTRACT_SHA256_KEY
        )
        onnx_command_digest = metadata.get(
            _PRECOMPUTED_COMMAND_CONTRACT_SHA256_KEY
        )
        if pt_command_digest != onnx_command_digest:
            raise RuntimeError(
                "PT and ONNX artifacts bind different precomputed command adapters."
            )
        if self._actor_sparse_root_command_mode() == "precomputed_turn_then_forward":
            _, expected_command_digest = (
                _precomputed_turn_then_forward_deployment_contract(
                    zero_root_command_when_drop_active=(
                        self._actor_zero_root_command_when_drop_active()
                    )
                )
            )
            if pt_command_digest != expected_command_digest:
                raise RuntimeError(
                    "PT/ONNX pair is missing the implemented precomputed command adapter digest."
                )
        pt_rolling_command_digest = checkpoint.get(
            _ROLLING_REFERENCE_DELTA_CONTRACT_SHA256_KEY
        )
        onnx_rolling_command_digest = metadata.get(
            _ROLLING_REFERENCE_DELTA_CONTRACT_SHA256_KEY
        )
        if pt_rolling_command_digest != onnx_rolling_command_digest:
            raise RuntimeError(
                "PT and ONNX artifacts bind different rolling reference-delta adapters."
            )
        if self._actor_sparse_root_command_mode() == "rolling_reference_delta":
            lookahead, zero_yaw_threshold_deg = (
                self._actor_rolling_reference_delta_parameters()
            )
            _, expected_rolling_command_digest = (
                _rolling_reference_delta_deployment_contract(
                    lookahead_motion_frames=lookahead,
                    zero_yaw_threshold_deg=zero_yaw_threshold_deg,
                    zero_root_command_when_drop_active=(
                        self._actor_zero_root_command_when_drop_active()
                    ),
                )
            )
            if pt_rolling_command_digest != expected_rolling_command_digest:
                raise RuntimeError(
                    "PT/ONNX pair is missing the implemented rolling reference-delta "
                    "command adapter digest."
                )

        for contract_key, digest_key in (
            (
                "motion_transition_contract",
                "motion_transition_contract_sha256",
            ),
            (
                _PRECOMPUTED_COMMAND_CONTRACT_KEY,
                _PRECOMPUTED_COMMAND_CONTRACT_SHA256_KEY,
            ),
            (
                _ROLLING_REFERENCE_DELTA_CONTRACT_KEY,
                _ROLLING_REFERENCE_DELTA_CONTRACT_SHA256_KEY,
            ),
            (
                "perception_observation_contract",
                "perception_observation_contract_sha256",
            ),
        ):
            contract_present = contract_key in metadata
            digest_present = digest_key in metadata
            if contract_present != digest_present:
                raise RuntimeError(
                    f"ONNX pair metadata must provide both {contract_key!r} and {digest_key!r}."
                )
            if contract_present:
                computed_digest = hashlib.sha256(
                    strict_json_payload(metadata[contract_key], label=contract_key)
                ).hexdigest()
                if metadata[digest_key] != computed_digest:
                    raise RuntimeError(
                        f"ONNX pair metadata digest mismatch for {contract_key!r}."
                    )

        checkpoint_geometry_support = checkpoint.get(
            _ACTOR_PERCEPTION_GEOMETRY_SUPPORT_KEY
        )
        perception_contract = metadata.get("perception_observation_contract")
        onnx_geometry_support = (
            perception_contract.get("training_geometry_support")
            if isinstance(perception_contract, Mapping)
            else None
        )
        if (checkpoint_geometry_support is None) != (onnx_geometry_support is None):
            raise RuntimeError(
                "PT and ONNX artifacts disagree on actor perception geometry support."
            )
        perception_geometry_payload = None
        if checkpoint_geometry_support is not None:
            checkpoint_geometry_payload = strict_json_payload(
                checkpoint_geometry_support,
                label=f"pt.{_ACTOR_PERCEPTION_GEOMETRY_SUPPORT_KEY}",
            )
            perception_geometry_payload = strict_json_payload(
                onnx_geometry_support,
                label="onnx.perception_observation_contract.training_geometry_support",
            )
            if checkpoint_geometry_payload != perception_geometry_payload:
                raise RuntimeError(
                    "PT actor perception geometry support differs from the ONNX observation contract."
                )

        provenance_payload = bound_metadata_payloads.get("training_provenance")
        transition_digest = metadata.get("motion_transition_contract_sha256")
        perception_digest = metadata.get("perception_observation_contract_sha256")
        bound_metadata_payload = strict_json_payload(
            {
                field_name: json.loads(payload)
                for field_name, payload in sorted(bound_metadata_payloads.items())
            },
            label="bound_metadata",
        )

        experiment_payload = bound_metadata_payloads["experiment_config"]
        checkpoint_stat = Path(checkpoint_path).stat()
        onnx_stat = Path(onnx_path).stat()
        manifest = {
            "version": 1,
            "semantics": "atomic_same_iteration_pt_onnx_policy_pair",
            "completed_iteration": completed_iteration,
            "next_iteration": int(next_iteration),
            "experiment_config_sha256": hashlib.sha256(
                experiment_payload
            ).hexdigest(),
            "bound_metadata_sha256": hashlib.sha256(
                bound_metadata_payload
            ).hexdigest(),
            "training_provenance_sha256": (
                None
                if provenance_payload is None
                else hashlib.sha256(provenance_payload).hexdigest()
            ),
            "motion_transition_contract_sha256": transition_digest,
            "perception_observation_contract_sha256": perception_digest,
            "perception_training_geometry_support_sha256": (
                None
                if perception_geometry_payload is None
                else hashlib.sha256(perception_geometry_payload).hexdigest()
            ),
            "precomputed_command_contract_sha256": pt_command_digest,
            "rolling_reference_delta_contract_sha256": (
                pt_rolling_command_digest
            ),
            "pt": {
                "name": Path(checkpoint_path).name,
                "sha256": checkpoint_sha256,
                "size_bytes": int(checkpoint_stat.st_size),
            },
            "onnx": {
                "name": Path(onnx_path).name,
                "sha256": onnx_sha256,
                "size_bytes": int(onnx_stat.st_size),
                "checker": validation["checker"],
                "runtime": validation["runtime"],
                "pytorch_vs_ort": True,
                "max_abs_error": float(validation["max_abs_error"]),
                "max_rel_error": float(validation["max_rel_error"]),
            },
        }
        manifest_path = str(Path(checkpoint_path).with_suffix(".pair.json"))
        parent = Path(manifest_path).parent
        payload = json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        fd, temporary_path_raw = tempfile.mkstemp(
            dir=parent,
            prefix=f".{Path(manifest_path).name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_path_raw)
        try:
            with os.fdopen(fd, "wb", closefd=True) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, manifest_path)
            directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary_path.unlink(missing_ok=True)
        return manifest_path

    def _save_policy_artifact_pair_with_distributed_outcome(
        self,
        *,
        checkpoint_path: str,
        onnx_path: str,
        next_iteration: int,
    ) -> str:
        """Generate, validate, and publish one required PT+ONNX boundary."""

        protocol_rng_state = capture_rng_checkpoint_state()
        onnx_ready = False
        try:
            self._save_checkpoint_with_distributed_outcome(
                checkpoint_path,
                next_iteration=next_iteration,
                upload=False,
            )
            onnx_sha256 = self._export_final_onnx_with_distributed_outcome(
                onnx_path,
                iteration=int(next_iteration) - 1,
                upload=False,
            )
            if not isinstance(onnx_sha256, str) or len(onnx_sha256) != 64:
                raise RuntimeError(
                    "Required ONNX pair export did not return an authenticated SHA-256."
                )
            onnx_ready = True

            local_error: Exception | None = None
            if self.is_main_process:
                try:
                    manifest_path = self._write_policy_artifact_pair_manifest(
                        checkpoint_path=checkpoint_path,
                        onnx_path=onnx_path,
                        next_iteration=next_iteration,
                        onnx_sha256=onnx_sha256,
                    )
                    # Publish the resumable PT last.  A partial W&B failure can
                    # therefore leave deployment/manifest diagnostics, but it
                    # can never advertise a resumable PT without its ONNX.
                    self.logging_helper.save_to_wandb(onnx_path)
                    self.logging_helper.save_to_wandb(manifest_path)
                    self.logging_helper.save_to_wandb(checkpoint_path)
                    logger.info(
                        "Published required PT+ONNX pair at next_iteration={} pt={} onnx={}.",
                        next_iteration,
                        Path(checkpoint_path).name,
                        Path(onnx_path).name,
                    )
                except Exception as exc:
                    local_error = exc
            self._synchronize_training_phase_error(
                local_error,
                operation=(
                    f"same-iteration PT+ONNX pair publication at {next_iteration}"
                ),
            )
            return onnx_sha256
        except Exception:
            if self.is_main_process and not onnx_ready:
                Path(checkpoint_path).unlink(missing_ok=True)
                Path(onnx_path).unlink(missing_ok=True)
                Path(checkpoint_path).with_suffix(".pair.json").unlink(
                    missing_ok=True
                )
            raise
        finally:
            restore_rng_checkpoint_state(
                protocol_rng_state,
                path="pre_policy_artifact_pair_protocol_rng_state",
            )

    def _post_epoch_logging_preserving_rng(
        self,
        it: int,
        loss_dict: dict[str, float],
        *,
        fixed_bc_eval_metrics: dict[str, float] | None = None,
    ) -> None:
        """Emit rank-zero observational logs with an all-rank outcome verdict."""

        local_error: Exception | None = None
        if self.is_main_process:
            logging_rng_state = None
            try:
                logging_rng_state = capture_rng_checkpoint_state()
                self._post_epoch_logging(
                    it,
                    loss_dict,
                    fixed_bc_eval_metrics=fixed_bc_eval_metrics,
                )
            except Exception as exc:
                local_error = exc
            finally:
                if logging_rng_state is not None:
                    try:
                        # TensorBoard/W&B/rich output is observational.  Keep
                        # library-internal global RNG use out of training.
                        restore_rng_checkpoint_state(
                            logging_rng_state,
                            path="pre_epoch_logging_rng_state",
                        )
                    except Exception as exc:
                        if local_error is None:
                            local_error = exc
                        else:
                            local_error = RuntimeError(
                                "Epoch logging and its RNG-state restoration both failed: "
                                f"logging={type(local_error).__name__}: {local_error}; "
                                f"restore={type(exc).__name__}: {exc}"
                            )
        self._synchronize_training_phase_error(
            local_error,
            operation="rank-zero epoch logging",
        )

    def _select_teacher_actions(
        self,
        teacher_obs_raw: torch.Tensor,
        obs_dict: dict[str, torch.Tensor],
        *,
        stochastic: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        use_stochastic_actions = self.teacher_use_stochastic_actions if stochastic is None else stochastic

        def teacher_act(teacher_actor: nn.Module, policy_state: dict[str, torch.Tensor]) -> torch.Tensor:
            if use_stochastic_actions:
                return teacher_actor.act(policy_state)
            return teacher_actor.act_inference(policy_state)

        if self.use_multi_teacher:
            if self.multi_teacher_select_obs_var not in obs_dict:
                raise ValueError(
                    f"Multi-teacher enabled but observation '{self.multi_teacher_select_obs_var}' not found."
                )
            raw_teacher_indices = obs_dict[self.multi_teacher_select_obs_var].view(-1)
            if raw_teacher_indices.numel() != teacher_obs_raw.shape[0]:
                raise ValueError(
                    "Multi-teacher selector must contain exactly one value per environment: "
                    f"got {raw_teacher_indices.numel()} for batch {teacher_obs_raw.shape[0]}."
                )
            if not torch.isfinite(raw_teacher_indices).all():
                raise ValueError("Multi-teacher selector contains NaN or infinity.")
            if raw_teacher_indices.is_complex():
                raise ValueError("Multi-teacher selector values must be real integers.")
            rounded_indices = (
                raw_teacher_indices
                if raw_teacher_indices.dtype == torch.bool
                else raw_teacher_indices.round()
            )
            if not torch.equal(raw_teacher_indices, rounded_indices):
                invalid = raw_teacher_indices[raw_teacher_indices != rounded_indices][:8].tolist()
                raise ValueError(f"Multi-teacher selector values must be integers; invalid values={invalid}.")
            teacher_indices = rounded_indices.to(dtype=torch.long)
            num_teachers = len(self.teacher_actors)
            if num_teachers == 0:
                raise ValueError("Multi-teacher mode has no loaded teacher actors.")
            invalid_range = (teacher_indices < 0) | (teacher_indices >= num_teachers)
            if invalid_range.any():
                invalid = teacher_indices[invalid_range][:8].tolist()
                raise ValueError(
                    f"Multi-teacher selector values must be in [0, {num_teachers - 1}]; invalid values={invalid}."
                )
            teacher_actions = torch.zeros((teacher_obs_raw.shape[0], self.num_act), device=teacher_obs_raw.device)
            for idx, (teacher_actor, normalizers) in enumerate(
                zip(self.teacher_actors, self.teacher_actor_obs_normalizers_list)
            ):
                mask = teacher_indices == idx
                if not mask.any():
                    continue
                teacher_obs = self._normalize_teacher_actor_obs(teacher_obs_raw[mask], normalizers=normalizers)
                teacher_policy_state = {"actor_obs": teacher_obs}
                teacher_perception_key = str(getattr(teacher_actor, "perception_input_name", "") or "")
                if teacher_perception_key:
                    if teacher_perception_key not in obs_dict:
                        raise ValueError(
                            f"Teacher perception obs '{teacher_perception_key}' not found in current observation dict."
                        )
                    teacher_policy_state[teacher_perception_key] = obs_dict[teacher_perception_key][mask]
                teacher_actions[mask] = teacher_act(teacher_actor, teacher_policy_state)
            return teacher_actions, teacher_indices

        assert self.teacher_actor is not None, "Teacher actor is not initialized."
        teacher_obs = self._normalize_teacher_actor_obs(teacher_obs_raw)
        teacher_policy_state = {"actor_obs": teacher_obs}
        teacher_perception_key = str(getattr(self.teacher_actor, "perception_input_name", "") or "")
        if teacher_perception_key:
            if teacher_perception_key not in obs_dict:
                raise ValueError(
                    f"Teacher perception obs '{teacher_perception_key}' not found in current observation dict."
                )
            teacher_policy_state[teacher_perception_key] = obs_dict[teacher_perception_key]
        teacher_actions = teacher_act(self.teacher_actor, teacher_policy_state)
        return teacher_actions, None

    def _try_select_teacher_actions_for_rollout(
        self,
        teacher_obs_raw: torch.Tensor,
        obs_dict: dict[str, torch.Tensor],
        *,
        stochastic: bool | None = None,
        preserve_rng: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, Exception | None]:
        """Capture a rank-local teacher failure for the synchronized pre-env verdict.

        Teacher selection and forward passes contain no distributed
        collectives.  A failed rank can therefore defer its exception to the
        per-step all-rank boundary while healthy ranks finish the same local
        computation, avoiding a peer blocking in that boundary.
        """

        rng_state = None
        actions = None
        indices = None
        local_error: Exception | None = None
        try:
            if preserve_rng:
                rng_state = capture_rng_checkpoint_state()
            actions, indices = self._select_teacher_actions(
                teacher_obs_raw,
                obs_dict,
                stochastic=stochastic,
            )
        except Exception as exc:
            local_error = exc
        if rng_state is not None:
            try:
                restore_rng_checkpoint_state(
                    rng_state,
                    path="pre_observational_teacher_label_rng_state",
                )
            except Exception as exc:
                if local_error is None:
                    local_error = exc
                else:
                    local_error = RuntimeError(
                        "Teacher label selection and observational RNG restoration both failed: "
                        f"selection={type(local_error).__name__}: {local_error}; "
                        f"restore={type(exc).__name__}: {exc}"
                    )
        if local_error is not None:
            return None, None, local_error
        return actions, indices, None

    def _build_teacher_obs_raw(
        self,
        obs_dict: dict[str, torch.Tensor],
        actor_obs_raw: torch.Tensor,
    ) -> torch.Tensor:
        if self.teacher_obs_keys == self.actor_obs_keys:
            return actor_obs_raw
        return torch.cat([obs_dict[key] for key in self.teacher_obs_keys], dim=1)

    def _compute_ppo_dagger_coeff_for_epoch(self, current_epoch: int) -> float:
        if not self.use_ppo_dagger_schedule:
            return 1.0

        if current_epoch < self.ppo_start_epoch:
            return 0.0
        if current_epoch >= self.dagger_end_epoch:
            return float(self.ppo_target_coeff)

        total_epochs = max(1, self.dagger_end_epoch - self.ppo_start_epoch)
        ppo_epochs = max(0, current_epoch - self.ppo_start_epoch)
        coeff_span = self.ppo_target_coeff - self.ppo_start_coeff
        if self.ppo_schedule_step_epochs > 0:
            step_epochs = max(1, self.ppo_schedule_step_epochs)
            total_steps = max(1, (total_epochs + step_epochs - 1) // step_epochs)
            completed_steps = max(0, ppo_epochs // step_epochs)
            progress = min(float(completed_steps) / float(total_steps), 1.0)
            return self.ppo_start_coeff + progress * coeff_span

        progress = min(float(ppo_epochs) / float(total_epochs), 1.0)
        return self.ppo_start_coeff + progress * coeff_span

    def _validate_teacher_rollout_action_config(self) -> None:
        """Reject rollout actions whose stored PPO likelihood is not valid.

        Rollout storage records the student action and its likelihood.  If the
        environment is stepped with a teacher action, those fields are only
        semantically valid while the actor objective is pure behavior cloning.
        """
        configured_static_mix_ratio = float(
            getattr(
                self,
                "_configured_teacher_action_mix_ratio",
                self.teacher_action_mix_ratio,
            )
        )
        scheduled_mix_values = (
            self.teacher_action_mix_ratio_start,
            self.teacher_action_mix_ratio_end,
        )
        scheduled_mix_is_positive = any(
            value is not None and value > 0.0 for value in scheduled_mix_values
        )
        if getattr(self, "use_teacher_action_mix_schedule", False) and configured_static_mix_ratio > 0.0:
            raise ValueError(
                "distill.teacher_action_mix_ratio is mutually exclusive with the scheduled "
                "teacher_action_mix_ratio_start/end fields; the static value would be silently ignored."
            )
        if self.take_teacher_actions and (
            configured_static_mix_ratio > 0.0 or scheduled_mix_is_positive
        ):
            raise ValueError(
                "distill.take_teacher_actions is mutually exclusive with a non-zero teacher-action mix ratio; "
                "otherwise the mix branch silently overrides the all-teacher rollout request."
            )
        teacher_controls_rollout = self.take_teacher_actions or self.teacher_action_mix_ratio > 0.0 or any(
            value is not None and value > 0.0 for value in scheduled_mix_values
        )
        if self.distill_mode != "dagger":
            ignored_options = []
            if teacher_controls_rollout:
                ignored_options.append(
                    "take_teacher_actions/teacher_action_mix_ratio"
                )
            elif getattr(self, "use_teacher_action_mix_schedule", False):
                # A zero-to-zero schedule still declares DAgger-only control
                # metadata and end-iteration state that legacy MSE never uses.
                ignored_options.append("teacher_action_mix_ratio_start/end schedule")
            if getattr(self, "teacher_use_stochastic_actions", False):
                ignored_options.append("teacher_use_stochastic_actions")
            if getattr(self, "switch_to_rl_after", -1) > 0:
                ignored_options.append("switch_to_rl_after")
            if getattr(self, "use_ppo_dagger_schedule", False):
                ignored_options.append("ppo_start_epoch/dagger_end_epoch schedule")
            if getattr(self, "use_multi_teacher", False):
                ignored_options.append("use_multi_teacher")
            if getattr(self, "dagger_match_std", False):
                ignored_options.append("dagger_match_std")
            if getattr(self, "dagger_ignore_episode_initial_steps", 0) > 0:
                ignored_options.append("dagger_ignore_episode_initial_steps")
            if getattr(self, "dagger_loss_coef", 10.0) != 10.0:
                ignored_options.append("dagger_loss_coef")
            if getattr(self, "ppo_start_coeff", 0.0) != 0.0:
                ignored_options.append("ppo_start_coeff")
            if getattr(self, "ppo_target_coeff", 0.9) != 0.9:
                ignored_options.append("ppo_target_coeff")
            if getattr(self, "ppo_start_noise_std_until_coeff", 0.1) != 0.1:
                ignored_options.append("ppo_start_noise_std_until_coeff")
            if not getattr(self, "dagger_ignore_zero_teacher_actions", True):
                ignored_options.append("dagger_ignore_zero_teacher_actions")
            if getattr(self, "multi_teacher_select_obs_var", "teacher_checkpoint_index") != "teacher_checkpoint_index":
                ignored_options.append("multi_teacher_select_obs_var")
            if ignored_options:
                raise ValueError(
                    "Legacy distill.mode='mse' does not implement these DAgger rollout options: "
                    + ", ".join(ignored_options)
                    + ". Use distill.mode='dagger' or disable them."
                )
            return

        if not self.use_ppo_dagger_schedule:
            ignored_schedule_options = []
            if getattr(self, "ppo_start_coeff", 0.0) != 0.0:
                ignored_schedule_options.append("ppo_start_coeff")
            if getattr(self, "ppo_target_coeff", 0.9) != 0.9:
                ignored_schedule_options.append("ppo_target_coeff")
            if getattr(self, "dagger_loss_coef", 10.0) != 10.0:
                ignored_schedule_options.append("dagger_loss_coef")
            if getattr(self, "ppo_start_noise_std_until_coeff", 0.1) != 0.1:
                ignored_schedule_options.append("ppo_start_noise_std_until_coeff")
            if ignored_schedule_options:
                raise ValueError(
                    "Unscheduled DAgger does not consume these PPO/DAgger schedule options: "
                    + ", ".join(ignored_schedule_options)
                    + ". Enable a ppo_start_epoch/dagger_end_epoch schedule or restore their defaults."
                )
        elif (
            getattr(self, "ppo_start_noise_std", None) is None
            and getattr(self, "ppo_start_noise_std_until_coeff", 0.1) != 0.1
        ):
            raise ValueError(
                "distill.ppo_start_noise_std_until_coeff is only consumed when "
                "distill.ppo_start_noise_std is configured."
            )

        if (
            not getattr(self, "use_multi_teacher", False)
            and getattr(
                self,
                "multi_teacher_select_obs_var",
                "teacher_checkpoint_index",
            )
            != "teacher_checkpoint_index"
        ):
            raise ValueError(
                "distill.multi_teacher_select_obs_var is only consumed when use_multi_teacher=True."
            )

        if not teacher_controls_rollout:
            return

        if self.use_ppo_dagger_schedule:
            ppo_can_contribute = self._ppo_can_contribute()
        else:
            # Non-scheduled DAgger uses (1 - bc_loss_coef) * PPO and a
            # switch_to_rl_after transition eventually enables full PPO.
            ppo_can_contribute = self._configured_bc_loss_coef < 1.0 or self.switch_to_rl_after > 0

        if ppo_can_contribute:
            raise ValueError(
                "Teacher rollout actions (take_teacher_actions/teacher_action_mix_ratio) cannot be combined "
                "with a non-zero PPO contribution: rollout storage contains student actions and student "
                "log-probabilities. Use pure DAgger with bc_loss_coef=1 and no PPO/switch schedule, or set "
                "teacher rollout action mixing to zero."
            )

    def _teacher_controls_current_rollout(self) -> bool:
        return bool(
            self.take_teacher_actions
            or float(getattr(self, "teacher_action_mix_ratio", 0.0)) > 0.0
        )

    def _effective_dagger_loss_weight(self) -> float:
        """Return the float32 BC multiplier used by the active actor graph."""

        if (
            getattr(self, "distill_mode", "mse") != "dagger"
            or not getattr(self, "dagger_enabled", False)
        ):
            return 0.0
        if getattr(self, "use_ppo_dagger_schedule", False):
            # Form the complementary BC weight from the canonical Python
            # schedule value before materializing that independent loss
            # weight in float32.  Rounding lambda to float32 first would erase
            # a representable positive BC term near lambda=1.
            lambda_ppo = float(getattr(self, "ppo_coeff", 1.0))
            weight = float(getattr(self, "dagger_loss_coef", 0.0)) * (1.0 - lambda_ppo)
        else:
            weight = max(0.0, float(getattr(self, "bc_loss_coef", 0.0)))
        return self._operational_float32_loss_weight(weight)

    def _teacher_observations_required_for_current_rollout(self) -> bool:
        if self._legacy_distillation_enabled():
            return True
        if not getattr(self, "dagger_enabled", False):
            return False
        return bool(
            self._effective_dagger_loss_weight() > 0.0
            or self._teacher_controls_current_rollout()
            or self._fixed_bc_eval_capture_pending()
        )

    def _pure_ppo_fixed_capture_error(self) -> RuntimeError | None:
        """Reject diagnostic capture that would perturb a pure-PPO stream."""

        if (
            getattr(self, "dagger_enabled", False)
            and self._effective_dagger_loss_weight() <= 0.0
            and not self._teacher_controls_current_rollout()
            and self._fixed_bc_eval_capture_pending()
        ):
            return RuntimeError(
                "The fixed-BC evaluation stratum is incomplete after the DAgger loss reached zero. "
                "Computing teacher-only observation groups now could consume RNG or mutate history and "
                "would make the nominal pure-PPO stream differ from pure PPO. Increase the capture "
                "budget availability during the BC phase or resume from a checkpoint with a complete "
                "fixed_bc_eval_by_rank dataset."
            )
        return None

    def _adjust_ppo_dagger_coeff(self, current_epoch: int) -> None:
        """PPO/DAgger curriculum mixing schedule.

        - epoch < ppo_start_epoch: ppo_coeff = 0.0
        - epoch >= dagger_end_epoch: ppo_coeff = ppo_target_coeff
        - otherwise: linear ramp from ppo_start_coeff to ppo_target_coeff, or
          staircase updates when ``ppo_schedule_step_epochs > 0``
        """
        self.ppo_coeff = self._compute_ppo_dagger_coeff_for_epoch(current_epoch)

    def _refresh_distillation_iteration_state(self, current_iteration: int) -> None:
        """Derive all iteration-dependent distillation state before rollout."""
        if not getattr(self, "dagger_enabled", False):
            return
        if self.use_ppo_dagger_schedule:
            self._adjust_ppo_dagger_coeff(current_iteration)
            return
        if self.switch_to_rl_after > 0:
            self.bc_loss_coef = (
                0.0 if current_iteration >= self.switch_to_rl_after else self._configured_bc_loss_coef
            )

    def _future_dagger_bc_positive_interval(
        self,
        start_iteration: int,
        end_iteration: int,
    ) -> tuple[int, int] | None:
        """Return the exact half-open future interval with an operational BC weight.

        Every supported DAgger curriculum starts with BC and can turn it off at
        most once.  Expressing that monotone contract as one interval keeps
        resume/short-run mask validation independent of mutable live coefficients.
        """

        start_iteration = int(start_iteration)
        end_iteration = int(end_iteration)
        if (
            start_iteration >= end_iteration
            or getattr(self, "distill_mode", "mse") != "dagger"
            or not getattr(self, "dagger_enabled", False)
        ):
            return None

        def bc_weight(iteration: int) -> float:
            if getattr(self, "use_ppo_dagger_schedule", False):
                lambda_ppo = self._compute_ppo_dagger_coeff_for_epoch(iteration)
                return self._operational_float32_loss_weight(
                    float(getattr(self, "dagger_loss_coef", 0.0))
                    * (1.0 - lambda_ppo)
                )

            configured_weight = max(
                0.0,
                float(getattr(self, "_configured_bc_loss_coef", 0.0)),
            )
            switch_iteration = int(getattr(self, "switch_to_rl_after", -1))
            if switch_iteration > 0 and iteration >= switch_iteration:
                return 0.0
            return self._operational_float32_loss_weight(configured_weight)

        # The supported schedules are monotone from BC toward PPO.  Evaluate
        # the same floating-point expression used at rollout time, then locate
        # its first non-positive weight.  This catches an IEEE-754 rounding or
        # underflow to zero before the nominal integer schedule endpoint.
        if bc_weight(start_iteration) <= 0.0:
            return None
        if bc_weight(end_iteration - 1) > 0.0:
            return start_iteration, end_iteration

        lower = start_iteration + 1
        upper = end_iteration - 1
        while lower < upper:
            midpoint = (lower + upper) // 2
            if bc_weight(midpoint) > 0.0:
                lower = midpoint + 1
            else:
                upper = midpoint
        bc_end = lower
        return start_iteration, bc_end

    @staticmethod
    def _maximum_canonical_rollout_block_iterations(
        start_iteration: int,
        end_iteration: int,
        save_interval: int,
    ) -> int:
        """Return the longest rollout block before checkpoint reset or ``end``."""

        if start_iteration >= end_iteration:
            return 0
        start_iteration = int(start_iteration)
        end_iteration = int(end_iteration)
        save_interval = int(save_interval)
        first_checkpoint_boundary = (
            (start_iteration // save_interval) + 1
        ) * save_interval
        first_block_iterations = (
            min(end_iteration, first_checkpoint_boundary) - start_iteration
        )
        remaining_iterations = max(
            0,
            end_iteration - first_checkpoint_boundary,
        )
        return max(
            first_block_iterations,
            min(save_interval, remaining_iterations),
        )

    @staticmethod
    def _start_at_zero_probability_one_intervals(
        motion_cfg: Any,
        start_iteration: int,
        end_iteration: int,
    ) -> list[tuple[int, int]]:
        """Return intervals where WBT's float32 reset-at-zero draw is certain."""

        start_iteration = int(start_iteration)
        end_iteration = int(end_iteration)
        if start_iteration >= end_iteration:
            return []

        start_probability = float(motion_cfg.start_at_timestep_zero_prob)
        end_probability = getattr(motion_cfg, "start_at_timestep_zero_prob_end", None)
        schedule_start = getattr(motion_cfg, "start_at_timestep_zero_prob_start_iter", None)
        schedule_end = getattr(motion_cfg, "start_at_timestep_zero_prob_end_iter", None)
        schedule_enabled = not (
            end_probability is None
            or schedule_start is None
            or schedule_end is None
        )
        if schedule_enabled:
            end_probability = float(end_probability)
            schedule_start = int(schedule_start)
            schedule_end = int(schedule_end)
        else:
            end_probability = start_probability

        def runtime_probability_is_one(iteration: int) -> bool:
            probability = start_probability
            if schedule_enabled:
                if iteration < schedule_start:
                    progress = 0.0
                elif schedule_end <= schedule_start:
                    progress = 1.0
                else:
                    progress = min(
                        max(
                            float(iteration - schedule_start)
                            / float(schedule_end - schedule_start),
                            0.0,
                        ),
                        1.0,
                    )
                probability = start_probability + (
                    float(end_probability) - start_probability
                ) * progress
            probability = max(0.0, min(1.0, float(probability)))
            # WBT materializes the Bernoulli probabilities with
            # ``torch.float32`` before comparing them to ``torch.rand``.  Use
            # the same conversion: a valid Python float below one can round to
            # an operationally certain reset probability.
            return float(np.float32(probability)) == 1.0

        increasing = float(end_probability) >= start_probability
        first_is_one = runtime_probability_is_one(start_iteration)
        last_is_one = runtime_probability_is_one(end_iteration - 1)
        if increasing:
            if first_is_one:
                return [(start_iteration, end_iteration)]
            if not last_is_one:
                return []
            lower = start_iteration + 1
            upper = end_iteration - 1
            while lower < upper:
                midpoint = (lower + upper) // 2
                if runtime_probability_is_one(midpoint):
                    upper = midpoint
                else:
                    lower = midpoint + 1
            return [(lower, end_iteration)]

        if not first_is_one:
            return []
        if last_is_one:
            return [(start_iteration, end_iteration)]
        lower = start_iteration + 1
        upper = end_iteration - 1
        while lower < upper:
            midpoint = (lower + upper) // 2
            if runtime_probability_is_one(midpoint):
                lower = midpoint + 1
            else:
                upper = midpoint
        return [(start_iteration, lower)]

    def _validate_future_dagger_bc_mask_signal(
        self,
        *,
        start_iteration: int,
        end_iteration: int,
    ) -> None:
        """Validate global label support and deterministic poisoned-reset branches."""

        bc_interval = self._future_dagger_bc_positive_interval(
            start_iteration,
            end_iteration,
        )
        if bc_interval is None:
            return

        ignore_steps = int(getattr(self, "dagger_ignore_episode_initial_steps", 0))
        env = getattr(self, "env", None)
        command_manager = getattr(env, "command_manager", None)
        motion_command = (
            command_manager.get_state("motion_command")
            if command_manager is not None
            else None
        )
        runtime_prepend_enabled = bool(
            getattr(motion_command, "_runtime_default_pose_prepend_enabled", False)
        )
        if ignore_steps <= 0 and not runtime_prepend_enabled:
            return

        episode_horizon = self._strict_positive_integer_equivalent(
            "env.max_episode_length",
            getattr(env, "max_episode_length", None),
        )
        save_interval = self._strict_positive_ppo_int(
            "save_interval",
            getattr(self.config, "save_interval", None),
        )
        rollout_steps = self._strict_positive_ppo_int(
            "num_steps_per_env",
            getattr(self.config, "num_steps_per_env", None),
        )
        max_block_iterations = self._maximum_canonical_rollout_block_iterations(
            bc_interval[0],
            bc_interval[1],
            save_interval,
        )
        max_block_rollout_transitions = max_block_iterations * rollout_steps
        max_episode_age_capacity = self._canonical_rollout_episode_age_capacity(
            episode_horizon,
            max_block_rollout_transitions,
            init_at_random_ep_len=bool(
                getattr(self.config, "init_at_random_ep_len", False)
            ),
        )
        if ignore_steps > 0:
            if ignore_steps >= max_episode_age_capacity:
                raise ValueError(
                    "The remaining DAgger BC interval cannot produce any valid sample after canonical "
                    "resets: "
                    f"iterations=[{bc_interval[0]}, {bc_interval[1]}), "
                    f"max_block_iterations={max_block_iterations}, "
                    f"num_steps_per_env={rollout_steps}, episode_horizon={episode_horizon}, "
                    f"dagger_ignore_episode_initial_steps={ignore_steps}. The largest possible episode "
                    f"age capacity is {max_episode_age_capacity} after accounting for reset_all()'s "
                    "dummy transition, so every BC mask in this run would be false."
                )

        if not runtime_prepend_enabled:
            return
        prepend_steps = self._strict_positive_ppo_int(
            "motion_command._runtime_default_pose_prepend_steps",
            getattr(motion_command, "_runtime_default_pose_prepend_steps", None),
        )
        motion_cfg = getattr(motion_command, "motion_cfg", None)
        if motion_cfg is None:
            raise ValueError(
                "Runtime default-pose prepend is enabled but motion_command.motion_cfg is unavailable."
            )
        probability_one_intervals = self._start_at_zero_probability_one_intervals(
            motion_cfg,
            bc_interval[0],
            bc_interval[1],
        )
        if not probability_one_intervals:
            return
        first_conflict_iteration = probability_one_intervals[0][0]
        first_valid_episode_age = max(ignore_steps, prepend_steps - 1)
        if first_valid_episode_age < max_episode_age_capacity:
            return
        raise ValueError(
            "Runtime default-pose prepend would permanently empty DAgger's BC mask for resets at "
            f"iteration {first_conflict_iteration}: prepend_steps={prepend_steps}, "
            f"dagger_ignore_episode_initial_steps={ignore_steps}, episode_horizon={episode_horizon}, "
            f"largest canonical-block episode-age capacity={max_episode_age_capacity}, and first valid "
            f"episode age={first_valid_episode_age}. The BC weight is positive and the effective "
            "float32 start_at_timestep_zero_prob is exactly 1, so no episode in even the longest "
            "available checkpoint block can finish the masked prefix. Shorten the prepend, lengthen "
            "the episode or block, end BC before this curriculum interval, or keep the effective "
            "reset-at-zero probability below 1."
        )

    def _legacy_distillation_enabled(self) -> bool:
        """Return whether the non-DAgger teacher-MSE objective is active.

        DAgger keeps ``distill_enabled`` true so rollout storage retains its
        teacher fields.  After ``switch_to_rl_after`` sets the DAgger BC
        coefficient to zero, that storage flag must not reactivate the legacy
        MSE objective and silently prevent the intended transition to pure RL.
        """
        return bool(
            getattr(self, "distill_enabled", False)
            and getattr(self, "distill_mode", "mse") != "dagger"
        )

    def _loaded_teacher_actors(self) -> list[nn.Module]:
        if getattr(self, "use_multi_teacher", False):
            return list(getattr(self, "teacher_actors", ()) or ())
        teacher_actor = getattr(self, "teacher_actor", None)
        return [] if teacher_actor is None else [teacher_actor]

    def _validate_loaded_teacher_inference_contract(self) -> None:
        """Ensure a requested deterministic label path consumes no flow noise."""

        loaded_teachers = self._loaded_teacher_actors()
        if getattr(self, "use_multi_teacher", False):
            recurrent_teacher_indices = [
                index
                for index, teacher_actor in enumerate(loaded_teachers)
                if getattr(teacher_actor, "perception_time_gru", None) is not None
            ]
            if recurrent_teacher_indices:
                raise ValueError(
                    "Multi-teacher DAgger does not support recurrent teacher policies: selector masks create "
                    "variable sub-batches, so recurrent hidden rows would no longer identify the same "
                    "environments and full-batch done resets would be invalid. Recurrent teacher indices: "
                    f"{recurrent_teacher_indices}. Use one recurrent teacher or non-recurrent multi-teachers."
                )

        deterministic_labels_required = bool(
            self.distill_mode != "dagger"
            or not self.teacher_use_stochastic_actions
        )
        if not deterministic_labels_required:
            return
        for teacher_index, teacher_actor in enumerate(loaded_teachers):
            if not bool(getattr(teacher_actor, "supports_flow_matching", False)):
                continue
            actor_impl = getattr(getattr(teacher_actor, "actor_module", None), "module", None)
            inference_noise_std = float(
                getattr(actor_impl, "inference_noise_std", 0.0) or 0.0
            )
            if inference_noise_std > 0.0:
                raise ValueError(
                    "Deterministic teacher labels require flow_inference_noise_std=0; "
                    f"teacher[{teacher_index}] has {inference_noise_std}. The flow actor's "
                    "act_inference path otherwise samples noise and advances the training RNG."
                )

    def _should_apply_ppo_start_noise_std_cap(self, current_epoch: int) -> bool:
        if (
            getattr(self, "ppo_start_noise_std", None) is None
            or not getattr(self, "use_ppo_dagger_schedule", False)
        ):
            return False
        if current_epoch < self.ppo_start_epoch:
            return False
        ppo_coeff = self._compute_ppo_dagger_coeff_for_epoch(current_epoch)
        if ppo_coeff <= self.ppo_start_noise_std_until_coeff + 1e-8:
            return True

        if self.ppo_schedule_step_epochs > 0:
            step_epochs = max(1, self.ppo_schedule_step_epochs)
            if self.ppo_start_coeff > 0.0:
                first_positive_tier_start = self.ppo_start_epoch
            else:
                first_positive_tier_start = self.ppo_start_epoch + step_epochs
            first_positive_tier_end = first_positive_tier_start + step_epochs
            return current_epoch < first_positive_tier_end

        return False

    def _apply_ppo_start_noise_std_cap(self, current_epoch: int) -> None:
        if not self._should_apply_ppo_start_noise_std_cap(current_epoch):
            return
        if not hasattr(self.actor, "std"):
            return
        assert self.ppo_start_noise_std is not None
        with torch.no_grad():
            std = self.actor.std.data
            finite_mask = torch.isfinite(std)
            lower = 1e-6
            min_noise_std = getattr(self.actor, "min_noise_std", None)
            if min_noise_std is not None:
                lower = max(lower, float(min_noise_std))
            min_mean_noise_std = getattr(self.actor, "min_mean_noise_std", None)
            configured_cap = float(self.ppo_start_noise_std)
            max_noise_std = getattr(self.actor, "max_noise_std", None)
            cap = (
                min(configured_cap, float(max_noise_std))
                if max_noise_std is not None
                else configured_cap
            )
            required_floor = max(
                lower,
                float(min_mean_noise_std)
                if min_mean_noise_std is not None
                else lower,
            )
            if cap < required_floor:
                raise ValueError(
                    "PPO-start noise cap cannot satisfy the actor's policy-noise constraints: "
                    f"effective_cap={cap}, required_floor={required_floor}."
                )
            projected_std = torch.clamp(std, min=lower, max=cap)
            if min_mean_noise_std is not None:
                target_mean = float(min_mean_noise_std)
                current_mean = projected_std.mean()
                alpha = (target_mean - current_mean) / (cap - current_mean)
                alpha = torch.clamp(alpha, min=0.0, max=1.0)
                raised_std = projected_std + alpha * (cap - projected_std)
                raised_std = torch.where(
                    current_mean <= 0.0,
                    torch.full_like(projected_std, target_mean),
                    raised_std,
                )
                needs_mean_floor = finite_mask.all() & (current_mean < target_mean)
                projected_std = torch.where(
                    needs_mean_floor,
                    raised_std,
                    projected_std,
                )
            capped_std = torch.where(finite_mask, projected_std, std)
            announce_constraint = not self._ppo_start_noise_std_cap_announced
            if announce_constraint:
                announcement = torch.stack(
                    (
                        torch.any(finite_mask & (capped_std < std)).to(dtype=torch.float64),
                        std.mean().to(dtype=torch.float64),
                        capped_std.mean().to(dtype=torch.float64),
                    )
                )
            self.actor.std.data.copy_(capped_std)

        if announce_constraint:
            did_cap_value, old_mean, new_mean = announcement.cpu().tolist()
            if bool(did_cap_value):
                logger.info(
                    "Capped actor noise std for PPO start: mean {:.6f} -> {:.6f} "
                    "(cap={}, until_ppo_coeff={}).",
                    old_mean,
                    new_mean,
                    self.ppo_start_noise_std,
                    self.ppo_start_noise_std_until_coeff,
                )
            else:
                logger.info(
                    "PPO-start noise constraint active without an initial clamp: "
                    "mean {:.6f}, cap={}, until_ppo_coeff={}.",
                    new_mean,
                    self.ppo_start_noise_std,
                    self.ppo_start_noise_std_until_coeff,
                )
            self._ppo_start_noise_std_cap_announced = True

    def _adjust_teacher_action_mix_ratio(self, current_iteration: int) -> None:
        if not getattr(self, "use_teacher_action_mix_schedule", False):
            return
        assert self.teacher_action_mix_ratio_start is not None
        assert self.teacher_action_mix_ratio_end is not None
        if self.teacher_action_mix_ratio_end_iteration <= 0:
            self.teacher_action_mix_ratio = self.teacher_action_mix_ratio_end
            return
        alpha = min(max(float(current_iteration), 0.0) / float(self.teacher_action_mix_ratio_end_iteration), 1.0)
        self.teacher_action_mix_ratio = (
            self.teacher_action_mix_ratio_start
            + (self.teacher_action_mix_ratio_end - self.teacher_action_mix_ratio_start) * alpha
        )

    def _prepare_rollout_objective_for_iteration(self, current_iteration: int) -> None:
        """Apply every objective dependency before observations are computed."""

        self._refresh_distillation_iteration_state(current_iteration)
        self._adjust_teacher_action_mix_ratio(current_iteration)
        self._apply_ppo_start_noise_std_cap(current_iteration)
        fixed_capture_error = self._pure_ppo_fixed_capture_error()
        pure_fixed_capture_preflight = bool(
            getattr(self, "dagger_enabled", False)
            and self._effective_dagger_loss_weight() <= 0.0
            and not self._teacher_controls_current_rollout()
            and int(getattr(self, "fixed_bc_eval_num_samples", 0) or 0) > 0
        )
        if pure_fixed_capture_preflight:
            self._synchronize_training_phase_error(
                fixed_capture_error,
                operation="PPO pure-stream fixed-BC capture preflight",
            )
        # Observation computation is part of the stochastic training stream.
        # Configure it before reset_all as well as before later rollouts.
        self._configure_active_observation_groups()

    def _sync_training_curriculum_state(self, *, current_iteration: int, total_iterations: int) -> None:
        command_manager = getattr(self.env, "command_manager", None)
        if command_manager is None:
            return
        motion_command = command_manager.get_state("motion_command")
        if motion_command is None or not hasattr(motion_command, "set_training_iteration"):
            return
        motion_command.set_training_iteration(current_iteration, total_iterations=total_iterations)

    def _motion_command_supports_runtime_default_pose_prepend_mask(self) -> bool:
        command_manager = getattr(self.env, "command_manager", None)
        if command_manager is None:
            return False
        motion_command = command_manager.get_state("motion_command")
        return motion_command is not None and hasattr(motion_command, "get_runtime_default_pose_prepend_mask")

    def _dagger_episode_age_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return samples past the configured number of actual post-reset steps."""
        episode_steps = getattr(self, "_dagger_episode_step_buf", None)
        if episode_steps is None:
            episode_steps = torch.zeros(batch_size, device=device, dtype=torch.long)
            self._dagger_episode_step_buf = episode_steps
        elif episode_steps.shape != (batch_size,):
            raise RuntimeError(
                "DAgger episode-step buffer shape changed during rollout: "
                f"expected={(batch_size,)}, got={tuple(episode_steps.shape)}."
            )
        elif episode_steps.device != device:
            episode_steps = episode_steps.to(device=device)
            self._dagger_episode_step_buf = episode_steps
        return (episode_steps >= self.dagger_ignore_episode_initial_steps).unsqueeze(1)

    def _advance_dagger_episode_age(self, dones: torch.Tensor) -> None:
        """Advance actual rollout age and reset completed environments to zero."""
        if self.dagger_ignore_episode_initial_steps <= 0:
            return
        episode_steps = getattr(self, "_dagger_episode_step_buf", None)
        if episode_steps is None:
            episode_steps = torch.zeros(dones.numel(), device=dones.device, dtype=torch.long)
        done_mask = dones.to(device=episode_steps.device, dtype=torch.bool).view(-1)
        if done_mask.shape != episode_steps.shape:
            raise RuntimeError(
                "DAgger done mask does not match episode-step buffer: "
                f"dones={tuple(done_mask.shape)}, steps={tuple(episode_steps.shape)}."
            )
        self._dagger_episode_step_buf = torch.where(
            done_mask,
            torch.zeros_like(episode_steps),
            episode_steps + 1,
        )

    def _use_deterministic_student_actions(self) -> bool:
        """Use mean actions during pure BC phases to reduce rollout noise drift."""
        if not self.dagger_enabled:
            return False
        if self.use_ppo_dagger_schedule:
            return self._operational_ppo_coefficient(self.ppo_coeff) <= 0.0
        return self.bc_loss_coef >= 1.0

    def _actor_uses_flow_matching(self) -> bool:
        return bool(
            getattr(
                getattr(self, "actor", None),
                "supports_flow_matching",
                False,
            )
        )

    def _recurrent_hidden_before_step(self, model: nn.Module, batch_size: int) -> torch.Tensor | None:
        gru = getattr(model, "perception_time_gru", None)
        if gru is None:
            return None
        hidden = gru.hidden
        if hidden is None:
            return torch.zeros(
                batch_size,
                gru.hidden_dim,
                device=self.device,
                dtype=next(gru.parameters()).dtype,
            )
        if hidden.shape != (1, batch_size, gru.hidden_dim):
            raise ValueError(
                "Recurrent rollout hidden state has an unexpected shape: "
                f"expected={(1, batch_size, gru.hidden_dim)}, got={tuple(hidden.shape)}."
            )
        return hidden.squeeze(0).detach().clone()

    def _evaluate_critic_preserving_recurrent_state(
        self,
        policy_state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate bootstrap observations without consuming live rollout state."""
        gru = getattr(self.critic, "perception_time_gru", None)
        if gru is None:
            return self.critic.evaluate(policy_state)
        saved_hidden = gru.hidden
        try:
            return self.critic.evaluate(policy_state)
        finally:
            gru.hidden = saved_hidden

    def _assert_rollout_tensors_finite(
        self,
        tensors: dict[str, torch.Tensor | None],
        *,
        phase: str,
        local_error: Exception | None = None,
    ) -> None:
        """Stop every rank at one boundary before invalid data reaches PhysX.

        All local tensors are collapsed into one compact verdict, so this adds
        exactly one small collective per call rather than one collective per
        observation or output.  NaN/Inf must be preserved by policy modules
        until this point; otherwise a corrupt rank could continue with
        apparently valid actions while its peers train on different dynamics.
        """

        finite_checks: list[tuple[str, torch.Tensor]] = []
        invalid_type_names: list[str] = []
        for name, value in tensors.items():
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                invalid_type_names.append(f"{name}<non-tensor:{type(value).__name__}>")
                continue
            if value.is_floating_point() or value.is_complex():
                finite_checks.append(
                    (name, torch.isfinite(value).all().to(device=self.device))
                )

        if finite_checks:
            local_bad = ~torch.stack([check for _, check in finite_checks]).all()
        else:
            local_bad = torch.zeros((), device=self.device, dtype=torch.bool)
        if invalid_type_names:
            local_bad = local_bad | torch.ones((), device=self.device, dtype=torch.bool)
        # Keep runtime exceptions and numerical corruption as separate bits so
        # a peer can report the correct failure class without a second success-
        # path collective.  A rank that failed to construct later tensors can
        # pass them as None and still enter this exact boundary.
        verdict = torch.stack(
            [
                torch.tensor(
                    1 if local_error is not None else 0,
                    device=self.device,
                    dtype=torch.int32,
                ),
                local_bad.to(dtype=torch.int32),
            ]
        )
        if (
            getattr(self, "is_multi_gpu", False)
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            verdict = self._all_reduce_small_tensor(
                verdict,
                op=torch.distributed.ReduceOp.MAX,
            )
        global_error = bool(verdict[0].item())
        global_bad = bool(verdict[1].item())
        if not global_error and not global_bad:
            return

        if global_error:
            if local_error is None:
                detail = "another rank failed while constructing the rollout transition"
            else:
                detail = f"this rank raised {type(local_error).__name__}: {local_error}"
            synchronized_error = RuntimeError(
                f"Rollout transition construction failed at {phase}; {detail}. "
                "All ranks stopped before env.step."
            )
            if local_error is not None:
                raise synchronized_error from local_error
            raise synchronized_error

        local_bad_names = list(invalid_type_names)
        if bool(local_bad.item()):
            local_bad_names.extend(
                name for name, check in finite_checks if not bool(check.item())
            )
        if local_bad_names:
            preview = ", ".join(local_bad_names[:12])
            suffix = "" if len(local_bad_names) <= 12 else f", ... ({len(local_bad_names)} total)"
            detail = f"this rank has NaN/Inf in: {preview}{suffix}"
        else:
            detail = "another rank reported NaN/Inf"
        raise FloatingPointError(
            f"Non-finite rollout state detected at {phase}; {detail}. "
            "Refusing to call env.step with an invalid policy transition."
        )

    def _try_compute_student_rollout_outputs(
        self,
        *,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actor_perception_obs: torch.Tensor | None,
        critic_perception_obs: torch.Tensor | None,
        timing,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        Exception | None,
    ]:
        """Compute rank-local student outputs while preserving the local failure.

        Actor/critic forward paths contain no distributed collectives.  A
        data-dependent exception is returned to the synchronized pre-env
        verdict so every rank stops before env.step.
        """

        try:
            actor_gru_hidden = self._recurrent_hidden_before_step(
                self.actor,
                actor_obs.shape[0],
            )
            critic_objective_enabled = self._critic_optimizer_objective_enabled()
            critic_gru_hidden = (
                self._recurrent_hidden_before_step(
                    self.critic,
                    critic_obs.shape[0],
                )
                if critic_objective_enabled
                else None
            )
            actor_policy_state = {"actor_obs": actor_obs}
            if actor_perception_obs is not None:
                actor_policy_state[self.actor_perception_key] = actor_perception_obs
            if timing is not None:
                with timing.record("rollout/actor_forward"):
                    if self._use_deterministic_student_actions():
                        # A pure-BC rollout consumes the distribution mean and
                        # has no stochastic policy objective. Calling act()
                        # first would draw and discard a sample, advancing the
                        # rank-local torch RNG for no scientific purpose.
                        self.actor.update_distribution_from_policy_state(actor_policy_state)
                        actions = self.actor.action_mean.detach()
                    else:
                        actions = self.actor.act(actor_policy_state)
            else:
                if self._use_deterministic_student_actions():
                    self.actor.update_distribution_from_policy_state(actor_policy_state)
                    actions = self.actor.action_mean.detach()
                else:
                    actions = self.actor.act(actor_policy_state)

            if critic_objective_enabled:
                critic_policy_state = {"critic_obs": critic_obs}
                if critic_perception_obs is not None:
                    critic_policy_state[self.critic_perception_key] = critic_perception_obs
                if timing is not None:
                    with timing.record("rollout/critic_forward"):
                        values = self.critic.evaluate(critic_policy_state).detach()
                else:
                    values = self.critic.evaluate(critic_policy_state).detach()
            else:
                # Pure supervised student training has no return, advantage, or
                # critic objective.  Keep the strict rollout schema finite while
                # avoiding a scientifically irrelevant critic forward (and any
                # dropout RNG / stateful-module side effects it could cause).
                values = actor_obs.new_zeros((actor_obs.shape[0], 1))
        except Exception as exc:
            return None, None, None, None, exc
        return actions, values, actor_gru_hidden, critic_gru_hidden, None

    def _rollout_step(self, obs_dict):
        # Replay must never train on the immutable fixed-BC gate.  Record the
        # readiness boundary before this rollout can capture any diagnostic
        # rows; collection is eligible only on a later rollout where every
        # rank was already complete at entry.
        self._dagger_replay_fixed_eval_ready_at_rollout_start = bool(
            getattr(self, "_fixed_bc_eval_ready", False)
        )
        debug_heartbeat = os.environ.get("HOLOSOMA_DEBUG_HEARTBEAT_VERBOSE", "").lower() not in (
            "",
            "0",
            "false",
            "no",
        )
        timing = self.algo_timing if self.algo_timing.enabled else None
        critic_objective_enabled = self._critic_optimizer_objective_enabled()
        with torch.no_grad():
            for rollout_step in range(self.config.num_steps_per_env):
                # Environment step
                if timing is not None:
                    with timing.record("rollout/obs_cat"):
                        actor_obs_raw = torch.cat([obs_dict[k] for k in self.actor_obs_keys], dim=1)
                        if critic_objective_enabled:
                            critic_obs_raw = torch.cat(
                                [obs_dict[k] for k in self.critic_obs_keys],
                                dim=1,
                            )
                        else:
                            critic_obs_raw = actor_obs_raw.new_zeros(
                                (actor_obs_raw.shape[0], self._get_obs_dim(self.critic_obs_keys))
                            )
                    with timing.record("rollout/obs_normalize"):
                        actor_obs = self._normalize_actor_obs(actor_obs_raw, update=True)
                        critic_obs = (
                            self._normalize_critic_obs(critic_obs_raw, update=True)
                            if critic_objective_enabled
                            else critic_obs_raw
                        )
                else:
                    actor_obs_raw = torch.cat([obs_dict[k] for k in self.actor_obs_keys], dim=1)
                    if critic_objective_enabled:
                        critic_obs_raw = torch.cat(
                            [obs_dict[k] for k in self.critic_obs_keys],
                            dim=1,
                        )
                    else:
                        critic_obs_raw = actor_obs_raw.new_zeros(
                            (actor_obs_raw.shape[0], self._get_obs_dim(self.critic_obs_keys))
                        )

                    actor_obs = self._normalize_actor_obs(actor_obs_raw, update=True)
                    critic_obs = (
                        self._normalize_critic_obs(critic_obs_raw, update=True)
                        if critic_objective_enabled
                        else critic_obs_raw
                    )

                # Keep perception aligned with the same pre-step state/action sample.
                actor_perception_obs_current = (
                    obs_dict.get(self.actor_perception_key) if self.actor_perception_key else None
                )
                critic_perception_obs_current = (
                    obs_dict.get(self.critic_perception_key)
                    if critic_objective_enabled and self.critic_perception_key
                    else None
                )
                required_perception_keys = [self.actor_perception_key]
                if critic_objective_enabled:
                    required_perception_keys.append(self.critic_perception_key)
                missing_perception_keys = [
                    key for key in required_perception_keys if key and key not in obs_dict
                ]
                if missing_perception_keys:
                    actions = None
                    values = None
                    actor_gru_hidden = None
                    critic_gru_hidden = None
                    rollout_compute_error = KeyError(
                        "Missing rollout perception observation(s): "
                        + ", ".join(sorted(set(missing_perception_keys)))
                    )
                else:
                    (
                        actions,
                        values,
                        actor_gru_hidden,
                        critic_gru_hidden,
                        rollout_compute_error,
                    ) = self._try_compute_student_rollout_outputs(
                        actor_obs=actor_obs,
                        critic_obs=critic_obs,
                        actor_perception_obs=actor_perception_obs_current,
                        critic_perception_obs=critic_perception_obs_current,
                        timing=timing,
                    )

                teacher_bc_mask_current = None
                if rollout_compute_error is None and (
                    self.dagger_ignore_episode_initial_steps > 0
                    or self._motion_command_supports_runtime_default_pose_prepend_mask()
                ):
                    try:
                        teacher_bc_mask_current = torch.ones(
                            (actions.shape[0], 1),
                            device=actions.device,
                            dtype=torch.bool,
                        )
                        motion_command = None
                        if self.env.command_manager is not None:
                            motion_command = self.env.command_manager.get_state("motion_command")
                        if self.dagger_ignore_episode_initial_steps > 0:
                            teacher_bc_mask_current &= self._dagger_episode_age_mask(
                                actions.shape[0],
                                actions.device,
                            )
                        if motion_command is not None and hasattr(
                            motion_command,
                            "get_runtime_default_pose_prepend_mask",
                        ):
                            teacher_bc_mask_current &= (
                                ~motion_command.get_runtime_default_pose_prepend_mask()
                            ).unsqueeze(1)
                    except Exception as exc:
                        rollout_compute_error = exc

                teacher_actions = None
                teacher_indices = None
                teacher_obs_raw = None
                actions_to_step = actions
                dagger_loss_active = self._effective_dagger_loss_weight() > 0.0
                teacher_controls_rollout = self._teacher_controls_current_rollout()
                observational_teacher_only = False
                teacher_labels_needed = dagger_loss_active or teacher_controls_rollout
                if self.dagger_enabled and not teacher_labels_needed:
                    observational_teacher_only = self._fixed_bc_eval_capture_pending()
                    if observational_teacher_only:
                        rollout_compute_error = self._pure_ppo_fixed_capture_error()
                if rollout_compute_error is None and self.dagger_enabled and teacher_labels_needed:
                    if timing is not None:
                        with timing.record("rollout/teacher_obs_cat"):
                            try:
                                teacher_obs_raw = self._build_teacher_obs_raw(obs_dict, actor_obs_raw)
                            except Exception as exc:
                                rollout_compute_error = exc
                        with timing.record("rollout/teacher_actions"):
                            if rollout_compute_error is None:
                                (
                                    teacher_actions,
                                    teacher_indices,
                                    rollout_compute_error,
                                ) = self._try_select_teacher_actions_for_rollout(
                                    teacher_obs_raw,
                                    obs_dict,
                                )
                        with timing.record("rollout/teacher_mix"):
                            if rollout_compute_error is None:
                                try:
                                    self._maybe_capture_fixed_bc_eval_samples(
                                        actor_obs_raw=actor_obs_raw,
                                        actor_perception_obs=actor_perception_obs_current,
                                        teacher_actions=teacher_actions,
                                        teacher_bc_mask=teacher_bc_mask_current,
                                    )
                                    if self.teacher_action_mix_ratio > 0.0:
                                        teacher_mask = (
                                            torch.rand((actions.shape[0], 1), device=actions.device)
                                            < self.teacher_action_mix_ratio
                                        )
                                        actions_to_step = torch.where(
                                            teacher_mask,
                                            teacher_actions,
                                            actions,
                                        )
                                    elif self.take_teacher_actions:
                                        actions_to_step = teacher_actions
                                except Exception as exc:
                                    rollout_compute_error = exc
                    else:
                        try:
                            teacher_obs_raw = self._build_teacher_obs_raw(obs_dict, actor_obs_raw)
                        except Exception as exc:
                            rollout_compute_error = exc
                        if rollout_compute_error is None:
                            (
                                teacher_actions,
                                teacher_indices,
                                rollout_compute_error,
                            ) = self._try_select_teacher_actions_for_rollout(
                                teacher_obs_raw,
                                obs_dict,
                            )
                        if rollout_compute_error is None:
                            try:
                                self._maybe_capture_fixed_bc_eval_samples(
                                    actor_obs_raw=actor_obs_raw,
                                    actor_perception_obs=actor_perception_obs_current,
                                    teacher_actions=teacher_actions,
                                    teacher_bc_mask=teacher_bc_mask_current,
                                )
                                if self.teacher_action_mix_ratio > 0.0:
                                    teacher_mask = (
                                        torch.rand((actions.shape[0], 1), device=actions.device)
                                        < self.teacher_action_mix_ratio
                                    )
                                    actions_to_step = torch.where(
                                        teacher_mask,
                                        teacher_actions,
                                        actions,
                                    )
                                elif self.take_teacher_actions:
                                    actions_to_step = teacher_actions
                            except Exception as exc:
                                rollout_compute_error = exc
                elif rollout_compute_error is None and self._legacy_distillation_enabled():
                    # Legacy MSE distillation also needs labels from the
                    # teacher's configured observation groups and perception
                    # input.  Capture them at rollout time while the complete
                    # observation dictionary for this transition is available.
                    try:
                        teacher_obs_raw = self._build_teacher_obs_raw(obs_dict, actor_obs_raw)
                    except Exception as exc:
                        rollout_compute_error = exc
                    if rollout_compute_error is None:
                        (
                            teacher_actions,
                            _,
                            rollout_compute_error,
                        ) = self._try_select_teacher_actions_for_rollout(
                            teacher_obs_raw,
                            obs_dict,
                            stochastic=False,
                        )

                # Compute and validate the exact likelihood metadata before
                # the environment transition.  Historically this happened
                # after env.step, allowing a sanitized NaN actor mean or a bad
                # teacher action to reach PhysX before any finite check.
                rollout_actions_log_prob = None
                if rollout_compute_error is None:
                    try:
                        rollout_actions_log_prob = (
                            self.actor.get_actions_log_prob(actions).detach().unsqueeze(1)
                        )
                    except Exception as exc:
                        rollout_compute_error = exc

                teacher_perception_checks: dict[str, torch.Tensor | None] = {}
                teacher_candidates = (
                    (
                        list(self.teacher_actors)
                        if self.use_multi_teacher
                        else ([self.teacher_actor] if self.teacher_actor is not None else [])
                    )
                    if teacher_actions is not None
                    else []
                )
                for teacher_index, teacher_candidate in enumerate(teacher_candidates):
                    perception_key = str(
                        getattr(teacher_candidate, "perception_input_name", "") or ""
                    )
                    if perception_key:
                        teacher_perception_checks[
                            f"teacher_perception_obs[{teacher_index}:{perception_key}]"
                        ] = obs_dict.get(perception_key)
                finite_tensors = {
                    "actor_obs_raw": actor_obs_raw,
                    "critic_obs_raw": critic_obs_raw,
                    "actor_obs": actor_obs,
                    "critic_obs": critic_obs,
                    "actor_perception_obs": actor_perception_obs_current,
                    "critic_perception_obs": critic_perception_obs_current,
                    "actor_actions": actions,
                    "actor_action_mean": self.actor.action_mean if actions is not None else None,
                    "actor_action_std": self.actor.action_std if actions is not None else None,
                    "actor_actions_log_prob": rollout_actions_log_prob,
                    "critic_values": values,
                    "teacher_obs_raw": teacher_obs_raw,
                    "teacher_actions": teacher_actions,
                    "multi_teacher_selector": (
                        obs_dict.get(self.multi_teacher_select_obs_var)
                        if teacher_indices is not None
                        else None
                    ),
                    "actions_to_step": actions_to_step,
                    **teacher_perception_checks,
                }
                finite_phase = (
                    f"iteration {self.current_learning_iteration} rollout step "
                    f"{rollout_step + 1}/{self.config.num_steps_per_env} before env.step"
                )
                self._assert_rollout_tensors_finite(
                    finite_tensors,
                    phase=finite_phase,
                    local_error=rollout_compute_error,
                )
                assert actions is not None
                assert values is not None
                assert actions_to_step is not None
                assert rollout_actions_log_prob is not None

                if debug_heartbeat:
                    logger.info(
                        "Heartbeat: iter {} rollout_step {}/{} before env.step",
                        self.current_learning_iteration,
                        rollout_step + 1,
                        self.config.num_steps_per_env,
                    )
                if timing is not None:
                    with timing.record("rollout/env_step"):
                        obs_dict, rewards, dones, infos = self.env.step({"actions": actions_to_step})
                else:
                    obs_dict, rewards, dones, infos = self.env.step({"actions": actions_to_step})
                if debug_heartbeat:
                    timeout_count = 0
                    if isinstance(infos, dict) and "time_outs" in infos and infos["time_outs"] is not None:
                        timeout_count = int(infos["time_outs"].sum().item())
                    logger.info(
                        "Heartbeat: iter {} rollout_step {}/{} after env.step (done_envs={}, timeout_envs={})",
                        self.current_learning_iteration,
                        rollout_step + 1,
                        self.config.num_steps_per_env,
                        int(dones.sum().item()),
                        timeout_count,
                    )

                if timing is not None:
                    with timing.record("rollout/device_transfer"):
                        for obs_key in obs_dict:
                            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                        rewards, dones = rewards.to(self.device), dones.to(self.device)
                else:
                    for obs_key in obs_dict:
                        obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                self._advance_dagger_episode_age(dones)

                # Compute bootstrap value for timeouts
                final_rewards = torch.zeros_like(rewards)
                has_timeout_final_observations = self._has_timeout_final_observations(infos)
                if timing is not None:
                    with timing.record("rollout/final_timeout_bootstrap"):
                        if critic_objective_enabled and has_timeout_final_observations:
                            final_critic_obs = torch.cat(
                                [infos["final_observations"][k] for k in self.critic_obs_keys], dim=1
                            )
                            # Timeout final observations are rank-local and conditional. Updating distributed
                            # normalizers here would desynchronize all_reduce order across ranks.
                            final_critic_obs = self._normalize_critic_obs(final_critic_obs, update=False)
                            final_policy_state = {"critic_obs": final_critic_obs}
                            if (
                                self.critic_perception_key
                                and self.critic_perception_key in infos["final_observations"]
                            ):
                                final_policy_state[self.critic_perception_key] = infos["final_observations"][
                                    self.critic_perception_key
                                ]
                            final_values = self._evaluate_critic_preserving_recurrent_state(final_policy_state).detach()
                            final_rewards += self.config.gamma * torch.squeeze(
                                final_values * infos["time_outs"].unsqueeze(1).to(self.device), 1
                            )
                else:
                    if critic_objective_enabled and has_timeout_final_observations:
                        final_critic_obs = torch.cat([infos["final_observations"][k] for k in self.critic_obs_keys], dim=1)
                        # Timeout final observations are rank-local and conditional. Updating distributed
                        # normalizers here would desynchronize all_reduce order across ranks.
                        final_critic_obs = self._normalize_critic_obs(final_critic_obs, update=False)
                        final_policy_state = {"critic_obs": final_critic_obs}
                        if (
                            self.critic_perception_key
                            and self.critic_perception_key in infos["final_observations"]
                        ):
                            final_policy_state[self.critic_perception_key] = infos["final_observations"][
                                self.critic_perception_key
                            ]
                        final_values = self._evaluate_critic_preserving_recurrent_state(final_policy_state).detach()
                        final_rewards += self.config.gamma * torch.squeeze(
                            final_values * infos["time_outs"].unsqueeze(1).to(self.device), 1
                        )

                if timing is not None:
                    with timing.record("rollout/storage_add"):
                        storage_kwargs = {
                            "actor_obs": actor_obs,
                            "critic_obs": critic_obs,
                            "actor_obs_raw": actor_obs_raw,
                            "critic_obs_raw": critic_obs_raw,
                            "actions": actions,
                            "values": values,
                            "actions_log_prob": rollout_actions_log_prob,
                            "action_mean": self.actor.action_mean.detach(),
                            "action_sigma": self.actor.action_std.detach(),
                            "rewards": (rewards + final_rewards).view(-1, 1),
                            "dones": dones.to(dtype=torch.bool).view(-1, 1),
                            "teacher_actions": teacher_actions.detach()
                            if teacher_actions is not None
                            else torch.zeros_like(actions),
                            "teacher_indices": teacher_indices.view(-1, 1)
                            if teacher_indices is not None
                            else torch.zeros(actions.shape[0], 1, device=actions.device, dtype=torch.long),
                            "actor_gru_hidden": actor_gru_hidden,
                            "critic_gru_hidden": critic_gru_hidden,
                        }
                        if teacher_bc_mask_current is not None:
                            storage_kwargs["teacher_bc_mask"] = teacher_bc_mask_current
                        if actor_perception_obs_current is not None:
                            storage_kwargs[self.actor_perception_key] = actor_perception_obs_current
                        if (
                            critic_perception_obs_current is not None
                            and self.critic_perception_key != self.actor_perception_key
                        ):
                            storage_kwargs[self.critic_perception_key] = critic_perception_obs_current
                        self._add_rollout_storage_transition(storage_kwargs)
                else:
                    storage_kwargs = {
                        "actor_obs": actor_obs,
                        "critic_obs": critic_obs,
                        "actor_obs_raw": actor_obs_raw,
                        "critic_obs_raw": critic_obs_raw,
                        "actions": actions,
                        "values": values,
                        "actions_log_prob": rollout_actions_log_prob,
                        "action_mean": self.actor.action_mean.detach(),
                        "action_sigma": self.actor.action_std.detach(),
                        "rewards": (rewards + final_rewards).view(-1, 1),
                        "dones": dones.to(dtype=torch.bool).view(-1, 1),
                        "teacher_actions": teacher_actions.detach()
                        if teacher_actions is not None
                        else torch.zeros_like(actions),
                        "teacher_indices": teacher_indices.view(-1, 1)
                        if teacher_indices is not None
                        else torch.zeros(actions.shape[0], 1, device=actions.device, dtype=torch.long),
                        "actor_gru_hidden": actor_gru_hidden,
                        "critic_gru_hidden": critic_gru_hidden,
                    }
                    if teacher_bc_mask_current is not None:
                        storage_kwargs["teacher_bc_mask"] = teacher_bc_mask_current
                    if actor_perception_obs_current is not None:
                        storage_kwargs[self.actor_perception_key] = actor_perception_obs_current
                    if (
                        critic_perception_obs_current is not None
                        and self.critic_perception_key != self.actor_perception_key
                    ):
                        storage_kwargs[self.critic_perception_key] = critic_perception_obs_current
                    self._add_rollout_storage_transition(storage_kwargs)

                # Reset actor and critic for completed envs
                if timing is not None:
                    with timing.record("rollout/model_reset"):
                        self.actor.reset(dones)
                        if critic_objective_enabled:
                            self.critic.reset(dones)
                        if teacher_actions is not None:
                            if self.use_multi_teacher:
                                for teacher_actor in self.teacher_actors:
                                    teacher_actor.reset(dones)
                            elif self.teacher_actor is not None:
                                self.teacher_actor.reset(dones)
                else:
                    self.actor.reset(dones)
                    if critic_objective_enabled:
                        self.critic.reset(dones)
                    if teacher_actions is not None:
                        if self.use_multi_teacher:
                            for teacher_actor in self.teacher_actors:
                                teacher_actor.reset(dones)
                        elif self.teacher_actor is not None:
                            self.teacher_actor.reset(dones)

                if self.log_dir is not None:
                    # Update episode stats using logging helper
                    if timing is not None:
                        with timing.record("rollout/episode_stats"):
                            self.logging_helper.update_episode_stats(rewards, dones, infos)
                    else:
                        self.logging_helper.update_episode_stats(rewards, dones, infos)

            # Return / Advantage computation
            if timing is not None:
                with timing.record("rollout/returns"):
                    if critic_objective_enabled:
                        last_critic_obs = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)
                        last_critic_obs = self._normalize_critic_obs(last_critic_obs, update=False)
                        last_policy_state = {"critic_obs": last_critic_obs}
                        if self.critic_perception_key and self.critic_perception_key in obs_dict:
                            last_policy_state[self.critic_perception_key] = obs_dict[self.critic_perception_key]
                        last_values = self._evaluate_critic_preserving_recurrent_state(last_policy_state).detach().to(self.device)
                        returns, advantages = self._compute_returns_and_advantages(
                            last_values,
                            self.storage["values"].to(self.device),
                            self.storage["dones"].to(self.device),
                            self.storage["rewards"].to(self.device),
                        )
                    else:
                        returns = torch.zeros_like(self.storage["values"])
                        advantages = torch.zeros_like(self.storage["values"])

                    self.storage["returns"] = returns
                    self.storage["advantages"] = advantages
            else:
                if critic_objective_enabled:
                    last_critic_obs = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)
                    last_critic_obs = self._normalize_critic_obs(last_critic_obs, update=False)
                    last_policy_state = {"critic_obs": last_critic_obs}
                    if self.critic_perception_key and self.critic_perception_key in obs_dict:
                        last_policy_state[self.critic_perception_key] = obs_dict[self.critic_perception_key]
                    last_values = self._evaluate_critic_preserving_recurrent_state(last_policy_state).detach().to(self.device)
                    returns, advantages = self._compute_returns_and_advantages(
                        last_values,
                        self.storage["values"].to(self.device),
                        self.storage["dones"].to(self.device),
                        self.storage["rewards"].to(self.device),
                    )
                else:
                    returns = torch.zeros_like(self.storage["values"])
                    advantages = torch.zeros_like(self.storage["values"])

                self.storage["returns"] = returns
                self.storage["advantages"] = advantages

        return obs_dict

    def _compute_returns_and_advantages(self, last_values, values, dones, rewards):
        advantage = 0
        returns = torch.zeros_like(values)
        num_steps = returns.shape[0]
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * self.config.gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * self.config.gamma * self.config.lam * advantage
            returns[step] = advantage + values[step]
        advantages = returns - values

        if self.is_multi_gpu:
            advantages = self._normalize_advantages_multi_gpu(advantages)
        else:
            local_mean = advantages.mean()
            local_variance = torch.mean((advantages - local_mean) ** 2)
            advantages = (advantages - local_mean) / torch.sqrt(local_variance + 1e-8)

        return returns, advantages

    def _assert_rollout_storage_finite(self) -> None:
        """Validate every filled floating rollout buffer before optimization."""

        storage = getattr(self, "storage", None)
        buffers = getattr(storage, "_buffers", {})
        filled_steps = int(getattr(storage, "step", 0) or 0)
        finite_checks: list[tuple[str, torch.Tensor]] = []
        if isinstance(buffers, dict) and filled_steps > 0:
            for name, buffer in buffers.items():
                if not isinstance(buffer, torch.Tensor) or not (
                    buffer.is_floating_point() or buffer.is_complex()
                ):
                    continue
                buffer_steps = min(filled_steps, int(buffer.shape[0]))
                buffer_is_finite = torch.ones(
                    (),
                    device=buffer.device,
                    dtype=torch.bool,
                )
                # Perception rollout buffers can be close to a GiB. Checking
                # the whole filled buffer at once creates an equally shaped
                # boolean temporary, so bound the peak to one rollout step.
                for step in range(buffer_steps):
                    buffer_is_finite.logical_and_(
                        torch.isfinite(buffer[step]).all()
                    )
                finite_checks.append(
                    (
                        name,
                        buffer_is_finite.to(device=self.device),
                    )
                )
        if finite_checks:
            local_bad = ~torch.stack([check for _, check in finite_checks]).all()
        else:
            local_bad = torch.zeros((), device=self.device, dtype=torch.bool)

        global_bad = local_bad.to(dtype=torch.int32)
        if (
            getattr(self, "is_multi_gpu", False)
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            global_bad = self._all_reduce_small_tensor(
                global_bad,
                op=torch.distributed.ReduceOp.MAX,
            )
        if not bool(global_bad.item()):
            return
        local_bad_names = [
            name for name, check in finite_checks if not bool(check.item())
        ]
        if bool(local_bad.item()):
            if local_bad_names == ["teacher_actions"]:
                detail = "this rank's filled teacher_actions rollout contains NaN/Inf"
            else:
                preview = ", ".join(local_bad_names[:12])
                suffix = "" if len(local_bad_names) <= 12 else f", ... ({len(local_bad_names)} total)"
                detail = f"this rank's filled rollout contains NaN/Inf in: {preview}{suffix}"
        else:
            detail = "another rank reported NaN/Inf in its filled rollout"
        raise FloatingPointError(
            f"Non-finite rollout storage detected before PPO minibatch generation; {detail}."
        )

    def _assert_rollout_teacher_actions_finite(self) -> None:
        """Backward-compatible entry point for the expanded storage check."""

        self._assert_rollout_storage_finite()

    def _all_ranks_fixed_bc_ready_before_rollout(self) -> bool:
        local_ready = bool(
            getattr(
                self,
                "_dagger_replay_fixed_eval_ready_at_rollout_start",
                False,
            )
        )
        ready_count = torch.tensor(
            int(local_ready), device=self.device, dtype=torch.int32
        )
        if getattr(self, "is_multi_gpu", False):
            ready_count = self._all_reduce_small_tensor(
                ready_count,
                op=torch.distributed.ReduceOp.SUM,
            )
        return int(ready_count.item()) == int(
            getattr(self, "gpu_world_size", 1) or 1
        )

    def _insert_current_rollout_into_dagger_replay(self) -> None:
        buffer = getattr(self, "_dagger_replay_buffer", None)
        if not self.dagger_replay_enabled or not isinstance(
            buffer, _DaggerReplayBuffer
        ):
            return
        filled_steps = int(getattr(self.storage, "step", 0) or 0)
        local_error: Exception | None = None
        try:
            if filled_steps <= 0:
                raise RuntimeError(
                    "Cannot aggregate DAgger replay before rollout storage is filled."
                )
            if not bool(
                getattr(self, "_dagger_replay_fixed_boundary_verified_runtime", False)
            ):
                # The collective digest call is intentionally outside this
                # local try block below.  Every rank enters it exactly once,
                # after the readiness SUM above established a common branch.
                pass
        except Exception as exc:
            local_error = exc
        self._synchronize_training_phase_error(
            local_error,
            operation="DAgger replay rollout preflight",
        )

        if not bool(
            getattr(self, "_dagger_replay_fixed_boundary_verified_runtime", False)
        ):
            _digest_by_rank, global_digest = (
                self._fixed_bc_guard_live_dataset_digests()
            )
            bind_error: Exception | None = None
            try:
                buffer.bind_fixed_dataset(
                    iteration=int(self.current_learning_iteration),
                    global_digest=global_digest,
                )
            except Exception as exc:
                bind_error = exc
            self._synchronize_training_phase_error(
                bind_error,
                operation="DAgger replay fixed-BC boundary binding",
            )
            self._dagger_replay_fixed_boundary_verified_runtime = True

        insert_error: Exception | None = None
        try:
            actor_obs_raw = self.storage["actor_obs_raw"][:filled_steps].reshape(
                -1, buffer.actor_obs_dim
            )
            teacher_actions = self.storage["teacher_actions"][:filled_steps].reshape(
                -1, buffer.action_dim
            )
            if self.actor_perception_key:
                actor_perception = self.storage[self.actor_perception_key][
                    :filled_steps
                ].reshape(-1, buffer.actor_perception_dim)
            else:
                actor_perception = None
            try:
                mask = self.storage["teacher_bc_mask"][:filled_steps].reshape(-1, 1)
            except KeyError:
                mask = torch.ones(
                    (actor_obs_raw.shape[0], 1),
                    device=actor_obs_raw.device,
                    dtype=torch.bool,
                )
            mask = mask.to(dtype=torch.bool)
            if self.dagger_ignore_zero_teacher_actions:
                mask = mask & (~torch.all(teacher_actions == 0.0, dim=-1)).view(-1, 1)
            buffer.insert(
                actor_obs_raw=actor_obs_raw,
                actor_perception=actor_perception,
                teacher_actions=teacher_actions,
                mask=mask,
                # _training_step entered through _assert_rollout_storage_finite;
                # avoid repeating three whole-rollout reductions and a host
                # synchronization inside the bounded replay copy.
                finite_prevalidated=True,
            )
        except Exception as exc:
            insert_error = exc
        self._synchronize_training_phase_error(
            insert_error,
            operation="DAgger replay deterministic reservoir insertion",
        )

    def _prepare_dagger_replay_update_plan(
        self,
        *,
        num_updates: int,
    ) -> list[dict[str, Any]] | None:
        """Insert once and precompute one collective-safe replay batch per update."""

        if not bool(getattr(self, "dagger_replay_enabled", False)):
            return None
        num_updates = self._strict_positive_ppo_int("replay num_updates", num_updates)
        if not self._all_ranks_fixed_bc_ready_before_rollout():
            return None
        self._insert_current_rollout_into_dagger_replay()
        buffer = getattr(self, "_dagger_replay_buffer", None)
        if not isinstance(buffer, _DaggerReplayBuffer):
            raise RuntimeError("Enabled DAgger replay has no initialized rank-local buffer.")

        batches: list[dict[str, torch.Tensor]] = []
        local_counts: list[float] = []
        sample_error: Exception | None = None
        try:
            for _ in range(num_updates):
                batch = buffer.sample(self.dagger_replay_batch_size)
                valid_mask = batch["mask"].view(-1).to(dtype=torch.bool)
                if self.dagger_ignore_zero_teacher_actions:
                    valid_mask &= ~torch.all(batch["teacher_actions"] == 0.0, dim=-1)
                batches.append(batch)
                local_counts.append(float(valid_mask.sum().item()))
        except Exception as exc:
            sample_error = exc
        self._synchronize_training_phase_error(
            sample_error,
            operation="DAgger replay deterministic sampling",
        )

        count_error: Exception | None = None
        weighted_counts: torch.Tensor | None = None
        try:
            weighted_counts = torch.tensor(
                local_counts, device=self.device, dtype=torch.float32
            )
            if getattr(self, "is_multi_gpu", False):
                # _reduce_parameters applies distributed_loss_weight before
                # averaging gradients across ranks.  Mirror that factor only
                # in the distributed denominator; a single-rank optimizer
                # step does not apply it and must use the ordinary local mean.
                weighted_counts = (
                    weighted_counts * self._get_distributed_loss_weight()
                )
        except Exception as exc:
            count_error = exc
        self._synchronize_training_phase_error(
            count_error,
            operation="DAgger replay weighted-count construction",
        )
        assert weighted_counts is not None
        if getattr(self, "is_multi_gpu", False):
            weighted_counts = self._all_reduce_small_tensor(
                weighted_counts,
                op=torch.distributed.ReduceOp.SUM,
            )
        global_presence = weighted_counts > 0.0
        if not bool(global_presence.any().item()):
            return None
        if not bool(global_presence.all().item()):
            raise RuntimeError(
                "DAgger replay sampling produced an inconsistent empty/non-empty update plan."
            )
        denominators = weighted_counts
        if getattr(self, "is_multi_gpu", False):
            denominators = denominators / float(
                getattr(self, "gpu_world_size", 1) or 1
            )
        if not bool(torch.isfinite(denominators).all().item()) or not bool(
            (denominators > 0.0).all().item()
        ):
            raise ValueError(
                "DAgger replay global weighted denominators must be finite and positive."
            )
        return [
            {
                "batch": batch,
                "denominator": denominators[index],
                "has_valid_samples": True,
            }
            for index, batch in enumerate(batches)
        ]

    def _training_step(self) -> dict[str, float]:
        timing = self.algo_timing if self.algo_timing.enabled else None
        debug_heartbeat = os.environ.get("HOLOSOMA_DEBUG_HEARTBEAT_VERBOSE", "").lower() not in (
            "",
            "0",
            "false",
            "no",
        )
        rank_label = f"{getattr(self, 'gpu_global_rank', 0)}/{getattr(self, 'gpu_world_size', 1)}"
        self._assert_rollout_storage_finite()
        num_updates = self.config.num_learning_epochs * self.config.num_mini_batches
        dagger_bc_denominator = self._rollout_bc_denominator_per_minibatch()
        dagger_bc_minibatch_presence = (
            self._rollout_bc_minibatch_presence()
            if dagger_bc_denominator is not None
            else None
        )
        dagger_controls_authoritative = bool(
            dagger_bc_denominator is not None
            and dagger_bc_minibatch_presence is not None
        )
        dagger_replay_plan = self._prepare_dagger_replay_update_plan(
            num_updates=num_updates
        )
        minibatch_keys = {
            "actor_obs",
            "critic_obs",
            "actions",
            "values",
            "advantages",
            "returns",
            "actions_log_prob",
            "action_mean",
            "action_sigma",
        }
        if self.distill_enabled:
            minibatch_keys.add("teacher_actions")
        if self.dagger_enabled:
            if self.use_multi_teacher:
                minibatch_keys.add("teacher_indices")
            if "teacher_bc_mask" in self.storage.registered_keys:
                minibatch_keys.add("teacher_bc_mask")
        if self.actor_perception_key:
            minibatch_keys.add(self.actor_perception_key)
        if self._critic_optimizer_objective_enabled() and self.critic_perception_key:
            minibatch_keys.add(self.critic_perception_key)
        if self.use_time_gru:
            minibatch_keys.add("dones")
        if self.use_symmetry:
            minibatch_keys.add("actor_obs_raw")
            minibatch_keys.add("critic_obs_raw")
        unregistered_minibatch_keys = minibatch_keys.difference(
            self.storage.registered_keys
        )
        if unregistered_minibatch_keys:
            raise RuntimeError(
                "PPO requested minibatch fields absent from rollout storage: "
                f"{sorted(unregistered_minibatch_keys)}"
            )
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} training_step enter (epochs={}, mini_batches={})",
                self.current_learning_iteration,
                rank_label,
                self.config.num_learning_epochs,
                self.config.num_mini_batches,
            )
        if timing is not None:
            with timing.record("training/generator_setup"):
                if self.use_time_gru:
                    generator = self.storage.sequence_mini_batch_generator(
                        self.config.num_mini_batches, self.config.num_learning_epochs
                    )
                else:
                    generator = self.storage.mini_batch_generator(
                        self.config.num_mini_batches, self.config.num_learning_epochs, keys=minibatch_keys
                    )
        else:
            if self.use_time_gru:
                generator = self.storage.sequence_mini_batch_generator(
                    self.config.num_mini_batches, self.config.num_learning_epochs
                )
            else:
                generator = self.storage.mini_batch_generator(
                    self.config.num_mini_batches, self.config.num_learning_epochs, keys=minibatch_keys
                )
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} training_step generator ready",
                self.current_learning_iteration,
                rank_label,
            )

        minibatch: Minibatch
        # Keep diagnostic losses on their producing device while minibatches
        # are being optimized.  Converting every field with ``.item()`` here
        # would serialize CUDA once or twice per field and per minibatch.  The
        # deferred accumulator retains detached scalars without any autograd
        # graph, performs one batched device-to-host transfer at the
        # synchronized iteration boundary below, and only then repeats the
        # historical Python-float additions in minibatch order.  This avoids
        # changing even the diagnostic-loss rounding order.
        loss_dict: dict[str, list[torch.Tensor | float]] = {
            "Value": [],
            "Surrogate": [],
            "Entropy": [],
            "KL": [],
        }
        minibatch_idx = 0
        self._debug_training_phase("training_loop_begin")
        for minibatch in generator:
            minibatch_idx += 1
            if dagger_bc_denominator is not None:
                # Every optimization epoch partitions the complete rollout
                # into exactly num_mini_batches pieces.  A fixed denominator
                # gives every valid rollout sample equal weight even when a
                # contiguous mask leaves minibatches with very different
                # valid counts.
                minibatch = dict(minibatch)
                minibatch["_dagger_bc_denominator"] = dagger_bc_denominator
                if dagger_bc_minibatch_presence is not None:
                    if minibatch_idx > len(dagger_bc_minibatch_presence):
                        raise RuntimeError(
                            "DAgger minibatch generator produced more batches than the cached "
                            "global-presence plan."
                        )
                    minibatch["_dagger_bc_has_valid_samples"] = (
                        dagger_bc_minibatch_presence[minibatch_idx - 1]
                    )
            if dagger_replay_plan is not None:
                if minibatch_idx > len(dagger_replay_plan):
                    raise RuntimeError(
                        "DAgger minibatch generator produced more batches than the replay plan."
                    )
                minibatch = dict(minibatch)
                minibatch["_dagger_replay"] = dagger_replay_plan[minibatch_idx - 1]
            self._debug_current_minibatch_idx = minibatch_idx
            self._debug_training_phase("minibatch_update_begin")
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} minibatch {} starting update",
                    self.current_learning_iteration,
                    rank_label,
                    minibatch_idx,
                )
            if timing is not None:
                with timing.record("training/update_algo_step"):
                    loss_dict = self._update_algo_step(
                        minibatch,
                        loss_dict,
                        defer_loss_host_sync=True,
                        dagger_controls_authoritative=dagger_controls_authoritative,
                    )
            else:
                loss_dict = self._update_algo_step(
                    minibatch,
                    loss_dict,
                    defer_loss_host_sync=True,
                    dagger_controls_authoritative=dagger_controls_authoritative,
                )
            self._debug_training_phase("minibatch_update_finished")
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} minibatch {} finished update",
                    self.current_learning_iteration,
                    rank_label,
                    minibatch_idx,
                )

        if (
            dagger_bc_minibatch_presence is not None
            and minibatch_idx != len(dagger_bc_minibatch_presence)
        ):
            raise RuntimeError(
                "DAgger minibatch generator did not consume the complete cached "
                "global-presence plan: "
                f"expected={len(dagger_bc_minibatch_presence)}, actual={minibatch_idx}."
            )
        if dagger_replay_plan is not None and minibatch_idx != len(dagger_replay_plan):
            raise RuntimeError(
                "DAgger minibatch generator did not consume the complete replay plan: "
                f"expected={len(dagger_replay_plan)}, actual={minibatch_idx}."
            )

        # Parameters only change in optimizer steps.  Check once at this
        # synchronized iteration boundary so a corrupt policy cannot enter the
        # next rollout while avoiding per-env-step/per-minibatch collectives.
        self._assert_model_parameters_finite(
            phase=f"training iteration {self.current_learning_iteration}",
            trainable_only=True,
            include_optimizer_state=True,
        )

        final_loss_extras: dict[str, torch.Tensor | float] = {}
        try:
            final_loss_extras["teacher_bc_mask_fraction"] = (
                self.storage["teacher_bc_mask"].float().mean()
            )
        except KeyError:
            pass
        replay_buffer = getattr(self, "_dagger_replay_buffer", None)
        if isinstance(replay_buffer, _DaggerReplayBuffer):
            final_loss_extras.update(
                {
                    "dagger_replay_buffer_size": float(replay_buffer.size),
                    "dagger_replay_seen_valid_count": float(
                        replay_buffer.seen_valid_count
                    ),
                    "dagger_replay_sample_draw_count": float(
                        replay_buffer.sample_draw_count
                    ),
                    "dagger_replay_active": float(dagger_replay_plan is not None),
                }
            )
        loss_dict = self._finalize_deferred_loss_dict(
            loss_dict,
            num_updates=num_updates,
            extras=final_loss_extras,
        )
        if timing is not None:
            with timing.record("training/storage_clear"):
                self.storage.clear()
        else:
            self.storage.clear()
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} training_step exit after {} minibatches",
                self.current_learning_iteration,
                rank_label,
                minibatch_idx,
            )
        self._debug_training_phase("training_loop_exit", minibatch_count=minibatch_idx)
        return loss_dict

    @staticmethod
    def _loss_to_float(loss: torch.Tensor | float | int) -> float:
        if torch.is_tensor(loss):
            loss = loss.detach()
            if loss.is_complex():
                raise TypeError("Loss/logging scalars must be real-valued, not complex.")
            if not bool(torch.isfinite(loss).all().item()):
                raise FloatingPointError("Refusing to convert a NaN/Inf loss value into a logging scalar.")
            if loss.numel() != 1:
                loss = loss.mean()
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(
                        "A finite loss tensor overflowed to NaN/Inf while reducing it for logging."
                    )
            return float(loss.item())
        loss_value = float(loss)
        if not math.isfinite(loss_value):
            raise FloatingPointError(
                f"Refusing to convert a non-finite loss value into a logging scalar: {loss_value!r}."
            )
        return loss_value

    @staticmethod
    def _loss_is_finite(loss: torch.Tensor | float | int) -> bool:
        if torch.is_tensor(loss):
            return bool(torch.isfinite(loss).all())
        loss_value = float(loss)
        return loss_value == loss_value and loss_value not in (float("inf"), float("-inf"))

    @staticmethod
    def _invalid_loss_fields_batched(
        losses: dict[str, torch.Tensor | float | int],
    ) -> list[str]:
        """Locate invalid loss fields with one host transfer per device.

        The former field-by-field ``bool(tensor)`` loop synchronized CUDA for
        every diagnostic scalar.  Keep all predicates on-device, batch-copy
        their boolean results once, and retain the exact field-level error
        report.  For non-scalar diagnostics also validate the source-dtype
        mean used by logging, so that deferred logging cannot hide a reduction
        overflow until after later optimizer steps.
        """

        tensor_checks: dict[torch.device, list[tuple[str, torch.Tensor]]] = {}
        invalid_names: set[str] = set()
        for field, value in losses.items():
            if torch.is_tensor(value):
                try:
                    detached = value.detach()
                    if detached.is_complex():
                        invalid_names.add(field)
                        continue
                    finite = torch.isfinite(detached).all()
                    if detached.numel() != 1:
                        finite = finite & torch.isfinite(detached.mean())
                    tensor_checks.setdefault(finite.device, []).append((field, finite))
                except (TypeError, ValueError, RuntimeError, OverflowError):
                    invalid_names.add(field)
                continue
            try:
                scalar = float(value)
                if not math.isfinite(scalar):
                    invalid_names.add(field)
            except (TypeError, ValueError, OverflowError):
                invalid_names.add(field)

        for entries in tensor_checks.values():
            host_checks = torch.stack([check.reshape(()) for _, check in entries]).cpu().tolist()
            invalid_names.update(
                field
                for (field, _), is_finite in zip(entries, host_checks)
                if not bool(is_finite)
            )
        return [field for field in losses if field in invalid_names]

    @staticmethod
    def _env_truthy(name: str) -> bool:
        return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")

    def _debug_training_phase(self, phase: str, **values) -> None:
        if not self._env_truthy("HOLOSOMA_DEBUG_TRAINING_PHASES"):
            return

        rank = int(getattr(self, "gpu_global_rank", 0))
        world_size = int(getattr(self, "gpu_world_size", 1))
        minibatch_idx = values.pop("minibatch", getattr(self, "_debug_current_minibatch_idx", None))
        parts = [
            f"iter={self.current_learning_iteration}",
            f"rank={rank}/{world_size}",
            f"minibatch={minibatch_idx}",
            f"phase={phase}",
        ]
        for key, value in values.items():
            parts.append(f"{key}={value}")
        message = "TrainPhase " + " ".join(parts)

        phase_dir = os.environ.get("HOLOSOMA_DEBUG_TRAINING_PHASE_DIR", "").strip()
        if phase_dir:
            path = Path(phase_dir).expanduser() / f"rank_{rank}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
            return
        print(message, flush=True)

    @staticmethod
    def _loss_to_deferred_scalar(loss: torch.Tensor | float | int) -> torch.Tensor | float:
        """Return a detached scalar without synchronizing a CUDA device."""

        if torch.is_tensor(loss):
            scalar = loss.detach()
            if scalar.is_complex():
                raise TypeError("Loss/logging scalars must be real-valued, not complex.")
            if scalar.numel() != 1:
                # Match the historical logging definition exactly: compute the
                # mean in the source dtype.  Widening, transfer, and Python-float
                # addition are deliberately deferred to the iteration boundary.
                scalar = scalar.mean()
            return scalar.reshape(())
        loss_value = float(loss)
        if not math.isfinite(loss_value):
            raise FloatingPointError(
                f"Refusing to accumulate a non-finite logging scalar: {loss_value!r}."
            )
        return loss_value

    @staticmethod
    def _add_deferred_loss_scalar(
        loss_dict: dict[str, Any],
        key: str,
        loss: torch.Tensor | float | int,
    ) -> None:
        scalar = PPO._loss_to_deferred_scalar(loss)
        current = loss_dict.get(key)
        if current is None:
            loss_dict[key] = [scalar]
        elif isinstance(current, list):
            current.append(scalar)
        else:
            # Backward-compatible support for callers that seed the mapping
            # with numeric zeros instead of empty lists.
            loss_dict[key] = [current, scalar]

    @staticmethod
    def _finalize_deferred_loss_dict(
        loss_dict: dict[str, Any],
        *,
        num_updates: int,
        extras: dict[str, torch.Tensor | float] | None = None,
    ) -> dict[str, float]:
        """Batch-copy detached loss scalars to the host once per device."""

        if num_updates <= 0:
            raise ValueError(f"num_updates must be positive, got {num_updates}.")

        pending: dict[torch.device, list[tuple[str, int, torch.Tensor]]] = {}
        ordered_values: dict[str, list[float | None]] = {}
        finalized: dict[str, float] = {}

        for key, raw_values in loss_dict.items():
            values = raw_values if isinstance(raw_values, list) else [raw_values]
            slots: list[float | None] = []
            for value in values:
                if torch.is_tensor(value):
                    scalar = value.detach().reshape(()).to(dtype=torch.float64)
                    slot_idx = len(slots)
                    slots.append(None)
                    pending.setdefault(scalar.device, []).append((key, slot_idx, scalar))
                else:
                    scalar_value = float(value)
                    if not math.isfinite(scalar_value):
                        raise FloatingPointError(
                            f"Refusing to finalize non-finite loss field {key}: {scalar_value!r}."
                        )
                    slots.append(scalar_value)
            ordered_values[key] = slots

        extra_values = extras or {}
        for key in extra_values:
            if key in loss_dict:
                raise KeyError(f"Deferred loss extra duplicates accumulated field {key!r}.")
        for key, value in extra_values.items():
            slots: list[float | None] = []
            if torch.is_tensor(value):
                scalar = value.detach().reshape(()).to(dtype=torch.float64)
                slots.append(None)
                pending.setdefault(scalar.device, []).append((key, 0, scalar))
            else:
                scalar_value = float(value)
                if not math.isfinite(scalar_value):
                    raise FloatingPointError(
                        f"Refusing to finalize non-finite loss field {key}: {scalar_value!r}."
                    )
                slots.append(scalar_value)
            ordered_values[key] = slots

        for entries in pending.values():
            # All tensors in a device bucket are scalar FP64 values.  One
            # stack/copy replaces O(fields * minibatches) host synchronizations.
            host_values = torch.stack([value for _, _, value in entries]).cpu().tolist()
            for (key, slot_idx, _), value in zip(entries, host_values):
                value = float(value)
                if not math.isfinite(value):
                    raise FloatingPointError(
                        f"Refusing to finalize non-finite loss field {key}: {value!r}."
                    )
                ordered_values[key][slot_idx] = value

        for key, values in ordered_values.items():
            if any(value is None for value in values):
                raise RuntimeError(f"Deferred loss field {key} was not copied to the host.")
            if key in extra_values:
                if len(values) != 1:
                    raise RuntimeError(f"Deferred loss extra {key} must contain exactly one scalar.")
                finalized[key] = float(values[0])
                continue
            total = 0.0
            for value in values:
                total += float(value)
            finalized[key] = total / num_updates
        return finalized

    def _accumulate_loss_dict(
        self,
        loss_dict: dict[str, Any],
        ppo_loss_dict: dict[str, torch.Tensor],
        *,
        defer_host_sync: bool = False,
    ):
        if defer_host_sync:
            add = lambda key, value: self._add_deferred_loss_scalar(loss_dict, key, value)
        else:
            add = lambda key, value: loss_dict.__setitem__(
                key,
                float(loss_dict.get(key, 0.0)) + self._loss_to_float(value),
            )

        add("Value", ppo_loss_dict.get("value_loss", 0.0))
        add("Surrogate", ppo_loss_dict.get("surrogate_loss", 0.0))
        add("Entropy", ppo_loss_dict.get("entropy_loss", 0.0))
        add("KL", ppo_loss_dict.get("kl_mean", 0.0))
        reserved = {"value_loss", "surrogate_loss", "entropy_loss", "kl_mean"}
        for key, loss in ppo_loss_dict.items():
            if key in reserved:
                continue
            add(key, loss)
        return loss_dict

    def _sanitize_actor_std(self):
        if not hasattr(self.actor, "std"):
            return
        with torch.no_grad():
            std = self._project_actor_std_constraints(self.actor.std.data)
            self.actor.std.data.copy_(std)

    def _validate_checkpoint_actor_std(
        self,
        actor_state: dict[str, Any],
        *,
        path: str,
        actor: nn.Module | None = None,
    ) -> None:
        """Reject checkpoint std values that runtime projection would alter."""
        actor = self.actor if actor is None else actor
        if not hasattr(actor, "std"):
            return
        checkpoint_std = actor_state.get("std")
        if not isinstance(checkpoint_std, torch.Tensor):
            raise ValueError(f"Checkpoint {path}.std must be a tensor.")
        if not bool(torch.isfinite(checkpoint_std).all().item()):
            raise ValueError(f"Checkpoint {path}.std contains NaN/Inf.")
        projected_std = self._project_actor_std_constraints(checkpoint_std, actor=actor)
        if torch.equal(checkpoint_std, projected_std):
            return

        min_noise_std = getattr(actor, "min_noise_std", None)
        min_mean_noise_std = getattr(actor, "min_mean_noise_std", None)
        max_noise_std = getattr(actor, "max_noise_std", None)
        raise ValueError(
            f"Checkpoint {path}.std violates the runtime policy-noise constraints and "
            "would be silently changed during restore; exact resume/policy init requires "
            "an already valid std. "
            f"checkpoint_range=[{float(checkpoint_std.min().item())}, "
            f"{float(checkpoint_std.max().item())}], "
            f"checkpoint_mean={float(checkpoint_std.mean().item())}, "
            f"projected_range=[{float(projected_std.min().item())}, "
            f"{float(projected_std.max().item())}], "
            f"projected_mean={float(projected_std.mean().item())}, "
            f"min_noise_std={min_noise_std}, min_mean_noise_std={min_mean_noise_std}, "
            f"max_noise_std={max_noise_std}."
        )

    def _synchronize_training_phase_error(
        self,
        local_error: Exception | None,
        *,
        operation: str,
    ) -> None:
        """Fail every rank after a rank-local numerical operation fails.

        The success path costs one scalar collective.  The more expensive
        all-rank error envelope is entered only after that scalar verdict says
        at least one worker failed, so every rank follows the same collective
        order and no peer can advance into the next minibatch.
        """

        distributed = (
            getattr(self, "is_multi_gpu", False)
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if not distributed:
            if local_error is not None:
                raise local_error
            return

        failed = torch.tensor(
            1 if local_error is not None else 0,
            device=self.device,
            dtype=torch.int32,
        )
        failed = self._all_reduce_small_tensor(
            failed,
            op=torch.distributed.ReduceOp.MAX,
        )
        if not bool(failed.item()):
            return
        self._synchronize_distributed_operation_error(
            local_error,
            operation=operation,
        )

    def _step_actor_optimizer(self) -> None:
        """Take one actor step and immediately project policy std constraints."""
        self.actor_optimizer.step()
        if hasattr(self.actor, "std") and not bool(torch.isfinite(self.actor.std).all().item()):
            raise FloatingPointError(
                "Actor optimizer produced NaN/Inf std; refusing silent projection or continued training."
            )
        self._sanitize_actor_std()

    def _supervised_dagger_only_enabled(self) -> bool:
        if hasattr(self, "_supervised_dagger_only"):
            return bool(self._supervised_dagger_only)
        return self._strict_environment_bool(
            "HOLOSOMA_DAGGER_SUPERVISED_ONLY",
            default=False,
        )

    def _supervised_actor_stream_backward_enabled(self) -> bool:
        if hasattr(self, "_supervised_actor_stream_backward"):
            return bool(self._supervised_actor_stream_backward)
        return self._strict_environment_bool(
            "HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD",
            default=False,
        )

    def _supervised_actor_only_step_enabled(self) -> bool:
        if hasattr(self, "_supervised_actor_only_step"):
            return bool(self._supervised_actor_only_step)
        # Fully initialized scientific runs always freeze this field during
        # setup.  Preserve the historical test/legacy-object fallback while
        # keeping the optimizer decision independent of ambient env changes.
        return self._supervised_dagger_only_enabled()

    def _supervised_actor_microbatch_size_value(self) -> int:
        if hasattr(self, "_supervised_actor_microbatch_size"):
            return int(self._supervised_actor_microbatch_size)
        return self._strict_environment_nonnegative_int(
            "HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH",
            default=0,
        )

    def _critic_optimizer_objective_enabled(self) -> bool:
        """Whether the critic has an objective that permits state mutation."""

        if self._supervised_dagger_only_enabled():
            return False
        config = getattr(self, "config", None)
        value_weight = float(getattr(config, "value_loss_coef", 1.0))
        symmetry_weight = float(getattr(config, "symmetry_critic_coef", 0.0))
        return bool(
            value_weight > 0.0
            or (getattr(self, "use_symmetry", False) and symmetry_weight > 0.0)
        )

    def _synchronize_cuda_before_gradient_reduction(self) -> bool:
        """Finish local CUDA work before packing and reducing gradients."""
        enabled = os.environ.get("HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE", "1").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if enabled:
            torch.cuda.synchronize(self.device)
        return enabled

    def _prepare_distributed_dagger_minibatch_controls(
        self,
        minibatch: Minibatch,
        *,
        controls_authoritative: bool,
    ) -> Minibatch:
        """Resolve any distributed BC controls before local loss construction.

        ``_compute_ppo_loss`` must be a collective-free region when it is
        called by ``_update_algo_step``.  Otherwise a rank-local actor/shape
        failure can abandon healthy peers in a BC-count collective.  The
        production contiguous generator already supplies both controls from
        one rollout-level reduction, so this is a no-op on its hot path.  The
        fallback below preserves randomized/direct-minibatch semantics while
        putting its fallible local work behind a synchronized preflight.
        """

        if not self.is_multi_gpu or self._effective_dagger_loss_weight() <= 0.0:
            return minibatch

        rollout_denominator = minibatch.get("_dagger_bc_denominator")
        cached_presence = minibatch.get("_dagger_bc_has_valid_samples")
        if controls_authoritative:
            # This mode is selected once from the rollout-level contiguous
            # presence plan, not from rank-local minibatch fields.  If one
            # rank's mapping is malformed, local loss construction raises and
            # every rank meets at the common outcome verdict; no rank can
            # peel off into the fallback count collective.
            return minibatch

        local_error: Exception | None = None
        weighted_count: torch.Tensor | None = None
        try:
            actions = minibatch["actions"]
            teacher_actions = minibatch["teacher_actions"][: actions.shape[0]]
            valid_mask = torch.ones(
                teacher_actions.shape[:-1],
                device=teacher_actions.device,
                dtype=torch.bool,
            )
            teacher_bc_mask = minibatch.get("teacher_bc_mask")
            if teacher_bc_mask is not None:
                valid_mask &= teacher_bc_mask[: actions.shape[0]].view_as(valid_mask).to(
                    dtype=torch.bool
                )
            if self.dagger_ignore_zero_teacher_actions:
                valid_mask &= ~torch.all(teacher_actions == 0.0, dim=-1)
            count_dtype = actions.dtype if actions.is_floating_point() else torch.float32
            local_valid_count = valid_mask.to(dtype=count_dtype).sum()
            weighted_count = local_valid_count * self._get_distributed_loss_weight()
            # This fallback is not used by the formal contiguous launcher.
            # Quiescing it here makes latent CUDA indexing/device failures
            # visible before the all-rank preflight verdict below.
            if weighted_count.is_cuda:
                torch.cuda.synchronize(weighted_count.device)
        except Exception as exc:
            local_error = exc

        self._synchronize_training_phase_error(
            local_error,
            operation="PPO distributed DAgger minibatch-control preflight",
        )
        assert weighted_count is not None
        global_weighted_count = self._all_reduce_small_tensor(
            weighted_count,
            op=torch.distributed.ReduceOp.SUM,
        )
        post_reduce_error: Exception | None = None
        prepared: dict[str, Any] | None = None
        try:
            has_valid_samples = bool((global_weighted_count > 0.0).item())
            prepared = dict(minibatch)
            if rollout_denominator is None:
                denominator = global_weighted_count / float(self.gpu_world_size)
                prepared["_dagger_bc_denominator"] = torch.where(
                    global_weighted_count > 0.0,
                    denominator,
                    torch.ones_like(denominator),
                )
            if cached_presence is None:
                prepared["_dagger_bc_has_valid_samples"] = has_valid_samples
        except Exception as exc:
            post_reduce_error = exc
        # The count SUM has completed everywhere, but mapping construction and
        # CUDA-to-host presence extraction can still fail rank-locally.  Keep
        # fallback peers in one protocol before any enters local loss compute.
        self._synchronize_training_phase_error(
            post_reduce_error,
            operation="PPO distributed DAgger minibatch-control finalization",
        )
        assert prepared is not None
        return prepared

    def _update_algo_step(
        self,
        minibatch: Minibatch,
        loss_dict: dict[str, Any],
        *,
        defer_loss_host_sync: bool = False,
        dagger_controls_authoritative: bool = False,
    ):
        debug_heartbeat = os.environ.get("HOLOSOMA_DEBUG_HEARTBEAT_VERBOSE", "").lower() not in (
            "",
            "0",
            "false",
            "no",
        )
        rank_label = f"{getattr(self, 'gpu_global_rank', 0)}/{getattr(self, 'gpu_world_size', 1)}"
        # Resolve the only optional DAgger count collective before entering
        # the rank-local forward/loss/backward region.
        minibatch = self._prepare_distributed_dagger_minibatch_controls(
            minibatch,
            controls_authoritative=dagger_controls_authoritative,
        )
        supervised_actor_only_step = self._supervised_actor_only_step_enabled()
        critic_optimizer_step_enabled = self._critic_optimizer_objective_enabled()
        stream_supervised_actor_backward = bool(
            supervised_actor_only_step
            and self._supervised_actor_stream_backward_enabled()
        )
        if stream_supervised_actor_backward:
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad(set_to_none=True)
        self._stream_supervised_actor_backward = stream_supervised_actor_backward
        self._debug_training_phase(
            "update_enter",
            stream_supervised_actor_backward=stream_supervised_actor_backward,
            supervised_actor_only_step=supervised_actor_only_step,
        )
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} update compute_loss begin (grad_enabled={} inference_mode={})",
                self.current_learning_iteration,
                rank_label,
                torch.is_grad_enabled(),
                torch.is_inference_mode_enabled(),
            )
        ppo_loss_dict: dict[str, Any] | None = None
        update_error: Exception | None = None
        self._compute_loss_requires_prepared_dagger_controls = True
        try:
            with torch.inference_mode(False), torch.enable_grad():
                self._debug_training_phase("compute_loss_begin")
                ppo_loss_dict = self._compute_ppo_loss(minibatch)
        except Exception as exc:
            update_error = exc
        finally:
            self._stream_supervised_actor_backward = False
            self._compute_loss_requires_prepared_dagger_controls = False
        self._debug_training_phase(
            "compute_loss_finished",
            local_compute_succeeded=update_error is None,
        )
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} update compute_loss finished (grad_enabled={} inference_mode={})",
                self.current_learning_iteration,
                rank_label,
                torch.is_grad_enabled(),
                torch.is_inference_mode_enabled(),
            )
        backward_already_done = False
        actor_optimizer_step_enabled = True
        expected_kl_reduction = not supervised_actor_only_step
        reduce_kl_before_optimizer = expected_kl_reduction
        distributed_kl_payload: torch.Tensor | None = None
        invalid_loss_fields: list[str] = []
        try:
            if update_error is not None:
                raise update_error
            assert ppo_loss_dict is not None
            backward_already_done = bool(ppo_loss_dict.pop("_backward_already_done", False))
            actor_optimizer_step_enabled = bool(
                ppo_loss_dict.pop("_actor_optimizer_step_enabled", True)
            )
            returned_kl_reduction = ppo_loss_dict.pop(
                "_reduce_kl_before_optimizer",
                expected_kl_reduction,
            )
            if type(returned_kl_reduction) is not bool:
                raise TypeError("_reduce_kl_before_optimizer must be a Python bool.")
            if returned_kl_reduction != expected_kl_reduction:
                raise RuntimeError(
                    "PPO loss returned a KL-reduction mode inconsistent with the frozen "
                    "supervised actor-only optimizer contract."
                )
            if reduce_kl_before_optimizer and self.is_multi_gpu:
                # Construct and validate every rank-local part before the
                # synchronized outcome verdict.  After that boundary the KL
                # path begins directly with its collective, so one rank cannot
                # fail in stack/device/allocation work while peers wait in SUM.
                distributed_kl_payload = self._build_distributed_kl_payload(
                    ppo_loss_dict["kl_mean"],
                    distributed_loss_weight=self._get_distributed_loss_weight(),
                )
            actor_loss = ppo_loss_dict["actor_loss"]
            critic_loss = ppo_loss_dict["critic_loss"]
            loss_fields_for_validation = ppo_loss_dict
            if distributed_kl_payload is not None:
                loss_fields_for_validation = {
                    **ppo_loss_dict,
                    "_distributed_kl_payload": distributed_kl_payload,
                }
            self._debug_training_phase(
                "loss_finite_check_begin",
                field_count=len(loss_fields_for_validation),
            )
            invalid_loss_fields = self._invalid_loss_fields_batched(
                loss_fields_for_validation
            )
            self._debug_training_phase(
                "loss_finite_check_finished",
                local_loss_finite=not invalid_loss_fields,
            )
            if invalid_loss_fields:
                update_error = FloatingPointError(
                    "Non-finite PPO loss verdict at iteration "
                    f"{self.current_learning_iteration}; this rank has non-finite fields: "
                    f"{', '.join(invalid_loss_fields)}. Refusing to skip the update or "
                    "record the value as zero."
                )
        except Exception as exc:
            update_error = exc

        if update_error is None and not backward_already_done:
            self._debug_training_phase("zero_grad_begin")
            self.actor_optimizer.zero_grad()
            if not critic_optimizer_step_enabled:
                self.critic_optimizer.zero_grad(set_to_none=True)
            else:
                self.critic_optimizer.zero_grad()
            self._debug_training_phase("zero_grad_finished")

        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} update loss/backward verdict begin",
                self.current_learning_iteration,
                rank_label,
            )
        self._debug_training_phase("backward_begin", backward_already_done=backward_already_done)
        if update_error is None:
            try:
                ppo_loss = (
                    actor_loss
                    if not critic_optimizer_step_enabled
                    else actor_loss + critic_loss
                )
                if backward_already_done:
                    if debug_heartbeat:
                        logger.info(
                            "Heartbeat: iter {} rank {} update backward skipped already_streamed",
                            self.current_learning_iteration,
                            rank_label,
                        )
                else:
                    ppo_loss.backward()
            except Exception as exc:
                update_error = exc

        # Forward/loss construction, validation, and backward are all local,
        # contain no distributed collective, and precede gradient
        # reduction/optimizer mutation.  One all-rank outcome therefore
        # protects every phase and replaces their former separate verdicts.
        try:
            self._synchronize_training_phase_error(
                update_error,
                operation="PPO loss validation/backward",
            )
        except Exception:
            # A healthy rank may have completed backward while another rank
            # reported invalid loss/forward/backward.  Clear every rank before
            # propagating the synchronized failure so no partial or stale
            # gradient can survive a caller-level recovery.
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            if invalid_loss_fields:
                self._debug_training_phase("update_exit_nonfinite_loss")
            raise
        self._debug_training_phase("backward_finished")
        if debug_heartbeat:
            if backward_already_done:
                logger.info(
                    "Heartbeat: iter {} rank {} update streamed backward verdict finished",
                    self.current_learning_iteration,
                    rank_label,
                )
            logger.info(
                "Heartbeat: iter {} rank {} update loss/backward verdict finished",
                self.current_learning_iteration,
                rank_label,
            )

        if self.is_multi_gpu:
            # Isaac Sim/PhysX and the policy backward pass can leave work on
            # multiple CUDA streams.  Quiesce simulator/backward work before
            # constructing the collective payload.  The flat-NCCL path also
            # establishes a second boundary after packing and waits for the
            # collective itself; both are required because zeros/copy/cat
            # below enqueue additional CUDA work after this synchronization.
            # These boundaries change scheduling only; gradients and optimizer
            # mathematics are unchanged.
            sync_before_grad_reduce = os.environ.get("HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE", "1").lower() in (
                "1", "true", "yes", "on"
            )
            if sync_before_grad_reduce:
                if debug_heartbeat:
                    logger.info(
                        "Heartbeat: iter {} rank {} update pre-grad all_reduce cuda sync begin",
                        self.current_learning_iteration,
                        rank_label,
                    )
                self._synchronize_cuda_before_gradient_reduction()
                if debug_heartbeat:
                    logger.info(
                        "Heartbeat: iter {} rank {} update pre-grad all_reduce cuda sync finished",
                        self.current_learning_iteration,
                        rank_label,
                    )
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} update grad all_reduce begin",
                    self.current_learning_iteration,
                    rank_label,
                )
            self._debug_training_phase("grad_reduce_begin")
            self._reduce_parameters(include_critic=critic_optimizer_step_enabled)
            self._debug_training_phase("grad_reduce_finished")
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} update grad all_reduce finished",
                    self.current_learning_iteration,
                    rank_label,
                )
            if os.environ.get("HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE", "").lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                if debug_heartbeat:
                    logger.info(
                        "Heartbeat: iter {} rank {} update grad all_reduce cuda sync begin",
                        self.current_learning_iteration,
                        rank_label,
                    )
                torch.cuda.synchronize(self.device)
                if debug_heartbeat:
                    logger.info(
                        "Heartbeat: iter {} rank {} update grad all_reduce cuda sync finished",
                        self.current_learning_iteration,
                        rank_label,
                    )

        # ``clip_grad_norm_(error_if_nonfinite=True)`` computes the total norm
        # over every present gradient and raises before applying a clip factor
        # when that norm is NaN/Inf.  Gradients have already passed through the
        # global reduction above, so a second per-parameter finite scan plus a
        # separate all-rank verdict is redundant.  Keep one exact scan here and
        # propagate its outcome to every rank before any optimizer mutates
        # model state.
        gradient_clip_error: Exception | None = None
        try:
            if reduce_kl_before_optimizer:
                assert ppo_loss_dict is not None
                ppo_loss_dict["kl_mean"] = self._reduce_kl_after_local_loss(
                    ppo_loss_dict["kl_mean"],
                    distributed_payload=distributed_kl_payload,
                )
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} update actor grad clip begin",
                    self.current_learning_iteration,
                    rank_label,
                )
            self._debug_training_phase("actor_grad_clip_begin")
            nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            self._debug_training_phase("actor_grad_clip_finished")
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} update actor grad clip finished",
                    self.current_learning_iteration,
                    rank_label,
                )
                logger.info(
                    "Heartbeat: iter {} rank {} update critic grad clip begin",
                    self.current_learning_iteration,
                    rank_label,
                )
            if not critic_optimizer_step_enabled:
                if debug_heartbeat:
                    logger.info(
                        "Heartbeat: iter {} rank {} update critic grad clip skipped actor-only",
                        self.current_learning_iteration,
                        rank_label,
                    )
            else:
                self._debug_training_phase("critic_grad_clip_begin")
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.max_grad_norm,
                    error_if_nonfinite=True,
                )
                self._debug_training_phase("critic_grad_clip_finished")
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} update critic grad clip finished",
                    self.current_learning_iteration,
                    rank_label,
                )
        except Exception as exc:
            gradient_clip_error = exc
        try:
            self._synchronize_training_phase_error(
                gradient_clip_error,
                operation="PPO KL adaptation/gradient clipping",
            )
        except Exception:
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            raise

        optimizer_step_error: Exception | None = None
        try:
            if debug_heartbeat:
                if actor_optimizer_step_enabled:
                    logger.info(
                        "Heartbeat: iter {} rank {} update actor optimizer step begin",
                        self.current_learning_iteration,
                        rank_label,
                    )
                else:
                    logger.info(
                        "Heartbeat: iter {} rank {} update actor optimizer step skipped no-signal",
                        self.current_learning_iteration,
                        rank_label,
                    )
            if actor_optimizer_step_enabled:
                self._debug_training_phase("actor_optimizer_step_begin")
                self._step_actor_optimizer()
                self._debug_training_phase("actor_optimizer_step_finished")
            else:
                self.actor_optimizer.zero_grad(set_to_none=True)
                self._debug_training_phase("actor_optimizer_step_skipped_no_signal")
            if debug_heartbeat:
                if actor_optimizer_step_enabled:
                    logger.info(
                        "Heartbeat: iter {} rank {} update actor optimizer step finished",
                        self.current_learning_iteration,
                        rank_label,
                    )
                logger.info(
                    "Heartbeat: iter {} rank {} update critic optimizer step begin",
                    self.current_learning_iteration,
                    rank_label,
                )
            if not critic_optimizer_step_enabled:
                if debug_heartbeat:
                    logger.info(
                        "Heartbeat: iter {} rank {} update critic optimizer step skipped actor-only",
                        self.current_learning_iteration,
                        rank_label,
                    )
            else:
                self._debug_training_phase("critic_optimizer_step_begin")
                self.critic_optimizer.step()
                self._debug_training_phase("critic_optimizer_step_finished")
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} update critic optimizer step finished",
                    self.current_learning_iteration,
                    rank_label,
                )
            self._debug_training_phase("noise_std_cap_begin")
            self._apply_ppo_start_noise_std_cap(self.current_learning_iteration)
            self._debug_training_phase("noise_std_cap_finished")
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} update optimizer step finished",
                    self.current_learning_iteration,
                    rank_label,
                )
            if os.environ.get("HOLOSOMA_SYNC_AFTER_OPTIMIZER_STEP", "").lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                if debug_heartbeat:
                    logger.info(
                        "Heartbeat: iter {} rank {} update optimizer cuda sync begin",
                        self.current_learning_iteration,
                        rank_label,
                    )
                torch.cuda.synchronize(self.device)
                if debug_heartbeat:
                    logger.info(
                        "Heartbeat: iter {} rank {} update optimizer cuda sync finished",
                        self.current_learning_iteration,
                        rank_label,
                    )
        except Exception as exc:
            optimizer_step_error = exc
        self._synchronize_training_phase_error(
            optimizer_step_error,
            operation="PPO optimizer update/std projection",
        )

        self._debug_training_phase("loss_dict_accumulate_begin")
        assert ppo_loss_dict is not None
        loss_dict = self._accumulate_loss_dict(
            loss_dict,
            ppo_loss_dict,
            defer_host_sync=defer_loss_host_sync,
        )
        self._debug_training_phase("update_exit")
        return loss_dict

    def _compute_dagger_replay_bc_loss(
        self,
        minibatch: Mapping[str, Any],
        *,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, bool, bool]:
        """Compute off-policy action rehearsal without constructing a PPO ratio.

        The replay term is defined only for the fail-closed pure-BC Stage1
        configuration validated during setup.  Distributed denominators are
        prepared before local forward/loss construction, so this method has no
        collectives and can share the ordinary synchronized error envelope.
        """

        replay = minibatch.get("_dagger_replay")
        if replay is None:
            return reference.new_zeros(()), False, False
        if not self.dagger_replay_enabled:
            raise RuntimeError("A replay minibatch was supplied while DAgger replay is disabled.")
        if not isinstance(replay, Mapping) or set(replay) != {
            "batch",
            "denominator",
            "has_valid_samples",
        }:
            raise ValueError("DAgger replay minibatch envelope has an invalid schema.")
        batch = replay["batch"]
        if not isinstance(batch, Mapping) or set(batch) != {
            "actor_obs_raw",
            "actor_perception",
            "teacher_actions",
            "mask",
        }:
            raise ValueError("DAgger replay batch has an invalid tensor schema.")
        denominator = replay["denominator"]
        if not isinstance(denominator, torch.Tensor) or denominator.numel() != 1:
            raise ValueError("DAgger replay denominator must be one scalar tensor.")
        denominator = denominator.to(device=self.device, dtype=reference.dtype)
        if not bool(torch.isfinite(denominator).item()) or not bool(
            (denominator > 0.0).item()
        ):
            raise ValueError("DAgger replay denominator must be finite and positive.")
        has_valid_samples = replay["has_valid_samples"]
        if type(has_valid_samples) is not bool:
            raise ValueError("DAgger replay presence must be a Python bool.")

        actor_obs_raw = batch["actor_obs_raw"]
        actor_perception = batch["actor_perception"]
        teacher_actions = batch["teacher_actions"]
        mask = batch["mask"]
        buffer = getattr(self, "_dagger_replay_buffer", None)
        if not isinstance(buffer, _DaggerReplayBuffer):
            raise RuntimeError("Enabled DAgger replay has no rank-local buffer.")
        rows = int(actor_obs_raw.shape[0]) if isinstance(actor_obs_raw, torch.Tensor) else -1
        expected_shapes = {
            "actor_obs_raw": (rows, buffer.actor_obs_dim),
            "actor_perception": (rows, buffer.actor_perception_dim),
            "teacher_actions": (rows, buffer.action_dim),
            "mask": (rows, 1),
        }
        for key, expected_shape in expected_shapes.items():
            value = batch[key]
            expected_dtype = torch.bool if key == "mask" else torch.float32
            if (
                not isinstance(value, torch.Tensor)
                or value.layout != torch.strided
                or tuple(value.shape) != expected_shape
                or value.dtype != expected_dtype
            ):
                raise ValueError(
                    f"DAgger replay {key} must have shape={expected_shape}, "
                    f"dtype={expected_dtype}."
                )
            if key != "mask" and not bool(torch.isfinite(value).all().item()):
                raise FloatingPointError(f"DAgger replay {key} contains NaN/Inf.")
        if rows == 0:
            return reference.new_zeros(()), has_valid_samples, True

        actor_obs_raw = actor_obs_raw.to(device=self.device)
        actor_obs = self._normalize_actor_obs(actor_obs_raw, update=False)
        teacher_actions = teacher_actions.to(device=self.device)
        mask = mask.to(device=self.device, dtype=torch.bool).view(-1)
        if self.clip_teacher_actions:
            teacher_actions = self._clip_teacher_actions_preserving_non_finite(
                teacher_actions,
                self.clip_actions_threshold,
            )
        actor_policy_state = {"actor_obs": actor_obs}
        if self.actor_perception_key:
            actor_policy_state[self.actor_perception_key] = actor_perception.to(
                device=self.device
            )
        if self._actor_uses_flow_matching():
            per_sample = self.actor.flow_matching_loss(
                actor_policy_state,
                teacher_actions,
                loss_fn=self.distill_loss_fn,
            )
        else:
            student_actions = self.actor.act_inference(actor_policy_state)
            per_element = self.distill_loss_fn(
                student_actions,
                teacher_actions,
                reduction="none",
            )
            per_sample = (
                per_element.mean(dim=-1)
                if per_element.ndim > 1
                else per_element
            )
        if self.dagger_ignore_zero_teacher_actions:
            mask &= ~torch.all(teacher_actions == 0.0, dim=-1)
        numerator = (per_sample * mask.to(dtype=per_sample.dtype)).sum()
        return numerator / denominator, has_valid_samples, True

    def _compute_ppo_loss(self, minibatch: Minibatch):
        if self.use_time_gru:
            return self._compute_ppo_loss_sequence(minibatch)
        debug_heartbeat = os.environ.get("HOLOSOMA_DEBUG_HEARTBEAT_VERBOSE", "").lower() not in (
            "",
            "0",
            "false",
            "no",
        )
        rank_label = f"{self.gpu_global_rank}/{self.gpu_world_size}" if self.is_multi_gpu else str(self.gpu_global_rank)
        def _clone_if_inference_tensor(value):
            if isinstance(value, torch.Tensor) and value.is_inference():
                return value.clone()
            return value

        critic_objective_enabled = self._critic_optimizer_objective_enabled()
        actor_obs = _clone_if_inference_tensor(minibatch["actor_obs"])
        critic_obs = _clone_if_inference_tensor(minibatch["critic_obs"])
        actions_batch = _clone_if_inference_tensor(minibatch["actions"])
        target_values_batch = _clone_if_inference_tensor(minibatch["values"])
        advantages_batch = _clone_if_inference_tensor(minibatch["advantages"])
        returns_batch = _clone_if_inference_tensor(minibatch["returns"])
        old_actions_log_prob_batch = _clone_if_inference_tensor(minibatch["actions_log_prob"])
        old_mu_batch = _clone_if_inference_tensor(minibatch["action_mean"])
        old_sigma_batch = _clone_if_inference_tensor(minibatch["action_sigma"])
        actor_perception_obs = (
            _clone_if_inference_tensor(minibatch.get(self.actor_perception_key)) if self.actor_perception_key else None
        )
        critic_perception_obs = (
            _clone_if_inference_tensor(minibatch.get(self.critic_perception_key))
            if critic_objective_enabled and self.critic_perception_key
            else None
        )
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} loss begin actor_obs={} actions={} actor_flow={}",
                self.current_learning_iteration,
                rank_label,
                tuple(actor_obs.shape),
                tuple(actions_batch.shape),
                self._actor_uses_flow_matching(),
            )

        # actor_obs/critic_obs are the exact normalized tensors used when the
        # transition was collected.  PPO likelihoods must be recomputed from
        # those tensors without updating normalization statistics again.
        original_batch_size = actions_batch.shape[0]

        if actor_perception_obs is not None and actor_perception_obs.is_inference():
            actor_perception_obs = actor_perception_obs.clone()
        if critic_perception_obs is not None and critic_perception_obs.is_inference():
            critic_perception_obs = critic_perception_obs.clone()

        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} loss normalized obs actor={} critic={}",
                self.current_learning_iteration,
                rank_label,
                tuple(actor_obs.shape),
                tuple(critic_obs.shape),
            )

        supervised_dagger_only = self._supervised_dagger_only_enabled()
        if supervised_dagger_only and self.distill_mode == "dagger" and self.dagger_enabled:
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} loss supervised_dagger begin",
                    self.current_learning_iteration,
                    rank_label,
                )
            teacher_actions_batch = minibatch.get("teacher_actions")
            if teacher_actions_batch is None:
                raise ValueError("Dagger supervised-only mode requires teacher_actions in rollout storage.")
            teacher_actions_batch = _clone_if_inference_tensor(teacher_actions_batch[:original_batch_size])
            if self.clip_teacher_actions:
                teacher_actions_batch = self._clip_teacher_actions_preserving_non_finite(
                    teacher_actions_batch,
                    self.clip_actions_threshold,
                )

            actor_policy_state = {"actor_obs": actor_obs[:original_batch_size]}
            if actor_perception_obs is not None:
                actor_policy_state[self.actor_perception_key] = actor_perception_obs[:original_batch_size]
            teacher_bc_mask_batch = minibatch.get("teacher_bc_mask")
            if teacher_bc_mask_batch is not None:
                teacher_bc_mask_batch = _clone_if_inference_tensor(
                    teacher_bc_mask_batch[:original_batch_size]
                ).view(-1)
            if self.use_ppo_dagger_schedule:
                lambda_ppo = float(self.ppo_coeff)
                dagger_weight = self.dagger_loss_coef * (1.0 - lambda_ppo)
            elif self.bc_loss_coef > 0.0:
                dagger_weight = self.bc_loss_coef
            else:
                dagger_weight = self.dagger_loss_coef
            dagger_weight = self._operational_float32_loss_weight(dagger_weight)

            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} loss supervised_dagger actor.inference begin",
                    self.current_learning_iteration,
                    rank_label,
                )
            bc_loss = None
            bc_has_valid_samples = False
            backward_already_done = False
            replay_requested = minibatch.get("_dagger_replay") is not None
            current_bc_fraction = (
                1.0 - self.dagger_replay_fraction if replay_requested else 1.0
            )
            if self._actor_uses_flow_matching():
                distill_per_sample = self.actor.flow_matching_loss(
                    actor_policy_state,
                    teacher_actions_batch,
                    loss_fn=self.distill_loss_fn,
                )
            else:
                actor_microbatch_size = self._supervised_actor_microbatch_size_value()
                if actor_microbatch_size > 0 and original_batch_size > actor_microbatch_size:
                    if debug_heartbeat:
                        logger.info(
                            "Heartbeat: iter {} rank {} loss supervised_dagger actor.inference microbatch size={} batch={}",
                            self.current_learning_iteration,
                            rank_label,
                            actor_microbatch_size,
                            original_batch_size,
                        )
                    stream_microbatch_backward = bool(getattr(self, "_stream_supervised_actor_backward", False))
                    distill_weighted_sum = torch.zeros((), device=self.device)
                    valid_count = torch.zeros((), device=self.device)
                    log_all_microbatches = os.environ.get("HOLOSOMA_DEBUG_MICROBATCH_ALL", "").lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    )
                    sync_after_microbatch_forward = os.environ.get(
                        "HOLOSOMA_SYNC_AFTER_MICROBATCH_FORWARD", ""
                    ).lower() in ("1", "true", "yes", "on")
                    stream_backward_error: Exception | None = None
                    if stream_microbatch_backward:
                        with torch.no_grad():
                            for micro_start in range(0, original_batch_size, actor_microbatch_size):
                                micro_end = min(micro_start + actor_microbatch_size, original_batch_size)
                                teacher_actions_micro = teacher_actions_batch[micro_start:micro_end]
                                valid_mask_micro = torch.ones(
                                    (micro_end - micro_start,), device=self.device, dtype=torch.bool
                                )
                                if teacher_bc_mask_batch is not None:
                                    valid_mask_micro &= teacher_bc_mask_batch[micro_start:micro_end].to(
                                        dtype=torch.bool
                                    )
                                if self.dagger_ignore_zero_teacher_actions:
                                    valid_mask_micro &= ~torch.all(teacher_actions_micro == 0.0, dim=-1)
                                valid_count = valid_count + valid_mask_micro.to(
                                    dtype=teacher_actions_batch.dtype
                                ).sum()
                            bc_denominator, bc_has_valid_samples = (
                                self._bc_denominator_and_presence_for_minibatch(
                                    minibatch,
                                    valid_count,
                                )
                            )
                    for micro_start in range(0, original_batch_size, actor_microbatch_size):
                        micro_end = min(micro_start + actor_microbatch_size, original_batch_size)
                        micro_policy_state = {
                            key: value[micro_start:micro_end]
                            for key, value in actor_policy_state.items()
                        }
                        if debug_heartbeat and (micro_start == 0 or log_all_microbatches):
                            logger.info(
                                "Heartbeat: iter {} rank {} loss supervised_dagger actor.inference microbatch begin {}:{} grad_enabled={} inference_mode={}",
                                self.current_learning_iteration,
                                rank_label,
                                micro_start,
                                micro_end,
                                torch.is_grad_enabled(),
                                torch.is_inference_mode_enabled(),
                            )
                        try:
                            with torch.inference_mode(False), torch.enable_grad():
                                student_actions_micro = self.actor.act_inference(micro_policy_state)
                        except Exception as exc:
                            if not stream_microbatch_backward:
                                raise
                            stream_backward_error = exc
                            break
                        if sync_after_microbatch_forward:
                            if debug_heartbeat and (micro_start == 0 or log_all_microbatches):
                                logger.info(
                                    "Heartbeat: iter {} rank {} loss supervised_dagger actor.inference microbatch cuda sync begin {}:{}",
                                    self.current_learning_iteration,
                                    rank_label,
                                    micro_start,
                                    micro_end,
                                )
                            try:
                                torch.cuda.synchronize(self.device)
                            except Exception as exc:
                                if not stream_microbatch_backward:
                                    raise
                                stream_backward_error = exc
                                break
                            if debug_heartbeat and (micro_start == 0 or log_all_microbatches):
                                logger.info(
                                    "Heartbeat: iter {} rank {} loss supervised_dagger actor.inference microbatch cuda sync finished {}:{}",
                                    self.current_learning_iteration,
                                    rank_label,
                                    micro_start,
                                    micro_end,
                                )
                        if debug_heartbeat and (micro_start == 0 or log_all_microbatches):
                            logger.info(
                                "Heartbeat: iter {} rank {} loss supervised_dagger actor.inference microbatch finished {}:{}",
                                self.current_learning_iteration,
                                rank_label,
                                micro_start,
                                micro_end,
                            )
                        teacher_actions_micro = teacher_actions_batch[micro_start:micro_end]
                        try:
                            distill_per_elem_micro = self.distill_loss_fn(
                                student_actions_micro,
                                teacher_actions_micro,
                                reduction="none",
                            )
                        except Exception as exc:
                            if not stream_microbatch_backward:
                                raise
                            stream_backward_error = exc
                            break
                        if distill_per_elem_micro.ndim > 1:
                            distill_per_sample_micro = distill_per_elem_micro.mean(dim=-1)
                        else:
                            distill_per_sample_micro = distill_per_elem_micro
                        valid_mask_micro = torch.ones_like(distill_per_sample_micro, dtype=torch.bool)
                        if teacher_bc_mask_batch is not None:
                            valid_mask_micro &= teacher_bc_mask_batch[micro_start:micro_end].to(dtype=torch.bool)
                        if self.dagger_ignore_zero_teacher_actions:
                            valid_mask_micro &= ~torch.all(teacher_actions_micro == 0.0, dim=-1)
                        valid_weight_micro = valid_mask_micro.to(dtype=distill_per_sample_micro.dtype)
                        distill_weighted_sum_micro = (
                            distill_per_sample_micro * valid_weight_micro
                        ).sum()
                        if stream_microbatch_backward:
                            if debug_heartbeat and (micro_start == 0 or log_all_microbatches):
                                logger.info(
                                    "Heartbeat: iter {} rank {} loss supervised_dagger actor.inference microbatch backward begin {}:{}",
                                    self.current_learning_iteration,
                                    rank_label,
                                    micro_start,
                                    micro_end,
                                )
                            try:
                                (
                                    dagger_weight
                                    * current_bc_fraction
                                    * distill_weighted_sum_micro
                                    / bc_denominator
                                ).backward()
                            except Exception as exc:
                                stream_backward_error = exc
                                break
                            if debug_heartbeat and (micro_start == 0 or log_all_microbatches):
                                logger.info(
                                    "Heartbeat: iter {} rank {} loss supervised_dagger actor.inference microbatch backward finished {}:{}",
                                    self.current_learning_iteration,
                                    rank_label,
                                    micro_start,
                                    micro_end,
                                )
                            distill_weighted_sum = distill_weighted_sum + distill_weighted_sum_micro.detach()
                        else:
                            distill_weighted_sum = distill_weighted_sum + distill_weighted_sum_micro
                            valid_count = valid_count + valid_weight_micro.sum()
                    if stream_microbatch_backward:
                        # The enclosing update synchronizes the complete
                        # forward/loss/backward outcome.  Raising locally here
                        # keeps this entire compute region collective-free;
                        # healthy ranks may finish their local microbatches and
                        # will then meet the failed rank at that one verdict.
                        if stream_backward_error is not None:
                            raise stream_backward_error
                        bc_loss = distill_weighted_sum / bc_denominator
                    else:
                        bc_denominator, bc_has_valid_samples = (
                            self._bc_denominator_and_presence_for_minibatch(
                                minibatch,
                                valid_count,
                            )
                        )
                        bc_loss = distill_weighted_sum / bc_denominator
                    backward_already_done = stream_microbatch_backward
                else:
                    with torch.inference_mode(False), torch.enable_grad():
                        student_actions = self.actor.act_inference(actor_policy_state)
                    distill_per_elem = self.distill_loss_fn(student_actions, teacher_actions_batch, reduction="none")
                    if distill_per_elem.ndim > 1:
                        distill_per_sample = distill_per_elem.mean(dim=-1)
                    else:
                        distill_per_sample = distill_per_elem
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} loss supervised_dagger actor.inference finished",
                    self.current_learning_iteration,
                    rank_label,
                )

            if bc_loss is None:
                valid_mask = torch.ones_like(distill_per_sample, dtype=torch.bool)
                if teacher_bc_mask_batch is not None:
                    valid_mask &= teacher_bc_mask_batch.to(dtype=torch.bool)
                if self.dagger_ignore_zero_teacher_actions:
                    valid_mask &= ~torch.all(teacher_actions_batch == 0.0, dim=-1)

                valid_weight = valid_mask.to(dtype=distill_per_sample.dtype)
                bc_denominator, bc_has_valid_samples = (
                    self._bc_denominator_and_presence_for_minibatch(
                        minibatch,
                        valid_weight.sum(),
                    )
                )
                bc_loss = (distill_per_sample * valid_weight).sum() / bc_denominator

            current_bc_loss = bc_loss
            (
                replay_bc_loss,
                replay_bc_has_valid_samples,
                replay_bc_active,
            ) = self._compute_dagger_replay_bc_loss(
                minibatch,
                reference=current_bc_loss,
            )
            if replay_bc_active:
                bc_loss = (
                    (1.0 - self.dagger_replay_fraction) * current_bc_loss
                    + self.dagger_replay_fraction * replay_bc_loss
                )
                if backward_already_done:
                    (
                        dagger_weight
                        * self.dagger_replay_fraction
                        * replay_bc_loss
                    ).backward()
            else:
                bc_loss = current_bc_loss

            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} loss supervised_dagger weight begin",
                    self.current_learning_iteration,
                    rank_label,
                )

            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} loss supervised_dagger actor_loss begin weight={}",
                    self.current_learning_iteration,
                    rank_label,
                    dagger_weight,
                )
            actor_loss = dagger_weight * bc_loss
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} loss supervised_dagger actor_loss finished",
                    self.current_learning_iteration,
                    rank_label,
                )
            zero = 0.0
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} loss supervised_dagger finished",
                    self.current_learning_iteration,
                    rank_label,
                )
            return {
                "actor_loss": actor_loss,
                "critic_loss": zero,
                "symmetry_actor_loss": zero,
                "symmetry_critic_loss": zero,
                "value_loss": zero,
                "surrogate_loss": zero,
                "entropy_loss": zero,
                "distill_loss": bc_loss,
                "bc_loss": bc_loss,
                "current_bc_loss": current_bc_loss,
                "replay_bc_loss": replay_bc_loss,
                "replay_bc_active": float(replay_bc_active),
                "ppo_coeff": float(self.ppo_coeff),
                "dagger_weight": dagger_weight,
                "kl_mean": zero,
                "_backward_already_done": backward_already_done,
                "_reduce_kl_before_optimizer": False,
                "_actor_optimizer_step_enabled": bool(
                    (bc_has_valid_samples or replay_bc_has_valid_samples)
                    and float(dagger_weight) > 0.0
                ),
                "actor_optimizer_step_skipped_no_signal": float(
                    (not bc_has_valid_samples and not replay_bc_has_valid_samples)
                    or float(dagger_weight) <= 0.0
                ),
            }

        actor_policy_state = {"actor_obs": actor_obs}
        if actor_perception_obs is not None:
            actor_policy_state[self.actor_perception_key] = actor_perception_obs
        if debug_heartbeat:
            logger.info("Heartbeat: iter {} rank {} loss actor.act begin", self.current_learning_iteration, rank_label)
        self.actor.update_distribution_from_policy_state(actor_policy_state)
        if debug_heartbeat:
            logger.info("Heartbeat: iter {} rank {} loss actor.act finished", self.current_learning_iteration, rank_label)

        if critic_objective_enabled:
            critic_policy_state = {"critic_obs": critic_obs}
            if critic_perception_obs is not None:
                critic_policy_state[self.critic_perception_key] = critic_perception_obs
            if debug_heartbeat:
                logger.info("Heartbeat: iter {} rank {} loss critic.evaluate begin", self.current_learning_iteration, rank_label)
            value_batch = self.critic.evaluate(critic_policy_state)
        else:
            value_batch = target_values_batch.new_zeros(target_values_batch.shape)
        # Pure BC has no likelihood-ratio objective. Do not construct its PPO
        # ratio: after BC moves the policy far from the rollout snapshot,
        # exp(new_logp-old_logp) can overflow, and exact-zero times NaN is still
        # NaN (which used to suppress both actor and critic optimizer steps).
        pure_bc_objective = bool(
            self.distill_mode == "dagger"
            and self.dagger_enabled
            and self._use_deterministic_student_actions()
        )
        if debug_heartbeat and critic_objective_enabled:
            logger.info("Heartbeat: iter {} rank {} loss critic.evaluate finished", self.current_learning_iteration, rank_label)
        if debug_heartbeat:
            logger.info("Heartbeat: iter {} rank {} loss action log_prob begin", self.current_learning_iteration, rank_label)
        actions_log_prob_batch = (
            None if pure_bc_objective else self.actor.get_actions_log_prob(actions_batch)
        )
        mu_batch = self.actor.action_mean[:original_batch_size]
        sigma_batch = self.actor.action_std[:original_batch_size]
        entropy_batch = self.actor.entropy[:original_batch_size]
        if debug_heartbeat:
            logger.info("Heartbeat: iter {} rank {} loss action log_prob finished", self.current_learning_iteration, rank_label)

        # KL is both a training-control signal and a scientific diagnostic.
        # Always measure the actual policy drift; the hybrid schedule may
        # suppress adaptive LR changes without suppressing observability.
        if debug_heartbeat:
            logger.info("Heartbeat: iter {} rank {} loss KL reduce begin", self.current_learning_iteration, rank_label)
        kl_mean = self._compute_kl_div(
            old_mu_batch,
            old_sigma_batch,
            mu_batch,
            sigma_batch,
            reduce_distributed=False,
        )
        if debug_heartbeat:
            logger.info("Heartbeat: iter {} rank {} loss KL reduce finished", self.current_learning_iteration, rank_label)

        # An exact zero is the scientifically correct PPO contribution during
        # pure BC and keeps irrelevant stale rollout likelihoods out of the
        # numerical graph.
        if pure_bc_objective:
            surrogate_loss = mu_batch.new_zeros(())
        else:
            assert actions_log_prob_batch is not None
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if critic_objective_enabled:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.config.clip_param, self.config.clip_param
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = actor_obs.new_zeros(())

        symmetry_actor_enabled = bool(
            self.use_symmetry and self.config.symmetry_actor_coef > 0.0
        )
        symmetry_critic_enabled = bool(
            critic_objective_enabled
            and self.use_symmetry
            and self.config.symmetry_critic_coef > 0.0
        )
        if symmetry_actor_enabled:
            raw_actor_obs = _clone_if_inference_tensor(minibatch["actor_obs_raw"])
            mirrored_actor_obs_raw = self.symmetry_utils.augment_observations(
                obs=raw_actor_obs,
                env=self.env,
                obs_list=self.actor_obs_keys,
            )[original_batch_size:]
            symmetry_original_actor_obs = self._normalize_actor_obs(raw_actor_obs, update=False)
            mirrored_actor_obs = self._normalize_actor_obs(mirrored_actor_obs_raw, update=False)

            mean_actions_for_original_batch = self.actor.act_inference(
                {"actor_obs": symmetry_original_actor_obs}
            )
            mean_actions_for_symmetry_batch = self.actor.act_inference(
                {"actor_obs": mirrored_actor_obs}
            )
            mean_symmetry_actions_batch = self.symmetry_utils.augment_actions(
                actions=mean_actions_for_original_batch,
            )[original_batch_size:]
            symmetry_actor_loss = F.mse_loss(
                mean_actions_for_symmetry_batch,
                mean_symmetry_actions_batch,
            )
        else:
            symmetry_actor_loss = actor_obs.new_zeros(())

        if symmetry_critic_enabled:
            raw_critic_obs = _clone_if_inference_tensor(minibatch["critic_obs_raw"])
            mirrored_critic_obs_raw = self.symmetry_utils.augment_observations(
                obs=raw_critic_obs,
                env=self.env,
                obs_list=self.critic_obs_keys,
            )[original_batch_size:]
            symmetry_original_critic_obs = self._normalize_critic_obs(
                raw_critic_obs,
                update=False,
            )
            mirrored_critic_obs = self._normalize_critic_obs(
                mirrored_critic_obs_raw,
                update=False,
            )
            symmetry_original_value_batch = self.critic.evaluate(
                {"critic_obs": symmetry_original_critic_obs}
            )
            mirrored_value_batch = self.critic.evaluate({"critic_obs": mirrored_critic_obs})
            symmetry_critic_loss = F.mse_loss(
                symmetry_original_value_batch,
                mirrored_value_batch,
            )
        else:
            symmetry_critic_loss = actor_obs.new_zeros(())

        entropy_loss = entropy_batch.mean()
        if pure_bc_objective:
            # Entropy is still useful as a diagnostic during deterministic
            # BC, but it is not part of that objective.  Leaving the live
            # entropy tensor behind an exact ``ppo_weight == 0`` gives the
            # policy std a material zero gradient.  Optimizers such as AdamW
            # then advance std's optimizer state and can apply decoupled
            # weight decay even though neither PPO nor std matching is
            # active.  Detach the diagnostic so std remains truly absent
            # from the BC graph; an explicit dagger_match_std loss below can
            # still provide its intended gradient.
            entropy_loss = entropy_loss.detach()
        entropy_coef = self._operational_entropy_coefficient()
        actor_loss_base = surrogate_loss - entropy_coef * entropy_loss
        actor_regularizer = self.config.symmetry_actor_coef * symmetry_actor_loss

        critic_loss = self.config.value_loss_coef * value_loss + self.config.symmetry_critic_coef * symmetry_critic_loss

        actor_loss = actor_loss_base + actor_regularizer
        distill_loss = torch.tensor(0.0, device=self.device)
        bc_loss = torch.tensor(0.0, device=self.device)
        current_bc_loss = torch.tensor(0.0, device=self.device)
        replay_bc_loss = torch.tensor(0.0, device=self.device)
        replay_bc_active = False
        dagger_weight = torch.tensor(0.0, device=self.device)
        actor_optimizer_step_enabled = True
        dagger_objective = self.distill_mode == "dagger" and self.dagger_enabled
        effective_dagger_weight = self._effective_dagger_loss_weight()
        if dagger_objective:
            if self.use_ppo_dagger_schedule:
                ppo_weight = self._operational_ppo_coefficient(self.ppo_coeff)
            else:
                ppo_weight = max(0.0, 1.0 - float(self.bc_loss_coef))
            dagger_weight = torch.tensor(effective_dagger_weight, device=self.device)
            actor_loss = ppo_weight * actor_loss_base + actor_regularizer
            actor_optimizer_step_enabled = bool(
                ppo_weight > 0.0
                or (self.use_symmetry and self.config.symmetry_actor_coef > 0.0)
            )

        if dagger_objective and effective_dagger_weight > 0.0:
            if debug_heartbeat:
                logger.info("Heartbeat: iter {} rank {} loss dagger begin", self.current_learning_iteration, rank_label)
            teacher_actions_batch = minibatch.get("teacher_actions")
            if teacher_actions_batch is None:
                raise ValueError("Dagger enabled but teacher_actions are missing from rollout storage.")
            teacher_actions_batch = teacher_actions_batch[:original_batch_size]
            if self.clip_teacher_actions:
                teacher_actions_batch = self._clip_teacher_actions_preserving_non_finite(
                    teacher_actions_batch,
                    self.clip_actions_threshold,
                )

            if self._actor_uses_flow_matching():
                distill_actor_policy_state = {"actor_obs": actor_obs[:original_batch_size]}
                if actor_perception_obs is not None:
                    distill_actor_policy_state[self.actor_perception_key] = actor_perception_obs[:original_batch_size]
                distill_per_sample = self.actor.flow_matching_loss(
                    distill_actor_policy_state,
                    teacher_actions_batch,
                    loss_fn=self.distill_loss_fn,
                )
            else:
                distill_per_elem = self.distill_loss_fn(mu_batch, teacher_actions_batch, reduction="none")
                if distill_per_elem.ndim > 1:
                    distill_per_sample = distill_per_elem.mean(dim=-1)
                else:
                    distill_per_sample = distill_per_elem

            valid_mask = torch.ones_like(distill_per_sample, dtype=torch.bool)
            teacher_bc_mask = minibatch.get("teacher_bc_mask")
            if teacher_bc_mask is not None:
                valid_mask &= teacher_bc_mask[:original_batch_size].view(-1).to(dtype=torch.bool)

            if self.dagger_ignore_zero_teacher_actions:
                expert_terminate = torch.all(teacher_actions_batch == 0.0, dim=-1)
                valid_mask &= ~expert_terminate

            valid_weight = valid_mask.to(dtype=distill_per_sample.dtype)
            bc_numerator = (distill_per_sample * valid_weight).sum()

            if self.dagger_match_std:
                if debug_heartbeat:
                    logger.info("Heartbeat: iter {} rank {} loss dagger std-match begin", self.current_learning_iteration, rank_label)
                if self.use_multi_teacher:
                    teacher_indices = minibatch.get("teacher_indices")
                    if teacher_indices is None:
                        raise ValueError("Multi-teacher enabled but teacher_indices are missing from rollout storage.")
                    teacher_indices = teacher_indices.view(-1)[:original_batch_size]
                    sigma_teacher = torch.zeros_like(sigma_batch)
                    for idx, teacher_actor in enumerate(self.teacher_actors):
                        mask = teacher_indices == idx
                        if mask.any():
                            sigma_teacher[mask] = self._get_actor_std_for_loss(teacher_actor).detach()
                else:
                    assert self.teacher_actor is not None, "Teacher actor is not initialized."
                    sigma_teacher = self._get_actor_std_for_loss(self.teacher_actor).detach()
                    sigma_teacher = sigma_teacher.unsqueeze(0).expand_as(sigma_batch)
                # Keep std matching on the same per-action scale as the action
                # BC loss above.  Summing here made its relative weight grow
                # linearly with the robot's action dimension.
                sigma_loss = self._std_matching_loss_per_sample(sigma_batch, sigma_teacher)
                bc_numerator = bc_numerator + (sigma_loss * valid_weight).sum()
                if debug_heartbeat:
                    logger.info("Heartbeat: iter {} rank {} loss dagger std-match finished", self.current_learning_iteration, rank_label)

            bc_denominator, bc_has_valid_samples = (
                self._bc_denominator_and_presence_for_minibatch(
                    minibatch,
                    valid_weight.sum(),
                )
            )
            current_bc_loss = bc_numerator / bc_denominator
            (
                replay_bc_loss,
                replay_bc_has_valid_samples,
                replay_bc_active,
            ) = self._compute_dagger_replay_bc_loss(
                minibatch,
                reference=current_bc_loss,
            )
            if replay_bc_active:
                bc_loss = (
                    (1.0 - self.dagger_replay_fraction) * current_bc_loss
                    + self.dagger_replay_fraction * replay_bc_loss
                )
            else:
                bc_loss = current_bc_loss

            # In DAgger mode, distillation objective is the BC term.
            distill_loss = bc_loss
            actor_loss = actor_loss + dagger_weight * bc_loss
            actor_optimizer_step_enabled = bool(
                actor_optimizer_step_enabled
                or bc_has_valid_samples
                or replay_bc_has_valid_samples
            )
            if debug_heartbeat:
                logger.info("Heartbeat: iter {} rank {} loss dagger finished", self.current_learning_iteration, rank_label)
        elif self._legacy_distillation_enabled():
            assert self.teacher_actor is not None, "Distillation enabled but teacher actor is not initialized."
            if debug_heartbeat:
                logger.info("Heartbeat: iter {} rank {} loss distill begin", self.current_learning_iteration, rank_label)
            teacher_actions = minibatch.get("teacher_actions")
            if teacher_actions is None:
                raise ValueError("Legacy distillation requires teacher_actions captured during rollout.")
            teacher_actions = teacher_actions[:original_batch_size]
            if self.clip_teacher_actions:
                teacher_actions = self._clip_teacher_actions_preserving_non_finite(
                    teacher_actions,
                    self.clip_actions_threshold,
                )
            if self._actor_uses_flow_matching():
                distill_actor_policy_state = {"actor_obs": actor_obs[:original_batch_size]}
                if actor_perception_obs is not None:
                    distill_actor_policy_state[self.actor_perception_key] = actor_perception_obs[:original_batch_size]
                distill_loss = self.actor.flow_matching_loss(
                    distill_actor_policy_state,
                    teacher_actions,
                    loss_fn=self.distill_loss_fn,
                ).mean()
            else:
                distill_loss = self.distill_loss_fn(mu_batch, teacher_actions)
            actor_loss = actor_loss + self.distill_loss_coef * distill_loss
            if debug_heartbeat:
                logger.info("Heartbeat: iter {} rank {} loss distill finished", self.current_learning_iteration, rank_label)

        if debug_heartbeat:
            logger.info("Heartbeat: iter {} rank {} loss finished", self.current_learning_iteration, rank_label)
        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "symmetry_actor_loss": symmetry_actor_loss,
            "symmetry_critic_loss": symmetry_critic_loss,
            "value_loss": value_loss,
            "surrogate_loss": surrogate_loss,
            "entropy_loss": entropy_loss,
            "entropy_coef": entropy_coef,
            "distill_loss": distill_loss,
            "bc_loss": bc_loss,
            "current_bc_loss": current_bc_loss,
            "replay_bc_loss": replay_bc_loss,
            "replay_bc_active": float(replay_bc_active),
            "ppo_coeff": float(self.ppo_coeff),
            "dagger_weight": dagger_weight,
            "kl_mean": kl_mean,
            "_reduce_kl_before_optimizer": True,
            "_actor_optimizer_step_enabled": actor_optimizer_step_enabled,
            "actor_optimizer_step_skipped_no_signal": float(not actor_optimizer_step_enabled),
        }

    def _encode_perception_sequence(
        self,
        encoder: nn.Module,
        obs_seq: torch.Tensor,
        dones_seq: torch.Tensor | None,
        initial_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not hasattr(encoder, "forward_sequence"):
            raise ValueError("Perception encoder does not support sequence encoding.")
        return encoder.forward_sequence(
            obs_seq,
            dones_seq=dones_seq,
            initial_hidden=initial_hidden,
        )

    def _compute_ppo_loss_sequence(self, minibatch: Minibatch):
        # Sequence shapes: [T, B, ...]
        if self.use_symmetry:
            raise ValueError("time_gru with symmetry is unsupported because recurrent perception mirroring is undefined.")
        actor_obs = minibatch["actor_obs"]
        actions_batch = minibatch["actions"]
        target_values_batch = minibatch["values"]
        advantages_batch = minibatch["advantages"]
        returns_batch = minibatch["returns"]
        old_actions_log_prob_batch = minibatch["actions_log_prob"]
        old_mu_batch = minibatch["action_mean"]
        old_sigma_batch = minibatch["action_sigma"]
        actor_perception_obs = minibatch.get(self.actor_perception_key) if self.actor_perception_key else None
        critic_perception_obs = minibatch.get(self.critic_perception_key) if self.critic_perception_key else None
        dones_seq = minibatch.get("dones")

        # actor_obs/critic_obs are the exact normalized tensors used during
        # collection.  Re-normalizing here changes the PPO likelihood ratio.
        actor_obs_flat = actor_obs.flatten(0, 1)
        critic_obs_flat = minibatch["critic_obs"].flatten(0, 1)

        # Encode perception sequences
        if actor_perception_obs is None or critic_perception_obs is None:
            raise ValueError("time_gru requires perception_obs for both actor and critic.")

        if hasattr(actor_perception_obs, "is_inference") and actor_perception_obs.is_inference():
            actor_perception_obs = actor_perception_obs.clone()
        if hasattr(critic_perception_obs, "is_inference") and critic_perception_obs.is_inference():
            critic_perception_obs = critic_perception_obs.clone()

        actor_hidden_seq = minibatch.get("actor_gru_hidden")
        critic_hidden_seq = minibatch.get("critic_gru_hidden")
        if actor_hidden_seq is None or critic_hidden_seq is None:
            raise ValueError("time_gru minibatches require stored actor_gru_hidden and critic_gru_hidden states.")
        actor_initial_hidden = actor_hidden_seq[0].unsqueeze(0)
        critic_initial_hidden = critic_hidden_seq[0].unsqueeze(0)
        actor_embed_seq = self._encode_perception_sequence(
            self.actor.perception_time_gru,
            actor_perception_obs,
            dones_seq,
            initial_hidden=actor_initial_hidden,
        )
        critic_embed_seq = self._encode_perception_sequence(
            self.critic.perception_time_gru,
            critic_perception_obs,
            dones_seq,
            initial_hidden=critic_initial_hidden,
        )

        # Flatten for PPO loss
        actor_embed_flat = actor_embed_seq.flatten(0, 1)
        critic_embed_flat = critic_embed_seq.flatten(0, 1)
        actions_flat = actions_batch.flatten(0, 1)
        target_values_flat = target_values_batch.flatten(0, 1)
        returns_flat = returns_batch.flatten(0, 1)
        advantages_flat = advantages_batch.flatten(0, 1)
        old_actions_log_prob_flat = old_actions_log_prob_batch.flatten(0, 1)
        old_mu_flat = old_mu_batch.flatten(0, 1)
        old_sigma_flat = old_sigma_batch.flatten(0, 1)

        original_batch_size = actions_flat.shape[0]

        actor_policy_state = {"actor_obs": actor_obs_flat, "extra_actor_input": actor_embed_flat}
        self.actor.update_distribution_from_policy_state(actor_policy_state)

        critic_policy_state = {"critic_obs": critic_obs_flat, "extra_critic_input": critic_embed_flat}
        value_batch = self.critic.evaluate(critic_policy_state)

        actions_log_prob_batch = self.actor.get_actions_log_prob(actions_flat)
        mu_batch = self.actor.action_mean[:original_batch_size]
        sigma_batch = self.actor.action_std[:original_batch_size]
        entropy_batch = self.actor.entropy[:original_batch_size]

        kl_mean = self._compute_kl_div(
            old_mu_flat,
            old_sigma_flat,
            mu_batch,
            sigma_batch,
            reduce_distributed=False,
        )

        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_flat))
        surrogate = -torch.squeeze(advantages_flat) * ratio
        surrogate_clipped = -torch.squeeze(advantages_flat) * torch.clamp(
            ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        value_clipped = target_values_flat + (value_batch - target_values_flat).clamp(
            -self.config.clip_param, self.config.clip_param
        )
        value_losses = (value_batch - returns_flat).pow(2)
        value_losses_clipped = (value_clipped - returns_flat).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()

        symmetry_actor_loss = torch.tensor(0.0, device=self.device)
        symmetry_critic_loss = torch.tensor(0.0, device=self.device)

        entropy_loss = entropy_batch.mean()
        entropy_coef = self._operational_entropy_coefficient()
        actor_loss_base = (
            surrogate_loss
            - entropy_coef * entropy_loss
            + self.config.symmetry_actor_coef * symmetry_actor_loss
        )
        actor_loss = actor_loss_base
        critic_loss = self.config.value_loss_coef * value_loss + self.config.symmetry_critic_coef * symmetry_critic_loss

        distill_loss = torch.tensor(0.0, device=self.device)
        bc_loss = torch.tensor(0.0, device=self.device)
        if self.distill_enabled or (self.distill_mode == "dagger" and self.dagger_enabled):
            raise ValueError("Distillation is not supported in time_gru mode.")

        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "symmetry_actor_loss": symmetry_actor_loss,
            "symmetry_critic_loss": symmetry_critic_loss,
            "value_loss": value_loss,
            "surrogate_loss": surrogate_loss,
            "entropy_loss": entropy_loss,
            "entropy_coef": entropy_coef,
            "distill_loss": distill_loss,
            "bc_loss": bc_loss,
            "ppo_coeff": float(self.ppo_coeff),
            "dagger_weight": 0.0,
            "kl_mean": kl_mean,
            "_reduce_kl_before_optimizer": True,
        }

    def _compute_kl_div(
        self,
        old_mu_batch,
        old_sigma_batch,
        mu_batch,
        sigma_batch,
        *,
        reduce_distributed: bool = True,
    ) -> torch.Tensor:
        with torch.inference_mode():
            # Compute the KL divergence between the old and new action distributions
            old_dist = Normal(old_mu_batch, old_sigma_batch)
            new_dist = Normal(mu_batch, sigma_batch)
            kl = kl_divergence(old_dist, new_dist).sum(-1)
            kl_mean = torch.mean(kl)

            # Reduce the KL divergence across all GPUs
            if self.is_multi_gpu and reduce_distributed:
                weight = self._get_distributed_loss_weight()
                payload = torch.stack([kl_mean * weight, kl_mean.new_tensor(weight)])
                payload = self._all_reduce_small_tensor(payload, op=torch.distributed.ReduceOp.SUM)
                if payload[1] <= 0.0:
                    raise ValueError("At least one rank must have a positive distributed_loss_weight.")
                kl_mean = payload[0] / payload[1]
        return kl_mean

    @staticmethod
    def _build_distributed_kl_payload(
        kl_mean: torch.Tensor,
        *,
        distributed_loss_weight: float,
    ) -> torch.Tensor:
        """Build all fallible rank-local KL reduction state before a verdict."""

        if not isinstance(kl_mean, torch.Tensor) or kl_mean.numel() != 1:
            raise ValueError("Local PPO KL must be one scalar tensor.")
        if kl_mean.is_complex():
            raise TypeError("Local PPO KL must be real-valued.")
        weight = float(distributed_loss_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                "distributed_loss_weight must be finite and non-negative before KL reduction."
            )
        payload = torch.stack(
            [
                kl_mean.reshape(()) * weight,
                kl_mean.new_tensor(weight).reshape(()),
            ]
        )
        return payload

    def _reduce_kl_after_local_loss(
        self,
        kl_mean: torch.Tensor,
        *,
        distributed_payload: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reduce local KL after every rank completed gradient reduction.

        Keeping the reduction outside ``_compute_ppo_loss`` means a rank-local
        forward/shape/device exception can enter the same all-rank outcome
        verdict as loss validation and backward instead of abandoning peers in
        a KL collective.  It runs after the gradient collective so any
        deterministic post-KL validation/adaptation error can share the
        existing gradient-clip verdict.  Moving adaptive-LR mutation to this
        point is training-equivalent: the current loss/backward graph and
        gradient reduction do not depend on optimizer LR, while the following
        optimizer step still observes the newly selected LR.
        """

        if self.is_multi_gpu:
            if distributed_payload is None:
                raise RuntimeError(
                    "Distributed KL reduction requires a prevalidated payload."
                )
            distributed_payload = self._all_reduce_small_tensor(
                distributed_payload,
                op=torch.distributed.ReduceOp.SUM,
            )
            if distributed_payload[1] <= 0.0:
                raise ValueError("At least one rank must have a positive distributed_loss_weight.")
            kl_mean = distributed_payload[0] / distributed_payload[1]
        if not bool(torch.isfinite(kl_mean).all().item()):
            raise FloatingPointError("Global PPO KL is NaN/Inf after distributed reduction.")
        if self._should_update_learning_rate_from_kl():
            self._update_learning_rate(kl_mean)
        return kl_mean

    def _should_update_learning_rate_from_kl(self) -> bool:
        """Whether measured KL may drive the adaptive learning-rate controller."""
        if self.config.desired_kl is None or self.config.schedule != "adaptive":
            return False
        # Pure BC has no likelihood-ratio objective, so KL is diagnostic-only.
        # As soon as PPO has any non-zero contribution, however, BC updates can
        # move the policy far enough to invalidate the rollout importance
        # ratios.  Keep the adaptive trust-region controller active even when
        # PPO is a small part of the hybrid objective.
        return not self._use_deterministic_student_actions()

    def _update_learning_rate(self, kl_mean: torch.Tensor):
        """Adapt the policy optimizer to policy KL, leaving the critic independent.

        Actor and critic have disjoint networks and optimizers in HoloSoma.
        Policy KL is therefore a trust-region signal only for the actor.  The
        upstream shared-optimizer PPO implementation necessarily changed both
        parameter sets together, but carrying that coupling into separate
        optimizers silently drives value learning to the actor LR floor during
        hybrid BC/PPO updates.
        """
        if kl_mean > self.config.desired_kl * 2.0:
            self.actor_learning_rate = max(self.min_actor_learning_rate, self.actor_learning_rate / 1.5)
        elif kl_mean < self.config.desired_kl / 2.0 and kl_mean > 0.0:
            self.actor_learning_rate = min(self.max_actor_learning_rate, self.actor_learning_rate * 1.5)

        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = self.actor_learning_rate

    @staticmethod
    def _move_checkpoint_value_to_device(value, device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: PPO._move_checkpoint_value_to_device(item, device) for key, item in value.items()}
        if isinstance(value, list):
            return [PPO._move_checkpoint_value_to_device(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(PPO._move_checkpoint_value_to_device(item, device) for item in value)
        return value

    def _move_optimizer_state_to_device(self, optimizer) -> None:
        for state in optimizer.state.values():
            for key, value in state.items():
                state[key] = self._move_checkpoint_value_to_device(value, self.device)

    @staticmethod
    def _canonical_motion_transition_contract(contract: Any) -> tuple[dict[str, Any], str]:
        # Keep the command implementation as the single training-side schema
        # authority without importing the WBT environment during PPO module
        # initialization (which would create an env/agent import cycle).
        from holosoma.managers.command.terms.wbt import (
            canonical_motion_transition_contract,
            motion_transition_contract_sha256,
        )

        canonical = canonical_motion_transition_contract(contract)
        return canonical, motion_transition_contract_sha256(canonical)

    def _local_motion_transition_contract(self) -> tuple[dict[str, Any] | None, str | None]:
        env = getattr(self, "unwrapped_env", None) or getattr(self, "env", None)
        if env is None:
            return None, None
        command_manager = getattr(env, "command_manager", None)
        if command_manager is None:
            return None, None
        motion_command = command_manager.get_state("motion_command")
        if motion_command is None:
            return None, None
        getter = getattr(motion_command, "get_motion_transition_contract", None)
        if not callable(getter):
            raise RuntimeError(
                "The live motion_command does not expose get_motion_transition_contract(); "
                "checkpoint/export cannot certify the effective prepend/append timeline."
            )
        return self._canonical_motion_transition_contract(getter())

    @staticmethod
    def _serialized_config_requests_motion_transitions(config: Any) -> bool:
        if not isinstance(config, Mapping):
            return False
        command = config.get("command")
        setup_terms = command.get("setup_terms") if isinstance(command, Mapping) else None
        motion_command = setup_terms.get("motion_command") if isinstance(setup_terms, Mapping) else None
        params = motion_command.get("params") if isinstance(motion_command, Mapping) else None
        motion_config = params.get("motion_config") if isinstance(params, Mapping) else None
        if not isinstance(motion_config, Mapping):
            return False
        for enabled_key, duration_key in (
            ("enable_default_pose_prepend", "default_pose_prepend_duration_s"),
            ("enable_default_pose_append", "default_pose_append_duration_s"),
        ):
            # MotionConfig historically defaulted both transitions to enabled
            # for two seconds, so absent serialized fields are not evidence of
            # an inactive legacy timeline.
            enabled = motion_config.get(enabled_key, True)
            duration = motion_config.get(duration_key, 2.0)
            if type(enabled) is not bool:
                return True
            if not enabled:
                continue
            if isinstance(duration, bool) or not isinstance(duration, numbers.Real):
                return True
            if not math.isfinite(float(duration)) or float(duration) > 0.0:
                return True
        return False

    def _validate_checkpoint_motion_transition_contract(
        self,
        loaded_dict: Mapping[str, Any],
        *,
        live_contract: dict[str, Any] | None,
        compare_live: bool,
        operation: str,
        allow_missing_contract: bool = False,
    ) -> tuple[dict[str, Any] | None, str | None]:
        contract_key = "motion_transition_contract"
        digest_key = "motion_transition_contract_sha256"
        has_contract = contract_key in loaded_dict
        has_digest = digest_key in loaded_dict
        if has_contract != has_digest:
            raise ValueError(
                f"{operation} checkpoint must provide both {contract_key} and {digest_key}, or neither."
            )

        canonical_live: dict[str, Any] | None = None
        if live_contract is not None:
            canonical_live, _ = self._canonical_motion_transition_contract(live_contract)

        if not has_contract:
            if allow_missing_contract:
                # Actor-only warm start is not a continuation of the source
                # rollout timeline.  The newly launched training environment's
                # live contract remains authoritative for future checkpoints.
                return None, None
            source_requests_transitions = self._serialized_config_requests_motion_transitions(
                loaded_dict.get("experiment_config")
            )
            if source_requests_transitions or (compare_live and canonical_live is not None):
                raise ValueError(
                    f"{operation} checkpoint predates motion_transition_contract while its source requests "
                    "transitions or the full-resume runtime has a live motion timeline that must be "
                    "authenticated. Requested flags cannot distinguish single-clip static append semantics "
                    "from global multi-clip runtime-prepend/no-append semantics; re-save/retrain with the "
                    "current contract format."
                )
            # A motion-free full resume and an explicitly transition-free legacy
            # evaluation have no timeline contract to preserve.
            return None, None

        raw_digest = loaded_dict[digest_key]
        if (
            type(raw_digest) is not str
            or len(raw_digest) != 64
            or any(char not in "0123456789abcdef" for char in raw_digest)
        ):
            raise ValueError(f"{operation} checkpoint has invalid {digest_key}: {raw_digest!r}.")
        saved_contract, computed_digest = self._canonical_motion_transition_contract(
            loaded_dict[contract_key]
        )
        if raw_digest != computed_digest:
            raise ValueError(
                f"{operation} checkpoint {digest_key} does not authenticate {contract_key}: "
                f"declared={raw_digest}, computed={computed_digest}."
            )
        if compare_live:
            if canonical_live is None:
                raise ValueError(
                    f"{operation} checkpoint declares a motion transition contract but the live runtime "
                    "has no motion_command contract to compare."
                )
            if saved_contract != canonical_live:
                raise ValueError(
                    f"{operation} motion transition contract differs from the live runtime: "
                    f"checkpoint={saved_contract}, live={canonical_live}."
                )
        return saved_contract, computed_digest

    def _collect_distributed_motion_transition_contract(
        self,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Collect and require one identical effective timeline across all ranks."""

        # Algorithms/environments without a command manager have no motion
        # timeline to certify.  Avoid inserting an unrelated collective into
        # pure-policy/locomotion checkpoint protocols.
        env = getattr(self, "unwrapped_env", None) or getattr(self, "env", None)
        if env is None:
            return None, None
        if getattr(env, "command_manager", None) is None:
            return None, None
        distributed = (
            getattr(self, "is_multi_gpu", False)
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if not distributed:
            return self._local_motion_transition_contract()

        group = self._setup_gloo_barrier_group()
        if group is None:
            raise RuntimeError("Distributed motion-transition validation requires an all-rank Gloo group.")
        world_size = torch.distributed.get_world_size(group=group)
        rank = torch.distributed.get_rank(group=group)
        try:
            local_contract, local_digest = self._local_motion_transition_contract()
            local_error = None
        except Exception as exc:
            local_contract = None
            local_digest = None
            local_error = f"{type(exc).__name__}: {exc}"
        gathered: list[dict[str, Any] | None] = [None] * world_size
        torch.distributed.all_gather_object(
            gathered,
            {
                "rank": rank,
                "error": local_error,
                "contract": local_contract,
                "digest": local_digest,
            },
            group=group,
        )
        malformed: list[str] = []
        failures: list[str] = []
        validated: list[tuple[dict[str, Any] | None, str | None]] = []
        for expected_rank, result in enumerate(gathered):
            if not isinstance(result, dict) or result.get("rank") != expected_rank:
                malformed.append(
                    f"slot={expected_rank}: malformed payload/reported_rank="
                    f"{result.get('rank') if isinstance(result, dict) else None}"
                )
                continue
            if result.get("error") is not None:
                failures.append(f"rank={expected_rank}: {result['error']}")
                continue
            contract = result.get("contract")
            digest = result.get("digest")
            if (contract is None) != (digest is None):
                malformed.append(f"rank={expected_rank}: unpaired contract/digest")
                continue
            if contract is None:
                validated.append((None, None))
                continue
            try:
                canonical, computed_digest = self._canonical_motion_transition_contract(contract)
            except Exception as exc:
                malformed.append(f"rank={expected_rank}: {type(exc).__name__}: {exc}")
                continue
            if digest != computed_digest:
                malformed.append(
                    f"rank={expected_rank}: declared digest {digest!r} != computed {computed_digest!r}"
                )
                continue
            validated.append((canonical, computed_digest))
        if malformed:
            raise RuntimeError(
                "Distributed motion-transition collection returned malformed payloads: "
                + "; ".join(malformed)
            )
        if failures:
            raise RuntimeError(
                "Motion-transition contract construction failed on at least one rank: "
                + "; ".join(failures)
            )
        if len(validated) != world_size:
            raise RuntimeError("Distributed motion-transition collection returned an incomplete rank set.")
        reference = validated[0]
        mismatched = [rank_id for rank_id, value in enumerate(validated) if value != reference]
        if mismatched:
            raise RuntimeError(
                "DDP ranks disagree on the effective motion-transition contract; "
                f"reference_rank=0 contract={reference[0]}, mismatched_ranks={mismatched}."
            )
        return reference

    def _prepare_motion_transition_contract_for_export(self) -> None:
        contract = getattr(self, "_motion_transition_contract", None)
        digest = getattr(self, "_motion_transition_contract_sha256", None)
        if (contract is None) != (digest is None):
            raise RuntimeError("Export has unpaired motion-transition contract metadata.")
        if contract is not None:
            canonical, computed = self._canonical_motion_transition_contract(contract)
            if digest != computed:
                raise RuntimeError(
                    "Export motion_transition_contract_sha256 does not match the retained contract."
                )
            self._motion_transition_contract = canonical
            return
        if bool(getattr(self, "_evaluation_only", False)):
            # Evaluation loading already validated the source checkpoint.  Never
            # replace its source timeline with the temporary single-clip eval env.
            return
        if (
            getattr(self, "is_multi_gpu", False)
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            raise RuntimeError(
                "Distributed ONNX export requires a preceding all-rank checkpoint save so the effective "
                "motion-transition contract is certified before rank-zero-only export."
            )
        contract, digest = self._local_motion_transition_contract()
        if contract is None:
            experiment_config = getattr(self, "_experiment_config", None)
            serialized = (
                experiment_config.to_serializable_dict()
                if experiment_config is not None and hasattr(experiment_config, "to_serializable_dict")
                else None
            )
            if self._serialized_config_requests_motion_transitions(serialized):
                raise RuntimeError(
                    "ONNX export configuration requests default-pose transitions but no live "
                    "motion_transition_contract is available."
                )
        self._motion_transition_contract = contract
        self._motion_transition_contract_sha256 = digest

    def _collect_distributed_env_states(self) -> dict[str, dict]:
        """Collect rank-local environment checkpoint state on every rank."""
        distributed = (
            getattr(self, "is_multi_gpu", False)
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if not distributed:
            local_state = self._move_checkpoint_value_to_device(self._collect_env_state(), "cpu")
            if not isinstance(local_state, dict):
                raise ValueError(
                    f"Environment checkpoint state must be a mapping, got {type(local_state).__name__}."
                )
            validate_finite_tree(local_state, path=f"env_state_by_rank[{self.gpu_global_rank}]")
            return {str(self.gpu_global_rank): local_state}

        # Establish the collective before rank-local serialization.  A local
        # environment exception is then carried in the same all-rank envelope
        # instead of letting peers block forever in all_gather_object.
        gloo_group = self._setup_gloo_barrier_group()
        if gloo_group is None:
            raise RuntimeError("Distributed environment checkpointing requires an all-rank Gloo group.")
        group_world_size = torch.distributed.get_world_size(group=gloo_group)
        group_rank = torch.distributed.get_rank(group=gloo_group)
        try:
            local_state = self._move_checkpoint_value_to_device(self._collect_env_state(), "cpu")
            if not isinstance(local_state, dict):
                raise ValueError(
                    f"Environment checkpoint state must be a mapping, got {type(local_state).__name__}."
                )
            validate_finite_tree(local_state, path=f"env_state_by_rank[{group_rank}]")
            local_error = None
        except Exception as exc:  # synchronize rank-local serialization failures
            local_state = None
            local_error = f"{type(exc).__name__}: {exc}"

        gathered_results: list[dict[str, Any] | None] = [None] * group_world_size
        torch.distributed.all_gather_object(
            gathered_results,
            {"rank": group_rank, "error": local_error, "state": local_state},
            group=gloo_group,
        )
        problems = []
        failures = []
        states: dict[str, dict] = {}
        for expected_rank, result in enumerate(gathered_results):
            if not isinstance(result, dict) or result.get("rank") != expected_rank:
                problems.append(
                    f"slot={expected_rank}: malformed payload/reported_rank="
                    f"{result.get('rank') if isinstance(result, dict) else None}"
                )
                continue
            if result.get("error") is not None:
                failures.append(f"rank={expected_rank}: {result['error']}")
                continue
            state = result.get("state")
            if not isinstance(state, dict):
                problems.append(
                    f"rank={expected_rank}: state_type={type(state).__name__}"
                )
                continue
            states[str(expected_rank)] = state
        if problems:
            raise RuntimeError(
                "Distributed environment checkpoint collection returned malformed payloads: "
                + "; ".join(problems)
            )
        if failures:
            raise RuntimeError(
                "Refusing to checkpoint because environment state collection failed on at "
                f"least one rank: {'; '.join(failures)}."
            )
        return states

    @staticmethod
    def _allow_nondeterministic_rng_resume() -> bool:
        """Return the exact legacy-checkpoint escape hatch."""

        raw_value = os.environ.get(ALLOW_NONDETERMINISTIC_RNG_RESUME_ENV)
        if raw_value is None or raw_value.strip() in ("", "0"):
            return False
        if raw_value.strip() == "1":
            return True
        raise ValueError(
            f"{ALLOW_NONDETERMINISTIC_RNG_RESUME_ENV} must be exactly 0 or 1; "
            f"got {raw_value!r}."
        )

    @staticmethod
    def _allow_legacy_rollout_restart_resume() -> bool:
        raw_value = os.environ.get(_ALLOW_LEGACY_ROLLOUT_RESUME_ENV)
        if raw_value is None or raw_value.strip() in ("", "0"):
            return False
        if raw_value.strip() == "1":
            return True
        raise ValueError(
            f"{_ALLOW_LEGACY_ROLLOUT_RESUME_ENV} must be exactly 0 or 1; "
            f"got {raw_value!r}."
        )

    def _reset_rollout_at_checkpoint_enabled(self) -> bool:
        config = getattr(self, "config", None)
        reset_at_checkpoint = getattr(
            self,
            "reset_rollout_at_checkpoint",
            getattr(config, "reset_rollout_at_checkpoint", False),
        )
        if type(reset_at_checkpoint) is not bool:
            raise ValueError(
                "PPO reset_rollout_at_checkpoint must be an explicit boolean, "
                f"got {reset_at_checkpoint!r}."
            )
        return reset_at_checkpoint

    def _rollout_resume_contract(self, next_iteration: int) -> dict[str, Any]:
        """Describe whether recovery is trajectory-equivalent at this boundary."""

        config = getattr(self, "config", None)
        reset_at_checkpoint = self._reset_rollout_at_checkpoint_enabled()
        if reset_at_checkpoint:
            version = _EXACT_ROLLOUT_RESUME_CONTRACT_VERSION
            mode = _EXACT_ROLLOUT_RESUME_CONTRACT_MODE
        else:
            version = _RECOVERY_ROLLOUT_RESUME_CONTRACT_VERSION
            mode = _RECOVERY_ROLLOUT_RESUME_CONTRACT_MODE
        return {
            "version": version,
            "mode": mode,
            "next_iteration": int(next_iteration),
            "save_interval": getattr(config, "save_interval", None),
            "init_at_random_ep_len": bool(
                getattr(config, "init_at_random_ep_len", False)
            ),
            "dagger_ignore_episode_initial_steps": int(
                getattr(self, "dagger_ignore_episode_initial_steps", 0)
            ),
            "reset_recurrent_hidden": True,
            "perception_state_mode": "checkpoint_stream_state_rebuild_derived_cache",
        }

    def _validate_rollout_resume_contract(
        self,
        loaded_dict: dict[str, Any],
        *,
        next_iteration: int,
    ) -> None:
        contract = loaded_dict.get("rollout_resume_contract")
        if contract is None:
            if not self._allow_legacy_rollout_restart_resume():
                raise RuntimeError(
                    "Checkpoint has no rollout_resume_contract. Its RNG snapshot cannot prove "
                    "equivalence to uninterrupted training because simulator/observation/DAgger/recurrent "
                    "episode state was not checkpointed. Refusing an incorrectly labelled exact resume. "
                    f"Set {_ALLOW_LEGACY_ROLLOUT_RESUME_ENV}=1 only to accept a new, non-equivalent "
                    "restart lineage from this legacy artifact."
                )
            logger.warning(
                "Checkpoint has no rollout_resume_contract. {}=1 is set; continuing as a legacy "
                "restart lineage that is not equivalent to uninterrupted training.",
                _ALLOW_LEGACY_ROLLOUT_RESUME_ENV,
            )
            return
        if not isinstance(contract, dict):
            raise ValueError("Checkpoint rollout_resume_contract must be a dictionary.")
        expected = self._rollout_resume_contract(next_iteration)
        if contract != expected:
            raise ValueError(
                "Checkpoint rollout-resume contract differs from the active runtime: "
                f"checkpoint={contract!r}, expected={expected!r}."
            )
        if contract.get("mode") == _RECOVERY_ROLLOUT_RESUME_CONTRACT_MODE:
            logger.warning(
                "Authenticated recovery checkpoint restores model, optimizer, curriculum, and RNG "
                "state, but starts a new simulator episode stream. This is intentional recovery/fine-tune "
                "semantics and is not trajectory-equivalent to uninterrupted training."
            )

    def _collect_distributed_rng_states(self) -> dict[str, dict[str, Any]]:
        """Capture one process-global RNG snapshot for every global rank.

        The Gloo object collective and checkpoint publication are boundary
        side effects, not training stochasticity.  Restore the just-captured
        local state after the collective so they cannot advance a stream.
        """

        rank = int(getattr(self, "gpu_global_rank", 0) or 0)
        distributed = (
            getattr(self, "is_multi_gpu", False)
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if not distributed:
            return {str(rank): capture_rng_checkpoint_state()}

        gloo_group = self._setup_gloo_barrier_group()
        group_error = (
            RuntimeError("Distributed RNG checkpointing requires an all-rank Gloo group.")
            if gloo_group is None
            else None
        )
        self._synchronize_training_phase_error(
            group_error,
            operation="distributed RNG checkpoint group validation",
        )
        assert gloo_group is not None
        group_world_size = torch.distributed.get_world_size(group=gloo_group)
        group_rank = torch.distributed.get_rank(group=gloo_group)
        expected_world_size = int(getattr(self, "gpu_world_size", 1) or 1)
        topology_error = None
        if group_world_size != expected_world_size or group_rank != rank:
            topology_error = RuntimeError(
                "Distributed RNG checkpoint Gloo topology disagrees with PPO topology: "
                f"group={group_rank}/{group_world_size}, PPO={rank}/{expected_world_size}."
            )
        self._synchronize_training_phase_error(
            topology_error,
            operation="distributed RNG checkpoint topology validation",
        )
        local_state: dict[str, Any] | None = None
        try:
            local_state = capture_rng_checkpoint_state()
            local_error = None
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"

        gathered_results: list[dict[str, Any] | None] = [None] * group_world_size
        gather_error: Exception | None = None
        try:
            torch.distributed.all_gather_object(
                gathered_results,
                {"rank": group_rank, "error": local_error, "state": local_state},
                group=gloo_group,
            )
        except Exception as exc:
            gather_error = exc

        restore_error: Exception | None = None
        if local_state is not None:
            try:
                restore_rng_checkpoint_state(
                    local_state,
                    path=f"rng_state_by_rank[{group_rank}]",
                )
            except Exception as exc:
                restore_error = exc
        self._synchronize_training_phase_error(
            restore_error,
            operation="distributed RNG checkpoint local-state restoration",
        )
        self._synchronize_training_phase_error(
            gather_error,
            operation="distributed RNG checkpoint gather outcome",
        )

        problems = []
        failures = []
        states: dict[str, dict[str, Any]] = {}
        expected_cuda_count = (
            int(local_state["torch_cuda_visible_device_count"])
            if isinstance(local_state, dict)
            else None
        )
        for expected_rank, result in enumerate(gathered_results):
            if not isinstance(result, dict) or result.get("rank") != expected_rank:
                problems.append(
                    f"slot={expected_rank}: malformed payload/reported_rank="
                    f"{result.get('rank') if isinstance(result, dict) else None}"
                )
                continue
            if result.get("error") is not None:
                failures.append(f"rank={expected_rank}: {result['error']}")
                continue
            try:
                states[str(expected_rank)] = validate_rng_checkpoint_state(
                    result.get("state"),
                    path=f"rng_state_by_rank[{expected_rank}]",
                    expected_cuda_device_count=expected_cuda_count,
                    validate_cuda_generators=False,
                )
            except ValueError as exc:
                problems.append(f"rank={expected_rank}: {exc}")
        validation_error: Exception | None = None
        if problems:
            validation_error = RuntimeError(
                "Distributed RNG checkpoint collection returned malformed payloads: "
                + "; ".join(problems)
            )
        elif failures:
            validation_error = RuntimeError(
                "Refusing to checkpoint because RNG-state capture failed on at least one rank: "
                + "; ".join(failures)
            )
        self._synchronize_training_phase_error(
            validation_error,
            operation="distributed RNG checkpoint payload validation",
        )
        return states

    def _local_fixed_bc_eval_checkpoint_state(self) -> dict[str, Any] | None:
        allocation = self._fixed_bc_eval_allocation_contract()
        ready = bool(getattr(self, "_fixed_bc_eval_ready", False) or allocation["local_target"] == 0)
        size = int(getattr(self, "_fixed_bc_eval_size", 0) or 0)
        local_target = int(allocation["local_target"])
        if ready and size != local_target:
            raise RuntimeError(
                "Refusing to checkpoint an inconsistent ready fixed BC stratum: "
                f"rank={allocation['rank']}, size={size}, local_target={local_target}."
            )
        if not ready and not (0 <= size < local_target):
            raise RuntimeError(
                "Refusing to checkpoint an inconsistent partial fixed BC stratum: "
                f"rank={allocation['rank']}, size={size}, local_target={local_target}."
            )
        if ready:
            tensors = getattr(self, "_fixed_bc_eval_dataset", {})
        else:
            tensors = {}
            parts_by_key = {
                "actor_obs_raw": getattr(self, "_fixed_bc_eval_actor_obs_parts", []),
                "teacher_actions": getattr(self, "_fixed_bc_eval_teacher_actions_parts", []),
                "actor_perception": getattr(self, "_fixed_bc_eval_actor_perception_parts", []),
            }
            for key, parts in parts_by_key.items():
                if parts:
                    tensors[key] = torch.cat(parts, dim=0)
        if not isinstance(tensors, dict):
            return None
        required = {"actor_obs_raw", "teacher_actions"}
        if getattr(self, "actor_perception_key", ""):
            required.add("actor_perception")
        if size > 0:
            missing = required - set(tensors)
            if missing:
                raise RuntimeError(
                    "Refusing to checkpoint a fixed BC stratum with missing tensors: "
                    f"rank={allocation['rank']}, missing={sorted(missing)}."
                )
            row_counts = {
                int(tensors[key].shape[0])
                for key in required
                if isinstance(tensors.get(key), torch.Tensor) and tensors[key].ndim >= 1
            }
            if row_counts != {size}:
                raise RuntimeError(
                    "Refusing to checkpoint a fixed BC stratum with inconsistent tensor rows: "
                    f"rank={allocation['rank']}, expected={size}, actual={sorted(row_counts)}."
                )
        state: dict[str, Any] = {
            "ready": ready,
            "size": size,
            **allocation,
        }
        state.update(
            {
                key: value.detach().cpu().clone()
                for key, value in tensors.items()
                if isinstance(value, torch.Tensor)
            }
        )
        return state

    def _fixed_bc_guard_checkpoint_dataset_digests(
        self,
        states_by_rank: Mapping[str, Any],
    ) -> tuple[dict[str, str], str]:
        world_size = int(getattr(self, "gpu_world_size", 1) or 1)
        budget = int(getattr(self, "fixed_bc_eval_num_samples", 0) or 0)
        expected_keys = {str(rank) for rank in range(world_size)}
        if not isinstance(states_by_rank, Mapping) or set(states_by_rank) != expected_keys:
            actual_keys = (
                sorted(str(key) for key in states_by_rank)
                if isinstance(states_by_rank, Mapping)
                else []
            )
            raise ValueError(
                "Fixed-BC guard checkpoint dataset map must contain every canonical rank: "
                f"checkpoint={actual_keys}, "
                f"expected={sorted(expected_keys)}."
            )
        quotient, remainder = divmod(budget, world_size)
        digest_by_rank: dict[str, str] = {}
        for rank in range(world_size):
            state = states_by_rank[str(rank)]
            expected_target = quotient + int(rank < remainder)
            if not isinstance(state, Mapping):
                raise ValueError(
                    f"fixed_bc_eval_by_rank[{rank}] must be a mapping for guard digest validation."
                )
            allocation_expectations = {
                "allocation_version": _FIXED_BC_EVAL_ALLOCATION_VERSION,
                "allocation_scheme": _FIXED_BC_EVAL_ALLOCATION_SCHEME,
                "global_sample_budget": budget,
                "world_size": world_size,
                "rank": rank,
                "local_target": expected_target,
                "ready": True,
                "size": expected_target,
            }
            mismatches = [
                f"{key}={state.get(key)!r} != {expected!r}"
                for key, expected in allocation_expectations.items()
                if type(state.get(key)) is not type(expected)
                or state.get(key) != expected
            ]
            if mismatches:
                raise ValueError(
                    f"fixed_bc_eval_by_rank[{rank}] is not a complete immutable guard stratum: "
                    + "; ".join(mismatches)
                )
            digest_by_rank[str(rank)] = self._fixed_bc_guard_digest_dataset(
                state,
                expected_rows=expected_target,
                context=f"checkpoint fixed BC dataset rank {rank}",
            )
        global_digest = self._fixed_bc_guard_global_dataset_digest(
            digest_by_rank,
            global_sample_budget=budget,
            world_size=world_size,
        )
        return digest_by_rank, global_digest

    def _fixed_bc_guard_checkpoint_state(
        self,
        fixed_bc_eval_by_rank: Mapping[str, Any],
        *,
        next_iteration: int,
        allow_tripped: bool = False,
    ) -> dict[str, Any] | None:
        if not bool(getattr(self, "fixed_bc_guard_enabled", False)):
            return None
        state: dict[str, Any] | None = None
        local_error: Exception | None = None
        try:
            digest_by_rank: dict[str, str] = {}
            global_digest = ""
            if next_iteration > 0:
                digest_by_rank, global_digest = (
                    self._fixed_bc_guard_checkpoint_dataset_digests(
                        fixed_bc_eval_by_rank
                    )
                )
            state = self._validate_fixed_bc_guard_checkpoint_state(
                self._fixed_bc_guard_state,
                next_iteration=next_iteration,
                digest_by_rank=digest_by_rank,
                global_digest=global_digest,
                allow_tripped=allow_tripped,
            )
        except Exception as exc:
            local_error = exc
        # A rank-local guard/dataset corruption must be agreed by every worker
        # before healthy peers enter the state-digest all-gather below.
        self._synchronize_training_phase_error(
            local_error,
            operation="fixed BC guard checkpoint-state validation",
        )
        assert state is not None
        self._require_all_rank_fixed_bc_guard_state_match(state)
        return state

    def _require_all_rank_fixed_bc_guard_state_match(
        self,
        state: Mapping[str, Any],
    ) -> None:
        local_error: Exception | None = None
        state_digest: str | None = None
        try:
            state_digest = hashlib.sha256(
                json.dumps(
                    state,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
        except Exception as exc:
            local_error = exc
        self._synchronize_training_phase_error(
            local_error,
            operation="fixed BC guard checkpoint-state canonicalization",
        )
        assert state_digest is not None
        distributed = (
            getattr(self, "is_multi_gpu", False)
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if not distributed:
            return
        group = self._setup_gloo_barrier_group()
        group_error = (
            RuntimeError(
                "Distributed fixed-BC guard checkpoint validation requires an all-rank Gloo group."
            )
            if group is None
            else None
        )
        self._synchronize_training_phase_error(
            group_error,
            operation="fixed BC guard state-digest group validation",
        )
        assert group is not None
        world_size = torch.distributed.get_world_size(group=group)
        rank = torch.distributed.get_rank(group=group)
        expected_world_size = int(getattr(self, "gpu_world_size", 1) or 1)
        expected_rank = int(getattr(self, "gpu_global_rank", 0) or 0)
        topology_error: Exception | None = None
        if world_size != expected_world_size or rank != expected_rank:
            topology_error = RuntimeError(
                "Fixed-BC guard checkpoint Gloo topology disagrees with PPO topology: "
                f"group={rank}/{world_size}, PPO={expected_rank}/{expected_world_size}."
            )
        self._synchronize_training_phase_error(
            topology_error,
            operation="fixed BC guard state-digest topology validation",
        )
        gathered: list[dict[str, Any] | None] = [None] * world_size
        torch.distributed.all_gather_object(
            gathered,
            {"rank": rank, "state_digest": state_digest},
            group=group,
        )
        problems = []
        for expected_rank, result in enumerate(gathered):
            if not isinstance(result, dict) or result.get("rank") != expected_rank:
                problems.append(f"slot={expected_rank}: malformed state envelope")
            elif result.get("state_digest") != state_digest:
                problems.append(
                    f"rank={expected_rank}: guard state differs from rank {rank}"
                )
        if problems:
            raise RuntimeError(
                "Refusing to checkpoint divergent all-rank fixed-BC guard state: "
                + "; ".join(problems)
            )

    def _validate_fixed_bc_guard_checkpoint_state(
        self,
        payload: Mapping[str, Any],
        *,
        next_iteration: int,
        digest_by_rank: Mapping[str, str],
        global_digest: str,
        allow_tripped: bool = False,
    ) -> dict[str, Any]:
        expected_fields = set(self._new_fixed_bc_guard_state())
        if not isinstance(payload, Mapping) or set(payload) != expected_fields:
            actual_fields = set(payload) if isinstance(payload, Mapping) else set()
            raise ValueError(
                "Checkpoint fixed_bc_guard_state schema mismatch: "
                f"missing={sorted(expected_fields - actual_fields)}, "
                f"extra={sorted(actual_fields - expected_fields)}."
            )
        if type(payload.get("version")) is not int or payload["version"] != _FIXED_BC_GUARD_STATE_VERSION:
            raise ValueError(
                "Checkpoint fixed_bc_guard_state version mismatch: "
                f"checkpoint={payload.get('version')!r}, "
                f"runtime={_FIXED_BC_GUARD_STATE_VERSION}."
            )
        expected_fingerprint = str(self._fixed_bc_guard_config_fingerprint)
        if payload.get("config_fingerprint") != expected_fingerprint:
            raise ValueError(
                "Checkpoint fixed-BC guard configuration fingerprint differs from the active run."
            )
        if type(payload.get("tripped")) is not bool:
            raise ValueError("Checkpoint fixed_bc_guard_state.tripped must be boolean.")
        tripped = payload["tripped"]
        if tripped and not allow_tripped:
            raise RuntimeError(
                "Checkpoint was published after the fixed-BC scientific guard tripped and "
                "cannot be resumed."
            )
        if allow_tripped and not tripped:
            raise ValueError(
                "Diagnostic fixed-BC guard checkpoint requires tripped=True."
            )

        interval = int(self.fixed_bc_eval_log_interval)
        completed_iteration = next_iteration - 1
        expected_last_eval = (
            None
            if completed_iteration < 0
            else (completed_iteration // interval) * interval
        )
        last_eval = payload.get("last_eval_iteration")
        if (
            (expected_last_eval is None and last_eval is not None)
            or (
                expected_last_eval is not None
                and (type(last_eval) is not int or last_eval != expected_last_eval)
            )
        ):
            raise ValueError(
                "Checkpoint fixed-BC guard timeline is incomplete: "
                f"last_eval={last_eval!r}, expected={expected_last_eval!r}."
            )
        reference_end = int(self.fixed_bc_guard_reference_end_epoch)
        if expected_last_eval is None:
            expected_reference_count = 0
        else:
            expected_reference_count = min(expected_last_eval, reference_end) // interval + 1
        reference_count = payload.get("reference_eval_count")
        if type(reference_count) is not int or reference_count != expected_reference_count:
            raise ValueError(
                "Checkpoint fixed-BC guard reference count mismatch: "
                f"checkpoint={reference_count!r}, expected={expected_reference_count}."
            )

        reference_min = payload.get("reference_min_mu_mse")
        reference_min_iteration = payload.get("reference_min_iteration")
        if expected_reference_count == 0:
            if reference_min is not None or reference_min_iteration is not None:
                raise ValueError(
                    "Checkpoint fixed-BC guard has reference values before any expected evaluation."
                )
        else:
            if (
                isinstance(reference_min, bool)
                or not isinstance(reference_min, numbers.Real)
                or not math.isfinite(float(reference_min))
                or float(reference_min) < 0.0
            ):
                raise ValueError(
                    "Checkpoint fixed-BC guard reference minimum must be finite and non-negative."
                )
            if (
                type(reference_min_iteration) is not int
                or reference_min_iteration < 0
                or reference_min_iteration > min(expected_last_eval, reference_end)
                or reference_min_iteration % interval != 0
            ):
                raise ValueError(
                    "Checkpoint fixed-BC guard reference-minimum iteration is invalid."
                )

        threshold = payload.get("threshold_mu_mse")
        reference_complete = expected_last_eval is not None and expected_last_eval >= reference_end
        if reference_complete:
            assert reference_min is not None
            expected_threshold = min(
                float(reference_min) * float(self.fixed_bc_guard_max_reference_ratio),
                float(self.fixed_bc_guard_absolute_max_mu_mse),
            )
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, numbers.Real)
                or not math.isfinite(float(threshold))
                or not math.isclose(
                    float(threshold),
                    expected_threshold,
                    rel_tol=1.0e-15,
                    abs_tol=0.0,
                )
            ):
                raise ValueError(
                    "Checkpoint fixed-BC guard frozen threshold is missing or inconsistent."
                )
        elif threshold is not None:
            raise ValueError(
                "Checkpoint fixed-BC guard froze a threshold before the reference period completed."
            )

        counter = payload.get("consecutive_exceedances")
        required_consecutive = int(self.fixed_bc_guard_consecutive_evals)
        if type(counter) is not int or counter < 0:
            raise ValueError(
                "Checkpoint fixed-BC guard consecutive-exceedance counter is invalid."
            )
        if tripped:
            if counter != required_consecutive:
                raise ValueError(
                    "Diagnostic fixed-BC guard checkpoint must trip at exactly the configured "
                    "consecutive-exceedance count."
                )
        elif counter >= required_consecutive:
            raise ValueError(
                "Checkpoint fixed-BC guard consecutive-exceedance counter is invalid."
            )
        if (
            expected_last_eval is None
            or expected_last_eval < int(self.fixed_bc_guard_start_epoch)
        ) and counter != 0:
            raise ValueError(
                "Checkpoint fixed-BC guard has exceedances before its configured start iteration."
            )
        trip_iteration = payload.get("trip_iteration")
        trip_mu_mse = payload.get("trip_mu_mse")
        if tripped:
            if (
                expected_last_eval is None
                or type(trip_iteration) is not int
                or trip_iteration != expected_last_eval
                or trip_iteration < int(self.fixed_bc_guard_start_epoch)
            ):
                raise ValueError(
                    "Diagnostic fixed-BC guard checkpoint has invalid trip iteration metadata."
                )
            if (
                isinstance(trip_mu_mse, bool)
                or not isinstance(trip_mu_mse, numbers.Real)
                or not math.isfinite(float(trip_mu_mse))
                or float(trip_mu_mse) < 0.0
            ):
                raise ValueError(
                    "Diagnostic fixed-BC guard checkpoint trip MSE must be finite and non-negative."
                )
        elif trip_iteration is not None or trip_mu_mse is not None:
            raise ValueError(
                "Non-tripped checkpoint fixed-BC guard state contains trip metadata."
            )
        last_mu_mse = payload.get("last_mu_mse")
        if expected_last_eval is None:
            if last_mu_mse is not None:
                raise ValueError(
                    "Checkpoint fixed-BC guard has a last value before its first evaluation."
                )
        elif (
            isinstance(last_mu_mse, bool)
            or not isinstance(last_mu_mse, numbers.Real)
            or not math.isfinite(float(last_mu_mse))
            or float(last_mu_mse) < 0.0
        ):
            raise ValueError(
                "Checkpoint fixed-BC guard last mean-action MSE must be finite and non-negative."
            )
        if tripped:
            assert last_mu_mse is not None and trip_mu_mse is not None and threshold is not None
            if float(trip_mu_mse) != float(last_mu_mse):
                raise ValueError(
                    "Diagnostic fixed-BC guard checkpoint trip MSE must equal its last evaluation."
                )
            if float(trip_mu_mse) <= float(threshold):
                raise ValueError(
                    "Diagnostic fixed-BC guard checkpoint trip MSE must exceed its frozen threshold."
                )

        stored_by_rank = payload.get("local_dataset_digest_by_rank")
        stored_global = payload.get("global_dataset_digest")
        if expected_last_eval is None:
            if stored_by_rank is not None or stored_global is not None:
                raise ValueError(
                    "Checkpoint fixed-BC guard has dataset digests before its first evaluation."
                )
        elif stored_by_rank != dict(digest_by_rank) or stored_global != global_digest:
            raise ValueError(
                "Checkpoint fixed-BC guard dataset content digest does not authenticate the "
                "serialized fixed strata."
            )
        restored = dict(payload)
        if stored_by_rank is not None:
            restored["local_dataset_digest_by_rank"] = dict(stored_by_rank)
        return restored

    def _prepare_fixed_bc_guard_checkpoint_state(
        self,
        loaded_dict: Mapping[str, Any],
        *,
        next_iteration: int,
        fixed_bc_plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = loaded_dict.get("fixed_bc_guard_state")
        if not bool(getattr(self, "fixed_bc_guard_enabled", False)):
            if payload is not None:
                raise ValueError(
                    "Checkpoint enables fixed-BC guard state while the active full-resume "
                    "configuration disables it."
                )
            return {"action": "noop"}
        if not isinstance(payload, Mapping):
            raise RuntimeError(
                "Fixed-BC guard is enabled but the full-resume checkpoint has no "
                "fixed_bc_guard_state."
            )
        if next_iteration > 0 and fixed_bc_plan.get("action") != "restore":
            raise RuntimeError(
                "Fixed-BC guard full resume requires exact restoration of the frozen dataset; "
                f"fixed-BC plan={fixed_bc_plan.get('action')!r}."
            )
        states_by_rank = loaded_dict.get("fixed_bc_eval_by_rank")
        if not isinstance(states_by_rank, Mapping):
            raise RuntimeError(
                "Fixed-BC guard full resume requires fixed_bc_eval_by_rank."
            )
        if next_iteration == 0:
            digest_by_rank: dict[str, str] = {}
            global_digest = ""
        else:
            digest_by_rank, global_digest = (
                self._fixed_bc_guard_checkpoint_dataset_digests(states_by_rank)
            )
        state = self._validate_fixed_bc_guard_checkpoint_state(
            payload,
            next_iteration=next_iteration,
            digest_by_rank=digest_by_rank,
            global_digest=global_digest,
        )
        return {"action": "restore", "state": state}

    def _commit_fixed_bc_guard_checkpoint_plan(self, plan: Mapping[str, Any]) -> None:
        action = plan.get("action")
        if action == "noop":
            return
        if action != "restore" or not isinstance(plan.get("state"), Mapping):
            raise RuntimeError(f"Unknown fixed-BC guard checkpoint plan: {plan!r}.")
        state = dict(plan["state"])
        if state.get("local_dataset_digest_by_rank") is not None:
            state["local_dataset_digest_by_rank"] = dict(
                state["local_dataset_digest_by_rank"]
            )
        self._fixed_bc_guard_state = state

    def _collect_distributed_fixed_bc_eval_states(self) -> dict[str, dict[str, Any]]:
        """Collect the fixed diagnostic strata so resume preserves its metric."""
        if not bool(getattr(self, "dagger_enabled", False)):
            sample_budget = int(getattr(self, "fixed_bc_eval_num_samples", 0) or 0)
            if sample_budget != 0:
                raise RuntimeError(
                    "Non-DAgger training must have an effective fixed BC evaluation budget of zero; "
                    f"got {sample_budget}."
                )
            return {}
        distributed = (
            getattr(self, "is_multi_gpu", False)
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if not distributed:
            sample_budget = int(getattr(self, "fixed_bc_eval_num_samples", 0) or 0)
            if sample_budget < 0:
                raise ValueError(
                    f"fixed_bc_eval_num_samples must be non-negative, got {sample_budget}."
                )
            if sample_budget == 0:
                return {}
            local_state = self._local_fixed_bc_eval_checkpoint_state()
            return (
                {str(getattr(self, "gpu_global_rank", 0)): local_state}
                if local_state is not None
                else {}
            )

        gloo_group = self._setup_gloo_barrier_group()
        if gloo_group is None:
            raise RuntimeError("Distributed fixed BC checkpointing requires an all-rank Gloo group.")
        group_world_size = torch.distributed.get_world_size(group=gloo_group)
        rank = torch.distributed.get_rank(group=gloo_group)
        local_contract = None
        try:
            sample_budget = int(getattr(self, "fixed_bc_eval_num_samples", 0) or 0)
            if sample_budget < 0:
                raise ValueError(
                    f"fixed_bc_eval_num_samples must be non-negative, got {sample_budget}."
                )
            local_contract = self._fixed_bc_eval_runtime_contract()
            local_state = (
                self._local_fixed_bc_eval_checkpoint_state()
                if sample_budget > 0
                else None
            )
            local_error = None
        except Exception as exc:  # synchronize rank-local validation failures before raising
            local_state = None
            local_error = f"{type(exc).__name__}: {exc}"

        gathered_results: list[dict[str, Any] | None] = [None] * group_world_size
        torch.distributed.all_gather_object(
            gathered_results,
            {
                "rank": rank,
                "error": local_error,
                "state": local_state,
                "contract": local_contract,
            },
            group=gloo_group,
        )

        envelope_problems = []
        failures = []
        for expected_rank, result in enumerate(gathered_results):
            if not isinstance(result, dict) or result.get("rank") != expected_rank:
                envelope_problems.append(
                    f"slot={expected_rank}: malformed payload/reported_rank="
                    f"{result.get('rank') if isinstance(result, dict) else None}"
                )
            elif result.get("error") is not None:
                failures.append(f"rank={expected_rank}: {result['error']}")
        if envelope_problems:
            raise RuntimeError(
                "Distributed fixed BC checkpoint save returned malformed all-rank envelopes: "
                + "; ".join(envelope_problems)
            )
        if failures:
            raise RuntimeError(
                "Refusing to checkpoint because fixed BC state validation failed on at "
                f"least one rank: {'; '.join(failures)}."
            )

        results = self._validate_gathered_fixed_bc_runtime_contracts(
            gathered_results,
            group_world_size=group_world_size,
            context="checkpoint save",
        )
        states: dict[str, dict[str, Any]] = {}
        shared_budget = int(results[0]["contract"]["global_sample_budget"])
        for expected_rank, result in enumerate(results):
            state = result.get("state")
            if shared_budget == 0 and state is None:
                continue
            if not isinstance(state, dict):
                failures.append(
                    f"rank={expected_rank}: expected a fixed BC state mapping for enabled collection, "
                    f"got {type(state).__name__}"
                )
                continue
            states[str(expected_rank)] = state

        if failures:
            raise RuntimeError(
                "Refusing to checkpoint because fixed BC state validation failed on at "
                f"least one rank: {'; '.join(failures)}."
            )
        if shared_budget == 0:
            return {}
        if len(states) != group_world_size:
            raise RuntimeError(
                "Distributed fixed BC checkpoint collection did not return every rank: "
                f"expected={group_world_size}, actual={len(states)}."
            )
        return states

    def _collect_distributed_dagger_replay_states(
        self,
        *,
        fixed_bc_eval_by_rank: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Authenticate and collect every bounded rank-local replay shard."""

        if not bool(getattr(self, "dagger_replay_enabled", False)):
            if getattr(self, "_dagger_replay_buffer", None) is not None:
                raise RuntimeError(
                    "Disabled DAgger replay unexpectedly owns a live buffer."
                )
            return {}
        buffer = getattr(self, "_dagger_replay_buffer", None)
        if not isinstance(buffer, _DaggerReplayBuffer):
            raise RuntimeError("Enabled DAgger replay has no rank-local buffer.")
        rank = int(getattr(self, "gpu_global_rank", 0) or 0)
        local_state: dict[str, Any] | None = None
        local_error: str | None = None
        try:
            local_state = buffer.state_dict()
            verifier = self._new_dagger_replay_buffer()
            verifier.load_state_dict(local_state)
            if verifier.fixed_bc_global_dataset_digest is not None:
                _digests, checkpoint_fixed_digest = (
                    self._fixed_bc_guard_checkpoint_dataset_digests(
                        fixed_bc_eval_by_rank
                    )
                )
                if verifier.fixed_bc_global_dataset_digest != checkpoint_fixed_digest:
                    raise ValueError(
                        "DAgger replay capture boundary disagrees with the fixed-BC dataset "
                        "being published in the same checkpoint."
                    )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"

        distributed = (
            getattr(self, "is_multi_gpu", False)
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if not distributed:
            if local_error is not None or local_state is None:
                raise RuntimeError(
                    "Refusing to checkpoint invalid DAgger replay state: "
                    f"{local_error or 'missing state'}."
                )
            return {str(rank): local_state}

        group = self._setup_gloo_barrier_group()
        if group is None:
            raise RuntimeError(
                "Distributed DAgger replay checkpointing requires an all-rank Gloo group."
            )
        group_world_size = torch.distributed.get_world_size(group=group)
        group_rank = torch.distributed.get_rank(group=group)
        gathered: list[dict[str, Any] | None] = [None] * group_world_size
        torch.distributed.all_gather_object(
            gathered,
            {"rank": group_rank, "error": local_error, "state": local_state},
            group=group,
        )
        problems: list[str] = []
        states: dict[str, dict[str, Any]] = {}
        for expected_rank, result in enumerate(gathered):
            if not isinstance(result, dict) or result.get("rank") != expected_rank:
                problems.append(f"slot={expected_rank}: malformed replay envelope")
                continue
            if result.get("error") is not None:
                problems.append(f"rank={expected_rank}: {result['error']}")
                continue
            state = result.get("state")
            if not isinstance(state, dict):
                problems.append(
                    f"rank={expected_rank}: state_type={type(state).__name__}"
                )
                continue
            states[str(expected_rank)] = state
        if problems or len(states) != group_world_size:
            raise RuntimeError(
                "Refusing to checkpoint invalid distributed DAgger replay state: "
                + "; ".join(problems)
            )
        return states

    def _prepare_dagger_replay_checkpoint_state(
        self,
        loaded_dict: Mapping[str, Any],
        *,
        next_iteration: int,
    ) -> dict[str, Any]:
        states_by_rank = loaded_dict.get(_DAGGER_REPLAY_STATE_KEY)
        if not bool(getattr(self, "dagger_replay_enabled", False)):
            if states_by_rank is not None and not (
                isinstance(states_by_rank, Mapping) and len(states_by_rank) == 0
            ):
                raise ValueError(
                    "Checkpoint contains DAgger replay state while replay is disabled in the "
                    "active full-resume configuration."
                )
            return {"action": "clear"}
        world_size = int(getattr(self, "gpu_world_size", 1) or 1)
        rank = int(getattr(self, "gpu_global_rank", 0) or 0)
        expected_keys = {str(index) for index in range(world_size)}
        if not isinstance(states_by_rank, Mapping) or set(states_by_rank) != expected_keys:
            actual = (
                sorted(str(key) for key in states_by_rank)
                if isinstance(states_by_rank, Mapping)
                else []
            )
            raise ValueError(
                "Exact DAgger replay resume requires one authenticated state per active rank: "
                f"checkpoint={actual}, expected={sorted(expected_keys)}."
            )
        restored = self._new_dagger_replay_buffer()
        restored.load_state_dict(states_by_rank[str(rank)])
        if (
            restored.capture_start_iteration is not None
            and restored.capture_start_iteration >= int(next_iteration)
        ):
            raise ValueError(
                "DAgger replay capture boundary must precede the next checkpoint iteration: "
                f"capture={restored.capture_start_iteration}, next={next_iteration}."
            )
        if restored.fixed_bc_global_dataset_digest is not None:
            fixed_states = loaded_dict.get("fixed_bc_eval_by_rank")
            _digests, fixed_digest = self._fixed_bc_guard_checkpoint_dataset_digests(
                fixed_states
            )
            if restored.fixed_bc_global_dataset_digest != fixed_digest:
                raise ValueError(
                    "DAgger replay checkpoint references a different fixed-BC dataset digest."
                )
        return {"action": "restore", "buffer": restored}

    def _prepare_dagger_replay_checkpoint_state_all_ranks(
        self,
        loaded_dict: Mapping[str, Any],
        *,
        next_iteration: int,
    ) -> dict[str, Any]:
        plan: dict[str, Any] | None = None
        local_error: Exception | None = None
        try:
            plan = self._prepare_dagger_replay_checkpoint_state(
                loaded_dict,
                next_iteration=next_iteration,
            )
        except Exception as exc:
            local_error = exc
        self._synchronize_full_resume_validation_error(
            local_error,
            phase="DAgger replay continuation validation",
        )
        assert plan is not None
        return plan

    def _commit_dagger_replay_checkpoint_plan(self, plan: Mapping[str, Any]) -> None:
        action = plan.get("action")
        if action == "clear":
            self._reset_dagger_replay_state()
            return
        buffer = plan.get("buffer")
        if action != "restore" or not isinstance(buffer, _DaggerReplayBuffer):
            raise RuntimeError(f"Unknown DAgger replay checkpoint plan: {plan!r}.")
        self._dagger_replay_buffer = buffer
        self._dagger_replay_fixed_eval_ready_at_rollout_start = False
        # A checkpoint may be taken before the immutable fixed-BC gate has
        # filled.  Such a replay state is valid but deliberately unbound; it
        # must still run the live all-rank digest/bind path once the gate is
        # ready.  Only a state carrying both authenticated boundary fields may
        # skip that path after exact resume.
        self._dagger_replay_fixed_boundary_verified_runtime = bool(
            buffer.capture_start_iteration is not None
            and buffer.fixed_bc_global_dataset_digest is not None
        )

    def _fixed_bc_eval_allocation_contract(self) -> dict[str, int | str]:
        return {
            "allocation_version": _FIXED_BC_EVAL_ALLOCATION_VERSION,
            "allocation_scheme": _FIXED_BC_EVAL_ALLOCATION_SCHEME,
            "global_sample_budget": int(getattr(self, "fixed_bc_eval_num_samples", 0) or 0),
            "world_size": int(getattr(self, "gpu_world_size", 1) or 1),
            "rank": int(getattr(self, "gpu_global_rank", 0) or 0),
            "local_target": self._fixed_bc_eval_local_target(),
        }

    def _fixed_bc_eval_runtime_contract(self) -> dict[str, int | str | bool]:
        return {
            **self._fixed_bc_eval_allocation_contract(),
            "dagger_enabled": bool(getattr(self, "dagger_enabled", False)),
            "fixed_bc_guard_enabled": bool(
                getattr(self, "fixed_bc_guard_enabled", False)
            ),
        }

    @staticmethod
    def _validate_gathered_fixed_bc_runtime_contracts(
        gathered_results: list[dict[str, Any] | None],
        *,
        group_world_size: int,
        context: str,
    ) -> list[dict[str, Any]]:
        """Validate rank identity and shared fixed-BC runtime semantics."""
        problems: list[str] = []
        results: list[dict[str, Any]] = []
        for expected_rank, result in enumerate(gathered_results):
            if not isinstance(result, dict) or type(result.get("rank")) is not int:
                problems.append(
                    f"slot={expected_rank}: payload/rank is malformed "
                    f"(payload_type={type(result).__name__})"
                )
                continue
            if result["rank"] != expected_rank:
                problems.append(
                    f"slot={expected_rank}: reported_rank={result['rank']}"
                )
                continue
            contract = result.get("contract")
            if not isinstance(contract, dict):
                problems.append(
                    f"rank={expected_rank}: contract_type={type(contract).__name__}"
                )
                continue
            results.append(result)

        if problems or len(results) != group_world_size:
            raise RuntimeError(
                f"Distributed fixed BC {context} returned malformed all-rank contracts: "
                + "; ".join(problems)
            )

        first = results[0]["contract"]
        shared_schema = {
            "allocation_version": int,
            "allocation_scheme": str,
            "global_sample_budget": int,
            "world_size": int,
            "dagger_enabled": bool,
            "fixed_bc_guard_enabled": bool,
        }
        for key, expected_type in shared_schema.items():
            if type(first.get(key)) is not expected_type:
                problems.append(
                    f"rank=0: {key} must be {expected_type.__name__}, got {first.get(key)!r}"
                )
        if problems:
            raise RuntimeError(
                f"Distributed fixed BC {context} has an invalid runtime contract: "
                + "; ".join(problems)
            )
        if first["global_sample_budget"] < 0:
            problems.append(
                f"rank=0: global_sample_budget={first['global_sample_budget']} is negative"
            )
        if first["world_size"] != group_world_size:
            problems.append(
                f"rank=0: configured world_size={first['world_size']} != group {group_world_size}"
            )

        quotient, remainder = divmod(first["global_sample_budget"], group_world_size)
        shared_keys = tuple(shared_schema)
        for expected_rank, result in enumerate(results):
            contract = result["contract"]
            for key in shared_keys:
                if type(contract.get(key)) is not shared_schema[key] or contract.get(key) != first[key]:
                    problems.append(
                        f"rank={expected_rank}: {key}={contract.get(key)!r} != rank0 {first[key]!r}"
                    )
            if type(contract.get("rank")) is not int or contract.get("rank") != expected_rank:
                problems.append(
                    f"rank={expected_rank}: allocation rank={contract.get('rank')!r}"
                )
            expected_target = quotient + int(expected_rank < remainder)
            if type(contract.get("local_target")) is not int or contract.get("local_target") != expected_target:
                problems.append(
                    f"rank={expected_rank}: local_target={contract.get('local_target')!r} "
                    f"!= expected {expected_target}"
                )

        if problems:
            raise RuntimeError(
                f"Distributed fixed BC {context} runtime contracts disagree: "
                + "; ".join(problems)
            )
        return results

    def _clear_fixed_bc_eval_state(self) -> None:
        self._fixed_bc_eval_dataset = {}
        self._fixed_bc_eval_size = 0
        self._fixed_bc_eval_actor_obs_parts.clear()
        self._fixed_bc_eval_teacher_actions_parts.clear()
        self._fixed_bc_eval_actor_perception_parts.clear()
        self._fixed_bc_eval_ready = self._fixed_bc_eval_local_target() == 0

    def _fixed_bc_reset_plan_or_raise(self, message: str) -> dict[str, Any]:
        if not self._fixed_bc_reset_on_resume_enabled():
            raise RuntimeError(
                message
                + " Set HOLOSOMA_ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME=1 only to explicitly "
                "start a new, non-comparable diagnostic series."
            )
        return {"action": "reset", "message": message}

    @staticmethod
    def _fixed_bc_reset_on_resume_enabled() -> bool:
        name = "HOLOSOMA_ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME"
        raw_value = os.environ.get(name)
        if raw_value is None or raw_value.strip() in {"", "0"}:
            return False
        if raw_value.strip() == "1":
            return True
        raise ValueError(f"{name} must be exactly 0 or 1, got {raw_value!r}.")

    def _fixed_bc_checkpoint_dtype(self) -> torch.dtype:
        for parameter in self.actor.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
        return torch.get_default_dtype()

    def _prepare_fixed_bc_eval_checkpoint_state(
        self,
        loaded_dict: dict,
        *,
        next_iteration: int,
    ) -> dict[str, Any]:
        """Validate this rank's fixed diagnostic stratum without mutation."""

        if int(getattr(self, "fixed_bc_eval_num_samples", 0) or 0) <= 0:
            return {"action": "noop"}
        states_by_rank = loaded_dict.get("fixed_bc_eval_by_rank")
        rank = int(getattr(self, "gpu_global_rank", 0))
        world_size = int(getattr(self, "gpu_world_size", 1) or 1)
        if not isinstance(states_by_rank, dict):
            if next_iteration > 0 and getattr(self, "dagger_enabled", False):
                return self._fixed_bc_reset_plan_or_raise(
                    "Training-resume checkpoint does not contain the fixed BC evaluation stratum "
                    "map fixed_bc_eval_by_rank; "
                    "recapturing it would make the metric discontinuous."
                )
            return {"action": "noop"}
        noncanonical_keys = [key for key in states_by_rank if type(key) is not str]
        expected_rank_keys = {str(index) for index in range(world_size)}
        if noncanonical_keys or set(states_by_rank) != expected_rank_keys:
            return self._fixed_bc_reset_plan_or_raise(
                "Training-resume checkpoint fixed BC rank map does not match the active world: "
                f"noncanonical_keys={noncanonical_keys!r}, "
                f"checkpoint={sorted(str(key) for key in states_by_rank)}, "
                f"current={sorted(expected_rank_keys)}."
            )
        state = states_by_rank[str(rank)]
        if not isinstance(state, dict):
            raise ValueError(f"fixed_bc_eval_by_rank[{rank}] must be a tensor dictionary.")

        expected_allocation = self._fixed_bc_eval_allocation_contract()
        allocation_fields = tuple(expected_allocation)
        missing_allocation_fields = [field for field in allocation_fields if field not in state]
        if missing_allocation_fields:
            return self._fixed_bc_reset_plan_or_raise(
                "Training-resume checkpoint fixed BC evaluation state predates the exact allocation contract "
                f"for global rank {rank}; missing={missing_allocation_fields}."
            )
        allocation_mismatches = [
            f"{field}: checkpoint={state[field]!r} current={expected!r}"
            for field, expected in expected_allocation.items()
            if type(state[field]) is not type(expected) or state[field] != expected
        ]
        if allocation_mismatches:
            return self._fixed_bc_reset_plan_or_raise(
                "Training-resume checkpoint fixed BC evaluation allocation does not match the active run for "
                f"global rank {rank}: " + "; ".join(allocation_mismatches)
            )

        raw_ready = state.get("ready")
        if type(raw_ready) is not bool:
            raise ValueError(f"fixed_bc_eval_by_rank[{rank}].ready must be a boolean.")
        ready = raw_ready
        declared_size = state.get("size")
        if type(declared_size) is not int:
            raise ValueError(
                f"fixed_bc_eval_by_rank[{rank}].size must be an integer, got {declared_size!r}."
            )
        local_target = int(expected_allocation["local_target"])
        if declared_size < 0 or declared_size > local_target:
            raise ValueError(
                f"fixed_bc_eval_by_rank[{rank}] size must be in [0, {local_target}], got {declared_size}."
            )
        if ready and declared_size != local_target:
            raise ValueError(
                f"fixed_bc_eval_by_rank[{rank}] is ready but size {declared_size} != local target {local_target}."
            )
        if not ready and declared_size >= local_target:
            raise ValueError(
                f"fixed_bc_eval_by_rank[{rank}] is partial but size {declared_size} is not below "
                f"local target {local_target}."
            )
        if declared_size == 0:
            if ready != bool(local_target == 0):
                raise ValueError(
                    f"fixed_bc_eval_by_rank[{rank}] has inconsistent zero-sized readiness for target {local_target}."
                )
            return {"action": "clear"}

        required = {"actor_obs_raw", "teacher_actions"}
        if getattr(self, "actor_perception_key", ""):
            required.add("actor_perception")
        missing = required - set(state)
        if missing:
            raise ValueError(f"fixed_bc_eval_by_rank[{rank}] is missing tensors: {sorted(missing)}")
        dataset: dict[str, torch.Tensor] = {}
        expected_dtype = self._fixed_bc_checkpoint_dtype()
        for key in required:
            value = state[key]
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                raise ValueError(
                    f"fixed_bc_eval_by_rank[{rank}][{key!r}] must be a dense rank-2 tensor."
                )
            if value.layout != torch.strided:
                raise ValueError(
                    f"fixed_bc_eval_by_rank[{rank}][{key!r}] must use strided layout."
                )
            if value.dtype != expected_dtype:
                raise ValueError(
                    f"fixed_bc_eval_by_rank[{rank}][{key!r}] dtype {value.dtype} != runtime {expected_dtype}."
                )
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"fixed_bc_eval_by_rank[{rank}][{key!r}] contains NaN or infinity.")
            dataset[key] = value.detach().cpu().clone()
        row_counts = {int(value.shape[0]) for value in dataset.values()}
        if len(row_counts) != 1 or not row_counts or next(iter(row_counts)) <= 0:
            raise ValueError(f"fixed_bc_eval_by_rank[{rank}] tensors have inconsistent/empty batches.")
        restored_size = next(iter(row_counts))
        if declared_size >= 0 and declared_size != restored_size:
            raise ValueError(
                f"fixed_bc_eval_by_rank[{rank}] declared size {declared_size} != tensor rows {restored_size}."
            )
        expected_widths = {
            "actor_obs_raw": self._get_obs_dim(self.actor_obs_keys),
            "teacher_actions": int(self.num_act),
        }
        if getattr(self, "actor_perception_key", ""):
            expected_widths["actor_perception"] = int(self.algo_obs_dim_dict[self.actor_perception_key])
        for key, expected_width in expected_widths.items():
            if dataset[key].shape[-1] != expected_width:
                raise ValueError(
                    f"fixed_bc_eval_by_rank[{rank}][{key!r}] width {dataset[key].shape[-1]} "
                    f"!= runtime width {expected_width}."
                )

        return {
            "action": "restore",
            "ready": ready,
            "size": restored_size,
            "dataset": dataset,
        }

    def _commit_fixed_bc_eval_checkpoint_plan(self, plan: dict[str, Any]) -> None:
        action = plan["action"]
        if action == "noop":
            return
        if action in {"reset", "clear"}:
            if action == "reset":
                logger.warning(
                    "{} HOLOSOMA_ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME=1 is set; "
                    "all ranks will recapture the diagnostic.",
                    plan.get("message", "Fixed BC checkpoint reset requested."),
                )
            self._clear_fixed_bc_eval_state()
            return
        if action != "restore":
            raise RuntimeError(f"Unknown fixed BC checkpoint plan action: {action!r}.")
        dataset = plan["dataset"]
        self._fixed_bc_eval_actor_obs_parts.clear()
        self._fixed_bc_eval_teacher_actions_parts.clear()
        self._fixed_bc_eval_actor_perception_parts.clear()
        self._fixed_bc_eval_size = int(plan["size"])
        self._fixed_bc_eval_ready = bool(plan["ready"])
        if self._fixed_bc_eval_ready:
            self._fixed_bc_eval_dataset = dataset
        else:
            self._fixed_bc_eval_dataset = {}
            self._fixed_bc_eval_actor_obs_parts.append(dataset["actor_obs_raw"])
            self._fixed_bc_eval_teacher_actions_parts.append(dataset["teacher_actions"])
            if "actor_perception" in dataset:
                self._fixed_bc_eval_actor_perception_parts.append(dataset["actor_perception"])

    def _prepare_fixed_bc_eval_checkpoint_state_all_ranks(
        self,
        loaded_dict: dict,
        *,
        next_iteration: int,
    ) -> dict[str, Any]:
        plan: dict[str, Any] | None = None
        local_error: Exception | None = None
        try:
            plan = self._prepare_fixed_bc_eval_checkpoint_state(
                loaded_dict,
                next_iteration=next_iteration,
            )
        except Exception as exc:  # synchronize the failure before re-raising
            if isinstance(exc, (ValueError, RuntimeError)) and self._fixed_bc_reset_on_resume_enabled():
                plan = {
                    "action": "reset",
                    "message": f"Rank-local fixed BC checkpoint validation failed: {exc}",
                }
            else:
                local_error = exc

        if (
            getattr(self, "is_multi_gpu", False)
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            group = self._setup_gloo_barrier_group()
            if group is None:
                raise RuntimeError("Distributed fixed BC resume validation requires an all-rank Gloo group.")
            group_world_size = torch.distributed.get_world_size(group=group)
            group_rank = torch.distributed.get_rank(group=group)
            try:
                local_contract = self._fixed_bc_eval_runtime_contract()
            except Exception as exc:  # carry contract-construction failures through the collective
                local_contract = None
                if local_error is None:
                    local_error = exc
            local_result = {
                "rank": group_rank,
                "error": None if local_error is None else f"{type(local_error).__name__}: {local_error}",
                "action": None if plan is None else plan["action"],
                "message": None if plan is None else plan.get("message"),
                "contract": local_contract,
            }
            gathered_results: list[dict[str, Any] | None] = [None] * group_world_size
            torch.distributed.all_gather_object(gathered_results, local_result, group=group)
            envelope_problems = []
            failures = []
            for expected_rank, result in enumerate(gathered_results):
                if not isinstance(result, dict) or result.get("rank") != expected_rank:
                    envelope_problems.append(
                        f"slot={expected_rank}: malformed payload/reported_rank="
                        f"{result.get('rank') if isinstance(result, dict) else None}"
                    )
                elif result.get("error") is not None:
                    failures.append(f"rank={expected_rank} {result['error']}")
            if envelope_problems:
                raise RuntimeError(
                    "Training-resume fixed BC validation returned malformed all-rank envelopes: "
                    + "; ".join(envelope_problems)
                )
            if failures:
                raise RuntimeError(
                    "Training-resume fixed BC validation failed before model mutation:\n  - "
                    + "\n  - ".join(failures)
                )
            results = self._validate_gathered_fixed_bc_runtime_contracts(
                gathered_results,
                group_world_size=group_world_size,
                context="resume validation",
            )
            reset_results = [
                result for result in results if result.get("action") == "reset"
            ]
            if reset_results:
                return {
                    "action": "reset",
                    "message": "At least one rank requires a fixed BC diagnostic reset: "
                    + "; ".join(
                        f"rank={result['rank']} {result['message']}" for result in reset_results
                    ),
                }
            actions = {result.get("action") for result in results}
            unknown_actions = actions - {"noop", "clear", "restore"}
            if unknown_actions:
                raise RuntimeError(
                    "Training-resume fixed BC validation returned unknown all-rank actions: "
                    f"{sorted(repr(action) for action in unknown_actions)}."
                )
            if "noop" in actions and len(actions) != 1:
                raise RuntimeError(
                    "Training-resume fixed BC validation produced an unsafe mixed all-rank "
                    f"action set {sorted(actions)}; noop cannot mix with restore/clear."
                )
        elif local_error is not None:
            raise local_error
        assert plan is not None
        return plan

    def _restore_fixed_bc_eval_checkpoint_state(self, loaded_dict: dict) -> None:
        """Compatibility wrapper for direct single-rank callers and tests."""

        try:
            plan = self._prepare_fixed_bc_eval_checkpoint_state(
                loaded_dict,
                next_iteration=int(getattr(self, "current_learning_iteration", 0) or 0),
            )
        except (ValueError, RuntimeError) as exc:
            if not self._fixed_bc_reset_on_resume_enabled():
                raise
            plan = {
                "action": "reset",
                "message": f"Fixed BC checkpoint validation failed: {exc}",
            }
        self._commit_fixed_bc_eval_checkpoint_plan(plan)

    def _select_checkpoint_env_state(self, loaded_dict: dict) -> dict | None:
        """Select this process' state, while accepting legacy checkpoints."""
        states_by_rank = loaded_dict.get("env_state_by_rank")
        if isinstance(states_by_rank, dict):
            noncanonical_keys = [key for key in states_by_rank if type(key) is not str]
            expected_keys = {str(rank) for rank in range(int(getattr(self, "gpu_world_size", 1) or 1))}
            if noncanonical_keys or set(states_by_rank) != expected_keys:
                raise KeyError(
                    "Checkpoint rank-local environment state keys must be canonical and complete: "
                    f"noncanonical={noncanonical_keys!r}, checkpoint={list(states_by_rank)}, "
                    f"expected={sorted(expected_keys)}."
                )
            return states_by_rank[str(self.gpu_global_rank)]
        if isinstance(states_by_rank, (list, tuple)):
            if int(getattr(self, "gpu_world_size", 1) or 1) != 1:
                raise ValueError(
                    "Legacy list-form env_state_by_rank is supported only for single-rank resume."
                )
            if len(states_by_rank) != 1:
                raise ValueError(
                    "Single-rank legacy env_state_by_rank must contain exactly one entry, "
                    f"got {len(states_by_rank)}."
                )
            if self.gpu_global_rank >= len(states_by_rank):
                raise KeyError(
                    "Checkpoint rank-local environment state has no entry for "
                    f"global rank {self.gpu_global_rank}; entries={len(states_by_rank)}"
                )
            return states_by_rank[self.gpu_global_rank]
        return loaded_dict.get("env_state")

    def _prepare_checkpoint_env_state(self, loaded_dict: dict) -> dict | None:
        """Validate this rank's environment state before any live-state mutation.

        All ranks exchange their validation outcome through Gloo.  A corrupt
        rank-local shard therefore makes every worker fail at the same boundary
        instead of letting healthy ranks advance to a later collective.
        """

        selected_env_state: dict | None = None
        local_error: Exception | None = None
        try:
            selected_env_state = self._select_checkpoint_env_state(loaded_dict)
            env = self._unwrap_env()
            if not selected_env_state and bool(
                getattr(env, "environment_state_checkpoint_required", False)
            ):
                raise RuntimeError(
                    "Checkpoint has no rank-local environment state, but the active environment "
                    "contains persistent perception calibration/stream state. Exact resume is "
                    "impossible; use a checkpoint containing env_state_by_rank or initialize only "
                    "the policy instead of requesting a full resume."
                )
            if not selected_env_state and self._curriculum_state_required_for_resume():
                message = (
                    "Checkpoint has no environment/curriculum state, but adaptive curriculum is enabled. "
                    "Resuming model and optimizer state with a fresh curriculum changes the AS sampling "
                    "distribution and is not a full resume. Use a checkpoint containing env_state(_by_rank)."
                )
                if not self._env_truthy("HOLOSOMA_ALLOW_FRESH_CURRICULUM_RESUME"):
                    raise RuntimeError(
                        message
                        + " To intentionally reset curriculum state, set "
                        "HOLOSOMA_ALLOW_FRESH_CURRICULUM_RESUME=1."
                    )
                logger.warning(
                    "{} HOLOSOMA_ALLOW_FRESH_CURRICULUM_RESUME=1 is set; continuing with fresh curriculum state.",
                    message,
                )
            if selected_env_state:
                if not isinstance(selected_env_state, dict):
                    raise ValueError("Selected rank-local environment checkpoint state must be a mapping.")
                validate_finite_tree(selected_env_state, path=f"env_state_by_rank[{self.gpu_global_rank}]")
                self._validate_env_state(selected_env_state)
        except Exception as exc:  # synchronize the failure before re-raising
            local_error = exc

        if (
            getattr(self, "is_multi_gpu", False)
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            group = self._setup_gloo_barrier_group()
            if group is None:
                raise RuntimeError("Distributed resume validation requires an all-rank Gloo group.")
            local_message = None
            if local_error is not None:
                local_message = (
                    f"rank={self.gpu_global_rank} {type(local_error).__name__}: {local_error}"
                )
            gathered_errors: list[str | None] = [None] * self.gpu_world_size
            torch.distributed.all_gather_object(
                gathered_errors,
                local_message,
                group=group,
            )
            failures = [message for message in gathered_errors if message is not None]
            if failures:
                raise RuntimeError(
                    "Training-resume environment validation failed before model mutation:\n  - "
                    + "\n  - ".join(failures)
                )
        elif local_error is not None:
            raise local_error
        return selected_env_state

    def _prepare_checkpoint_rng_state(self, loaded_dict: dict) -> dict[str, Any] | None:
        """Validate and select this global rank's process RNG state.

        Legacy checkpoints can be used only through an exact, explicit escape
        hatch.  That path restores model/curriculum state but is intentionally
        labelled non-deterministic because its stochastic lineage is absent.
        """

        selected_state: dict[str, Any] | None = None
        local_error: Exception | None = None
        try:
            states_by_rank = loaded_dict.get("rng_state_by_rank")
            if states_by_rank is None:
                if not self._allow_nondeterministic_rng_resume():
                    raise RuntimeError(
                        "Checkpoint has no rng_state_by_rank, so Python/NumPy/torch stochastic streams "
                        "cannot be continued. Refusing a scientifically incomplete resume. To explicitly "
                        f"accept a non-deterministic legacy resume, set "
                        f"{ALLOW_NONDETERMINISTIC_RNG_RESUME_ENV}=1."
                    )
                logger.warning(
                    "Checkpoint has no rng_state_by_rank. {}=1 is set; model/optimizer/curriculum state "
                    "will resume, but Python/NumPy/torch stochastic streams will not.",
                    ALLOW_NONDETERMINISTIC_RNG_RESUME_ENV,
                )
            else:
                if not isinstance(states_by_rank, dict):
                    raise ValueError("Checkpoint rng_state_by_rank must be a mapping.")
                noncanonical_keys = [key for key in states_by_rank if type(key) is not str]
                world_size = int(getattr(self, "gpu_world_size", 1) or 1)
                expected_keys = {str(rank) for rank in range(world_size)}
                if noncanonical_keys or set(states_by_rank) != expected_keys:
                    raise KeyError(
                        "Checkpoint rank-local RNG state keys must be canonical and complete: "
                        f"noncanonical={noncanonical_keys!r}, checkpoint={list(states_by_rank)}, "
                        f"expected={sorted(expected_keys)}."
                    )
                rank = int(getattr(self, "gpu_global_rank", 0) or 0)
                runtime_cuda_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
                runtime_cuda_device = int(torch.cuda.current_device()) if runtime_cuda_count else None
                selected_state = validate_rng_checkpoint_state(
                    states_by_rank[str(rank)],
                    path=f"rng_state_by_rank[{rank}]",
                    expected_cuda_device_count=runtime_cuda_count,
                    expected_cuda_device_index=runtime_cuda_device,
                    validate_cuda_generators=True,
                )
        except Exception as exc:
            local_error = exc

        if (
            getattr(self, "is_multi_gpu", False)
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            group = self._setup_gloo_barrier_group()
            if group is None:
                raise RuntimeError("Distributed RNG resume validation requires an all-rank Gloo group.")
            group_world_size = torch.distributed.get_world_size(group=group)
            group_rank = torch.distributed.get_rank(group=group)
            local_result = {
                "rank": group_rank,
                "error": None if local_error is None else f"{type(local_error).__name__}: {local_error}",
                "has_state": selected_state is not None,
            }
            gathered_results: list[dict[str, Any] | None] = [None] * group_world_size
            torch.distributed.all_gather_object(gathered_results, local_result, group=group)
            envelope_problems = []
            failures = []
            modes = set()
            for expected_rank, result in enumerate(gathered_results):
                if not isinstance(result, dict) or result.get("rank") != expected_rank:
                    envelope_problems.append(
                        f"slot={expected_rank}: malformed payload/reported_rank="
                        f"{result.get('rank') if isinstance(result, dict) else None}"
                    )
                    continue
                if result.get("error") is not None:
                    failures.append(f"rank={expected_rank}: {result['error']}")
                modes.add(bool(result.get("has_state")))
            if envelope_problems:
                raise RuntimeError(
                    "Training-resume RNG validation returned malformed all-rank envelopes: "
                    + "; ".join(envelope_problems)
                )
            if failures:
                raise RuntimeError(
                    "Training-resume RNG validation failed before model mutation:\n  - "
                    + "\n  - ".join(failures)
                )
            if len(modes) != 1:
                raise RuntimeError(
                    "Training-resume RNG validation produced mixed deterministic/legacy modes across ranks."
                )
        elif local_error is not None:
            raise local_error
        return selected_state

    def _synchronize_distributed_operation_error(
        self,
        local_error: Exception | None,
        *,
        operation: str,
    ) -> None:
        """Expose one rank-local operation failure to every worker."""

        if (
            getattr(self, "is_multi_gpu", False)
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            group = self._setup_gloo_barrier_group()
            if group is None:
                raise RuntimeError(
                    f"Distributed {operation} requires an all-rank Gloo group."
                )
            group_world_size = torch.distributed.get_world_size(group=group)
            group_rank = torch.distributed.get_rank(group=group)
            local_result = {
                "rank": group_rank,
                "error": (
                    None
                    if local_error is None
                    else f"{type(local_error).__name__}: {local_error}"
                ),
            }
            gathered_results: list[dict[str, Any] | None] = [None] * group_world_size
            torch.distributed.all_gather_object(gathered_results, local_result, group=group)
            envelope_problems = []
            failures = []
            for expected_rank, result in enumerate(gathered_results):
                if not isinstance(result, dict) or result.get("rank") != expected_rank:
                    envelope_problems.append(
                        f"slot={expected_rank}: malformed payload/reported_rank="
                        f"{result.get('rank') if isinstance(result, dict) else None}"
                    )
                elif result.get("error") is not None:
                    failures.append(f"rank={expected_rank}: {result['error']}")
            if envelope_problems:
                raise RuntimeError(
                    f"{operation} returned malformed all-rank envelopes: "
                    + "; ".join(envelope_problems)
                )
            if failures:
                synchronized_error = RuntimeError(
                    f"{operation} failed on at least one rank:\n  - "
                    + "\n  - ".join(failures)
                )
                if local_error is not None:
                    raise synchronized_error from local_error
                raise synchronized_error
        elif local_error is not None:
            raise local_error

    def _synchronize_full_resume_validation_error(
        self,
        local_error: Exception | None,
        *,
        phase: str = "checkpoint deserialization/core validation",
    ) -> None:
        """Make one full-resume phase failure atomic across ranks."""

        self._synchronize_distributed_operation_error(
            local_error,
            operation=f"Training-resume {phase}",
        )

    @staticmethod
    def _checkpoint_normalization_flag(loaded_dict: dict, field: str) -> bool | None:
        try:
            value = loaded_dict["experiment_config"]["algo"]["config"][field]
        except (KeyError, TypeError):
            return None
        return value if isinstance(value, bool) else None

    @staticmethod
    def _validate_normalizer_module_state(
        normalizer: nn.Module,
        state: dict[str, Any],
        *,
        path: str,
    ) -> None:
        """Validate one normalizer payload completely before module mutation."""
        validate_finite_tree(state, path=path)
        validate_module_state_compatibility(
            state,
            reference_state=normalizer.state_dict(),
            path=path,
            allow_legacy_integral_count=isinstance(normalizer, EmpiricalNormalization),
        )
        if not isinstance(normalizer, EmpiricalNormalization):
            return

        count = state.get("count")
        variance = state.get("_var")
        std = state.get("_std")
        if not isinstance(count, torch.Tensor) or count.numel() != 1:
            raise ValueError(f"Checkpoint {path}.count must be one scalar tensor.")
        count_value = float(count.item())
        if not math.isfinite(count_value) or count_value < 0.0:
            raise ValueError(
                f"Checkpoint {path}.count must be finite and non-negative, got {count_value}."
            )
        if not isinstance(variance, torch.Tensor) or bool((variance < 0.0).any().item()):
            raise ValueError(f"Checkpoint {path}._var must be a non-negative tensor.")
        if not isinstance(std, torch.Tensor) or bool((std <= 0.0).any().item()):
            raise ValueError(f"Checkpoint {path}._std must be a positive tensor.")
        if count_value > 0.0:
            expected_std = torch.sqrt(variance + float(normalizer.eps))
            if not torch.allclose(std, expected_std, rtol=1.0e-5, atol=1.0e-7):
                max_error = float(torch.max(torch.abs(std - expected_std)).item())
                raise ValueError(
                    f"Checkpoint {path} has inconsistent _std/_var/eps statistics "
                    f"(max_abs_error={max_error})."
                )

    def _restore_checkpoint_normalizers(
        self,
        loaded_dict: dict,
        *,
        kind: str,
        runtime_enabled: bool,
        normalizers: dict[str, nn.Module],
        operation: str = "Full-resume",
    ) -> None:
        state_dict = self._validate_checkpoint_normalizers(
            loaded_dict,
            kind=kind,
            runtime_enabled=runtime_enabled,
            normalizers=normalizers,
            operation=operation,
        )
        if state_dict is None:
            return

        for key in set(normalizers) & set(state_dict):
            state = state_dict[key]
            if state is None:
                continue
            normalizer = normalizers[key]
            try:
                normalizer.load_state_dict(state, strict=True)
            except RuntimeError as exc:
                raise ValueError(
                    f"Checkpoint {kind}_obs_normalizer_state[{key!r}] is shape/state incompatible "
                    "with the runtime normalizer."
                ) from exc

    def _validate_checkpoint_normalizers(
        self,
        loaded_dict: dict,
        *,
        kind: str,
        runtime_enabled: bool,
        normalizers: dict[str, nn.Module],
        operation: str = "Full-resume",
    ) -> dict | None:
        """Validate normalizer state without mutating the live modules."""

        flag_name = f"normalize_{kind}_obs"
        state_name = f"{kind}_obs_normalizer_state"
        checkpoint_enabled = self._checkpoint_normalization_flag(loaded_dict, flag_name)
        if operation == "Policy-init" and checkpoint_enabled is None:
            raise ValueError(
                f"Policy-init checkpoint experiment_config must declare a boolean {flag_name}; "
                "the actor normalization contract cannot be inferred from tensor shapes."
            )
        if checkpoint_enabled is not None and checkpoint_enabled != runtime_enabled:
            raise ValueError(
                f"{operation} normalization mismatch for {kind}: checkpoint {flag_name}="
                f"{checkpoint_enabled}, runtime={runtime_enabled}."
            )
        normalization_required = runtime_enabled or checkpoint_enabled is True
        state_dict = loaded_dict.get(state_name)
        if normalization_required and not isinstance(state_dict, dict):
            raise ValueError(
                f"Checkpoint requires {kind} observation normalization but {state_name} is missing."
            )
        if not normalization_required:
            # A disabled normalizer is not part of the policy/value function;
            # stale legacy payloads are intentionally ignored rather than
            # loaded into Identity modules.
            return None
        if not isinstance(state_dict, dict):
            return None

        expected_keys = set(normalizers)
        actual_keys = set(state_dict)
        if normalization_required and actual_keys != expected_keys:
            raise ValueError(
                f"Checkpoint {state_name} keys do not match runtime groups: "
                f"missing={sorted(expected_keys - actual_keys)}, extra={sorted(actual_keys - expected_keys)}."
            )
        for key in expected_keys & actual_keys:
            state = state_dict[key]
            if state is None:
                if normalization_required:
                    raise ValueError(f"Checkpoint {state_name}[{key!r}] is empty.")
                continue
            normalizer = normalizers[key]
            if normalization_required and not isinstance(normalizer, EmpiricalNormalization):
                raise ValueError(
                    f"Runtime {kind} normalizer for {key!r} is not empirical despite enabled normalization."
                )
            if not isinstance(state, dict):
                raise ValueError(
                    f"Checkpoint {state_name}[{key!r}] must be a mapping."
                )
            self._validate_normalizer_module_state(
                normalizer,
                state,
                path=f"{state_name}[{key!r}]",
            )
        return state_dict

    def _validate_full_resume_checkpoint(self, loaded_dict: Any) -> tuple[int, float | None, float | None]:
        """Validate every numerical state used by full resume before mutation."""

        if not isinstance(loaded_dict, dict):
            raise ValueError("Training-resume checkpoint payload must be a mapping.")
        completed_iteration, next_iter = validate_checkpoint_iterations(loaded_dict)
        self._validate_terminal_fixed_bc_eval_artifact_payload(
            loaded_dict,
            expected_completed_iteration=completed_iteration,
            compare_runtime_guard_config=True,
        )

        actor_state = require_mapping(loaded_dict, "actor_model_state_dict")
        critic_state = require_mapping(loaded_dict, "critic_model_state_dict")
        validate_finite_tree(actor_state, path="actor_model_state_dict")
        validate_finite_tree(critic_state, path="critic_model_state_dict")
        validate_module_state_compatibility(
            actor_state,
            reference_state=self.actor.state_dict(),
            path="actor_model_state_dict",
        )
        self._validate_checkpoint_actor_std(
            actor_state,
            path="actor_model_state_dict",
        )
        validate_module_state_compatibility(
            critic_state,
            reference_state=self.critic.state_dict(),
            path="critic_model_state_dict",
        )

        self._validate_checkpoint_normalizers(
            loaded_dict,
            kind="actor",
            runtime_enabled=bool(getattr(self.config, "normalize_actor_obs", False)),
            normalizers=self.actor_obs_normalizers,
        )
        self._validate_checkpoint_normalizers(
            loaded_dict,
            kind="critic",
            runtime_enabled=bool(getattr(self.config, "normalize_critic_obs", False)),
            normalizers=self.critic_obs_normalizers,
        )

        actor_lr: float | None = None
        critic_lr: float | None = None
        if self.config.load_optimizer:
            missing_optimizer_keys = [
                key
                for key in ("actor_optimizer_state_dict", "critic_optimizer_state_dict")
                if not isinstance(loaded_dict.get(key), dict)
            ]
            if missing_optimizer_keys:
                raise ValueError(
                    "config.load_optimizer=True but checkpoint is missing optimizer state: "
                    f"{missing_optimizer_keys}."
                )
            actor_optimizer_state = require_mapping(loaded_dict, "actor_optimizer_state_dict")
            critic_optimizer_state = require_mapping(loaded_dict, "critic_optimizer_state_dict")
            actor_lr = validate_optimizer_state(
                actor_optimizer_state,
                path="actor_optimizer_state_dict",
                minimum_lr=self.min_actor_learning_rate,
                maximum_lr=self.max_actor_learning_rate,
            )
            critic_lr = validate_optimizer_state(
                critic_optimizer_state,
                path="critic_optimizer_state_dict",
                minimum_lr=self.min_critic_learning_rate,
                maximum_lr=self.max_critic_learning_rate,
            )
            validate_optimizer_compatibility(
                actor_optimizer_state,
                optimizer=self.actor_optimizer,
                path="actor_optimizer_state_dict",
            )
            validate_optimizer_compatibility(
                critic_optimizer_state,
                optimizer=self.critic_optimizer,
                path="critic_optimizer_state_dict",
            )
        return next_iter, actor_lr, critic_lr

    def load(self, ckpt_path: str | None) -> dict | None:
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict: dict[str, Any] | None = None
            next_iter: int | None = None
            actor_lr: float | None = None
            critic_lr: float | None = None
            live_motion_transition_contract, _live_motion_transition_digest = (
                self._collect_distributed_motion_transition_contract()
            )
            restored_motion_transition_contract: dict[str, Any] | None = None
            restored_motion_transition_digest: str | None = None
            local_error: Exception | None = None
            try:
                training_provenance = getattr(self, "_training_provenance", None)
                legacy_unprovenanced_resume = allow_legacy_unprovenanced_resume()
                current_provenance_present = training_provenance is not None
                if current_provenance_present and not isinstance(training_provenance, dict):
                    raise ValueError(
                        "Attached training provenance must be a mapping when present; the legacy "
                        "override cannot bypass malformed provenance."
                    )
                experiment_config = getattr(self, "_experiment_config", None)
                current_config = (
                    experiment_config.to_serializable_dict()
                    if experiment_config is not None
                    and hasattr(experiment_config, "to_serializable_dict")
                    else None
                )
                if not legacy_unprovenanced_resume:
                    if not isinstance(training_provenance, dict):
                        raise ValueError(
                            "Scientific full resume requires finalized current training provenance. "
                            f"Set {ALLOW_LEGACY_UNPROVENANCED_RESUME_ENV}=1 only for an explicitly "
                            "non-scientific legacy lineage."
                        )
                    if not isinstance(current_config, dict):
                        raise ValueError(
                            "Scientific full resume requires attached experiment_config metadata so "
                            "reward, termination, randomization, observation, and optimizer semantics "
                            "can be compared before mutation."
                        )
                expected_resume_sha256 = None
                if isinstance(training_provenance, dict):
                    if training_provenance.get("training_resume_enabled") is not True:
                        raise ValueError(
                            "PPO.load was called while attached training provenance does not enable "
                            "a full training resume."
                        )
                    expected_resume_sha256 = training_provenance.get("training_resume_sha256")
                    if not isinstance(expected_resume_sha256, str):
                        raise ValueError(
                            "Attached training provenance enables full resume but has no authenticated "
                            "training_resume_sha256."
                        )
                # Re-authenticate a stable no-follow descriptor immediately
                # before loading.  This closes the preflight->simulator->PPO
                # path-reopen window and uses only the weights-only schema.
                # Direct callers without provenance still receive the stable
                # safe-load guarantee, but cannot claim an authenticated
                # scientific lineage.
                loaded_dict, actual_resume_sha256 = load_verified_torch_checkpoint(
                    ckpt_path,
                    expected_sha256=expected_resume_sha256,
                    map_location="cpu",
                )
                checkpoint_provenance_present = (
                    loaded_dict.get("training_provenance") is not None
                )
                if checkpoint_provenance_present or current_provenance_present:
                    if checkpoint_provenance_present != current_provenance_present:
                        raise ValueError(
                            "Training-resume provenance is present on exactly one side; refusing an "
                            "unpaired full resume."
                        )
                    if not isinstance(current_config, dict):
                        raise ValueError(
                            "A provenance-authenticated full resume also requires attached "
                            "experiment_config metadata; the legacy override applies only when "
                            "provenance is absent on both sides."
                        )
                    validate_resume_payload_identity(
                        loaded_dict,
                        current_config,
                        current_provenance=training_provenance,
                        actual_resume_sha256=actual_resume_sha256,
                    )
                elif isinstance(current_config, dict):
                    # This invokes the exact legacy-hatch check and still
                    # compares every scientific config field available in an
                    # otherwise unprovenanced checkpoint.
                    validate_resume_payload_identity(
                        loaded_dict,
                        current_config,
                        current_provenance=None,
                        actual_resume_sha256=actual_resume_sha256,
                    )
                else:
                    # Only the both-unprovenanced, both-unconfigured legacy
                    # lineage may reach this branch.  The strict parser above
                    # has already established that the exact hatch is enabled.
                    logger.warning(
                        "{}=1: allowing an explicitly non-scientific legacy full-resume path "
                        "without provenance or serialized config comparison.",
                        ALLOW_LEGACY_UNPROVENANCED_RESUME_ENV,
                    )
                (
                    restored_motion_transition_contract,
                    restored_motion_transition_digest,
                ) = self._validate_checkpoint_motion_transition_contract(
                    loaded_dict,
                    live_contract=live_motion_transition_contract,
                    compare_live=True,
                    operation="Training-resume",
                )
                next_iter, actor_lr, critic_lr = self._validate_full_resume_checkpoint(
                    loaded_dict
                )
            except Exception as exc:
                local_error = exc
            self._synchronize_full_resume_validation_error(local_error)
            assert loaded_dict is not None and next_iter is not None
            self._motion_transition_contract = restored_motion_transition_contract
            self._motion_transition_contract_sha256 = restored_motion_transition_digest
            logger.info("Checkpoint deserialized on CPU; restoring tensors to {}.", self.device)
            selected_env_state = self._prepare_checkpoint_env_state(loaded_dict)
            fixed_bc_plan = self._prepare_fixed_bc_eval_checkpoint_state_all_ranks(
                loaded_dict,
                next_iteration=next_iter,
            )
            dagger_replay_plan = (
                self._prepare_dagger_replay_checkpoint_state_all_ranks(
                    loaded_dict,
                    next_iteration=next_iter,
                )
            )
            fixed_bc_guard_plan: dict[str, Any] | None = None
            fixed_bc_guard_error: Exception | None = None
            try:
                fixed_bc_guard_plan = self._prepare_fixed_bc_guard_checkpoint_state(
                    loaded_dict,
                    next_iteration=next_iter,
                    fixed_bc_plan=fixed_bc_plan,
                )
            except Exception as exc:
                fixed_bc_guard_error = exc
            self._synchronize_full_resume_validation_error(
                fixed_bc_guard_error,
                phase="fixed-BC guard continuation validation",
            )
            assert fixed_bc_guard_plan is not None
            selected_rng_state = self._prepare_checkpoint_rng_state(loaded_dict)
            rollout_contract_error: Exception | None = None
            try:
                self._validate_rollout_resume_contract(
                    loaded_dict,
                    next_iteration=next_iter,
                )
            except Exception as exc:
                rollout_contract_error = exc
            self._synchronize_full_resume_validation_error(
                rollout_contract_error,
                phase="rollout-resume contract validation",
            )
            optimizer_contract_error: Exception | None = None
            if getattr(self.config, "load_optimizer", True) is not True:
                optimizer_contract_error = ValueError(
                    "Exact training resume requires config.load_optimizer=True. Resetting optimizer "
                    "moments and adaptive learning rates creates a warm-start lineage, not a continuation."
                )
            self._synchronize_full_resume_validation_error(
                optimizer_contract_error,
                phase="optimizer-continuation contract validation",
            )
            commit_error: Exception | None = None
            try:
                self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
                self._sanitize_actor_std()
                self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])
                self._restore_checkpoint_normalizers(
                    loaded_dict,
                    kind="actor",
                    runtime_enabled=bool(getattr(self.config, "normalize_actor_obs", False)),
                    normalizers=self.actor_obs_normalizers,
                )
                self._restore_checkpoint_normalizers(
                    loaded_dict,
                    kind="critic",
                    runtime_enabled=bool(getattr(self.config, "normalize_critic_obs", False)),
                    normalizers=self.critic_obs_normalizers,
                )
                if self.config.load_optimizer:
                    self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
                    self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
                    self._move_optimizer_state_to_device(self.actor_optimizer)
                    self._move_optimizer_state_to_device(self.critic_optimizer)
                    assert actor_lr is not None and critic_lr is not None
                    self.actor_learning_rate = actor_lr
                    self.critic_learning_rate = critic_lr
                    logger.info("Optimizer loaded from checkpoint")
                self.current_learning_iteration = next_iter
                self._commit_fixed_bc_eval_checkpoint_plan(fixed_bc_plan)
                self._commit_dagger_replay_checkpoint_plan(dagger_replay_plan)
                self._commit_fixed_bc_guard_checkpoint_plan(fixed_bc_guard_plan)
                # A validated old final proves its source artifact only.  It
                # must not become sticky state on a target extension.
                self._terminal_fixed_bc_eval_state = None
                env_state = self._move_checkpoint_value_to_device(selected_env_state, self.device)
                self._restore_env_state(env_state)
                self._refresh_distillation_iteration_state(self.current_learning_iteration)
                self._apply_ppo_start_noise_std_cap(self.current_learning_iteration)
            except Exception as exc:
                commit_error = exc
            # A runtime/device failure during the validated commit phase must
            # not let healthy peers enter model synchronization alone.
            self._synchronize_full_resume_validation_error(
                commit_error,
                phase="validated state commit",
            )
            if getattr(self, "is_multi_gpu", False):
                # Resume inputs are provenance-checked, and rank zero remains
                # the explicit model source of truth before rollout.
                self._synchronize_model_weights()
            self._assert_model_parameters_finite(
                phase="full-resume load",
                trainable_only=False,
            )
            if selected_rng_state is not None:
                rng_restore_error: Exception | None = None
                try:
                    restore_rng_checkpoint_state(
                        selected_rng_state,
                        path=f"rng_state_by_rank[{self.gpu_global_rank}]",
                    )
                except Exception as exc:
                    rng_restore_error = exc
                if getattr(self, "is_multi_gpu", False):
                    # Surface a driver/runtime restore failure on every rank.
                    # The outcome collective is observational; restore once
                    # more below so it cannot become the final stochastic
                    # boundary of an otherwise successful resume.
                    self._synchronize_full_resume_validation_error(
                        rng_restore_error,
                        phase="rank-local RNG state restore",
                    )
                    restore_rng_checkpoint_state(
                        selected_rng_state,
                        path=f"rng_state_by_rank[{self.gpu_global_rank}]",
                    )
                elif rng_restore_error is not None:
                    raise rng_restore_error
            return loaded_dict.get("infos")
        return None

    def _live_actor_perception_manager(self) -> PerceptionManager:
        env = self._unwrap_env()
        topology_getter = getattr(env, "_perception_checkpoint_topology", None)
        if callable(topology_getter):
            role_owners, managers_by_owner = topology_getter()
            if not isinstance(role_owners, dict) or not isinstance(managers_by_owner, dict):
                raise RuntimeError("Live perception checkpoint topology is malformed.")
            owner = role_owners.get("actor")
            manager = managers_by_owner.get(owner) if isinstance(owner, str) else None
        else:
            manager = getattr(env, "perception_manager", None)
        if not isinstance(manager, PerceptionManager) or not bool(getattr(manager, "enabled", False)):
            raise RuntimeError(
                "A perception actor requires one enabled live student perception manager."
            )
        return manager

    def _aggregate_actor_perception_geometry_support(
        self,
        states_by_rank: Any,
        *,
        allow_legacy_missing: bool,
        path: str = "env_state_by_rank",
    ) -> dict[str, Any] | None:
        """Extract all-rank student geometry without restoring environment state."""

        if not getattr(self, "actor_perception_key", ""):
            return None
        if not isinstance(states_by_rank, Mapping) or not states_by_rank:
            if allow_legacy_missing:
                return None
            raise ValueError(
                "Perception policy checkpoint publication requires non-empty env_state_by_rank."
            )
        expected_keys = {str(index) for index in range(len(states_by_rank))}
        if any(type(key) is not str for key in states_by_rank) or set(states_by_rank) != expected_keys:
            raise ValueError(
                f"{path} keys must be canonical contiguous global-rank strings; "
                f"expected={sorted(expected_keys)}, actual={sorted(str(key) for key in states_by_rank)}."
            )

        rank_supports: list[dict[str, Any]] = []
        missing_ranks: list[int] = []
        for rank in range(len(states_by_rank)):
            rank_path = f"{path}[{rank}]"
            env_state = states_by_rank[str(rank)]
            if not isinstance(env_state, Mapping):
                raise ValueError(f"{rank_path} must be a mapping.")
            envelope = env_state.get("perception_managers")
            if envelope is None:
                missing_ranks.append(rank)
                continue
            if not isinstance(envelope, Mapping) or set(envelope) != {
                "version",
                "role_owners",
                "states",
            }:
                raise ValueError(f"{rank_path}.perception_managers is malformed.")
            if type(envelope.get("version")) is not int or envelope["version"] != 1:
                raise ValueError(
                    f"{rank_path}.perception_managers.version must equal integer 1."
                )
            role_owners = envelope.get("role_owners")
            states = envelope.get("states")
            if not isinstance(role_owners, Mapping) or set(role_owners) != {
                "actor",
                "teacher",
                "critic",
            }:
                raise ValueError(f"{rank_path}.perception_managers.role_owners is malformed.")
            if not isinstance(states, Mapping):
                raise ValueError(f"{rank_path}.perception_managers.states must be a mapping.")
            actor_owner = role_owners.get("actor")
            if not isinstance(actor_owner, str) or actor_owner not in states:
                raise ValueError(
                    f"{rank_path}.perception_managers has no state for its actor owner."
                )
            manager_state = states[actor_owner]
            if not isinstance(manager_state, Mapping):
                raise ValueError(
                    f"{rank_path}.perception_managers.states[{actor_owner!r}] must be a mapping."
                )
            if type(manager_state.get("version")) is not int or manager_state["version"] != 1:
                raise ValueError(
                    f"{rank_path}.perception_managers.states[{actor_owner!r}].version must equal integer 1."
                )
            rank_supports.append(
                PerceptionManager.geometry_support_from_checkpoint_semantics(
                    manager_state.get("semantics"),
                    path=(
                        f"{rank_path}.perception_managers.states[{actor_owner!r}].semantics"
                    ),
                )
            )

        if missing_ranks:
            if allow_legacy_missing and len(missing_ranks) == len(states_by_rank):
                return None
            raise ValueError(
                "Perception manager checkpoint state is missing on only part of the training world: "
                f"ranks={missing_ranks}."
            )
        return PerceptionManager.aggregate_training_geometry_support(rank_supports)

    def _prepare_evaluation_perception_geometry_support(
        self,
        loaded_dict: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Authenticate checkpoint geometry against the already-setup eval manager."""

        if not getattr(self, "actor_perception_key", ""):
            return None
        derived = self._aggregate_actor_perception_geometry_support(
            loaded_dict.get("env_state_by_rank"),
            allow_legacy_missing=True,
        )
        declared = loaded_dict.get(_ACTOR_PERCEPTION_GEOMETRY_SUPPORT_KEY)
        if derived is None:
            if declared is not None:
                raise ValueError(
                    "Checkpoint declares actor perception training geometry but has no all-rank "
                    "perception manager state from which to authenticate it."
                )
            return None
        if declared is not None:
            normalized_declared = PerceptionManager._normalize_training_geometry_support(
                declared,
                path=_ACTOR_PERCEPTION_GEOMETRY_SUPPORT_KEY,
            )
            if normalized_declared != derived:
                raise ValueError(
                    "Checkpoint actor perception training geometry does not match env_state_by_rank."
                )
        manager = self._live_actor_perception_manager()
        allow_ood_object_geometry = bool(
            getattr(self, "_evaluation_allow_ood_object_geometry", False)
        )
        if allow_ood_object_geometry:
            manager.validate_deployment_geometry_support(
                derived,
                allow_unknown_object_geometry=True,
            )
            live_support = manager.get_local_geometry_support()

            def object_identity(item: Mapping[str, Any]) -> tuple[str, str, int, str]:
                mesh = item["mesh"]
                return (
                    str(item["source_name"]),
                    str(mesh["sha256"]),
                    int(mesh["size_bytes"]),
                    str(mesh["suffix"]),
                )

            training_objects = {
                object_identity(item) for item in derived["object_mesh_support"]
            }
            live_objects = {
                object_identity(item) for item in live_support["object_mesh_support"]
            }
            unknown_objects = sorted(live_objects - training_objects)
            if not unknown_objects:
                raise ValueError(
                    "Explicit OOD object-geometry evaluation was requested, but the "
                    "selected live object is already in the authenticated training support."
                )
            self._evaluation_ood_object_geometry_audit = {
                "version": 1,
                "semantics": (
                    "evaluation_only_checkpoint_actor_with_explicit_"
                    "out_of_training_support_object_geometry"
                ),
                "training_object_mesh_count": len(training_objects),
                "live_object_mesh_count": len(live_objects),
                "unknown_live_objects": [
                    {
                        "source_name": source_name,
                        "mesh": {
                            "sha256": digest,
                            "size_bytes": size_bytes,
                            "suffix": suffix,
                        },
                    }
                    for source_name, digest, size_bytes, suffix in unknown_objects
                ],
                "training_geometry_support": derived,
                "live_geometry_support": live_support,
                "camera_source_exact": True,
                "robot_mesh_bindings_exact": True,
            }
        else:
            manager.validate_deployment_geometry_support(derived)
            self._evaluation_ood_object_geometry_audit = None
        return derived

    def load_policy_init(self, ckpt_path: str | None) -> dict | None:
        """Initialize only actor policy parameters from a checkpoint.

        This intentionally does not restore critic, optimizers, iteration
        counters, or environment state.  When actor observation normalization
        is enabled, its state is part of the policy function and is restored as
        well. The actor state dict must be fully compatible; partial policy
        initialization is explicitly forbidden because it can look like a
        successful warm start while leaving large parts of the policy randomly
        initialized.
        """
        if ckpt_path is None:
            return None
        rng_state = capture_rng_checkpoint_state()
        try:
            (
                self._motion_transition_contract,
                self._motion_transition_contract_sha256,
            ) = self._collect_distributed_motion_transition_contract()
            return self._load_policy_init_impl(ckpt_path)
        finally:
            restore_rng_checkpoint_state(
                rng_state,
                path="pre_policy_init_load_rng_state",
            )

    def load_stage4_init(self, ckpt_path: str | None) -> dict | None:
        """Initialize actor+critic for Stage 4 without continuing training state.

        Actor and critic parameters plus enabled observation normalizers are
        restored.  Optimizers, iteration, RNG, environment/curriculum state,
        DAgger replay, and W&B identity remain fresh.  Policy exploration std
        is reset to the target run's configured ``init_noise_std``.
        """

        if ckpt_path is None:
            return None
        rng_state = capture_rng_checkpoint_state()
        try:
            live_motion_contract, live_motion_digest = (
                self._collect_distributed_motion_transition_contract()
            )
            prepared: tuple[
                dict[str, Any],
                dict[str, Any],
                dict[str, Any],
                bool,
                bool,
                str,
            ] | None = None
            validation_error: Exception | None = None
            try:
                provenance = getattr(self, "_training_provenance", None)
                legacy_unverified = allow_legacy_unverified_policy_load()
                expected_sha256: str | None = None
                if provenance is not None:
                    provenance = validate_training_provenance(
                        provenance,
                        require_finalized=True,
                    )
                    if provenance.get("stage4_init_enabled") is not True:
                        raise ValueError(
                            "PPO.load_stage4_init was called while attached training "
                            "provenance does not enable Stage-4 initialization."
                        )
                    expected_sha256 = provenance["stage4_init_sha256"]
                elif not legacy_unverified:
                    raise ValueError(
                        "Scientific Stage-4 initialization requires finalized current "
                        "training provenance with an authenticated stage4_init_sha256."
                    )
                runtime_config = getattr(self, "_policy_load_runtime_config", None)
                current_config = (
                    runtime_config.to_serializable_dict()
                    if runtime_config is not None
                    and hasattr(runtime_config, "to_serializable_dict")
                    else None
                )
                if not isinstance(current_config, dict):
                    raise ValueError(
                        "Stage-4 initialization requires an attached runtime experiment config."
                    )
                loaded_dict, actual_sha256 = load_verified_torch_checkpoint(
                    ckpt_path,
                    expected_sha256=expected_sha256,
                    map_location="cpu",
                )
                validate_stage4_init_payload_identity(loaded_dict, current_config)
                actor_state = require_mapping(loaded_dict, "actor_model_state_dict")
                critic_state = require_mapping(loaded_dict, "critic_model_state_dict")
                validate_finite_tree(actor_state, path="actor_model_state_dict")
                validate_finite_tree(critic_state, path="critic_model_state_dict")
                validate_module_state_compatibility(
                    actor_state,
                    reference_state=self.actor.state_dict(),
                    path="actor_model_state_dict",
                )
                self._validate_checkpoint_actor_std(
                    actor_state,
                    path="actor_model_state_dict",
                )
                validate_module_state_compatibility(
                    critic_state,
                    reference_state=self.critic.state_dict(),
                    path="critic_model_state_dict",
                )
                actor_norm = bool(getattr(self.config, "normalize_actor_obs", False))
                critic_norm = bool(getattr(self.config, "normalize_critic_obs", False))
                self._validate_checkpoint_normalizers(
                    loaded_dict,
                    kind="actor",
                    runtime_enabled=actor_norm,
                    normalizers=self.actor_obs_normalizers,
                    operation="Stage-4 init",
                )
                self._validate_checkpoint_normalizers(
                    loaded_dict,
                    kind="critic",
                    runtime_enabled=critic_norm,
                    normalizers=self.critic_obs_normalizers,
                    operation="Stage-4 init",
                )
                # Validate source metadata, but retain the target run's live
                # transition contract because this is a fresh lineage.
                self._validate_checkpoint_motion_transition_contract(
                    loaded_dict,
                    live_contract=None,
                    compare_live=False,
                    operation="Stage-4 init",
                    allow_missing_contract=True,
                )
                if self.current_learning_iteration != 0:
                    raise ValueError(
                        "Stage-4 initialization must begin at fresh iteration 0, got "
                        f"{self.current_learning_iteration}."
                    )
                if self.actor_optimizer.state or self.critic_optimizer.state:
                    raise ValueError(
                        "Stage-4 initialization requires fresh empty optimizer state."
                    )
                prepared = (
                    loaded_dict,
                    actor_state,
                    critic_state,
                    actor_norm,
                    critic_norm,
                    actual_sha256,
                )
            except Exception as exc:
                validation_error = exc
            self._synchronize_distributed_operation_error(
                validation_error,
                operation="Stage-4 checkpoint deserialization/validation",
            )
            assert prepared is not None
            (
                loaded_dict,
                actor_state,
                critic_state,
                actor_norm,
                critic_norm,
                actual_sha256,
            ) = prepared

            commit_error: Exception | None = None
            try:
                self.actor.load_state_dict(actor_state, strict=True)
                self.critic.load_state_dict(critic_state, strict=True)
                configured_std = torch.full_like(
                    self.actor.std.data,
                    float(self.config.init_noise_std),
                )
                projected_std = self._project_actor_std_constraints(configured_std)
                if not torch.equal(projected_std, configured_std):
                    raise ValueError(
                        "Stage-4 init_noise_std violates actor noise constraints; "
                        "refusing a silent projection."
                    )
                self.actor.std.data.copy_(configured_std)
                if actor_norm:
                    self._restore_checkpoint_normalizers(
                        loaded_dict,
                        kind="actor",
                        runtime_enabled=True,
                        normalizers=self.actor_obs_normalizers,
                        operation="Stage-4 init",
                    )
                if critic_norm:
                    self._restore_checkpoint_normalizers(
                        loaded_dict,
                        kind="critic",
                        runtime_enabled=True,
                        normalizers=self.critic_obs_normalizers,
                        operation="Stage-4 init",
                    )
                self._reset_dagger_replay_state()
                self._terminal_fixed_bc_eval_state = None
                self._source_checkpoint_sha256 = actual_sha256
                self._motion_transition_contract = live_motion_contract
                self._motion_transition_contract_sha256 = live_motion_digest
                if self.actor_optimizer.state or self.critic_optimizer.state:
                    raise RuntimeError(
                        "Stage-4 validated commit unexpectedly mutated optimizer state."
                    )
            except Exception as exc:
                commit_error = exc
            self._synchronize_distributed_operation_error(
                commit_error,
                operation="Stage-4 validated state commit",
            )
            if getattr(self, "is_multi_gpu", False):
                self._synchronize_model_weights()
            self._assert_model_parameters_finite(
                phase="stage4-init validated load",
                trainable_only=False,
            )
            logger.info(
                "Stage-4 initialized actor+critic from {}; reset std to {}; ignored "
                "checkpoint iteration, optimizers, RNG, env/curriculum, replay, and W&B identity.",
                ckpt_path,
                float(self.config.init_noise_std),
            )
            return loaded_dict.get("infos")
        finally:
            restore_rng_checkpoint_state(
                rng_state,
                path="pre_stage4_init_load_rng_state",
            )

    def load_evaluation(self, ckpt_path: str | None) -> dict | None:
        """Load only the policy function required by evaluation.

        Evaluation is neither an optimizer continuation nor a policy-init
        lineage edge.  It therefore uses the same validated actor/normalizer
        payload as policy initialization without interpreting the current
        launch's training provenance as a policy-init authorization.
        """

        if ckpt_path is None:
            return None
        rng_state = (
            getattr(self, "_evaluation_rng_boundary_state", None)
            or capture_rng_checkpoint_state()
        )
        try:
            result = self._load_policy_init_impl(
                ckpt_path,
                operation="Evaluation policy load",
                enforce_policy_init_provenance=False,
                restore_evaluation_metadata=True,
            )
            self._prepare_selected_evaluation_policy()
            return result
        finally:
            restore_rng_checkpoint_state(
                rng_state,
                path="pre_evaluation_load_rng_state",
            )

    def _prepare_policy_init_checkpoint(
        self,
        ckpt_path: str,
        *,
        enforce_policy_init_provenance: bool = True,
        operation: str = "Policy-init",
        motion_transition_load_mode: str = "actor_warm_start",
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        bool,
        str,
        dict[str, Any] | None,
        str | None,
    ]:
        """Safely deserialize and fully validate policy-init state."""

        if motion_transition_load_mode not in {"actor_warm_start", "evaluation"}:
            raise ValueError(
                "motion_transition_load_mode must be actor_warm_start or evaluation, got "
                f"{motion_transition_load_mode!r}."
            )

        allow_authenticated_legacy_evaluation_motion_contract = False
        if motion_transition_load_mode == "evaluation":
            raw_legacy_motion_contract = os.environ.get(
                _ALLOW_AUTHENTICATED_LEGACY_EVALUATION_MOTION_CONTRACT_ENV
            )
            if raw_legacy_motion_contract is not None:
                if raw_legacy_motion_contract.strip() == "1":
                    allow_authenticated_legacy_evaluation_motion_contract = True
                elif raw_legacy_motion_contract.strip() not in {"", "0"}:
                    raise ValueError(
                        f"{_ALLOW_AUTHENTICATED_LEGACY_EVALUATION_MOTION_CONTRACT_ENV} "
                        "must be exactly 0 or 1; "
                        f"got {raw_legacy_motion_contract!r}."
                    )

        if allow_authenticated_legacy_evaluation_motion_contract:
            pinned_evaluation_sha256 = os.environ.get(
                _EXPECTED_EVALUATION_CHECKPOINT_SHA256_ENV
            )
            if (
                not isinstance(pinned_evaluation_sha256, str)
                or len(pinned_evaluation_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in pinned_evaluation_sha256
                )
            ):
                raise ValueError(
                    f"{_ALLOW_AUTHENTICATED_LEGACY_EVALUATION_MOTION_CONTRACT_ENV}=1 "
                    "requires HOLOSOMA_EXPECTED_EVALUATION_CHECKPOINT_SHA256 to be "
                    "exactly 64 lowercase hexadecimal characters."
                )

        legacy_unverified_policy_load = allow_legacy_unverified_policy_load()
        required_terminal_target = (
            required_policy_init_terminal_target_from_env()
            if enforce_policy_init_provenance
            else None
        )
        training_provenance = (
            getattr(self, "_training_provenance", None)
            if enforce_policy_init_provenance
            else None
        )
        if training_provenance is not None and not isinstance(training_provenance, dict):
            raise ValueError(
                "Attached policy-init training provenance must be a mapping when present; "
                "the legacy override cannot bypass malformed provenance."
            )
        runtime_experiment_config = getattr(self, "_policy_load_runtime_config", None)
        current_config = (
            runtime_experiment_config.to_serializable_dict()
            if runtime_experiment_config is not None
            and hasattr(runtime_experiment_config, "to_serializable_dict")
            else None
        )
        expected_policy_init_sha256 = None
        if isinstance(training_provenance, dict):
            if training_provenance.get("policy_init_enabled") is not True:
                raise ValueError(
                    "PPO.load_policy_init was called while attached training provenance does not enable "
                    "policy initialization."
                )
            expected_policy_init_sha256 = training_provenance.get("policy_init_sha256")
            if not isinstance(expected_policy_init_sha256, str):
                raise ValueError(
                    "Attached training provenance enables policy initialization but has no authenticated "
                    "policy_init_sha256."
                )
            training_provenance = validate_training_provenance(
                training_provenance,
                require_finalized=True,
            )
            expected_policy_init_sha256 = training_provenance["policy_init_sha256"]
            if not isinstance(current_config, dict):
                raise ValueError(
                    "Provenance-authenticated policy initialization requires an attached runtime "
                    "experiment config; the legacy override cannot downgrade a scientific lineage."
                )
        elif enforce_policy_init_provenance:
            if not legacy_unverified_policy_load:
                raise ValueError(
                    "Scientific policy initialization requires finalized current training "
                    "provenance with an authenticated policy_init_sha256. Set "
                    f"{ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV}=1 only for an explicitly "
                    "non-scientific legacy warm start."
                )
            logger.warning(
                "{}=1: allowing policy initialization without authenticated current provenance.",
                ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV,
            )
        if motion_transition_load_mode == "evaluation":
            expected_evaluation_sha256 = os.environ.get(
                _EXPECTED_EVALUATION_CHECKPOINT_SHA256_ENV
            )
            if expected_evaluation_sha256 is not None:
                if (
                    expected_policy_init_sha256 is not None
                    and expected_policy_init_sha256 != expected_evaluation_sha256
                ):
                    raise ValueError(
                        "Evaluation checkpoint SHA256 disagrees with the authenticated "
                        "policy-init provenance digest."
                    )
                expected_policy_init_sha256 = expected_evaluation_sha256
        if not isinstance(current_config, dict) and not legacy_unverified_policy_load:
            raise ValueError(
                f"{operation} requires an attached runtime experiment config so equal-shaped "
                "observation/action tensors cannot hide semantic drift. Set "
                f"{ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV}=1 only for an explicitly "
                "non-scientific legacy policy load."
            )
        if (
            allow_authenticated_legacy_evaluation_motion_contract
            and not isinstance(current_config, dict)
        ):
            raise ValueError(
                f"{_ALLOW_AUTHENTICATED_LEGACY_EVALUATION_MOTION_CONTRACT_ENV}=1 "
                "requires an attached runtime experiment config and cannot be combined "
                "with an unverified semantic policy load."
            )
        loaded_dict, actual_checkpoint_sha256 = load_verified_torch_checkpoint(
            ckpt_path,
            expected_sha256=expected_policy_init_sha256,
            map_location="cpu",
        )
        if not isinstance(loaded_dict, dict):
            raise ValueError("Policy init checkpoint payload must be a mapping.")
        if not isinstance(loaded_dict.get("experiment_config"), dict):
            raise ValueError(
                "Policy init checkpoint has no serialized experiment_config; "
                f"refusing an unverifiable actor warm start: {ckpt_path}"
            )
        if allow_authenticated_legacy_evaluation_motion_contract and (
            "motion_transition_contract" in loaded_dict
            or "motion_transition_contract_sha256" in loaded_dict
        ):
            raise ValueError(
                f"{_ALLOW_AUTHENTICATED_LEGACY_EVALUATION_MOTION_CONTRACT_ENV}=1 is "
                "only valid for an authenticated legacy evaluation checkpoint that "
                "provides neither motion-transition contract field."
            )
        if required_terminal_target is not None:
            validate_policy_init_terminal_source_payload(
                loaded_dict,
                required_target=required_terminal_target,
            )
        if (
            _TERMINAL_FIXED_BC_EVAL_STATE_KEY in loaded_dict
            or _TERMINAL_FIXED_BC_EVAL_STATE_SHA256_KEY in loaded_dict
        ):
            completed_iteration, _ = validate_checkpoint_iterations(loaded_dict)
            self._validate_terminal_fixed_bc_eval_artifact_payload(
                loaded_dict,
                expected_completed_iteration=completed_iteration,
            )
        if isinstance(current_config, dict):
            validate_policy_init_payload_identity(loaded_dict, current_config)
        else:
            logger.warning(
                "{}=1: allowing {} without a live actor semantic-contract comparison.",
                ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV,
                operation,
            )
        actor_state = loaded_dict.get("actor_model_state_dict")
        if not isinstance(actor_state, dict):
            raise KeyError(f"Checkpoint does not contain actor_model_state_dict: {ckpt_path}")
        validate_finite_tree(actor_state, path="actor_model_state_dict")
        validate_module_state_compatibility(
            actor_state,
            reference_state=self.actor.state_dict(),
            path="actor_model_state_dict",
        )
        self._validate_checkpoint_actor_std(
            actor_state,
            path="actor_model_state_dict",
        )

        actor_normalization_enabled = bool(getattr(self.config, "normalize_actor_obs", False))
        self._validate_checkpoint_normalizers(
            loaded_dict,
            kind="actor",
            runtime_enabled=actor_normalization_enabled,
            normalizers=self.actor_obs_normalizers,
            operation=operation,
        )
        (
            restored_motion_transition_contract,
            restored_motion_transition_digest,
        ) = self._validate_checkpoint_motion_transition_contract(
            loaded_dict,
            live_contract=None,
            compare_live=False,
            operation=operation,
            allow_missing_contract=(
                motion_transition_load_mode == "actor_warm_start"
                or allow_authenticated_legacy_evaluation_motion_contract
            ),
        )
        if allow_authenticated_legacy_evaluation_motion_contract:
            logger.warning(
                "{}=1: evaluating exact checkpoint SHA256 {} without a persisted "
                "motion-transition contract. Actor and observation semantics were "
                "validated, but the evaluation timeline must be specified explicitly "
                "and cannot be claimed as an authenticated training-resume timeline.",
                _ALLOW_AUTHENTICATED_LEGACY_EVALUATION_MOTION_CONTRACT_ENV,
                actual_checkpoint_sha256,
            )
        return (
            loaded_dict,
            actor_state,
            actor_normalization_enabled,
            actual_checkpoint_sha256,
            restored_motion_transition_contract,
            restored_motion_transition_digest,
        )

    def _load_policy_init_impl(
        self,
        ckpt_path: str,
        *,
        operation: str = "Policy-init",
        enforce_policy_init_provenance: bool = True,
        restore_evaluation_metadata: bool = False,
    ) -> dict | None:
        """Run policy-init with all-rank validation and commit envelopes."""

        logger.info("{} from checkpoint: {}", operation, ckpt_path)
        prepared: tuple[
            dict[str, Any],
            dict[str, Any],
            bool,
            str,
            dict[str, Any] | None,
            str | None,
        ] | None = None
        evaluation_completed_iteration: int | None = None
        evaluation_source_provenance: dict[str, Any] | None = None
        evaluation_perception_geometry_support: dict[str, Any] | None = None
        evaluation_terminal_fixed_bc_state: dict[str, Any] | None = None
        validation_error: Exception | None = None
        try:
            prepared = self._prepare_policy_init_checkpoint(
                ckpt_path,
                enforce_policy_init_provenance=enforce_policy_init_provenance,
                operation=operation,
                motion_transition_load_mode=(
                    "evaluation" if restore_evaluation_metadata else "actor_warm_start"
                ),
            )
            if restore_evaluation_metadata:
                loaded_dict = prepared[0]
                evaluation_perception_geometry_support = (
                    self._prepare_evaluation_perception_geometry_support(loaded_dict)
                )
                (
                    evaluation_completed_iteration,
                    _evaluation_next_iteration,
                ) = validate_checkpoint_iterations(loaded_dict)
                evaluation_terminal_fixed_bc_state = (
                    self._validate_checkpoint_terminal_fixed_bc_eval(
                        loaded_dict,
                        expected_completed_iteration=evaluation_completed_iteration,
                    )
                )
                raw_source_provenance = loaded_dict.get("training_provenance")
                if raw_source_provenance is not None:
                    evaluation_source_provenance = validate_training_provenance(
                        raw_source_provenance,
                        require_finalized=True,
                    )
        except Exception as exc:
            validation_error = exc
        self._synchronize_distributed_operation_error(
            validation_error,
            operation=f"{operation} checkpoint deserialization/validation",
        )
        assert prepared is not None
        (
            loaded_dict,
            actor_state,
            actor_normalization_enabled,
            actual_checkpoint_sha256,
            restored_motion_transition_contract,
            restored_motion_transition_digest,
        ) = prepared

        commit_error: Exception | None = None
        try:
            self.actor.load_state_dict(actor_state, strict=True)
            self._sanitize_actor_std()
            if actor_normalization_enabled:
                self._restore_checkpoint_normalizers(
                    loaded_dict,
                    kind="actor",
                    runtime_enabled=True,
                    normalizers=self.actor_obs_normalizers,
                    operation=operation,
                )
            if restore_evaluation_metadata:
                assert evaluation_completed_iteration is not None
                self._evaluation_completed_iteration = evaluation_completed_iteration
                self._training_provenance = evaluation_source_provenance
                self._source_checkpoint_sha256 = actual_checkpoint_sha256
                self._actor_perception_training_geometry_support = (
                    evaluation_perception_geometry_support
                )
                self._source_experiment_config_dict = dict(
                    loaded_dict["experiment_config"]
                )
                self._motion_transition_contract = restored_motion_transition_contract
                self._motion_transition_contract_sha256 = restored_motion_transition_digest
                # Evaluation may re-export the exact loaded final actor; keep
                # its already validated terminal proof for that corresponding
                # ONNX only.  Actor warm-start and full-resume paths discard it.
                self._terminal_fixed_bc_eval_state = (
                    evaluation_terminal_fixed_bc_state
                )
            else:
                self._terminal_fixed_bc_eval_state = None
                # Actor warm-start is a fresh training lineage.  Replay data,
                # reservoir counters, and its sampler state belong only to an
                # exact full resume and must never be inherited from the
                # source checkpoint or a prior live invocation.
                self._reset_dagger_replay_state()
        except Exception as exc:
            commit_error = exc
        self._synchronize_distributed_operation_error(
            commit_error,
            operation=f"{operation} validated state commit",
        )
        if getattr(self, "is_multi_gpu", False):
            # Policy init is loaded after setup() performed its initial model
            # broadcast, so synchronize the newly loaded actor explicitly.
            # The unchanged critic is re-broadcast at the same collective
            # boundary to keep every rank's call order identical.
            self._synchronize_model_weights()
        self._assert_model_parameters_finite(
            phase=f"{operation.lower()} validated load",
            trainable_only=False,
        )

        checkpoint_iter = loaded_dict.get("iter", loaded_dict.get("iteration", "<unknown>"))
        if restore_evaluation_metadata:
            logger.info(
                "Loaded actor policy parameters from {} for evaluation at completed iteration={}; "
                "ignored critic, optimizers, critic normalizers, and env_state; actor normalizers "
                "restored={}; source checkpoint SHA256 and provenance retained for export metadata.",
                ckpt_path,
                checkpoint_iter,
                actor_normalization_enabled,
            )
        else:
            logger.info(
                "Loaded actor policy parameters from {}; ignored checkpoint iteration={}, critic, optimizers, "
                "critic normalizers, and env_state; actor normalizers restored={}. "
                "Training will start from iteration {}.",
                ckpt_path,
                checkpoint_iter,
                actor_normalization_enabled,
                self.current_learning_iteration,
            )
        return loaded_dict.get("infos")

    def _validate_checkpoint_publish_state(self) -> None:
        """Reject a non-finite live training state before checkpoint collection."""

        local_error: Exception | None = None
        try:
            actor_model_state = self.actor.state_dict()
            critic_model_state = self.critic.state_dict()
            validate_finite_tree(
                actor_model_state,
                path="live.actor_model_state_dict",
            )
            validate_finite_tree(
                critic_model_state,
                path="live.critic_model_state_dict",
            )
            self._validate_checkpoint_actor_std(
                actor_model_state,
                path="live.actor_model_state_dict",
            )

            actor_optimizer_state = self.actor_optimizer.state_dict()
            critic_optimizer_state = self.critic_optimizer.state_dict()
            validate_optimizer_state(
                actor_optimizer_state,
                path="live.actor_optimizer_state_dict",
                minimum_lr=float(getattr(self, "min_actor_learning_rate", 0.0)),
                maximum_lr=float(getattr(self, "max_actor_learning_rate", float("inf"))),
            )
            validate_optimizer_state(
                critic_optimizer_state,
                path="live.critic_optimizer_state_dict",
                minimum_lr=float(getattr(self, "min_critic_learning_rate", 0.0)),
                maximum_lr=float(getattr(self, "max_critic_learning_rate", float("inf"))),
            )
            validate_optimizer_compatibility(
                actor_optimizer_state,
                optimizer=self.actor_optimizer,
                path="live.actor_optimizer_state_dict",
            )
            validate_optimizer_compatibility(
                critic_optimizer_state,
                optimizer=self.critic_optimizer,
                path="live.critic_optimizer_state_dict",
            )
            actor_normalizer_state = {
                key: normalizer.state_dict() if hasattr(normalizer, "state_dict") else None
                for key, normalizer in getattr(self, "actor_obs_normalizers", {}).items()
            }
            critic_normalizer_state = {
                key: normalizer.state_dict() if hasattr(normalizer, "state_dict") else None
                for key, normalizer in getattr(self, "critic_obs_normalizers", {}).items()
            }
            validate_finite_tree(
                actor_normalizer_state,
                path="live.actor_obs_normalizer_state",
            )
            validate_finite_tree(
                critic_normalizer_state,
                path="live.critic_obs_normalizer_state",
            )
            for kind, normalizers, states in (
                ("actor", getattr(self, "actor_obs_normalizers", {}), actor_normalizer_state),
                ("critic", getattr(self, "critic_obs_normalizers", {}), critic_normalizer_state),
            ):
                for key, normalizer in normalizers.items():
                    state = states[key]
                    if not isinstance(state, dict):
                        raise ValueError(
                            f"Live {kind} normalizer {key!r} did not produce a state mapping."
                        )
                    self._validate_normalizer_module_state(
                        normalizer,
                        state,
                        path=f"live.{kind}_obs_normalizer_state[{key!r}]",
                    )
        except Exception as exc:
            local_error = exc
        self._synchronize_distributed_operation_error(
            local_error,
            operation="Checkpoint finite-state preflight",
        )

    def save(
        self,
        path,
        infos=None,
        *,
        next_iteration: int | None = None,
        allow_tripped_fixed_bc_guard: bool = False,
        upload: bool = True,
    ):
        # Capture the stochastic boundary before *any* checkpoint-side
        # inspection or collection.  State-dict hooks, validation,
        # environment/fixed-BC serialization, their object collectives,
        # torch.save, and W&B publication are observational side effects, so
        # none may perturb either the serialized boundary or the next training
        # draw.
        rng_states_by_rank = self._collect_distributed_rng_states()
        rank = int(getattr(self, "gpu_global_rank", 0) or 0)
        local_rng_state = rng_states_by_rank[str(rank)]
        try:
            # Do not collect rank-local state or publish an artifact from a
            # corrupt actor, critic, module buffer, normalizer, or optimizer
            # state.  This remains inside the RNG restoration scope because
            # custom state-dict hooks are allowed to execute arbitrary code.
            self._validate_checkpoint_publish_state()
            if next_iteration is None:
                # current_learning_iteration is consistently the next update to
                # execute outside the body of an active update.
                next_iteration = int(self.current_learning_iteration)
            next_iteration = int(next_iteration)
            env_states_by_rank = self._collect_distributed_env_states()
            (
                motion_transition_contract,
                motion_transition_contract_sha256,
            ) = self._collect_distributed_motion_transition_contract()
            self._motion_transition_contract = motion_transition_contract
            self._motion_transition_contract_sha256 = motion_transition_contract_sha256
            actor_perception_geometry_support = (
                self._aggregate_actor_perception_geometry_support(
                    env_states_by_rank,
                    allow_legacy_missing=False,
                )
            )
            self._actor_perception_training_geometry_support = (
                actor_perception_geometry_support
            )
            fixed_bc_eval_by_rank = self._collect_distributed_fixed_bc_eval_states()
            dagger_replay_by_rank = self._collect_distributed_dagger_replay_states(
                fixed_bc_eval_by_rank=fixed_bc_eval_by_rank,
            )
            fixed_bc_guard_state = self._fixed_bc_guard_checkpoint_state(
                fixed_bc_eval_by_rank,
                next_iteration=next_iteration,
                allow_tripped=allow_tripped_fixed_bc_guard,
            )
            if not self.is_main_process:
                return
            completed_iteration = next_iteration - 1

            def normalizer_states(normalizers: dict[str, nn.Module]):
                states: dict[str, dict | None] = {}
                for key, normalizer in normalizers.items():
                    states[key] = normalizer.state_dict() if hasattr(normalizer, "state_dict") else None
                return states

            checkpoint_dict = {
                "actor_model_state_dict": self.actor.state_dict(),
                "critic_model_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "iter": completed_iteration,
                "next_iter": next_iteration,
                "infos": infos,
                "actor_obs_normalizer_state": normalizer_states(self.actor_obs_normalizers),
                "critic_obs_normalizer_state": normalizer_states(self.critic_obs_normalizers),
                "rng_state_by_rank": rng_states_by_rank,
                "rollout_resume_contract": self._rollout_resume_contract(next_iteration),
            }
            checkpoint_dict.update(self._checkpoint_metadata(iteration=completed_iteration))
            if actor_perception_geometry_support is not None:
                checkpoint_dict[_ACTOR_PERCEPTION_GEOMETRY_SUPPORT_KEY] = (
                    actor_perception_geometry_support
                )
            if fixed_bc_eval_by_rank:
                checkpoint_dict["fixed_bc_eval_by_rank"] = fixed_bc_eval_by_rank
            if dagger_replay_by_rank:
                checkpoint_dict[_DAGGER_REPLAY_STATE_KEY] = dagger_replay_by_rank
            if fixed_bc_guard_state is not None:
                checkpoint_dict["fixed_bc_guard_state"] = fixed_bc_guard_state
            if any(env_states_by_rank.values()):
                checkpoint_dict["env_state_by_rank"] = env_states_by_rank
                # Keep a rank-zero alias so older single-rank consumers can still
                # inspect checkpoints written by this version.  New PPO loads
                # always prefer env_state_by_rank.
                rank_zero_state = env_states_by_rank.get("0")
                if rank_zero_state:
                    checkpoint_dict["env_state"] = rank_zero_state
            configured_run_target = getattr(
                getattr(self, "config", None),
                "num_learning_iterations",
                None,
            )
            is_configured_final = (
                configured_run_target is not None
                and next_iteration == int(configured_run_target)
            )
            terminal_fixed_bc_required = bool(
                is_configured_final
                and not allow_tripped_fixed_bc_guard
                and getattr(self, "dagger_enabled", False)
                and int(getattr(self, "fixed_bc_eval_num_samples", 0) or 0) > 0
                and int(getattr(self, "fixed_bc_eval_log_interval", 0) or 0) > 0
            )
            if is_configured_final and not allow_tripped_fixed_bc_guard:
                terminal_metadata = self._terminal_fixed_bc_eval_artifact_metadata(
                    completed_iteration=completed_iteration,
                )
                if terminal_fixed_bc_required and not terminal_metadata:
                    raise RuntimeError(
                        "Refusing to publish the final DAgger checkpoint without its "
                        "validated terminal fixed-BC observation."
                    )
                if terminal_metadata:
                    terminal_candidate = dict(checkpoint_dict)
                    terminal_candidate.update(terminal_metadata)
                    self._validate_terminal_fixed_bc_eval_artifact_payload(
                        terminal_candidate,
                        expected_completed_iteration=completed_iteration,
                        compare_runtime_guard_config=True,
                    )
                    checkpoint_dict.update(terminal_metadata)
            validate_finite_tree(
                checkpoint_dict,
                path="checkpoint_publish_payload",
            )
            if upload:
                self.logging_helper.save_checkpoint_artifact(checkpoint_dict, path)
            else:
                self.logging_helper.save_checkpoint_artifact(
                    checkpoint_dict,
                    path,
                    upload=False,
                )
        finally:
            # torch serialization and W&B artifact publication are not part
            # of the training stochastic process.  Preserve the checkpointed
            # state even if those libraries use a global RNG internally.
            restore_rng_checkpoint_state(
                local_rng_state,
                path=f"rng_state_by_rank[{rank}]",
            )

    def export(
        self,
        onnx_file_path: str,
        *,
        iteration: int | None = None,
        upload: bool = True,
    ):
        """Export the `.onnx` of the policy to & save it to `path`.

        This is intended to enable deployment, but not resuming training.
        For storing checkpoints to resume training, see `PPO.save()`
        """
        # Save current training state
        was_training = self.actor.training

        # Set model to evaluation mode for export so we don't affect gradients mid-rollout
        self._eval_mode()
        try:
            self._prepare_motion_transition_contract_for_export()
            # Save a pure policy .onnx for deployment. Motion replay/reference
            # tensors belong in debug/demo tooling, not in the policy artifact.
            example_obs_dict = {"actor_obs": self._get_zero_input()}
            zero_perception = self._get_zero_perception_input()
            perception_observation_contract = None
            perception_observation_contract_sha256 = None
            if zero_perception is not None:
                example_obs_dict[self.actor_perception_key] = zero_perception
                perception_manager = self._live_actor_perception_manager()
                contract_getter = getattr(perception_manager, "get_observation_contract", None)
                digest_getter = getattr(perception_manager, "get_observation_contract_sha256", None)
                if not callable(contract_getter) or not callable(digest_getter):
                    raise RuntimeError(
                        "Perception policy ONNX export requires the live student perception manager's "
                        "effective observation contract."
                    )
                training_geometry_support = getattr(
                    self,
                    "_actor_perception_training_geometry_support",
                    None,
                )
                if training_geometry_support is None:
                    raise RuntimeError(
                        "Perception policy ONNX export requires all-rank training geometry support "
                        "authenticated by the checkpoint environment state. Re-save/retrain with the "
                        "current checkpoint format; a fresh eval scene cannot certify old weights."
                    )
                perception_manager.validate_deployment_geometry_support(
                    training_geometry_support
                )
                perception_observation_contract = contract_getter(
                    training_geometry_support=training_geometry_support
                )
                perception_observation_contract_sha256 = digest_getter(
                    training_geometry_support=training_geometry_support
                )
            export_policy_as_onnx(
                wrapper=self.actor_onnx_wrapper,
                onnx_file_path=onnx_file_path,
                example_obs_dict=example_obs_dict,
                perception_input_name=self.actor_perception_key or None,
            )

            # Extract control gains and velocity limits & attach to onnx as metadata
            kp_list, kd_list = get_control_gains_from_config(self.env.robot_config)
            cmd_ranges = get_command_ranges_from_env(self.env)
            # Extract URDF text from the robot config
            urdf_file_path, urdf_str = get_urdf_text_from_robot_config(self.env.robot_config)

            metadata = {
                "dof_names": self.env.robot_config.dof_names,
                "kp": kp_list,
                "kd": kd_list,
                "command_ranges": cmd_ranges,
                "robot_urdf": urdf_str,
                "robot_urdf_path": urdf_file_path,
            }
            if perception_observation_contract is not None:
                metadata["perception_observation_contract"] = perception_observation_contract
                metadata["perception_observation_contract_sha256"] = perception_observation_contract_sha256
            if iteration is None:
                evaluation_iteration = getattr(
                    self,
                    "_evaluation_completed_iteration",
                    None,
                )
                completed_iteration = (
                    int(evaluation_iteration)
                    if evaluation_iteration is not None
                    else int(self.current_learning_iteration) - 1
                )
            else:
                completed_iteration = int(iteration)
            metadata.update(self._checkpoint_metadata(iteration=completed_iteration))
            terminal_metadata = self._terminal_fixed_bc_eval_artifact_metadata(
                completed_iteration=completed_iteration,
            )
            configured_run_target = getattr(
                getattr(self, "config", None),
                "num_learning_iterations",
                None,
            )
            terminal_fixed_bc_required = bool(
                configured_run_target is not None
                and completed_iteration + 1 == int(configured_run_target)
                and getattr(self, "dagger_enabled", False)
                and int(getattr(self, "fixed_bc_eval_num_samples", 0) or 0) > 0
                and int(getattr(self, "fixed_bc_eval_log_interval", 0) or 0) > 0
            )
            if terminal_fixed_bc_required and not terminal_metadata:
                raise RuntimeError(
                    "Refusing to publish the final DAgger ONNX without its validated "
                    "terminal fixed-BC observation."
                )
            metadata.update(terminal_metadata)

            attach_onnx_metadata(
                onnx_path=onnx_file_path,
                metadata=metadata,
            )

            parity_report = validate_exported_policy_onnx(
                wrapper=self.actor_onnx_wrapper,
                onnx_file_path=onnx_file_path,
                example_obs_dict=example_obs_dict,
                perception_input_name=self.actor_perception_key or None,
            )
            parity_report.update(
                {
                    "completed_iteration": completed_iteration,
                    "actor_graph_semantics": (
                        "raw_actor_observation_plus_authenticated_external_observation_adapter"
                    ),
                    "precomputed_command_contract_sha256": metadata.get(
                        _PRECOMPUTED_COMMAND_CONTRACT_SHA256_KEY
                    ),
                    "rolling_reference_delta_contract_sha256": metadata.get(
                        _ROLLING_REFERENCE_DELTA_CONTRACT_SHA256_KEY
                    ),
                }
            )
            attach_onnx_metadata(
                onnx_path=onnx_file_path,
                metadata={"onnx_validation_contract": parity_report},
            )
            validate_exported_policy_onnx(
                wrapper=self.actor_onnx_wrapper,
                onnx_file_path=onnx_file_path,
                example_obs_dict=example_obs_dict,
                perception_input_name=self.actor_perception_key or None,
            )

            # Upload the .onnx file to wandb
            if upload:
                self.logging_helper.save_to_wandb(onnx_file_path)
        finally:
            # An exporter/metadata failure must not leave the live training
            # policy in eval mode for all subsequent iterations.
            if was_training:
                self._train_mode()

    def _post_epoch_logging(self, it, loss_dict, *, fixed_bc_eval_metrics=None):
        mean_noise_std_tensor = self.actor.std.detach().mean()
        extra_log_dicts = {
            "Policy": {
                "mean_noise_std": self._loss_to_float(mean_noise_std_tensor),
                "mean_noise_std_is_finite": float(torch.isfinite(mean_noise_std_tensor).item()),
            },
        }
        motion_command = None
        if self.env.command_manager is not None:
            motion_command = self.env.command_manager.get_state("motion_command")
        if motion_command is not None:
            train_logs = extra_log_dicts.setdefault("Train", {})
            motion_total = float(motion_command.motion.time_step_total)
            train_logs["mean_episode_length_motion_total"] = motion_total
            train_logs["mean_episode_length_motion_total/time"] = motion_total
            train_logs["command_goal_training_iteration"] = float(getattr(motion_command, "_training_iteration", it) or it)
            if hasattr(motion_command, "get_clean_noisy_clip_curriculum_log_state"):
                train_logs.update(motion_command.get_clean_noisy_clip_curriculum_log_state())
        if self.dagger_enabled and self.use_ppo_dagger_schedule:
            train_logs = extra_log_dicts.setdefault("Train", {})
            train_logs["ppo_dagger_target_coeff"] = float(self.ppo_target_coeff)
            train_logs["ppo_dagger_start_coeff"] = float(self.ppo_start_coeff)
            train_logs["ppo_dagger_coeff"] = float(self.ppo_coeff)
            train_logs["ppo_dagger_bc_weight"] = float(self.dagger_loss_coef * max(0.0, 1.0 - float(self.ppo_coeff)))
            if self.ppo_start_noise_std is not None:
                train_logs["ppo_start_noise_std"] = float(self.ppo_start_noise_std)
                train_logs["ppo_start_noise_std_until_coeff"] = float(self.ppo_start_noise_std_until_coeff)
        if self.dagger_enabled:
            train_logs = extra_log_dicts.setdefault("Train", {})
            train_logs["teacher_action_mix_ratio"] = float(self.teacher_action_mix_ratio)
            if self.use_teacher_action_mix_schedule:
                train_logs["teacher_action_mix_ratio_start"] = float(self.teacher_action_mix_ratio_start)
                train_logs["teacher_action_mix_ratio_end"] = float(self.teacher_action_mix_ratio_end)
                train_logs["teacher_action_mix_ratio_end_iteration"] = float(self.teacher_action_mix_ratio_end_iteration)
            if fixed_bc_eval_metrics:
                extra_log_dicts.setdefault("Eval", {}).update(fixed_bc_eval_metrics)
        extra_log_dicts.setdefault("Train", {})["distributed_loss_weight"] = self._get_distributed_loss_weight()
        if self.is_multi_gpu:
            extra_log_dicts["Train"]["distributed_loss_weight_sum"] = float(
                self.logging_helper.distributed_loss_weight_sum
            )
            extra_log_dicts["Train"]["distributed_effective_episode_count"] = float(
                self.logging_helper.distributed_effective_episode_count
            )
        self._add_step_timing_logs(extra_log_dicts)
        loss_dict["actor_learning_rate"] = self.actor_learning_rate
        loss_dict["critic_learning_rate"] = self.critic_learning_rate
        # Use logging helper
        self.logging_helper.post_epoch_logging(it=it, loss_dict=loss_dict, extra_log_dicts=extra_log_dicts)

    def _get_distributed_loss_weight(self) -> float:
        env = self._unwrap_env()
        raw_weight = getattr(env, "distributed_loss_weight", 1.0)
        if torch.is_tensor(raw_weight):
            if raw_weight.numel() != 1:
                raise ValueError("env.distributed_loss_weight must be a scalar.")
            raw_weight = raw_weight.detach().item()
        weight = float(raw_weight)
        if weight != weight or weight in (float("inf"), float("-inf")) or weight < 0.0:
            raise ValueError(
                "env.distributed_loss_weight must be finite and non-negative, "
                f"got {weight}."
            )
        return weight

    def _global_bc_denominator_and_presence(
        self,
        local_valid_count: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """Return the weighted denominator and whether any valid sample exists.

        Gradients are multiplied by ``distributed_loss_weight`` and averaged
        over ranks in ``_reduce_parameters``.  Dividing each local BC sum by
        ``sum_r(weight_r * count_r) / world_size`` therefore yields the exact
        globally weighted valid-sample mean after that reduction.
        """
        local_valid_count = local_valid_count.detach()
        if not self.is_multi_gpu:
            return torch.clamp(local_valid_count, min=1.0), bool((local_valid_count > 0.0).item())
        weighted_count = local_valid_count * self._get_distributed_loss_weight()
        global_weighted_count = self._all_reduce_small_tensor(
            weighted_count,
            op=torch.distributed.ReduceOp.SUM,
        )
        denominator = global_weighted_count / float(self.gpu_world_size)
        # A sparse mask can legitimately produce a denominator below one
        # after the DDP averaging factor (for example one valid sample across
        # two ranks -> 0.5).  Clamping that value to one silently shrinks BC
        # gradients.  Only the all-zero case needs a safe nonzero divisor.
        has_valid_samples = bool((global_weighted_count > 0.0).item())
        denominator = torch.where(
            global_weighted_count > 0.0,
            denominator,
            torch.ones_like(denominator),
        )
        return denominator, has_valid_samples

    def _global_bc_denominator(self, local_valid_count: torch.Tensor) -> torch.Tensor:
        """Backward-compatible denominator-only wrapper."""
        denominator, _ = self._global_bc_denominator_and_presence(local_valid_count)
        return denominator

    def _rollout_bc_denominator_per_minibatch(self) -> torch.Tensor | None:
        """Return one rollout-wide BC scale shared by every minibatch.

        Equal-sized minibatches are optimizer steps, so the per-step
        denominator is the global weighted valid count divided by the number
        of minibatches.  Normalizing each step by its own valid count instead
        weights sparse and dense mask regions equally rather than weighting
        individual valid samples equally.
        """

        if self._effective_dagger_loss_weight() <= 0.0:
            return None
        num_mini_batches = int(getattr(self.config, "num_mini_batches", 0))
        if num_mini_batches <= 0:
            raise ValueError(
                f"num_mini_batches must be positive, got {num_mini_batches}."
            )
        filled_steps = int(getattr(self.storage, "step", 0) or 0)
        if filled_steps <= 0:
            raise RuntimeError("Cannot normalize DAgger BC before rollout storage is filled.")
        teacher_actions = self.storage["teacher_actions"][:filled_steps]
        valid_mask = torch.ones(
            teacher_actions.shape[:-1],
            device=teacher_actions.device,
            dtype=torch.bool,
        )
        try:
            teacher_bc_mask = self.storage["teacher_bc_mask"][:filled_steps]
        except KeyError:
            teacher_bc_mask = None
        if teacher_bc_mask is not None:
            valid_mask &= teacher_bc_mask.view_as(valid_mask).to(dtype=torch.bool)
        if self.dagger_ignore_zero_teacher_actions:
            valid_mask &= ~torch.all(teacher_actions == 0.0, dim=-1)
        local_valid_count = valid_mask.to(dtype=teacher_actions.dtype).sum()
        global_denominator, _ = self._global_bc_denominator_and_presence(
            local_valid_count
        )
        return global_denominator / float(num_mini_batches)

    def _rollout_bc_minibatch_presence(self) -> tuple[bool, ...] | None:
        """Batch the current-minibatch BC-presence votes for contiguous PPO.

        The production configuration slices the time-major flattened rollout
        contiguously.  Its complete presence plan is therefore known before
        the first optimizer step and can be reduced as one short vector rather
        than as one 104-rank collective per minibatch.  Randomized and
        recurrent generators retain the legacy per-minibatch fallback because
        their exact index plan is not available here.
        """

        if self._effective_dagger_loss_weight() <= 0.0 or self.use_time_gru:
            return None
        use_contiguous = os.environ.get("HOLOSOMA_CONTIGUOUS_MINIBATCHES", "").lower() not in (
            "",
            "0",
            "false",
            "no",
        )
        if not use_contiguous:
            return None

        num_mini_batches = int(getattr(self.config, "num_mini_batches", 0))
        num_learning_epochs = int(getattr(self.config, "num_learning_epochs", 0))
        if num_mini_batches <= 0 or num_learning_epochs <= 0:
            raise ValueError(
                "num_mini_batches and num_learning_epochs must both be positive "
                "when constructing the DAgger presence plan."
            )
        filled_steps = int(getattr(self.storage, "step", 0) or 0)
        if filled_steps <= 0:
            raise RuntimeError("Cannot plan DAgger BC presence before rollout storage is filled.")

        teacher_actions = self.storage["teacher_actions"][:filled_steps]
        valid_mask = torch.ones(
            teacher_actions.shape[:-1],
            device=teacher_actions.device,
            dtype=torch.bool,
        )
        try:
            teacher_bc_mask = self.storage["teacher_bc_mask"][:filled_steps]
        except KeyError:
            teacher_bc_mask = None
        if teacher_bc_mask is not None:
            valid_mask &= teacher_bc_mask.view_as(valid_mask).to(dtype=torch.bool)
        if self.dagger_ignore_zero_teacher_actions:
            valid_mask &= ~torch.all(teacher_actions == 0.0, dim=-1)

        flat_valid_mask = valid_mask.reshape(-1)
        if flat_valid_mask.numel() % num_mini_batches != 0:
            raise ValueError(
                "Filled rollout size must be divisible by num_mini_batches for "
                "contiguous DAgger presence planning: "
                f"batch_size={flat_valid_mask.numel()}, num_mini_batches={num_mini_batches}."
            )
        local_counts = flat_valid_mask.reshape(num_mini_batches, -1).to(
            dtype=teacher_actions.dtype
        ).sum(dim=1)
        if self.is_multi_gpu:
            local_counts = local_counts * self._get_distributed_loss_weight()
            global_counts = self._all_reduce_small_tensor(
                local_counts,
                op=torch.distributed.ReduceOp.SUM,
            )
        else:
            global_counts = local_counts

        one_epoch_presence = tuple(
            bool(value)
            for value in (global_counts > 0.0).detach().cpu().tolist()
        )
        if len(one_epoch_presence) != num_mini_batches:
            raise RuntimeError(
                "DAgger presence plan length does not match num_mini_batches: "
                f"expected={num_mini_batches}, actual={len(one_epoch_presence)}."
            )
        return one_epoch_presence * num_learning_epochs

    def _bc_denominator_and_presence_for_minibatch(
        self,
        minibatch: Mapping[str, Any],
        local_valid_count: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """Use rollout-wide scaling while retaining current-batch presence."""

        rollout_denominator = minibatch.get("_dagger_bc_denominator")
        if rollout_denominator is None:
            if self.is_multi_gpu and getattr(
                self,
                "_compute_loss_requires_prepared_dagger_controls",
                False,
            ):
                raise RuntimeError(
                    "Distributed DAgger denominator was not prepared before local loss construction."
                )
            return self._global_bc_denominator_and_presence(local_valid_count)
        if not isinstance(rollout_denominator, torch.Tensor) or rollout_denominator.numel() != 1:
            raise ValueError("_dagger_bc_denominator must be one scalar tensor.")
        rollout_denominator = rollout_denominator.to(
            device=local_valid_count.device,
            dtype=local_valid_count.dtype,
        )
        if not bool(torch.isfinite(rollout_denominator).item()) or not bool(
            (rollout_denominator > 0.0).item()
        ):
            raise ValueError("_dagger_bc_denominator must be finite and positive.")
        cached_presence = minibatch.get("_dagger_bc_has_valid_samples")
        if cached_presence is not None:
            if type(cached_presence) is not bool:
                raise ValueError("_dagger_bc_has_valid_samples must be a Python bool.")
            return rollout_denominator, cached_presence
        if self.is_multi_gpu and getattr(
            self,
            "_compute_loss_requires_prepared_dagger_controls",
            False,
        ):
            raise RuntimeError(
                "Distributed DAgger presence was not prepared before local loss construction."
            )
        _, has_valid_samples = self._global_bc_denominator_and_presence(
            local_valid_count
        )
        return rollout_denominator, has_valid_samples

    def _reduce_parameters(self, include_critic: bool = True):
        models = [self.actor]
        if include_critic:
            models.append(self.critic)
        # Frozen encoders are already synchronized as model parameters during
        # distributed setup and can never receive gradients.  Packing them in
        # every minibatch payload is both unnecessary and catastrophic for a
        # large frozen perception backbone (for example DeFM).
        params = [
            param
            for model in models
            for param in model.parameters()
            if param.requires_grad
        ]
        if not params:
            return

        debug_grad_reduce = os.environ.get("HOLOSOMA_DEBUG_GRAD_REDUCE", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        rank_label = f"{getattr(self, 'gpu_global_rank', 0)}/{getattr(self, 'gpu_world_size', 1)}"
        total_numel = sum(param.numel() for param in params)
        first_grad = next((param.grad for param in params if param.grad is not None), None)
        dtype = first_grad.dtype if first_grad is not None else params[0].dtype
        device = params[0].device
        all_grads = torch.zeros(total_numel, device=device, dtype=dtype)
        grad_mask = torch.zeros(len(params), device=device, dtype=dtype)
        distributed_loss_weight = self._get_distributed_loss_weight()

        offset = 0
        local_grad_param_count = 0
        local_grad_numel = 0
        first_missing_param_idx = None
        for param_idx, param in enumerate(params):
            numel = param.numel()
            if param.grad is not None:
                grad = param.grad.detach()
                all_grads[offset : offset + numel].copy_(
                    grad.reshape(-1).to(dtype=dtype) * distributed_loss_weight
                )
                grad_mask[param_idx] = 1.0
                local_grad_param_count += 1
                local_grad_numel += numel
            elif first_missing_param_idx is None:
                first_missing_param_idx = param_idx
            offset += numel

        debug_minibatch_idx = getattr(self, "_debug_current_minibatch_idx", None)
        if debug_grad_reduce:
            print(
                "GradReducePrint "
                f"iter={self.current_learning_iteration} rank={rank_label} minibatch={debug_minibatch_idx} "
                f"phase=begin include_critic={include_critic} "
                f"local_grad_params={local_grad_param_count}/{len(params)} "
                f"local_grad_numel={local_grad_numel}/{total_numel} "
                f"first_missing_param_idx={first_missing_param_idx}",
                flush=True,
            )
            logger.info(
                "GradReduce: iter {} rank {} begin include_critic={} local_grad_params={}/{} "
                "local_grad_numel={}/{} first_missing_param_idx={}",
                self.current_learning_iteration,
                rank_label,
                include_critic,
                local_grad_param_count,
                len(params),
                local_grad_numel,
                total_numel,
                first_missing_param_idx,
            )

        payload = torch.cat((all_grads, grad_mask))
        reduce_path = self._all_reduce_grad_payload(payload)
        reduced_grads = payload[:total_numel].div_(self.gpu_world_size)
        grad_counts = payload[total_numel:].detach().cpu()

        if debug_grad_reduce:
            global_grad_param_count = int((grad_counts > 0).sum().item())
            print(
                "GradReducePrint "
                f"iter={self.current_learning_iteration} rank={rank_label} minibatch={debug_minibatch_idx} "
                f"phase=finished path={reduce_path} global_grad_params={global_grad_param_count}/{len(params)}",
                flush=True,
            )
            logger.info(
                "GradReduce: iter {} rank {} reduced path={} global_grad_params={}/{}",
                self.current_learning_iteration,
                rank_label,
                reduce_path,
                global_grad_param_count,
                len(params),
            )

        offset = 0
        for param_idx, param in enumerate(params):
            numel = param.numel()
            if grad_counts[param_idx].item() > 0:
                reduced_view = reduced_grads[offset : offset + numel].view_as(param)
                if param.grad is None:
                    param.grad = torch.empty_like(param, memory_format=torch.preserve_format)
                param.grad.detach().copy_(reduced_view.to(dtype=param.grad.dtype))
            else:
                param.grad = None
            offset += numel

    def _hierarchical_grad_reduce_enabled(self) -> bool:
        return os.environ.get("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    @staticmethod
    def _topology_local_world_size() -> int:
        raw = os.environ.get("HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE") or os.environ.get("LOCAL_WORLD_SIZE", "1")
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid distributed topology local world size: {raw!r}.") from exc
        if value <= 0:
            raise ValueError(f"Distributed topology local world size must be positive, got {value}.")
        return value

    @staticmethod
    def _topology_local_rank() -> int:
        raw = os.environ.get("HOLOSOMA_ORIGINAL_LOCAL_RANK") or os.environ.get("LOCAL_RANK", "0")
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid distributed topology local rank: {raw!r}.") from exc
        if value < 0:
            raise ValueError(f"Distributed topology local rank must be non-negative, got {value}.")
        return value

    @staticmethod
    def _is_node_local_main_process() -> bool:
        """Return true exactly once per torchrun node, including rank-visible launches."""
        return PPO._topology_local_rank() == 0

    def _gloo_grad_reduce_enabled(self) -> bool:
        return os.environ.get("HOLOSOMA_GLOO_GRAD_REDUCE", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _setup_gloo_grad_reduce_group(self):
        if self._gloo_grad_reduce_ready:
            return self._gloo_grad_reduce_group
        self._gloo_grad_reduce_ready = True
        if not self.is_multi_gpu or not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return None
        if self._gloo_barrier_group is not None:
            self._gloo_grad_reduce_group = self._gloo_barrier_group
        else:
            self._gloo_grad_reduce_group = torch.distributed.new_group(
                ranks=list(range(self.gpu_world_size)),
                backend="gloo",
            )
        if self.is_main_process:
            logger.info("Gloo CPU gradient reduce enabled across {} ranks.", self.gpu_world_size)
        return self._gloo_grad_reduce_group

    def _setup_gloo_barrier_group(self):
        if self._gloo_barrier_ready:
            return self._gloo_barrier_group
        self._gloo_barrier_ready = True
        if not self.is_multi_gpu or not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return None
        if self._gloo_grad_reduce_group is not None:
            self._gloo_barrier_group = self._gloo_grad_reduce_group
        else:
            self._gloo_barrier_group = torch.distributed.new_group(
                ranks=list(range(self.gpu_world_size)),
                backend="gloo",
            )
        if self.is_main_process:
            logger.info("Gloo distributed barrier enabled across {} ranks.", self.gpu_world_size)
        return self._gloo_barrier_group

    def _hierarchical_grad_reduce_cpu_leader_enabled(self) -> bool:
        return os.environ.get("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _setup_hierarchical_grad_reduce_groups(self) -> bool:
        if self._hierarchical_grad_reduce_ready:
            return self._hierarchical_grad_reduce_available

        self._hierarchical_grad_reduce_ready = True
        topology_local_rank = getattr(self, "gpu_topology_local_rank", None)
        if topology_local_rank is None:
            topology_local_rank = self._topology_local_rank()
        topology_local_rank = int(topology_local_rank)
        if (
            not self.is_multi_gpu
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or self.gpu_local_world_size <= 1
            or self.gpu_world_size <= self.gpu_local_world_size
            or self.gpu_world_size % self.gpu_local_world_size != 0
        ):
            return False

        if topology_local_rank >= self.gpu_local_world_size:
            raise RuntimeError(
                "Hierarchical gradient reduction received an invalid local topology: "
                f"local_rank={topology_local_rank}, local_world_size={self.gpu_local_world_size}."
            )
        expected_local_rank = self.gpu_global_rank % self.gpu_local_world_size
        if topology_local_rank != expected_local_rank:
            raise RuntimeError(
                "Hierarchical gradient reduction requires contiguous torchrun ranks on each node: "
                f"global_rank={self.gpu_global_rank}, expected_local_rank={expected_local_rank}, "
                f"actual_local_rank={topology_local_rank}, local_world_size={self.gpu_local_world_size}."
            )

        node_count = self.gpu_world_size // self.gpu_local_world_size
        local_node_idx = self.gpu_global_rank // self.gpu_local_world_size
        self._hierarchical_grad_reduce_cpu_leader = self._hierarchical_grad_reduce_cpu_leader_enabled()
        hierarchical_small_collectives = self._hierarchical_small_collectives_enabled()
        subgroup_timeout = self._hierarchical_pg_timeout()
        for node_idx in range(node_count):
            start_rank = node_idx * self.gpu_local_world_size
            local_ranks = list(range(start_rank, start_rank + self.gpu_local_world_size))
            # Bind the payload group explicitly to NCCL instead of inheriting
            # the default process-group backend.  This lets the job use a
            # 104-rank Gloo default group while creating NCCL communicators
            # only for the eight GPUs that share one node.
            local_group = torch.distributed.new_group(
                ranks=local_ranks,
                backend="nccl",
                timeout=subgroup_timeout,
            )
            local_barrier_group = torch.distributed.new_group(
                ranks=local_ranks,
                backend="gloo",
                timeout=subgroup_timeout,
            )
            if node_idx == local_node_idx:
                self._hierarchical_local_group = local_group
                self._hierarchical_local_barrier_group = local_barrier_group
                self._hierarchical_local_leader_rank = start_rank

        leader_ranks = list(range(0, self.gpu_world_size, self.gpu_local_world_size))
        if self._hierarchical_grad_reduce_cpu_leader:
            # Do not even instantiate a cross-node NCCL communicator in the
            # CPU-leader mode.  Only the eight ranks within a node communicate
            # through NCCL; the node leaders exchange the flattened payload on
            # CPU through this 13-rank Gloo group.
            self._hierarchical_leader_gloo_group = torch.distributed.new_group(
                ranks=leader_ranks,
                backend="gloo",
                timeout=subgroup_timeout,
            )
        else:
            self._hierarchical_leader_group = torch.distributed.new_group(
                ranks=leader_ranks,
                backend="nccl",
                timeout=subgroup_timeout,
            )
            if hierarchical_small_collectives:
                # GPU leader gradient reduction and CPU leader control
                # reduction intentionally use distinct groups.  All ranks
                # create both groups in the same order; only leaders enter
                # their collectives at runtime.
                self._hierarchical_leader_gloo_group = torch.distributed.new_group(
                    ranks=leader_ranks,
                    backend="gloo",
                    timeout=subgroup_timeout,
                )
        self._hierarchical_is_leader_rank = self.gpu_global_rank in leader_ranks
        leader_group_available = (
            self._hierarchical_leader_gloo_group is not None
            if self._hierarchical_grad_reduce_cpu_leader
            else self._hierarchical_leader_group is not None
        )
        self._hierarchical_grad_reduce_available = (
            self._hierarchical_local_group is not None
            and self._hierarchical_local_barrier_group is not None
            and leader_group_available
        )
        if self.is_main_process:
            logger.info(
                "Hierarchical gradient reduce enabled: world_size={} local_world_size={} nodes={} "
                "cpu_leader={} hierarchical_small_collectives={} subgroup_timeout_sec={}",
                self.gpu_world_size,
                self.gpu_local_world_size,
                node_count,
                self._hierarchical_grad_reduce_cpu_leader,
                hierarchical_small_collectives,
                int(subgroup_timeout.total_seconds()),
            )
        return self._hierarchical_grad_reduce_available

    def _all_reduce_grad_payload(self, payload: torch.Tensor) -> str:
        gloo_grad_reduce = self._gloo_grad_reduce_enabled()
        hierarchical_grad_reduce = self._hierarchical_grad_reduce_enabled()
        cpu_leader = self._hierarchical_grad_reduce_cpu_leader_enabled()
        if gloo_grad_reduce and hierarchical_grad_reduce:
            raise RuntimeError(
                "HOLOSOMA_GLOO_GRAD_REDUCE and HOLOSOMA_HIERARCHICAL_GRAD_REDUCE are mutually exclusive."
            )
        if cpu_leader and not hierarchical_grad_reduce:
            raise RuntimeError(
                "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER requires "
                "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1."
            )

        if gloo_grad_reduce:
            gloo_group = self._setup_gloo_grad_reduce_group()
            if gloo_group is not None:
                cpu_payload = payload.detach().cpu()
                torch.distributed.all_reduce(
                    cpu_payload,
                    op=torch.distributed.ReduceOp.SUM,
                    group=gloo_group,
                )
                payload.copy_(cpu_payload.to(device=payload.device, dtype=payload.dtype))
                return "gloo_cpu"

        if hierarchical_grad_reduce:
            if not self._setup_hierarchical_grad_reduce_groups():
                raise RuntimeError(
                    "Hierarchical gradient reduction was requested but the distributed topology is unsupported: "
                    f"world_size={self.gpu_world_size}, local_world_size={self.gpu_local_world_size}."
                )
            # The payload was assembled by CUDA kernels immediately before
            # this call.  Fence packing before the first NCCL operation, then
            # fence each backend boundary so every node executes the same
            # reduce -> leader all-reduce -> broadcast sequence.
            if payload.device.type == "cuda":
                torch.cuda.synchronize(payload.device)
            torch.distributed.reduce(
                payload,
                dst=self._hierarchical_local_leader_rank,
                op=torch.distributed.ReduceOp.SUM,
                group=self._hierarchical_local_group,
            )
            if payload.device.type == "cuda":
                torch.cuda.synchronize(payload.device)
            if self._hierarchical_is_leader_rank:
                if self._hierarchical_grad_reduce_cpu_leader:
                    cpu_payload = payload.detach().cpu()
                    torch.distributed.all_reduce(
                        cpu_payload,
                        op=torch.distributed.ReduceOp.SUM,
                        group=self._hierarchical_leader_gloo_group,
                    )
                    payload.copy_(cpu_payload.to(device=payload.device, dtype=payload.dtype))
                    if payload.device.type == "cuda":
                        torch.cuda.synchronize(payload.device)
                else:
                    torch.distributed.all_reduce(
                        payload,
                        op=torch.distributed.ReduceOp.SUM,
                        group=self._hierarchical_leader_group,
                    )
                    if payload.device.type == "cuda":
                        torch.cuda.synchronize(payload.device)
            torch.distributed.barrier(group=self._hierarchical_local_barrier_group)
            if payload.device.type == "cuda":
                torch.cuda.synchronize(payload.device)
            torch.distributed.broadcast(
                payload,
                src=self._hierarchical_local_leader_rank,
                group=self._hierarchical_local_group,
            )
            if payload.device.type == "cuda":
                torch.cuda.synchronize(payload.device)
            if self._hierarchical_grad_reduce_cpu_leader:
                return "hierarchical_cpu_leader"
            return "hierarchical"

        # ``all_reduce`` on an NCCL process group enqueues work on a CUDA
        # stream even when ``async_op`` is false.  The flattened payload was
        # just assembled with zeros/copy/cat kernels, so the earlier
        # post-backward device boundary cannot order this newly enqueued work
        # against NCCL on every rank.  At 104 ranks this manifested as a
        # different single rank occasionally blocking while the other ranks
        # advanced to the next collective.  Match the independently verified
        # NCCL stress path exactly: finish payload packing before enqueueing,
        # then wait for collective completion before any rank can repack or
        # enter the next minibatch.
        if payload.device.type == "cuda":
            torch.cuda.synchronize(payload.device)
        torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.SUM)
        if payload.device.type == "cuda":
            torch.cuda.synchronize(payload.device)
        return "flat"

    @staticmethod
    def _module_synchronization_entries(role: str, module: nn.Module) -> list[tuple[str, torch.Tensor]]:
        entries = [
            (f"{role}.parameter.{name}", parameter.data)
            for name, parameter in module.named_parameters()
        ]
        entries.extend(
            (f"{role}.buffer.{name}", buffer.data)
            for name, buffer in module.named_buffers()
        )
        return entries

    @staticmethod
    def _model_synchronization_digest(
        entries: list[tuple[str, torch.Tensor]],
        *,
        include_values: bool,
    ) -> bytes:
        digest = hashlib.sha256()
        for name, tensor in entries:
            descriptor = {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "layout": str(tensor.layout),
            }
            digest.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\0")
            if include_values:
                if tensor.layout != torch.strided:
                    raise ValueError(
                        f"Distributed model synchronization does not support non-strided tensor {name}: "
                        f"layout={tensor.layout}."
                    )
                raw = (
                    tensor.detach()
                    .cpu()
                    .contiguous()
                    .reshape(-1)
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
                digest.update(raw)
            digest.update(b"\xff")
        return digest.digest()

    def _require_rank0_model_digest(
        self,
        local_digest: bytes,
        *,
        phase: str,
        device: torch.device,
    ) -> None:
        if int(getattr(self, "gpu_world_size", 1) or 1) <= 1:
            return
        expected = torch.zeros(32, dtype=torch.uint8, device=device)
        if int(getattr(self, "gpu_global_rank", 0) or 0) == 0:
            expected.copy_(torch.tensor(list(local_digest), dtype=torch.uint8, device=device))
        self._broadcast_tensor(expected, src=0)
        mismatch = torch.tensor(
            [0.0 if bytes(expected.detach().cpu().tolist()) == local_digest else 1.0],
            dtype=torch.float32,
            device=device,
        )
        mismatch = self._all_reduce_small_tensor(mismatch, op=torch.distributed.ReduceOp.SUM)
        mismatch_count = int(mismatch.item())
        if mismatch_count:
            raise RuntimeError(
                f"Distributed model {phase} differs from rank 0 on {mismatch_count}/"
                f"{self.gpu_world_size} ranks."
            )

    def _synchronize_model_weights(self):
        """Synchronize every actor/critic parameter and buffer across GPUs."""
        debug_heartbeat = os.environ.get("HOLOSOMA_DEBUG_HEARTBEAT", "").lower() not in ("", "0", "false", "no")
        skip_critic_weight_sync = os.environ.get("HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        entries = self._module_synchronization_entries("actor", self.actor)
        if not skip_critic_weight_sync:
            entries.extend(self._module_synchronization_entries("critic", self.critic))
        if not entries:
            raise RuntimeError("Actor/critic model synchronization found no parameters or buffers.")
        digest_device = entries[0][1].device

        schema_digest = self._model_synchronization_digest(entries, include_values=False)
        self._require_rank0_model_digest(
            schema_digest,
            phase="schema",
            device=digest_device,
        )
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} broadcast full model state begin", self.gpu_global_rank)
        for _, tensor in entries:
            self._broadcast_tensor(tensor, src=0)
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} broadcast full model state finished", self.gpu_global_rank)

        state_digest = self._model_synchronization_digest(entries, include_values=True)
        self._require_rank0_model_digest(
            state_digest,
            phase="state after broadcast",
            device=digest_device,
        )

        if skip_critic_weight_sync:
            logger.info(
                "Synchronized actor parameters and buffers across {} GPUs; skipped critic state sync",
                self.gpu_world_size,
            )
        else:
            logger.info(
                "Synchronized model parameters and buffers across {} GPUs and verified state digests",
                self.gpu_world_size,
            )

    def _normalize_advantages_multi_gpu(self, advantages):
        weight = self._get_distributed_loss_weight()
        local_stats = torch.stack(
            [
                advantages.mean() * weight,
                (advantages**2).mean() * weight,
                advantages.new_tensor(weight),
            ]
        )
        local_stats = self._all_reduce_small_tensor(local_stats, op=torch.distributed.ReduceOp.SUM)

        total_weight = local_stats[2]
        if total_weight <= 0.0:
            raise ValueError("At least one rank must have a positive distributed_loss_weight.")
        expected_weight = float(getattr(self, "gpu_world_size", 0) or 0)
        if expected_weight > 0.0 and not math.isclose(
            float(total_weight.item()),
            expected_weight,
            rel_tol=1.0e-5,
            abs_tol=1.0e-5,
        ):
            raise ValueError(
                "Distributed loss weights must sum to world_size before optimization because gradient "
                f"reduction divides by world_size: sum={float(total_weight.item())}, "
                f"world_size={expected_weight}."
            )
        global_mean = local_stats[0] / total_weight
        global_sq_mean = local_stats[1] / total_weight
        global_variance = torch.clamp(global_sq_mean - global_mean**2, min=0.0)
        global_std = torch.sqrt(global_variance + 1e-8)

        return (advantages - global_mean) / global_std

    ##########################################################################################
    # Code for Evaluation
    ##########################################################################################

    @property
    def actor_onnx_wrapper(self):
        self._validate_actor_onnx_compatibility()

        class ActorWrapper(nn.Module):
            def __init__(self, actor, normalizers, keys, slices, perception_key):
                super().__init__()
                self.actor = actor
                self.keys = keys
                self.slices = slices
                self.perception_key = perception_key
                self.normalizers = nn.ModuleDict({key: normalizers[key] for key in keys})

            def forward(self, actor_obs, perception_obs=None):
                parts = []
                for key in self.keys:
                    part = actor_obs[..., self.slices[key]]
                    normalizer = self.normalizers[key]
                    if isinstance(normalizer, EmpiricalNormalization):
                        part = normalizer(part, update=False)
                    else:
                        part = normalizer(part)
                    parts.append(part)
                actor_obs = torch.cat(parts, dim=-1)
                policy_state = {"actor_obs": actor_obs}
                if self.perception_key and perception_obs is not None:
                    policy_state[self.perception_key] = perception_obs
                return self.actor.act_inference(policy_state)

        return ActorWrapper(
            self.actor,
            self.actor_obs_normalizers,
            self.actor_obs_keys,
            self.actor_obs_slices,
            self.actor_perception_key,
        )

    def env_step(self, actor_state):
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update({"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras})
        return actor_state

    @torch.no_grad()
    def get_example_obs(self):
        """Used for exporting policy as onnx."""
        obs_dict = self.env.reset_all()
        example = {
            "actor_obs": torch.cat([obs_dict[k] for k in self.actor_obs_keys], dim=1),
            "critic_obs": torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1),
        }
        if self.actor_perception_key and self.actor_perception_key in obs_dict:
            example[self.actor_perception_key] = obs_dict[self.actor_perception_key]
        return example

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None):
        self._create_eval_callbacks()
        obs_dict = self._pre_evaluate_policy()
        actor_state = self._create_actor_state()
        self.eval_policy = self.get_inference_policy()

        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})

        critic_obs = torch.cat([actor_state["obs"][k] for k in self.critic_obs_keys], dim=1)
        actor_state["obs"]["critic_obs"] = critic_obs

        for step in itertools.islice(itertools.count(), max_eval_steps):
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)

        self._post_evaluate_policy()

    def _create_actor_state(self):
        return {"done_indices": [], "stop": False}

    def _create_eval_callbacks(self):
        # evaluate_policy() may be called repeatedly in one process.  Callback
        # instances are scoped to one evaluation run and must not accumulate.
        self.eval_callbacks.clear()
        if self.config.eval_callbacks is not None:
            for cb in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb], training_loop=self))

    def _pre_evaluate_policy(self, reset_env=True):
        self._eval_mode()
        self.env.set_is_evaluating()
        obs_dict = None
        if reset_env:
            obs_dict = self.env.reset_all()
            reset_mask = torch.ones(self.env.num_envs, device=self.device, dtype=torch.bool)
            self.actor.reset(reset_mask)
            self.critic.reset(reset_mask)
            if getattr(self, "_evaluation_policy_mode", "checkpoint_actor") \
                    in {
                        "distill_label_teacher",
                        "distill_label_teacher_bc_target",
                    }:
                assert self.teacher_actor is not None
                self.teacher_actor.reset(reset_mask)

        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()
        return obs_dict

    def _post_evaluate_policy(self):
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _maybe_debug_eval_policy_io(
        self,
        *,
        step: int | None,
        actor_obs_raw: torch.Tensor,
        actor_obs: torch.Tensor,
        policy_state: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> None:
        debug_path = os.environ.get("HOLOSOMA_EVAL_DEBUG_PATH", "").strip()
        if not debug_path:
            return
        debug_limit = int(os.environ.get("HOLOSOMA_EVAL_DEBUG_LIMIT", "12"))
        debug_count = int(getattr(self, "_eval_debug_count", 0))
        if debug_count >= debug_limit:
            return

        path = Path(debug_path)
        if not getattr(self, "_eval_debug_initialized", False):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
            self._eval_debug_initialized = True

        perception_obs = policy_state.get(self.actor_perception_key) if self.actor_perception_key else None
        torque_record: dict[str, list[float] | float | int] = {}
        action_term = None
        action_manager = getattr(self.env, "action_manager", None)
        if action_manager is not None and hasattr(action_manager, "get_term"):
            try:
                action_term = action_manager.get_term("joint_control")
            except Exception:
                action_term = None
        if action_term is not None:
            with torch.no_grad():
                actions_scaled = actions * action_term.action_scales
                control_type = self.env.robot_config.control.control_type
                if control_type == "P":
                    torques_unclipped = (
                        action_term._kp_scale
                        * action_term.p_gains
                        * (actions_scaled + self.env.default_dof_pos - self.env.simulator.dof_pos)
                        - action_term._kd_scale * action_term.d_gains * self.env.simulator.dof_vel
                    )
                elif control_type == "V":
                    torques_unclipped = (
                        action_term._kp_scale * action_term.p_gains * (actions_scaled - self.env.simulator.dof_vel)
                        - action_term._kd_scale
                        * action_term.d_gains
                        * (self.env.simulator.dof_vel - action_term._prev_dof_vel)
                        / self.env.sim_dt
                    )
                elif control_type == "T":
                    torques_unclipped = actions_scaled
                else:
                    torques_unclipped = None

                if torques_unclipped is not None:
                    torques_clipped = torques_unclipped
                    if self.env.robot_config.control.clip_torques:
                        torques_clipped = torch.clip(torques_clipped, -self.env.torque_limits, self.env.torque_limits)
                    sat_ratio = torch.abs(torques_clipped) / torch.clamp(self.env.torque_limits, min=1.0e-6)
                    torque_record = {
                        "torque_unclipped_values": torques_unclipped.detach().cpu().reshape(-1).to(torch.float32).tolist(),
                        "torque_clipped_values": torques_clipped.detach().cpu().reshape(-1).to(torch.float32).tolist(),
                        "torque_sat_ratio_values": sat_ratio.detach().cpu().reshape(-1).to(torch.float32).tolist(),
                        "torque_saturated_joint_count": int(
                            torch.count_nonzero(torch.abs(torques_unclipped) >= self.env.torque_limits - 1.0e-5).item()
                        ),
                    }
        record = {
            "count": debug_count,
            "step": None if step is None else int(step),
            "actor_obs_raw_values": actor_obs_raw.detach().cpu().reshape(-1).to(torch.float32).tolist(),
            "actor_obs_norm_values": actor_obs.detach().cpu().reshape(-1).to(torch.float32).tolist(),
            "perception_obs_values": (
                None
                if perception_obs is None
                else perception_obs.detach().cpu().reshape(-1).to(torch.float32).tolist()
            ),
            "action_values": actions.detach().cpu().reshape(-1).to(torch.float32).tolist(),
        }
        simulator = getattr(self.env, "simulator", None)
        if simulator is not None:
            try:
                record["sim_time_ms"] = float(simulator.time()) * 1000.0
            except Exception:
                pass
            try:
                record["robot_root_state"] = simulator.robot_root_states[0].detach().cpu().reshape(-1).to(torch.float32).tolist()
                record["robot_dof_pos"] = simulator.dof_pos[0].detach().cpu().reshape(-1).to(torch.float32).tolist()
                record["robot_dof_vel"] = simulator.dof_vel[0].detach().cpu().reshape(-1).to(torch.float32).tolist()
            except Exception:
                pass
            actor_states: dict[str, list[float]] = {}
            try:
                env_ids = torch.tensor([0], device=simulator.device, dtype=torch.long)
                actor_metadata = getattr(simulator, "_actor_root_metadata", {})
                if isinstance(actor_metadata, dict) and actor_metadata:
                    actor_names = [name for name in actor_metadata if name != "robot"]
                else:
                    actor_names = list(getattr(simulator, "_object_urdf_by_name", {}).keys())
                for name in actor_names:
                    try:
                        actor_state = simulator.get_actor_states([name], env_ids)
                    except Exception:
                        continue
                    if actor_state.numel() == 0:
                        continue
                    actor_states[str(name)] = actor_state[0].detach().cpu().reshape(-1).to(torch.float32).tolist()
            except Exception:
                actor_states = {}
            if actor_states:
                record["actors"] = actor_states
        perception_manager = getattr(self.env, "perception_manager", None)
        if perception_manager is not None and getattr(perception_manager, "enabled", False):
            try:
                env_ids = torch.tensor([0], device=perception_manager.device, dtype=torch.long)
                cam_body_pos, cam_body_quat = perception_manager.get_camera_pose(
                    env_ids,
                    apply_sensor_offset=False,
                    apply_pitch=False,
                )
                record["camera_body_pose_pos"] = (
                    cam_body_pos[0].detach().cpu().reshape(-1).to(torch.float32).tolist()
                )
                record["camera_body_pose_quat_xyzw"] = (
                    cam_body_quat[0].detach().cpu().reshape(-1).to(torch.float32).tolist()
                )
            except Exception:
                pass
            try:
                cam_pos, cam_quat = perception_manager.get_camera_pose(
                    env_ids,
                    apply_sensor_offset=True,
                    apply_pitch=True,
                )
                record["camera_pose_pos"] = cam_pos[0].detach().cpu().reshape(-1).to(torch.float32).tolist()
                record["camera_pose_quat_xyzw"] = cam_quat[0].detach().cpu().reshape(-1).to(torch.float32).tolist()
            except Exception:
                pass
        record.update(torque_record)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._eval_debug_count = debug_count + 1

    def _pre_eval_env_step(self, actor_state: dict):
        actor_obs_raw = torch.cat([actor_state["obs"][k] for k in self.actor_obs_keys], dim=1)
        evaluation_policy_mode = getattr(
            self,
            "_evaluation_policy_mode",
            "checkpoint_actor",
        )
        if evaluation_policy_mode in {
            "distill_label_teacher",
            "distill_label_teacher_bc_target",
        }:
            teacher_obs_raw = self._build_teacher_obs_raw(actor_state["obs"], actor_obs_raw)
            teacher_obs = self._normalize_teacher_actor_obs(teacher_obs_raw)
            actions, _teacher_indices = self._select_teacher_actions(
                teacher_obs_raw,
                actor_state["obs"],
                stochastic=False,
            )
            if bool(
                getattr(
                    self,
                    "_evaluation_teacher_action_clip_enabled",
                    False,
                )
            ):
                clip_threshold = getattr(
                    self,
                    "_evaluation_teacher_action_clip_threshold",
                    None,
                )
                if clip_threshold is None:
                    raise RuntimeError(
                        "Clipped distillation-label teacher evaluation is missing "
                        "its authenticated BC target threshold."
                    )
                actions = self._clip_teacher_actions_preserving_non_finite(
                    actions,
                    float(clip_threshold),
                )
            policy_state = {"actor_obs": teacher_obs}
            actor_obs_raw_for_debug = teacher_obs_raw
            actor_obs_for_debug = teacher_obs
        else:
            actor_obs = self._normalize_actor_obs(actor_obs_raw, update=False)
            policy_state = {"actor_obs": actor_obs}
            if self.actor_perception_key and self.actor_perception_key in actor_state["obs"]:
                policy_state[self.actor_perception_key] = actor_state["obs"][self.actor_perception_key]
            # policy_state is already normalized here; invoke the actor directly so
            # get_inference_policy() cannot normalize it a second time.
            actions = self.actor.act_inference(policy_state)
            actor_obs_raw_for_debug = actor_obs_raw
            actor_obs_for_debug = actor_obs
        self._maybe_debug_eval_policy_io(
            step=actor_state.get("step"),
            actor_obs_raw=actor_obs_raw_for_debug,
            actor_obs=actor_obs_for_debug,
            policy_state=policy_state,
            actions=actions,
        )
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state):
        dones = actor_state.get("dones")
        if dones is not None:
            self.actor.reset(dones)
            self.critic.reset(dones)
            if getattr(self, "_evaluation_policy_mode", "checkpoint_actor") \
                    in {
                        "distill_label_teacher",
                        "distill_label_teacher_bc_target",
                    }:
                assert self.teacher_actor is not None
                self.teacher_actor.reset(dones)
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state

    def get_inference_policy(self, device=None):
        self.actor.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.actor.to(device)
            for normalizer in self.actor_obs_normalizers.values():
                normalizer.to(device)
        for normalizer in self.actor_obs_normalizers.values():
            normalizer.eval()

        def _policy(obs_dict):
            actor_obs = obs_dict["actor_obs"]
            actor_obs = self._normalize_actor_obs(actor_obs, update=False)
            policy_state = {"actor_obs": actor_obs}
            if self.actor_perception_key and self.actor_perception_key in obs_dict:
                policy_state[self.actor_perception_key] = obs_dict[self.actor_perception_key]
            return self.actor.act_inference(policy_state)

        return _policy
