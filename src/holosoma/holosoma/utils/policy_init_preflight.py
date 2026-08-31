"""Fail-closed semantic validation for actor-only policy initialization.

The preflight runs in a short-lived process before Isaac is imported.  A
strict state-dict load proves tensor compatibility later, while this module
proves that equal-shaped tensors still describe the same observation-to-action
mapping.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from holosoma.utils.checkpoint_validation import (
    checkpoint_saved_run_target,
    load_verified_torch_checkpoint,
    require_mapping,
    validate_checkpoint_iterations,
    validate_finite_tree,
    validate_terminal_fixed_bc_eval_artifact_payload,
)
from holosoma.utils.training_provenance import (
    parse_training_provenance,
    validate_training_provenance,
)


ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV = (
    "HOLOSOMA_ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD"
)
POLICY_INIT_REQUIRED_TERMINAL_TARGET_ENV = (
    "HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET"
)
TRACKING_TO_PRECOMPUTED_DROP_EXCLUSIVE_MIGRATION = (
    "tracking_error_to_precomputed_turn_then_forward_drop_exclusive_v1"
)
PRECOMPUTED_TO_HMI_TERMINAL_GOAL_MIGRATION = (
    "precomputed_turn_then_forward_to_hmi_terminal_goal_unfreeze_native_depth_v1"
)


def allow_legacy_unverified_policy_load() -> bool:
    """Return the exact, explicitly non-scientific legacy policy-load hatch."""

    raw_value = os.environ.get(ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV)
    if raw_value is None or raw_value.strip() in ("", "0"):
        return False
    if raw_value.strip() == "1":
        return True
    raise ValueError(
        f"{ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV} must be exactly 0 or 1; "
        f"got {raw_value!r}."
    )


def required_policy_init_terminal_target_from_env(
    environ: Mapping[str, str] | None = None,
) -> int | None:
    """Parse the worker's exact terminal-source requirement.

    An explicitly empty value is equivalent to unset so generic/legacy policy
    initialization remains available.  Any enabled value is canonical and
    positive; permissive integer parsing would let launch aliases disagree
    while appearing numerically equal.
    """

    if environ is None:
        environ = os.environ
    raw_value = environ.get(POLICY_INIT_REQUIRED_TERMINAL_TARGET_ENV)
    if raw_value is None or raw_value == "":
        return None
    if (
        raw_value != raw_value.strip()
        or not raw_value.isascii()
        or not raw_value.isdecimal()
        or raw_value.startswith("0")
    ):
        raise ValueError(
            f"{POLICY_INIT_REQUIRED_TERMINAL_TARGET_ENV} must be a canonical "
            f"ASCII positive integer, got {raw_value!r}."
        )
    target = int(raw_value, 10)
    if target < 1:
        raise ValueError(
            f"{POLICY_INIT_REQUIRED_TERMINAL_TARGET_ENV} must be positive."
        )
    return target


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Policy-init config {path} must be a mapping, got {type(value).__name__}.")
    return value


def _require_path(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    traversed: list[str] = []
    for key in path:
        traversed.append(key)
        if not isinstance(value, dict) or key not in value:
            raise ValueError(
                "Policy-init config is missing required actor-contract field " + ".".join(traversed) + "."
            )
        value = value[key]
    return value


def _json_value(value: Any) -> Any:
    """Normalize tuple/list details without erasing mapping insertion order."""

    if isinstance(value, dict):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Policy-init actor contract contains a non-serializable value: {value!r}.")


def _ordered_observation_group(group_name: str, raw_group: Any) -> dict[str, Any]:
    group = _require_mapping(raw_group, f"observation.groups.{group_name}")
    terms = _require_mapping(
        _require_path(group, ("terms",)),
        f"observation.groups.{group_name}.terms",
    )
    ordered_terms: list[dict[str, Any]] = []
    required_term_fields = ("func", "params", "scale", "noise", "clip")
    for term_name, raw_term in terms.items():
        term = _require_mapping(raw_term, f"observation.groups.{group_name}.terms.{term_name}")
        contract: dict[str, Any] = {"name": str(term_name)}
        for field in required_term_fields:
            if field not in term:
                raise ValueError(
                    "Policy-init config is missing required actor-contract field "
                    f"observation.groups.{group_name}.terms.{term_name}.{field}."
                )
            contract[field] = _json_value(term[field])
        ordered_terms.append(contract)

    required_group_fields = ("concatenate", "enable_noise", "history_length")
    result: dict[str, Any] = {"name": group_name, "terms": ordered_terms}
    for field in required_group_fields:
        if field not in group:
            raise ValueError(
                "Policy-init config is missing required actor-contract field "
                f"observation.groups.{group_name}.{field}."
            )
        result[field] = _json_value(group[field])
    return result


def _canonical_actor_module(actor: dict[str, Any], *, actions_dim: int) -> dict[str, Any]:
    """Resolve symbolic dimensions and explicit current defaults for legacy checkpoints."""

    result = copy.deepcopy(actor)
    output_dim = result.get("output_dim")
    if not isinstance(output_dim, (list, tuple)):
        raise ValueError("Policy-init actor output_dim must be an ordered list.")
    result["output_dim"] = [actions_dim if value == "robot_action_dim" else value for value in output_dim]

    # These fields were added with None defaults after the box initializer was
    # written.  Absence in that checkpoint is exactly the dataclass default,
    # not permission to ignore a non-default current value.
    result.setdefault("min_noise_std", None)
    result.setdefault("min_mean_noise_std", None)
    result.setdefault("max_noise_std", None)

    layer_config = _require_mapping(result.get("layer_config"), "algo.config.module_dict.actor.layer_config")
    layer_defaults = {
        "flow_integration_steps": 4,
        "flow_train_noise_std": 1.0,
        "flow_time_epsilon": 1e-4,
        "flow_inference_noise_std": 0.0,
        # This authenticated-local-weight identity was added after the legacy
        # box initializers were written.  Missing meant the same thing as the
        # current dataclass default: no pretrained perception payload.
        "perception_pretrained_sha256": None,
    }
    for field, default in layer_defaults.items():
        layer_config.setdefault(field, default)
    return _json_value(result)


def _canonical_perception_config(
    config: dict[str, Any],
    raw_perception: Any,
) -> dict[str, Any]:
    """Materialize only historically exact perception defaults.

    The digest field below did not exist in legacy serialized configs and its
    absence was exactly the current ``None`` default.  A missing reset
    lifecycle is *legacy_full_v1*, never targeted_v2; keeping that distinction
    makes an old initializer fail closed when a current run requests the new
    targeted reset producer.

    Historical absence of camera_warp_hole_reference_batch_size bound the
    Perlin field to that producer's live batch, as does current ``None``.  The
    effective integer therefore comes from this config's already-localized
    training.num_envs; preserving that number detects a 4096-env legacy field
    being reused in a 64-env rank even though both raw values appear null.
    """

    perception = copy.deepcopy(
        _require_mapping(raw_perception, "perception")
    )
    holes_active = bool(perception.get("camera_warp_enable_holes", False)) and float(
        perception.get("camera_warp_hole_prob", 0.0) or 0.0
    ) > 0.0
    if holes_active:
        training = _require_mapping(_require_path(config, ("training",)), "training")
        live_batch_size = _require_path(training, ("num_envs",))
        if (
            isinstance(live_batch_size, bool)
            or not isinstance(live_batch_size, int)
            or live_batch_size <= 0
        ):
            raise ValueError(
                "Policy-init training.num_envs must be a positive integer for the "
                f"perception contract, got {live_batch_size!r}."
            )
        hole_reference_batch_size = perception.get(
            "camera_warp_hole_reference_batch_size"
        )
        if hole_reference_batch_size is None:
            hole_reference_batch_size = live_batch_size
        if (
            isinstance(hole_reference_batch_size, bool)
            or not isinstance(hole_reference_batch_size, int)
            or hole_reference_batch_size < live_batch_size
        ):
            raise ValueError(
                "Policy-init perception.camera_warp_hole_reference_batch_size must be "
                "an integer no smaller than effective training.num_envs "
                f"({live_batch_size}), got {hole_reference_batch_size!r}."
            )
        perception["camera_warp_hole_reference_batch_size"] = hole_reference_batch_size
    else:
        # The generator is not constructed, so neither the configured
        # reference nor the live batch can affect actor observations.
        perception["camera_warp_hole_reference_batch_size"] = None
    perception.setdefault("encoder_pretrained_sha256", None)
    perception.setdefault("reset_refresh_semantics", "legacy_full_v1")
    # Missing fields describe the exact historical producer.  In particular,
    # never relabel an old actor as rank-local-v2 merely because its serialized
    # config predates the seed contract.
    perception.setdefault("camera_warp_hole_seed_semantics", "legacy_fixed_v1")
    perception.setdefault(
        "camera_warp_hole_octave_profile",
        "legacy_single_octave_v1",
    )
    return _json_value(perception)


def _contact_aware_command_contract(config: dict[str, Any], ordered_groups: list[dict[str, Any]]) -> dict[str, Any] | None:
    contact_aware_func = "sparse_target_root_trajectory_command_contact_aware"
    actor_observation_funcs = {
        str(term.get("func", "")).rsplit(":", 1)[-1]
        for group in ordered_groups
        for term in group["terms"]
    }
    uses_contact_aware_root_command = contact_aware_func in actor_observation_funcs
    uses_button_window = bool(
        actor_observation_funcs.intersection({"drop_button", "pickup_button"})
    )
    uses_contact_window = bool(
        actor_observation_funcs.intersection({contact_aware_func, "drop_button", "pickup_button"})
    )
    if not uses_contact_window:
        return None

    command = _require_mapping(_require_path(config, ("command",)), "command")
    setup_terms = _require_mapping(_require_path(command, ("setup_terms",)), "command.setup_terms")
    motion_term = _require_mapping(
        _require_path(setup_terms, ("motion_command",)),
        "command.setup_terms.motion_command",
    )
    params = _require_mapping(
        _require_path(motion_term, ("params",)),
        "command.setup_terms.motion_command.params",
    )
    motion_config = _require_mapping(
        _require_path(params, ("motion_config",)),
        "command.setup_terms.motion_command.params.motion_config",
    )
    raw_mode = str(motion_config.get("contact_aware_sparse_root_command_mode", "tracking_error"))
    mode = raw_mode.strip().lower().replace("-", "_")
    if mode in {"tracking", "default", "robot_tracking_error"}:
        mode = "tracking_error"
    elif mode in {"segment", "segment_30"}:
        mode = "t1_aligned_segment"

    carry_window_mode = str(motion_config.get("contact_aware_carry_window_mode", "rel_z"))
    carry_window_mode = carry_window_mode.strip().lower().replace("-", "_")
    result: dict[str, Any] = {"contact_aware_carry_window_mode": carry_window_mode}
    if uses_button_window:
        button_window_mode = motion_config.get(
            "contact_aware_button_window_mode",
            "contact_interval",
        )
        if not isinstance(button_window_mode, str) or button_window_mode not in {
            "contact_interval",
            "kinematic_lift",
        }:
            raise ValueError(
                "command.setup_terms.motion_command.params.motion_config."
                "contact_aware_button_window_mode must be exactly "
                f"'contact_interval' or 'kinematic_lift', got {button_window_mode!r}."
            )
        result["contact_aware_button_window_mode"] = button_window_mode
    if carry_window_mode == "peak_height":
        result["contact_aware_peak_height_alpha"] = _json_value(
            motion_config.get("contact_aware_peak_height_alpha", 0.91)
        )
        result["contact_aware_peak_height_smoothing_steps"] = _json_value(
            motion_config.get("contact_aware_peak_height_smoothing_steps", 5)
        )
    if uses_contact_aware_root_command:
        result["contact_aware_sparse_root_command_mode"] = mode
        zero_root_when_drop_active = motion_config.get(
            "zero_root_command_when_drop_active",
            False,
        )
        if type(zero_root_when_drop_active) is not bool:
            raise ValueError(
                "command.setup_terms.motion_command.params.motion_config."
                "zero_root_command_when_drop_active must be boolean, got "
                f"{zero_root_when_drop_active!r}."
            )
        result["zero_root_command_when_drop_active"] = zero_root_when_drop_active
        if mode == "t1_aligned_segment":
            result["contact_aware_sparse_root_segment_steps"] = _json_value(
                motion_config.get("contact_aware_sparse_root_segment_steps", 30)
            )
            result["contact_aware_sparse_root_zero_yaw_threshold_deg"] = _json_value(
                motion_config.get("contact_aware_sparse_root_zero_yaw_threshold_deg", 0.0)
            )
    return result


def canonical_actor_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the ordered config contract that defines actor semantics."""

    config = _require_mapping(config, "experiment_config")
    actor = _require_mapping(
        _require_path(config, ("algo", "config", "module_dict", "actor")),
        "algo.config.module_dict.actor",
    )
    actor_inputs_raw = _require_path(actor, ("input_dim",))
    if not isinstance(actor_inputs_raw, (list, tuple)) or not all(
        isinstance(group, str) and group for group in actor_inputs_raw
    ):
        raise ValueError("Policy-init actor input_dim must be a non-empty ordered list of observation groups.")
    actor_inputs = list(actor_inputs_raw)
    if len(set(actor_inputs)) != len(actor_inputs):
        raise ValueError(f"Policy-init actor input_dim contains duplicate groups: {actor_inputs!r}.")

    layer_config = _require_mapping(
        _require_path(actor, ("layer_config",)),
        "algo.config.module_dict.actor.layer_config",
    )
    perception_key_raw = layer_config.get("perception_input_name", "")
    if perception_key_raw is None:
        perception_key_raw = ""
    if not isinstance(perception_key_raw, str):
        raise ValueError("Policy-init actor perception_input_name must be a string or null.")
    perception_key = perception_key_raw.strip()

    observation = _require_mapping(_require_path(config, ("observation",)), "observation")
    groups = _require_mapping(_require_path(observation, ("groups",)), "observation.groups")
    ordered_group_names = list(actor_inputs)
    if perception_key and perception_key not in ordered_group_names:
        ordered_group_names.append(perception_key)
    ordered_groups = []
    for group_name in ordered_group_names:
        if group_name not in groups:
            raise ValueError(
                f"Policy-init actor references observation group {group_name!r}, but it is missing from config."
            )
        ordered_groups.append(_ordered_observation_group(group_name, groups[group_name]))

    algo_config = _require_mapping(_require_path(config, ("algo", "config")), "algo.config")
    normalization = {}
    for field in ("normalize_actor_obs", "obs_normalizer_eps", "obs_normalizer_until"):
        if field not in algo_config:
            raise ValueError(f"Policy-init config is missing required actor-contract field algo.config.{field}.")
        normalization[field] = _json_value(algo_config[field])
    if not isinstance(normalization["normalize_actor_obs"], bool):
        raise ValueError("Policy-init algo.config.normalize_actor_obs must be boolean.")
    eps = normalization["obs_normalizer_eps"]
    if not isinstance(eps, (int, float)) or isinstance(eps, bool) or float(eps) <= 0.0:
        raise ValueError(f"Policy-init algo.config.obs_normalizer_eps must be positive, got {eps!r}.")
    until = normalization["obs_normalizer_until"]
    if until is not None and (
        not isinstance(until, int) or isinstance(until, bool) or until < 0
    ):
        raise ValueError(
            "Policy-init algo.config.obs_normalizer_until must be null or a non-negative integer, "
            f"got {until!r}."
        )

    if "clip_observations" not in observation:
        raise ValueError("Policy-init config is missing required actor-contract field observation.clip_observations.")

    robot = _require_mapping(_require_path(config, ("robot",)), "robot")
    robot_action_contract = {}
    for field in (
        "actions_dim",
        "dof_names",
        "dof_effort_limit_list",
        "init_state",
        "control",
    ):
        if field not in robot:
            raise ValueError(f"Policy-init config is missing required actor-contract field robot.{field}.")
        robot_action_contract[field] = _json_value(robot[field])
    actions_dim = robot_action_contract["actions_dim"]
    if not isinstance(actions_dim, int) or isinstance(actions_dim, bool) or actions_dim <= 0:
        raise ValueError(f"Policy-init robot.actions_dim must be a positive integer, got {actions_dim!r}.")

    action = _require_mapping(_require_path(config, ("action",)), "action")
    action_terms = _require_mapping(_require_path(action, ("terms",)), "action.terms")
    ordered_action_terms = [
        {"name": str(term_name), "config": _json_value(term_config)}
        for term_name, term_config in action_terms.items()
    ]

    perception = None
    if perception_key:
        perception = _canonical_perception_config(
            config,
            _require_path(config, ("perception",))
        )

    return {
        "algorithm": "ppo",
        "actor_module": _canonical_actor_module(actor, actions_dim=actions_dim),
        "actor_input_groups": actor_inputs,
        "observation_groups": ordered_groups,
        "observation_clip": _json_value(observation["clip_observations"]),
        "command_observation_semantics": _contact_aware_command_contract(config, ordered_groups),
        "normalization": normalization,
        "perception_input_name": perception_key,
        "perception": perception,
        "robot_action": robot_action_contract,
        "action_terms": ordered_action_terms,
    }


def canonical_critic_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the ordered config contract that defines a PPO value function."""

    config = _require_mapping(config, "experiment_config")
    critic = _require_mapping(
        _require_path(config, ("algo", "config", "module_dict", "critic")),
        "algo.config.module_dict.critic",
    )
    critic_inputs_raw = _require_path(critic, ("input_dim",))
    if not isinstance(critic_inputs_raw, (list, tuple)) or not all(
        isinstance(group, str) and group for group in critic_inputs_raw
    ):
        raise ValueError(
            "Stage-4 critic input_dim must be a non-empty ordered list of observation groups."
        )
    critic_inputs = list(critic_inputs_raw)
    if len(set(critic_inputs)) != len(critic_inputs):
        raise ValueError(
            f"Stage-4 critic input_dim contains duplicate groups: {critic_inputs!r}."
        )

    layer_config = _require_mapping(
        _require_path(critic, ("layer_config",)),
        "algo.config.module_dict.critic.layer_config",
    )
    perception_key_raw = layer_config.get("perception_input_name", "")
    if perception_key_raw is None:
        perception_key_raw = ""
    if not isinstance(perception_key_raw, str):
        raise ValueError("Stage-4 critic perception_input_name must be a string or null.")
    perception_key = perception_key_raw.strip()

    observation = _require_mapping(_require_path(config, ("observation",)), "observation")
    groups = _require_mapping(_require_path(observation, ("groups",)), "observation.groups")
    ordered_group_names = list(critic_inputs)
    if perception_key and perception_key not in ordered_group_names:
        ordered_group_names.append(perception_key)
    ordered_groups: list[dict[str, Any]] = []
    for group_name in ordered_group_names:
        if group_name not in groups:
            raise ValueError(
                f"Stage-4 critic references observation group {group_name!r}, "
                "but it is missing from config."
            )
        ordered_groups.append(_ordered_observation_group(group_name, groups[group_name]))

    algo_config = _require_mapping(_require_path(config, ("algo", "config")), "algo.config")
    normalization: dict[str, Any] = {}
    for field in ("normalize_critic_obs", "obs_normalizer_eps", "obs_normalizer_until"):
        if field not in algo_config:
            raise ValueError(
                f"Stage-4 config is missing required critic-contract field algo.config.{field}."
            )
        normalization[field] = _json_value(algo_config[field])
    if not isinstance(normalization["normalize_critic_obs"], bool):
        raise ValueError("Stage-4 algo.config.normalize_critic_obs must be boolean.")
    eps = normalization["obs_normalizer_eps"]
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or float(eps) <= 0.0:
        raise ValueError(f"Stage-4 algo.config.obs_normalizer_eps must be positive, got {eps!r}.")
    until = normalization["obs_normalizer_until"]
    if until is not None and (type(until) is not int or until < 0):
        raise ValueError(
            "Stage-4 algo.config.obs_normalizer_until must be null or a non-negative "
            f"integer, got {until!r}."
        )
    if "clip_observations" not in observation:
        raise ValueError("Stage-4 config is missing observation.clip_observations.")

    robot = _require_mapping(_require_path(config, ("robot",)), "robot")
    actions_dim = robot.get("actions_dim")
    if type(actions_dim) is not int or actions_dim <= 0:
        raise ValueError(
            f"Stage-4 robot.actions_dim must be a positive integer, got {actions_dim!r}."
        )
    perception = None
    if perception_key:
        perception = _canonical_perception_config(
            config,
            _require_path(config, ("perception",)),
        )
    return {
        "algorithm": "ppo",
        "critic_module": _canonical_actor_module(critic, actions_dim=actions_dim),
        "critic_input_groups": critic_inputs,
        "observation_groups": ordered_groups,
        "observation_clip": _json_value(observation["clip_observations"]),
        "command_observation_semantics": _contact_aware_command_contract(
            config,
            ordered_groups,
        ),
        "normalization": normalization,
        "perception_input_name": perception_key,
        "perception": perception,
    }


_FAST_SAC_ACTOR_CONFIG_FIELDS = (
    "actor_obs_keys",
    "actor_hidden_dim",
    "log_std_max",
    "log_std_min",
    "use_tanh",
    "action_boundary_mode",
    "use_layer_norm",
    "obs_normalization",
    "use_cnn_encoder",
    "encoder_obs_key",
    "encoder_obs_shape",
)


def _policy_algorithm_kind(config: dict[str, Any]) -> str:
    algo = _require_mapping(_require_path(config, ("algo",)), "algo")
    algo_config = _require_mapping(_require_path(algo, ("config",)), "algo.config")
    module_dict = algo_config.get("module_dict")
    if isinstance(module_dict, dict) and isinstance(module_dict.get("actor"), dict):
        return "ppo"
    if "actor_obs_keys" in algo_config:
        return "fast_sac"
    raise ValueError(
        "Policy-load config does not describe a supported PPO or FastSAC actor contract."
    )


def _require_fast_sac_bool(algo_config: dict[str, Any], field: str) -> bool:
    value = _require_path(algo_config, (field,))
    if type(value) is not bool:
        raise ValueError(f"FastSAC actor config {field} must be boolean, got {value!r}.")
    return value


def canonical_fast_sac_actor_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the complete inference function represented by a FastSAC actor."""

    config = _require_mapping(config, "experiment_config")
    algo = _require_mapping(_require_path(config, ("algo",)), "algo")
    algo_target = _require_path(algo, ("_target_",))
    if not isinstance(algo_target, str) or not algo_target:
        raise ValueError("FastSAC algo._target_ must be a non-empty string.")
    algo_config = _require_mapping(_require_path(algo, ("config",)), "algo.config")

    raw_actor_obs_keys = _require_path(algo_config, ("actor_obs_keys",))
    if not isinstance(raw_actor_obs_keys, (list, tuple)) or not raw_actor_obs_keys:
        raise ValueError("FastSAC actor_obs_keys must be a non-empty ordered list.")
    if not all(isinstance(key, str) and key for key in raw_actor_obs_keys):
        raise ValueError("FastSAC actor_obs_keys must contain only non-empty strings.")
    actor_obs_keys = list(raw_actor_obs_keys)
    if len(set(actor_obs_keys)) != len(actor_obs_keys):
        raise ValueError(f"FastSAC actor_obs_keys contains duplicates: {actor_obs_keys!r}.")

    actor_hidden_dim = _require_path(algo_config, ("actor_hidden_dim",))
    if type(actor_hidden_dim) is not int or actor_hidden_dim <= 0:
        raise ValueError(
            f"FastSAC actor_hidden_dim must be a positive integer, got {actor_hidden_dim!r}."
        )
    log_std_min = _require_path(algo_config, ("log_std_min",))
    log_std_max = _require_path(algo_config, ("log_std_max",))
    for field, value in (("log_std_min", log_std_min), ("log_std_max", log_std_max)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"FastSAC {field} must be finite, got {value!r}.")
    if float(log_std_min) >= float(log_std_max):
        raise ValueError(
            f"FastSAC log_std_min must be less than log_std_max, got {log_std_min!r} >= {log_std_max!r}."
        )

    use_tanh = _require_fast_sac_bool(algo_config, "use_tanh")
    action_boundary_mode = algo_config.get(
        "action_boundary_mode",
        "legacy_max_range_scalar_v1",
    )
    if action_boundary_mode not in {
        "legacy_max_range_scalar_v1",
        "joint_limit_affine_v2",
    }:
        raise ValueError(
            "FastSAC action_boundary_mode must be 'legacy_max_range_scalar_v1' or "
            f"'joint_limit_affine_v2', got {action_boundary_mode!r}."
        )
    use_layer_norm = _require_fast_sac_bool(algo_config, "use_layer_norm")
    obs_normalization = _require_fast_sac_bool(algo_config, "obs_normalization")
    use_cnn_encoder = _require_fast_sac_bool(algo_config, "use_cnn_encoder")
    encoder_obs_key = _require_path(algo_config, ("encoder_obs_key",))
    if not isinstance(encoder_obs_key, str) or not encoder_obs_key:
        raise ValueError("FastSAC encoder_obs_key must be a non-empty string.")
    raw_encoder_shape = _require_path(algo_config, ("encoder_obs_shape",))
    if not isinstance(raw_encoder_shape, (list, tuple)) or len(raw_encoder_shape) != 3:
        raise ValueError("FastSAC encoder_obs_shape must contain exactly three dimensions.")
    encoder_obs_shape = list(raw_encoder_shape)
    if any(type(size) is not int or size <= 0 for size in encoder_obs_shape):
        raise ValueError(
            f"FastSAC encoder_obs_shape dimensions must be positive integers, got {encoder_obs_shape!r}."
        )
    if use_cnn_encoder and encoder_obs_key not in actor_obs_keys:
        raise ValueError(
            "FastSAC CNN encoder_obs_key must be present in actor_obs_keys: "
            f"encoder_obs_key={encoder_obs_key!r}, actor_obs_keys={actor_obs_keys!r}."
        )

    observation = _require_mapping(_require_path(config, ("observation",)), "observation")
    groups = _require_mapping(_require_path(observation, ("groups",)), "observation.groups")
    ordered_groups: list[dict[str, Any]] = []
    perception_used = False
    for group_name in actor_obs_keys:
        if group_name not in groups:
            raise ValueError(
                f"FastSAC actor references observation group {group_name!r}, but it is missing from config."
            )
        group_contract = _ordered_observation_group(group_name, groups[group_name])
        if group_contract["concatenate"] is not True:
            raise ValueError(
                f"FastSAC actor observation group {group_name!r} must concatenate terms into one tensor."
            )
        ordered_groups.append(group_contract)
        perception_used = perception_used or group_name == "perception_obs" or any(
            "perception" in str(term["func"]).lower() for term in group_contract["terms"]
        )
    if use_cnn_encoder:
        perception_used = perception_used or encoder_obs_key == "perception_obs"

    if "clip_observations" not in observation:
        raise ValueError("FastSAC config is missing observation.clip_observations.")

    robot = _require_mapping(_require_path(config, ("robot",)), "robot")
    robot_action_contract: dict[str, Any] = {}
    for field in (
        "actions_dim",
        "dof_names",
        "dof_pos_lower_limit_list",
        "dof_pos_upper_limit_list",
        "dof_effort_limit_list",
        "init_state",
        "control",
    ):
        if field not in robot:
            raise ValueError(f"FastSAC config is missing robot.{field}.")
        robot_action_contract[field] = _json_value(robot[field])
    actions_dim = robot_action_contract["actions_dim"]
    if type(actions_dim) is not int or actions_dim <= 0:
        raise ValueError(f"FastSAC robot.actions_dim must be positive, got {actions_dim!r}.")

    action = _require_mapping(_require_path(config, ("action",)), "action")
    action_terms = _require_mapping(_require_path(action, ("terms",)), "action.terms")
    ordered_action_terms = [
        {"name": str(term_name), "config": _json_value(term_config)}
        for term_name, term_config in action_terms.items()
    ]

    perception = None
    if perception_used:
        perception = _canonical_perception_config(
            config,
            _require_path(config, ("perception",))
        )

    return {
        "algorithm": "fast_sac",
        "algo_target": algo_target,
        "actor_input_groups": actor_obs_keys,
        "actor_module": {
            "actor_hidden_dim": actor_hidden_dim,
            "log_std_min": _json_value(log_std_min),
            "log_std_max": _json_value(log_std_max),
            "use_tanh": use_tanh,
            "action_boundary_mode": action_boundary_mode,
            "use_layer_norm": use_layer_norm,
            "use_cnn_encoder": use_cnn_encoder,
            "encoder_obs_key": encoder_obs_key if use_cnn_encoder else None,
            "encoder_obs_shape": encoder_obs_shape if use_cnn_encoder else None,
        },
        "observation_groups": ordered_groups,
        "observation_clip": _json_value(observation["clip_observations"]),
        "normalization": {
            "normalize_actor_obs": obs_normalization,
            "implementation": "fast_sac_empirical_v1",
            "eps": 1e-2,
            "until": None,
        },
        "perception": perception,
        "robot_action": robot_action_contract,
        "action_terms": ordered_action_terms,
        "policy_implementation": {
            "version": 1,
            "action_boundary": action_boundary_mode,
        },
    }


def _diff(expected: Any, actual: Any, path: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in expected:
                differences.append(f"{child_path}: checkpoint=<missing> current={actual[key]!r}")
            elif key not in actual:
                differences.append(f"{child_path}: checkpoint={expected[key]!r} current=<missing>")
            else:
                differences.extend(_diff(expected[key], actual[key], child_path))
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: checkpoint_len={len(expected)} current_len={len(actual)}"]
        differences = []
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            differences.extend(_diff(left, right, f"{path}[{index}]"))
        return differences
    if expected != actual:
        return [f"{path}: checkpoint={expected!r} current={actual!r}"]
    return []


def _actor_contract_migration_profile(
    current_config: dict[str, Any],
    *,
    field_name: str = "policy_init_actor_contract_migration",
) -> str | None:
    training = _require_mapping(
        _require_path(current_config, ("training",)),
        "training",
    )
    profile = training.get(field_name)
    if profile is None:
        return None
    if not isinstance(profile, str) or profile != profile.strip() or not profile:
        raise ValueError(
            f"training.{field_name} must be null or one "
            f"non-empty stripped string, got {profile!r}."
        )
    if profile not in {
        TRACKING_TO_PRECOMPUTED_DROP_EXCLUSIVE_MIGRATION,
        PRECOMPUTED_TO_HMI_TERMINAL_GOAL_MIGRATION,
    }:
        raise ValueError(
            f"Unsupported training.{field_name} profile: "
            f"{profile!r}."
        )
    return profile


def _apply_explicit_policy_init_actor_contract_migration(
    saved_contract: dict[str, Any],
    current_contract: dict[str, Any],
    *,
    profile: str,
) -> None:
    """Accept one narrowly scoped, user-requested command-input migration.

    The initialized actor keeps the exact tensor layout, observation producer,
    perception preprocessing, robot/action contract, and normalization.  Only
    the values carried by the existing root-command slots change from live
    tracking error to the immutable precomputed turn/forward schedule, while
    drop becomes exclusive.  Any additional drift remains fail-closed.
    """

    if profile == PRECOMPUTED_TO_HMI_TERMINAL_GOAL_MIGRATION:
        _apply_precomputed_to_hmi_terminal_goal_migration(
            saved_contract,
            current_contract,
            profile=profile,
        )
        return
    if profile != TRACKING_TO_PRECOMPUTED_DROP_EXCLUSIVE_MIGRATION:
        raise AssertionError(f"Unhandled actor-contract migration profile: {profile!r}")
    saved_command = saved_contract.get("command_observation_semantics")
    current_command = current_contract.get("command_observation_semantics")
    if not isinstance(saved_command, dict) or not isinstance(current_command, dict):
        raise ValueError(
            f"Policy-init migration {profile!r} requires contact-aware root-command semantics."
        )
    expected_source = {
        "contact_aware_sparse_root_command_mode": "tracking_error",
        "zero_root_command_when_drop_active": False,
    }
    expected_target = {
        "contact_aware_sparse_root_command_mode": "precomputed_turn_then_forward",
        "zero_root_command_when_drop_active": True,
    }
    for field, expected in expected_source.items():
        if saved_command.get(field) != expected:
            raise ValueError(
                f"Policy-init migration {profile!r} source {field} must be "
                f"{expected!r}, got {saved_command.get(field)!r}."
            )
    for field, expected in expected_target.items():
        if current_command.get(field) != expected:
            raise ValueError(
                f"Policy-init migration {profile!r} target {field} must be "
                f"{expected!r}, got {current_command.get(field)!r}."
            )

    migrated = copy.deepcopy(saved_contract)
    migrated_command = migrated["command_observation_semantics"]
    for field, value in expected_target.items():
        migrated_command[field] = value
    residual_differences = _diff(migrated, current_contract)
    if residual_differences:
        preview = "\n  - ".join(residual_differences[:30])
        suffix = (
            ""
            if len(residual_differences) <= 30
            else f"\n  ... and {len(residual_differences) - 30} more difference(s)"
        )
        raise ValueError(
            f"Policy-init migration {profile!r} permits only the declared root-command "
            "mode and drop-exclusivity changes; residual actor semantic drift:\n  - "
            + preview
            + suffix
        )
    print(
        "[INFO] policy_init_actor_contract_migration_verified "
        f"profile={profile} "
        "source_mode=tracking_error target_mode=precomputed_turn_then_forward "
        "source_drop_exclusive=false target_drop_exclusive=true",
        flush=True,
    )


def _apply_precomputed_to_hmi_terminal_goal_migration(
    saved_contract: dict[str, Any],
    current_contract: dict[str, Any],
    *,
    profile: str,
) -> None:
    """Audit the equal-width m8 command-slot to HMI goal-slot migration.

    The source actor consumes the three-scalar precomputed root command and a
    reference drop button.  HMI Stage 2 deliberately reuses those four tensor
    slots for a terminal object goal and a constant-zero drop value.  The
    native far-tracking CNN is also unfrozen under the current implementation;
    its authenticated source weights are still loaded strictly.  No camera,
    proprioception, robot-control, action, or tensor-topology drift is allowed.
    """

    if profile != PRECOMPUTED_TO_HMI_TERMINAL_GOAL_MIGRATION:
        raise AssertionError(f"Unhandled HMI migration profile: {profile!r}")

    expected_source_command = {
        "contact_aware_carry_window_mode": "peak_height",
        "contact_aware_button_window_mode": "contact_interval",
        "contact_aware_peak_height_alpha": 0.91,
        "contact_aware_peak_height_smoothing_steps": 5,
        "contact_aware_sparse_root_command_mode": "precomputed_turn_then_forward",
        "zero_root_command_when_drop_active": True,
    }
    if saved_contract.get("command_observation_semantics") != expected_source_command:
        raise ValueError(
            f"Policy-init migration {profile!r} requires the exact m8 precomputed-command "
            "source semantics."
        )
    if current_contract.get("command_observation_semantics") is not None:
        raise ValueError(
            f"Policy-init migration {profile!r} requires an HMI target without "
            "reference contact-window command semantics."
        )

    source_groups = [
        "actor_obs_root_contact_aware",
        "actor_obs_drop_button",
        "actor_obs_proprio_with_actions_no_linvel",
    ]
    target_groups = [
        "actor_obs_hmi_goal_command",
        "actor_obs_drop_button",
        "actor_obs_proprio_with_actions_no_linvel",
    ]
    if saved_contract.get("actor_input_groups") != source_groups:
        raise ValueError(
            f"Policy-init migration {profile!r} source actor groups must be "
            f"{source_groups!r}."
        )
    if current_contract.get("actor_input_groups") != target_groups:
        raise ValueError(
            f"Policy-init migration {profile!r} target actor groups must be "
            f"{target_groups!r}."
        )

    source_observation_groups = saved_contract.get("observation_groups")
    target_observation_groups = current_contract.get("observation_groups")
    if not isinstance(source_observation_groups, list) or not isinstance(
        target_observation_groups, list
    ):
        raise ValueError(f"Policy-init migration {profile!r} requires ordered observation groups.")
    if len(source_observation_groups) != len(target_observation_groups) or len(source_observation_groups) < 3:
        raise ValueError(f"Policy-init migration {profile!r} observation group counts differ.")

    expected_source_terms = (
        (
            "sparse_target_root_trajectory_command_contact_aware",
            "holosoma.managers.observation.terms.wbt:"
            "sparse_target_root_trajectory_command_contact_aware",
        ),
        (
            "drop_button",
            "holosoma.managers.observation.terms.wbt:drop_button",
        ),
    )
    expected_target_terms = (
        (
            "hmi_object_goal_command",
            "holosoma.managers.observation.terms.wbt:hmi_object_goal_command",
        ),
        (
            "hmi_zero_drop_button",
            "holosoma.managers.observation.terms.wbt:hmi_zero_drop_button",
        ),
    )
    for index, (expected_name, expected_func) in enumerate(expected_source_terms):
        terms = source_observation_groups[index].get("terms")
        if not isinstance(terms, list) or len(terms) != 1:
            raise ValueError(f"Policy-init migration {profile!r} source group {index} must have one term.")
        if (terms[0].get("name"), terms[0].get("func")) != (expected_name, expected_func):
            raise ValueError(f"Policy-init migration {profile!r} source group {index} has unexpected semantics.")
    for index, (expected_name, expected_func) in enumerate(expected_target_terms):
        terms = target_observation_groups[index].get("terms")
        if not isinstance(terms, list) or len(terms) != 1:
            raise ValueError(f"Policy-init migration {profile!r} target group {index} must have one term.")
        if (terms[0].get("name"), terms[0].get("func")) != (expected_name, expected_func):
            raise ValueError(f"Policy-init migration {profile!r} target group {index} has unexpected semantics.")

    migrated = copy.deepcopy(saved_contract)
    migrated["actor_input_groups"] = copy.deepcopy(target_groups)
    migrated["actor_module"]["input_dim"][0] = target_groups[0]
    migrated["actor_module"]["layer_config"]["module_input_name"][0] = target_groups[0]
    migrated["command_observation_semantics"] = None

    # Preserve every group setting and term field except the two explicitly
    # declared value producers and the renamed command group.
    migrated["observation_groups"][0]["name"] = target_groups[0]
    for index, (target_name, target_func) in enumerate(expected_target_terms):
        migrated["observation_groups"][index]["terms"][0]["name"] = target_name
        migrated["observation_groups"][index]["terms"][0]["func"] = target_func

    saved_layer = saved_contract["actor_module"]["layer_config"]
    current_layer = current_contract["actor_module"]["layer_config"]
    if saved_layer.get("perception_encoder_type") != "far_tracking_cnn_small" or current_layer.get(
        "perception_encoder_type"
    ) != "far_tracking_cnn_small":
        raise ValueError(f"Policy-init migration {profile!r} requires the native far-tracking CNN.")
    expected_native_flag_transition = (True, False)
    for field in ("perception_pretrained", "perception_freeze_backbone"):
        if (saved_layer.get(field), current_layer.get(field)) != expected_native_flag_transition:
            raise ValueError(
                f"Policy-init migration {profile!r} requires native encoder {field} "
                "to transition exactly true->false."
            )
        migrated["actor_module"]["layer_config"][field] = False

    saved_perception = saved_contract.get("perception")
    current_perception = current_contract.get("perception")
    if not isinstance(saved_perception, dict) or not isinstance(current_perception, dict):
        raise ValueError(f"Policy-init migration {profile!r} requires a perception contract.")
    if saved_perception.get("encoder_type") != "far_tracking_cnn_small" or current_perception.get(
        "encoder_type"
    ) != "far_tracking_cnn_small":
        raise ValueError(f"Policy-init migration {profile!r} perception encoder is not far_tracking_cnn_small.")
    for field in ("encoder_pretrained", "encoder_freeze_backbone"):
        if (saved_perception.get(field), current_perception.get(field)) != expected_native_flag_transition:
            raise ValueError(
                f"Policy-init migration {profile!r} requires perception {field} "
                "to transition exactly true->false."
            )
        migrated["perception"][field] = False

    residual_differences = _diff(migrated, current_contract)
    if residual_differences:
        preview = "\n  - ".join(residual_differences[:30])
        suffix = (
            ""
            if len(residual_differences) <= 30
            else f"\n  ... and {len(residual_differences) - 30} more difference(s)"
        )
        raise ValueError(
            f"Policy-init migration {profile!r} permits only the declared command/drop "
            "producers and native-depth unfreeze; residual actor semantic drift:\n  - "
            + preview
            + suffix
        )
    print(
        "[INFO] policy_init_actor_contract_migration_verified "
        f"profile={profile} source=precomputed_turn_then_forward "
        "target=hmi_terminal_object_goal native_depth=true_to_trainable",
        flush=True,
    )


def validate_fast_sac_actor_config_identity(
    saved_config: dict[str, Any],
    current_config: dict[str, Any],
) -> dict[str, Any]:
    """Compare FastSAC actor semantics without requiring checkpoint tensors."""

    saved_contract = canonical_fast_sac_actor_contract(saved_config)
    current_contract = canonical_fast_sac_actor_contract(current_config)
    differences = _diff(saved_contract, current_contract)
    if differences:
        preview = "\n  - ".join(differences[:30])
        suffix = (
            ""
            if len(differences) <= 30
            else f"\n  ... and {len(differences) - 30} more difference(s)"
        )
        raise ValueError(
            "FastSAC actor semantic contract mismatch; equal tensor shapes are not sufficient:\n  - "
            + preview
            + suffix
        )
    return saved_contract


def _validate_normalizer_payload(checkpoint: dict[str, Any], actor_contract: dict[str, Any]) -> None:
    normalization = actor_contract["normalization"]
    if normalization["normalize_actor_obs"] is not True:
        return
    state = checkpoint.get("actor_obs_normalizer_state")
    if not isinstance(state, dict):
        raise ValueError(
            "Policy-init checkpoint enables normalize_actor_obs but actor_obs_normalizer_state is missing."
        )
    expected_keys = set(actor_contract["actor_input_groups"])
    actual_keys = set(state)
    if actual_keys != expected_keys:
        raise ValueError(
            "Policy-init actor_obs_normalizer_state keys do not match actor input groups: "
            f"missing={sorted(expected_keys - actual_keys)}, extra={sorted(actual_keys - expected_keys)}."
        )
    empty = [key for key in actor_contract["actor_input_groups"] if not isinstance(state[key], dict) or not state[key]]
    if empty:
        raise ValueError(
            "Policy-init checkpoint has empty actor observation normalizer state for groups "
            f"{empty!r}."
        )
    validate_finite_tree(state, path="actor_obs_normalizer_state")


def _validate_critic_normalizer_payload(
    checkpoint: dict[str, Any],
    critic_contract: dict[str, Any],
) -> None:
    normalization = critic_contract["normalization"]
    if normalization["normalize_critic_obs"] is not True:
        return
    state = checkpoint.get("critic_obs_normalizer_state")
    if not isinstance(state, dict):
        raise ValueError(
            "Stage-4 checkpoint enables normalize_critic_obs but "
            "critic_obs_normalizer_state is missing."
        )
    expected_keys = set(critic_contract["critic_input_groups"])
    actual_keys = set(state)
    if actual_keys != expected_keys:
        raise ValueError(
            "Stage-4 critic_obs_normalizer_state keys do not match critic input groups: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}."
        )
    empty = [
        key
        for key in critic_contract["critic_input_groups"]
        if not isinstance(state[key], dict) or not state[key]
    ]
    if empty:
        raise ValueError(
            "Stage-4 checkpoint has empty critic observation normalizer state for groups "
            f"{empty!r}."
        )
    validate_finite_tree(state, path="critic_obs_normalizer_state")


def _fast_sac_actor_args(config: dict[str, Any]) -> dict[str, Any]:
    algo_config = _require_mapping(
        _require_path(config, ("algo", "config")),
        "algo.config",
    )
    values: dict[str, Any] = {}
    for field in _FAST_SAC_ACTOR_CONFIG_FIELDS:
        if field == "action_boundary_mode":
            value = algo_config.get(field, "legacy_max_range_scalar_v1")
        elif field in algo_config:
            value = algo_config[field]
        else:
            raise ValueError(f"FastSAC experiment config is missing actor field algo.config.{field}.")
        values[field] = _json_value(value)
    return values


def _validate_fast_sac_args_payload(
    checkpoint: dict[str, Any],
    saved_config: dict[str, Any],
) -> None:
    """Reject checkpoint metadata that contradicts the config used to build the actor."""

    raw_args = checkpoint.get("args")
    if not isinstance(raw_args, dict):
        raise ValueError("FastSAC checkpoint args must be a mapping.")
    saved_args = _fast_sac_actor_args(saved_config)
    checkpoint_args: dict[str, Any] = {}
    for field in _FAST_SAC_ACTOR_CONFIG_FIELDS:
        if field == "action_boundary_mode":
            value = raw_args.get(field, "legacy_max_range_scalar_v1")
        elif field in raw_args:
            value = raw_args[field]
        else:
            raise ValueError(f"FastSAC checkpoint args is missing actor field {field!r}.")
        checkpoint_args[field] = _json_value(value)
    differences = _diff(saved_args, checkpoint_args, "args")
    if differences:
        raise ValueError(
            "FastSAC checkpoint args contradict serialized experiment_config:\n  - "
            + "\n  - ".join(differences[:30])
        )


def _validate_fast_sac_normalizer_payload(
    checkpoint: dict[str, Any],
    actor_contract: dict[str, Any],
) -> None:
    state = checkpoint.get("obs_normalizer_state")
    if not isinstance(state, dict):
        raise ValueError("FastSAC checkpoint obs_normalizer_state must be a mapping.")
    enabled = actor_contract["normalization"]["normalize_actor_obs"] is True
    if not enabled:
        if state:
            raise ValueError(
                "FastSAC checkpoint disables observation normalization but carries non-empty "
                "obs_normalizer_state."
            )
        return

    expected_keys = {"_mean", "_var", "_std", "count"}
    if set(state) != expected_keys:
        raise ValueError(
            "FastSAC obs_normalizer_state keys are invalid: "
            f"missing={sorted(expected_keys - set(state))}, extra={sorted(set(state) - expected_keys)}."
        )
    validate_finite_tree(state, path="obs_normalizer_state")
    mean = state["_mean"]
    variance = state["_var"]
    std = state["_std"]
    count = state["count"]
    if not all(isinstance(value, torch.Tensor) for value in (mean, variance, std, count)):
        raise ValueError("FastSAC obs_normalizer_state values must all be tensors.")
    if mean.ndim != 2 or mean.shape[0] != 1 or mean.numel() == 0:
        raise ValueError(
            f"FastSAC normalizer mean must have shape [1, actor_obs_dim], got {tuple(mean.shape)}."
        )
    if variance.shape != mean.shape or std.shape != mean.shape:
        raise ValueError(
            "FastSAC normalizer mean/variance/std shapes must match: "
            f"mean={tuple(mean.shape)}, var={tuple(variance.shape)}, std={tuple(std.shape)}."
        )
    if not (mean.is_floating_point() and variance.is_floating_point() and std.is_floating_point()):
        raise ValueError("FastSAC normalizer mean/variance/std must use floating tensor dtypes.")
    if count.numel() != 1 or count.dtype == torch.bool or count.is_floating_point() or count.is_complex():
        raise ValueError("FastSAC normalizer count must be one integral tensor value.")
    if int(count.item()) < 0:
        raise ValueError(f"FastSAC normalizer count must be non-negative, got {int(count.item())}.")
    if bool((variance < 0).any().item()):
        raise ValueError("FastSAC normalizer variance must be non-negative.")
    if bool((std < 0).any().item()):
        raise ValueError("FastSAC normalizer std must be non-negative.")
    if not torch.allclose(std.square(), variance, rtol=1e-4, atol=1e-6):
        raise ValueError("FastSAC normalizer std is inconsistent with sqrt(variance).")


def validate_policy_init_payload_identity(
    checkpoint: dict[str, Any],
    current_config: dict[str, Any],
    *,
    migration_field_name: str = "policy_init_actor_contract_migration",
) -> dict[str, Any]:
    """Prove the semantic actor contract of an already authenticated payload.

    Shape-compatible tensors are insufficient for a student/evaluation policy:
    ordered observation terms, history, normalization, perception, robot
    control, and action semantics must describe the same function as the
    runtime configuration.
    """

    if not isinstance(checkpoint, dict):
        raise ValueError("Policy-init checkpoint payload must be a mapping.")
    if not isinstance(current_config, dict):
        raise ValueError("Current policy-load config must be a mapping.")
    saved_config = checkpoint.get("experiment_config")
    if not isinstance(saved_config, dict):
        raise ValueError(
            "Policy-init checkpoint has no serialized experiment_config; "
            "refusing an unverifiable actor warm start."
        )
    saved_kind = _policy_algorithm_kind(saved_config)
    current_kind = _policy_algorithm_kind(current_config)
    if saved_kind != current_kind:
        raise ValueError(
            f"Policy-load algorithm mismatch: checkpoint={saved_kind!r}, current={current_kind!r}."
        )
    if saved_kind == "ppo":
        actor_state_key = "actor_model_state_dict"
        saved_contract = canonical_actor_contract(saved_config)
        current_contract = canonical_actor_contract(current_config)
    else:
        actor_state_key = "actor_state_dict"
        saved_contract = canonical_fast_sac_actor_contract(saved_config)
        current_contract = canonical_fast_sac_actor_contract(current_config)
        _validate_fast_sac_args_payload(checkpoint, saved_config)
    actor_state = require_mapping(checkpoint, actor_state_key)
    validate_finite_tree(actor_state, path=actor_state_key)

    migration_profile = _actor_contract_migration_profile(
        current_config,
        field_name=migration_field_name,
    )
    differences = _diff(saved_contract, current_contract)
    if differences:
        if saved_kind == "ppo" and migration_profile is not None:
            _apply_explicit_policy_init_actor_contract_migration(
                saved_contract,
                current_contract,
                profile=migration_profile,
            )
            differences = []
    elif migration_profile is not None:
        raise ValueError(
            f"training.{migration_field_name} is set, but the checkpoint "
            "and current actor semantic contracts are already identical."
        )
    if differences:
        preview = "\n  - ".join(differences[:30])
        suffix = (
            ""
            if len(differences) <= 30
            else f"\n  ... and {len(differences) - 30} more difference(s)"
        )
        raise ValueError(
            "Policy-init actor semantic contract mismatch; equal tensor shapes are not sufficient:\n  - "
            + preview
            + suffix
        )
    if saved_kind == "ppo":
        _validate_normalizer_payload(checkpoint, saved_contract)
    else:
        _validate_fast_sac_normalizer_payload(checkpoint, saved_contract)
    return saved_contract


def validate_stage4_init_payload_identity(
    checkpoint: dict[str, Any],
    current_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the actor and critic functions restored by a Stage-4 warm start."""

    actor_contract = validate_policy_init_payload_identity(
        checkpoint,
        current_config,
        migration_field_name="stage4_init_contract_migration",
    )
    saved_config = checkpoint.get("experiment_config")
    if not isinstance(saved_config, dict):
        raise ValueError("Stage-4 checkpoint has no serialized experiment_config.")
    if _policy_algorithm_kind(saved_config) != "ppo" or _policy_algorithm_kind(current_config) != "ppo":
        raise ValueError("Stage-4 actor+critic initialization currently supports PPO checkpoints only.")
    critic_state = require_mapping(checkpoint, "critic_model_state_dict")
    validate_finite_tree(critic_state, path="critic_model_state_dict")
    saved_contract = canonical_critic_contract(saved_config)
    current_contract = canonical_critic_contract(current_config)
    differences = _diff(saved_contract, current_contract)
    migration_profile = _actor_contract_migration_profile(
        current_config,
        field_name="stage4_init_contract_migration",
    )
    if differences and migration_profile is not None:
        _apply_explicit_policy_init_actor_contract_migration(
            saved_contract,
            current_contract,
            profile=migration_profile,
        )
        differences = []
    if differences:
        preview = "\n  - ".join(differences[:30])
        suffix = (
            ""
            if len(differences) <= 30
            else f"\n  ... and {len(differences) - 30} more difference(s)"
        )
        raise ValueError(
            "Stage-4 critic semantic contract mismatch; equal tensor shapes are not sufficient:\n  - "
            + preview
            + suffix
        )
    _validate_critic_normalizer_payload(checkpoint, saved_contract)
    return actor_contract, saved_contract


def validate_policy_init_terminal_source_payload(
    checkpoint: dict[str, Any],
    *,
    required_target: int,
) -> dict[str, Any]:
    """Require one current-format terminal DAgger checkpoint as actor source."""

    if not isinstance(checkpoint, dict):
        raise ValueError("Policy-init terminal source payload must be a mapping.")
    if type(required_target) is not int or required_target < 1:
        raise ValueError(
            f"Required policy-init terminal target must be a positive integer, got {required_target!r}."
        )
    expected_completed_iteration = required_target - 1
    required_iteration_fields = {
        "iter": expected_completed_iteration,
        "iteration": expected_completed_iteration,
        "next_iter": required_target,
    }
    for field, expected in required_iteration_fields.items():
        if field not in checkpoint:
            raise ValueError(
                "Policy-init terminal source must use the current checkpoint iteration schema; "
                f"missing explicit {field!r}."
            )
        value = checkpoint[field]
        if type(value) is not int or value != expected:
            raise ValueError(
                f"Policy-init terminal source {field} must equal {expected}, got {value!r}."
            )
    completed_iteration, next_iteration = validate_checkpoint_iterations(checkpoint)
    if (
        completed_iteration != expected_completed_iteration
        or next_iteration != required_target
    ):
        raise ValueError(
            "Policy-init terminal source iteration counters do not match the required target: "
            f"completed={completed_iteration}, next={next_iteration}, target={required_target}."
        )
    saved_target = checkpoint_saved_run_target(checkpoint)
    if saved_target != required_target:
        raise ValueError(
            "Policy-init terminal source saved run target does not match the required target: "
            f"saved={saved_target}, required={required_target}."
        )

    saved_config = checkpoint.get("experiment_config")
    if not isinstance(saved_config, dict):
        raise ValueError("Policy-init terminal source has no serialized experiment_config.")
    try:
        actor_config = saved_config["algo"]["config"]["module_dict"]["actor"]
        layer_config = actor_config["layer_config"]
        actions_dim = saved_config["robot"]["actions_dim"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Policy-init terminal source lacks actor/action metadata needed to validate "
            "its frozen fixed-BC dataset."
        ) from exc
    if not isinstance(actor_config, dict) or not isinstance(layer_config, dict):
        raise ValueError("Policy-init terminal source actor metadata must be mappings.")
    if type(actions_dim) is not int or actions_dim < 1:
        raise ValueError(
            "Policy-init terminal source robot.actions_dim must be a positive integer."
        )
    perception_key = layer_config.get("perception_input_name")
    if perception_key is None:
        perception_key = ""
    if not isinstance(perception_key, str):
        raise ValueError(
            "Policy-init terminal source actor perception_input_name must be a string or null."
        )
    perception_key = perception_key.strip()
    required_tensor_keys = {"actor_obs_raw", "teacher_actions"}
    if perception_key:
        required_tensor_keys.add("actor_perception")

    terminal_state = validate_terminal_fixed_bc_eval_artifact_payload(
        checkpoint,
        expected_completed_iteration=expected_completed_iteration,
        required_tensor_keys=required_tensor_keys,
        expected_widths={"teacher_actions": actions_dim},
        require_terminal=True,
    )
    assert terminal_state is not None
    if (
        terminal_state["next_iteration"] != required_target
        or terminal_state["run_target_iteration"] != required_target
    ):
        raise ValueError(
            "Policy-init terminal fixed-BC proof does not identify the required target: "
            f"next={terminal_state['next_iteration']}, "
            f"run_target={terminal_state['run_target_iteration']}, "
            f"required={required_target}."
        )
    return terminal_state


def validate_policy_init_checkpoint(
    checkpoint_path: Path,
    current_config: dict[str, Any],
    *,
    current_provenance: dict[str, Any] | None = None,
    required_terminal_target: int | None = None,
) -> None:
    legacy_unverified_policy_load = allow_legacy_unverified_policy_load()
    # Keep the lexical final component intact so the verified loader can
    # reject symlinks with O_NOFOLLOW.
    checkpoint_path = Path(
        os.path.abspath(os.fspath(checkpoint_path.expanduser()))
    )
    expected_sha256: str | None = None
    if current_provenance is not None:
        current_provenance = validate_training_provenance(
            current_provenance,
            require_finalized=True,
        )
        if current_provenance.get("policy_init_enabled") is not True:
            raise ValueError(
                "Current training provenance does not enable policy initialization."
            )
        expected_sha256 = current_provenance["policy_init_sha256"]
    elif not legacy_unverified_policy_load:
        raise ValueError(
            "Scientific policy initialization requires finalized current training provenance "
            "with an authenticated policy_init_sha256. Set "
            f"{ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV}=1 only for an explicitly "
            "non-scientific legacy warm start."
        )
    else:
        print(
            "[WARN] legacy_unverified_policy_load_allowed "
            f"override={ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV}=1: checkpoint file identity "
            "is not authenticated by current training provenance.",
            flush=True,
        )
    try:
        checkpoint, _actual_sha256 = load_verified_torch_checkpoint(
            checkpoint_path,
            expected_sha256=expected_sha256,
            map_location="cpu",
        )
    except ValueError as exc:
        if expected_sha256 is not None and "SHA256 does not match" in str(exc):
            raise ValueError(
                "Current training provenance does not identify the policy-init checkpoint being loaded: "
                f"declared={expected_sha256}."
            ) from exc
        raise
    terminal_state = None
    if required_terminal_target is not None:
        terminal_state = validate_policy_init_terminal_source_payload(
            checkpoint,
            required_target=required_terminal_target,
        )
    saved_contract = validate_policy_init_payload_identity(checkpoint, current_config)

    terminal_detail = (
        ""
        if terminal_state is None
        else f" terminal_target={terminal_state['run_target_iteration']}"
    )
    print(
        "[INFO] policy_init_preflight_verified "
        f"checkpoint={checkpoint_path} actor_inputs={saved_contract['actor_input_groups']} "
        f"normalize_actor_obs={saved_contract['normalization']['normalize_actor_obs']}"
        f"{terminal_detail}",
        flush=True,
    )


def validate_stage4_init_checkpoint(
    checkpoint_path: Path,
    current_config: dict[str, Any],
    *,
    current_provenance: dict[str, Any] | None = None,
) -> None:
    """Authenticate and validate an actor+critic fresh-lineage initializer."""

    legacy_unverified_policy_load = allow_legacy_unverified_policy_load()
    checkpoint_path = Path(os.path.abspath(os.fspath(checkpoint_path.expanduser())))
    expected_sha256: str | None = None
    if current_provenance is not None:
        current_provenance = validate_training_provenance(
            current_provenance,
            require_finalized=True,
        )
        if current_provenance.get("stage4_init_enabled") is not True:
            raise ValueError("Current training provenance does not enable Stage-4 initialization.")
        expected_sha256 = current_provenance["stage4_init_sha256"]
    elif not legacy_unverified_policy_load:
        raise ValueError(
            "Scientific Stage-4 initialization requires finalized current training provenance "
            "with an authenticated stage4_init_sha256. Set "
            f"{ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV}=1 only for an explicitly "
            "non-scientific legacy warm start."
        )
    checkpoint, _actual_sha256 = load_verified_torch_checkpoint(
        checkpoint_path,
        expected_sha256=expected_sha256,
        map_location="cpu",
    )
    actor_contract, critic_contract = validate_stage4_init_payload_identity(
        checkpoint,
        current_config,
    )
    print(
        "[INFO] stage4_init_preflight_verified "
        f"checkpoint={checkpoint_path} "
        f"actor_inputs={actor_contract['actor_input_groups']} "
        f"critic_inputs={critic_contract['critic_input_groups']} "
        f"normalize_actor_obs={actor_contract['normalization']['normalize_actor_obs']} "
        f"normalize_critic_obs={critic_contract['normalization']['normalize_critic_obs']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--current-provenance-json")
    parser.add_argument("--require-terminal-target", type=int)
    parser.add_argument(
        "--load-mode",
        choices=("policy_init", "stage4_init"),
        default="policy_init",
    )
    args = parser.parse_args()
    current_config = json.load(sys.stdin)
    current_provenance = parse_training_provenance(args.current_provenance_json)
    if args.load_mode == "stage4_init":
        if args.require_terminal_target is not None:
            raise ValueError("--require-terminal-target is only valid for actor-only policy init.")
        validate_stage4_init_checkpoint(
            args.checkpoint,
            current_config,
            current_provenance=current_provenance,
        )
    else:
        validate_policy_init_checkpoint(
            args.checkpoint,
            current_config,
            current_provenance=current_provenance,
            required_terminal_target=args.require_terminal_target,
        )


if __name__ == "__main__":
    main()
