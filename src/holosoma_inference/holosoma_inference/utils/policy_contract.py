"""Validation helpers for checkpoint/ONNX deployment contracts.

The ONNX feature dimension alone is not enough to identify an observation
layout.  This module validates the serialized training metadata against the
runtime inference configuration before the first policy action is produced.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

from holosoma_inference.config.config_types.observation import (
    ObservationConfig,
    ObservationTermDescriptor,
)


class PolicyContractError(ValueError):
    """Raised when an ONNX policy and the runtime configuration disagree."""


_MAX_CONTACT_AWARE_SMOOTHING_STEPS = 4096
_MAX_MOTION_TRANSITION_STEPS = 4096


def _validate_contact_aware_carry_window_metadata(motion_config: Mapping[str, Any]) -> None:
    """Reject serialized carry-window values that training would not accept.

    Deployment metadata is an untrusted boundary.  In particular, coercing an
    unknown mode to the rel-z implementation or allowing an unbounded smoothing
    kernel changes the learned observation/command semantics and can allocate an
    attacker-sized convolution kernel.
    """

    button_mode_key = "contact_aware_button_window_mode"
    button_mode = motion_config.get(button_mode_key, "contact_interval")
    if not isinstance(button_mode, str) or button_mode not in {
        "contact_interval",
        "kinematic_lift",
    }:
        raise PolicyContractError(
            f"Policy metadata field motion_config.{button_mode_key} must be exactly "
            f"'contact_interval' or 'kinematic_lift', got {button_mode!r}."
        )

    mode_key = "contact_aware_carry_window_mode"
    mode = motion_config.get(mode_key, "rel_z")
    if not isinstance(mode, str) or mode not in {"rel_z", "peak_height"}:
        raise PolicyContractError(
            f"Policy metadata field motion_config.{mode_key} must be exactly "
            f"'rel_z' or 'peak_height', got {mode!r}."
        )

    alpha_key = "contact_aware_peak_height_alpha"
    alpha = motion_config.get(alpha_key, 0.91)
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, Real)
        or not math.isfinite(float(alpha))
        or not 0.0 <= float(alpha) <= 1.0
    ):
        raise PolicyContractError(
            f"Policy metadata field motion_config.{alpha_key} must be a finite real number "
            f"in [0, 1], got {alpha!r}."
        )

    smoothing_key = "contact_aware_peak_height_smoothing_steps"
    smoothing_steps = motion_config.get(smoothing_key, 5)
    if (
        isinstance(smoothing_steps, bool)
        or not isinstance(smoothing_steps, Integral)
        or not 1 <= int(smoothing_steps) <= _MAX_CONTACT_AWARE_SMOOTHING_STEPS
    ):
        raise PolicyContractError(
            f"Policy metadata field motion_config.{smoothing_key} must be an integer in "
            f"[1, {_MAX_CONTACT_AWARE_SMOOTHING_STEPS}], got {smoothing_steps!r}."
        )


def _require_exact_mapping_keys(
    value: Any,
    *,
    expected: set[str],
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyContractError(f"{path} must be a mapping.")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(repr(key) for key in actual - expected)
        raise PolicyContractError(
            f"{path} must contain exactly {sorted(expected)}; "
            f"missing={missing}, unexpected={unexpected}."
        )
    return value


def _require_nonempty_canonical_string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PolicyContractError(f"{path} must be a non-empty stripped string.")
    return value


def _require_finite_vector(value: Any, *, length: int, path: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise PolicyContractError(f"{path} must be a JSON list of {length} finite numbers.")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, Real) or not math.isfinite(float(item)):
            raise PolicyContractError(f"{path}[{index}] must be a finite number.")
        result.append(float(item))
    return result


def _validate_training_geometry_mesh(value: Any, *, path: str) -> None:
    mesh = _require_exact_mapping_keys(
        value,
        expected={"suffix", "size_bytes", "sha256"},
        path=path,
    )
    suffix = mesh["suffix"]
    if (
        not isinstance(suffix, str)
        or len(suffix) <= 1
        or not suffix.startswith(".")
        or suffix != suffix.strip()
        or suffix != suffix.lower()
    ):
        raise PolicyContractError(
            f"{path}.suffix must be a non-empty lowercase file suffix beginning with '.'."
        )
    size_bytes = mesh["size_bytes"]
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, Integral) or int(size_bytes) <= 0:
        raise PolicyContractError(f"{path}.size_bytes must be a positive integer.")
    sha256 = mesh["sha256"]
    if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise PolicyContractError(f"{path}.sha256 must be 64 lowercase hexadecimal characters.")


def _require_canonical_sorted_unique(entries: list[Any], *, path: str) -> None:
    try:
        sort_keys = [
            json.dumps(
                entry,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            for entry in entries
        ]
    except (TypeError, ValueError) as exc:
        raise PolicyContractError(f"{path} must contain strict finite JSON values.") from exc
    if sort_keys != sorted(sort_keys) or len(sort_keys) != len(set(sort_keys)):
        raise PolicyContractError(
            f"{path} must be a canonical sorted unique list using strict JSON sort keys."
        )


def _validate_training_geometry_support(contract: Mapping[str, Any]) -> None:
    path = "Perception training_geometry_support"
    support = _require_exact_mapping_keys(
        contract.get("training_geometry_support"),
        expected={
            "version",
            "camera_source",
            "training_rank_count",
            "robot_mesh_bindings",
            "object_mesh_support",
        },
        path=path,
    )
    version = support["version"]
    if isinstance(version, bool) or not isinstance(version, Integral) or int(version) != 1:
        raise PolicyContractError(f"{path}.version must equal integer 1.")
    camera_source = _require_nonempty_canonical_string(
        support["camera_source"],
        path=f"{path}.camera_source",
    )
    if camera_source != contract.get("camera_source"):
        raise PolicyContractError(
            f"{path}.camera_source must match perception_observation_contract.camera_source."
        )
    training_rank_count = support["training_rank_count"]
    if (
        isinstance(training_rank_count, bool)
        or not isinstance(training_rank_count, Integral)
        or int(training_rank_count) <= 0
    ):
        raise PolicyContractError(f"{path}.training_rank_count must be a positive integer.")

    robot_bindings = support["robot_mesh_bindings"]
    if not isinstance(robot_bindings, list):
        raise PolicyContractError(f"{path}.robot_mesh_bindings must be a JSON list.")
    robot_keys = {
        "slot_name",
        "mesh",
        "tracking_body_name",
        "fixed_position_xyz",
        "fixed_quaternion_xyzw",
    }
    for index, raw_binding in enumerate(robot_bindings):
        binding_path = f"{path}.robot_mesh_bindings[{index}]"
        binding = _require_exact_mapping_keys(
            raw_binding,
            expected=robot_keys,
            path=binding_path,
        )
        _require_nonempty_canonical_string(binding["slot_name"], path=f"{binding_path}.slot_name")
        _require_nonempty_canonical_string(
            binding["tracking_body_name"],
            path=f"{binding_path}.tracking_body_name",
        )
        _validate_training_geometry_mesh(binding["mesh"], path=f"{binding_path}.mesh")
        _require_finite_vector(
            binding["fixed_position_xyz"],
            length=3,
            path=f"{binding_path}.fixed_position_xyz",
        )
        quaternion = _require_finite_vector(
            binding["fixed_quaternion_xyzw"],
            length=4,
            path=f"{binding_path}.fixed_quaternion_xyzw",
        )
        quaternion_norm = math.sqrt(sum(component * component for component in quaternion))
        if not math.isclose(quaternion_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-4):
            raise PolicyContractError(
                f"{binding_path}.fixed_quaternion_xyzw must be a unit quaternion."
            )
    _require_canonical_sorted_unique(robot_bindings, path=f"{path}.robot_mesh_bindings")

    object_support = support["object_mesh_support"]
    if not isinstance(object_support, list):
        raise PolicyContractError(f"{path}.object_mesh_support must be a JSON list.")
    object_keys = {"source_name", "mesh", "training_active_env_count"}
    for index, raw_entry in enumerate(object_support):
        entry_path = f"{path}.object_mesh_support[{index}]"
        entry = _require_exact_mapping_keys(
            raw_entry,
            expected=object_keys,
            path=entry_path,
        )
        _require_nonempty_canonical_string(entry["source_name"], path=f"{entry_path}.source_name")
        _validate_training_geometry_mesh(entry["mesh"], path=f"{entry_path}.mesh")
        active_count = entry["training_active_env_count"]
        if isinstance(active_count, bool) or not isinstance(active_count, Integral) or int(active_count) <= 0:
            raise PolicyContractError(
                f"{entry_path}.training_active_env_count must be a positive integer."
            )
    _require_canonical_sorted_unique(object_support, path=f"{path}.object_mesh_support")

    if camera_source != "far_tracking_warp" and (robot_bindings or object_support):
        raise PolicyContractError(
            f"{path} mesh lists must both be empty when camera_source is not 'far_tracking_warp'."
        )


def perception_observation_contract_sha256_from_metadata(metadata: Mapping[str, Any]) -> str | None:
    """Validate and return an authenticated effective perception contract digest."""

    contract = metadata.get("perception_observation_contract")
    declared_digest = metadata.get("perception_observation_contract_sha256")
    if contract is None and declared_digest is None:
        return None
    if contract is None or declared_digest is None:
        raise PolicyContractError(
            "Perception observation contract metadata must include both the contract and its SHA-256."
        )
    if not isinstance(contract, Mapping) or contract.get("version") != 2:
        raise PolicyContractError(
            "Perception observation contract must be a version-2 mapping; legacy v1 artifacts "
            "do not authenticate producer lifecycle/reset cadence and must be re-exported."
        )
    required_fields = {
        "producer_tick_dt",
        "producer_lifecycle",
        "camera_reset_randomization",
        "camera_setup_randomization",
        "camera_ray_correction_quaternion_xyzw",
        "hole_generator_schema",
        "training_geometry_support",
    }
    missing_fields = sorted(required_fields - set(contract))
    if missing_fields:
        raise PolicyContractError(
            "Perception observation contract v2 is missing required producer fields: "
            f"{missing_fields}."
        )
    producer_tick_dt = contract.get("producer_tick_dt")
    if (
        isinstance(producer_tick_dt, bool)
        or not isinstance(producer_tick_dt, (int, float))
        or not math.isfinite(float(producer_tick_dt))
        or float(producer_tick_dt) <= 0.0
    ):
        raise PolicyContractError(
            f"Perception producer_tick_dt must be finite and positive, got {producer_tick_dt!r}."
        )
    lifecycle = contract.get("producer_lifecycle")
    if not isinstance(lifecycle, Mapping):
        raise PolicyContractError("Perception producer_lifecycle must be a mapping.")
    semantics = lifecycle.get("reset_refresh_semantics")
    expected_lifecycle = {
        "legacy_full_v1": (
            "full_vectorized_batch",
            True,
            True,
            "not_replayable_one_env",
        ),
        "targeted_v2": (
            "reset_env_subset",
            False,
            False,
            "distribution_only",
        ),
    }
    if semantics not in expected_lifecycle:
        raise PolicyContractError(
            "Perception producer_lifecycle.reset_refresh_semantics must be "
            f"one of {sorted(expected_lifecycle)}, got {semantics!r}."
        )
    ordinary_updates = lifecycle.get("ordinary_manager_update_calls_per_control_tick")
    if isinstance(ordinary_updates, bool) or ordinary_updates != 1:
        raise PolicyContractError(
            "Perception producer_lifecycle.ordinary_manager_update_calls_per_control_tick "
            "must equal integer 1."
        )
    initialization_ticks = lifecycle.get(
        "initialization_control_ticks_before_first_reset_output"
    )
    if isinstance(initialization_ticks, bool) or initialization_ticks != 1:
        raise PolicyContractError(
            "Perception producer_lifecycle."
            "initialization_control_ticks_before_first_reset_output must equal integer 1."
        )
    initialization_updates = lifecycle.get(
        "initialization_ordinary_manager_update_calls_before_first_reset_output"
    )
    if isinstance(initialization_updates, bool) or initialization_updates != 1:
        raise PolicyContractError(
            "Perception producer_lifecycle."
            "initialization_ordinary_manager_update_calls_before_first_reset_output "
            "must equal integer 1."
        )
    if lifecycle.get("reset_output_republished_until_physics_advances") is not True:
        raise PolicyContractError(
            "Perception producer_lifecycle."
            "reset_output_republished_until_physics_advances must be true."
        )
    expected_scope, expected_hole_advance, expected_phase_advance, expected_equivalence = (
        expected_lifecycle[str(semantics)]
    )
    if lifecycle.get("reset_output_scope") != expected_scope:
        raise PolicyContractError(
            "Perception producer_lifecycle.reset_output_scope is inconsistent with "
            f"{semantics!r}."
        )
    for field_name, expected in (
        ("hole_clock_advances_on_reset_refresh", expected_hole_advance),
        ("camera_frequency_phase_advances_on_reset_refresh", expected_phase_advance),
    ):
        value = lifecycle.get(field_name)
        if not isinstance(value, bool) or value is not expected:
            raise PolicyContractError(
                f"Perception producer_lifecycle.{field_name} is inconsistent with {semantics!r}."
            )
    consumes_global_rng = lifecycle.get(
        "camera_producer_reset_refresh_consumes_process_global_rng"
    )
    if not isinstance(consumes_global_rng, bool):
        raise PolicyContractError(
            "Perception producer_lifecycle."
            "camera_producer_reset_refresh_consumes_process_global_rng must be boolean."
        )
    expected_future_peer_coupled = bool(
        semantics == "legacy_full_v1" or consumes_global_rng
    )
    if lifecycle.get("future_noise_sample_path_peer_reset_coupled") is not expected_future_peer_coupled:
        raise PolicyContractError(
            "Perception producer_lifecycle.future_noise_sample_path_peer_reset_coupled "
            "is inconsistent with reset lifecycle/global-RNG consumption."
        )
    if lifecycle.get("batch_size_invariant_sample_path") is not False:
        raise PolicyContractError(
            "Perception producer_lifecycle.batch_size_invariant_sample_path must be false "
            "for the current vectorized stochastic producer."
        )
    if lifecycle.get("stochastic_equivalence") != expected_equivalence:
        raise PolicyContractError(
            "Perception producer_lifecycle.stochastic_equivalence is inconsistent with "
            f"{semantics!r}; one-environment deployment authenticates distributions, not an "
            "identical training sample path."
        )
    if lifecycle.get("seed_replay_scope") != "same_execution_trace_only":
        raise PolicyContractError(
            "Perception producer_lifecycle.seed_replay_scope must equal "
            "'same_execution_trace_only'."
        )
    hole_schema = contract.get("hole_generator_schema")
    if hole_schema is not None:
        if not isinstance(hole_schema, Mapping):
            raise PolicyContractError("Perception hole_generator_schema must be a mapping or null.")
        if hole_schema.get("normalization_scope") != "reference_batch":
            raise PolicyContractError(
                "Perception hole_generator_schema.normalization_scope must equal 'reference_batch'."
            )
        reference_batch_size = hole_schema.get("reference_batch_size")
        if (
            isinstance(reference_batch_size, bool)
            or not isinstance(reference_batch_size, Integral)
            or int(reference_batch_size) < 1
        ):
            raise PolicyContractError(
                "Perception hole_generator_schema.reference_batch_size must be a positive integer."
            )
        rank_local_fields = {
            "seed_semantics",
            "effective_seed",
            "gradient_seed_mixer",
            "octave_profile",
        }
        present_rank_local_fields = rank_local_fields.intersection(hole_schema)
        if present_rank_local_fields and present_rank_local_fields != rank_local_fields:
            raise PolicyContractError(
                "Perception rank-local hole_generator_schema seed fields must be all present or all absent."
            )
        if present_rank_local_fields:
            if hole_schema.get("seed_semantics") != "rank_local_v2":
                raise PolicyContractError(
                    "Perception hole_generator_schema.seed_semantics must equal 'rank_local_v2'."
                )
            effective_seed = hole_schema.get("effective_seed")
            if (
                isinstance(effective_seed, bool)
                or not isinstance(effective_seed, Integral)
                or not 0 <= int(effective_seed) <= 2**64 - 1
            ):
                raise PolicyContractError(
                    "Perception hole_generator_schema.effective_seed must be an integer in "
                    "[0, 2**64 - 1]."
                )
            if hole_schema.get("gradient_seed_mixer") != "sha256_u63_be_v1":
                raise PolicyContractError(
                    "Perception hole_generator_schema.gradient_seed_mixer must equal "
                    "'sha256_u63_be_v1'."
                )
            if hole_schema.get("octave_profile") != "legacy_single_octave_v1":
                raise PolicyContractError(
                    "Perception hole_generator_schema.octave_profile must equal "
                    "'legacy_single_octave_v1'."
                )
    _validate_training_geometry_support(contract)
    if not isinstance(declared_digest, str) or len(declared_digest) != 64:
        raise PolicyContractError("Perception observation contract SHA-256 must be a 64-character hex string.")
    try:
        bytes.fromhex(declared_digest)
        payload = json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PolicyContractError("Perception observation contract must contain strict finite JSON values.") from exc
    computed_digest = hashlib.sha256(payload).hexdigest()
    if declared_digest != computed_digest:
        raise PolicyContractError(
            "Perception observation contract SHA-256 does not match its serialized contract: "
            f"declared={declared_digest}, computed={computed_digest}."
        )
    return computed_digest


def motion_transition_contract_from_metadata(
    metadata: Mapping[str, Any],
    *,
    required: bool = False,
) -> Mapping[str, Any] | None:
    """Validate the live training transition contract embedded in an artifact.

    Requested MotionConfig flags are not an effective contract: global
    multi-clip training implements a runtime prepend and intentionally skips
    the requested append.  This digest-bound record describes what the live
    training command actually applied.
    """

    contract = metadata.get("motion_transition_contract")
    declared_digest = metadata.get("motion_transition_contract_sha256")
    if contract is None and declared_digest is None:
        if required:
            raise PolicyContractError(
                "Policy artifact is missing motion_transition_contract metadata. Requested "
                "default-pose transitions cannot be reconstructed from raw config flags; "
                "re-export the source checkpoint with the current code."
            )
        return None
    if contract is None or declared_digest is None:
        raise PolicyContractError(
            "Motion transition metadata must include both motion_transition_contract and its SHA-256."
        )

    contract = _require_exact_mapping_keys(
        contract,
        expected={"version", "control_dt_s", "source_semantics", "prepend", "append"},
        path="motion_transition_contract",
    )
    version = contract["version"]
    if isinstance(version, bool) or not isinstance(version, Integral) or int(version) != 1:
        raise PolicyContractError("motion_transition_contract.version must equal integer 1.")
    control_dt_s = contract["control_dt_s"]
    if (
        isinstance(control_dt_s, bool)
        or not isinstance(control_dt_s, Real)
        or not math.isfinite(float(control_dt_s))
        or float(control_dt_s) <= 0.0
    ):
        raise PolicyContractError(
            "motion_transition_contract.control_dt_s must be a finite positive real number."
        )
    source_semantics = contract["source_semantics"]
    if not isinstance(source_semantics, str) or source_semantics not in {
        "single_clip_static",
        "global_multi_clip_runtime",
    }:
        raise PolicyContractError(
            "motion_transition_contract.source_semantics must be exactly "
            "'single_clip_static' or 'global_multi_clip_runtime'."
        )

    phases: dict[str, Mapping[str, Any]] = {}
    for phase_name in ("prepend", "append"):
        phase = _require_exact_mapping_keys(
            contract[phase_name],
            expected={"implementation", "applied", "steps"},
            path=f"motion_transition_contract.{phase_name}",
        )
        implementation = phase["implementation"]
        allowed_implementations = (
            {"none", "static_splice", "runtime_hold"}
            if phase_name == "prepend"
            else {"none", "static_splice"}
        )
        if not isinstance(implementation, str) or implementation not in allowed_implementations:
            raise PolicyContractError(
                f"motion_transition_contract.{phase_name}.implementation must be one of "
                f"{sorted(allowed_implementations)}, got {implementation!r}."
            )
        applied = phase["applied"]
        if not isinstance(applied, bool):
            raise PolicyContractError(
                f"motion_transition_contract.{phase_name}.applied must be boolean."
            )
        steps = phase["steps"]
        if isinstance(steps, bool) or not isinstance(steps, Integral):
            raise PolicyContractError(
                f"motion_transition_contract.{phase_name}.steps must be an integer."
            )
        steps = int(steps)
        if applied:
            if implementation == "none" or not 2 <= steps <= _MAX_MOTION_TRANSITION_STEPS:
                raise PolicyContractError(
                    f"Applied motion transition {phase_name} must use a concrete implementation "
                    f"and steps in [2, {_MAX_MOTION_TRANSITION_STEPS}]."
                )
        elif implementation != "none" or steps != 0:
            raise PolicyContractError(
                f"Inactive motion transition {phase_name} must use implementation='none' and steps=0."
            )
        phases[phase_name] = phase

    prepend_impl = phases["prepend"]["implementation"]
    append_impl = phases["append"]["implementation"]
    if source_semantics == "global_multi_clip_runtime":
        if prepend_impl not in {"none", "runtime_hold"} or append_impl != "none":
            raise PolicyContractError(
                "Global multi-clip training permits only a runtime-hold prepend and no append."
            )
    elif prepend_impl not in {"none", "static_splice"} or append_impl not in {
        "none",
        "static_splice",
    }:
        raise PolicyContractError(
            "Single-clip training permits only static-splice default-pose transitions."
        )

    if not isinstance(declared_digest, str) or re.fullmatch(r"[0-9a-f]{64}", declared_digest) is None:
        raise PolicyContractError(
            "Motion transition contract SHA-256 must be 64 lowercase hexadecimal characters."
        )
    try:
        payload = json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PolicyContractError(
            "Motion transition contract must contain strict finite JSON values."
        ) from exc
    computed_digest = hashlib.sha256(payload).hexdigest()
    if declared_digest != computed_digest:
        raise PolicyContractError(
            "Motion transition contract SHA-256 does not match its serialized contract: "
            f"declared={declared_digest}, computed={computed_digest}."
        )
    return contract


def effective_motion_transition_settings_from_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve eval/deployment flags from the authenticated effective timeline.

    Legacy artifacts are accepted only when their serialized config proves
    both phases were explicitly inactive. Requested flags are never used as a
    substitute for the single-vs-global implementation contract.
    """

    experiment_config = metadata.get("experiment_config")
    command = experiment_config.get("command") if isinstance(experiment_config, Mapping) else None
    setup_terms = command.get("setup_terms") if isinstance(command, Mapping) else None
    motion_command = setup_terms.get("motion_command") if isinstance(setup_terms, Mapping) else None
    params = motion_command.get("params") if isinstance(motion_command, Mapping) else None
    motion_cfg = params.get("motion_config") if isinstance(params, Mapping) else None

    def serialized_config_requests_transitions() -> bool:
        if not isinstance(motion_cfg, Mapping):
            # This helper is used only by WBT timeline consumers. Absence of a
            # motion config cannot prove an explicitly inactive legacy source.
            return True
        for enabled_key, duration_key in (
            ("enable_default_pose_prepend", "default_pose_prepend_duration_s"),
            ("enable_default_pose_append", "default_pose_append_duration_s"),
        ):
            # Historical MotionConfig defaults were enabled/two seconds.
            # Missing or malformed fields are therefore ambiguous.
            enabled = motion_cfg.get(enabled_key, True)
            duration = motion_cfg.get(duration_key, 2.0)
            if type(enabled) is not bool:
                return True
            if not enabled:
                continue
            if (
                isinstance(duration, bool)
                or not isinstance(duration, Real)
                or not math.isfinite(float(duration))
                or float(duration) > 0.0
            ):
                return True
        return False

    contract = motion_transition_contract_from_metadata(
        metadata,
        required=serialized_config_requests_transitions(),
    )
    if contract is None:
        inactive = {
            "implementation": "none",
            "applied": False,
            "steps": 0,
            "duration_s": 0.0,
        }
        return {
            "source_semantics": "legacy_explicitly_inactive",
            "control_dt_s": None,
            "contract_sha256": None,
            "prepend": dict(inactive),
            "append": dict(inactive),
        }

    if not isinstance(experiment_config, Mapping) or not isinstance(motion_cfg, Mapping):
        raise PolicyContractError(
            "motion_transition_contract requires serialized experiment_config motion_config metadata."
        )
    simulator = experiment_config.get("simulator")
    simulator_config = simulator.get("config") if isinstance(simulator, Mapping) else None
    sim = simulator_config.get("sim") if isinstance(simulator_config, Mapping) else None
    fps = sim.get("fps") if isinstance(sim, Mapping) else None
    decimation = sim.get("control_decimation") if isinstance(sim, Mapping) else None
    if (
        isinstance(fps, bool)
        or not isinstance(fps, Real)
        or not math.isfinite(float(fps))
        or float(fps) <= 0.0
        or isinstance(decimation, bool)
        or not isinstance(decimation, Integral)
        or int(decimation) <= 0
    ):
        raise PolicyContractError(
            "motion_transition_contract requires finite positive simulator fps and a positive "
            "integer control_decimation."
        )
    serialized_control_dt = int(decimation) / float(fps)
    control_dt = float(contract["control_dt_s"])
    if not math.isclose(control_dt, serialized_control_dt, rel_tol=0.0, abs_tol=1.0e-12):
        raise PolicyContractError(
            "motion_transition_contract control_dt_s does not match experiment_config."
        )

    source_semantics = str(contract["source_semantics"])
    has_applied_transition = any(
        bool(contract[phase_name]["applied"])
        for phase_name in ("prepend", "append")
    )
    if has_applied_transition:
        simulator_target = simulator.get("_target_") if isinstance(simulator, Mapping) else None
        simulator_name = (
            simulator_config.get("name")
            if isinstance(simulator_config, Mapping)
            else None
        )
        if not isinstance(simulator_target, str) or not simulator_target.strip():
            raise PolicyContractError(
                "Applied motion transitions require a non-empty serialized simulator._target_."
            )
        if not isinstance(simulator_name, str) or not simulator_name.strip():
            raise PolicyContractError(
                "Applied motion transitions require a non-empty serialized simulator.config.name."
            )
        target_name = simulator_target.rsplit(".", 1)[-1].lower()
        configured_name = simulator_name.lower()
        if target_name != configured_name:
            raise PolicyContractError(
                "Applied motion transitions require matching simulator._target_ and "
                "simulator.config.name."
            )
        if target_name != "isaacsim":
            raise PolicyContractError(
                "Applied motion transitions are only realizable by the exact IsaacSim backend; "
                f"serialized backend is {target_name!r}."
            )
    result: dict[str, Any] = {
        "source_semantics": source_semantics,
        "control_dt_s": control_dt,
        "contract_sha256": metadata["motion_transition_contract_sha256"],
    }
    for phase_name in ("prepend", "append"):
        enabled_key = f"enable_default_pose_{phase_name}"
        duration_key = f"default_pose_{phase_name}_duration_s"
        enabled = motion_cfg.get(enabled_key)
        duration = motion_cfg.get(duration_key)
        if type(enabled) is not bool:
            raise PolicyContractError(f"motion_config.{enabled_key} must be boolean.")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, Real)
            or not math.isfinite(float(duration))
            or float(duration) < 0.0
        ):
            raise PolicyContractError(
                f"motion_config.{duration_key} must be finite and non-negative."
            )
        requested_steps = round(float(duration) / control_dt)
        phase = contract[phase_name]
        if bool(phase["applied"]):
            if not enabled or requested_steps != int(phase["steps"]):
                raise PolicyContractError(
                    f"Effective {phase_name} transition contradicts serialized motion_config."
                )
        elif source_semantics == "single_clip_static" and enabled and requested_steps > 1:
            raise PolicyContractError(
                f"Single-clip contract omitted the requested {phase_name} static transition."
            )
        steps = int(phase["steps"])
        result[phase_name] = {
            "implementation": phase["implementation"],
            "applied": bool(phase["applied"]),
            "steps": steps,
            "duration_s": steps * control_dt if bool(phase["applied"]) else 0.0,
        }
    return result


def _feature_dim(shapes: Mapping[str, Sequence[Any]], name: str) -> int:
    shape = shapes.get(name)
    if shape is None:
        raise PolicyContractError(f"ONNX tensor {name!r} has no declared shape.")
    if isinstance(shape, (str, bytes)) or len(shape) != 2:
        raise PolicyContractError(
            f"ONNX tensor {name!r} must have rank 2 [batch, features], got shape={shape!r}."
        )
    batch = shape[0]
    if isinstance(batch, bool):
        raise PolicyContractError(f"ONNX tensor {name!r} has invalid boolean batch dimension {batch!r}.")
    if isinstance(batch, Integral):
        if int(batch) != 1:
            raise PolicyContractError(
                f"ONNX tensor {name!r} has fixed batch dimension {batch}; runtime requires batch 1 "
                "or a dynamic batch dimension."
            )
    elif batch is None:
        pass
    elif isinstance(batch, str) and batch:
        pass
    else:
        raise PolicyContractError(
            f"ONNX tensor {name!r} batch dimension must be fixed integer 1 or a non-empty dynamic "
            f"symbol/None, got {batch!r}."
        )
    value = shape[1]
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) <= 0:
        raise PolicyContractError(
            f"ONNX tensor {name!r} must have a static positive feature dimension, got shape={shape!r}."
        )
    return int(value)


def _validate_float32_tensor_types(
    *,
    input_shapes: Mapping[str, Sequence[Any]],
    output_shapes: Mapping[str, Sequence[Any]],
    input_types: Mapping[str, str] | None,
    output_types: Mapping[str, str] | None,
) -> None:
    """Reject graphs that cannot consume the runtime's float32 tensor contract."""

    if input_types is not None:
        for name in input_shapes:
            tensor_type = input_types.get(name)
            if tensor_type != "tensor(float)":
                raise PolicyContractError(
                    f"ONNX input {name!r} must have type tensor(float), got {tensor_type!r}."
                )

    if output_types is not None:
        supported_outputs = {
            "action",
            "actions",
            "joint_pos",
            "joint_vel",
            "ref_pos_xyz",
            "ref_quat_xyzw",
        }
        for name in supported_outputs.intersection(output_shapes):
            tensor_type = output_types.get(name)
            if tensor_type != "tensor(float)":
                raise PolicyContractError(
                    f"ONNX output {name!r} must have type tensor(float), got {tensor_type!r}."
                )


def _actor_groups_from_metadata(
    metadata: Mapping[str, Any],
) -> tuple[list[str], Mapping[str, Any], Mapping[str, Any]] | None:
    if "experiment_config" not in metadata:
        return None

    experiment = metadata["experiment_config"]
    if not isinstance(experiment, Mapping):
        raise PolicyContractError("Policy experiment_config metadata must be a mapping.")

    current: Any = experiment
    for key in ("algo", "config", "module_dict", "actor"):
        if not isinstance(current, Mapping) or key not in current:
            raise PolicyContractError(
                "Policy experiment metadata is missing "
                f"algo.config.module_dict.actor (stopped at {key!r})."
            )
        current = current[key]
    actor = current
    if not isinstance(actor, Mapping):
        raise PolicyContractError("Policy metadata algo.config.module_dict.actor must be a mapping.")

    saved_observation = experiment.get("observation")
    if not isinstance(saved_observation, Mapping):
        raise PolicyContractError("Policy experiment metadata is missing observation configuration.")
    if "groups" not in saved_observation or not isinstance(saved_observation["groups"], Mapping):
        raise PolicyContractError("Policy experiment metadata is missing observation.groups.")
    groups = saved_observation["groups"]

    actor_groups = actor.get("input_dim")
    if (
        not isinstance(actor_groups, (list, tuple))
        or not actor_groups
        or not all(isinstance(name, str) and name for name in actor_groups)
    ):
        raise PolicyContractError(
            "Policy metadata actor.input_dim must be a non-empty ordered list of group names."
        )
    if len(set(actor_groups)) != len(actor_groups):
        raise PolicyContractError(f"Policy metadata actor.input_dim contains duplicates: {actor_groups!r}.")
    return list(actor_groups), groups, actor


def actor_perception_input_name_from_metadata(metadata: Mapping[str, Any]) -> str | None:
    """Return the authenticated actor perception input name, if configured."""

    extracted = _actor_groups_from_metadata(metadata)
    if extracted is None:
        return None
    _, _, actor = extracted
    layer_config = actor.get("layer_config")
    if not isinstance(layer_config, Mapping):
        raise PolicyContractError("Policy metadata actor.layer_config must be a mapping.")
    raw_name = layer_config.get("perception_input_name")
    if raw_name is None or raw_name == "":
        return None
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise PolicyContractError(
            "Policy metadata actor.layer_config.perception_input_name must be null or a non-empty string."
        )
    return raw_name.strip()


def _canonical_metadata_value(value: Any, *, label: str) -> tuple[Any, ...]:
    """Return a JSON-like, type-aware value for exact semantic comparison."""

    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, int):
        return ("number", float(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PolicyContractError(f"{label} contains a non-finite numeric value {value!r}.")
        return ("number", float(value))
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise PolicyContractError(f"{label} must contain only string mapping keys.")
        return (
            "mapping",
            tuple(
                (key, _canonical_metadata_value(value[key], label=f"{label}.{key}"))
                for key in sorted(value)
            ),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return (
            "sequence",
            tuple(
                _canonical_metadata_value(item, label=f"{label}[{index}]")
                for index, item in enumerate(value)
            ),
        )
    raise PolicyContractError(
        f"{label} contains unsupported serialized value type {type(value).__name__}."
    )


def _finite_scalar(value: Any, *, label: str, non_negative: bool = False) -> float:
    if isinstance(value, bool):
        raise PolicyContractError(f"{label} must be numeric, got boolean {value!r}.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PolicyContractError(f"{label} must be numeric, got {value!r}.") from exc
    if not math.isfinite(result) or (non_negative and result < 0.0):
        qualifier = "finite and non-negative" if non_negative else "finite"
        raise PolicyContractError(f"{label} must be {qualifier}, got {value!r}.")
    return result


def _canonical_clip(value: Any, *, label: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise PolicyContractError(f"{label} must be null or a two-element numeric sequence.")
    low = _finite_scalar(value[0], label=f"{label}[0]")
    high = _finite_scalar(value[1], label=f"{label}[1]")
    if low > high:
        raise PolicyContractError(f"{label} lower bound {low} exceeds upper bound {high}.")
    return (low, high)


def _configured_actor_groups(observation: ObservationConfig) -> list[str]:
    return [
        name
        for name in observation.obs_dict
        if name.startswith("actor_obs") or name == "motion_future_target_poses"
    ]


def _configured_group_dim(observation: ObservationConfig, group_name: str) -> int:
    history = int(observation.history_length_dict.get(group_name, 1))
    if history < 1:
        raise PolicyContractError(f"Runtime observation group {group_name!r} has invalid history length {history}.")
    try:
        frame_dim = sum(int(observation.obs_dims[term]) for term in observation.obs_dict[group_name])
    except KeyError as exc:
        raise PolicyContractError(
            f"Runtime observation group {group_name!r} references a term with no configured dimension: {exc.args[0]!r}."
        ) from exc
    return frame_dim * history


def _object_actor_contract_requires_metadata(observation: ObservationConfig) -> bool:
    """Return whether shape-only deployment could silently alter object semantics.

    This is derived from the selected terms instead of a user-overridable config
    flag, so a CLI option cannot disable the fail-closed boundary.
    """

    actor_terms = {
        term
        for group_name, terms in observation.obs_dict.items()
        if group_name.startswith("actor_obs")
        for term in terms
    }
    current_split = {
        "obj_size",
        "obj_target_ori_b",
        "obj_target_pos_b",
        "obj_pos_b",
        "obj_ori_b",
    }
    legacy_combined = {"obj_target_pose_size_b", "obj_pos_b", "obj_ori_b"}
    return current_split.issubset(actor_terms) or legacy_combined.issubset(actor_terms)


def validate_onnx_policy_contract(
    *,
    metadata: Mapping[str, Any],
    input_shapes: Mapping[str, Sequence[Any]],
    output_shapes: Mapping[str, Sequence[Any]],
    observation: ObservationConfig,
    runtime_dof_names: Sequence[str],
    runtime_default_dof_angles: Sequence[float],
    runtime_motor_effort_limits: Sequence[float] | None = None,
    runtime_joint2motor: Sequence[int] | None = None,
    input_types: Mapping[str, str] | None = None,
    output_types: Mapping[str, str] | None = None,
) -> bool:
    """Validate a metadata-backed ONNX policy against inference configuration.

    Returns ``True`` when a complete actor observation contract was present in
    metadata and validated.  Legacy artifacts without that metadata return
    ``False`` so callers may use their existing compatibility path.  Shape
    checks are still mandatory for legacy artifacts: an explicitly selected
    runtime term contract may never be padded or truncated to fit ONNX.
    """

    _validate_float32_tensor_types(
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        input_types=input_types,
        output_types=output_types,
    )
    motion_transition_contract_from_metadata(metadata)

    configured_groups = _configured_actor_groups(observation)
    obs_input_name = "actor_obs" if "actor_obs" in input_shapes else "obs" if "obs" in input_shapes else None
    if obs_input_name is None:
        raise PolicyContractError(f"Policy has no supported actor input ('actor_obs' or 'obs'): {sorted(input_shapes)}")
    onnx_obs_dim = _feature_dim(input_shapes, obs_input_name)
    configured_obs_dim = sum(_configured_group_dim(observation, name) for name in configured_groups)
    if onnx_obs_dim != configured_obs_dim:
        raise PolicyContractError(
            f"Runtime actor observation dimension does not match ONNX: expected={onnx_obs_dim}, "
            f"configured={configured_obs_dim}. Refusing to pad or truncate the selected term contract."
        )

    action_outputs = [name for name in ("action", "actions") if name in output_shapes]
    if not action_outputs:
        raise PolicyContractError(
            f"Policy has no supported action output ('action' or 'actions'): {sorted(output_shapes)}"
        )
    if len(action_outputs) != 1:
        raise PolicyContractError(
            "Policy must expose exactly one supported action output, "
            f"found={action_outputs}."
        )
    action_name = action_outputs[0]
    action_dim = _feature_dim(output_shapes, action_name)
    if action_dim != len(runtime_dof_names):
        raise PolicyContractError(
            f"ONNX action dimension {action_dim} does not match runtime DOF count {len(runtime_dof_names)}."
        )

    motion_output_dims = {
        "joint_pos": len(runtime_dof_names),
        "joint_vel": len(runtime_dof_names),
        "ref_pos_xyz": 3,
        "ref_quat_xyzw": 4,
    }
    for output_name, expected_dim in motion_output_dims.items():
        if output_name not in output_shapes:
            continue
        actual_dim = _feature_dim(output_shapes, output_name)
        if actual_dim != expected_dim:
            raise PolicyContractError(
                f"ONNX motion output {output_name!r} dimension {actual_dim} does not match "
                f"the runtime-required dimension {expected_dim}."
            )

    present_motion_outputs = set(motion_output_dims).intersection(output_shapes)
    required_motion_outputs = {"joint_pos", "joint_vel", "ref_quat_xyzw"}
    configured_terms = {
        term
        for group_name in configured_groups
        for term in observation.obs_dict.get(group_name, ())
    }
    uses_motion_command = bool(
        configured_terms.intersection(
            {"motion_command", "motion_ref_ori_b", "motion_future_target_poses"}
        )
    )
    if (present_motion_outputs or "time_step" in input_shapes) and not uses_motion_command:
        raise PolicyContractError(
            "ONNX time_step/motion outputs belong only to a motion-command observation contract; "
            f"runtime actor terms are {sorted(configured_terms)}."
        )
    if present_motion_outputs and "time_step" not in input_shapes:
        raise PolicyContractError(
            "ONNX motion outputs require a time_step input so runtime targets can advance every "
            f"control step; found motion outputs {sorted(present_motion_outputs)} without time_step."
        )
    if "time_step" in input_shapes:
        time_step_dim = _feature_dim(input_shapes, "time_step")
        if time_step_dim != 1:
            raise PolicyContractError(
                f"ONNX time_step input must have feature dimension 1, got {time_step_dim}."
            )
        if not required_motion_outputs.issubset(output_shapes):
            raise PolicyContractError(
                "ONNX time_step input is supported only for policies exposing joint_pos, joint_vel, "
                "and ref_quat_xyzw motion outputs."
            )

    extracted = _actor_groups_from_metadata(metadata)
    if extracted is None:
        if _object_actor_contract_requires_metadata(observation):
            raise PolicyContractError(
                "The selected observation layout rejects shape-only deployment and requires complete "
                "experiment_config metadata. Observation semantics are not authenticated by a tensor "
                "width; in particular, the semantically ambiguous 875-D current split object-target "
                "and legacy combined object-target contracts have identical shapes."
            )
        extra_inputs = sorted(set(input_shapes) - {obs_input_name})
        if extra_inputs:
            raise PolicyContractError(
                "Student/perception ONNX policies require complete experiment_config metadata; "
                f"shape-only deployment cannot authenticate the semantics, order, history, units, or "
                f"preprocessing of extra inputs {extra_inputs}. Re-export the checkpoint with metadata."
            )
        return False
    expected_groups, saved_groups, actor = extracted

    experiment = metadata.get("experiment_config")
    if isinstance(experiment, Mapping):
        saved_observation = experiment.get("observation")
        if not isinstance(saved_observation, Mapping) or "clip_observations" not in saved_observation:
            raise PolicyContractError("Policy metadata is missing observation.clip_observations.")
        saved_clip = _finite_scalar(
            saved_observation["clip_observations"],
            label="Policy metadata observation.clip_observations",
        )
        runtime_clip = _finite_scalar(
            observation.clip_observations,
            label="Runtime observation.clip_observations",
        )
        if not math.isfinite(saved_clip) or saved_clip <= 0.0:
            raise PolicyContractError(f"Policy metadata has invalid observation clip {saved_clip!r}.")
        if not math.isfinite(runtime_clip) or runtime_clip <= 0.0:
            raise PolicyContractError(f"Runtime observation clip must be finite and > 0, got {runtime_clip!r}.")
        if not math.isclose(runtime_clip, saved_clip, rel_tol=0.0, abs_tol=1.0e-9):
            raise PolicyContractError(
                "Runtime global observation clip does not match policy metadata: "
                f"expected={saved_clip}, configured={runtime_clip}."
            )

        command = experiment.get("command")
        setup_terms = command.get("setup_terms") if isinstance(command, Mapping) else None
        motion_command = setup_terms.get("motion_command") if isinstance(setup_terms, Mapping) else None
        params = motion_command.get("params") if isinstance(motion_command, Mapping) else None
        motion_config = params.get("motion_config") if isinstance(params, Mapping) else None
        if isinstance(motion_config, Mapping):
            _validate_contact_aware_carry_window_metadata(motion_config)
            compensation_key = "contact_interval_runtime_prepend_compensation"
            compensation = motion_config.get(compensation_key, False)
            if not isinstance(compensation, bool):
                raise PolicyContractError(
                    f"Policy metadata field motion_config.{compensation_key} must be boolean, "
                    f"got {compensation!r}."
                )
            sparse_mode_key = "contact_aware_sparse_root_command_mode"
            if sparse_mode_key in motion_config:
                sparse_mode_raw = motion_config[sparse_mode_key]
                if not isinstance(sparse_mode_raw, str):
                    raise PolicyContractError(
                        f"Policy metadata field motion_config.{sparse_mode_key} must be a string, "
                        f"got {sparse_mode_raw!r}."
                    )
                sparse_mode = sparse_mode_raw.strip().lower().replace("-", "_")
                if sparse_mode in {"tracking", "default", "robot_tracking_error"}:
                    sparse_mode = "tracking_error"
                if sparse_mode != "tracking_error":
                    raise PolicyContractError(
                        "Inference implements only tracking_error/default contact-aware sparse-root "
                        "commands; policy metadata requests "
                        f"motion_config.{sparse_mode_key}={sparse_mode_raw!r}."
                    )

        saved_action = experiment.get("action")
        action_terms = saved_action.get("terms") if isinstance(saved_action, Mapping) else None
        if not isinstance(action_terms, Mapping) or len(action_terms) != 1:
            raise PolicyContractError(
                "Policy metadata must contain exactly one serialized action term."
            )
        action_term = next(iter(action_terms.values()))
        if not isinstance(action_term, Mapping):
            raise PolicyContractError("Policy metadata contains an invalid action term.")
        action_func = action_term.get("func")
        expected_action_func = (
            "holosoma.managers.action.terms.joint_control:JointPositionActionTerm"
        )
        if action_func != expected_action_func:
            raise PolicyContractError(
                f"Inference supports only the exact action implementation {expected_action_func!r}, "
                f"got {action_func!r}."
            )
        if action_term.get("params") != {}:
            raise PolicyContractError(
                "Inference supports JointPositionActionTerm only with an explicit empty params mapping."
            )
        try:
            action_term_scale = float(action_term.get("scale", 1.0))
        except (TypeError, ValueError) as exc:
            raise PolicyContractError("Serialized action-term scale must be numeric.") from exc
        if not math.isfinite(action_term_scale) or action_term_scale != 1.0:
            raise PolicyContractError(
                "Inference does not implement a separate action-term scale; serialized scale must be 1.0."
            )
        if action_term.get("clip") is not None:
            raise PolicyContractError(
                "Inference does not implement action-term clipping; serialized action term clip must be null."
            )

        saved_robot = experiment.get("robot")
        if not isinstance(saved_robot, Mapping):
            raise PolicyContractError("Policy metadata is missing robot configuration.")
        control = saved_robot.get("control")
        if not isinstance(control, Mapping):
            raise PolicyContractError("Policy metadata is missing robot.control configuration.")
        if control.get("control_type") != "P":
            raise PolicyContractError(
                "Inference supports only position-control policies (robot.control.control_type='P')."
            )
        clip_actions = control.get("clip_actions")
        if clip_actions is not True:
            raise PolicyContractError(
                "Inference requires robot.control.clip_actions=true in the serialized training contract."
            )
        try:
            action_clip = float(control["action_clip_value"])
            action_scale = float(control["action_scale"])
        except KeyError as exc:
            raise PolicyContractError(
                f"Policy metadata is missing robot.control.{exc.args[0]}."
            ) from exc
        except (TypeError, ValueError) as exc:
            raise PolicyContractError("Serialized action clip/scale values must be numeric.") from exc
        if not math.isfinite(action_clip) or action_clip <= 0.0:
            raise PolicyContractError(
                f"Serialized robot.control.action_clip_value must be finite and > 0, got {action_clip!r}."
            )
        if not math.isfinite(action_scale) or action_scale <= 0.0:
            raise PolicyContractError(
                f"Serialized robot.control.action_scale must be finite and > 0, got {action_scale!r}."
            )
        per_joint_scale = control.get("action_scales_by_effort_limit_over_p_gain", False)
        if not isinstance(per_joint_scale, bool):
            raise PolicyContractError(
                "Policy metadata field robot.control.action_scales_by_effort_limit_over_p_gain must be boolean."
            )
        if per_joint_scale:
            saved_effort = saved_robot.get("dof_effort_limit_list")
            if not isinstance(saved_effort, Sequence) or isinstance(saved_effort, (str, bytes)):
                raise PolicyContractError(
                    "Per-joint action scaling requires robot.dof_effort_limit_list in policy metadata."
                )
            if runtime_motor_effort_limits is None or runtime_joint2motor is None:
                raise PolicyContractError(
                    "Per-joint action scaling requires runtime motor effort limits and joint2motor mapping."
                )
            dof_count = len(runtime_dof_names)
            if len(saved_effort) != dof_count or len(runtime_joint2motor) != dof_count:
                raise PolicyContractError(
                    "Action-scale contract dimensions disagree: "
                    f"saved_effort={len(saved_effort)}, joint2motor={len(runtime_joint2motor)}, dofs={dof_count}."
                )
            mapping: list[int] = []
            for value in runtime_joint2motor:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise PolicyContractError(f"runtime joint2motor must contain integer indices, got {value!r}.")
                mapping.append(int(value))
            if len(set(mapping)) != len(mapping):
                raise PolicyContractError(f"runtime joint2motor must be one-to-one, got {mapping}.")
            motor_effort = [float(value) for value in runtime_motor_effort_limits]
            if not mapping or min(mapping) < 0 or max(mapping) >= len(motor_effort):
                raise PolicyContractError(
                    f"runtime joint2motor indices {mapping} are outside {len(motor_effort)} motor effort limits."
                )
            saved_joint_effort = [float(value) for value in saved_effort]
            runtime_joint_effort = [motor_effort[motor_idx] for motor_idx in mapping]
            if not all(math.isfinite(value) and value >= 0.0 for value in saved_joint_effort + runtime_joint_effort):
                raise PolicyContractError("Action-scale effort limits must be finite and non-negative.")
            mismatches = [
                index
                for index, (expected, actual) in enumerate(zip(saved_joint_effort, runtime_joint_effort, strict=True))
                if not math.isclose(expected, actual, rel_tol=1.0e-6, abs_tol=1.0e-6)
            ]
            if mismatches:
                preview = [
                    (index, saved_joint_effort[index], runtime_joint_effort[index])
                    for index in mismatches[:8]
                ]
                raise PolicyContractError(
                    "Runtime motor effort limits change the training action-scale contract; "
                    f"joint(expected, runtime)={preview}."
                )

    if configured_groups != expected_groups:
        raise PolicyContractError(
            "Runtime actor observation group order does not match the policy metadata: "
            f"expected={expected_groups}, configured={configured_groups}. "
            "Select the inference preset recorded by the checkpoint instead of overriding the observation layout."
        )

    for group_name in expected_groups:
        saved_group = saved_groups.get(group_name)
        if not isinstance(saved_group, Mapping):
            raise PolicyContractError(f"Policy metadata is missing actor observation group {group_name!r}.")
        if "terms" not in saved_group:
            raise PolicyContractError(
                f"Policy metadata is missing terms for actor observation group {group_name!r}."
            )
        saved_terms_cfg = saved_group["terms"]
        if not isinstance(saved_terms_cfg, Mapping):
            raise PolicyContractError(f"Policy metadata has invalid terms for actor group {group_name!r}.")
        if not all(isinstance(name, str) and name for name in saved_terms_cfg):
            raise PolicyContractError(
                f"Policy metadata actor group {group_name!r} must use non-empty string term names."
            )

        # Both the training and inference observation managers concatenate
        # terms in lexical order.
        expected_terms = sorted(saved_terms_cfg)
        configured_terms = sorted(observation.obs_dict[group_name])
        if configured_terms != expected_terms:
            raise PolicyContractError(
                f"Runtime terms for actor group {group_name!r} do not match policy metadata: "
                f"expected={expected_terms}, configured={configured_terms}."
            )

        if "history_length" not in saved_group:
            raise PolicyContractError(
                f"Policy metadata is missing history_length for actor group {group_name!r}."
            )
        expected_history = saved_group["history_length"]
        if isinstance(expected_history, bool) or not isinstance(expected_history, int) or expected_history < 1:
            raise PolicyContractError(
                f"Policy metadata actor group {group_name!r} has invalid history_length "
                f"{expected_history!r}."
            )
        configured_history = observation.history_length_dict.get(group_name)
        if (
            isinstance(configured_history, bool)
            or not isinstance(configured_history, int)
            or configured_history < 1
        ):
            raise PolicyContractError(
                f"Runtime actor group {group_name!r} has invalid or missing history_length "
                f"{configured_history!r}."
            )
        if configured_history != expected_history:
            raise PolicyContractError(
                f"Runtime history for actor group {group_name!r} does not match policy metadata: "
                f"expected={expected_history}, configured={configured_history}."
            )

        if "concatenate" not in saved_group or not isinstance(saved_group["concatenate"], bool):
            raise PolicyContractError(
                f"Policy metadata actor group {group_name!r} must declare boolean concatenate."
            )
        expected_concatenate = saved_group["concatenate"]
        configured_concatenate = observation.group_concatenate.get(group_name)
        if not isinstance(configured_concatenate, bool):
            raise PolicyContractError(
                f"Runtime actor group {group_name!r} has no canonical concatenate descriptor."
            )
        if not configured_concatenate:
            raise PolicyContractError(
                f"Runtime actor group {group_name!r} requests concatenate=false, which inference does not implement."
            )
        if configured_concatenate != expected_concatenate:
            raise PolicyContractError(
                f"Runtime concatenate setting for actor group {group_name!r} does not match policy metadata: "
                f"expected={expected_concatenate}, configured={configured_concatenate}."
            )

        if "enable_noise" not in saved_group or not isinstance(saved_group["enable_noise"], bool):
            raise PolicyContractError(
                f"Policy metadata actor group {group_name!r} must declare boolean enable_noise."
            )
        expected_enable_noise = saved_group["enable_noise"]
        configured_enable_noise = observation.group_enable_noise.get(group_name)
        if not isinstance(configured_enable_noise, bool):
            raise PolicyContractError(
                f"Runtime actor group {group_name!r} has no canonical enable_noise descriptor."
            )
        if configured_enable_noise != expected_enable_noise:
            raise PolicyContractError(
                f"Runtime training-noise setting for actor group {group_name!r} does not match policy metadata: "
                f"expected={expected_enable_noise}, configured={configured_enable_noise}."
            )

        for term_name in expected_terms:
            saved_term = saved_terms_cfg[term_name]
            if not isinstance(saved_term, Mapping):
                raise PolicyContractError(
                    f"Policy metadata observation term {group_name!r}.{term_name!r} must be a mapping."
                )
            descriptor = observation.term_descriptors.get(term_name)
            if not isinstance(descriptor, ObservationTermDescriptor):
                raise PolicyContractError(
                    f"Runtime observation term {term_name!r} has no canonical semantic descriptor."
                )

            required_term_fields = ("func", "params", "scale", "noise", "clip")
            missing_fields = [field for field in required_term_fields if field not in saved_term]
            if missing_fields:
                raise PolicyContractError(
                    f"Policy metadata observation term {group_name!r}.{term_name!r} is incomplete; "
                    f"missing={missing_fields}."
                )

            expected_func = saved_term["func"]
            if not isinstance(expected_func, str) or not expected_func:
                raise PolicyContractError(
                    f"Policy metadata observation term {term_name!r} must declare a non-empty func path."
                )
            if expected_func != descriptor.func:
                raise PolicyContractError(
                    f"Runtime func for observation term {term_name!r} does not match policy metadata: "
                    f"expected={expected_func!r}, configured={descriptor.func!r}."
                )

            expected_params = saved_term["params"]
            if not isinstance(expected_params, Mapping):
                raise PolicyContractError(
                    f"Policy metadata observation term {term_name!r}.params must be a mapping."
                )
            configured_params = descriptor.params
            if not isinstance(configured_params, Mapping):
                raise PolicyContractError(
                    f"Runtime observation term {term_name!r}.params descriptor must be a mapping."
                )
            expected_params_value = _canonical_metadata_value(
                expected_params,
                label=f"Policy metadata observation term {term_name}.params",
            )
            configured_params_value = _canonical_metadata_value(
                configured_params,
                label=f"Runtime observation term {term_name}.params",
            )
            if configured_params_value != expected_params_value:
                raise PolicyContractError(
                    f"Runtime params for observation term {term_name!r} do not match policy metadata: "
                    f"expected={dict(expected_params)!r}, configured={dict(configured_params)!r}."
                )

            expected_scale = _finite_scalar(
                saved_term["scale"],
                label=f"Policy metadata observation term {term_name}.scale",
            )
            try:
                configured_scale_raw = observation.obs_scales[term_name]
            except KeyError as exc:
                raise PolicyContractError(
                    f"Runtime observation term {term_name!r} has no configured scale."
                ) from exc
            configured_scale = _finite_scalar(
                configured_scale_raw,
                label=f"Runtime observation term {term_name}.scale",
            )
            if not math.isclose(configured_scale, expected_scale, rel_tol=0.0, abs_tol=1.0e-12):
                raise PolicyContractError(
                    f"Runtime scale for observation term {term_name!r} does not match policy metadata: "
                    f"expected={expected_scale}, configured={configured_scale}."
                )

            expected_noise = _finite_scalar(
                saved_term["noise"],
                label=f"Policy metadata observation term {term_name}.noise",
                non_negative=True,
            )
            configured_noise = _finite_scalar(
                descriptor.noise,
                label=f"Runtime observation term {term_name}.noise descriptor",
                non_negative=True,
            )
            if not math.isclose(configured_noise, expected_noise, rel_tol=0.0, abs_tol=1.0e-12):
                raise PolicyContractError(
                    f"Runtime training-noise descriptor for observation term {term_name!r} "
                    "does not match policy metadata: "
                    f"expected={expected_noise}, configured={configured_noise}."
                )

            expected_term_clip = _canonical_clip(
                saved_term["clip"],
                label=f"Policy metadata observation term {term_name}.clip",
            )
            configured_term_clip = _canonical_clip(
                descriptor.clip,
                label=f"Runtime observation term {term_name}.clip descriptor",
            )
            if configured_term_clip != expected_term_clip:
                raise PolicyContractError(
                    f"Runtime clip for observation term {term_name!r} does not match policy metadata: "
                    f"expected={expected_term_clip}, configured={configured_term_clip}."
                )

    layer_config = actor.get("layer_config", {})
    layer_config = layer_config if isinstance(layer_config, Mapping) else {}
    perception_name = actor_perception_input_name_from_metadata(metadata) or ""
    perception_contract_digest = perception_observation_contract_sha256_from_metadata(metadata)
    if perception_name:
        if perception_name in {"obs", "actor_obs", "time_step"} or perception_name == obs_input_name:
            raise PolicyContractError(
                f"Perception input name {perception_name!r} collides with a reserved actor/time input."
            )
        if perception_name not in input_shapes:
            raise PolicyContractError(
                f"Policy metadata requires perception input {perception_name!r}, "
                f"but ONNX inputs are {sorted(input_shapes)}."
            )
        height = layer_config.get("perception_input_height")
        width = layer_config.get("perception_input_width")
        onnx_perception_dim = _feature_dim(input_shapes, perception_name)
        if (
            not isinstance(height, Integral)
            or isinstance(height, bool)
            or int(height) <= 0
            or not isinstance(width, Integral)
            or isinstance(width, bool)
            or int(width) <= 0
        ):
            raise PolicyContractError(
                "Perception policy metadata must declare positive integer "
                "actor.layer_config.perception_input_height and perception_input_width; "
                f"got height={height!r}, width={width!r}."
            )
        expected_perception_dim = int(height) * int(width)
        if onnx_perception_dim != expected_perception_dim:
            raise PolicyContractError(
                f"ONNX perception input {perception_name!r} has dimension {onnx_perception_dim}, "
                f"but policy metadata requires {height}x{width}={expected_perception_dim}."
            )
        if perception_contract_digest is not None:
            perception_contract = metadata["perception_observation_contract"]
            contract_shape = perception_contract.get("camera_obs_shape")
            if (
                isinstance(contract_shape, (str, bytes))
                or not isinstance(contract_shape, Sequence)
                or len(contract_shape) != 2
                or any(
                    not isinstance(value, Integral) or isinstance(value, bool) or int(value) <= 0
                    for value in contract_shape
                )
            ):
                raise PolicyContractError(
                    "Perception observation contract camera_obs_shape must be two positive integers."
                )
            authenticated_shape = [int(value) for value in contract_shape]
            actor_shape = [int(height), int(width)]
            if authenticated_shape != actor_shape:
                raise PolicyContractError(
                    "Perception observation contract camera_obs_shape does not match the actor input geometry: "
                    f"contract={authenticated_shape}, actor={actor_shape}."
                )

        actor_encoder_raw = layer_config.get("perception_encoder_type")
        actor_encoder = actor_encoder_raw.strip().lower() if isinstance(actor_encoder_raw, str) else ""
        perception_cfg = experiment.get("perception") if isinstance(experiment, Mapping) else None
        metadata_encoder_raw = perception_cfg.get("encoder_type") if isinstance(perception_cfg, Mapping) else None
        metadata_encoder = (
            metadata_encoder_raw.strip().lower()
            if isinstance(metadata_encoder_raw, str)
            else ""
        )
        if actor_encoder.startswith("defm_") or metadata_encoder.startswith("defm_"):
            if not actor_encoder or not metadata_encoder or actor_encoder != metadata_encoder:
                raise PolicyContractError(
                    "DeFM policy metadata must declare the same encoder in "
                    "actor.layer_config.perception_encoder_type and perception.encoder_type; "
                    f"got actor={actor_encoder_raw!r}, perception={metadata_encoder_raw!r}."
                )
            if perception_cfg.get("output_mode") != "camera_depth":
                raise PolicyContractError(
                    f"DeFM encoder {actor_encoder!r} requires perception.output_mode='camera_depth'."
                )
            if perception_cfg.get("camera_warp_normalize") is not False:
                raise PolicyContractError(
                    f"DeFM encoder {actor_encoder!r} requires authenticated metric depth in meters; "
                    "perception.camera_warp_normalize must be explicitly false. Artifacts trained "
                    "with true use incompatible legacy depth semantics and require retraining."
                )
    elif "perception_obs" in input_shapes:
        raise PolicyContractError(
            "ONNX exposes perception_obs, but the serialized actor metadata has no perception input."
        )
    elif perception_contract_digest is not None:
        raise PolicyContractError(
            "ONNX metadata declares a perception observation contract, but the actor has no perception input."
        )

    allowed_inputs = {obs_input_name}
    if perception_name:
        allowed_inputs.add(perception_name)
    if "time_step" in input_shapes:
        allowed_inputs.add("time_step")
    unexpected_inputs = sorted(set(input_shapes) - allowed_inputs)
    if unexpected_inputs:
        raise PolicyContractError(
            "ONNX exposes inputs that are not declared by the authenticated actor contract: "
            f"{unexpected_inputs}."
        )

    saved_dof_names = metadata.get("dof_names")
    if not isinstance(saved_dof_names, (list, tuple)) or not all(
        isinstance(name, str) for name in saved_dof_names
    ):
        raise PolicyContractError("Policy metadata must contain a DOF-ordered string list in dof_names.")

    expected_dof_names = list(saved_dof_names)
    configured_dof_names = list(runtime_dof_names)
    if not all(isinstance(name, str) for name in configured_dof_names):
        raise PolicyContractError("Runtime DOF names must all be strings.")
    if len(expected_dof_names) != len(configured_dof_names):
        raise PolicyContractError(
            "Policy/runtime DOF counts do not match: "
            f"metadata={len(expected_dof_names)}, runtime={len(configured_dof_names)}."
        )
    if len(set(expected_dof_names)) != len(expected_dof_names):
        raise PolicyContractError(f"Policy metadata DOF names must be unique, got {expected_dof_names}.")
    if len(set(configured_dof_names)) != len(configured_dof_names):
        raise PolicyContractError(f"Runtime DOF names must be unique, got {configured_dof_names}.")
    if configured_dof_names != expected_dof_names:
        raise PolicyContractError(
            "Runtime DOF order does not match policy metadata: "
            f"expected={expected_dof_names}, configured={configured_dof_names}."
        )

    saved_robot = experiment.get("robot") if isinstance(experiment, Mapping) else None
    saved_init_state = saved_robot.get("init_state") if isinstance(saved_robot, Mapping) else None
    saved_defaults = (
        saved_init_state.get("default_joint_angles") if isinstance(saved_init_state, Mapping) else None
    )
    if not isinstance(saved_defaults, Mapping):
        raise PolicyContractError(
            "Policy metadata is missing robot.init_state.default_joint_angles."
        )
    missing_defaults = [name for name in expected_dof_names if name not in saved_defaults]
    if missing_defaults:
        raise PolicyContractError(
            "Policy metadata default joint-angle mapping is incomplete; "
            f"missing={missing_defaults}."
        )
    if len(runtime_default_dof_angles) != len(configured_dof_names):
        raise PolicyContractError(
            "Runtime default joint-angle count does not match the runtime DOF count: "
            f"defaults={len(runtime_default_dof_angles)}, dofs={len(configured_dof_names)}."
        )
    try:
        expected_defaults = [float(saved_defaults[name]) for name in expected_dof_names]
        configured_defaults = [float(value) for value in runtime_default_dof_angles]
    except (TypeError, ValueError) as exc:
        raise PolicyContractError("Default joint angles must be numeric.") from exc
    if not all(math.isfinite(value) for value in expected_defaults):
        raise PolicyContractError("Policy metadata default joint angles must all be finite.")
    if not all(math.isfinite(value) for value in configured_defaults):
        raise PolicyContractError("Runtime default joint angles must all be finite.")
    if configured_defaults != expected_defaults:
        raise PolicyContractError(
            "Runtime default joint angles do not match policy metadata; dof_pos would be shifted. "
            f"expected={expected_defaults}, configured={configured_defaults}."
        )

    for key, label in (("kp", "KP"), ("kd", "KD")):
        values = metadata.get(key)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise PolicyContractError(f"Policy metadata must contain DOF-ordered {label} values in {key}.")
        if len(values) != len(expected_dof_names):
            raise PolicyContractError(
                f"Policy metadata {label} count {len(values)} does not match DOF count "
                f"{len(expected_dof_names)}."
            )
        try:
            numeric_values = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise PolicyContractError(f"Policy metadata {label} values must be numeric.") from exc
        if not all(math.isfinite(value) and value >= 0.0 for value in numeric_values):
            raise PolicyContractError(
                f"Policy metadata {label} values must be finite and non-negative."
            )

    return True
