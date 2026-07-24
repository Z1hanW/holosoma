from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from holosoma_inference.utils.policy_contract import (
    PolicyContractError,
    effective_motion_transition_settings_from_metadata,
    motion_transition_contract_from_metadata,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _metadata(*, semantics: str, prepend_steps: int, append_steps: int) -> dict:
    prepend_impl = "none"
    if prepend_steps:
        prepend_impl = "runtime_hold" if semantics == "global_multi_clip_runtime" else "static_splice"
    append_impl = "static_splice" if append_steps else "none"
    contract = {
        "version": 1,
        "control_dt_s": 0.02,
        "source_semantics": semantics,
        "prepend": {
            "implementation": prepend_impl,
            "applied": prepend_steps > 0,
            "steps": prepend_steps,
        },
        "append": {
            "implementation": append_impl,
            "applied": append_steps > 0,
            "steps": append_steps,
        },
    }
    digest = hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "motion_transition_contract": contract,
        "motion_transition_contract_sha256": digest,
        "experiment_config": {
            "simulator": {
                "_target_": "holosoma.simulator.isaacsim.isaacsim.IsaacSim",
                "config": {
                    "name": "isaacsim",
                    "sim": {
                        "fps": 200,
                        "control_decimation": 4,
                    }
                }
            },
            "command": {
                "setup_terms": {
                    "motion_command": {
                        "params": {
                            "motion_config": {
                                "enable_default_pose_prepend": prepend_steps > 0,
                                "default_pose_prepend_duration_s": prepend_steps * 0.02,
                                "enable_default_pose_append": append_steps > 0,
                                "default_pose_append_duration_s": append_steps * 0.02,
                            }
                        }
                    }
                }
            },
        },
    }


def test_global_eval_defaults_ignore_raw_requested_append() -> None:
    metadata = _metadata(
        semantics="global_multi_clip_runtime",
        prepend_steps=10,
        append_steps=0,
    )
    motion_cfg = (
        metadata["experiment_config"]["command"]["setup_terms"]["motion_command"]["params"][
            "motion_config"
        ]
    )
    # Global training intentionally ignored this raw append request.
    motion_cfg["enable_default_pose_append"] = True
    motion_cfg["default_pose_append_duration_s"] = 0.2

    settings = effective_motion_transition_settings_from_metadata(metadata)

    assert settings["source_semantics"] == "global_multi_clip_runtime"
    assert settings["prepend"] == {
        "implementation": "runtime_hold",
        "applied": True,
        "steps": 10,
        "duration_s": 0.2,
    }
    assert settings["append"] == {
        "implementation": "none",
        "applied": False,
        "steps": 0,
        "duration_s": 0.0,
    }


def test_missing_contract_with_ambiguous_requested_timeline_fails_closed() -> None:
    metadata = _metadata(
        semantics="global_multi_clip_runtime",
        prepend_steps=10,
        append_steps=0,
    )
    del metadata["motion_transition_contract"]
    del metadata["motion_transition_contract_sha256"]

    with pytest.raises(PolicyContractError, match="missing motion_transition_contract"):
        effective_motion_transition_settings_from_metadata(metadata)


def test_single_clip_eval_defaults_preserve_both_authenticated_phases() -> None:
    settings = effective_motion_transition_settings_from_metadata(
        _metadata(
            semantics="single_clip_static",
            prepend_steps=10,
            append_steps=10,
        )
    )

    assert settings["source_semantics"] == "single_clip_static"
    assert settings["prepend"]["steps"] == 10
    assert settings["append"]["steps"] == 10
    assert settings["prepend"]["applied"] is True
    assert settings["append"]["applied"] is True


@pytest.mark.parametrize(
    ("field", "unhashable_value"),
    [
        ("source_semantics", ["global_multi_clip_runtime"]),
        ("prepend_implementation", {"value": "runtime_hold"}),
    ],
)
def test_unhashable_transition_enum_values_raise_policy_contract_error(
    field: str,
    unhashable_value: object,
) -> None:
    metadata = _metadata(
        semantics="global_multi_clip_runtime",
        prepend_steps=10,
        append_steps=0,
    )
    contract = metadata["motion_transition_contract"]
    if field == "source_semantics":
        contract["source_semantics"] = unhashable_value
    else:
        contract["prepend"]["implementation"] = unhashable_value
    metadata["motion_transition_contract_sha256"] = hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(PolicyContractError, match="must be"):
        motion_transition_contract_from_metadata(metadata, required=True)


def test_requested_step_rounding_matches_training_half_even_semantics() -> None:
    metadata = _metadata(
        semantics="single_clip_static",
        prepend_steps=2,
        append_steps=0,
    )
    motion_cfg = (
        metadata["experiment_config"]["command"]["setup_terms"]["motion_command"]["params"][
            "motion_config"
        ]
    )
    # Training uses Python round(duration / dt): round(2.5) == 2.
    motion_cfg["default_pose_prepend_duration_s"] = 0.05

    settings = effective_motion_transition_settings_from_metadata(metadata)

    assert settings["prepend"]["steps"] == 2


@pytest.mark.parametrize("control_decimation", [2.5, True, "4"])
def test_effective_transition_rejects_non_integer_control_decimation(
    control_decimation: object,
) -> None:
    metadata = _metadata(
        semantics="single_clip_static",
        prepend_steps=10,
        append_steps=10,
    )
    metadata["experiment_config"]["simulator"]["config"]["sim"][
        "control_decimation"
    ] = control_decimation

    with pytest.raises(PolicyContractError, match="positive integer control_decimation"):
        effective_motion_transition_settings_from_metadata(metadata)


@pytest.mark.parametrize(
    ("target", "name", "message"),
    [
        ("holosoma.simulator.mujoco.mujoco.MuJoCo", "mujoco", "exact IsaacSim"),
        ("holosoma.simulator.isaacgym.isaacgym.IsaacGym", "isaacgym", "exact IsaacSim"),
        ("holosoma.simulator.isaacsim.isaacsim.IsaacSim", "mujoco", "matching simulator"),
        (None, "isaacsim", "simulator._target_"),
        ("holosoma.simulator.isaacsim.isaacsim.IsaacSim", None, "simulator.config.name"),
    ],
)
def test_applied_transition_requires_exact_consistent_isaacsim_backend(
    target: object,
    name: object,
    message: str,
) -> None:
    metadata = _metadata(
        semantics="single_clip_static",
        prepend_steps=10,
        append_steps=10,
    )
    simulator = metadata["experiment_config"]["simulator"]
    if target is None:
        simulator.pop("_target_")
    else:
        simulator["_target_"] = target
    if name is None:
        simulator["config"].pop("name")
    else:
        simulator["config"]["name"] = name

    with pytest.raises(PolicyContractError, match=message):
        effective_motion_transition_settings_from_metadata(metadata)


def test_inactive_global_contract_can_describe_mujoco_backend() -> None:
    metadata = _metadata(
        semantics="global_multi_clip_runtime",
        prepend_steps=0,
        append_steps=0,
    )
    simulator = metadata["experiment_config"]["simulator"]
    simulator["_target_"] = "holosoma.simulator.mujoco.mujoco.MuJoCo"
    simulator["config"]["name"] = "mujoco"

    settings = effective_motion_transition_settings_from_metadata(metadata)

    assert settings["prepend"]["applied"] is False
    assert settings["append"]["applied"] is False


def test_infer_box_tracking_launcher_is_closed_over_effective_helper_and_both_phases() -> None:
    source = (_REPO_ROOT / "infer_box_tracking.sh").read_text(encoding="utf-8")

    assert "effective_motion_transition_settings_from_metadata" in source
    assert "resolve_exact_checkpoint.py" in source
    assert "HOLOSOMA_EXPECTED_EVALUATION_CHECKPOINT_SHA256" in source
    assert "CHECKPOINT_SAVED_ENABLE_DEFAULT_POSE_PREPEND" not in source
    assert "CHECKPOINT_SAVED_DEFAULT_POSE_PREPEND_DURATION_S" not in source
    assert "checkpoint_metadata_error" in source
    assert "refusing to silently disable motion transitions" in source
    assert "motion_config.enable_default_pose_append \"${ENABLE_DEFAULT_POSE_APPEND}\"" in source
    assert (
        "motion_config.default_pose_append_duration_s \"${DEFAULT_POSE_APPEND_DURATION_S}\""
        in source
    )


def test_mj_track_launch_defaults_use_effective_transition_helper() -> None:
    source = (_REPO_ROOT / "mj_track.sh").read_text(encoding="utf-8")
    marker = "apply_training_motion_launch_defaults()"
    block = source[source.index(marker) : source.index("apply_training_motion_launch_defaults ", source.index(marker) + len(marker))]

    assert "effective_motion_transition_settings_from_metadata" in block
    assert "motion_transition_contract_from_metadata" not in block
    assert "requested_transition" not in block
