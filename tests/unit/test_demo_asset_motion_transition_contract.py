from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from holosoma_inference.utils.policy_contract import PolicyContractError


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "mujoco-web-wobj-track-demo" / "scripts" / "prepare_demo_assets.py"
_SPEC = importlib.util.spec_from_file_location("prepare_demo_assets_contract_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


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
        # These requested flags deliberately contradict global effective append
        # semantics and must never drive deployment asset lengths.
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
                                "enable_default_pose_prepend": True,
                                "default_pose_prepend_duration_s": 0.2,
                                "enable_default_pose_append": True,
                                "default_pose_append_duration_s": 0.2,
                            }
                        }
                    }
                }
            }
        },
    }


def test_demo_assets_use_effective_global_contract_not_raw_append_request() -> None:
    metadata = _metadata(
        semantics="global_multi_clip_runtime",
        prepend_steps=10,
        append_steps=0,
    )

    assert _MODULE._transition_step_counts(metadata) == (10, 0)


def test_demo_assets_preserve_authenticated_single_clip_static_transitions() -> None:
    metadata = _metadata(
        semantics="single_clip_static",
        prepend_steps=10,
        append_steps=10,
    )

    assert _MODULE._transition_step_counts(metadata) == (10, 10)


def test_demo_assets_fail_closed_without_authenticated_transition_contract() -> None:
    with pytest.raises(PolicyContractError, match="missing motion_transition_contract"):
        _MODULE._transition_step_counts({"experiment_config": {}})


def test_demo_asset_cli_defaults_to_effective_timeline_and_names_unsafe_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT_PATH)])
    assert _MODULE.parse_args().unsafe_skip_training_motion_transitions is False

    monkeypatch.setattr(
        sys,
        "argv",
        [str(_SCRIPT_PATH), "--unsafe-skip-training-motion-transitions"],
    )
    assert _MODULE.parse_args().unsafe_skip_training_motion_transitions is True
