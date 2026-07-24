from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import pickle
import subprocess
from unittest.mock import patch

import pytest
import torch

from holosoma.utils.policy_init_preflight import (
    ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV,
    POLICY_INIT_REQUIRED_TERMINAL_TARGET_ENV,
    canonical_actor_contract,
    required_policy_init_terminal_target_from_env,
    validate_policy_init_checkpoint,
    validate_policy_init_terminal_source_payload,
)
from holosoma.utils.checkpoint_validation import (
    fixed_bc_dataset_sha256,
    fixed_bc_global_dataset_sha256,
    terminal_fixed_bc_eval_state_sha256,
)
from holosoma.utils.training_provenance import (
    disabled_checkpoint_sha256,
    embedded_runtime_asset_manifest_sha256,
)


@pytest.fixture(autouse=True)
def _explicit_legacy_identity_hatch_for_unprovenanced_fixtures(monkeypatch):
    monkeypatch.setenv(ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV, "1")


def _term(func: str) -> dict:
    return {"func": func, "params": {}, "scale": 1.0, "noise": 0.0, "clip": None}


def _group(*terms: tuple[str, str], history_length: int = 1) -> dict:
    return {
        "terms": {name: _term(func) for name, func in terms},
        "concatenate": True,
        "enable_noise": False,
        "history_length": history_length,
    }


def _config(*, normalize_actor_obs: bool = False) -> dict:
    return {
        "training": {"num_envs": 32},
        "algo": {
            "config": {
                "module_dict": {
                    "actor": {
                        "type": "MLPPerceptionEncoder",
                        "input_dim": ["root", "proprio"],
                        "output_dim": [2],
                        "layer_config": {
                            "hidden_dims": [8, 4],
                            "activation": "ELU",
                            "module_input_name": ["root", "proprio"],
                            "perception_input_name": "perception_obs",
                            "perception_input_height": 2,
                            "perception_input_width": 3,
                        },
                        "min_noise_std": 0.01,
                    },
                    "critic": {"type": "MLP", "input_dim": ["critic"]},
                },
                "normalize_actor_obs": normalize_actor_obs,
                "obs_normalizer_eps": 0.01,
                "obs_normalizer_until": None,
            }
        },
        "observation": {
            "groups": {
                "root": _group(("target", "pkg:target")),
                "proprio": _group(
                    ("joint_pos", "pkg:joint_pos"),
                    ("joint_vel", "pkg:joint_vel"),
                    history_length=1,
                ),
                "perception_obs": _group(("depth", "pkg:depth")),
                "critic": _group(("privileged", "pkg:privileged")),
            },
            "clip_observations": 100.0,
        },
        "perception": {
            "enabled": True,
            "output_mode": "camera_depth",
            "camera_width": 6,
            "camera_height": 4,
            "camera_warp_resize": [2, 3],
            "camera_warp_normalize": True,
        },
        "robot": {
            "actions_dim": 2,
            "dof_names": ["left", "right"],
            "dof_effort_limit_list": [20.0, 20.0],
            "init_state": {"default_joint_angles": {"left": 0.1, "right": -0.1}},
            "control": {
                "control_type": "P",
                "stiffness": {".*": 40.0},
                "damping": {".*": 2.0},
                "action_scale": 0.25,
                "action_clip_value": 100.0,
                "clip_actions": True,
                "clip_torques": True,
                "action_scales_by_effort_limit_over_p_gain": True,
            },
        },
        "action": {
            "terms": {
                "joint_control": {
                    "func": "pkg:JointPositionActionTerm",
                    "params": {},
                    "scale": 1.0,
                    "clip": None,
                }
            }
        },
    }


def _save(tmp_path, config: dict, *, normalizer_state=None):
    payload = {
        "experiment_config": config,
        "actor_model_state_dict": {"weight": torch.ones(2, 2)},
    }
    if normalizer_state is not None:
        payload["actor_obs_normalizer_state"] = normalizer_state
    path = tmp_path / "policy.pt"
    torch.save(payload, path)
    return path


def _terminal_payload(config: dict, *, target: int = 8) -> dict:
    config = copy.deepcopy(config)
    config["algo"]["config"]["num_learning_iterations"] = target
    budget = 3
    world_size = 2
    required_tensors = {"actor_obs_raw", "teacher_actions", "actor_perception"}
    rows_by_rank = (2, 1)
    states_by_rank = {}
    digest_by_rank = {}
    for rank, rows in enumerate(rows_by_rank):
        rank_state = {
            "allocation_version": 1,
            "allocation_scheme": "rank_quotient_remainder",
            "global_sample_budget": budget,
            "world_size": world_size,
            "rank": rank,
            "local_target": rows,
            "ready": True,
            "size": rows,
            "actor_obs_raw": torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3),
            "teacher_actions": torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2),
            "actor_perception": torch.arange(rows * 6, dtype=torch.float32).reshape(rows, 6),
        }
        states_by_rank[str(rank)] = rank_state
        digest_by_rank[str(rank)] = fixed_bc_dataset_sha256(
            rank_state,
            expected_rows=rows,
            required_tensor_keys=required_tensors,
            context=f"fixture rank {rank}",
        )
    global_digest = fixed_bc_global_dataset_sha256(
        digest_by_rank,
        global_sample_budget=budget,
        world_size=world_size,
    )
    completed = target - 1
    terminal_state = {
        "version": 1,
        "terminal_observation": True,
        "completed_iteration": completed,
        "next_iteration": target,
        "run_target_iteration": target,
        "scheduled_evaluation": False,
        "guard_enabled": False,
        "guard_applied": False,
        "fixed_bc_eval_log_interval": 2,
        "fixed_bc_eval_num_samples": budget,
        "world_size": world_size,
        "fixed_bc_global_dataset_sha256": global_digest,
        "fixed_bc_guard_config_sha256": "f" * 64,
        "fixed_bc_guard_state_sha256": None,
        "fixed_bc_guard_threshold_mu_mse": None,
        "fixed_bc_terminal_within_threshold": None,
        "fixed_bc_mu_mse": 0.04,
        "fixed_bc_num_samples": budget,
        "fixed_bc_weighted_num_samples": float(budget),
        "fixed_bc_expected_weighted_num_samples": float(budget),
        "fixed_bc_rank_strata": world_size,
    }
    return {
        "iter": completed,
        "iteration": completed,
        "next_iter": target,
        "experiment_config": config,
        "actor_model_state_dict": {"weight": torch.ones(2, 2)},
        "fixed_bc_eval_by_rank": states_by_rank,
        "terminal_fixed_bc_eval": terminal_state,
        "terminal_fixed_bc_eval_sha256": terminal_fixed_bc_eval_state_sha256(
            terminal_state
        ),
    }


def _provenance(policy_sha256: str) -> dict:
    runtime_asset_manifest = {"version": 2, "fixture": "policy-init-preflight"}
    return {
        "version": 2,
        "teacher_sha256": "a" * 64,
        "policy_init_enabled": True,
        "policy_init_sha256": policy_sha256,
        "training_resume_enabled": True,
        "training_resume_sha256": "b" * 64,
        "motion_shard_manifest_sha256": "c" * 64,
        "contact_sidecar_manifest_sha256": "d" * 64,
        "source_bundle_sha256": "e" * 64,
        "runtime_asset_manifest_phase": "final",
        "runtime_asset_manifest_sha256": embedded_runtime_asset_manifest_sha256(
            runtime_asset_manifest
        ),
        "runtime_asset_manifest": runtime_asset_manifest,
    }


class _UnsafeCheckpointPayload:
    """Fixture that would create a marker under unrestricted pickle loading."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self):
        return subprocess.call, (["touch", self.marker],)


def test_policy_init_accepts_identical_actor_contract(tmp_path):
    config = _config()
    checkpoint = _save(tmp_path, config)
    validate_policy_init_checkpoint(checkpoint, copy.deepcopy(config))


def test_required_terminal_policy_init_accepts_exact_current_checkpoint(tmp_path):
    config = _config()
    payload = _terminal_payload(config)
    checkpoint = tmp_path / "terminal-policy.pt"
    torch.save(payload, checkpoint)

    terminal_state = validate_policy_init_terminal_source_payload(
        payload,
        required_target=8,
    )
    validate_policy_init_checkpoint(
        checkpoint,
        copy.deepcopy(config),
        required_terminal_target=8,
    )

    assert terminal_state["completed_iteration"] == 7
    assert terminal_state["next_iteration"] == 8
    assert terminal_state["run_target_iteration"] == 8


@pytest.mark.parametrize("missing_field", ["iter", "iteration", "next_iter"])
def test_required_terminal_policy_init_rejects_missing_explicit_iteration_field(
    missing_field,
):
    payload = _terminal_payload(_config())
    del payload[missing_field]

    with pytest.raises(ValueError, match=f"missing explicit {missing_field!r}"):
        validate_policy_init_terminal_source_payload(payload, required_target=8)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("iter", 6),
        ("iter", True),
        ("iteration", 6),
        ("next_iter", 9),
        ("next_iter", "8"),
    ],
)
def test_required_terminal_policy_init_rejects_nonexact_iteration_fields(field, value):
    payload = _terminal_payload(_config())
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        validate_policy_init_terminal_source_payload(payload, required_target=8)


def test_required_terminal_policy_init_rejects_wrong_saved_target():
    payload = _terminal_payload(_config())
    payload["experiment_config"]["algo"]["config"]["num_learning_iterations"] = 9

    with pytest.raises(ValueError, match="saved run target"):
        validate_policy_init_terminal_source_payload(payload, required_target=8)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("terminal_fixed_bc_eval"),
        lambda payload: payload.pop("terminal_fixed_bc_eval_sha256"),
        lambda payload: payload.__setitem__("terminal_fixed_bc_eval_sha256", "0" * 64),
    ],
)
def test_required_terminal_policy_init_rejects_missing_or_wrong_terminal_digest(mutation):
    payload = _terminal_payload(_config())
    mutation(payload)

    with pytest.raises(ValueError, match="terminal fixed-BC|authenticate its state"):
        validate_policy_init_terminal_source_payload(payload, required_target=8)


def test_required_terminal_policy_init_rejects_mutated_frozen_dataset():
    payload = _terminal_payload(_config())
    payload["fixed_bc_eval_by_rank"]["0"]["actor_obs_raw"][0, 0] += 1.0

    with pytest.raises(ValueError, match="does not authenticate.*frozen dataset"):
        validate_policy_init_terminal_source_payload(payload, required_target=8)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, None), ("", None), ("1", 1), ("8", 8), ("40000", 40000)],
)
def test_required_terminal_target_environment_parser_accepts_canonical_values(raw, expected):
    environ = {} if raw is None else {POLICY_INIT_REQUIRED_TERMINAL_TARGET_ENV: raw}

    assert required_policy_init_terminal_target_from_env(environ) == expected


@pytest.mark.parametrize("raw", ["0", "00", "08", "+8", " 8", "8 ", "8.0", "true"])
def test_required_terminal_target_environment_parser_rejects_aliases(raw):
    with pytest.raises(ValueError, match="canonical ASCII positive integer|must be positive"):
        required_policy_init_terminal_target_from_env(
            {POLICY_INIT_REQUIRED_TERMINAL_TARGET_ENV: raw}
        )


def test_policy_init_requires_authenticated_current_provenance_by_default(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV, raising=False)
    checkpoint = _save(tmp_path, _config())

    with pytest.raises(ValueError, match="requires finalized current training provenance"):
        validate_policy_init_checkpoint(checkpoint, _config())


def test_policy_init_legacy_identity_hatch_must_be_exact(tmp_path, monkeypatch):
    monkeypatch.setenv(ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV, "true")
    checkpoint = _save(tmp_path, _config())

    with pytest.raises(ValueError, match="must be exactly 0 or 1"):
        validate_policy_init_checkpoint(checkpoint, _config())


def test_policy_init_preflight_rejects_disabled_current_lineage_before_load(tmp_path):
    checkpoint = _save(tmp_path, _config())
    provenance = _provenance(hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    provenance["policy_init_enabled"] = False
    provenance["policy_init_sha256"] = disabled_checkpoint_sha256("policy_init")

    with (
        patch(
            "holosoma.utils.policy_init_preflight.load_verified_torch_checkpoint"
        ) as load_mock,
        pytest.raises(ValueError, match="does not enable policy initialization"),
    ):
        validate_policy_init_checkpoint(
            checkpoint,
            _config(),
            current_provenance=provenance,
        )

    load_mock.assert_not_called()


def test_policy_init_preflight_parses_hatch_before_provenanced_load(
    tmp_path,
    monkeypatch,
):
    checkpoint = _save(tmp_path, _config())
    provenance = _provenance(hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    monkeypatch.setenv(ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV, "true")

    with (
        patch(
            "holosoma.utils.policy_init_preflight.load_verified_torch_checkpoint"
        ) as load_mock,
        pytest.raises(ValueError, match="must be exactly 0 or 1"),
    ):
        validate_policy_init_checkpoint(
            checkpoint,
            _config(),
            current_provenance=provenance,
        )

    load_mock.assert_not_called()


def test_policy_init_preflight_never_executes_arbitrary_pickle_globals(tmp_path):
    marker = tmp_path / "pickle_executed"
    checkpoint = tmp_path / "unsafe.pt"
    torch.save(
        {
            "experiment_config": _config(),
            "actor_model_state_dict": {"weight": torch.ones(2, 2)},
            "unused": _UnsafeCheckpointPayload(str(marker)),
        },
        checkpoint,
    )

    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        validate_policy_init_checkpoint(checkpoint, _config())
    assert not marker.exists()


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
def test_policy_init_rejects_non_finite_actor_state(tmp_path, non_finite):
    config = _config()
    checkpoint = _save(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["actor_model_state_dict"]["weight"][0, 0] = non_finite
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="actor_model_state_dict.*non-finite"):
        validate_policy_init_checkpoint(checkpoint, config)


def test_policy_init_finite_check_ignores_unused_critic_state(tmp_path):
    config = _config()
    checkpoint = _save(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["critic_model_state_dict"] = {"unused": torch.tensor(float("nan"))}
    torch.save(payload, checkpoint)

    validate_policy_init_checkpoint(checkpoint, config)


def test_policy_init_requires_serialized_experiment_config(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    torch.save({"actor_model_state_dict": {}}, checkpoint)
    with pytest.raises(ValueError, match="no serialized experiment_config"):
        validate_policy_init_checkpoint(checkpoint, _config())


def test_policy_init_rejects_equal_shape_observation_term_reordering(tmp_path):
    saved = _config()
    current = copy.deepcopy(saved)
    current["observation"]["groups"]["proprio"]["terms"] = {
        "joint_vel": _term("pkg:joint_vel"),
        "joint_pos": _term("pkg:joint_pos"),
    }
    checkpoint = _save(tmp_path, saved)
    with pytest.raises(ValueError, match=r"observation_groups\[1\]\.terms\[0\]\.name"):
        validate_policy_init_checkpoint(checkpoint, current)


@pytest.mark.parametrize(
    ("mutate", "expected_path"),
    [
        (
            lambda cfg: cfg["algo"]["config"]["module_dict"]["actor"].update(
                input_dim=["proprio", "root"]
            ),
            "actor_input_groups",
        ),
        (
            lambda cfg: cfg["observation"]["groups"]["proprio"]["terms"]["joint_pos"].update(
                scale=2.0
            ),
            "scale",
        ),
        (
            lambda cfg: cfg["observation"]["groups"]["proprio"].update(history_length=2),
            "history_length",
        ),
        (
            lambda cfg: cfg["algo"]["config"].update(obs_normalizer_eps=0.1),
            "obs_normalizer_eps",
        ),
        (
            lambda cfg: cfg["perception"].update(camera_warp_normalize=False),
            "camera_warp_normalize",
        ),
        (
            lambda cfg: cfg["robot"].update(dof_names=["right", "left"]),
            "dof_names",
        ),
        (
            lambda cfg: cfg["robot"]["control"].update(action_scale=0.5),
            "action_scale",
        ),
        (
            lambda cfg: cfg["action"]["terms"]["joint_control"].update(scale=0.5),
            "action_terms",
        ),
    ],
)
def test_policy_init_rejects_actor_semantic_drift(tmp_path, mutate, expected_path):
    saved = _config()
    current = copy.deepcopy(saved)
    mutate(current)
    checkpoint = _save(tmp_path, saved)
    with pytest.raises(ValueError, match=expected_path):
        validate_policy_init_checkpoint(checkpoint, current)


def test_policy_init_rejects_contact_aware_root_command_mode_drift(tmp_path):
    saved = _config()
    saved["observation"]["groups"]["root"]["terms"]["target"]["func"] = (
        "holosoma.managers.observation.terms.wbt:sparse_target_root_trajectory_command_contact_aware"
    )
    saved["command"] = {
        "setup_terms": {
            "motion_command": {
                "params": {
                    "motion_config": {
                        # Missing in the historical box initializer means the
                        # runtime default, tracking_error.
                    }
                }
            }
        }
    }
    current = copy.deepcopy(saved)
    current["command"]["setup_terms"]["motion_command"]["params"]["motion_config"].update(
        contact_aware_sparse_root_command_mode="t1_aligned_segment",
        contact_aware_sparse_root_segment_steps=30,
        contact_aware_sparse_root_zero_yaw_threshold_deg=0.0,
    )
    checkpoint = _save(tmp_path, saved)

    with pytest.raises(ValueError, match="contact_aware_sparse_root_command_mode"):
        validate_policy_init_checkpoint(checkpoint, current)


def test_policy_init_contact_aware_legacy_default_matches_explicit_tracking_error():
    saved = _config()
    saved["observation"]["groups"]["root"]["terms"]["target"]["func"] = (
        "holosoma.managers.observation.terms.wbt:sparse_target_root_trajectory_command_contact_aware"
    )
    saved["command"] = {
        "setup_terms": {"motion_command": {"params": {"motion_config": {}}}}
    }
    current = copy.deepcopy(saved)
    current["command"]["setup_terms"]["motion_command"]["params"]["motion_config"] = {
        "contact_aware_sparse_root_command_mode": "tracking_error"
    }

    assert canonical_actor_contract(saved) == canonical_actor_contract(current)


def test_policy_init_rejects_contact_aware_carry_window_drift(tmp_path):
    saved = _config()
    saved["observation"]["groups"]["root"]["terms"]["target"]["func"] = (
        "holosoma.managers.observation.terms.wbt:sparse_target_root_trajectory_command_contact_aware"
    )
    saved["command"] = {
        "setup_terms": {
            "motion_command": {
                "params": {
                    "motion_config": {
                        "contact_aware_carry_window_mode": "peak_height",
                        "contact_aware_peak_height_alpha": 0.91,
                        "contact_aware_peak_height_smoothing_steps": 5,
                    }
                }
            }
        }
    }
    current = copy.deepcopy(saved)
    current["command"]["setup_terms"]["motion_command"]["params"]["motion_config"][
        "contact_aware_peak_height_alpha"
    ] = 0.75
    checkpoint = _save(tmp_path, saved)

    with pytest.raises(ValueError, match="contact_aware_peak_height_alpha"):
        validate_policy_init_checkpoint(checkpoint, current)


def test_policy_init_button_window_legacy_default_and_drift_are_fail_closed(tmp_path):
    saved = _config()
    saved["observation"]["groups"]["root"]["terms"]["target"]["func"] = (
        "holosoma.managers.observation.terms.wbt:drop_button"
    )
    saved["command"] = {
        "setup_terms": {"motion_command": {"params": {"motion_config": {}}}}
    }
    current = copy.deepcopy(saved)
    motion_config = current["command"]["setup_terms"]["motion_command"]["params"][
        "motion_config"
    ]
    motion_config["contact_aware_button_window_mode"] = "contact_interval"
    checkpoint = _save(tmp_path, saved)

    validate_policy_init_checkpoint(checkpoint, current)

    motion_config["contact_aware_button_window_mode"] = "kinematic_lift"
    with pytest.raises(ValueError, match="contact_aware_button_window_mode"):
        validate_policy_init_checkpoint(checkpoint, current)


def test_policy_init_ignores_button_window_mode_without_button_consumer():
    saved = _config()
    saved["command"] = {
        "setup_terms": {
            "motion_command": {
                "params": {
                    "motion_config": {
                        "contact_aware_button_window_mode": "contact_interval"
                    }
                }
            }
        }
    }
    current = copy.deepcopy(saved)
    current["command"]["setup_terms"]["motion_command"]["params"]["motion_config"][
        "contact_aware_button_window_mode"
    ] = "kinematic_lift"

    assert canonical_actor_contract(saved) == canonical_actor_contract(current)


def test_policy_init_normalized_actor_requires_complete_state(tmp_path):
    config = _config(normalize_actor_obs=True)
    checkpoint = _save(tmp_path, config, normalizer_state={"root": {"count": torch.tensor(1)}})
    with pytest.raises(ValueError, match="normalizer_state keys"):
        validate_policy_init_checkpoint(checkpoint, config)

    checkpoint = _save(
        tmp_path,
        config,
        normalizer_state={
            "root": {"count": torch.tensor(1)},
            "proprio": {"count": torch.tensor(1)},
        },
    )
    validate_policy_init_checkpoint(checkpoint, config)


def test_policy_init_rejects_non_finite_enabled_normalizer_state(tmp_path):
    config = _config(normalize_actor_obs=True)
    checkpoint = _save(
        tmp_path,
        config,
        normalizer_state={
            "root": {"count": torch.tensor(float("nan"))},
            "proprio": {"count": torch.tensor(1.0)},
        },
    )

    with pytest.raises(ValueError, match="actor_obs_normalizer_state.*non-finite"):
        validate_policy_init_checkpoint(checkpoint, config)


def test_policy_init_verifies_current_training_provenance_digest(tmp_path):
    config = _config()
    checkpoint = _save(tmp_path, config)
    actual_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    validate_policy_init_checkpoint(checkpoint, config, current_provenance=_provenance(actual_sha256))

    with pytest.raises(ValueError, match="does not identify the policy-init checkpoint"):
        validate_policy_init_checkpoint(checkpoint, config, current_provenance=_provenance("f" * 64))


def test_canonical_actor_contract_ignores_critic_and_training_schedule():
    saved = _config()
    current = copy.deepcopy(saved)
    current["algo"]["config"]["module_dict"]["critic"] = {"type": "OtherCritic"}
    current["algo"]["config"]["num_learning_iterations"] = 123
    assert canonical_actor_contract(saved) == canonical_actor_contract(current)


def test_canonical_actor_contract_resolves_legacy_defaults_and_symbolic_action_dim():
    saved = _config()
    current = copy.deepcopy(saved)
    current["algo"]["config"]["module_dict"]["actor"]["output_dim"] = ["robot_action_dim"]
    current_actor = current["algo"]["config"]["module_dict"]["actor"]
    current_actor["max_noise_std"] = None
    current_actor["layer_config"].update(
        flow_integration_steps=4,
        flow_train_noise_std=1.0,
        flow_time_epsilon=1e-4,
        flow_inference_noise_std=0.0,
    )
    assert canonical_actor_contract(saved) == canonical_actor_contract(current)


def test_canonical_actor_contract_materializes_exact_legacy_perception_defaults():
    saved = _config()
    current = copy.deepcopy(saved)
    current["algo"]["config"]["module_dict"]["actor"]["layer_config"][
        "perception_pretrained_sha256"
    ] = None
    current["perception"].update(
        encoder_pretrained_sha256=None,
        reset_refresh_semantics="legacy_full_v1",
        camera_warp_hole_seed_semantics="legacy_fixed_v1",
        camera_warp_hole_octave_profile="legacy_single_octave_v1",
    )

    assert canonical_actor_contract(saved) == canonical_actor_contract(current)


def test_policy_init_rejects_silent_legacy_to_targeted_reset_refresh_migration(tmp_path):
    saved = _config()
    current = copy.deepcopy(saved)
    current["algo"]["config"]["module_dict"]["actor"]["layer_config"][
        "perception_pretrained_sha256"
    ] = None
    current["perception"].update(
        encoder_pretrained_sha256=None,
        reset_refresh_semantics="targeted_v2",
    )
    checkpoint = _save(tmp_path, saved)

    with pytest.raises(ValueError, match="reset_refresh_semantics"):
        validate_policy_init_checkpoint(checkpoint, current)


def test_policy_init_does_not_relabel_missing_hole_seed_contract_as_rank_local_v2(tmp_path):
    saved = _config()
    current = copy.deepcopy(saved)
    saved["perception"].update(
        camera_warp_enable_holes=True,
        camera_warp_hole_prob=0.2,
    )
    current["perception"].update(
        camera_warp_enable_holes=True,
        camera_warp_hole_prob=0.2,
        camera_warp_hole_seed_semantics="rank_local_v2",
    )
    checkpoint = _save(tmp_path, saved)

    with pytest.raises(ValueError, match="camera_warp_hole_seed_semantics"):
        validate_policy_init_checkpoint(checkpoint, current)


def test_policy_init_accepts_matching_rank_local_hole_seed_contract(tmp_path):
    saved = _config()
    saved["perception"].update(
        camera_warp_enable_holes=True,
        camera_warp_hole_prob=0.2,
        camera_warp_hole_seed_semantics="rank_local_v2",
        camera_warp_hole_octave_profile="legacy_single_octave_v1",
    )
    checkpoint = _save(tmp_path, saved)

    validate_policy_init_checkpoint(checkpoint, copy.deepcopy(saved))


def test_policy_init_rejects_implicit_hole_reference_batch_migration(tmp_path):
    saved = _config()
    current = copy.deepcopy(saved)
    saved["perception"].update(
        camera_warp_enable_holes=True,
        camera_warp_hole_prob=0.2,
    )
    current["perception"].update(
        camera_warp_enable_holes=True,
        camera_warp_hole_prob=0.2,
    )
    saved["training"]["num_envs"] = 4096
    current["training"]["num_envs"] = 64
    checkpoint = _save(tmp_path, saved)

    with pytest.raises(ValueError, match="camera_warp_hole_reference_batch_size"):
        validate_policy_init_checkpoint(checkpoint, current)


def test_policy_init_explicit_hole_reference_preserves_legacy_effective_batch(tmp_path):
    saved = _config()
    current = copy.deepcopy(saved)
    saved["perception"].update(
        camera_warp_enable_holes=True,
        camera_warp_hole_prob=0.2,
    )
    current["perception"].update(
        camera_warp_enable_holes=True,
        camera_warp_hole_prob=0.2,
    )
    saved["training"]["num_envs"] = 4096
    current["training"]["num_envs"] = 64
    current["perception"]["camera_warp_hole_reference_batch_size"] = 4096
    checkpoint = _save(tmp_path, saved)

    validate_policy_init_checkpoint(checkpoint, current)


def test_policy_init_ignores_hole_reference_batch_when_holes_are_inactive(tmp_path):
    saved = _config()
    current = copy.deepcopy(saved)
    saved["training"]["num_envs"] = 4096
    current["training"]["num_envs"] = 64
    current["perception"]["camera_warp_hole_reference_batch_size"] = 8192
    checkpoint = _save(tmp_path, saved)

    validate_policy_init_checkpoint(checkpoint, current)


def test_train_agent_policy_init_preflight_runs_before_sim_and_installs_local_path(
    tmp_path,
    monkeypatch,
):
    from holosoma.config_values.experiment import DEFAULTS
    from holosoma.train_agent import _preflight_policy_init_before_sim

    checkpoint = tmp_path / "policy.pt"
    torch.save({"placeholder": True}, checkpoint)
    config = copy.deepcopy(DEFAULTS["g1_29dof_wbt_w_object_distill_sparse_root_cmd"])
    config = dataclasses.replace(
        config,
        training=dataclasses.replace(config.training, policy_init_checkpoint=str(checkpoint)),
    )
    monkeypatch.setenv("WORLD_SIZE", "2")
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="verified\n", stderr="")

    with (
        patch("holosoma.train_agent.apply_observation_overrides", side_effect=lambda value: value),
        patch("holosoma.train_agent.apply_perception_overrides", side_effect=lambda value: value),
        patch("holosoma.train_agent.training_provenance_from_env", return_value=None),
        patch("holosoma.train_agent.subprocess.run", return_value=completed) as run,
    ):
        resolved = _preflight_policy_init_before_sim(config)

    assert resolved.training.policy_init_checkpoint == str(checkpoint.resolve())
    command = run.call_args.args[0]
    assert command[1:4] == ["-m", "holosoma.utils.policy_init_preflight", "--checkpoint"]
    assert command[4] == str(checkpoint.resolve())
    submitted_config = json.loads(run.call_args.kwargs["input"])
    assert submitted_config["training"]["num_envs"] == config.training.num_envs // 2
    assert run.call_args.kwargs["text"] is True
    assert run.call_args.kwargs["capture_output"] is True


def test_train_agent_threads_required_terminal_target_to_worker_preflight(
    tmp_path,
    monkeypatch,
):
    from holosoma.config_values.experiment import DEFAULTS
    from holosoma.train_agent import _preflight_policy_init_before_sim

    checkpoint = tmp_path / "policy.pt"
    torch.save({"placeholder": True}, checkpoint)
    config = copy.deepcopy(DEFAULTS["g1_29dof_wbt_w_object_distill_sparse_root_cmd"])
    config = dataclasses.replace(
        config,
        training=dataclasses.replace(config.training, policy_init_checkpoint=str(checkpoint)),
    )
    monkeypatch.setenv(POLICY_INIT_REQUIRED_TERMINAL_TARGET_ENV, "8")
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="verified\n", stderr="")

    with (
        patch("holosoma.train_agent.apply_observation_overrides", side_effect=lambda value: value),
        patch("holosoma.train_agent.apply_perception_overrides", side_effect=lambda value: value),
        patch("holosoma.train_agent.training_provenance_from_env", return_value=None),
        patch("holosoma.train_agent.subprocess.run", return_value=completed) as run,
    ):
        _preflight_policy_init_before_sim(config)

    command = run.call_args.args[0]
    assert command[-2:] == ["--require-terminal-target", "8"]


def test_train_agent_required_terminal_target_requires_policy_initializer(monkeypatch):
    from holosoma.config_values.experiment import DEFAULTS
    from holosoma.train_agent import _preflight_policy_init_before_sim

    config = copy.deepcopy(DEFAULTS["g1_29dof_wbt_w_object_distill_sparse_root_cmd"])
    config = dataclasses.replace(
        config,
        training=dataclasses.replace(config.training, policy_init_checkpoint=None),
    )
    monkeypatch.setenv(POLICY_INIT_REQUIRED_TERMINAL_TARGET_ENV, "8")

    with pytest.raises(ValueError, match="policy-init terminal target.*checkpoint is empty"):
        _preflight_policy_init_before_sim(config)


def test_train_agent_policy_init_preflight_rejects_symlink_before_subprocess(
    tmp_path,
):
    from holosoma.config_values.experiment import DEFAULTS
    from holosoma.train_agent import _preflight_policy_init_before_sim

    checkpoint = tmp_path / "policy.pt"
    alias = tmp_path / "alias.pt"
    torch.save({"placeholder": True}, checkpoint)
    alias.symlink_to(checkpoint)
    config = copy.deepcopy(DEFAULTS["g1_29dof_wbt_w_object_distill_sparse_root_cmd"])
    config = dataclasses.replace(
        config,
        training=dataclasses.replace(config.training, policy_init_checkpoint=str(alias)),
    )

    with (
        patch("holosoma.train_agent.subprocess.run") as run,
        pytest.raises(ValueError, match="non-symlink regular file"),
    ):
        _preflight_policy_init_before_sim(config)

    run.assert_not_called()


def test_direct_train_required_terminal_source_fails_before_simulator_import(
    monkeypatch,
):
    from holosoma.config_values.experiment import DEFAULTS
    import holosoma.train_agent as train_agent

    config = copy.deepcopy(DEFAULTS["g1_29dof_wbt_w_object_distill_sparse_root_cmd"])
    config = dataclasses.replace(
        config,
        training=dataclasses.replace(config.training, policy_init_checkpoint=None),
    )
    monkeypatch.setenv(POLICY_INIT_REQUIRED_TERMINAL_TARGET_ENV, "8")

    with (
        patch("holosoma.train_agent._effective_runtime_config", side_effect=lambda value: value),
        patch("holosoma.train_agent._current_rank_training_seed"),
        patch("holosoma.train_agent._configure_defm_materialization_mode"),
        patch("holosoma.train_agent.finalize_runtime_asset_provenance", return_value=None),
        patch("holosoma.train_agent._validate_prestarted_runtime_provenance"),
        patch("holosoma.train_agent._preflight_checkpoint_lineage_before_sim"),
        patch("holosoma.train_agent._preflight_data_assets_before_sim"),
        patch("holosoma.train_agent.init_sim_imports") as init_sim,
        pytest.raises(
            ValueError,
            match="policy-init terminal target.*checkpoint is empty",
        ),
    ):
        train_agent.train(config)

    init_sim.assert_not_called()


def test_training_context_invalid_policy_initializer_fails_before_simulator_import(
    tmp_path,
):
    from holosoma.config_values.experiment import DEFAULTS
    import holosoma.train_agent as train_agent

    missing_checkpoint = tmp_path / "missing-policy.pt"
    config = copy.deepcopy(DEFAULTS["g1_29dof_wbt_w_object_distill_sparse_root_cmd"])
    config = dataclasses.replace(
        config,
        training=dataclasses.replace(
            config.training,
            policy_init_checkpoint=str(missing_checkpoint),
        ),
    )

    with (
        patch("holosoma.train_agent._effective_runtime_config", side_effect=lambda value: value),
        patch("holosoma.train_agent._current_rank_training_seed"),
        patch("holosoma.train_agent.finalize_runtime_asset_provenance", return_value=None),
        patch("holosoma.train_agent._validate_prestarted_runtime_provenance"),
        patch("holosoma.train_agent._preflight_checkpoint_lineage_before_sim"),
        patch("holosoma.train_agent._preflight_data_assets_before_sim"),
        patch("holosoma.train_agent.init_sim_imports") as init_sim,
        pytest.raises(FileNotFoundError, match="not a readable local file"),
    ):
        train_agent.TrainingContext(config).__enter__()

    init_sim.assert_not_called()


def test_repeated_policy_init_preflight_does_not_redownload_wandb_checkpoint(
    tmp_path,
):
    from holosoma.config_values.experiment import DEFAULTS
    from holosoma.train_agent import _preflight_policy_init_before_sim

    checkpoint = tmp_path / "downloaded-policy.pt"
    torch.save({"placeholder": True}, checkpoint)
    config = copy.deepcopy(DEFAULTS["g1_29dof_wbt_w_object_distill_sparse_root_cmd"])
    config = dataclasses.replace(
        config,
        training=dataclasses.replace(
            config.training,
            policy_init_checkpoint="wandb://entity/project/artifact:latest",
        ),
    )
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="verified\n",
        stderr="",
    )

    with (
        patch("holosoma.train_agent.load_checkpoint", return_value=checkpoint) as download,
        patch("holosoma.train_agent.apply_observation_overrides", side_effect=lambda value: value),
        patch("holosoma.train_agent.apply_perception_overrides", side_effect=lambda value: value),
        patch("holosoma.train_agent.training_provenance_from_env", return_value=None),
        patch("holosoma.train_agent.subprocess.run", return_value=completed),
    ):
        first = _preflight_policy_init_before_sim(config)
        second = _preflight_policy_init_before_sim(first)

    download.assert_called_once()
    assert first.training.policy_init_checkpoint == str(checkpoint.resolve())
    assert second.training.policy_init_checkpoint == str(checkpoint.resolve())
