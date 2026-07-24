from __future__ import annotations

import copy
import hashlib
import os
from unittest.mock import patch

import pytest
import torch

from holosoma.train_agent import (
    _canonicalize_fresh_curriculum_resume_env,
    _per_rank_env_count,
)
from holosoma.utils.resume_preflight import (
    ALLOW_LEGACY_UNPROVENANCED_RESUME_ENV,
    ALLOW_RUNTIME_DRIFT_ENV,
    _RUNTIME_SEMANTIC_ENVIRONMENT_FIELDS,
    canonical_resume_manifest,
    validate_resume_payload_identity,
    validate_resume_checkpoint,
)
from holosoma.utils.rng_checkpoint import (
    ALLOW_NONDETERMINISTIC_RNG_RESUME_ENV,
    capture_rng_checkpoint_state,
)
from holosoma.utils.training_provenance import (
    disabled_checkpoint_sha256,
    embedded_runtime_asset_manifest_sha256,
)


_RUNTIME_ASSET_MANIFEST = {"version": 2, "fixture": "resume-preflight"}
_RUNTIME_ASSET_DIGEST = embedded_runtime_asset_manifest_sha256(_RUNTIME_ASSET_MANIFEST)
_HOSTILE_MARKER_ENV = "HOLOSOMA_TEST_UNSAFE_CHECKPOINT_EXECUTED"


def _execute_hostile_checkpoint_global() -> None:
    os.environ[_HOSTILE_MARKER_ENV] = "1"


class _HostileCheckpointValue:
    def __reduce__(self):
        return (_execute_hostile_checkpoint_global, ())


def _config() -> dict:
    return {
        "training": {
            "num_envs": 16,
            "checkpoint": None,
            "project": "project-a",
            "name": "run-a",
            "export_onnx": True,
        },
        "logger": {"type": "wandb", "name": "run-a"},
        "algo": {
            "config": {
                "num_learning_iterations": 40_000,
                "save_interval": 500,
                "reset_rollout_at_checkpoint": True,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "normalize_actor_obs": False,
                "normalize_critic_obs": False,
                "module_dict": {
                    "actor": {
                        "type": "MLPPerceptionEncoder",
                        "input_dim": ["root_a", "proprio"],
                        "layer_config": {"hidden_dims": [64, 32], "perception_input_name": "perception_obs"},
                    },
                    "critic": {"type": "MLP", "input_dim": ["critic_obs"]},
                },
                "distill": {
                    "policy_to_clone": "/cache/model_67000.pt",
                    "ppo_start_epoch": 0,
                    "dagger_end_epoch": 4000,
                    "ppo_start_coeff": 0.1,
                    "ppo_target_coeff": 0.9,
                    "dagger_match_std": True,
                    "schedule_name": "display-only",
                    "schedule_notes": "display-only",
                },
            }
        },
        "observation": {
            "groups": {
                "root_a": {"history_length": 1, "terms": {"root": {"func": "root_a"}}},
                "proprio": {"history_length": 5, "terms": {"joint": {"func": "joint"}}},
            }
        },
        "perception": {"enabled": True, "camera_width": 87, "camera_height": 58},
        "action": {"terms": {"joint_pos": {"scale": 1.0}}},
        "command": {"params": {"motion_file": "/data/bank-a"}},
        "robot": {"actions_dim": 29},
    }


def _save_checkpoint(tmp_path, config: dict, *, states=True):
    reset_at_checkpoint = config["algo"]["config"].get(
        "reset_rollout_at_checkpoint",
        False,
    )
    payload = {
        "experiment_config": config,
        "iter": 10,
        "next_iter": 11,
        "rng_state_by_rank": {
            "0": capture_rng_checkpoint_state(),
            "1": capture_rng_checkpoint_state(),
        },
        "rollout_resume_contract": {
            "version": 2 if reset_at_checkpoint else 3,
            "mode": (
                "canonical_reset_after_checkpoint"
                if reset_at_checkpoint
                else "new_episode_on_resume"
            ),
            "next_iteration": 11,
            "save_interval": config["algo"]["config"]["save_interval"],
            "init_at_random_ep_len": bool(
                config["algo"]["config"].get("init_at_random_ep_len", False)
            ),
            "dagger_ignore_episode_initial_steps": int(
                config["algo"]["config"].get("distill", {}).get(
                    "dagger_ignore_episode_initial_steps", 0
                )
            ),
            "reset_recurrent_hidden": True,
            "perception_state_mode": "checkpoint_stream_state_rebuild_derived_cache",
        },
    }
    if states:
        payload["env_state_by_rank"] = {"0": {"command": {"version": 1}}, "1": {"command": {"version": 1}}}
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    return path


def test_recovery_resume_contract_allows_checkpoint_without_live_rollout_reset(
    tmp_path,
) -> None:
    config = _config()
    config["algo"]["config"]["reset_rollout_at_checkpoint"] = False
    checkpoint = _save_checkpoint(tmp_path, config)

    validate_resume_checkpoint(
        checkpoint,
        config,
        world_size=2,
        allow_fresh_curriculum=False,
    )


def _provenance(fill: str = "a") -> dict:
    return {
        "version": 2,
        "teacher_sha256": fill * 64,
        "policy_init_enabled": True,
        "policy_init_sha256": "f" * 64,
        "training_resume_enabled": True,
        "training_resume_sha256": "1" * 64,
        "motion_shard_manifest_sha256": "b" * 64,
        "contact_sidecar_manifest_sha256": "c" * 64,
        "source_bundle_sha256": "e" * 64,
        "runtime_asset_manifest_phase": "final",
        "runtime_asset_manifest_sha256": _RUNTIME_ASSET_DIGEST,
        "runtime_asset_manifest": copy.deepcopy(_RUNTIME_ASSET_MANIFEST),
        "teacher_motion_end_mode": "episodic",
        "teacher_uses_action_history": True,
        "student_motion_end_mode": "episodic",
        "contact_interval_runtime_prepend_compensation": True,
        "environment": {
            "python": "3.11.15",
            "platform": "Linux-6.17.0-audit-only",
            "torch": "2.7.0+cu128",
            "torch_cuda": "12.8",
            "python_runtime_manifest_sha256": "9" * 64,
            "execution_runtime": {
                "NCCL_LIB_SHA256": "8" * 64,
                "TORCH_DIST_BACKEND": "gloo",
                "PYTHONHASHSEED": "0",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": True,
                "HOLOSOMA_GLOO_BARRIER": True,
                "HOLOSOMA_GLOO_GRAD_REDUCE": True,
                "HOLOSOMA_GLOO_SMALL_COLLECTIVES": False,
                "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE": False,
                "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES": False,
                "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER": False,
                "HOLOSOMA_RANK_VISIBLE_DEVICES": True,
                "NPROC": 8,
                "NNODES": 2,
                "HOLOSOMA_CONTIGUOUS_MINIBATCHES": True,
                "HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP": False,
                "HOLOSOMA_DAGGER_SUPERVISED_ONLY": False,
                "HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH": 16,
                "HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD": True,
                "HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC": False,
                "semantic_environment": {
                    name: None for name in _RUNTIME_SEMANTIC_ENVIRONMENT_FIELDS
                },
            },
            "packages": {
                "torch": "2.7.0",
                "isaacsim": "4.5.0.0",
                "isaaclab": "2.1.1",
                "numpy": "1.26.0",
                "omegaconf": "2.3.0",
                "antlr4-python3-runtime": "4.9.3",
                "PyYAML": "6.0.2",
                "attrs": "25.1.0",
                "audit-only-package": "1.0",
            },
        },
    }


def _current_resume_provenance(path, fill: str = "a") -> dict:
    provenance = _provenance(fill)
    provenance["policy_init_sha256"] = "0" * 64
    provenance["training_resume_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return provenance


@pytest.fixture(autouse=True)
def _deny_runtime_drift_by_default(monkeypatch):
    # Most fixtures intentionally predate provenance and exercise orthogonal
    # resume state.  Their non-scientific status is explicit; dedicated tests
    # below delete this hatch to verify the production fail-closed default.
    monkeypatch.setenv(ALLOW_LEGACY_UNPROVENANCED_RESUME_ENV, "1")
    monkeypatch.delenv(ALLOW_RUNTIME_DRIFT_ENV, raising=False)
    monkeypatch.delenv(ALLOW_NONDETERMINISTIC_RNG_RESUME_ENV, raising=False)
    monkeypatch.delenv("ALLOW_FRESH_CURRICULUM_RESUME", raising=False)
    monkeypatch.delenv("HOLOSOMA_ALLOW_FRESH_CURRICULUM_RESUME", raising=False)


def test_per_rank_env_count_requires_exact_division():
    assert _per_rank_env_count(40, 4) == 10
    with pytest.raises(ValueError, match="must be divisible"):
        _per_rank_env_count(42, 4)
    with pytest.raises(ValueError, match="too small"):
        _per_rank_env_count(2, 4)


def test_resume_payload_identity_requires_provenance_by_default(monkeypatch):
    monkeypatch.delenv(ALLOW_LEGACY_UNPROVENANCED_RESUME_ENV, raising=False)
    config = _config()

    with pytest.raises(ValueError, match="neither current nor checkpoint training provenance"):
        validate_resume_payload_identity(
            {"experiment_config": copy.deepcopy(config)},
            config,
            current_provenance=None,
            actual_resume_sha256="0" * 64,
        )


def test_resume_payload_identity_legacy_hatch_must_be_exact(monkeypatch):
    monkeypatch.setenv(ALLOW_LEGACY_UNPROVENANCED_RESUME_ENV, "true")
    config = _config()

    with pytest.raises(ValueError, match="must be exactly 0 or 1"):
        validate_resume_payload_identity(
            {"experiment_config": copy.deepcopy(config)},
            config,
            current_provenance=None,
            actual_resume_sha256="0" * 64,
        )


def test_resume_payload_identity_always_requires_serialized_config():
    with pytest.raises(ValueError, match="no serialized experiment_config"):
        validate_resume_payload_identity(
            {},
            _config(),
            current_provenance=None,
            actual_resume_sha256="0" * 64,
        )


def test_resume_preflight_rejects_disabled_current_lineage_before_load(
    tmp_path,
):
    checkpoint = _save_checkpoint(tmp_path, _config())
    provenance = _current_resume_provenance(checkpoint)
    provenance["training_resume_enabled"] = False
    provenance["training_resume_sha256"] = disabled_checkpoint_sha256(
        "training_resume"
    )

    with (
        patch(
            "holosoma.utils.resume_preflight.load_verified_torch_checkpoint"
        ) as load_mock,
        pytest.raises(ValueError, match="does not enable a full training resume"),
    ):
        validate_resume_checkpoint(
            checkpoint,
            _config(),
            world_size=2,
            allow_fresh_curriculum=False,
            current_provenance=provenance,
        )

    load_mock.assert_not_called()


def test_resume_preflight_parses_hatch_before_provenanced_load(
    tmp_path,
    monkeypatch,
):
    checkpoint = _save_checkpoint(tmp_path, _config())
    provenance = _current_resume_provenance(checkpoint)
    monkeypatch.setenv(ALLOW_LEGACY_UNPROVENANCED_RESUME_ENV, "true")

    with (
        patch(
            "holosoma.utils.resume_preflight.load_verified_torch_checkpoint"
        ) as load_mock,
        pytest.raises(ValueError, match="must be exactly 0 or 1"),
    ):
        validate_resume_checkpoint(
            checkpoint,
            _config(),
            world_size=2,
            allow_fresh_curriculum=False,
            current_provenance=provenance,
        )

    load_mock.assert_not_called()


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("ALLOW_FRESH_CURRICULUM_RESUME", "1", True),
        ("ALLOW_FRESH_CURRICULUM_RESUME", "0", False),
        ("HOLOSOMA_ALLOW_FRESH_CURRICULUM_RESUME", "true", True),
        ("HOLOSOMA_ALLOW_FRESH_CURRICULUM_RESUME", "false", False),
    ],
)
def test_fresh_curriculum_resume_alias_reaches_ppo_runtime(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv(name, value)

    assert _canonicalize_fresh_curriculum_resume_env() is expected
    assert os.environ["HOLOSOMA_ALLOW_FRESH_CURRICULUM_RESUME"] == ("1" if expected else "0")


def test_fresh_curriculum_resume_alias_rejects_split_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALLOW_FRESH_CURRICULUM_RESUME", "1")
    monkeypatch.setenv("HOLOSOMA_ALLOW_FRESH_CURRICULUM_RESUME", "0")

    with pytest.raises(ValueError, match="disagree"):
        _canonicalize_fresh_curriculum_resume_env()


def test_resume_preflight_rejects_unsupported_pickle_global_without_execution(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(_HOSTILE_MARKER_ENV, raising=False)
    checkpoint = tmp_path / "hostile.pt"
    torch.save({"hostile": _HostileCheckpointValue()}, checkpoint)

    with pytest.raises(Exception, match="Weights only load failed|Unsupported global"):
        validate_resume_checkpoint(
            checkpoint,
            _config(),
            world_size=2,
            allow_fresh_curriculum=False,
        )

    assert _HOSTILE_MARKER_ENV not in os.environ


def test_resume_manifest_allows_target_log_debug_but_not_save_interval_changes():
    saved = _config()
    current = copy.deepcopy(saved)
    current["algo"]["config"]["num_learning_iterations"] = 50_000
    current["training"]["checkpoint"] = "/cache/model.pt"
    current["training"]["name"] = "new-display-name"
    current["training"]["export_onnx"] = False
    current["logger"] = {"type": "disabled"}
    assert canonical_resume_manifest(saved) == canonical_resume_manifest(current)

    current["algo"]["config"]["save_interval"] = 50
    assert canonical_resume_manifest(saved) != canonical_resume_manifest(current)
    current["algo"]["config"]["save_interval"] = saved["algo"]["config"]["save_interval"]

    current["algo"]["config"]["distill"]["policy_to_clone"] = "wandb://e/p/r/model_67000.pt"
    assert canonical_resume_manifest(saved) != canonical_resume_manifest(current)
    assert canonical_resume_manifest(
        saved, teacher_identity_verified=True
    ) == canonical_resume_manifest(current, teacher_identity_verified=True)


def _config_with_ppo_learning_rate_bounds() -> dict:
    config = _config()
    config["algo"]["config"].update(
        actor_learning_rate=1.0e-6,
        min_actor_learning_rate=None,
        max_actor_learning_rate=None,
        critic_learning_rate=1.0e-1,
        min_critic_learning_rate=None,
        max_critic_learning_rate=None,
    )
    return config


def test_resume_manifest_matches_none_learning_rate_bounds_to_runtime_defaults():
    saved = _config_with_ppo_learning_rate_bounds()
    current = copy.deepcopy(saved)
    current["algo"]["config"].update(
        min_actor_learning_rate=1.0e-6,
        max_actor_learning_rate=1.0e-2,
        min_critic_learning_rate=1.0e-5,
        max_critic_learning_rate=1.0e-1,
    )

    saved_manifest = canonical_resume_manifest(saved)
    current_manifest = canonical_resume_manifest(current)

    assert saved_manifest == current_manifest
    saved_algo = saved_manifest["algo"]["config"]
    assert saved_algo["min_actor_learning_rate"] == pytest.approx(1.0e-6)
    assert saved_algo["max_actor_learning_rate"] == pytest.approx(1.0e-2)
    assert saved_algo["min_critic_learning_rate"] == pytest.approx(1.0e-5)
    assert saved_algo["max_critic_learning_rate"] == pytest.approx(1.0e-1)


@pytest.mark.parametrize(
    ("field", "different_value"),
    [
        ("min_actor_learning_rate", 5.0e-7),
        ("max_actor_learning_rate", 2.0e-2),
        ("min_critic_learning_rate", 5.0e-6),
        ("max_critic_learning_rate", 2.0e-1),
    ],
)
def test_resume_manifest_rejects_explicit_learning_rate_bound_drift(
    field: str,
    different_value: float,
):
    saved = _config_with_ppo_learning_rate_bounds()
    current = copy.deepcopy(saved)
    current["algo"]["config"].update(
        min_actor_learning_rate=1.0e-6,
        max_actor_learning_rate=1.0e-2,
        min_critic_learning_rate=1.0e-5,
        max_critic_learning_rate=1.0e-1,
    )
    current["algo"]["config"][field] = different_value

    with pytest.raises(ValueError, match=field):
        validate_resume_payload_identity(
            {"experiment_config": saved},
            current,
            current_provenance=None,
            actual_resume_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value", "expected"),
    [
        ("actor_learning_rate", 0.0, "actor_learning_rate"),
        ("actor_learning_rate", float("nan"), "actor_learning_rate"),
        ("min_actor_learning_rate", True, "min_actor_learning_rate"),
        ("max_actor_learning_rate", float("inf"), "max_actor_learning_rate"),
        ("min_critic_learning_rate", 2.0e-1, "critic learning-rate bounds"),
        ("max_critic_learning_rate", 5.0e-2, "critic learning-rate bounds"),
    ],
)
def test_resume_manifest_rejects_invalid_learning_rate_bounds(
    field: str,
    invalid_value,
    expected: str,
):
    config = _config_with_ppo_learning_rate_bounds()
    config["algo"]["config"][field] = invalid_value

    with pytest.raises(ValueError, match=expected):
        canonical_resume_manifest(config)


def test_resume_manifest_does_not_invent_missing_learning_rate_schema_fields():
    config = _config_with_ppo_learning_rate_bounds()
    del config["algo"]["config"]["actor_learning_rate"]

    with pytest.raises(ValueError, match=r"actor_learning_rate.*missing"):
        canonical_resume_manifest(config)

    saved = _config_with_ppo_learning_rate_bounds()
    current = copy.deepcopy(saved)
    del current["algo"]["config"]["min_actor_learning_rate"]
    assert canonical_resume_manifest(saved) != canonical_resume_manifest(current)


@pytest.mark.parametrize(
    ("mutate", "expected_path"),
    [
        (lambda cfg: cfg["algo"]["config"]["module_dict"]["actor"].update(input_dim=["root_b", "proprio"]), "input_dim"),
        (lambda cfg: cfg["observation"]["groups"]["proprio"].update(history_length=6), "history_length"),
        (lambda cfg: cfg["algo"]["config"].update(normalize_actor_obs=True), "normalize_actor_obs"),
        (lambda cfg: cfg["algo"]["config"]["distill"].update(ppo_start_coeff=0.9), "ppo_start_coeff"),
        (lambda cfg: cfg["algo"]["config"].update(num_learning_epochs=1), "num_learning_epochs"),
        (lambda cfg: cfg["perception"].update(camera_width=88), "camera_width"),
    ],
)
def test_resume_manifest_rejects_training_semantic_mismatch(tmp_path, mutate, expected_path):
    saved = _config()
    current = copy.deepcopy(saved)
    mutate(current)
    checkpoint = _save_checkpoint(tmp_path, saved)
    with pytest.raises(ValueError, match=expected_path):
        validate_resume_checkpoint(
            checkpoint,
            current,
            world_size=2,
            allow_fresh_curriculum=False,
        )


def test_resume_rejects_pre_perception_stream_rollout_contract(tmp_path) -> None:
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["rollout_resume_contract"]["version"] = 1
    payload["rollout_resume_contract"].pop("perception_state_mode")
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="rollout contract mismatch"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
        )


def test_resume_requires_rank_local_curriculum_state_by_default(tmp_path, capsys):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config, states=False)
    with pytest.raises(ValueError, match="not a curriculum-correct training resume"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
        )

    validate_resume_checkpoint(
        checkpoint,
        config,
        world_size=2,
        allow_fresh_curriculum=True,
    )
    assert "AS adaptive sampler" in capsys.readouterr().out


def test_resume_requires_rank_local_rng_state_or_exact_legacy_override(
    tmp_path,
    monkeypatch,
    capsys,
):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload.pop("rng_state_by_rank")
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="no saved rank-local Python/NumPy/torch RNG state"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
        )

    monkeypatch.setenv(ALLOW_NONDETERMINISTIC_RNG_RESUME_ENV, "true")
    with pytest.raises(ValueError, match="must be exactly 0 or 1"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
        )

    monkeypatch.setenv(ALLOW_NONDETERMINISTIC_RNG_RESUME_ENV, "1")
    validate_resume_checkpoint(
        checkpoint,
        config,
        world_size=2,
        allow_fresh_curriculum=False,
    )
    output = capsys.readouterr().out
    assert "nondeterministic_rng_resume_allowed" in output
    assert "resume_mode=legacy_rng_missing_not_bitwise_trajectory" in output


def test_resume_rejects_rng_rank_or_payload_corruption(tmp_path):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["rng_state_by_rank"].pop("1")
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="RNG world-size/rank-state mismatch"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
        )

    payload["rng_state_by_rank"]["1"] = capture_rng_checkpoint_state()
    payload["rng_state_by_rank"]["0"]["torch_cpu_rng_state"] = torch.zeros(
        3,
        dtype=torch.uint8,
    )
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="not a valid torch RNG state"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
        )


def test_resume_requires_matching_sha256_provenance_not_teacher_basename(tmp_path):
    saved = _config()
    current = copy.deepcopy(saved)
    current["algo"]["config"]["distill"]["policy_to_clone"] = "wandb://e/p/r/model_67000.pt"
    checkpoint = _save_checkpoint(tmp_path, saved)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_provenance = _provenance("a")
    saved_provenance.update(
        policy_init_enabled=True,
        training_resume_enabled=False,
        training_resume_sha256=disabled_checkpoint_sha256("training_resume"),
    )
    payload["training_provenance"] = saved_provenance
    torch.save(payload, checkpoint)

    current_provenance = _current_resume_provenance(checkpoint, "a")
    current_provenance.update(
        policy_init_enabled=False,
        policy_init_sha256=disabled_checkpoint_sha256("policy_init"),
        training_resume_enabled=True,
    )
    validate_resume_checkpoint(
        checkpoint,
        current,
        world_size=2,
        allow_fresh_curriculum=False,
        current_provenance=current_provenance,
    )
    mismatched_provenance = dict(current_provenance)
    mismatched_provenance["teacher_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="input provenance mismatch"):
        validate_resume_checkpoint(
            checkpoint,
            current,
            world_size=2,
            allow_fresh_curriculum=False,
            current_provenance=mismatched_provenance,
        )


def test_resume_rejects_runtime_asset_manifest_digest_mismatch(tmp_path):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_provenance = _provenance()
    payload["training_provenance"] = saved_provenance
    torch.save(payload, checkpoint)
    current_provenance = _current_resume_provenance(checkpoint)
    current_provenance["runtime_asset_manifest"]["fixture"] = "different-assets"
    current_provenance["runtime_asset_manifest_sha256"] = embedded_runtime_asset_manifest_sha256(
        current_provenance["runtime_asset_manifest"]
    )

    with pytest.raises(
        ValueError,
        match=r"(?s)input provenance mismatch.*runtime_asset_manifest_sha256",
    ):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
            current_provenance=current_provenance,
        )


@pytest.mark.parametrize(
    ("mutate", "expected_path"),
    [
        (
            lambda provenance: provenance["environment"].update(
                python_runtime_manifest_sha256="8" * 64
            ),
            "python_runtime_manifest_sha256",
        ),
        (lambda provenance: provenance["environment"].update(torch="2.8.0+cu128"), "torch"),
        (lambda provenance: provenance["environment"].update(torch_cuda="12.9"), "torch_cuda"),
        (
            lambda provenance: provenance["environment"]["packages"].update(
                isaacsim="5.0.0.0"
            ),
            "isaacsim",
        ),
        (
            lambda provenance: provenance["environment"]["packages"].update(isaaclab="2.2.0"),
            "isaaclab",
        ),
        (
            lambda provenance: provenance["environment"]["packages"].update(numpy="2.0.0"),
            "numpy",
        ),
        (
            lambda provenance: provenance["environment"]["packages"].update(
                omegaconf="2.3.1"
            ),
            "omegaconf",
        ),
        (
            lambda provenance: provenance["environment"]["packages"].update(
                **{"antlr4-python3-runtime": "4.10.0"}
            ),
            "antlr4-python3-runtime",
        ),
        (
            lambda provenance: provenance["environment"]["packages"].update(
                PyYAML="6.1.0"
            ),
            "PyYAML",
        ),
        (
            lambda provenance: provenance["environment"]["packages"].update(
                attrs="25.2.0"
            ),
            "attrs",
        ),
    ],
)
def test_resume_rejects_core_runtime_identity_drift_by_default(
    tmp_path,
    mutate,
    expected_path,
):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_provenance = _provenance()
    payload["training_provenance"] = saved_provenance
    torch.save(payload, checkpoint)
    current_provenance = _current_resume_provenance(checkpoint)
    mutate(current_provenance)

    with pytest.raises(ValueError, match=expected_path):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
            current_provenance=current_provenance,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("PYTHONHASHSEED", "7"),
        ("CUBLAS_WORKSPACE_CONFIG", ":16:8"),
        ("NCCL_LIB_SHA256", "7" * 64),
        ("TORCH_DIST_BACKEND", "nccl"),
        ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", False),
        ("HOLOSOMA_GLOO_BARRIER", False),
        ("HOLOSOMA_GLOO_GRAD_REDUCE", False),
        ("HOLOSOMA_GLOO_SMALL_COLLECTIVES", True),
        ("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", True),
        ("HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES", True),
        ("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER", True),
        ("HOLOSOMA_RANK_VISIBLE_DEVICES", False),
        ("NPROC", 4),
        ("NNODES", 1),
        ("HOLOSOMA_CONTIGUOUS_MINIBATCHES", False),
        ("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", True),
        ("HOLOSOMA_DAGGER_SUPERVISED_ONLY", True),
        ("HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH", 8),
        ("HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD", False),
        ("HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC", True),
    ],
)
def test_resume_rejects_execution_runtime_drift_by_default(tmp_path, field, value):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_provenance = _provenance()
    if field == "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES":
        saved_execution_runtime = saved_provenance["environment"]["execution_runtime"]
        saved_execution_runtime["HOLOSOMA_GLOO_GRAD_REDUCE"] = False
        saved_execution_runtime["HOLOSOMA_GLOO_SMALL_COLLECTIVES"] = True
        saved_execution_runtime["HOLOSOMA_HIERARCHICAL_GRAD_REDUCE"] = True
    payload["training_provenance"] = saved_provenance
    torch.save(payload, checkpoint)
    current_provenance = _current_resume_provenance(checkpoint)
    current_execution_runtime = current_provenance["environment"]["execution_runtime"]
    if field == "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES":
        current_execution_runtime["HOLOSOMA_GLOO_GRAD_REDUCE"] = False
        current_execution_runtime["HOLOSOMA_GLOO_SMALL_COLLECTIVES"] = True
        current_execution_runtime["HOLOSOMA_HIERARCHICAL_GRAD_REDUCE"] = True
    current_execution_runtime[field] = value

    with pytest.raises(ValueError, match=field):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
            current_provenance=current_provenance,
        )


def test_resume_normalizes_missing_legacy_hierarchical_small_field_to_false(tmp_path):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_provenance = _provenance()
    saved_provenance["environment"]["execution_runtime"].pop(
        "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES"
    )
    payload["training_provenance"] = saved_provenance
    torch.save(payload, checkpoint)

    current_provenance = _current_resume_provenance(checkpoint)
    validate_resume_checkpoint(
        checkpoint,
        config,
        world_size=2,
        allow_fresh_curriculum=False,
        current_provenance=current_provenance,
    )


def test_resume_rejects_partial_hierarchical_small_collectives_contract(tmp_path):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_provenance = _provenance()
    saved_execution_runtime = saved_provenance["environment"]["execution_runtime"]
    saved_execution_runtime["HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES"] = True
    payload["training_provenance"] = saved_provenance
    torch.save(payload, checkpoint)
    current_provenance = _current_resume_provenance(checkpoint)
    current_provenance["environment"]["execution_runtime"][
        "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES"
    ] = True

    with pytest.raises(
        ValueError,
        match=(
            "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=1 requires "
            "HOLOSOMA_GLOO_SMALL_COLLECTIVES=1 and "
            "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1"
        ),
    ):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
            current_provenance=current_provenance,
        )


def test_resume_semantic_environment_schema_matches_provenance_generator():
    from scripts.compute_training_provenance import SEMANTIC_ENVIRONMENT_FIELDS

    assert _RUNTIME_SEMANTIC_ENVIRONMENT_FIELDS == SEMANTIC_ENVIRONMENT_FIELDS


def test_resume_rejects_unset_to_explicit_false_semantic_environment_drift(tmp_path):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["training_provenance"] = _provenance()
    torch.save(payload, checkpoint)
    current_provenance = _current_resume_provenance(checkpoint)
    current_provenance["environment"]["execution_runtime"]["semantic_environment"][
        "HOLOSOMA_ONLINE_CONTACT_PRIOR"
    ] = "0"

    with pytest.raises(
        ValueError,
        match=r"semantic_environment\.HOLOSOMA_ONLINE_CONTACT_PRIOR",
    ):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
            current_provenance=current_provenance,
        )


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda execution: execution.pop("semantic_environment"),
            r"semantic_environment: missing or not a JSON object",
        ),
        (
            lambda execution: execution["semantic_environment"].pop(
                "HOLOSOMA_DEFM_FORWARD_BATCH_SIZE"
            ),
            r"semantic_environment: keys must exactly match",
        ),
        (
            lambda execution: execution["semantic_environment"].__setitem__(
                "HOLOSOMA_UNDECLARED_TRAINING_OVERRIDE", "1"
            ),
            r"semantic_environment: keys must exactly match",
        ),
        (
            lambda execution: execution["semantic_environment"].__setitem__(17, "1"),
            r"semantic_environment: keys must exactly match",
        ),
        (
            lambda execution: execution["semantic_environment"].__setitem__(
                "HOLOSOMA_DEFM_FORWARD_BATCH_SIZE", False
            ),
            r"expected a string or null",
        ),
        (
            lambda execution: execution["semantic_environment"].__setitem__(
                "HOLOSOMA_DEFM_FORWARD_BATCH_SIZE", " 0 "
            ),
            r"expected a stripped canonical string",
        ),
    ],
)
def test_resume_rejects_invalid_semantic_environment_schema_by_default(
    tmp_path,
    mutate,
    expected,
):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["training_provenance"] = _provenance()
    torch.save(payload, checkpoint)
    current_provenance = _current_resume_provenance(checkpoint)
    mutate(current_provenance["environment"]["execution_runtime"])

    with pytest.raises(ValueError, match=expected):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
            current_provenance=current_provenance,
        )


def test_resume_rejects_missing_or_stringly_typed_execution_runtime_by_default(tmp_path):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_provenance = _provenance()
    del saved_provenance["environment"]["execution_runtime"]
    payload["training_provenance"] = saved_provenance
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="checkpoint.environment.execution_runtime: missing"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
            current_provenance=_current_resume_provenance(checkpoint),
        )

    payload["training_provenance"] = _provenance()
    torch.save(payload, checkpoint)
    current_provenance = _current_resume_provenance(checkpoint)
    current_provenance["environment"]["execution_runtime"]["HOLOSOMA_GLOO_GRAD_REDUCE"] = "1"
    with pytest.raises(ValueError, match="expected a boolean"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
            current_provenance=current_provenance,
        )


def test_resume_runtime_contract_ignores_platform_and_unrelated_packages(tmp_path):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["training_provenance"] = _provenance()
    torch.save(payload, checkpoint)
    current_provenance = _current_resume_provenance(checkpoint)
    current_provenance["environment"]["platform"] = "Linux-6.17.0-different-kernel"
    current_provenance["environment"]["hostname"] = "different-node"
    current_provenance["environment"]["packages"]["audit-only-package"] = "9.9"

    validate_resume_checkpoint(
        checkpoint,
        config,
        world_size=2,
        allow_fresh_curriculum=False,
        current_provenance=current_provenance,
    )


def test_resume_runtime_drift_requires_exact_override_and_emits_lineage(
    tmp_path,
    monkeypatch,
    capsys,
):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["training_provenance"] = _provenance()
    torch.save(payload, checkpoint)
    current_provenance = _current_resume_provenance(checkpoint)
    current_provenance["environment"]["packages"]["numpy"] = "2.0.0"

    monkeypatch.setenv(ALLOW_RUNTIME_DRIFT_ENV, "true")
    with pytest.raises(ValueError, match="must be exactly 0 or 1"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
            current_provenance=current_provenance,
        )

    monkeypatch.setenv(ALLOW_RUNTIME_DRIFT_ENV, "1")
    validate_resume_checkpoint(
        checkpoint,
        config,
        world_size=2,
        allow_fresh_curriculum=False,
        current_provenance=current_provenance,
    )
    output = capsys.readouterr().out
    assert "runtime_drift_on_resume_allowed" in output
    assert f"override={ALLOW_RUNTIME_DRIFT_ENV}=1" in output
    assert "lineage_parent_checkpoint_sha256=" in output
    assert '"numpy":"2.0.0"' in output


def test_resume_rejects_missing_runtime_manifest_by_default(tmp_path, monkeypatch, capsys):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_provenance = _provenance()
    del saved_provenance["environment"]["python_runtime_manifest_sha256"]
    payload["training_provenance"] = saved_provenance
    torch.save(payload, checkpoint)
    current_provenance = _current_resume_provenance(checkpoint)

    with pytest.raises(ValueError, match="checkpoint.environment.python_runtime_manifest_sha256: missing"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
            current_provenance=current_provenance,
        )

    monkeypatch.setenv(ALLOW_RUNTIME_DRIFT_ENV, "1")
    validate_resume_checkpoint(
        checkpoint,
        config,
        world_size=2,
        allow_fresh_curriculum=False,
        current_provenance=current_provenance,
    )
    assert "runtime_drift_on_resume_allowed" in capsys.readouterr().out


def test_resume_treats_missing_legacy_contact_compensation_as_false(tmp_path):
    saved = _config()
    saved["command"] = {
        "setup_terms": {"motion_command": {"params": {"motion_config": {}}}}
    }
    current = copy.deepcopy(saved)
    current["command"]["setup_terms"]["motion_command"]["params"]["motion_config"][
        "contact_interval_runtime_prepend_compensation"
    ] = False
    checkpoint = _save_checkpoint(tmp_path, saved)
    validate_resume_checkpoint(
        checkpoint,
        current,
        world_size=2,
        allow_fresh_curriculum=False,
    )
    current["command"]["setup_terms"]["motion_command"]["params"]["motion_config"][
        "contact_interval_runtime_prepend_compensation"
    ] = True
    with pytest.raises(ValueError, match="contact_interval_runtime_prepend_compensation"):
        validate_resume_checkpoint(
            checkpoint,
            current,
            world_size=2,
            allow_fresh_curriculum=False,
        )


def test_resume_treats_missing_legacy_button_window_mode_as_contact_interval(tmp_path):
    saved = _config()
    saved["command"] = {
        "setup_terms": {"motion_command": {"params": {"motion_config": {}}}}
    }
    current = copy.deepcopy(saved)
    motion_config = current["command"]["setup_terms"]["motion_command"]["params"][
        "motion_config"
    ]
    motion_config["contact_aware_button_window_mode"] = "contact_interval"
    checkpoint = _save_checkpoint(tmp_path, saved)

    validate_resume_checkpoint(
        checkpoint,
        current,
        world_size=2,
        allow_fresh_curriculum=False,
    )

    motion_config["contact_aware_button_window_mode"] = "kinematic_lift"
    with pytest.raises(ValueError, match="contact_aware_button_window_mode"):
        validate_resume_checkpoint(
            checkpoint,
            current,
            world_size=2,
            allow_fresh_curriculum=False,
        )


def test_resume_rejects_world_size_rank_state_mismatch(tmp_path):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    with pytest.raises(ValueError, match="world-size/rank-state mismatch"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=3,
            allow_fresh_curriculum=False,
        )


def test_resume_rejects_noncanonical_or_nonfinite_rank_state(tmp_path):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["env_state_by_rank"] = {
        0: {"command": {"version": 1}},
        "0": {"command": {"version": 1}},
        "1": {"command": {"version": 1}},
    }
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="keys must be canonical decimal strings"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
        )

    payload["env_state_by_rank"] = {
        "0": {"command": {"failed": torch.tensor(float("nan"))}},
        "1": {"command": {"version": 1}},
    }
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="env_state_by_rank.*non-finite"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
        )


def test_resume_rejects_malformed_or_nonfinite_fixed_bc_rank_state(tmp_path):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["fixed_bc_eval_by_rank"] = {
        "0": {"ready": True, "actor_obs_raw": torch.zeros(1, 2)},
    }
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="fixed-BC world-size/rank-state mismatch"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
        )

    payload["fixed_bc_eval_by_rank"]["1"] = {
        "ready": True,
        "actor_obs_raw": torch.tensor([[float("inf"), 0.0]]),
    }
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="fixed_bc_eval_by_rank.*non-finite"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
        )


def test_resume_fixed_bc_reset_override_is_exact_and_explicit(tmp_path, monkeypatch, capsys):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["fixed_bc_eval_by_rank"] = {
        "0": {"ready": True, "actor_obs_raw": torch.zeros(1, 2)},
    }
    torch.save(payload, checkpoint)

    monkeypatch.setenv("HOLOSOMA_ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME", "true")
    with pytest.raises(ValueError, match="must be exactly 0 or 1"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
        )

    monkeypatch.setenv("HOLOSOMA_ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME", "1")
    validate_resume_checkpoint(
        checkpoint,
        config,
        world_size=2,
        allow_fresh_curriculum=False,
    )
    output = capsys.readouterr().out
    assert "fixed_bc_reset_on_resume_allowed" in output
    assert "HOLOSOMA_ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME=1" in output

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("iter", True),
        ("iter", 10.5),
        ("next_iter", False),
        ("next_iter", 11.5),
    ],
)
def test_resume_rejects_non_integral_iteration_metadata(tmp_path, field, value):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload[field] = value
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match=rf"Checkpoint {field} must be an integer"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
        )


@pytest.mark.parametrize(("field", "value"), [("iter", -1), ("next_iter", -1)])
def test_resume_rejects_negative_iteration_metadata(tmp_path, field, value):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload[field] = value
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match=rf"Checkpoint {field} must be >= 0"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
        )


@pytest.mark.parametrize("next_iter", [10, 12])
def test_resume_rejects_explicit_next_iter_that_is_not_iter_plus_one(tmp_path, next_iter):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["next_iter"] = next_iter
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match=r"explicit next_iter must equal iter \+ 1"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
        )


def test_resume_rejects_disagreeing_iter_aliases(tmp_path):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["iteration"] = 9
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="iter=10, iteration=9"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
        )


def test_resume_accepts_legacy_iteration_alias_without_next_iter(tmp_path, capsys):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["iteration"] = payload.pop("iter")
    payload.pop("next_iter")
    torch.save(payload, checkpoint)

    validate_resume_checkpoint(
        checkpoint,
        config,
        world_size=2,
        allow_fresh_curriculum=False,
    )
    assert "iter=10 next_iter=11" in capsys.readouterr().out


def test_resume_target_is_absolute_and_legacy_iter_is_not_repeated(tmp_path, capsys):
    config = _config()
    config["algo"]["config"]["num_learning_iterations"] = 40_000
    checkpoint = _save_checkpoint(tmp_path, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["iter"] = 31_500
    payload.pop("next_iter")
    payload["rollout_resume_contract"]["next_iteration"] = 31_501
    torch.save(payload, checkpoint)

    validate_resume_checkpoint(
        checkpoint,
        config,
        world_size=2,
        allow_fresh_curriculum=False,
    )
    output = capsys.readouterr().out
    assert "next_iter=31501" in output
    assert "remaining_iterations=8499" in output

    config["algo"]["config"]["num_learning_iterations"] = 31_501
    with pytest.raises(ValueError, match="must be greater than checkpoint next_iter 31501"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
        )


def test_resume_preflight_rejects_optimizer_reset_warm_start(tmp_path):
    config = _config()
    checkpoint = _save_checkpoint(tmp_path, config)
    config["algo"]["config"]["load_optimizer"] = False

    with pytest.raises(ValueError, match="load_optimizer=true"):
        validate_resume_checkpoint(
            checkpoint,
            config,
            world_size=2,
            allow_fresh_curriculum=False,
        )
