from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from holosoma.agents.ppo.ppo import PPO
from holosoma.utils.policy_init_preflight import (
    canonical_critic_contract,
    validate_stage4_init_checkpoint,
    validate_stage4_init_payload_identity,
)
from holosoma.utils.training_provenance import (
    disabled_checkpoint_sha256,
    embedded_runtime_asset_manifest_sha256,
)


def _term(func: str) -> dict:
    return {"func": func, "params": {}, "scale": 1.0, "noise": 0.0, "clip": None}


def _group(name: str, func: str) -> dict:
    return {
        "terms": {name: _term(func)},
        "concatenate": True,
        "enable_noise": False,
        "history_length": 1,
    }


def _config() -> dict:
    layer = {
        "hidden_dims": [8, 4],
        "activation": "ELU",
        "module_input_name": ["actor_obs"],
        "perception_input_name": "",
    }
    return {
        "training": {
            "num_envs": 32,
            "policy_init_actor_contract_migration": None,
            "stage4_init_contract_migration": None,
        },
        "algo": {
            "config": {
                "module_dict": {
                    "actor": {
                        "type": "MLP",
                        "input_dim": ["actor_obs"],
                        "output_dim": ["robot_action_dim"],
                        "layer_config": copy.deepcopy(layer),
                        "min_noise_std": 0.01,
                    },
                    "critic": {
                        "type": "MLP",
                        "input_dim": ["critic_obs"],
                        "output_dim": [1],
                        "layer_config": {
                            **copy.deepcopy(layer),
                            "module_input_name": ["critic_obs"],
                        },
                    },
                },
                "normalize_actor_obs": False,
                "normalize_critic_obs": False,
                "obs_normalizer_eps": 0.01,
                "obs_normalizer_until": None,
            }
        },
        "observation": {
            "groups": {
                "actor_obs": _group("actor", "pkg:actor"),
                "critic_obs": _group("critic", "pkg:critic"),
            },
            "clip_observations": 100.0,
        },
        "robot": {
            "actions_dim": 2,
            "dof_names": ["left", "right"],
            "dof_effort_limit_list": [20.0, 20.0],
            "init_state": {"default_joint_angles": {"left": 0.0, "right": 0.0}},
            "control": {"control_type": "P", "action_scale": 0.25},
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


def _payload(config: dict) -> dict:
    return {
        "experiment_config": copy.deepcopy(config),
        "actor_model_state_dict": {"weight": torch.ones(2, 2)},
        "critic_model_state_dict": {"weight": torch.ones(1, 2)},
    }


def _stage4_provenance(checkpoint_sha256: str) -> dict:
    manifest = {"version": 2, "fixture": "stage4-init"}
    return {
        "version": 2,
        "teacher_sha256": "a" * 64,
        "policy_init_enabled": False,
        "policy_init_sha256": disabled_checkpoint_sha256("policy_init"),
        "stage4_init_enabled": True,
        "stage4_init_sha256": checkpoint_sha256,
        "training_resume_enabled": False,
        "training_resume_sha256": disabled_checkpoint_sha256("training_resume"),
        "motion_shard_manifest_sha256": "b" * 64,
        "contact_sidecar_manifest_sha256": "c" * 64,
        "source_bundle_sha256": "d" * 64,
        "runtime_asset_manifest_phase": "final",
        "runtime_asset_manifest_sha256": embedded_runtime_asset_manifest_sha256(manifest),
        "runtime_asset_manifest": manifest,
    }


def test_stage4_preflight_validates_actor_and_critic(tmp_path):
    config = _config()
    payload = _payload(config)
    actor_contract, critic_contract = validate_stage4_init_payload_identity(
        payload,
        copy.deepcopy(config),
    )
    assert actor_contract["actor_input_groups"] == ["actor_obs"]
    assert critic_contract == canonical_critic_contract(config)

    checkpoint = tmp_path / "stage4.pt"
    torch.save(payload, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    validate_stage4_init_checkpoint(
        checkpoint,
        copy.deepcopy(config),
        current_provenance=_stage4_provenance(digest),
    )


def test_stage4_preflight_rejects_equal_shape_critic_semantic_drift():
    saved = _config()
    current = copy.deepcopy(saved)
    current["observation"]["groups"]["critic_obs"]["terms"]["critic"]["func"] = "pkg:other"
    with pytest.raises(ValueError, match="critic semantic contract mismatch"):
        validate_stage4_init_payload_identity(_payload(saved), current)


class _Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1))
        self.std = nn.Parameter(torch.tensor([0.2]))
        self.min_noise_std = 0.01
        self.min_mean_noise_std = None
        self.max_noise_std = 1.0


def test_stage4_load_restores_both_models_but_resets_std_and_continuation_state():
    ppo = object.__new__(PPO)
    ppo.actor = _Actor()
    ppo.critic = nn.Linear(1, 1, bias=False)
    ppo.actor_optimizer = torch.optim.SGD(ppo.actor.parameters(), lr=0.1)
    ppo.critic_optimizer = torch.optim.SGD(ppo.critic.parameters(), lr=0.1)
    ppo.actor_obs_normalizers = {"actor_obs": nn.Identity()}
    ppo.critic_obs_normalizers = {"critic_obs": nn.Identity()}
    ppo.config = SimpleNamespace(
        normalize_actor_obs=False,
        normalize_critic_obs=False,
        init_noise_std=0.5,
    )
    ppo.current_learning_iteration = 0
    ppo.gpu_global_rank = 0
    ppo.gpu_world_size = 1
    ppo.is_multi_gpu = False
    ppo._policy_load_runtime_config = SimpleNamespace(
        to_serializable_dict=lambda: {"runtime": "stage4"}
    )
    ppo._training_provenance = _stage4_provenance("e" * 64)
    live_contract = {"version": 1, "fixture": "target"}
    ppo._collect_distributed_motion_transition_contract = Mock(
        return_value=(live_contract, "f" * 64)
    )
    ppo._validate_checkpoint_motion_transition_contract = Mock(return_value=(None, None))
    ppo._reset_dagger_replay_state = Mock()
    ppo._assert_model_parameters_finite = Mock()
    ppo._terminal_fixed_bc_eval_state = {"stale": True}

    actor_state = {key: value.detach().clone() for key, value in ppo.actor.state_dict().items()}
    actor_state["weight"].fill_(7.0)
    actor_state["std"].fill_(0.01)
    critic_state = {key: value.detach().clone() for key, value in ppo.critic.state_dict().items()}
    critic_state["weight"].fill_(9.0)
    checkpoint = {
        "experiment_config": {
            "algo": {
                "config": {
                    "normalize_actor_obs": False,
                    "normalize_critic_obs": False,
                }
            }
        },
        "actor_model_state_dict": actor_state,
        "critic_model_state_dict": critic_state,
        "iter": 39999,
        "actor_optimizer_state_dict": {"source": "ignored"},
        "critic_optimizer_state_dict": {"source": "ignored"},
        "env_state": {"source": "ignored"},
    }

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, "e" * 64),
        ),
        patch("holosoma.agents.ppo.ppo.validate_stage4_init_payload_identity"),
    ):
        ppo.load_stage4_init("stage4.pt")

    assert ppo.actor.weight.item() == pytest.approx(7.0)
    assert ppo.critic.weight.item() == pytest.approx(9.0)
    assert ppo.actor.std.item() == pytest.approx(0.5)
    assert ppo.current_learning_iteration == 0
    assert not ppo.actor_optimizer.state
    assert not ppo.critic_optimizer.state
    assert ppo._terminal_fixed_bc_eval_state is None
    assert ppo._source_checkpoint_sha256 == "e" * 64
    assert ppo._motion_transition_contract == live_contract
    ppo._reset_dagger_replay_state.assert_called_once_with()
