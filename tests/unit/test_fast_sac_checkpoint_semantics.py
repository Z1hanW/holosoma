from __future__ import annotations

import copy
import hashlib
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent, FastSACEnv
from holosoma.agents.fast_sac.fast_sac_utils import EmpiricalNormalization
from holosoma.config_types.algo import FastSACConfig
from holosoma.managers.action.terms.joint_control import JointPositionActionTerm
from holosoma.utils.policy_init_preflight import (
    canonical_fast_sac_actor_contract,
    validate_policy_init_payload_identity,
)
from holosoma.utils.rng_checkpoint import (
    capture_rng_checkpoint_state,
    restore_rng_checkpoint_state,
)


def test_fast_sac_env_requests_dense_episode_stats_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_contracts: list[bool] = []
    env = SimpleNamespace(
        set_collection_extras_contract=lambda *, dense_episode_stats: requested_contracts.append(
            dense_episode_stats
        )
    )
    monkeypatch.setattr(
        FastSACEnv,
        "_compute_action_transform",
        lambda _self: (torch.ones(1), torch.zeros(1)),
    )

    FastSACEnv(
        env,
        actor_obs_keys=["actor_obs"],
        critic_obs_keys=["critic_obs"],
        action_boundary_mode="joint_limit_affine_v2",
    )

    assert requested_contracts == [True]


def _fast_sac_config(*, normalization: bool = True, boundary_mode: str = "joint_limit_affine_v2") -> dict:
    return {
        "algo": {
            "_target_": "holosoma.agents.fast_sac.fast_sac_agent.FastSACAgent",
            "config": {
                "actor_obs_keys": ["actor_obs"],
                "actor_hidden_dim": 16,
                "log_std_max": 0.0,
                "log_std_min": -5.0,
                "use_tanh": True,
                "action_boundary_mode": boundary_mode,
                "use_layer_norm": True,
                "obs_normalization": normalization,
                "use_cnn_encoder": False,
                "encoder_obs_key": "perception_obs",
                "encoder_obs_shape": [1, 2, 2],
            },
        },
        "observation": {
            "groups": {
                "actor_obs": {
                    "terms": {
                        "state": {
                            "func": "pkg:state",
                            "params": {},
                            "scale": 1.0,
                            "noise": 0.0,
                            "clip": None,
                        }
                    },
                    "concatenate": True,
                    "enable_noise": False,
                    "history_length": 1,
                }
            },
            "clip_observations": 100.0,
        },
        "robot": {
            "actions_dim": 2,
            "dof_names": ["j0", "j1"],
            "dof_pos_lower_limit_list": [-1.0, -0.5],
            "dof_pos_upper_limit_list": [3.0, 1.5],
            "dof_effort_limit_list": [10.0, 20.0],
            "init_state": {"default_joint_angles": {"j0": 0.0, "j1": 0.5}},
            "control": {
                "control_type": "P",
                "action_scale": 0.25,
                "action_clip_value": 100.0,
                "clip_actions": True,
                "clip_torques": True,
                "action_scales_by_effort_limit_over_p_gain": True,
                "stiffness": {"j0": 5.0, "j1": 10.0},
                "damping": {"j0": 1.0, "j1": 1.0},
            },
        },
        "action": {
            "terms": {
                "joint_control": {
                    "func": "holosoma.managers.action.terms.joint_control:JointPositionActionTerm",
                    "params": {},
                    "scale": 1.0,
                    "clip": None,
                }
            }
        },
    }


def _fast_sac_args(config: dict) -> dict:
    return copy.deepcopy(config["algo"]["config"])


class _RuntimeConfig:
    def __init__(self, payload: dict):
        self.payload = payload

    def to_serializable_dict(self) -> dict:
        return copy.deepcopy(self.payload)


class _TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2))
        self.register_buffer("action_scale", torch.tensor([4.0, 0.5]))
        self.register_buffer("action_bias", torch.tensor([2.0, 0.0]))


def _evaluation_agent(config: dict) -> FastSACAgent:
    agent = FastSACAgent.__new__(FastSACAgent)
    agent.actor = _TinyActor()
    agent.obs_normalizer = EmpiricalNormalization(shape=2, device="cpu")
    agent._policy_load_runtime_config = _RuntimeConfig(config)
    agent._evaluation_only = True
    agent._training_provenance = {"stale": "must be replaced"}
    agent._source_checkpoint_sha256 = None
    agent._source_experiment_config_dict = None
    agent._evaluation_completed_iteration = None
    agent.global_step = 777
    return agent


def _evaluation_checkpoint(agent: FastSACAgent, config: dict, *, completed_step: int = 12) -> dict:
    actor_state = copy.deepcopy(agent.actor.state_dict())
    actor_state["weight"] = torch.full_like(actor_state["weight"], 3.0)
    normalizer_state = copy.deepcopy(agent.obs_normalizer.state_dict())
    normalizer_state["_mean"] = torch.tensor([[1.0, -1.0]])
    normalizer_state["_var"] = torch.tensor([[4.0, 9.0]])
    normalizer_state["_std"] = torch.tensor([[2.0, 3.0]])
    normalizer_state["count"] = torch.tensor(20, dtype=torch.long)
    return {
        "actor_state_dict": actor_state,
        "obs_normalizer_state": normalizer_state,
        "experiment_config": copy.deepcopy(config),
        "args": _fast_sac_args(config),
        "global_step": completed_step,
        "iteration": completed_step,
        "infos": {"source": "test"},
    }


def test_fast_sac_contract_rejects_equal_shape_observation_semantic_drift():
    saved = _fast_sac_config()
    current = copy.deepcopy(saved)
    current["observation"]["groups"]["actor_obs"]["terms"]["state"]["scale"] = 2.0
    checkpoint = {
        "actor_state_dict": {"weight": torch.ones(1)},
        "obs_normalizer_state": {
            "_mean": torch.zeros(1, 1),
            "_var": torch.ones(1, 1),
            "_std": torch.ones(1, 1),
            "count": torch.tensor(0, dtype=torch.long),
        },
        "experiment_config": saved,
        "args": _fast_sac_args(saved),
    }
    with pytest.raises(ValueError, match="semantic contract mismatch"):
        validate_policy_init_payload_identity(checkpoint, current)


def test_fast_sac_missing_boundary_version_is_explicit_legacy_contract():
    legacy = _fast_sac_config(boundary_mode="legacy_max_range_scalar_v1")
    del legacy["algo"]["config"]["action_boundary_mode"]
    contract = canonical_fast_sac_actor_contract(legacy)
    assert contract["actor_module"]["action_boundary_mode"] == "legacy_max_range_scalar_v1"
    assert contract["policy_implementation"]["action_boundary"] == "legacy_max_range_scalar_v1"


def test_fast_sac_affine_action_transform_maps_tanh_endpoints_to_joint_limits():
    term = JointPositionActionTerm.__new__(JointPositionActionTerm)
    term._action_dim = 2
    term.action_scales = torch.tensor([0.5, 2.0])
    action_manager = SimpleNamespace(
        total_action_dim=2,
        iter_terms=lambda: [("joint_control", term)],
    )
    robot_config = SimpleNamespace(
        dof_names=["j0", "j1"],
        dof_pos_lower_limit_list=[-1.0, -0.5],
        dof_pos_upper_limit_list=[3.0, 1.5],
        init_state=SimpleNamespace(default_joint_angles={"j0": 0.0, "j1": 0.5}),
        control=SimpleNamespace(
            control_type="P",
            action_scale=0.25,
            clip_actions=True,
            action_clip_value=100.0,
        ),
    )
    wrapped = FastSACEnv.__new__(FastSACEnv)
    wrapped._env = SimpleNamespace(
        robot_config=robot_config,
        action_manager=action_manager,
        device="cpu",
    )
    wrapped._action_boundary_mode = "joint_limit_affine_v2"
    scale, bias = wrapped._compute_action_transform()
    assert torch.allclose(scale, torch.tensor([4.0, 0.5]))
    assert torch.allclose(bias, torch.tensor([2.0, 0.0]))

    default = torch.tensor([0.0, 0.5])
    effective_scale = term.action_scales
    assert torch.allclose(default + (bias - scale) * effective_scale, torch.tensor([-1.0, -0.5]))
    assert torch.allclose(default + (bias + scale) * effective_scale, torch.tensor([3.0, 1.5]))


def test_fast_sac_legacy_action_transform_reproduces_old_scalar_max_range():
    robot_config = SimpleNamespace(
        dof_names=["j0", "j1"],
        dof_pos_lower_limit_list=[-1.0, -0.5],
        dof_pos_upper_limit_list=[3.0, 1.5],
        init_state=SimpleNamespace(default_joint_angles={"j0": 0.0, "j1": 0.5}),
        control=SimpleNamespace(action_scale=0.25),
    )
    wrapped = FastSACEnv.__new__(FastSACEnv)
    wrapped._env = SimpleNamespace(robot_config=robot_config, device="cpu")
    wrapped._action_boundary_mode = "legacy_max_range_scalar_v1"
    scale, bias = wrapped._compute_action_transform()
    assert torch.allclose(scale, torch.tensor([12.0, 4.0]))
    assert torch.equal(bias, torch.zeros(2))


def test_fast_sac_delayed_actor_schedule_counts_critic_updates_across_collections():
    agent = FastSACAgent.__new__(FastSACAgent)
    agent.config = SimpleNamespace(policy_frequency=4)
    agent._critic_update_step = 0

    # Model two collection iterations with three critic updates each.  A
    # per-collection index would reset and never reach frequency four.
    first_collection = [
        agent._record_critic_update_and_should_update_policy() for _ in range(3)
    ]
    second_collection = [
        agent._record_critic_update_and_should_update_policy() for _ in range(3)
    ]
    assert first_collection == [False, False, False]
    assert second_collection == [True, False, False]
    assert agent._critic_update_step == 6

    agent.config = SimpleNamespace(policy_frequency=1)
    assert agent._record_critic_update_and_should_update_policy() is True


def test_fast_sac_evaluation_load_is_actor_only_and_preserves_rng(tmp_path):
    config = _fast_sac_config()
    agent = _evaluation_agent(config)
    checkpoint = _evaluation_checkpoint(agent, config)
    checkpoint_path = tmp_path / "fast_sac.pt"
    torch.save(checkpoint, checkpoint_path)
    expected_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    random.seed(91)
    np.random.seed(92)
    torch.manual_seed(93)
    boundary = capture_rng_checkpoint_state()
    expected_rng = (random.random(), float(np.random.random()), float(torch.rand(())))
    restore_rng_checkpoint_state(boundary, path="test_boundary")
    agent._evaluation_rng_boundary_state = boundary

    infos = agent.load_evaluation(str(checkpoint_path))
    actual_rng = (random.random(), float(np.random.random()), float(torch.rand(())))

    assert infos == {"source": "test"}
    assert actual_rng == expected_rng
    assert torch.equal(agent.actor.weight, torch.full((2, 2), 3.0))
    assert torch.equal(agent.obs_normalizer.state_dict()["_mean"], torch.tensor([[1.0, -1.0]]))
    assert agent.global_step == 777
    assert not hasattr(agent, "qnet")
    assert not hasattr(agent, "actor_optimizer")
    assert agent._evaluation_completed_iteration == 12
    assert agent._source_checkpoint_sha256 == expected_sha
    assert agent._source_experiment_config_dict == config
    assert agent._training_provenance is None


def test_fast_sac_evaluation_rejects_step_conflict_before_actor_mutation(tmp_path):
    config = _fast_sac_config()
    agent = _evaluation_agent(config)
    checkpoint = _evaluation_checkpoint(agent, config)
    checkpoint["global_step"] = 13
    checkpoint_path = tmp_path / "conflict.pt"
    torch.save(checkpoint, checkpoint_path)
    agent._evaluation_rng_boundary_state = capture_rng_checkpoint_state()

    before = copy.deepcopy(agent.actor.state_dict())
    with pytest.raises(ValueError, match="step metadata is inconsistent"):
        agent.load_evaluation(str(checkpoint_path))
    for key, value in before.items():
        assert torch.equal(agent.actor.state_dict()[key], value)


def test_fast_sac_full_resume_fails_closed_without_replay_and_rng_state():
    agent = FastSACAgent.__new__(FastSACAgent)
    with pytest.raises(RuntimeError, match="exact full resume is intentionally disabled"):
        agent.load("checkpoint.pt")


def test_fast_sac_evaluation_only_setup_skips_training_state():
    actor_group = SimpleNamespace(concatenate=True)
    critic_group = SimpleNamespace(concatenate=True)

    class _ObsManager:
        cfg = SimpleNamespace(groups={"actor_obs": actor_group, "critic_obs": critic_group})

        def __init__(self):
            self.requested = None

        def get_obs_dims(self, group_names):
            self.requested = list(group_names)
            return {"actor_obs": 2}

    observation_manager = _ObsManager()
    wrapped_env = SimpleNamespace(
        observation_manager=observation_manager,
        robot_config=SimpleNamespace(actions_dim=2),
        num_envs=4,
        _action_boundaries=torch.ones(2),
        _action_bias=torch.zeros(2),
        _include_critic_obs=True,
    )
    agent = FastSACAgent.__new__(FastSACAgent)
    agent.config = FastSACConfig(
        actor_hidden_dim=16,
        actor_obs_keys=["actor_obs"],
        critic_obs_keys=["critic_obs"],
        use_layer_norm=False,
        compile=False,
        amp=False,
    )
    agent.device = "cpu"
    agent.env = wrapped_env
    agent.is_multi_gpu = False
    agent._evaluation_only = True

    agent.setup()

    assert observation_manager.requested == ["actor_obs"]
    assert wrapped_env._include_critic_obs is False
    assert hasattr(agent, "actor")
    assert hasattr(agent, "obs_normalizer")
    for attribute in (
        "qnet",
        "qnet_target",
        "actor_optimizer",
        "q_optimizer",
        "alpha_optimizer",
        "scaler",
        "rb",
        "critic_obs_normalizer",
    ):
        assert not hasattr(agent, attribute)
