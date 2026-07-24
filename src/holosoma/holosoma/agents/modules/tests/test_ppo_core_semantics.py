from __future__ import annotations

import dataclasses
import hashlib
import itertools
import math
import random
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import pytest
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from holosoma.agents.modules.data_utils import RolloutStorage
from holosoma.agents.modules.modules import PerceptionTimeGRU
from holosoma.agents.ppo.ppo import PPO
from holosoma.config_types.algo import DistillationConfig, LayerConfig
from holosoma.config_types.observation import ObsTermCfg
from holosoma.config_types.perception import PerceptionConfig
from holosoma.managers.command.terms.wbt import motion_transition_contract_sha256
from holosoma.managers.perception.manager import PerceptionManager
from holosoma.utils.normalization import EmpiricalNormalization
from holosoma.utils.policy_init_preflight import (
    ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV,
    POLICY_INIT_REQUIRED_TERMINAL_TARGET_ENV,
)
from holosoma.utils.resume_preflight import (
    ALLOW_LEGACY_UNPROVENANCED_RESUME_ENV,
)
from holosoma.utils.rng_checkpoint import (
    ALLOW_NONDETERMINISTIC_RNG_RESUME_ENV,
    capture_rng_checkpoint_state,
    restore_rng_checkpoint_state,
)
from holosoma.utils.training_provenance import (
    embedded_runtime_asset_manifest_sha256,
    validate_training_provenance,
)


@pytest.fixture(autouse=True)
def _explicit_legacy_hatches_for_numerical_checkpoint_stubs(monkeypatch):
    """Keep legacy fixture payloads focused on their numerical contracts."""

    monkeypatch.setenv(ALLOW_LEGACY_UNPROVENANCED_RESUME_ENV, "1")
    monkeypatch.setenv(ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV, "1")


def test_ppo_requests_sparse_collection_extras_contract() -> None:
    requested_contracts = []
    env = SimpleNamespace(
        set_collection_extras_contract=lambda *, dense_episode_stats: requested_contracts.append(
            dense_episode_stats
        )
    )

    PPO._configure_collection_extras_contract(env)

    assert requested_contracts == [False]


def test_ppo_sparse_collection_contract_has_fake_env_fallback() -> None:
    env = SimpleNamespace()

    PPO._configure_collection_extras_contract(env)

    assert env._dense_episode_stats_each_step is False


def test_timeout_bootstrap_presence_uses_transition_local_final_observations(monkeypatch) -> None:
    def reject_device_reduction(*_args, **_kwargs):
        raise AssertionError("timeout presence must not call Tensor.any()")

    monkeypatch.setattr(torch.Tensor, "any", reject_device_reduction)
    time_outs = torch.tensor([False, True])

    assert not PPO._has_timeout_final_observations({"time_outs": time_outs})
    assert not PPO._has_timeout_final_observations(
        {"time_outs": time_outs, "final_observations": {}}
    )
    assert PPO._has_timeout_final_observations(
        {
            "time_outs": time_outs,
            "final_observations": {"critic_obs": torch.zeros(2, 3)},
        }
    )


class _TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.last_distribution_obs = None
        self.inference_obs = []
        self.distribution = None

    def update_distribution_from_policy_state(self, policy_state):
        obs = policy_state["actor_obs"]
        self.last_distribution_obs = obs.detach().clone()
        mean = obs[:, :1] * self.weight
        self.distribution = Normal(mean, torch.ones_like(mean))

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def act_inference(self, policy_state):
        obs = policy_state["actor_obs"]
        self.inference_obs.append(obs.detach().clone())
        return obs[:, :1] * self.weight


class _TinyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))

    def evaluate(self, policy_state):
        return policy_state["critic_obs"][:, :1] * self.weight


def _loss_stub(*, use_symmetry: bool = False) -> PPO:
    ppo = object.__new__(PPO)
    ppo.device = "cpu"
    ppo.current_learning_iteration = 0
    ppo.gpu_global_rank = 0
    ppo.gpu_world_size = 1
    ppo.is_multi_gpu = False
    ppo.use_time_gru = False
    ppo.use_symmetry = use_symmetry
    ppo.actor_perception_key = ""
    ppo.critic_perception_key = ""
    ppo.actor_obs_keys = ["actor_obs"]
    ppo.critic_obs_keys = ["critic_obs"]
    ppo.actor = _TinyActor()
    ppo.critic = _TinyCritic()
    ppo.dagger_enabled = False
    ppo.distill_enabled = False
    ppo.distill_mode = "mse"
    ppo.ppo_coeff = 1.0
    ppo.config = SimpleNamespace(
        desired_kl=None,
        schedule="fixed",
        clip_param=0.2,
        entropy_coef=0.0,
        symmetry_actor_coef=1.0 if use_symmetry else 0.0,
        symmetry_critic_coef=1.0 if use_symmetry else 0.0,
        value_loss_coef=1.0,
    )
    return ppo


def _optimizer_update_stub() -> PPO:
    ppo = _loss_stub()
    ppo.config.max_grad_norm = 1.0
    ppo.max_grad_norm = 1.0
    ppo.ppo_start_noise_std = None
    ppo.actor_optimizer = torch.optim.SGD(ppo.actor.parameters(), lr=0.1)
    ppo.critic_optimizer = torch.optim.SGD(ppo.critic.parameters(), lr=0.1)
    ppo._get_distributed_loss_weight = MethodType(lambda self: 1.0, ppo)
    return ppo


def _finite_update_loss(ppo: PPO, **extra):
    losses = {
        "actor_loss": (ppo.actor.weight - 1.0).pow(2),
        "critic_loss": (ppo.critic.weight - 1.0).pow(2),
        "value_loss": torch.tensor(0.5),
        "surrogate_loss": torch.tensor(0.25),
        "entropy_loss": torch.tensor(0.1),
        "kl_mean": torch.tensor(0.01),
    }
    losses.update(extra)
    return losses


def _minibatch(actor_obs: torch.Tensor) -> dict[str, torch.Tensor]:
    batch = actor_obs.shape[0]
    old_mean = actor_obs[:, :1] * 0.5
    actions = old_mean.clone()
    old_log_prob = Normal(old_mean, torch.ones_like(old_mean)).log_prob(actions).sum(-1, keepdim=True)
    return {
        "actor_obs": actor_obs,
        "critic_obs": actor_obs + 1.0,
        "actor_obs_raw": actor_obs * 10.0,
        "critic_obs_raw": (actor_obs + 1.0) * 10.0,
        "actions": actions,
        "values": torch.zeros(batch, 1),
        "advantages": torch.ones(batch, 1),
        "returns": torch.zeros(batch, 1),
        "actions_log_prob": old_log_prob,
        "action_mean": old_mean,
        "action_sigma": torch.ones_like(old_mean),
    }


def test_observation_slice_builder_rejects_duplicate_policy_groups() -> None:
    ppo = object.__new__(PPO)
    ppo.algo_obs_dim_dict = {"proprio": 3, "command": 2}

    with pytest.raises(ValueError, match="observation input groups must be unique"):
        ppo._build_obs_slices(["proprio", "command", "proprio"])


def test_model_setup_materializes_before_sync_and_optimizer(monkeypatch: pytest.MonkeyPatch):
    """Setup must expose lazy parameters to both broadcast and optimizers."""

    events: list[str] = []

    class _LazySetupModel(nn.Module):
        def __init__(self, role: str):
            super().__init__()
            self.role = role
            self.base = nn.Parameter(torch.tensor(1.0))
            self.backbone: nn.Linear | None = None

        def materialize_for_setup(self, device):
            events.append(f"materialize_{self.role}")
            self.backbone = nn.Linear(2, 2).to(device)

    actor = _LazySetupModel("actor")
    critic = _LazySetupModel("critic")
    ppo = object.__new__(PPO)
    ppo.device = "cpu"
    ppo.gpu_global_rank = 0
    ppo.is_multi_gpu = True
    ppo.use_symmetry = False
    ppo.algo_obs_dim_dict = {}
    ppo.algo_history_length_dict = {}
    ppo.num_act = 1
    ppo.actor_learning_rate = 1.0e-3
    ppo.critic_learning_rate = 1.0e-3
    optimizer_cfg = SimpleNamespace(_target_="torch.optim.SGD", weight_decay=0.0)
    ppo.config = SimpleNamespace(
        module_dict=SimpleNamespace(actor=object(), critic=object()),
        init_noise_std=0.1,
        actor_optimizer=optimizer_cfg,
        critic_optimizer=optimizer_cfg,
    )
    ppo._setup_distillation = MethodType(lambda self: None, ppo)
    ppo._validate_training_objective_configuration = MethodType(lambda self: None, ppo)

    def assert_materialized_at_sync(self):
        assert actor.backbone is not None
        assert critic.backbone is not None
        events.append("sync")

    ppo._synchronize_model_weights = MethodType(assert_materialized_at_sync, ppo)
    monkeypatch.setattr("holosoma.agents.ppo.ppo.setup_ppo_actor_module", lambda **kwargs: actor)
    monkeypatch.setattr("holosoma.agents.ppo.ppo.setup_ppo_critic_module", lambda **kwargs: critic)

    ppo._setup_models_and_optimizer()

    assert events == ["materialize_actor", "materialize_critic", "sync"]
    actor_optimizer_ids = {
        id(parameter)
        for parameter_group in ppo.actor_optimizer.param_groups
        for parameter in parameter_group["params"]
    }
    critic_optimizer_ids = {
        id(parameter)
        for parameter_group in ppo.critic_optimizer.param_groups
        for parameter in parameter_group["params"]
    }
    assert actor.backbone is not None
    assert critic.backbone is not None
    assert id(actor.backbone.weight) in actor_optimizer_ids
    assert id(critic.backbone.weight) in critic_optimizer_ids


def test_evaluation_only_model_setup_skips_teacher_and_optimizers(monkeypatch):
    ppo = object.__new__(PPO)
    ppo.device = "cpu"
    ppo.gpu_global_rank = 0
    ppo.is_multi_gpu = False
    ppo._evaluation_only = True
    ppo.use_symmetry = False
    ppo.algo_obs_dim_dict = {}
    ppo.algo_history_length_dict = {}
    ppo.num_act = 1
    ppo.config = SimpleNamespace(
        module_dict=SimpleNamespace(actor=object(), critic=object()),
        init_noise_std=0.1,
    )
    actor = nn.Linear(1, 1)
    critic = nn.Linear(1, 1)
    ppo._validate_training_objective_configuration = Mock(
        side_effect=AssertionError("training objective validation must not run for evaluation")
    )
    ppo._load_teacher_actor = Mock(
        side_effect=AssertionError("evaluation must not construct the training teacher")
    )
    instantiate_mock = Mock(
        side_effect=AssertionError("evaluation must not construct training optimizers")
    )
    monkeypatch.setattr(
        "holosoma.agents.ppo.ppo.setup_ppo_actor_module",
        lambda **_kwargs: actor,
    )
    monkeypatch.setattr(
        "holosoma.agents.ppo.ppo.setup_ppo_critic_module",
        lambda **_kwargs: critic,
    )
    monkeypatch.setattr("holosoma.agents.ppo.ppo.instantiate", instantiate_mock)

    ppo._setup_models_and_optimizer()

    assert ppo.actor is actor
    assert ppo.critic is critic
    assert ppo.distill_enabled is False
    assert ppo.dagger_enabled is False
    assert ppo.teacher_actor is None
    ppo._load_teacher_actor.assert_not_called()
    ppo._validate_training_objective_configuration.assert_not_called()
    instantiate_mock.assert_not_called()
    assert not hasattr(ppo, "actor_optimizer")
    assert not hasattr(ppo, "critic_optimizer")


def test_evaluation_only_setup_skips_rollout_storage():
    ppo = object.__new__(PPO)
    ppo._evaluation_only = True
    ppo.gpu_global_rank = 0
    ppo.is_multi_gpu = False
    ppo._setup_models_and_optimizer = Mock()
    ppo._configure_active_observation_groups = Mock()
    ppo._setup_storage = Mock(
        side_effect=AssertionError("evaluation must not allocate training rollout storage")
    )

    ppo.setup()

    ppo._setup_models_and_optimizer.assert_called_once_with()
    ppo._configure_active_observation_groups.assert_called_once_with()
    ppo._setup_storage.assert_not_called()


def test_lazy_model_materialization_does_not_advance_process_rng():
    class _NoisyLazyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.materialized_parameter: nn.Parameter | None = None

        def materialize_for_setup(self, device):
            random.random()
            np.random.random()
            self.materialized_parameter = nn.Parameter(torch.rand(3, device=device))

    original_rng = capture_rng_checkpoint_state()
    try:
        ppo = object.__new__(PPO)
        ppo.device = "cpu"
        model = _NoisyLazyModel()
        random.seed(3101)
        np.random.seed(3102)
        torch.manual_seed(3103)
        boundary = capture_rng_checkpoint_state()
        expected = (random.random(), float(np.random.random()), torch.rand(3))
        restore_rng_checkpoint_state(boundary)

        ppo._materialize_lazy_model_modules(model)

        assert model.materialized_parameter is not None
        assert random.random() == expected[0]
        assert float(np.random.random()) == expected[1]
        assert torch.equal(torch.rand(3), expected[2])
    finally:
        restore_rng_checkpoint_state(original_rng)


@pytest.mark.parametrize(
    ("checkpoint_restore", "expected_event"),
    [(False, "fresh"), (True, "restore")],
)
def test_lazy_model_materialization_selects_checkpoint_safe_hook(
    checkpoint_restore: bool,
    expected_event: str,
):
    events: list[str] = []

    class _CheckpointAwareLazyModel(nn.Module):
        def materialize_for_setup(self, _device):
            events.append("fresh")

        def materialize_for_checkpoint_restore(self, _device):
            events.append("restore")

    ppo = object.__new__(PPO)
    ppo.device = "cpu"
    ppo._materialize_lazy_model_modules(
        _CheckpointAwareLazyModel(),
        checkpoint_restore=checkpoint_restore,
    )

    assert events == [expected_event]


def test_optimizer_coverage_validation_rejects_missing_trainable_parameter():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD([model.bias], lr=1.0e-3)

    with pytest.raises(RuntimeError, match=r"actor optimizer.*weight"):
        PPO._validate_optimizer_parameter_coverage(model, optimizer, role="actor")


def test_parameter_finite_boundary_rejects_actor_nan_hidden_by_output_sanitization():
    ppo = _loss_stub()
    with torch.no_grad():
        ppo.actor.weight.fill_(float("nan"))

    with pytest.raises(
        FloatingPointError,
        match=r"training iteration 7.*actor\.weight.*corrupt policy state",
    ):
        ppo._assert_model_parameters_finite(
            phase="training iteration 7",
            trainable_only=True,
        )


def test_iteration_finite_boundary_rejects_non_finite_model_buffer():
    ppo = _loss_stub()
    ppo.actor.register_buffer("running_stat", torch.tensor(float("nan")))

    with pytest.raises(FloatingPointError, match=r"actor\.running_stat"):
        ppo._assert_model_parameters_finite(
            phase="training iteration 7",
            trainable_only=True,
        )


def test_parameter_finite_boundary_ignores_frozen_parameters_after_setup():
    ppo = _loss_stub()
    ppo.actor.weight.requires_grad_(False)
    with torch.no_grad():
        ppo.actor.weight.fill_(float("nan"))

    ppo._assert_model_parameters_finite(
        phase="training iteration 7",
        trainable_only=True,
    )

    with pytest.raises(FloatingPointError, match=r"model setup.*actor\.weight"):
        ppo._assert_model_parameters_finite(
            phase="model setup",
            trainable_only=False,
        )


def test_parameter_finite_boundary_propagates_remote_rank_failure(monkeypatch):
    ppo = _loss_stub()
    ppo.is_multi_gpu = True
    reduced_flags = []

    def report_remote_failure(self, flag, *, op):
        assert op == torch.distributed.ReduceOp.MAX
        assert flag.item() == 0
        reduced_flags.append(flag.detach().clone())
        return flag.new_tensor(1)

    ppo._all_reduce_small_tensor = MethodType(report_remote_failure, ppo)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    with pytest.raises(FloatingPointError, match="another rank reported"):
        ppo._assert_model_parameters_finite(
            phase="training iteration 7",
            trainable_only=True,
        )

    assert len(reduced_flags) == 1


def test_iteration_finite_boundary_rejects_non_finite_optimizer_state():
    ppo = _loss_stub()
    ppo.actor_optimizer = torch.optim.Adam(ppo.actor.parameters(), lr=1.0e-3)
    ppo.critic_optimizer = torch.optim.Adam(ppo.critic.parameters(), lr=1.0e-3)
    ppo.actor_optimizer.state[ppo.actor.weight]["exp_avg_sq"] = torch.tensor(float("inf"))

    with pytest.raises(
        FloatingPointError,
        match=r"actor_optimizer\.state\.0\.exp_avg_sq",
    ):
        ppo._assert_model_parameters_finite(
            phase="training iteration 7",
            trainable_only=True,
            include_optimizer_state=True,
        )


def test_iteration_finite_boundary_rejects_non_finite_optimizer_learning_rate():
    ppo = _loss_stub()
    ppo.actor_optimizer = torch.optim.SGD(ppo.actor.parameters(), lr=1.0e-3)
    ppo.critic_optimizer = torch.optim.SGD(ppo.critic.parameters(), lr=1.0e-3)
    ppo.actor_optimizer.param_groups[0]["lr"] = float("nan")

    with pytest.raises(
        FloatingPointError,
        match=r"actor_optimizer\.param_groups\[0\]\.lr",
    ):
        ppo._assert_model_parameters_finite(
            phase="training iteration 7",
            trainable_only=True,
            include_optimizer_state=True,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_loss_to_float_rejects_non_finite_values_instead_of_logging_zero(value):
    with pytest.raises(FloatingPointError, match="non-finite|NaN/Inf"):
        PPO._loss_to_float(torch.tensor(value))


def test_deferred_loss_accumulation_exactly_matches_eager_python_order():
    ppo = object.__new__(PPO)
    eager = {"Value": 0.0, "Surrogate": 0.0, "Entropy": 0.0, "KL": 0.0}
    deferred = {"Value": [], "Surrogate": [], "Entropy": [], "KL": []}
    minibatch_losses = [
        {
            "value_loss": torch.tensor(0.1, dtype=torch.float32),
            "surrogate_loss": torch.tensor(-0.2, dtype=torch.float32),
            "entropy_loss": torch.tensor(0.3, dtype=torch.float64),
            "kl_mean": torch.tensor(0.4, dtype=torch.float32),
            "actor_loss": torch.tensor(0.5, dtype=torch.float32),
            "vector_metric": torch.tensor([0.25, 0.75], dtype=torch.float32),
            "python_metric": 0.125,
        },
        {
            "value_loss": torch.tensor(0.6, dtype=torch.float32),
            "surrogate_loss": torch.tensor(-0.7, dtype=torch.float32),
            "entropy_loss": torch.tensor(0.8, dtype=torch.float64),
            "kl_mean": torch.tensor(0.9, dtype=torch.float32),
            "actor_loss": torch.tensor(1.0, dtype=torch.float32),
            "vector_metric": torch.tensor([1.25, 1.75], dtype=torch.float32),
            "python_metric": 0.375,
        },
    ]

    for losses in minibatch_losses:
        ppo._accumulate_loss_dict(eager, losses)
        ppo._accumulate_loss_dict(deferred, losses, defer_host_sync=True)

    expected = {key: value / len(minibatch_losses) for key, value in eager.items()}
    expected["teacher_bc_mask_fraction"] = float(torch.tensor(0.625).item())
    actual = ppo._finalize_deferred_loss_dict(
        deferred,
        num_updates=len(minibatch_losses),
        extras={"teacher_bc_mask_fraction": torch.tensor(0.625)},
    )

    assert actual == expected


def test_deferred_loss_success_path_never_calls_tensor_item():
    ppo = object.__new__(PPO)
    deferred = {"Value": [], "Surrogate": [], "Entropy": [], "KL": []}
    losses = {
        "value_loss": torch.tensor(1.0),
        "surrogate_loss": torch.tensor(2.0),
        "entropy_loss": torch.tensor(3.0),
        "kl_mean": torch.tensor(4.0),
        "vector_metric": torch.tensor([5.0, 7.0]),
    }

    with patch.object(torch.Tensor, "item", side_effect=AssertionError("unexpected item()")):
        ppo._accumulate_loss_dict(deferred, losses, defer_host_sync=True)
        finalized = ppo._finalize_deferred_loss_dict(deferred, num_updates=1)

    assert finalized == {
        "Value": 1.0,
        "Surrogate": 2.0,
        "Entropy": 3.0,
        "KL": 4.0,
        "vector_metric": 6.0,
    }


def test_batched_loss_finite_check_reports_fields_without_tensor_item():
    losses = {
        "finite": torch.tensor(1.0),
        "nan_field": torch.tensor(float("nan")),
        "inf_vector": torch.tensor([1.0, float("inf")]),
        "python_inf": float("-inf"),
    }

    with patch.object(torch.Tensor, "item", side_effect=AssertionError("unexpected item()")):
        invalid = PPO._invalid_loss_fields_batched(losses)

    assert invalid == ["nan_field", "inf_vector", "python_inf"]


def test_loss_logging_rejects_complex_values_without_discarding_imaginary_part():
    complex_loss = torch.tensor(1.0 + 2.0j)

    assert PPO._invalid_loss_fields_batched({"complex_loss": complex_loss}) == [
        "complex_loss"
    ]
    with pytest.raises(TypeError, match="real-valued"):
        PPO._loss_to_float(complex_loss)
    with pytest.raises(TypeError, match="real-valued"):
        PPO._loss_to_deferred_scalar(complex_loss)


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, "invalid"],
)
def test_max_grad_norm_validation_rejects_non_positive_or_non_finite_values(value):
    with pytest.raises(ValueError, match="max_grad_norm must be finite and > 0"):
        PPO._validate_max_grad_norm(value)


def test_max_grad_norm_validation_normalizes_positive_numeric_value():
    assert PPO._validate_max_grad_norm(0.5) == pytest.approx(0.5)


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_distillation_integer_fields_reject_bool_fractional_and_string_values(value):
    with pytest.raises(ValueError, match="fixed_bc_eval_num_samples must be an integer"):
        PPO._strict_config_int("fixed_bc_eval_num_samples", value)


@pytest.mark.parametrize("value", [0, 1, "False", None])
def test_distillation_boolean_fields_reject_truthy_or_falsy_non_booleans(value):
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        PPO._strict_config_bool("enabled", value)


@pytest.mark.parametrize(
    "name",
    [
        "HOLOSOMA_SKIP_GRAD_FINITE_CHECK",
        "HOLOSOMA_SKIP_LOSS_FINITE_CHECK",
        "HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION",
        "HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC",
    ],
)
def test_direct_ppo_rejects_scientific_skip_switches(monkeypatch, name):
    for variable in (
        "HOLOSOMA_SKIP_GRAD_FINITE_CHECK",
        "HOLOSOMA_SKIP_LOSS_FINITE_CHECK",
        "HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION",
        "HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv(name, "true")

    with pytest.raises(RuntimeError, match=rf"{name} cannot be enabled"):
        PPO._validate_scientific_fail_closed_environment()


def test_direct_ppo_rejects_invalid_scientific_skip_value(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_SKIP_LOSS_FINITE_CHECK", "sometimes")

    with pytest.raises(ValueError, match="must be an explicit boolean"):
        PPO._validate_scientific_fail_closed_environment()


@pytest.mark.parametrize(
    "name",
    [
        "HOLOSOMA_DISABLE_ACTIVE_OBS_GROUP_FILTER",
        "HOLOSOMA_DISABLE_AUTO_RESET",
        "HOLOSOMA_DISABLE_CLIP_END_RESET",
        "HOLOSOMA_DISABLE_MOTION_END_RESET",
        "HOLOSOMA_DISABLE_BAD_TRACKING_RESET",
    ],
)
def test_scientific_training_rejects_ambient_semantic_bypasses(monkeypatch, name):
    ppo = _batch_norm_objective_validation_stub(permanently_pure_bc=True)
    monkeypatch.setenv(name, "true")

    with pytest.raises(RuntimeError, match=rf"{name} cannot be enabled"):
        ppo._validate_training_objective_configuration()


def test_scientific_training_requires_pre_gradient_sync(monkeypatch):
    ppo = _batch_norm_objective_validation_stub(permanently_pure_bc=True)
    monkeypatch.setenv("HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE", "false")

    with pytest.raises(RuntimeError, match="SYNC_BEFORE_GRAD_ALLREDUCE cannot be disabled"):
        ppo._validate_training_objective_configuration()


def test_scientific_training_rejects_invalid_semantic_override_boolean(monkeypatch):
    ppo = _batch_norm_objective_validation_stub(permanently_pure_bc=True)
    monkeypatch.setenv("HOLOSOMA_DISABLE_AUTO_RESET", "sometimes")

    with pytest.raises(ValueError, match="must be an explicit boolean"):
        ppo._validate_training_objective_configuration()


def test_update_rejects_non_finite_auxiliary_loss_before_backward_or_step():
    ppo = _optimizer_update_stub()
    original_actor = ppo.actor.weight.detach().clone()
    original_critic = ppo.critic.weight.detach().clone()
    ppo._compute_ppo_loss = MethodType(
        lambda self, minibatch: _finite_update_loss(self, kl_mean=torch.tensor(float("nan"))),
        ppo,
    )

    with pytest.raises(FloatingPointError, match=r"non-finite fields: kl_mean"):
        ppo._update_algo_step({}, {"Value": 0.0, "Surrogate": 0.0, "Entropy": 0.0, "KL": 0.0})

    assert torch.equal(ppo.actor.weight, original_actor)
    assert torch.equal(ppo.critic.weight, original_critic)


def test_update_synchronizes_compute_failure_and_clears_all_gradients():
    ppo = _optimizer_update_stub()
    ppo.actor.weight.grad = torch.ones_like(ppo.actor.weight)
    ppo.critic.weight.grad = torch.ones_like(ppo.critic.weight)
    ppo._compute_ppo_loss = MethodType(
        lambda self, minibatch: (_ for _ in ()).throw(RuntimeError("actor forward failed")),
        ppo,
    )

    with pytest.raises(RuntimeError, match="actor forward failed"):
        ppo._update_algo_step(
            {},
            {"Value": 0.0, "Surrogate": 0.0, "Entropy": 0.0, "KL": 0.0},
        )

    assert ppo.actor.weight.grad is None
    assert ppo.critic.weight.grad is None
    assert ppo.actor_optimizer.state == {}
    assert ppo.critic_optimizer.state == {}


def test_authoritative_dagger_control_mode_never_branches_on_rank_local_fields():
    ppo = _loss_stub()
    ppo.is_multi_gpu = True
    ppo.distill_mode = "dagger"
    ppo.dagger_enabled = True
    ppo.use_ppo_dagger_schedule = False
    ppo.bc_loss_coef = 1.0
    ppo._all_reduce_small_tensor = Mock(
        side_effect=AssertionError("authoritative controls must not enter fallback reduction")
    )
    complete = {
        "_dagger_bc_denominator": torch.tensor(2.0),
        "_dagger_bc_has_valid_samples": True,
    }
    missing: dict[str, object] = {}

    assert ppo._prepare_distributed_dagger_minibatch_controls(
        complete,
        controls_authoritative=True,
    ) is complete
    assert ppo._prepare_distributed_dagger_minibatch_controls(
        missing,
        controls_authoritative=True,
    ) is missing
    ppo._compute_loss_requires_prepared_dagger_controls = True
    try:
        denominator, presence = ppo._bc_denominator_and_presence_for_minibatch(
            complete,
            torch.tensor(1.0),
        )
        assert denominator.item() == pytest.approx(2.0)
        assert presence is True
        with pytest.raises(RuntimeError, match="denominator was not prepared"):
            ppo._bc_denominator_and_presence_for_minibatch(
                missing,
                torch.tensor(1.0),
            )
    finally:
        ppo._compute_loss_requires_prepared_dagger_controls = False
    ppo._all_reduce_small_tensor.assert_not_called()


def test_dagger_fallback_synchronizes_post_count_finalization_failure():
    ppo = _loss_stub()
    ppo.is_multi_gpu = True
    ppo.gpu_world_size = 2
    ppo.distill_mode = "dagger"
    ppo.dagger_enabled = True
    ppo.use_ppo_dagger_schedule = False
    ppo.bc_loss_coef = 1.0
    ppo.dagger_ignore_zero_teacher_actions = False
    ppo._get_distributed_loss_weight = MethodType(lambda self: 1.0, ppo)
    ppo._all_reduce_small_tensor = MethodType(
        lambda self, tensor, *, op: tensor,
        ppo,
    )
    verdicts: list[tuple[Exception | None, str]] = []

    def synchronize(self, local_error, *, operation):
        verdicts.append((local_error, operation))
        if local_error is not None:
            raise local_error

    ppo._synchronize_training_phase_error = MethodType(synchronize, ppo)
    minibatch = {
        "actions": torch.zeros(2, 1),
        "teacher_actions": torch.ones(2, 1),
    }

    with (
        patch.object(torch.Tensor, "item", side_effect=RuntimeError("presence copy failed")),
        pytest.raises(RuntimeError, match="presence copy failed"),
    ):
        ppo._prepare_distributed_dagger_minibatch_controls(
            minibatch,
            controls_authoritative=False,
        )

    assert verdicts[0] == (
        None,
        "PPO distributed DAgger minibatch-control preflight",
    )
    assert isinstance(verdicts[1][0], RuntimeError)
    assert verdicts[1][1] == (
        "PPO distributed DAgger minibatch-control finalization"
    )


def test_distributed_kl_payload_failure_joins_common_verdict_before_collectives():
    ppo = _optimizer_update_stub()
    ppo.is_multi_gpu = True
    ppo._compute_ppo_loss = MethodType(
        lambda self, minibatch: _finite_update_loss(
            self,
            _reduce_kl_before_optimizer=True,
        ),
        ppo,
    )
    ppo._build_distributed_kl_payload = Mock(
        side_effect=RuntimeError("injected KL payload allocation failure")
    )
    ppo._reduce_parameters = Mock(
        side_effect=AssertionError("gradient collective must not be entered")
    )
    verdicts: list[tuple[Exception | None, str]] = []

    def synchronize(self, local_error, *, operation):
        verdicts.append((local_error, operation))
        if local_error is not None:
            raise local_error

    ppo._synchronize_training_phase_error = MethodType(synchronize, ppo)

    with pytest.raises(RuntimeError, match="injected KL payload allocation failure"):
        ppo._update_algo_step(
            {},
            {"Value": 0.0, "Surrogate": 0.0, "Entropy": 0.0, "KL": 0.0},
        )

    assert len(verdicts) == 1
    assert verdicts[0][1] == "PPO loss validation/backward"
    assert isinstance(verdicts[0][0], RuntimeError)
    ppo._reduce_parameters.assert_not_called()
    assert ppo.actor.weight.grad is None
    assert ppo.critic.weight.grad is None


def test_update_propagates_remote_rank_non_finite_loss(monkeypatch):
    ppo = _optimizer_update_stub()
    ppo.is_multi_gpu = True
    ppo._compute_ppo_loss = MethodType(
        lambda self, minibatch: _finite_update_loss(self),
        ppo,
    )
    verdicts = []

    def inject_remote_loss_failure(self, local_error, *, operation):
        verdicts.append((local_error, operation))
        raise FloatingPointError("another rank reported a non-finite PPO loss field")

    ppo._synchronize_training_phase_error = MethodType(
        inject_remote_loss_failure,
        ppo,
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    with pytest.raises(FloatingPointError, match="another rank reported a non-finite PPO loss field"):
        ppo._update_algo_step({}, {"Value": 0.0, "Surrogate": 0.0, "Entropy": 0.0, "KL": 0.0})

    assert verdicts == [(None, "PPO loss validation/backward")]
    assert ppo.actor.weight.grad is None
    assert ppo.critic.weight.grad is None


def test_update_clips_actor_and_critic_with_error_if_nonfinite():
    ppo = _optimizer_update_stub()
    ppo._compute_ppo_loss = MethodType(
        lambda self, minibatch: _finite_update_loss(self),
        ppo,
    )
    clip_calls = []

    def record_clip(parameters, max_norm, *, error_if_nonfinite):
        clip_calls.append((tuple(parameters), max_norm, error_if_nonfinite))
        return torch.tensor(0.0)

    with patch("holosoma.agents.ppo.ppo.nn.utils.clip_grad_norm_", side_effect=record_clip):
        ppo._update_algo_step({}, {"Value": 0.0, "Surrogate": 0.0, "Entropy": 0.0, "KL": 0.0})

    assert len(clip_calls) == 2
    assert all(max_norm == 1.0 and error_if_nonfinite is True for _, max_norm, error_if_nonfinite in clip_calls)


def test_update_reduces_kl_after_backward_and_before_optimizer_step():
    ppo = _optimizer_update_stub()
    events: list[str] = []
    ppo._compute_ppo_loss = MethodType(
        lambda self, minibatch: _finite_update_loss(
            self,
            _reduce_kl_before_optimizer=True,
        ),
        ppo,
    )

    def reduce_kl(self, kl_mean, *, distributed_payload=None):
        assert distributed_payload is None
        assert self.actor.weight.grad is not None
        assert self.critic.weight.grad is not None
        events.append("kl")
        return kl_mean * 2.0

    ppo._reduce_kl_after_local_loss = MethodType(reduce_kl, ppo)
    ppo._step_actor_optimizer = MethodType(
        lambda self: events.append("optimizer"),
        ppo,
    )

    def record_clip(parameters, max_norm, *, error_if_nonfinite):  # noqa: ARG001
        assert events and events[0] == "kl"
        events.append("clip")
        return torch.tensor(0.0)

    loss_dict = {"Value": 0.0, "Surrogate": 0.0, "Entropy": 0.0, "KL": 0.0}
    with patch(
        "holosoma.agents.ppo.ppo.nn.utils.clip_grad_norm_",
        side_effect=record_clip,
    ):
        result = ppo._update_algo_step({}, loss_dict)

    assert events == ["kl", "clip", "clip", "optimizer"]
    assert result["KL"] == pytest.approx(0.02)


def test_gradient_clip_failure_prevents_all_optimizer_steps():
    ppo = _optimizer_update_stub()
    ppo._compute_ppo_loss = MethodType(
        lambda self, minibatch: _finite_update_loss(self),
        ppo,
    )
    original_actor = ppo.actor.weight.detach().clone()
    original_critic = ppo.critic.weight.detach().clone()

    with (
        patch(
            "holosoma.agents.ppo.ppo.nn.utils.clip_grad_norm_",
            side_effect=RuntimeError("non-finite total norm"),
        ),
        pytest.raises(RuntimeError, match="non-finite total norm"),
    ):
        ppo._update_algo_step({}, {"Value": 0.0, "Surrogate": 0.0, "Entropy": 0.0, "KL": 0.0})

    assert torch.equal(ppo.actor.weight, original_actor)
    assert torch.equal(ppo.critic.weight, original_critic)


def test_non_finite_backward_gradient_is_rejected_by_clip_before_optimizer_steps():
    ppo = _optimizer_update_stub()
    ppo._compute_ppo_loss = MethodType(
        lambda self, minibatch: _finite_update_loss(self),
        ppo,
    )
    original_actor = ppo.actor.weight.detach().clone()
    original_critic = ppo.critic.weight.detach().clone()

    hook = ppo.actor.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("nan"))
    )
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            ppo._update_algo_step(
                {},
                {"Value": 0.0, "Surrogate": 0.0, "Entropy": 0.0, "KL": 0.0},
            )
    finally:
        hook.remove()

    assert torch.equal(ppo.actor.weight, original_actor)
    assert torch.equal(ppo.critic.weight, original_critic)
    assert ppo.actor_optimizer.state == {}
    assert ppo.critic_optimizer.state == {}


def test_zero_critic_objective_skips_adamw_step_and_weight_decay(monkeypatch):
    monkeypatch.delenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", raising=False)
    ppo = _optimizer_update_stub()
    ppo.config.value_loss_coef = 0.0
    ppo.config.symmetry_critic_coef = 0.0
    ppo.critic_optimizer = torch.optim.AdamW(
        ppo.critic.parameters(),
        lr=0.1,
        weight_decay=0.5,
    )
    ppo._compute_ppo_loss = MethodType(
        lambda self, minibatch: _finite_update_loss(
            self,
            critic_loss=self.critic.weight.pow(2).sum() * 0.0,
        ),
        ppo,
    )
    original_critic = ppo.critic.weight.detach().clone()

    ppo._update_algo_step(
        {},
        {"Value": 0.0, "Surrogate": 0.0, "Entropy": 0.0, "KL": 0.0},
    )

    assert torch.equal(ppo.critic.weight, original_critic)
    assert ppo.critic_optimizer.state == {}


def test_supervised_only_train_mode_keeps_frozen_critic_state_in_eval():
    ppo = object.__new__(PPO)
    ppo._supervised_dagger_only = True
    ppo.use_symmetry = False
    ppo.config = SimpleNamespace(value_loss_coef=1.0, symmetry_critic_coef=0.0)
    ppo.actor = nn.Sequential(nn.Dropout(p=0.5))
    ppo.critic = nn.Sequential(nn.Dropout(p=0.5), nn.BatchNorm1d(2))
    ppo.actor_obs_normalizers = {"actor": nn.BatchNorm1d(2)}
    ppo.critic_obs_normalizers = {"critic": nn.BatchNorm1d(2)}
    ppo.teacher_actor = None
    ppo.teacher_actors = []

    ppo._train_mode()

    assert ppo.actor.training is True
    assert ppo.actor_obs_normalizers["actor"].training is True
    assert ppo.critic.training is False
    assert ppo.critic_obs_normalizers["critic"].training is False


def test_ppo_update_uses_exact_collection_input_without_renormalizing():
    ppo = _loss_stub()
    ppo._normalize_actor_obs = MethodType(
        lambda self, obs, update: (_ for _ in ()).throw(AssertionError("unexpected actor renormalization")),
        ppo,
    )
    ppo._normalize_critic_obs = MethodType(
        lambda self, obs, update: (_ for _ in ()).throw(AssertionError("unexpected critic renormalization")),
        ppo,
    )
    actor_obs = torch.tensor([[0.25], [1.5], [-2.0]])

    losses = ppo._compute_ppo_loss(_minibatch(actor_obs))

    assert torch.equal(ppo.actor.last_distribution_obs, actor_obs)
    assert losses["surrogate_loss"].item() == pytest.approx(-1.0)


class _NegatingSymmetry:
    @staticmethod
    def augment_observations(obs, env, obs_list):
        return torch.cat((obs, -obs), dim=0)

    @staticmethod
    def augment_actions(actions):
        return torch.cat((actions, -actions), dim=0)


def test_symmetry_is_auxiliary_and_does_not_duplicate_ppo_samples():
    ppo = _loss_stub(use_symmetry=True)
    ppo.env = object()
    ppo.symmetry_utils = _NegatingSymmetry()
    ppo._normalize_actor_obs = MethodType(lambda self, obs, update: obs / 10.0, ppo)
    ppo._normalize_critic_obs = MethodType(lambda self, obs, update: obs / 10.0, ppo)
    actor_obs = torch.tensor([[1.0], [2.0]])

    losses = ppo._compute_ppo_loss(_minibatch(actor_obs))

    assert ppo.actor.last_distribution_obs.shape[0] == 2
    assert losses["surrogate_loss"].item() == pytest.approx(-1.0)
    assert losses["symmetry_actor_loss"].item() == pytest.approx(0.0)


def test_time_gru_sequence_matches_rollout_hidden_and_post_transition_resets():
    torch.manual_seed(7)
    gru = PerceptionTimeGRU(3, 4, LayerConfig())
    x_seq = torch.randn(5, 2, 3)
    dones = torch.tensor(
        [
            [[False], [False]],
            [[True], [False]],
            [[False], [False]],
            [[False], [True]],
            [[False], [False]],
        ]
    )
    initial_hidden = torch.randn(1, 2, 4)
    gru.hidden = initial_hidden.clone()
    rollout_outputs = []
    for step in range(x_seq.shape[0]):
        rollout_outputs.append(gru.step(x_seq[step]))
        gru.reset(dones[step])

    sequence_outputs = gru.forward_sequence(
        x_seq,
        dones_seq=dones,
        initial_hidden=initial_hidden,
    )

    assert torch.allclose(sequence_outputs, torch.stack(rollout_outputs), atol=1e-6)


def test_ppo_rollout_storage_adapter_submits_exact_registered_transition_schema():
    ppo = object.__new__(PPO)
    ppo.storage = RolloutStorage(num_envs=2, num_transitions_per_env=1)
    ppo.storage.register("actor_obs", shape=(1,))
    ppo.storage.register("actions", shape=(1,))
    ppo.storage.register("returns", shape=(1,), required_on_add=False)

    actor_obs = torch.tensor([[1.0], [2.0]])
    actions = torch.tensor([[3.0], [4.0]])
    ppo._add_rollout_storage_transition(
        {
            "actor_obs": actor_obs,
            "actions": actions,
            # PPO computes one superset across optional configurations. These
            # values are deliberately absent from the active storage schema.
            "teacher_actions": torch.zeros_like(actions),
            "actor_gru_hidden": None,
        }
    )

    assert ppo.storage.step == 1
    assert torch.equal(ppo.storage["actor_obs"][0], actor_obs)
    assert torch.equal(ppo.storage["actions"][0], actions)
    assert ppo.storage.registered_keys == frozenset({"actor_obs", "actions", "returns"})


def test_ppo_storage_schema_marks_only_returns_and_advantages_as_derived():
    ppo = object.__new__(PPO)
    ppo.device = "cpu"
    ppo.env = SimpleNamespace(num_envs=2)
    ppo.config = SimpleNamespace(num_steps_per_env=3)
    ppo.num_act = 2
    ppo.actor_obs_keys = ["actor_obs"]
    ppo.critic_obs_keys = ["critic_obs"]
    ppo.algo_obs_dim_dict = {"actor_obs": 4, "critic_obs": 5}
    ppo.use_symmetry = False
    ppo.distill_enabled = False
    ppo.dagger_enabled = False
    ppo.actor_perception_key = ""
    ppo.critic_perception_key = ""
    ppo.use_time_gru = False

    ppo._setup_storage()

    assert ppo.storage.derived_keys == frozenset({"returns", "advantages"})
    assert ppo.storage.required_on_add_keys == ppo.storage.registered_keys.difference(
        ppo.storage.derived_keys
    )


def test_supervised_only_storage_and_training_step_do_not_request_critic_perception(
    monkeypatch,
) -> None:
    ppo = object.__new__(PPO)
    ppo._supervised_dagger_only = True
    ppo.device = "cpu"
    ppo.env = SimpleNamespace(num_envs=2)
    ppo.config = SimpleNamespace(
        num_steps_per_env=3,
        num_learning_epochs=1,
        num_mini_batches=1,
    )
    ppo.num_act = 2
    ppo.actor_obs_keys = ["actor_obs"]
    ppo.critic_obs_keys = ["critic_obs"]
    ppo.algo_obs_dim_dict = {
        "actor_obs": 4,
        "critic_obs": 5,
        "actor_depth": 6,
        "critic_depth": 7,
    }
    ppo.use_symmetry = False
    ppo.distill_enabled = True
    ppo.dagger_enabled = True
    ppo.use_multi_teacher = False
    ppo.dagger_ignore_episode_initial_steps = 0
    ppo.actor_perception_key = "actor_depth"
    ppo.critic_perception_key = "critic_depth"
    ppo.use_time_gru = False

    ppo._setup_storage()

    assert "actor_depth" in ppo.storage.registered_keys
    assert "critic_depth" not in ppo.storage.registered_keys
    assert "critic_obs" in ppo.storage.registered_keys
    assert "values" in ppo.storage.registered_keys

    requested_keys = None

    def capture_minibatch_keys(num_mini_batches, num_epochs, *, keys):
        nonlocal requested_keys
        assert num_mini_batches == 1
        assert num_epochs == 1
        requested_keys = set(keys)
        return iter(())

    monkeypatch.setattr(ppo.storage, "mini_batch_generator", capture_minibatch_keys)
    ppo.algo_timing = SimpleNamespace(enabled=False)
    ppo.current_learning_iteration = 0
    ppo._assert_rollout_storage_finite = MethodType(lambda self: None, ppo)
    ppo._rollout_bc_denominator_per_minibatch = MethodType(lambda self: None, ppo)
    ppo._assert_model_parameters_finite = MethodType(lambda self, **kwargs: None, ppo)
    ppo._debug_training_phase = MethodType(lambda self, *args, **kwargs: None, ppo)

    ppo._training_step()

    assert requested_keys is not None
    assert "actor_depth" in requested_keys
    assert "critic_depth" not in requested_keys


def test_ppo_rollout_storage_adapter_rejects_missing_registered_transition_field():
    ppo = object.__new__(PPO)
    ppo.storage = RolloutStorage(num_envs=1, num_transitions_per_env=1)
    ppo.storage.register("actor_obs", shape=(1,))
    ppo.storage.register("actions", shape=(1,))

    with pytest.raises(KeyError, match="missing fields required by storage"):
        ppo._add_rollout_storage_transition({"actor_obs": torch.ones(1, 1)})
    assert ppo.storage.step == 0


def test_global_bc_denominator_composes_with_rank_weighted_gradient_reduction():
    ppo = object.__new__(PPO)
    ppo.is_multi_gpu = True
    ppo.gpu_world_size = 2
    ppo._get_distributed_loss_weight = MethodType(lambda self: 2.0, ppo)
    ppo._all_reduce_small_tensor = MethodType(
        lambda self, value, op: value + value.new_tensor(3.0),
        ppo,
    )

    denominator = ppo._global_bc_denominator(torch.tensor(2.0))

    # Local weighted count=2*2=4; remote weighted count=3; divide by
    # world-size because the gradient reducer averages ranks.
    assert denominator.item() == pytest.approx(3.5)


def test_advantage_statistics_use_distributed_loss_weights():
    ppo = object.__new__(PPO)
    ppo._get_distributed_loss_weight = MethodType(lambda self: 2.0, ppo)
    ppo._all_reduce_small_tensor = MethodType(
        lambda self, payload, op: payload + payload.new_tensor([12.0, 148.0, 1.0]),
        ppo,
    )
    advantages = torch.tensor([1.0, 3.0])

    normalized = ppo._normalize_advantages_multi_gpu(advantages)

    expected_mean = 16.0 / 3.0
    expected_variance = 158.0 / 3.0 - expected_mean**2
    assert torch.allclose(
        normalized,
        (advantages - expected_mean) / (expected_variance + 1e-8) ** 0.5,
    )


def test_single_rank_advantage_normalization_matches_population_formula_and_is_finite_for_one_sample():
    ppo = object.__new__(PPO)
    ppo.is_multi_gpu = False
    ppo.config = SimpleNamespace(gamma=0.0, lam=0.0)

    _, normalized = ppo._compute_returns_and_advantages(
        last_values=torch.zeros(1),
        values=torch.zeros(1, 1),
        dones=torch.ones(1, 1, dtype=torch.bool),
        rewards=torch.tensor([[3.0]]),
    )

    assert torch.equal(normalized, torch.zeros_like(normalized))
    assert torch.isfinite(normalized).all()


def test_gradient_reduction_excludes_frozen_backbone_from_payload():
    class _ActorWithFrozenBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen_backbone = nn.Parameter(torch.zeros(10_000), requires_grad=False)
            self.trainable_head = nn.Parameter(torch.tensor([1.0, 2.0]))

    ppo = object.__new__(PPO)
    ppo.actor = _ActorWithFrozenBackbone()
    ppo.critic = nn.Linear(1, 1)
    ppo.actor.trainable_head.grad = torch.tensor([3.0, 4.0])
    ppo.gpu_world_size = 1
    ppo.gpu_global_rank = 0
    ppo.current_learning_iteration = 0
    ppo._get_distributed_loss_weight = MethodType(lambda self: 1.0, ppo)
    captured_payloads: list[torch.Tensor] = []

    def capture_payload(self, payload: torch.Tensor) -> str:
        captured_payloads.append(payload.detach().clone())
        return "unit-test"

    ppo._all_reduce_grad_payload = MethodType(capture_payload, ppo)

    ppo._reduce_parameters(include_critic=False)

    assert len(captured_payloads) == 1
    # Two trainable gradient elements plus one trainable-parameter mask bit;
    # none of the 10,000 frozen elements may enter the collective.
    assert torch.equal(captured_payloads[0], torch.tensor([3.0, 4.0, 1.0]))
    assert torch.equal(ppo.actor.trainable_head.grad, torch.tensor([3.0, 4.0]))


def test_canonical_rollout_reset_clears_all_policy_and_teacher_recurrent_state():
    def recurrent_model(value: float):
        return SimpleNamespace(
            perception_time_gru=SimpleNamespace(hidden=torch.tensor([value]))
        )

    ppo = object.__new__(PPO)
    ppo.actor = recurrent_model(1.0)
    ppo.critic = recurrent_model(2.0)
    ppo.teacher_actor = recurrent_model(3.0)
    ppo.teacher_actors = [recurrent_model(4.0), recurrent_model(5.0)]

    ppo._reset_recurrent_rollout_state()

    models = [ppo.actor, ppo.critic, ppo.teacher_actor, *ppo.teacher_actors]
    assert all(model.perception_time_gru.hidden is None for model in models)


@pytest.mark.parametrize(
    ("configured", "expected_calls"),
    [(None, 1), ("1", 1), ("true", 1), ("0", 0), ("false", 0)],
)
def test_multi_gpu_update_synchronizes_cuda_before_gradient_reduction(
    monkeypatch: pytest.MonkeyPatch,
    configured: str | None,
    expected_calls: int,
):
    """The pre-reduce device boundary defaults safe and remains explicitly configurable."""
    ppo = object.__new__(PPO)
    ppo.device = "cuda:3"

    if configured is None:
        monkeypatch.delenv("HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE", raising=False)
    else:
        monkeypatch.setenv("HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE", configured)

    with patch("torch.cuda.synchronize") as synchronize:
        enabled = ppo._synchronize_cuda_before_gradient_reduction()

    assert enabled is bool(expected_calls)
    assert synchronize.call_count == expected_calls
    if expected_calls:
        synchronize.assert_called_once_with("cuda:3")


def test_advantage_statistics_reject_weight_sum_that_changes_gradient_scale():
    ppo = object.__new__(PPO)
    ppo.gpu_world_size = 2
    ppo._get_distributed_loss_weight = MethodType(lambda self: 1.0, ppo)
    ppo._all_reduce_small_tensor = MethodType(
        lambda self, payload, op: payload + payload.new_tensor([0.0, 0.0, 0.25]),
        ppo,
    )

    with pytest.raises(ValueError, match="must sum to world_size before optimization"):
        ppo._normalize_advantages_multi_gpu(torch.tensor([1.0, 3.0]))


def test_actor_std_sanitization_projects_negative_raw_parameter():
    ppo = object.__new__(PPO)
    ppo.actor = SimpleNamespace(
        std=nn.Parameter(torch.tensor([-0.1, 0.2])),
        min_noise_std=0.01,
        min_mean_noise_std=None,
        max_noise_std=0.5,
    )
    ppo.config = SimpleNamespace(init_noise_std=0.2)

    ppo._sanitize_actor_std()

    assert torch.isfinite(ppo.actor.std).all()
    assert ppo.actor.std.detach().tolist() == pytest.approx([0.01, 0.2])


@pytest.mark.parametrize("invalid_std", [float("nan"), float("inf"), float("-inf")])
def test_actor_std_projection_preserves_non_finite_parameter(invalid_std):
    ppo = object.__new__(PPO)
    ppo.actor = SimpleNamespace(
        std=nn.Parameter(torch.tensor([0.2, invalid_std])),
        min_noise_std=0.01,
        min_mean_noise_std=None,
        max_noise_std=0.5,
    )

    ppo._sanitize_actor_std()

    assert not torch.isfinite(ppo.actor.std[1])


@pytest.mark.parametrize("invalid_std", [float("nan"), float("inf"), float("-inf")])
def test_actor_optimizer_step_rejects_non_finite_std_before_projection(invalid_std):
    ppo = object.__new__(PPO)
    ppo.actor = SimpleNamespace(
        std=nn.Parameter(torch.tensor([0.2, 0.3])),
        min_noise_std=0.01,
        min_mean_noise_std=None,
        max_noise_std=0.5,
    )
    ppo.actor_optimizer = SimpleNamespace(
        step=lambda: ppo.actor.std.data.copy_(torch.tensor([0.2, invalid_std]))
    )

    with pytest.raises(FloatingPointError, match="optimizer produced NaN/Inf std"):
        ppo._step_actor_optimizer()

    assert not torch.isfinite(ppo.actor.std[1])


def test_teacher_action_clipping_never_turns_non_finite_values_finite():
    teacher_actions = torch.tensor([[2.0, -2.0, float("inf"), float("-inf"), float("nan")]])

    clipped = PPO._clip_teacher_actions_preserving_non_finite(teacher_actions, 0.5)

    assert clipped[0, :2].tolist() == pytest.approx([0.5, -0.5])
    assert torch.isposinf(clipped[0, 2])
    assert torch.isneginf(clipped[0, 3])
    assert torch.isnan(clipped[0, 4])


def test_training_boundary_rejects_non_finite_teacher_action_rollout():
    ppo = object.__new__(PPO)
    ppo.device = "cpu"
    ppo.is_multi_gpu = False
    ppo.storage = SimpleNamespace(
        step=1,
        _buffers={"teacher_actions": torch.tensor([[[0.0, float("inf")]]])},
    )

    with pytest.raises(FloatingPointError, match="filled teacher_actions rollout contains NaN/Inf"):
        ppo._assert_rollout_teacher_actions_finite()


def test_pre_env_rollout_boundary_rejects_non_finite_action():
    ppo = object.__new__(PPO)
    ppo.device = "cpu"
    ppo.is_multi_gpu = False

    with pytest.raises(
        FloatingPointError,
        match=r"before env\.step.*actor_actions.*Refusing to call env\.step",
    ):
        ppo._assert_rollout_tensors_finite(
            {
                "actor_obs": torch.zeros(2, 3),
                "actor_actions": torch.tensor([[0.0], [float("nan")]]),
            },
            phase="iteration 2 rollout step 3/24 before env.step",
        )


def test_pre_env_rollout_boundary_synchronizes_local_runtime_error():
    ppo = object.__new__(PPO)
    ppo.device = "cpu"
    ppo.is_multi_gpu = False

    with pytest.raises(RuntimeError, match=r"selector failed.*before env\.step") as exc_info:
        ppo._assert_rollout_tensors_finite(
            {"actor_obs": torch.zeros(2, 3)},
            phase="iteration 2 rollout step 3/24 before env.step",
            local_error=ValueError("selector failed"),
        )

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_training_boundary_checks_all_floating_rollout_buffers():
    ppo = object.__new__(PPO)
    ppo.device = "cpu"
    ppo.is_multi_gpu = False
    ppo.storage = SimpleNamespace(
        step=1,
        _buffers={
            "teacher_actions": torch.zeros(1, 1, 2),
            "rewards": torch.tensor([[[float("inf")]]]),
        },
    )

    with pytest.raises(FloatingPointError, match=r"filled rollout.*rewards"):
        ppo._assert_rollout_storage_finite()


def test_every_actor_optimizer_step_hard_projects_max_noise_std():
    ppo = object.__new__(PPO)
    ppo.actor = SimpleNamespace(
        std=nn.Parameter(torch.tensor([0.2, 0.3])),
        min_noise_std=0.01,
        min_mean_noise_std=None,
        max_noise_std=0.5,
    )
    ppo.config = SimpleNamespace(init_noise_std=0.2)
    ppo.actor_optimizer = SimpleNamespace(
        step=lambda: ppo.actor.std.data.copy_(torch.tensor([5.0, 0.7]))
    )

    ppo._step_actor_optimizer()

    assert ppo.actor.std.detach().tolist() == pytest.approx([0.5, 0.5])


def test_actor_std_projection_caps_components_before_enforcing_mean_floor():
    ppo = object.__new__(PPO)
    ppo.actor = SimpleNamespace(
        std=nn.Parameter(torch.tensor([2.0, 0.1])),
        min_noise_std=None,
        min_mean_noise_std=0.8,
        max_noise_std=0.8,
    )
    ppo.config = SimpleNamespace(init_noise_std=0.2)

    ppo._sanitize_actor_std()

    assert ppo.actor.std.detach().tolist() == pytest.approx([0.8, 0.8])
    assert ppo.actor.std.detach().mean().item() == pytest.approx(0.8)


def test_adaptive_kl_is_rank_weighted_and_updates_only_actor_learning_rate():
    ppo = object.__new__(PPO)
    ppo.device = "cpu"
    ppo.is_multi_gpu = True
    ppo._get_distributed_loss_weight = MethodType(lambda self: 2.0, ppo)
    ppo._all_reduce_small_tensor = MethodType(
        lambda self, payload, op: payload + payload.new_tensor([3.0, 1.0]),
        ppo,
    )
    old_mu = torch.zeros(2, 1)
    old_sigma = torch.ones(2, 1)
    new_mu = torch.ones(2, 1)
    new_sigma = torch.ones(2, 1)

    weighted_kl = ppo._compute_kl_div(old_mu, old_sigma, new_mu, new_sigma)

    assert weighted_kl.item() == pytest.approx(4.0 / 3.0)

    actor_param = nn.Parameter(torch.zeros(()))
    critic_param = nn.Parameter(torch.zeros(()))
    ppo.actor_optimizer = torch.optim.SGD([actor_param], lr=0.3)
    ppo.critic_optimizer = torch.optim.SGD([critic_param], lr=0.15)
    ppo.actor_learning_rate = 0.3
    ppo.critic_learning_rate = 0.15
    ppo.min_actor_learning_rate = 0.01
    ppo.min_critic_learning_rate = 0.01
    ppo.max_actor_learning_rate = 1.0
    ppo.max_critic_learning_rate = 1.0
    ppo.config = SimpleNamespace(desired_kl=0.1)

    ppo._update_learning_rate(weighted_kl)

    assert ppo.actor_learning_rate == pytest.approx(0.2)
    assert ppo.actor_optimizer.param_groups[0]["lr"] == pytest.approx(0.2)
    assert ppo.critic_learning_rate == pytest.approx(0.15)
    assert ppo.critic_optimizer.param_groups[0]["lr"] == pytest.approx(0.15)


def test_hybrid_kl_controller_cannot_collapse_independent_critic_lr():
    ppo = object.__new__(PPO)
    actor_param = nn.Parameter(torch.zeros(()))
    critic_param = nn.Parameter(torch.zeros(()))
    ppo.actor_optimizer = torch.optim.SGD([actor_param], lr=1.0e-3)
    ppo.critic_optimizer = torch.optim.SGD([critic_param], lr=1.0e-3)
    ppo.actor_learning_rate = 1.0e-3
    ppo.critic_learning_rate = 1.0e-3
    ppo.min_actor_learning_rate = 1.0e-5
    ppo.max_actor_learning_rate = 1.0e-2
    ppo.min_critic_learning_rate = 1.0e-5
    ppo.max_critic_learning_rate = 1.0e-2
    ppo.config = SimpleNamespace(desired_kl=0.01)

    # r20 uses 64 minibatches.  Sustained high actor KL is allowed to bring
    # the actor to its trust-region floor, but it must not alter value learning.
    for _ in range(64):
        ppo._update_learning_rate(torch.tensor(1.0))

    assert ppo.actor_learning_rate == pytest.approx(1.0e-5)
    assert ppo.actor_optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-5)
    assert ppo.critic_learning_rate == pytest.approx(1.0e-3)
    assert ppo.critic_optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)


def test_learning_rate_bounds_preserve_explicit_values():
    assert PPO._resolve_learning_rate_bounds("actor", 1.0e-3, 2.0e-5, 3.0e-3) == pytest.approx(
        (1.0e-3, 2.0e-5, 3.0e-3)
    )


@pytest.mark.parametrize(
    ("initial", "minimum", "maximum", "message"),
    [
        (0.0, None, None, "finite and > 0"),
        (1.0e-3, 0.0, 1.0e-2, "finite and > 0"),
        (1.0e-3, 2.0e-3, 1.0e-2, "minimum <= initial <= maximum"),
        (1.0e-3, 1.0e-5, 5.0e-4, "minimum <= initial <= maximum"),
        (float("nan"), None, None, "finite and > 0"),
    ],
)
def test_learning_rate_bounds_reject_invalid_values(initial, minimum, maximum, message):
    with pytest.raises(ValueError, match=message):
        PPO._resolve_learning_rate_bounds("actor", initial, minimum, maximum)


def _enable_hybrid_adaptive_kl(ppo: PPO, *, ppo_coeff: float) -> Mock:
    ppo.dagger_enabled = True
    ppo.use_ppo_dagger_schedule = True
    ppo.ppo_coeff = ppo_coeff
    ppo.config.desired_kl = 0.01
    ppo.config.schedule = "adaptive"
    update_learning_rate = Mock()
    ppo._update_learning_rate = update_learning_rate
    return update_learning_rate


def test_pure_bc_minibatch_records_kl_without_updating_learning_rate():
    ppo = _loss_stub()
    update_learning_rate = _enable_hybrid_adaptive_kl(ppo, ppo_coeff=0.0)
    minibatch = _minibatch(torch.tensor([[1.0], [2.0]]))
    minibatch["action_mean"] = torch.zeros_like(minibatch["action_mean"])

    losses = ppo._compute_ppo_loss(minibatch)

    assert losses["kl_mean"].item() > 0.0
    update_learning_rate.assert_not_called()


@pytest.mark.parametrize("ppo_coeff", [0.01, 0.1])
def test_any_nonzero_ppo_contribution_defers_adaptive_kl_until_before_optimizer(
    ppo_coeff: float,
):
    ppo = _loss_stub()
    update_learning_rate = _enable_hybrid_adaptive_kl(ppo, ppo_coeff=ppo_coeff)
    minibatch = _minibatch(torch.tensor([[1.0], [2.0]]))
    minibatch["action_mean"] = torch.zeros_like(minibatch["action_mean"])

    losses = ppo._compute_ppo_loss(minibatch)

    assert losses["kl_mean"].item() > 0.0
    assert losses["_reduce_kl_before_optimizer"] is True
    update_learning_rate.assert_not_called()

    reduced_kl = ppo._reduce_kl_after_local_loss(losses["kl_mean"])

    assert torch.equal(reduced_kl, losses["kl_mean"])
    update_learning_rate.assert_called_once_with(losses["kl_mean"])


def test_non_scheduled_pure_bc_keeps_adaptive_kl_diagnostic_only():
    ppo = _loss_stub()
    ppo.dagger_enabled = True
    ppo.use_ppo_dagger_schedule = False
    ppo.bc_loss_coef = 1.0
    ppo.config.desired_kl = 0.01
    ppo.config.schedule = "adaptive"

    assert ppo._should_update_learning_rate_from_kl() is False


def test_pure_bc_rollout_uses_distribution_mean_without_advancing_torch_rng():
    ppo = object.__new__(PPO)
    ppo.device = "cpu"
    ppo.dagger_enabled = True
    ppo.use_ppo_dagger_schedule = False
    ppo.bc_loss_coef = 1.0
    mean = torch.tensor([[0.25, -0.5]])
    ppo.actor = SimpleNamespace(
        perception_time_gru=None,
        action_mean=mean,
        update_distribution_from_policy_state=Mock(),
        act=Mock(side_effect=lambda _state: torch.rand_like(mean)),
    )
    ppo.critic = SimpleNamespace(
        perception_time_gru=None,
        evaluate=Mock(return_value=torch.tensor([[1.0]])),
    )
    actor_obs = torch.tensor([[1.0, 2.0]])
    critic_obs = torch.tensor([[3.0]])

    rng_before = torch.get_rng_state().clone()
    actions, values, actor_hidden, critic_hidden, error = ppo._try_compute_student_rollout_outputs(
        actor_obs=actor_obs,
        critic_obs=critic_obs,
        actor_perception_obs=None,
        critic_perception_obs=None,
        timing=None,
    )

    assert error is None
    assert torch.equal(actions, mean)
    assert torch.equal(values, torch.tensor([[1.0]]))
    assert actor_hidden is None
    assert critic_hidden is None
    assert torch.equal(torch.get_rng_state(), rng_before)
    ppo.actor.update_distribution_from_policy_state.assert_called_once_with(
        {"actor_obs": actor_obs}
    )
    ppo.actor.act.assert_not_called()


@pytest.mark.parametrize("bc_loss_coef", [0.99, 0.0])
def test_non_scheduled_hybrid_or_switched_rl_keeps_adaptive_kl_control(bc_loss_coef: float):
    ppo = _loss_stub()
    ppo.dagger_enabled = True
    ppo.use_ppo_dagger_schedule = False
    ppo.bc_loss_coef = bc_loss_coef
    ppo.config.desired_kl = 0.01
    ppo.config.schedule = "adaptive"

    assert ppo._should_update_learning_rate_from_kl() is True


class _SequencePassthrough(nn.Module):
    def forward_sequence(self, obs_seq, dones_seq, initial_hidden):  # noqa: ARG002
        return obs_seq


def test_pure_bc_gru_minibatch_records_kl_without_updating_learning_rate():
    ppo = _loss_stub()
    ppo.actor_perception_key = "actor_perception"
    ppo.critic_perception_key = "critic_perception"
    ppo.actor.perception_time_gru = _SequencePassthrough()
    ppo.critic.perception_time_gru = _SequencePassthrough()
    update_learning_rate = _enable_hybrid_adaptive_kl(ppo, ppo_coeff=0.0)

    time_steps, batch_size = 2, 2
    actor_obs = torch.tensor([[[1.0], [2.0]], [[1.5], [2.5]]])
    actions = torch.zeros(time_steps, batch_size, 1)
    old_mu = torch.zeros_like(actions)
    old_sigma = torch.ones_like(actions)
    old_log_prob = Normal(old_mu, old_sigma).log_prob(actions).sum(-1, keepdim=True)
    minibatch = {
        "actor_obs": actor_obs,
        "critic_obs": actor_obs + 1.0,
        "actions": actions,
        "values": torch.zeros_like(actions),
        "advantages": torch.ones_like(actions),
        "returns": torch.zeros_like(actions),
        "actions_log_prob": old_log_prob,
        "action_mean": old_mu,
        "action_sigma": old_sigma,
        "actor_perception": torch.zeros(time_steps, batch_size, 1),
        "critic_perception": torch.zeros(time_steps, batch_size, 1),
        "dones": torch.zeros(time_steps, batch_size, 1, dtype=torch.bool),
        "actor_gru_hidden": torch.zeros(time_steps, batch_size, 1),
        "critic_gru_hidden": torch.zeros(time_steps, batch_size, 1),
    }

    losses = ppo._compute_ppo_loss_sequence(minibatch)

    assert losses["kl_mean"].item() > 0.0
    update_learning_rate.assert_not_called()


@pytest.mark.parametrize("selector", [torch.tensor([[0.5]]), torch.tensor([[float("nan")]]), torch.tensor([[2.0]])])
def test_multi_teacher_selector_rejects_fractional_nonfinite_and_out_of_range(selector):
    ppo = object.__new__(PPO)
    ppo.use_multi_teacher = True
    ppo.multi_teacher_select_obs_var = "teacher_index"
    ppo.teacher_use_stochastic_actions = False
    ppo.teacher_actors = [object()]
    ppo.teacher_actor_obs_normalizers_list = [{}]
    ppo.num_act = 1

    with pytest.raises(ValueError, match="Multi-teacher selector"):
        ppo._select_teacher_actions(torch.zeros(1, 1), {"teacher_index": selector})


def test_multi_teacher_rejects_recurrent_teacher_hidden_state_aliasing():
    ppo = object.__new__(PPO)
    ppo.use_multi_teacher = True
    ppo.teacher_actors = [
        SimpleNamespace(perception_time_gru=None, supports_flow_matching=False),
        SimpleNamespace(perception_time_gru=object(), supports_flow_matching=False),
    ]
    ppo.teacher_actor = None
    ppo.distill_mode = "dagger"
    ppo.teacher_use_stochastic_actions = False

    with pytest.raises(ValueError, match="variable sub-batches.*hidden rows"):
        ppo._validate_loaded_teacher_inference_contract()


def test_non_recurrent_multi_teacher_inference_contract_is_supported():
    ppo = object.__new__(PPO)
    ppo.use_multi_teacher = True
    ppo.teacher_actors = [
        SimpleNamespace(perception_time_gru=None, supports_flow_matching=False),
        SimpleNamespace(perception_time_gru=None, supports_flow_matching=False),
    ]
    ppo.teacher_actor = None
    ppo.distill_mode = "dagger"
    ppo.teacher_use_stochastic_actions = False

    ppo._validate_loaded_teacher_inference_contract()


def test_distillation_rejects_unknown_mode_before_setup_side_effects():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(distill=SimpleNamespace(mode="not-a-mode"))

    with pytest.raises(ValueError, match="distill.mode"):
        ppo._setup_distillation()


def test_distillation_rejects_conflicting_teacher_checkpoint_aliases():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            policy_to_clone="teacher-a.pt",
            teacher_checkpoint="teacher-b.pt",
        )
    )

    with pytest.raises(ValueError, match="identify different teacher sources"):
        ppo._setup_distillation()


def test_dagger_mode_requires_explicit_distillation_enable():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=False,
            mode="dagger",
            policy_to_clone="teacher.pt",
        )
    )

    with pytest.raises(ValueError, match="requires distill.enabled=True"):
        ppo._setup_distillation()


def test_enabled_dagger_rejects_silent_noop_pure_ppo_configuration():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=True,
            mode="dagger",
            policy_to_clone="teacher.pt",
            bc_loss_coef=0.0,
            switch_to_rl_after=-1,
            ppo_start_epoch=-1,
            dagger_end_epoch=-1,
        )
    )

    with pytest.raises(ValueError, match="silently ignore the teacher and run pure PPO"):
        ppo._setup_distillation()


def test_enabled_dagger_rejects_bc_weight_that_rounds_to_float32_zero():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=True,
            mode="dagger",
            policy_to_clone="teacher.pt",
            bc_loss_coef=math.nextafter(0.0, 1.0),
        )
    )

    with pytest.raises(ValueError, match="rounds to zero.*float32 actor loss graph"):
        ppo._setup_distillation()


def test_enabled_legacy_mse_rejects_zero_weight_noop_teacher_path():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=True,
            mode="mse",
            teacher_checkpoint="teacher.pt",
            loss_coef=0.0,
        )
    )

    with pytest.raises(ValueError, match="teacher-only observations.*pure PPO"):
        ppo._setup_distillation()


@pytest.mark.parametrize(
    "loss_coef",
    [math.nextafter(0.0, 1.0), 1.0e-50, 1.0e50],
)
def test_enabled_legacy_mse_rejects_nonoperational_float32_weight(loss_coef):
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=True,
            mode="mse",
            teacher_checkpoint="teacher.pt",
            loss_coef=loss_coef,
        )
    )

    with pytest.raises(ValueError, match="finite and positive.*float32 actor loss graph"):
        ppo._setup_distillation()


def test_enabled_legacy_mse_rejects_dagger_only_bc_loss_alias():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=True,
            mode="mse",
            teacher_checkpoint="teacher.pt",
            loss_coef=0.25,
            bc_loss_coef=0.75,
        )
    )

    with pytest.raises(ValueError, match="bc_loss_coef is DAgger-only.*loss_coef only"):
        ppo._setup_distillation()


def test_mse_mode_rejects_huber_loss_mislabelling():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=True,
            mode="mse",
            teacher_checkpoint="teacher.pt",
            distill_loss_type="huber",
        )
    )

    with pytest.raises(ValueError, match="mode='mse'.*distill_loss_type='mse'"):
        ppo._setup_distillation()


def test_dagger_rejects_conflicting_nondefault_loss_coefficient_aliases():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=True,
            mode="dagger",
            policy_to_clone="teacher.pt",
            loss_coef=0.25,
            bc_loss_coef=0.75,
        )
    )

    with pytest.raises(ValueError, match="loss_coef and distill.bc_loss_coef conflict"):
        ppo._setup_distillation()


def test_non_dagger_setup_uses_zero_effective_fixed_bc_budget():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=False,
            mode="mse",
            fixed_bc_eval_num_samples=4096,
        )
    )

    ppo._setup_distillation()

    assert ppo._configured_fixed_bc_eval_num_samples == 4096
    assert ppo.fixed_bc_eval_num_samples == 0


@pytest.mark.parametrize(
    ("policy_to_clone", "selector_key", "obs_dims", "message"),
    [
        ("one.pt", "selector", {"selector": 1}, "requires policy_to_clone to be a real list"),
        (["one.pt"], "selector", {"selector": 1}, "requires at least two teacher checkpoint paths"),
        (["one.pt", "  "], "selector", {"selector": 1}, "entries must be non-empty strings"),
        (["one.pt", "one.pt"], "selector", {"selector": 1}, "duplicate checkpoint paths"),
        (["one.pt", "two.pt"], "missing", {"selector": 1}, "must exist and contain exactly one scalar"),
        (["one.pt", "two.pt"], "selector", {"selector": 2}, "must exist and contain exactly one scalar"),
    ],
)
def test_multi_teacher_requires_distinct_list_and_scalar_selector_before_loading(
    policy_to_clone,
    selector_key,
    obs_dims,
    message,
):
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=True,
            mode="dagger",
            policy_to_clone=policy_to_clone,
            bc_loss_coef=1.0,
            use_multi_teacher=True,
            multi_teacher_select_obs_var=selector_key,
        )
    )
    ppo.algo_obs_dim_dict = obs_dims

    with pytest.raises(ValueError, match=message):
        ppo._setup_distillation()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dagger_loss_coef": 0.0}, "requires distill.dagger_loss_coef > 0"),
        (
            {"dagger_loss_coef": math.nextafter(0.0, 1.0)},
            "float32 actor loss graph",
        ),
        ({"bc_loss_coef": 0.5}, "Scheduled PPO/DAgger ignores distill.bc_loss_coef"),
    ],
)
def test_scheduled_dagger_rejects_empty_or_silently_ignored_bc_configuration(
    overrides,
    message,
):
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=True,
            mode="dagger",
            policy_to_clone="teacher.pt",
            ppo_start_epoch=10,
            dagger_end_epoch=20,
            **overrides,
        )
    )

    with pytest.raises(ValueError, match=message):
        ppo._setup_distillation()


def test_scheduled_dagger_rejects_epoch_zero_unit_start_as_nominal_dagger_pure_ppo():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=True,
            mode="dagger",
            policy_to_clone="teacher.pt",
            ppo_start_epoch=0,
            dagger_end_epoch=20,
            ppo_start_coeff=1.0,
            ppo_target_coeff=1.0,
        )
    )

    with pytest.raises(ValueError, match="nominal DAgger run is actually pure PPO"):
        ppo._setup_distillation()


@pytest.mark.parametrize(
    ("ppo_start_coeff", "ppo_target_coeff", "coefficient_name"),
    [
        (math.nextafter(0.0, 1.0), 0.5, "ppo_start_coeff"),
        (0.0, math.nextafter(0.0, 1.0), "ppo_target_coeff"),
    ],
)
def test_scheduled_dagger_rejects_positive_endpoint_that_underflows_float32(
    ppo_start_coeff,
    ppo_target_coeff,
    coefficient_name,
):
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=True,
            mode="dagger",
            policy_to_clone="teacher.pt",
            ppo_start_epoch=0,
            dagger_end_epoch=20,
            ppo_start_coeff=ppo_start_coeff,
            ppo_target_coeff=ppo_target_coeff,
        )
    )

    with pytest.raises(
        ValueError,
        match=rf"{coefficient_name}.*rounds to zero.*float32 PPO actor loss graph",
    ):
        ppo._setup_distillation()


def test_scheduled_dagger_rejects_first_positive_tier_that_underflows_float32():
    smallest_float32 = float(
        torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)).item()
    )
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=True,
            mode="dagger",
            policy_to_clone="teacher.pt",
            ppo_start_epoch=0,
            dagger_end_epoch=2,
            ppo_start_coeff=0.0,
            ppo_target_coeff=smallest_float32,
        )
    )

    with pytest.raises(ValueError, match="positive Python PPO tier.*rounds to zero"):
        ppo._setup_distillation()


def test_scheduled_dagger_rejects_operational_float32_weight_underflow():
    smallest_float32 = float(
        torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)).item()
    )
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=True,
            mode="dagger",
            policy_to_clone="teacher.pt",
            ppo_start_epoch=0,
            dagger_end_epoch=20,
            ppo_start_coeff=0.5,
            ppo_target_coeff=1.0,
            dagger_loss_coef=smallest_float32,
        )
    )

    with pytest.raises(ValueError, match="no operational float32 BC phase"):
        ppo._setup_distillation()


@pytest.mark.parametrize("ppo_target_coeff", [0.9, 1.0])
def test_scheduled_dagger_rejects_future_positive_bc_tier_that_underflows_float32(
    ppo_target_coeff,
):
    smallest_float32 = float(
        torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)).item()
    )
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=True,
            mode="dagger",
            policy_to_clone="teacher.pt",
            ppo_start_epoch=0,
            dagger_end_epoch=10,
            ppo_start_coeff=0.0,
            ppo_target_coeff=ppo_target_coeff,
            dagger_loss_coef=smallest_float32,
        )
    )

    with pytest.raises(ValueError, match="positive future BC tier.*rounds to zero"):
        ppo._setup_distillation()


@pytest.mark.parametrize(
    ("ppo_start_epoch", "ppo_start_coeff", "expected_initial_coeff"),
    [
        (1, 1.0, 0.0),
        (0, 0.9, 0.9),
    ],
)
def test_scheduled_dagger_accepts_boundary_configurations_with_a_real_bc_phase(
    ppo_start_epoch,
    ppo_start_coeff,
    expected_initial_coeff,
):
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=True,
            mode="dagger",
            policy_to_clone="teacher.pt",
            ppo_start_epoch=ppo_start_epoch,
            dagger_end_epoch=20,
            ppo_start_coeff=ppo_start_coeff,
            ppo_target_coeff=1.0,
        )
    )
    ppo.algo_obs_dim_dict = {"actor_obs": 1}
    ppo.actor_obs_keys = ["actor_obs"]
    ppo.current_learning_iteration = 0
    ppo._load_teacher_actor = Mock(
        return_value=(
            SimpleNamespace(
                perception_time_gru=None,
                supports_flow_matching=False,
            ),
            {},
        )
    )

    ppo._setup_distillation()

    assert ppo.use_ppo_dagger_schedule is True
    assert ppo.ppo_coeff == pytest.approx(expected_initial_coeff)
    assert ppo._effective_dagger_loss_weight() > 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("switch_to_rl_after", 10),
        ("use_multi_teacher", True),
        ("dagger_match_std", True),
        ("dagger_ignore_episode_initial_steps", 1),
    ],
)
def test_legacy_mse_rejects_dagger_only_options(field, value):
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            mode="mse",
            teacher_checkpoint="teacher.pt",
            **{field: value},
        )
    )
    ppo.algo_obs_dim_dict = {"actor_obs": 1}
    ppo.actor_obs_keys = ["actor_obs"]
    ppo.teacher_action_mix_ratio_start = None
    ppo.teacher_action_mix_ratio_end = None

    with pytest.raises(ValueError, match=field):
        ppo._setup_distillation()


def test_legacy_mse_rejects_ppo_dagger_schedule():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            mode="mse",
            teacher_checkpoint="teacher.pt",
            ppo_start_epoch=0,
            dagger_end_epoch=10,
        )
    )
    ppo.algo_obs_dim_dict = {"actor_obs": 1}
    ppo.actor_obs_keys = ["actor_obs"]
    ppo.teacher_action_mix_ratio_start = None
    ppo.teacher_action_mix_ratio_end = None

    with pytest.raises(ValueError, match="ppo_start_epoch/dagger_end_epoch schedule"):
        ppo._setup_distillation()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dagger_loss_coef", 3.0),
        ("ppo_start_coeff", 0.2),
        ("ppo_target_coeff", 0.7),
        ("ppo_start_noise_std_until_coeff", 0.4),
        ("dagger_ignore_zero_teacher_actions", False),
        ("multi_teacher_select_obs_var", "another_selector"),
    ],
)
def test_legacy_mse_rejects_nondefault_dagger_only_noop_options(field, value):
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            mode="mse",
            teacher_checkpoint="teacher.pt",
            **{field: value},
        )
    )
    ppo.algo_obs_dim_dict = {"actor_obs": 1}
    ppo.actor_obs_keys = ["actor_obs"]

    with pytest.raises(ValueError, match=field):
        ppo._setup_distillation()


def test_legacy_mse_rejects_even_zero_to_zero_teacher_mix_schedule():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            mode="mse",
            teacher_checkpoint="teacher.pt",
            teacher_action_mix_ratio_start=0.0,
            teacher_action_mix_ratio_end=0.0,
            teacher_action_mix_ratio_end_iteration=10,
        )
    )
    ppo.algo_obs_dim_dict = {"actor_obs": 1}
    ppo.actor_obs_keys = ["actor_obs"]

    with pytest.raises(ValueError, match="schedule start and end must differ"):
        ppo._setup_distillation()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dagger_loss_coef", 3.0),
        ("ppo_start_coeff", 0.2),
        ("ppo_target_coeff", 0.7),
        ("ppo_start_noise_std_until_coeff", 0.4),
    ],
)
def test_unscheduled_dagger_rejects_nondefault_schedule_only_options(field, value):
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=True,
            mode="dagger",
            policy_to_clone="teacher.pt",
            bc_loss_coef=1.0,
            **{field: value},
        )
    )

    with pytest.raises(ValueError, match=field):
        ppo._setup_distillation()


def test_scheduled_dagger_rejects_noise_threshold_without_noise_cap():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=True,
            mode="dagger",
            policy_to_clone="teacher.pt",
            bc_loss_coef=1.0,
            ppo_start_epoch=10,
            dagger_end_epoch=20,
            ppo_start_noise_std=None,
            ppo_start_noise_std_until_coeff=0.4,
        )
    )

    with pytest.raises(ValueError, match="only consumed when.*ppo_start_noise_std"):
        ppo._setup_distillation()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "teacher_action_mix_ratio_start": 0.5,
                "teacher_action_mix_ratio_end": 0.5,
                "teacher_action_mix_ratio_end_iteration": 10,
            },
            "schedule start and end must differ",
        ),
        (
            {
                "teacher_action_mix_ratio_start": 0.5,
                "teacher_action_mix_ratio_end": 0.0,
                "teacher_action_mix_ratio_end_iteration": 0,
            },
            "end_iteration must be > 0",
        ),
        (
            {
                "ppo_start_epoch": 10,
                "dagger_end_epoch": 20,
                "ppo_start_coeff": 0.4,
                "ppo_start_noise_std": 0.1,
                "ppo_start_noise_std_until_coeff": 0.2,
                "ppo_schedule_step_epochs": 0,
            },
            "noise cap would never apply",
        ),
        (
            {"clip_teacher_actions": False, "clip_actions_threshold": 8.0},
            "clip_actions_threshold is only consumed",
        ),
        (
            {"fixed_bc_eval_num_samples": 0, "fixed_bc_eval_log_interval": 100},
            "fixed_bc_eval_log_interval is only consumed",
        ),
        (
            {"multi_teacher_select_obs_var": "unused_selector"},
            "multi_teacher_select_obs_var is only consumed",
        ),
    ],
)
def test_dagger_rejects_additional_silent_noop_configuration(overrides, message):
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=True,
            mode="dagger",
            policy_to_clone="teacher.pt",
            bc_loss_coef=1.0,
            **overrides,
        )
    )

    with pytest.raises(ValueError, match=message):
        ppo._setup_distillation()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("loss_coef", float("nan")),
        ("bc_loss_coef", float("inf")),
        ("clip_actions_threshold", float("-inf")),
        ("teacher_action_mix_ratio", float("nan")),
        ("ppo_target_coeff", float("inf")),
        ("ppo_start_coeff", float("nan")),
        ("ppo_start_noise_std", float("inf")),
        ("ppo_start_noise_std_until_coeff", float("nan")),
        ("dagger_loss_coef", float("-inf")),
    ],
)
def test_distillation_rejects_nonfinite_numerical_configuration(field, value):
    config_values = {field: value}
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(distill=DistillationConfig(**config_values))

    with pytest.raises(ValueError, match=field):
        ppo._setup_distillation()


@pytest.mark.parametrize(
    ("config_values", "message"),
    [
        (
            {"teacher_action_mix_ratio_end_iteration": 10},
            "teacher_action_mix_ratio_end_iteration is only valid",
        ),
        (
            {"ppo_start_noise_std": 0.1},
            "ppo_start_noise_std requires an enabled",
        ),
        (
            {"ppo_schedule_step_epochs": 10},
            "ppo_schedule_step_epochs requires an enabled",
        ),
    ],
)
def test_distillation_rejects_schedule_knobs_that_would_be_ignored(config_values, message):
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(distill=DistillationConfig(**config_values))

    with pytest.raises(ValueError, match=message):
        ppo._setup_distillation()


@pytest.mark.parametrize(
    ("config_values", "message"),
    [
        (
            {"dagger_ignore_episode_initial_steps": -1},
            "dagger_ignore_episode_initial_steps must be >= 0",
        ),
        (
            {"fixed_bc_eval_num_samples": -1},
            "fixed_bc_eval_num_samples must be >= 0",
        ),
        (
            {"fixed_bc_eval_log_interval": 0},
            "fixed_bc_eval_log_interval must be > 0",
        ),
        (
            {"switch_to_rl_after": -2},
            "switch_to_rl_after must be -1/0",
        ),
        (
            {
                "enabled": True,
                "mode": "dagger",
                "policy_to_clone": "teacher.pt",
                "bc_loss_coef": 0.0,
                "switch_to_rl_after": 10,
            },
            "requires a positive pre-switch bc_loss_coef",
        ),
    ],
)
def test_distillation_rejects_invalid_diagnostic_and_mask_counts(config_values, message):
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(distill=DistillationConfig(**config_values))
    ppo.teacher_action_mix_ratio_start = None
    ppo.teacher_action_mix_ratio_end = None

    with pytest.raises(ValueError, match=message):
        ppo._setup_distillation()


def _dagger_initial_mask_setup_stub(
    *,
    ignore_steps: int,
    episode_horizon=4,
    include_env: bool = True,
    save_interval=10,
    num_steps_per_env=4,
) -> PPO:
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        save_interval=save_interval,
        num_steps_per_env=num_steps_per_env,
        init_at_random_ep_len=False,
        distill=DistillationConfig(
            enabled=True,
            mode="dagger",
            policy_to_clone="teacher.pt",
            bc_loss_coef=1.0,
            dagger_ignore_episode_initial_steps=ignore_steps,
        )
    )
    if include_env:
        ppo.env = SimpleNamespace(max_episode_length=episode_horizon)
    ppo.algo_obs_dim_dict = {"actor_obs": 1}
    ppo.actor_obs_keys = ["actor_obs"]
    ppo.current_learning_iteration = 0
    ppo._load_teacher_actor = Mock(return_value=(object(), {}))
    ppo._validate_loaded_teacher_inference_contract = Mock()
    return ppo


@pytest.mark.parametrize(
    "episode_horizon",
    [None, True, 0, -1, 3.5, float("nan"), float("inf"), "4"],
)
def test_dagger_initial_step_mask_requires_finite_positive_integer_horizon(
    episode_horizon,
):
    ppo = _dagger_initial_mask_setup_stub(
        ignore_steps=1,
        episode_horizon=episode_horizon,
    )

    with pytest.raises(ValueError, match="finite positive integer-equivalent horizon"):
        ppo._setup_distillation()


def test_dagger_initial_step_mask_requires_environment_horizon_when_enabled():
    ppo = _dagger_initial_mask_setup_stub(ignore_steps=1, include_env=False)

    with pytest.raises(ValueError, match="env.max_episode_length"):
        ppo._setup_distillation()


@pytest.mark.parametrize("ignore_steps", [4, 5])
def test_dagger_initial_step_mask_rejects_permanently_empty_episode_horizon(
    ignore_steps,
):
    ppo = _dagger_initial_mask_setup_stub(
        ignore_steps=ignore_steps,
        episode_horizon=4,
    )

    with pytest.raises(
        ValueError,
        match="permanently empty the BC mask.*freeze the student actor",
    ):
        ppo._setup_distillation()


def test_dagger_initial_step_mask_accepts_horizon_minus_one():
    ppo = _dagger_initial_mask_setup_stub(
        ignore_steps=3,
        episode_horizon=np.float64(4.0),
    )

    ppo._setup_distillation()

    assert ppo.dagger_ignore_episode_initial_steps == 3
    assert ppo.dagger_enabled is True


def test_zero_dagger_initial_step_mask_does_not_require_environment_horizon():
    ppo = _dagger_initial_mask_setup_stub(ignore_steps=0, include_env=False)

    ppo._setup_distillation()

    assert ppo.dagger_ignore_episode_initial_steps == 0
    assert ppo.dagger_enabled is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("save_interval", None),
        ("save_interval", True),
        ("save_interval", 0),
        ("save_interval", -1),
        ("save_interval", 1.0),
        ("num_steps_per_env", None),
        ("num_steps_per_env", True),
        ("num_steps_per_env", 0),
        ("num_steps_per_env", -1),
        ("num_steps_per_env", 4.0),
    ],
)
def test_dagger_initial_step_mask_requires_positive_integer_rollout_counts(
    field,
    value,
):
    kwargs = {field: value}
    ppo = _dagger_initial_mask_setup_stub(
        ignore_steps=1,
        episode_horizon=100,
        **kwargs,
    )

    with pytest.raises(ValueError, match=rf"PPO {field} must be a positive integer"):
        ppo._setup_distillation()


@pytest.mark.parametrize("ignore_steps", [8, 9])
def test_dagger_initial_step_mask_rejects_checkpoint_block_capacity(
    ignore_steps,
):
    ppo = _dagger_initial_mask_setup_stub(
        ignore_steps=ignore_steps,
        episode_horizon=100,
        save_interval=2,
        num_steps_per_env=4,
    )

    with pytest.raises(ValueError, match="one canonical checkpoint block"):
        ppo._setup_distillation()


def test_dagger_initial_step_mask_accepts_checkpoint_capacity_minus_one():
    ppo = _dagger_initial_mask_setup_stub(
        ignore_steps=7,
        episode_horizon=100,
        save_interval=2,
        num_steps_per_env=4,
    )

    ppo._setup_distillation()

    assert ppo.dagger_ignore_episode_initial_steps == 7
    assert ppo.dagger_enabled is True


@pytest.mark.parametrize(
    ("episode_horizon", "block_transitions", "random_length", "expected"),
    [
        (4, 1, False, 1),
        (4, 3, False, 3),
        (4, 4, False, 3),
        (4, 6, False, 3),
        (4, 7, False, 4),
        (4, 4, True, 4),
        (1, 1, False, 1),
    ],
)
def test_canonical_rollout_episode_age_capacity_accounts_for_dummy_step(
    episode_horizon,
    block_transitions,
    random_length,
    expected,
):
    assert (
        PPO._canonical_rollout_episode_age_capacity(
            episode_horizon,
            block_transitions,
            init_at_random_ep_len=random_length,
        )
        == expected
    )


def test_dagger_initial_step_mask_rejects_dummy_step_checkpoint_cycle():
    ppo = _dagger_initial_mask_setup_stub(
        ignore_steps=1,
        episode_horizon=2,
        save_interval=1,
        num_steps_per_env=2,
    )

    with pytest.raises(ValueError, match=r"reset_all\(\)'s dummy transition"):
        ppo._setup_distillation()


def _runtime_prepend_motion_command(
    *,
    prepend_steps: int,
    start_probability: float,
    end_probability: float | None = None,
    schedule_start: int | None = None,
    schedule_end: int | None = None,
):
    motion_cfg = SimpleNamespace(
        start_at_timestep_zero_prob=start_probability,
        start_at_timestep_zero_prob_end=end_probability,
        start_at_timestep_zero_prob_start_iter=schedule_start,
        start_at_timestep_zero_prob_end_iter=schedule_end,
    )
    return SimpleNamespace(
        _runtime_default_pose_prepend_enabled=True,
        _runtime_default_pose_prepend_steps=prepend_steps,
        motion_cfg=motion_cfg,
    )


def _future_dagger_mask_validation_stub(
    *,
    current_iteration: int = 0,
    end_iteration: int = 20,
    ignore_steps: int = 0,
    episode_horizon: int = 100,
    save_interval: int = 10,
    num_steps_per_env: int = 4,
    switch_to_rl_after: int = -1,
    motion_command=None,
) -> PPO:
    ppo = object.__new__(PPO)
    ppo.current_learning_iteration = current_iteration
    ppo.distill_mode = "dagger"
    ppo.dagger_enabled = True
    ppo.use_ppo_dagger_schedule = False
    ppo._configured_bc_loss_coef = 1.0
    ppo.bc_loss_coef = 1.0
    ppo.switch_to_rl_after = switch_to_rl_after
    ppo.dagger_ignore_episode_initial_steps = ignore_steps
    ppo.config = SimpleNamespace(
        num_learning_iterations=end_iteration,
        save_interval=save_interval,
        num_steps_per_env=num_steps_per_env,
        init_at_random_ep_len=False,
    )
    command_manager = None
    if motion_command is not None:
        command_manager = SimpleNamespace(
            get_state=Mock(return_value=motion_command),
        )
    ppo.env = SimpleNamespace(
        max_episode_length=episode_horizon,
        command_manager=command_manager,
    )
    return ppo


@pytest.mark.parametrize(
    (
        "dagger_loss_coef",
        "ppo_start_coeff",
        "ppo_target_coeff",
        "run_start",
        "run_end",
        "expected",
    ),
    [
        (1.0, 0.0, 1.0, 3, 12, (3, 10)),
        (1.0, 1.0, 1.0, 0, 12, (0, 5)),
        (1.0, 0.0, 0.9, 3, 12, (3, 12)),
        (1.0, 0.0, 1.0, 10, 12, None),
        (0.0, 0.0, 0.9, 3, 12, None),
    ],
)
def test_future_dagger_bc_interval_matches_scheduled_coefficient_support(
    dagger_loss_coef,
    ppo_start_coeff,
    ppo_target_coeff,
    run_start,
    run_end,
    expected,
):
    ppo = _future_dagger_mask_validation_stub(
        current_iteration=run_start,
        end_iteration=run_end,
    )
    ppo.use_ppo_dagger_schedule = True
    ppo.dagger_loss_coef = dagger_loss_coef
    ppo.ppo_start_epoch = 5
    ppo.dagger_end_epoch = 10
    ppo.ppo_start_coeff = ppo_start_coeff
    ppo.ppo_target_coeff = ppo_target_coeff
    ppo.ppo_schedule_step_epochs = 0

    assert ppo._future_dagger_bc_positive_interval(run_start, run_end) == expected


def test_future_dagger_bc_interval_uses_actual_ieee_weight_cutoff():
    ppo = _future_dagger_mask_validation_stub(
        current_iteration=0,
        end_iteration=2,
        ignore_steps=1,
        episode_horizon=100,
        save_interval=10,
        num_steps_per_env=1,
    )
    ppo.use_ppo_dagger_schedule = True
    ppo.dagger_loss_coef = 1.0
    ppo.ppo_start_epoch = 0
    ppo.dagger_end_epoch = 2
    ppo.ppo_start_coeff = math.nextafter(1.0, 0.0)
    ppo.ppo_target_coeff = 1.0
    ppo.ppo_schedule_step_epochs = 0

    assert ppo._compute_ppo_dagger_coeff_for_epoch(1) == 1.0
    assert ppo._future_dagger_bc_positive_interval(0, 2) == (0, 1)
    with pytest.raises(ValueError, match=r"iterations=\[0, 1\)"):
        ppo._validate_future_dagger_bc_mask_signal(
            start_iteration=0,
            end_iteration=2,
        )


@pytest.mark.parametrize(
    ("current_iteration", "end_iteration"),
    [(0, 1), (9, 10)],
)
def test_future_dagger_mask_rejects_fresh_or_resumed_one_iteration_run(
    current_iteration,
    end_iteration,
):
    ppo = _future_dagger_mask_validation_stub(
        current_iteration=current_iteration,
        end_iteration=end_iteration,
        ignore_steps=4,
        episode_horizon=100,
        save_interval=10,
        num_steps_per_env=4,
    )

    with pytest.raises(
        ValueError,
        match="remaining DAgger BC interval cannot produce any valid sample",
    ):
        ppo._validate_future_dagger_bc_mask_signal(
            start_iteration=current_iteration,
            end_iteration=end_iteration,
        )


def test_future_dagger_mask_uses_longest_post_resume_checkpoint_block():
    ppo = _future_dagger_mask_validation_stub(
        current_iteration=9,
        end_iteration=21,
        ignore_steps=39,
        episode_horizon=100,
        save_interval=10,
        num_steps_per_env=4,
    )

    ppo._validate_future_dagger_bc_mask_signal(
        start_iteration=9,
        end_iteration=21,
    )


def test_future_dagger_mask_rejects_dummy_step_resume_tail_cycle():
    ppo = _future_dagger_mask_validation_stub(
        current_iteration=0,
        end_iteration=1,
        ignore_steps=1,
        episode_horizon=2,
        save_interval=1,
        num_steps_per_env=2,
    )

    with pytest.raises(ValueError, match="largest possible episode age capacity is 1"):
        ppo._validate_future_dagger_bc_mask_signal(
            start_iteration=0,
            end_iteration=1,
        )


def test_maximum_canonical_rollout_block_is_exact_across_boundaries():
    assert PPO._maximum_canonical_rollout_block_iterations(9, 10, 10) == 1
    assert PPO._maximum_canonical_rollout_block_iterations(9, 21, 10) == 10
    assert PPO._maximum_canonical_rollout_block_iterations(10, 15, 10) == 5
    assert (
        PPO._maximum_canonical_rollout_block_iterations(0, 1_000_001, 1)
        == 1
    )


def test_future_dagger_mask_uses_bc_cutoff_not_nominal_run_end():
    ppo = _future_dagger_mask_validation_stub(
        current_iteration=0,
        end_iteration=20,
        ignore_steps=4,
        episode_horizon=100,
        save_interval=10,
        num_steps_per_env=4,
        switch_to_rl_after=1,
    )

    with pytest.raises(ValueError, match=r"iterations=\[0, 1\)"):
        ppo._validate_future_dagger_bc_mask_signal(
            start_iteration=0,
            end_iteration=20,
        )


def test_future_dagger_mask_does_not_report_after_bc_is_already_zero():
    motion_command = _runtime_prepend_motion_command(
        prepend_steps=5,
        start_probability=1.0,
    )
    ppo = _future_dagger_mask_validation_stub(
        current_iteration=10,
        end_iteration=12,
        ignore_steps=100,
        episode_horizon=4,
        save_interval=1,
        num_steps_per_env=1,
        switch_to_rl_after=10,
        motion_command=motion_command,
    )

    ppo._validate_future_dagger_bc_mask_signal(
        start_iteration=10,
        end_iteration=12,
    )


def test_learn_rejects_empty_future_bc_mask_before_entering_train_mode():
    ppo = _future_dagger_mask_validation_stub(
        current_iteration=0,
        end_iteration=1,
        ignore_steps=4,
        num_steps_per_env=4,
    )
    ppo._train_mode = Mock()

    with pytest.raises(ValueError, match="remaining DAgger BC interval"):
        ppo.learn()

    ppo._train_mode.assert_not_called()


@pytest.mark.parametrize(
    (
        "start_probability",
        "end_probability",
        "schedule_start",
        "schedule_end",
        "run_start",
        "run_end",
        "expected",
    ),
    [
        (1.0, None, None, None, 3, 8, [(3, 8)]),
        (1.0, 0.0, 5, 10, 3, 8, [(3, 6)]),
        (0.0, 1.0, 5, 10, 8, 12, [(10, 12)]),
        (1.0, 0.0, 5, 5, 4, 7, [(4, 5)]),
        (0.0, 1.0, 5, 5, 4, 7, [(5, 7)]),
        (0.5, None, None, None, 3, 8, []),
    ],
)
def test_probability_one_intervals_match_wbt_curriculum_boundaries(
    start_probability,
    end_probability,
    schedule_start,
    schedule_end,
    run_start,
    run_end,
    expected,
):
    motion_command = _runtime_prepend_motion_command(
        prepend_steps=5,
        start_probability=start_probability,
        end_probability=end_probability,
        schedule_start=schedule_start,
        schedule_end=schedule_end,
    )

    assert PPO._start_at_zero_probability_one_intervals(
        motion_command.motion_cfg,
        run_start,
        run_end,
    ) == expected


@pytest.mark.parametrize(
    ("start_probability", "end_probability", "expected"),
    [
        (0.99999998, None, [(0, 11)]),
        (0.9999999, 1.0, [(8, 11)]),
        (1.0, 0.9999999, [(0, 3)]),
    ],
)
def test_probability_one_intervals_match_runtime_float32_rounding(
    start_probability,
    end_probability,
    expected,
):
    scheduled = end_probability is not None
    motion_command = _runtime_prepend_motion_command(
        prepend_steps=5,
        start_probability=start_probability,
        end_probability=end_probability,
        schedule_start=0 if scheduled else None,
        schedule_end=10 if scheduled else None,
    )

    assert PPO._start_at_zero_probability_one_intervals(
        motion_command.motion_cfg,
        0,
        11,
    ) == expected


@pytest.mark.parametrize(
    ("motion_command", "run_start", "run_end", "first_conflict"),
    [
        (
            _runtime_prepend_motion_command(
                prepend_steps=101,
                start_probability=1.0,
            ),
            7,
            9,
            7,
        ),
        (
            _runtime_prepend_motion_command(
                prepend_steps=101,
                start_probability=1.0,
                end_probability=0.0,
                schedule_start=5,
                schedule_end=10,
            ),
            5,
            8,
            5,
        ),
        (
            _runtime_prepend_motion_command(
                prepend_steps=101,
                start_probability=0.0,
                end_probability=1.0,
                schedule_start=5,
                schedule_end=10,
            ),
            8,
            12,
            10,
        ),
    ],
)
def test_runtime_prepend_reports_first_probability_one_bc_iteration(
    motion_command,
    run_start,
    run_end,
    first_conflict,
):
    ppo = _future_dagger_mask_validation_stub(
        current_iteration=run_start,
        end_iteration=run_end,
        episode_horizon=100,
        motion_command=motion_command,
    )

    with pytest.raises(
        ValueError,
        match=rf"resets at iteration {first_conflict}",
    ):
        ppo._validate_future_dagger_bc_mask_signal(
            start_iteration=run_start,
            end_iteration=run_end,
        )


def test_runtime_prepend_equal_to_horizon_rejects_short_checkpoint_blocks():
    motion_command = _runtime_prepend_motion_command(
        prepend_steps=100,
        start_probability=1.0,
    )
    ppo = _future_dagger_mask_validation_stub(
        episode_horizon=100,
        motion_command=motion_command,
    )

    with pytest.raises(ValueError, match="first valid episode age=99"):
        ppo._validate_future_dagger_bc_mask_signal(
            start_iteration=0,
            end_iteration=20,
        )


def test_runtime_prepend_equal_to_horizon_accepts_a_long_enough_block():
    motion_command = _runtime_prepend_motion_command(
        prepend_steps=100,
        start_probability=1.0,
    )
    ppo = _future_dagger_mask_validation_stub(
        end_iteration=50,
        episode_horizon=100,
        save_interval=50,
        num_steps_per_env=4,
        motion_command=motion_command,
    )

    ppo._validate_future_dagger_bc_mask_signal(
        start_iteration=0,
        end_iteration=50,
    )


def test_runtime_prepend_below_horizon_still_rejects_a_too_short_block():
    motion_command = _runtime_prepend_motion_command(
        prepend_steps=50,
        start_probability=1.0,
    )
    ppo = _future_dagger_mask_validation_stub(
        end_iteration=10,
        episode_horizon=100,
        save_interval=10,
        num_steps_per_env=4,
        motion_command=motion_command,
    )

    with pytest.raises(ValueError, match="first valid episode age=49"):
        ppo._validate_future_dagger_bc_mask_signal(
            start_iteration=0,
            end_iteration=10,
        )


def test_runtime_prepend_is_legal_when_bc_ends_before_probability_one():
    motion_command = _runtime_prepend_motion_command(
        prepend_steps=101,
        start_probability=0.2,
        end_probability=1.0,
        schedule_start=5,
        schedule_end=10,
    )
    ppo = _future_dagger_mask_validation_stub(
        end_iteration=20,
        episode_horizon=100,
        switch_to_rl_after=10,
        motion_command=motion_command,
    )

    ppo._validate_future_dagger_bc_mask_signal(
        start_iteration=0,
        end_iteration=20,
    )


def test_runtime_prepend_rejects_first_iteration_of_rising_one_plateau():
    motion_command = _runtime_prepend_motion_command(
        prepend_steps=101,
        start_probability=0.2,
        end_probability=1.0,
        schedule_start=5,
        schedule_end=10,
    )
    ppo = _future_dagger_mask_validation_stub(
        end_iteration=20,
        episode_horizon=100,
        switch_to_rl_after=11,
        motion_command=motion_command,
    )

    with pytest.raises(ValueError, match="resets at iteration 10"):
        ppo._validate_future_dagger_bc_mask_signal(
            start_iteration=0,
            end_iteration=20,
        )


def test_runtime_prepend_degenerate_decline_excludes_jump_iteration():
    motion_command = _runtime_prepend_motion_command(
        prepend_steps=101,
        start_probability=1.0,
        end_probability=0.0,
        schedule_start=5,
        schedule_end=5,
    )
    ppo = _future_dagger_mask_validation_stub(
        current_iteration=5,
        end_iteration=6,
        episode_horizon=100,
        motion_command=motion_command,
    )

    ppo._validate_future_dagger_bc_mask_signal(
        start_iteration=5,
        end_iteration=6,
    )


def test_supervised_only_rejects_ppo_schedule_conflict(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", "1")
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", "1")
    ppo = object.__new__(PPO)
    ppo.use_symmetry = False
    ppo.use_time_gru = False
    ppo.actor_perception_key = ""
    ppo.critic_perception_key = ""
    ppo.dagger_enabled = True
    ppo.use_ppo_dagger_schedule = True
    ppo.ppo_start_coeff = 0.0
    ppo.ppo_target_coeff = 0.1
    ppo.switch_to_rl_after = -1
    ppo._configured_bc_loss_coef = 1.0
    ppo.dagger_match_std = False

    with pytest.raises(ValueError, match="schedule with an operational PPO contribution"):
        ppo._validate_training_objective_configuration()


def test_supervised_only_rejects_silently_ignored_symmetry_objective(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", "1")
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", "1")
    ppo = object.__new__(PPO)
    ppo.use_symmetry = True
    ppo.actor_perception_key = ""
    ppo.critic_perception_key = ""
    ppo.use_time_gru = False
    ppo.dagger_enabled = True
    ppo.use_ppo_dagger_schedule = False
    ppo.switch_to_rl_after = -1
    ppo._configured_bc_loss_coef = 1.0
    ppo.dagger_match_std = False

    with pytest.raises(ValueError, match="use_symmetry.*not implemented"):
        ppo._validate_training_objective_configuration()


def test_supervised_only_rejects_even_tiny_positive_ppo_weight(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", "1")
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", "1")
    ppo = object.__new__(PPO)
    ppo.use_symmetry = False
    ppo.use_time_gru = False
    ppo.actor_perception_key = ""
    ppo.critic_perception_key = ""
    ppo.dagger_enabled = True
    ppo.use_ppo_dagger_schedule = False
    ppo.switch_to_rl_after = -1
    ppo._configured_bc_loss_coef = 0.999999995
    ppo.dagger_match_std = False

    with pytest.raises(ValueError, match="bc_loss_coef=0.999999995"):
        ppo._validate_training_objective_configuration()


def _batch_norm_objective_validation_stub(*, permanently_pure_bc: bool) -> PPO:
    ppo = object.__new__(PPO)
    ppo.use_symmetry = False
    ppo.use_time_gru = False
    ppo.actor_perception_key = ""
    ppo.critic_perception_key = ""
    ppo.actor = nn.Sequential(nn.BatchNorm1d(2))
    ppo.dagger_enabled = permanently_pure_bc
    ppo.use_ppo_dagger_schedule = False
    ppo._configured_bc_loss_coef = 1.0
    ppo.bc_loss_coef = 1.0
    ppo.switch_to_rl_after = -1
    ppo.dagger_match_std = False
    ppo.config = SimpleNamespace(value_loss_coef=1.0, symmetry_critic_coef=0.0)
    return ppo


def test_supervised_only_accepts_explicit_zero_ppo_schedule(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", "1")
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", "1")
    ppo = _batch_norm_objective_validation_stub(permanently_pure_bc=True)
    ppo.use_ppo_dagger_schedule = True
    ppo.ppo_start_coeff = 0.0
    ppo.ppo_target_coeff = 0.0

    ppo._validate_training_objective_configuration()


def test_actor_batch_norm_is_rejected_when_ppo_can_contribute(monkeypatch):
    monkeypatch.delenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", raising=False)
    monkeypatch.delenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", raising=False)
    ppo = _batch_norm_objective_validation_stub(permanently_pure_bc=False)

    with pytest.raises(ValueError, match=r"Actor BatchNorm.*Unsafe modules: 0"):
        ppo._validate_training_objective_configuration()


def test_ppo_contribution_requires_a_return_trained_critic(monkeypatch):
    monkeypatch.delenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", raising=False)
    monkeypatch.delenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", raising=False)
    ppo = _batch_norm_objective_validation_stub(permanently_pure_bc=False)
    ppo.actor = nn.Linear(2, 2)
    ppo.config.value_loss_coef = 0.0

    with pytest.raises(ValueError, match="PPO can contribute.*value_loss_coef must be > 0"):
        ppo._validate_training_objective_configuration()


def test_actor_batch_norm_is_allowed_for_permanently_pure_bc(monkeypatch):
    monkeypatch.delenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", raising=False)
    monkeypatch.delenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", raising=False)
    ppo = _batch_norm_objective_validation_stub(permanently_pure_bc=True)

    ppo._validate_training_objective_configuration()


def test_multi_gpu_pure_bc_rejects_rank_local_batch_norm_buffers(monkeypatch):
    monkeypatch.delenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", raising=False)
    monkeypatch.delenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", raising=False)
    ppo = _batch_norm_objective_validation_stub(permanently_pure_bc=True)
    ppo.is_multi_gpu = True

    with pytest.raises(ValueError, match="multi-GPU.*running buffers.*rank-zero student"):
        ppo._validate_training_objective_configuration()


def test_standalone_supervised_actor_only_step_is_rejected(monkeypatch):
    monkeypatch.delenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", raising=False)
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", "1")
    ppo = object.__new__(PPO)
    ppo.use_symmetry = False
    ppo.use_time_gru = False
    ppo.actor_perception_key = ""
    ppo.critic_perception_key = ""

    with pytest.raises(ValueError, match="silently freeze the critic"):
        ppo._validate_training_objective_configuration()


def test_supervised_only_rejects_false_actor_only_provenance(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", "1")
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", "0")
    ppo = object.__new__(PPO)
    ppo.use_symmetry = False
    ppo.use_time_gru = False
    ppo.actor_perception_key = ""
    ppo.critic_perception_key = ""

    with pytest.raises(ValueError, match="requires.*ACTOR_ONLY_STEP=1"):
        ppo._validate_training_objective_configuration()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD", "1"),
        ("HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH", "8"),
    ],
)
def test_supervised_only_tuning_knobs_are_effectively_neutral_when_mode_is_disabled(
    monkeypatch,
    name,
    value,
):
    monkeypatch.delenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", raising=False)
    monkeypatch.delenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", raising=False)
    monkeypatch.delenv("HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD", raising=False)
    monkeypatch.delenv("HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH", raising=False)
    monkeypatch.setenv(name, value)
    ppo = _batch_norm_objective_validation_stub(permanently_pure_bc=True)

    ppo._validate_training_objective_configuration()

    assert ppo._supervised_actor_stream_backward_enabled() is False
    assert ppo._supervised_actor_microbatch_size_value() == 0
    assert ppo._configured_supervised_actor_stream_backward is (name.endswith("STREAM_BACKWARD"))
    assert ppo._configured_supervised_actor_microbatch_size == (
        8 if name.endswith("MICROBATCH") else 0
    )


def test_supervised_only_runtime_knobs_are_strict_and_frozen_at_setup(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", "1")
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", "1")
    monkeypatch.setenv("HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD", "1")
    monkeypatch.setenv("HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH", "8")
    ppo = _batch_norm_objective_validation_stub(permanently_pure_bc=True)

    ppo._validate_training_objective_configuration()
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", "0")
    monkeypatch.setenv("HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD", "0")
    monkeypatch.setenv("HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH", "64")

    assert ppo._supervised_dagger_only_enabled() is True
    assert ppo._supervised_actor_only_step_enabled() is True
    assert ppo._supervised_actor_stream_backward_enabled() is True
    assert ppo._supervised_actor_microbatch_size_value() == 8


def _supervised_flow_objective_validation_stub() -> PPO:
    ppo = _batch_norm_objective_validation_stub(permanently_pure_bc=True)
    ppo.actor = nn.Linear(2, 2)
    ppo.actor.supports_flow_matching = True
    return ppo


def test_supervised_only_flow_rejects_silently_ignored_microbatch(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", "1")
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", "1")
    monkeypatch.setenv("HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD", "0")
    monkeypatch.setenv("HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH", "8")
    ppo = _supervised_flow_objective_validation_stub()

    with pytest.raises(
        ValueError,
        match=r"Supervised-only Flow actor microbatch/stream-backward training is not implemented",
    ):
        ppo._validate_training_objective_configuration()


def test_supervised_only_flow_rejects_silently_ignored_stream_backward(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", "1")
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", "1")
    monkeypatch.setenv("HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD", "1")
    monkeypatch.setenv("HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH", "8")
    ppo = _supervised_flow_objective_validation_stub()

    with pytest.raises(
        ValueError,
        match=r"Supervised-only Flow actor microbatch/stream-backward training is not implemented",
    ):
        ppo._validate_training_objective_configuration()


def test_supervised_only_flow_default_whole_batch_contract_remains_supported(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", "1")
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", "1")
    monkeypatch.delenv("HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD", raising=False)
    monkeypatch.delenv("HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH", raising=False)
    ppo = _supervised_flow_objective_validation_stub()

    ppo._validate_training_objective_configuration()

    assert ppo._supervised_actor_stream_backward_enabled() is False
    assert ppo._supervised_actor_microbatch_size_value() == 0


@pytest.mark.parametrize("value", ["-1", "1.5", "true", "８"])
def test_supervised_actor_microbatch_rejects_non_decimal_counts(monkeypatch, value):
    monkeypatch.setenv("HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH", value)

    with pytest.raises(ValueError, match="base-10 non-negative integer"):
        PPO._strict_environment_nonnegative_int(
            "HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH",
            default=0,
        )


def test_time_gru_distillation_is_rejected_before_rollout(monkeypatch):
    monkeypatch.delenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", raising=False)
    monkeypatch.delenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", raising=False)
    ppo = object.__new__(PPO)
    ppo.use_symmetry = False
    ppo.use_time_gru = True
    ppo.actor_perception_key = "perception_obs"
    ppo.critic_perception_key = "perception_obs"
    ppo.actor = SimpleNamespace(perception_time_gru=object())
    ppo.critic = SimpleNamespace(perception_time_gru=object())
    ppo.distill_enabled = True
    ppo.distill_mode = "dagger"
    ppo.dagger_enabled = True

    with pytest.raises(ValueError, match="not supported in time_gru mode"):
        ppo._validate_training_objective_configuration()


def test_onnx_export_rejects_stochastic_flow_inference_contract() -> None:
    ppo = object.__new__(PPO)
    ppo.actor = SimpleNamespace(
        supports_flow_matching=True,
        actor_module=SimpleNamespace(
            module=SimpleNamespace(inference_noise_std=0.25),
        ),
    )

    with pytest.raises(ValueError, match="flow_inference_noise_std=0"):
        _ = ppo.actor_onnx_wrapper


@pytest.mark.parametrize("distill_mode", ["dagger", "mse"])
def test_deterministic_teacher_labels_reject_stochastic_flow_inference(
    distill_mode: str,
) -> None:
    ppo = object.__new__(PPO)
    ppo.distill_mode = distill_mode
    ppo.teacher_use_stochastic_actions = False
    ppo.use_multi_teacher = False
    ppo.teacher_actor = SimpleNamespace(
        supports_flow_matching=True,
        actor_module=SimpleNamespace(
            module=SimpleNamespace(inference_noise_std=0.25),
        ),
    )

    with pytest.raises(ValueError, match="Deterministic teacher labels require flow_inference_noise_std=0"):
        ppo._validate_loaded_teacher_inference_contract()


def test_explicit_stochastic_dagger_teacher_allows_stochastic_flow_inference() -> None:
    ppo = object.__new__(PPO)
    ppo.distill_mode = "dagger"
    ppo.teacher_use_stochastic_actions = True
    ppo.use_multi_teacher = False
    ppo.teacher_actor = SimpleNamespace(
        supports_flow_matching=True,
        actor_module=SimpleNamespace(
            module=SimpleNamespace(inference_noise_std=0.25),
        ),
    )

    ppo._validate_loaded_teacher_inference_contract()


@pytest.mark.parametrize("multi_teacher", [False, True])
def test_active_observation_groups_follow_actual_teacher_side_inputs_and_drop_at_pure_ppo(
    multi_teacher: bool,
) -> None:
    ppo = object.__new__(PPO)
    ppo.actor_obs_keys = ["actor_obs"]
    ppo.critic_obs_keys = ["critic_obs"]
    ppo.teacher_obs_keys = ["teacher_obs"]
    ppo.actor_perception_key = "student_depth"
    ppo.critic_perception_key = ""
    ppo.teacher_perception_obs_key = ""
    ppo.distill_enabled = True
    ppo.distill_mode = "dagger"
    ppo.dagger_enabled = True
    ppo.use_ppo_dagger_schedule = True
    ppo.ppo_coeff = 0.5
    ppo.dagger_loss_coef = 1.0
    ppo.take_teacher_actions = False
    ppo.teacher_action_mix_ratio = 0.0
    ppo.fixed_bc_eval_num_samples = 0
    ppo._fixed_bc_eval_ready = True
    ppo.use_multi_teacher = multi_teacher
    ppo.multi_teacher_select_obs_var = "teacher_index"
    ppo.teacher_actor = None if multi_teacher else SimpleNamespace(
        perception_input_name="teacher_depth"
    )
    ppo.teacher_actors = (
        [
            SimpleNamespace(perception_input_name="teacher_depth_a"),
            SimpleNamespace(perception_input_name="teacher_depth_b"),
        ]
        if multi_teacher
        else []
    )
    set_active_groups = Mock()
    observation_manager = SimpleNamespace(
        active_group_names=None,
        set_active_groups=set_active_groups,
        cfg=SimpleNamespace(groups={}),
    )
    ppo.env = SimpleNamespace(observation_manager=observation_manager)
    ppo.is_main_process = False

    ppo._configure_active_observation_groups()

    active_with_bc = set_active_groups.call_args.args[0]
    assert "teacher_obs" in active_with_bc
    if multi_teacher:
        assert "teacher_depth_a" in active_with_bc
        assert "teacher_depth_b" in active_with_bc
        assert "teacher_index" in active_with_bc
    else:
        assert "teacher_depth" in active_with_bc

    set_active_groups.reset_mock()
    ppo.ppo_coeff = 1.0
    ppo._configure_active_observation_groups()

    assert set_active_groups.call_args.args[0] == [
        "actor_obs",
        "critic_obs",
        "student_depth",
    ]


def test_supervised_only_active_groups_exclude_critic_only_inputs() -> None:
    ppo = object.__new__(PPO)
    ppo._supervised_dagger_only = True
    ppo.actor_obs_keys = ["actor_obs"]
    ppo.critic_obs_keys = ["critic_obs", "critic_history"]
    ppo.teacher_obs_keys = ["teacher_obs"]
    ppo.actor_perception_key = "student_depth"
    ppo.critic_perception_key = "critic_depth"
    ppo.teacher_perception_obs_key = ""
    ppo.distill_enabled = True
    ppo.distill_mode = "dagger"
    ppo.dagger_enabled = True
    ppo.use_ppo_dagger_schedule = False
    ppo.bc_loss_coef = 1.0
    ppo.dagger_loss_coef = 1.0
    ppo.take_teacher_actions = False
    ppo.teacher_action_mix_ratio = 0.0
    ppo.fixed_bc_eval_num_samples = 0
    ppo._fixed_bc_eval_ready = True
    ppo.use_multi_teacher = False
    ppo.multi_teacher_select_obs_var = "teacher_index"
    ppo.teacher_actor = SimpleNamespace(perception_input_name="teacher_depth")
    ppo.teacher_actors = []
    set_active_groups = Mock()
    ppo.env = SimpleNamespace(
        observation_manager=SimpleNamespace(
            active_group_names=None,
            set_active_groups=set_active_groups,
            cfg=SimpleNamespace(groups={}),
        )
    )
    ppo.is_main_process = False

    ppo._configure_active_observation_groups()

    active_groups = set_active_groups.call_args.args[0]
    assert active_groups == [
        "actor_obs",
        "student_depth",
        "teacher_obs",
        "teacher_depth",
    ]
    assert "critic_obs" not in active_groups
    assert "critic_history" not in active_groups
    assert "critic_depth" not in active_groups


def test_eval_callback_creation_is_idempotent_and_instantiates_each_once():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(eval_callbacks={"a": {"id": "a"}, "b": {"id": "b"}})
    ppo.eval_callbacks = [object(), object()]
    created = []

    def fake_instantiate(config, *, training_loop):
        callback = SimpleNamespace(config=config, training_loop=training_loop)
        created.append(callback)
        return callback

    with patch("holosoma.agents.ppo.ppo.instantiate", side_effect=fake_instantiate):
        ppo._create_eval_callbacks()
        assert len(ppo.eval_callbacks) == 2
        ppo._create_eval_callbacks()

    assert len(ppo.eval_callbacks) == 2
    assert len(created) == 4
    assert [callback.config["id"] for callback in ppo.eval_callbacks] == ["a", "b"]


def test_eval_step_normalizes_and_advances_actor_exactly_once(monkeypatch):
    monkeypatch.delenv("HOLOSOMA_EVAL_DEBUG_PATH", raising=False)
    ppo = object.__new__(PPO)
    ppo.actor_obs_keys = ["obs"]
    ppo.actor_perception_key = ""
    ppo.eval_callbacks = []
    ppo.actor = SimpleNamespace(act_inference=Mock(return_value=torch.tensor([[3.0]])))
    normalize = Mock(side_effect=lambda obs, update: obs + 1.0)
    ppo._normalize_actor_obs = normalize
    actor_state = {"obs": {"obs": torch.tensor([[2.0]])}, "step": 0}

    result = ppo._pre_eval_env_step(actor_state)

    normalize.assert_called_once()
    ppo.actor.act_inference.assert_called_once()
    assert torch.equal(result["actions"], torch.tensor([[3.0]]))


def test_learn_at_completed_target_is_noop_before_mode_sync_or_reset():
    ppo = object.__new__(PPO)
    ppo.current_learning_iteration = 40000
    ppo.config = SimpleNamespace(num_learning_iterations=40000)
    ppo._train_mode = Mock(side_effect=AssertionError("must not change mode"))
    ppo.env = SimpleNamespace(reset_all=Mock(side_effect=AssertionError("must not reset")))

    ppo.learn()

    ppo._train_mode.assert_not_called()
    ppo.env.reset_all.assert_not_called()


def test_time_gru_required_onnx_export_fails_before_mode_change_or_reset():
    ppo = object.__new__(PPO)
    ppo.current_learning_iteration = 0
    ppo.config = SimpleNamespace(num_learning_iterations=1)
    ppo.use_time_gru = True
    ppo._should_export_onnx = MethodType(lambda self: True, ppo)
    ppo._train_mode = Mock(side_effect=AssertionError("must not change mode"))
    ppo.env = SimpleNamespace(reset_all=Mock(side_effect=AssertionError("must not reset")))

    with pytest.raises(ValueError, match=r"time_gru training.*--training\.export-onnx False"):
        ppo.learn()

    ppo._train_mode.assert_not_called()
    ppo.env.reset_all.assert_not_called()


def test_pure_bc_stochastic_flow_required_onnx_fails_before_any_training_work():
    ppo = object.__new__(PPO)
    ppo.current_learning_iteration = 0
    ppo.config = SimpleNamespace(num_learning_iterations=1)
    ppo.use_time_gru = False
    ppo.dagger_enabled = True
    ppo.use_ppo_dagger_schedule = False
    ppo.bc_loss_coef = 1.0
    ppo.actor = SimpleNamespace(
        supports_flow_matching=True,
        actor_module=SimpleNamespace(
            module=SimpleNamespace(inference_noise_std=0.25),
        ),
    )
    ppo._should_export_onnx = MethodType(lambda self: True, ppo)
    ppo._validate_future_dagger_bc_mask_signal = Mock(
        side_effect=AssertionError("must not run later preflight")
    )
    ppo._train_mode = Mock(side_effect=AssertionError("must not change mode"))
    ppo.env = SimpleNamespace(reset_all=Mock(side_effect=AssertionError("must not reset")))

    with pytest.raises(
        ValueError,
        match=r"Flow training.*flow_inference_noise_std.*--training\.export-onnx False",
    ):
        ppo.learn()

    ppo._validate_future_dagger_bc_mask_signal.assert_not_called()
    ppo._train_mode.assert_not_called()
    ppo.env.reset_all.assert_not_called()


def test_t1_aligned_sparse_root_required_onnx_fails_before_any_training_work():
    ppo = object.__new__(PPO)
    ppo.current_learning_iteration = 0
    ppo.config = SimpleNamespace(num_learning_iterations=1)
    ppo.use_time_gru = False
    ppo.actor = SimpleNamespace(
        supports_flow_matching=False,
        actor_module=SimpleNamespace(module=SimpleNamespace(inference_noise_std=0.0)),
    )
    ppo._experiment_config = SimpleNamespace(
        training=SimpleNamespace(export_onnx=True),
        command=SimpleNamespace(
            setup_terms={
                "motion_command": SimpleNamespace(
                    params={
                        "motion_config": SimpleNamespace(
                            contact_aware_sparse_root_command_mode="t1_aligned_segment"
                        )
                    }
                )
            }
        ),
    )
    ppo._validate_future_dagger_bc_mask_signal = Mock(
        side_effect=AssertionError("must not run later preflight")
    )
    ppo._train_mode = Mock(side_effect=AssertionError("must not change mode"))
    ppo.env = SimpleNamespace(reset_all=Mock(side_effect=AssertionError("must not reset")))

    with pytest.raises(
        ValueError,
        match=r"cannot require ONNX export.*t1_aligned_segment.*--training\.export-onnx False",
    ):
        ppo.learn()

    ppo._validate_future_dagger_bc_mask_signal.assert_not_called()
    ppo._train_mode.assert_not_called()
    ppo.env.reset_all.assert_not_called()


def test_learn_applies_resumed_iteration_schedule_before_initial_reset():
    events: list[tuple[str, int | None, int | None]] = []
    ppo = object.__new__(PPO)
    ppo.current_learning_iteration = 8
    ppo.config = SimpleNamespace(num_learning_iterations=10)
    ppo.is_multi_gpu = False
    ppo._train_mode = Mock()
    ppo._sync_training_curriculum_state = Mock(
        side_effect=lambda *, current_iteration, total_iterations: events.append(
            ("sync", current_iteration, total_iterations)
        )
    )

    def stop_after_reset():
        events.append(("reset", None, None))
        raise RuntimeError("stop after observing initial reset order")

    ppo.env = SimpleNamespace(reset_all=stop_after_reset)

    with pytest.raises(RuntimeError, match="stop after observing"):
        ppo.learn()

    assert events == [("sync", 8, 10), ("reset", None, None)]


def test_resumed_pure_ppo_drops_teacher_groups_before_canonical_reset() -> None:
    ppo = object.__new__(PPO)
    ppo.current_learning_iteration = 8
    ppo.config = SimpleNamespace(num_learning_iterations=10)
    ppo._train_mode = Mock()
    ppo.distill_enabled = True
    ppo.distill_mode = "dagger"
    ppo.dagger_enabled = True
    ppo.use_ppo_dagger_schedule = True
    ppo.ppo_start_epoch = 0
    ppo.dagger_end_epoch = 8
    ppo.ppo_start_coeff = 0.0
    ppo.ppo_target_coeff = 1.0
    ppo.ppo_schedule_step_epochs = 0
    ppo.ppo_coeff = 0.0
    ppo.dagger_loss_coef = 1.0
    ppo.ppo_start_noise_std = None
    ppo.take_teacher_actions = False
    ppo.teacher_action_mix_ratio = 0.0
    ppo.use_teacher_action_mix_schedule = False
    ppo.fixed_bc_eval_num_samples = 0
    ppo._fixed_bc_eval_ready = True
    ppo.actor_obs_keys = ["actor_obs"]
    ppo.critic_obs_keys = ["critic_obs"]
    ppo.teacher_obs_keys = ["teacher_obs"]
    ppo.actor_perception_key = "student_depth"
    ppo.critic_perception_key = ""
    ppo.teacher_perception_obs_key = ""
    ppo.teacher_actor = SimpleNamespace(perception_input_name="teacher_depth")
    ppo.teacher_actors = []
    ppo.use_multi_teacher = False
    ppo.multi_teacher_select_obs_var = "teacher_index"
    set_active_groups = Mock()
    observation_manager = SimpleNamespace(
        active_group_names=None,
        set_active_groups=set_active_groups,
        cfg=SimpleNamespace(groups={}),
    )
    ppo.env = SimpleNamespace(observation_manager=observation_manager)
    ppo.is_main_process = False

    def stop_at_reset(**_kwargs):
        assert ppo.ppo_coeff == pytest.approx(1.0)
        assert set_active_groups.call_args.args[0] == [
            "actor_obs",
            "critic_obs",
            "student_depth",
        ]
        raise RuntimeError("stop after checking pre-reset active groups")

    ppo._reset_rollout_stream_at_canonical_boundary = Mock(side_effect=stop_at_reset)

    with pytest.raises(RuntimeError, match="pre-reset active groups"):
        ppo.learn()


def test_learn_synchronizes_curriculum_once_before_first_rollout_and_once_per_later_iteration():
    sync_iterations: list[int] = []
    ppo = object.__new__(PPO)
    ppo.current_learning_iteration = 5
    ppo.config = SimpleNamespace(
        num_learning_iterations=8,
        init_at_random_ep_len=False,
        save_interval=2,
    )
    ppo.device = "cpu"
    ppo.log_dir = "/tmp/holosoma-test"
    ppo.is_multi_gpu = True
    ppo.is_main_process = False
    ppo._experiment_config = SimpleNamespace(
        training=SimpleNamespace(export_onnx=False)
    )
    ppo.env = SimpleNamespace(reset_all=Mock(return_value={"obs": torch.zeros(1, 1)}))
    ppo.algo_timing = SimpleNamespace(enabled=False)
    ppo.logging_helper = SimpleNamespace(
        record_collection_time=lambda: nullcontext(),
        record_learn_time=lambda: nullcontext(),
        synchronize_distributed_metrics=lambda metrics, **_kwargs: metrics,
    )
    ppo._train_mode = Mock()
    ppo._sync_training_curriculum_state = Mock()
    ppo._curriculum_state_sync_enabled = Mock(return_value=True)
    ppo._synchronize_curriculum_metrics = Mock(
        side_effect=lambda: sync_iterations.append(ppo.current_learning_iteration)
    )
    ppo._reset_step_timing = Mock()
    ppo._sync_iteration_boundary = Mock()
    ppo._refresh_distillation_iteration_state = Mock()
    ppo._adjust_teacher_action_mix_ratio = Mock()
    ppo._apply_ppo_start_noise_std_cap = Mock()
    ppo._rollout_step = Mock(side_effect=lambda obs: obs)
    ppo._training_step = Mock(return_value={})
    ppo._capture_step_timing = Mock()
    ppo._emit_step_timing_summary = Mock()
    ppo._setup_gloo_barrier_group = Mock(return_value=object())
    ppo._get_distributed_loss_weight = Mock(return_value=1.0)
    ppo._distributed_barrier = Mock()
    ppo.save = Mock()

    ppo.learn()

    # Iteration 5 is synchronized once before reset_all(); its loop body must
    # not repeat that collective.  Iterations 6 and 7 each synchronize the AS
    # state produced by their preceding rollout.
    assert sync_iterations == [5, 6, 7]
    assert ppo._synchronize_curriculum_metrics.call_count == 3
    assert ppo._rollout_step.call_count == 3
    assert [call.kwargs["next_iteration"] for call in ppo.save.call_args_list] == [6, 8]
    assert [Path(call.args[0]).name for call in ppo.save.call_args_list] == [
        "model_00006.pt",
        "model_00008.pt",
    ]
    # Ordinary checkpoint publication is observational by default.  Only the
    # initial learn() boundary resets the environment; model_00006.pt must not
    # truncate the live rollout.
    assert ppo.env.reset_all.call_count == 1


def test_checkpoint_rank_zero_io_failure_is_gathered_before_peer_barrier():
    ppo = object.__new__(PPO)
    ppo.is_multi_gpu = True
    ppo.is_main_process = True
    ppo.gpu_world_size = 2
    ppo.gpu_global_rank = 0
    ppo._synchronize_training_phase_error = MethodType(
        lambda self, error, *, operation: (_ for _ in ()).throw(error) if error is not None else None,
        ppo,
    )
    ppo.save = Mock(side_effect=OSError("injected disk failure"))
    group = object()
    ppo._setup_gloo_barrier_group = Mock(return_value=group)

    def gather_rank_zero_failure(results, local_result, **kwargs):
        assert kwargs == {"group": group}
        assert local_result == {"rank": 0, "error": "OSError: injected disk failure"}
        results[:] = [local_result, {"rank": 1, "error": None}]

    with (
        patch("holosoma.agents.ppo.ppo.torch.distributed.is_initialized", return_value=True),
        patch("holosoma.agents.ppo.ppo.torch.distributed.get_world_size", return_value=2),
        patch("holosoma.agents.ppo.ppo.torch.distributed.get_rank", return_value=0),
        patch(
            "holosoma.agents.ppo.ppo.torch.distributed.all_gather_object",
            side_effect=gather_rank_zero_failure,
        ),
        pytest.raises(RuntimeError, match="injected disk failure"),
    ):
        ppo._save_checkpoint_with_distributed_outcome(
            "/tmp/model_00010.pt",
            next_iteration=10,
        )



def test_checkpoint_peer_raises_rank_zero_publication_failure_from_gather():
    ppo = object.__new__(PPO)
    ppo.is_multi_gpu = True
    ppo.is_main_process = False
    ppo.gpu_world_size = 2
    ppo.gpu_global_rank = 1
    ppo._synchronize_training_phase_error = MethodType(
        lambda self, error, *, operation: (_ for _ in ()).throw(error) if error is not None else None,
        ppo,
    )
    ppo.save = Mock()
    group = object()
    ppo._setup_gloo_barrier_group = Mock(return_value=group)

    def inject_rank_zero_failure(results, local_result, **kwargs):
        assert kwargs == {"group": group}
        assert local_result == {"rank": 1, "error": None}
        results[:] = [
            {"rank": 0, "error": "OSError: remote disk full"},
            local_result,
        ]

    with (
        patch("holosoma.agents.ppo.ppo.torch.distributed.is_initialized", return_value=True),
        patch("holosoma.agents.ppo.ppo.torch.distributed.get_world_size", return_value=2),
        patch("holosoma.agents.ppo.ppo.torch.distributed.get_rank", return_value=1),
        patch(
            "holosoma.agents.ppo.ppo.torch.distributed.all_gather_object",
            side_effect=inject_rank_zero_failure,
        ),
        pytest.raises(RuntimeError, match="remote disk full"),
    ):
        ppo._save_checkpoint_with_distributed_outcome(
            "/tmp/model_00010.pt",
            next_iteration=10,
        )

    ppo.save.assert_called_once_with("/tmp/model_00010.pt", next_iteration=10)


def test_checkpoint_non_main_local_failure_is_gathered_instead_of_escaping_early():
    ppo = object.__new__(PPO)
    ppo.is_multi_gpu = True
    ppo.is_main_process = False
    ppo.gpu_world_size = 2
    ppo.gpu_global_rank = 1
    ppo._synchronize_training_phase_error = MethodType(
        lambda self, error, *, operation: (_ for _ in ()).throw(error) if error is not None else None,
        ppo,
    )
    ppo.save = Mock(side_effect=RuntimeError("rank-local RNG restore failed"))
    group = object()
    ppo._setup_gloo_barrier_group = Mock(return_value=group)

    def gather_peer_failure(results, local_result, **kwargs):
        assert kwargs == {"group": group}
        assert local_result == {
            "rank": 1,
            "error": "RuntimeError: rank-local RNG restore failed",
        }
        results[:] = [{"rank": 0, "error": None}, local_result]

    with (
        patch("holosoma.agents.ppo.ppo.torch.distributed.is_initialized", return_value=True),
        patch("holosoma.agents.ppo.ppo.torch.distributed.get_world_size", return_value=2),
        patch("holosoma.agents.ppo.ppo.torch.distributed.get_rank", return_value=1),
        patch(
            "holosoma.agents.ppo.ppo.torch.distributed.all_gather_object",
            side_effect=gather_peer_failure,
        ),
        pytest.raises(RuntimeError, match="rank=1: RuntimeError: rank-local RNG restore failed"),
    ):
        ppo._save_checkpoint_with_distributed_outcome(
            "/tmp/model_00010.pt",
            next_iteration=10,
        )


def test_complete_distributed_checkpoint_protocol_preserves_rng() -> None:
    original_rng = capture_rng_checkpoint_state()
    try:
        ppo = object.__new__(PPO)
        ppo.is_multi_gpu = True
        ppo.is_main_process = True
        ppo.gpu_world_size = 2
        ppo.gpu_global_rank = 0
        ppo._synchronize_training_phase_error = MethodType(
            lambda self, error, *, operation: (_ for _ in ()).throw(error) if error is not None else None,
            ppo,
        )
        group = object()
        ppo._setup_gloo_barrier_group = Mock(return_value=group)

        def noisy_save(*_args, **_kwargs):
            random.random()
            np.random.random()
            torch.rand(1)

        ppo.save = Mock(side_effect=noisy_save)

        def noisy_outcome_gather(results, local_result, **kwargs):
            assert kwargs == {"group": group}
            random.random()
            np.random.random()
            torch.rand(1)
            results[:] = [local_result, {"rank": 1, "error": None}]

        random.seed(34)
        np.random.seed(35)
        torch.manual_seed(36)
        boundary = capture_rng_checkpoint_state()
        expected = (random.random(), float(np.random.random()), torch.rand(2))
        restore_rng_checkpoint_state(boundary)

        with (
            patch("holosoma.agents.ppo.ppo.torch.distributed.is_initialized", return_value=True),
            patch("holosoma.agents.ppo.ppo.torch.distributed.get_world_size", return_value=2),
            patch("holosoma.agents.ppo.ppo.torch.distributed.get_rank", return_value=0),
            patch(
                "holosoma.agents.ppo.ppo.torch.distributed.all_gather_object",
                side_effect=noisy_outcome_gather,
            ),
        ):
            ppo._save_checkpoint_with_distributed_outcome(
                "/tmp/model_00010.pt",
                next_iteration=10,
            )

        assert random.random() == expected[0]
        assert float(np.random.random()) == expected[1]
        assert torch.equal(torch.rand(2), expected[2])
    finally:
        restore_rng_checkpoint_state(original_rng)


def test_onnx_checkpoint_export_side_effects_do_not_advance_training_rng(tmp_path) -> None:
    original_rng = capture_rng_checkpoint_state()
    try:
        ppo = object.__new__(PPO)
        ppo._should_export_onnx = MethodType(lambda self: True, ppo)

        def noisy_export(self, **_kwargs):
            random.random()
            np.random.random()
            torch.rand(1)
            raise RuntimeError("synthetic optional export failure")

        ppo.export = MethodType(noisy_export, ppo)
        random.seed(14)
        np.random.seed(15)
        torch.manual_seed(16)
        boundary = capture_rng_checkpoint_state()
        expected = (random.random(), float(np.random.random()), torch.rand(2))
        restore_rng_checkpoint_state(boundary)

        ppo._export_onnx_checkpoint(str(tmp_path / "policy.onnx"), iteration=3)

        assert random.random() == expected[0]
        assert float(np.random.random()) == expected[1]
        assert torch.equal(torch.rand(2), expected[2])
    finally:
        restore_rng_checkpoint_state(original_rng)


def test_required_final_onnx_failure_is_fatal_cleans_partial_and_preserves_rng(tmp_path) -> None:
    original_rng = capture_rng_checkpoint_state()
    try:
        ppo = object.__new__(PPO)
        ppo._should_export_onnx = MethodType(lambda self: True, ppo)
        artifact = tmp_path / "policy.onnx"

        def failed_export(self, **_kwargs):
            artifact.write_bytes(b"partial")
            random.random()
            np.random.random()
            torch.rand(1)
            raise RuntimeError("synthetic final export failure")

        ppo.export = MethodType(failed_export, ppo)
        random.seed(114)
        np.random.seed(115)
        torch.manual_seed(116)
        boundary = capture_rng_checkpoint_state()
        expected = (random.random(), float(np.random.random()), torch.rand(2))
        restore_rng_checkpoint_state(boundary)

        with pytest.raises(RuntimeError, match="synthetic final export failure"):
            ppo._export_onnx_checkpoint(
                str(artifact),
                iteration=3,
                required=True,
            )

        assert not artifact.exists()
        assert random.random() == expected[0]
        assert float(np.random.random()) == expected[1]
        assert torch.equal(torch.rand(2), expected[2])
    finally:
        restore_rng_checkpoint_state(original_rng)


def test_successful_onnx_export_returns_sha256_of_published_bytes(tmp_path) -> None:
    ppo = object.__new__(PPO)
    ppo._should_export_onnx = MethodType(lambda self: True, ppo)
    artifact = tmp_path / "policy.onnx"
    payload = b"authenticated-onnx-test-payload"
    ppo.export = MethodType(
        lambda self, **kwargs: Path(kwargs["onnx_file_path"]).write_bytes(payload),
        ppo,
    )

    digest = ppo._export_onnx_checkpoint(
        str(artifact),
        iteration=3,
        required=True,
    )

    assert digest == hashlib.sha256(payload).hexdigest()
    assert artifact.read_bytes() == payload


def test_time_gru_policy_export_fails_closed_until_hidden_state_is_explicit_io() -> None:
    ppo = object.__new__(PPO)
    ppo.use_time_gru = True

    with pytest.raises(ValueError, match="recurrent hidden state.*explicit ONNX input/output"):
        _ = ppo.actor_onnx_wrapper


def test_epoch_logging_side_effects_do_not_advance_training_rng() -> None:
    original_rng = capture_rng_checkpoint_state()
    try:
        ppo = object.__new__(PPO)
        ppo.is_main_process = True
        ppo.is_multi_gpu = False

        def noisy_logging(self, it, loss_dict, *, fixed_bc_eval_metrics=None):
            assert it == 3
            assert loss_dict == {"loss": 1.0}
            assert fixed_bc_eval_metrics == {"bc": 2.0}
            random.random()
            np.random.random()
            torch.rand(1)

        ppo._post_epoch_logging = MethodType(noisy_logging, ppo)
        random.seed(24)
        np.random.seed(25)
        torch.manual_seed(26)
        boundary = capture_rng_checkpoint_state()
        expected = (random.random(), float(np.random.random()), torch.rand(2))
        restore_rng_checkpoint_state(boundary)

        ppo._post_epoch_logging_preserving_rng(
            3,
            {"loss": 1.0},
            fixed_bc_eval_metrics={"bc": 2.0},
        )

        assert random.random() == expected[0]
        assert float(np.random.random()) == expected[1]
        assert torch.equal(torch.rand(2), expected[2])
    finally:
        restore_rng_checkpoint_state(original_rng)


def _checkpoint_load_stub(
    *,
    normalize_actor=False,
    normalize_critic=False,
    load_optimizer=False,
    optimizer_momentum=0.0,
):
    ppo = object.__new__(PPO)
    ppo.actor = nn.Linear(1, 1)
    ppo.critic = nn.Linear(1, 1)
    ppo.actor_optimizer = torch.optim.SGD(
        ppo.actor.parameters(), lr=0.1, momentum=optimizer_momentum
    )
    ppo.critic_optimizer = torch.optim.SGD(
        ppo.critic.parameters(), lr=0.1, momentum=optimizer_momentum
    )
    ppo.actor_obs_normalizers = {
        "actor_obs": EmpiricalNormalization((1,), "cpu") if normalize_actor else nn.Identity()
    }
    ppo.critic_obs_normalizers = {
        "critic_obs": EmpiricalNormalization((1,), "cpu") if normalize_critic else nn.Identity()
    }
    ppo.config = SimpleNamespace(
        load_optimizer=load_optimizer,
        normalize_actor_obs=normalize_actor,
        normalize_critic_obs=normalize_critic,
        init_noise_std=0.2,
    )
    ppo.actor_learning_rate = 0.1
    ppo.min_actor_learning_rate = 1.0e-5
    ppo.max_actor_learning_rate = 0.1
    ppo.critic_learning_rate = 0.1
    ppo.min_critic_learning_rate = 1.0e-5
    ppo.max_critic_learning_rate = 0.1
    ppo.device = "cpu"
    ppo.gpu_global_rank = 0
    ppo.is_multi_gpu = False
    ppo.dagger_enabled = False
    ppo.ppo_start_noise_std = None
    ppo.env = SimpleNamespace(load_checkpoint_state=Mock())
    return ppo


def _perception_geometry_semantics(*, object_digest: str, robot_digest: str = "a" * 64) -> dict:
    return {
        "num_envs": 1,
        "camera_source": "far_tracking_warp",
        "far_tracking_geometry": (
            ("torso_link", ".stl", 3, robot_digest),
            ("object__variant_000", ".obj", 7, object_digest),
        ),
        "far_tracking_topology": {
            "robot_slot_indices": (0.0,),
            "robot_body_indices": (4.0,),
            "robot_body_names": ("torso_link",),
            "robot_body_offset_positions": (0.0, 0.0, 0.0),
            "robot_body_offset_quaternions": (0.0, 0.0, 0.0, 1.0),
            "object_slot_indices": (1.0,),
            "object_source_indices": (0.0,),
            "primitive_source_indices": (),
            "object_names": ("object",),
            "object_active_env_ids": (None,),
        },
    }


def _perception_rank_env_state(*, object_digest: str, robot_digest: str = "a" * 64) -> dict:
    return {
        "version": 4,
        "perception_managers": {
            "version": 1,
            "role_owners": {"actor": "actor", "teacher": None, "critic": "actor"},
            "states": {
                "actor": {
                    "version": 1,
                    "semantics": _perception_geometry_semantics(
                        object_digest=object_digest,
                        robot_digest=robot_digest,
                    ),
                }
            },
        },
    }


def test_perception_geometry_support_aggregates_every_training_rank() -> None:
    ppo = _checkpoint_load_stub()
    ppo.actor_perception_key = "perception_obs"

    support = ppo._aggregate_actor_perception_geometry_support(
        {
            "0": _perception_rank_env_state(object_digest="b" * 64),
            "1": _perception_rank_env_state(object_digest="c" * 64),
        },
        allow_legacy_missing=False,
    )

    assert support is not None
    assert support["training_rank_count"] == 2
    assert [item["mesh"]["sha256"] for item in support["object_mesh_support"]] == [
        "b" * 64,
        "c" * 64,
    ]


def test_perception_geometry_support_rejects_rank_local_robot_drift() -> None:
    ppo = _checkpoint_load_stub()
    ppo.actor_perception_key = "perception_obs"

    with pytest.raises(ValueError, match="robot mesh bindings differ on training rank 1"):
        ppo._aggregate_actor_perception_geometry_support(
            {
                "0": _perception_rank_env_state(object_digest="b" * 64),
                "1": _perception_rank_env_state(
                    object_digest="c" * 64,
                    robot_digest="d" * 64,
                ),
            },
            allow_legacy_missing=False,
        )


def test_evaluation_geometry_validation_precedes_actor_state_commit() -> None:
    ppo = _checkpoint_load_stub(normalize_actor=False)
    ppo.actor_perception_key = "perception_obs"
    ppo._policy_load_runtime_config = SimpleNamespace(
        to_serializable_dict=lambda: {"runtime": "actor-contract"}
    )
    manager = object.__new__(PerceptionManager)
    manager.enabled = True
    manager.validate_deployment_geometry_support = Mock(
        side_effect=ValueError("unknown live geometry")
    )
    ppo.env._perception_checkpoint_topology = Mock(
        return_value=(
            {"actor": "actor", "teacher": None, "critic": "actor"},
            {"actor": manager},
        )
    )
    actor_state = {
        key: value.detach().clone() for key, value in ppo.actor.state_dict().items()
    }
    actor_state["weight"].fill_(7.0)
    checkpoint = {
        "actor_model_state_dict": actor_state,
        "experiment_config": {"algo": {"config": {"normalize_actor_obs": False}}},
        "env_state_by_rank": {
            "0": _perception_rank_env_state(object_digest="b" * 64)
        },
        "iter": 12,
        "next_iter": 13,
    }
    original_actor_weight = ppo.actor.weight.detach().clone()

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, "a" * 64),
        ),
        patch("holosoma.agents.ppo.ppo.validate_policy_init_payload_identity"),
        pytest.raises(ValueError, match="unknown live geometry"),
    ):
        ppo.load_evaluation("evaluation.pt")

    assert torch.equal(ppo.actor.weight, original_actor_weight)


def test_evaluation_retains_all_rank_perception_geometry_for_export() -> None:
    ppo = _checkpoint_load_stub(normalize_actor=False)
    ppo.actor_perception_key = "perception_obs"
    ppo._policy_load_runtime_config = SimpleNamespace(
        to_serializable_dict=lambda: {"runtime": "actor-contract"}
    )
    manager = object.__new__(PerceptionManager)
    manager.enabled = True
    manager.validate_deployment_geometry_support = Mock(
        side_effect=lambda support: support
    )
    ppo.env._perception_checkpoint_topology = Mock(
        return_value=(
            {"actor": "actor", "teacher": None, "critic": "actor"},
            {"actor": manager},
        )
    )
    checkpoint = {
        "actor_model_state_dict": ppo.actor.state_dict(),
        "experiment_config": {"algo": {"config": {"normalize_actor_obs": False}}},
        "env_state_by_rank": {
            "0": _perception_rank_env_state(object_digest="b" * 64),
            "1": _perception_rank_env_state(object_digest="c" * 64),
        },
        "iter": 12,
        "next_iter": 13,
    }

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, "a" * 64),
        ),
        patch("holosoma.agents.ppo.ppo.validate_policy_init_payload_identity"),
    ):
        ppo.load_evaluation("evaluation.pt")

    support = ppo._actor_perception_training_geometry_support
    assert support["training_rank_count"] == 2
    assert len(support["object_mesh_support"]) == 2
    manager.validate_deployment_geometry_support.assert_called_once_with(support)


def _finalized_test_training_provenance() -> dict:
    runtime_asset_manifest = {"version": 2, "fixture": "ppo-evaluation-source"}
    return {
        "version": 2,
        "teacher_sha256": "a" * 64,
        "policy_init_enabled": True,
        "policy_init_sha256": "b" * 64,
        "training_resume_enabled": True,
        "training_resume_sha256": "c" * 64,
        "motion_shard_manifest_sha256": "d" * 64,
        "contact_sidecar_manifest_sha256": "e" * 64,
        "source_bundle_sha256": "f" * 64,
        "runtime_asset_manifest_phase": "final",
        "runtime_asset_manifest_sha256": embedded_runtime_asset_manifest_sha256(
            runtime_asset_manifest
        ),
        "runtime_asset_manifest": runtime_asset_manifest,
    }


def _full_resume_checkpoint(ppo):
    return {
        "actor_model_state_dict": ppo.actor.state_dict(),
        "critic_model_state_dict": ppo.critic.state_dict(),
        "actor_optimizer_state_dict": ppo.actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": ppo.critic_optimizer.state_dict(),
        "iter": 3,
        "next_iter": 4,
        "rng_state_by_rank": {"0": capture_rng_checkpoint_state()},
        "rollout_resume_contract": ppo._rollout_resume_contract(4),
    }


def _motion_transition_contract(
    *,
    source_semantics: str = "global_multi_clip_runtime",
    prepend_steps: int = 10,
    append_steps: int = 0,
) -> dict:
    return {
        "version": 1,
        "control_dt_s": 0.02,
        "source_semantics": source_semantics,
        "prepend": {
            "implementation": (
                "runtime_hold"
                if source_semantics == "global_multi_clip_runtime" and prepend_steps > 0
                else "static_splice"
                if prepend_steps > 0
                else "none"
            ),
            "applied": prepend_steps > 0,
            "steps": prepend_steps,
        },
        "append": {
            "implementation": "static_splice" if append_steps > 0 else "none",
            "applied": append_steps > 0,
            "steps": append_steps,
        },
    }


def _add_checkpoint_std_parameter(ppo, *, value=0.1, min_noise_std=0.01):
    ppo.actor.std = nn.Parameter(torch.tensor([value], dtype=ppo.actor.weight.dtype))
    ppo.actor.min_noise_std = min_noise_std
    ppo.actor.min_mean_noise_std = None
    ppo.actor.max_noise_std = 0.5
    ppo.actor_optimizer = torch.optim.SGD(ppo.actor.parameters(), lr=0.1)
    return ppo


def test_motion_transition_checkpoint_contract_rejects_legacy_ambiguous_timeline() -> None:
    ppo = _checkpoint_load_stub()
    live = _motion_transition_contract()
    checkpoint = {
        "experiment_config": {
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
        }
    }

    with pytest.raises(ValueError, match="predates motion_transition_contract"):
        ppo._validate_checkpoint_motion_transition_contract(
            checkpoint,
            live_contract=live,
            compare_live=True,
            operation="Policy-init",
        )


def test_motion_transition_checkpoint_contract_allows_legacy_explicitly_inactive_evaluation() -> None:
    ppo = _checkpoint_load_stub()
    checkpoint = {
        "experiment_config": {
            "command": {
                "setup_terms": {
                    "motion_command": {
                        "params": {
                            "motion_config": {
                                "enable_default_pose_prepend": False,
                                "enable_default_pose_append": False,
                            }
                        }
                    }
                }
            }
        }
    }

    restored, digest = ppo._validate_checkpoint_motion_transition_contract(
        checkpoint,
        live_contract=None,
        compare_live=False,
        operation="Evaluation",
    )

    assert restored is None
    assert digest is None


@pytest.mark.parametrize("malformed_enabled", [1, "true", None])
def test_legacy_transition_detection_treats_malformed_enable_as_ambiguous(
    malformed_enabled,
) -> None:
    ppo = _checkpoint_load_stub()
    checkpoint = {
        "experiment_config": {
            "command": {
                "setup_terms": {
                    "motion_command": {
                        "params": {
                            "motion_config": {
                                "enable_default_pose_prepend": malformed_enabled,
                                "default_pose_prepend_duration_s": 0.0,
                                "enable_default_pose_append": False,
                            }
                        }
                    }
                }
            }
        }
    }

    with pytest.raises(ValueError, match="predates motion_transition_contract"):
        ppo._validate_checkpoint_motion_transition_contract(
            checkpoint,
            live_contract=None,
            compare_live=False,
            operation="Evaluation",
        )


def test_full_resume_requires_contract_even_when_live_motion_transitions_are_inactive() -> None:
    ppo = _checkpoint_load_stub()
    inactive = _motion_transition_contract(
        source_semantics="single_clip_static",
        prepend_steps=0,
        append_steps=0,
    )
    with pytest.raises(ValueError, match="predates motion_transition_contract"):
        ppo._validate_checkpoint_motion_transition_contract(
            {"experiment_config": {}},
            live_contract=inactive,
            compare_live=True,
            operation="Training-resume",
        )


def test_motion_transition_checkpoint_contract_rejects_digest_or_live_mismatch() -> None:
    ppo = _checkpoint_load_stub()
    saved = _motion_transition_contract()
    checkpoint = {
        "experiment_config": {},
        "motion_transition_contract": saved,
        "motion_transition_contract_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="does not authenticate"):
        ppo._validate_checkpoint_motion_transition_contract(
            checkpoint,
            live_contract=saved,
            compare_live=True,
            operation="Training-resume",
        )

    checkpoint["motion_transition_contract_sha256"] = motion_transition_contract_sha256(saved)
    live_single = _motion_transition_contract(
        source_semantics="single_clip_static",
        prepend_steps=10,
        append_steps=10,
    )
    with pytest.raises(ValueError, match="differs from the live runtime"):
        ppo._validate_checkpoint_motion_transition_contract(
            checkpoint,
            live_contract=live_single,
            compare_live=True,
            operation="Training-resume",
        )


def test_distributed_motion_transition_contract_requires_rank_exact_equality() -> None:
    ppo = _checkpoint_load_stub()
    ppo.env = SimpleNamespace(command_manager=object())
    ppo.is_multi_gpu = True
    ppo.gpu_world_size = 2
    group = object()
    ppo._setup_gloo_barrier_group = Mock(return_value=group)
    rank_zero = _motion_transition_contract()
    rank_one = _motion_transition_contract(
        source_semantics="single_clip_static",
        prepend_steps=10,
        append_steps=10,
    )
    ppo._local_motion_transition_contract = Mock(
        return_value=(rank_zero, motion_transition_contract_sha256(rank_zero))
    )

    def gather(gathered, local_result, **kwargs):
        assert kwargs == {"group": group}
        gathered[:] = [
            local_result,
            {
                "rank": 1,
                "error": None,
                "contract": rank_one,
                "digest": motion_transition_contract_sha256(rank_one),
            },
        ]

    with (
        patch("holosoma.agents.ppo.ppo.torch.distributed.is_available", return_value=True),
        patch("holosoma.agents.ppo.ppo.torch.distributed.is_initialized", return_value=True),
        patch("holosoma.agents.ppo.ppo.torch.distributed.get_world_size", return_value=2),
        patch("holosoma.agents.ppo.ppo.torch.distributed.get_rank", return_value=0),
        patch(
            "holosoma.agents.ppo.ppo.torch.distributed.all_gather_object",
            side_effect=gather,
        ),
        pytest.raises(RuntimeError, match="DDP ranks disagree"),
    ):
        ppo._collect_distributed_motion_transition_contract()


def test_legacy_requested_transition_policy_init_keeps_new_live_global_contract() -> None:
    ppo = _checkpoint_load_stub(normalize_actor=False)
    ppo.current_learning_iteration = 0
    live = _motion_transition_contract()
    motion_command = SimpleNamespace(get_motion_transition_contract=lambda: live)
    ppo.env = SimpleNamespace(
        command_manager=SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None),
        load_checkpoint_state=Mock(),
    )
    runtime_config = {
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
        },
        "algo": {"config": {"normalize_actor_obs": False}},
    }
    ppo._policy_load_runtime_config = SimpleNamespace(to_serializable_dict=lambda: runtime_config)
    ppo._experiment_config = SimpleNamespace(to_serializable_dict=lambda: runtime_config)
    ppo._source_experiment_config_dict = None
    ppo._wandb_run_path = None
    ppo._training_provenance = None
    ppo._source_checkpoint_sha256 = None
    ppo._actor_perception_training_geometry_support = None
    ppo._evaluation_only = False
    checkpoint = {
        "actor_model_state_dict": ppo.actor.state_dict(),
        # Legacy source requested transitions but did not serialize their
        # effective single-vs-global implementation.
        "experiment_config": runtime_config,
    }

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, "7f621" + "0" * 59),
        ),
        patch("holosoma.agents.ppo.ppo.validate_policy_init_payload_identity"),
    ):
        ppo.load_policy_init("legacy-box-init.pt")

    assert ppo._motion_transition_contract == live
    assert ppo._motion_transition_contract_sha256 == motion_transition_contract_sha256(live)
    metadata = ppo._checkpoint_metadata(iteration=0)
    assert metadata["motion_transition_contract"] == live
    assert metadata["motion_transition_contract_sha256"] == motion_transition_contract_sha256(live)


def test_checkpoint_save_publishes_collected_live_motion_transition_contract() -> None:
    ppo = _checkpoint_load_stub()
    ppo.is_main_process = True
    ppo.current_learning_iteration = 4
    ppo.fixed_bc_eval_num_samples = 0
    live = _motion_transition_contract()
    ppo._experiment_config = SimpleNamespace(to_serializable_dict=lambda: {"training": "current"})
    ppo._source_experiment_config_dict = None
    ppo._wandb_run_path = None
    ppo._training_provenance = None
    ppo._source_checkpoint_sha256 = None
    ppo._actor_perception_training_geometry_support = None
    ppo._motion_transition_contract = None
    ppo._motion_transition_contract_sha256 = None
    ppo._validate_checkpoint_publish_state = Mock()
    rng_state = capture_rng_checkpoint_state()
    ppo._collect_distributed_rng_states = Mock(return_value={"0": rng_state})
    ppo._collect_distributed_env_states = Mock(return_value={"0": {}})
    ppo._collect_distributed_motion_transition_contract = Mock(
        return_value=(live, motion_transition_contract_sha256(live))
    )
    ppo._aggregate_actor_perception_geometry_support = Mock(return_value=None)
    ppo._collect_distributed_fixed_bc_eval_states = Mock(return_value={})
    ppo._rollout_resume_contract = Mock(return_value={"version": 1})
    published = {}
    ppo.logging_helper = SimpleNamespace(
        save_checkpoint_artifact=lambda payload, path: published.update(payload)
    )

    ppo.save("model_00004.pt", next_iteration=4)

    assert published["motion_transition_contract"] == live
    assert published["motion_transition_contract_sha256"] == motion_transition_contract_sha256(live)


def test_checkpoint_publication_side_effects_do_not_advance_training_rng() -> None:
    original_rng = capture_rng_checkpoint_state()
    try:
        ppo = _checkpoint_load_stub()
        ppo.is_main_process = True
        ppo.current_learning_iteration = 4
        ppo.fixed_bc_eval_num_samples = 0
        def noisy_env_state_collection():
            random.random()
            np.random.random()
            torch.rand(1)
            return {}

        ppo.env = SimpleNamespace(get_checkpoint_state=noisy_env_state_collection)
        ppo._checkpoint_metadata = MethodType(lambda self, iteration: {}, ppo)

        original_preflight = ppo._validate_checkpoint_publish_state

        def noisy_preflight():
            random.random()
            np.random.random()
            torch.rand(1)
            original_preflight()

        ppo._validate_checkpoint_publish_state = noisy_preflight

        published = {}

        def noisy_publication(checkpoint, _path):
            published.update(checkpoint)
            random.random()
            np.random.random()
            torch.rand(1)

        ppo.logging_helper = SimpleNamespace(save_checkpoint_artifact=noisy_publication)
        random.seed(31)
        np.random.seed(32)
        torch.manual_seed(33)
        boundary = capture_rng_checkpoint_state()
        expected = (random.random(), float(np.random.random()), torch.rand(2))
        restore_rng_checkpoint_state(boundary)

        ppo.save("model_00004.pt", next_iteration=4)

        assert random.random() == expected[0]
        assert float(np.random.random()) == expected[1]
        assert torch.equal(torch.rand(2), expected[2])

        # The serialized rank-local state is the same pre-save boundary, not
        # a state already advanced by environment/fixed-BC collection.
        restore_rng_checkpoint_state(published["rng_state_by_rank"]["0"])
        assert random.random() == expected[0]
        assert float(np.random.random()) == expected[1]
        assert torch.equal(torch.rand(2), expected[2])
    finally:
        restore_rng_checkpoint_state(original_rng)


def test_failed_checkpoint_preflight_does_not_advance_training_rng() -> None:
    original_rng = capture_rng_checkpoint_state()
    try:
        ppo = _checkpoint_load_stub()

        def noisy_failed_preflight():
            random.random()
            np.random.random()
            torch.rand(1)
            raise ValueError("injected checkpoint preflight failure")

        ppo._validate_checkpoint_publish_state = noisy_failed_preflight
        random.seed(41)
        np.random.seed(42)
        torch.manual_seed(43)
        boundary = capture_rng_checkpoint_state()
        expected = (random.random(), float(np.random.random()), torch.rand(2))
        restore_rng_checkpoint_state(boundary)

        with pytest.raises(ValueError, match="injected checkpoint preflight failure"):
            ppo.save("must-not-publish.pt", next_iteration=4)

        assert random.random() == expected[0]
        assert float(np.random.random()) == expected[1]
        assert torch.equal(torch.rand(2), expected[2])
    finally:
        restore_rng_checkpoint_state(original_rng)


def test_save_preflight_rejects_non_finite_optimizer_state_before_publication():
    ppo = _checkpoint_load_stub(optimizer_momentum=0.9)
    ppo.is_main_process = True
    ppo.logging_helper = SimpleNamespace(save_checkpoint_artifact=Mock())
    first_parameter = next(iter(ppo.actor.parameters()))
    ppo.actor_optimizer.state[first_parameter]["momentum_buffer"] = torch.full_like(
        first_parameter,
        float("inf"),
    )

    with pytest.raises(ValueError, match=r"live\.actor_optimizer_state_dict.*non-finite"):
        ppo.save("must-not-publish.pt")

    ppo.logging_helper.save_checkpoint_artifact.assert_not_called()


def test_save_preflight_rejects_actor_std_that_runtime_would_project():
    ppo = _add_checkpoint_std_parameter(_checkpoint_load_stub(), value=0.0, min_noise_std=0.01)
    ppo.is_main_process = True
    ppo.logging_helper = SimpleNamespace(save_checkpoint_artifact=Mock())

    with pytest.raises(ValueError, match=r"live\.actor_model_state_dict\.std violates"):
        ppo.save("must-not-publish.pt")

    ppo.logging_helper.save_checkpoint_artifact.assert_not_called()


def test_save_preflight_rejects_non_finite_live_normalizer_state():
    ppo = _checkpoint_load_stub(normalize_actor=True)
    ppo.is_main_process = True
    ppo.logging_helper = SimpleNamespace(save_checkpoint_artifact=Mock())
    ppo.actor_obs_normalizers["actor_obs"]._std.fill_(float("nan"))

    with pytest.raises(ValueError, match=r"live\.actor_obs_normalizer_state.*non-finite"):
        ppo.save("must-not-publish.pt")

    ppo.logging_helper.save_checkpoint_artifact.assert_not_called()


def test_save_rejects_non_finite_collected_payload_before_publication():
    ppo = _checkpoint_load_stub()
    ppo.is_main_process = True
    ppo.current_learning_iteration = 4
    ppo.fixed_bc_eval_num_samples = 0
    ppo._collect_distributed_env_states = MethodType(
        lambda self: {"0": {"bad_buffer": torch.tensor(float("nan"))}},
        ppo,
    )
    ppo._checkpoint_metadata = MethodType(lambda self, iteration: {}, ppo)
    ppo.logging_helper = SimpleNamespace(save_checkpoint_artifact=Mock())

    with pytest.raises(ValueError, match=r"checkpoint_publish_payload.*non-finite"):
        ppo.save("must-not-publish.pt", next_iteration=4)

    ppo.logging_helper.save_checkpoint_artifact.assert_not_called()


@pytest.mark.parametrize(
    ("values", "min_noise_std", "min_mean_noise_std", "max_noise_std"),
    [
        ([0.6], 0.01, None, 0.5),
        ([0.2, 0.8], None, 0.8, 0.8),
    ],
)
def test_checkpoint_actor_std_validator_rejects_upper_or_mean_constraint_drift(
    values, min_noise_std, min_mean_noise_std, max_noise_std
):
    ppo = _checkpoint_load_stub()
    ppo.actor.std = nn.Parameter(torch.tensor(values))
    ppo.actor.min_noise_std = min_noise_std
    ppo.actor.min_mean_noise_std = min_mean_noise_std
    ppo.actor.max_noise_std = max_noise_std
    state = {"std": ppo.actor.std.detach().clone()}

    with pytest.raises(ValueError, match="runtime policy-noise constraints"):
        ppo._validate_checkpoint_actor_std(state, path="actor_model_state_dict")


def test_checkpoint_actor_std_validator_accepts_exact_constraint_boundaries():
    ppo = _checkpoint_load_stub()
    ppo.actor.std = nn.Parameter(torch.tensor([0.01, 0.5]))
    ppo.actor.min_noise_std = 0.01
    ppo.actor.min_mean_noise_std = None
    ppo.actor.max_noise_std = 0.5
    state = {"std": ppo.actor.std.detach().clone()}

    ppo._validate_checkpoint_actor_std(state, path="actor_model_state_dict")


def _replace_stub_optimizers_with_initialized_adamw(ppo):
    ppo.actor_optimizer = torch.optim.AdamW(ppo.actor.parameters(), lr=0.1)
    ppo.critic_optimizer = torch.optim.AdamW(ppo.critic.parameters(), lr=0.1)
    for model, optimizer in (
        (ppo.actor, ppo.actor_optimizer),
        (ppo.critic, ppo.critic_optimizer),
    ):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()


def test_legacy_checkpoint_without_env_state_fails_for_adaptive_curriculum(monkeypatch):
    monkeypatch.delenv("HOLOSOMA_ALLOW_FRESH_CURRICULUM_RESUME", raising=False)
    ppo = _checkpoint_load_stub()
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: True, ppo)
    checkpoint = {
        "actor_model_state_dict": ppo.actor.state_dict(),
        "critic_model_state_dict": ppo.critic.state_dict(),
        "iter": 3,
    }

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(RuntimeError, match="ALLOW_FRESH_CURRICULUM_RESUME"),
    ):
        ppo.load("legacy.pt")


def test_full_resume_direct_load_requires_provenance_by_default(monkeypatch):
    monkeypatch.delenv(ALLOW_LEGACY_UNPROVENANCED_RESUME_ENV, raising=False)
    ppo = _checkpoint_load_stub()

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint") as load_mock,
        pytest.raises(ValueError, match="requires finalized current training provenance"),
    ):
        ppo.load("unprovenanced.pt")

    load_mock.assert_not_called()


def test_full_resume_legacy_provenance_hatch_must_be_exact(monkeypatch):
    monkeypatch.setenv(ALLOW_LEGACY_UNPROVENANCED_RESUME_ENV, "true")
    ppo = _checkpoint_load_stub()

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint") as load_mock,
        pytest.raises(ValueError, match="must be exactly 0 or 1"),
    ):
        ppo.load("unprovenanced.pt")

    load_mock.assert_not_called()


def test_full_resume_rejects_one_sided_provenance_even_with_legacy_hatch():
    ppo = _checkpoint_load_stub()
    checkpoint = {"training_provenance": {}}

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, "0" * 64),
        ),
        pytest.raises(ValueError, match="present on exactly one side"),
    ):
        ppo.load("one-sided.pt")


def test_full_resume_rejects_current_only_provenance_even_with_legacy_hatch():
    ppo = _checkpoint_load_stub()
    ppo._training_provenance = {
        "training_resume_enabled": True,
        "training_resume_sha256": "0" * 64,
    }

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=({}, "0" * 64),
        ),
        pytest.raises(ValueError, match="present on exactly one side"),
    ):
        ppo.load("current-only.pt")


def test_full_resume_legacy_hatch_cannot_downgrade_paired_provenance():
    ppo = _checkpoint_load_stub()
    ppo._training_provenance = {
        "training_resume_enabled": True,
        "training_resume_sha256": "0" * 64,
    }
    checkpoint = {"training_provenance": {}}

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, "0" * 64),
        ),
        pytest.raises(ValueError, match="also requires attached experiment_config"),
    ):
        ppo.load("paired-but-unconfigured.pt")


def test_full_resume_without_required_perception_state_fails_before_escape_hatch(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_ALLOW_FRESH_CURRICULUM_RESUME", "1")
    ppo = object.__new__(PPO)
    ppo.gpu_global_rank = 0
    ppo.gpu_world_size = 1
    ppo.is_multi_gpu = False
    ppo.env = SimpleNamespace(environment_state_checkpoint_required=True)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)

    with pytest.raises(RuntimeError, match="persistent perception.*Exact resume is impossible"):
        ppo._prepare_checkpoint_env_state({})


def test_full_resume_missing_rng_state_fails_before_model_mutation(monkeypatch):
    monkeypatch.delenv(ALLOW_NONDETERMINISTIC_RNG_RESUME_ENV, raising=False)
    ppo = _checkpoint_load_stub()
    ppo.current_learning_iteration = 17
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    original_weight = ppo.actor.weight.detach().clone()
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint.pop("rng_state_by_rank")
    checkpoint["actor_model_state_dict"] = {
        key: value.detach().clone() for key, value in checkpoint["actor_model_state_dict"].items()
    }
    checkpoint["actor_model_state_dict"]["weight"].fill_(7.0)

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(RuntimeError, match="no rng_state_by_rank"),
    ):
        ppo.load("legacy-without-rng.pt")

    assert torch.equal(ppo.actor.weight, original_weight)
    assert ppo.current_learning_iteration == 17


def test_full_resume_reauthenticates_checkpoint_after_preflight_path_drift(tmp_path):
    ppo = _checkpoint_load_stub()
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint_path = tmp_path / "resume.pt"
    torch.save(checkpoint, checkpoint_path)
    authenticated_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    ppo._training_provenance = {
        "training_resume_enabled": True,
        "training_resume_sha256": authenticated_sha256,
    }
    original_weight = ppo.actor.weight.detach().clone()

    # Simulate replacement after the launch-time preflight but before PPO.load.
    drifted = dict(checkpoint)
    drifted["actor_model_state_dict"] = {
        key: value.detach().clone()
        for key, value in checkpoint["actor_model_state_dict"].items()
    }
    drifted["actor_model_state_dict"]["weight"].fill_(7.0)
    torch.save(drifted, checkpoint_path)

    with pytest.raises(ValueError, match="does not match the authenticated training provenance"):
        ppo.load(str(checkpoint_path))

    assert torch.equal(ppo.actor.weight, original_weight)


def test_full_resume_enabled_provenance_cannot_omit_authenticated_digest():
    ppo = _checkpoint_load_stub()
    ppo._training_provenance = {"training_resume_enabled": True}

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint") as load_mock,
        pytest.raises(ValueError, match="no authenticated training_resume_sha256"),
    ):
        ppo.load("resume.pt")

    load_mock.assert_not_called()


def test_full_resume_restores_python_numpy_and_torch_rng_last(monkeypatch):
    monkeypatch.delenv(ALLOW_NONDETERMINISTIC_RNG_RESUME_ENV, raising=False)
    original_rng = capture_rng_checkpoint_state()
    try:
        ppo = _checkpoint_load_stub(load_optimizer=True)
        ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
        random.seed(111)
        np.random.seed(222)
        torch.manual_seed(333)
        checkpoint = _full_resume_checkpoint(ppo)
        expected = (random.random(), float(np.random.random()), torch.rand(3))

        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        with patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, "0" * 64),
        ) as load_mock:
            ppo.load("rng-complete.pt")

        load_mock.assert_called_once_with(
            "rng-complete.pt",
            expected_sha256=None,
            map_location="cpu",
        )
        assert random.random() == expected[0]
        assert float(np.random.random()) == expected[1]
        assert torch.equal(torch.rand(3), expected[2])
    finally:
        restore_rng_checkpoint_state(original_rng)


def test_distributed_peer_core_resume_failure_is_raised_before_local_mutation():
    ppo = _checkpoint_load_stub()
    ppo.is_multi_gpu = True
    ppo.gpu_world_size = 2
    ppo._gloo_barrier_group = object()
    ppo._setup_gloo_barrier_group = Mock(return_value=ppo._gloo_barrier_group)
    original_weight = ppo.actor.weight.detach().clone()
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint["actor_model_state_dict"] = {
        key: value.detach().clone()
        for key, value in checkpoint["actor_model_state_dict"].items()
    }
    checkpoint["actor_model_state_dict"]["weight"].fill_(7.0)

    def inject_peer_failure(gathered, local_result, **_kwargs):
        assert local_result == {"rank": 0, "error": None}
        gathered[:] = [
            {"rank": 0, "error": None},
            {"rank": 1, "error": "ValueError: corrupt critic state"},
        ]

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        patch("holosoma.agents.ppo.ppo.torch.distributed.is_available", return_value=True),
        patch("holosoma.agents.ppo.ppo.torch.distributed.is_initialized", return_value=True),
        patch("holosoma.agents.ppo.ppo.torch.distributed.get_world_size", return_value=2),
        patch("holosoma.agents.ppo.ppo.torch.distributed.get_rank", return_value=0),
        patch(
            "holosoma.agents.ppo.ppo.torch.distributed.all_gather_object",
            side_effect=inject_peer_failure,
        ),
        pytest.raises(RuntimeError, match="rank=1: ValueError: corrupt critic state"),
    ):
        ppo.load("peer-corrupt.pt")

    assert torch.equal(ppo.actor.weight, original_weight)


def test_distributed_peer_commit_failure_is_reported_before_model_sync():
    ppo = object.__new__(PPO)
    ppo.is_multi_gpu = True
    group = object()
    ppo._setup_gloo_barrier_group = Mock(return_value=group)

    def inject_peer_failure(gathered, local_result, **kwargs):
        assert kwargs == {"group": group}
        assert local_result == {"rank": 0, "error": None}
        gathered[:] = [
            local_result,
            {"rank": 1, "error": "RuntimeError: env restore failed"},
        ]

    with (
        patch("holosoma.agents.ppo.ppo.torch.distributed.is_available", return_value=True),
        patch("holosoma.agents.ppo.ppo.torch.distributed.is_initialized", return_value=True),
        patch("holosoma.agents.ppo.ppo.torch.distributed.get_world_size", return_value=2),
        patch("holosoma.agents.ppo.ppo.torch.distributed.get_rank", return_value=0),
        patch(
            "holosoma.agents.ppo.ppo.torch.distributed.all_gather_object",
            side_effect=inject_peer_failure,
        ),
        pytest.raises(
            RuntimeError,
            match=r"(?s)validated state commit.*rank=1: RuntimeError: env restore failed",
        ),
    ):
        ppo._synchronize_full_resume_validation_error(
            None,
            phase="validated state commit",
        )


def test_legacy_rng_resume_override_must_be_exact(monkeypatch):
    ppo = _checkpoint_load_stub(load_optimizer=True)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint.pop("rng_state_by_rank")

    monkeypatch.setenv(ALLOW_NONDETERMINISTIC_RNG_RESUME_ENV, "true")
    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match="must be exactly 0 or 1"),
    ):
        ppo.load("legacy-without-rng.pt")

    monkeypatch.setenv(ALLOW_NONDETERMINISTIC_RNG_RESUME_ENV, "1")
    with patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)):
        ppo.load("legacy-without-rng.pt")
    assert ppo.current_learning_iteration == 4


def test_full_resume_requires_enabled_normalizer_state():
    ppo = _checkpoint_load_stub(normalize_actor=True)

    with pytest.raises(ValueError, match="actor_obs_normalizer_state is missing"):
        ppo._restore_checkpoint_normalizers(
            {},
            kind="actor",
            runtime_enabled=True,
            normalizers=ppo.actor_obs_normalizers,
        )


def test_full_resume_rejects_normalizer_group_key_drift():
    ppo = _checkpoint_load_stub(normalize_actor=True)

    with pytest.raises(ValueError, match="extra=.*wrong"):
        ppo._restore_checkpoint_normalizers(
            {"actor_obs_normalizer_state": {"wrong": {}}},
            kind="actor",
            runtime_enabled=True,
            normalizers=ppo.actor_obs_normalizers,
        )


def test_policy_init_restores_enabled_actor_normalizer_state():
    ppo = _checkpoint_load_stub(normalize_actor=True)
    ppo.current_learning_iteration = 0
    checkpoint_normalizer = EmpiricalNormalization((1,), "cpu")
    checkpoint_normalizer._mean.fill_(3.0)
    checkpoint_normalizer._var.fill_(4.0)
    checkpoint_normalizer._std.fill_((4.0 + checkpoint_normalizer.eps) ** 0.5)
    checkpoint_normalizer.count.fill_(17)
    checkpoint = {
        "actor_model_state_dict": ppo.actor.state_dict(),
        "actor_obs_normalizer_state": {"actor_obs": checkpoint_normalizer.state_dict()},
        "experiment_config": {"algo": {"config": {"normalize_actor_obs": True}}},
        "iter": 12,
    }

    with patch(
        "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
        return_value=(checkpoint, "0" * 64),
    ):
        ppo.load_policy_init("normalized-policy.pt")

    restored = ppo.actor_obs_normalizers["actor_obs"]
    assert isinstance(restored, EmpiricalNormalization)
    assert restored.mean.item() == pytest.approx(3.0)
    assert restored.std.item() == pytest.approx((4.0 + checkpoint_normalizer.eps) ** 0.5)
    assert restored.count.item() == 17


def test_direct_policy_load_requires_runtime_semantic_contract_by_default(monkeypatch):
    monkeypatch.delenv(ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV, raising=False)
    ppo = _checkpoint_load_stub(normalize_actor=False)

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint") as load_mock,
        pytest.raises(ValueError, match="requires finalized current training provenance"),
    ):
        ppo.load_policy_init("unverified-policy.pt")

    load_mock.assert_not_called()


def test_policy_init_config_identity_does_not_replace_checkpoint_authentication(monkeypatch):
    monkeypatch.delenv(ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV, raising=False)
    ppo = _checkpoint_load_stub(normalize_actor=False)
    ppo._policy_load_runtime_config = SimpleNamespace(
        to_serializable_dict=lambda: {"runtime": "actor-contract"}
    )

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint") as load_mock,
        pytest.raises(ValueError, match="authenticated policy_init_sha256"),
    ):
        ppo.load_policy_init("unauthenticated-policy.pt")

    load_mock.assert_not_called()


def test_direct_policy_load_legacy_hatch_must_be_exact(monkeypatch):
    monkeypatch.setenv(ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV, "true")
    ppo = _checkpoint_load_stub(normalize_actor=False)

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint") as load_mock,
        pytest.raises(ValueError, match="must be exactly 0 or 1"),
    ):
        ppo.load_policy_init("unverified-policy.pt")

    load_mock.assert_not_called()


def test_evaluation_load_validates_actor_contract_and_retains_source_identity():
    ppo = _checkpoint_load_stub(normalize_actor=False)
    ppo.current_learning_iteration = 0
    ppo._policy_load_runtime_config = SimpleNamespace(
        to_serializable_dict=lambda: {"runtime": "actor-contract"}
    )
    ppo._training_provenance = {"unrelated_current_eval_process": True}
    actor_state = {
        key: value.detach().clone() for key, value in ppo.actor.state_dict().items()
    }
    actor_state["weight"].fill_(7.0)
    checkpoint = {
        "actor_model_state_dict": actor_state,
        "experiment_config": {"algo": {"config": {"normalize_actor_obs": False}}},
        "iter": 12,
        "next_iter": 13,
    }
    original_critic_weight = ppo.critic.weight.detach().clone()
    original_actor_optimizer_lr = ppo.actor_optimizer.param_groups[0]["lr"]
    original_rng = capture_rng_checkpoint_state()
    random.seed(1201)
    np.random.seed(1202)
    torch.manual_seed(1203)
    boundary = capture_rng_checkpoint_state()
    expected_rng = (random.random(), float(np.random.random()), torch.rand(3))
    restore_rng_checkpoint_state(boundary)

    def noisy_checkpoint_load(*_args, **_kwargs):
        random.random()
        np.random.random()
        torch.rand(3)
        return checkpoint, "a" * 64

    try:
        with (
            patch(
                "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
                side_effect=noisy_checkpoint_load,
            ),
            patch(
                "holosoma.agents.ppo.ppo.validate_policy_init_payload_identity"
            ) as identity_validator,
        ):
            ppo.load_evaluation("evaluation.pt")

        identity_validator.assert_called_once_with(
            checkpoint,
            {"runtime": "actor-contract"},
        )
        assert ppo.actor.weight.item() == pytest.approx(7.0)
        assert torch.equal(ppo.critic.weight, original_critic_weight)
        assert ppo.actor_optimizer.param_groups[0]["lr"] == original_actor_optimizer_lr
        ppo.env.load_checkpoint_state.assert_not_called()
        assert ppo.current_learning_iteration == 0
        assert ppo._evaluation_completed_iteration == 12
        assert ppo._source_checkpoint_sha256 == "a" * 64
        assert ppo._training_provenance is None
        assert ppo._source_experiment_config_dict == checkpoint["experiment_config"]
        assert random.random() == expected_rng[0]
        assert float(np.random.random()) == expected_rng[1]
        assert torch.equal(torch.rand(3), expected_rng[2])
    finally:
        restore_rng_checkpoint_state(original_rng)


def test_evaluation_load_passes_launcher_pinned_checkpoint_digest(monkeypatch):
    expected_sha256 = "7" * 64
    monkeypatch.setenv(
        "HOLOSOMA_EXPECTED_EVALUATION_CHECKPOINT_SHA256",
        expected_sha256,
    )
    ppo = _checkpoint_load_stub(normalize_actor=False)
    ppo._policy_load_runtime_config = SimpleNamespace(
        to_serializable_dict=lambda: {"runtime": "actor-contract"}
    )
    checkpoint = {
        "actor_model_state_dict": ppo.actor.state_dict(),
        "experiment_config": {"algo": {"config": {"normalize_actor_obs": False}}},
        "iter": 12,
        "next_iter": 13,
    }

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, expected_sha256),
        ) as load_mock,
        patch("holosoma.agents.ppo.ppo.validate_policy_init_payload_identity"),
    ):
        ppo.load_evaluation("evaluation.pt")

    load_mock.assert_called_once_with(
        "evaluation.pt",
        expected_sha256=expected_sha256,
        map_location="cpu",
    )


def test_evaluation_load_rejects_malformed_launcher_pinned_digest(monkeypatch):
    monkeypatch.setenv(
        "HOLOSOMA_EXPECTED_EVALUATION_CHECKPOINT_SHA256",
        "not-a-sha256",
    )
    ppo = _checkpoint_load_stub(normalize_actor=False)
    ppo._policy_load_runtime_config = SimpleNamespace(
        to_serializable_dict=lambda: {"runtime": "actor-contract"}
    )

    with pytest.raises(ValueError, match="Expected checkpoint SHA256"):
        ppo.load_evaluation("evaluation.pt")


def test_authenticated_legacy_evaluation_motion_contract_requires_pinned_digest(
    monkeypatch,
):
    monkeypatch.setenv(
        "HOLOSOMA_EVAL_ALLOW_AUTHENTICATED_LEGACY_MOTION_CONTRACT",
        "1",
    )
    monkeypatch.delenv(
        "HOLOSOMA_EXPECTED_EVALUATION_CHECKPOINT_SHA256",
        raising=False,
    )
    ppo = _checkpoint_load_stub(normalize_actor=False)
    ppo._policy_load_runtime_config = SimpleNamespace(
        to_serializable_dict=lambda: {"runtime": "actor-contract"}
    )

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint") as load_mock,
        pytest.raises(ValueError, match="requires HOLOSOMA_EXPECTED_EVALUATION"),
    ):
        ppo.load_evaluation("legacy-evaluation.pt")

    load_mock.assert_not_called()


def test_authenticated_legacy_evaluation_motion_contract_is_eval_only(
    monkeypatch,
):
    monkeypatch.setenv(
        "HOLOSOMA_EVAL_ALLOW_AUTHENTICATED_LEGACY_MOTION_CONTRACT",
        "1",
    )
    ppo = _checkpoint_load_stub(normalize_actor=False)

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint") as load_mock,
        pytest.raises(ValueError, match="predates motion_transition_contract"),
    ):
        ppo._validate_checkpoint_motion_transition_contract(
            {
                "experiment_config": {
                    "command": {
                        "setup_terms": {
                            "motion_command": {
                                "params": {
                                    "motion_config": {
                                        "enable_default_pose_append": True,
                                        "default_pose_append_duration_s": 2.0,
                                    }
                                }
                            }
                        }
                    }
                }
            },
            live_contract=None,
            compare_live=False,
            operation="Full resume",
        )

    load_mock.assert_not_called()


def test_authenticated_legacy_evaluation_motion_contract_accepts_exact_hash(
    monkeypatch,
):
    expected_sha256 = "7" * 64
    monkeypatch.setenv(
        "HOLOSOMA_EVAL_ALLOW_AUTHENTICATED_LEGACY_MOTION_CONTRACT",
        "1",
    )
    monkeypatch.setenv(
        "HOLOSOMA_EXPECTED_EVALUATION_CHECKPOINT_SHA256",
        expected_sha256,
    )
    ppo = _checkpoint_load_stub(normalize_actor=False)
    ppo._policy_load_runtime_config = SimpleNamespace(
        to_serializable_dict=lambda: {"runtime": "actor-contract"}
    )
    checkpoint = {
        "actor_model_state_dict": ppo.actor.state_dict(),
        "experiment_config": {
            "algo": {"config": {"normalize_actor_obs": False}},
            "command": {
                "setup_terms": {
                    "motion_command": {
                        "params": {
                            "motion_config": {
                                "enable_default_pose_append": True,
                                "default_pose_append_duration_s": 2.0,
                            }
                        }
                    }
                }
            },
        },
        "iter": 12,
        "next_iter": 13,
    }

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, expected_sha256),
        ) as load_mock,
        patch(
            "holosoma.agents.ppo.ppo.validate_policy_init_payload_identity"
        ) as identity_validator,
    ):
        ppo.load_evaluation("legacy-evaluation.pt")

    load_mock.assert_called_once_with(
        "legacy-evaluation.pt",
        expected_sha256=expected_sha256,
        map_location="cpu",
    )
    identity_validator.assert_called_once_with(
        checkpoint,
        {"runtime": "actor-contract"},
    )
    assert ppo._source_checkpoint_sha256 == expected_sha256


def test_authenticated_legacy_evaluation_motion_contract_rejects_modern_payload(
    monkeypatch,
):
    expected_sha256 = "7" * 64
    monkeypatch.setenv(
        "HOLOSOMA_EVAL_ALLOW_AUTHENTICATED_LEGACY_MOTION_CONTRACT",
        "1",
    )
    monkeypatch.setenv(
        "HOLOSOMA_EXPECTED_EVALUATION_CHECKPOINT_SHA256",
        expected_sha256,
    )
    ppo = _checkpoint_load_stub(normalize_actor=False)
    ppo._policy_load_runtime_config = SimpleNamespace(
        to_serializable_dict=lambda: {"runtime": "actor-contract"}
    )
    checkpoint = {
        "actor_model_state_dict": ppo.actor.state_dict(),
        "experiment_config": {"algo": {"config": {"normalize_actor_obs": False}}},
        "motion_transition_contract": _motion_transition_contract(),
        "motion_transition_contract_sha256": motion_transition_contract_sha256(
            _motion_transition_contract()
        ),
        "iter": 12,
        "next_iter": 13,
    }

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, expected_sha256),
        ),
        pytest.raises(ValueError, match="only valid for an authenticated legacy"),
    ):
        ppo.load_evaluation("modern-evaluation.pt")


def test_evaluation_load_rejects_malformed_source_provenance_before_mutation():
    ppo = _checkpoint_load_stub(normalize_actor=False)
    ppo._policy_load_runtime_config = SimpleNamespace(
        to_serializable_dict=lambda: {"runtime": "actor-contract"}
    )
    original_weight = ppo.actor.weight.detach().clone()
    actor_state = {
        key: value.detach().clone() for key, value in ppo.actor.state_dict().items()
    }
    actor_state["weight"].fill_(7.0)
    checkpoint = {
        "actor_model_state_dict": actor_state,
        "experiment_config": {"algo": {"config": {"normalize_actor_obs": False}}},
        "training_provenance": {},
        "iter": 12,
        "next_iter": 13,
    }
    original_rng = capture_rng_checkpoint_state()
    random.seed(1301)
    np.random.seed(1302)
    torch.manual_seed(1303)
    boundary = capture_rng_checkpoint_state()
    expected_rng = (random.random(), float(np.random.random()), torch.rand(3))
    restore_rng_checkpoint_state(boundary)

    def noisy_checkpoint_load(*_args, **_kwargs):
        random.random()
        np.random.random()
        torch.rand(3)
        return checkpoint, "a" * 64

    try:
        with (
            patch(
                "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
                side_effect=noisy_checkpoint_load,
            ),
            patch("holosoma.agents.ppo.ppo.validate_policy_init_payload_identity"),
            pytest.raises(ValueError, match="unsupported training provenance version"),
        ):
            ppo.load_evaluation("evaluation.pt")

        assert torch.equal(ppo.actor.weight, original_weight)
        assert random.random() == expected_rng[0]
        assert float(np.random.random()) == expected_rng[1]
        assert torch.equal(torch.rand(3), expected_rng[2])
    finally:
        restore_rng_checkpoint_state(original_rng)


def test_evaluation_load_requires_attached_runtime_contract_by_default(monkeypatch):
    monkeypatch.delenv(ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV, raising=False)
    ppo = _checkpoint_load_stub(normalize_actor=False)

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint") as load_mock,
        pytest.raises(ValueError, match="Evaluation policy load requires an attached runtime"),
    ):
        ppo.load_evaluation("evaluation.pt")

    load_mock.assert_not_called()


def test_evaluation_load_preserves_valid_source_provenance_for_publication():
    ppo = _checkpoint_load_stub(normalize_actor=False)
    ppo.current_learning_iteration = 0
    ppo._policy_load_runtime_config = SimpleNamespace(
        to_serializable_dict=lambda: {"runtime": "actor-contract"}
    )
    ppo._experiment_config = SimpleNamespace(
        to_serializable_dict=lambda: {"stale": "source-config"}
    )
    ppo._wandb_run_path = None
    source_provenance = _finalized_test_training_provenance()
    source_transition_contract = _motion_transition_contract()
    checkpoint = {
        "actor_model_state_dict": ppo.actor.state_dict(),
        "experiment_config": {"algo": {"config": {"normalize_actor_obs": False}}},
        "training_provenance": source_provenance,
        "motion_transition_contract": source_transition_contract,
        "motion_transition_contract_sha256": motion_transition_contract_sha256(
            source_transition_contract
        ),
        "iter": 22,
        "next_iter": 23,
    }

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, "9" * 64),
        ),
        patch("holosoma.agents.ppo.ppo.validate_policy_init_payload_identity"),
    ):
        ppo.load_evaluation("evaluation.pt")

    expected_provenance = validate_training_provenance(
        source_provenance,
        require_finalized=True,
    )
    assert ppo._training_provenance == expected_provenance
    metadata = ppo._checkpoint_metadata(iteration=22)
    assert metadata["training_provenance"] == expected_provenance
    assert metadata["source_checkpoint_sha256"] == "9" * 64
    assert metadata["experiment_config"] == checkpoint["experiment_config"]
    assert metadata["motion_transition_contract"] == source_transition_contract
    assert metadata["motion_transition_contract_sha256"] == motion_transition_contract_sha256(
        source_transition_contract
    )
    assert metadata["iteration"] == 22


def test_evaluation_load_restores_only_actor_normalizer_state():
    ppo = _checkpoint_load_stub(normalize_actor=True, normalize_critic=True)
    ppo._policy_load_runtime_config = SimpleNamespace(
        to_serializable_dict=lambda: {"runtime": "actor-contract"}
    )
    checkpoint_actor_normalizer = EmpiricalNormalization((1,), "cpu")
    checkpoint_actor_normalizer._mean.fill_(3.0)
    checkpoint_actor_normalizer._var.fill_(4.0)
    checkpoint_actor_normalizer._std.fill_((4.0 + checkpoint_actor_normalizer.eps) ** 0.5)
    checkpoint_actor_normalizer.count.fill_(17)
    checkpoint_critic_normalizer = EmpiricalNormalization((1,), "cpu")
    checkpoint_critic_normalizer._mean.fill_(99.0)
    critic_before = {
        key: value.detach().clone()
        for key, value in ppo.critic_obs_normalizers["critic_obs"].state_dict().items()
    }
    checkpoint = {
        "actor_model_state_dict": ppo.actor.state_dict(),
        "actor_obs_normalizer_state": {
            "actor_obs": checkpoint_actor_normalizer.state_dict()
        },
        "critic_obs_normalizer_state": {
            "critic_obs": checkpoint_critic_normalizer.state_dict()
        },
        "experiment_config": {"algo": {"config": {"normalize_actor_obs": True}}},
        "iter": 32,
        "next_iter": 33,
    }

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, "8" * 64),
        ),
        patch("holosoma.agents.ppo.ppo.validate_policy_init_payload_identity"),
    ):
        ppo.load_evaluation("normalized-evaluation.pt")

    actor_normalizer = ppo.actor_obs_normalizers["actor_obs"]
    assert actor_normalizer.mean.item() == pytest.approx(3.0)
    assert actor_normalizer.count.item() == 17
    critic_after = ppo.critic_obs_normalizers["critic_obs"].state_dict()
    for key, expected in critic_before.items():
        assert torch.equal(critic_after[key], expected)


def test_policy_init_enabled_provenance_cannot_omit_authenticated_digest():
    ppo = _checkpoint_load_stub(normalize_actor=False)
    ppo._training_provenance = {"policy_init_enabled": True}

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint") as load_mock,
        pytest.raises(ValueError, match="no authenticated policy_init_sha256"),
    ):
        ppo.load_policy_init("policy-init.pt")

    load_mock.assert_not_called()


def test_policy_init_required_terminal_target_rejects_missing_terminal_payload(
    monkeypatch,
):
    ppo = _checkpoint_load_stub(normalize_actor=False)
    checkpoint = {
        "actor_model_state_dict": ppo.actor.state_dict(),
        "experiment_config": {"algo": {"config": {"normalize_actor_obs": False}}},
    }
    monkeypatch.setenv(POLICY_INIT_REQUIRED_TERMINAL_TARGET_ENV, "8")

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, "0" * 64),
        ),
        pytest.raises(ValueError, match="missing explicit 'iter'"),
    ):
        ppo.load_policy_init("policy-init.pt")


def test_policy_init_required_terminal_target_alias_fails_before_checkpoint_load(
    monkeypatch,
):
    ppo = _checkpoint_load_stub(normalize_actor=False)
    monkeypatch.setenv(POLICY_INIT_REQUIRED_TERMINAL_TARGET_ENV, "08")

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint") as load_mock,
        pytest.raises(ValueError, match="canonical ASCII positive integer"),
    ):
        ppo.load_policy_init("policy-init.pt")

    load_mock.assert_not_called()


def test_distributed_policy_init_peer_validation_failure_precedes_local_mutation():
    ppo = _checkpoint_load_stub(normalize_actor=False)
    ppo.is_multi_gpu = True
    ppo.gpu_world_size = 2
    group = object()
    ppo._setup_gloo_barrier_group = Mock(return_value=group)
    ppo._synchronize_model_weights = Mock()
    original_weight = ppo.actor.weight.detach().clone()
    actor_state = {
        key: value.detach().clone() for key, value in ppo.actor.state_dict().items()
    }
    actor_state["weight"].fill_(7.0)
    checkpoint = {
        "actor_model_state_dict": actor_state,
        "experiment_config": {"algo": {"config": {"normalize_actor_obs": False}}},
    }

    def inject_peer_failure(gathered, local_result, **kwargs):
        assert kwargs == {"group": group}
        assert local_result == {"rank": 0, "error": None}
        gathered[:] = [
            local_result,
            {"rank": 1, "error": "OSError: rank-local policy checkpoint read failed"},
        ]

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, "0" * 64),
        ),
        patch("holosoma.agents.ppo.ppo.torch.distributed.is_available", return_value=True),
        patch("holosoma.agents.ppo.ppo.torch.distributed.is_initialized", return_value=True),
        patch("holosoma.agents.ppo.ppo.torch.distributed.get_world_size", return_value=2),
        patch("holosoma.agents.ppo.ppo.torch.distributed.get_rank", return_value=0),
        patch(
            "holosoma.agents.ppo.ppo.torch.distributed.all_gather_object",
            side_effect=inject_peer_failure,
        ),
        pytest.raises(
            RuntimeError,
            match="rank=1: OSError: rank-local policy checkpoint read failed",
        ),
    ):
        ppo.load_policy_init("policy-init.pt")

    assert torch.equal(ppo.actor.weight, original_weight)
    ppo._synchronize_model_weights.assert_not_called()


def test_distributed_policy_init_peer_commit_failure_precedes_model_broadcast():
    ppo = _checkpoint_load_stub(normalize_actor=False)
    ppo.is_multi_gpu = True
    ppo.gpu_world_size = 2
    group = object()
    ppo._setup_gloo_barrier_group = Mock(return_value=group)
    ppo._synchronize_model_weights = Mock()
    checkpoint = {
        "actor_model_state_dict": {
            key: value.detach().clone() for key, value in ppo.actor.state_dict().items()
        },
        "experiment_config": {"algo": {"config": {"normalize_actor_obs": False}}},
    }
    gather_count = 0

    def inject_second_phase_failure(gathered, local_result, **kwargs):
        nonlocal gather_count
        assert kwargs == {"group": group}
        gather_count += 1
        peer_error = (
            None
            if gather_count == 1
            else "RuntimeError: rank-local actor device commit failed"
        )
        gathered[:] = [
            local_result,
            {"rank": 1, "error": peer_error},
        ]

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, "0" * 64),
        ),
        patch("holosoma.agents.ppo.ppo.torch.distributed.is_available", return_value=True),
        patch("holosoma.agents.ppo.ppo.torch.distributed.is_initialized", return_value=True),
        patch("holosoma.agents.ppo.ppo.torch.distributed.get_world_size", return_value=2),
        patch("holosoma.agents.ppo.ppo.torch.distributed.get_rank", return_value=0),
        patch(
            "holosoma.agents.ppo.ppo.torch.distributed.all_gather_object",
            side_effect=inject_second_phase_failure,
        ),
        pytest.raises(
            RuntimeError,
            match=r"(?s)Policy-init validated state commit.*rank=1: RuntimeError: rank-local actor device",
        ),
    ):
        ppo.load_policy_init("policy-init.pt")

    assert gather_count == 2
    ppo._synchronize_model_weights.assert_not_called()


def test_policy_init_direct_load_rejects_non_finite_actor_before_mutation():
    ppo = _checkpoint_load_stub(normalize_actor=False)
    ppo.current_learning_iteration = 0
    original_weight = ppo.actor.weight.detach().clone()
    actor_state = {key: value.detach().clone() for key, value in ppo.actor.state_dict().items()}
    actor_state["weight"].fill_(float("nan"))
    checkpoint = {
        "actor_model_state_dict": actor_state,
        "experiment_config": {"algo": {"config": {"normalize_actor_obs": False}}},
    }

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, "0" * 64),
        ),
        pytest.raises(ValueError, match="actor_model_state_dict.*non-finite"),
    ):
        ppo.load_policy_init("non-finite-policy.pt")

    assert torch.equal(ppo.actor.weight, original_weight)


def test_policy_init_rejects_out_of_domain_actor_std_before_mutation():
    ppo = _add_checkpoint_std_parameter(_checkpoint_load_stub(normalize_actor=False))
    ppo.current_learning_iteration = 0
    original_weight = ppo.actor.weight.detach().clone()
    original_std = ppo.actor.std.detach().clone()
    actor_state = {
        key: value.detach().clone() for key, value in ppo.actor.state_dict().items()
    }
    actor_state["weight"].fill_(7.0)
    actor_state["std"].fill_(-3.0)
    checkpoint = {
        "actor_model_state_dict": actor_state,
        "experiment_config": {"algo": {"config": {"normalize_actor_obs": False}}},
    }

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, "0" * 64),
        ),
        pytest.raises(ValueError, match=r"actor_model_state_dict\.std.*exact resume/policy init"),
    ):
        ppo.load_policy_init("invalid-std-policy.pt")

    assert torch.equal(ppo.actor.weight, original_weight)
    assert torch.equal(ppo.actor.std, original_std)


def test_policy_init_enabled_normalization_fails_when_state_is_missing():
    ppo = _checkpoint_load_stub(normalize_actor=True)
    ppo.current_learning_iteration = 0
    checkpoint = {
        "actor_model_state_dict": ppo.actor.state_dict(),
        "experiment_config": {"algo": {"config": {"normalize_actor_obs": True}}},
    }

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, "0" * 64),
        ),
        pytest.raises(ValueError, match="actor_obs_normalizer_state is missing"),
    ):
        ppo.load_policy_init("missing-normalizer.pt")


def test_policy_init_direct_load_requires_experiment_config():
    ppo = _checkpoint_load_stub(normalize_actor=False)
    ppo.current_learning_iteration = 0
    checkpoint = {"actor_model_state_dict": ppo.actor.state_dict()}

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, "0" * 64),
        ),
        pytest.raises(ValueError, match="no serialized experiment_config"),
    ):
        ppo.load_policy_init("unverifiable-policy.pt")


def test_policy_init_direct_load_requires_explicit_normalization_contract():
    ppo = _checkpoint_load_stub(normalize_actor=False)
    ppo.current_learning_iteration = 0
    checkpoint = {
        "actor_model_state_dict": ppo.actor.state_dict(),
        "experiment_config": {"algo": {"config": {}}},
    }

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, "0" * 64),
        ),
        pytest.raises(ValueError, match="must declare a boolean normalize_actor_obs"),
    ):
        ppo.load_policy_init("missing-normalization-contract.pt")


def test_policy_init_disabled_normalization_keeps_identity_behavior():
    ppo = _checkpoint_load_stub(normalize_actor=False)
    ppo.current_learning_iteration = 0
    original_normalizer = ppo.actor_obs_normalizers["actor_obs"]
    checkpoint = {
        "actor_model_state_dict": ppo.actor.state_dict(),
        # Identity normalizer state is intentionally ignored in False/False mode.
        "actor_obs_normalizer_state": {"unexpected": {"bad": torch.tensor(1)}},
        "experiment_config": {"algo": {"config": {"normalize_actor_obs": False}}},
    }

    with patch(
        "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
        return_value=(checkpoint, "0" * 64),
    ):
        ppo.load_policy_init("unnormalized-policy.pt")

    assert ppo.actor_obs_normalizers["actor_obs"] is original_normalizer


def test_policy_init_rejects_checkpoint_normalization_when_runtime_disables_it_before_mutation():
    ppo = _checkpoint_load_stub(normalize_actor=False)
    ppo.current_learning_iteration = 0
    original_weight = ppo.actor.weight.detach().clone()
    checkpoint_normalizer = EmpiricalNormalization((1,), "cpu")
    checkpoint = {
        "actor_model_state_dict": {
            key: value.detach().clone() for key, value in ppo.actor.state_dict().items()
        },
        "actor_obs_normalizer_state": {"actor_obs": checkpoint_normalizer.state_dict()},
        "experiment_config": {"algo": {"config": {"normalize_actor_obs": True}}},
    }
    checkpoint["actor_model_state_dict"]["weight"].fill_(7.0)

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, "0" * 64),
        ),
        pytest.raises(ValueError, match="Policy-init normalization mismatch for actor"),
    ):
        ppo.load_policy_init("normalization-mismatch.pt")

    assert torch.equal(ppo.actor.weight, original_weight)


def test_full_resume_load_optimizer_requires_both_states():
    ppo = _checkpoint_load_stub(load_optimizer=True)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    checkpoint = {
        "actor_model_state_dict": ppo.actor.state_dict(),
        "critic_model_state_dict": ppo.critic.state_dict(),
        "iter": 3,
    }

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match="missing optimizer state"),
    ):
        ppo.load("legacy.pt")


def test_full_resume_rejects_optimizer_reset_as_non_exact_continuation():
    ppo = _checkpoint_load_stub(load_optimizer=False)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    checkpoint = _full_resume_checkpoint(ppo)

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, "0" * 64),
        ),
        pytest.raises(ValueError, match="Exact training resume requires config.load_optimizer=True"),
    ):
        ppo.load("optimizer-reset.pt")


def test_full_resume_rejects_non_finite_model_before_mutation():
    ppo = _checkpoint_load_stub(load_optimizer=False)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    original_weight = ppo.actor.weight.detach().clone()
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint["actor_model_state_dict"] = {
        key: value.detach().clone() for key, value in checkpoint["actor_model_state_dict"].items()
    }
    checkpoint["actor_model_state_dict"]["weight"].fill_(float("inf"))

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match="actor_model_state_dict.*non-finite"),
    ):
        ppo.load("non-finite-model.pt")

    assert torch.equal(ppo.actor.weight, original_weight)


def test_full_resume_rejects_out_of_domain_actor_std_before_mutation():
    ppo = _add_checkpoint_std_parameter(_checkpoint_load_stub(load_optimizer=False))
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    original_weight = ppo.actor.weight.detach().clone()
    original_std = ppo.actor.std.detach().clone()
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint["actor_model_state_dict"] = {
        key: value.detach().clone()
        for key, value in checkpoint["actor_model_state_dict"].items()
    }
    checkpoint["actor_model_state_dict"]["weight"].fill_(7.0)
    checkpoint["actor_model_state_dict"]["std"].fill_(-3.0)

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match=r"actor_model_state_dict\.std.*exact resume/policy init"),
    ):
        ppo.load("invalid-std-resume.pt")

    assert torch.equal(ppo.actor.weight, original_weight)
    assert torch.equal(ppo.actor.std, original_std)


def test_full_resume_rejects_shape_incompatible_model_before_partial_mutation():
    ppo = _checkpoint_load_stub(load_optimizer=False)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    original_weight = ppo.actor.weight.detach().clone()
    original_bias = ppo.actor.bias.detach().clone()
    checkpoint = _full_resume_checkpoint(ppo)
    actor_state = {
        key: value.detach().clone() for key, value in checkpoint["actor_model_state_dict"].items()
    }
    actor_state["weight"].fill_(7.0)
    actor_state["bias"] = torch.zeros(2)
    checkpoint["actor_model_state_dict"] = actor_state

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match="actor_model_state_dict.*shape"),
    ):
        ppo.load("shape-incompatible-model.pt")

    assert torch.equal(ppo.actor.weight, original_weight)
    assert torch.equal(ppo.actor.bias, original_bias)


def test_full_resume_direct_load_rejects_inconsistent_iteration_before_mutation():
    ppo = _checkpoint_load_stub(load_optimizer=False)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    original_weight = ppo.actor.weight.detach().clone()
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint["next_iter"] = 5

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match=r"explicit next_iter must equal iter \+ 1"),
    ):
        ppo.load("inconsistent-iteration.pt")

    assert torch.equal(ppo.actor.weight, original_weight)


def test_full_resume_rejects_invalid_environment_state_before_model_and_iteration_mutation():
    ppo = _checkpoint_load_stub(load_optimizer=False)
    ppo.current_learning_iteration = 17
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    ppo.env = SimpleNamespace(
        validate_checkpoint_state=Mock(side_effect=ValueError("invalid AS sampler state")),
        load_checkpoint_state=Mock(),
    )
    original_weight = ppo.actor.weight.detach().clone()
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint["actor_model_state_dict"] = {
        key: value.detach().clone() for key, value in checkpoint["actor_model_state_dict"].items()
    }
    checkpoint["actor_model_state_dict"]["weight"].fill_(7.0)
    checkpoint["env_state"] = {"version": 1, "motion_command": {"corrupt": True}}

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match="invalid AS sampler state"),
    ):
        ppo.load("invalid-env-state.pt")

    assert torch.equal(ppo.actor.weight, original_weight)
    assert ppo.current_learning_iteration == 17
    ppo.env.load_checkpoint_state.assert_not_called()


def test_full_resume_rejects_invalid_fixed_bc_state_before_model_and_iteration_mutation():
    ppo = _checkpoint_load_stub(load_optimizer=False)
    ppo.current_learning_iteration = 17
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    ppo.fixed_bc_eval_num_samples = 1
    ppo.dagger_enabled = True
    ppo.actor_obs_keys = ["actor_obs"]
    ppo.algo_obs_dim_dict = {"actor_obs": 1}
    ppo.num_act = 1
    original_weight = ppo.actor.weight.detach().clone()
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint["actor_model_state_dict"] = {
        key: value.detach().clone() for key, value in checkpoint["actor_model_state_dict"].items()
    }
    checkpoint["actor_model_state_dict"]["weight"].fill_(7.0)
    checkpoint["fixed_bc_eval_by_rank"] = {
        "0": {
            "ready": True,
            "size": 1,
            "allocation_version": 1,
            "allocation_scheme": "rank_quotient_remainder",
            "global_sample_budget": 1,
            "world_size": 1,
            "rank": 0,
            "local_target": 1,
            "actor_obs_raw": torch.zeros(1, 1, dtype=torch.float64),
            "teacher_actions": torch.zeros(1, 1),
        }
    }

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match="actor_obs_raw.*dtype torch.float64"),
    ):
        ppo.load("invalid-fixed-bc-state.pt")

    assert torch.equal(ppo.actor.weight, original_weight)
    assert ppo.current_learning_iteration == 17


def test_full_resume_rejects_non_finite_enabled_normalizer_before_mutation():
    ppo = _checkpoint_load_stub(normalize_actor=True, load_optimizer=False)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    original_weight = ppo.actor.weight.detach().clone()
    normalizer_state = ppo.actor_obs_normalizers["actor_obs"].state_dict()
    normalizer_state["_mean"] = normalizer_state["_mean"].clone().fill_(float("nan"))
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint["actor_obs_normalizer_state"] = {"actor_obs": normalizer_state}
    checkpoint["experiment_config"] = {"algo": {"config": {"normalize_actor_obs": True}}}

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match="actor_obs_normalizer_state.*non-finite"),
    ):
        ppo.load("non-finite-normalizer.pt")

    assert torch.equal(ppo.actor.weight, original_weight)


def test_full_resume_rejects_shape_incompatible_normalizer_before_actor_mutation():
    ppo = _checkpoint_load_stub(normalize_actor=True, load_optimizer=False)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    original_weight = ppo.actor.weight.detach().clone()
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint["actor_model_state_dict"] = {
        key: value.detach().clone() for key, value in checkpoint["actor_model_state_dict"].items()
    }
    checkpoint["actor_model_state_dict"]["weight"].fill_(7.0)
    normalizer_state = {
        key: value.detach().clone()
        for key, value in ppo.actor_obs_normalizers["actor_obs"].state_dict().items()
    }
    normalizer_state["_mean"] = torch.zeros(2)
    checkpoint["actor_obs_normalizer_state"] = {"actor_obs": normalizer_state}
    checkpoint["experiment_config"] = {"algo": {"config": {"normalize_actor_obs": True}}}

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match="actor_obs_normalizer_state.*shape"),
    ):
        ppo.load("shape-incompatible-normalizer.pt")

    assert torch.equal(ppo.actor.weight, original_weight)


@pytest.mark.parametrize(
    ("field", "invalid_value", "expected"),
    [
        ("count", -1.0, "count must be finite and non-negative"),
        ("_var", -1.0, "_var must be a non-negative"),
        ("_std", -1.0, "_std must be a positive"),
    ],
)
def test_full_resume_rejects_invalid_normalizer_statistics_before_actor_mutation(
    field, invalid_value, expected
):
    ppo = _checkpoint_load_stub(normalize_actor=True, load_optimizer=False)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    original_weight = ppo.actor.weight.detach().clone()
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint["actor_model_state_dict"] = {
        key: value.detach().clone() for key, value in checkpoint["actor_model_state_dict"].items()
    }
    checkpoint["actor_model_state_dict"]["weight"].fill_(7.0)
    normalizer_state = {
        key: value.detach().clone()
        for key, value in ppo.actor_obs_normalizers["actor_obs"].state_dict().items()
    }
    normalizer_state[field].fill_(invalid_value)
    checkpoint["actor_obs_normalizer_state"] = {"actor_obs": normalizer_state}
    checkpoint["experiment_config"] = {"algo": {"config": {"normalize_actor_obs": True}}}

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match=expected),
    ):
        ppo.load("invalid-normalizer-statistics.pt")

    assert torch.equal(ppo.actor.weight, original_weight)


def test_full_resume_accepts_legacy_integer_normalizer_count():
    ppo = _checkpoint_load_stub(normalize_actor=True, load_optimizer=True)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    checkpoint = _full_resume_checkpoint(ppo)
    normalizer_state = {
        key: value.detach().clone()
        for key, value in ppo.actor_obs_normalizers["actor_obs"].state_dict().items()
    }
    normalizer_state["count"] = torch.tensor(0, dtype=torch.int64)
    checkpoint["actor_obs_normalizer_state"] = {"actor_obs": normalizer_state}
    checkpoint["experiment_config"] = {"algo": {"config": {"normalize_actor_obs": True}}}

    with patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)):
        ppo.load("legacy-integer-normalizer-count.pt")

    assert ppo.actor_obs_normalizers["actor_obs"].count.dtype == torch.float64
    assert ppo.actor_obs_normalizers["actor_obs"].count.item() == 0.0


@pytest.mark.parametrize(
    ("role", "invalid_lr", "expected"),
    [
        ("actor", float("nan"), "finite and > 0"),
        ("actor", float("inf"), "finite and > 0"),
        ("actor", 0.0, "finite and > 0"),
        ("critic", -1.0, "finite and > 0"),
        ("actor", 1.0e-6, "outside the configured bounds"),
        ("critic", 0.2, "outside the configured bounds"),
    ],
)
def test_full_resume_rejects_invalid_optimizer_lr_before_mutation(role, invalid_lr, expected):
    ppo = _checkpoint_load_stub(load_optimizer=True)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    original_weight = ppo.actor.weight.detach().clone()
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint[f"{role}_optimizer_state_dict"]["param_groups"][0]["lr"] = invalid_lr

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match=expected),
    ):
        ppo.load("invalid-lr.pt")

    assert torch.equal(ppo.actor.weight, original_weight)


def test_full_resume_validates_lr_in_every_optimizer_param_group():
    ppo = _checkpoint_load_stub(load_optimizer=True)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    checkpoint = _full_resume_checkpoint(ppo)
    extra_group = dict(checkpoint["actor_optimizer_state_dict"]["param_groups"][0])
    extra_group["params"] = []
    extra_group["lr"] = float("nan")
    checkpoint["actor_optimizer_state_dict"]["param_groups"].append(extra_group)

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match=r"param_groups\[1\]\.lr must be finite and > 0"),
    ):
        ppo.load("invalid-second-group-lr.pt")


def test_full_resume_rejects_optimizer_topology_before_actor_mutation():
    ppo = _checkpoint_load_stub(load_optimizer=True)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    original_weight = ppo.actor.weight.detach().clone()
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint["actor_model_state_dict"] = {
        key: value.detach().clone() for key, value in checkpoint["actor_model_state_dict"].items()
    }
    checkpoint["actor_model_state_dict"]["weight"].fill_(7.0)
    checkpoint["actor_optimizer_state_dict"]["param_groups"][0]["params"] = [0]

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match="has 1 parameters.*runtime group has 2"),
    ):
        ppo.load("optimizer-topology-mismatch.pt")

    assert torch.equal(ppo.actor.weight, original_weight)


def test_full_resume_rejects_optimizer_option_drift_before_actor_mutation():
    ppo = _checkpoint_load_stub(load_optimizer=True)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    original_weight = ppo.actor.weight.detach().clone()
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint["actor_model_state_dict"] = {
        key: value.detach().clone() for key, value in checkpoint["actor_model_state_dict"].items()
    }
    checkpoint["actor_model_state_dict"]["weight"].fill_(7.0)
    checkpoint["actor_optimizer_state_dict"]["param_groups"][0]["momentum"] = 2.0

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match="optimizer options differ.*momentum"),
    ):
        ppo.load("optimizer-option-drift.pt")

    assert torch.equal(ppo.actor.weight, original_weight)


def test_full_resume_rejects_optimizer_moment_dtype_before_overflowing_cast():
    ppo = _checkpoint_load_stub(load_optimizer=True, optimizer_momentum=0.9)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    original_weight = ppo.actor.weight.detach().clone()
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint["actor_model_state_dict"] = {
        key: value.detach().clone() for key, value in checkpoint["actor_model_state_dict"].items()
    }
    checkpoint["actor_model_state_dict"]["weight"].fill_(7.0)
    checkpoint["actor_optimizer_state_dict"]["state"] = {
        0: {"momentum_buffer": torch.full_like(ppo.actor.weight, 1.0e300, dtype=torch.float64)}
    }

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match="momentum_buffer.*dtype"),
    ):
        ppo.load("optimizer-moment-dtype.pt")

    assert torch.equal(ppo.actor.weight, original_weight)


def test_full_resume_rejects_negative_adam_step_before_actor_mutation():
    ppo = _checkpoint_load_stub(load_optimizer=True)
    _replace_stub_optimizers_with_initialized_adamw(ppo)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    original_weight = ppo.actor.weight.detach().clone()
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint["actor_model_state_dict"] = {
        key: value.detach().clone() for key, value in checkpoint["actor_model_state_dict"].items()
    }
    checkpoint["actor_model_state_dict"]["weight"].fill_(7.0)
    first_state = next(iter(checkpoint["actor_optimizer_state_dict"]["state"].values()))
    first_state["step"] = first_state["step"].new_tensor(-1.0)

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match="step must be finite, integral, and non-negative"),
    ):
        ppo.load("negative-adam-step.pt")

    assert torch.equal(ppo.actor.weight, original_weight)


def test_full_resume_rejects_incomplete_adam_state_before_actor_mutation():
    ppo = _checkpoint_load_stub(load_optimizer=True)
    _replace_stub_optimizers_with_initialized_adamw(ppo)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    original_weight = ppo.actor.weight.detach().clone()
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint["actor_model_state_dict"] = {
        key: value.detach().clone() for key, value in checkpoint["actor_model_state_dict"].items()
    }
    checkpoint["actor_model_state_dict"]["weight"].fill_(7.0)
    first_state = next(iter(checkpoint["actor_optimizer_state_dict"]["state"].values()))
    del first_state["exp_avg_sq"]

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match="Adam state keys are invalid.*exp_avg_sq"),
    ):
        ppo.load("incomplete-adam-state.pt")

    assert torch.equal(ppo.actor.weight, original_weight)


def test_full_resume_rejects_scalar_adam_moment_for_one_element_parameter():
    ppo = _checkpoint_load_stub(load_optimizer=True)
    _replace_stub_optimizers_with_initialized_adamw(ppo)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    original_weight = ppo.actor.weight.detach().clone()
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint["actor_model_state_dict"] = {
        key: value.detach().clone() for key, value in checkpoint["actor_model_state_dict"].items()
    }
    checkpoint["actor_model_state_dict"]["weight"].fill_(7.0)
    bias_state = checkpoint["actor_optimizer_state_dict"]["state"][1]
    assert bias_state["exp_avg"].shape == torch.Size([1])
    bias_state["exp_avg"] = bias_state["exp_avg"].squeeze(0)

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match="exp_avg shape.*parameter shape"),
    ):
        ppo.load("scalar-adam-moment.pt")

    assert torch.equal(ppo.actor.weight, original_weight)


def test_full_resume_rejects_reordered_optimizer_parameter_ids_before_actor_mutation():
    ppo = _checkpoint_load_stub(load_optimizer=True)
    _replace_stub_optimizers_with_initialized_adamw(ppo)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    original_weight = ppo.actor.weight.detach().clone()
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint["actor_model_state_dict"] = {
        key: value.detach().clone() for key, value in checkpoint["actor_model_state_dict"].items()
    }
    checkpoint["actor_model_state_dict"]["weight"].fill_(7.0)
    saved_parameters = checkpoint["actor_optimizer_state_dict"]["param_groups"][0]["params"]
    assert saved_parameters == [0, 1]
    saved_parameters[:] = [1, 0]

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match="params order/ids.*canonical order/ids"),
    ):
        ppo.load("reordered-optimizer-parameters.pt")

    assert torch.equal(ppo.actor.weight, original_weight)


def test_full_resume_rejects_non_finite_optimizer_tensor_state():
    ppo = _checkpoint_load_stub(load_optimizer=True)
    ppo._curriculum_state_required_for_resume = MethodType(lambda self: False, ppo)
    checkpoint = _full_resume_checkpoint(ppo)
    checkpoint["actor_optimizer_state_dict"]["state"] = {
        0: {"momentum_buffer": torch.tensor([float("nan")])}
    }

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)),
        pytest.raises(ValueError, match="actor_optimizer_state_dict.*non-finite"),
    ):
        ppo.load("non-finite-optimizer-state.pt")


def test_full_resume_preserves_divergent_lrs_and_enters_ppo_boundary_once():
    source = _checkpoint_load_stub(load_optimizer=True)
    source.current_learning_iteration = 700
    source.is_main_process = True
    source.fixed_bc_eval_num_samples = 0
    source.env = SimpleNamespace(get_checkpoint_state=Mock(return_value={}))
    source.actor_learning_rate = 1.0e-5
    source.critic_learning_rate = 1.0e-3
    source.actor_optimizer.param_groups[0]["lr"] = source.actor_learning_rate
    source.critic_optimizer.param_groups[0]["lr"] = source.critic_learning_rate
    source._checkpoint_metadata = MethodType(
        lambda self, iteration: {"iteration": int(iteration)},
        source,
    )
    saved = {}
    source.logging_helper = SimpleNamespace(
        save_checkpoint_artifact=lambda checkpoint, path: saved.update(
            checkpoint=checkpoint,
            path=path,
        )
    )

    source.save("model_00700.pt", next_iteration=700)

    checkpoint = saved["checkpoint"]
    assert saved["path"] == "model_00700.pt"
    assert checkpoint["iter"] == 699
    assert checkpoint["next_iter"] == 700
    assert checkpoint["actor_optimizer_state_dict"]["param_groups"][0]["lr"] == pytest.approx(1.0e-5)
    assert checkpoint["critic_optimizer_state_dict"]["param_groups"][0]["lr"] == pytest.approx(1.0e-3)

    resumed = _checkpoint_load_stub(load_optimizer=True)
    resumed._curriculum_state_required_for_resume = MethodType(lambda self: False, resumed)
    resumed.dagger_enabled = True
    resumed.use_ppo_dagger_schedule = True
    resumed.ppo_start_epoch = 0
    resumed.dagger_end_epoch = 6300
    resumed.ppo_start_coeff = 0.0
    resumed.ppo_target_coeff = 0.9
    resumed.ppo_schedule_step_epochs = 700
    resumed.ppo_coeff = 0.0
    resumed._configured_bc_loss_coef = 1.0
    resumed.bc_loss_coef = 1.0
    resumed.switch_to_rl_after = -1

    with patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)):
        resumed.load("model_00700.pt")

    assert resumed.actor_learning_rate == pytest.approx(1.0e-5)
    assert resumed.actor_optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-5)
    assert resumed.critic_learning_rate == pytest.approx(1.0e-3)
    assert resumed.critic_optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)
    assert resumed.current_learning_iteration == 700
    assert resumed.ppo_coeff == pytest.approx(0.1)
    assert resumed._use_deterministic_student_actions() is False
    assert list(range(resumed.current_learning_iteration, 702)) == [700, 701]


def test_rank_local_adaptive_sampler_requires_checkpoint_state_without_cross_rank_sync():
    ppo = _checkpoint_load_stub()
    motion_command = SimpleNamespace(
        adaptive_timesteps_sampler=object(),
        _rank_local_shard_metadata={"rank": 0, "world_size": 2},
    )
    ppo.env = SimpleNamespace(
        curriculum_state_sync_enabled=False,
        command_manager=SimpleNamespace(
            get_state=lambda name: motion_command if name == "motion_command" else None
        ),
    )

    assert ppo._curriculum_state_sync_enabled() is False
    assert ppo._curriculum_state_required_for_resume() is True


def test_teacher_normalizer_uses_checkpoint_epsilon_and_until():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        normalize_actor_obs=False,
        obs_normalizer_eps=0.01,
        obs_normalizer_until=None,
    )
    enabled, eps, until = ppo._teacher_normalization_config(
        {
            "experiment_config": {
                "algo": {
                    "config": {
                        "normalize_actor_obs": True,
                        "obs_normalizer_eps": 0.125,
                        "obs_normalizer_until": 123,
                    }
                }
            }
        }
    )

    assert enabled is True
    assert eps == pytest.approx(0.125)
    assert until == 123


def test_strict_teacher_normalization_requires_explicit_boolean_contract() -> None:
    ppo = object.__new__(PPO)
    ppo.strict_teacher_load = True
    ppo.config = SimpleNamespace(
        normalize_actor_obs=False,
        obs_normalizer_eps=0.01,
        obs_normalizer_until=None,
    )

    with pytest.raises(ValueError, match="boolean normalize_actor_obs"):
        ppo._teacher_normalization_config(
            {"experiment_config": {"algo": {"config": {}}}}
        )


def test_strict_normalized_teacher_requires_checkpoint_epsilon() -> None:
    ppo = object.__new__(PPO)
    ppo.strict_teacher_load = True
    ppo.config = SimpleNamespace(
        normalize_actor_obs=False,
        obs_normalizer_eps=0.01,
        obs_normalizer_until=None,
    )

    with pytest.raises(ValueError, match="must declare obs_normalizer_eps"):
        ppo._teacher_normalization_config(
            {
                "experiment_config": {
                    "algo": {"config": {"normalize_actor_obs": True}}
                }
            }
        )


def test_strict_teacher_rejects_checkpoint_without_experiment_config():
    ppo = object.__new__(PPO)
    ppo.strict_teacher_load = True

    with pytest.raises(ValueError, match="no verifiable experiment_config"):
        ppo._validate_teacher_checkpoint_runtime_config(
            {"actor_model_state_dict": {}},
            obs_keys=["actor_obs"],
            teacher_actor_cfg=SimpleNamespace(layer_config=SimpleNamespace(perception_input_name="")),
        )


def test_strict_teacher_alias_validation_checks_term_semantics():
    ppo = object.__new__(PPO)
    ppo.strict_teacher_load = True
    runtime_term = ObsTermCfg(func="pkg:expected", params={"frame": "root"}, scale=2.0)
    runtime_group = SimpleNamespace(
        history_length=1,
        concatenate=True,
        enable_noise=False,
        terms={"term": runtime_term},
    )
    robot_contract = {
        "actions_dim": 2,
        "dof_names": ["left", "right"],
        "dof_effort_limit_list": [20.0, 20.0],
        "init_state": {"default_joint_angles": {"left": 0.0, "right": 0.0}},
        "control": {"action_scale": 0.25},
    }
    action_terms = {"joint_control": {"func": "pkg:joint", "params": {}, "scale": 1.0}}
    ppo.env = SimpleNamespace(
        observation_manager=SimpleNamespace(
            cfg=SimpleNamespace(
                groups={"actor_obs_teacher_compat": runtime_group},
                clip_observations=100.0,
            )
        ),
        robot_config=robot_contract,
        action_manager=SimpleNamespace(cfg={"terms": action_terms}),
    )
    teacher_actor_cfg = SimpleNamespace(layer_config=SimpleNamespace(perception_input_name=""))
    checkpoint = {
        "experiment_config": {
            "algo": {"config": {"module_dict": {"actor": {"input_dim": ["actor_obs"]}}}},
            "observation": {
                "clip_observations": 100.0,
                "groups": {
                    "actor_obs": {
                        "history_length": 1,
                        "concatenate": True,
                        "enable_noise": False,
                        "terms": {
                            "term": {
                                "func": "pkg:wrong",
                                "params": {"frame": "root"},
                                "scale": 2.0,
                                "noise": 0.0,
                                "clip": None,
                            }
                        },
                    }
                }
            },
            "robot": robot_contract,
            "action": {"terms": action_terms},
        }
    }

    with pytest.raises(ValueError, match=r"term\.func"):
        ppo._validate_teacher_checkpoint_runtime_config(
            checkpoint,
            obs_keys=["actor_obs_teacher_compat"],
            teacher_actor_cfg=teacher_actor_cfg,
        )


@pytest.mark.parametrize(
    ("field_name", "checkpoint_value", "error_pattern"),
    [
        ("noise", 0.0, r"term\.noise"),
        ("clip", [-2.0, 2.0], r"term\.clip"),
    ],
)
def test_strict_teacher_validation_checks_term_transforms(field_name, checkpoint_value, error_pattern):
    ppo = object.__new__(PPO)
    ppo.strict_teacher_load = True
    runtime_term = ObsTermCfg(
        func="pkg:expected",
        params={"frame": "root"},
        scale=2.0,
        noise=0.1,
        clip=(-1.0, 1.0),
    )
    runtime_group = SimpleNamespace(
        history_length=1,
        concatenate=True,
        enable_noise=True,
        terms={"term": runtime_term},
    )
    ppo.env = _strict_teacher_runtime_env(runtime_group)
    teacher_actor_cfg = SimpleNamespace(layer_config=SimpleNamespace(perception_input_name=""))
    checkpoint_term = {
        "func": "pkg:expected",
        "params": {"frame": "root"},
        "scale": 2.0,
        "noise": 0.1,
        "clip": [-1.0, 1.0],
    }
    checkpoint_term[field_name] = checkpoint_value
    checkpoint = {
        "experiment_config": {
            "algo": {"config": {"module_dict": {"actor": {"input_dim": ["actor_obs"]}}}},
            "observation": {
                "clip_observations": 100.0,
                "groups": {
                    "actor_obs": {
                        "history_length": 1,
                        "concatenate": True,
                        "enable_noise": True,
                        "terms": {"term": checkpoint_term},
                    }
                }
            },
            "robot": ppo.env.robot_config,
            "action": {"terms": ppo.env.action_manager.cfg["terms"]},
        }
    }

    with pytest.raises(ValueError, match=error_pattern):
        ppo._validate_teacher_checkpoint_runtime_config(
            checkpoint,
            obs_keys=["actor_obs"],
            teacher_actor_cfg=teacher_actor_cfg,
        )


@pytest.mark.parametrize(
    ("field_name", "checkpoint_value"),
    [("concatenate", False), ("enable_noise", False)],
)
def test_strict_teacher_validation_checks_group_transforms(field_name, checkpoint_value):
    ppo = object.__new__(PPO)
    ppo.strict_teacher_load = True
    runtime_term = ObsTermCfg(func="pkg:expected", noise=0.1)
    runtime_group = SimpleNamespace(
        history_length=1,
        concatenate=True,
        enable_noise=True,
        terms={"term": runtime_term},
    )
    ppo.env = _strict_teacher_runtime_env(runtime_group)
    teacher_actor_cfg = SimpleNamespace(layer_config=SimpleNamespace(perception_input_name=""))
    checkpoint_group = {
        "history_length": 1,
        "concatenate": True,
        "enable_noise": True,
        "terms": {
            "term": {
                "func": "pkg:expected",
                "params": {},
                "scale": 1.0,
                "noise": 0.1,
                "clip": None,
            }
        },
    }
    checkpoint_group[field_name] = checkpoint_value
    checkpoint = {
        "experiment_config": {
            "algo": {"config": {"module_dict": {"actor": {"input_dim": ["actor_obs"]}}}},
            "observation": {
                "clip_observations": 100.0,
                "groups": {"actor_obs": checkpoint_group},
            },
            "robot": ppo.env.robot_config,
            "action": {"terms": ppo.env.action_manager.cfg["terms"]},
        }
    }

    with pytest.raises(ValueError, match=field_name):
        ppo._validate_teacher_checkpoint_runtime_config(
            checkpoint,
            obs_keys=["actor_obs"],
            teacher_actor_cfg=teacher_actor_cfg,
        )


def _strict_teacher_runtime_env(runtime_group):
    robot_contract = {
        "actions_dim": 2,
        "dof_names": ["left", "right"],
        "dof_effort_limit_list": [20.0, 20.0],
        "init_state": {"default_joint_angles": {"left": 0.0, "right": 0.0}},
        "control": {"action_scale": 0.25},
    }
    action_terms = {"joint_control": {"func": "pkg:joint", "params": {}, "scale": 1.0}}
    return SimpleNamespace(
        observation_manager=SimpleNamespace(
            cfg=SimpleNamespace(
                groups={"actor_obs": runtime_group},
                clip_observations=100.0,
            )
        ),
        robot_config=robot_contract,
        action_manager=SimpleNamespace(cfg={"terms": action_terms}),
    )


def _strict_teacher_perception_contract(*, runtime_cfg: PerceptionConfig | None = None):
    checkpoint_cfg = PerceptionConfig(
        enabled=True,
        output_mode="camera_depth",
        camera_width=87,
        camera_height=58,
        camera_warp_preprocess=False,
        camera_warp_normalize=False,
        encoder_type="defm_vit_s14",
        encoder_output_dim=384,
        encoder_pretrained=False,
        encoder_freeze_backbone=True,
        encoder_target_size=224,
        encoder_patch_size=14,
    )
    runtime_cfg = checkpoint_cfg if runtime_cfg is None else runtime_cfg
    checkpoint_perception_group = {
        "history_length": 1,
        "concatenate": True,
        "enable_noise": False,
        "terms": {
            "depth": {
                "func": "holosoma.managers.observation.terms.perception:perception_obs",
                "params": {},
                "scale": 1.0,
                "noise": 0.0,
                "clip": None,
            }
        },
    }
    runtime_perception_group = SimpleNamespace(
        history_length=1,
        concatenate=True,
        enable_noise=False,
        terms={
            "depth": ObsTermCfg(
                func="holosoma.managers.observation.terms.perception:teacher_perception_obs"
            )
        },
    )
    checkpoint_actor_group = {
        "history_length": 1,
        "concatenate": True,
        "enable_noise": False,
        "terms": {
            "proprio": {
                "func": "pkg:proprio",
                "params": {},
                "scale": 1.0,
                "noise": 0.0,
                "clip": None,
            }
        },
    }
    runtime_actor_group = SimpleNamespace(
        history_length=1,
        concatenate=True,
        enable_noise=False,
        terms={"proprio": ObsTermCfg(func="pkg:proprio")},
    )
    env = _strict_teacher_runtime_env(runtime_actor_group)
    env.observation_manager.cfg.groups = {
        "teacher_actor_obs": runtime_actor_group,
        "teacher_depth": runtime_perception_group,
    }
    env.perception_manager = None
    env.teacher_perception_manager = SimpleNamespace(cfg=runtime_cfg)
    env.critic_perception_manager = None
    teacher_actor_cfg = SimpleNamespace(
        layer_config=SimpleNamespace(perception_input_name="teacher_depth")
    )
    checkpoint = {
        "experiment_config": {
            "algo": {
                "config": {
                    "module_dict": {
                        "actor": {
                            # Perception is a side input and is deliberately
                            # absent from ModuleConfig.input_dim in production.
                            "input_dim": ["actor_obs"],
                            "layer_config": {
                                "perception_input_name": "perception_obs",
                                "perception_encoder_type": checkpoint_cfg.encoder_type,
                                "perception_output_dim": checkpoint_cfg.encoder_output_dim,
                                "perception_freeze_backbone": checkpoint_cfg.encoder_freeze_backbone,
                                "perception_target_size": checkpoint_cfg.encoder_target_size,
                                "perception_patch_size": checkpoint_cfg.encoder_patch_size,
                                "perception_input_height": checkpoint_cfg.camera_height,
                                "perception_input_width": checkpoint_cfg.camera_width,
                            },
                        }
                    }
                }
            },
            "observation": {
                "clip_observations": 100.0,
                "groups": {
                    "actor_obs": checkpoint_actor_group,
                    "perception_obs": checkpoint_perception_group,
                },
            },
            "perception": dataclasses.asdict(checkpoint_cfg),
            "robot": env.robot_config,
            "action": {"terms": env.action_manager.cfg["terms"]},
        }
    }
    return checkpoint_cfg, checkpoint, env, teacher_actor_cfg


def test_strict_teacher_validation_accepts_exact_perception_manager_alias_contract():
    ppo = object.__new__(PPO)
    ppo.strict_teacher_load = True
    _, checkpoint, ppo.env, teacher_actor_cfg = _strict_teacher_perception_contract()

    ppo._validate_teacher_checkpoint_runtime_config(
        checkpoint,
        obs_keys=["teacher_actor_obs"],
        teacher_actor_cfg=teacher_actor_cfg,
    )


@pytest.mark.parametrize(
    ("runtime_change", "error_pattern"),
    [
        ({"encoder_type": "defm_regnet_y_800mf"}, "perception_encoder_type/encoder_type"),
        ({"camera_warp_normalize": True}, "metric meters|sensor semantics"),
    ],
)
def test_strict_teacher_validation_rejects_perception_manager_contract_mismatch(
    runtime_change,
    error_pattern,
):
    checkpoint_cfg, _, _, _ = _strict_teacher_perception_contract()
    runtime_cfg = dataclasses.replace(checkpoint_cfg, **runtime_change)
    _, checkpoint, env, teacher_actor_cfg = _strict_teacher_perception_contract(
        runtime_cfg=runtime_cfg
    )
    ppo = object.__new__(PPO)
    ppo.strict_teacher_load = True
    ppo.env = env

    with pytest.raises(ValueError, match=error_pattern):
        ppo._validate_teacher_checkpoint_runtime_config(
            checkpoint,
            obs_keys=["teacher_actor_obs"],
            teacher_actor_cfg=teacher_actor_cfg,
        )


def test_strict_teacher_validation_rejects_global_observation_clip_mismatch():
    _, checkpoint, env, teacher_actor_cfg = _strict_teacher_perception_contract()
    checkpoint["experiment_config"]["observation"]["clip_observations"] = 10.0
    ppo = object.__new__(PPO)
    ppo.strict_teacher_load = True
    ppo.env = env

    with pytest.raises(ValueError, match=r"observation\.clip_observations mismatch"):
        ppo._validate_teacher_checkpoint_runtime_config(
            checkpoint,
            obs_keys=["teacher_actor_obs"],
            teacher_actor_cfg=teacher_actor_cfg,
        )


@pytest.mark.parametrize("contract_path", ["dof_names", "control", "action_terms"])
def test_strict_teacher_validation_checks_action_semantics(contract_path):
    ppo = object.__new__(PPO)
    ppo.strict_teacher_load = True
    runtime_term = ObsTermCfg(func="pkg:expected")
    runtime_group = SimpleNamespace(
        history_length=1,
        concatenate=True,
        enable_noise=False,
        terms={"term": runtime_term},
    )
    ppo.env = _strict_teacher_runtime_env(runtime_group)
    checkpoint_robot = dict(ppo.env.robot_config)
    checkpoint_robot["dof_names"] = list(checkpoint_robot["dof_names"])
    checkpoint_robot["control"] = dict(checkpoint_robot["control"])
    checkpoint_action_terms = {
        name: dict(config) for name, config in ppo.env.action_manager.cfg["terms"].items()
    }
    if contract_path == "dof_names":
        checkpoint_robot["dof_names"].reverse()
    elif contract_path == "control":
        checkpoint_robot["control"]["action_scale"] = 0.5
    else:
        checkpoint_action_terms["joint_control"]["scale"] = 0.5
    checkpoint = {
        "experiment_config": {
            "algo": {"config": {"module_dict": {"actor": {"input_dim": ["actor_obs"]}}}},
            "observation": {
                "clip_observations": 100.0,
                "groups": {
                    "actor_obs": {
                        "history_length": 1,
                        "concatenate": True,
                        "enable_noise": False,
                        "terms": {
                            "term": {
                                "func": "pkg:expected",
                                "params": {},
                                "scale": 1.0,
                                "noise": 0.0,
                                "clip": None,
                            }
                        },
                    }
                }
            },
            "robot": checkpoint_robot,
            "action": {"terms": checkpoint_action_terms},
        }
    }

    with pytest.raises(ValueError, match="teacher (robot|action)"):
        ppo._validate_teacher_checkpoint_runtime_config(
            checkpoint,
            obs_keys=["actor_obs"],
            teacher_actor_cfg=SimpleNamespace(layer_config=SimpleNamespace(perception_input_name="")),
        )


def test_standalone_export_metadata_uses_completed_iteration():
    ppo = object.__new__(PPO)
    ppo.current_learning_iteration = 40000
    ppo.actor = nn.Linear(1, 1)
    ppo.actor.train()
    ppo.actor_perception_key = ""
    ppo._eval_mode = MethodType(lambda self: self.actor.eval(), ppo)
    ppo._train_mode = MethodType(lambda self: self.actor.train(), ppo)
    ppo._get_zero_input = MethodType(lambda self: torch.zeros(1, 1), ppo)
    ppo._get_zero_perception_input = MethodType(lambda self: None, ppo)
    ppo.env = SimpleNamespace(robot_config=SimpleNamespace(dof_names=[]))
    ppo.logging_helper = SimpleNamespace(save_to_wandb=lambda path: None)
    seen_iterations = []
    ppo._checkpoint_metadata = MethodType(
        lambda self, iteration=None: seen_iterations.append(iteration) or {},
        ppo,
    )

    with (
        patch.object(PPO, "actor_onnx_wrapper", new_callable=PropertyMock, return_value=object()),
        patch("holosoma.agents.ppo.ppo.export_policy_as_onnx"),
        patch("holosoma.agents.ppo.ppo.get_control_gains_from_config", return_value=([], [])),
        patch("holosoma.agents.ppo.ppo.get_command_ranges_from_env", return_value={}),
        patch("holosoma.agents.ppo.ppo.get_urdf_text_from_robot_config", return_value=("", "")),
        patch("holosoma.agents.ppo.ppo.attach_onnx_metadata"),
    ):
        ppo.export("policy.onnx")

    assert seen_iterations == [39999]


def test_evaluation_export_metadata_uses_source_checkpoint_iteration():
    ppo = object.__new__(PPO)
    ppo.current_learning_iteration = 0
    ppo._evaluation_completed_iteration = 12
    ppo.actor = nn.Linear(1, 1)
    ppo.actor.train()
    ppo.actor_perception_key = ""
    ppo._eval_mode = MethodType(lambda self: self.actor.eval(), ppo)
    ppo._train_mode = MethodType(lambda self: self.actor.train(), ppo)
    ppo._get_zero_input = MethodType(lambda self: torch.zeros(1, 1), ppo)
    ppo._get_zero_perception_input = MethodType(lambda self: None, ppo)
    ppo.env = SimpleNamespace(robot_config=SimpleNamespace(dof_names=[]))
    ppo.logging_helper = SimpleNamespace(save_to_wandb=lambda path: None)
    seen_iterations = []
    ppo._checkpoint_metadata = MethodType(
        lambda self, iteration=None: seen_iterations.append(iteration) or {},
        ppo,
    )

    with (
        patch.object(PPO, "actor_onnx_wrapper", new_callable=PropertyMock, return_value=object()),
        patch("holosoma.agents.ppo.ppo.export_policy_as_onnx"),
        patch("holosoma.agents.ppo.ppo.get_control_gains_from_config", return_value=([], [])),
        patch("holosoma.agents.ppo.ppo.get_command_ranges_from_env", return_value={}),
        patch("holosoma.agents.ppo.ppo.get_urdf_text_from_robot_config", return_value=("", "")),
        patch("holosoma.agents.ppo.ppo.attach_onnx_metadata"),
    ):
        ppo.export("policy.onnx")

    assert seen_iterations == [12]


def test_model_synchronization_broadcasts_parameters_and_persistent_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC", raising=False)
    ppo = object.__new__(PPO)
    ppo.actor = nn.Sequential(nn.Linear(3, 2), nn.BatchNorm1d(2))
    ppo.critic = nn.Linear(3, 1)
    ppo.critic.register_buffer("frozen_feature_scale", torch.tensor([3.0]))
    ppo.gpu_world_size = 1
    ppo.gpu_global_rank = 0
    broadcast_ptrs: list[int] = []
    ppo._broadcast_tensor = MethodType(
        lambda self, tensor, *, src=0: broadcast_ptrs.append(tensor.data_ptr()),
        ppo,
    )

    expected_ptrs = [
        tensor.data_ptr()
        for module in (ppo.actor, ppo.critic)
        for tensor in itertools.chain(module.parameters(), module.buffers())
    ]

    ppo._synchronize_model_weights()

    assert broadcast_ptrs == expected_ptrs
    assert ppo.actor[1].running_mean.data_ptr() in broadcast_ptrs
    assert ppo.actor[1].running_var.data_ptr() in broadcast_ptrs
    assert ppo.actor[1].num_batches_tracked.data_ptr() in broadcast_ptrs
    assert ppo.critic.frozen_feature_scale.data_ptr() in broadcast_ptrs
