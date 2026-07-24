from __future__ import annotations

import copy
import random
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from holosoma.agents.ppo.ppo import PPO
from holosoma.config_types.algo import DistillationConfig
from holosoma.utils.rng_checkpoint import (
    capture_rng_checkpoint_state,
    restore_rng_checkpoint_state,
)


class _DummyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def act_inference(self, policy_state_dict):
        return policy_state_dict["actor_obs"] + 1.0


class _DummyTeacherActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.perception_input_name = ""

    def act(self, policy_state_dict):
        return policy_state_dict["actor_obs"] + 10.0

    def act_inference(self, policy_state_dict):
        return policy_state_dict["actor_obs"] + 1.0


class _RngConsumingActor(_DummyActor):
    def act_inference(self, policy_state_dict):
        random.random()
        np.random.random()
        return policy_state_dict["actor_obs"] + torch.randn_like(
            policy_state_dict["actor_obs"]
        )


def _make_stub_ppo() -> PPO:
    ppo = object.__new__(PPO)
    ppo.is_main_process = True
    ppo.is_multi_gpu = False
    ppo.gpu_global_rank = 0
    ppo.gpu_world_size = 1
    ppo.current_learning_iteration = 0
    ppo.fixed_bc_eval_num_samples = 2
    ppo.fixed_bc_eval_log_interval = 1
    ppo.dagger_enabled = True
    ppo.dagger_ignore_zero_teacher_actions = True
    ppo.actor_perception_key = ""
    ppo.actor_obs_keys = ["actor_obs"]
    ppo.algo_obs_dim_dict = {"actor_obs": 1}
    ppo.num_act = 2
    ppo._fixed_bc_eval_ready = False
    ppo._fixed_bc_eval_size = 0
    ppo._fixed_bc_eval_actor_obs_parts = []
    ppo._fixed_bc_eval_teacher_actions_parts = []
    ppo._fixed_bc_eval_actor_perception_parts = []
    ppo._fixed_bc_eval_dataset = {}
    ppo.device = "cpu"
    ppo.actor = _DummyActor()
    ppo.actor_obs_normalizers = {"actor_obs": nn.Identity()}
    ppo._normalize_actor_obs = MethodType(lambda self, obs, update=False: obs, ppo)
    return ppo


def _make_guard_ppo() -> PPO:
    ppo = _make_stub_ppo()
    ppo.fixed_bc_guard_enabled = True
    ppo.fixed_bc_guard_reference_end_epoch = 2
    ppo.fixed_bc_guard_max_reference_ratio = 2.0
    ppo.fixed_bc_guard_absolute_max_mu_mse = 0.160
    ppo.fixed_bc_guard_start_epoch = 3
    ppo.fixed_bc_guard_consecutive_evals = 2
    ppo.distill_mode = "dagger"
    ppo.ppo_start_epoch = 0
    ppo.dagger_end_epoch = 3
    ppo.ppo_start_coeff = 0.0
    ppo.ppo_target_coeff = 0.7
    ppo.ppo_schedule_step_epochs = 3
    ppo._get_distributed_loss_weight = MethodType(lambda self: 1.0, ppo)
    ppo._fixed_bc_eval_ready = True
    ppo._fixed_bc_eval_size = 2
    ppo._fixed_bc_eval_dataset = {
        "actor_obs_raw": torch.tensor([[1.0], [2.0]]),
        "teacher_actions": torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
    }
    ppo._fixed_bc_guard_config_fingerprint = (
        ppo._fixed_bc_guard_runtime_config_fingerprint()
    )
    ppo._fixed_bc_guard_state = ppo._new_fixed_bc_guard_state()
    return ppo


def _guard_metrics(mu_mse: float) -> dict[str, float]:
    return {
        "fixed_bc_mu_mse": mu_mse,
        "fixed_bc_num_samples": 2.0,
        "fixed_bc_weighted_num_samples": 2.0,
        "fixed_bc_rank_strata": 1.0,
    }


def _make_terminal_guard_state(
    *,
    mu_mse: float = 0.04,
) -> tuple[PPO, dict, dict[str, dict], dict]:
    """Create an off-cadence final observation with a frozen guard threshold."""

    ppo = _make_guard_ppo()
    ppo.fixed_bc_eval_log_interval = 2
    ppo.fixed_bc_guard_reference_end_epoch = 2
    ppo.fixed_bc_guard_start_epoch = 4
    ppo.config = SimpleNamespace(num_learning_iterations=4)
    ppo._fixed_bc_guard_config_fingerprint = (
        ppo._fixed_bc_guard_runtime_config_fingerprint()
    )
    ppo._fixed_bc_guard_state = ppo._new_fixed_bc_guard_state()
    assert not ppo._update_fixed_bc_guard(
        current_iteration=0,
        metrics=_guard_metrics(0.05),
    )
    assert not ppo._update_fixed_bc_guard(
        current_iteration=2,
        metrics=_guard_metrics(0.03),
    )
    _, global_dataset_digest = ppo._fixed_bc_guard_live_dataset_digests()
    state = ppo._build_terminal_fixed_bc_eval_state(
        completed_iteration=3,
        metrics=_guard_metrics(mu_mse),
        scheduled_evaluation=False,
        global_dataset_digest=global_dataset_digest,
        expected_weighted_num_samples=(
            ppo._fixed_bc_guard_expected_weighted_sample_count()
        ),
    )
    fixed_states = {"0": ppo._local_fixed_bc_eval_checkpoint_state()}
    guard_state = ppo._fixed_bc_guard_checkpoint_state(
        fixed_states,
        next_iteration=4,
    )
    assert guard_state is not None
    return ppo, state, fixed_states, guard_state


def _terminal_checkpoint_payload(
    ppo: PPO,
    state: dict,
    fixed_states: dict[str, dict],
    guard_state: dict,
) -> dict:
    return {
        "iter": 3,
        "next_iter": 4,
        "experiment_config": {
            "algo": {"config": {"num_learning_iterations": 4}}
        },
        "fixed_bc_eval_by_rank": fixed_states,
        "fixed_bc_guard_state": guard_state,
        "terminal_fixed_bc_eval": state,
        "terminal_fixed_bc_eval_sha256": (
            ppo._terminal_fixed_bc_eval_state_sha256(state)
        ),
    }


def test_capture_fixed_bc_eval_samples_respects_mask_and_zero_teacher():
    ppo = _make_stub_ppo()
    actor_obs_raw = torch.tensor([[1.0], [2.0], [3.0]])
    teacher_actions = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    teacher_bc_mask = torch.tensor([[True], [True], [False]])

    ppo._maybe_capture_fixed_bc_eval_samples(
        actor_obs_raw=actor_obs_raw,
        actor_perception_obs=None,
        teacher_actions=teacher_actions,
        teacher_bc_mask=teacher_bc_mask,
    )

    assert ppo._fixed_bc_eval_ready is False
    assert ppo._fixed_bc_eval_size == 1

    ppo._maybe_capture_fixed_bc_eval_samples(
        actor_obs_raw=torch.tensor([[4.0], [5.0]]),
        actor_perception_obs=None,
        teacher_actions=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        teacher_bc_mask=torch.tensor([[True], [True]]),
    )

    assert ppo._fixed_bc_eval_ready is True
    assert ppo._fixed_bc_eval_dataset["actor_obs_raw"].shape == (2, 1)
    assert ppo._fixed_bc_eval_dataset["teacher_actions"].shape == (2, 2)
    assert torch.equal(ppo._fixed_bc_eval_dataset["actor_obs_raw"].squeeze(-1), torch.tensor([2.0, 4.0]))


def test_non_main_rank_captures_bounded_fixed_bc_stratum():
    ppo = _make_stub_ppo()
    ppo.is_main_process = False
    ppo.is_multi_gpu = True
    ppo.gpu_world_size = 2

    ppo._maybe_capture_fixed_bc_eval_samples(
        actor_obs_raw=torch.tensor([[1.0], [2.0]]),
        actor_perception_obs=None,
        teacher_actions=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        teacher_bc_mask=None,
    )

    assert ppo._fixed_bc_eval_ready is True
    assert ppo._fixed_bc_eval_dataset["actor_obs_raw"].shape[0] == 1


def test_fixed_bc_allocation_realizes_exact_global_budget():
    ppo = _make_stub_ppo()
    ppo.fixed_bc_eval_num_samples = 4096
    ppo.gpu_world_size = 104

    targets = []
    for rank in range(ppo.gpu_world_size):
        ppo.gpu_global_rank = rank
        targets.append(ppo._fixed_bc_eval_local_target())

    assert targets[:40] == [40] * 40
    assert targets[40:] == [39] * 64
    assert sum(targets) == 4096


def test_fixed_bc_eval_metrics_use_deterministic_actor_mean():
    ppo = _make_stub_ppo()
    ppo._fixed_bc_eval_ready = True
    ppo._fixed_bc_eval_dataset = {
        "actor_obs_raw": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "teacher_actions": torch.tensor([[2.0, 3.0], [5.0, 4.0]]),
    }

    metrics = ppo._get_fixed_bc_eval_metrics(current_iteration=4)

    assert metrics["fixed_bc_num_samples"] == 2.0
    assert metrics["fixed_bc_mu_mse"] == 0.5
    assert metrics["fixed_bc_rank_strata"] == 1.0


def test_fixed_bc_guard_config_defaults_are_disabled():
    config = DistillationConfig()

    assert config.fixed_bc_guard_enabled is False
    assert config.fixed_bc_guard_reference_end_epoch == 600
    assert config.fixed_bc_guard_max_reference_ratio == 2.0
    assert config.fixed_bc_guard_absolute_max_mu_mse == pytest.approx(0.160)
    assert config.fixed_bc_guard_start_epoch == -1
    assert config.fixed_bc_guard_consecutive_evals == 3


@pytest.mark.parametrize("value", [True, np.bool_(False), "2.0", float("inf")])
def test_fixed_bc_guard_core_rejects_coerced_or_nonfinite_real(value):
    with pytest.raises(ValueError, match="must be (?:a finite real number|finite)"):
        PPO._strict_config_real("fixed_bc_guard_max_reference_ratio", value)


def test_fixed_bc_guard_core_requires_canonical_disabled_start():
    distill = DistillationConfig(
        enabled=True,
        mode="dagger",
        policy_to_clone="teacher.pt",
        bc_loss_coef=1.0,
    )
    distill_values = dict(distill.__dict__)
    distill_values["fixed_bc_guard_enabled"] = False
    distill_values["fixed_bc_guard_start_epoch"] = 7
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        num_learning_iterations=20,
        distill=SimpleNamespace(**distill_values),
    )

    with pytest.raises(ValueError, match="Disabled fixed-BC guard requires"):
        ppo._setup_distillation()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "ppo_start_epoch": 0,
                "dagger_end_epoch": 10,
                "fixed_bc_guard_reference_end_epoch": 2,
                "fixed_bc_guard_start_epoch": 10,
            },
            "reference period must remain pure BC",
        ),
        (
            {
                "ppo_start_epoch": 3,
                "dagger_end_epoch": 10,
                "fixed_bc_guard_reference_end_epoch": 2,
                "fixed_bc_guard_start_epoch": 9,
            },
            "must be >= dagger_end_epoch",
        ),
        (
            {
                "ppo_start_epoch": 3,
                "dagger_end_epoch": 4,
                "fixed_bc_eval_log_interval": 2,
                "fixed_bc_guard_reference_end_epoch": 2,
                "fixed_bc_guard_start_epoch": 4,
            },
            "at least three expected evaluations",
        ),
        (
            {
                "ppo_start_epoch": 3,
                "dagger_end_epoch": 10,
                "fixed_bc_guard_reference_end_epoch": 2,
                "fixed_bc_guard_start_epoch": 19,
                "fixed_bc_guard_consecutive_evals": 2,
            },
            "enough scheduled evaluations",
        ),
    ],
)
def test_fixed_bc_guard_rejects_scientifically_invalid_schedule(overrides, message):
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        num_learning_iterations=20,
        distill=DistillationConfig(
            enabled=True,
            mode="dagger",
            policy_to_clone="teacher.pt",
            bc_loss_coef=1.0,
            fixed_bc_guard_enabled=True,
            **overrides,
        ),
    )

    with pytest.raises(ValueError, match=message):
        ppo._setup_distillation()


def test_fixed_bc_guard_reference_threshold_recovery_and_trip():
    ppo = _make_guard_ppo()

    assert ppo._update_fixed_bc_guard(current_iteration=0, metrics=_guard_metrics(0.10)) is False
    assert ppo._update_fixed_bc_guard(current_iteration=1, metrics=_guard_metrics(0.08)) is False
    assert ppo._update_fixed_bc_guard(current_iteration=2, metrics=_guard_metrics(0.09)) is False
    assert ppo._fixed_bc_guard_state["reference_min_mu_mse"] == pytest.approx(0.08)
    assert ppo._fixed_bc_guard_state["reference_min_iteration"] == 1
    assert ppo._fixed_bc_guard_state["threshold_mu_mse"] == pytest.approx(0.16)

    assert ppo._update_fixed_bc_guard(current_iteration=3, metrics=_guard_metrics(0.17)) is False
    assert ppo._fixed_bc_guard_state["consecutive_exceedances"] == 1
    assert ppo._update_fixed_bc_guard(current_iteration=4, metrics=_guard_metrics(0.15)) is False
    assert ppo._fixed_bc_guard_state["consecutive_exceedances"] == 0
    assert ppo._update_fixed_bc_guard(current_iteration=5, metrics=_guard_metrics(0.18)) is False
    assert ppo._update_fixed_bc_guard(current_iteration=6, metrics=_guard_metrics(0.19)) is True
    assert ppo._fixed_bc_guard_state["tripped"] is True
    assert ppo._fixed_bc_guard_state["trip_iteration"] == 6
    assert ppo._fixed_bc_guard_state["last_mu_mse"] == pytest.approx(0.19)

    log_metrics = ppo._fixed_bc_guard_log_metrics()
    assert log_metrics == {
        "fixed_bc_guard_reference_min_mu_mse": pytest.approx(0.08),
        "fixed_bc_guard_effective_threshold_mu_mse": pytest.approx(0.16),
        "fixed_bc_guard_consecutive_exceedances": 2.0,
        "fixed_bc_guard_last_mu_mse": pytest.approx(0.19),
    }


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("fixed_bc_num_samples", 1.0, "sample budget mismatch"),
        ("fixed_bc_weighted_num_samples", 1.0, "weighted sample count mismatch"),
        ("fixed_bc_rank_strata", 2.0, "rank-strata mismatch"),
        ("fixed_bc_mu_mse", float("nan"), "non-finite"),
    ],
)
def test_fixed_bc_guard_rejects_malformed_aggregated_metrics(key, value, message):
    ppo = _make_guard_ppo()
    metrics = _guard_metrics(0.1)
    metrics[key] = value

    with pytest.raises((ValueError, RuntimeError), match=message):
        ppo._update_fixed_bc_guard(current_iteration=0, metrics=metrics)


def test_fixed_bc_guard_missing_metric_fails_closed():
    ppo = _make_guard_ppo()
    metrics = _guard_metrics(0.1)
    del metrics["fixed_bc_mu_mse"]

    with pytest.raises(RuntimeError, match="missing required metrics"):
        ppo._update_fixed_bc_guard(current_iteration=0, metrics=metrics)


def test_fixed_bc_guard_synchronizes_metric_failure_before_next_collective():
    ppo = _make_guard_ppo()
    operations: list[tuple[str, Exception | None]] = []

    def synchronize(self, local_error, *, operation):
        operations.append((operation, local_error))
        if local_error is not None:
            raise RuntimeError("synchronized metric failure") from local_error

    ppo._synchronize_training_phase_error = MethodType(synchronize, ppo)
    ppo._fixed_bc_guard_expected_weighted_sample_count = MethodType(
        lambda self: pytest.fail("weighted-count collective must not be entered"),
        ppo,
    )
    metrics = _guard_metrics(0.1)
    del metrics["fixed_bc_mu_mse"]

    with pytest.raises(RuntimeError, match="synchronized metric failure"):
        ppo._update_fixed_bc_guard(current_iteration=0, metrics=metrics)

    assert any(
        operation == "fixed BC guard metric-envelope validation"
        and isinstance(error, RuntimeError)
        for operation, error in operations
    )


def test_fixed_bc_guard_synchronizes_timeline_failure_before_dataset_collective():
    ppo = _make_guard_ppo()
    ppo._fixed_bc_guard_state["last_eval_iteration"] = 99
    operations: list[tuple[str, Exception | None]] = []

    def synchronize(self, local_error, *, operation):
        operations.append((operation, local_error))
        if local_error is not None:
            raise RuntimeError("synchronized timeline failure") from local_error

    ppo._synchronize_training_phase_error = MethodType(synchronize, ppo)
    ppo._fixed_bc_guard_live_dataset_digests = MethodType(
        lambda self: pytest.fail("dataset collective must not be entered"),
        ppo,
    )

    with pytest.raises(RuntimeError, match="synchronized timeline failure"):
        ppo._update_fixed_bc_guard(current_iteration=0, metrics=_guard_metrics(0.1))

    assert any(
        operation == "fixed BC guard timeline validation"
        and isinstance(error, RuntimeError)
        for operation, error in operations
    )


def test_fixed_bc_guard_requires_dynamic_state_agreement_before_trip_return():
    ppo = _make_guard_ppo()
    for iteration, value in enumerate((0.10, 0.08, 0.09, 0.15, 0.15, 0.18)):
        assert ppo._update_fixed_bc_guard(
            current_iteration=iteration,
            metrics=_guard_metrics(value),
        ) is False

    observed_states: list[dict[str, object]] = []

    def reject_state(self, state):
        observed_states.append(dict(state))
        raise RuntimeError("all-rank dynamic state disagreement")

    ppo._require_all_rank_fixed_bc_guard_state_match = MethodType(reject_state, ppo)
    with pytest.raises(RuntimeError, match="all-rank dynamic state disagreement"):
        ppo._update_fixed_bc_guard(current_iteration=6, metrics=_guard_metrics(0.19))

    assert observed_states[-1]["tripped"] is True
    assert observed_states[-1]["trip_iteration"] == 6


def test_fixed_bc_guard_detects_frozen_dataset_content_drift():
    ppo = _make_guard_ppo()
    ppo._update_fixed_bc_guard(current_iteration=0, metrics=_guard_metrics(0.1))
    ppo._fixed_bc_eval_dataset["actor_obs_raw"][0, 0] += 1.0

    with pytest.raises((ValueError, RuntimeError), match="dataset content digest"):
        ppo._update_fixed_bc_guard(current_iteration=1, metrics=_guard_metrics(0.09))


def test_fixed_bc_guard_refuses_late_dataset_digest_rebaseline():
    ppo = _make_guard_ppo()
    ppo._update_fixed_bc_guard(current_iteration=0, metrics=_guard_metrics(0.1))
    ppo._fixed_bc_guard_state["local_dataset_digest_by_rank"] = None
    ppo._fixed_bc_guard_state["global_dataset_digest"] = None
    ppo._fixed_bc_eval_dataset["actor_obs_raw"][0, 0] += 1.0

    with pytest.raises((ValueError, RuntimeError), match="dataset content digest"):
        ppo._update_fixed_bc_guard(current_iteration=1, metrics=_guard_metrics(0.09))


def test_fixed_bc_guard_validates_complete_state_before_next_transition():
    ppo = _make_guard_ppo()
    ppo._update_fixed_bc_guard(current_iteration=0, metrics=_guard_metrics(0.1))
    ppo._fixed_bc_guard_state["reference_min_mu_mse"] = None
    ppo._fixed_bc_guard_state["reference_min_iteration"] = None

    with pytest.raises((ValueError, RuntimeError), match="reference minimum"):
        ppo._update_fixed_bc_guard(current_iteration=1, metrics=_guard_metrics(0.2))


def test_fixed_bc_guard_digest_protocol_is_observational_to_training_rng():
    original_rng = capture_rng_checkpoint_state()
    try:
        ppo = _make_guard_ppo()
        original_digest = ppo._fixed_bc_guard_live_dataset_digests

        def noisy_digest(self):
            random.random()
            np.random.random()
            torch.rand(2)
            return original_digest()

        ppo._fixed_bc_guard_live_dataset_digests = MethodType(noisy_digest, ppo)
        random.seed(7101)
        np.random.seed(7102)
        torch.manual_seed(7103)
        boundary = capture_rng_checkpoint_state()
        expected_next = (random.random(), float(np.random.random()), torch.rand(3))
        restore_rng_checkpoint_state(boundary)

        ppo._update_fixed_bc_guard(
            current_iteration=0,
            metrics=_guard_metrics(0.1),
        )
        observed_next = (random.random(), float(np.random.random()), torch.rand(3))

        assert observed_next[0] == expected_next[0]
        assert observed_next[1] == expected_next[1]
        assert torch.equal(observed_next[2], expected_next[2])
    finally:
        restore_rng_checkpoint_state(original_rng)


def test_fixed_bc_guard_checkpoint_round_trip_is_strict():
    source = _make_guard_ppo()
    for iteration, value in enumerate((0.10, 0.08, 0.09, 0.17)):
        source._update_fixed_bc_guard(
            current_iteration=iteration,
            metrics=_guard_metrics(value),
        )
    fixed_state = source._local_fixed_bc_eval_checkpoint_state()
    assert fixed_state is not None
    fixed_states = {"0": fixed_state}
    guard_state = source._fixed_bc_guard_checkpoint_state(
        fixed_states, next_iteration=4
    )
    assert guard_state is not None

    resumed = _make_guard_ppo()
    fixed_plan = resumed._prepare_fixed_bc_eval_checkpoint_state(
        {"fixed_bc_eval_by_rank": fixed_states},
        next_iteration=4,
    )
    guard_plan = resumed._prepare_fixed_bc_guard_checkpoint_state(
        {
            "fixed_bc_eval_by_rank": fixed_states,
            "fixed_bc_guard_state": guard_state,
        },
        next_iteration=4,
        fixed_bc_plan=fixed_plan,
    )
    resumed._commit_fixed_bc_eval_checkpoint_plan(fixed_plan)
    resumed._commit_fixed_bc_guard_checkpoint_plan(guard_plan)

    assert resumed._fixed_bc_guard_state == source._fixed_bc_guard_state
    assert resumed._fixed_bc_guard_state["consecutive_exceedances"] == 1


def test_fixed_bc_guard_checkpoint_refuses_semantically_corrupt_live_state():
    source = _make_guard_ppo()
    source._update_fixed_bc_guard(current_iteration=0, metrics=_guard_metrics(0.10))
    fixed_state = source._local_fixed_bc_eval_checkpoint_state()
    assert fixed_state is not None
    source._fixed_bc_guard_state["reference_eval_count"] = -1

    with pytest.raises((ValueError, RuntimeError), match="reference count mismatch"):
        source._fixed_bc_guard_checkpoint_state(
            {"0": fixed_state},
            next_iteration=1,
        )


def test_fixed_bc_guard_diagnostic_checkpoint_is_strict_and_nonresumable():
    source = _make_guard_ppo()
    values = (0.10, 0.08, 0.09, 0.17, 0.15, 0.18, 0.19)
    for iteration, value in enumerate(values):
        tripped = source._update_fixed_bc_guard(
            current_iteration=iteration,
            metrics=_guard_metrics(value),
        )
    assert tripped is True
    fixed_state = source._local_fixed_bc_eval_checkpoint_state()
    assert fixed_state is not None
    fixed_states = {"0": fixed_state}

    diagnostic = source._fixed_bc_guard_checkpoint_state(
        fixed_states,
        next_iteration=7,
        allow_tripped=True,
    )
    assert diagnostic is not None
    assert diagnostic["tripped"] is True
    assert diagnostic["trip_iteration"] == 6

    with pytest.raises(RuntimeError, match="cannot be resumed"):
        source._fixed_bc_guard_checkpoint_state(
            fixed_states,
            next_iteration=7,
        )


def test_fixed_bc_guard_iteration_zero_checkpoint_round_trip_allows_empty_stratum():
    source = _make_guard_ppo()
    source._clear_fixed_bc_eval_state()
    fixed_state = source._local_fixed_bc_eval_checkpoint_state()
    assert fixed_state is not None
    assert fixed_state["ready"] is False
    assert fixed_state["size"] == 0
    fixed_states = {"0": fixed_state}
    guard_state = source._fixed_bc_guard_checkpoint_state(
        fixed_states,
        next_iteration=0,
    )
    assert guard_state is not None
    assert guard_state["local_dataset_digest_by_rank"] is None

    resumed = _make_guard_ppo()
    fixed_plan = resumed._prepare_fixed_bc_eval_checkpoint_state(
        {"fixed_bc_eval_by_rank": fixed_states},
        next_iteration=0,
    )
    guard_plan = resumed._prepare_fixed_bc_guard_checkpoint_state(
        {
            "fixed_bc_eval_by_rank": fixed_states,
            "fixed_bc_guard_state": guard_state,
        },
        next_iteration=0,
        fixed_bc_plan=fixed_plan,
    )
    resumed._commit_fixed_bc_eval_checkpoint_plan(fixed_plan)
    resumed._commit_fixed_bc_guard_checkpoint_plan(guard_plan)

    assert resumed._fixed_bc_eval_ready is False
    assert resumed._fixed_bc_eval_size == 0
    assert resumed._fixed_bc_guard_state == source._new_fixed_bc_guard_state()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda state: state.pop("config_fingerprint"), "schema mismatch"),
        (
            lambda state: state.__setitem__("config_fingerprint", "0" * 64),
            "configuration fingerprint",
        ),
        (lambda state: state.__setitem__("tripped", True), "cannot be resumed"),
        (
            lambda state: state.__setitem__("global_dataset_digest", "0" * 64),
            "dataset content digest",
        ),
    ],
)
def test_fixed_bc_guard_resume_rejects_missing_mismatched_or_tripped_state(mutation, message):
    source = _make_guard_ppo()
    for iteration, value in enumerate((0.10, 0.08, 0.09, 0.15)):
        source._update_fixed_bc_guard(
            current_iteration=iteration,
            metrics=_guard_metrics(value),
        )
    fixed_state = source._local_fixed_bc_eval_checkpoint_state()
    assert fixed_state is not None
    fixed_states = {"0": fixed_state}
    guard_state = source._fixed_bc_guard_checkpoint_state(
        fixed_states, next_iteration=4
    )
    assert guard_state is not None
    mutation(guard_state)

    resumed = _make_guard_ppo()
    with pytest.raises((ValueError, RuntimeError), match=message):
        resumed._prepare_fixed_bc_guard_checkpoint_state(
            {
                "fixed_bc_eval_by_rank": fixed_states,
                "fixed_bc_guard_state": guard_state,
            },
            next_iteration=4,
            fixed_bc_plan={"action": "restore"},
        )


def test_fixed_bc_guard_trip_saves_diagnostic_before_nonzero_exit(tmp_path):
    ppo = _make_guard_ppo()
    ppo.log_dir = str(tmp_path)
    ppo._fixed_bc_guard_state.update(
        {
            "threshold_mu_mse": 0.16,
            "consecutive_exceedances": 2,
            "trip_iteration": 6,
            "trip_mu_mse": 0.19,
            "tripped": True,
        }
    )
    events = []
    ppo._distributed_barrier = MethodType(
        lambda self: events.append(("barrier",)), ppo
    )
    ppo._save_checkpoint_with_distributed_outcome = MethodType(
        lambda self, path, *, next_iteration, allow_tripped_fixed_bc_guard=False: events.append(
            ("save", path, next_iteration, allow_tripped_fixed_bc_guard)
        ),
        ppo,
    )
    ppo._is_node_local_main_process = MethodType(lambda self: False, ppo)

    with pytest.raises(RuntimeError, match="diagnostic checkpoint"):
        ppo._abort_for_fixed_bc_guard_trip(next_iteration=7)

    assert events[0] == ("barrier",)
    assert events[1] == (
        "save",
        str(tmp_path / "diagnostic_fixed_bc_guard_00007.pt"),
        7,
        True,
    )
    assert events[2] == ("barrier",)


def test_fixed_bc_guard_checkpoint_rejects_all_rank_dynamic_state_divergence(monkeypatch):
    ppo = _make_guard_ppo()
    ppo.is_multi_gpu = True
    ppo.gpu_world_size = 2
    ppo.gpu_global_rank = 0
    ppo._all_reduce_small_tensor = MethodType(lambda self, tensor, op: tensor, ppo)
    group = object()
    ppo._setup_gloo_barrier_group = MethodType(lambda self: group, ppo)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)

    def gather(results, local_result, group):
        results[0] = local_result
        results[1] = {
            "rank": 1,
            "state_digest": "0" * 64,
        }

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    with pytest.raises(RuntimeError, match="divergent all-rank"):
        ppo._require_all_rank_fixed_bc_guard_state_match(
            ppo._fixed_bc_guard_state
        )


def test_fixed_bc_eval_metrics_aggregate_weighted_rank_strata():
    ppo = _make_stub_ppo()
    ppo.is_multi_gpu = True
    ppo.gpu_world_size = 2
    ppo.fixed_bc_eval_num_samples = 4
    ppo._fixed_bc_eval_ready = True
    ppo._fixed_bc_eval_dataset = {
        "actor_obs_raw": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "teacher_actions": torch.tensor([[2.0, 3.0], [5.0, 4.0]]),
    }
    ppo._get_distributed_loss_weight = MethodType(lambda self: 1.0, ppo)

    def reduce(self, tensor, op):
        if tensor.numel() == 1:
            return tensor
        return tensor + tensor.new_tensor([8.0, 4.0, 2.0, 2.0, 1.0])

    ppo._all_reduce_small_tensor = MethodType(reduce, ppo)

    metrics = ppo._get_fixed_bc_eval_metrics(current_iteration=0)

    assert metrics["fixed_bc_mu_mse"] == 1.25
    assert metrics["fixed_bc_num_samples"] == 4.0
    assert metrics["fixed_bc_weighted_num_samples"] == 4.0
    assert metrics["fixed_bc_rank_strata"] == 2.0


def test_fixed_bc_eval_actor_and_collectives_do_not_change_next_training_draw():
    original_rng = capture_rng_checkpoint_state()
    try:
        ppo = _make_stub_ppo()
        ppo.actor = _RngConsumingActor()
        ppo.is_multi_gpu = True
        ppo.gpu_world_size = 2
        ppo.fixed_bc_eval_num_samples = 4
        ppo.fixed_bc_eval_log_interval = 2
        ppo._fixed_bc_eval_ready = True
        ppo._fixed_bc_eval_dataset = {
            "actor_obs_raw": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "teacher_actions": torch.tensor([[2.0, 3.0], [5.0, 4.0]]),
        }
        ppo._get_distributed_loss_weight = MethodType(lambda self: 1.0, ppo)

        def noisy_reduce(self, tensor, op):
            random.random()
            np.random.random()
            torch.rand(1)
            if tensor.numel() == 1:
                return tensor
            return tensor + tensor.new_tensor([8.0, 4.0, 2.0, 2.0, 1.0])

        ppo._all_reduce_small_tensor = MethodType(noisy_reduce, ppo)
        random.seed(1101)
        np.random.seed(1102)
        torch.manual_seed(1103)
        boundary = capture_rng_checkpoint_state()

        # The interval-disabled diagnostic is the reference trajectory.
        assert ppo._get_fixed_bc_eval_metrics(current_iteration=1) == {}
        disabled_next = (random.random(), float(np.random.random()), torch.rand(3))

        restore_rng_checkpoint_state(boundary)
        metrics = ppo._get_fixed_bc_eval_metrics(current_iteration=2)
        enabled_next = (random.random(), float(np.random.random()), torch.rand(3))

        assert metrics["fixed_bc_num_samples"] == 4.0
        assert enabled_next[0] == disabled_next[0]
        assert enabled_next[1] == disabled_next[1]
        assert torch.equal(enabled_next[2], disabled_next[2])
    finally:
        restore_rng_checkpoint_state(original_rng)


def test_fixed_bc_eval_random_actor_uses_stable_rank_specific_draw():
    original_rng = capture_rng_checkpoint_state()
    try:
        ppo = _make_stub_ppo()
        ppo.actor = _RngConsumingActor()
        ppo._fixed_bc_eval_ready = True
        ppo._fixed_bc_eval_dataset = {
            "actor_obs_raw": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "teacher_actions": torch.tensor([[2.0, 3.0], [5.0, 4.0]]),
        }

        random.seed(4101)
        np.random.seed(4102)
        torch.manual_seed(4103)
        first = ppo._get_fixed_bc_eval_metrics(current_iteration=0)

        random.seed(5101)
        np.random.seed(5102)
        torch.manual_seed(5103)
        second = ppo._get_fixed_bc_eval_metrics(current_iteration=1)

        assert second == first
    finally:
        restore_rng_checkpoint_state(original_rng)


def test_fixed_bc_eval_terminal_force_bypasses_interval_and_not_ready_fails_closed():
    ppo = _make_stub_ppo()
    ppo.fixed_bc_eval_log_interval = 100
    ppo._fixed_bc_eval_ready = True
    ppo._fixed_bc_eval_size = 2
    ppo._fixed_bc_eval_dataset = {
        "actor_obs_raw": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "teacher_actions": torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
    }

    assert ppo._get_fixed_bc_eval_metrics(current_iteration=7) == {}
    forced = ppo._get_fixed_bc_eval_metrics(
        current_iteration=7,
        terminal_observation=True,
    )

    assert forced["fixed_bc_num_samples"] == 2.0
    assert forced["fixed_bc_mu_mse"] == 0.0
    ppo._fixed_bc_eval_ready = False
    with pytest.raises(RuntimeError, match="Final fixed-BC observation is unavailable"):
        ppo._get_fixed_bc_eval_metrics(
            current_iteration=7,
            terminal_observation=True,
        )


def test_fixed_bc_eval_seed_failure_restores_every_rng_stream(monkeypatch):
    original_rng = capture_rng_checkpoint_state()
    original_manual_seed = torch.manual_seed
    try:
        ppo = _make_stub_ppo()
        random.seed(7101)
        np.random.seed(7102)
        original_manual_seed(7103)
        boundary = capture_rng_checkpoint_state()
        expected = (random.random(), float(np.random.random()), torch.rand(3))
        restore_rng_checkpoint_state(boundary)

        def partially_mutating_seed(seed):
            random.random()
            np.random.random()
            original_manual_seed(seed)
            torch.rand(1)
            raise RuntimeError("synthetic deterministic seed failure")

        monkeypatch.setattr(torch, "manual_seed", partially_mutating_seed)
        with pytest.raises(RuntimeError, match="synthetic deterministic seed failure"):
            ppo._get_fixed_bc_eval_metrics(current_iteration=0)

        assert random.random() == expected[0]
        assert float(np.random.random()) == expected[1]
        assert torch.equal(torch.rand(3), expected[2])
    finally:
        restore_rng_checkpoint_state(original_rng)


def test_terminal_fixed_bc_state_binds_dataset_guard_threshold_and_saved_target():
    ppo, state, fixed_states, guard_state = _make_terminal_guard_state()
    payload = _terminal_checkpoint_payload(
        ppo,
        state,
        fixed_states,
        guard_state,
    )

    restored = ppo._validate_terminal_fixed_bc_eval_artifact_payload(
        payload,
        expected_completed_iteration=3,
        compare_runtime_guard_config=True,
    )

    assert restored == state
    assert state["scheduled_evaluation"] is False
    assert state["guard_applied"] is False
    assert state["fixed_bc_guard_threshold_mu_mse"] == pytest.approx(0.06)
    assert state["fixed_bc_terminal_within_threshold"] is True
    assert state["fixed_bc_weighted_num_samples"] == pytest.approx(
        state["fixed_bc_expected_weighted_num_samples"]
    )

    bad_digest = copy.deepcopy(payload)
    bad_digest["terminal_fixed_bc_eval_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not authenticate its state"):
        ppo._validate_terminal_fixed_bc_eval_artifact_payload(
            bad_digest,
            expected_completed_iteration=3,
        )

    bad_dataset = copy.deepcopy(payload)
    bad_dataset["fixed_bc_eval_by_rank"]["0"]["actor_obs_raw"][0, 0] += 1.0
    with pytest.raises(ValueError, match="does not authenticate.*frozen dataset"):
        ppo._validate_terminal_fixed_bc_eval_artifact_payload(
            bad_dataset,
            expected_completed_iteration=3,
        )

    nonfinal = copy.deepcopy(payload)
    nonfinal["experiment_config"]["algo"]["config"][
        "num_learning_iterations"
    ] = 5
    with pytest.raises(ValueError, match="non-final checkpoint"):
        ppo._validate_terminal_fixed_bc_eval_artifact_payload(
            nonfinal,
            expected_completed_iteration=3,
        )

    bad_guard = copy.deepcopy(payload)
    bad_guard["fixed_bc_guard_state"]["threshold_mu_mse"] = 0.05
    with pytest.raises(ValueError, match="periodic guard state"):
        ppo._validate_terminal_fixed_bc_eval_artifact_payload(
            bad_guard,
            expected_completed_iteration=3,
        )


def test_terminal_fixed_bc_state_fails_closed_above_frozen_threshold():
    ppo, _, _, _ = _make_terminal_guard_state()
    _, global_dataset_digest = ppo._fixed_bc_guard_live_dataset_digests()

    with pytest.raises(RuntimeError, match="exceeds the frozen scientific guard threshold"):
        ppo._build_terminal_fixed_bc_eval_state(
            completed_iteration=3,
            metrics=_guard_metrics(0.061),
            scheduled_evaluation=False,
            global_dataset_digest=global_dataset_digest,
            expected_weighted_num_samples=2.0,
        )


def test_terminal_fixed_bc_state_guard_disabled_uses_strict_null_semantics():
    ppo = _make_stub_ppo()
    ppo.config = SimpleNamespace(num_learning_iterations=4)
    ppo.fixed_bc_eval_log_interval = 2
    ppo.fixed_bc_guard_enabled = False
    ppo._fixed_bc_guard_config_fingerprint = (
        ppo._fixed_bc_guard_runtime_config_fingerprint()
    )

    state = ppo._build_terminal_fixed_bc_eval_state(
        completed_iteration=3,
        metrics=_guard_metrics(0.04),
        scheduled_evaluation=False,
        global_dataset_digest="a" * 64,
        expected_weighted_num_samples=2.0,
    )

    assert state["fixed_bc_guard_state_sha256"] is None
    assert state["fixed_bc_guard_threshold_mu_mse"] is None
    assert state["fixed_bc_terminal_within_threshold"] is None


@pytest.mark.parametrize(
    ("metric_key", "bad_value"),
    [
        ("fixed_bc_mu_mse", True),
        ("fixed_bc_mu_mse", "0.04"),
        ("fixed_bc_weighted_num_samples", False),
        ("fixed_bc_weighted_num_samples", "2.0"),
        ("fixed_bc_expected_weighted_num_samples", "2.0"),
    ],
)
def test_terminal_fixed_bc_builder_rejects_coerced_metrics(metric_key, bad_value):
    ppo = _make_stub_ppo()
    ppo.config = SimpleNamespace(num_learning_iterations=4)
    ppo.fixed_bc_eval_log_interval = 2
    ppo.fixed_bc_guard_enabled = False
    ppo._fixed_bc_guard_config_fingerprint = (
        ppo._fixed_bc_guard_runtime_config_fingerprint()
    )
    metrics = _guard_metrics(0.04)
    expected_weighted_num_samples = 2.0
    if metric_key == "fixed_bc_expected_weighted_num_samples":
        expected_weighted_num_samples = bad_value
    else:
        metrics[metric_key] = bad_value

    with pytest.raises(ValueError, match="must be a real scalar"):
        ppo._build_terminal_fixed_bc_eval_state(
            completed_iteration=3,
            metrics=metrics,
            scheduled_evaluation=False,
            global_dataset_digest="a" * 64,
            expected_weighted_num_samples=expected_weighted_num_samples,
        )


def test_terminal_fixed_bc_proof_construction_preserves_all_rng_streams():
    original_rng = capture_rng_checkpoint_state()
    try:
        ppo = _make_stub_ppo()
        ppo.config = SimpleNamespace(num_learning_iterations=4)
        ppo.fixed_bc_eval_log_interval = 2
        ppo.fixed_bc_guard_enabled = False
        ppo._fixed_bc_guard_config_fingerprint = (
            ppo._fixed_bc_guard_runtime_config_fingerprint()
        )

        def consume_rng():
            random.random()
            np.random.random()
            torch.rand(1)

        ppo._fixed_bc_guard_live_dataset_digests = Mock(
            side_effect=lambda: consume_rng() or ({"0": "a" * 64}, "a" * 64)
        )
        ppo._fixed_bc_guard_expected_weighted_sample_count = Mock(
            side_effect=lambda: consume_rng() or 2.0
        )
        ppo._require_all_rank_fixed_bc_guard_state_match = Mock(
            side_effect=lambda _state: consume_rng()
        )
        random.seed(8101)
        np.random.seed(8102)
        torch.manual_seed(8103)
        boundary = capture_rng_checkpoint_state()
        expected = (random.random(), float(np.random.random()), torch.rand(3))
        restore_rng_checkpoint_state(boundary)

        state = ppo._build_terminal_fixed_bc_eval_state_preserving_rng(
            completed_iteration=3,
            metrics=_guard_metrics(0.04),
            scheduled_evaluation=False,
        )

        assert state["completed_iteration"] == 3
        assert random.random() == expected[0]
        assert float(np.random.random()) == expected[1]
        assert torch.equal(torch.rand(3), expected[2])
    finally:
        restore_rng_checkpoint_state(original_rng)


def _make_terminal_learn_stub(
    events: list,
    *,
    final_eval_error: Exception | None = None,
) -> tuple[PPO, list[dict[str, float]]]:
    ppo = object.__new__(PPO)
    ppo.current_learning_iteration = 6
    ppo.config = SimpleNamespace(
        num_learning_iterations=8,
        save_interval=4,
        init_at_random_ep_len=False,
    )
    ppo.device = "cpu"
    ppo.log_dir = "/tmp/terminal-fixed-bc-learn-test"
    ppo.is_multi_gpu = False
    ppo.is_main_process = False
    ppo.gpu_world_size = 1
    ppo.gpu_global_rank = 0
    ppo.dagger_enabled = True
    ppo.fixed_bc_eval_num_samples = 2
    ppo.fixed_bc_eval_log_interval = 2
    ppo.fixed_bc_guard_enabled = False
    ppo._fixed_bc_guard_config_fingerprint = "b" * 64
    ppo._terminal_fixed_bc_eval_state = None
    ppo._experiment_config = SimpleNamespace(
        training=SimpleNamespace(export_onnx=False)
    )
    ppo.algo_timing = SimpleNamespace(enabled=False)
    ppo.logging_helper = SimpleNamespace(
        record_collection_time=lambda: nullcontext(),
        record_learn_time=lambda: nullcontext(),
    )
    ppo._validate_future_dagger_bc_mask_signal = Mock()
    ppo._train_mode = Mock()
    ppo._prepare_rollout_objective_for_iteration = Mock()
    ppo._reset_rollout_stream_at_canonical_boundary = Mock(
        return_value={"obs": torch.zeros(1, 1)}
    )
    ppo._reset_step_timing = Mock()
    ppo._sync_iteration_boundary = Mock()
    ppo._sync_training_curriculum_state = Mock()
    ppo._curriculum_state_sync_enabled = Mock(return_value=False)
    ppo._rollout_step = Mock(side_effect=lambda obs: obs)

    def train_step():
        events.append(("train", ppo.current_learning_iteration))
        return {}

    ppo._training_step = Mock(side_effect=train_step)
    ppo._capture_step_timing = Mock()
    ppo._emit_step_timing_summary = Mock()
    metrics = _guard_metrics(0.04)

    def evaluate(*, current_iteration, terminal_observation=False):
        events.append(("eval", current_iteration, terminal_observation))
        if terminal_observation and final_eval_error is not None:
            raise final_eval_error
        return dict(metrics)

    ppo._get_fixed_bc_eval_metrics = Mock(side_effect=evaluate)

    def guard(*, current_iteration, metrics):
        events.append(("guard", current_iteration, dict(metrics)))
        return False

    ppo._update_fixed_bc_guard = Mock(side_effect=guard)
    ppo._fixed_bc_guard_log_metrics = Mock(return_value={})
    ppo._fixed_bc_guard_live_dataset_digests = Mock(
        side_effect=lambda: events.append(("digest",)) or ({"0": "a" * 64}, "a" * 64)
    )
    ppo._fixed_bc_guard_expected_weighted_sample_count = Mock(
        side_effect=lambda: events.append(("weight",)) or 2.0
    )
    original_build = ppo._build_terminal_fixed_bc_eval_state

    def build_terminal(**kwargs):
        events.append(
            (
                "build",
                kwargs["completed_iteration"],
                kwargs["scheduled_evaluation"],
            )
        )
        return original_build(**kwargs)

    ppo._build_terminal_fixed_bc_eval_state = Mock(side_effect=build_terminal)
    ppo._require_all_rank_fixed_bc_guard_state_match = Mock(
        side_effect=lambda state: events.append(("agree", state["completed_iteration"]))
    )
    logged_metrics: list[dict[str, float]] = []

    def log(it, _losses, *, fixed_bc_eval_metrics):
        logged_metrics.append(dict(fixed_bc_eval_metrics))
        events.append(("log", it))

    ppo._post_epoch_logging_preserving_rng = Mock(side_effect=log)
    ppo._distributed_barrier = Mock(side_effect=lambda: events.append(("barrier",)))
    ppo._save_checkpoint_with_distributed_outcome = Mock(
        side_effect=lambda path, *, next_iteration: events.append(
            ("save", next_iteration)
        )
    )
    ppo._export_final_onnx_with_distributed_outcome = Mock(
        side_effect=lambda path, *, iteration: events.append(("export", iteration))
        or None
    )
    ppo._is_node_local_main_process = Mock(return_value=False)
    return ppo, logged_metrics


def test_learn_terminal_fixed_bc_is_single_log_guard_isolated_and_before_artifacts():
    events: list = []
    ppo, logged_metrics = _make_terminal_learn_stub(events)

    ppo.learn()

    assert events == [
        ("train", 6),
        ("eval", 6, False),
        ("guard", 6, _guard_metrics(0.04)),
        ("log", 6),
        ("train", 7),
        ("eval", 7, True),
        ("digest",),
        ("weight",),
        ("build", 7, False),
        ("agree", 7),
        ("log", 7),
        ("barrier",),
        ("save", 8),
        ("export", 7),
        ("barrier",),
    ]
    assert len(logged_metrics) == 2
    terminal_log = logged_metrics[-1]
    assert terminal_log["fixed_bc_terminal_observation"] == 1.0
    assert terminal_log["fixed_bc_scheduled_evaluation"] == 0.0
    assert terminal_log["fixed_bc_guard_applied"] == 0.0
    assert terminal_log["fixed_bc_final_mu_mse"] == pytest.approx(0.04)
    ppo._save_checkpoint_with_distributed_outcome.assert_called_once()
    ppo._export_final_onnx_with_distributed_outcome.assert_called_once()


def test_learn_off_cadence_skips_fixed_bc_eval_and_guard_protocols():
    events: list = []
    ppo, _ = _make_terminal_learn_stub(events)
    ppo.current_learning_iteration = 5
    ppo.config.num_learning_iterations = 7

    ppo.learn()

    train_5_index = events.index(("train", 5))
    log_5_index = events.index(("log", 5))
    assert events[train_5_index : log_5_index + 1] == [
        ("train", 5),
        ("log", 5),
    ]
    assert not any(
        event[0] in {"eval", "guard"} and event[1] == 5
        for event in events
    )
    assert ("eval", 6, True) in events
    assert ("guard", 6, _guard_metrics(0.04)) in events


def test_learn_terminal_fixed_bc_failure_stops_before_final_log_save_and_export():
    events: list = []
    ppo, logged_metrics = _make_terminal_learn_stub(
        events,
        final_eval_error=RuntimeError("terminal stratum not frozen"),
    )

    with pytest.raises(RuntimeError, match="terminal stratum not frozen"):
        ppo.learn()

    assert events == [
        ("train", 6),
        ("eval", 6, False),
        ("guard", 6, _guard_metrics(0.04)),
        ("log", 6),
        ("train", 7),
        ("eval", 7, True),
    ]
    assert len(logged_metrics) == 1
    ppo._save_checkpoint_with_distributed_outcome.assert_not_called()
    ppo._export_final_onnx_with_distributed_outcome.assert_not_called()


def _configure_terminal_checkpoint_save(
    ppo: PPO,
    state: dict,
    fixed_states: dict[str, dict],
) -> list[dict]:
    ppo._terminal_fixed_bc_eval_state = state
    ppo.is_main_process = True
    ppo.current_learning_iteration = 4
    ppo.critic = nn.Linear(1, 1)
    ppo.actor_optimizer = SimpleNamespace(state_dict=lambda: {})
    ppo.critic_optimizer = SimpleNamespace(state_dict=lambda: {})
    ppo.critic_obs_normalizers = {}
    ppo._validate_checkpoint_publish_state = Mock()
    rng_state = capture_rng_checkpoint_state()
    ppo._collect_distributed_rng_states = Mock(return_value={"0": rng_state})
    ppo._collect_distributed_env_states = Mock(return_value={"0": {}})
    ppo._collect_distributed_motion_transition_contract = Mock(
        return_value=(None, None)
    )
    ppo._aggregate_actor_perception_geometry_support = Mock(return_value=None)
    ppo._collect_distributed_fixed_bc_eval_states = Mock(
        return_value=fixed_states
    )
    ppo._rollout_resume_contract = Mock(return_value={"version": 1})
    ppo._checkpoint_metadata = MethodType(
        lambda self, iteration=None: {
            "experiment_config": {
                "algo": {"config": {"num_learning_iterations": 4}}
            },
            "iteration": int(iteration),
        },
        ppo,
    )
    published: list[dict] = []
    ppo.logging_helper = SimpleNamespace(
        save_checkpoint_artifact=lambda checkpoint, _path: published.append(
            copy.deepcopy(checkpoint)
        )
    )
    return published


def test_checkpoint_save_only_binds_terminal_fixed_bc_to_final_artifact():
    ppo, state, fixed_states, _ = _make_terminal_guard_state()
    published = _configure_terminal_checkpoint_save(ppo, state, fixed_states)

    ppo.save("model_00003.pt", next_iteration=3)
    ppo.save("model_00004.pt", next_iteration=4)

    assert "terminal_fixed_bc_eval" not in published[0]
    assert "terminal_fixed_bc_eval_sha256" not in published[0]
    assert published[1]["terminal_fixed_bc_eval"] == state
    assert published[1]["terminal_fixed_bc_eval_sha256"] == (
        ppo._terminal_fixed_bc_eval_state_sha256(state)
    )
    ppo._validate_terminal_fixed_bc_eval_artifact_payload(
        published[1],
        expected_completed_iteration=3,
        compare_runtime_guard_config=True,
    )


def test_checkpoint_save_requires_terminal_fixed_bc_for_configured_final():
    ppo, state, fixed_states, _ = _make_terminal_guard_state()
    published = _configure_terminal_checkpoint_save(ppo, state, fixed_states)
    ppo._terminal_fixed_bc_eval_state = None

    with pytest.raises(RuntimeError, match="final DAgger checkpoint"):
        ppo.save("model_00004.pt", next_iteration=4)

    assert published == []


def test_aligned_final_guard_trip_diagnostic_omits_terminal_success_proof():
    ppo, state, fixed_states, _ = _make_terminal_guard_state()
    published = _configure_terminal_checkpoint_save(ppo, state, fixed_states)
    ppo._fixed_bc_guard_checkpoint_state = Mock(
        return_value={"tripped": True}
    )

    ppo.save(
        "diagnostic_fixed_bc_guard_00004.pt",
        next_iteration=4,
        allow_tripped_fixed_bc_guard=True,
    )

    assert len(published) == 1
    assert published[0]["fixed_bc_guard_state"] == {"tripped": True}
    assert "terminal_fixed_bc_eval" not in published[0]
    assert "terminal_fixed_bc_eval_sha256" not in published[0]


def _configure_terminal_onnx_export(ppo: PPO, state: dict) -> None:
    ppo._terminal_fixed_bc_eval_state = state
    ppo.current_learning_iteration = 4
    ppo.actor.train()
    ppo.actor_perception_key = ""
    ppo._prepare_motion_transition_contract_for_export = Mock()
    ppo._eval_mode = MethodType(lambda self: self.actor.eval(), ppo)
    ppo._train_mode = MethodType(lambda self: self.actor.train(), ppo)
    ppo._get_zero_input = MethodType(lambda self: torch.zeros(1, 1), ppo)
    ppo._get_zero_perception_input = MethodType(lambda self: None, ppo)
    ppo.env = SimpleNamespace(robot_config=SimpleNamespace(dof_names=[]))
    ppo.logging_helper = SimpleNamespace(save_to_wandb=Mock())
    ppo._checkpoint_metadata = MethodType(
        lambda self, iteration=None: {
            "experiment_config": {
                "algo": {"config": {"num_learning_iterations": 4}}
            },
            "iteration": int(iteration),
        },
        ppo,
    )


def test_onnx_export_binds_same_terminal_fixed_bc_state_and_digest():
    ppo, state, _, _ = _make_terminal_guard_state()
    _configure_terminal_onnx_export(ppo, state)
    metadata_by_iteration: list[dict] = []

    with (
        patch.object(
            PPO,
            "actor_onnx_wrapper",
            new_callable=PropertyMock,
            return_value=object(),
        ),
        patch("holosoma.agents.ppo.ppo.export_policy_as_onnx"),
        patch(
            "holosoma.agents.ppo.ppo.get_control_gains_from_config",
            return_value=([], []),
        ),
        patch("holosoma.agents.ppo.ppo.get_command_ranges_from_env", return_value={}),
        patch(
            "holosoma.agents.ppo.ppo.get_urdf_text_from_robot_config",
            return_value=("", ""),
        ),
        patch(
            "holosoma.agents.ppo.ppo.attach_onnx_metadata",
            side_effect=lambda *, onnx_path, metadata: metadata_by_iteration.append(
                copy.deepcopy(metadata)
            ),
        ),
    ):
        ppo.export("periodic.onnx", iteration=2)
        ppo.export("final.onnx", iteration=3)

    assert "terminal_fixed_bc_eval" not in metadata_by_iteration[0]
    assert "terminal_fixed_bc_eval_sha256" not in metadata_by_iteration[0]
    assert metadata_by_iteration[1]["terminal_fixed_bc_eval"] == state
    assert metadata_by_iteration[1]["terminal_fixed_bc_eval_sha256"] == (
        ppo._terminal_fixed_bc_eval_state_sha256(state)
    )


def test_onnx_export_requires_terminal_fixed_bc_for_configured_final():
    ppo, state, _, _ = _make_terminal_guard_state()
    _configure_terminal_onnx_export(ppo, state)
    ppo._terminal_fixed_bc_eval_state = None

    with (
        patch.object(
            PPO,
            "actor_onnx_wrapper",
            new_callable=PropertyMock,
            return_value=object(),
        ),
        patch("holosoma.agents.ppo.ppo.export_policy_as_onnx"),
        patch(
            "holosoma.agents.ppo.ppo.get_control_gains_from_config",
            return_value=([], []),
        ),
        patch("holosoma.agents.ppo.ppo.get_command_ranges_from_env", return_value={}),
        patch(
            "holosoma.agents.ppo.ppo.get_urdf_text_from_robot_config",
            return_value=("", ""),
        ),
        patch("holosoma.agents.ppo.ppo.attach_onnx_metadata") as attach,
        pytest.raises(RuntimeError, match="final DAgger ONNX"),
    ):
        ppo.export("final.onnx", iteration=3)

    attach.assert_not_called()
    ppo.logging_helper.save_to_wandb.assert_not_called()


def test_incomplete_fixed_capture_is_rejected_after_dagger_weight_reaches_zero() -> None:
    ppo = _make_stub_ppo()
    ppo.distill_mode = "dagger"
    ppo.use_ppo_dagger_schedule = True
    ppo.ppo_coeff = 1.0
    ppo.dagger_loss_coef = 1.0
    ppo.take_teacher_actions = False
    ppo.teacher_action_mix_ratio = 0.0
    ppo._fixed_bc_eval_ready = False

    error = ppo._pure_ppo_fixed_capture_error()

    assert isinstance(error, RuntimeError)
    assert "nominal pure-PPO stream" in str(error)
    ppo._fixed_bc_eval_ready = True
    assert ppo._pure_ppo_fixed_capture_error() is None


def test_observational_teacher_selection_restores_all_training_rng_streams() -> None:
    class _RngTeacher(_DummyTeacherActor):
        def act_inference(self, policy_state_dict):
            random.random()
            np.random.random()
            return policy_state_dict["actor_obs"] + torch.randn_like(
                policy_state_dict["actor_obs"]
            )

    original_rng = capture_rng_checkpoint_state()
    try:
        ppo = _make_stub_ppo()
        ppo.use_multi_teacher = False
        ppo.teacher_actor = _RngTeacher()
        ppo.teacher_actor_obs_normalizers = {}
        ppo.teacher_use_stochastic_actions = False
        ppo._normalize_teacher_actor_obs = MethodType(
            lambda self, obs, normalizers=None: obs,
            ppo,
        )
        random.seed(3101)
        np.random.seed(3102)
        torch.manual_seed(3103)
        boundary = capture_rng_checkpoint_state()

        reference_next = (random.random(), float(np.random.random()), torch.rand(3))
        restore_rng_checkpoint_state(boundary)
        actions, indices, error = ppo._try_select_teacher_actions_for_rollout(
            torch.ones(2, 1),
            {},
            preserve_rng=True,
        )
        observed_next = (random.random(), float(np.random.random()), torch.rand(3))

        assert error is None
        assert actions is not None
        assert indices is None
        assert observed_next[0] == reference_next[0]
        assert observed_next[1] == reference_next[1]
        assert torch.equal(observed_next[2], reference_next[2])
    finally:
        restore_rng_checkpoint_state(original_rng)


def test_fixed_bc_eval_restores_actor_and_normalizer_modes_on_failure():
    class RaisingActor(_RngConsumingActor):
        def act_inference(self, policy_state_dict):
            super().act_inference(policy_state_dict)
            raise RuntimeError("synthetic fixed-BC inference failure")

    ppo = _make_stub_ppo()
    ppo.actor = RaisingActor().train(True)
    ppo.actor_obs_normalizers["actor_obs"].train(False)
    ppo._fixed_bc_eval_ready = True
    ppo._fixed_bc_eval_dataset = {
        "actor_obs_raw": torch.tensor([[1.0], [2.0]]),
        "teacher_actions": torch.tensor([[2.0], [3.0]]),
    }

    with pytest.raises(RuntimeError, match="synthetic fixed-BC inference failure"):
        ppo._get_fixed_bc_eval_metrics(current_iteration=0)

    assert ppo.actor.training is True
    assert ppo.actor_obs_normalizers["actor_obs"].training is False


def test_fixed_bc_eval_synchronizes_local_failure_before_statistics_collective():
    class RaisingActor(_DummyActor):
        def act_inference(self, policy_state_dict):
            raise RuntimeError("synthetic rank-local fixed-BC failure")

    ppo = _make_stub_ppo()
    ppo.actor = RaisingActor()
    ppo.is_multi_gpu = True
    ppo.gpu_world_size = 2
    ppo.fixed_bc_eval_num_samples = 4
    ppo._fixed_bc_eval_ready = True
    ppo._fixed_bc_eval_dataset = {
        "actor_obs_raw": torch.tensor([[1.0], [2.0]]),
        "teacher_actions": torch.tensor([[2.0], [3.0]]),
    }
    ppo._get_distributed_loss_weight = MethodType(lambda self: 1.0, ppo)
    collective_sizes = []
    ppo._all_reduce_small_tensor = MethodType(
        lambda self, tensor, op: collective_sizes.append(tensor.numel()) or tensor,
        ppo,
    )
    synchronized_errors = []

    def synchronize(self, local_error, *, operation):
        synchronized_errors.append((local_error, operation))
        if local_error is not None:
            raise local_error

    ppo._synchronize_training_phase_error = MethodType(synchronize, ppo)

    with pytest.raises(RuntimeError, match="synthetic rank-local fixed-BC failure"):
        ppo._get_fixed_bc_eval_metrics(current_iteration=0)

    assert any(
        operation == "fixed BC evaluation local inference"
        and isinstance(error, RuntimeError)
        for error, operation in synchronized_errors
    )
    # Only the readiness scalar ran; the five-field statistics collective was
    # never entered after the local failure verdict.
    assert collective_sizes == [1]


def test_fixed_bc_eval_checkpoint_state_restores_exact_dataset():
    ppo = _make_stub_ppo()
    ppo.current_learning_iteration = 7
    ppo._fixed_bc_eval_ready = True
    ppo._fixed_bc_eval_size = 2
    ppo._fixed_bc_eval_dataset = {
        "actor_obs_raw": torch.tensor([[1.0], [2.0]]),
        "teacher_actions": torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
    }
    state = ppo._local_fixed_bc_eval_checkpoint_state()
    assert state is not None

    resumed = _make_stub_ppo()
    resumed.current_learning_iteration = 7
    resumed._restore_fixed_bc_eval_checkpoint_state({"fixed_bc_eval_by_rank": {"0": state}})

    assert resumed._fixed_bc_eval_ready is True
    assert resumed._fixed_bc_eval_size == 2
    assert torch.equal(
        resumed._fixed_bc_eval_dataset["teacher_actions"],
        ppo._fixed_bc_eval_dataset["teacher_actions"],
    )


def test_partial_fixed_bc_eval_checkpoint_resumes_capture_exactly():
    source = _make_stub_ppo()
    source._fixed_bc_eval_size = 1
    source._fixed_bc_eval_actor_obs_parts = [torch.tensor([[1.0]])]
    source._fixed_bc_eval_teacher_actions_parts = [torch.tensor([[2.0, 3.0]])]
    state = source._local_fixed_bc_eval_checkpoint_state()
    assert state is not None and state["ready"] is False and state["size"] == 1

    resumed = _make_stub_ppo()
    resumed.current_learning_iteration = 7
    resumed._restore_fixed_bc_eval_checkpoint_state({"fixed_bc_eval_by_rank": {"0": state}})

    assert resumed._fixed_bc_eval_ready is False
    assert resumed._fixed_bc_eval_size == 1
    resumed._maybe_capture_fixed_bc_eval_samples(
        actor_obs_raw=torch.tensor([[4.0]]),
        actor_perception_obs=None,
        teacher_actions=torch.tensor([[5.0, 6.0]]),
        teacher_bc_mask=None,
    )
    assert resumed._fixed_bc_eval_ready is True
    assert torch.equal(resumed._fixed_bc_eval_dataset["actor_obs_raw"], torch.tensor([[1.0], [4.0]]))


def test_fixed_bc_capture_remains_pending_after_bc_switch_until_ready():
    ppo = _make_stub_ppo()
    ppo.bc_loss_coef = 0.0
    ppo.use_ppo_dagger_schedule = False

    assert ppo._fixed_bc_eval_capture_pending() is True
    ppo._fixed_bc_eval_ready = True
    assert ppo._fixed_bc_eval_capture_pending() is False


def test_fixed_bc_eval_resume_without_dataset_fails_closed():
    ppo = _make_stub_ppo()
    ppo.current_learning_iteration = 7

    with pytest.raises(RuntimeError, match="fixed BC evaluation stratum"):
        ppo._restore_fixed_bc_eval_checkpoint_state({})


def test_legacy_fixed_bc_eval_state_fails_closed():
    ppo = _make_stub_ppo()
    ppo.current_learning_iteration = 7
    legacy_state = {
        "actor_obs_raw": torch.tensor([[1.0], [2.0]]),
        "teacher_actions": torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
    }

    with pytest.raises(RuntimeError, match="predates the exact allocation contract"):
        ppo._restore_fixed_bc_eval_checkpoint_state({"fixed_bc_eval_by_rank": {"0": legacy_state}})


def test_legacy_fixed_bc_eval_state_can_only_be_explicitly_reset(monkeypatch):
    ppo = _make_stub_ppo()
    ppo.current_learning_iteration = 7
    ppo._fixed_bc_eval_size = 1
    ppo._fixed_bc_eval_actor_obs_parts = [torch.tensor([[9.0]])]
    ppo._fixed_bc_eval_teacher_actions_parts = [torch.tensor([[9.0, 9.0]])]
    monkeypatch.setenv("HOLOSOMA_ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME", "1")

    ppo._restore_fixed_bc_eval_checkpoint_state(
        {
            "fixed_bc_eval_by_rank": {
                "0": {
                    "actor_obs_raw": torch.tensor([[1.0], [2.0]]),
                    "teacher_actions": torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
                }
            }
        }
    )

    assert ppo._fixed_bc_eval_ready is False
    assert ppo._fixed_bc_eval_size == 0
    assert ppo._fixed_bc_eval_actor_obs_parts == []


@pytest.mark.parametrize("invalid_size", [True, 1.9, "1"])
def test_fixed_bc_resume_requires_strict_integer_size(invalid_size):
    source = _make_stub_ppo()
    source._fixed_bc_eval_ready = True
    source._fixed_bc_eval_size = 2
    source._fixed_bc_eval_dataset = {
        "actor_obs_raw": torch.tensor([[1.0], [2.0]]),
        "teacher_actions": torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
    }
    state = source._local_fixed_bc_eval_checkpoint_state()
    assert state is not None
    state["size"] = invalid_size

    resumed = _make_stub_ppo()
    resumed.current_learning_iteration = 7
    with pytest.raises(ValueError, match="size must be an integer"):
        resumed._restore_fixed_bc_eval_checkpoint_state({"fixed_bc_eval_by_rank": {"0": state}})


@pytest.mark.parametrize(
    ("replacement", "expected"),
    [
        (torch.zeros(2, 1, 1), "dense rank-2 tensor"),
        (torch.zeros(2, 1, dtype=torch.float64), "dtype torch.float64"),
    ],
)
def test_fixed_bc_resume_rejects_shape_or_dtype_before_live_state_mutation(replacement, expected):
    source = _make_stub_ppo()
    source._fixed_bc_eval_ready = True
    source._fixed_bc_eval_size = 2
    source._fixed_bc_eval_dataset = {
        "actor_obs_raw": torch.tensor([[1.0], [2.0]]),
        "teacher_actions": torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
    }
    state = source._local_fixed_bc_eval_checkpoint_state()
    assert state is not None
    state["actor_obs_raw"] = replacement

    resumed = _make_stub_ppo()
    resumed.current_learning_iteration = 7
    resumed._fixed_bc_eval_size = 1
    resumed._fixed_bc_eval_actor_obs_parts = [torch.tensor([[9.0]])]
    before = resumed._fixed_bc_eval_actor_obs_parts[0].clone()
    with pytest.raises(ValueError, match=expected):
        resumed._restore_fixed_bc_eval_checkpoint_state({"fixed_bc_eval_by_rank": {"0": state}})

    assert resumed._fixed_bc_eval_size == 1
    assert torch.equal(resumed._fixed_bc_eval_actor_obs_parts[0], before)


def test_any_rank_fixed_bc_reset_plan_forces_global_reset(monkeypatch):
    source_states = {}
    for rank in range(2):
        source = _make_stub_ppo()
        source.is_multi_gpu = True
        source.gpu_world_size = 2
        source.gpu_global_rank = rank
        source._fixed_bc_eval_ready = True
        source._fixed_bc_eval_size = 1
        source._fixed_bc_eval_dataset = {
            "actor_obs_raw": torch.tensor([[float(rank + 1)]]),
            "teacher_actions": torch.tensor([[1.0, 2.0]]),
        }
        source_states[str(rank)] = source._local_fixed_bc_eval_checkpoint_state()

    resumed = _make_stub_ppo()
    resumed.is_multi_gpu = True
    resumed.gpu_world_size = 2
    resumed.gpu_global_rank = 0
    resumed._setup_gloo_barrier_group = MethodType(lambda self: object(), resumed)
    monkeypatch.setenv("HOLOSOMA_ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME", "1")
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)

    def gather(results, local_result, group):
        assert group is not None
        results[0] = local_result
        results[1] = {
            "rank": 1,
            "error": None,
            "action": "reset",
            "message": "rank 1 has no compatible stratum",
            "contract": {
                **local_result["contract"],
                "rank": 1,
                "local_target": 1,
            },
        }

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    plan = resumed._prepare_fixed_bc_eval_checkpoint_state_all_ranks(
        {"fixed_bc_eval_by_rank": source_states},
        next_iteration=7,
    )

    assert plan["action"] == "reset"
    assert "rank=1" in plan["message"]


def test_fixed_bc_checkpoint_local_validation_failure_is_gathered_before_raise(monkeypatch):
    ppo = _make_stub_ppo()
    ppo.is_multi_gpu = True
    ppo.gpu_world_size = 2
    ppo.gpu_global_rank = 0
    ppo._fixed_bc_eval_ready = True
    ppo._fixed_bc_eval_size = 0  # rank 0 target is one sample
    gloo_group = object()
    ppo._setup_gloo_barrier_group = MethodType(lambda self: gloo_group, ppo)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    gather_calls = []

    def gather(results, local_result, group):
        gather_calls.append(local_result)
        assert group is gloo_group
        assert local_result["rank"] == 0
        assert "inconsistent ready fixed BC stratum" in local_result["error"]
        assert local_result["state"] is None
        results[0] = local_result
        results[1] = {
            "rank": 1,
            "error": None,
            "state": {"rank": 1},
            "contract": {
                **local_result["contract"],
                "rank": 1,
                "local_target": 1,
            },
        }

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    with pytest.raises(RuntimeError, match=r"rank=0: RuntimeError:.*inconsistent ready"):
        ppo._collect_distributed_fixed_bc_eval_states()

    assert len(gather_calls) == 1


def test_non_dagger_fixed_bc_checkpoint_state_is_a_collective_free_noop(monkeypatch):
    ppo = _make_stub_ppo()
    ppo.dagger_enabled = False
    ppo.fixed_bc_eval_num_samples = 0
    ppo.is_multi_gpu = True
    monkeypatch.setattr(
        ppo,
        "_setup_gloo_barrier_group",
        MethodType(
            lambda self: pytest.fail("non-DAgger fixed-BC save must not create a collective"),
            ppo,
        ),
    )

    assert ppo._collect_distributed_fixed_bc_eval_states() == {}


def test_fixed_bc_checkpoint_remote_validation_failure_fails_every_rank(monkeypatch):
    ppo = _make_stub_ppo()
    ppo.is_multi_gpu = True
    ppo.gpu_world_size = 2
    ppo.gpu_global_rank = 0
    ppo._fixed_bc_eval_ready = True
    ppo._fixed_bc_eval_size = 1
    ppo._fixed_bc_eval_dataset = {
        "actor_obs_raw": torch.tensor([[1.0]]),
        "teacher_actions": torch.tensor([[2.0, 3.0]]),
    }
    gloo_group = object()
    ppo._setup_gloo_barrier_group = MethodType(lambda self: gloo_group, ppo)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)

    def gather(results, local_result, group):
        assert group is gloo_group
        results[0] = local_result
        results[1] = {
            "rank": 1,
            "error": "ValueError: corrupt remote fixed BC state",
            "state": None,
            "contract": {
                **local_result["contract"],
                "rank": 1,
                "local_target": 1,
            },
        }

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    with pytest.raises(RuntimeError, match=r"rank=1: ValueError: corrupt remote"):
        ppo._collect_distributed_fixed_bc_eval_states()


def test_fixed_bc_checkpoint_disabled_rank_still_gathers_and_rejects_budget_drift(monkeypatch):
    ppo = _make_stub_ppo()
    ppo.fixed_bc_eval_num_samples = 0
    ppo.is_multi_gpu = True
    ppo.gpu_world_size = 2
    ppo.gpu_global_rank = 0
    gloo_group = object()
    ppo._setup_gloo_barrier_group = MethodType(lambda self: gloo_group, ppo)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    gather_calls = []

    def gather(results, local_result, group):
        gather_calls.append(local_result)
        assert local_result["state"] is None
        results[0] = local_result
        results[1] = {
            "rank": 1,
            "error": None,
            "state": {"rank": 1},
            "contract": {
                **local_result["contract"],
                "global_sample_budget": 2,
                "rank": 1,
                "local_target": 1,
            },
        }

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    with pytest.raises(RuntimeError, match="global_sample_budget"):
        ppo._collect_distributed_fixed_bc_eval_states()

    assert len(gather_calls) == 1


def test_fixed_bc_checkpoint_contract_error_is_gathered_before_raise(monkeypatch):
    ppo = _make_stub_ppo()
    ppo.is_multi_gpu = True
    ppo.gpu_world_size = 2
    ppo.gpu_global_rank = 0
    ppo._fixed_bc_eval_runtime_contract = MethodType(
        lambda self: (_ for _ in ()).throw(ValueError("invalid local fixed BC contract")),
        ppo,
    )
    group = object()
    ppo._setup_gloo_barrier_group = MethodType(lambda self: group, ppo)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    gather_calls = []

    def gather(results, local_result, group):
        gather_calls.append(local_result)
        assert local_result["contract"] is None
        results[0] = local_result
        results[1] = {
            "rank": 1,
            "error": None,
            "state": {"rank": 1},
            "contract": {},
        }

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    with pytest.raises(RuntimeError, match="invalid local fixed BC contract"):
        ppo._collect_distributed_fixed_bc_eval_states()

    assert len(gather_calls) == 1


def test_fixed_bc_resume_contract_error_is_gathered_before_raise(monkeypatch):
    ppo = _make_stub_ppo()
    ppo.is_multi_gpu = True
    ppo.gpu_world_size = 2
    ppo.gpu_global_rank = 0
    ppo._fixed_bc_eval_runtime_contract = MethodType(
        lambda self: (_ for _ in ()).throw(ValueError("invalid local resume contract")),
        ppo,
    )
    group = object()
    ppo._setup_gloo_barrier_group = MethodType(lambda self: group, ppo)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    gather_calls = []

    def gather(results, local_result, group):
        gather_calls.append(local_result)
        assert local_result["contract"] is None
        results[0] = local_result
        results[1] = {
            "rank": 1,
            "error": None,
            "action": "noop",
            "message": None,
            "contract": {},
        }

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    with pytest.raises(RuntimeError, match="invalid local resume contract"):
        ppo._prepare_fixed_bc_eval_checkpoint_state_all_ranks({}, next_iteration=0)

    assert len(gather_calls) == 1


def test_fixed_bc_resume_rejects_noop_mixed_with_restore(monkeypatch):
    source_states = {}
    for rank in range(2):
        source = _make_stub_ppo()
        source.is_multi_gpu = True
        source.gpu_world_size = 2
        source.gpu_global_rank = rank
        source._fixed_bc_eval_ready = True
        source._fixed_bc_eval_size = 1
        source._fixed_bc_eval_dataset = {
            "actor_obs_raw": torch.tensor([[float(rank + 1)]]),
            "teacher_actions": torch.tensor([[1.0, 2.0]]),
        }
        source_states[str(rank)] = source._local_fixed_bc_eval_checkpoint_state()

    resumed = _make_stub_ppo()
    resumed.is_multi_gpu = True
    resumed.gpu_world_size = 2
    resumed.gpu_global_rank = 0
    group = object()
    resumed._setup_gloo_barrier_group = MethodType(lambda self: group, resumed)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)

    def gather(results, local_result, group):
        results[0] = local_result
        results[1] = {
            "rank": 1,
            "error": None,
            "action": "noop",
            "message": None,
            "contract": {
                **local_result["contract"],
                "rank": 1,
                "local_target": 1,
            },
        }

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    with pytest.raises(RuntimeError, match="noop cannot mix"):
        resumed._prepare_fixed_bc_eval_checkpoint_state_all_ranks(
            {"fixed_bc_eval_by_rank": source_states},
            next_iteration=7,
        )


def test_fixed_bc_eval_uses_same_clipped_teacher_target_as_training():
    ppo = _make_stub_ppo()
    ppo.clip_teacher_actions = True
    ppo.clip_actions_threshold = 2.0

    ppo._maybe_capture_fixed_bc_eval_samples(
        actor_obs_raw=torch.tensor([[1.0], [2.0]]),
        actor_perception_obs=None,
        teacher_actions=torch.tensor([[5.0, -6.0], [1.0, -1.0]]),
        teacher_bc_mask=torch.tensor([[True], [True]]),
    )

    assert torch.equal(
        ppo._fixed_bc_eval_dataset["teacher_actions"],
        torch.tensor([[2.0, -2.0], [1.0, -1.0]]),
    )


def test_select_teacher_actions_uses_teacher_mean_by_default():
    ppo = object.__new__(PPO)
    ppo.use_multi_teacher = False
    ppo.teacher_use_stochastic_actions = False
    ppo.teacher_actor = _DummyTeacherActor()
    ppo._normalize_teacher_actor_obs = MethodType(lambda self, obs, normalizers=None: obs, ppo)

    teacher_actions, teacher_indices = ppo._select_teacher_actions(
        teacher_obs_raw=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        obs_dict={},
    )

    assert teacher_indices is None
    assert torch.equal(teacher_actions, torch.tensor([[2.0, 3.0], [4.0, 5.0]]))


def test_select_teacher_actions_can_opt_into_stochastic_teacher_samples():
    ppo = object.__new__(PPO)
    ppo.use_multi_teacher = False
    ppo.teacher_use_stochastic_actions = True
    ppo.teacher_actor = _DummyTeacherActor()
    ppo._normalize_teacher_actor_obs = MethodType(lambda self, obs, normalizers=None: obs, ppo)

    teacher_actions, _ = ppo._select_teacher_actions(
        teacher_obs_raw=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        obs_dict={},
    )

    assert torch.equal(teacher_actions, torch.tensor([[11.0, 12.0], [13.0, 14.0]]))
