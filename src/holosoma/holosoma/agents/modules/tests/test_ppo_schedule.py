from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from holosoma.agents.ppo.ppo import PPO
from holosoma.utils.rng_checkpoint import capture_rng_checkpoint_state


def _make_stub_ppo(
    *,
    ppo_start_epoch: int,
    dagger_end_epoch: int,
    ppo_target_coeff: float,
    step_epochs: int,
    ppo_start_coeff: float = 0.0,
) -> PPO:
    ppo = object.__new__(PPO)
    ppo.use_ppo_dagger_schedule = True
    ppo.ppo_start_epoch = ppo_start_epoch
    ppo.dagger_end_epoch = dagger_end_epoch
    ppo.ppo_target_coeff = ppo_target_coeff
    ppo.ppo_start_coeff = ppo_start_coeff
    ppo.ppo_schedule_step_epochs = step_epochs
    ppo.ppo_coeff = 0.0
    return ppo


def _make_entropy_schedule(
    *,
    start: float = 0.005,
    end: float | None = 0.0,
    start_iteration: int = 2000,
    end_iteration: int = 10000,
) -> PPO:
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        entropy_coef=start,
        entropy_coef_end=end,
        entropy_coef_decay_start_iteration=start_iteration,
        entropy_coef_decay_end_iteration=end_iteration,
    )
    (
        ppo._entropy_coef_start,
        ppo._entropy_coef_end,
        ppo._entropy_coef_decay_start_iteration,
        ppo._entropy_coef_decay_end_iteration,
    ) = ppo._validate_entropy_coefficient_schedule()
    ppo.current_learning_iteration = 0
    return ppo


def test_entropy_coefficient_holds_then_decays_to_zero_by_absolute_iteration():
    ppo = _make_entropy_schedule()

    expected = {
        0: 0.005,
        1999: 0.005,
        2000: 0.005,
        6000: 0.0025,
        9999: 0.005 / 8000,
        10000: 0.0,
        60000: 0.0,
    }
    for iteration, coefficient in expected.items():
        assert ppo._operational_entropy_coefficient(iteration) == pytest.approx(coefficient)


def test_entropy_coefficient_fixed_mode_preserves_legacy_behavior():
    ppo = _make_entropy_schedule(end=None)

    assert ppo._operational_entropy_coefficient(0) == pytest.approx(0.005)
    assert ppo._operational_entropy_coefficient(60000) == pytest.approx(0.005)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("entropy_coef", -0.1, "finite non-negative real"),
        ("entropy_coef_end", float("nan"), "finite non-negative real"),
        ("entropy_coef_decay_start_iteration", True, "non-negative integer"),
        ("entropy_coef_decay_end_iteration", 2000, "requires.*end_iteration"),
    ],
)
def test_entropy_coefficient_schedule_rejects_invalid_contract(field, value, match):
    ppo = _make_entropy_schedule()
    setattr(ppo.config, field, value)

    with pytest.raises(ValueError, match=match):
        ppo._validate_entropy_coefficient_schedule()


def test_adjust_ppo_dagger_coeff_uses_linear_ramp_by_default():
    ppo = _make_stub_ppo(ppo_start_epoch=0, dagger_end_epoch=3000, ppo_target_coeff=0.3, step_epochs=0)

    ppo._adjust_ppo_dagger_coeff(1500)

    assert ppo.ppo_coeff == pytest.approx(0.15)


def test_adjust_ppo_dagger_coeff_supports_step_schedule():
    ppo = _make_stub_ppo(ppo_start_epoch=0, dagger_end_epoch=4500, ppo_target_coeff=0.9, step_epochs=500)

    checkpoints = {
        0: 0.0,
        499: 0.0,
        500: 0.1,
        1000: 0.2,
        2500: 0.5,
        4000: 0.8,
        4500: 0.9,
        4700: 0.9,
    }

    for current_epoch, expected in checkpoints.items():
        ppo._adjust_ppo_dagger_coeff(current_epoch)
        assert ppo.ppo_coeff == pytest.approx(expected)


def test_adjust_ppo_dagger_coeff_supports_nonzero_start_coeff():
    ppo = _make_stub_ppo(
        ppo_start_epoch=0,
        dagger_end_epoch=4000,
        ppo_target_coeff=0.9,
        step_epochs=500,
        ppo_start_coeff=0.1,
    )

    checkpoints = {
        0: 0.1,
        499: 0.1,
        500: 0.2,
        1000: 0.3,
        2500: 0.6,
        3500: 0.8,
        4000: 0.9,
        4500: 0.9,
    }

    for current_epoch, expected in checkpoints.items():
        ppo._adjust_ppo_dagger_coeff(current_epoch)
        assert ppo.ppo_coeff == pytest.approx(expected)


def test_ppo_start_noise_std_cap_applies_only_through_first_ppo_tier():
    ppo = _make_stub_ppo(ppo_start_epoch=2000, dagger_end_epoch=6500, ppo_target_coeff=0.9, step_epochs=500)
    ppo.ppo_start_noise_std = 0.1
    ppo.ppo_start_noise_std_until_coeff = 0.1
    ppo._ppo_start_noise_std_cap_announced = True
    ppo.config = type("Config", (), {"init_noise_std": 0.8})()
    ppo.actor = type("Actor", (), {})()
    ppo.actor.std = torch.nn.Parameter(torch.tensor([0.5, 0.2, 0.05]))

    ppo._apply_ppo_start_noise_std_cap(2500)

    assert ppo.actor.std.detach().tolist() == pytest.approx([0.1, 0.1, 0.05])

    ppo.actor.std.data.copy_(torch.tensor([0.5, 0.2, 0.05]))

    ppo._apply_ppo_start_noise_std_cap(3000)

    assert ppo.actor.std.detach().tolist() == pytest.approx([0.5, 0.2, 0.05])


def test_ppo_start_noise_std_cap_covers_first_step_tier_above_until_coeff():
    ppo = _make_stub_ppo(ppo_start_epoch=1000, dagger_end_epoch=4500, ppo_target_coeff=0.9, step_epochs=500)
    ppo.ppo_start_noise_std = 0.1
    ppo.ppo_start_noise_std_until_coeff = 0.1
    ppo._ppo_start_noise_std_cap_announced = True
    ppo.config = type("Config", (), {"init_noise_std": 0.8})()
    ppo.actor = type("Actor", (), {})()
    ppo.actor.std = torch.nn.Parameter(torch.tensor([0.5]))

    ppo._apply_ppo_start_noise_std_cap(1500)

    assert ppo.actor.std.detach().item() == pytest.approx(0.1)

    ppo.actor.std.data.fill_(0.5)

    ppo._apply_ppo_start_noise_std_cap(2000)

    assert ppo.actor.std.detach().item() == pytest.approx(0.5)


def test_ppo_start_noise_cap_preserves_mean_floor_without_exceeding_cap():
    ppo = _make_stub_ppo(
        ppo_start_epoch=0,
        dagger_end_epoch=10,
        ppo_target_coeff=0.9,
        step_epochs=1,
    )
    ppo.ppo_start_noise_std = 0.8
    ppo.ppo_start_noise_std_until_coeff = 0.1
    ppo._ppo_start_noise_std_cap_announced = True
    ppo.actor = type("Actor", (), {})()
    ppo.actor.std = torch.nn.Parameter(torch.tensor([1.0, 0.1]))
    ppo.actor.min_noise_std = None
    ppo.actor.min_mean_noise_std = 0.8
    ppo.actor.max_noise_std = 1.0

    ppo._apply_ppo_start_noise_std_cap(0)

    assert ppo.actor.std.detach().mean().item() == pytest.approx(0.8)
    assert ppo.actor.std.detach().max().item() == pytest.approx(0.8)


def test_ppo_start_noise_cap_rejects_incompatible_mean_floor():
    ppo = _make_stub_ppo(
        ppo_start_epoch=0,
        dagger_end_epoch=10,
        ppo_target_coeff=0.9,
        step_epochs=1,
    )
    ppo.ppo_start_noise_std = 0.1
    ppo.ppo_start_noise_std_until_coeff = 0.1
    ppo._ppo_start_noise_std_cap_announced = True
    ppo.actor = type("Actor", (), {})()
    ppo.actor.std = torch.nn.Parameter(torch.tensor([0.5]))
    ppo.actor.min_noise_std = None
    ppo.actor.min_mean_noise_std = 0.2
    ppo.actor.max_noise_std = 1.0

    with pytest.raises(ValueError, match="cannot satisfy.*policy-noise constraints"):
        ppo._apply_ppo_start_noise_std_cap(0)


def test_refresh_distillation_state_sets_coeff_before_action_selection():
    ppo = _make_stub_ppo(
        ppo_start_epoch=0,
        dagger_end_epoch=4000,
        ppo_target_coeff=0.9,
        step_epochs=500,
        ppo_start_coeff=0.1,
    )
    ppo.dagger_enabled = True
    ppo.bc_loss_coef = 1.0
    ppo._configured_bc_loss_coef = 1.0
    ppo.switch_to_rl_after = -1

    ppo._refresh_distillation_iteration_state(0)

    assert ppo.ppo_coeff == pytest.approx(0.1)
    assert ppo._use_deterministic_student_actions() is False


def test_refresh_distillation_state_applies_switch_on_or_after_resume_iteration():
    ppo = _make_stub_ppo(ppo_start_epoch=-1, dagger_end_epoch=-1, ppo_target_coeff=0.9, step_epochs=0)
    ppo.use_ppo_dagger_schedule = False
    ppo.dagger_enabled = True
    ppo._configured_bc_loss_coef = 1.0
    ppo.bc_loss_coef = 1.0
    ppo.switch_to_rl_after = 100

    ppo._refresh_distillation_iteration_state(150)

    assert ppo.bc_loss_coef == 0.0


def test_switch_to_rl_does_not_fall_back_to_legacy_teacher_mse():
    ppo = object.__new__(PPO)
    ppo.distill_enabled = True
    ppo.distill_mode = "dagger"
    ppo.dagger_enabled = True
    ppo.use_ppo_dagger_schedule = False
    ppo._configured_bc_loss_coef = 1.0
    ppo.bc_loss_coef = 1.0
    ppo.switch_to_rl_after = 100

    ppo._refresh_distillation_iteration_state(100)

    assert ppo.bc_loss_coef == 0.0
    assert ppo._legacy_distillation_enabled() is False


def test_legacy_teacher_mse_mode_remains_enabled():
    ppo = object.__new__(PPO)
    ppo.distill_enabled = True
    ppo.distill_mode = "mse"

    assert ppo._legacy_distillation_enabled() is True


def test_load_derives_scheduled_coeff_from_restored_iteration(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_ALLOW_LEGACY_UNPROVENANCED_RESUME", "1")
    ppo = _make_stub_ppo(
        ppo_start_epoch=0,
        dagger_end_epoch=3000,
        ppo_target_coeff=0.9,
        step_epochs=0,
    )
    ppo.dagger_enabled = True
    ppo._configured_bc_loss_coef = 1.0
    ppo.bc_loss_coef = 1.0
    ppo.switch_to_rl_after = -1
    ppo.ppo_start_noise_std = None
    ppo.actor = torch.nn.Linear(1, 1)
    ppo.critic = torch.nn.Linear(1, 1)
    ppo.actor_optimizer = torch.optim.SGD(ppo.actor.parameters(), lr=0.1)
    ppo.critic_optimizer = torch.optim.SGD(ppo.critic.parameters(), lr=0.1)
    ppo.actor_obs_normalizers = {}
    ppo.critic_obs_normalizers = {}
    ppo.config = SimpleNamespace(
        load_optimizer=True,
        normalize_actor_obs=False,
        normalize_critic_obs=False,
    )
    ppo.actor_learning_rate = 0.1
    ppo.min_actor_learning_rate = 1.0e-5
    ppo.max_actor_learning_rate = 0.1
    ppo.critic_learning_rate = 0.1
    ppo.min_critic_learning_rate = 1.0e-5
    ppo.max_critic_learning_rate = 0.1
    ppo.device = "cpu"
    ppo.gpu_global_rank = 0
    ppo.env = SimpleNamespace(load_checkpoint_state=lambda state: None)
    checkpoint = {
        "actor_model_state_dict": ppo.actor.state_dict(),
        "critic_model_state_dict": ppo.critic.state_dict(),
        "actor_optimizer_state_dict": ppo.actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": ppo.critic_optimizer.state_dict(),
        "iter": 1500,
        "rng_state_by_rank": {"0": capture_rng_checkpoint_state()},
        "rollout_resume_contract": ppo._rollout_resume_contract(1501),
    }

    with patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint", return_value=(checkpoint, "0" * 64)):
        ppo.load("resume.pt")

    # Legacy ``iter`` is the completed update, so resume starts at iter + 1
    # and schedule state is derived for that next rollout.
    assert ppo.current_learning_iteration == 1501
    assert ppo.ppo_coeff == pytest.approx(0.4503)


def test_legacy_resume_range_does_not_repeat_completed_iteration():
    completed_iteration = 31500
    next_iteration = completed_iteration + 1
    target_iteration = 40000

    updates = range(next_iteration, target_iteration)

    assert updates.start == 31501
    assert updates.stop == 40000
    assert updates[-1] == 39999
    assert len(updates) == 8499


def _teacher_rollout_validation_stub(*, pure_bc: bool) -> PPO:
    ppo = object.__new__(PPO)
    ppo.distill_mode = "dagger"
    ppo.take_teacher_actions = False
    ppo.teacher_action_mix_ratio = 0.5
    ppo._configured_teacher_action_mix_ratio = 0.5
    ppo.teacher_action_mix_ratio_start = None
    ppo.teacher_action_mix_ratio_end = None
    ppo.use_teacher_action_mix_schedule = False
    ppo.use_ppo_dagger_schedule = False
    ppo.ppo_start_coeff = 0.0
    # Schedule coefficients are inactive in this helper and must remain at
    # their neutral configuration defaults.
    ppo.ppo_target_coeff = 0.9
    ppo._configured_bc_loss_coef = 1.0 if pure_bc else 0.5
    ppo.switch_to_rl_after = -1
    return ppo


def test_teacher_rollout_actions_are_allowed_for_pure_dagger():
    _teacher_rollout_validation_stub(pure_bc=True)._validate_teacher_rollout_action_config()


def test_teacher_rollout_actions_fail_fast_when_ppo_contributes():
    with pytest.raises(ValueError, match="cannot be combined with a non-zero PPO"):
        _teacher_rollout_validation_stub(pure_bc=False)._validate_teacher_rollout_action_config()


def test_teacher_rollout_actions_reject_even_tiny_positive_ppo_weight():
    ppo = _teacher_rollout_validation_stub(pure_bc=True)
    ppo._configured_bc_loss_coef = 0.999999995

    with pytest.raises(ValueError, match="cannot be combined with a non-zero PPO"):
        ppo._validate_teacher_rollout_action_config()


def test_take_teacher_actions_fail_fast_for_scheduled_ppo():
    ppo = _teacher_rollout_validation_stub(pure_bc=True)
    ppo.teacher_action_mix_ratio = 0.0
    ppo._configured_teacher_action_mix_ratio = 0.0
    ppo.take_teacher_actions = True
    ppo.use_ppo_dagger_schedule = True
    ppo.ppo_target_coeff = 0.9

    with pytest.raises(ValueError, match="cannot be combined with a non-zero PPO"):
        ppo._validate_teacher_rollout_action_config()


def test_take_teacher_actions_rejects_nonzero_mix_in_pure_bc():
    ppo = _teacher_rollout_validation_stub(pure_bc=True)
    ppo.take_teacher_actions = True

    with pytest.raises(ValueError, match="mutually exclusive.*mix ratio"):
        ppo._validate_teacher_rollout_action_config()


def test_teacher_mix_schedule_rejects_ignored_static_ratio():
    ppo = _teacher_rollout_validation_stub(pure_bc=True)
    ppo.use_teacher_action_mix_schedule = True
    ppo.teacher_action_mix_ratio = 0.8
    ppo.teacher_action_mix_ratio_start = 0.8
    ppo.teacher_action_mix_ratio_end = 0.0

    with pytest.raises(ValueError, match="static value would be silently ignored"):
        ppo._validate_teacher_rollout_action_config()


def test_legacy_mse_rejects_ignored_teacher_rollout_options():
    ppo = _teacher_rollout_validation_stub(pure_bc=True)
    ppo.distill_mode = "mse"
    ppo.teacher_use_stochastic_actions = False

    with pytest.raises(ValueError, match="does not implement.*teacher_action_mix_ratio"):
        ppo._validate_teacher_rollout_action_config()
