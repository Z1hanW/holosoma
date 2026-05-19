from __future__ import annotations

import pytest
import torch

from holosoma.agents.ppo.ppo import PPO


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
