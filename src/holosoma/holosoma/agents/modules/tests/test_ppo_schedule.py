from __future__ import annotations

import pytest

from holosoma.agents.ppo.ppo import PPO


def _make_stub_ppo(*, ppo_start_epoch: int, dagger_end_epoch: int, ppo_target_coeff: float, step_epochs: int) -> PPO:
    ppo = object.__new__(PPO)
    ppo.use_ppo_dagger_schedule = True
    ppo.ppo_start_epoch = ppo_start_epoch
    ppo.dagger_end_epoch = dagger_end_epoch
    ppo.ppo_target_coeff = ppo_target_coeff
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
