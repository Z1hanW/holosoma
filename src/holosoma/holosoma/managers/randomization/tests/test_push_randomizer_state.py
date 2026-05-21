from types import SimpleNamespace

import torch

from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg
from holosoma.managers.randomization.manager import RandomizationManager
from holosoma.managers.randomization.terms.locomotion import PushRandomizerState


class _DummySimulator:
    pass


class _DummyEnv(SimpleNamespace):
    def _push_robots(self, env_ids):
        self.pushed_env_ids.append(env_ids.detach().cpu().tolist())


def test_push_randomizer_state_preserves_fractional_interval_seconds():
    env = SimpleNamespace(device="cpu", num_envs=4)
    cfg = SimpleNamespace(
        params={
            "push_interval_s": [0.5, 0.5],
            "max_push_vel": [0.7, 0.7, 0.25, 0.7, 0.7, 1.0],
            "enabled": True,
        }
    )
    state = PushRandomizerState(cfg, env)

    state.setup()

    assert state.push_interval_s is not None
    assert state.push_robot_counter is not None
    assert torch.allclose(state.push_interval_s, torch.full((env.num_envs,), 0.5))

    state.push_robot_counter[:] = 25
    due_envs = state.due_envs(0.02)

    assert torch.equal(due_envs, torch.arange(env.num_envs))


def test_push_randomizer_state_setup_config_survives_reset_and_step_terms():
    env = _DummyEnv(
        device="cpu",
        num_envs=2,
        dt=0.02,
        is_evaluating=False,
        simulator=_DummySimulator(),
        pushed_env_ids=[],
    )
    cfg = RandomizationManagerCfg(
        setup_terms={
            "push_randomizer_state": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState",
                params={
                    "push_interval_s": [0.5, 0.5],
                    "max_push_vel": [0.7, 0.7, 0.25, 0.7, 0.7, 1.0],
                    "enabled": True,
                },
            )
        },
        reset_terms={
            "push_randomizer_state": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
            ),
            "randomize_push_schedule": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:randomize_push_schedule"
            ),
        },
        step_terms={
            "push_randomizer_state": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
            ),
            "apply_pushes": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:apply_pushes"
            ),
        },
    )
    manager = RandomizationManager(cfg, env, "cpu")
    env.randomization_manager = manager

    manager.setup()
    state = manager.get_state("push_randomizer_state")
    assert isinstance(state, PushRandomizerState)
    assert state.push_interval_range == [0.5, 0.5]

    manager.reset(torch.arange(env.num_envs))
    assert state.push_interval_range == [0.5, 0.5]
    assert state.push_interval_s is not None
    assert torch.allclose(state.push_interval_s, torch.full((env.num_envs,), 0.5))

    for _ in range(24):
        manager.step()
    assert env.pushed_env_ids == []

    manager.step()
    assert env.pushed_env_ids == [[0, 1]]
