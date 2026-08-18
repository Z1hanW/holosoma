from __future__ import annotations

from types import SimpleNamespace

import torch

from holosoma.envs.base_task.base_task import BaseTask
from holosoma.managers.termination.terms.common import timeout_exceeded
from holosoma.managers.termination.terms.wbt import BadTracking


def test_timeout_triggers_at_configured_control_step_horizon() -> None:
    env = SimpleNamespace(
        episode_length_buf=torch.tensor([9, 10, 11], dtype=torch.long),
        max_episode_length=10,
    )

    assert torch.equal(
        timeout_exceeded(env),
        torch.tensor([False, True, True]),
    )


def test_simultaneous_terminal_condition_suppresses_timeout_bootstrap(monkeypatch) -> None:
    # env 0: timeout-only truncation; env 1: terminal-only; env 2: both.
    reset_flags = torch.tensor([False, True, True, False])
    timeout_flags = torch.tensor([True, False, True, False])
    env = object.__new__(BaseTask)
    env.reset_buf = torch.zeros(4, dtype=torch.long)
    env.time_out_buf = torch.zeros(4, dtype=torch.bool)
    env.termination_manager = SimpleNamespace(check=lambda: (reset_flags, timeout_flags))
    monkeypatch.setattr(env, "_log_termination_masks", lambda *_: None)

    env._check_termination()

    assert torch.equal(env.reset_buf.bool(), reset_flags | timeout_flags)
    assert torch.equal(env.time_out_buf, torch.tensor([True, False, False, False]))


def test_final_timeout_observation_uses_regular_observation_clip_domain() -> None:
    env = object.__new__(BaseTask)
    env.observation_manager = SimpleNamespace(
        cfg=SimpleNamespace(clip_observations=1.0),
        compute=lambda **_: {"critic_obs": torch.tensor([[-2.0, 0.5, 3.0]])},
    )

    final_obs = env._compute_final_observations()

    assert torch.equal(final_obs["critic_obs"], torch.tensor([[-1.0, 0.5, 1.0]]))


def test_termination_logs_distinguish_raw_timeout_timeout_only_and_done() -> None:
    env = object.__new__(BaseTask)
    env.log_dict = {}
    env.termination_manager = SimpleNamespace(_term_names=[])
    reset_flags = torch.tensor([False, True, True, False])
    timeout_flags = torch.tensor([True, False, True, False])

    env._log_termination_masks(reset_flags, timeout_flags)

    assert env.log_dict["termination/reset_frac"].item() == 0.5
    assert env.log_dict["termination/timeout_frac"].item() == 0.5
    assert env.log_dict["termination/timeout_only_frac"].item() == 0.25
    assert env.log_dict["termination/done_frac"].item() == 0.75


def test_bad_tracking_exposes_each_overlapping_threshold_condition(monkeypatch) -> None:
    monkeypatch.delenv("HOLOSOMA_DISABLE_BAD_TRACKING_RESET", raising=False)
    motion_command = SimpleNamespace(
        motion_cfg=SimpleNamespace(body_names_to_track=["torso"]),
        motion=SimpleNamespace(has_object=True),
    )
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        is_evaluating=False,
        command_manager=SimpleNamespace(get_state=lambda _: motion_command),
    )
    term = object.__new__(BadTracking)
    term.env = env
    term.body_names_to_track = ["torso"]
    term._last_component_results = {}
    expected_components = {
        "robot_ref_position": torch.tensor([True, False, False, False]),
        "robot_ref_orientation": torch.tensor([False, True, False, False]),
        "robot_body_position": torch.tensor([False, False, True, False]),
        "object_position": torch.tensor([True, False, False, False]),
        "object_orientation": torch.tensor([False, False, False, True]),
    }
    term.bad_ref_pos = lambda _: expected_components["robot_ref_position"]
    term.bad_ref_ori = lambda _: expected_components["robot_ref_orientation"]
    term.bad_motion_body_pos = lambda _: expected_components["robot_body_position"]
    term.bad_object_pos = lambda _: expected_components["object_position"]
    term.bad_object_ori = lambda _: expected_components["object_orientation"]

    result = term(env)

    actual_components = term.get_last_component_results()
    assert actual_components.keys() == expected_components.keys()
    for name, expected in expected_components.items():
        assert torch.equal(actual_components[name], expected)
    assert torch.equal(result, torch.tensor([True, True, True, True]))


def test_termination_logs_bad_tracking_component_fractions() -> None:
    env = object.__new__(BaseTask)
    env.log_dict = {}
    term_result = torch.tensor([True, True, False, False])
    components = {
        "robot_ref_position": torch.tensor([True, False, False, False]),
        "object_position": torch.tensor([True, True, False, False]),
    }
    env.termination_manager = SimpleNamespace(
        _term_names=["bad_tracking"],
        get_last_term_result=lambda _: term_result,
        get_last_term_components=lambda _: components,
    )

    env._log_termination_masks(term_result, torch.zeros_like(term_result))

    assert env.log_dict["termination/bad_tracking_frac"].item() == 0.5
    assert env.log_dict[
        "termination/bad_tracking/condition_robot_ref_position_frac"
    ].item() == 0.25
    assert env.log_dict[
        "termination/bad_tracking/condition_object_position_frac"
    ].item() == 0.5
