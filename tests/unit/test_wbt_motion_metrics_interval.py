from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.curriculum.terms.locomotion import WObjectDifficultyCurriculum


_LIVE_OBJECT_ERROR_KEY = "motion/error_object_ref_pos"


class _ObjectMetricCommand(MotionCommand):
    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._test_object_pos

    @property
    def simulator_object_pos_w(self) -> torch.Tensor:
        return self._test_simulator_object_pos


def _make_object_metric_command() -> _ObjectMetricCommand:
    command = object.__new__(_ObjectMetricCommand)
    command.device = "cpu"
    command.num_envs = 2
    command.motion = SimpleNamespace(has_object=True)
    command.metrics = {"diagnostic/sentinel": torch.tensor([7.0])}
    command._test_object_pos = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=torch.float32,
    )
    command._test_simulator_object_pos = torch.zeros((2, 3), dtype=torch.float32)
    return command


def test_selective_live_metric_refresh_assigns_a_new_tensor() -> None:
    command = _make_object_metric_command()

    command.update_live_metrics({_LIVE_OBJECT_ERROR_KEY})
    first = command.metrics[_LIVE_OBJECT_ERROR_KEY]
    assert torch.equal(first, torch.tensor([1.0, 2.0]))
    assert set(command.metrics) == {"diagnostic/sentinel", _LIVE_OBJECT_ERROR_KEY}

    command._test_object_pos = torch.tensor(
        [[0.0, 0.0, 3.0], [0.0, 4.0, 0.0]],
        dtype=torch.float32,
    )
    command.update_live_metrics({_LIVE_OBJECT_ERROR_KEY})
    second = command.metrics[_LIVE_OBJECT_ERROR_KEY]

    assert torch.equal(first, torch.tensor([1.0, 2.0]))
    assert torch.equal(second, torch.tensor([3.0, 4.0]))
    assert second is not first
    assert second.data_ptr() != first.data_ptr()


@pytest.mark.parametrize(
    "metric_key",
    [
        "motion/error_object_ref_position_typo",
        "motion/adaptive_timesteps_sampler_entropy",
        "lift/root_z",
    ],
)
def test_curriculum_live_metric_validation_fails_closed(metric_key: str) -> None:
    command = SimpleNamespace(
        supported_live_metric_keys=MotionCommand.supported_live_metric_keys,
    )

    with pytest.raises(ValueError, match="not a supported live tracking error"):
        WholeBodyTrackingManager._validate_live_motion_metric_keys(
            command,
            frozenset({metric_key}),
        )


def test_only_enabled_curriculum_similarity_key_enters_live_set() -> None:
    enabled = SimpleNamespace(enabled=True, similarity_metric_key=_LIVE_OBJECT_ERROR_KEY)
    disabled = SimpleNamespace(enabled=False, similarity_metric_key="motion/error_body_pos")
    env = object.__new__(WholeBodyTrackingManager)
    env.curriculum_manager = SimpleNamespace(
        iter_terms=lambda: iter((("enabled", enabled), ("disabled", disabled))),
    )

    assert env._curriculum_live_motion_metric_keys() == frozenset({_LIVE_OBJECT_ERROR_KEY})


def test_misspelled_curriculum_metric_prefix_is_not_silently_ignored() -> None:
    term = SimpleNamespace(enabled=True, similarity_metric_key="motoin/error_object_ref_pos")
    env = object.__new__(WholeBodyTrackingManager)
    env.curriculum_manager = SimpleNamespace(iter_terms=lambda: iter((("wobject", term),)))

    metric_keys = env._curriculum_live_motion_metric_keys()

    assert metric_keys == frozenset({"motoin/error_object_ref_pos"})
    with pytest.raises(ValueError, match="not a supported live tracking error"):
        env._validate_live_motion_metric_keys(
            SimpleNamespace(supported_live_metric_keys=MotionCommand.supported_live_metric_keys),
            metric_keys,
        )


class _CadenceCommand:
    def __init__(self) -> None:
        self.metrics: dict[str, torch.Tensor] = {}
        self.full_calls = 0
        self.live_calls: list[frozenset[str]] = []

    def update_metrics(self) -> None:
        self.full_calls += 1
        self.metrics[_LIVE_OBJECT_ERROR_KEY] = torch.tensor([float(self.full_calls)])
        self.metrics["diagnostic/expensive"] = torch.tensor([float(self.full_calls)])

    def update_live_metrics(self, metric_keys) -> None:
        keys = frozenset(metric_keys)
        self.live_calls.append(keys)
        for key in keys:
            self.metrics[key] = torch.tensor([100.0 + len(self.live_calls)])


def _make_cadence_env(command: _CadenceCommand, *, live: bool) -> WholeBodyTrackingManager:
    env = object.__new__(WholeBodyTrackingManager)
    env._motion_metrics_interval = 4
    env._motion_metrics_step = 0
    env._live_motion_metric_keys = frozenset({_LIVE_OBJECT_ERROR_KEY}) if live else frozenset()
    env.step_timing = SimpleNamespace(enabled=False)
    env.log_dict = {}
    env.command_manager = SimpleNamespace(get_state=lambda name: command)
    tracker = SimpleNamespace(get_average=lambda: torch.tensor(5.0))
    env.curriculum_manager = SimpleNamespace(
        get_term=lambda name: tracker if name == "average_episode_tracker" else None,
    )
    return env


def test_full_diagnostics_follow_interval_while_live_metric_stays_fresh() -> None:
    command = _CadenceCommand()
    env = _make_cadence_env(command, live=True)

    for _ in range(6):
        env._update_log_dict()

    assert command.full_calls == 2  # steps 1 and 5
    assert command.live_calls == [
        frozenset({_LIVE_OBJECT_ERROR_KEY}),
        frozenset({_LIVE_OBJECT_ERROR_KEY}),
        frozenset({_LIVE_OBJECT_ERROR_KEY}),
        frozenset({_LIVE_OBJECT_ERROR_KEY}),
    ]
    assert "diagnostic/expensive" in env.log_dict


def test_disabled_curriculum_does_not_force_selective_refresh() -> None:
    command = _CadenceCommand()
    env = _make_cadence_env(command, live=False)

    for _ in range(6):
        env._update_log_dict()

    assert command.full_calls == 2
    assert command.live_calls == []


class _PreResetMetricCommand:
    def __init__(self) -> None:
        self.metrics = {
            _LIVE_OBJECT_ERROR_KEY: torch.tensor([9.0, 9.0]),
            "diagnostic/expensive": torch.tensor([3.0]),
        }
        self.next_error = torch.tensor([0.25, 0.75])
        self.full_calls = 0

    def update_metrics(self) -> None:
        self.full_calls += 1

    def update_live_metrics(self, metric_keys) -> None:
        assert frozenset(metric_keys) == frozenset({_LIVE_OBJECT_ERROR_KEY})
        self.metrics[_LIVE_OBJECT_ERROR_KEY] = self.next_error.clone()


def test_non_diagnostic_step_publishes_fresh_pre_reset_curriculum_error() -> None:
    command = _PreResetMetricCommand()
    stale_error = command.metrics[_LIVE_OBJECT_ERROR_KEY]
    env = object.__new__(WholeBodyTrackingManager)
    env.device = "cpu"
    env.num_envs = 2
    env.is_evaluating = False
    env.time_out_buf = torch.tensor([False, False])
    env.termination_manager = SimpleNamespace(
        get_last_term_result=lambda name: torch.tensor([False, False]),
    )
    env.command_manager = SimpleNamespace(get_state=lambda name: command)
    env._motion_metrics_interval = 4
    env._motion_metrics_step = 1  # step 2 is deliberately not a full refresh
    env._live_motion_metric_keys = frozenset({_LIVE_OBJECT_ERROR_KEY})
    env.step_timing = SimpleNamespace(enabled=False)
    env.log_dict = {}

    term = WObjectDifficultyCurriculum(
        SimpleNamespace(
            params={
                "enabled": True,
                "initial_lambda": 0.0,
                "similarity_metric_key": _LIVE_OBJECT_ERROR_KEY,
            }
        ),
        env,
    )
    term.setup()
    tracker = SimpleNamespace(get_average=lambda: torch.tensor(1.0))
    env.curriculum_manager = SimpleNamespace(
        get_term=lambda name: tracker if name == "average_episode_tracker" else term,
    )

    env._update_log_dict()
    term.reset(torch.tensor([1], dtype=torch.long))

    fresh_error = env.log_dict[_LIVE_OBJECT_ERROR_KEY]
    assert command.full_calls == 0
    assert fresh_error is not stale_error
    assert torch.equal(stale_error, torch.tensor([9.0, 9.0]))
    assert torch.equal(fresh_error, torch.tensor([0.25, 0.75]))
    assert term._pending_similarity_error_sum == pytest.approx(0.75)
    assert term._pending_similarity_count == 1
