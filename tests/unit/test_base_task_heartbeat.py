from __future__ import annotations

from types import SimpleNamespace

import torch

import holosoma.envs.base_task.base_task as base_task_module
from holosoma.envs.base_task.base_task import BaseTask


class _ScalarResult:
    def __init__(self, value: int) -> None:
        self.value = value
        self.item_calls = 0

    def item(self) -> int:
        self.item_calls += 1
        return self.value


class _ResetBuffer:
    def __init__(self, reset_count: int) -> None:
        self.sum_calls = 0
        self.sum_result = _ScalarResult(reset_count)

    def sum(self) -> _ScalarResult:
        self.sum_calls += 1
        return self.sum_result


class _LoggerRecorder:
    def __init__(self) -> None:
        self.info_calls: list[tuple[object, ...]] = []

    def info(self, *args: object) -> None:
        self.info_calls.append(args)


def _make_task() -> tuple[BaseTask, _ResetBuffer, list[str]]:
    task = object.__new__(BaseTask)
    task.num_envs = 2
    task.step_timing = SimpleNamespace(enabled=False)
    task.reset_buf = reset_buf = _ResetBuffer(reset_count=1)
    task.obs_buf_dict = {"actor_obs": torch.zeros(2, 3)}
    task.rew_buf = torch.zeros(2)
    task.extras = {}

    phase_calls: list[str] = []
    task._pre_physics_step = lambda actions: phase_calls.append("pre")
    task._physics_step = lambda: phase_calls.append("physics")
    task._post_physics_step = lambda: phase_calls.append("post")
    return task, reset_buf, phase_calls


def test_coarse_heartbeat_does_not_enable_per_step_logging_or_reset_sync(monkeypatch) -> None:
    monkeypatch.setenv("HOLOSOMA_DEBUG_HEARTBEAT", "1")
    monkeypatch.setenv("HOLOSOMA_DEBUG_HEARTBEAT_VERBOSE", "0")
    logger = _LoggerRecorder()
    monkeypatch.setattr(base_task_module, "logger", logger)
    task, reset_buf, phase_calls = _make_task()

    BaseTask._step_impl(task, {"actions": torch.zeros(2, 3)})

    assert phase_calls == ["pre", "physics", "post"]
    assert logger.info_calls == []
    assert reset_buf.sum_calls == 0
    assert reset_buf.sum_result.item_calls == 0


def test_verbose_heartbeat_keeps_per_step_logging_and_reset_count(monkeypatch) -> None:
    monkeypatch.setenv("HOLOSOMA_DEBUG_HEARTBEAT", "1")
    monkeypatch.setenv("HOLOSOMA_DEBUG_HEARTBEAT_VERBOSE", "1")
    logger = _LoggerRecorder()
    monkeypatch.setattr(base_task_module, "logger", logger)
    task, reset_buf, phase_calls = _make_task()

    BaseTask._step_impl(task, {"actions": torch.zeros(2, 3)})

    assert phase_calls == ["pre", "physics", "post"]
    assert len(logger.info_calls) == 4
    assert "BaseTask.step begin" in str(logger.info_calls[0][0])
    assert "reset_envs={}" in str(logger.info_calls[-1][0])
    assert logger.info_calls[-1][1] == 1
    assert reset_buf.sum_calls == 1
    assert reset_buf.sum_result.item_calls == 1
