from __future__ import annotations

from types import SimpleNamespace

import torch

from holosoma.envs.base_task.base_task import BaseTask
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


class _ResetRecorder:
    def __init__(self, events: list[object], name: str) -> None:
        self.events = events
        self.name = name

    def reset(self, _env_ids: torch.Tensor) -> None:
        self.events.append(self.name)


class _SceneResetRecorder:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def reset_scene(self, _env_ids: torch.Tensor) -> None:
        self.events.append("reset_scene")


def _make_reset_task(simulator: object, events: list[object]) -> BaseTask:
    task = object.__new__(BaseTask)
    task.simulator = simulator
    task.observation_manager = _ResetRecorder(events, "observation_reset")
    task.perception_manager = None
    task.teacher_perception_manager = None
    task.critic_perception_manager = None
    task.episode_length_buf = torch.zeros(8, dtype=torch.long)
    task._pending_episode_lengths = torch.zeros(8, dtype=torch.long)
    task._pending_episode_update_mask = torch.zeros(8, dtype=torch.bool)
    task.randomization_manager = None
    task.action_manager = None
    task.command_manager = None
    task.curriculum_manager = None
    task.termination_manager = None
    task.reset_manager = _SceneResetRecorder(events)
    task._reset_refresh_pending = False
    task._finalize_depth_logging_if_needed = lambda: events.append("finalize_depth")
    task._finalize_startup_depth_video_if_needed = (
        lambda _env_ids: events.append("finalize_startup_depth")
    )
    task._reset_envs_idx_impl = (
        lambda _env_ids, _target_states, _target_buf: events.append("reset_impl")
    )
    task._start_depth_logging_if_needed = lambda: events.append("start_depth")
    return task


def _make_base_simulator(*, video_recorder: object | None = None) -> BaseSimulator:
    simulator = object.__new__(BaseSimulator)
    simulator.video_recorder = video_recorder
    simulator.virtual_gantry = None
    return simulator


def test_builtin_noop_episode_callbacks_do_not_materialize_or_iterate_ids(monkeypatch) -> None:
    simulator = _make_base_simulator()
    assert not simulator.requires_episode_callbacks()

    events: list[object] = []
    task = _make_reset_task(simulator, events)
    env_ids = torch.tensor([4, 1, 7], dtype=torch.long)

    def unexpected_item(_self):
        raise AssertionError("no-op episode callbacks must not call Tensor.item()")

    def unexpected_tolist(_self):
        raise AssertionError("no-op episode callbacks must not copy reset IDs to the host")

    monkeypatch.setattr(torch.Tensor, "item", unexpected_item)
    monkeypatch.setattr(torch.Tensor, "tolist", unexpected_tolist)

    task.reset_envs_idx(env_ids)

    assert events == [
        "finalize_depth",
        "finalize_startup_depth",
        "observation_reset",
        "reset_impl",
        "reset_scene",
        "start_depth",
    ]
    assert task._reset_refresh_pending is True


def test_active_builtin_callbacks_copy_ids_once_and_preserve_reset_order(monkeypatch) -> None:
    events: list[object] = []

    class _VideoRecorder:
        def on_episode_end(self, env_id: int) -> None:
            events.append(("episode_end", env_id))

        def on_episode_start(self, env_id: int) -> None:
            events.append(("episode_start", env_id))

    simulator = _make_base_simulator(video_recorder=_VideoRecorder())
    assert simulator.requires_episode_callbacks()
    task = _make_reset_task(simulator, events)
    env_ids = torch.tensor([4, 1, 7], dtype=torch.long)

    original_tolist = torch.Tensor.tolist
    tolist_calls = 0

    def counted_tolist(self):
        nonlocal tolist_calls
        tolist_calls += 1
        return original_tolist(self)

    def unexpected_item(_self):
        raise AssertionError("batched callback dispatch must not call Tensor.item()")

    monkeypatch.setattr(torch.Tensor, "tolist", counted_tolist)
    monkeypatch.setattr(torch.Tensor, "item", unexpected_item)

    task.reset_envs_idx(env_ids)

    assert tolist_calls == 1
    assert events == [
        ("episode_end", 4),
        ("episode_end", 1),
        ("episode_end", 7),
        "finalize_depth",
        "finalize_startup_depth",
        "observation_reset",
        "reset_impl",
        "reset_scene",
        ("episode_start", 4),
        ("episode_start", 1),
        ("episode_start", 7),
        "start_depth",
    ]
    assert all(type(event[1]) is int for event in events if isinstance(event, tuple))


def test_legacy_third_party_scalar_callbacks_remain_supported() -> None:
    events: list[object] = []

    class _LegacySimulator:
        def on_episode_end(self, env_id: int) -> None:
            events.append(("legacy_end", env_id))

        def on_episode_start(self, env_id: int) -> None:
            events.append(("legacy_start", env_id))

    task = object.__new__(BaseTask)
    task.simulator = _LegacySimulator()
    callback_ids = task._simulator_episode_callback_ids(torch.tensor([5, 2]))

    task._notify_simulator_episode_callbacks("end", callback_ids)
    task._notify_simulator_episode_callbacks("start", callback_ids)

    assert callback_ids == [5, 2]
    assert events == [
        ("legacy_end", 5),
        ("legacy_end", 2),
        ("legacy_start", 5),
        ("legacy_start", 2),
    ]


def test_explicit_third_party_batch_callbacks_take_precedence() -> None:
    events: list[object] = []
    simulator = SimpleNamespace(
        requires_episode_callbacks=True,
        on_episodes_end=lambda env_ids: events.append(("batch_end", tuple(env_ids))),
        on_episodes_start=lambda env_ids: events.append(("batch_start", tuple(env_ids))),
        on_episode_end=lambda _env_id: events.append("unexpected_scalar_end"),
        on_episode_start=lambda _env_id: events.append("unexpected_scalar_start"),
    )
    task = object.__new__(BaseTask)
    task.simulator = simulator
    callback_ids = task._simulator_episode_callback_ids(torch.tensor([6, 3]))

    task._notify_simulator_episode_callbacks("end", callback_ids)
    task._notify_simulator_episode_callbacks("start", callback_ids)

    assert events == [("batch_end", (6, 3)), ("batch_start", (6, 3))]
