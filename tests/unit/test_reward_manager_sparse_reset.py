from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.envs.base_task.base_task import BaseTask
from holosoma.managers.reward.manager import RewardManager


_SCALED_SUMS = {
    "alpha": torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0]),
    "beta": torch.tensor([-2.0, 1.0, 0.0, 3.0, 7.0]),
}
_RAW_SUMS = {
    "alpha": torch.tensor([20.0, 40.0, 60.0, 80.0, 100.0]),
    "beta": torch.tensor([-20.0, 10.0, 0.0, 30.0, 70.0]),
}
_MAX_EPISODE_LENGTH_S = 2.0
_DT = 0.1
_EPISODE_STEPS = torch.tensor([2, 4, 0, 8, 5], dtype=torch.long)


class _StatefulResetRecorder:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor | None] = []

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.calls.append(None if env_ids is None else env_ids.detach().clone())


def _make_reward_manager() -> tuple[RewardManager, _StatefulResetRecorder]:
    stateful = _StatefulResetRecorder()
    manager = object.__new__(RewardManager)
    manager.env = SimpleNamespace(
        max_episode_length_s=_MAX_EPISODE_LENGTH_S,
        dt=_DT,
        _pending_episode_lengths=_EPISODE_STEPS.clone(),
    )
    manager.device = "cpu"
    manager._term_names = list(_SCALED_SUMS)
    manager._episode_sums = {name: values.clone() for name, values in _SCALED_SUMS.items()}
    manager._episode_sums_raw = {name: values.clone() for name, values in _RAW_SUMS.items()}
    manager._term_instances = {"stateful": stateful}
    return manager, stateful


def _expected_extras(
    env_ids: torch.Tensor | None,
    *,
    include_all: bool,
) -> dict[str, dict[str, torch.Tensor]]:
    selected = slice(None) if env_ids is None else env_ids
    extras: dict[str, dict[str, torch.Tensor]] = {
        "episode": {},
        "episode_all": {},
        "raw_episode": {},
        "raw_episode_all": {},
        "episode_rate": {},
        "raw_episode_mean": {},
    }
    selected_steps = _EPISODE_STEPS[selected]

    def normalize_by_steps(values: torch.Tensor, *, seconds: bool) -> torch.Tensor:
        denominator = selected_steps.to(dtype=values.dtype)
        if seconds:
            denominator = denominator * _DT
        valid = denominator > 0
        safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
        return (values[selected] / safe_denominator).masked_fill(~valid, 0.0)

    for term_name, values in _SCALED_SUMS.items():
        normalized = values / _MAX_EPISODE_LENGTH_S
        extras["episode"][f"rew_{term_name}"] = normalized[selected].clone()
        extras["episode_rate"][f"rew_{term_name}"] = normalize_by_steps(values, seconds=True)
        if include_all:
            extras["episode_all"][f"rew_{term_name}"] = normalized.clone()
    for term_name, values in _RAW_SUMS.items():
        normalized = values / _MAX_EPISODE_LENGTH_S
        extras["raw_episode"][f"raw_rew_{term_name}"] = normalized[selected].clone()
        extras["raw_episode_mean"][f"raw_rew_{term_name}"] = normalize_by_steps(
            values,
            seconds=False,
        )
        if include_all:
            extras["raw_episode_all"][f"raw_rew_{term_name}"] = normalized.clone()
    return extras


def _expected_remaining(
    source: dict[str, torch.Tensor],
    env_ids: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    selected = slice(None) if env_ids is None else env_ids
    remaining = {name: values.clone() for name, values in source.items()}
    for values in remaining.values():
        values[selected] = 0.0
    return remaining


def _assert_tensor_dicts_equal(
    actual: dict[str, dict[str, torch.Tensor]],
    expected: dict[str, dict[str, torch.Tensor]],
) -> None:
    assert actual.keys() == expected.keys()
    for section in expected:
        assert actual[section].keys() == expected[section].keys()
        for key, expected_tensor in expected[section].items():
            assert torch.equal(actual[section][key], expected_tensor), (section, key)


def _assert_episode_buffers_equal(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    assert actual.keys() == expected.keys()
    for key, expected_tensor in expected.items():
        assert torch.equal(actual[key], expected_tensor), key


def _assert_base_task_extras(
    task: BaseTask,
    expected_reward_extras: dict[str, dict[str, torch.Tensor]],
) -> None:
    actual_reward_extras = {
        section: task.extras[section]
        for section in (
            "episode",
            "episode_all",
            "raw_episode",
            "raw_episode_all",
            "episode_rate",
            "raw_episode_mean",
        )
    }
    _assert_tensor_dicts_equal(actual_reward_extras, expected_reward_extras)
    assert task.extras["time_outs"] is task.time_out_buf


@pytest.mark.parametrize(
    "env_ids",
    [None, torch.tensor([3, 1], dtype=torch.long), torch.empty(0, dtype=torch.long)],
    ids=["all", "subset", "empty"],
)
def test_dense_reset_matches_legacy_oracle_exactly(env_ids: torch.Tensor | None) -> None:
    manager, stateful = _make_reward_manager()

    actual = manager.reset(env_ids)

    _assert_tensor_dicts_equal(actual, _expected_extras(env_ids, include_all=True))
    _assert_episode_buffers_equal(
        manager._episode_sums,
        _expected_remaining(_SCALED_SUMS, env_ids),
    )
    _assert_episode_buffers_equal(
        manager._episode_sums_raw,
        _expected_remaining(_RAW_SUMS, env_ids),
    )
    assert len(stateful.calls) == 1
    if env_ids is None:
        assert stateful.calls[0] is None
    else:
        assert torch.equal(stateful.calls[0], env_ids)


@pytest.mark.parametrize(
    "env_ids",
    [None, torch.tensor([3, 1], dtype=torch.long), torch.empty(0, dtype=torch.long)],
    ids=["all", "subset", "empty"],
)
def test_sparse_reset_materializes_only_completed_rows(env_ids: torch.Tensor | None) -> None:
    manager, stateful = _make_reward_manager()

    actual = manager.reset(env_ids, include_all=False)

    _assert_tensor_dicts_equal(actual, _expected_extras(env_ids, include_all=False))
    assert actual["episode_all"] == {}
    assert actual["raw_episode_all"] == {}
    _assert_episode_buffers_equal(
        manager._episode_sums,
        _expected_remaining(_SCALED_SUMS, env_ids),
    )
    _assert_episode_buffers_equal(
        manager._episode_sums_raw,
        _expected_remaining(_RAW_SUMS, env_ids),
    )
    assert len(stateful.calls) == 1
    if env_ids is None:
        assert stateful.calls[0] is None
    else:
        assert torch.equal(stateful.calls[0], env_ids)


def test_sparse_subset_never_divides_or_clones_a_full_batch(monkeypatch) -> None:
    manager, _ = _make_reward_manager()
    manager._term_instances = {}
    divided_sizes: list[int] = []
    cloned_sizes: list[int] = []
    original_divide_in_place = torch.Tensor.div_
    original_clone = torch.Tensor.clone

    def tracked_divide_in_place(self, *args, **kwargs):
        divided_sizes.append(self.numel())
        return original_divide_in_place(self, *args, **kwargs)

    def tracked_clone(self, *args, **kwargs):
        cloned_sizes.append(self.numel())
        return original_clone(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "div_", tracked_divide_in_place)
    monkeypatch.setattr(torch.Tensor, "clone", tracked_clone)

    manager.reset(torch.tensor([4, 1]), include_all=False)

    assert divided_sizes == [2, 2, 2, 2]
    assert all(size != 5 for size in cloned_sizes)


def test_actual_duration_metrics_remove_early_termination_length_factor() -> None:
    manager, _ = _make_reward_manager()
    manager._term_instances = {}
    steps = _EPISODE_STEPS.to(dtype=torch.float32)
    for term_name in manager._term_names:
        manager._episode_sums[term_name] = steps * _DT * 2.0
        manager._episode_sums_raw[term_name] = steps * 3.0

    actual = manager.reset(None, include_all=False)

    positive_length = _EPISODE_STEPS > 0
    for term_name in manager._term_names:
        assert torch.equal(
            actual["episode_rate"][f"rew_{term_name}"][positive_length],
            torch.full((4,), 2.0),
        )
        assert torch.equal(
            actual["raw_episode_mean"][f"raw_rew_{term_name}"][positive_length],
            torch.full((4,), 3.0),
        )
        assert actual["episode_rate"][f"rew_{term_name}"][~positive_length].item() == 0.0
        assert actual["raw_episode_mean"][f"raw_rew_{term_name}"][~positive_length].item() == 0.0
        assert torch.unique(actual["episode"][f"rew_{term_name}"][positive_length]).numel() > 1


def _make_fill_extras_task(
    reward_manager: object,
    *,
    dense_episode_stats: bool,
) -> BaseTask:
    task = object.__new__(BaseTask)
    task.reward_manager = reward_manager
    task._dense_episode_stats_each_step = dense_episode_stats
    task.extras = {}
    task.time_out_buf = torch.tensor([False, True, False, False, False])
    return task


def test_ppo_sparse_contract_is_forwarded_by_base_task() -> None:
    manager, _ = _make_reward_manager()
    task = _make_fill_extras_task(manager, dense_episode_stats=False)
    env_ids = torch.tensor([3, 1])

    task._fill_extras(env_ids)

    _assert_base_task_extras(task, _expected_extras(env_ids, include_all=False))


def test_fast_sac_dense_contract_is_forwarded_by_base_task() -> None:
    manager, _ = _make_reward_manager()
    task = _make_fill_extras_task(manager, dense_episode_stats=True)
    env_ids = torch.empty(0, dtype=torch.long)

    task._fill_extras(env_ids)

    _assert_base_task_extras(task, _expected_extras(env_ids, include_all=True))


def test_base_task_supports_new_fake_signature_without_capability_marker() -> None:
    calls: list[tuple[torch.Tensor, bool]] = []

    class _FakeRewardManager:
        def reset(self, env_ids: torch.Tensor, *, include_all: bool = True):
            calls.append((env_ids, include_all))
            return {"episode": {"rew_fake": torch.tensor([1.0])}}

    task = _make_fill_extras_task(_FakeRewardManager(), dense_episode_stats=False)
    env_ids = torch.tensor([2])

    task._fill_extras(env_ids)

    assert calls == [(env_ids, False)]
    assert task.extras["episode_all"] == {}
    assert task.extras["raw_episode"] == {}
    assert task.extras["raw_episode_all"] == {}
    assert task.extras["episode_rate"] == {}
    assert task.extras["raw_episode_mean"] == {}


def test_base_task_preserves_legacy_reward_manager_signature() -> None:
    calls: list[torch.Tensor] = []

    class _LegacyRewardManager:
        def reset(self, env_ids: torch.Tensor):
            calls.append(env_ids)
            return {
                "episode": {"rew_legacy": torch.tensor([2.0])},
                "episode_all": {"rew_legacy": torch.arange(5.0)},
            }

    task = _make_fill_extras_task(_LegacyRewardManager(), dense_episode_stats=False)
    env_ids = torch.tensor([2])

    task._fill_extras(env_ids)

    assert calls == [env_ids]
    assert torch.equal(task.extras["episode"]["rew_legacy"], torch.tensor([2.0]))
    assert torch.equal(task.extras["episode_all"]["rew_legacy"], torch.arange(5.0))
    assert task.extras["raw_episode"] == {}
    assert task.extras["raw_episode_all"] == {}
    assert task.extras["episode_rate"] == {}
    assert task.extras["raw_episode_mean"] == {}


@pytest.mark.parametrize("invalid_value", [None, 0, 1, "false"])
def test_reward_reset_include_all_requires_bool(invalid_value: object) -> None:
    manager, _ = _make_reward_manager()

    with pytest.raises(TypeError, match="include_all must be a boolean"):
        manager.reset(torch.tensor([0]), include_all=invalid_value)  # type: ignore[arg-type]
