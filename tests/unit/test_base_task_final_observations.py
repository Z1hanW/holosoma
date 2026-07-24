from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from holosoma.envs.base_task.base_task import BaseTask


class _DirectRefreshTask(BaseTask):
    supports_direct_post_reset_refresh_ids = True


class _InheritedDirectRefreshTask(_DirectRefreshTask):
    pass


class _StepTiming:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.records: list[str] = []

    @contextmanager
    def record(self, name: str):
        self.records.append(name)
        yield


def _make_post_physics_task(
    *,
    reset_mask: list[int],
    timeout_mask: list[bool],
    timing_enabled: bool,
    task_cls: type[BaseTask] = BaseTask,
) -> tuple[BaseTask, dict[str, object]]:
    num_envs = len(reset_mask)
    task = object.__new__(task_cls)
    task.device = "cpu"
    task.num_envs = num_envs
    task.step_timing = _StepTiming(enabled=timing_enabled)
    task._dense_episode_stats_each_step = False
    task.reset_buf = torch.tensor(reset_mask, dtype=torch.long)
    task.time_out_buf = torch.tensor(timeout_mask, dtype=torch.bool)
    task.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    task.obs_buf_dict = {"critic_obs": torch.full((num_envs, 2), -1.0)}
    task.extras = {
        "sentinel": "preserved",
        "time_outs": task.time_out_buf,
        "final_observations": {"critic_obs": torch.full((num_envs, 2), 999.0)},
    }
    task.log_dict = {"metric": torch.tensor(1.0)}
    task.viewer = False

    calls: dict[str, object] = {
        "final": 0,
        "reset_ids": [],
        "refresh_select": 0,
        "refresh_ids": [],
        "tasks": 0,
        "observations": 0,
    }

    task._refresh_sim_tensors = lambda: None
    task._update_counters_each_step = lambda: None
    task._pre_compute_observations_callback = lambda: None
    task._check_termination = lambda: None
    task._compute_reward = lambda: None
    task._update_log_dict = lambda: None
    task._draw_scandots_in_viewer = lambda: None

    def reset_envs_idx(env_ids: torch.Tensor) -> None:
        reset_ids = calls["reset_ids"]
        assert isinstance(reset_ids, list)
        reset_ids.append(env_ids.clone())

    task.reset_envs_idx = reset_envs_idx
    task._test_refresh_env_ids = torch.empty(0, dtype=torch.long)

    def get_envs_to_refresh() -> torch.Tensor:
        calls["refresh_select"] = int(calls["refresh_select"]) + 1
        return task._test_refresh_env_ids

    task._get_envs_to_refresh = get_envs_to_refresh

    def refresh_envs_after_reset(env_ids: torch.Tensor) -> None:
        refresh_ids = calls["refresh_ids"]
        assert isinstance(refresh_ids, list)
        refresh_ids.append(env_ids.clone())

    task._refresh_envs_after_reset = refresh_envs_after_reset

    def update_tasks() -> None:
        calls["tasks"] = int(calls["tasks"]) + 1

    task._update_tasks_callback = update_tasks

    def compute_observations() -> None:
        calls["observations"] = int(calls["observations"]) + 1
        task.obs_buf_dict = {
            "critic_obs": torch.arange(num_envs * 2, dtype=torch.float32).reshape(num_envs, 2) + 100.0
        }

    task._compute_observations = compute_observations
    task._post_compute_observations_callback = lambda: None
    task._clip_observations = lambda: None
    return task, calls


@pytest.mark.parametrize("invalid_value", [None, 0, 1, "false"])
def test_collection_extras_contract_requires_a_bool(invalid_value: object) -> None:
    task = object.__new__(BaseTask)

    with pytest.raises(TypeError, match="dense_episode_stats must be a bool"):
        task.set_collection_extras_contract(dense_episode_stats=invalid_value)  # type: ignore[arg-type]


@pytest.mark.parametrize("timing_enabled", [False, True])
def test_ordinary_terminal_reset_skips_final_observation_compute(timing_enabled: bool) -> None:
    task, calls = _make_post_physics_task(
        reset_mask=[1, 0, 0],
        timeout_mask=[False, False, False],
        timing_enabled=timing_enabled,
    )

    def unexpected_final_observation_compute():
        calls["final"] = int(calls["final"]) + 1
        raise AssertionError("ordinary terminal reset must not compute final observations")

    task._compute_final_observations = unexpected_final_observation_compute

    BaseTask._post_physics_step(task)

    assert calls["final"] == 0
    assert len(calls["reset_ids"]) == 1
    assert torch.equal(calls["reset_ids"][0], torch.tensor([0]))
    assert "final_observations" not in task.extras
    assert task.extras["sentinel"] == "preserved"
    assert task.extras["to_log"] is task.log_dict


@pytest.mark.parametrize("timing_enabled", [False, True])
def test_timeout_computes_once_and_stores_only_bootstrap_eligible_rows(timing_enabled: bool) -> None:
    task, calls = _make_post_physics_task(
        reset_mask=[1, 1, 0],
        timeout_mask=[True, False, False],
        timing_enabled=timing_enabled,
    )
    terminal_observations = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
    )

    def compute_final_observations():
        calls["final"] = int(calls["final"]) + 1
        return {"critic_obs": terminal_observations.clone()}

    task._compute_final_observations = compute_final_observations

    BaseTask._post_physics_step(task)

    assert calls["final"] == 1
    assert torch.equal(calls["reset_ids"][0], torch.tensor([0, 1]))
    stored = task.extras["final_observations"]["critic_obs"]
    assert torch.equal(stored[0], terminal_observations[0])
    assert torch.equal(stored[1:], torch.zeros(2, 2))
    assert task.extras["time_outs"] is task.time_out_buf
    assert torch.equal(task.extras["time_outs"], torch.tensor([True, False, False]))


@pytest.mark.parametrize("timing_enabled", [False, True])
def test_no_reset_path_keeps_regular_observation_and_extras_semantics(timing_enabled: bool) -> None:
    task, calls = _make_post_physics_task(
        reset_mask=[0, 0, 0],
        timeout_mask=[False, False, False],
        timing_enabled=timing_enabled,
    )

    def unexpected_final_observation_compute():
        calls["final"] = int(calls["final"]) + 1
        raise AssertionError("no-reset transition must not compute final observations")

    task._compute_final_observations = unexpected_final_observation_compute

    BaseTask._post_physics_step(task)

    assert calls["final"] == 0
    assert calls["tasks"] == 1
    assert calls["observations"] == 1
    assert torch.equal(
        task.obs_buf_dict["critic_obs"],
        torch.tensor([[100.0, 101.0], [102.0, 103.0], [104.0, 105.0]]),
    )
    # Empty transitions must not run the reset stack merely to refresh extras.
    # In particular, reward/stateful-manager reset hooks are reset-only work.
    assert calls["reset_ids"] == []
    assert calls["refresh_select"] == 0
    assert calls["refresh_ids"] == []
    assert task.extras["reset_env_ids"].numel() == 0
    for key in ("episode", "episode_all", "raw_episode", "raw_episode_all"):
        assert task.extras[key] == {}
    assert task.extras["time_outs"] is task.time_out_buf
    assert not torch.any(task.extras["time_outs"])
    assert "final_observations" not in task.extras
    assert task.extras["sentinel"] == "preserved"
    assert task.extras["to_log"] is task.log_dict


@pytest.mark.parametrize("timing_enabled", [False, True])
def test_reset_then_no_reset_clears_transition_extras_without_reentering_reset_hooks(
    timing_enabled: bool,
) -> None:
    task, calls = _make_post_physics_task(
        reset_mask=[1, 0, 0],
        timeout_mask=[True, False, False],
        timing_enabled=timing_enabled,
    )
    task._compute_final_observations = lambda: {
        "critic_obs": torch.arange(6, dtype=torch.float32).reshape(3, 2)
    }

    reset_hook_calls: list[torch.Tensor] = []
    record_reset = task.reset_envs_idx

    def reset_with_episode_extras(env_ids: torch.Tensor) -> None:
        record_reset(env_ids)
        reset_hook_calls.append(env_ids.clone())
        task.extras["episode"] = {"rew_term": torch.tensor([1.0])}
        task.extras["episode_all"] = {"rew_term": torch.tensor([1.0, 2.0, 3.0])}
        task.extras["raw_episode"] = {"raw_rew_term": torch.tensor([4.0])}
        task.extras["raw_episode_all"] = {
            "raw_rew_term": torch.tensor([4.0, 5.0, 6.0])
        }

    task.reset_envs_idx = reset_with_episode_extras

    BaseTask._post_physics_step(task)

    assert len(reset_hook_calls) == 1
    assert torch.equal(reset_hook_calls[0], torch.tensor([0]))
    assert task.extras["episode"]
    assert task.extras["episode_all"]
    assert task.extras["raw_episode"]
    assert task.extras["raw_episode_all"]
    assert torch.equal(task.extras["time_outs"], torch.tensor([True, False, False]))
    assert torch.equal(task.extras["reset_env_ids"], torch.tensor([0]))
    assert "final_observations" in task.extras

    # The next transition has no reset.  BaseTask normally zeroes both masks in
    # _check_termination(); this lightweight fixture does so explicitly because
    # its termination callback is a no-op.
    task.reset_buf.zero_()
    task.time_out_buf.zero_()
    BaseTask._post_physics_step(task)

    assert len(reset_hook_calls) == 1
    assert len(calls["reset_ids"]) == 1
    assert calls["refresh_select"] == 1
    assert calls["refresh_ids"] == []
    for key in ("episode", "episode_all", "raw_episode", "raw_episode_all"):
        assert task.extras[key] == {}
    assert task.extras["time_outs"] is task.time_out_buf
    assert not torch.any(task.extras["time_outs"])
    assert task.extras["reset_env_ids"].numel() == 0
    assert "final_observations" not in task.extras


@pytest.mark.parametrize("timing_enabled", [False, True])
def test_pending_refresh_from_reset_all_survives_a_no_reset_transition(
    timing_enabled: bool,
) -> None:
    task, calls = _make_post_physics_task(
        reset_mask=[0, 0, 0],
        timeout_mask=[False, False, False],
        timing_enabled=timing_enabled,
    )
    task._compute_final_observations = lambda: (_ for _ in ()).throw(
        AssertionError("no-reset transition must not compute final observations")
    )
    task._reset_refresh_pending = True
    task._test_refresh_env_ids = torch.tensor([0, 1, 2], dtype=torch.long)

    BaseTask._post_physics_step(task)

    assert calls["reset_ids"] == []
    assert task.extras["reset_env_ids"].numel() == 0
    assert calls["refresh_select"] == 1
    assert len(calls["refresh_ids"]) == 1
    assert torch.equal(calls["refresh_ids"][0], torch.tensor([0, 1, 2]))
    assert task._reset_refresh_pending is False


@pytest.mark.parametrize("timing_enabled", [False, True])
def test_in_loop_reset_reuses_exact_reset_ids_without_second_dynamic_selector(
    timing_enabled: bool,
) -> None:
    task, calls = _make_post_physics_task(
        reset_mask=[0, 1, 0, 1],
        timeout_mask=[False, False, False, False],
        timing_enabled=timing_enabled,
        task_cls=_DirectRefreshTask,
    )

    BaseTask._post_physics_step(task)

    assert calls["refresh_select"] == 0
    assert len(calls["refresh_ids"]) == 1
    assert torch.equal(calls["refresh_ids"][0], torch.tensor([1, 3]))


@pytest.mark.parametrize("timing_enabled", [False, True])
def test_preexisting_pending_refresh_uses_mask_selector_even_with_direct_capability(
    timing_enabled: bool,
) -> None:
    task, calls = _make_post_physics_task(
        reset_mask=[0, 1, 0, 0],
        timeout_mask=[False, False, False, False],
        timing_enabled=timing_enabled,
        task_cls=_DirectRefreshTask,
    )
    task._reset_refresh_pending = True
    task._test_refresh_env_ids = torch.tensor([0, 1, 3], dtype=torch.long)

    BaseTask._post_physics_step(task)

    assert calls["refresh_select"] == 1
    assert len(calls["refresh_ids"]) == 1
    assert torch.equal(calls["refresh_ids"][0], torch.tensor([0, 1, 3]))


@pytest.mark.parametrize("timing_enabled", [False, True])
def test_direct_refresh_capability_is_not_inherited_implicitly(
    timing_enabled: bool,
) -> None:
    task, calls = _make_post_physics_task(
        reset_mask=[0, 1, 0, 0],
        timeout_mask=[False, False, False, False],
        timing_enabled=timing_enabled,
        task_cls=_InheritedDirectRefreshTask,
    )
    task._test_refresh_env_ids = torch.tensor([1, 3], dtype=torch.long)

    BaseTask._post_physics_step(task)

    assert calls["refresh_select"] == 1
    assert len(calls["refresh_ids"]) == 1
    assert torch.equal(calls["refresh_ids"][0], torch.tensor([1, 3]))


@pytest.mark.parametrize("timing_enabled", [False, True])
def test_dense_episode_stats_contract_preserves_legacy_empty_reset_behavior(
    timing_enabled: bool,
) -> None:
    task, calls = _make_post_physics_task(
        reset_mask=[0, 0, 0],
        timeout_mask=[False, False, False],
        timing_enabled=timing_enabled,
    )
    task.set_collection_extras_contract(dense_episode_stats=True)
    task._compute_final_observations = lambda: (_ for _ in ()).throw(
        AssertionError("no-reset transition must not compute final observations")
    )
    record_reset = task.reset_envs_idx

    def dense_empty_reset(env_ids: torch.Tensor) -> None:
        record_reset(env_ids)
        task.extras["episode"] = {"rew_term": torch.empty(0)}
        task.extras["episode_all"] = {"rew_term": torch.arange(3, dtype=torch.float32)}
        task.extras["raw_episode"] = {"raw_rew_term": torch.empty(0)}
        task.extras["raw_episode_all"] = {
            "raw_rew_term": torch.arange(3, dtype=torch.float32) + 10.0
        }

    task.reset_envs_idx = dense_empty_reset

    BaseTask._post_physics_step(task)

    assert len(calls["reset_ids"]) == 1
    assert calls["reset_ids"][0].numel() == 0
    assert calls["refresh_select"] == 1
    assert len(task.extras["episode_all"]["rew_term"]) == task.num_envs
    assert len(task.extras["raw_episode_all"]["raw_rew_term"]) == task.num_envs
    assert task.extras["episode"]["rew_term"].numel() == 0
    assert task.extras["raw_episode"]["raw_rew_term"].numel() == 0


@pytest.mark.parametrize("timing_enabled", [False, True])
def test_rejected_timeout_preview_is_not_published_as_final_observation(timing_enabled: bool) -> None:
    task, calls = _make_post_physics_task(
        reset_mask=[1, 0, 0],
        timeout_mask=[True, False, False],
        timing_enabled=timing_enabled,
    )

    def reject_timeout_preview():
        calls["final"] = int(calls["final"]) + 1
        task.time_out_buf[0] = False
        return {"critic_obs": torch.full((3, 2), 123.0)}

    task._compute_final_observations = reject_timeout_preview

    BaseTask._post_physics_step(task)

    assert calls["final"] == 1
    assert not torch.any(task.time_out_buf)
    assert "final_observations" not in task.extras
