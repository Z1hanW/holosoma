from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import torch

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.managers.observation.base import reusable_observation_base_term
from holosoma.managers.observation.manager import ObservationManager


class _NoTiming:
    enabled = False


def _env(values: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        values=values,
        num_envs=values.shape[0],
        device=str(values.device),
        step_timing=_NoTiming(),
    )


def _manager_cfg(*, reuse: bool, groups: dict[str, ObsGroupCfg]) -> ObservationManagerCfg:
    return ObservationManagerCfg(groups=groups, reuse_exact_base_terms=reuse)


def _group(
    func_path: str,
    *,
    noise: float = 0.0,
    enable_noise: bool = False,
    history_length: int = 1,
    concatenate: bool = True,
    params: dict | None = None,
) -> ObsGroupCfg:
    return ObsGroupCfg(
        concatenate=concatenate,
        enable_noise=enable_noise,
        history_length=history_length,
        terms={
            "value": ObsTermCfg(
                func=func_path,
                params={} if params is None else params,
                noise=noise,
            )
        },
    )


def _install_resolver(monkeypatch, functions: dict[str, Callable]) -> None:
    monkeypatch.setattr(
        "holosoma.managers.observation.manager.resolve_callable",
        lambda path, *, context: functions[path],
    )


def test_exact_base_reuse_requires_both_opt_ins_and_empty_params(monkeypatch) -> None:
    calls = {"marked": 0, "unmarked": 0}

    @reusable_observation_base_term
    def marked(env, *, offset: float = 0.0):
        calls["marked"] += 1
        return env.values + offset

    def unmarked(env):
        calls["unmarked"] += 1
        return env.values

    _install_resolver(monkeypatch, {"marked": marked, "unmarked": unmarked})
    env = _env(torch.arange(6, dtype=torch.float32).reshape(2, 3))

    groups = {"a": _group("marked"), "b": _group("marked")}
    ObservationManager(_manager_cfg(reuse=False, groups=groups), env, env.device).compute()
    assert calls["marked"] == 2

    calls["marked"] = 0
    ObservationManager(_manager_cfg(reuse=True, groups=groups), env, env.device).compute()
    assert calls["marked"] == 1

    calls["marked"] = 0
    parameterized = {
        "a": _group("marked", params={"offset": 1.0}),
        "b": _group("marked", params={"offset": 1.0}),
    }
    ObservationManager(_manager_cfg(reuse=True, groups=parameterized), env, env.device).compute()
    assert calls["marked"] == 2

    unmarked_groups = {"a": _group("unmarked"), "b": _group("unmarked")}
    ObservationManager(_manager_cfg(reuse=True, groups=unmarked_groups), env, env.device).compute()
    assert calls["unmarked"] == 2


def test_exact_base_reuse_preserves_noise_values_and_rng_state(monkeypatch) -> None:
    calls = {"count": 0}

    @reusable_observation_base_term
    def marked(env):
        calls["count"] += 1
        return env.values

    _install_resolver(monkeypatch, {"marked": marked})
    env = _env(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    groups = {
        "noisy_a": _group("marked", noise=0.2, enable_noise=True),
        "plain": _group("marked"),
        "noisy_b": _group("marked", noise=0.7, enable_noise=True),
    }
    baseline = ObservationManager(_manager_cfg(reuse=False, groups=groups), env, env.device)
    candidate = ObservationManager(_manager_cfg(reuse=True, groups=groups), env, env.device)

    torch.manual_seed(20260717)
    baseline_obs = baseline.compute()
    baseline_rng = torch.random.get_rng_state().clone()

    calls["count"] = 0
    torch.manual_seed(20260717)
    candidate_obs = candidate.compute()
    candidate_rng = torch.random.get_rng_state().clone()

    assert calls["count"] == 1
    assert baseline_obs.keys() == candidate_obs.keys()
    for key in baseline_obs:
        assert torch.equal(baseline_obs[key], candidate_obs[key])
    assert torch.equal(baseline_rng, candidate_rng)


def test_exact_base_reuse_preserves_history_reset_and_preview(monkeypatch) -> None:
    @reusable_observation_base_term
    def marked(env):
        return env.values

    _install_resolver(monkeypatch, {"marked": marked})
    baseline_env = _env(torch.zeros(3, 2))
    candidate_env = _env(torch.zeros(3, 2))
    groups = {
        "history": _group("marked", history_length=3),
        "plain": _group("marked"),
    }
    baseline = ObservationManager(_manager_cfg(reuse=False, groups=groups), baseline_env, baseline_env.device)
    candidate = ObservationManager(_manager_cfg(reuse=True, groups=groups), candidate_env, candidate_env.device)

    for step in range(4):
        values = torch.arange(6, dtype=torch.float32).reshape(3, 2) + step * 10
        baseline_env.values = values.clone()
        candidate_env.values = values.clone()
        baseline_obs = baseline.compute()
        candidate_obs = candidate.compute()
        for key in baseline_obs:
            assert torch.equal(baseline_obs[key], candidate_obs[key])

    reset_ids = torch.tensor([0, 2], dtype=torch.long)
    baseline.reset(reset_ids)
    candidate.reset(reset_ids)
    baseline_env.values.add_(100)
    candidate_env.values.add_(100)

    baseline_preview = baseline.compute(modify_history=False)
    candidate_preview = candidate.compute(modify_history=False)
    baseline_after_preview = baseline.compute()
    candidate_after_preview = candidate.compute()
    for key in baseline_preview:
        assert torch.equal(baseline_preview[key], candidate_preview[key])
        assert torch.equal(baseline_after_preview[key], candidate_after_preview[key])


def test_exact_base_cache_is_ephemeral_and_tracks_active_groups(monkeypatch) -> None:
    calls = {"count": 0}

    @reusable_observation_base_term
    def marked(env):
        calls["count"] += 1
        return env.values

    _install_resolver(monkeypatch, {"marked": marked})
    env = _env(torch.tensor([[1.0, 2.0]]))
    groups = {"a": _group("marked"), "b": _group("marked")}
    manager = ObservationManager(_manager_cfg(reuse=True, groups=groups), env, env.device)

    first = manager.compute()
    env.values = torch.tensor([[7.0, 9.0]])
    second = manager.compute()
    assert calls["count"] == 2
    assert torch.equal(first["a"], torch.tensor([[1.0, 2.0]]))
    assert torch.equal(second["a"], torch.tensor([[7.0, 9.0]]))

    manager.set_active_groups(["b"])
    manager.compute()
    assert calls["count"] == 3
    manager.set_active_groups(["a", "b"])
    manager.compute()
    assert calls["count"] == 4


def test_exact_base_reuse_preserves_clone_isolation_for_views(monkeypatch) -> None:
    @reusable_observation_base_term
    def marked(env):
        return env.values

    _install_resolver(monkeypatch, {"marked": marked})
    env = _env(torch.arange(4, dtype=torch.float32).reshape(2, 2))
    groups = {
        "a": _group("marked", concatenate=False),
        "b": _group("marked", concatenate=False),
    }
    manager = ObservationManager(_manager_cfg(reuse=True, groups=groups), env, env.device)
    obs = manager.compute()

    assert isinstance(obs["a"], dict)
    assert isinstance(obs["b"], dict)
    a = obs["a"]["value"]
    b = obs["b"]["value"]
    assert a.data_ptr() != b.data_ptr()
    assert a.data_ptr() != env.values.data_ptr()
    a.add_(100)
    assert torch.equal(b, env.values)
