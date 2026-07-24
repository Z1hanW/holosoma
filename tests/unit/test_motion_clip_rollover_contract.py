from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.termination.manager import TerminationManager

_STANDARD_MOTION_END = "holosoma.managers.termination.terms.wbt:motion_ends"


def _make_command(
    *,
    term_func: str = _STANDARD_MOTION_END,
    is_timeout: bool = False,
) -> MotionCommand:
    env = SimpleNamespace(num_envs=1, device="cpu")
    cfg = TerminationManagerCfg(
        terms={
            "motion_ends": TerminationTermCfg(
                func=term_func,
                is_timeout=is_timeout,
            )
        }
    )
    env.termination_manager = TerminationManager(cfg, env, "cpu")

    command = object.__new__(MotionCommand)
    command._env = env
    command._disable_clip_end_reset = False
    command.time_steps = torch.tensor([5], dtype=torch.long)
    return command


def _install_rollover_spies(command: MotionCommand, *, clip_length: int = 5) -> list[tuple[str, list[int]]]:
    events: list[tuple[str, list[int]]] = []

    def record(name: str, env_ids: torch.Tensor) -> None:
        events.append((name, env_ids.tolist()))

    command._current_clip_lengths = lambda: torch.tensor([clip_length], dtype=torch.long)
    command.reset = lambda env_ids: record("reset", env_ids)
    command._env.simulator = SimpleNamespace(
        robot_root_states=torch.zeros((1, 13), dtype=torch.float32),
        set_actor_root_state_tensor_robots=lambda env_ids, _states: record("root", env_ids),
        set_dof_state_tensor_robots=lambda env_ids: record("dof", env_ids),
        refresh_sim_tensors=lambda: events.append(("refresh", [])),
    )
    return events


def _clear_reset_disable_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HOLOSOMA_DISABLE_AUTO_RESET", raising=False)
    monkeypatch.delenv("HOLOSOMA_DISABLE_MOTION_END_RESET", raising=False)


def test_standard_episodic_motion_end_skips_rollover_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reset_disable_flags(monkeypatch)
    command = _make_command()

    def unexpected_clip_length_read() -> torch.Tensor:
        raise AssertionError("episodic motion_ends must skip the rollover scan")

    command._current_clip_lengths = unexpected_clip_length_read
    command.reset = lambda _env_ids: pytest.fail("episodic motion_ends must own reset")

    assert command._has_standard_episodic_motion_end_contract()
    command._handle_clip_rollover()
    assert command.time_steps.tolist() == [5]


@pytest.mark.parametrize(
    ("term_func", "is_timeout"),
    [
        ("holosoma.managers.termination.terms.common:timeout_exceeded", False),
        (_STANDARD_MOTION_END, True),
    ],
)
def test_nonstandard_or_timeout_term_falls_back_to_rollover(
    term_func: str,
    is_timeout: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_reset_disable_flags(monkeypatch)
    command = _make_command(term_func=term_func, is_timeout=is_timeout)
    events = _install_rollover_spies(command)

    assert not command._has_standard_episodic_motion_end_contract()
    command._handle_clip_rollover()

    assert events == [("reset", [0]), ("root", [0]), ("dof", [0]), ("refresh", [])]


def test_custom_manager_fails_closed_even_with_matching_visible_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reset_disable_flags(monkeypatch)
    command = _make_command()
    real_manager = command._env.termination_manager
    command._env.termination_manager = SimpleNamespace(
        cfg=real_manager.cfg,
        _term_names=real_manager._term_names,
        _term_funcs=real_manager._term_funcs,
        _term_instances=real_manager._term_instances,
    )
    events = _install_rollover_spies(command)

    assert not command._has_standard_episodic_motion_end_contract()
    command._handle_clip_rollover()

    assert events == [("reset", [0]), ("root", [0]), ("dof", [0]), ("refresh", [])]


def test_missing_motion_end_config_falls_back_to_rollover(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reset_disable_flags(monkeypatch)
    command = _make_command()
    command._env.termination_manager.cfg.terms.clear()
    events = _install_rollover_spies(command)

    assert not command._has_standard_episodic_motion_end_contract()
    command._handle_clip_rollover()

    assert events == [("reset", [0]), ("root", [0]), ("dof", [0]), ("refresh", [])]


def test_wrapped_resolved_motion_end_falls_back_to_rollover(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reset_disable_flags(monkeypatch)
    command = _make_command()
    command._env.termination_manager._term_funcs["motion_ends"] = lambda _env: torch.tensor([False])
    events = _install_rollover_spies(command)

    assert not command._has_standard_episodic_motion_end_contract()
    command._handle_clip_rollover()

    assert events == [("reset", [0]), ("root", [0]), ("dof", [0]), ("refresh", [])]


@pytest.mark.parametrize(
    "flag_name",
    ["HOLOSOMA_DISABLE_AUTO_RESET", "HOLOSOMA_DISABLE_MOTION_END_RESET"],
)
def test_runtime_reset_disable_flags_restore_rollover_fallback(
    flag_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_reset_disable_flags(monkeypatch)
    command = _make_command()
    events = _install_rollover_spies(command)
    monkeypatch.setenv(flag_name, "true")

    assert command._has_standard_episodic_motion_end_contract()
    assert not command._termination_owns_clip_rollover()
    command._handle_clip_rollover()

    assert events == [("reset", [0]), ("root", [0]), ("dof", [0]), ("refresh", [])]


def test_disable_clip_end_reset_preserves_clamp_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reset_disable_flags(monkeypatch)
    command = _make_command()
    command._disable_clip_end_reset = True
    events = _install_rollover_spies(command)

    command._handle_clip_rollover()

    assert command.time_steps.tolist() == [4]
    assert events == []
