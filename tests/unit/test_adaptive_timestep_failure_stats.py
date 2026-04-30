from __future__ import annotations

from types import SimpleNamespace

import torch

from holosoma.managers.command.terms.wbt import MotionCommand


class _RecordingAdaptiveSampler:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def update_current_bin_failed_count(
        self,
        failed_at_time_step: torch.Tensor,
        clip_ids: torch.Tensor | None = None,
    ) -> None:
        self.calls.append(
            (
                failed_at_time_step.detach().clone(),
                None if clip_ids is None else clip_ids.detach().clone(),
            )
        )


def _make_motion_command(terminated: torch.Tensor) -> tuple[MotionCommand, _RecordingAdaptiveSampler]:
    sampler = _RecordingAdaptiveSampler()
    command = object.__new__(MotionCommand)
    command.use_adaptive_timesteps_sampler = True
    command._env = SimpleNamespace(
        termination_manager=SimpleNamespace(terminated=terminated),
    )
    command.time_steps = torch.tensor([4, 17, 29, 8], dtype=torch.long)
    command.clip_ids = torch.tensor([0, 2, 1, 3], dtype=torch.long)
    command.adaptive_timesteps_sampler = sampler
    return command, sampler


def test_adaptive_timestep_failure_stats_use_previous_clip_ids_before_resample() -> None:
    command, sampler = _make_motion_command(torch.tensor([False, True, False, True]))
    env_ids = torch.tensor([1, 3], dtype=torch.long)

    command._update_adaptive_timestep_failure_stats_before_resample(env_ids)
    command.clip_ids[env_ids] = torch.tensor([0, 0], dtype=torch.long)

    assert len(sampler.calls) == 1
    failed_steps, failed_clip_ids = sampler.calls[0]
    assert torch.equal(failed_steps, torch.tensor([17, 8], dtype=torch.long))
    assert torch.equal(failed_clip_ids, torch.tensor([2, 3], dtype=torch.long))


def test_adaptive_timestep_failure_stats_skip_when_no_env_failed() -> None:
    command, sampler = _make_motion_command(torch.tensor([False, False, False, False]))

    command._update_adaptive_timestep_failure_stats_before_resample(torch.tensor([1, 3], dtype=torch.long))

    assert sampler.calls == []
