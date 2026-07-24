from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.replay import (
    _expected_replay_frame_count,
    _replay_reached_full_source_terminal,
    _source_frame_to_materialized_index,
)


def _global_runtime_contract(*, prepend: int = 10, append: int = 0) -> dict:
    return {
        "source_semantics": "global_multi_clip_runtime",
        "prepend": {"steps": prepend},
        "append": {"steps": append},
    }


def test_rule90_ball_materialized_replay_count_and_window_mapping() -> None:
    source_frames = 319
    prepend_frames = 10

    assert _expected_replay_frame_count(
        source_frames,
        _global_runtime_contract(prepend=prepend_frames),
    ) == 329
    assert _source_frame_to_materialized_index(0, prepend_frames) == 10
    assert _source_frame_to_materialized_index(59, prepend_frames) == 69
    assert _source_frame_to_materialized_index(232, prepend_frames) == 242
    assert _source_frame_to_materialized_index(318, prepend_frames) == 328


def test_direct_replay_terminal_is_full_source_frame_only() -> None:
    command = SimpleNamespace(
        time_steps=torch.tensor([317], dtype=torch.long),
        current_clip_lengths=torch.tensor([319], dtype=torch.long),
        _runtime_default_pose_prepend_active=torch.tensor([False]),
    )

    assert not _replay_reached_full_source_terminal(command, 0)
    command.time_steps[0] = 318
    assert _replay_reached_full_source_terminal(command, 0)

    # Even a terminal-looking source clock cannot end while the materialized
    # runtime prefix is active.
    command._runtime_default_pose_prepend_active[0] = True
    assert not _replay_reached_full_source_terminal(command, 0)


def test_episodic_motion_end_margin_and_visualize_default_are_unchanged() -> None:
    command = SimpleNamespace(
        time_steps=torch.tensor([316, 317, 318], dtype=torch.long),
        _current_clip_lengths=lambda: torch.tensor([319, 319, 319], dtype=torch.long),
    )

    assert MotionCommand.motion_end_mask(command).tolist() == [False, True, True]

    parameter = inspect.signature(
        WholeBodyTrackingManager.step_visualize_motion
    ).parameters["advance_motion"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is True
