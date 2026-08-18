from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "export_heading_path_commands.py"
SPEC = importlib.util.spec_from_file_location("export_heading_path_commands", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _yaw_to_wxyz(yaw: np.ndarray) -> np.ndarray:
    result = np.zeros((yaw.size, 4), dtype=np.float64)
    result[:, 0] = np.cos(yaw / 2.0)
    result[:, 3] = np.sin(yaw / 2.0)
    return result


def _compute(xy: np.ndarray, yaw: np.ndarray, velocity_xy: np.ndarray, **overrides):
    root_pos = np.column_stack((xy, np.full((xy.shape[0],), 0.8)))
    root_velocity = np.column_stack((velocity_xy, np.zeros((xy.shape[0],))))
    kwargs = {
        "carry_start": 0,
        "carry_end": xy.shape[0],
        "lookahead_m": 0.15,
        "smoothing_steps": 1,
        "rdp_epsilon_m": 0.005,
        "hard_turn_deg": 60.0,
        "minimum_leg_m": 0.08,
        "stop_speed_mps": 0.001,
        "stop_min_steps": 100,
    }
    kwargs.update(overrides)
    return MODULE.compute_heading_path_commands(
        root_pos,
        _yaw_to_wxyz(yaw),
        root_velocity,
        **kwargs,
    )


def test_straight_path_matches_constant_forward_lookahead() -> None:
    xy = np.column_stack((np.linspace(0.0, 1.0, 21), np.zeros((21,))))
    yaw = np.zeros((21,))
    result = _compute(xy, yaw, np.tile([0.5, 0.0], (21, 1)))

    np.testing.assert_allclose(result["command"][:17, 0], 0.15, atol=1.0e-8)
    np.testing.assert_allclose(result["command"][:, 1:], 0.0, atol=1.0e-8)


def test_right_angle_lookahead_does_not_cut_across_corner() -> None:
    first_leg = np.column_stack((np.linspace(0.0, 1.0, 11), np.zeros((11,))))
    second_leg = np.column_stack((np.ones((10,)), np.linspace(0.1, 1.0, 10)))
    xy = np.concatenate((first_leg, second_leg), axis=0)
    yaw = np.concatenate((np.zeros((11,)), np.full((10,), np.pi / 2.0)))
    velocity = np.gradient(xy, axis=0) * 10.0
    result = _compute(xy, yaw, velocity)

    assert 10 in result["hard_boundary_frames"].tolist()
    np.testing.assert_allclose(result["target_xy_w"][9], [1.0, 0.0], atol=1.0e-8)
    np.testing.assert_allclose(result["command"][9, :2], [0.1, 0.0], atol=1.0e-8)
    np.testing.assert_allclose(result["command"][11, :2], [0.15, 0.0], atol=1.0e-8)


def test_target_frames_preserve_order_on_returning_path() -> None:
    outbound = np.column_stack((np.linspace(0.0, 1.0, 11), np.zeros((11,))))
    inbound = np.column_stack((np.linspace(0.9, 0.0, 10), np.zeros((10,))))
    xy = np.concatenate((outbound, inbound), axis=0)
    yaw = np.concatenate((np.zeros((11,)), np.full((10,), np.pi)))
    velocity = np.gradient(xy, axis=0) * 10.0
    result = _compute(xy, yaw, velocity)

    assert 10 in result["hard_boundary_frames"].tolist()
    assert np.all(result["target_frame"] >= np.arange(xy.shape[0]))
    assert result["target_frame"][9] <= 10.0
    assert result["target_frame"][11] > 11.0


def test_xm0_pickup_window_stays_latched_after_object_lowers() -> None:
    root_pos = np.column_stack(
        (
            np.zeros((15,)),
            np.zeros((15,)),
            np.full((15,), 0.8),
        )
    )
    object_pos = root_pos.copy()
    object_pos[:, 2] = np.asarray(
        [0.0] * 3 + [0.3] * 7 + [0.0] * 5,
        dtype=np.float64,
    )

    start, end, _ = MODULE.xm0_post_pickup_window_from_rel_z(root_pos, object_pos)

    assert start == 3
    assert end == 15


def test_heading_relative_commands_are_world_se2_invariant() -> None:
    xy = np.column_stack((np.linspace(0.0, 1.0, 21), np.linspace(0.0, 0.3, 21)))
    yaw = np.full((21,), 0.2)
    velocity = np.tile([0.5, 0.15], (21, 1))
    reference = _compute(xy, yaw, velocity)

    rotation_angle = 1.1
    rotation = np.asarray(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )
    transformed_xy = xy @ rotation.T + np.asarray([3.0, -2.0])
    transformed_velocity = velocity @ rotation.T
    transformed = _compute(transformed_xy, yaw + rotation_angle, transformed_velocity)

    np.testing.assert_allclose(transformed["command"], reference["command"], atol=1.0e-8)


def test_turn_then_forward_right_angle_never_couples_command_axes() -> None:
    first_leg = np.column_stack((np.linspace(0.0, 1.0, 11), np.zeros((11,))))
    second_leg = np.column_stack((np.ones((10,)), np.linspace(0.1, 1.0, 10)))
    xy = np.concatenate((first_leg, second_leg), axis=0)
    root_pos = np.column_stack((xy, np.full((xy.shape[0],), 0.8)))
    yaw = np.concatenate((np.zeros((11,)), np.full((10,), np.pi / 2.0)))

    result = MODULE.compute_turn_then_forward_commands(
        root_pos,
        _yaw_to_wxyz(yaw),
        carry_start=0,
        carry_end=root_pos.shape[0],
        smoothing_steps=1,
        rdp_epsilon_m=0.005,
        minimum_leg_m=0.08,
        minimum_turn_deg=25.0,
    )

    command = result["command"]
    assert np.all(command[:, 1] == 0.0)
    assert not np.any((command[:, 0] != 0.0) & (command[:, 2] != 0.0))
    yaw_frames = np.flatnonzero(result["phase"] == 2)
    assert yaw_frames.size > 0
    assert yaw_frames[-1] == 9
    assert np.all(command[yaw_frames, 2] == pytest.approx(np.pi / 2.0))
    assert np.all(command[result["phase"] == 1, 0] > 0.0)


def test_turn_then_forward_initial_alignment_precedes_forward_phase() -> None:
    xy = np.column_stack((np.zeros((31,)), np.linspace(0.0, 1.0, 31)))
    root_pos = np.column_stack((xy, np.full((31,), 0.8)))
    result = MODULE.compute_turn_then_forward_commands(
        root_pos,
        _yaw_to_wxyz(np.zeros((31,))),
        carry_start=0,
        carry_end=31,
        smoothing_steps=1,
        rdp_epsilon_m=0.005,
        minimum_leg_m=0.08,
        minimum_turn_deg=25.0,
    )

    phases = result["phase"]
    yaw_frames = np.flatnonzero(phases == 2)
    forward_frames = np.flatnonzero(phases == 1)
    assert yaw_frames[0] == 0
    assert yaw_frames[-1] < forward_frames[0]
    np.testing.assert_allclose(result["command"][yaw_frames, 2], np.pi / 2.0)
    np.testing.assert_allclose(result["command"][forward_frames, 2], 0.0)


def test_prune_short_polyline_leg_removes_spurious_vertex() -> None:
    path = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.04, 0.02],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )

    pruned = MODULE.prune_short_polyline_legs(path, [0, 1, 2, 3], minimum_leg_m=0.10)

    assert pruned in ([0, 1, 3], [0, 2, 3])
