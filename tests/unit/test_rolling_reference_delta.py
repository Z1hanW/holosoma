from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from holosoma.managers.observation.terms.wbt import (
    _contact_aware_rolling_reference_delta_command,
)
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy


def _training_fixture(time_steps: list[int]) -> SimpleNamespace:
    frame_count = 6
    positions = torch.zeros((frame_count, 1, 3), dtype=torch.float32)
    positions[:, 0, 1] = torch.arange(frame_count, dtype=torch.float32) * 0.1
    half_yaw = torch.tensor(np.pi / 4.0, dtype=torch.float32)
    quat_xyzw = torch.tensor(
        [0.0, 0.0, torch.sin(half_yaw), torch.cos(half_yaw)],
        dtype=torch.float32,
    )
    quaternions = quat_xyzw.view(1, 1, 4).repeat(frame_count, 1, 1)
    count = len(time_steps)
    command = SimpleNamespace(
        motion=SimpleNamespace(
            has_object=True,
            body_pos_w=positions,
            body_quat_w=quaternions,
        ),
        motion_cfg=SimpleNamespace(
            contact_aware_sparse_root_segment_steps=2,
            contact_aware_sparse_root_zero_yaw_threshold_deg=0.0,
        ),
        num_envs=count,
        device=torch.device("cpu"),
        clip_ids=torch.zeros(count, dtype=torch.long),
        time_steps=torch.tensor(time_steps, dtype=torch.long),
        current_clip_lengths=torch.full((count,), frame_count, dtype=torch.long),
        _get_contact_aware_carry_window_by_clip=lambda: torch.tensor(
            [[1, 5]], dtype=torch.long
        ),
        _get_motion_indices=lambda steps: steps,
    )
    return command


def _inference_fixture() -> WholeBodyTrackingPolicy:
    frame_count = 6
    positions = np.zeros((frame_count, 3), dtype=np.float32)
    positions[:, 1] = np.arange(frame_count, dtype=np.float32) * np.float32(0.1)
    half_yaw = np.float32(np.pi / 4.0)
    quat_wxyz = np.asarray(
        [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
        dtype=np.float32,
    )
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._motion_cfg = {
        "contact_aware_sparse_root_command_mode": "rolling_reference_delta",
        "contact_aware_sparse_root_segment_steps": 2,
        "contact_aware_sparse_root_zero_yaw_threshold_deg": 0.0,
    }
    policy._motion_data = SimpleNamespace(
        has_object=True,
        frame_count=frame_count,
        root_pos_w=positions,
        root_quat_w=np.repeat(quat_wxyz.reshape(1, 4), frame_count, axis=0),
    )
    policy._motion_index_for_test = 1
    policy._get_motion_index = lambda: policy._motion_index_for_test
    policy._get_contact_aware_carry_window = lambda: (1, 5)
    policy._apply_external_sparse_root_command = lambda command: command
    return policy


def test_training_rolling_reference_delta_recomputes_each_frame_and_zeros_tail() -> None:
    actual = _contact_aware_rolling_reference_delta_command(
        _training_fixture([0, 1, 2, 3])
    )

    torch.testing.assert_close(
        actual,
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_training_and_inference_rolling_reference_delta_are_numerically_equal() -> None:
    training = _contact_aware_rolling_reference_delta_command(
        _training_fixture([1])
    ).numpy()
    policy = _inference_fixture()
    inference = policy._get_rolling_reference_delta_command()

    np.testing.assert_allclose(inference, training, rtol=0.0, atol=1.0e-7)

    policy._motion_index_for_test = 3
    np.testing.assert_array_equal(
        policy._get_rolling_reference_delta_command(),
        np.zeros((1, 3), dtype=np.float32),
    )
