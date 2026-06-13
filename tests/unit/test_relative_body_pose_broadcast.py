from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import holosoma.managers.observation.terms.wbt as obs_wbt
from holosoma.utils.rotations import (
    quat_apply,
    quat_apply_broadcast_left,
    quat_inverse,
    quat_mul,
    quat_mul_broadcast_left,
    quaternion_to_matrix,
    subtract_frame_transforms,
    yaw_quat,
)


def _random_quat(*shape: int) -> torch.Tensor:
    quat = torch.randn(*shape, dtype=torch.float32)
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)


@pytest.mark.parametrize("w_last", [True, False])
def test_quat_mul_broadcast_left_matches_repeat(w_last: bool) -> None:
    torch.manual_seed(1)
    num_envs = 6
    num_bodies = 7
    left = _random_quat(num_envs, 4)
    right = _random_quat(num_envs, num_bodies, 4)

    expected = quat_mul(left[:, None, :].repeat(1, num_bodies, 1), right, w_last=w_last)
    actual = quat_mul_broadcast_left(left, right, w_last=w_last)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("w_last", [True, False])
def test_quat_apply_broadcast_left_matches_repeat(w_last: bool) -> None:
    torch.manual_seed(2)
    num_envs = 6
    num_bodies = 7
    quat = _random_quat(num_envs, 4)
    vec = torch.randn(num_envs, num_bodies, 3, dtype=torch.float32)

    expected = quat_apply(quat[:, None, :].repeat(1, num_bodies, 1), vec, w_last=w_last)
    actual = quat_apply_broadcast_left(quat, vec, w_last=w_last)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_relative_body_pose_formula_matches_repeat_path() -> None:
    torch.manual_seed(3)
    num_envs = 8
    num_bodies = 5
    use_root = torch.randint(0, 2, (num_envs, 1), dtype=torch.float32)

    root_pos_w = torch.randn(num_envs, 3)
    ref_pos_w_raw = torch.randn(num_envs, 3)
    robot_root_pos_w = torch.randn(num_envs, 3)
    robot_ref_pos_w_raw = torch.randn(num_envs, 3)
    root_quat_w = _random_quat(num_envs, 4)
    ref_quat_w_raw = _random_quat(num_envs, 4)
    robot_root_quat_w = _random_quat(num_envs, 4)
    robot_ref_quat_w_raw = _random_quat(num_envs, 4)
    body_pos_w = torch.randn(num_envs, num_bodies, 3)
    body_quat_w = _random_quat(num_envs, num_bodies, 4)

    ref_pos_w = root_pos_w * use_root + ref_pos_w_raw * (1.0 - use_root)
    ref_quat_w = root_quat_w * use_root + ref_quat_w_raw * (1.0 - use_root)
    robot_ref_pos_w = robot_root_pos_w * use_root + robot_ref_pos_w_raw * (1.0 - use_root)
    robot_ref_quat_w = robot_root_quat_w * use_root + robot_ref_quat_w_raw * (1.0 - use_root)

    ref_pos_w_repeat = ref_pos_w[:, None, :].repeat(1, num_bodies, 1)
    ref_quat_w_repeat = ref_quat_w[:, None, :].repeat(1, num_bodies, 1)
    robot_ref_pos_w_repeat = robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1)
    robot_ref_quat_w_repeat = robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1)
    legacy_delta_quat_w = yaw_quat(
        quat_mul(robot_ref_quat_w_repeat, quat_inverse(ref_quat_w_repeat, w_last=True), w_last=True),
        w_last=True,
    )
    expected_quat = quat_mul(legacy_delta_quat_w, body_quat_w, w_last=True)
    legacy_delta_pos_w_height = ref_pos_w_repeat - robot_ref_pos_w_repeat
    legacy_delta_pos_w_height[..., :2] = 0.0
    expected_pos = (
        robot_ref_pos_w_repeat
        + legacy_delta_pos_w_height
        + quat_apply(legacy_delta_quat_w, body_pos_w - ref_pos_w_repeat, w_last=True)
    )

    delta_quat_w = yaw_quat(
        quat_mul(robot_ref_quat_w, quat_inverse(ref_quat_w, w_last=True), w_last=True),
        w_last=True,
    )
    actual_quat = quat_mul_broadcast_left(delta_quat_w, body_quat_w, w_last=True)
    delta_pos_w_height = ref_pos_w - robot_ref_pos_w
    delta_pos_w_height[..., :2] = 0.0
    actual_pos = (
        robot_ref_pos_w[:, None, :]
        + delta_pos_w_height[:, None, :]
        + quat_apply_broadcast_left(delta_quat_w, body_pos_w - ref_pos_w[:, None, :], w_last=True)
    )

    torch.testing.assert_close(actual_quat, expected_quat, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_pos, expected_pos, rtol=1e-6, atol=1e-6)


def test_robot_body_observation_formula_matches_subtract_frame_repeat(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(4)
    num_envs = 5
    num_bodies = 4
    motion_command = SimpleNamespace(
        motion_cfg=SimpleNamespace(body_names_to_track=[f"body_{idx}" for idx in range(num_bodies)]),
        robot_ref_pos_w=torch.randn(num_envs, 3),
        robot_ref_quat_w=_random_quat(num_envs, 4),
        robot_body_pos_w=torch.randn(num_envs, num_bodies, 3),
        robot_body_quat_w=_random_quat(num_envs, num_bodies, 4),
    )
    env = SimpleNamespace(num_envs=num_envs)
    monkeypatch.setattr(obs_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)

    expected_pos, expected_quat = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
        w_last=True,
    )

    actual_pos = obs_wbt.robot_body_pos_b(env).view(num_envs, num_bodies, 3)
    expected_ori = quaternion_to_matrix(expected_quat, w_last=True)[..., :2].reshape(num_envs, -1)
    actual_ori = obs_wbt.robot_body_ori_b(env)

    torch.testing.assert_close(actual_pos, expected_pos, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_ori, expected_ori, rtol=0.0, atol=0.0)
