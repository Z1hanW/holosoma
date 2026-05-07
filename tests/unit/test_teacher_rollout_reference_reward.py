from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import holosoma.managers.reward.terms.wbt as reward_wbt


class _DummyMotion:
    def __init__(self, clip_ids: list[str]):
        self.clip_ids = clip_ids
        self.num_clips = len(clip_ids)
        self.has_object = True


class _DummyMotionCommand:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.num_envs = 2
        self.motion = _DummyMotion(["box_10"])
        self.motion_cfg = SimpleNamespace(
            body_names_to_track=["pelvis", "torso_link"],
            body_name_ref=["torso_link"],
        )
        self.ref_body_index = 1
        self.clip_ids = torch.tensor([0, 0], dtype=torch.long)
        self.time_steps = torch.tensor([1, 1], dtype=torch.long)
        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

        self.robot_body_pos_w = torch.tensor(
            [
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.2]],
                [[0.3, 0.0, 1.0], [0.0, 0.0, 1.2]],
            ],
            dtype=torch.float32,
        )
        self.robot_body_quat_w = identity_quat.view(1, 1, 4).repeat(self.num_envs, 2, 1)
        self.robot_body_lin_vel_w = torch.zeros((self.num_envs, 2, 3), dtype=torch.float32)
        self.robot_body_ang_vel_w = torch.zeros((self.num_envs, 2, 3), dtype=torch.float32)

        self.robot_root_pos_w = self.robot_body_pos_w[:, 0, :]
        self.robot_root_quat_w = identity_quat.view(1, 4).repeat(self.num_envs, 1)
        self.robot_ref_pos_w = self.robot_body_pos_w[:, self.ref_body_index, :]
        self.robot_ref_quat_w = identity_quat.view(1, 4).repeat(self.num_envs, 1)

        self.simulator_object_pos_w = torch.tensor(
            [
                [1.0, 0.0, 0.5],
                [1.25, 0.0, 0.5],
            ],
            dtype=torch.float32,
        )
        self.simulator_object_quat_w = identity_quat.view(1, 4).repeat(self.num_envs, 1)

    def _get_env_offsets(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        offsets = torch.zeros((self.num_envs, 3), dtype=torch.float32)
        if env_ids is None:
            return offsets
        return offsets.index_select(0, env_ids.to(dtype=torch.long))


def _write_teacher_rollout_reference(export_root: Path) -> None:
    clip_dir = export_root / "clips" / "0000_box_10"
    clip_dir.mkdir(parents=True, exist_ok=True)

    valid_steps = np.asarray([True, True, True], dtype=np.bool_)
    body_pos_local = np.asarray(
        [
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.2]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.2]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.2]],
        ],
        dtype=np.float32,
    )
    identity_quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    body_quat_w = np.broadcast_to(identity_quat, (3, 2, 4)).copy()
    zeros_body = np.zeros((3, 2, 3), dtype=np.float32)
    zeros_vec = np.zeros((3, 3), dtype=np.float32)
    ref_pos_local = np.asarray(
        [
            [0.0, 0.0, 1.2],
            [0.0, 0.0, 1.2],
            [0.0, 0.0, 1.2],
        ],
        dtype=np.float32,
    )
    root_pos_local = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    object_pos_local = np.asarray(
        [
            [1.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
        ],
        dtype=np.float32,
    )
    object_quat_w = np.broadcast_to(identity_quat, (3, 4)).copy()

    np.savez_compressed(
        clip_dir / "teacher_rollout_reference.npz",
        clip_id=np.asarray("box_10"),
        clip_index=np.asarray(0, dtype=np.int32),
        tracked_body_names=np.asarray(["pelvis", "torso_link"]),
        ref_body_name=np.asarray("torso_link"),
        valid_steps=valid_steps,
        body_pos_local=body_pos_local,
        body_quat_w=body_quat_w,
        body_lin_vel_w=zeros_body,
        body_ang_vel_w=zeros_body,
        ref_pos_local=ref_pos_local,
        ref_quat_w=np.broadcast_to(identity_quat, (3, 4)).copy(),
        ref_lin_vel_w=zeros_vec,
        ref_ang_vel_w=zeros_vec,
        root_pos_local=root_pos_local,
        root_quat_w=np.broadcast_to(identity_quat, (3, 4)).copy(),
        root_lin_vel_w=zeros_vec,
        root_ang_vel_w=zeros_vec,
        object_pos_local=object_pos_local,
        object_quat_w=object_quat_w,
        object_lin_vel_w=zeros_vec,
        object_ang_vel_w=zeros_vec,
    )


def _build_env(tmp_path: Path) -> tuple[SimpleNamespace, _DummyMotionCommand, Path]:
    export_root = tmp_path / "teacher_rollout_export"
    _write_teacher_rollout_reference(export_root)
    motion_command = _DummyMotionCommand()
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        episode_length_buf=torch.ones((2,), dtype=torch.long),
        command_manager=SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None),
    )
    return env, motion_command, export_root


def test_teacher_rollout_object_reference_reward_uses_exported_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, motion_command, export_root = _build_env(tmp_path)
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)

    reward = reward_wbt.teacher_rollout_object_global_ref_position_error_exp(
        env,
        sigma=0.1,
        rollout_reference_root=str(export_root),
    )

    assert reward.shape == (2,)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    assert reward[1].item() < 0.01


def test_teacher_rollout_global_ref_reward_uses_exported_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, motion_command, export_root = _build_env(tmp_path)
    motion_command.robot_ref_pos_w = torch.tensor(
        [
            [0.0, 0.0, 1.2],
            [0.2, 0.0, 1.2],
        ],
        dtype=torch.float32,
    )
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)

    reward = reward_wbt.teacher_rollout_global_ref_position_error_exp(
        env,
        sigma=0.1,
        rollout_reference_root=str(export_root),
    )

    assert reward.shape == (2,)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    assert reward[1].item() < 0.05


def test_teacher_rollout_relative_body_reward_uses_exported_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, motion_command, export_root = _build_env(tmp_path)
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)

    reward = reward_wbt.teacher_rollout_relative_body_position_error_exp(
        env,
        sigma=0.1,
        rollout_reference_root=str(export_root),
    )

    assert reward.shape == (2,)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    assert reward[1].item() < 0.2


def test_teacher_rollout_global_body_lin_vel_reward_uses_exported_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, motion_command, export_root = _build_env(tmp_path)
    motion_command.robot_body_lin_vel_w[1, 0, 0] = 1.0
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)

    reward = reward_wbt.teacher_rollout_global_body_lin_vel(
        env,
        sigma=0.1,
        rollout_reference_root=str(export_root),
    )

    assert reward.shape == (2,)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    assert reward[1].item() < 0.1
