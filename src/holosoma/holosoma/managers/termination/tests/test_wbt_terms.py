from __future__ import annotations

from types import SimpleNamespace

import torch

from holosoma.config_types.termination import TerminationTermCfg
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.termination.terms.wbt import RobotFallenByTiltAfterIteration
from holosoma.utils.rotations import quat_from_euler_xyz


class _StubMotionCommand:
    def __init__(self, training_iteration: int):
        self._training_iteration = training_iteration


class _StubCommandManager:
    def __init__(self, motion_command: _StubMotionCommand):
        self._motion_command = motion_command

    def get_state(self, name: str):
        assert name == "motion_command"
        return self._motion_command


class _StubSimulator:
    def __init__(self, num_envs: int):
        self.robot_root_states = torch.zeros((num_envs, 13), dtype=torch.float32)


class _StubEnv:
    def __init__(self, *, base_quat: torch.Tensor, training_iteration: int, is_evaluating: bool = False):
        self.num_envs = int(base_quat.shape[0])
        self.device = "cpu"
        self.base_quat = base_quat
        self.is_evaluating = is_evaluating
        self.simulator = _StubSimulator(self.num_envs)
        self.command_manager = _StubCommandManager(_StubMotionCommand(training_iteration))


def _make_quat(*, pitch_deg: float) -> torch.Tensor:
    pitch = torch.tensor([torch.deg2rad(torch.tensor(float(pitch_deg))).item()], dtype=torch.float32)
    zeros = torch.zeros_like(pitch)
    return quat_from_euler_xyz(zeros, pitch, zeros)


def test_robot_fallen_by_tilt_is_disabled_during_dagger_until_iteration_gate(monkeypatch):
    monkeypatch.setenv("PPO_START_EPOCH", "2000")
    base_quat = torch.cat((_make_quat(pitch_deg=80.0), _make_quat(pitch_deg=0.0)), dim=0)
    cfg = TerminationTermCfg(
        func="holosoma.managers.termination.terms.wbt:RobotFallenByTiltAfterIteration",
        params={
            "max_tilt_deg": 60.0,
            "hold_steps": 1,
            "enable_after_iteration_env_var": "PPO_START_EPOCH",
        },
    )

    dagger_env = _StubEnv(base_quat=base_quat, training_iteration=500)
    term = RobotFallenByTiltAfterIteration(cfg, dagger_env)
    assert torch.equal(term(dagger_env), torch.tensor([False, False]))

    ppo_env = _StubEnv(base_quat=base_quat, training_iteration=2500)
    term = RobotFallenByTiltAfterIteration(cfg, ppo_env)
    assert torch.equal(term(ppo_env), torch.tensor([True, False]))


def test_robot_fallen_by_tilt_applies_during_evaluation_even_before_iteration_gate(monkeypatch):
    monkeypatch.setenv("PPO_START_EPOCH", "2000")
    cfg = TerminationTermCfg(
        func="holosoma.managers.termination.terms.wbt:RobotFallenByTiltAfterIteration",
        params={
            "max_tilt_deg": 60.0,
            "hold_steps": 1,
            "enable_after_iteration_env_var": "PPO_START_EPOCH",
            "apply_during_evaluation": True,
        },
    )

    eval_env = _StubEnv(base_quat=_make_quat(pitch_deg=80.0), training_iteration=0, is_evaluating=True)
    term = RobotFallenByTiltAfterIteration(cfg, eval_env)
    assert torch.equal(term(eval_env), torch.tensor([True]))


def test_sparse_goal_external_mask_is_disabled_when_curriculum_is_off():
    motion_command = object.__new__(MotionCommand)
    motion_command.num_envs = 2
    motion_command.device = "cpu"
    motion_command.motion = SimpleNamespace(has_object=True)
    motion_command._sparse_goal_curriculum_enabled = False
    motion_command.manual_goal_enabled = True
    motion_command.manual_goal_object_pos_w = torch.ones((2, 3), dtype=torch.float32)
    motion_command.manual_goal_object_rot6d_w = torch.ones((2, 6), dtype=torch.float32)
    motion_command.manual_goal_is_external = torch.ones((2,), dtype=torch.bool)
    motion_command.clip_goal_object_pos_w = torch.zeros((2, 3), dtype=torch.float32)
    motion_command.clip_goal_object_rot6d_w = torch.zeros((2, 6), dtype=torch.float32)

    assert torch.equal(
        motion_command.get_sparse_goal_external_mask(),
        torch.zeros((2,), dtype=torch.bool),
    )
