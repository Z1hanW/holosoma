from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.reward.terms import wbt as reward_terms
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward_w_object_generalist_tracking_no_contact


class _CommandManager:
    def __init__(self, motion_command: MotionCommand):
        self.motion_command = motion_command

    def get_state(self, name: str):
        assert name == "motion_command"
        return self.motion_command


def _make_env(time_steps: list[int], *, valid: bool = True):
    motion_command = object.__new__(MotionCommand)
    motion_command.clip_ids = torch.zeros(len(time_steps), dtype=torch.long)
    motion_command.time_steps = torch.tensor(time_steps, dtype=torch.long)
    motion_command._adaptive_sampling_contact_window_by_clip = torch.tensor([[100, 140]], dtype=torch.long)
    motion_command._adaptive_sampling_contact_window_valid_by_clip = torch.tensor([valid], dtype=torch.bool)
    motion_command._runtime_default_pose_prepend_active = torch.zeros(len(time_steps), dtype=torch.bool)
    motion_command.metrics = {}
    env = SimpleNamespace(
        num_envs=len(time_steps),
        _reward_compute_counter=1,
        command_manager=_CommandManager(motion_command),
    )
    return env, motion_command


def test_t1_precontact_gate_has_expected_piecewise_linear_window():
    env, motion_command = _make_env([49, 50, 51, 55, 100, 105, 106, 109, 110, 111])

    gate = reward_terms._t1_precontact_smooth_gate(
        env,
        motion_command,
        lead_steps=50,
        tail_steps=10,
        ramp_steps=5,
        require_complete_contact_window=True,
    )

    assert gate.tolist() == pytest.approx([0.0, 0.0, 0.2, 1.0, 1.0, 1.0, 0.8, 0.2, 0.0, 0.0])
    assert motion_command.metrics["t1_precontact_gate_active_frac"].item() == pytest.approx(0.6)
    assert motion_command.metrics["t1_precontact_gate_mean"].item() == pytest.approx(0.42)
    assert motion_command.metrics["t1_precontact_contact_window_valid_frac"].item() == 1.0


def test_t1_precontact_gate_masks_runtime_default_pose_prepend():
    env, motion_command = _make_env([80, 80])
    motion_command._runtime_default_pose_prepend_active[0] = True

    gate = reward_terms._t1_precontact_smooth_gate(
        env,
        motion_command,
        lead_steps=50,
        tail_steps=10,
        ramp_steps=5,
        require_complete_contact_window=True,
    )

    assert gate.tolist() == pytest.approx([0.0, 1.0])


def test_t1_precontact_gate_fails_closed_on_missing_contact_window():
    env, motion_command = _make_env([80], valid=False)

    with pytest.raises(RuntimeError, match="requires a valid converted contact window"):
        reward_terms._t1_precontact_smooth_gate(
            env,
            motion_command,
            lead_steps=50,
            tail_steps=10,
            ramp_steps=5,
            require_complete_contact_window=True,
        )


def test_t1_precontact_joint_reward_multiplies_existing_score_by_gate(monkeypatch):
    env, _ = _make_env([51, 55, 110])
    monkeypatch.setattr(
        reward_terms,
        "motion_joint_position_error_exp",
        lambda *args, **kwargs: torch.tensor([0.5, 0.25, 1.0]),
    )

    reward = reward_terms.t1_precontact_motion_joint_position_error_exp(
        env,
        sigma=0.3,
        dof_names=["unused"],
        lead_steps=50,
        tail_steps=10,
        ramp_steps=5,
        require_complete_contact_window=True,
    )

    assert reward.tolist() == pytest.approx([0.1, 0.25, 0.0])


def test_generalist_reward_exposes_opt_in_t1_precontact_terms():
    terms = g1_29dof_wbt_reward_w_object_generalist_tracking_no_contact.terms
    lower = terms["t1_precontact_motion_joint_position_lower"]
    waist = terms["t1_precontact_motion_joint_position_waist"]

    assert lower.weight == 0.0
    assert waist.weight == 0.0
    assert lower.params == {
        "sigma": 0.3,
        "dof_names": [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ],
        "lead_steps": 50,
        "tail_steps": 10,
        "ramp_steps": 5,
        "require_complete_contact_window": True,
    }
    assert waist.params["sigma"] == 0.25
    assert waist.params["dof_names"] == [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ]
    assert waist.params["lead_steps"] == 50
    assert waist.params["tail_steps"] == 10
    assert waist.params["ramp_steps"] == 5
    assert waist.params["require_complete_contact_window"] is True


@pytest.mark.parametrize(
    ("lead_steps", "tail_steps", "ramp_steps"),
    [(0, 10, 5), (50, 0, 5), (50, 10, 0), (4, 10, 5), (50, 4, 5)],
)
def test_t1_precontact_gate_rejects_invalid_window_parameters(lead_steps, tail_steps, ramp_steps):
    env, motion_command = _make_env([80])

    with pytest.raises(ValueError):
        reward_terms._t1_precontact_smooth_gate(
            env,
            motion_command,
            lead_steps=lead_steps,
            tail_steps=tail_steps,
            ramp_steps=ramp_steps,
            require_complete_contact_window=True,
        )
