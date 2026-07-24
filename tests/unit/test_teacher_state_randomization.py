from __future__ import annotations

from types import SimpleNamespace

import pytest

from holosoma.config_types.command import NoiseToInitialPoseConfig
from holosoma.config_types.randomization import RandomizationTermCfg
from holosoma.config_values.wbt.g1.randomization import (
    g1_29dof_wbt_randomization_w_object_with_action_delay,
    g1_29dof_wbt_randomization_w_object_teacher_state_robust,
)
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.randomization.terms.locomotion import (
    MotionRelativeResetRandomizerState,
)


def _state(params: dict) -> MotionRelativeResetRandomizerState:
    cfg = RandomizationTermCfg(func="unused:test", params=params)
    return MotionRelativeResetRandomizerState(cfg, SimpleNamespace())


def test_teacher_preset_randomizes_states_without_sim2real_terms() -> None:
    cfg = g1_29dof_wbt_randomization_w_object_teacher_state_robust

    assert set(cfg.setup_terms) == {
        "motion_relative_reset_randomizer_state",
        "push_randomizer_state",
    }
    assert set(cfg.reset_terms) == {
        "push_randomizer_state",
        "randomize_push_schedule",
    }
    assert set(cfg.step_terms) == {"push_randomizer_state", "apply_pushes"}

    serialized = repr(cfg).lower()
    for forbidden in (
        "action_delay",
        "actuator_randomizer",
        "pd_gain",
        "rfi",
        "torque",
        "mass",
        "friction",
        "restitution",
        "camera",
    ):
        assert forbidden not in serialized

    state = _state(cfg.setup_terms["motion_relative_reset_randomizer_state"].params)
    assert state.noise_config.dof_pos == pytest.approx(0.20)
    assert state.noise_config.dof_vel == pytest.approx(0.35)
    assert state.noise_config.object_pos[2] == 0.0


def test_teacher_push_distribution_covers_the_student_distribution() -> None:
    teacher = g1_29dof_wbt_randomization_w_object_teacher_state_robust
    student = g1_29dof_wbt_randomization_w_object_with_action_delay

    teacher_push = teacher.setup_terms["push_randomizer_state"].params
    student_push = student.setup_terms["push_randomizer_state"].params

    # A label teacher should see at least the external-disturbance states that
    # the student visits.  Keep the two push distributions identical; the
    # teacher/student difference belongs to sim-to-real nuisance terms, not to
    # recovery-state coverage.
    assert teacher_push["push_interval_s"] == student_push["push_interval_s"] == [0.5, 2.0]
    assert teacher_push["max_push_vel"] == student_push["max_push_vel"] == [
        0.7,
        0.7,
        0.25,
        0.7,
        0.7,
        1.0,
    ]


def test_student_object_mass_and_inertia_scale_contract() -> None:
    student = g1_29dof_wbt_randomization_w_object_with_action_delay
    term = student.setup_terms["randomize_object_rigid_body_mass_inertia_scale_startup"]

    assert term.params["mass_scale_distribution_params"] == [0.33, 3.0]


def test_student_object_friction_uses_coupled_ratio_contract() -> None:
    student = g1_29dof_wbt_randomization_w_object_with_action_delay
    term = student.setup_terms["randomize_object_rigid_body_material_startup"]

    assert term.params["static_friction_range"] == [0.1, 0.7]
    assert term.params["dynamic_friction_ratio_range"] == [0.7, 0.99]
    assert "dynamic_friction_range" not in term.params


def test_motion_command_uses_final_writer_randomization_state() -> None:
    configured = _state(
        {
            "overall_noise_scale": 1.0,
            "dof_pos": 0.2,
            "dof_vel": 0.35,
            "root_pos": [0.08, 0.08, 0.025],
            "root_rot": [0.15, 0.15, 0.3],
            "root_lin_vel": [0.2, 0.2, 0.1],
            "root_ang_vel": [0.25, 0.25, 0.35],
            "object_pos": [0.08, 0.08, 0.0],
        }
    )
    manager = SimpleNamespace(
        get_state=lambda name: configured
        if name == "motion_relative_reset_randomizer_state"
        else None
    )
    command = object.__new__(MotionCommand)
    command._env = SimpleNamespace(randomization_manager=manager)
    command.init_pose_cfg = NoiseToInitialPoseConfig(dof_pos=0.1)

    assert command._effective_initial_pose_noise_config() is configured.noise_config


def test_motion_command_falls_back_to_command_noise_without_state_preset() -> None:
    command = object.__new__(MotionCommand)
    command._env = SimpleNamespace(randomization_manager=None)
    command.init_pose_cfg = NoiseToInitialPoseConfig(dof_pos=0.1, dof_vel=0.05)

    assert command._effective_initial_pose_noise_config() is command.init_pose_cfg


@pytest.mark.parametrize(
    "params",
    (
        {"dof_pos": -0.1},
        {"dof_vel": float("nan")},
        {"root_pos": [0.1, 0.1]},
        {"object_pos": [0.1, -0.1, 0.0]},
        {"action_delay": 1},
    ),
)
def test_motion_relative_reset_randomizer_rejects_invalid_config(params: dict) -> None:
    with pytest.raises(ValueError):
        _state(params)
