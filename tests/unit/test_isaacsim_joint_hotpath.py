from __future__ import annotations

import ast
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import torch

from holosoma.managers.action.terms.joint_control import JointPositionActionTerm
from holosoma.simulator.isaacsim.joint_hotpath import (
    build_ideal_pd_actuator_groups,
    cached_dof_selector,
    select_dof_write_batch,
)


class _RecordingActuatorCfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _resolve_exact_dict(values: dict[str, float], joint_names: list[str]) -> list[float]:
    """Mirror IsaacLab's strict re.fullmatch-based per-joint resolution."""

    resolved = []
    for joint_name in joint_names:
        matches = [
            value
            for pattern, value in values.items()
            if re.fullmatch(pattern, joint_name)
        ]
        assert len(matches) == 1, (joint_name, matches)
        resolved.append(matches[0])
    return resolved


def test_single_actuator_group_preserves_every_per_joint_property_and_order():
    # Include regex metacharacters so this also proves that "exact name" does
    # not accidentally broaden IsaacLab's regular-expression matching.
    joint_names = ["hip_joint", "knee.joint", "wrist[0]", "wrist0"]
    effort_limits = [88.0, 91.0, 23.5, 17.0]
    velocity_limits = [12.0, 13.0, 6.5, 5.0]
    armatures = [0.01, 0.02, 0.03, 0.04]
    frictions = [0.1, 0.2, 0.3, 0.4]

    groups = build_ideal_pd_actuator_groups(
        actuator_cfg_type=_RecordingActuatorCfg,
        joint_names=joint_names,
        effort_limits=effort_limits,
        velocity_limits=velocity_limits,
        armatures=armatures,
        frictions=frictions,
    )

    assert list(groups) == ["all_joints"]
    cfg = groups["all_joints"]
    assert cfg.joint_names_expr == [re.escape(name) for name in joint_names]
    assert _resolve_exact_dict(cfg.effort_limit, joint_names) == effort_limits
    assert _resolve_exact_dict(cfg.velocity_limit, joint_names) == velocity_limits
    assert _resolve_exact_dict(cfg.armature, joint_names) == armatures
    assert _resolve_exact_dict(cfg.friction, joint_names) == frictions
    assert cfg.stiffness == 0
    assert cfg.damping == 0


@pytest.mark.parametrize(
    "field,values",
    [
        ("effort_limits", [1.0]),
        ("velocity_limits", [1.0]),
        ("armatures", [1.0]),
        ("frictions", [1.0]),
    ],
)
def test_actuator_group_rejects_incomplete_per_joint_properties(field, values):
    kwargs = {
        "effort_limits": [1.0, 2.0],
        "velocity_limits": [3.0, 4.0],
        "armatures": [5.0, 6.0],
        "frictions": [7.0, 8.0],
    }
    kwargs[field] = values

    with pytest.raises(ValueError, match=field):
        build_ideal_pd_actuator_groups(
            actuator_cfg_type=_RecordingActuatorCfg,
            joint_names=["j0", "j1"],
            **kwargs,
        )


def test_actuator_group_rejects_duplicate_joint_names():
    with pytest.raises(ValueError, match="unique"):
        build_ideal_pd_actuator_groups(
            actuator_cfg_type=_RecordingActuatorCfg,
            joint_names=["same", "same"],
            effort_limits=[1.0, 2.0],
            velocity_limits=[3.0, 4.0],
            armatures=[5.0, 6.0],
            frictions=[7.0, 8.0],
        )


def test_cached_dof_selector_uses_slice_only_for_identity_mapping():
    dof_state = torch.arange(12).reshape(3, 4)
    identity = [0, 1, 2, 3]
    identity_selector = cached_dof_selector(identity, total_num_dofs=4)
    assert identity_selector == slice(None)
    identity_state = dof_state[:, identity_selector]
    assert identity_state.data_ptr() == dof_state.data_ptr()
    assert torch.equal(identity_state, dof_state)

    permutation = [0, 2, 1, 3]
    permutation_selector = cached_dof_selector(permutation, total_num_dofs=4)
    assert permutation_selector is permutation
    permutation_state = dof_state[:, permutation_selector]
    assert permutation_state.data_ptr() != dof_state.data_ptr()
    assert torch.equal(permutation_state, dof_state[:, [0, 2, 1, 3]])


def test_cached_dof_selector_does_not_expand_a_leading_identity_subset():
    dof_state = torch.arange(12).reshape(3, 4)
    leading_subset = [0, 1]

    selector = cached_dof_selector(leading_subset, total_num_dofs=4)

    assert selector is leading_subset
    assert torch.equal(dof_state[:, selector], dof_state[:, :2])


def test_cached_dof_selector_rejects_negative_total_dof_count():
    with pytest.raises(ValueError, match="total_num_dofs"):
        cached_dof_selector([], total_num_dofs=-1)


def test_current_dof_write_selects_rows_without_assembling_full_state():
    current_pos = torch.arange(15, dtype=torch.float32).view(5, 3)
    current_vel = current_pos + 100.0
    env_ids = torch.tensor([4, 1], dtype=torch.long)

    pos, vel = select_dof_write_batch(
        current_pos,
        current_vel,
        env_ids,
        None,
        num_envs=5,
    )

    assert torch.equal(pos, current_pos[env_ids])
    assert torch.equal(vel, current_vel[env_ids])
    assert pos.shape == (2, 3)
    assert vel.shape == (2, 3)


def test_explicit_dof_write_accepts_full_or_compact_batches():
    current_pos = torch.zeros(5, 3)
    current_vel = torch.zeros(5, 3)
    env_ids = torch.tensor([4, 1], dtype=torch.long)
    full = torch.arange(30, dtype=torch.float32).view(5, 3, 2)
    compact = full[env_ids].clone()

    full_pos, full_vel = select_dof_write_batch(
        current_pos,
        current_vel,
        env_ids,
        full,
        num_envs=5,
    )
    compact_pos, compact_vel = select_dof_write_batch(
        current_pos,
        current_vel,
        env_ids,
        compact,
        num_envs=5,
    )

    assert torch.equal(compact_pos, full_pos)
    assert torch.equal(compact_vel, full_vel)


@pytest.mark.parametrize(
    "bad_state",
    [
        torch.zeros(2, 3),
        torch.zeros(2, 3, 3),
        torch.zeros(2, 4, 2),
        torch.zeros(3, 3, 2),
    ],
)
def test_explicit_dof_write_rejects_ambiguous_or_invalid_shapes(bad_state):
    with pytest.raises(ValueError):
        select_dof_write_batch(
            torch.zeros(5, 3),
            torch.zeros(5, 3),
            torch.tensor([4, 1], dtype=torch.long),
            bad_state,
            num_envs=5,
        )


class _FakeSimulator:
    def __init__(self):
        self.dof_vel = torch.tensor([[4.0, 5.0]])
        self.applied_torques = None

    def apply_torques_at_dof(self, torques):
        self.applied_torques = torques.clone()


@pytest.mark.parametrize(
    ("control_type", "expected"),
    [("P", False), ("T", False), ("V", True)],
)
def test_previous_dof_velocity_gate_is_initialized_from_control_type(control_type, expected):
    control = SimpleNamespace(
        control_type=control_type,
        stiffness={"joint": 10.0},
        damping={"joint": 1.0},
        integral={},
        action_scales_by_effort_limit_over_p_gain=False,
        action_scale=0.25,
    )
    robot_config = SimpleNamespace(
        control=control,
        init_state=SimpleNamespace(default_joint_angles={"joint": 0.0}),
    )
    env = SimpleNamespace(
        num_dof=1,
        num_envs=2,
        device="cpu",
        dof_names=["joint"],
        robot_config=robot_config,
    )

    term = JointPositionActionTerm(SimpleNamespace(), env)

    assert term._uses_prev_dof_vel is expected


@pytest.mark.parametrize("uses_history", [False, True])
def test_previous_dof_velocity_copy_is_gated_by_controller_need(uses_history):
    simulator = _FakeSimulator()
    term = JointPositionActionTerm.__new__(JointPositionActionTerm)
    term.env = SimpleNamespace(simulator=simulator)
    term.torques = torch.zeros(1, 2)
    term._actions_after_delay = torch.tensor([[1.0, 2.0]])
    term._compute_torques = lambda actions: actions + 10.0
    term._uses_prev_dof_vel = uses_history
    term._prev_dof_vel = torch.tensor([[-1.0, -2.0]])

    term.apply_actions()

    assert torch.equal(simulator.applied_torques, torch.tensor([[11.0, 12.0]]))
    expected_history = simulator.dof_vel if uses_history else torch.tensor([[-1.0, -2.0]])
    assert torch.equal(term._prev_dof_vel, expected_history)


@pytest.mark.parametrize("uses_history", [False, True])
def test_previous_dof_velocity_reset_is_gated_by_controller_need(uses_history):
    term = JointPositionActionTerm.__new__(JointPositionActionTerm)
    term.env = SimpleNamespace(_randomize_ctrl_delay=False)
    term._raw_actions = None
    term._processed_actions = None
    term.action_queue = None
    term.torques = torch.ones(2, 2)
    term._uses_prev_dof_vel = uses_history
    term._prev_dof_vel = torch.full((2, 2), 7.0)

    term.reset()

    assert torch.count_nonzero(term.torques) == 0
    expected_history = torch.zeros(2, 2) if uses_history else torch.full((2, 2), 7.0)
    assert torch.equal(term._prev_dof_vel, expected_history)


def test_isaacsim_hot_methods_use_the_cached_selector_contract():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "src/holosoma/holosoma/simulator/isaacsim/isaacsim.py"
    )
    tree = ast.parse(source_path.read_text())
    isaacsim_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "IsaacSim"
    )
    methods = {
        node.name: ast.get_source_segment(source_path.read_text(), node)
        for node in isaacsim_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert methods["_setup_scene"].count("build_ideal_pd_actuator_groups(") == 1
    for method_name in ("refresh_sim_tensors", "apply_torques_at_dof", "simulate_at_each_physics_step"):
        assert "self._dof_selector" in methods[method_name]
