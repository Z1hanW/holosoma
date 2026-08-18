from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt_w_object_distill_sparse_root_cmd,
    g1_29dof_wbt_w_object_hybrid_stage2,
    g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift,
)
from holosoma.config_values.wbt.g1.reward import (
    g1_29dof_wbt_reward_w_object_generalist_tracking_no_contact,
)
from holosoma.config_values.wbt.g1.termination import (
    g1_29dof_wbt_termination_generalist,
)
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.terms import wbt as observation_terms
from holosoma.managers.reward.terms import wbt as reward_terms


class _CommandManager:
    def __init__(self, command: MotionCommand):
        self.command = command

    def get_state(self, name: str):
        assert name == "motion_command"
        return self.command


def _bare_hybrid_command(num_envs: int = 4) -> MotionCommand:
    command = MotionCommand.__new__(MotionCommand)
    command.num_envs = num_envs
    command.device = "cpu"
    command.motion_cfg = SimpleNamespace(
        hybrid_stage2_enabled=True,
        hybrid_stage2_task_env_fraction=0.5,
        hybrid_stage2_forward_command_m=0.15,
        contact_aware_sparse_root_command_mode="tracking_error",
    )
    command.motion = SimpleNamespace(has_object=True)
    command.hybrid_stage2_task_env_mask = command._build_hybrid_stage2_task_env_mask()
    command.pickup_anchor_set = torch.tensor([False, False, True, True])
    command.pickup_anchor_root_pos_w = torch.zeros((num_envs, 3))
    command.pickup_anchor_root_quat_w = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0]] * num_envs
    )
    command.pickup_anchor_object_pos_b = torch.tensor(
        [[0.4, 0.0, 0.1]] * num_envs
    )
    command.pickup_anchor_object_quat_b = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0]] * num_envs
    )
    simulator = SimpleNamespace(
        robot_root_states=torch.tensor(
            [[0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]]
            * num_envs
        )
    )
    command._env = SimpleNamespace(simulator=simulator)
    command._simulator_object_state_snapshot = torch.tensor(
        [[0.4, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]]
        * num_envs
    )
    command._simulator_object_state_snapshot_ready = True
    command.manual_control_enabled = False
    return command


def test_hybrid_assignment_is_exact_spread_and_rng_free():
    command = MotionCommand.__new__(MotionCommand)
    command.num_envs = 8
    command.device = "cpu"
    command.motion_cfg = SimpleNamespace(
        hybrid_stage2_enabled=True,
        hybrid_stage2_task_env_fraction=0.5,
        hybrid_stage2_forward_command_m=0.15,
    )

    before = torch.random.get_rng_state().clone()
    mask = command._build_hybrid_stage2_task_env_mask()
    after = torch.random.get_rng_state()

    assert mask.tolist() == [False, True, False, True, False, True, False, True]
    assert torch.equal(before, after)


def test_hybrid_assignment_is_stratified_with_two_clip_round_robin():
    command = MotionCommand.__new__(MotionCommand)
    command.num_envs = 8
    command.device = "cpu"
    command.motion_cfg = SimpleNamespace(
        hybrid_stage2_enabled=True,
        hybrid_stage2_task_env_fraction=0.5,
        hybrid_stage2_forward_command_m=0.15,
    )
    command._fixed_clip_ids = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    mask = command._build_hybrid_stage2_task_env_mask()

    assert int(mask.sum()) == 4
    assert int(mask[command._fixed_clip_ids == 0].sum()) == 2
    assert int(mask[command._fixed_clip_ids == 1].sum()) == 2


def test_hybrid_actor_contract_is_unchanged_and_task_id_is_critic_only():
    base = g1_29dof_wbt_w_object_distill_sparse_root_cmd
    hybrid = g1_29dof_wbt_w_object_hybrid_stage2

    assert hybrid.algo.config.module_dict.actor == base.algo.config.module_dict.actor
    for group_name, base_group in base.observation.groups.items():
        if group_name == "critic_obs":
            continue
        assert hybrid.observation.groups[group_name] == base_group
    assert "hybrid_stage2_task_indicator" not in hybrid.observation.groups[
        "actor_obs_root_contact_aware"
    ].terms
    assert "hybrid_stage2_task_indicator" in hybrid.observation.groups["critic_obs"].terms


def test_pure_rl_changes_only_policy_command_and_keeps_tracking_contract():
    base = g1_29dof_wbt_w_object_distill_sparse_root_cmd
    pure_rl = g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift
    motion_cfg = pure_rl.command.setup_terms["motion_command"].params["motion_config"]

    assert pure_rl.algo.config.distill.enabled is False
    assert motion_cfg.hybrid_stage2_enabled is False
    assert motion_cfg.pure_rl_policy_command_after_lift_enabled is True
    assert motion_cfg.pure_rl_policy_forward_command_m == 0.5
    assert pure_rl.algo.config.module_dict == base.algo.config.module_dict
    assert pure_rl.observation == base.observation
    assert pure_rl.reward == g1_29dof_wbt_reward_w_object_generalist_tracking_no_contact
    assert pure_rl.reward.terms["offline_contact_guidance"].weight == 0.0
    assert pure_rl.termination == g1_29dof_wbt_termination_generalist
    assert "hybrid_stage2_task_indicator" not in pure_rl.observation.groups[
        "critic_obs"
    ].terms
    serialized_terms = repr(pure_rl.reward.terms)
    assert "hybrid_stage2" not in serialized_terms
    assert "pure_rl" not in serialized_terms
    assert pure_rl.termination.terms["bad_tracking"].func.endswith(":BadTracking")


def test_hybrid_command_switches_only_task_rows_after_pickup(monkeypatch):
    command = _bare_hybrid_command()
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        command_manager=_CommandManager(command),
    )
    base = torch.tensor(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    )
    monkeypatch.setattr(
        observation_terms,
        "sparse_target_root_trajectory_command",
        lambda _env: base.clone(),
    )
    command.get_contact_aware_root_command_active_mask = lambda: torch.ones(
        4, dtype=torch.bool
    )

    result = observation_terms.sparse_target_root_trajectory_command_contact_aware(env)

    assert result.tolist() == [
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        pytest.approx([0.15, 0.0, 0.0]),
    ]


def test_policy_command_override_does_not_compute_reference_offset(monkeypatch):
    command = _bare_hybrid_command()
    command.motion_cfg.hybrid_stage2_enabled = False
    command.motion_cfg.pure_rl_policy_command_after_lift_enabled = True
    command.motion_cfg.pure_rl_policy_forward_command_m = 0.5
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        command_manager=_CommandManager(command),
    )

    def fail_if_reference_offset_is_computed(_env):
        raise AssertionError("policy command override must not compute tracking error")

    monkeypatch.setattr(
        observation_terms,
        "sparse_target_root_trajectory_command",
        fail_if_reference_offset_is_computed,
    )

    result = observation_terms.sparse_target_root_trajectory_command_contact_aware(env)

    assert result.tolist() == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ]


def test_policy_command_override_does_not_change_drop_button():
    command = _bare_hybrid_command()
    command.motion_cfg.hybrid_stage2_enabled = False
    command.motion_cfg.pure_rl_policy_command_after_lift_enabled = True
    command.motion_cfg.pure_rl_policy_forward_command_m = 0.5
    command.get_contact_aware_drop_button = lambda: torch.tensor(
        [False, True, False, True]
    )
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        command_manager=_CommandManager(command),
    )

    result = observation_terms.drop_button(env)
    assert result.tolist() == [[0.0], [1.0], [0.0], [1.0]]


def test_hybrid_task_reward_is_zero_outside_active_task_and_one_at_target():
    command = _bare_hybrid_command()
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        command_manager=_CommandManager(command),
        _reward_compute_counter=1,
    )

    reward = reward_terms.hybrid_stage2_forward_velocity_exp(
        env,
        target_velocity=0.5,
        sigma=0.25,
    )
    hold = reward_terms.hybrid_stage2_object_position_hold_exp(env, sigma=0.12)

    assert reward.tolist() == [0.0, 0.0, 0.0, 1.0]
    assert hold.tolist() == [0.0, 0.0, 0.0, 1.0]
