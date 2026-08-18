from __future__ import annotations

from types import SimpleNamespace

import torch

from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt_w_object_distill_sparse_root_cmd,
    g1_29dof_wbt_w_object_hybrid_velocity,
    g1_29dof_wbt_w_object_hybrid_world_velocity,
    g1_29dof_wbt_w_object_policy_world_root_error,
    g1_29dof_wbt_w_object_policy_world_velocity,
    g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift,
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


def _motion_cfg(**overrides):
    values = {
        "hybrid_velocity_enabled": True,
        "hybrid_velocity_command_frame": "heading",
        "hybrid_stage2_enabled": False,
        "pure_rl_policy_command_after_lift_enabled": False,
        "hybrid_velocity_task_env_fraction_start": 0.0,
        "hybrid_velocity_task_env_fraction_end": 0.5,
        "hybrid_velocity_task_env_fraction_start_iter": 0,
        "hybrid_velocity_task_env_fraction_end_iter": 100,
        "hybrid_velocity_forward_command_mps": 0.5,
        "hybrid_velocity_lift_height_m": 0.10,
        "align_motion_to_init_yaw": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _bare_velocity_command(num_envs: int = 3) -> MotionCommand:
    command = MotionCommand.__new__(MotionCommand)
    command.num_envs = num_envs
    command.device = "cpu"
    command.motion_cfg = _motion_cfg()
    command.motion = SimpleNamespace(has_object=True)
    command._training_iteration = 0
    command._env = SimpleNamespace(
        is_evaluating=False,
        simulator=SimpleNamespace(
            robot_root_states=torch.tensor(
                [
                    [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.2],
                    [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ),
    )
    body_lin_vel = torch.tensor(
        [
            [[0.2, 0.1, 0.0]],
            [[0.3, -0.1, 0.0]],
            [[0.4, 0.2, 0.0]],
        ]
    )
    body_ang_vel = torch.tensor(
        [
            [[0.0, 0.0, 0.2]],
            [[0.0, 0.0, -0.1]],
            [[0.0, 0.0, 0.3]],
        ]
    )
    command._raw_motion_body_lin_vel_w = lambda env_ids=None: (
        body_lin_vel if env_ids is None else body_lin_vel[env_ids]
    )
    command._raw_motion_body_ang_vel_w = lambda env_ids=None: (
        body_ang_vel if env_ids is None else body_ang_vel[env_ids]
    )
    command.hybrid_velocity_task_env_mask = torch.tensor([False, True, True])
    command.pickup_anchor_set = torch.tensor([False, False, True])
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
    command.hybrid_velocity_object_z_baseline = torch.tensor([0.8, 0.8, 0.8])
    command._simulator_object_state_snapshot = torch.tensor(
        [
            [0.4, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.4, 0.0, 0.85, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.4, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    command._simulator_object_state_snapshot_ready = True
    return command


def test_new_preset_is_isolated_and_uses_one_actor_goal_semantics():
    base = g1_29dof_wbt_w_object_distill_sparse_root_cmd
    old_pure_rl = g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift
    hybrid = g1_29dof_wbt_w_object_hybrid_velocity
    old_cfg = old_pure_rl.command.setup_terms["motion_command"].params["motion_config"]
    new_cfg = hybrid.command.setup_terms["motion_command"].params["motion_config"]

    assert old_cfg.hybrid_velocity_enabled is False
    assert new_cfg.hybrid_velocity_enabled is True
    assert new_cfg.hybrid_stage2_enabled is False
    assert new_cfg.pure_rl_policy_command_after_lift_enabled is False
    assert hybrid.algo.config.module_dict.actor == base.algo.config.module_dict.actor

    actor_terms = hybrid.observation.groups["actor_obs_root_contact_aware"].terms
    assert list(actor_terms) == ["hybrid_velocity_command"]
    assert actor_terms["hybrid_velocity_command"].func.endswith(":hybrid_velocity_command")
    assert "hybrid_velocity_task_indicator" not in actor_terms

    critic_terms = hybrid.observation.groups["critic_obs"].terms
    assert critic_terms["motion_command"].func.endswith(":hybrid_velocity_masked_motion_command")
    assert critic_terms["motion_ref_pos_b"].func.endswith(":hybrid_velocity_masked_motion_ref_pos_b")
    assert critic_terms["motion_ref_ori_b"].func.endswith(":hybrid_velocity_masked_motion_ref_ori_b")
    assert critic_terms["obj_target_pos_b"].func.endswith(":hybrid_velocity_masked_obj_target_pos_b")
    assert critic_terms["obj_target_ori_b"].func.endswith(":hybrid_velocity_masked_obj_target_ori_b")
    assert "hybrid_velocity_command" in critic_terms
    assert "hybrid_velocity_task_indicator" in critic_terms
    assert hybrid.reward.terms["offline_contact_guidance"].weight == 0.0
    assert hybrid.termination.terms["bad_tracking"].func.endswith(":HybridVelocityBadTracking")


def test_task_fraction_is_stratified_monotonic_rng_free_and_reset_scoped():
    command = MotionCommand.__new__(MotionCommand)
    command.num_envs = 16
    command.device = "cpu"
    command.motion_cfg = _motion_cfg()
    command.motion = SimpleNamespace(has_object=True)
    command._env = SimpleNamespace(is_evaluating=False)
    command._training_iteration = 0
    command._fixed_clip_ids = torch.tensor([0, 1] * 8)
    command.hybrid_velocity_task_env_mask = torch.zeros(16, dtype=torch.bool)
    command._hybrid_velocity_task_priority = torch.ones(16)

    before = torch.random.get_rng_state().clone()
    command._configure_hybrid_velocity_task_assignment()
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    assert not torch.any(command.hybrid_velocity_task_env_mask)

    command._training_iteration = 50
    command._refresh_hybrid_velocity_task_env_mask(torch.arange(8))
    assert int(command.hybrid_velocity_task_env_mask[:8].sum()) == 2
    assert int(command.hybrid_velocity_task_env_mask[8:].sum()) == 0

    command._refresh_hybrid_velocity_task_env_mask()
    quarter_mask = command.hybrid_velocity_task_env_mask.clone()
    assert int(quarter_mask.sum()) == 4
    assert int(quarter_mask[command._fixed_clip_ids == 0].sum()) == 2
    assert int(quarter_mask[command._fixed_clip_ids == 1].sum()) == 2

    command._training_iteration = 100
    command._refresh_hybrid_velocity_task_env_mask()
    half_mask = command.hybrid_velocity_task_env_mask
    assert int(half_mask.sum()) == 8
    assert torch.all(half_mask[quarter_mask])
    assert int(half_mask[command._fixed_clip_ids == 0].sum()) == 4
    assert int(half_mask[command._fixed_clip_ids == 1].sum()) == 4


def test_velocity_command_uses_npz_root_velocity_and_task_latch():
    command = _bare_velocity_command()
    result = command.get_hybrid_velocity_command()
    assert torch.allclose(
        result,
        torch.tensor(
            [
                [0.2, 0.1, 0.2],
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ]
        ),
    )


def test_world_velocity_command_does_not_rotate_with_robot_heading():
    command = _bare_velocity_command()
    half_angle = torch.tensor(torch.pi / 4.0)
    command._env.simulator.robot_root_states[0, 3:7] = torch.tensor(
        [0.0, 0.0, torch.sin(half_angle), torch.cos(half_angle)]
    )

    heading_command = command.get_hybrid_velocity_command()[0]
    command.motion_cfg.hybrid_velocity_command_frame = "world"
    world_command = command.get_hybrid_velocity_command()[0]

    assert torch.allclose(heading_command, torch.tensor([0.1, -0.2, 0.2]), atol=1.0e-6)
    assert torch.allclose(world_command, torch.tensor([0.2, 0.1, 0.2]), atol=1.0e-6)


def test_manual_command_override_respects_each_policy_command_contract():
    command = _bare_velocity_command()
    command.manual_control_enabled = True
    command.manual_xy_rel = torch.tensor([[0.05, 0.0], [0.15, 0.0], [0.50, 0.0]])
    command.manual_yaw_rel = torch.zeros((3, 1))
    command._manual_forward_after_lift_command_semantics = "robot_heading_velocity_mps"

    assert torch.equal(
        command.get_hybrid_velocity_command(),
        torch.tensor([[0.05, 0.0, 0.0], [0.15, 0.0, 0.0], [0.50, 0.0, 0.0]]),
    )

    env = SimpleNamespace(num_envs=3, device="cpu", command_manager=_CommandManager(command))
    command.motion_cfg.hybrid_velocity_command_frame = "world"
    command._manual_forward_after_lift_command_semantics = "world_velocity_mps"
    assert torch.equal(
        observation_terms.target_root_world_velocity_command(env),
        torch.tensor([[0.05, 0.0, 0.0], [0.15, 0.0, 0.0], [0.50, 0.0, 0.0]]),
    )

    command._manual_forward_after_lift_command_semantics = "world_root_error_m"
    assert torch.equal(
        observation_terms.target_root_world_xy_yaw_error_command(env),
        torch.tensor([[0.05, 0.0, 0.0], [0.15, 0.0, 0.0], [0.50, 0.0, 0.0]]),
    )


def test_policy_world_velocity_preserves_per_frame_turns_and_reversals():
    command = _bare_velocity_command()
    body_lin_vel = torch.tensor(
        [
            [[0.7, 0.3, 0.0]],
            [[-0.4, 0.2, 0.0]],
            [[0.1, -0.8, 0.0]],
        ]
    )
    body_ang_vel = torch.tensor(
        [
            [[0.0, 0.0, 0.6]],
            [[0.0, 0.0, -0.5]],
            [[0.0, 0.0, 0.1]],
        ]
    )
    command._raw_motion_body_lin_vel_w = lambda env_ids=None: (
        body_lin_vel if env_ids is None else body_lin_vel[env_ids]
    )
    command._raw_motion_body_ang_vel_w = lambda env_ids=None: (
        body_ang_vel if env_ids is None else body_ang_vel[env_ids]
    )
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        command_manager=_CommandManager(command),
    )

    result = observation_terms.target_root_world_velocity_command(env)

    assert torch.equal(
        result,
        torch.tensor(
            [
                [0.7, 0.3, 0.6],
                [-0.4, 0.2, -0.5],
                [0.1, -0.8, 0.1],
            ]
        ),
    )


def test_world_root_error_keeps_global_xy_and_wraps_yaw():
    target_yaw = torch.tensor(-torch.pi + 0.05)
    robot_yaw = torch.tensor(torch.pi - 0.05)
    target_half = target_yaw / 2.0
    robot_half = robot_yaw / 2.0
    command = SimpleNamespace(
        root_pos_w=torch.tensor([[3.0, 5.0, 0.8]]),
        robot_root_pos_w=torch.tensor([[2.0, 3.0, 0.8]]),
        root_quat_w=torch.tensor(
            [[0.0, 0.0, torch.sin(target_half), torch.cos(target_half)]]
        ),
        robot_root_quat_w=torch.tensor(
            [[0.0, 0.0, torch.sin(robot_half), torch.cos(robot_half)]]
        ),
    )

    rel_xy, rel_yaw = observation_terms._root_world_xy_yaw_error_command(command)

    assert torch.equal(rel_xy, torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(rel_yaw, torch.tensor([[0.1]]), atol=1.0e-6)


def test_new_world_presets_are_pure_policy_contracts_and_keep_critic_reference():
    world_velocity = g1_29dof_wbt_w_object_policy_world_velocity
    world_error = g1_29dof_wbt_w_object_policy_world_root_error
    hybrid_world = g1_29dof_wbt_w_object_hybrid_world_velocity

    assert world_velocity.algo.config.module_dict.actor.input_dim == [
        "actor_obs_world_velocity_command",
        "actor_obs_drop_button",
        "actor_obs_proprio_with_actions_no_linvel",
    ]
    assert world_error.algo.config.module_dict.actor.input_dim[0] == (
        "actor_obs_world_root_error_command"
    )
    assert world_velocity.reward.terms["offline_contact_guidance"].weight == 0.0
    assert world_velocity.termination.terms["bad_tracking"].func.endswith(":BadTracking")
    assert world_velocity.observation.groups["critic_obs"] == (
        g1_29dof_wbt_w_object_distill_sparse_root_cmd.observation.groups["critic_obs"]
    )

    hybrid_cfg = hybrid_world.command.setup_terms["motion_command"].params[
        "motion_config"
    ]
    assert hybrid_cfg.hybrid_velocity_command_frame == "world"
    assert hybrid_world.algo.config.module_dict.actor.input_dim[0] == (
        "actor_obs_hybrid_world_velocity_command"
    )
    old_cfg = g1_29dof_wbt_w_object_hybrid_velocity.command.setup_terms[
        "motion_command"
    ].params["motion_config"]
    assert old_cfg.hybrid_velocity_command_frame == "heading"


def test_world_hybrid_task_rewards_use_world_axes_not_pickup_heading():
    command = _bare_velocity_command()
    command.motion_cfg.hybrid_velocity_command_frame = "world"
    half_angle = torch.tensor(torch.pi / 4.0)
    command.pickup_anchor_root_quat_w[2] = torch.tensor(
        [0.0, 0.0, torch.sin(half_angle), torch.cos(half_angle)]
    )
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        command_manager=_CommandManager(command),
        _reward_compute_counter=1,
    )

    forward = reward_terms.hybrid_velocity_forward_velocity_exp(env, sigma=0.25)
    lateral = reward_terms.hybrid_velocity_lateral_velocity_exp(env, sigma=0.20)

    assert torch.allclose(forward, torch.tensor([0.0, 1.0, 1.0]), atol=1.0e-6)
    assert torch.allclose(lateral, torch.tensor([0.0, 1.0, 1.0]), atol=1.0e-6)


def test_task_reference_mask_and_drop_button_are_task_only():
    command = _bare_velocity_command()
    command.get_contact_aware_drop_button = lambda: torch.tensor([True, True, True])
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        command_manager=_CommandManager(command),
    )
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    masked = observation_terms._mask_hybrid_velocity_reference_rows(env, value)
    button = observation_terms.drop_button(env)

    assert masked.tolist() == [[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]]
    assert button.tolist() == [[1.0], [0.0], [0.0]]


def test_task_rewards_use_same_command_and_world_lift_progress():
    command = _bare_velocity_command()
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        command_manager=_CommandManager(command),
        _reward_compute_counter=1,
    )

    lift = reward_terms.hybrid_velocity_lift_progress(env)
    forward = reward_terms.hybrid_velocity_forward_velocity_exp(env, sigma=0.25)
    hold = reward_terms.hybrid_velocity_object_position_hold_exp(env, sigma=0.12)

    assert torch.allclose(lift, torch.tensor([0.0, 0.5, 1.0]), atol=1.0e-6)
    assert torch.allclose(forward, torch.tensor([0.0, 1.0, 1.0]), atol=1.0e-6)
    assert torch.allclose(hold, torch.tensor([0.0, 0.0, 1.0]), atol=1.0e-6)
