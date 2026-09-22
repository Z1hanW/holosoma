from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holosoma.config_types.command import MotionConfig
from holosoma.agents.ppo.ppo import PPO
from holosoma.managers.command.terms.wbt import MotionCommand, motion_transition_contract_sha256
from holosoma.utils.simulator_config import SimulatorType
from holosoma_inference.policies.wbt import _apply_transition_segment_np
from holosoma_inference.utils.policy_contract import motion_transition_contract_from_metadata


def _command(duration: float = 0.2, *, lengths=(3, 5)) -> MotionCommand:
    command = object.__new__(MotionCommand)
    command.device = "cpu"
    command.num_envs = len(lengths)
    command.num_future_steps = 0
    command.multi_clip = len(lengths) > 1
    command._rank_local_shard_metadata = {"global_clip_count": 137}
    command.motion_cfg = SimpleNamespace(runtime_default_pose_append_duration_s=duration)
    command._env = SimpleNamespace(
        dt=0.02,
        simulator=SimpleNamespace(get_simulator_type=lambda: SimulatorType.ISAACSIM),
    )
    command.clip_ids = torch.arange(len(lengths))
    command.time_steps = torch.tensor(lengths) - 1
    total = sum(lengths)
    joint = torch.arange(total * 2, dtype=torch.float32).view(total, 2) + 1
    pos = torch.arange(total * 3, dtype=torch.float32).view(total, 1, 3)
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(total, 1, 1)
    command.motion = SimpleNamespace(
        num_clips=len(lengths), clip_ids=[f"clip_{i}" for i in range(len(lengths))],
        clip_lengths=torch.tensor(lengths),
        clip_offsets=torch.tensor([0, *np.cumsum(lengths[:-1])]),
        time_step_total=total, has_object=True,
        joint_pos=joint, joint_vel=torch.ones_like(joint),
        body_pos_w=pos, body_quat_w=quat,
        body_lin_vel_w=torch.ones_like(pos), body_ang_vel_w=torch.ones_like(pos),
        object_pos_w=pos[:, 0] + 100, object_quat_w=quat[:, 0],
        object_lin_vel_w=torch.zeros(total, 3),
        precomputed_root_command=torch.ones(total, 3),
        precomputed_root_command_phase=torch.ones(total, dtype=torch.uint8),
    )
    command._runtime_default_pose_prepend_enabled = False
    command._runtime_default_pose_prepend_steps = 0
    command._disable_clip_end_reset = True
    command.precomputed_turn_then_forward_enabled = lambda: True
    command.pickup_anchor_set = torch.ones(len(lengths), dtype=torch.bool)
    command._termination_owns_clip_rollover = lambda: False

    def default(idx):
        p = pos[idx].clone()
        p[:, 2] = 0.78
        return {
            "joint_pos": torch.zeros(2), "joint_vel": torch.zeros(2),
            "body_pos": p,
            "body_quat": torch.tensor([[0.0, 0.0, 0.6, 0.8]]),
            "body_lin_vel": torch.zeros(1, 3), "body_ang_vel": torch.zeros(1, 3),
            "object_pos": command.motion.object_pos_w[idx].clone(),
            "object_quat": command.motion.object_quat_w[idx].clone(),
            "object_lin_vel": torch.zeros(3),
        }

    command._build_default_pose_state_robot_order = default
    command._configure_runtime_default_pose_append()
    return command


def test_ten_appended_steps_end_exactly_at_default_without_mutating_bank():
    command = _command()
    source = command.motion.joint_pos.clone()
    source_lengths = command.motion.clip_lengths.clone()
    ends = command.motion.clip_offsets + source_lengths - 1
    starts = source[ends]
    for step in range(11):
        command.time_steps = source_lengths - 1 + step
        torch.testing.assert_close(command._raw_motion_joint_pos(), starts * (1 - step / 10))
        assert command.motion_end_mask().tolist() == [step == 10] * 2
        assert command._get_motion_indices(command.time_steps).tolist() == ends.tolist()
        torch.testing.assert_close(command._raw_motion_object_pos_w(), command.motion.object_pos_w[ends])
    assert torch.equal(command._raw_motion_joint_pos(), torch.zeros_like(starts))
    assert torch.equal(source, command.motion.joint_pos)
    assert torch.equal(source_lengths, command.motion.clip_lengths)
    assert command.current_clip_lengths.tolist() == [13, 15]
    assert command._valid_start_counts().tolist() == [1, 3]


def test_subset_and_future_gathers_keep_each_clips_own_endpoint():
    command = _command()
    command.time_steps = torch.tensor([7, 14])
    torch.testing.assert_close(command._raw_motion_joint_pos(torch.tensor([1])), torch.zeros(1, 2))
    future = torch.tensor([[1, 2, 7, 12], [3, 4, 9, 14]])
    indices = command._get_motion_indices(future)
    assert indices.tolist() == [[1, 2, 2, 2], [6, 7, 7, 7]]
    values = command.motion.body_pos_w[indices]
    actual = command._blend_runtime_default_pose_append(values, "body_pos", steps=future)
    assert torch.equal(actual[:, :2], values[:, :2])
    targets = command._runtime_default_pose_append_defaults["body_pos"]
    torch.testing.assert_close(actual[:, 2], (values[:, 2] + targets) / 2)
    assert torch.equal(actual[:, 3], targets)
    quats = command._blend_runtime_default_pose_append(
        command.motion.body_quat_w[indices], "body_quat", steps=future, quaternion=True
    )
    torch.testing.assert_close(torch.linalg.vector_norm(quats, dim=-1), torch.ones(2, 4, 1))
    assert torch.equal(quats[:, -1], command._runtime_default_pose_append_defaults["body_quat"])


def test_append_clock_clamps_only_after_endpoint_when_reset_disabled():
    command = _command()
    command.time_steps = torch.tensor([12, 14])
    command._handle_clip_rollover()
    assert command.time_steps.tolist() == [12, 14]
    command.time_steps += 1
    command._handle_clip_rollover()
    assert command.time_steps.tolist() == [12, 14]


def test_precomputed_commands_preserved_in_source_and_zero_during_return():
    command = _command()
    assert torch.equal(command.get_precomputed_turn_then_forward_command(), torch.ones(2, 3))
    command.time_steps += 1
    assert torch.count_nonzero(command.get_precomputed_turn_then_forward_command()) == 0
    assert torch.count_nonzero(command.get_precomputed_turn_then_forward_phase()) == 0


def test_historical_config_and_termination_are_unchanged():
    assert MotionConfig(motion_file="test.npz", body_name_ref=["pelvis"], body_names_to_track=["pelvis"]).runtime_default_pose_append_duration_s == 0.0
    command = _command(0.0)
    command.motion_cfg.enable_default_pose_append = True
    command.motion_cfg.default_pose_append_duration_s = 2.0
    command._configure_runtime_default_pose_append()
    assert command._runtime_default_pose_append_steps == 0
    assert command.current_clip_lengths.tolist() == [3, 5]
    command.time_steps = torch.tensor([1, 2])
    assert command.motion_end_mask().tolist() == [True, False]


def test_single_clip_shard_uses_global_runtime_append():
    command = _command(lengths=(3,))
    command.time_steps[:] = 12
    assert command._runtime_default_pose_append_steps == 10
    assert torch.equal(command._raw_motion_joint_pos(), torch.zeros(1, 2))


@pytest.mark.parametrize("duration", [True, -0.1, float("nan"), float("inf"), 0.01, 100.0])
def test_invalid_duration_fails_instead_of_skipping(duration):
    with pytest.raises(ValueError, match="append"):
        _command(duration)


def test_runtime_contract_is_accepted_by_deployment_and_bound_to_digest():
    command = _command()
    contract = command.get_motion_transition_contract()
    assert contract["append"] == {"implementation": "runtime_blend", "applied": True, "steps": 10}
    metadata = {"motion_transition_contract": contract,
                "motion_transition_contract_sha256": motion_transition_contract_sha256(contract)}
    assert motion_transition_contract_from_metadata(metadata, required=True) == contract
    contract["append"]["steps"] = 11
    with pytest.raises(ValueError, match="SHA-256"):
        motion_transition_contract_from_metadata(metadata, required=True)


def test_exact_resume_rejects_new_append_on_historical_checkpoint():
    old = _command(0.0).get_motion_transition_contract()
    checkpoint = {"motion_transition_contract": old,
                  "motion_transition_contract_sha256": motion_transition_contract_sha256(old)}
    ppo = object.__new__(PPO)
    with pytest.raises(ValueError, match="differs from the live runtime"):
        ppo._validate_checkpoint_motion_transition_contract(
            checkpoint, live_contract=_command().get_motion_transition_contract(),
            compare_live=True, operation="Full resume"
        )


def test_no_runtime_append_fallback_on_unsupported_backend_or_source():
    command = _command(0.0, lengths=(3,))
    command.motion_cfg.runtime_default_pose_append_duration_s = 0.2
    command._env.simulator.get_simulator_type = lambda: SimulatorType.MUJOCO
    with pytest.raises(ValueError, match="IsaacSim"):
        command._configure_runtime_default_pose_append()
    command._env.simulator.get_simulator_type = lambda: SimulatorType.ISAACSIM
    command._rank_local_shard_metadata = None
    with pytest.raises(ValueError, match="global multi-clip"):
        command._configure_runtime_default_pose_append()


def test_torch_runtime_matches_numpy_deployment_materialization():
    command = _command(lengths=(3,))
    end = 2
    start = command._raw_motion_joint_pos()[0].numpy()
    default = command._runtime_default_pose_append_defaults
    state = {
        "joint_pos": start, "joint_vel": np.ones(2, dtype=np.float32),
        "root_pos": command.motion.body_pos_w[end, 0].numpy(),
        "ref_pos": command.motion.body_pos_w[end, 0].numpy(),
        "root_quat": np.array([1, 0, 0, 0], dtype=np.float32),
        "ref_quat": np.array([1, 0, 0, 0], dtype=np.float32),
    }
    target = dict(state, joint_pos=default["joint_pos"][0].numpy(),
                  joint_vel=default["joint_vel"][0].numpy(),
                  root_pos=default["body_pos"][0, 0].numpy(),
                  ref_pos=default["body_pos"][0, 0].numpy(),
                  root_quat=default["body_quat"][0, 0, [3, 0, 1, 2]].numpy(),
                  ref_quat=default["body_quat"][0, 0, [3, 0, 1, 2]].numpy())
    motion = {name: value[None] for name, value in {
        "joint_pos": state["joint_pos"], "joint_vel": state["joint_vel"],
        "root_pos_w": state["root_pos"], "ref_pos_w": state["ref_pos"],
        "root_quat_w": state["root_quat"], "ref_quat_w": state["ref_quat"],
    }.items()}
    _apply_transition_segment_np(motion, start_state=state, target_state=target,
                                 num_steps=10, prepend=False, drop_first=True, drop_last=False)
    for step in range(11):
        command.time_steps[:] = end + step
        np.testing.assert_allclose(command._raw_motion_joint_pos()[0].numpy(), motion["joint_pos"][step], atol=1e-6)
        np.testing.assert_allclose(command._raw_motion_body_pos_w()[0, 0].numpy(), motion["root_pos_w"][step], atol=1e-6)
        np.testing.assert_allclose(command._raw_motion_body_quat_w()[0, 0, [3, 0, 1, 2]].numpy(), motion["root_quat_w"][step], atol=1e-6)
