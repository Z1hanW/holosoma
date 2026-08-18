from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holosoma.managers.command.terms.wbt import MotionCommand, MotionLoader
from holosoma.utils.rotations import slerp


def _write_shuffled_motion(path) -> dict[str, torch.Tensor]:
    num_frames = 3
    # Include one source-only column on both axes.  MotionLoader's public
    # contract exposes exactly the configured robot, so canonicalization must
    # both reorder the retained fields and drop these extra source columns.
    num_joints = 4
    num_bodies = 4

    joint_pos = torch.arange(num_frames * num_joints, dtype=torch.float32).view(num_frames, num_joints) + 10.0
    joint_vel = torch.arange(num_frames * num_joints, dtype=torch.float32).view(num_frames, num_joints) + 30.0
    body_pos = torch.arange(num_frames * num_bodies * 3, dtype=torch.float32).view(num_frames, num_bodies, 3)
    body_quat_wxyz = (
        torch.arange(num_frames * num_bodies * 4, dtype=torch.float32).view(num_frames, num_bodies, 4) + 1.0
    )
    body_lin_vel = body_pos + 100.0
    body_ang_vel = body_pos + 200.0

    np.savez(
        path,
        joint_pos=np.concatenate((np.zeros((num_frames, 7), dtype=np.float32), joint_pos.numpy()), axis=1),
        joint_vel=np.concatenate((np.zeros((num_frames, 6), dtype=np.float32), joint_vel.numpy()), axis=1),
        body_pos_w=body_pos.numpy(),
        body_quat_w=body_quat_wxyz.numpy(),
        body_lin_vel_w=body_lin_vel.numpy(),
        body_ang_vel_w=body_ang_vel.numpy(),
        joint_names=np.asarray(["joint_b", "joint_a", "joint_c", "source_only_joint"]),
        body_names=np.asarray(["body_b", "body_c", "body_a", "source_only_body"]),
        fps=np.asarray([50.0], dtype=np.float32),
    )
    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat_wxyz[..., [1, 2, 3, 0]],
        "body_lin_vel_w": body_lin_vel,
        "body_ang_vel_w": body_ang_vel,
    }


def test_motion_loader_canonicalizes_all_robot_fields_once_without_getter_reindex(tmp_path) -> None:
    motion_path = tmp_path / "shuffled_motion.npz"
    source = _write_shuffled_motion(motion_path)
    joint_order = torch.tensor([2, 1, 0], dtype=torch.long)
    body_order = torch.tensor([2, 0, 1], dtype=torch.long)

    loader = MotionLoader(
        str(motion_path),
        robot_body_names=["body_a", "body_b", "body_c"],
        robot_joint_names=["joint_c", "joint_a", "joint_b"],
        device="cpu",
    )

    expected_by_property = {
        "joint_pos": source["joint_pos"].index_select(1, joint_order),
        "joint_vel": source["joint_vel"].index_select(1, joint_order),
        "body_pos_w": source["body_pos_w"].index_select(1, body_order),
        "body_quat_w": source["body_quat_w"].index_select(1, body_order),
        "body_lin_vel_w": source["body_lin_vel_w"].index_select(1, body_order),
        "body_ang_vel_w": source["body_ang_vel_w"].index_select(1, body_order),
    }
    backing_by_property = {
        "joint_pos": "_joint_pos",
        "joint_vel": "_joint_vel",
        "body_pos_w": "_body_pos_w",
        "body_quat_w": "_body_quat_w",
        "body_lin_vel_w": "_body_lin_vel_w",
        "body_ang_vel_w": "_body_ang_vel_w",
    }

    for property_name, expected in expected_by_property.items():
        actual = getattr(loader, property_name)
        backing = getattr(loader, backing_by_property[property_name])
        assert torch.equal(actual, expected), property_name
        # A property-time advanced-index reorder allocates different storage.
        # Canonical robot order must instead be paid once while loading.
        assert actual.data_ptr() == backing.data_ptr(), property_name
        assert actual.stride() == backing.stride(), property_name
        assert actual.is_contiguous(), property_name

    assert torch.equal(loader._joint_indexes, torch.arange(3, dtype=torch.long))
    assert torch.equal(loader._body_indexes, torch.arange(3, dtype=torch.long))
    assert loader.joint_pos.shape[1] == 3
    assert loader.body_pos_w.shape[1] == 3


def test_canonicalized_extra_columns_remain_robot_order_through_static_prepend(tmp_path) -> None:
    motion_path = tmp_path / "shuffled_motion.npz"
    _write_shuffled_motion(motion_path)
    loader = MotionLoader(
        str(motion_path),
        robot_body_names=["body_a", "body_b", "body_c"],
        robot_joint_names=["joint_c", "joint_a", "joint_b"],
        device="cpu",
    )
    original = {
        "joint_pos": loader.joint_pos.clone(),
        "joint_vel": loader.joint_vel.clone(),
        "body_pos": loader.body_pos_w.clone(),
        "body_quat": loader.body_quat_w.clone(),
        "body_lin_vel": loader.body_lin_vel_w.clone(),
        "body_ang_vel": loader.body_ang_vel_w.clone(),
    }

    command = object.__new__(MotionCommand)
    command.device = "cpu"
    command.motion = loader
    command._joint_indexes_in_motion = loader._joint_indexes
    command._body_indexes_in_motion = loader._body_indexes
    default_state = {
        "joint_pos": torch.tensor([101.0, 102.0, 103.0]),
        "joint_vel": torch.tensor([201.0, 202.0, 203.0]),
        "body_pos": torch.arange(9, dtype=torch.float32).view(3, 3) + 301.0,
        "body_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1),
        "body_lin_vel": torch.arange(9, dtype=torch.float32).view(3, 3) + 401.0,
        "body_ang_vel": torch.arange(9, dtype=torch.float32).view(3, 3) + 501.0,
    }

    command._add_transition_to_motion(default_state, num_steps=2, prepend=True)

    # The prepend contributes alpha=0 and alpha=0.5; the original canonical
    # robot-order motion starts immediately afterwards.  No source-only axis
    # may reappear while the transition is mapped and spliced.
    assert command.motion.joint_pos.shape == (5, 3)
    assert command.motion.body_pos_w.shape == (5, 3, 3)
    assert torch.equal(command.motion.joint_pos[0], default_state["joint_pos"])
    assert torch.equal(command.motion.joint_vel[0], default_state["joint_vel"])
    assert torch.equal(command.motion.body_pos_w[0], default_state["body_pos"])
    assert torch.equal(command.motion.body_quat_w[0], default_state["body_quat"])
    assert torch.equal(command.motion.body_lin_vel_w[0], default_state["body_lin_vel"])
    assert torch.equal(command.motion.body_ang_vel_w[0], default_state["body_ang_vel"])
    assert torch.equal(command.motion.joint_pos[2:], original["joint_pos"])
    assert torch.equal(command.motion.joint_vel[2:], original["joint_vel"])
    assert torch.equal(command.motion.body_pos_w[2:], original["body_pos"])
    assert torch.equal(command.motion.body_quat_w[2:], original["body_quat"])
    assert torch.equal(command.motion.body_lin_vel_w[2:], original["body_lin_vel"])
    assert torch.equal(command.motion.body_ang_vel_w[2:], original["body_ang_vel"])


_LERP_SHAPES = {
    "joint_pos": (5,),
    "joint_vel": (5,),
    "body_pos": (4, 3),
    "body_lin_vel": (4, 3),
    "body_ang_vel": (4, 3),
    "object_pos": (3,),
    "object_lin_vel": (3,),
}


def _normalized_quaternions(*shape: int, generator: torch.Generator) -> torch.Tensor:
    values = torch.randn(*shape, 4, generator=generator)
    return values / torch.linalg.vector_norm(values, dim=-1, keepdim=True)


def _runtime_prepend_command() -> MotionCommand:
    command = object.__new__(MotionCommand)
    command.device = "cpu"
    command.num_envs = 6
    command.clip_ids = torch.tensor([2, 0, 1, 2, 1, 0], dtype=torch.long)
    command._runtime_default_pose_prepend_enabled = True
    command._runtime_default_pose_prepend_steps = 10
    command._runtime_default_pose_prepend_active = torch.tensor([True, False, True, False, True, False])
    command._runtime_default_pose_prepend_step = torch.tensor([0, 7, 4, 1, 9, 3], dtype=torch.long)
    return command


def _legacy_subset_lerp(command: MotionCommand, current: torch.Tensor, defaults: torch.Tensor) -> torch.Tensor:
    active = command._runtime_default_pose_prepend_active
    assert active is not None
    env_ids = torch.nonzero(active, as_tuple=False).flatten()
    clip_ids = command.clip_ids[env_ids]
    step = command._runtime_default_pose_prepend_step
    assert step is not None
    alpha = step[env_ids].to(torch.float32) / float(command._runtime_default_pose_prepend_steps)
    alpha = alpha.view(-1, *([1] * (current.ndim - 1)))
    expected = current.clone()
    expected[env_ids] = defaults[clip_ids] + alpha * (current[env_ids] - defaults[clip_ids])
    return expected


def _legacy_subset_slerp(command: MotionCommand, current: torch.Tensor, defaults: torch.Tensor) -> torch.Tensor:
    active = command._runtime_default_pose_prepend_active
    assert active is not None
    env_ids = torch.nonzero(active, as_tuple=False).flatten()
    start = defaults[command.clip_ids[env_ids]]
    end = current[env_ids]
    step = command._runtime_default_pose_prepend_step
    assert step is not None
    alpha = step[env_ids].to(torch.float32) / float(command._runtime_default_pose_prepend_steps)
    if current.ndim == 2:
        blended = slerp(start, end, alpha.unsqueeze(-1))
    else:
        alpha_flat = alpha.unsqueeze(1).expand(-1, start.shape[1]).reshape(-1, 1)
        blended = slerp(start.reshape(-1, 4), end.reshape(-1, 4), alpha_flat).view_as(start)
    expected = current.clone()
    expected[env_ids] = blended
    return expected


def _forbid_host_mask_materialization(*_args, **_kwargs):
    raise AssertionError("runtime prepend hot path must not materialize CUDA masks in Python")


@pytest.mark.parametrize("key", tuple(_LERP_SHAPES))
def test_runtime_prepend_branchless_lerp_matches_legacy_subset_for_mixed_envs(
    key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _runtime_prepend_command()
    generator = torch.Generator().manual_seed(4100 + tuple(_LERP_SHAPES).index(key))
    tail_shape = _LERP_SHAPES[key]
    current = torch.randn(command.num_envs, *tail_shape, generator=generator)
    defaults = torch.randn(3, *tail_shape, generator=generator)
    command._runtime_default_pose_prepend_defaults = {key: defaults}
    expected = _legacy_subset_lerp(command, current, defaults)
    active = command._runtime_default_pose_prepend_active
    assert active is not None

    with monkeypatch.context() as guarded:
        guarded.setattr(torch, "nonzero", _forbid_host_mask_materialization)
        guarded.setattr(torch, "any", _forbid_host_mask_materialization)
        actual = command._blend_runtime_default_pose_prepend_lerp(current, key)

    assert torch.equal(actual[~active], current[~active])
    assert torch.allclose(actual[active], expected[active], rtol=1e-6, atol=1e-7)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("tail_shape,key", [((4,), "object_quat"), ((4, 4), "body_quat")])
def test_runtime_prepend_branchless_slerp_matches_legacy_subset_for_mixed_envs(
    tail_shape: tuple[int, ...],
    key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _runtime_prepend_command()
    generator = torch.Generator().manual_seed(8901 + len(tail_shape))
    current = _normalized_quaternions(command.num_envs, *tail_shape[:-1], generator=generator)
    defaults = _normalized_quaternions(3, *tail_shape[:-1], generator=generator)
    # Exercise slerp's shortest-arc sign correction in both ranks.
    current[2] = -current[2]
    command._runtime_default_pose_prepend_defaults = {key: defaults}
    expected = _legacy_subset_slerp(command, current, defaults)
    active = command._runtime_default_pose_prepend_active
    assert active is not None

    with monkeypatch.context() as guarded:
        guarded.setattr(torch, "nonzero", _forbid_host_mask_materialization)
        guarded.setattr(torch, "any", _forbid_host_mask_materialization)
        actual = command._blend_runtime_default_pose_prepend_quat(current, key)

    assert torch.equal(actual[~active], current[~active])
    assert torch.allclose(actual[active], expected[active], rtol=1e-6, atol=1e-7)
    assert torch.isfinite(actual).all()


def _subset_motion_command(*, aligned: bool) -> MotionCommand:
    command = object.__new__(MotionCommand)
    command.device = "cpu"
    command.num_envs = 6
    command.clip_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    command.time_steps = torch.tensor([0, 1, 3, 0, 2, 3], dtype=torch.long)
    command.tracked_body_indexes = torch.tensor([0, 2], dtype=torch.long)
    command.motion_cfg = SimpleNamespace(align_motion_to_init_yaw=aligned)
    command._clip_terrain_offsets = None
    command._terrain_row_ids = None
    command._env = SimpleNamespace(
        terrain_manager=None,
        simulator=SimpleNamespace(
            scene=SimpleNamespace(
                env_origins=torch.arange(18, dtype=torch.float32).view(6, 3) / 10.0,
            ),
        ),
    )

    generator = torch.Generator().manual_seed(74129)
    num_frames = 8
    num_bodies = 3
    num_joints = 5
    body_quat = _normalized_quaternions(num_frames, num_bodies, generator=generator)
    object_quat = _normalized_quaternions(num_frames, generator=generator)
    command.motion = SimpleNamespace(
        num_clips=2,
        clip_offsets=torch.tensor([0, 4], dtype=torch.long),
        has_object=True,
        joint_pos=torch.randn(num_frames, num_joints, generator=generator),
        joint_vel=torch.randn(num_frames, num_joints, generator=generator),
        body_pos_w=torch.randn(num_frames, num_bodies, 3, generator=generator),
        body_quat_w=body_quat,
        body_lin_vel_w=torch.randn(num_frames, num_bodies, 3, generator=generator),
        body_ang_vel_w=torch.randn(num_frames, num_bodies, 3, generator=generator),
        object_pos_w=torch.randn(num_frames, 3, generator=generator),
        object_quat_w=object_quat,
        object_lin_vel_w=torch.randn(num_frames, 3, generator=generator),
    )
    command._runtime_default_pose_prepend_enabled = True
    command._runtime_default_pose_prepend_steps = 10
    command._runtime_default_pose_prepend_active = torch.tensor(
        [True, False, True, False, True, False], dtype=torch.bool
    )
    command._runtime_default_pose_prepend_step = torch.tensor([0, 8, 3, 4, 7, 2], dtype=torch.long)
    command._runtime_default_pose_prepend_defaults = {
        "joint_pos": torch.randn(2, num_joints, generator=generator),
        "joint_vel": torch.randn(2, num_joints, generator=generator),
        "body_pos": torch.randn(2, num_bodies, 3, generator=generator),
        "body_quat": _normalized_quaternions(2, num_bodies, generator=generator),
        "body_lin_vel": torch.randn(2, num_bodies, 3, generator=generator),
        "body_ang_vel": torch.randn(2, num_bodies, 3, generator=generator),
        "object_pos": torch.randn(2, 3, generator=generator),
        "object_quat": _normalized_quaternions(2, generator=generator),
        "object_lin_vel": torch.randn(2, 3, generator=generator),
    }
    command._align_quat = _normalized_quaternions(command.num_envs, generator=generator)
    command._align_pos = torch.randn(command.num_envs, 3, generator=generator)
    return command


@pytest.mark.parametrize("aligned", [False, True])
def test_sparse_reset_motion_gathers_match_full_batch_targets(aligned: bool) -> None:
    command = _subset_motion_command(aligned=aligned)
    env_ids = torch.tensor([4, 0], dtype=torch.long)

    full_and_sparse = (
        (command.joint_pos, command._motion_joint_pos(env_ids)),
        (command.joint_vel, command._motion_joint_vel(env_ids)),
        (command.body_pos_w, command._motion_body_pos_w(env_ids)),
        (command.body_quat_w, command._motion_body_quat_w(env_ids)),
        (command.body_lin_vel_w, command._motion_body_lin_vel_w(env_ids)),
        (command.body_ang_vel_w, command._motion_body_ang_vel_w(env_ids)),
        (command.object_pos_w, command._motion_object_pos_w(env_ids)),
        (command.object_quat_w, command._motion_object_quat_w(env_ids)),
        (command.object_lin_vel_w, command._motion_object_lin_vel_w(env_ids)),
    )

    for full, sparse in full_and_sparse:
        assert sparse.shape[0] == env_ids.numel()
        assert torch.allclose(sparse, full[env_ids], rtol=1e-6, atol=1e-7)


class _ClockOnlyMotionCommand(MotionCommand):
    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._test_root_pos

    @property
    def ref_pos_w(self) -> torch.Tensor:
        return self._test_root_pos

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self._test_root_pos

    @property
    def robot_ref_pos_w(self) -> torch.Tensor:
        return self._test_root_pos

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self._test_root_quat

    @property
    def ref_quat_w(self) -> torch.Tensor:
        return self._test_root_quat

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self._test_root_quat

    @property
    def robot_ref_quat_w(self) -> torch.Tensor:
        return self._test_root_quat

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._test_body_pos

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._test_body_quat

    def _record_adaptive_timestep_exposure_before_advance(self) -> None:
        return None

    def _current_freeze_at_timestep_zero_prob(self) -> float:
        return 0.0

    def _current_clip_lengths(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        lengths = torch.full((self.num_envs,), 10_000, dtype=torch.long)
        return lengths if env_ids is None else lengths[env_ids]

    def _update_future_target_poses(self) -> None:
        return None

    def _update_pickup_anchor_state(self) -> None:
        return None

    def _update_contact_prior_state(self) -> None:
        return None


def _clock_only_command() -> _ClockOnlyMotionCommand:
    command = object.__new__(_ClockOnlyMotionCommand)
    command.device = "cpu"
    command.num_envs = 3
    command.time_steps = torch.tensor([0, 5, 9], dtype=torch.long)
    command._runtime_default_pose_prepend_enabled = True
    command._runtime_default_pose_prepend_steps = 4
    command._runtime_default_pose_prepend_active = torch.tensor([True, True, False])
    command._runtime_default_pose_prepend_step = torch.tensor([0, 2, 0], dtype=torch.long)
    command._disable_clip_end_reset = False
    command.use_adaptive_timesteps_sampler = False
    command._manual_forward_after_lift_enabled = False
    command._manual_forward_heading_lock_enabled = False
    command._env = SimpleNamespace(step_timing=None, episode_length_buf=torch.ones(3, dtype=torch.long))
    command._test_root_pos = torch.zeros(3, 3)
    command._test_root_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1)
    command._test_body_pos = torch.zeros(3, 1, 3)
    command._test_body_quat = command._test_root_quat.unsqueeze(1)
    return command


def _clock_state(command: MotionCommand) -> tuple[list[int], list[int], list[bool]]:
    step = command._runtime_default_pose_prepend_step
    active = command._runtime_default_pose_prepend_active
    assert step is not None and active is not None
    return command.time_steps.tolist(), step.tolist(), active.tolist()


def test_runtime_prepend_clock_advances_from_zero_through_deactivation_without_host_mask_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _clock_only_command()
    states = [_clock_state(command)]

    with monkeypatch.context() as guarded:
        guarded.setattr(torch, "nonzero", _forbid_host_mask_materialization)
        guarded.setattr(torch, "any", _forbid_host_mask_materialization)
        for _ in range(5):
            command.step()
            states.append(_clock_state(command))

    assert states == [
        ([0, 5, 9], [0, 2, 0], [True, True, False]),
        ([0, 5, 10], [1, 3, 0], [True, True, False]),
        ([0, 5, 11], [2, 3, 0], [True, False, False]),
        ([0, 6, 12], [3, 3, 0], [True, False, False]),
        ([0, 7, 13], [3, 3, 0], [False, False, False]),
        ([1, 8, 14], [3, 3, 0], [False, False, False]),
    ]
