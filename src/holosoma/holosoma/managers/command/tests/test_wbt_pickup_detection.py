from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch

from holosoma.managers.command.terms.wbt import MotionCommand, _pickup_step_and_threshold_from_rel_z


_PICKUP_STATE_NAMES = (
    "pickup_anchor_set",
    "pickup_anchor_root_pos_w",
    "pickup_anchor_root_quat_w",
    "pickup_object_rel_z_baseline",
    "pickup_consecutive_counter",
)


class _PickupAnchorCommand(MotionCommand):
    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self._test_robot_root_pos_w

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self._test_robot_root_quat_w

    @property
    def simulator_object_pos_w(self) -> torch.Tensor:
        return self._test_simulator_object_pos_w

    def _get_clip_pickup_thresholds_by_clip(self) -> torch.Tensor:
        return self._test_clip_pickup_thresholds

    def _get_clip_pickup_steps_by_clip(self) -> torch.Tensor:
        return self._test_clip_pickup_steps


def _make_pickup_anchor_command(
    *,
    anchor_set: list[bool],
    counters: list[int],
    relative_object_heights: list[float],
) -> _PickupAnchorCommand:
    num_envs = len(anchor_set)
    command = object.__new__(_PickupAnchorCommand)
    command.device = "cpu"
    command.num_envs = num_envs
    command.motion = SimpleNamespace(has_object=True)
    command.clip_ids = torch.tensor([index % 2 for index in range(num_envs)], dtype=torch.long)
    command.time_steps = torch.zeros(num_envs, dtype=torch.long)
    command.pickup_anchor_set = torch.tensor(anchor_set, dtype=torch.bool)
    command.pickup_consecutive_counter = torch.tensor(counters, dtype=torch.long)
    command.pickup_anchor_root_pos_w = (
        torch.arange(num_envs * 3, dtype=torch.float32).view(num_envs, 3) + 100.0
    )
    command.pickup_anchor_root_quat_w = (
        torch.arange(num_envs * 4, dtype=torch.float32).view(num_envs, 4) + 200.0
    )
    command.pickup_object_rel_z_baseline = torch.arange(num_envs, dtype=torch.float32) + 0.2
    command._test_robot_root_pos_w = torch.arange(num_envs * 3, dtype=torch.float32).view(num_envs, 3) + 10.0
    command._test_robot_root_quat_w = (
        torch.arange(num_envs * 4, dtype=torch.float32).view(num_envs, 4) + 20.0
    )
    relative_heights = torch.tensor(relative_object_heights, dtype=torch.float32)
    command._test_simulator_object_pos_w = command._test_robot_root_pos_w.clone()
    command._test_simulator_object_pos_w[:, 2] += relative_heights
    command._test_clip_pickup_thresholds = torch.tensor([0.30, 0.40], dtype=torch.float32)
    command._test_clip_pickup_steps = torch.tensor([3, 7], dtype=torch.long)
    return command


def _clone_pickup_anchor_command(command: _PickupAnchorCommand) -> _PickupAnchorCommand:
    cloned = _make_pickup_anchor_command(
        anchor_set=command.pickup_anchor_set.tolist(),
        counters=command.pickup_consecutive_counter.tolist(),
        relative_object_heights=(
            command._test_simulator_object_pos_w[:, 2] - command._test_robot_root_pos_w[:, 2]
        ).tolist(),
    )
    cloned.clip_ids.copy_(command.clip_ids)
    cloned.time_steps.copy_(command.time_steps)
    for name in _PICKUP_STATE_NAMES:
        getattr(cloned, name).copy_(getattr(command, name))
    cloned._test_robot_root_pos_w.copy_(command._test_robot_root_pos_w)
    cloned._test_robot_root_quat_w.copy_(command._test_robot_root_quat_w)
    cloned._test_simulator_object_pos_w.copy_(command._test_simulator_object_pos_w)
    cloned._test_clip_pickup_thresholds.copy_(command._test_clip_pickup_thresholds)
    cloned._test_clip_pickup_steps.copy_(command._test_clip_pickup_steps)
    return cloned


def _legacy_update_pickup_anchor_state(command: _PickupAnchorCommand) -> None:
    current_rel_z = command.simulator_object_pos_w[:, 2] - command.robot_root_pos_w[:, 2]
    clip_pickup_thresholds = command._get_clip_pickup_thresholds_by_clip()[command.clip_ids]
    lifted = current_rel_z >= clip_pickup_thresholds
    command.pickup_consecutive_counter = torch.where(
        lifted,
        command.pickup_consecutive_counter + 1,
        torch.zeros_like(command.pickup_consecutive_counter),
    )
    newly_picked = (~command.pickup_anchor_set) & (command.pickup_consecutive_counter >= 5)
    if not newly_picked.any():
        return
    command.pickup_anchor_set[newly_picked] = True
    command.pickup_anchor_root_pos_w[newly_picked] = command.robot_root_pos_w[newly_picked]
    command.pickup_anchor_root_quat_w[newly_picked] = command.robot_root_quat_w[newly_picked]


def _legacy_reset_pickup_anchor_state(
    command: _PickupAnchorCommand,
    env_ids: torch.Tensor,
    *,
    root_pos_w: torch.Tensor | None = None,
    root_quat_w: torch.Tensor | None = None,
    object_pos_w: torch.Tensor | None = None,
) -> None:
    command.pickup_anchor_set[env_ids] = False
    command.pickup_consecutive_counter[env_ids] = 0
    command.pickup_anchor_root_pos_w[env_ids] = 0.0
    command.pickup_anchor_root_quat_w[env_ids] = 0.0
    command.pickup_anchor_root_quat_w[env_ids, 3] = 1.0
    command.pickup_object_rel_z_baseline[env_ids] = 0.0
    if root_pos_w is None or root_quat_w is None or object_pos_w is None:
        return
    command.pickup_anchor_root_pos_w[env_ids] = root_pos_w
    command.pickup_anchor_root_quat_w[env_ids] = root_quat_w
    command.pickup_object_rel_z_baseline[env_ids] = object_pos_w[:, 2] - root_pos_w[:, 2]

    clip_pickup_steps = command._get_clip_pickup_steps_by_clip()[command.clip_ids[env_ids]]
    already_picked_mask = command.time_steps[env_ids] >= clip_pickup_steps
    if not torch.any(already_picked_mask):
        return
    prime_env_ids = env_ids[already_picked_mask]
    command.pickup_anchor_set[prime_env_ids] = True
    command.pickup_consecutive_counter[prime_env_ids] = 5
    command.pickup_anchor_root_pos_w[prime_env_ids] = root_pos_w[already_picked_mask]
    command.pickup_anchor_root_quat_w[prime_env_ids] = root_quat_w[already_picked_mask]


def _assert_pickup_state_equal(actual: _PickupAnchorCommand, expected: _PickupAnchorCommand) -> None:
    for name in _PICKUP_STATE_NAMES:
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


def _forbid_pickup_host_mask_materialization(*_args, **_kwargs):
    raise AssertionError("pickup-anchor hot path must not materialize a device mask in Python")


def test_pickup_step_and_threshold_uses_max_of_absolute_and_ratio_thresholds():
    rel_z = torch.tensor([0.20, 0.20, 0.35, 0.50, 0.60], dtype=torch.float32)

    pickup_step, pickup_threshold = _pickup_step_and_threshold_from_rel_z(
        rel_z,
        lift_height_threshold=0.10,
        lift_ratio_threshold=0.35,
        consecutive_steps=2,
    )

    assert torch.isclose(pickup_threshold, torch.tensor(0.34))
    assert pickup_step == 2


def test_runtime_pickup_anchor_state_uses_clip_threshold_not_reset_baseline():
    motion_command = object.__new__(MotionCommand)
    motion_command.device = "cpu"
    motion_command.num_envs = 1
    motion_command.motion = SimpleNamespace(has_object=True)
    motion_command.pickup_anchor_set = torch.tensor([False])
    motion_command.pickup_anchor_root_pos_w = torch.zeros((1, 3), dtype=torch.float32)
    motion_command.pickup_anchor_root_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    motion_command.pickup_object_rel_z_baseline = torch.tensor([0.25], dtype=torch.float32)
    motion_command.pickup_consecutive_counter = torch.tensor([4], dtype=torch.long)
    motion_command.clip_ids = torch.tensor([0], dtype=torch.long)
    motion_command._env = SimpleNamespace(
        simulator=SimpleNamespace(
            robot_root_states=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
            all_root_states=torch.tensor(
                [[0.0, 0.0, 0.34, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
            ),
        )
    )

    motion_command._get_clip_pickup_thresholds_by_clip = MethodType(
        lambda self: torch.tensor([0.34], dtype=torch.float32),
        motion_command,
    )
    motion_command._get_active_object_indices = MethodType(
        lambda self: torch.tensor([0], dtype=torch.long),
        motion_command,
    )

    motion_command._update_pickup_anchor_state()

    assert bool(motion_command.pickup_anchor_set[0].item()) is True
    assert motion_command.pickup_consecutive_counter[0].item() == 5


@pytest.mark.parametrize(
    "anchor_set,counters,relative_object_heights",
    [
        pytest.param(
            [False, False, False, False],
            [0, 4, 2, 1],
            [0.10, 0.20, 0.29, 0.39],
            id="never",
        ),
        pytest.param(
            [False, False, False, False],
            [4, 1, 2, 3],
            [0.30, 0.20, 0.10, 0.20],
            id="one",
        ),
        pytest.param(
            [False, False, False, False],
            [4, 4, 5, 7],
            [0.30, 0.40, 0.31, 0.41],
            id="multiple",
        ),
        pytest.param(
            [True, False, True, False],
            [8, 4, 2, 0],
            [0.50, 0.40, 0.31, 0.10],
            id="already-set",
        ),
    ],
)
def test_pickup_anchor_device_only_update_matches_legacy_reference_bitwise(
    anchor_set: list[bool],
    counters: list[int],
    relative_object_heights: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = _make_pickup_anchor_command(
        anchor_set=anchor_set,
        counters=counters,
        relative_object_heights=relative_object_heights,
    )
    expected = _clone_pickup_anchor_command(actual)
    _legacy_update_pickup_anchor_state(expected)

    state_buffer_ptrs = {
        name: getattr(actual, name).data_ptr()
        for name in (
            "pickup_anchor_set",
            "pickup_anchor_root_pos_w",
            "pickup_anchor_root_quat_w",
            "pickup_consecutive_counter",
        )
    }
    with monkeypatch.context() as guarded:
        guarded.setattr(torch, "any", _forbid_pickup_host_mask_materialization)
        guarded.setattr(torch.Tensor, "any", _forbid_pickup_host_mask_materialization)
        actual._update_pickup_anchor_state()

    _assert_pickup_state_equal(actual, expected)
    assert {
        name: getattr(actual, name).data_ptr()
        for name in state_buffer_ptrs
    } == state_buffer_ptrs


def test_pickup_anchor_device_only_reset_and_followup_sequence_matches_legacy_reference_bitwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = _make_pickup_anchor_command(
        anchor_set=[True, True, False, True],
        counters=[8, 6, 3, 9],
        relative_object_heights=[0.50, 0.50, 0.10, 0.50],
    )
    actual.time_steps.copy_(torch.tensor([8, 7, 5, 1], dtype=torch.long))
    expected = _clone_pickup_anchor_command(actual)

    env_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    reset_root_pos = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=torch.float32,
    )
    reset_root_quat = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.1, 0.2, 0.3, 0.9],
            [0.2, 0.3, 0.4, 0.8],
        ],
        dtype=torch.float32,
    )
    reset_object_pos = reset_root_pos.clone()
    reset_object_pos[:, 2] += torch.tensor([0.45, 0.35, 0.20], dtype=torch.float32)
    _legacy_reset_pickup_anchor_state(
        expected,
        env_ids,
        root_pos_w=reset_root_pos,
        root_quat_w=reset_root_quat,
        object_pos_w=reset_object_pos,
    )
    with monkeypatch.context() as guarded:
        guarded.setattr(torch, "any", _forbid_pickup_host_mask_materialization)
        guarded.setattr(torch.Tensor, "any", _forbid_pickup_host_mask_materialization)
        actual._reset_pickup_anchor_state(
            env_ids,
            root_pos_w=reset_root_pos,
            root_quat_w=reset_root_quat,
            object_pos_w=reset_object_pos,
        )
    _assert_pickup_state_equal(actual, expected)

    relative_height_sequence = (
        [0.10, 0.50, 0.35, 0.20],
        [0.31, 0.50, 0.35, 0.45],
        [0.31, 0.50, 0.35, 0.45],
        [0.31, 0.50, 0.35, 0.45],
        [0.31, 0.50, 0.35, 0.45],
        [0.31, 0.50, 0.35, 0.45],
    )
    for step, relative_heights in enumerate(relative_height_sequence):
        root_delta = float(step + 1)
        for command in (actual, expected):
            command._test_robot_root_pos_w.add_(root_delta)
            command._test_robot_root_quat_w.add_(root_delta * 0.01)
            command._test_simulator_object_pos_w.copy_(command._test_robot_root_pos_w)
            command._test_simulator_object_pos_w[:, 2] += torch.tensor(relative_heights, dtype=torch.float32)

        _legacy_update_pickup_anchor_state(expected)
        with monkeypatch.context() as guarded:
            guarded.setattr(torch, "any", _forbid_pickup_host_mask_materialization)
            guarded.setattr(torch.Tensor, "any", _forbid_pickup_host_mask_materialization)
            actual._update_pickup_anchor_state()
        _assert_pickup_state_equal(actual, expected)
