from __future__ import annotations

from types import MethodType, SimpleNamespace

import torch

from holosoma.managers.command.terms.wbt import MotionCommand, _pickup_step_and_threshold_from_rel_z


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
            all_root_states=torch.tensor([[0.0, 0.0, 0.34, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        )
    )

    applied: dict[str, torch.Tensor] = {}

    motion_command._get_clip_pickup_thresholds_by_clip = MethodType(
        lambda self: torch.tensor([0.34], dtype=torch.float32),
        motion_command,
    )
    motion_command._get_active_object_indices = MethodType(
        lambda self: torch.tensor([0], dtype=torch.long),
        motion_command,
    )
    motion_command._apply_manual_goal_world_from_command = MethodType(
        lambda self, env_ids, anchor_pos_w, anchor_quat_w: applied.update(
            {
                "env_ids": env_ids.clone(),
                "anchor_pos_w": anchor_pos_w.clone(),
                "anchor_quat_w": anchor_quat_w.clone(),
            }
        ),
        motion_command,
    )

    motion_command._update_pickup_anchor_state()

    assert bool(motion_command.pickup_anchor_set[0].item()) is True
    assert motion_command.pickup_consecutive_counter[0].item() == 5
    assert torch.equal(applied["env_ids"], torch.tensor([0], dtype=torch.long))
