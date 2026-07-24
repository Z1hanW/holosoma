from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.managers.command.terms.wbt import MotionCommand


def _bare_motion_command() -> MotionCommand:
    command = MotionCommand.__new__(MotionCommand)
    command.num_envs = 1
    command.device = torch.device("cpu")
    command.motion = SimpleNamespace(has_object=True)
    command.manual_control_enabled = False
    command.manual_xy_rel = torch.full((1, 2), 9.0)
    command.manual_yaw_rel = torch.full((1, 1), 9.0)
    command.manual_drop_button_override_enabled = False
    command.manual_drop_button = torch.full((1, 1), 9.0)
    command._simulator_object_state_snapshot = torch.zeros((1, 13))
    command._simulator_object_state_snapshot[0, 2] = 0.1
    command._simulator_object_state_snapshot_ready = True
    command._env = SimpleNamespace(episode_length_buf=torch.tensor([20], dtype=torch.long))
    return command


def test_manual_forward_after_lift_holds_zero_then_replaces_command() -> None:
    command = _bare_motion_command()
    command.configure_manual_forward_after_lift(
        command_m=0.15,
        rel_z_delta_m=0.3,
        consecutive_steps=3,
    )

    assert command.manual_control_enabled is True
    assert command.manual_drop_button_override_enabled is True
    torch.testing.assert_close(command.manual_xy_rel, torch.zeros((1, 2)))
    torch.testing.assert_close(command.manual_yaw_rel, torch.zeros((1, 1)))
    torch.testing.assert_close(command.manual_drop_button, torch.zeros((1, 1)))

    command._simulator_object_state_snapshot[0, 2] = 0.399
    command._update_manual_forward_after_lift()
    assert command.get_manual_forward_after_lift_status()["consecutive_count"] == 0

    command._simulator_object_state_snapshot[0, 2] = 0.401
    for expected_count in (1, 2):
        command._update_manual_forward_after_lift()
        status = command.get_manual_forward_after_lift_status()
        assert status["phase"] == "pickup_zero"
        assert status["consecutive_count"] == expected_count
        assert status["active_forward_command_m"] == 0.0

    command._env.episode_length_buf[0] = 23
    command._update_manual_forward_after_lift()
    status = command.get_manual_forward_after_lift_status()
    assert status["phase"] == "forward"
    assert status["triggered"] is True
    assert status["trigger_episode_step"] == 23
    assert status["active_forward_command_m"] == pytest.approx(0.15)
    torch.testing.assert_close(command.manual_xy_rel, torch.tensor([[0.15, 0.0]]))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"command_m": float("nan"), "rel_z_delta_m": 0.3, "consecutive_steps": 3}, "command_m"),
        ({"command_m": 0.1, "rel_z_delta_m": 0.0, "consecutive_steps": 3}, "rel_z_delta_m"),
        ({"command_m": 0.1, "rel_z_delta_m": 0.3, "consecutive_steps": 0}, "consecutive_steps"),
        ({"command_m": 0.1, "rel_z_delta_m": 0.3, "consecutive_steps": 1.5}, "consecutive_steps"),
    ],
)
def test_manual_forward_after_lift_rejects_invalid_contract(kwargs, error: str) -> None:
    command = _bare_motion_command()
    with pytest.raises(ValueError, match=error):
        command.configure_manual_forward_after_lift(**kwargs)
