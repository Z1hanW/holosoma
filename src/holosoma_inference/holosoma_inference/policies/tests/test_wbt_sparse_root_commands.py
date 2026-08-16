from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy


def _make_policy(*, drop_button_command: float = 0.0) -> WholeBodyTrackingPolicy:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy.obs_dims = {"drop_button": 1}
    policy._drop_button_command = drop_button_command
    policy._force_zero_sparse_root_command = False
    policy._logged_zero_sparse_root_command = False
    policy._manual_sparse_root_command_offset = np.zeros((1, 3), dtype=np.float32)
    policy._joystick_sparse_root_command_offset = np.zeros((1, 3), dtype=np.float32)
    policy.logger = Mock()
    return policy


def test_full_forward_joystick_command_is_point_five() -> None:
    policy = _make_policy()
    policy.interface = SimpleNamespace(
        get_joystick_msg=lambda: SimpleNamespace(keys=0, lx=0.0, ly=1.0, rx=0.0)
    )

    policy._update_sparse_root_joystick_command()

    np.testing.assert_allclose(policy._joystick_sparse_root_command_offset, [[0.5, 0.0, 0.0]])


def test_forward_command_uses_point_zero_two_threshold() -> None:
    policy = _make_policy()
    joystick = SimpleNamespace(keys=0, lx=0.0, ly=0.021, rx=0.0)
    policy.interface = SimpleNamespace(get_joystick_msg=lambda: joystick)

    policy._update_sparse_root_joystick_command()
    np.testing.assert_allclose(policy._joystick_sparse_root_command_offset, [[0.5, 0.0, 0.0]])

    joystick.ly = 0.02
    policy._update_sparse_root_joystick_command()
    np.testing.assert_array_equal(policy._joystick_sparse_root_command_offset, np.zeros((1, 3)))


def test_keyboard_w_sets_constant_point_fifteen_forward_command() -> None:
    policy = _make_policy()

    assert policy._handle_sparse_root_keyboard_command("w")
    np.testing.assert_allclose(policy._manual_sparse_root_command_offset, [[0.15, 0.0, 0.0]])

    assert policy._handle_sparse_root_keyboard_command("w")
    np.testing.assert_allclose(policy._manual_sparse_root_command_offset, [[0.15, 0.0, 0.0]])


def test_activating_drop_clears_and_suppresses_sparse_root_commands() -> None:
    policy = _make_policy()
    policy._manual_sparse_root_command_offset[:] = [[0.1, -0.1, 0.1]]
    policy._joystick_sparse_root_command_offset[:] = [[0.5, -0.1, -0.1]]

    policy._toggle_drop_button_command()

    np.testing.assert_array_equal(policy._manual_sparse_root_command_offset, np.zeros((1, 3)))
    np.testing.assert_array_equal(policy._joystick_sparse_root_command_offset, np.zeros((1, 3)))
    np.testing.assert_array_equal(
        policy._get_sparse_target_root_trajectory_command(robot_state_data=None),
        np.zeros((1, 3), dtype=np.float32),
    )


def test_drop_ignores_joystick_until_toggled_off() -> None:
    policy = _make_policy(drop_button_command=1.0)
    policy.interface = SimpleNamespace(
        get_joystick_msg=lambda: SimpleNamespace(keys=0, lx=1.0, ly=1.0, rx=1.0)
    )

    policy._update_sparse_root_joystick_command()

    np.testing.assert_array_equal(policy._joystick_sparse_root_command_offset, np.zeros((1, 3)))
