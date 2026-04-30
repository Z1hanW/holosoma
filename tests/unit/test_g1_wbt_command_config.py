from __future__ import annotations

from holosoma.config_values.wbt.g1.command import init_pose_config


def test_g1_wbt_reset_velocity_noise_matches_success_teacher_profile() -> None:
    assert init_pose_config.root_lin_vel == [0.1, 0.1, 0.05]
    assert init_pose_config.root_ang_vel == [0.1, 0.1, 0.1]
