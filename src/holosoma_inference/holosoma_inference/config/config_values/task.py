"""Default task configurations for holosoma_inference."""

from __future__ import annotations

from holosoma_inference.config.config_types.task import TaskConfig

# Locomotion task
locomotion = TaskConfig(
    model_path="",  # Must be provided by user
    rl_rate=50,
    policy_action_scale=0.25,
    use_phase=True,
    gait_period=1.0,
    desired_base_height=0.75,
    residual_upper_body_action=False,
    domain_id=0,
    interface="lo",
    use_joystick=False,
    joystick_type="xbox",
    joystick_device=0,
    sim_clock_port=5555,
    auto_start_policy=False,
    auto_start_motion=False,
    auto_start_motion_clip=False,
    auto_start_stiff_hold_sec=1.0,
    auto_start_stiff_pose_tolerance=0.12,
    auto_start_stiff_max_wait_sec=4.0,
    defer_policy_start_until_valid_state=False,
    motion_file=None,
    apply_training_motion_transitions=False,
    use_sim_state=False,
    sim_state_port=5557,
    sim_control_port=5559,
    use_zmq_lowcmd=False,
    sim_object_name="object",
    use_root_reference_at_clip_start=False,
    prefer_sim_ref_from_sim_state=False,
    use_ros=False,
    wandb_download_dir="/tmp",
)

# Whole-body tracking task
wbt = TaskConfig(
    model_path="",  # Must be provided by user
    rl_rate=50,
    policy_action_scale=1.0,
    use_phase=False,
    gait_period=1.0,
    desired_base_height=0.75,
    residual_upper_body_action=False,
    domain_id=0,
    interface="lo",
    use_joystick=False,
    joystick_type="xbox",
    joystick_device=0,
    use_sim_time=False,
    sim_clock_port=5555,
    auto_start_policy=False,
    auto_start_motion=False,
    auto_start_motion_clip=False,
    auto_start_stiff_hold_sec=1.0,
    auto_start_stiff_pose_tolerance=0.12,
    auto_start_stiff_max_wait_sec=4.0,
    defer_policy_start_until_valid_state=False,
    motion_file=None,
    apply_training_motion_transitions=False,
    use_sim_state=False,
    sim_state_port=5557,
    sim_control_port=5559,
    use_zmq_lowcmd=False,
    sim_object_name="object",
    use_root_reference_at_clip_start=False,
    prefer_sim_ref_from_sim_state=False,
    use_ros=False,
    wandb_download_dir="/tmp",
)

DEFAULTS = {
    "locomotion": locomotion,
    "wbt": wbt,
}
