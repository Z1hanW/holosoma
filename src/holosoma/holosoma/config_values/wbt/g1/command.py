"""Whole Body Tracking command presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.command import (
    CommandManagerCfg,
    CommandTermCfg,
    HMIGoalPoseNoiseConfig,
    HMIMotionConfig,
    MotionConfig,
    NoiseToInitialPoseConfig,
)

init_pose_config = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.1, 0.1, 0.05],
    root_ang_vel=[0.1, 0.1, 0.1],
    object_pos=[0.05, 0.05, 0.0],
)

motion_config = MotionConfig(
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz",
    body_names_to_track=[
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ],
    body_name_ref=["torso_link"],
    use_adaptive_timesteps_sampler=True,
    noise_to_initial_pose=init_pose_config,
)

motion_config_motion_tracking = replace(
    motion_config,
    num_future_steps=10,
    target_pose_type="max-coords-future-rel-with-time",
)

motion_config_w_object = replace(
    motion_config,
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz",
)

motion_config_w_object_generalist = replace(
    motion_config_w_object,
    # Keep clip-internal timestep resets uniform; across-clip curriculum is handled by
    # the clip weighting strategy.
    clip_weighting_strategy="uniform_clip",
    use_adaptive_timesteps_sampler=False,
    freeze_at_timestep_zero_prob=0.0,
)

motion_config_w_object_hybrid_stage2 = replace(
    motion_config_w_object_generalist,
    hybrid_stage2_enabled=True,
    hybrid_stage2_task_env_fraction=0.5,
    hybrid_stage2_forward_command_m=0.15,
)

motion_config_w_object_hybrid_velocity = replace(
    motion_config_w_object_generalist,
    hybrid_velocity_enabled=True,
    hybrid_velocity_task_env_fraction_start=0.0,
    hybrid_velocity_task_env_fraction_end=0.5,
    hybrid_velocity_task_env_fraction_start_iter=0,
    hybrid_velocity_task_env_fraction_end_iter=5000,
    hybrid_velocity_forward_command_mps=0.5,
    hybrid_velocity_lift_height_m=0.10,
)

motion_config_w_object_hybrid_world_velocity = replace(
    motion_config_w_object_hybrid_velocity,
    hybrid_velocity_command_frame="world",
)

motion_config_w_object_pure_rl_policy_command_after_lift = replace(
    motion_config_w_object_generalist,
    pure_rl_policy_command_after_lift_enabled=True,
    pure_rl_policy_forward_command_m=0.5,
    enable_default_pose_prepend=False,
    default_pose_prepend_duration_s=0.0,
    enable_default_pose_append=False,
    default_pose_append_duration_s=0.0,
    contact_interval_runtime_prepend_compensation=False,
)

_hmi_stage1_object_goal_noise = HMIGoalPoseNoiseConfig(
    pos_std_xyz=[0.2, 0.2, 0.001],
    pos_clip_xyz=[0.3, 0.3, 0.001],
    rpy_std=[0.001, 0.001, 0.2],
    rpy_clip=[0.001, 0.001, 0.4],
)

_hmi_stage2_object_goal_noise = HMIGoalPoseNoiseConfig(
    pos_std_xyz=[0.5, 0.5, 0.001],
    pos_clip_xyz=[1.0, 1.0, 0.001],
    rpy_std=[0.001, 0.001, 0.4],
    rpy_clip=[0.001, 0.001, 0.8],
)

_hmi_stage1_cfg = HMIMotionConfig(
    track_ratio=1.0,
    env_partition_seed=0,
    gen_start_at_timestep_zero_prob=0.2,
    object_goal_noise=_hmi_stage1_object_goal_noise,
    gen_step_zero_root_pos_std_xyz=[0.01, 0.01, 0.003],
    gen_step_zero_root_pos_clip_xyz=[0.05, 0.05, 0.01],
    gen_step_zero_root_rpy_std=[0.01, 0.01, 0.03],
    gen_step_zero_root_rpy_clip=[0.05, 0.05, 0.1],
)

_hmi_stage2_cfg = replace(
    _hmi_stage1_cfg,
    track_ratio=0.5,
    object_goal_noise=_hmi_stage2_object_goal_noise,
)

_hmi_initial_pose_noise = replace(
    init_pose_config,
    dof_pos=0.1,
    dof_vel=0.0,
    root_pos=[0.025, 0.025, 0.01],
    root_rot=[0.05, 0.05, 0.1],
    root_lin_vel=[0.25, 0.25, 0.1],
    root_ang_vel=[0.26, 0.26, 0.39],
    object_pos=[0.025, 0.025, 0.0],
)

motion_config_w_object_hmi_depth_stage1 = replace(
    motion_config_w_object,
    # Keep a checked-in, real-mesh clip with object metadata as the runnable
    # preset default. Formal launches override it with their immutable bank.
    motion_file="data_demo/sub10_largebox_032_mj_w_obj.npz",
    hmi=_hmi_stage1_cfg,
    noise_to_initial_pose=_hmi_initial_pose_noise,
    start_at_timestep_zero_prob=0.05,
    freeze_at_timestep_zero_prob=0.0,
    enable_default_pose_prepend=False,
    default_pose_prepend_duration_s=0.0,
    enable_default_pose_append=False,
    default_pose_append_duration_s=0.0,
)

motion_config_w_object_hmi_depth_stage2 = replace(
    motion_config_w_object_hmi_depth_stage1,
    hmi=_hmi_stage2_cfg,
)

g1_29dof_wbt_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config,
            },
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
    step_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
)

g1_29dof_wbt_command_motion_tracking = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_motion_tracking,
            },
        )
    },
)

g1_29dof_wbt_command_w_object = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_w_object,
            },
        )
    },
)

g1_29dof_wbt_command_w_object_generalist = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_w_object_generalist,
            },
        )
    },
)

g1_29dof_wbt_command_w_object_hybrid_stage2 = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_w_object_hybrid_stage2,
            },
        )
    },
)

g1_29dof_wbt_command_w_object_hybrid_velocity = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_w_object_hybrid_velocity,
            },
        )
    },
)

g1_29dof_wbt_command_w_object_hybrid_world_velocity = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_w_object_hybrid_world_velocity,
            },
        )
    },
)

g1_29dof_wbt_command_w_object_pure_rl_policy_command_after_lift = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_w_object_pure_rl_policy_command_after_lift,
            },
        )
    },
)

g1_29dof_wbt_command_w_object_hmi_depth_stage1 = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={"motion_config": motion_config_w_object_hmi_depth_stage1},
        )
    },
)

g1_29dof_wbt_command_w_object_hmi_depth_stage2 = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={"motion_config": motion_config_w_object_hmi_depth_stage2},
        )
    },
)

__all__ = [
    "g1_29dof_wbt_command",
    "g1_29dof_wbt_command_motion_tracking",
    "g1_29dof_wbt_command_w_object",
    "g1_29dof_wbt_command_w_object_generalist",
    "g1_29dof_wbt_command_w_object_hybrid_stage2",
    "g1_29dof_wbt_command_w_object_hybrid_velocity",
    "g1_29dof_wbt_command_w_object_hybrid_world_velocity",
    "g1_29dof_wbt_command_w_object_pure_rl_policy_command_after_lift",
    "g1_29dof_wbt_command_w_object_hmi_depth_stage1",
    "g1_29dof_wbt_command_w_object_hmi_depth_stage2",
]
