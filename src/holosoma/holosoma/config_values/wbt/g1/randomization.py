"""Whole Body Tracking randomization presets for the G1 robot."""

from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg

CAMERA_RAYCAST_MESH_ALLOWLIST = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

robot_state_dr_at_setup = {
    "randomize_robot_rigid_body_material_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_robot_rigid_body_material_startup",
        params={
            "static_friction_range": [0.3, 1.6],
            "dynamic_friction_range": [0.3, 1.2],
            "restitution_range": [0.0, 0.5],
        },
    ),
    "randomize_base_com_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_base_com_startup",
        params={
            "base_com_range": {"x": [-0.025, 0.025], "y": [-0.05, 0.05], "z": [-0.05, 0.05]},
            "enabled": True,
        },
    ),
    "setup_dof_pos_bias": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:setup_dof_pos_bias",
        params={
            "dof_pos_bias_range": [-0.01, 0.01],
            "enabled": True,
        },
    ),
    "setup_camera_raycast_randomization": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:setup_camera_raycast_randomization",
        params={
            "enabled": True,
            "mesh_allowlist": CAMERA_RAYCAST_MESH_ALLOWLIST,
        },
    ),
}

object_state_dr_at_setup = {
    "randomize_object_rigid_body_material_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_object_rigid_body_material_startup",
        params={
            "static_friction_range": [0.1, 0.6],
            "dynamic_friction_range": [0.1, 0.6],
            "restitution_range": [0.0, 1.0],
        },
    ),
    "randomize_object_rigid_body_mass_inertia_scale_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_object_rigid_body_mass_inertia_scale_startup",
        params={
            "mass_scale_distribution_params": [0.5, 1.5],
        },
    ),
}

base_setup_terms = {
    "push_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState",
        params={
            "push_interval_s": [0.5, 2.0],
            "max_push_vel": [0.7, 0.7, 0.25, 0.7, 0.7, 1.0],
            "enabled": True,
        },
    ),
    "actuator_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState",
        params={
            "kp_range": [0.9, 1.1],
            "kd_range": [0.9, 1.1],
            "rfi_lim_range": [1.0, 1.0],
            "enable_pd_gain": False,
            "enable_rfi_lim": False,
        },
    ),
    "setup_action_delay_buffers": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:setup_action_delay_buffers",
        params={
            "ctrl_delay_step_range": [0, 1],
            "enabled": False,
        },
    ),
    **robot_state_dr_at_setup,
}

base_reset_terms = {
    "push_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
    ),
    "randomize_push_schedule": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_push_schedule",
    ),
    "randomize_action_delay": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_action_delay",
    ),
    "actuator_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState"
    ),
    # TODO: what is the difference between reset and setup? for joint_pos_bias_range?
    "randomize_dof_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_dof_state",
        params={
            "joint_pos_scale_range": [1.0, 1.0],
            "joint_vel_range": [0.0, 0.0],
            "joint_pos_bias_range": [-0.01, 0.01],
            "randomize_dof_pos_bias": False,
        },
    ),
    "randomize_camera_raycast": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_camera_raycast",
        params={
            "enabled": True,
            "translation_range": {"x": [-0.025, 0.025], "y": [-0.025, 0.025], "z": [-0.025, 0.025]},
            "rotation_range_deg": {"roll": [-2.5, 2.5], "pitch": [-3.0, 3.0], "yaw": [-2.5, 2.5]},
            "noise_std_mult_range": [0.0, 0.05],
            "noise_drop_prob_range": [0.0, 0.025],
        },
    ),
}

base_step_terms = {
    "push_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
    ),
    "apply_pushes": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:apply_pushes",
    ),
}

g1_29dof_wbt_randomization = RandomizationManagerCfg(
    setup_terms={**base_setup_terms},
    reset_terms={**base_reset_terms},
    step_terms={**base_step_terms},
)

distill_setup_terms = {
    **base_setup_terms,
    "randomize_base_com_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_base_com_startup",
        params={
            "base_com_range": {"x": [-0.055, 0.055], "y": [-0.08, 0.08], "z": [-0.1, 0.1]},
            "enabled": True,
        },
    ),
    "setup_torque_rfi": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:setup_torque_rfi",
        params={
            "enabled": True,
            "rfi_lim": 0.01,
        },
    ),
    "actuator_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState",
        params={
            **base_setup_terms["actuator_randomizer_state"].params,
            "enable_pd_gain": True,
            "rfi_lim_range": [0.0, 1.0],
            "enable_rfi_lim": True,
        },
    ),
    "mass_randomizer": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_mass_startup",
        params={
            "enable_link_mass": True,
            "link_mass_range": [0.9, 1.2],
            "enable_base_mass": True,
            "added_mass_range": [-1.0, 3.0],
        },
    ),
    "setup_action_delay_buffers": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:setup_action_delay_buffers",
        params={
            **base_setup_terms["setup_action_delay_buffers"].params,
            "enabled": True,
        },
    ),
}

g1_29dof_wbt_randomization_with_action_delay = RandomizationManagerCfg(
    setup_terms={**distill_setup_terms},
    reset_terms={**base_reset_terms},
    step_terms={**base_step_terms},
)

g1_29dof_wbt_randomization_w_object = RandomizationManagerCfg(
    setup_terms={
        **base_setup_terms,
        **object_state_dr_at_setup,
    },
    reset_terms={
        **base_reset_terms,
    },
    step_terms={
        **base_step_terms,
    },
)

g1_29dof_wbt_randomization_w_object_with_action_delay = RandomizationManagerCfg(
    setup_terms={
        **distill_setup_terms,
        **object_state_dr_at_setup,
    },
    reset_terms={
        **base_reset_terms,
    },
    step_terms={
        **base_step_terms,
    },
)

__all__ = [
    "g1_29dof_wbt_randomization",
    "g1_29dof_wbt_randomization_with_action_delay",
    "g1_29dof_wbt_randomization_w_object",
    "g1_29dof_wbt_randomization_w_object_with_action_delay",
]
