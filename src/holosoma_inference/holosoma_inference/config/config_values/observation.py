"""Default observation configurations for holosoma_inference.

This module provides pre-configured observation spaces for different
robot types and tasks, converted from the original YAML configurations.
"""

from __future__ import annotations

from holosoma_inference.config.config_types.observation import ObservationConfig

# =============================================================================
# Locomotion Observation Configurations
# =============================================================================

loco_g1_29dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase",
        ]
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "sin_phase": 1,
        "cos_phase": 1,
    },
    obs_scales={
        "base_lin_vel": 2.0,
        "base_ang_vel": 0.25,
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

loco_t1_29dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase",
        ]
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "sin_phase": 2,
        "cos_phase": 2,
    },
    obs_scales={
        "base_lin_vel": 1.0,  # T1 uses 1.0 (vs G1's 2.0)
        "base_ang_vel": 1.0,  # T1 uses 1.0 (vs G1's 0.25)
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.1,  # T1 uses 0.1 (vs G1's 0.05)
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)


# =============================================================================
# WBT (Whole Body Tracking) Observation Configurations
# =============================================================================

wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "motion_command",
            "motion_ref_ori_b",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ]
    },
    obs_dims={
        "motion_command": 58,
        "motion_ref_pos_b": 3,
        "motion_ref_ori_b": 6,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "actions": 1.0,
        "motion_command": 1.0,
        "motion_ref_pos_b": 1.0,
        "motion_ref_ori_b": 1.0,
        "base_lin_vel": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "robot_body_pos_b": 1.0,
        "robot_body_ori_b": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)


loco_manip_stand_height_waist = ObservationConfig(
    obs_intervals={
        "actor_obs": 1,
        "perception_obs": 5,
        "command_lin_vel_obs": 1,
    },
    obs_dict={
        "actor_obs": [
            "base_ang_vel",  # checked
            "projected_gravity",  # checked
            "command_lin_vel",  # checked
            "command_ang_vel",  # checked
            "command_stand",  # checked
            "command_base_height",
            "command_waist_dofs",
            "ref_upper_dof_pos",
            "dof_pos",  # checked
            "dof_vel",  # checked
            "actions",  # checked
        ],
        "perception_obs": [
            "cam_depth",
        ],
        "command_lin_vel_obs": [
            "command_lin_vel",
        ],
    },
    obs_dims={
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "command_stand": 1,
        "command_base_height": 1,
        "command_waist_dofs": 3,
        "ref_upper_dof_pos": 14,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        # cam_depth: ${eval:${robot.cameras.props.resized_width} * ${robot.cameras.props.resized_height} * ${len:${robot.cameras.poses}}}
        "cam_depth": 2592,  # 2 cameras, 48 width, 27 height
    },
    obs_scales={
        "base_ang_vel": 0.25,
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "command_stand": 1.0,
        "command_base_height": 2.0,
        "command_waist_dofs": 1.0,
        "ref_upper_dof_pos": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "history": 1.0,
        "actions": 1.0,
        "cam_depth": 1.0,
    },
    history_length_dict={
        "actor_obs": 5,
        "perception_obs": 2,
        "command_lin_vel_obs": 1,
    },
    # Note: obs_intervals from YAML not included as it's not present in other Python configs
    # Original intervals were: actor_state_obs: 1, perception_obs: ${eval:${task.policy.config.rl_rate} / ${robot.cameras.props.frame_rate}}, command_lin_vel_obs: 1
)
# =============================================================================
# Default Configurations Dictionary
# =============================================================================

DEFAULTS = {
    "loco-g1-29dof": loco_g1_29dof,
    "loco-t1-29dof": loco_t1_29dof,
    "wbt": wbt,
    "loco-manip-stand-height-waist": loco_manip_stand_height_waist,
}
"""Dictionary of all available observation configurations.

Keys use hyphen-case naming convention for CLI compatibility.
"""
