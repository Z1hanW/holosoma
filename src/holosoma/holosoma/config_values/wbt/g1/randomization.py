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
            "static_friction_range": [0.1, 0.7],
            "dynamic_friction_ratio_range": [0.7, 0.99],
            "restitution_range": [0.0, 1.0],
        },
    ),
    "randomize_object_rigid_body_mass_inertia_scale_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_object_rigid_body_mass_inertia_scale_startup",
        params={
            "mass_scale_distribution_params": [0.25, 4.0],
            "mass_scale_distribution": "log_uniform",
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
            "translation_range": {"x": [-0.035, 0.035], "y": [-0.035, 0.035], "z": [-0.035, 0.035]},
            "rotation_range_deg": {"roll": [-3.5, 3.5], "pitch": [-3.5, 3.5], "yaw": [-3.5, 3.5]},
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
    # Current student contract: keep state/dynamics/perception randomization,
    # but use nominal joint calibration and actuator semantics.  Keep these
    # terms present as explicit disabled no-ops so existing launchers and
    # checkpoint config schemas remain compatible.
    "setup_dof_pos_bias": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:setup_dof_pos_bias",
        params={
            "dof_pos_bias_range": [0.0, 0.0],
            "enabled": False,
        },
    ),
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
            "enabled": False,
            "rfi_lim": 0.0,
        },
    ),
    "actuator_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState",
        params={
            **base_setup_terms["actuator_randomizer_state"].params,
            "kp_range": [1.0, 1.0],
            "kd_range": [1.0, 1.0],
            "enable_pd_gain": False,
            "rfi_lim_range": [1.0, 1.0],
            "enable_rfi_lim": False,
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
            "ctrl_delay_step_range": [0, 0],
            "enabled": False,
        },
    ),
}

# Pure PPO keeps the current state, rigid-body, object, and perception
# randomization contract, but restores the actuator/control-chain uncertainty
# used by the original tracking policy. Keep this separate from
# ``distill_setup_terms``: label-based student training intentionally uses the
# nominal control chain, while pure RL can learn under the perturbations.
pure_rl_control_chain_setup_terms = {
    **distill_setup_terms,
    "setup_dof_pos_bias": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:setup_dof_pos_bias",
        params={
            "dof_pos_bias_range": [-0.01, 0.01],
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
            "kp_range": [0.9, 1.1],
            "kd_range": [0.9, 1.1],
            "enable_pd_gain": True,
            "rfi_lim_range": [0.0, 1.0],
            "enable_rfi_lim": True,
        },
    ),
    "setup_action_delay_buffers": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:setup_action_delay_buffers",
        params={
            "ctrl_delay_step_range": [0, 1],
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

g1_29dof_wbt_randomization_w_object_pure_rl = RandomizationManagerCfg(
    setup_terms={
        **pure_rl_control_chain_setup_terms,
        **object_state_dr_at_setup,
    },
    reset_terms={
        **base_reset_terms,
    },
    step_terms={
        **base_step_terms,
    },
)

# Privileged-teacher coverage preset.  The legacy ``state_robust`` name is kept
# for launcher/checkpoint-schema compatibility, but the teacher now samples a
# modest guard band around every physical-parameter distribution used by the
# student.  This keeps teacher action labels in-distribution when they are
# queried in the student's randomized simulator.  Actuator/calibration and
# perception randomization remain absent because those terms are disabled for
# the student or are not consumed by the privileged teacher.
#
# MotionCommand consumes the reset state below at its final simulator-state
# write so the joint perturbation cannot be overwritten.
teacher_state_robust_setup_terms = {
    "motion_relative_reset_randomizer_state": RandomizationTermCfg(
        func=(
            "holosoma.managers.randomization.terms.locomotion:"
            "MotionRelativeResetRandomizerState"
        ),
        params={
            "overall_noise_scale": 1.0,
            "dof_pos": 0.20,
            "dof_vel": 0.35,
            "root_pos": [0.08, 0.08, 0.025],
            "root_rot": [0.15, 0.15, 0.30],
            "root_lin_vel": [0.20, 0.20, 0.10],
            "root_ang_vel": [0.25, 0.25, 0.35],
            # Keep z exact so an otherwise valid motion object cannot spawn
            # below the support surface.
            "object_pos": [0.08, 0.08, 0.0],
        },
    ),
    "push_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState",
        params={
            # Cover the student's [0.5, 2.0] s schedule and
            # [0.7, 0.7, 0.25, 0.7, 0.7, 1.0] velocity envelope with a
            # deliberately modest recovery margin.
            "push_interval_s": [0.4, 2.2],
            "max_push_vel": [0.8, 0.8, 0.30, 0.8, 0.8, 1.1],
            "enabled": True,
        },
    ),
    "randomize_robot_rigid_body_material_startup": RandomizationTermCfg(
        func=(
            "holosoma.managers.randomization.terms.locomotion:"
            "randomize_robot_rigid_body_material_startup"
        ),
        params={
            "static_friction_range": [0.25, 1.7],
            "dynamic_friction_range": [0.25, 1.3],
            "restitution_range": [0.0, 0.6],
        },
    ),
    "randomize_base_com_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_base_com_startup",
        params={
            "base_com_range": {
                "x": [-0.065, 0.065],
                "y": [-0.095, 0.095],
                "z": [-0.12, 0.12],
            },
            "enabled": True,
        },
    ),
    "mass_randomizer": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_mass_startup",
        params={
            "enable_link_mass": True,
            "link_mass_range": [0.85, 1.25],
            "enable_base_mass": True,
            "added_mass_range": [-1.5, 3.5],
        },
    ),
    "randomize_object_rigid_body_material_startup": RandomizationTermCfg(
        func=(
            "holosoma.managers.randomization.terms.locomotion:"
            "randomize_object_rigid_body_material_startup"
        ),
        params={
            "static_friction_range": [0.08, 0.8],
            "dynamic_friction_ratio_range": [0.65, 0.99],
            # The student already spans the complete physically valid range.
            "restitution_range": [0.0, 1.0],
        },
    ),
    "randomize_object_rigid_body_mass_inertia_scale_startup": RandomizationTermCfg(
        func=(
            "holosoma.managers.randomization.terms.locomotion:"
            "randomize_object_rigid_body_mass_inertia_scale_startup"
        ),
        params={
            # Strictly cover the student's log-uniform [0.25, 4.0]
            # density support while retaining a modest teacher guard band.
            "mass_scale_distribution_params": [0.20, 5.0],
            "mass_scale_distribution": "log_uniform",
        },
    ),
}

teacher_state_robust_reset_terms = {
    "push_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
    ),
    "randomize_push_schedule": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_push_schedule",
    ),
}

teacher_state_robust_step_terms = {
    "push_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
    ),
    "apply_pushes": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:apply_pushes",
    ),
}

g1_29dof_wbt_randomization_w_object_teacher_state_robust = RandomizationManagerCfg(
    setup_terms=teacher_state_robust_setup_terms,
    reset_terms=teacher_state_robust_reset_terms,
    step_terms=teacher_state_robust_step_terms,
)

# Keep the privileged teacher's physical/state distribution unchanged while
# exercising the same depth-camera stochastic contract as a depth student.
# This is a separate preset because the standard teacher does not consume
# perception and should retain its existing behavior by default.
g1_29dof_wbt_randomization_w_object_teacher_state_robust_with_camera = (
    RandomizationManagerCfg(
        setup_terms={
            **teacher_state_robust_setup_terms,
            "setup_camera_raycast_randomization": robot_state_dr_at_setup[
                "setup_camera_raycast_randomization"
            ],
        },
        reset_terms={
            **teacher_state_robust_reset_terms,
            "randomize_camera_raycast": base_reset_terms["randomize_camera_raycast"],
        },
        step_terms=teacher_state_robust_step_terms,
    )
)

__all__ = [
    "g1_29dof_wbt_randomization",
    "g1_29dof_wbt_randomization_with_action_delay",
    "g1_29dof_wbt_randomization_w_object",
    "g1_29dof_wbt_randomization_w_object_pure_rl",
    "g1_29dof_wbt_randomization_w_object_with_action_delay",
    "g1_29dof_wbt_randomization_w_object_teacher_state_robust",
    "g1_29dof_wbt_randomization_w_object_teacher_state_robust_with_camera",
]
