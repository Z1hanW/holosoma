"""Whole Body Tracking termination presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

g1_29dof_wbt_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTrackingZOnly",
            params={
                # robot tracking
                "bad_ref_pos_threshold": 1.0,
                "bad_ref_ori_threshold": 1.1,
                "bad_motion_body_pos_threshold": 0.5,
                # NOTE: body_names_to_track is shared with command_manager
                "body_names_to_track": [
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
                "bad_motion_body_pos_body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                # object tracking
                # only triggered when has_object=True
                "bad_object_pos_threshold": 0.5,
                "bad_object_ori_threshold": 1.1,
            },
        ),
    }
)

g1_29dof_wbt_termination_generalist = TerminationManagerCfg(
    terms={
        **g1_29dof_wbt_termination.terms,
        "bad_tracking": replace(
            g1_29dof_wbt_termination.terms["bad_tracking"],
            func="holosoma.managers.termination.terms.wbt:BadTracking",
        ),
        "motion_ends": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:motion_ends",
            is_timeout=False,
        ),
    }
)

g1_29dof_wbt_termination_generalist_z_only = TerminationManagerCfg(
    terms={
        **g1_29dof_wbt_termination.terms,
        "motion_ends": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:motion_ends",
            is_timeout=False,
        ),
    }
)

g1_29dof_wbt_termination_generalist_all_position_z_only = TerminationManagerCfg(
    terms={
        **g1_29dof_wbt_termination.terms,
        "bad_tracking": replace(
            g1_29dof_wbt_termination.terms["bad_tracking"],
            func=(
                "holosoma.managers.termination.terms.wbt:"
                "BadTrackingAllPositionZOnly"
            ),
        ),
        "motion_ends": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:motion_ends",
            is_timeout=False,
        ),
    }
)

g1_29dof_wbt_termination_hybrid_stage2 = TerminationManagerCfg(
    terms={
        **g1_29dof_wbt_termination_generalist.terms,
        "bad_tracking": replace(
            g1_29dof_wbt_termination_generalist.terms["bad_tracking"],
            func="holosoma.managers.termination.terms.wbt:HybridStage2BadTracking",
            params={
                **g1_29dof_wbt_termination_generalist.terms["bad_tracking"].params,
                "task_object_hold_pos_threshold": 0.35,
            },
        ),
    }
)

g1_29dof_wbt_termination_hybrid_velocity = TerminationManagerCfg(
    terms={
        **g1_29dof_wbt_termination_generalist.terms,
        "bad_tracking": replace(
            g1_29dof_wbt_termination_generalist.terms["bad_tracking"],
            func="holosoma.managers.termination.terms.wbt:HybridVelocityBadTracking",
            params={
                **g1_29dof_wbt_termination_generalist.terms["bad_tracking"].params,
                "task_object_hold_pos_threshold": 0.35,
                "task_max_tilt_deg": 60.0,
            },
        ),
    }
)

g1_29dof_wbt_termination_hmi = TerminationManagerCfg(
    terms={
        "timeout": g1_29dof_wbt_termination_generalist.terms["timeout"],
        "body_proximity": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BodyGroupProximity",
            params={
                "min_distance": 0.05,
                "body_groups": [
                    ["left_foot_contact_point", "right_foot_contact_point"]
                ],
            },
        ),
        "bad_tracking": replace(
            g1_29dof_wbt_termination_generalist.terms["bad_tracking"],
            func="holosoma.managers.termination.terms.wbt:HMIBadTracking",
            params={
                **g1_29dof_wbt_termination_generalist.terms["bad_tracking"].params,
                "bad_ref_pos_threshold": 0.40,
                "bad_ref_ori_threshold": 0.40,
                "bad_motion_body_pos_threshold": 0.25,
                "bad_object_pos_threshold": 0.30,
                "bad_object_ori_threshold": 0.80,
                "min_root_height": 0.45,
                "gen_bad_ref_pos_z_threshold": 0.40,
                "gen_bad_ref_pos_xyz_threshold": 100.0,
                "gen_bad_object_pos_z_threshold": 0.60,
                "gen_bad_object_ori_threshold": 1.0,
            },
        ),
    }
)

g1_29dof_wbt_termination_distill = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTracking",
            params={
                # Keep the default aligned with gt/generalist; launchers can pass
                # a small threshold augment (for example 1.1x or 1.2x) explicitly.
                # robot tracking
                "bad_ref_pos_threshold": 0.5,
                "bad_ref_ori_threshold": 0.8,
                "bad_motion_body_pos_threshold": 0.25,
                # NOTE: body_names_to_track is shared with command_manager
                "body_names_to_track": [
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
                "bad_motion_body_pos_body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                # object tracking
                # only triggered when has_object=True
                "bad_object_pos_threshold": 0.25,
                "bad_object_ori_threshold": 0.8,
            },
        ),
    }
)

__all__ = [
    "g1_29dof_wbt_termination",
    "g1_29dof_wbt_termination_generalist",
    "g1_29dof_wbt_termination_generalist_z_only",
    "g1_29dof_wbt_termination_generalist_all_position_z_only",
    "g1_29dof_wbt_termination_hybrid_stage2",
    "g1_29dof_wbt_termination_hybrid_velocity",
    "g1_29dof_wbt_termination_hmi",
    "g1_29dof_wbt_termination_distill",
]
