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

g1_29dof_wbt_termination_distill = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTrackingZOnly",
            params={
                # Distillation rollouts benefit from more tolerance before early reset
                # so the student can recover from teacher/student mismatch.
                # robot tracking
                "bad_ref_pos_threshold": 1.0,
                "bad_ref_ori_threshold": 1.2,
                "bad_motion_body_pos_threshold": 0.55,
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
                "bad_object_pos_threshold": 0.65,
                "bad_object_ori_threshold": 1.2,
            },
        ),
    }
)

g1_29dof_wbt_termination_distill_sparse_goal_mixed = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "motion_ends_clip_goal": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:motion_ends_if_clip_goal",
            params={"only_clip_goal": True},
        ),
        "bad_tracking_non_external_goal": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTrackingZOnly",
            params={
                # Keep bad-tracking resets on all non-external episodes so the
                # student continues to train on the stable motion-tracking
                # distribution. External-goal episodes remain exempt.
                "only_clip_goal": True,
                "bad_ref_pos_threshold": 1.0,
                "bad_ref_ori_threshold": 1.2,
                "bad_motion_body_pos_threshold": 0.55,
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
                "bad_object_pos_threshold": 0.65,
                "bad_object_ori_threshold": 1.2,
            },
        ),
        "sparse_goal_success": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:SparseGoalSuccess",
            params={
                "only_external": True,
                "xy_threshold": 0.10,
                "yaw_threshold": 0.35,
                "z_threshold": 0.06,
                "lin_vel_threshold": 1.0e6,
                "ang_vel_threshold": 1.0e6,
                "ignore_yaw": True,
                "hold_steps": 10,
            },
        ),
        "sparse_goal_dropped_away": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:SparseGoalDroppedAway",
            params={
                "only_external": True,
                "xy_fail_threshold": 0.35,
                "release_height_margin": 0.08,
                "hold_steps": 2,
            },
        ),
    }
)

g1_29dof_wbt_termination_distill_sparse_goal_pickup = TerminationManagerCfg(
    terms={
        name: term
        for name, term in g1_29dof_wbt_termination_distill_sparse_goal_mixed.terms.items()
        if name not in ("sparse_goal_success", "sparse_goal_dropped_away")
    }
)

g1_29dof_wbt_termination_command_curriculum = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "motion_ends_clip_goal": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:motion_ends_if_clip_goal",
            params={"only_clip_goal": True},
        ),
        "bad_tracking_clip_goal": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTrackingZOnly",
            params={
                "only_clip_goal": True,
                "bad_ref_pos_threshold": 1.0,
                "bad_ref_ori_threshold": 1.2,
                "bad_motion_body_pos_threshold": 0.55,
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
                "bad_object_pos_threshold": 0.65,
                "bad_object_ori_threshold": 1.2,
            },
        ),
        "sparse_goal_success": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:SparseGoalSuccess",
            params={
                "only_external": True,
                "xy_threshold": 0.10,
                "yaw_threshold": 0.35,
                "z_threshold": 0.06,
                "lin_vel_threshold": 1.0e6,
                "ang_vel_threshold": 1.0e6,
                "ignore_yaw": True,
                "hold_steps": 10,
            },
        ),
        "sparse_goal_dropped_away": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:SparseGoalDroppedAway",
            params={
                "only_external": True,
                "xy_fail_threshold": 0.35,
                "release_height_margin": 0.08,
                "hold_steps": 2,
            },
        ),
    }
)

__all__ = [
    "g1_29dof_wbt_termination",
    "g1_29dof_wbt_termination_generalist",
    "g1_29dof_wbt_termination_command_curriculum",
    "g1_29dof_wbt_termination_distill",
    "g1_29dof_wbt_termination_distill_sparse_goal_mixed",
    "g1_29dof_wbt_termination_distill_sparse_goal_pickup",
]
