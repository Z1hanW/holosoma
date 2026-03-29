"""Whole Body Tracking reward presets for the G1 robot."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

_LOWER_TRACKED_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
]
_TORSO_TRACKED_BODY_NAMES = ["torso_link"]
_UPPER_TRACKED_BODY_NAMES = [
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

_LOWER_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
_WAIST_DOF_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
_UPPER_DOF_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# The current whole-body object asset uses a fixed palm joint
# (left/right_hand_palm_joint) that gets collapsed into wrist_yaw_link.
_PALM_CONTACT_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
_ARM_SUPPORT_CONTACT_BODY_NAMES = [
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
]
# On the current G1 object-carry asset, chest/trunk bracing against the box
# is represented by torso_link contact.
_TORSO_SUPPORT_CONTACT_BODY_NAMES = ["torso_link"]
_FOOT_OBJECT_CONTACT_BODY_NAMES = [
    "left_foot_contact_point",
    "right_foot_contact_point",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

g1_29dof_wbt_reward = RewardManagerCfg(
    terms={
        # Motion tracking rewards - global reference frame
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=0.5,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        # Motion tracking rewards - body velocities
        "motion_global_body_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={"sigma": 1.0},
            weight=1.0,
        ),
        "motion_global_body_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={"sigma": 3.14},
            weight=1.0,
        ),
        # Regularization rewards
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-0.1,
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},
            weight=-100.0,
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": (
                    "^(?!left_foot_contact_point$)(?!right_foot_contact_point$)"
                    "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                    "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"
                ),
            },
            weight=-0.5,
        ),
    }
)

g1_29dof_wbt_fast_sac_reward = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-1.0,
        ),
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=2.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
    }
)

g1_29dof_wbt_reward_w_object = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        # Penalize only foot/ankle contacts with the box itself. Floor contacts are handled elsewhere.
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:ObjectUndesiredContacts",
            params={
                "threshold": 1.0,
                "body_names": _FOOT_OBJECT_CONTACT_BODY_NAMES,
            },
            weight=-0.5,
        ),
        # Motion tracking rewards - global reference frame
        "object_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "object_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
    }
)

g1_29dof_wbt_reward_w_object_distill_sparse_goal_mixed = RewardManagerCfg(
    terms={
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3, "only_clip_goal": True},
            weight=1.5,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4, "only_clip_goal": True},
            weight=0.75,
        ),
        "motion_global_body_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={"sigma": 1.0, "only_clip_goal": True},
            weight=0.2,
        ),
        "motion_global_body_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={"sigma": 3.14, "only_clip_goal": True},
            weight=0.1,
        ),
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-0.05,
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},
            weight=-100.0,
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:ObjectUndesiredContacts",
            params={
                "threshold": 1.0,
                "body_names": _FOOT_OBJECT_CONTACT_BODY_NAMES,
            },
            weight=-0.5,
        ),
        "body_contact_reward_palms": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:body_object_contact_reward",
            params={
                "threshold": 1.0,
                "force_scale": 25.0,
                "reward_mode": "tanh",
                "body_names": _PALM_CONTACT_BODY_NAMES,
            },
            weight=0.10,
        ),
        "body_contact_reward_arms": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:body_object_contact_reward",
            params={
                "threshold": 1.0,
                "force_scale": 25.0,
                "reward_mode": "tanh",
                "body_names": _ARM_SUPPORT_CONTACT_BODY_NAMES,
            },
            weight=0.20,
        ),
        "body_contact_reward_torso": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:body_object_contact_reward",
            params={
                "threshold": 1.0,
                "force_scale": 25.0,
                "reward_mode": "tanh",
                "body_names": _TORSO_SUPPORT_CONTACT_BODY_NAMES,
            },
            weight=0.30,
        ),
        "sparse_goal_pickup_height_reward": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:sparse_goal_pickup_height_reward",
            params={
                "only_external": True,
                "stop_after_pick": True,
                "target_height_delta": 0.12,
            },
            weight=0.0,
        ),
        "sparse_goal_object_pose_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:sparse_goal_object_pose_error_exp",
            params={
                "sigma_xy": 0.20,
                "sigma_yaw": 0.50,
                "sigma_z": 0.05,
                "only_external": True,
                "picked_only": True,
                "ignore_yaw": True,
            },
            weight=4.0,
        ),
        "sparse_goal_hover_height_penalty": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:sparse_goal_hover_height_penalty",
            params={
                "only_external": True,
                "picked_only": True,
                "near_goal_xy_threshold": 0.20,
                "near_goal_yaw_threshold": 0.60,
                "target_height_margin": 0.10,
                "height_scale": 0.12,
                "ignore_yaw": True,
            },
            weight=-2.0,
        ),
        "sparse_goal_success_bonus": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:sparse_goal_success_bonus",
            params={
                "only_external": True,
                "xy_threshold": 0.10,
                "yaw_threshold": 0.35,
                "z_threshold": 0.06,
                "lin_vel_threshold": 0.30,
                "ang_vel_threshold": 1.50,
                "ignore_yaw": True,
            },
            weight=20.0,
        ),
    }
)

g1_29dof_wbt_reward_w_object_extend = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward_w_object.terms,
        "motion_relative_body_position_error_lower": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3, "body_names": _LOWER_TRACKED_BODY_NAMES},
            weight=0.0,
        ),
        "motion_relative_body_orientation_error_lower": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4, "body_names": _LOWER_TRACKED_BODY_NAMES},
            weight=0.0,
        ),
        "motion_global_body_lin_vel_lower": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={"sigma": 1.0, "body_names": _LOWER_TRACKED_BODY_NAMES},
            weight=0.0,
        ),
        "motion_global_body_ang_vel_lower": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={"sigma": 3.14, "body_names": _LOWER_TRACKED_BODY_NAMES},
            weight=0.0,
        ),
        "motion_relative_body_position_error_torso": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.25, "body_names": _TORSO_TRACKED_BODY_NAMES},
            weight=0.0,
        ),
        "motion_relative_body_orientation_error_torso": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.3, "body_names": _TORSO_TRACKED_BODY_NAMES},
            weight=0.0,
        ),
        "motion_relative_body_position_error_upper": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3, "body_names": _UPPER_TRACKED_BODY_NAMES},
            weight=0.0,
        ),
        "motion_relative_body_orientation_error_upper": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4, "body_names": _UPPER_TRACKED_BODY_NAMES},
            weight=0.0,
        ),
        "motion_global_body_lin_vel_upper": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={"sigma": 1.0, "body_names": _UPPER_TRACKED_BODY_NAMES},
            weight=0.0,
        ),
        "motion_global_body_ang_vel_upper": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={"sigma": 3.14, "body_names": _UPPER_TRACKED_BODY_NAMES},
            weight=0.0,
        ),
        "motion_joint_position_error_lower": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_joint_position_error_exp",
            params={"sigma": 0.3, "dof_names": _LOWER_DOF_NAMES},
            weight=0.0,
        ),
        "motion_joint_velocity_error_lower": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_joint_velocity_error_exp",
            params={"sigma": 2.0, "dof_names": _LOWER_DOF_NAMES},
            weight=0.0,
        ),
        "motion_joint_position_error_waist": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_joint_position_error_exp",
            params={"sigma": 0.25, "dof_names": _WAIST_DOF_NAMES},
            weight=0.0,
        ),
        "motion_joint_velocity_error_waist": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_joint_velocity_error_exp",
            params={"sigma": 2.0, "dof_names": _WAIST_DOF_NAMES},
            weight=0.0,
        ),
        "motion_joint_position_error_upper": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_joint_position_error_exp",
            params={"sigma": 0.35, "dof_names": _UPPER_DOF_NAMES},
            weight=0.0,
        ),
        "motion_joint_velocity_error_upper": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_joint_velocity_error_exp",
            params={"sigma": 2.5, "dof_names": _UPPER_DOF_NAMES},
            weight=0.0,
        ),
        "body_contact_reward_palms": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:body_object_contact_reward",
            params={
                "threshold": 1.0,
                "force_scale": 25.0,
                "reward_mode": "binary",
                "body_names": _PALM_CONTACT_BODY_NAMES,
            },
            weight=0.0,
        ),
        "body_contact_reward_arms": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:body_object_contact_reward",
            params={
                "threshold": 1.0,
                "force_scale": 25.0,
                "reward_mode": "binary",
                "body_names": _ARM_SUPPORT_CONTACT_BODY_NAMES,
            },
            weight=0.0,
        ),
        "body_contact_reward_torso": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:body_object_contact_reward",
            params={
                "threshold": 1.0,
                "force_scale": 25.0,
                "reward_mode": "binary",
                "body_names": _TORSO_SUPPORT_CONTACT_BODY_NAMES,
            },
            weight=0.0,
        ),
    }
)

g1_29dof_wbt_reward_w_object_command_curriculum = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward_w_object_extend.terms,
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3, "only_clip_goal": True},
            weight=0.5,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4, "only_clip_goal": True},
            weight=0.5,
        ),
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3, "only_clip_goal": True},
            weight=1.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4, "only_clip_goal": True},
            weight=1.0,
        ),
        "motion_global_body_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={"sigma": 1.0, "only_clip_goal": True},
            weight=1.0,
        ),
        "motion_global_body_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={"sigma": 3.14, "only_clip_goal": True},
            weight=1.0,
        ),
        "object_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_position_error_exp",
            params={"sigma": 0.3, "only_clip_goal": True},
            weight=1.0,
        ),
        "object_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_orientation_error_exp",
            params={"sigma": 0.4, "only_clip_goal": True},
            weight=1.0,
        ),
        "motion_relative_body_position_error_lower": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3, "body_names": _LOWER_TRACKED_BODY_NAMES, "only_clip_goal": True},
            weight=0.0,
        ),
        "motion_relative_body_orientation_error_lower": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4, "body_names": _LOWER_TRACKED_BODY_NAMES, "only_clip_goal": True},
            weight=0.0,
        ),
        "motion_global_body_lin_vel_lower": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={"sigma": 1.0, "body_names": _LOWER_TRACKED_BODY_NAMES, "only_clip_goal": True},
            weight=0.0,
        ),
        "motion_global_body_ang_vel_lower": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={"sigma": 3.14, "body_names": _LOWER_TRACKED_BODY_NAMES, "only_clip_goal": True},
            weight=0.0,
        ),
        "motion_relative_body_position_error_torso": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.25, "body_names": _TORSO_TRACKED_BODY_NAMES, "only_clip_goal": True},
            weight=0.0,
        ),
        "motion_relative_body_orientation_error_torso": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.3, "body_names": _TORSO_TRACKED_BODY_NAMES, "only_clip_goal": True},
            weight=0.0,
        ),
        "motion_relative_body_position_error_upper": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3, "body_names": _UPPER_TRACKED_BODY_NAMES, "only_clip_goal": True},
            weight=0.0,
        ),
        "motion_relative_body_orientation_error_upper": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4, "body_names": _UPPER_TRACKED_BODY_NAMES, "only_clip_goal": True},
            weight=0.0,
        ),
        "motion_global_body_lin_vel_upper": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={"sigma": 1.0, "body_names": _UPPER_TRACKED_BODY_NAMES, "only_clip_goal": True},
            weight=0.0,
        ),
        "motion_global_body_ang_vel_upper": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={"sigma": 3.14, "body_names": _UPPER_TRACKED_BODY_NAMES, "only_clip_goal": True},
            weight=0.0,
        ),
        "motion_joint_position_error_lower": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_joint_position_error_exp",
            params={"sigma": 0.3, "dof_names": _LOWER_DOF_NAMES, "only_clip_goal": True},
            weight=0.0,
        ),
        "motion_joint_velocity_error_lower": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_joint_velocity_error_exp",
            params={"sigma": 2.0, "dof_names": _LOWER_DOF_NAMES, "only_clip_goal": True},
            weight=0.0,
        ),
        "motion_joint_position_error_waist": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_joint_position_error_exp",
            params={"sigma": 0.25, "dof_names": _WAIST_DOF_NAMES, "only_clip_goal": True},
            weight=0.0,
        ),
        "motion_joint_velocity_error_waist": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_joint_velocity_error_exp",
            params={"sigma": 2.0, "dof_names": _WAIST_DOF_NAMES, "only_clip_goal": True},
            weight=0.0,
        ),
        "motion_joint_position_error_upper": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_joint_position_error_exp",
            params={"sigma": 0.35, "dof_names": _UPPER_DOF_NAMES, "only_clip_goal": True},
            weight=0.0,
        ),
        "motion_joint_velocity_error_upper": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_joint_velocity_error_exp",
            params={"sigma": 2.5, "dof_names": _UPPER_DOF_NAMES, "only_clip_goal": True},
            weight=0.0,
        ),
        "sparse_goal_pickup_height_reward": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:sparse_goal_pickup_height_reward",
            params={
                "only_external": True,
                "stop_after_pick": True,
                "target_height_delta": 0.12,
            },
            weight=0.0,
        ),
        "sparse_goal_object_pose_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:sparse_goal_object_pose_error_exp",
            params={
                "sigma_xy": 0.35,
                "sigma_yaw": 1.0,
                "sigma_z": 0.10,
                "only_external": True,
                "picked_only": False,
                "ignore_yaw": True,
            },
            weight=4.0,
        ),
        "sparse_goal_hover_height_penalty": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:sparse_goal_hover_height_penalty",
            params={
                "only_external": True,
                "picked_only": True,
                "near_goal_xy_threshold": 0.20,
                "near_goal_yaw_threshold": 0.60,
                "target_height_margin": 0.10,
                "height_scale": 0.12,
                "ignore_yaw": True,
            },
            weight=0.0,
        ),
        "sparse_goal_success_bonus": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:sparse_goal_success_bonus",
            params={
                "only_external": True,
                "xy_threshold": 0.10,
                "yaw_threshold": 0.35,
                "z_threshold": 0.06,
                "lin_vel_threshold": 0.30,
                "ang_vel_threshold": 1.50,
                "ignore_yaw": True,
            },
            weight=20.0,
        ),
        "command_contact_prior_guidance": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:CommandCurriculumContactPrior",
            params={
                "only_command_env": True,
                "contact_threshold": 1.0,
                "force_scale": 25.0,
                "force_match_sigma": 0.25,
                "position_sigma": 0.18,
                "force_weight": 0.55,
                "position_weight": 0.45,
                "expected_contact_min_occupancy": 0.15,
            },
            weight=1.0,
        ),
    }
)

g1_29dof_wbt_reward_w_object_generalist = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward_w_object_extend.terms,
        "body_contact_reward_palms": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:body_object_contact_reward",
            params={
                "threshold": 1.0,
                "force_scale": 25.0,
                "reward_mode": "tanh",
                "body_names": _PALM_CONTACT_BODY_NAMES,
            },
            weight=0.10,
        ),
        "body_contact_reward_arms": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:body_object_contact_reward",
            params={
                "threshold": 1.0,
                "force_scale": 25.0,
                "reward_mode": "tanh",
                "body_names": _ARM_SUPPORT_CONTACT_BODY_NAMES,
            },
            weight=0.20,
        ),
        "body_contact_reward_torso": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:body_object_contact_reward",
            params={
                "threshold": 1.0,
                "force_scale": 25.0,
                "reward_mode": "tanh",
                "body_names": _TORSO_SUPPORT_CONTACT_BODY_NAMES,
            },
            weight=0.30,
        ),
    }
)

__all__ = [
    "g1_29dof_wbt_fast_sac_reward",
    "g1_29dof_wbt_reward",
    "g1_29dof_wbt_reward_w_object",
    "g1_29dof_wbt_reward_w_object_command_curriculum",
    "g1_29dof_wbt_reward_w_object_distill_sparse_goal_mixed",
    "g1_29dof_wbt_reward_w_object_generalist",
    "g1_29dof_wbt_reward_w_object_extend",
]
