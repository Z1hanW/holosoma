"""Whole Body Tracking reward presets for the G1 robot."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

_FOOT_OBJECT_CONTACT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
_PALM_CONTACT_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
_ARM_SUPPORT_CONTACT_BODY_NAMES = ["left_elbow_link", "right_elbow_link"]
_TORSO_SUPPORT_CONTACT_BODY_NAMES = ["torso_link"]

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
            weight=-10.0,
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
            weight=-0.1,
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

g1_29dof_wbt_reward_w_object_generalist = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward_w_object.terms,
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
            weight=0.0,
        ),
    }
)

g1_29dof_wbt_reward_w_object_generalist_offline_contact_guidance = g1_29dof_wbt_reward_w_object_generalist

__all__ = [
    "g1_29dof_wbt_fast_sac_reward",
    "g1_29dof_wbt_reward",
    "g1_29dof_wbt_reward_w_object",
    "g1_29dof_wbt_reward_w_object_generalist",
    "g1_29dof_wbt_reward_w_object_generalist_offline_contact_guidance",
]
