"""Whole Body Tracking reward presets for the G1 robot."""

from dataclasses import replace

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

_WRIST_CONTACT_BODY_NAMES = {
    "left_wrist_yaw": "left_wrist_yaw_link",
    "right_wrist_yaw": "right_wrist_yaw_link",
}
_PALM_CONTACT_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
_ARM_SUPPORT_CONTACT_BODY_NAMES = [
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
]
_OFFLINE_CONTACT_GUIDANCE_REGION_NAMES = [
    "left_wrist",
    "right_wrist",
    "left_elbow",
    "right_elbow",
    "left_wrist_roll",
    "right_wrist_roll",
    "left_wrist_pitch",
    "right_wrist_pitch",
    "torso",
]
_AS_KEEP169_CONTACT_EXPORT_ROOT = (
    "data/ds_as_data/carryany_filter_scale_noscale_keep169_20260513/contact_export_from_retarget"
)
# On the current G1 object-carry asset, chest/trunk bracing against the box
# is represented by torso_link contact.
_TORSO_SUPPORT_CONTACT_BODY_NAMES = ["torso_link"]
_FOOT_OBJECT_CONTACT_BODY_NAMES = [
    "left_foot_contact_point",
    "right_foot_contact_point",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]
_LOWER_BODY_UNDESIRED_CONTACT_BODY_NAMES = [
    # head_link is fixed-collapsed into torso_link in IsaacSim, so torso contact
    # also catches contact on the head collision mesh.
    "torso_link",
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
]


def _lower_body_undesired_contacts_term() -> RewardTermCfg:
    return RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
        params={
            "threshold": 1.0,
            "undesired_contacts_body_names": _LOWER_BODY_UNDESIRED_CONTACT_BODY_NAMES,
            "required_selected_body_names": ["torso_link"],
            "forbidden_sim_body_names": ["head_link"],
        },
        weight=-0.1,
    )


def _fused_offline_contact_guidance_term(
    *,
    contact_export_root: str,
    wrist_weight: float,
    contact_weight: float,
    force_threshold: float,
    position_sigma: float = 0.08,
    force_sigma: float = 10.0,
    contact_schedule_relax_steps: int = 5,
) -> RewardTermCfg:
    return RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:FusedOfflineContactPointGuidance",
        params={
            "contact_export_root": contact_export_root,
            "wrist_region_names": ["left_wrist", "right_wrist"],
            "contact_region_names": _OFFLINE_CONTACT_GUIDANCE_REGION_NAMES,
            "position_sigma": position_sigma,
            "force_threshold": force_threshold,
            "force_sigma": force_sigma,
            "use_force_term": True,
            "force_gate_mode": "binary",
            "use_contact_schedule": True,
            "contact_schedule_relax_steps": contact_schedule_relax_steps,
            "contact_schedule_missing_mode": "after_pickup",
            "require_stable_contact": True,
            "min_target_points": 1,
            "wrist_weight": wrist_weight,
            "contact_weight": contact_weight,
        },
        weight=1.0,
    )


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
        # Penalize only foot/ankle contacts with the box itself. Floor contacts are handled elsewhere.
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:ObjectUndesiredContacts",
            params={
                "threshold": 1.0,
                "body_names": _FOOT_OBJECT_CONTACT_BODY_NAMES,
            },
            weight=-0.5,
        ),
        "lower_body_undesired_contacts": _lower_body_undesired_contacts_term(),
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
        **{
            f"body_contact_reward_{region_name}": RewardTermCfg(
                func="holosoma.managers.reward.terms.wbt:body_object_contact_reward",
                params={
                    "threshold": 1.0,
                    "force_scale": 25.0,
                    "reward_mode": "binary",
                    "body_names": [body_name],
                },
                weight=0.0,
            )
            for region_name, body_name in _WRIST_CONTACT_BODY_NAMES.items()
        },
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
            weight=0.0,
        ),
    }
)

g1_29dof_wbt_reward_w_object_generalist_offline_contact_guidance = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward_w_object_generalist.terms,
        # Replace coarse positive body-object contact rewards with retargeted
        # object-frame contact-point guidance. Keep the foot/ankle object-contact
        # penalty inherited from the object reward.
        "body_contact_reward_palms": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:body_object_contact_reward",
            params={
                "threshold": 1.0,
                "force_scale": 25.0,
                "reward_mode": "tanh",
                "body_names": _PALM_CONTACT_BODY_NAMES,
            },
            weight=0.0,
        ),
        "body_contact_reward_arms": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:body_object_contact_reward",
            params={
                "threshold": 1.0,
                "force_scale": 25.0,
                "reward_mode": "tanh",
                "body_names": _ARM_SUPPORT_CONTACT_BODY_NAMES,
            },
            weight=0.0,
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
        "offline_contact_guidance": _fused_offline_contact_guidance_term(
            contact_export_root=_AS_KEEP169_CONTACT_EXPORT_ROOT,
            wrist_weight=5.0,
            contact_weight=10.0,
            force_threshold=1.0,
        ),
    }
)

_FORWARD_AFTER_LIFT_TRACKING_TERM_NAMES = {
    "motion_global_ref_position_error_exp",
    "motion_global_ref_orientation_error_exp",
    "motion_relative_body_position_error_exp",
    "motion_relative_body_orientation_error_exp",
    "motion_global_body_lin_vel",
    "motion_global_body_ang_vel",
    "object_global_ref_position_error_exp",
    "object_global_ref_orientation_error_exp",
}


def _hybrid_stage2_tracking_term(term: RewardTermCfg) -> RewardTermCfg:
    return RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:hybrid_stage2_tracking_reward",
        params={
            "reward_func": term.func.rsplit(":", 1)[-1],
            "reward_params": dict(term.params),
        },
        weight=term.weight,
        tags=[*term.tags, "hybrid_stage2_tracking"],
    )


def _hybrid_velocity_tracking_term(term: RewardTermCfg) -> RewardTermCfg:
    return RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:hybrid_velocity_tracking_reward",
        params={
            "reward_func": term.func.rsplit(":", 1)[-1],
            "reward_params": dict(term.params),
        },
        weight=term.weight,
        tags=[*term.tags, "hybrid_velocity_tracking"],
    )


g1_29dof_wbt_reward_w_object_hybrid_stage2 = RewardManagerCfg(
    terms={
        **{
            name: (
                _hybrid_stage2_tracking_term(term)
                if name in _FORWARD_AFTER_LIFT_TRACKING_TERM_NAMES
                else term
            )
            for name, term in (
                g1_29dof_wbt_reward_w_object_generalist_offline_contact_guidance.terms.items()
            )
        },
        "hybrid_task_forward_velocity": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hybrid_stage2_forward_velocity_exp",
            params={"target_velocity": 0.5, "sigma": 0.25},
            weight=2.5,
            tags=["hybrid_stage2_task", "straight_line"],
        ),
        "hybrid_task_lateral_velocity": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hybrid_stage2_lateral_velocity_exp",
            params={"sigma": 0.20},
            weight=1.0,
            tags=["hybrid_stage2_task", "straight_line"],
        ),
        "hybrid_task_heading": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hybrid_stage2_heading_error_exp",
            params={"sigma": 0.25},
            weight=2.0,
            tags=["hybrid_stage2_task", "straight_line"],
        ),
        "hybrid_task_cross_track": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hybrid_stage2_cross_track_error_exp",
            params={"sigma": 0.15},
            weight=2.0,
            tags=["hybrid_stage2_task", "straight_line"],
        ),
        "hybrid_task_yaw_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hybrid_stage2_yaw_rate_exp",
            params={"sigma": 0.30},
            weight=1.0,
            tags=["hybrid_stage2_task", "straight_line"],
        ),
        "hybrid_task_object_position_hold": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hybrid_stage2_object_position_hold_exp",
            params={"sigma": 0.12},
            weight=2.0,
            tags=["hybrid_stage2_task", "object_hold"],
        ),
        "hybrid_task_object_orientation_hold": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hybrid_stage2_object_orientation_hold_exp",
            params={"sigma": 0.35},
            weight=0.5,
            tags=["hybrid_stage2_task", "object_hold"],
        ),
    }
)

g1_29dof_wbt_reward_w_object_generalist_tracking_no_contact = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward_w_object_generalist_offline_contact_guidance.terms,
        "offline_contact_guidance": replace(
            g1_29dof_wbt_reward_w_object_generalist_offline_contact_guidance.terms[
                "offline_contact_guidance"
            ],
            weight=0.0,
        ),
    }
)

g1_29dof_wbt_reward_w_object_hybrid_velocity = RewardManagerCfg(
    terms={
        **{
            name: (
                _hybrid_velocity_tracking_term(term)
                if name in _FORWARD_AFTER_LIFT_TRACKING_TERM_NAMES
                else term
            )
            for name, term in (
                g1_29dof_wbt_reward_w_object_generalist_tracking_no_contact.terms.items()
            )
        },
        "hybrid_velocity_lift_progress": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hybrid_velocity_lift_progress",
            weight=2.0,
            tags=["hybrid_velocity_task", "pickup"],
        ),
        "hybrid_velocity_forward": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hybrid_velocity_forward_velocity_exp",
            params={"sigma": 0.25},
            weight=2.0,
            tags=["hybrid_velocity_task", "command_tracking"],
        ),
        "hybrid_velocity_lateral": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hybrid_velocity_lateral_velocity_exp",
            params={"sigma": 0.20},
            weight=0.75,
            tags=["hybrid_velocity_task", "command_tracking"],
        ),
        "hybrid_velocity_heading": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hybrid_velocity_heading_error_exp",
            params={"sigma": 0.25},
            weight=1.0,
            tags=["hybrid_velocity_task", "straight_line"],
        ),
        "hybrid_velocity_cross_track": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hybrid_velocity_cross_track_error_exp",
            params={"sigma": 0.15},
            weight=1.0,
            tags=["hybrid_velocity_task", "straight_line"],
        ),
        "hybrid_velocity_yaw_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hybrid_velocity_yaw_rate_exp",
            params={"sigma": 0.30},
            weight=0.5,
            tags=["hybrid_velocity_task", "command_tracking"],
        ),
        "hybrid_velocity_object_position_hold": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hybrid_velocity_object_position_hold_exp",
            params={"sigma": 0.12},
            weight=1.5,
            tags=["hybrid_velocity_task", "object_hold"],
        ),
        "hybrid_velocity_object_orientation_hold": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hybrid_velocity_object_orientation_hold_exp",
            params={"sigma": 0.35},
            weight=0.5,
            tags=["hybrid_velocity_task", "object_hold"],
        ),
    }
)

g1_29dof_wbt_reward_w_object_r2s_contact_guidance = RewardManagerCfg(
    terms={
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=0.2,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.2,
        ),
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
            func="holosoma.managers.reward.terms.wbt:ObjectUndesiredContacts",
            params={
                "threshold": 1.0,
                "body_names": _FOOT_OBJECT_CONTACT_BODY_NAMES,
            },
            weight=-0.5,
        ),
        "lower_body_undesired_contacts": _lower_body_undesired_contacts_term(),
        "object_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_position_error_exp",
            params={"sigma": 0.7},
            weight=0.25,
        ),
        "object_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_orientation_error_exp",
            params={"sigma": 0.7},
            weight=0.25,
        ),
        "offline_contact_guidance": _fused_offline_contact_guidance_term(
            contact_export_root="outputs/clips",
            wrist_weight=3.0,
            contact_weight=4.0,
            force_threshold=1.4,
        ),
    }
)

g1_29dof_wbt_reward_w_object_r2s_rollout_reference_guidance = RewardManagerCfg(
    terms={
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={
                "sigma": 0.3,
                "rollout_reference_root": "outputs/clips",
            },
            weight=0.5,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={
                "sigma": 0.4,
                "rollout_reference_root": "outputs/clips",
            },
            weight=0.5,
        ),
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={
                "sigma": 0.3,
                "rollout_reference_root": "outputs/clips",
            },
            weight=1.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={
                "sigma": 0.4,
                "rollout_reference_root": "outputs/clips",
            },
            weight=1.0,
        ),
        "motion_global_body_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={
                "sigma": 1.0,
                "rollout_reference_root": "outputs/clips",
            },
            weight=1.0,
        ),
        "motion_global_body_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={
                "sigma": 3.14,
                "rollout_reference_root": "outputs/clips",
            },
            weight=1.0,
        ),
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            params={"max_penalty": -1.0},
            weight=-0.1,
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},
            weight=-10.0,
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:ObjectUndesiredContacts",
            params={
                "threshold": 1.0,
                "body_names": _FOOT_OBJECT_CONTACT_BODY_NAMES,
            },
            weight=-0.5,
        ),
        "lower_body_undesired_contacts": _lower_body_undesired_contacts_term(),
        "object_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_position_error_exp",
            params={
                "sigma": 0.3,
                "rollout_reference_root": "outputs/clips",
            },
            weight=1.0,
        ),
        "object_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_orientation_error_exp",
            params={
                "sigma": 0.4,
                "rollout_reference_root": "outputs/clips",
            },
            weight=1.0,
        ),
        "offline_contact_guidance": _fused_offline_contact_guidance_term(
            contact_export_root="outputs/clips",
            wrist_weight=4.0,
            contact_weight=7.0,
            force_threshold=1.4,
        ),
    }
)

__all__ = [
    "g1_29dof_wbt_fast_sac_reward",
    "g1_29dof_wbt_reward",
    "g1_29dof_wbt_reward_w_object",
    "g1_29dof_wbt_reward_w_object_generalist",
    "g1_29dof_wbt_reward_w_object_generalist_offline_contact_guidance",
    "g1_29dof_wbt_reward_w_object_generalist_tracking_no_contact",
    "g1_29dof_wbt_reward_w_object_hybrid_stage2",
    "g1_29dof_wbt_reward_w_object_hybrid_velocity",
    "g1_29dof_wbt_reward_w_object_r2s_contact_guidance",
    "g1_29dof_wbt_reward_w_object_r2s_rollout_reference_guidance",
    "g1_29dof_wbt_reward_w_object_extend",
]
