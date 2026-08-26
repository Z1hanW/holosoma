"""Whole Body Tracking observation presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg

DEFAULT_WBT_POLICY_HISTORY_LENGTH = 5
DEFAULT_WBT_DISTILL_POLICY_HISTORY_LENGTH = 1
DEFAULT_WBT_DISTILL_PROPRIO_HISTORY_LENGTH = 1

_WRIST_YAW_CONTACT_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
_ARM_LINK_CONTACT_BODY_NAMES = {
    "left_elbow": "left_elbow_link",
    "right_elbow": "right_elbow_link",
    "left_wrist_roll": "left_wrist_roll_link",
    "right_wrist_roll": "right_wrist_roll_link",
    "left_wrist_pitch": "left_wrist_pitch_link",
    "right_wrist_pitch": "right_wrist_pitch_link",
}
_TORSO_SUPPORT_CONTACT_BODY_NAMES = ["torso_link"]
_FOOT_CONTACT_BODY_NAMES = ["left_foot_contact_point", "right_foot_contact_point"]
_ANKLE_CONTACT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]

actor_obs_shared = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
    terms={
        "motion_command": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_command",
            scale=1.0,
            noise=0.0,
        ),
        "motion_ref_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
            scale=1.0,
            noise=0.05,
        ),
        "base_ang_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:base_ang_vel",
            scale=1.0,
            noise=0.2,
        ),
        "dof_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_pos",
            scale=1.0,
            noise=0.01,
        ),
        "dof_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_vel",
            scale=1.0,
            noise=0.5,
        ),
        "actions": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:actions",
            scale=1.0,
            noise=0.0,
        ),
    },
)

actor_obs_motion_tracking_terms = actor_obs_shared.terms.copy()
actor_obs_motion_tracking_terms["motion_future_target_poses"] = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:motion_future_target_poses",
    scale=1.0,
    noise=0.0,
)
actor_obs_motion_tracking = ObsGroupCfg(
    concatenate=actor_obs_shared.concatenate,
    enable_noise=actor_obs_shared.enable_noise,
    history_length=actor_obs_shared.history_length,
    terms=actor_obs_motion_tracking_terms,
)

actor_obs_w_object_terms = actor_obs_shared.terms.copy()
actor_obs_w_object_terms.update(
    {
        "obj_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ori_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_target_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_target_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_target_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_target_ori_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_size": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_size",
            scale=1.0,
            noise=0.0,
        ),
    }
)
actor_obs_w_object = ObsGroupCfg(
    concatenate=actor_obs_shared.concatenate,
    enable_noise=actor_obs_shared.enable_noise,
    history_length=actor_obs_shared.history_length,
    terms=actor_obs_w_object_terms,
)

# Privileged teacher-only variant.  The base linear velocity is available from
# simulator state and is intentionally not added to any student proprioception
# group.  Keep it as the final term so its three dimensions have an explicit,
# stable checkpoint contract without changing the legacy teacher layout.
actor_obs_w_object_teacher_linvel_terms = actor_obs_w_object_terms.copy()
actor_obs_w_object_teacher_linvel_terms["base_lin_vel"] = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:base_lin_vel",
    scale=1.0,
    noise=0.0,
)
actor_obs_w_object_teacher_linvel = replace(
    actor_obs_w_object,
    terms=actor_obs_w_object_teacher_linvel_terms,
)

actor_obs_w_object_legacy_terms = actor_obs_shared.terms.copy()
actor_obs_w_object_legacy_terms.update(
    {
        "obj_target_pose_size_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_target_pose_size_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ori_b",
            scale=1.0,
            noise=0.0,
        ),
    }
)

actor_obs_w_object_legacy = ObsGroupCfg(
    concatenate=actor_obs_shared.concatenate,
    enable_noise=actor_obs_shared.enable_noise,
    history_length=actor_obs_shared.history_length,
    terms=actor_obs_w_object_legacy_terms,
)

# Exact term order used by legacy teacher checkpoints such as 5vlz6pj8/model_24000.pt.
# Keep this separate from actor_obs_legacy so we can improve teacher compatibility
# without disturbing the student-facing observation structure.
actor_obs_w_object_teacher_compat_terms = {
    "actions": actor_obs_w_object_legacy_terms["actions"],
    "base_ang_vel": actor_obs_w_object_legacy_terms["base_ang_vel"],
    "dof_pos": actor_obs_w_object_legacy_terms["dof_pos"],
    "dof_vel": actor_obs_w_object_legacy_terms["dof_vel"],
    "motion_command": actor_obs_w_object_legacy_terms["motion_command"],
    "motion_ref_ori_b": actor_obs_w_object_legacy_terms["motion_ref_ori_b"],
    "obj_ori_b": actor_obs_w_object_legacy_terms["obj_ori_b"],
    "obj_pos_b": actor_obs_w_object_legacy_terms["obj_pos_b"],
    "obj_target_pose_size_b": actor_obs_w_object_legacy_terms["obj_target_pose_size_b"],
}

actor_obs_w_object_teacher_compat = ObsGroupCfg(
    concatenate=actor_obs_shared.concatenate,
    enable_noise=actor_obs_shared.enable_noise,
    history_length=actor_obs_shared.history_length,
    terms=actor_obs_w_object_teacher_compat_terms,
)

motion_future_target_poses_group = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "motion_future_target_poses": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_future_target_poses",
            scale=1.0,
            noise=0.0,
        )
    },
)

terrain_transformer_self_terms = {
    "motion_command": actor_obs_shared.terms["motion_command"],
    "motion_ref_ori_b": actor_obs_shared.terms["motion_ref_ori_b"],
    "base_ang_vel": actor_obs_shared.terms["base_ang_vel"],
    "dof_pos": actor_obs_shared.terms["dof_pos"],
    "dof_vel": actor_obs_shared.terms["dof_vel"],
    "actions_history": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:ActionsHistory",
        params={"history_steps": DEFAULT_WBT_POLICY_HISTORY_LENGTH},
        scale=1.0,
        noise=0.0,
    ),
}

terrain_transformer_target_terms = {
    "target_joints": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:target_joints",
        scale=1.0,
        noise=0.0,
    ),
    "target_root_roll": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:target_root_roll",
        scale=1.0,
        noise=0.0,
    ),
    "target_root_pitch": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:target_root_pitch",
        scale=1.0,
        noise=0.0,
    ),
}

critic_obs_shared_terms = {
    "motion_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_command",
        scale=1.0,
        noise=0.0,
    ),
    "motion_ref_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_pos_b",
        scale=1.0,
        noise=0.25,
    ),
    "motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
        scale=1.0,
        noise=0.05,
    ),
    "robot_body_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_pos_b",
        scale=1.0,
        noise=0.0,
    ),
    "robot_body_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_ori_b",
        scale=1.0,
        noise=0.0,
    ),
    "base_lin_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_lin_vel",
        scale=1.0,
        noise=0.0,
    ),
    "base_ang_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_ang_vel",
        scale=1.0,
        noise=0.2,
    ),
    "dof_pos": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_pos",
        scale=1.0,
        noise=0.01,
    ),
    "dof_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_vel",
        scale=1.0,
        noise=0.5,
    ),
    "actions": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:actions",
        scale=1.0,
        noise=0.0,
    ),
}

critic_obs_motion_tracking_terms = critic_obs_shared_terms.copy()
critic_obs_motion_tracking_terms["motion_future_target_poses"] = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:motion_future_target_poses",
    scale=1.0,
    noise=0.0,
)

critic_obs_w_object_terms = critic_obs_shared_terms.copy()
critic_obs_w_object_terms.update(
    {
        "obj_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ori_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_target_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_target_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_target_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_target_ori_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_size": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_size",
            scale=1.0,
            noise=0.0,
        ),
        "obj_lin_vel_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_lin_vel_b_v2",
            scale=1.0,
            noise=0.0,
        ),
    }
)

critic_obs_w_object_command_privileged_terms = critic_obs_w_object_terms.copy()
critic_obs_w_object_command_privileged_terms.update(
    {
        "obj_ang_vel_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ang_vel_b",
            scale=1.0,
            noise=0.0,
        ),
        "contact_prior_confidence": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:contact_prior_confidence",
            scale=1.0,
            noise=0.0,
        ),
        "left_wrist_contact_prior_occupancy": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:contact_prior_region_occupancy",
            params={"region_name": "left_wrist"},
            scale=1.0,
            noise=0.0,
        ),
        "right_wrist_contact_prior_occupancy": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:contact_prior_region_occupancy",
            params={"region_name": "right_wrist"},
            scale=1.0,
            noise=0.0,
        ),
        **{
            f"{region_name}_contact_prior_occupancy": ObsTermCfg(
                func="holosoma.managers.observation.terms.wbt:contact_prior_region_occupancy",
                params={"region_name": region_name},
                scale=1.0,
                noise=0.0,
            )
            for region_name in _ARM_LINK_CONTACT_BODY_NAMES
        },
        "torso_contact_prior_occupancy": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:contact_prior_region_occupancy",
            params={"region_name": "torso"},
            scale=1.0,
            noise=0.0,
        ),
        "left_wrist_contact_prior_force": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:contact_prior_region_force",
            params={"region_name": "left_wrist"},
            scale=1.0,
            noise=0.0,
        ),
        "right_wrist_contact_prior_force": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:contact_prior_region_force",
            params={"region_name": "right_wrist"},
            scale=1.0,
            noise=0.0,
        ),
        **{
            f"{region_name}_contact_prior_force": ObsTermCfg(
                func="holosoma.managers.observation.terms.wbt:contact_prior_region_force",
                params={"region_name": region_name},
                scale=1.0,
                noise=0.0,
            )
            for region_name in _ARM_LINK_CONTACT_BODY_NAMES
        },
        "torso_contact_prior_force": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:contact_prior_region_force",
            params={"region_name": "torso"},
            scale=1.0,
            noise=0.0,
        ),
        "left_wrist_contact_prior_pos_obj": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:contact_prior_region_pos_obj",
            params={"region_name": "left_wrist"},
            scale=1.0,
            noise=0.0,
        ),
        "right_wrist_contact_prior_pos_obj": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:contact_prior_region_pos_obj",
            params={"region_name": "right_wrist"},
            scale=1.0,
            noise=0.0,
        ),
        **{
            f"{region_name}_contact_prior_pos_obj": ObsTermCfg(
                func="holosoma.managers.observation.terms.wbt:contact_prior_region_pos_obj",
                params={"region_name": region_name},
                scale=1.0,
                noise=0.0,
            )
            for region_name in _ARM_LINK_CONTACT_BODY_NAMES
        },
        "torso_contact_prior_pos_obj": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:contact_prior_region_pos_obj",
            params={"region_name": "torso"},
            scale=1.0,
            noise=0.0,
        ),
        "left_wrist_object_contact_force": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:body_contact_force_magnitude",
            params={"body_names": ["left_wrist_yaw_link"], "object_only": True, "reduction": "max"},
            scale=1.0,
            noise=0.0,
        ),
        "right_wrist_object_contact_force": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:body_contact_force_magnitude",
            params={"body_names": ["right_wrist_yaw_link"], "object_only": True, "reduction": "max"},
            scale=1.0,
            noise=0.0,
        ),
        "left_wrist_object_contact_flag": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:body_contact_binary_flag",
            params={"body_names": ["left_wrist_yaw_link"], "object_only": True, "threshold": 1.0},
            scale=1.0,
            noise=0.0,
        ),
        "right_wrist_object_contact_flag": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:body_contact_binary_flag",
            params={"body_names": ["right_wrist_yaw_link"], "object_only": True, "threshold": 1.0},
            scale=1.0,
            noise=0.0,
        ),
        **{
            f"{region_name}_object_contact_force": ObsTermCfg(
                func="holosoma.managers.observation.terms.wbt:body_contact_force_magnitude",
                params={"body_names": [body_name], "object_only": True, "reduction": "max"},
                scale=1.0,
                noise=0.0,
            )
            for region_name, body_name in _ARM_LINK_CONTACT_BODY_NAMES.items()
        },
        **{
            f"{region_name}_object_contact_flag": ObsTermCfg(
                func="holosoma.managers.observation.terms.wbt:body_contact_binary_flag",
                params={"body_names": [body_name], "object_only": True, "threshold": 1.0},
                scale=1.0,
                noise=0.0,
            )
            for region_name, body_name in _ARM_LINK_CONTACT_BODY_NAMES.items()
        },
        "torso_object_contact_force": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:body_contact_force_magnitude",
            params={"body_names": _TORSO_SUPPORT_CONTACT_BODY_NAMES, "object_only": True, "reduction": "max"},
            scale=1.0,
            noise=0.0,
        ),
        "torso_object_contact_flag": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:body_contact_binary_flag",
            params={"body_names": _TORSO_SUPPORT_CONTACT_BODY_NAMES, "object_only": True, "threshold": 1.0},
            scale=1.0,
            noise=0.0,
        ),
        "feet_object_contact_force": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:body_contact_force_magnitude",
            params={"body_names": _FOOT_CONTACT_BODY_NAMES, "object_only": True, "reduction": "max"},
            scale=1.0,
            noise=0.0,
        ),
        "feet_object_contact_flag": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:body_contact_binary_flag",
            params={"body_names": _FOOT_CONTACT_BODY_NAMES, "object_only": True, "threshold": 1.0},
            scale=1.0,
            noise=0.0,
        ),
        "ankle_object_contact_force": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:body_contact_force_magnitude",
            params={"body_names": _ANKLE_CONTACT_BODY_NAMES, "object_only": True, "reduction": "max"},
            scale=1.0,
            noise=0.0,
        ),
        "ankle_object_contact_flag": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:body_contact_binary_flag",
            params={"body_names": _ANKLE_CONTACT_BODY_NAMES, "object_only": True, "threshold": 1.0},
            scale=1.0,
            noise=0.0,
        ),
        "feet_support_contact_force": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:body_contact_force_magnitude",
            params={"body_names": _FOOT_CONTACT_BODY_NAMES, "non_object_only": True, "reduction": "max"},
            scale=1.0,
            noise=0.0,
        ),
        "feet_support_contact_flag": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:body_contact_binary_flag",
            params={"body_names": _FOOT_CONTACT_BODY_NAMES, "non_object_only": True, "threshold": 1.0},
            scale=1.0,
            noise=0.0,
        ),
    }
)

critic_obs_w_object_command_distill_terms = critic_obs_w_object_terms.copy()
critic_obs_w_object_command_distill_terms.pop("actions")
critic_obs_w_object_command_distill_terms.update(
    {
        "obj_ang_vel_b": critic_obs_w_object_command_privileged_terms["obj_ang_vel_b"],
    }
)

critic_obs_w_object_command_distill_legacy_target_terms = {
    "dof_pos": critic_obs_shared_terms["dof_pos"],
    "dof_vel": critic_obs_shared_terms["dof_vel"],
    "obj_ori_b": critic_obs_w_object_terms["obj_ori_b"],
    "obj_pos_b": critic_obs_w_object_terms["obj_pos_b"],
    "base_ang_vel": critic_obs_shared_terms["base_ang_vel"],
    "base_lin_vel": critic_obs_shared_terms["base_lin_vel"],
    "obj_ang_vel_b": critic_obs_w_object_command_privileged_terms["obj_ang_vel_b"],
    "obj_lin_vel_b": critic_obs_w_object_terms["obj_lin_vel_b"],
    "motion_command": critic_obs_shared_terms["motion_command"],
    "motion_ref_ori_b": critic_obs_shared_terms["motion_ref_ori_b"],
    "motion_ref_pos_b": critic_obs_shared_terms["motion_ref_pos_b"],
    "robot_body_ori_b": critic_obs_shared_terms["robot_body_ori_b"],
    "robot_body_pos_b": critic_obs_shared_terms["robot_body_pos_b"],
    "obj_target_pose_size_b": actor_obs_w_object_legacy_terms["obj_target_pose_size_b"],
}

g1_29dof_wbt_observation = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
    },
)

g1_29dof_wbt_observation_motion_tracking = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_motion_tracking,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_motion_tracking_terms,
        ),
    },
)

g1_29dof_wbt_observation_motion_tracking_split = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
        "motion_future_target_poses": motion_future_target_poses_group,
    },
)

g1_29dof_wbt_observation_terrain_transformer = ObservationManagerCfg(
    groups={
        "actor_obs_self": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=terrain_transformer_self_terms,
        ),
        "actor_obs_target": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=terrain_transformer_target_terms,
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_w_object,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
            terms=critic_obs_w_object_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object_teacher_linvel = replace(
    g1_29dof_wbt_observation_w_object,
    groups={
        **g1_29dof_wbt_observation_w_object.groups,
        "actor_obs": actor_obs_w_object_teacher_linvel,
    },
)

g1_29dof_wbt_observation_w_object_legacy = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_w_object_legacy,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
            terms=critic_obs_w_object_terms,
        ),
    },
)

object_distill_sparse_root_cmd_terms = {
    # Sim2real default: do NOT expose clip_phase to the student.
    # Keep sparse target root trajectory only.
    "sparse_target_root_trajectory_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:sparse_target_root_trajectory_command",
        scale=1.0,
        noise=0.0,
    ),
}

object_distill_sparse_root_cmd_terms_contact_aware = {
    "sparse_target_root_trajectory_command_contact_aware": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:sparse_target_root_trajectory_command_contact_aware",
        scale=1.0,
        noise=0.0,
    ),
}

object_distill_drop_button_terms = {
    "drop_button": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:drop_button",
        scale=1.0,
        noise=0.0,
    ),
}

object_distill_pickup_button_terms = {
    "pickup_button": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:pickup_button",
        scale=1.0,
        noise=0.0,
    ),
}

# Legacy distill torso observation (kept only for backward compatibility):
# includes clip_phase and should not be used for sim2real-oriented training.
object_distill_sparse_root_cmd_terms_legacy = {
    "sparse_target_root_trajectory_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:sparse_target_root_trajectory_command",
        scale=1.0,
        noise=0.0,
    ),
    "clip_phase": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:clip_phase",
        scale=1.0,
        noise=0.0,
    ),
}

object_distill_sparse_root_cmd_terms_contact_aware_legacy = {
    "sparse_target_root_trajectory_command_contact_aware": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:sparse_target_root_trajectory_command_contact_aware",
        scale=1.0,
        noise=0.0,
    ),
    "clip_phase": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:clip_phase",
        scale=1.0,
        noise=0.0,
    ),
}

object_distill_proprio_terms = {
    "base_lin_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_lin_vel",
        scale=1.0,
        noise=0.0,
    ),
    "base_ang_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_ang_vel",
        scale=1.0,
        noise=0.2,
    ),
    "dof_pos": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_pos",
        scale=1.0,
        noise=0.01,
    ),
    "dof_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_vel",
        scale=1.0,
        noise=0.5,
    ),
    "actions": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:actions",
        scale=1.0,
        noise=0.0,
    ),
}

object_distill_proprio_terms_no_linvel = object_distill_proprio_terms.copy()
object_distill_proprio_terms_no_linvel.pop("base_lin_vel")

object_distill_proprio_history_terms = object_distill_proprio_terms.copy()
object_distill_proprio_history_terms.pop("actions")

object_distill_proprio_history_terms_no_linvel = object_distill_proprio_terms_no_linvel.copy()
object_distill_proprio_history_terms_no_linvel.pop("actions")

object_distill_action_terms = {
    "actions": object_distill_proprio_terms["actions"],
}

object_distill_box_terms = {
    "obj_current_pose_size_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_current_pose_size_b",
        scale=1.0,
        noise=0.0,
    ),
}

object_distill_drop_terms = {
    "obj_goal_xy_pick_root_heading": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_goal_xy_pick_root_heading",
        params={
            "lift_height_threshold": 0.10,
            "lift_ratio_threshold": 0.35,
            "consecutive_steps": 5,
        },
        scale=1.0,
        noise=0.0,
    ),
}

g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd = ObservationManagerCfg(
    reuse_exact_base_terms=True,
    groups={
        # Keep full teacher actor observation available for teacher policy queries.
        "actor_obs": replace(actor_obs_w_object, history_length=1),
        # Legacy teacher observation group (without object velocities).
        "actor_obs_legacy": replace(actor_obs_w_object_legacy, history_length=1),
        # Exact-order teacher compatibility group for legacy checkpoints.
        "actor_obs_teacher_compat": replace(actor_obs_w_object_teacher_compat, history_length=1),
        # Student sparse root-trajectory command.
        "actor_obs_root": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_POLICY_HISTORY_LENGTH,
            terms=object_distill_sparse_root_cmd_terms,
        ),
        # Backward-compatible alias; semantics are root-relative, not torso-relative.
        "actor_obs_torso": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_POLICY_HISTORY_LENGTH,
            terms=object_distill_sparse_root_cmd_terms,
        ),
        "actor_obs_root_contact_aware": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_POLICY_HISTORY_LENGTH,
            terms=object_distill_sparse_root_cmd_terms_contact_aware,
        ),
        "actor_obs_torso_contact_aware": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_POLICY_HISTORY_LENGTH,
            terms=object_distill_sparse_root_cmd_terms_contact_aware,
        ),
        "actor_obs_pickup_button": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_pickup_button_terms,
        ),
        "actor_obs_drop_button": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_drop_button_terms,
        ),
        # Student proprioception state.
        "actor_obs_proprio": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_PROPRIO_HISTORY_LENGTH,
            terms=object_distill_proprio_history_terms,
        ),
        # Student proprioception state without base linear velocity.
        "actor_obs_proprio_no_linvel": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_PROPRIO_HISTORY_LENGTH,
            terms=object_distill_proprio_history_terms_no_linvel,
        ),
        # Student proprioception state with actions, without base linear velocity.
        "actor_obs_proprio_with_actions_no_linvel": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_PROPRIO_HISTORY_LENGTH,
            terms=object_distill_proprio_terms_no_linvel,
        ),
        "actor_obs_proprio_no_linvel_actions": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_PROPRIO_HISTORY_LENGTH,
            terms=object_distill_proprio_terms_no_linvel,
        ),
        "actor_obs_actions": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_action_terms,
        ),
        # Student object state: current object [pos(3), rot6d(6), size(3)] only.
        "actor_obs_box": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_POLICY_HISTORY_LENGTH,
            terms=object_distill_box_terms,
        ),
        # Student drop target: final object [dx, dy] in pickup-time pelvis-heading frame.
        # Keep this single-frame because it is already a frozen command.
        "actor_obs_drop": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_drop_terms,
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_command_distill_terms,
        ),
        "critic_proprio_history": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_PROPRIO_HISTORY_LENGTH,
            terms=object_distill_proprio_history_terms,
        ),
        "critic_actions": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_action_terms,
        ),
    },
)

# Pure-RL critic contract without the redundant one-frame proprio copy.  The
# base critic_obs group already contains the current base linear/angular
# velocity and joint position/velocity, while critic_actions retains the true
# previous-action signal.  Keep this as a separate preset so existing 377D and
# 381D runs remain immutable and resumable under their original contracts.
g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_critic317 = replace(
    g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd,
    groups={
        group_name: group_cfg
        for group_name, group_cfg in
        g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd.groups.items()
        if group_name != "critic_proprio_history"
    },
)

hmi_goal_command_terms = {
    "hmi_object_goal_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:hmi_object_goal_command",
        scale=1.0,
        noise=0.0,
    ),
}

hmi_zero_drop_button_terms = {
    "hmi_zero_drop_button": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:hmi_zero_drop_button",
        scale=1.0,
        noise=0.0,
    ),
}

hmi_critic_terms = critic_obs_w_object_command_distill_terms.copy()
hmi_critic_terms["motion_command"] = replace(
    hmi_critic_terms["motion_command"],
    func="holosoma.managers.observation.terms.wbt:hmi_masked_motion_command",
)
hmi_critic_terms["motion_ref_pos_b"] = replace(
    hmi_critic_terms["motion_ref_pos_b"],
    func="holosoma.managers.observation.terms.wbt:hmi_masked_motion_ref_pos_b",
)
hmi_critic_terms["motion_ref_ori_b"] = replace(
    hmi_critic_terms["motion_ref_ori_b"],
    func="holosoma.managers.observation.terms.wbt:hmi_masked_motion_ref_ori_b",
)
hmi_critic_terms["obj_target_pos_b"] = replace(
    hmi_critic_terms["obj_target_pos_b"],
    func="holosoma.managers.observation.terms.wbt:hmi_masked_obj_target_pos_b",
)
hmi_critic_terms["obj_target_ori_b"] = replace(
    hmi_critic_terms["obj_target_ori_b"],
    func="holosoma.managers.observation.terms.wbt:hmi_masked_obj_target_ori_b",
)

g1_29dof_wbt_observation_w_object_hmi_depth = replace(
    g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_critic317,
    groups={
        **g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_critic317.groups,
        "actor_obs_hmi_goal_command": replace(
            g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_critic317.groups[
                "actor_obs_root_contact_aware"
            ],
            terms=hmi_goal_command_terms,
        ),
        "actor_obs_drop_button": replace(
            g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_critic317.groups[
                "actor_obs_drop_button"
            ],
            terms=hmi_zero_drop_button_terms,
        ),
        "critic_obs": replace(
            g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_critic317.groups[
                "critic_obs"
            ],
            terms=hmi_critic_terms,
        ),
    },
)

hybrid_stage2_critic_terms = critic_obs_w_object_command_distill_terms.copy()
hybrid_stage2_critic_terms["hybrid_stage2_task_indicator"] = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:hybrid_stage2_task_indicator",
    scale=1.0,
    noise=0.0,
)

g1_29dof_wbt_observation_w_object_hybrid_stage2 = replace(
    g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd,
    groups={
        **g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd.groups,
        "critic_obs": replace(
            g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd.groups["critic_obs"],
            terms=hybrid_stage2_critic_terms,
        ),
    },
)

hybrid_velocity_command_terms = {
    "hybrid_velocity_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:hybrid_velocity_command",
        scale=1.0,
        noise=0.0,
    ),
}

policy_world_velocity_command_terms = {
    "target_root_world_velocity_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:target_root_world_velocity_command",
        scale=1.0,
        noise=0.0,
    ),
}

policy_world_root_error_command_terms = {
    "target_root_world_xy_yaw_error_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:target_root_world_xy_yaw_error_command",
        scale=1.0,
        noise=0.0,
    ),
}

hybrid_velocity_critic_terms = critic_obs_w_object_command_distill_terms.copy()
hybrid_velocity_critic_terms["motion_command"] = replace(
    hybrid_velocity_critic_terms["motion_command"],
    func="holosoma.managers.observation.terms.wbt:hybrid_velocity_masked_motion_command",
)
hybrid_velocity_critic_terms["motion_ref_pos_b"] = replace(
    hybrid_velocity_critic_terms["motion_ref_pos_b"],
    func="holosoma.managers.observation.terms.wbt:hybrid_velocity_masked_motion_ref_pos_b",
)
hybrid_velocity_critic_terms["motion_ref_ori_b"] = replace(
    hybrid_velocity_critic_terms["motion_ref_ori_b"],
    func="holosoma.managers.observation.terms.wbt:hybrid_velocity_masked_motion_ref_ori_b",
)
hybrid_velocity_critic_terms["obj_target_pos_b"] = replace(
    hybrid_velocity_critic_terms["obj_target_pos_b"],
    func="holosoma.managers.observation.terms.wbt:hybrid_velocity_masked_obj_target_pos_b",
)
hybrid_velocity_critic_terms["obj_target_ori_b"] = replace(
    hybrid_velocity_critic_terms["obj_target_ori_b"],
    func="holosoma.managers.observation.terms.wbt:hybrid_velocity_masked_obj_target_ori_b",
)
hybrid_velocity_critic_terms["hybrid_velocity_command"] = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:hybrid_velocity_command",
    scale=1.0,
    noise=0.0,
)
hybrid_velocity_critic_terms["hybrid_velocity_task_indicator"] = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:hybrid_velocity_task_indicator",
    scale=1.0,
    noise=0.0,
)

g1_29dof_wbt_observation_w_object_hybrid_velocity = replace(
    g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd,
    groups={
        **g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd.groups,
        **{
            group_name: replace(
                g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd.groups[group_name],
                terms=hybrid_velocity_command_terms,
            )
            for group_name in (
                "actor_obs_root",
                "actor_obs_torso",
                "actor_obs_root_contact_aware",
                "actor_obs_torso_contact_aware",
            )
        },
        "critic_obs": replace(
            g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd.groups["critic_obs"],
            terms=hybrid_velocity_critic_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object_policy_world_velocity = replace(
    g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd,
    groups={
        **g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd.groups,
        "actor_obs_world_velocity_command": replace(
            g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd.groups[
                "actor_obs_root_contact_aware"
            ],
            terms=policy_world_velocity_command_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object_policy_world_root_error = replace(
    g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd,
    groups={
        **g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd.groups,
        "actor_obs_world_root_error_command": replace(
            g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd.groups[
                "actor_obs_root_contact_aware"
            ],
            terms=policy_world_root_error_command_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object_hybrid_world_velocity = replace(
    g1_29dof_wbt_observation_w_object_hybrid_velocity,
    groups={
        **g1_29dof_wbt_observation_w_object_hybrid_velocity.groups,
        "actor_obs_hybrid_world_velocity_command": replace(
            g1_29dof_wbt_observation_w_object_hybrid_velocity.groups[
                "actor_obs_root_contact_aware"
            ],
            terms=hybrid_velocity_command_terms,
        ),
    },
)

# Compatibility preset for distilling a teacher trained with privileged
# base_lin_vel.  Only the teacher query group changes; all student actor groups
# remain byte-for-byte equivalent to the ordinary sparse-root configuration.
g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_teacher_linvel = replace(
    g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd,
    groups={
        **g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd.groups,
        "actor_obs": replace(actor_obs_w_object_teacher_linvel, history_length=1),
    },
)

g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_legacy = ObservationManagerCfg(
    groups={
        # Keep full teacher actor observation available for teacher policy queries.
        "actor_obs": replace(actor_obs_w_object, history_length=1),
        # Legacy teacher observation group (without object velocities).
        "actor_obs_legacy": replace(actor_obs_w_object_legacy, history_length=1),
        # Student sparse root-trajectory command (legacy includes clip_phase).
        "actor_obs_root": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_POLICY_HISTORY_LENGTH,
            terms=object_distill_sparse_root_cmd_terms_legacy,
        ),
        # Backward-compatible alias; semantics are root-relative, not torso-relative.
        "actor_obs_torso": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_POLICY_HISTORY_LENGTH,
            terms=object_distill_sparse_root_cmd_terms_legacy,
        ),
        "actor_obs_root_contact_aware": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_POLICY_HISTORY_LENGTH,
            terms=object_distill_sparse_root_cmd_terms_contact_aware_legacy,
        ),
        "actor_obs_torso_contact_aware": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_POLICY_HISTORY_LENGTH,
            terms=object_distill_sparse_root_cmd_terms_contact_aware_legacy,
        ),
        "actor_obs_pickup_button": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_pickup_button_terms,
        ),
        "actor_obs_drop_button": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_drop_button_terms,
        ),
        # Student proprioception state.
        "actor_obs_proprio": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_PROPRIO_HISTORY_LENGTH,
            terms=object_distill_proprio_history_terms,
        ),
        "actor_obs_proprio_no_linvel": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_PROPRIO_HISTORY_LENGTH,
            terms=object_distill_proprio_history_terms_no_linvel,
        ),
        "actor_obs_proprio_with_actions_no_linvel": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_PROPRIO_HISTORY_LENGTH,
            terms=object_distill_proprio_terms_no_linvel,
        ),
        "actor_obs_proprio_no_linvel_actions": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_PROPRIO_HISTORY_LENGTH,
            terms=object_distill_proprio_terms_no_linvel,
        ),
        "actor_obs_actions": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_action_terms,
        ),
        # Student object state: current object [pos(3), rot6d(6), size(3)] only.
        "actor_obs_box": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_POLICY_HISTORY_LENGTH,
            terms=object_distill_box_terms,
        ),
        "actor_obs_drop": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_drop_terms,
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_POLICY_HISTORY_LENGTH,
            terms=critic_obs_w_object_command_privileged_terms,
        ),
        "critic_proprio_history": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_PROPRIO_HISTORY_LENGTH,
            terms=object_distill_proprio_history_terms,
        ),
        "critic_actions": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_action_terms,
        ),
    },
)

# VideoMimic-style observation: history for torso signals + target pose terms.
actor_obs_videomimic_terms = {
    "torso_real": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:torso_real",
        scale=1.0,
        noise=0.0,
    ),
    "torso_xy_rel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:torso_xy_rel",
        scale=1.0,
        noise=0.0,
    ),
    "torso_yaw_rel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:torso_yaw_rel",
        scale=1.0,
        noise=0.0,
    ),
}

critic_obs_videomimic_terms = actor_obs_videomimic_terms.copy()
critic_obs_videomimic_terms["base_lin_vel"] = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:base_lin_vel",
    scale=1.0,
    noise=0.0,
)

videomimic_target_terms = {
    "target_joints": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:target_joints",
        scale=1.0,
        noise=0.0,
    ),
    "target_root_roll": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:target_root_roll",
        scale=1.0,
        noise=0.0,
    ),
    "target_root_pitch": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:target_root_pitch",
        scale=1.0,
        noise=0.0,
    ),
}

g1_29dof_wbt_observation_terrain_distill_sparse_root_cmd = ObservationManagerCfg(
    groups={
        # Keep the teacher videomimic groups available for teacher policy queries.
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
            terms=actor_obs_videomimic_terms,
        ),
        "actor_obs_target": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=videomimic_target_terms,
        ),
        # Student sparse root command [rel_xy(2), rel_yaw(1)] in root-heading frame.
        "actor_obs_root": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_POLICY_HISTORY_LENGTH,
            terms=object_distill_sparse_root_cmd_terms,
        ),
        # Student proprioception only; no privileged target poses in the actor.
        "actor_obs_proprio": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_PROPRIO_HISTORY_LENGTH,
            terms=object_distill_proprio_history_terms,
        ),
        "actor_obs_actions": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_action_terms,
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_videomimic_terms,
        ),
        "critic_obs_target": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=videomimic_target_terms,
        ),
    },
)

g1_29dof_wbt_observation_videomimic = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
            terms=actor_obs_videomimic_terms,
        ),
        "actor_obs_target": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=videomimic_target_terms,
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=5,
            terms=critic_obs_videomimic_terms,
        ),
        "critic_obs_target": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=videomimic_target_terms,
        ),
    },
)

__all__ = [
    "g1_29dof_wbt_observation",
    "g1_29dof_wbt_observation_motion_tracking",
    "g1_29dof_wbt_observation_motion_tracking_split",
    "g1_29dof_wbt_observation_terrain_transformer",
    "g1_29dof_wbt_observation_terrain_distill_sparse_root_cmd",
    "g1_29dof_wbt_observation_w_object",
    "g1_29dof_wbt_observation_w_object_teacher_linvel",
    "g1_29dof_wbt_observation_w_object_legacy",
    "g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd",
    "g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_critic317",
    "g1_29dof_wbt_observation_w_object_hmi_depth",
    "g1_29dof_wbt_observation_w_object_hybrid_stage2",
    "g1_29dof_wbt_observation_w_object_hybrid_velocity",
    "g1_29dof_wbt_observation_w_object_hybrid_world_velocity",
    "g1_29dof_wbt_observation_w_object_policy_world_velocity",
    "g1_29dof_wbt_observation_w_object_policy_world_root_error",
    "g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_teacher_linvel",
    "g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_legacy",
    "g1_29dof_wbt_observation_videomimic",
]
