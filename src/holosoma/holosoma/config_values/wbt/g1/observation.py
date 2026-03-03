"""Whole Body Tracking observation presets for the G1 robot."""

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg

actor_obs_shared = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
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
        "obj_lin_vel_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_lin_vel_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_ang_vel_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ang_vel_b",
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

actor_obs_w_object_legacy_terms = actor_obs_w_object_terms.copy()
actor_obs_w_object_legacy_terms.pop("obj_lin_vel_b", None)
actor_obs_w_object_legacy_terms.pop("obj_ang_vel_b", None)

actor_obs_w_object_legacy = ObsGroupCfg(
    concatenate=actor_obs_shared.concatenate,
    enable_noise=actor_obs_shared.enable_noise,
    history_length=actor_obs_shared.history_length,
    terms=actor_obs_w_object_legacy_terms,
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
        "obj_lin_vel_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_lin_vel_b",
            scale=1.0,
            noise=0.0,
        ),
    }
)

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

g1_29dof_wbt_observation_w_object = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_w_object,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object_legacy = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_w_object_legacy,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_terms,
        ),
    },
)

object_distill_torso_terms = {
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

object_distill_box_terms = {
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
    "obj_target_pose_size_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_target_pose_size_b",
        scale=1.0,
        noise=0.0,
    ),
}

object_distill_box_goal_terms = {
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
    "obj_goal_pos_size_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_goal_pos_size_b",
        scale=1.0,
        noise=0.0,
    ),
}

g1_29dof_wbt_observation_w_object_distill_torso_box = ObservationManagerCfg(
    groups={
        # Keep full teacher actor observation available for teacher policy queries.
        "actor_obs": actor_obs_w_object,
        # Student torso command state.
        "actor_obs_torso": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_torso_terms,
        ),
        # Student proprioception state (keep proprio; remove tracking pose terms only).
        "actor_obs_proprio": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_proprio_terms,
        ),
        # Student object-aware state (current object pose + target pose/size).
        "actor_obs_box": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_box_terms,
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object_distill_torso_box_goal = ObservationManagerCfg(
    groups={
        # Keep full teacher actor observation available for teacher policy queries.
        "actor_obs": actor_obs_w_object,
        # Student torso command state.
        "actor_obs_torso": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_torso_terms,
        ),
        # Student proprioception state (keep proprio; remove tracking pose terms only).
        "actor_obs_proprio": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_proprio_terms,
        ),
        # Student object state in robot base frame:
        # - current object pose (obj_pos_b)
        # - final clip goal position + size (obj_goal_pos_size_b)
        "actor_obs_box": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_box_goal_terms,
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_terms,
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

g1_29dof_wbt_observation_videomimic = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=5,
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

g1_29dof_wbt_observation_videomimic_distill = ObservationManagerCfg(
    groups={
        # Teacher-style actor obs (history on torso + goals) kept for distillation inputs.
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=5,
            terms=actor_obs_videomimic_terms,
        ),
        # Student actor obs: history on torso_real only.
        "actor_obs_torso": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=5,
            terms={
                "torso_real": actor_obs_videomimic_terms["torso_real"],
            },
        ),
        # Student actor obs: single-frame goals (no history on torso_xy_rel/yaw_rel).
        "actor_obs_goal": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                "torso_xy_rel": actor_obs_videomimic_terms["torso_xy_rel"],
                "torso_yaw_rel": actor_obs_videomimic_terms["torso_yaw_rel"],
            },
        ),
        # Keep target terms for teacher input (actor) and critic.
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
    "g1_29dof_wbt_observation_w_object",
    "g1_29dof_wbt_observation_w_object_legacy",
    "g1_29dof_wbt_observation_w_object_distill_torso_box",
    "g1_29dof_wbt_observation_w_object_distill_torso_box_goal",
    "g1_29dof_wbt_observation_videomimic",
    "g1_29dof_wbt_observation_videomimic_distill",
]
