"""Whole Body Tracking observation presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg

DEFAULT_WBT_POLICY_HISTORY_LENGTH = 10

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
            history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
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
    "obj_current_pose_size_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_current_pose_size_b",
        scale=1.0,
        noise=0.0,
    ),
}

object_distill_drop_terms = {
    "obj_goal_xy_yaw_pick_root_heading": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_goal_xy_yaw_pick_root_heading",
        params={
            "lift_height_threshold": 0.10,
            "lift_ratio_threshold": 0.35,
            "consecutive_steps": 5,
        },
        scale=1.0,
        noise=0.0,
    ),
}

object_distill_drop_mixed_terms = {
    "obj_sparse_goal_xy_yaw_pick_root_heading": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_sparse_goal_xy_yaw_pick_root_heading",
        scale=1.0,
        noise=0.0,
    ),
    "obj_picked_flag": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_picked_flag",
        scale=1.0,
        noise=0.0,
    ),
}

object_command_curriculum_track_terms = {
    "motion_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:command_curriculum_motion_command",
        scale=1.0,
        noise=0.0,
    ),
    "motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:command_curriculum_motion_ref_ori_b",
        scale=1.0,
        noise=0.0,
    ),
    "obj_target_pose_size_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:command_curriculum_obj_target_pose_size_b",
        scale=1.0,
        noise=0.0,
    ),
}

object_command_curriculum_goal_terms = {
    "obj_sparse_goal_xy_yaw_pick_root_heading": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:command_curriculum_obj_sparse_goal_xy_yaw_pick_root_heading",
        scale=1.0,
        noise=0.0,
    ),
    "obj_picked_flag": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:command_curriculum_obj_picked_flag",
        scale=1.0,
        noise=0.0,
    ),
}

object_command_curriculum_mode_terms = {
    "command_only_flag": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:command_curriculum_command_only_flag",
        scale=1.0,
        noise=0.0,
    ),
}

g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd = ObservationManagerCfg(
    groups={
        # Keep full teacher actor observation available for teacher policy queries.
        "actor_obs": replace(actor_obs_w_object, history_length=1),
        # Legacy teacher observation group (without object velocities).
        "actor_obs_legacy": replace(actor_obs_w_object_legacy, history_length=1),
        # Student sparse root-trajectory command.
        "actor_obs_root": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
            terms=object_distill_sparse_root_cmd_terms,
        ),
        # Backward-compatible alias; semantics are root-relative, not torso-relative.
        "actor_obs_torso": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
            terms=object_distill_sparse_root_cmd_terms,
        ),
        # Student proprioception state.
        "actor_obs_proprio": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
            terms=object_distill_proprio_terms,
        ),
        # Student object state: current object [pos(3), rot6d(6), size(3)] only.
        "actor_obs_box": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
            terms=object_distill_box_terms,
        ),
        # Student drop target: final object [dx, dy, dyaw] in pickup-time pelvis-heading frame.
        # Keep this single-frame because it is already a frozen command.
        "actor_obs_drop": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_drop_terms,
        ),
        "actor_obs_drop_mixed": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_drop_mixed_terms,
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
            terms=critic_obs_w_object_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object_command_curriculum = ObservationManagerCfg(
    groups={
        # Keep the legacy full observation around for debugging/analysis.
        "actor_obs": replace(actor_obs_w_object, history_length=1),
        "actor_obs_track": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
            terms=object_command_curriculum_track_terms,
        ),
        "actor_obs_proprio": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
            terms=object_distill_proprio_terms,
        ),
        "actor_obs_box": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
            terms=object_distill_box_terms,
        ),
        "actor_obs_goal": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_command_curriculum_goal_terms,
        ),
        "actor_obs_mode": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_command_curriculum_mode_terms,
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
            terms=critic_obs_w_object_terms,
        ),
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
            history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
            terms=object_distill_sparse_root_cmd_terms_legacy,
        ),
        # Backward-compatible alias; semantics are root-relative, not torso-relative.
        "actor_obs_torso": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
            terms=object_distill_sparse_root_cmd_terms_legacy,
        ),
        # Student proprioception state.
        "actor_obs_proprio": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
            terms=object_distill_proprio_terms,
        ),
        # Student object state: current object [pos(3), rot6d(6), size(3)] only.
        "actor_obs_box": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
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
            history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
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
    "g1_29dof_wbt_observation_w_object",
    "g1_29dof_wbt_observation_w_object_legacy",
    "g1_29dof_wbt_observation_w_object_command_curriculum",
    "g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd",
    "g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_legacy",
    "g1_29dof_wbt_observation_videomimic",
]
