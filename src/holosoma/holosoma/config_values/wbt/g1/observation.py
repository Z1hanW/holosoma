"""Whole Body Tracking observation presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg

DEFAULT_WBT_POLICY_HISTORY_LENGTH = 5
DEFAULT_WBT_DISTILL_POLICY_HISTORY_LENGTH = 1
DEFAULT_WBT_DISTILL_PROPRIO_HISTORY_LENGTH = 1

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
        "obj_lin_vel_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_lin_vel_b",
            scale=1.0,
            noise=0.0,
        ),
    }
)

critic_obs_w_object_generalist_terms = critic_obs_w_object_terms.copy()
critic_obs_w_object_generalist_terms.update(
    {
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

actor_obs_w_object_generalist_terms = actor_obs_shared.terms.copy()
actor_obs_w_object_generalist_terms.update(
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

actor_obs_w_object_generalist = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
    terms=actor_obs_w_object_generalist_terms,
)

object_distill_sparse_root_cmd_terms = {
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

object_distill_proprio_terms_no_linvel = {
    "base_ang_vel": actor_obs_shared.terms["base_ang_vel"],
    "dof_pos": actor_obs_shared.terms["dof_pos"],
    "dof_vel": actor_obs_shared.terms["dof_vel"],
    "actions": actor_obs_shared.terms["actions"],
}

object_distill_box_terms = {
    "obj_current_pose_size_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_current_pose_size_b",
        scale=1.0,
        noise=0.0,
    ),
}

object_distill_depth_terms = {
    "object_depth_map_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:object_depth_map_b",
        params={
            "height": 17,
            "width": 17,
            "near": 0.15,
            "max_distance": 3.0,
            "normalize": True,
        },
        scale=1.0,
        noise=0.0,
        clip=(-0.5, 0.5),
    ),
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

g1_29dof_wbt_observation_w_object = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object_generalist = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_w_object_generalist,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_POLICY_HISTORY_LENGTH,
            terms=critic_obs_w_object_generalist_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd = ObservationManagerCfg(
    groups={
        "actor_obs": replace(actor_obs_w_object_generalist, history_length=1),
        "actor_obs_root": ObsGroupCfg(
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
        "actor_obs_proprio_with_actions_no_linvel": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=DEFAULT_WBT_DISTILL_PROPRIO_HISTORY_LENGTH,
            terms=object_distill_proprio_terms_no_linvel,
        ),
        "actor_obs_box": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_box_terms,
        ),
        "perception_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=object_distill_depth_terms,
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_generalist_terms,
        ),
    },
)

__all__ = [
    "g1_29dof_wbt_observation",
    "g1_29dof_wbt_observation_w_object",
    "g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd",
    "g1_29dof_wbt_observation_w_object_generalist",
]
