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

height_scan_obs_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:height_scan",
    scale=1.0,
    noise=0.0,
)

depth_camera_obs_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:depth_camera",
    scale=1.0,
    noise=0.0,
)

actor_obs_with_height_scan = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    terms={
        **actor_obs_shared.terms,
        "height_scan": height_scan_obs_term,
    },
)

actor_obs_with_depth_camera = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    terms={
        **actor_obs_shared.terms,
        "depth_camera": depth_camera_obs_term,
    },
)

actor_height_scan = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "height_scan": height_scan_obs_term,
    },
)

critic_height_scan = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "height_scan": height_scan_obs_term,
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

critic_obs_shared_height_scan_terms = {
    **critic_obs_shared_terms,
    "height_scan": height_scan_obs_term,
}

critic_obs_shared_depth_camera_terms = {
    **critic_obs_shared_terms,
    "depth_camera": depth_camera_obs_term,
}

critic_obs_w_object_terms = critic_obs_shared_terms.copy()
object_current_obs_terms = {
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

object_obs_terms = {
    **object_current_obs_terms,
    "obj_ref_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_ref_pos_b",
        scale=1.0,
        noise=0.0,
    ),
    "obj_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_ref_ori_b",
        scale=1.0,
        noise=0.0,
    ),
    "obj_ref_lin_vel_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_ref_lin_vel_b",
        scale=1.0,
        noise=0.0,
    ),
}
critic_obs_w_object_terms.update(object_current_obs_terms)

object_next_target_obs_terms = {
    "obj_ref_pos_next_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_ref_pos_next_b",
        scale=1.0,
        noise=0.0,
    ),
    "obj_ref_ori_next_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_ref_ori_next_b",
        scale=1.0,
        noise=0.0,
    ),
    "obj_ref_lin_vel_next_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_ref_lin_vel_next_b",
        scale=1.0,
        noise=0.0,
    ),
}

object_obs_with_next_target_terms = {
    **object_obs_terms,
    **object_next_target_obs_terms,
}

actor_object_obs = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms=object_obs_terms,
)

critic_object_obs = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms=object_obs_terms,
)

actor_object_obs_with_next_target = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms=object_obs_with_next_target_terms,
)

critic_object_obs_with_next_target = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms=object_obs_with_next_target_terms,
)

critic_obs_w_object_height_scan_terms = {
    **critic_obs_w_object_terms,
    "height_scan": height_scan_obs_term,
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

g1_29dof_wbt_observation_with_height_scan = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_with_height_scan,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_height_scan_terms,
        ),
    },
)

g1_29dof_wbt_observation_height_scan_tokenhsi = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "actor_height_scan": actor_height_scan,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
        "critic_height_scan": critic_height_scan,
    },
)

g1_29dof_wbt_observation_with_depth_camera = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_with_depth_camera,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_depth_camera_terms,
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

g1_29dof_wbt_observation_w_object_with_height_scan = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_with_height_scan,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_height_scan_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object_height_scan_tokenhsi = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "actor_height_scan": actor_height_scan,
        "actor_object_obs": actor_object_obs,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
        "critic_height_scan": critic_height_scan,
        "critic_object_obs": critic_object_obs,
    },
)

g1_29dof_wbt_observation_w_object_height_scan_tokenhsi_next_target = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "actor_height_scan": actor_height_scan,
        "actor_object_obs": actor_object_obs_with_next_target,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
        "critic_height_scan": critic_height_scan,
        "critic_object_obs": critic_object_obs_with_next_target,
    },
)

__all__ = [
    "g1_29dof_wbt_observation",
    "g1_29dof_wbt_observation_with_height_scan",
    "g1_29dof_wbt_observation_height_scan_tokenhsi",
    "g1_29dof_wbt_observation_with_depth_camera",
    "g1_29dof_wbt_observation_w_object",
    "g1_29dof_wbt_observation_w_object_with_height_scan",
    "g1_29dof_wbt_observation_w_object_height_scan_tokenhsi",
    "g1_29dof_wbt_observation_w_object_height_scan_tokenhsi_next_target",
]
