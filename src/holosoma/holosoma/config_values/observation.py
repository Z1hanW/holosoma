"""Default observation manager configurations."""

from holosoma.config_values.loco.g1.observation import g1_29dof_loco_single_wolinvel
from holosoma.config_values.loco.t1.observation import t1_29dof_loco_single_wolinvel
from holosoma.config_values.wbt.g1.observation import (
    critic_obs_w_object_command_distill_legacy_target_terms,
    g1_29dof_wbt_observation,
    g1_29dof_wbt_observation_motion_tracking,
    g1_29dof_wbt_observation_motion_tracking_split,
    g1_29dof_wbt_observation_terrain_distill_sparse_root_cmd,
    g1_29dof_wbt_observation_terrain_transformer,
    g1_29dof_wbt_observation_videomimic,
    g1_29dof_wbt_observation_w_object_legacy,
    g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd,
    g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_critic317,
    g1_29dof_wbt_observation_w_object_hybrid_stage2,
    g1_29dof_wbt_observation_w_object_hybrid_velocity,
    g1_29dof_wbt_observation_w_object_hybrid_world_velocity,
    g1_29dof_wbt_observation_w_object_hmi_depth,
    g1_29dof_wbt_observation_w_object_hmi_depth_object_xy,
    g1_29dof_wbt_observation_w_object_hmi_depth_root_xy,
    g1_29dof_wbt_observation_w_object_policy_world_root_error,
    g1_29dof_wbt_observation_w_object_policy_world_velocity,
    g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_teacher_linvel,
    g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_legacy,
    g1_29dof_wbt_observation_w_object,
    g1_29dof_wbt_observation_w_object_teacher_linvel,
)

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof_loco_single_wolinvel": t1_29dof_loco_single_wolinvel,
    "g1_29dof_loco_single_wolinvel": g1_29dof_loco_single_wolinvel,
    "g1_29dof_wbt": g1_29dof_wbt_observation,
    "g1_29dof_wbt_motion_tracking": g1_29dof_wbt_observation_motion_tracking,
    "g1_29dof_wbt_motion_tracking_split": g1_29dof_wbt_observation_motion_tracking_split,
    "g1_29dof_wbt_terrain_distill_sparse_root_cmd": g1_29dof_wbt_observation_terrain_distill_sparse_root_cmd,
    "g1_29dof_wbt_terrain_transformer": g1_29dof_wbt_observation_terrain_transformer,
    "g1_29dof_wbt_videomimic": g1_29dof_wbt_observation_videomimic,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_observation_w_object,
    "g1_29dof_wbt_w_object_teacher_linvel": g1_29dof_wbt_observation_w_object_teacher_linvel,
    "g1_29dof_wbt_w_object_legacy": g1_29dof_wbt_observation_w_object_legacy,
    "g1_29dof_wbt_w_object_distill_sparse_root_cmd": g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd,
    "g1_29dof_wbt_w_object_distill_sparse_root_cmd_critic317": (
        g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_critic317
    ),
    "g1_29dof_wbt_w_object_hybrid_stage2": g1_29dof_wbt_observation_w_object_hybrid_stage2,
    "g1_29dof_wbt_w_object_hybrid_velocity": g1_29dof_wbt_observation_w_object_hybrid_velocity,
    "g1_29dof_wbt_w_object_hybrid_world_velocity": (
        g1_29dof_wbt_observation_w_object_hybrid_world_velocity
    ),
    "g1_29dof_wbt_w_object_hmi_depth": (
        g1_29dof_wbt_observation_w_object_hmi_depth
    ),
    "g1_29dof_wbt_w_object_policy_world_velocity": (
        g1_29dof_wbt_observation_w_object_policy_world_velocity
    ),
    "g1_29dof_wbt_w_object_policy_world_root_error": (
        g1_29dof_wbt_observation_w_object_policy_world_root_error
    ),
    "g1_29dof_wbt_w_object_distill_sparse_root_cmd_teacher_linvel": g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_teacher_linvel,
    "g1_29dof_wbt_w_object_distill_sparse_root_cmd_legacy": g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_legacy,
}
