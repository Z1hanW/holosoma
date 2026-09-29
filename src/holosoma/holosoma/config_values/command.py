"""Default command manager configurations."""

from holosoma.config_values.loco.g1.command import g1_29dof_command
from holosoma.config_values.loco.t1.command import t1_29dof_command
from holosoma.config_values.wbt.g1.command import (
    g1_29dof_wbt_command,
    g1_29dof_wbt_command_motion_tracking,
    g1_29dof_wbt_command_w_object,
    g1_29dof_wbt_command_w_object_generalist,
    g1_29dof_wbt_command_w_object_hybrid_stage2,
    g1_29dof_wbt_command_w_object_hybrid_velocity,
    g1_29dof_wbt_command_w_object_hybrid_world_velocity,
    g1_29dof_wbt_command_w_object_hmi_depth_stage1,
    g1_29dof_wbt_command_w_object_hmi_depth_stage2,
    g1_29dof_wbt_command_w_object_hmi_depth_stage2_object_xy,
    g1_29dof_wbt_command_w_object_hmi_depth_stage2_root_xy,
    g1_29dof_wbt_command_w_object_pure_rl_policy_command_after_lift,
)

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof": t1_29dof_command,
    "g1_29dof": g1_29dof_command,
    "g1_29dof_wbt": g1_29dof_wbt_command,
    "g1_29dof_wbt_motion_tracking": g1_29dof_wbt_command_motion_tracking,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_command_w_object,
    "g1_29dof_wbt_w_object_generalist": g1_29dof_wbt_command_w_object_generalist,
    "g1_29dof_wbt_w_object_hybrid_stage2": g1_29dof_wbt_command_w_object_hybrid_stage2,
    "g1_29dof_wbt_w_object_hybrid_velocity": g1_29dof_wbt_command_w_object_hybrid_velocity,
    "g1_29dof_wbt_w_object_hybrid_world_velocity": (
        g1_29dof_wbt_command_w_object_hybrid_world_velocity
    ),
    "g1_29dof_wbt_w_object_hmi_depth_stage1": (
        g1_29dof_wbt_command_w_object_hmi_depth_stage1
    ),
    "g1_29dof_wbt_w_object_hmi_depth_stage2": (
        g1_29dof_wbt_command_w_object_hmi_depth_stage2
    ),
    "g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift": (
        g1_29dof_wbt_command_w_object_pure_rl_policy_command_after_lift
    ),
}
