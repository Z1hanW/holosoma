"""Default reward manager configurations."""

from holosoma.config_values.loco.g1.reward import g1_29dof_loco, g1_29dof_loco_fast_sac
from holosoma.config_values.loco.t1.reward import t1_29dof_loco, t1_29dof_loco_fast_sac
from holosoma.config_values.wbt.g1.reward import (
    g1_29dof_wbt_fast_sac_reward,
    g1_29dof_wbt_reward,
    g1_29dof_wbt_reward_w_object_generalist,
    g1_29dof_wbt_reward_w_object_generalist_offline_contact_guidance,
    g1_29dof_wbt_reward_w_object_generalist_tracking_no_contact,
    g1_29dof_wbt_reward_w_object_hybrid_stage2,
    g1_29dof_wbt_reward_w_object_hybrid_velocity,
    g1_29dof_wbt_reward_w_object_hmi,
    g1_29dof_wbt_reward_w_object_hmi_object_xy,
    g1_29dof_wbt_reward_w_object_hmi_root_xy,
    g1_29dof_wbt_reward_w_object_r2s_contact_guidance,
    g1_29dof_wbt_reward_w_object_r2s_rollout_reference_guidance,
    g1_29dof_wbt_reward_w_object_extend,
    g1_29dof_wbt_reward_w_object,
)

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof_loco": t1_29dof_loco,
    "t1_29dof_loco_fast_sac": t1_29dof_loco_fast_sac,
    "g1_29dof_loco": g1_29dof_loco,
    "g1_29dof_loco_fast_sac": g1_29dof_loco_fast_sac,
    "g1_29dof_wbt": g1_29dof_wbt_reward,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_reward_w_object,
    "g1_29dof_wbt_w_object_generalist": g1_29dof_wbt_reward_w_object_generalist,
    "g1_29dof_wbt_w_object_generalist_offline_contact_guidance": (
        g1_29dof_wbt_reward_w_object_generalist_offline_contact_guidance
    ),
    "g1_29dof_wbt_w_object_generalist_tracking_no_contact": (
        g1_29dof_wbt_reward_w_object_generalist_tracking_no_contact
    ),
    "g1_29dof_wbt_w_object_hybrid_stage2": g1_29dof_wbt_reward_w_object_hybrid_stage2,
    "g1_29dof_wbt_w_object_hybrid_velocity": g1_29dof_wbt_reward_w_object_hybrid_velocity,
    "g1_29dof_wbt_w_object_hmi": g1_29dof_wbt_reward_w_object_hmi,
    "g1_29dof_wbt_w_object_r2s_contact_guidance": g1_29dof_wbt_reward_w_object_r2s_contact_guidance,
    "g1_29dof_wbt_w_object_r2s_rollout_reference_guidance": g1_29dof_wbt_reward_w_object_r2s_rollout_reference_guidance,
    "g1_29dof_wbt_w_object_extend": g1_29dof_wbt_reward_w_object_extend,
    "g1_29dof_wbt_fast_sac": g1_29dof_wbt_fast_sac_reward,
}
