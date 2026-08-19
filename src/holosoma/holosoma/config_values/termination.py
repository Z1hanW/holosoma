"""Default termination manager configurations."""

from holosoma.config_values.loco.g1.termination import g1_29dof_termination
from holosoma.config_values.loco.t1.termination import t1_29dof_termination
from holosoma.config_values.wbt.g1.termination import (
    g1_29dof_wbt_termination,
    g1_29dof_wbt_termination_distill,
    g1_29dof_wbt_termination_generalist,
    g1_29dof_wbt_termination_generalist_all_position_z_only,
    g1_29dof_wbt_termination_generalist_z_only,
    g1_29dof_wbt_termination_hybrid_stage2,
    g1_29dof_wbt_termination_hybrid_velocity,
)

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof": t1_29dof_termination,
    "g1_29dof": g1_29dof_termination,
    "g1_29dof_wbt": g1_29dof_wbt_termination,
    "g1_29dof_wbt_generalist": g1_29dof_wbt_termination_generalist,
    "g1_29dof_wbt_generalist_z_only": g1_29dof_wbt_termination_generalist_z_only,
    "g1_29dof_wbt_generalist_all_position_z_only": (
        g1_29dof_wbt_termination_generalist_all_position_z_only
    ),
    "g1_29dof_wbt_hybrid_stage2": g1_29dof_wbt_termination_hybrid_stage2,
    "g1_29dof_wbt_hybrid_velocity": g1_29dof_wbt_termination_hybrid_velocity,
    "g1_29dof_wbt_distill": g1_29dof_wbt_termination_distill,
}
