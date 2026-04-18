"""Default termination manager configurations."""

from holosoma.config_values.loco.g1.termination import g1_29dof_termination
from holosoma.config_values.loco.t1.termination import t1_29dof_termination
from holosoma.config_values.wbt.g1.termination import (
    g1_29dof_wbt_termination,
    g1_29dof_wbt_termination_command_curriculum,
    g1_29dof_wbt_termination_distill,
    g1_29dof_wbt_termination_distill_sparse_goal_mixed,
    g1_29dof_wbt_termination_distill_sparse_goal_pickup,
)

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof": t1_29dof_termination,
    "g1_29dof": g1_29dof_termination,
    "g1_29dof_wbt": g1_29dof_wbt_termination,
    "g1_29dof_wbt_command_curriculum": g1_29dof_wbt_termination_command_curriculum,
    "g1_29dof_wbt_distill": g1_29dof_wbt_termination_distill,
    "g1_29dof_wbt_distill_sparse_goal_mixed": g1_29dof_wbt_termination_distill_sparse_goal_mixed,
    "g1_29dof_wbt_distill_sparse_goal_pickup": g1_29dof_wbt_termination_distill_sparse_goal_pickup,
}
