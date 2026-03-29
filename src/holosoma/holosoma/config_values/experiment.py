import tyro
from typing_extensions import Annotated

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.loco.g1.experiment import g1_29dof, g1_29dof_fast_sac
from holosoma.config_values.loco.t1.experiment import t1_29dof, t1_29dof_fast_sac
from holosoma.config_values.wbt.g1.experiment import (
    g1_terrain_transformer,
    g1_29dof_wbt,
    g1_29dof_wbt_fast_sac,
    g1_29dof_wbt_fast_sac_w_object,
    g1_29dof_wbt_motion_tracking,
    g1_29dof_wbt_motion_tracking_mlp_encoder,
    g1_29dof_wbt_motion_tracking_transformer,
    g1_29dof_wbt_videomimic_terrain_transformer,
    g1_29dof_wbt_videomimic_mlp,
    g1_29dof_wbt_videomimic_mlp_w_gru,
    g1_29dof_wbt_videomimic_transformer,
    g1_29dof_wbt_w_object,
    g1_29dof_wbt_w_object_command_curriculum,
    g1_29dof_wbt_w_object_distill_sparse_goal_mixed,
    g1_29dof_wbt_w_object_distill_sparse_root_cmd,
    g1_29dof_wbt_w_object_distill_sparse_root_cmd_legacy,
    g1_29dof_wbt_w_object_extend,
    g1_29dof_wbt_w_object_generalist,
    g1_29dof_wbt_w_object_generalist_legacy_obs,
)

DEFAULTS = {
    "g1_29dof": g1_29dof,
    "g1_29dof_fast_sac": g1_29dof_fast_sac,
    "t1_29dof": t1_29dof,
    "t1_29dof_fast_sac": t1_29dof_fast_sac,
    "g1_29dof_wbt": g1_29dof_wbt,
    "g1_29dof_wbt_motion_tracking": g1_29dof_wbt_motion_tracking,
    "g1_29dof_wbt_motion_tracking_mlp_encoder": g1_29dof_wbt_motion_tracking_mlp_encoder,
    "g1_29dof_wbt_motion_tracking_transformer": g1_29dof_wbt_motion_tracking_transformer,
    "g1_terrain_transformer": g1_terrain_transformer,
    "g1_29dof_wbt_videomimic_mlp": g1_29dof_wbt_videomimic_mlp,
    "w_gru": g1_29dof_wbt_videomimic_mlp_w_gru,
    "g1_29dof_wbt_videomimic_transformer": g1_29dof_wbt_videomimic_transformer,
    "g1_29dof_wbt_videomimic_terrain_transformer": g1_29dof_wbt_videomimic_terrain_transformer,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_w_object,
    "g1_29dof_wbt_w_object_extend": g1_29dof_wbt_w_object_extend,
    "g1_29dof_wbt_w_object_generalist": g1_29dof_wbt_w_object_generalist,
    "g1_29dof_wbt_w_object_generalist_legacy_obs": g1_29dof_wbt_w_object_generalist_legacy_obs,
    "g1_29dof_wbt_w_object_command_curriculum": g1_29dof_wbt_w_object_command_curriculum,
    "g1_29dof_wbt_w_object_distill_sparse_goal_mixed": g1_29dof_wbt_w_object_distill_sparse_goal_mixed,
    "g1_29dof_wbt_w_object_distill_sparse_root_cmd": g1_29dof_wbt_w_object_distill_sparse_root_cmd,
    "g1_29dof_wbt_w_object_distill_sparse_root_cmd_legacy": g1_29dof_wbt_w_object_distill_sparse_root_cmd_legacy,
    "g1_29dof_wbt_fast_sac": g1_29dof_wbt_fast_sac,
    "g1_29dof_wbt_fast_sac_w_object": g1_29dof_wbt_fast_sac_w_object,
}

AnnotatedExperimentConfig = Annotated[
    ExperimentConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults(
            {f"exp:{k.replace('_', '-')}": v for k, v in DEFAULTS.items()}
        )
    ),
]
