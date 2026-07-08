"""Default observation manager configurations."""

from holosoma.config_values.loco.g1.observation import g1_29dof_loco_single_wolinvel
from holosoma.config_values.loco.t1.observation import t1_29dof_loco_single_wolinvel
from holosoma.config_values.wbt.g1.observation import (
    g1_29dof_wbt_observation,
    g1_29dof_wbt_observation_height_scan_tokenhsi,
    g1_29dof_wbt_observation_with_depth_camera,
    g1_29dof_wbt_observation_with_height_scan,
    g1_29dof_wbt_observation_w_object,
    g1_29dof_wbt_observation_w_object_height_scan_tokenhsi,
    g1_29dof_wbt_observation_w_object_height_scan_tokenhsi_next_target,
    g1_29dof_wbt_observation_w_object_with_height_scan,
)

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof_loco_single_wolinvel": t1_29dof_loco_single_wolinvel,
    "g1_29dof_loco_single_wolinvel": g1_29dof_loco_single_wolinvel,
    "g1_29dof_wbt": g1_29dof_wbt_observation,
    "g1_29dof_wbt_height_scan": g1_29dof_wbt_observation_with_height_scan,
    "g1_29dof_wbt_height_scan_tokenhsi": g1_29dof_wbt_observation_height_scan_tokenhsi,
    "g1_29dof_wbt_depth_camera": g1_29dof_wbt_observation_with_depth_camera,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_observation_w_object,
    "g1_29dof_wbt_w_object_height_scan": g1_29dof_wbt_observation_w_object_with_height_scan,
    "g1_29dof_wbt_w_object_height_scan_tokenhsi": g1_29dof_wbt_observation_w_object_height_scan_tokenhsi,
    "g1_29dof_wbt_w_object_height_scan_tokenhsi_next_target": (
        g1_29dof_wbt_observation_w_object_height_scan_tokenhsi_next_target
    ),
}
