"""Default inference configurations for holosoma_inference."""

from __future__ import annotations

from dataclasses import replace

import tyro
from typing_extensions import Annotated

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values import observation, robot, task

# G1 Locomotion
g1_29dof_loco = InferenceConfig(
    robot=robot.g1_29dof,
    observation=observation.loco_g1_29dof,
    task=task.locomotion,
)

# T1 Locomotion
t1_29dof_loco = InferenceConfig(
    robot=robot.t1_29dof,
    observation=observation.loco_t1_29dof,
    task=task.locomotion,
)

# G1 Whole-Body Tracking
g1_29dof_wbt = InferenceConfig(
    robot=replace(
        robot.g1_29dof,
        stiff_startup_pos=(
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg
            0.0, 0.0, 0.0,  # waist
            0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # left arm
            0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # right arm
        ),
        stiff_startup_kp=(
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,  # left leg
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,  # right leg
            200.0, 200.0, 200.0,  # waist
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,  # left arm
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,  # right arm
        ),
        stiff_startup_kd=(
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,  # left leg
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,  # right leg
            5.0, 5.0, 5.0,  # waist
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,  # left arm
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,  # right arm
        ),
    ),
    observation=observation.wbt,
    task=task.wbt,
)

g1_29dof_wbt_w_object = replace(
    g1_29dof_wbt,
    observation=observation.wbt_w_object,
    task=task.wbt,
)

g1_29dof_wbt_object_generalist = replace(
    g1_29dof_wbt,
    observation=observation.wbt_object_generalist,
    task=replace(
        task.wbt,
        use_sim_time=True,
        auto_start_motion=True,
        use_sim_state=True,
        prefer_sim_ref_from_sim_state=True,
        restart_motion_on_clock_reset=True,
    ),
)

g1_29dof_wbt_depth_distill = replace(
    g1_29dof_wbt,
    observation=observation.wbt_depth_distill,
    task=replace(
        task.wbt,
        use_sim_time=True,
        use_sim_state=True,
        prefer_sim_ref_from_sim_state=True,
        restart_motion_on_clock_reset=True,
    ),
)

g1_29dof_wbt_terrain_aware = replace(
    g1_29dof_wbt,
    task=replace(
        task.wbt,
        use_sim_time=True,
        use_sim_state=True,
        use_split_perception_obs=True,
        defer_policy_start_until_valid_state=True,
        prefer_sim_ref_from_sim_state=True,
        use_root_reference_at_clip_start=True,
        restart_motion_on_clock_reset=True,
    ),
)

# G1 Whole-Body Tracking (VideoMimic)
g1_29dof_videomimic = InferenceConfig(
    robot=replace(
        robot.g1_29dof,
        stiff_startup_pos=(
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg
            0.0, 0.0, 0.0,  # waist
            0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # left arm
            0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # right arm
        ),
        stiff_startup_kp=(
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,  # left leg
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,  # right leg
            200.0, 200.0, 200.0,  # waist
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,  # left arm
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,  # right arm
        ),
        stiff_startup_kd=(
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,  # left leg
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,  # right leg
            5.0, 5.0, 5.0,  # waist
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,  # left arm
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,  # right arm
        ),
    ),
    observation=observation.wbt_videomimic,
    task=task.wbt,
)

DEFAULTS = {
    "g1-29dof-loco": g1_29dof_loco,
    "t1-29dof-loco": t1_29dof_loco,
    "g1-29dof-wbt": g1_29dof_wbt,
    "g1-29dof-wbt-terrain-aware": g1_29dof_wbt_terrain_aware,
    "g1-29dof-wbt-terrain": g1_29dof_wbt_terrain_aware,
    "g1-29dof-wbt-depth-distill": g1_29dof_wbt_depth_distill,
    "g1-29dof-wbt-w-object": g1_29dof_wbt_w_object,
    "g1-29dof-wbt-object-generalist": g1_29dof_wbt_object_generalist,
    "g1-29dof-wbt-w-obj": g1_29dof_wbt_object_generalist,
    "g1-29dof-w-obj": g1_29dof_wbt_object_generalist,
    "g1-29dof-videomimic": g1_29dof_videomimic,
}

# Annotated version for Tyro CLI with subcommands
AnnotatedInferenceConfig = Annotated[
    InferenceConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults({f"inference:{k}": v for k, v in DEFAULTS.items()})
    ),
]
