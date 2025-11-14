"""Default inference configurations for holosoma_inference."""

from __future__ import annotations

import tyro
from typing_extensions import Annotated

from holosoma_inference.config2.config_types.inference import InferenceConfig
from holosoma_inference.config2.config_values import observation, robot, task

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
    robot=robot.g1_29dof,
    observation=observation.wbt,
    task=task.wbt,
)

DEFAULTS = {
    "g1-29dof-loco": g1_29dof_loco,
    "t1-29dof-loco": t1_29dof_loco,
    "g1-29dof-wbt": g1_29dof_wbt,
}

# Annotated version for Tyro CLI with subcommands
AnnotatedInferenceConfig = Annotated[
    InferenceConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults({f"inference:{k}": v for k, v in DEFAULTS.items()})
    ),
]
