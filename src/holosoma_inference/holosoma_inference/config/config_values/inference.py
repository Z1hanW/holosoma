"""Default inference configurations for holosoma_inference."""

from dataclasses import replace
from importlib.metadata import entry_points

import tyro
from typing_extensions import Annotated

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values import camera, observation, robot, task

g1_29dof_loco = InferenceConfig(
    robot=robot.g1_29dof,
    observation=observation.loco_g1_29dof,
    task=task.locomotion,
)

t1_29dof_loco = InferenceConfig(
    robot=robot.t1_29dof,
    observation=observation.loco_t1_29dof,
    task=task.locomotion,
)

# fmt: off
g1_29dof_wbt = InferenceConfig(
    robot=replace(
        robot.g1_29dof,
        stiff_startup_pos=(
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
            0.0, 0.0, 0.0,                          # waist
            0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # left arm
            0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # right arm
        ),
        stiff_startup_kp=(
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
            200.0, 200.0, 200.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
        ),
        stiff_startup_kd=(
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
            5.0, 5.0, 5.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        ),
    ),
# fmt: on
    observation=observation.wbt,
    task=task.wbt,
)

g1_box_task = replace(task.wbt, policy_type="g1_box")

g1_box_distill = replace(
    g1_29dof_wbt,
    observation=observation.wbt_depth_distill,
    task=g1_box_task,
)

g1_box_contact_aware_depth_distill = replace(
    g1_29dof_wbt,
    observation=observation.wbt_contact_aware_depth_distill,
    task=g1_box_task,
)

g1_box_linvel_depth_distill = replace(
    g1_29dof_wbt,
    observation=observation.wbt_linvel_depth_distill,
    task=g1_box_task,
)

g1_box_action_history_depth_distill = replace(
    g1_29dof_wbt,
    observation=observation.wbt_action_history_depth_distill,
    task=g1_box_task,
)

g1_box_linvel_action_history_depth_distill = replace(
    g1_29dof_wbt,
    observation=observation.wbt_linvel_action_history_depth_distill,
    task=g1_box_task,
)

g1_box_linvel_contact_aware_depth_distill = replace(
    g1_29dof_wbt,
    observation=observation.wbt_linvel_contact_aware_depth_distill,
    task=g1_box_task,
)

g1_29dof_loco_manip_stand_height_waist = InferenceConfig(
    robot=robot.g1_29dof_loco_manip_stand_height_waist,
    observation=observation.loco_manip_stand_height_waist,
    task=task.loco_manip_stand_height_waist,
    camera=camera.dual_depth_cameras,
)

g1_wbt_distillation = InferenceConfig(
    robot=replace(
        robot.g1_29dof_wbt_distillation,
        stiff_startup_pos=(
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
            0.0, 0.0, 0.0,                          # waist
            0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # left arm
            0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # right arm
        ),
        stiff_startup_kp=(
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
            200.0, 200.0, 200.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
        ),
        stiff_startup_kd=(
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
            5.0, 5.0, 5.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        ),
    ),
    observation=observation.wbt_distillation_g1,
    task=task.wbt_distillation,
    camera=camera.single_zed2i_depth,
)

# fmt: off
g1_blind_fall_recovery = InferenceConfig(
    robot=replace(
        robot.g1_29dof,
        stiff_startup_pos=(
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
            0.0, 0.0, 0.0,                          # waist
            0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # left arm
            0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # right arm
        ),
        stiff_startup_kp=(
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
            200.0, 200.0, 200.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
        ),
        stiff_startup_kd=(
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
            5.0, 5.0, 5.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        ),
    ),
# fmt: on
    observation=observation.blind_fall_recovery_g1,
    task=task.blind_fall_recovery,
)

DEFAULTS = {
    "g1-29dof-loco": g1_29dof_loco,
    "t1-29dof-loco": t1_29dof_loco,
    "g1-29dof-wbt": g1_29dof_wbt,
    "g1-29dof-wbt-object-distill": g1_box_distill,
    "g1-29dof-wbt-depth-distill": g1_box_distill,
    "g1-29dof-wbt-object-contact-aware-depth-distill": g1_box_contact_aware_depth_distill,
    "g1-29dof-wbt-contact-aware-depth-distill": g1_box_contact_aware_depth_distill,
    "g1-29dof-wbt-object-linvel-depth-distill": g1_box_linvel_depth_distill,
    "g1-29dof-wbt-linvel-depth-distill": g1_box_linvel_depth_distill,
    "g1-29dof-wbt-object-action-history-depth-distill": g1_box_action_history_depth_distill,
    "g1-29dof-wbt-action-history-depth-distill": g1_box_action_history_depth_distill,
    "g1-29dof-wbt-object-linvel-action-history-depth-distill": g1_box_linvel_action_history_depth_distill,
    "g1-29dof-wbt-linvel-action-history-depth-distill": g1_box_linvel_action_history_depth_distill,
    "g1-29dof-wbt-object-linvel-contact-aware-depth-distill": g1_box_linvel_contact_aware_depth_distill,
    "g1-29dof-wbt-linvel-contact-aware-depth-distill": g1_box_linvel_contact_aware_depth_distill,
    "g1-box-contact-aware-near0p3": g1_box_contact_aware_depth_distill,
    "g1-box-xxehngzo": g1_box_contact_aware_depth_distill,
    "g1-box-g1_box_perception_pure_sd_ppo_first_contact14": g1_box_distill,
    "g1-box-shoo7sr1": g1_box_distill,
    "g1-box-w5qostjn": g1_box_distill,
    "g1-box-w5qostjn_linvel_contact_aware": g1_box_linvel_contact_aware_depth_distill,
    "g1-box-w5qostjn-linvel-contact-aware": g1_box_linvel_contact_aware_depth_distill,
    "g1-box-tvtwx4to": g1_box_linvel_contact_aware_depth_distill,
    "g1-box-w5qostjn_linvel_action_history": g1_box_linvel_action_history_depth_distill,
    "g1-box-w5qostjn-linvel-action-history": g1_box_linvel_action_history_depth_distill,
    "g1-box-haap1tjl": g1_box_linvel_action_history_depth_distill,
    "g1-box-w5qostjn_action_history": g1_box_action_history_depth_distill,
    "g1-box-w5qostjn-action-history": g1_box_action_history_depth_distill,
    "g1-box-5aotqbdq": g1_box_action_history_depth_distill,
    "g1-box-w5qostjn_linvel": g1_box_linvel_depth_distill,
    "g1-box-w5qostjn-linvel": g1_box_linvel_depth_distill,
    "g1-box-c1gaknfu": g1_box_linvel_depth_distill,
    "g1-29dof-loco-manip-stand-height-waist": g1_29dof_loco_manip_stand_height_waist,
    "g1-wbt-distillation": g1_wbt_distillation,
    "g1-blind-fall-recovery": g1_blind_fall_recovery,
}

# Auto-discover inference configs from installed extensions
for ep in entry_points(group="holosoma.config.inference"):
    DEFAULTS[ep.name] = ep.load()

AnnotatedInferenceConfig = Annotated[
    InferenceConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults({f"inference:{k}": v for k, v in DEFAULTS.items()})
        ),
]
