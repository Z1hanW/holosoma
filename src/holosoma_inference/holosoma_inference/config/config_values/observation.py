"""Default observation configurations for holosoma_inference.

This module provides pre-configured observation spaces for different
robot types and tasks, converted from the original YAML configurations.
"""

from __future__ import annotations

from dataclasses import replace

from holosoma_inference.config.config_types.observation import (
    ObservationConfig,
    ObservationTermDescriptor,
)


def _term(
    namespace: str,
    name: str,
    *,
    func_name: str | None = None,
    noise: float = 0.0,
) -> ObservationTermDescriptor:
    return ObservationTermDescriptor(
        func=f"holosoma.managers.observation.terms.{namespace}:{func_name or name}",
        params={},
        noise=noise,
        clip=None,
    )


_LOCOMOTION_TERM_DESCRIPTORS = {
    "base_ang_vel": _term("locomotion", "base_ang_vel"),
    "projected_gravity": _term("locomotion", "projected_gravity"),
    "command_lin_vel": _term("locomotion", "command_lin_vel"),
    "command_ang_vel": _term("locomotion", "command_ang_vel"),
    "dof_pos": _term("locomotion", "dof_pos", noise=0.01),
    "dof_vel": _term("locomotion", "dof_vel", noise=0.1),
    "actions": _term("locomotion", "actions"),
    "sin_phase": _term("locomotion", "sin_phase"),
    "cos_phase": _term("locomotion", "cos_phase"),
}

_WBT_TERM_DESCRIPTORS = {
    "motion_command": _term("wbt", "motion_command"),
    "motion_ref_ori_b": _term("wbt", "motion_ref_ori_b", noise=0.05),
    "base_lin_vel": _term("wbt", "base_lin_vel"),
    "base_ang_vel": _term("wbt", "base_ang_vel", noise=0.2),
    "dof_pos": _term("wbt", "dof_pos", noise=0.01),
    "dof_vel": _term("wbt", "dof_vel", noise=0.5),
    "actions": _term("wbt", "actions"),
    "obj_size": _term("wbt", "obj_size"),
    "obj_target_ori_b": _term("wbt", "obj_target_ori_b"),
    "obj_target_pos_b": _term("wbt", "obj_target_pos_b"),
    "obj_target_pose_size_b": _term("wbt", "obj_target_pose_size_b"),
    "obj_pos_b": _term("wbt", "obj_pos_b"),
    "obj_ori_b": _term("wbt", "obj_ori_b"),
    # ``obj_lin_vel_b_v2`` is the scientifically correct root-frame velocity
    # implementation.  Keeping the term key stable while pinning the function
    # path makes old same-shape checkpoints fail closed.
    "obj_lin_vel_b": _term("wbt", "obj_lin_vel_b", func_name="obj_lin_vel_b_v2"),
    "obj_ang_vel_b": _term("wbt", "obj_ang_vel_b"),
    "sparse_target_root_trajectory_command": _term("wbt", "sparse_target_root_trajectory_command"),
    "sparse_target_root_trajectory_command_contact_aware": _term(
        "wbt", "sparse_target_root_trajectory_command_contact_aware"
    ),
    "pickup_button": _term("wbt", "pickup_button"),
    "drop_button": _term("wbt", "drop_button"),
    "obj_current_pose_size_b": _term("wbt", "obj_current_pose_size_b"),
    "torso_real": _term("wbt", "torso_real"),
    "torso_xy_rel": _term("wbt", "torso_xy_rel"),
    "torso_yaw_rel": _term("wbt", "torso_yaw_rel"),
    "target_joints": _term("wbt", "target_joints"),
    "target_root_roll": _term("wbt", "target_root_roll"),
    "target_root_pitch": _term("wbt", "target_root_pitch"),
}


def _with_canonical_contract(
    config: ObservationConfig,
    catalog: dict[str, ObservationTermDescriptor],
    *,
    noise_enabled_groups: frozenset[str] = frozenset(),
) -> ObservationConfig:
    term_names = {term for terms in config.obs_dict.values() for term in terms}
    missing = sorted(term_names.difference(catalog))
    if missing:
        raise ValueError(f"Inference observation terms have no canonical descriptor: {missing}.")
    unknown_noise_groups = sorted(noise_enabled_groups.difference(config.obs_dict))
    if unknown_noise_groups:
        raise ValueError(f"Unknown noise-enabled inference observation groups: {unknown_noise_groups}.")
    return replace(
        config,
        term_descriptors={name: catalog[name] for name in sorted(term_names)},
        group_concatenate={name: True for name in config.obs_dict},
        group_enable_noise={name: name in noise_enabled_groups for name in config.obs_dict},
    )

DEFAULT_WBT_POLICY_HISTORY_LENGTH = 1

# =============================================================================
# Locomotion Observation Configurations
# =============================================================================

loco_g1_29dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase",
        ]
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "sin_phase": 1,
        "cos_phase": 1,
    },
    obs_scales={
        "base_lin_vel": 2.0,
        "base_ang_vel": 0.25,
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

loco_t1_29dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase",
        ]
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "sin_phase": 2,
        "cos_phase": 2,
    },
    obs_scales={
        "base_lin_vel": 1.0,  # T1 uses 1.0 (vs G1's 2.0)
        "base_ang_vel": 1.0,  # T1 uses 1.0 (vs G1's 0.25)
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.1,  # T1 uses 0.1 (vs G1's 0.05)
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)


# =============================================================================
# WBT (Whole Body Tracking) Observation Configurations
# =============================================================================

wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "motion_command",
            "motion_ref_ori_b",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ]
    },
    obs_dims={
        "motion_command": 58,
        "motion_ref_pos_b": 3,
        "motion_ref_ori_b": 6,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "actions": 1.0,
        "motion_command": 1.0,
        "motion_ref_pos_b": 1.0,
        "motion_ref_ori_b": 1.0,
        "base_lin_vel": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "robot_body_pos_b": 1.0,
        "robot_body_ori_b": 1.0,
    },
    history_length_dict={
        "actor_obs": DEFAULT_WBT_POLICY_HISTORY_LENGTH,
    },
)

wbt_object_velocity_generalist = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "motion_command",
            "motion_ref_ori_b",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "obj_target_pose_size_b",
            "obj_pos_b",
            "obj_ori_b",
            "obj_lin_vel_b",
            "obj_ang_vel_b",
        ]
    },
    obs_dims={
        "motion_command": 58,
        "motion_ref_ori_b": 6,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "obj_target_pose_size_b": 12,
        "obj_pos_b": 3,
        "obj_ori_b": 6,
        "obj_lin_vel_b": 3,
        "obj_ang_vel_b": 3,
    },
    obs_scales={
        "motion_command": 1.0,
        "motion_ref_ori_b": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
        "obj_target_pose_size_b": 1.0,
        "obj_pos_b": 1.0,
        "obj_ori_b": 1.0,
        "obj_lin_vel_b": 1.0,
        "obj_ang_vel_b": 1.0,
    },
    history_length_dict={
        "actor_obs": DEFAULT_WBT_POLICY_HISTORY_LENGTH,
    },
)

wbt_w_object = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "motion_command",
            "motion_ref_ori_b",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "obj_size",
            "obj_target_ori_b",
            "obj_target_pos_b",
            "obj_pos_b",
            "obj_ori_b",
        ]
    },
    obs_dims={
        "motion_command": 58,
        "motion_ref_ori_b": 6,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "obj_size": 3,
        "obj_target_ori_b": 6,
        "obj_target_pos_b": 3,
        "obj_pos_b": 3,
        "obj_ori_b": 6,
    },
    obs_scales={
        "motion_command": 1.0,
        "motion_ref_ori_b": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
        "obj_size": 1.0,
        "obj_target_ori_b": 1.0,
        "obj_target_pos_b": 1.0,
        "obj_pos_b": 1.0,
        "obj_ori_b": 1.0,
    },
    history_length_dict={
        "actor_obs": 5,
    },
)

wbt_w_object_legacy = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "motion_command",
            "motion_ref_ori_b",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "obj_target_pose_size_b",
            "obj_pos_b",
            "obj_ori_b",
        ]
    },
    obs_dims={
        "motion_command": 58,
        "motion_ref_ori_b": 6,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "obj_target_pose_size_b": 12,
        "obj_pos_b": 3,
        "obj_ori_b": 6,
    },
    obs_scales={
        "motion_command": 1.0,
        "motion_ref_ori_b": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
        "obj_target_pose_size_b": 1.0,
        "obj_pos_b": 1.0,
        "obj_ori_b": 1.0,
    },
    history_length_dict={
        "actor_obs": 5,
    },
)

# Exact default student contracts emitted by ``distill_as_perception.sh``.
# The non-contact layout is intentionally not an alias of the older 96-D
# contact-aware layout below: their tensor widths match but their commands and
# group boundaries have different semantics.
wbt_as_depth_distill = ObservationConfig(
    obs_dict={
        "actor_obs_root": [
            "sparse_target_root_trajectory_command",
        ],
        "actor_obs_proprio": [
            "base_lin_vel",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
        ],
        "actor_obs_actions": [
            "actions",
        ],
    },
    obs_dims={
        "sparse_target_root_trajectory_command": 3,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "sparse_target_root_trajectory_command": 1.0,
        "base_lin_vel": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={
        "actor_obs_root": 1,
        "actor_obs_proprio": 1,
        "actor_obs_actions": 1,
    },
)

# Generic AS contact-aware runs use actions inside the no-linear-velocity
# proprio group, yielding 93 dimensions rather than the older split 96-D
# contact-aware contract.
wbt_as_contact_aware_depth_distill = ObservationConfig(
    obs_dict={
        "actor_obs_root_contact_aware": [
            "sparse_target_root_trajectory_command_contact_aware",
        ],
        "actor_obs_proprio_with_actions_no_linvel": [
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ],
    },
    obs_dims={
        "sparse_target_root_trajectory_command_contact_aware": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "sparse_target_root_trajectory_command_contact_aware": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={
        "actor_obs_root_contact_aware": 1,
        "actor_obs_proprio_with_actions_no_linvel": 1,
    },
)

wbt_depth_distill = ObservationConfig(
    obs_dict={
        "actor_obs_root": [
            "sparse_target_root_trajectory_command",
        ],
        "actor_obs_proprio_no_linvel": [
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
        ],
    },
    obs_dims={
        "sparse_target_root_trajectory_command": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
    },
    obs_scales={
        "sparse_target_root_trajectory_command": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
    },
    history_length_dict={
        "actor_obs_root": 1,
        "actor_obs_proprio_no_linvel": 5,
    },
)

wbt_contact_aware_depth_distill = ObservationConfig(
    obs_dict={
        "actor_obs_root_contact_aware": [
            "sparse_target_root_trajectory_command_contact_aware",
        ],
        "actor_obs_proprio": [
            "base_lin_vel",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
        ],
        "actor_obs_actions": [
            "actions",
        ],
    },
    obs_dims={
        "sparse_target_root_trajectory_command_contact_aware": 3,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "sparse_target_root_trajectory_command_contact_aware": 1.0,
        "base_lin_vel": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={
        "actor_obs_root_contact_aware": 1,
        "actor_obs_proprio": 1,
        "actor_obs_actions": 1,
    },
)

wbt_contact_aware_drop_button_depth_distill = ObservationConfig(
    obs_dict={
        "actor_obs_root_contact_aware": [
            "sparse_target_root_trajectory_command_contact_aware",
        ],
        "actor_obs_drop_button": [
            "drop_button",
        ],
        "actor_obs_proprio_with_actions_no_linvel": [
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ],
    },
    obs_dims={
        "sparse_target_root_trajectory_command_contact_aware": 3,
        "drop_button": 1,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "sparse_target_root_trajectory_command_contact_aware": 1.0,
        "drop_button": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={
        "actor_obs_root_contact_aware": 1,
        "actor_obs_drop_button": 1,
        "actor_obs_proprio_with_actions_no_linvel": 1,
    },
)

wbt_contact_aware_dual_button_depth_distill = ObservationConfig(
    obs_dict={
        "actor_obs_root_contact_aware": [
            "sparse_target_root_trajectory_command_contact_aware",
        ],
        "actor_obs_pickup_button": [
            "pickup_button",
        ],
        "actor_obs_drop_button": [
            "drop_button",
        ],
        "actor_obs_proprio_with_actions_no_linvel": [
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ],
    },
    obs_dims={
        "sparse_target_root_trajectory_command_contact_aware": 3,
        "pickup_button": 1,
        "drop_button": 1,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "sparse_target_root_trajectory_command_contact_aware": 1.0,
        "pickup_button": 1.0,
        "drop_button": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={
        "actor_obs_root_contact_aware": 1,
        "actor_obs_pickup_button": 1,
        "actor_obs_drop_button": 1,
        "actor_obs_proprio_with_actions_no_linvel": 1,
    },
)

wbt_object_mocap_distill = ObservationConfig(
    obs_dict={
        "actor_obs_root": [
            "sparse_target_root_trajectory_command",
        ],
        "actor_obs_proprio_no_linvel": [
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
        ],
        "actor_obs_actions": [
            "actions",
        ],
        "actor_obs_box": [
            "obj_current_pose_size_b",
        ],
    },
    obs_dims={
        "sparse_target_root_trajectory_command": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "obj_current_pose_size_b": 12,
    },
    obs_scales={
        "sparse_target_root_trajectory_command": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
        "obj_current_pose_size_b": 1.0,
    },
    history_length_dict={
        "actor_obs_root": 1,
        "actor_obs_proprio_no_linvel": 1,
        "actor_obs_actions": 1,
        "actor_obs_box": 1,
    },
)

# =============================================================================
# WBT VideoMimic Observation Configurations
# =============================================================================

wbt_videomimic = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "torso_real",
            "torso_xy_rel",
            "torso_yaw_rel",
        ],
        "actor_obs_target": [
            "target_joints",
            "target_root_roll",
            "target_root_pitch",
        ],
    },
    obs_dims={
        "torso_real": 93,
        "torso_xy_rel": 2,
        "torso_yaw_rel": 1,
        "target_joints": 29,
        "target_root_roll": 1,
        "target_root_pitch": 1,
    },
    obs_scales={
        "torso_real": 1.0,
        "torso_xy_rel": 1.0,
        "torso_yaw_rel": 1.0,
        "target_joints": 1.0,
        "target_root_roll": 1.0,
        "target_root_pitch": 1.0,
    },
    history_length_dict={
        "actor_obs": 5,
        "actor_obs_target": 1,
    },
)


# Attach a complete, deployment-side description of the serialized training
# semantics.  Shape-only compatibility remains available only to artifacts
# that have no experiment metadata at all.
loco_g1_29dof = _with_canonical_contract(
    loco_g1_29dof,
    _LOCOMOTION_TERM_DESCRIPTORS,
    noise_enabled_groups=frozenset({"actor_obs"}),
)
loco_t1_29dof = _with_canonical_contract(
    loco_t1_29dof,
    _LOCOMOTION_TERM_DESCRIPTORS,
    noise_enabled_groups=frozenset({"actor_obs"}),
)
wbt = _with_canonical_contract(
    wbt,
    _WBT_TERM_DESCRIPTORS,
    noise_enabled_groups=frozenset({"actor_obs"}),
)
wbt_object_velocity_generalist = _with_canonical_contract(
    wbt_object_velocity_generalist,
    _WBT_TERM_DESCRIPTORS,
    noise_enabled_groups=frozenset({"actor_obs"}),
)
wbt_w_object = _with_canonical_contract(
    wbt_w_object,
    _WBT_TERM_DESCRIPTORS,
    noise_enabled_groups=frozenset({"actor_obs"}),
)
# ``train_object_generalist_ds.sh`` intentionally defaults to disabling actor
# history, while ``train_as_general.sh`` explicitly trains history-5.  These
# checkpoints share term semantics but not tensor shape; keep both deployment
# contracts named instead of making one launcher silently incompatible.
wbt_w_object_history1 = replace(
    wbt_w_object,
    history_length_dict={"actor_obs": 1},
)
wbt_w_object_legacy = _with_canonical_contract(
    wbt_w_object_legacy,
    _WBT_TERM_DESCRIPTORS,
    noise_enabled_groups=frozenset({"actor_obs"}),
)

# The training experiment named ``wbt_w_object_generalist`` now uses the same
# split-target actor observation as ``wbt_w_object``.  Keep the Python alias
# aligned with that source of truth; the velocity-bearing v2 layout is
# exposed only under the explicit ``*_velocity_generalist`` name above.
wbt_object_generalist = wbt_w_object
wbt_as_depth_distill = _with_canonical_contract(
    wbt_as_depth_distill,
    _WBT_TERM_DESCRIPTORS,
)
wbt_as_contact_aware_depth_distill = _with_canonical_contract(
    wbt_as_contact_aware_depth_distill,
    _WBT_TERM_DESCRIPTORS,
)
# ``distill_as_perception.sh contact-aware-history`` is a fixed experiment
# contract: the sparse command remains single-frame while the combined
# proprio/action group contains five old-to-new frames.
wbt_as_contact_aware_history5_depth_distill = replace(
    wbt_as_contact_aware_depth_distill,
    history_length_dict={
        "actor_obs_root_contact_aware": 1,
        "actor_obs_proprio_with_actions_no_linvel": 5,
    },
)
wbt_depth_distill = _with_canonical_contract(wbt_depth_distill, _WBT_TERM_DESCRIPTORS)
wbt_contact_aware_depth_distill = _with_canonical_contract(
    wbt_contact_aware_depth_distill,
    _WBT_TERM_DESCRIPTORS,
)
wbt_contact_aware_drop_button_depth_distill = _with_canonical_contract(
    wbt_contact_aware_drop_button_depth_distill,
    _WBT_TERM_DESCRIPTORS,
)
wbt_contact_aware_dual_button_depth_distill = _with_canonical_contract(
    wbt_contact_aware_dual_button_depth_distill,
    _WBT_TERM_DESCRIPTORS,
)
wbt_object_mocap_distill = _with_canonical_contract(
    wbt_object_mocap_distill,
    _WBT_TERM_DESCRIPTORS,
)
wbt_videomimic = _with_canonical_contract(wbt_videomimic, _WBT_TERM_DESCRIPTORS)

# =============================================================================
# Default Configurations Dictionary
# =============================================================================

DEFAULTS = {
    "loco-g1-29dof": loco_g1_29dof,
    "loco-t1-29dof": loco_t1_29dof,
    "wbt": wbt,
    "wbt-object-generalist": wbt_object_generalist,
    "wbt-object-velocity-generalist": wbt_object_velocity_generalist,
    "wbt-w-object": wbt_w_object,
    "wbt-w-object-history1": wbt_w_object_history1,
    "wbt-w-object-legacy": wbt_w_object_legacy,
    "wbt-as-depth-distill": wbt_as_depth_distill,
    "wbt-as-contact-aware-depth-distill": wbt_as_contact_aware_depth_distill,
    "wbt-as-contact-aware-history5-depth-distill": wbt_as_contact_aware_history5_depth_distill,
    "wbt-depth-distill": wbt_depth_distill,
    "wbt-contact-aware-depth-distill": wbt_contact_aware_depth_distill,
    "wbt-contact-aware-drop-button-depth-distill": wbt_contact_aware_drop_button_depth_distill,
    "wbt-contact-aware-dual-button-depth-distill": wbt_contact_aware_dual_button_depth_distill,
    "wbt-object-mocap-distill": wbt_object_mocap_distill,
    "wbt-videomimic": wbt_videomimic,
}
"""Dictionary of all available observation configurations.

Keys use hyphen-case naming convention for CLI compatibility.
"""
