from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.experiment import ObservationOverridesConfig
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object_distill_sparse_root_cmd
from holosoma.observation.config_utils import apply_observation_overrides


def _contact_button_with_actions_config():
    config = g1_29dof_wbt_w_object_distill_sparse_root_cmd
    actor = replace(
        config.algo.config.module_dict.actor,
        input_dim=[
            "actor_obs_root_contact_aware",
            "actor_obs_drop_button",
            "actor_obs_proprio_with_actions_no_linvel",
        ],
    )
    module_dict = replace(config.algo.config.module_dict, actor=actor)
    algo = replace(config.algo, config=replace(config.algo.config, module_dict=module_dict))
    overrides = ObservationOverridesConfig(
        distill_proprio_history_only=True,
        distill_proprio_history_length=5,
    )
    return replace(config, algo=algo, observation_overrides=overrides)


def test_distill_proprio_history_splits_actions_from_active_with_actions_group():
    original = _contact_button_with_actions_config()

    updated = apply_observation_overrides(original)

    actor_inputs = updated.algo.config.module_dict.actor.input_dim
    assert actor_inputs == [
        "actor_obs_root_contact_aware",
        "actor_obs_drop_button",
        "actor_obs_proprio_with_actions_no_linvel",
        "actor_obs_actions",
    ]

    groups = updated.observation.groups
    proprio = groups["actor_obs_proprio_with_actions_no_linvel"]
    assert proprio.history_length == 5
    assert "actions" not in proprio.terms
    assert groups["actor_obs_actions"].history_length == 1
    assert list(groups["actor_obs_actions"].terms) == ["actions"]


def test_distill_proprio_history_keeps_student_commands_single_frame_and_teacher_history_unchanged():
    original = _contact_button_with_actions_config()
    original_teacher_history = original.observation.groups["actor_obs"].history_length

    updated = apply_observation_overrides(original)
    groups = updated.observation.groups

    for group_name in (
        "actor_obs_root_contact_aware",
        "actor_obs_torso_contact_aware",
        "actor_obs_pickup_button",
        "actor_obs_drop_button",
    ):
        assert groups[group_name].history_length == 1
    assert groups["actor_obs"].history_length == original_teacher_history


def test_distill_proprio_history_does_not_mutate_the_source_preset():
    original = _contact_button_with_actions_config()

    apply_observation_overrides(original)

    source_group = original.observation.groups["actor_obs_proprio_with_actions_no_linvel"]
    assert source_group.history_length == 1
    assert "actions" in source_group.terms


def test_distill_proprio_history_override_is_idempotent_after_checkpoint_reload():
    once = apply_observation_overrides(_contact_button_with_actions_config())

    twice = apply_observation_overrides(once)

    assert twice.algo.config.module_dict.actor.input_dim == once.algo.config.module_dict.actor.input_dim
    assert twice.observation.groups == once.observation.groups
