from __future__ import annotations


def test_pure_rl_critic317_removes_redundant_proprio_group() -> None:
    from holosoma.config_values.wbt.g1.experiment import (
        g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift_critic317,
    )

    config = g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift_critic317
    actor = config.algo.config.module_dict.actor
    critic = config.algo.config.module_dict.critic

    assert actor.input_dim == [
        "actor_obs_root_contact_aware",
        "actor_obs_drop_button",
        "actor_obs_proprio_with_actions_no_linvel",
    ]
    assert critic.input_dim == [
        "critic_obs",
        "critic_actions",
        "actor_obs_root_contact_aware",
        "actor_obs_drop_button",
    ]
    assert critic.layer_config.module_input_name == tuple(critic.input_dim)
    assert "critic_proprio_history" not in config.observation.groups


def test_pure_rl_critic317_is_registered_without_changing_legacy_preset() -> None:
    from holosoma.config_values.experiment import DEFAULTS
    from holosoma.config_values.wbt.g1.experiment import (
        g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift,
    )

    name = "g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift_critic317"
    assert DEFAULTS[name].training.name == name
    assert "critic_proprio_history" in (
        g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift.observation.groups
    )
