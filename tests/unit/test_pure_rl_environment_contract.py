from holosoma.config_values import randomization as randomization_defaults
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt_w_object_distill_sparse_root_cmd,
    g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift,
)
from holosoma.config_values.wbt.g1.randomization import (
    g1_29dof_wbt_randomization_w_object_pure_rl,
)


def test_pure_rl_disables_default_pose_motion_transitions() -> None:
    config = g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift
    motion = config.command.setup_terms["motion_command"].params["motion_config"]

    assert motion.enable_default_pose_prepend is False
    assert motion.default_pose_prepend_duration_s == 0.0
    assert motion.enable_default_pose_append is False
    assert motion.default_pose_append_duration_s == 0.0
    assert motion.contact_interval_runtime_prepend_compensation is False


def test_pure_rl_uses_joint_bias_and_pd_gain_dr_without_rfi_or_delay() -> None:
    config = g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift
    assert config.randomization is g1_29dof_wbt_randomization_w_object_pure_rl
    assert (
        randomization_defaults.DEFAULTS["g1_29dof_wbt_w_object_pure_rl"]
        is g1_29dof_wbt_randomization_w_object_pure_rl
    )

    terms = config.randomization.setup_terms
    assert terms["setup_dof_pos_bias"].params == {
        "dof_pos_bias_range": [-0.01, 0.01],
        "enabled": True,
    }
    assert terms["setup_torque_rfi"].params == {
        "enabled": False,
        "rfi_lim": 0.0,
    }
    assert terms["actuator_randomizer_state"].params == {
        "kp_range": [0.9, 1.1],
        "kd_range": [0.9, 1.1],
        "rfi_lim_range": [1.0, 1.0],
        "enable_pd_gain": True,
        "enable_rfi_lim": False,
    }
    assert terms["setup_action_delay_buffers"].params == {
        "ctrl_delay_step_range": [0, 0],
        "enabled": False,
    }


def test_distill_control_chain_contract_is_not_changed_by_pure_rl() -> None:
    distill_randomization = (
        g1_29dof_wbt_w_object_distill_sparse_root_cmd.randomization
    )
    pure_rl_randomization = (
        g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift.randomization
    )
    terms = distill_randomization.setup_terms

    assert terms["setup_dof_pos_bias"].params["enabled"] is False
    assert terms["setup_torque_rfi"].params["enabled"] is False
    assert terms["actuator_randomizer_state"].params["enable_pd_gain"] is False
    assert terms["actuator_randomizer_state"].params["enable_rfi_lim"] is False
    assert terms["setup_action_delay_buffers"].params["enabled"] is False

    control_chain_terms = {
        "setup_dof_pos_bias",
        "setup_torque_rfi",
        "actuator_randomizer_state",
        "setup_action_delay_buffers",
    }
    assert pure_rl_randomization.setup_terms.keys() == distill_randomization.setup_terms.keys()
    for name in distill_randomization.setup_terms.keys() - control_chain_terms:
        assert pure_rl_randomization.setup_terms[name] == distill_randomization.setup_terms[name]
    assert pure_rl_randomization.reset_terms == distill_randomization.reset_terms
    assert pure_rl_randomization.step_terms == distill_randomization.step_terms
