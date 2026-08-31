import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pytest
import torch
from torch import nn

from holosoma.agents.modules.ppo_modules import PPOActorEncoder
from holosoma.config_types.command import HMIMotionConfig
from holosoma.config_types.reward import RewardTermCfg
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt_w_object_hmi_depth_stage1,
    g1_29dof_wbt_w_object_hmi_depth_stage2,
    g1_29dof_wbt_w_object_hmi_depth_stage2_object_xy,
    g1_29dof_wbt_w_object_hmi_depth_stage2_root_xy,
)
from holosoma.managers.command.terms.wbt import (
    MotionCommand,
    MotionLoader,
    build_fixed_hmi_track_mask,
)
from holosoma.managers.observation.terms.wbt import _mask_hmi_generation_reference_rows
from holosoma.managers.reward.terms.wbt import (
    HMIObjectGoalReachedOnce,
    HMIXYGoalReachedOnce,
)
from holosoma.managers.termination.terms.wbt import BodyGroupProximity
from holosoma.perception.config_utils import apply_perception_overrides
from holosoma.utils.inference_helpers import (
    export_policy_as_onnx,
    validate_exported_policy_onnx,
)


class _CommandManager:
    def __init__(self, motion_command: MotionCommand):
        self.motion_command = motion_command

    def get_state(self, name: str):
        assert name == "motion_command"
        return self.motion_command


class _HMIActorONNXWrapper(nn.Module):
    def __init__(self, actor: PPOActorEncoder):
        super().__init__()
        self.actor = actor

    def forward(
        self,
        actor_obs: torch.Tensor,
        perception_obs: torch.Tensor,
    ) -> torch.Tensor:
        return self.actor.act_inference(
            {"actor_obs": actor_obs, "perception_obs": perception_obs}
        )


def _minimal_hmi_command(track_mask: torch.Tensor) -> MotionCommand:
    command = object.__new__(MotionCommand)
    command.hmi_cfg = HMIMotionConfig(track_ratio=0.5)
    command.hmi_track_env_mask = track_mask.clone()
    command.hmi_gen_env_mask = ~track_mask
    command.num_envs = int(track_mask.numel())
    command.device = torch.device("cpu")
    command.hmi_goal_reached = torch.zeros(track_mask.numel(), dtype=torch.bool)
    return command


def test_fixed_hmi_partition_is_deterministic_exact_and_rng_isolated():
    torch.manual_seed(123)
    expected_next_sample = torch.rand(4)

    torch.manual_seed(123)
    first = build_fixed_hmi_track_mask(10, 0.5, 17)
    actual_next_sample = torch.rand(4)
    second = build_fixed_hmi_track_mask(10, 0.5, 17)
    different_seed = build_fixed_hmi_track_mask(10, 0.5, 18)

    assert int(first.sum()) == 5
    assert torch.equal(first, second)
    assert not torch.equal(first, different_seed)
    assert torch.equal(actual_next_sample, expected_next_sample)


def test_hmi_depth_presets_keep_the_production_actor_interface():
    for raw_config, expected_track_ratio, expected_iterations in (
        (g1_29dof_wbt_w_object_hmi_depth_stage1, 1.0, 15000),
        (g1_29dof_wbt_w_object_hmi_depth_stage2, 0.5, 20000),
    ):
        config = apply_perception_overrides(raw_config)
        actor = config.algo.config.module_dict.actor
        critic = config.algo.config.module_dict.critic

        assert config.training.export_onnx is True
        assert config.simulator.config.sim.max_episode_length_s == 10.0
        assert config.algo.config.num_learning_iterations == expected_iterations
        motion_config = config.command.setup_terms["motion_command"].params[
            "motion_config"
        ]
        assert motion_config.motion_file == (
            "data_demo/sub10_largebox_032_mj_w_obj.npz"
        )
        assert config.robot.object.object_urdf_path == (
            "data_demo/objects/objects_largebox.urdf"
        )
        assert motion_config.hmi.track_ratio == expected_track_ratio
        assert motion_config.use_adaptive_timesteps_sampler is True
        assert motion_config.hmi.gen_start_at_timestep_zero_prob == 0.2
        assert actor.input_dim == [
            "actor_obs_hmi_goal_command",
            "actor_obs_drop_button",
            "actor_obs_proprio_with_actions_no_linvel",
        ]
        assert actor.output_dim == ["robot_action_dim"]
        assert actor.type == "MLPPerceptionEncoder"
        assert actor.layer_config.hidden_dims == [512, 256, 128]
        assert actor.layer_config.perception_input_name == "perception_obs"
        assert actor.layer_config.perception_encoder_type == "far_tracking_cnn_small"
        assert actor.layer_config.perception_input_height == 58
        assert actor.layer_config.perception_input_width == 87
        assert actor.layer_config.perception_output_dim == 32
        assert critic.input_dim == [
            "critic_obs",
            "critic_actions",
            "actor_obs_hmi_goal_command",
            "actor_obs_drop_button",
        ]
        assert set(config.termination.terms) == {
            "timeout",
            "body_proximity",
            "bad_tracking",
        }
        bad_tracking = config.termination.terms["bad_tracking"].params
        assert bad_tracking["bad_ref_pos_threshold"] == 0.4
        assert bad_tracking["bad_motion_body_pos_threshold"] == 0.25
        assert bad_tracking["bad_object_pos_threshold"] == 0.3
        assert bad_tracking["bad_object_ori_threshold"] == 0.8

    assert not inspect.isabstract(BodyGroupProximity)


@pytest.mark.parametrize(
    ("raw_config", "goal_target", "term_name", "interface"),
    (
        (
            g1_29dof_wbt_w_object_hmi_depth_stage2_object_xy,
            "object_xy",
            "hmi_object_xy_goal_command",
            "actor93_depth5046_action29_terminal_object_xy_v2",
        ),
        (
            g1_29dof_wbt_w_object_hmi_depth_stage2_root_xy,
            "root_xy",
            "hmi_root_xy_goal_command",
            "actor93_depth5046_action29_terminal_root_xy_v2",
        ),
    ),
)
def test_hmi_xy_presets_remove_yaw_from_goal_observation_and_reward(
    raw_config, goal_target, term_name, interface
):
    config = apply_perception_overrides(raw_config)
    motion_config = config.command.setup_terms["motion_command"].params[
        "motion_config"
    ]
    hmi = motion_config.hmi
    group = config.observation.groups["actor_obs_hmi_goal_command"]

    assert hmi.goal_target == goal_target
    assert hmi.actor_interface_semantics == interface
    assert list(group.terms) == [term_name]
    noise = hmi.object_goal_noise if goal_target == "object_xy" else hmi.root_goal_noise
    assert noise.pos_std_xyz == [0.5, 0.5, 0.0]
    assert noise.pos_clip_xyz == [1.0, 1.0, 0.0]
    assert noise.rpy_std == [0.0, 0.0, 0.0]
    assert noise.rpy_clip == [0.0, 0.0, 0.0]
    sparse_terms = [
        term
        for name, term in config.reward.terms.items()
        if name.startswith(f"hmi_{goal_target}_goal_reached_once")
    ]
    assert len(sparse_terms) == 1
    assert sparse_terms[0].params == {
        "goal_target": goal_target,
        "pos_threshold": 0.2,
        "bonus": 3.0,
    }


def test_hmi_default_motion_bank_has_complete_object_metadata():
    repo_root = Path(__file__).resolve().parents[6]
    config = g1_29dof_wbt_w_object_hmi_depth_stage1
    motion_config = config.command.setup_terms[
        "motion_command"
    ].params["motion_config"]
    motion_path = repo_root / motion_config.motion_file
    object_map_path = motion_path.parent / "_clip_object_urdf_map.json"
    object_path = repo_root / config.robot.object.object_urdf_path
    object_map = json.loads(object_map_path.read_text(encoding="utf-8"))["clips"]

    assert motion_path.is_file()
    object_urdf = object_map[motion_path.stem]["object_urdf_path"]
    assert (motion_path.parent / object_urdf).resolve() == object_path.resolve()

    urdf_root = ElementTree.parse(object_path).getroot()
    assert urdf_root.findall(".//mesh")
    for primitive in ("box", "sphere", "cylinder", "capsule"):
        assert not urdf_root.findall(f".//{primitive}")
    for mesh in urdf_root.findall(".//mesh"):
        assert (object_path.parent / mesh.attrib["filename"]).is_file()


def test_hmi_single_npz_loader_prefers_portable_bank_object_map():
    repo_root = Path(__file__).resolve().parents[6]
    config = g1_29dof_wbt_w_object_hmi_depth_stage1
    motion_config = config.command.setup_terms["motion_command"].params[
        "motion_config"
    ]
    motion_path = repo_root / motion_config.motion_file
    expected_object_path = (
        repo_root / config.robot.object.object_urdf_path
    ).resolve()
    with np.load(motion_path, allow_pickle=False) as data:
        source_absolute_path = str(np.asarray(data["object_urdf_path"]).item())
        body_names = data["body_names"].tolist()
        joint_names = data["joint_names"].tolist()

    assert Path(source_absolute_path) != expected_object_path
    loader = MotionLoader(
        str(motion_path),
        robot_body_names=body_names,
        robot_joint_names=joint_names,
        device="cpu",
    )
    assert loader.clip_object_urdf_paths == [str(expected_object_path)]


def test_hmi_stage_change_is_an_explicit_checkpoint_contract_change():
    stage1 = object.__new__(MotionCommand)
    stage1.hmi_cfg = g1_29dof_wbt_w_object_hmi_depth_stage1.command.setup_terms[
        "motion_command"
    ].params["motion_config"].hmi
    stage2 = object.__new__(MotionCommand)
    stage2.hmi_cfg = g1_29dof_wbt_w_object_hmi_depth_stage2.command.setup_terms[
        "motion_command"
    ].params["motion_config"].hmi

    stage1_contract = stage1._hmi_checkpoint_contract()
    stage2_contract = stage2._hmi_checkpoint_contract()

    assert stage1_contract["upstream_commit"] == (
        "c353731999b3578c41ad5a00f896415b45e6a9f5"
    )
    assert stage1_contract["track_ratio"] == 1.0
    assert stage2_contract["track_ratio"] == 0.5
    assert stage1_contract != stage2_contract


def test_hmi_stage2_uses_released_goal_noise_recipe_and_curriculum():
    hmi = g1_29dof_wbt_w_object_hmi_depth_stage2.command.setup_terms[
        "motion_command"
    ].params["motion_config"].hmi

    assert hmi.object_goal_noise.pos_std_xyz == [0.5, 0.5, 0.001]
    assert hmi.object_goal_noise.pos_clip_xyz == [1.0, 1.0, 0.001]
    assert hmi.object_goal_noise.rpy_std == [0.001, 0.001, 0.4]
    assert hmi.object_goal_noise.rpy_clip == [0.001, 0.001, 0.8]
    assert hmi.goal_noise_initial_scale == 0.3
    assert hmi.goal_noise_update_interval == 20


def test_hmi_goal_noise_curriculum_updates_only_at_iteration_boundary():
    command = _minimal_hmi_command(torch.ones(4, dtype=torch.bool))
    command.hmi_cfg = g1_29dof_wbt_w_object_hmi_depth_stage2.command.setup_terms[
        "motion_command"
    ].params["motion_config"].hmi
    command.hmi_goal_noise_scale = torch.tensor(0.3)
    command.hmi_goal_success_ema = torch.tensor(0.0)
    command.hmi_goal_success_ema_initialized = False
    command.hmi_goal_success_sum = torch.tensor(17.0)
    command.hmi_goal_success_count = torch.tensor(20, dtype=torch.long)
    command.hmi_last_curriculum_update_iteration = 0

    command._update_hmi_goal_noise_curriculum(19)
    assert torch.isclose(command.hmi_goal_noise_scale, torch.tensor(0.3))
    assert command.hmi_goal_success_count.item() == 20

    command._update_hmi_goal_noise_curriculum(20)
    assert torch.isclose(command.hmi_goal_success_ema, torch.tensor(0.85))
    assert torch.isclose(command.hmi_goal_noise_scale, torch.tensor(0.3))
    assert command.hmi_goal_success_count.item() == 0

    command.hmi_goal_success_sum.fill_(20.0)
    command.hmi_goal_success_count.fill_(20)
    command._update_hmi_goal_noise_curriculum(40)
    assert torch.isclose(command.hmi_goal_success_ema, torch.tensor(0.925))
    assert torch.isclose(command.hmi_goal_noise_scale, torch.tensor(0.31))


def test_hmi_curriculum_checkpoint_state_round_trips_strictly():
    command = _minimal_hmi_command(torch.tensor([True, False]))
    command.hmi_goal_noise_scale = torch.tensor(0.37)
    command.hmi_goal_success_ema = torch.tensor(0.91)
    command.hmi_goal_success_ema_initialized = True
    command.hmi_goal_success_sum = torch.tensor(7.0)
    command.hmi_goal_success_count = torch.tensor(9, dtype=torch.long)
    command.hmi_last_curriculum_update_iteration = 60

    state = command._hmi_checkpoint_state()
    restored = command._prepare_hmi_checkpoint_state(state)

    assert torch.equal(restored["goal_noise_scale"], torch.tensor(0.37))
    assert torch.equal(restored["goal_success_ema"], torch.tensor(0.91))
    assert torch.equal(restored["goal_success_sum"], torch.tensor(7.0))
    assert torch.equal(restored["goal_success_count"], torch.tensor(9))
    assert restored["goal_success_ema_initialized"] is True
    assert restored["last_curriculum_update_iteration"] == 60


def test_hmi_reference_mask_preserves_track_rows_and_zeros_gen_rows():
    track_mask = torch.tensor([True, False, True, False])
    command = _minimal_hmi_command(track_mask)
    env = SimpleNamespace(
        num_envs=4,
        device=torch.device("cpu"),
        command_manager=_CommandManager(command),
    )
    values = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    masked = _mask_hmi_generation_reference_rows(env, values)

    assert torch.equal(masked[track_mask], values[track_mask])
    assert torch.equal(masked[~track_mask], torch.zeros_like(masked[~track_mask]))


def test_hmi_goal_command_uses_manual_eval_override_without_changing_native_mode():
    command = _minimal_hmi_command(torch.ones(2, dtype=torch.bool))
    command.hmi_goal_object_pos_w = torch.tensor(
        [[1.0, 2.0, 0.5], [-3.0, 4.0, 0.5]], dtype=torch.float32
    )
    command.hmi_goal_object_quat_w = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32
    )
    command.ref_body_index = 0
    command._env = SimpleNamespace(
        simulator=SimpleNamespace(
            _rigid_body_pos=torch.zeros((2, 1, 3), dtype=torch.float32),
            _rigid_body_rot=torch.tensor(
                [[[0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 1.0]]],
                dtype=torch.float32,
            ),
        )
    )
    command.manual_control_enabled = False

    native = command.get_hmi_object_goal_command()
    assert torch.equal(
        native,
        torch.tensor([[1.0, 2.0, 0.0], [-3.0, 4.0, 0.0]]),
    )

    command.manual_control_enabled = True
    command.manual_xy_rel = torch.tensor([[0.0, 0.0], [0.15, 0.0]])
    command.manual_yaw_rel = torch.zeros((2, 1), dtype=torch.float32)
    command._manual_forward_after_lift_command_semantics = (
        "legacy_constant_robot_heading_frame"
    )
    assert torch.equal(
        command.get_hmi_object_goal_command(),
        torch.tensor([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]]),
    )

    command._manual_forward_after_lift_command_semantics = "robot_heading_velocity_mps"
    with pytest.raises(
        RuntimeError,
        match="Manual HMI evaluation command semantics mismatch",
    ):
        command.get_hmi_object_goal_command()


@pytest.mark.parametrize("goal_target", ("object_xy", "root_xy"))
def test_hmi_xy_goal_command_has_exactly_two_heading_frame_components(goal_target):
    command = _minimal_hmi_command(torch.ones(2, dtype=torch.bool))
    interface = (
        "actor93_depth5046_action29_terminal_object_xy_v2"
        if goal_target == "object_xy"
        else "actor93_depth5046_action29_terminal_root_xy_v2"
    )
    command.hmi_cfg = HMIMotionConfig(
        actor_interface_semantics=interface,
        goal_target=goal_target,
    )
    command.hmi_goal_xy_pos_w = torch.tensor(
        [[1.0, 2.0, 9.0], [-3.0, 4.0, -9.0]], dtype=torch.float32
    )
    command.ref_body_index = 0
    command._env = SimpleNamespace(
        simulator=SimpleNamespace(
            _rigid_body_pos=torch.zeros((2, 1, 3), dtype=torch.float32),
            _rigid_body_rot=torch.tensor(
                [[[0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 1.0]]],
                dtype=torch.float32,
            ),
        )
    )
    command.manual_control_enabled = False

    value = command.get_hmi_xy_goal_command(expected_target=goal_target)

    assert value.shape == (2, 2)
    assert torch.equal(value, torch.tensor([[1.0, 2.0], [-3.0, 4.0]]))
    wrong_target = "root_xy" if goal_target == "object_xy" else "object_xy"
    with pytest.raises(RuntimeError, match="observation/command mismatch"):
        command.get_hmi_xy_goal_command(expected_target=wrong_target)


def test_hmi_sparse_goal_bonus_is_gen_only_and_once_per_goal_version():
    track_mask = torch.tensor([True, False, False, True])
    command = _minimal_hmi_command(track_mask)
    command.motion = SimpleNamespace(has_object=True)
    command.hmi_goal_object_pos_w = torch.zeros((4, 3), dtype=torch.float32)
    command.hmi_goal_object_quat_w = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0]] * 4, dtype=torch.float32
    )
    command.hmi_goal_version = torch.ones(4, dtype=torch.long)
    command._simulator_object_state_snapshot = torch.zeros((4, 13), dtype=torch.float32)
    command._simulator_object_state_snapshot[:, 6] = 1.0
    command._simulator_object_state_snapshot[2, 0] = 1.0
    command._simulator_object_state_snapshot_ready = True

    env = SimpleNamespace(
        num_envs=4,
        device=torch.device("cpu"),
        dt=0.02,
        command_manager=_CommandManager(command),
    )
    term = HMIObjectGoalReachedOnce(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:HMIObjectGoalReachedOnce",
            params={"pos_threshold": 0.2, "ori_threshold": 0.5, "bonus": 3.0},
        ),
        env,
    )

    first = term(env)
    second = term(env)
    command.hmi_goal_version[1] += 1
    third = term(env)

    assert torch.equal(first, torch.tensor([0.0, 150.0, 0.0, 0.0]))
    assert torch.count_nonzero(second) == 0
    assert torch.equal(third, torch.tensor([0.0, 150.0, 0.0, 0.0]))


@pytest.mark.parametrize("goal_target", ("object_xy", "root_xy"))
def test_hmi_xy_sparse_goal_ignores_z_and_orientation(goal_target):
    track_mask = torch.tensor([True, False, False, True])
    command = _minimal_hmi_command(track_mask)
    interface = (
        "actor93_depth5046_action29_terminal_object_xy_v2"
        if goal_target == "object_xy"
        else "actor93_depth5046_action29_terminal_root_xy_v2"
    )
    command.hmi_cfg = HMIMotionConfig(
        actor_interface_semantics=interface,
        goal_target=goal_target,
    )
    command.hmi_goal_xy_pos_w = torch.tensor(
        [[0.0, 0.0, 50.0], [0.0, 0.0, 50.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    command.hmi_goal_version = torch.ones(4, dtype=torch.long)
    command.motion = SimpleNamespace(has_object=True)
    command._simulator_object_state_snapshot = torch.zeros((4, 13), dtype=torch.float32)
    command._simulator_object_state_snapshot[:, 6] = 1.0
    command._simulator_object_state_snapshot_ready = True
    command.ref_body_index = 0
    command._env = SimpleNamespace(
        simulator=SimpleNamespace(
            _rigid_body_pos=torch.zeros((4, 1, 3), dtype=torch.float32),
            _rigid_body_rot=torch.tensor(
                [[[0.0, 0.0, 0.0, 1.0]]] * 4, dtype=torch.float32
            ),
        )
    )
    env = SimpleNamespace(
        num_envs=4,
        device=torch.device("cpu"),
        dt=0.02,
        command_manager=_CommandManager(command),
    )
    term = HMIXYGoalReachedOnce(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:HMIXYGoalReachedOnce",
            params={"goal_target": goal_target, "pos_threshold": 0.2, "bonus": 3.0},
        ),
        env,
    )

    reward = term(env)

    assert torch.equal(reward, torch.tensor([0.0, 150.0, 0.0, 0.0]))


def test_hmi_depth_actor_real_onnx_checker_and_ort_parity(tmp_path):
    config = apply_perception_overrides(g1_29dof_wbt_w_object_hmi_depth_stage1)
    obs_dims = {
        "actor_obs_hmi_goal_command": 3,
        "actor_obs_drop_button": 1,
        "actor_obs_proprio_with_actions_no_linvel": 90,
        "perception_obs": 58 * 87,
    }
    actor = PPOActorEncoder(
        obs_dim_dict=obs_dims,
        module_config_dict=config.algo.config.module_dict.actor,
        num_actions=29,
        init_noise_std=1.0,
        history_length={name: 1 for name in obs_dims},
    )
    wrapper = _HMIActorONNXWrapper(actor).eval()
    assert actor.actor_module.module[0].in_features == 94 + 32

    example = {
        "actor_obs": torch.zeros((1, 94), dtype=torch.float32),
        "perception_obs": torch.zeros((1, 58 * 87), dtype=torch.float32),
    }
    onnx_path = tmp_path / "hmi_depth_actor.onnx"
    export_policy_as_onnx(
        wrapper,
        str(onnx_path),
        example,
        perception_input_name="perception_obs",
    )
    report = validate_exported_policy_onnx(
        wrapper=wrapper,
        onnx_file_path=str(onnx_path),
        example_obs_dict=example,
        perception_input_name="perception_obs",
    )

    assert report["checker"] == "onnx.checker.check_model"
    assert report["runtime"] == "onnxruntime_cpu"
    assert report["pytorch_vs_ort"] is True
    assert report["input_names"] == ["actor_obs", "perception_obs"]
    assert report["output_names"] == ["action"]


@pytest.mark.parametrize(
    "raw_config",
    (
        g1_29dof_wbt_w_object_hmi_depth_stage2_object_xy,
        g1_29dof_wbt_w_object_hmi_depth_stage2_root_xy,
    ),
)
def test_hmi_xy_actor_is_93d_and_has_real_onnx_parity(tmp_path, raw_config):
    config = apply_perception_overrides(raw_config)
    obs_dims = {
        "actor_obs_hmi_goal_command": 2,
        "actor_obs_drop_button": 1,
        "actor_obs_proprio_with_actions_no_linvel": 90,
        "perception_obs": 58 * 87,
    }
    actor = PPOActorEncoder(
        obs_dim_dict=obs_dims,
        module_config_dict=config.algo.config.module_dict.actor,
        num_actions=29,
        init_noise_std=1.0,
        history_length={name: 1 for name in obs_dims},
    )
    wrapper = _HMIActorONNXWrapper(actor).eval()
    assert actor.actor_module.module[0].in_features == 93 + 32
    example = {
        "actor_obs": torch.zeros((1, 93), dtype=torch.float32),
        "perception_obs": torch.zeros((1, 58 * 87), dtype=torch.float32),
    }
    onnx_path = tmp_path / f"{config.training.name}.onnx"
    export_policy_as_onnx(
        wrapper,
        str(onnx_path),
        example,
        perception_input_name="perception_obs",
    )
    report = validate_exported_policy_onnx(
        wrapper=wrapper,
        onnx_file_path=str(onnx_path),
        example_obs_dict=example,
        perception_input_name="perception_obs",
    )
    assert report["pytorch_vs_ort"] is True
