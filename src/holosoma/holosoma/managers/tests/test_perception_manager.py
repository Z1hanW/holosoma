import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from yourdfpy import URDF

from holosoma.config_values import perception as perception_presets
from holosoma.managers.perception import manager as perception_manager_module
from holosoma.managers.perception.manager import PerceptionManager
from holosoma.managers.randomization.terms.locomotion import (
    _camera_raycast_enabled,
    randomize_camera_raycast,
)
from holosoma.utils.rotations import quat_from_euler_xyz
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.simulator_config import SimulatorType


def test_camera_depth_d435i_matches_rev1_urdf_and_neutral_ankle_midpoint_pose() -> None:
    cfg = perception_presets.camera_depth_d435i
    assert cfg.sensor_offset == pytest.approx([0.0576235, 0.01753, 0.42987])

    urdf_path = Path(__file__).resolve().parents[2] / "data/robots/g1/g1_29dof.urdf"
    robot_urdf = URDF.load(str(urdf_path), load_meshes=False)
    robot_urdf.update_cfg({name: 0.0 for name in robot_urdf.actuated_joint_names})

    torso_to_camera = robot_urdf.get_transform("d435_link", "torso_link")
    np.testing.assert_allclose(
        torso_to_camera[:3, 3],
        cfg.sensor_offset,
        atol=1.0e-9,
        rtol=0.0,
    )

    left_foot = robot_urdf.get_transform("left_ankle_roll_link")[:3, 3]
    right_foot = robot_urdf.get_transform("right_ankle_roll_link")[:3, 3]
    camera = robot_urdf.get_transform("d435_link")[:3, 3]
    midfeet_to_camera = camera - 0.5 * (left_foot + right_foot)
    np.testing.assert_allclose(
        midfeet_to_camera,
        [0.05366232609678057, 0.01753, 1.230733752422211],
        atol=1.0e-9,
        rtol=0.0,
    )

    mount_rpy = torch.tensor([0.0, 0.8307767239493009, 0.0])
    expected_mount_quat = quat_from_euler_xyz(
        mount_rpy[0],
        mount_rpy[1],
        mount_rpy[2],
    )
    assert torch.allclose(
        torch.tensor(cfg.camera_mount_quat),
        expected_mount_quat,
        atol=1.0e-7,
        rtol=0.0,
    )
    assert cfg.camera_frame_quat == [-0.5, 0.5, -0.5, 0.5]


def test_camera_depth_d435i_corl_freezes_historical_mount_and_residual_pitch() -> None:
    cfg = perception_presets.camera_depth_d435i_corl

    assert cfg.camera_body_name == "torso_link"
    assert cfg.sensor_offset == pytest.approx([0.01, 0.01, 0.44])
    assert cfg.camera_mount_quat == pytest.approx(
        [0.00644801, 0.23350163, 0.00644801, 0.97231365]
    )
    assert cfg.camera_frame_quat == [-0.5, 0.5, -0.5, 0.5]
    assert cfg.camera_pitch_deg == pytest.approx(10.0)


def test_strict_camera_mount_rotation_is_derived_from_configured_quaternion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        perception_manager_module,
        "get_simulator_type",
        lambda: SimulatorType.ISAACSIM,
    )
    cfg = replace(
        perception_presets.camera_depth_d435i,
        camera_warp_enable_holes=False,
    )
    manager = PerceptionManager(
        cfg,
        SimpleNamespace(num_envs=1, logger=None, randomization_manager=None),
        "cpu",
    )

    assert torch.allclose(
        manager._strict_camera_mount_rotation_deg,
        torch.tensor([0.0, 47.6, 0.0]),
        atol=1.0e-5,
        rtol=0.0,
    )


def _make_checkpoint_perception_manager() -> PerceptionManager:
    manager = object.__new__(PerceptionManager)
    manager.enabled = True
    manager.num_envs = 2
    manager.device = "cpu"
    manager.cfg = SimpleNamespace(
        output_mode="camera_depth",
        camera_fps=30.0,
        camera_near=0.3,
        camera_far=3.0,
        max_distance=3.0,
        camera_pitch_deg=0.0,
        camera_target_pitch_deg=None,
        camera_distortion=[0.0] * 5,
        use_heading_only=True,
        heightmap_obs_offset=0.5,
    )
    manager.env = SimpleNamespace(randomization_manager=None)
    manager._camera_source = "far_tracking_warp"
    manager._simulator_backend = "isaacgym"
    manager._is_mujoco_perception = False
    manager._sensor_offset = torch.tensor([0.01, 0.01, 0.44], dtype=torch.float32)
    manager._camera_height = 2
    manager._camera_width = 3
    manager._camera_fx = torch.tensor(2.0)
    manager._camera_fy = torch.tensor(2.0)
    manager._camera_cx = torch.tensor(1.5)
    manager._camera_cy = torch.tensor(1.0)
    manager._camera_vfov_deg = 60.0
    manager._camera_hfov_deg = 90.0
    manager._camera_obs_height = 2
    manager._camera_obs_width = 3
    manager._camera_body_name = "torso_link"
    manager._camera_body_index = 4
    manager._camera_body_offset_pos = torch.zeros(3)
    manager._camera_body_offset_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    manager._rendered_camera_env_id = 0
    manager._camera_mount_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    manager._use_camera_mount_quat = True
    manager._camera_frame_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    manager._use_camera_frame_quat = True
    manager._camera_auto_fix_backward = False
    manager._camera_backward_ratio_threshold = 0.6
    manager._strict_camera_mount_rotation_deg = torch.tensor([1.0, 27.0, 1.0])
    manager._heightmap_grid_x = 17
    manager._heightmap_grid_y = 17
    manager._heightmap_interval_x = 0.1
    manager._heightmap_interval_y = 0.1
    manager._heightmap_body_name = "pelvis"
    manager._heightmap_body_index = 0
    manager._heightmap_body_offset_pos = torch.zeros(3)
    manager._heightmap_body_offset_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    manager._ray_start_offset = torch.tensor([0.0, 0.0, 20.0])
    manager._camera_strict_warp = True
    manager._camera_disable_offsets = False
    manager._update_interval = 1.0 / 30.0
    manager._camera_warp_preprocess = True
    manager._camera_warp_freq_ratio = 2
    manager._camera_warp_buffer_len = 6
    manager._camera_warp_latency_frame = 3
    manager._camera_warp_latency_frame_range = (3, 4)
    manager._camera_warp_depth_offset = torch.tensor([0.1, -0.2], dtype=torch.float32)
    manager._camera_ray_correction_quat = torch.tensor(
        [0.0, 0.0, 0.0, 1.0],
        dtype=torch.float32,
    )
    manager._camera_obs_step_counter = 7
    manager._time_since_update = 0.013
    manager._camera_warp_hole_generator = SimpleNamespace(
        shape=(64, 96),
        resolutions=[(2, 2), (4, 4)],
        periods=[32, 16],
        factors=[0.3, 0.09],
        frame_idx=11,
        gradient_cache=[{1: torch.ones(1)}, {2: torch.ones(1)}],
    )
    manager._shared_camera_sensor_local_position = torch.tensor(
        [[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]],
        dtype=torch.float32,
    )
    identity = torch.zeros((2, 1, 4), dtype=torch.float32)
    identity[..., 3] = 1.0
    manager._shared_camera_sensor_local_orientation = identity.clone()
    manager._shared_camera_sensor_data_frame_quat = identity.clone()
    manager._far_tracking_camera_sensor = SimpleNamespace(
        camera_sensor_local_position=torch.zeros((2, 1, 3), dtype=torch.float32),
        camera_sensor_local_orientation=torch.zeros((2, 1, 4), dtype=torch.float32),
        camera_sensor_data_frame_quat=torch.zeros((2, 1, 4), dtype=torch.float32),
    )
    manager._far_tracking_geometry_fingerprint = (
        ("torso_link", ".stl", 3, "a" * 64),
    )
    manager._far_tracking_base_link_indices = torch.tensor([4], dtype=torch.long)
    manager._far_tracking_robot_slot_indices = torch.tensor([0], dtype=torch.long)
    manager._far_tracking_robot_body_indices = torch.tensor([4], dtype=torch.long)
    manager._far_tracking_robot_body_names = ["torso_link"]
    manager._far_tracking_robot_body_offset_pos = torch.zeros((1, 3))
    manager._far_tracking_robot_body_offset_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    manager._far_tracking_object_slot_indices = torch.empty((0,), dtype=torch.long)
    manager._far_tracking_object_source_indices = torch.empty((0,), dtype=torch.long)
    manager._far_tracking_primitive_source_indices = torch.empty((0,), dtype=torch.long)
    manager._far_tracking_object_names = []
    manager._far_tracking_object_active_env_ids = []
    manager._camera_ray_dirs_base = None
    manager._camera_scandots_ray_dirs_base = None
    return manager


def _set_single_object_geometry(
    manager: PerceptionManager,
    *,
    digest: str,
    size_bytes: int,
) -> None:
    manager._far_tracking_geometry_fingerprint = (
        ("torso_link", ".stl", 3, "a" * 64),
        ("object__variant_000", ".obj", size_bytes, digest),
    )
    manager._far_tracking_object_slot_indices = torch.tensor([1], dtype=torch.long)
    manager._far_tracking_object_source_indices = torch.tensor([0], dtype=torch.long)
    manager._far_tracking_object_names = ["object"]
    manager._far_tracking_object_active_env_ids = [None]


def test_perception_persistent_checkpoint_round_trip_is_atomic_and_non_aliasing() -> None:
    manager = _make_checkpoint_perception_manager()
    state = manager.get_persistent_checkpoint_state()
    expected_offset = state["camera_warp_depth_offset"].clone()
    expected_position = state["shared_mount"]["local_position"].clone()

    manager._camera_warp_depth_offset.zero_()
    manager._shared_camera_sensor_local_position.zero_()
    manager._camera_ray_correction_quat.copy_(torch.tensor([0.0, 0.0, 1.0, 0.0]))
    manager._camera_obs_step_counter = 99
    manager._time_since_update = 0.5
    manager._camera_warp_hole_generator.frame_idx = 100
    manager._camera_warp_hole_frame_stats = (100, torch.tensor(-1.0), torch.tensor(1.0))

    assert torch.equal(state["camera_warp_depth_offset"], expected_offset)
    assert torch.equal(state["shared_mount"]["local_position"], expected_position)

    manager.load_persistent_checkpoint_state(state)

    assert torch.equal(manager._camera_warp_depth_offset, expected_offset)
    assert torch.equal(manager._shared_camera_sensor_local_position, expected_position)
    assert manager._camera_obs_step_counter == 7
    assert manager._time_since_update == pytest.approx(0.013)
    assert manager._camera_warp_hole_generator.frame_idx == 11
    assert manager._camera_warp_hole_generator.gradient_cache == [{}, {}]
    assert manager._camera_warp_hole_frame_stats is None
    sensor = manager._far_tracking_camera_sensor
    assert torch.equal(sensor.camera_sensor_local_position, expected_position)
    assert torch.equal(
        sensor.camera_sensor_local_orientation,
        manager._shared_camera_sensor_local_orientation,
    )
    assert torch.equal(
        sensor.camera_sensor_data_frame_quat,
        manager._shared_camera_sensor_data_frame_quat,
    )

    corrupt = copy.deepcopy(state)
    corrupt["shared_mount"]["local_position"][0, 0, 0] = float("nan")
    before = manager._camera_warp_depth_offset.clone()
    before_position = manager._shared_camera_sensor_local_position.clone()
    with pytest.raises(ValueError, match="finite floating"):
        manager.load_persistent_checkpoint_state(corrupt)
    assert torch.equal(manager._camera_warp_depth_offset, before)
    assert torch.equal(manager._shared_camera_sensor_local_position, before_position)

    nonunit = copy.deepcopy(state)
    nonunit["camera_ray_correction_quat"] *= 2.0
    with pytest.raises(ValueError, match="unit quaternions"):
        manager.load_persistent_checkpoint_state(nonunit)
    assert torch.equal(manager._camera_warp_depth_offset, before)
    assert torch.equal(manager._shared_camera_sensor_local_position, before_position)

    manager._sensor_offset[0] += 0.1
    with pytest.raises(ValueError, match="semantics differ"):
        manager.load_persistent_checkpoint_state(state)
    assert torch.equal(manager._camera_warp_depth_offset, before)
    assert torch.equal(manager._shared_camera_sensor_local_position, before_position)


@pytest.mark.parametrize(
    ("owner", "attribute", "replacement"),
    [
        ("manager", "_camera_fx", torch.tensor(2.5)),
        ("manager", "_camera_body_offset_pos", torch.tensor([0.01, 0.0, 0.0])),
        ("manager", "_heightmap_grid_x", 19),
        ("manager", "_heightmap_interval_y", 0.2),
        ("cfg", "max_distance", 4.0),
        ("cfg", "heightmap_obs_offset", 0.25),
    ],
)
def test_perception_checkpoint_rejects_changed_effective_geometry(
    owner: str,
    attribute: str,
    replacement,
) -> None:
    manager = _make_checkpoint_perception_manager()
    state = manager.get_persistent_checkpoint_state()

    setattr(manager if owner == "manager" else manager.cfg, attribute, replacement)

    with pytest.raises(ValueError, match="semantics differ"):
        manager.load_persistent_checkpoint_state(state)


def test_far_tracking_geometry_fingerprint_is_content_addressed_and_path_independent(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left" / "mesh.obj"
    right = tmp_path / "right" / "generated-cache-name.obj"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_bytes(b"same mesh bytes")
    right.write_bytes(b"same mesh bytes")

    left_identity = PerceptionManager._fingerprint_far_tracking_geometry(
        {"slot": str(left)},
        asset_meshes_root=tmp_path,
    )
    right_identity = PerceptionManager._fingerprint_far_tracking_geometry(
        {"slot": str(right)},
        asset_meshes_root=tmp_path,
    )
    assert left_identity == right_identity

    right.write_bytes(b"different mesh bytes")
    changed_identity = PerceptionManager._fingerprint_far_tracking_geometry(
        {"slot": str(right)},
        asset_meshes_root=tmp_path,
    )
    assert changed_identity != left_identity


def test_far_tracking_fixed_link_mesh_pose_applies_parent_to_child_transform() -> None:
    half_turn_z = torch.tensor(
        [[[0.0, 0.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )
    parent_position = torch.tensor([[[1.0, 2.0, 3.0]]])
    parent_orientation = half_turn_z
    child_offset_position = torch.tensor([[[0.25, 0.0, 0.0]]])
    child_offset_orientation = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])

    child_position, child_orientation = PerceptionManager._compose_fixed_body_pose(
        parent_position,
        parent_orientation,
        child_offset_position,
        child_offset_orientation,
    )

    assert torch.allclose(child_position, torch.tensor([[[0.75, 2.0, 3.0]]]), atol=1.0e-6)
    assert torch.allclose(child_orientation, half_turn_z, atol=1.0e-6)


def test_perception_checkpoint_rejects_changed_camera_randomization_ranges() -> None:
    manager = _make_checkpoint_perception_manager()
    term = SimpleNamespace(
        func="holosoma.managers.randomization.terms.locomotion:randomize_camera_raycast",
        params={
            "enabled": True,
            "translation_range": {
                "x": [-0.01, 0.01],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
            },
            "rotation_range_deg": [-2.0, 2.0],
            "noise_std_mult_range": [0.0, 0.05],
            "noise_drop_prob_range": [0.0, 0.025],
        },
    )
    manager.env.randomization_manager = SimpleNamespace(
        cfg=SimpleNamespace(reset_terms={"camera": term})
    )
    state = manager.get_persistent_checkpoint_state()

    term.params["rotation_range_deg"] = [-3.0, 3.0]

    with pytest.raises(ValueError, match="semantics differ"):
        manager.load_persistent_checkpoint_state(state)


@pytest.mark.parametrize(
    ("semantics", "scope", "advances", "equivalence"),
    [
        ("legacy_full_v1", "full_vectorized_batch", True, "not_replayable_one_env"),
        ("targeted_v2", "reset_env_subset", False, "distribution_only"),
    ],
)
def test_observation_contract_authenticates_reset_refresh_lifecycle(
    semantics: str,
    scope: str,
    advances: bool,
    equivalence: str,
) -> None:
    manager = _make_checkpoint_perception_manager()
    manager._reset_refresh_semantics = semantics

    lifecycle = manager.get_observation_contract()["producer_lifecycle"]

    assert lifecycle["reset_refresh_semantics"] == semantics
    assert lifecycle["ordinary_manager_update_calls_per_control_tick"] == 1
    assert lifecycle["initialization_control_ticks_before_first_reset_output"] == 1
    assert lifecycle["initialization_ordinary_manager_update_calls_before_first_reset_output"] == 1
    assert lifecycle["reset_output_republished_until_physics_advances"] is True
    assert lifecycle["reset_output_scope"] == scope
    assert lifecycle["hole_clock_advances_on_reset_refresh"] is advances
    assert lifecycle["camera_frequency_phase_advances_on_reset_refresh"] is advances
    # This fixture has a randomized latency frame, so even targeted subset
    # refresh consumes the process-global RNG and shifts future peer samples.
    assert lifecycle["camera_producer_reset_refresh_consumes_process_global_rng"] is True
    assert lifecycle["future_noise_sample_path_peer_reset_coupled"] is True
    assert lifecycle["batch_size_invariant_sample_path"] is False
    assert lifecycle["stochastic_equivalence"] == equivalence
    assert lifecycle["seed_replay_scope"] == "same_execution_trace_only"


def test_observation_contract_authenticates_portable_training_geometry_support() -> None:
    manager = _make_checkpoint_perception_manager()
    manager._reset_refresh_semantics = "targeted_v2"

    support = manager.get_local_geometry_support()
    contract = manager.get_observation_contract()

    assert contract["training_geometry_support"] == support
    assert support["training_rank_count"] == 1
    assert support["robot_mesh_bindings"] == [
        {
            "slot_name": "torso_link",
            "mesh": {"suffix": ".stl", "size_bytes": 3, "sha256": "a" * 64},
            "tracking_body_name": "torso_link",
            "fixed_position_xyz": [0.0, 0.0, 0.0],
            "fixed_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    ]
    assert support["object_mesh_support"] == []


def test_training_geometry_support_unions_rank_shards_and_direct_checks_membership() -> None:
    first = _make_checkpoint_perception_manager()
    second = _make_checkpoint_perception_manager()
    _set_single_object_geometry(first, digest="b" * 64, size_bytes=11)
    _set_single_object_geometry(second, digest="c" * 64, size_bytes=13)

    aggregated = PerceptionManager.aggregate_training_geometry_support(
        [first.get_local_geometry_support(), second.get_local_geometry_support()]
    )

    assert aggregated["training_rank_count"] == 2
    assert [item["mesh"]["sha256"] for item in aggregated["object_mesh_support"]] == [
        "b" * 64,
        "c" * 64,
    ]
    assert [item["training_active_env_count"] for item in aggregated["object_mesh_support"]] == [
        2,
        2,
    ]
    second.validate_deployment_geometry_support(aggregated)

    unknown = _make_checkpoint_perception_manager()
    _set_single_object_geometry(unknown, digest="d" * 64, size_bytes=17)
    with pytest.raises(ValueError, match="not a member"):
        unknown.validate_deployment_geometry_support(aggregated)
    assert (
        unknown.validate_deployment_geometry_support(
            aggregated,
            allow_unknown_object_geometry=True,
        )
        == aggregated
    )

    with pytest.raises(TypeError, match="must be a bool"):
        unknown.validate_deployment_geometry_support(
            aggregated,
            allow_unknown_object_geometry=1,
        )


def test_direct_manager_publishes_exact_onnx_contract_only_after_live_validation() -> None:
    training_rank = _make_checkpoint_perception_manager()
    direct = _make_checkpoint_perception_manager()
    _set_single_object_geometry(training_rank, digest="b" * 64, size_bytes=11)
    _set_single_object_geometry(direct, digest="b" * 64, size_bytes=11)
    training_rank._reset_refresh_semantics = "targeted_v2"
    direct._reset_refresh_semantics = "targeted_v2"
    support = training_rank.get_local_geometry_support()
    contract = training_rank.get_observation_contract(
        training_geometry_support=support
    )
    digest = training_rank.get_observation_contract_sha256(
        training_geometry_support=support
    )
    wire_contract = perception_manager_module.json.loads(
        perception_manager_module.json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )

    assert direct.authenticate_observation_contract(
        wire_contract,
        declared_sha256=digest,
    ) == digest
    assert direct.get_observation_contract() == wire_contract
    assert direct.get_observation_contract_sha256() == digest

    changed = copy.deepcopy(wire_contract)
    changed["producer_tick_dt"] = 0.123
    changed_digest = perception_manager_module.hashlib.sha256(
        perception_manager_module.json.dumps(
            changed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="transform/noise/cadence"):
        direct.authenticate_observation_contract(
            changed,
            declared_sha256=changed_digest,
        )


def test_perception_canonical_reset_preserves_stream_clock_and_clears_only_derived_cache() -> None:
    manager = _make_checkpoint_perception_manager()
    manager._camera_warp_hole_frame_stats = (10, torch.tensor(-1.0), torch.tensor(1.0))

    manager.reset_canonical_rollout_state()

    assert manager._camera_obs_step_counter == 7
    assert manager._time_since_update == pytest.approx(0.013)
    assert manager._camera_warp_hole_generator.frame_idx == 11
    assert manager._camera_warp_hole_generator.gradient_cache == [{}, {}]
    assert manager._camera_warp_hole_frame_stats is None


def test_perception_checkpoint_boundary_matches_fresh_process_resume_stream() -> None:
    uninterrupted = _make_checkpoint_perception_manager()
    uninterrupted._camera_warp_hole_generator = perception_manager_module._InfiniteFractalPerlin3D(
        (4, 4),
        [(2, 2)],
        [4],
        [1.0],
        batch_size=2,
        device="cpu",
        seed_semantics="rank_local_v2",
        effective_seed=123,
    )
    uninterrupted._camera_warp_hole_generator.frame_idx = 11
    checkpoint_state = uninterrupted.get_persistent_checkpoint_state()
    checkpoint_rng = torch.get_rng_state().clone()

    def canonical_warmup(manager: PerceptionManager) -> tuple[torch.Tensor, ...]:
        manager.reset_canonical_rollout_state()
        manager.cfg.camera_source = "far_tracking_warp"
        manager.env.device = "cpu"
        manager.env.num_envs = manager.num_envs
        manager.env.perception_manager = manager
        randomize_camera_raycast(
            manager.env,
            torch.arange(manager.num_envs),
            translation_range={
                "x": [-0.02, 0.02],
                "y": [-0.01, 0.01],
                "z": [-0.03, 0.03],
            },
            rotation_range_deg={
                "roll": [-2.0, 2.0],
                "pitch": [-3.0, 3.0],
                "yaw": [-4.0, 4.0],
            },
            noise_std_mult_range=[0.0, 0.05],
            noise_drop_prob_range=[0.0, 0.025],
        )
        warmup_random = torch.rand(4)
        manager._camera_obs_step_counter += 1
        manager._time_since_update += 0.02
        next_hole_frame = manager._camera_warp_hole_generator.generate_frame()
        next_random = torch.rand(4)
        return (
            manager.env._perception_camera_offset_pos.clone(),
            manager.env._perception_camera_offset_rpy.clone(),
            manager.env._perception_camera_offset_quat.clone(),
            manager.env._perception_camera_noise_std_mult.clone(),
            manager.env._perception_camera_noise_drop_prob.clone(),
            warmup_random,
            next_hole_frame,
            next_random,
        )

    uninterrupted_result = canonical_warmup(uninterrupted)
    uninterrupted_state = uninterrupted.get_persistent_checkpoint_state()

    resumed = _make_checkpoint_perception_manager()
    resumed._camera_warp_hole_generator = perception_manager_module._InfiniteFractalPerlin3D(
        (4, 4),
        [(2, 2)],
        [4],
        [1.0],
        batch_size=2,
        device="cpu",
        seed_semantics="rank_local_v2",
        effective_seed=123,
    )
    resumed._camera_warp_depth_offset.fill_(9.0)
    resumed._camera_obs_step_counter = 0
    resumed._time_since_update = 0.0
    resumed._camera_warp_hole_generator.frame_idx = 0
    resumed.load_persistent_checkpoint_state(checkpoint_state)
    torch.set_rng_state(checkpoint_rng)
    resumed_result = canonical_warmup(resumed)
    resumed_state = resumed.get_persistent_checkpoint_state()

    assert all(torch.equal(left, right) for left, right in zip(uninterrupted_result, resumed_result, strict=True))
    assert uninterrupted_state["camera_obs_step_counter"] == resumed_state["camera_obs_step_counter"]
    assert uninterrupted_state["time_since_update"] == pytest.approx(resumed_state["time_since_update"])
    assert uninterrupted_state["hole_frame_idx"] == resumed_state["hole_frame_idx"]
    assert torch.equal(
        uninterrupted_state["camera_warp_depth_offset"],
        resumed_state["camera_warp_depth_offset"],
    )


def test_perception_checkpoint_rejects_rank_local_hole_seed_change_atomically() -> None:
    manager = _make_checkpoint_perception_manager()
    manager._camera_warp_hole_generator = perception_manager_module._InfiniteFractalPerlin3D(
        (4, 4),
        [(2, 2)],
        [4],
        [1.0],
        batch_size=2,
        device="cpu",
        seed_semantics="rank_local_v2",
        effective_seed=123,
    )
    state = manager.get_persistent_checkpoint_state()
    manager._camera_warp_hole_generator = perception_manager_module._InfiniteFractalPerlin3D(
        (4, 4),
        [(2, 2)],
        [4],
        [1.0],
        batch_size=2,
        device="cpu",
        seed_semantics="rank_local_v2",
        effective_seed=124,
    )
    before_offset = manager._camera_warp_depth_offset.clone()
    before_frame = manager._camera_warp_hole_generator.frame_idx

    with pytest.raises(ValueError, match="semantics differ"):
        manager.load_persistent_checkpoint_state(state)

    assert torch.equal(manager._camera_warp_depth_offset, before_offset)
    assert manager._camera_warp_hole_generator.frame_idx == before_frame


def test_perception_checkpoint_load_rebuilds_camera_ray_caches() -> None:
    manager = _make_checkpoint_perception_manager()
    manager.cfg = SimpleNamespace(
        output_mode="camera_depth",
        camera_pitch_deg=0.0,
        camera_scandots_width=1,
        camera_scandots_height=1,
        camera_scandots_stride=1,
    )
    manager._camera_width = 2
    manager._camera_height = 1
    manager._camera_fx = torch.tensor(1.0)
    manager._camera_fy = torch.tensor(1.0)
    manager._camera_cx = torch.tensor(0.5)
    manager._camera_cy = torch.tensor(0.0)
    manager._camera_strict_warp = False
    manager._use_camera_mount_quat = False
    manager._camera_mount_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    manager._use_camera_frame_quat = False
    manager._camera_frame_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    expected_camera_rays = manager._build_camera_rays()
    expected_scandots_rays = manager._build_camera_scandots_rays()
    state = manager.get_persistent_checkpoint_state()

    manager._camera_ray_correction_quat.copy_(torch.tensor([0.0, 0.0, 1.0, 0.0]))
    manager._camera_ray_dirs_base = manager._build_camera_rays()
    manager._camera_scandots_ray_dirs_base = manager._build_camera_scandots_rays()
    stale_camera_rays = manager._camera_ray_dirs_base.clone()

    manager.load_persistent_checkpoint_state(state)

    assert not torch.allclose(stale_camera_rays, expected_camera_rays)
    assert torch.allclose(manager._camera_ray_dirs_base, expected_camera_rays)
    assert torch.allclose(manager._camera_scandots_ray_dirs_base, expected_scandots_rays)


def test_strict_rendered_camera_mount_is_initialized_eagerly_during_setup() -> None:
    manager = object.__new__(PerceptionManager)
    manager.enabled = True
    manager.cfg = SimpleNamespace(output_mode="camera_depth")
    manager._camera_strict_warp = True
    calls: list[str] = []
    manager._ensure_shared_strict_warp_camera_mount = lambda: calls.append("mount")
    manager._uses_raycast = lambda: False
    manager._uses_camera_raycast = lambda: False
    manager._uses_camera_far_tracking = lambda: False
    manager._uses_camera_scandots = lambda: False
    manager._uses_pytorch3d = lambda: False
    manager._uses_rendered_camera = lambda: False
    manager._wants_camera_scandots = lambda: False
    manager._maybe_fix_camera_backward = lambda: None
    manager._log_camera_ray_alignment = lambda: None

    manager.setup()

    assert calls == ["mount"]


def test_isaac_rendered_camera_exact_resume_fails_closed() -> None:
    manager = _make_checkpoint_perception_manager()
    manager._camera_source = "rendered"

    with pytest.raises(RuntimeError, match="external annotator/render-frame queue"):
        manager.validate_exact_resume_supported()

    manager._is_mujoco_perception = True
    manager.validate_exact_resume_supported()


def test_enabled_heightmap_requires_stream_phase_checkpoint_state() -> None:
    manager = _make_checkpoint_perception_manager()
    manager.cfg.output_mode = "heightmap"

    assert manager.persistent_checkpoint_state_required() is True


def _make_depth_preprocessing_manager(cfg) -> PerceptionManager:
    manager = object.__new__(PerceptionManager)
    manager.cfg = cfg
    manager._camera_warp_preprocess = bool(cfg.camera_warp_preprocess)
    manager._camera_warp_crop_top = int(cfg.camera_warp_crop_top)
    manager._camera_warp_crop_bottom = int(cfg.camera_warp_crop_bottom)
    manager._camera_warp_crop_left = int(cfg.camera_warp_crop_left)
    manager._camera_warp_crop_right = int(cfg.camera_warp_crop_right)
    manager._camera_obs_height = int(cfg.camera_warp_resize[0])
    manager._camera_obs_width = int(cfg.camera_warp_resize[1])
    manager._camera_warp_min_valid_depth = float(cfg.camera_warp_min_valid_depth)
    manager._camera_warp_edge_noise = False
    manager._camera_warp_enable_holes = False
    manager._camera_warp_hole_prob = 0.0
    manager._camera_warp_additive_noise_std = 0.0
    manager._camera_warp_depth_offset_std = 0.0
    manager._camera_warp_normalize = bool(cfg.camera_warp_normalize)
    return manager


@pytest.mark.parametrize(
    "preset",
    [
        perception_presets.camera_depth_d435i_defm_vit_s14,
        perception_presets.camera_depth_d435i_defm_regnet_y_800mf,
        perception_presets.camera_depth_d435i_defm_efficientnet_b2,
    ],
)
def test_defm_camera_preprocessing_preserves_metric_depth_in_meters(preset) -> None:
    manager = _make_depth_preprocessing_manager(preset)
    depth_m = torch.full((2, int(preset.camera_height), int(preset.camera_width)), 1.25)

    processed = manager._process_camera_depth_for_obs(depth_m)

    assert processed.shape == (2, 58, 87)
    assert torch.allclose(processed, torch.full_like(processed, 1.25))


def _make_update_manager(backend: str, *, sensor_noise: bool) -> tuple[PerceptionManager, list[torch.Tensor]]:
    manager = object.__new__(PerceptionManager)
    manager.enabled = True
    manager.num_envs = 3
    manager.device = "cpu"
    manager.cfg = SimpleNamespace(camera_near=0.1, max_distance=3.0, output_mode="camera_depth")
    manager.env = SimpleNamespace(
        dt=0.02,
        _perception_camera_noise_std_mult=torch.zeros(3),
        _perception_camera_noise_drop_prob=torch.tensor([0.0, 0.0, 1.0]),
    )
    manager._camera_apply_sensor_noise = sensor_noise
    manager._debug_update_counter = 0
    manager._update_interval = 0.0
    manager._camera_height = 2
    manager._camera_width = 2
    manager._camera_depth = torch.zeros((3, 2, 2))
    manager._rendered_camera_env_id = 2
    manager._rendered_camera = SimpleNamespace(capture_depth=lambda: torch.ones((1, 2, 2)))
    manager._warned_invalid_rendered_depth = False
    manager.logger = None

    manager._uses_rendered_camera = lambda: backend == "rendered"
    manager._uses_pytorch3d = lambda: False
    manager._uses_camera_far_tracking = lambda: backend == "far_tracking_warp"
    manager._uses_camera_scandots = lambda: False
    manager._uses_camera_raycast = lambda: False
    manager._compute_far_tracking_camera_depth = (
        lambda env_ids=None: torch.ones((3 if env_ids is None else env_ids.numel(), 2, 2))
    )
    manager._consume_camera_obs_refresh_flag = lambda: True
    committed_depths: list[torch.Tensor] = []
    manager._update_camera_depth_observation = (
        lambda _idx, depth, *, refresh, advance_temporal_noise=True: committed_depths.append(
            depth.detach().clone()
        )
    )
    manager._maybe_dump_camera_debug = lambda **_kwargs: None
    manager._maybe_log_runtime_camera_alignment = lambda: None
    return manager, committed_depths


@pytest.mark.parametrize("backend", ["rendered", "far_tracking_warp"])
@pytest.mark.parametrize("sensor_noise", [False, True])
def test_camera_update_applies_sensor_noise_once_with_backend_env_ids(
    backend: str,
    sensor_noise: bool,
) -> None:
    manager, committed_depths = _make_update_manager(backend, sensor_noise=sensor_noise)
    requested_env_ids = torch.tensor([2]) if backend == "rendered" else torch.tensor([2, 0])

    original_apply_noise = manager._apply_camera_depth_noise
    apply_calls: list[torch.Tensor | None] = []

    def tracked_apply_noise(depth: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        apply_calls.append(None if env_ids is None else env_ids.detach().clone())
        return original_apply_noise(depth, env_ids=env_ids)

    manager._apply_camera_depth_noise = tracked_apply_noise
    manager.update(requested_env_ids)

    assert len(apply_calls) == 1
    assert torch.equal(apply_calls[0], requested_env_ids)
    assert len(committed_depths) == 1
    expected = torch.ones_like(committed_depths[0])
    if sensor_noise:
        expected[0] = manager.cfg.max_distance
    assert torch.equal(committed_depths[0], expected)


def test_camera_noise_selects_partial_env_multipliers(monkeypatch: pytest.MonkeyPatch) -> None:
    manager, _ = _make_update_manager("far_tracking_warp", sensor_noise=True)
    manager.env._perception_camera_noise_std_mult = torch.tensor([0.0, 0.5, 1.0])
    manager.env._perception_camera_noise_drop_prob = None
    monkeypatch.setattr(perception_manager_module.torch, "randn_like", torch.ones_like)

    depth = torch.ones((2, 1, 1))
    actual = manager._apply_camera_depth_noise(depth, env_ids=torch.tensor([2, 0]))

    assert torch.equal(actual, torch.tensor([[[2.0]], [[1.0]]]))


@pytest.mark.parametrize(
    ("func", "enabled", "expected"),
    [
        ("holosoma.managers.randomization.terms.locomotion:randomize_camera_raycast", True, True),
        ("holosoma.managers.randomization.terms.locomotion:randomize_camera_raycast", False, False),
        ("holosoma.managers.randomization.terms.locomotion:randomize_action_delay", True, False),
    ],
)
def test_camera_reset_randomization_requires_enabled_matching_term(
    func: str,
    enabled: bool,
    expected: bool,
) -> None:
    manager = object.__new__(PerceptionManager)
    term = SimpleNamespace(func=func, params={"enabled": enabled})
    manager.env = SimpleNamespace(
        randomization_manager=SimpleNamespace(cfg=SimpleNamespace(reset_terms={"term": term}))
    )

    assert manager._has_camera_reset_randomization() is expected


@pytest.mark.parametrize(
    "source",
    ["mesh_raycast", "far_tracking_warp", "rendered", "rendered_depth_sensor"],
)
def test_camera_randomization_gate_covers_every_supported_depth_backend(source: str) -> None:
    env = SimpleNamespace(
        perception_manager=SimpleNamespace(
            cfg=SimpleNamespace(output_mode="camera_depth", camera_source=source)
        )
    )

    assert _camera_raycast_enabled(env) is True


def test_camera_randomization_gate_covers_teacher_only_camera_manager() -> None:
    actor = SimpleNamespace(
        enabled=True,
        cfg=SimpleNamespace(output_mode="heightmap", camera_source="far_tracking_warp"),
    )
    teacher = SimpleNamespace(
        enabled=True,
        cfg=SimpleNamespace(output_mode="camera_depth", camera_source="rendered"),
    )
    env = SimpleNamespace(
        perception_manager=actor,
        teacher_perception_manager=teacher,
        critic_perception_manager=teacher,
        device="cpu",
        num_envs=2,
    )

    assert _camera_raycast_enabled(env) is True
    randomize_camera_raycast(
        env,
        torch.tensor([0, 1]),
        translation_range={
            "x": [-0.01, 0.01],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
        },
        rotation_range_deg={
            "roll": [0.0, 0.0],
            "pitch": [-3.0, 3.0],
            "yaw": [0.0, 0.0],
        },
        noise_std_mult_range=[0.0, 0.05],
        noise_drop_prob_range=[0.0, 0.025],
    )
    assert env._perception_camera_offset_pos.shape == (2, 3)
    assert env._perception_camera_offset_rpy.shape == (2, 3)
    assert env._perception_camera_offset_quat.shape == (2, 4)
    assert env._perception_camera_noise_std_mult.shape == (2,)
    assert env._perception_camera_noise_drop_prob.shape == (2,)


def test_strict_camera_randomization_perturbs_mount_without_rotating_sensor_lever_arm() -> None:
    manager = _make_checkpoint_perception_manager()
    manager._camera_disable_offsets = False
    manager._camera_strict_warp = True
    manager.env._perception_camera_offset_pos = torch.tensor([[0.02, -0.01, 0.03], [0.0, 0.0, 0.0]])
    manager.env._perception_camera_offset_rpy = torch.deg2rad(
        torch.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 0.0]])
    )
    manager.env._perception_camera_offset_quat = torch.zeros((2, 4))
    manager.env._perception_camera_offset_quat[:, 3] = 1.0
    local_position = torch.tensor([[0.01, 0.01, 0.44], [0.01, 0.01, 0.44]])
    base_rpy = torch.deg2rad(manager._strict_camera_mount_rotation_deg)
    local_orientation = quat_from_euler_xyz(
        base_rpy[0].repeat(2),
        base_rpy[1].repeat(2),
        base_rpy[2].repeat(2),
    )

    actual_position, actual_orientation = manager._apply_runtime_camera_mount_offsets(
        local_position,
        local_orientation,
        idx=slice(None),
    )

    assert torch.allclose(
        actual_position[0],
        torch.tensor([0.03, 0.0, 0.47]),
        atol=1.0e-7,
    )
    expected_rpy = base_rpy + torch.deg2rad(torch.tensor([0.0, 3.0, 0.0]))
    expected_orientation = quat_from_euler_xyz(
        expected_rpy[0],
        expected_rpy[1],
        expected_rpy[2],
    )
    assert torch.allclose(actual_orientation[0], expected_orientation, atol=1.0e-6)


def test_nonstrict_mujoco_render_pose_consumes_mount_local_randomization() -> None:
    manager = object.__new__(PerceptionManager)
    manager.enabled = True
    manager.device = "cpu"
    manager.cfg = SimpleNamespace(output_mode="camera_depth")
    manager._camera_strict_warp = False
    manager._camera_disable_offsets = False
    manager._camera_width = 3
    manager._camera_height = 3
    manager._sensor_offset = torch.tensor([0.0, 0.0, 0.44])
    manager._camera_body_index = 0
    manager._camera_body_offset_pos = torch.zeros(3)
    manager._camera_body_offset_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    identity = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
    manager.env = SimpleNamespace(
        simulator=SimpleNamespace(
            _rigid_body_pos=torch.zeros((1, 1, 3)),
            _rigid_body_rot=identity,
        ),
        _perception_camera_offset_pos=torch.tensor([[0.1, 0.0, 0.0]]),
        _perception_camera_offset_rpy=torch.tensor([[0.0, 0.0, torch.pi / 2.0]]),
        _perception_camera_offset_quat=torch.tensor(
            [[0.0, 0.0, 2.0**-0.5, 2.0**-0.5]]
        ),
    )
    manager._build_camera_rays_from_coords = lambda _u, _v: torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    camera_position, camera_quat = manager.get_mujoco_render_camera_pose()

    assert torch.allclose(camera_position, torch.tensor([[0.1, 0.0, 0.44]]), atol=1.0e-6)
    assert torch.allclose(
        perception_manager_module.quat_apply(
            camera_quat,
            torch.tensor([[1.0, 0.0, 0.0]]),
            w_last=True,
        ),
        torch.tensor([[0.0, 1.0, 0.0]]),
        atol=1.0e-6,
    )
    assert torch.allclose(
        perception_manager_module.quat_apply(
            camera_quat,
            torch.tensor([[0.0, 0.0, -1.0]]),
            w_last=True,
        ),
        torch.tensor([[0.0, 0.0, 1.0]]),
        atol=1.0e-6,
    )


def test_camera_randomization_state_fails_closed_when_configured_sample_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The narrow video-eval escape hatch must never bypass a configured
    # randomizer whose sampled runtime tensors are absent.
    monkeypatch.setenv("HOLOSOMA_EVAL_ALLOW_MISSING_CAMERA_SENSOR_NOISE_STATE", "1")
    manager = object.__new__(PerceptionManager)
    manager.cfg = SimpleNamespace(output_mode="camera_depth")
    manager._camera_randomization_log_done = False
    manager._camera_apply_sensor_noise = True
    manager._camera_source = "far_tracking_warp"
    term = SimpleNamespace(
        func="holosoma.managers.randomization.terms.locomotion:randomize_camera_raycast",
        params={"enabled": True, "noise_std_mult_range": [0.0, 0.05]},
    )
    manager.env = SimpleNamespace(
        randomization_manager=SimpleNamespace(cfg=SimpleNamespace(reset_terms={"camera": term}))
    )

    with pytest.raises(RuntimeError, match="noise_std_mult_range"):
        manager._log_camera_randomization_state_once()


def test_camera_sensor_noise_fails_closed_without_runtime_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HOLOSOMA_EVAL_ALLOW_MISSING_CAMERA_SENSOR_NOISE_STATE", raising=False)
    manager = object.__new__(PerceptionManager)
    manager.cfg = SimpleNamespace(output_mode="camera_depth")
    manager._camera_randomization_log_done = False
    manager._camera_apply_sensor_noise = True
    manager._camera_source = "rendered"
    manager.env = SimpleNamespace(randomization_manager=None)

    with pytest.raises(RuntimeError, match="Refusing to advertise sensor noise while applying none"):
        manager._log_camera_randomization_state_once()


def test_video_eval_can_explicitly_allow_missing_camera_sensor_noise_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOLOSOMA_EVAL_ALLOW_MISSING_CAMERA_SENSOR_NOISE_STATE", "1")
    manager = object.__new__(PerceptionManager)
    manager.cfg = SimpleNamespace(output_mode="camera_depth")
    manager._camera_randomization_log_done = False
    manager._camera_apply_sensor_noise = True
    manager._camera_source = "rendered"
    manager.env = SimpleNamespace(randomization_manager=None)

    manager._log_camera_randomization_state_once()

    assert manager._camera_randomization_log_done is True


class _FarTrackingSubsetSimulatorStub:
    def __init__(self, num_envs: int) -> None:
        self._rigid_body_pos = (
            torch.arange(num_envs * 2 * 3, dtype=torch.float32).view(num_envs, 2, 3) * 0.01
        )
        self._rigid_body_rot = torch.zeros((num_envs, 2, 4), dtype=torch.float32)
        self._rigid_body_rot[..., 3] = 1.0
        self.object_states = torch.zeros((num_envs, 2, 13), dtype=torch.float32)
        env_axis = torch.arange(num_envs, dtype=torch.float32)
        self.object_states[:, 0, :3] = torch.stack(
            (0.10 + env_axis * 0.03, 0.20 + env_axis * 0.01, 0.30 + env_axis * 0.02),
            dim=-1,
        )
        self.object_states[:, 1, :3] = torch.stack(
            (0.40 + env_axis * 0.02, 0.50 + env_axis * 0.03, 0.60 + env_axis * 0.01),
            dim=-1,
        )
        self.object_states[..., 6] = 1.0
        self.actor_state_queries: list[tuple[tuple[str, ...], torch.Tensor]] = []

    def get_actor_states(self, names: list[str], env_ids: torch.Tensor) -> torch.Tensor:
        self.actor_state_queries.append((tuple(names), env_ids.detach().clone()))
        # Match the simulator contract: object-major batches, each retaining
        # the caller's env order (including unsorted selections).
        return self.object_states[env_ids].permute(1, 0, 2).reshape(-1, 13)


class _FarTrackingSubsetSensorStub:
    def __init__(self, num_envs: int) -> None:
        self.ray_cast_body_poses_tensor = torch.full((num_envs, 3, 3), -7.0)
        self.ray_cast_body_quats_tensor = torch.zeros((num_envs, 3, 4))
        self.ray_cast_body_quats_tensor[..., 3] = 1.0
        self.primitive_body_poses_tensor = torch.full((num_envs, 2, 3), -8.0)
        self.primitive_body_quats_tensor = torch.zeros((num_envs, 2, 4))
        self.primitive_body_quats_tensor[..., 3] = 1.0
        self.camera_sensor_local_position = torch.zeros((num_envs, 1, 3))
        self.camera_sensor_local_orientation = torch.zeros((num_envs, 1, 4))
        self.camera_sensor_local_orientation[..., 3] = 1.0
        self.camera_sensor_data_frame_quat = self.camera_sensor_local_orientation.clone()
        self.camera_sensor_position = torch.full((num_envs, 1, 3), -9.0)
        self.camera_sensor_orientation = torch.zeros((num_envs, 1, 4))
        self.camera_sensor_orientation[..., 3] = 1.0
        self.depth = torch.full((num_envs, 1, 1, 1), 0.1)
        self.capture_calls: list[torch.Tensor | None] = []

    def capture(self, debug: bool = False, active_env_ids: torch.Tensor | None = None) -> torch.Tensor:
        del debug
        copied_ids = None if active_env_ids is None else active_env_ids.detach().clone()
        self.capture_calls.append(copied_ids)
        produced = (
            0.5
            + self.camera_sensor_position[:, 0, 0]
            + 0.1 * self.ray_cast_body_poses_tensor[:, 0, 0]
            + 0.01 * self.primitive_body_poses_tensor[:, :, 0].sum(dim=1)
        )
        if active_env_ids is None:
            self.depth[:, 0, 0, 0] = produced
        else:
            self.depth[active_env_ids, 0, 0, 0] = produced[active_env_ids]
        return self.depth


def _make_far_tracking_subset_manager(num_envs: int = 4) -> PerceptionManager:
    manager = object.__new__(PerceptionManager)
    manager.num_envs = num_envs
    manager.device = "cpu"
    manager.cfg = SimpleNamespace(camera_near=0.01, max_distance=1000.0, camera_pitch_deg=0.0)
    manager._camera_disable_offsets = False
    manager._camera_strict_warp = True
    manager._camera_body_index = 0
    manager._camera_body_offset_pos = torch.zeros(3)
    manager._camera_body_offset_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    manager._strict_camera_mount_rotation_deg = torch.zeros(3)
    manager._sensor_offset = torch.zeros(3)
    manager._debug_dump_dir = ""
    manager._debug_dump_done = False

    identity_quat = torch.zeros((num_envs, 1, 4))
    identity_quat[..., 3] = 1.0
    simulator = _FarTrackingSubsetSimulatorStub(num_envs)
    camera_offsets = torch.stack(
        (
            torch.arange(num_envs, dtype=torch.float32) * 0.02,
            torch.arange(num_envs, dtype=torch.float32) * -0.01,
            torch.arange(num_envs, dtype=torch.float32) * 0.005,
        ),
        dim=-1,
    )
    manager.env = SimpleNamespace(
        simulator=simulator,
        command_manager=None,
        randomization_manager=None,
        _perception_camera_offset_pos=camera_offsets,
        _perception_camera_offset_quat=identity_quat[:, 0].clone(),
        _perception_camera_offset_rpy=torch.zeros((num_envs, 3)),
    )

    sensor = _FarTrackingSubsetSensorStub(num_envs)
    manager._far_tracking_camera_sensor = sensor
    # These helpers remain part of the initialized far-tracking contract even
    # though fixed-link composition now uses the shared manager operations.
    manager._far_tracking_tf_apply = object()
    manager._far_tracking_quat_mul = object()
    manager._far_tracking_base_link_indices = torch.tensor([0])
    manager._far_tracking_robot_slot_indices = torch.tensor([0], dtype=torch.long)
    manager._far_tracking_robot_body_indices = torch.tensor([1], dtype=torch.long)
    manager._far_tracking_robot_body_offset_pos = torch.tensor([[0.01, -0.02, 0.03]])
    manager._far_tracking_robot_body_offset_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    manager._far_tracking_object_slot_indices = torch.tensor([1, 2], dtype=torch.long)
    manager._far_tracking_object_source_indices = torch.tensor([0, 1], dtype=torch.long)
    manager._far_tracking_primitive_source_indices = torch.tensor([0, 1], dtype=torch.long)
    manager._far_tracking_object_names = ["object_a", "object_b"]
    manager._far_tracking_object_active_env_ids = [
        torch.tensor([0, 2], dtype=torch.long),
        torch.tensor([1, 3], dtype=torch.long),
    ]
    manager._shared_camera_sensor_local_position = torch.zeros((num_envs, 1, 3))
    manager._shared_camera_sensor_local_orientation = identity_quat.clone()
    manager._shared_camera_sensor_data_frame_quat = identity_quat.clone()
    manager._initialize_far_tracking_object_slots()
    return manager


def _advance_far_tracking_subset_sources(manager: PerceptionManager) -> None:
    manager.env.simulator._rigid_body_pos.add_(0.25)
    manager.env.simulator.object_states[..., :3].add_(0.40)
    manager.env._perception_camera_offset_pos.add_(
        torch.tensor([0.30, -0.10, 0.05], dtype=torch.float32)
    )


def _far_tracking_sensor_dynamic_state(manager: PerceptionManager) -> dict[str, torch.Tensor]:
    sensor = manager._far_tracking_camera_sensor
    return {
        "robot_and_object_pos": sensor.ray_cast_body_poses_tensor.clone(),
        "robot_and_object_quat": sensor.ray_cast_body_quats_tensor.clone(),
        "primitive_pos": sensor.primitive_body_poses_tensor.clone(),
        "primitive_quat": sensor.primitive_body_quats_tensor.clone(),
        "camera_pos": sensor.camera_sensor_position.clone(),
        "camera_quat": sensor.camera_sensor_orientation.clone(),
        "depth": sensor.depth.clone(),
    }


def test_far_tracking_targeted_prep_matches_full_and_preserves_survivors_for_unsorted_multiobject_ids() -> None:
    targeted = _make_far_tracking_subset_manager()
    reference = _make_far_tracking_subset_manager()
    targeted._compute_far_tracking_camera_depth()
    reference._compute_far_tracking_camera_depth()
    before = _far_tracking_sensor_dynamic_state(targeted)

    _advance_far_tracking_subset_sources(targeted)
    _advance_far_tracking_subset_sources(reference)
    full_depth = reference._compute_far_tracking_camera_depth()
    selected = torch.tensor([3, 0], dtype=torch.long)
    survivor = torch.tensor([1, 2], dtype=torch.long)
    targeted_depth = targeted._compute_far_tracking_camera_depth(selected)

    assert torch.equal(targeted_depth, full_depth[selected])
    targeted_state = _far_tracking_sensor_dynamic_state(targeted)
    reference_state = _far_tracking_sensor_dynamic_state(reference)
    for key in targeted_state:
        assert torch.equal(targeted_state[key][selected], reference_state[key][selected]), key
        assert torch.equal(targeted_state[key][survivor], before[key][survivor]), key

    # Env 0 owns object slot 1 and env 3 owns slot 2.  Their other slots must
    # remain parked instead of being populated from the wrong object source.
    assert torch.equal(
        targeted._far_tracking_camera_sensor.ray_cast_body_poses_tensor[0, 2],
        before["robot_and_object_pos"][0, 2],
    )
    assert torch.equal(
        targeted._far_tracking_camera_sensor.ray_cast_body_poses_tensor[3, 1],
        before["robot_and_object_pos"][3, 1],
    )
    assert torch.equal(targeted.env.simulator.actor_state_queries[-1][1], selected)
    assert torch.equal(targeted._far_tracking_camera_sensor.capture_calls[-1], selected)


def test_far_tracking_targeted_single_env_uses_latest_camera_and_geometry_state() -> None:
    targeted = _make_far_tracking_subset_manager()
    reference = _make_far_tracking_subset_manager()
    targeted._compute_far_tracking_camera_depth()
    reference._compute_far_tracking_camera_depth()
    before = _far_tracking_sensor_dynamic_state(targeted)
    _advance_far_tracking_subset_sources(targeted)
    _advance_far_tracking_subset_sources(reference)

    full_depth = reference._compute_far_tracking_camera_depth()
    selected = torch.tensor([2], dtype=torch.long)
    camera_body_pose_rows: list[torch.Tensor | slice] = []
    original_camera_body_pose = targeted._get_camera_body_pose

    def record_camera_body_pose(idx: torch.Tensor | slice):
        camera_body_pose_rows.append(idx.detach().clone() if isinstance(idx, torch.Tensor) else idx)
        return original_camera_body_pose(idx)

    targeted._get_camera_body_pose = record_camera_body_pose
    targeted_depth = targeted._compute_far_tracking_camera_depth(selected)

    assert targeted_depth.shape == (1, 1, 1)
    assert torch.equal(targeted_depth, full_depth[selected])
    assert len(camera_body_pose_rows) == 1
    assert torch.equal(camera_body_pose_rows[0], selected)
    assert torch.equal(
        targeted._far_tracking_camera_sensor.camera_sensor_position[2, 0],
        reference._far_tracking_camera_sensor.camera_sensor_position[2, 0],
    )
    survivor = torch.tensor([0, 1, 3], dtype=torch.long)
    after = _far_tracking_sensor_dynamic_state(targeted)
    for key in after:
        assert torch.equal(after[key][survivor], before[key][survivor]), key
    assert torch.equal(targeted.env.simulator.actor_state_queries[-1][1], selected)


def test_far_tracking_empty_target_does_not_prepare_or_modify_any_rows() -> None:
    manager = _make_far_tracking_subset_manager()
    manager._compute_far_tracking_camera_depth()
    before = _far_tracking_sensor_dynamic_state(manager)
    query_count = len(manager.env.simulator.actor_state_queries)
    _advance_far_tracking_subset_sources(manager)

    empty = torch.empty(0, dtype=torch.long)
    depth = manager._compute_far_tracking_camera_depth(empty)

    assert depth.shape == (0, 1, 1)
    after = _far_tracking_sensor_dynamic_state(manager)
    for key in after:
        assert torch.equal(after[key], before[key]), key
    assert len(manager.env.simulator.actor_state_queries) == query_count
    assert torch.equal(manager._far_tracking_camera_sensor.capture_calls[-1], empty)
