from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import trimesh
import warp as wp

from holosoma.third_party.ft_warp_sensors import camera_sensor as camera_sensor_module
from holosoma.third_party.ft_warp_sensors.camera_kernels_warp import DepthCameraWarpKernels
from holosoma.third_party.ft_warp_sensors.camera_sensor import CameraSensor


def _mapping_only_sensor(num_envs: int = 6) -> CameraSensor:
    wp.init()
    sensor = object.__new__(CameraSensor)
    sensor.num_envs = num_envs
    sensor.device = "cpu"
    sensor.full_capture_env_ids_tensor = torch.arange(num_envs, dtype=torch.int32)
    sensor.full_capture_env_ids = wp.from_torch(
        sensor.full_capture_env_ids_tensor, dtype=wp.int32
    )
    sensor._capture_env_ids_tensor = sensor.full_capture_env_ids_tensor
    sensor._capture_env_ids = sensor.full_capture_env_ids
    return sensor


def test_capture_mapping_defaults_to_identity() -> None:
    sensor = _mapping_only_sensor()
    sensor.set_capture_env_ids(torch.tensor([5, 2]))

    count = sensor.set_capture_env_ids(None)

    assert count == 6
    assert sensor._capture_env_ids_tensor is sensor.full_capture_env_ids_tensor
    assert torch.equal(sensor._capture_env_ids_tensor, torch.arange(6, dtype=torch.int32))


def test_compact_capture_mapping_preserves_order_and_duplicates() -> None:
    sensor = _mapping_only_sensor()

    count = sensor.set_capture_env_ids(torch.tensor([4, 1, 4]))

    assert count == 3
    assert torch.equal(sensor._capture_env_ids_tensor, torch.tensor([4, 1, 4], dtype=torch.int32))


def test_compact_capture_mapping_supports_empty_and_singleton_selections() -> None:
    sensor = _mapping_only_sensor()

    assert sensor.set_capture_env_ids(torch.empty(0, dtype=torch.long)) == 0
    assert sensor._capture_env_ids_tensor.shape == (0,)

    assert sensor.set_capture_env_ids(torch.tensor([3])) == 1
    assert torch.equal(sensor._capture_env_ids_tensor, torch.tensor([3], dtype=torch.int32))


def test_compact_capture_mapping_rejects_out_of_range_ids_before_warp_launch() -> None:
    sensor = _mapping_only_sensor()

    with pytest.raises((IndexError, RuntimeError)):
        sensor.set_capture_env_ids(torch.tensor([6]))


def test_cuda_sensor_setup_establishes_warp_to_torch_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensor = object.__new__(CameraSensor)
    sensor.device = "cuda:3"
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        camera_sensor_module.wp,
        "synchronize_device",
        lambda device: calls.append(("warp", device)),
    )
    monkeypatch.setattr(
        camera_sensor_module.torch.cuda,
        "synchronize",
        lambda *, device: calls.append(("torch", device)),
    )

    sensor._synchronize_cuda_initialization()

    assert calls == [("warp", "cuda:3"), ("torch", torch.device("cuda:3"))]


@pytest.mark.parametrize("dynamic", [False, True])
def test_depth_launch_grid_uses_compact_count_and_global_mapping(
    monkeypatch: pytest.MonkeyPatch,
    dynamic: bool,
) -> None:
    sensor = object.__new__(CameraSensor)
    sensor.is_dyna_mesh = dynamic
    sensor.num_sensors = 2
    sensor.width = 87
    sensor.height = 58
    sensor.device = "cpu"
    sensor.terrain_mesh_id = 1
    sensor.camera_position_array = object()
    sensor.camera_orientation_array = object()
    sensor.K_inv = object()
    sensor.far_plane = 5.0
    sensor.pixels = object()
    sensor.c_x = 43
    sensor.c_y = 29
    sensor.calculate_depth = True
    sensor.robot_mesh_ids = object()
    sensor.primitive_body_active = object()
    sensor.primitive_body_half_extents = object()
    sensor.ray_cast_body_poses = object()
    sensor.ray_cast_body_quats = object()
    sensor.primitive_body_poses = object()
    sensor.primitive_body_quats = object()
    sensor.num_robot_bodies = 3
    sensor.primitive_bodies = ["box", "crate"]
    mapping = object()
    launch: dict[str, object] = {}

    monkeypatch.setattr(camera_sensor_module.wp, "launch", lambda **kwargs: launch.update(kwargs))
    sensor._launch_depth_range(mapping, 3)

    expected_kernel = (
        DepthCameraWarpKernels.draw_optimized_kernel_depth_range_dynamic
        if dynamic
        else DepthCameraWarpKernels.draw_optimized_kernel_depth_range
    )
    assert launch["kernel"] is expected_kernel
    assert launch["dim"] == (3, 2, 87, 58)
    assert launch["inputs"][1] is mapping


def test_pointcloud_launch_grid_uses_compact_count_and_global_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensor = object.__new__(CameraSensor)
    sensor.num_sensors = 2
    sensor.width = 87
    sensor.height = 58
    sensor.device = "cpu"
    sensor.terrain_mesh_id = 1
    sensor.camera_position_array = object()
    sensor.camera_orientation_array = object()
    sensor.K_inv = object()
    sensor.far_plane = 5.0
    sensor.pixels = object()
    sensor.c_x = 43
    sensor.c_y = 29
    sensor.pointcloud_in_world_frame = False
    mapping = object()
    launch: dict[str, object] = {}

    monkeypatch.setattr(camera_sensor_module.wp, "launch", lambda **kwargs: launch.update(kwargs))
    sensor._launch_pointcloud(mapping, 1)

    assert launch["kernel"] is DepthCameraWarpKernels.draw_optimized_kernel_pointcloud
    assert launch["dim"] == (1, 2, 87, 58)
    assert launch["inputs"][1] is mapping


def _camera_config(*, dynamic: bool) -> SimpleNamespace:
    return SimpleNamespace(
        num_sensors=1,
        base_link_frame={"camera": "pelvis"},
        width=6,
        height=4,
        horizontal_fov_deg=60.0,
        max_range=20.0,
        calculate_depth=True,
        offset_rot_base=(0.0, 0.0, 0.0),
        offset={"camera": {"offset_pos": (0.0, 0.0, 0.0), "offset_rot": (0.0, 0.0, 0.0)}},
        randomize_placement=False,
        return_pointcloud=False,
        pointcloud_in_world_frame=False,
        segmentation_camera=False,
        ray_cast_bodies={},
        primitive_bodies=["box", "crate"] if dynamic else [],
        add_offpath_obstacle=False,
        offpath_obstacle_bodies={},
        offpath_obstacle_meshes_root="",
        dynamic_meshes=dynamic,
        asset_meshes_root="",
    )


def _plane_at_z(z: float = 5.0) -> trimesh.Trimesh:
    vertices = np.asarray(
        [
            [-100.0, -100.0, z],
            [100.0, -100.0, z],
            [100.0, 100.0, z],
            [-100.0, 100.0, z],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


@pytest.mark.parametrize("dynamic", [False, True])
def test_compact_cpu_launch_updates_only_global_target_rows(dynamic: bool) -> None:
    """Exercise both terrain-only and multi-primitive mapping kernels on CPU."""
    sensor = CameraSensor(4, _camera_config(dynamic=dynamic), _plane_at_z(), device="cpu")
    sensor.camera_sensor_position.zero_()
    sensor.camera_sensor_orientation.zero_()
    sensor.camera_sensor_orientation[..., 3] = 1.0

    if dynamic:
        sensor.primitive_body_active_tensor.fill_(1)
        sensor.primitive_body_half_extents_tensor.fill_(0.25)
        sensor.primitive_body_poses_tensor[..., 2] = 3.0
        sensor.primitive_body_poses_tensor[:, 1, 2] = 4.0

    # CPU graph capture is intentionally bypassed; this still exercises the
    # exact full and compact Warp kernels used by the CUDA graph/direct paths.
    full = sensor.capture(debug=True).clone()
    sensor.depth_tensors.fill_(-7.0)

    targeted = sensor.capture(active_env_ids=torch.tensor([3, 1])).clone()
    assert torch.equal(targeted[[3, 1]], full[[3, 1]])
    assert torch.equal(targeted[[0, 2]], torch.full_like(targeted[[0, 2]], -7.0))

    singleton = sensor.capture(active_env_ids=torch.tensor([2])).clone()
    assert torch.equal(singleton[2], full[2])
    assert torch.equal(singleton[0], torch.full_like(singleton[0], -7.0))
    assert torch.equal(singleton[[1, 3]], targeted[[1, 3]])

    before_empty = singleton.clone()
    after_empty = sensor.capture(active_env_ids=torch.empty(0, dtype=torch.long)).clone()
    assert torch.equal(after_empty, before_empty)
