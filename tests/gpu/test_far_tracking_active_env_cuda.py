from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import trimesh

from holosoma.third_party.ft_warp_sensors.camera_sensor import CameraSensor


_RUN_GPU_WARP_TEST = os.environ.get("HOLOSOMA_RUN_GPU_WARP_TEST", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _camera_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_sensors=1,
        base_link_frame={"camera": "pelvis"},
        width=12,
        height=8,
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
        primitive_bodies=[],
        add_offpath_obstacle=False,
        offpath_obstacle_bodies={},
        offpath_obstacle_meshes_root="",
        dynamic_meshes=False,
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


@pytest.mark.skipif(
    not _RUN_GPU_WARP_TEST or not torch.cuda.is_available(),
    reason="explicit isolated-GPU Warp parity test",
)
def test_targeted_graph_capture_matches_full_capture_on_nondefault_torch_stream() -> None:
    device = "cuda:0"
    sensor = CameraSensor(5, _camera_config(), _plane_at_z(), device=device)
    stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream):
        sensor.camera_sensor_position.zero_()
        sensor.camera_sensor_orientation.zero_()
        sensor.camera_sensor_orientation[..., 3] = 1.0

        # A targeted capture uses a compact direct launch and deliberately does
        # not instantiate the fixed full-batch graph. Inactive rows retain
        # their initialized pixels while selected rows execute normally.
        first_ids = torch.tensor([3, 1, 3], device=device)
        first_targeted = sensor.capture(active_env_ids=first_ids).clone()
        assert sensor.graph is None
        first_full = sensor.capture(active_env_ids=None).clone()
        assert sensor.graph is not None

        # Move every camera without synchronizing the non-default Torch stream.
        # capture() must launch Warp on this same stream and see the new pose.
        sensor.camera_sensor_position[..., 2] = 1.0
        second_ids = torch.tensor([4, 0], device=device)
        second_targeted = sensor.capture(active_env_ids=second_ids).clone()
        second_full = sensor.capture(active_env_ids=None).clone()

        unchanged_after_empty = sensor.capture(
            active_env_ids=torch.empty(0, device=device, dtype=torch.long)
        ).clone()

    stream.synchronize()

    assert torch.equal(first_targeted[first_ids.unique()], first_full[first_ids.unique()])
    inactive_first = torch.tensor([0, 2, 4], device=device)
    assert torch.equal(first_targeted[inactive_first], torch.zeros_like(first_targeted[inactive_first]))

    assert torch.equal(second_targeted[second_ids], second_full[second_ids])
    inactive_second = torch.tensor([1, 2, 3], device=device)
    assert torch.equal(second_targeted[inactive_second], first_full[inactive_second])
    assert torch.equal(unchanged_after_empty, second_full)

    # The pose change must actually alter depth, otherwise stream parity above
    # could pass while Warp read stale camera tensors.
    assert not torch.equal(first_full[second_ids], second_full[second_ids])
