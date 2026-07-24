from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.simulator.isaacsim.perception_camera import (
    IsaacSimDepthCamera,
    IsaacSimDepthSensorCamera,
)
from holosoma.utils.rotations import quat_apply


class _PoseView:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def set_world_poses(self, position, orientation, env_ids) -> None:
        self.calls.append((position.clone(), orientation.clone(), env_ids.clone()))


def test_isaac_rendered_camera_converts_manager_optical_pose_to_usd_axes() -> None:
    camera = object.__new__(IsaacSimDepthCamera)
    primary_view = _PoseView()
    rgb_view = _PoseView()
    requested_ids: list[torch.Tensor] = []

    def pose_provider(env_ids):
        requested_ids.append(env_ids.clone())
        return (
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        )

    camera._view = primary_view
    camera._rgb_view = rgb_view
    camera._pose_provider = pose_provider
    camera._env_id = 2
    camera._device = "cpu"
    camera._env = SimpleNamespace()

    camera._update_pose()

    assert len(requested_ids) == 1
    assert torch.equal(requested_ids[0], torch.tensor([2]))
    expected_position = torch.tensor([[1.0, 2.0, 3.0]])
    # Optical identity (+Z forward/+Y down) becomes USD Rx(pi), whose local
    # -Z points world +Z and local +Y points world -Y.
    expected_wxyz = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    for view in (primary_view, rgb_view):
        assert len(view.calls) == 1
        position, orientation, env_ids = view.calls[0]
        assert torch.equal(position, expected_position)
        assert torch.equal(orientation, expected_wxyz)
        # XFormPrim wraps one exact camera prim; indices are view-local rather
        # than simulator environment ids.
        assert torch.equal(env_ids, torch.tensor([0], dtype=torch.int32))
    usd_quat_xyzw = expected_wxyz[:, [1, 2, 3, 0]]
    assert torch.allclose(
        quat_apply(usd_quat_xyzw, torch.tensor([[0.0, 0.0, -1.0]]), w_last=True),
        torch.tensor([[0.0, 0.0, 1.0]]),
        atol=1.0e-6,
    )
    assert torch.allclose(
        quat_apply(usd_quat_xyzw, torch.tensor([[0.0, 1.0, 0.0]]), w_last=True),
        torch.tensor([[0.0, -1.0, 0.0]]),
        atol=1.0e-6,
    )


def test_isaac_rendered_camera_axis_conversion_is_postmultiplied_in_optical_frame() -> None:
    camera = object.__new__(IsaacSimDepthCamera)
    primary_view = _PoseView()
    sin_cos_45 = 2.0**-0.5
    provider_quat = torch.tensor([[0.0, 0.0, sin_cos_45, sin_cos_45]])
    camera._view = primary_view
    camera._rgb_view = None
    camera._pose_provider = lambda _ids: (torch.zeros((1, 3)), provider_quat.clone())
    camera._env_id = 0
    camera._device = "cpu"
    camera._env = SimpleNamespace()

    camera._update_pose()

    captured_wxyz = primary_view.calls[0][1]
    captured_xyzw = captured_wxyz[:, [1, 2, 3, 0]]
    # USD local -Z/+Y/+X must map to optical +Z/-Y/+X after the
    # provider's non-commuting world Rz(90) rotation.  A left-multiplied
    # Rx(pi) fails these comparisons.
    for usd_axis, optical_axis in (
        ([0.0, 0.0, -1.0], [0.0, 0.0, 1.0]),
        ([0.0, 1.0, 0.0], [0.0, -1.0, 0.0]),
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
    ):
        actual_world = quat_apply(
            captured_xyzw,
            torch.tensor([usd_axis]),
            w_last=True,
        )
        expected_world = quat_apply(
            provider_quat,
            torch.tensor([optical_axis]),
            w_last=True,
        )
        assert torch.allclose(actual_world, expected_world, atol=1.0e-6)


def test_isaac_rendered_camera_rejects_invalid_provider_pose() -> None:
    camera = object.__new__(IsaacSimDepthCamera)
    camera._view = _PoseView()
    camera._rgb_view = None
    camera._pose_provider = lambda _ids: (
        torch.tensor([[float("nan"), 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
    )
    camera._env_id = 0
    camera._device = "cpu"
    camera._env = SimpleNamespace()

    with pytest.raises(RuntimeError, match="non-finite"):
        camera._update_pose()


def test_isaac_rendered_camera_rejects_nonunit_provider_quaternion() -> None:
    camera = object.__new__(IsaacSimDepthCamera)
    camera._view = _PoseView()
    camera._rgb_view = None
    camera._pose_provider = lambda _ids: (
        torch.zeros((1, 3)),
        torch.tensor([[0.0, 0.0, 0.0, 2.0]]),
    )
    camera._env_id = 0
    camera._device = "cpu"
    camera._env = SimpleNamespace()

    with pytest.raises(RuntimeError, match="non-unit quaternion"):
        camera._update_pose()


def test_depth_sensor_asset_rejects_uncompensated_optical_pose_provider() -> None:
    camera = object.__new__(IsaacSimDepthSensorCamera)
    camera._pose_provider = lambda _ids: (
        torch.zeros((1, 3)),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
    )

    with pytest.raises(RuntimeError, match="child-local transform"):
        camera.setup()
