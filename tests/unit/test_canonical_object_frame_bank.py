from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_builder():
    path = Path(__file__).resolve().parents[2] / "scripts" / "build_canonical_object_frame_bank.py"
    spec = importlib.util.spec_from_file_location("canonical_object_frame_builder_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_canonical_axes_recover_geometry_axes_and_right_handedness() -> None:
    builder = _load_builder()
    angle = np.deg2rad(31.0)
    old_from_geometry = np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    inertia_old = old_from_geometry @ np.diag([1.0, 2.0, 3.0]) @ old_from_geometry.T
    world_up_in_old = old_from_geometry[:, 2]

    rotation, _, symmetry, alignment, _ = builder._canonical_axes(
        inertia_old,
        np.repeat(world_up_in_old[None, :], 5, axis=0),
    )

    assert symmetry == "asymmetric_principal_axes"
    assert alignment < 1.0e-6
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-10)
    assert np.linalg.det(rotation) > 0.999999
    canonical_inertia = rotation.T @ inertia_old @ rotation
    np.testing.assert_allclose(canonical_inertia, np.diag(np.diag(canonical_inertia)), atol=1.0e-10)


def test_pose_velocity_transform_preserves_surface_world_state() -> None:
    builder = _load_builder()
    rotation_old_from_canonical = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotation_quat = builder._matrix_to_quat_wxyz(rotation_old_from_canonical)
    old_quat = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.9238795325, 0.0, 0.0, 0.3826834324]])
    old_pos = np.asarray([[1.0, 2.0, 3.0], [-0.5, 0.2, 1.3]], dtype=np.float32)
    old_lin = np.asarray([[0.1, 0.2, -0.1], [0.3, -0.2, 0.4]], dtype=np.float32)
    old_ang = np.asarray([[0.2, -0.3, 0.1], [-0.4, 0.1, 0.2]], dtype=np.float32)
    offset = np.asarray([0.15, -0.04, 0.08])

    new_pos, new_quat, new_lin, _ = builder._transform_pose_velocity(
        old_pos,
        old_quat,
        old_lin,
        old_ang,
        offset_old=offset,
        rotation_quat_wxyz=rotation_quat,
    )
    point_old = np.asarray([0.31, -0.07, 0.12])
    point_new = rotation_old_from_canonical.T @ (point_old - offset)
    old_rot = builder._quat_to_matrix_wxyz(old_quat)
    new_rot = builder._quat_to_matrix_wxyz(new_quat)
    old_world = old_pos + np.einsum("tij,j->ti", old_rot, point_old)
    new_world = new_pos + np.einsum("tij,j->ti", new_rot, point_new)
    np.testing.assert_allclose(new_world, old_world, atol=2.0e-7)
    old_velocity = old_lin + np.cross(old_ang, old_world - old_pos)
    new_velocity = new_lin + np.cross(old_ang, new_world - new_pos)
    np.testing.assert_allclose(new_velocity, old_velocity, atol=2.0e-7)
