"""Validated object-pose convention handling for trajectory optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ObjectPoseTransforms:
    positions: np.ndarray
    rotations: np.ndarray
    quaternions_wxyz: np.ndarray
    pose_layout: str


def resolve_object_pose_layout(
    object_poses: np.ndarray,
    pose_layout: str = "auto",
) -> str:
    """Resolve whether a seven-vector stores quaternion or position first."""

    object_poses = np.asarray(object_poses, dtype=np.float64)
    if object_poses.ndim != 2 or object_poses.shape[1] != 7:
        raise ValueError("object_poses must have shape (T, 7)")
    if pose_layout in {"quat_pos", "pos_quat"}:
        return pose_layout
    if pose_layout != "auto":
        raise ValueError("pose_layout must be 'auto', 'quat_pos', or 'pos_quat'")

    quat_pos_error = float(
        np.median(np.abs(np.linalg.norm(object_poses[:, :4], axis=1) - 1.0))
    )
    pos_quat_error = float(
        np.median(np.abs(np.linalg.norm(object_poses[:, 3:], axis=1) - 1.0))
    )
    best_error = min(quat_pos_error, pos_quat_error)
    error_gap = abs(quat_pos_error - pos_quat_error)
    if best_error > 1e-3 or error_gap < 1e-4:
        raise ValueError(
            "cannot infer object pose layout from quaternion norms; "
            "pass pose_layout explicitly"
        )
    return "quat_pos" if quat_pos_error < pos_quat_error else "pos_quat"


def decode_object_poses(
    object_poses: np.ndarray,
    *,
    quaternion_order: str,
    pose_layout: str = "auto",
) -> ObjectPoseTransforms:
    """Decode object poses into positions, rotations, and MuJoCo quaternions."""

    object_poses = np.asarray(object_poses, dtype=np.float64)
    resolved_layout = resolve_object_pose_layout(object_poses, pose_layout)
    if resolved_layout == "quat_pos":
        quaternions = object_poses[:, :4].copy()
        positions = object_poses[:, 4:].copy()
    else:
        positions = object_poses[:, :3].copy()
        quaternions = object_poses[:, 3:].copy()

    if quaternion_order == "xyzw":
        quaternions_xyzw = quaternions
        quaternions_wxyz = quaternions[:, [3, 0, 1, 2]]
    elif quaternion_order == "wxyz":
        quaternions_wxyz = quaternions
        quaternions_xyzw = quaternions[:, [1, 2, 3, 0]]
    else:
        raise ValueError("quaternion_order must be 'xyzw' or 'wxyz'")
    quaternion_norms = np.linalg.norm(quaternions_xyzw, axis=1)
    if np.any(quaternion_norms < 1e-12):
        raise ValueError("object_poses contain a zero-norm quaternion")
    quaternions_xyzw = quaternions_xyzw / quaternion_norms[:, None]
    quaternions_wxyz = quaternions_wxyz / quaternion_norms[:, None]
    rotations = Rotation.from_quat(quaternions_xyzw).as_matrix()
    arrays = (positions, rotations, quaternions_wxyz)
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("object_poses contain non-finite values")
    return ObjectPoseTransforms(
        positions=positions,
        rotations=rotations,
        quaternions_wxyz=quaternions_wxyz,
        pose_layout=resolved_layout,
    )
