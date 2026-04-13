from __future__ import annotations

from pathlib import Path

import numpy as np
from holosoma.utils.object_geometry import (
    get_largebox_best_iou_primitive_center_offset,
    get_largebox_best_iou_primitive_extents,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional at import time for lightweight tooling
    torch = None

_OMOMO_LARGEBOX_TARGET_RESIDUAL_RPY_DEG = (180.0, 0.0, 0.0)
_OMOMO_LARGEBOX_PRIMITIVE_LOCAL_ALIGNMENT_RPY_DEG = (0.0, 0.0, -63.24443673940713)


def _quat_mul_wxyz_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def _quat_inverse_wxyz_np(q: np.ndarray) -> np.ndarray:
    inv = np.asarray(q, dtype=np.float64).copy()
    inv[..., 1:] *= -1.0
    return inv


def _quat_rotate_vec_wxyz_np(quat_wxyz: np.ndarray, vec_xyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    vec = np.asarray(vec_xyz, dtype=np.float64)
    q_vec = quat[..., 1:]
    uv = np.cross(q_vec, vec)
    uuv = np.cross(q_vec, uv)
    return vec + 2.0 * (quat[..., :1] * uv + uuv)


def _quat_from_rpy_deg_wxyz(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll = np.deg2rad(float(roll_deg))
    pitch = np.deg2rad(float(pitch_deg))
    yaw = np.deg2rad(float(yaw_deg))
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


OMOMO_LARGEBOX_TARGET_RESIDUAL_WXYZ = _quat_from_rpy_deg_wxyz(
    *_OMOMO_LARGEBOX_TARGET_RESIDUAL_RPY_DEG
).astype(np.float32)
OMOMO_LARGEBOX_PRIMITIVE_LOCAL_ALIGNMENT_WXYZ = _quat_from_rpy_deg_wxyz(
    *_OMOMO_LARGEBOX_PRIMITIVE_LOCAL_ALIGNMENT_RPY_DEG
).astype(np.float32)
OMOMO_LARGEBOX_CANONICAL_LOCAL_CORRECTION_WXYZ = np.asarray(
    [
        0.9968672571552186,
        -0.05573667084973842,
        -0.05602919466692427,
        0.0031344025579125466,
    ],
    dtype=np.float32,
)
OMOMO_LARGEBOX_PRIMITIVE_CENTER_OFFSET_XYZ = np.asarray(
    get_largebox_best_iou_primitive_center_offset(),
    dtype=np.float32,
)
OMOMO_LARGEBOX_PRIMITIVE_EXTENTS_XYZ = np.asarray(
    get_largebox_best_iou_primitive_extents(),
    dtype=np.float32,
)


def _box_corners_from_extents_xyz(extents_xyz: np.ndarray) -> np.ndarray:
    half = 0.5 * np.asarray(extents_xyz, dtype=np.float64).reshape(3)
    return np.asarray(
        [
            [sx, sy, sz]
            for sx in (-half[0], half[0])
            for sy in (-half[1], half[1])
            for sz in (-half[2], half[2])
        ],
        dtype=np.float64,
    )


def is_omomo_largebox_clip(clip_id: str, object_name: str = "", object_urdf_path: str = "") -> bool:
    clip_key = str(clip_id or "").strip().lower()
    object_key = str(object_name or "").strip().lower()
    urdf_stem = Path(str(object_urdf_path or "").strip()).stem.lower()

    if clip_key.startswith("omomo__"):
        clip_key = clip_key[len("omomo__") :]

    is_omomo = clip_key.startswith("sub")
    is_largebox = (
        "largebox" in clip_key
        or object_key in {"largebox", "objects_largebox"}
        or urdf_stem in {"largebox", "objects_largebox"}
    )
    return bool(is_omomo and is_largebox)


def uses_omomo_largebox_primitive_semantics(object_name: str = "", object_urdf_path: str = "") -> bool:
    object_key = str(object_name or "").strip().lower()
    urdf_stem = Path(str(object_urdf_path or "").strip()).stem.lower()
    return bool(object_key in {"largebox", "objects_largebox"} or urdf_stem in {"largebox", "objects_largebox"})


def yaw_from_quat_wxyz_np(quat_wxyz: np.ndarray) -> float:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def derive_local_zup_correction_wxyz_from_first_frame(
    first_quat_wxyz: np.ndarray,
    *,
    target_residual_wxyz: np.ndarray | None = None,
) -> np.ndarray:
    target = (
        np.asarray(target_residual_wxyz, dtype=np.float64).reshape(4)
        if target_residual_wxyz is not None
        else OMOMO_LARGEBOX_TARGET_RESIDUAL_WXYZ.astype(np.float64)
    )
    yaw = yaw_from_quat_wxyz_np(first_quat_wxyz)
    yaw_only = _quat_from_rpy_deg_wxyz(0.0, 0.0, np.rad2deg(yaw))
    residual = _quat_mul_wxyz_np(_quat_inverse_wxyz_np(yaw_only), np.asarray(first_quat_wxyz, dtype=np.float64))
    correction = _quat_mul_wxyz_np(_quat_inverse_wxyz_np(residual), target)
    correction /= np.linalg.norm(correction)
    if correction[0] < 0.0:
        correction *= -1.0
    return correction.astype(np.float32)


def apply_local_quat_correction_wxyz_np(quat_wxyz: np.ndarray, correction_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float32)
    correction = np.asarray(correction_wxyz, dtype=np.float32)
    if quat.ndim == 1:
        quat = quat.reshape(1, 4)
        squeeze = True
    else:
        squeeze = False
    correction_batch = np.broadcast_to(correction.reshape(1, 4), quat.shape)
    corrected = _quat_mul_wxyz_np(quat.astype(np.float64), correction_batch.astype(np.float64)).astype(np.float32)
    norms = np.linalg.norm(corrected, axis=-1, keepdims=True)
    corrected = corrected / np.clip(norms, 1.0e-8, None)
    return corrected[0] if squeeze else corrected


def get_omomo_largebox_primitive_center_offset_xyz_np() -> np.ndarray:
    return OMOMO_LARGEBOX_PRIMITIVE_CENTER_OFFSET_XYZ.copy()


def get_omomo_largebox_primitive_extents_xyz_np() -> np.ndarray:
    return OMOMO_LARGEBOX_PRIMITIVE_EXTENTS_XYZ.copy()


def get_omomo_largebox_canonical_local_correction_wxyz_np() -> np.ndarray:
    return OMOMO_LARGEBOX_CANONICAL_LOCAL_CORRECTION_WXYZ.copy()


def get_omomo_largebox_primitive_local_alignment_wxyz_np() -> np.ndarray:
    return OMOMO_LARGEBOX_PRIMITIVE_LOCAL_ALIGNMENT_WXYZ.copy()


def get_omomo_largebox_primitive_fit_local_correction_wxyz_np() -> np.ndarray:
    return _quat_mul_wxyz_np(
        OMOMO_LARGEBOX_CANONICAL_LOCAL_CORRECTION_WXYZ.astype(np.float64),
        OMOMO_LARGEBOX_PRIMITIVE_LOCAL_ALIGNMENT_WXYZ.astype(np.float64),
    ).astype(np.float32)


def compute_box_ground_contact_translation_xyz_wxyz_np(
    pos_w: np.ndarray,
    quat_wxyz: np.ndarray,
    extents_xyz: np.ndarray,
) -> np.ndarray:
    pos = np.asarray(pos_w, dtype=np.float32)
    quat = np.asarray(quat_wxyz, dtype=np.float32)
    if pos.ndim == 1:
        pos = pos.reshape(1, 3)
    if quat.ndim == 1:
        quat = quat.reshape(1, 4)
    if pos.shape[0] == 0 or quat.shape[0] == 0:
        return np.zeros(3, dtype=np.float32)

    first_pos = pos[0].astype(np.float64)
    first_quat = quat[0].astype(np.float64)
    corners_local = _box_corners_from_extents_xyz(extents_xyz)
    quat_batch = np.broadcast_to(first_quat.reshape(1, 4), (corners_local.shape[0], 4))
    corners_world = first_pos.reshape(1, 3) + _quat_rotate_vec_wxyz_np(quat_batch, corners_local)
    lowest_z = float(np.min(corners_world[:, 2]))
    return np.asarray([0.0, 0.0, -lowest_z], dtype=np.float32)


def apply_local_position_offset_wxyz_np(
    pos_w: np.ndarray,
    quat_wxyz: np.ndarray,
    local_offset_xyz: np.ndarray,
) -> np.ndarray:
    pos = np.asarray(pos_w, dtype=np.float32)
    quat = np.asarray(quat_wxyz, dtype=np.float32)
    offset = np.asarray(local_offset_xyz, dtype=np.float32)
    if pos.ndim == 1:
        pos = pos.reshape(1, 3)
        squeeze = True
    else:
        squeeze = False
    rotated_offset = _quat_rotate_vec_wxyz_np(quat.astype(np.float64), offset.astype(np.float64)).astype(np.float32)
    corrected = pos + rotated_offset
    return corrected[0] if squeeze else corrected


def apply_local_pose_correction_wxyz_np(
    pos_w: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    local_correction_wxyz: np.ndarray | None = None,
    local_offset_xyz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    quat = np.asarray(quat_wxyz, dtype=np.float32)
    corrected_quat = quat
    if local_correction_wxyz is not None:
        corrected_quat = apply_local_quat_correction_wxyz_np(quat, np.asarray(local_correction_wxyz, dtype=np.float32))

    pos = np.asarray(pos_w, dtype=np.float32)
    corrected_pos = pos
    if local_offset_xyz is not None:
        corrected_pos = apply_local_position_offset_wxyz_np(
            pos,
            corrected_quat,
            np.asarray(local_offset_xyz, dtype=np.float32),
        )
    return corrected_pos, corrected_quat


def remove_local_pose_correction_wxyz_np(
    pos_w: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    local_correction_wxyz: np.ndarray | None = None,
    local_offset_xyz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    quat = np.asarray(quat_wxyz, dtype=np.float32)
    pos = np.asarray(pos_w, dtype=np.float32)

    corrected_quat = quat
    if local_correction_wxyz is not None:
        correction = np.asarray(local_correction_wxyz, dtype=np.float32)
        corrected_quat = _quat_mul_wxyz_np(
            quat.astype(np.float64),
            _quat_inverse_wxyz_np(correction.astype(np.float64)),
        ).astype(np.float32)
        norms = np.linalg.norm(corrected_quat, axis=-1, keepdims=True) if corrected_quat.ndim > 1 else np.linalg.norm(corrected_quat)
        corrected_quat = corrected_quat / np.clip(norms, 1.0e-8, None)

    corrected_pos = pos
    if local_offset_xyz is not None:
        offset = np.asarray(local_offset_xyz, dtype=np.float32)
        rotated_offset = _quat_rotate_vec_wxyz_np(quat.astype(np.float64), offset.astype(np.float64)).astype(np.float32)
        corrected_pos = pos - rotated_offset
    return corrected_pos, corrected_quat


def apply_omomo_largebox_primitive_local_alignment_wxyz_np(quat_wxyz: np.ndarray) -> np.ndarray:
    return apply_local_quat_correction_wxyz_np(
        quat_wxyz,
        OMOMO_LARGEBOX_PRIMITIVE_LOCAL_ALIGNMENT_WXYZ,
    )


def apply_local_quat_correction_xyzw_torch(quat_xyzw: torch.Tensor, correction_xyzw: torch.Tensor) -> torch.Tensor:
    if torch is None:
        raise ModuleNotFoundError("torch is required for apply_local_quat_correction_xyzw_torch")
    if quat_xyzw.ndim == 1:
        quat_xyzw = quat_xyzw.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    x1, y1, z1, w1 = quat_xyzw.unbind(dim=-1)
    x2, y2, z2, w2 = correction_xyzw.expand_as(quat_xyzw).unbind(dim=-1)
    corrected = torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )
    corrected = torch.nn.functional.normalize(corrected, dim=-1)
    return corrected[0] if squeeze else corrected


def apply_omomo_largebox_zup_correction_wxyz_np(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float32)
    if quat.ndim == 1:
        first = quat
    else:
        first = quat[0]
    correction = derive_local_zup_correction_wxyz_from_first_frame(first)
    return apply_local_quat_correction_wxyz_np(quat, correction)


def apply_omomo_largebox_primitive_frame_correction_wxyz_np(quat_wxyz: np.ndarray) -> np.ndarray:
    corrected = apply_omomo_largebox_zup_correction_wxyz_np(quat_wxyz)
    return apply_omomo_largebox_primitive_local_alignment_wxyz_np(corrected)


def apply_omomo_largebox_center_offset_wxyz_np(pos_w: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    return apply_local_position_offset_wxyz_np(
        pos_w,
        quat_wxyz,
        OMOMO_LARGEBOX_PRIMITIVE_CENTER_OFFSET_XYZ,
    )


def apply_omomo_largebox_ground_contact_wxyz_np(pos_w: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos_w, dtype=np.float32)
    translation = compute_box_ground_contact_translation_xyz_wxyz_np(
        pos,
        quat_wxyz,
        OMOMO_LARGEBOX_PRIMITIVE_EXTENTS_XYZ,
    )
    return pos + translation.reshape(1, 3) if pos.ndim > 1 else pos + translation


def apply_omomo_largebox_semantic_pose_correction_wxyz_np(
    pos_w: np.ndarray,
    quat_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return apply_local_pose_correction_wxyz_np(
        pos_w,
        quat_wxyz,
        local_correction_wxyz=OMOMO_LARGEBOX_CANONICAL_LOCAL_CORRECTION_WXYZ,
        local_offset_xyz=OMOMO_LARGEBOX_PRIMITIVE_CENTER_OFFSET_XYZ,
    )


def remove_omomo_largebox_semantic_pose_correction_wxyz_np(
    pos_w: np.ndarray,
    quat_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return remove_local_pose_correction_wxyz_np(
        pos_w,
        quat_wxyz,
        local_correction_wxyz=OMOMO_LARGEBOX_CANONICAL_LOCAL_CORRECTION_WXYZ,
        local_offset_xyz=OMOMO_LARGEBOX_PRIMITIVE_CENTER_OFFSET_XYZ,
    )
