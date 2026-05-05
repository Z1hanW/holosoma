"""Default camera manager configurations."""

import math
import os

from holosoma.config_types.camera import CameraManagerCfg, CameraPose, CameraProps, CameraTermCfg
from holosoma.config_values.loco.g1.camera import dual_depth_cameras

none = CameraManagerCfg()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(float(raw))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _crop_end_from_right(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = int(float(raw))
    return -value if value > 0 else None


def _env_latency_frame(name: str, default: int | tuple[int, int]) -> int | tuple[int, int]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(float(raw))


def _euler_xyz_deg_to_quat_wxyz(euler_deg: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (math.radians(value) for value in euler_deg)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        cy * cr * cp + sy * sr * sp,
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp,
    )


def _quat_mul_wxyz(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_wxyz_to_euler_xyz_deg(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def _compose_d435i_mount_with_pitch(pitch_deg: float) -> tuple[float, float, float]:
    mount_quat = _euler_xyz_deg_to_quat_wxyz((1.0, 27.0, 1.0))
    pitch_quat = _euler_xyz_deg_to_quat_wxyz((0.0, pitch_deg, 0.0))
    return _quat_wxyz_to_euler_xyz_deg(_quat_mul_wxyz(pitch_quat, mount_quat))


def _d435i_camera_rotation() -> tuple[float, float, float]:
    raw = os.environ.get("HOLOSOMA_D435I_CAMERA_ROTATION")
    if raw:
        values = tuple(float(part.strip()) for part in raw.split(","))
        if len(values) != 3:
            raise ValueError("HOLOSOMA_D435I_CAMERA_ROTATION must be 'roll,pitch,yaw'")
        return values

    pitch_raw = os.environ.get("PERCEPTION_CAMERA_PITCH_DEG")
    if pitch_raw is not None and pitch_raw != "":
        return _compose_d435i_mount_with_pitch(float(pitch_raw))
    return (1.0, 37.0, 1.0)


# D435i depth camera props for far-tracking distillation
d435i_depth_props = CameraProps(
    image_type="depth",
    width=_env_int("PERCEPTION_CAMERA_WIDTH", 848),
    height=_env_int("PERCEPTION_CAMERA_HEIGHT", 480),
    resized_width=87,
    resized_height=58,
    horizontal_fov=_env_float("PERCEPTION_CAMERA_HFOV_DEG", 89.5),
    vertical_fov=_env_float("PERCEPTION_CAMERA_VFOV_DEG", 89.5 * (60 / 106)),
    near_clip=_env_float("PERCEPTION_CAMERA_NEAR", 0.3),
    far_clip=_env_float("PERCEPTION_CAMERA_FAR", 3.0),
    image_show=True,
    frame_rate=_env_int("PERCEPTION_CAMERA_FPS", 50),
    depth_delay=0,
    crop_y_start=_env_int("PERCEPTION_CAMERA_WARP_CROP_TOP", 16),
    crop_y_end=_crop_end_from_right("PERCEPTION_CAMERA_WARP_CROP_BOTTOM", None),
    crop_x_start=_env_int("PERCEPTION_CAMERA_WARP_CROP_LEFT", 32),
    crop_x_end=_crop_end_from_right("PERCEPTION_CAMERA_WARP_CROP_RIGHT", -32),
    latency_frame=_env_latency_frame("PERCEPTION_CAMERA_WARP_LATENCY_FRAME", (7, 8)),
    buffer_len=_env_int("PERCEPTION_CAMERA_WARP_BUFFER_LEN", 9),
)

single_d435i_depth = CameraManagerCfg(terms={
    "d435i_depth": CameraTermCfg(
        func="holosoma.managers.camera.terms.depth:DepthCamera",
        params={
            "pose": CameraPose(
                camera_body_link="torso_link",
                camera_offset=(0.01, 0.01, 0.44),
                camera_rotation=_d435i_camera_rotation(),
            ),
            "props": d435i_depth_props,
        },
    ),
})

# ZED 2i depth camera props for far-tracking distillation (G1FlatZed2iConfig)
# Native: 1280x720, warp raycast: 240x135, policy resize: (58, 87)
# Training uses "resize": (58, 87) in distillation_env_cfg.py WarpObservationsCfg
zed2i_depth_props = CameraProps(
    image_type="depth",
    width=1280,
    height=720,
    resized_width=87,
    resized_height=58,
    horizontal_fov=101.41,
    vertical_fov=101.41 * (135 / 240),
    near_clip=0.1,
    far_clip=2.0,
    image_show=False,
    frame_rate=10,
)

single_zed2i_depth = CameraManagerCfg(terms={
    "zed2i_depth": CameraTermCfg(
        func="holosoma.managers.camera.terms.depth:DepthCamera",
        params={
            "pose": CameraPose(
                camera_body_link="torso_link",
                camera_offset=(0.1, 0.0, 0.1),
                camera_rotation=(0.0, 75.0, 0.0),     # warp convention, same as training G1FlatZed2iConfig
            ),
            "props": zed2i_depth_props,
        },
    ),
})

DEFAULTS = {
    "none": none,
    "dual_depth_cameras": dual_depth_cameras,
    "single_d435i_depth": single_d435i_depth,
    "single_zed2i_depth": single_zed2i_depth,
}

__all__ = ["dual_depth_cameras", "single_d435i_depth", "single_zed2i_depth"]
