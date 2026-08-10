#!/usr/bin/env python3
"""Publish static MuJoCo ground-truth depth for the real-debug Viser view."""

from __future__ import annotations

import argparse
import math
import signal
import time
from multiprocessing import shared_memory
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = (
    REPO_ROOT
    / "src"
    / "holosoma"
    / "holosoma"
    / "data"
    / "robots"
    / "g1"
    / "scenes"
    / "scene_g1_29dof_wbt_plane.xml"
)
DEFAULT_MOTION = REPO_ROOT / "data_demo" / "box_75.npz"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--motion-frame", type=int, default=0)
    parser.add_argument("--shm-name", default="sim_gt_depth_raw_shm")
    parser.add_argument("--width", type=int, default=106)
    parser.add_argument("--height", type=int, default=60)
    parser.add_argument("--horizontal-fov-deg", type=float, default=89.5)
    parser.add_argument("--vertical-fov-deg", type=float, default=58.6)
    parser.add_argument("--near", type=float, default=0.3)
    parser.add_argument("--far", type=float, default=3.0)
    parser.add_argument("--rate-hz", type=float, default=30.0)
    return parser.parse_args(argv)


def _euler_xyz_to_quat_wxyz(euler_rad: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = euler_rad
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.asarray(
        [
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ],
        dtype=np.float64,
    )


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def d435i_urdf_mujoco_quaternion() -> np.ndarray:
    """Use the same Warp/Isaac-to-MuJoCo camera conversion as SceneManager."""
    user = _euler_xyz_to_quat_wxyz(np.deg2rad([0.0, 47.6, 0.0]))
    base = _euler_xyz_to_quat_wxyz(np.deg2rad([-90.0, 0.0, -90.0]))
    flip = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    quaternion = _quat_mul_wxyz(_quat_mul_wxyz(user, base), flip)
    return quaternion / np.linalg.norm(quaternion)


def _motion_frame(path: Path, frame_index: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as motion:
        frame_count = int(motion["joint_pos"].shape[0])
        frame_index = min(max(frame_index, 0), frame_count - 1)
        root_qpos = np.asarray(motion["joint_pos"][frame_index, :7], dtype=np.float64).copy()
        root_qpos[2] = 0.76
        w, x, y, z = root_qpos[3:7]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        root_qpos[3:7] = [math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)]
        return {
            "root_qpos": root_qpos,
            "object_pos": np.asarray(motion["object_pos_w"][frame_index], dtype=np.float64),
            "object_quat": np.asarray(motion["object_quat_w"][frame_index], dtype=np.float64),
            "object_size": np.asarray(motion["object_size"], dtype=np.float64),
        }


def build_scene(scene_path: Path, motion_path: Path, motion_frame: int, vertical_fov_deg: float):
    """Build a static G1 zero-joint scene with the selected motion object."""
    state = _motion_frame(motion_path, motion_frame)
    spec = mujoco.MjSpec.from_file(str(scene_path))

    existing_camera = spec.camera("cam_d435i_depth")
    if existing_camera is not None:
        spec.delete(existing_camera)
    torso = spec.body("torso_link")
    if torso is None:
        raise RuntimeError("torso_link is missing from the MuJoCo G1 scene")
    camera_name = "sim_gt_d435i_urdf"
    torso.add_camera(
        name=camera_name,
        pos=[0.0576235, 0.01753, 0.41987],
        quat=d435i_urdf_mujoco_quaternion().tolist(),
        fovy=vertical_fov_deg,
    )

    object_body = spec.worldbody.add_body(
        name="sim_gt_box",
        pos=state["object_pos"].tolist(),
        quat=state["object_quat"].tolist(),
    )
    object_body.add_geom(
        name="sim_gt_box_visual",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(state["object_size"] * 0.5).tolist(),
        rgba=[0.7, 0.8, 0.9, 1.0],
        contype=0,
        conaffinity=0,
    )

    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[:7] = state["root_qpos"]
    mujoco.mj_forward(model, data)
    return model, data, camera_name


def run(args: argparse.Namespace) -> None:
    scene_path = args.scene.expanduser().resolve()
    motion_path = args.motion_file.expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"MuJoCo scene not found: {scene_path}")
    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Depth dimensions must be positive")
    if not (0.0 < args.horizontal_fov_deg < 180.0 and 0.0 < args.vertical_fov_deg < 180.0):
        raise ValueError("Camera FOV must be between 0 and 180 degrees")

    model, data, camera_name = build_scene(scene_path, motion_path, args.motion_frame, args.vertical_fov_deg)
    # MuJoCo stores vertical FOV. The configured 106x60 aspect and 58.6-degree
    # vertical FOV produce the requested horizontal field of view.
    implied_hfov = math.degrees(
        2.0
        * math.atan(
            (args.width / args.height)
            * math.tan(math.radians(args.vertical_fov_deg) * 0.5)
        )
    )
    if abs(implied_hfov - args.horizontal_fov_deg) > 1.0:
        print(
            f"[sim_gt_depth] warning: render aspect implies hfov={implied_hfov:.2f}, "
            f"configured hfov={args.horizontal_fov_deg:.2f}",
            flush=True,
        )

    shape = (1, 1, args.height, args.width)
    size_bytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
    created = False
    try:
        shm = shared_memory.SharedMemory(name=args.shm_name, create=True, size=size_bytes)
        created = True
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=args.shm_name)
        if shm.size != size_bytes:
            shm.close()
            raise RuntimeError(
                f"Existing /dev/shm/{args.shm_name} has {shm.size} bytes; expected {size_bytes}"
            )
    output = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)

    stop = False

    def _stop(_signum=None, _frame=None) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    renderer.enable_depth_rendering()
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = 0
    scene_option.geomgroup[3] = 0
    print(
        f"[sim_gt_depth] MuJoCo GT ready: /dev/shm/{args.shm_name} "
        f"shape={shape} motion={motion_path.name} frame={args.motion_frame}",
        flush=True,
    )

    period = 1.0 / max(float(args.rate_hz), 1.0)
    try:
        while not stop:
            started = time.monotonic()
            renderer.update_scene(data, camera=camera_name, scene_option=scene_option)
            output[0, 0] = np.asarray(renderer.render(), dtype=np.float32)
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    finally:
        renderer.close()
        shm.close()
        if created:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


def main(argv: Sequence[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
