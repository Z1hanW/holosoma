#!/usr/bin/env python3
"""Publish flat-ground MuJoCo GT plus a robot-only comparison channel."""

from __future__ import annotations

import argparse
import json
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
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--state-path", type=Path)
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


def build_scene(scene_path: Path, vertical_fov_deg: float):
    """Build a static all-zero G1 scene; non-robot geoms are filtered at render time."""
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
    floor = spec.geom("floor")
    if floor is None:
        raise RuntimeError("flat-ground geom 'floor' is missing from the MuJoCo scene")
    floor.group = 0

    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[:7] = [0.0, 0.0, 0.76, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    return model, data, camera_name


def read_robot_status(path: Path | None) -> dict:
    """Read one atomic real-robot telemetry snapshot."""
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def apply_robot_status(model: mujoco.MjModel, data: mujoco.MjData, status: dict) -> bool:
    """Update MuJoCo with measured joints/base pose from Viser telemetry."""
    names = status.get("dof_names")
    positions = status.get("q_actual")
    if not isinstance(names, list) or not isinstance(positions, list) or len(names) != len(positions):
        return False

    updated = 0
    for name, position in zip(names, positions):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if joint_id < 0 or not np.isfinite(position):
            continue
        data.qpos[model.jnt_qposadr[joint_id]] = float(position)
        updated += 1
    if updated == 0:
        return False

    base_position = np.asarray(status.get("base_position", ()), dtype=np.float64).reshape(-1)
    if base_position.size == 3 and np.isfinite(base_position).all() and abs(float(base_position[2])) > 0.1:
        data.qpos[2] = float(base_position[2])
    base_wxyz = np.asarray(status.get("base_wxyz", ()), dtype=np.float64).reshape(-1)
    if base_wxyz.size == 4 and np.isfinite(base_wxyz).all():
        norm = float(np.linalg.norm(base_wxyz))
        if norm > 1.0e-6:
            data.qpos[3:7] = base_wxyz / norm
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return True


def run(args: argparse.Namespace) -> None:
    scene_path = args.scene.expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"MuJoCo scene not found: {scene_path}")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Depth dimensions must be positive")
    if not (0.0 < args.horizontal_fov_deg < 180.0 and 0.0 < args.vertical_fov_deg < 180.0):
        raise ValueError("Camera FOV must be between 0 and 180 degrees")

    model, data, camera_name = build_scene(scene_path, args.vertical_fov_deg)
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

    # Channel 0 is the visible robot + flat-ground scene. Channel 1 is the
    # robot-only depth used to keep comparison metrics off the background.
    shape = (1, 2, args.height, args.width)
    size_bytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
    try:
        shm = shared_memory.SharedMemory(name=args.shm_name, create=True, size=size_bytes)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=args.shm_name)
        if shm.size != size_bytes:
            stale_size = shm.size
            shm.close()
            shm.unlink()
            print(
                f"[sim_gt_depth] replaced stale {stale_size}-byte /dev/shm/{args.shm_name} "
                f"with {size_bytes}-byte two-channel buffer",
                flush=True,
            )
            shm = shared_memory.SharedMemory(name=args.shm_name, create=True, size=size_bytes)
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
    scene_option.geomgroup[:] = 0
    scene_option.geomgroup[0] = 1  # Flat ground.
    scene_option.geomgroup[1] = 1  # G1 visual meshes.
    robot_option = mujoco.MjvOption()
    robot_option.geomgroup[:] = 0
    robot_option.geomgroup[1] = 1
    print(
        f"[sim_gt_depth] MuJoCo GT ready: /dev/shm/{args.shm_name} "
        f"shape={shape} flat-ground scene + robot-only comparison mask",
        flush=True,
    )

    period = 1.0 / max(float(args.rate_hz), 1.0)
    telemetry_live = False
    try:
        while not stop:
            started = time.monotonic()
            status = read_robot_status(args.state_path)
            status_applied = apply_robot_status(model, data, status)
            if status_applied and not telemetry_live:
                print(f"[sim_gt_depth] following measured robot pose from {args.state_path}", flush=True)
            telemetry_live = status_applied
            renderer.update_scene(data, camera=camera_name, scene_option=scene_option)
            output[0, 0] = np.asarray(renderer.render(), dtype=np.float32)
            renderer.update_scene(data, camera=camera_name, scene_option=robot_option)
            output[0, 1] = np.asarray(renderer.render(), dtype=np.float32)
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    finally:
        renderer.close()
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
