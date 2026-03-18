#!/usr/bin/env python3
"""Check strict-warp camera alignment for the staged MuJoCo web demo."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    import mujoco as mj
except ModuleNotFoundError as exc:
    raise SystemExit(
        "MuJoCo Python bindings are not available in the current interpreter. "
        "Use `npm run check-camera` from this demo folder, or run the script with "
        "`/home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python`."
    ) from exc

THREE_CAMERA_TO_STRICT_XYZW = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
ROOT_BODY_CANDIDATES = ("pelvis", "pelvis_link", "base_link", "torso_link")


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm <= 1.0e-12:
        return vec.copy()
    return vec / norm


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.asarray(a, dtype=np.float64)
    bx, by, bz, bw = np.asarray(b, dtype=np.float64)
    return np.asarray(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _quat_conj_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return np.asarray([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float64)


def _quat_apply_xyzw(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat = _normalize(np.asarray(quat, dtype=np.float64))
    vec_quat = np.asarray([vec[0], vec[1], vec[2], 0.0], dtype=np.float64)
    rotated = _quat_mul_xyzw(_quat_mul_xyzw(quat, vec_quat), _quat_conj_xyzw(quat))
    return rotated[:3]


def _quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize(a)
    b = _normalize(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def _resolve_intrinsics(width: int, height: int, vfov_deg: float, hfov_deg: float) -> tuple[float, float, float, float]:
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    fy = height / (2.0 * math.tan(math.radians(vfov_deg) / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def _resolve_root_body_name(model: mj.MjModel) -> str:
    for name in ROOT_BODY_CANDIDATES:
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0 and model.body_jntnum[body_id] > 0:
            return name
    raise RuntimeError(f"Could not resolve a root body from {ROOT_BODY_CANDIDATES!r}")


def _load_manifest(asset_root: Path) -> dict:
    manifest_path = asset_root / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "default_clip_id": "default",
        "clips": [{"id": "default", "label": "Default", "config_path": "demo-config.json"}],
    }


def _resolve_config(asset_root: Path, *, clip_id: str | None, config_path: Path | None) -> tuple[str, Path, dict]:
    if config_path is not None:
        config_abspath = config_path.expanduser().resolve()
        return config_abspath.stem, config_abspath, json.loads(config_abspath.read_text(encoding="utf-8"))

    manifest = _load_manifest(asset_root)
    clips = manifest.get("clips", [])
    if not clips:
        raise RuntimeError(f"No clips found under {asset_root}")
    selected_id = clip_id or manifest.get("default_clip_id") or clips[0]["id"]
    for clip in clips:
        if clip["id"] == selected_id:
            config_abspath = (asset_root / clip["config_path"]).resolve()
            return clip["id"], config_abspath, json.loads(config_abspath.read_text(encoding="utf-8"))
    raise RuntimeError(f"Unknown clip id {selected_id!r}. Available: {[clip['id'] for clip in clips]}")


def _reset_model_state(model: mj.MjModel, data: mj.MjData, config: dict) -> None:
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    root_body_name = _resolve_root_body_name(model)
    root_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, root_body_name)
    root_joint_id = model.body_jntadr[root_body_id]
    root_qpos_adr = int(model.jnt_qposadr[root_joint_id])

    root_pos = np.asarray(config["motion"]["initial_root_pos_w"], dtype=np.float64)
    root_quat = np.asarray(config["motion"]["initial_root_quat_wxyz"], dtype=np.float64)
    data.qpos[root_qpos_adr : root_qpos_adr + 3] = root_pos
    data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = root_quat

    initial_joint_pos = np.asarray(config["motion"]["initial_joint_pos"], dtype=np.float64)
    initial_joint_vel = np.asarray(config["motion"]["initial_joint_vel"], dtype=np.float64)
    for index, joint_name in enumerate(config["dof_names"]):
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"Joint {joint_name!r} not found in scene.")
        qpos_adr = int(model.jnt_qposadr[joint_id])
        qvel_adr = int(model.jnt_dofadr[joint_id])
        data.qpos[qpos_adr] = initial_joint_pos[index]
        data.qvel[qvel_adr] = initial_joint_vel[index]

    object_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "largebox_link")
    if (
        object_body_id >= 0
        and model.body_jntnum[object_body_id] > 0
        and config["motion"].get("initial_object_pos_w") is not None
        and config["motion"].get("initial_object_quat_wxyz") is not None
    ):
        object_joint_id = model.body_jntadr[object_body_id]
        object_qpos_adr = int(model.jnt_qposadr[object_joint_id])
        data.qpos[object_qpos_adr : object_qpos_adr + 3] = np.asarray(
            config["motion"]["initial_object_pos_w"], dtype=np.float64
        )
        data.qpos[object_qpos_adr + 3 : object_qpos_adr + 7] = np.asarray(
            config["motion"]["initial_object_quat_wxyz"], dtype=np.float64
        )

    mj.mj_forward(model, data)


def _format_vec(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value): .6f}" for value in np.asarray(vec).reshape(-1)) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "public" / "demo-assets",
    )
    parser.add_argument("--clip-id", type=str, default=None)
    parser.add_argument("--config-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset_root = args.asset_root.expanduser().resolve()
    clip_id, config_path, config = _resolve_config(asset_root, clip_id=args.clip_id, config_path=args.config_path)

    scene_path = (asset_root / config["scene_path"]).resolve()
    model = mj.MjModel.from_xml_path(str(scene_path))
    data = mj.MjData(model)
    _reset_model_state(model, data, config)

    perception = config["perception"]
    camera_body_name = perception["camera_body_name"]
    camera_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, camera_body_name)
    if camera_body_id < 0:
        raise RuntimeError(f"Camera body {camera_body_name!r} not found in {scene_path}")

    body_pos = np.asarray(data.xpos[camera_body_id], dtype=np.float64)
    body_quat_wxyz = np.asarray(data.xquat[camera_body_id], dtype=np.float64)
    body_quat_xyzw = _quat_wxyz_to_xyzw(body_quat_wxyz)

    sensor_offset = np.asarray(perception["sensor_offset"], dtype=np.float64)
    camera_mount_quat = np.asarray(perception["camera_mount_quat"], dtype=np.float64)
    camera_frame_quat = np.asarray(perception["camera_frame_quat"], dtype=np.float64)

    camera_pos = body_pos + _quat_apply_xyzw(body_quat_xyzw, sensor_offset)
    strict_combo = _quat_mul_xyzw(camera_mount_quat, camera_frame_quat)
    world_strict_quat = _quat_mul_xyzw(body_quat_xyzw, strict_combo)
    three_quat = _quat_mul_xyzw(world_strict_quat, THREE_CAMERA_TO_STRICT_XYZW)

    strict_forward = _quat_apply_xyzw(world_strict_quat, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    three_forward = _quat_apply_xyzw(three_quat, np.asarray([0.0, 0.0, -1.0], dtype=np.float64))
    strict_right = _quat_apply_xyzw(world_strict_quat, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    three_right = _quat_apply_xyzw(three_quat, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    strict_down = _quat_apply_xyzw(world_strict_quat, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
    three_down = _quat_apply_xyzw(three_quat, np.asarray([0.0, -1.0, 0.0], dtype=np.float64))

    width = int(perception["camera_width"])
    height = int(perception["camera_height"])
    fx, fy, cx, cy = _resolve_intrinsics(
        width,
        height,
        vfov_deg=float(perception["camera_vfov_deg"]),
        hfov_deg=float(perception["camera_hfov_deg"]),
    )
    principal_ray_strict = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    center_u = width // 2
    center_v = height // 2
    center_ray_strict = _normalize(np.asarray([(center_u - cx) / fx, (center_v - cy) / fy, 1.0], dtype=np.float64))
    top_left_ray_strict = _normalize(np.asarray([(0.0 - cx) / fx, (0.0 - cy) / fy, 1.0], dtype=np.float64))
    bottom_left_ray_strict = _normalize(np.asarray([(0.0 - cx) / fx, ((height - 1) - cy) / fy, 1.0], dtype=np.float64))

    principal_ray_world = _quat_apply_xyzw(world_strict_quat, principal_ray_strict)
    center_ray_world = _quat_apply_xyzw(world_strict_quat, center_ray_strict)

    print(f"Asset root: {asset_root}")
    print(f"Clip id: {clip_id}")
    print(f"Config path: {config_path}")
    print(f"Scene path: {scene_path}")
    print()
    print("Camera body pose")
    print(f"  body_name: {camera_body_name}")
    print(f"  body_pos_world: {_format_vec(body_pos)}")
    print(f"  body_quat_wxyz: {_format_vec(body_quat_wxyz)}")
    print(f"  sensor_offset_body: {_format_vec(sensor_offset)}")
    print(f"  camera_pos_world: {_format_vec(camera_pos)}")
    print()
    print("Camera quaternions")
    print(f"  strict_world_quat_xyzw: {_format_vec(world_strict_quat)}")
    print(f"  three_world_quat_xyzw: {_format_vec(three_quat)}")
    print()
    print("Axis alignment")
    print(f"  forward_error_deg: {_angle_deg(strict_forward, three_forward):.6f}")
    print(f"  right_error_deg: {_angle_deg(strict_right, three_right):.6f}")
    print(f"  down_error_deg: {_angle_deg(strict_down, three_down):.6f}")
    print(f"  principal_vs_forward_deg: {_angle_deg(principal_ray_world, strict_forward):.6f}")
    print(f"  center_pixel_vs_forward_deg: {_angle_deg(center_ray_world, strict_forward):.6f}")
    print()
    print("Camera intrinsics")
    print(f"  resolution: {width} x {height}")
    print(f"  fx/fy: {fx:.6f} / {fy:.6f}")
    print(f"  cx/cy: {cx:.6f} / {cy:.6f}")
    print()
    print("Strict-frame rays")
    print(f"  principal_ray: {_format_vec(principal_ray_strict)}")
    print(f"  center_pixel_ray: {_format_vec(center_ray_strict)}")
    print(f"  top_left_ray: {_format_vec(top_left_ray_strict)}")
    print(f"  bottom_left_ray: {_format_vec(bottom_left_ray_strict)}")
    print()
    print("World-frame rays")
    print(f"  forward_world: {_format_vec(strict_forward)}")
    print(f"  principal_world: {_format_vec(principal_ray_world)}")
    print(f"  center_pixel_world: {_format_vec(center_ray_world)}")


if __name__ == "__main__":
    main()
