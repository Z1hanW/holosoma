#!/usr/bin/env python3
"""Compare MuJoCo reset modes for the web demo and report ground penetration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnx

try:
    import mujoco as mj
except ModuleNotFoundError as exc:
    raise SystemExit(
        "MuJoCo Python bindings are not available in the current interpreter. "
        "Use `/home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python` to run this script."
    ) from exc


def _load_manifest(asset_root: Path) -> dict:
    manifest_path = asset_root / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "default_clip_id": "default",
        "clips": [{"id": "default", "label": "Default", "config_path": "demo-config.json"}],
    }


def _resolve_config(asset_root: Path, clip_id: str | None) -> tuple[str, dict]:
    manifest = _load_manifest(asset_root)
    clips = manifest.get("clips", [])
    if not clips:
        raise RuntimeError(f"No clips found under {asset_root}")
    selected_id = clip_id or manifest.get("default_clip_id") or clips[0]["id"]
    for clip in clips:
        if clip["id"] == selected_id:
            config_path = asset_root / clip["config_path"]
            return clip["id"], json.loads(config_path.read_text(encoding="utf-8"))
    raise RuntimeError(f"Unknown clip id {selected_id!r}")


def _quat_xyzw_to_euler_xyz(quat_xyzw: np.ndarray) -> tuple[float, float, float]:
    x, y, z, w = [float(v) for v in quat_xyzw]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw


def _quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return np.asarray(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


def _geom_lowest_z(model: mj.MjModel, data: mj.MjData, geom_id: int) -> float:
    geom_type = int(model.geom_type[geom_id])
    xpos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64).reshape(3)
    xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64).reshape(-1)

    if geom_type == mj.mjtGeom.mjGEOM_PLANE:
        return 0.0
    if geom_type == mj.mjtGeom.mjGEOM_SPHERE:
        return float(xpos[2] - size[0])
    if geom_type == mj.mjtGeom.mjGEOM_CAPSULE:
        half, radius = size[1], size[0]
        points = np.asarray([[0.0, 0.0, half], [0.0, 0.0, -half]], dtype=np.float64)
        world = (xmat @ points.T).T + xpos
        return float(world[:, 2].min() - radius)
    if geom_type == mj.mjtGeom.mjGEOM_BOX:
        extents = size[:3]
        corners = np.asarray(
            [[sx, sy, sz] for sx in (-extents[0], extents[0]) for sy in (-extents[1], extents[1]) for sz in (-extents[2], extents[2])],
            dtype=np.float64,
        )
        world = (xmat @ corners.T).T + xpos
        return float(world[:, 2].min())
    if geom_type == mj.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        vert_adr = int(model.mesh_vertadr[mesh_id])
        vert_num = int(model.mesh_vertnum[mesh_id])
        verts = np.asarray(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=np.float64).reshape(-1, 3)
        if verts.size == 0:
            return float(xpos[2])
        world = (xmat @ verts.T).T + xpos
        return float(world[:, 2].min())
    return float(xpos[2] - float(model.geom_rbound[geom_id]))


def _set_state(
    *,
    model: mj.MjModel,
    data: mj.MjData,
    config: dict,
    init_state: dict,
    mode: str,
) -> dict:
    root_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pelvis")
    root_joint_id = int(model.body_jntadr[root_body_id])
    root_qpos_adr = int(model.jnt_qposadr[root_joint_id])
    root_qvel_adr = int(model.jnt_dofadr[root_joint_id])

    object_body_name = config.get("object_body_name", "object_baseLink")
    object_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, object_body_name)
    if object_body_id < 0:
        object_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "largebox_link")
    object_joint_id = int(model.body_jntadr[object_body_id]) if object_body_id >= 0 and model.body_jntnum[object_body_id] > 0 else -1
    object_qpos_adr = int(model.jnt_qposadr[object_joint_id]) if object_joint_id >= 0 else -1

    mj.mj_resetData(model, data)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    motion = config["motion"]
    root_pos = np.asarray(motion["initial_root_pos_w"], dtype=np.float64)
    root_quat_wxyz = np.asarray(motion["initial_root_quat_wxyz"], dtype=np.float64)
    raw_joint_pos = np.asarray(motion["initial_joint_pos"], dtype=np.float64)
    raw_joint_vel = np.asarray(motion["initial_joint_vel"], dtype=np.float64)

    if mode == "demo_raw":
        set_root_pos = root_pos.copy()
        set_root_quat_wxyz = root_quat_wxyz.copy()
        set_joint_pos = raw_joint_pos.copy()
        set_joint_vel = raw_joint_vel.copy()
    elif mode == "isaac_training_default_pose":
        init_root_quat_xyzw = np.asarray(init_state["rot"], dtype=np.float64)
        init_roll, init_pitch, _ = _quat_xyzw_to_euler_xyz(init_root_quat_xyzw)
        motion_quat_xyzw = np.asarray([root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]], dtype=np.float64)
        _, _, motion_yaw = _quat_xyzw_to_euler_xyz(motion_quat_xyzw)
        default_quat_xyzw = _quat_from_euler_xyz(init_roll, init_pitch, motion_yaw)
        set_root_pos = root_pos.copy()
        set_root_pos[2] = float(init_state["pos"][2])
        set_root_quat_wxyz = np.asarray(
            [default_quat_xyzw[3], default_quat_xyzw[0], default_quat_xyzw[1], default_quat_xyzw[2]],
            dtype=np.float64,
        )
        set_joint_pos = np.asarray(
            [float(init_state["default_joint_angles"].get(name, 0.0)) for name in config["dof_names"]],
            dtype=np.float64,
        )
        set_joint_vel = np.zeros_like(set_joint_pos)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    data.qpos[root_qpos_adr : root_qpos_adr + 3] = set_root_pos
    data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = set_root_quat_wxyz
    data.qvel[root_qvel_adr : root_qvel_adr + 6] = 0.0

    for index, joint_name in enumerate(config["dof_names"]):
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[int(model.jnt_qposadr[joint_id])] = set_joint_pos[index]
        data.qvel[int(model.jnt_dofadr[joint_id])] = set_joint_vel[index]

    if object_qpos_adr >= 0 and motion.get("initial_object_pos_w") is not None:
        data.qpos[object_qpos_adr : object_qpos_adr + 3] = np.asarray(motion["initial_object_pos_w"], dtype=np.float64)
        data.qpos[object_qpos_adr + 3 : object_qpos_adr + 7] = np.asarray(
            motion["initial_object_quat_wxyz"],
            dtype=np.float64,
        )

    mj.mj_forward(model, data)
    return {
        "root_pos": set_root_pos,
        "root_quat_wxyz": set_root_quat_wxyz,
        "joint_l2_vs_default": float(
            np.linalg.norm(
                raw_joint_pos
                - np.asarray(
                    [float(init_state["default_joint_angles"].get(name, 0.0)) for name in config["dof_names"]],
                    dtype=np.float64,
                )
            )
        ),
    }


def _analyze_mode(model: mj.MjModel, data: mj.MjData, mode: str) -> dict:
    geoms = []
    for geom_id in range(model.ngeom):
        geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id) or ""
        geoms.append(
            {
                "geom_id": geom_id,
                "geom_name": geom_name,
                "body_name": body_name,
                "lowest_z": _geom_lowest_z(model, data, geom_id),
                "contype": int(model.geom_contype[geom_id]),
                "group": int(model.geom_group[geom_id]),
            }
        )
    geoms.sort(key=lambda row: row["lowest_z"])

    contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        contacts.append(
            {
                "dist": float(contact.dist),
                "geom1": mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or str(int(contact.geom1)),
                "geom2": mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or str(int(contact.geom2)),
            }
        )
    contacts.sort(key=lambda row: row["dist"])
    return {
        "geoms": geoms,
        "contacts": contacts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "public" / "demo-assets",
    )
    parser.add_argument("--clip-id", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset_root = args.asset_root.expanduser().resolve()
    clip_id, config = _resolve_config(asset_root, args.clip_id)
    model = mj.MjModel.from_xml_path(str(asset_root / config["scene_path"]))
    data = mj.MjData(model)

    onnx_model = onnx.load(str(asset_root / config["model_path"]))
    metadata = {prop.key: json.loads(prop.value) for prop in onnx_model.metadata_props}
    init_state = metadata["experiment_config"]["robot"]["init_state"]

    print(f"Clip: {clip_id}")
    print(f"Scene: {asset_root / config['scene_path']}")
    print(f"Motion root z: {float(config['motion']['initial_root_pos_w'][2]):.6f}")
    print(f"Training init z: {float(init_state['pos'][2]):.6f}")
    print()

    for mode in ("demo_raw", "isaac_training_default_pose"):
        mode_state = _set_state(model=model, data=data, config=config, init_state=init_state, mode=mode)
        report = _analyze_mode(model, data, mode)
        print(f"Mode: {mode}")
        print(f"  root_pos: {mode_state['root_pos'].tolist()}")
        print(f"  root_quat_wxyz: {mode_state['root_quat_wxyz'].tolist()}")
        print(f"  joint_l2_vs_default: {mode_state['joint_l2_vs_default']:.6f}")
        print("  lowest geoms:")
        for row in report["geoms"][: args.top_k]:
            print(
                "   ",
                f"{row['lowest_z']:+.6f}",
                row["geom_name"],
                f"(body={row['body_name']}, contype={row['contype']}, group={row['group']})",
            )
        print("  most negative contacts:")
        for row in report["contacts"][: args.top_k]:
            print("   ", f"{row['dist']:+.6f}", row["geom1"], "<->", row["geom2"])
        print()


if __name__ == "__main__":
    main()
