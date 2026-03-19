#!/usr/bin/env python3
"""Verify MuJoCo free-joint angular velocity frame semantics for the web demo scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


DEFAULT_SCENE = Path(__file__).resolve().parents[1] / "public" / "demo-assets" / "scene.xml"


def quat_apply_wxyz(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w = quat_wxyz[0]
    xyz = quat_wxyz[1:]
    t = 2.0 * np.cross(xyz, vec)
    return vec + w * t + np.cross(xyz, t)


def check_body(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, quat_wxyz: np.ndarray, angvel_local: np.ndarray) -> dict:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body {body_name!r} not found.")

    joint_adr = model.body_jntadr[body_id]
    if joint_adr < 0:
        raise ValueError(f"Body {body_name!r} does not have a joint.")

    qpos_adr = model.jnt_qposadr[joint_adr]
    qvel_adr = model.jnt_dofadr[joint_adr]

    data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat_wxyz
    data.qvel[qvel_adr : qvel_adr + 3] = 0.0
    data.qvel[qvel_adr + 3 : qvel_adr + 6] = angvel_local
    mujoco.mj_forward(model, data)

    world = np.zeros(6, dtype=np.float64)
    local = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, world, 0)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, local, 1)

    expected_world = quat_apply_wxyz(quat_wxyz, angvel_local)
    expected_local = angvel_local

    return {
        "body_name": body_name,
        "quat_wxyz": quat_wxyz.tolist(),
        "qvel_ang_local": angvel_local.tolist(),
        "mj_object_velocity_world_ang": world[:3].tolist(),
        "mj_object_velocity_local_ang": local[:3].tolist(),
        "rotated_local_to_world": expected_world.tolist(),
        "world_error_norm": float(np.linalg.norm(world[:3] - expected_world)),
        "local_error_norm": float(np.linalg.norm(local[:3] - expected_local)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--body", type=str, default="pelvis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)

    quat_wxyz = np.array([np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)], dtype=np.float64)
    angvel_local = np.array([0.2, -0.1, 0.3], dtype=np.float64)

    result = check_body(model, data, args.body, quat_wxyz, angvel_local)
    print(json.dumps(result, indent=2))

    if result["world_error_norm"] > 1e-5 or result["local_error_norm"] > 1e-3:
        raise SystemExit("Velocity frame check failed.")


if __name__ == "__main__":
    main()
