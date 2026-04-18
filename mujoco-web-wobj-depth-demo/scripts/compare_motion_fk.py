#!/usr/bin/env python3
"""Compare motion-clip world poses against MuJoCo forward kinematics."""

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
        "Use `/home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python` to run this script."
    ) from exc


def _decode_names(values: np.ndarray) -> list[str]:
    names: list[str] = []
    for item in values.tolist():
        if isinstance(item, (bytes, bytearray, np.bytes_)):
            names.append(item.decode("utf-8"))
        else:
            names.append(str(item))
    return names


def _load_manifest(asset_root: Path) -> dict:
    manifest_path = asset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_clip_config(asset_root: Path, clip_id: str | None) -> tuple[str, dict]:
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


def _quat_error_deg(quat_a_wxyz: np.ndarray, quat_b_wxyz: np.ndarray) -> float:
    dot = abs(float(np.dot(quat_a_wxyz, quat_b_wxyz)))
    dot = min(1.0, max(-1.0, dot))
    return float(2.0 * math.degrees(math.acos(dot)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "public" / "demo-assets",
    )
    parser.add_argument("--clip-id", type=str, default=None)
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset_root = args.asset_root.expanduser().resolve()
    clip_id, config = _load_clip_config(asset_root, args.clip_id)
    motion_path = Path(config["motion_file"]).expanduser().resolve()

    with np.load(motion_path, allow_pickle=True) as motion_data:
        body_names = _decode_names(np.asarray(motion_data["body_names"]))
        joint_names = _decode_names(np.asarray(motion_data["joint_names"]))
        body_pos_w = np.asarray(motion_data["body_pos_w"], dtype=np.float64)
        body_quat_w = np.asarray(motion_data["body_quat_w"], dtype=np.float64)
        joint_pos = np.asarray(motion_data["joint_pos"], dtype=np.float64)
        if joint_pos.shape[1] == len(joint_names) + 7:
            joint_pos = joint_pos[:, 7:]

    frame_idx = int(np.clip(args.frame_idx, 0, body_pos_w.shape[0] - 1))
    model = mj.MjModel.from_xml_path(str(asset_root / config["scene_path"]))
    data = mj.MjData(model)

    root_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pelvis")
    root_joint_id = int(model.body_jntadr[root_body_id])
    root_qpos_adr = int(model.jnt_qposadr[root_joint_id])

    root_idx = body_names.index("pelvis") if "pelvis" in body_names else 0
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qpos[root_qpos_adr : root_qpos_adr + 3] = body_pos_w[frame_idx, root_idx]
    data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = body_quat_w[frame_idx, root_idx]

    joint_name_to_index = {name: idx for idx, name in enumerate(joint_names)}
    for joint_name in config["dof_names"]:
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[int(model.jnt_qposadr[joint_id])] = joint_pos[frame_idx, joint_name_to_index[joint_name]]

    if config["motion"].get("initial_object_pos_w") is not None:
        object_body_name = config.get("object_body_name", "object_baseLink")
        object_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, object_body_name)
        if object_body_id < 0:
            object_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "largebox_link")
        if object_body_id >= 0 and model.body_jntnum[object_body_id] > 0:
            object_joint_id = int(model.body_jntadr[object_body_id])
            object_qpos_adr = int(model.jnt_qposadr[object_joint_id])
            data.qpos[object_qpos_adr : object_qpos_adr + 3] = np.asarray(config["motion"]["initial_object_pos_w"], dtype=np.float64)
            data.qpos[object_qpos_adr + 3 : object_qpos_adr + 7] = np.asarray(
                config["motion"]["initial_object_quat_wxyz"],
                dtype=np.float64,
            )

    mj.mj_forward(model, data)

    rows: list[dict] = []
    for clip_body_idx, body_name in enumerate(body_names):
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue
        mujoco_pos = np.asarray(data.xpos[body_id], dtype=np.float64).reshape(3)
        mujoco_quat = np.asarray(data.xquat[body_id], dtype=np.float64).reshape(4)
        clip_pos = body_pos_w[frame_idx, clip_body_idx]
        clip_quat = body_quat_w[frame_idx, clip_body_idx]
        rows.append(
            {
                "body_name": body_name,
                "pos_err_m": float(np.linalg.norm(mujoco_pos - clip_pos)),
                "quat_err_deg": _quat_error_deg(mujoco_quat, clip_quat),
                "mujoco_z": float(mujoco_pos[2]),
                "clip_z": float(clip_pos[2]),
            }
        )

    rows.sort(key=lambda row: (row["pos_err_m"], row["quat_err_deg"]), reverse=True)
    pos_errors = np.asarray([row["pos_err_m"] for row in rows], dtype=np.float64)
    quat_errors = np.asarray([row["quat_err_deg"] for row in rows], dtype=np.float64)

    print(f"Clip: {clip_id}")
    print(f"Frame: {frame_idx}")
    print(f"Scene: {asset_root / config['scene_path']}")
    print(f"Motion: {motion_path}")
    print(f"Common bodies: {len(rows)}")
    print()
    print("Error summary")
    print(f"  median_pos_err_m: {float(np.median(pos_errors)):.9f}")
    print(f"  max_pos_err_m: {float(pos_errors.max(initial=0.0)):.9f}")
    print(f"  median_quat_err_deg: {float(np.median(quat_errors)):.9f}")
    print(f"  max_quat_err_deg: {float(quat_errors.max(initial=0.0)):.9f}")
    print()
    print("Top bodies by position error")
    for row in rows[: args.top_k]:
        print(
            "  "
            f"{row['body_name']:<30} "
            f"pos_err={row['pos_err_m']:.9f} m "
            f"quat_err={row['quat_err_deg']:.6f} deg "
            f"mujoco_z={row['mujoco_z']:.6f} "
            f"clip_z={row['clip_z']:.6f}"
        )


if __name__ == "__main__":
    main()
