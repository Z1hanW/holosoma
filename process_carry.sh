#!/usr/bin/env bash
set -euo pipefail

# Filter motion/geometry pairs by "box between hands" condition and save *_carry copies.
#
# Usage:
#   MOTION_DIR=/ABS/PATH/to/motions GEOMETRY_DIR=/ABS/PATH/to/geometry OBJECT_URDF_DIR=/ABS/PATH/to/urdfs ./process_carry.sh
#
# Optional overrides:
#   ROBOT=g1_29dof LEFT_LINK=left_wrist_yaw_link RIGHT_LINK=right_wrist_yaw_link
#   OUTPUT_MOTION_DIR=/ABS/PATH OUT_GEOMETRY_DIR=/ABS/PATH OUT_OBJECT_DIR=/ABS/PATH
#   DIST_THRESH=0.15  # (optional) max distance to hand-line segment

MOTION_DIR=${MOTION_DIR:-""}
GEOMETRY_DIR=${GEOMETRY_DIR:-""}
OBJECT_URDF_DIR=${OBJECT_URDF_DIR:-""}
ROBOT=${ROBOT:-"g1_29dof"}
LEFT_LINK=${LEFT_LINK:-""}
RIGHT_LINK=${RIGHT_LINK:-""}
OUTPUT_MOTION_DIR=${OUTPUT_MOTION_DIR:-""}
OUTPUT_GEOMETRY_DIR=${OUTPUT_GEOMETRY_DIR:-""}
OUTPUT_OBJECT_DIR=${OUTPUT_OBJECT_DIR:-""}
DIST_THRESH=${DIST_THRESH:-""}

if [[ -z "${MOTION_DIR}" ]]; then
  echo "[ERROR] MOTION_DIR is required." >&2
  exit 1
fi

export MOTION_DIR GEOMETRY_DIR OBJECT_URDF_DIR ROBOT LEFT_LINK RIGHT_LINK
export OUTPUT_MOTION_DIR OUTPUT_GEOMETRY_DIR OUTPUT_OBJECT_DIR DIST_THRESH

python - <<'PY'
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from holosoma.config_values import robot as robot_values
from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.path import resolve_data_file_path

try:
    import yourdfpy  # type: ignore
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError("yourdfpy is required to run process_carry.sh") from exc


def _resolve_data_path(path: str) -> Path:
    if path.startswith("@holosoma/"):
        return Path(get_holosoma_root()) / path[len("@holosoma/") :]
    return Path(resolve_data_file_path(path))


def _decode_names(arr: np.ndarray) -> list[str]:
    names: list[str] = []
    for item in arr.tolist():
        if isinstance(item, (bytes, np.bytes_)):
            names.append(item.decode("utf-8"))
        else:
            names.append(str(item))
    return names


def _first_present(candidates: list[str], names: list[str]) -> str | None:
    for cand in candidates:
        if cand in names:
            return cand
    return None


def _build_joint_index_map(robot_cfg) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(robot_cfg.dof_names)}


def _load_urdf(robot_cfg):
    asset_root = _resolve_data_path(robot_cfg.asset.asset_root)
    urdf_path = _resolve_data_path(str(Path(asset_root) / robot_cfg.asset.urdf_file))
    return yourdfpy.URDF.load(str(urdf_path), load_meshes=False, build_scene_graph=True)


def _hand_positions_from_body_pos(
    body_pos_w: np.ndarray,
    body_names: list[str],
    left_name: str,
    right_name: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if left_name not in body_names or right_name not in body_names:
        return None
    left_idx = body_names.index(left_name)
    right_idx = body_names.index(right_name)
    return body_pos_w[:, left_idx, :], body_pos_w[:, right_idx, :]


def _hand_positions_from_qpos(
    qpos: np.ndarray,
    robot_cfg,
    urdf,
    left_link: str,
    right_link: str,
) -> tuple[np.ndarray, np.ndarray]:
    joint_count = len(robot_cfg.dof_names)
    if qpos.shape[1] < 7 + joint_count:
        raise ValueError(f"qpos has insufficient dims: {qpos.shape[1]} < {7 + joint_count}")

    urdf_act = urdf.actuated_joint_names
    name_to_qpos_idx = _build_joint_index_map(robot_cfg)
    missing = [name for name in urdf_act if name not in name_to_qpos_idx]
    if missing:
        raise ValueError(f"URDF actuated joints not found in robot config: {missing}")
    urdf_order_idx = [name_to_qpos_idx[name] for name in urdf_act]

    left_positions = []
    right_positions = []
    for frame in qpos:
        root_pos = frame[:3]
        root_quat_wxyz = frame[3:7]
        root_rot = Rotation.from_quat(
            [root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]]
        )
        joint_vals = frame[7 : 7 + joint_count]
        urdf.update_cfg(joint_vals[urdf_order_idx])
        t_left = urdf.get_transform(frame_to=left_link, frame_from="world")
        t_right = urdf.get_transform(frame_to=right_link, frame_from="world")
        left_local = t_left[:3, 3]
        right_local = t_right[:3, 3]
        left_positions.append(root_pos + root_rot.apply(left_local))
        right_positions.append(root_pos + root_rot.apply(right_local))
    return np.asarray(left_positions), np.asarray(right_positions)


def _object_positions_from_npz(data: dict, qpos: np.ndarray | None, joint_count: int) -> np.ndarray | None:
    if "object_pos_w" in data:
        obj = np.asarray(data["object_pos_w"], dtype=np.float32)
        if obj.ndim == 2 and obj.shape[1] == 3:
            return obj
    if qpos is None:
        return None
    if qpos.shape[1] >= 7 + joint_count + 7:
        return qpos[:, -7:-4]
    return None


def _between_hands(
    left: np.ndarray,
    right: np.ndarray,
    obj: np.ndarray,
    dist_thresh: float | None,
) -> bool:
    n = min(left.shape[0], right.shape[0], obj.shape[0])
    if n == 0:
        return False
    left = left[:n]
    right = right[:n]
    obj = obj[:n]
    vec = right - left
    denom = np.sum(vec * vec, axis=1)
    valid = denom > 1e-8
    if not np.any(valid):
        return False
    t = np.sum((obj - left) * vec, axis=1) / denom
    cond = valid & (t >= 0.0) & (t <= 1.0)
    if dist_thresh is not None:
        proj = left + vec * t[:, None]
        dist = np.linalg.norm(obj - proj, axis=1)
        cond = cond & (dist <= dist_thresh)
    return bool(np.any(cond))


def main() -> None:
    env = dict(**{k: v for k, v in vars(__import__("os").environ).items()})

    motion_dir = _resolve_data_path(env["MOTION_DIR"])
    geometry_dir = _resolve_data_path(env["GEOMETRY_DIR"]) if env.get("GEOMETRY_DIR") else None
    object_dir = _resolve_data_path(env["OBJECT_URDF_DIR"]) if env.get("OBJECT_URDF_DIR") else None
    robot_name = env.get("ROBOT", "g1_29dof")
    left_link_override = env.get("LEFT_LINK", "").strip()
    right_link_override = env.get("RIGHT_LINK", "").strip()
    out_motion = _resolve_data_path(env["OUTPUT_MOTION_DIR"]) if env.get("OUTPUT_MOTION_DIR") else None
    out_geometry = _resolve_data_path(env["OUTPUT_GEOMETRY_DIR"]) if env.get("OUTPUT_GEOMETRY_DIR") else None
    out_object = _resolve_data_path(env["OUTPUT_OBJECT_DIR"]) if env.get("OUTPUT_OBJECT_DIR") else None
    dist_thresh = env.get("DIST_THRESH", "").strip()
    dist_thresh_val = float(dist_thresh) if dist_thresh else None

    if not motion_dir.is_dir():
        raise FileNotFoundError(f"Motion dir not found: {motion_dir}")

    motion_paths = sorted(list(motion_dir.glob("*.npz")) + list(motion_dir.glob("*.NPZ")))
    if not motion_paths:
        raise FileNotFoundError(f"No .npz files in {motion_dir}")
    motion_map = {p.stem: p for p in motion_paths}

    geom_map: dict[str, Path] = {}
    obj_map: dict[str, Path] = {}
    pair_names = sorted(motion_map)

    if geometry_dir is not None:
        geom_paths = sorted(list(geometry_dir.glob("*.obj")) + list(geometry_dir.glob("*.OBJ")))
        geom_map = {p.stem: p for p in geom_paths}
        pair_names = sorted(set(pair_names) & set(geom_map))

    if object_dir is not None:
        obj_paths = sorted(list(object_dir.glob("*.urdf")) + list(object_dir.glob("*.URDF")))
        obj_map = {p.stem: p for p in obj_paths}
        pair_names = sorted(set(pair_names) & set(obj_map))

    if not pair_names:
        raise RuntimeError("No matching motion/geometry/object pairs found.")

    if out_motion is None:
        out_motion = motion_dir.parent / f"{motion_dir.name}_carry"
    if geometry_dir is not None and out_geometry is None:
        out_geometry = geometry_dir.parent / f"{geometry_dir.name}_carry"
    if object_dir is not None and out_object is None:
        out_object = object_dir.parent / f"{object_dir.name}_carry"

    out_motion.mkdir(parents=True, exist_ok=True)
    if out_geometry is not None:
        out_geometry.mkdir(parents=True, exist_ok=True)
    if out_object is not None:
        out_object.mkdir(parents=True, exist_ok=True)

    defaults = robot_values.DEFAULTS
    if robot_name not in defaults:
        raise ValueError(f"Unknown robot '{robot_name}'. Available: {sorted(defaults.keys())}")
    robot_cfg = defaults[robot_name]
    joint_count = len(robot_cfg.dof_names)
    urdf = _load_urdf(robot_cfg)

    candidate_pairs = [
        ("left_rubber_hand", "right_rubber_hand"),
        ("left_hand_link", "right_hand_link"),
        ("left_wrist_yaw_link", "right_wrist_yaw_link"),
        ("left_wrist_pitch_link", "right_wrist_pitch_link"),
        ("left_wrist_roll_link", "right_wrist_roll_link"),
    ]
    left_link = left_link_override or None
    right_link = right_link_override or None

    if left_link is None or right_link is None:
        urdf_links = list(urdf.link_map.keys())
        for cand_left, cand_right in candidate_pairs:
            if cand_left in urdf_links and cand_right in urdf_links:
                left_link = cand_left
                right_link = cand_right
                break

    if left_link is None or right_link is None:
        raise RuntimeError("Could not resolve left/right hand link names. Set LEFT_LINK/RIGHT_LINK.")

    kept = 0
    dropped = 0
    skipped = 0

    for name in pair_names:
        motion_path = motion_map[name]
        with np.load(motion_path, allow_pickle=True) as data:
            payload = {k: data[k] for k in data.files}

        qpos = payload.get("qpos")
        if qpos is not None:
            qpos = np.asarray(qpos, dtype=np.float32)

        obj_pos = _object_positions_from_npz(payload, qpos, joint_count)
        if obj_pos is None:
            skipped += 1
            continue

        left_pos = right_pos = None
        if "body_pos_w" in payload and "body_names" in payload:
            body_pos_w = np.asarray(payload["body_pos_w"], dtype=np.float32)
            body_names = _decode_names(np.asarray(payload["body_names"]))
            left_candidate = _first_present([left_link], body_names) or _first_present(
                [c[0] for c in candidate_pairs], body_names
            )
            right_candidate = _first_present([right_link], body_names) or _first_present(
                [c[1] for c in candidate_pairs], body_names
            )
            if left_candidate and right_candidate:
                hand_pos = _hand_positions_from_body_pos(body_pos_w, body_names, left_candidate, right_candidate)
                if hand_pos is not None:
                    left_pos, right_pos = hand_pos

        if left_pos is None or right_pos is None:
            if qpos is None:
                skipped += 1
                continue
            left_pos, right_pos = _hand_positions_from_qpos(qpos, robot_cfg, urdf, left_link, right_link)

        if not _between_hands(left_pos, right_pos, obj_pos, dist_thresh_val):
            dropped += 1
            continue

        out_motion_path = out_motion / f"{motion_path.stem}_carry{motion_path.suffix}"
        shutil.copy2(motion_path, out_motion_path)

        if out_geometry is not None and name in geom_map:
            geom_path = geom_map[name]
            out_geom_path = out_geometry / f"{geom_path.stem}_carry{geom_path.suffix}"
            shutil.copy2(geom_path, out_geom_path)

        if out_object is not None and name in obj_map:
            obj_path = obj_map[name]
            out_obj_path = out_object / f"{obj_path.stem}_carry{obj_path.suffix}"
            shutil.copy2(obj_path, out_obj_path)

        kept += 1

    print(f"[process_carry] scanned={len(pair_names)} kept={kept} dropped={dropped} skipped={skipped}")
    print(f"[process_carry] output motion dir: {out_motion}")
    if out_geometry is not None:
        print(f"[process_carry] output geometry dir: {out_geometry}")
    if out_object is not None:
        print(f"[process_carry] output object dir: {out_object}")


if __name__ == "__main__":
    main()
PY
