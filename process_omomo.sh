#!/usr/bin/env bash
set -euo pipefail

# Filter OMOMO retargeted sequences by stable hand-object contact and object init position.
#
# Keep only sequences that:
# 1) have stable contact with the object from both left/right hand groups;
# 2) look like "holding" (dual-hand overlap + object between hands);
# 3) start with object position close to [0, 0, 0].
#
# Usage:
#   MOTION_DIR=/ABS/PATH/to/motions \
#   GEOMETRY_DIR=/ABS/PATH/to/geometry \
#   OBJECT_URDF_DIR=/ABS/PATH/to/object_urdf_or_dir \
#   ./process_omomo.sh
#
# Optional overrides:
#   ROBOT=g1_29dof
#   LEFT_LINK=left_wrist_yaw_link RIGHT_LINK=right_wrist_yaw_link
#   OUTPUT_MOTION_DIR=/ABS/PATH OUTPUT_GEOMETRY_DIR=/ABS/PATH OUTPUT_OBJECT_DIR=/ABS/PATH
#   OUTPUT_SUFFIX=   # default empty: keep original file names
#   CONTACT_THRESH=0.25
#   MIN_STABLE_RATIO=0.20
#   MIN_CONSEC_FRAMES=12
#   MIN_DUAL_RATIO=0.15
#   MIN_BETWEEN_RATIO=0.15
#   SEGMENT_DIST_THRESH=0.18
#   OBJ_INIT_TOL=1e-4
#   REQUIRE_BOTH_HANDS=true

MOTION_DIR=${MOTION_DIR:-"/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/omomo"}
GEOMETRY_DIR=${GEOMETRY_DIR:-""}
OBJECT_URDF_DIR=${OBJECT_URDF_DIR:-"/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/models/largebox/largebox.urdf"}
ROBOT=${ROBOT:-"g1_29dof"}
LEFT_LINK=${LEFT_LINK:-""}
RIGHT_LINK=${RIGHT_LINK:-""}
OUTPUT_MOTION_DIR=${OUTPUT_MOTION_DIR:-""}
OUTPUT_GEOMETRY_DIR=${OUTPUT_GEOMETRY_DIR:-""}
OUTPUT_OBJECT_DIR=${OUTPUT_OBJECT_DIR:-""}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-""}
CONTACT_THRESH=${CONTACT_THRESH:-"0.25"}
MIN_STABLE_RATIO=${MIN_STABLE_RATIO:-"0.20"}
MIN_CONSEC_FRAMES=${MIN_CONSEC_FRAMES:-"12"}
MIN_DUAL_RATIO=${MIN_DUAL_RATIO:-"0.15"}
MIN_BETWEEN_RATIO=${MIN_BETWEEN_RATIO:-"0.15"}
SEGMENT_DIST_THRESH=${SEGMENT_DIST_THRESH:-"0.18"}
OBJ_INIT_TOL=${OBJ_INIT_TOL:-"1e-4"}
REQUIRE_BOTH_HANDS=${REQUIRE_BOTH_HANDS:-"true"}

export MOTION_DIR GEOMETRY_DIR OBJECT_URDF_DIR ROBOT LEFT_LINK RIGHT_LINK
export OUTPUT_MOTION_DIR OUTPUT_GEOMETRY_DIR OUTPUT_OBJECT_DIR OUTPUT_SUFFIX
export CONTACT_THRESH MIN_STABLE_RATIO MIN_CONSEC_FRAMES MIN_DUAL_RATIO MIN_BETWEEN_RATIO
export SEGMENT_DIST_THRESH OBJ_INIT_TOL REQUIRE_BOTH_HANDS

python - <<'PY'
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from holosoma.config_values import robot as robot_values
from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.path import resolve_data_file_path

try:
    import yourdfpy  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("yourdfpy is required to run process_omomo.sh") from exc


def _resolve_data_path(path: str) -> Path:
    if path.startswith("@holosoma/"):
        return Path(get_holosoma_root()) / path[len("@holosoma/") :]
    return Path(resolve_data_file_path(path))


def _decode_names(arr: np.ndarray) -> list[str]:
    out: list[str] = []
    for item in arr.tolist():
        if isinstance(item, (bytes, np.bytes_)):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def _build_joint_index_map(robot_cfg: Any) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(robot_cfg.dof_names)}


def _load_urdf(robot_cfg: Any):
    asset_root = _resolve_data_path(robot_cfg.asset.asset_root)
    urdf_path = _resolve_data_path(str(Path(asset_root) / robot_cfg.asset.urdf_file))
    return yourdfpy.URDF.load(str(urdf_path), load_meshes=False, build_scene_graph=True)


def _object_positions_from_npz(data: dict[str, np.ndarray], qpos: np.ndarray | None, joint_count: int) -> np.ndarray | None:
    if "object_pos_w" in data:
        obj = np.asarray(data["object_pos_w"], dtype=np.float32)
        if obj.ndim == 2 and obj.shape[1] == 3:
            return obj
    if qpos is None:
        return None
    if qpos.shape[1] >= 7 + joint_count + 7:
        # last freejoint convention: [x, y, z, qw, qx, qy, qz]
        return qpos[:, -7:-4]
    return None


def _hand_positions_from_qpos(
    qpos: np.ndarray,
    robot_cfg: Any,
    urdf: Any,
    links: list[str],
) -> dict[str, np.ndarray]:
    joint_count = len(robot_cfg.dof_names)
    if qpos.shape[1] < 7 + joint_count:
        raise ValueError(f"qpos has insufficient dims: {qpos.shape[1]} < {7 + joint_count}")

    urdf_act = urdf.actuated_joint_names
    name_to_qpos_idx = _build_joint_index_map(robot_cfg)
    missing = [name for name in urdf_act if name not in name_to_qpos_idx]
    if missing:
        raise ValueError(f"URDF actuated joints not found in robot config: {missing}")
    urdf_order_idx = [name_to_qpos_idx[name] for name in urdf_act]

    out = {name: [] for name in links}
    for frame in qpos:
        root_pos = frame[:3]
        root_quat_wxyz = frame[3:7]
        root_rot = Rotation.from_quat(
            [root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]]
        )
        joint_vals = frame[7 : 7 + joint_count]
        urdf.update_cfg(joint_vals[urdf_order_idx])
        for name in links:
            t = urdf.get_transform(frame_to=name, frame_from="world")
            local = t[:3, 3]
            out[name].append(root_pos + root_rot.apply(local))

    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def _longest_true_run(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in mask.tolist():
        if bool(v):
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _stable_contact_mask(dist: np.ndarray, thresh: float) -> np.ndarray:
    return dist <= thresh


def _between_hands_mask(
    left: np.ndarray,
    right: np.ndarray,
    obj: np.ndarray,
    segment_dist_thresh: float,
) -> np.ndarray:
    n = min(left.shape[0], right.shape[0], obj.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=bool)
    left = left[:n]
    right = right[:n]
    obj = obj[:n]
    vec = right - left
    denom = np.sum(vec * vec, axis=1)
    valid = denom > 1e-8
    t = np.zeros((n,), dtype=np.float32)
    t[valid] = np.sum((obj[valid] - left[valid]) * vec[valid], axis=1) / denom[valid]
    proj = left + vec * t[:, None]
    dist = np.linalg.norm(obj - proj, axis=1)
    return valid & (t >= 0.0) & (t <= 1.0) & (dist <= segment_dist_thresh)


def _normalize_bool_env(v: str) -> bool:
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    import os

    motion_dir = _resolve_data_path(
        os.environ.get(
            "MOTION_DIR",
            "/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/omomo",
        )
    )
    geometry_raw = os.environ.get("GEOMETRY_DIR", "").strip()
    object_raw = os.environ.get(
        "OBJECT_URDF_DIR",
        "/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/models/largebox/largebox.urdf",
    ).strip()
    robot_name = os.environ.get("ROBOT", "g1_29dof").strip()
    left_override = os.environ.get("LEFT_LINK", "").strip()
    right_override = os.environ.get("RIGHT_LINK", "").strip()
    out_motion_raw = os.environ.get("OUTPUT_MOTION_DIR", "").strip()
    out_geometry_raw = os.environ.get("OUTPUT_GEOMETRY_DIR", "").strip()
    out_object_raw = os.environ.get("OUTPUT_OBJECT_DIR", "").strip()
    suffix = os.environ.get("OUTPUT_SUFFIX", "")
    contact_thresh = float(os.environ.get("CONTACT_THRESH", "0.25"))
    min_stable_ratio = float(os.environ.get("MIN_STABLE_RATIO", "0.20"))
    min_consec_frames = int(os.environ.get("MIN_CONSEC_FRAMES", "12"))
    min_dual_ratio = float(os.environ.get("MIN_DUAL_RATIO", "0.15"))
    min_between_ratio = float(os.environ.get("MIN_BETWEEN_RATIO", "0.15"))
    segment_dist_thresh = float(os.environ.get("SEGMENT_DIST_THRESH", "0.18"))
    obj_init_tol = float(os.environ.get("OBJ_INIT_TOL", "1e-4"))
    require_both_hands = _normalize_bool_env(os.environ.get("REQUIRE_BOTH_HANDS", "true"))

    geometry_dir = _resolve_data_path(geometry_raw) if geometry_raw else None
    out_motion = _resolve_data_path(out_motion_raw) if out_motion_raw else None
    out_geometry = _resolve_data_path(out_geometry_raw) if out_geometry_raw else None
    out_object = _resolve_data_path(out_object_raw) if out_object_raw else None

    if not motion_dir.is_dir():
        raise FileNotFoundError(f"Motion dir not found: {motion_dir}")

    motion_paths = sorted(list(motion_dir.glob("*.npz")) + list(motion_dir.glob("*.NPZ")))
    if not motion_paths:
        raise FileNotFoundError(f"No .npz files in {motion_dir}")
    motion_map = {p.stem: p for p in motion_paths}
    pair_names = sorted(motion_map)

    geom_map: dict[str, Path] = {}
    if geometry_dir is not None and geometry_dir.is_dir():
        geom_paths = sorted(list(geometry_dir.glob("*.obj")) + list(geometry_dir.glob("*.OBJ")))
        geom_map = {p.stem: p for p in geom_paths}
        pair_names = sorted(set(pair_names) & set(geom_map))

    object_dir: Path | None = None
    object_urdf_path: Path | None = None
    obj_map: dict[str, Path] = {}
    if object_raw:
        obj_path = _resolve_data_path(object_raw)
        if obj_path.is_file():
            object_urdf_path = obj_path
        elif obj_path.is_dir():
            object_dir = obj_path

    if object_dir is not None and object_dir.is_dir():
        obj_paths = sorted(list(object_dir.glob("*.urdf")) + list(object_dir.glob("*.URDF")))
        obj_map = {p.stem: p for p in obj_paths}
        pair_names = sorted(set(pair_names) & set(obj_map))

    if not pair_names:
        raise RuntimeError("No matching motion/geometry/object pairs found.")

    if out_motion is None:
        out_motion = motion_dir.parent / f"{motion_dir.name}_carry"
    if geometry_dir is not None and out_geometry is None:
        out_geometry = geometry_dir.parent / f"{geometry_dir.name}_carry"
    if (object_dir is not None or object_urdf_path is not None) and out_object is None:
        if object_dir is not None:
            out_object = object_dir.parent / f"{object_dir.name}_carry"
        else:
            out_object = object_urdf_path.parent / f"{object_urdf_path.stem}_carry"

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

    # Candidate hand-related links used to evaluate contact stability.
    left_candidates = [
        "left_rubber_hand",
        "left_hand_link",
        "left_hand_palm_link",
        "left_wrist_yaw_link",
        "left_wrist_pitch_link",
        "left_wrist_roll_link",
    ]
    right_candidates = [
        "right_rubber_hand",
        "right_hand_link",
        "right_hand_palm_link",
        "right_wrist_yaw_link",
        "right_wrist_pitch_link",
        "right_wrist_roll_link",
    ]

    urdf_links = set(urdf.link_map.keys())
    left_eval = [n for n in left_candidates if n in urdf_links]
    right_eval = [n for n in right_candidates if n in urdf_links]
    if left_override:
        if left_override not in urdf_links:
            raise RuntimeError(f"LEFT_LINK not found in URDF: {left_override}")
        left_eval = [left_override] + [x for x in left_eval if x != left_override]
    if right_override:
        if right_override not in urdf_links:
            raise RuntimeError(f"RIGHT_LINK not found in URDF: {right_override}")
        right_eval = [right_override] + [x for x in right_eval if x != right_override]

    if not left_eval or not right_eval:
        raise RuntimeError(
            "Could not resolve hand link candidates from URDF. "
            "Try setting LEFT_LINK and RIGHT_LINK explicitly."
        )

    left_repr = left_eval[0]
    right_repr = right_eval[0]

    counters = {
        "kept": 0,
        "dropped_contact": 0,
        "dropped_init_not_zero": 0,
        "skipped_no_object": 0,
        "skipped_no_hands": 0,
    }
    kept_contact_summary: list[tuple[str, list[str], list[str], float, float]] = []

    for name in pair_names:
        motion_path = motion_map[name]
        with np.load(motion_path, allow_pickle=True) as data:
            payload = {k: data[k] for k in data.files}

        qpos = payload.get("qpos")
        if qpos is not None:
            qpos = np.asarray(qpos, dtype=np.float32)

        obj_pos = _object_positions_from_npz(payload, qpos, joint_count)
        if obj_pos is None:
            counters["skipped_no_object"] += 1
            continue

        # Strictly remove sequences whose object start position is not ~0.
        if float(np.linalg.norm(obj_pos[0])) > obj_init_tol:
            counters["dropped_init_not_zero"] += 1
            continue

        link_pos: dict[str, np.ndarray] = {}
        if "body_pos_w" in payload and "body_names" in payload:
            body_pos_w = np.asarray(payload["body_pos_w"], dtype=np.float32)
            body_names = _decode_names(np.asarray(payload["body_names"]))
            for ln in (left_eval + right_eval):
                if ln in body_names:
                    link_pos[ln] = body_pos_w[:, body_names.index(ln), :]

        missing_links = [ln for ln in (left_eval + right_eval) if ln not in link_pos]
        if missing_links:
            if qpos is None:
                counters["skipped_no_hands"] += 1
                continue
            # Fallback to FK for all missing hand links.
            fk_links = sorted(set(missing_links))
            fk_pos = _hand_positions_from_qpos(qpos, robot_cfg, urdf, fk_links)
            link_pos.update(fk_pos)

        if left_repr not in link_pos or right_repr not in link_pos:
            counters["skipped_no_hands"] += 1
            continue

        n = min(obj_pos.shape[0], *(arr.shape[0] for arr in link_pos.values()))
        if n <= 0:
            counters["dropped_contact"] += 1
            continue
        obj = obj_pos[:n]

        stable_left: list[str] = []
        stable_right: list[str] = []
        left_any = np.zeros((n,), dtype=bool)
        right_any = np.zeros((n,), dtype=bool)

        for ln in left_eval:
            if ln not in link_pos:
                continue
            d = np.linalg.norm(link_pos[ln][:n] - obj, axis=1)
            c = _stable_contact_mask(d, contact_thresh)
            ratio = float(np.mean(c))
            run = _longest_true_run(c)
            if ratio >= min_stable_ratio and run >= min_consec_frames:
                stable_left.append(ln)
            left_any |= c

        for rn in right_eval:
            if rn not in link_pos:
                continue
            d = np.linalg.norm(link_pos[rn][:n] - obj, axis=1)
            c = _stable_contact_mask(d, contact_thresh)
            ratio = float(np.mean(c))
            run = _longest_true_run(c)
            if ratio >= min_stable_ratio and run >= min_consec_frames:
                stable_right.append(rn)
            right_any |= c

        dual_contact = left_any & right_any
        dual_ratio = float(np.mean(dual_contact))
        dual_run = _longest_true_run(dual_contact)

        between = _between_hands_mask(
            left=link_pos[left_repr][:n],
            right=link_pos[right_repr][:n],
            obj=obj,
            segment_dist_thresh=segment_dist_thresh,
        )
        between_ratio = float(np.mean(between))
        between_run = _longest_true_run(between)

        if require_both_hands:
            hands_ok = bool(stable_left) and bool(stable_right)
        else:
            hands_ok = bool(stable_left) or bool(stable_right)

        hold_ok = (
            hands_ok
            and dual_ratio >= min_dual_ratio
            and dual_run >= max(1, min_consec_frames // 2)
            and between_ratio >= min_between_ratio
            and between_run >= max(1, min_consec_frames // 2)
        )
        if not hold_ok:
            counters["dropped_contact"] += 1
            continue

        out_motion_path = out_motion / f"{motion_path.stem}{suffix}{motion_path.suffix}"
        shutil.copy2(motion_path, out_motion_path)

        if out_geometry is not None and name in geom_map:
            geom_path = geom_map[name]
            out_geom_path = out_geometry / f"{geom_path.stem}{suffix}{geom_path.suffix}"
            shutil.copy2(geom_path, out_geom_path)

        if out_object is not None:
            if object_urdf_path is not None:
                out_obj_path = out_object / f"{object_urdf_path.stem}{suffix}{object_urdf_path.suffix}"
                if not out_obj_path.exists():
                    shutil.copy2(object_urdf_path, out_obj_path)
            elif name in obj_map:
                obj_path = obj_map[name]
                out_obj_path = out_object / f"{obj_path.stem}{suffix}{obj_path.suffix}"
                shutil.copy2(obj_path, out_obj_path)

        counters["kept"] += 1
        kept_contact_summary.append((name, stable_left, stable_right, dual_ratio, between_ratio))

    scanned = len(pair_names)
    print(
        "[process_omomo] scanned={scanned} kept={kept} dropped_contact={dropped_contact} "
        "dropped_init_not_zero={dropped_init_not_zero} skipped_no_object={skipped_no_object} "
        "skipped_no_hands={skipped_no_hands}".format(scanned=scanned, **counters)
    )
    print(f"[process_omomo] output motion dir: {out_motion}")
    if out_geometry is not None:
        print(f"[process_omomo] output geometry dir: {out_geometry}")
    if out_object is not None:
        print(f"[process_omomo] output object dir: {out_object}")

    # Show stable contact parts for kept sequences.
    for name, left_links, right_links, dual_ratio, between_ratio in kept_contact_summary:
        print(
            f"[process_omomo] keep {name}: "
            f"left={left_links} right={right_links} "
            f"dual_ratio={dual_ratio:.3f} between_ratio={between_ratio:.3f}"
        )


if __name__ == "__main__":
    main()
PY
