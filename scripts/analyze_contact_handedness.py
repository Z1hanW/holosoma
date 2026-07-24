#!/usr/bin/env python3
"""Analyze whether rollout yaw/drift correlates with left/right wrist contact.

This is an offline reader for outputs produced by infer_teacher_as_contacts.sh /
export_teacher_box_contacts.py. It does not launch Isaac Sim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


LEFT_LABEL = "left_wrist"
RIGHT_LABEL = "right_wrist"


def wrap_pi(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    x = quat[..., 0]
    y = quat[..., 1]
    z = quat[..., 2]
    w = quat[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def resolve_clips_dir(root: Path) -> Path:
    root = root.expanduser().resolve()
    if root.name == "clips" and root.is_dir():
        return root
    if (root / "clips").is_dir():
        return (root / "clips").resolve()

    candidates = sorted(root.glob("contact_export*/clips"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    if len(candidates) > 1:
        names = "\n  ".join(str(p) for p in candidates)
        raise SystemExit(f"[ERROR] Multiple contact export roots found; pass one explicitly:\n  {names}")
    raise SystemExit(f"[ERROR] Could not find clips directory under {root}")


def read_csv_by_key(path: Path, key_field: str) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return {str(row.get(key_field, "")): row for row in reader if row.get(key_field)}


def read_interval(clip_dir: Path, label: str) -> tuple[int, int]:
    npy_path = clip_dir / f"{label}_contact_interval_steps.npy"
    if npy_path.is_file():
        arr = np.asarray(np.load(npy_path), dtype=np.int64).reshape(-1)
        if arr.size >= 2:
            return int(arr[0]), int(arr[1])

    json_path = clip_dir / "contact_intervals.json"
    if json_path.is_file():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        raw = payload.get(label, [-1, -1])
        if isinstance(raw, list | tuple) and len(raw) >= 2:
            return int(raw[0]), int(raw[1])
    return -1, -1


def valid_interval(interval: tuple[int, int]) -> bool:
    start, end = interval
    return start >= 0 and end > start


def has_reward_target(clip_dir: Path, label: str, interval: tuple[int, int]) -> bool:
    points_path = clip_dir / f"{label}_contact_points.npy"
    if not points_path.is_file() or not valid_interval(interval):
        return False
    try:
        points = np.asarray(np.load(points_path), dtype=np.float32).reshape(-1, 3)
    except Exception:
        return False
    return int(points.shape[0]) >= 1


def side_group(has_left: bool, has_right: bool) -> str:
    if has_left and has_right:
        return "both"
    if has_left:
        return "left_only"
    if has_right:
        return "right_only"
    return "neither"


def interval_mask(valid: np.ndarray, interval: tuple[int, int]) -> np.ndarray:
    if not valid_interval(interval):
        return np.zeros_like(valid, dtype=np.bool_)
    steps = np.arange(valid.shape[0])
    start, end = interval
    return valid & (steps >= start) & (steps < end)


def mean_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return math.nan
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(finite.mean())


def final_heading_frame_delta(pos: np.ndarray, yaw0: float, valid_indices: np.ndarray) -> tuple[float, float]:
    if pos.ndim != 2 or pos.shape[0] == 0 or valid_indices.size == 0:
        return math.nan, math.nan
    start = pos[valid_indices[0], :2].astype(np.float64)
    end = pos[valid_indices[-1], :2].astype(np.float64)
    delta = end - start
    c = math.cos(float(yaw0))
    s = math.sin(float(yaw0))
    forward = c * delta[0] + s * delta[1]
    left = -s * delta[0] + c * delta[1]
    return float(forward), float(left)


def pearson(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or float(np.std(x)) < 1.0e-12 or float(np.std(y)) < 1.0e-12:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def analyze_clip(clip_dir: Path) -> dict[str, Any]:
    ref_path = clip_dir / "teacher_rollout_reference.npz"
    if not ref_path.is_file():
        raise FileNotFoundError(ref_path)
    ref = np.load(ref_path, allow_pickle=False)
    valid = np.asarray(ref["valid_steps"], dtype=np.bool_).reshape(-1)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        valid_indices = np.asarray([0], dtype=np.int64)

    root_yaw = yaw_from_xyzw(ref["root_quat_w"])
    target_root_yaw = yaw_from_xyzw(ref["target_root_quat_w"]) if "target_root_quat_w" in ref.files else root_yaw
    yaw_error = wrap_pi(root_yaw - target_root_yaw)

    root_yaw_valid = root_yaw[valid_indices]
    target_yaw_valid = target_root_yaw[valid_indices]
    yaw_error_valid = yaw_error[valid_indices]
    root_delta = float(wrap_pi(root_yaw_valid[-1] - root_yaw_valid[0]))
    target_delta = float(wrap_pi(target_yaw_valid[-1] - target_yaw_valid[0]))
    yaw_error_delta = float(wrap_pi(yaw_error_valid[-1] - yaw_error_valid[0]))

    left_interval = read_interval(clip_dir, LEFT_LABEL)
    right_interval = read_interval(clip_dir, RIGHT_LABEL)
    left_mask = interval_mask(valid, left_interval)
    right_mask = interval_mask(valid, right_interval)
    either_mask = left_mask | right_mask

    region_stats = read_csv_by_key(clip_dir / "region_contact_stats.csv", "region")
    left_stats = region_stats.get(LEFT_LABEL, {})
    right_stats = region_stats.get(RIGHT_LABEL, {})
    left_frames = int(finite_float(left_stats.get("contact_frames", 0)))
    right_frames = int(finite_float(right_stats.get("contact_frames", 0)))
    left_avg_force = finite_float(left_stats.get("avg_force_over_contact_frames", 0.0))
    right_avg_force = finite_float(right_stats.get("avg_force_over_contact_frames", 0.0))
    left_max_force = finite_float(left_stats.get("max_force", 0.0))
    right_max_force = finite_float(right_stats.get("max_force", 0.0))

    has_left_target = has_reward_target(clip_dir, LEFT_LABEL, left_interval)
    has_right_target = has_reward_target(clip_dir, RIGHT_LABEL, right_interval)
    target_group = side_group(has_left_target, has_right_target)
    force_group = side_group(left_frames > 0, right_frames > 0)

    denom = max(left_frames + right_frames, 1)
    right_share = float(right_frames / denom)
    force_hand_score = float((right_frames - left_frames) / denom)
    target_denom = max(int(has_left_target) + int(has_right_target), 1)
    target_hand_score = float((int(has_right_target) - int(has_left_target)) / target_denom)

    root_forward, root_left = final_heading_frame_delta(ref["root_pos_local"], float(root_yaw_valid[0]), valid_indices)
    if "object_pos_local" in ref.files:
        object_forward, object_left = final_heading_frame_delta(ref["object_pos_local"], float(root_yaw_valid[0]), valid_indices)
    else:
        object_forward, object_left = math.nan, math.nan

    def deg(x: float) -> float:
        return float(math.degrees(x)) if math.isfinite(float(x)) else math.nan

    clip_id = clip_dir.name
    metadata_path = clip_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            clip_id = str(metadata.get("clip_id", clip_id))
        except Exception:
            pass

    return {
        "clip_dir": clip_dir.name,
        "clip_id": clip_id,
        "target_group": target_group,
        "force_group": force_group,
        "valid_step_count": int(valid_indices.size),
        "left_frames": left_frames,
        "right_frames": right_frames,
        "right_share": right_share,
        "force_hand_score_right_minus_left": force_hand_score,
        "target_hand_score_right_minus_left": target_hand_score,
        "has_left_target": bool(has_left_target),
        "has_right_target": bool(has_right_target),
        "left_interval_start": int(left_interval[0]),
        "left_interval_end": int(left_interval[1]),
        "right_interval_start": int(right_interval[0]),
        "right_interval_end": int(right_interval[1]),
        "left_avg_force": left_avg_force,
        "right_avg_force": right_avg_force,
        "left_max_force": left_max_force,
        "right_max_force": right_max_force,
        "root_yaw_start_deg": deg(float(root_yaw_valid[0])),
        "root_yaw_end_deg": deg(float(root_yaw_valid[-1])),
        "root_yaw_delta_deg": deg(root_delta),
        "target_root_yaw_delta_deg": deg(target_delta),
        "yaw_error_start_deg": deg(float(yaw_error_valid[0])),
        "yaw_error_end_deg": deg(float(yaw_error_valid[-1])),
        "yaw_error_delta_deg": deg(yaw_error_delta),
        "yaw_error_mean_deg": deg(mean_or_nan(yaw_error_valid)),
        "yaw_error_left_contact_mean_deg": deg(mean_or_nan(yaw_error[left_mask])),
        "yaw_error_right_contact_mean_deg": deg(mean_or_nan(yaw_error[right_mask])),
        "yaw_error_any_wrist_contact_mean_deg": deg(mean_or_nan(yaw_error[either_mask])),
        "root_forward_final_m": root_forward,
        "root_left_final_m": root_left,
        "object_forward_final_m": object_forward,
        "object_left_final_m": object_left,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_clips": len(rows),
        "positive_yaw_deg_means_left_turn": True,
        "positive_lateral_m_means_left_in_initial_root_heading_frame": True,
    }
    metric_keys = [
        "right_share",
        "force_hand_score_right_minus_left",
        "target_hand_score_right_minus_left",
        "root_yaw_delta_deg",
        "yaw_error_delta_deg",
        "yaw_error_mean_deg",
        "yaw_error_right_contact_mean_deg",
        "yaw_error_left_contact_mean_deg",
        "root_left_final_m",
        "object_left_final_m",
        "right_avg_force",
        "left_avg_force",
    ]
    for group_field in ["target_group", "force_group"]:
        for group in ["right_only", "left_only", "both", "neither"]:
            subset = [r for r in rows if r[group_field] == group]
            prefix = f"{group_field}/{group}"
            summary[f"{prefix}/count"] = len(subset)
            for key in metric_keys:
                summary[f"{prefix}/{key}_mean"] = mean_or_nan(
                    np.asarray([finite_float(r.get(key), math.nan) for r in subset])
                )

    target_hand_scores = [finite_float(r["target_hand_score_right_minus_left"], math.nan) for r in rows]
    force_hand_scores = [finite_float(r["force_hand_score_right_minus_left"], math.nan) for r in rows]
    summary["corr/target_hand_score_vs_yaw_error_delta"] = pearson(
        target_hand_scores,
        [finite_float(r["yaw_error_delta_deg"], math.nan) for r in rows],
    )
    summary["corr/target_hand_score_vs_yaw_error_mean"] = pearson(
        target_hand_scores,
        [finite_float(r["yaw_error_mean_deg"], math.nan) for r in rows],
    )
    summary["corr/target_hand_score_vs_root_left_final"] = pearson(
        target_hand_scores,
        [finite_float(r["root_left_final_m"], math.nan) for r in rows],
    )
    summary["corr/target_hand_score_vs_root_yaw_delta"] = pearson(
        target_hand_scores,
        [finite_float(r["root_yaw_delta_deg"], math.nan) for r in rows],
    )
    summary["corr/force_hand_score_vs_yaw_error_delta"] = pearson(
        force_hand_scores,
        [finite_float(r["yaw_error_delta_deg"], math.nan) for r in rows],
    )
    summary["corr/force_hand_score_vs_yaw_error_mean"] = pearson(
        force_hand_scores,
        [finite_float(r["yaw_error_mean_deg"], math.nan) for r in rows],
    )
    summary["corr/force_hand_score_vs_root_left_final"] = pearson(
        force_hand_scores,
        [finite_float(r["root_left_final_m"], math.nan) for r in rows],
    )
    summary["corr/force_hand_score_vs_root_yaw_delta"] = pearson(
        force_hand_scores,
        [finite_float(r["root_yaw_delta_deg"], math.nan) for r in rows],
    )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_root", type=Path, help="Export dir, contact export dir, or clips dir.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/contact_handedness"),
        help="Directory for handedness_summary.json and handedness_per_clip.csv.",
    )
    args = parser.parse_args()

    clips_dir = resolve_clips_dir(args.rollout_root)
    clip_dirs = sorted(p for p in clips_dir.iterdir() if (p / "teacher_rollout_reference.npz").is_file())
    if not clip_dirs:
        raise SystemExit(f"[ERROR] No teacher_rollout_reference.npz files found under {clips_dir}")

    rows = [analyze_clip(clip_dir) for clip_dir in clip_dirs]
    summary = summarize(rows)
    summary["clips_dir"] = str(clips_dir)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "handedness_per_clip.csv"
    json_path = out_dir / "handedness_summary.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[INFO] clips_dir={clips_dir}")
    print(f"[INFO] wrote {csv_path}")
    print(f"[INFO] wrote {json_path}")
    print("[SUMMARY]")
    for key in [
        "num_clips",
        "target_group/right_only/count",
        "target_group/left_only/count",
        "target_group/both/count",
        "target_group/neither/count",
        "force_group/right_only/count",
        "force_group/left_only/count",
        "force_group/both/count",
        "force_group/neither/count",
        "target_group/right_only/yaw_error_delta_deg_mean",
        "target_group/right_only/yaw_error_mean_deg_mean",
        "target_group/right_only/root_left_final_m_mean",
        "target_group/left_only/yaw_error_delta_deg_mean",
        "target_group/left_only/yaw_error_mean_deg_mean",
        "target_group/left_only/root_left_final_m_mean",
        "corr/target_hand_score_vs_yaw_error_delta",
        "corr/target_hand_score_vs_yaw_error_mean",
        "corr/target_hand_score_vs_root_left_final",
        "corr/force_hand_score_vs_yaw_error_delta",
        "corr/force_hand_score_vs_yaw_error_mean",
        "corr/force_hand_score_vs_root_left_final",
    ]:
        print(f"{key}: {summary.get(key)}")


if __name__ == "__main__":
    main()
