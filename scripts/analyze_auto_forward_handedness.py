#!/usr/bin/env python3
"""Analyze auto-forward yaw/drift against left/right wrist object contact.

Input is the JSONL produced by scripts/run_auto_forward_command_sweep.sh with
VISER_AUTO_FORWARD_AFTER_LIFT enabled. Positive yaw is a left turn; positive
lateral displacement is leftward drift in the initial root-heading frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def wrap_pi(value: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(value) + math.pi) % (2.0 * math.pi) - math.pi


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def list_jsonl_inputs(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SystemExit(f"[ERROR] Input path does not exist: {path}")

    preferred = sorted(path.glob("infer_cmd_*.jsonl"))
    if preferred:
        return preferred
    matches = sorted(path.rglob("*.jsonl"))
    if matches:
        return matches
    raise SystemExit(f"[ERROR] No JSONL files found under {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"[ERROR] Invalid JSON in {path}:{line_no}: {exc}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def as_vec3(value: Any) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return np.full(3, np.nan, dtype=np.float64)
    if arr.size < 3:
        return np.full(3, np.nan, dtype=np.float64)
    return arr[:3]


def command_norm(value: Any) -> float:
    if not isinstance(value, list | tuple):
        return 0.0
    vals = np.asarray([finite_float(v) for v in value], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    return float(np.linalg.norm(vals))


def heading_frame_delta(start_pos: np.ndarray, end_pos: np.ndarray, yaw0: float) -> tuple[float, float]:
    if not np.all(np.isfinite(start_pos[:2])) or not np.all(np.isfinite(end_pos[:2])) or not math.isfinite(yaw0):
        return math.nan, math.nan
    delta = end_pos[:2] - start_pos[:2]
    c = math.cos(yaw0)
    s = math.sin(yaw0)
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


def mean_finite(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return math.nan if arr.size == 0 else float(arr.mean())


def max_finite(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return math.nan if arr.size == 0 else float(arr.max())


def field_series(rows: list[dict[str, Any]], *field_names: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        selected = math.nan
        for field_name in field_names:
            candidate = finite_float(row.get(field_name))
            if math.isfinite(candidate):
                selected = candidate
                break
        values.append(selected)
    return np.asarray(values, dtype=np.float64)


def analyze_file(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    active_rows = [row for row in rows if str(row.get("state", "")).lower() == "active"]
    used_rows = active_rows if active_rows else rows
    if len(used_rows) < 2:
        return {
            "jsonl_path": str(path),
            "status": "insufficient_frames",
            "num_rows": len(rows),
            "num_active_rows": len(active_rows),
        }

    first_active_index = None
    for idx, row in enumerate(rows):
        if str(row.get("state", "")).lower() == "active":
            first_active_index = idx
            break
    pre_active_rows = rows[: first_active_index if first_active_index is not None else len(rows)]
    pre_active_applied_nonzero_count = sum(command_norm(row.get("applied_command")) > 1.0e-6 for row in pre_active_rows)
    pre_active_effective_nonzero_count = sum(command_norm(row.get("effective_command")) > 1.0e-6 for row in pre_active_rows)
    trigger_rel_z_delta = math.nan
    if first_active_index is not None:
        trigger_rel_z_delta = finite_float(rows[first_active_index].get("object_rel_z_delta"))

    root_pos = np.stack([as_vec3(row.get("root_pos_w")) for row in used_rows], axis=0)
    object_pos = np.stack([as_vec3(row.get("object_pos_w")) for row in used_rows], axis=0)
    root_yaw = np.asarray([finite_float(row.get("root_yaw")) for row in used_rows], dtype=np.float64)
    root_yaw_rate = np.asarray([finite_float(row.get("root_yaw_rate_est")) for row in used_rows], dtype=np.float64)
    left_wrist_yaw_force = field_series(used_rows, "left_wrist_yaw_object_force", "left_wrist_object_force")
    right_wrist_yaw_force = field_series(used_rows, "right_wrist_yaw_object_force", "right_wrist_object_force")
    left_rubber_hand_force = field_series(used_rows, "left_rubber_hand_object_force")
    right_rubber_hand_force = field_series(used_rows, "right_rubber_hand_object_force")
    left_force = field_series(used_rows, "left_hand_object_force", "left_wrist_object_force")
    right_force = field_series(used_rows, "right_hand_object_force", "right_wrist_object_force")
    right_minus_left = field_series(
        used_rows,
        "right_minus_left_hand_object_force",
        "right_minus_left_wrist_object_force",
    )
    right_share = field_series(
        used_rows,
        "right_hand_object_force_share",
        "right_wrist_object_force_share",
    )

    valid_yaw = np.flatnonzero(np.isfinite(root_yaw))
    valid_pos = np.flatnonzero(np.all(np.isfinite(root_pos[:, :2]), axis=1))
    if valid_yaw.size < 2 or valid_pos.size < 2:
        return {
            "jsonl_path": str(path),
            "status": "missing_pose",
            "num_rows": len(rows),
            "num_active_rows": len(active_rows),
        }

    first = int(max(valid_yaw[0], valid_pos[0]))
    last = int(min(valid_yaw[-1], valid_pos[-1]))
    if last <= first:
        first = int(valid_yaw[0])
        last = int(valid_yaw[-1])

    yaw_delta = float(wrap_pi(root_yaw[last] - root_yaw[first]))
    root_forward, root_left = heading_frame_delta(root_pos[first], root_pos[last], float(root_yaw[first]))
    object_forward, object_left = heading_frame_delta(object_pos[first], object_pos[last], float(root_yaw[first]))
    force_sum = left_force + right_force
    force_score = np.divide(
        right_force - left_force,
        force_sum,
        out=np.full_like(force_sum, np.nan, dtype=np.float64),
        where=np.isfinite(force_sum) & (force_sum > 1.0e-9),
    )

    command = None
    for row in used_rows:
        candidate = row.get("effective_command") or row.get("applied_command") or row.get("configured_command")
        if isinstance(candidate, list) and candidate:
            command = [finite_float(v) for v in candidate]
            break

    errors = sorted({str(row.get("wrist_object_contact_error")) for row in rows if row.get("wrist_object_contact_error")})
    sources = sorted({str(row.get("wrist_object_contact_source")) for row in rows if row.get("wrist_object_contact_source")})

    return {
        "jsonl_path": str(path),
        "status": "ok",
        "command": command,
        "num_rows": len(rows),
        "num_active_rows": len(active_rows),
        "num_pre_active_rows": len(pre_active_rows),
        "pre_active_applied_nonzero_count": pre_active_applied_nonzero_count,
        "pre_active_effective_nonzero_count": pre_active_effective_nonzero_count,
        "trigger_object_rel_z_delta_m": trigger_rel_z_delta,
        "used_state": "active" if active_rows else "all",
        "contact_sources": ",".join(sources),
        "contact_errors": "; ".join(errors),
        "yaw_delta_deg": math.degrees(yaw_delta),
        "yaw_left_positive": True,
        "root_yaw_rate_mean_deg_s": math.degrees(mean_finite(root_yaw_rate)),
        "root_yaw_rate_abs_mean_deg_s": math.degrees(mean_finite(np.abs(root_yaw_rate))),
        "root_forward_delta_m": root_forward,
        "root_left_delta_m": root_left,
        "object_forward_delta_m": object_forward,
        "object_left_delta_m": object_left,
        "left_hand_force_mean": mean_finite(left_force),
        "right_hand_force_mean": mean_finite(right_force),
        "left_hand_force_max": max_finite(left_force),
        "right_hand_force_max": max_finite(right_force),
        "left_wrist_yaw_force_mean": mean_finite(left_wrist_yaw_force),
        "right_wrist_yaw_force_mean": mean_finite(right_wrist_yaw_force),
        "left_wrist_yaw_force_max": max_finite(left_wrist_yaw_force),
        "right_wrist_yaw_force_max": max_finite(right_wrist_yaw_force),
        "left_rubber_hand_force_mean": mean_finite(left_rubber_hand_force),
        "right_rubber_hand_force_mean": mean_finite(right_rubber_hand_force),
        "left_rubber_hand_force_max": max_finite(left_rubber_hand_force),
        "right_rubber_hand_force_max": max_finite(right_rubber_hand_force),
        "left_wrist_force_mean": mean_finite(left_wrist_yaw_force),
        "right_wrist_force_mean": mean_finite(right_wrist_yaw_force),
        "left_wrist_force_max": max_finite(left_wrist_yaw_force),
        "right_wrist_force_max": max_finite(right_wrist_yaw_force),
        "right_minus_left_force_mean": mean_finite(right_minus_left),
        "right_force_share_mean": mean_finite(right_share),
        "force_score_right_minus_left_mean": mean_finite(force_score),
        "corr_right_minus_left_force_vs_yaw_rate": pearson(right_minus_left.tolist(), root_yaw_rate.tolist()),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    out: dict[str, Any] = {
        "num_runs": len(rows),
        "num_ok_runs": len(ok_rows),
        "yaw_left_positive": True,
    }
    metric_keys = [
        "yaw_delta_deg",
        "root_yaw_rate_mean_deg_s",
        "root_forward_delta_m",
        "root_left_delta_m",
        "left_hand_force_mean",
        "right_hand_force_mean",
        "left_rubber_hand_force_mean",
        "right_rubber_hand_force_mean",
        "left_wrist_yaw_force_mean",
        "right_wrist_yaw_force_mean",
        "left_wrist_force_mean",
        "right_wrist_force_mean",
        "right_minus_left_force_mean",
        "right_force_share_mean",
        "force_score_right_minus_left_mean",
        "corr_right_minus_left_force_vs_yaw_rate",
    ]
    for key in metric_keys:
        out[f"{key}_mean"] = mean_finite([finite_float(row.get(key)) for row in ok_rows])
    out["corr_run_force_score_vs_yaw_delta"] = pearson(
        [finite_float(row.get("force_score_right_minus_left_mean")) for row in ok_rows],
        [finite_float(row.get("yaw_delta_deg")) for row in ok_rows],
    )
    out["corr_run_force_score_vs_root_left_delta"] = pearson(
        [finite_float(row.get("force_score_right_minus_left_mean")) for row in ok_rows],
        [finite_float(row.get("root_left_delta_m")) for row in ok_rows],
    )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Auto-forward JSONL file or sweep directory")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for summary JSON/CSV")
    args = parser.parse_args()

    inputs = list_jsonl_inputs(args.input)
    rows = [analyze_file(path) for path in inputs]
    summary = aggregate(rows)
    payload = {
        "note": "Positive yaw_delta_deg means left turn; positive root_left_delta_m means left drift.",
        "summary": summary,
        "runs": rows,
    }

    if args.out_dir is not None:
        out_dir = args.out_dir.expanduser().resolve()
    else:
        out_dir = (args.input.expanduser().resolve() if args.input.is_dir() else args.input.expanduser().resolve().parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "auto_forward_handedness_summary.json"
    csv_path = out_dir / "auto_forward_handedness_per_run.csv"
    safe_payload = json_safe(payload)
    summary_path.write_text(json.dumps(safe_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)

    print(f"[OK] analyzed {summary['num_ok_runs']}/{summary['num_runs']} run(s)")
    print(f"[OK] wrote {summary_path}")
    print(f"[OK] wrote {csv_path}")
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
