#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_MOTION_FILE="$ROOT_DIR/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
DEFAULT_MODEL_PATH="/data/logs_new/boxer/20260316_200048-g1_29dof_wbt_w_object_extend_20260316_200027_s01_scale_1p0-g1_29dof_wbt_w_object_extend_20260316_200027/model_23500.onnx"

MOTION_FILE="${1:-$DEFAULT_MOTION_FILE}"
MODEL_PATH="${2:-$DEFAULT_MODEL_PATH}"
RUN_SECONDS="${RUN_SECONDS:-8}"
TRACE_PATH="${TRACE_PATH:-$ROOT_DIR/logs/live_debug/validated_wobj_$(date +%s).jsonl}"

export SIM_USE_ZMQ_LOWCMD="${SIM_USE_ZMQ_LOWCMD:-1}"
export INFERENCE_CONFIG="${INFERENCE_CONFIG:-g1-29dof-w-obj}"
export PREFER_SIM_REF_FROM_SIM_STATE="${PREFER_SIM_REF_FROM_SIM_STATE:-1}"
export USE_ROOT_REFERENCE_AT_CLIP_START="${USE_ROOT_REFERENCE_AT_CLIP_START:-1}"
export SIM_HOLD_INITIAL_POSE_UNTIL_FIRST_COMMAND="${SIM_HOLD_INITIAL_POSE_UNTIL_FIRST_COMMAND:-0}"
export SIM_FREEZE_UNTIL_FIRST_COMMAND="${SIM_FREEZE_UNTIL_FIRST_COMMAND:-1}"
export AUTO_START_STIFF_HOLD_SEC="${AUTO_START_STIFF_HOLD_SEC:-1.0}"
export AUTO_START_STIFF_MAX_WAIT_SEC="${AUTO_START_STIFF_MAX_WAIT_SEC:-1.0}"
export HOLOSOMA_SPLIT_SIM_STATE_TRACE_PATH="$TRACE_PATH"
export HOLOSOMA_SIM_STATE_INCLUDE_OBJECT_CONTACT_DETAILS="${HOLOSOMA_SIM_STATE_INCLUDE_OBJECT_CONTACT_DETAILS:-1}"
export HOLOSOMA_SIM_STATE_INCLUDE_KEY_BODY_STATES="${HOLOSOMA_SIM_STATE_INCLUDE_KEY_BODY_STATES:-1}"
export RUN_SECONDS

echo "[run_wobj_tracking_validated] motion: $MOTION_FILE"
echo "[run_wobj_tracking_validated] model:  $MODEL_PATH"
echo "[run_wobj_tracking_validated] trace:  $TRACE_PATH"
echo "[run_wobj_tracking_validated] seconds:$RUN_SECONDS"

"$ROOT_DIR/sim2sim_box_split_tracking.sh" "$MOTION_FILE" "$MODEL_PATH"

python3 - <<'PY' "$TRACE_PATH"
import json
import sys
from pathlib import Path

import numpy as np

trace_path = Path(sys.argv[1]).resolve()
if not trace_path.is_file():
    raise SystemExit(f"trace file not found: {trace_path}")

rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    raise SystemExit(f"trace file is empty: {trace_path}")

actor_keys = list(rows[0].get("actors", {}).keys())
if not actor_keys:
    raise SystemExit("trace does not contain any simulator actors")
object_name = actor_keys[0]

robot_xy = np.asarray([np.asarray(row["robot_root_state"], dtype=np.float64)[:2] for row in rows], dtype=np.float64)
object_xy = np.asarray(
    [np.asarray(row["actors"][object_name], dtype=np.float64)[:2] for row in rows],
    dtype=np.float64,
)
robot_z = np.asarray([float(np.asarray(row["robot_root_state"], dtype=np.float64)[2]) for row in rows], dtype=np.float64)
object_robot_contacts = np.asarray(
    [float(row.get("object_robot_contact_count", 0.0)) for row in rows],
    dtype=np.float64,
)

summary = {
    "trace_path": str(trace_path),
    "rows": int(len(rows)),
    "object_name": object_name,
    "robot_disp_m": float(np.linalg.norm(robot_xy[-1] - robot_xy[0])),
    "object_disp_m": float(np.linalg.norm(object_xy[-1] - object_xy[0])),
    "end_gap_m": float(np.linalg.norm(robot_xy[-1] - object_xy[-1])),
    "robot_z_min_m": float(robot_z.min()),
    "max_object_robot_contacts": int(object_robot_contacts.max()),
    "first_contact_idx": None,
    "first_contact_bodies": [],
}

first_contact_idx = next((idx for idx, value in enumerate(object_robot_contacts) if value > 0.0), None)
if first_contact_idx is not None:
    summary["first_contact_idx"] = int(first_contact_idx)
    summary["first_contact_bodies"] = rows[first_contact_idx].get("object_robot_contact_bodies", [])

print("[run_wobj_tracking_validated] summary:")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
