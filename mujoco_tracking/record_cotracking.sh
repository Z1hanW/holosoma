#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOLOSOMA="$(cd "$HERE/.." && pwd)"
FAR="$(cd "$HOLOSOMA/.." && pwd)"

CLIP="${1:-box_74}"
DATASET_NAME="carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success133_final0p5"
if [[ -e "$HOLOSOMA/data/$DATASET_NAME" ]]; then
  DATASET="$HOLOSOMA/data/$DATASET_NAME"
else
  DATASET="$FAR/data/$DATASET_NAME"
fi
MODEL="$HOLOSOMA/checkpoints/cotracking/bcleb5oi_model_52000.onnx"
MUJOCO_PY="/home/user/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python"
INFER_PY="/home/user/.holosoma_deps/miniconda3/envs/hsinference/bin/python"

if [[ "$CLIP" == "-h" || "$CLIP" == "--help" ]]; then
  echo "Usage: ./mujoco_tracking/record_cotracking.sh [clip_name]"
  echo "Example: ./mujoco_tracking/record_cotracking.sh box_74"
  exit 0
fi

MOTION="$DATASET/$CLIP.npz"
[[ -f "$MOTION" ]] || { echo "motion not found: $MOTION" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }

SECONDS_TO_RECORD="${SECONDS_TO_RECORD:-$("$MUJOCO_PY" - "$MOTION" <<'PY'
import math
import sys

import numpy as np

motion = np.load(sys.argv[1])
fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
frames = 0
for key in ("joint_pos", "body_pos_w", "object_pos_w"):
    if key in motion:
        frames = int(motion[key].shape[0])
        break
duration = max(0.0, (frames - 1) / fps)
print(int(math.ceil(duration + 8.0)))
PY
)}"

OBJECT_URDF="$("$MUJOCO_PY" - "$DATASET" "$CLIP" <<'PY'
import json
import sys
from pathlib import Path

dataset = Path(sys.argv[1])
clip = sys.argv[2]
mapping = dataset / "_clip_object_urdf_map.json"

if mapping.exists():
    clips = json.loads(mapping.read_text()).get("clips", {})
    item = clips.get(clip)
    if item:
        path = Path(item["object_urdf_path"])
        print(path if path.is_absolute() else dataset / path)
        raise SystemExit

for path in (
    dataset / "objects" / f"motion_bank_{clip}" / f"{clip}.urdf",
    dataset / "objects" / f"motion_bank_{clip}" / f"motion_bank_{clip}.urdf",
):
    if path.exists():
        print(path)
        raise SystemExit

raise SystemExit(f"object urdf not found for {clip}")
PY
)"

if [[ "$CLIP" != box_* ]]; then
  CONVEX_DIR="${CONVEX_DIR:-$HERE/coacd_outputs}"
  CONVEX_URDF="$CONVEX_DIR/${CLIP}_mujoco_convex.urdf"
  "$MUJOCO_PY" "$HERE/tools/coacd_decompose_for_mujoco.py" \
    "$OBJECT_URDF" \
    --output "$CONVEX_URDF" \
    --absolute-paths
  OBJECT_URDF="$CONVEX_URDF"
fi

RUN_DIR="${RUN_DIR:-$HOLOSOMA/logs/cotracking_record_${CLIP}_$(date +%Y%m%d_%H%M%S)}"
TRACE="$RUN_DIR/sim_state_trace.jsonl"
MODEL_XML="$RUN_DIR/compiled_model.xml"
MP4="${OUTPUT_MP4:-$RUN_DIR/cotracking_${CLIP}.mp4}"
mkdir -p "$RUN_DIR"

echo "clip: $CLIP"
echo "recording physics rollout for ${SECONDS_TO_RECORD}s"

HOLOSOMA_MJ_TRACK_INTERNAL_CORE=1 \
MUJOCO_PY="$MUJOCO_PY" \
INFER_PY="$INFER_PY" \
OBJECT_URDF="$OBJECT_URDF" \
RUN_DIR="$RUN_DIR" \
RUN_SECONDS="$SECONDS_TO_RECORD" \
TRAINING_HEADLESS=True \
SIM_DEBUG_VIZ=False \
SIM_USE_TRAINING_URDF_OBJECT_SCENE=1 \
SIM_ADD_DEFAULT_OBJECT_ACTUATORS=1 \
HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES:-1}" \
HOLOSOMA_MUJOCO_LOAD_OBJECT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_LOAD_OBJECT_VISUAL_MESHES:-1}" \
SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML="${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML:-1}" \
SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML="${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML:-1}" \
HOLOSOMA_MUJOCO_EXPORT_XML_PATH="$MODEL_XML" \
HOLOSOMA_SPLIT_SIM_STATE_TRACE_PATH="$TRACE" \
POLICY_STDIO=log \
bash "$HERE/mj_track.sh" "$MOTION" "$MODEL" >"$RUN_DIR/launcher.log" 2>&1

echo "rendering mp4"
cp -a "$HERE/src/holosoma/holosoma/data/robots/g1/meshes" "$RUN_DIR/meshes"
rsync -a \
  --include='*/' \
  --include='*.obj' \
  --include='*.OBJ' \
  --include='*.stl' \
  --include='*.STL' \
  --exclude='*' \
  "$(dirname "$OBJECT_URDF")"/ "$RUN_DIR"/
MUJOCO_GL="${MUJOCO_GL:-egl}" "$MUJOCO_PY" "$HERE/tools/render_trace_mp4.py" \
  --model-xml "$MODEL_XML" \
  --trace "$TRACE" \
  --output "$MP4" \
  --stride "${VIDEO_STRIDE:-7}"

echo "mp4: $MP4"
echo "logs: $RUN_DIR"
