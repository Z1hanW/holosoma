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
  echo "Usage: ./mujoco_tracking/run_cotracking.sh [clip_name]"
  echo "Example: ./mujoco_tracking/run_cotracking.sh box_74"
  exit 0
fi

MOTION="$DATASET/$CLIP.npz"
[[ -f "$MOTION" ]] || { echo "motion not found: $MOTION" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }

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
  CONVEX_URDF="$HERE/coacd_outputs/${CLIP}_mujoco_convex.urdf"
  "$MUJOCO_PY" "$HERE/tools/coacd_decompose_for_mujoco.py" \
    "$OBJECT_URDF" \
    --output "$CONVEX_URDF" \
    --absolute-paths
  OBJECT_URDF="$CONVEX_URDF"
fi

RUN_DIR="$HOLOSOMA/logs/cotracking_${CLIP}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo "clip: $CLIP"
echo "logs: $RUN_DIR"
echo "Close the MuJoCo window to stop."

HOLOSOMA_MJ_TRACK_INTERNAL_CORE=1 \
MUJOCO_PY="$MUJOCO_PY" \
INFER_PY="$INFER_PY" \
OBJECT_URDF="$OBJECT_URDF" \
RUN_DIR="$RUN_DIR" \
RUN_SECONDS=0 \
TRAINING_HEADLESS=False \
SIM_DEBUG_VIZ=True \
SIM_USE_TRAINING_URDF_OBJECT_SCENE=1 \
SIM_ADD_DEFAULT_OBJECT_ACTUATORS=1 \
HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES:-1}" \
HOLOSOMA_MUJOCO_LOAD_OBJECT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_LOAD_OBJECT_VISUAL_MESHES:-1}" \
SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML="${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML:-1}" \
SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML="${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML:-1}" \
POLICY_STDIO=log \
bash "$HERE/mj_track.sh" "$MOTION" "$MODEL" >"$RUN_DIR/launcher.log" 2>&1
