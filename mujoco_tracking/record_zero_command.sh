#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOLOSOMA="$(cd "$HERE/.." && pwd)"
FAR="$(cd "$HOLOSOMA/.." && pwd)"

CLIP="${1:-box_10}"
DATASET_NAME="carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success133_final0p5"
if [[ -e "$HOLOSOMA/data/$DATASET_NAME" ]]; then
  DATASET="$HOLOSOMA/data/$DATASET_NAME"
else
  DATASET="$FAR/data/$DATASET_NAME"
fi

MUJOCO_PY="${MUJOCO_PY:-/home/user/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python}"
INFER_PY="${INFER_PY:-/home/user/.holosoma_deps/miniconda3/envs/hsinference/bin/python}"
LOCAL_MODEL="${HOLOSOMA_ZERO_RECORD_LOCAL_MODEL:-/tmp/zxz3hd8h_model_07500.onnx}"
MODEL="${ZERO_MODEL:-${HOLOSOMA_ZERO_RECORD_MODEL:-$LOCAL_MODEL}}"
WANDB_RUN="${HOLOSOMA_ZERO_RECORD_WANDB_RUN:-zihanw22/carry-any/zxz3hd8h}"
WANDB_FILE="${HOLOSOMA_ZERO_RECORD_WANDB_FILE:-model_07500.onnx}"
OBJECT_MASS="${HOLOSOMA_MJ_OBJECT_MASS:-2.0}"
SECONDS_TO_RECORD="${SECONDS_TO_RECORD:-20}"

if [[ "$CLIP" == "-h" || "$CLIP" == "--help" ]]; then
  echo "Usage: ./mujoco_tracking/record_zero_command.sh [clip_name]"
  echo "Example: ./mujoco_tracking/record_zero_command.sh box_10"
  exit 0
fi

if [[ "$MODEL" == wandb://* ]]; then
  MODEL="$LOCAL_MODEL"
fi

if [[ ! -f "$MODEL" ]]; then
  echo "model not found locally, downloading ${WANDB_RUN}/${WANDB_FILE} -> $MODEL"
  "$INFER_PY" - "$MODEL" "$WANDB_RUN" "$WANDB_FILE" <<'PY'
import shutil
import sys
from pathlib import Path

import wandb

out_path = Path(sys.argv[1]).expanduser()
run_path = sys.argv[2]
file_name = sys.argv[3]
out_path.parent.mkdir(parents=True, exist_ok=True)

tmp_dir = out_path.parent / f".{out_path.name}.wandb_download"
if tmp_dir.exists():
    shutil.rmtree(tmp_dir)
tmp_dir.mkdir(parents=True)

api = wandb.Api()
downloaded = api.run(run_path).file(file_name).download(root=str(tmp_dir), replace=True)
shutil.copy2(downloaded.name, out_path)
shutil.rmtree(tmp_dir, ignore_errors=True)
PY
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
  CONVEX_DIR="${CONVEX_DIR:-$HERE/coacd_outputs}"
  CONVEX_URDF="$CONVEX_DIR/${CLIP}_mujoco_convex.urdf"
  "$MUJOCO_PY" "$HERE/tools/coacd_decompose_for_mujoco.py" \
    "$OBJECT_URDF" \
    --output "$CONVEX_URDF" \
    --absolute-paths
  OBJECT_URDF="$CONVEX_URDF"
fi

RUN_DIR="${RUN_DIR:-$HOLOSOMA/logs/zero_command_record_${CLIP}_$(date +%Y%m%d_%H%M%S)}"
TRACE="$RUN_DIR/sim_state_trace.jsonl"
MODEL_XML="$RUN_DIR/compiled_model.xml"
MP4="${OUTPUT_MP4:-$RUN_DIR/zero_command_${CLIP}.mp4}"
mkdir -p "$RUN_DIR"
rm -f "$TRACE" "$MODEL_XML" "$MP4"

echo "clip: $CLIP"
echo "model: $MODEL"
echo "object_mass: ${OBJECT_MASS}kg"
echo "recording physics rollout for ${SECONDS_TO_RECORD}s with sparse root command [0, 0, 0]"

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
MUJOCO_OBJECT_MASS_OVERRIDE="$OBJECT_MASS" \
HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND="${HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND:-1}" \
HOLOSOMA_POLICY_PICKUP_BUTTON="${HOLOSOMA_POLICY_PICKUP_BUTTON:-1}" \
HOLOSOMA_POLICY_DROP_BUTTON="${HOLOSOMA_POLICY_DROP_BUTTON:-0}" \
HOLOSOMA_KEYBOARD_ROOT_COMMAND=1 \
HOLOSOMA_KEYBOARD_ROOT_COMMAND_MODE=manual \
HOLOSOMA_KEYBOARD_ROOT_COMMAND_VALUE=0 \
HOLOSOMA_KEYBOARD_ROOT_COMMAND_YAW_VALUE=0 \
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
