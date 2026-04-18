#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MOTION_DIR="$ROOT_DIR/data/ds_box_data/train_g1_w_obj_prepared"
DEFAULT_MODEL_INPUT="wandb://zihanw22/boxer/shoo7sr1/model_07500.onnx"
DEFAULT_OBJECT_MAP="$DEFAULT_MOTION_DIR/_clip_object_urdf_map.json"

INFER_PYTHON_BIN="${INFER_PY:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python}"
if [[ ! -x "$INFER_PYTHON_BIN" ]]; then
  INFER_PYTHON_BIN="${INFER_PYTHON_BIN:-python3}"
fi

usage() {
  cat <<EOF
Usage:
  MOTION_DIR=/path/to/prepared OBJECT_URDF=/path/to/_clip_object_urdf_map.json bash mj_box_depth_track.sh [depth] [clip_name|motion.npz] [model.onnx|wandb://...] [viser args...]

Defaults:
  motion_dir = ${DEFAULT_MOTION_DIR}
  object_map = ${DEFAULT_OBJECT_MAP}
  model      = ${DEFAULT_MODEL_INPUT}
EOF
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

MOTION_DIR="${MOTION_DIR:-$DEFAULT_MOTION_DIR}"
OBJECT_MAP_INPUT="${OBJECT_URDF:-$DEFAULT_OBJECT_MAP}"
MODEL_INPUT="${MODEL_INPUT:-${MODEL_PATH:-$DEFAULT_MODEL_INPUT}}"
MOTION_FILE="${MOTION_FILE:-}"
MOTION_CLIP_NAME="${MOTION_CLIP_NAME:-${MOTION_CLIP:-}}"
EXTRA_ARGS=()
POSITIONAL_MODE=1

for arg in "$@"; do
  if [[ "$POSITIONAL_MODE" == "0" ]]; then
    EXTRA_ARGS+=("$arg")
    continue
  fi
  case "$arg" in
    depth)
      ;;
    *.npz)
      MOTION_FILE="$arg"
      ;;
    wandb://*|https://*|*.onnx|*.pt)
      MODEL_INPUT="$arg"
      ;;
    --*)
      POSITIONAL_MODE=0
      EXTRA_ARGS+=("$arg")
      ;;
    *)
      if [[ -z "$MOTION_CLIP_NAME" ]]; then
        MOTION_CLIP_NAME="$arg"
      else
        EXTRA_ARGS+=("$arg")
      fi
      ;;
  esac
done

if [[ -z "$MOTION_FILE" ]]; then
  if [[ -n "$MOTION_CLIP_NAME" ]]; then
    MOTION_FILE="$MOTION_DIR/${MOTION_CLIP_NAME%.npz}.npz"
  else
    MOTION_FILE="$(find "$MOTION_DIR" -maxdepth 1 -name '*.npz' | sort | head -n 1)"
  fi
fi

if [[ -z "$MOTION_FILE" || ! -f "$MOTION_FILE" ]]; then
  echo "[ERROR] Motion clip not found: ${MOTION_FILE:-<empty>}" >&2
  exit 1
fi
MOTION_FILE="$(cd "$(dirname "$MOTION_FILE")" && pwd)/$(basename "$MOTION_FILE")"

if [[ "$MODEL_INPUT" == wandb://*.pt ]]; then
  MODEL_INPUT="${MODEL_INPUT%.pt}.onnx"
fi

MODEL_LOCAL="$(
  "$INFER_PYTHON_BIN" - <<'PY' "$MODEL_INPUT" "$ROOT_DIR/logs/wandb_runs"
import sys
from pathlib import Path

from holosoma_inference.utils.wandb import load_checkpoint

model = sys.argv[1]
root = Path(sys.argv[2])
download_dir = root / "box_depth"
if model.startswith("wandb://"):
    parts = model[len("wandb://") :].split("/", 3)
    if len(parts) >= 3:
        download_dir = root / parts[2]
path = load_checkpoint(None, model, str(download_dir))
print(Path(path).expanduser().resolve())
PY
)"
MODEL_LOCAL="$(printf '%s\n' "$MODEL_LOCAL" | tail -n 1)"

OBJECT_URDF_RESOLVED="$(
  "$INFER_PYTHON_BIN" - <<'PY' "$OBJECT_MAP_INPUT" "$MOTION_FILE"
import json
import sys
from pathlib import Path

import numpy as np

raw = sys.argv[1]
motion_path = Path(sys.argv[2]).expanduser().resolve()
stem = motion_path.stem

candidate = Path(raw).expanduser() if raw else None
if candidate is not None and candidate.is_file() and candidate.suffix.lower() == ".json":
    data = json.loads(candidate.read_text())
    clips = data.get("clips", data) if isinstance(data, dict) else {}
    entry = clips.get(stem) if isinstance(clips, dict) else None
    if not isinstance(entry, dict):
        raise SystemExit(f"Object map has no entry for clip '{stem}': {candidate}")
    path = entry.get("object_urdf_path") or entry.get("urdf_path")
    if not path:
        raise SystemExit(f"Object map entry for clip '{stem}' has no object_urdf_path")
    print(Path(path).expanduser().resolve())
elif candidate is not None and str(candidate):
    print(candidate.expanduser().resolve())
else:
    with np.load(motion_path, allow_pickle=True) as data:
        if "object_urdf_path" not in data:
            raise SystemExit(f"No OBJECT_URDF map provided and motion has no object_urdf_path: {motion_path}")
        print(Path(str(np.asarray(data["object_urdf_path"]).item())).expanduser().resolve())
PY
)"

export OBJECT_URDF="$OBJECT_URDF_RESOLVED"
export ENABLE_SPLIT_PERCEPTION_OBS="${ENABLE_SPLIT_PERCEPTION_OBS:-1}"
export PERCEPTION_PRESET="${PERCEPTION_PRESET:-camera_depth_d435i}"
export PERCEPTION_CAMERA_SOURCE="${PERCEPTION_CAMERA_SOURCE:-far_tracking_warp}"
export PERCEPTION_OBJECT_GEOMETRY_MODE="${PERCEPTION_OBJECT_GEOMETRY_MODE:-mesh}"
export SIM_USE_TRAINING_URDF_OBJECT_SCENE="${SIM_USE_TRAINING_URDF_OBJECT_SCENE:-1}"
if [[ "$PERCEPTION_CAMERA_SOURCE" == "far_tracking_warp" ]]; then
  export SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
fi
export INFERENCE_CONFIG="${INFERENCE_CONFIG:-g1-29dof-wbt-object-distill}"
export RUN_SECONDS="${RUN_SECONDS:-0}"

echo "[INFO] motion_file=$MOTION_FILE"
echo "[INFO] object_urdf=$OBJECT_URDF"
echo "[INFO] model=$MODEL_LOCAL"
echo "[INFO] inference_config=$INFERENCE_CONFIG"
echo "[INFO] perception=${ENABLE_SPLIT_PERCEPTION_OBS} preset=${PERCEPTION_PRESET} camera_source=${PERCEPTION_CAMERA_SOURCE}"

exec bash "$ROOT_DIR/mj_track.sh" "$MOTION_FILE" "$MODEL_LOCAL" "${EXTRA_ARGS[@]}"
