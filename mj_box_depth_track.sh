#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MOTION_DIR="$ROOT_DIR/outputs/motion_bank_success_box_0_92_0p3"
DEFAULT_MODEL_INPUT="${ROOT_DIR}/logs/wandb_runs/shoo7sr1/model_18500.onnx"
DEFAULT_OBJECT_MAP="$ROOT_DIR/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"
DEFAULT_GT_BOX_SCENE="/home/ubuntu/FAR/holosoma_gt/src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof_w_largebox.xml"
DEFAULT_PERCEPTION_CAMERA_SOURCE="${PERCEPTION_CAMERA_SOURCE_DEFAULT:-rendered}"

INFER_PYTHON_BIN="${INFER_PY:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python}"
if [[ ! -x "$INFER_PYTHON_BIN" ]]; then
  INFER_PYTHON_BIN="${INFER_PYTHON_BIN:-python3}"
fi

usage() {
  cat <<EOF
Usage:
  MOTION_DIR=/path/to/rollout_motion_bank OBJECT_URDF=/path/to/object.urdf bash mj_box_depth_track.sh [depth|rendered|warp] [clip_name|motion.npz] [model.onnx|wandb://...] [viser args...]

Defaults:
  motion_dir    = ${DEFAULT_MOTION_DIR}
  object_map    = ${DEFAULT_OBJECT_MAP}
  gt_box_scene  = ${DEFAULT_GT_BOX_SCENE}
  model         = ${DEFAULT_MODEL_INPUT}
  depth_source  = ${DEFAULT_PERCEPTION_CAMERA_SOURCE}
  object_mass   = MUJOCO_OBJECT_MASS_OVERRIDE:-<unset>
  lateral_friction = MUJOCO_OBJECT_LATERAL_FRICTION:-<unset> (GT XML already has 0.9)
  rolling_friction = MUJOCO_OBJECT_ROLLING_FRICTION:-<unset> (do not force condim=6 by default)
  contact_stiffness= MUJOCO_OBJECT_CONTACT_STIFFNESS:-<unset>
  contact_damping  = MUJOCO_OBJECT_CONTACT_DAMPING:-<unset>
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
    rendered|render|mujoco)
      PERCEPTION_CAMERA_SOURCE="rendered"
      ;;
    warp|far_tracking_warp)
      PERCEPTION_CAMERA_SOURCE="far_tracking_warp"
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
export PERCEPTION_CAMERA_SOURCE="${PERCEPTION_CAMERA_SOURCE:-$DEFAULT_PERCEPTION_CAMERA_SOURCE}"
export PERCEPTION_OBJECT_GEOMETRY_MODE="${PERCEPTION_OBJECT_GEOMETRY_MODE:-mesh}"
export HOLOSOMA_MUJOCO_USE_GT_BOX_ENV="${HOLOSOMA_MUJOCO_USE_GT_BOX_ENV:-1}"
export HOLOSOMA_MUJOCO_OBJECT_SCENE_XML="${HOLOSOMA_MUJOCO_OBJECT_SCENE_XML:-$DEFAULT_GT_BOX_SCENE}"
export HOLOSOMA_MUJOCO_SKIP_SCENE_TERRAIN="${HOLOSOMA_MUJOCO_SKIP_SCENE_TERRAIN:-1}"
export SIM_TERRAIN_MESH_TYPE="${SIM_TERRAIN_MESH_TYPE:-}"
export SIM_USE_TRAINING_URDF_OBJECT_SCENE="${SIM_USE_TRAINING_URDF_OBJECT_SCENE:-0}"
export SIM_ROBOT_MJCF_FILTER_ENABLE="${SIM_ROBOT_MJCF_FILTER_ENABLE:-False}"
export HOLOSOMA_MUJOCO_GT_GROUND_CONTACTS="${HOLOSOMA_MUJOCO_GT_GROUND_CONTACTS:-0}"
export SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML="${SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML:-0}"
export SIM_COPY_TENDONS_FROM_ROBOT_XML="${SIM_COPY_TENDONS_FROM_ROBOT_XML:-0}"
export SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML="${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML:-0}"
export SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML="${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML:-0}"
export SIM_ADD_DEFAULT_OBJECT_ACTUATORS="${SIM_ADD_DEFAULT_OBJECT_ACTUATORS:-1}"
export HOLOSOMA_W_OBJECT_URDF="${HOLOSOMA_W_OBJECT_URDF:-g1/main_mesh_collision_halfspherehand.urdf}"
export MUJOCO_OBJECT_CONTACT_BODY_MARKERS="${MUJOCO_OBJECT_CONTACT_BODY_MARKERS:-}"
export MUJOCO_OBJECT_MASS_OVERRIDE="${MUJOCO_OBJECT_MASS_OVERRIDE:-}"
export MUJOCO_OBJECT_GEOM_FRICTION="${MUJOCO_OBJECT_GEOM_FRICTION:-}"
export MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION="${MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION:-}"
export MUJOCO_OBJECT_LATERAL_FRICTION="${MUJOCO_OBJECT_LATERAL_FRICTION:-}"
export MUJOCO_OBJECT_ROLLING_FRICTION="${MUJOCO_OBJECT_ROLLING_FRICTION:-}"
export MUJOCO_OBJECT_CONTACT_STIFFNESS="${MUJOCO_OBJECT_CONTACT_STIFFNESS:-}"
export MUJOCO_OBJECT_CONTACT_DAMPING="${MUJOCO_OBJECT_CONTACT_DAMPING:-}"
export HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS="${HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS:-0}"
case "$(printf '%s' "$HOLOSOMA_MUJOCO_USE_GT_BOX_ENV" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    export MUJOCO_OBJECT_MASS_OVERRIDE=""
    export MUJOCO_OBJECT_GEOM_FRICTION=""
    export MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION=""
    export MUJOCO_OBJECT_LATERAL_FRICTION=""
    export MUJOCO_OBJECT_ROLLING_FRICTION=""
    export MUJOCO_OBJECT_CONTACT_STIFFNESS=""
    export MUJOCO_OBJECT_CONTACT_DAMPING=""
    export HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS="0"
    ;;
esac
if [[ "$PERCEPTION_CAMERA_SOURCE" == "far_tracking_warp" ]]; then
  export SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
fi
if [[ "$PERCEPTION_CAMERA_SOURCE" == "rendered" && -z "${MUJOCO_GL:-}" ]]; then
  case "$(printf '%s' "${TRAINING_HEADLESS:-${HEADLESS:-True}}" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off)
      export MUJOCO_GL=glfw
      ;;
    *)
      export MUJOCO_GL=egl
      ;;
  esac
fi
if [[ "$PERCEPTION_CAMERA_SOURCE" == "rendered" ]]; then
  export HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES:-1}"
  export HOLOSOMA_MUJOCO_LOAD_OBJECT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_LOAD_OBJECT_VISUAL_MESHES:-1}"
  export HOLOSOMA_MUJOCO_DEPTH_PREFER_VISUAL_MESHES="${HOLOSOMA_MUJOCO_DEPTH_PREFER_VISUAL_MESHES:-1}"
  export HOLOSOMA_MUJOCO_RENDERED_DEPTH_FLIPUD="${HOLOSOMA_MUJOCO_RENDERED_DEPTH_FLIPUD:-0}"
fi
export INFERENCE_CONFIG="${INFERENCE_CONFIG:-g1-29dof-wbt-object-distill}"
export RUN_SECONDS="${RUN_SECONDS:-0}"

echo "[INFO] motion_file=$MOTION_FILE"
echo "[INFO] object_urdf=$OBJECT_URDF"
echo "[INFO] model=$MODEL_LOCAL"
echo "[INFO] inference_config=$INFERENCE_CONFIG"
echo "[INFO] perception=${ENABLE_SPLIT_PERCEPTION_OBS} preset=${PERCEPTION_PRESET} camera_source=${PERCEPTION_CAMERA_SOURCE}"
echo "[INFO] gt_box_env=${HOLOSOMA_MUJOCO_USE_GT_BOX_ENV} scene_xml=${HOLOSOMA_MUJOCO_OBJECT_SCENE_XML} skip_scene_terrain=${HOLOSOMA_MUJOCO_SKIP_SCENE_TERRAIN} terrain_mesh_type=${SIM_TERRAIN_MESH_TYPE:-<training>}"
echo "[INFO] object_contact_body_markers=${MUJOCO_OBJECT_CONTACT_BODY_MARKERS:-<all robot collision bodies>}"
echo "[INFO] mujoco_scene_filter=${SIM_ROBOT_MJCF_FILTER_ENABLE} gt_ground_contacts=${HOLOSOMA_MUJOCO_GT_GROUND_CONTACTS}"
echo "[INFO] object_mass_override=${MUJOCO_OBJECT_MASS_OVERRIDE} object_geom_friction=${MUJOCO_OBJECT_GEOM_FRICTION:-<none>} lateral_friction=${MUJOCO_OBJECT_LATERAL_FRICTION:-<none>} rolling_friction=${MUJOCO_OBJECT_ROLLING_FRICTION:-<none>} contact_stiffness=${MUJOCO_OBJECT_CONTACT_STIFFNESS:-<none>} contact_damping=${MUJOCO_OBJECT_CONTACT_DAMPING:-<none>} web_demo_object_contacts=${HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS}"

exec bash "$ROOT_DIR/mj_track.sh" "$MOTION_FILE" "$MODEL_LOCAL" "${EXTRA_ARGS[@]}"
