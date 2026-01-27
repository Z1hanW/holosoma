#!/usr/bin/env bash
set -euo pipefail

# Evaluate a multi-perception policy checkpoint with motion + geometry in Isaac Sim (GUI on by default).
#
# Usage:
#   CKPT=/abs/path/to/model.pt ./eval_multi_perception.sh
#
# Optional overrides:
#   DEPTH_IMPL=rendered|depth_sensor|raycast|scandots
#   MOTION_FILE=/abs/path/to/motion_or_dir
#   OBJ_DIR=/abs/path/to/obj_dir_or_obj_file
#   NUM_ENVS=1 HEADLESS=False NUM_ROWS=1 NUM_COLS=
#   PAIR_TERRAIN=True START_AT_TIMESTEP_ZERO_PROB=0.05
#   ENABLE_DEFAULT_POSE_APPEND=False DEFAULT_POSE_APPEND_DURATION_S=0
#   ENABLE_DEFAULT_POSE_PREPEND=False DEFAULT_POSE_PREPEND_DURATION_S=0
#   IMAGE_WIDTH=128 IMAGE_HEIGHT=72
#   PHYSX_GPU_COLLISION_STACK_SIZE=4294967295
#   MAX_EVAL_STEPS=1000

DEPTH_IMPL=${1:-${DEPTH_IMPL:-rendered}} # rendered|depth_sensor|raycast|scandots
CKPT=${2:-${CKPT:-}}
IMAGE_WIDTH=${IMAGE_WIDTH:-128}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-72}
PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-4294967295}
NUM_ENVS=${NUM_ENVS:-1}
HEADLESS=${HEADLESS:-False}
MAX_EVAL_STEPS=${MAX_EVAL_STEPS:-}

case "${DEPTH_IMPL}" in
  rendered)
    PERCEPTION_PRESET="camera_depth_d435i_rendered"
    ;;
  depth_sensor)
    PERCEPTION_PRESET="camera_depth_d435i_depth_sensor"
    ;;
  raycast)
    PERCEPTION_PRESET="camera_depth_d435i"
    ;;
  scandots)
    PERCEPTION_PRESET="camera_depth_d435i_scandots"
    ;;
  *)
    echo "Unknown DEPTH_IMPL=${DEPTH_IMPL}. Use rendered|depth_sensor|raycast|scandots." >&2
    exit 1
    ;;
esac

if [[ -z "${CKPT}" ]]; then
  echo "Set CKPT to your checkpoint path." >&2
  exit 1
fi

OBJ_DIR=${OBJ_DIR:-"/home/ubuntu/FAR/Store/vmm_data/___zero_pad_geo_trans"}
MOTION_FILE=${MOTION_FILE:-"/home/ubuntu/FAR/Store/vmm_data/___zero_pad_data_trans"}
NUM_ROWS=${NUM_ROWS:-1}
NUM_COLS=${NUM_COLS:-}
FUSED_OBJ="../tmp_fused.obj"
FUSED_META="../tmp_fused.meta.json"
REBUILD_FUSED=${REBUILD_FUSED:-0}
PAIR_TERRAIN=${PAIR_TERRAIN:-True}

START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.05}
ENABLE_DEFAULT_POSE_APPEND=${ENABLE_DEFAULT_POSE_APPEND:-False}
DEFAULT_POSE_APPEND_DURATION_S=${DEFAULT_POSE_APPEND_DURATION_S:-0}
ENABLE_DEFAULT_POSE_PREPEND=${ENABLE_DEFAULT_POSE_PREPEND:-False}
DEFAULT_POSE_PREPEND_DURATION_S=${DEFAULT_POSE_PREPEND_DURATION_S:-0}

NUM_TILES=1
if [[ -d "${OBJ_DIR}" ]]; then
  mapfile -t OBJ_FILES < <(find "${OBJ_DIR}" -maxdepth 1 -type f \( -name "*.obj" -o -name "*.OBJ" \) | sort)
  NUM_TILES=${#OBJ_FILES[@]}
  if [[ "${NUM_TILES}" -eq 0 ]]; then
    echo "No OBJ files found in ${OBJ_DIR}" >&2
    exit 1
  fi
fi

if [[ -z "${NUM_COLS}" ]]; then
  NUM_COLS=${NUM_TILES}
fi

echo "[INFO] DEPTH_IMPL=${DEPTH_IMPL} -> perception:${PERCEPTION_PRESET}"
echo "[INFO] CKPT=${CKPT}"
echo "[INFO] NUM_ENVS=${NUM_ENVS} NUM_TILES=${NUM_TILES} NUM_ROWS=${NUM_ROWS}"

OBJ_PATH="${OBJ_DIR}"
OBJ_META=""
if [[ -d "${OBJ_DIR}" ]]; then
  NEEDS_REBUILD=0
  if [[ -f "${FUSED_META}" ]]; then
    read -r META_ROWS META_COLS < <(
      python - <<'PY'
import json,sys
path = sys.argv[1] if len(sys.argv) > 1 else ""
rows = 0
cols = 0
try:
    with open(path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    rows = int(meta.get("tile_rows", 0) or 0)
    cols = int(meta.get("tile_cols", 0) or 0)
except Exception:
    rows = 0
    cols = 0
print(rows, cols)
PY
      "${FUSED_META}"
    )
    if [[ "${META_ROWS}" -ne "${NUM_ROWS}" || "${META_COLS}" -ne "${NUM_TILES}" ]]; then
      NEEDS_REBUILD=1
    fi
  else
    NEEDS_REBUILD=1
  fi
  if [[ "${REBUILD_FUSED}" == "1" || "${NEEDS_REBUILD}" == "1" || ! -f "${FUSED_OBJ}" ]]; then
    python preprocess/build_obj_terrain_tiles.py \
      --obj-dir "${OBJ_DIR}" \
      --out-obj "${FUSED_OBJ}" \
      --out-meta "${FUSED_META}" \
      --num-rows "${NUM_ROWS}"
  fi
  OBJ_PATH="${FUSED_OBJ}"
  OBJ_META="${FUSED_META}"
fi

cmd=(
  python src/holosoma/holosoma/eval_agent.py
  --checkpoint "${CKPT}"
  exp:g1-29dof-wbt-videomimic-mlp
  "perception:${PERCEPTION_PRESET}"
  --perception.camera_width="${IMAGE_WIDTH}"
  --perception.camera_height="${IMAGE_HEIGHT}"
  --simulator.config.sim.physx.gpu_collision_stack_size="${PHYSX_GPU_COLLISION_STACK_SIZE}"
  --training.num_envs="${NUM_ENVS}"
  --training.headless="${HEADLESS}"
  --simulator.config.scene.env_spacing=0.0
  terrain:terrain-load-obj
  --terrain.terrain-term.obj-file-path "${OBJ_PATH}"
  --terrain.terrain-term.num-rows "${NUM_ROWS}"
  --terrain.terrain-term.num-cols "${NUM_COLS}"
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_FILE}"
  --command.setup_terms.motion_command.params.motion_config.pair_terrain_with_motion "${PAIR_TERRAIN}"
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob "${START_AT_TIMESTEP_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append "${ENABLE_DEFAULT_POSE_APPEND}"
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s "${DEFAULT_POSE_APPEND_DURATION_S}"
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend "${ENABLE_DEFAULT_POSE_PREPEND}"
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s "${DEFAULT_POSE_PREPEND_DURATION_S}"
)

if [[ -n "${OBJ_META}" ]]; then
  cmd+=(--terrain.terrain-term.obj-metadata-path "${OBJ_META}")
fi

if [[ -n "${MAX_EVAL_STEPS}" ]]; then
  cmd+=(--training.max_eval_steps "${MAX_EVAL_STEPS}")
fi

"${cmd[@]}"
