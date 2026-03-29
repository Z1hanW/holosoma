#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "${SCRIPT_DIR}"

SIM_ENV_BIN=/home/ubuntu/miniconda3/envs/sim/bin
if ! command -v torchrun >/dev/null 2>&1 && [[ -x "${SIM_ENV_BIN}/torchrun" ]]; then
  export PATH="${SIM_ENV_BIN}:${PATH}"
fi
if [[ -x "${SIM_ENV_BIN}/python" ]]; then
  DEFAULT_PYTHON_BIN="${SIM_ENV_BIN}/python"
else
  DEFAULT_PYTHON_BIN="$(command -v python)"
fi
PYTHON_BIN=${PYTHON_BIN:-"${DEFAULT_PYTHON_BIN}"}

PERCEPTION_PRESET=${1:-${PERCEPTION_PRESET:-camera_depth_d435i}}
case "${PERCEPTION_PRESET}" in
  camera_depth_d435i|heightmap)
    if [[ $# -gt 0 ]]; then
      shift
    fi
    ;;
  *)
    echo "[ERROR] Unknown PERCEPTION_PRESET=${PERCEPTION_PRESET}. Use camera_depth_d435i|heightmap." >&2
    exit 1
    ;;
esac

EXP=${EXP:-g1-29dof-wbt-terrain-transformer}
if [[ "${EXP}" == exp:* ]]; then
  EXP_ARG="${EXP}"
else
  EXP_ARG="exp:${EXP}"
fi

DEFAULT_CUDA_VISIBLE_DEVICES=${DEFAULT_CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}
NPROC=${NPROC:-$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")}
PER_GPU_ENVS=${PER_GPU_ENVS:-8192}
NUM_ENVS=${NUM_ENVS:-$((NPROC * PER_GPU_ENVS))}
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

WANDB_PROJECT=${WANDB_PROJECT:-boxer}
TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_terrain_generalist}
LOGGER_NAME=${LOGGER_NAME:-g1_terrain_generalist}
RESUME_CKPT=${RESUME_CKPT:-}

ACTOR_LR=${ACTOR_LR:-7e-5}
CRITIC_LR=${CRITIC_LR:-7e-5}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
LOAD_OPTIMIZER=${LOAD_OPTIMIZER:-False}
PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-268435456}

MOTION_DIR=${MOTION_DIR:-/data/terrain/___crisp_clean_motion}
OBJ_SOURCE=${OBJ_SOURCE:-/data/terrain/___crisp_clean_geometry}
OBJ_META_PATH=${OBJ_META_PATH:-}
NUM_ROWS=${NUM_ROWS:-}
NUM_COLS=${NUM_COLS:-}
REBUILD_FUSED=${REBUILD_FUSED:-0}
FUSED_OUT_DIR=${FUSED_OUT_DIR:-${SCRIPT_DIR}/multi-terrain/generated}
FUSED_PREFIX=${FUSED_PREFIX:-terrain_generalist}

PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-True}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.05}
ENABLE_DEFAULT_POSE_APPEND=${ENABLE_DEFAULT_POSE_APPEND:-False}
DEFAULT_POSE_APPEND_DURATION_S=${DEFAULT_POSE_APPEND_DURATION_S:-0}
ENABLE_DEFAULT_POSE_PREPEND=${ENABLE_DEFAULT_POSE_PREPEND:-False}
DEFAULT_POSE_PREPEND_DURATION_S=${DEFAULT_POSE_PREPEND_DURATION_S:-0}

HEADLESS=${HEADLESS:-True}
ENABLE_VISER=${ENABLE_VISER:-0}
VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_ENV_COUNT=${VISER_ENV_COUNT:-${NUM_ENVS}}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_SYNC_TO_SIM=${VISER_SYNC_TO_SIM:-True}
VISER_FORCE_DT=${VISER_FORCE_DT:-True}
VISER_RECENTER=${VISER_RECENTER:-True}
VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS:-False}
VISER_MULTI_ENV_SPACING=${VISER_MULTI_ENV_SPACING:-0.0}

IMAGE_WIDTH=${IMAGE_WIDTH:-106}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-60}
CAMERA_WARP_PREPROCESS=${CAMERA_WARP_PREPROCESS:-True}
DISABLE_CAMERA_RANDOMIZATION=${DISABLE_CAMERA_RANDOMIZATION:-0}

read_meta_rows_cols() {
  local meta_path="$1"
  "${PYTHON_BIN}" - "${meta_path}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    meta = json.load(handle)
rows = int(meta.get("tile_rows", 1) or 1)
cols = int(meta.get("tile_cols", 1) or 1)
print(rows, cols)
PY
}

if [[ ! -e "${OBJ_SOURCE}" ]]; then
  echo "[ERROR] OBJ_SOURCE not found: ${OBJ_SOURCE}" >&2
  exit 1
fi
if [[ ! -e "${MOTION_DIR}" ]]; then
  echo "[ERROR] MOTION_DIR not found: ${MOTION_DIR}" >&2
  exit 1
fi

OBJ_PATH="${OBJ_SOURCE}"
if [[ -d "${OBJ_SOURCE}" ]]; then
  mapfile -t OBJ_FILES < <(find "${OBJ_SOURCE}" -maxdepth 1 -type f \( -name "*.obj" -o -name "*.OBJ" \) | sort)
  NUM_TILES=${#OBJ_FILES[@]}
  if [[ "${NUM_TILES}" -eq 0 ]]; then
    echo "[ERROR] No OBJ files found in ${OBJ_SOURCE}" >&2
    exit 1
  fi

  if [[ -z "${NUM_ROWS}" ]]; then
    NUM_ROWS=1
  fi

  mkdir -p "${FUSED_OUT_DIR}"
  FUSED_OBJ="${FUSED_OUT_DIR}/${FUSED_PREFIX}_${NUM_ROWS}x${NUM_TILES}.obj"
  FUSED_META="${FUSED_OUT_DIR}/${FUSED_PREFIX}_${NUM_ROWS}x${NUM_TILES}.json"

  NEEDS_REBUILD=0
  if [[ ! -f "${FUSED_OBJ}" || ! -f "${FUSED_META}" ]]; then
    NEEDS_REBUILD=1
  else
    read -r META_ROWS META_COLS < <(read_meta_rows_cols "${FUSED_META}")
    if [[ "${META_ROWS}" != "${NUM_ROWS}" || "${META_COLS}" != "${NUM_TILES}" ]]; then
      NEEDS_REBUILD=1
    fi
  fi

  if [[ "${REBUILD_FUSED}" == "1" || "${NEEDS_REBUILD}" == "1" ]]; then
    "${PYTHON_BIN}" preprocess/build_obj_terrain_tiles.py \
      --obj-dir "${OBJ_SOURCE}" \
      --out-obj "${FUSED_OBJ}" \
      --out-meta "${FUSED_META}" \
      --num-rows "${NUM_ROWS}"
  fi

  OBJ_PATH="${FUSED_OBJ}"
  OBJ_META_PATH="${FUSED_META}"
fi

if [[ -z "${OBJ_META_PATH}" && -f "${OBJ_PATH%.*}.json" ]]; then
  OBJ_META_PATH="${OBJ_PATH%.*}.json"
fi

if [[ -n "${OBJ_META_PATH}" ]]; then
  if [[ ! -f "${OBJ_META_PATH}" ]]; then
    echo "[ERROR] OBJ_META_PATH not found: ${OBJ_META_PATH}" >&2
    exit 1
  fi
  read -r META_ROWS META_COLS < <(read_meta_rows_cols "${OBJ_META_PATH}")
  NUM_ROWS=${NUM_ROWS:-${META_ROWS}}
  NUM_COLS=${NUM_COLS:-${META_COLS}}
fi

NUM_ROWS=${NUM_ROWS:-1}
NUM_COLS=${NUM_COLS:-1}

PERCEPTION_OVERRIDES=()
if [[ "${PERCEPTION_PRESET}" == "camera_depth_d435i" ]]; then
  PERCEPTION_OVERRIDES=(
    --perception.camera_width="${IMAGE_WIDTH}"
    --perception.camera_height="${IMAGE_HEIGHT}"
    --perception.camera_warp_preprocess="${CAMERA_WARP_PREPROCESS}"
    --perception.camera_warp_freq_ratio=1
    --perception.camera_warp_latency_frame=0
    --perception.camera_warp_buffer_len=3
    --perception.camera_warp_crop_top=2
    --perception.camera_warp_crop_bottom=0
    --perception.camera_warp_crop_left=4
    --perception.camera_warp_crop_right=4
    --perception.camera_warp_min_valid_depth=0.15
    --perception.camera_warp_normalize=True
    --perception.camera_warp_edge_noise=True
    --perception.camera_warp_edge_border=3
    --perception.camera_warp_edge_shuffle_prob=0.9
    --perception.camera_warp_edge_empty_prob=0.7
    --perception.camera_warp_edge_thresh_primary=1.0
    --perception.camera_warp_edge_thresh_secondary=0.6
    --perception.camera_warp_edge_far_depth_thresh=2.5
    --perception.camera_warp_enable_holes=False
    --perception.camera_warp_hole_prob=0.0
  )
fi

RANDOMIZATION_OVERRIDES=()
if [[ "${DISABLE_CAMERA_RANDOMIZATION}" == "1" ]]; then
  RANDOMIZATION_OVERRIDES=(
    --randomization.setup_terms.setup_camera_raycast_randomization.params.enabled=False
    --randomization.reset_terms.randomize_camera_raycast.params.enabled=False
  )
fi

VISER_OVERRIDES=()
if [[ "${ENABLE_VISER}" == "1" ]]; then
  VISER_OVERRIDES=(
    --training.enable_viser=True
    --training.viser_port="${VISER_PORT}"
    --training.viser_env_id="${VISER_ENV_ID}"
    --training.viser_env_count="${VISER_ENV_COUNT}"
    --training.viser_update_hz="${VISER_UPDATE_HZ}"
    --training.viser_sync_to_sim="${VISER_SYNC_TO_SIM}"
    --training.viser_force_dt="${VISER_FORCE_DT}"
    --training.viser_recenter="${VISER_RECENTER}"
    --training.viser_show_scandots="${VISER_SHOW_SCANDOTS}"
    --training.viser_multi_env_spacing="${VISER_MULTI_ENV_SPACING}"
  )
fi

CHECKPOINT_OVERRIDES=()
if [[ -n "${RESUME_CKPT}" ]]; then
  CHECKPOINT_OVERRIDES=(
    --training.checkpoint "${RESUME_CKPT}"
  )
fi

echo "[INFO] EXP=${EXP_ARG}"
echo "[INFO] PERCEPTION=${PERCEPTION_PRESET}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] NPROC=${NPROC} PER_GPU_ENVS=${PER_GPU_ENVS} NUM_ENVS=${NUM_ENVS}"
echo "[INFO] MOTION_DIR=${MOTION_DIR}"
echo "[INFO] OBJ_PATH=${OBJ_PATH}"
if [[ -n "${OBJ_META_PATH}" ]]; then
  echo "[INFO] OBJ_META_PATH=${OBJ_META_PATH}"
fi
echo "[INFO] TERRAIN_GRID=${NUM_ROWS}x${NUM_COLS}"
echo "[INFO] PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION}"
if [[ "${ENABLE_VISER}" == "1" ]]; then
  echo "[INFO] VISER=http://localhost:${VISER_PORT}"
fi

cmd=(
  torchrun
  --nproc_per_node="${NPROC}"
  --master_port="${MASTER_PORT}"
  src/holosoma/holosoma/train_agent.py
  "${EXP_ARG}"
  "perception:${PERCEPTION_PRESET}"
  terrain:terrain-load-obj
  --training.project="${WANDB_PROJECT}"
  --training.name="${TRAINING_NAME}"
  --training.num_envs="${NUM_ENVS}"
  --training.headless="${HEADLESS}"
  --simulator.config.scene.env_spacing=0.0
  --simulator.config.sim.physx.gpu_collision_stack_size="${PHYSX_GPU_COLLISION_STACK_SIZE}"
  --terrain.terrain-term.obj-file-path "${OBJ_PATH}"
  --terrain.terrain-term.num-rows "${NUM_ROWS}"
  --terrain.terrain-term.num-cols "${NUM_COLS}"
  --algo.config.actor_learning_rate="${ACTOR_LR}"
  --algo.config.critic_learning_rate="${CRITIC_LR}"
  --algo.config.normalize_actor_obs=False
  --algo.config.normalize_critic_obs=False
  --algo.config.load_optimizer="${LOAD_OPTIMIZER}"
  --algo.config.save_interval="${SAVE_INTERVAL}"
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_DIR}"
  --command.setup_terms.motion_command.params.motion_config.pair_terrain_with_motion="${PAIR_TERRAIN_WITH_MOTION}"
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob="${START_AT_TIMESTEP_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append="${ENABLE_DEFAULT_POSE_APPEND}"
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s="${DEFAULT_POSE_APPEND_DURATION_S}"
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend="${ENABLE_DEFAULT_POSE_PREPEND}"
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s="${DEFAULT_POSE_PREPEND_DURATION_S}"
)

if [[ -n "${OBJ_META_PATH}" ]]; then
  cmd+=(--terrain.terrain-term.obj-metadata-path "${OBJ_META_PATH}")
fi

cmd+=("${PERCEPTION_OVERRIDES[@]}")
cmd+=("${RANDOMIZATION_OVERRIDES[@]}")
cmd+=("${VISER_OVERRIDES[@]}")
cmd+=("${CHECKPOINT_OVERRIDES[@]}")
cmd+=(
  logger:wandb
  --logger.video.enabled=False
  --logger.headless_recording=False
  --logger.video.upload_to_wandb=False
  --logger.name="${LOGGER_NAME}"
)
cmd+=("$@")

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}"
