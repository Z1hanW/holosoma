#!/usr/bin/env bash
set -euo pipefail

# Perception-aware VideoMimic tracking with depth (D435i-style camera).

DEPTH_IMPL=${1:-${DEPTH_IMPL:-rendered}} # rendered|depth_sensor|raycast|scandots
IMAGE_WIDTH=${IMAGE_WIDTH:-128}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-72}
PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-536870912}
NUM_ENVS=${NUM_ENVS:-12280}
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
echo "[INFO] DEPTH_IMPL=${DEPTH_IMPL} -> perception:${PERCEPTION_PRESET}"

OBJ_DIR="/home/ubuntu/FAR/Store/___zero_pad_geo"
MOTION_DIR="/home/ubuntu/FAR/Store/___zero_pad_data"
NUM_ROWS=${NUM_ROWS:-}
NUM_COLS=${NUM_COLS:-}
FUSED_OBJ="../tmp_fused.obj"
FUSED_META="../tmp_fused.meta.json"
REBUILD_FUSED=${REBUILD_FUSED:-0}

NUM_TILES=1
if [[ -d "${OBJ_DIR}" ]]; then
  mapfile -t OBJ_FILES < <(find "${OBJ_DIR}" -maxdepth 1 -type f \( -name "*.obj" -o -name "*.OBJ" \) | sort)
  NUM_TILES=${#OBJ_FILES[@]}
  if [[ "${NUM_TILES}" -eq 0 ]]; then
    echo "No OBJ files found in ${OBJ_DIR}" >&2
    exit 1
  fi
fi

if [[ -z "${NUM_ROWS}" ]]; then
  if [[ "${NUM_TILES}" -gt 0 ]]; then
    NUM_ROWS=$(( (NUM_ENVS + NUM_TILES - 1) / NUM_TILES ))
  else
    NUM_ROWS=1
  fi
fi

if [[ -z "${NUM_COLS}" ]]; then
  NUM_COLS=${NUM_TILES}
fi

echo "[INFO] NUM_ENVS=${NUM_ENVS} NUM_TILES=${NUM_TILES} NUM_ROWS=${NUM_ROWS}"

OBJ_PATH="${OBJ_DIR}"
OBJ_META=""
if [[ -d "${OBJ_DIR}" ]]; then
  NEEDS_REBUILD=0
  if [[ -f "${FUSED_META}" ]]; then
    read -r META_ROWS META_COLS < <(python - <<'PY'
import json
import sys

path = sys.argv[1]
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
"${FUSED_META}")
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

CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 --master_port=$((29500 + RANDOM % 1000)) src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-videomimic-mlp \
  "perception:${PERCEPTION_PRESET}" \
  --perception.camera_width="$IMAGE_WIDTH" \
  --perception.camera_height="$IMAGE_HEIGHT" \
  --simulator.config.sim.physx.gpu_collision_stack_size="${PHYSX_GPU_COLLISION_STACK_SIZE}" \
  terrain:terrain-load-obj \
  --training.num_envs="${NUM_ENVS}" \
  --simulator.config.scene.env_spacing=0.0 \
  --terrain.terrain-term.obj-file-path "${OBJ_PATH}" \
  ${OBJ_META:+--terrain.terrain-term.obj-metadata-path "${OBJ_META}"} \
  --terrain.terrain-term.num-rows "${NUM_ROWS}" \
  --terrain.terrain-term.num-cols "${NUM_COLS}" \
  \
  --algo.config.actor_learning_rate=7e-5 \
  --algo.config.critic_learning_rate=7e-5 \
  --algo.config.normalize_actor_obs=False \
  --algo.config.normalize_critic_obs=False \
  --algo.config.save_interval=1000 \
  \
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_DIR}" \
  --command.setup_terms.motion_command.params.motion_config.pair_terrain_with_motion=True \
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob=0.05 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=0 \
  logger:wandb \
  --logger.video.interval=1000 \
  --logger.name="g1_videomimic_multiclip_terrain_depth"
