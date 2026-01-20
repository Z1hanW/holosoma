#!/usr/bin/env bash
set -euo pipefail

# Perception-aware VideoMimic tracking with depth (D435i-style camera).

DEPTH_IMPL=${DEPTH_IMPL:-rendered} # rendered|depth_sensor|raycast
IMAGE_WIDTH=${IMAGE_WIDTH:-1280}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-720}
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
    PERCEPTION_PRESET="perception:camera-depth-d435i-scandots"
  *)
    echo "Unknown DEPTH_IMPL=${DEPTH_IMPL}. Use rendered|depth_sensor|raycast." >&2
    exit 1
    ;;
esac

MOTION_DIR="multi-geometry/test"
OBJ_DIR="multi-motion/test"
NUM_ROWS=1
NUM_COLS=1
FUSED_OBJ="../tmp_fused.obj"
FUSED_META="../tmp_fused.meta.json"
REBUILD_FUSED=${REBUILD_FUSED:-0}

OBJ_PATH="${OBJ_DIR}"
OBJ_META=""
if [[ -d "${OBJ_DIR}" ]]; then
  if [[ "${REBUILD_FUSED}" == "1" || ! -f "${FUSED_OBJ}" || ! -f "${FUSED_META}" ]]; then
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
  terrain:terrain-load-obj \
  --training.num_envs=30720 \
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
