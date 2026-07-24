#!/usr/bin/env bash
set -euo pipefail

# Perception-aware VideoMimic tracking.
# Supported perception presets are intentionally restricted to:
#   - camera_depth_d435i (default, far-tracking aligned)
#   - heightmap

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/scripts/gpu_launch_defaults.sh"

PERCEPTION_PRESET=${1:-${PERCEPTION_PRESET:-camera_depth_d435i}} # camera_depth_d435i|heightmap
STAGE1_CKPT=${2:-${STAGE1_CKPT:-}}
RESUME_CKPT=${RESUME_CKPT:-}
IMAGE_WIDTH=${IMAGE_WIDTH:-106}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-60}
PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-268435456}
CUDA_VISIBLE_DEVICES="$(default_cuda_visible_devices_all "${CUDA_VISIBLE_DEVICES:-}")"
NUM_GPUS=${NUM_GPUS:-$(count_cuda_visible_devices "${CUDA_VISIBLE_DEVICES}")}
PER_GPU_ENVS=${PER_GPU_ENVS:-4096}
NUM_ENVS=${NUM_ENVS:-$((NUM_GPUS * PER_GPU_ENVS))}
TTTEST=${TTTEST:-0}
PPO_START_EPOCH=${PPO_START_EPOCH:-0}
DAGGER_END_EPOCH=${DAGGER_END_EPOCH:-10000}
DAGGER_LOSS_COEF=${DAGGER_LOSS_COEF:-10.0}
DISTILL_LOSS_TYPE=${DISTILL_LOSS_TYPE:-mse}
TEACHER_OBS_KEYS=${TEACHER_OBS_KEYS:-actor_obs,actor_obs_target}

case "${PERCEPTION_PRESET}" in
  camera_depth_d435i|heightmap)
    ;;
  *)
    echo "Unknown PERCEPTION_PRESET=${PERCEPTION_PRESET}. Use camera_depth_d435i|heightmap." >&2
    exit 1
    ;;
esac
echo "[INFO] perception:${PERCEPTION_PRESET}"

PERCEPTION_OVERRIDES=()
if [[ "${PERCEPTION_PRESET}" == "camera_depth_d435i" ]]; then
  PERCEPTION_OVERRIDES=(
    --perception.camera_width="${IMAGE_WIDTH}"
    --perception.camera_height="${IMAGE_HEIGHT}"
    --perception.camera_warp_preprocess=True
    --perception.camera_warp_freq_ratio=1
    --perception.camera_warp_latency_frame=0
    --perception.camera_warp_buffer_len=3
    --perception.camera_warp_crop_top=2
    --perception.camera_warp_crop_bottom=0
    --perception.camera_warp_crop_left=4
    --perception.camera_warp_crop_right=4
    --perception.camera_warp_min_valid_depth=0.3
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

EXP_NAME="exp:g1-29dof-wbt-videomimic-mlp"
DISTILL_OVERRIDES=()
CHECKPOINT_OVERRIDES=()

if [[ -n "${RESUME_CKPT}" ]]; then
  CHECKPOINT_OVERRIDES+=(--training.checkpoint "${RESUME_CKPT}")
fi

if [[ "${PERCEPTION_PRESET}" == "camera_depth_d435i" ]]; then
  if [[ -z "${STAGE1_CKPT}" ]]; then
    echo "camera_depth_d435i requires Stage1 teacher checkpoint as arg2." >&2
    exit 1
  fi
  EXP_NAME="exp:g1-29dof-wbt-videomimic-distill-mlp"
  DISTILL_OVERRIDES=(
    --algo.config.distill.enabled=True
    --algo.config.distill.mode=dagger
    --algo.config.distill.policy_to_clone="${STAGE1_CKPT}"
    --algo.config.distill.teacher_obs_keys="${TEACHER_OBS_KEYS}"
    --algo.config.distill.bc_loss_coef=1.0
    --algo.config.distill.distill_loss_type="${DISTILL_LOSS_TYPE}"
    --algo.config.distill.ppo_start_epoch="${PPO_START_EPOCH}"
    --algo.config.distill.dagger_end_epoch="${DAGGER_END_EPOCH}"
    --algo.config.distill.dagger_loss_coef="${DAGGER_LOSS_COEF}"
    --algo.config.distill.dagger_ignore_zero_teacher_actions=True
    --algo.config.distill.dagger_match_std=False
  )
  echo "[INFO] teacher_ckpt=${STAGE1_CKPT}"
  echo "[INFO] ppo_start_epoch=${PPO_START_EPOCH} dagger_end_epoch=${DAGGER_END_EPOCH} dagger_loss_coef=${DAGGER_LOSS_COEF}"
elif [[ -n "${STAGE1_CKPT}" && -z "${RESUME_CKPT}" ]]; then
  # Preserve legacy behavior for heightmap: arg2 resumes training.
  CHECKPOINT_OVERRIDES+=(--training.checkpoint "${STAGE1_CKPT}")
fi

OBJ_DIR="/data/terrain/___crisp_clean_geometry"
MOTION_DIR="/data/terrain/___crisp_clean_motion"
NUM_ROWS=${NUM_ROWS:-1}
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

if [[ -z "${NUM_COLS}" ]]; then
  NUM_COLS=${NUM_TILES}
fi

echo "[INFO] NUM_GPUS=${NUM_GPUS} PER_GPU_ENVS=${PER_GPU_ENVS} TOTAL_ENVS=${NUM_ENVS} NUM_TILES=${NUM_TILES} NUM_ROWS=${NUM_ROWS}"

OBJ_PATH="${OBJ_DIR}"
OBJ_META=""
if [[ -d "${OBJ_DIR}" ]]; then
  NEEDS_REBUILD=0
  if [[ -f "${FUSED_META}" ]]; then
    read -r META_ROWS META_COLS < <(
      python -c 'import json,sys
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
' "${FUSED_META}"
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

if [[ "${TTTEST}" != "0" ]]; then
  echo "[INFO] TTTEST enabled: launching Viser physics preview with 1 env"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python src/holosoma/holosoma/train_agent.py \
    "${EXP_NAME}" \
    "perception:${PERCEPTION_PRESET}" \
    "${PERCEPTION_OVERRIDES[@]}" \
    --simulator.config.sim.physx.gpu_collision_stack_size="${PHYSX_GPU_COLLISION_STACK_SIZE}" \
    terrain:terrain-load-obj \
    --training.num_envs=1 \
    --training.headless=False \
    --training.enable_viser=True \
    --training.viser_env_id=0 \
    --training.viser_update_hz=30 \
    --training.viser_recenter=True \
    --training.viser_show_scandots=True \
    --training.isaac_show_scandots=True \
    --training.isaac_scandots_point_size=3.0 \
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
    --algo.config.load_optimizer=False \
    "${DISTILL_OVERRIDES[@]}" \
    \
    --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_DIR}" \
    --command.setup_terms.motion_command.params.motion_config.pair_terrain_with_motion=True \
    --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob=0.05 \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False \
    --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0 \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=False \
    --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=0 \
    "${CHECKPOINT_OVERRIDES[@]}" \
    logger:disabled
  exit 0
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun --nproc_per_node="${NUM_GPUS}" --master_port=$((29500 + RANDOM % 1000)) src/holosoma/holosoma/train_agent.py \
  "${EXP_NAME}" \
  "perception:${PERCEPTION_PRESET}" \
  "${PERCEPTION_OVERRIDES[@]}" \
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
  --algo.config.load_optimizer=False \
  --algo.config.save_interval=1000 \
  "${DISTILL_OVERRIDES[@]}" \
  \
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_DIR}" \
  --command.setup_terms.motion_command.params.motion_config.pair_terrain_with_motion=True \
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob=0.05 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=0 \
  "${CHECKPOINT_OVERRIDES[@]}" \
  logger:wandb \
  --logger.video.enabled=False \
  --logger.headless_recording=False \
  --logger.video.upload_to_wandb=False \
  --logger.name="g1_videomimic_multiclip_terrain_depth"
