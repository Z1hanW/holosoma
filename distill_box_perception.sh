#!/usr/bin/env bash
set -euo pipefail

# Distill object-carry generalist -> non-goal student with depth perception access.
#
# Student policy observation (actor):
# - actor_obs_root: sparse root command
# - actor_obs_proprio (base_lin_vel, base_ang_vel, dof_pos, dof_vel, actions)
# - perception_obs (camera depth)
# - No actor box state is used by student actor.
#
# Teacher policy observation:
# - actor_obs (full teacher state, includes object terms)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

DEFAULT_TEACHER_CHECKPOINT=${DEFAULT_TEACHER_CHECKPOINT:-"wandb://zihanw22/boxer/kge4jozt/model_12000.pt"}
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${DEFAULT_TEACHER_CHECKPOINT}}"
POSITIONAL_RUN_NAME=""

if [[ $# -gt 0 ]]; then
  if [[ "$1" == wandb://* || "$1" == /* || "$1" == ./* || "$1" == ../* || "$1" == *.pt ]]; then
    TEACHER_CHECKPOINT="$1"
    shift
  elif [[ "$1" != -* ]]; then
    POSITIONAL_RUN_NAME="$1"
    shift
  fi
fi

if [[ -z "${TEACHER_CHECKPOINT}" ]]; then
  echo "Usage: $0 <teacher_checkpoint.pt> [extra train args...]" >&2
  exit 1
fi

# Sim2real default: sparse root-command distill without clip_phase in student torso observation.
# Legacy option (old behavior with clip_phase):
#   EXP=g1-29dof-wbt-w-object-distill-sparse-root-cmd-legacy
EXP=${EXP:-g1-29dof-wbt-w-object-distill-sparse-root-cmd}
RUN_NAME=${RUN_NAME:-g1_w_object_distill_box_perception}
TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_box_perception_access_to_depth}
TRAINING_PROJECT=${TRAINING_PROJECT:-boxer}

if [[ -n "${POSITIONAL_RUN_NAME}" ]]; then
  RUN_NAME="${POSITIONAL_RUN_NAME}"
fi

# Keep launcher self-contained: direct `bash ./distill_box_perception.sh` works out of box.
HSSIM_BIN_DIR=${HSSIM_BIN_DIR:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin}
if [[ -d "${HSSIM_BIN_DIR}" ]]; then
  export PATH="${HSSIM_BIN_DIR}:${PATH}"
fi
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
if [[ -z "${NPROC:-}" ]]; then
  IFS=',' read -r -a _visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC=${#_visible_gpus[@]}
fi

# Default teacher (kge4jozt/model_12000.pt) uses actor_obs-only input.
# For legacy teachers, override TEACHER_OBS_KEYS explicitly (e.g., actor_obs_legacy,perception_obs).
TEACHER_OBS_KEYS=${TEACHER_OBS_KEYS:-actor_obs}
TEACHER_ACTION_MIX_RATIO=${TEACHER_ACTION_MIX_RATIO:-0.0}
BC_LOSS_COEF=${BC_LOSS_COEF:-1.0}
NUM_LEARNING_ITERATIONS=${NUM_LEARNING_ITERATIONS:-3000}
PPO_START_EPOCH=${PPO_START_EPOCH:-0}
# Perception distill benefits from a shorter curriculum and a stronger PPO tail.
DAGGER_END_EPOCH=${DAGGER_END_EPOCH:-2000}
DAGGER_LOSS_COEF=${DAGGER_LOSS_COEF:-5.0}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.7}
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
PERCEPTION_PRESET=${PERCEPTION_PRESET:-camera_depth_d435i}

# Teacher expects perception encoder input dim 289 => 17x17.
IMAGE_WIDTH=${IMAGE_WIDTH:-17}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-17}
CAMERA_NEAR=${CAMERA_NEAR:-0.001}
CAMERA_FAR=${CAMERA_FAR:-3.0}
CAMERA_MAX_DISTANCE=${CAMERA_MAX_DISTANCE:-3.0}
PERCEPTION_WARP_PREPROCESS=${PERCEPTION_WARP_PREPROCESS:-True}

echo "[INFO] teacher_checkpoint=${TEACHER_CHECKPOINT}"
echo "[INFO] run_name=${RUN_NAME} training_name=${TRAINING_NAME}"
echo "[INFO] exp=${EXP} perception=${PERCEPTION_PRESET}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES} nproc=${NPROC}"
echo "[INFO] num_learning_iterations=${NUM_LEARNING_ITERATIONS}"
echo "[INFO] bc_loss_coef=${BC_LOSS_COEF} dagger_loss_coef=${DAGGER_LOSS_COEF} teacher_action_mix_ratio=${TEACHER_ACTION_MIX_RATIO}"
echo "[INFO] ppo_start_epoch=${PPO_START_EPOCH} dagger_end_epoch=${DAGGER_END_EPOCH}"
echo "[INFO] start_at_timestep_zero_prob=${START_AT_TIMESTEP_ZERO_PROB}"

exec env \
  EXP="${EXP}" \
  RUN_NAME="${RUN_NAME}" \
  TRAINING_NAME="${TRAINING_NAME}" \
  TRAINING_PROJECT="${TRAINING_PROJECT}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  NPROC="${NPROC}" \
  TEACHER_OBS_KEYS="${TEACHER_OBS_KEYS}" \
  TEACHER_ACTION_MIX_RATIO="${TEACHER_ACTION_MIX_RATIO}" \
  BC_LOSS_COEF="${BC_LOSS_COEF}" \
  NUM_LEARNING_ITERATIONS="${NUM_LEARNING_ITERATIONS}" \
  PPO_START_EPOCH="${PPO_START_EPOCH}" \
  DAGGER_END_EPOCH="${DAGGER_END_EPOCH}" \
  DAGGER_LOSS_COEF="${DAGGER_LOSS_COEF}" \
  START_AT_TIMESTEP_ZERO_PROB="${START_AT_TIMESTEP_ZERO_PROB}" \
  PAIR_TERRAIN_WITH_MOTION="${PAIR_TERRAIN_WITH_MOTION}" \
  bash "${SCRIPT_DIR}/distill_root_box.sh" "${TEACHER_CHECKPOINT}" \
    "perception:${PERCEPTION_PRESET}" \
    --algo.config.module-dict.actor.input-dim "['actor_obs_root','actor_obs_proprio']" \
    --perception.camera-width="${IMAGE_WIDTH}" \
    --perception.camera-height="${IMAGE_HEIGHT}" \
    --perception.camera-near="${CAMERA_NEAR}" \
    --perception.camera-far="${CAMERA_FAR}" \
    --perception.max-distance="${CAMERA_MAX_DISTANCE}" \
    --perception.camera-warp-preprocess="${PERCEPTION_WARP_PREPROCESS}" \
    "$@"
