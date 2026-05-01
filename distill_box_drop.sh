#!/usr/bin/env bash
set -euo pipefail

# Distill an OMOMO box-drop student with depth perception.
#
# Student policy observation (actor):
# - actor/student groups are single-frame, plus actor_obs_actions single-step action
# - actor_obs_drop: final clip object target [dx, dy, dyaw] in the pickup-time pelvis-heading frame
# - perception_obs: camera depth
#
# Teacher policy observation:
# - actor_obs_legacy + perception_obs (single-frame legacy teacher state + perception)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/scripts/gpu_launch_defaults.sh"

DEFAULT_TEACHER_CHECKPOINT=${DEFAULT_TEACHER_CHECKPOINT:-"wandb://zihanw22/boxer/5vlz6pj8/model_24000.pt"}
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
EXP=${EXP:-g1-29dof-wbt-w-object-distill-sparse-root-cmd}
RUN_NAME=${RUN_NAME:-g1_w_object_distill_box_drop}
TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_box_drop_depth}
TRAINING_PROJECT=${TRAINING_PROJECT:-boxer}

if [[ -n "${POSITIONAL_RUN_NAME}" ]]; then
  RUN_NAME="${POSITIONAL_RUN_NAME}"
fi

# Keep launcher self-contained: direct `bash ./distill_box_drop.sh` works out of box.
HSSIM_BIN_DIR=${HSSIM_BIN_DIR:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin}
if [[ -d "${HSSIM_BIN_DIR}" ]]; then
  export PATH="${HSSIM_BIN_DIR}:${PATH}"
fi
CUDA_VISIBLE_DEVICES="$(default_cuda_visible_devices_all "${CUDA_VISIBLE_DEVICES:-}")"
if [[ -z "${NPROC:-}" ]]; then
  NPROC="$(count_cuda_visible_devices "${CUDA_VISIBLE_DEVICES}")"
fi

DEFAULT_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"
MOTION_DIR=${MOTION_DIR:-"${DEFAULT_MOTION_DIR}"}

# Teacher alignment for 5vlz6pj8/model_24000.pt.
TEACHER_OBS_KEYS=${TEACHER_OBS_KEYS:-actor_obs_legacy,perception_obs}
TEACHER_ACTION_MIX_RATIO=${TEACHER_ACTION_MIX_RATIO:-0.0}
BC_LOSS_COEF=${BC_LOSS_COEF:-1.0}
NUM_LEARNING_ITERATIONS=${NUM_LEARNING_ITERATIONS:-40000}
PPO_START_EPOCH=${PPO_START_EPOCH:-1000}
DAGGER_END_EPOCH=${DAGGER_END_EPOCH:-2000}
DAGGER_LOSS_COEF=${DAGGER_LOSS_COEF:-5.0}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.7}
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
PERCEPTION_PRESET=${PERCEPTION_PRESET:-camera_depth_d435i_17x17}
STUDENT_ACTOR_INPUTS=${STUDENT_ACTOR_INPUTS:-"['actor_obs_proprio','actor_obs_actions','actor_obs_drop']"}
VISER_DISTILL_MINIMAL_UI=${VISER_DISTILL_MINIMAL_UI:-1}

# Runtime perception must stay 17x17 to match the teacher's 289-d encoder input.
IMAGE_WIDTH=${IMAGE_WIDTH:-17}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-17}
CAMERA_NEAR=${CAMERA_NEAR:-0.3}
CAMERA_FAR=${CAMERA_FAR:-3.0}
CAMERA_MAX_DISTANCE=${CAMERA_MAX_DISTANCE:-3.0}
PERCEPTION_WARP_PREPROCESS=${PERCEPTION_WARP_PREPROCESS:-True}

if [[ ! -e "${MOTION_DIR}" ]]; then
  echo "MOTION_DIR not found: ${MOTION_DIR}" >&2
  exit 1
fi

echo "[INFO] teacher_checkpoint=${TEACHER_CHECKPOINT}"
echo "[INFO] run_name=${RUN_NAME} training_name=${TRAINING_NAME}"
echo "[INFO] exp=${EXP} perception=${PERCEPTION_PRESET}"
echo "[INFO] motion_dir=${MOTION_DIR}"
echo "[INFO] student_actor_inputs=${STUDENT_ACTOR_INPUTS}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES} nproc=${NPROC}"
echo "[INFO] num_learning_iterations=${NUM_LEARNING_ITERATIONS}"
echo "[INFO] bc_loss_coef=${BC_LOSS_COEF} dagger_loss_coef=${DAGGER_LOSS_COEF} teacher_action_mix_ratio=${TEACHER_ACTION_MIX_RATIO}"
echo "[INFO] ppo_start_epoch=${PPO_START_EPOCH} dagger_end_epoch=${DAGGER_END_EPOCH}"
echo "[INFO] start_at_timestep_zero_prob=${START_AT_TIMESTEP_ZERO_PROB}"
echo "[INFO] viser_distill_minimal_ui=${VISER_DISTILL_MINIMAL_UI}"
echo "[INFO] student actor history=single-frame groups plus explicit actor_obs_actions single-step action; teacher groups=single-frame"

exec env \
  EXP="${EXP}" \
  RUN_NAME="${RUN_NAME}" \
  TRAINING_NAME="${TRAINING_NAME}" \
  TRAINING_PROJECT="${TRAINING_PROJECT}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  NPROC="${NPROC}" \
  MOTION_DIR="${MOTION_DIR}" \
  TEACHER_OBS_KEYS="${TEACHER_OBS_KEYS}" \
  TEACHER_ACTION_MIX_RATIO="${TEACHER_ACTION_MIX_RATIO}" \
  BC_LOSS_COEF="${BC_LOSS_COEF}" \
  NUM_LEARNING_ITERATIONS="${NUM_LEARNING_ITERATIONS}" \
  PPO_START_EPOCH="${PPO_START_EPOCH}" \
  DAGGER_END_EPOCH="${DAGGER_END_EPOCH}" \
  DAGGER_LOSS_COEF="${DAGGER_LOSS_COEF}" \
  START_AT_TIMESTEP_ZERO_PROB="${START_AT_TIMESTEP_ZERO_PROB}" \
  VISER_DISTILL_MINIMAL_UI="${VISER_DISTILL_MINIMAL_UI}" \
  PAIR_TERRAIN_WITH_MOTION="${PAIR_TERRAIN_WITH_MOTION}" \
  bash "${SCRIPT_DIR}/distill_root_box.sh" "${TEACHER_CHECKPOINT}" \
    "perception:${PERCEPTION_PRESET}" \
    --algo.config.module-dict.actor.input-dim "${STUDENT_ACTOR_INPUTS}" \
    --perception.camera-width="${IMAGE_WIDTH}" \
    --perception.camera-height="${IMAGE_HEIGHT}" \
    --perception.camera-near="${CAMERA_NEAR}" \
    --perception.camera-far="${CAMERA_FAR}" \
    --perception.max-distance="${CAMERA_MAX_DISTANCE}" \
    --perception.camera-warp-preprocess="${PERCEPTION_WARP_PREPROCESS}" \
    "$@"
