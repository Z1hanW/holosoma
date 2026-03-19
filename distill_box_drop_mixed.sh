#!/usr/bin/env bash
set -euo pipefail

# Mixed sparse-goal box-drop distillation on OMOMO with depth perception.
#
# Student policy observation (actor):
# - actor_obs_proprio: proprio history
# - actor_obs_drop_mixed: [goal_dx, goal_dy, goal_dyaw, picked_flag]
# - perception_obs: camera depth
#
# Mixed supervision:
# - clip-consistent sparse goals keep DAgger supervision
# - external sparse goals are trained with PPO/task rewards only

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

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

EXP=${EXP:-g1-29dof-wbt-w-object-distill-sparse-goal-mixed}
RUN_NAME=${RUN_NAME:-g1_w_object_distill_box_drop_mixed}
TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_box_drop_mixed}
TRAINING_PROJECT=${TRAINING_PROJECT:-boxer}

if [[ -n "${POSITIONAL_RUN_NAME}" ]]; then
  RUN_NAME="${POSITIONAL_RUN_NAME}"
fi

HSSIM_BIN_DIR=${HSSIM_BIN_DIR:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin}
if [[ -d "${HSSIM_BIN_DIR}" ]]; then
  export PATH="${HSSIM_BIN_DIR}:${PATH}"
fi
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
if [[ -z "${NPROC:-}" ]]; then
  IFS=',' read -r -a _visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC=${#_visible_gpus[@]}
fi

DEFAULT_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"
MOTION_DIR=${MOTION_DIR:-"${DEFAULT_MOTION_DIR}"}

TEACHER_OBS_KEYS=${TEACHER_OBS_KEYS:-actor_obs_legacy,perception_obs}
TEACHER_ACTION_MIX_RATIO=${TEACHER_ACTION_MIX_RATIO:-0.0}
BC_LOSS_COEF=${BC_LOSS_COEF:-1.0}
NUM_LEARNING_ITERATIONS=${NUM_LEARNING_ITERATIONS:-4000}
PPO_START_EPOCH=${PPO_START_EPOCH:-0}
DAGGER_END_EPOCH=${DAGGER_END_EPOCH:-2500}
DAGGER_LOSS_COEF=${DAGGER_LOSS_COEF:-5.0}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-1.0}
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
PERCEPTION_PRESET=${PERCEPTION_PRESET:-camera_depth_d435i_17x17}
STUDENT_ACTOR_INPUTS=${STUDENT_ACTOR_INPUTS:-"['actor_obs_proprio','actor_obs_drop_mixed']"}
DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES=${DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES:-True}
MAX_EPISODE_LENGTH_S=${MAX_EPISODE_LENGTH_S:-8.0}

SPARSE_GOAL_ENABLED=${SPARSE_GOAL_ENABLED:-True}
CLIP_GOAL_DELTA_MIN_STEPS=${CLIP_GOAL_DELTA_MIN_STEPS:-45}
CLIP_GOAL_DELTA_MAX_STEPS=${CLIP_GOAL_DELTA_MAX_STEPS:-120}
EXTERNAL_GOAL_PROB_START=${EXTERNAL_GOAL_PROB_START:-0.0}
EXTERNAL_GOAL_PROB_END=${EXTERNAL_GOAL_PROB_END:-0.7}
EXTERNAL_GOAL_PROB_RAMP_RESETS=${EXTERNAL_GOAL_PROB_RAMP_RESETS:-150000}
EVAL_EXTERNAL_GOAL_PROB=${EVAL_EXTERNAL_GOAL_PROB:-1.0}
EXTERNAL_GOAL_RANGE_RAMP_RESETS=${EXTERNAL_GOAL_RANGE_RAMP_RESETS:-${EXTERNAL_GOAL_PROB_RAMP_RESETS}}
EXTERNAL_GOAL_POS_LOCAL_MIN_START=${EXTERNAL_GOAL_POS_LOCAL_MIN_START:-"[0.40, -0.20, 0.185]"}
EXTERNAL_GOAL_POS_LOCAL_MAX_START=${EXTERNAL_GOAL_POS_LOCAL_MAX_START:-"[0.65, 0.20, 0.185]"}
EXTERNAL_GOAL_POS_LOCAL_MIN=${EXTERNAL_GOAL_POS_LOCAL_MIN:-"[0.25, -0.75, 0.185]"}
EXTERNAL_GOAL_POS_LOCAL_MAX=${EXTERNAL_GOAL_POS_LOCAL_MAX:-"[1.00, 0.75, 0.185]"}
EXTERNAL_GOAL_RPY_MIN_START=${EXTERNAL_GOAL_RPY_MIN_START:-"[0.0, 0.0, -0.80]"}
EXTERNAL_GOAL_RPY_MAX_START=${EXTERNAL_GOAL_RPY_MAX_START:-"[0.0, 0.0, 0.80]"}
EXTERNAL_GOAL_RPY_MIN=${EXTERNAL_GOAL_RPY_MIN:-"[0.0, 0.0, -3.1415926]"}
EXTERNAL_GOAL_RPY_MAX=${EXTERNAL_GOAL_RPY_MAX:-"[0.0, 0.0, 3.1415926]"}

IMAGE_WIDTH=${IMAGE_WIDTH:-17}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-17}
CAMERA_NEAR=${CAMERA_NEAR:-0.001}
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
echo "[INFO] sparse_goal_enabled=${SPARSE_GOAL_ENABLED} ext_prob=${EXTERNAL_GOAL_PROB_START}->${EXTERNAL_GOAL_PROB_END}"
echo "[INFO] external_goal_range_xy_start=${EXTERNAL_GOAL_POS_LOCAL_MIN_START} -> ${EXTERNAL_GOAL_POS_LOCAL_MAX_START}"
echo "[INFO] external_goal_range_xy_end=${EXTERNAL_GOAL_POS_LOCAL_MIN} -> ${EXTERNAL_GOAL_POS_LOCAL_MAX}"
echo "[INFO] external_goal_range_yaw_start=${EXTERNAL_GOAL_RPY_MIN_START} -> ${EXTERNAL_GOAL_RPY_MAX_START}"
echo "[INFO] external_goal_range_yaw_end=${EXTERNAL_GOAL_RPY_MIN} -> ${EXTERNAL_GOAL_RPY_MAX}"
echo "[INFO] clip_goal_delta_steps=${CLIP_GOAL_DELTA_MIN_STEPS}-${CLIP_GOAL_DELTA_MAX_STEPS}"
echo "[INFO] start_at_timestep_zero_prob=${START_AT_TIMESTEP_ZERO_PROB}"
echo "[INFO] dagger_ignore_external_goal_samples=${DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES}"
echo "[INFO] max_episode_length_s=${MAX_EPISODE_LENGTH_S}"

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
  PAIR_TERRAIN_WITH_MOTION="${PAIR_TERRAIN_WITH_MOTION}" \
  bash "${SCRIPT_DIR}/distill_root_box.sh" "${TEACHER_CHECKPOINT}" \
    "perception:${PERCEPTION_PRESET}" \
    --algo.config.module-dict.actor.input-dim "${STUDENT_ACTOR_INPUTS}" \
    --algo.config.distill.dagger-ignore-external-goal-samples="${DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.enabled="${SPARSE_GOAL_ENABLED}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.clip-goal-delta-min-steps="${CLIP_GOAL_DELTA_MIN_STEPS}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.clip-goal-delta-max-steps="${CLIP_GOAL_DELTA_MAX_STEPS}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-prob-start="${EXTERNAL_GOAL_PROB_START}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-prob-end="${EXTERNAL_GOAL_PROB_END}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-prob-ramp-resets="${EXTERNAL_GOAL_PROB_RAMP_RESETS}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.eval-external-goal-prob="${EVAL_EXTERNAL_GOAL_PROB}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-range-ramp-resets="${EXTERNAL_GOAL_RANGE_RAMP_RESETS}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-pos-local-min-start "${EXTERNAL_GOAL_POS_LOCAL_MIN_START}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-pos-local-max-start "${EXTERNAL_GOAL_POS_LOCAL_MAX_START}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-pos-local-min "${EXTERNAL_GOAL_POS_LOCAL_MIN}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-pos-local-max "${EXTERNAL_GOAL_POS_LOCAL_MAX}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-rpy-min-start "${EXTERNAL_GOAL_RPY_MIN_START}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-rpy-max-start "${EXTERNAL_GOAL_RPY_MAX_START}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-rpy-min "${EXTERNAL_GOAL_RPY_MIN}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-rpy-max "${EXTERNAL_GOAL_RPY_MAX}" \
    --simulator.config.sim.max_episode_length_s "${MAX_EPISODE_LENGTH_S}" \
    --perception.camera-width="${IMAGE_WIDTH}" \
    --perception.camera-height="${IMAGE_HEIGHT}" \
    --perception.camera-near="${CAMERA_NEAR}" \
    --perception.camera-far="${CAMERA_FAR}" \
    --perception.max-distance="${CAMERA_MAX_DISTANCE}" \
    --perception.camera-warp-preprocess="${PERCEPTION_WARP_PREPROCESS}" \
    "$@"
