#!/usr/bin/env bash
set -euo pipefail

# Goal-conditioned distill (depth perception) with goal-consistent defaults:
# - student actor: actor_obs_torso + actor_obs_proprio + perception_obs
# - sparse object-goal curriculum is enabled
# - external random goals are disabled by default during distill to avoid teacher/goal mismatch
#
# Stage-A default (goal-consistent distill):
#   GOAL_EXTERNAL_PROB_START=0.0
#   GOAL_EXTERNAL_PROB_END=0.0
#
# You can still override any flag from CLI or env.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

DEFAULT_TEACHER_CHECKPOINT=${DEFAULT_TEACHER_CHECKPOINT:-"wandb://zihanw22/boxer/5vlz6pj8/model_10000.pt"}
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${DEFAULT_TEACHER_CHECKPOINT}}"

if [[ $# -gt 0 ]]; then
  if [[ "$1" == wandb://* || "$1" == /* || "$1" == ./* || "$1" == ../* || "$1" == *.pt ]]; then
    TEACHER_CHECKPOINT="$1"
    shift
  fi
fi

if [[ -z "${TEACHER_CHECKPOINT}" ]]; then
  echo "Usage: $0 <teacher_checkpoint.pt> [extra train args...]" >&2
  exit 1
fi

EXP=${EXP:-g1-29dof-wbt-w-object-distill-sparse-goal-cmd}
RUN_NAME=${RUN_NAME:-g1_w_object_distill_goal_box_perception}
TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_goal_box_perception}
TRAINING_PROJECT=${TRAINING_PROJECT:-boxer}

HSSIM_BIN_DIR=${HSSIM_BIN_DIR:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin}
if [[ -d "${HSSIM_BIN_DIR}" ]]; then
  export PATH="${HSSIM_BIN_DIR}:${PATH}"
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
if [[ -z "${NPROC:-}" ]]; then
  IFS=',' read -r -a _visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC=${#_visible_gpus[@]}
fi

TEACHER_OBS_KEYS=${TEACHER_OBS_KEYS:-actor_obs_legacy,perception_obs}
_teacher_obs_keys_no_space="$(echo "${TEACHER_OBS_KEYS}" | tr -d '[:space:]')"
case "${_teacher_obs_keys_no_space}" in
  "actor_obs"|"['actor_obs']"|"[\"actor_obs\"]"|"actor_obs,perception_obs"|"['actor_obs','perception_obs']"|"[\"actor_obs\",\"perception_obs\"]")
    echo "[WARN] Remapping TEACHER_OBS_KEYS to actor_obs_legacy,perception_obs to match teacher checkpoint dim/config."
    TEACHER_OBS_KEYS="actor_obs_legacy,perception_obs"
    ;;
esac

TEACHER_ACTION_MIX_RATIO=${TEACHER_ACTION_MIX_RATIO:-0.0}
BC_LOSS_COEF=${BC_LOSS_COEF:-1.0}
PPO_START_EPOCH=${PPO_START_EPOCH:-0}
DAGGER_END_EPOCH=${DAGGER_END_EPOCH:-10000}
DAGGER_LOSS_COEF=${DAGGER_LOSS_COEF:-1.0}
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
PERCEPTION_PRESET=${PERCEPTION_PRESET:-camera_depth_d435i}

IMAGE_WIDTH=${IMAGE_WIDTH:-17}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-17}
CAMERA_NEAR=${CAMERA_NEAR:-0.001}
CAMERA_FAR=${CAMERA_FAR:-3.0}
CAMERA_MAX_DISTANCE=${CAMERA_MAX_DISTANCE:-3.0}
PERCEPTION_WARP_PREPROCESS=${PERCEPTION_WARP_PREPROCESS:-False}

# Sparse object-goal curriculum knobs.
GOAL_CLIP_DELTA_MIN_STEPS=${GOAL_CLIP_DELTA_MIN_STEPS:-30}
GOAL_CLIP_DELTA_MAX_STEPS=${GOAL_CLIP_DELTA_MAX_STEPS:-180}
GOAL_EXTERNAL_PROB_START=${GOAL_EXTERNAL_PROB_START:-0.0}
GOAL_EXTERNAL_PROB_END=${GOAL_EXTERNAL_PROB_END:-0.0}
GOAL_EXTERNAL_PROB_RAMP_RESETS=${GOAL_EXTERNAL_PROB_RAMP_RESETS:-500000}
GOAL_EVAL_EXTERNAL_PROB=${GOAL_EVAL_EXTERNAL_PROB:-0.0}

echo "[INFO] distill mode: goal-box perception"
echo "[INFO] teacher checkpoint: ${TEACHER_CHECKPOINT}"
echo "[INFO] exp=${EXP} perception=${PERCEPTION_PRESET}"
echo "[INFO] training_project=${TRAINING_PROJECT}"
echo "[INFO] sparse goal: delta=[${GOAL_CLIP_DELTA_MIN_STEPS}, ${GOAL_CLIP_DELTA_MAX_STEPS}]"
echo "[INFO] sparse goal external prob: train ${GOAL_EXTERNAL_PROB_START} -> ${GOAL_EXTERNAL_PROB_END}, eval=${GOAL_EVAL_EXTERNAL_PROB}"
echo "[INFO] student actor uses actor_obs_torso + actor_obs_proprio + perception_obs"

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
  PPO_START_EPOCH="${PPO_START_EPOCH}" \
  DAGGER_END_EPOCH="${DAGGER_END_EPOCH}" \
  DAGGER_LOSS_COEF="${DAGGER_LOSS_COEF}" \
  PAIR_TERRAIN_WITH_MOTION="${PAIR_TERRAIN_WITH_MOTION}" \
  bash "${SCRIPT_DIR}/distill_torso_box.sh" "${TEACHER_CHECKPOINT}" \
    "perception:${PERCEPTION_PRESET}" \
    --algo.config.module-dict.actor.input-dim "['actor_obs_torso','actor_obs_proprio']" \
    --perception.camera-width="${IMAGE_WIDTH}" \
    --perception.camera-height="${IMAGE_HEIGHT}" \
    --perception.camera-near="${CAMERA_NEAR}" \
    --perception.camera-far="${CAMERA_FAR}" \
    --perception.max-distance="${CAMERA_MAX_DISTANCE}" \
    --perception.camera-warp-preprocess="${PERCEPTION_WARP_PREPROCESS}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.enabled=True \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.clip-goal-delta-min-steps="${GOAL_CLIP_DELTA_MIN_STEPS}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.clip-goal-delta-max-steps="${GOAL_CLIP_DELTA_MAX_STEPS}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-prob-start="${GOAL_EXTERNAL_PROB_START}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-prob-end="${GOAL_EXTERNAL_PROB_END}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-prob-ramp-resets="${GOAL_EXTERNAL_PROB_RAMP_RESETS}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.eval-external-goal-prob="${GOAL_EVAL_EXTERNAL_PROB}" \
    "$@"
