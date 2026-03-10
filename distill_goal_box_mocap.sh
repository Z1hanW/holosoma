#!/usr/bin/env bash
set -euo pipefail

# Goal-conditioned distill (mocap box state) with goal-consistent defaults:
# - student actor: actor_obs_torso + actor_obs_proprio + actor_obs_box
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

DEFAULT_TEACHER_CHECKPOINT=${DEFAULT_TEACHER_CHECKPOINT:-"wandb://zihanw22/boxer/kge4jozt/model_12000.pt"}
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
RUN_NAME=${RUN_NAME:-g1_w_object_distill_goal_box_mocap}
TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_goal_box_mocap}
MOTION_DIR=${MOTION_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_sq_carry_aug_mix_ml"}

# Default teacher (kge4jozt/model_12000.pt) uses actor_obs-only input.
# For legacy teachers, override TEACHER_OBS_KEYS explicitly (e.g., actor_obs_legacy,perception_obs).
TEACHER_OBS_KEYS=${TEACHER_OBS_KEYS:-actor_obs}

PERCEPTION_PRESET=${PERCEPTION_PRESET:-heightmap}
PERCEPTION_INTO_POLICY_MODULES=${PERCEPTION_INTO_POLICY_MODULES:-False}
TEACHER_ACTION_MIX_RATIO=${TEACHER_ACTION_MIX_RATIO:-0.0}
BC_LOSS_COEF=${BC_LOSS_COEF:-1.0}
PPO_START_EPOCH=${PPO_START_EPOCH:-0}
DAGGER_END_EPOCH=${DAGGER_END_EPOCH:-10000}
DAGGER_LOSS_COEF=${DAGGER_LOSS_COEF:-1.0}
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
ACTOR_LR=${ACTOR_LR:-5e-5}
CRITIC_LR=${CRITIC_LR:-5e-5}

# Sparse object-goal curriculum knobs.
GOAL_CLIP_DELTA_MIN_STEPS=${GOAL_CLIP_DELTA_MIN_STEPS:-30}
GOAL_CLIP_DELTA_MAX_STEPS=${GOAL_CLIP_DELTA_MAX_STEPS:-180}
GOAL_EXTERNAL_PROB_START=${GOAL_EXTERNAL_PROB_START:-0.0}
GOAL_EXTERNAL_PROB_END=${GOAL_EXTERNAL_PROB_END:-0.0}
GOAL_EXTERNAL_PROB_RAMP_RESETS=${GOAL_EXTERNAL_PROB_RAMP_RESETS:-500000}

echo "[INFO] distill mode: goal-box mocap"
echo "[INFO] teacher checkpoint: ${TEACHER_CHECKPOINT}"
echo "[INFO] exp=${EXP}"
echo "[INFO] motion_dir=${MOTION_DIR}"
echo "[INFO] perception preset for teacher=${PERCEPTION_PRESET}"
echo "[INFO] actor box state: obj_current_pose_size_b + obj_goal_pose_size_b"
echo "[INFO] sparse goal: delta=[${GOAL_CLIP_DELTA_MIN_STEPS}, ${GOAL_CLIP_DELTA_MAX_STEPS}]"
echo "[INFO] sparse goal external prob: train ${GOAL_EXTERNAL_PROB_START} -> ${GOAL_EXTERNAL_PROB_END}"

exec env \
  EXP="${EXP}" \
  RUN_NAME="${RUN_NAME}" \
  TRAINING_NAME="${TRAINING_NAME}" \
  MOTION_DIR="${MOTION_DIR}" \
  PERCEPTION_INTO_POLICY_MODULES="${PERCEPTION_INTO_POLICY_MODULES}" \
  TEACHER_OBS_KEYS="${TEACHER_OBS_KEYS}" \
  TEACHER_ACTION_MIX_RATIO="${TEACHER_ACTION_MIX_RATIO}" \
  BC_LOSS_COEF="${BC_LOSS_COEF}" \
  PPO_START_EPOCH="${PPO_START_EPOCH}" \
  DAGGER_END_EPOCH="${DAGGER_END_EPOCH}" \
  DAGGER_LOSS_COEF="${DAGGER_LOSS_COEF}" \
  PAIR_TERRAIN_WITH_MOTION="${PAIR_TERRAIN_WITH_MOTION}" \
  ACTOR_LR="${ACTOR_LR}" \
  CRITIC_LR="${CRITIC_LR}" \
  bash "${SCRIPT_DIR}/distill_torso_box.sh" "${TEACHER_CHECKPOINT}" \
    "perception:${PERCEPTION_PRESET}" \
    --algo.config.module-dict.actor.input-dim "['actor_obs_torso','actor_obs_proprio','actor_obs_box']" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.enabled=True \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.clip-goal-delta-min-steps="${GOAL_CLIP_DELTA_MIN_STEPS}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.clip-goal-delta-max-steps="${GOAL_CLIP_DELTA_MAX_STEPS}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-prob-start="${GOAL_EXTERNAL_PROB_START}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-prob-end="${GOAL_EXTERNAL_PROB_END}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-prob-ramp-resets="${GOAL_EXTERNAL_PROB_RAMP_RESETS}" \
    "$@"
