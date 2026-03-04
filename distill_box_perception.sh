#!/usr/bin/env bash
set -euo pipefail

# Distill object-carry generalist -> sim2real student with depth perception access.
#
# Student policy observation (actor):
# - actor_obs_torso: sparse target root trajectory command
# - actor_obs_proprio (base_lin_vel, base_ang_vel, dof_pos, dof_vel, actions)
# - perception_obs (camera depth)
# - No actor box state is used by student actor.
#
# Teacher policy observation:
# - actor_obs (full teacher state, includes object terms)

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

EXP=${EXP:-g1-29dof-wbt-w-object-distill-sparse-root-cmd}
RUN_NAME=${RUN_NAME:-g1_w_object_distill_box_perception}
TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_box_perception_access_to_depth}
TEACHER_OBS_KEYS=${TEACHER_OBS_KEYS:-actor_obs_legacy}
TEACHER_ACTION_MIX_RATIO=${TEACHER_ACTION_MIX_RATIO:-0.0}
BC_LOSS_COEF=${BC_LOSS_COEF:-1.0}
PPO_START_EPOCH=${PPO_START_EPOCH:--1}
DAGGER_END_EPOCH=${DAGGER_END_EPOCH:--1}
DAGGER_LOSS_COEF=${DAGGER_LOSS_COEF:-10.0}
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
PERCEPTION_PRESET=${PERCEPTION_PRESET:-camera_depth_d435i}

IMAGE_WIDTH=${IMAGE_WIDTH:-106}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-60}
CAMERA_NEAR=${CAMERA_NEAR:-0.001}
CAMERA_FAR=${CAMERA_FAR:-3.0}
CAMERA_MAX_DISTANCE=${CAMERA_MAX_DISTANCE:-3.0}

echo "[INFO] distill mode: depth-access-no-box-state"
echo "[INFO] teacher checkpoint: ${TEACHER_CHECKPOINT}"
echo "[INFO] exp=${EXP} perception=${PERCEPTION_PRESET}"
echo "[INFO] student actor uses actor_obs_torso + actor_obs_proprio + perception_obs (no actor box state)"
echo "[INFO] pure_dagger_default=True"
echo "[INFO] teacher_action_mix_ratio=${TEACHER_ACTION_MIX_RATIO}"
echo "[INFO] bc_loss_coef=${BC_LOSS_COEF} ppo_start_epoch=${PPO_START_EPOCH} dagger_end_epoch=${DAGGER_END_EPOCH} dagger_loss_coef=${DAGGER_LOSS_COEF}"

exec env \
  EXP="${EXP}" \
  RUN_NAME="${RUN_NAME}" \
  TRAINING_NAME="${TRAINING_NAME}" \
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
    "$@"
