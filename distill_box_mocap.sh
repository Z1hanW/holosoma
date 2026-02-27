#!/usr/bin/env bash
set -euo pipefail

# Distill object-carry generalist -> sim2real student with mocap-access box state.
#
# Student policy observation (actor):
# - actor_obs_torso
# - actor_obs_box, where:
#   - obj_pos_b: current box position in robot base frame
#   - obj_goal_pos_size_b: final clip goal position + box size in robot base frame
#
# Teacher policy observation:
# - actor_obs (full teacher state)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"


DEFAULT_TEACHER_CHECKPOINT=${DEFAULT_TEACHER_CHECKPOINT:-"/home/ubuntu/FAR/model_17000.pt"}
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

EXP=${EXP:-g1-29dof-wbt-w-object-distill-torso-box-goal}
RUN_NAME=${RUN_NAME:-g1_w_object_distill_box_mocap}
TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_box_mocap_access_to_mocap_data}
TEACHER_OBS_KEYS=${TEACHER_OBS_KEYS:-actor_obs}
TEACHER_ACTION_MIX_RATIO=${TEACHER_ACTION_MIX_RATIO:-0.5}
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
ACTOR_LR=${ACTOR_LR:-5e-5}
CRITIC_LR=${CRITIC_LR:-5e-5}

echo "[INFO] distill mode: mocap-access-to-box"
echo "[INFO] teacher checkpoint: ${TEACHER_CHECKPOINT}"
echo "[INFO] exp=${EXP}"
echo "[INFO] actor box state is in robot base frame (b): obj_pos_b + obj_goal_pos_size_b"
echo "[INFO] actor_lr=${ACTOR_LR} critic_lr=${CRITIC_LR}"
echo "[INFO] teacher_action_mix_ratio=${TEACHER_ACTION_MIX_RATIO}"

exec env \
  EXP="${EXP}" \
  RUN_NAME="${RUN_NAME}" \
  TRAINING_NAME="${TRAINING_NAME}" \
  TEACHER_OBS_KEYS="${TEACHER_OBS_KEYS}" \
  TEACHER_ACTION_MIX_RATIO="${TEACHER_ACTION_MIX_RATIO}" \
  PAIR_TERRAIN_WITH_MOTION="${PAIR_TERRAIN_WITH_MOTION}" \
  ACTOR_LR="${ACTOR_LR}" \
  CRITIC_LR="${CRITIC_LR}" \
  bash "${SCRIPT_DIR}/distill_torso_box.sh" "${TEACHER_CHECKPOINT}" \
    "$@"
