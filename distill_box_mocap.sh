#!/usr/bin/env bash
set -euo pipefail

# Distill object-carry generalist -> sim2real student with mocap-access box state.
#
# Student policy observation (actor):
# - actor_obs_torso: sparse target root trajectory command
# - actor_obs_proprio (base_lin_vel, base_ang_vel, dof_pos, dof_vel, actions)
# - actor_obs_box: obj_target_pose_size_b = [obj_pos(3), obj_rot6d(6), obj_scale(3)]
#
# Teacher policy observation:
# - actor_obs_legacy + perception_obs (heightmap by default)

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
RUN_NAME=${RUN_NAME:-g1_w_object_distill_box_mocap}
TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_box_mocap_access_to_mocap_data}
MOTION_DIR=${MOTION_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"}
TEACHER_OBS_KEYS=${TEACHER_OBS_KEYS:-actor_obs_legacy,perception_obs}
_teacher_obs_keys_no_space="$(echo "${TEACHER_OBS_KEYS}" | tr -d '[:space:]')"
case "${_teacher_obs_keys_no_space}" in
  "actor_obs"|"['actor_obs']"|"[\"actor_obs\"]"|"actor_obs,perception_obs"|"['actor_obs','perception_obs']"|"[\"actor_obs\",\"perception_obs\"]")
    echo "[WARN] Remapping TEACHER_OBS_KEYS to actor_obs_legacy,perception_obs to match teacher checkpoint dim/config."
    TEACHER_OBS_KEYS="actor_obs_legacy,perception_obs"
    ;;
esac
PERCEPTION_PRESET=${PERCEPTION_PRESET:-heightmap}
PERCEPTION_INTO_POLICY_MODULES=${PERCEPTION_INTO_POLICY_MODULES:-False}
TEACHER_ACTION_MIX_RATIO=${TEACHER_ACTION_MIX_RATIO:-0.0}
BC_LOSS_COEF=${BC_LOSS_COEF:-1.0}
PPO_START_EPOCH=${PPO_START_EPOCH:--1}
DAGGER_END_EPOCH=${DAGGER_END_EPOCH:--1}
DAGGER_LOSS_COEF=${DAGGER_LOSS_COEF:-10.0}
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
ACTOR_LR=${ACTOR_LR:-5e-5}
CRITIC_LR=${CRITIC_LR:-5e-5}

echo "[INFO] distill mode: mocap-access-to-box"
echo "[INFO] teacher checkpoint: ${TEACHER_CHECKPOINT}"
echo "[INFO] exp=${EXP}"
echo "[INFO] motion_dir=${MOTION_DIR}"
echo "[INFO] teacher_obs_keys=${TEACHER_OBS_KEYS}"
echo "[INFO] perception preset for teacher=${PERCEPTION_PRESET}"
echo "[INFO] perception.inject_into_policy_modules=${PERCEPTION_INTO_POLICY_MODULES} (student stays non-perception)"
echo "[INFO] actor box state: obj_target_pose_size_b = [obj_pos(3), obj_rot6d(6), obj_scale(3)]"
echo "[INFO] actor_lr=${ACTOR_LR} critic_lr=${CRITIC_LR}"
echo "[INFO] pure_dagger_default=True"
echo "[INFO] teacher_action_mix_ratio=${TEACHER_ACTION_MIX_RATIO}"
echo "[INFO] bc_loss_coef=${BC_LOSS_COEF} ppo_start_epoch=${PPO_START_EPOCH} dagger_end_epoch=${DAGGER_END_EPOCH} dagger_loss_coef=${DAGGER_LOSS_COEF}"

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
    "$@"
