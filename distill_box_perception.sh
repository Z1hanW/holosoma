#!/usr/bin/env bash
set -euo pipefail

# Distill object-carry generalist -> sim2real student with depth perception access.
#
# Student policy observation (actor):
# - actor_obs_torso (robot proprio torso terms)
# - perception_obs (camera depth)
# - No actor box state is used by student actor.
#
# Teacher policy observation:
# - actor_obs (full teacher state, includes object terms)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

DEFAULT_TEACHER_CHECKPOINT=${DEFAULT_TEACHER_CHECKPOINT:-"/home/ubuntu/FAR/holosoma/logs/WholeBodyTracking/20260216_214200-g1_29dof_wbt_w_object_generalist-locomotion/model_17000.pt"}
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
RUN_NAME=${RUN_NAME:-g1_w_object_distill_box_perception}
TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_box_perception_access_to_depth}
TEACHER_OBS_KEYS=${TEACHER_OBS_KEYS:-actor_obs}
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
echo "[INFO] student actor uses actor_obs_torso + perception_obs only"

exec env \
  EXP="${EXP}" \
  RUN_NAME="${RUN_NAME}" \
  TRAINING_NAME="${TRAINING_NAME}" \
  TEACHER_OBS_KEYS="${TEACHER_OBS_KEYS}" \
  PAIR_TERRAIN_WITH_MOTION="${PAIR_TERRAIN_WITH_MOTION}" \
  bash "${SCRIPT_DIR}/distill_torso_box.sh" "${TEACHER_CHECKPOINT}" \
    "perception:${PERCEPTION_PRESET}" \
    --algo.config.module_dict.actor.input_dim "['actor_obs_torso']" \
    --perception.camera_width="${IMAGE_WIDTH}" \
    --perception.camera_height="${IMAGE_HEIGHT}" \
    --perception.camera_near="${CAMERA_NEAR}" \
    --perception.camera_far="${CAMERA_FAR}" \
    --perception.max_distance="${CAMERA_MAX_DISTANCE}" \
    "$@"
