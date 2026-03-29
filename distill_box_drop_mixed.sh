#!/usr/bin/env bash
set -euo pipefail

# Single-run sparse-goal box-drop distillation on OMOMO with depth perception.
#
# Student policy observation (actor):
# - actor_obs_proprio: proprio history
# - actor_obs_drop_command: fixed pickup-frame command [goal_dx, goal_dy, goal_dyaw]
# - perception_obs: camera depth
#
# Single-run curriculum:
# - 0..2000 iters: clip/training-distribution only, PPO 0->0.5, DAgger 1->0.5
# - >=2000 iters: keep 50% envs on training distribution, open command curriculum on the other 50%
# - within the command curriculum, external goals ramp conservatively so training remains teacher-anchored

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

TEACHER_OBS_KEYS_EXPLICIT=0
[[ -n "${TEACHER_OBS_KEYS+x}" ]] && TEACHER_OBS_KEYS_EXPLICIT=1
TEACHER_PERCEPTION_PRESET_EXPLICIT=0
[[ -n "${TEACHER_PERCEPTION_PRESET+x}" ]] && TEACHER_PERCEPTION_PRESET_EXPLICIT=1
TEACHER_PERCEPTION_OBS_KEY_EXPLICIT=0
[[ -n "${TEACHER_PERCEPTION_OBS_KEY+x}" ]] && TEACHER_PERCEPTION_OBS_KEY_EXPLICIT=1
START_AT_TIMESTEP_ZERO_PROB_EXPLICIT=0
[[ -n "${START_AT_TIMESTEP_ZERO_PROB+x}" ]] && START_AT_TIMESTEP_ZERO_PROB_EXPLICIT=1
FREEZE_AT_TIMESTEP_ZERO_PROB_EXPLICIT=0
[[ -n "${FREEZE_AT_TIMESTEP_ZERO_PROB+x}" ]] && FREEZE_AT_TIMESTEP_ZERO_PROB_EXPLICIT=1

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
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}
if [[ -z "${NPROC:-}" ]]; then
  IFS=',' read -r -a _visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC=${#_visible_gpus[@]}
fi
DEFAULT_TOTAL_ENVS=${DEFAULT_TOTAL_ENVS:-98304}
NUM_ENVS=${NUM_ENVS:-${DEFAULT_TOTAL_ENVS}}

DEFAULT_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"
MOTION_DIR=${MOTION_DIR:-"${DEFAULT_MOTION_DIR}"}
FILTER_NON_PLACEMENT_CLIPS=${FILTER_NON_PLACEMENT_CLIPS:-True}
FINAL_PLACEMENT_MAX_DELTA_Z=${FINAL_PLACEMENT_MAX_DELTA_Z:-0.15}

TEACHER_OBS_KEYS=${TEACHER_OBS_KEYS:-actor_obs_legacy,perception_obs}
TEACHER_PERCEPTION_PRESET=${TEACHER_PERCEPTION_PRESET:-none}
TEACHER_PERCEPTION_OBS_KEY=${TEACHER_PERCEPTION_OBS_KEY:-teacher_perception_obs}
TEACHER_COMPAT_PROFILE=${TEACHER_COMPAT_PROFILE:-auto}
TEACHER_COMPAT_NOTES=${TEACHER_COMPAT_NOTES:-}
TEACHER_ACTION_MIX_RATIO=${TEACHER_ACTION_MIX_RATIO:-0.0}
BC_LOSS_COEF=${BC_LOSS_COEF:-1.0}
NUM_LEARNING_ITERATIONS=${NUM_LEARNING_ITERATIONS:-4000}
PPO_START_EPOCH=${PPO_START_EPOCH:-0}
DAGGER_END_EPOCH=${DAGGER_END_EPOCH:-2000}
PPO_TARGET_COEFF=${PPO_TARGET_COEFF:-0.5}
DAGGER_LOSS_COEF=${DAGGER_LOSS_COEF:-1.0}
SCHEDULE_NAME=${SCHEDULE_NAME:-teacher_anchor_then_goal_curriculum}
SCHEDULE_NOTES=${SCHEDULE_NOTES:-"0-2000 clip-only with PPO 0->0.5 and DAgger 1->0.5; >=2000 command_only_env_prob=0.5; external_goal_prob 0.10->0.35; external goal range ramps to final span"}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-1.0}
FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
PERCEPTION_PRESET=${PERCEPTION_PRESET:-camera_depth_d435i_17x17}
STUDENT_ACTOR_INPUTS=${STUDENT_ACTOR_INPUTS:-"['actor_obs_proprio','actor_obs_drop_command']"}
DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES=${DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES:-True}
DAGGER_IGNORE_EPISODE_INITIAL_STEPS=${DAGGER_IGNORE_EPISODE_INITIAL_STEPS:-0}
MAX_EPISODE_LENGTH_S=${MAX_EPISODE_LENGTH_S:-8.0}
RESET_TO_DEFAULT_POSE=${RESET_TO_DEFAULT_POSE:-True}
ENABLE_DEFAULT_POSE_PREPEND=${ENABLE_DEFAULT_POSE_PREPEND:-True}
DEFAULT_POSE_PREPEND_DURATION_S=${DEFAULT_POSE_PREPEND_DURATION_S:-0.5}
ENABLE_DEFAULT_POSE_APPEND=${ENABLE_DEFAULT_POSE_APPEND:-False}
DEFAULT_POSE_APPEND_DURATION_S=${DEFAULT_POSE_APPEND_DURATION_S:-0.0}

SPARSE_GOAL_ENABLED=${SPARSE_GOAL_ENABLED:-True}
CLIP_GOAL_DELTA_MIN_STEPS=${CLIP_GOAL_DELTA_MIN_STEPS:-45}
CLIP_GOAL_DELTA_MAX_STEPS=${CLIP_GOAL_DELTA_MAX_STEPS:-120}
COMMAND_ONLY_ENV_PROB_START=${COMMAND_ONLY_ENV_PROB_START:-0.5}
COMMAND_ONLY_ENV_PROB_END=${COMMAND_ONLY_ENV_PROB_END:-0.5}
COMMAND_ONLY_ENV_PROB_START_ITER=${COMMAND_ONLY_ENV_PROB_START_ITER:-2000}
COMMAND_ONLY_ENV_PROB_END_ITER=${COMMAND_ONLY_ENV_PROB_END_ITER:-2000}
EVAL_COMMAND_ONLY_ENV_PROB=${EVAL_COMMAND_ONLY_ENV_PROB:-1.0}
EXTERNAL_GOAL_PROB_START=${EXTERNAL_GOAL_PROB_START:-0.10}
EXTERNAL_GOAL_PROB_END=${EXTERNAL_GOAL_PROB_END:-0.35}
EXTERNAL_GOAL_PROB_START_ITER=${EXTERNAL_GOAL_PROB_START_ITER:-2000}
EXTERNAL_GOAL_PROB_END_ITER=${EXTERNAL_GOAL_PROB_END_ITER:-${NUM_LEARNING_ITERATIONS}}
EXTERNAL_GOAL_PROB_RAMP_RESETS=${EXTERNAL_GOAL_PROB_RAMP_RESETS:-150000}
EVAL_EXTERNAL_GOAL_PROB=${EVAL_EXTERNAL_GOAL_PROB:-1.0}
EXTERNAL_GOAL_RANGE_RAMP_RESETS=${EXTERNAL_GOAL_RANGE_RAMP_RESETS:-${EXTERNAL_GOAL_PROB_RAMP_RESETS}}
EXTERNAL_GOAL_RANGE_START_ITER=${EXTERNAL_GOAL_RANGE_START_ITER:-2000}
EXTERNAL_GOAL_RANGE_END_ITER=${EXTERNAL_GOAL_RANGE_END_ITER:-${NUM_LEARNING_ITERATIONS}}
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

TEACHER_REF_RUN_ID="5vlz6pj8"
TEACHER_REF_LOCAL_CHECKPOINT="${SCRIPT_DIR}/.teacher_checkpoints/model_24000.pt"
TEACHER_REF_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_sq_aug_mix_ml"
TEACHER_REF_PERCEPTION_PRESET="heightmap"
TEACHER_COMPAT_PROFILE_RESOLVED="${TEACHER_COMPAT_PROFILE}"
TEACHER_COMPAT_NOTES_AUTO=""

append_teacher_compat_note() {
  local note="$1"
  if [[ -z "${note}" ]]; then
    return
  fi
  if [[ -z "${TEACHER_COMPAT_NOTES_AUTO}" ]]; then
    TEACHER_COMPAT_NOTES_AUTO="${note}"
  else
    TEACHER_COMPAT_NOTES_AUTO="${TEACHER_COMPAT_NOTES_AUTO}; ${note}"
  fi
}

if [[ "${TEACHER_COMPAT_PROFILE_RESOLVED}" == "auto" ]]; then
  if [[ "${TEACHER_CHECKPOINT}" == *"${TEACHER_REF_RUN_ID}"* || "${TEACHER_CHECKPOINT}" == "${TEACHER_REF_LOCAL_CHECKPOINT}" ]]; then
    TEACHER_COMPAT_PROFILE_RESOLVED="soft_5vlz6pj8"
  else
    TEACHER_COMPAT_PROFILE_RESOLVED="none"
  fi
fi

case "${TEACHER_COMPAT_PROFILE_RESOLVED}" in
  none)
    ;;
  soft_5vlz6pj8)
    if [[ "${TEACHER_OBS_KEYS_EXPLICIT}" -eq 0 ]]; then
      TEACHER_OBS_KEYS="actor_obs_teacher_compat"
    fi
    if [[ "${TEACHER_PERCEPTION_PRESET_EXPLICIT}" -eq 0 ]]; then
      TEACHER_PERCEPTION_PRESET="${TEACHER_REF_PERCEPTION_PRESET}"
    fi
    if [[ "${TEACHER_PERCEPTION_OBS_KEY_EXPLICIT}" -eq 0 ]]; then
      TEACHER_PERCEPTION_OBS_KEY="teacher_perception_obs"
    fi
    append_teacher_compat_note "teacher_obs_keys defaulted to actor_obs_teacher_compat for exact legacy ordering"
    append_teacher_compat_note "teacher now consumes ${TEACHER_PERCEPTION_PRESET} via ${TEACHER_PERCEPTION_OBS_KEY} instead of reusing student depth perception"
    if [[ "${PERCEPTION_PRESET}" != "${TEACHER_REF_PERCEPTION_PRESET}" ]]; then
      append_teacher_compat_note "student perception kept at ${PERCEPTION_PRESET} to preserve current student structure; teacher used ${TEACHER_REF_PERCEPTION_PRESET}"
    fi
    if [[ "${MOTION_DIR}" != "${TEACHER_REF_MOTION_DIR}" ]]; then
      append_teacher_compat_note "motion_dir kept at ${MOTION_DIR}; teacher used ${TEACHER_REF_MOTION_DIR}"
    fi
    ;;
  *)
    echo "Unknown TEACHER_COMPAT_PROFILE: ${TEACHER_COMPAT_PROFILE_RESOLVED}" >&2
    exit 1
    ;;
esac

if [[ -n "${TEACHER_COMPAT_NOTES_AUTO}" ]]; then
  if [[ -n "${TEACHER_COMPAT_NOTES}" ]]; then
    TEACHER_COMPAT_NOTES="${TEACHER_COMPAT_NOTES}; ${TEACHER_COMPAT_NOTES_AUTO}"
  else
    TEACHER_COMPAT_NOTES="${TEACHER_COMPAT_NOTES_AUTO}"
  fi
fi

if [[ ! -e "${MOTION_DIR}" ]]; then
  echo "MOTION_DIR not found: ${MOTION_DIR}" >&2
  exit 1
fi

if [[ "${FILTER_NON_PLACEMENT_CLIPS}" == "True" || "${FILTER_NON_PLACEMENT_CLIPS}" == "true" || "${FILTER_NON_PLACEMENT_CLIPS}" == "1" ]]; then
  FILTERED_MOTION_DIR=$(
    MOTION_DIR="${MOTION_DIR}" FINAL_PLACEMENT_MAX_DELTA_Z="${FINAL_PLACEMENT_MAX_DELTA_Z}" python - <<'PY'
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

motion_dir = Path(os.environ["MOTION_DIR"]).expanduser().resolve()
threshold = float(os.environ["FINAL_PLACEMENT_MAX_DELTA_Z"])

if not motion_dir.is_dir():
    print(str(motion_dir))
    raise SystemExit(0)

cache_key = hashlib.sha1(f"{motion_dir}:{threshold:.4f}".encode("utf-8")).hexdigest()[:10]
cache_dir = motion_dir.parent / f"{motion_dir.name}_drop_final_{cache_key}"
cache_dir.mkdir(parents=True, exist_ok=True)

kept = 0
excluded = 0
for path in sorted(motion_dir.iterdir()):
    target = cache_dir / path.name
    if path.suffix != ".npz":
        if not target.exists():
            target.symlink_to(path)
        continue

    data = np.load(path)
    if "object_pos_w" not in data:
        keep = True
    else:
        object_z = data["object_pos_w"][:, 2]
        keep = float(object_z[-1] - object_z.min()) <= threshold

    if keep:
        kept += 1
        if not target.exists():
            target.symlink_to(path)
    else:
        excluded += 1
        if target.exists():
            target.unlink()

print(f"{cache_dir}|{kept}|{excluded}")
PY
  )
  MOTION_DIR_FILTERED_PATH="${FILTERED_MOTION_DIR%%|*}"
  FILTERED_MOTION_DIR_STATS="${FILTERED_MOTION_DIR#*|}"
  FILTERED_MOTION_DIR_KEPT="${FILTERED_MOTION_DIR_STATS%%|*}"
  FILTERED_MOTION_DIR_EXCLUDED="${FILTERED_MOTION_DIR_STATS##*|}"
  MOTION_DIR="${MOTION_DIR_FILTERED_PATH}"
fi

echo "[INFO] teacher_checkpoint=${TEACHER_CHECKPOINT}"
echo "[INFO] teacher_compat_profile=${TEACHER_COMPAT_PROFILE_RESOLVED}"
echo "[INFO] teacher_obs_keys=${TEACHER_OBS_KEYS}"
echo "[INFO] teacher_perception_preset=${TEACHER_PERCEPTION_PRESET} teacher_perception_obs_key=${TEACHER_PERCEPTION_OBS_KEY}"
echo "[INFO] run_name=${RUN_NAME} training_name=${TRAINING_NAME}"
echo "[INFO] exp=${EXP} perception=${PERCEPTION_PRESET}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES} nproc=${NPROC} num_envs=${NUM_ENVS}"
echo "[INFO] motion_dir=${MOTION_DIR}"
if [[ -n "${FILTERED_MOTION_DIR_KEPT:-}" ]]; then
  echo "[INFO] motion_filter_non_placement_clips=${FILTER_NON_PLACEMENT_CLIPS} final_placement_max_delta_z=${FINAL_PLACEMENT_MAX_DELTA_Z} kept=${FILTERED_MOTION_DIR_KEPT} excluded=${FILTERED_MOTION_DIR_EXCLUDED}"
fi
echo "[INFO] student_actor_inputs=${STUDENT_ACTOR_INPUTS}"
echo "[INFO] schedule_name=${SCHEDULE_NAME}"
echo "[INFO] schedule_notes=${SCHEDULE_NOTES}"
echo "[INFO] ppo_schedule=${PPO_START_EPOCH}->${DAGGER_END_EPOCH} target=${PPO_TARGET_COEFF} dagger_loss_coef=${DAGGER_LOSS_COEF}"
echo "[INFO] command_only_env_prob=${COMMAND_ONLY_ENV_PROB_START}->${COMMAND_ONLY_ENV_PROB_END} iter=${COMMAND_ONLY_ENV_PROB_START_ITER}->${COMMAND_ONLY_ENV_PROB_END_ITER}"
echo "[INFO] sparse_goal_enabled=${SPARSE_GOAL_ENABLED} ext_prob=${EXTERNAL_GOAL_PROB_START}->${EXTERNAL_GOAL_PROB_END}"
echo "[INFO] external_goal_prob_iter=${EXTERNAL_GOAL_PROB_START_ITER}->${EXTERNAL_GOAL_PROB_END_ITER}"
echo "[INFO] external_goal_range_xy_start=${EXTERNAL_GOAL_POS_LOCAL_MIN_START} -> ${EXTERNAL_GOAL_POS_LOCAL_MAX_START}"
echo "[INFO] external_goal_range_xy_end=${EXTERNAL_GOAL_POS_LOCAL_MIN} -> ${EXTERNAL_GOAL_POS_LOCAL_MAX}"
echo "[INFO] external_goal_range_yaw_start=${EXTERNAL_GOAL_RPY_MIN_START} -> ${EXTERNAL_GOAL_RPY_MAX_START}"
echo "[INFO] external_goal_range_yaw_end=${EXTERNAL_GOAL_RPY_MIN} -> ${EXTERNAL_GOAL_RPY_MAX}"
echo "[INFO] external_goal_range_iter=${EXTERNAL_GOAL_RANGE_START_ITER}->${EXTERNAL_GOAL_RANGE_END_ITER}"
echo "[INFO] clip_goal_delta_steps=${CLIP_GOAL_DELTA_MIN_STEPS}-${CLIP_GOAL_DELTA_MAX_STEPS} (legacy/unused; clip-goal now uses final placement)"
echo "[INFO] start_at_timestep_zero_prob=${START_AT_TIMESTEP_ZERO_PROB}"
echo "[INFO] freeze_at_timestep_zero_prob=${FREEZE_AT_TIMESTEP_ZERO_PROB}"
echo "[INFO] reset_to_default_pose=${RESET_TO_DEFAULT_POSE}"
echo "[INFO] default_pose_prepend=${ENABLE_DEFAULT_POSE_PREPEND} duration_s=${DEFAULT_POSE_PREPEND_DURATION_S} default_pose_append=${ENABLE_DEFAULT_POSE_APPEND} append_duration_s=${DEFAULT_POSE_APPEND_DURATION_S}"
echo "[INFO] dagger_ignore_external_goal_samples=${DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES}"
echo "[INFO] dagger_ignore_episode_initial_steps=${DAGGER_IGNORE_EPISODE_INITIAL_STEPS}"
echo "[INFO] max_episode_length_s=${MAX_EPISODE_LENGTH_S}"
if [[ -n "${TEACHER_COMPAT_NOTES}" ]]; then
  echo "[WARN] teacher_compat_notes=${TEACHER_COMPAT_NOTES}"
fi

exec env \
  EXP="${EXP}" \
  RUN_NAME="${RUN_NAME}" \
  TRAINING_NAME="${TRAINING_NAME}" \
  TRAINING_PROJECT="${TRAINING_PROJECT}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  NPROC="${NPROC}" \
  NUM_ENVS="${NUM_ENVS}" \
  MOTION_DIR="${MOTION_DIR}" \
  TEACHER_OBS_KEYS="${TEACHER_OBS_KEYS}" \
  TEACHER_ACTION_MIX_RATIO="${TEACHER_ACTION_MIX_RATIO}" \
  BC_LOSS_COEF="${BC_LOSS_COEF}" \
  NUM_LEARNING_ITERATIONS="${NUM_LEARNING_ITERATIONS}" \
  PPO_START_EPOCH="${PPO_START_EPOCH}" \
  DAGGER_END_EPOCH="${DAGGER_END_EPOCH}" \
  PPO_TARGET_COEFF="${PPO_TARGET_COEFF}" \
  DAGGER_LOSS_COEF="${DAGGER_LOSS_COEF}" \
  START_AT_TIMESTEP_ZERO_PROB="${START_AT_TIMESTEP_ZERO_PROB}" \
  FREEZE_AT_TIMESTEP_ZERO_PROB="${FREEZE_AT_TIMESTEP_ZERO_PROB}" \
  HOLOSOMA_RESET_TO_DEFAULT_POSE="${RESET_TO_DEFAULT_POSE}" \
  ENABLE_DEFAULT_POSE_PREPEND="${ENABLE_DEFAULT_POSE_PREPEND}" \
  DEFAULT_POSE_PREPEND_DURATION_S="${DEFAULT_POSE_PREPEND_DURATION_S}" \
  ENABLE_DEFAULT_POSE_APPEND="${ENABLE_DEFAULT_POSE_APPEND}" \
  DEFAULT_POSE_APPEND_DURATION_S="${DEFAULT_POSE_APPEND_DURATION_S}" \
  PAIR_TERRAIN_WITH_MOTION="${PAIR_TERRAIN_WITH_MOTION}" \
  bash "${SCRIPT_DIR}/distill_root_box.sh" "${TEACHER_CHECKPOINT}" \
    "perception:${PERCEPTION_PRESET}" \
    --algo.config.module-dict.actor.input-dim "${STUDENT_ACTOR_INPUTS}" \
    --algo.config.distill.schedule-name="${SCHEDULE_NAME}" \
    --algo.config.distill.schedule-notes="${SCHEDULE_NOTES}" \
    --algo.config.distill.teacher-compat-profile="${TEACHER_COMPAT_PROFILE_RESOLVED}" \
    --algo.config.distill.teacher-compat-notes="${TEACHER_COMPAT_NOTES}" \
    --algo.config.distill.teacher-perception-preset="${TEACHER_PERCEPTION_PRESET}" \
    --algo.config.distill.teacher-perception-obs-key="${TEACHER_PERCEPTION_OBS_KEY}" \
    --algo.config.distill.dagger-ignore-external-goal-samples="${DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES}" \
    --algo.config.distill.dagger-ignore-episode-initial-steps="${DAGGER_IGNORE_EPISODE_INITIAL_STEPS}" \
    --algo.config.distill.ppo-target-coeff="${PPO_TARGET_COEFF}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.enabled="${SPARSE_GOAL_ENABLED}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.clip-goal-delta-min-steps="${CLIP_GOAL_DELTA_MIN_STEPS}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.clip-goal-delta-max-steps="${CLIP_GOAL_DELTA_MAX_STEPS}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.command-only-env-prob-start="${COMMAND_ONLY_ENV_PROB_START}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.command-only-env-prob-end="${COMMAND_ONLY_ENV_PROB_END}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.command-only-env-prob-start-iter="${COMMAND_ONLY_ENV_PROB_START_ITER}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.command-only-env-prob-end-iter="${COMMAND_ONLY_ENV_PROB_END_ITER}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.eval-command-only-env-prob="${EVAL_COMMAND_ONLY_ENV_PROB}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-prob-start="${EXTERNAL_GOAL_PROB_START}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-prob-end="${EXTERNAL_GOAL_PROB_END}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-prob-start-iter="${EXTERNAL_GOAL_PROB_START_ITER}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-prob-end-iter="${EXTERNAL_GOAL_PROB_END_ITER}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-prob-ramp-resets="${EXTERNAL_GOAL_PROB_RAMP_RESETS}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.eval-external-goal-prob="${EVAL_EXTERNAL_GOAL_PROB}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-range-ramp-resets="${EXTERNAL_GOAL_RANGE_RAMP_RESETS}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-range-start-iter="${EXTERNAL_GOAL_RANGE_START_ITER}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-range-end-iter="${EXTERNAL_GOAL_RANGE_END_ITER}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-pos-local-min-start "${EXTERNAL_GOAL_POS_LOCAL_MIN_START}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-pos-local-max-start "${EXTERNAL_GOAL_POS_LOCAL_MAX_START}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-pos-local-min "${EXTERNAL_GOAL_POS_LOCAL_MIN}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-pos-local-max "${EXTERNAL_GOAL_POS_LOCAL_MAX}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-rpy-min-start "${EXTERNAL_GOAL_RPY_MIN_START}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-rpy-max-start "${EXTERNAL_GOAL_RPY_MAX_START}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-rpy-min "${EXTERNAL_GOAL_RPY_MIN}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-rpy-max "${EXTERNAL_GOAL_RPY_MAX}" \
    --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob="${FREEZE_AT_TIMESTEP_ZERO_PROB}" \
    --simulator.config.sim.max_episode_length_s "${MAX_EPISODE_LENGTH_S}" \
    --perception.camera-width="${IMAGE_WIDTH}" \
    --perception.camera-height="${IMAGE_HEIGHT}" \
    --perception.camera-near="${CAMERA_NEAR}" \
    --perception.camera-far="${CAMERA_FAR}" \
    --perception.max-distance="${CAMERA_MAX_DISTANCE}" \
    --perception.camera-warp-preprocess="${PERCEPTION_WARP_PREPROCESS}" \
    "$@"
