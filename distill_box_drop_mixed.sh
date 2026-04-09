#!/usr/bin/env bash
set -euo pipefail

# Single-run sparse-goal box-drop distillation on OMOMO with depth perception.
#
# Student policy observation (actor):
# - actor_obs_proprio: proprio history
# - actor_obs_drop_command: fixed pickup-frame command [goal_dx, goal_dy]
# - perception_obs: camera depth
#
# Single-run curriculum:
# - 0..2000 iters: clip/training-distribution only, PPO 0->0.5, DAgger 1->0.5
# - >=2000 iters: keep 50% envs on training distribution, open command curriculum on the other 50%
# - within the command curriculum, external goals ramp conservatively so training remains teacher-anchored

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

DEFAULT_TEACHER_CHECKPOINT=${DEFAULT_TEACHER_CHECKPOINT:-"https://wandb.ai/zihanw22/boxer/runs/u5lguxvl"}
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${DEFAULT_TEACHER_CHECKPOINT}}"
POSITIONAL_RUN_NAME=""
DATA_MODE=${DATA_MODE:-mix-naive}
SCHEDULE_VARIANT=${SCHEDULE_VARIANT:-default}
REWARD_VARIANT=${REWARD_VARIANT:-default}
DISTRIBUTION_VARIANT=${DISTRIBUTION_VARIANT:-default}

PYTHON_BIN=${PYTHON_BIN:-python}

parse_wandb_run_url() {
  local ref="$1"
  local clean_ref="${ref%%\?*}"
  if [[ "${clean_ref}" != https://wandb.ai/*/runs/* ]]; then
    return 1
  fi
  local trimmed="${clean_ref#https://wandb.ai/}"
  local entity=""
  local project=""
  local run_id=""
  local explicit_file=""
  IFS='/' read -r -a parts <<< "${trimmed}"
  if [[ "${#parts[@]}" -lt 4 || "${parts[2]}" != "runs" ]]; then
    return 1
  fi
  entity="${parts[0]}"
  project="${parts[1]}"
  run_id="${parts[3]}"
  if [[ -z "${entity}" || -z "${project}" || -z "${run_id}" ]]; then
    return 1
  fi
  if [[ "${#parts[@]}" -ge 6 && "${parts[4]}" == "files" ]]; then
    explicit_file="${trimmed#${entity}/${project}/runs/${run_id}/files/}"
  fi
  printf '%s\t%s\t%s\t%s\n' "${entity}" "${project}" "${run_id}" "${explicit_file}"
}

resolve_remote_wandb_checkpoint_name() {
  local entity="$1"
  local project="$2"
  local run_id="$3"
  "${PYTHON_BIN}" - "${entity}" "${project}" "${run_id}" <<'PY' 2>/dev/null || true
import re
import sys

try:
    import wandb
except Exception:
    sys.exit(0)

entity, project, run_id = sys.argv[1:4]
api = wandb.Api(timeout=30)
run = api.run(f"{entity}/{project}/{run_id}")

def _coerce_int(value):
    try:
        return int(value)
    except Exception:
        return None

summary = getattr(run, "summary", {}) or {}
step_hint = _coerce_int(summary.get("_step"))
if step_hint is None:
    step_hint = _coerce_int(summary.get("global_step"))
if step_hint is None:
    step_hint = _coerce_int(getattr(run, "lastHistoryStep", None))
if step_hint is None:
    step_hint = 0

save_interval = None
cfg = getattr(run, "config", {}) or {}
algo_cfg = cfg.get("algo") if isinstance(cfg, dict) else None
if isinstance(algo_cfg, dict):
    algo_cfg = algo_cfg.get("config")
if isinstance(algo_cfg, dict):
    save_interval = _coerce_int(algo_cfg.get("save_interval"))
if save_interval is None or save_interval <= 0:
    save_interval = 500

pattern = re.compile(r"^model_(\d+)\.pt$")
start_step = max(step_hint - (step_hint % save_interval), 0)
best_step = -1
best_name = ""
for step in range(start_step, -1, -save_interval):
    name = f"model_{step:05d}.pt"
    try:
        file_obj = run.file(name)
    except Exception:
        continue
    size = _coerce_int(getattr(file_obj, "size", None))
    match = pattern.match(getattr(file_obj, "name", ""))
    if match is None or size is None or size <= 0:
        continue
    file_step = int(match.group(1))
    if file_step >= best_step:
        best_step = file_step
        best_name = name
        break

if best_name:
    print(best_name)
PY
}

normalize_checkpoint_ref() {
  local ref="$1"
  if [[ "${ref}" != https://wandb.ai/*/runs/* ]]; then
    echo "${ref}"
    return 0
  fi

  local parsed=""
  local entity=""
  local project=""
  local run_id=""
  local explicit_file=""
  local model_file="${WANDB_MODEL_FILE:-}"

  parsed="$(parse_wandb_run_url "${ref}" || true)"
  if [[ -z "${parsed}" ]]; then
    echo "${ref}"
    return 0
  fi

  IFS=$'\t' read -r entity project run_id explicit_file <<< "${parsed}"
  if [[ -n "${explicit_file}" ]]; then
    model_file="${explicit_file}"
  elif [[ -z "${model_file}" ]]; then
    model_file="$(resolve_remote_wandb_checkpoint_name "${entity}" "${project}" "${run_id}")"
    if [[ -n "${model_file}" ]]; then
      echo "[INFO] Resolved wandb run URL to latest remote checkpoint: ${model_file}" >&2
    fi
  fi

  if [[ -z "${model_file}" ]]; then
    echo "[ERROR] Could not determine checkpoint for W&B run URL: ${ref}" >&2
    echo "[ERROR] Pass a /files/<checkpoint>.pt URL or set WANDB_MODEL_FILE." >&2
    return 2
  fi

  echo "wandb://${entity}/${project}/${run_id}/${model_file}"
}

while [[ $# -gt 0 ]]; do
  first_arg_normalized=$(echo "$1" | tr '[:upper:]' '[:lower:]')
  case "${first_arg_normalized}" in
    pure-sd|pure-ds)
      DATA_MODE="pure-sd"
      shift
      ;;
    pure-real|pure-omomo)
      DATA_MODE="pure-real"
      shift
      ;;
    omomo-debug|omomo_debug)
      DATA_MODE="omomo-debug"
      shift
      ;;
    mix-naive)
      DATA_MODE="mix-naive"
      shift
      ;;
    mix-curriculum|mix-clean-noisy|mix-curr)
      DATA_MODE="mix-curriculum"
      shift
      ;;
    default)
      SCHEDULE_VARIANT="default"
      shift
      ;;
    dag_first|dag-first|dagger-first)
      SCHEDULE_VARIANT="dag_first"
      shift
      ;;
    pickup_reward|pickup-reward|pickup)
      REWARD_VARIANT="pickup_reward"
      shift
      ;;
    stable_half|stable-half|half_stable|half-stable)
      DISTRIBUTION_VARIANT="stable_half"
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -gt 0 ]]; then
  if [[ "$1" == wandb://* || "$1" == https://wandb.ai/*/runs/* || "$1" == /* || "$1" == ./* || "$1" == ../* || "$1" == *.pt ]]; then
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
TEACHER_ACTOR_OBS_HISTORY_LENGTH_EXPLICIT=0
[[ -n "${TEACHER_ACTOR_OBS_HISTORY_LENGTH+x}" ]] && TEACHER_ACTOR_OBS_HISTORY_LENGTH_EXPLICIT=1
TEACHER_ACTION_MIX_RATIO_EXPLICIT=0
[[ -n "${TEACHER_ACTION_MIX_RATIO+x}" ]] && TEACHER_ACTION_MIX_RATIO_EXPLICIT=1
TEACHER_ACTION_MIX_RATIO_START_EXPLICIT=0
[[ -n "${TEACHER_ACTION_MIX_RATIO_START+x}" ]] && TEACHER_ACTION_MIX_RATIO_START_EXPLICIT=1
TEACHER_ACTION_MIX_RATIO_END_EXPLICIT=0
[[ -n "${TEACHER_ACTION_MIX_RATIO_END+x}" ]] && TEACHER_ACTION_MIX_RATIO_END_EXPLICIT=1
TEACHER_ACTION_MIX_RATIO_END_ITERATION_EXPLICIT=0
[[ -n "${TEACHER_ACTION_MIX_RATIO_END_ITERATION+x}" ]] && TEACHER_ACTION_MIX_RATIO_END_ITERATION_EXPLICIT=1
PPO_START_EPOCH_EXPLICIT=0
[[ -n "${PPO_START_EPOCH+x}" ]] && PPO_START_EPOCH_EXPLICIT=1
DAGGER_END_EPOCH_EXPLICIT=0
[[ -n "${DAGGER_END_EPOCH+x}" ]] && DAGGER_END_EPOCH_EXPLICIT=1
SCHEDULE_NAME_EXPLICIT=0
[[ -n "${SCHEDULE_NAME+x}" ]] && SCHEDULE_NAME_EXPLICIT=1
SCHEDULE_NOTES_EXPLICIT=0
[[ -n "${SCHEDULE_NOTES+x}" ]] && SCHEDULE_NOTES_EXPLICIT=1
START_AT_TIMESTEP_ZERO_PROB_EXPLICIT=0
[[ -n "${START_AT_TIMESTEP_ZERO_PROB+x}" ]] && START_AT_TIMESTEP_ZERO_PROB_EXPLICIT=1
START_AT_TIMESTEP_ZERO_PROB_END_EXPLICIT=0
[[ -n "${START_AT_TIMESTEP_ZERO_PROB_END+x}" ]] && START_AT_TIMESTEP_ZERO_PROB_END_EXPLICIT=1
FREEZE_AT_TIMESTEP_ZERO_PROB_EXPLICIT=0
[[ -n "${FREEZE_AT_TIMESTEP_ZERO_PROB+x}" ]] && FREEZE_AT_TIMESTEP_ZERO_PROB_EXPLICIT=1
FREEZE_AT_TIMESTEP_ZERO_PROB_END_EXPLICIT=0
[[ -n "${FREEZE_AT_TIMESTEP_ZERO_PROB_END+x}" ]] && FREEZE_AT_TIMESTEP_ZERO_PROB_END_EXPLICIT=1
COMMAND_ONLY_ENV_PROB_END_EXPLICIT=0
[[ -n "${COMMAND_ONLY_ENV_PROB_END+x}" ]] && COMMAND_ONLY_ENV_PROB_END_EXPLICIT=1
EXTERNAL_GOAL_PROB_END_EXPLICIT=0
[[ -n "${EXTERNAL_GOAL_PROB_END+x}" ]] && EXTERNAL_GOAL_PROB_END_EXPLICIT=1
DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES_EXPLICIT=0
[[ -n "${DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES+x}" ]] && DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES_EXPLICIT=1
EXP_EXPLICIT=0
[[ -n "${EXP+x}" ]] && EXP_EXPLICIT=1
RUN_NAME_EXPLICIT=0
[[ -n "${RUN_NAME+x}" ]] && RUN_NAME_EXPLICIT=1
if [[ -n "${POSITIONAL_RUN_NAME}" ]]; then
  RUN_NAME_EXPLICIT=1
fi
TRAINING_NAME_EXPLICIT=0
[[ -n "${TRAINING_NAME+x}" ]] && TRAINING_NAME_EXPLICIT=1

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
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}
if [[ -z "${NPROC:-}" ]]; then
  IFS=',' read -r -a _visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC=${#_visible_gpus[@]}
fi
DEFAULT_TOTAL_ENVS=${DEFAULT_TOTAL_ENVS:-7168}
NUM_ENVS=${NUM_ENVS:-${DEFAULT_TOTAL_ENVS}}

DS_DATA_ROOT=${DS_DATA_ROOT:-"${SCRIPT_DIR}/data/ds_box_data"}
DEFAULT_DS_PREPARED_MOTION_DIR="${DS_DATA_ROOT}/train_g1_w_obj_prepared"
DEFAULT_MIX_NAIVE_MOTION_DIR="${DS_DATA_ROOT}/train_g1_w_obj_prepared_plus_omomo_orig"
MOTION_DIR_FROM_ENV=0
if [[ -n "${MOTION_DIR+x}" ]]; then
  MOTION_DIR_FROM_ENV=1
fi
FILTER_NON_PLACEMENT_CLIPS=${FILTER_NON_PLACEMENT_CLIPS:-True}
FINAL_PLACEMENT_MAX_DELTA_Z=${FINAL_PLACEMENT_MAX_DELTA_Z:-0.15}
MIX_CURRICULUM_OMOMO_PREFIXES=${MIX_CURRICULUM_OMOMO_PREFIXES:-'["sub"]'}
MIX_CURRICULUM_STAGE_START_ITERATIONS=${MIX_CURRICULUM_STAGE_START_ITERATIONS:-'[0, 1500, 2000, 2500, 3000, 3500]'}
MIX_CURRICULUM_OMOMO_PROBABILITIES=${MIX_CURRICULUM_OMOMO_PROBABILITIES:-'[1.0, 0.9, 0.8, 0.7, 0.6, 0.5]'}
PURE_REAL_OMOMO_PREFIXES=${PURE_REAL_OMOMO_PREFIXES:-'["sub"]'}
OMOMO_DEBUG_PREFIXES=${OMOMO_DEBUG_PREFIXES:-'["sub"]'}
OBJECT_SPEC_PATH=${OBJECT_SPEC_PATH:-""}
ASSERT_ACTIVE_MULTI_URDF=${ASSERT_ACTIVE_MULTI_URDF:-auto}

TEACHER_OBS_KEYS=${TEACHER_OBS_KEYS:-actor_obs_legacy,perception_obs}
TEACHER_PERCEPTION_PRESET=${TEACHER_PERCEPTION_PRESET:-none}
TEACHER_PERCEPTION_OBS_KEY=${TEACHER_PERCEPTION_OBS_KEY:-teacher_perception_obs}
CRITIC_PERCEPTION_PRESET=${CRITIC_PERCEPTION_PRESET:-heightmap}
CRITIC_PERCEPTION_OBS_KEY=${CRITIC_PERCEPTION_OBS_KEY:-critic_perception_obs}
TEACHER_ACTOR_OBS_HISTORY_LENGTH=${TEACHER_ACTOR_OBS_HISTORY_LENGTH:-}
TEACHER_COMPAT_PROFILE=${TEACHER_COMPAT_PROFILE:-auto}
TEACHER_COMPAT_NOTES=${TEACHER_COMPAT_NOTES:-}
TEACHER_ACTION_MIX_RATIO=${TEACHER_ACTION_MIX_RATIO:-0.0}
TEACHER_ACTION_MIX_RATIO_START=${TEACHER_ACTION_MIX_RATIO_START:-}
TEACHER_ACTION_MIX_RATIO_END=${TEACHER_ACTION_MIX_RATIO_END:-}
TEACHER_ACTION_MIX_RATIO_END_ITERATION=${TEACHER_ACTION_MIX_RATIO_END_ITERATION:-}
BC_LOSS_COEF=${BC_LOSS_COEF:-1.0}
NUM_LEARNING_ITERATIONS=${NUM_LEARNING_ITERATIONS:-4000}
PPO_START_EPOCH=${PPO_START_EPOCH:-0}
DAGGER_END_EPOCH=${DAGGER_END_EPOCH:-3000}
PPO_TARGET_COEFF=${PPO_TARGET_COEFF:-0.3}
DAGGER_LOSS_COEF=${DAGGER_LOSS_COEF:-1.0}
SCHEDULE_NAME=${SCHEDULE_NAME:-teacher_anchor_then_goal_curriculum_v2}
SCHEDULE_NOTES=${SCHEDULE_NOTES:-"0-700 teacher rollout mix decays 0.7->0.0. 0-2500 teacher-anchored clip-only; PPO ramps 0->0.3 over 0-3000 while DAgger weight stays dominant. 2500-3500 command_only_env_prob ramps 0->0.5. 2500-end external_goal_prob ramps 0->0.25 and reset curriculum ramps start_at_zero 0.2->1.0 / freeze_at_zero 0.95->0.0. Goal range ramps with the same delayed schedule."}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.2}
START_AT_TIMESTEP_ZERO_PROB_END=${START_AT_TIMESTEP_ZERO_PROB_END:-1.0}
START_AT_TIMESTEP_ZERO_PROB_START_ITER=${START_AT_TIMESTEP_ZERO_PROB_START_ITER:-2500}
START_AT_TIMESTEP_ZERO_PROB_END_ITER=${START_AT_TIMESTEP_ZERO_PROB_END_ITER:-${NUM_LEARNING_ITERATIONS}}
FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.95}
FREEZE_AT_TIMESTEP_ZERO_PROB_END=${FREEZE_AT_TIMESTEP_ZERO_PROB_END:-0.0}
FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=${FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER:-2500}
FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=${FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER:-${NUM_LEARNING_ITERATIONS}}
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
PERCEPTION_PRESET=${PERCEPTION_PRESET:-camera_depth_d435i_defm_regnet_y_800mf}
STUDENT_ACTOR_INPUTS=${STUDENT_ACTOR_INPUTS:-"['actor_obs_proprio','actor_obs_drop_command']"}
DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES=${DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES:-True}
DAGGER_MATCH_STD=${DAGGER_MATCH_STD:-True}
ENTROPY_COEF=${ENTROPY_COEF:-0.0}
DAGGER_IGNORE_EPISODE_INITIAL_STEPS=${DAGGER_IGNORE_EPISODE_INITIAL_STEPS:-0}
MAX_EPISODE_LENGTH_S=${MAX_EPISODE_LENGTH_S:-8.0}
# Keep non-zero clip starts motion-aligned; t=0 starts still train from default pose via runtime prepend.
RESET_TO_DEFAULT_POSE=${RESET_TO_DEFAULT_POSE:-False}
ENABLE_DEFAULT_POSE_PREPEND=${ENABLE_DEFAULT_POSE_PREPEND:-True}
DEFAULT_POSE_PREPEND_DURATION_S=${DEFAULT_POSE_PREPEND_DURATION_S:-0.5}
ENABLE_DEFAULT_POSE_APPEND=${ENABLE_DEFAULT_POSE_APPEND:-False}
DEFAULT_POSE_APPEND_DURATION_S=${DEFAULT_POSE_APPEND_DURATION_S:-0.0}
# Distill UI: keep only goal/reset-box/clip + essential sim controls by default.
VISER_DISTILL_MINIMAL_UI=${VISER_DISTILL_MINIMAL_UI:-1}
# Distill mixed launcher: hide Viser track/target keypoints by default.
VISER_SHOW_TARGET_KEYPOINTS=${VISER_SHOW_TARGET_KEYPOINTS:-0}

if [[ "${START_AT_TIMESTEP_ZERO_PROB_EXPLICIT}" -eq 1 && "${START_AT_TIMESTEP_ZERO_PROB_END_EXPLICIT}" -eq 0 ]]; then
  START_AT_TIMESTEP_ZERO_PROB_END="${START_AT_TIMESTEP_ZERO_PROB}"
fi
if [[ "${FREEZE_AT_TIMESTEP_ZERO_PROB_EXPLICIT}" -eq 1 && "${FREEZE_AT_TIMESTEP_ZERO_PROB_END_EXPLICIT}" -eq 0 ]]; then
  FREEZE_AT_TIMESTEP_ZERO_PROB_END="${FREEZE_AT_TIMESTEP_ZERO_PROB}"
fi
if [[ "${TEACHER_ACTION_MIX_RATIO_EXPLICIT}" -eq 0 && "${TEACHER_ACTION_MIX_RATIO_START_EXPLICIT}" -eq 0 && "${TEACHER_ACTION_MIX_RATIO_END_EXPLICIT}" -eq 0 && "${TEACHER_ACTION_MIX_RATIO_END_ITERATION_EXPLICIT}" -eq 0 ]]; then
  TEACHER_ACTION_MIX_RATIO="0.0"
  TEACHER_ACTION_MIX_RATIO_START="0.7"
  TEACHER_ACTION_MIX_RATIO_END="0.0"
  TEACHER_ACTION_MIX_RATIO_END_ITERATION="700"
fi

case "${SCHEDULE_VARIANT}" in
  default)
    ;;
  dag_first)
    if [[ "${PPO_START_EPOCH_EXPLICIT}" -eq 0 ]]; then
      PPO_START_EPOCH=2000
    fi
    if [[ "${DAGGER_END_EPOCH_EXPLICIT}" -eq 0 ]]; then
      DAGGER_END_EPOCH=3000
    fi
    if [[ "${SCHEDULE_NAME_EXPLICIT}" -eq 0 ]]; then
      SCHEDULE_NAME="teacher_anchor_then_goal_curriculum_v2_dag_first"
    fi
    if [[ "${SCHEDULE_NOTES_EXPLICIT}" -eq 0 ]]; then
      SCHEDULE_NOTES="0-700 teacher rollout mix decays 0.7->0.0. 0-2000 pure DAgger with PPO disabled. 2000-3000 PPO ramps 0->0.3 while DAgger stays dominant. 2500-3500 command_only_env_prob ramps 0->0.5. 2500-end external_goal_prob ramps 0->0.25 and reset curriculum ramps start_at_zero 0.2->1.0 / freeze_at_zero 0.95->0.0. Goal range ramps with the same delayed schedule."
    fi
    ;;
  *)
    echo "[ERROR] Unsupported SCHEDULE_VARIANT='${SCHEDULE_VARIANT}'. Use one of: default, dag_first" >&2
    exit 2
    ;;
esac

case "${REWARD_VARIANT}" in
  default)
    ;;
  pickup_reward)
    if [[ "${EXP_EXPLICIT}" -eq 0 ]]; then
      EXP="g1-29dof-wbt-w-object-distill-sparse-goal-mixed-pickup"
    fi
    if [[ "${RUN_NAME_EXPLICIT}" -eq 0 ]]; then
      RUN_NAME="g1_w_object_distill_box_drop_mixed_pickup"
    fi
    if [[ "${TRAINING_NAME_EXPLICIT}" -eq 0 ]]; then
      TRAINING_NAME="g1_29dof_wbt_w_object_distill_box_drop_mixed_pickup"
    fi
    ;;
  *)
    echo "[ERROR] Unsupported REWARD_VARIANT='${REWARD_VARIANT}'. Use one of: default, pickup_reward" >&2
    exit 2
    ;;
esac

DATA_MODE=$(echo "${DATA_MODE}" | tr '[:upper:]' '[:lower:]')
case "${DATA_MODE}" in
  pure-ds)
    DATA_MODE="pure-sd"
    ;;
  pure-omomo)
    DATA_MODE="pure-real"
    ;;
  omomo_debug)
    DATA_MODE="omomo-debug"
    ;;
  mix-clean-noisy|mix-curr)
    DATA_MODE="mix-curriculum"
    ;;
esac
case "${DATA_MODE}" in
  pure-sd|pure-real|omomo-debug|mix-naive|mix-curriculum)
    ;;
  *)
    echo "[ERROR] Unsupported DATA_MODE='${DATA_MODE}'. Use one of: pure-sd, pure-real, omomo-debug, mix-naive, mix-curriculum" >&2
    exit 2
    ;;
esac

case "${DATA_MODE}" in
  pure-sd)
    MODE_DEFAULT_MOTION_DIR="${DEFAULT_DS_PREPARED_MOTION_DIR}"
    ;;
  omomo-debug)
    MODE_DEFAULT_MOTION_DIR="${DEFAULT_MIX_NAIVE_MOTION_DIR}"
    ;;
  pure-real|mix-naive|mix-curriculum)
    MODE_DEFAULT_MOTION_DIR="${DEFAULT_MIX_NAIVE_MOTION_DIR}"
    ;;
esac

if [[ "${DATA_MODE}" == "omomo-debug" ]]; then
  if [[ "${TRAINING_NAME_EXPLICIT}" -eq 0 ]]; then
    if [[ "${REWARD_VARIANT}" == "pickup_reward" ]]; then
      TRAINING_NAME="g1_29dof_wbt_w_object_distill_box_drop_omomo_debug_pickup"
    else
      TRAINING_NAME="g1_29dof_wbt_w_object_distill_box_drop_omomo_debug"
    fi
  fi
fi

if [[ "${RUN_NAME_EXPLICIT}" -eq 0 ]]; then
  RUN_NAME="${DATA_MODE}"
fi

MOTION_DIR=${MOTION_DIR:-"${MODE_DEFAULT_MOTION_DIR}"}

if [[ -z "${OBJECT_SPEC_PATH}" ]]; then
  default_map="${MOTION_DIR}/_clip_object_urdf_map.json"
  if [[ -f "${default_map}" ]]; then
    OBJECT_SPEC_PATH="${default_map}"
  fi
fi

SPARSE_GOAL_ENABLED=${SPARSE_GOAL_ENABLED:-True}
CLIP_GOAL_DELTA_MIN_STEPS=${CLIP_GOAL_DELTA_MIN_STEPS:-45}
CLIP_GOAL_DELTA_MAX_STEPS=${CLIP_GOAL_DELTA_MAX_STEPS:-120}
COMMAND_ONLY_ENV_PROB_START=${COMMAND_ONLY_ENV_PROB_START:-0.0}
COMMAND_ONLY_ENV_PROB_END=${COMMAND_ONLY_ENV_PROB_END:-0.5}
COMMAND_ONLY_ENV_PROB_START_ITER=${COMMAND_ONLY_ENV_PROB_START_ITER:-2500}
COMMAND_ONLY_ENV_PROB_END_ITER=${COMMAND_ONLY_ENV_PROB_END_ITER:-3500}
EVAL_COMMAND_ONLY_ENV_PROB=${EVAL_COMMAND_ONLY_ENV_PROB:-1.0}
EXTERNAL_GOAL_PROB_START=${EXTERNAL_GOAL_PROB_START:-0.0}
EXTERNAL_GOAL_PROB_END=${EXTERNAL_GOAL_PROB_END:-0.25}
EXTERNAL_GOAL_PROB_START_ITER=${EXTERNAL_GOAL_PROB_START_ITER:-2500}
EXTERNAL_GOAL_PROB_END_ITER=${EXTERNAL_GOAL_PROB_END_ITER:-${NUM_LEARNING_ITERATIONS}}
EXTERNAL_GOAL_PROB_RAMP_RESETS=${EXTERNAL_GOAL_PROB_RAMP_RESETS:-150000}
EVAL_EXTERNAL_GOAL_PROB=${EVAL_EXTERNAL_GOAL_PROB:-1.0}
EXTERNAL_GOAL_RANGE_RAMP_RESETS=${EXTERNAL_GOAL_RANGE_RAMP_RESETS:-${EXTERNAL_GOAL_PROB_RAMP_RESETS}}
EXTERNAL_GOAL_RANGE_START_ITER=${EXTERNAL_GOAL_RANGE_START_ITER:-2500}
EXTERNAL_GOAL_RANGE_END_ITER=${EXTERNAL_GOAL_RANGE_END_ITER:-${NUM_LEARNING_ITERATIONS}}
EXTERNAL_GOAL_POS_LOCAL_MIN_START=${EXTERNAL_GOAL_POS_LOCAL_MIN_START:-"[0.40, -0.20, 0.185]"}
EXTERNAL_GOAL_POS_LOCAL_MAX_START=${EXTERNAL_GOAL_POS_LOCAL_MAX_START:-"[0.65, 0.20, 0.185]"}
EXTERNAL_GOAL_POS_LOCAL_MIN=${EXTERNAL_GOAL_POS_LOCAL_MIN:-"[0.25, -0.75, 0.185]"}
EXTERNAL_GOAL_POS_LOCAL_MAX=${EXTERNAL_GOAL_POS_LOCAL_MAX:-"[1.00, 0.75, 0.185]"}

case "${DISTRIBUTION_VARIANT}" in
  default)
    ;;
  stable_half)
    if [[ "${COMMAND_ONLY_ENV_PROB_END_EXPLICIT}" -eq 0 ]]; then
      COMMAND_ONLY_ENV_PROB_END="0.5"
    fi
    if [[ "${EXTERNAL_GOAL_PROB_END_EXPLICIT}" -eq 0 ]]; then
      EXTERNAL_GOAL_PROB_END="0.5"
    fi
    if [[ "${START_AT_TIMESTEP_ZERO_PROB_EXPLICIT}" -eq 0 ]]; then
      START_AT_TIMESTEP_ZERO_PROB="0.2"
    fi
    if [[ "${START_AT_TIMESTEP_ZERO_PROB_END_EXPLICIT}" -eq 0 ]]; then
      START_AT_TIMESTEP_ZERO_PROB_END="${START_AT_TIMESTEP_ZERO_PROB}"
    fi
    if [[ "${FREEZE_AT_TIMESTEP_ZERO_PROB_EXPLICIT}" -eq 0 ]]; then
      FREEZE_AT_TIMESTEP_ZERO_PROB="0.95"
    fi
    if [[ "${FREEZE_AT_TIMESTEP_ZERO_PROB_END_EXPLICIT}" -eq 0 ]]; then
      FREEZE_AT_TIMESTEP_ZERO_PROB_END="${FREEZE_AT_TIMESTEP_ZERO_PROB}"
    fi
    if [[ "${DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES_EXPLICIT}" -eq 0 ]]; then
      DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES="True"
    fi
    if [[ "${RUN_NAME_EXPLICIT}" -eq 0 ]]; then
      RUN_NAME="${RUN_NAME}_stable_half"
    fi
    if [[ "${TRAINING_NAME_EXPLICIT}" -eq 0 ]]; then
      TRAINING_NAME="${TRAINING_NAME}_stable_half"
    fi
    if [[ "${SCHEDULE_NAME_EXPLICIT}" -eq 0 ]]; then
      if [[ "${SCHEDULE_VARIANT}" == "dag_first" ]]; then
        SCHEDULE_NAME="teacher_anchor_then_goal_curriculum_v2_dag_first_stable_half"
      else
        SCHEDULE_NAME="teacher_anchor_then_goal_curriculum_v2_stable_half"
      fi
    fi
    if [[ "${SCHEDULE_NOTES_EXPLICIT}" -eq 0 ]]; then
      if [[ "${SCHEDULE_VARIANT}" == "dag_first" ]]; then
        SCHEDULE_NOTES="0-700 teacher rollout mix decays 0.7->0.0. 0-2000 pure DAgger with PPO disabled. 2000-3000 PPO ramps 0->0.3 while DAgger stays dominant. 2500-3500 command_only_env_prob ramps 0->0.5. 2500-end external_goal_prob ramps 0->0.5 so half the envs remain on stable clip / motion-tracking distribution while half train external goals. Reset curriculum is frozen at start_at_zero=0.2 and freeze_at_zero=0.95 to preserve copy-style reset distribution."
      else
        SCHEDULE_NOTES="0-700 teacher rollout mix decays 0.7->0.0. 0-2500 teacher-anchored clip-only; PPO ramps 0->0.3 over 0-3000 while DAgger weight stays dominant. 2500-3500 command_only_env_prob ramps 0->0.5. 2500-end external_goal_prob ramps 0->0.5 so half the envs remain on stable clip / motion-tracking distribution while half train external goals. Reset curriculum is frozen at start_at_zero=0.2 and freeze_at_zero=0.95 to preserve copy-style reset distribution."
      fi
    fi
    ;;
  *)
    echo "[ERROR] Unsupported DISTRIBUTION_VARIANT='${DISTRIBUTION_VARIANT}'. Use one of: default, stable_half" >&2
    exit 2
    ;;
esac

IMAGE_WIDTH_EXPLICIT=0
[[ -n "${IMAGE_WIDTH+x}" ]] && IMAGE_WIDTH_EXPLICIT=1
IMAGE_HEIGHT_EXPLICIT=0
[[ -n "${IMAGE_HEIGHT+x}" ]] && IMAGE_HEIGHT_EXPLICIT=1
CAMERA_NEAR_EXPLICIT=0
[[ -n "${CAMERA_NEAR+x}" ]] && CAMERA_NEAR_EXPLICIT=1
CAMERA_FAR_EXPLICIT=0
[[ -n "${CAMERA_FAR+x}" ]] && CAMERA_FAR_EXPLICIT=1
CAMERA_MAX_DISTANCE_EXPLICIT=0
[[ -n "${CAMERA_MAX_DISTANCE+x}" ]] && CAMERA_MAX_DISTANCE_EXPLICIT=1
PERCEPTION_WARP_PREPROCESS_EXPLICIT=0
[[ -n "${PERCEPTION_WARP_PREPROCESS+x}" ]] && PERCEPTION_WARP_PREPROCESS_EXPLICIT=1

TEACHER_REF_RUN_ID="5vlz6pj8"
TEACHER_REF_LOCAL_CHECKPOINT="${SCRIPT_DIR}/.teacher_checkpoints/model_24000.pt"
TEACHER_REF_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_sq_aug_mix_ml"
TEACHER_REF_PERCEPTION_PRESET="heightmap"
TEACHER_U5LGUXVL_RUN_ID="u5lguxvl"
TEACHER_U5LGUXVL_MODEL_FILE="model_06000.pt"
TEACHER_U5LGUXVL_LOCAL_CHECKPOINT="${SCRIPT_DIR}/.teacher_checkpoints/${TEACHER_U5LGUXVL_MODEL_FILE}"
TEACHER_COMPAT_PROFILE_RESOLVED="${TEACHER_COMPAT_PROFILE}"
TEACHER_COMPAT_NOTES_AUTO=""

if [[ "${TEACHER_CHECKPOINT}" == https://wandb.ai/*/runs/* ]]; then
  TEACHER_CHECKPOINT="$(normalize_checkpoint_ref "${TEACHER_CHECKPOINT}")"
fi

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
  elif [[ "${TEACHER_CHECKPOINT}" == *"${TEACHER_U5LGUXVL_RUN_ID}"* || "${TEACHER_CHECKPOINT}" == "${TEACHER_U5LGUXVL_LOCAL_CHECKPOINT}" ]]; then
    TEACHER_COMPAT_PROFILE_RESOLVED="u5lguxvl_generalist"
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
  u5lguxvl_generalist)
    if [[ "${TEACHER_OBS_KEYS_EXPLICIT}" -eq 0 ]]; then
      TEACHER_OBS_KEYS="actor_obs"
    fi
    if [[ "${TEACHER_PERCEPTION_PRESET_EXPLICIT}" -eq 0 ]]; then
      TEACHER_PERCEPTION_PRESET="none"
    fi
    if [[ "${TEACHER_PERCEPTION_OBS_KEY_EXPLICIT}" -eq 0 ]]; then
      TEACHER_PERCEPTION_OBS_KEY=""
    fi
    if [[ "${TEACHER_ACTOR_OBS_HISTORY_LENGTH_EXPLICIT}" -eq 0 ]]; then
      TEACHER_ACTOR_OBS_HISTORY_LENGTH="5"
    fi
    append_teacher_compat_note "teacher_obs_keys defaulted to actor_obs to match u5lguxvl teacher"
    append_teacher_compat_note "teacher perception disabled to match u5lguxvl teacher"
    append_teacher_compat_note "teacher perception obs key cleared because this teacher does not consume perception input"
    append_teacher_compat_note "actor_obs history length set to ${TEACHER_ACTOR_OBS_HISTORY_LENGTH} to match teacher checkpoint"
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

if [[ "${DATA_MODE}" == "omomo-debug" ]]; then
  OMOMO_DEBUG_FILTER_RESULT=$(
    MOTION_DIR="${MOTION_DIR}" OMOMO_DEBUG_PREFIXES="${OMOMO_DEBUG_PREFIXES}" python - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

motion_dir = Path(os.environ["MOTION_DIR"]).expanduser().resolve()
prefixes_raw = os.environ.get("OMOMO_DEBUG_PREFIXES", '["sub"]')

if not motion_dir.is_dir():
    print(str(motion_dir))
    raise SystemExit(0)

try:
    parsed = json.loads(prefixes_raw)
    if isinstance(parsed, str):
        prefixes = [parsed.strip()]
    else:
        prefixes = [str(item).strip() for item in parsed if str(item).strip()]
except Exception:
    prefixes = [part.strip() for part in prefixes_raw.split(",") if part.strip()]

if not prefixes:
    raise SystemExit("[ERROR] OMOMO_DEBUG_PREFIXES resolved to an empty prefix list.")

cache_key = hashlib.sha1(f"{motion_dir}:{prefixes}".encode("utf-8")).hexdigest()[:10]
cache_dir = motion_dir.parent / f"{motion_dir.name}_omomo_debug_{cache_key}"
cache_dir.mkdir(parents=True, exist_ok=True)

kept = 0
excluded = 0
for path in sorted(motion_dir.iterdir()):
    target = cache_dir / path.name
    if path.suffix != ".npz":
        if not target.exists():
            target.symlink_to(path)
        continue

    keep = any(path.stem.startswith(prefix) for prefix in prefixes)
    if keep:
        kept += 1
        if not target.exists():
            target.symlink_to(path)
    else:
        excluded += 1
        if target.exists():
            target.unlink()

print(f"{cache_dir}|{kept}|{excluded}|{json.dumps(prefixes)}")
PY
  )
  MOTION_DIR="${OMOMO_DEBUG_FILTER_RESULT%%|*}"
  OMOMO_DEBUG_FILTER_STATS="${OMOMO_DEBUG_FILTER_RESULT#*|}"
  OMOMO_DEBUG_FILTER_KEPT="${OMOMO_DEBUG_FILTER_STATS%%|*}"
  OMOMO_DEBUG_FILTER_REST="${OMOMO_DEBUG_FILTER_STATS#*|}"
  OMOMO_DEBUG_FILTER_EXCLUDED="${OMOMO_DEBUG_FILTER_REST%%|*}"
  OMOMO_DEBUG_FILTER_PREFIXES="${OMOMO_DEBUG_FILTER_REST#*|}"
  if [[ "${OMOMO_DEBUG_FILTER_KEPT}" == "0" ]]; then
    echo "[ERROR] omomo-debug matched zero clips under ${MOTION_DIR} for prefixes ${OMOMO_DEBUG_FILTER_PREFIXES}" >&2
    exit 1
  fi
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

OBJECT_SPEC_KIND=""
if [[ -n "${OBJECT_SPEC_PATH}" ]]; then
  object_spec_suffix=$(printf '%s' "${OBJECT_SPEC_PATH}" | tr '[:upper:]' '[:lower:]')
  if [[ "${object_spec_suffix}" == *.json ]]; then
    OBJECT_SPEC_KIND="clip-map"
  else
    OBJECT_SPEC_KIND="single-urdf"
  fi
fi

if [[ "${OBJECT_SPEC_KIND}" == "clip-map" ]]; then
  "${PYTHON_BIN}" - "${MOTION_DIR}" "${OBJECT_SPEC_PATH}" "${DATA_MODE}" "${ASSERT_ACTIVE_MULTI_URDF}" <<'PY'
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

motion_dir = Path(sys.argv[1]).expanduser().resolve()
object_spec_path = Path(sys.argv[2]).expanduser().resolve()
data_mode = sys.argv[3].strip().lower()
assert_mode_raw = sys.argv[4].strip().lower()

if not motion_dir.is_dir():
    raise SystemExit(f"[ERROR] Motion dir not found for object-bank validation: {motion_dir}")
if not object_spec_path.is_file():
    raise SystemExit(f"[ERROR] Object spec map not found for object-bank validation: {object_spec_path}")

payload = json.loads(object_spec_path.read_text(encoding="utf-8"))
clips = payload["clips"] if isinstance(payload, dict) and isinstance(payload.get("clips"), dict) else payload
if not isinstance(clips, dict) or not clips:
    raise SystemExit(f"[ERROR] Invalid clip-object map payload: {object_spec_path}")

active_clip_ids = [path.stem for path in sorted(motion_dir.glob("*.npz"))]
if not active_clip_ids:
    raise SystemExit(f"[ERROR] No .npz clips found under active MOTION_DIR: {motion_dir}")

resolved_urdfs: list[str] = []
missing: list[str] = []
for clip_id in active_clip_ids:
    entry = clips.get(clip_id)
    urdf = ""
    if isinstance(entry, str):
        urdf = entry.strip()
    elif isinstance(entry, dict):
        urdf = str(entry.get("object_urdf_path", "")).strip()

    if not urdf:
        npz_path = motion_dir / f"{clip_id}.npz"
        data = np.load(npz_path, allow_pickle=True)
        if "object_urdf_path" in data:
            arr = np.asarray(data["object_urdf_path"])
            if arr.size:
                item = arr.item() if arr.shape == () else arr.reshape(-1)[0]
                urdf = str(item).strip()

    if not urdf:
        missing.append(clip_id)
        continue
    resolved_urdfs.append(str(Path(urdf).expanduser().resolve()))

if missing:
    preview = ", ".join(missing[:10])
    raise SystemExit(
        f"[ERROR] Active motion clips missing object_urdf_path resolution in {object_spec_path}: {preview}"
    )

counts = Counter(resolved_urdfs)
unique_urdfs = sorted(counts)
top = ", ".join(f"{Path(path).name}:{count}" for path, count in counts.most_common(5))
print(
    f"[INFO] active_object_bank={len(active_clip_ids)} clips {len(unique_urdfs)} unique_urdfs "
    f"(top={top})"
)

assert_multi = False
if assert_mode_raw in {"1", "true", "yes", "on"}:
    assert_multi = True
elif assert_mode_raw in {"0", "false", "no", "off"}:
    assert_multi = False
elif assert_mode_raw == "auto":
    assert_multi = data_mode in {"mix-naive", "mix-curriculum"}

if assert_multi and len(unique_urdfs) <= 1:
    only = unique_urdfs[0] if unique_urdfs else "<none>"
    raise SystemExit(
        f"[ERROR] Expected multiple active object URDFs for DATA_MODE={data_mode}, "
        f"but only found {len(unique_urdfs)} under active MOTION_DIR {motion_dir}. "
        f"Resolved URDF: {only}"
    )
PY
elif [[ "${OBJECT_SPEC_KIND}" == "single-urdf" ]]; then
  if [[ ! -f "${OBJECT_SPEC_PATH}" ]]; then
    echo "[ERROR] Object URDF not found: ${OBJECT_SPEC_PATH}" >&2
    exit 1
  fi
fi

echo "[INFO] teacher_checkpoint=${TEACHER_CHECKPOINT}"
echo "[INFO] teacher_compat_profile=${TEACHER_COMPAT_PROFILE_RESOLVED}"
echo "[INFO] teacher_obs_keys=${TEACHER_OBS_KEYS}"
echo "[INFO] teacher_perception_preset=${TEACHER_PERCEPTION_PRESET} teacher_perception_obs_key=${TEACHER_PERCEPTION_OBS_KEY}"
echo "[INFO] critic_perception_preset=${CRITIC_PERCEPTION_PRESET} critic_perception_obs_key=${CRITIC_PERCEPTION_OBS_KEY}"
if [[ -n "${TEACHER_ACTOR_OBS_HISTORY_LENGTH}" ]]; then
  echo "[INFO] teacher_actor_obs_history_length=${TEACHER_ACTOR_OBS_HISTORY_LENGTH}"
fi
echo "[INFO] run_name=${RUN_NAME} training_name=${TRAINING_NAME}"
echo "[INFO] exp=${EXP} perception=${PERCEPTION_PRESET}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES} nproc=${NPROC} num_envs=${NUM_ENVS}"
echo "[INFO] data_mode=${DATA_MODE}"
echo "[INFO] schedule_variant=${SCHEDULE_VARIANT}"
echo "[INFO] reward_variant=${REWARD_VARIANT}"
echo "[INFO] distribution_variant=${DISTRIBUTION_VARIANT}"
echo "[INFO] motion_dir=${MOTION_DIR}"
if [[ -n "${OMOMO_DEBUG_FILTER_KEPT:-}" ]]; then
  echo "[INFO] omomo_debug_prefixes=${OMOMO_DEBUG_FILTER_PREFIXES} kept=${OMOMO_DEBUG_FILTER_KEPT} excluded=${OMOMO_DEBUG_FILTER_EXCLUDED}"
fi
if [[ "${OBJECT_SPEC_KIND}" == "clip-map" ]]; then
  echo "[INFO] object_spec_path=${OBJECT_SPEC_PATH}"
elif [[ "${OBJECT_SPEC_KIND}" == "single-urdf" ]]; then
  echo "[INFO] object_urdf_path=${OBJECT_SPEC_PATH}"
fi
if [[ -n "${FILTERED_MOTION_DIR_KEPT:-}" ]]; then
  echo "[INFO] motion_filter_non_placement_clips=${FILTER_NON_PLACEMENT_CLIPS} final_placement_max_delta_z=${FINAL_PLACEMENT_MAX_DELTA_Z} kept=${FILTERED_MOTION_DIR_KEPT} excluded=${FILTERED_MOTION_DIR_EXCLUDED}"
fi
echo "[INFO] student_actor_inputs=${STUDENT_ACTOR_INPUTS}"
echo "[INFO] schedule_name=${SCHEDULE_NAME}"
echo "[INFO] schedule_notes=${SCHEDULE_NOTES}"
echo "[INFO] ppo_schedule=${PPO_START_EPOCH}->${DAGGER_END_EPOCH} target=${PPO_TARGET_COEFF} dagger_loss_coef=${DAGGER_LOSS_COEF}"
echo "[INFO] teacher_action_mix_ratio=${TEACHER_ACTION_MIX_RATIO}"
if [[ -n "${TEACHER_ACTION_MIX_RATIO_START}" || -n "${TEACHER_ACTION_MIX_RATIO_END}" || -n "${TEACHER_ACTION_MIX_RATIO_END_ITERATION}" ]]; then
  echo "[INFO] teacher_action_mix_schedule=${TEACHER_ACTION_MIX_RATIO_START}->${TEACHER_ACTION_MIX_RATIO_END} end_iter=${TEACHER_ACTION_MIX_RATIO_END_ITERATION}"
fi
echo "[INFO] entropy_coef=${ENTROPY_COEF} dagger_match_std=${DAGGER_MATCH_STD}"
echo "[INFO] command_only_env_prob=${COMMAND_ONLY_ENV_PROB_START}->${COMMAND_ONLY_ENV_PROB_END} iter=${COMMAND_ONLY_ENV_PROB_START_ITER}->${COMMAND_ONLY_ENV_PROB_END_ITER}"
echo "[INFO] sparse_goal_enabled=${SPARSE_GOAL_ENABLED} ext_prob=${EXTERNAL_GOAL_PROB_START}->${EXTERNAL_GOAL_PROB_END}"
echo "[INFO] external_goal_prob_iter=${EXTERNAL_GOAL_PROB_START_ITER}->${EXTERNAL_GOAL_PROB_END_ITER}"
echo "[INFO] external_goal_range_xy_start=${EXTERNAL_GOAL_POS_LOCAL_MIN_START} -> ${EXTERNAL_GOAL_POS_LOCAL_MAX_START}"
echo "[INFO] external_goal_range_xy_end=${EXTERNAL_GOAL_POS_LOCAL_MIN} -> ${EXTERNAL_GOAL_POS_LOCAL_MAX}"
echo "[INFO] external_goal_range_iter=${EXTERNAL_GOAL_RANGE_START_ITER}->${EXTERNAL_GOAL_RANGE_END_ITER}"
echo "[INFO] clip_goal_delta_steps=${CLIP_GOAL_DELTA_MIN_STEPS}-${CLIP_GOAL_DELTA_MAX_STEPS} (legacy/unused; clip-goal now uses final placement)"
echo "[INFO] start_at_timestep_zero_prob=${START_AT_TIMESTEP_ZERO_PROB}->${START_AT_TIMESTEP_ZERO_PROB_END} iter=${START_AT_TIMESTEP_ZERO_PROB_START_ITER}->${START_AT_TIMESTEP_ZERO_PROB_END_ITER}"
echo "[INFO] freeze_at_timestep_zero_prob=${FREEZE_AT_TIMESTEP_ZERO_PROB}->${FREEZE_AT_TIMESTEP_ZERO_PROB_END} iter=${FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER}->${FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER}"
echo "[INFO] reset_to_default_pose=${RESET_TO_DEFAULT_POSE}"
echo "[INFO] default_pose_prepend=${ENABLE_DEFAULT_POSE_PREPEND} duration_s=${DEFAULT_POSE_PREPEND_DURATION_S} default_pose_append=${ENABLE_DEFAULT_POSE_APPEND} append_duration_s=${DEFAULT_POSE_APPEND_DURATION_S}"
echo "[INFO] viser_distill_minimal_ui=${VISER_DISTILL_MINIMAL_UI}"
echo "[INFO] viser_show_target_keypoints=${VISER_SHOW_TARGET_KEYPOINTS}"
echo "[INFO] dagger_ignore_external_goal_samples=${DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES}"
echo "[INFO] dagger_ignore_episode_initial_steps=${DAGGER_IGNORE_EPISODE_INITIAL_STEPS}"
echo "[INFO] max_episode_length_s=${MAX_EPISODE_LENGTH_S}"
if [[ "${RESET_TO_DEFAULT_POSE}" == "True" || "${RESET_TO_DEFAULT_POSE}" == "true" || "${RESET_TO_DEFAULT_POSE}" == "1" ]]; then
  echo "[WARN] reset_to_default_pose=True applies to every reset, including non-zero motion starts; this is much harder than runtime prepend only."
fi
if [[ -n "${TEACHER_COMPAT_NOTES}" ]]; then
  echo "[WARN] teacher_compat_notes=${TEACHER_COMPAT_NOTES}"
fi

EXTRA_DISTILL_ARGS=()
if [[ -n "${TEACHER_ACTOR_OBS_HISTORY_LENGTH}" ]]; then
  EXTRA_DISTILL_ARGS+=(--observation.groups.actor_obs.history-length="${TEACHER_ACTOR_OBS_HISTORY_LENGTH}")
fi
if [[ "${DATA_MODE}" == "mix-curriculum" ]]; then
  EXTRA_DISTILL_ARGS+=(
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.enabled=True
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.clean-clip-name-prefixes="${MIX_CURRICULUM_OMOMO_PREFIXES}"
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.stage-start-iterations="${MIX_CURRICULUM_STAGE_START_ITERATIONS}"
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.clean-group-probabilities="${MIX_CURRICULUM_OMOMO_PROBABILITIES}"
  )
elif [[ "${DATA_MODE}" == "pure-real" ]]; then
  EXTRA_DISTILL_ARGS+=(
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.enabled=True
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.clean-clip-name-prefixes="${PURE_REAL_OMOMO_PREFIXES}"
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.stage-start-iterations='[0]'
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.clean-group-probabilities='[1.0]'
  )
fi

TEACHER_PERCEPTION_ARGS=(
  --algo.config.distill.teacher-perception-preset="${TEACHER_PERCEPTION_PRESET}"
)
if [[ -n "${TEACHER_PERCEPTION_OBS_KEY}" ]]; then
  TEACHER_PERCEPTION_ARGS+=(
    --algo.config.distill.teacher-perception-obs-key="${TEACHER_PERCEPTION_OBS_KEY}"
  )
fi

CRITIC_PERCEPTION_ARGS=(
  --algo.config.distill.critic-perception-preset="${CRITIC_PERCEPTION_PRESET}"
)
if [[ -n "${CRITIC_PERCEPTION_OBS_KEY}" ]]; then
  CRITIC_PERCEPTION_ARGS+=(
    --algo.config.distill.critic-perception-obs-key="${CRITIC_PERCEPTION_OBS_KEY}"
  )
fi

PERCEPTION_OVERRIDE_ARGS=()
if [[ "${IMAGE_WIDTH_EXPLICIT}" -eq 1 ]]; then
  PERCEPTION_OVERRIDE_ARGS+=(--perception.camera-width="${IMAGE_WIDTH}")
fi
if [[ "${IMAGE_HEIGHT_EXPLICIT}" -eq 1 ]]; then
  PERCEPTION_OVERRIDE_ARGS+=(--perception.camera-height="${IMAGE_HEIGHT}")
fi
if [[ "${CAMERA_NEAR_EXPLICIT}" -eq 1 ]]; then
  PERCEPTION_OVERRIDE_ARGS+=(--perception.camera-near="${CAMERA_NEAR}")
fi
if [[ "${CAMERA_FAR_EXPLICIT}" -eq 1 ]]; then
  PERCEPTION_OVERRIDE_ARGS+=(--perception.camera-far="${CAMERA_FAR}")
fi
if [[ "${CAMERA_MAX_DISTANCE_EXPLICIT}" -eq 1 ]]; then
  PERCEPTION_OVERRIDE_ARGS+=(--perception.max-distance="${CAMERA_MAX_DISTANCE}")
fi
if [[ "${PERCEPTION_WARP_PREPROCESS_EXPLICIT}" -eq 1 ]]; then
  PERCEPTION_OVERRIDE_ARGS+=(--perception.camera-warp-preprocess="${PERCEPTION_WARP_PREPROCESS}")
fi

OBJECT_URDF_ENV=()
if [[ -n "${OBJECT_SPEC_PATH}" ]]; then
  OBJECT_URDF_ENV=(OBJECT_URDF="${OBJECT_SPEC_PATH}")
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
  TEACHER_ACTION_MIX_RATIO_START="${TEACHER_ACTION_MIX_RATIO_START}" \
  TEACHER_ACTION_MIX_RATIO_END="${TEACHER_ACTION_MIX_RATIO_END}" \
  TEACHER_ACTION_MIX_RATIO_END_ITERATION="${TEACHER_ACTION_MIX_RATIO_END_ITERATION}" \
  BC_LOSS_COEF="${BC_LOSS_COEF}" \
  NUM_LEARNING_ITERATIONS="${NUM_LEARNING_ITERATIONS}" \
  PPO_START_EPOCH="${PPO_START_EPOCH}" \
  DAGGER_END_EPOCH="${DAGGER_END_EPOCH}" \
  PPO_TARGET_COEFF="${PPO_TARGET_COEFF}" \
  DAGGER_LOSS_COEF="${DAGGER_LOSS_COEF}" \
  DAGGER_MATCH_STD="${DAGGER_MATCH_STD}" \
  ENTROPY_COEF="${ENTROPY_COEF}" \
  START_AT_TIMESTEP_ZERO_PROB="${START_AT_TIMESTEP_ZERO_PROB}" \
  FREEZE_AT_TIMESTEP_ZERO_PROB="${FREEZE_AT_TIMESTEP_ZERO_PROB}" \
  HOLOSOMA_RESET_TO_DEFAULT_POSE="${RESET_TO_DEFAULT_POSE}" \
  ENABLE_DEFAULT_POSE_PREPEND="${ENABLE_DEFAULT_POSE_PREPEND}" \
  DEFAULT_POSE_PREPEND_DURATION_S="${DEFAULT_POSE_PREPEND_DURATION_S}" \
  ENABLE_DEFAULT_POSE_APPEND="${ENABLE_DEFAULT_POSE_APPEND}" \
  DEFAULT_POSE_APPEND_DURATION_S="${DEFAULT_POSE_APPEND_DURATION_S}" \
  VISER_DISTILL_MINIMAL_UI="${VISER_DISTILL_MINIMAL_UI}" \
  VISER_SHOW_TARGET_KEYPOINTS="${VISER_SHOW_TARGET_KEYPOINTS}" \
  PAIR_TERRAIN_WITH_MOTION="${PAIR_TERRAIN_WITH_MOTION}" \
  "${OBJECT_URDF_ENV[@]}" \
  bash "${SCRIPT_DIR}/distill_root_box.sh" "${TEACHER_CHECKPOINT}" \
    "perception:${PERCEPTION_PRESET}" \
    --algo.config.module-dict.actor.input-dim "${STUDENT_ACTOR_INPUTS}" \
    --algo.config.distill.schedule-name="${SCHEDULE_NAME}" \
    --algo.config.distill.schedule-notes="${SCHEDULE_NOTES}" \
    --algo.config.distill.teacher-compat-profile="${TEACHER_COMPAT_PROFILE_RESOLVED}" \
    --algo.config.distill.teacher-compat-notes="${TEACHER_COMPAT_NOTES}" \
    "${TEACHER_PERCEPTION_ARGS[@]}" \
    "${CRITIC_PERCEPTION_ARGS[@]}" \
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
    --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end="${START_AT_TIMESTEP_ZERO_PROB_END}" \
    --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-start-iter="${START_AT_TIMESTEP_ZERO_PROB_START_ITER}" \
    --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end-iter="${START_AT_TIMESTEP_ZERO_PROB_END_ITER}" \
    --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob="${FREEZE_AT_TIMESTEP_ZERO_PROB}" \
    --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end="${FREEZE_AT_TIMESTEP_ZERO_PROB_END}" \
    --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-start-iter="${FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER}" \
    --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end-iter="${FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER}" \
    --simulator.config.sim.max_episode_length_s "${MAX_EPISODE_LENGTH_S}" \
    "${PERCEPTION_OVERRIDE_ARGS[@]}" \
    "${EXTRA_DISTILL_ARGS[@]}" \
    "$@"
