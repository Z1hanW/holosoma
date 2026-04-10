#!/usr/bin/env bash
set -euo pipefail

# Distill object-carry generalist -> non-goal student with depth perception access.
#
# Student policy observation (actor):
# - actor_obs_root: root-position command [root_x, root_y, root_yaw]
# - actor_obs_proprio (base_lin_vel, base_ang_vel, dof_pos, dof_vel, actions)
# - perception_obs (camera depth)
# - No actor box state is used by student actor.
#
# Teacher policy observation:
# - auto-matched to the selected teacher checkpoint, following the same teacher
#   compatibility interface pattern as distill_box_drop_mixed.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

DEFAULT_TEACHER_CHECKPOINT=${DEFAULT_TEACHER_CHECKPOINT:-"wandb://zihanw22/boxer/u5lguxvl/model_17000.pt"}
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${DEFAULT_TEACHER_CHECKPOINT}}"
POSITIONAL_RUN_NAME=""
DATA_MODE=${DATA_MODE:-mix-naive}
SCHEDULE_VARIANT=${SCHEDULE_VARIANT:-default}
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

is_checkpoint_ref() {
  local ref="$1"
  [[ "${ref}" == wandb://* || "${ref}" == https://wandb.ai/*/runs/* || "${ref}" == /* || "${ref}" == ./* || "${ref}" == ../* || "${ref}" == *.pt ]]
}

while [[ $# -gt 0 ]]; do
  first_arg_normalized=$(echo "$1" | tr '[:upper:]' '[:lower:]')
  case "${first_arg_normalized}" in
    mix-naive)
      DATA_MODE="mix-naive"
      shift
      ;;
    pure-sd|pure-ds)
      DATA_MODE="pure-sd"
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
    *)
      break
      ;;
  esac
done

if [[ $# -gt 0 ]] && is_checkpoint_ref "$1"; then
  TEACHER_CHECKPOINT="$1"
  shift
fi
if [[ $# -gt 0 && "$1" != -* ]]; then
  POSITIONAL_RUN_NAME="$1"
  shift
fi

if [[ -z "${TEACHER_CHECKPOINT}" ]]; then
  echo "Usage: $0 [default|dag_first] [teacher_checkpoint.pt|wandb_run_url] [run_name] [extra train args...]" >&2
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

# Sim2real default: root-position distill without clip_phase in student torso observation.
EXP=${EXP:-g1-29dof-wbt-w-object-distill-root-pos-cmd}
RUN_NAME=${RUN_NAME:-g1_w_object_distill_box_perception_root_pos_cmd}
TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_box_perception_root_pos_cmd_access_to_depth}
TRAINING_PROJECT=${TRAINING_PROJECT:-boxer}
if [[ -n "${POSITIONAL_RUN_NAME}" ]]; then
  RUN_NAME="${POSITIONAL_RUN_NAME}"
fi

# Keep launcher self-contained: direct `bash ./distill_box_perception.sh` works out of box.
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

case "${DATA_MODE}" in
  default|mix-naive)
    DATA_MODE="mix-naive"
    MOTION_DIR=${MOTION_DIR:-"${DEFAULT_MIX_NAIVE_MOTION_DIR}"}
    ;;
  pure-sd)
    MOTION_DIR=${MOTION_DIR:-"${DEFAULT_DS_PREPARED_MOTION_DIR}"}
    ;;
  *)
    echo "[ERROR] Unsupported DATA_MODE='${DATA_MODE}'. Use one of: default, mix-naive, pure-sd" >&2
    exit 2
    ;;
esac

TEACHER_OBS_KEYS=${TEACHER_OBS_KEYS:-actor_obs}
TEACHER_PERCEPTION_PRESET=${TEACHER_PERCEPTION_PRESET:-none}
TEACHER_PERCEPTION_OBS_KEY=${TEACHER_PERCEPTION_OBS_KEY:-teacher_perception_obs}
CRITIC_PERCEPTION_PRESET=${CRITIC_PERCEPTION_PRESET:-none}
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
FIXED_BC_EVAL_LOG_INTERVAL=${FIXED_BC_EVAL_LOG_INTERVAL:-1000}
SCHEDULE_NAME=${SCHEDULE_NAME:-root_pos_teacher_anchor_v2}
SCHEDULE_NOTES=${SCHEDULE_NOTES:-"0-700 teacher rollout mix decays 0.7->0.0. PPO starts immediately with target coeff 0.3 while DAgger remains active until iteration 3000. Root-position command replaces the drop-command-specific curricula from distill_box_drop_mixed.sh; other launcher defaults stay aligned."}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.2}
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
PERCEPTION_PRESET=${PERCEPTION_PRESET:-camera_depth_d435i}
STUDENT_ACTOR_INPUTS=${STUDENT_ACTOR_INPUTS:-"['actor_obs_root','actor_obs_proprio']"}
DAGGER_MATCH_STD=${DAGGER_MATCH_STD:-True}
ENTROPY_COEF=${ENTROPY_COEF:-0.0}
DAGGER_IGNORE_EPISODE_INITIAL_STEPS=${DAGGER_IGNORE_EPISODE_INITIAL_STEPS:-0}
MAX_EPISODE_LENGTH_S=${MAX_EPISODE_LENGTH_S:-8.0}
RESET_TO_DEFAULT_POSE=${RESET_TO_DEFAULT_POSE:-False}
ENABLE_DEFAULT_POSE_PREPEND=${ENABLE_DEFAULT_POSE_PREPEND:-True}
DEFAULT_POSE_PREPEND_DURATION_S=${DEFAULT_POSE_PREPEND_DURATION_S:-0.5}
ENABLE_DEFAULT_POSE_APPEND=${ENABLE_DEFAULT_POSE_APPEND:-False}
DEFAULT_POSE_APPEND_DURATION_S=${DEFAULT_POSE_APPEND_DURATION_S:-0.0}
VISER_DISTILL_MINIMAL_UI=${VISER_DISTILL_MINIMAL_UI:-1}
VISER_SHOW_TARGET_KEYPOINTS=${VISER_SHOW_TARGET_KEYPOINTS:-0}

# Use the preset defaults unless the caller explicitly overrides camera settings.
IMAGE_WIDTH=${IMAGE_WIDTH:-}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-}
CAMERA_NEAR=${CAMERA_NEAR:-}
CAMERA_FAR=${CAMERA_FAR:-}
CAMERA_MAX_DISTANCE=${CAMERA_MAX_DISTANCE:-}
PERCEPTION_WARP_PREPROCESS=${PERCEPTION_WARP_PREPROCESS:-}
OBJECT_GEOMETRY_MODE=${OBJECT_GEOMETRY_MODE:-}
OBJECT_GEOMETRY_MODE_NORM=""
HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE=""
PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE=""

if [[ -n "${OBJECT_GEOMETRY_MODE}" ]]; then
  case "$(echo "${OBJECT_GEOMETRY_MODE}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on|primitive|primitives|box|cuboid)
      OBJECT_GEOMETRY_MODE_NORM="primitive"
      HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE="primitive"
      PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE="primitive"
      ;;
    0|false|no|off|mesh|urdf|disable|disabled)
      OBJECT_GEOMETRY_MODE_NORM="mesh"
      HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE="urdf"
      PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE="mesh"
      ;;
    *)
      echo "[ERROR] OBJECT_GEOMETRY_MODE must be one of: on/off/primitive/mesh. Got: ${OBJECT_GEOMETRY_MODE}" >&2
      exit 2
      ;;
  esac
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
      SCHEDULE_NAME="root_pos_teacher_anchor_v2_dag_first"
    fi
    if [[ "${SCHEDULE_NOTES_EXPLICIT}" -eq 0 ]]; then
      SCHEDULE_NOTES="0-700 teacher rollout mix decays 0.7->0.0. 0-2000 pure DAgger with PPO disabled. 2000-3000 PPO ramps 0->0.3 while DAgger stays dominant. Root-position command replaces the drop-command-specific curricula from distill_box_drop_mixed.sh; other launcher defaults stay aligned."
    fi
    ;;
  *)
    echo "[ERROR] Unsupported SCHEDULE_VARIANT='${SCHEDULE_VARIANT}'. Use one of: default, dag_first" >&2
    exit 2
    ;;
esac

TEACHER_REF_RUN_ID="5vlz6pj8"
TEACHER_REF_MODEL_FILE="model_24000.pt"
TEACHER_REF_LOCAL_CHECKPOINT="${SCRIPT_DIR}/.teacher_checkpoints/${TEACHER_REF_MODEL_FILE}"
TEACHER_REF_PERCEPTION_PRESET="heightmap"
TEACHER_U5LGUXVL_RUN_ID="u5lguxvl"
TEACHER_U5LGUXVL_MODEL_FILE="model_17000.pt"
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
    if [[ "${TEACHER_ACTOR_OBS_HISTORY_LENGTH_EXPLICIT}" -eq 0 ]]; then
      TEACHER_ACTOR_OBS_HISTORY_LENGTH=""
    fi
    append_teacher_compat_note "teacher_obs_keys defaulted to actor_obs_teacher_compat for exact legacy ordering"
    append_teacher_compat_note "teacher now consumes ${TEACHER_PERCEPTION_PRESET} via ${TEACHER_PERCEPTION_OBS_KEY}"
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
if [[ -n "${OBJECT_GEOMETRY_MODE_NORM}" ]]; then
  echo "[INFO] object_geometry_mode=${OBJECT_GEOMETRY_MODE_NORM} simulator_object_spawn_mode=${HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE}"
else
  echo "[INFO] object_geometry_mode=<default>"
fi
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES} nproc=${NPROC} num_envs=${NUM_ENVS}"
echo "[INFO] data_mode=${DATA_MODE}"
if [[ -n "${MOTION_DIR:-}" ]]; then
  echo "[INFO] motion_dir=${MOTION_DIR}"
fi
echo "[INFO] schedule_variant=${SCHEDULE_VARIANT}"
echo "[INFO] schedule_name=${SCHEDULE_NAME}"
echo "[INFO] schedule_notes=${SCHEDULE_NOTES}"
echo "[INFO] student_actor_inputs=${STUDENT_ACTOR_INPUTS}"
echo "[INFO] num_learning_iterations=${NUM_LEARNING_ITERATIONS}"
echo "[INFO] bc_loss_coef=${BC_LOSS_COEF} dagger_loss_coef=${DAGGER_LOSS_COEF} teacher_action_mix_ratio=${TEACHER_ACTION_MIX_RATIO}"
if [[ -n "${TEACHER_ACTION_MIX_RATIO_START}" || -n "${TEACHER_ACTION_MIX_RATIO_END}" || -n "${TEACHER_ACTION_MIX_RATIO_END_ITERATION}" ]]; then
  echo "[INFO] teacher_action_mix_schedule=${TEACHER_ACTION_MIX_RATIO_START}->${TEACHER_ACTION_MIX_RATIO_END} end_iter=${TEACHER_ACTION_MIX_RATIO_END_ITERATION}"
fi
echo "[INFO] ppo_schedule=${PPO_START_EPOCH}->${DAGGER_END_EPOCH} target=${PPO_TARGET_COEFF} dagger_loss_coef=${DAGGER_LOSS_COEF}"
echo "[INFO] fixed_bc_eval_log_interval=${FIXED_BC_EVAL_LOG_INTERVAL}"
echo "[INFO] start_at_timestep_zero_prob=${START_AT_TIMESTEP_ZERO_PROB}"
echo "[INFO] entropy_coef=${ENTROPY_COEF} dagger_match_std=${DAGGER_MATCH_STD}"
echo "[INFO] default_pose_prepend=${ENABLE_DEFAULT_POSE_PREPEND} duration_s=${DEFAULT_POSE_PREPEND_DURATION_S} default_pose_append=${ENABLE_DEFAULT_POSE_APPEND} append_duration_s=${DEFAULT_POSE_APPEND_DURATION_S}"
echo "[INFO] viser_distill_minimal_ui=${VISER_DISTILL_MINIMAL_UI}"
echo "[INFO] viser_show_target_keypoints=${VISER_SHOW_TARGET_KEYPOINTS}"
echo "[INFO] dagger_ignore_episode_initial_steps=${DAGGER_IGNORE_EPISODE_INITIAL_STEPS}"
echo "[INFO] max_episode_length_s=${MAX_EPISODE_LENGTH_S}"
if [[ -n "${TEACHER_COMPAT_NOTES}" ]]; then
  echo "[WARN] teacher_compat_notes=${TEACHER_COMPAT_NOTES}"
fi

EXTRA_DISTILL_ARGS=()
if [[ -n "${TEACHER_ACTOR_OBS_HISTORY_LENGTH}" ]]; then
  EXTRA_DISTILL_ARGS+=(--observation.groups.actor_obs.history-length="${TEACHER_ACTOR_OBS_HISTORY_LENGTH}")
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
if [[ -n "${PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE}" ]]; then
  PERCEPTION_OVERRIDE_ARGS+=(--perception.object-geometry-mode="${PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE}")
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

OBJECT_GEOMETRY_MODE_ENV=()
if [[ -n "${HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE}" ]]; then
  OBJECT_GEOMETRY_MODE_ENV=(HOLOSOMA_OBJECT_SPAWN_MODE="${HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE}")
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
  HOLOSOMA_RESET_TO_DEFAULT_POSE="${RESET_TO_DEFAULT_POSE}" \
  ENABLE_DEFAULT_POSE_PREPEND="${ENABLE_DEFAULT_POSE_PREPEND}" \
  DEFAULT_POSE_PREPEND_DURATION_S="${DEFAULT_POSE_PREPEND_DURATION_S}" \
  ENABLE_DEFAULT_POSE_APPEND="${ENABLE_DEFAULT_POSE_APPEND}" \
  DEFAULT_POSE_APPEND_DURATION_S="${DEFAULT_POSE_APPEND_DURATION_S}" \
  VISER_DISTILL_MINIMAL_UI="${VISER_DISTILL_MINIMAL_UI}" \
  VISER_SHOW_TARGET_KEYPOINTS="${VISER_SHOW_TARGET_KEYPOINTS}" \
  PAIR_TERRAIN_WITH_MOTION="${PAIR_TERRAIN_WITH_MOTION}" \
  "${OBJECT_GEOMETRY_MODE_ENV[@]}" \
  bash "${SCRIPT_DIR}/distill_root_box.sh" "${TEACHER_CHECKPOINT}" \
    "perception:${PERCEPTION_PRESET}" \
    --algo.config.module-dict.actor.input-dim "${STUDENT_ACTOR_INPUTS}" \
    --algo.config.distill.schedule-name="${SCHEDULE_NAME}" \
    --algo.config.distill.schedule-notes="${SCHEDULE_NOTES}" \
    --algo.config.distill.teacher-compat-profile="${TEACHER_COMPAT_PROFILE_RESOLVED}" \
    --algo.config.distill.teacher-compat-notes="${TEACHER_COMPAT_NOTES}" \
    "${TEACHER_PERCEPTION_ARGS[@]}" \
    "${CRITIC_PERCEPTION_ARGS[@]}" \
    --algo.config.distill.dagger-ignore-episode-initial-steps="${DAGGER_IGNORE_EPISODE_INITIAL_STEPS}" \
    --algo.config.distill.fixed-bc-eval-log-interval="${FIXED_BC_EVAL_LOG_INTERVAL}" \
    --algo.config.distill.ppo-target-coeff="${PPO_TARGET_COEFF}" \
    --simulator.config.sim.max_episode_length_s "${MAX_EPISODE_LENGTH_S}" \
    "${EXTRA_DISTILL_ARGS[@]}" \
    "${PERCEPTION_OVERRIDE_ARGS[@]}" \
    "$@"
