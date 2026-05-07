#!/usr/bin/env bash
set -euo pipefail

# Teacher-policy physics rollout aligned with distill_box_perception.sh defaults.
#
# Design choice:
# - Load the teacher checkpoint directly as the rollout policy.
# - Keep motion/object defaults aligned with distill_box_perception.sh.
# - Do not route through distill student runtime / take_teacher_actions.
#
# Usage:
#   bash infer_teacher_rollout.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra tyro args...]
#
# Examples:
#   bash infer_teacher_rollout.sh
#   bash infer_teacher_rollout.sh wandb://zihanw22/boxer/u5lguxvl/model_17000.pt
#   bash infer_teacher_rollout.sh https://wandb.ai/zihanw22/boxer/runs/u5lguxvl
#   MOTION_CLIP_NAME=sub3_largebox_003_mj_w_obj bash infer_teacher_rollout.sh

usage() {
  cat <<'EOF'
Usage:
  bash infer_teacher_rollout.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra tyro args...]

Optional env vars:
  TEACHER_CHECKPOINT        (default: parsed from distill_box_perception.sh, else pinned fallback)
  WANDB_MODEL_FILE          (optional; used when checkpoint is a W&B run URL without /files/<checkpoint>)
  INFER_DATASET             (default: rollout-ref; options: rollout-ref|teacher-rollout|mix-naive|default|pure-sd)
  DATA_MODE                 (legacy alias for INFER_DATASET old DS modes)
  TEACHER_ROLLOUT_MOTION_DIR
                            (default: ./outputs/motion_bank; used by rollout-ref)
  TEACHER_ROLLOUT_FILTER_ENABLED
                            (default: True; prefer TEACHER_ROLLOUT_FILTERED_MOTION_DIR when available)
  TEACHER_ROLLOUT_FILTERED_MOTION_DIR
                            (default: ./outputs/motion_bank_success_box_0_92_0p3)
  MOTION_DIR                (optional override)
  MOTION_CLIP_NAME          (optional single clip name)
  MOTION_CLIP_ID            (optional single clip id)
  OBJECT_URDF               (optional override; default: MOTION_DIR/_clip_object_urdf_map.json when present)
  OBJECT_GEOMETRY_MODE      (optional; `on`/`primitive` forces cuboid primitive path,
                             `off`/`mesh` forces legacy URDF/mesh path)
  NUM_ENVS                  (default: 1)
  HEADLESS                  (default: False)
  VISER_PORT                (default: random)
  VISER_ENV_ID              (default: 0)
  VISER_UPDATE_HZ           (default: 30)
  VISER_RECENTER            (default: True)
  VIS_GPU                   (default: auto; IsaacSim uses HOLOSOMA_DEVICE=cuda:<idx> when possible)
  PAIR_TERRAIN_WITH_MOTION  (default: False)
  DISABLE_RANDOMIZATION     (default: True)
  START_AT_TIMESTEP_ZERO_PROB (default: 1.0)
  FREEZE_AT_TIMESTEP_ZERO_PROB (default: 0.0)
  RESET_NOISE_SCALE         (default: 0.0)
  MAX_EPISODE_LENGTH_S      (default: 1000000)
  MAX_EVAL_STEPS            (optional; overrides training.max_eval_steps)
  PHYSX_GPU_COLLISION_STACK_SIZE (default: 268435456)
  DRY_RUN                   (default: 0; set 1/true to print the final command)
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help|help)
      usage
      exit 0
      ;;
  esac
fi

extract_default_teacher_checkpoint_from_distill_box_perception() {
  "$PYTHON_BIN" - "${SCRIPT_DIR}/distill_box_perception.sh" <<'PY' 2>/dev/null || true
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    text = path.read_text(encoding="utf-8")
except Exception:
    sys.exit(0)

match = re.search(r'^DEFAULT_TEACHER_CHECKPOINT=\$\{DEFAULT_TEACHER_CHECKPOINT:-"([^"]+)"\}', text, re.M)
if match:
    print(match.group(1))
PY
}

DISTILL_DEFAULT_TEACHER_CHECKPOINT="$(extract_default_teacher_checkpoint_from_distill_box_perception)"
DISTILL_DEFAULT_TEACHER_CHECKPOINT="${DISTILL_DEFAULT_TEACHER_CHECKPOINT:-wandb://zihanw22/boxer/u5lguxvl/model_17000.pt}"
DEFAULT_TEACHER_CHECKPOINT="${DEFAULT_TEACHER_CHECKPOINT:-${DISTILL_DEFAULT_TEACHER_CHECKPOINT}}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${DEFAULT_TEACHER_CHECKPOINT}}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    wandb://*|https://wandb.ai/*|/*|./*|../*|*.pt)
      TEACHER_CHECKPOINT="$1"
      shift
      ;;
  esac
fi

parse_wandb_run_url() {
  local ref="$1"
  local clean_ref="${ref%%#*}"
  clean_ref="${clean_ref%%\?*}"
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

  "$PYTHON_BIN" - "${entity}" "${project}" "${run_id}" <<'PY' 2>/dev/null || true
import re
import sys
from pathlib import Path

repo_root = Path.cwd().resolve()
sanitized_sys_path: list[str] = []
for path_entry in sys.path:
    if path_entry in {"", "."}:
        continue
    try:
        if Path(path_entry).resolve() == repo_root:
            continue
    except Exception:
        pass
    sanitized_sys_path.append(path_entry)
sys.path = sanitized_sys_path

try:
    import wandb
except Exception:
    sys.exit(0)

entity, project, run_id = sys.argv[1:4]
api = wandb.Api(timeout=30)
run = api.run(f"{entity}/{project}/{run_id}")
model_pattern = re.compile(r"^model_(\d+)\.pt$")
latest_step = -1
latest_name = ""
for file_obj in run.files():
    name = getattr(file_obj, "name", "")
    match = model_pattern.match(name)
    if not match:
        continue
    step = int(match.group(1))
    if step >= latest_step:
        latest_step = step
        latest_name = name
if latest_name:
    print(latest_name)
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

if [[ "${TEACHER_CHECKPOINT}" == https://wandb.ai/*/runs/* ]]; then
  TEACHER_CHECKPOINT="$(normalize_checkpoint_ref "${TEACHER_CHECKPOINT}")"
fi

if [[ -z "${TEACHER_CHECKPOINT}" ]]; then
  echo "[ERROR] Missing teacher checkpoint." >&2
  usage >&2
  exit 2
fi
if [[ "${TEACHER_CHECKPOINT}" != wandb://* ]] && [[ ! -f "${TEACHER_CHECKPOINT}" ]]; then
  echo "[ERROR] checkpoint not found: ${TEACHER_CHECKPOINT}" >&2
  exit 1
fi

INFER_DATASET="${INFER_DATASET:-${DATA_MODE:-rollout-ref}}"
INFER_DATASET_RAW="$(echo "${INFER_DATASET}" | tr '[:upper:]' '[:lower:]' | tr -d '[][:space:]')"
DS_DATA_ROOT="${DS_DATA_ROOT:-${SCRIPT_DIR}/data/ds_box_data}"
TEACHER_ROLLOUT_MOTION_DIR=${TEACHER_ROLLOUT_MOTION_DIR:-"${SCRIPT_DIR}/outputs/motion_bank"}
TEACHER_ROLLOUT_FILTER_ENABLED=${TEACHER_ROLLOUT_FILTER_ENABLED:-True}
TEACHER_ROLLOUT_FILTERED_MOTION_DIR=${TEACHER_ROLLOUT_FILTERED_MOTION_DIR:-"${SCRIPT_DIR}/outputs/motion_bank_success_box_0_92_0p3"}
DEFAULT_MIX_NAIVE_MOTION_DIR="${DS_DATA_ROOT}/train_g1_w_obj_prepared_plus_omomo_orig"
DEFAULT_DS_PREPARED_MOTION_DIR="${DS_DATA_ROOT}/train_g1_w_obj_prepared"
DEFAULT_SINGLE_OBJECT_URDF="${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"

resolve_teacher_rollout_motion_dir() {
  local filter_norm
  filter_norm="$(echo "${TEACHER_ROLLOUT_FILTER_ENABLED}" | tr '[:upper:]' '[:lower:]')"
  case "${filter_norm}" in
    1|true|yes|on)
      if compgen -G "${TEACHER_ROLLOUT_FILTERED_MOTION_DIR}/box_*.npz" > /dev/null; then
        echo "${TEACHER_ROLLOUT_FILTERED_MOTION_DIR}"
        return 0
      fi
      ;;
  esac
  echo "${TEACHER_ROLLOUT_MOTION_DIR}"
}
DEFAULT_TEACHER_ROLLOUT_MOTION_DIR="$(resolve_teacher_rollout_motion_dir)"

case "${INFER_DATASET_RAW}" in
  rollout-ref|rollout_ref|teacher-rollout|teacher_rollout|teacherrollout|rollout)
    INFER_DATASET="rollout-ref"
    DATA_MODE="rollout-ref"
    MOTION_DIR="${MOTION_DIR:-${DEFAULT_TEACHER_ROLLOUT_MOTION_DIR}}"
    ;;
  default|mix-naive)
    INFER_DATASET="mix-naive"
    DATA_MODE="mix-naive"
    MOTION_DIR="${MOTION_DIR:-${DEFAULT_MIX_NAIVE_MOTION_DIR}}"
    ;;
  pure-sd|pure-ds)
    INFER_DATASET="pure-sd"
    DATA_MODE="pure-sd"
    MOTION_DIR="${MOTION_DIR:-${DEFAULT_DS_PREPARED_MOTION_DIR}}"
    ;;
  *)
    echo "[ERROR] INFER_DATASET must be one of: rollout-ref|teacher-rollout|mix-naive|default|pure-sd. Got: ${INFER_DATASET}" >&2
    exit 2
    ;;
esac

if [[ -z "${OBJECT_URDF+x}" ]]; then
  DEFAULT_OBJECT_MAP="${MOTION_DIR}/_clip_object_urdf_map.json"
  if [[ "${INFER_DATASET}" == "rollout-ref" ]]; then
    OBJECT_URDF="${DEFAULT_OBJECT_MAP}"
  elif [[ -f "${DEFAULT_OBJECT_MAP}" ]]; then
    OBJECT_URDF="${DEFAULT_OBJECT_MAP}"
  else
    OBJECT_URDF="${DEFAULT_SINGLE_OBJECT_URDF}"
  fi
fi

if [[ ! -e "${MOTION_DIR}" ]]; then
  echo "[ERROR] MOTION_DIR not found: ${MOTION_DIR}" >&2
  exit 1
fi
if [[ "${INFER_DATASET}" == "rollout-ref" ]] && ! compgen -G "${MOTION_DIR}/box_*.npz" > /dev/null; then
  echo "[ERROR] rollout-ref requires teacher rollout motion clips under MOTION_DIR=${MOTION_DIR}" >&2
  echo "[ERROR] Set TEACHER_ROLLOUT_MOTION_DIR, TEACHER_ROLLOUT_FILTERED_MOTION_DIR, or MOTION_DIR explicitly." >&2
  exit 2
fi
if [[ ! -f "${OBJECT_URDF}" ]]; then
  echo "[ERROR] OBJECT_URDF not found: ${OBJECT_URDF}" >&2
  exit 1
fi

NUM_ENVS="${NUM_ENVS:-1}"
HEADLESS="${HEADLESS:-False}"
VISER_PORT="${VISER_PORT:-18080}"
VISER_ENV_ID="${VISER_ENV_ID:-0}"
VISER_UPDATE_HZ="${VISER_UPDATE_HZ:-30}"
VISER_RECENTER="${VISER_RECENTER:-True}"
PAIR_TERRAIN_WITH_MOTION="${PAIR_TERRAIN_WITH_MOTION:-False}"
DISABLE_RANDOMIZATION="${DISABLE_RANDOMIZATION:-True}"
START_AT_TIMESTEP_ZERO_PROB="${START_AT_TIMESTEP_ZERO_PROB:-1.0}"
FREEZE_AT_TIMESTEP_ZERO_PROB="${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}"
RESET_NOISE_SCALE="${RESET_NOISE_SCALE:-0.0}"
MAX_EPISODE_LENGTH_S="${MAX_EPISODE_LENGTH_S:-1000000}"
PHYSX_GPU_COLLISION_STACK_SIZE="${PHYSX_GPU_COLLISION_STACK_SIZE:-268435456}"
DRY_RUN="${DRY_RUN:-0}"
VIS_GPU="${VIS_GPU:-auto}"
OBJECT_GEOMETRY_MODE_RAW=${OBJECT_GEOMETRY_MODE:-}
OBJECT_GEOMETRY_MODE=""
HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE=""
PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE=""

if [[ -n "${OBJECT_GEOMETRY_MODE_RAW}" ]]; then
  case "$(echo "${OBJECT_GEOMETRY_MODE_RAW}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on|primitive|primitives|box|cuboid)
      OBJECT_GEOMETRY_MODE="primitive"
      HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE="primitive"
      PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE="primitive"
      ;;
    0|false|no|off|mesh|urdf|disable|disabled)
      OBJECT_GEOMETRY_MODE="mesh"
      HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE="urdf"
      PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE="mesh"
      ;;
    *)
      echo "[ERROR] OBJECT_GEOMETRY_MODE must be one of: on/off/primitive/mesh. Got: ${OBJECT_GEOMETRY_MODE_RAW}" >&2
      exit 2
      ;;
  esac
fi

MOTION_CLIP_NAME="${MOTION_CLIP_NAME:-}"
MOTION_CLIP_ID="${MOTION_CLIP_ID:-}"

VISER_ENABLE_CLIP_GUI="${VISER_ENABLE_CLIP_GUI:-1}"
VISER_ENABLE_MANUAL_GUI="${VISER_ENABLE_MANUAL_GUI:-0}"
export VISER_ENABLE_CLIP_GUI
export VISER_ENABLE_MANUAL_GUI
export VISER_SHOW_TARGET_KEYPOINTS="${VISER_SHOW_TARGET_KEYPOINTS:-0}"
export HOLOSOMA_DISABLE_AUTO_RESET="${HOLOSOMA_DISABLE_AUTO_RESET:-1}"
export HOLOSOMA_DISABLE_CLIP_END_RESET="${HOLOSOMA_DISABLE_CLIP_END_RESET:-1}"

is_truthy() {
  case "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

normalize_bool_flag() {
  if is_truthy "${1:-}"; then
    echo "True"
  else
    echo "False"
  fi
}

SELECTED_GPU=""
if [[ "${VIS_GPU}" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    SELECTED_GPU="$(
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2,2n \
        | head -n1 \
        | cut -d, -f1 \
        | tr -d ' '
    )"
  fi
elif [[ "${VIS_GPU}" =~ ^[0-9]+$ ]]; then
  SELECTED_GPU="${VIS_GPU}"
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[WARN] CUDA_VISIBLE_DEVICES is set for an IsaacSim launch; prefer VIS_GPU or HOLOSOMA_DEVICE." >&2
elif [[ -z "${HOLOSOMA_DEVICE:-}" && -n "${SELECTED_GPU}" ]]; then
  export HOLOSOMA_DEVICE="cuda:${SELECTED_GPU}"
fi

HETEROGENEOUS_OBJECT_SINGLE_SLOT_DISABLE_EXPLICIT=0
[[ -n "${HOLOSOMA_DISABLE_HETEROGENEOUS_OBJECT_SINGLE_SLOT+x}" ]] && HETEROGENEOUS_OBJECT_SINGLE_SLOT_DISABLE_EXPLICIT=1
AUTO_SWITCH_MULTI_OBJECT_MODE=0
if is_truthy "${VISER_ENABLE_CLIP_GUI}" && [[ "${NUM_ENVS}" == "1" ]] && [[ "${OBJECT_URDF}" == *.json ]]; then
  # Single-env clip switching with an object-map needs per-asset simulator objects.
  # Otherwise env_0 keeps its initial object asset while the selected clip metadata changes.
  if [[ -z "${OBJECT_GEOMETRY_MODE_RAW}" ]]; then
    OBJECT_GEOMETRY_MODE="mesh"
    HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE="urdf"
    PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE="mesh"
    AUTO_SWITCH_MULTI_OBJECT_MODE=1
  fi
  if [[ "${HETEROGENEOUS_OBJECT_SINGLE_SLOT_DISABLE_EXPLICIT}" -eq 0 ]]; then
    export HOLOSOMA_DISABLE_HETEROGENEOUS_OBJECT_SINGLE_SLOT=1
    AUTO_SWITCH_MULTI_OBJECT_MODE=1
  fi
fi
if [[ -n "${HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE}" ]]; then
  export HOLOSOMA_OBJECT_SPAWN_MODE="${HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE}"
fi

HEADLESS_FLAG="$(normalize_bool_flag "${HEADLESS}")"
if is_truthy "${HEADLESS}"; then
  export HEADLESS=1
else
  export HEADLESS=0
fi
EXTRA_ARGS=("$@")

cmd=(
  "${PYTHON_BIN}" -m holosoma.visualize physics
  --checkpoint "${TEACHER_CHECKPOINT}"
  --motion-dir "${MOTION_DIR}"
  --num-envs "${NUM_ENVS}"
  --headless "${HEADLESS_FLAG}"
  --pair-terrain-with-motion "${PAIR_TERRAIN_WITH_MOTION}"
  --viser-port "${VISER_PORT}"
  --viser-env-id "${VISER_ENV_ID}"
  --viser-update-hz "${VISER_UPDATE_HZ}"
  --viser-recenter "${VISER_RECENTER}"
  --simulator.config.sim.max_episode_length_s "${MAX_EPISODE_LENGTH_S}"
  --simulator.config.sim.physx.gpu_collision_stack_size "${PHYSX_GPU_COLLISION_STACK_SIZE}"
  --robot.object.enabled True
  --robot.object.object_urdf_path "${OBJECT_URDF}"
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob "${START_AT_TIMESTEP_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.freeze_at_timestep_zero_prob "${FREEZE_AT_TIMESTEP_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.overall_noise_scale "${RESET_NOISE_SCALE}"
  --algo.config.distill.bc_loss_coef 0.0
  --algo.config.distill.loss_coef 0.0
  --algo.config.distill.switch_to_rl_after -1
  --algo.config.distill.take_teacher_actions False
  --algo.config.distill.teacher_action_mix_ratio 0.0
  --algo.config.distill.enabled False
  --algo.config.distill.mode mse
  --algo.config.distill.ppo_start_epoch -1
  --algo.config.distill.dagger_end_epoch -1
)

if [[ -n "${MOTION_CLIP_NAME}" ]]; then
  cmd+=(--command.setup_terms.motion_command.params.motion_config.motion_clip_name "${MOTION_CLIP_NAME}")
fi
if [[ -n "${MOTION_CLIP_ID}" ]]; then
  cmd+=(--command.setup_terms.motion_command.params.motion_config.motion_clip_id "${MOTION_CLIP_ID}")
fi
if [[ -n "${MAX_EVAL_STEPS:-}" ]]; then
  cmd+=(--training.max_eval_steps "${MAX_EVAL_STEPS}")
fi
if [[ -n "${PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE}" ]]; then
  cmd+=(--perception.object_geometry_mode "${PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE}")
fi

if [[ "${DISABLE_RANDOMIZATION}" == "True" || "${DISABLE_RANDOMIZATION}" == "true" ]]; then
  cmd+=(
    --randomization.setup_terms.push_randomizer_state.params.enabled False
    --randomization.reset_terms.randomize_push_schedule.params.enabled False
    --randomization.step_terms.apply_pushes.params.enabled False
    --randomization.setup_terms.actuator_randomizer_state.params.enable_pd_gain False
    --randomization.setup_terms.actuator_randomizer_state.params.enable_rfi_lim False
    --randomization.setup_terms.setup_action_delay_buffers.params.enabled False
    --randomization.reset_terms.randomize_action_delay.params.enabled False
    --randomization.setup_terms.randomize_robot_rigid_body_material_startup.params.enabled False
    --randomization.setup_terms.randomize_base_com_startup.params.enabled False
    --randomization.setup_terms.setup_dof_pos_bias.params.enabled False
    --randomization.reset_terms.randomize_dof_state.params.randomize_dof_pos_bias False
    --randomization.setup_terms.setup_camera_raycast_randomization.params.enabled False
    --randomization.reset_terms.randomize_camera_raycast.params.enabled False
    --randomization.setup_terms.randomize_object_rigid_body_material_startup.params.enabled False
    --randomization.setup_terms.randomize_object_rigid_body_mass_startup.params.enabled False
    --randomization.setup_terms.randomize_object_rigid_body_inertia_startup.params.enabled False
  )
fi

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] teacher_checkpoint=${TEACHER_CHECKPOINT}"
echo "[INFO] distill_box_perception_default_teacher=${DISTILL_DEFAULT_TEACHER_CHECKPOINT}"
echo "[INFO] infer_dataset=${INFER_DATASET}"
echo "[INFO] data_mode=${DATA_MODE}"
echo "[INFO] motion_dir=${MOTION_DIR}"
echo "[INFO] motion_clip_name=${MOTION_CLIP_NAME:-<auto>}"
echo "[INFO] motion_clip_id=${MOTION_CLIP_ID:-<auto>}"
echo "[INFO] object_urdf=${OBJECT_URDF}"
if [[ -n "${OBJECT_GEOMETRY_MODE}" ]]; then
  echo "[INFO] object_geometry_mode=${OBJECT_GEOMETRY_MODE} simulator_object_spawn_mode=${HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE}"
else
  echo "[INFO] object_geometry_mode=<default>"
fi
echo "[INFO] headless=${HEADLESS_FLAG}"
echo "[INFO] num_envs=${NUM_ENVS}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] holosoma_device=${HOLOSOMA_DEVICE:-<unset>}"
echo "[INFO] disable_randomization=${DISABLE_RANDOMIZATION}"
echo "[INFO] disable_auto_reset=${HOLOSOMA_DISABLE_AUTO_RESET} disable_clip_end_reset=${HOLOSOMA_DISABLE_CLIP_END_RESET}"
echo "[INFO] start_at_timestep_zero_prob=${START_AT_TIMESTEP_ZERO_PROB} freeze_at_timestep_zero_prob=${FREEZE_AT_TIMESTEP_ZERO_PROB} reset_noise_scale=${RESET_NOISE_SCALE}"
if [[ "${AUTO_SWITCH_MULTI_OBJECT_MODE}" == "1" ]]; then
  echo "[INFO] auto_enabled_per_clip_object_switching=True heterogeneous_single_slot_disabled=${HOLOSOMA_DISABLE_HETEROGENEOUS_OBJECT_SINGLE_SLOT:-0}"
fi

if is_truthy "${DRY_RUN}"; then
  printf '[DRY_RUN] '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${cmd[@]}"
