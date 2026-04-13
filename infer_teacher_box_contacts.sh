#!/usr/bin/env bash
set -euo pipefail

# Teacher-policy contact export for box sequences.
#
# Runs multiple environments in parallel, assigns clips batch-by-batch to matching
# object slots, forces every rollout to start at timestep 0, and disables rollout
# randomization/noise by default.
#
# Usage:
#   bash infer_teacher_box_contacts.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra tyro args...]
#
# Examples:
#   bash infer_teacher_box_contacts.sh
#   NUM_ENVS=16 bash infer_teacher_box_contacts.sh
#   DATA_MODE=mix-naive NUM_ENVS=12 bash infer_teacher_box_contacts.sh

usage() {
  cat <<'EOF'
Usage:
  bash infer_teacher_box_contacts.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra tyro args...]

Optional env vars:
  TEACHER_CHECKPOINT        Default: parsed from distill_box_perception.sh, else pinned fallback
  WANDB_MODEL_FILE          Optional; used when checkpoint is a W&B run URL without /files/<checkpoint>
  DATA_MODE                 Default: pure-sd; options: pure-sd|mix-naive
  DS_DATA_ROOT              Default: ./data/ds_box_data
  MOTION_DIR                Optional override
  OBJECT_URDF               Optional override; defaults to MOTION_DIR/_clip_object_urdf_map.json when present
  NUM_ENVS                  Default: 8
  HEADLESS                  Default: True
  OUTPUT_DIR                Default: outputs/teacher_box_contacts/<timestamp>
  MIN_CONTACT_FRAMES        Default: 10
  CONTACT_FORCE_THRESHOLD   Default: 1.0
  CONTACT_VOXEL_SIZE        Default: 0.01
  SUCCESS_POSITION_THRESHOLD Default: 0.10
  MAX_ROLLOUT_STEPS         Optional per-clip step cap
  DISABLE_RANDOMIZATION     Default: True
  START_AT_TIMESTEP_ZERO_PROB Default: 1.0
  FREEZE_AT_TIMESTEP_ZERO_PROB Default: 0.0
  RESET_NOISE_SCALE         Default: 0.0
  USE_ADAPTIVE_TIMESTEPS_SAMPLER Default: False
  MAX_EPISODE_LENGTH_S      Default: 1000000
  PHYSX_GPU_COLLISION_STACK_SIZE Default: 268435456
  DRY_RUN                   Default: 0
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
  "${PYTHON_BIN}" - "${SCRIPT_DIR}/distill_box_perception.sh" <<'PY' 2>/dev/null || true
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

  "${PYTHON_BIN}" - "${entity}" "${project}" "${run_id}" <<'PY' 2>/dev/null || true
import re
import sys
from pathlib import Path

repo_root = Path.cwd().resolve()
sanitized_sys_path = []
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
  fi

  if [[ -z "${model_file}" ]]; then
    echo "[ERROR] Could not determine checkpoint for W&B run URL: ${ref}" >&2
    echo "[ERROR] Pass a /files/<checkpoint>.pt URL or set WANDB_MODEL_FILE." >&2
    return 2
  fi

  echo "wandb://${entity}/${project}/${run_id}/${model_file}"
}

DISTILL_DEFAULT_TEACHER_CHECKPOINT="$(extract_default_teacher_checkpoint_from_distill_box_perception)"
DISTILL_DEFAULT_TEACHER_CHECKPOINT="${DISTILL_DEFAULT_TEACHER_CHECKPOINT:-wandb://zihanw22/boxer/u5lguxvl/model_17000.pt}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${DISTILL_DEFAULT_TEACHER_CHECKPOINT}}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    wandb://*|https://wandb.ai/*|/*|./*|../*|*.pt)
      TEACHER_CHECKPOINT="$1"
      shift
      ;;
  esac
fi

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

DATA_MODE="${DATA_MODE:-pure-sd}"
DATA_MODE="$(echo "${DATA_MODE}" | tr '[:upper:]' '[:lower:]')"
DS_DATA_ROOT="${DS_DATA_ROOT:-${SCRIPT_DIR}/data/ds_box_data}"
DEFAULT_PURE_SD_MOTION_DIR="${DS_DATA_ROOT}/train_g1_w_obj_prepared"
DEFAULT_MIX_NAIVE_MOTION_DIR="${DS_DATA_ROOT}/train_g1_w_obj_prepared_plus_omomo_orig"
DEFAULT_SINGLE_OBJECT_URDF="${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"

case "${DATA_MODE}" in
  pure-sd|pure-ds)
    DATA_MODE="pure-sd"
    MOTION_DIR="${MOTION_DIR:-${DEFAULT_PURE_SD_MOTION_DIR}}"
    ;;
  mix-naive|mix|default)
    DATA_MODE="mix-naive"
    MOTION_DIR="${MOTION_DIR:-${DEFAULT_MIX_NAIVE_MOTION_DIR}}"
    ;;
  *)
    echo "[ERROR] DATA_MODE must be one of: pure-sd|mix-naive. Got: ${DATA_MODE}" >&2
    exit 2
    ;;
esac

if [[ -z "${OBJECT_URDF+x}" ]]; then
  DEFAULT_OBJECT_MAP="${MOTION_DIR}/_clip_object_urdf_map.json"
  if [[ -f "${DEFAULT_OBJECT_MAP}" ]]; then
    OBJECT_URDF="${DEFAULT_OBJECT_MAP}"
  else
    OBJECT_URDF="${DEFAULT_SINGLE_OBJECT_URDF}"
  fi
fi

NUM_ENVS="${NUM_ENVS:-8}"
HEADLESS="${HEADLESS:-True}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/teacher_box_contacts}"
MIN_CONTACT_FRAMES="${MIN_CONTACT_FRAMES:-10}"
CONTACT_FORCE_THRESHOLD="${CONTACT_FORCE_THRESHOLD:-1.0}"
CONTACT_VOXEL_SIZE="${CONTACT_VOXEL_SIZE:-0.01}"
SUCCESS_POSITION_THRESHOLD="${SUCCESS_POSITION_THRESHOLD:-0.10}"
DISABLE_RANDOMIZATION="${DISABLE_RANDOMIZATION:-True}"
START_AT_TIMESTEP_ZERO_PROB="${START_AT_TIMESTEP_ZERO_PROB:-1.0}"
FREEZE_AT_TIMESTEP_ZERO_PROB="${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}"
RESET_NOISE_SCALE="${RESET_NOISE_SCALE:-0.0}"
USE_ADAPTIVE_TIMESTEPS_SAMPLER="${USE_ADAPTIVE_TIMESTEPS_SAMPLER:-False}"
MAX_EPISODE_LENGTH_S="${MAX_EPISODE_LENGTH_S:-1000000}"
PHYSX_GPU_COLLISION_STACK_SIZE="${PHYSX_GPU_COLLISION_STACK_SIZE:-268435456}"
MAX_ROLLOUT_STEPS="${MAX_ROLLOUT_STEPS:-}"
DRY_RUN="${DRY_RUN:-0}"

normalize_bool_flag() {
  local value="${1:-}"
  case "$(echo "${value}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) echo "True" ;;
    0|false|no|off) echo "False" ;;
    *)
      echo "[ERROR] Invalid boolean value: ${value}" >&2
      exit 2
      ;;
  esac
}

HEADLESS_FLAG="$(normalize_bool_flag "${HEADLESS}")"
if [[ "${HEADLESS_FLAG}" == "True" ]]; then
  export HEADLESS=1
else
  export HEADLESS=0
fi
export HOLOSOMA_DISABLE_HETEROGENEOUS_OBJECT_SINGLE_SLOT=1

cmd=(
  "${PYTHON_BIN}" src/holosoma/holosoma/export_teacher_box_contacts.py
  --checkpoint "${TEACHER_CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --min-contact-frames "${MIN_CONTACT_FRAMES}"
  --contact-force-threshold "${CONTACT_FORCE_THRESHOLD}"
  --contact-voxel-size "${CONTACT_VOXEL_SIZE}"
  --success-position-threshold "${SUCCESS_POSITION_THRESHOLD}"
  --training.num-envs "${NUM_ENVS}"
  --training.headless "${HEADLESS_FLAG}"
  --simulator.config.sim.max_episode_length_s "${MAX_EPISODE_LENGTH_S}"
  --simulator.config.sim.physx.gpu_collision_stack_size "${PHYSX_GPU_COLLISION_STACK_SIZE}"
  --robot.object.enabled True
  --robot.object.object-urdf-path "${OBJECT_URDF}"
  --command.setup-terms.motion-command.params.motion-config.motion-file "${MOTION_DIR}"
  --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler "${USE_ADAPTIVE_TIMESTEPS_SAMPLER}"
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob "${START_AT_TIMESTEP_ZERO_PROB}"
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob "${FREEZE_AT_TIMESTEP_ZERO_PROB}"
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale "${RESET_NOISE_SCALE}"
)

if [[ -n "${MAX_ROLLOUT_STEPS}" ]]; then
  cmd+=(--max-rollout-steps "${MAX_ROLLOUT_STEPS}")
fi

if [[ "${DISABLE_RANDOMIZATION}" == "True" || "${DISABLE_RANDOMIZATION}" == "true" ]]; then
  cmd+=(randomization:disabled)
fi

if [[ $# -gt 0 ]]; then
  cmd+=("$@")
fi

echo "[INFO] teacher_checkpoint=${TEACHER_CHECKPOINT}"
echo "[INFO] data_mode=${DATA_MODE}"
echo "[INFO] motion_dir=${MOTION_DIR}"
echo "[INFO] object_urdf=${OBJECT_URDF}"
echo "[INFO] num_envs=${NUM_ENVS}"
echo "[INFO] start_at_timestep_zero_prob=${START_AT_TIMESTEP_ZERO_PROB}"
echo "[INFO] freeze_at_timestep_zero_prob=${FREEZE_AT_TIMESTEP_ZERO_PROB}"
echo "[INFO] reset_noise_scale=${RESET_NOISE_SCALE}"
echo "[INFO] use_adaptive_timesteps_sampler=${USE_ADAPTIVE_TIMESTEPS_SAMPLER}"
echo "[INFO] disable_randomization=${DISABLE_RANDOMIZATION}"
echo "[INFO] output_dir=${OUTPUT_DIR}"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" || "${DRY_RUN}" == "True" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${cmd[@]}"
