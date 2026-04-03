#!/usr/bin/env bash
set -euo pipefail

# Teacher-policy inference for box tracking on the same motion-box pairs used by
# train_object_generalist.sh, with Isaac Sim <-> Viser sync enabled.
#
# Branches:
# - omomo:       infer on OMOMO carry clips
# - real:        infer on behave+omomo mixed clips
# - pure-sd:     infer on prepared Seedance/DS clips
# - naive-mixed: infer on OMOMO + Seedance/DS naive-mixed clips
#
# Usage:
#   bash infer_box_tracking.sh [omomo|real|pure-sd|naive-mixed] [teacher_checkpoint.pt|model_XXXXX.pt|wandb://...] [extra tyro args...]
#
# Optional env vars:
#   TEACHER_CHECKPOINT        (default: wandb://zihanw22/boxer/a5ohxuta/model_09000.pt)
#   WANDB_MODEL_FILE          (optional; used when TEACHER_CHECKPOINT is a W&B run URL without /files/<checkpoint>)
#   LEGACY_OBS                (default: 0; set 1/true to require legacy checkpoint observation layout)
#   REQUIRE_HEIGHTMAP         (default: 0; set 1/true to require checkpoint perception.enabled=True and output_mode=heightmap)
#   DEFAULT_LEGACY_TEACHER_CHECKPOINT
#                             (optional; used as default checkpoint when LEGACY_OBS=1 and no checkpoint is explicitly provided)
#   PURE_SD_DEFAULT_TEACHER_RUN_URL
#                             (default: https://wandb.ai/zihanw22/boxer/runs/6pzxdnr6; resolved to latest checkpoint at runtime)
#   INFER_DATASET             (default: mixed; options: omomo|behave|behave_carry|behave_sq_carry|mixed|real|pure-sd|naive-mixed|mix-naive)
#   DS_DATA_ROOT              (default: ./data/ds_box_data; used by pure-sd / naive-mixed)
#   MOTION_DIR                (optional override; if unset, chosen by INFER_DATASET)
#   MOTION_CLIP_NAME          (optional: pin a single clip)
#   OBJECT_URDF               (optional override; if unset, chosen by INFER_DATASET)
#   NUM_ENVS                  (default: 1)
#   HEADLESS                  (default: True; set False for local interactive eval)
#   VISER_PORT                (default: random)
#   VISER_ENV_ID              (default: 0)
#   VISER_UPDATE_HZ           (default: 30)
#   VISER_RECENTER            (default: True)
#   VISER_SYNC_TO_SIM         (default: True)
#   VISER_FORCE_DT            (default: True)
#   VISER_LOAD_URDF           (default: 1; URDF meshes are shown in Viser, but pose/object selection comes from Isaac Sim runtime state)
#   VISER_DEFER_INIT          (default: 1; defer Viser startup until first simulator step)
#   START_AT_TIMESTEP_ZERO_PROB
#                             (default: 0.2; matches checkpoint default)
#   FREEZE_AT_TIMESTEP_ZERO_PROB
#                             (default: 0.95; matches checkpoint default)
#   ENABLE_DEFAULT_POSE_PREPEND
#                             (default: False; disable runtime default-pose warmup for more stable interactive resets)
#   DEFAULT_POSE_PREPEND_DURATION_S
#                             (default: 0.0; only used when ENABLE_DEFAULT_POSE_PREPEND=True)
#   DISABLE_RANDOMIZATION     (default: True)
#   VIS_GPU                   (default: auto; picks least-used GPU if CUDA_VISIBLE_DEVICES is unset)
#   PYTHON_BIN                (default: python)
#   WANDB_MODEL_FILE          (optional preferred model file for W&B run URLs)

usage() {
  cat <<'EOF'
Usage:
  bash infer_box_tracking.sh [omomo|real|pure-sd|naive-mixed] [teacher_checkpoint.pt|model_XXXXX.pt|wandb://...] [extra tyro args...]

Examples:
  bash infer_box_tracking.sh
  bash infer_box_tracking.sh omomo
  bash infer_box_tracking.sh real
  bash infer_box_tracking.sh pure-sd
  bash infer_box_tracking.sh pure-sd model_00500.pt
  bash infer_box_tracking.sh naive-mixed
  bash infer_box_tracking.sh /abs/path/to/model_17000.pt
  bash infer_box_tracking.sh real /abs/path/to/model_17000.pt
  bash infer_box_tracking.sh pure-sd simulator:isaacsim
  MOTION_CLIP_NAME=sub3_largebox_003_mj_w_obj bash infer_box_tracking.sh

Dataset selection examples:
  INFER_DATASET=omomo bash infer_box_tracking.sh
  INFER_DATASET=behave bash infer_box_tracking.sh
  INFER_DATASET=behave_carry bash infer_box_tracking.sh
  INFER_DATASET=mixed bash infer_box_tracking.sh
  INFER_DATASET=pure-sd bash infer_box_tracking.sh
  INFER_DATASET=naive-mixed bash infer_box_tracking.sh
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"

is_checkpoint_ref() {
  local ref="$1"
  [[ "${ref}" == wandb://* || "${ref}" == https://wandb.ai/*/runs/* || "${ref}" == /* || "${ref}" == ./* || "${ref}" == ../* || "${ref}" == *.pt ]]
}

is_bare_checkpoint_name() {
  local ref="$1"
  [[ "${ref}" == *.pt && "${ref}" != */* && "${ref}" != ./* && "${ref}" != ../* ]]
}

default_model_file_for_run_id() {
  local run_id="$1"
  case "${run_id}" in
    6pzxdnr6) echo "model_00500.pt" ;;
    *) echo "" ;;
  esac
}

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

parse_wandb_uri() {
  local ref="$1"
  if [[ "${ref}" != wandb://* ]]; then
    return 1
  fi

  local trimmed="${ref#wandb://}"
  local entity=""
  local project=""
  local run_id=""
  local explicit_file=""
  IFS='/' read -r -a parts <<< "${trimmed}"
  if [[ "${#parts[@]}" -lt 3 ]]; then
    return 1
  fi

  entity="${parts[0]}"
  project="${parts[1]}"
  run_id="${parts[2]}"
  if [[ -z "${entity}" || -z "${project}" || -z "${run_id}" ]]; then
    return 1
  fi

  if [[ "${#parts[@]}" -gt 3 ]]; then
    explicit_file="${trimmed#${entity}/${project}/${run_id}/}"
  fi

  printf '%s\t%s\t%s\t%s\n' "${entity}" "${project}" "${run_id}" "${explicit_file}"
}

parse_wandb_reference() {
  local ref="$1"
  parse_wandb_run_url "${ref}" || parse_wandb_uri "${ref}"
}

checkpoint_ref_with_model_file() {
  local base_ref="$1"
  local model_file="$2"
  local parsed=""
  local entity=""
  local project=""
  local run_id=""

  parsed="$(parse_wandb_reference "${base_ref}" || true)"
  if [[ -z "${parsed}" ]]; then
    return 1
  fi

  IFS=$'\t' read -r entity project run_id _explicit_file <<< "${parsed}"
  if [[ -z "${entity}" || -z "${project}" || -z "${run_id}" ]]; then
    return 1
  fi

  echo "wandb://${entity}/${project}/${run_id}/${model_file}"
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
  local remote_model_file=""
  local builtin_model_file=""

  parsed="$(parse_wandb_run_url "${ref}" || true)"
  if [[ -z "${parsed}" ]]; then
    echo "${ref}"
    return 0
  fi

  IFS=$'\t' read -r entity project run_id explicit_file <<< "${parsed}"
  if [[ -n "${explicit_file}" ]]; then
    model_file="${explicit_file}"
  elif [[ -z "${model_file}" ]]; then
    remote_model_file="$(resolve_remote_wandb_checkpoint_name "${entity}" "${project}" "${run_id}")"
    if [[ -n "${remote_model_file}" ]]; then
      model_file="${remote_model_file}"
      echo "[INFO] Resolved wandb run URL to latest remote checkpoint: ${model_file}" >&2
    else
      builtin_model_file="$(default_model_file_for_run_id "${run_id}")"
      if [[ -n "${builtin_model_file}" ]]; then
        model_file="${builtin_model_file}"
        echo "[INFO] Falling back to pinned checkpoint file for run ${run_id}: ${model_file}" >&2
      fi
    fi
  fi

  if [[ -z "${model_file}" ]]; then
    echo "[ERROR] Could not determine a .pt checkpoint for W&B run URL: ${ref}" >&2
    echo "[ERROR] Pass a /files/<checkpoint>.pt URL or set WANDB_MODEL_FILE." >&2
    return 2
  fi

  echo "wandb://${entity}/${project}/${run_id}/${model_file}"
}

resolve_local_checkpoint_from_run_url() {
  local ref="$1"
  local preferred_model_file="${2:-}"
  local parsed=""
  local run_id=""
  local explicit_file=""
  local wandb_run_dir=""
  local run_log_dir=""
  local local_ckpt=""
  local target_model_file=""

  parsed="$(parse_wandb_run_url "${ref}" || true)"
  if [[ -z "${parsed}" ]]; then
    echo ""
    return 0
  fi
  IFS=$'\t' read -r _entity _project run_id explicit_file <<< "${parsed}"

  target_model_file="${explicit_file}"
  if [[ -z "${target_model_file}" ]]; then
    target_model_file="${preferred_model_file}"
  fi
  if [[ -z "${target_model_file}" ]]; then
    target_model_file="$(default_model_file_for_run_id "${run_id}")"
  fi
  if [[ -z "${target_model_file}" ]]; then
    echo ""
    return 0
  fi

  wandb_run_dir="$(find /data/logs_new -maxdepth 8 -type d -name "run-*-${run_id}" 2>/dev/null | head -n 1 || true)"
  if [[ -z "${wandb_run_dir}" ]]; then
    echo ""
    return 0
  fi

  run_log_dir="$(dirname "$(dirname "$(dirname "${wandb_run_dir}")")")"
  if [[ -f "${run_log_dir}/${target_model_file}" ]]; then
    local_ckpt="${run_log_dir}/${target_model_file}"
  fi
  echo "${local_ckpt}"
}

resolve_local_checkpoint_from_wandb_ref() {
  local ref="$1"
  local parsed=""
  local run_id=""
  local explicit_file=""
  local wandb_run_dir=""
  local run_log_dir=""
  local local_ckpt=""

  parsed="$(parse_wandb_uri "${ref}" || true)"
  if [[ -z "${parsed}" ]]; then
    echo ""
    return 0
  fi
  IFS=$'\t' read -r _entity _project run_id explicit_file <<< "${parsed}"

  if [[ -z "${explicit_file}" ]]; then
    echo ""
    return 0
  fi

  wandb_run_dir="$(find /data/logs_new -maxdepth 8 -type d -name "run-*-${run_id}" 2>/dev/null | head -n 1 || true)"
  if [[ -z "${wandb_run_dir}" ]]; then
    echo ""
    return 0
  fi

  run_log_dir="$(dirname "$(dirname "$(dirname "${wandb_run_dir}")")")"
  if [[ -f "${run_log_dir}/${explicit_file}" ]]; then
    local_ckpt="${run_log_dir}/${explicit_file}"
  fi
  echo "${local_ckpt}"
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help|help)
      usage
      exit 0
      ;;
  esac
fi

# https://wandb.ai/zihanw22/boxer/runs/a5ohxuta/files?nw=nwuserz1hanw
DEFAULT_TEACHER_CHECKPOINT=${DEFAULT_TEACHER_CHECKPOINT:-"wandb://zihanw22/boxer/a5ohxuta/model_14000.pt"}
PURE_SD_DEFAULT_TEACHER_RUN_URL=${PURE_SD_DEFAULT_TEACHER_RUN_URL:-"https://wandb.ai/zihanw22/boxer/runs/6pzxdnr6"}
DEFAULT_LEGACY_TEACHER_CHECKPOINT="${DEFAULT_LEGACY_TEACHER_CHECKPOINT:-}"
LEGACY_OBS=${LEGACY_OBS:-0}
legacy_obs_normalized=$(echo "${LEGACY_OBS}" | tr '[:upper:]' '[:lower:]')
if [[ "${legacy_obs_normalized}" == "1" || "${legacy_obs_normalized}" == "true" ]]; then
  LEGACY_OBS_ENABLED=1
else
  LEGACY_OBS_ENABLED=0
fi
REQUIRE_HEIGHTMAP=${REQUIRE_HEIGHTMAP:-0}
require_heightmap_normalized=$(echo "${REQUIRE_HEIGHTMAP}" | tr '[:upper:]' '[:lower:]')
if [[ "${require_heightmap_normalized}" == "1" || "${require_heightmap_normalized}" == "true" ]]; then
  HEIGHTMAP_REQUIRED=1
else
  HEIGHTMAP_REQUIRED=0
fi

TEACHER_CHECKPOINT_FROM_ENV=0
if [[ -n "${TEACHER_CHECKPOINT+x}" || -n "${CKPT+x}" ]]; then
  TEACHER_CHECKPOINT_FROM_ENV=1
fi
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${CKPT:-${DEFAULT_TEACHER_CHECKPOINT}}}"
TEACHER_CHECKPOINT_FROM_ARG=0
TEACHER_CHECKPOINT_NAME_FROM_ARG=0
TEACHER_CHECKPOINT_MODEL_FILE=""
INFER_DATASET_FROM_ARG=0

if [[ $# -gt 0 ]]; then
  first_arg_normalized=$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr -d '[][:space:]')
  case "${first_arg_normalized}" in
    omomo)
      INFER_DATASET="omomo"
      INFER_DATASET_FROM_ARG=1
      shift
      ;;
    real|mixed|behave+omomo|behave-omomo)
      INFER_DATASET="real"
      INFER_DATASET_FROM_ARG=1
      shift
      ;;
    pure-sd|seedance|sd)
      INFER_DATASET="pure-sd"
      INFER_DATASET_FROM_ARG=1
      shift
      ;;
    naive-mixed|mix-naive)
      INFER_DATASET="naive-mixed"
      INFER_DATASET_FROM_ARG=1
      shift
      ;;
  esac
fi

if [[ $# -gt 0 ]]; then
  if is_bare_checkpoint_name "$1" && [[ ! -f "$1" ]]; then
    TEACHER_CHECKPOINT_NAME_FROM_ARG=1
    TEACHER_CHECKPOINT_MODEL_FILE="$1"
    shift
  elif is_checkpoint_ref "$1"; then
    TEACHER_CHECKPOINT="$1"
    TEACHER_CHECKPOINT_FROM_ARG=1
    shift
  fi
fi

if [[ "${LEGACY_OBS_ENABLED}" == "1" ]]; then
  if [[ "${TEACHER_CHECKPOINT_FROM_ENV}" != "1" && "${TEACHER_CHECKPOINT_FROM_ARG}" != "1" ]]; then
    if [[ -n "${DEFAULT_LEGACY_TEACHER_CHECKPOINT}" ]]; then
      TEACHER_CHECKPOINT="${DEFAULT_LEGACY_TEACHER_CHECKPOINT}"
    else
      echo "[ERROR] LEGACY_OBS=1 requires an explicit legacy checkpoint." >&2
      echo "[ERROR] Provide TEACHER_CHECKPOINT/CKPT/positional .pt, or set DEFAULT_LEGACY_TEACHER_CHECKPOINT." >&2
      exit 2
    fi
  fi
fi

pick_first_existing_path() {
  local candidate=""
  for candidate in "$@"; do
    if [[ -e "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  if [[ $# -gt 0 ]]; then
    echo "$1"
  fi
}

if [[ "${INFER_DATASET_FROM_ARG}" != "1" ]]; then
  INFER_DATASET=${INFER_DATASET:-${DATASET:-mixed}}
fi
INFER_DATASET=$(echo "${INFER_DATASET}" | tr '[:upper:]' '[:lower:]' | tr -d '[][:space:]')
case "${INFER_DATASET}" in
  omomo|behave|behave_carry|behave_sq_carry|mixed|real|pure-sd|naive-mixed|mix-naive) ;;
  *)
    echo "[ERROR] INFER_DATASET must be one of: omomo, behave, behave_carry, behave_sq_carry, mixed, real, pure-sd, naive-mixed, mix-naive. Got: ${INFER_DATASET}" >&2
    exit 2
    ;;
esac

if [[ "${INFER_DATASET}" == "pure-sd" && "${TEACHER_CHECKPOINT_FROM_ENV}" != "1" && "${TEACHER_CHECKPOINT_FROM_ARG}" != "1" && "${LEGACY_OBS_ENABLED}" != "1" ]]; then
  TEACHER_CHECKPOINT="${PURE_SD_DEFAULT_TEACHER_RUN_URL}"
fi

if [[ "${TEACHER_CHECKPOINT_NAME_FROM_ARG}" == "1" ]]; then
  resolved_checkpoint_ref="$(checkpoint_ref_with_model_file "${TEACHER_CHECKPOINT}" "${TEACHER_CHECKPOINT_MODEL_FILE}" || true)"
  if [[ -z "${resolved_checkpoint_ref}" ]]; then
    echo "[ERROR] Cannot resolve checkpoint shorthand '${TEACHER_CHECKPOINT_MODEL_FILE}' from base reference: ${TEACHER_CHECKPOINT}" >&2
    echo "[ERROR] Pass a full checkpoint path / wandb:// ref, or use a W&B run URL as the base teacher checkpoint." >&2
    exit 2
  fi
  TEACHER_CHECKPOINT="${resolved_checkpoint_ref}"
fi

if [[ "${TEACHER_CHECKPOINT}" == https://wandb.ai/*/runs/* ]]; then
  LOCAL_WANDB_CKPT="$(resolve_local_checkpoint_from_run_url "${TEACHER_CHECKPOINT}" "${WANDB_MODEL_FILE:-}")"
  if [[ -n "${LOCAL_WANDB_CKPT}" && -f "${LOCAL_WANDB_CKPT}" ]]; then
    TEACHER_CHECKPOINT="${LOCAL_WANDB_CKPT}"
    echo "[INFO] Resolved wandb run URL to local checkpoint: ${TEACHER_CHECKPOINT}"
  else
    TEACHER_CHECKPOINT="$(normalize_checkpoint_ref "${TEACHER_CHECKPOINT}")"
  fi
fi

if [[ "${TEACHER_CHECKPOINT}" == wandb://* ]]; then
  LOCAL_WANDB_CKPT="$(resolve_local_checkpoint_from_wandb_ref "${TEACHER_CHECKPOINT}")"
  if [[ -n "${LOCAL_WANDB_CKPT}" && -f "${LOCAL_WANDB_CKPT}" ]]; then
    TEACHER_CHECKPOINT="${LOCAL_WANDB_CKPT}"
    echo "[INFO] Resolved wandb reference to local checkpoint: ${TEACHER_CHECKPOINT}"
  fi
fi

DEFAULT_OMOMO_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"
DEFAULT_BEHAVE_MOTION_DIR="$(pick_first_existing_path \
  "${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/behave_carry" \
  "${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/behave_sq_carry")"
DEFAULT_REAL_MOTION_DIR="$(pick_first_existing_path \
  "${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_carry_aug_mix_ml" \
  "${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_sq_carry_aug_mix_ml")"
DS_DATA_ROOT="${DS_DATA_ROOT:-"${SCRIPT_DIR}/data/ds_box_data"}"
DEFAULT_PURE_SD_MOTION_DIR="${DS_DATA_ROOT}/train_g1_w_obj_prepared"
DEFAULT_NAIVE_MIXED_MOTION_DIR="${DS_DATA_ROOT}/train_g1_w_obj_prepared_plus_omomo_orig"
DEFAULT_OMOMO_URDF="${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"
DEFAULT_BEHAVE_MAP_FILE="${DEFAULT_BEHAVE_MOTION_DIR}/_clip_object_urdf_map.json"
DEFAULT_REAL_MAP_FILE="${DEFAULT_REAL_MOTION_DIR}/_clip_object_urdf_map.json"
DEFAULT_PURE_SD_MAP_FILE="${DEFAULT_PURE_SD_MOTION_DIR}/_clip_object_urdf_map.json"
DEFAULT_NAIVE_MIXED_MAP_FILE="${DEFAULT_NAIVE_MIXED_MOTION_DIR}/_clip_object_urdf_map.json"

MOTION_DIR_FROM_ENV=0
if [[ -n "${MOTION_DIR+x}" ]]; then
  MOTION_DIR_FROM_ENV=1
fi
OBJECT_URDF_FROM_ENV=0
if [[ -n "${OBJECT_URDF+x}" ]]; then
  OBJECT_URDF_FROM_ENV=1
fi

if [[ "${MOTION_DIR_FROM_ENV}" != "1" ]]; then
  case "${INFER_DATASET}" in
    omomo)
      MOTION_DIR="${DEFAULT_OMOMO_MOTION_DIR}"
      ;;
    behave|behave_carry|behave_sq_carry)
      MOTION_DIR="${DEFAULT_BEHAVE_MOTION_DIR}"
      ;;
    mixed|real)
      MOTION_DIR="${DEFAULT_REAL_MOTION_DIR}"
      ;;
    pure-sd)
      MOTION_DIR="${DEFAULT_PURE_SD_MOTION_DIR}"
      ;;
    naive-mixed|mix-naive)
      MOTION_DIR="${DEFAULT_NAIVE_MIXED_MOTION_DIR}"
      ;;
  esac
fi

MOTION_CLIP_NAME=${MOTION_CLIP_NAME:-}
if [[ "${OBJECT_URDF_FROM_ENV}" != "1" ]]; then
  case "${INFER_DATASET}" in
    omomo)
      OBJECT_URDF="${DEFAULT_OMOMO_URDF}"
      ;;
    behave|behave_carry|behave_sq_carry)
      if [[ -f "${DEFAULT_BEHAVE_MAP_FILE}" ]]; then
        OBJECT_URDF="${DEFAULT_BEHAVE_MAP_FILE}"
      else
        echo "[ERROR] BEHAVE map file not found: ${DEFAULT_BEHAVE_MAP_FILE}" >&2
        exit 2
      fi
      ;;
    mixed|real)
      if [[ -f "${DEFAULT_REAL_MAP_FILE}" ]]; then
        OBJECT_URDF="${DEFAULT_REAL_MAP_FILE}"
      else
        echo "[ERROR] behave+omomo map file not found: ${DEFAULT_REAL_MAP_FILE}" >&2
        exit 2
      fi
      ;;
    pure-sd)
      if [[ -f "${DEFAULT_PURE_SD_MAP_FILE}" ]]; then
        OBJECT_URDF="${DEFAULT_PURE_SD_MAP_FILE}"
      else
        echo "[ERROR] Seedance/DS map file not found: ${DEFAULT_PURE_SD_MAP_FILE}" >&2
        exit 2
      fi
      ;;
    naive-mixed|mix-naive)
      if [[ -f "${DEFAULT_NAIVE_MIXED_MAP_FILE}" ]]; then
        OBJECT_URDF="${DEFAULT_NAIVE_MIXED_MAP_FILE}"
      else
        echo "[ERROR] OMOMO+Seedance/DS mix map file not found: ${DEFAULT_NAIVE_MIXED_MAP_FILE}" >&2
        exit 2
      fi
      ;;
  esac
fi

NUM_ENVS=${NUM_ENVS:-1}
HEADLESS_RAW=${HEADLESS:-True}
HEADLESS_NORM=$(echo "${HEADLESS_RAW}" | tr '[:upper:]' '[:lower:]')
case "${HEADLESS_NORM}" in
  1|true|yes|on)
    HEADLESS_FLAG=True
    export HEADLESS=1
    ;;
  0|false|no|off|"")
    HEADLESS_FLAG=False
    export HEADLESS=0
    ;;
  *)
    echo "[ERROR] HEADLESS must be one of: 0/1/true/false/yes/no/on/off. Got: ${HEADLESS_RAW}" >&2
    exit 2
    ;;
esac
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
VISER_PORT_FROM_ENV=0
if [[ -n "${VISER_PORT+x}" && -n "${VISER_PORT}" ]]; then
  VISER_PORT_FROM_ENV=1
fi
VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_RECENTER=${VISER_RECENTER:-True}
VISER_SYNC_TO_SIM=${VISER_SYNC_TO_SIM:-True}
VISER_FORCE_DT=${VISER_FORCE_DT:-True}
VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS:-False}
VISER_LOAD_URDF=${VISER_LOAD_URDF:-1}

is_truthy() {
  local raw="${1:-}"
  case "$(echo "${raw}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

port_in_use() {
  local port="$1"
  "${PYTHON_BIN}" - "${port}" <<'PY' >/dev/null 2>&1
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("0.0.0.0", port))
except OSError:
    sys.exit(0)
finally:
    try:
        sock.close()
    except Exception:
        pass
sys.exit(1)
PY
}

if [[ ! "${VISER_PORT}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] VISER_PORT must be an integer. Got: ${VISER_PORT}" >&2
  exit 2
fi
if (( VISER_PORT < 1 || VISER_PORT > 65535 )); then
  echo "[ERROR] VISER_PORT must be in [1, 65535]. Got: ${VISER_PORT}" >&2
  exit 2
fi
if port_in_use "${VISER_PORT}"; then
  if [[ "${VISER_PORT_FROM_ENV}" == "1" ]]; then
    echo "[ERROR] VISER_PORT ${VISER_PORT} is already in use. Choose a different VISER_PORT." >&2
    exit 2
  fi
  found_port=0
  for _attempt in {1..30}; do
    candidate_port=$((RANDOM % 8976 + 1024))
    if ! port_in_use "${candidate_port}"; then
      VISER_PORT="${candidate_port}"
      found_port=1
      break
    fi
  done
  if [[ "${found_port}" != "1" ]]; then
    echo "[ERROR] Could not find a free random VISER_PORT after 30 attempts." >&2
    exit 2
  fi
fi
export HOLOSOMA_VISER_PORT="${VISER_PORT}"

if [[ -n "${RANK+x}" && "${RANK}" != "0" ]]; then
  echo "[WARN] RANK=${RANK} would disable Viser (viewer only runs on rank 0). Forcing rank-0 env vars." >&2
  export RANK=0
  export LOCAL_RANK=0
  export WORLD_SIZE=1
fi

if ! "${PYTHON_BIN}" - <<'PY' >/tmp/holosoma_viser_check.err 2>&1
import inspect
import viser

if not hasattr(viser, "ViserServer"):
    module_file = getattr(viser, "__file__", None)
    raise RuntimeError(
        f"Imported 'viser' module does not provide ViserServer "
        f"(module={module_file!r}, inspect={inspect.getmodule(viser)!r})"
    )
PY
then
  echo "[ERROR] Invalid/missing Viser runtime in ${PYTHON_BIN}." >&2
  sed -n '1,6p' /tmp/holosoma_viser_check.err >&2 || true
  echo "[ERROR] Install the correct 'viser' package in this env (must expose viser.ViserServer), or set PYTHON_BIN to an env that has it." >&2
  exit 2
fi
if is_truthy "${VISER_LOAD_URDF}"; then
  if ! "${PYTHON_BIN}" - <<'PY' >/tmp/holosoma_viser_urdf_check.err 2>&1
import viser  # noqa: F401
from viser.extras import ViserUrdf  # noqa: F401
PY
  then
    echo "[WARN] viser.extras.ViserUrdf unavailable in ${PYTHON_BIN}; falling back to VISER_LOAD_URDF=0." >&2
    sed -n '1,3p' /tmp/holosoma_viser_urdf_check.err >&2 || true
    VISER_LOAD_URDF=0
  fi
fi

START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.2}
FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.95}
ENABLE_DEFAULT_POSE_PREPEND=${ENABLE_DEFAULT_POSE_PREPEND:-False}
DEFAULT_POSE_PREPEND_DURATION_S=${DEFAULT_POSE_PREPEND_DURATION_S:-0.0}
RESET_NOISE_SCALE=${RESET_NOISE_SCALE:-0.0}
MAX_EPISODE_LENGTH_S=${MAX_EPISODE_LENGTH_S:-1000000}
SIM_ENV_SPACING=${SIM_ENV_SPACING:-0.0}
PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-67108864}
DISABLE_RANDOMIZATION=${DISABLE_RANDOMIZATION:-True}
VIS_GPU=${VIS_GPU:-auto}

# Pick a less-loaded GPU by default for IsaacSim startup stability.
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" || -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  if [[ "${VIS_GPU}" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      AUTO_GPU="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2,2n | head -n1 | cut -d, -f1 | tr -d ' ')"
      if [[ -n "${AUTO_GPU}" ]]; then
        export CUDA_VISIBLE_DEVICES="${AUTO_GPU}"
      fi
    fi
  elif [[ "${VIS_GPU}" =~ ^[0-9]+$ ]]; then
    export CUDA_VISIBLE_DEVICES="${VIS_GPU}"
  fi
fi

# Useful defaults for interactive motion/clip inspection.
export VISER_ENABLE_CLIP_GUI=${VISER_ENABLE_CLIP_GUI:-1}
export VISER_ENABLE_MANUAL_GUI=${VISER_ENABLE_MANUAL_GUI:-0}
export VISER_SHOW_TARGET_KEYPOINTS=${VISER_SHOW_TARGET_KEYPOINTS:-1}
export VISER_START_PAUSED=${VISER_START_PAUSED:-0}
export VISER_LOAD_URDF
export VISER_DEFER_INIT=${VISER_DEFER_INIT:-1}
export LOGURU_LEVEL=${LOGURU_LEVEL:-WARNING}
export PY_LOG_LEVEL=${PY_LOG_LEVEL:-WARNING}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

if [[ "${TEACHER_CHECKPOINT}" != wandb://* ]] && [[ ! -f "${TEACHER_CHECKPOINT}" ]]; then
  echo "[ERROR] teacher checkpoint not found: ${TEACHER_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -e "${MOTION_DIR}" ]]; then
  echo "[ERROR] MOTION_DIR not found: ${MOTION_DIR}" >&2
  exit 1
fi
if [[ ! -f "${OBJECT_URDF}" ]]; then
  echo "[ERROR] OBJECT_URDF not found: ${OBJECT_URDF}" >&2
  exit 1
fi

if [[ -d "${MOTION_DIR}" && -n "${MOTION_CLIP_NAME}" && ! -f "${MOTION_DIR}/${MOTION_CLIP_NAME}.npz" ]]; then
  echo "[ERROR] MOTION_CLIP_NAME not found in MOTION_DIR: ${MOTION_CLIP_NAME}.npz" >&2
  exit 2
fi

if [[ "${LEGACY_OBS_ENABLED}" == "1" || "${HEIGHTMAP_REQUIRED}" == "1" ]]; then
  "${PYTHON_BIN}" - <<'PY' "${TEACHER_CHECKPOINT}" "${LEGACY_OBS_ENABLED}" "${HEIGHTMAP_REQUIRED}" || exit 2
import sys
import tempfile
from pathlib import Path

import torch


def parse_bool(v: str) -> bool:
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _parse_wandb_reference(reference: str) -> tuple[str, str]:
    if not reference.startswith("wandb://"):
        raise ValueError("Not a wandb:// reference")
    remainder = reference[len("wandb://") :]
    parts = remainder.split("/")
    if len(parts) < 4:
        raise ValueError(
            "Invalid wandb checkpoint path. Expected wandb://<entity>/<project>/<run_id>/<checkpoint_name>"
        )
    entity, project = parts[0], parts[1]
    run_id_index = 2
    if len(parts) > 4 and parts[2] == "runs":
        run_id_index = 3
    if run_id_index >= len(parts):
        raise ValueError(
            "Invalid wandb checkpoint path. Expected wandb://<entity>/<project>/<run_id>/<checkpoint_name>"
        )
    run_id = parts[run_id_index]
    ckpt_name = "/".join(parts[run_id_index + 1 :]).strip()
    if not ckpt_name:
        raise ValueError(
            "wandb checkpoint reference must include checkpoint filename, e.g. model_12000.pt"
        )
    return f"{entity}/{project}/{run_id}", ckpt_name


def load_payload(checkpoint_ref: str):
    if checkpoint_ref.startswith("wandb://"):
        import wandb

        run_path, ckpt_name = _parse_wandb_reference(checkpoint_ref)
        run = wandb.Api().run(run_path)
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloaded = run.file(ckpt_name).download(root=tmp_dir, replace=True)
            ckpt_path = Path(downloaded.name)
            if not ckpt_path.is_absolute():
                ckpt_path = (Path.cwd() / ckpt_path).resolve()
            payload = torch.load(ckpt_path, map_location="cpu")
            return payload
    return torch.load(checkpoint_ref, map_location="cpu")


checkpoint_ref = sys.argv[1]
require_legacy = parse_bool(sys.argv[2])
require_heightmap = parse_bool(sys.argv[3])

payload = load_payload(checkpoint_ref)
cfg = payload.get("experiment_config")
if not isinstance(cfg, dict):
    raise SystemExit(f"[ERROR] checkpoint has no experiment_config dict: {checkpoint_ref}")

obs_cfg = cfg.get("observation")
groups = obs_cfg.get("groups", {}) if isinstance(obs_cfg, dict) else {}
actor_obs = groups.get("actor_obs", {}) if isinstance(groups, dict) else {}
terms = actor_obs.get("terms", {}) if isinstance(actor_obs, dict) else {}
if not isinstance(terms, dict):
    raise SystemExit(f"[ERROR] checkpoint actor_obs.terms is invalid: {checkpoint_ref}")

if require_legacy:
    legacy_forbidden = ("obj_lin_vel_b", "obj_ang_vel_b")
    present = [name for name in legacy_forbidden if name in terms]
    if present:
        raise SystemExit(
            "[ERROR] LEGACY_OBS=1 but checkpoint actor_obs is non-legacy "
            f"(contains {present}): {checkpoint_ref}"
        )

if require_heightmap:
    perception_cfg = cfg.get("perception")
    if not isinstance(perception_cfg, dict):
        raise SystemExit(
            "[ERROR] REQUIRE_HEIGHTMAP=1 but checkpoint has no perception config dict: "
            f"{checkpoint_ref}"
        )
    enabled = bool(perception_cfg.get("enabled", False))
    output_mode = str(perception_cfg.get("output_mode", "")).strip()
    if not enabled:
        raise SystemExit(
            "[ERROR] REQUIRE_HEIGHTMAP=1 but checkpoint perception.enabled is False: "
            f"{checkpoint_ref}"
        )
    if output_mode != "heightmap":
        raise SystemExit(
            "[ERROR] REQUIRE_HEIGHTMAP=1 but checkpoint perception.output_mode is "
            f"'{output_mode}' (expected 'heightmap'): {checkpoint_ref}"
        )
    if "perception_obs" not in groups:
        raise SystemExit(
            "[ERROR] REQUIRE_HEIGHTMAP=1 but observation groups has no 'perception_obs': "
            f"{checkpoint_ref}"
        )

print(
    f"[INFO] Checkpoint validation passed (legacy={require_legacy}, "
    f"heightmap={require_heightmap}): {checkpoint_ref}"
)
PY
fi

SIMULATOR_SUBCOMMAND=""
EXTRA_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    simulator:*)
      if [[ -n "${SIMULATOR_SUBCOMMAND}" && "${SIMULATOR_SUBCOMMAND}" != "${arg}" ]]; then
        echo "[ERROR] Multiple simulator subcommands requested: ${SIMULATOR_SUBCOMMAND} and ${arg}" >&2
        exit 2
      fi
      SIMULATOR_SUBCOMMAND="${arg}"
      ;;
    *)
      EXTRA_ARGS+=("${arg}")
      ;;
  esac
done

cmd=(
  "${PYTHON_BIN}" -m holosoma.visualize physics
)

if [[ -n "${SIMULATOR_SUBCOMMAND}" ]]; then
  cmd+=("${SIMULATOR_SUBCOMMAND}")
fi

cmd+=(
  --checkpoint "${TEACHER_CHECKPOINT}"
  --motion-dir "${MOTION_DIR}"
  --num-envs "${NUM_ENVS}"
  --headless "${HEADLESS_FLAG}"
  --pair-terrain-with-motion "${PAIR_TERRAIN_WITH_MOTION}"
  --viser-port "${VISER_PORT}"
  --viser-env-id "${VISER_ENV_ID}"
  --viser-update-hz "${VISER_UPDATE_HZ}"
  --viser-recenter "${VISER_RECENTER}"
  --training.viser_sync_to_sim "${VISER_SYNC_TO_SIM}"
  --training.viser_force_dt "${VISER_FORCE_DT}"
  --training.viser_show_scandots "${VISER_SHOW_SCANDOTS}"
  --simulator.config.scene.env_spacing "${SIM_ENV_SPACING}"
  --simulator.config.sim.max_episode_length_s "${MAX_EPISODE_LENGTH_S}"
  --robot.object.enabled True
  --robot.object.object_urdf_path "${OBJECT_URDF}"
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob "${START_AT_TIMESTEP_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.freeze_at_timestep_zero_prob "${FREEZE_AT_TIMESTEP_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend "${ENABLE_DEFAULT_POSE_PREPEND}"
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s "${DEFAULT_POSE_PREPEND_DURATION_S}"
  --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.overall_noise_scale "${RESET_NOISE_SCALE}"
)

if [[ "${SIMULATOR_SUBCOMMAND}" != "simulator:mujoco" ]]; then
  cmd+=(--simulator.config.sim.physx.gpu_collision_stack_size "${PHYSX_GPU_COLLISION_STACK_SIZE}")
else
  cmd+=(--randomization.ignore_unsupported True)
fi

if [[ -n "${MOTION_CLIP_NAME}" ]]; then
  cmd+=(
    --command.setup_terms.motion_command.params.motion_config.motion_clip_name "${MOTION_CLIP_NAME}"
  )
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
echo "[INFO] legacy_obs_enabled=${LEGACY_OBS_ENABLED}"
echo "[INFO] require_heightmap=${HEIGHTMAP_REQUIRED}"
echo "[INFO] infer_dataset=${INFER_DATASET}"
echo "[INFO] motion_dir=${MOTION_DIR}"
echo "[INFO] motion_clip_name=${MOTION_CLIP_NAME:-<auto>}"
echo "[INFO] object_urdf=${OBJECT_URDF}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] headless=${HEADLESS_FLAG} (env HEADLESS=${HEADLESS})"
echo "[INFO] viser=http://localhost:${VISER_PORT}"
echo "[INFO] holosoma_viser_port=${HOLOSOMA_VISER_PORT}"
echo "[INFO] viser_sync_to_sim=${VISER_SYNC_TO_SIM} viser_force_dt=${VISER_FORCE_DT}"
echo "[INFO] viser_load_urdf=${VISER_LOAD_URDF}"
echo "[INFO] viser_defer_init=${VISER_DEFER_INIT}"
if is_truthy "${VISER_DEFER_INIT}"; then
  echo "[INFO] Viser startup is deferred until the first simulator step."
fi
echo "[INFO] simulator_subcommand=${SIMULATOR_SUBCOMMAND:-<default>}"
echo "[INFO] enable_default_pose_prepend=${ENABLE_DEFAULT_POSE_PREPEND} duration_s=${DEFAULT_POSE_PREPEND_DURATION_S}"
echo "[INFO] disable_randomization=${DISABLE_RANDOMIZATION}"
if command -v hostname >/dev/null 2>&1; then
  HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  if [[ -n "${HOST_IP}" ]]; then
    echo "[INFO] Remote URL: http://${HOST_IP}:${VISER_PORT}"
    echo "[INFO] SSH tunnel example: ssh -N -L ${VISER_PORT}:localhost:${VISER_PORT} <user>@<host>"
  fi
fi

"${cmd[@]}"
