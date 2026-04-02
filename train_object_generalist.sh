#!/usr/bin/env bash
set -euo pipefail

# Generalist whole-body tracking training with dynamic object from a motion bank directory.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SIM_ENV_BIN=/home/ubuntu/miniconda3/envs/sim/bin
if ! command -v torchrun >/dev/null 2>&1 && [[ -x "${SIM_ENV_BIN}/torchrun" ]]; then
  export PATH="${SIM_ENV_BIN}:${PATH}"
fi
if [[ -x "${SIM_ENV_BIN}/python" ]]; then
  DEFAULT_PYTHON_BIN="${SIM_ENV_BIN}/python"
else
  DEFAULT_PYTHON_BIN="$(command -v python)"
fi
PYTHON_BIN=${PYTHON_BIN:-"${DEFAULT_PYTHON_BIN}"}

DEFAULT_CUDA_VISIBLE_DEVICES=4,5,6,7
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}
WANDB_PROJECT_FROM_ENV=0
if [[ -n "${WANDB_PROJECT+x}" ]]; then
  WANDB_PROJECT_FROM_ENV=1
fi
EXP=${EXP:-g1-29dof-wbt-w-object-generalist}
WANDB_PROJECT=${WANDB_PROJECT:-boxer}
WANDB_ENTITY=${WANDB_ENTITY:-""}
WANDB_RUN_ID=${WANDB_RUN_ID:-${RESUME_WANDB_ID:-""}}
WANDB_RESUME=${WANDB_RESUME:-""}
WANDB_RESUME_SAME_RUN=${WANDB_RESUME_SAME_RUN:-auto}
WANDB_MODEL_FILE=${WANDB_MODEL_FILE:-${RESUME_MODEL_FILE:-""}}
RESUME_CKPT=${RESUME_CKPT:-${RESUME_CHECKPOINT:-""}}
RESUME_STEP_RAW=${RESUME_STEP:-""}
DEFAULT_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_sq_carry_aug_mix_ml"
MOTION_DIR_FROM_ENV=0
if [[ -n "${MOTION_DIR+x}" ]]; then
  MOTION_DIR_FROM_ENV=1
fi
MOTION_DIR=${MOTION_DIR:-"${DEFAULT_MOTION_DIR}"}
OBJECT_SPEC_PATH=${OBJECT_SPEC_PATH:-""}
NUM_ENVS=${NUM_ENVS:-65536}
NPROC=${NPROC:-$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")}
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
PHYSX_GPU_MAX_RIGID_PATCH_COUNT=${PHYSX_GPU_MAX_RIGID_PATCH_COUNT:-655360}

TRAIN_DATASETS=${TRAIN_DATASETS:-"omomo,behave"}
AUTO_PREP_MIXED_BANK=${AUTO_PREP_MIXED_BANK:-1}
MIXED_CLEAN_OUT=${MIXED_CLEAN_OUT:-1}
MIXED_LINK_MODE=${MIXED_LINK_MODE:-symlink}
MIXED_BEHAVE_FILTER=${MIXED_BEHAVE_FILTER:-boxmedium,boxlarge}
MIXED_OMOMO_DIR=${MIXED_OMOMO_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"}
MIXED_BEHAVE_DIR=${MIXED_BEHAVE_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/behave_sq_carry"}
MIXED_BEHAVE_MAP_FILE=${MIXED_BEHAVE_MAP_FILE:-"${MIXED_BEHAVE_DIR}/_clip_object_urdf_map.json"}
DEFAULT_POSE_PREPEND_ENABLED=${DEFAULT_POSE_PREPEND_ENABLED:-1}
DEFAULT_POSE_PREPEND_DURATION_S=${DEFAULT_POSE_PREPEND_DURATION_S:-0.2}

VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_SYNC_TO_SIM=${VISER_SYNC_TO_SIM:-True}
VISER_FORCE_DT=${VISER_FORCE_DT:-True}
VISER_RECENTER=${VISER_RECENTER:-True}
VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS:-False}
VISER_LOAD_URDF=${VISER_LOAD_URDF:-1}
ENABLE_VISER=${ENABLE_VISER:-0}
DEBUG_MODE=${DEBUG_MODE:-${DEBUG_MODEL:-off}}
CURRICULUM=${CURRICULUM:-0}
PERCEPTION=${PERCEPTION:-none}
LEGACY_OBS=${LEGACY_OBS:-0}
GENERALIST_CONTACT_REWARD_ENABLED=${GENERALIST_CONTACT_REWARD_ENABLED:-1}
GENERALIST_CONTACT_REWARD_MODE=${GENERALIST_CONTACT_REWARD_MODE:-tanh}
GENERALIST_CONTACT_REWARD_THRESHOLD=${GENERALIST_CONTACT_REWARD_THRESHOLD:-1.0}
GENERALIST_CONTACT_REWARD_FORCE_SCALE=${GENERALIST_CONTACT_REWARD_FORCE_SCALE:-25.0}
GENERALIST_TORSO_CONTACT_REWARD_WEIGHT=${GENERALIST_TORSO_CONTACT_REWARD_WEIGHT:-0.30}
GENERALIST_ARM_CONTACT_REWARD_WEIGHT=${GENERALIST_ARM_CONTACT_REWARD_WEIGHT:-0.20}
GENERALIST_PALM_CONTACT_REWARD_WEIGHT=${GENERALIST_PALM_CONTACT_REWARD_WEIGHT:-0.10}

normalize_resume_step() {
  local raw="$1"
  local compact="${raw//[[:space:]_]/}"
  if [[ -z "${compact}" ]]; then
    echo ""
    return 0
  fi
  if [[ "${compact}" =~ ^([0-9]+)[kK]$ ]]; then
    echo $((10#${BASH_REMATCH[1]} * 1000))
    return 0
  fi
  if [[ "${compact}" =~ ^[0-9]+$ ]]; then
    echo $((10#${compact}))
    return 0
  fi
  return 1
}

RESUME_STEP=""
if ! RESUME_STEP="$(normalize_resume_step "${RESUME_STEP_RAW}")"; then
  echo "[ERROR] RESUME_STEP must be an integer step count or '<N>k'. Got: ${RESUME_STEP_RAW}" >&2
  exit 2
fi

is_checkpoint_ref() {
  local ref="$1"
  [[ "${ref}" == wandb://* || "${ref}" == https://wandb.ai/* || "${ref}" == /* || "${ref}" == ./* || "${ref}" == ../* || "${ref}" == *.pt ]]
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

resolve_remote_wandb_checkpoint_name() {
  local entity="$1"
  local project="$2"
  local run_id="$3"
  local requested_step="${4:-}"

  "${PYTHON_BIN}" - "${entity}" "${project}" "${run_id}" "${requested_step}" <<'PY' 2>/dev/null || true
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

entity, project, run_id, requested_step = sys.argv[1:5]
requested_step_int = int(requested_step) if requested_step else None
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
    if requested_step_int is not None:
        if step == requested_step_int:
            print(name)
            sys.exit(0)
        continue
    if step >= latest_step:
        latest_step = step
        latest_name = name
if latest_name and requested_step_int is None:
    print(latest_name)
PY
}

find_local_checkpoint_by_step() {
  local run_log_dir="$1"
  local requested_step="$2"
  local candidate=""
  local file=""
  local basename=""
  local file_step=""
  while IFS= read -r file; do
    basename="$(basename "${file}")"
    if [[ "${basename}" =~ ^model_0*([0-9]+)\.pt$ ]]; then
      file_step=$((10#${BASH_REMATCH[1]}))
      if (( file_step == requested_step )); then
        candidate="${file}"
        break
      fi
    fi
  done < <(find "${run_log_dir}" -maxdepth 1 -type f -name 'model_*.pt' | sort -V)
  echo "${candidate}"
}

resolve_local_checkpoint_from_wandb_ref() {
  local ref="$1"
  local parsed=""
  local run_id=""
  local explicit_file=""
  local wandb_run_dir=""
  local run_log_dir=""
  local local_ckpt=""

  parsed="$(parse_wandb_reference "${ref}" || true)"
  if [[ -z "${parsed}" ]]; then
    echo ""
    return 0
  fi
  IFS=$'\t' read -r _entity _project run_id explicit_file <<< "${parsed}"

  wandb_run_dir=$(find /data/logs_new -maxdepth 8 -type d -name "run-*-${run_id}" 2>/dev/null | head -n 1 || true)
  if [[ -z "${wandb_run_dir}" ]]; then
    echo ""
    return 0
  fi

  run_log_dir="$(dirname "$(dirname "$(dirname "${wandb_run_dir}")")")"
  if [[ -n "${explicit_file}" && -f "${run_log_dir}/${explicit_file}" ]]; then
    local_ckpt="${run_log_dir}/${explicit_file}"
  elif [[ -n "${WANDB_MODEL_FILE}" && -f "${run_log_dir}/${WANDB_MODEL_FILE}" ]]; then
    local_ckpt="${run_log_dir}/${WANDB_MODEL_FILE}"
  elif [[ -n "${RESUME_STEP}" ]]; then
    local_ckpt="$(find_local_checkpoint_by_step "${run_log_dir}" "${RESUME_STEP}")"
  else
    local_ckpt=$(ls -1 "${run_log_dir}"/model_*.pt 2>/dev/null | sort -V | tail -n 1 || true)
  fi
  echo "${local_ckpt}"
}

normalize_resume_checkpoint_ref() {
  local ref="$1"
  local parsed=""
  local entity=""
  local project=""
  local run_id=""
  local explicit_file=""
  local model_file=""

  parsed="$(parse_wandb_reference "${ref}" || true)"
  if [[ -z "${parsed}" ]]; then
    echo "${ref}"
    return 0
  fi
  IFS=$'\t' read -r entity project run_id explicit_file <<< "${parsed}"
  model_file="${explicit_file}"

  if [[ -z "${model_file}" && -n "${WANDB_MODEL_FILE}" ]]; then
    model_file="${WANDB_MODEL_FILE}"
    echo "[INFO] Resolved wandb reference to requested checkpoint file: ${model_file}" >&2
  fi

  if [[ -z "${model_file}" && -n "${RESUME_STEP}" ]]; then
    model_file="$(resolve_remote_wandb_checkpoint_name "${entity}" "${project}" "${run_id}" "${RESUME_STEP}")"
    if [[ -n "${model_file}" ]]; then
      echo "[INFO] Resolved wandb reference to step ${RESUME_STEP} checkpoint: ${model_file}" >&2
    fi
  fi

  if [[ -z "${model_file}" ]]; then
    model_file="$(resolve_remote_wandb_checkpoint_name "${entity}" "${project}" "${run_id}" "")"
    if [[ -n "${model_file}" ]]; then
      echo "[INFO] Resolved wandb reference to latest remote checkpoint: ${model_file}" >&2
    fi
  fi

  if [[ -z "${model_file}" ]]; then
    echo "[ERROR] Could not determine a .pt checkpoint for W&B reference: ${ref}" >&2
    echo "[ERROR] Pass a /files/<checkpoint>.pt URL, set WANDB_MODEL_FILE, or set RESUME_STEP." >&2
    exit 1
  fi

  echo "wandb://${entity}/${project}/${run_id}/${model_file}"
}

SEQUENCE_NAME=${SEQUENCE_NAME:-""}
if [[ "$#" -gt 0 ]]; then
  if is_checkpoint_ref "$1"; then
    RESUME_CKPT="$1"
    shift
  else
    SEQUENCE_NAME="$1"
    shift
    if [[ "$#" -gt 0 ]] && is_checkpoint_ref "$1"; then
      RESUME_CKPT="$1"
      shift
    fi
  fi
fi
EXTRA_ARGS=("$@")
if [[ -n "${SEQUENCE_NAME}" ]]; then
  echo "[INFO] Sequence name: ${SEQUENCE_NAME}"
fi

RESUME_WANDB_ENTITY=""
RESUME_WANDB_PROJECT=""
RESUME_WANDB_RUN_ID=""
if [[ -n "${RESUME_CKPT}" ]]; then
  RESUME_SOURCE_REF="${RESUME_CKPT}"
  parsed_resume_ref="$(parse_wandb_reference "${RESUME_SOURCE_REF}" || true)"
  if [[ -n "${parsed_resume_ref}" ]]; then
    IFS=$'\t' read -r RESUME_WANDB_ENTITY RESUME_WANDB_PROJECT RESUME_WANDB_RUN_ID _resume_explicit_file <<< "${parsed_resume_ref}"
    LOCAL_WANDB_CKPT="$(resolve_local_checkpoint_from_wandb_ref "${RESUME_SOURCE_REF}")"
    if [[ -n "${LOCAL_WANDB_CKPT}" && -f "${LOCAL_WANDB_CKPT}" ]]; then
      RESUME_CKPT="${LOCAL_WANDB_CKPT}"
      echo "[INFO] Resolved wandb reference to local checkpoint: ${RESUME_CKPT}"
    else
      RESUME_CKPT="$(normalize_resume_checkpoint_ref "${RESUME_SOURCE_REF}")"
    fi
  fi

  if [[ "${RESUME_CKPT}" != wandb://* ]] && [[ ! -f "${RESUME_CKPT}" ]]; then
    echo "[ERROR] Resume checkpoint not found: ${RESUME_CKPT}" >&2
    exit 1
  fi
  echo "[INFO] Resume checkpoint: ${RESUME_CKPT}"
fi

AUTO_ATTACH_WANDB_RUN=0
resume_same_run_normalized=$(echo "${WANDB_RESUME_SAME_RUN}" | tr '[:upper:]' '[:lower:]')
case "${resume_same_run_normalized}" in
  auto|"")
    if [[ -n "${RESUME_WANDB_RUN_ID}" ]]; then
      AUTO_ATTACH_WANDB_RUN=1
    fi
    ;;
  1|true|yes|on)
    AUTO_ATTACH_WANDB_RUN=1
    ;;
  0|false|no|off)
    AUTO_ATTACH_WANDB_RUN=0
    ;;
  *)
    echo "[ERROR] WANDB_RESUME_SAME_RUN must be one of: auto, 0/1, true/false, yes/no, on/off. Got: ${WANDB_RESUME_SAME_RUN}" >&2
    exit 2
    ;;
esac

if [[ "${AUTO_ATTACH_WANDB_RUN}" == "1" && -n "${RESUME_WANDB_RUN_ID}" ]]; then
  if [[ "${WANDB_PROJECT_FROM_ENV}" != "1" ]]; then
    WANDB_PROJECT="${RESUME_WANDB_PROJECT}"
  fi
  if [[ -z "${WANDB_ENTITY}" ]]; then
    WANDB_ENTITY="${RESUME_WANDB_ENTITY}"
  fi
  if [[ -z "${WANDB_RUN_ID}" ]]; then
    WANDB_RUN_ID="${RESUME_WANDB_RUN_ID}"
  fi
  if [[ -z "${WANDB_RESUME}" ]]; then
    WANDB_RESUME="must"
  fi
  echo "[INFO] W&B same-run resume enabled: ${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_RUN_ID} (resume=${WANDB_RESUME})"
fi

datasets_normalized=$(echo "${TRAIN_DATASETS}" | tr '[:upper:]' '[:lower:]' | tr -d '[]')
IFS=',' read -r -a dataset_tokens <<< "${datasets_normalized}"
USE_OMOMO=0
USE_BEHAVE=0
for token in "${dataset_tokens[@]}"; do
  dataset_key=$(echo "${token}" | tr -d '[:space:]')
  if [[ -z "${dataset_key}" ]]; then
    continue
  fi
  case "${dataset_key}" in
    omomo)
      USE_OMOMO=1
      ;;
    behave)
      USE_BEHAVE=1
      ;;
    *)
      echo "[ERROR] Unsupported dataset '${dataset_key}' in TRAIN_DATASETS='${TRAIN_DATASETS}'. Use only omomo,behave." >&2
      exit 2
      ;;
  esac
done
if [[ "${USE_OMOMO}" != "1" && "${USE_BEHAVE}" != "1" ]]; then
  echo "[ERROR] TRAIN_DATASETS='${TRAIN_DATASETS}' selected no datasets. Use omomo and/or behave." >&2
  exit 2
fi

selected_datasets=()
if [[ "${USE_OMOMO}" == "1" ]]; then
  selected_datasets+=("omomo")
fi
if [[ "${USE_BEHAVE}" == "1" ]]; then
  selected_datasets+=("behave")
fi
if [[ "${MOTION_DIR_FROM_ENV}" != "1" ]]; then
  if [[ "${USE_OMOMO}" == "1" && "${USE_BEHAVE}" == "1" ]]; then
    MOTION_DIR="${DEFAULT_MOTION_DIR}"
  elif [[ "${USE_OMOMO}" == "1" ]]; then
    MOTION_DIR="${MIXED_OMOMO_DIR}"
  else
    MOTION_DIR="${MIXED_BEHAVE_DIR}"
  fi
fi
echo "[INFO] TRAIN_DATASETS (resolved): $(IFS=,; echo "${selected_datasets[*]}")"
echo "[INFO] MOTION_DIR: ${MOTION_DIR}"

if [[ "${AUTO_PREP_MIXED_BANK}" != "0" ]]; then
  if [[ "${USE_OMOMO}" == "1" && "${USE_BEHAVE}" == "1" ]]; then
    echo "[INFO] Preparing mixed motion bank into: ${MOTION_DIR}"
    OMOMO_DIR="${MIXED_OMOMO_DIR}" \
    BEHAVE_DIR="${MIXED_BEHAVE_DIR}" \
    OUT_DIR="${MOTION_DIR}" \
    BEHAVE_FILTER="${MIXED_BEHAVE_FILTER}" \
    LINK_MODE="${MIXED_LINK_MODE}" \
    CLEAN_OUT="${MIXED_CLEAN_OUT}" \
    BEHAVE_MAP_FILE="${MIXED_BEHAVE_MAP_FILE}" \
    PREFIX_DATASET=1 \
    bash "${SCRIPT_DIR}/prepare_mixed_object_bank.sh"
  else
    echo "[INFO] AUTO_PREP_MIXED_BANK is enabled but skipped for single-dataset training ($(IFS=,; echo "${selected_datasets[*]}"))."
  fi
fi

if [[ -z "${OBJECT_SPEC_PATH}" ]]; then
  default_map="${MOTION_DIR}/_clip_object_urdf_map.json"
  if [[ -f "${default_map}" ]]; then
    OBJECT_SPEC_PATH="${default_map}"
    echo "[INFO] Using clip-object URDF map: ${OBJECT_SPEC_PATH}"
  elif [[ "${USE_BEHAVE}" == "1" && -f "${MIXED_BEHAVE_MAP_FILE}" ]]; then
    OBJECT_SPEC_PATH="${MIXED_BEHAVE_MAP_FILE}"
    echo "[INFO] Using BEHAVE clip-object URDF map: ${OBJECT_SPEC_PATH}"
  elif [[ "${USE_BEHAVE}" == "1" ]]; then
    echo "[WARN] BEHAVE selected but no clip-object URDF map found. Training may fallback to single-object URDF." >&2
  fi
fi

# BEHAVE requires per-clip URDF mapping; do not silently fall back to a single URDF.
if [[ "${USE_BEHAVE}" == "1" ]]; then
  if [[ -z "${OBJECT_SPEC_PATH}" || ! -f "${OBJECT_SPEC_PATH}" ]]; then
    echo "[ERROR] BEHAVE training requires a valid _clip_object_urdf_map.json, but OBJECT_SPEC_PATH is missing." >&2
    echo "[ERROR] Expected map example: ${MIXED_BEHAVE_DIR}/_clip_object_urdf_map.json" >&2
    exit 2
  fi
  python - <<'PY' "${OBJECT_SPEC_PATH}" || exit 2
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
    payload = payload["clips"]
if not isinstance(payload, dict) or not payload:
    raise SystemExit(f"[ERROR] Invalid or empty object map: {path}")
has_urdf = False
for entry in payload.values():
    if isinstance(entry, str):
        urdf = entry.strip()
    elif isinstance(entry, dict):
        urdf = str(entry.get("object_urdf_path", "")).strip()
    else:
        urdf = ""
    if urdf:
        has_urdf = True
        break
if not has_urdf:
    raise SystemExit(f"[ERROR] Object map has no valid object_urdf_path entries: {path}")
print(f"[INFO] Validated BEHAVE object map: {path}")
PY
fi

DEBUG_MODE=$(echo "${DEBUG_MODE}" | tr '[:upper:]' '[:lower:]')
case "${DEBUG_MODE}" in
  ""|0|off|none)
    DEBUG_MODE="off"
    ;;
  1|replay)
    DEBUG_MODE="replay"
    ;;
  toy)
    DEBUG_MODE="toy"
    ;;
  *)
    echo "[ERROR] Unsupported DEBUG_MODE='${DEBUG_MODE}'. Use one of: off, replay, toy"
    exit 2
    ;;
esac

if [[ "${DEBUG_MODE}" == "replay" || "${DEBUG_MODE}" == "toy" ]]; then
  if [[ -n "${OBJECT_SPEC_PATH}" && -f "${OBJECT_SPEC_PATH}" ]]; then
    DEBUG_URDF_COUNT=$(python - <<'PY' "${OBJECT_SPEC_PATH}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
    payload = payload["clips"]
if not isinstance(payload, dict):
    print(0)
    raise SystemExit(0)

seen = set()
for _, entry in payload.items():
    if isinstance(entry, str):
        urdf = entry.strip()
    elif isinstance(entry, dict):
        urdf = str(entry.get("object_urdf_path", "")).strip()
    else:
        urdf = ""
    if urdf:
        seen.add(str(Path(urdf).resolve()))
print(len(seen))
PY
)
    if [[ "${DEBUG_URDF_COUNT}" =~ ^[0-9]+$ ]] && (( DEBUG_URDF_COUNT > 0 )); then
      NUM_ENVS="${DEBUG_URDF_COUNT}"
      echo "[INFO] DEBUG_MODE=${DEBUG_MODE}: using one env per unique URDF => NUM_ENVS=${NUM_ENVS}"
    else
      echo "[WARN] DEBUG_MODE=${DEBUG_MODE}: failed to infer URDF count from ${OBJECT_SPEC_PATH}; keeping NUM_ENVS=${NUM_ENVS}"
    fi
  else
    echo "[WARN] DEBUG_MODE=${DEBUG_MODE}: OBJECT_SPEC_PATH missing; keeping NUM_ENVS=${NUM_ENVS}"
  fi
  ENABLE_VISER=1
  NPROC=1
fi

if [[ "${ENABLE_VISER}" == "1" ]]; then
  echo "[INFO] Starting training with live Viser on port ${VISER_PORT}"
  echo "[INFO] Open: http://localhost:${VISER_PORT}"
  echo "[INFO] Viser runtime source: Isaac Sim state; URDF mesh loading in Viser = ${VISER_LOAD_URDF}"
else
  echo "[INFO] Starting training without Viser"
fi

legacy_obs_normalized=$(echo "${LEGACY_OBS}" | tr '[:upper:]' '[:lower:]')
if [[ "${legacy_obs_normalized}" == "1" || "${legacy_obs_normalized}" == "true" ]]; then
  if [[ "${EXP}" == "g1-29dof-wbt-w-object-generalist" ]]; then
    EXP="g1-29dof-wbt-w-object-generalist-legacy-obs"
  fi
  echo "[INFO] LEGACY_OBS enabled: using legacy actor observation (175-dim, no object velocity terms)."
  echo "[INFO] Resolved EXP: ${EXP}"
fi

contact_reward_enabled_normalized=$(echo "${GENERALIST_CONTACT_REWARD_ENABLED}" | tr '[:upper:]' '[:lower:]')
case "${contact_reward_enabled_normalized}" in
  1|true|yes|on)
    GENERALIST_CONTACT_REWARD_ENABLED_FLAG=1
    ;;
  0|false|no|off|"")
    GENERALIST_CONTACT_REWARD_ENABLED_FLAG=0
    ;;
  *)
    echo "[ERROR] GENERALIST_CONTACT_REWARD_ENABLED must be one of: 0/1/true/false/yes/no/on/off. Got: ${GENERALIST_CONTACT_REWARD_ENABLED}" >&2
    exit 2
    ;;
esac

if [[ "${GENERALIST_CONTACT_REWARD_ENABLED_FLAG}" != "1" ]]; then
  GENERALIST_TORSO_CONTACT_REWARD_WEIGHT=0.0
  GENERALIST_ARM_CONTACT_REWARD_WEIGHT=0.0
  GENERALIST_PALM_CONTACT_REWARD_WEIGHT=0.0
fi

default_pose_prepend_enabled_normalized=$(echo "${DEFAULT_POSE_PREPEND_ENABLED}" | tr '[:upper:]' '[:lower:]')
case "${default_pose_prepend_enabled_normalized}" in
  1|true|yes|on)
    DEFAULT_POSE_PREPEND_ENABLED_FLAG=True
    ;;
  0|false|no|off)
    DEFAULT_POSE_PREPEND_ENABLED_FLAG=False
    ;;
  *)
    echo "[ERROR] DEFAULT_POSE_PREPEND_ENABLED must be one of: 0/1/true/false/yes/no/on/off. Got: ${DEFAULT_POSE_PREPEND_ENABLED}" >&2
    exit 2
    ;;
esac

echo "[INFO] Generalist contact reward enabled: ${GENERALIST_CONTACT_REWARD_ENABLED_FLAG}"
echo "[INFO] Generalist contact reward mode=${GENERALIST_CONTACT_REWARD_MODE} threshold=${GENERALIST_CONTACT_REWARD_THRESHOLD} force_scale=${GENERALIST_CONTACT_REWARD_FORCE_SCALE}"
echo "[INFO] Generalist contact reward weights torso=${GENERALIST_TORSO_CONTACT_REWARD_WEIGHT} arms=${GENERALIST_ARM_CONTACT_REWARD_WEIGHT} palms=${GENERALIST_PALM_CONTACT_REWARD_WEIGHT}"
echo "[INFO] Motion default-pose prepend enabled: ${DEFAULT_POSE_PREPEND_ENABLED_FLAG}"
echo "[INFO] Motion default-pose prepend duration: ${DEFAULT_POSE_PREPEND_DURATION_S}s"

train_cmd=(
  src/holosoma/holosoma/train_agent.py
  "exp:${EXP}"
  "perception:${PERCEPTION}"
  --training.project="${WANDB_PROJECT}"
  --training.num-envs="${NUM_ENVS}"
  --command.setup-terms.motion-command.params.motion-config.motion-file "${MOTION_DIR}"
  --algo.config.save-interval=500
  --simulator.config.sim.physx.gpu-max-rigid-patch-count="${PHYSX_GPU_MAX_RIGID_PATCH_COUNT}"
  --reward.terms.body_contact_reward_torso.weight="${GENERALIST_TORSO_CONTACT_REWARD_WEIGHT}"
  --reward.terms.body_contact_reward_arms.weight="${GENERALIST_ARM_CONTACT_REWARD_WEIGHT}"
  --reward.terms.body_contact_reward_palms.weight="${GENERALIST_PALM_CONTACT_REWARD_WEIGHT}"
  --reward.terms.body_contact_reward_torso.params.reward_mode="${GENERALIST_CONTACT_REWARD_MODE}"
  --reward.terms.body_contact_reward_arms.params.reward_mode="${GENERALIST_CONTACT_REWARD_MODE}"
  --reward.terms.body_contact_reward_palms.params.reward_mode="${GENERALIST_CONTACT_REWARD_MODE}"
  --reward.terms.body_contact_reward_torso.params.threshold="${GENERALIST_CONTACT_REWARD_THRESHOLD}"
  --reward.terms.body_contact_reward_arms.params.threshold="${GENERALIST_CONTACT_REWARD_THRESHOLD}"
  --reward.terms.body_contact_reward_palms.params.threshold="${GENERALIST_CONTACT_REWARD_THRESHOLD}"
  --reward.terms.body_contact_reward_torso.params.force_scale="${GENERALIST_CONTACT_REWARD_FORCE_SCALE}"
  --reward.terms.body_contact_reward_arms.params.force_scale="${GENERALIST_CONTACT_REWARD_FORCE_SCALE}"
  --reward.terms.body_contact_reward_palms.params.force_scale="${GENERALIST_CONTACT_REWARD_FORCE_SCALE}"
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-prepend="${DEFAULT_POSE_PREPEND_ENABLED_FLAG}"
  --command.setup-terms.motion-command.params.motion-config.default-pose-prepend-duration-s="${DEFAULT_POSE_PREPEND_DURATION_S}"
)
if [[ "${DEBUG_MODE}" == "replay" || "${DEBUG_MODE}" == "toy" ]]; then
  train_cmd=("${PYTHON_BIN}" "${train_cmd[@]}")
else
  train_cmd=(torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" "${train_cmd[@]}")
fi
if [[ "${DEBUG_MODE}" == "replay" ]]; then
  train_cmd+=(--training.debug=True)
fi
if [[ "${DEBUG_MODE}" == "toy" ]]; then
  train_cmd+=(--training.toy-mode=True)
  train_cmd+=(--training.viser-env-count="${NUM_ENVS}")
fi
if [[ "${ENABLE_VISER}" == "1" ]]; then
  train_cmd+=(
    --training.enable-viser=True
    --training.viser-port="${VISER_PORT}"
    --training.viser-env-id="${VISER_ENV_ID}"
    --training.viser-update-hz="${VISER_UPDATE_HZ}"
    --training.viser-sync-to-sim="${VISER_SYNC_TO_SIM}"
    --training.viser-force-dt="${VISER_FORCE_DT}"
    --training.viser-recenter="${VISER_RECENTER}"
    --training.viser-show-scandots="${VISER_SHOW_SCANDOTS}"
  )
fi
if [[ -n "${OBJECT_SPEC_PATH}" ]]; then
  train_cmd+=(--robot.object.object-urdf-path "${OBJECT_SPEC_PATH}")
fi
if [[ -n "${SEQUENCE_NAME}" ]]; then
  train_cmd+=(--training.name="${SEQUENCE_NAME}")
fi
if [[ -n "${RESUME_CKPT}" ]]; then
  train_cmd+=(--training.checkpoint="${RESUME_CKPT}")
fi
if [[ "${CURRICULUM}" == "1" || "${CURRICULUM,,}" == "true" ]]; then
  echo "[INFO] Enabling w-object curriculum."
  train_cmd+=(--curriculum.setup-terms.w-object-difficulty-curriculum.params.enabled=True)
fi
train_cmd+=("${EXTRA_ARGS[@]}")
train_cmd+=(logger:wandb)
if [[ -n "${WANDB_ENTITY}" ]]; then
  train_cmd+=(--logger.entity="${WANDB_ENTITY}")
fi
if [[ -n "${WANDB_RUN_ID}" ]]; then
  train_cmd+=(--logger.id="${WANDB_RUN_ID}")
fi
if [[ -n "${WANDB_RESUME}" ]]; then
  train_cmd+=(--logger.resume="${WANDB_RESUME}")
fi
if [[ -n "${SEQUENCE_NAME}" ]]; then
  train_cmd+=(--logger.name="${SEQUENCE_NAME}")
fi
echo "[INFO] Training video recording disabled."
train_cmd+=(--logger.video.enabled=False)
train_cmd+=(--logger.headless_recording=False)
train_cmd+=(--logger.video.upload_to_wandb=False)
VISER_LOAD_URDF="${VISER_LOAD_URDF}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${train_cmd[@]}"
