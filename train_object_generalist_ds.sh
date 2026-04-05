#!/usr/bin/env bash
set -euo pipefail

# DS-box object generalist training.
#
# This launcher prepares a trainable motion bank from:
# - raw motion clips in `data/ds_box_data/train_g1_w_obj`
# - per-clip box geometry in `data/ds_box_data/train_g1_w_obj_geometry`
#
# The prepared bank augments each clip with:
# - `object_size`
# - `object_name`
# - `object_urdf_path`
# - `_clip_object_urdf_map.json`

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

DEFAULT_CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
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
DS_DATA_ROOT=${DS_DATA_ROOT:-"${SCRIPT_DIR}/data/ds_box_data"}
DEFAULT_DS_RAW_MOTION_DIR="${DS_DATA_ROOT}/train_g1_w_obj"
DEFAULT_DS_GEOMETRY_DIR="${DS_DATA_ROOT}/train_g1_w_obj_geometry"
DEFAULT_DS_PREPARED_MOTION_DIR="${DS_DATA_ROOT}/train_g1_w_obj_prepared"
DEFAULT_MIX_NAIVE_MOTION_DIR="${DS_DATA_ROOT}/train_g1_w_obj_prepared_plus_omomo_orig"
DEFAULT_MOTION_DIR="${DEFAULT_DS_PREPARED_MOTION_DIR}"
MOTION_DIR_FROM_ENV=0
if [[ -n "${MOTION_DIR+x}" ]]; then
  MOTION_DIR_FROM_ENV=1
fi
MOTION_DIR=${MOTION_DIR:-"${DEFAULT_MOTION_DIR}"}
RAW_MOTION_DIR=${RAW_MOTION_DIR:-"${DEFAULT_DS_RAW_MOTION_DIR}"}
OBJ_DIR=${OBJ_DIR:-"${DEFAULT_DS_GEOMETRY_DIR}"}
PREPARED_MOTION_DIR=${PREPARED_MOTION_DIR:-""}
OBJECT_SPEC_PATH=${OBJECT_SPEC_PATH:-""}
NUM_ENVS=${NUM_ENVS:-49152}
NPROC=${NPROC:-$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")}
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
PHYSX_GPU_MAX_RIGID_PATCH_COUNT=${PHYSX_GPU_MAX_RIGID_PATCH_COUNT:-4194304}

AUTO_PREP_DS_BANK=${AUTO_PREP_DS_BANK:-1}
DS_PREP_CLEAN_OUT=${DS_PREP_CLEAN_OUT:-1}
DS_OBJECT_MASS=${DS_OBJECT_MASS:-0.1}
DS_OBJECT_COLOR_RGBA=${DS_OBJECT_COLOR_RGBA:-"0.7 0.8 0.9 1"}
PREP_ONLY=${PREP_ONLY:-0}
DATA_MODE=${DATA_MODE:-pure-sd}
STRICT_DEFAULT_DS_BANK_VALIDATION=${STRICT_DEFAULT_DS_BANK_VALIDATION:-1}
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
PURE_SD_REWARD_PROFILE_RAW=${PURE_SD_REWARD_PROFILE:-default}
PURE_SD_REWARD_PROFILE=$(echo "${PURE_SD_REWARD_PROFILE_RAW}" | tr '[:upper:]' '[:lower:]' | tr -d '[][:space:]')
GENERALIST_CONTACT_REWARD_ENABLED=${GENERALIST_CONTACT_REWARD_ENABLED:-1}
GENERALIST_CONTACT_REWARD_MODE=${GENERALIST_CONTACT_REWARD_MODE:-tanh}
GENERALIST_CONTACT_REWARD_THRESHOLD=${GENERALIST_CONTACT_REWARD_THRESHOLD:-1.0}
GENERALIST_CONTACT_REWARD_FORCE_SCALE=${GENERALIST_CONTACT_REWARD_FORCE_SCALE:-25.0}

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

resolve_reward_profile_defaults() {
  local requested_profile="$1"
  local requested_profile_raw="$2"

  case "${requested_profile}" in
    ""|default)
      ACTIVE_REWARD_PROFILE="default"
      DEFAULT_GENERALIST_TORSO_CONTACT_REWARD_WEIGHT=0.30
      DEFAULT_GENERALIST_ARM_CONTACT_REWARD_WEIGHT=0.20
      DEFAULT_GENERALIST_PALM_CONTACT_REWARD_WEIGHT=0.10
      DEFAULT_ROOT_POS_W=0.5
      DEFAULT_ROOT_ORI_W=0.5
      DEFAULT_FULL_BODY_POS_W=1.0
      DEFAULT_FULL_BODY_ORI_W=1.0
      DEFAULT_FULL_BODY_LIN_VEL_W=1.0
      DEFAULT_FULL_BODY_ANG_VEL_W=1.0
      DEFAULT_OBJECT_POS_W=1.0
      DEFAULT_OBJECT_ORI_W=1.0
      DEFAULT_ROOT_POS_SIGMA=0.3
      DEFAULT_ROOT_ORI_SIGMA=0.4
      DEFAULT_FULL_BODY_POS_SIGMA=0.3
      DEFAULT_FULL_BODY_ORI_SIGMA=0.4
      DEFAULT_FULL_BODY_LIN_VEL_SIGMA=1.0
      DEFAULT_FULL_BODY_ANG_VEL_SIGMA=3.14
      DEFAULT_OBJECT_POS_SIGMA=0.3
      DEFAULT_OBJECT_ORI_SIGMA=0.4
      ;;
    loose-cotrack|loose-cotracking|cotrack)
      ACTIVE_REWARD_PROFILE="loose-cotrack"
      DEFAULT_GENERALIST_TORSO_CONTACT_REWARD_WEIGHT=0.45
      DEFAULT_GENERALIST_ARM_CONTACT_REWARD_WEIGHT=0.30
      DEFAULT_GENERALIST_PALM_CONTACT_REWARD_WEIGHT=0.15
      DEFAULT_ROOT_POS_W=0.5
      DEFAULT_ROOT_ORI_W=0.5
      DEFAULT_FULL_BODY_POS_W=1.0
      DEFAULT_FULL_BODY_ORI_W=1.0
      DEFAULT_FULL_BODY_LIN_VEL_W=0.75
      DEFAULT_FULL_BODY_ANG_VEL_W=0.75
      DEFAULT_OBJECT_POS_W=1.5
      DEFAULT_OBJECT_ORI_W=1.25
      DEFAULT_ROOT_POS_SIGMA=0.45
      DEFAULT_ROOT_ORI_SIGMA=0.6
      DEFAULT_FULL_BODY_POS_SIGMA=0.45
      DEFAULT_FULL_BODY_ORI_SIGMA=0.6
      DEFAULT_FULL_BODY_LIN_VEL_SIGMA=1.5
      DEFAULT_FULL_BODY_ANG_VEL_SIGMA=4.5
      DEFAULT_OBJECT_POS_SIGMA=0.45
      DEFAULT_OBJECT_ORI_SIGMA=0.6
      ;;
    *)
      echo "[ERROR] PURE_SD_REWARD_PROFILE must be one of: default, loose-cotrack. Got: ${requested_profile_raw}" >&2
      exit 2
      ;;
  esac
}

refresh_effective_sequence_name() {
  local run_name_suffix=""
  if [[ "${ACTIVE_REWARD_PROFILE:-default}" != "default" ]]; then
    run_name_suffix="-${ACTIVE_REWARD_PROFILE}"
  fi

  if [[ -n "${SEQUENCE_NAME:-}" ]]; then
    EFFECTIVE_SEQUENCE_NAME="${SEQUENCE_NAME}${run_name_suffix}"
  elif [[ -n "${run_name_suffix}" && "${AUTO_ATTACH_WANDB_RUN:-0}" != "1" ]]; then
    EFFECTIVE_SEQUENCE_NAME="${EXP}-${DATA_MODE}${run_name_suffix}"
  else
    EFFECTIVE_SEQUENCE_NAME=""
  fi
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

validate_object_spec_map() {
  local map_path="$1"
  "${PYTHON_BIN}" - "${map_path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser().resolve()
payload = json.loads(path.read_text(encoding="utf-8"))
if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
    payload = payload["clips"]
if not isinstance(payload, dict) or not payload:
    raise SystemExit(f"[ERROR] Invalid or empty object map: {path}")

missing = []
for clip_id, entry in payload.items():
    if isinstance(entry, str):
        urdf = entry.strip()
    elif isinstance(entry, dict):
        urdf = str(entry.get("object_urdf_path", "")).strip()
    else:
        urdf = ""
    if not urdf:
        missing.append((clip_id, "<missing>"))
        continue
    resolved = Path(urdf).expanduser().resolve()
    if not resolved.is_file():
        missing.append((clip_id, str(resolved)))

if missing:
    sample = ", ".join(f"{clip}:{urdf}" for clip, urdf in missing[:10])
    raise SystemExit(f"[ERROR] Object map has missing URDFs in {path}: {sample}")

print(f"[INFO] Validated clip-object URDF map: {path} ({len(payload)} clips)")
PY
}

validate_default_ds_bank() {
  local motion_dir="$1"
  local expected_count="${2:-43}"
  "${PYTHON_BIN}" - "${motion_dir}" "${expected_count}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

motion_dir = Path(sys.argv[1]).expanduser().resolve()
expected = int(sys.argv[2])
npz_files = sorted(motion_dir.glob('*.npz'))
if len(npz_files) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} DS motion clips under {motion_dir}, found {len(npz_files)}")

map_path = motion_dir / '_clip_object_urdf_map.json'
payload = json.loads(map_path.read_text(encoding='utf-8'))
clips = payload['clips'] if isinstance(payload, dict) and isinstance(payload.get('clips'), dict) else payload
if not isinstance(clips, dict):
    raise SystemExit(f"[ERROR] Invalid DS clip-object map payload: {map_path}")
if len(clips) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} map entries in {map_path}, found {len(clips)}")

unique_names = set()
unique_sizes = set()
missing = []
for npz_path in npz_files:
    data = np.load(npz_path, allow_pickle=True)
    object_name = data['object_name'].item() if 'object_name' in data else ''
    object_urdf = data['object_urdf_path'].item() if 'object_urdf_path' in data else ''
    object_size = np.asarray(data['object_size']).reshape(-1).tolist() if 'object_size' in data else None
    if not object_name or not object_urdf or object_size is None or len(object_size) != 3:
        missing.append(npz_path.name)
        continue
    unique_names.add(str(object_name))
    unique_sizes.add(tuple(round(float(v), 6) for v in object_size))

if missing:
    preview = ', '.join(missing[:10])
    raise SystemExit(f"[ERROR] DS prepared bank is missing object fields in: {preview}")
if len(unique_names) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} unique object_name entries, found {len(unique_names)}")
if len(unique_sizes) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} unique object_size entries, found {len(unique_sizes)}")

print(
    f"[INFO] Validated default DS prepared bank: {motion_dir} ({len(npz_files)} clips, {len(unique_sizes)} unique sizes)"
)
PY
}


validate_mix_naive_bank() {
  local motion_dir="$1"
  local expected_total="${2:-105}"
  local expected_ds="${3:-43}"
  local expected_omomo="${4:-62}"
  "${PYTHON_BIN}" - "${motion_dir}" "${expected_total}" "${expected_ds}" "${expected_omomo}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

motion_dir = Path(sys.argv[1]).expanduser().resolve()
expected_total = int(sys.argv[2])
expected_ds = int(sys.argv[3])
expected_omomo = int(sys.argv[4])
npz_files = sorted(motion_dir.glob('*.npz'))
if len(npz_files) != expected_total:
    raise SystemExit(f"[ERROR] Expected {expected_total} mix-naive clips under {motion_dir}, found {len(npz_files)}")

map_path = motion_dir / '_clip_object_urdf_map.json'
payload = json.loads(map_path.read_text(encoding='utf-8'))
clips = payload['clips'] if isinstance(payload, dict) and isinstance(payload.get('clips'), dict) else payload
if not isinstance(clips, dict):
    raise SystemExit(f"[ERROR] Invalid mix-naive clip-object map payload: {map_path}")
if len(clips) != expected_total:
    raise SystemExit(f"[ERROR] Expected {expected_total} map entries in {map_path}, found {len(clips)}")

missing = []
ds_count = 0
omomo_count = 0
unique_names = set()
for npz_path in npz_files:
    data = np.load(npz_path, allow_pickle=True)
    object_name = data['object_name'].item() if 'object_name' in data else ''
    object_urdf = data['object_urdf_path'].item() if 'object_urdf_path' in data else ''
    object_size = np.asarray(data['object_size']).reshape(-1).tolist() if 'object_size' in data else None
    if not object_name or not object_urdf or object_size is None or len(object_size) != 3:
        missing.append(npz_path.name)
        continue
    unique_names.add(str(object_name))
    if npz_path.stem.startswith('sub'):
        omomo_count += 1
    else:
        ds_count += 1

if missing:
    preview = ', '.join(missing[:10])
    raise SystemExit(f"[ERROR] mix-naive bank is missing object fields in: {preview}")
if ds_count != expected_ds or omomo_count != expected_omomo:
    raise SystemExit(
        f"[ERROR] mix-naive bank split mismatch under {motion_dir}: ds={ds_count} omomo={omomo_count} "
        f"(expected ds={expected_ds}, omomo={expected_omomo})"
    )
print(
    f"[INFO] Validated mix-naive bank: {motion_dir} ({len(npz_files)} clips = {ds_count} ds + {omomo_count} omomo, {len(unique_names)} unique object names)"
)
PY
}


prepare_ds_motion_bank() {
  local raw_motion_dir="$1"
  local geometry_dir="$2"
  local out_dir="$3"
  local clean_out="$4"
  local object_mass="$5"
  local color_rgba="$6"

  if [[ ! -d "${raw_motion_dir}" ]]; then
    echo "[ERROR] RAW_MOTION_DIR does not exist: ${raw_motion_dir}" >&2
    exit 2
  fi
  if [[ ! -d "${geometry_dir}" ]]; then
    echo "[ERROR] OBJ_DIR does not exist: ${geometry_dir}" >&2
    exit 2
  fi

  echo "[INFO] Preparing DS motion bank:"
  echo "[INFO]   RAW_MOTION_DIR=${raw_motion_dir}"
  echo "[INFO]   OBJ_DIR=${geometry_dir}"
  echo "[INFO]   OUT_DIR=${out_dir}"

  "${PYTHON_BIN}" - "${raw_motion_dir}" "${geometry_dir}" "${out_dir}" "${clean_out}" "${object_mass}" "${color_rgba}" <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

raw_motion_dir = Path(sys.argv[1]).expanduser().resolve()
geometry_dir = Path(sys.argv[2]).expanduser().resolve()
out_dir = Path(sys.argv[3]).expanduser().resolve()
clean_out = sys.argv[4].strip().lower() not in {"", "0", "false", "no", "off"}
object_mass = float(sys.argv[5])
color_rgba = str(sys.argv[6]).strip()

if out_dir == raw_motion_dir:
    raise SystemExit(f"[ERROR] Refusing to prepare in-place over RAW_MOTION_DIR: {out_dir}")
if out_dir == geometry_dir:
    raise SystemExit(f"[ERROR] Refusing to prepare in-place over OBJ_DIR: {out_dir}")

motion_files = sorted(raw_motion_dir.glob('*.npz'))
if not motion_files:
    raise SystemExit(f"[ERROR] No .npz clips found in {raw_motion_dir}")

generated_urdf_dir = out_dir / '_generated_urdfs'
if clean_out and out_dir.exists():
    for child in out_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
out_dir.mkdir(parents=True, exist_ok=True)
generated_urdf_dir.mkdir(parents=True, exist_ok=True)


def ensure_removed(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def symlink_or_copy(src: Path, dst: Path) -> None:
    ensure_removed(dst)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def parse_obj_extents(obj_path: Path) -> np.ndarray:
    mins = None
    maxs = None
    with obj_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.startswith('v '):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            xyz = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            if mins is None:
                mins = xyz.copy()
                maxs = xyz.copy()
            else:
                mins = np.minimum(mins, xyz)
                maxs = np.maximum(maxs, xyz)
    if mins is None or maxs is None:
        raise ValueError(f'No OBJ vertices found in {obj_path}')
    return np.maximum(maxs - mins, 1.0e-4).astype(np.float32)


def build_urdf_text(robot_name: str, mesh_filename: str, mass: float, color: str, extents: np.ndarray) -> str:
    x, y, z = [float(v) for v in extents]
    ixx = mass * (y * y + z * z) / 12.0
    iyy = mass * (x * x + z * z) / 12.0
    izz = mass * (x * x + y * y) / 12.0
    return f"""<?xml version="1.0" ?>
<robot name="{robot_name}">
  <dynamics damping="0.5" friction="0.9"/>
  <link name="baseLink">
    <inertial>
      <mass value="{mass:.8g}"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="{ixx:.8g}" ixy="0" ixz="0" iyy="{iyy:.8g}" iyz="0" izz="{izz:.8g}"/>
    </inertial>
    <contact>
      <lateral_friction value="0.9"/>
      <rolling_friction value="0.5"/>
      <stiffness value="30000"/>
      <damping value="1000"/>
    </contact>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_filename}" scale="1.0 1.0 1.0"/>
      </geometry>
      <material name="mat">
        <color rgba="{color}"/>
      </material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_filename}" scale="1.0 1.0 1.0"/>
      </geometry>
    </collision>
  </link>
</robot>
"""


clip_map = {}
missing_geometry = []
for src_npz in motion_files:
    clip_id = src_npz.stem
    obj_src = geometry_dir / f'{clip_id}.obj'
    if not obj_src.is_file():
        missing_geometry.append(clip_id)
        continue

    object_size = parse_obj_extents(obj_src)
    obj_dst = generated_urdf_dir / obj_src.name
    symlink_or_copy(obj_src, obj_dst)

    urdf_path = generated_urdf_dir / f'{clip_id}.urdf'
    urdf_path.write_text(
        build_urdf_text(clip_id, obj_dst.name, object_mass, color_rgba, object_size),
        encoding='utf-8',
    )

    with np.load(src_npz, allow_pickle=True) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    payload['object_size'] = object_size.astype(np.float32)
    payload['object_name'] = np.array(clip_id)
    payload['object_urdf_path'] = np.array(str(urdf_path.resolve()))
    np.savez_compressed(out_dir / src_npz.name, **payload)

    clip_map[clip_id] = {
        'object_name': clip_id,
        'object_urdf_path': str(urdf_path.resolve()),
        'object_size': [float(v) for v in object_size.tolist()],
    }

if missing_geometry:
    sample = ', '.join(missing_geometry[:10])
    raise SystemExit(f'[ERROR] Missing OBJ geometry for {len(missing_geometry)} clips: {sample}')

map_path = out_dir / '_clip_object_urdf_map.json'
map_path.write_text(json.dumps({'clips': clip_map}, indent=2, sort_keys=True), encoding='utf-8')

print(f'[INFO] Prepared DS motion bank with {len(clip_map)} clips at {out_dir}')
print(f'[INFO] Wrote clip-object map: {map_path}')
PY
}

if [[ "$#" -gt 0 ]]; then
  first_arg_normalized=$(echo "$1" | tr '[:upper:]' '[:lower:]')
  case "${first_arg_normalized}" in
    pure-sd|pure-ds)
      DATA_MODE="pure-sd"
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
  esac
fi
DATA_MODE=$(echo "${DATA_MODE}" | tr '[:upper:]' '[:lower:]')
case "${DATA_MODE}" in
  pure-ds)
    DATA_MODE="pure-sd"
    ;;
  mix-clean-noisy|mix-curr)
    DATA_MODE="mix-curriculum"
    ;;
esac
case "${DATA_MODE}" in
  pure-sd|mix-naive|mix-curriculum)
    ;;
  *)
    echo "[ERROR] Unsupported DATA_MODE='${DATA_MODE}'. Use one of: pure-sd, mix-naive, mix-curriculum" >&2
    exit 2
    ;;
esac
if [[ "${DATA_MODE}" == "mix-naive" && -n "${MIX_NAIVE_CLEAN_NOISY_CURRICULUM+x}" ]]; then
  legacy_mix_curriculum_normalized=$(echo "${MIX_NAIVE_CLEAN_NOISY_CURRICULUM}" | tr '[:upper:]' '[:lower:]')
  case "${legacy_mix_curriculum_normalized}" in
    1|true|yes|on)
      echo "[ERROR] MIX_NAIVE_CLEAN_NOISY_CURRICULUM is no longer supported with DATA_MODE=mix-naive." >&2
      echo "[ERROR] Use the third mode directly instead: bash train_object_generalist_ds.sh mix-curriculum" >&2
      exit 2
      ;;
  esac
fi
if [[ "${DATA_MODE}" == "pure-sd" ]]; then
  resolve_reward_profile_defaults "${PURE_SD_REWARD_PROFILE}" "${PURE_SD_REWARD_PROFILE_RAW}"
else
  resolve_reward_profile_defaults "default" "default"
  if [[ -n "${PURE_SD_REWARD_PROFILE}" && "${PURE_SD_REWARD_PROFILE}" != "default" ]]; then
    echo "[WARN] Ignoring PURE_SD_REWARD_PROFILE=${PURE_SD_REWARD_PROFILE_RAW} for DATA_MODE=${DATA_MODE}; reward profile presets only apply to pure-sd."
  fi
fi
GENERALIST_TORSO_CONTACT_REWARD_WEIGHT=${GENERALIST_TORSO_CONTACT_REWARD_WEIGHT:-${DEFAULT_GENERALIST_TORSO_CONTACT_REWARD_WEIGHT}}
GENERALIST_ARM_CONTACT_REWARD_WEIGHT=${GENERALIST_ARM_CONTACT_REWARD_WEIGHT:-${DEFAULT_GENERALIST_ARM_CONTACT_REWARD_WEIGHT}}
GENERALIST_PALM_CONTACT_REWARD_WEIGHT=${GENERALIST_PALM_CONTACT_REWARD_WEIGHT:-${DEFAULT_GENERALIST_PALM_CONTACT_REWARD_WEIGHT}}
ROOT_POS_W=${ROOT_POS_W:-${DEFAULT_ROOT_POS_W}}
ROOT_ORI_W=${ROOT_ORI_W:-${DEFAULT_ROOT_ORI_W}}
FULL_BODY_POS_W=${FULL_BODY_POS_W:-${DEFAULT_FULL_BODY_POS_W}}
FULL_BODY_ORI_W=${FULL_BODY_ORI_W:-${DEFAULT_FULL_BODY_ORI_W}}
FULL_BODY_LIN_VEL_W=${FULL_BODY_LIN_VEL_W:-${DEFAULT_FULL_BODY_LIN_VEL_W}}
FULL_BODY_ANG_VEL_W=${FULL_BODY_ANG_VEL_W:-${DEFAULT_FULL_BODY_ANG_VEL_W}}
OBJECT_POS_W=${OBJECT_POS_W:-${DEFAULT_OBJECT_POS_W}}
OBJECT_ORI_W=${OBJECT_ORI_W:-${DEFAULT_OBJECT_ORI_W}}
ROOT_POS_SIGMA=${ROOT_POS_SIGMA:-${DEFAULT_ROOT_POS_SIGMA}}
ROOT_ORI_SIGMA=${ROOT_ORI_SIGMA:-${DEFAULT_ROOT_ORI_SIGMA}}
FULL_BODY_POS_SIGMA=${FULL_BODY_POS_SIGMA:-${DEFAULT_FULL_BODY_POS_SIGMA}}
FULL_BODY_ORI_SIGMA=${FULL_BODY_ORI_SIGMA:-${DEFAULT_FULL_BODY_ORI_SIGMA}}
FULL_BODY_LIN_VEL_SIGMA=${FULL_BODY_LIN_VEL_SIGMA:-${DEFAULT_FULL_BODY_LIN_VEL_SIGMA}}
FULL_BODY_ANG_VEL_SIGMA=${FULL_BODY_ANG_VEL_SIGMA:-${DEFAULT_FULL_BODY_ANG_VEL_SIGMA}}
OBJECT_POS_SIGMA=${OBJECT_POS_SIGMA:-${DEFAULT_OBJECT_POS_SIGMA}}
OBJECT_ORI_SIGMA=${OBJECT_ORI_SIGMA:-${DEFAULT_OBJECT_ORI_SIGMA}}
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
else
  echo "[INFO] No resume checkpoint requested; training will start from scratch."
  if [[ -n "${WANDB_RUN_ID}" || -n "${WANDB_RESUME}" ]]; then
    echo "[WARN] WANDB_RUN_ID/WANDB_RESUME is set without RESUME_CKPT. Training still starts without a model checkpoint, but W&B may attach to an existing run."
  fi
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

refresh_effective_sequence_name
echo "[INFO] Data mode: ${DATA_MODE}"
if [[ -n "${SEQUENCE_NAME}" ]]; then
  echo "[INFO] Sequence name: ${SEQUENCE_NAME}"
fi
if [[ -n "${EFFECTIVE_SEQUENCE_NAME}" ]]; then
  echo "[INFO] Effective run name: ${EFFECTIVE_SEQUENCE_NAME}"
fi
if [[ "${DATA_MODE}" == "mix-curriculum" ]]; then
  MIX_NAIVE_CLEAN_NOISY_CURRICULUM_FLAG=1
  echo "[INFO] mix-curriculum enabled: clean=sub* noisy=non-sub* schedule=100/0 -> 90/10@1500 -> 80/20@2000 -> 70/30@2500 -> 60/40@3000 -> 50/50@4000+"
fi

case "${DATA_MODE}" in
  pure-sd)
    MODE_DEFAULT_MOTION_DIR="${DEFAULT_DS_PREPARED_MOTION_DIR}"
    ;;
  mix-naive|mix-curriculum)
    MODE_DEFAULT_MOTION_DIR="${DEFAULT_MIX_NAIVE_MOTION_DIR}"
    ;;
esac

if [[ -z "${PREPARED_MOTION_DIR}" ]]; then
  if [[ "${MOTION_DIR_FROM_ENV}" == "1" ]]; then
    if [[ -d "${MOTION_DIR}" && ! -f "${MOTION_DIR}/_clip_object_urdf_map.json" ]]; then
      PREPARED_MOTION_DIR="${MOTION_DIR%/}_prepared"
    else
      PREPARED_MOTION_DIR="${MOTION_DIR}"
    fi
  else
    PREPARED_MOTION_DIR="${MODE_DEFAULT_MOTION_DIR}"
  fi
fi

if [[ "${DATA_MODE}" == "pure-sd" && "${MOTION_DIR_FROM_ENV}" == "1" && -d "${MOTION_DIR}" && ! -f "${MOTION_DIR}/_clip_object_urdf_map.json" ]]; then
  RAW_MOTION_DIR="${MOTION_DIR}"
  echo "[INFO] MOTION_DIR points to a raw DS bank; using it as RAW_MOTION_DIR source: ${RAW_MOTION_DIR}"
fi

if [[ "${DATA_MODE}" == "pure-sd" ]]; then
  if [[ "${AUTO_PREP_DS_BANK}" != "0" ]]; then
    prepare_ds_motion_bank "${RAW_MOTION_DIR}" "${OBJ_DIR}" "${PREPARED_MOTION_DIR}" "${DS_PREP_CLEAN_OUT}" "${DS_OBJECT_MASS}" "${DS_OBJECT_COLOR_RGBA}"
    MOTION_DIR="${PREPARED_MOTION_DIR}"
  elif [[ "${MOTION_DIR_FROM_ENV}" != "1" ]]; then
    MOTION_DIR="${PREPARED_MOTION_DIR}"
  fi
else
  if [[ "${MOTION_DIR_FROM_ENV}" != "1" ]]; then
    MOTION_DIR="${MODE_DEFAULT_MOTION_DIR}"
  fi
fi

echo "[INFO] RAW_MOTION_DIR: ${RAW_MOTION_DIR}"
echo "[INFO] OBJ_DIR: ${OBJ_DIR}"
echo "[INFO] MOTION_DIR: ${MOTION_DIR}"

if [[ -z "${OBJECT_SPEC_PATH}" ]]; then
  default_map="${MOTION_DIR}/_clip_object_urdf_map.json"
  if [[ -f "${default_map}" ]]; then
    OBJECT_SPEC_PATH="${default_map}"
    echo "[INFO] Using clip-object URDF map: ${OBJECT_SPEC_PATH}"
  fi
fi

if [[ -z "${OBJECT_SPEC_PATH}" || ! -f "${OBJECT_SPEC_PATH}" ]]; then
  echo "[ERROR] DS object generalist training requires a valid _clip_object_urdf_map.json." >&2
  echo "[ERROR] Current MOTION_DIR: ${MOTION_DIR}" >&2
  echo "[ERROR] Either enable AUTO_PREP_DS_BANK=1 or point OBJECT_SPEC_PATH to a valid map." >&2
  exit 2
fi
validate_object_spec_map "${OBJECT_SPEC_PATH}"

if [[ "${STRICT_DEFAULT_DS_BANK_VALIDATION}" != "0" ]]; then
  case "${DATA_MODE}" in
    pure-sd)
      if [[ "$(realpath "${MOTION_DIR}")" == "$(realpath "${DEFAULT_DS_PREPARED_MOTION_DIR}")" ]]; then
        validate_default_ds_bank "${MOTION_DIR}" 43
      fi
      ;;
    mix-naive|mix-curriculum)
      if [[ "$(realpath "${MOTION_DIR}")" == "$(realpath "${DEFAULT_MIX_NAIVE_MOTION_DIR}")" ]]; then
        validate_mix_naive_bank "${MOTION_DIR}" 105 43 62
      fi
      ;;
  esac
fi

prep_only_normalized=$(echo "${PREP_ONLY}" | tr '[:upper:]' '[:lower:]')
case "${prep_only_normalized}" in
  1|true|yes|on)
    echo "[INFO] PREP_ONLY enabled; skipping training launch."
    exit 0
    ;;
  0|false|no|off|"")
    ;;
  *)
    echo "[ERROR] PREP_ONLY must be one of: 0/1/true/false/yes/no/on/off. Got: ${PREP_ONLY}" >&2
    exit 2
    ;;
esac

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
    DEBUG_URDF_COUNT=$("${PYTHON_BIN}" - <<'PY' "${OBJECT_SPEC_PATH}"
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
echo "[INFO] pure_sd_reward_profile=${ACTIVE_REWARD_PROFILE}"
echo "[INFO] Generalist contact reward mode=${GENERALIST_CONTACT_REWARD_MODE} threshold=${GENERALIST_CONTACT_REWARD_THRESHOLD} force_scale=${GENERALIST_CONTACT_REWARD_FORCE_SCALE}"
echo "[INFO] Generalist contact reward weights torso=${GENERALIST_TORSO_CONTACT_REWARD_WEIGHT} arms=${GENERALIST_ARM_CONTACT_REWARD_WEIGHT} palms=${GENERALIST_PALM_CONTACT_REWARD_WEIGHT}"
echo "[INFO] Reference tracking reward weights root_pos=${ROOT_POS_W} root_ori=${ROOT_ORI_W} body_pos=${FULL_BODY_POS_W} body_ori=${FULL_BODY_ORI_W} body_lin_vel=${FULL_BODY_LIN_VEL_W} body_ang_vel=${FULL_BODY_ANG_VEL_W}"
echo "[INFO] Box tracking reward weights object_pos=${OBJECT_POS_W} object_ori=${OBJECT_ORI_W}"
echo "[INFO] Reference tracking reward sigmas root_pos=${ROOT_POS_SIGMA} root_ori=${ROOT_ORI_SIGMA} body_pos=${FULL_BODY_POS_SIGMA} body_ori=${FULL_BODY_ORI_SIGMA} body_lin_vel=${FULL_BODY_LIN_VEL_SIGMA} body_ang_vel=${FULL_BODY_ANG_VEL_SIGMA}"
echo "[INFO] Box tracking reward sigmas object_pos=${OBJECT_POS_SIGMA} object_ori=${OBJECT_ORI_SIGMA}"
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
  --reward.terms.motion_global_ref_position_error_exp.weight="${ROOT_POS_W}"
  --reward.terms.motion_global_ref_orientation_error_exp.weight="${ROOT_ORI_W}"
  --reward.terms.motion_relative_body_position_error_exp.weight="${FULL_BODY_POS_W}"
  --reward.terms.motion_relative_body_orientation_error_exp.weight="${FULL_BODY_ORI_W}"
  --reward.terms.motion_global_body_lin_vel.weight="${FULL_BODY_LIN_VEL_W}"
  --reward.terms.motion_global_body_ang_vel.weight="${FULL_BODY_ANG_VEL_W}"
  --reward.terms.object_global_ref_position_error_exp.weight="${OBJECT_POS_W}"
  --reward.terms.object_global_ref_orientation_error_exp.weight="${OBJECT_ORI_W}"
  --reward.terms.motion_global_ref_position_error_exp.params.sigma="${ROOT_POS_SIGMA}"
  --reward.terms.motion_global_ref_orientation_error_exp.params.sigma="${ROOT_ORI_SIGMA}"
  --reward.terms.motion_relative_body_position_error_exp.params.sigma="${FULL_BODY_POS_SIGMA}"
  --reward.terms.motion_relative_body_orientation_error_exp.params.sigma="${FULL_BODY_ORI_SIGMA}"
  --reward.terms.motion_global_body_lin_vel.params.sigma="${FULL_BODY_LIN_VEL_SIGMA}"
  --reward.terms.motion_global_body_ang_vel.params.sigma="${FULL_BODY_ANG_VEL_SIGMA}"
  --reward.terms.object_global_ref_position_error_exp.params.sigma="${OBJECT_POS_SIGMA}"
  --reward.terms.object_global_ref_orientation_error_exp.params.sigma="${OBJECT_ORI_SIGMA}"
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
if [[ -n "${EFFECTIVE_SEQUENCE_NAME}" ]]; then
  train_cmd+=(--training.name="${EFFECTIVE_SEQUENCE_NAME}")
fi
if [[ -n "${RESUME_CKPT}" ]]; then
  train_cmd+=(--training.checkpoint="${RESUME_CKPT}")
fi
if [[ "${CURRICULUM}" == "1" || "${CURRICULUM,,}" == "true" ]]; then
  echo "[INFO] Enabling w-object curriculum."
  train_cmd+=(--curriculum.setup-terms.w-object-difficulty-curriculum.params.enabled=True)
fi
if [[ "${DATA_MODE}" == "mix-curriculum" ]]; then
  train_cmd+=(--command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.enabled=True)
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
if [[ -n "${EFFECTIVE_SEQUENCE_NAME}" ]]; then
  train_cmd+=(--logger.name="${EFFECTIVE_SEQUENCE_NAME}")
fi
echo "[INFO] Training video recording disabled."
train_cmd+=(--logger.video.enabled=False)
train_cmd+=(--logger.headless_recording=False)
train_cmd+=(--logger.video.upload_to_wandb=False)
VISER_LOAD_URDF="${VISER_LOAD_URDF}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${train_cmd[@]}"
