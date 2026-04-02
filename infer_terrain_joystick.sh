#!/usr/bin/env bash
set -euo pipefail

# Only edit this line. Then run: bash infer_terrain_joystick.sh
WANDB_RUN_URL="${WANDB_RUN_URL:-}"

# IsaacSim + Viser manual-control inference for terrain sparse-root distill students.
# This is the perception-enabled student side of the terrain pipeline.
#
# Usage:
#   bash infer_terrain_joystick.sh [checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra tyro args...]

usage() {
  cat <<'EOF'
Usage:
  bash infer_terrain_joystick.sh [checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra tyro args...]
  # or edit WANDB_RUN_URL at the top, then just run:
  bash infer_terrain_joystick.sh

Optional env vars:
  CHECKPOINT / CKPT
  WANDB_MODEL_FILE
  MOTION_DIR
  MOTION_CLIP_NAME
  MOTION_CLIP_ID
  GEOMETRY_DIR
  GEOMETRY_METADATA
  PERCEPTION_PRESET          (default: checkpoint; options: checkpoint|none|heightmap|camera_depth_d435i|camera_depth_d435i_17x17)
  PAIR_TERRAIN_WITH_MOTION   (default: checkpoint value, else True)
  NUM_ENVS                   (default: 1)
  HEADLESS                   (default: True)
  VISER_PORT                 (default: random)
  VISER_ENV_ID               (default: 0)
  VISER_UPDATE_HZ            (default: 30)
  VISER_RECENTER             (default: True)
  VISER_SYNC_TO_SIM          (default: True)
  VISER_FORCE_DT             (default: True)
  VISER_SHOW_SCANDOTS        (default: False)
  VIS_GPU                    (default: auto)
  DISABLE_RANDOMIZATION      (default: True)
  START_AT_TIMESTEP_ZERO_PROB
  FREEZE_AT_TIMESTEP_ZERO_PROB
  RESET_NOISE_SCALE
  NUM_ROWS / TERRAIN_NUM_ROWS
  NUM_COLS / TERRAIN_NUM_COLS
  MAX_EPISODE_LENGTH_S       (default: 1000000)
  DRY_RUN                    (default: 0)
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_CHECKPOINT="${DEFAULT_CHECKPOINT:-${WANDB_RUN_URL:-}}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

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
from pathlib import Path

repo_root = Path.cwd().resolve()
sanitized_sys_path = []
for entry in sys.path:
    if entry in {"", "."}:
        continue
    try:
        if Path(entry).resolve() == repo_root:
            continue
    except Exception:
        pass
    sanitized_sys_path.append(entry)
sys.path = sanitized_sys_path

try:
    import wandb
except Exception:
    sys.exit(0)

entity, project, run_id = sys.argv[1:4]
api = wandb.Api(timeout=30)
run = api.run(f"{entity}/{project}/{run_id}")
pattern = re.compile(r"^model_(\d+)\.pt$")
latest_step = -1
latest_name = ""
for file_obj in run.files():
    name = getattr(file_obj, "name", "")
    match = pattern.match(name)
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

resolve_local_checkpoint_from_run_url() {
  local ref="$1"
  local preferred_model_file="${2:-}"
  local parsed=""
  local run_id=""
  local explicit_file=""
  local target_model_file=""
  local latest_ckpt=""
  parsed="$(parse_wandb_run_url "${ref}" || true)"
  if [[ -z "${parsed}" ]]; then
    echo ""
    return 0
  fi
  IFS=$'\t' read -r _entity _project run_id explicit_file <<< "${parsed}"
  target_model_file="${explicit_file:-${preferred_model_file}}"
  while IFS= read -r run_dir; do
    [[ -z "${run_dir}" ]] && continue
    if [[ -n "${target_model_file}" && -f "${run_dir}/${target_model_file}" ]]; then
      echo "${run_dir}/${target_model_file}"
      return 0
    fi
    latest_ckpt="$(find "${run_dir}" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n1 || true)"
    if [[ -n "${latest_ckpt}" ]]; then
      echo "${latest_ckpt}"
      return 0
    fi
  done < <(find /data/logs_new -maxdepth 2 -type d -name "*${run_id}*" 2>/dev/null | sort)
  echo ""
}

find_latest_local_distill_ckpt() {
  local latest_ckpt=""
  local latest_mtime=0
  local ckpt=""
  local mtime=0
  local root=""
  IFS=',' read -r -a roots <<< "${DISTILL_LOG_ROOTS:-/data/logs_new/terrain-aware,/data/logs_new/boxer}"
  for root in "${roots[@]}"; do
    [[ -d "${root}" ]] || continue
    while IFS= read -r dir; do
      ckpt="$(find "${dir}" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n1 || true)"
      [[ -n "${ckpt}" ]] || continue
      mtime="$(stat -c %Y "${ckpt}" 2>/dev/null || echo 0)"
      if [[ "${mtime}" -gt "${latest_mtime}" ]]; then
        latest_mtime="${mtime}"
        latest_ckpt="${ckpt}"
      fi
    done < <(find "${root}" -maxdepth 1 -type d \( -iname '*terrain*distill*' -o -iname '*distill*terrain*' \) 2>/dev/null)
  done
  echo "${latest_ckpt}"
}

resolve_data_path() {
  local path_value="$1"
  if [[ -z "${path_value}" ]]; then
    echo ""
    return 0
  fi
  if [[ "${path_value}" == s3://* || "${path_value}" == /* ]]; then
    echo "${path_value}"
    return 0
  fi
  if [[ "${path_value}" == @holosoma/* ]]; then
    echo "${SCRIPT_DIR}/src/holosoma/${path_value#@holosoma/}"
    return 0
  fi
  if [[ "${path_value}" == holosoma/data/* ]]; then
    echo "${SCRIPT_DIR}/src/holosoma/${path_value}"
    return 0
  fi
  "${PYTHON_BIN}" - "${path_value}" <<'PY'
import sys
from pathlib import Path
print(str(Path(sys.argv[1]).expanduser().resolve()))
PY
}

extract_checkpoint_terrain_defaults() {
  local checkpoint_ref="$1"
  "${PYTHON_BIN}" - "${checkpoint_ref}" <<'PY' 2>/dev/null || true
import sys
import tempfile
from pathlib import Path
import torch

def parse_ref(reference: str) -> tuple[str, str]:
    remainder = reference[len("wandb://"):]
    parts = remainder.split("/")
    entity, project = parts[0], parts[1]
    run_id_index = 2
    if len(parts) > 4 and parts[2] == "runs":
        run_id_index = 3
    run_id = parts[run_id_index]
    ckpt_name = "/".join(parts[run_id_index + 1 :]).strip()
    return f"{entity}/{project}/{run_id}", ckpt_name

def load_payload(reference: str):
    if reference.startswith("wandb://"):
        import wandb
        run_path, ckpt_name = parse_ref(reference)
        run = wandb.Api(timeout=30).run(run_path)
        with tempfile.TemporaryDirectory() as td:
            downloaded = run.file(ckpt_name).download(root=td, replace=True)
            path = Path(downloaded.name)
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            return torch.load(path, map_location="cpu")
    return torch.load(reference, map_location="cpu")

payload = load_payload(sys.argv[1])
cfg = payload.get("experiment_config")
if not isinstance(cfg, dict):
    sys.exit(0)
command_cfg = cfg.get("command")
setup_terms = command_cfg.get("setup_terms", {}) if isinstance(command_cfg, dict) else {}
motion_command = setup_terms.get("motion_command", {}) if isinstance(setup_terms, dict) else {}
params = motion_command.get("params", {}) if isinstance(motion_command, dict) else {}
motion_cfg = params.get("motion_config", {}) if isinstance(params, dict) else {}
terrain_cfg = cfg.get("terrain")
terrain_term = terrain_cfg.get("terrain_term", {}) if isinstance(terrain_cfg, dict) else {}
perception_cfg = cfg.get("perception")
noise_cfg = motion_cfg.get("noise_to_initial_pose", {}) if isinstance(motion_cfg, dict) else {}
values = [
    motion_cfg.get("motion_file") if isinstance(motion_cfg, dict) else None,
    motion_cfg.get("motion_clip_name") if isinstance(motion_cfg, dict) else None,
    motion_cfg.get("motion_clip_id") if isinstance(motion_cfg, dict) else None,
    motion_cfg.get("pair_terrain_with_motion") if isinstance(motion_cfg, dict) else None,
    motion_cfg.get("start_at_timestep_zero_prob") if isinstance(motion_cfg, dict) else None,
    motion_cfg.get("freeze_at_timestep_zero_prob") if isinstance(motion_cfg, dict) else None,
    noise_cfg.get("overall_noise_scale") if isinstance(noise_cfg, dict) else None,
    terrain_term.get("obj_file_path") if isinstance(terrain_term, dict) else None,
    terrain_term.get("obj_metadata_path") if isinstance(terrain_term, dict) else None,
    terrain_term.get("num_rows") if isinstance(terrain_term, dict) else None,
    terrain_term.get("num_cols") if isinstance(terrain_term, dict) else None,
    perception_cfg.get("enabled") if isinstance(perception_cfg, dict) else None,
    perception_cfg.get("output_mode") if isinstance(perception_cfg, dict) else None,
]
for value in values:
    print("" if value is None else str(value))
PY
}

normalize_bool_flag() {
  local name="$1"
  local raw="${2:-}"
  case "$(echo "${raw}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) echo "True" ;;
    0|false|no|off|"") echo "False" ;;
    *)
      echo "[ERROR] ${name} must be one of: 0/1/true/false/yes/no/on/off. Got: ${raw}" >&2
      exit 2
      ;;
  esac
}

normalize_bool_int() {
  local name="$1"
  local raw="${2:-}"
  case "$(echo "${raw}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) echo "1" ;;
    0|false|no|off|"") echo "0" ;;
    *)
      echo "[ERROR] ${name} must be one of: 0/1/true/false/yes/no/on/off. Got: ${raw}" >&2
      exit 2
      ;;
  esac
}

empty_if_none() {
  local value="${1:-}"
  if [[ "${value}" == "None" || "${value}" == "null" ]]; then
    echo ""
  else
    echo "${value}"
  fi
}

looks_like_checkpoint_ref() {
  local value="$1"
  [[ "${value}" == wandb://* || "${value}" == https://wandb.ai/*/runs/* || "${value}" == /* || "${value}" == ./* || "${value}" == ../* || "${value}" == *.pt ]]
}

run_with_delayed_local_url() {
  local port="$1"
  shift
  local fifo_path=""
  local child_pid=""
  local child_status=0
  local announced=0
  local viewer_disabled=0
  fifo_path="$(mktemp -u "${TMPDIR:-/tmp}/infer_terrain_joystick.XXXXXX.fifo")"
  mkfifo "${fifo_path}"
  "$@" >"${fifo_path}" 2>&1 &
  child_pid=$!
  while IFS= read -r line || [[ -n "${line}" ]]; do
    printf '%s\n' "${line}"
    case "${line}" in
      *"Viser live viewer disabled:"*)
        viewer_disabled=1
        ;;
    esac
    if [[ "${announced}" == "0" ]]; then
      case "${line}" in
        *"Registering key:"*)
          if [[ "${viewer_disabled}" != "1" ]]; then
            echo "[INFO] Local URL: http://localhost:${port}"
            announced=1
          fi
          ;;
      esac
    fi
  done < "${fifo_path}"
  wait "${child_pid}" || child_status=$?
  rm -f "${fifo_path}"
  if [[ "${announced}" == "0" && "${viewer_disabled}" == "1" ]]; then
    echo "[WARN] Viser viewer was disabled during startup; no localhost viewer will be available."
  elif [[ "${announced}" == "0" && "${child_status}" == "0" ]]; then
    echo "[INFO] Local URL: http://localhost:${port}"
  fi
  return "${child_status}"
}

CHECKPOINT="${CHECKPOINT:-${CKPT:-${DEFAULT_CHECKPOINT:-}}}"
if [[ $# -gt 0 ]] && looks_like_checkpoint_ref "$1"; then
  CHECKPOINT="$1"
  shift
fi
if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT="$(find_latest_local_distill_ckpt)"
fi
if [[ -z "${CHECKPOINT}" ]]; then
  echo "[ERROR] Could not auto-resolve terrain distill checkpoint. Pass one explicitly." >&2
  exit 1
fi

if [[ "${CHECKPOINT}" == https://wandb.ai/*/runs/* ]]; then
  LOCAL_WANDB_CKPT="$(resolve_local_checkpoint_from_run_url "${CHECKPOINT}" "${WANDB_MODEL_FILE:-}")"
  if [[ -n "${LOCAL_WANDB_CKPT}" && -f "${LOCAL_WANDB_CKPT}" ]]; then
    CHECKPOINT="${LOCAL_WANDB_CKPT}"
    echo "[INFO] Resolved wandb run URL to local checkpoint: ${CHECKPOINT}"
  else
    CHECKPOINT="$(normalize_checkpoint_ref "${CHECKPOINT}")"
  fi
fi
if [[ "${CHECKPOINT}" != wandb://* ]] && [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

MOTION_DIR_SET=0
if [[ -n "${MOTION_DIR+x}" && -n "${MOTION_DIR}" ]]; then
  MOTION_DIR_SET=1
fi
GEOMETRY_DIR_SET=0
if [[ -n "${GEOMETRY_DIR+x}" && -n "${GEOMETRY_DIR}" ]]; then
  GEOMETRY_DIR_SET=1
fi
if [[ -n "${GEOMETRY_METADATA+x}" ]]; then
  GEOMETRY_METADATA_SET=1
else
  GEOMETRY_METADATA_SET=0
fi
MOTION_CLIP_NAME_FROM_ENV=0
if [[ -n "${MOTION_CLIP_NAME+x}" ]]; then
  MOTION_CLIP_NAME_FROM_ENV=1
fi
MOTION_CLIP_ID_FROM_ENV=0
if [[ -n "${MOTION_CLIP_ID+x}" ]]; then
  MOTION_CLIP_ID_FROM_ENV=1
fi

CHECKPOINT_MOTION_DIR=""
CHECKPOINT_MOTION_CLIP_NAME=""
CHECKPOINT_MOTION_CLIP_ID=""
CHECKPOINT_PAIR_TERRAIN_WITH_MOTION=""
CHECKPOINT_START_AT_TIMESTEP_ZERO_PROB=""
CHECKPOINT_FREEZE_AT_TIMESTEP_ZERO_PROB=""
CHECKPOINT_RESET_NOISE_SCALE=""
CHECKPOINT_GEOMETRY_DIR=""
CHECKPOINT_GEOMETRY_METADATA=""
CHECKPOINT_NUM_ROWS=""
CHECKPOINT_NUM_COLS=""
CHECKPOINT_PERCEPTION_ENABLED=""
CHECKPOINT_PERCEPTION_OUTPUT_MODE=""

mapfile -t checkpoint_defaults_lines < <(extract_checkpoint_terrain_defaults "${CHECKPOINT}")
CHECKPOINT_MOTION_DIR="$(empty_if_none "${checkpoint_defaults_lines[0]:-}")"
CHECKPOINT_MOTION_CLIP_NAME="$(empty_if_none "${checkpoint_defaults_lines[1]:-}")"
CHECKPOINT_MOTION_CLIP_ID="$(empty_if_none "${checkpoint_defaults_lines[2]:-}")"
CHECKPOINT_PAIR_TERRAIN_WITH_MOTION="$(empty_if_none "${checkpoint_defaults_lines[3]:-}")"
CHECKPOINT_START_AT_TIMESTEP_ZERO_PROB="$(empty_if_none "${checkpoint_defaults_lines[4]:-}")"
CHECKPOINT_FREEZE_AT_TIMESTEP_ZERO_PROB="$(empty_if_none "${checkpoint_defaults_lines[5]:-}")"
CHECKPOINT_RESET_NOISE_SCALE="$(empty_if_none "${checkpoint_defaults_lines[6]:-}")"
CHECKPOINT_GEOMETRY_DIR="$(empty_if_none "${checkpoint_defaults_lines[7]:-}")"
CHECKPOINT_GEOMETRY_METADATA="$(empty_if_none "${checkpoint_defaults_lines[8]:-}")"
CHECKPOINT_NUM_ROWS="$(empty_if_none "${checkpoint_defaults_lines[9]:-}")"
CHECKPOINT_NUM_COLS="$(empty_if_none "${checkpoint_defaults_lines[10]:-}")"
CHECKPOINT_PERCEPTION_ENABLED="$(empty_if_none "${checkpoint_defaults_lines[11]:-}")"
CHECKPOINT_PERCEPTION_OUTPUT_MODE="$(empty_if_none "${checkpoint_defaults_lines[12]:-}")"

if [[ -n "${CHECKPOINT_MOTION_DIR}" ]]; then
  CHECKPOINT_MOTION_DIR="$(resolve_data_path "${CHECKPOINT_MOTION_DIR}")"
fi
if [[ -n "${CHECKPOINT_GEOMETRY_DIR}" ]]; then
  CHECKPOINT_GEOMETRY_DIR="$(resolve_data_path "${CHECKPOINT_GEOMETRY_DIR}")"
fi
if [[ -n "${CHECKPOINT_GEOMETRY_METADATA}" ]]; then
  CHECKPOINT_GEOMETRY_METADATA="$(resolve_data_path "${CHECKPOINT_GEOMETRY_METADATA}")"
fi

if [[ "${MOTION_DIR_SET}" != "1" ]]; then
  if [[ -n "${CHECKPOINT_MOTION_DIR}" && ( "${CHECKPOINT_MOTION_DIR}" == s3://* || -e "${CHECKPOINT_MOTION_DIR}" ) ]]; then
    MOTION_DIR="${CHECKPOINT_MOTION_DIR}"
  else
    MOTION_DIR="/data/terrain/___crisp_clean_motion"
  fi
fi
if [[ "${GEOMETRY_DIR_SET}" != "1" ]]; then
  if [[ -n "${CHECKPOINT_GEOMETRY_DIR}" && -e "${CHECKPOINT_GEOMETRY_DIR}" ]]; then
    GEOMETRY_DIR="${CHECKPOINT_GEOMETRY_DIR}"
  else
    GEOMETRY_DIR="/data/terrain/___crisp_clean_geometry"
  fi
fi
if [[ "${GEOMETRY_METADATA_SET}" != "1" ]]; then
  if [[ -n "${CHECKPOINT_GEOMETRY_METADATA}" && -f "${CHECKPOINT_GEOMETRY_METADATA}" && -n "${CHECKPOINT_GEOMETRY_DIR}" && "${GEOMETRY_DIR}" == "${CHECKPOINT_GEOMETRY_DIR}" ]]; then
    GEOMETRY_METADATA="${CHECKPOINT_GEOMETRY_METADATA}"
  elif [[ -f "${GEOMETRY_DIR}" && -f "${GEOMETRY_DIR%.*}.json" ]]; then
    GEOMETRY_METADATA="${GEOMETRY_DIR%.*}.json"
  else
    GEOMETRY_METADATA=""
  fi
fi

if [[ "${MOTION_CLIP_NAME_FROM_ENV}" != "1" && -n "${CHECKPOINT_MOTION_CLIP_NAME}" ]]; then
  MOTION_CLIP_NAME="${CHECKPOINT_MOTION_CLIP_NAME}"
else
  MOTION_CLIP_NAME="${MOTION_CLIP_NAME:-}"
fi
if [[ "${MOTION_CLIP_ID_FROM_ENV}" != "1" && -n "${CHECKPOINT_MOTION_CLIP_ID}" ]]; then
  MOTION_CLIP_ID="${CHECKPOINT_MOTION_CLIP_ID}"
else
  MOTION_CLIP_ID="${MOTION_CLIP_ID:-}"
fi

if [[ ! -e "${MOTION_DIR}" && "${MOTION_DIR}" != s3://* ]]; then
  echo "[ERROR] MOTION_DIR not found: ${MOTION_DIR}" >&2
  exit 1
fi
if [[ ! -e "${GEOMETRY_DIR}" ]]; then
  echo "[ERROR] GEOMETRY_DIR not found: ${GEOMETRY_DIR}" >&2
  exit 1
fi
if [[ -n "${GEOMETRY_METADATA:-}" && ! -f "${GEOMETRY_METADATA}" ]]; then
  echo "[ERROR] GEOMETRY_METADATA not found: ${GEOMETRY_METADATA}" >&2
  exit 1
fi
if [[ -d "${MOTION_DIR}" && -n "${MOTION_CLIP_NAME}" && ! -f "${MOTION_DIR}/${MOTION_CLIP_NAME}.npz" ]]; then
  echo "[ERROR] MOTION_CLIP_NAME not found in MOTION_DIR: ${MOTION_CLIP_NAME}.npz" >&2
  exit 2
fi

HEADLESS_RAW="${HEADLESS:-True}"
HEADLESS_FLAG="$(normalize_bool_flag HEADLESS "${HEADLESS_RAW}")"
export HEADLESS="$(normalize_bool_int HEADLESS "${HEADLESS_RAW}")"
DISABLE_RANDOMIZATION="${DISABLE_RANDOMIZATION:-True}"
DISABLE_RANDOMIZATION_FLAG="$(normalize_bool_flag DISABLE_RANDOMIZATION "${DISABLE_RANDOMIZATION}")"
DRY_RUN_RAW="${DRY_RUN:-0}"
DRY_RUN_FLAG="$(normalize_bool_int DRY_RUN "${DRY_RUN_RAW}")"

if [[ -n "${PAIR_TERRAIN_WITH_MOTION+x}" ]]; then
  PAIR_TERRAIN_WITH_MOTION_RAW="${PAIR_TERRAIN_WITH_MOTION}"
elif [[ -n "${CHECKPOINT_PAIR_TERRAIN_WITH_MOTION}" ]]; then
  PAIR_TERRAIN_WITH_MOTION_RAW="${CHECKPOINT_PAIR_TERRAIN_WITH_MOTION}"
else
  PAIR_TERRAIN_WITH_MOTION_RAW="True"
fi
PAIR_TERRAIN_WITH_MOTION_FLAG="$(normalize_bool_flag PAIR_TERRAIN_WITH_MOTION "${PAIR_TERRAIN_WITH_MOTION_RAW}")"

NUM_ENVS="${NUM_ENVS:-1}"
VISER_PORT="${VISER_PORT:-$((RANDOM % 8976 + 1024))}"
VISER_ENV_ID="${VISER_ENV_ID:-0}"
VISER_UPDATE_HZ="${VISER_UPDATE_HZ:-30}"
VISER_RECENTER="${VISER_RECENTER:-True}"
VISER_SYNC_TO_SIM="${VISER_SYNC_TO_SIM:-True}"
VISER_FORCE_DT="${VISER_FORCE_DT:-True}"
VISER_SHOW_SCANDOTS="${VISER_SHOW_SCANDOTS:-False}"
VIS_GPU="${VIS_GPU:-auto}"
START_AT_TIMESTEP_ZERO_PROB="${START_AT_TIMESTEP_ZERO_PROB:-${CHECKPOINT_START_AT_TIMESTEP_ZERO_PROB:-1.0}}"
FREEZE_AT_TIMESTEP_ZERO_PROB="${FREEZE_AT_TIMESTEP_ZERO_PROB:-${CHECKPOINT_FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}}"
RESET_NOISE_SCALE="${RESET_NOISE_SCALE:-${CHECKPOINT_RESET_NOISE_SCALE:-0.0}}"
MAX_EPISODE_LENGTH_S="${MAX_EPISODE_LENGTH_S:-1000000}"
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:-}"
PHYSX_GPU_COLLISION_STACK_SIZE="${PHYSX_GPU_COLLISION_STACK_SIZE:-536870912}"
SIM_ENV_SPACING="${SIM_ENV_SPACING:-0.0}"
TERRAIN_NUM_ROWS="${TERRAIN_NUM_ROWS:-${NUM_ROWS:-${CHECKPOINT_NUM_ROWS:-}}}"
TERRAIN_NUM_COLS="${TERRAIN_NUM_COLS:-${NUM_COLS:-${CHECKPOINT_NUM_COLS:-}}}"

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

export VISER_ENABLE_CLIP_GUI="${VISER_ENABLE_CLIP_GUI:-1}"
export VISER_ENABLE_MANUAL_GUI="${VISER_ENABLE_MANUAL_GUI:-1}"
export VISER_MANUAL_CONTROL_DEFAULT="${VISER_MANUAL_CONTROL_DEFAULT:-1}"
export VISER_FORCE_MANUAL_CONTROL="${VISER_FORCE_MANUAL_CONTROL:-0}"
export VISER_SHOW_TARGET_KEYPOINTS="${VISER_SHOW_TARGET_KEYPOINTS:-1}"
export VISER_START_PAUSED="${VISER_START_PAUSED:-0}"
export VISER_MANUAL_USE_HW_JOYSTICK="${VISER_MANUAL_USE_HW_JOYSTICK:-0}"
export VISER_MANUAL_HW_BACKEND="${VISER_MANUAL_HW_BACKEND:-auto}"
export VISER_MANUAL_HW_DEVICE="${VISER_MANUAL_HW_DEVICE:-0}"
export VISER_MANUAL_HW_TYPE="${VISER_MANUAL_HW_TYPE:-xbox}"
export VISER_CLIP_LOCK_DEFAULT="${VISER_CLIP_LOCK_DEFAULT:-1}"
export HOLOSOMA_DISABLE_MOTION_END_RESET="${HOLOSOMA_DISABLE_MOTION_END_RESET:-1}"
export HOLOSOMA_RESET_TO_DEFAULT_POSE="${HOLOSOMA_RESET_TO_DEFAULT_POSE:-1}"
export LOGURU_LEVEL="${LOGURU_LEVEL:-WARNING}"
export PY_LOG_LEVEL="${PY_LOG_LEVEL:-WARNING}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PERCEPTION_PRESET="${PERCEPTION_PRESET:-checkpoint}"
case "$(echo "${PERCEPTION_PRESET}" | tr '[:upper:]' '[:lower:]')" in
  checkpoint|auto|"")
    if [[ "$(echo "${CHECKPOINT_PERCEPTION_ENABLED}" | tr '[:upper:]' '[:lower:]')" == "true" && "$(echo "${CHECKPOINT_PERCEPTION_OUTPUT_MODE}" | tr '[:upper:]' '[:lower:]')" == "camera_depth" ]]; then
      export VISER_PERCEPTION_IMAGE_MODE="${VISER_PERCEPTION_IMAGE_MODE:-depth}"
      export VISER_SHOW_PERCEPTION_FRUSTUM="${VISER_SHOW_PERCEPTION_FRUSTUM:-1}"
    fi
    ;;
  camera_depth_d435i|camera-depth-d435i|d435i|camera_depth)
    export VISER_PERCEPTION_IMAGE_MODE="${VISER_PERCEPTION_IMAGE_MODE:-depth}"
    export VISER_SHOW_PERCEPTION_FRUSTUM="${VISER_SHOW_PERCEPTION_FRUSTUM:-1}"
    ;;
  camera_depth_d435i_17x17|camera-depth-d435i-17x17|d435i_17x17)
    export VISER_PERCEPTION_IMAGE_MODE="${VISER_PERCEPTION_IMAGE_MODE:-depth}"
    export VISER_SHOW_PERCEPTION_FRUSTUM="${VISER_SHOW_PERCEPTION_FRUSTUM:-1}"
    ;;
  heightmap|none)
    ;;
  *)
    echo "[ERROR] PERCEPTION_PRESET must be one of: checkpoint|none|heightmap|camera_depth_d435i|camera_depth_d435i_17x17. Got: ${PERCEPTION_PRESET}" >&2
    exit 2
    ;;
esac

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
  --checkpoint "${CHECKPOINT}"
  --motion-dir "${MOTION_DIR}"
  --num-envs "${NUM_ENVS}"
  --headless "${HEADLESS_FLAG}"
  --pair-terrain-with-motion "${PAIR_TERRAIN_WITH_MOTION_FLAG}"
  --viser-port "${VISER_PORT}"
  --viser-env-id "${VISER_ENV_ID}"
  --viser-update-hz "${VISER_UPDATE_HZ}"
  --viser-recenter "${VISER_RECENTER}"
  --training.viser_sync_to_sim "${VISER_SYNC_TO_SIM}"
  --training.viser_force_dt "${VISER_FORCE_DT}"
  --training.viser_show_scandots "${VISER_SHOW_SCANDOTS}"
  --simulator.config.scene.env_spacing "${SIM_ENV_SPACING}"
  --simulator.config.sim.max_episode_length_s "${MAX_EPISODE_LENGTH_S}"
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
  --geometry-dir "${GEOMETRY_DIR}"
)
if [[ "${SIMULATOR_SUBCOMMAND}" != "simulator:mujoco" ]]; then
  cmd+=(--simulator.config.sim.physx.gpu_collision_stack_size "${PHYSX_GPU_COLLISION_STACK_SIZE}")
else
  cmd+=(--randomization.ignore_unsupported True)
fi
if [[ -n "${GEOMETRY_METADATA:-}" ]]; then
  cmd+=(--geometry-metadata "${GEOMETRY_METADATA}")
fi
if [[ -n "${TERRAIN_NUM_ROWS:-}" ]]; then
  cmd+=(--num-rows "${TERRAIN_NUM_ROWS}")
fi
if [[ -n "${TERRAIN_NUM_COLS:-}" ]]; then
  cmd+=(--num-cols "${TERRAIN_NUM_COLS}")
fi
if [[ -n "${MOTION_CLIP_NAME}" ]]; then
  cmd+=(--command.setup_terms.motion_command.params.motion_config.motion_clip_name "${MOTION_CLIP_NAME}")
fi
if [[ -n "${MOTION_CLIP_ID}" ]]; then
  cmd+=(--command.setup_terms.motion_command.params.motion_config.motion_clip_id "${MOTION_CLIP_ID}")
fi
if [[ -n "${MAX_EVAL_STEPS}" ]]; then
  cmd+=(--training.max_eval_steps "${MAX_EVAL_STEPS}")
fi

if [[ "${DISABLE_RANDOMIZATION_FLAG}" == "True" ]]; then
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
  )
fi

case "$(echo "${PERCEPTION_PRESET}" | tr '[:upper:]' '[:lower:]')" in
  checkpoint|auto|"")
    ;;
  none)
    cmd+=(perception:none)
    ;;
  heightmap)
    cmd+=(perception:heightmap)
    ;;
  camera_depth_d435i|camera-depth-d435i|d435i|camera_depth)
    cmd+=(perception:camera_depth_d435i)
    ;;
  camera_depth_d435i_17x17|camera-depth-d435i-17x17|d435i_17x17)
    cmd+=(perception:camera_depth_d435i_17x17)
    ;;
esac

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] checkpoint=${CHECKPOINT}"
echo "[INFO] motion_dir=${MOTION_DIR}"
echo "[INFO] geometry_dir=${GEOMETRY_DIR}"
echo "[INFO] geometry_metadata=${GEOMETRY_METADATA:-<none>}"
echo "[INFO] headless=${HEADLESS_FLAG} (env HEADLESS=${HEADLESS})"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] viser_port=${VISER_PORT}"
echo "[INFO] manual_gui=${VISER_ENABLE_MANUAL_GUI} clip_gui=${VISER_ENABLE_CLIP_GUI}"
echo "[INFO] hw_joystick=${VISER_MANUAL_USE_HW_JOYSTICK} backend=${VISER_MANUAL_HW_BACKEND}"
echo "[INFO] pair_terrain_with_motion=${PAIR_TERRAIN_WITH_MOTION_FLAG}"
echo "[INFO] perception_preset=${PERCEPTION_PRESET}"
echo "[INFO] checkpoint_perception=${CHECKPOINT_PERCEPTION_ENABLED:-<unknown>}/${CHECKPOINT_PERCEPTION_OUTPUT_MODE:-<unknown>}"
if [[ -n "${MOTION_CLIP_NAME}" ]]; then
  echo "[INFO] motion_clip_name=${MOTION_CLIP_NAME}"
fi

if [[ "${DRY_RUN_FLAG}" == "1" ]]; then
  echo "[INFO] local_url=http://localhost:${VISER_PORT} (dry run preview)"
  printf '[DRY_RUN] '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

run_with_delayed_local_url "${VISER_PORT}" "${cmd[@]}"
