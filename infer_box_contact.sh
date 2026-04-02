#!/usr/bin/env bash
set -euo pipefail

# Inference launcher for checkpoints produced by train_object_extend.sh.
#
# Supports both:
# - classic extend checkpoints (co-tracking focused)
# - command-curriculum extend checkpoints (sparse-goal capable)
#
# The script is checkpoint-driven:
# - if no checkpoint is provided, it prefers the latest local command-curriculum checkpoint
# - if none is found, it falls back to the pinned W&B command-curriculum run below
# - if that cannot be resolved, it falls back to the latest local smoke checkpoint
# - motion/object defaults come from the checkpoint's serialized experiment_config
# - runtime mode can override sparse-goal eval behavior when the checkpoint supports it
#
# Usage:
#   bash infer_box_contact.sh [auto|track|manual|goal] [checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra tyro args...]
#
# Modes:
#   auto    Use checkpoint eval settings as-is
#   track   Force pure co-tracking eval (command-only fraction = 0)
#   manual  Force command-only eval and use Viser manual-goal GUI
#   goal    Force command-only external sparse-goal eval

usage() {
  cat <<'EOF'
Usage:
  bash infer_box_contact.sh [auto|track|manual|goal] [checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra tyro args...]

Examples:
  bash infer_box_contact.sh
  bash infer_box_contact.sh track
  bash infer_box_contact.sh manual /abs/path/to/model.pt
  bash infer_box_contact.sh goal https://wandb.ai/zihanw22/boxer/runs/abcdef12

Default checkpoint resolution:
  1. Latest local non-smoke g1_29dof_wbt_w_object_command_curriculum checkpoint
  2. FALLBACK_CHECKPOINT_REF / pinned W&B command-curriculum run
  3. Latest local smoke g1_29dof_wbt_w_object_command_curriculum checkpoint

Optional env vars:
  CHECKPOINT / CKPT             Optional checkpoint override
  WANDB_MODEL_FILE              Used when a W&B run URL does not include /files/<checkpoint>.pt
  LOGGER_BASE_DIR               Default: /data/logs_new
  WANDB_PROJECT                 Default: boxer
  MOTION_DIR                    Optional motion source override
  MOTION_CLIP_NAME              Optional single-clip override; default pins inference to sub3_largebox_003_mj_w_obj when MOTION_DIR is a directory
  OBJECT_URDF                   Optional object URDF / map override
  NUM_ENVS                      Default: 1
  HEADLESS                      Default: False
  VISER_PORT                    Default: random
  VISER_ENV_ID                  Default: 0
  VISER_UPDATE_HZ               Default: 30
  VISER_RECENTER                Default: True
  VISER_SYNC_TO_SIM             Default: True
  VISER_FORCE_DT                Default: True
  VISER_LOAD_URDF               Default: 1
  VIS_GPU                       Default: auto
  DISABLE_RANDOMIZATION         Default: True
  START_AT_TIMESTEP_ZERO_PROB   Default: checkpoint / 1.0 fallback
  FREEZE_AT_TIMESTEP_ZERO_PROB  Default: checkpoint / 0.0 fallback
  RESET_NOISE_SCALE             Default: checkpoint / 0.0 fallback
  MAX_EPISODE_LENGTH_S          Default: 1000000
  MAX_EVAL_STEPS                Optional eval step cap
  PHYSX_GPU_COLLISION_STACK_SIZE Default: 268435456
  EVAL_COMMAND_ONLY_ENV_PROB    Optional sparse-goal eval override
  EVAL_EXTERNAL_GOAL_PROB       Optional sparse-goal eval override
  EVAL_CARRY_EXTENSION_PROB     Optional sparse-goal eval override
  DRY_RUN                       Default: 0
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOGGER_BASE_DIR="${LOGGER_BASE_DIR:-/data/logs_new}"
WANDB_PROJECT="${WANDB_PROJECT:-boxer}"
DEFAULT_FALLBACK_CHECKPOINT_REF="https://wandb.ai/zihanw22/boxer/runs/yoecm2af"
FALLBACK_CHECKPOINT_REF="${FALLBACK_CHECKPOINT_REF:-${DEFAULT_FALLBACK_CHECKPOINT_REF}}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help|help)
      usage
      exit 0
      ;;
  esac
fi

MODE="${MODE:-auto}"
if [[ $# -gt 0 ]]; then
  case "$1" in
    auto|track|manual|goal)
      MODE="$1"
      shift
      ;;
  esac
fi

case "${MODE}" in
  auto|track|manual|goal) ;;
  *)
    echo "[ERROR] mode must be one of: auto|track|manual|goal. Got: ${MODE}" >&2
    exit 2
    ;;
esac

DEFAULT_SINGLE_MOTION_SOURCE="${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
DEFAULT_SINGLE_OBJECT_URDF="${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"
DEFAULT_SINGLE_MOTION_CLIP_NAME="sub3_largebox_003_mj_w_obj"

WANDB_MODEL_FILE_FROM_ENV=0
if [[ -n "${WANDB_MODEL_FILE+x}" && -n "${WANDB_MODEL_FILE}" ]]; then
  WANDB_MODEL_FILE_FROM_ENV=1
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
  local model_file=""

  parsed="$(parse_wandb_run_url "${ref}" || true)"
  if [[ -z "${parsed}" ]]; then
    echo "${ref}"
    return 0
  fi

  IFS=$'\t' read -r entity project run_id explicit_file <<< "${parsed}"
  if [[ -n "${explicit_file}" ]]; then
    model_file="${explicit_file}"
  elif [[ "${WANDB_MODEL_FILE_FROM_ENV}" == "1" ]]; then
    model_file="${WANDB_MODEL_FILE}"
  else
    model_file="$(resolve_remote_wandb_checkpoint_name "${entity}" "${project}" "${run_id}")"
    if [[ -n "${model_file}" ]]; then
      echo "[INFO] Resolved wandb run URL to remote checkpoint: ${model_file}" >&2
    fi
  fi

  if [[ -z "${model_file}" ]]; then
    echo "[ERROR] Could not determine a .pt checkpoint for W&B run URL: ${ref}" >&2
    echo "[ERROR] Pass a /files/<checkpoint>.pt URL or set WANDB_MODEL_FILE." >&2
    return 2
  fi

  echo "wandb://${entity}/${project}/${run_id}/${model_file}"
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

find_latest_command_curriculum_checkpoint() {
  LOGGER_BASE_DIR="${LOGGER_BASE_DIR}" WANDB_PROJECT="${WANDB_PROJECT}" "${PYTHON_BIN}" - <<'PY'
import os
import re
from pathlib import Path

base = Path(os.environ["LOGGER_BASE_DIR"]) / os.environ["WANDB_PROJECT"]
if not base.is_dir():
    raise SystemExit(0)

name_pattern = re.compile(r"g1_29dof_wbt_w_object_command_curriculum")
dirs = []
for path in base.iterdir():
    if not path.is_dir():
        continue
    name = path.name
    if "-eval" in name or "smoke" in name:
        continue
    if not name_pattern.search(name):
        continue
    models = sorted(path.glob("model_*.pt"))
    if not models:
        continue
    dirs.append((path.stat().st_mtime, path, models))

if not dirs:
    raise SystemExit(0)

_mtime, run_dir, models = max(dirs, key=lambda item: item[0])
print(str(models[-1]))
PY
}

find_latest_command_curriculum_checkpoint_including_smoke() {
  LOGGER_BASE_DIR="${LOGGER_BASE_DIR}" WANDB_PROJECT="${WANDB_PROJECT}" "${PYTHON_BIN}" - <<'PY'
import os
import re
from pathlib import Path

base = Path(os.environ["LOGGER_BASE_DIR"]) / os.environ["WANDB_PROJECT"]
if not base.is_dir():
    raise SystemExit(0)

name_pattern = re.compile(r"g1_29dof_wbt_w_object_command_curriculum")
dirs = []
for path in base.iterdir():
    if not path.is_dir():
        continue
    name = path.name
    if "-eval" in name:
        continue
    if not name_pattern.search(name):
        continue
    models = sorted(path.glob("model_*.pt"))
    if not models:
        continue
    dirs.append((path.stat().st_mtime, path, models))

if not dirs:
    raise SystemExit(0)

_mtime, run_dir, models = max(dirs, key=lambda item: item[0])
print(str(models[-1]))
PY
}

extract_checkpoint_metadata() {
  local checkpoint_ref="$1"

  "${PYTHON_BIN}" - "${checkpoint_ref}" <<'PY' 2>/dev/null || true
import json
import sys
import tempfile
from pathlib import Path

import torch


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
        run = wandb.Api(timeout=30).run(run_path)
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloaded = run.file(ckpt_name).download(root=tmp_dir, replace=True)
            ckpt_path = Path(downloaded.name)
            if not ckpt_path.is_absolute():
                ckpt_path = (Path.cwd() / ckpt_path).resolve()
            return torch.load(ckpt_path, map_location="cpu")
    return torch.load(checkpoint_ref, map_location="cpu")


payload = load_payload(sys.argv[1])
cfg = payload.get("experiment_config")
if not isinstance(cfg, dict):
    raise SystemExit(0)

training_cfg = cfg.get("training", {})
algo_cfg = cfg.get("algo", {})
algo_inner_cfg = algo_cfg.get("config", {}) if isinstance(algo_cfg, dict) else {}
module_dict = algo_inner_cfg.get("module_dict", {}) if isinstance(algo_inner_cfg, dict) else {}
actor_cfg = module_dict.get("actor", {}) if isinstance(module_dict, dict) else {}
actor_inputs = actor_cfg.get("input_dim", []) if isinstance(actor_cfg, dict) else []

command_cfg = cfg.get("command", {})
setup_terms = command_cfg.get("setup_terms", {}) if isinstance(command_cfg, dict) else {}
motion_command = setup_terms.get("motion_command", {}) if isinstance(setup_terms, dict) else {}
params = motion_command.get("params", {}) if isinstance(motion_command, dict) else {}
motion_cfg = params.get("motion_config", {}) if isinstance(params, dict) else {}
sparse_goal_cfg = motion_cfg.get("sparse_object_goal", {}) if isinstance(motion_cfg, dict) else {}

robot_cfg = cfg.get("robot", {})
object_cfg = robot_cfg.get("object", {}) if isinstance(robot_cfg, dict) else {}

motion_file = motion_cfg.get("motion_file") if isinstance(motion_cfg, dict) else None
motion_clip_name = motion_cfg.get("motion_clip_name") if isinstance(motion_cfg, dict) else None
object_urdf_path = object_cfg.get("object_urdf_path") if isinstance(object_cfg, dict) else None
training_name = training_cfg.get("name") if isinstance(training_cfg, dict) else None
exp_name = cfg.get("name")
sparse_goal_enabled = bool(sparse_goal_cfg.get("enabled", False)) if isinstance(sparse_goal_cfg, dict) else False
command_capable = sparse_goal_enabled and any(
    name in set(actor_inputs)
    for name in (
        "actor_obs_goal",
        "actor_obs_mode",
        "actor_obs_drop",
        "actor_obs_drop_command",
        "actor_obs_drop_mixed",
        "actor_obs_root",
        "actor_obs_torso",
    )
)

def resolve_scheduled_prob(
    cfg: dict[str, object] | None,
    *,
    base_key: str,
    end_key: str,
    start_iter_key: str,
    end_iter_key: str,
):
    if not isinstance(cfg, dict):
        return None
    end_value = cfg.get(end_key)
    start_iter = cfg.get(start_iter_key)
    end_iter = cfg.get(end_iter_key)
    if end_value is not None and start_iter is not None and end_iter is not None:
        return end_value
    return cfg.get(base_key)


start_at_zero_prob = resolve_scheduled_prob(
    motion_cfg,
    base_key="start_at_timestep_zero_prob",
    end_key="start_at_timestep_zero_prob_end",
    start_iter_key="start_at_timestep_zero_prob_start_iter",
    end_iter_key="start_at_timestep_zero_prob_end_iter",
)
freeze_at_zero_prob = resolve_scheduled_prob(
    motion_cfg,
    base_key="freeze_at_timestep_zero_prob",
    end_key="freeze_at_timestep_zero_prob_end",
    start_iter_key="freeze_at_timestep_zero_prob_start_iter",
    end_iter_key="freeze_at_timestep_zero_prob_end_iter",
)
noise_cfg = motion_cfg.get("noise_to_initial_pose", {}) if isinstance(motion_cfg, dict) else {}
reset_noise_scale = noise_cfg.get("overall_noise_scale") if isinstance(noise_cfg, dict) else None

for value in (
    motion_file,
    motion_clip_name,
    object_urdf_path,
    training_name,
    exp_name,
    json.dumps(actor_inputs),
    "1" if sparse_goal_enabled else "0",
    "1" if command_capable else "0",
    "" if start_at_zero_prob is None else str(start_at_zero_prob),
    "" if freeze_at_zero_prob is None else str(freeze_at_zero_prob),
    "" if reset_noise_scale is None else str(reset_noise_scale),
):
    print("" if value is None else str(value))
PY
}

is_truthy() {
  local value="${1,,}"
  [[ "${value}" == "1" || "${value}" == "true" || "${value}" == "yes" || "${value}" == "on" ]]
}

CHECKPOINT="${CHECKPOINT:-${CKPT:-}}"
if [[ $# -gt 0 ]]; then
  if [[ "$1" == wandb://* || "$1" == https://wandb.ai/*/runs/* || "$1" == /* || "$1" == ./* || "$1" == ../* || "$1" == *.pt ]]; then
    CHECKPOINT="$1"
    shift
  fi
fi

if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT="$(find_latest_command_curriculum_checkpoint || true)"
  if [[ -n "${CHECKPOINT}" ]]; then
    echo "[INFO] Using latest local command-curriculum checkpoint: ${CHECKPOINT}"
  elif [[ -n "${FALLBACK_CHECKPOINT_REF}" ]]; then
    CHECKPOINT="$(normalize_checkpoint_ref "${FALLBACK_CHECKPOINT_REF}" 2>/dev/null || true)"
    if [[ -n "${CHECKPOINT}" ]]; then
      echo "[INFO] No non-smoke local command-curriculum checkpoint found; using pinned W&B fallback: ${CHECKPOINT}"
    fi
  fi
  if [[ -z "${CHECKPOINT}" ]]; then
    CHECKPOINT="$(find_latest_command_curriculum_checkpoint_including_smoke || true)"
    if [[ -n "${CHECKPOINT}" ]]; then
      echo "[INFO] No non-smoke command-curriculum checkpoint found and pinned W&B fallback was unavailable; using latest local smoke checkpoint: ${CHECKPOINT}"
    else
      echo "[ERROR] No local command-curriculum checkpoint found under ${LOGGER_BASE_DIR}/${WANDB_PROJECT}." >&2
      echo "[ERROR] Pass a checkpoint explicitly or set FALLBACK_CHECKPOINT_REF." >&2
      exit 1
    fi
  fi
fi

if [[ "${CHECKPOINT}" == https://wandb.ai/*/runs/* ]]; then
  CHECKPOINT="$(normalize_checkpoint_ref "${CHECKPOINT}")"
fi

if [[ "${CHECKPOINT}" != wandb://* ]] && [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

mapfile -t checkpoint_meta_lines < <(extract_checkpoint_metadata "${CHECKPOINT}")
CHECKPOINT_MOTION_SOURCE="$(resolve_data_path "${checkpoint_meta_lines[0]:-}")"
CHECKPOINT_MOTION_CLIP_NAME="${checkpoint_meta_lines[1]:-}"
CHECKPOINT_OBJECT_URDF="$(resolve_data_path "${checkpoint_meta_lines[2]:-}")"
CHECKPOINT_TRAINING_NAME="${checkpoint_meta_lines[3]:-}"
CHECKPOINT_EXP_NAME="${checkpoint_meta_lines[4]:-}"
CHECKPOINT_ACTOR_INPUTS_JSON="${checkpoint_meta_lines[5]:-[]}"
CHECKPOINT_SPARSE_GOAL_ENABLED="${checkpoint_meta_lines[6]:-0}"
CHECKPOINT_COMMAND_CAPABLE="${checkpoint_meta_lines[7]:-0}"
CHECKPOINT_START_AT_ZERO_PROB="${checkpoint_meta_lines[8]:-}"
CHECKPOINT_FREEZE_AT_ZERO_PROB="${checkpoint_meta_lines[9]:-}"
CHECKPOINT_RESET_NOISE_SCALE="${checkpoint_meta_lines[10]:-}"

MOTION_DIR_FROM_ENV=0
[[ -n "${MOTION_DIR+x}" ]] && MOTION_DIR_FROM_ENV=1
OBJECT_URDF_FROM_ENV=0
[[ -n "${OBJECT_URDF+x}" ]] && OBJECT_URDF_FROM_ENV=1
MOTION_CLIP_NAME_FROM_ENV=0
[[ -n "${MOTION_CLIP_NAME+x}" ]] && MOTION_CLIP_NAME_FROM_ENV=1

if [[ "${MOTION_DIR_FROM_ENV}" != "1" ]]; then
  if [[ -n "${CHECKPOINT_MOTION_SOURCE}" ]]; then
    MOTION_DIR="${CHECKPOINT_MOTION_SOURCE}"
  else
    MOTION_DIR="${DEFAULT_SINGLE_MOTION_SOURCE}"
  fi
fi
if [[ "${OBJECT_URDF_FROM_ENV}" != "1" ]]; then
  if [[ -n "${CHECKPOINT_OBJECT_URDF}" ]]; then
    OBJECT_URDF="${CHECKPOINT_OBJECT_URDF}"
  else
    OBJECT_URDF="${DEFAULT_SINGLE_OBJECT_URDF}"
  fi
fi
if [[ "${MOTION_CLIP_NAME_FROM_ENV}" != "1" && -n "${CHECKPOINT_MOTION_CLIP_NAME}" ]]; then
  MOTION_CLIP_NAME="${CHECKPOINT_MOTION_CLIP_NAME}"
elif [[ "${MOTION_CLIP_NAME_FROM_ENV}" != "1" && -d "${MOTION_DIR}" ]]; then
  MOTION_CLIP_NAME="${DEFAULT_SINGLE_MOTION_CLIP_NAME}"
else
  MOTION_CLIP_NAME="${MOTION_CLIP_NAME:-}"
fi

NUM_ENVS="${NUM_ENVS:-1}"
HEADLESS_RAW="${HEADLESS:-False}"
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

PAIR_TERRAIN_WITH_MOTION="${PAIR_TERRAIN_WITH_MOTION:-False}"
VISER_PORT="${VISER_PORT:-$((RANDOM % 8976 + 1024))}"
VISER_ENV_ID="${VISER_ENV_ID:-0}"
VISER_UPDATE_HZ="${VISER_UPDATE_HZ:-30}"
VISER_RECENTER="${VISER_RECENTER:-True}"
VISER_SYNC_TO_SIM="${VISER_SYNC_TO_SIM:-True}"
VISER_FORCE_DT="${VISER_FORCE_DT:-True}"
VISER_SHOW_SCANDOTS="${VISER_SHOW_SCANDOTS:-False}"
VISER_LOAD_URDF="${VISER_LOAD_URDF:-1}"
START_AT_TIMESTEP_ZERO_PROB="${START_AT_TIMESTEP_ZERO_PROB:-${CHECKPOINT_START_AT_ZERO_PROB:-1.0}}"
FREEZE_AT_TIMESTEP_ZERO_PROB="${FREEZE_AT_TIMESTEP_ZERO_PROB:-${CHECKPOINT_FREEZE_AT_ZERO_PROB:-0.0}}"
RESET_NOISE_SCALE="${RESET_NOISE_SCALE:-${CHECKPOINT_RESET_NOISE_SCALE:-0.0}}"
MAX_EPISODE_LENGTH_S="${MAX_EPISODE_LENGTH_S:-1000000}"
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:-}"
SIM_ENV_SPACING="${SIM_ENV_SPACING:-0.0}"
PHYSX_GPU_COLLISION_STACK_SIZE="${PHYSX_GPU_COLLISION_STACK_SIZE:-268435456}"
DISABLE_RANDOMIZATION="${DISABLE_RANDOMIZATION:-True}"
VIS_GPU="${VIS_GPU:-auto}"
EVAL_COMMAND_ONLY_ENV_PROB="${EVAL_COMMAND_ONLY_ENV_PROB:-}"
EVAL_EXTERNAL_GOAL_PROB="${EVAL_EXTERNAL_GOAL_PROB:-}"
EVAL_CARRY_EXTENSION_PROB="${EVAL_CARRY_EXTENSION_PROB:-}"
DRY_RUN_RAW="${DRY_RUN:-0}"

DRY_RUN_NORM=$(echo "${DRY_RUN_RAW}" | tr '[:upper:]' '[:lower:]')
case "${DRY_RUN_NORM}" in
  1|true|yes|on) DRY_RUN_FLAG=1 ;;
  0|false|no|off|"") DRY_RUN_FLAG=0 ;;
  *)
    echo "[ERROR] DRY_RUN must be one of: 0/1/true/false/yes/no/on/off. Got: ${DRY_RUN_RAW}" >&2
    exit 2
    ;;
esac

if [[ "${CHECKPOINT_COMMAND_CAPABLE}" != "1" ]]; then
  if [[ "${MODE}" == "manual" || "${MODE}" == "goal" ]]; then
    echo "[ERROR] Checkpoint is not command-capable; mode='${MODE}' is not supported." >&2
    echo "[ERROR] training_name=${CHECKPOINT_TRAINING_NAME} exp=${CHECKPOINT_EXP_NAME}" >&2
    exit 2
  fi
fi

case "${MODE}" in
  track)
    if [[ "${CHECKPOINT_COMMAND_CAPABLE}" == "1" ]]; then
      EVAL_COMMAND_ONLY_ENV_PROB="${EVAL_COMMAND_ONLY_ENV_PROB:-0.0}"
      EVAL_EXTERNAL_GOAL_PROB="${EVAL_EXTERNAL_GOAL_PROB:-0.0}"
      EVAL_CARRY_EXTENSION_PROB="${EVAL_CARRY_EXTENSION_PROB:-0.0}"
    fi
    ;;
  manual)
    EVAL_COMMAND_ONLY_ENV_PROB="${EVAL_COMMAND_ONLY_ENV_PROB:-1.0}"
    EVAL_EXTERNAL_GOAL_PROB="${EVAL_EXTERNAL_GOAL_PROB:-0.0}"
    EVAL_CARRY_EXTENSION_PROB="${EVAL_CARRY_EXTENSION_PROB:-0.0}"
    ;;
  goal)
    EVAL_COMMAND_ONLY_ENV_PROB="${EVAL_COMMAND_ONLY_ENV_PROB:-1.0}"
    EVAL_EXTERNAL_GOAL_PROB="${EVAL_EXTERNAL_GOAL_PROB:-1.0}"
    EVAL_CARRY_EXTENSION_PROB="${EVAL_CARRY_EXTENSION_PROB:-0.0}"
    ;;
esac

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
if [[ "${CHECKPOINT_COMMAND_CAPABLE}" == "1" ]]; then
  export VISER_ENABLE_MANUAL_GOAL_GUI="${VISER_ENABLE_MANUAL_GOAL_GUI:-1}"
  export VISER_SHOW_TARGET_BOX="${VISER_SHOW_TARGET_BOX:-1}"
  export HOLOSOMA_DISABLE_BAD_TRACKING_RESET="${HOLOSOMA_DISABLE_BAD_TRACKING_RESET:-1}"
  export HOLOSOMA_DISABLE_AUTO_RESET="${HOLOSOMA_DISABLE_AUTO_RESET:-1}"
else
  export VISER_ENABLE_MANUAL_GOAL_GUI="${VISER_ENABLE_MANUAL_GOAL_GUI:-0}"
  export VISER_SHOW_TARGET_BOX="${VISER_SHOW_TARGET_BOX:-0}"
fi
export VISER_SHOW_TARGET_KEYPOINTS="${VISER_SHOW_TARGET_KEYPOINTS:-1}"
export VISER_START_PAUSED="${VISER_START_PAUSED:-0}"
export VISER_LOAD_URDF

if [[ "${CHECKPOINT}" != wandb://* ]] && [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] checkpoint not found: ${CHECKPOINT}" >&2
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

EXTRA_ARGS=("$@")

cmd=(
  "${PYTHON_BIN}" -m holosoma.visualize physics
  --checkpoint "${CHECKPOINT}"
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
  --simulator.config.sim.physx.gpu_collision_stack_size "${PHYSX_GPU_COLLISION_STACK_SIZE}"
  --robot.object.enabled True
  --robot.object.object_urdf_path "${OBJECT_URDF}"
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob "${START_AT_TIMESTEP_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.freeze_at_timestep_zero_prob "${FREEZE_AT_TIMESTEP_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.overall_noise_scale "${RESET_NOISE_SCALE}"
)

if [[ -n "${MOTION_CLIP_NAME}" ]]; then
  cmd+=(--command.setup_terms.motion_command.params.motion_config.motion_clip_name "${MOTION_CLIP_NAME}")
fi

if [[ -n "${MAX_EVAL_STEPS}" ]]; then
  cmd+=(--training.max_eval_steps "${MAX_EVAL_STEPS}")
fi

if [[ "${CHECKPOINT_COMMAND_CAPABLE}" == "1" ]]; then
  cmd+=(--command.setup_terms.motion_command.params.motion_config.sparse_object_goal.enabled True)
  if [[ -n "${EVAL_COMMAND_ONLY_ENV_PROB}" ]]; then
    cmd+=(
      --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.eval_command_only_env_prob
      "${EVAL_COMMAND_ONLY_ENV_PROB}"
    )
  fi
  if [[ -n "${EVAL_EXTERNAL_GOAL_PROB}" ]]; then
    cmd+=(
      --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.eval_external_goal_prob
      "${EVAL_EXTERNAL_GOAL_PROB}"
    )
  fi
  if [[ -n "${EVAL_CARRY_EXTENSION_PROB}" ]]; then
    cmd+=(
      --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.eval_carry_extension_prob
      "${EVAL_CARRY_EXTENSION_PROB}"
    )
  fi
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

echo "[INFO] mode=${MODE}"
echo "[INFO] checkpoint=${CHECKPOINT}"
echo "[INFO] checkpoint_training_name=${CHECKPOINT_TRAINING_NAME:-<unknown>}"
echo "[INFO] checkpoint_exp=${CHECKPOINT_EXP_NAME:-<unknown>}"
echo "[INFO] checkpoint_sparse_goal_enabled=${CHECKPOINT_SPARSE_GOAL_ENABLED}"
echo "[INFO] checkpoint_command_capable=${CHECKPOINT_COMMAND_CAPABLE}"
echo "[INFO] checkpoint_actor_inputs=${CHECKPOINT_ACTOR_INPUTS_JSON}"
echo "[INFO] motion_dir=${MOTION_DIR}"
echo "[INFO] motion_clip_name=${MOTION_CLIP_NAME:-<auto>}"
echo "[INFO] object_urdf=${OBJECT_URDF}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] headless=${HEADLESS_FLAG} (env HEADLESS=${HEADLESS})"
echo "[INFO] viser=http://localhost:${VISER_PORT}"
echo "[INFO] viser_sync_to_sim=${VISER_SYNC_TO_SIM} viser_force_dt=${VISER_FORCE_DT}"
echo "[INFO] viser_load_urdf=${VISER_LOAD_URDF}"
echo "[INFO] disable_randomization=${DISABLE_RANDOMIZATION}"
if [[ "${CHECKPOINT_COMMAND_CAPABLE}" == "1" ]]; then
  echo "[INFO] eval_command_only_env_prob=${EVAL_COMMAND_ONLY_ENV_PROB:-<checkpoint>}"
  echo "[INFO] eval_external_goal_prob=${EVAL_EXTERNAL_GOAL_PROB:-<checkpoint>}"
  echo "[INFO] eval_carry_extension_prob=${EVAL_CARRY_EXTENSION_PROB:-<checkpoint>}"
fi

if [[ "${DRY_RUN_FLAG}" == "1" ]]; then
  printf '[DRY_RUN] '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
