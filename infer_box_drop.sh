#!/usr/bin/env bash
set -euo pipefail

# IsaacSim + Viser evaluation for box-drop student policies.
#
# Branches:
# - clip:  clip-conditioned drop student (default run: oitf644a)
# - mixed: sparse-goal mixed drop student (default run: s221l5eo)
#
# Usage:
#   bash infer_box_drop.sh [clip|mixed] [checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra tyro args...]
#
# Examples:
#   bash infer_box_drop.sh
#   bash infer_box_drop.sh https://wandb.ai/zihanw22/boxer/runs/kmux2yeq/logs
#   bash infer_box_drop.sh clip
#   bash infer_box_drop.sh mixed
#   bash infer_box_drop.sh mixed https://wandb.ai/zihanw22/boxer/runs/s221l5eo
#   MOTION_CLIP_NAME=sub3_largebox_003_mj_w_obj bash infer_box_drop.sh clip

usage() {
  cat <<'EOF'
Usage:
  bash infer_box_drop.sh [clip|mixed] [checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra tyro args...]

Accepted W&B URLs:
  https://wandb.ai/<entity>/<project>/runs/<run_id>
  https://wandb.ai/<entity>/<project>/runs/<run_id>/logs
  https://wandb.ai/<entity>/<project>/runs/<run_id>/files/<checkpoint>.pt

Modes:
  clip   Evaluate the clip-conditioned drop student (oitf644a by default)
  mixed  Evaluate the sparse-goal mixed drop student (s221l5eo by default; also the default mode if omitted)

Default W&B runs:
  clip   https://wandb.ai/zihanw22/boxer/runs/oitf644a
  mixed  https://wandb.ai/zihanw22/boxer/runs/s221l5eo

  Optional env vars:
    CKPT / CHECKPOINT        (optional checkpoint override)
    WANDB_MODEL_FILE         (optional; used when checkpoint is a W&B run URL without /files/<checkpoint>)
    ROBOT_INIT_STATE_POS     (optional absolute robot init-state position override: [x,y,z] or x,y,z)
    ROBOT_INIT_STATE_XY_OFFSET
                             (optional relative x,y offset applied to checkpoint default init-state XY
                              when default-pose init is enabled at launch)
    INFER_DATASET            (default: omomo; options: omomo|behave|mixed)
  MOTION_DIR               (optional override)
  MOTION_CLIP_NAME         (optional single clip name)
  MOTION_CLIP_ID           (optional single clip id)
  OBJECT_URDF              (optional override)
  OBJECT_SCALE             (optional scalar or x,y,z)
  GEOMETRY_DIR             (optional OBJ file/dir for terrain visualization)
  NUM_ENVS                 (default: 1)
  HEADLESS                 (default: True)
  VISER_PORT               (default: random)
  VISER_ENV_ID             (default: 0)
  VISER_UPDATE_HZ          (default: 30)
  VISER_RECENTER           (default: True)
  VIS_GPU                  (default: auto)
  PAIR_TERRAIN_WITH_MOTION (default: False)
  DISABLE_RANDOMIZATION    (default: True)
  START_AT_TIMESTEP_ZERO_PROB (default: 1.0)
  FREEZE_AT_TIMESTEP_ZERO_PROB (default: 0.0)
  RESET_NOISE_SCALE        (default: 0.0; s221l5eo mixed profile defaults to 1.0)
  MAX_EPISODE_LENGTH_S     (default: 1000000)
  MAX_EVAL_STEPS           (optional; if set, overrides training.max_eval_steps)
  PHYSX_GPU_COLLISION_STACK_SIZE (default: 268435456)
  DEPTH_PERCEPTION_PRESET  (default: checkpoint; options: checkpoint|d435i_17x17)
  MIXED_PROFILE            (default: auto; options: auto|none|s221l5eo)
  EVAL_COMMAND_ONLY_ENV_PROB (mixed mode default: 1.0 for s221l5eo; clip mode leaves checkpoint logic unchanged)
  EVAL_EXTERNAL_GOAL_PROB  (mixed mode default: 1.0; clip mode leaves checkpoint logic unchanged)
  HOLOSOMA_DISABLE_BAD_TRACKING_RESET (default: 1 for infer)
  HOLOSOMA_DISABLE_AUTO_RESET (default: 1 for infer; only GUI/manual reset will reset)
  DRY_RUN                  (default: 0; set 1/true to print the command without launching)
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"

is_checkpoint_arg() {
  local arg="${1:-}"
  [[ "${arg}" == wandb://* || "${arg}" == https://wandb.ai/* || "${arg}" == /* || "${arg}" == ./* || "${arg}" == ../* || "${arg}" == *.pt ]]
}

MODE="mixed"
MODE_INPUT="<default:mixed>"
if [[ $# -gt 0 ]]; then
  case "$1" in
    clip|drop|oitf644a)
      MODE_INPUT="$1"
      MODE="clip"
      shift
      ;;
    mixed|sparse_goal|sparse-goal|hw5jbitz|q3t3ntf4|s221l5eo)
      MODE_INPUT="$1"
      MODE="mixed"
      shift
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      if ! is_checkpoint_arg "$1"; then
        echo "[ERROR] first argument must be mode clip|mixed or a checkpoint/W&B run reference. Got: $1" >&2
        exit 2
      fi
      ;;
  esac
fi

DEFAULT_CLIP_RUN_URL="${DEFAULT_CLIP_RUN_URL:-https://wandb.ai/zihanw22/boxer/runs/oitf644a}"
DEFAULT_MIXED_RUN_URL="${DEFAULT_MIXED_RUN_URL:-https://wandb.ai/zihanw22/boxer/runs/s221l5eo}"
DEFAULT_CLIP_CHECKPOINT="${DEFAULT_CLIP_CHECKPOINT:-wandb://zihanw22/boxer/oitf644a/model_01600.pt}"
DEFAULT_MIXED_CHECKPOINT="${DEFAULT_MIXED_CHECKPOINT:-wandb://zihanw22/boxer/s221l5eo/model_03600.pt}"

default_model_file_for_run_id() {
  local run_id="$1"
  case "${run_id}" in
    kmux2yeq) echo "model_01200.pt" ;;
    oitf644a) echo "model_01600.pt" ;;
    q3t3ntf4) echo "model_01400.pt" ;;
    hw5jbitz) echo "model_02800.pt" ;;
    s221l5eo) echo "model_03600.pt" ;;
    *) echo "" ;;
  esac
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

  "$PYTHON_BIN" - "${entity}" "${project}" "${run_id}" <<'PY' 2>/dev/null || true
import re
import sys
import time
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
model_pattern = re.compile(r"^model_(\d+)\.pt$")
latest_name = ""
for attempt in range(3):
    try:
        api = wandb.Api(timeout=30)
        run = api.run(f"{entity}/{project}/{run_id}")
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
            break
    except Exception:
        if attempt == 2:
            break
        time.sleep(1.0)
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
  local builtin_model_file=""
  local remote_model_file=""

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
    echo ""
    return 0
  fi

  wandb_run_dir=$(find /data/logs_new -maxdepth 8 -type d -name "run-*-${run_id}" 2>/dev/null | head -n 1 || true)
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

normalize_xy_offset() {
  local raw="$1"
  local compact="${raw//[\[\]\(\)[:space:]]/}"
  local values=()
  if [[ -z "${compact}" ]]; then
    echo ""
    return 0
  fi
  IFS=',' read -r -a values <<< "${compact}"
  if [[ "${#values[@]}" -ne 2 ]]; then
    echo "[ERROR] ROBOT_INIT_STATE_XY_OFFSET must be a comma-separated x,y pair. Got: ${raw}" >&2
    exit 2
  fi
  printf '%s,%s' "${values[0]}" "${values[1]}"
}

is_truthy() {
  local raw="${1:-}"
  case "$(echo "${raw}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_robot_init_state_pos_with_xy_offset() {
  local checkpoint_ref="$1"
  local normalized_xy_offset="$2"
  "$PYTHON_BIN" - "${checkpoint_ref}" "${normalized_xy_offset}" <<'PY'
import json
import sys

from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config

checkpoint_ref = sys.argv[1]
dx_str, dy_str = sys.argv[2].split(",", 1)
dx = float(dx_str)
dy = float(dy_str)

saved_cfg, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=checkpoint_ref))
init_pos = list(saved_cfg.robot.init_state.pos)
if len(init_pos) != 3:
    raise ValueError(f"Expected robot.init_state.pos to have 3 values, got: {init_pos}")

init_pos[0] = float(init_pos[0]) + dx
init_pos[1] = float(init_pos[1]) + dy
init_pos[2] = float(init_pos[2])
print(json.dumps(init_pos, separators=(",", ":")))
PY
}

load_checkpoint_saved_motion_defaults() {
  local checkpoint_ref="$1"
  "$PYTHON_BIN" - "${checkpoint_ref}" "${SCRIPT_DIR}" <<'PY' 2>/dev/null || true
import json
import re
import sys
import tempfile
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
    import torch
    from holosoma.utils.eval_utils import load_checkpoint
except Exception:
    print(json.dumps({}))
    sys.exit(0)

checkpoint_ref = sys.argv[1]
script_dir = Path(sys.argv[2]).resolve()
retarget_root = script_dir / "src" / "holosoma_retargeting"
holosoma_root = script_dir / "src" / "holosoma"


def resolve_saved_path(raw_path: str | None) -> str | None:
    if not raw_path:
        return None

    original = Path(raw_path).expanduser()
    candidates: list[Path] = [original]

    alias_roots = [
        (Path("/data/holosoma_moved/src/holosoma_retargeting"), retarget_root),
        (Path("/home/ubuntu/FAR/holosoma/src/holosoma_retargeting"), retarget_root),
        (Path("/data/holosoma_moved/src/holosoma"), holosoma_root),
        (Path("/home/ubuntu/FAR/holosoma/src/holosoma"), holosoma_root),
    ]
    for old_root, new_root in alias_roots:
        try:
            rel = original.relative_to(old_root)
        except Exception:
            continue
        candidates.append(new_root / rel)

    seen: set[str] = set()
    deduped: list[Path] = []
    for candidate in candidates:
        resolved_key = str(candidate)
        if resolved_key in seen:
            continue
        seen.add(resolved_key)
        deduped.append(candidate)

    for candidate in deduped:
        if candidate.exists():
            return str(candidate)

    stem_match = re.match(r"^(?P<prefix>.+_)[0-9a-f]{8,}$", original.name)
    if stem_match:
        prefix = stem_match.group("prefix")
        for parent in deduped:
            parent_dir = parent.parent
            if not parent_dir.is_dir():
                continue
            matches = sorted(p for p in parent_dir.glob(f"{prefix}*") if p.exists())
            if len(matches) == 1:
                return str(matches[0])

    return None


try:
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = load_checkpoint(checkpoint_ref, temp_dir)
        blob = torch.load(checkpoint_path, map_location="cpu")
except Exception:
    print(json.dumps({}))
    sys.exit(0)

experiment_config = blob.get("experiment_config", {})
motion_cfg = (
    experiment_config.get("command", {})
    .get("setup_terms", {})
    .get("motion_command", {})
    .get("params", {})
    .get("motion_config", {})
)
robot_cfg = experiment_config.get("robot", {}).get("object", {})

motion_path = motion_cfg.get("motion_dir") or motion_cfg.get("motion_file")
object_urdf_path = robot_cfg.get("object_urdf_path")

print(
    json.dumps(
        {
            "motion_path": resolve_saved_path(motion_path),
            "saved_motion_path": motion_path,
            "object_urdf_path": resolve_saved_path(object_urdf_path),
            "saved_object_urdf_path": object_urdf_path,
        }
    )
)
PY
}

CKPT="${CHECKPOINT:-${CKPT:-}}"
if [[ $# -gt 0 ]]; then
  if is_checkpoint_arg "$1"; then
    CKPT="$1"
    shift
  fi
fi

if [[ -z "${CKPT}" ]]; then
  if [[ "${MODE}" == "mixed" ]]; then
    CKPT="${DEFAULT_MIXED_CHECKPOINT}"
  else
    CKPT="${DEFAULT_CLIP_CHECKPOINT}"
  fi
fi

if [[ "${CKPT}" == https://wandb.ai/*/runs/* ]]; then
  LOCAL_WANDB_CKPT="$(resolve_local_checkpoint_from_run_url "${CKPT}" "${WANDB_MODEL_FILE:-}")"
  if [[ -n "${LOCAL_WANDB_CKPT}" && -f "${LOCAL_WANDB_CKPT}" ]]; then
    CKPT="${LOCAL_WANDB_CKPT}"
    echo "[INFO] Resolved wandb run URL to local checkpoint: ${CKPT}"
  else
    CKPT="$(normalize_checkpoint_ref "${CKPT}")"
  fi
fi

if [[ "${CKPT}" != wandb://* ]] && [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] checkpoint not found: ${CKPT}" >&2
  exit 1
fi

ROBOT_INIT_STATE_POS_ARG="${ROBOT_INIT_STATE_POS:-}"
ROBOT_INIT_STATE_XY_OFFSET_RAW="${ROBOT_INIT_STATE_XY_OFFSET:-}"
ROBOT_INIT_STATE_XY_OFFSET_ARG=""
if [[ -n "${ROBOT_INIT_STATE_POS_ARG}" && -n "${ROBOT_INIT_STATE_XY_OFFSET_RAW}" ]]; then
  echo "[ERROR] Set only one of ROBOT_INIT_STATE_POS or ROBOT_INIT_STATE_XY_OFFSET." >&2
  exit 2
fi
if [[ -n "${ROBOT_INIT_STATE_XY_OFFSET_RAW}" ]]; then
  ROBOT_INIT_STATE_XY_OFFSET_ARG="$(normalize_xy_offset "${ROBOT_INIT_STATE_XY_OFFSET_RAW}")"
  DEFAULT_POSE_INIT_ENABLED=0
  if is_truthy "${HOLOSOMA_RESET_TO_DEFAULT_POSE:-}"; then
    DEFAULT_POSE_INIT_ENABLED=1
  elif is_truthy "${HOLOSOMA_DEFAULT_POSE_INIT:-}"; then
    DEFAULT_POSE_INIT_ENABLED=1
  elif [[ "$(echo "${SIM_MOTION_INIT_MODE:-}" | tr '[:upper:]' '[:lower:]')" == "training_default_pose" ]]; then
    DEFAULT_POSE_INIT_ENABLED=1
  fi

  if [[ "${DEFAULT_POSE_INIT_ENABLED}" == "1" ]]; then
    ROBOT_INIT_STATE_POS_ARG="$(
      resolve_robot_init_state_pos_with_xy_offset "${CKPT}" "${ROBOT_INIT_STATE_XY_OFFSET_ARG}"
    )"
  else
    echo "[INFO] Ignoring ROBOT_INIT_STATE_XY_OFFSET because default-pose init is not enabled at launch." >&2
  fi
fi

MIXED_PROFILE=${MIXED_PROFILE:-auto}
INFER_DATASET_EXPLICIT=0
[[ -n "${INFER_DATASET+x}" ]] && INFER_DATASET_EXPLICIT=1
MOTION_DIR_EXPLICIT=0
[[ -n "${MOTION_DIR+x}" ]] && MOTION_DIR_EXPLICIT=1
OBJECT_URDF_EXPLICIT=0
[[ -n "${OBJECT_URDF+x}" ]] && OBJECT_URDF_EXPLICIT=1
PAIR_TERRAIN_WITH_MOTION_EXPLICIT=0
[[ -n "${PAIR_TERRAIN_WITH_MOTION+x}" ]] && PAIR_TERRAIN_WITH_MOTION_EXPLICIT=1
START_AT_TIMESTEP_ZERO_PROB_EXPLICIT=0
[[ -n "${START_AT_TIMESTEP_ZERO_PROB+x}" ]] && START_AT_TIMESTEP_ZERO_PROB_EXPLICIT=1
FREEZE_AT_TIMESTEP_ZERO_PROB_EXPLICIT=0
[[ -n "${FREEZE_AT_TIMESTEP_ZERO_PROB+x}" ]] && FREEZE_AT_TIMESTEP_ZERO_PROB_EXPLICIT=1
RESET_NOISE_SCALE_EXPLICIT=0
[[ -n "${RESET_NOISE_SCALE+x}" ]] && RESET_NOISE_SCALE_EXPLICIT=1
MAX_EPISODE_LENGTH_S_EXPLICIT=0
[[ -n "${MAX_EPISODE_LENGTH_S+x}" ]] && MAX_EPISODE_LENGTH_S_EXPLICIT=1
DEPTH_PERCEPTION_PRESET_EXPLICIT=0
[[ -n "${DEPTH_PERCEPTION_PRESET+x}" ]] && DEPTH_PERCEPTION_PRESET_EXPLICIT=1
EVAL_COMMAND_ONLY_ENV_PROB_EXPLICIT=0
[[ -n "${EVAL_COMMAND_ONLY_ENV_PROB+x}" ]] && EVAL_COMMAND_ONLY_ENV_PROB_EXPLICIT=1
EVAL_EXTERNAL_GOAL_PROB_EXPLICIT=0
[[ -n "${EVAL_EXTERNAL_GOAL_PROB+x}" ]] && EVAL_EXTERNAL_GOAL_PROB_EXPLICIT=1
CHECKPOINT_SAVED_MOTION_PATH=""
CHECKPOINT_SAVED_OBJECT_URDF=""

if [[ "${MOTION_DIR_EXPLICIT}" -eq 0 || "${OBJECT_URDF_EXPLICIT}" -eq 0 ]]; then
  CHECKPOINT_DEFAULTS_JSON="$(load_checkpoint_saved_motion_defaults "${CKPT}")"
  if [[ -n "${CHECKPOINT_DEFAULTS_JSON}" && "${CHECKPOINT_DEFAULTS_JSON}" != "{}" ]]; then
    while IFS='=' read -r key value; do
      case "${key}" in
        motion_path) CHECKPOINT_SAVED_MOTION_PATH="${value}" ;;
        object_urdf_path) CHECKPOINT_SAVED_OBJECT_URDF="${value}" ;;
      esac
    done < <(
      CHECKPOINT_DEFAULTS_JSON="${CHECKPOINT_DEFAULTS_JSON}" "$PYTHON_BIN" - <<'PY'
import json
import os

payload = json.loads(os.environ["CHECKPOINT_DEFAULTS_JSON"])
for key in ("motion_path", "object_urdf_path"):
    value = payload.get(key) or ""
    print(f"{key}={value}")
PY
    )
  fi
fi

MIXED_PROFILE_RESOLVED="none"
if [[ "${MODE}" == "mixed" ]]; then
  if [[ "${MIXED_PROFILE}" == "auto" ]]; then
    if [[ "${CKPT}" == *"s221l5eo"* ]]; then
      MIXED_PROFILE_RESOLVED="s221l5eo"
    else
      MIXED_PROFILE_RESOLVED="none"
    fi
  else
    MIXED_PROFILE_RESOLVED="${MIXED_PROFILE}"
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

DEFAULT_INFER_DATASET="omomo"
if [[ "${MODE}" == "mixed" ]]; then
  DEFAULT_INFER_DATASET="mixed"
fi
INFER_DATASET=${INFER_DATASET:-${DEFAULT_INFER_DATASET}}
INFER_DATASET=$(echo "${INFER_DATASET}" | tr '[:upper:]' '[:lower:]' | tr -d '[][:space:]')
case "${INFER_DATASET}" in
  omomo|behave|mixed) ;;
  *)
    echo "[ERROR] INFER_DATASET must be one of: omomo|behave|mixed. Got: ${INFER_DATASET}" >&2
    exit 2
    ;;
esac

DS_DATA_ROOT="${DS_DATA_ROOT:-${SCRIPT_DIR}/data/ds_box_data}"
DEFAULT_OMOMO_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"
DEFAULT_BEHAVE_MOTION_DIR="$(pick_first_existing_path \
  "${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/behave_carry" \
  "${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/behave_sq_carry")"
DEFAULT_MIXED_MOTION_DIR="$(pick_first_existing_path \
  "${DS_DATA_ROOT}/train_g1_w_obj_prepared_plus_omomo_orig" \
  "${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_carry_aug_mix_ml" \
  "${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_sq_carry_aug_mix_ml")"
DEFAULT_OMOMO_URDF="$(pick_first_existing_path \
  "${SCRIPT_DIR}/src/holosoma_retargeting/models/largebox/largebox.urdf" \
  "${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf")"
DEFAULT_BEHAVE_MAP_FILE="${DEFAULT_BEHAVE_MOTION_DIR}/_clip_object_urdf_map.json"
DEFAULT_MIXED_MAP_FILE="${DEFAULT_MIXED_MOTION_DIR}/_clip_object_urdf_map.json"

if [[ "${MOTION_DIR_EXPLICIT}" -eq 0 && -n "${CHECKPOINT_SAVED_MOTION_PATH}" && "${INFER_DATASET_EXPLICIT}" -eq 0 ]]; then
  MOTION_DIR="${CHECKPOINT_SAVED_MOTION_PATH}"
fi

if [[ "${OBJECT_URDF_EXPLICIT}" -eq 0 && -n "${CHECKPOINT_SAVED_OBJECT_URDF}" && "${INFER_DATASET_EXPLICIT}" -eq 0 ]]; then
  OBJECT_URDF="${CHECKPOINT_SAVED_OBJECT_URDF}"
fi

if [[ -z "${MOTION_DIR+x}" ]]; then
  case "${INFER_DATASET}" in
    omomo) MOTION_DIR="${DEFAULT_OMOMO_MOTION_DIR}" ;;
    behave) MOTION_DIR="${DEFAULT_BEHAVE_MOTION_DIR}" ;;
    mixed) MOTION_DIR="${DEFAULT_MIXED_MOTION_DIR}" ;;
  esac
fi

if [[ -z "${OBJECT_URDF+x}" ]]; then
  case "${INFER_DATASET}" in
    omomo) OBJECT_URDF="${DEFAULT_OMOMO_URDF}" ;;
    behave) OBJECT_URDF="${DEFAULT_BEHAVE_MAP_FILE}" ;;
    mixed) OBJECT_URDF="${DEFAULT_MIXED_MAP_FILE}" ;;
  esac
fi

MOTION_CLIP_NAME=${MOTION_CLIP_NAME:-}
MOTION_CLIP_ID=${MOTION_CLIP_ID:-}
GEOMETRY_DIR=${GEOMETRY_DIR:-}
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
NUM_ENVS=${NUM_ENVS:-1}
HEADLESS_RAW=${HEADLESS:-True}
VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_RECENTER=${VISER_RECENTER:-True}
VIS_GPU=${VIS_GPU:-auto}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-1.0}
FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}
RESET_NOISE_SCALE=${RESET_NOISE_SCALE:-0.0}
MAX_EPISODE_LENGTH_S=${MAX_EPISODE_LENGTH_S:-1000000}
PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-268435456}
DISABLE_RANDOMIZATION=${DISABLE_RANDOMIZATION:-True}
DEPTH_PERCEPTION_PRESET=${DEPTH_PERCEPTION_PRESET:-checkpoint}
IMAGE_WIDTH=${IMAGE_WIDTH:-17}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-17}
CAMERA_NEAR=${CAMERA_NEAR:-0.001}
CAMERA_FAR=${CAMERA_FAR:-3.0}
CAMERA_MAX_DISTANCE=${CAMERA_MAX_DISTANCE:-3.0}
MAX_EVAL_STEPS=${MAX_EVAL_STEPS:-}
EVAL_COMMAND_ONLY_ENV_PROB=${EVAL_COMMAND_ONLY_ENV_PROB:-}
EVAL_EXTERNAL_GOAL_PROB=${EVAL_EXTERNAL_GOAL_PROB:-}
DRY_RUN_RAW=${DRY_RUN:-0}

case "${MIXED_PROFILE_RESOLVED}" in
  none)
    ;;
  s221l5eo)
    if [[ "${PAIR_TERRAIN_WITH_MOTION_EXPLICIT}" -eq 0 ]]; then
      PAIR_TERRAIN_WITH_MOTION="False"
    fi
    if [[ "${START_AT_TIMESTEP_ZERO_PROB_EXPLICIT}" -eq 0 ]]; then
      START_AT_TIMESTEP_ZERO_PROB="1.0"
    fi
    if [[ "${FREEZE_AT_TIMESTEP_ZERO_PROB_EXPLICIT}" -eq 0 ]]; then
      FREEZE_AT_TIMESTEP_ZERO_PROB="0.0"
    fi
    if [[ "${RESET_NOISE_SCALE_EXPLICIT}" -eq 0 ]]; then
      RESET_NOISE_SCALE="1.0"
    fi
    if [[ "${DEPTH_PERCEPTION_PRESET_EXPLICIT}" -eq 0 ]]; then
      DEPTH_PERCEPTION_PRESET="checkpoint"
    fi
    if [[ "${EVAL_COMMAND_ONLY_ENV_PROB_EXPLICIT}" -eq 0 ]]; then
      EVAL_COMMAND_ONLY_ENV_PROB="1.0"
    fi
    if [[ "${EVAL_EXTERNAL_GOAL_PROB_EXPLICIT}" -eq 0 ]]; then
      EVAL_EXTERNAL_GOAL_PROB="1.0"
    fi
    ;;
  *)
    echo "[ERROR] MIXED_PROFILE must be one of: auto|none|s221l5eo. Got: ${MIXED_PROFILE_RESOLVED}" >&2
    exit 2
    ;;
esac

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

DRY_RUN_NORM=$(echo "${DRY_RUN_RAW}" | tr '[:upper:]' '[:lower:]')
case "${DRY_RUN_NORM}" in
  1|true|yes|on) DRY_RUN_FLAG=1 ;;
  0|false|no|off|"") DRY_RUN_FLAG=0 ;;
  *)
    echo "[ERROR] DRY_RUN must be one of: 0/1/true/false/yes/no/on/off. Got: ${DRY_RUN_RAW}" >&2
    exit 2
    ;;
esac

if [[ ! -e "${MOTION_DIR}" ]]; then
  echo "[ERROR] MOTION_DIR not found: ${MOTION_DIR}" >&2
  exit 1
fi
if [[ ! -f "${OBJECT_URDF}" ]]; then
  echo "[ERROR] OBJECT_URDF not found: ${OBJECT_URDF}" >&2
  exit 1
fi
if [[ -n "${GEOMETRY_DIR}" && ! -e "${GEOMETRY_DIR}" ]]; then
  echo "[ERROR] GEOMETRY_DIR not found: ${GEOMETRY_DIR}" >&2
  exit 1
fi
if [[ -n "${MOTION_CLIP_NAME}" && -d "${MOTION_DIR}" && ! -f "${MOTION_DIR}/${MOTION_CLIP_NAME}.npz" ]]; then
  echo "[ERROR] MOTION_CLIP_NAME not found in MOTION_DIR: ${MOTION_CLIP_NAME}.npz" >&2
  exit 1
fi

normalize_object_scale() {
  local raw="$1"
  local compact="${raw//[\[\]\(\)[:space:]]/}"
  local values=()
  if [[ -z "${compact}" ]]; then
    echo ""
    return 0
  fi
  IFS=',' read -r -a values <<< "${compact}"
  if [[ "${#values[@]}" -eq 1 ]]; then
    printf '[%s,%s,%s]' "${values[0]}" "${values[0]}" "${values[0]}"
    return 0
  fi
  if [[ "${#values[@]}" -eq 3 ]]; then
    printf '[%s,%s,%s]' "${values[0]}" "${values[1]}" "${values[2]}"
    return 0
  fi
  echo "[ERROR] OBJECT_SCALE must be a scalar or comma-separated x,y,z triple. Got: ${raw}" >&2
  exit 2
}

OBJECT_SCALE_ARG=""
if [[ -n "${OBJECT_SCALE+x}" && -n "${OBJECT_SCALE}" ]]; then
  OBJECT_SCALE_ARG="$(normalize_object_scale "${OBJECT_SCALE}")"
fi

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

export VISER_ENABLE_CLIP_GUI=${VISER_ENABLE_CLIP_GUI:-1}
export VISER_ENABLE_MANUAL_GUI=${VISER_ENABLE_MANUAL_GUI:-1}
export VISER_ENABLE_MANUAL_GOAL_GUI=${VISER_ENABLE_MANUAL_GOAL_GUI:-1}
export VISER_SHOW_TARGET_KEYPOINTS=${VISER_SHOW_TARGET_KEYPOINTS:-1}
export VISER_SHOW_TARGET_BOX=${VISER_SHOW_TARGET_BOX:-1}
export VISER_PERCEPTION_IMAGE_MODE=${VISER_PERCEPTION_IMAGE_MODE:-depth}
export VISER_SHOW_PERCEPTION_FRUSTUM=${VISER_SHOW_PERCEPTION_FRUSTUM:-1}
export HOLOSOMA_DISABLE_BAD_TRACKING_RESET=${HOLOSOMA_DISABLE_BAD_TRACKING_RESET:-1}
export HOLOSOMA_DISABLE_AUTO_RESET=${HOLOSOMA_DISABLE_AUTO_RESET:-1}
export LOGURU_LEVEL=${LOGURU_LEVEL:-WARNING}
export PY_LOG_LEVEL=${PY_LOG_LEVEL:-WARNING}

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
  "$PYTHON_BIN" -m holosoma.visualize physics
)

if [[ -n "${SIMULATOR_SUBCOMMAND}" ]]; then
  cmd+=("${SIMULATOR_SUBCOMMAND}")
fi

cmd+=(
  --checkpoint "${CKPT}"
  --motion-dir "${MOTION_DIR}"
  --num-envs "${NUM_ENVS}"
  --headless "${HEADLESS_FLAG}"
  --pair-terrain-with-motion "${PAIR_TERRAIN_WITH_MOTION}"
  --viser-port "${VISER_PORT}"
  --viser-env-id "${VISER_ENV_ID}"
  --viser-update-hz "${VISER_UPDATE_HZ}"
  --viser-recenter "${VISER_RECENTER}"
  --simulator.config.sim.max_episode_length_s "${MAX_EPISODE_LENGTH_S}"
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

if [[ "${SIMULATOR_SUBCOMMAND}" != "simulator:mujoco" ]]; then
  cmd+=(--simulator.config.sim.physx.gpu_collision_stack_size "${PHYSX_GPU_COLLISION_STACK_SIZE}")
else
  cmd+=(--randomization.ignore_unsupported True)
fi

if [[ -n "${MOTION_CLIP_NAME}" ]]; then
  cmd+=(--command.setup_terms.motion_command.params.motion_config.motion_clip_name "${MOTION_CLIP_NAME}")
fi
if [[ -n "${MOTION_CLIP_ID}" ]]; then
  cmd+=(--command.setup_terms.motion_command.params.motion_config.motion_clip_id "${MOTION_CLIP_ID}")
fi
if [[ -n "${OBJECT_SCALE_ARG}" ]]; then
  cmd+=(--robot.object.scale "${OBJECT_SCALE_ARG}")
fi
if [[ -n "${GEOMETRY_DIR}" ]]; then
  cmd+=(--geometry-dir "${GEOMETRY_DIR}")
fi
if [[ -n "${ROBOT_INIT_STATE_POS_ARG}" ]]; then
  cmd+=(--robot.init-state.pos "${ROBOT_INIT_STATE_POS_ARG}")
fi
if [[ -n "${MAX_EVAL_STEPS}" ]]; then
  cmd+=(--training.max_eval_steps "${MAX_EVAL_STEPS}")
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

case "$(echo "${DEPTH_PERCEPTION_PRESET}" | tr '[:upper:]' '[:lower:]')" in
  checkpoint|auto|"")
    ;;
  d435i_17x17|d435i)
    cmd+=(
      perception:camera_depth_d435i_17x17
      --perception.camera_width "${IMAGE_WIDTH}"
      --perception.camera_height "${IMAGE_HEIGHT}"
      --perception.camera_near "${CAMERA_NEAR}"
      --perception.camera_far "${CAMERA_FAR}"
      --perception.max_distance "${CAMERA_MAX_DISTANCE}"
    )
    ;;
  *)
    echo "[ERROR] DEPTH_PERCEPTION_PRESET must be one of: checkpoint|d435i_17x17. Got: ${DEPTH_PERCEPTION_PRESET}" >&2
    exit 2
    ;;
esac

if [[ "${MODE}" == "mixed" ]]; then
  if [[ -z "${EVAL_EXTERNAL_GOAL_PROB}" ]]; then
    EVAL_EXTERNAL_GOAL_PROB="1.0"
  fi
  cmd+=(
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.enabled True
  )
  if [[ -n "${EVAL_COMMAND_ONLY_ENV_PROB}" ]]; then
    cmd+=(
      --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.eval_command_only_env_prob "${EVAL_COMMAND_ONLY_ENV_PROB}"
    )
  fi
  if [[ -n "${EVAL_EXTERNAL_GOAL_PROB}" ]]; then
    cmd+=(
      --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.eval_external_goal_prob "${EVAL_EXTERNAL_GOAL_PROB}"
    )
  fi
fi

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] mode_input=${MODE_INPUT} runtime_mode=${MODE}"
if [[ "${MODE}" == "mixed" ]]; then
  echo "[INFO] mixed_profile=${MIXED_PROFILE_RESOLVED}"
fi
echo "[INFO] checkpoint=${CKPT}"
echo "[INFO] infer_dataset=${INFER_DATASET}"
echo "[INFO] motion_dir=${MOTION_DIR}"
echo "[INFO] object_urdf=${OBJECT_URDF}"
if [[ -n "${CHECKPOINT_SAVED_MOTION_PATH}" ]]; then
  echo "[INFO] checkpoint_saved_motion_path=${CHECKPOINT_SAVED_MOTION_PATH}"
fi
if [[ -n "${CHECKPOINT_SAVED_OBJECT_URDF}" ]]; then
  echo "[INFO] checkpoint_saved_object_urdf=${CHECKPOINT_SAVED_OBJECT_URDF}"
fi
echo "[INFO] preserving checkpoint actor/critic observation history"
echo "[INFO] headless=${HEADLESS_FLAG} (env HEADLESS=${HEADLESS})"
echo "[INFO] viser=http://localhost:${VISER_PORT}"
echo "[INFO] clip_gui=${VISER_ENABLE_CLIP_GUI} manual_gui=${VISER_ENABLE_MANUAL_GUI} manual_goal_gui=${VISER_ENABLE_MANUAL_GOAL_GUI}"
if [[ -n "${MOTION_CLIP_NAME}" ]]; then
  echo "[INFO] motion_clip_name=${MOTION_CLIP_NAME}"
fi
if [[ -n "${MOTION_CLIP_ID}" ]]; then
  echo "[INFO] motion_clip_id=${MOTION_CLIP_ID}"
fi
if [[ -n "${OBJECT_SCALE_ARG}" ]]; then
  echo "[INFO] object_scale=${OBJECT_SCALE_ARG}"
fi
if [[ -n "${GEOMETRY_DIR}" ]]; then
  echo "[INFO] geometry_dir=${GEOMETRY_DIR}"
fi
if [[ -n "${ROBOT_INIT_STATE_POS_ARG}" ]]; then
  echo "[INFO] robot_init_state_pos=${ROBOT_INIT_STATE_POS_ARG}"
fi
if [[ -n "${ROBOT_INIT_STATE_XY_OFFSET_ARG}" ]]; then
  echo "[INFO] robot_init_state_xy_offset=${ROBOT_INIT_STATE_XY_OFFSET_ARG}"
fi
if [[ "${MODE}" == "mixed" ]]; then
  echo "[INFO] eval_command_only_env_prob=${EVAL_COMMAND_ONLY_ENV_PROB:-<checkpoint>}"
  echo "[INFO] eval_external_goal_prob=${EVAL_EXTERNAL_GOAL_PROB}"
fi
if command -v hostname >/dev/null 2>&1; then
  HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  if [[ -n "${HOST_IP}" ]]; then
    echo "[INFO] Remote URL: http://${HOST_IP}:${VISER_PORT}"
  fi
fi

if [[ "${DRY_RUN_FLAG}" == "1" ]]; then
  printf '[DRY_RUN] '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
