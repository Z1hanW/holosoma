#!/usr/bin/env bash
set -euo pipefail

# Unified IsaacSim + Viser interactive inference for distilled box-carry policies.
#
# Features:
# - Two branches: mocap | depth
# - Viser clip selection GUI
# - Viser manual root-command GUI (root-frame relative dx/dy/dyaw)
# - Optional hardware joystick via pygame/bridge backend
#
# Usage:
#   bash infer_box_joystick.sh <mocap|depth> [checkpoint.pt|wandb://...|https://wandb.ai/.../runs/.../files] [extra tyro args...]
#
# Examples:
#   bash infer_box_joystick.sh mocap /abs/path/model.pt
#   bash infer_box_joystick.sh depth /abs/path/model.pt --viser-port 18080
#   bash infer_box_joystick.sh mocap https://wandb.ai/zihanw22/WholeBodyTracking/runs/d20ktze6/files
# Boxer reference checkpoints:
#   https://wandb.ai/zihanw22/boxer/runs/wttqf2em/files/model_04000.pt -> wandb://zihanw22/boxer/wttqf2em/model_04000.pt
#   https://wandb.ai/zihanw22/boxer/runs/wttqf2em?nw=nwuserz1hanw -> wandb://zihanw22/boxer/wttqf2em/model_04000.pt
#   https://wandb.ai/zihanw22/boxer/runs/wttqf2em -> wandb://zihanw22/boxer/wttqf2em/model_04000.pt
#   https://wandb.ai/zihanw22/boxer/runs/0z2aggr2 -> wandb://zihanw22/boxer/0z2aggr2/model_00200.pt

usage() {
  cat <<'EOF'
Usage:
  bash infer_box_joystick.sh <mocap|depth> [checkpoint.pt|wandb://...|https://wandb.ai/.../runs/.../files] [extra tyro args...]

Modes:
  mocap   Distilled policy with box-state (mocap) actor observation
  depth   Distilled policy with camera depth perception (D435i)

Optional env vars:
  INFER_DATASET           (default: omomo; options: omomo|behave|mixed)
  MOTION_DIR              (optional override; default chosen by INFER_DATASET)
  OBJECT_URDF             (optional override; default chosen by INFER_DATASET)
  OBJECT_SCALE            (optional; scalar or x,y,z spawn scale for current object URDF)
  OBJECT_GEOMETRY_MODE    (optional; `on`/`primitive` forces cuboid primitive path,
                           `off`/`mesh` forces legacy URDF/mesh path)
  GEOMETRY_DIR            (optional; OBJ file/dir for terrain visualization)
  PAIR_TERRAIN_WITH_MOTION (default: False)
  NUM_ENVS                (default: 1)
  HEADLESS                (default: True; accepts 0/1/true/false/yes/no/on/off)
  VISER_PORT              (default: random)
  VISER_ENV_ID            (default: 0)
  VISER_UPDATE_HZ         (default: 30)
  VISER_RECENTER          (default: True)
  WANDB_MODEL_FILE        (default varies by mode; used when checkpoint is a wandb run URL)
  MOCAP_PERCEPTION_PRESET (default: checkpoint; checkpoint|none|heightmap)
  DEPTH_PERCEPTION_PRESET (default: checkpoint; checkpoint|d435i)
  CAMERA_*                (optional explicit camera overrides; default preserves checkpoint camera config)
  CAMERA_PITCH_DEG        (optional explicit camera pitch override; default preserves checkpoint camera pitch)
  DISTILL_PROPRIO_HISTORY_ONLY (default: 1; keep 5-frame history only on proprio groups, keep actions single-frame)
  DISTILL_PROPRIO_HISTORY_LENGTH (default: 5)
  VISER_ENABLE_EXTERNAL_SPARSE_GOAL
                          (default: 0; set 1 to enable the same sparse-object-goal
                           command path used by box-drop distillation and draw that target)
  EVAL_EXTERNAL_GOAL_PROB (default with VISER_ENABLE_EXTERNAL_SPARSE_GOAL=1: 1.0)
  EVAL_COMMAND_ONLY_ENV_PROB
                          (default with VISER_ENABLE_EXTERNAL_SPARSE_GOAL=1: 1.0)
  EXTERNAL_GOAL_SAMPLING_MODE
                          (default with VISER_ENABLE_EXTERNAL_SPARSE_GOAL=1: annulus)
  DRY_RUN                 (default: 0; set 1/true to print the command without launching)

Hardware joystick (optional):
  VISER_MANUAL_USE_HW_JOYSTICK=1
  VISER_MANUAL_HW_BACKEND=auto|pygame|bridge
  VISER_MANUAL_HW_DEVICE=0
  VISER_MANUAL_HW_TYPE=xbox|switch
  USE_HW_JOYSTICK_BRIDGE=True   # optional legacy bridge joystick path (requires real joystick)
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

MODE_INPUT="$1"
shift

case "${MODE_INPUT}" in
  mocap|depth) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  mixed_goal)
    echo "[ERROR] mixed_goal inference has been removed. Use mode=depth or mode=mocap." >&2
    exit 1
    ;;
  *)
    echo "[ERROR] mode must be one of: mocap|depth. Got: ${MODE_INPUT}" >&2
    exit 1
    ;;
esac

MODE="${MODE_INPUT}"

LOG_ROOT="/data/logs_new/WholeBodyTracking"
MOCAP_TRAINING_NAME_DEFAULT="g1_29dof_wbt_w_object_distill_box_mocap_access_to_mocap_data"
DEPTH_TRAINING_NAME_DEFAULT="g1_29dof_wbt_w_object_distill_box_perception_access_to_depth"
MOCAP_CHECKPOINT_DEFAULT=${MOCAP_CHECKPOINT_DEFAULT:-"wandb://zihanw22/boxer/d20ktze6/model_00800.pt"}
# DEPTH_CHECKPOINT_DEFAULT=${DEPTH_CHECKPOINT_DEFAULT:-"wandb://zihanw22/boxer/7fxeaecw/model_03000.pt"}
DEPTH_CHECKPOINT_DEFAULT=${DEPTH_CHECKPOINT_DEFAULT:-"wandb://zihanw22/boxer/0z2aggr2/model_05000.pt"}
if [[ -n "${WANDB_MODEL_FILE+x}" && -n "${WANDB_MODEL_FILE}" ]]; then
  WANDB_MODEL_FILE_FROM_ENV=1
else
  WANDB_MODEL_FILE_FROM_ENV=0
  if [[ "${MODE}" == "depth" ]]; then
    WANDB_MODEL_FILE="model_01000.pt"
  else
    WANDB_MODEL_FILE="model_00800.pt"
  fi
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

extract_wandb_run_id_from_url() {
  local ref="$1"
  local parsed=""
  parsed="$(parse_wandb_run_url "${ref}" || true)"
  if [[ -n "${parsed}" ]]; then
    IFS=$'\t' read -r _entity _project run_id _explicit_file <<< "${parsed}"
    echo "${run_id}"
    return 0
  fi
  echo ""
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
  if [[ "${ref}" == https://wandb.ai/*/runs/* ]]; then
    local parsed=""
    local entity=""
    local project=""
    local run_id=""
    local explicit_file=""
    local model_file="${WANDB_MODEL_FILE}"

    parsed="$(parse_wandb_run_url "${ref}" || true)"
    if [[ -n "${parsed}" ]]; then
      IFS=$'\t' read -r entity project run_id explicit_file <<< "${parsed}"
      if [[ -n "${explicit_file}" ]]; then
        model_file="${explicit_file}"
      elif [[ "${WANDB_MODEL_FILE_FROM_ENV}" != "1" ]]; then
        local remote_model_file=""
        remote_model_file="$(resolve_remote_wandb_checkpoint_name "${entity}" "${project}" "${run_id}")"
        if [[ -n "${remote_model_file}" ]]; then
          model_file="${remote_model_file}"
          echo "[INFO] Resolved wandb run URL to latest remote checkpoint: ${model_file}" >&2
        fi
      fi
      if [[ -n "${entity}" && -n "${project}" && -n "${run_id}" && -n "${model_file}" ]]; then
        ref="wandb://${entity}/${project}/${run_id}/${model_file}"
      fi
    fi
  fi
  echo "${ref}"
}

resolve_local_checkpoint_from_run_url() {
  local ref="$1"
  local parsed=""
  local run_id=""
  local explicit_file=""
  local wandb_run_dir=""
  local run_log_dir=""
  local local_ckpt=""

  parsed="$(parse_wandb_run_url "${ref}" || true)"
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
  else
    local_ckpt=$(ls -1 "${run_log_dir}"/model_*.pt 2>/dev/null | sort -V | tail -n 1 || true)
  fi
  echo "${local_ckpt}"
}

find_latest_ckpt() {
  local root="$1"
  local training_name="$2"
  local latest_run=""
  local latest_ckpt=""

  latest_run=$(ls -dt "${root}"/*-"${training_name}"* 2>/dev/null | head -n 1 || true)
  if [[ -z "${latest_run}" ]]; then
    echo ""
    return 0
  fi

  latest_ckpt=$(ls -1 "${latest_run}"/model_*.pt 2>/dev/null | sort -V | tail -n 1 || true)
  echo "${latest_ckpt}"
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
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
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
observation_overrides = experiment_config.get("observation_overrides", {})
if not isinstance(observation_overrides, dict):
    observation_overrides = {}

motion_path = motion_cfg.get("motion_dir") or motion_cfg.get("motion_file")
object_urdf_path = robot_cfg.get("object_urdf_path")

print(
    json.dumps(
        {
            "motion_path": resolve_saved_path(motion_path),
            "saved_motion_path": motion_path,
            "object_urdf_path": resolve_saved_path(object_urdf_path),
            "saved_object_urdf_path": object_urdf_path,
            "distill_proprio_history_only": observation_overrides.get("distill_proprio_history_only", False),
            "distill_proprio_history_length": observation_overrides.get("distill_proprio_history_length", 5),
        }
    )
)
PY
}

augment_object_map_from_motion_metadata() {
  local motion_dir="$1"
  local object_spec_path="$2"
  "$PYTHON_BIN" - "${motion_dir}" "${object_spec_path}" <<'PY' 2>/dev/null || true
import hashlib
import json
import sys
import zipfile
from pathlib import Path

try:
    import numpy as np
except Exception:
    print(sys.argv[2])
    sys.exit(0)

motion_dir = Path(sys.argv[1]).resolve()
object_spec_path = Path(sys.argv[2]).resolve()
if object_spec_path.suffix.lower() != ".json" or not motion_dir.is_dir() or not object_spec_path.is_file():
    print(str(object_spec_path))
    sys.exit(0)

try:
    payload = json.loads(object_spec_path.read_text(encoding="utf-8"))
except Exception:
    print(str(object_spec_path))
    sys.exit(0)

if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
    clips = payload["clips"]
else:
    clips = payload
if not isinstance(clips, dict):
    print(str(object_spec_path))
    sys.exit(0)

retarget_roots = [
    motion_dir.parents[2] / "src" / "holosoma_retargeting",
    Path("/home/ubuntu/FAR/holosoma/src/holosoma_retargeting"),
]

def scalar_str(value) -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    item = arr.item() if arr.shape == () else arr.reshape(-1)[0]
    if isinstance(item, (bytes, np.bytes_)):
        return item.decode("utf-8")
    return str(item)

def resolve_urdf(raw: str, *, base_dir: Path) -> str:
    raw = str(raw).strip()
    if not raw:
        return ""
    candidate = Path(raw)
    if candidate.is_absolute():
        return str(candidate)
    resolved = (base_dir / raw).resolve()
    if resolved.exists():
        return str(resolved)
    for root in retarget_roots:
        fallback = (root / raw).resolve()
        if fallback.exists():
            return str(fallback)
    return str(resolved)

changed = False
normalized_clips: dict[str, dict[str, str]] = {}

for clip_id, entry in clips.items():
    if not isinstance(clip_id, str):
        continue
    if isinstance(entry, str):
        normalized_clips[clip_id] = {"object_name": "", "object_urdf_path": entry.strip()}
    elif isinstance(entry, dict):
        normalized_clips[clip_id] = {
            "object_name": str(entry.get("object_name", "")).strip(),
            "object_urdf_path": str(entry.get("object_urdf_path", "")).strip(),
        }
    else:
        normalized_clips[clip_id] = {"object_name": "", "object_urdf_path": ""}

for clip_path in sorted(motion_dir.glob("*.npz")):
    if not zipfile.is_zipfile(clip_path):
        continue
    clip_id = clip_path.stem
    entry = normalized_clips.get(clip_id, {"object_name": "", "object_urdf_path": ""})
    try:
        with np.load(clip_path, allow_pickle=True) as data:
            object_name = scalar_str(data["object_name"]) if "object_name" in data else ""
            object_urdf_path = scalar_str(data["object_urdf_path"]) if "object_urdf_path" in data else ""
    except Exception:
        continue
    if object_urdf_path:
        object_urdf_path = resolve_urdf(object_urdf_path, base_dir=clip_path.parent)
    if not entry.get("object_name") and object_name:
        entry["object_name"] = object_name
        changed = True
    if not entry.get("object_urdf_path") and object_urdf_path:
        entry["object_urdf_path"] = object_urdf_path
        changed = True
    if clip_id not in normalized_clips and (object_name or object_urdf_path):
        normalized_clips[clip_id] = entry
        changed = True
    else:
        normalized_clips[clip_id] = entry

if not changed:
    print(str(object_spec_path))
    sys.exit(0)

out_dir = Path("/tmp/holosoma_object_maps")
out_dir.mkdir(parents=True, exist_ok=True)
digest = hashlib.sha1(f"{motion_dir}|{object_spec_path}".encode("utf-8")).hexdigest()[:12]
out_path = out_dir / f"{object_spec_path.stem}_{digest}.json"
out_path.write_text(json.dumps({"clips": normalized_clips}, ensure_ascii=True, sort_keys=True), encoding="utf-8")
print(str(out_path))
PY
}

CKPT=""
if [[ $# -gt 0 ]]; then
  if [[ "$1" == wandb://* || "$1" == https://wandb.ai/* || "$1" == /* || "$1" == ./* || "$1" == ../* || "$1" == *.pt ]]; then
    CKPT="$1"
    shift
  fi
fi

if [[ -n "${CKPT}" ]]; then
  if [[ "${CKPT}" == https://wandb.ai/*/runs/* ]]; then
    LOCAL_WANDB_CKPT="$(resolve_local_checkpoint_from_run_url "${CKPT}")"
    if [[ -n "${LOCAL_WANDB_CKPT}" && -f "${LOCAL_WANDB_CKPT}" ]]; then
      CKPT="${LOCAL_WANDB_CKPT}"
      echo "[INFO] Resolved wandb run URL to local checkpoint: ${CKPT}"
    fi
  fi
  CKPT="$(normalize_checkpoint_ref "${CKPT}")"
fi

if [[ -z "${CKPT}" ]]; then
  if [[ "${MODE}" == "mocap" ]]; then
    if [[ "${MOCAP_CHECKPOINT_DEFAULT}" == wandb://* ]] || [[ -f "${MOCAP_CHECKPOINT_DEFAULT}" ]]; then
      CKPT="${MOCAP_CHECKPOINT_DEFAULT}"
    else
      CKPT="$(find_latest_ckpt "${LOG_ROOT}" "${MOCAP_TRAINING_NAME_DEFAULT}")"
    fi
  else
    if [[ "${DEPTH_CHECKPOINT_DEFAULT}" == wandb://* ]] || [[ -f "${DEPTH_CHECKPOINT_DEFAULT}" ]]; then
      CKPT="${DEPTH_CHECKPOINT_DEFAULT}"
    else
      CKPT="$(find_latest_ckpt "${LOG_ROOT}" "${DEPTH_TRAINING_NAME_DEFAULT}")"
    fi
  fi
fi

if [[ -z "${CKPT}" ]]; then
  echo "[ERROR] Could not auto-resolve checkpoint. Please pass checkpoint path explicitly." >&2
  exit 1
fi
if [[ "${CKPT}" != wandb://* ]] && [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] checkpoint not found: ${CKPT}" >&2
  exit 1
fi

INFER_DATASET_EXPLICIT=0
[[ -n "${INFER_DATASET+x}" ]] && INFER_DATASET_EXPLICIT=1
MOTION_DIR_EXPLICIT=0
[[ -n "${MOTION_DIR+x}" ]] && MOTION_DIR_EXPLICIT=1
OBJECT_URDF_EXPLICIT=0
[[ -n "${OBJECT_URDF+x}" ]] && OBJECT_URDF_EXPLICIT=1
DISTILL_PROPRIO_HISTORY_ONLY_EXPLICIT=0
[[ -n "${DISTILL_PROPRIO_HISTORY_ONLY+x}" ]] && DISTILL_PROPRIO_HISTORY_ONLY_EXPLICIT=1
DISTILL_PROPRIO_HISTORY_LENGTH_EXPLICIT=0
[[ -n "${DISTILL_PROPRIO_HISTORY_LENGTH+x}" ]] && DISTILL_PROPRIO_HISTORY_LENGTH_EXPLICIT=1
HETEROGENEOUS_OBJECT_SINGLE_SLOT_DISABLE_EXPLICIT=0
[[ -n "${HOLOSOMA_DISABLE_HETEROGENEOUS_OBJECT_SINGLE_SLOT+x}" ]] && HETEROGENEOUS_OBJECT_SINGLE_SLOT_DISABLE_EXPLICIT=1
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
CAMERA_PITCH_DEG_EXPLICIT=0
[[ -n "${CAMERA_PITCH_DEG+x}" ]] && CAMERA_PITCH_DEG_EXPLICIT=1
CHECKPOINT_SAVED_MOTION_PATH=""
CHECKPOINT_SAVED_OBJECT_URDF=""
CHECKPOINT_SAVED_DISTILL_PROPRIO_HISTORY_ONLY=""
CHECKPOINT_SAVED_DISTILL_PROPRIO_HISTORY_LENGTH=""

if [[ "${MOTION_DIR_EXPLICIT}" -eq 0 || "${OBJECT_URDF_EXPLICIT}" -eq 0 || "${DISTILL_PROPRIO_HISTORY_ONLY_EXPLICIT}" -eq 0 || "${DISTILL_PROPRIO_HISTORY_LENGTH_EXPLICIT}" -eq 0 ]]; then
  CHECKPOINT_DEFAULTS_JSON="$(load_checkpoint_saved_motion_defaults "${CKPT}")"
  if [[ -n "${CHECKPOINT_DEFAULTS_JSON}" && "${CHECKPOINT_DEFAULTS_JSON}" != "{}" ]]; then
    while IFS='=' read -r key value; do
      case "${key}" in
        motion_path) CHECKPOINT_SAVED_MOTION_PATH="${value}" ;;
        object_urdf_path) CHECKPOINT_SAVED_OBJECT_URDF="${value}" ;;
        distill_proprio_history_only) CHECKPOINT_SAVED_DISTILL_PROPRIO_HISTORY_ONLY="${value}" ;;
        distill_proprio_history_length) CHECKPOINT_SAVED_DISTILL_PROPRIO_HISTORY_LENGTH="${value}" ;;
      esac
    done < <(
      CHECKPOINT_DEFAULTS_JSON="${CHECKPOINT_DEFAULTS_JSON}" "$PYTHON_BIN" - <<'PY'
import json
import os

payload = json.loads(os.environ["CHECKPOINT_DEFAULTS_JSON"])
for key in (
    "motion_path",
    "object_urdf_path",
    "distill_proprio_history_only",
    "distill_proprio_history_length",
):
    value = payload.get(key)
    if value is None:
        value = ""
    print(f"{key}={value}")
PY
    )
  fi
fi

INFER_DATASET_DEFAULT="omomo"
INFER_DATASET=${INFER_DATASET:-${INFER_DATASET_DEFAULT}}
INFER_DATASET=$(echo "${INFER_DATASET}" | tr '[:upper:]' '[:lower:]' | tr -d '[][:space:]')
case "${INFER_DATASET}" in
  omomo|behave|mixed) ;;
  *)
    echo "[ERROR] INFER_DATASET must be one of: omomo|behave|mixed. Got: ${INFER_DATASET}" >&2
    exit 2
    ;;
esac

DEFAULT_OMOMO_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"
DEFAULT_BEHAVE_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/behave_sq_carry"
DEFAULT_MIXED_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_sq_carry_aug_mix_ml"
DEFAULT_OMOMO_URDF="${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"
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

AUGMENTED_OBJECT_URDF_PATH=""
if [[ "${OBJECT_URDF_EXPLICIT}" -eq 0 ]]; then
  AUGMENTED_OBJECT_URDF_PATH="$(augment_object_map_from_motion_metadata "${MOTION_DIR}" "${OBJECT_URDF}")"
  if [[ -n "${AUGMENTED_OBJECT_URDF_PATH}" ]]; then
    OBJECT_URDF="${AUGMENTED_OBJECT_URDF_PATH}"
  fi
fi

GEOMETRY_DIR=${GEOMETRY_DIR:-}

PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
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
VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_RECENTER=${VISER_RECENTER:-True}

START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-1.0}
FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}
RESET_NOISE_SCALE=${RESET_NOISE_SCALE:-0.0}
PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-268435456}
FORCE_SINGLE_FRAME_HISTORY=${FORCE_SINGLE_FRAME_HISTORY:-0}
if [[ "${DISTILL_PROPRIO_HISTORY_ONLY_EXPLICIT}" -eq 0 && -n "${CHECKPOINT_SAVED_DISTILL_PROPRIO_HISTORY_ONLY}" ]]; then
  DISTILL_PROPRIO_HISTORY_ONLY="${CHECKPOINT_SAVED_DISTILL_PROPRIO_HISTORY_ONLY}"
else
  DISTILL_PROPRIO_HISTORY_ONLY=${DISTILL_PROPRIO_HISTORY_ONLY:-1}
fi
if [[ "${DISTILL_PROPRIO_HISTORY_LENGTH_EXPLICIT}" -eq 0 && -n "${CHECKPOINT_SAVED_DISTILL_PROPRIO_HISTORY_LENGTH}" ]]; then
  DISTILL_PROPRIO_HISTORY_LENGTH="${CHECKPOINT_SAVED_DISTILL_PROPRIO_HISTORY_LENGTH}"
else
  DISTILL_PROPRIO_HISTORY_LENGTH=${DISTILL_PROPRIO_HISTORY_LENGTH:-5}
fi

DISABLE_RANDOMIZATION=${DISABLE_RANDOMIZATION:-True}
MAX_EPISODE_LENGTH_S=${MAX_EPISODE_LENGTH_S:-1000000}

IMAGE_WIDTH=${IMAGE_WIDTH:-}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-}
CAMERA_NEAR=${CAMERA_NEAR:-}
CAMERA_FAR=${CAMERA_FAR:-}
CAMERA_MAX_DISTANCE=${CAMERA_MAX_DISTANCE:-}
CAMERA_PITCH_DEG=${CAMERA_PITCH_DEG:-}
MOCAP_PERCEPTION_PRESET=${MOCAP_PERCEPTION_PRESET:-checkpoint}
DEPTH_PERCEPTION_PRESET=${DEPTH_PERCEPTION_PRESET:-checkpoint}
OBJECT_GEOMETRY_MODE_RAW=${OBJECT_GEOMETRY_MODE:-}
OBJECT_GEOMETRY_MODE=""
HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE=""
PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE=""
DRY_RUN_RAW=${DRY_RUN:-0}
VISER_ENABLE_EXTERNAL_SPARSE_GOAL_RAW=${VISER_ENABLE_EXTERNAL_SPARSE_GOAL:-0}

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

DRY_RUN_NORM=$(echo "${DRY_RUN_RAW}" | tr '[:upper:]' '[:lower:]')
case "${DRY_RUN_NORM}" in
  1|true|yes|on) DRY_RUN_FLAG=1 ;;
  0|false|no|off|"") DRY_RUN_FLAG=0 ;;
  *)
    echo "[ERROR] DRY_RUN must be one of: 0/1/true/false/yes/no/on/off. Got: ${DRY_RUN_RAW}" >&2
    exit 2
    ;;
esac

VISER_ENABLE_EXTERNAL_SPARSE_GOAL_NORM=$(echo "${VISER_ENABLE_EXTERNAL_SPARSE_GOAL_RAW}" | tr '[:upper:]' '[:lower:]')
case "${VISER_ENABLE_EXTERNAL_SPARSE_GOAL_NORM}" in
  1|true|yes|on) VISER_EXTERNAL_SPARSE_GOAL_FLAG=1 ;;
  0|false|no|off|"") VISER_EXTERNAL_SPARSE_GOAL_FLAG=0 ;;
  *)
    echo "[ERROR] VISER_ENABLE_EXTERNAL_SPARSE_GOAL must be one of: 0/1/true/false/yes/no/on/off. Got: ${VISER_ENABLE_EXTERNAL_SPARSE_GOAL_RAW}" >&2
    exit 2
    ;;
esac

if [[ "${VISER_EXTERNAL_SPARSE_GOAL_FLAG}" -eq 1 ]]; then
  # Mirror distill_box_drop_mixed.sh defaults for the sparse object-goal command path.
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
  EXTERNAL_GOAL_PROB_END_ITER=${EXTERNAL_GOAL_PROB_END_ITER:-${NUM_LEARNING_ITERATIONS:-10000}}
  EXTERNAL_GOAL_PROB_RAMP_RESETS=${EXTERNAL_GOAL_PROB_RAMP_RESETS:-150000}
  EVAL_EXTERNAL_GOAL_PROB=${EVAL_EXTERNAL_GOAL_PROB:-1.0}
  EVAL_CARRY_EXTENSION_PROB=${EVAL_CARRY_EXTENSION_PROB:-0.0}
  EXTERNAL_GOAL_RANGE_RAMP_RESETS=${EXTERNAL_GOAL_RANGE_RAMP_RESETS:-${EXTERNAL_GOAL_PROB_RAMP_RESETS}}
  EXTERNAL_GOAL_RANGE_START_ITER=${EXTERNAL_GOAL_RANGE_START_ITER:-2500}
  EXTERNAL_GOAL_RANGE_END_ITER=${EXTERNAL_GOAL_RANGE_END_ITER:-${NUM_LEARNING_ITERATIONS:-10000}}
  EXTERNAL_GOAL_SAMPLING_MODE=${EXTERNAL_GOAL_SAMPLING_MODE:-annulus}
  EXTERNAL_GOAL_RADIUS_MIN_START=${EXTERNAL_GOAL_RADIUS_MIN_START:-1.00}
  EXTERNAL_GOAL_RADIUS_MAX_START=${EXTERNAL_GOAL_RADIUS_MAX_START:-1.70}
  EXTERNAL_GOAL_RADIUS_MIN=${EXTERNAL_GOAL_RADIUS_MIN:-1.00}
  EXTERNAL_GOAL_RADIUS_MAX=${EXTERNAL_GOAL_RADIUS_MAX:-3.40}
  EXTERNAL_GOAL_POS_LOCAL_MIN_START=${EXTERNAL_GOAL_POS_LOCAL_MIN_START:-"[1.00, -0.20, 0.185]"}
  EXTERNAL_GOAL_POS_LOCAL_MAX_START=${EXTERNAL_GOAL_POS_LOCAL_MAX_START:-"[1.25, 0.20, 0.185]"}
  EXTERNAL_GOAL_POS_LOCAL_MIN=${EXTERNAL_GOAL_POS_LOCAL_MIN:-"[1.00, -0.75, 0.185]"}
  EXTERNAL_GOAL_POS_LOCAL_MAX=${EXTERNAL_GOAL_POS_LOCAL_MAX:-"[1.75, 0.75, 0.185]"}
fi

# Viser GUI defaults aligned with VideoMimic-style manual + clip control.
export VISER_ENABLE_CLIP_GUI=${VISER_ENABLE_CLIP_GUI:-1}
export VISER_ENABLE_MANUAL_GUI=${VISER_ENABLE_MANUAL_GUI:-1}
export VISER_ENABLE_MANUAL_GOAL_GUI=${VISER_ENABLE_MANUAL_GOAL_GUI:-0}
export VISER_TARGET_BOX_REQUIRES_MANUAL_GOAL=${VISER_TARGET_BOX_REQUIRES_MANUAL_GOAL:-1}
export VISER_MANUAL_CONTROL_DEFAULT=${VISER_MANUAL_CONTROL_DEFAULT:-1}
export VISER_FORCE_MANUAL_CONTROL=${VISER_FORCE_MANUAL_CONTROL:-0}
export VISER_SHOW_TARGET_KEYPOINTS=${VISER_SHOW_TARGET_KEYPOINTS:-0}
export VISER_START_PAUSED=${VISER_START_PAUSED:-0}
export VISER_MANUAL_USE_HW_JOYSTICK=${VISER_MANUAL_USE_HW_JOYSTICK:-0}
export VISER_MANUAL_HW_DEADZONE=${VISER_MANUAL_HW_DEADZONE:-0.08}
export VISER_CLIP_LOCK_DEFAULT=${VISER_CLIP_LOCK_DEFAULT:-1}
export HOLOSOMA_DISABLE_AUTO_RESET=${HOLOSOMA_DISABLE_AUTO_RESET:-1}
export HOLOSOMA_DISABLE_MOTION_END_RESET=${HOLOSOMA_DISABLE_MOTION_END_RESET:-1}
export HOLOSOMA_DISABLE_CLIP_END_RESET=${HOLOSOMA_DISABLE_CLIP_END_RESET:-1}
export HOLOSOMA_RESET_TO_DEFAULT_POSE=${HOLOSOMA_RESET_TO_DEFAULT_POSE:-1}

# VideoMimic-style default:
# - manual control comes from Viser GUI toggles.
# - hardware joystick is optional and should not crash startup when no device exists.
USE_HW_JOYSTICK_BRIDGE=${USE_HW_JOYSTICK_BRIDGE:-False}
export VISER_MANUAL_HW_BACKEND=${VISER_MANUAL_HW_BACKEND:-auto}
export VISER_MANUAL_HW_DEVICE=${VISER_MANUAL_HW_DEVICE:-0}
export VISER_MANUAL_HW_TYPE=${VISER_MANUAL_HW_TYPE:-xbox}
# Keep noisy third-party debug logs off by default.
export LOGURU_LEVEL=${LOGURU_LEVEL:-WARNING}
export PY_LOG_LEVEL=${PY_LOG_LEVEL:-WARNING}

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
  # Hard-disable distillation teacher loading during inference.
  --algo.config.distill.enabled False
  --algo.config.distill.mode mse
  --algo.config.distill.ppo_start_epoch -1
  --algo.config.distill.dagger_end_epoch -1
)

if [[ "${VISER_EXTERNAL_SPARSE_GOAL_FLAG}" -eq 1 ]]; then
  cmd+=(
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.enabled "${SPARSE_GOAL_ENABLED}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.clip_goal_delta_min_steps "${CLIP_GOAL_DELTA_MIN_STEPS}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.clip_goal_delta_max_steps "${CLIP_GOAL_DELTA_MAX_STEPS}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.command_only_env_prob_start "${COMMAND_ONLY_ENV_PROB_START}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.command_only_env_prob_end "${COMMAND_ONLY_ENV_PROB_END}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.command_only_env_prob_start_iter "${COMMAND_ONLY_ENV_PROB_START_ITER}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.command_only_env_prob_end_iter "${COMMAND_ONLY_ENV_PROB_END_ITER}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.eval_command_only_env_prob "${EVAL_COMMAND_ONLY_ENV_PROB}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external_goal_prob_start "${EXTERNAL_GOAL_PROB_START}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external_goal_prob_end "${EXTERNAL_GOAL_PROB_END}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external_goal_prob_start_iter "${EXTERNAL_GOAL_PROB_START_ITER}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external_goal_prob_end_iter "${EXTERNAL_GOAL_PROB_END_ITER}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external_goal_prob_ramp_resets "${EXTERNAL_GOAL_PROB_RAMP_RESETS}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.eval_external_goal_prob "${EVAL_EXTERNAL_GOAL_PROB}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.eval_carry_extension_prob "${EVAL_CARRY_EXTENSION_PROB}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external_goal_range_ramp_resets "${EXTERNAL_GOAL_RANGE_RAMP_RESETS}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external_goal_range_start_iter "${EXTERNAL_GOAL_RANGE_START_ITER}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external_goal_range_end_iter "${EXTERNAL_GOAL_RANGE_END_ITER}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external_goal_sampling_mode "${EXTERNAL_GOAL_SAMPLING_MODE}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external_goal_radius_min_start "${EXTERNAL_GOAL_RADIUS_MIN_START}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external_goal_radius_max_start "${EXTERNAL_GOAL_RADIUS_MAX_START}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external_goal_radius_min "${EXTERNAL_GOAL_RADIUS_MIN}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external_goal_radius_max "${EXTERNAL_GOAL_RADIUS_MAX}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external_goal_pos_local_min_start "${EXTERNAL_GOAL_POS_LOCAL_MIN_START}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external_goal_pos_local_max_start "${EXTERNAL_GOAL_POS_LOCAL_MAX_START}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external_goal_pos_local_min "${EXTERNAL_GOAL_POS_LOCAL_MIN}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external_goal_pos_local_max "${EXTERNAL_GOAL_POS_LOCAL_MAX}"
  )
fi

case "$(echo "${FORCE_SINGLE_FRAME_HISTORY}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    cmd+=(
      --observation_overrides.disable_actor_history True
      --observation_overrides.disable_critic_history True
    )
    ;;
  0|false|no|off|"")
    ;;
  *)
    echo "[ERROR] FORCE_SINGLE_FRAME_HISTORY must be one of: 0/1/true/false/yes/no/on/off. Got: ${FORCE_SINGLE_FRAME_HISTORY}" >&2
    exit 2
    ;;
esac

case "$(echo "${DISTILL_PROPRIO_HISTORY_ONLY}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    if [[ "$(echo "${FORCE_SINGLE_FRAME_HISTORY}" | tr '[:upper:]' '[:lower:]')" != "1" && "$(echo "${FORCE_SINGLE_FRAME_HISTORY}" | tr '[:upper:]' '[:lower:]')" != "true" && "$(echo "${FORCE_SINGLE_FRAME_HISTORY}" | tr '[:upper:]' '[:lower:]')" != "yes" && "$(echo "${FORCE_SINGLE_FRAME_HISTORY}" | tr '[:upper:]' '[:lower:]')" != "on" ]]; then
      cmd+=(
        --observation_overrides.distill_proprio_history_only True
        --observation_overrides.distill_proprio_history_length "${DISTILL_PROPRIO_HISTORY_LENGTH}"
      )
    fi
    ;;
  0|false|no|off|"")
    ;;
  *)
    echo "[ERROR] DISTILL_PROPRIO_HISTORY_ONLY must be one of: 0/1/true/false/yes/no/on/off. Got: ${DISTILL_PROPRIO_HISTORY_ONLY}" >&2
    exit 2
    ;;
esac

if [[ "${SIMULATOR_SUBCOMMAND}" != "simulator:mujoco" ]]; then
  cmd+=(--simulator.config.sim.physx.gpu_collision_stack_size "${PHYSX_GPU_COLLISION_STACK_SIZE}")
else
  cmd+=(--randomization.ignore_unsupported True)
fi

if [[ -n "${OBJECT_SCALE_ARG}" ]]; then
  cmd+=(--robot.object.scale "${OBJECT_SCALE_ARG}")
fi

if [[ -n "${GEOMETRY_DIR}" ]]; then
  cmd+=(--geometry-dir "${GEOMETRY_DIR}")
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

if [[ "${USE_HW_JOYSTICK_BRIDGE}" == "True" || "${USE_HW_JOYSTICK_BRIDGE}" == "true" || "${VISER_MANUAL_HW_BACKEND:-auto}" == "bridge" ]]; then
  cmd+=(
    --simulator.config.bridge.enabled True
    --simulator.config.bridge.use_joystick True
    --simulator.config.bridge.joystick_device "${VISER_MANUAL_HW_DEVICE}"
    --simulator.config.bridge.joystick_type "${VISER_MANUAL_HW_TYPE}"
  )
fi

if [[ -n "${PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE}" ]]; then
  cmd+=(--perception.object_geometry_mode "${PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE}")
fi

append_explicit_camera_overrides() {
  if [[ "${IMAGE_WIDTH_EXPLICIT}" -eq 1 ]]; then
    cmd+=(--perception.camera_width "${IMAGE_WIDTH}")
  fi
  if [[ "${IMAGE_HEIGHT_EXPLICIT}" -eq 1 ]]; then
    cmd+=(--perception.camera_height "${IMAGE_HEIGHT}")
  fi
  if [[ "${CAMERA_NEAR_EXPLICIT}" -eq 1 ]]; then
    cmd+=(--perception.camera_near "${CAMERA_NEAR}")
  fi
  if [[ "${CAMERA_FAR_EXPLICIT}" -eq 1 ]]; then
    cmd+=(--perception.camera_far "${CAMERA_FAR}")
  fi
  if [[ "${CAMERA_MAX_DISTANCE_EXPLICIT}" -eq 1 ]]; then
    cmd+=(--perception.max_distance "${CAMERA_MAX_DISTANCE}")
  fi
  if [[ "${CAMERA_PITCH_DEG_EXPLICIT}" -eq 1 ]]; then
    cmd+=(--perception.camera_pitch_deg "${CAMERA_PITCH_DEG}")
  fi
}

if [[ "${MODE}" == "mocap" ]]; then
  case "$(echo "${MOCAP_PERCEPTION_PRESET}" | tr '[:upper:]' '[:lower:]')" in
    checkpoint|auto|"")
      # Keep checkpoint-saved perception config (required by some distill checkpoints).
      ;;
    none)
      cmd+=(perception:none)
      ;;
    heightmap)
      cmd+=(perception:heightmap)
      ;;
    *)
      echo "[ERROR] MOCAP_PERCEPTION_PRESET must be one of: checkpoint|none|heightmap. Got: ${MOCAP_PERCEPTION_PRESET}" >&2
      exit 2
      ;;
  esac
else
  # Depth branch defaults to checkpoint-native perception so actor dims match checkpoint.
  export VISER_PERCEPTION_IMAGE_MODE=${VISER_PERCEPTION_IMAGE_MODE:-depth}
  export VISER_SHOW_PERCEPTION_FRUSTUM=${VISER_SHOW_PERCEPTION_FRUSTUM:-1}
  case "$(echo "${DEPTH_PERCEPTION_PRESET}" | tr '[:upper:]' '[:lower:]')" in
    checkpoint|auto|"")
      # Keep checkpoint-saved perception config.
      ;;
    d435i)
      cmd+=(
        perception:camera-depth-d435i
      )
      ;;
    *)
      echo "[ERROR] DEPTH_PERCEPTION_PRESET must be one of: checkpoint|d435i. Got: ${DEPTH_PERCEPTION_PRESET}" >&2
      exit 2
      ;;
  esac
fi

append_explicit_camera_overrides

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] mode_input=${MODE_INPUT} runtime_mode=${MODE}"
echo "[INFO] simulator_subcommand=${SIMULATOR_SUBCOMMAND:-<default>}"
if [[ -n "${CHECKPOINT_SAVED_MOTION_PATH}" && "${INFER_DATASET_EXPLICIT}" -eq 0 ]]; then
  echo "[INFO] infer_dataset=${INFER_DATASET} (checkpoint overrides motion/object defaults)"
else
  echo "[INFO] infer_dataset=${INFER_DATASET}"
fi
echo "[INFO] checkpoint=${CKPT}"
echo "[INFO] motion_dir=${MOTION_DIR}"
echo "[INFO] object_urdf=${OBJECT_URDF}"
echo "[INFO] force_single_frame_history=${FORCE_SINGLE_FRAME_HISTORY}"
echo "[INFO] distill_proprio_history_only=${DISTILL_PROPRIO_HISTORY_ONLY} distill_proprio_history_length=${DISTILL_PROPRIO_HISTORY_LENGTH}"
if [[ -n "${OBJECT_GEOMETRY_MODE}" ]]; then
  echo "[INFO] object_geometry_mode=${OBJECT_GEOMETRY_MODE} simulator_object_spawn_mode=${HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE}"
else
  echo "[INFO] object_geometry_mode=<default>"
fi
if [[ "${AUTO_SWITCH_MULTI_OBJECT_MODE}" == "1" ]]; then
  echo "[INFO] auto_enabled_per_clip_object_switching=True heterogeneous_single_slot_disabled=${HOLOSOMA_DISABLE_HETEROGENEOUS_OBJECT_SINGLE_SLOT:-0}"
fi
if [[ -n "${CHECKPOINT_SAVED_MOTION_PATH}" ]]; then
  echo "[INFO] checkpoint_saved_motion_path=${CHECKPOINT_SAVED_MOTION_PATH}"
fi
if [[ -n "${CHECKPOINT_SAVED_OBJECT_URDF}" ]]; then
  echo "[INFO] checkpoint_saved_object_urdf=${CHECKPOINT_SAVED_OBJECT_URDF}"
fi
if [[ -n "${CHECKPOINT_SAVED_DISTILL_PROPRIO_HISTORY_ONLY}" ]]; then
  echo "[INFO] checkpoint_saved_distill_proprio_history_only=${CHECKPOINT_SAVED_DISTILL_PROPRIO_HISTORY_ONLY}"
fi
if [[ -n "${CHECKPOINT_SAVED_DISTILL_PROPRIO_HISTORY_LENGTH}" ]]; then
  echo "[INFO] checkpoint_saved_distill_proprio_history_length=${CHECKPOINT_SAVED_DISTILL_PROPRIO_HISTORY_LENGTH}"
fi
if [[ -n "${AUGMENTED_OBJECT_URDF_PATH}" && "${AUGMENTED_OBJECT_URDF_PATH}" != "${CHECKPOINT_SAVED_OBJECT_URDF}" && "${AUGMENTED_OBJECT_URDF_PATH}" != "${DEFAULT_OMOMO_URDF}" ]]; then
  echo "[INFO] augmented_object_urdf=${AUGMENTED_OBJECT_URDF_PATH}"
fi
if [[ -n "${OBJECT_SCALE_ARG}" ]]; then
  echo "[INFO] object_scale=${OBJECT_SCALE_ARG}"
fi
if [[ -n "${GEOMETRY_DIR}" ]]; then
  echo "[INFO] geometry_dir=${GEOMETRY_DIR}"
fi
echo "[INFO] headless=${HEADLESS_FLAG} (env HEADLESS=${HEADLESS})"
echo "[INFO] viser=http://localhost:${VISER_PORT}"
echo "[INFO] manual_gui=${VISER_ENABLE_MANUAL_GUI} manual_goal_gui=${VISER_ENABLE_MANUAL_GOAL_GUI} clip_gui=${VISER_ENABLE_CLIP_GUI}"
echo "[INFO] external_sparse_goal_viz=${VISER_EXTERNAL_SPARSE_GOAL_FLAG}"
if [[ "${VISER_EXTERNAL_SPARSE_GOAL_FLAG}" -eq 1 ]]; then
  echo "[INFO] sparse_goal_enabled=${SPARSE_GOAL_ENABLED} eval_command_only=${EVAL_COMMAND_ONLY_ENV_PROB} eval_external=${EVAL_EXTERNAL_GOAL_PROB} eval_carry=${EVAL_CARRY_EXTENSION_PROB}"
  echo "[INFO] external_goal_sampling_mode=${EXTERNAL_GOAL_SAMPLING_MODE} radius_start=${EXTERNAL_GOAL_RADIUS_MIN_START}->${EXTERNAL_GOAL_RADIUS_MAX_START} radius_end=${EXTERNAL_GOAL_RADIUS_MIN}->${EXTERNAL_GOAL_RADIUS_MAX}"
fi
echo "[INFO] manual_control_default=${VISER_MANUAL_CONTROL_DEFAULT} force_manual=${VISER_FORCE_MANUAL_CONTROL}"
echo "[INFO] hw_joystick=${VISER_MANUAL_USE_HW_JOYSTICK}"
echo "[INFO] hw_backend=${VISER_MANUAL_HW_BACKEND:-auto} bridge_joystick=${USE_HW_JOYSTICK_BRIDGE}"
echo "[INFO] disable_auto_reset=${HOLOSOMA_DISABLE_AUTO_RESET} disable_clip_end_reset=${HOLOSOMA_DISABLE_CLIP_END_RESET}"
echo "[INFO] disable_motion_end_reset=${HOLOSOMA_DISABLE_MOTION_END_RESET}"
echo "[INFO] reset_to_default_pose=${HOLOSOMA_RESET_TO_DEFAULT_POSE}"
if [[ "${MODE}" == "mocap" ]]; then
  echo "[INFO] mocap_perception_preset=${MOCAP_PERCEPTION_PRESET}"
else
  echo "[INFO] depth_perception_preset=${DEPTH_PERCEPTION_PRESET}"
fi
CAMERA_OVERRIDE_SUMMARY=()
[[ "${IMAGE_WIDTH_EXPLICIT}" -eq 1 ]] && CAMERA_OVERRIDE_SUMMARY+=("camera_width=${IMAGE_WIDTH}")
[[ "${IMAGE_HEIGHT_EXPLICIT}" -eq 1 ]] && CAMERA_OVERRIDE_SUMMARY+=("camera_height=${IMAGE_HEIGHT}")
[[ "${CAMERA_NEAR_EXPLICIT}" -eq 1 ]] && CAMERA_OVERRIDE_SUMMARY+=("camera_near=${CAMERA_NEAR}")
[[ "${CAMERA_FAR_EXPLICIT}" -eq 1 ]] && CAMERA_OVERRIDE_SUMMARY+=("camera_far=${CAMERA_FAR}")
[[ "${CAMERA_MAX_DISTANCE_EXPLICIT}" -eq 1 ]] && CAMERA_OVERRIDE_SUMMARY+=("max_distance=${CAMERA_MAX_DISTANCE}")
[[ "${CAMERA_PITCH_DEG_EXPLICIT}" -eq 1 ]] && CAMERA_OVERRIDE_SUMMARY+=("camera_pitch_deg=${CAMERA_PITCH_DEG}")
if [[ "${#CAMERA_OVERRIDE_SUMMARY[@]}" -gt 0 ]]; then
  echo "[INFO] explicit_camera_overrides=${CAMERA_OVERRIDE_SUMMARY[*]}"
else
  echo "[INFO] explicit_camera_overrides=<none; checkpoint camera preserved>"
fi
echo "[INFO] Viser controls:"
echo "  1) Open 'Manual Control' and enable 'Enable Manual Root Command'."
echo "  2) Set 'Root dX/dY/dYaw' as the desired root-frame relative command."
echo "  3) Use 'Zero Root Command' to reset the relative root command to zero."
echo "  4) Use 'Advanced > Reset Object' to add box position/rotation offsets for the next reset."
echo "  5) Use 'Clip Playback' to select clip/start frame and click 'Apply Clip'."
echo "  6) Use 'Advanced > Simulation Control' for Play/Step/Reset (Reset returns to the default pose)."
if command -v hostname >/dev/null 2>&1; then
  HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  if [[ -n "${HOST_IP}" ]]; then
    echo "[INFO] Remote URL: http://${HOST_IP}:${VISER_PORT}"
    echo "[INFO] SSH tunnel example: ssh -N -L ${VISER_PORT}:localhost:${VISER_PORT} <user>@<host>"
  fi
fi

if [[ "${DRY_RUN_FLAG}" == "1" ]]; then
  printf '[DRY_RUN] '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
