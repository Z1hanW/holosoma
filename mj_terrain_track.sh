#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Terrain-aware split sim2sim launcher for single-clip policy inference.
#
# This path is intentionally ONNX-first:
# 1. resolve/download the ONNX for a W&B run or local checkpoint
# 2. patch the ONNX to a single motion clip
# 3. resolve the paired terrain OBJ for that clip
# 4. launch run_sim.py + run_policy.py directly

WANDB_RUN_URL="${WANDB_RUN_URL:-https://wandb.ai/zihanw22/terrain-aware/runs/qs0vyrn2}"
DEFAULT_MODEL_INPUT="${DEFAULT_MODEL_INPUT:-${WANDB_RUN_URL}}"
DEFAULT_TERRAIN_MOTION_DIR="${DEFAULT_TERRAIN_MOTION_DIR:-$ROOT_DIR/data/ds_crisp_data/_generated/___crisp_clean_motion_gmr_g1_trainready_rebuilt_20260423}"
DEFAULT_TERRAIN_GEOMETRY_DIR="${DEFAULT_TERRAIN_GEOMETRY_DIR:-$ROOT_DIR/data/ds_crisp_data/_generated/___crisp_clean_geometry_s0p7415730337}"

usage() {
  cat <<EOF
Usage:
  bash mj_terrain_track.sh [checkpoint.pt|model.onnx|wandb://...|https://wandb.ai/.../runs/...] [motion_dir_or_file] [geometry_obj_or_dir] [motion_clip_name]

Defaults:
  model    = ${DEFAULT_MODEL_INPUT}
  motion   = ${DEFAULT_TERRAIN_MOTION_DIR}
  geometry = ${DEFAULT_TERRAIN_GEOMETRY_DIR}

Examples:
  bash mj_terrain_track.sh
  MOTION_CLIP_NAME=stair_16 bash mj_terrain_track.sh
  bash mj_terrain_track.sh https://wandb.ai/zihanw22/terrain-aware/runs/qs0vyrn2
  RUN_SECONDS=10 HEADLESS=True MOTION_CLIP_NAME=stair_16 bash mj_terrain_track.sh

Optional env vars:
  MOTION_CLIP_NAME / MOTION_CLIP_ID
  WANDB_MODEL_FILE
  WANDB_DOWNLOAD_DIR   (default: /tmp/wandb_terrain_track)
  PATCH_DIR            (default: $ROOT_DIR/logs/sim2sim_exports)
  RUN_SECONDS          (default: 0, run until interrupted)
  HEADLESS             (default: True)
  INFERENCE_CONFIG     (default: g1-29dof-wbt-terrain-aware)
  SIM_FPS / SIM_CONTROL_DECIMATION
  SIM_CLOCK_PORT / SIM_STATE_PORT / SIM_PERCEPTION_PORT / SIM_CONTROL_PORT / POLICY_CONTROL_PORT
  SIM_USE_ZMQ_LOWCMD   (default: 1)
  SIM_LOG_FIRST_COMMAND_SUMMARY (default: 0)
  USE_SIM_TIME         (default: 1)
  PREFER_SIM_REF_FROM_SIM_STATE (default: 1)
  USE_ROOT_REFERENCE_AT_CLIP_START (default: 1)
  RESTART_MOTION_ON_CLOCK_RESET (default: 1)
  POLICY_DEFER_UNTIL_VALID_STATE (default: 1)
  APPLY_TRAINING_MOTION_TRANSITIONS (default: 0)
  SIM_MOTION_INIT_MODE (default: raw_motion)
  DRY_RUN              (default: 0)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

MODEL_INPUT="${1:-$DEFAULT_MODEL_INPUT}"
MOTION_SOURCE="${2:-${MOTION_SOURCE:-}}"
GEOMETRY_SOURCE="${3:-${GEOMETRY_SOURCE:-}}"
POSITIONAL_CLIP_NAME="${4:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MUJOCO_PY="${MUJOCO_PY:-}"
INFER_PY="${INFER_PY:-}"

WANDB_DOWNLOAD_DIR="${WANDB_DOWNLOAD_DIR:-/tmp/wandb_terrain_track}"
PATCH_DIR="${PATCH_DIR:-$ROOT_DIR/logs/sim2sim_exports}"
RUN_SECONDS="${RUN_SECONDS:-0}"
HEADLESS_RAW="${HEADLESS:-True}"
INFERENCE_CONFIG="${INFERENCE_CONFIG:-g1-29dof-wbt-terrain-aware}"

SIM_FPS="${SIM_FPS:-}"
SIM_CONTROL_DECIMATION="${SIM_CONTROL_DECIMATION:-}"
SIM_SUBSTEPS="${SIM_SUBSTEPS:-}"
SIM_RUN_DEVICE="${SIM_RUN_DEVICE:-}"
SIM_CLOCK_PORT="${SIM_CLOCK_PORT:-5755}"
SIM_STATE_PORT="${SIM_STATE_PORT:-5757}"
SIM_PERCEPTION_PORT="${SIM_PERCEPTION_PORT:-5758}"
SIM_CONTROL_PORT="${SIM_CONTROL_PORT:-5759}"
POLICY_CONTROL_PORT="${POLICY_CONTROL_PORT:-0}"
INTERFACE_NAME="${INTERFACE_NAME:-lo}"
SIM_USE_ZMQ_LOWCMD="${SIM_USE_ZMQ_LOWCMD:-1}"
SIM_IGNORE_DEFAULT_IDLE_COMMAND="${SIM_IGNORE_DEFAULT_IDLE_COMMAND:-1}"
SIM_LOG_FIRST_COMMAND_SUMMARY="${SIM_LOG_FIRST_COMMAND_SUMMARY:-0}"
SIM_HOLD_DEFAULT_POSE_UNTIL_FIRST_COMMAND="${SIM_HOLD_DEFAULT_POSE_UNTIL_FIRST_COMMAND:-0}"
SIM_HOLD_INITIAL_POSE_UNTIL_FIRST_COMMAND="${SIM_HOLD_INITIAL_POSE_UNTIL_FIRST_COMMAND:-0}"
SIM_FREEZE_UNTIL_FIRST_COMMAND="${SIM_FREEZE_UNTIL_FIRST_COMMAND:-1}"
SIM_READY_TIMEOUT="${SIM_READY_TIMEOUT:-45}"
SIM_READY_PATTERN="${SIM_READY_PATTERN:-Starting direct simulation loop...}"
SIM_STARTUP_WAIT="${SIM_STARTUP_WAIT:-0}"
SIM_DEBUG_VIZ="${SIM_DEBUG_VIZ:-True}"

USE_SIM_TIME="${USE_SIM_TIME:-1}"
PREFER_SIM_REF_FROM_SIM_STATE="${PREFER_SIM_REF_FROM_SIM_STATE:-1}"
USE_ROOT_REFERENCE_AT_CLIP_START="${USE_ROOT_REFERENCE_AT_CLIP_START:-1}"
RESTART_MOTION_ON_CLOCK_RESET="${RESTART_MOTION_ON_CLOCK_RESET:-1}"
POLICY_DEFER_UNTIL_VALID_STATE="${POLICY_DEFER_UNTIL_VALID_STATE:-1}"
APPLY_TRAINING_MOTION_TRANSITIONS="${APPLY_TRAINING_MOTION_TRANSITIONS:-0}"
SIM_MOTION_INIT_MODE="${SIM_MOTION_INIT_MODE:-raw_motion}"
AUTO_START_STIFF_HOLD_SEC="${AUTO_START_STIFF_HOLD_SEC:-0.0}"
AUTO_START_STIFF_MAX_WAIT_SEC="${AUTO_START_STIFF_MAX_WAIT_SEC:-0.0}"
AUTO_START_STIFF_POSE_TOL="${AUTO_START_STIFF_POSE_TOL:-0.12}"
DRY_RUN="${DRY_RUN:-0}"
SIM_PERCEPTION_CAMERA_SOURCE_OVERRIDE="${SIM_PERCEPTION_CAMERA_SOURCE_OVERRIDE:-}"

mkdir -p "$PATCH_DIR" "$WANDB_DOWNLOAD_DIR"

export PYTHONPATH="$ROOT_DIR/src/holosoma:$ROOT_DIR/src/holosoma_inference${PYTHONPATH:+:$PYTHONPATH}"

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

HEADLESS="$(normalize_bool_flag HEADLESS "$HEADLESS_RAW")"
SIM_USE_ZMQ_LOWCMD="$(normalize_bool_int SIM_USE_ZMQ_LOWCMD "$SIM_USE_ZMQ_LOWCMD")"
USE_SIM_TIME="$(normalize_bool_int USE_SIM_TIME "$USE_SIM_TIME")"
PREFER_SIM_REF_FROM_SIM_STATE="$(normalize_bool_int PREFER_SIM_REF_FROM_SIM_STATE "$PREFER_SIM_REF_FROM_SIM_STATE")"
USE_ROOT_REFERENCE_AT_CLIP_START="$(normalize_bool_int USE_ROOT_REFERENCE_AT_CLIP_START "$USE_ROOT_REFERENCE_AT_CLIP_START")"
RESTART_MOTION_ON_CLOCK_RESET="$(normalize_bool_int RESTART_MOTION_ON_CLOCK_RESET "$RESTART_MOTION_ON_CLOCK_RESET")"
POLICY_DEFER_UNTIL_VALID_STATE="$(normalize_bool_int POLICY_DEFER_UNTIL_VALID_STATE "$POLICY_DEFER_UNTIL_VALID_STATE")"
APPLY_TRAINING_MOTION_TRANSITIONS="$(normalize_bool_int APPLY_TRAINING_MOTION_TRANSITIONS "$APPLY_TRAINING_MOTION_TRANSITIONS")"
DRY_RUN="$(normalize_bool_int DRY_RUN "$DRY_RUN")"

INFERENCE_SUBCOMMAND="$INFERENCE_CONFIG"
if [[ "$INFERENCE_SUBCOMMAND" != inference:* ]]; then
  INFERENCE_SUBCOMMAND="inference:${INFERENCE_SUBCOMMAND}"
fi

resolve_python() {
  local configured="$1"
  shift
  if [[ -n "$configured" ]]; then
    if [[ ! -x "$configured" ]]; then
      echo "Configured python is not executable: $configured" >&2
      exit 1
    fi
    printf '%s\n' "$configured"
    return
  fi
  local candidate
  for candidate in "$@"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo "No usable python interpreter found" >&2
  exit 1
}

python_has_module() {
  local python_bin="$1"
  local module_name="$2"
  "$python_bin" - <<'PY' "$module_name" >/dev/null 2>&1
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec(sys.argv[1]) is not None else 1)
PY
}

resolve_python_with_module() {
  local module_name="$1"
  shift
  local candidate
  for candidate in "$@"; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    if python_has_module "$candidate" "$module_name"; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  echo "No usable python interpreter with module '$module_name' found" >&2
  exit 1
}

MUJOCO_PY="$(resolve_python_with_module mujoco \
  "$(resolve_python "$MUJOCO_PY")" \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python)"
INFER_PY="$(resolve_python "$INFER_PY" \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python)"

resolve_data_path() {
  local path_value="$1"
  if [[ -z "${path_value}" ]]; then
    echo ""
    return 0
  fi
  if [[ "${path_value}" == /* ]]; then
    echo "${path_value}"
    return 0
  fi
  if [[ "${path_value}" == @holosoma/* ]]; then
    echo "${ROOT_DIR}/src/holosoma/${path_value#@holosoma/}"
    return 0
  fi
  if [[ "${path_value}" == holosoma/data/* ]]; then
    echo "${ROOT_DIR}/src/holosoma/${path_value}"
    return 0
  fi
  "$INFER_PY" - <<'PY' "${path_value}"
import sys
from pathlib import Path

print(str(Path(sys.argv[1]).expanduser().resolve()))
PY
}

resolve_model_input() {
  "$INFER_PY" - <<'PY' "$1" "${WANDB_MODEL_FILE:-}" "$WANDB_DOWNLOAD_DIR"
from __future__ import annotations

import sys
from pathlib import Path

def parse_reference(reference: str) -> tuple[str, str, str, str]:
    if reference.startswith("wandb://"):
        parts = reference[len("wandb://") :].split("/")
        if len(parts) < 4:
            raise SystemExit(f"Invalid wandb reference: {reference}")
        entity, project = parts[0], parts[1]
        run_id_index = 2
        if len(parts) > 4 and parts[2] == "runs":
            run_id_index = 3
        run_id = parts[run_id_index]
        file_name = "/".join(parts[run_id_index + 1 :])
        return entity, project, run_id, file_name
    if reference.startswith("https://wandb.ai/"):
        clean = reference.split("?", 1)[0]
        trimmed = clean[len("https://wandb.ai/") :]
        parts = trimmed.split("/")
        if len(parts) < 4 or parts[2] != "runs":
            raise SystemExit(f"Unsupported W&B URL: {reference}")
        entity, project, run_id = parts[0], parts[1], parts[3]
        explicit_file = ""
        if len(parts) >= 6 and parts[4] == "files":
            explicit_file = "/".join(parts[5:])
        return entity, project, run_id, explicit_file
    raise SystemExit("not-wandb")


reference = sys.argv[1]
preferred_file = sys.argv[2].strip()
download_root = Path(sys.argv[3]).expanduser().resolve()

local_path = Path(reference).expanduser()
if local_path.is_file():
    resolved = local_path.resolve()
    if resolved.suffix == ".pt":
        sibling = resolved.with_suffix(".onnx")
        if not sibling.is_file():
            raise SystemExit(f"Expected sibling ONNX next to checkpoint: {sibling}")
        print(str(sibling))
    else:
        print(str(resolved))
    raise SystemExit(0)

try:
    entity, project, run_id, explicit_file = parse_reference(reference)
except SystemExit as exc:
    if str(exc) == "not-wandb":
        raise SystemExit(f"Model input not found: {reference}")
    raise

import wandb

api = wandb.Api(timeout=30)
run = api.run(f"{entity}/{project}/{run_id}")

candidates: list[str] = []
requested = explicit_file or preferred_file
if requested:
    requested = requested.strip()
    if requested.endswith(".onnx"):
        candidates.append(requested)
    elif requested.endswith(".pt"):
        candidates.append(f"{Path(requested).stem}.onnx")

if not candidates:
    history_line_count = run._attrs.get("historyLineCount")
    if isinstance(history_line_count, int) and history_line_count > 0:
        step = history_line_count - 1
        candidates.append(f"model_{step:05d}.onnx")
        candidates.append(f"model_{step}.onnx")
    candidates.append("model.onnx")

download_root.mkdir(parents=True, exist_ok=True)
for candidate in candidates:
    try:
        downloaded = run.file(candidate).download(root=str(download_root), replace=True)
    except Exception:
        continue
    path = Path(downloaded.name)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if path.suffix != ".onnx":
        sibling = path.with_suffix(".onnx")
        if sibling.is_file():
            path = sibling
        else:
            continue
    print(str(path))
    raise SystemExit(0)

raise SystemExit(
    f"Unable to download an ONNX for W&B run {entity}/{project}/{run_id}. Candidates={candidates}"
)
PY
}

extract_model_defaults() {
  "$INFER_PY" - <<'PY' "$1"
from __future__ import annotations

import json
import sys

import onnx

model = onnx.load(sys.argv[1])
metadata = {}
for prop in model.metadata_props:
    try:
        metadata[prop.key] = json.loads(prop.value)
    except Exception:
        metadata[prop.key] = prop.value

exp_cfg = metadata.get("experiment_config", {})
if not isinstance(exp_cfg, dict):
    raise SystemExit(0)

motion_cfg = (
    exp_cfg.get("command", {})
    .get("setup_terms", {})
    .get("motion_command", {})
    .get("params", {})
    .get("motion_config", {})
)
terrain_term = exp_cfg.get("terrain", {}).get("terrain_term", {})
perception_cfg = exp_cfg.get("perception", {})
sim_cfg = exp_cfg.get("simulator", {}).get("config", {}).get("sim", {})

values = [
    motion_cfg.get("motion_file"),
    motion_cfg.get("motion_clip_name"),
    motion_cfg.get("motion_clip_id"),
    terrain_term.get("obj_file_path"),
    terrain_term.get("obj_metadata_path"),
    terrain_term.get("num_rows"),
    terrain_term.get("num_cols"),
    perception_cfg.get("enabled"),
    perception_cfg.get("output_mode"),
    sim_cfg.get("fps"),
    sim_cfg.get("control_decimation"),
]
for value in values:
    print("" if value is None else str(value))
PY
}

resolve_motion_file_and_clip() {
  "$INFER_PY" - <<'PY' "$1" "${2:-}" "${3:-}"
from __future__ import annotations

import sys
from pathlib import Path

motion_source = Path(sys.argv[1]).expanduser().resolve()
clip_name = str(sys.argv[2] or "").strip()
clip_id_raw = str(sys.argv[3] or "").strip()

if motion_source.is_file():
    print(str(motion_source))
    print(motion_source.stem)
    raise SystemExit(0)

if not motion_source.is_dir():
    raise SystemExit(f"Motion source not found: {motion_source}")

files = sorted(list(motion_source.glob("*.npz")) + list(motion_source.glob("*.NPZ")))
if not files:
    raise SystemExit(f"No .npz clips found in {motion_source}")

if clip_name:
    candidate = motion_source / f"{clip_name}.npz"
    if not candidate.is_file():
        candidate = motion_source / f"{clip_name}.NPZ"
    if not candidate.is_file():
        raise SystemExit(f"Motion clip not found: {clip_name}")
    print(str(candidate.resolve()))
    print(candidate.stem)
    raise SystemExit(0)

if clip_id_raw:
    clip_id = int(clip_id_raw)
    if clip_id < 0 or clip_id >= len(files):
        raise SystemExit(f"MOTION_CLIP_ID out of range: {clip_id}")
    print(str(files[clip_id].resolve()))
    print(files[clip_id].stem)
    raise SystemExit(0)

print(str(files[0].resolve()))
print(files[0].stem)
PY
}

resolve_geometry_obj() {
  "$INFER_PY" - <<'PY' "$1" "$2"
from __future__ import annotations

import sys
from pathlib import Path

geometry_source = Path(sys.argv[1]).expanduser().resolve()
clip_name = str(sys.argv[2]).strip()

if geometry_source.is_file():
    print(str(geometry_source))
    raise SystemExit(0)

if not geometry_source.is_dir():
    raise SystemExit(f"Geometry source not found: {geometry_source}")

candidate = geometry_source / f"{clip_name}.obj"
if not candidate.is_file():
    candidate = geometry_source / f"{clip_name}.OBJ"
if not candidate.is_file():
    raise SystemExit(f"Paired terrain OBJ not found for clip '{clip_name}' in {geometry_source}")

print(str(candidate.resolve()))
PY
}

model_uses_perception_obs() {
  "$INFER_PY" - <<'PY' "$1"
import sys

import onnx

model = onnx.load(sys.argv[1])
input_names = {value.name for value in model.graph.input}
print("1" if "perception_obs" in input_names else "0")
PY
}

build_training_perception_args() {
  "$INFER_PY" - <<'PY' "$1" "$2"
import json
import sys

import onnx

model = onnx.load(sys.argv[1])
camera_source_override = sys.argv[2].strip()
metadata = {}
for prop in model.metadata_props:
    try:
        metadata[prop.key] = json.loads(prop.value)
    except Exception:
        metadata[prop.key] = prop.value

perception_cfg = metadata.get("experiment_config", {}).get("perception", {})
if not isinstance(perception_cfg, dict) or not bool(perception_cfg.get("enabled", False)):
    raise SystemExit(0)

perception_cfg = dict(perception_cfg)
if camera_source_override:
    perception_cfg["camera_source"] = camera_source_override

output_mode = str(perception_cfg.get("output_mode", "")).strip().lower()
if output_mode == "camera_depth":
    print("perception:camera_depth_d435i")
elif output_mode == "heightmap":
    print("perception:heightmap")
else:
    raise SystemExit(f"Unsupported split sim perception output_mode: {output_mode!r}")

for key, value in perception_cfg.items():
    if key in {"enabled", "output_mode"} or value is None:
        continue
    if isinstance(value, bool):
        text = "True" if value else "False"
    elif isinstance(value, (int, float, str)):
        text = str(value)
    elif isinstance(value, (list, tuple)):
        if len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
            text = f"({value[0]},{value[1]})"
        else:
            text = json.dumps(value, separators=(",", ":"))
    elif isinstance(value, dict):
        text = json.dumps(value, separators=(",", ":"))
    else:
        continue
    print(f"--perception.{key}")
    print(text)
PY
}

extract_perception_arg_value() {
  local key="$1"
  shift
  local args=("$@")
  local idx=0
  local needle="--perception.${key}"
  while [[ $idx -lt ${#args[@]} ]]; do
    if [[ "${args[$idx]}" == "$needle" ]]; then
      local next_idx=$((idx + 1))
      if [[ $next_idx -lt ${#args[@]} ]]; then
        printf '%s\n' "${args[$next_idx]}"
      fi
      return 0
    fi
    idx=$((idx + 1))
  done
  return 1
}

python_cuda_available() {
  local python_bin="$1"
  "$python_bin" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
}

MODEL_ONNX="$(resolve_model_input "$MODEL_INPUT")"
mapfile -t MODEL_DEFAULTS < <(extract_model_defaults "$MODEL_ONNX")
CKPT_MOTION_SOURCE="$(resolve_data_path "${MODEL_DEFAULTS[0]:-}")"
CKPT_MOTION_CLIP_NAME="${MODEL_DEFAULTS[1]:-}"
CKPT_MOTION_CLIP_ID="${MODEL_DEFAULTS[2]:-}"
CKPT_GEOMETRY_SOURCE="$(resolve_data_path "${MODEL_DEFAULTS[3]:-}")"
CKPT_GEOMETRY_METADATA="$(resolve_data_path "${MODEL_DEFAULTS[4]:-}")"
CKPT_NUM_ROWS="${MODEL_DEFAULTS[5]:-}"
CKPT_NUM_COLS="${MODEL_DEFAULTS[6]:-}"
CKPT_PERCEPTION_ENABLED="${MODEL_DEFAULTS[7]:-}"
CKPT_PERCEPTION_OUTPUT_MODE="${MODEL_DEFAULTS[8]:-}"
CKPT_SIM_FPS="${MODEL_DEFAULTS[9]:-}"
CKPT_SIM_CONTROL_DECIMATION="${MODEL_DEFAULTS[10]:-}"

if [[ -z "$SIM_FPS" && -n "$CKPT_SIM_FPS" ]]; then
  SIM_FPS="$CKPT_SIM_FPS"
fi
if [[ -z "$SIM_CONTROL_DECIMATION" && -n "$CKPT_SIM_CONTROL_DECIMATION" ]]; then
  SIM_CONTROL_DECIMATION="$CKPT_SIM_CONTROL_DECIMATION"
fi
SIM_FPS="${SIM_FPS:-200}"
SIM_CONTROL_DECIMATION="${SIM_CONTROL_DECIMATION:-4}"

if [[ -z "$MOTION_SOURCE" ]]; then
  if [[ -n "$CKPT_MOTION_SOURCE" && -e "$CKPT_MOTION_SOURCE" ]]; then
    MOTION_SOURCE="$CKPT_MOTION_SOURCE"
  else
    MOTION_SOURCE="$DEFAULT_TERRAIN_MOTION_DIR"
  fi
fi
if [[ -z "$GEOMETRY_SOURCE" ]]; then
  if [[ -d "$DEFAULT_TERRAIN_GEOMETRY_DIR" ]]; then
    GEOMETRY_SOURCE="$DEFAULT_TERRAIN_GEOMETRY_DIR"
  else
    GEOMETRY_SOURCE="$CKPT_GEOMETRY_SOURCE"
  fi
fi

if [[ ! -e "$MOTION_SOURCE" ]]; then
  echo "[ERROR] Motion source not found: $MOTION_SOURCE" >&2
  exit 1
fi
if [[ ! -e "$GEOMETRY_SOURCE" ]]; then
  echo "[ERROR] Geometry source not found: $GEOMETRY_SOURCE" >&2
  exit 1
fi

if [[ -n "$POSITIONAL_CLIP_NAME" ]]; then
  MOTION_CLIP_NAME="$POSITIONAL_CLIP_NAME"
fi
MOTION_CLIP_NAME="${MOTION_CLIP_NAME:-$CKPT_MOTION_CLIP_NAME}"
MOTION_CLIP_ID="${MOTION_CLIP_ID:-$CKPT_MOTION_CLIP_ID}"

mapfile -t MOTION_SELECTION < <(resolve_motion_file_and_clip "$MOTION_SOURCE" "${MOTION_CLIP_NAME:-}" "${MOTION_CLIP_ID:-}")
MOTION_FILE="${MOTION_SELECTION[0]}"
RESOLVED_CLIP_NAME="${MOTION_SELECTION[1]}"

if [[ -z "${MOTION_CLIP_NAME:-}" && -z "${MOTION_CLIP_ID:-}" ]]; then
  echo "[INFO] No motion clip was pinned; defaulting to first sorted clip: ${RESOLVED_CLIP_NAME}" >&2
fi

GEOMETRY_OBJ="$(resolve_geometry_obj "$GEOMETRY_SOURCE" "$RESOLVED_CLIP_NAME")"

MODEL_STEM="$(basename "${MODEL_ONNX%.*}")"
PATCHED_ONNX="$PATCH_DIR/${MODEL_STEM}__${RESOLVED_CLIP_NAME}.onnx"
RUN_DIR="$ROOT_DIR/logs/sim2sim_runs/${RESOLVED_CLIP_NAME}__terrain_tracking"
mkdir -p "$RUN_DIR"

"$INFER_PY" "$ROOT_DIR/src/holosoma_inference/holosoma_inference/tools/patch_motion_onnx.py" \
  --model-path "$MODEL_ONNX" \
  --motion-file "$MOTION_FILE" \
  $( [[ "$APPLY_TRAINING_MOTION_TRANSITIONS" == "1" ]] && printf '%s' "--apply-training-motion-transitions" ) \
  --output-path "$PATCHED_ONNX"

MODEL_EXPECTS_PERCEPTION_OBS="$(model_uses_perception_obs "$PATCHED_ONNX")"
PERCEPTION_ARGS=()
if [[ "$MODEL_EXPECTS_PERCEPTION_OBS" == "1" ]]; then
  mapfile -t PERCEPTION_ARGS < <(build_training_perception_args "$PATCHED_ONNX" "$SIM_PERCEPTION_CAMERA_SOURCE_OVERRIDE")
  if [[ "${#PERCEPTION_ARGS[@]}" -eq 0 ]]; then
    echo "[ERROR] Model expects perception_obs, but no training perception config was found in $PATCHED_ONNX" >&2
    exit 1
  fi
fi

SIM_PERCEPTION_CAMERA_SOURCE=""
if [[ "${#PERCEPTION_ARGS[@]}" -gt 0 ]]; then
  SIM_PERCEPTION_CAMERA_SOURCE="$(extract_perception_arg_value "camera_source" "${PERCEPTION_ARGS[@]}" || true)"
fi
if [[ -z "$SIM_RUN_DEVICE" && "$SIM_PERCEPTION_CAMERA_SOURCE" == "far_tracking_warp" ]]; then
  if python_cuda_available "$MUJOCO_PY"; then
    SIM_RUN_DEVICE="cuda:0"
  fi
fi

SIM_LOG="$RUN_DIR/mujoco.log"
POLICY_LOG="$RUN_DIR/policy.log"
: >"$SIM_LOG"
: >"$POLICY_LOG"

cleanup() {
  if [[ -n "${POLICY_PID:-}" ]] && kill -0 "$POLICY_PID" 2>/dev/null; then
    kill "$POLICY_PID" 2>/dev/null || true
    wait "$POLICY_PID" 2>/dev/null || true
  fi
  if [[ -n "${SIM_PID:-}" ]] && kill -0 "$SIM_PID" 2>/dev/null; then
    kill "$SIM_PID" 2>/dev/null || true
    wait "$SIM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_sim_ready() {
  local deadline=$((SECONDS + SIM_READY_TIMEOUT))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$SIM_PID" 2>/dev/null; then
      echo "MuJoCo simulator exited during startup. See $SIM_LOG" >&2
      tail -n 60 "$SIM_LOG" >&2 || true
      return 1
    fi
    if [[ -f "$SIM_LOG" ]] && grep -qF "$SIM_READY_PATTERN" "$SIM_LOG"; then
      return 0
    fi
    sleep 0.5
  done
  echo "Timed out waiting for MuJoCo readiness pattern '$SIM_READY_PATTERN'. See $SIM_LOG" >&2
  tail -n 60 "$SIM_LOG" >&2 || true
  return 1
}

SIM_CMD=(
  "$MUJOCO_PY" "$ROOT_DIR/src/holosoma/holosoma/run_sim.py"
  simulator:mujoco
  robot:g1_29dof
  terrain:terrain-load-obj
  --training.headless "$HEADLESS"
  --simulator.config.debug-viz "$SIM_DEBUG_VIZ"
  --simulator.config.sim.fps "$SIM_FPS"
  --simulator.config.sim.control-decimation "$SIM_CONTROL_DECIMATION"
  --terrain.terrain-term.obj-file-path "$GEOMETRY_OBJ"
  --terrain.terrain-term.num-rows 1
  --terrain.terrain-term.num-cols 1
  --simulator.config.bridge.interface "$INTERFACE_NAME"
  --simulator.config.bridge.clock-port "$SIM_CLOCK_PORT"
  --simulator.config.bridge.publish-sim-state=True
  --simulator.config.bridge.listen-control=True
  --simulator.config.bridge.sim-state-port "$SIM_STATE_PORT"
  --simulator.config.bridge.control-port "$SIM_CONTROL_PORT"
  --motion-init.enabled=True
  --motion-init.motion-file "$MOTION_FILE"
  --motion-init.mode "$SIM_MOTION_INIT_MODE"
)
if [[ -n "$SIM_RUN_DEVICE" ]]; then
  SIM_CMD+=(--device "$SIM_RUN_DEVICE")
fi
if [[ -n "$SIM_SUBSTEPS" ]]; then
  SIM_CMD+=(--simulator.config.sim.substeps "$SIM_SUBSTEPS")
fi
if [[ "$MODEL_EXPECTS_PERCEPTION_OBS" == "1" ]]; then
  SIM_CMD+=(
    "${PERCEPTION_ARGS[@]}"
    --simulator.config.bridge.publish-perception-obs=True
    --simulator.config.bridge.perception-obs-port "$SIM_PERCEPTION_PORT"
  )
fi
if [[ "$SIM_USE_ZMQ_LOWCMD" == "1" ]]; then
  SIM_CMD+=(--simulator.config.bridge.use-zmq-lowcmd True)
fi
if [[ "$SIM_IGNORE_DEFAULT_IDLE_COMMAND" == "1" ]]; then
  SIM_CMD+=(--simulator.config.bridge.ignore-default-idle-command True)
fi
if [[ "$SIM_LOG_FIRST_COMMAND_SUMMARY" == "1" ]]; then
  SIM_CMD+=(--simulator.config.bridge.log-first-command-summary True)
fi
if [[ "$SIM_HOLD_DEFAULT_POSE_UNTIL_FIRST_COMMAND" == "1" ]]; then
  SIM_CMD+=(--simulator.config.bridge.hold-default-pose-until-first-command True)
fi
if [[ "$SIM_HOLD_INITIAL_POSE_UNTIL_FIRST_COMMAND" == "1" ]]; then
  SIM_CMD+=(--simulator.config.bridge.hold-initial-pose-until-first-command True)
fi
if [[ "$SIM_FREEZE_UNTIL_FIRST_COMMAND" == "1" ]]; then
  SIM_CMD+=(--simulator.config.bridge.freeze-until-first-command True)
fi

POLICY_CMD=(
  "$INFER_PY" "$ROOT_DIR/src/holosoma_inference/holosoma_inference/run_policy.py"
  "$INFERENCE_SUBCOMMAND"
  --task.model-path "$PATCHED_ONNX"
  --task.motion-file "$MOTION_FILE"
  --task.interface "$INTERFACE_NAME"
  --task.use-sim-state
  --task.sim-clock-port "$SIM_CLOCK_PORT"
  --task.sim-state-port "$SIM_STATE_PORT"
  --task.sim-control-port "$SIM_CONTROL_PORT"
  --task.no-auto-start-motion
  --task.auto-start-motion-clip
  --task.auto-start-stiff-hold-sec "$AUTO_START_STIFF_HOLD_SEC"
  --task.auto-start-stiff-max-wait-sec "$AUTO_START_STIFF_MAX_WAIT_SEC"
  --task.auto-start-stiff-pose-tolerance "$AUTO_START_STIFF_POSE_TOL"
)
if [[ "$POLICY_CONTROL_PORT" != "0" ]]; then
  POLICY_CMD+=(--task.policy-control-port "$POLICY_CONTROL_PORT")
fi
if [[ "$SIM_USE_ZMQ_LOWCMD" == "1" ]]; then
  POLICY_CMD+=(--task.use-zmq-lowcmd)
fi
if [[ "$MODEL_EXPECTS_PERCEPTION_OBS" == "1" ]]; then
  POLICY_CMD+=(--task.use-split-perception-obs --task.perception-obs-port "$SIM_PERCEPTION_PORT")
fi
if [[ "$USE_SIM_TIME" == "1" ]]; then
  POLICY_CMD+=(--task.use-sim-time)
fi
if [[ "$PREFER_SIM_REF_FROM_SIM_STATE" == "1" ]]; then
  POLICY_CMD+=(--task.prefer-sim-ref-from-sim-state)
fi
if [[ "$USE_ROOT_REFERENCE_AT_CLIP_START" == "1" ]]; then
  POLICY_CMD+=(--task.use-root-reference-at-clip-start)
fi
if [[ "$RESTART_MOTION_ON_CLOCK_RESET" == "1" ]]; then
  POLICY_CMD+=(--task.restart-motion-on-clock-reset)
fi
if [[ "$POLICY_DEFER_UNTIL_VALID_STATE" == "1" ]]; then
  POLICY_CMD+=(--task.defer-policy-start-until-valid-state)
fi
if [[ "$APPLY_TRAINING_MOTION_TRANSITIONS" == "1" ]]; then
  POLICY_CMD+=(--task.apply-training-motion-transitions)
fi

echo "[INFO] model_onnx=$MODEL_ONNX"
echo "[INFO] patched_onnx=$PATCHED_ONNX"
echo "[INFO] motion_file=$MOTION_FILE"
echo "[INFO] clip_name=$RESOLVED_CLIP_NAME"
echo "[INFO] geometry_obj=$GEOMETRY_OBJ"
echo "[INFO] headless=$HEADLESS"
echo "[INFO] sim_fps=$SIM_FPS sim_control_decimation=$SIM_CONTROL_DECIMATION"
echo "[INFO] checkpoint_perception=${CKPT_PERCEPTION_ENABLED:-<unknown>}/${CKPT_PERCEPTION_OUTPUT_MODE:-<unknown>}"
echo "[INFO] model_expects_perception_obs=$MODEL_EXPECTS_PERCEPTION_OBS"
echo "[INFO] logs: sim=$SIM_LOG policy=$POLICY_LOG"

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[DRY_RUN][SIM] '
  printf '%q ' "${SIM_CMD[@]}"
  printf '\n'
  printf '[DRY_RUN][POLICY] '
  printf '%q ' "${POLICY_CMD[@]}"
  printf '\n'
  exit 0
fi

"${SIM_CMD[@]}" >"$SIM_LOG" 2>&1 &
SIM_PID=$!

if ! wait_for_sim_ready; then
  exit 1
fi

if [[ "$SIM_STARTUP_WAIT" != "0" ]]; then
  sleep "$SIM_STARTUP_WAIT"
fi

set +e
if [[ "$RUN_SECONDS" == "0" ]]; then
  "${POLICY_CMD[@]}" >"$POLICY_LOG" 2>&1 &
else
  timeout --signal=INT "${RUN_SECONDS}s" "${POLICY_CMD[@]}" >"$POLICY_LOG" 2>&1 &
fi
POLICY_PID=$!
set -e

set +e
wait "$POLICY_PID"
STATUS=$?
set -e

if [[ "$STATUS" -ne 0 && "$STATUS" -ne 124 && "$STATUS" -ne 130 ]]; then
  echo "Policy run failed. See $POLICY_LOG" >&2
  tail -n 80 "$POLICY_LOG" >&2 || true
  exit "$STATUS"
fi

echo "Patched ONNX: $PATCHED_ONNX"
echo "MuJoCo log:   $SIM_LOG"
echo "Policy log:   $POLICY_LOG"
