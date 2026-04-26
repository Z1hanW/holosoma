#!/usr/bin/env bash
set -euo pipefail

# MuJoCo split-sim launcher for the object-generalist box tracking teacher.
# This wrapper is intentionally thin: it resolves the clip/model/object URDF,
# sets the generalist-specific defaults, then delegates to mj_track.sh.

usage() {
  cat <<'EOF'
Usage:
  bash mj_track_generalist.sh [clip_name|motion.npz] [model.onnx|wandb://...] [mj_track args...]

Defaults:
  model      = wandb://zihanw22/boxer/u5lguxvl/latest.onnx
  motion dir = data/ds_box_data/train_g1_w_obj_prepared
  clip       = box_74
  object map = <motion dir>/_clip_object_urdf_map.json

Options:
  --clip NAME          Select a clip from MOTION_DIR, e.g. box_74
  --motion-file PATH   Select an explicit .npz motion clip
  --motion-dir PATH    Directory containing prepared .npz clips
  --model-ref REF      Local ONNX/PT or wandb:// / https://wandb.ai reference
  --object-urdf PATH   Object URDF or _clip_object_urdf_map.json
  --dry-run            Print the resolved command without launching
  -h, --help           Show this help

Useful env vars:
  MOTION_CLIP_NAME, MOTION_FILE, MOTION_DIR, MODEL_REF, OBJECT_URDF
  RUN_SECONDS, MJ_VIEWER, SIM_READY_TIMEOUT, WANDB_MODEL_FILE

Examples:
  bash mj_track_generalist.sh
  bash mj_track_generalist.sh box_74
  bash mj_track_generalist.sh box_74 wandb://zihanw22/boxer/u5lguxvl/latest.onnx
  RUN_SECONDS=20 bash mj_track_generalist.sh --clip box_74
  MJ_VIEWER=mjviser bash mj_track_generalist.sh box_74 -- --viewer mjviser
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DEFAULT_MODEL_REF="${DEFAULT_MODEL_REF:-wandb://zihanw22/boxer/u5lguxvl/latest.onnx}"
DEFAULT_MOTION_DIR="${DEFAULT_MOTION_DIR:-${SCRIPT_DIR}/data/ds_box_data/train_g1_w_obj_prepared}"
DEFAULT_CLIP_NAME="${DEFAULT_CLIP_NAME:-box_74}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${SCRIPT_DIR}/logs/sim2sim_remote_models}"
INFER_PY="${INFER_PY:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python}"

if [[ ! -x "${INFER_PY}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    INFER_PY="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    INFER_PY="$(command -v python)"
  else
    echo "[ERROR] No usable Python found. Set INFER_PY explicitly." >&2
    exit 1
  fi
fi

MODEL_REF="${MODEL_REF:-${MODEL_PATH:-${DEFAULT_MODEL_REF}}}"
MOTION_DIR="${MOTION_DIR:-${DEFAULT_MOTION_DIR}}"
MOTION_FILE="${MOTION_FILE:-}"
MOTION_CLIP_NAME="${MOTION_CLIP_NAME:-${MOTION_CLIP:-${DEFAULT_CLIP_NAME}}}"
OBJECT_URDF_INPUT="${OBJECT_URDF:-}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS=()
POSITIONAL_MODE=1

is_model_ref() {
  local value="${1:-}"
  [[ "${value}" == wandb://* || "${value}" == https://wandb.ai/* || "${value}" == *.onnx || "${value}" == *.pt ]]
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help|help)
      usage
      exit 0
      ;;
    --clip|--motion-clip|--motion-clip-name)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires a value." >&2; exit 2; }
      MOTION_CLIP_NAME="$2"
      MOTION_FILE=""
      shift 2
      ;;
    --motion-file)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires a value." >&2; exit 2; }
      MOTION_FILE="$2"
      shift 2
      ;;
    --motion-dir)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires a value." >&2; exit 2; }
      MOTION_DIR="$2"
      shift 2
      ;;
    --model-ref|--model|--model-path)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires a value." >&2; exit 2; }
      MODEL_REF="$2"
      shift 2
      ;;
    --object-urdf|--object-map)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires a value." >&2; exit 2; }
      OBJECT_URDF_INPUT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -*)
      POSITIONAL_MODE=0
      EXTRA_ARGS+=("$1")
      shift
      ;;
    *)
      if [[ "${POSITIONAL_MODE}" == "1" && "$1" == *.npz ]]; then
        MOTION_FILE="$1"
        shift
      elif [[ "${POSITIONAL_MODE}" == "1" ]] && is_model_ref "$1"; then
        MODEL_REF="$1"
        shift
      elif [[ "${POSITIONAL_MODE}" == "1" && -z "${MOTION_FILE}" && -n "$1" ]]; then
        MOTION_CLIP_NAME="${1%.npz}"
        shift
      else
        POSITIONAL_MODE=0
        EXTRA_ARGS+=("$1")
        shift
      fi
      ;;
  esac
done

if [[ -z "${MOTION_FILE}" ]]; then
  MOTION_FILE="${MOTION_DIR%/}/${MOTION_CLIP_NAME%.npz}.npz"
fi

MOTION_FILE="$("${INFER_PY}" - "${MOTION_FILE}" <<'PY'
import os
from pathlib import Path
import sys

path = Path(os.path.abspath(os.path.expanduser(sys.argv[1])))
if not path.is_file():
    raise SystemExit(f"[ERROR] motion file not found: {path}")
print(path)
PY
)"

if [[ -z "${OBJECT_URDF_INPUT}" ]]; then
  OBJECT_URDF_INPUT="${MOTION_DIR%/}/_clip_object_urdf_map.json"
fi

OBJECT_URDF_RESOLVED="$("${INFER_PY}" - "${OBJECT_URDF_INPUT}" "${MOTION_FILE}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

raw = sys.argv[1]
motion_path = Path(sys.argv[2]).expanduser().resolve()
stem = motion_path.stem
candidate = Path(raw).expanduser() if raw else None

if candidate is not None and candidate.is_file() and candidate.suffix.lower() == ".json":
    data = json.loads(candidate.read_text())
    clips = data.get("clips", data) if isinstance(data, dict) else {}
    entry = clips.get(stem) if isinstance(clips, dict) else None
    if not isinstance(entry, dict):
        raise SystemExit(f"[ERROR] Object map has no entry for clip '{stem}': {candidate}")
    path = entry.get("object_urdf_path") or entry.get("urdf_path")
    if not path:
        raise SystemExit(f"[ERROR] Object map entry for '{stem}' has no object_urdf_path")
    print(Path(path).expanduser().resolve())
elif candidate is not None and str(candidate):
    if not candidate.is_file():
        raise SystemExit(f"[ERROR] object URDF/map not found: {candidate.expanduser().resolve()}")
    print(candidate.expanduser().resolve())
else:
    with np.load(motion_path, allow_pickle=True) as data:
        if "object_urdf_path" not in data:
            raise SystemExit(f"[ERROR] No object map provided and motion has no object_urdf_path: {motion_path}")
        print(Path(str(np.asarray(data["object_urdf_path"]).item())).expanduser().resolve())
PY
)"

resolve_model_path() {
  local ref="$1"
  if [[ "${ref}" != wandb://* && "${ref}" != https://wandb.ai/* ]]; then
    "${INFER_PY}" - "${ref}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser().resolve()
if path.suffix == ".pt":
    candidate = path.with_suffix(".onnx")
    if not candidate.is_file():
        raise SystemExit(f"[ERROR] Expected sibling ONNX next to checkpoint: {candidate}")
    path = candidate
if not path.is_file():
    raise SystemExit(f"[ERROR] model path not found: {path}")
print(path)
PY
    return
  fi

  mkdir -p "${MODEL_CACHE_DIR}"
  WANDB_SILENT=true "${INFER_PY}" - "${ref}" "${MODEL_CACHE_DIR}" <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

import wandb


LATEST_SENTINEL = "__LATEST_ONNX__"


def parse_ref(ref: str) -> tuple[str, str]:
    if ref.startswith("wandb://"):
        parts = ref[len("wandb://") :].split("/")
        if len(parts) < 3:
            raise SystemExit("[ERROR] Expected wandb://<entity>/<project>/<run_id>/<model.onnx>")
        run_idx = 3 if len(parts) > 4 and parts[2] == "runs" else 2
        entity, project, run_id = parts[0], parts[1], parts[run_idx]
        filename = "/".join(parts[run_idx + 1 :]).strip()
        if not filename:
            filename = os.environ.get("WANDB_MODEL_FILE", "").strip()
        if not filename or filename.lower() in {"latest", "latest.onnx"}:
            filename = LATEST_SENTINEL
        return f"{entity}/{project}/{run_id}", filename

    clean = ref.split("#", 1)[0].split("?", 1)[0]
    if not clean.startswith("https://wandb.ai/"):
        raise SystemExit(f"[ERROR] Unsupported remote model reference: {ref}")
    parts = clean[len("https://wandb.ai/") :].split("/")
    if len(parts) < 4 or parts[2] != "runs":
        raise SystemExit("[ERROR] Expected https://wandb.ai/<entity>/<project>/runs/<run_id>[/files/<model.onnx>]")
    entity, project, run_id = parts[0], parts[1], parts[3]
    filename = ""
    if len(parts) >= 6 and parts[4] == "files":
        filename = "/".join(parts[5:]).strip()
    elif len(parts) >= 5:
        filename = "/".join(parts[4:]).strip()
    if not filename:
        filename = os.environ.get("WANDB_MODEL_FILE", "").strip()
    if not filename or filename.lower() in {"latest", "latest.onnx"}:
        filename = LATEST_SENTINEL
    return f"{entity}/{project}/{run_id}", filename


ref = sys.argv[1]
cache_root = Path(sys.argv[2]).expanduser().resolve()
run_path, filename = parse_ref(ref)
refresh = os.environ.get("REFRESH_MODEL", "0").lower() in {"1", "true", "yes", "on"}
api = None
run = None
if filename == LATEST_SENTINEL:
    api = wandb.Api(timeout=30)
    run = api.run(run_path)
    onnx_files = [file_obj for file_obj in run.files() if file_obj.name.endswith(".onnx")]
    if not onnx_files:
        raise SystemExit(f"[ERROR] No ONNX files found for W&B run: {run_path}")
    latest_file = max(onnx_files, key=lambda file_obj: ((file_obj.updated_at or ""), file_obj.name))
    filename = latest_file.name

dest = cache_root / run_path / filename
dest.parent.mkdir(parents=True, exist_ok=True)

if refresh or not dest.is_file() or dest.stat().st_size == 0:
    if run is None:
        api = wandb.Api(timeout=30)
        run = api.run(run_path)
    file_obj = run.file(filename)
    downloaded = file_obj.download(root=str(dest.parent), replace=True)
    downloaded_path = Path(downloaded.name)
    if not downloaded_path.is_absolute():
        downloaded_path = (dest.parent / downloaded_path).resolve()
    if downloaded_path != dest:
        dest.write_bytes(downloaded_path.read_bytes())

if dest.suffix != ".onnx":
    raise SystemExit(f"[ERROR] mj_track_generalist.sh expects an ONNX model, got: {dest.name}")
print(dest)
PY
}

MODEL_LOCAL="$(resolve_model_path "${MODEL_REF}")"
MODEL_LOCAL="$(printf '%s\n' "${MODEL_LOCAL}" | tail -n 1)"

export OBJECT_URDF="${OBJECT_URDF_RESOLVED}"
export INFERENCE_CONFIG="${INFERENCE_CONFIG:-g1-29dof-wbt-w-object}"
export ENABLE_SPLIT_PERCEPTION_OBS="${ENABLE_SPLIT_PERCEPTION_OBS:-0}"
export RUN_SECONDS="${RUN_SECONDS:-0}"
export SIM_READY_TIMEOUT="${SIM_READY_TIMEOUT:-180}"
export USE_TRAINING_SIM_CONFIG="${USE_TRAINING_SIM_CONFIG:-1}"
export SIM_USE_TRAINING_URDF_OBJECT_SCENE="${SIM_USE_TRAINING_URDF_OBJECT_SCENE:-}"
export HOLOSOMA_FORCE_MOTION_ALIGNMENT="${HOLOSOMA_FORCE_MOTION_ALIGNMENT:-1}"
export HOLOSOMA_W_OBJECT_URDF="${HOLOSOMA_W_OBJECT_URDF:-g1/g1_29dof.urdf}"
export HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES:-1}"
export MUJOCO_SHOW_OBJECT_COLLISION="${MUJOCO_SHOW_OBJECT_COLLISION:-0}"
export MUJOCO_HIDE_OBJECT_VISUALS_WHEN_SHOWING_COLLISION="${MUJOCO_HIDE_OBJECT_VISUALS_WHEN_SHOWING_COLLISION:-0}"

MODEL_OBS_SUMMARY="$(
  PYTHONPATH="${SCRIPT_DIR}/src/holosoma:${SCRIPT_DIR}/src/holosoma_inference${PYTHONPATH:+:${PYTHONPATH}}" \
    "${INFER_PY}" - "${MODEL_LOCAL}" "${INFERENCE_CONFIG}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import onnx

from holosoma_inference.config.config_values.inference import DEFAULTS


model_path = Path(sys.argv[1]).expanduser().resolve()
inference_name = sys.argv[2]
if inference_name not in DEFAULTS:
    raise SystemExit(f"[ERROR] Unknown inference config for validation: {inference_name}")

cfg = DEFAULTS[inference_name]
obs_cfg = cfg.observation
actor_terms = list(obs_cfg.obs_dict.get("actor_obs", ()))
if not actor_terms:
    raise SystemExit(f"[ERROR] Inference config '{inference_name}' has no actor_obs terms")
history_length = int(obs_cfg.history_length_dict.get("actor_obs", 1))
expected_dim = sum(int(obs_cfg.obs_dims[term]) for term in actor_terms) * history_length

model = onnx.load(str(model_path))
input_shape = model.graph.input[0].type.tensor_type.shape.dim
input_dim = int(input_shape[-1].dim_value)
if input_dim != expected_dim:
    raise SystemExit(
        f"[ERROR] Observation dim mismatch for {model_path.name}: model obs={input_dim}, "
        f"inference:{inference_name} expects actor_obs={expected_dim}"
    )

metadata = {}
for prop in model.metadata_props:
    try:
        metadata[prop.key] = json.loads(prop.value)
    except Exception:
        metadata[prop.key] = prop.value

groups = metadata.get("experiment_config", {}).get("observation", {}).get("groups", {})
model_actor = groups.get("actor_obs", {}) if isinstance(groups, dict) else {}
model_terms = list(model_actor.get("terms", {}).keys()) if isinstance(model_actor.get("terms"), dict) else []
model_history_length = int(model_actor.get("history_length", history_length)) if isinstance(model_actor, dict) else history_length

if model_terms and model_terms != actor_terms:
    raise SystemExit(
        f"[ERROR] Observation term mismatch for {model_path.name}: "
        f"model terms={model_terms}, inference:{inference_name} terms={actor_terms}"
    )
if model_history_length != history_length:
    raise SystemExit(
        f"[ERROR] Observation history mismatch for {model_path.name}: "
        f"model history={model_history_length}, inference:{inference_name} history={history_length}"
    )

print(
    f"actor_obs_dim={input_dim} history={history_length} "
    f"terms={','.join(actor_terms)}"
)
PY
)"

CMD=(
  bash
  "${SCRIPT_DIR}/mj_track.sh"
  "${MOTION_FILE}"
  "${MODEL_LOCAL}"
  "${EXTRA_ARGS[@]}"
)

echo "[INFO] mj_track_generalist"
echo "[INFO] motion_file      = ${MOTION_FILE}"
echo "[INFO] object_urdf      = ${OBJECT_URDF}"
echo "[INFO] model_ref        = ${MODEL_REF}"
echo "[INFO] model_onnx       = ${MODEL_LOCAL}"
echo "[INFO] inference_config = ${INFERENCE_CONFIG}"
echo "[INFO] observation_ok   = ${MODEL_OBS_SUMMARY}"
echo "[INFO] split_perception = ${ENABLE_SPLIT_PERCEPTION_OBS}"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
  printf '[DRY_RUN]'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

exec "${CMD[@]}"
