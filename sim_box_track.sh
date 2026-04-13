#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash sim_box_track.sh [model_ref|motion_file] [motion_file|model_ref] [mj_track args...]

Defaults:
  model_ref   = wandb://zihanw22/boxer/u5lguxvl/model_17000.onnx
  motion_file = src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz

Options:
  --model-ref PATH_OR_URI   Override rollout model.
                            Supports local ONNX, wandb://..., or https://wandb.ai/.../files/...
  --motion-file PATH        Override MuJoCo init clip (.npz)
  --object-urdf PATH        Override object URDF
  --skip-setup              Fail instead of running setup scripts when envs are missing
  --no-warp                 When setup is needed, install MuJoCo without Warp
  -h, --help                Show this message

Examples:
  bash sim_box_track.sh
  bash sim_box_track.sh wandb://zihanw22/boxer/u5lguxvl/model_17000.onnx
  bash sim_box_track.sh ./local/model_17000.onnx ./data_demo/sub10_largebox_032_mj_w_obj.npz
  RUN_SECONDS=0 MJ_VIEWER=mjviser bash sim_box_track.sh -- --viewer mjviser
  DRY_RUN=1 bash sim_box_track.sh

Notes:
  - This wrapper is for sim2sim verification: it ensures inference + MuJoCo envs,
    caches remote ONNX locally, then launches mj_track.sh.
  - Additional arguments are passed through to mj_track.sh.
  - Useful env vars: RUN_SECONDS, MJ_VIEWER, SIM_READY_TIMEOUT, WANDB_MODEL_FILE, DRY_RUN.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

source "${SCRIPT_DIR}/scripts/source_common.sh"

DEFAULT_MODEL_REF="${DEFAULT_MODEL_REF:-wandb://zihanw22/boxer/u5lguxvl/model_17000.onnx}"
DEFAULT_MOTION_FILE="${DEFAULT_MOTION_FILE:-${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz}"
DEFAULT_OBJECT_URDF="${DEFAULT_OBJECT_URDF:-${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${SCRIPT_DIR}/logs/sim2sim_remote_models}"

MODEL_REF="${MODEL_REF:-${DEFAULT_MODEL_REF}}"
MOTION_FILE="${MOTION_FILE:-${DEFAULT_MOTION_FILE}}"
OBJECT_URDF="${OBJECT_URDF:-${DEFAULT_OBJECT_URDF}}"
APPLY_TRAINING_MOTION_TRANSITIONS_SET="${APPLY_TRAINING_MOTION_TRANSITIONS+x}"
SIM_MOTION_INIT_MODE_SET="${SIM_MOTION_INIT_MODE+x}"

RUN_SECONDS="${RUN_SECONDS:-0}"
MJ_VIEWER="${MJ_VIEWER:-sim_state}"
SIM_READY_TIMEOUT="${SIM_READY_TIMEOUT:-90}"
SIM_LOG_FIRST_COMMAND_SUMMARY="${SIM_LOG_FIRST_COMMAND_SUMMARY:-1}"
USE_TRAINING_SIM_CONFIG="${USE_TRAINING_SIM_CONFIG:-1}"
SIM_USE_TRAINING_URDF_OBJECT_SCENE="${SIM_USE_TRAINING_URDF_OBJECT_SCENE:-1}"
POLICY_DEFER_UNTIL_VALID_STATE="${POLICY_DEFER_UNTIL_VALID_STATE:-1}"
APPLY_TRAINING_MOTION_TRANSITIONS="${APPLY_TRAINING_MOTION_TRANSITIONS:-}"
SIM_MOTION_INIT_MODE="${SIM_MOTION_INIT_MODE:-}"
MUJOCO_OBJECT_MASS_SCALE="${MUJOCO_OBJECT_MASS_SCALE:-2.0}"
MUJOCO_OBJECT_MASS_OVERRIDE="${MUJOCO_OBJECT_MASS_OVERRIDE:-}"
MUJOCO_OBJECT_GEOM_FRICTION="${MUJOCO_OBJECT_GEOM_FRICTION:-}"
MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION="${MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION:-}"
MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES="${MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES:-1}"
MUJOCO_OBJECT_CONTACT_BODY_MARKERS="${MUJOCO_OBJECT_CONTACT_BODY_MARKERS:-[\"torso\",\"shoulder\",\"elbow\",\"wrist\",\"hand\"]}"
DRY_RUN="${DRY_RUN:-0}"

SKIP_SETUP=0
MUJOCO_SETUP_ARGS=()
MJ_TRACK_ARGS=()
POSITIONAL_MODEL_SET=0
POSITIONAL_MOTION_SET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help|help)
      usage
      exit 0
      ;;
    --skip-setup)
      SKIP_SETUP=1
      shift
      ;;
    --no-warp)
      MUJOCO_SETUP_ARGS+=(--no-warp)
      shift
      ;;
    --model-ref|--model)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] $1 requires a value." >&2
        exit 2
      fi
      MODEL_REF="$2"
      POSITIONAL_MODEL_SET=1
      shift 2
      ;;
    --motion-file)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] $1 requires a value." >&2
        exit 2
      fi
      MOTION_FILE="$2"
      POSITIONAL_MOTION_SET=1
      shift 2
      ;;
    --object-urdf)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] $1 requires a value." >&2
        exit 2
      fi
      OBJECT_URDF="$2"
      shift 2
      ;;
    --)
      shift
      MJ_TRACK_ARGS+=("$@")
      break
      ;;
    wandb://*|https://wandb.ai/*|*.onnx|*.pt|/*|./*|../*)
      if [[ "$1" == *.npz ]]; then
        MOTION_FILE="$1"
        POSITIONAL_MOTION_SET=1
      elif [[ $POSITIONAL_MODEL_SET -eq 0 ]]; then
        MODEL_REF="$1"
        POSITIONAL_MODEL_SET=1
      elif [[ $POSITIONAL_MOTION_SET -eq 0 ]]; then
        MOTION_FILE="$1"
        POSITIONAL_MOTION_SET=1
      else
        MJ_TRACK_ARGS+=("$1")
      fi
      shift
      ;;
    *.npz)
      if [[ $POSITIONAL_MOTION_SET -eq 0 ]]; then
        MOTION_FILE="$1"
        POSITIONAL_MOTION_SET=1
      else
        MJ_TRACK_ARGS+=("$1")
      fi
      shift
      ;;
    *)
      MJ_TRACK_ARGS+=("$1")
      shift
      ;;
  esac
done

python_has_modules() {
  local python_bin="$1"
  shift
  [[ -x "${python_bin}" ]] || return 1
  "${python_bin}" - "$@" <<'PY' >/dev/null 2>&1
import importlib.util
import sys

modules = sys.argv[1:]
missing = [name for name in modules if importlib.util.find_spec(name) is None]
raise SystemExit(1 if missing else 0)
PY
}

run_setup_or_fail() {
  local label="$1"
  shift
  if [[ "${SKIP_SETUP}" == "1" ]]; then
    echo "[ERROR] ${label} environment is missing or incomplete, and --skip-setup was set." >&2
    exit 1
  fi
  echo "[INFO] Building ${label} environment via: $*"
  "$@"
}

ensure_inference_env() {
  local python_bin="${CONDA_ROOT}/envs/hsinference/bin/python"
  if ! python_has_modules "${python_bin}" wandb onnx onnxruntime holosoma_inference; then
    run_setup_or_fail "inference" bash "${SCRIPT_DIR}/scripts/setup_inference.sh"
  fi
  if ! python_has_modules "${python_bin}" wandb onnx onnxruntime holosoma_inference; then
    echo "[ERROR] inference environment validation failed after setup." >&2
    exit 1
  fi
  INFER_PY="${python_bin}"
}

ensure_mujoco_env() {
  local python_bin="${CONDA_ROOT}/envs/hsmujoco/bin/python"
  if ! python_has_modules "${python_bin}" mujoco torch holosoma; then
    run_setup_or_fail "MuJoCo" bash "${SCRIPT_DIR}/scripts/setup_mujoco.sh" "${MUJOCO_SETUP_ARGS[@]}"
  fi
  if ! python_has_modules "${python_bin}" mujoco torch holosoma; then
    echo "[ERROR] MuJoCo environment validation failed after setup." >&2
    exit 1
  fi
  MUJOCO_PY="${python_bin}"
}

resolve_model_path() {
  local ref="$1"
  if [[ "${ref}" != wandb://* && "${ref}" != https://wandb.ai/* ]]; then
    "${INFER_PY}" - "${ref}" <<'PY'
from pathlib import Path
import sys

raw = sys.argv[1]
path = Path(raw).expanduser().resolve()
if path.suffix == ".pt":
    candidate = path.with_suffix(".onnx")
    if not candidate.is_file():
        raise SystemExit(f"[ERROR] Expected sibling ONNX next to checkpoint: {candidate}")
    path = candidate
if not path.is_file():
    raise SystemExit(f"[ERROR] model path not found: {path}")
print(path)
PY
    return 0
  fi

  mkdir -p "${MODEL_CACHE_DIR}"
  WANDB_SILENT=true "${INFER_PY}" - "${ref}" "${MODEL_CACHE_DIR}" <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

import wandb


def parse_wandb_ref(ref: str) -> tuple[str, str]:
    if ref.startswith("wandb://"):
        remainder = ref[len("wandb://") :]
        parts = remainder.split("/")
        if len(parts) < 4:
            raise SystemExit(
                "[ERROR] Invalid wandb URI. Expected "
                "wandb://<entity>/<project>/<run_id>/<model.onnx>"
            )
        entity, project = parts[0], parts[1]
        run_idx = 3 if len(parts) > 4 and parts[2] == "runs" else 2
        if run_idx >= len(parts):
            raise SystemExit(
                "[ERROR] Invalid wandb URI. Expected "
                "wandb://<entity>/<project>/<run_id>/<model.onnx>"
            )
        run_id = parts[run_idx]
        filename = "/".join(parts[run_idx + 1 :]).strip()
        if not filename:
            filename = os.environ.get("WANDB_MODEL_FILE", "").strip()
        if not filename:
            raise SystemExit("[ERROR] Missing ONNX filename in wandb URI.")
        return f"{entity}/{project}/{run_id}", filename

    clean = ref.split("#", 1)[0].split("?", 1)[0]
    if not clean.startswith("https://wandb.ai/"):
        raise SystemExit(f"[ERROR] Unsupported remote model ref: {ref}")
    trimmed = clean[len("https://wandb.ai/") :]
    parts = trimmed.split("/")
    if len(parts) < 4 or parts[2] != "runs":
        raise SystemExit(
            "[ERROR] Invalid W&B URL. Expected "
            "https://wandb.ai/<entity>/<project>/runs/<run_id>[/files/<model.onnx>]"
        )
    entity, project, run_id = parts[0], parts[1], parts[3]
    filename = ""
    if len(parts) >= 6 and parts[4] == "files":
        filename = "/".join(parts[5:]).strip()
    if not filename:
        filename = os.environ.get("WANDB_MODEL_FILE", "").strip()
    if not filename:
        raise SystemExit(
            "[ERROR] W&B run URL does not include /files/<model.onnx>; set WANDB_MODEL_FILE."
        )
    return f"{entity}/{project}/{run_id}", filename


ref = sys.argv[1]
cache_root = Path(sys.argv[2]).expanduser().resolve()
refresh = os.environ.get("SIM_BOX_TRACK_REFRESH_MODEL", "0").strip().lower() in {"1", "true", "yes", "on"}
run_path, filename = parse_wandb_ref(ref)
dest = cache_root / run_path / filename
dest.parent.mkdir(parents=True, exist_ok=True)

if refresh or not dest.is_file() or dest.stat().st_size == 0:
    api = wandb.Api(timeout=30)
    run = api.run(run_path)
    file_obj = run.file(filename)
    if file_obj is None:
        raise SystemExit(f"[ERROR] W&B file not found: {run_path}/{filename}")
    downloaded = file_obj.download(root=str(dest.parent), replace=True)
    downloaded_path = Path(downloaded.name)
    if not downloaded_path.is_absolute():
        downloaded_path = (dest.parent / downloaded_path).resolve()
    if downloaded_path != dest:
      dest.write_bytes(downloaded_path.read_bytes())

if dest.suffix != ".onnx":
    raise SystemExit(f"[ERROR] sim_box_track.sh expects an ONNX model, got: {dest.name}")
print(dest)
PY
}

ensure_inference_env
ensure_mujoco_env

MOTION_FILE="$("${INFER_PY}" - "${MOTION_FILE}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser().resolve()
if not path.is_file():
    raise SystemExit(f"[ERROR] motion file not found: {path}")
print(path)
PY
)"

OBJECT_URDF="$("${INFER_PY}" - "${OBJECT_URDF}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser().resolve()
if not path.is_file():
    raise SystemExit(f"[ERROR] object URDF not found: {path}")
print(path)
PY
)"

LOCAL_MODEL_PATH="$(resolve_model_path "${MODEL_REF}")"

echo "[INFO] sim2sim verification rollout"
echo "[INFO] motion file : ${MOTION_FILE}"
echo "[INFO] model ref   : ${MODEL_REF}"
echo "[INFO] model onnx  : ${LOCAL_MODEL_PATH}"
echo "[INFO] object urdf : ${OBJECT_URDF}"
echo "[INFO] viewer      : ${MJ_VIEWER}"
echo "[INFO] run seconds : ${RUN_SECONDS}"

CMD=(
  env
  MUJOCO_PY="${MUJOCO_PY}"
  INFER_PY="${INFER_PY}"
  VIEWER_PYTHON_BIN="${MUJOCO_PY}"
  MJ_VIEWER="${MJ_VIEWER}"
  RUN_SECONDS="${RUN_SECONDS}"
  SIM_READY_TIMEOUT="${SIM_READY_TIMEOUT}"
  SIM_LOG_FIRST_COMMAND_SUMMARY="${SIM_LOG_FIRST_COMMAND_SUMMARY}"
  USE_TRAINING_SIM_CONFIG="${USE_TRAINING_SIM_CONFIG}"
  SIM_USE_TRAINING_URDF_OBJECT_SCENE="${SIM_USE_TRAINING_URDF_OBJECT_SCENE}"
  POLICY_DEFER_UNTIL_VALID_STATE="${POLICY_DEFER_UNTIL_VALID_STATE}"
)

if [[ -n "${APPLY_TRAINING_MOTION_TRANSITIONS_SET}" ]]; then
  CMD+=(APPLY_TRAINING_MOTION_TRANSITIONS="${APPLY_TRAINING_MOTION_TRANSITIONS}")
fi

if [[ -n "${SIM_MOTION_INIT_MODE_SET}" ]]; then
  CMD+=(SIM_MOTION_INIT_MODE="${SIM_MOTION_INIT_MODE}")
fi

CMD+=(
  MUJOCO_OBJECT_MASS_SCALE="${MUJOCO_OBJECT_MASS_SCALE}"
  MUJOCO_OBJECT_MASS_OVERRIDE="${MUJOCO_OBJECT_MASS_OVERRIDE}"
  MUJOCO_OBJECT_GEOM_FRICTION="${MUJOCO_OBJECT_GEOM_FRICTION}"
  MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION="${MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION}"
  MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES="${MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES}"
  MUJOCO_OBJECT_CONTACT_BODY_MARKERS="${MUJOCO_OBJECT_CONTACT_BODY_MARKERS}"
  OBJECT_URDF="${OBJECT_URDF}"
  bash
  "${SCRIPT_DIR}/mj_track.sh"
  "${MOTION_FILE}"
  "${LOCAL_MODEL_PATH}"
  "${MJ_TRACK_ARGS[@]}"
)

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
  printf '[DRY_RUN]'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

exec "${CMD[@]}"
