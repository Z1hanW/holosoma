#!/usr/bin/env bash
set -euo pipefail

# MuJoCo split-sim launcher for zihanw22/boxer/9ofult64 mocap object-state student.
# The policy input is validated against the training actor obs:
# actor_obs_root + actor_obs_proprio_no_linvel + actor_obs_actions + actor_obs_box = 105 dims.

usage() {
  cat <<'EOF'
Usage:
  bash mj_mocap.sh [clip_name|motion.npz] [model.onnx|wandb://...|https://wandb.ai/...]

Defaults:
  model      = /data/logs_new/boxer/20260426_085912-g1_29dof_wbt_w_object_distill_box_mocap_sparse_root_cmd_r2s_rollout_ref_access_to_box_state-locomotion/model_13000.onnx
  motion dir = outputs/motion_bank_success_box_0_92_0p3
  clip       = box_74
  object map = <motion dir>/_clip_object_urdf_map.json

Options:
  --clip NAME          Select a clip from MOTION_DIR, e.g. box_10, box_74, box_75
  --motion-file PATH   Select an explicit .npz motion clip
  --motion-dir PATH    Directory containing prepared .npz clips
  --model-ref REF      Local ONNX/PT or W&B reference
  --object-urdf PATH   Object URDF or _clip_object_urdf_map.json
  --dry-run            Print the resolved command without launching
  -h, --help           Show this help

Examples:
  bash mj_mocap.sh box_74
  RUN_SECONDS=12 bash mj_mocap.sh --clip box_10
  DRY_RUN=1 bash mj_mocap.sh box_75
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RUN_DIR_9OFULT64="/data/logs_new/boxer/20260426_085912-g1_29dof_wbt_w_object_distill_box_mocap_sparse_root_cmd_r2s_rollout_ref_access_to_box_state-locomotion"
DEFAULT_MODEL_REF="${DEFAULT_MODEL_REF:-${RUN_DIR_9OFULT64}/model_13000.onnx}"
DEFAULT_MOTION_DIR="${DEFAULT_MOTION_DIR:-${SCRIPT_DIR}/outputs/motion_bank_success_box_0_92_0p3}"
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

MODEL_REF="${MODEL_REF:-${MODEL_PATH:-${MODEL_INPUT:-${DEFAULT_MODEL_REF}}}}"
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
api = None
run = None
refresh = os.environ.get("REFRESH_MODEL", "0").lower() in {"1", "true", "yes", "on"}
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
    raise SystemExit(f"[ERROR] mj_mocap.sh expects an ONNX model, got: {dest.name}")
print(dest)
PY
}

MODEL_LOCAL="$(resolve_model_path "${MODEL_REF}")"
MODEL_LOCAL="$(printf '%s\n' "${MODEL_LOCAL}" | tail -n 1)"

MODEL_OBS_SUMMARY="$(
  PYTHONPATH="${SCRIPT_DIR}/src/holosoma:${SCRIPT_DIR}/src/holosoma_inference${PYTHONPATH:+:${PYTHONPATH}}" \
    "${INFER_PY}" - "${MODEL_LOCAL}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import onnx

model_path = Path(sys.argv[1]).expanduser().resolve()
model = onnx.load(str(model_path))
input_dims = {}
for value in model.graph.input:
    dims = [dim.dim_value or dim.dim_param for dim in value.type.tensor_type.shape.dim]
    input_dims[value.name] = dims

obs_shape = input_dims.get("obs") or input_dims.get("actor_obs")
if obs_shape is None or len(obs_shape) < 2:
    raise SystemExit(f"[ERROR] {model_path.name} has no obs/actor_obs input")
obs_dim = int(obs_shape[1])
if obs_dim != 105:
    raise SystemExit(f"[ERROR] Observation dim mismatch: model obs={obs_dim}, expected mocap obs=105")
if "perception_obs" in input_dims:
    raise SystemExit("[ERROR] Mocap object-state student must not expose perception_obs input")

metadata = {}
for prop in model.metadata_props:
    try:
        metadata[prop.key] = json.loads(prop.value)
    except Exception:
        metadata[prop.key] = prop.value

cfg = metadata.get("experiment_config", {})
actor_input = (
    cfg.get("algo", {})
    .get("config", {})
    .get("module_dict", {})
    .get("actor", {})
    .get("input_dim")
)
expected_actor_input = [
    "actor_obs_root",
    "actor_obs_proprio_no_linvel",
    "actor_obs_actions",
    "actor_obs_box",
]
if actor_input != expected_actor_input:
    raise SystemExit(f"[ERROR] Actor input group mismatch: model={actor_input}, expected={expected_actor_input}")

expected = {
    "actor_obs_root": ["sparse_target_root_trajectory_command"],
    "actor_obs_proprio_no_linvel": ["base_ang_vel", "dof_pos", "dof_vel"],
    "actor_obs_actions": ["actions"],
    "actor_obs_box": ["obj_current_pose_size_b"],
}
groups = cfg.get("observation", {}).get("groups", {})
for group, expected_terms in expected.items():
    group_cfg = groups.get(group, {})
    if int(group_cfg.get("history_length", -1)) != 1:
        raise SystemExit(f"[ERROR] {group} history mismatch: {group_cfg.get('history_length')} != 1")
    terms_cfg = group_cfg.get("terms", {})
    terms = list(terms_cfg.keys()) if isinstance(terms_cfg, dict) else []
    if terms != expected_terms:
        raise SystemExit(f"[ERROR] {group} terms mismatch: model={terms}, expected={expected_terms}")

run_path = metadata.get("wandb_run_path", "<missing>")
print(f"obs_dim=105 run={run_path} groups={','.join(expected_actor_input)}")
PY
)"

export OBJECT_URDF="${OBJECT_URDF_RESOLVED}"
export INFERENCE_CONFIG="${INFERENCE_CONFIG:-g1-29dof-wbt-object-mocap-distill}"
export ENABLE_SPLIT_PERCEPTION_OBS="${ENABLE_SPLIT_PERCEPTION_OBS:-0}"
export RUN_SECONDS="${RUN_SECONDS:-0}"
export SIM_READY_TIMEOUT="${SIM_READY_TIMEOUT:-180}"
export USE_TRAINING_SIM_CONFIG="${USE_TRAINING_SIM_CONFIG:-1}"
export HOLOSOMA_FORCE_MOTION_ALIGNMENT="${HOLOSOMA_FORCE_MOTION_ALIGNMENT:-1}"
export HOLOSOMA_SKIP_STIFF_PROMPT="${HOLOSOMA_SKIP_STIFF_PROMPT:-1}"
export POLICY_DEFER_UNTIL_VALID_STATE="${POLICY_DEFER_UNTIL_VALID_STATE:-1}"
export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET="${HOLOSOMA_POLICY_MOTION_INDEX_OFFSET:-0}"
export USE_ROOT_REFERENCE_AT_CLIP_START="${USE_ROOT_REFERENCE_AT_CLIP_START:-0}"
export HOLOSOMA_RESET_TO_DEFAULT_POSE="${HOLOSOMA_RESET_TO_DEFAULT_POSE:-1}"
export HOLOSOMA_DEFAULT_POSE_INIT="${HOLOSOMA_DEFAULT_POSE_INIT:-1}"
export HOLOSOMA_MOTION_INIT_ROOT_POS_DELTA="${HOLOSOMA_MOTION_INIT_ROOT_POS_DELTA:-0,0,-0.002452}"
export HOLOSOMA_MOTION_INIT_ROOT_LIN_VEL="${HOLOSOMA_MOTION_INIT_ROOT_LIN_VEL:-0,0,-0.1962}"
export HOLOSOMA_W_OBJECT_URDF="${HOLOSOMA_W_OBJECT_URDF:-g1/g1_29dof.urdf}"
export HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES:-1}"
export MUJOCO_SHOW_OBJECT_COLLISION="${MUJOCO_SHOW_OBJECT_COLLISION:-0}"
export MUJOCO_HIDE_OBJECT_VISUALS_WHEN_SHOWING_COLLISION="${MUJOCO_HIDE_OBJECT_VISUALS_WHEN_SHOWING_COLLISION:-0}"

if [[ -z "${GT_MUJOCO_PHYSICS+x}" && -z "${HOLOSOMA_GT_MUJOCO_PHYSICS+x}" ]]; then
  export GT_MUJOCO_PHYSICS=0
  export HOLOSOMA_GT_MUJOCO_PHYSICS=0
fi
export SIM_USE_TRAINING_URDF_OBJECT_SCENE="${SIM_USE_TRAINING_URDF_OBJECT_SCENE:-1}"
export SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML="${SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML:-1}"
export SIM_COPY_TENDONS_FROM_ROBOT_XML="${SIM_COPY_TENDONS_FROM_ROBOT_XML:-1}"
export SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML="${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML:-1}"
export SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML="${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML:-1}"
export MUJOCO_OBJECT_MASS_OVERRIDE="${MUJOCO_OBJECT_MASS_OVERRIDE:-1.0}"
export HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS="${HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS:-0}"
export HOLOSOMA_MUJOCO_KEEP_REFERENCE_HAND_COLLISION="${HOLOSOMA_MUJOCO_KEEP_REFERENCE_HAND_COLLISION:-0}"
export HOLOSOMA_MUJOCO_CARRY_ARM_OBJECT_CONTACTS="${HOLOSOMA_MUJOCO_CARRY_ARM_OBJECT_CONTACTS:-0}"
export HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_PAIRS="${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_PAIRS:-1}"
export MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES="${MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES:-0}"
export HOLOSOMA_ZMQ_LOWCMD_LOCKSTEP_CONTROL_BOUNDARY="${HOLOSOMA_ZMQ_LOWCMD_LOCKSTEP_CONTROL_BOUNDARY:-1}"
export SIM_MOTION_INIT_MODE="${SIM_MOTION_INIT_MODE:-training_default_pose}"

CMD=(
  bash
  "${SCRIPT_DIR}/mj_track.sh"
  "${MOTION_FILE}"
  "${MODEL_LOCAL}"
  "${EXTRA_ARGS[@]}"
)

echo "[INFO] mj_mocap"
echo "[INFO] motion_file      = ${MOTION_FILE}"
echo "[INFO] object_urdf      = ${OBJECT_URDF}"
echo "[INFO] model_ref        = ${MODEL_REF}"
echo "[INFO] model_onnx       = ${MODEL_LOCAL}"
echo "[INFO] inference_config = ${INFERENCE_CONFIG}"
echo "[INFO] observation_ok   = ${MODEL_OBS_SUMMARY}"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
  printf '[DRY_RUN]'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

exec "${CMD[@]}"
