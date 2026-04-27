#!/usr/bin/env bash
set -euo pipefail

# Mocap-access box policy inference for run 9ofult64.
#
# This is a thin wrapper around infer_box_joystick.sh mocap. It pins inference
# to the same 28-clip rollout-ref motion bank used by the training run.
#
# Usage:
#   bash infer_mocap.sh [checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra infer_box_joystick.sh args...]
#
# Examples:
#   bash infer_mocap.sh
#   DRY_RUN=1 bash infer_mocap.sh
#   bash infer_mocap.sh --command.setup_terms.motion_command.params.motion_config.motion_clip_name box_74

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RUN_URL_DEFAULT="https://wandb.ai/zihanw22/boxer/runs/9ofult64"
RUN_DIR_DEFAULT="/data/logs_new/boxer/20260426_085912-g1_29dof_wbt_w_object_distill_box_mocap_sparse_root_cmd_r2s_rollout_ref_access_to_box_state-locomotion"
MOTION_DIR_DEFAULT="${SCRIPT_DIR}/outputs/motion_bank_success_box_0_92_0p3"
OBJECT_MAP_DEFAULT="${MOTION_DIR_DEFAULT}/_clip_object_urdf_map.json"
EXPECTED_MOTION_COUNT="${EXPECTED_MOTION_COUNT:-28}"

is_checkpoint_ref() {
  local ref="$1"
  [[ "${ref}" == wandb://* || "${ref}" == https://wandb.ai/*/runs/* || "${ref}" == /* || "${ref}" == ./* || "${ref}" == ../* || "${ref}" == *.pt ]]
}

latest_local_model_file() {
  local run_dir="$1"
  ls -1 "${run_dir}"/model_*.pt 2>/dev/null | sort -V | tail -n 1 | xargs -r basename
}

validate_training_bank() {
  local motion_dir="$1"
  local object_map="$2"
  local expected_count="$3"

  if [[ ! -d "${motion_dir}" ]]; then
    echo "[ERROR] training motion bank not found: ${motion_dir}" >&2
    exit 1
  fi
  if [[ ! -f "${object_map}" ]]; then
    echo "[ERROR] training object map not found: ${object_map}" >&2
    exit 1
  fi

  if [[ "${expected_count}" == "0" || -z "${expected_count}" ]]; then
    return 0
  fi

  "${PYTHON_BIN:-python}" - "${motion_dir}" "${object_map}" "${expected_count}" <<'PY'
import json
import sys
from pathlib import Path

motion_dir = Path(sys.argv[1])
object_map = Path(sys.argv[2])
expected_count = int(sys.argv[3])

clips = sorted(path.stem for path in motion_dir.glob("*.npz"))
payload = json.loads(object_map.read_text(encoding="utf-8"))
mapped = payload.get("clips", payload) if isinstance(payload, dict) else {}
if not isinstance(mapped, dict):
    raise SystemExit(f"[ERROR] invalid object map format: {object_map}")

missing = [clip_id for clip_id in clips if clip_id not in mapped]
if len(clips) != expected_count or len(mapped) != expected_count or missing:
    preview = ", ".join(missing[:8])
    raise SystemExit(
        "[ERROR] training bank mismatch: "
        f"npz={len(clips)} map={len(mapped)} expected={expected_count} "
        f"missing_in_map=[{preview}]"
    )
PY
}

CKPT="${CKPT:-${CHECKPOINT:-${TEACHER_CHECKPOINT:-${MOCAP_CHECKPOINT:-${RUN_URL_DEFAULT}}}}}"
if [[ $# -gt 0 ]] && is_checkpoint_ref "$1"; then
  CKPT="$1"
  shift
fi

MOTION_DIR="${MOTION_DIR:-${MOTION_DIR_DEFAULT}}"
OBJECT_URDF="${OBJECT_URDF:-${OBJECT_MAP_DEFAULT}}"
TEACHER_ROLLOUT_FILTERED_MOTION_DIR="${TEACHER_ROLLOUT_FILTERED_MOTION_DIR:-${MOTION_DIR}}"
TEACHER_ROLLOUT_MOTION_DIR="${TEACHER_ROLLOUT_MOTION_DIR:-${MOTION_DIR}}"
TEACHER_ROLLOUT_FILTER_ENABLED="${TEACHER_ROLLOUT_FILTER_ENABLED:-True}"
INFER_DATASET="${INFER_DATASET:-rollout-ref}"
MOCAP_PERCEPTION_PRESET="${MOCAP_PERCEPTION_PRESET:-checkpoint}"
OBJECT_GEOMETRY_MODE="${OBJECT_GEOMETRY_MODE:-primitive}"
VISER_MANUAL_CONTROL_DEFAULT="${VISER_MANUAL_CONTROL_DEFAULT:-0}"
VISER_FORCE_MANUAL_CONTROL="${VISER_FORCE_MANUAL_CONTROL:-0}"

# 9ofult64 was trained with large multi-env PhysX GPU buffers. Loading that
# checkpoint config directly for 1-env inference can allocate gigabytes for
# contact-pair buffers before the first rollout step.
PHYSX_GPU_MAX_RIGID_CONTACT_COUNT="${PHYSX_GPU_MAX_RIGID_CONTACT_COUNT:-1048576}"
PHYSX_GPU_MAX_RIGID_PATCH_COUNT="${PHYSX_GPU_MAX_RIGID_PATCH_COUNT:-65536}"
PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY="${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY:-1048576}"
PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY="${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY:-1048576}"
PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY="${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY:-262144}"
PHYSX_GPU_COLLISION_STACK_SIZE="${PHYSX_GPU_COLLISION_STACK_SIZE:-67108864}"
PHYSX_GPU_HEAP_CAPACITY="${PHYSX_GPU_HEAP_CAPACITY:-67108864}"
PHYSX_GPU_TEMP_BUFFER_CAPACITY="${PHYSX_GPU_TEMP_BUFFER_CAPACITY:-16777216}"

if [[ -z "${WANDB_MODEL_FILE+x}" || -z "${WANDB_MODEL_FILE}" ]]; then
  latest_model_file="$(latest_local_model_file "${RUN_DIR_DEFAULT}")"
  if [[ -n "${latest_model_file}" ]]; then
    WANDB_MODEL_FILE="${latest_model_file}"
  fi
fi

export MOTION_DIR
export OBJECT_URDF
export TEACHER_ROLLOUT_FILTERED_MOTION_DIR
export TEACHER_ROLLOUT_MOTION_DIR
export TEACHER_ROLLOUT_FILTER_ENABLED
export INFER_DATASET
export MOCAP_PERCEPTION_PRESET
export OBJECT_GEOMETRY_MODE
export VISER_MANUAL_CONTROL_DEFAULT
export VISER_FORCE_MANUAL_CONTROL
export MOCAP_CHECKPOINT_DEFAULT="${CKPT}"
export WANDB_MODEL_FILE="${WANDB_MODEL_FILE:-}"
export PHYSX_GPU_MAX_RIGID_CONTACT_COUNT
export PHYSX_GPU_MAX_RIGID_PATCH_COUNT
export PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY
export PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY
export PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY
export PHYSX_GPU_COLLISION_STACK_SIZE
export PHYSX_GPU_HEAP_CAPACITY
export PHYSX_GPU_TEMP_BUFFER_CAPACITY

validate_training_bank "${MOTION_DIR}" "${OBJECT_URDF}" "${EXPECTED_MOTION_COUNT}"

echo "[INFO] checkpoint=${CKPT}"
echo "[INFO] motion_dir=${MOTION_DIR}"
echo "[INFO] object_map=${OBJECT_URDF}"
echo "[INFO] expected_motion_count=${EXPECTED_MOTION_COUNT}"
echo "[INFO] physx_gpu_max_rigid_contact_count=${PHYSX_GPU_MAX_RIGID_CONTACT_COUNT}"
echo "[INFO] physx_gpu_found_lost_pairs_capacity=${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY}"
echo "[INFO] physx_gpu_collision_stack_size=${PHYSX_GPU_COLLISION_STACK_SIZE}"
if [[ -n "${WANDB_MODEL_FILE}" ]]; then
  echo "[INFO] wandb_model_file=${WANDB_MODEL_FILE}"
fi

exec bash "${SCRIPT_DIR}/infer_box_joystick.sh" mocap "${CKPT}" "$@"
