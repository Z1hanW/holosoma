#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_SOURCE_BANK="carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout"
SOURCE_AS_DATA_DIR="${SOURCE_AS_DATA_DIR:-${SCRIPT_DIR}/data/ds_as_data/${DEFAULT_SOURCE_BANK}}"
SOURCE_AS_OBJECT_MAP="${SOURCE_AS_OBJECT_MAP:-${SOURCE_AS_DATA_DIR}/_clip_object_urdf_map.json}"
SOURCE_EXPECTED_TOTAL="${SOURCE_EXPECTED_TOTAL:-195}"

TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-https://wandb.ai/zihanw22/carry-any/runs/bcleb5oi/files/model_67000.pt}"
RUN_ID="${RUN_ID:-as_teacher_realmesh_rollout_$(date -u +%Y%m%d_%H%M%S)_bcleb5oi_model67000_box_bin_barrel_ball}"
SHARD_ROOT="${SHARD_ROOT:-${SCRIPT_DIR}/data/ds_as_data/_teacher_rollout_shards/${RUN_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/teacher_as_contacts/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs/runtime/${RUN_ID}}"
TMP_ROOT="${TMP_ROOT:-/data/tmp/holosoma_realmesh_rollout/${RUN_ID}}"
REALMESH_BANK_NAME="${REALMESH_BANK_NAME:-${DEFAULT_SOURCE_BANK}_realmesh_rollout_bcleb5oi67000_box_bin_barrel_ball}"
TARGET_BANK="${TARGET_BANK:-${SCRIPT_DIR}/data/ds_as_data/${REALMESH_BANK_NAME}}"
CONTACT_EXPORT_NAME="${CONTACT_EXPORT_NAME:-contact_export_from_teacher_realmesh_rollout}"

ALLOWED_CATEGORIES="${ALLOWED_CATEGORIES:-box,ball,bin,barrel}"
REALMESH_EXCLUDE_CLIPS="${REALMESH_EXCLUDE_CLIPS:-scale__any_bin_3,scale__any_bin_8,box_21,box_39}"
SUCCESS_POSITION_THRESHOLD="${SUCCESS_POSITION_THRESHOLD:-0.5}"
MIN_CONTACT_FRAMES="${MIN_CONTACT_FRAMES:-10}"
CONTACT_FORCE_THRESHOLD="${CONTACT_FORCE_THRESHOLD:-1.0}"
CONTACT_VOXEL_SIZE="${CONTACT_VOXEL_SIZE:-0.01}"
FOOT_OBJECT_CONTACT_FORCE_THRESHOLD="${FOOT_OBJECT_CONTACT_FORCE_THRESHOLD:-1.0}"
MIDDLE_FOOT_CONTACT_START_FRAC="${MIDDLE_FOOT_CONTACT_START_FRAC:-0.20}"
MIDDLE_FOOT_CONTACT_END_FRAC="${MIDDLE_FOOT_CONTACT_END_FRAC:-0.80}"

HEADLESS="${HEADLESS:-True}"
DISABLE_RANDOMIZATION="${DISABLE_RANDOMIZATION:-True}"
START_AT_TIMESTEP_ZERO_PROB="${START_AT_TIMESTEP_ZERO_PROB:-1.0}"
FREEZE_AT_TIMESTEP_ZERO_PROB="${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}"
RESET_NOISE_SCALE="${RESET_NOISE_SCALE:-0.0}"
USE_ADAPTIVE_TIMESTEPS_SAMPLER="${USE_ADAPTIVE_TIMESTEPS_SAMPLER:-False}"
MAX_EPISODE_LENGTH_S="${MAX_EPISODE_LENGTH_S:-1000000}"
PHYSX_GPU_COLLISION_STACK_SIZE="${PHYSX_GPU_COLLISION_STACK_SIZE:-268435456}"
LOCAL_RANK_OFFSET="${LOCAL_RANK_OFFSET:-40}"
PER_GPU_ENVS="${PER_GPU_ENVS:-0}"
FORCE="${FORCE:-1}"
SAVE_OBJECT_FRAME_VIS="${SAVE_OBJECT_FRAME_VIS:-1}"
SAVE_SHARD_GLB="${SAVE_SHARD_GLB:-0}"
SAVE_SHARD_PREVIEW_PNG="${SAVE_SHARD_PREVIEW_PNG:-0}"
SAVE_SHARD_FACE_HEATMAP_PNG="${SAVE_SHARD_FACE_HEATMAP_PNG:-0}"
LAUNCH_VISER="${LAUNCH_VISER:-1}"
VIEWER_BACKGROUND="${VIEWER_BACKGROUND:-1}"
VISER_HOST="${VISER_HOST:-0.0.0.0}"
VISER_PORT="${VISER_PORT:-$((RANDOM % 8976 + 1024))}"
VIEWER_LOG="${VIEWER_LOG:-${LOG_ROOT}/realmesh_rollout_viewer.log}"
DRY_RUN="${DRY_RUN:-0}"

is_truthy() {
  case "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

detect_gpu_list() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local detected
    detected="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)"
    if [[ -n "${detected}" ]]; then
      echo "${detected}"
      return 0
    fi
  fi
  echo "0"
}

GPU_LIST="${GPU_LIST:-$(detect_gpu_list)}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
NUM_SHARDS="${NUM_SHARDS:-${#GPUS[@]}}"
if [[ "${#GPUS[@]}" -ne "${NUM_SHARDS}" ]]; then
  echo "[ERROR] GPU_LIST must contain NUM_SHARDS entries. GPU_LIST=${GPU_LIST} NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 2
fi

mkdir -p "${SHARD_ROOT}" "${OUTPUT_ROOT}/shards" "${LOG_ROOT}" "${TMP_ROOT}"
export TMPDIR="${TMP_ROOT}"
export TMP="${TMP_ROOT}"
export TEMP="${TMP_ROOT}"

prepare_args=(
  "${PYTHON_BIN}" scripts/prepare_teacher_as_realmesh_rollout.py prepare-shards
  --source-bank "${SOURCE_AS_DATA_DIR}"
  --source-map "${SOURCE_AS_OBJECT_MAP}"
  --shard-root "${SHARD_ROOT}"
  --num-shards "${NUM_SHARDS}"
  --per-gpu-envs "${PER_GPU_ENVS}"
  --expected-total "${SOURCE_EXPECTED_TOTAL}"
  --allowed-categories "${ALLOWED_CATEGORIES}"
  --exclude-clips "${REALMESH_EXCLUDE_CLIPS}"
)
"${prepare_args[@]}" > "${SHARD_ROOT}/prepare_stdout.json"
PER_GPU_ENVS="$("${PYTHON_BIN}" - "${SHARD_ROOT}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text())["per_gpu_envs"])
PY
)"
SELECTED_CLIPS="$("${PYTHON_BIN}" - "${SHARD_ROOT}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text())["selected_clip_count"])
PY
)"

echo "[INFO] run_id=${RUN_ID}"
echo "[INFO] teacher_checkpoint=${TEACHER_CHECKPOINT}"
echo "[INFO] source_bank=${SOURCE_AS_DATA_DIR}"
echo "[INFO] source_object_map=${SOURCE_AS_OBJECT_MAP}"
echo "[INFO] allowed_categories=${ALLOWED_CATEGORIES}"
echo "[INFO] excluded_clips=${REALMESH_EXCLUDE_CLIPS}"
echo "[INFO] selected_clips=${SELECTED_CLIPS}"
echo "[INFO] num_shards=${NUM_SHARDS}"
echo "[INFO] gpu_list=${GPU_LIST}"
echo "[INFO] per_gpu_envs=${PER_GPU_ENVS}"
echo "[INFO] shard_root=${SHARD_ROOT}"
echo "[INFO] output_root=${OUTPUT_ROOT}"
echo "[INFO] tmp_root=${TMP_ROOT}"
echo "[INFO] target_bank=${TARGET_BANK}"

if is_truthy "${DRY_RUN}"; then
  for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_name="$(printf 'shard_%02d' "${shard_idx}")"
    shard_dir="${SHARD_ROOT}/${shard_name}"
    shard_output="${OUTPUT_ROOT}/shards/${shard_name}"
    gpu="${GPUS[${shard_idx}]}"
    local_rank=$((LOCAL_RANK_OFFSET + shard_idx))
    expected_count="$(wc -l < "${shard_dir}/clip_ids.txt" | tr -d ' ')"
    echo "[DRY_RUN] ${shard_name}: gpu=${gpu} local_rank=${local_rank} envs=${PER_GPU_ENVS} clips=${expected_count}"
    echo "[DRY_RUN] AS_DATA_DIR=${shard_dir}"
    echo "[DRY_RUN] OUTPUT_DIR=${shard_output}"
    echo "[DRY_RUN] TMPDIR=${TMP_ROOT}/${shard_name}"
    echo "[DRY_RUN] REAL_MESH_OBJECT_SPAWN=1"
    echo "[DRY_RUN] HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=0"
    printf '[DRY_RUN] command:'
    printf ' %q' bash ./infer_teacher_as_contacts.sh "${TEACHER_CHECKPOINT}" \
      --require-final-position-success-for-success \
      --require-no-middle-foot-object-contact-for-success \
      --middle-foot-contact-start-frac "${MIDDLE_FOOT_CONTACT_START_FRAC}" \
      --middle-foot-contact-end-frac "${MIDDLE_FOOT_CONTACT_END_FRAC}" \
      --foot-object-contact-force-threshold "${FOOT_OBJECT_CONTACT_FORCE_THRESHOLD}"
    if ! is_truthy "${SAVE_SHARD_GLB}"; then
      printf ' %q' --no-save-glb
    fi
    if ! is_truthy "${SAVE_SHARD_PREVIEW_PNG}"; then
      printf ' %q' --no-save-preview-png
    fi
    if ! is_truthy "${SAVE_SHARD_FACE_HEATMAP_PNG}"; then
      printf ' %q' --no-save-face-heatmap-png
    fi
    printf '\n'
  done
  echo "[DRY_RUN] merge_target_bank=${TARGET_BANK}"
  exit 0
fi

declare -a PIDS=()
declare -a SHARD_LOGS=()

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_name="$(printf 'shard_%02d' "${shard_idx}")"
  shard_dir="${SHARD_ROOT}/${shard_name}"
  shard_output="${OUTPUT_ROOT}/shards/${shard_name}"
  shard_log="${LOG_ROOT}/${shard_name}.log"
  gpu="${GPUS[${shard_idx}]}"
  local_rank=$((LOCAL_RANK_OFFSET + shard_idx))
  expected_count="$(wc -l < "${shard_dir}/clip_ids.txt" | tr -d ' ')"
  SHARD_LOGS+=("${shard_log}")

  echo "[INFO] Starting ${shard_name}: gpu=${gpu} local_rank=${local_rank} envs=${PER_GPU_ENVS} clips=${expected_count}"
  (
    set -euo pipefail
    cd "${SCRIPT_DIR}"
    export HOLOSOMA_DEVICE="cuda:${gpu}"
    export LOCAL_RANK="${local_rank}"
    export AS_DATA_DIR="${shard_dir}"
    export AS_OBJECT_MAP="${shard_dir}/_clip_object_urdf_map.json"
    export AS_EXPECTED_TOTAL="${expected_count}"
    export AS_SINGLE_SLOT_MOTION_DIR="${shard_dir}/_single_slot_motion_bank"
    export NUM_ENVS="${PER_GPU_ENVS}"
    export HEADLESS="${HEADLESS}"
    export OUTPUT_DIR="${shard_output}"
    export SUCCESS_POSITION_THRESHOLD="${SUCCESS_POSITION_THRESHOLD}"
    export MIN_CONTACT_FRAMES="${MIN_CONTACT_FRAMES}"
    export CONTACT_FORCE_THRESHOLD="${CONTACT_FORCE_THRESHOLD}"
    export CONTACT_VOXEL_SIZE="${CONTACT_VOXEL_SIZE}"
    export PHYSX_GPU_COLLISION_STACK_SIZE="${PHYSX_GPU_COLLISION_STACK_SIZE}"
    export PUBLISH_FOR_INFER_BOX=0
    export LAUNCH_VISER=0
    export VALIDATE_OUTPUT_FORMAT=1
    export DISABLE_RANDOMIZATION="${DISABLE_RANDOMIZATION}"
    export START_AT_TIMESTEP_ZERO_PROB="${START_AT_TIMESTEP_ZERO_PROB}"
    export FREEZE_AT_TIMESTEP_ZERO_PROB="${FREEZE_AT_TIMESTEP_ZERO_PROB}"
    export RESET_NOISE_SCALE="${RESET_NOISE_SCALE}"
    export USE_ADAPTIVE_TIMESTEPS_SAMPLER="${USE_ADAPTIVE_TIMESTEPS_SAMPLER}"
    export MAX_EPISODE_LENGTH_S="${MAX_EPISODE_LENGTH_S}"
    export PYTHON_BIN="${PYTHON_BIN}"
    export REAL_MESH_OBJECT_SPAWN=1
    export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=0
    shard_tmp="${TMP_ROOT}/${shard_name}"
    mkdir -p "${shard_tmp}"
    export TMPDIR="${shard_tmp}"
    export TMP="${shard_tmp}"
    export TEMP="${shard_tmp}"

    cmd=(bash ./infer_teacher_as_contacts.sh "${TEACHER_CHECKPOINT}"
      --require-final-position-success-for-success
      --require-no-middle-foot-object-contact-for-success
      --middle-foot-contact-start-frac "${MIDDLE_FOOT_CONTACT_START_FRAC}"
      --middle-foot-contact-end-frac "${MIDDLE_FOOT_CONTACT_END_FRAC}"
      --foot-object-contact-force-threshold "${FOOT_OBJECT_CONTACT_FORCE_THRESHOLD}")

    if ! is_truthy "${SAVE_SHARD_GLB}"; then
      cmd+=(--no-save-glb)
    fi
    if ! is_truthy "${SAVE_SHARD_PREVIEW_PNG}"; then
      cmd+=(--no-save-preview-png)
    fi
    if ! is_truthy "${SAVE_SHARD_FACE_HEATMAP_PNG}"; then
      cmd+=(--no-save-face-heatmap-png)
    fi

    "${cmd[@]}"
  ) > "${shard_log}" 2>&1 &
  PIDS+=("$!")
done

status=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  if wait "${pid}"; then
    echo "[INFO] ${SHARD_LOGS[$idx]} finished"
  else
    echo "[ERROR] ${SHARD_LOGS[$idx]} failed" >&2
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "[ERROR] At least one realmesh rollout shard failed; leaving partial outputs under ${OUTPUT_ROOT}/shards" >&2
  exit "${status}"
fi

merge_cmd=(
  "${PYTHON_BIN}" scripts/prepare_teacher_as_realmesh_rollout.py merge
  --output-root "${OUTPUT_ROOT}"
  --target-bank "${TARGET_BANK}"
  --source-bank "${SOURCE_AS_DATA_DIR}"
  --contact-export-name "${CONTACT_EXPORT_NAME}"
)
if is_truthy "${FORCE}"; then
  merge_cmd+=(--force)
fi
if is_truthy "${SAVE_OBJECT_FRAME_VIS}"; then
  merge_cmd+=(--save-visualization)
fi
"${merge_cmd[@]}" | tee "${LOG_ROOT}/merge.log"

if is_truthy "${LAUNCH_VISER}"; then
  export PYTHONPATH="${SCRIPT_DIR}/src/holosoma:${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
  viewer_cmd=(
    "${PYTHON_BIN}" -m holosoma.debug_rollout_viewer
    --data-root "${OUTPUT_ROOT}"
    --vis-root "${OUTPUT_ROOT}"
    --stats-root "${OUTPUT_ROOT}"
    --original-motion-dir "${SOURCE_AS_DATA_DIR}"
    --host "${VISER_HOST}"
    --port "${VISER_PORT}"
    --show-original-motion
  )
  echo "[INFO] Launching Viser object-frame contact viewer"
  echo "[INFO] viewer_url=http://localhost:${VISER_PORT}"
  echo "[INFO] viewer_log=${VIEWER_LOG}"
  if is_truthy "${VIEWER_BACKGROUND}"; then
    mkdir -p "$(dirname "${VIEWER_LOG}")"
    {
      printf '[INFO] command:'
      printf ' %q' "${viewer_cmd[@]}"
      printf '\n'
    } > "${VIEWER_LOG}"
    setsid "${viewer_cmd[@]}" >> "${VIEWER_LOG}" 2>&1 < /dev/null &
    echo "[INFO] viewer_pid=$!"
  else
    exec "${viewer_cmd[@]}"
  fi
fi

echo "[INFO] Done."
echo "[INFO] output_root=${OUTPUT_ROOT}"
echo "[INFO] target_bank=${TARGET_BANK}"
