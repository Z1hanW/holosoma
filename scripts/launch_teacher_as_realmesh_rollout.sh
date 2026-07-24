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

release_launch_lock() {
  if [[ -n "${LAUNCH_LOCK_KEEPALIVE_FD:-}" ]]; then
    exec {LAUNCH_LOCK_KEEPALIVE_FD}>&-
    LAUNCH_LOCK_KEEPALIVE_FD=""
  fi
  if [[ -n "${LAUNCH_LOCK_HOLDER_PROCESS_PID:-}" ]]; then
    wait "${LAUNCH_LOCK_HOLDER_PROCESS_PID}" 2>/dev/null || true
    LAUNCH_LOCK_HOLDER_PROCESS_PID=""
  fi
}

launch_lock_only_exit_cleanup() {
  local status=$?
  trap - EXIT HUP INT TERM
  release_launch_lock
  exit "${status}"
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
if ! [[ "${NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] NUM_SHARDS must be a positive integer. Got: ${NUM_SHARDS}" >&2
  exit 2
fi
if [[ "${#GPUS[@]}" -ne "${NUM_SHARDS}" ]]; then
  echo "[ERROR] GPU_LIST must contain NUM_SHARDS entries. GPU_LIST=${GPU_LIST} NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 2
fi
declare -A _SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  if ! [[ "${gpu}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "[ERROR] GPU_LIST entries must be canonical non-negative integer indices. Got: ${gpu@Q}" >&2
    exit 2
  fi
  if [[ -n "${_SEEN_GPUS[${gpu}]+x}" ]]; then
    echo "[ERROR] GPU_LIST contains duplicate GPU index ${gpu}; independent shards require unique devices." >&2
    exit 2
  fi
  _SEEN_GPUS["${gpu}"]=1
done
if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t _AVAILABLE_GPUS < <(
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null \
      | sed 's/[[:space:]]//g; /^[[:space:]]*$/d'
  )
  declare -A _AVAILABLE_GPU_SET=()
  for gpu in "${_AVAILABLE_GPUS[@]}"; do
    [[ -n "${gpu}" ]] && _AVAILABLE_GPU_SET["${gpu}"]=1
  done
  for gpu in "${GPUS[@]}"; do
    if [[ -z "${_AVAILABLE_GPU_SET[${gpu}]+x}" ]]; then
      echo "[ERROR] GPU_LIST selects unavailable GPU index ${gpu}; available=${_AVAILABLE_GPUS[*]:-<none>}." >&2
      exit 2
    fi
  done
  unset _AVAILABLE_GPUS _AVAILABLE_GPU_SET
fi
unset _SEEN_GPUS

ROLLOUT_CONTRACT_JSON="$("${PYTHON_BIN}" - \
  "${HEADLESS}" "${DISABLE_RANDOMIZATION}" "${START_AT_TIMESTEP_ZERO_PROB}" \
  "${FREEZE_AT_TIMESTEP_ZERO_PROB}" "${RESET_NOISE_SCALE}" "${USE_ADAPTIVE_TIMESTEPS_SAMPLER}" \
  "${MAX_EPISODE_LENGTH_S}" "${PHYSX_GPU_COLLISION_STACK_SIZE}" "${SUCCESS_POSITION_THRESHOLD}" \
  "${MIN_CONTACT_FRAMES}" "${CONTACT_FORCE_THRESHOLD}" "${CONTACT_VOXEL_SIZE}" \
  "${FOOT_OBJECT_CONTACT_FORCE_THRESHOLD}" "${MIDDLE_FOOT_CONTACT_START_FRAC}" \
  "${MIDDLE_FOOT_CONTACT_END_FRAC}" "${HOLOSOMA_OBJECT_COLLIDER_TYPE:-convex_decomposition}" \
  "PCI_BUS_ID" <<'PY'
import json
import sys

keys = (
    "headless",
    "disable_randomization",
    "start_at_timestep_zero_prob",
    "freeze_at_timestep_zero_prob",
    "reset_noise_scale",
    "use_adaptive_timesteps_sampler",
    "max_episode_length_s",
    "physx_gpu_collision_stack_size",
    "success_position_threshold",
    "min_contact_frames",
    "contact_force_threshold",
    "contact_voxel_size",
    "foot_object_contact_force_threshold",
    "middle_foot_contact_start_frac",
    "middle_foot_contact_end_frac",
    "object_collider_type",
    "cuda_device_order",
)
print(json.dumps(dict(zip(keys, sys.argv[1:], strict=True)), sort_keys=True, separators=(",", ":")))
PY
)"

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
  --rollout-contract-json "${ROLLOUT_CONTRACT_JSON}"
)

if is_truthy "${DRY_RUN}"; then
  PLAN_JSON="$("${prepare_args[@]}" --dry-run)"
  mapfile -t PLAN_VALUES < <(printf '%s' "${PLAN_JSON}" | "${PYTHON_BIN}" -c '
import json
import sys
payload = json.load(sys.stdin)
print(payload["per_gpu_envs"])
print(payload["selected_clip_count"])
for shard in payload["shards"]:
    print(shard["count"])
')
  PER_GPU_ENVS="${PLAN_VALUES[0]}"
  SELECTED_CLIPS="${PLAN_VALUES[1]}"
else
  if ! command -v setsid >/dev/null 2>&1; then
    echo "[ERROR] setsid is required for owned shard process-group cleanup." >&2
    exit 2
  fi
  if ((BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1))); then
    echo "[ERROR] Bash >= 5.1 is required for fail-fast wait -n -p shard supervision." >&2
    exit 2
  fi
  # All rollout launches can write one or more of the shard, output, target,
  # generation, and compatibility-view namespaces.  A single lexical global
  # lock avoids both partial-resource overlap and a TARGET_BANK symlink changing
  # its resolved lock identity after an atomic generation switch.
  LAUNCH_LOCK_ROOT="${SCRIPT_DIR}/data/ds_as_data/_teacher_rollout_launch_locks"
  LAUNCH_LOCK_NAME="global.lock"
  LAUNCH_LOCK_TIMEOUT_S="${TEACHER_ROLLOUT_LAUNCH_LOCK_TIMEOUT_S:-0}"
  coproc LAUNCH_LOCK_HOLDER {
    "${PYTHON_BIN}" scripts/hold_no_follow_lock.py \
      --root "${LAUNCH_LOCK_ROOT}" \
      --name "${LAUNCH_LOCK_NAME}" \
      --timeout-seconds "${LAUNCH_LOCK_TIMEOUT_S}"
  }
  LAUNCH_LOCK_HOLDER_PROCESS_PID="${LAUNCH_LOCK_HOLDER_PID}"
  LAUNCH_LOCK_READ_FD="${LAUNCH_LOCK_HOLDER[0]}"
  if ! IFS= read -r LAUNCH_LOCK_STATUS <&"${LAUNCH_LOCK_READ_FD}" || [[ "${LAUNCH_LOCK_STATUS}" != "LOCKED" ]]; then
    wait "${LAUNCH_LOCK_HOLDER_PROCESS_PID}" 2>/dev/null || true
    echo "[ERROR] Another teacher rollout owns this shard/output/target scope." >&2
    exit 2
  fi
  exec {LAUNCH_LOCK_READ_FD}<&-
  LAUNCH_LOCK_KEEPALIVE_FD="${LAUNCH_LOCK_HOLDER[1]}"
  trap launch_lock_only_exit_cleanup EXIT
  trap 'exit 129' HUP
  trap 'exit 130' INT
  trap 'exit 143' TERM
  mkdir -p "${SHARD_ROOT}" "${OUTPUT_ROOT}/shards" "${LOG_ROOT}" "${TMP_ROOT}"
  export TMPDIR="${TMP_ROOT}"
  export TMP="${TMP_ROOT}"
  export TEMP="${TMP_ROOT}"
  "${prepare_args[@]}" > "${LOG_ROOT}/prepare_manifest_stdout.json"
  PREPARED_MANIFEST_SHA256="$(sha256sum "${SHARD_ROOT}/manifest.json" | awk '{print $1}')"
  mapfile -t PLAN_VALUES < <("${PYTHON_BIN}" - "${SHARD_ROOT}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
print(payload["per_gpu_envs"])
print(payload["selected_clip_count"])
for shard in payload["shards"]:
    print(shard["count"])
PY
  )
  PER_GPU_ENVS="${PLAN_VALUES[0]}"
  SELECTED_CLIPS="${PLAN_VALUES[1]}"
fi

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
    expected_count="${PLAN_VALUES[$((shard_idx + 2))]}"
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

# Resolve/download exactly once, publish under the content digest, and pass the
# same immutable local file to every independent shard. Re-resolving a mutable
# W&B run file per GPU can otherwise create a mixed-teacher output bank.
export PYTHONPATH="${SCRIPT_DIR}/src/holosoma:${SCRIPT_DIR}/src/holosoma_inference:${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
TEACHER_CHECKPOINT_SOURCE="${TEACHER_CHECKPOINT}"
TEACHER_CHECKPOINT_CACHE_ROOT="${TEACHER_CHECKPOINT_CACHE_ROOT:-${SCRIPT_DIR}/.checkpoint_cache/teacher_realmesh_rollout}"
TEACHER_CHECKPOINT="$("${PYTHON_BIN}" scripts/resolve_exact_checkpoint.py \
  --ref "${TEACHER_CHECKPOINT_SOURCE}" \
  --cache-root "${TEACHER_CHECKPOINT_CACHE_ROOT}")"
TEACHER_CHECKPOINT_SHA256="$(sha256sum "${TEACHER_CHECKPOINT}" | awk '{print $1}')"
if [[ -n "${TEACHER_CHECKPOINT_EXPECTED_SHA256:-}" && "${TEACHER_CHECKPOINT_SHA256}" != "${TEACHER_CHECKPOINT_EXPECTED_SHA256}" ]]; then
  echo "[ERROR] Resolved teacher SHA256 mismatch: actual=${TEACHER_CHECKPOINT_SHA256} expected=${TEACHER_CHECKPOINT_EXPECTED_SHA256}" >&2
  exit 2
fi
echo "[INFO] immutable_teacher_checkpoint=${TEACHER_CHECKPOINT}"
echo "[INFO] teacher_checkpoint_sha256=${TEACHER_CHECKPOINT_SHA256}"

declare -a PIDS=()
declare -a SHARD_LOGS=()
declare -A ACTIVE_SHARDS=()
declare -A OWNED_SHARD_GROUPS=()
CLEANUP_DISARMED=0
CLEANUP_RUNNING=0

terminate_owned_shards() {
  local pid deadline any_running
  local -A cleanup_pids=()
  local -a owned_pids=()
  ((CLEANUP_RUNNING == 0)) || return 0
  CLEANUP_RUNNING=1
  for pid in "${!OWNED_SHARD_GROUPS[@]}"; do
    cleanup_pids["${pid}"]=1
  done
  # Close the tiny signal window between `cmd &` and recording `$!`.  The lock
  # holder is the only non-shard background job and is excluded explicitly.
  while IFS= read -r pid; do
    [[ -n "${pid}" && "${pid}" != "${LAUNCH_LOCK_HOLDER_PROCESS_PID:-}" ]] && cleanup_pids["${pid}"]=1
  done < <(jobs -pr)
  owned_pids=("${!cleanup_pids[@]}")
  if ((${#owned_pids[@]} > 0)); then
    echo "[INFO] Terminating ${#owned_pids[@]} owned rollout shard process group(s)." >&2
  fi
  for pid in "${owned_pids[@]}"; do
    kill -TERM -- "${pid}" 2>/dev/null || true
    kill -TERM -- "-${pid}" 2>/dev/null || true
  done
  deadline=$((SECONDS + 10))
  while ((SECONDS < deadline)); do
    any_running=0
    for pid in "${owned_pids[@]}"; do
      if kill -0 -- "${pid}" 2>/dev/null || kill -0 -- "-${pid}" 2>/dev/null; then
        any_running=1
        break
      fi
    done
    ((any_running == 0)) && break
    sleep 0.2
  done
  for pid in "${owned_pids[@]}"; do
    kill -KILL -- "${pid}" 2>/dev/null || true
    kill -KILL -- "-${pid}" 2>/dev/null || true
  done
  for pid in "${owned_pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
    unset 'ACTIVE_SHARDS[$pid]'
  done
  release_launch_lock
}

launcher_exit_cleanup() {
  local status=$?
  trap - EXIT HUP INT TERM
  if ((CLEANUP_DISARMED == 0)); then
    terminate_owned_shards
  else
    release_launch_lock
  fi
  exit "${status}"
}

launcher_hup() { exit 129; }
launcher_int() { exit 130; }
launcher_term() { exit 143; }
trap launcher_exit_cleanup EXIT
trap launcher_hup HUP
trap launcher_int INT
trap launcher_term TERM

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
    exec {LAUNCH_LOCK_KEEPALIVE_FD}>&-
    cd "${SCRIPT_DIR}"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export HOLOSOMA_DEVICE="cuda:0"
    # Each shard is an independent single-process rollout. Never inherit a
    # parent torchrun topology, which would reinterpret LOCAL_RANK as a device
    # index and can route shard 40+ to a nonexistent GPU.
    unset WORLD_SIZE RANK GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
    export LOCAL_RANK=0
    export AS_DATA_DIR="${shard_dir}"
    export AS_OBJECT_MAP="${shard_dir}/_clip_object_urdf_map.json"
    export AS_EXPECTED_TOTAL="${expected_count}"
    export TEACHER_ROLLOUT_PREPARED_MANIFEST_SHA256="${PREPARED_MANIFEST_SHA256}"
    export TEACHER_ROLLOUT_EXPECTED_CLIP_IDS_FILE="${shard_dir}/clip_ids.txt"
    export TEACHER_ROLLOUT_SHARD_NAME="${shard_name}"
    # The prepared shard itself already is the authenticated single-slot bank;
    # never let inference create a mutable child inside the read-only snapshot.
    export AS_SINGLE_SLOT_MOTION_DIR="${shard_dir}"
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
    export HOLOSOMA_OBJECT_COLLIDER_TYPE="${HOLOSOMA_OBJECT_COLLIDER_TYPE:-convex_decomposition}"
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

    # Make every owned shard its own process group so a launcher-side failure
    # can terminate that shard and all descendants without touching unrelated
    # jobs on the host.
    exec setsid "${cmd[@]}"
  ) > "${shard_log}" 2>&1 &
  shard_pid="$!"
  PIDS+=("${shard_pid}")
  ACTIVE_SHARDS["${shard_pid}"]="${shard_idx}"
  OWNED_SHARD_GROUPS["${shard_pid}"]=1
done

while ((${#ACTIVE_SHARDS[@]} > 0)); do
  completed_pid=""
  if wait -n -p completed_pid "${!ACTIVE_SHARDS[@]}"; then
    shard_status=0
  else
    shard_status=$?
  fi
  if [[ -z "${completed_pid}" || -z "${ACTIVE_SHARDS[${completed_pid}]+x}" ]]; then
    echo "[ERROR] Could not identify the completed owned rollout shard (wait status=${shard_status})." >&2
    exit 1
  fi
  completed_idx="${ACTIVE_SHARDS[${completed_pid}]}"
  if ((shard_status == 0)); then
    unset 'ACTIVE_SHARDS[$completed_pid]'
    if kill -0 -- "-${completed_pid}" 2>/dev/null; then
      echo "[ERROR] ${SHARD_LOGS[$completed_idx]} left descendants in its owned process group." >&2
      exit 1
    fi
    echo "[INFO] ${SHARD_LOGS[$completed_idx]} finished"
    continue
  fi
  echo "[ERROR] ${SHARD_LOGS[$completed_idx]} failed with status ${shard_status}" >&2
  echo "[ERROR] A realmesh rollout shard failed; leaving diagnostic outputs under ${OUTPUT_ROOT}/shards" >&2
  exit "${shard_status}"
done

merge_cmd=(
  "${PYTHON_BIN}" scripts/prepare_teacher_as_realmesh_rollout.py merge
  --output-root "${OUTPUT_ROOT}"
  --target-bank "${TARGET_BANK}"
  --source-bank "${SOURCE_AS_DATA_DIR}"
  --prepared-manifest "${SHARD_ROOT}/manifest.json"
  --prepared-manifest-sha256 "${PREPARED_MANIFEST_SHA256}"
  --contact-export-name "${CONTACT_EXPORT_NAME}"
  --expected-teacher-checkpoint-sha256 "${TEACHER_CHECKPOINT_SHA256}"
  --teacher-checkpoint-path "${TEACHER_CHECKPOINT}"
  --teacher-checkpoint-source "${TEACHER_CHECKPOINT_SOURCE}"
)
if is_truthy "${FORCE}"; then
  merge_cmd+=(--force)
fi
if is_truthy "${SAVE_OBJECT_FRAME_VIS}"; then
  merge_cmd+=(--save-visualization)
fi
"${merge_cmd[@]}" | tee "${LOG_ROOT}/merge.log"
CLEANUP_DISARMED=1
release_launch_lock
trap - EXIT HUP INT TERM

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
