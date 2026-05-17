#!/usr/bin/env bash
set -euo pipefail

# Replay and record AS motion/object pairs one at a time in Isaac Sim.
#
# Defaults mirror train_as_general.sh:
#   AS_DATA_DIR=data/ds_as_data/carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout
#   AS_OBJECT_MAP=${AS_DATA_DIR}/_clip_object_urdf_map.json
#   OBJECT_SPAWN_MODE=single_slot_multi_urdf
#
# Examples:
#   bash ./debug_as_replay_record.sh --list
#   LIMIT=1 bash ./debug_as_replay_record.sh
#   bash ./debug_as_replay_record.sh --clip scale__any_ball_0
#   CLIP_REGEX='chair|table' bash ./debug_as_replay_record.sh
#   HEADLESS=False ENABLE_VISER=1 bash ./debug_as_replay_record.sh --clip scale__any_ball_0
#   bash ./debug_as_replay_record.sh -- --logger.video.camera.offset '[3.0,3.0,1.6]'

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

usage() {
  sed -n '1,22p' "$0"
  cat <<'EOF'

Environment knobs:
  CLIP / CLIP_REGEX / CLIP_LIST     Select clips. Empty selection means all clips.
  LIMIT / START_INDEX               Batch slicing after selection.
  AS_DEBUG_ROOT                     Default: outputs/debug_as_replay_record
  AS_VIDEO_DIR                      Default: ${AS_DEBUG_ROOT}/videos
  AS_PAIR_WORK_ROOT                 Default: ${AS_DEBUG_ROOT}/single_pair_banks
  PYTHON_BIN                        Isaac Sim python. Auto-detected if unset.
  VIS_GPU                           GPU id or "auto". Default: auto.
  HEADLESS                          True/False. Default: True.
  ENABLE_VISER                      1 to open Viser alongside recording. Default: 0.
  VIDEO_WIDTH/VIDEO_HEIGHT          Default: 1280x720.
  VIDEO_FORMAT                      h264 or mp4. Default: h264.
  DRY_RUN                           1 prints replay commands without launching.
  SKIP_EXISTING                     1 skips clips whose video dir already has mp4 files.
  KEEP_GOING                        1 continues after a failed clip. Default: 1.
  EXIT_AFTER_VIDEO                  1 treats saved video as success if Isaac Sim shutdown hangs. Default: 1.
EOF
}

CLIP=${CLIP:-}
LIST_ONLY=${LIST_ONLY:-0}
EXTRA_REPLAY_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    list|--list)
      LIST_ONLY=1
      shift
      ;;
    all|--all)
      CLIP=""
      shift
      ;;
    --clip)
      CLIP="${2:-}"
      shift 2
      ;;
    --clip-regex)
      CLIP_REGEX="${2:-}"
      shift 2
      ;;
    --clip-list)
      CLIP_LIST="${2:-}"
      shift 2
      ;;
    --limit)
      LIMIT="${2:-}"
      shift 2
      ;;
    --start-index)
      START_INDEX="${2:-}"
      shift 2
      ;;
    --)
      shift
      EXTRA_REPLAY_ARGS=("$@")
      break
      ;;
    *)
      if [[ -z "${CLIP}" ]]; then
        CLIP="$1"
      else
        EXTRA_REPLAY_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

DEFAULT_AS_BANK=carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout
AS_DATA_DIR=${AS_DATA_DIR:-${OMOMO_DATA_DIR:-"data/ds_as_data/${DEFAULT_AS_BANK}"}}
AS_OBJECT_MAP=${AS_OBJECT_MAP:-${OMOMO_OBJECT_MAP:-"${AS_DATA_DIR}/_clip_object_urdf_map.json"}}
AS_EXPECTED_TOTAL=${AS_EXPECTED_TOTAL:-${OMOMO_EXPECTED_TOTAL:-197}}

EXP=${EXP:-g1-29dof-wbt-w-object-generalist}
COMMAND_CONFIG=${COMMAND_CONFIG:-g1-29dof-wbt-w-object-generalist}
OBJECT_SPAWN_MODE=${OBJECT_SPAWN_MODE:-single_slot_multi_urdf}
AS_OBJECT_GEOMETRY_MODE=${OBJECT_GEOMETRY_MODE:-mesh}
HOLOSOMA_OBJECT_COLLIDER_TYPE=${HOLOSOMA_OBJECT_COLLIDER_TYPE:-convex_decomposition}
HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=${HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK:-0}

case "$(echo "${AS_OBJECT_GEOMETRY_MODE}" | tr '[:upper:]' '[:lower:]')" in
  mesh|urdf|off|disable|disabled|0|false|no)
    AS_OBJECT_GEOMETRY_MODE=mesh
    ;;
  *)
    echo "[ERROR] AS replay debugging expects mesh/URDF object geometry. Got OBJECT_GEOMETRY_MODE=${AS_OBJECT_GEOMETRY_MODE}" >&2
    exit 2
    ;;
esac

case "$(echo "${OBJECT_SPAWN_MODE}" | tr '[:upper:]' '[:lower:]')" in
  urdf|mesh)
    OBJECT_SPAWN_MODE=urdf
    SINGLE_SLOT=0
    ;;
  single_slot_multi_urdf|single-slot-multi-urdf|single_slot|single-slot|heterogeneous_single_slot|heterogeneous-single-slot)
    OBJECT_SPAWN_MODE=single_slot_multi_urdf
    SINGLE_SLOT=1
    ;;
  *)
    echo "[ERROR] OBJECT_SPAWN_MODE must be urdf or single_slot_multi_urdf for this AS replay script. Got: ${OBJECT_SPAWN_MODE}" >&2
    exit 2
    ;;
esac

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in \
    "${HOME}/.holosoma_deps/miniconda3/envs/hssim/bin/python" \
    "${HOME}/.holosoma_deps/miniconda3/envs/sim/bin/python" \
    "/home/ubuntu/miniconda3/envs/sim/bin/python" \
    "$(command -v python3 || true)" \
    "$(command -v python || true)"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN:-}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Could not find an executable Python. Set PYTHON_BIN to the Isaac Sim environment python." >&2
  exit 2
fi

VIS_GPU=${VIS_GPU:-auto}
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" || -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  if [[ "${VIS_GPU}" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      gpu_pick=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2n | head -n1 | cut -d',' -f1 | xargs || true)
      if [[ -n "${gpu_pick}" ]]; then
        export CUDA_VISIBLE_DEVICES="${gpu_pick}"
      fi
    fi
  elif [[ -n "${VIS_GPU}" ]]; then
    export CUDA_VISIBLE_DEVICES="${VIS_GPU}"
  fi
fi

HEADLESS_FLAG=${HEADLESS:-True}
case "$(echo "${HEADLESS_FLAG}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    HEADLESS_FLAG=True
    export HEADLESS=1
    ;;
  0|false|no|off)
    HEADLESS_FLAG=False
    export HEADLESS=0
    ;;
  *)
    echo "[ERROR] HEADLESS must be true/false. Got: ${HEADLESS_FLAG}" >&2
    exit 2
    ;;
esac

AS_DEBUG_ROOT=${AS_DEBUG_ROOT:-"${SCRIPT_DIR}/outputs/debug_as_replay_record"}
AS_PAIR_WORK_ROOT=${AS_PAIR_WORK_ROOT:-"${AS_DEBUG_ROOT}/single_pair_banks"}
AS_VIDEO_DIR=${AS_VIDEO_DIR:-"${AS_DEBUG_ROOT}/videos"}
AS_MANIFEST=${AS_MANIFEST:-"${AS_DEBUG_ROOT}/manifest.tsv"}
AS_LOG_DIR=${AS_LOG_DIR:-"${AS_DEBUG_ROOT}/logs"}
mkdir -p "${AS_DEBUG_ROOT}" "${AS_LOG_DIR}"

PREPARE_ARGS=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/prepare_as_replay_pairs.py"
  --motion-dir "${AS_DATA_DIR}"
  --object-map "${AS_OBJECT_MAP}"
  --work-root "${AS_PAIR_WORK_ROOT}"
  --video-root "${AS_VIDEO_DIR}"
  --manifest "${AS_MANIFEST}"
  --force
)
if [[ -n "${AS_EXPECTED_TOTAL}" ]]; then
  PREPARE_ARGS+=(--expected-total "${AS_EXPECTED_TOTAL}")
fi
if [[ -n "${CLIP}" ]]; then
  PREPARE_ARGS+=(--clip "${CLIP}")
fi
if [[ -n "${CLIP_REGEX:-}" ]]; then
  PREPARE_ARGS+=(--clip-regex "${CLIP_REGEX}")
fi
if [[ -n "${CLIP_LIST:-}" ]]; then
  PREPARE_ARGS+=(--clip-list "${CLIP_LIST}")
fi
if [[ -n "${START_INDEX:-}" ]]; then
  PREPARE_ARGS+=(--start-index "${START_INDEX}")
fi
if [[ -n "${LIMIT:-}" ]]; then
  PREPARE_ARGS+=(--limit "${LIMIT}")
fi
if [[ "${SINGLE_SLOT}" == "1" ]]; then
  PREPARE_ARGS+=(--single-slot)
fi

echo "[INFO] Preparing AS replay pairs from train_as_general defaults"
echo "[INFO] AS_DATA_DIR=${AS_DATA_DIR}"
echo "[INFO] AS_OBJECT_MAP=${AS_OBJECT_MAP}"
echo "[INFO] OBJECT_SPAWN_MODE=${OBJECT_SPAWN_MODE}"
"${PREPARE_ARGS[@]}"

if [[ "${LIST_ONLY}" == "1" ]]; then
  echo "[INFO] Selected clips and object URDFs:"
  awk -F '\t' '{printf "%s\t%s\n", $1, $4}' "${AS_MANIFEST}"
  exit 0
fi

export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
export HOLOSOMA_OBJECT_SPAWN_MODE="${OBJECT_SPAWN_MODE}"
export HOLOSOMA_OBJECT_COLLIDER_TYPE
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK
export HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE="${AS_OBJECT_GEOMETRY_MODE}"
export HOLOSOMA_REPLAY_KEEP_OPEN=${HOLOSOMA_REPLAY_KEEP_OPEN:-0}
export VISER_ENABLE_CLIP_GUI=${VISER_ENABLE_CLIP_GUI:-0}
export VISER_ENABLE_MANUAL_GUI=${VISER_ENABLE_MANUAL_GUI:-0}
export VISER_MANUAL_USE_HW_JOYSTICK=${VISER_MANUAL_USE_HW_JOYSTICK:-0}

NUM_ENVS=${NUM_ENVS:-1}
ENABLE_VISER=${ENABLE_VISER:-0}
VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_SYNC_TO_SIM=${VISER_SYNC_TO_SIM:-True}
VISER_FORCE_DT=${VISER_FORCE_DT:-True}
VISER_RECENTER=${VISER_RECENTER:-True}
VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS:-False}

VIDEO_WIDTH=${VIDEO_WIDTH:-1280}
VIDEO_HEIGHT=${VIDEO_HEIGHT:-720}
VIDEO_FORMAT=${VIDEO_FORMAT:-h264}
VIDEO_PLAYBACK_RATE=${VIDEO_PLAYBACK_RATE:-1.0}
VIDEO_CAMERA_SMOOTHING=${VIDEO_CAMERA_SMOOTHING:-0.90}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-1.0}
FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}
RESET_NOISE_SCALE=${RESET_NOISE_SCALE:-0.0}
DISABLE_RANDOMIZATION=${DISABLE_RANDOMIZATION:-1}
DRY_RUN=${DRY_RUN:-0}
SKIP_EXISTING=${SKIP_EXISTING:-0}
KEEP_GOING=${KEEP_GOING:-1}
TEE_LOGS=${TEE_LOGS:-1}
EXIT_AFTER_VIDEO=${EXIT_AFTER_VIDEO:-1}
VIDEO_SAVE_GRACE_SECONDS=${VIDEO_SAVE_GRACE_SECONDS:-5}
VIDEO_SAVE_POLL_SECONDS=${VIDEO_SAVE_POLL_SECONDS:-2}

is_truthy() {
  case "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

run_replay_with_video_watchdog() {
  local log_path="$1"
  local watch_clip_id="$2"
  shift 2
  local child_pid
  local saved_at=0
  local status=0

  : >"${log_path}"
  if command -v setsid >/dev/null 2>&1; then
    setsid "$@" </dev/null >>"${log_path}" 2>&1 &
  else
    "$@" </dev/null >>"${log_path}" 2>&1 &
  fi
  child_pid=$!

  while kill -0 "${child_pid}" 2>/dev/null; do
    if grep -q "Successfully saved video file:" "${log_path}" 2>/dev/null; then
      if [[ "${saved_at}" == "0" ]]; then
        saved_at=$(date +%s)
        echo "[INFO]   saved video for ${watch_clip_id}; waiting ${VIDEO_SAVE_GRACE_SECONDS}s for Isaac Sim shutdown"
      elif (( $(date +%s) - saved_at >= VIDEO_SAVE_GRACE_SECONDS )); then
        echo "[WARN]   Isaac Sim shutdown still running after video save for ${watch_clip_id}; stopping child process"
        kill -TERM "-${child_pid}" 2>/dev/null || kill -TERM "${child_pid}" 2>/dev/null || true
        sleep 2
        if kill -0 "${child_pid}" 2>/dev/null; then
          kill -KILL "-${child_pid}" 2>/dev/null || kill -KILL "${child_pid}" 2>/dev/null || true
        fi
        wait "${child_pid}" 2>/dev/null || true
        return 0
      fi
    fi
    sleep "${VIDEO_SAVE_POLL_SECONDS}"
  done

  if wait "${child_pid}"; then
    status=0
  else
    status=$?
  fi
  if [[ "${status}" != "0" ]] && grep -q "Successfully saved video file:" "${log_path}" 2>/dev/null; then
    echo "[WARN]   replay exited with status ${status} after saving video for ${watch_clip_id}; treating as success"
    return 0
  fi
  return "${status}"
}

total=$(wc -l < "${AS_MANIFEST}" | tr -d '[:space:]')
index=0
failed=0

while IFS=$'\t' read -r clip_id pair_dir pair_map object_urdf source_npz video_dir; do
  index=$((index + 1))
  if [[ "${SKIP_EXISTING}" == "1" ]] && compgen -G "${video_dir}/*.mp4" >/dev/null; then
    echo "[INFO] [${index}/${total}] Skipping existing video for ${clip_id}: ${video_dir}"
    continue
  fi

  cmd=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/src/holosoma/holosoma/replay.py"
    "exp:${EXP}"
    "command:${COMMAND_CONFIG}"
    "logger:disabled"
    --training.headless="${HEADLESS_FLAG}"
    --training.num-envs="${NUM_ENVS}"
    --command.setup-terms.motion-command.params.motion-config.motion-file "${pair_dir}"
    --command.setup-terms.motion-command.params.motion-config.motion-clip-name "${clip_id}"
    --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler=False
    --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob "${START_AT_TIMESTEP_ZERO_PROB}"
    --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob "${FREEZE_AT_TIMESTEP_ZERO_PROB}"
    --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale "${RESET_NOISE_SCALE}"
    --robot.object.enabled=True
    --robot.object.object-urdf-path "${pair_map}"
    --logger.video.enabled=True
    --logger.headless_recording=True
    --logger.video.upload_to_wandb=False
    --logger.video.save_dir "${video_dir}"
    --logger.video.width "${VIDEO_WIDTH}"
    --logger.video.height "${VIDEO_HEIGHT}"
    --logger.video.output_format "${VIDEO_FORMAT}"
    --logger.video.playback_rate "${VIDEO_PLAYBACK_RATE}"
    --logger.video.camera_smoothing "${VIDEO_CAMERA_SMOOTHING}"
    --logger.video.show_command_overlay=False
    --simulator.config.sim.physx.gpu-max-rigid-contact-count="${PHYSX_GPU_MAX_RIGID_CONTACT_COUNT:-33554432}"
    --simulator.config.sim.physx.gpu-max-rigid-patch-count="${PHYSX_GPU_MAX_RIGID_PATCH_COUNT:-4194304}"
    --simulator.config.sim.physx.gpu-found-lost-pairs-capacity="${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY:-134217728}"
    --simulator.config.sim.physx.gpu-found-lost-aggregate-pairs-capacity="${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY:-134217728}"
    --simulator.config.sim.physx.gpu-total-aggregate-pairs-capacity="${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY:-16777216}"
    --simulator.config.sim.physx.gpu-collision-stack-size="${PHYSX_GPU_COLLISION_STACK_SIZE:-67108864}"
    --simulator.config.sim.physx.gpu-heap-capacity="${PHYSX_GPU_HEAP_CAPACITY:-67108864}"
    --simulator.config.sim.physx.gpu-temp-buffer-capacity="${PHYSX_GPU_TEMP_BUFFER_CAPACITY:-16777216}"
  )

  if [[ "$(echo "${ENABLE_VISER}" | tr '[:upper:]' '[:lower:]')" =~ ^(1|true|yes|on)$ ]]; then
    cmd+=(
      --training.enable-viser=True
      --training.viser-port="${VISER_PORT}"
      --training.viser-env-id="${VISER_ENV_ID}"
      --training.viser-update-hz="${VISER_UPDATE_HZ}"
      --training.viser-sync-to-sim="${VISER_SYNC_TO_SIM}"
      --training.viser-force-dt="${VISER_FORCE_DT}"
      --training.viser-recenter="${VISER_RECENTER}"
      --training.viser-show-scandots="${VISER_SHOW_SCANDOTS}"
    )
  fi

  if [[ "$(echo "${DISABLE_RANDOMIZATION}" | tr '[:upper:]' '[:lower:]')" =~ ^(1|true|yes|on)$ ]]; then
    cmd+=(randomization:disabled)
  fi
  if [[ -n "${VIDEO_CAMERA_OFFSET:-}" ]]; then
    cmd+=(--logger.video.camera.offset "${VIDEO_CAMERA_OFFSET}")
  fi
  if [[ -n "${VIDEO_CAMERA_TARGET_OFFSET:-}" ]]; then
    cmd+=(--logger.video.camera.target_offset "${VIDEO_CAMERA_TARGET_OFFSET}")
  fi
  cmd+=("${EXTRA_REPLAY_ARGS[@]}")

  echo "[INFO] [${index}/${total}] Replaying ${clip_id}"
  echo "[INFO]   motion=${source_npz}"
  echo "[INFO]   object=${object_urdf}"
  echo "[INFO]   video_dir=${video_dir}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    continue
  fi

  safe_clip_log_name="${clip_id//[^A-Za-z0-9_.-]/_}"
  log_path="${AS_LOG_DIR}/${index}_${safe_clip_log_name}.log"
  clip_failed=0
  if is_truthy "${EXIT_AFTER_VIDEO}"; then
    if ! run_replay_with_video_watchdog "${log_path}" "${clip_id}" "${cmd[@]}"; then
      clip_failed=1
    fi
  elif [[ "${TEE_LOGS}" == "1" ]]; then
    if ! "${cmd[@]}" </dev/null 2>&1 | tee "${log_path}"; then
      clip_failed=1
    fi
  else
    if ! "${cmd[@]}" </dev/null >"${log_path}" 2>&1; then
      clip_failed=1
    fi
  fi

  if [[ "${clip_failed}" == "0" ]] && ! compgen -G "${video_dir}/*.mp4" >/dev/null; then
    echo "[ERROR] Replay produced no mp4 for ${clip_id}; log=${log_path}" >&2
    clip_failed=1
  fi
  if [[ "${clip_failed}" != "0" ]]; then
    failed=$((failed + 1))
    echo "[ERROR] Replay failed for ${clip_id}; log=${log_path}" >&2
    if [[ "${KEEP_GOING}" != "1" ]]; then
      exit 1
    fi
  fi
done < "${AS_MANIFEST}"

if [[ "${failed}" != "0" ]]; then
  echo "[ERROR] Completed with ${failed} failed replay(s). Logs: ${AS_LOG_DIR}" >&2
  exit 1
fi

echo "[INFO] Completed ${total} replay recording(s)."
echo "[INFO] Videos: ${AS_VIDEO_DIR}"
