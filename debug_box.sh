#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash debug_box.sh

Environment overrides:
  DATA_DIR                 Default: ./data_demo
  VISER_PORT               Default: 18085
  VISER_ENV_ID             Default: 0
  VISER_GRID_SPACING       Default: 2.8
  VISER_MULTI_ENV_COLS     Default: 3
  VISER_UPDATE_HZ          Default: 30
  VISER_START_PAUSED       Default: 0
  VISER_PLAY_RESTARTS_VISIBLE_REPLAY
                           Default: 1
  HEADLESS                 Default: False (training.headless)
  ISAAC_APP_HEADLESS       Default: auto (1 when no DISPLAY, else follows HEADLESS)
  GPU_ID                   Default: auto
  DRY_RUN                  Default: 0

Notes:
  - Replays the 3 demo pairs from data_demo in Isaac Sim with 3 envs.
  - Viser geometry is sourced from live Isaac Sim USD + simulator state, not URDF playback.
  - GUI exposes Mesh Mode for visual mesh / collision mesh switching.
  - On machines without X/Wayland, Isaac AppLauncher is forced headless even if training.headless=False.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

source "${ROOT_DIR}/scripts/source_common.sh"
source "${ROOT_DIR}/scripts/source_isaacsim_setup.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data_demo}"
OBJECT_MAP="${OBJECT_MAP:-${DATA_DIR}/_clip_object_urdf_map.json}"
NUM_ENVS="${NUM_ENVS:-3}"
VISER_PORT="${VISER_PORT:-18085}"
VISER_ENV_ID="${VISER_ENV_ID:-0}"
VISER_GRID_SPACING="${VISER_GRID_SPACING:-2.8}"
VISER_MULTI_ENV_COLS="${VISER_MULTI_ENV_COLS:-3}"
VISER_UPDATE_HZ="${VISER_UPDATE_HZ:-30}"
VISER_START_PAUSED="${VISER_START_PAUSED:-0}"
TRAINING_NAME="${TRAINING_NAME:-debug_box_replay}"
GPU_ID="${GPU_ID:-auto}"
DRY_RUN="${DRY_RUN:-0}"

TRAINING_HEADLESS_RAW="${HEADLESS:-False}"
case "$(printf '%s' "${TRAINING_HEADLESS_RAW}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    TRAINING_HEADLESS="True"
    ;;
  0|false|no|off)
    TRAINING_HEADLESS="False"
    ;;
  *)
    echo "[ERROR] HEADLESS must be True/False/1/0, got: ${TRAINING_HEADLESS_RAW}" >&2
    exit 2
    ;;
esac

APP_HEADLESS_RAW="${ISAAC_APP_HEADLESS:-auto}"
case "$(printf '%s' "${APP_HEADLESS_RAW}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    HEADLESS_ENV=1
    ;;
  0|false|no|off)
    HEADLESS_ENV=0
    ;;
  auto|"")
    if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
      if [[ "${TRAINING_HEADLESS}" == "True" ]]; then
        HEADLESS_ENV=1
      else
        HEADLESS_ENV=0
      fi
    else
      HEADLESS_ENV=1
    fi
    ;;
  *)
    echo "[ERROR] ISAAC_APP_HEADLESS must be auto/True/False/1/0, got: ${APP_HEADLESS_RAW}" >&2
    exit 2
    ;;
esac

if [[ "${NUM_ENVS}" != "3" ]]; then
  echo "[ERROR] debug_box.sh expects NUM_ENVS=3, got ${NUM_ENVS}" >&2
  exit 2
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[ERROR] DATA_DIR not found: ${DATA_DIR}" >&2
  exit 1
fi
if [[ ! -f "${OBJECT_MAP}" ]]; then
  echo "[ERROR] object map not found: ${OBJECT_MAP}" >&2
  exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES+x}" || -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  if [[ "${GPU_ID}" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      gpu_pick="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2n | head -n1 | cut -d',' -f1 | xargs)"
      if [[ -n "${gpu_pick}" ]]; then
        export CUDA_VISIBLE_DEVICES="${gpu_pick}"
      fi
    fi
  else
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  fi
fi

export PYTHONUNBUFFERED=1
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export HEADLESS="${HEADLESS_ENV}"
export VISER_MESH_SOURCE="${VISER_MESH_SOURCE:-sim}"
export VISER_MESH_MODE="${VISER_MESH_MODE:-both}"
export VISER_LOAD_URDF=0
export VISER_MULTI_ENV_COLS="${VISER_MULTI_ENV_COLS}"
export VISER_START_PAUSED="${VISER_START_PAUSED}"
export VISER_PLAY_RESTARTS_VISIBLE_REPLAY="${VISER_PLAY_RESTARTS_VISIBLE_REPLAY:-1}"
export VISER_RESET_RESTARTS_VISIBLE_REPLAY="${VISER_RESET_RESTARTS_VISIBLE_REPLAY:-1}"
export VISER_ENABLE_CLIP_GUI="${VISER_ENABLE_CLIP_GUI:-0}"
export VISER_ENABLE_MANUAL_GUI="${VISER_ENABLE_MANUAL_GUI:-0}"
export VISER_SHOW_TARGET_KEYPOINTS="${VISER_SHOW_TARGET_KEYPOINTS:-0}"
export HOLOSOMA_REPLAY_KEEP_OPEN="${HOLOSOMA_REPLAY_KEEP_OPEN:-1}"

cmd=(
  "${PYTHON_BIN}" src/holosoma/holosoma/replay.py
  exp:g1-29dof-wbt-w-object-generalist
  randomization:disabled
  logger:disabled
  --training.name="${TRAINING_NAME}"
  --training.headless="${TRAINING_HEADLESS}"
  --training.debug=True
  --training.num-envs=3
  --training.enable-viser=True
  --training.viser-port="${VISER_PORT}"
  --training.viser-env-id="${VISER_ENV_ID}"
  --training.viser-env-count=3
  --training.viser-multi-env-spacing="${VISER_GRID_SPACING}"
  --training.viser-update-hz="${VISER_UPDATE_HZ}"
  --training.viser-sync-to-sim=True
  --training.viser-force-dt=True
  --training.viser-recenter=True
  --training.viser-show-scandots=False
  --command.setup-terms.motion-command.params.motion-config.motion-file "${DATA_DIR}"
  --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler False
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob 1.0
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-prepend False
  --command.setup-terms.motion-command.params.motion-config.default-pose-prepend-duration-s 0.0
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-append False
  --command.setup-terms.motion-command.params.motion-config.default-pose-append-duration-s 0.0
  --robot.object.object-urdf-path "${OBJECT_MAP}"
)

echo "[INFO] data_dir=${DATA_DIR}"
echo "[INFO] object_map=${OBJECT_MAP}"
echo "[INFO] viser=http://localhost:${VISER_PORT}"
echo "[INFO] training_headless=${TRAINING_HEADLESS} isaac_app_headless=${HEADLESS_ENV}"
echo "[INFO] mesh_source=${VISER_MESH_SOURCE} mesh_mode=${VISER_MESH_MODE}"
echo "[INFO] play_restarts_visible_replay=${VISER_PLAY_RESTARTS_VISIBLE_REPLAY}"
echo "[INFO] reset_restarts_visible_replay=${VISER_RESET_RESTARTS_VISIBLE_REPLAY}"
printf '[INFO] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$(printf '%s' "${DRY_RUN}" | tr '[:upper:]' '[:lower:]')" =~ ^(1|true|yes|on)$ ]]; then
  exit 0
fi

"${cmd[@]}"
