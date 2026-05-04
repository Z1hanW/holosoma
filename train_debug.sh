#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEBUG_ROOT="${DEBUG_ROOT:-${SCRIPT_DIR}/data/ds_box_debug/debug_data/v1_named_sequences_20260502T054723Z}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/train_object_generalist_ds.sh}"

usage() {
  cat <<EOF
Usage:
  bash train_debug.sh <mode> [train_object_generalist_ds.sh extra args...]
  bash train_debug.sh --list

Modes are the .npz folders under:
  ${DEBUG_ROOT}
EOF
}

list_modes() {
  find "${DEBUG_ROOT}" -mindepth 1 -maxdepth 1 -type d -print \
    | while IFS= read -r dir; do
        if find "${dir}" -maxdepth 1 -name '*.npz' -print -quit | grep -q .; then
          basename "${dir}"
        fi
      done \
    | sort
}

if [[ ! -d "${DEBUG_ROOT}" ]]; then
  echo "[ERROR] Debug data root does not exist: ${DEBUG_ROOT}" >&2
  echo "[ERROR] Expected debug_data.zip to be extracted and organized under data/ds_box_debug." >&2
  exit 2
fi

if [[ "$#" -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  echo
  echo "Available modes:"
  list_modes | sed 's/^/  /'
  exit 0
fi

if [[ "${1}" == "--list" ]]; then
  list_modes
  exit 0
fi

MODE="$1"
shift
MOTION_DIR="${DEBUG_ROOT}/${MODE}"
OBJECT_SPEC_PATH="${MOTION_DIR}/_clip_object_urdf_map.json"

if [[ ! -d "${MOTION_DIR}" ]]; then
  echo "[ERROR] Unknown debug mode: ${MODE}" >&2
  echo "[ERROR] Available modes:" >&2
  list_modes | sed 's/^/[ERROR]   /' >&2
  exit 2
fi
if ! find "${MOTION_DIR}" -maxdepth 1 -name '*.npz' -print -quit | grep -q .; then
  echo "[ERROR] Debug mode has no .npz clips: ${MOTION_DIR}" >&2
  exit 2
fi
if [[ ! -f "${OBJECT_SPEC_PATH}" ]]; then
  echo "[ERROR] Debug mode is missing object map: ${OBJECT_SPEC_PATH}" >&2
  exit 2
fi
if [[ ! -d "${MOTION_DIR}/objects" ]]; then
  echo "[ERROR] Debug mode is missing objects directory: ${MOTION_DIR}/objects" >&2
  exit 2
fi

export MOTION_DIR
export OBJECT_SPEC_PATH
export AUTO_PREP_DS_BANK=0
export STRICT_DEFAULT_DS_BANK_VALIDATION=0
export NPROC=1
export PER_GPU_ENVS="${PER_GPU_ENVS:-128}"
export NUM_ENVS="${NUM_ENVS:-128}"
export FORCE_PYTHON_SINGLE_PROCESS=1
export HEADLESS=0
export TRAINING_HEADLESS=False
export LOGGER_BASE_DIR="${LOGGER_BASE_DIR:-${SCRIPT_DIR}/logs/debug_train}"
export EXP="${EXP:-g1-29dof-wbt-w-object-generalist}"
export SEQUENCE_NAME="${SEQUENCE_NAME:-debug-${MODE}}"

echo "[INFO] Debug mode: ${MODE}"
echo "[INFO] MOTION_DIR=${MOTION_DIR}"
echo "[INFO] OBJECT_SPEC_PATH=${OBJECT_SPEC_PATH}"
echo "[INFO] NPROC=${NPROC} NUM_ENVS=${NUM_ENVS} FORCE_PYTHON_SINGLE_PROCESS=${FORCE_PYTHON_SINGLE_PROCESS}"
echo "[INFO] LOGGER_BASE_DIR=${LOGGER_BASE_DIR}"
echo "[INFO] training.headless=False"

exec bash "${TRAIN_SCRIPT}" pure-real "${SEQUENCE_NAME}" "$@"
