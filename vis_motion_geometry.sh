#!/usr/bin/env bash
set -euo pipefail

# Motion geometry viewer.
#
# `kinematic` skips the simulator and visualizes motion directly.
# `replay` launches Isaac Sim and Viser reads simulator states.
#
# Usage:
#   bash ./vis_motion_geometry.sh behave
#   bash ./vis_motion_geometry.sh omomo
#   bash ./vis_motion_geometry.sh behave replay
#   START_CLIP=sub10_largebox_032_mj_w_obj bash ./vis_motion_geometry.sh omomo
#   MOTION_DIR=/abs/path OBJECT_URDF=/abs/path/to/spec_or_urdf bash ./vis_motion_geometry.sh behave

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,14p' "$0"
  exit 0
fi

CLI_DATASET_KNOB=${1:-""}
CLI_VIS_MODE=${2:-""}
DATASET_KNOB=${CLI_DATASET_KNOB:-${DATASET_KNOB:-"behave"}}

case "${DATASET_KNOB}" in
  crisp)
    DEFAULT_MOTION_DIR="/data/terrain/___crisp_clean_motion"
    DEFAULT_GEOMETRY_DIR="/data/terrain/___crisp_clean_geometry"
    DEFAULT_OBJECT_URDF_DIR="${SCRIPT_DIR}/crisp/vmm_data/___crisp_object_urdf"
    DEFAULT_OBJECT_URDF=""
    DEFAULT_OBJECT_URDF_MODE="stem"
    DEFAULT_OBJECT_FILTER=""
    DEFAULT_EXP="g1-29dof-wbt"
    ;;
  omomo)
    DEFAULT_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"
    DEFAULT_GEOMETRY_DIR=""
    DEFAULT_OBJECT_URDF_DIR=""
    DEFAULT_OBJECT_URDF="${SCRIPT_DIR}/src/holosoma_retargeting/models/largebox/largebox.urdf"
    DEFAULT_OBJECT_URDF_MODE="stem"
    DEFAULT_OBJECT_FILTER=""
    DEFAULT_EXP="g1-29dof-wbt-w-object-generalist"
    ;;
  behave)
    DEFAULT_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/behave_sq_carry_xy_0p5_1p5_flat"
    DEFAULT_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/behave_sq_carry"
    DEFAULT_GEOMETRY_DIR=""
    DEFAULT_OBJECT_URDF_DIR=""
    DEFAULT_OBJECT_URDF="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/behave_sq_carry_xy_0p5_1p5_flat/_clip_object_urdf_map.json"
    DEFAULT_OBJECT_URDF="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/behave_sq_carry/_clip_object_urdf_map.json"
    DEFAULT_OBJECT_URDF_MODE="behave"
    DEFAULT_OBJECT_FILTER="boxmedium,boxlarge"
    DEFAULT_EXP="g1-29dof-wbt-w-object-generalist"
    ;;
  *)
    echo "[ERROR] Unknown DATASET_KNOB=${DATASET_KNOB}. Use crisp|omomo|behave." >&2
    exit 1
    ;;
esac

MOTION_DIR="${MOTION_DIR:-"${DEFAULT_MOTION_DIR}"}"
GEOMETRY_DIR="${GEOMETRY_DIR:-"${DEFAULT_GEOMETRY_DIR}"}"
OBJECT_URDF_DIR="${OBJECT_URDF_DIR:-"${DEFAULT_OBJECT_URDF_DIR}"}"
OBJECT_URDF="${OBJECT_URDF:-"${DEFAULT_OBJECT_URDF}"}"
OBJECT_URDF_MODE="${OBJECT_URDF_MODE:-"${DEFAULT_OBJECT_URDF_MODE}"}"
OBJECT_FILTER="${OBJECT_FILTER:-"${DEFAULT_OBJECT_FILTER}"}"

VIS_MODE=${CLI_VIS_MODE:-${VIS_MODE:-"kinematic"}}
EXP=${EXP:-"${DEFAULT_EXP}"}
ROBOT=${ROBOT:-"g1_29dof"}
PYTHON_BIN=${PYTHON_BIN:-python3}
HEADLESS_FLAG=${HEADLESS:-True}
NUM_ENVS=${NUM_ENVS:-1}
PORT=${PORT:-"$((RANDOM % 8976 + 1024))"}
START_CLIP=${START_CLIP:-""}
FPS=${FPS:-""}
AUTOPLAY=${AUTOPLAY:-True}
LOOP=${LOOP:-True}
PRELOAD=${PRELOAD:-False}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_SYNC_TO_SIM=${VISER_SYNC_TO_SIM:-True}
VISER_FORCE_DT=${VISER_FORCE_DT:-True}
VISER_RECENTER=${VISER_RECENTER:-True}
VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS:-False}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-1.0}
FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}
RESET_NOISE_SCALE=${RESET_NOISE_SCALE:-0.0}
VIS_GPU=${VIS_GPU:-auto}
DISABLE_RANDOMIZATION=${DISABLE_RANDOMIZATION:-True}

headless_lc=$(echo "${HEADLESS_FLAG}" | tr '[:upper:]' '[:lower:]')
case "${headless_lc}" in
  1|true|yes|on)
    HEADLESS_ENV=1
    HEADLESS_FLAG=True
    ;;
  0|false|no|off)
    HEADLESS_ENV=0
    HEADLESS_FLAG=False
    ;;
  *)
    echo "[WARN] Unknown HEADLESS='${HEADLESS_FLAG}', fallback to True."
    HEADLESS_ENV=1
    HEADLESS_FLAG=True
    ;;
esac
# IsaacLab launcher expects HEADLESS as integer env var.
export HEADLESS="${HEADLESS_ENV}"

# Select a single GPU for visualization to avoid PhysX scene creation failure on busy GPU0.
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" || -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  if [[ "${VIS_GPU}" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      _gpu_pick=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2n | head -n1 | cut -d',' -f1 | xargs)
      if [[ -n "${_gpu_pick}" ]]; then
        export CUDA_VISIBLE_DEVICES="${_gpu_pick}"
      fi
    fi
  else
    export CUDA_VISIBLE_DEVICES="${VIS_GPU}"
  fi
fi

if [[ ! -e "${MOTION_DIR}" ]]; then
  echo "[ERROR] motion dir not found: ${MOTION_DIR}" >&2
  exit 1
fi

OBJECT_SPEC=""
if [[ -n "${OBJECT_URDF}" ]]; then
  if [[ ! -f "${OBJECT_URDF}" ]]; then
    echo "[WARN] object urdf not found: ${OBJECT_URDF} (disabling object asset)"
  else
    OBJECT_SPEC="${OBJECT_URDF}"
  fi
elif [[ -n "${OBJECT_URDF_DIR}" ]]; then
  if [[ ! -d "${OBJECT_URDF_DIR}" ]]; then
    echo "[WARN] object urdf dir not found: ${OBJECT_URDF_DIR} (disabling object asset)"
  else
    OBJECT_SPEC="${OBJECT_URDF_DIR}"
  fi
fi

if [[ -z "${START_CLIP}" && -n "${OBJECT_FILTER}" && -d "${MOTION_DIR}" ]]; then
  IFS=',' read -r -a _filter_terms <<< "${OBJECT_FILTER}"
  while IFS= read -r _npz; do
    _stem=$(basename "${_npz}" .npz)
    _stem_lc=$(echo "${_stem}" | tr '[:upper:]' '[:lower:]')
    for _term in "${_filter_terms[@]}"; do
      _term_lc=$(echo "${_term}" | tr '[:upper:]' '[:lower:]' | xargs)
      if [[ -n "${_term_lc}" && "${_stem_lc}" == *"${_term_lc}"* ]]; then
        START_CLIP="${_stem}"
        break 2
      fi
    done
  done < <(find "${MOTION_DIR}" -maxdepth 1 -type f -name "*.npz" | sort)
fi

vis_mode_lc=$(echo "${VIS_MODE}" | tr '[:upper:]' '[:lower:]')
case "${vis_mode_lc}" in
  kinematic)
    if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1; then
import trimesh
import viser
import tyro
PY
      echo "[ERROR] Missing kinematic viewer dependencies in ${PYTHON_BIN} environment (need: trimesh, viser, tyro)." >&2
      exit 1
    fi

    cmd=(
      "${PYTHON_BIN}" src/holosoma/holosoma/viser_motion_geometry.py
      --motion-dir "${MOTION_DIR}"
      --robot "${ROBOT}"
      --port "${PORT}"
      --autoplay "${AUTOPLAY}"
      --loop "${LOOP}"
      --preload "${PRELOAD}"
      --object-urdf-mode "${OBJECT_URDF_MODE}"
    )

    if [[ -n "${GEOMETRY_DIR}" ]]; then
      if [[ -e "${GEOMETRY_DIR}" ]]; then
        cmd+=(--geometry-dir "${GEOMETRY_DIR}")
      else
        echo "[WARN] geometry path not found: ${GEOMETRY_DIR} (using ground-only view)"
      fi
    fi

    if [[ -n "${OBJECT_SPEC}" ]]; then
      if [[ -d "${OBJECT_SPEC}" ]]; then
        cmd+=(--object-urdf-dir "${OBJECT_SPEC}")
      else
        cmd+=(--object-urdf "${OBJECT_SPEC}")
      fi
    fi

    if [[ -n "${OBJECT_FILTER}" ]]; then
      cmd+=(--object-filter-csv "${OBJECT_FILTER}")
    fi

    if [[ -n "${START_CLIP}" ]]; then
      cmd+=(--start-clip "${START_CLIP}")
    fi

    if [[ -n "${FPS}" ]]; then
      cmd+=(--fps "${FPS}")
    fi

    echo "[INFO] Viewer backend: kinematic"
    echo "[INFO] DATASET_KNOB=${DATASET_KNOB}"
    echo "[INFO] motion_dir=${MOTION_DIR}"
    echo "[INFO] start_clip=${START_CLIP:-<auto>}"
    echo "[INFO] object_spec=${OBJECT_SPEC:-<none>}"
    echo "[INFO] preload=${PRELOAD}"
    echo "[INFO] viser=http://localhost:${PORT}"
    "${cmd[@]}"
    exit $?
    ;;
  replay)
    ;;
  *)
    echo "[ERROR] Unknown VIS_MODE=${VIS_MODE}. Use replay|kinematic." >&2
    exit 1
    ;;
esac

# Keep replay GUI controls consistent with infer/debug behavior.
export VISER_ENABLE_CLIP_GUI=${VISER_ENABLE_CLIP_GUI:-1}
export VISER_ENABLE_MANUAL_GUI=${VISER_ENABLE_MANUAL_GUI:-0}
export VISER_MANUAL_USE_HW_JOYSTICK=${VISER_MANUAL_USE_HW_JOYSTICK:-0}
export VISER_SHOW_TARGET_KEYPOINTS=${VISER_SHOW_TARGET_KEYPOINTS:-1}
export VISER_START_PAUSED=${VISER_START_PAUSED:-0}
export HOLOSOMA_REPLAY_KEEP_OPEN=${HOLOSOMA_REPLAY_KEEP_OPEN:-1}
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}

cmd=(
  "${PYTHON_BIN}" src/holosoma/holosoma/replay.py
  "exp:${EXP}"
  --training.headless="${HEADLESS_FLAG}"
  --training.num-envs="${NUM_ENVS}"
  --training.enable-viser=True
  --training.viser-port="${PORT}"
  --training.viser-env-id="${VISER_ENV_ID}"
  --training.viser-update-hz="${VISER_UPDATE_HZ}"
  --training.viser-sync-to-sim="${VISER_SYNC_TO_SIM}"
  --training.viser-force-dt="${VISER_FORCE_DT}"
  --training.viser-recenter="${VISER_RECENTER}"
  --training.viser-show-scandots="${VISER_SHOW_SCANDOTS}"
  --command.setup-terms.motion-command.params.motion-config.motion-file "${MOTION_DIR}"
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob "${START_AT_TIMESTEP_ZERO_PROB}"
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob "${FREEZE_AT_TIMESTEP_ZERO_PROB}"
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale "${RESET_NOISE_SCALE}"
)

if [[ -n "${START_CLIP}" ]]; then
  cmd+=(--command.setup-terms.motion-command.params.motion-config.motion-clip-name "${START_CLIP}")
fi

if [[ -n "${OBJECT_SPEC}" ]]; then
  cmd+=(--robot.object.enabled=True)
  cmd+=(--robot.object.object-urdf-path "${OBJECT_SPEC}")
fi

disable_randomization_lc=$(echo "${DISABLE_RANDOMIZATION}" | tr '[:upper:]' '[:lower:]')
case "${disable_randomization_lc}" in
  1|true|yes|on)
    cmd+=(randomization:disabled)
    ;;
  0|false|no|off)
    ;;
  *)
    echo "[WARN] Unknown DISABLE_RANDOMIZATION='${DISABLE_RANDOMIZATION}', leaving randomization config unchanged."
    ;;
esac

if [[ -n "${GEOMETRY_DIR}" ]]; then
  if [[ -e "${GEOMETRY_DIR}" ]]; then
    cmd+=(--terrain.terrain-term.mesh-type=LOAD_OBJ)
    cmd+=(--terrain.terrain-term.obj-file-path "${GEOMETRY_DIR}")
  else
    echo "[WARN] geometry path not found: ${GEOMETRY_DIR} (using default terrain)"
  fi
fi

echo "[INFO] Viewer backend: replay"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<default>}"
echo "[INFO] DATASET_KNOB=${DATASET_KNOB}"
echo "[INFO] motion_dir=${MOTION_DIR}"
echo "[INFO] start_clip=${START_CLIP:-<auto>}"
echo "[INFO] object_spec=${OBJECT_SPEC:-<none>}"
echo "[INFO] disable_randomization=${DISABLE_RANDOMIZATION}"
echo "[INFO] viser=http://localhost:${PORT}"

"${cmd[@]}"
