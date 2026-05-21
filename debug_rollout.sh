#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash ./debug_rollout.sh [extra viewer args...]

Visualizes exported teacher rollout/contact products with a Viser GUI.

Environment overrides:
  ROLLOUT_ASSET_ROOT  Optional root containing outputs/, outputs_vis/, outputs_sts/.
                      Default: repo root when local outputs exist, else
                      /nfs/zzzihanw/tao/teacher_rollout_assets_20260415 if present.
  DATA_ROOT           Default: ${ROLLOUT_ASSET_ROOT}/outputs or ./outputs
  VIS_ROOT            Default: ${ROLLOUT_ASSET_ROOT}/outputs_vis or ./outputs_vis
  STATS_ROOT          Default: ${ROLLOUT_ASSET_ROOT}/outputs_sts or ./outputs_sts
  SEQUENCE            Optional initial clip id or clip directory name.
  VISER_PORT          Default: 18092
  VISER_HOST          Default: 0.0.0.0
  ROBOT_URDF          G1 URDF rendered in the rollout view.
                      Default: src/holosoma/holosoma/data/robots/g1/g1_29dof.urdf
  ORIGINAL_MOTION_DIR Original input motion directory to compare against rollout.
                      Default: data/ds_box_data/train_g1_w_obj_prepared
  SHOW_ROBOT          Default: 1; set 0/false to hide the training G1 overlay.
  AUTOPLAY            Default: 0; set 1/true to start playback immediately.
  PLAYBACK_FULL_MOTION Default: 0; set 1/true to play the full motion timeline,
                      not only valid rollout steps.
  PLAYBACK_FPS        Default: 30; initial playback FPS.
  LOOP                Default: 1; set 0/false to advance instead of looping.
  REPLAY_ONLY         Default: 0; set 1/true to skip contact/path/static overlays
                      and only replay object + G1 meshes.
  SUCCESS_ONLY        Default: 0; set 1/true to only show success=True clips.
  STRICT_SUCCESS_ONLY Default: 0; set 1/true to only show stable-contact +
                      final-position success clips.
  SOLID_ONLY          Default: 0; set 1/true to only show box/bin/barrel/ball.
  EXCLUDE_CLIPS       Optional comma/space-separated clip ids to remove.
  EXCLUDE_CLIPS_FILE  Optional text file with one clip id per line to remove.
  LIST_ONLY           Default: 0; print available sequences and exit.
  DRY_RUN             Default: 0; print command without running.

Examples:
  bash ./debug_rollout.sh
  SEQUENCE=box_10 bash ./debug_rollout.sh
  ROLLOUT_ASSET_ROOT=/nfs/zzzihanw/tao/teacher_rollout_assets_20260415 bash ./debug_rollout.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

is_truthy() {
  case "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
VISER_PORT="${VISER_PORT:-18092}"
VISER_HOST="${VISER_HOST:-0.0.0.0}"
ROBOT_URDF="${ROBOT_URDF:-${ROOT_DIR}/src/holosoma/holosoma/data/robots/g1/g1_29dof.urdf}"
ORIGINAL_MOTION_DIR="${ORIGINAL_MOTION_DIR:-${ROOT_DIR}/data/ds_box_data/train_g1_w_obj_prepared}"
SHOW_ROBOT="${SHOW_ROBOT:-1}"
AUTOPLAY="${AUTOPLAY:-0}"
PLAYBACK_FULL_MOTION="${PLAYBACK_FULL_MOTION:-0}"
PLAYBACK_FPS="${PLAYBACK_FPS:-30}"
LOOP="${LOOP:-1}"
REPLAY_ONLY="${REPLAY_ONLY:-0}"
SUCCESS_ONLY="${SUCCESS_ONLY:-0}"
STRICT_SUCCESS_ONLY="${STRICT_SUCCESS_ONLY:-0}"
SOLID_ONLY="${SOLID_ONLY:-0}"
EXCLUDE_CLIPS="${EXCLUDE_CLIPS:-}"
EXCLUDE_CLIPS_FILE="${EXCLUDE_CLIPS_FILE:-}"
LIST_ONLY="${LIST_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "${ROLLOUT_ASSET_ROOT:-}" ]]; then
  DEFAULT_DATA_ROOT="${ROLLOUT_ASSET_ROOT}/outputs"
  DEFAULT_VIS_ROOT="${ROLLOUT_ASSET_ROOT}/outputs_vis"
  DEFAULT_STATS_ROOT="${ROLLOUT_ASSET_ROOT}/outputs_sts"
elif [[ -d "${ROOT_DIR}/outputs/clips" ]]; then
  DEFAULT_DATA_ROOT="${ROOT_DIR}/outputs"
  DEFAULT_VIS_ROOT="${ROOT_DIR}/outputs_vis"
  DEFAULT_STATS_ROOT="${ROOT_DIR}/outputs_sts"
elif [[ -d "/nfs/zzzihanw/tao/teacher_rollout_assets_20260415/outputs/clips" ]]; then
  DEFAULT_DATA_ROOT="/nfs/zzzihanw/tao/teacher_rollout_assets_20260415/outputs"
  DEFAULT_VIS_ROOT="/nfs/zzzihanw/tao/teacher_rollout_assets_20260415/outputs_vis"
  DEFAULT_STATS_ROOT="/nfs/zzzihanw/tao/teacher_rollout_assets_20260415/outputs_sts"
else
  DEFAULT_DATA_ROOT="${ROOT_DIR}/outputs"
  DEFAULT_VIS_ROOT="${ROOT_DIR}/outputs_vis"
  DEFAULT_STATS_ROOT="${ROOT_DIR}/outputs_sts"
fi

DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
VIS_ROOT="${VIS_ROOT:-${DEFAULT_VIS_ROOT}}"
STATS_ROOT="${STATS_ROOT:-${DEFAULT_STATS_ROOT}}"

if [[ ! -d "${DATA_ROOT}/clips" ]]; then
  echo "[ERROR] Missing ${DATA_ROOT}/clips. Run cp_tao.sh or set ROLLOUT_ASSET_ROOT/DATA_ROOT." >&2
  exit 2
fi
if [[ ! -d "${STATS_ROOT}/clips" ]]; then
  echo "[ERROR] Missing ${STATS_ROOT}/clips. Run cp_tao.sh or set STATS_ROOT." >&2
  exit 2
fi

cmd=(
  "${PYTHON_BIN}" -m holosoma.debug_rollout_viewer
  --data-root "${DATA_ROOT}"
  --vis-root "${VIS_ROOT}"
  --stats-root "${STATS_ROOT}"
  --original-motion-dir "${ORIGINAL_MOTION_DIR}"
  --host "${VISER_HOST}"
  --port "${VISER_PORT}"
)

if [[ "${SHOW_ROBOT}" == "0" || "${SHOW_ROBOT,,}" == "false" ]]; then
  cmd+=(--no-robot)
else
  cmd+=(--robot-urdf "${ROBOT_URDF}")
fi
if [[ -n "${SEQUENCE:-}" ]]; then
  cmd+=(--sequence "${SEQUENCE}")
fi
if is_truthy "${AUTOPLAY}"; then
  cmd+=(--autoplay)
fi
if is_truthy "${PLAYBACK_FULL_MOTION}"; then
  cmd+=(--playback-full-motion)
fi
cmd+=(--fps "${PLAYBACK_FPS}")
if ! is_truthy "${LOOP}"; then
  cmd+=(--no-loop)
fi
if is_truthy "${REPLAY_ONLY}"; then
  cmd+=(--replay-only)
fi
if is_truthy "${SUCCESS_ONLY}"; then
  cmd+=(--success-only)
fi
if is_truthy "${STRICT_SUCCESS_ONLY}"; then
  cmd+=(--strict-success-only)
fi
if is_truthy "${SOLID_ONLY}"; then
  cmd+=(--solid-only)
fi
if [[ -n "${EXCLUDE_CLIPS}" ]]; then
  for clip_id in ${EXCLUDE_CLIPS//,/ }; do
    if [[ -n "${clip_id}" ]]; then
      cmd+=(--exclude-clip "${clip_id}")
    fi
  done
fi
if [[ -n "${EXCLUDE_CLIPS_FILE}" ]]; then
  cmd+=(--exclude-clips-file "${EXCLUDE_CLIPS_FILE}")
fi
if [[ "${LIST_ONLY}" == "1" || "${LIST_ONLY,,}" == "true" ]]; then
  cmd+=(--list-only)
fi
if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] VIS_ROOT=${VIS_ROOT}"
echo "[INFO] STATS_ROOT=${STATS_ROOT}"
echo "[INFO] ORIGINAL_MOTION_DIR=${ORIGINAL_MOTION_DIR}"
echo "[INFO] ROBOT_URDF=${ROBOT_URDF}"
echo "[INFO] AUTOPLAY=${AUTOPLAY} PLAYBACK_FULL_MOTION=${PLAYBACK_FULL_MOTION} PLAYBACK_FPS=${PLAYBACK_FPS} LOOP=${LOOP} REPLAY_ONLY=${REPLAY_ONLY} SUCCESS_ONLY=${SUCCESS_ONLY} STRICT_SUCCESS_ONLY=${STRICT_SUCCESS_ONLY} SOLID_ONLY=${SOLID_ONLY} EXCLUDE_CLIPS=${EXCLUDE_CLIPS} EXCLUDE_CLIPS_FILE=${EXCLUDE_CLIPS_FILE}"
echo "[INFO] VISER_URL=http://localhost:${VISER_PORT}"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
  printf '[DRY_RUN]'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${cmd[@]}"
