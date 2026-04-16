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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
VISER_PORT="${VISER_PORT:-18092}"
VISER_HOST="${VISER_HOST:-0.0.0.0}"
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
  --host "${VISER_HOST}"
  --port "${VISER_PORT}"
)

if [[ -n "${SEQUENCE:-}" ]]; then
  cmd+=(--sequence "${SEQUENCE}")
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
echo "[INFO] VISER_URL=http://localhost:${VISER_PORT}"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
  printf '[DRY_RUN]'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${cmd[@]}"
