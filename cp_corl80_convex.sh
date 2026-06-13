#!/usr/bin/env bash
set -euo pipefail

# Copy the convex-hull CoRL solid80-clean AS distillation bank from NFS into
# this repo's data/ds_as_data tree.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "${SCRIPT_DIR}"

DEFAULT_AS_SUCCESS155_BANK_NAME="carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success155_bcleb5oi58000_final0p5_primitiveproj"
DEFAULT_CORL80_BANK_NAME="${DEFAULT_AS_SUCCESS155_BANK_NAME}_solid80_clean_box_bin_barrel_ball_convexhull"

CORL80_BANK_NAME=${CORL80_BANK_NAME:-"${DEFAULT_CORL80_BANK_NAME}"}
DEFAULT_NFS_CORL80_BANK="/nfs/zzzihanw/ds_as_data/_distill/${CORL80_BANK_NAME}"
DEFAULT_NFS_CORL80_TAR="${DEFAULT_NFS_CORL80_BANK}.tar"

if [[ -z "${NFS_CORL80_BANK:-}" ]]; then
  if [[ -f "${DEFAULT_NFS_CORL80_TAR}" ]]; then
    NFS_CORL80_BANK="${DEFAULT_NFS_CORL80_TAR}"
  else
    NFS_CORL80_BANK="${DEFAULT_NFS_CORL80_BANK}"
  fi
fi

export CORL_BANK_NAME="${CORL_BANK_NAME:-${CORL80_BANK_NAME}}"
export NFS_CORL_BANK="${NFS_CORL_BANK:-${NFS_CORL80_BANK}}"
export LOCAL_BANK_NAME="${LOCAL_BANK_NAME:-${CORL80_BANK_NAME}}"
export EXPECTED_CLIP_COUNT="${EXPECTED_CLIP_COUNT:-79}"

echo "[INFO] Copying CoRL solid80-clean convex-hull bank for distill_as_button_solid_convex.sh"
echo "[INFO] NFS_CORL_BANK=${NFS_CORL_BANK}"
echo "[INFO] LOCAL_BANK_NAME=${LOCAL_BANK_NAME}"
echo "[INFO] EXPECTED_CLIP_COUNT=${EXPECTED_CLIP_COUNT}"

exec bash "${SCRIPT_DIR}/cp_corl.sh"
