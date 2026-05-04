#!/usr/bin/env bash
set -euo pipefail

# OMOMO+DS128 setup, but loosen global reference position tracking.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export DATA_MODE=mix-naive
export AUTO_PREP_DS_BANK=${AUTO_PREP_DS_BANK:-0}
export STRICT_DEFAULT_DS_BANK_VALIDATION=${STRICT_DEFAULT_DS_BANK_VALIDATION:-0}
export DATA_SUBSET_SEED=${DATA_SUBSET_SEED:-0}

resolve_omomo_ds128_source_dir() {
  local candidate
  for candidate in \
    "${OMOMO_DS128_SOURCE_DIR:-}" \
    "${SCRIPT_DIR}/data/ds_box_data/scale_mix_all/train_g1_w_obj_prepared_plus_omomo_orig" \
    "/nfs/zzzihanw/ds_box_data/scale_mix_all/train_g1_w_obj_prepared_plus_omomo_orig"
  do
    [[ -z "${candidate}" ]] && continue
    if [[ -d "${candidate}" && -f "${candidate}/_clip_object_urdf_map.json" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

resolve_omomo_ds128_subset_dir() {
  local candidate
  for candidate in \
    "${OMOMO_DS128_MOTION_DIR:-}" \
    "${SCRIPT_DIR}/data/ds_box_data/scale_mix_all/train_g1_w_obj_prepared_plus_omomo_orig_omomo_ds128_seed${DATA_SUBSET_SEED}"
  do
    [[ -z "${candidate}" ]] && continue
    if [[ -d "${candidate}" && -f "${candidate}/_clip_object_urdf_map.json" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

unset OBJECT_SPEC_PATH
if MOTION_DIR="$(resolve_omomo_ds128_subset_dir)"; then
  export MOTION_DIR
  export OBJECT_SPEC_PATH="${MOTION_DIR}/_clip_object_urdf_map.json"
  unset DATA_SUBSET_MODE
else
  if ! MOTION_DIR="$(resolve_omomo_ds128_source_dir)"; then
    echo "[ERROR] Could not find OMOMO+DS128 data or its scale_mix_all source bank." >&2
    echo "[ERROR] Expected one of:" >&2
    echo "[ERROR]   OMOMO_DS128_MOTION_DIR=<prepared subset bank>" >&2
    echo "[ERROR]   OMOMO_DS128_SOURCE_DIR=<scale_mix_all/train_g1_w_obj_prepared_plus_omomo_orig>" >&2
    echo "[ERROR]   ${SCRIPT_DIR}/data/ds_box_data/scale_mix_all/train_g1_w_obj_prepared_plus_omomo_orig" >&2
    echo "[ERROR]   /nfs/zzzihanw/ds_box_data/scale_mix_all/train_g1_w_obj_prepared_plus_omomo_orig" >&2
    exit 2
  fi
  export MOTION_DIR
  export DATA_SUBSET_MODE=omomo+ds128
fi

export NPROC=${NPROC:-8}
export PER_GPU_ENVS=${PER_GPU_ENVS:-4096}
export NUM_ENVS=${NUM_ENVS:-$((NPROC * PER_GPU_ENVS))}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1000}

export CLIP_WEIGHTING_STRATEGY=${CLIP_WEIGHTING_STRATEGY:-success_rate_adaptive}
export USE_ADAPTIVE_TIMESTEPS_SAMPLER=${USE_ADAPTIVE_TIMESTEPS_SAMPLER:-False}
export FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}
export GENERALIST_CONTACT_REWARD_MODE=${GENERALIST_CONTACT_REWARD_MODE:-tanh}
export SEQUENCE_NAME=${SEQUENCE_NAME:-rootpos-w025-sigma06}

ROOT_POS_WEIGHT=${ROOT_POS_WEIGHT:-0.25}
ROOT_POS_SIGMA_OVERRIDE=${ROOT_POS_SIGMA_OVERRIDE:-0.6}
export REFERENCE_ROOT_POS_W="${ROOT_POS_WEIGHT}"
export REFERENCE_ROOT_POS_SIGMA="${ROOT_POS_SIGMA_OVERRIDE}"

if [[ ! -d "${MOTION_DIR}" ]]; then
  echo "[ERROR] MOTION_DIR does not exist: ${MOTION_DIR}" >&2
  exit 2
fi
if [[ -n "${OBJECT_SPEC_PATH:-}" && ! -f "${OBJECT_SPEC_PATH}" ]]; then
  echo "[ERROR] OBJECT_SPEC_PATH does not exist: ${OBJECT_SPEC_PATH}" >&2
  exit 2
fi

exec bash "${SCRIPT_DIR}/train_object_generalist_ds.sh" "${SEQUENCE_NAME}"
