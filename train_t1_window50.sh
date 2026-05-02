#!/usr/bin/env bash
set -euo pipefail

# Current mixed-bank setup, but density-boost sampling around loaded contact t1.
# The target fraction applies to clips that have loaded t1/t2 contact windows.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export DATA_MODE=mix-naive
export AUTO_PREP_DS_BANK=${AUTO_PREP_DS_BANK:-0}
export STRICT_DEFAULT_DS_BANK_VALIDATION=${STRICT_DEFAULT_DS_BANK_VALIDATION:-0}
export MOTION_DIR=${MOTION_DIR:-"${SCRIPT_DIR}/data/ds_box_data/train_g1_w_obj_prepared_plus_omomo_orig"}
export OBJECT_SPEC_PATH=${OBJECT_SPEC_PATH:-"${MOTION_DIR}/_clip_object_urdf_map.json"}

export NPROC=${NPROC:-8}
export PER_GPU_ENVS=${PER_GPU_ENVS:-4096}
export NUM_ENVS=${NUM_ENVS:-$((NPROC * PER_GPU_ENVS))}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1000}

export CLIP_WEIGHTING_STRATEGY=${CLIP_WEIGHTING_STRATEGY:-success_rate_adaptive}
export USE_ADAPTIVE_TIMESTEPS_SAMPLER=False
export FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}
export GENERALIST_CONTACT_REWARD_MODE=${GENERALIST_CONTACT_REWARD_MODE:-tanh}
export SEQUENCE_NAME=${SEQUENCE_NAME:-t1-window70-frac50}

CONTACT_INTERVAL_ROOT=${CONTACT_INTERVAL_ROOT:-"${SCRIPT_DIR}/outputs/clips"}
T1_WINDOW_HALF_WIDTH_STEPS=${T1_WINDOW_HALF_WIDTH_STEPS:-70}
T1_WINDOW_TARGET_SAMPLE_FRAC=${T1_WINDOW_TARGET_SAMPLE_FRAC:-0.5}

if [[ ! -d "${MOTION_DIR}" ]]; then
  echo "[ERROR] MOTION_DIR does not exist: ${MOTION_DIR}" >&2
  exit 2
fi
if [[ ! -f "${OBJECT_SPEC_PATH}" ]]; then
  echo "[ERROR] OBJECT_SPEC_PATH does not exist: ${OBJECT_SPEC_PATH}" >&2
  exit 2
fi
if [[ ! -d "${CONTACT_INTERVAL_ROOT}" ]]; then
  echo "[ERROR] CONTACT_INTERVAL_ROOT does not exist: ${CONTACT_INTERVAL_ROOT}" >&2
  exit 2
fi

exec bash "${SCRIPT_DIR}/train_object_generalist_ds.sh" mix-naive \
  --command.setup-terms.motion-command.params.motion-config.adaptive-sampling-contact-interval-root="${CONTACT_INTERVAL_ROOT}" \
  --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-sampling-enabled=True \
  --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-half-width-steps="${T1_WINDOW_HALF_WIDTH_STEPS}" \
  --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-target-sample-frac="${T1_WINDOW_TARGET_SAMPLE_FRAC}"
