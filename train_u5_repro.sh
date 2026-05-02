#!/usr/bin/env bash
set -euo pipefail

# Reproduce the key training settings from W&B run u5lguxvl using the current code.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
export DATA_MODE=mix-naive
export AUTO_PREP_DS_BANK=${AUTO_PREP_DS_BANK:-0}
export STRICT_DEFAULT_DS_BANK_VALIDATION=${STRICT_DEFAULT_DS_BANK_VALIDATION:-0}
export MOTION_DIR=${MOTION_DIR:-"${SCRIPT_DIR}/data/ds_box_data/train_g1_w_obj_prepared_plus_omomo_orig"}
export OBJECT_SPEC_PATH=${OBJECT_SPEC_PATH:-"${MOTION_DIR}/_clip_object_urdf_map.json"}

export NPROC=${NPROC:-6}
export PER_GPU_ENVS=${PER_GPU_ENVS:-10240}
export NUM_ENVS=${NUM_ENVS:-$((NPROC * PER_GPU_ENVS))}
export TRAINING_SEED=${TRAINING_SEED:-42}
export SAVE_INTERVAL=${SAVE_INTERVAL:-500}
export NUM_LEARNING_EPOCHS=${NUM_LEARNING_EPOCHS:-8}
export ACTOR_LR=${ACTOR_LR:-1e-05}
export CRITIC_LR=${CRITIC_LR:-1e-05}

export CLIP_WEIGHTING_STRATEGY=${CLIP_WEIGHTING_STRATEGY:-success_rate_adaptive}
export USE_ADAPTIVE_TIMESTEPS_SAMPLER=${USE_ADAPTIVE_TIMESTEPS_SAMPLER:-True}
export FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.95}
export DEFAULT_POSE_PREPEND_ENABLED=${DEFAULT_POSE_PREPEND_ENABLED:-1}
export DEFAULT_POSE_PREPEND_DURATION_S=${DEFAULT_POSE_PREPEND_DURATION_S:-0.2}

export GENERALIST_CONTACT_REWARD_MODE=${GENERALIST_CONTACT_REWARD_MODE:-tanh}
export GENERALIST_CONTACT_REWARD_THRESHOLD=${GENERALIST_CONTACT_REWARD_THRESHOLD:-1.0}
export GENERALIST_CONTACT_REWARD_FORCE_SCALE=${GENERALIST_CONTACT_REWARD_FORCE_SCALE:-25.0}
export SEQUENCE_NAME=${SEQUENCE_NAME:-u5-repro-current-code}

if [[ ! -d "${MOTION_DIR}" ]]; then
  echo "[ERROR] MOTION_DIR does not exist: ${MOTION_DIR}" >&2
  exit 2
fi
if [[ ! -f "${OBJECT_SPEC_PATH}" ]]; then
  echo "[ERROR] OBJECT_SPEC_PATH does not exist: ${OBJECT_SPEC_PATH}" >&2
  exit 2
fi

exec bash "${SCRIPT_DIR}/train_object_generalist_ds.sh" mix-naive \
  --observation-overrides.disable-actor-history=False \
  --observation-overrides.disable-critic-history=False \
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=0.2 \
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-append=True \
  --command.setup-terms.motion-command.params.motion-config.default-pose-append-duration-s=2.0
