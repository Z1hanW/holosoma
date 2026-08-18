#!/usr/bin/env bash
set -euo pipefail

# Pure-RL contact-window bootstrap.
#
# This launcher is intentionally separate from debug_pure_rl_bootstrap.sh so the
# existing baseline/staged runs keep their previous defaults. It targets the
# failure mode where pure PPO quickly learns survival/body tracking but never
# discovers grasp/lift.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "${SCRIPT_DIR}"

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}

export RUN_STAMP
export NPROC=${NPROC:-8}
export NNODES=${NNODES:-1}
export PER_GPU_ENVS=${PER_GPU_ENVS:-2048}
export NUM_MINI_BATCHES=${NUM_MINI_BATCHES:-$((NPROC * NNODES))}
export NUM_LEARNING_EPOCHS=${NUM_LEARNING_EPOCHS:-2}
export NUM_LEARNING_ITERATIONS=${NUM_LEARNING_ITERATIONS:-40000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-500}

# Keep PPO in the stable range observed in xcuib3we. The old 1e-3 / 0.01-noise
# run saturated action clipping and drove action-rate loss to O(1e9).
export ACTOR_LR=${ACTOR_LR:-3e-4}
export CRITIC_LR=${CRITIC_LR:-3e-4}
export INIT_NOISE_STD=${INIT_NOISE_STD:-0.30}
export ACTOR_MIN_NOISE_STD=${ACTOR_MIN_NOISE_STD:-0.05}
export ENTROPY_COEF=${ENTROPY_COEF:-0.001}
export CLIP_PARAM=${CLIP_PARAM:-0.1}
export MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.5}

# Sample both from the beginning and from the contact/lift window. This shortens
# credit assignment without converting the run into distillation.
export START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.30}
export USE_ADAPTIVE_TIMESTEPS_SAMPLER=${USE_ADAPTIVE_TIMESTEPS_SAMPLER:-False}
export RESET_NOISE_SCALE=${RESET_NOISE_SCALE:-0.08}

# Keep early episodes alive; tighten in a later run only after lift/contact
# appears in checkpoint inference.
export PURE_RL_BAD_REF_POS_THRESHOLD=${PURE_RL_BAD_REF_POS_THRESHOLD:-1.0}
export PURE_RL_BAD_REF_ORI_THRESHOLD=${PURE_RL_BAD_REF_ORI_THRESHOLD:-1.5}
export PURE_RL_BAD_MOTION_BODY_POS_THRESHOLD=${PURE_RL_BAD_MOTION_BODY_POS_THRESHOLD:-0.5}
export PURE_RL_BAD_OBJECT_POS_THRESHOLD=${PURE_RL_BAD_OBJECT_POS_THRESHOLD:-1.0}
export PURE_RL_BAD_OBJECT_ORI_THRESHOLD=${PURE_RL_BAD_OBJECT_ORI_THRESHOLD:-1.5}

# Bootstrap contact as a dense position prior first. Force-gated/stable contact
# is too sparse before the hands have learned to reach the object.
export PURE_RL_WRIST_WEIGHT=${PURE_RL_WRIST_WEIGHT:-3.0}
export PURE_RL_CONTACT_WEIGHT=${PURE_RL_CONTACT_WEIGHT:-5.0}
export PURE_RL_CONTACT_USE_FORCE_TERM=${PURE_RL_CONTACT_USE_FORCE_TERM:-False}
export PURE_RL_CONTACT_FORCE_GATE_MODE=${PURE_RL_CONTACT_FORCE_GATE_MODE:-soft}
export PURE_RL_REQUIRE_STABLE_CONTACT=${PURE_RL_REQUIRE_STABLE_CONTACT:-False}
export PURE_RL_USE_CONTACT_SCHEDULE=${PURE_RL_USE_CONTACT_SCHEDULE:-True}
export PURE_RL_CONTACT_SCHEDULE_RELAX_STEPS=${PURE_RL_CONTACT_SCHEDULE_RELAX_STEPS:-25}
export PURE_RL_POSITION_SIGMA=${PURE_RL_POSITION_SIGMA:-0.16}

# Do not push objects during the discovery phase.
export PUSH_INTERVAL_S=${PUSH_INTERVAL_S:-'[1000000.0,1000001.0]'}
export PUSH_MAX_VEL=${PUSH_MAX_VEL:-'[0.0,0.0,0.0,0.0,0.0,0.0]'}

export RUN_NAME=${RUN_NAME:-debug_pure_rl_contact_window_bootstrap_debug30_bigmlp_${RUN_STAMP}}
export TRAINING_NAME=${TRAINING_NAME:-debug_pure_rl_contact_window_bootstrap_debug30_bigmlp_${RUN_STAMP}}

contact_window_args=(
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob=0.0
  --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-target-sample-frac=0.65
  --reward.terms.object-global-ref-position-error-exp.weight=2.0
  --reward.terms.object-global-ref-orientation-error-exp.weight=0.5
  --reward.terms.motion-global-body-lin-vel.weight=0.6
  --reward.terms.motion-global-body-ang-vel.weight=0.6
)

echo "[INFO] pure_rl_contact_window_bootstrap=1"
echo "[INFO] reset start_at_zero=${START_AT_TIMESTEP_ZERO_PROB} uniform_t1_target=0.65 adaptive=${USE_ADAPTIVE_TIMESTEPS_SAMPLER} reset_noise=${RESET_NOISE_SCALE}"
echo "[INFO] ppo lr=${ACTOR_LR}/${CRITIC_LR} init_noise=${INIT_NOISE_STD} min_noise=${ACTOR_MIN_NOISE_STD} entropy=${ENTROPY_COEF}"
echo "[INFO] contact wrist=${PURE_RL_WRIST_WEIGHT} contact=${PURE_RL_CONTACT_WEIGHT} sigma=${PURE_RL_POSITION_SIGMA} relax=${PURE_RL_CONTACT_SCHEDULE_RELAX_STEPS}"

exec bash "${SCRIPT_DIR}/debug_pure_rl_bootstrap.sh" "${contact_window_args[@]}" "$@"
