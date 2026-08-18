#!/usr/bin/env bash
set -euo pipefail

# Isolated single-node launcher for the velocity-conditioned tracking/task
# curriculum. Existing hybrid, distillation, and pure-RL presets are not used.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "${SCRIPT_DIR}"

source "${SCRIPT_DIR}/scripts/gpu_launch_defaults.sh"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

NPROC=${NPROC:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-32341}
PER_GPU_ENVS=${PER_GPU_ENVS:-1024}
NUM_ENVS=${NUM_ENVS:-$((NPROC * NNODES * PER_GPU_ENVS))}
TRAINING_SEED=${TRAINING_SEED:-42}

if [[ "${NPROC}" != 8 || "${NNODES}" != 1 || "${NODE_RANK}" != 0 ]]; then
  echo "[ERROR] train_hybrid_velocity.sh owns one node with exactly 8 ranks." >&2
  exit 2
fi
if (( PER_GPU_ENVS <= 0 || NUM_ENVS != NPROC * PER_GPU_ENVS )); then
  echo "[ERROR] NUM_ENVS must equal 8 * PER_GPU_ENVS; got ${NUM_ENVS} and ${PER_GPU_ENVS}." >&2
  exit 2
fi

DEFAULT_VIEW="data/ds_as_data/debug30_original_realmesh_cominertia_categorymass_v2_scientific_single_slot__src_1f3966245545689a2e14909dda31d6673790bdf888168362449a0ac1902f3510"
MOTION_DIR=$(realpath -e "${MOTION_DIR:-${DEFAULT_VIEW}}")
OBJECT_SPEC_PATH=$(realpath -e "${OBJECT_SPEC_PATH:-${MOTION_DIR}/_clip_object_urdf_map.json}")
CONTACT_EXPORT_ROOT=$(realpath -e "${CONTACT_EXPORT_ROOT:-${MOTION_DIR}/contact_export_from_teacher_realmesh_rollout}")

for forbidden in \
  RESUME_CKPT RESUME_CHECKPOINT RESUME_MODEL_FILE RESUME_STEP \
  POLICY_INIT_CKPT POLICY_INIT_CHECKPOINT RESUME_FROM_BOX BOX_RESUME_CKPT \
  TEACHER_CHECKPOINT TEACHER_CHECKPOINT_EXPECTED_SHA256; do
  if [[ -n "${!forbidden:-}" ]]; then
    echo "[ERROR] ${forbidden} is set; this contract requires a fresh actor, critic, and optimizer." >&2
    exit 2
  fi
done

if [[ "${HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED:-0}" != 1 ]]; then
  echo "[ERROR] HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED=1 is required." >&2
  exit 2
fi
for required_path in \
  "${HOLOSOMA_RANK_LOCAL_MOTION_ROOT:-}" \
  "${HOLOSOMA_MOTION_SHARD_MANIFEST:-}"; do
  if [[ -z "${required_path}" || ! -e "${required_path}" ]]; then
    echo "[ERROR] missing authenticated rank-shard path: ${required_path:-<unset>}" >&2
    exit 2
  fi
done

export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
export ACCEPT_EULA=${ACCEPT_EULA:-Y}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HOLOSOMA_OBJECT_SPAWN_MODE=single_slot_multi_urdf
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=0
export HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS=1
export HOLOSOMA_REQUIRE_OBJECT_MESH_ASSETS=1
export HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE=mesh
export HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES=True
export HOLOSOMA_PERCEPTION_INCLUDE_ROBOT_MESH=1
export HOLOSOMA_OBJECT_COLLIDER_TYPE=convex_decomposition
export HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=0
export HOLOSOMA_REQUIRE_CONTACT_INTERVAL_COVERAGE=1
export HOLOSOMA_REQUIRE_CONTACT_TARGET_COVERAGE=1
export HOLOSOMA_SKIP_INITIAL_CHECKPOINT=1
unset HOLOSOMA_SKIP_CHECKPOINT_UPLOAD WANDB_SKIP_CHECKPOINT_UPLOAD

RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}
TRAINING_PROJECT=${TRAINING_PROJECT:-carry-any}
TRAINING_NAME=${TRAINING_NAME:-student_hybrid_velocity_original30_ws8_e${PER_GPU_ENVS}_${RUN_STAMP}}
LOGGER=${LOGGER:-logger:disabled}
WANDB_ENTITY=${WANDB_ENTITY:-zihanw22}
WANDB_RUN_ID=${WANDB_RUN_ID:-}
LOGGER_BASE_DIR=${LOGGER_BASE_DIR:-${SCRIPT_DIR}/logs/hybrid_velocity/${TRAINING_NAME}}

NUM_LEARNING_ITERATIONS=${NUM_LEARNING_ITERATIONS:-40000}
NUM_STEPS_PER_ENV=${NUM_STEPS_PER_ENV:-24}
NUM_LEARNING_EPOCHS=${NUM_LEARNING_EPOCHS:-7}
NUM_MINI_BATCHES=${NUM_MINI_BATCHES:-4}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
ACTOR_LR=${ACTOR_LR:-0.001}
CRITIC_LR=${CRITIC_LR:-0.001}
TASK_FRACTION_START=${TASK_FRACTION_START:-0.0}
TASK_FRACTION_END=${TASK_FRACTION_END:-0.5}
TASK_FRACTION_START_ITER=${TASK_FRACTION_START_ITER:-0}
TASK_FRACTION_END_ITER=${TASK_FRACTION_END_ITER:-5000}
FORWARD_COMMAND_MPS=${FORWARD_COMMAND_MPS:-0.5}
LIFT_HEIGHT_M=${LIFT_HEIGHT_M:-0.10}
RESET_CURRICULUM_END_ITER=${RESET_CURRICULUM_END_ITER:-$((NUM_LEARNING_ITERATIONS - 1))}

if [[ "${SAVE_INTERVAL}" != 1000 ]]; then
  echo "[ERROR] formal checkpoint cadence is fixed at SAVE_INTERVAL=1000." >&2
  exit 2
fi

TRAIN_ARGS=(
  exp:g1-29dof-wbt-w-object-hybrid-velocity
  command:g1-29dof-wbt-w-object-hybrid-velocity
  observation:g1-29dof-wbt-w-object-hybrid-velocity
  reward:g1-29dof-wbt-w-object-hybrid-velocity
  termination:g1-29dof-wbt-hybrid-velocity
  perception:camera_depth_d435i
  randomization:g1_29dof_wbt_w_object_with_action_delay
  "${LOGGER}"
  --training.project="${TRAINING_PROJECT}"
  --training.name="${TRAINING_NAME}"
  --training.num-envs="${NUM_ENVS}"
  --training.seed="${TRAINING_SEED}"
  --training.multigpu=True
  --training.export-onnx=True
  --algo.config.distill.enabled=False
  --algo.config.num-learning-iterations="${NUM_LEARNING_ITERATIONS}"
  --algo.config.num-steps-per-env="${NUM_STEPS_PER_ENV}"
  --algo.config.num-learning-epochs="${NUM_LEARNING_EPOCHS}"
  --algo.config.num-mini-batches="${NUM_MINI_BATCHES}"
  --algo.config.clip-param=0.2
  --algo.config.gamma=0.99
  --algo.config.lam=0.95
  --algo.config.value-loss-coef=1.0
  --algo.config.entropy-coef=0.005
  --algo.config.max-grad-norm=1.0
  --algo.config.schedule=adaptive
  --algo.config.desired-kl=0.01
  --algo.config.actor-learning-rate="${ACTOR_LR}"
  --algo.config.critic-learning-rate="${CRITIC_LR}"
  --algo.config.min-actor-learning-rate=0.00001
  --algo.config.max-actor-learning-rate=0.01
  --algo.config.min-critic-learning-rate=0.00001
  --algo.config.max-critic-learning-rate=0.01
  --algo.config.init-noise-std=1.0
  --algo.config.module-dict.actor.min-noise-std=0.01
  --algo.config.normalize-actor-obs=False
  --algo.config.normalize-critic-obs=False
  --algo.config.save-interval="${SAVE_INTERVAL}"
  --algo.config.reset-rollout-at-checkpoint=False
  --algo.config.module-dict.actor.type=MLP
  --algo.config.module-dict.actor.input-dim="['actor_obs_root_contact_aware','actor_obs_drop_button','actor_obs_proprio_with_actions_no_linvel']"
  --algo.config.module-dict.actor.layer-config.hidden-dims='[512,256,128]'
  --algo.config.module-dict.critic.layer-config.hidden-dims='[512,256,128]'
  --observation.groups.actor_obs_root_contact_aware.history-length=1
  --observation.groups.actor_obs_drop_button.history-length=1
  --observation.groups.actor_obs_proprio_with_actions_no_linvel.history-length=1
  --observation.groups.critic_proprio_history.history-length=1
  --command.setup-terms.motion-command.params.motion-config.motion-file="${MOTION_DIR}"
  --command.setup-terms.motion-command.params.motion-config.hybrid-velocity-enabled=True
  --command.setup-terms.motion-command.params.motion-config.hybrid-stage2-enabled=False
  --command.setup-terms.motion-command.params.motion-config.pure-rl-policy-command-after-lift-enabled=False
  --command.setup-terms.motion-command.params.motion-config.hybrid-velocity-task-env-fraction-start="${TASK_FRACTION_START}"
  --command.setup-terms.motion-command.params.motion-config.hybrid-velocity-task-env-fraction-end="${TASK_FRACTION_END}"
  --command.setup-terms.motion-command.params.motion-config.hybrid-velocity-task-env-fraction-start-iter="${TASK_FRACTION_START_ITER}"
  --command.setup-terms.motion-command.params.motion-config.hybrid-velocity-task-env-fraction-end-iter="${TASK_FRACTION_END_ITER}"
  --command.setup-terms.motion-command.params.motion-config.hybrid-velocity-forward-command-mps="${FORWARD_COMMAND_MPS}"
  --command.setup-terms.motion-command.params.motion-config.hybrid-velocity-lift-height-m="${LIFT_HEIGHT_M}"
  --command.setup-terms.motion-command.params.motion-config.clip-weighting-strategy=uniform_clip
  --command.setup-terms.motion-command.params.motion-config.pair-terrain-with-motion=False
  --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler=True
  --command.setup-terms.motion-command.params.motion-config.adaptive-sampling-contact-interval-root="${CONTACT_EXPORT_ROOT}/clips"
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=0.2
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end=0.2
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-start-iter=0
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end-iter="${RESET_CURRICULUM_END_ITER}"
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob=0.0
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end=0.0
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-start-iter=0
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end-iter="${RESET_CURRICULUM_END_ITER}"
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale=1.0
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-prepend=True
  --command.setup-terms.motion-command.params.motion-config.default-pose-prepend-duration-s=0.2
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-append=True
  --command.setup-terms.motion-command.params.motion-config.default-pose-append-duration-s=2.0
  --command.setup-terms.motion-command.params.motion-config.contact-interval-runtime-prepend-compensation=True
  --command.setup-terms.motion-command.params.motion-config.contact-aware-button-window-mode=contact_interval
  --command.setup-terms.motion-command.params.motion-config.contact-aware-carry-window-mode=peak_height
  --command.setup-terms.motion-command.params.motion-config.contact-aware-peak-height-alpha=0.91
  --command.setup-terms.motion-command.params.motion-config.contact-aware-peak-height-smoothing-steps=5
  --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-sampling-enabled=False
  --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-half-width-steps=0
  --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-density-boost=1.0
  --reward.terms.offline-contact-guidance.params.contact-export-root="${CONTACT_EXPORT_ROOT}"
  --reward.terms.offline-contact-guidance.weight=0.0
  --randomization.setup_terms.push_randomizer_state.params.push_interval_s='[0.5,2.0]'
  --randomization.setup_terms.push_randomizer_state.params.max_push_vel='[0.7,0.7,0.25,0.7,0.7,1.0]'
  --perception.camera-apply-sensor-noise=True
  --perception.camera-warp-edge-noise=True
  --perception.camera-warp-enable-holes=True
  --perception.camera-warp-hole-prob=0.2
  --perception.camera-warp-additive-noise-std=0.03
  --perception.camera-warp-depth-offset-std=0.03
  --perception.object-geometry-mode=mesh
  --robot.object.enabled=True
  --robot.object.object-urdf-path="${OBJECT_SPEC_PATH}"
  --simulator.config.sim.max-episode-length-s=8.0
  --simulator.config.sim.physx.gpu-found-lost-pairs-capacity=268435456
  --simulator.config.sim.physx.gpu-found-lost-aggregate-pairs-capacity=268435456
  --simulator.config.sim.physx.gpu-total-aggregate-pairs-capacity=67108864
  --simulator.config.sim.physx.gpu-collision-stack-size=268435456
  --logger.video.enabled=False
  --logger.headless-recording=False
  --logger.video.upload-to-wandb=False
)

if [[ "${LOGGER}" == logger:wandb ]]; then
  [[ -n "${WANDB_RUN_ID}" ]] || { echo "[ERROR] WANDB_RUN_ID is required for formal W&B launch." >&2; exit 2; }
  TRAIN_ARGS+=(
    --logger.entity="${WANDB_ENTITY}"
    --logger.id="${WANDB_RUN_ID}"
    --logger.resume=must
    --logger.name="${TRAINING_NAME}"
    --logger.base-dir="${LOGGER_BASE_DIR}"
  )
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/validate_train_cli.py" \
  --expected-motion-end-mode episodic \
  -- "${TRAIN_ARGS[@]}"

echo "[INFO] hybrid_velocity_preflight_ok velocity_units=mps,mps,radps task_fraction=${TASK_FRACTION_START}->${TASK_FRACTION_END} task_fraction_iters=${TASK_FRACTION_START_ITER}->${TASK_FRACTION_END_ITER} command_pre=0,0,0 command_post=${FORWARD_COMMAND_MPS},0,0 task_mode_changes_only_on_reset=true critic_reference_masked_on_task=true distill=false fresh_policy=true envs_per_rank=${PER_GPU_ENVS}"

if [[ "${DRY_RUN:-0}" == 1 || "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  exit 0
fi

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node=8 \
  --max_restarts=0 \
  --master_port="${MASTER_PORT}" \
  src/holosoma/holosoma/train_agent_rank_visible.py \
  "${TRAIN_ARGS[@]}"
