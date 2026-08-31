#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "${REPO_ROOT}"

if rg -n '_object_bank_wandb\.env' distill_box_button.sh distill_box_perception.sh; then
  echo '[FAIL] launcher metadata must never be written inside an immutable motion bank' >&2
  exit 1
fi

if rg -n 'torch\.load\(' \
  batch_ne.sh distill_as_button.sh distill_as_perception.sh \
  distill_box_button.sh distill_box_perception.sh; then
  echo '[FAIL] training launchers must route checkpoint inspection through the stable weights-only loader' >&2
  exit 1
fi
for checkpoint_launcher in \
  batch_ne.sh distill_as_button.sh distill_as_perception.sh \
  distill_box_button.sh distill_box_perception.sh; do
  if ! grep -F 'load_verified_torch_checkpoint' "${checkpoint_launcher}" >/dev/null; then
    echo "[FAIL] ${checkpoint_launcher} must use the verified checkpoint loader for inline metadata reads" >&2
    exit 1
  fi
done

TMP_DIR=$(mktemp -d)
FIXTURE_PROCESS_SPECS=()
LEGACY_FIXTURE_ROOT_PIDS=()
LEGACY_FIXTURE_UNITS=()
FIXTURE_GATE_MAX_POLLS=4000
cleanup_test() {
  local spec pid token unit
  for spec in "${FIXTURE_PROCESS_SPECS[@]}"; do
    pid=${spec%%:*}
    token=${spec#*:}
    if [[ -r "/proc/${pid}/environ" ]] \
        && tr '\0' '\n' <"/proc/${pid}/environ" 2>/dev/null \
          | grep -Fx "HOLOSOMA_LAUNCH_TOKEN=${token}" >/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  for unit in "${LEGACY_FIXTURE_UNITS[@]}"; do
    if declare -F cleanup_legacy_fixture_unit >/dev/null; then
      cleanup_legacy_fixture_unit "${unit}"
    else
      systemctl --user stop "${unit}" >/dev/null 2>&1 || true
    fi
  done
  for pid in "${LEGACY_FIXTURE_ROOT_PIDS[@]}"; do
    if declare -F terminate_legacy_fixture_tree >/dev/null; then
      terminate_legacy_fixture_tree "${pid}"
    else
      kill -KILL "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  if [[ "${KEEP_TEST_TMP:-0}" == 1 ]]; then
    echo "[INFO] preserved launcher-contract TMP_DIR=${TMP_DIR}" >&2
  else
    chmod -R u+w "${TMP_DIR}" 2>/dev/null || true
    rm -rf "${TMP_DIR}"
  fi
}
trap cleanup_test EXIT

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

# Most launcher fixtures exercise lifecycle machinery without constructing a
# real scientific runtime archive.  Opt those fixtures into the explicit
# legacy compatibility mode; dedicated assertions below verify that production
# invocations now default to the fail-closed overlay requirement.
export HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY=0

expect_failure() {
  local output_file="$1"
  local expected="$2"
  shift 2
  if "$@" >"${output_file}" 2>&1; then
    fail "command unexpectedly succeeded: $*"
  fi
  grep -F "${expected}" "${output_file}" >/dev/null || {
    sed -n '1,40p' "${output_file}" >&2
    fail "missing expected failure text: ${expected}"
  }
}

bash -n batch_ne.sh batch_track.sh train_as_general.sh train_object_generalist_ds.sh \
  cp*.sh distill*.sh scripts/build_run_snapshot.sh scripts/build_numpy_runtime_overlay.sh \
  scripts/reset_curriculum_contract.sh

bash tests/launcher/test_reset_curriculum_contract.sh
bash tests/launcher/test_ppo_lr_controller_contract.sh
bash tests/launcher/test_legacy_stop_uid_contract.sh
bash tests/launcher/test_legacy_stop_real_tmux_contract.sh
bash tests/launcher/test_lifecycle_state_security_contract.sh
bash tests/launcher/test_mujoco_perception_metadata_contracts.sh
bash tests/launcher/test_numpy_runtime_overlay_contract.sh
bash tests/launcher/test_dual_button_entrypoint_contract.sh

if rg -n '^[[:space:]]*python3([[:space:]]|$)|torchrun_args=\(torchrun' \
  train_as_general.sh train_object_generalist_ds.sh >/dev/null; then
  fail 'AS generalist launchers must use the verified PYTHON_BIN interpreter'
fi
grep -F 'GLOBAL_WORLD_SIZE=$((NPROC * NNODES))' train_object_generalist_ds.sh >/dev/null ||
  fail 'object-generalist launcher must define a global multi-node world size'
grep -F 'NUM_ENVS=${NUM_ENVS:-$((GLOBAL_WORLD_SIZE * PER_GPU_ENVS))}' \
  train_object_generalist_ds.sh >/dev/null ||
  fail 'object-generalist default NUM_ENVS must cover all nodes and GPUs'
grep -F 'Multi-node launch requires explicit shared MASTER_ADDR and MASTER_PORT.' \
  train_object_generalist_ds.sh >/dev/null ||
  fail 'object-generalist multi-node launch must reject implicit rendezvous defaults'
grep -F 'Multi-node launch requires explicit shared MASTER_ADDR and MASTER_PORT on every node.' \
  distill_torso_box.sh >/dev/null ||
  fail 'base distill multi-node launch must reject per-node rendezvous defaults'
grep -F 'AS_RANK_LOCAL_SHARDS must be a boolean.' distill_as_perception.sh >/dev/null ||
  fail 'AS rank-local sharding control must reject typo values instead of silently disabling sharding'
grep -F 'NPROC=${NPROC} exceeds CUDA_VISIBLE_DEVICES count=${AS_VISIBLE_DEVICE_COUNT}' \
  distill_as_perception.sh >/dev/null ||
  fail 'AS shard preparation must reject impossible local GPU topology before allocating shards'
grep -F 'scripts/resolve_exact_checkpoint.py' distill_torso_box.sh >/dev/null ||
  fail 'base distill launcher must resolve teachers through the shared verified checkpoint path'
if rg -n 'run\.file\(file_name\)\.download|replace=True' distill_torso_box.sh >/dev/null; then
  fail 'base distill launcher must not use an unchecked mutable W&B download path'
fi
grep -F 'Snapshot regular-file set/content does not exactly match its signed manifest:' \
  batch_ne.sh >/dev/null ||
  fail 'installed source snapshots must reject executable files omitted from the signed manifest'
grep -F 'Installed snapshot top-level directory closure changed after prepare.' \
  batch_ne.sh >/dev/null ||
  fail 'installed source snapshots must reject injected top-level package directories'
grep -F 'Installed snapshot source mode closure changed after prepare.' \
  batch_ne.sh >/dev/null ||
  fail 'installed source snapshots must reject permission-bit drift after prepare'
grep -F -- '--no-same-owner --same-permissions' batch_ne.sh >/dev/null ||
  fail 'snapshot extraction must preserve signed modes independently of node umask'
grep -F 'ip link show $(quote "${GLOO_SOCKET_IFNAME}")' batch_ne.sh >/dev/null ||
  fail 'batch node health must validate the configured Gloo interface as well as NCCL'
for rank_alias in \
  HOLOSOMA_ORIGINAL_LOCAL_RANK \
  HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE \
  HOLOSOMA_ORIGINAL_CUDA_VISIBLE_DEVICES; do
  grep -F "unset ${rank_alias}" batch_ne.sh >/dev/null ||
    fail "batch launch must clear inherited worker-only topology alias ${rank_alias}"
  grep -F "unset ${rank_alias}" distill_torso_box.sh >/dev/null ||
    fail "base distill launch must clear inherited worker-only topology alias ${rank_alias}"
done
unset rank_alias

# Generated node controls must erase every manual resume/policy/data-source
# fallback before exporting the controller's canonical launch identity.
canonical_contract_line=$(grep -n '^export HOLOSOMA_SOURCE_SNAPSHOT_ID=' batch_ne.sh \
  | tail -1 | cut -d: -f1)
[[ "${canonical_contract_line}" =~ ^[1-9][0-9]*$ ]] ||
  fail 'could not locate the canonical node-control export boundary'
for inherited_launch_alias in \
  RESUME_TRAINING_CKPT RESUME_CHECKPOINT RESUME_WANDB_ID WANDB_RUN_ID WANDB_RESUME \
  POLICY_INIT_CKPT POLICY_INIT_CHECKPOINT POLICY_INIT_SOURCE_REF \
  BOX_POLICY_INIT_EXPECTED_SHA256 HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET \
  BOX_POLICY_INIT_CONTROL_CACHE_ROOT BOX_POLICY_INIT_EXPECTED_WORLD_SIZE \
  BOX_POLICY_INIT_EXPECTED_WANDB_RUN_PATH BOX_POLICY_INIT_EXPECTED_SOURCE_SNAPSHOT_ID \
  BOX_POLICY_INIT_REQUIRE_FRESH_SOURCE \
  BOX_RESUME_CKPT RESUME_FROM_BOX_CKPT DEFAULT_BOX_RESUME_RUN \
  DEFAULT_BOX_RESUME_MODEL_FILE BOX_RESUME_MODEL_FILE DEFAULT_BOX_RESUME_CHECKPOINT \
  RESUME_FROM_PREVIOUS PREVIOUS_RESUME_CKPT RESUME_FROM_PREVIOUS_CKPT \
  PREVIOUS_RESUME_RUN PREVIOUS_RESUME_MODEL_FILE DEFAULT_PREVIOUS_RESUME_RUN \
  PREVIOUS_POLICY_INIT_CACHE_ROOT AS_POLICY_INIT_PROFILE POLICY_INIT_EXPECTED_SHA256 \
  POLICY_INIT_CACHE_ROOT RESUME_MODEL_FILE WANDB_MODEL_FILE RESUME_STEP WANDB_NAME \
  WANDB_MODE WANDB_DISABLED WANDB_BASE_URL WANDB_DIR WANDB_INIT_TIMEOUT WANDB_CONSOLE \
  WANDB_RUN_GROUP WANDB_TAGS WANDB_SWEEP_ID HOLOSOMA_REQUIRE_WANDB_RUN LOGGER \
  AS_SUCCESS133_BANK_NAME OMOMO_DATA_DIR OMOMO_OBJECT_MAP AS_CONTACT_EXPORT_ROOT \
  AS_RESUME_DATA_DIR AS_RESUME_OBJECT_MAP AS_TRAINING_RESUME_REF \
  AS_TRAINING_RESUME_CACHE_ROOT DEFAULT_RESUME_FROM_BOX_AS_BANK \
  RESUME_FROM_BOX_AS_DATA_DIR RESUME_FROM_BOX_AS_OBJECT_MAP \
  RESUME_FROM_BOX_CONTACT_EXPORT_ROOT; do
  alias_unset_line=$(grep -nFx "unset ${inherited_launch_alias}" batch_ne.sh \
    | tail -1 | cut -d: -f1)
  [[ "${alias_unset_line}" =~ ^[1-9][0-9]*$ \
        && "${alias_unset_line}" -lt "${canonical_contract_line}" ]] ||
    fail "node control does not clear ${inherited_launch_alias} before canonical exports"
done
unset inherited_launch_alias alias_unset_line canonical_contract_line
canonical_contract_line=$(grep -n '^export HOLOSOMA_SOURCE_SNAPSHOT_ID=' batch_ne.sh \
  | tail -1 | cut -d: -f1)
for ambient_semantic in \
  HOLOSOMA_EVAL_DISABLE_ROLLOUT_REFERENCE_REWARDS \
  HOLOSOMA_EVAL_ALLOW_MISSING_CAMERA_SENSOR_NOISE_STATE \
  HOLOSOMA_EVAL_DEBUG_PATH HOLOSOMA_EVAL_DEBUG_LIMIT \
  HOLOSOMA_ALLOW_LEGACY_OBJECT_URDF_FALLBACK \
  HOLOSOMA_ALLOW_LEGACY_ROLLOUT_RESTART_RESUME \
  HOLOSOMA_ALLOW_LEGACY_UNPROVENANCED_RESUME \
  HOLOSOMA_ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD \
  HOLOSOMA_ALLOW_LEGACY_UNVERIFIED_TEACHER_LOAD \
  HOLOSOMA_CLEAN_ROBOT_USD_CACHE \
  HOLOSOMA_DISABLE_HETEROGENEOUS_OBJECT_SINGLE_SLOT \
  HOLOSOMA_FORCE_HETEROGENEOUS_OBJECT_SINGLE_SLOT \
  HOLOSOMA_PROVENANCE_TIMEOUT_SEC HOLOSOMA_DEBUG_ACTOR_ALL \
  HOLOSOMA_DEBUG_GRAD_REDUCE HOLOSOMA_DEBUG_STATE_SYNC \
  HOLOSOMA_DEBUG_TRAINING_PHASES HOLOSOMA_DEBUG_TRAINING_PHASE_DIR \
  HOLOSOMA_FORCE_RICH_LIVE_LOGGING HOLOSOMA_STEP_TIMING \
  HOLOSOMA_STEP_TIMING_PROFILE HOLOSOMA_STEP_TIMING_SYNC_CUDA \
  HOLOSOMA_STEP_TIMING_INTERVAL HOLOSOMA_CAMERA_LOG_ROOT_BACK_EVERY \
  HOLOSOMA_CAMERA_WARN_ROOT_BACK_RATIO \
  HOLOSOMA_CAMERA_AUTOFIX_BACKWARD HOLOSOMA_CAMERA_BACKWARD_RATIO_THRESHOLD \
  HOLOSOMA_CAMERA_DISABLE_OFFSETS HOLOSOMA_CAMERA_EXTRA_YAW_DEG \
  HOLOSOMA_CAMERA_RANDOMIZE_PLACEMENT HOLOSOMA_CAMERA_STRICT_WARP \
  HOLOSOMA_DEFM_FORWARD_BATCH_SIZE HOLOSOMA_DEFAULT_POSE_INIT \
  HOLOSOMA_DISABLE_ACTIVE_OBS_GROUP_FILTER HOLOSOMA_DISABLE_AUTO_RESET \
  HOLOSOMA_DISABLE_BAD_TRACKING_RESET HOLOSOMA_DISABLE_CLIP_END_RESET \
  HOLOSOMA_DISABLE_MOTION_END_RESET HOLOSOMA_DISABLE_ONLINE_CONTACT_PRIOR \
  HOLOSOMA_FAR_TRACKING_DISABLE_COMBINED_DEPTH_MESHES \
  HOLOSOMA_FORCE_ROUND_ROBIN_CLIP_ASSIGNMENT HOLOSOMA_FORCE_ROUND_ROBIN_CLIP_START \
  HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE HOLOSOMA_MUJOCO_RESET_NOISE \
  HOLOSOMA_ONLINE_CONTACT_PRIOR HOLOSOMA_PERCEPTION_INCLUDE_ROBOT_MESH \
  HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES \
  HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE HOLOSOMA_PERCEPTION_SENSOR_OFFSET_DELTA \
  HOLOSOMA_PERCEPTION_SENSOR_OFFSET_OVERRIDE HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS \
  HOLOSOMA_RESET_TO_DEFAULT_POSE HOLOSOMA_STRICT_PERCEPTION_CAMERA_SOURCE \
  HOLOSOMA_STRICT_PERCEPTION_OBJECT_MESHES ISAAC_SCANDOTS_INCLUDE_MISSES \
  ISAAC_SCANDOTS_USE_DEPTH_MASK HOLOSOMA_SYNC_EACH_ITERATION \
  AS_CONTACT_AWARE AS_CONTACT_AWARE_HISTORY CONTACT_AWARE CONTACT_AWARE_HISTORY \
  CONTACT_AWARE_HISTORY_LENGTH CONTACT_AWARE_CARRY_WINDOW_MODE \
  CONTACT_AWARE_PEAK_HEIGHT_ALPHA CONTACT_AWARE_PEAK_HEIGHT_SMOOTHING_STEPS \
  CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE CONTACT_AWARE_SPARSE_ROOT_SEGMENT_STEPS \
  CONTACT_AWARE_SPARSE_ROOT_ZERO_YAW_THRESHOLD_DEG \
  ZERO_ROOT_COMMAND_WHEN_DROP_ACTIVE \
  CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION STUDENT_ACTOR_INPUTS \
  STUDENT_PROPRIO_HISTORY_LENGTH STUDENT_ACTION_HISTORY_LENGTH \
  CRITIC_PROPRIO_HISTORY_LENGTH TEACHER_ACTOR_OBS_HISTORY_LENGTH \
  FORCE_EIGHT_GPU_CONFIG \
  CORL_128 DATA_MODE EXP ROOT_COMMAND_MODE SCHEDULE_VARIANT \
  TEACHER_COMPAT_PROFILE TEACHER_OBS_KEYS TEACHER_PERCEPTION_PRESET \
  TEACHER_PERCEPTION_OBS_KEY TRACKER_PROFILE PERCEPTION_PRESET \
  CRITIC_PERCEPTION_PRESET CRITIC_PERCEPTION_OBS_KEY \
  PERCEPTION_INTO_POLICY_MODULES PERCEPTION_INTO_CRITIC_MODULES \
  DISTILL_MODE DISTILL_LOSS_TYPE DISTILL_ENABLED \
  BC_LOSS_COEF ACTOR_LR CRITIC_LR PPO_LR_SCHEDULE PPO_DESIRED_KL \
  ACTOR_MIN_LR ACTOR_MAX_LR CRITIC_MIN_LR CRITIC_MAX_LR \
  SWITCH_TO_RL_AFTER TEACHER_ACTION_MIX_RATIO \
  TEACHER_ACTION_MIX_RATIO_START TEACHER_ACTION_MIX_RATIO_END \
  TEACHER_ACTION_MIX_RATIO_END_ITERATION CLIP_TEACHER_ACTIONS \
  CLIP_ACTIONS_THRESHOLD USE_ADAPTIVE_TIMESTEPS_SAMPLER \
  START_AT_TIMESTEP_ZERO_PROB START_AT_TIMESTEP_ZERO_PROB_END \
  START_AT_TIMESTEP_ZERO_PROB_START_ITER START_AT_TIMESTEP_ZERO_PROB_END_ITER \
  FREEZE_AT_TIMESTEP_ZERO_PROB FREEZE_AT_TIMESTEP_ZERO_PROB_END \
  FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER \
  UNIFORM_T1_WINDOW_SAMPLING_ENABLED UNIFORM_T1_WINDOW_HALF_WIDTH_STEPS \
  UNIFORM_T1_WINDOW_DENSITY_BOOST PAIR_TERRAIN_WITH_MOTION \
  DAGGER_IGNORE_EPISODE_INITIAL_STEPS MAX_EPISODE_LENGTH_S RESET_TO_DEFAULT_POSE \
  ENABLE_DEFAULT_POSE_PREPEND DEFAULT_POSE_PREPEND_DURATION_S \
  ENABLE_DEFAULT_POSE_APPEND DEFAULT_POSE_APPEND_DURATION_S \
  BAD_TRACKING_THRESHOLD_AUGMENT BAD_TRACKING_THRESHOLD_MULTIPLIER \
  BAD_TRACKING_THRESHOLD_SCALE BAD_TRACKING_REF_POS_THRESHOLD_OVERRIDE \
  BAD_TRACKING_REF_ORI_THRESHOLD_OVERRIDE BAD_TRACKING_BODY_POS_THRESHOLD_OVERRIDE \
  BAD_TRACKING_OBJECT_POS_THRESHOLD_OVERRIDE BAD_TRACKING_OBJECT_ORI_THRESHOLD_OVERRIDE \
  IMAGE_WIDTH IMAGE_HEIGHT CAMERA_PITCH_DEG \
  CAMERA_FAR CAMERA_MAX_DISTANCE CAMERA_APPLY_SENSOR_NOISE CAMERA_WARP_FREQ_RATIO \
  CAMERA_WARP_EDGE_NOISE CAMERA_WARP_ENABLE_HOLES CAMERA_WARP_HOLE_PROB \
  CAMERA_WARP_ADDITIVE_NOISE_STD CAMERA_WARP_DEPTH_OFFSET_STD \
  PERCEPTION_WARP_PREPROCESS AS_PUSH_INTERVAL_S AS_MAX_PUSH_VEL \
  HOLOSOMA_OBJECT_COLLIDER_TYPE HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS \
  HOLOSOMA_REQUIRE_OBJECT_MESH_ASSETS HOLOSOMA_DEBUG_TILE_LAYOUT \
  HOLOSOMA_W_OBJECT_URDF HOLOSOMA_ISAACSIM_KIT_ARGS ISAACSIM_KIT_ARGS \
  CHECK_ONLY DRY_RUN; do
  ambient_unset_line=$(grep -nFx "unset ${ambient_semantic}" batch_ne.sh \
    | tail -1 | cut -d: -f1)
  [[ "${ambient_unset_line}" =~ ^[1-9][0-9]*$ \
        && "${ambient_unset_line}" -lt "${canonical_contract_line}" ]] ||
    fail "node control does not clear ambient scientific override ${ambient_semantic}"
done
unset ambient_semantic ambient_unset_line canonical_contract_line
for canonical_wandb_export in \
  'export LOGGER=logger:wandb' \
  'export WANDB_BASE_URL=$(quote "${WANDB_BASE_URL}")' \
  'export WANDB_INIT_TIMEOUT=$(quote "${WANDB_INIT_TIMEOUT}")' \
  'export WANDB_CONSOLE=off' \
  'export HOLOSOMA_REQUIRE_WANDB_RUN=$(quote "${HOLOSOMA_REQUIRE_WANDB_RUN}")'; do
  grep -F "${canonical_wandb_export}" batch_ne.sh >/dev/null ||
    fail "batch launch is missing canonical required W&B export: ${canonical_wandb_export}"
done
unset canonical_wandb_export
for canonical_ppo_lr_export in \
  'export ACTOR_LR=$(quote "${ACTOR_LR}")' \
  'export CRITIC_LR=$(quote "${CRITIC_LR}")' \
  'export PPO_LR_SCHEDULE=$(quote "${PPO_LR_SCHEDULE}")' \
  'export PPO_DESIRED_KL=$(quote "${PPO_DESIRED_KL}")' \
  'export ACTOR_MIN_LR=$(quote "${ACTOR_MIN_LR}")' \
  'export ACTOR_MAX_LR=$(quote "${ACTOR_MAX_LR}")' \
  'export CRITIC_MIN_LR=$(quote "${CRITIC_MIN_LR}")' \
  'export CRITIC_MAX_LR=$(quote "${CRITIC_MAX_LR}")'; do
  [[ "$(grep -Fxc -- "${canonical_ppo_lr_export}" batch_ne.sh)" -eq 1 ]] ||
    fail "batch launch must export each controller-owned PPO LR field exactly once: ${canonical_ppo_lr_export}"
done
unset canonical_ppo_lr_export
grep -F 'ppo_lr_controller schedule=\${PPO_LR_SCHEDULE} desired_kl=\${PPO_DESIRED_KL}' \
  batch_ne.sh >/dev/null ||
  fail 'batch node startup log must expose the effective PPO LR controller contract'
for wandb_sanitization_contract in \
  'for _wandb_env_name in \${!WANDB_@} \${!_WANDB_@}; do' \
  'WANDB_API_KEY|WANDB_IDENTITY_TOKEN_FILE|WANDB_CREDENTIALS_FILE)' \
  'unset "\${_wandb_env_name}"' \
  'TRAIN_EXTRA_ARGS=(--logger.entity="\${WANDB_ENTITY}")'; do
  grep -F "${wandb_sanitization_contract}" batch_ne.sh >/dev/null ||
    fail "batch launch is missing W&B ambient/explicit-identity contract: ${wandb_sanitization_contract}"
done
unset wandb_sanitization_contract
grep -F 'export BOX_RESUME_CKPT' batch_ne.sh >/dev/null ||
  fail 'verified box policy initializer must be re-exported after alias cleanup'

expect_failure \
  "${TMP_DIR}/legacy_batch_track_quarantine.out" \
  'batch_track.sh launch is quarantined' \
  env DRY_RUN=1 NODES=test-node NNODES=1 NPROC=8 PER_GPU_ENVS=4096 \
  bash batch_track.sh launch

expect_failure \
  "${TMP_DIR}/implicit_history_box_warmstart.out" \
  'requires an explicit target policy history contract' \
  env RESUME_FROM_BOX=1 BOX_RESUME_CKPT="${TMP_DIR}/box.pt" \
  bash train_as_general.sh

expect_failure \
  "${TMP_DIR}/poisoned_python_verified_marker.out" \
  'PYTHON_BIN cannot import torch:' \
  env PYTHON_BIN=/usr/bin/python3 HOLOSOMA_TORCH_PYTHON_VERIFIED=/usr/bin/python3 \
  bash -c 'source scripts/gpu_launch_defaults.sh'

# Objective-label mistakes must fail before the launcher sources Python/GPU
# helpers or resolves a teacher. A deliberately unusable interpreter proves
# that no such preflight I/O ran first.
expect_failure \
  "${TMP_DIR}/distill_mse_non_mse_loss_early.out" \
  'DISTILL_MODE=mse requires DISTILL_LOSS_TYPE=mse. Got: huber' \
  env PYTHON_BIN="${TMP_DIR}/must-not-be-executed" \
    DISTILL_MODE=mse DISTILL_LOSS_TYPE=huber bash distill_torso_box.sh
expect_failure \
  "${TMP_DIR}/distill_dagger_unknown_loss_early.out" \
  "DISTILL_LOSS_TYPE must be exactly 'mse' or 'huber' in DAgger mode. Got: l1" \
  env PYTHON_BIN="${TMP_DIR}/must-not-be-executed" \
    DISTILL_MODE=dagger DISTILL_LOSS_TYPE=l1 bash distill_torso_box.sh
expect_failure \
  "${TMP_DIR}/distill_invalid_ppo_lr_schedule_early.out" \
  'PPO_LR_SCHEDULE must be exactly adaptive or fixed. Got: Adaptive' \
  env PYTHON_BIN="${TMP_DIR}/must-not-be-executed" \
    PPO_LR_SCHEDULE=Adaptive bash distill_torso_box.sh
expect_failure \
  "${TMP_DIR}/distill_forwarded_loss_override_early.out" \
  'Do not override launcher-owned distillation field via forwarded CLI: --algo.config.distill.distill-loss-type=huber' \
  env PYTHON_BIN="${TMP_DIR}/must-not-be-executed" \
    DISTILL_MODE=mse DISTILL_LOSS_TYPE=mse bash distill_torso_box.sh \
    --algo.config.distill.distill-loss-type=huber
expect_failure \
  "${TMP_DIR}/distill_forwarded_mode_override_early.out" \
  'Do not override launcher-owned distillation field via forwarded CLI: --algo.config.distill.mode=mse' \
  env PYTHON_BIN="${TMP_DIR}/must-not-be-executed" \
    DISTILL_MODE=dagger DISTILL_LOSS_TYPE=huber bash distill_torso_box.sh \
    --algo.config.distill.mode=mse
expect_failure \
  "${TMP_DIR}/distill_forwarded_checkpoint_mode_override_early.out" \
  'Do not override launcher-owned training/reset field via forwarded CLI: --training.checkpoint=/tmp/resume.pt' \
  env PYTHON_BIN="${TMP_DIR}/must-not-be-executed" \
    bash distill_torso_box.sh --training.checkpoint=/tmp/resume.pt
expect_failure \
  "${TMP_DIR}/distill_forwarded_policy_init_mode_override_early.out" \
  'Do not override launcher-owned training/reset field via forwarded CLI: --training.policy-init-checkpoint=/tmp/init.pt' \
  env PYTHON_BIN="${TMP_DIR}/must-not-be-executed" \
    bash distill_torso_box.sh --training.policy-init-checkpoint=/tmp/init.pt
expect_failure \
  "${TMP_DIR}/distill_forwarded_ppo_lr_controller_override_early.out" \
  'Do not override launcher-owned PPO LR controller field via forwarded CLI: --algo.config.desired-kl=0.02' \
  env PYTHON_BIN="${TMP_DIR}/must-not-be-executed" \
    bash distill_torso_box.sh --algo.config.desired-kl=0.02
expect_failure \
  "${TMP_DIR}/perception_forwarded_target_override_early.out" \
  'is launcher-owned and cannot be overridden in forwarded argv/EXTRA_ARGS' \
  env PYTHON_BIN="${TMP_DIR}/must-not-be-executed" \
    bash distill_box_perception.sh --algo.config.num_learning_iterations=8
expect_failure \
  "${TMP_DIR}/button_forwarded_reset_endpoint_override_early.out" \
  'is launcher-owned and cannot be overridden in forwarded argv/EXTRA_ARGS' \
  env PYTHON_BIN="${TMP_DIR}/must-not-be-executed" \
    bash distill_box_button.sh \
    --command.setup_terms.motion_command.params.motion_config.freeze_at_timestep_zero_prob_end_iter=8
expect_failure \
  "${TMP_DIR}/perception_forwarded_policy_init_mode_override_early.out" \
  'is launcher-owned and cannot be overridden in forwarded argv/EXTRA_ARGS' \
  env PYTHON_BIN="${TMP_DIR}/must-not-be-executed" \
    bash distill_box_perception.sh --training.policy_init_checkpoint=/tmp/init.pt

touch "${TMP_DIR}/teacher.pt"
expect_failure \
  "${TMP_DIR}/distill_implicit_multinode_rendezvous.out" \
  'Multi-node launch requires explicit shared MASTER_ADDR and MASTER_PORT on every node.' \
  env TEACHER_CHECKPOINT="${TMP_DIR}/teacher.pt" NNODES=2 NODE_RANK=0 NPROC=1 \
    CUDA_VISIBLE_DEVICES=0 bash distill_torso_box.sh
expect_failure \
  "${TMP_DIR}/distill_elastic_replay.out" \
  'Scientific distillation requires MAX_RESTARTS=0' \
  env TEACHER_CHECKPOINT="${TMP_DIR}/teacher.pt" NNODES=1 NPROC=1 \
    CUDA_VISIBLE_DEVICES=0 MAX_RESTARTS=1 bash distill_torso_box.sh
expect_failure \
  "${TMP_DIR}/distill_invalid_physx_capacity.out" \
  'PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY must be a positive integer' \
  env TEACHER_CHECKPOINT="${TMP_DIR}/teacher.pt" NNODES=1 NPROC=1 \
    CUDA_VISIBLE_DEVICES=0 PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY='x[$(false)]' \
    bash distill_torso_box.sh
expect_failure \
  "${TMP_DIR}/distill_desired_kl_nan.out" \
  'PPO_DESIRED_KL must be finite. Got: nan' \
  env TEACHER_CHECKPOINT="${TMP_DIR}/teacher.pt" NNODES=1 NPROC=1 \
    CUDA_VISIBLE_DEVICES=0 PPO_DESIRED_KL=nan bash distill_torso_box.sh
expect_failure \
  "${TMP_DIR}/distill_actor_lr_lower_bound.out" \
  'ACTOR_MIN_LR must be <= ACTOR_LR; got 0.002>0.001.' \
  env TEACHER_CHECKPOINT="${TMP_DIR}/teacher.pt" NNODES=1 NPROC=1 \
    CUDA_VISIBLE_DEVICES=0 ACTOR_LR=0.001 ACTOR_MIN_LR=0.002 \
    bash distill_torso_box.sh
expect_failure \
  "${TMP_DIR}/distill_critic_lr_upper_bound.out" \
  'CRITIC_LR must be <= CRITIC_MAX_LR; got 0.001>0.0005.' \
  env TEACHER_CHECKPOINT="${TMP_DIR}/teacher.pt" NNODES=1 NPROC=1 \
    CUDA_VISIBLE_DEVICES=0 CRITIC_LR=0.001 CRITIC_MAX_LR=0.0005 \
    bash distill_torso_box.sh
expect_failure \
  "${TMP_DIR}/distill_duplicate_visible_gpu.out" \
  'CUDA_VISIBLE_DEVICES must not select the same GPU token twice' \
  env TEACHER_CHECKPOINT="${TMP_DIR}/teacher.pt" NNODES=1 NPROC=2 \
    CUDA_VISIBLE_DEVICES=0,0 bash distill_torso_box.sh
expect_failure \
  "${TMP_DIR}/distill_invalid_force_eight.out" \
  'FORCE_EIGHT_GPU_CONFIG must be a boolean' \
  env TEACHER_CHECKPOINT="${TMP_DIR}/teacher.pt" NNODES=1 NPROC=1 \
    CUDA_VISIBLE_DEVICES=0 FORCE_EIGHT_GPU_CONFIG=perhaps bash distill_torso_box.sh
expect_failure \
  "${TMP_DIR}/distill_invalid_mode.out" \
  "DISTILL_MODE must be exactly 'mse' or 'dagger'. Got: MSE" \
  env TEACHER_CHECKPOINT="${TMP_DIR}/teacher.pt" NNODES=1 NPROC=1 \
    CUDA_VISIBLE_DEVICES=0 DISTILL_MODE=MSE bash distill_torso_box.sh
expect_failure \
  "${TMP_DIR}/distill_mse_dagger_schedule.out" \
  'DISTILL_MODE=mse requires PPO_START_EPOCH=-1 and DAGGER_END_EPOCH=-1' \
  env TEACHER_CHECKPOINT="${TMP_DIR}/teacher.pt" NNODES=1 NPROC=1 \
    CUDA_VISIBLE_DEVICES=0 DISTILL_MODE=mse PPO_START_EPOCH=0 bash distill_torso_box.sh
expect_failure \
  "${TMP_DIR}/distill_mse_teacher_mix.out" \
  'DISTILL_MODE=mse requires TEACHER_ACTION_MIX_RATIO=0' \
  env TEACHER_CHECKPOINT="${TMP_DIR}/teacher.pt" NNODES=1 NPROC=1 \
    CUDA_VISIBLE_DEVICES=0 DISTILL_MODE=mse TEACHER_ACTION_MIX_RATIO=0.5 bash distill_torso_box.sh
expect_failure \
  "${TMP_DIR}/distill_mse_teacher_mix_schedule.out" \
  'DISTILL_MODE=mse does not accept TEACHER_ACTION_MIX_RATIO_START/END/END_ITERATION' \
  env TEACHER_CHECKPOINT="${TMP_DIR}/teacher.pt" NNODES=1 NPROC=1 \
    CUDA_VISIBLE_DEVICES=0 DISTILL_MODE=mse TEACHER_ACTION_MIX_RATIO_START=0.5 bash distill_torso_box.sh
expect_failure \
  "${TMP_DIR}/distill_mse_dagger_match.out" \
  'DISTILL_MODE=mse requires DAGGER_MATCH_STD=False' \
  env TEACHER_CHECKPOINT="${TMP_DIR}/teacher.pt" NNODES=1 NPROC=1 \
    CUDA_VISIBLE_DEVICES=0 DISTILL_MODE=mse DAGGER_MATCH_STD=True bash distill_torso_box.sh
expect_failure \
  "${TMP_DIR}/distill_mse_switch.out" \
  'DISTILL_MODE=mse requires SWITCH_TO_RL_AFTER to be empty, -1, or 0' \
  env TEACHER_CHECKPOINT="${TMP_DIR}/teacher.pt" NNODES=1 NPROC=1 \
    CUDA_VISIBLE_DEVICES=0 DISTILL_MODE=mse SWITCH_TO_RL_AFTER=10 bash distill_torso_box.sh
expect_failure \
  "${TMP_DIR}/distill_mse_dagger_loss.out" \
  'DISTILL_MODE=mse requires DAGGER_LOSS_COEF=0' \
  env TEACHER_CHECKPOINT="${TMP_DIR}/teacher.pt" NNODES=1 NPROC=1 \
    CUDA_VISIBLE_DEVICES=0 DISTILL_MODE=mse DAGGER_LOSS_COEF=10 bash distill_torso_box.sh
expect_failure \
  "${TMP_DIR}/distill_mse_dagger_ignore.out" \
  'DISTILL_MODE=mse requires DAGGER_IGNORE_ZERO_TEACHER_ACTIONS=False' \
  env TEACHER_CHECKPOINT="${TMP_DIR}/teacher.pt" NNODES=1 NPROC=1 \
    CUDA_VISIBLE_DEVICES=0 DISTILL_MODE=mse DAGGER_IGNORE_ZERO_TEACHER_ACTIONS=True \
    bash distill_torso_box.sh

mkdir -p "${TMP_DIR}/distill_mode_motion"
touch "${TMP_DIR}/distill_mode_object.urdf"
python - "${TMP_DIR}/distill_mode_teacher.pt" <<'PY'
from pathlib import Path
import sys

import torch

torch.save({}, Path(sys.argv[1]))
PY

DISTILL_MODE_DRY_RUN_ENV=(
  env
  DRY_RUN=1
  NPROC=1
  NNODES=1
  PER_GPU_ENVS=2
  CUDA_VISIBLE_DEVICES=0
  LOGGER=logger:disabled
  TEACHER_CHECKPOINT="${TMP_DIR}/distill_mode_teacher.pt"
  TEACHER_CACHE_ROOT="${TMP_DIR}/distill_mode_teacher_cache"
  MOTION_DIR="${TMP_DIR}/distill_mode_motion"
  OBJECT_URDF="${TMP_DIR}/distill_mode_object.urdf"
)
"${DISTILL_MODE_DRY_RUN_ENV[@]}" DISTILL_MODE=mse BC_LOSS_COEF=0.375 \
  ACTOR_LR=1e-6 CRITIC_LR=0.1 \
  bash distill_torso_box.sh >"${TMP_DIR}/distill_mode_mse.out"
"${DISTILL_MODE_DRY_RUN_ENV[@]}" DISTILL_MODE=dagger DISTILL_LOSS_TYPE=huber \
  BC_LOSS_COEF=0.625 PPO_START_EPOCH=-1 DAGGER_END_EPOCH=-1 \
  PPO_LR_SCHEDULE=fixed PPO_DESIRED_KL=0.023 \
  ACTOR_LR=0.000321 ACTOR_MIN_LR=0.000123 ACTOR_MAX_LR=0.000456 \
  CRITIC_LR=0.000654 CRITIC_MIN_LR=0.000234 CRITIC_MAX_LR=0.000765 \
  bash distill_torso_box.sh >"${TMP_DIR}/distill_mode_dagger.out"

grep -F '[INFO] distill_mode=mse loss_coef=0.375' \
  "${TMP_DIR}/distill_mode_mse.out" >/dev/null ||
  fail 'MSE dry-run must report the coefficient that the MSE objective consumes'
grep -F '[INFO] distill_mode=dagger bc_loss_coef=0.625 ' \
  "${TMP_DIR}/distill_mode_dagger.out" >/dev/null ||
  fail 'DAgger dry-run must report the coefficient that the DAgger objective consumes'
grep -F '[INFO] ppo_lr_controller schedule=fixed desired_kl=0.023 actor_lr=0.000321 actor_bounds=[0.000123,0.000456] critic_lr=0.000654 critic_bounds=[0.000234,0.000765]' \
  "${TMP_DIR}/distill_mode_dagger.out" >/dev/null ||
  fail 'DAgger dry-run must report its complete effective PPO LR controller contract'
python - "${TMP_DIR}/distill_mode_mse.out" "${TMP_DIR}/distill_mode_dagger.out" <<'PY'
from __future__ import annotations

import shlex
import sys
from pathlib import Path

prefix = "[INFO] final_train_command:"
cases = (
    (Path(sys.argv[1]), "mse", "0.375", "mse", "-1", "-1"),
    (Path(sys.argv[2]), "dagger", "0.625", "huber", "-1", "-1"),
)
for output_path, mode, coefficient, loss_type, ppo_start, dagger_end in cases:
    commands = [
        shlex.split(line[len(prefix) :])
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.startswith(prefix)
    ]
    if len(commands) != 1:
        raise SystemExit(
            f"[FAIL] {mode} dry-run expected one final_train_command, got {len(commands)}"
        )
    args = commands[0]
    mode_arg = f"--algo.config.distill.mode={mode}"
    if args.count(mode_arg) != 1:
        raise SystemExit(
            f"[FAIL] {mode} dry-run must forward its exact mode once: count={args.count(mode_arg)}"
        )
    loss_type_arg = f"--algo.config.distill.distill-loss-type={loss_type}"
    if args.count(loss_type_arg) != 1:
        raise SystemExit(
            f"[FAIL] {mode} dry-run must forward its validated loss type once: "
            f"arg={loss_type_arg!r} count={args.count(loss_type_arg)}"
        )

    if mode == "mse":
        expected_lr_controller = {
            "--algo.config.schedule=adaptive",
            "--algo.config.desired-kl=0.01",
            "--algo.config.actor-learning-rate=1e-6",
            "--algo.config.critic-learning-rate=0.1",
            "--algo.config.min-actor-learning-rate=1e-06",
            "--algo.config.max-actor-learning-rate=0.01",
            "--algo.config.min-critic-learning-rate=1e-05",
            "--algo.config.max-critic-learning-rate=0.1",
        }
    else:
        expected_lr_controller = {
            "--algo.config.schedule=fixed",
            "--algo.config.desired-kl=0.023",
            "--algo.config.actor-learning-rate=0.000321",
            "--algo.config.critic-learning-rate=0.000654",
            "--algo.config.min-actor-learning-rate=0.000123",
            "--algo.config.max-actor-learning-rate=0.000456",
            "--algo.config.min-critic-learning-rate=0.000234",
            "--algo.config.max-critic-learning-rate=0.000765",
        }
    controller_prefixes = {
        expected.split("=", 1)[0].replace("_", "-")
        for expected in expected_lr_controller
    }
    actual_lr_controller = [
        arg
        for arg in args
        if arg.split("=", 1)[0].replace("_", "-") in controller_prefixes
    ]
    if set(actual_lr_controller) != expected_lr_controller or len(actual_lr_controller) != len(
        expected_lr_controller
    ):
        raise SystemExit(
            f"[FAIL] {mode} final CLI must expose every PPO LR controller field exactly once: "
            f"actual={actual_lr_controller!r} expected={sorted(expected_lr_controller)!r}"
        )

    mse_coef_args = [
        arg
        for arg in args
        if arg.startswith("--algo.config.distill.loss-coef=")
        or arg.startswith("--algo.config.distill.loss_coef=")
    ]
    dagger_coef_args = [
        arg
        for arg in args
        if arg.startswith("--algo.config.distill.bc-loss-coef=")
        or arg.startswith("--algo.config.distill.bc_loss_coef=")
    ]
    if mode == "mse":
        expected = [f"--algo.config.distill.loss-coef={coefficient}"]
        if mse_coef_args != expected or dagger_coef_args:
            raise SystemExit(
                "[FAIL] MSE must forward only its effective loss_coef: "
                f"loss={mse_coef_args!r} bc={dagger_coef_args!r}"
            )
        dagger_only_fields = (
            "teacher-action-mix-ratio",
            "teacher-action-mix-ratio-start",
            "teacher-action-mix-ratio-end",
            "teacher-action-mix-ratio-end-iteration",
            "ppo-start-epoch",
            "dagger-end-epoch",
            "dagger-loss-coef",
            "dagger-match-std",
            "dagger-ignore-zero-teacher-actions",
            "switch-to-rl-after",
        )
        leaked = [
            arg
            for arg in args
            if any(
                arg.replace("_", "-").startswith(
                    f"--algo.config.distill.{field}="
                )
                for field in dagger_only_fields
            )
        ]
        if leaked:
            raise SystemExit(
                f"[FAIL] MSE final CLI leaked DAgger-only options: {leaked!r}"
            )
    else:
        expected = [f"--algo.config.distill.bc-loss-coef={coefficient}"]
        if dagger_coef_args != expected or mse_coef_args:
            raise SystemExit(
                "[FAIL] DAgger must forward only its effective bc_loss_coef: "
                f"bc={dagger_coef_args!r} loss={mse_coef_args!r}"
            )
        expected_dagger_options = {
            "--algo.config.distill.teacher-action-mix-ratio=0.0",
            f"--algo.config.distill.ppo-start-epoch={ppo_start}",
            f"--algo.config.distill.dagger-end-epoch={dagger_end}",
            "--algo.config.distill.dagger-loss-coef=10.0",
            "--algo.config.distill.dagger-match-std=False",
            "--algo.config.distill.dagger-ignore-zero-teacher-actions=True",
        }
        missing = sorted(expected_dagger_options.difference(args))
        if missing:
            raise SystemExit(
                f"[FAIL] DAgger defaults changed while splitting mode-specific CLI: {missing!r}"
            )
PY
unset DISTILL_MODE_DRY_RUN_ENV

BATCH_ENV=(
  env
  NODES=test-node
  NPROC=1
  NNODES=1
  PER_GPU_ENVS=1024
  DRY_RUN=1
  SKIP_GIT_PULL=1
  SOURCE_SNAPSHOT_CACHE="${TMP_DIR}/snapshot-cache"
)
NCCL_LIB_SHA_SENTINEL='0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef'
HUGE_UNSIGNED_DECIMAL=18446744073709551617

# Emergency control actions must remain available even when a stale training
# shell exports a malformed runtime contract.  Those values are neither
# trusted nor evaluated by status/stop.
for control_action in status stop; do
  "${BATCH_ENV[@]}" \
    HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY=invalid \
    PYTHON_RUNTIME_SITEPACKAGES=malformed \
    PYTHON_RUNTIME_MANIFEST_SHA256=bad \
    PYTHON_RUNTIME_ARCHIVE=/bad \
    PYTHON_RUNTIME_ARCHIVE_SHA256=bad \
    bash batch_ne.sh "${control_action}" \
    >"${TMP_DIR}/batch_poisoned_runtime_${control_action}.out"
  if grep -E 'HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY|PYTHON_RUNTIME_.*must|runtime overlay.*required' \
      "${TMP_DIR}/batch_poisoned_runtime_${control_action}.out" >/dev/null; then
    fail "${control_action} evaluated a stale scientific runtime contract"
  fi
done
unset control_action
if rg -n '(^|[[:space:]])python3[[:space:]]+-' batch_ne.sh >/dev/null; then
  fail 'batch controller Python helpers must use the resolved absolute PYTHON_BIN, not ambient python3'
fi

expect_failure \
  "${TMP_DIR}/batch_extra_positional_arg.out" \
  'accepts exactly one action and no additional positional arguments' \
  "${BATCH_ENV[@]}" bash batch_ne.sh launch --dry-run
expect_failure \
  "${TMP_DIR}/batch_default_runtime_overlay_required.out" \
  'This scientific launch requires PYTHON_RUNTIME_SITEPACKAGES and PYTHON_RUNTIME_MANIFEST_SHA256' \
  env -u HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY \
  "${BATCH_ENV[@]}" bash batch_ne.sh launch
if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' \
    "${TMP_DIR}/batch_default_runtime_overlay_required.out" >/dev/null; then
  fail 'default runtime-overlay requirement must fail before snapshot construction or remote actions'
fi
expect_failure \
  "${TMP_DIR}/batch_invalid_runtime_overlay_requirement.out" \
  'HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY must be exactly 0 or 1' \
  "${BATCH_ENV[@]}" HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY=invalid \
  bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_dry_run.out" \
  'DRY_RUN must be a boolean. Got: typo' \
  "${BATCH_ENV[@]}" DRY_RUN=typo bash batch_ne.sh launch
remote_root_injection_marker="${TMP_DIR}/remote-root-injection-marker"
expect_failure \
  "${TMP_DIR}/batch_unsafe_remote_run_root.out" \
  'REMOTE_RUN_ROOT must be a canonical safe non-root absolute path' \
  "${BATCH_ENV[@]}" \
  REMOTE_RUN_ROOT="/tmp/unsafe;\$(touch\${IFS}${remote_root_injection_marker})" \
  bash batch_ne.sh launch
[[ ! -e "${remote_root_injection_marker}" ]] ||
  fail 'unsafe REMOTE_RUN_ROOT reached remote/local shell evaluation'
unset remote_root_injection_marker
expect_failure \
  "${TMP_DIR}/batch_non_cloud_wandb_url.out" \
  'WANDB_BASE_URL must be exactly https://api.wandb.ai' \
  "${BATCH_ENV[@]}" WANDB_BASE_URL=https://wandb.internal.example bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_wandb_timeout.out" \
  'WANDB_INIT_TIMEOUT must be a canonical integer in [1, 3600]' \
  "${BATCH_ENV[@]}" WANDB_INIT_TIMEOUT=0 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_excessive_wandb_timeout.out" \
  'WANDB_INIT_TIMEOUT must be a canonical integer in [1, 3600]' \
  "${BATCH_ENV[@]}" WANDB_INIT_TIMEOUT=3601 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_require_wandb.out" \
  'HOLOSOMA_REQUIRE_WANDB_RUN must be a boolean' \
  "${BATCH_ENV[@]}" HOLOSOMA_REQUIRE_WANDB_RUN=invalid bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_disabled_required_wandb.out" \
  'Scientific batch launch requires HOLOSOMA_REQUIRE_WANDB_RUN=1 exactly' \
  "${BATCH_ENV[@]}" HOLOSOMA_REQUIRE_WANDB_RUN=0 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_wandb_entity.out" \
  'WANDB_ENTITY must be one canonical W&B URL-path segment' \
  "${BATCH_ENV[@]}" WANDB_ENTITY='wrong/entity' bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_wandb_resume_id.out" \
  'RESUME_WANDB_RUN_ID must be one canonical W&B URL-path segment' \
  "${BATCH_ENV[@]}" RESUME_WANDB_RUN_ID='wrong/run' bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_non_must_wandb_resume.out" \
  'Scientific same-run W&B resume requires WANDB_RESUME_MODE=must exactly' \
  "${BATCH_ENV[@]}" RESUME_WANDB_RUN_ID=run123 WANDB_RESUME_MODE=allow \
  bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_contradictory_wandb_resume.out" \
  'RESUME_WANDB_RUN_ID requires WANDB_RESUME_SAME_RUN=1' \
  "${BATCH_ENV[@]}" RESUME_WANDB_RUN_ID=run123 WANDB_RESUME_SAME_RUN=0 \
  bash batch_ne.sh launch

expect_failure \
  "${TMP_DIR}/batch_huge_target.out" \
  'TARGET_LEARNING_ITERATION must be a canonical integer in [1, 2147483647]' \
  "${BATCH_ENV[@]}" TARGET_LEARNING_ITERATION="${HUGE_UNSIGNED_DECIMAL}" \
  bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_huge_nproc.out" \
  'NPROC must be a canonical integer in [1, 1024]' \
  "${BATCH_ENV[@]}" NPROC="${HUGE_UNSIGNED_DECIMAL}" bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_huge_physx.out" \
  'PHYSX_GPU_COLLISION_STACK_SIZE must be a canonical integer in [1, 2147483647]' \
  "${BATCH_ENV[@]}" PHYSX_GPU_COLLISION_STACK_SIZE="${HUGE_UNSIGNED_DECIMAL}" \
  bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_equal_schedule_boundaries.out" \
  'PPO_START_EPOCH must be < DAGGER_END_EPOCH; got 700>=700' \
  "${BATCH_ENV[@]}" PPO_START_EPOCH=700 DAGGER_END_EPOCH=700 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_excessive_flow_steps.out" \
  'STUDENT_FLOW_STEPS must be a canonical positive integer in [1, 4096]' \
  "${BATCH_ENV[@]}" STUDENT_FLOW_STEPS=4097 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_excessive_flow_train_noise.out" \
  'STUDENT_FLOW_TRAIN_NOISE_STD must be finite and [0.0, 1e+18]' \
  "${BATCH_ENV[@]}" STUDENT_FLOW_TRAIN_NOISE_STD=1e19 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_excessive_flow_inference_noise.out" \
  'STUDENT_FLOW_INFERENCE_NOISE_STD must be finite and [0.0, 1e+18]' \
  "${BATCH_ENV[@]}" STUDENT_FLOW_INFERENCE_NOISE_STD=1e19 bash batch_ne.sh launch
for flow_preflight_output in \
  "${TMP_DIR}/batch_excessive_flow_steps.out" \
  "${TMP_DIR}/batch_excessive_flow_train_noise.out" \
  "${TMP_DIR}/batch_excessive_flow_inference_noise.out"; do
  if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' "${flow_preflight_output}" >/dev/null; then
    fail 'unsafe Flow numerics must fail before snapshot construction or remote actions'
  fi
done
unset flow_preflight_output

expect_failure \
  "${TMP_DIR}/batch_actor_lr_nan.out" \
  'ACTOR_LR must be finite' \
  "${BATCH_ENV[@]}" ACTOR_LR=nan bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_critic_lr_negative.out" \
  'CRITIC_LR must be finite and > 0' \
  "${BATCH_ENV[@]}" CRITIC_LR=-1 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_ppo_lr_schedule.out" \
  'PPO_LR_SCHEDULE must be exactly adaptive or fixed. Got: Adaptive' \
  "${BATCH_ENV[@]}" PPO_LR_SCHEDULE=Adaptive bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_desired_kl_nan.out" \
  'PPO_DESIRED_KL must be finite. Got: nan' \
  "${BATCH_ENV[@]}" PPO_DESIRED_KL=nan bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_desired_kl_zero.out" \
  'PPO_DESIRED_KL must be finite and > 0. Got: 0' \
  "${BATCH_ENV[@]}" PPO_DESIRED_KL=0 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_actor_lr_lower_bound.out" \
  'ACTOR_MIN_LR must be <= ACTOR_LR; got 0.002>0.001.' \
  "${BATCH_ENV[@]}" ACTOR_LR=0.001 ACTOR_MIN_LR=0.002 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_actor_lr_upper_bound.out" \
  'ACTOR_LR must be <= ACTOR_MAX_LR; got 0.001>0.0005.' \
  "${BATCH_ENV[@]}" ACTOR_LR=0.001 ACTOR_MAX_LR=0.0005 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_critic_lr_lower_bound.out" \
  'CRITIC_MIN_LR must be <= CRITIC_LR; got 0.002>0.001.' \
  "${BATCH_ENV[@]}" CRITIC_LR=0.001 CRITIC_MIN_LR=0.002 bash batch_ne.sh launch
for ppo_lr_preflight_output in \
  "${TMP_DIR}/batch_invalid_ppo_lr_schedule.out" \
  "${TMP_DIR}/batch_desired_kl_nan.out" \
  "${TMP_DIR}/batch_desired_kl_zero.out" \
  "${TMP_DIR}/batch_actor_lr_lower_bound.out" \
  "${TMP_DIR}/batch_actor_lr_upper_bound.out" \
  "${TMP_DIR}/batch_critic_lr_lower_bound.out"; do
  if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' "${ppo_lr_preflight_output}" >/dev/null; then
    fail 'invalid PPO LR controller settings must fail before snapshot construction or remote actions'
  fi
done
unset ppo_lr_preflight_output
expect_failure \
  "${TMP_DIR}/batch_actor_min_noise_inf.out" \
  'ACTOR_MIN_NOISE_STD must be finite' \
  "${BATCH_ENV[@]}" ACTOR_MIN_NOISE_STD=inf bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_init_noise_zero.out" \
  'INIT_NOISE_STD must be finite and > 0' \
  "${BATCH_ENV[@]}" INIT_NOISE_STD=0 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_entropy_nan.out" \
  'ENTROPY_COEF must be finite' \
  "${BATCH_ENV[@]}" ENTROPY_COEF=nan bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_ppo_start_coeff_nan.out" \
  'PPO_START_COEFF must be finite' \
  "${BATCH_ENV[@]}" PPO_START_COEFF=nan bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_ppo_target_coeff_range.out" \
  'PPO_TARGET_COEFF must be a finite probability in [0, 1]' \
  "${BATCH_ENV[@]}" PPO_TARGET_COEFF=2 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_ppo_coeff_relation.out" \
  'PPO_START_COEFF must be <= PPO_TARGET_COEFF' \
  "${BATCH_ENV[@]}" PPO_START_COEFF=0.8 PPO_TARGET_COEFF=0.7 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_dagger_loss_negative.out" \
  'DAGGER_LOSS_COEF must be finite and > 0' \
  "${BATCH_ENV[@]}" DAGGER_LOSS_COEF=-1 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_dagger_match_invalid.out" \
  'DAGGER_MATCH_STD must be a boolean' \
  "${BATCH_ENV[@]}" DAGGER_MATCH_STD=banana bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_ppo_start_noise_negative.out" \
  'PPO_START_NOISE_STD must be finite and > 0' \
  "${BATCH_ENV[@]}" PPO_START_NOISE_STD=-1 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_ppo_noise_threshold_inf.out" \
  'PPO_START_NOISE_STD_UNTIL_COEFF must be finite' \
  "${BATCH_ENV[@]}" PPO_START_NOISE_STD_UNTIL_COEFF=inf bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_reverse_reset_curriculum.out" \
  'START_AT_TIMESTEP_ZERO_PROB_START_ITER must be <= START_AT_TIMESTEP_ZERO_PROB_END_ITER' \
  "${BATCH_ENV[@]}" TARGET_LEARNING_ITERATION=5 FIXED_BC_GUARD_ENABLED=False \
  START_AT_TIMESTEP_ZERO_PROB_START_ITER=6 \
  START_AT_TIMESTEP_ZERO_PROB_END_ITER=5 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_reset_curriculum_beyond_target.out" \
  'FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER must be <= TARGET_LEARNING_ITERATION' \
  "${BATCH_ENV[@]}" TARGET_LEARNING_ITERATION=5 FIXED_BC_GUARD_ENABLED=False \
  FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=0 \
  FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=6 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_fresh_reset_curriculum_at_exclusive_target.out" \
  'Fresh/policy-init FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER must be < TARGET_LEARNING_ITERATION' \
  "${BATCH_ENV[@]}" TARGET_LEARNING_ITERATION=5 FIXED_BC_GUARD_ENABLED=False \
  FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=0 \
  FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=5 bash batch_ne.sh launch
for controller_numeric_output in \
  "${TMP_DIR}/batch_actor_lr_nan.out" \
  "${TMP_DIR}/batch_critic_lr_negative.out" \
  "${TMP_DIR}/batch_actor_min_noise_inf.out" \
  "${TMP_DIR}/batch_init_noise_zero.out" \
  "${TMP_DIR}/batch_entropy_nan.out" \
  "${TMP_DIR}/batch_ppo_start_coeff_nan.out" \
  "${TMP_DIR}/batch_ppo_target_coeff_range.out" \
  "${TMP_DIR}/batch_ppo_coeff_relation.out" \
  "${TMP_DIR}/batch_dagger_loss_negative.out" \
  "${TMP_DIR}/batch_dagger_match_invalid.out" \
  "${TMP_DIR}/batch_ppo_start_noise_negative.out" \
  "${TMP_DIR}/batch_ppo_noise_threshold_inf.out" \
  "${TMP_DIR}/batch_reverse_reset_curriculum.out" \
  "${TMP_DIR}/batch_reset_curriculum_beyond_target.out"; do
  if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' "${controller_numeric_output}" >/dev/null; then
    fail 'invalid controller scientific numerics must fail before snapshot construction or remote actions'
  fi
done
unset controller_numeric_output

expect_failure \
  "${TMP_DIR}/batch_nodes_glob.out" \
  'Unsafe node identifier in NODES: *' \
  env NODES='*' NNODES=1 NPROC=1 PER_GPU_ENVS=1024 DRY_RUN=1 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_elastic_replay.out" \
  'Scientific launch requires MAX_RESTARTS=0 exactly' \
  "${BATCH_ENV[@]}" MAX_RESTARTS=1 bash batch_ne.sh launch
if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' "${TMP_DIR}/batch_elastic_replay.out" >/dev/null; then
  fail 'unsafe torchrun automatic restart must fail before snapshot construction or remote actions'
fi
expect_failure \
  "${TMP_DIR}/batch_unsafe_master.out" \
  'MASTER_ADDR must be a safe host/IP identifier' \
  "${BATCH_ENV[@]}" MASTER_ADDR=-oProxyCommand=bad bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_non_rank0_master.out" \
  'MASTER_ADDR must exactly equal the rank-0 host NODES[0]' \
  "${BATCH_ENV[@]}" MASTER_ADDR=other-test-node bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_noncanonical_nnodes.out" \
  'NNODES must equal node list length' \
  "${BATCH_ENV[@]}" NNODES=01 bash batch_ne.sh launch

arithmetic_marker="${TMP_DIR}/batch_arithmetic_injection"
expect_failure \
  "${TMP_DIR}/batch_unsafe_master_port.out" \
  'MASTER_PORT must be an integer in [1, 65535]' \
  "${BATCH_ENV[@]}" MASTER_PORT='port[$(touch '"${arithmetic_marker}"')]' \
  bash batch_ne.sh launch
[[ ! -e "${arithmetic_marker}" ]] ||
  fail 'MASTER_PORT was evaluated as shell arithmetic before validation'
expect_failure \
  "${TMP_DIR}/batch_unsafe_min_envs.out" \
  'MIN_PER_GPU_ENVS must be a canonical integer in [1, 2147483647]' \
  "${BATCH_ENV[@]}" MIN_PER_GPU_ENVS='minimum[$(touch '"${arithmetic_marker}"')]' \
  bash batch_ne.sh launch
[[ ! -e "${arithmetic_marker}" ]] ||
  fail 'MIN_PER_GPU_ENVS was evaluated as shell arithmetic before validation'

expect_failure \
  "${TMP_DIR}/invalid_batch_cublas.out" \
  'CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8' \
  "${BATCH_ENV[@]}" CUBLAS_WORKSPACE_CONFIG=:bad bash batch_ne.sh launch
if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' "${TMP_DIR}/invalid_batch_cublas.out" >/dev/null; then
  fail 'invalid cuBLAS runtime settings must fail before snapshot construction or remote actions'
fi

expect_failure \
  "${TMP_DIR}/invalid_batch_hash_seed.out" \
  'PYTHONHASHSEED must be an integer in [0, 4294967295]' \
  "${BATCH_ENV[@]}" PYTHONHASHSEED=4294967296 bash batch_ne.sh launch
if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' "${TMP_DIR}/invalid_batch_hash_seed.out" >/dev/null; then
  fail 'invalid hash seeds must fail before snapshot construction or remote actions'
fi

expect_failure \
  "${TMP_DIR}/invalid_batch_training_seed.out" \
  'TRAINING_SEED/SEED must be an integer in [0, 4294967295]' \
  "${BATCH_ENV[@]}" TRAINING_SEED=not-an-integer bash batch_ne.sh launch
if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' "${TMP_DIR}/invalid_batch_training_seed.out" >/dev/null; then
  fail 'invalid training seeds must fail before snapshot construction or remote actions'
fi

expect_failure \
  "${TMP_DIR}/overflowing_batch_rank_seed.out" \
  'TRAINING_SEED plus rank offsets must stay <= 4294967295' \
  env NODES=test-node NPROC=2 NNODES=1 PER_GPU_ENVS=1024 DRY_RUN=1 SKIP_GIT_PULL=1 \
    SOURCE_SNAPSHOT_CACHE="${TMP_DIR}/snapshot-cache" TRAINING_SEED=4294967295 \
    bash batch_ne.sh launch
if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' "${TMP_DIR}/overflowing_batch_rank_seed.out" >/dev/null; then
  fail 'rank-offset seed overflow must fail before snapshot construction or remote actions'
fi

for physx_capacity_name in \
  PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY \
  PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY \
  PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY \
  PHYSX_GPU_COLLISION_STACK_SIZE; do
  physx_failure_output="${TMP_DIR}/invalid_${physx_capacity_name}.out"
  expect_failure \
    "${physx_failure_output}" \
    "${physx_capacity_name} must be a canonical integer in [1, 2147483647]" \
    "${BATCH_ENV[@]}" "${physx_capacity_name}=not-a-number" bash batch_ne.sh launch
  if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' "${physx_failure_output}" >/dev/null; then
    fail "invalid ${physx_capacity_name} must fail before snapshot construction or remote actions"
  fi
done
unset physx_capacity_name physx_failure_output

expect_failure \
  "${TMP_DIR}/invalid_object_collider_type.out" \
  'HOLOSOMA_OBJECT_COLLIDER_TYPE must be exactly convex_decomposition or convex_hull' \
  "${BATCH_ENV[@]}" HOLOSOMA_OBJECT_COLLIDER_TYPE=triangle_mesh bash batch_ne.sh launch
if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' "${TMP_DIR}/invalid_object_collider_type.out" >/dev/null; then
  fail 'invalid object collider type must fail before snapshot construction or remote actions'
fi

"${BATCH_ENV[@]}" bash batch_ne.sh launch >"${TMP_DIR}/batch_default.out"
"${BATCH_ENV[@]}" HOLOSOMA_OBJECT_COLLIDER_TYPE=convex_hull \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_convex_hull.out"
grep -F 'export HOLOSOMA_OBJECT_COLLIDER_TYPE=convex_hull' \
  "${TMP_DIR}/batch_convex_hull.out" >/dev/null ||
  fail 'batch must carry the explicit convex-hull collider into the sealed node payload'
"${BATCH_ENV[@]}" SHOO7SR1_NEAR03_DEBUG=1 SHOO7SR1_OBS_VARIANT=baseline \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_poisoned_shoo.out"
grep -F 'unset SHOO7SR1_NEAR03_DEBUG' "${TMP_DIR}/batch_poisoned_shoo.out" >/dev/null
grep -F 'unset SHOO7SR1_OBS_VARIANT' "${TMP_DIR}/batch_poisoned_shoo.out" >/dev/null
grep -F 'export SHOO7SR1_NEAR03_DEBUG=0' "${TMP_DIR}/batch_poisoned_shoo.out" >/dev/null
grep -F 'export DAGGER_END_EPOCH=4900' "${TMP_DIR}/batch_poisoned_shoo.out" >/dev/null
grep -F 'export PPO_TARGET_COEFF=0.7' "${TMP_DIR}/batch_poisoned_shoo.out" >/dev/null

"${BATCH_ENV[@]}" TARGET_LEARNING_ITERATION=8 SAVE_INTERVAL=1000 \
  FIXED_BC_EVAL_LOG_INTERVAL=1 FIXED_BC_GUARD_ENABLED=False \
  PPO_START_EPOCH=0 DAGGER_END_EPOCH=7 PPO_START_COEFF=0.0 \
  PPO_TARGET_COEFF=0.7 PPO_SCHEDULE_STEP_EPOCHS=1 \
  START_AT_TIMESTEP_ZERO_PROB_START_ITER=0 \
  FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=0 \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_eight_iter_canary.out"
for canary_export in \
  'export TARGET_LEARNING_ITERATION=8' \
  'export SAVE_INTERVAL=1000' \
  'export FIXED_BC_EVAL_LOG_INTERVAL=1' \
  'export FIXED_BC_GUARD_ENABLED=False' \
  'export FIXED_BC_GUARD_START_EPOCH=-1' \
  'export PPO_START_EPOCH=0' \
  'export DAGGER_END_EPOCH=7' \
  'export PPO_TARGET_COEFF=0.7' \
  'export PPO_SCHEDULE_STEP_EPOCHS=1' \
  'export START_AT_TIMESTEP_ZERO_PROB_END_ITER=7' \
  'export FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=7'; do
  grep -F "${canary_export}" "${TMP_DIR}/batch_eight_iter_canary.out" >/dev/null ||
    fail "8-iteration canary lost ${canary_export}"
done
unset canary_export
for required_wandb_line in \
  'for _wandb_env_name in ${!WANDB_@} ${!_WANDB_@}; do' \
  'unset WANDB_MODE' \
  'unset WANDB_DISABLED' \
  'unset WANDB_BASE_URL' \
  'unset WANDB_DIR' \
  'unset WANDB_INIT_TIMEOUT' \
  'unset WANDB_CONSOLE' \
  'unset HOLOSOMA_REQUIRE_WANDB_RUN' \
  'unset LOGGER' \
  'export LOGGER=logger:wandb' \
  'TRAIN_EXTRA_ARGS=(--logger.entity="${WANDB_ENTITY}")' \
  'export WANDB_BASE_URL=https://api.wandb.ai' \
  'export WANDB_INIT_TIMEOUT=120' \
  'export WANDB_CONSOLE=off' \
  'export HOLOSOMA_REQUIRE_WANDB_RUN=1'; do
  grep -F "${required_wandb_line}" "${TMP_DIR}/batch_default.out" >/dev/null ||
    fail "default dry-run is missing required W&B contract line: ${required_wandb_line}"
done
unset required_wandb_line

# Node/login settings must not select W&B's rank-zero-only console wrapper.
# The scientific launcher fixes tee as the authoritative cross-rank log even
# when an ambient caller explicitly requests redirect mode.
"${BATCH_ENV[@]}" WANDB_CONSOLE=redirect \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_wandb_console_override.out"
grep -F 'unset WANDB_CONSOLE' "${TMP_DIR}/batch_wandb_console_override.out" >/dev/null ||
  fail 'batch dry-run did not erase ambient WANDB_CONSOLE'
grep -F 'export WANDB_CONSOLE=off' "${TMP_DIR}/batch_wandb_console_override.out" >/dev/null ||
  fail 'batch dry-run did not restore the canonical unwrapped W&B console contract'
if grep -F 'export WANDB_CONSOLE=redirect' \
    "${TMP_DIR}/batch_wandb_console_override.out" >/dev/null; then
  fail 'ambient WANDB_CONSOLE overrode the canonical tee logging path'
fi

# Short canaries must not inherit the delegated wrapper's historical
# 2500->TARGET reversal.  The controller owns and serializes a bounded 0->5
# reset curriculum, and normalizes permissive boolean spellings before launch.
"${BATCH_ENV[@]}" TARGET_LEARNING_ITERATION=5 DAGGER_MATCH_STD=yes FIXED_BC_GUARD_ENABLED=False \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_short_canary.out"
grep -F 'export DAGGER_MATCH_STD=True' "${TMP_DIR}/batch_short_canary.out" >/dev/null ||
  fail 'batch did not normalize DAGGER_MATCH_STD=yes to the canonical Tyro boolean'
for canary_schedule_export in \
  'export START_AT_TIMESTEP_ZERO_PROB=0.2' \
  'export START_AT_TIMESTEP_ZERO_PROB_END=1.0' \
  'export START_AT_TIMESTEP_ZERO_PROB_START_ITER=0' \
  'export START_AT_TIMESTEP_ZERO_PROB_END_ITER=4' \
  'export FREEZE_AT_TIMESTEP_ZERO_PROB=0.0' \
  'export FREEZE_AT_TIMESTEP_ZERO_PROB_END=0.0' \
  'export FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=0' \
  'export FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=4'; do
  grep -F "${canary_schedule_export}" "${TMP_DIR}/batch_short_canary.out" >/dev/null ||
    fail "short canary is missing bounded reset curriculum export: ${canary_schedule_export}"
done
unset canary_schedule_export

# TARGET is an exclusive rollout bound.  The controller-owned defaults must
# reach their end value on TARGET-1 across the degenerate one-rollout case and
# both sides of the historical iteration-2500 start boundary.
"${BATCH_ENV[@]}" TARGET_LEARNING_ITERATION=1 FIXED_BC_GUARD_ENABLED=False \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_target_one_curriculum.out"
for target_one_export in \
  'export START_AT_TIMESTEP_ZERO_PROB_START_ITER=0' \
  'export START_AT_TIMESTEP_ZERO_PROB_END_ITER=0' \
  'export FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=0' \
  'export FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=0'; do
  grep -F "${target_one_export}" "${TMP_DIR}/batch_target_one_curriculum.out" >/dev/null ||
    fail "target=1 reset curriculum lost its only reachable rollout: ${target_one_export}"
done
unset target_one_export

"${BATCH_ENV[@]}" TARGET_LEARNING_ITERATION=2500 FIXED_BC_GUARD_ENABLED=False \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_target_2500_curriculum.out"
for target_2500_export in \
  'export START_AT_TIMESTEP_ZERO_PROB_START_ITER=0' \
  'export START_AT_TIMESTEP_ZERO_PROB_END_ITER=2499' \
  'export FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=0' \
  'export FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=2499'; do
  grep -F "${target_2500_export}" "${TMP_DIR}/batch_target_2500_curriculum.out" >/dev/null ||
    fail "target=2500 reset curriculum crossed its final reachable rollout: ${target_2500_export}"
done
unset target_2500_export

"${BATCH_ENV[@]}" TARGET_LEARNING_ITERATION=2501 FIXED_BC_GUARD_ENABLED=False \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_target_2501_curriculum.out"
for target_2501_export in \
  'export START_AT_TIMESTEP_ZERO_PROB_START_ITER=2500' \
  'export START_AT_TIMESTEP_ZERO_PROB_END_ITER=2500' \
  'export FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=2500' \
  'export FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=2500'; do
  grep -F "${target_2501_export}" "${TMP_DIR}/batch_target_2501_curriculum.out" >/dev/null ||
    fail "target=2501 reset curriculum lost the exact iteration-2500 boundary: ${target_2501_export}"
done
unset target_2501_export

# A legacy full-training resume may have an exact-bound saved schedule whose
# end equals the new exclusive target.  Preserve that manifest-bound lineage;
# fresh and actor-only launches are rejected by the failure case above.
"${BATCH_ENV[@]}" TARGET_LEARNING_ITERATION=8 FIXED_BC_GUARD_ENABLED=False \
  RESUME_TRAINING_CKPT=wandb://entity/project/run/model_1.pt \
  START_AT_TIMESTEP_ZERO_PROB_START_ITER=0 START_AT_TIMESTEP_ZERO_PROB_END_ITER=8 \
  FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=0 FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=8 \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_full_resume_legacy_equal_target_curriculum.out"
grep -F 'export START_AT_TIMESTEP_ZERO_PROB_END_ITER=8' \
  "${TMP_DIR}/batch_full_resume_legacy_equal_target_curriculum.out" >/dev/null ||
  fail 'full resume did not preserve an exact-bound legacy start-at-zero schedule'
grep -F 'export FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=8' \
  "${TMP_DIR}/batch_full_resume_legacy_equal_target_curriculum.out" >/dev/null ||
  fail 'full resume did not preserve an exact-bound legacy freeze-at-zero schedule'

# A full-resume checkpoint is authoritative for an implicit policy type.  Do
# not reject explicit flow parameters against the temporary controller-side
# MLP default before the node verifies and parses the persisted actor type.
"${BATCH_ENV[@]}" \
  RESUME_TRAINING_CKPT=wandb://entity/project/run/model_100.pt \
  STUDENT_FLOW_STEPS=8 STUDENT_FLOW_TIME_EPSILON=0.01 \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_deferred_resume_policy_type.out"
grep -F 'export STUDENT_POLICY_TYPE_EXPLICIT=0' \
  "${TMP_DIR}/batch_deferred_resume_policy_type.out" >/dev/null ||
  fail 'resume launch did not preserve implicit policy-type provenance for node-side inference'
grep -F 'export STUDENT_FLOW_STEPS_EXPLICIT=1' \
  "${TMP_DIR}/batch_deferred_resume_policy_type.out" >/dev/null ||
  fail 'resume launch lost the explicit flow-step comparison contract'

# All accepted true spellings must remain non-mutating.  In particular,
# DRY_RUN=true may never fall through to the real SSH path.
DRY_RUN_REMOTE_MARKER="${TMP_DIR}/dry_run_true_invoked_ssh"
DRY_RUN_PYTHON_MARKER="${TMP_DIR}/dry_run_true_invoked_ambient_python"
mkdir -p "${TMP_DIR}/dry-run-fake-bin"
cat >"${TMP_DIR}/dry-run-fake-bin/ssh" <<EOF
#!/usr/bin/env bash
touch "${DRY_RUN_REMOTE_MARKER}"
exit 97
EOF
cat >"${TMP_DIR}/dry-run-fake-bin/python3" <<EOF
#!/usr/bin/env bash
touch "${DRY_RUN_PYTHON_MARKER}"
exit 98
EOF
chmod +x "${TMP_DIR}/dry-run-fake-bin/ssh" "${TMP_DIR}/dry-run-fake-bin/python3"
"${BATCH_ENV[@]}" PATH="${TMP_DIR}/dry-run-fake-bin:${PATH}" DRY_RUN=TrUe \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_dry_run_true.out"
grep -F '[DRY_RUN]' "${TMP_DIR}/batch_dry_run_true.out" >/dev/null ||
  fail 'DRY_RUN=true did not enter the dry-run path'
[[ ! -e "${DRY_RUN_REMOTE_MARKER}" ]] ||
  fail 'DRY_RUN=true executed a real SSH operation'
[[ ! -e "${DRY_RUN_PYTHON_MARKER}" ]] ||
  fail 'batch controller validation selected ambient python3 instead of the resolved PYTHON_BIN'

# Controller-level interpreter determinism settings must be serialized into
# every independent remote shell. ssh does not forward arbitrary environment
# variables, so relying on the remote defaults would silently ignore an
# explicitly configured experiment contract.
"${BATCH_ENV[@]}" PYTHONHASHSEED=7 CUBLAS_WORKSPACE_CONFIG=:16:8 \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_explicit_runtime.out"
[[ "$(grep -Fc 'export PYTHONHASHSEED=7' "${TMP_DIR}/batch_explicit_runtime.out")" -ge 3 ]] ||
  fail 'launch must forward PYTHONHASHSEED to launch-preflight, health, and train shells'
[[ "$(grep -Fc 'export CUBLAS_WORKSPACE_CONFIG=:16:8' "${TMP_DIR}/batch_explicit_runtime.out")" -ge 3 ]] ||
  fail 'launch must forward CUBLAS_WORKSPACE_CONFIG to every remote Python shell'

"${BATCH_ENV[@]}" TRAINING_SEED=123 bash batch_ne.sh launch \
  >"${TMP_DIR}/batch_explicit_training_seed.out"
grep -F 'export TRAINING_SEED=123' "${TMP_DIR}/batch_explicit_training_seed.out" >/dev/null ||
  fail 'batch launch must forward the explicit training base seed across SSH'
grep -F 'unset SEED' "${TMP_DIR}/batch_explicit_training_seed.out" >/dev/null ||
  fail 'batch launch must clear the legacy SEED alias after canonicalizing TRAINING_SEED'
"${BATCH_ENV[@]}" SEED=456 bash batch_ne.sh launch \
  >"${TMP_DIR}/batch_legacy_training_seed.out"
grep -F 'export TRAINING_SEED=456' "${TMP_DIR}/batch_legacy_training_seed.out" >/dev/null ||
  fail 'batch launch must canonicalize and forward the legacy SEED alias'

"${BATCH_ENV[@]}" PYTHONHASHSEED=7 CUBLAS_WORKSPACE_CONFIG=:16:8 PREPARE_DATA=0 \
  bash batch_ne.sh prepare >"${TMP_DIR}/batch_prepare_explicit_runtime.out"
grep -F 'export PYTHONHASHSEED=7' "${TMP_DIR}/batch_prepare_explicit_runtime.out" >/dev/null ||
  fail 'prepare must forward PYTHONHASHSEED before its first remote Python process'
grep -F 'export CUBLAS_WORKSPACE_CONFIG=:16:8' \
  "${TMP_DIR}/batch_prepare_explicit_runtime.out" >/dev/null ||
  fail 'prepare must forward CUBLAS_WORKSPACE_CONFIG before its first remote Python process'

eula_semicolon_marker="${TMP_DIR}/eula-semicolon-injected"
eula_newline_marker="${TMP_DIR}/eula-newline-injected"
"${BATCH_ENV[@]}" \
  OMNI_KIT_ACCEPT_EULA="YES; touch ${eula_semicolon_marker}" \
  ACCEPT_EULA=$'Y\ntouch '"${eula_newline_marker}" \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_quoted_eula.out"
sed -n '/^export OMNI_KIT_ACCEPT_EULA=/,/^export NPROC=/p' \
  "${TMP_DIR}/batch_quoted_eula.out" >"${TMP_DIR}/quoted_eula_fragment.sh"
bash "${TMP_DIR}/quoted_eula_fragment.sh"
if [[ -e "${eula_semicolon_marker}" || -e "${eula_newline_marker}" ]]; then
  fail 'EULA values must remain quoted assignment data in the generated remote shell'
fi

for scientific_skip_name in \
  HOLOSOMA_SKIP_GRAD_FINITE_CHECK \
  HOLOSOMA_SKIP_LOSS_FINITE_CHECK \
  HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION \
  HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC; do
  scientific_skip_output="${TMP_DIR}/forbidden_${scientific_skip_name}.out"
  expect_failure \
    "${scientific_skip_output}" \
    "Scientific batch launch forbids ${scientific_skip_name}=1" \
    "${BATCH_ENV[@]}" "${scientific_skip_name}=yes" bash batch_ne.sh launch
  if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' "${scientific_skip_output}" >/dev/null; then
    fail "${scientific_skip_name}=1 must fail before snapshot construction or remote actions"
  fi
done
unset scientific_skip_name scientific_skip_output

expect_failure \
  "${TMP_DIR}/batch_missing_pre_gradient_sync.out" \
  'Scientific batch launch requires HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE=1' \
  "${BATCH_ENV[@]}" HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE=false bash batch_ne.sh launch
if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' "${TMP_DIR}/batch_missing_pre_gradient_sync.out" >/dev/null; then
  fail 'disabled pre-gradient synchronization must fail before snapshot construction or remote actions'
fi

grep -F '[INFO] Launching 1 nodes x 1 GPUs' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_SKIP_GRAD_FINITE_CHECK=0' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_SKIP_LOSS_FINITE_CHECK=0' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION=0' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC=0' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE=1' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_DAGGER_SUPERVISED_ONLY=0' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP=0' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH=16' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD=1' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export ACTOR_MIN_NOISE_STD=0.01' "${TMP_DIR}/batch_default.out" >/dev/null ||
  fail 'batch must pin the scientific actor minimum noise standard deviation'
grep -F 'export INIT_NOISE_STD=0.01' "${TMP_DIR}/batch_default.out" >/dev/null ||
  fail 'batch must pin the scientific initial policy noise standard deviation'
grep -F 'export ENTROPY_COEF=0.0' "${TMP_DIR}/batch_default.out" >/dev/null ||
  fail 'batch must pin zero entropy regularization'
grep -F 'supervised_actor_microbatch requested=${HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH} effective=${effective_supervised_actor_microbatch}' \
  "${TMP_DIR}/batch_default.out" >/dev/null ||
  fail 'batch launch must log requested and effective supervised-only microbatch semantics'
grep -F 'export OMNI_KIT_ACCEPT_EULA=YES' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export ACCEPT_EULA=Y' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=301989888' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY=301989888' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=134217728' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export PHYSX_GPU_COLLISION_STACK_SIZE=268435456' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_OBJECT_COLLIDER_TYPE=convex_decomposition' "${TMP_DIR}/batch_default.out" >/dev/null ||
  fail 'batch must preserve the validated convex-decomposition collider by default'
grep -F 'export HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=0' "${TMP_DIR}/batch_default.out" >/dev/null ||
  fail 'batch must preserve the validated object reporter/sleep contract by default'
grep -F 'export DAGGER_MATCH_STD=False' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export PPO_START_NOISE_STD=0.1' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export PPO_START_COEFF=0.0' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export START_AT_TIMESTEP_ZERO_PROB=0.2' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export START_AT_TIMESTEP_ZERO_PROB_END=1.0' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export START_AT_TIMESTEP_ZERO_PROB_START_ITER=2500' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export START_AT_TIMESTEP_ZERO_PROB_END_ITER=39999' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export FREEZE_AT_TIMESTEP_ZERO_PROB=0.0' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export FREEZE_AT_TIMESTEP_ZERO_PROB_END=0.0' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=2500' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=39999' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export TEACHER_CHECKPOINT=wandb://zihanw22/carry-any/bcleb5oi/model_67000.pt' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export TARGET_LEARNING_ITERATION=40000' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export EXPORT_ONNX=False' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export ZERO_ROOT_COMMAND_WHEN_DROP_ACTIVE=False' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export STUDENT_MOTION_END_MODE=episodic' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export FIXED_BC_EVAL_LOG_INTERVAL=100' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export FIXED_BC_GUARD_ENABLED=True' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export FIXED_BC_GUARD_REFERENCE_END_EPOCH=600' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export FIXED_BC_GUARD_MAX_REFERENCE_RATIO=2.0' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export FIXED_BC_GUARD_ABSOLUTE_MAX_MU_MSE=0.160' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export FIXED_BC_GUARD_START_EPOCH=4900' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export FIXED_BC_GUARD_CONSECUTIVE_EVALS=3' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export DAGGER_END_EPOCH=4900' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export PPO_TARGET_COEFF=0.7' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_MOTION_METRICS_INTERVAL=16' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export TORCH_DIST_TIMEOUT_SEC=3600' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC=300' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export MAX_RESTARTS=0' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export TORCH_NCCL_ENABLE_MONITORING=1' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export TORCH_NCCL_TRACE_BUFFER_SIZE=65536' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export NCCL_SOCKET_RETRY_CNT=34' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export NCCL_SOCKET_RETRY_SLEEP_MSEC=100' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export SAVE_INTERVAL=1000' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export TORCH_DIST_BACKEND=gloo' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export NCCL_LIB_SHA256=' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'if [[ gloo == nccl || 0 == 1 ]]; then' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'if [[ "${TORCH_DIST_BACKEND}" == "nccl" || "${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE}" == "1" ]]; then' \
  "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_GLOO_GRAD_REDUCE=1' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_GLOO_BARRIER=1' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_GLOO_SMALL_COLLECTIVES=1' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=0' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=0' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER=0' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_RANK_VISIBLE_DEVICES=0' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export ALLOW_NONDETERMINISTIC_RNG_RESUME=0' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export ALLOW_FRESH_CURRICULUM_RESUME=0' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_ALLOW_FRESH_CURRICULUM_RESUME=0' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME=0' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_ALLOW_RUNTIME_DRIFT_ON_RESUME=0' "${TMP_DIR}/batch_default.out" >/dev/null
if grep -F 'export CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION=' "${TMP_DIR}/batch_default.out" >/dev/null; then
  fail 'batch must leave contact prepend compensation unset so a training resume can infer the saved contract'
fi
grep -F 'scripts/resolve_exact_checkpoint.py' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'Python interpreter split-brain:' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F '[INFO][python-preflight] executable=' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'CUDA_VISIBLE_DEVICES selects an unknown GPU index/UUID' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'scripts/validate_contact_sidecars.py' "${TMP_DIR}/batch_default.out" >/dev/null
grep -E 'source_snapshot_id=src-[0-9a-f]{64}' "${TMP_DIR}/batch_default.out" >/dev/null
grep -E 'remote_run_repo=/home/ubuntu/FAR/holosoma_runs/src-[0-9a-f]{64} remote_asset_repo=/home/ubuntu/FAR/holosoma' \
  "${TMP_DIR}/batch_default.out" >/dev/null
grep -E 'export PYTHONPATH=/home/ubuntu/FAR/holosoma_runs/src-[0-9a-f]{64}/src/holosoma:' \
  "${TMP_DIR}/batch_default.out" >/dev/null
grep -E 'export PYTHON_BIN=.*/envs/hssim/bin/python3(\.[0-9]+)?' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export HOLOSOMA_PYTHON_PROFILE=hssim' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export PYTHON_RUNTIME_SITEPACKAGES=' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export PYTHON_RUNTIME_MANIFEST_SHA256=' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export PYTHONDONTWRITEBYTECODE=1' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export PYTHONNOUSERSITE=1' "${TMP_DIR}/batch_default.out" >/dev/null
[[ "$(grep -Fc 'export PYTHONNOUSERSITE=1' "${TMP_DIR}/batch_default.out")" -ge 3 ]] ||
  fail 'launch preflight, health preflight, and payload must all disable user site-packages'
[[ "$(grep -Ec '^export PATH=.*/envs/hssim/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin$' "${TMP_DIR}/batch_default.out")" -ge 3 ]] ||
  fail 'launch preflight, health preflight, and payload must share one exact PATH'
[[ "$(grep -Ec '^export PYTHONPATH=.*/src/holosoma:.*/src/holosoma_inference:.*/src$' "${TMP_DIR}/batch_default.out")" -ge 3 ]] ||
  fail 'launch preflight, health preflight, and payload must share one exact PYTHONPATH'
[[ "$(grep -Fc 'export LD_LIBRARY_PATH=/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/nvidia/nccl/lib' "${TMP_DIR}/batch_default.out")" -ge 3 ]] ||
  fail 'launch preflight, health preflight, and payload must share one exact LD_LIBRARY_PATH'
[[ "$(grep -Fc 'unset LD_PRELOAD' "${TMP_DIR}/batch_default.out")" -ge 3 ]] ||
  fail 'launch preflight, health preflight, and payload must clear ambient LD_PRELOAD'
[[ "$(grep -Fc 'unset BASH_ENV ENV CDPATH' "${TMP_DIR}/batch_default.out")" -ge 3 ]] ||
  fail 'launch preflight, health preflight, and payload must clear inherited shell startup hooks'
[[ "$(grep -Fc 'unset PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONOPTIMIZE PYTHONWARNINGS PYTHONBREAKPOINT PYTHONSAFEPATH' "${TMP_DIR}/batch_default.out")" -ge 3 ]] ||
  fail 'launch preflight, health preflight, and payload must clear inherited Python startup controls'
if grep -F '${PYTHONPATH:+:${PYTHONPATH}}' "${TMP_DIR}/batch_default.out" >/dev/null; then
  fail 'batch node control must not append an inherited remote PYTHONPATH'
fi
if grep -F '${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' "${TMP_DIR}/batch_default.out" >/dev/null; then
  fail 'batch node control must not append an inherited remote LD_LIBRARY_PATH'
fi
if grep -F '${LD_PRELOAD:+:${LD_PRELOAD}}' "${TMP_DIR}/batch_default.out" >/dev/null; then
  fail 'batch node control must not append an inherited remote LD_PRELOAD'
fi
grep -E 'export PATH=.*/envs/hssim/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' \
  "${TMP_DIR}/batch_default.out" >/dev/null ||
  fail 'batch node control must pin one exact executable search path'
grep -F 'scripts/verify_python_runtime_overlay.py' "${TMP_DIR}/batch_default.out" >/dev/null
[[ "$(grep -Fc -- '--require-distribution-closure' batch_ne.sh)" == 5 ]] ||
  fail 'prepare, pre-intent barrier, launch preflight, health preflight, and train payload must all require the exact distribution closure'
[[ "$(grep -Fc -- '--require-current-runtime-binding' batch_ne.sh)" == 5 ]] ||
  fail 'prepare, pre-intent barrier, launch preflight, health preflight, and train payload must all prove live interpreter binding'
grep -F 'python_runtime_launch_overlay_verified=${PYTHON_RUNTIME_SITEPACKAGES}' \
  "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'python_runtime_train_overlay_verified=${PYTHON_RUNTIME_SITEPACKAGES}' \
  "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'python_runtime_prepare_overlay_verified=\${PYTHON_RUNTIME_SITEPACKAGES}' \
  batch_ne.sh >/dev/null
[[ "$(grep -Fc 'export PYTHONNOUSERSITE=1' "${TMP_DIR}/batch_prepare_explicit_runtime.out")" -ge 1 ]] ||
  fail 'prepare must disable user site-packages before its first remote Python process'
grep -E '^export PATH=.*/envs/hssim/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin$' \
  "${TMP_DIR}/batch_prepare_explicit_runtime.out" >/dev/null ||
  fail 'prepare must use the same exact executable search path as launch'
grep -F 'unset LD_PRELOAD' "${TMP_DIR}/batch_prepare_explicit_runtime.out" >/dev/null ||
  fail 'prepare must clear ambient LD_PRELOAD before its first remote Python process'
grep -F 'find "${SITE_PACKAGES}" -type d -exec chmod 555 {} +' \
  scripts/build_numpy_runtime_overlay.sh >/dev/null ||
  fail 'runtime overlay builder must canonicalize every directory to 0555'
grep -F 'find "${SITE_PACKAGES}" -type f -exec chmod 444 {} +' \
  scripts/build_numpy_runtime_overlay.sh >/dev/null ||
  fail 'runtime overlay builder must canonicalize every file to 0444'
grep -F 'export LOGGER_BASE_DIR=/home/ubuntu/FAR/holosoma_runs/training_logs' \
  "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'logger_base_dir_health_verified=${LOGGER_BASE_DIR}' \
  "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'export PYTHONPATH=$(quote "${runtime_pythonpath}")' \
  batch_ne.sh >/dev/null
grep -F 'CONTACT_VALIDATOR_RUNTIME_ARGS+=(--runtime-prepend-compensation --runtime-prepend-duration-s 0.2)' \
  batch_ne.sh >/dev/null
grep -F 'train_args+=(--logger.base-dir="${LOGGER_BASE_DIR}")' distill_torso_box.sh >/dev/null
grep -E '\[DRY_RUN\] stream launch script over ssh stdin node=test-node path=.*/\.run_control/train-[0-9a-f]{64}\.sh sha256=[0-9a-f]{64} bytes=[1-9][0-9]*' \
  "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'cat > "${INCOMING}"' "${TMP_DIR}/batch_default.out" >/dev/null
grep -F 'installed_verified_launch_script=${FINAL} sha256=${final_sha256}' \
  "${TMP_DIR}/batch_default.out" >/dev/null
grep -E 'tmux new-session -d -s .*exec\\ bash\\ /home/ubuntu/FAR/holosoma_runs/src-[0-9a-f]{64}/\.run_control/train-[0-9a-f]{64}\.sh 8>&-$' \
  "${TMP_DIR}/batch_default.out" >/dev/null
if awk '/tmux new-session/ && length($0) >= 4096 { found=1 } END { exit(found ? 0 : 1) }' \
  "${TMP_DIR}/batch_default.out"; then
  fail 'tmux launch argv must stay short and reference the streamed control script'
fi
grep -F 'tmux set-option -t ' batch_ne.sh | grep -F '@holosoma_launch_token' >/dev/null ||
  fail 'batch launch must stamp newly created tmux sessions with an ownership token'
grep -F '@holosoma_command_sha256' batch_ne.sh >/dev/null ||
  fail 'batch launch must stamp tmux with the streamed command SHA256'
grep -F 'validate_and_release_rendezvous_pair' batch_ne.sh >/dev/null ||
  fail 'rank0/controller cleanup must use the strict rendezvous-pair validator'
grep -F '"\${RENDEZVOUS_MAIN_STATE}" "\${RENDEZVOUS_PROVENANCE_STATE}"' batch_ne.sh >/dev/null &&
  grep -F '$(quote "${HOLOSOMA_PROVENANCE_MASTER_PORT}") 1' batch_ne.sh >/dev/null ||
  fail 'rank0 clean completion must require both exact reservation records to be present'
grep -F "printf '2\\tlaunching\\t%s\\t%s\\t%s\\t%s\\tpending\\t%s\\n'" batch_ne.sh >/dev/null ||
  fail 'batch launch must prepublish v2 all-node intents before starting tmux'
grep -F 'Refusing to stop ${SESSION} during launch rollback: session is not owned by this launch.' \
  batch_ne.sh >/dev/null || fail 'batch rollback must preserve sessions not owned by the current launch'
grep -F 'RESTART=${RESTART:-0}' batch_ne.sh >/dev/null ||
  fail 'multi-node launcher must default to non-destructive RESTART=0'
grep -F 'ACTIVE_STATE=' "${TMP_DIR}/batch_default.out" >/dev/null ||
  fail 'batch launch must publish active snapshot/log metadata for exact status binding'
grep -F 'completion_total=\$(grep -Ec '\''^HOLOSOMA_RUN_COMPLETE '\''' batch_ne.sh >/dev/null ||
  fail 'batch status must recognize authoritative successful completion markers'
grep -F 'effective BC weight is ${DAGGER_LOSS_COEF}*(1-PPO)' batch_ne.sh >/dev/null ||
  fail 'schedule provenance must record the configured DAgger coefficient in effective BC weight'
grep -F 'saved_iter, next_iter = validate_checkpoint_iterations(checkpoint)' batch_ne.sh >/dev/null ||
  fail 'training-resume launcher metadata must use the shared strict checkpoint iteration validator'
if grep -F 'checkpoint.get("next_iter", saved_iter + 1)' batch_ne.sh >/dev/null; then
  fail 'training-resume launcher must not silently accept contradictory iteration metadata'
fi
grep -F 'Launch predecessor CAS mismatch:' batch_ne.sh >/dev/null ||
  fail 'launch intent publication must compare-and-swap its exact preflight predecessor'
grep -F 'verify_no_launch_token_epoch_processes' batch_ne.sh >/dev/null ||
  fail 'intent cancellation closure must scan every process by token and epoch before release'
grep -F 'set -o noclobber; cat > "\${INCOMING}"' batch_ne.sh >/dev/null ||
  fail 'streamed launch scripts must use no-clobber creation for their incoming file'
grep -F '[[ ! -f "\${CONTROL_SCRIPT}" || -L "\${CONTROL_SCRIPT}" ]]' batch_ne.sh >/dev/null ||
  fail 'tmux prelaunch must reject a symlinked content-addressed control script'
grep -F 'Refusing to reuse pre-existing run-specific log directory' batch_ne.sh >/dev/null ||
  fail 'launch must require a fresh run-specific log directory before tmux creation'
grep -F 'flock -w ' "${TMP_DIR}/batch_default.out" | grep -F -- '-x 8' >/dev/null ||
  fail 'same-session tmux mutation must use a bounded exclusive launcher lock'
grep -F 'HOLOSOMA_STARTUP_READY token=%s launch_epoch=%s source_snapshot=%s phase=batch_preflight_complete' \
  batch_ne.sh >/dev/null || fail 'batch launch must emit a launch-bound post-preflight readiness marker'
grep -F "grep -Ec '^\\[INFO\\] final_train_command:'" batch_ne.sh >/dev/null ||
  fail 'batch startup handshake must require the terminal launcher torchrun-boundary signal'
grep -F "grep -Ec '^\\[INFO\\] cross_rank_training_provenance_verified world_size=" \
  batch_ne.sh >/dev/null ||
  fail 'batch startup handshake must require an actual cross-rank provenance rendezvous'
grep -F "grep -Ec '^\\[INFO\\] final_worker_preflight_verified '" batch_ne.sh >/dev/null ||
  fail 'batch startup handshake must require final launch-bound worker readiness markers'
grep -F 'final_worker_preflight_verified ' \
  src/holosoma/holosoma/train_agent.py >/dev/null ||
  fail 'workers must publish rank-unique readiness after real env/algo/checkpoint setup'
grep -F 'wait_for_launch_startup "${launch_token}" "${launch_epoch}"' batch_ne.sh >/dev/null ||
  fail 'batch launch must not return after tmux creation without the bounded startup handshake'
grep -F 'timeout --foreground --signal=TERM --kill-after=5s' batch_ne.sh >/dev/null ||
  fail 'startup SSH probes must have a controller-side execution bound'
if rg -n '(^|[[:space:]])remote_run[[:space:]]' batch_ne.sh; then
  fail 'batch controller must not retain calls to the legacy unbounded remote_run helper'
fi
controller_io_lines=$(rg -n '^[[:space:]]*(ssh|scp)[[:space:]]+\$\{SSH_OPTS\}' batch_ne.sh || true)
if [[ $(wc -l <<<"${controller_io_lines}") -ne 4 ]] \
    || [[ $(grep -Ec 'ssh \$\{SSH_OPTS\} "\$\{node\}" "\$\{cmd\}"$' <<<"${controller_io_lines}") -ne 2 ]] \
    || [[ $(grep -Ec 'ssh \$\{SSH_OPTS\} "\$\{node\}" "\$\{wrapped_cmd\}"$' <<<"${controller_io_lines}") -ne 1 ]] \
    || [[ $(grep -Ec 'scp \$\{SSH_OPTS\} "\$\{local_path\}" "\$\{node\}:\$\{remote_path\}"$' <<<"${controller_io_lines}") -ne 1 ]]; then
  printf '%s\n' "${controller_io_lines}" >&2
  fail 'raw controller ssh/scp commands may appear only inside the four hard-timeout helpers'
fi
unbounded_flocks=$(rg -n '^[[:space:]]*(if ![[:space:]]+)?flock[[:space:]]' batch_ne.sh \
  | rg -v 'flock -w \$\(quote "\$\{LAUNCH_LOCK_TIMEOUT_SECONDS\}"\) -x [0-9]' || true)
if [[ -n "${unbounded_flocks}" ]]; then
  printf '%s\n' "${unbounded_flocks}" >&2
  fail 'every remote launcher flock must have the validated LAUNCH_LOCK_TIMEOUT_SECONDS bound'
fi
unset controller_io_lines unbounded_flocks

expect_failure \
  "${TMP_DIR}/multinode_restart_nontransactional.out" \
  'RESTART=1 is not a supported transactional operation' \
  "${BATCH_ENV[@]}" NODES='test-node-a test-node-b' NNODES=2 RESTART=1 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/single_node_restart_nontransactional.out" \
  'RESTART=1 is not a supported transactional operation' \
  "${BATCH_ENV[@]}" RESTART=1 bash batch_ne.sh launch
if rg -n '\[INFO\] Launching|source_snapshot_id=|\[DRY_RUN\].*ssh' \
  "${TMP_DIR}/single_node_restart_nontransactional.out" >/dev/null; then
  fail 'unsupported RESTART=1 must fail before snapshot or remote launch work'
fi

expect_failure \
  "${TMP_DIR}/invalid_restart.out" \
  'RESTART must be a boolean' \
  "${BATCH_ENV[@]}" RESTART=replace bash batch_ne.sh launch

for completion_path_case in logger training; do
  completion_path_output="${TMP_DIR}/invalid_completion_path_${completion_path_case}.out"
  if [[ "${completion_path_case}" == logger ]]; then
    expect_failure \
      "${completion_path_output}" \
      'LOGGER_BASE_DIR cannot contain whitespace or control characters' \
      "${BATCH_ENV[@]}" REMOTE_RUN_ROOT="${TMP_DIR}/run-root" \
      LOGGER_BASE_DIR="${TMP_DIR}/run-root/training logs" bash batch_ne.sh launch
  else
    expect_failure \
      "${completion_path_output}" \
      'TRAINING_NAME cannot contain whitespace or control characters' \
      "${BATCH_ENV[@]}" TRAINING_NAME='scientifically ambiguous name' bash batch_ne.sh launch
  fi
  if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' "${completion_path_output}" >/dev/null; then
    fail "invalid ${completion_path_case} completion-path field must fail before snapshot or remote work"
  fi
done
unset completion_path_case completion_path_output

expect_failure \
  "${TMP_DIR}/invalid_training_name_traversal.out" \
  'TRAINING_NAME must be a safe basename of 1-128 characters' \
  "${BATCH_ENV[@]}" TRAINING_NAME='run/../../../outside-logger-root' \
  bash batch_ne.sh launch
if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' \
  "${TMP_DIR}/invalid_training_name_traversal.out" >/dev/null; then
  fail 'traversing TRAINING_NAME must fail before snapshot or remote work'
fi

expect_failure \
  "${TMP_DIR}/invalid_hierarchical_bool.out" \
  'HOLOSOMA_HIERARCHICAL_GRAD_REDUCE must be a boolean' \
  "${BATCH_ENV[@]}" HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=maybe bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/invalid_hierarchical_small_collectives_bool.out" \
  'HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES must be a boolean' \
  "${BATCH_ENV[@]}" HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=maybe bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/invalid_rank_visible_bool.out" \
  'HOLOSOMA_RANK_VISIBLE_DEVICES must be a boolean' \
  "${BATCH_ENV[@]}" HOLOSOMA_RANK_VISIBLE_DEVICES=maybe bash batch_ne.sh launch
for invalid_tf32_flag in YES on true 2; do
  invalid_tf32_output="${TMP_DIR}/invalid_tf32_${invalid_tf32_flag}.out"
  expect_failure \
    "${invalid_tf32_output}" \
    'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE must be exactly 0 or 1' \
    "${BATCH_ENV[@]}" TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="${invalid_tf32_flag}" \
    bash batch_ne.sh launch
  if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' "${invalid_tf32_output}" >/dev/null; then
    fail 'invalid PyTorch TF32 pre-start flag must fail before snapshot construction or remote actions'
  fi
done
unset invalid_tf32_flag invalid_tf32_output
expect_failure \
  "${TMP_DIR}/standalone_actor_only_step.out" \
  'HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP must equal HOLOSOMA_DAGGER_SUPERVISED_ONLY' \
  "${BATCH_ENV[@]}" HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP=1 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/supervised_only_actor_step_mismatch.out" \
  'HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP must equal HOLOSOMA_DAGGER_SUPERVISED_ONLY' \
  "${BATCH_ENV[@]}" HOLOSOMA_DAGGER_SUPERVISED_ONLY=1 \
  HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP=0 bash batch_ne.sh launch
for invalid_supervised_bool_name in \
  HOLOSOMA_DAGGER_SUPERVISED_ONLY \
  HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP \
  HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD; do
  invalid_supervised_output="${TMP_DIR}/invalid_${invalid_supervised_bool_name}.out"
  expect_failure \
    "${invalid_supervised_output}" \
    "${invalid_supervised_bool_name} must be a boolean" \
    "${BATCH_ENV[@]}" "${invalid_supervised_bool_name}=maybe" bash batch_ne.sh launch
  if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' "${invalid_supervised_output}" >/dev/null; then
    fail "invalid ${invalid_supervised_bool_name} must fail before snapshot construction or remote actions"
  fi
done
unset invalid_supervised_bool_name invalid_supervised_output
for invalid_microbatch in -1 1.5 true; do
  invalid_microbatch_output="${TMP_DIR}/invalid_supervised_microbatch_${invalid_microbatch//[^[:alnum:]]/_}.out"
  expect_failure \
    "${invalid_microbatch_output}" \
    'HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH must be a canonical integer in [0, 2147483647]' \
    "${BATCH_ENV[@]}" HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH="${invalid_microbatch}" \
    bash batch_ne.sh launch
  if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' "${invalid_microbatch_output}" >/dev/null; then
    fail 'invalid supervised actor microbatch must fail before snapshot construction or remote actions'
  fi
done
unset invalid_microbatch invalid_microbatch_output
expect_failure \
  "${TMP_DIR}/zero_streamed_supervised_microbatch.out" \
  'HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD=1 requires a positive HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH' \
  "${BATCH_ENV[@]}" HOLOSOMA_DAGGER_SUPERVISED_ONLY=1 \
  HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD=1 HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH=0 \
  bash batch_ne.sh launch
if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' \
  "${TMP_DIR}/zero_streamed_supervised_microbatch.out" >/dev/null; then
  fail 'zero streamed supervised microbatch must fail before snapshot construction or remote actions'
fi

"${BATCH_ENV[@]}" HOLOSOMA_DAGGER_SUPERVISED_ONLY=0 \
  HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP=0 \
  HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH=8 \
  HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD=false \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_explicit_supervised_runtime.out"
grep -F 'export HOLOSOMA_DAGGER_SUPERVISED_ONLY=0' \
  "${TMP_DIR}/batch_explicit_supervised_runtime.out" >/dev/null
grep -F 'export HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP=0' \
  "${TMP_DIR}/batch_explicit_supervised_runtime.out" >/dev/null
grep -F 'export HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH=8' \
  "${TMP_DIR}/batch_explicit_supervised_runtime.out" >/dev/null
grep -F 'export HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD=0' \
  "${TMP_DIR}/batch_explicit_supervised_runtime.out" >/dev/null
expect_failure \
  "${TMP_DIR}/hierarchical_requires_nccl_digest.out" \
  'NCCL_LIB_SHA256 is required when TORCH_DIST_BACKEND=nccl or HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1' \
  "${BATCH_ENV[@]}" TORCH_DIST_BACKEND=gloo HOLOSOMA_GLOO_GRAD_REDUCE=0 \
  HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1 \
  bash batch_ne.sh launch

expect_failure \
  "${TMP_DIR}/hierarchical_single_node_topology.out" \
  'HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1 requires NPROC>1 and NNODES>1; got world_size=2, local_world_size=2, NNODES=1' \
  "${BATCH_ENV[@]}" NPROC=2 CUDA_VISIBLE_DEVICES=0,1 \
  NCCL_LIB_SHA256="${NCCL_LIB_SHA_SENTINEL}" HOLOSOMA_GLOO_GRAD_REDUCE=0 \
  HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1 bash batch_ne.sh launch
if rg -n '\[INFO\] Launching|\[DRY_RUN\].*ssh|source_snapshot_id=' \
  "${TMP_DIR}/hierarchical_single_node_topology.out" >/dev/null; then
  fail 'unsupported single-node hierarchy must fail before snapshot/remote launch work'
fi
expect_failure \
  "${TMP_DIR}/hierarchical_single_local_rank.out" \
  'HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1 requires NPROC>1 and NNODES>1; got world_size=2, local_world_size=1, NNODES=2' \
  "${BATCH_ENV[@]}" NODES='test-node-a test-node-b' NNODES=2 NPROC=1 \
  NCCL_LIB_SHA256="${NCCL_LIB_SHA_SENTINEL}" HOLOSOMA_GLOO_GRAD_REDUCE=0 \
  HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/hierarchical_conflicts_with_flat_gloo.out" \
  'HOLOSOMA_GLOO_GRAD_REDUCE and HOLOSOMA_HIERARCHICAL_GRAD_REDUCE are mutually exclusive' \
  "${BATCH_ENV[@]}" NCCL_LIB_SHA256="${NCCL_LIB_SHA_SENTINEL}" \
  HOLOSOMA_GLOO_GRAD_REDUCE=1 HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/hierarchical_cpu_leader_requires_hierarchy.out" \
  'HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER requires HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1' \
  "${BATCH_ENV[@]}" HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=0 \
  HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER=1 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/hierarchical_small_collectives_requires_gloo_small.out" \
  'HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=1 requires HOLOSOMA_GLOO_SMALL_COLLECTIVES=1 and HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1' \
  "${BATCH_ENV[@]}" NODES='test-node-a test-node-b' NNODES=2 NPROC=2 \
  CUDA_VISIBLE_DEVICES=0,1 NCCL_LIB_SHA256="${NCCL_LIB_SHA_SENTINEL}" \
  HOLOSOMA_GLOO_GRAD_REDUCE=0 HOLOSOMA_GLOO_SMALL_COLLECTIVES=0 \
  HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1 \
  HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=1 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/hierarchical_small_collectives_requires_hierarchy.out" \
  'HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=1 requires HOLOSOMA_GLOO_SMALL_COLLECTIVES=1 and HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1' \
  "${BATCH_ENV[@]}" HOLOSOMA_GLOO_SMALL_COLLECTIVES=1 \
  HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=0 \
  HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=1 bash batch_ne.sh launch

# Exercise the non-dry-run rollback boundary with local ssh/tmux fakes. An
# existing same-name session must survive both the RESTART=0 rejection and a
# failure before the launcher reaches tmux creation.
FAKE_BIN="${TMP_DIR}/rollback-fake-bin"
FAKE_TMUX_STATE_DIR="${TMP_DIR}/rollback-tmux-state"
ROLLBACK_REMOTE_ROOT="${TMP_DIR}/rollback-remote-root"
mkdir -p "${FAKE_BIN}" "${FAKE_TMUX_STATE_DIR}" "${ROLLBACK_REMOTE_ROOT}"
cat >"${FAKE_BIN}/ssh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
node="${@: -2:1}"
cmd="${@: -1}"
if [[ "${FAKE_SSH_FAIL_PREFLIGHT_NODE:-}" == "${node}" && "${cmd}" == *'.holosoma_snapshot/source_manifest.sha256'* ]]; then
  echo "[FAKE] forced preflight failure on ${node}" >&2
  exit 97
fi
if [[ "${FAKE_SSH_FAIL_NODE_HEALTH_NODE:-}" == "${node}" \
      && "${cmd}" == *'node_data_sidecar_health_check_ok'* ]]; then
  echo "[FAKE] forced node GPU/data health preflight failure on ${node}" >&2
  exit 96
fi
if [[ "${FAKE_SSH_FAIL_PRELAUNCH_IDLE_NODE:-}" == "${node}" \
      && "${cmd}" == *'selected_gpu_idle_preflight_ok'* \
      && "${cmd}" == *'PROBE_PHASE=pre-launch'* ]]; then
  echo "[FAKE] forced pre-launch selected-GPU idle failure on ${node}" >&2
  exit 92
fi
# Both selected-GPU checks are mandatory even when the optional health bundle
# is skipped.  Model an idle node without requiring this test host to provide
# nvidia-smi; the branch above still injects the post-intent pre-launch failure
# needed for transactional rollback coverage.
if [[ "${cmd}" == *'selected_gpu_idle_preflight_ok'* ]]; then
  echo "[INFO][${node}] selected_gpu_idle_preflight_ok fixture=True"
  exit 0
fi
if [[ "${FAKE_SSH_FAIL_IDENTITY_RECOVERY_NODE:-}" == "${node}" \
      && "${cmd}" == *'lifecycle identity-recovery lock'* ]]; then
  echo "[FAKE] forced launch identity-recovery failure on ${node}" >&2
  exit 95
fi
if [[ -n "${FAKE_SSH_DELAY_PUBLISH_GATE_DIR:-}" \
      && "${cmd}" == *'Launch token was durably cancelled before intent publication'* ]]; then
  mkdir -p "${FAKE_SSH_DELAY_PUBLISH_GATE_DIR}"
  : >"${FAKE_SSH_DELAY_PUBLISH_GATE_DIR}/reached"
  (
    set +e
    for ((gate_poll = 0; gate_poll < 4000; gate_poll++)); do
      [[ -f "${FAKE_SSH_DELAY_PUBLISH_GATE_DIR}/release" ]] && break
      sleep 0.005
    done
    if [[ ! -f "${FAKE_SSH_DELAY_PUBLISH_GATE_DIR}/release" ]]; then
      echo '[FAKE] delayed publish gate timed out'
      printf '98\n' >"${FAKE_SSH_DELAY_PUBLISH_GATE_DIR}/late.rc"
      exit 0
    fi
    FAKE_SSH_NODE="${node}" bash -c "${cmd}"
    late_rc=$?
    printf '%s\n' "${late_rc}" >"${FAKE_SSH_DELAY_PUBLISH_GATE_DIR}/late.rc"
  ) >"${FAKE_SSH_DELAY_PUBLISH_GATE_DIR}/late.out" 2>&1 </dev/null &
  echo "[FAKE] delayed publish shell detached before commit on ${node}" >&2
  exit 97
fi
if [[ -n "${FAKE_SSH_DELAY_RESERVATION_GATE_DIR:-}" \
      && "${cmd}" == *'reserved rendezvous ports session='* ]]; then
  mkdir -p "${FAKE_SSH_DELAY_RESERVATION_GATE_DIR}"
  : >"${FAKE_SSH_DELAY_RESERVATION_GATE_DIR}/reached"
  (
    set +e
    for ((gate_poll = 0; gate_poll < 4000; gate_poll++)); do
      [[ -f "${FAKE_SSH_DELAY_RESERVATION_GATE_DIR}/release" ]] && break
      sleep 0.005
    done
    if [[ ! -f "${FAKE_SSH_DELAY_RESERVATION_GATE_DIR}/release" ]]; then
      echo '[FAKE] delayed reservation gate timed out'
      printf '98\n' >"${FAKE_SSH_DELAY_RESERVATION_GATE_DIR}/late.rc"
      exit 0
    fi
    FAKE_SSH_NODE="${node}" bash -c "${cmd}"
    late_rc=$?
    printf '%s\n' "${late_rc}" >"${FAKE_SSH_DELAY_RESERVATION_GATE_DIR}/late.rc"
  ) >"${FAKE_SSH_DELAY_RESERVATION_GATE_DIR}/late.out" 2>&1 </dev/null &
  echo "[FAKE] delayed reservation shell detached before commit on ${node}" >&2
  exit 97
fi
if [[ "${FAKE_SSH_FAIL_AFTER_RESERVE_NODE:-}" == "${node}" \
      && "${cmd}" == *'reserved rendezvous ports session='* ]]; then
  set +e
  FAKE_SSH_NODE="${node}" bash -c "${cmd}"
  reserve_rc=$?
  set -e
  (( reserve_rc == 0 )) || exit "${reserve_rc}"
  echo "[FAKE] forced SSH failure after committed rendezvous reservation on ${node}" >&2
  exit 97
fi
if [[ "${FAKE_SSH_FAIL_AFTER_PUBLISH_INTENT_NODE:-}" == "${node}" \
      && "${cmd}" == *'exact launch intent already published for token='* ]]; then
  # Execute the complete remote transaction first, then lose only its success
  # reply.  This reproduces an SSH post-commit ambiguity rather than a
  # pre-publication failure.
  set +e
  FAKE_SSH_NODE="${node}" bash -c "${cmd}"
  publish_rc=$?
  set -e
  (( publish_rc == 0 )) || exit "${publish_rc}"
  echo "[FAKE] forced SSH failure after committed launch intent on ${node}" >&2
  exit 97
fi
if [[ -n "${FAKE_SSH_PUBLISH_GATE_DIR:-}" \
      && "${cmd}" == *'Refusing concurrent launch intent:'* ]]; then
  mkdir -p "${FAKE_SSH_PUBLISH_GATE_DIR}"
  : >"${FAKE_SSH_PUBLISH_GATE_DIR}/reached"
  for ((gate_poll = 0; gate_poll < 4000; gate_poll++)); do
    [[ -f "${FAKE_SSH_PUBLISH_GATE_DIR}/release" ]] && break
    sleep 0.025
  done
  [[ -f "${FAKE_SSH_PUBLISH_GATE_DIR}/release" ]] || {
    echo '[FAKE] timed out waiting for publish gate release' >&2
    exit 98
  }
fi
if [[ "${FAKE_SSH_REPLACE_CONTROL_WITH_SYMLINK:-0}" == 1 \
      && "${cmd}" == *'installed_verified_launch_script='* ]]; then
  FAKE_SSH_NODE="${node}" bash -c "${cmd}"
  control_script=$(sed -n 's/^FINAL=//p' <<<"${cmd}" | head -1)
  if [[ ! "${control_script}" =~ ^${REMOTE_RUN_ROOT}/src-[0-9a-f]{64}/\.run_control/train-[0-9a-f]{64}\.sh$ \
        || ! -f "${control_script}" || -L "${control_script}" ]]; then
    echo '[FAKE] could not identify the installed regular control script' >&2
    exit 94
  fi
  mv -T "${control_script}" "${control_script}.symlink-target"
  ln -s "$(basename "${control_script}.symlink-target")" "${control_script}"
  echo "[FAKE] replaced installed control script with symlink: ${control_script}" >&2
  exit 0
fi
if [[ "${FAKE_SSH_HANG_ROLLBACK:-0}" == 1 \
      && ( -z "${FAKE_SSH_HANG_ROLLBACK_NODE:-}" \
        || "${FAKE_SSH_HANG_ROLLBACK_NODE}" == "${node}" ) \
      && "${cmd}" == *'lifecycle rollback lock; owned session cleanup is unconfirmed'* ]]; then
  sleep 60
fi
if [[ -n "${FAKE_SSH_REQUIRE_MUTATION_WRAPPER_SESSION:-}" \
      && "${cmd}" == *"${FAKE_SSH_REQUIRE_MUTATION_WRAPPER_SESSION}"* \
      && "${cmd}" == *'lifecycle rollback lock; owned session cleanup is unconfirmed'* ]]; then
  : "${FAKE_SSH_MUTATION_WRAPPER_MARKER:?}"
  if [[ ! "${cmd}" =~ ^timeout[[:space:]]+--signal=TERM[[:space:]]+--kill-after=[1-9][0-9]*s[[:space:]]+[1-9][0-9]*s[[:space:]]+bash[[:space:]]+-c[[:space:]] ]]; then
    echo '[FAKE] lifecycle rollback mutation lacks its remote timeout wrapper' >&2
    exit 92
  fi
  : >"${FAKE_SSH_MUTATION_WRAPPER_MARKER}"
fi
FAKE_SSH_NODE="${node}" bash -c "${cmd}"
EOF
cat >"${FAKE_BIN}/tmux" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
: "${FAKE_TMUX_STATE_DIR:?}"
: "${FAKE_SSH_NODE:?}"
action="${1:?}"
shift
session=''
previous=''
for arg in "$@"; do
  if [[ "${previous}" == '-t' || "${previous}" == '-s' ]]; then
    session="${arg}"
    break
  fi
  previous="${arg}"
done
[[ -n "${session}" ]] || { echo '[FAKE] missing tmux session argument' >&2; exit 2; }
state_prefix="${FAKE_TMUX_STATE_DIR}/${FAKE_SSH_NODE}.${session}"
case "${action}" in
  has-session)
    if [[ -n "${FAKE_TMUX_HAS_SESSION_RC:-}" ]]; then
      exit "${FAKE_TMUX_HAS_SESSION_RC}"
    fi
    [[ -f "${state_prefix}.session" ]]
    ;;
  new-session)
    if [[ -e /proc/$$/fd/8 ]]; then
      echo '[FAKE] launcher lock fd 8 leaked into tmux new-session' >&2
      exit 86
    fi
    [[ ! -f "${state_prefix}.session" ]] || exit 1
    : >"${state_prefix}.session"
    new_session_args=("$@")
    for ((arg_index = 0; arg_index < ${#new_session_args[@]}; arg_index++)); do
      [[ "${new_session_args[arg_index]}" == -e ]] || continue
      (( arg_index + 1 < ${#new_session_args[@]} )) || {
        echo '[FAKE] tmux new-session -e lacks an environment assignment' >&2
        exit 2
      }
      environment_assignment=${new_session_args[arg_index + 1]}
      environment_name=${environment_assignment%%=*}
      environment_value=${environment_assignment#*=}
      case "${environment_name}" in
        HOLOSOMA_LAUNCH_TOKEN) environment_file=env_token ;;
        HOLOSOMA_COMMAND_SHA256) environment_file=env_command_sha256 ;;
        HOLOSOMA_LAUNCH_EPOCH) environment_file=env_launch_epoch ;;
        *) echo "[FAKE] unsupported tmux session environment: ${environment_name}" >&2; exit 2 ;;
      esac
      printf '%s\n' "${environment_value}" >"${state_prefix}.${environment_file}"
      arg_index=$((arg_index + 1))
    done
    case "${FAKE_TMUX_POST_CREATE_FAILURE:-}" in
      '')
        ;;
      exact)
        echo '[FAKE] forced SSH-visible failure after atomic tmux creation' >&2
        exit 97
        ;;
      conflicting-option)
        printf '%s\n' conflicting-other-token >"${state_prefix}.token"
        echo '[FAKE] forced post-create failure with a conflicting existing ownership option' >&2
        exit 97
        ;;
      legacy-options-only)
        cp "${state_prefix}.env_token" "${state_prefix}.token"
        cp "${state_prefix}.env_command_sha256" "${state_prefix}.command_sha256"
        cp "${state_prefix}.env_launch_epoch" "${state_prefix}.launch_epoch"
        rm -f "${state_prefix}.env_token" "${state_prefix}.env_command_sha256" \
          "${state_prefix}.env_launch_epoch"
        echo '[FAKE] forced post-create failure with exact legacy ownership options only' >&2
        exit 97
        ;;
      partial-environment)
        cp "${state_prefix}.env_token" "${state_prefix}.token"
        cp "${state_prefix}.env_command_sha256" "${state_prefix}.command_sha256"
        cp "${state_prefix}.env_launch_epoch" "${state_prefix}.launch_epoch"
        rm -f "${state_prefix}.env_command_sha256" "${state_prefix}.env_launch_epoch"
        echo '[FAKE] forced post-create failure with a partial atomic environment' >&2
        exit 97
        ;;
      empty-environment)
        cp "${state_prefix}.env_token" "${state_prefix}.token"
        cp "${state_prefix}.env_command_sha256" "${state_prefix}.command_sha256"
        cp "${state_prefix}.env_launch_epoch" "${state_prefix}.launch_epoch"
        : >"${state_prefix}.env_token"
        : >"${state_prefix}.env_command_sha256"
        : >"${state_prefix}.env_launch_epoch"
        echo '[FAKE] forced post-create failure with explicitly empty atomic environment fields' >&2
        exit 97
        ;;
      mismatched-environment)
        cp "${state_prefix}.env_token" "${state_prefix}.token"
        cp "${state_prefix}.env_command_sha256" "${state_prefix}.command_sha256"
        cp "${state_prefix}.env_launch_epoch" "${state_prefix}.launch_epoch"
        printf '%s\n' mismatched-atomic-token >"${state_prefix}.env_token"
        echo '[FAKE] forced post-create failure with a mismatched atomic environment' >&2
        exit 97
        ;;
      empty-option)
        cp "${state_prefix}.env_token" "${state_prefix}.token"
        cp "${state_prefix}.env_command_sha256" "${state_prefix}.command_sha256"
        cp "${state_prefix}.env_launch_epoch" "${state_prefix}.launch_epoch"
        : >"${state_prefix}.token"
        echo '[FAKE] forced post-create failure with an explicitly empty ownership option' >&2
        exit 97
        ;;
      removed-environment)
        cp "${state_prefix}.env_token" "${state_prefix}.token"
        cp "${state_prefix}.env_command_sha256" "${state_prefix}.command_sha256"
        cp "${state_prefix}.env_launch_epoch" "${state_prefix}.launch_epoch"
        rm -f "${state_prefix}.env_token" "${state_prefix}.env_command_sha256" \
          "${state_prefix}.env_launch_epoch"
        : >"${state_prefix}.env_token_removed"
        : >"${state_prefix}.env_command_sha256_removed"
        : >"${state_prefix}.env_launch_epoch_removed"
        echo '[FAKE] forced post-create failure with explicitly removed atomic environment fields' >&2
        exit 97
        ;;
      identity-read-failure)
        cp "${state_prefix}.env_token" "${state_prefix}.token"
        cp "${state_prefix}.env_command_sha256" "${state_prefix}.command_sha256"
        cp "${state_prefix}.env_launch_epoch" "${state_prefix}.launch_epoch"
        : >"${state_prefix}.identity_read_failure"
        echo '[FAKE] forced post-create failure with tmux identity reads unavailable' >&2
        exit 97
        ;;
      *)
        echo "[FAKE] unsupported post-create failure mode: ${FAKE_TMUX_POST_CREATE_FAILURE}" >&2
        exit 2
        ;;
    esac
    ;;
  set-option)
    [[ -f "${state_prefix}.session" ]] || exit 1
    option_name="${@: -2:1}"
    option_value="${@: -1}"
    case "${option_name}" in
      @holosoma_launch_token) option_file=token ;;
      @holosoma_command_sha256) option_file=command_sha256 ;;
      @holosoma_launch_epoch) option_file=launch_epoch ;;
      *) echo "[FAKE] unsupported tmux option: ${option_name}" >&2; exit 2 ;;
    esac
    printf '%s\n' "${option_value}" >"${state_prefix}.${option_file}"
    if [[ "${option_name}" == @holosoma_launch_epoch ]]; then
      session_sha=$(printf '%s' "${session}" | sha256sum | awk '{print $1}')
      active_state="${REMOTE_RUN_ROOT}/.active/${session_sha}_${FAKE_SSH_NODE}.state"
      if [[ -f "${active_state}" ]]; then
        IFS=$'\t' read -r _version _phase snapshot log_dir _target token _command epoch <"${active_state}" || true
        active_log=$(find "${REMOTE_RUN_ROOT}/${snapshot}/${log_dir}" -maxdepth 1 \
          -type f -name "node_*_${FAKE_SSH_NODE}.log" -print -quit 2>/dev/null || true)
        if [[ -n "${active_log}" ]]; then
          case "${FAKE_TMUX_STARTUP_MODE:-ready}" in
            ready|boundary-only|missing-worker|missing-provenance|progressive-worker|malformed-worker|duplicate-worker|mismatched-ready|mismatched-provenance|fatal-long-ready)
              printf 'HOLOSOMA_STARTUP_READY token=%s launch_epoch=%s source_snapshot=%s phase=batch_preflight_complete\n' \
                "${token}" "${epoch}" "${snapshot}" >>"${active_log}"
              ;;
          esac
          if [[ "${FAKE_TMUX_STARTUP_MODE:-ready}" == mismatched-ready ]]; then
            printf 'HOLOSOMA_STARTUP_READY token=wrong-launch-token launch_epoch=%s source_snapshot=%s phase=batch_preflight_complete\n' \
              "${epoch}" "${snapshot}" >>"${active_log}"
          fi
          if [[ "${FAKE_TMUX_STARTUP_MODE:-ready}" == ready \
                || "${FAKE_TMUX_STARTUP_MODE:-ready}" == boundary-only \
                || "${FAKE_TMUX_STARTUP_MODE:-ready}" == missing-worker \
                || "${FAKE_TMUX_STARTUP_MODE:-ready}" == missing-provenance \
                || "${FAKE_TMUX_STARTUP_MODE:-ready}" == progressive-worker \
                || "${FAKE_TMUX_STARTUP_MODE:-ready}" == malformed-worker \
                || "${FAKE_TMUX_STARTUP_MODE:-ready}" == duplicate-worker \
                || "${FAKE_TMUX_STARTUP_MODE:-ready}" == mismatched-ready \
                || "${FAKE_TMUX_STARTUP_MODE:-ready}" == mismatched-provenance \
                || "${FAKE_TMUX_STARTUP_MODE:-ready}" == fatal-long-ready ]]; then
            printf '[INFO] final_train_command: fake torch.distributed.run\n' >>"${active_log}"
          fi
          if [[ "${FAKE_TMUX_STARTUP_MODE:-ready}" == ready \
                || "${FAKE_TMUX_STARTUP_MODE:-ready}" == missing-worker \
                || "${FAKE_TMUX_STARTUP_MODE:-ready}" == missing-provenance \
                || "${FAKE_TMUX_STARTUP_MODE:-ready}" == progressive-worker \
                || "${FAKE_TMUX_STARTUP_MODE:-ready}" == malformed-worker \
                || "${FAKE_TMUX_STARTUP_MODE:-ready}" == duplicate-worker \
                || "${FAKE_TMUX_STARTUP_MODE:-ready}" == mismatched-ready \
                || "${FAKE_TMUX_STARTUP_MODE:-ready}" == mismatched-provenance \
                || "${FAKE_TMUX_STARTUP_MODE:-ready}" == fatal-long-ready ]]; then
            log_name=${active_log##*/}
            node_rank=${log_name#node_}
            node_rank=${node_rank%%_*}
            [[ "${node_rank}" =~ ^[0-9]+$ ]] || { echo '[FAKE] invalid node rank in active log' >&2; exit 2; }
            provenance_count=${NPROC}
            if [[ "${FAKE_TMUX_STARTUP_MODE:-ready}" == missing-provenance ]]; then
              provenance_count=$((NPROC - 1))
            fi
            for ((marker_rank = 0; marker_rank < provenance_count; marker_rank++)); do
              printf '[INFO] cross_rank_training_provenance_verified world_size=%s training_regime=fake\n' \
                "$((NPROC * NNODES))" >>"${active_log}"
            done
            if [[ "${FAKE_TMUX_STARTUP_MODE:-ready}" == mismatched-provenance ]]; then
              printf '[INFO] cross_rank_training_provenance_verified world_size=%s training_regime=wrong-launch\n' \
                "$((NPROC * NNODES + 1))" >>"${active_log}"
            fi
            worker_count=${NPROC}
            if [[ "${FAKE_TMUX_STARTUP_MODE:-ready}" == missing-worker ]]; then
              worker_count=$((NPROC - 1))
            elif [[ "${FAKE_TMUX_STARTUP_MODE:-ready}" == progressive-worker \
                  || "${FAKE_TMUX_STARTUP_MODE:-ready}" == malformed-worker \
                  || "${FAKE_TMUX_STARTUP_MODE:-ready}" == duplicate-worker ]]; then
              worker_count=$((NPROC - 1))
            fi
            for ((local_rank = 0; local_rank < worker_count; local_rank++)); do
              global_rank=$((node_rank * NPROC + local_rank))
              printf '[INFO] final_worker_preflight_verified global_rank=%s local_rank=%s world_size=%s source_snapshot=%s launch_token=%s launch_epoch=%s\n' \
                "${global_rank}" "${local_rank}" "$((NPROC * NNODES))" \
                "${snapshot}" "${token}" "${epoch}" >>"${active_log}"
            done
            if [[ "${FAKE_TMUX_STARTUP_MODE:-ready}" == progressive-worker ]]; then
              local_rank=$((NPROC - 1))
              global_rank=$((node_rank * NPROC + local_rank))
              # Leave an exact worker record temporarily incomplete. GNU grep
              # counts this prefix as a marker even though -Fxc cannot yet
              # validate it, reproducing a probe against a live tee append.
              printf '[INFO] final_worker_preflight_verified global_rank=%s local_rank=%s' \
                "${global_rank}" "${local_rank}" >>"${active_log}"
              (
                sleep "${FAKE_TMUX_PROGRESSIVE_DELAY_SECONDS:-2}"
                printf ' world_size=%s source_snapshot=%s launch_token=%s launch_epoch=%s\n' \
                  "$((NPROC * NNODES))" "${snapshot}" "${token}" "${epoch}" \
                  >>"${active_log}"
              ) >/dev/null 2>&1 &
            elif [[ "${FAKE_TMUX_STARTUP_MODE:-ready}" == malformed-worker ]]; then
              printf '[INFO] final_worker_preflight_verified malformed_launch_identity=true\n' \
                >>"${active_log}"
            elif [[ "${FAKE_TMUX_STARTUP_MODE:-ready}" == duplicate-worker ]]; then
              global_rank=$((node_rank * NPROC))
              printf '[INFO] final_worker_preflight_verified global_rank=%s local_rank=0 world_size=%s source_snapshot=%s launch_token=%s launch_epoch=%s\n' \
                "${global_rank}" "$((NPROC * NNODES))" \
                "${snapshot}" "${token}" "${epoch}" >>"${active_log}"
            fi
          fi
          if [[ "${FAKE_TMUX_STARTUP_MODE:-ready}" == fatal-long-ready ]]; then
            printf 'Traceback (most recent call last): fake startup failure before long buffered output\n' >>"${active_log}"
            for ((padding_line = 0; padding_line < 1100; padding_line++)); do
              printf 'fatal-padding-%04d %4096s\n' "${padding_line}" x >>"${active_log}"
            done
          elif [[ "${FAKE_TMUX_STARTUP_MODE:-ready}" == fatal ]]; then
            printf 'Traceback (most recent call last): fake startup failure\n' >>"${active_log}"
          fi
        fi
      fi
    fi
    ;;
  show-options)
    [[ -f "${state_prefix}.session" ]] || exit 1
    [[ ! -f "${state_prefix}.identity_read_failure" ]] || exit 99
    for option_spec in \
      '@holosoma_launch_token:token' \
      '@holosoma_command_sha256:command_sha256' \
      '@holosoma_launch_epoch:launch_epoch'; do
      option_name=${option_spec%%:*}
      option_file=${option_spec#*:}
      [[ -f "${state_prefix}.${option_file}" ]] || continue
      option_value=$(cat "${state_prefix}.${option_file}")
      if [[ -n "${option_value}" ]]; then
        printf '%s %s\n' "${option_name}" "${option_value}"
      else
        printf "%s ''\n" "${option_name}"
      fi
    done
    ;;
  show-environment)
    [[ -f "${state_prefix}.session" ]] || exit 1
    [[ ! -f "${state_prefix}.identity_read_failure" ]] || exit 99
    for environment_spec in \
      'HOLOSOMA_LAUNCH_TOKEN:env_token' \
      'HOLOSOMA_COMMAND_SHA256:env_command_sha256' \
      'HOLOSOMA_LAUNCH_EPOCH:env_launch_epoch'; do
      environment_name=${environment_spec%%:*}
      environment_file=${environment_spec#*:}
      if [[ -f "${state_prefix}.${environment_file}_removed" ]]; then
        printf -- '-%s\n' "${environment_name}"
      elif [[ -f "${state_prefix}.${environment_file}" ]]; then
        printf '%s=%s\n' "${environment_name}" "$(cat "${state_prefix}.${environment_file}")"
      fi
    done
    ;;
  list-panes)
    [[ -f "${state_prefix}.session" ]] || exit 1
    [[ -s "${state_prefix}.pane_pid" ]] || {
      echo '[FAKE] pane PID is unavailable' >&2
      exit 2
    }
    printf '%s\t%s\t0\n' "${session}" "$(cat "${state_prefix}.pane_pid")"
    if [[ -s "${state_prefix}.extra_pane_pid" ]]; then
      printf '%s\t%s\t0\n' "${session}" "$(cat "${state_prefix}.extra_pane_pid")"
    fi
    ;;
  display-message)
    [[ -f "${state_prefix}.session" ]]
    if [[ "${FAKE_TMUX_STARTUP_IDENTITY_MODE:-}" == removed-environment ]]; then
      # The launcher has already completed its post-create exact identity
      # check.  Remove all atomic fields at the last controller-side tmux
      # probe so only the startup-health handshake can catch the drift.
      rm -f "${state_prefix}.env_token" "${state_prefix}.env_command_sha256" \
        "${state_prefix}.env_launch_epoch"
    fi
    ;;
  kill-session)
    [[ -f "${state_prefix}.session" ]] || exit 1
    if [[ -n "${FAKE_TMUX_KILL_SESSION_RC:-}" ]]; then
      echo '[FAKE] forced legacy kill-session failure' >&2
      exit "${FAKE_TMUX_KILL_SESSION_RC}"
    fi
    if [[ "${FAKE_TMUX_HANG_KILL_SESSION:-}" == "${session}" ]]; then
      : "${FAKE_TMUX_HANG_MARKER_PREFIX:?}"
      : >"${FAKE_TMUX_HANG_MARKER_PREFIX}.entered"
      printf '%s\n' "$$" >"${FAKE_TMUX_HANG_MARKER_PREFIX}.shell-pid"
      sleep 12 &
      fake_hang_sleep_pid=$!
      printf '%s\n' "${fake_hang_sleep_pid}" \
        >"${FAKE_TMUX_HANG_MARKER_PREFIX}.sleep-pid"
      wait "${fake_hang_sleep_pid}"
      : >"${FAKE_TMUX_HANG_MARKER_PREFIX}.late"
    fi
    rm -f "${state_prefix}.session" "${state_prefix}.token" \
      "${state_prefix}.command_sha256" "${state_prefix}.launch_epoch" \
      "${state_prefix}.env_token" "${state_prefix}.env_command_sha256" \
      "${state_prefix}.env_launch_epoch" "${state_prefix}.env_token_removed" \
      "${state_prefix}.env_command_sha256_removed" \
      "${state_prefix}.env_launch_epoch_removed" "${state_prefix}.identity_read_failure" \
      "${state_prefix}.pane_pid" "${state_prefix}.extra_pane_pid"
    printf '%s %s\n' "${FAKE_SSH_NODE}" "${session}" >>"${FAKE_TMUX_STATE_DIR}/kills.log"
    ;;
  *)
    echo "[FAKE] unsupported tmux action: ${action}" >&2
    exit 2
    ;;
esac
EOF
cat >"${FAKE_BIN}/ss" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${FAKE_BUSY_PORT:-}" && "$*" == *":${FAKE_BUSY_PORT}"* ]]; then
  printf 'LISTEN 0 128 0.0.0.0:%s 0.0.0.0:*\n' "${FAKE_BUSY_PORT}"
fi
EOF
cat >"${FAKE_BIN}/mktemp" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
path=$(/usr/bin/mktemp "$@")
if [[ "${FAKE_CORRUPT_LAUNCH_RESULT:-0}" == 1 \
      && "$*" == *'holosoma-launch-result.XXXXXX'* ]]; then
  (
    for ((poll = 0; poll < 4000; poll++)); do
      if [[ -s "${path}" ]]; then
        printf '%s\n' corrupt-result-without-command-identity >"${path}"
        exit 0
      fi
      sleep 0.001
    done
    exit 1
  ) >/dev/null 2>&1 &
fi
printf '%s\n' "${path}"
EOF
cat >"${FAKE_BIN}/date" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${FAKE_DATE_SECONDS:-}" && "$#" == 1 && "$1" == +%s ]]; then
  printf '%s\n' "${FAKE_DATE_SECONDS}"
else
  exec /usr/bin/date "$@"
fi
EOF
cat >"${FAKE_BIN}/od" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${FAKE_OD_OUTPUT+x}" ]]; then
  printf '%s\n' "${FAKE_OD_OUTPUT}"
else
  exec /usr/bin/od "$@"
fi
EOF
chmod +x "${FAKE_BIN}/ssh" "${FAKE_BIN}/tmux" "${FAKE_BIN}/ss" \
  "${FAKE_BIN}/mktemp" "${FAKE_BIN}/date" "${FAKE_BIN}/od"

rollback_snapshot_id=$(sed -nE 's/.*source_snapshot_id=(src-[0-9a-f]{64}).*/\1/p' \
  "${TMP_DIR}/batch_default.out" | head -1)
rollback_snapshot_archive=$(sed -nE 's/.*source_snapshot_archive=([^ ]+) archive_sha256=.*/\1/p' \
  "${TMP_DIR}/batch_default.out" | head -1)
rollback_snapshot_archive_sha=$(sha256sum "${rollback_snapshot_archive}" | awk '{print $1}')
rollback_snapshot_manifest_sha=${rollback_snapshot_id#src-}
mkdir -p "${ROLLBACK_REMOTE_ROOT}/${rollback_snapshot_id}"
tar -xzf "${rollback_snapshot_archive}" \
  -C "${ROLLBACK_REMOTE_ROOT}/${rollback_snapshot_id}" --same-permissions
rollback_snapshot_root="${ROLLBACK_REMOTE_ROOT}/${rollback_snapshot_id}"
rollback_asset_repo="${TMP_DIR}/rollback-asset-repo"
rollback_source_bank_name=launcher_contract_single_box_28
rollback_reference_bank="${REPO_ROOT}/data/ds_as_data/as_realmesh67000_finalpos_convexsurface51_convexhull"
rollback_source_bank="${rollback_asset_repo}/data/ds_as_data/${rollback_source_bank_name}"
while IFS=$'\t' read -r _rollback_link_path rollback_asset_path; do
  [[ -n "${rollback_asset_path}" ]] || continue
  mkdir -p -- "${rollback_asset_repo}/${rollback_asset_path}"
done <"${rollback_snapshot_root}/.holosoma_snapshot/asset_links.tsv"
mkdir -p -- \
  "${rollback_source_bank}/contact_export_from_teacher_success133_final0p5/clips"
cp -- "${rollback_reference_bank}/box_28.npz" \
  "${rollback_source_bank}/box_28.npz"
cp -a -- \
  "${rollback_reference_bank}/contact_export_from_teacher_success133_final0p5/clips/0002_box_28" \
  "${rollback_source_bank}/contact_export_from_teacher_success133_final0p5/clips/0002_box_28"
"${PYTHON_BIN:-python3}" - \
    "${rollback_reference_bank}/_clip_object_urdf_map.json" \
    "${rollback_source_bank}/_clip_object_urdf_map.json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
payload = json.loads(source_path.read_text(encoding="utf-8"))
clips = payload.get("clips")
if not isinstance(clips, dict) or "box_28" not in clips:
    raise SystemExit("[FAIL] lifecycle fixture source map does not contain box_28")
payload["clips"] = {"box_28": clips["box_28"]}
target_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
declare -A rollback_opened_source_modes=()
open_rollback_snapshot_source_dir() {
  local directory="$1"
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    fail "rollback snapshot install parent is missing or symlinked: ${directory}"
  fi
  if [[ -z "${rollback_opened_source_modes[${directory}]+x}" ]]; then
    rollback_opened_source_modes["${directory}"]=$(stat -c '%a' -- "${directory}")
    chmod u+w "${directory}"
  fi
}
while IFS=$'\t' read -r rollback_link_path rollback_asset_path; do
  [[ -n "${rollback_link_path}" ]] || continue
  rollback_link_parent=$(dirname -- "${rollback_snapshot_root}/${rollback_link_path}")
  open_rollback_snapshot_source_dir "${rollback_link_parent}"
  ln -s "${rollback_asset_repo}/${rollback_asset_path}" \
    "${rollback_snapshot_root}/${rollback_link_path}"
done <"${rollback_snapshot_root}/.holosoma_snapshot/asset_links.tsv"
open_rollback_snapshot_source_dir "${rollback_snapshot_root}"
mkdir -p \
  "${rollback_snapshot_root}/.checkpoint_cache" \
  "${rollback_snapshot_root}/.teacher_checkpoints" \
  "${rollback_snapshot_root}/.run_control" \
  "${rollback_snapshot_root}/logs/batch_ne"
chmod 700 \
  "${rollback_snapshot_root}/.checkpoint_cache" \
  "${rollback_snapshot_root}/.teacher_checkpoints" \
  "${rollback_snapshot_root}/.run_control" \
  "${rollback_snapshot_root}/logs" \
  "${rollback_snapshot_root}/logs/batch_ne"
for rollback_opened_dir in "${!rollback_opened_source_modes[@]}"; do
  chmod "${rollback_opened_source_modes[${rollback_opened_dir}]}" \
    "${rollback_opened_dir}"
done
unset rollback_opened_source_modes rollback_opened_dir rollback_link_parent
unset -f open_rollback_snapshot_source_dir

ROLLBACK_ENV=(
  env
  PATH="${FAKE_BIN}:${PATH}"
  FAKE_TMUX_STATE_DIR="${FAKE_TMUX_STATE_DIR}"
  MOTION_GENERATOR_TEACHER_EXPECTED_SHA256="$(printf 'a%.0s' {1..64})"
  NPROC=1
  NNODES=1
  PER_GPU_ENVS=1024
  DRY_RUN=0
  SSH_OPTS=
  SKIP_NODE_HEALTH_CHECK=1
  SKIP_GIT_PULL=1
  REMOTE_REPO="${rollback_asset_repo}"
  REMOTE_RUN_ROOT="${ROLLBACK_REMOTE_ROOT}"
  LOGGER_BASE_DIR="${ROLLBACK_REMOTE_ROOT}/training_logs"
  CH_BANK_NAME="${rollback_source_bank_name}"
  CORL_SOLID80_BANK_NAME="${rollback_source_bank_name}"
  OMOMO_EXPECTED_TOTAL=1
  RESUME_FROM_BOX_EXPECTED_TOTAL=1
  SOURCE_SNAPSHOT_CACHE="${TMP_DIR}/snapshot-cache"
  SOURCE_SNAPSHOT_ID="${rollback_snapshot_id}"
  SOURCE_SNAPSHOT_ARCHIVE="${rollback_snapshot_archive}"
  SOURCE_SNAPSHOT_ARCHIVE_SHA256="${rollback_snapshot_archive_sha}"
  SOURCE_MANIFEST_SHA256="${rollback_snapshot_manifest_sha}"
  SESSION=rollback-contract
  RUN_STAMP=rollback-contract
  LAUNCH_STARTUP_TIMEOUT_SECONDS=5
  LAUNCH_STARTUP_PROBE_TIMEOUT_SECONDS=2
  LAUNCH_STARTUP_POLL_SECONDS=1
  LAUNCH_STARTUP_STABILITY_SECONDS=1
)

snapshot_scripts_dir="${ROLLBACK_REMOTE_ROOT}/${rollback_snapshot_id}/scripts"
snapshot_scripts_sealed_mode=$(stat -c '%a' "${snapshot_scripts_dir}")
chmod u+w "${snapshot_scripts_dir}"
printf '%s\n' 'injected = True' >"${snapshot_scripts_dir}/unmanifested_injection.py"
chmod "${snapshot_scripts_sealed_mode}" "${snapshot_scripts_dir}"
expect_failure \
  "${TMP_DIR}/snapshot_extra_executable.out" \
  'Installed snapshot executable file closure changed after prepare.' \
  "${ROLLBACK_ENV[@]}" NODES=closure-node bash batch_ne.sh launch
chmod u+w "${snapshot_scripts_dir}"
rm -f "${snapshot_scripts_dir}/unmanifested_injection.py"
chmod "${snapshot_scripts_sealed_mode}" "${snapshot_scripts_dir}"

snapshot_mode_target="${ROLLBACK_REMOTE_ROOT}/${rollback_snapshot_id}/scripts/build_run_snapshot.sh"
snapshot_original_mode=$(stat -c '%a' "${snapshot_mode_target}")
if [[ -x "${snapshot_mode_target}" ]]; then
  chmod u-x "${snapshot_mode_target}"
else
  chmod u+x "${snapshot_mode_target}"
fi
expect_failure \
  "${TMP_DIR}/snapshot_mode_drift.out" \
  'Installed snapshot source mode closure changed after prepare.' \
  "${ROLLBACK_ENV[@]}" NODES=closure-node bash batch_ne.sh launch
chmod "${snapshot_original_mode}" "${snapshot_mode_target}"

snapshot_mode_dir="${ROLLBACK_REMOTE_ROOT}/${rollback_snapshot_id}/scripts"
snapshot_original_dir_mode=$(stat -c '%a' "${snapshot_mode_dir}")
if [[ "$(stat -c '%A' "${snapshot_mode_dir}")" == ?????w???? ]]; then
  chmod g-w "${snapshot_mode_dir}"
else
  chmod g+w "${snapshot_mode_dir}"
fi
expect_failure \
  "${TMP_DIR}/snapshot_directory_mode_drift.out" \
  'Installed snapshot source mode closure changed after prepare.' \
  "${ROLLBACK_ENV[@]}" NODES=closure-node bash batch_ne.sh launch
chmod "${snapshot_original_dir_mode}" "${snapshot_mode_dir}"

snapshot_original_root_mode=$(stat -c '%a' "${rollback_snapshot_root}")
chmod u+w "${rollback_snapshot_root}"
expect_failure \
  "${TMP_DIR}/snapshot_root_mode_drift.out" \
  'Installed snapshot source mode closure changed after prepare.' \
  "${ROLLBACK_ENV[@]}" NODES=closure-node bash batch_ne.sh launch
chmod "${snapshot_original_root_mode}" "${rollback_snapshot_root}"

snapshot_metadata_dir="${rollback_snapshot_root}/.holosoma_snapshot"
snapshot_original_metadata_dir_mode=$(stat -c '%a' "${snapshot_metadata_dir}")
chmod u+w "${snapshot_metadata_dir}"
expect_failure \
  "${TMP_DIR}/snapshot_metadata_dir_mode_drift.out" \
  'Installed snapshot source mode closure changed after prepare.' \
  "${ROLLBACK_ENV[@]}" NODES=closure-node bash batch_ne.sh launch
chmod "${snapshot_original_metadata_dir_mode}" "${snapshot_metadata_dir}"

snapshot_metadata_file="${snapshot_metadata_dir}/id"
snapshot_original_metadata_file_mode=$(stat -c '%a' "${snapshot_metadata_file}")
chmod u+w "${snapshot_metadata_file}"
expect_failure \
  "${TMP_DIR}/snapshot_metadata_file_mode_drift.out" \
  'Installed snapshot metadata file is not sealed 0444 after prepare:' \
  "${ROLLBACK_ENV[@]}" NODES=closure-node bash batch_ne.sh launch
chmod "${snapshot_original_metadata_file_mode}" "${snapshot_metadata_file}"

snapshot_runtime_dir="${rollback_snapshot_root}/.checkpoint_cache"
snapshot_original_runtime_dir_mode=$(stat -c '%a' "${snapshot_runtime_dir}")
chmod 755 "${snapshot_runtime_dir}"
expect_failure \
  "${TMP_DIR}/snapshot_runtime_dir_mode_drift.out" \
  'Installed snapshot runtime directory boundary changed after prepare:' \
  "${ROLLBACK_ENV[@]}" NODES=closure-node bash batch_ne.sh launch
chmod "${snapshot_original_runtime_dir_mode}" "${snapshot_runtime_dir}"

snapshot_root_sealed_mode=$(stat -c '%a' "${rollback_snapshot_root}")
chmod u+w "${rollback_snapshot_root}"
mkdir "${rollback_snapshot_root}/unmanifested_package"
printf '%s\n' 'raise RuntimeError("must never be importable")' \
  >"${rollback_snapshot_root}/unmanifested_package/__init__.py"
chmod "${snapshot_root_sealed_mode}" "${rollback_snapshot_root}"
expect_failure \
  "${TMP_DIR}/snapshot_extra_package.out" \
  'Installed snapshot top-level directory closure changed after prepare.' \
  "${ROLLBACK_ENV[@]}" NODES=closure-node bash batch_ne.sh launch
chmod u+w "${rollback_snapshot_root}"
rm -rf "${rollback_snapshot_root}/unmanifested_package"
chmod "${snapshot_root_sealed_mode}" "${rollback_snapshot_root}"

chmod u+w "${snapshot_scripts_dir}"
mkfifo "${snapshot_scripts_dir}/unmanifested_fifo"
chmod "${snapshot_scripts_sealed_mode}" "${snapshot_scripts_dir}"
expect_failure \
  "${TMP_DIR}/snapshot_special_entry.out" \
  'Installed snapshot gained an unsupported special filesystem entry after prepare.' \
  "${ROLLBACK_ENV[@]}" NODES=closure-node bash batch_ne.sh launch
chmod u+w "${snapshot_scripts_dir}"
rm -f "${snapshot_scripts_dir}/unmanifested_fifo"
chmod "${snapshot_scripts_sealed_mode}" "${snapshot_scripts_dir}"

for fixture_node in existing-node preflight-node; do
  : >"${FAKE_TMUX_STATE_DIR}/${fixture_node}.rollback-contract.session"
  printf '%s\n' preexisting-owner >"${FAKE_TMUX_STATE_DIR}/${fixture_node}.rollback-contract.token"
done
expect_failure \
  "${TMP_DIR}/rollback_existing_session.out" \
  'tmux session already exists: rollback-contract' \
  "${ROLLBACK_ENV[@]}" NODES=existing-node RESTART=0 bash batch_ne.sh launch
[[ -f "${FAKE_TMUX_STATE_DIR}/existing-node.rollback-contract.session" ]] ||
  fail 'RESTART=0 launch failure killed the pre-existing same-name session'
[[ "$(cat "${FAKE_TMUX_STATE_DIR}/existing-node.rollback-contract.token")" == preexisting-owner ]] ||
  fail 'RESTART=0 launch failure replaced the pre-existing session ownership token'

expect_failure \
  "${TMP_DIR}/rollback_preflight_failure.out" \
  '[FAKE] forced preflight failure on preflight-node' \
  "${ROLLBACK_ENV[@]}" NODES=preflight-node RESTART=0 \
  FAKE_SSH_FAIL_PREFLIGHT_NODE=preflight-node bash batch_ne.sh launch
[[ -f "${FAKE_TMUX_STATE_DIR}/preflight-node.rollback-contract.session" ]] ||
  fail 'preflight failure killed a same-name session not created by this launch'
[[ "$(cat "${FAKE_TMUX_STATE_DIR}/preflight-node.rollback-contract.token")" == preexisting-owner ]] ||
  fail 'preflight failure changed the pre-existing session ownership token'
if [[ -s "${FAKE_TMUX_STATE_DIR}/kills.log" ]]; then
  sed -n '1,20p' "${FAKE_TMUX_STATE_DIR}/kills.log" >&2
  fail 'ownership-safe rollback killed a pre-existing session'
fi

# Exercise v2 launch lifecycle state with local ssh/tmux/ss fakes. These
# cases intentionally reuse one snapshot while binding every status result to
# a unique launch token, command digest, and epoch.
LIFECYCLE_NODE=lifecycle-node
LIFECYCLE_TARGET=40000
LIFECYCLE_ENV=(
  "${ROLLBACK_ENV[@]}"
  NODES="${LIFECYCLE_NODE}"
  MASTER_ADDR="${LIFECYCLE_NODE}"
  TARGET_LEARNING_ITERATION="${LIFECYCLE_TARGET}"
  RESTART=0
)
lifecycle_active_path() {
  local session="$1"
  local session_sha
  session_sha=$(printf '%s' "${session}" | sha256sum | awk '{print $1}')
  printf '%s/.active/%s_%s.state\n' "${ROLLBACK_REMOTE_ROOT}" "${session_sha}" "${LIFECYCLE_NODE}"
}
lifecycle_log_path() {
  local session="$1"
  printf '%s/%s/logs/batch_ne/%s_contract/node_0_%s.log\n' \
    "${ROLLBACK_REMOTE_ROOT}" "${rollback_snapshot_id}" "${session}" "${LIFECYCLE_NODE}"
}
write_lifecycle_active() {
  local session="$1" phase="$2" token="$3" command_sha="$4" epoch="$5"
  local state
  state=$(lifecycle_active_path "${session}")
  mkdir -p "$(dirname "${state}")"
  (umask 077; printf '2\t%s\t%s\tlogs/batch_ne/%s_contract\t1\t%s\t%s\t%s\n' \
    "${phase}" "${rollback_snapshot_id}" "${session}" "${token}" "${command_sha}" "${epoch}" >"${state}")
}
write_lifecycle_log() {
  local session="$1" token="$2" command_sha="$3" epoch="$4"
  local log
  log=$(lifecycle_log_path "${session}")
  mkdir -p "$(dirname "${log}")"
  printf 'HOLOSOMA_LAUNCH_BINDING token=%s command_sha256=%s launch_epoch=%s\n' \
    "${token}" "${command_sha}" "${epoch}" >"${log}"
}
assert_quarantined_rendezvous_for_token() {
  local expected_token="$1" label="$2" state version token session port created_at
  local -a states=()
  if [[ -d "${ROLLBACK_REMOTE_ROOT}/.rendezvous" ]]; then
    mapfile -t states < <(
      list_rendezvous_endpoint_states
    )
  fi
  [[ "${#states[@]}" == 2 ]] ||
    fail "${label} did not quarantine exactly two rendezvous reservations (found ${#states[@]})"
  for state in "${states[@]}"; do
    version='' token='' session='' port='' created_at=''
    IFS=$'\t' read -r version token session port created_at <"${state}" || true
    [[ "${version}" == 2 && "${token}" == "${expected_token}" ]] ||
      fail "${label} quarantined a rendezvous reservation with the wrong token: ${state}"
  done
}
list_rendezvous_endpoint_states() {
  [[ -d "${ROLLBACK_REMOTE_ROOT}/.rendezvous" ]] || return 0
  find "${ROLLBACK_REMOTE_ROOT}/.rendezvous" -type f -name '*.state' -print \
    | sed -nE '/\/[0-9a-f]{64}_[1-9][0-9]*\.state$/p' \
    | sort
}
rendezvous_endpoint_path() {
  local port="$1" master_key
  master_key=$(printf '%s' "${LIFECYCLE_NODE}" | sha256sum | awk '{print $1}')
  printf '%s/.rendezvous/%s_%s.state\n' \
    "${ROLLBACK_REMOTE_ROOT}" "${master_key}" "${port}"
}
clear_quarantined_rendezvous() {
  [[ -d "${ROLLBACK_REMOTE_ROOT}/.rendezvous" ]] || return 0
  find "${ROLLBACK_REMOTE_ROOT}/.rendezvous" -type f -name '*.state' -delete
}

wait_for_fixture_process_token() {
  local pid="$1" token="$2" label="$3" poll
  for ((poll = 0; poll < 400; poll++)); do
    if [[ -r "/proc/${pid}/environ" ]] \
        && tr '\0' '\n' <"/proc/${pid}/environ" 2>/dev/null \
          | grep -Fx "HOLOSOMA_LAUNCH_TOKEN=${token}" >/dev/null; then
      FIXTURE_PROCESS_SPECS+=("${pid}:${token}")
      return 0
    fi
    kill -0 "${pid}" 2>/dev/null || break
    sleep 0.005
  done
  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
  fail "${label} process did not expose its exact launch token before lifecycle probing"
}

# Launch ownership must come from an exact 256-bit kernel CSPRNG read.  A
# short/malformed read fails before active metadata or rendezvous mutation.
expect_failure \
  "${TMP_DIR}/malformed_csprng_token.out" \
  'Kernel CSPRNG returned a malformed launch ownership token' \
  "${LIFECYCLE_ENV[@]}" SESSION=malformed-csprng RUN_STAMP=malformed-csprng \
  MASTER_PORT=31251 HOLOSOMA_PROVENANCE_MASTER_PORT=31252 \
  FAKE_OD_OUTPUT=abc123 bash batch_ne.sh launch
[[ ! -e "$(lifecycle_active_path malformed-csprng)" ]] ||
  fail 'malformed CSPRNG output published active metadata'
[[ -z "$(list_rendezvous_endpoint_states)" ]] ||
  fail 'malformed CSPRNG output reserved rendezvous endpoints'
if rg -n '\$RANDOM|date \+%s.*launch_token|launch_token=.*date' batch_ne.sh >/dev/null; then
  fail 'launch ownership token must not use predictable shell randomness or wall-clock material'
fi

# tmux presence is a three-state query: rc=1 alone means absent.  A transport
# or server error must fail before active intent publication or reservations.
expect_failure \
  "${TMP_DIR}/tmux_presence_query_failure.out" \
  'tmux session-presence query failed for tmux-query-failure (rc=2)' \
  "${LIFECYCLE_ENV[@]}" SESSION=tmux-query-failure RUN_STAMP=tmux-query-failure \
  MASTER_PORT=30975 HOLOSOMA_PROVENANCE_MASTER_PORT=30976 \
  FAKE_TMUX_HAS_SESSION_RC=2 bash batch_ne.sh launch
[[ ! -e "$(lifecycle_active_path tmux-query-failure)" ]] ||
  fail 'tmux presence-query failure published active launch intent'
[[ -z "$(list_rendezvous_endpoint_states)" ]] ||
  fail 'tmux presence-query failure reserved rendezvous endpoints'

# Every active-state consumer must reject appended records and unsafe log-dir
# traversal/glob metadata before deciding terminal health or ownership.
ACTIVE_SHAPE_TOKEN=$(printf active-shape-token | sha256sum | awk '{print $1}')
ACTIVE_SHAPE_COMMAND=$(printf active-shape-command | sha256sum | awk '{print $1}')
ACTIVE_SHAPE_EPOCH=$(date +%s)
write_lifecycle_active active-extra-line rolled_back "${ACTIVE_SHAPE_TOKEN}" \
  "${ACTIVE_SHAPE_COMMAND}" "${ACTIVE_SHAPE_EPOCH}"
printf 'unexpected\tsecond\trecord\n' >>"$(lifecycle_active_path active-extra-line)"
expect_failure \
  "${TMP_DIR}/active_extra_line_status.out" \
  'must contain exactly one eight-field TSV record' \
  "${LIFECYCLE_ENV[@]}" SESSION=active-extra-line bash batch_ne.sh status
unsafe_active_state=$(lifecycle_active_path active-unsafe-log-dir)
mkdir -p "$(dirname "${unsafe_active_state}")"
(umask 077; printf '2\trolled_back\t%s\tlogs/batch_ne/../../*\t1\t%s\t%s\t%s\n' \
  "${rollback_snapshot_id}" "${ACTIVE_SHAPE_TOKEN}" "${ACTIVE_SHAPE_COMMAND}" \
  "${ACTIVE_SHAPE_EPOCH}" >"${unsafe_active_state}")
expect_failure \
  "${TMP_DIR}/active_unsafe_log_dir_status.out" \
  'malformed v2 identity fields' \
  "${LIFECYCLE_ENV[@]}" SESSION=active-unsafe-log-dir bash batch_ne.sh status
write_lifecycle_active active-stopped-pending stopped "${ACTIVE_SHAPE_TOKEN}" \
  pending "${ACTIVE_SHAPE_EPOCH}"
expect_failure \
  "${TMP_DIR}/active_stopped_pending_status.out" \
  'active launch metadata is malformed or unsupported' \
  "${LIFECYCLE_ENV[@]}" SESSION=active-stopped-pending bash batch_ne.sh status
write_lifecycle_active active-wrong-session rolled_back "${ACTIVE_SHAPE_TOKEN}" \
  "${ACTIVE_SHAPE_COMMAND}" "${ACTIVE_SHAPE_EPOCH}"
wrong_session_state=$(lifecycle_active_path active-wrong-session)
awk -F '\t' -v OFS='\t' '{$4="logs/batch_ne/different-session_contract"; print}' \
  "${wrong_session_state}" >"${wrong_session_state}.incoming-test"
mv -T "${wrong_session_state}.incoming-test" "${wrong_session_state}"
wrong_session_sha=$(sha256sum "${wrong_session_state}" | awk '{print $1}')
expect_failure \
  "${TMP_DIR}/active_wrong_session_preflight.out" \
  'Launch predecessor active metadata is malformed' \
  "${LIFECYCLE_ENV[@]}" SESSION=active-wrong-session RUN_STAMP=active-wrong-session \
  MASTER_PORT=31241 HOLOSOMA_PROVENANCE_MASTER_PORT=31242 bash batch_ne.sh launch
[[ "$(sha256sum "${wrong_session_state}" | awk '{print $1}')" == "${wrong_session_sha}" ]] ||
  fail 'wrong-session predecessor namespace was mutated during failed preflight'
write_lifecycle_active active-future-huge rolled_back "${ACTIVE_SHAPE_TOKEN}" \
  "${ACTIVE_SHAPE_COMMAND}" 99999999999999999999999999999999999999999999999999
expect_failure \
  "${TMP_DIR}/active_future_huge_status.out" \
  'active launch epoch exceeds the bounded node-clock skew without safe subtraction' \
  "${LIFECYCLE_ENV[@]}" SESSION=active-future-huge bash batch_ne.sh status

PROCESS_CLOSURE_TOKEN=$(printf process-closure-token | sha256sum | awk '{print $1}')
PROCESS_CLOSURE_COMMAND=$(printf process-closure-command | sha256sum | awk '{print $1}')
PROCESS_CLOSURE_EPOCH=$(date +%s)
write_lifecycle_active terminal-process-closure rolled_back \
  "${PROCESS_CLOSURE_TOKEN}" "${PROCESS_CLOSURE_COMMAND}" "${PROCESS_CLOSURE_EPOCH}"
env HOLOSOMA_LAUNCH_TOKEN="${PROCESS_CLOSURE_TOKEN}" \
  HOLOSOMA_COMMAND_SHA256="${PROCESS_CLOSURE_COMMAND}" \
  HOLOSOMA_LAUNCH_EPOCH="${PROCESS_CLOSURE_EPOCH}" sleep 3600 &
terminal_process_pid=$!
wait_for_fixture_process_token \
  "${terminal_process_pid}" "${PROCESS_CLOSURE_TOKEN}" terminal-process-closure
expect_failure \
  "${TMP_DIR}/terminal_process_status.out" \
  'exact launch-identity processes remain' \
  "${LIFECYCLE_ENV[@]}" SESSION=terminal-process-closure bash batch_ne.sh status
kill "${terminal_process_pid}" 2>/dev/null || true
wait "${terminal_process_pid}" 2>/dev/null || true
"${LIFECYCLE_ENV[@]}" SESSION=terminal-process-closure \
  bash batch_ne.sh status >"${TMP_DIR}/terminal_process_status_closed.out"

# The exact token+command scanner used by status/stop must also fail closed on
# a same-token process whose epoch is wrong.  This exercises the second /proc
# parser independently of the pending-intent predecessor checks below.
EXACT_WRONG_EPOCH_TOKEN=$(printf exact-wrong-epoch-token | sha256sum | awk '{print $1}')
EXACT_WRONG_EPOCH_COMMAND=$(printf exact-wrong-epoch-command | sha256sum | awk '{print $1}')
EXACT_WRONG_EPOCH=$(date +%s)
write_lifecycle_active terminal-exact-wrong-epoch rolled_back \
  "${EXACT_WRONG_EPOCH_TOKEN}" "${EXACT_WRONG_EPOCH_COMMAND}" "${EXACT_WRONG_EPOCH}"
env HOLOSOMA_LAUNCH_TOKEN="${EXACT_WRONG_EPOCH_TOKEN}" \
  HOLOSOMA_COMMAND_SHA256="${EXACT_WRONG_EPOCH_COMMAND}" \
  HOLOSOMA_LAUNCH_EPOCH="$((EXACT_WRONG_EPOCH + 1))" sleep 3600 &
exact_wrong_epoch_pid=$!
wait_for_fixture_process_token \
  "${exact_wrong_epoch_pid}" "${EXACT_WRONG_EPOCH_TOKEN}" terminal-exact-wrong-epoch
expect_failure \
  "${TMP_DIR}/terminal_exact_wrong_epoch_status.out" \
  "Process ${exact_wrong_epoch_pid} has conflicting/duplicate exact launch identity for token=${EXACT_WRONG_EPOCH_TOKEN}" \
  "${LIFECYCLE_ENV[@]}" SESSION=terminal-exact-wrong-epoch bash batch_ne.sh status
kill "${exact_wrong_epoch_pid}" 2>/dev/null || true
wait "${exact_wrong_epoch_pid}" 2>/dev/null || true
"${LIFECYCLE_ENV[@]}" SESSION=terminal-exact-wrong-epoch \
  bash batch_ne.sh status >"${TMP_DIR}/terminal_exact_wrong_epoch_status_closed.out"

PREDECESSOR_PROCESS_TOKEN=$(printf predecessor-process-token | sha256sum | awk '{print $1}')
PREDECESSOR_PROCESS_COMMAND=$(printf predecessor-process-command | sha256sum | awk '{print $1}')
PREDECESSOR_PROCESS_EPOCH=$(date +%s)
write_lifecycle_active predecessor-process-corrupt rolled_back \
  "${PREDECESSOR_PROCESS_TOKEN}" "${PREDECESSOR_PROCESS_COMMAND}" \
  "${PREDECESSOR_PROCESS_EPOCH}"
env HOLOSOMA_LAUNCH_TOKEN="${PREDECESSOR_PROCESS_TOKEN}" \
  HOLOSOMA_COMMAND_SHA256=malformed-command-digest \
  HOLOSOMA_LAUNCH_EPOCH="${PREDECESSOR_PROCESS_EPOCH}" sleep 3600 &
predecessor_corrupt_process_pid=$!
wait_for_fixture_process_token \
  "${predecessor_corrupt_process_pid}" "${PREDECESSOR_PROCESS_TOKEN}" predecessor-process-corrupt
expect_failure \
  "${TMP_DIR}/predecessor_corrupt_process.out" \
  "Process ${predecessor_corrupt_process_pid} has malformed environment for launch token=${PREDECESSOR_PROCESS_TOKEN}" \
  "${LIFECYCLE_ENV[@]}" SESSION=predecessor-process-corrupt \
  RUN_STAMP=predecessor-process-corrupt MASTER_PORT=30977 \
  HOLOSOMA_PROVENANCE_MASTER_PORT=30978 bash batch_ne.sh launch
kill "${predecessor_corrupt_process_pid}" 2>/dev/null || true
wait "${predecessor_corrupt_process_pid}" 2>/dev/null || true
[[ "$(cut -f6 "$(lifecycle_active_path predecessor-process-corrupt)")" == \
      "${PREDECESSOR_PROCESS_TOKEN}" ]] ||
  fail 'malformed predecessor process preflight mutated terminal active metadata'

run_corrupt_predecessor_epoch_case() {
  local mode="$1" session="predecessor-${1}-epoch" main_port="$2" provenance_port="$3"
  local token command epoch process_pid
  token=$(printf '%s-token' "${session}" | sha256sum | awk '{print $1}')
  command=$(printf '%s-command' "${session}" | sha256sum | awk '{print $1}')
  epoch=$(date +%s)
  write_lifecycle_active "${session}" rolled_back "${token}" "${command}" "${epoch}"
  case "${mode}" in
    wrong)
      env HOLOSOMA_LAUNCH_TOKEN="${token}" \
        HOLOSOMA_COMMAND_SHA256="${command}" \
        HOLOSOMA_LAUNCH_EPOCH="$((epoch + 1))" sleep 3600 &
      ;;
    missing)
      env -u HOLOSOMA_LAUNCH_EPOCH HOLOSOMA_LAUNCH_TOKEN="${token}" \
        HOLOSOMA_COMMAND_SHA256="${command}" sleep 3600 &
      ;;
    *) fail "unsupported corrupt predecessor epoch case: ${mode}" ;;
  esac
  process_pid=$!
  wait_for_fixture_process_token "${process_pid}" "${token}" "${session}"
  expect_failure \
    "${TMP_DIR}/${session}.out" \
    "Process ${process_pid} has malformed environment for launch token=${token} epoch=${epoch}" \
    "${LIFECYCLE_ENV[@]}" SESSION="${session}" RUN_STAMP="${session}" \
    MASTER_PORT="${main_port}" HOLOSOMA_PROVENANCE_MASTER_PORT="${provenance_port}" \
    bash batch_ne.sh launch
  kill "${process_pid}" 2>/dev/null || true
  wait "${process_pid}" 2>/dev/null || true
  [[ "$(cut -f6 "$(lifecycle_active_path "${session}")")" == "${token}" ]] ||
    fail "${mode}-epoch predecessor process preflight mutated terminal metadata"
  [[ -z "$(list_rendezvous_endpoint_states)" ]] ||
    fail "${mode}-epoch predecessor process preflight reserved rendezvous endpoints"
}

run_corrupt_predecessor_epoch_case wrong 31253 31254
run_corrupt_predecessor_epoch_case missing 31255 31256

# `launch_node` must not inherit Bash's `if ! function` errexit suppression.
# A failed GPU/data health SSH preflight is terminal before command identity or
# tmux creation, and the intent/ports must still roll back transactionally.
expect_failure \
  "${TMP_DIR}/node_health_preflight_failure.out" \
  '[FAKE] forced node GPU/data health preflight failure on lifecycle-node' \
  "${LIFECYCLE_ENV[@]}" SESSION=node-health-failure RUN_STAMP=node-health-failure \
  MASTER_PORT=30979 HOLOSOMA_PROVENANCE_MASTER_PORT=30980 \
  SKIP_NODE_HEALTH_CHECK=0 FAKE_SSH_FAIL_NODE_HEALTH_NODE="${LIFECYCLE_NODE}" \
  bash batch_ne.sh launch
node_health_failure_state=$(lifecycle_active_path node-health-failure)
IFS=$'\t' read -r _node_health_version node_health_phase _node_health_snapshot \
  _node_health_log _node_health_target _node_health_token node_health_command \
  _node_health_epoch <"${node_health_failure_state}"
[[ "${node_health_phase}" == rolled_back && "${node_health_command}" == pending ]] ||
  fail 'node health preflight failure did not remain intent-only and roll back'
[[ ! -f "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.node-health-failure.session" ]] ||
  fail 'node health preflight failure incorrectly continued into tmux creation'
if [[ -n "$(list_rendezvous_endpoint_states)" ]]; then
  fail 'node health preflight failure leaked a rendezvous reservation'
fi

# The selected-GPU inventory immediately before staging/tmux is mandatory even
# when the optional repo/logger/data health bundle is skipped.  Its failure is
# post-intent/post-reservation but must still leave an exact rolled-back state.
expect_failure \
  "${TMP_DIR}/prelaunch_idle_failure.out" \
  '[FAKE] forced pre-launch selected-GPU idle failure on lifecycle-node' \
  "${LIFECYCLE_ENV[@]}" SESSION=prelaunch-idle-failure RUN_STAMP=prelaunch-idle-failure \
  MASTER_PORT=31401 HOLOSOMA_PROVENANCE_MASTER_PORT=31402 \
  SKIP_NODE_HEALTH_CHECK=1 \
  FAKE_SSH_FAIL_PRELAUNCH_IDLE_NODE="${LIFECYCLE_NODE}" \
  bash batch_ne.sh launch
prelaunch_idle_failure_state=$(lifecycle_active_path prelaunch-idle-failure)
IFS=$'\t' read -r _prelaunch_idle_version prelaunch_idle_phase \
  _prelaunch_idle_snapshot _prelaunch_idle_log _prelaunch_idle_target \
  _prelaunch_idle_token prelaunch_idle_command _prelaunch_idle_epoch \
  <"${prelaunch_idle_failure_state}"
[[ "${prelaunch_idle_phase}" == rolled_back \
      && "${prelaunch_idle_command}" == pending ]] ||
  fail 'pre-launch selected-GPU idle failure did not remain intent-only and roll back'
[[ ! -f "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.prelaunch-idle-failure.session" ]] ||
  fail 'pre-launch selected-GPU idle failure incorrectly continued into tmux creation'
if [[ -n "$(list_rendezvous_endpoint_states)" ]]; then
  fail 'pre-launch selected-GPU idle failure leaked a rendezvous reservation'
fi

# A streamed control payload is executable/stop-authoritative only as the
# exact regular file that was hashed.  Replacing it with a same-content
# symlink between installation and tmux launch must fail before session start.
expect_failure \
  "${TMP_DIR}/control_script_symlink_launch.out" \
  'Verified launch script is missing, non-regular, or symlinked' \
  "${LIFECYCLE_ENV[@]}" SESSION=control-script-symlink RUN_STAMP=control-script-symlink \
  MASTER_PORT=31243 HOLOSOMA_PROVENANCE_MASTER_PORT=31244 \
  FAKE_SSH_REPLACE_CONTROL_WITH_SYMLINK=1 bash batch_ne.sh launch
[[ "$(cut -f2 "$(lifecycle_active_path control-script-symlink)")" == rolled_back ]] ||
  fail 'symlinked control-script rejection did not close the published intent'
[[ ! -f "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.control-script-symlink.session" ]] ||
  fail 'symlinked control script reached tmux creation'
[[ -z "$(list_rendezvous_endpoint_states)" ]] ||
  fail 'symlinked control-script rejection leaked rendezvous endpoints'

control_incoming_dir="${ROLLBACK_REMOTE_ROOT}/${rollback_snapshot_id}/.run_control/.incoming"
rm -rf -- "${control_incoming_dir}"
ln -s "${control_incoming_dir}.outside" "${control_incoming_dir}"
expect_failure \
  "${TMP_DIR}/control_incoming_symlink_launch.out" \
  'Refusing non-directory or symlinked launch-control directory' \
  "${LIFECYCLE_ENV[@]}" SESSION=control-incoming-symlink RUN_STAMP=control-incoming-symlink \
  MASTER_PORT=31257 HOLOSOMA_PROVENANCE_MASTER_PORT=31258 bash batch_ne.sh launch
[[ -L "${control_incoming_dir}" && ! -e "${control_incoming_dir}.outside" ]] ||
  fail 'launch-control incoming symlink was followed or replaced'
[[ "$(cut -f2 "$(lifecycle_active_path control-incoming-symlink)")" == rolled_back ]] ||
  fail 'launch-control incoming symlink rejection did not close launch intent'
[[ -z "$(list_rendezvous_endpoint_states)" ]] ||
  fail 'launch-control incoming symlink rejection leaked rendezvous endpoints'
rm -f -- "${control_incoming_dir}"
mkdir -- "${control_incoming_dir}"

run_preexisting_log_case() {
  local kind="$1" session="preexisting-log-${1}" main_port="$2" provenance_port="$3"
  local log_dir="${ROLLBACK_REMOTE_ROOT}/${rollback_snapshot_id}/logs/batch_ne/${session}_${session}"
  local log_file="${log_dir}/node_0_${LIFECYCLE_NODE}.log"
  local sentinel_sha=''
  case "${kind}" in
    old-run)
      mkdir -p "${log_dir}"
      printf 'scientific-old-run-sentinel\n' >"${log_file}"
      sentinel_sha=$(sha256sum "${log_file}" | awk '{print $1}')
      ;;
    dangling)
      mkdir -p "$(dirname "${log_dir}")"
      ln -s "${log_dir}.missing-target" "${log_dir}"
      ;;
    *) fail "unsupported pre-existing log case: ${kind}" ;;
  esac
  expect_failure \
    "${TMP_DIR}/preexisting_log_${kind}.out" \
    'Refusing to reuse pre-existing run-specific log directory' \
    "${LIFECYCLE_ENV[@]}" SESSION="${session}" RUN_STAMP="${session}" \
    MASTER_PORT="${main_port}" HOLOSOMA_PROVENANCE_MASTER_PORT="${provenance_port}" \
    bash batch_ne.sh launch
  [[ "$(cut -f2 "$(lifecycle_active_path "${session}")")" == rolled_back ]] ||
    fail "${kind} log collision did not close the published intent"
  [[ ! -f "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.${session}.session" ]] ||
    fail "${kind} log collision reached tmux creation"
  [[ -z "$(list_rendezvous_endpoint_states)" ]] ||
    fail "${kind} log collision leaked rendezvous endpoints"
  if [[ "${kind}" == old-run ]]; then
    [[ "$(sha256sum "${log_file}" | awk '{print $1}')" == "${sentinel_sha}" ]] ||
      fail 'pre-existing run log sentinel was modified'
  else
    [[ -L "${log_dir}" && ! -e "${log_dir}.missing-target" ]] ||
      fail 'dangling run-log symlink was followed or replaced'
  fi
}

run_preexisting_log_case old-run 31245 31246
run_preexisting_log_case dangling 31247 31248

run_preexisting_owner_marker_case() {
  local kind="$1" session="preexisting-owner-${1}" main_port="$2" provenance_port="$3"
  local token epoch log_rel log_dir owner_file owner_sha target
  token=$(printf '%s-fixed-csprng' "${session}" | sha256sum | awk '{print $1}')
  epoch=$((1784041600 + main_port))
  log_rel="logs/batch_ne/${session}_${session}"
  log_dir="${ROLLBACK_REMOTE_ROOT}/${rollback_snapshot_id}/${log_rel}"
  owner_file="${log_dir}/.holosoma_launch_owner_v2"
  mkdir -p "${log_dir}"
  target="${LIFECYCLE_TARGET}"
  case "${kind}" in
    wrong-target) target=$((LIFECYCLE_TARGET - 1)) ;;
    extra-field) ;;
    *) fail "unsupported pre-existing owner marker case: ${kind}" ;;
  esac
  printf '2\t%s\t%s\t%s\t%s\t1\t%s\t%s' \
    "${rollback_snapshot_id}" "${session}" "${log_rel}" "${target}" "${token}" "${epoch}" \
    >"${owner_file}"
  if [[ "${kind}" == extra-field ]]; then
    printf '\tunexpected' >>"${owner_file}"
  fi
  printf '\n' >>"${owner_file}"
  owner_sha=$(sha256sum "${owner_file}" | awk '{print $1}')
  expect_failure \
    "${TMP_DIR}/${session}.out" \
    'Refusing to reuse pre-existing run-specific log directory without this exact launch owner' \
    "${LIFECYCLE_ENV[@]}" SESSION="${session}" RUN_STAMP="${session}" \
    MASTER_PORT="${main_port}" HOLOSOMA_PROVENANCE_MASTER_PORT="${provenance_port}" \
    FAKE_OD_OUTPUT="${token}" FAKE_DATE_SECONDS="${epoch}" bash batch_ne.sh launch
  [[ "$(sha256sum "${owner_file}" | awk '{print $1}')" == "${owner_sha}" ]] ||
    fail "${kind} pre-existing owner marker was modified"
  [[ "$(cut -f2 "$(lifecycle_active_path "${session}")")" == rolled_back ]] ||
    fail "${kind} owner-marker collision did not close the launch intent"
  [[ ! -f "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.${session}.session" ]] ||
    fail "${kind} owner-marker collision reached tmux creation"
  [[ -z "$(list_rendezvous_endpoint_states)" ]] ||
    fail "${kind} owner-marker collision leaked rendezvous endpoints"
}

run_preexisting_owner_marker_case wrong-target 31261 31262
run_preexisting_owner_marker_case extra-field 31263 31264

dangling_rendezvous_state=$(rendezvous_endpoint_path 31249)
mkdir -p "$(dirname "${dangling_rendezvous_state}")"
ln -s "${dangling_rendezvous_state}.missing-target" "${dangling_rendezvous_state}"
expect_failure \
  "${TMP_DIR}/dangling_rendezvous_reserve.out" \
  'Rendezvous port is already reserved' \
  "${LIFECYCLE_ENV[@]}" SESSION=dangling-rendezvous RUN_STAMP=dangling-rendezvous \
  MASTER_PORT=31249 HOLOSOMA_PROVENANCE_MASTER_PORT=31250 bash batch_ne.sh launch
[[ -L "${dangling_rendezvous_state}" \
      && ! -e "${dangling_rendezvous_state}.missing-target" ]] ||
  fail 'rendezvous reservation followed or replaced a dangling endpoint symlink'
[[ "$(cut -f2 "$(lifecycle_active_path dangling-rendezvous)")" == rolled_back ]] ||
  fail 'dangling rendezvous reservation rejection did not close launch intent'
[[ ! -f "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.dangling-rendezvous.session" ]] ||
  fail 'dangling rendezvous reservation reached tmux creation'
rm -f "${dangling_rendezvous_state}"

# If the remote launch reports success but the controller-local result channel
# loses/corrupts the command digest, the live current node is not intent-only.
# When token-bound active-state recovery also fails, preserve both the session
# and its port reservations in rolling_back quarantine.
expect_failure \
  "${TMP_DIR}/successful_launch_missing_identity.out" \
  '[FAKE] forced launch identity-recovery failure on lifecycle-node' \
  "${LIFECYCLE_ENV[@]}" SESSION=success-without-identity RUN_STAMP=success-without-identity \
  MASTER_PORT=31017 HOLOSOMA_PROVENANCE_MASTER_PORT=31018 \
  FAKE_CORRUPT_LAUNCH_RESULT=1 \
  FAKE_SSH_FAIL_IDENTITY_RECOVERY_NODE="${LIFECYCLE_NODE}" \
  bash batch_ne.sh launch
success_without_identity_state=$(lifecycle_active_path success-without-identity)
IFS=$'\t' read -r _success_without_identity_version success_without_identity_phase \
  _success_without_identity_snapshot _success_without_identity_log \
  _success_without_identity_target success_without_identity_token \
  success_without_identity_command _success_without_identity_epoch \
  <"${success_without_identity_state}"
[[ "${success_without_identity_phase}" == rolling_back \
      && "${success_without_identity_command}" =~ ^[0-9a-f]{64}$ ]] ||
  fail 'successful launch without controller identity was falsely marked intent-only/rolled_back'
success_without_identity_prefix="${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.success-without-identity"
[[ -f "${success_without_identity_prefix}.session" ]] ||
  fail 'successful launch without recoverable controller identity lost its unconfirmed session'
assert_quarantined_rendezvous_for_token \
  "${success_without_identity_token}" 'successful launch without recoverable controller identity'
clear_quarantined_rendezvous
rm -f "${success_without_identity_prefix}.session" "${success_without_identity_prefix}.token" \
  "${success_without_identity_prefix}.command_sha256" \
  "${success_without_identity_prefix}.launch_epoch" \
  "${success_without_identity_prefix}.env_token" \
  "${success_without_identity_prefix}.env_command_sha256" \
  "${success_without_identity_prefix}.env_launch_epoch"

# Losing the SSH reply after the current node atomically publishes its launch
# intent must still include that ambiguous node in the exact-token rollback.
# A new controller/token must then be able to reuse the stopped lifecycle slot.
expect_failure \
  "${TMP_DIR}/post_commit_intent_disconnect.out" \
  '[FAKE] forced SSH failure after committed launch intent on lifecycle-node' \
  "${LIFECYCLE_ENV[@]}" SESSION=post-commit-intent RUN_STAMP=post-commit-intent-failed \
  MASTER_PORT=30981 HOLOSOMA_PROVENANCE_MASTER_PORT=30982 \
  FAKE_SSH_FAIL_AFTER_PUBLISH_INTENT_NODE="${LIFECYCLE_NODE}" \
  bash batch_ne.sh launch
post_commit_intent_state=$(lifecycle_active_path post-commit-intent)
IFS=$'\t' read -r _post_intent_version post_intent_phase _post_intent_snapshot \
  _post_intent_log _post_intent_target post_intent_failed_token post_intent_command \
  _post_intent_epoch <"${post_commit_intent_state}"
[[ "${post_intent_phase}" == rolled_back && "${post_intent_command}" == pending ]] ||
  fail 'post-commit launch-intent disconnect did not roll the ambiguous current node back'
[[ ! -f "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.post-commit-intent.session" ]] ||
  fail 'post-commit launch-intent disconnect unexpectedly created a tmux session'
if [[ -n "$(list_rendezvous_endpoint_states)" ]]; then
  fail 'post-commit launch-intent disconnect leaked a rendezvous reservation'
fi
"${LIFECYCLE_ENV[@]}" SESSION=post-commit-intent RUN_STAMP=post-commit-intent-retry \
  MASTER_PORT=30981 HOLOSOMA_PROVENANCE_MASTER_PORT=30982 \
  bash batch_ne.sh launch >"${TMP_DIR}/post_commit_intent_retry.out"
IFS=$'\t' read -r _post_intent_retry_version post_intent_retry_phase _post_intent_retry_snapshot \
  _post_intent_retry_log _post_intent_retry_target post_intent_retry_token \
  _post_intent_retry_command _post_intent_retry_epoch <"${post_commit_intent_state}"
[[ "${post_intent_retry_phase}" == running \
      && "${post_intent_retry_token}" != "${post_intent_failed_token}" ]] ||
  fail 'a new launch token could not proceed after ambiguous intent rollback'
"${LIFECYCLE_ENV[@]}" SESSION=post-commit-intent RUN_STAMP=post-commit-intent-retry \
  MASTER_PORT=30981 HOLOSOMA_PROVENANCE_MASTER_PORT=30982 \
  bash batch_ne.sh stop >"${TMP_DIR}/post_commit_intent_retry_stop.out"

# A remote publish shell can outlive a lost controller SSH reply.  Cleanup's
# same-lock tombstone must commit first and make the delayed publisher unable
# to resurrect launching metadata.
delayed_publish_gate="${TMP_DIR}/delayed-publish-gate"
expect_failure \
  "${TMP_DIR}/delayed_publish_controller.out" \
  '[FAKE] delayed publish shell detached before commit on lifecycle-node' \
  "${LIFECYCLE_ENV[@]}" SESSION=delayed-publish RUN_STAMP=delayed-publish \
  MASTER_PORT=30983 HOLOSOMA_PROVENANCE_MASTER_PORT=30984 \
  FAKE_SSH_DELAY_PUBLISH_GATE_DIR="${delayed_publish_gate}" bash batch_ne.sh launch
[[ -f "${delayed_publish_gate}/reached" ]] || fail 'delayed publish shell was not scheduled'
touch "${delayed_publish_gate}/release"
for ((gate_wait = 0; gate_wait < FIXTURE_GATE_MAX_POLLS; gate_wait++)); do
  [[ -f "${delayed_publish_gate}/late.rc" ]] && break
  sleep 0.025
done
[[ "$(cat "${delayed_publish_gate}/late.rc" 2>/dev/null || true)" == 3 ]] || {
  sed -n '1,80p' "${delayed_publish_gate}/late.out" >&2 || true
  fail 'delayed publisher was not rejected by the durable intent tombstone'
}
[[ ! -e "$(lifecycle_active_path delayed-publish)" ]] ||
  fail 'delayed publisher resurrected active metadata after absent-intent cancellation'
[[ -z "$(list_rendezvous_endpoint_states)" ]] ||
  fail 'delayed publish failure leaked rendezvous endpoints'

# The publisher itself must compare-and-swap the exact terminal record observed
# by preflight.  These gates run the old publisher before cancellation exists,
# replace its predecessor with a different transaction, and then prove that
# both equal and arbitrarily large epochs are preserved without authorization
# by wall-clock ordering.
run_publish_predecessor_cas_case() {
  local label="$1" replacement_epoch="$2" main_port="$3" provenance_port="$4"
  local session="cas-${label}" gate="${TMP_DIR}/cas-${label}-gate"
  local old_token old_command replacement_token replacement_command
  local controller_pid session_sha active_state replacement_state_sha
  local cas_version cas_token cas_epoch cas_snapshot cas_log cas_target cas_session cas_node
  local -a cas_tombstones=()
  old_token=$(printf 'old-%s-token' "${label}" | sha256sum | awk '{print $1}')
  old_command=$(printf 'old-%s-command' "${label}" | sha256sum | awk '{print $1}')
  replacement_token=$(printf 'replacement-%s-token' "${label}" | sha256sum | awk '{print $1}')
  replacement_command=$(printf 'replacement-%s-command' "${label}" | sha256sum | awk '{print $1}')
  write_lifecycle_active "${session}" rolled_back "${old_token}" "${old_command}" 1999999999
  active_state=$(lifecycle_active_path "${session}")
  mkdir -p "${gate}"
  "${LIFECYCLE_ENV[@]}" SESSION="${session}" RUN_STAMP="${session}" \
    MASTER_PORT="${main_port}" HOLOSOMA_PROVENANCE_MASTER_PORT="${provenance_port}" \
    FAKE_DATE_SECONDS=2000000000 FAKE_SSH_PUBLISH_GATE_DIR="${gate}" \
    bash batch_ne.sh launch >"${TMP_DIR}/cas_${label}_controller.out" 2>&1 &
  controller_pid=$!
  for ((gate_wait = 0; gate_wait < FIXTURE_GATE_MAX_POLLS; gate_wait++)); do
    [[ -f "${gate}/reached" ]] && break
    sleep 0.025
  done
  [[ -f "${gate}/reached" ]] || fail "${label} publisher did not reach its post-preflight gate"
  session_sha=$(printf '%s' "${session}" | sha256sum | awk '{print $1}')
  if compgen -G "${ROLLBACK_REMOTE_ROOT}/.active/cancelled.${session_sha}.${LIFECYCLE_NODE}.*.state" >/dev/null; then
    fail "${label} cancellation tombstone existed before the publisher-first CAS"
  fi
  write_lifecycle_active "${session}" rolled_back \
    "${replacement_token}" "${replacement_command}" "${replacement_epoch}"
  replacement_state_sha=$(sha256sum "${active_state}" | awk '{print $1}')
  touch "${gate}/release"
  if wait "${controller_pid}"; then
    fail "${label} stale publisher unexpectedly replaced a changed terminal predecessor"
  fi
  grep -F 'Launch predecessor CAS mismatch: terminal active metadata changed after preflight.' \
    "${TMP_DIR}/cas_${label}_controller.out" >/dev/null || {
      sed -n '1,100p' "${TMP_DIR}/cas_${label}_controller.out" >&2
      fail "${label} publisher did not reject the changed predecessor via CAS"
    }
  [[ "$(cut -f6 "${active_state}")" == "${replacement_token}" \
        && "$(cut -f7 "${active_state}")" == "${replacement_command}" \
        && "$(cut -f8 "${active_state}")" == "${replacement_epoch}" \
        && "$(sha256sum "${active_state}" | awk '{print $1}')" == "${replacement_state_sha}" ]] ||
    fail "${label} cancellation overwrote the preserved different-token terminal record"
  mapfile -t cas_tombstones < <(
    find "${ROLLBACK_REMOTE_ROOT}/.active" -maxdepth 1 -type f \
      -name "cancelled.${session_sha}.${LIFECYCLE_NODE}.*.state" -print
  )
  [[ "${#cas_tombstones[@]}" == 1 ]] ||
    fail "${label} CAS cleanup did not leave exactly one durable cancellation tombstone"
  cas_version='' cas_token='' cas_epoch='' cas_snapshot='' cas_log='' cas_target=''
  cas_session='' cas_node=''
  IFS=$'\t' read -r cas_version cas_token cas_epoch cas_snapshot cas_log cas_target \
    cas_session cas_node <"${cas_tombstones[0]}" || true
  [[ "$(awk 'END { print NR }' "${cas_tombstones[0]}")" == 1 \
        && "$(awk -F '\t' 'NR == 1 { print NF }' "${cas_tombstones[0]}")" == 8 \
        && "${cas_version}" == 1 && "${cas_token}" =~ ^[0-9a-f]{64}$ \
        && "${cas_epoch}" == 2000000000 && "${cas_snapshot}" == "${rollback_snapshot_id}" \
        && "${cas_log}" == "logs/batch_ne/${session}_${session}" \
        && "${cas_target}" == "${LIFECYCLE_TARGET}" \
        && "${cas_session}" == "${session}" \
        && "${cas_node}" == "${LIFECYCLE_NODE}" \
        && "${cas_tombstones[0]}" == *."${cas_token}".state ]] ||
    fail "${label} CAS cleanup tombstone has the wrong exact launch identity"
  grep -F 'verified cancelled-intent closure active_disposition=other' \
    "${TMP_DIR}/cas_${label}_controller.out" >/dev/null ||
    fail "${label} CAS cleanup did not prove preserved-other closure"
  [[ -z "$(list_rendezvous_endpoint_states)" ]] ||
    fail "${label} predecessor-CAS failure leaked rendezvous endpoints"
}

run_publish_predecessor_cas_case equal 2000000000 31105 31106
run_publish_predecessor_cas_case huge \
  99999999999999999999999999999999999999999999999999 31107 31108

# The reservation transaction uses the same cancellation protocol under its
# global lock: whether reserve or cancel gets the lock first, no late endpoint
# pair may appear after controller cleanup returns.
delayed_reservation_gate="${TMP_DIR}/delayed-reservation-gate"
expect_failure \
  "${TMP_DIR}/delayed_reservation_controller.out" \
  '[FAKE] delayed reservation shell detached before commit on lifecycle-node' \
  "${LIFECYCLE_ENV[@]}" SESSION=delayed-reservation RUN_STAMP=delayed-reservation \
  MASTER_PORT=30985 HOLOSOMA_PROVENANCE_MASTER_PORT=30986 \
  FAKE_SSH_DELAY_RESERVATION_GATE_DIR="${delayed_reservation_gate}" bash batch_ne.sh launch
[[ -f "${delayed_reservation_gate}/reached" ]] || fail 'delayed reservation shell was not scheduled'
touch "${delayed_reservation_gate}/release"
for ((gate_wait = 0; gate_wait < FIXTURE_GATE_MAX_POLLS; gate_wait++)); do
  [[ -f "${delayed_reservation_gate}/late.rc" ]] && break
  sleep 0.025
done
[[ "$(cat "${delayed_reservation_gate}/late.rc" 2>/dev/null || true)" == 3 ]] || {
  sed -n '1,80p' "${delayed_reservation_gate}/late.out" >&2 || true
  fail 'delayed reservation was not rejected by the durable token tombstone'
}
[[ "$(cut -f2 "$(lifecycle_active_path delayed-reservation)")" == rolled_back ]] ||
  fail 'delayed reservation cleanup did not close launch intents'
[[ -z "$(list_rendezvous_endpoint_states)" ]] ||
  fail 'delayed reservation shell recreated rendezvous endpoints after cleanup'

# The opposite ordering is equally safe: if reserve committed before its SSH
# reply was lost, cancellation waits for the reservation lock and exact-pair
# release removes the committed endpoints only after intent closure.
expect_failure \
  "${TMP_DIR}/post_commit_reservation_disconnect.out" \
  '[FAKE] forced SSH failure after committed rendezvous reservation on lifecycle-node' \
  "${LIFECYCLE_ENV[@]}" SESSION=post-commit-reservation RUN_STAMP=post-commit-reservation \
  MASTER_PORT=31103 HOLOSOMA_PROVENANCE_MASTER_PORT=31104 \
  FAKE_SSH_FAIL_AFTER_RESERVE_NODE="${LIFECYCLE_NODE}" bash batch_ne.sh launch
[[ "$(cut -f2 "$(lifecycle_active_path post-commit-reservation)")" == rolled_back ]] ||
  fail 'post-commit reservation failure did not close published launch intents'
[[ -z "$(list_rendezvous_endpoint_states)" ]] ||
  fail 'post-commit reservation failure left its exact endpoint pair reserved'

# tmux creation writes the ownership environment atomically.  If the remote
# reply fails before any @option is installed, exact-env rollback must remove
# that orphan while preserving a different session and releasing both ports.
post_create_unrelated_prefix="${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.post-create-unrelated"
: >"${post_create_unrelated_prefix}.session"
printf '%s\n' unrelated-other-token >"${post_create_unrelated_prefix}.token"
expect_failure \
  "${TMP_DIR}/post_create_disconnect.out" \
  '[FAKE] forced SSH-visible failure after atomic tmux creation' \
  "${LIFECYCLE_ENV[@]}" SESSION=post-create-disconnect RUN_STAMP=post-create-disconnect \
  MASTER_PORT=30983 HOLOSOMA_PROVENANCE_MASTER_PORT=30984 \
  FAKE_TMUX_POST_CREATE_FAILURE=exact bash batch_ne.sh launch
post_create_state=$(lifecycle_active_path post-create-disconnect)
[[ "$(cut -f2 "${post_create_state}")" == rolled_back ]] ||
  fail 'post-create disconnect did not publish rolled_back lifecycle state'
[[ ! -f "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.post-create-disconnect.session" ]] ||
  fail 'post-create disconnect left its exact atomic-env orphan alive'
[[ -f "${post_create_unrelated_prefix}.session" \
      && "$(cat "${post_create_unrelated_prefix}.token")" == unrelated-other-token ]] ||
  fail 'post-create rollback damaged an unrelated session'
if [[ -n "$(list_rendezvous_endpoint_states)" ]]; then
  fail 'post-create disconnect leaked rendezvous reservations'
fi
rm -f "${post_create_unrelated_prefix}.session" "${post_create_unrelated_prefix}.token"

# A corrupted/missing controller result must be recovered even when the remote
# launch call itself failed after atomic tmux creation.  The recovered SHA is
# then sufficient for exact rollback and final closure/release.
expect_failure \
  "${TMP_DIR}/post_create_corrupt_result_recovery.out" \
  'Recovered exact command SHA256 from token-bound remote identity for rollback.' \
  "${LIFECYCLE_ENV[@]}" SESSION=post-create-recovered RUN_STAMP=post-create-recovered \
  MASTER_PORT=31101 HOLOSOMA_PROVENANCE_MASTER_PORT=31102 \
  FAKE_CORRUPT_LAUNCH_RESULT=1 FAKE_TMUX_POST_CREATE_FAILURE=exact \
  bash batch_ne.sh launch
post_create_recovered_state=$(lifecycle_active_path post-create-recovered)
[[ "$(cut -f2 "${post_create_recovered_state}")" == rolled_back \
      && "$(cut -f7 "${post_create_recovered_state}")" =~ ^[0-9a-f]{64}$ ]] ||
  fail 'corrupt controller result was not recovered into exact rolled_back identity'
[[ ! -f "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.post-create-recovered.session" ]] ||
  fail 'recovered post-create identity did not remove its exact tmux orphan'
[[ -z "$(list_rendezvous_endpoint_states)" ]] ||
  fail 'recovered post-create identity did not release its reservation pair'

# Even an exact atomic environment is insufficient when a pre-existing option
# conflicts.  This models a same-name/other-token identity and proves the
# fallback never broadens into an unsafe name-only kill.
expect_failure \
  "${TMP_DIR}/post_create_conflicting_option.out" \
  'Exact token/command/epoch mismatch across atomic environment/options' \
  "${LIFECYCLE_ENV[@]}" SESSION=post-create-conflict RUN_STAMP=post-create-conflict \
  MASTER_PORT=30985 HOLOSOMA_PROVENANCE_MASTER_PORT=30986 \
  FAKE_TMUX_POST_CREATE_FAILURE=conflicting-option bash batch_ne.sh launch
post_create_conflict_state=$(lifecycle_active_path post-create-conflict)
post_create_conflict_prefix="${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.post-create-conflict"
[[ "$(cut -f2 "${post_create_conflict_state}")" == rolling_back ]] ||
  fail 'unconfirmed conflicting-option rollback was falsely marked rolled_back'
[[ -f "${post_create_conflict_prefix}.session" \
      && "$(cat "${post_create_conflict_prefix}.token")" == conflicting-other-token ]] ||
  fail 'atomic-env fallback killed a same-name session with a conflicting option token'
post_create_conflict_token=$(cut -f6 "${post_create_conflict_state}")
assert_quarantined_rendezvous_for_token \
  "${post_create_conflict_token}" 'conflicting-option post-create failure'
clear_quarantined_rendezvous
rm -f "${post_create_conflict_prefix}.session" "${post_create_conflict_prefix}.token" \
  "${post_create_conflict_prefix}.command_sha256" "${post_create_conflict_prefix}.launch_epoch" \
  "${post_create_conflict_prefix}.env_token" "${post_create_conflict_prefix}.env_command_sha256" \
  "${post_create_conflict_prefix}.env_launch_epoch"

# Sessions launched by an older controller have no atomic environment.  The
# rollback path remains backward compatible only when all three legacy
# @options exactly match the failed launch identity.
expect_failure \
  "${TMP_DIR}/post_create_legacy_options_only.out" \
  '[FAKE] forced post-create failure with exact legacy ownership options only' \
  "${LIFECYCLE_ENV[@]}" SESSION=post-create-legacy RUN_STAMP=post-create-legacy \
  MASTER_PORT=30987 HOLOSOMA_PROVENANCE_MASTER_PORT=30988 \
  FAKE_TMUX_POST_CREATE_FAILURE=legacy-options-only bash batch_ne.sh launch
post_create_legacy_state=$(lifecycle_active_path post-create-legacy)
post_create_legacy_prefix="${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.post-create-legacy"
[[ "$(cut -f2 "${post_create_legacy_state}")" == rolled_back ]] ||
  fail 'exact legacy options-only rollback was not marked rolled_back'
[[ ! -f "${post_create_legacy_prefix}.session" ]] ||
  fail 'exact legacy options-only rollback left its owned session alive'
if [[ -n "$(list_rendezvous_endpoint_states)" ]]; then
  fail 'legacy options-only post-create failure leaked rendezvous reservations'
fi

# The presence of only part of the new atomic environment is corruption, not
# a legacy session.  Exact options must not weaken that fail-closed decision.
expect_failure \
  "${TMP_DIR}/post_create_partial_environment.out" \
  'Exact token/command/epoch mismatch across atomic environment/options' \
  "${LIFECYCLE_ENV[@]}" SESSION=post-create-partial RUN_STAMP=post-create-partial \
  MASTER_PORT=30989 HOLOSOMA_PROVENANCE_MASTER_PORT=30990 \
  FAKE_TMUX_POST_CREATE_FAILURE=partial-environment bash batch_ne.sh launch
post_create_partial_state=$(lifecycle_active_path post-create-partial)
post_create_partial_prefix="${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.post-create-partial"
[[ "$(cut -f2 "${post_create_partial_state}")" == rolling_back ]] ||
  fail 'partial atomic-environment rollback was falsely marked rolled_back'
[[ -f "${post_create_partial_prefix}.session" \
      && -f "${post_create_partial_prefix}.env_token" \
      && ! -f "${post_create_partial_prefix}.env_command_sha256" \
      && ! -f "${post_create_partial_prefix}.env_launch_epoch" ]] ||
  fail 'partial atomic-environment rollback did not preserve the unconfirmed session'
post_create_partial_token=$(cut -f6 "${post_create_partial_state}")
assert_quarantined_rendezvous_for_token \
  "${post_create_partial_token}" 'partial atomic-environment post-create failure'
clear_quarantined_rendezvous
rm -f "${post_create_partial_prefix}.session" "${post_create_partial_prefix}.token" \
  "${post_create_partial_prefix}.command_sha256" "${post_create_partial_prefix}.launch_epoch" \
  "${post_create_partial_prefix}.env_token" "${post_create_partial_prefix}.env_command_sha256" \
  "${post_create_partial_prefix}.env_launch_epoch"

run_corrupt_post_create_identity_case() {
  local mode="$1" session="$2" master_port="$3" provenance_port="$4" label="$5"
  local output="${TMP_DIR}/${session}.out"
  expect_failure \
    "${output}" \
    'Exact token/command/epoch mismatch across atomic environment/options' \
    "${LIFECYCLE_ENV[@]}" SESSION="${session}" RUN_STAMP="${session}" \
    MASTER_PORT="${master_port}" HOLOSOMA_PROVENANCE_MASTER_PORT="${provenance_port}" \
    FAKE_TMUX_POST_CREATE_FAILURE="${mode}" bash batch_ne.sh launch
  local state prefix token
  state=$(lifecycle_active_path "${session}")
  prefix="${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.${session}"
  [[ "$(cut -f2 "${state}")" == rolling_back ]] ||
    fail "${label} rollback was falsely marked rolled_back"
  [[ -f "${prefix}.session" ]] || fail "${label} rollback killed the unconfirmed session"
  case "${mode}" in
    empty-environment)
      [[ -f "${prefix}.env_token" && ! -s "${prefix}.env_token" \
            && -f "${prefix}.env_command_sha256" && ! -s "${prefix}.env_command_sha256" \
            && -f "${prefix}.env_launch_epoch" && ! -s "${prefix}.env_launch_epoch" ]] ||
        fail "${label} fixture did not retain three explicitly empty environment fields"
      ;;
    mismatched-environment)
      [[ "$(cat "${prefix}.env_token")" == mismatched-atomic-token ]] ||
        fail "${label} fixture lost its mismatched atomic token"
      ;;
    empty-option)
      [[ -f "${prefix}.token" && ! -s "${prefix}.token" ]] ||
        fail "${label} fixture lost its explicitly empty option"
      ;;
    removed-environment)
      [[ -f "${prefix}.env_token_removed" \
            && -f "${prefix}.env_command_sha256_removed" \
            && -f "${prefix}.env_launch_epoch_removed" ]] ||
        fail "${label} fixture lost its explicit removed-environment markers"
      ;;
    identity-read-failure)
      [[ -f "${prefix}.identity_read_failure" ]] ||
        fail "${label} fixture lost its forced tmux read-failure marker"
      ;;
    *)
      fail "unsupported corrupt identity test mode: ${mode}"
      ;;
  esac
  token=$(cut -f6 "${state}")
  assert_quarantined_rendezvous_for_token "${token}" "${label}"
  clear_quarantined_rendezvous
  rm -f "${prefix}.session" "${prefix}.token" "${prefix}.command_sha256" \
    "${prefix}.launch_epoch" "${prefix}.env_token" "${prefix}.env_command_sha256" \
    "${prefix}.env_launch_epoch" "${prefix}.env_token_removed" \
    "${prefix}.env_command_sha256_removed" "${prefix}.env_launch_epoch_removed" \
    "${prefix}.identity_read_failure"
}

# Empty values are present and corrupt, not absent legacy identity.  Likewise,
# one mismatched atomic field or one explicitly empty option must fail closed.
run_corrupt_post_create_identity_case \
  empty-environment post-create-empty-env 31009 31010 'empty atomic-environment post-create failure'
run_corrupt_post_create_identity_case \
  mismatched-environment post-create-mismatched-env 31011 31012 'mismatched atomic-environment post-create failure'
run_corrupt_post_create_identity_case \
  empty-option post-create-empty-option 31013 31014 'empty ownership-option post-create failure'
run_corrupt_post_create_identity_case \
  removed-environment post-create-removed-env 31019 31020 'removed atomic-environment post-create failure'
run_corrupt_post_create_identity_case \
  identity-read-failure post-create-identity-read-failure 31021 31022 'tmux identity read-failure post-create failure'

run_active_cleanup_corruption_case() {
  local mode="$1" session="$2" main_port="$3" provenance_port="$4"
  local state prefix token launch_pid other_token
  state=$(lifecycle_active_path "${session}")
  prefix="${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.${session}"
  "${LIFECYCLE_ENV[@]}" SESSION="${session}" RUN_STAMP="${session}" \
    MASTER_PORT="${main_port}" HOLOSOMA_PROVENANCE_MASTER_PORT="${provenance_port}" \
    LAUNCH_STARTUP_TIMEOUT_SECONDS=5 LAUNCH_STARTUP_PROBE_TIMEOUT_SECONDS=1 \
    FAKE_TMUX_STARTUP_MODE=boundary-only \
    bash batch_ne.sh launch >"${TMP_DIR}/${session}.out" 2>&1 &
  launch_pid=$!
  for ((state_wait = 0; state_wait < FIXTURE_GATE_MAX_POLLS; state_wait++)); do
    [[ -f "${state}" && "$(cut -f2 "${state}" 2>/dev/null || true)" == running ]] && break
    sleep 0.025
  done
  [[ -f "${state}" && "$(cut -f2 "${state}")" == running ]] || {
    kill "${launch_pid}" 2>/dev/null || true
    wait "${launch_pid}" 2>/dev/null || true
    fail "${mode} active cleanup fixture never reached running"
  }
  token=$(cut -f6 "${state}")
  case "${mode}" in
    missing)
      rm -f "${state}"
      ;;
    other-token)
      other_token=$(printf '%s' "${session}-other-token" | sha256sum | awk '{print $1}')
      awk -F '\t' -v OFS='\t' -v token="${other_token}" '{$6=token; print}' \
        "${state}" >"${state}.corrupt"
      mv "${state}.corrupt" "${state}"
      ;;
    malformed)
      printf 'extra\tactive\trecord\n' >>"${state}"
      ;;
    *) fail "unsupported active cleanup corruption mode: ${mode}" ;;
  esac
  if wait "${launch_pid}"; then
    fail "${mode} active corruption unexpectedly allowed startup acceptance"
  fi
  grep -F 'Preserving owned rendezvous reservations as quarantine' \
    "${TMP_DIR}/${session}.out" >/dev/null ||
    fail "${mode} active corruption did not quarantine rendezvous reservations"
  assert_quarantined_rendezvous_for_token "${token}" "${mode} active metadata corruption"
  [[ -f "${prefix}.session" ]] ||
    fail "${mode} active corruption caused an ownership-unproven tmux kill"
  clear_quarantined_rendezvous
  rm -f "${prefix}.session" "${prefix}.token" "${prefix}.command_sha256" \
    "${prefix}.launch_epoch" "${prefix}.env_token" \
    "${prefix}.env_command_sha256" "${prefix}.env_launch_epoch"
}

# Missing, cross-token, or appended active metadata can never be treated as a
# completed rollback merely because tmux cleanup happened to report absence.
run_active_cleanup_corruption_case missing active-cleanup-missing 31231 31232
run_active_cleanup_corruption_case other-token active-cleanup-other-token 31233 31234
run_active_cleanup_corruption_case malformed active-cleanup-malformed 31235 31236

# A successful non-dry launch publishes one running v2 record whose command
# digest is identical in active metadata, atomic tmux environment, tmux
# options, and the log binding.
"${LIFECYCLE_ENV[@]}" SESSION=successful-lifecycle RUN_STAMP=successful-lifecycle \
  MASTER_PORT=30991 HOLOSOMA_PROVENANCE_MASTER_PORT=30992 \
  bash batch_ne.sh launch >"${TMP_DIR}/successful_lifecycle.out"
successful_state=$(lifecycle_active_path successful-lifecycle)
IFS=$'\t' read -r successful_version successful_phase successful_snapshot successful_log_dir \
  successful_target successful_token successful_command_sha successful_epoch <"${successful_state}"
[[ "${successful_version}" == 2 && "${successful_phase}" == running ]] ||
  fail 'successful launch did not publish running v2 active metadata'
[[ "${successful_token}" =~ ^[0-9a-f]{64}$ && "${successful_command_sha}" =~ ^[0-9a-f]{64}$ ]] ||
  fail 'running v2 active metadata lacks exact token/command SHA256'
successful_tmux_prefix="${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.successful-lifecycle"
[[ "$(cat "${successful_tmux_prefix}.token")" == "${successful_token}" ]] ||
  fail 'tmux ownership token differs from v2 active metadata'
[[ "$(cat "${successful_tmux_prefix}.command_sha256")" == "${successful_command_sha}" ]] ||
  fail 'tmux command SHA256 differs from v2 active metadata'
[[ "$(cat "${successful_tmux_prefix}.launch_epoch")" == "${successful_epoch}" ]] ||
  fail 'tmux launch epoch differs from v2 active metadata'
[[ "$(cat "${successful_tmux_prefix}.env_token")" == "${successful_token}" ]] ||
  fail 'atomic tmux environment token differs from v2 active metadata'
[[ "$(cat "${successful_tmux_prefix}.env_command_sha256")" == "${successful_command_sha}" ]] ||
  fail 'atomic tmux environment command SHA256 differs from v2 active metadata'
[[ "$(cat "${successful_tmux_prefix}.env_launch_epoch")" == "${successful_epoch}" ]] ||
  fail 'atomic tmux environment launch epoch differs from v2 active metadata'
successful_log="${ROLLBACK_REMOTE_ROOT}/${successful_snapshot}/${successful_log_dir}/node_0_${LIFECYCLE_NODE}.log"
successful_binding="HOLOSOMA_LAUNCH_BINDING token=${successful_token} command_sha256=${successful_command_sha} launch_epoch=${successful_epoch}"
grep -Fx "${successful_binding}" "${successful_log}" >/dev/null ||
  fail 'launch log is not bound to the active token/command/epoch'
[[ "$(list_rendezvous_endpoint_states | wc -l)" -eq 2 ]] ||
  fail 'successful launch did not retain both rendezvous reservations'
grep -F 'Startup-health handshake passed for all 1 node(s)' \
  "${TMP_DIR}/successful_lifecycle.out" >/dev/null ||
  fail 'successful launch returned without the bounded startup-health handshake'

# Status may consume only the one regular log selected by active metadata.
# A same-path symlink must not redirect health checks to mutable bytes.
mv -T "${successful_log}" "${successful_log}.regular"
ln -s "$(basename "${successful_log}.regular")" "${successful_log}"
expect_failure \
  "${TMP_DIR}/successful_symlink_log_status.out" \
  'unique active log candidate is non-regular or symlinked' \
  "${LIFECYCLE_ENV[@]}" SESSION=successful-lifecycle \
  MASTER_PORT=30991 HOLOSOMA_PROVENANCE_MASTER_PORT=30992 bash batch_ne.sh status
rm -f -- "${successful_log}"
mv -T "${successful_log}.regular" "${successful_log}"

status_root="${ROLLBACK_REMOTE_ROOT}/.status"
status_root_backup="${ROLLBACK_REMOTE_ROOT}/.status.contract-backup"
status_root_target="${ROLLBACK_REMOTE_ROOT}/.status.contract-target"
rm -rf -- "${status_root_backup}" "${status_root_target}"
if [[ -e "${status_root}" || -L "${status_root}" ]]; then
  mv -T "${status_root}" "${status_root_backup}"
fi
mkdir -- "${status_root_target}"
ln -s "$(basename "${status_root_target}")" "${status_root}"
expect_failure \
  "${TMP_DIR}/successful_symlink_status_root.out" \
  'status progress root is non-directory or symlinked' \
  "${LIFECYCLE_ENV[@]}" SESSION=successful-lifecycle \
  MASTER_PORT=30991 HOLOSOMA_PROVENANCE_MASTER_PORT=30992 bash batch_ne.sh status
rm -f -- "${status_root}"
rm -rf -- "${status_root_target}"
if [[ -e "${status_root_backup}" ]]; then
  mv -T "${status_root_backup}" "${status_root}"
else
  mkdir -- "${status_root}"
fi

status_state="${status_root}/successful-lifecycle_${LIFECYCLE_NODE}.state"
printf 'must-not-be-followed\n' >"${status_state}.target"
ln -s "$(basename "${status_state}.target")" "${status_state}"
expect_failure \
  "${TMP_DIR}/successful_symlink_status_state.out" \
  'status progress state is non-regular or symlinked' \
  "${LIFECYCLE_ENV[@]}" SESSION=successful-lifecycle \
  MASTER_PORT=30991 HOLOSOMA_PROVENANCE_MASTER_PORT=30992 bash batch_ne.sh status
rm -f -- "${status_state}" "${status_state}.target"

# Persisted status progress is untrusted metadata.  A huge future change time
# must be reset before any subtraction instead of wrapping into a stale result.
printf 'HOLOSOMA_PROGRESS completed_iteration=1\n' >>"${successful_log}"
printf '%s %s 1 %s\n' "${successful_token}" "${successful_log}" \
  99999999999999999999999999999999999999999999999999 \
  >"${status_state}"
"${LIFECYCLE_ENV[@]}" SESSION=successful-lifecycle RUN_STAMP=successful-lifecycle \
  MASTER_PORT=30991 HOLOSOMA_PROVENANCE_MASTER_PORT=30992 \
  bash batch_ne.sh status >"${TMP_DIR}/successful_huge_previous_change_status.out"
grep -F 'ignoring malformed prior status progress metadata' \
  "${TMP_DIR}/successful_huge_previous_change_status.out" >/dev/null ||
  fail 'status did not reject huge future previous_change before subtraction'
[[ -f "${status_state}" && ! -L "${status_state}" \
      && "$(awk 'END {print NR}' "${status_state}")" == 1 \
      && "$(awk 'NR == 1 {print NF}' "${status_state}")" == 4 \
      && "$(awk 'NR == 1 {print $1}' "${status_state}")" == "${successful_token}" \
      && "$(awk 'NR == 1 {print $3}' "${status_state}")" == 1 ]] ||
  fail 'status did not atomically replace malformed progress metadata with one canonical four-field record'

# Training-only environment drift must not block the control-plane status
# path; active v2 metadata remains authoritative for target/log identity.
"${LIFECYCLE_ENV[@]}" SESSION=successful-lifecycle RUN_STAMP='../../stale' \
  MASTER_PORT=30991 HOLOSOMA_PROVENANCE_MASTER_PORT=30992 \
  LOGGER_BASE_DIR=/outside RESUME_TRAINING_CKPT=/missing-resume.pt \
  TEACHER_CHECKPOINT=/missing-teacher.pt BOX_POLICY_INIT_REF=/missing-box.pt \
  STUDENT_POLICY_TYPE=unsupported STUDENT_FLOW_STEPS="${HUGE_UNSIGNED_DECIMAL}" \
  TARGET_LEARNING_ITERATION=not-an-integer NUM_MINI_BATCHES=invalid \
  NCCL_LIB_SHA256=invalid PYTHONHASHSEED=invalid \
  bash batch_ne.sh status >"${TMP_DIR}/control_only_status.out"
grep -F "active_state=${successful_state}" "${TMP_DIR}/control_only_status.out" >/dev/null ||
  fail 'control-only status did not reach exact active lifecycle metadata'
# Status must consume the same unique binding/completion evidence contract as
# clean-success release; head/tail selection cannot hide conflicting markers.
printf '%s\n' "${successful_binding}" >>"${successful_log}"
expect_failure \
  "${TMP_DIR}/successful_duplicate_binding_status.out" \
  'duplicate or launch-conflicting binding records' \
  "${LIFECYCLE_ENV[@]}" SESSION=successful-lifecycle RUN_STAMP=successful-lifecycle \
  MASTER_PORT=30991 HOLOSOMA_PROVENANCE_MASTER_PORT=30992 \
  bash batch_ne.sh status
sed -i '$d' "${successful_log}"
successful_model_iteration=$(printf '%05d' "${successful_target}")
printf 'HOLOSOMA_RUN_COMPLETE target_iteration=%s checkpoint=/tmp/model_99999.pt\n' \
  "${successful_target}" >>"${successful_log}"
expect_failure \
  "${TMP_DIR}/successful_noncanonical_completion_status.out" \
  'duplicate or non-canonical completion records' \
  "${LIFECYCLE_ENV[@]}" SESSION=successful-lifecycle RUN_STAMP=successful-lifecycle \
  MASTER_PORT=30991 HOLOSOMA_PROVENANCE_MASTER_PORT=30992 \
  bash batch_ne.sh status
sed -i '$d' "${successful_log}"
printf 'HOLOSOMA_RUN_COMPLETE target_iteration=%s checkpoint=/tmp/model_%s.pt\n' \
  "${successful_target}" "${successful_model_iteration}" >>"${successful_log}"
printf 'HOLOSOMA_RUN_COMPLETE target_iteration=%s checkpoint=/tmp/model_%s.pt\n' \
  "${successful_target}" "${successful_model_iteration}" >>"${successful_log}"
expect_failure \
  "${TMP_DIR}/successful_duplicate_completion_status.out" \
  'duplicate or non-canonical completion records' \
  "${LIFECYCLE_ENV[@]}" SESSION=successful-lifecycle RUN_STAMP=successful-lifecycle \
  MASTER_PORT=30991 HOLOSOMA_PROVENANCE_MASTER_PORT=30992 \
  bash batch_ne.sh status
sed -i '$d' "${successful_log}"
sed -i '$d' "${successful_log}"
[[ -f "${successful_tmux_prefix}.session" ]] ||
  fail 'strict duplicate evidence status checks mutated the running session'
# A valid v2 record stored under this session hash must still carry this
# session's log namespace before stop can treat it as ownership evidence.
cp "${successful_state}" "${successful_state}.namespace-backup"
(umask 077; awk -F '\t' -v OFS='\t' \
  '{$4="logs/batch_ne/different-session_contract"; print}' \
  "${successful_state}.namespace-backup" >"${successful_state}.namespace-incoming")
mv -T "${successful_state}.namespace-incoming" "${successful_state}"
[[ "$(stat -c %a -- "${successful_state}")" == 600 ]] ||
  fail 'wrong-namespace fixture accidentally changed the active-state mode'
expect_failure \
  "${TMP_DIR}/successful_lifecycle_wrong_namespace_stop.out" \
  'exact v2 active metadata is unavailable' \
  "${LIFECYCLE_ENV[@]}" SESSION=successful-lifecycle RUN_STAMP=successful-lifecycle \
  MASTER_PORT=30991 HOLOSOMA_PROVENANCE_MASTER_PORT=30992 \
  bash batch_ne.sh stop
[[ -f "${successful_tmux_prefix}.session" ]] ||
  fail 'wrong-session active namespace caused stop to kill the running owner'
mv -T "${successful_state}.namespace-backup" "${successful_state}"
# Stop must recover the original master/port topology from the immutable,
# hash-bound control script before mutating any node.
expect_failure \
  "${TMP_DIR}/successful_lifecycle_wrong_ports_stop.out" \
  'embedded session/log/topology/master/ports do not match this controller invocation' \
  "${LIFECYCLE_ENV[@]}" SESSION=successful-lifecycle RUN_STAMP=successful-lifecycle \
  MASTER_PORT=31991 HOLOSOMA_PROVENANCE_MASTER_PORT=31992 \
  bash batch_ne.sh stop
[[ -f "${successful_tmux_prefix}.session" && "$(cut -f2 "${successful_state}")" == running ]] ||
  fail 'wrong-port explicit stop mutated the running launch before topology validation'
# Periodic status is a scientific-health check for a session created by this
# launcher, so deleting all three atomic fields must be unhealthy rather than
# silently downgrading to legacy.  Teardown below may still use exact options
# to stop this now-legacy-shaped session safely.
rm -f "${successful_tmux_prefix}.env_token" \
  "${successful_tmux_prefix}.env_command_sha256" \
  "${successful_tmux_prefix}.env_launch_epoch"
expect_failure \
  "${TMP_DIR}/successful_lifecycle_removed_env_status.out" \
  'tmux atomic environment/options ownership does not match v2 active metadata' \
  "${LIFECYCLE_ENV[@]}" SESSION=successful-lifecycle RUN_STAMP=successful-lifecycle \
  MASTER_PORT=30991 HOLOSOMA_PROVENANCE_MASTER_PORT=30992 \
  bash batch_ne.sh status
[[ -f "${successful_tmux_prefix}.session" ]] ||
  fail 'strict status check killed a session after atomic identity removal'
"${LIFECYCLE_ENV[@]}" SESSION=successful-lifecycle RUN_STAMP='../../stale-stop' \
  MASTER_PORT=30991 HOLOSOMA_PROVENANCE_MASTER_PORT=30992 \
  LOGGER_BASE_DIR=/outside RESUME_TRAINING_CKPT=/missing-resume.pt \
  TEACHER_CHECKPOINT=/missing-teacher.pt BOX_POLICY_INIT_REF=/missing-box.pt \
  RESUME_FROM_BOX=not-a-boolean STUDENT_POLICY_TYPE=unsupported \
  STUDENT_FLOW_STEPS="${HUGE_UNSIGNED_DECIMAL}" \
  TARGET_LEARNING_ITERATION=not-an-integer NUM_MINI_BATCHES=invalid \
  NCCL_LIB_SHA256=invalid PYTHONHASHSEED=invalid \
  bash batch_ne.sh stop >"${TMP_DIR}/successful_stop.out"
[[ "$(cut -f2 "${successful_state}")" == stopped ]] ||
  fail 'explicit stop did not publish the stopped lifecycle phase'
[[ ! -f "${successful_tmux_prefix}.session" ]] ||
  fail 'explicit stop left the owned tmux session alive'
if [[ -n "$(list_rendezvous_endpoint_states)" ]]; then
  fail 'explicit stop leaked rendezvous reservations'
fi

# Namespace prefix similarity is not ownership.  A hash-bound foo_bar
# control script copied under foo's active-state key must fail the embedded
# exact SESSION check before tmux or lifecycle mutation.
"${LIFECYCLE_ENV[@]}" SESSION=foo_bar RUN_STAMP=foo_bar \
  MASTER_PORT=31259 HOLOSOMA_PROVENANCE_MASTER_PORT=31260 \
  bash batch_ne.sh launch >"${TMP_DIR}/session_prefix_owner_launch.out"
foo_bar_state=$(lifecycle_active_path foo_bar)
foo_state=$(lifecycle_active_path foo)
cp "${foo_bar_state}" "${foo_state}"
foo_state_sha=$(sha256sum "${foo_state}" | awk '{print $1}')
expect_failure \
  "${TMP_DIR}/session_prefix_false_owner_stop.out" \
  'embedded session/log/topology/master/ports do not match this controller invocation' \
  "${LIFECYCLE_ENV[@]}" SESSION=foo RUN_STAMP=foo \
  MASTER_PORT=31259 HOLOSOMA_PROVENANCE_MASTER_PORT=31260 bash batch_ne.sh stop
[[ "$(sha256sum "${foo_state}" | awk '{print $1}')" == "${foo_state_sha}" ]] ||
  fail 'foo stop mutated the copied foo_bar active identity'
[[ -f "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.foo_bar.session" ]] ||
  fail 'foo stop killed the distinct foo_bar session'
rm -f -- "${foo_state}"
"${LIFECYCLE_ENV[@]}" SESSION=foo_bar RUN_STAMP=foo_bar \
  MASTER_PORT=31259 HOLOSOMA_PROVENANCE_MASTER_PORT=31260 \
  bash batch_ne.sh stop >"${TMP_DIR}/session_prefix_owner_stop.out"

# Explicit stop must prove the full original ordered node membership.  A
# subset or reordered NODE_LIST cannot stop rank0 and release shared ports
# while workers from omitted ranks remain alive.
TOPOLOGY_NODE_B=lifecycle-node-b
TOPOLOGY_ENV=(
  "${ROLLBACK_ENV[@]}"
  NODES="${LIFECYCLE_NODE} ${TOPOLOGY_NODE_B}"
  MASTER_ADDR="${LIFECYCLE_NODE}"
  NNODES=2
  SESSION=topology-stop
  RUN_STAMP=topology-stop
  MASTER_PORT=31111
  HOLOSOMA_PROVENANCE_MASTER_PORT=31112
)
"${TOPOLOGY_ENV[@]}" bash batch_ne.sh launch >"${TMP_DIR}/topology_stop_launch.out"
topology_prefix_a="${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.topology-stop"
topology_prefix_b="${FAKE_TMUX_STATE_DIR}/${TOPOLOGY_NODE_B}.topology-stop"
[[ -f "${topology_prefix_a}.session" && -f "${topology_prefix_b}.session" ]] ||
  fail 'two-node topology fixture did not launch both sessions'
topology_session_sha=$(printf '%s' topology-stop | sha256sum | awk '{print $1}')
topology_state_a="${ROLLBACK_REMOTE_ROOT}/.active/${topology_session_sha}_${LIFECYCLE_NODE}.state"
topology_state_b="${ROLLBACK_REMOTE_ROOT}/.active/${topology_session_sha}_${TOPOLOGY_NODE_B}.state"
topology_token_a=$(cut -f6 "${topology_state_a}")
topology_token_b=$(cut -f6 "${topology_state_b}")
topology_epoch_a=$(cut -f8 "${topology_state_a}")
topology_epoch_b=$(cut -f8 "${topology_state_b}")
[[ "${topology_token_a}" == "${topology_token_b}" \
      && "${topology_epoch_a}" == "${topology_epoch_b}" ]] ||
  fail 'two-node shared-log fixture did not retain one launch token/epoch'
topology_log_rel=logs/batch_ne/topology-stop_topology-stop
topology_owner="${ROLLBACK_REMOTE_ROOT}/${rollback_snapshot_id}/${topology_log_rel}/.holosoma_launch_owner_v2"
topology_expected_owner=$(printf '2\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
  "${rollback_snapshot_id}" topology-stop "${topology_log_rel}" "${LIFECYCLE_TARGET}" 2 \
  "${topology_token_a}" "${topology_epoch_a}")
[[ -f "${topology_owner}" && ! -L "${topology_owner}" \
      && "$(awk 'END { print NR }' "${topology_owner}")" == 1 \
      && "$(awk -F '\t' 'NR == 1 { print NF }' "${topology_owner}")" == 8 \
      && "$(<"${topology_owner}")" == "${topology_expected_owner}" ]] ||
  fail 'two-node shared log directory lacks one exact immutable launch-owner marker'
expect_failure \
  "${TMP_DIR}/topology_subset_stop.out" \
  'Embedded topology does not prove complete ordered membership' \
  "${LIFECYCLE_ENV[@]}" SESSION=topology-stop RUN_STAMP=topology-stop \
  MASTER_PORT=31111 HOLOSOMA_PROVENANCE_MASTER_PORT=31112 \
  bash batch_ne.sh stop
[[ -f "${topology_prefix_a}.session" && -f "${topology_prefix_b}.session" ]] ||
  fail 'subset explicit stop mutated a two-node launch'
expect_failure \
  "${TMP_DIR}/topology_reordered_stop.out" \
  'embedded session/log/topology/master/ports do not match this controller invocation' \
  "${ROLLBACK_ENV[@]}" NODES="${TOPOLOGY_NODE_B} ${LIFECYCLE_NODE}" \
  MASTER_ADDR="${TOPOLOGY_NODE_B}" NNODES=2 SESSION=topology-stop RUN_STAMP=topology-stop \
  MASTER_PORT=31111 HOLOSOMA_PROVENANCE_MASTER_PORT=31112 \
  bash batch_ne.sh stop
[[ -f "${topology_prefix_a}.session" && -f "${topology_prefix_b}.session" ]] ||
  fail 'reordered explicit stop mutated the original ordered topology'
"${TOPOLOGY_ENV[@]}" bash batch_ne.sh stop >"${TMP_DIR}/topology_stop_success.out"
[[ ! -f "${topology_prefix_a}.session" && ! -f "${topology_prefix_b}.session" ]] ||
  fail 'full ordered topology stop left an owned session alive'
[[ -z "$(list_rendezvous_endpoint_states)" ]] ||
  fail 'full ordered topology stop leaked rendezvous endpoints'

# tmux teardown alone is insufficient: exact launch-identity descendants must
# be closed, while an unrelated process with a different token must survive.
"${LIFECYCLE_ENV[@]}" SESSION=process-stop RUN_STAMP=process-stop \
  MASTER_PORT=31113 HOLOSOMA_PROVENANCE_MASTER_PORT=31114 \
  bash batch_ne.sh launch >"${TMP_DIR}/process_stop_launch.out"
process_stop_state=$(lifecycle_active_path process-stop)
process_stop_token=$(cut -f6 "${process_stop_state}")
process_stop_command=$(cut -f7 "${process_stop_state}")
process_stop_epoch=$(cut -f8 "${process_stop_state}")
env HOLOSOMA_LAUNCH_TOKEN="${process_stop_token}" \
  HOLOSOMA_COMMAND_SHA256="${process_stop_command}" \
  HOLOSOMA_LAUNCH_EPOCH="${process_stop_epoch}" sleep 3600 &
owned_process_pid=$!
wait_for_fixture_process_token \
  "${owned_process_pid}" "${process_stop_token}" process-stop-owned
unrelated_process_token=$(printf unrelated-process-token | sha256sum | awk '{print $1}')
env HOLOSOMA_LAUNCH_TOKEN="${unrelated_process_token}" \
  HOLOSOMA_COMMAND_SHA256="${process_stop_command}" \
  HOLOSOMA_LAUNCH_EPOCH="${process_stop_epoch}" sleep 3600 &
unrelated_process_pid=$!
wait_for_fixture_process_token \
  "${unrelated_process_pid}" "${unrelated_process_token}" process-stop-unrelated
"${LIFECYCLE_ENV[@]}" SESSION=process-stop RUN_STAMP=process-stop \
  MASTER_PORT=31113 HOLOSOMA_PROVENANCE_MASTER_PORT=31114 \
  bash batch_ne.sh stop >"${TMP_DIR}/process_stop_success.out"
set +e
wait "${owned_process_pid}"
owned_process_rc=$?
set -e
(( owned_process_rc != 0 )) || fail 'explicit stop did not terminate the exact launch-identity process'
kill -0 "${unrelated_process_pid}" 2>/dev/null ||
  fail 'explicit stop terminated an unrelated different-token process'
kill "${unrelated_process_pid}" 2>/dev/null || true
wait "${unrelated_process_pid}" 2>/dev/null || true
[[ "$(cut -f2 "${process_stop_state}")" == stopped ]] ||
  fail 'process-closure stop did not publish stopped terminal metadata'
[[ -z "$(list_rendezvous_endpoint_states)" ]] ||
  fail 'process-closure stop leaked rendezvous endpoints'

# Pre-atomic r21 controls have no SESSION/RUN_STAMP/active-log or launch
# identity exports.  Their narrow compatibility stop is authorized by five
# explicit values and an exact options-only, single-pane tmux identity.  Use
# real local process trees here.  Every fake pane is launched in one unique,
# current-user cgroup-v2 scope whose name and placement match a real tmux
# server's tmux-spawn scope.  One worker deliberately calls setsid while
# retaining its exact parent chain inside that scope.
declare -A LEGACY_FIXTURE_CONTROL_BY_SESSION=()
declare -A LEGACY_FIXTURE_ROOT_BY_SESSION=()
declare -A LEGACY_FIXTURE_UNIT_BY_SESSION=()
declare -A LEGACY_FIXTURE_CGROUP_BY_SESSION=()
declare -A LEGACY_FIXTURE_START_BY_ROOT=()
legacy_fixture_counter=0
legacy_fixture_distill_installed=0
legacy_fixture_distill_backup=''
legacy_fixture_snapshot_root_mode=''

legacy_test_proc_identity() {
  local pid="$1" record tail ignored
  [[ -r "/proc/${pid}/stat" ]] || return 1
  IFS= read -r record <"/proc/${pid}/stat" || return 1
  tail=${record##*) }
  legacy_test_state='' legacy_test_ppid='' legacy_test_pgrp=''
  legacy_test_session='' legacy_test_start='' legacy_test_uid=''
  IFS=' ' read -r legacy_test_state legacy_test_ppid legacy_test_pgrp legacy_test_session \
    ignored ignored ignored ignored ignored ignored ignored ignored ignored \
    ignored ignored ignored ignored ignored ignored legacy_test_start ignored <<<"${tail}" || true
  legacy_test_uid=$(stat -c %u -- "/proc/${pid}" 2>/dev/null) || return 1
  [[ "${legacy_test_ppid}" =~ ^[0-9]+$ \
        && "${legacy_test_pgrp}" =~ ^[1-9][0-9]*$ \
        && "${legacy_test_session}" =~ ^[0-9]+$ \
        && "${legacy_test_start}" =~ ^[1-9][0-9]*$ \
        && "${legacy_test_uid}" =~ ^[0-9]+$ ]]
}

legacy_test_read_cgroup_identity() {
  local pid="$1" record_count
  local -a records=()
  legacy_fixture_cgroup_path=''
  mapfile -t records <"/proc/${pid}/cgroup" 2>/dev/null || return 1
  record_count=${#records[@]}
  (( record_count == 1 )) || return 1
  [[ "${records[0]}" =~ ^0::/ ]] || return 1
  legacy_fixture_cgroup_path=${records[0]#0::}
  legacy_fixture_cgroup_dir="/sys/fs/cgroup${legacy_fixture_cgroup_path}"
  [[ -d "${legacy_fixture_cgroup_dir}" && ! -L "${legacy_fixture_cgroup_dir}" ]] || return 1
  legacy_fixture_cgroup_dev=$(stat -c %d -- "${legacy_fixture_cgroup_dir}") || return 1
  legacy_fixture_cgroup_ino=$(stat -c %i -- "${legacy_fixture_cgroup_dir}") || return 1
  legacy_fixture_cgroup_uid=$(stat -c %u -- "${legacy_fixture_cgroup_dir}") || return 1
}

legacy_test_read_cgroup_events() {
  local cgroup_dir="$1" key value extra
  legacy_test_cgroup_populated=''
  legacy_test_cgroup_frozen=''
  while read -r key value extra; do
    [[ -z "${extra}" ]] || return 1
    case "${key}" in
      populated) legacy_test_cgroup_populated=${value} ;;
      frozen) legacy_test_cgroup_frozen=${value} ;;
    esac
  done <"${cgroup_dir}/cgroup.events"
  [[ "${legacy_test_cgroup_populated}" =~ ^[01]$ \
        && "${legacy_test_cgroup_frozen}" =~ ^[01]$ ]]
}

legacy_test_wait_cgroup_frozen() {
  local cgroup_dir="$1" expected="$2" poll
  for ((poll = 0; poll < 200; poll++)); do
    if legacy_test_read_cgroup_events "${cgroup_dir}" \
        && [[ "$(<"${cgroup_dir}/cgroup.freeze")" == "${expected}" \
          && "${legacy_test_cgroup_frozen}" == "${expected}" ]]; then
      return 0
    fi
    sleep 0.01
  done
  return 1
}

legacy_test_assert_cgroup_terminal() {
  local cgroup_path="$1" expected_dev="$2" expected_ino="$3" label="$4"
  local cgroup_dir="/sys/fs/cgroup${cgroup_path}"
  if [[ ! -e "${cgroup_dir}" && ! -L "${cgroup_dir}" ]]; then
    return 0
  fi
  [[ -d "${cgroup_dir}" && ! -L "${cgroup_dir}" \
        && "$(stat -c %d -- "${cgroup_dir}")" == "${expected_dev}" \
        && "$(stat -c %i -- "${cgroup_dir}")" == "${expected_ino}" ]] ||
    fail "${label} cgroup path was replaced before terminal verification"
  legacy_test_read_cgroup_events "${cgroup_dir}" ||
    fail "${label} cgroup has malformed terminal events"
  [[ "${legacy_test_cgroup_populated}" == 0 ]] ||
    fail "${label} cgroup remains populated after cgroup.kill"
}

cleanup_legacy_fixture_unit() {
  local unit="$1" cgroup_path cgroup_dir
  [[ "${unit}" =~ ^tmux-spawn-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.scope$ ]] || return 0
  cgroup_path=$(systemctl --user show --property=ControlGroup --value "${unit}" 2>/dev/null || true)
  if [[ "${cgroup_path}" == "/user.slice/user-$(id -u).slice/user@$(id -u).service/${unit}" ]]; then
    cgroup_dir="/sys/fs/cgroup${cgroup_path}"
    if [[ -d "${cgroup_dir}" && ! -L "${cgroup_dir}" ]]; then
      printf '0\n' >"${cgroup_dir}/cgroup.freeze" 2>/dev/null || true
      printf '1\n' >"${cgroup_dir}/cgroup.kill" 2>/dev/null || true
    fi
  fi
  systemctl --user stop "${unit}" >/dev/null 2>&1 || true
  systemctl --user reset-failed "${unit}" >/dev/null 2>&1 || true
}

collect_legacy_fixture_tree() {
  local root="$1" pid ppid added
  local -A included=()
  legacy_fixture_tree_pids=()
  [[ "${root}" =~ ^[1-9][0-9]*$ ]] || return 0
  included[${root}]=1
  legacy_fixture_tree_pids+=("${root}")
  while :; do
    added=0
    while read -r pid ppid; do
      [[ "${pid}" =~ ^[1-9][0-9]*$ && "${ppid}" =~ ^[0-9]+$ ]] || continue
      [[ -z "${included[${pid}]+x}" && -n "${included[${ppid}]+x}" ]] || continue
      included[${pid}]=1
      legacy_fixture_tree_pids+=("${pid}")
      added=1
    done < <(ps -eo pid=,ppid=)
    (( added != 0 )) || break
  done
}

terminate_legacy_fixture_tree() {
  local root="$1" index pid expected_start
  expected_start=${LEGACY_FIXTURE_START_BY_ROOT[${root}]:-}
  if [[ -n "${expected_start}" ]]; then
    if ! legacy_test_proc_identity "${root}"; then
      wait "${root}" 2>/dev/null || true
      return 0
    fi
    [[ "${legacy_test_start}" == "${expected_start}" ]] || return 0
  fi
  collect_legacy_fixture_tree "${root}"
  for ((index = ${#legacy_fixture_tree_pids[@]} - 1; index >= 0; index--)); do
    pid=${legacy_fixture_tree_pids[${index}]}
    kill -CONT "${pid}" 2>/dev/null || true
    kill -TERM "${pid}" 2>/dev/null || true
  done
  sleep 0.02
  for ((index = ${#legacy_fixture_tree_pids[@]} - 1; index >= 0; index--)); do
    kill -KILL "${legacy_fixture_tree_pids[${index}]}" 2>/dev/null || true
  done
  wait "${root}" 2>/dev/null || true
}

make_legacy_stop_fixture() {
  local session="$1" main_port="$2" provenance_port="$3" layout="${4:-valid}"
  local topology="${5:-closed}"
  local control_dir control_incoming control_sha control log_dir absolute_log log_file
  local prefix active session_sha created_at poll root_session pid distill_fixture
  local current_uid expected_cgroup_path reparent_parent
  legacy_fixture_counter=$((legacy_fixture_counter + 1))
  legacy_fixture_session=${session}
  legacy_fixture_run_stamp="${session}-stamp"
  legacy_fixture_token=$(printf 'legacy-stop-token:%s' "${session}" | sha256sum | awk '{print $1}')
  legacy_fixture_epoch=$((1700000000 + legacy_fixture_counter))
  legacy_fixture_target=${LIFECYCLE_TARGET}
  legacy_fixture_main_port=${main_port}
  legacy_fixture_provenance_port=${provenance_port}
  legacy_fixture_log_dir="logs/batch_ne/${session}_${legacy_fixture_run_stamp}"
  absolute_log="${ROLLBACK_REMOTE_ROOT}/${rollback_snapshot_id}/${legacy_fixture_log_dir}"
  log_file="${absolute_log}/node_0_${LIFECYCLE_NODE}.log"
  mkdir -p "${absolute_log}"
  distill_fixture="${ROLLBACK_REMOTE_ROOT}/${rollback_snapshot_id}/distill_as_button_solid.sh"
  if (( legacy_fixture_distill_installed == 0 )); then
    legacy_fixture_snapshot_root_mode=$(stat -c %a \
      "${ROLLBACK_REMOTE_ROOT}/${rollback_snapshot_id}")
    legacy_fixture_distill_backup="${distill_fixture}.legacy-test-backup"
    chmod u+w "${ROLLBACK_REMOTE_ROOT}/${rollback_snapshot_id}"
    cp -p "${distill_fixture}" "${legacy_fixture_distill_backup}"
    chmod u+w "${distill_fixture}"
    cat >"${distill_fixture}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
bash -c '
if [[ "${LEGACY_FIXTURE_REPARENT_EXTRA:-0}" == 1 ]]; then
  # The intermediate shell exits immediately.  Its sleep stays inside the
  # inherited tmux-spawn scope but is reparented outside the pane PPID tree.
  bash -c "sleep 3600 &"
fi
setsid sleep 3600 &
sleep 3600 &
wait
'
EOF
    chmod 0500 "${distill_fixture}"
    legacy_fixture_distill_installed=1
  fi
  control_dir="${ROLLBACK_REMOTE_ROOT}/${rollback_snapshot_id}/.run_control"
  mkdir -p "${control_dir}"
  control_incoming="${control_dir}/.legacy-${session}.incoming"
  cat >"${control_incoming}" <<EOF
set -euo pipefail
cd ${ROLLBACK_REMOTE_ROOT}/${rollback_snapshot_id}
mkdir -p ${absolute_log}
export HOLOSOMA_SOURCE_SNAPSHOT_ID=${rollback_snapshot_id}
export HOLOSOMA_SOURCE_MANIFEST_SHA256=${rollback_snapshot_manifest_sha}
export NPROC=1
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=${LIFECYCLE_NODE}
export MASTER_PORT=${main_port}
export HOLOSOMA_PROVENANCE_MASTER_PORT=${provenance_port}
export TARGET_LEARNING_ITERATION=${LIFECYCLE_TARGET}
EOF
  if [[ "${layout}" == partial ]]; then
    printf 'export SESSION=%s\n' "${session}" >>"${control_incoming}"
  elif [[ "${layout}" == modern ]]; then
    printf 'export SESSION=%s\nexport RUN_STAMP=%s\nexport HOLOSOMA_ACTIVE_LOG_DIR=%s\n' \
      "${session}" "${legacy_fixture_run_stamp}" "${legacy_fixture_log_dir}" \
      >>"${control_incoming}"
  elif [[ "${layout}" != valid ]]; then
    fail "unknown legacy fixture layout: ${layout}"
  fi
  cat >>"${control_incoming}" <<EOF
export RUN_NAME=${session}
echo "[INFO][${LIFECYCLE_NODE}] master=${LIFECYCLE_NODE}:${main_port} log=${log_file}"
TRAIN_EXTRA_ARGS=()
bash distill_as_button_solid.sh "\${TRAIN_EXTRA_ARGS[@]}" 2>&1 | tee -a ${log_file}
EOF
  control_sha=$(sha256sum "${control_incoming}" | awk '{print $1}')
  control="${control_dir}/train-${control_sha}.sh"
  mv -T "${control_incoming}" "${control}"
  chmod 0500 "${control}"
  case "${topology}" in
    closed) legacy_fixture_reparent_flag=0 ;;
    reparented) legacy_fixture_reparent_flag=1 ;;
    *) fail "unknown legacy fixture topology: ${topology}" ;;
  esac
  legacy_fixture_unit="tmux-spawn-$(cat /proc/sys/kernel/random/uuid).scope"
  LEGACY_FIXTURE_REPARENT_EXTRA="${legacy_fixture_reparent_flag}" \
    systemd-run --user --scope --slice=- --quiet \
      --unit="${legacy_fixture_unit}" setsid bash "${control}" \
      >"${TMP_DIR}/${session}.pane.out" 2>&1 &
  legacy_fixture_root_pid=$!
  LEGACY_FIXTURE_ROOT_PIDS+=("${legacy_fixture_root_pid}")
  LEGACY_FIXTURE_UNITS+=("${legacy_fixture_unit}")
  LEGACY_FIXTURE_CONTROL_BY_SESSION[${session}]=${control}
  LEGACY_FIXTURE_ROOT_BY_SESSION[${session}]=${legacy_fixture_root_pid}
  LEGACY_FIXTURE_UNIT_BY_SESSION[${session}]=${legacy_fixture_unit}
  for ((poll = 0; poll < 400; poll++)); do
    if legacy_test_proc_identity "${legacy_fixture_root_pid}"; then
      mapfile -d '' -t legacy_fixture_root_argv <"/proc/${legacy_fixture_root_pid}/cmdline" || true
      if (( ${#legacy_fixture_root_argv[@]} == 2 )) \
          && [[ "${legacy_fixture_root_argv[0]##*/}" == bash \
            && "${legacy_fixture_root_argv[1]}" == "${control}" ]]; then
        break
      fi
    fi
    sleep 0.005
  done
  (( poll < 400 )) || fail "legacy fixture ${session} did not expose exact pane argv"
  legacy_test_proc_identity "${legacy_fixture_root_pid}" ||
    fail "legacy fixture ${session} root disappeared"
  legacy_fixture_root_start=${legacy_test_start}
  LEGACY_FIXTURE_START_BY_ROOT[${legacy_fixture_root_pid}]=${legacy_fixture_root_start}
  legacy_fixture_root_uid=${legacy_test_uid}
  legacy_fixture_root_ppid=${legacy_test_ppid}
  legacy_fixture_root_pgrp=${legacy_test_pgrp}
  legacy_fixture_root_session=${legacy_test_session}
  root_session=${legacy_test_session}
  legacy_test_read_cgroup_identity "${legacy_fixture_root_pid}" ||
    fail "legacy fixture ${session} root lacks one exact cgroup-v2 identity"
  current_uid=$(id -u)
  expected_cgroup_path="/user.slice/user-${current_uid}.slice/user@${current_uid}.service/${legacy_fixture_unit}"
  [[ "${legacy_fixture_cgroup_path}" == "${expected_cgroup_path}" \
        && "${legacy_fixture_cgroup_uid}" == "${current_uid}" \
        && -r "${legacy_fixture_cgroup_dir}/cgroup.events" \
        && -w "${legacy_fixture_cgroup_dir}/cgroup.freeze" \
        && -w "${legacy_fixture_cgroup_dir}/cgroup.kill" ]] ||
    fail "legacy fixture ${session} is outside its exact writable current-user tmux-spawn scope"
  LEGACY_FIXTURE_CGROUP_BY_SESSION[${session}]=${legacy_fixture_cgroup_path}
  legacy_fixture_cgroup_fingerprint=$(printf '%s\0%s\0%s\0%s' \
    "${legacy_fixture_cgroup_path}" "${legacy_fixture_cgroup_dev}" \
    "${legacy_fixture_cgroup_ino}" "${legacy_fixture_cgroup_uid}" \
    | sha256sum | awk '{print $1}')
  legacy_test_read_cgroup_events "${legacy_fixture_cgroup_dir}" ||
    fail "legacy fixture ${session} cgroup events are malformed"
  [[ "${legacy_test_cgroup_frozen}" == 0 \
        && "${legacy_test_cgroup_populated}" == 1 ]] ||
    fail "legacy fixture ${session} did not start thawed and populated"
  legacy_fixture_setsid_pid=''
  legacy_fixture_setsid_start=''
  for ((poll = 0; poll < 400; poll++)); do
    collect_legacy_fixture_tree "${legacy_fixture_root_pid}"
    for pid in "${legacy_fixture_tree_pids[@]}"; do
      [[ "${pid}" != "${legacy_fixture_root_pid}" ]] || continue
      if legacy_test_proc_identity "${pid}" \
          && [[ "${legacy_test_session}" != "${root_session}" ]]; then
        legacy_fixture_setsid_pid=${pid}
        legacy_fixture_setsid_start=${legacy_test_start}
        break 2
      fi
    done
    sleep 0.005
  done
  [[ -n "${legacy_fixture_setsid_pid}" ]] ||
    fail "legacy fixture ${session} did not create a PPID descendant in a new session"
  legacy_fixture_reparented_pid=''
  legacy_fixture_reparented_start=''
  if [[ "${topology}" == reparented ]]; then
    for ((poll = 0; poll < 400; poll++)); do
      mapfile -t legacy_fixture_cgroup_pids \
        <"${legacy_fixture_cgroup_dir}/cgroup.procs" || true
      declare -A legacy_fixture_cgroup_pid_set=()
      for pid in "${legacy_fixture_cgroup_pids[@]}"; do
        legacy_fixture_cgroup_pid_set[${pid}]=1
      done
      for pid in "${legacy_fixture_cgroup_pids[@]}"; do
        [[ "${pid}" != "${legacy_fixture_root_pid}" ]] || continue
        if legacy_test_proc_identity "${pid}"; then
          reparent_parent=${legacy_test_ppid}
          if [[ -z "${legacy_fixture_cgroup_pid_set[${reparent_parent}]+x}" ]]; then
            legacy_fixture_reparented_pid=${pid}
            legacy_fixture_reparented_start=${legacy_test_start}
            break 2
          fi
        fi
      done
      unset legacy_fixture_cgroup_pid_set
      sleep 0.005
    done
    [[ -n "${legacy_fixture_reparented_pid}" ]] ||
      fail "legacy fixture ${session} did not retain a reparented member inside its cgroup"
  fi

  prefix="${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.${session}"
  rm -f "${prefix}."*
  : >"${prefix}.session"
  printf '%s\n' "${legacy_fixture_token}" >"${prefix}.token"
  printf '%s\n' "${control_sha}" >"${prefix}.command_sha256"
  printf '%s\n' "${legacy_fixture_epoch}" >"${prefix}.launch_epoch"
  printf '%s\n' "${legacy_fixture_root_pid}" >"${prefix}.pane_pid"
  session_sha=$(printf '%s' "${session}" | sha256sum | awk '{print $1}')
  active="${ROLLBACK_REMOTE_ROOT}/.active/${session_sha}_${LIFECYCLE_NODE}.state"
  mkdir -p "$(dirname "${active}")"
  printf '2\trunning\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${rollback_snapshot_id}" "${legacy_fixture_log_dir}" "${LIFECYCLE_TARGET}" \
    "${legacy_fixture_token}" "${control_sha}" "${legacy_fixture_epoch}" >"${active}"
  legacy_fixture_prefix=${prefix}
  legacy_fixture_active=${active}
  legacy_fixture_control=${control}
  legacy_fixture_command_sha=${control_sha}
  legacy_fixture_receipt_identity=$(printf 'legacy-process-v2-cgroup\t%s\t%s\t%s' \
    "${legacy_fixture_token}" "${control_sha}" "${legacy_fixture_epoch}" \
    | sha256sum | awk '{print $1}')
  legacy_fixture_receipt="${active}.legacy-processes.${legacy_fixture_receipt_identity}"
  clear_quarantined_rendezvous
  mkdir -p "${ROLLBACK_REMOTE_ROOT}/.rendezvous"
  created_at=$((legacy_fixture_epoch + 100))
  printf '2\t%s\t%s\t%s\t%s\n' \
    "${legacy_fixture_token}" "${session}" "${main_port}" "${created_at}" \
    >"$(rendezvous_endpoint_path "${main_port}")"
  printf '2\t%s\t%s\t%s\t%s\n' \
    "${legacy_fixture_token}" "${session}" "${provenance_port}" "${created_at}" \
    >"$(rendezvous_endpoint_path "${provenance_port}")"
}

cleanup_legacy_stop_fixture() {
  local session="$1" root active prefix control unit registered_pid registered_unit
  local -a retained_roots=()
  local -a retained_units=()
  root=${LEGACY_FIXTURE_ROOT_BY_SESSION[${session}]:-}
  unit=${LEGACY_FIXTURE_UNIT_BY_SESSION[${session}]:-}
  active=$(lifecycle_active_path "${session}")
  prefix="${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.${session}"
  control=${LEGACY_FIXTURE_CONTROL_BY_SESSION[${session}]:-}
  [[ -z "${unit}" ]] || cleanup_legacy_fixture_unit "${unit}"
  [[ -z "${root}" ]] || terminate_legacy_fixture_tree "${root}"
  rm -f "${prefix}."* "${active}" "${active}.legacy-processes."*
  [[ -z "${control}" ]] || rm -f "${control}"
  clear_quarantined_rendezvous
  for registered_pid in "${LEGACY_FIXTURE_ROOT_PIDS[@]}"; do
    [[ "${registered_pid}" == "${root}" ]] || retained_roots+=("${registered_pid}")
  done
  LEGACY_FIXTURE_ROOT_PIDS=("${retained_roots[@]}")
  [[ -z "${root}" ]] || unset 'LEGACY_FIXTURE_START_BY_ROOT['"${root}"']'
  for registered_unit in "${LEGACY_FIXTURE_UNITS[@]}"; do
    [[ "${registered_unit}" == "${unit}" ]] || retained_units+=("${registered_unit}")
  done
  LEGACY_FIXTURE_UNITS=("${retained_units[@]}")
  unset 'LEGACY_FIXTURE_ROOT_BY_SESSION['"${session}"']'
  unset 'LEGACY_FIXTURE_CONTROL_BY_SESSION['"${session}"']'
  unset 'LEGACY_FIXTURE_UNIT_BY_SESSION['"${session}"']'
  unset 'LEGACY_FIXTURE_CGROUP_BY_SESSION['"${session}"']'
}

legacy_stop_with_expected() {
  local session="$1" run_stamp="$2" token="$3" epoch="$4"
  local snapshot="$5" target="$6" main_port="$7" provenance_port="$8"
  shift 8
  "${LIFECYCLE_ENV[@]}" SESSION="${session}" RUN_STAMP="${run_stamp}" \
    MASTER_PORT="${main_port}" HOLOSOMA_PROVENANCE_MASTER_PORT="${provenance_port}" \
    LEGACY_STOP_EXPECTED_SNAPSHOT_ID="${snapshot}" \
    LEGACY_STOP_EXPECTED_TOKEN="${token}" LEGACY_STOP_EXPECTED_EPOCH="${epoch}" \
    LEGACY_STOP_EXPECTED_RUN_STAMP="${run_stamp}" \
    LEGACY_STOP_EXPECTED_TARGET="${target}" "$@" bash batch_ne.sh stop
}

make_legacy_stop_fixture legacy-missing 31301 31302 valid
chmod 0664 "${legacy_fixture_active}"
legacy_missing_active_sha=$(sha256sum "${legacy_fixture_active}" | awk '{print $1}')
[[ "$(stat -c %a -- "${legacy_fixture_active}")" == 664 ]] ||
  fail 'legacy missing-authorization fixture did not exercise historical mode 0664'
expect_failure \
  "${TMP_DIR}/legacy_stop_nonprivate_without_expected.out" \
  'Private lifecycle state mode is not 0600' \
  "${LIFECYCLE_ENV[@]}" SESSION=legacy-missing RUN_STAMP=legacy-missing-stamp \
  MASTER_PORT=31301 HOLOSOMA_PROVENANCE_MASTER_PORT=31302 bash batch_ne.sh stop
[[ "$(sha256sum "${legacy_fixture_active}" | awk '{print $1}')" == "${legacy_missing_active_sha}" \
      && "$(stat -c %a -- "${legacy_fixture_active}")" == 664 ]] ||
  fail 'legacy mode authorization refusal rewrote or migrated the active state'
[[ -f "${legacy_fixture_prefix}.session" && "$(cut -f2 "${legacy_fixture_active}")" == running ]] ||
  fail 'legacy stop without explicit authorization mutated its fixture'
kill -0 "${legacy_fixture_root_pid}" 2>/dev/null ||
  fail 'legacy stop without explicit authorization killed its pane'

# The launcher cannot distinguish a legacy body from corrupt modern content
# without reading it, so an unauthorized 0664 record must fail at metadata.
# Once the fixture is private, body validation may identify the legacy layout
# and issue the more specific five-field authorization error.
chmod 0600 "${legacy_fixture_active}"
expect_failure \
  "${TMP_DIR}/legacy_stop_missing_expected.out" \
  'Legacy control layout requires all five explicit LEGACY_STOP_EXPECTED_* fields' \
  "${LIFECYCLE_ENV[@]}" SESSION=legacy-missing RUN_STAMP=legacy-missing-stamp \
  MASTER_PORT=31301 HOLOSOMA_PROVENANCE_MASTER_PORT=31302 bash batch_ne.sh stop
[[ "$(sha256sum "${legacy_fixture_active}" | awk '{print $1}')" == "${legacy_missing_active_sha}" \
      && "$(stat -c %a -- "${legacy_fixture_active}")" == 600 \
      && -f "${legacy_fixture_prefix}.session" ]] ||
  fail 'private legacy authorization refusal mutated its lifecycle fixture'
cleanup_legacy_stop_fixture legacy-missing

# Five-field legacy authorization may authenticate a historical record, but it
# must not relax 0600 for a control script that declares the modern layout.
make_legacy_stop_fixture modern-nonprivate 31321 31322 modern
chmod 0664 "${legacy_fixture_active}"
modern_nonprivate_sha=$(sha256sum "${legacy_fixture_active}" | awk '{print $1}')
expect_failure \
  "${TMP_DIR}/modern_nonprivate_with_legacy_auth.out" \
  'Non-private active-state mode is supported only for an explicitly authorized legacy control layout' \
  legacy_stop_with_expected modern-nonprivate "${legacy_fixture_run_stamp}" \
  "${legacy_fixture_token}" "${legacy_fixture_epoch}" "${rollback_snapshot_id}" \
  "${legacy_fixture_target}" 31321 31322
[[ "$(sha256sum "${legacy_fixture_active}" | awk '{print $1}')" == "${modern_nonprivate_sha}" \
      && "$(stat -c %a -- "${legacy_fixture_active}")" == 664 \
      && -f "${legacy_fixture_prefix}.session" ]] ||
  fail 'non-private modern-layout refusal mutated or migrated its lifecycle fixture'
kill -0 "${legacy_fixture_root_pid}" 2>/dev/null ||
  fail 'non-private modern-layout refusal killed its pane'
cleanup_legacy_stop_fixture modern-nonprivate

make_legacy_stop_fixture legacy-partial 31303 31304 partial
expect_failure \
  "${TMP_DIR}/legacy_stop_partial_layout.out" \
  'exports form a partial layout' \
  legacy_stop_with_expected legacy-partial "${legacy_fixture_run_stamp}" \
  "${legacy_fixture_token}" "${legacy_fixture_epoch}" "${rollback_snapshot_id}" \
  "${legacy_fixture_target}" 31303 31304
[[ -f "${legacy_fixture_prefix}.session" && "$(cut -f2 "${legacy_fixture_active}")" == running ]] ||
  fail 'partial legacy layout mutated active/tmux state'
cleanup_legacy_stop_fixture legacy-partial

# A receipt body can never be recovered without the durable pre-freeze intent.
# A bounded, regular `.in` residue is recognized but must fail before either
# freezer or lifecycle mutation.
make_legacy_stop_fixture legacy-stale-receipt-in 31307 31308 valid
stale_receipt_in="${legacy_fixture_receipt}.in"
: >"${stale_receipt_in}"
chmod 0600 "${stale_receipt_in}"
expect_failure \
  "${TMP_DIR}/legacy_stop_stale_receipt_in.out" \
  'Receipt incoming residue lacks its durable exact freeze intent' \
  legacy_stop_with_expected "${legacy_fixture_session}" "${legacy_fixture_run_stamp}" \
  "${legacy_fixture_token}" "${legacy_fixture_epoch}" "${rollback_snapshot_id}" \
  "${legacy_fixture_target}" 31307 31308
[[ "$(cut -f2 "${legacy_fixture_active}")" == running \
      && -f "${legacy_fixture_prefix}.session" \
      && -f "${stale_receipt_in}" ]] ||
  fail 'orphan receipt residue mutated legacy lifecycle/tmux metadata'
legacy_test_wait_cgroup_frozen "${legacy_fixture_cgroup_dir}" 0 ||
  fail 'orphan receipt residue froze the exact legacy cgroup'
cleanup_legacy_stop_fixture legacy-stale-receipt-in

# An unpublished intent residue is safe to discard while the authenticated
# scope is still thawed.  The stop transaction must recapture from scratch and
# leave no intent/residue beside its terminal v2 receipt.
make_legacy_stop_fixture legacy-stale-intent-in 31309 31310 valid
stale_intent_receipt=${legacy_fixture_receipt}
stale_intent_in="${stale_intent_receipt}.freeze-intent.in"
stale_intent_cgroup_path=${legacy_fixture_cgroup_path}
stale_intent_cgroup_dev=${legacy_fixture_cgroup_dev}
stale_intent_cgroup_ino=${legacy_fixture_cgroup_ino}
: >"${stale_intent_in}"
chmod 0600 "${stale_intent_in}"
legacy_stop_with_expected "${legacy_fixture_session}" "${legacy_fixture_run_stamp}" \
  "${legacy_fixture_token}" "${legacy_fixture_epoch}" "${rollback_snapshot_id}" \
  "${legacy_fixture_target}" 31309 31310 \
  >"${TMP_DIR}/legacy_stop_stale_intent_in_success.out"
[[ "$(cut -f2 "${legacy_fixture_active}")" == stopped \
      && ! -f "${legacy_fixture_prefix}.session" \
      && -f "${stale_intent_receipt}" \
      && ! -e "${stale_intent_in}" \
      && ! -e "${stale_intent_receipt}.freeze-intent" \
      && ! -e "${stale_intent_receipt}.in" ]] ||
  fail 'thawed intent residue was not replaced by one exact terminal receipt'
legacy_test_assert_cgroup_terminal \
  "${stale_intent_cgroup_path}" "${stale_intent_cgroup_dev}" \
  "${stale_intent_cgroup_ino}" 'stale-intent-recapture'
cleanup_legacy_stop_fixture legacy-stale-intent-in

# A valid-looking receipt observed beside a thawed running scope is not an
# immutable membership snapshot.  Force a one-record stale receipt and prove
# the transaction replaces it with a freshly frozen full-scope capture.
make_legacy_stop_fixture legacy-thawed-receipt 31311 31312 valid
thawed_receipt=${legacy_fixture_receipt}
thawed_cgroup_path=${legacy_fixture_cgroup_path}
thawed_cgroup_dev=${legacy_fixture_cgroup_dev}
thawed_cgroup_ino=${legacy_fixture_cgroup_ino}
printf '2\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t1\n%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${legacy_fixture_token}" "${legacy_fixture_epoch}" \
  "${legacy_fixture_command_sha}" "${rollback_snapshot_id}" \
  "${legacy_fixture_log_dir}" "${legacy_fixture_target}" \
  "${legacy_fixture_root_pid}" "${legacy_fixture_root_start}" \
  "${legacy_fixture_cgroup_path}" "${legacy_fixture_cgroup_dev}" \
  "${legacy_fixture_cgroup_ino}" "${legacy_fixture_root_pid}" \
  "${legacy_fixture_root_start}" "${legacy_fixture_root_uid}" \
  "${legacy_fixture_root_ppid}" "${legacy_fixture_root_pgrp}" \
  "${legacy_fixture_root_session}" >"${thawed_receipt}"
chmod 0400 "${thawed_receipt}"
legacy_stop_with_expected "${legacy_fixture_session}" "${legacy_fixture_run_stamp}" \
  "${legacy_fixture_token}" "${legacy_fixture_epoch}" "${rollback_snapshot_id}" \
  "${legacy_fixture_target}" 31311 31312 \
  >"${TMP_DIR}/legacy_stop_thawed_receipt_success.out"
thawed_recaptured_count=$(awk -F '\t' 'NR == 1 { print $13 }' "${thawed_receipt}")
[[ "${thawed_recaptured_count}" =~ ^[1-9][0-9]*$ ]] \
  && (( thawed_recaptured_count > 1 )) ||
  fail 'thawed legacy receipt was trusted instead of recaptured under freezer=1'
[[ "$(cut -f2 "${legacy_fixture_active}")" == stopped \
      && ! -e "${thawed_receipt}.freeze-intent" \
      && ! -e "${thawed_receipt}.freeze-intent.in" \
      && ! -e "${thawed_receipt}.in" ]] ||
  fail 'thawed receipt recapture did not reach one canonical stopped closure'
legacy_test_assert_cgroup_terminal \
  "${thawed_cgroup_path}" "${thawed_cgroup_dev}" \
  "${thawed_cgroup_ino}" 'thawed-receipt-recapture'
cleanup_legacy_stop_fixture legacy-thawed-receipt

# A task which daemonized/reparented before the freezer boundary remains in
# the inherited scope but is outside the exact pane PPID authorization.  The
# narrow r21 compatibility stop must thaw and fail closed, not kill that wider
# same-UID cgroup.
make_legacy_stop_fixture legacy-reparented 31313 31314 valid reparented
reparented_pid=${legacy_fixture_reparented_pid}
reparented_start=${legacy_fixture_reparented_start}
reparented_receipt=${legacy_fixture_receipt}
expect_failure \
  "${TMP_DIR}/legacy_stop_reparented_member.out" \
  'outside the exact pane PPID closure' \
  legacy_stop_with_expected "${legacy_fixture_session}" "${legacy_fixture_run_stamp}" \
  "${legacy_fixture_token}" "${legacy_fixture_epoch}" "${rollback_snapshot_id}" \
  "${legacy_fixture_target}" 31313 31314
[[ "$(cut -f2 "${legacy_fixture_active}")" == running \
      && -f "${legacy_fixture_prefix}.session" \
      && ! -e "${reparented_receipt}" \
      && ! -e "${reparented_receipt}.freeze-intent" \
      && ! -e "${reparented_receipt}.freeze-intent.in" \
      && ! -e "${reparented_receipt}.in" ]] ||
  fail 'reparented cgroup-member refusal left lifecycle or capture metadata mutated'
legacy_test_wait_cgroup_frozen "${legacy_fixture_cgroup_dir}" 0 ||
  fail 'reparented cgroup-member refusal did not restore effective freezer=0'
legacy_test_proc_identity "${reparented_pid}" \
  && [[ "${legacy_test_start}" == "${reparented_start}" ]] ||
  fail 'fail-closed reparented-member check killed the unauthorized task'
cleanup_legacy_stop_fixture legacy-reparented

make_legacy_stop_fixture legacy-strict 31305 31306 valid
strict_session=${legacy_fixture_session}
strict_stamp=${legacy_fixture_run_stamp}
strict_token=${legacy_fixture_token}
strict_epoch=${legacy_fixture_epoch}
strict_target=${legacy_fixture_target}
strict_prefix=${legacy_fixture_prefix}
strict_active=${legacy_fixture_active}
strict_receipt=${legacy_fixture_receipt}
strict_root_pid=${legacy_fixture_root_pid}
strict_root_start=${legacy_fixture_root_start}
strict_root_uid=${legacy_fixture_root_uid}
strict_root_ppid=${legacy_fixture_root_ppid}
strict_root_pgrp=${legacy_fixture_root_pgrp}
strict_root_session=${legacy_fixture_root_session}
strict_setsid_pid=${legacy_fixture_setsid_pid}
strict_setsid_start=${legacy_fixture_setsid_start}
strict_command_sha=${legacy_fixture_command_sha}
strict_cgroup_path=${legacy_fixture_cgroup_path}
strict_cgroup_dir=${legacy_fixture_cgroup_dir}
strict_cgroup_dev=${legacy_fixture_cgroup_dev}
strict_cgroup_ino=${legacy_fixture_cgroup_ino}
strict_cgroup_fingerprint=${legacy_fixture_cgroup_fingerprint}

printf '%s\n' "$(printf legacy-wrong-option | sha256sum | awk '{print $1}')" \
  >"${strict_prefix}.token"
expect_failure \
  "${TMP_DIR}/legacy_stop_option_mismatch.out" \
  'tmux options-only ownership does not match active identity' \
  legacy_stop_with_expected "${strict_session}" "${strict_stamp}" "${strict_token}" \
  "${strict_epoch}" "${rollback_snapshot_id}" "${strict_target}" 31305 31306
printf '%s\n' "${strict_token}" >"${strict_prefix}.token"

printf '%s\n' "$$" >"${strict_prefix}.extra_pane_pid"
expect_failure \
  "${TMP_DIR}/legacy_stop_extra_pane.out" \
  'Legacy stop requires exactly one pane' \
  legacy_stop_with_expected "${strict_session}" "${strict_stamp}" "${strict_token}" \
  "${strict_epoch}" "${rollback_snapshot_id}" "${strict_target}" 31305 31306
rm -f "${strict_prefix}.extra_pane_pid"

# Do not use the test shell's own PID as the malformed pane fixture.  Under a
# PTY the top-level shell can itself be the process-group/session leader, so
# the production validator correctly advances to its argv check and the test
# becomes invocation-context dependent.  A plain background child inherits
# the caller's group/session while having a distinct PID, which guarantees the
# exact leader invariant is false under both PTY and non-PTY runners.
legacy_nonleader_token=$(printf legacy-nonleader-pane | sha256sum | awk '{print $1}')
env HOLOSOMA_LAUNCH_TOKEN="${legacy_nonleader_token}" sleep 3600 &
legacy_nonleader_pid=$!
wait_for_fixture_process_token \
  "${legacy_nonleader_pid}" "${legacy_nonleader_token}" legacy-nonleader-pane
legacy_test_proc_identity "${legacy_nonleader_pid}" ||
  fail 'legacy nonleader pane fixture disappeared before validation'
[[ "${legacy_test_pgrp}" != "${legacy_nonleader_pid}" \
      || "${legacy_test_session}" != "${legacy_nonleader_pid}" ]] ||
  fail 'legacy nonleader pane fixture unexpectedly became a group/session leader'
printf '%s\n' "${legacy_nonleader_pid}" >"${strict_prefix}.pane_pid"
expect_failure \
  "${TMP_DIR}/legacy_stop_wrong_pane_pid.out" \
  'is not its exact process-group/session leader' \
  legacy_stop_with_expected "${strict_session}" "${strict_stamp}" "${strict_token}" \
  "${strict_epoch}" "${rollback_snapshot_id}" "${strict_target}" 31305 31306
printf '%s\n' "${strict_root_pid}" >"${strict_prefix}.pane_pid"
kill "${legacy_nonleader_pid}" 2>/dev/null || true
wait "${legacy_nonleader_pid}" 2>/dev/null || true

printf '2\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t1\n%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${strict_token}" "${strict_epoch}" "${strict_command_sha}" \
  "${rollback_snapshot_id}" "logs/batch_ne/${strict_session}_${strict_stamp}" \
  "${strict_target}" "${strict_root_pid}" "$((strict_root_start + 1))" \
  "${strict_cgroup_path}" "${strict_cgroup_dev}" "${strict_cgroup_ino}" \
  "${strict_root_pid}" "$((strict_root_start + 1))" "${strict_root_uid}" \
  "${strict_root_ppid}" "${strict_root_pgrp}" "${strict_root_session}" \
  >"${strict_receipt}"
chmod 0400 "${strict_receipt}"
expect_failure \
  "${TMP_DIR}/legacy_stop_wrong_starttime.out" \
  'Legacy receipt root does not match the exact current pane identity' \
  legacy_stop_with_expected "${strict_session}" "${strict_stamp}" "${strict_token}" \
  "${strict_epoch}" "${rollback_snapshot_id}" "${strict_target}" 31305 31306
rm -f "${strict_receipt}"

printf 'corrupt\treceipt\n' >"${strict_receipt}"
chmod 0400 "${strict_receipt}"
expect_failure \
  "${TMP_DIR}/legacy_stop_corrupt_receipt.out" \
  'Legacy process-capture receipt header/identity is malformed' \
  legacy_stop_with_expected "${strict_session}" "${strict_stamp}" "${strict_token}" \
  "${strict_epoch}" "${rollback_snapshot_id}" "${strict_target}" 31305 31306
rm -f "${strict_receipt}"
[[ "$(cut -f2 "${strict_active}")" == running && -f "${strict_prefix}.session" ]] ||
  fail 'legacy preflight corruption cases mutated the live fixture'
kill -0 "${strict_root_pid}" 2>/dev/null ||
  fail 'legacy preflight corruption cases killed the exact pane root'

legacy_unrelated_token=$(printf legacy-unrelated | sha256sum | awk '{print $1}')
env HOLOSOMA_LAUNCH_TOKEN="${legacy_unrelated_token}" sleep 3600 &
legacy_unrelated_pid=$!
wait_for_fixture_process_token \
  "${legacy_unrelated_pid}" "${legacy_unrelated_token}" legacy-stop-unrelated
expect_failure \
  "${TMP_DIR}/legacy_stop_retry_receipt.out" \
  '[FAKE] forced legacy kill-session failure' \
  legacy_stop_with_expected "${strict_session}" "${strict_stamp}" "${strict_token}" \
  "${strict_epoch}" "${rollback_snapshot_id}" "${strict_target}" 31305 31306 \
  FAKE_TMUX_KILL_SESSION_RC=97
[[ "$(cut -f2 "${strict_active}")" == stopping && -f "${strict_prefix}.session" ]] ||
  fail 'failed legacy tmux kill did not retain exact stopping/tmux retry state'
[[ -f "${strict_receipt}" && ! -L "${strict_receipt}" \
      && "$(stat -c %a "${strict_receipt}")" == 400 \
      && "$(stat -c %h "${strict_receipt}")" == 1 ]] ||
  fail 'failed legacy stop did not retain one durable canonical capture receipt'
awk -F '\t' -v token="${strict_token}" -v epoch="${strict_epoch}" \
  -v command="${strict_command_sha}" -v cgroup="${strict_cgroup_path}" \
  -v dev="${strict_cgroup_dev}" -v ino="${strict_cgroup_ino}" '
    NR == 1 && NF == 13 && $1 == 2 && $2 == token && $3 == epoch \
      && $4 == command && $10 == cgroup && $11 == dev && $12 == ino { found = 1 }
    END { exit(found ? 0 : 1) }
  ' "${strict_receipt}" ||
  fail 'legacy capture receipt omitted its exact v2 cgroup identity header'
awk -F '\t' -v pid="${strict_root_pid}" -v start="${strict_root_start}" '
    NR > 1 && NF == 6 && $1 == pid && $2 == start { found = 1 }
    END { exit(found ? 0 : 1) }
  ' "${strict_receipt}" ||
  fail 'legacy capture receipt omitted its exact pane PID/starttime'
awk -F '\t' -v pid="${strict_setsid_pid}" -v start="${strict_setsid_start}" '
    NR > 1 && NF == 6 && $1 == pid && $2 == start { found = 1 }
    END { exit(found ? 0 : 1) }
  ' "${strict_receipt}" ||
  fail 'legacy cgroup receipt omitted a PPID descendant which entered a new session'
[[ ! -e "${strict_receipt}.freeze-intent" \
      && ! -e "${strict_receipt}.freeze-intent.in" \
      && ! -e "${strict_receipt}.in" ]] ||
  fail 'committed legacy receipt retained a freeze intent or publication residue'
legacy_test_assert_cgroup_terminal \
  "${strict_cgroup_path}" "${strict_cgroup_dev}" "${strict_cgroup_ino}" \
  'failed-tmux-retry'
if legacy_test_proc_identity "${strict_root_pid}"; then
  fail 'committed legacy cleanup ran a catchable pane EXIT path before tmux retry'
fi
while IFS=$'\t' read -r receipt_pid receipt_start receipt_uid receipt_ppid \
    receipt_pgrp receipt_session receipt_extra; do
  [[ "${receipt_pid}" =~ ^[1-9][0-9]*$ ]] || continue
  [[ -z "${receipt_extra}" ]] ||
    fail 'legacy v2 receipt contains a non-canonical process record'
  if legacy_test_proc_identity "${receipt_pid}" \
      && [[ "${legacy_test_start}" == "${receipt_start}" ]]; then
    fail 'committed legacy cleanup left a captured process alive after tmux-only failure'
  fi
done < <(tail -n +2 "${strict_receipt}")

legacy_stop_with_expected "${strict_session}" "${strict_stamp}" "${strict_token}" \
  "${strict_epoch}" "${rollback_snapshot_id}" "${strict_target}" 31305 31306 \
  >"${TMP_DIR}/legacy_stop_retry_success.out"
set +e
wait "${strict_root_pid}"
strict_root_rc=$?
set -e
(( strict_root_rc != 0 )) || fail 'legacy retry did not terminate the exact pane process'
kill -0 "${legacy_unrelated_pid}" 2>/dev/null ||
  fail 'legacy cgroup cleanup terminated an unrelated process'
kill "${legacy_unrelated_pid}" 2>/dev/null || true
wait "${legacy_unrelated_pid}" 2>/dev/null || true
[[ "$(cut -f2 "${strict_active}")" == stopped && ! -f "${strict_prefix}.session" ]] ||
  fail 'legacy retry did not publish exact stopped/tmux closure'
[[ -z "$(list_rendezvous_endpoint_states)" ]] ||
  fail 'legacy retry leaked token-bound rendezvous reservations'
cleanup_legacy_stop_fixture "${strict_session}"
if (( legacy_fixture_distill_installed == 1 )); then
  mv -fT "${legacy_fixture_distill_backup}" \
    "${ROLLBACK_REMOTE_ROOT}/${rollback_snapshot_id}/distill_as_button_solid.sh"
  chmod "${legacy_fixture_snapshot_root_mode}" \
    "${ROLLBACK_REMOTE_ROOT}/${rollback_snapshot_id}"
  legacy_fixture_distill_installed=0
fi

run_rendezvous_corruption_case() {
  local mode="$1" session="$2" main_port="$3" provenance_port="$4"
  local main_state provenance_state token created_at
  "${LIFECYCLE_ENV[@]}" SESSION="${session}" RUN_STAMP="${session}" \
    MASTER_PORT="${main_port}" HOLOSOMA_PROVENANCE_MASTER_PORT="${provenance_port}" \
    bash batch_ne.sh launch >"${TMP_DIR}/${session}_launch.out"
  token=$(cut -f6 "$(lifecycle_active_path "${session}")")
  main_state=$(rendezvous_endpoint_path "${main_port}")
  provenance_state=$(rendezvous_endpoint_path "${provenance_port}")
  created_at=$(cut -f5 "${main_state}")
  case "${mode}" in
    wrong-port)
      printf '2\t%s\t%s\t%s\t%s\n' \
        "${token}" "${session}" "$((provenance_port + 7))" "${created_at}" >"${provenance_state}"
      ;;
    wrong-session)
      printf '2\t%s\t%s\t%s\t%s\n' \
        "${token}" other-session "${provenance_port}" "${created_at}" >"${provenance_state}"
      ;;
    empty-session)
      printf '2\t%s\t\t%s\t%s\n' \
        "${token}" "${provenance_port}" "${created_at}" >"${provenance_state}"
      ;;
    extra-field)
      printf '2\t%s\t%s\t%s\t%s\textra\n' \
        "${token}" "${session}" "${provenance_port}" "${created_at}" >"${provenance_state}"
      ;;
    mismatched-created-at)
      printf '2\t%s\t%s\t%s\t%s\n' \
        "${token}" "${session}" "${provenance_port}" "$((created_at + 1))" >"${provenance_state}"
      ;;
    symlink)
      mv "${provenance_state}" "${provenance_state}.target"
      ln -s "${provenance_state}.target" "${provenance_state}"
      ;;
    *) fail "unsupported rendezvous corruption mode: ${mode}" ;;
  esac
  expect_failure \
    "${TMP_DIR}/${session}_stop.out" \
    'token-bound rendezvous release failed; reservations remain quarantined' \
    "${LIFECYCLE_ENV[@]}" SESSION="${session}" RUN_STAMP="${session}" \
    MASTER_PORT="${main_port}" HOLOSOMA_PROVENANCE_MASTER_PORT="${provenance_port}" \
    bash batch_ne.sh stop
  [[ -e "${main_state}" && ( -e "${provenance_state}" || -L "${provenance_state}" ) ]] ||
    fail "${mode} rendezvous validation partially deleted a corrupt pair"
  [[ "$(cut -f2 "$(lifecycle_active_path "${session}")")" == stopped ]] ||
    fail "${mode} rendezvous fixture did not prove process/session stop closure before release refusal"
  rm -f "${main_state}" "${provenance_state}" "${provenance_state}.target"
}

# Pair deletion is two-phase and exact: corrupt provenance must preserve main,
# including empty/session/port/shape/symlink and cross-transaction timestamp
# mismatches.
run_rendezvous_corruption_case wrong-port rendezvous-wrong-port 31201 31202
run_rendezvous_corruption_case wrong-session rendezvous-wrong-session 31203 31204
run_rendezvous_corruption_case empty-session rendezvous-empty-session 31205 31206
run_rendezvous_corruption_case extra-field rendezvous-extra-field 31207 31208
run_rendezvous_corruption_case mismatched-created-at rendezvous-created-at 31209 31210
run_rendezvous_corruption_case symlink rendezvous-symlink 31211 31212

# An unrelated same-session record at another port/token is not matched by a
# wildcard teardown and must survive exact release of the active pair.
"${LIFECYCLE_ENV[@]}" SESSION=rendezvous-extra-token RUN_STAMP=rendezvous-extra-token \
  MASTER_PORT=31213 HOLOSOMA_PROVENANCE_MASTER_PORT=31214 \
  bash batch_ne.sh launch >"${TMP_DIR}/rendezvous_extra_token_launch.out"
extra_token_other=$(printf rendezvous-other-token | sha256sum | awk '{print $1}')
extra_token_state=$(rendezvous_endpoint_path 31215)
extra_token_created=$(cut -f5 "$(rendezvous_endpoint_path 31213)")
printf '2\t%s\t%s\t31215\t%s\n' \
  "${extra_token_other}" rendezvous-extra-token "${extra_token_created}" >"${extra_token_state}"
"${LIFECYCLE_ENV[@]}" SESSION=rendezvous-extra-token RUN_STAMP=rendezvous-extra-token \
  MASTER_PORT=31213 HOLOSOMA_PROVENANCE_MASTER_PORT=31214 \
  bash batch_ne.sh stop >"${TMP_DIR}/rendezvous_extra_token_stop.out"
[[ -f "${extra_token_state}" && "$(cut -f2 "${extra_token_state}")" == "${extra_token_other}" ]] ||
  fail 'exact stop release removed a same-session reservation owned by another token/port'
rm -f "${extra_token_state}"

# Startup acceptance is for a session created by this same controller and may
# not silently downgrade to the legacy options-only contract.  Remove all
# atomic fields after the post-create check but before the first health probe.
expect_failure \
  "${TMP_DIR}/startup_removed_atomic_environment.out" \
  'tmux atomic environment/options do not match active metadata' \
  "${LIFECYCLE_ENV[@]}" SESSION=startup-removed-env RUN_STAMP=startup-removed-env \
  MASTER_PORT=31015 HOLOSOMA_PROVENANCE_MASTER_PORT=31016 \
  FAKE_TMUX_STARTUP_IDENTITY_MODE=removed-environment bash batch_ne.sh launch
startup_removed_state=$(lifecycle_active_path startup-removed-env)
[[ "$(cut -f2 "${startup_removed_state}")" == rolled_back ]] ||
  fail 'startup accepted or failed to roll back a newly launched session with removed atomic identity'
[[ ! -f "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.startup-removed-env.session" ]] ||
  fail 'startup atomic-identity failure left its exactly option-bound session alive'
if [[ -n "$(list_rendezvous_endpoint_states)" ]]; then
  fail 'startup atomic-identity failure leaked rendezvous reservations'
fi

# Batch and terminal-launcher boundary markers are necessary but not sufficient:
# torchrun workers must also finish the all-rank provenance rendezvous. Timeout
# rollback targets only the cryptographically owned session and leaves an
# unrelated tmux session on the same host untouched.
unrelated_prefix="${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.startup-unrelated"
: >"${unrelated_prefix}.session"
printf '%s\n' unrelated-owner >"${unrelated_prefix}.token"
expect_failure \
  "${TMP_DIR}/startup_boundary_only_timeout.out" \
  'Startup-health handshake timed out' \
  "${LIFECYCLE_ENV[@]}" SESSION=startup-boundary-only RUN_STAMP=startup-boundary-only \
  MASTER_PORT=30993 HOLOSOMA_PROVENANCE_MASTER_PORT=30994 \
  LAUNCH_STARTUP_TIMEOUT_SECONDS=2 LAUNCH_STARTUP_PROBE_TIMEOUT_SECONDS=1 \
  FAKE_TMUX_STARTUP_MODE=boundary-only bash batch_ne.sh launch
grep -F 'pending batch_preflight=1/1 torchrun_boundary=1/1 distributed_provenance=0/1 final_workers=0/1' \
  "${TMP_DIR}/startup_boundary_only_timeout.out" >/dev/null ||
  fail 'launcher-boundary readiness was not held pending for cross-rank provenance'
boundary_only_state=$(lifecycle_active_path startup-boundary-only)
[[ "$(cut -f2 "${boundary_only_state}")" == rolled_back ]] ||
  fail 'startup timeout did not publish rolled_back lifecycle state'
[[ ! -f "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.startup-boundary-only.session" ]] ||
  fail 'startup timeout left its owned tmux session alive'
[[ -f "${unrelated_prefix}.session" ]] ||
  fail 'startup timeout killed an unrelated tmux session'
grep -Fx "${LIFECYCLE_NODE} startup-boundary-only" "${FAKE_TMUX_STATE_DIR}/kills.log" >/dev/null ||
  fail 'startup timeout did not kill the session owned by its launch token'
if [[ -n "$(list_rendezvous_endpoint_states)" ]]; then
  fail 'startup timeout leaked rendezvous reservations'
fi

# Fatal evidence after tmux creation must fail immediately rather than waiting
# for the startup timeout, and must use the same ownership-safe rollback path.
expect_failure \
  "${TMP_DIR}/startup_fatal.out" \
  'fatal/non-finite evidence appeared during startup' \
  "${LIFECYCLE_ENV[@]}" SESSION=startup-fatal RUN_STAMP=startup-fatal \
  MASTER_PORT=30995 HOLOSOMA_PROVENANCE_MASTER_PORT=30996 \
  FAKE_TMUX_STARTUP_MODE=fatal bash batch_ne.sh launch
fatal_state=$(lifecycle_active_path startup-fatal)
[[ "$(cut -f2 "${fatal_state}")" == rolled_back ]] ||
  fail 'fatal startup did not publish rolled_back lifecycle state'
[[ ! -f "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.startup-fatal.session" ]] ||
  fail 'fatal startup left its owned tmux session alive'
[[ -f "${unrelated_prefix}.session" ]] ||
  fail 'fatal startup rollback killed an unrelated tmux session'
if [[ -n "$(list_rendezvous_endpoint_states)" ]]; then
  fail 'fatal startup leaked rendezvous reservations'
fi

# A fatal near the front of a large recent tail must not be inverted by
# pipefail when grep has already found it and closes its input early.
expect_failure \
  "${TMP_DIR}/startup_fatal_long.out" \
  'fatal/non-finite evidence appeared during startup' \
  "${LIFECYCLE_ENV[@]}" SESSION=startup-fatal-long RUN_STAMP=startup-fatal-long \
  MASTER_PORT=30997 HOLOSOMA_PROVENANCE_MASTER_PORT=30998 \
  FAKE_TMUX_STARTUP_MODE=fatal-long-ready bash batch_ne.sh launch
fatal_long_state=$(lifecycle_active_path startup-fatal-long)
[[ "$(cut -f2 "${fatal_long_state}")" == rolled_back ]] ||
  fail 'long-log fatal startup did not publish rolled_back lifecycle state'
[[ ! -f "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.startup-fatal-long.session" ]] ||
  fail 'long-log fatal startup left its exact owned tmux session alive'

# Every local torchrun worker must independently finish both provenance and the
# later env/algo/checkpoint barrier. One marker on a two-worker node is pending.
expect_failure \
  "${TMP_DIR}/startup_missing_worker.out" \
  'Startup-health handshake timed out' \
  "${LIFECYCLE_ENV[@]}" SESSION=startup-missing-worker RUN_STAMP=startup-missing-worker \
  NPROC=2 CUDA_VISIBLE_DEVICES=0,1 \
  MASTER_PORT=30999 HOLOSOMA_PROVENANCE_MASTER_PORT=31000 \
  LAUNCH_STARTUP_TIMEOUT_SECONDS=2 LAUNCH_STARTUP_PROBE_TIMEOUT_SECONDS=1 \
  FAKE_TMUX_STARTUP_MODE=missing-worker bash batch_ne.sh launch
grep -F 'distributed_provenance=2/2 final_workers=1/2' \
  "${TMP_DIR}/startup_missing_worker.out" >/dev/null ||
  fail 'one missing final worker marker was not held pending'
[[ "$(cut -f2 "$(lifecycle_active_path startup-missing-worker)")" == rolled_back ]] ||
  fail 'missing-worker startup did not roll back its exact owned session'

expect_failure \
  "${TMP_DIR}/startup_missing_provenance.out" \
  'Startup-health handshake timed out' \
  "${LIFECYCLE_ENV[@]}" SESSION=startup-missing-provenance RUN_STAMP=startup-missing-provenance \
  NPROC=2 CUDA_VISIBLE_DEVICES=0,1 \
  MASTER_PORT=31003 HOLOSOMA_PROVENANCE_MASTER_PORT=31004 \
  LAUNCH_STARTUP_TIMEOUT_SECONDS=2 LAUNCH_STARTUP_PROBE_TIMEOUT_SECONDS=1 \
  FAKE_TMUX_STARTUP_MODE=missing-provenance bash batch_ne.sh launch
grep -F 'distributed_provenance=1/2 final_workers=2/2' \
  "${TMP_DIR}/startup_missing_provenance.out" >/dev/null ||
  fail 'one missing provenance marker was not held pending'

# A startup probe reads a log which torchrun/tee is still appending.  A marker
# prefix can therefore be visible to the aggregate grep before the same line
# is newline-complete and exact-rank-valid.  That transient mismatch must stay
# pending, then become healthy after the writer completes the record.
"${LIFECYCLE_ENV[@]}" SESSION=startup-progressive-worker RUN_STAMP=startup-progressive-worker \
  NPROC=2 CUDA_VISIBLE_DEVICES=0,1 \
  MASTER_PORT=32101 HOLOSOMA_PROVENANCE_MASTER_PORT=32102 \
  LAUNCH_STARTUP_TIMEOUT_SECONDS=12 LAUNCH_STARTUP_PROBE_TIMEOUT_SECONDS=1 \
  FAKE_TMUX_STARTUP_MODE=progressive-worker FAKE_TMUX_PROGRESSIVE_DELAY_SECONDS=5 \
  bash batch_ne.sh launch >"${TMP_DIR}/startup_progressive_worker.out"
grep -F 'final_workers=1/2' "${TMP_DIR}/startup_progressive_worker.out" \
  | grep -F 'observed_worker_markers=2/2' >/dev/null ||
  fail 'an in-flight worker marker mismatch was not held pending'
grep -F 'Startup-health handshake passed for all 1 node(s)' \
  "${TMP_DIR}/startup_progressive_worker.out" >/dev/null ||
  fail 'completed progressive worker markers never became startup-healthy'
[[ "$(cut -f2 "$(lifecycle_active_path startup-progressive-worker)")" == running ]] ||
  fail 'progressive worker launch did not retain running lifecycle state'
"${LIFECYCLE_ENV[@]}" SESSION=startup-progressive-worker RUN_STAMP=startup-progressive-worker \
  NPROC=2 CUDA_VISIBLE_DEVICES=0,1 \
  MASTER_PORT=32101 HOLOSOMA_PROVENANCE_MASTER_PORT=32102 \
  bash batch_ne.sh stop >"${TMP_DIR}/startup_progressive_worker_stop.out"
[[ ! -f "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.startup-progressive-worker.session" ]] ||
  fail 'progressive worker fixture left its accepted tmux session running'

# A complete malformed marker is not sufficient evidence for acceptance, but
# while the log remains live it is indistinguishable from a write in progress.
# Keep it pending until the bounded timeout instead of rolling back on the
# first inconsistent pair of greps.
expect_failure \
  "${TMP_DIR}/startup_malformed_worker.out" \
  'Startup-health handshake timed out' \
  "${LIFECYCLE_ENV[@]}" SESSION=startup-malformed-worker RUN_STAMP=startup-malformed-worker \
  NPROC=2 CUDA_VISIBLE_DEVICES=0,1 \
  MASTER_PORT=32103 HOLOSOMA_PROVENANCE_MASTER_PORT=32104 \
  LAUNCH_STARTUP_TIMEOUT_SECONDS=2 LAUNCH_STARTUP_PROBE_TIMEOUT_SECONDS=1 \
  FAKE_TMUX_STARTUP_MODE=malformed-worker bash batch_ne.sh launch
grep -F 'final_workers=1/2' "${TMP_DIR}/startup_malformed_worker.out" \
  | grep -F 'observed_worker_markers=2/2' >/dev/null ||
  fail 'malformed worker evidence was not held pending and rejected by timeout'

# An exact rank marker observed twice is monotonic, unrecoverable evidence even
# when another rank is still absent and the aggregate count has not exceeded
# NPROC.  It must remain an immediate fatal rather than being softened by the
# live-log race handling.
expect_failure \
  "${TMP_DIR}/startup_duplicate_worker.out" \
  'duplicate or launch-mismatched startup evidence is not valid' \
  "${LIFECYCLE_ENV[@]}" SESSION=startup-duplicate-worker RUN_STAMP=startup-duplicate-worker \
  NPROC=2 CUDA_VISIBLE_DEVICES=0,1 \
  MASTER_PORT=32105 HOLOSOMA_PROVENANCE_MASTER_PORT=32106 \
  LAUNCH_STARTUP_TIMEOUT_SECONDS=8 LAUNCH_STARTUP_PROBE_TIMEOUT_SECONDS=1 \
  FAKE_TMUX_STARTUP_MODE=duplicate-worker bash batch_ne.sh launch
grep -F 'worker_markers=2 valid_unique_workers=0 duplicate_worker_ranks=1' \
  "${TMP_DIR}/startup_duplicate_worker.out" >/dev/null ||
  fail 'duplicate exact worker rank was not diagnosed independently of the aggregate bound'

# Exact-marker counters alone cannot close the observed log: one wrong launch
# marker plus the required valid markers would otherwise satisfy acceptance.
# Total prefix counts make these extra identities immediately fail closed.
expect_failure \
  "${TMP_DIR}/startup_mismatched_ready.out" \
  'duplicate or launch-mismatched startup evidence is not valid' \
  "${LIFECYCLE_ENV[@]}" SESSION=startup-mismatched-ready RUN_STAMP=startup-mismatched-ready \
  NPROC=2 CUDA_VISIBLE_DEVICES=0,1 \
  MASTER_PORT=32107 HOLOSOMA_PROVENANCE_MASTER_PORT=32108 \
  LAUNCH_STARTUP_TIMEOUT_SECONDS=8 LAUNCH_STARTUP_PROBE_TIMEOUT_SECONDS=1 \
  FAKE_TMUX_STARTUP_MODE=mismatched-ready bash batch_ne.sh launch
grep -F 'ready=1/2' "${TMP_DIR}/startup_mismatched_ready.out" >/dev/null ||
  fail 'wrong-launch ready marker was not included in the observed total'

expect_failure \
  "${TMP_DIR}/startup_mismatched_provenance.out" \
  'duplicate or launch-mismatched startup evidence is not valid' \
  "${LIFECYCLE_ENV[@]}" SESSION=startup-mismatched-provenance RUN_STAMP=startup-mismatched-provenance \
  NPROC=2 CUDA_VISIBLE_DEVICES=0,1 \
  MASTER_PORT=32109 HOLOSOMA_PROVENANCE_MASTER_PORT=32110 \
  LAUNCH_STARTUP_TIMEOUT_SECONDS=8 LAUNCH_STARTUP_PROBE_TIMEOUT_SECONDS=1 \
  FAKE_TMUX_STARTUP_MODE=mismatched-provenance bash batch_ne.sh launch
grep -F 'distributed_provenance=2/3' \
  "${TMP_DIR}/startup_mismatched_provenance.out" >/dev/null ||
  fail 'wrong-world-size provenance marker was not included in the observed total'

# Even a connected SSH which hangs in rollback is bounded. Because cleanup was
# not confirmed, metadata must remain rolling_back and the exact session is
# reported/preserved for a later retry rather than falsely marked rolled_back.
# The inner controller bound sends TERM after the configured cleanup timeout
# and deliberately retains a five-second hard-kill grace for an SSH client
# which ignores TERM. Leave enough outer-test budget for startup, that full
# grace, and the controller's resulting durable error report.
expect_failure \
  "${TMP_DIR}/startup_bounded_rollback.out" \
  'Exact owned tmux cleanup was not confirmed within the bound' \
  timeout 20s "${LIFECYCLE_ENV[@]}" \
  SESSION=startup-bounded-rollback RUN_STAMP=startup-bounded-rollback \
  MASTER_PORT=31005 HOLOSOMA_PROVENANCE_MASTER_PORT=31006 \
  LAUNCH_STARTUP_TIMEOUT_SECONDS=2 LAUNCH_STARTUP_PROBE_TIMEOUT_SECONDS=1 \
  LAUNCH_CLEANUP_TIMEOUT_SECONDS=2 LAUNCH_LOCK_TIMEOUT_SECONDS=1 \
  FAKE_TMUX_STARTUP_MODE=boundary-only FAKE_SSH_HANG_ROLLBACK=1 \
  bash batch_ne.sh launch
bounded_rollback_state=$(lifecycle_active_path startup-bounded-rollback)
[[ "$(cut -f2 "${bounded_rollback_state}")" == rolling_back ]] ||
  fail 'unconfirmed bounded rollback was falsely recorded as rolled_back'
bounded_rollback_prefix="${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.startup-bounded-rollback"
[[ -f "${bounded_rollback_prefix}.session" ]] ||
  fail 'bounded rollback fixture unexpectedly removed the unconfirmed session'
bounded_rollback_token=$(cut -f6 "${bounded_rollback_state}")
assert_quarantined_rendezvous_for_token \
  "${bounded_rollback_token}" 'bounded unconfirmed startup rollback'
clear_quarantined_rendezvous
rm -f "${bounded_rollback_prefix}.session" "${bounded_rollback_prefix}.token" \
  "${bounded_rollback_prefix}.command_sha256" "${bounded_rollback_prefix}.launch_epoch" \
  "${bounded_rollback_prefix}.env_token" "${bounded_rollback_prefix}.env_command_sha256" \
  "${bounded_rollback_prefix}.env_launch_epoch"

# A controller timeout must also terminate the remote lifecycle transaction,
# not merely its local SSH client.  Hang fake tmux after the durable
# rolling_back transition but before session deletion; the remote wrapper must
# kill that complete process group and leave no late metadata/session writer.
remote_mutation_session=startup-remote-mutation-timeout
remote_mutation_marker="${TMP_DIR}/startup-remote-mutation-timeout"
remote_mutation_wrapper_marker="${remote_mutation_marker}.wrapper"
expect_failure \
  "${TMP_DIR}/startup_remote_mutation_timeout.out" \
  'Exact owned tmux cleanup was not confirmed within the bound' \
  timeout 30s "${LIFECYCLE_ENV[@]}" \
  SESSION="${remote_mutation_session}" RUN_STAMP="${remote_mutation_session}" \
  MASTER_PORT=31007 HOLOSOMA_PROVENANCE_MASTER_PORT=31008 \
  LAUNCH_STARTUP_TIMEOUT_SECONDS=2 LAUNCH_STARTUP_PROBE_TIMEOUT_SECONDS=1 \
  LAUNCH_CLEANUP_TIMEOUT_SECONDS=8 LAUNCH_LOCK_TIMEOUT_SECONDS=1 \
  FAKE_TMUX_STARTUP_MODE=boundary-only \
  FAKE_SSH_REQUIRE_MUTATION_WRAPPER_SESSION="${remote_mutation_session}" \
  FAKE_SSH_MUTATION_WRAPPER_MARKER="${remote_mutation_wrapper_marker}" \
  FAKE_TMUX_HANG_KILL_SESSION="${remote_mutation_session}" \
  FAKE_TMUX_HANG_MARKER_PREFIX="${remote_mutation_marker}" \
  bash batch_ne.sh launch
remote_mutation_state=$(lifecycle_active_path "${remote_mutation_session}")
remote_mutation_prefix="${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.${remote_mutation_session}"
[[ -f "${remote_mutation_wrapper_marker}" ]] ||
  fail 'startup rollback did not traverse the remote mutation timeout wrapper'
[[ -f "${remote_mutation_marker}.entered" ]] ||
  fail 'remote mutation timeout fixture never entered the tmux kill boundary'
[[ "$(cut -f2 "${remote_mutation_state}")" == rolling_back ]] ||
  fail 'interrupted remote rollback was falsely recorded as rolled_back'
[[ -f "${remote_mutation_prefix}.session" ]] ||
  fail 'interrupted remote rollback removed its unconfirmed tmux session'
for remote_mutation_pid_file in \
  "${remote_mutation_marker}.shell-pid" "${remote_mutation_marker}.sleep-pid"; do
  [[ -s "${remote_mutation_pid_file}" ]] ||
    fail "remote mutation fixture did not record a PID: ${remote_mutation_pid_file}"
  remote_mutation_pid=$(<"${remote_mutation_pid_file}")
  [[ "${remote_mutation_pid}" =~ ^[1-9][0-9]*$ ]] ||
    fail "remote mutation fixture recorded a non-canonical PID: ${remote_mutation_pid_file}"
  for ((remote_mutation_poll = 0; remote_mutation_poll < 200; remote_mutation_poll++)); do
    [[ -e "/proc/${remote_mutation_pid}" ]] || break
    sleep 0.01
  done
  [[ ! -e "/proc/${remote_mutation_pid}" ]] ||
    fail "remote mutation timeout left a live process: ${remote_mutation_pid}"
done
[[ ! -e "${remote_mutation_marker}.late" ]] ||
  fail 'remote rollback continued after its mutation timeout'
for remote_mutation_residue in "${remote_mutation_state}".legacy-processes.*; do
  [[ ! -e "${remote_mutation_residue}" && ! -L "${remote_mutation_residue}" ]] ||
    fail 'startup rollback incorrectly entered the legacy cgroup receipt path'
done
remote_mutation_state_sha=$(sha256sum "${remote_mutation_state}" | awk '{print $1}')
sleep 2
[[ "$(sha256sum "${remote_mutation_state}" | awk '{print $1}')" == "${remote_mutation_state_sha}" ]] ||
  fail 'timed-out remote rollback mutated active state after controller return'
[[ ! -e "${remote_mutation_marker}.late" && -f "${remote_mutation_prefix}.session" ]] ||
  fail 'timed-out remote rollback performed a delayed tmux mutation'
remote_mutation_token=$(cut -f6 "${remote_mutation_state}")
assert_quarantined_rendezvous_for_token \
  "${remote_mutation_token}" 'remote mutation timeout startup rollback'
clear_quarantined_rendezvous
rm -f "${remote_mutation_prefix}.session" "${remote_mutation_prefix}.token" \
  "${remote_mutation_prefix}.command_sha256" "${remote_mutation_prefix}.launch_epoch" \
  "${remote_mutation_prefix}.env_token" "${remote_mutation_prefix}.env_command_sha256" \
  "${remote_mutation_prefix}.env_launch_epoch"
unset remote_mutation_session remote_mutation_marker remote_mutation_wrapper_marker
unset remote_mutation_state remote_mutation_prefix remote_mutation_pid_file
unset remote_mutation_pid remote_mutation_poll remote_mutation_residue remote_mutation_state_sha
unset remote_mutation_token

# Multi-node cleanup may be partially successful.  If rank0 closes but another
# node's rollback hangs, neither endpoint may be released for reuse.
expect_failure \
  "${TMP_DIR}/two_node_partial_rollback.out" \
  'Preserving owned rendezvous reservations as quarantine' \
  timeout 40s "${ROLLBACK_ENV[@]}" \
  NODES="${LIFECYCLE_NODE} ${TOPOLOGY_NODE_B}" MASTER_ADDR="${LIFECYCLE_NODE}" NNODES=2 \
  SESSION=two-node-partial-rollback RUN_STAMP=two-node-partial-rollback \
  MASTER_PORT=31221 HOLOSOMA_PROVENANCE_MASTER_PORT=31222 \
  LAUNCH_STARTUP_TIMEOUT_SECONDS=2 LAUNCH_STARTUP_PROBE_TIMEOUT_SECONDS=1 \
  LAUNCH_CLEANUP_TIMEOUT_SECONDS=8 LAUNCH_LOCK_TIMEOUT_SECONDS=1 \
  FAKE_TMUX_STARTUP_MODE=boundary-only FAKE_SSH_HANG_ROLLBACK=1 \
  FAKE_SSH_HANG_ROLLBACK_NODE="${TOPOLOGY_NODE_B}" \
  bash batch_ne.sh launch
partial_state_a=$(lifecycle_active_path two-node-partial-rollback)
partial_session_a="${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.two-node-partial-rollback"
partial_session_b="${FAKE_TMUX_STATE_DIR}/${TOPOLOGY_NODE_B}.two-node-partial-rollback"
[[ "$(cut -f2 "${partial_state_a}")" == rolled_back && ! -f "${partial_session_a}.session" ]] ||
  fail 'successful rank0 rollback did not reach exact terminal closure'
[[ -f "${partial_session_b}.session" ]] ||
  fail 'hanging non-rank0 rollback fixture unexpectedly lost its session'
partial_rollback_token=$(cut -f6 "${partial_state_a}")
assert_quarantined_rendezvous_for_token \
  "${partial_rollback_token}" 'two-node partial startup rollback'
clear_quarantined_rendezvous
rm -f "${partial_session_b}.session" "${partial_session_b}.token" \
  "${partial_session_b}.command_sha256" "${partial_session_b}.launch_epoch" \
  "${partial_session_b}.env_token" "${partial_session_b}.env_command_sha256" \
  "${partial_session_b}.env_launch_epoch"

# A controller that passed preflight before another controller launched must
# recheck under the publication lock. It may not overwrite the accepted job's
# running active identity after it resumes.
publish_gate="${TMP_DIR}/concurrent-publish-gate"
mkdir -p "${publish_gate}"
"${LIFECYCLE_ENV[@]}" SESSION=concurrent-intent RUN_STAMP=concurrent-intent \
  MASTER_PORT=31007 HOLOSOMA_PROVENANCE_MASTER_PORT=31008 \
  FAKE_SSH_PUBLISH_GATE_DIR="${publish_gate}" \
  bash batch_ne.sh launch >"${TMP_DIR}/concurrent_intent_late.out" 2>&1 &
late_controller_pid=$!
for ((gate_wait = 0; gate_wait < FIXTURE_GATE_MAX_POLLS; gate_wait++)); do
  [[ -f "${publish_gate}/reached" ]] && break
  sleep 0.025
done
[[ -f "${publish_gate}/reached" ]] || fail 'late controller did not reach the publish-intent gate'
"${LIFECYCLE_ENV[@]}" SESSION=concurrent-intent RUN_STAMP=concurrent-intent \
  MASTER_PORT=31007 HOLOSOMA_PROVENANCE_MASTER_PORT=31008 \
  bash batch_ne.sh launch >"${TMP_DIR}/concurrent_intent_winner.out"
concurrent_state=$(lifecycle_active_path concurrent-intent)
IFS=$'\t' read -r _concurrent_version concurrent_phase _concurrent_snapshot _concurrent_log \
  _concurrent_target concurrent_token concurrent_command concurrent_epoch <"${concurrent_state}"
touch "${publish_gate}/release"
if wait "${late_controller_pid}"; then
  fail 'late concurrent controller unexpectedly replaced the accepted launch'
fi
grep -F 'Refusing to publish launch intent while tmux session concurrent-intent exists.' \
  "${TMP_DIR}/concurrent_intent_late.out" >/dev/null ||
  fail 'late concurrent controller did not fail at the locked publication recheck'
grep -F 'verified cancelled-intent closure active_disposition=other' \
  "${TMP_DIR}/concurrent_intent_late.out" >/dev/null ||
  fail 'late concurrent controller did not prove closure beside the preserved running owner'
IFS=$'\t' read -r _version_after phase_after _snapshot_after _log_after _target_after \
  token_after command_after epoch_after <"${concurrent_state}"
[[ "${phase_after}" == running && "${token_after}" == "${concurrent_token}" \
      && "${command_after}" == "${concurrent_command}" && "${epoch_after}" == "${concurrent_epoch}" ]] ||
  fail 'late concurrent controller corrupted the accepted launch identity'
concurrent_tmux_prefix="${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.concurrent-intent"
[[ "$(cat "${concurrent_tmux_prefix}.token")" == "${concurrent_token}" ]] ||
  fail 'late concurrent controller changed the accepted tmux owner'
"${LIFECYCLE_ENV[@]}" SESSION=concurrent-intent RUN_STAMP=concurrent-intent \
  MASTER_PORT=31007 HOLOSOMA_PROVENANCE_MASTER_PORT=31008 \
  bash batch_ne.sh stop >"${TMP_DIR}/concurrent_intent_stop.out"

# Preserving another running owner is safe only when its atomic environment
# and all three stable @options are complete.  Missing one option must
# quarantine cleanup without mutating the winner.
incomplete_gate="${TMP_DIR}/concurrent-incomplete-gate"
mkdir -p "${incomplete_gate}"
"${LIFECYCLE_ENV[@]}" SESSION=concurrent-incomplete RUN_STAMP=concurrent-incomplete \
  MASTER_PORT=31239 HOLOSOMA_PROVENANCE_MASTER_PORT=31240 \
  FAKE_SSH_PUBLISH_GATE_DIR="${incomplete_gate}" \
  bash batch_ne.sh launch >"${TMP_DIR}/concurrent_incomplete_late.out" 2>&1 &
incomplete_late_pid=$!
for ((gate_wait = 0; gate_wait < FIXTURE_GATE_MAX_POLLS; gate_wait++)); do
  [[ -f "${incomplete_gate}/reached" ]] && break
  sleep 0.025
done
[[ -f "${incomplete_gate}/reached" ]] ||
  fail 'incomplete-identity late controller did not reach the publish gate'
"${LIFECYCLE_ENV[@]}" SESSION=concurrent-incomplete RUN_STAMP=concurrent-incomplete \
  MASTER_PORT=31239 HOLOSOMA_PROVENANCE_MASTER_PORT=31240 \
  bash batch_ne.sh launch >"${TMP_DIR}/concurrent_incomplete_winner.out"
incomplete_state=$(lifecycle_active_path concurrent-incomplete)
incomplete_command=$(cut -f7 "${incomplete_state}")
incomplete_prefix="${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.concurrent-incomplete"
rm -f "${incomplete_prefix}.command_sha256"
touch "${incomplete_gate}/release"
if wait "${incomplete_late_pid}"; then
  fail 'late controller accepted a preserved running owner with an incomplete @option identity'
fi
grep -F 'Different-token active metadata was preserved, but same-name tmux is not its exact atomic identity.' \
  "${TMP_DIR}/concurrent_incomplete_late.out" >/dev/null || {
    sed -n '1,100p' "${TMP_DIR}/concurrent_incomplete_late.out" >&2
    fail 'incomplete preserved-other tmux identity did not fail closed'
  }
[[ "$(cut -f2 "${incomplete_state}")" == running \
      && -f "${incomplete_prefix}.session" ]] ||
  fail 'failed old-token closure mutated the preserved running winner'
[[ "$(list_rendezvous_endpoint_states | wc -l)" == 2 ]] ||
  fail 'failed old-token closure released the preserved winner rendezvous pair'
printf '%s\n' "${incomplete_command}" >"${incomplete_prefix}.command_sha256"
"${LIFECYCLE_ENV[@]}" SESSION=concurrent-incomplete RUN_STAMP=concurrent-incomplete \
  MASTER_PORT=31239 HOLOSOMA_PROVENANCE_MASTER_PORT=31240 \
  bash batch_ne.sh stop >"${TMP_DIR}/concurrent_incomplete_stop.out"

# A same-name tmux whose token/command/epoch do not match v2 active metadata
# belongs to another launch and must never be killed by the normal stop path.
UNOWNED_SESSION=unowned-stop
UNOWNED_TMUX_TOKEN=$(printf other-owner | sha256sum | awk '{print $1}')
"${LIFECYCLE_ENV[@]}" SESSION="${UNOWNED_SESSION}" RUN_STAMP="${UNOWNED_SESSION}" \
  MASTER_PORT=31121 HOLOSOMA_PROVENANCE_MASTER_PORT=31122 \
  bash batch_ne.sh launch >"${TMP_DIR}/unowned_normal_stop_launch.out"
unowned_state=$(lifecycle_active_path "${UNOWNED_SESSION}")
UNOWNED_ACTIVE_TOKEN=$(cut -f6 "${unowned_state}")
UNOWNED_COMMAND_SHA=$(cut -f7 "${unowned_state}")
UNOWNED_EPOCH=$(cut -f8 "${unowned_state}")
unowned_tmux_prefix="${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.${UNOWNED_SESSION}"
printf '%s\n' "${UNOWNED_TMUX_TOKEN}" >"${unowned_tmux_prefix}.token"
printf '%s\n' "${UNOWNED_TMUX_TOKEN}" >"${unowned_tmux_prefix}.env_token"
expect_failure \
  "${TMP_DIR}/unowned_normal_stop.out" \
  'same-name tmux is not bound to the active token/command/epoch' \
  "${LIFECYCLE_ENV[@]}" SESSION="${UNOWNED_SESSION}" RUN_STAMP="${UNOWNED_SESSION}" \
  MASTER_PORT=31121 HOLOSOMA_PROVENANCE_MASTER_PORT=31122 bash batch_ne.sh stop
[[ -f "${unowned_tmux_prefix}.session" ]] ||
  fail 'normal stop killed a same-name tmux owned by another launch'
[[ "$(cut -f2 "${unowned_state}")" == running ]] ||
  fail 'failed ownership check mutated the active lifecycle phase'
printf '%s\n' "${UNOWNED_ACTIVE_TOKEN}" >"${unowned_tmux_prefix}.token"
printf '%s\n' "${UNOWNED_ACTIVE_TOKEN}" >"${unowned_tmux_prefix}.env_token"
"${LIFECYCLE_ENV[@]}" SESSION="${UNOWNED_SESSION}" RUN_STAMP="${UNOWNED_SESSION}" \
  MASTER_PORT=31121 HOLOSOMA_PROVENANCE_MASTER_PORT=31122 \
  bash batch_ne.sh stop >"${TMP_DIR}/unowned_normal_stop_cleanup.out"

# Either of the two rendezvous endpoints being busy must abort before any tmux
# starts, roll the published intent back, and leave no owned reservation.
expect_failure \
  "${TMP_DIR}/busy_provenance_port.out" \
  'Rendezvous TCP port is already listening' \
  "${LIFECYCLE_ENV[@]}" SESSION=port-conflict RUN_STAMP=port-conflict \
  MASTER_PORT=31001 HOLOSOMA_PROVENANCE_MASTER_PORT=31002 FAKE_BUSY_PORT=31002 \
  bash batch_ne.sh launch
port_conflict_state=$(lifecycle_active_path port-conflict)
[[ "$(cut -f2 "${port_conflict_state}")" == rolled_back ]] ||
  fail 'port-conflict launch intent was not moved to rolled_back'
if [[ -n "$(list_rendezvous_endpoint_states)" ]]; then
  fail 'failed dual-port reservation leaked rendezvous state'
fi
[[ ! -f "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.port-conflict.session" ]] ||
  fail 'port-conflict launch unexpectedly started tmux'

NEW_TOKEN=$(printf new-token | sha256sum | awk '{print $1}')
OLD_TOKEN=$(printf old-token | sha256sum | awk '{print $1}')
COMMAND_SHA=$(printf command | sha256sum | awk '{print $1}')
NOW=$(date +%s)

# A completion marker from a previous launch is never authoritative without
# the exact active token/command/epoch log binding.
write_lifecycle_active stale-completion running "${NEW_TOKEN}" "${COMMAND_SHA}" "${NOW}"
write_lifecycle_log stale-completion "${OLD_TOKEN}" "${COMMAND_SHA}" "${NOW}"
printf 'HOLOSOMA_RUN_COMPLETE target_iteration=1 checkpoint=/tmp/model_00001.pt\n' \
  >>"$(lifecycle_log_path stale-completion)"
expect_failure \
  "${TMP_DIR}/stale_completion.out" \
  'active log is not bound to the v2 launch token/command/epoch' \
  "${LIFECYCLE_ENV[@]}" SESSION=stale-completion TARGET_LEARNING_ITERATION=1 \
  STATUS_STALE_SECONDS=1 STATUS_STARTUP_GRACE_SECONDS=5 bash batch_ne.sh status
if grep -F 'run_state=completed' "${TMP_DIR}/stale_completion.out" >/dev/null; then
  fail 'stale completion marker was accepted for a different launch token'
fi

# A valid completion marker while tmux remains alive is finalizing, not
# completed. It remains subject to stale-log health checks.
FINALIZING_EPOCH=$((NOW - 10))
write_lifecycle_active finalizing running "${NEW_TOKEN}" "${COMMAND_SHA}" "${FINALIZING_EPOCH}"
write_lifecycle_log finalizing "${NEW_TOKEN}" "${COMMAND_SHA}" "${FINALIZING_EPOCH}"
printf 'HOLOSOMA_RUN_COMPLETE target_iteration=1 checkpoint=/tmp/model_00001.pt\n' \
  >>"$(lifecycle_log_path finalizing)"
touch -d '10 seconds ago' "$(lifecycle_log_path finalizing)"
: >"${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.finalizing.session"
printf '%s\n' "${NEW_TOKEN}" >"${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.finalizing.token"
printf '%s\n' "${COMMAND_SHA}" >"${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.finalizing.command_sha256"
printf '%s\n' "${FINALIZING_EPOCH}" >"${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.finalizing.launch_epoch"
expect_failure \
  "${TMP_DIR}/finalizing_stale.out" \
  'training log has not changed' \
  "${LIFECYCLE_ENV[@]}" SESSION=finalizing TARGET_LEARNING_ITERATION=1 \
  STATUS_STALE_SECONDS=1 STATUS_STARTUP_GRACE_SECONDS=5 bash batch_ne.sh status
grep -F 'run_state=finalizing target_iteration=1' "${TMP_DIR}/finalizing_stale.out" >/dev/null ||
  fail 'live tmux with a completion marker was not reported as finalizing'
if grep -F 'run_state=completed' "${TMP_DIR}/finalizing_stale.out" >/dev/null; then
  fail 'live tmux was incorrectly reported completed'
fi

# Once the exact tmux session exits, the same bound marker is authoritative.
rm -f "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.finalizing.session" \
  "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.finalizing.token" \
  "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.finalizing.command_sha256" \
  "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.finalizing.launch_epoch"
"${LIFECYCLE_ENV[@]}" SESSION=finalizing TARGET_LEARNING_ITERATION=1 \
  STATUS_STALE_SECONDS=1 STATUS_STARTUP_GRACE_SECONDS=5 \
  bash batch_ne.sh status >"${TMP_DIR}/completed.out"
grep -F 'run_state=completed target_iteration=1' "${TMP_DIR}/completed.out" >/dev/null ||
  fail 'bound completion marker after tmux exit was not reported completed'

# Prepublished launching metadata is healthy during startup grace even before
# any node has created its log or tmux session.
STARTING_NOW=$(date +%s)
write_lifecycle_active startup-grace launching "${NEW_TOKEN}" pending "${STARTING_NOW}"
rm -f "$(lifecycle_log_path startup-grace)" \
  "${FAKE_TMUX_STATE_DIR}/${LIFECYCLE_NODE}.startup-grace.session"
"${LIFECYCLE_ENV[@]}" SESSION=startup-grace TARGET_LEARNING_ITERATION=1 \
  STATUS_STALE_SECONDS=1 STATUS_STARTUP_GRACE_SECONDS=30 \
  bash batch_ne.sh status >"${TMP_DIR}/startup_grace.out"
grep -F 'run_state=starting' "${TMP_DIR}/startup_grace.out" >/dev/null ||
  fail 'startup grace incorrectly rejected a not-yet-created log'

expect_failure \
  "${TMP_DIR}/unsafe_session_path.out" \
  'SESSION must be a safe basename' \
  "${BATCH_ENV[@]}" SESSION='../../unsafe' bash batch_ne.sh status
expect_failure \
  "${TMP_DIR}/unsafe_session_space.out" \
  'SESSION must be a safe basename' \
  "${BATCH_ENV[@]}" SESSION='unsafe session' bash batch_ne.sh status

# A payload much larger than the failed inline-ssh command remains stdin data;
# only the short content-addressed script path may appear in the tmux argv.
LONG_SCHEDULE_NOTES=$(head -c 70000 /dev/zero | tr '\0' x)
"${BATCH_ENV[@]}" FORCE_EIGHT_GPU_CONFIG=1 SCHEDULE_NOTES="${LONG_SCHEDULE_NOTES}" \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_long_launch_payload.out"
for canonical_semantic_line in \
  'unset FORCE_EIGHT_GPU_CONFIG' \
  'unset PERCEPTION_INTO_POLICY_MODULES' \
  'unset RESET_TO_DEFAULT_POSE' \
  'export PERCEPTION_INTO_POLICY_MODULES=True' \
  'export RESET_TO_DEFAULT_POSE=False' \
  'export HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES=True' \
  'export HOLOSOMA_RESET_TO_DEFAULT_POSE=False'; do
  [[ "$(grep -Fxc -- "${canonical_semantic_line}" \
      "${TMP_DIR}/batch_long_launch_payload.out")" -eq 1 ]] ||
    fail "generated one-node payload must contain exactly one canonical semantic line: ${canonical_semantic_line}"
done
unset canonical_semantic_line
if grep -E '^(export )?FORCE_EIGHT_GPU_CONFIG=1$' \
    "${TMP_DIR}/batch_long_launch_payload.out" >/dev/null; then
  fail 'controller ambient FORCE_EIGHT_GPU_CONFIG=1 leaked into the generated node payload'
fi
[[ "$(grep -Fxc -- 'export NPROC=1' \
    "${TMP_DIR}/batch_long_launch_payload.out")" -eq 1 ]] ||
  fail 'ambient FORCE_EIGHT_GPU_CONFIG changed the generated payload NPROC contract'
long_payload_bytes=$(sed -nE 's/.* stream launch script .* bytes=([0-9]+)$/\1/p' \
  "${TMP_DIR}/batch_long_launch_payload.out" | tail -1)
[[ "${long_payload_bytes}" =~ ^[0-9]+$ && "${long_payload_bytes}" -gt 70000 ]] \
  || fail "large launch payload was not streamed intact: bytes=${long_payload_bytes:-missing}"
if awk '/tmux new-session/ && length($0) >= 4096 { found=1 } END { exit(found ? 0 : 1) }' \
  "${TMP_DIR}/batch_long_launch_payload.out"; then
  fail 'large streamed payload leaked back into the tmux argv'
fi
grep -E 'tmux new-session -d -s .*\.run_control/train-[0-9a-f]{64}\.sh 8>&-$' \
  "${TMP_DIR}/batch_long_launch_payload.out" >/dev/null

"${BATCH_ENV[@]}" \
  NODES='test-node-a test-node-b' \
  NNODES=2 \
  NPROC=2 \
  CUDA_VISIBLE_DEVICES=0,1 \
  FIXED_BC_EVAL_LOG_INTERVAL=37 \
  FIXED_BC_GUARD_ENABLED=False \
  HOLOSOMA_MOTION_METRICS_INTERVAL=19 \
  TORCH_DIST_BACKEND=NCCL \
  NCCL_LIB_SHA256="${NCCL_LIB_SHA_SENTINEL}" \
  TORCH_DIST_TIMEOUT_SEC=47 \
  HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC=53 \
  MAX_RESTARTS=0 \
  SAVE_INTERVAL=1000 \
  HOLOSOMA_GLOO_GRAD_REDUCE=false \
  HOLOSOMA_GLOO_BARRIER=yes \
  HOLOSOMA_GLOO_SMALL_COLLECTIVES=on \
  HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=yes \
  HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=true \
  HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER=false \
  HOLOSOMA_RANK_VISIBLE_DEVICES=on \
  PYTHON_RUNTIME_SITEPACKAGES="/home/ubuntu/FAR/holosoma_runs/.runtime/python/python-runtime-v2-${NCCL_LIB_SHA_SENTINEL}/site-packages" \
  PYTHON_RUNTIME_MANIFEST_SHA256="${NCCL_LIB_SHA_SENTINEL}" \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_launcher_sentinels.out"
grep -F '[INFO] Launching 2 nodes x 2 GPUs' "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'export FIXED_BC_EVAL_LOG_INTERVAL=37' "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'export HOLOSOMA_MOTION_METRICS_INTERVAL=19' "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'export TORCH_DIST_BACKEND=nccl' "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F "export NCCL_LIB_SHA256=${NCCL_LIB_SHA_SENTINEL}" "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F "export NCCL_LIB_DIR=/home/ubuntu/FAR/holosoma_runs/.runtime/nccl/${NCCL_LIB_SHA_SENTINEL}" \
  "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'export TORCH_DIST_TIMEOUT_SEC=47' "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'export HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC=53' "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'export MAX_RESTARTS=0' "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'export SAVE_INTERVAL=1000' "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'export HOLOSOMA_GLOO_GRAD_REDUCE=0' "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'export HOLOSOMA_GLOO_BARRIER=1' "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'export HOLOSOMA_GLOO_SMALL_COLLECTIVES=1' "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'export HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1' "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'export HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=1' "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'export HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER=0' "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'export HOLOSOMA_RANK_VISIBLE_DEVICES=1' "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'hierarchical_grad_reduce=${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE} hierarchical_small_collectives=${HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES} cpu_leader=${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER} hierarchical_pg_timeout_sec=${HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC} hierarchical_small_scope=eligible_integral_verdict_control_only floating_reductions=flat_gloo' \
  "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'rank_visible_devices=${HOLOSOMA_RANK_VISIBLE_DEVICES}' \
  "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'python_runtime_train_overlay_verified=${PYTHON_RUNTIME_SITEPACKAGES}' \
  "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'python_runtime_pre_intent_verified=$SITE_PACKAGES' \
  "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null ||
  fail 'launch with an overlay must verify its live binding on every node before intent publication'
pre_intent_line=$(grep -nF 'python_runtime_pre_intent_verified=$SITE_PACKAGES' \
  "${TMP_DIR}/batch_launcher_sentinels.out" | head -1 | cut -d: -f1)
intent_preflight_line=$(grep -nF '[INFO] Preflighting launch intent on ' \
  "${TMP_DIR}/batch_launcher_sentinels.out" | head -1 | cut -d: -f1)
[[ "${pre_intent_line}" =~ ^[1-9][0-9]*$ \
      && "${intent_preflight_line}" =~ ^[1-9][0-9]*$ \
      && "${pre_intent_line}" -lt "${intent_preflight_line}" ]] ||
  fail 'Python runtime pre-intent barrier did not precede launch-intent preflight'
unset pre_intent_line intent_preflight_line
grep -F 'save_interval=${SAVE_INTERVAL} fixed_bc_eval_log_interval=${FIXED_BC_EVAL_LOG_INTERVAL} motion_metrics_interval=${HOLOSOMA_MOTION_METRICS_INTERVAL}' \
  "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'torch_dist_backend=${TORCH_DIST_BACKEND} timeout_sec=${TORCH_DIST_TIMEOUT_SEC} max_restarts=${MAX_RESTARTS}' \
  "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F "NCCL_RUNTIME_EXPECTED_SHA256=${NCCL_LIB_SHA_SENTINEL}" \
  "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'actual_nccl_lib_sha256=$(sha256sum ${NCCL_RUNTIME_LIB} | awk' \
  "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F "export LD_PRELOAD=/home/ubuntu/FAR/holosoma_runs/.runtime/nccl/${NCCL_LIB_SHA_SENTINEL}/libnccl.so.2" \
  "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'ctypes.CDLL(None).ncclGetVersion' "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'Path("/proc/self/maps")' "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'NCCL runtime mapping mismatch after torch import' "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null
grep -F 'nccl_lib_dir=${NCCL_LIB_DIR} nccl_lib_sha256=${NCCL_LIB_SHA256:-<not-required>}' \
  "${TMP_DIR}/batch_launcher_sentinels.out" >/dev/null

expect_failure \
  "${TMP_DIR}/batch_invalid_backend.out" \
  'TORCH_DIST_BACKEND must be nccl or gloo' \
  "${BATCH_ENV[@]}" TORCH_DIST_BACKEND=mpi bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_nccl_missing_lib_hash.out" \
  'NCCL_LIB_SHA256 is required when TORCH_DIST_BACKEND=nccl' \
  "${BATCH_ENV[@]}" TORCH_DIST_BACKEND=nccl NCCL_LIB_SHA256= bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_nccl_lib_hash.out" \
  'NCCL_LIB_SHA256 must be a 64-character lowercase SHA256 hex digest' \
  "${BATCH_ENV[@]}" NCCL_LIB_SHA256=ABCDEF bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_incomplete_python_runtime_overlay.out" \
  'PYTHON_RUNTIME_SITEPACKAGES and PYTHON_RUNTIME_MANIFEST_SHA256 must be set together' \
  "${BATCH_ENV[@]}" \
  PYTHON_RUNTIME_SITEPACKAGES=/home/ubuntu/FAR/holosoma_runs/.runtime/python/test/site-packages \
  bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_mismatched_python_runtime_overlay_identity.out" \
  'PYTHON_RUNTIME_SITEPACKAGES must exactly bind its manifest identity' \
  "${BATCH_ENV[@]}" \
  PYTHON_RUNTIME_SITEPACKAGES=/home/ubuntu/FAR/holosoma_runs/.runtime/python/python-runtime-v2-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/site-packages \
  PYTHON_RUNTIME_MANIFEST_SHA256="${NCCL_LIB_SHA_SENTINEL}" \
  bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_incomplete_python_runtime_archive.out" \
  'PYTHON_RUNTIME_ARCHIVE and PYTHON_RUNTIME_ARCHIVE_SHA256 must be set together' \
  "${BATCH_ENV[@]}" \
  PYTHON_RUNTIME_ARCHIVE="${TMP_DIR}/missing-runtime.tar.gz" \
  bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_python_runtime_archive_digest.out" \
  'PYTHON_RUNTIME_ARCHIVE_SHA256 must be a 64-character lowercase SHA256 hex digest' \
  "${BATCH_ENV[@]}" \
  PYTHON_RUNTIME_ARCHIVE="${TMP_DIR}/missing-runtime.tar.gz" \
  PYTHON_RUNTIME_ARCHIVE_SHA256=invalid \
  bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_prepare_runtime_without_archive.out" \
  'prepare/all requires PYTHON_RUNTIME_ARCHIVE and PYTHON_RUNTIME_ARCHIVE_SHA256 when the runtime overlay is enabled' \
  "${BATCH_ENV[@]}" \
  PYTHON_RUNTIME_SITEPACKAGES="/home/ubuntu/FAR/holosoma_runs/.runtime/python/python-runtime-v2-${NCCL_LIB_SHA_SENTINEL}/site-packages" \
  PYTHON_RUNTIME_MANIFEST_SHA256="${NCCL_LIB_SHA_SENTINEL}" \
  PREPARE_DATA=0 bash batch_ne.sh prepare

invalid_runtime_archive="${TMP_DIR}/python-runtime-v2-${NCCL_LIB_SHA_SENTINEL}.tar.gz"
printf 'invalid-runtime-archive\n' >"${invalid_runtime_archive}"
chmod 444 "${invalid_runtime_archive}"
controller_transfer_count_before=$(find /tmp -maxdepth 1 -type d \
  -name "holosoma-python-runtime-transfer.$(id -u).*" -user "$(id -u)" | wc -l)
expect_failure \
  "${TMP_DIR}/batch_wrong_python_runtime_archive_digest.out" \
  'Python runtime archive changed or failed its exact SHA256 contract' \
  "${BATCH_ENV[@]}" \
  PYTHON_RUNTIME_SITEPACKAGES="/home/ubuntu/FAR/holosoma_runs/.runtime/python/python-runtime-v2-${NCCL_LIB_SHA_SENTINEL}/site-packages" \
  PYTHON_RUNTIME_MANIFEST_SHA256="${NCCL_LIB_SHA_SENTINEL}" \
  PYTHON_RUNTIME_ARCHIVE="${invalid_runtime_archive}" \
  PYTHON_RUNTIME_ARCHIVE_SHA256="${NCCL_LIB_SHA_SENTINEL}" \
  PREPARE_DATA=0 bash batch_ne.sh prepare
controller_transfer_count_after=$(find /tmp -maxdepth 1 -type d \
  -name "holosoma-python-runtime-transfer.$(id -u).*" -user "$(id -u)" | wc -l)
[[ "${controller_transfer_count_after}" == "${controller_transfer_count_before}" ]] ||
  fail 'failed controller runtime authentication leaked a private transfer snapshot'
runtime_digest_idle_calls=$(grep -c '^\[DRY_RUN\] bounded-ssh ' \
  "${TMP_DIR}/batch_wrong_python_runtime_archive_digest.out" || true)
[[ "${runtime_digest_idle_calls}" == 1 ]] ||
  fail 'controller runtime digest mismatch did not perform exactly one mandatory bounded early idle probe'
grep -F 'PROBE_PHASE=early' \
  "${TMP_DIR}/batch_wrong_python_runtime_archive_digest.out" >/dev/null ||
  fail 'controller runtime digest mismatch remote call was not the mandatory early idle probe'
grep -F 'selected_gpu_idle_preflight_ok' \
  "${TMP_DIR}/batch_wrong_python_runtime_archive_digest.out" >/dev/null ||
  fail 'controller runtime digest mismatch did not bind the selected-GPU idle contract'
if grep -F '[DRY_RUN] scp ' \
    "${TMP_DIR}/batch_wrong_python_runtime_archive_digest.out" >/dev/null; then
  fail 'controller runtime digest mismatch reached runtime/source transfer'
fi
unset invalid_runtime_archive controller_transfer_count_before controller_transfer_count_after
unset runtime_digest_idle_calls

# A minimal controller-authentic archive is sufficient for dry-run because no
# remote extraction occurs.  Exercise the complete ordered prepare protocol
# without depending on any pre-existing runtime cache.
runtime_controller_fixture="${TMP_DIR}/runtime-controller-fixture"
mkdir -p "${runtime_controller_fixture}/tree/site-packages"
printf 'fixture runtime manifest payload\n' \
  >"${runtime_controller_fixture}/tree/site-packages/.holosoma-runtime-manifest.sha256"
runtime_controller_manifest_sha=$(sha256sum \
  "${runtime_controller_fixture}/tree/site-packages/.holosoma-runtime-manifest.sha256" \
  | awk '{print $1}')
runtime_controller_id="python-runtime-v2-${runtime_controller_manifest_sha}"
runtime_controller_archive="${runtime_controller_fixture}/${runtime_controller_id}.tar.gz"
tar -czf "${runtime_controller_archive}" \
  -C "${runtime_controller_fixture}/tree" site-packages
chmod 444 "${runtime_controller_archive}"
runtime_controller_archive_sha=$(sha256sum "${runtime_controller_archive}" | awk '{print $1}')
runtime_controller_site="/home/ubuntu/FAR/holosoma_runs/.runtime/python/${runtime_controller_id}/site-packages"
controller_transfer_count_before=$(find /tmp -maxdepth 1 -type d \
  -name "holosoma-python-runtime-transfer.$(id -u).*" -user "$(id -u)" | wc -l)
"${BATCH_ENV[@]}" \
  PYTHON_RUNTIME_SITEPACKAGES="${runtime_controller_site}" \
  PYTHON_RUNTIME_MANIFEST_SHA256="${runtime_controller_manifest_sha}" \
  PYTHON_RUNTIME_ARCHIVE="${runtime_controller_archive}" \
  PYTHON_RUNTIME_ARCHIVE_SHA256="${runtime_controller_archive_sha}" \
  PREPARE_DATA=0 bash batch_ne.sh prepare \
  >"${TMP_DIR}/batch_valid_python_runtime_prepare.out"
controller_transfer_count_after=$(find /tmp -maxdepth 1 -type d \
  -name "holosoma-python-runtime-transfer.$(id -u).*" -user "$(id -u)" | wc -l)
[[ "${controller_transfer_count_after}" == "${controller_transfer_count_before}" ]] ||
  fail 'successful dry-run runtime prepare leaked its private controller snapshot'
for ordered_runtime_marker in \
  'controller_python_runtime_archive_snapshot_verified=' \
  'for spec in ".runtime:755" ".runtime/python:700"' \
  '--probe-only' \
  'Runtime transfer token path already exists.' \
  '[DRY_RUN] scp /tmp/holosoma-python-runtime-transfer.' \
  'python_runtime_prepare_installed=$SITE_PACKAGES' \
  'python_runtime_prepare_overlay_verified=${PYTHON_RUNTIME_SITEPACKAGES}'; do
  marker_line=$(grep -nF -- "${ordered_runtime_marker}" \
    "${TMP_DIR}/batch_valid_python_runtime_prepare.out" | head -1 | cut -d: -f1)
  [[ "${marker_line}" =~ ^[1-9][0-9]*$ ]] ||
    fail "valid runtime prepare omitted ordered stage marker: ${ordered_runtime_marker}"
  if [[ -n "${previous_marker_line:-}" \
        && "${marker_line}" -le "${previous_marker_line}" ]]; then
    fail "valid runtime prepare reordered stage marker: ${ordered_runtime_marker}"
  fi
  previous_marker_line=${marker_line}
done
unset runtime_controller_fixture runtime_controller_manifest_sha runtime_controller_id \
  runtime_controller_archive runtime_controller_archive_sha runtime_controller_site \
  controller_transfer_count_before controller_transfer_count_after \
  ordered_runtime_marker marker_line previous_marker_line
expect_failure \
  "${TMP_DIR}/batch_invalid_fixed_bc_interval.out" \
  'FIXED_BC_EVAL_LOG_INTERVAL must be a canonical integer in [1, 2147483647]' \
  "${BATCH_ENV[@]}" FIXED_BC_EVAL_LOG_INTERVAL=0 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_fixed_bc_guard_bool.out" \
  'FIXED_BC_GUARD_ENABLED must be a boolean' \
  "${BATCH_ENV[@]}" FIXED_BC_GUARD_ENABLED=invalid bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_fixed_bc_guard_patience.out" \
  'FIXED_BC_GUARD_CONSECUTIVE_EVALS must be a canonical integer in [1, 2147483647]' \
  "${BATCH_ENV[@]}" FIXED_BC_GUARD_CONSECUTIVE_EVALS=0 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_fixed_bc_guard_ratio.out" \
  'FIXED_BC_GUARD_MAX_REFERENCE_RATIO must be finite and > 0' \
  "${BATCH_ENV[@]}" FIXED_BC_GUARD_MAX_REFERENCE_RATIO=0 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_too_small_fixed_bc_guard_ratio.out" \
  'FIXED_BC_GUARD_MAX_REFERENCE_RATIO must be finite and >= 1' \
  "${BATCH_ENV[@]}" FIXED_BC_GUARD_MAX_REFERENCE_RATIO=0.5 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_fixed_bc_guard_absolute.out" \
  'FIXED_BC_GUARD_ABSOLUTE_MAX_MU_MSE must be finite and > 0' \
  "${BATCH_ENV[@]}" FIXED_BC_GUARD_ABSOLUTE_MAX_MU_MSE=0 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_fixed_bc_guard_reference_after_start.out" \
  'FIXED_BC_GUARD_REFERENCE_END_EPOCH < FIXED_BC_GUARD_START_EPOCH' \
  "${BATCH_ENV[@]}" FIXED_BC_GUARD_REFERENCE_END_EPOCH=4900 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_fixed_bc_guard_too_few_reference_evals.out" \
  'requires at least 3 reference evaluations' \
  "${BATCH_ENV[@]}" FIXED_BC_EVAL_LOG_INTERVAL=600 FIXED_BC_GUARD_START_EPOCH=6000 \
  bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_fixed_bc_guard_unaligned_reference.out" \
  'REFERENCE_END_EPOCH to be divisible by FIXED_BC_EVAL_LOG_INTERVAL' \
  "${BATCH_ENV[@]}" FIXED_BC_GUARD_REFERENCE_END_EPOCH=650 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_fixed_bc_guard_unaligned_start.out" \
  'START_EPOCH to be divisible by FIXED_BC_EVAL_LOG_INTERVAL' \
  "${BATCH_ENV[@]}" FIXED_BC_GUARD_START_EPOCH=4950 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_fixed_bc_guard_non_bc_reference.out" \
  'reference window must remain pure BC' \
  "${BATCH_ENV[@]}" PPO_START_COEFF=0.1 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_disabled_fixed_bc_guard_noncanonical_start.out" \
  'Disabled fixed-BC guard requires FIXED_BC_GUARD_START_EPOCH=-1' \
  "${BATCH_ENV[@]}" FIXED_BC_GUARD_ENABLED=False FIXED_BC_GUARD_START_EPOCH=7 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_fixed_bc_guard_insufficient_trip_window.out" \
  'must have enough evaluations to reach FIXED_BC_GUARD_CONSECUTIVE_EVALS' \
  "${BATCH_ENV[@]}" FIXED_BC_GUARD_START_EPOCH=39900 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_motion_metrics_interval.out" \
  'HOLOSOMA_MOTION_METRICS_INTERVAL must be a canonical integer in [1, 2147483647]' \
  "${BATCH_ENV[@]}" HOLOSOMA_MOTION_METRICS_INTERVAL=invalid bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_save_interval.out" \
  'SAVE_INTERVAL must be a canonical integer in [1, 2147483647]' \
  "${BATCH_ENV[@]}" SAVE_INTERVAL=0 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_noncanonical_scientific_save_interval.out" \
  'Scientific AS launch requires SAVE_INTERVAL=1000 exactly; got 100' \
  "${BATCH_ENV[@]}" SAVE_INTERVAL=100 bash batch_ne.sh launch
if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' \
    "${TMP_DIR}/batch_noncanonical_scientific_save_interval.out" >/dev/null; then
  fail 'non-1000 scientific save interval must fail before snapshot construction or remote actions'
fi
expect_failure \
  "${TMP_DIR}/batch_invalid_dist_timeout.out" \
  'TORCH_DIST_TIMEOUT_SEC must be a canonical integer in [1, 2147483647]' \
  "${BATCH_ENV[@]}" TORCH_DIST_TIMEOUT_SEC=0 bash batch_ne.sh launch
for invalid_hierarchical_timeout in 0 0300 "${HUGE_UNSIGNED_DECIMAL}"; do
  invalid_hierarchical_timeout_output="${TMP_DIR}/batch_invalid_hierarchical_timeout_${invalid_hierarchical_timeout}.out"
  expect_failure \
    "${invalid_hierarchical_timeout_output}" \
    'HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC must be a canonical integer in [1, 2147483647]' \
    "${BATCH_ENV[@]}" \
    HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC="${invalid_hierarchical_timeout}" \
    bash batch_ne.sh launch
done
unset invalid_hierarchical_timeout invalid_hierarchical_timeout_output
expect_failure \
  "${TMP_DIR}/batch_invalid_max_restarts.out" \
  'Scientific launch requires MAX_RESTARTS=0 exactly' \
  "${BATCH_ENV[@]}" MAX_RESTARTS=-1 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_gloo_grad_reduce.out" \
  'HOLOSOMA_GLOO_GRAD_REDUCE must be a boolean' \
  "${BATCH_ENV[@]}" HOLOSOMA_GLOO_GRAD_REDUCE=invalid bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_gloo_barrier.out" \
  'HOLOSOMA_GLOO_BARRIER must be a boolean' \
  "${BATCH_ENV[@]}" HOLOSOMA_GLOO_BARRIER=invalid bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_gloo_small_collectives.out" \
  'HOLOSOMA_GLOO_SMALL_COLLECTIVES must be a boolean' \
  "${BATCH_ENV[@]}" HOLOSOMA_GLOO_SMALL_COLLECTIVES=invalid bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_hierarchical_small_collectives.out" \
  'HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES must be a boolean' \
  "${BATCH_ENV[@]}" HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=invalid bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_hierarchical_cpu_leader.out" \
  'HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER must be a boolean' \
  "${BATCH_ENV[@]}" HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER=invalid bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_nondeterministic_rng_resume.out" \
  'ALLOW_NONDETERMINISTIC_RNG_RESUME must be a boolean' \
  "${BATCH_ENV[@]}" ALLOW_NONDETERMINISTIC_RNG_RESUME=invalid bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_fixed_bc_eval_reset_on_resume.out" \
  'ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME must be a boolean' \
  "${BATCH_ENV[@]}" ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME=invalid bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_invalid_runtime_drift_on_resume.out" \
  'ALLOW_RUNTIME_DRIFT_ON_RESUME must be a boolean' \
  "${BATCH_ENV[@]}" ALLOW_RUNTIME_DRIFT_ON_RESUME=invalid bash batch_ne.sh launch

expect_failure \
  "${TMP_DIR}/batch_invalid_required_terminal_target.out" \
  'HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET must be a canonical integer' \
  "${BATCH_ENV[@]}" HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET=08 \
  bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_terminal_target_without_policy_init.out" \
  'HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET requires RESUME_FROM_BOX=1' \
  "${BATCH_ENV[@]}" HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET=8 \
  bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_terminal_target_wrong_active_world.out" \
  'BOX_POLICY_INIT_EXPECTED_WORLD_SIZE must equal the active NNODES*NPROC topology' \
  "${BATCH_ENV[@]}" RESUME_FROM_BOX=1 \
  HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET=8 \
  BOX_POLICY_INIT_EXPECTED_SHA256="${NCCL_LIB_SHA_SENTINEL}" \
  BOX_POLICY_INIT_CONTROL_CACHE_ROOT="${TMP_DIR}/not-used" \
  BOX_POLICY_INIT_EXPECTED_WORLD_SIZE=2 \
  bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_terminal_target_remote_ref.out" \
  'accepts only an authenticated control-local checkpoint' \
  "${BATCH_ENV[@]}" RESUME_FROM_BOX=1 \
  HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET=8 \
  BOX_POLICY_INIT_EXPECTED_SHA256="${NCCL_LIB_SHA_SENTINEL}" \
  BOX_POLICY_INIT_CONTROL_CACHE_ROOT="${TMP_DIR}/not-used" \
  BOX_POLICY_INIT_EXPECTED_WORLD_SIZE=1 \
  BOX_POLICY_INIT_EXPECTED_WANDB_RUN_PATH=zihanw22/carry-any/run \
  BOX_POLICY_INIT_EXPECTED_SOURCE_SNAPSHOT_ID="src-${NCCL_LIB_SHA_SENTINEL}" \
  SOURCE_SNAPSHOT_ID="src-${NCCL_LIB_SHA_SENTINEL}" \
  BOX_POLICY_INIT_REQUIRE_FRESH_SOURCE=1 \
  BOX_POLICY_INIT_REF=wandb://entity/project/run/model_8.pt \
  bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_terminal_target_missing_digest.out" \
  'requires BOX_POLICY_INIT_EXPECTED_SHA256' \
  "${BATCH_ENV[@]}" RESUME_FROM_BOX=1 \
  HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET=8 \
  BOX_POLICY_INIT_REF="${TMP_DIR}/model_39999.pt" \
  bash batch_ne.sh launch

printf '%s\n' 'control-local policy init fixture' >"${TMP_DIR}/model_39999.pt"
policy_init_sha=$(sha256sum "${TMP_DIR}/model_39999.pt" | awk '{print $1}')
"${BATCH_ENV[@]}" \
  RESUME_FROM_BOX=1 \
  BOX_POLICY_INIT_REF="${TMP_DIR}/model_39999.pt" \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_local_policy_init.out"
grep -F "control_box_policy_init=${TMP_DIR}/model_39999.pt sha256=${policy_init_sha}" \
  "${TMP_DIR}/batch_local_policy_init.out" >/dev/null
grep -E "\[DRY_RUN\] scp ${TMP_DIR}/model_39999\.pt test-node:/home/ubuntu/FAR/holosoma_runs/src-[0-9a-f]{64}/\.checkpoint_cache/policy_init/\.${policy_init_sha}\.0\.[0-9]+\.incoming" \
  "${TMP_DIR}/batch_local_policy_init.out" >/dev/null
grep -E "FINAL=/home/ubuntu/FAR/holosoma_runs/src-[0-9a-f]{64}/\.checkpoint_cache/policy_init/${policy_init_sha}\.pt" \
  "${TMP_DIR}/batch_local_policy_init.out" >/dev/null
grep -F "EXPECTED=${policy_init_sha}" "${TMP_DIR}/batch_local_policy_init.out" >/dev/null
grep -E "export BOX_POLICY_INIT_REF=/home/ubuntu/FAR/holosoma_runs/src-[0-9a-f]{64}/\.checkpoint_cache/policy_init/${policy_init_sha}\.pt" \
  "${TMP_DIR}/batch_local_policy_init.out" >/dev/null
grep -F "export BOX_POLICY_INIT_EXPECTED_SHA256=${policy_init_sha}" \
  "${TMP_DIR}/batch_local_policy_init.out" >/dev/null
grep -F 'Resolved policy-init SHA256 mismatch:' "${TMP_DIR}/batch_local_policy_init.out" >/dev/null

wandb_policy_init_ref='wandb://entity/boxer/run123/model_39999.pt'
"${BATCH_ENV[@]}" \
  RESUME_FROM_BOX=1 \
  BOX_POLICY_INIT_REF="${wandb_policy_init_ref}" \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_wandb_policy_init.out"
grep -F "export BOX_POLICY_INIT_REF=${wandb_policy_init_ref}" "${TMP_DIR}/batch_wandb_policy_init.out" >/dev/null
grep -F 'export BOX_POLICY_INIT_EXPECTED_SHA256=' "${TMP_DIR}/batch_wandb_policy_init.out" >/dev/null
if grep -F '[DRY_RUN] scp ' "${TMP_DIR}/batch_wandb_policy_init.out" | grep -F '/.checkpoint_cache/policy_init/' >/dev/null; then
  fail 'W&B policy-init references must remain remote artifacts rather than control-local scp inputs'
fi

expect_failure \
  "${TMP_DIR}/batch_missing_local_policy_init.out" \
  'Control-local BOX_POLICY_INIT_REF does not exist' \
  "${BATCH_ENV[@]}" RESUME_FROM_BOX=1 BOX_POLICY_INIT_REF="${TMP_DIR}/missing.pt" bash batch_ne.sh launch

ln -s "${TMP_DIR}/model_39999.pt" "${TMP_DIR}/terminal-policy-init-link.pt"
expect_failure \
  "${TMP_DIR}/batch_terminal_target_symlink.out" \
  'refuses a symlink BOX_POLICY_INIT_REF' \
  "${BATCH_ENV[@]}" RESUME_FROM_BOX=1 \
  HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET=8 \
  BOX_POLICY_INIT_EXPECTED_SHA256="${policy_init_sha}" \
  BOX_POLICY_INIT_CONTROL_CACHE_ROOT="${TMP_DIR}/not-used" \
  BOX_POLICY_INIT_EXPECTED_WORLD_SIZE=1 \
  BOX_POLICY_INIT_EXPECTED_WANDB_RUN_PATH=zihanw22/carry-any/run \
  BOX_POLICY_INIT_EXPECTED_SOURCE_SNAPSHOT_ID="src-${NCCL_LIB_SHA_SENTINEL}" \
  SOURCE_SNAPSHOT_ID="src-${NCCL_LIB_SHA_SENTINEL}" \
  BOX_POLICY_INIT_REQUIRE_FRESH_SOURCE=1 \
  BOX_POLICY_INIT_REF="${TMP_DIR}/terminal-policy-init-link.pt" \
  bash batch_ne.sh launch

IFS=$'\t' read -r terminal_source_snapshot_id terminal_source_snapshot_archive \
  terminal_source_snapshot_archive_sha terminal_source_manifest_sha < <(
    bash scripts/build_run_snapshot.sh \
      --repo-root "${REPO_ROOT}" \
      --cache-root "${TMP_DIR}/snapshot-cache"
  )
mapfile -t terminal_policy_init_fixture < <(
  python - "${TMP_DIR}/terminal-policy-init-controller" \
    "${terminal_source_snapshot_id}" "${terminal_source_manifest_sha}" <<'PY'
import sys
from pathlib import Path

from tests.unit.test_terminal_policy_init_controller import (
    _fresh_finalized_provenance,
    _payload,
    _publish_private,
)

payload = _payload()
payload["training_provenance"] = _fresh_finalized_provenance()
payload["wandb_run_path"] = "zihanw22/carry-any/replacement-canary"
payload["training_provenance"]["source_snapshot_id"] = sys.argv[2]
payload["training_provenance"]["source_manifest_sha256"] = sys.argv[3]
cache_root, checkpoint, digest = _publish_private(Path(sys.argv[1]), payload)
print(cache_root)
print(checkpoint)
print(digest)
PY
)
[[ "${#terminal_policy_init_fixture[@]}" -eq 3 ]] ||
  fail 'terminal policy-init fixture publisher returned an incomplete identity'
terminal_policy_init_cache_root=${terminal_policy_init_fixture[0]}
terminal_policy_init_path=${terminal_policy_init_fixture[1]}
terminal_policy_init_sha=${terminal_policy_init_fixture[2]}
"${BATCH_ENV[@]}" RESUME_FROM_BOX=1 \
  HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET=8 \
  BOX_POLICY_INIT_EXPECTED_SHA256="${terminal_policy_init_sha}" \
  BOX_POLICY_INIT_CONTROL_CACHE_ROOT="${terminal_policy_init_cache_root}" \
  BOX_POLICY_INIT_EXPECTED_WORLD_SIZE=1 \
  BOX_POLICY_INIT_EXPECTED_WANDB_RUN_PATH=zihanw22/carry-any/replacement-canary \
  BOX_POLICY_INIT_EXPECTED_SOURCE_SNAPSHOT_ID="${terminal_source_snapshot_id}" \
  BOX_POLICY_INIT_REQUIRE_FRESH_SOURCE=1 \
  BOX_POLICY_INIT_REF="${terminal_policy_init_path}" \
  SOURCE_SNAPSHOT_ID="${terminal_source_snapshot_id}" \
  SOURCE_SNAPSHOT_ARCHIVE="${terminal_source_snapshot_archive}" \
  SOURCE_SNAPSHOT_ARCHIVE_SHA256="${terminal_source_snapshot_archive_sha}" \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_terminal_policy_init_verified.out"
grep -F '[INFO] controller_terminal_policy_init_verified completed_iteration=7 next_iteration=8 target_iteration=8 world_size=1' \
  "${TMP_DIR}/batch_terminal_policy_init_verified.out" >/dev/null ||
  fail 'controller did not authenticate the complete terminal policy-init source'
grep -F 'export HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET=8' \
  "${TMP_DIR}/batch_terminal_policy_init_verified.out" >/dev/null ||
  fail 'worker node control lost the required terminal target gate'
grep -F 'validate_policy_init_terminal_source_payload' batch_ne.sh >/dev/null ||
  fail 'node policy-init architecture preflight does not revalidate terminal proof before actor inspection'
unset terminal_policy_init_fixture terminal_policy_init_cache_root
unset terminal_policy_init_path terminal_policy_init_sha
unset terminal_source_snapshot_id terminal_source_snapshot_archive
unset terminal_source_snapshot_archive_sha terminal_source_manifest_sha

"${BATCH_ENV[@]}" bash batch_ne.sh prepare >"${TMP_DIR}/batch_prepare.out"
grep -E '\[DRY_RUN\] scp .*/src-[0-9a-f]{64}\.tar\.gz test-node:/home/ubuntu/FAR/holosoma_runs/\.incoming/' \
  "${TMP_DIR}/batch_prepare.out" >/dev/null
grep -F 'installed_verified_source_snapshot=' "${TMP_DIR}/batch_prepare.out" >/dev/null
grep -F 'remote dirty repository source is never modified' "${TMP_DIR}/batch_prepare.out" >/dev/null
if grep -E '(^|[[:space:]])git (fetch|pull)' "${TMP_DIR}/batch_prepare.out" >/dev/null; then
  fail 'isolated prepare must never fetch or pull in the remote dirty repository'
fi

printf '%s\n' 'content-addressed custom bank fixture' >"${TMP_DIR}/source-bank.tar"
fallback_sha=$(sha256sum "${TMP_DIR}/source-bank.tar" | awk '{print $1}')
"${BATCH_ENV[@]}" \
  RUN_STAMP=fallback-contract \
  PREPARE_COPY_SCRIPT=cp_corl \
  NFS_CORL_BANK="${TMP_DIR}/source-bank.tar" \
  LOCAL_BANK_NAME=installed-bank \
  CORL_SOLID80_BANK_NAME=installed-bank \
  EXPECTED_CLIP_COUNT=1 \
  bash batch_ne.sh prepare >"${TMP_DIR}/batch_data_fallback.out"
grep -F "control_corl_package=${TMP_DIR}/source-bank.tar sha256=${fallback_sha}" \
  "${TMP_DIR}/batch_data_fallback.out" >/dev/null
grep -E "\[DRY_RUN\] scp ${TMP_DIR}/source-bank\.tar test-node:/home/ubuntu/FAR/holosoma_runs/\.data-packages/\.incoming/${fallback_sha}\.[0-9]+\.tar" \
  "${TMP_DIR}/batch_data_fallback.out" >/dev/null
grep -F "FINAL=/home/ubuntu/FAR/holosoma_runs/.data-packages/${fallback_sha}.tar" \
  "${TMP_DIR}/batch_data_fallback.out" >/dev/null
grep -F "EXPECTED=${fallback_sha}" "${TMP_DIR}/batch_data_fallback.out" >/dev/null
grep -F 'flock -w 10 -x 9' "${TMP_DIR}/batch_data_fallback.out" >/dev/null ||
  fail 'data-package publication must use the validated bounded exclusive lock'
if rg -n '^[[:space:]]*flock[[:space:]]+[0-9]' "${TMP_DIR}/batch_data_fallback.out"; then
  fail 'data-package dry-run payload must never regress to an unbounded remote flock'
fi
grep -F 'mv -T --no-clobber "${INCOMING}" "${FINAL}"' "${TMP_DIR}/batch_data_fallback.out" >/dev/null
grep -F 'Refusing to overwrite corrupt data-package cache entry' "${TMP_DIR}/batch_data_fallback.out" >/dev/null
grep -F "NFS_CORL_BANK=/home/ubuntu/FAR/holosoma_runs/.data-packages/${fallback_sha}.tar" \
  "${TMP_DIR}/batch_data_fallback.out" >/dev/null
if grep -F '/home/ubuntu/FAR/holosoma/.data-packages/' "${TMP_DIR}/batch_data_fallback.out" >/dev/null; then
  fail 'custom data-package fallback cache must stay outside the source/asset repository'
fi
expect_failure \
  "${TMP_DIR}/data_cache_inside_repo.out" \
  'REMOTE_DATA_PACKAGE_CACHE must be outside the mutable asset/source repository' \
  "${BATCH_ENV[@]}" \
  REMOTE_DATA_PACKAGE_CACHE=/home/ubuntu/FAR/holosoma_runs/../holosoma/.data-packages \
  bash batch_ne.sh prepare

"${BATCH_ENV[@]}" bash batch_ne.sh status >"${TMP_DIR}/batch_status.out"
grep -F 'training log has not changed for' "${TMP_DIR}/batch_status.out" >/dev/null
grep -F 'NaN/Inf/non-finite value or Python training exception detected' "${TMP_DIR}/batch_status.out" >/dev/null
grep -F 'HOLOSOMA_PROGRESS completed_iteration=([0-9]+)' "${TMP_DIR}/batch_status.out" >/dev/null
grep -F 'Heartbeat: iter[[:space:]]+([0-9]+)' "${TMP_DIR}/batch_status.out" >/dev/null
if grep -F '[Ii]ter[ =:]+' "${TMP_DIR}/batch_status.out" >/dev/null; then
  fail 'batch status must not use a generic iter= matcher that mistakes configuration for progress'
fi

status_fixture="${TMP_DIR}/status_iteration_fixture.log"
cat >"${status_fixture}" <<'EOF'
[INFO] start_at_timestep_prob=0.25 start_at_timestep_until_iter=2500
[INFO] ppo_schedule step_epochs=700 end_iteration=6300
2026-07-12 20:00:00 | INFO | Entering PPO.learn at iteration 7.
2026-07-12 20:00:01 | INFO | Heartbeat: iter 8 starting rollout
 Learning iteration 9/40000
2026-07-12 20:00:02 | INFO | HOLOSOMA_PROGRESS completed_iteration=10
EOF
parsed_status_iter=$(sed -nE \
  -e 's/.*HOLOSOMA_PROGRESS completed_iteration=([0-9]+)([^0-9].*)?$/\1/p' \
  -e 's/.*Heartbeat: iter[[:space:]]+([0-9]+)([^0-9].*)?$/\1/p' \
  -e 's/.*Entering PPO\.learn at iteration[[:space:]]+([0-9]+)\..*$/\1/p' \
  -e 's/.*Learning iteration[[:space:]]+([0-9]+)\/[0-9]+.*$/\1/p' \
  "${status_fixture}" | tail -1)
[[ "${parsed_status_iter}" == "10" ]] || fail "status parser returned ${parsed_status_iter:-empty}, expected 10"
sed -i '/Entering PPO.learn/d; /Heartbeat: iter/d; /Learning iteration/d; /HOLOSOMA_PROGRESS/d' "${status_fixture}"
if sed -nE \
  -e 's/.*HOLOSOMA_PROGRESS completed_iteration=([0-9]+)([^0-9].*)?$/\1/p' \
  -e 's/.*Heartbeat: iter[[:space:]]+([0-9]+)([^0-9].*)?$/\1/p' \
  -e 's/.*Entering PPO\.learn at iteration[[:space:]]+([0-9]+)\..*$/\1/p' \
  -e 's/.*Learning iteration[[:space:]]+([0-9]+)\/[0-9]+.*$/\1/p' \
  "${status_fixture}" | grep -q .; then
  fail 'status parser mistook configuration iter=2500 for training progress'
fi

for as_launcher in distill_as_button_solid.sh distill_as_button.sh distill_as_perception.sh; do
  if grep -nE '(^|[;&|()[:space:]])python3([[:space:]]|$)' "${as_launcher}"; then
    fail "${as_launcher} must route every Python invocation through PYTHON_BIN"
  fi
done
grep -F 'export HOLOSOMA_REQUIRE_CONTACT_INTERVAL_COVERAGE=1' distill_as_perception.sh >/dev/null ||
  fail 'the real AS distill path must require complete runtime contact-window coverage'
grep -F 'export HOLOSOMA_REQUIRE_CONTACT_TARGET_COVERAGE=1' distill_as_perception.sh >/dev/null ||
  fail 'the real AS distill path must require complete runtime contact-target coverage'
grep -F 'ALLOW_PARTIAL_CONTACT_SIDECARS requires ENABLE_OFFLINE_CONTACT_GUIDANCE=False.' \
  distill_as_perception.sh >/dev/null ||
  fail 'partial rollout sidecars must be forbidden while positive offline contact guidance is enabled'
grep -F 'export HOLOSOMA_REQUIRE_CONTACT_INTERVAL_COVERAGE=0' distill_as_perception.sh >/dev/null ||
  fail 'an explicit no-contact rollout profile must be able to retain honest partial interval coverage'
grep -F 'export HOLOSOMA_REQUIRE_CONTACT_TARGET_COVERAGE=0' distill_as_perception.sh >/dev/null ||
  fail 'an explicit no-contact rollout profile must be able to retain honest partial target coverage'
grep -F 'AS_CONTACT_AWARE_DEPLOYMENT_HISTORY_LENGTH=5' distill_as_perception.sh >/dev/null ||
  fail 'AS contact-aware-history must pin its deployable actor history to five frames'
grep -F 'STUDENT_PROPRIO_HISTORY_LENGTH CRITIC_PROPRIO_HISTORY_LENGTH' \
  distill_as_perception.sh >/dev/null ||
  fail 'AS contact-aware-history must pin both serialized student and critic history overrides'
grep -F -- '--observation.groups.actor_obs_root_contact_aware.history-length=1' \
  distill_as_perception.sh >/dev/null ||
  fail 'AS contact-aware-history must pin its sparse command group to one frame'
grep -F -- '--observation.groups.actor_obs_proprio_with_actions_no_linvel.history-length="${STUDENT_PROPRIO_HISTORY_LENGTH}"' \
  distill_box_perception.sh >/dev/null ||
  fail 'delegated AS launch must serialize the fixed student history into the effective training config'
expect_failure \
  "${TMP_DIR}/as_history_length4.out" \
  'contact-aware-history is a fixed deployment contract with CONTACT_AWARE_HISTORY_LENGTH=5; got 4.' \
  env AS_CONTACT_AWARE=1 AS_CONTACT_AWARE_HISTORY=1 CONTACT_AWARE_HISTORY_LENGTH=4 \
  TEACHER_CHECKPOINT=/does/not/need/to/exist.pt bash distill_as_perception.sh
expect_failure \
  "${TMP_DIR}/as_history_student_override.out" \
  'contact-aware-history requires STUDENT_PROPRIO_HISTORY_LENGTH=5; got 4.' \
  env AS_CONTACT_AWARE=1 AS_CONTACT_AWARE_HISTORY=1 CONTACT_AWARE_HISTORY_LENGTH=5 \
  STUDENT_PROPRIO_HISTORY_LENGTH=4 TEACHER_CHECKPOINT=/does/not/need/to/exist.pt \
  bash distill_as_perception.sh
expect_failure \
  "${TMP_DIR}/as_history_actor_override.out" \
  'contact-aware-history requires exact actor inputs' \
  env AS_CONTACT_AWARE=1 AS_CONTACT_AWARE_HISTORY=1 CONTACT_AWARE_HISTORY_LENGTH=5 \
  STUDENT_ACTOR_INPUTS="['actor_obs_root_contact_aware','actor_obs_proprio','actor_obs_actions']" \
  TEACHER_CHECKPOINT=/does/not/need/to/exist.pt bash distill_as_perception.sh
expect_failure \
  "${TMP_DIR}/as_invalid_perception_policy_modules.out" \
  'PERCEPTION_INTO_POLICY_MODULES must be a boolean. Got: typo' \
  env PERCEPTION_INTO_POLICY_MODULES=typo \
  TEACHER_CHECKPOINT=/does/not/need/to/exist.pt bash distill_as_perception.sh
expect_failure \
  "${TMP_DIR}/as_invalid_reset_to_default_pose.out" \
  'RESET_TO_DEFAULT_POSE must be a boolean. Got: typo' \
  env RESET_TO_DEFAULT_POSE=typo \
  TEACHER_CHECKPOINT=/does/not/need/to/exist.pt bash distill_as_perception.sh
expect_failure \
  "${TMP_DIR}/as_perception_semantic_alias_conflict.out" \
  'HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES conflicts with PERCEPTION_INTO_POLICY_MODULES' \
  env PERCEPTION_INTO_POLICY_MODULES=True \
  HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES=False \
  TEACHER_CHECKPOINT=/does/not/need/to/exist.pt bash distill_as_perception.sh
expect_failure \
  "${TMP_DIR}/as_reset_semantic_alias_conflict.out" \
  'HOLOSOMA_RESET_TO_DEFAULT_POSE conflicts with RESET_TO_DEFAULT_POSE' \
  env RESET_TO_DEFAULT_POSE=False HOLOSOMA_RESET_TO_DEFAULT_POSE=True \
  TEACHER_CHECKPOINT=/does/not/need/to/exist.pt bash distill_as_perception.sh
provenance_compute_line=$(grep -n -m1 'compute_training_provenance.py' distill_as_perception.sh | cut -d: -f1)
for semantic_export in \
  HOLOSOMA_OBJECT_SPAWN_MODE \
  HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE \
  HOLOSOMA_OBJECT_COLLIDER_TYPE \
  HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS \
  HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK \
  HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS \
  HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES \
  HOLOSOMA_RESET_TO_DEFAULT_POSE; do
  semantic_export_line=$(grep -n -m1 "^export ${semantic_export}=" distill_as_perception.sh | cut -d: -f1)
  if [[ -z "${semantic_export_line}" || "${semantic_export_line}" -ge "${provenance_compute_line}" ]]; then
    fail "AS distill must finalize ${semantic_export} before computing training provenance"
  fi
done
unset provenance_compute_line semantic_export semantic_export_line
grep -F -- '--source-digest-only' distill_as_perception.sh >/dev/null ||
  fail 'AS distill launcher must content-address rank-local shard generations'
grep -F -- '_rank_shards/by-source/${AS_RANK_SHARD_SOURCE_DIGEST}/ws${AS_GLOBAL_WORLD_SIZE}' \
  distill_as_perception.sh >/dev/null ||
  fail 'AS distill launcher default rank-local shard root must be immutable by source digest'
grep -F -- '--expected-source-digest "${AS_RANK_SHARD_SOURCE_DIGEST}"' \
  distill_as_perception.sh >/dev/null ||
  fail 'AS distill launcher must fail closed if shard inputs drift after root selection'
grep -F -- 'prepare_immutable_single_slot_bank.py' train_as_general.sh >/dev/null ||
  fail 'AS object-generalist launcher must use a content-addressed single-slot bank'
if grep -nE 'shutil\.rmtree\(child\)|target\.symlink_to\(npz_path' train_as_general.sh; then
  fail 'AS object-generalist launcher must not clear/rebuild a mutable symlink motion view'
fi
if grep -nE '^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]+[[:space:]]+)*(exec[[:space:]]+)?torchrun([[:space:]]|$)|torchrun_args=\([[:space:]]*torchrun([[:space:]]|\)|$)' distill_torso_box.sh; then
  fail 'the AS terminal launcher must start torch.distributed.run through PYTHON_BIN'
fi

GENERALIST_PREPARED_FIXTURE='data/ds_box_data/train_g1_w_obj_prepared'
if [[ -f "${GENERALIST_PREPARED_FIXTURE}/_clip_object_urdf_map.json" ]] \
    && command -v nvidia-smi >/dev/null 2>&1 \
    && nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
  env DRY_RUN=1 NPROC=1 NNODES=1 PER_GPU_ENVS=2 CUDA_VISIBLE_DEVICES=0 \
    ASSERT_NEW_DS_DATA=0 AUTO_PREP_DS_BANK=0 STRICT_DEFAULT_DS_BANK_VALIDATION=0 \
    MOTION_DIR="${GENERALIST_PREPARED_FIXTURE}" \
    OBJECT_SPEC_PATH="${GENERALIST_PREPARED_FIXTURE}/_clip_object_urdf_map.json" \
    TORCH_DIST_BACKEND=gloo HOLOSOMA_RANK_VISIBLE_DEVICES=on \
    bash train_object_generalist_ds.sh >"${TMP_DIR}/generalist_rank_visible.out"
  python3 - "${TMP_DIR}/generalist_rank_visible.out" <<'PY'
from __future__ import annotations

import shlex
import sys
from pathlib import Path

prefix = "[INFO] Final train command:"
commands = [
    shlex.split(line[len(prefix) :])
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.startswith(prefix)
]
if len(commands) != 1:
    raise SystemExit(f"[FAIL] expected one generalist final train command, got {len(commands)}")
args = commands[0]
rank_visible_entry = "src/holosoma/holosoma/train_agent_rank_visible.py"
plain_entry = "src/holosoma/holosoma/train_agent.py"
if args.count(rank_visible_entry) != 1 or plain_entry in args:
    raise SystemExit(
        "[FAIL] rank-visible generalist mode must select only the rank-visible wrapper: "
        f"rank_visible_count={args.count(rank_visible_entry)} plain_present={plain_entry in args}"
    )
PY
fi

"${BATCH_ENV[@]}" \
  RESUME_TRAINING_CKPT='https://wandb.ai/entity/project/runs/run123/files/model_12000.pt?download=1' \
  STUDENT_ACTOR_HIDDEN_DIMS='[512,256,128]' \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_wandb_resume.out"
grep -F 'export RESUME_CKPT=wandb://entity/project/run123/model_12000.pt' \
  "${TMP_DIR}/batch_wandb_resume.out" >/dev/null
grep -F 'training_resume_checkpoint_local=' "${TMP_DIR}/batch_wandb_resume.out" >/dev/null

python - "${TMP_DIR}/local_resume.pt" "${TMP_DIR}/local_resume_bad_actor.pt" <<'PY'
import sys
import torch

def checkpoint(hidden_dims):
    return {
        "iter": 10,
        "experiment_config": {
            "algo": {
                "config": {
                    "module_dict": {
                        "actor": {
                            "type": "MLPPerceptionEncoder",
                            "input_dim": ["actor_obs"],
                            "layer_config": {"hidden_dims": hidden_dims},
                        },
                    }
                }
            }
        },
    }


torch.save(checkpoint([64, 32]), sys.argv[1])
torch.save(checkpoint([True, 32]), sys.argv[2])
PY
local_resume_sha=$(sha256sum "${TMP_DIR}/local_resume.pt" | awk '{print $1}')
"${BATCH_ENV[@]}" RUN_STAMP=contract \
  RESUME_TRAINING_CKPT="${TMP_DIR}/local_resume.pt" \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_local_resume.out"
grep -F "control_training_resume=${TMP_DIR}/local_resume.pt sha256=${local_resume_sha}" \
  "${TMP_DIR}/batch_local_resume.out" >/dev/null
grep -E "\[DRY_RUN\] scp ${TMP_DIR}/local_resume\.pt test-node:/home/ubuntu/FAR/holosoma_runs/src-[0-9a-f]{64}/\.checkpoint_cache/training_resume/\.${local_resume_sha}\.0\.[0-9]+\.incoming" \
  "${TMP_DIR}/batch_local_resume.out" >/dev/null
grep -E "FINAL=/home/ubuntu/FAR/holosoma_runs/src-[0-9a-f]{64}/\.checkpoint_cache/training_resume/${local_resume_sha}\.pt" \
  "${TMP_DIR}/batch_local_resume.out" >/dev/null
grep -E "export RESUME_SOURCE_REF=/home/ubuntu/FAR/holosoma_runs/src-[0-9a-f]{64}/\.checkpoint_cache/training_resume/${local_resume_sha}\.pt" \
  "${TMP_DIR}/batch_local_resume.out" >/dev/null
grep -F "export RESUME_SOURCE_EXPECTED_SHA256=${local_resume_sha}" \
  "${TMP_DIR}/batch_local_resume.out" >/dev/null
grep -F 'Resolved training-resume SHA256 mismatch:' "${TMP_DIR}/batch_local_resume.out" >/dev/null

expect_failure \
  "${TMP_DIR}/batch_local_resume_bad_actor.out" \
  'Invalid actor hidden dims in training-resume checkpoint: [True, 32]' \
  "${BATCH_ENV[@]}" RESUME_TRAINING_CKPT="${TMP_DIR}/local_resume_bad_actor.pt" \
  bash batch_ne.sh launch
if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' \
  "${TMP_DIR}/batch_local_resume_bad_actor.out" >/dev/null; then
  fail 'invalid local resume actor contract must fail before snapshot construction or remote actions'
fi

"${BATCH_ENV[@]}" \
  TEACHER_CHECKPOINT="${TMP_DIR}/local_resume.pt" \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_local_teacher.out"
grep -F "control_teacher_checkpoint=${TMP_DIR}/local_resume.pt sha256=${local_resume_sha}" \
  "${TMP_DIR}/batch_local_teacher.out" >/dev/null
grep -E "\[DRY_RUN\] scp ${TMP_DIR}/local_resume\.pt test-node:/home/ubuntu/FAR/holosoma_runs/src-[0-9a-f]{64}/\.checkpoint_cache/teacher/\.${local_resume_sha}\.0\.[0-9]+\.incoming" \
  "${TMP_DIR}/batch_local_teacher.out" >/dev/null
grep -E "export TEACHER_CHECKPOINT=/home/ubuntu/FAR/holosoma_runs/src-[0-9a-f]{64}/\.checkpoint_cache/teacher/${local_resume_sha}\.pt" \
  "${TMP_DIR}/batch_local_teacher.out" >/dev/null
grep -F "export TEACHER_CHECKPOINT_EXPECTED_SHA256=${local_resume_sha}" \
  "${TMP_DIR}/batch_local_teacher.out" >/dev/null
grep -F 'Resolved teacher-checkpoint SHA256 mismatch:' "${TMP_DIR}/batch_local_teacher.out" >/dev/null

expect_failure \
  "${TMP_DIR}/batch_missing_local_resume.out" \
  'Control-local RESUME_TRAINING_CKPT does not exist' \
  "${BATCH_ENV[@]}" RESUME_TRAINING_CKPT="${TMP_DIR}/missing-resume.pt" bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_missing_local_teacher.out" \
  'Control-local TEACHER_CHECKPOINT does not exist' \
  "${BATCH_ENV[@]}" TEACHER_CHECKPOINT="${TMP_DIR}/missing-teacher.pt" bash batch_ne.sh launch

"${BATCH_ENV[@]}" \
  UNIFORM_T1_WINDOW_TARGET_SAMPLE_FRAC=0.61 \
  EXPORT_ONNX=True \
  CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE=precomputed-turn-then-forward \
  ZERO_ROOT_COMMAND_WHEN_DROP_ACTIVE=True \
  CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION=False \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_optional_forwarding.out"
grep -F 'export UNIFORM_T1_WINDOW_TARGET_SAMPLE_FRAC=0.61' "${TMP_DIR}/batch_optional_forwarding.out" >/dev/null
grep -F 'export EXPORT_ONNX=True' "${TMP_DIR}/batch_optional_forwarding.out" >/dev/null
grep -F 'export CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE=precomputed_turn_then_forward' \
  "${TMP_DIR}/batch_optional_forwarding.out" >/dev/null
grep -F 'export ZERO_ROOT_COMMAND_WHEN_DROP_ACTIVE=True' "${TMP_DIR}/batch_optional_forwarding.out" >/dev/null
grep -F 'export CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION=False' "${TMP_DIR}/batch_optional_forwarding.out" >/dev/null

"${BATCH_ENV[@]}" \
  NFS_CORL_BANK=/tmp/source-bank \
  LOCAL_BANK_NAME=installed-bank \
  bash batch_ne.sh launch >"${TMP_DIR}/batch_custom_bank.out"
grep -F 'export CORL_SOLID80_BANK_NAME=installed-bank' "${TMP_DIR}/batch_custom_bank.out" >/dev/null

expect_failure \
  "${TMP_DIR}/same_run_without_checkpoint.out" \
  'RESUME_WANDB_RUN_ID requests same-run logging resume but RESUME_TRAINING_CKPT is empty' \
  "${BATCH_ENV[@]}" RESUME_WANDB_RUN_ID=run123 bash batch_ne.sh launch

expect_failure \
  "${TMP_DIR}/bare_teacher.out" \
  'TEACHER_CHECKPOINT must identify an exact .pt artifact' \
  "${BATCH_ENV[@]}" TEACHER_CHECKPOINT='https://wandb.ai/entity/project/runs/run123' bash batch_ne.sh launch

expect_failure \
  "${TMP_DIR}/non_pt_resume.out" \
  'RESUME_TRAINING_CKPT must be a .pt checkpoint path or wandb:// URI' \
  "${BATCH_ENV[@]}" RESUME_TRAINING_CKPT='/tmp/model.ckpt' bash batch_ne.sh launch

expect_failure \
  "${TMP_DIR}/target_alias_conflict.out" \
  'TARGET_LEARNING_ITERATION and legacy NUM_LEARNING_ITERATIONS disagree' \
  "${BATCH_ENV[@]}" TARGET_LEARNING_ITERATION=40000 NUM_LEARNING_ITERATIONS=41000 bash batch_ne.sh launch

expect_failure \
  "${TMP_DIR}/invalid_t1_target.out" \
  'UNIFORM_T1_WINDOW_TARGET_SAMPLE_FRAC must be a finite probability in [0, 1]' \
  "${BATCH_ENV[@]}" UNIFORM_T1_WINDOW_TARGET_SAMPLE_FRAC=1.1 bash batch_ne.sh launch

expect_failure \
  "${TMP_DIR}/flow_policy_init.out" \
  'drop-button policy-init profile cannot initialize a flow actor' \
  env RESUME_FROM_BOX=1 STUDENT_POLICY_TYPE=flow bash distill_as_button.sh

expect_failure \
  "${TMP_DIR}/dual_policy_init.out" \
  'RESUME_FROM_BOX=1 is incompatible with the dual-button actor' \
  env RESUME_FROM_BOX=1 bash distill_as_dual_button.sh

expect_failure \
  "${TMP_DIR}/generic_policy_init.out" \
  'Generic distill_as_perception.sh cannot safely use RESUME_FROM_BOX=1' \
  env RESUME_FROM_BOX=1 bash distill_as_perception.sh

expect_failure \
  "${TMP_DIR}/dual_previous_policy_init.out" \
  'saved single-button policy and is incompatible with the dual-button actor' \
  env RESUME_FROM_PREVIOUS=1 bash distill_as_dual_button_solid.sh

expect_failure \
  "${TMP_DIR}/flow_previous_policy_init.out" \
  'saved single-button MLP policy profile' \
  env RESUME_FROM_PREVIOUS=1 STUDENT_POLICY_TYPE=flow bash distill_as_button_solid.sh

# These dry-run checks require the repository's local box/AS fixture banks. Keep
# the static syntax/contracts above portable, and exercise the full chain when
# those fixtures are available (as on the training workspace).
BOX_MOTION_FIXTURE="outputs/motion_bank_success_box_0_92_0p3"
if [[ -d "${BOX_MOTION_FIXTURE}" ]]; then
  BOX_TEACHER_REF=wandb://zihanw22/boxer/u5lguxvl/model_17000.pt
  if [[ -f .teacher_checkpoints/model_17000.pt ]]; then
    BOX_TEACHER_REF="${REPO_ROOT}/.teacher_checkpoints/model_17000.pt"
  fi
  BOX_ENV=(
    env
    DRY_RUN=1
    USE_LEGACY_DS=0
    TEACHER_CHECKPOINT="${BOX_TEACHER_REF}"
    TEACHER_ACTOR_OBS_HISTORY_LENGTH=5
  )

  "${BOX_ENV[@]}" NPROC=2 NNODES=2 NODE_RANK=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29571 PER_GPU_ENVS=10 \
    bash distill_box_button.sh >"${TMP_DIR}/button_multinode.out"
  grep -F 'global_world_size=4 per_gpu_envs=10 total_num_envs=40' "${TMP_DIR}/button_multinode.out" >/dev/null
  grep -F -- '-m torch.distributed.run --nnodes=2 --node_rank=1 --master_addr=127.0.0.1 --nproc_per_node=2' \
    "${TMP_DIR}/button_multinode.out" >/dev/null

  expect_failure \
    "${TMP_DIR}/button_nondivisible.out" \
    'TOTAL_NUM_ENVS must be divisible by global world size' \
    "${BOX_ENV[@]}" NPROC=2 NNODES=2 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29571 TOTAL_NUM_ENVS=42 bash distill_box_button.sh

  expect_failure \
    "${TMP_DIR}/button_missing_rendezvous.out" \
    'Multi-node launch requires explicit shared MASTER_ADDR and MASTER_PORT' \
    "${BOX_ENV[@]}" NPROC=2 NNODES=2 NODE_RANK=0 PER_GPU_ENVS=10 bash distill_box_button.sh

  "${BOX_ENV[@]}" NPROC=1 NNODES=1 PER_GPU_ENVS=2 UNIFORM_T1_WINDOW_TARGET_SAMPLE_FRAC=0.61 \
    bash distill_box_button.sh >"${TMP_DIR}/button_t1_target.out"
  [[ "$(grep -o -- '--command.setup-terms.motion-command.params.motion-config.uniform-t1-window-target-sample-frac=0.61' \
    "${TMP_DIR}/button_t1_target.out" | wc -l)" -eq 1 ]] || fail 'T1 target flag must appear exactly once'

  "${BOX_ENV[@]}" NPROC=1 NNODES=1 PER_GPU_ENVS=2 SHOO7SR1_NEAR03_DEBUG=1 EXPORT_ONNX=False \
    bash distill_box_perception.sh >"${TMP_DIR}/shoo_export_false.out"
  grep -F 'export_onnx=False' "${TMP_DIR}/shoo_export_false.out" >/dev/null
  grep -F -- '--training.export-onnx=False' "${TMP_DIR}/shoo_export_false.out" >/dev/null
  grep -F 'baseline_action_semantics=actor_obs_proprio_with_actions_no_linvel_includes_current_action_history' \
    "${TMP_DIR}/shoo_export_false.out" >/dev/null

  "${BOX_ENV[@]}" NPROC=1 NNODES=1 PER_GPU_ENVS=2 \
    bash distill_box_perception.sh termination:g1_29dof_wbt_generalist \
    >"${TMP_DIR}/box_forwarded_component_selector.out"
  grep -F 'termination:g1_29dof_wbt_generalist' \
    "${TMP_DIR}/box_forwarded_component_selector.out" >/dev/null ||
    fail 'forwarded Tyro component selector was consumed as a positional run name'
  if grep -F 'run_name=termination:g1_29dof_wbt_generalist' \
    "${TMP_DIR}/box_forwarded_component_selector.out" >/dev/null; then
    fail 'forwarded Tyro component selector silently changed the run name'
  fi

  expect_failure \
    "${TMP_DIR}/shoo_action_history.out" \
    'SHOO7SR1_OBS_VARIANT=action_history is not implemented' \
    "${BOX_ENV[@]}" SHOO7SR1_NEAR03_DEBUG=1 SHOO7SR1_OBS_VARIANT=action_history bash distill_box_perception.sh
  expect_failure \
    "${TMP_DIR}/shoo_false_no_actions.out" \
    'saved shoo7sr1 baseline uses actor_obs_proprio_with_actions_no_linvel' \
    "${BOX_ENV[@]}" SHOO7SR1_NEAR03_DEBUG=1 SHOO7SR1_OBS_VARIANT=no_linvel_no_actions bash distill_box_perception.sh
fi

AS_BANK_FIXTURE='data/ds_as_data/carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success133_final0p5'
if [[ -d "${AS_BANK_FIXTURE}" && -f .teacher_checkpoints/model_67000.pt ]]; then
  env DRY_RUN=1 NPROC=1 NNODES=1 PER_GPU_ENVS=2 TORCH_DIST_BACKEND=gloo \
    ACTOR_LR=0.000321 CRITIC_LR=0.000654 ACTOR_MIN_NOISE_STD=0.017 \
    INIT_NOISE_STD=0.23 ENTROPY_COEF=0.004 \
    PPO_START_EPOCH=1 DAGGER_END_EPOCH=9 PPO_START_COEFF=0.2 PPO_TARGET_COEFF=0.8 \
    PPO_SCHEDULE_STEP_EPOCHS=2 DAGGER_LOSS_COEF=1.7 DAGGER_MATCH_STD=True \
    PPO_START_NOISE_STD=0.31 PPO_START_NOISE_STD_UNTIL_COEFF=0.25 \
    START_AT_TIMESTEP_ZERO_PROB=0.15 START_AT_TIMESTEP_ZERO_PROB_END=0.85 \
    START_AT_TIMESTEP_ZERO_PROB_START_ITER=0 START_AT_TIMESTEP_ZERO_PROB_END_ITER=17 \
    FREEZE_AT_TIMESTEP_ZERO_PROB=0.05 FREEZE_AT_TIMESTEP_ZERO_PROB_END=0.25 \
    FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=1 FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=15 \
    CAMERA_PITCH_DEG=11.25 CAMERA_FAR=4.0 CAMERA_MAX_DISTANCE=3.5 \
    CAMERA_WARP_HOLE_PROB=0.13 CAMERA_WARP_ADDITIVE_NOISE_STD=0.02 \
    CAMERA_WARP_DEPTH_OFFSET_STD=0.01 \
    AS_PUSH_INTERVAL_S='[0.75,1.25]' AS_MAX_PUSH_VEL='[0.11,0.22,0.33,0.44,0.55,0.66]' \
    TEACHER_CHECKPOINT="${REPO_ROOT}/.teacher_checkpoints/model_67000.pt" \
    bash distill_as_perception.sh success133 contact-aware-history \
    >"${TMP_DIR}/as_contact_history5.out"
  grep -F 'student_actor_inputs=['"'"'actor_obs_root_contact_aware'"'"','"'"'actor_obs_proprio_with_actions_no_linvel'"'"']' \
    "${TMP_DIR}/as_contact_history5.out" >/dev/null ||
    fail 'AS history5 dry-run did not preserve the canonical two-group actor contract'
  grep -F 'student_proprio_history_length=5' "${TMP_DIR}/as_contact_history5.out" >/dev/null ||
    fail 'AS history5 dry-run did not apply the five-frame student history override'
  python3 - "${TMP_DIR}/as_contact_history5.out" <<'PY'
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

prefix = "[INFO] final_train_command:"
lines = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
commands = [
    shlex.split(line[len(prefix) :])
    for line in lines
    if line.startswith(prefix)
]
if len(commands) != 1:
    raise SystemExit(f"[FAIL] expected one AS history5 final_train_command, got {len(commands)}")
args = commands[0]
provenance_prefix = "[INFO] training_provenance="
provenance_payloads = [
    json.loads(line[len(provenance_prefix) :])
    for line in lines
    if line.startswith(provenance_prefix)
]
if len(provenance_payloads) != 1:
    raise SystemExit(
        f"[FAIL] expected one AS history5 training provenance payload, got {len(provenance_payloads)}"
    )
semantic_environment = provenance_payloads[0]["environment"]["execution_runtime"][
    "semantic_environment"
]
expected_semantics = {
    "HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES": "True",
    "HOLOSOMA_RESET_TO_DEFAULT_POSE": "False",
}
actual_semantics = {name: semantic_environment.get(name) for name in expected_semantics}
if actual_semantics != expected_semantics:
    raise SystemExit(
        "[FAIL] AS provenance did not serialize the canonical worker semantic environment: "
        f"actual={actual_semantics!r} expected={expected_semantics!r}"
    )
if args.count("HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES=True") != 1:
    raise SystemExit(
        "[FAIL] terminal worker command did not retain the provenance-bound perception injection value"
    )
expected_once = {
    "--observation.groups.actor_obs_root_contact_aware.history-length=1",
    "--observation.groups.actor_obs_proprio_with_actions_no_linvel.history-length=5",
    "--observation.groups.critic_proprio_history.history-length=5",
    "--algo.config.actor-learning-rate=0.000321",
    "--algo.config.critic-learning-rate=0.000654",
    "--algo.config.module-dict.actor.min-noise-std=0.017",
    "--algo.config.init-noise-std=0.23",
    "--algo.config.entropy-coef=0.004",
    "--algo.config.distill.ppo-start-epoch=1",
    "--algo.config.distill.dagger-end-epoch=9",
    "--algo.config.distill.ppo-start-coeff=0.2",
    "--algo.config.distill.ppo-target-coeff=0.8",
    "--algo.config.distill.ppo-schedule-step-epochs=2",
    "--algo.config.distill.dagger-loss-coef=1.7",
    "--algo.config.distill.dagger-match-std=True",
    "--algo.config.distill.fixed-bc-guard-enabled=False",
    "--algo.config.distill.fixed-bc-guard-reference-end-epoch=600",
    "--algo.config.distill.fixed-bc-guard-max-reference-ratio=2.0",
    "--algo.config.distill.fixed-bc-guard-absolute-max-mu-mse=0.160",
    "--algo.config.distill.fixed-bc-guard-start-epoch=-1",
    "--algo.config.distill.fixed-bc-guard-consecutive-evals=3",
    "--algo.config.distill.ppo-start-noise-std=0.31",
    "--algo.config.distill.ppo-start-noise-std-until-coeff=0.25",
    "--perception.camera-pitch-deg=11.25",
    "--perception.camera-far=4.0",
    "--perception.max-distance=3.5",
    "--perception.camera-warp-hole-prob=0.13",
    "--perception.camera-warp-additive-noise-std=0.02",
    "--perception.camera-warp-depth-offset-std=0.01",
    "--randomization.setup_terms.push_randomizer_state.params.push_interval_s=[0.75,1.25]",
    "--randomization.setup_terms.push_randomizer_state.params.max_push_vel=[0.11,0.22,0.33,0.44,0.55,0.66]",
    "--command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=0.15",
    "--command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end=0.85",
    "--command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-start-iter=0",
    "--command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end-iter=17",
    "--command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob=0.05",
    "--command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end=0.25",
    "--command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-start-iter=1",
    "--command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end-iter=15",
}
bad_counts = {arg: args.count(arg) for arg in expected_once if args.count(arg) != 1}
if bad_counts:
    raise SystemExit(
        "[FAIL] AS history5 effective training command does not contain its exact serialized "
        f"history contract once each: {bad_counts!r}"
    )
actor_option = "--algo.config.module-dict.actor.input-dim"
if args.count(actor_option) != 1:
    raise SystemExit(f"[FAIL] AS history5 actor input option count is {args.count(actor_option)}")
actor_index = args.index(actor_option)
expected_actor = "['actor_obs_root_contact_aware','actor_obs_proprio_with_actions_no_linvel']"
if actor_index + 1 >= len(args) or args[actor_index + 1] != expected_actor:
    raise SystemExit("[FAIL] AS history5 final actor input contract drifted")
PY

  env DRY_RUN=1 NPROC=1 NNODES=1 PER_GPU_ENVS=2 TORCH_DIST_BACKEND=gloo SCHEDULE_VARIANT=dagger_mix SEED=123 \
    HOLOSOMA_DISABLE_AUTO_RESET=1 HOLOSOMA_DISABLE_CLIP_END_RESET=1 \
    HOLOSOMA_DISABLE_MOTION_END_RESET=1 \
    TEACHER_CHECKPOINT="${REPO_ROOT}/.teacher_checkpoints/model_67000.pt" \
    bash distill_as_button.sh >"${TMP_DIR}/as_dagger_mix.out"
  grep -F 'schedule_variant=dagger_mix' "${TMP_DIR}/as_dagger_mix.out" >/dev/null
  grep -F 'target=0.0 step_epochs=0' "${TMP_DIR}/as_dagger_mix.out" >/dev/null
  grep -F 'schedule_name=as_success133_final0p5_sparse_root_dagger_mix_contact_drop_button' \
    "${TMP_DIR}/as_dagger_mix.out" >/dev/null
  grep -F 'contact_sidecar_contract_verified' "${TMP_DIR}/as_dagger_mix.out" >/dev/null
  grep -F 'offline_wrist_region_names=["left_wrist","right_wrist"]' \
    "${TMP_DIR}/as_dagger_mix.out" >/dev/null
  grep -F 'offline_contact_region_names=["left_wrist","right_wrist","left_elbow","right_elbow","left_wrist_roll","right_wrist_roll","left_wrist_pitch","right_wrist_pitch","torso"]' \
    "${TMP_DIR}/as_dagger_mix.out" >/dev/null
  grep -F 'training_provenance={"contact_interval_runtime_prepend_compensation":true' \
    "${TMP_DIR}/as_dagger_mix.out" >/dev/null
  grep -F 'student_motion_end_mode=episodic termination_profile=g1_29dof_wbt_generalist' \
    "${TMP_DIR}/as_dagger_mix.out" >/dev/null
  grep -F 'training_reset_contract disable_auto_reset=0 disable_clip_end_reset=0 disable_motion_end_reset=0' \
    "${TMP_DIR}/as_dagger_mix.out" >/dev/null
  grep -F 'termination:g1_29dof_wbt_generalist' "${TMP_DIR}/as_dagger_mix.out" >/dev/null
  grep -F "train_cli_preflight_ok expected_motion_end_mode=episodic termination_terms=['bad_tracking', 'motion_ends', 'timeout']" \
    "${TMP_DIR}/as_dagger_mix.out" >/dev/null
  python3 - "${TMP_DIR}/as_dagger_mix.out" <<'PY'
from __future__ import annotations

import shlex
import sys
from pathlib import Path

lines = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
prefix = "[INFO] final_train_command:"
commands = [shlex.split(line[len(prefix) :]) for line in lines if line.startswith(prefix)]
if len(commands) != 1:
    raise SystemExit(f"[FAIL] expected one final_train_command, got {len(commands)}")
args = commands[0]
if args.count("--training.seed=123") != 1:
    raise SystemExit(
        "[FAIL] canonical training seed must reach the real terminal train command exactly once: "
        f"matches={args.count('--training.seed=123')}"
    )
expected_mix_args = {
    "--algo.config.distill.teacher-action-mix-ratio=0.0",
    "--algo.config.distill.teacher-action-mix-ratio-start=0.7",
    "--algo.config.distill.teacher-action-mix-ratio-end=0.0",
    "--algo.config.distill.teacher-action-mix-ratio-end-iteration=3500",
}
missing_mix_args = sorted(expected_mix_args.difference(args))
if missing_mix_args:
    raise SystemExit(
        "[FAIL] dagger_mix must keep the static ratio disabled while forwarding "
        f"the complete schedule; missing={missing_mix_args!r}"
    )
exp_index = next(i for i, arg in enumerate(args) if arg.startswith("exp:"))
perception_index = next(i for i, arg in enumerate(args) if arg.startswith("perception:"))
termination_index = next(i for i, arg in enumerate(args) if arg.startswith("termination:"))
logger_index = next(i for i, arg in enumerate(args) if arg.startswith("logger:"))
termination_override_index = next(i for i, arg in enumerate(args) if arg.startswith("--termination."))
if not exp_index < perception_index < termination_index < logger_index < termination_override_index:
    raise SystemExit(
        "[FAIL] Tyro component subcommands must follow exp and precede namespace flags: "
        f"exp={exp_index} perception={perception_index} termination={termination_index} "
        f"logger={logger_index} termination_override={termination_override_index}"
    )
PY
  grep -F -- '--command.setup-terms.motion-command.params.motion-config.contact-interval-runtime-prepend-compensation=True' \
    "${TMP_DIR}/as_dagger_mix.out" >/dev/null
fi

CONVEX_BANK='data/ds_as_data/carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success155_bcleb5oi58000_final0p5_primitiveproj_solid80_clean_box_bin_barrel_ball_convexhull'
if [[ -d "${CONVEX_BANK}" ]]; then
  CHECK_ONLY=1 bash distill_as_button_solid_convex.sh >"${TMP_DIR}/convex_check.out"
  grep -F 'selected_solid_clips=' "${TMP_DIR}/convex_check.out" >/dev/null
fi

echo "[PASS] launcher contracts"
