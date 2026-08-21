#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 16 || ! $1 =~ ^(canary|formal)$ || ! $2 =~ ^(gap|spatial)$ ]]; then
  echo "usage: $0 MODE ENCODER EXPECTED_IP SOURCE_ROOT PERSIST_ROOT MASTER_PORT RUN_ID RUN_NAME CONTRACT_PATH CONTRACT_SHA RULE90_PATH RULE90_SHA CANARY_PATH CANARY_SHA COMMIT_SHA TREE_SHA" >&2
  exit 2
fi

readonly MODE=$1 ENCODER_ARM=$2 EXPECTED_IP=$3 SOURCE_ROOT=$4 PERSIST_ROOT=$5 MASTER_PORT=$6
readonly RUN_ID=$7 RUN_NAME=$8 CONTRACT_PATH=$9 CONTRACT_SHA=${10} RULE90_PATH=${11}
readonly RULE90_SHA=${12} CANARY_PATH=${13} CANARY_SHA=${14} COMMIT_SHA=${15} TREE_SHA=${16}
readonly REMOTE_URL=https://github.com/Z1hanW/holosoma
readonly REMOTE_REF=main
readonly WORLD_SIZE=8 ENVIRONMENTS_PER_RANK=2048 TOTAL_ENVIRONMENTS=16384
readonly PYTHON_BIN=/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3.11
readonly PYTHON_RUNTIME_ROOT=/data/holosoma_runs/.runtime/python/python-runtime-v2-dd7ca81fa848917c362b3a239893a7a26f4c89d42b4f85cb515d91622f1690bc
readonly PYTHON_RUNTIME=${PYTHON_RUNTIME_ROOT}/site-packages
readonly PYTHON_RUNTIME_SHA256=dd7ca81fa848917c362b3a239893a7a26f4c89d42b4f85cb515d91622f1690bc
readonly NCCL_ROOT=/home/ubuntu/FAR/holosoma_runs/.runtime/nccl/e4a7aee9c3eecf53fac780441d2f03b578ab8db8874b71f8e391bcec7adb2899
readonly NCCL_SHA256=e4a7aee9c3eecf53fac780441d2f03b578ab8db8874b71f8e391bcec7adb2899
readonly MOTION_DIGEST=307e9662d498bd507b9d17ca9abf74a3654f7bf66ac6ab989c6f19c3889bddef
readonly SOURCE_MOTION_DIR=/data/holosoma_inputs/corl79_plus_debug30_decoupled_turn_forward_v1/by-source/${MOTION_DIGEST}
readonly SOURCE_OBJECT_SPEC_PATH=${SOURCE_MOTION_DIR}/_clip_object_urdf_map.json
readonly TEACHER_ROOT=/data/holosoma_runs/formal_distill_7hvy40000_depthab_ws8x2_20260821/teacher
readonly TEACHER=${TEACHER_ROOT}/model_40000.pt
readonly TEACHER_SHA256=d627b6b5f5a2037761889810c4ef2e158d911941ef8a773f93257a283a68b129
readonly TEACHER_ONNX=${TEACHER_ROOT}/model_40000.onnx
readonly TEACHER_ONNX_SHA256=a003a77abbeface5bde627f7ac8d4289f81820264a31eb9ae31d0cc8ba86ce92
readonly TEACHER_PAIR=${TEACHER_ROOT}/model_40000.pair.json
readonly TEACHER_PAIR_SHA256=859619bd64f76e5ad364c721ea7b9f156ff56e81bae40d11c352169e3f0a5599
readonly IMMUTABLE_DATA_ROOT=/data/holosoma_runs/formal_distill_7hvy40000_depthab_ws8x2_20260821/immutable_data_v2
readonly SINGLE_SLOT_SOURCE_DIGEST=df2514603dffeaa0f2719e1ff90ae0cf969f79024e51dbbdd391b1db7148f5b5
readonly SINGLE_SLOT_VIEW_DIGEST=714b24d1d4a77f2d7103ea84a01173c27fd548fea66b0dd5328591be88dd348d
readonly SINGLE_SLOT_MANIFEST_SHA256=b901afc66d9fefcddb436f6f235cf4b0efd9d5de7a2577f08560b9eb982a3e1d
readonly SINGLE_SLOT_MAP_SHA256=241154190f28300d74f008dd099662a134ff4e73f95a3056e7e260cae1f80cbd
readonly RANK_SHARD_SOURCE_DIGEST=383bdc8b3755645121ca80a986b9930af994874615781c6641d34fc0ade7bc6a
readonly RANK_SHARD_MANIFEST_SHA256=a639242ca42dec7ce7b0a60257a467e3b85960e1b2b9b4a43feeaa87c4c4b1aa

case ${ENCODER_ARM} in
  gap) readonly ENCODER_TYPE=far_tracking_cnn_small ;;
  spatial) readonly ENCODER_TYPE=far_tracking_cnn_spatial_softmax ;;
esac

if [[ ${MODE} == formal ]]; then
  readonly TARGET_ITERATIONS=40000 SAVE_INTERVAL=1000 CURRICULUM_END_ITER=39999
  [[ ${RUN_ID} =~ ^[a-z0-9]{8}$ && ${RUN_NAME} != - && ${CONTRACT_PATH} != - \
     && ${RULE90_PATH} != - && ${CANARY_PATH} != - ]]
else
  readonly TARGET_ITERATIONS=2 SAVE_INTERVAL=2 CURRICULUM_END_ITER=1
fi

check_sha() {
  local expected=$1 path=$2 actual
  [[ -f ${path} && ! -L ${path} ]] || { echo "[ERROR] missing regular integrity input: ${path}" >&2; exit 2; }
  actual=$(sha256sum "${path}" | awk '{print $1}')
  [[ ${actual} == "${expected}" ]] || {
    echo "[ERROR] SHA mismatch: ${path} expected=${expected} actual=${actual}" >&2
    exit 2
  }
}

hostname -I | tr ' ' '\n' | grep -Fxq "${EXPECTED_IP}" || {
  echo "[ERROR] expected node IP ${EXPECTED_IP}, got $(hostname -I)" >&2; exit 2;
}
readonly VERIFY_ROOT=${PERSIST_ROOT}/${MODE}_${ENCODER_ARM}_git_verification
mkdir -p "${VERIFY_ROOT}"
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/verify_formal_git_checkout.py" \
  --source-root "${SOURCE_ROOT}" --remote-url "${REMOTE_URL}" --remote-ref "${REMOTE_REF}" \
  --commit "${COMMIT_SHA}" --tree "${TREE_SHA}" --output "${VERIFY_ROOT}/node_0.json"
readonly GIT_MANIFEST_SHA256=$(git -C "${SOURCE_ROOT}" ls-tree -r --full-tree "${COMMIT_SHA}" | sha256sum | awk '{print $1}')
readonly SOURCE_SNAPSHOT_ID=src-${GIT_MANIFEST_SHA256}

check_sha 2de9ee5ca188b70e877c32dd9f0d2975eea99d11aa077bb077cf06ea9ab897bb "${SOURCE_MOTION_DIR}/manifest.json"
check_sha 70b466aad04837a79f6dd0f4491cb345a73c687209981acd3eb7f4a0365d8f5c "${SOURCE_OBJECT_SPEC_PATH}"
check_sha "${TEACHER_SHA256}" "${TEACHER}"
check_sha "${TEACHER_ONNX_SHA256}" "${TEACHER_ONNX}"
check_sha "${TEACHER_PAIR_SHA256}" "${TEACHER_PAIR}"
check_sha "${NCCL_SHA256}" "${NCCL_ROOT}/libnccl.so.2"
if [[ ${MODE} == formal ]]; then
  check_sha "${CONTRACT_SHA}" "${CONTRACT_PATH}"
  check_sha "${RULE90_SHA}" "${RULE90_PATH}"
  check_sha "${CANARY_SHA}" "${CANARY_PATH}"
fi

# Keep generated data outside the clean Git checkout at one identical absolute
# path on every node.  The single-slot/view digest currently authenticates the
# source paths as well as file bytes, so putting this under SOURCE_ROOT would
# make canary and formal clones receive different execution identities.
readonly LOCAL_MOTION_DIR=${IMMUTABLE_DATA_ROOT}/exact109_source
if [[ ! -d ${LOCAL_MOTION_DIR} ]]; then
  readonly LOCAL_MOTION_INCOMING=${LOCAL_MOTION_DIR}.incoming.$$
  [[ ! -e ${LOCAL_MOTION_INCOMING} ]]
  mkdir -p "$(dirname "${LOCAL_MOTION_DIR}")"
  cp -al "${SOURCE_MOTION_DIR}" "${LOCAL_MOTION_INCOMING}"
  mv "${LOCAL_MOTION_INCOMING}" "${LOCAL_MOTION_DIR}"
fi
readonly MOTION_DIR=${LOCAL_MOTION_DIR}
readonly OBJECT_SPEC_PATH=${MOTION_DIR}/_clip_object_urdf_map.json
readonly CONTACT_ROOT=${MOTION_DIR}/contact_export_corl79_success133_plus_debug30_realmesh_model05000
readonly SINGLE_SLOT_DIR=${MOTION_DIR}/_single_slot_motion_bank/by-source/${SINGLE_SLOT_VIEW_DIGEST}
readonly RANK_SHARD_DIR=${SINGLE_SLOT_DIR}/_rank_shards/by-source/${RANK_SHARD_SOURCE_DIGEST}/ws8
check_sha 2de9ee5ca188b70e877c32dd9f0d2975eea99d11aa077bb077cf06ea9ab897bb "${MOTION_DIR}/manifest.json"
check_sha 70b466aad04837a79f6dd0f4491cb345a73c687209981acd3eb7f4a0365d8f5c "${OBJECT_SPEC_PATH}"
check_sha "${SINGLE_SLOT_MANIFEST_SHA256}" "${SINGLE_SLOT_DIR}/manifest.json"
check_sha "${SINGLE_SLOT_MAP_SHA256}" "${SINGLE_SLOT_DIR}/_clip_object_urdf_map.json"
check_sha "${RANK_SHARD_MANIFEST_SHA256}" "${RANK_SHARD_DIR}/manifest.json"
[[ $(find "${MOTION_DIR}" -maxdepth 1 -type f -name '*.npz' | wc -l) -eq 109 ]]

readonly RUN_ROOT=${PERSIST_ROOT}/${MODE}_${ENCODER_ARM}_${RUN_ID}
readonly LOGGER_BASE_DIR=${RUN_ROOT}/training_logs
readonly SCRATCH_ROOT=/dev/shm/holosoma_distill_7hvy_${MODE}_${ENCODER_ARM}
mkdir -p "${RUN_ROOT}" "${LOGGER_BASE_DIR}" "${PERSIST_ROOT}/wandb" \
  "${SCRATCH_ROOT}/tmp" "${SCRATCH_ROOT}/xdg-cache" "${SCRATCH_ROOT}/robot-usd-cache" \
  "${SCRATCH_ROOT}/object-usd-cache" "${SCRATCH_ROOT}/perception-mesh-cache" \
  "${SCRATCH_ROOT}/derived-data-cache" "${SCRATCH_ROOT}/provenance-cache"

unset BASH_ENV ENV CDPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONOPTIMIZE
unset PYTHONWARNINGS PYTHONBREAKPOINT PYTHONSAFEPATH LD_LIBRARY_PATH LD_PRELOAD
unset WANDB_RUN_ID WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS
export HOME=/home/ubuntu
export PATH="$(dirname "${PYTHON_BIN}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="${PYTHON_RUNTIME}:${SOURCE_ROOT}/src/holosoma:${SOURCE_ROOT}/src/holosoma_inference:${SOURCE_ROOT}/src"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export PYTHON_RUNTIME_SITEPACKAGES=${PYTHON_RUNTIME}
export PYTHON_RUNTIME_MANIFEST_SHA256=${PYTHON_RUNTIME_SHA256}
export HOLOSOMA_PYTHON_RUNTIME_MANIFEST_SHA256=${PYTHON_RUNTIME_SHA256}
export HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY=1
export NCCL_LIB_DIR=${NCCL_ROOT} NCCL_LIB_SHA256=${NCCL_SHA256}
export LD_LIBRARY_PATH=${NCCL_ROOT} LD_PRELOAD=${NCCL_ROOT}/libnccl.so.2
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y HEADLESS=1 OMP_NUM_THREADS=1
export TMPDIR=${SCRATCH_ROOT}/tmp XDG_CACHE_HOME=${SCRATCH_ROOT}/xdg-cache
export WANDB_DIR=${PERSIST_ROOT}/wandb
export HOLOSOMA_RUNTIME_SCRATCH_ROOT=${SCRATCH_ROOT}
export HOLOSOMA_ROBOT_USD_CACHE_DIR=${SCRATCH_ROOT}/robot-usd-cache
export HOLOSOMA_OBJECT_USD_CACHE_DIR=${SCRATCH_ROOT}/object-usd-cache
export HOLOSOMA_PERCEPTION_MESH_CACHE_DIR=${SCRATCH_ROOT}/perception-mesh-cache
export HOLOSOMA_ISAACSIM_KIT_ARGS="--/UJITSO/datastore/localCachePath=${SCRATCH_ROOT}/derived-data-cache --/UJITSO/datastore/localDataStore/largeChunkDiskBudgetMB=1024"
export HOLOSOMA_DATA_PROVENANCE_CACHE_ROOT=${SCRATCH_ROOT}/provenance-cache

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/verify_python_runtime_overlay.py" \
  --site-packages "${PYTHON_RUNTIME}" --manifest-sha256 "${PYTHON_RUNTIME_SHA256}" \
  --require-distribution-closure --require-current-runtime-binding
"${PYTHON_BIN}" -c 'import onnx, onnxruntime; assert onnx.__version__ == "1.21.0"; assert onnxruntime.__version__ == "1.25.1"'

mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,name,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits)
[[ ${#gpu_rows[@]} -eq 8 ]] || { echo "[ERROR] expected exactly 8 GPUs" >&2; exit 2; }
for row in "${gpu_rows[@]}"; do
  [[ ${row} == *"NVIDIA L40S"* && ${row##*, } == 0 ]] || { echo "[ERROR] GPU health: ${row}" >&2; exit 2; }
done
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d') ]]; then
  echo "[ERROR] node is not GPU-idle" >&2; exit 2
fi

if [[ ${MODE} == formal ]]; then
  "${PYTHON_BIN}" - "${CANARY_PATH}" "${COMMIT_SHA}" "${TREE_SHA}" "${SOURCE_SNAPSHOT_ID}" "${ENCODER_TYPE}" <<'PY'
import json, sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text())
expected={"accepted":True,"world_size":8,"environments_per_rank":2048,
          "completed_iterations_per_rank":2,"commit_sha":sys.argv[2],"tree_sha":sys.argv[3],
          "source_snapshot_id":sys.argv[4],"encoder_type":sys.argv[5],
          "teacher_pt_sha256":"d627b6b5f5a2037761889810c4ef2e158d911941ef8a773f93257a283a68b129",
          "ppo_start_coefficient":0.01,"ppo_target_coefficient":1.0,
          "positive_contact_reward":False,"export_onnx":True,
          "onnx_checker_passed":True,"onnxruntime_load_passed":True,
          "pytorch_ort_parity_passed":True,"finite_metrics":True,"distillation_enabled":True}
for key,value in expected.items():
    if p.get(key)!=value: raise SystemExit(f"invalid canary {key}: {p.get(key)!r} != {value!r}")
PY
  "${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/wandb_replay_preflight.py" verify \
    --manifest "${RULE90_PATH}" --expected-manifest-sha256 "${RULE90_SHA}" \
    --required-manifest-version 1 --expected-source-snapshot-id "${SOURCE_SNAPSHOT_ID}" \
    --expected-entity zihanw22 --expected-project carry-any --expected-run-id "${RUN_ID}" \
    --expected-run-name "${RUN_NAME}" --expected-world-size 8
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT
export TORCH_DIST_BACKEND=gloo TORCH_DIST_TIMEOUT_SEC=3600 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1
export HOLOSOMA_GLOO_GRAD_REDUCE=1 HOLOSOMA_GLOO_BARRIER=1 HOLOSOMA_GLOO_SMALL_COLLECTIVES=1
export HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=0 HOLOSOMA_RANK_VISIBLE_DEVICES=1 HOLOSOMA_RANK_LOCAL_CPU_AFFINITY=1
export HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE=1 HOLOSOMA_CONTIGUOUS_MINIBATCHES=1

export HOLOSOMA_SOURCE_ROOT=${SOURCE_ROOT} HOLOSOMA_SOURCE_SNAPSHOT_ID=${SOURCE_SNAPSHOT_ID}
export HOLOSOMA_SOURCE_MANIFEST_SHA256=${GIT_MANIFEST_SHA256}
export HOLOSOMA_GIT_REMOTE_URL=${REMOTE_URL} HOLOSOMA_GIT_REMOTE_REF=${REMOTE_REF}
export HOLOSOMA_GIT_COMMIT_SHA=${COMMIT_SHA} HOLOSOMA_GIT_TREE_SHA=${TREE_SHA}
export HOLOSOMA_FORMAL_GIT_VERIFICATION_PATH=${VERIFY_ROOT}/node_0.json
export OMOMO_DATA_DIR=${MOTION_DIR} OMOMO_OBJECT_MAP=${OBJECT_SPEC_PATH}
export MOTION_DIR OBJECT_URDF=${OBJECT_SPEC_PATH} CONTACT_EXPORT_ROOT=${CONTACT_ROOT} AS_CONTACT_EXPORT_ROOT=${CONTACT_ROOT}
export AS_SINGLE_SLOT_MOTION_BASE=${MOTION_DIR}/_single_slot_motion_bank
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST=${SINGLE_SLOT_SOURCE_DIGEST}
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST=${SINGLE_SLOT_VIEW_DIGEST}
export HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST=${RANK_SHARD_SOURCE_DIGEST}
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_DIR=${SINGLE_SLOT_DIR}
export HOLOSOMA_EXTERNAL_AS_WORLD_SIZE=${WORLD_SIZE}
export OMOMO_EXPECTED_TOTAL=109 RESUME_FROM_BOX_EXPECTED_TOTAL=109 AS_SUCCESS133_FINAL0P5=0 AS_RANK_LOCAL_SHARDS=1
export SOLID_ALLOWED_OBJECT_CATEGORIES='["box","ball","barrel","bin"]'
export CONTACT_SIDECAR_MODE=full-sidecars REQUIRE_MOTION_GENERATOR_TEACHER_MATCH=0
export HOLOSOMA_OBJECT_SPAWN_MODE=single_slot_multi_urdf HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS=1
export HOLOSOMA_REQUIRE_OBJECT_MESH_ASSETS=1 HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE=mesh
export HOLOSOMA_OBJECT_COLLIDER_TYPE=convex_decomposition HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=0
export HOLOSOMA_REQUIRE_CONTACT_INTERVAL_COVERAGE=1 HOLOSOMA_REQUIRE_CONTACT_TARGET_COVERAGE=1
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=0 HOLOSOMA_PERCEPTION_INCLUDE_ROBOT_MESH=1
export HOLOSOMA_MOTION_METRICS_INTERVAL=16 HOLOSOMA_SKIP_INITIAL_CHECKPOINT=1
export HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES=True

export DISTILL_EXPERIMENT_CONFIG=g1-29dof-wbt-w-object-distill-sparse-root-cmd-teacher-linvel
export EXP=${DISTILL_EXPERIMENT_CONFIG}
export DISTILL_REWARD_CONFIG=g1-29dof-wbt-w-object-generalist-tracking-no-contact
export ENABLE_OFFLINE_CONTACT_GUIDANCE=False
export TEACHER_CHECKPOINT=${TEACHER} TEACHER_CHECKPOINT_EXPECTED_SHA256=${TEACHER_SHA256}
export DEFAULT_AS_TEACHER_CHECKPOINT=${TEACHER}
export EXPORT_ONNX=True
export RESUME_FROM_BOX=0 RESUME_FROM_PREVIOUS=0 WANDB_RESUME_SAME_RUN=0
unset RESUME_TRAINING_CKPT RESUME_CHECKPOINT RESUME_CKPT POLICY_INIT_CKPT POLICY_INIT_CHECKPOINT
export CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE=precomputed_turn_then_forward
export ZERO_ROOT_COMMAND_WHEN_DROP_ACTIVE=True CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION=True CAMERA_PITCH_DEG=0
export STUDENT_ACTOR_INPUTS="['actor_obs_root_contact_aware','actor_obs_drop_button','actor_obs_proprio_with_actions_no_linvel']"
export STUDENT_ACTOR_HIDDEN_DIMS='[512,256,128]' STUDENT_POLICY_TYPE=mlp
export PER_GPU_ENVS=${ENVIRONMENTS_PER_RANK} MIN_PER_GPU_ENVS=${ENVIRONMENTS_PER_RANK} TOTAL_NUM_ENVS=${TOTAL_ENVIRONMENTS}
export TRAINING_SEED=42 NUM_LEARNING_ITERATIONS=${TARGET_ITERATIONS} TARGET_LEARNING_ITERATION=${TARGET_ITERATIONS}
export SAVE_INTERVAL NUM_MINI_BATCHES=4 NUM_LEARNING_EPOCHS=7
export PPO_START_EPOCH=0 DAGGER_END_EPOCH=4000 PPO_START_COEFF=0.01 PPO_TARGET_COEFF=1.0 PPO_SCHEDULE_STEP_EPOCHS=500
export PPO_START_NOISE_STD=0.1 PPO_START_NOISE_STD_UNTIL_COEFF=0.1
export DAGGER_LOSS_COEF=1.0 DAGGER_MATCH_STD=False DAGGER_REPLAY_ENABLED=False
export TEACHER_ACTION_MIX_RATIO=0 DAGGER_IGNORE_ZERO_TEACHER_ACTIONS=True
export FIXED_BC_EVAL_LOG_INTERVAL=100 FIXED_BC_GUARD_ENABLED=False
export START_AT_TIMESTEP_ZERO_PROB=0.2 START_AT_TIMESTEP_ZERO_PROB_END=0.2
export START_AT_TIMESTEP_ZERO_PROB_START_ITER=0 START_AT_TIMESTEP_ZERO_PROB_END_ITER=${CURRICULUM_END_ITER}
export FREEZE_AT_TIMESTEP_ZERO_PROB=0.0 FREEZE_AT_TIMESTEP_ZERO_PROB_END=0.0
export FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=0 FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=${CURRICULUM_END_ITER}
export PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=402653184 PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY=402653184
export PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=134217728 PHYSX_GPU_COLLISION_STACK_SIZE=268435456
export RUN_NAME TRAINING_NAME=${RUN_NAME} TRAINING_PROJECT=carry-any LOGGER_BASE_DIR PRINT_TRAIN_CMD=1
export SCHEDULE_NAME=distill_7hvy40000_ppo0p01_to1p0_depth_ab
export SCHEDULE_NOTES="Fresh 7hvy6x2i/model_40000 frozen label-teacher distillation on exact109; PPO/BC ramps 0.01/0.99 to 1.0/0.0 by iteration 4000; no positive contact reward; depth arm=${ENCODER_ARM} encoder=${ENCODER_TYPE}; all other A/B fields fixed."

if [[ ${MODE} == formal ]]; then
  unset WANDB_DISABLED
  export WANDB_MODE=online WANDB_CONSOLE=off WANDB_ENTITY=zihanw22 WANDB_RESUME_MODE=must HOLOSOMA_REQUIRE_WANDB_RUN=1
else
  export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_CONSOLE=off HOLOSOMA_REQUIRE_WANDB_RUN=0
fi

EXTRA_ARGS=(
  "reward:${DISTILL_REWARD_CONFIG}"
  randomization:g1_29dof_wbt_w_object_with_action_delay
  --algo.config.reset-rollout-at-checkpoint=False
  --algo.config.num-steps-per-env=24
  --randomization.setup-terms.setup-dof-pos-bias.params.dof-pos-bias-range='[-0.01,0.01]'
  --randomization.setup-terms.setup-dof-pos-bias.params.enabled=True
  --randomization.setup-terms.actuator-randomizer-state.params.kp-range='[0.9,1.1]'
  --randomization.setup-terms.actuator-randomizer-state.params.kd-range='[0.9,1.1]'
  --randomization.setup-terms.actuator-randomizer-state.params.enable-pd-gain=True
  --randomization.setup-terms.setup-action-delay-buffers.params.ctrl-delay-step-range='[0,1]'
  --randomization.setup-terms.setup-action-delay-buffers.params.enabled=True
  --randomization.setup-terms.setup-torque-rfi.params.enabled=True
  --randomization.setup-terms.setup-torque-rfi.params.rfi-lim=0.01
  --perception.sensor-offset='[0.0576235,0.01753,0.42987]'
  --perception.camera-mount-quat='[0.0,0.40354529635239006,0.0,0.9149596678498247]'
  --perception.camera-frame-quat='[-0.5,0.5,-0.5,0.5]'
  --perception.encoder-type="${ENCODER_TYPE}"
  --reward.terms.offline-contact-guidance.params.contact-export-root "${CONTACT_ROOT}"
  --reward.terms.offline-contact-guidance.params.contact-region-names='["left_wrist","right_wrist","left_elbow","right_elbow","left_wrist_roll","right_wrist_roll","left_wrist_pitch","right_wrist_pitch","torso"]'
  --reward.terms.offline-contact-guidance.params.wrist-region-names='["left_wrist","right_wrist"]'
  --reward.terms.body-contact-reward-arms.weight=0.0
  --reward.terms.body-contact-reward-palms.weight=0.0
  --reward.terms.body-contact-reward-torso.weight=0.0
  --reward.terms.body-contact-reward-left-wrist-yaw.weight=0.0
  --reward.terms.body-contact-reward-right-wrist-yaw.weight=0.0
)
if [[ ${MODE} == formal ]]; then EXTRA_ARGS+=(--logger.id="${RUN_ID}" --logger.resume=must); fi

if [[ ${PREFLIGHT_ONLY:-0} == 1 ]]; then
  echo "[INFO] worker_preflight_ok mode=${MODE} encoder=${ENCODER_TYPE} teacher=${TEACHER_SHA256} ppo=0.01->1.0 contact_reward=0 export_onnx=true"
  exit 0
fi
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d') ]]; then
  echo "[ERROR] GPU apps appeared after preflight" >&2; exit 2
fi
cd "${SOURCE_ROOT}"
echo "[INFO] launch mode=${MODE} encoder=${ENCODER_TYPE} teacher_sha=${TEACHER_SHA256} ppo_schedule=0.01_to_1.0 no_positive_contact_reward=true export_onnx=true"
exec bash "${SOURCE_ROOT}/distill_as_button.sh" "${TEACHER}" "${EXTRA_ARGS[@]}"
