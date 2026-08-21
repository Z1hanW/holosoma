#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 17 || ! $1 =~ ^(canary|formal)$ || ! $2 =~ ^[0-3]$ ]]; then
  echo "usage: $0 MODE NODE_RANK EXPECTED_IP SOURCE_ROOT PERSIST_ROOT MASTER_ADDR MASTER_PORT RUN_ID RUN_NAME CONTRACT_PATH CONTRACT_SHA RULE90_PATH RULE90_SHA CANARY_PATH CANARY_SHA COMMIT_SHA TREE_SHA" >&2
  exit 2
fi

readonly MODE=$1 NODE_RANK=$2 EXPECTED_IP=$3 SOURCE_ROOT=$4 PERSIST_ROOT=$5
readonly MASTER_ADDR=$6 MASTER_PORT=$7 RUN_ID=$8 RUN_NAME=$9 CONTRACT_PATH=${10}
readonly CONTRACT_SHA=${11} RULE90_PATH=${12} RULE90_SHA=${13} CANARY_PATH=${14}
readonly CANARY_SHA=${15} COMMIT_SHA=${16} TREE_SHA=${17}
readonly REMOTE_URL=https://github.com/Z1hanW/holosoma
readonly REMOTE_REF=main
readonly NPROC=8 NNODES=4 WORLD_SIZE=32 ENVIRONMENTS_PER_RANK=2048
readonly TOTAL_ENVIRONMENTS=$((WORLD_SIZE * ENVIRONMENTS_PER_RANK))
readonly PYTHON_BIN=/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3.11
readonly PYTHON_RUNTIME_ROOT=/data/holosoma_runs/.runtime/python/python-runtime-v2-dd7ca81fa848917c362b3a239893a7a26f4c89d42b4f85cb515d91622f1690bc
readonly PYTHON_RUNTIME=${PYTHON_RUNTIME_ROOT}/site-packages
readonly PYTHON_RUNTIME_SHA256=dd7ca81fa848917c362b3a239893a7a26f4c89d42b4f85cb515d91622f1690bc
readonly NCCL_ROOT=/home/ubuntu/FAR/holosoma_runs/.runtime/nccl/e4a7aee9c3eecf53fac780441d2f03b578ab8db8874b71f8e391bcec7adb2899
readonly NCCL_SHA256=e4a7aee9c3eecf53fac780441d2f03b578ab8db8874b71f8e391bcec7adb2899
readonly MOTION_DIGEST=307e9662d498bd507b9d17ca9abf74a3654f7bf66ac6ab989c6f19c3889bddef
readonly MOTION_DIR=/data/holosoma_inputs/corl79_plus_debug30_decoupled_turn_forward_v1/by-source/${MOTION_DIGEST}
readonly OBJECT_SPEC_PATH=${MOTION_DIR}/_clip_object_urdf_map.json
readonly CONTACT_ROOT=${MOTION_DIR}/contact_export_corl79_success133_plus_debug30_realmesh_model05000
readonly SHARD_DIGEST=13db668f710806bf4bc6b0541f1c99a3e2b36ad3e5179cccd31d3af8f1ab4928
readonly SHARD_MANIFEST_SHA256=19500fe84e4fef7c70cadc574b2581a41f1e85b5bf5e6cede7fa58a57ab8c858
readonly SHARD_ROOT=/data/holosoma_inputs/corl79_plus_debug30_decoupled_turn_forward_v1_rank_shards_ws32/by-source/${SHARD_DIGEST}/ws32
readonly ACTOR_INPUTS="['actor_obs_root_contact_aware','actor_obs_drop_button','actor_obs_proprio_with_actions_no_linvel']"

if [[ ${MODE} == formal ]]; then
  readonly TARGET_ITERATIONS=60000 SAVE_INTERVAL=1000
  [[ ${RUN_ID} != - && ${RUN_NAME} != - && ${CONTRACT_PATH} != - && ${RULE90_PATH} != - && ${CANARY_PATH} != - ]]
else
  readonly TARGET_ITERATIONS=2 SAVE_INTERVAL=2
fi

check_sha() {
  local expected=$1 path=$2 actual
  [[ -f ${path} ]] || { echo "[ERROR] missing integrity input: ${path}" >&2; exit 2; }
  actual=$(sha256sum "${path}" | awk '{print $1}')
  [[ ${actual} == "${expected}" ]] || {
    echo "[ERROR] SHA mismatch: ${path} expected=${expected} actual=${actual}" >&2
    exit 2
  }
}

hostname -I | tr ' ' '\n' | grep -Fxq "${EXPECTED_IP}" || {
  echo "[ERROR] node-rank/IP mismatch: rank=${NODE_RANK} expected=${EXPECTED_IP} actual=$(hostname -I)" >&2
  exit 2
}
readonly VERIFY_ROOT=${PERSIST_ROOT}/${MODE}_git_verification
mkdir -p "${VERIFY_ROOT}"
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/verify_formal_git_checkout.py" \
  --source-root "${SOURCE_ROOT}" --remote-url "${REMOTE_URL}" --remote-ref "${REMOTE_REF}" \
  --commit "${COMMIT_SHA}" --tree "${TREE_SHA}" \
  --output "${VERIFY_ROOT}/node_${NODE_RANK}.json"

check_sha 2de9ee5ca188b70e877c32dd9f0d2975eea99d11aa077bb077cf06ea9ab897bb "${MOTION_DIR}/manifest.json"
check_sha 70b466aad04837a79f6dd0f4491cb345a73c687209981acd3eb7f4a0365d8f5c "${OBJECT_SPEC_PATH}"
check_sha "${SHARD_MANIFEST_SHA256}" "${SHARD_ROOT}/manifest.json"
check_sha "${NCCL_SHA256}" "${NCCL_ROOT}/libnccl.so.2"
if [[ ${MODE} == formal ]]; then
  check_sha "${CONTRACT_SHA}" "${CONTRACT_PATH}"
  check_sha "${RULE90_SHA}" "${RULE90_PATH}"
  check_sha "${CANARY_SHA}" "${CANARY_PATH}"
fi

readonly RUN_ROOT=${PERSIST_ROOT}/${MODE}_t1precontact_w030_ws32_e2048/node_${NODE_RANK}
readonly LOGGER_BASE_DIR=${PERSIST_ROOT}/${MODE}_t1precontact_w030_ws32_e2048/shared_training_logs
readonly SCRATCH_ROOT=/dev/shm/holosoma_${MODE}_t1precontact_w030_ws32_e2048
mkdir -p "${RUN_ROOT}" "${LOGGER_BASE_DIR}" "${SCRATCH_ROOT}/tmp" "${SCRATCH_ROOT}/xdg-cache" \
  "${SCRATCH_ROOT}/robot-usd-cache" "${SCRATCH_ROOT}/object-usd-cache" \
  "${SCRATCH_ROOT}/perception-mesh-cache" "${SCRATCH_ROOT}/derived-data-cache" \
  "${SCRATCH_ROOT}/provenance-cache"
unset BASH_ENV ENV CDPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONOPTIMIZE
unset PYTHONWARNINGS PYTHONBREAKPOINT PYTHONSAFEPATH LD_PRELOAD
export HOME=/home/ubuntu
export PATH="$(dirname "${PYTHON_BIN}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="${PYTHON_RUNTIME}:${SOURCE_ROOT}/src/holosoma:${SOURCE_ROOT}/src/holosoma_inference:${SOURCE_ROOT}/src"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export PYTHON_RUNTIME_SITEPACKAGES="${PYTHON_RUNTIME}"
export PYTHON_RUNTIME_MANIFEST_SHA256="${PYTHON_RUNTIME_SHA256}"
export HOLOSOMA_PYTHON_RUNTIME_MANIFEST_SHA256="${PYTHON_RUNTIME_SHA256}"
export HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y HEADLESS=1 OMP_NUM_THREADS=1
export TMPDIR="${SCRATCH_ROOT}/tmp" XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg-cache"
export HOLOSOMA_RUNTIME_SCRATCH_ROOT="${SCRATCH_ROOT}"
export HOLOSOMA_ROBOT_USD_CACHE_DIR="${SCRATCH_ROOT}/robot-usd-cache"
export HOLOSOMA_OBJECT_USD_CACHE_DIR="${SCRATCH_ROOT}/object-usd-cache"
export HOLOSOMA_PERCEPTION_MESH_CACHE_DIR="${SCRATCH_ROOT}/perception-mesh-cache"
export HOLOSOMA_ISAACSIM_KIT_ARGS="--/UJITSO/datastore/localCachePath=${SCRATCH_ROOT}/derived-data-cache --/UJITSO/datastore/localDataStore/largeChunkDiskBudgetMB=1024"
export HOLOSOMA_DATA_PROVENANCE_CACHE_ROOT="${SCRATCH_ROOT}/provenance-cache"

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/verify_python_runtime_overlay.py" \
  --site-packages "${PYTHON_RUNTIME}" --manifest-sha256 "${PYTHON_RUNTIME_SHA256}" \
  --require-distribution-closure --require-current-runtime-binding

"${PYTHON_BIN}" - "${SOURCE_ROOT}/scripts/prepare_as_rank_shards.py" "${MOTION_DIR}" \
  "${OBJECT_SPEC_PATH}" "${SHARD_ROOT}" <<'PY'
import importlib.util, sys
from pathlib import Path
module_path, motion_dir, object_map, output_root = sys.argv[1:]
spec = importlib.util.spec_from_file_location("formal_rank_shards", module_path)
if spec is None or spec.loader is None: raise RuntimeError(module_path)
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
manifest = module.validate_published_rank_shards(
    motion_dir=Path(motion_dir), object_map=Path(object_map), output_root=Path(output_root),
    world_size=32, environments_per_rank=2048,
    expected_source_digest="13db668f710806bf4bc6b0541f1c99a3e2b36ad3e5179cccd31d3af8f1ab4928",
)
if manifest["clip_count"] != 109 or not manifest["exact_clip_partition"]:
    raise RuntimeError("Unexpected exact109 partition")
if set(manifest["clip_cover_counts"].values()) != {1}:
    raise RuntimeError("Every clip must appear exactly once globally")
if not manifest["rank_clip_counts_divide_environments_per_rank"]:
    raise RuntimeError("Rank clip count must divide 2048")
PY

mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,name,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits)
[[ ${#gpu_rows[@]} -eq 8 ]] || { echo "[ERROR] expected 8 GPUs" >&2; exit 2; }
for row in "${gpu_rows[@]}"; do
  [[ ${row} == *"NVIDIA L40S"* && ${row##*, } == 0 ]] || { echo "[ERROR] GPU health: ${row}" >&2; exit 2; }
done
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d') ]]; then
  echo "[ERROR] node is not GPU-idle" >&2; exit 2
fi

if [[ ${MODE} == formal ]]; then
  "${PYTHON_BIN}" - "${CANARY_PATH}" "${COMMIT_SHA}" "${TREE_SHA}" <<'PY'
import json,sys
from pathlib import Path
payload=json.loads(Path(sys.argv[1]).read_text())
expected={"accepted":True,"world_size":32,"environments_per_rank":2048,
          "completed_iterations_per_rank":2,"commit_sha":sys.argv[2],"tree_sha":sys.argv[3],
          "export_onnx":True,"onnx_checker_passed":True,"onnxruntime_load_passed":True,
          "pytorch_ort_parity_passed":True,"finite_metrics":True,"pure_ppo":True,
          "t1_precontact_total_weight":0.3}
for key,value in expected.items():
    if payload.get(key)!=value: raise SystemExit(f"invalid canary {key}: {payload.get(key)!r}")
PY
  if [[ ${NODE_RANK} == 0 ]]; then
    "${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/wandb_replay_preflight.py" verify \
      --manifest "${RULE90_PATH}" --expected-manifest-sha256 "${RULE90_SHA}" \
      --required-manifest-version 1 --expected-source-snapshot-id "git-${COMMIT_SHA}" \
      --expected-entity zihanw22 --expected-project carry-any --expected-run-id "${RUN_ID}" \
      --expected-run-name "${RUN_NAME}" --expected-world-size 32
  fi
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC NNODES MASTER_ADDR MASTER_PORT
export TORCH_DIST_BACKEND=gloo TORCH_DIST_TIMEOUT_SEC=3600
export NCCL_LIB_DIR="${NCCL_ROOT}" NCCL_LIB_SHA256="${NCCL_SHA256}"
export LD_LIBRARY_PATH="${NCCL_ROOT}" LD_PRELOAD="${NCCL_ROOT}/libnccl.so.2"
export GLOO_SOCKET_IFNAME=enp135s0 NCCL_SOCKET_IFNAME=enp135s0 NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN NCCL_SOCKET_FAMILY=AF_INET NCCL_SOCKET_RETRY_CNT=34 NCCL_SOCKET_RETRY_SLEEP_MSEC=100
export NCCL_SOCKET_NTHREADS=2 NCCL_NSOCKS_PERTHREAD=4 HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC=300
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300
export TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_TRACE_BUFFER_SIZE=65536 TORCH_NCCL_PROPAGATE_ERROR=1
export TORCH_NCCL_DESYNC_DEBUG=0 TORCH_NCCL_ENABLE_TIMING=0 TORCH_NCCL_BLOCKING_WAIT=0
export HOLOSOMA_GLOO_GRAD_REDUCE=0 HOLOSOMA_GLOO_BARRIER=1 HOLOSOMA_GLOO_SMALL_COLLECTIVES=1
export HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1 HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=0
export HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER=1 HOLOSOMA_RANK_VISIBLE_DEVICES=1
export HOLOSOMA_RANK_LOCAL_CPU_AFFINITY=1 HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE=1
export HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE=0 HOLOSOMA_SYNC_AFTER_OPTIMIZER_STEP=0
export HOLOSOMA_SYNC_AFTER_MICROBATCH_FORWARD=0 HOLOSOMA_CONTIGUOUS_MINIBATCHES=1

readonly GIT_MANIFEST_SHA256=$(git -C "${SOURCE_ROOT}" ls-tree -r --full-tree "${COMMIT_SHA}" | sha256sum | awk '{print $1}')
export HOLOSOMA_SOURCE_ROOT="${SOURCE_ROOT}"
export HOLOSOMA_SOURCE_SNAPSHOT_ID="src-${GIT_MANIFEST_SHA256}"
export HOLOSOMA_SOURCE_MANIFEST_SHA256="${GIT_MANIFEST_SHA256}"
export HOLOSOMA_GIT_REMOTE_URL="${REMOTE_URL}" HOLOSOMA_GIT_REMOTE_REF="${REMOTE_REF}"
export HOLOSOMA_GIT_COMMIT_SHA="${COMMIT_SHA}" HOLOSOMA_GIT_TREE_SHA="${TREE_SHA}"
export MOTION_DIR OBJECT_SPEC_PATH OBJECT_URDF="${OBJECT_SPEC_PATH}" CONTACT_EXPORT_ROOT="${CONTACT_ROOT}"
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST=0d1ae14db44e06fd9806e8757d3a26051697ee2f60ce446deed5b25ac9bfe6c5
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST="${MOTION_DIGEST}" HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_DIR="${MOTION_DIR}"
export HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST="${SHARD_DIGEST}" HOLOSOMA_EXTERNAL_AS_WORLD_SIZE=32
export HOLOSOMA_RANK_LOCAL_MOTION_ROOT="${SHARD_ROOT}" HOLOSOMA_MOTION_SHARD_MANIFEST="${SHARD_ROOT}/manifest.json"
export HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED=1 HOLOSOMA_REQUIRE_RANK_LOCAL_SHARD_PROVENANCE=1
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=0 HOLOSOMA_OBJECT_SPAWN_MODE=single_slot_multi_urdf
export HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS=1 HOLOSOMA_REQUIRE_OBJECT_MESH_ASSETS=1
export HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE=mesh HOLOSOMA_OBJECT_COLLIDER_TYPE=convex_decomposition
export HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=0 HOLOSOMA_REQUIRE_CONTACT_INTERVAL_COVERAGE=1
export HOLOSOMA_REQUIRE_CONTACT_TARGET_COVERAGE=1 HOLOSOMA_MOTION_METRICS_INTERVAL=16
export HOLOSOMA_DISABLE_AUTO_RESET=0 HOLOSOMA_DISABLE_CLIP_END_RESET=0 HOLOSOMA_DISABLE_MOTION_END_RESET=0
export HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES=True HOLOSOMA_PERCEPTION_INCLUDE_ROBOT_MESH=1
export HOLOSOMA_SKIP_INITIAL_CHECKPOINT=1
if [[ ${MODE} == formal ]]; then
  unset WANDB_DISABLED
  export WANDB_MODE=online WANDB_CONSOLE=off WANDB_ENTITY=zihanw22 HOLOSOMA_REQUIRE_WANDB_RUN=1 WANDB_RESUME=must
else
  export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_CONSOLE=off HOLOSOMA_REQUIRE_WANDB_RUN=0
fi

TRAIN_ARGS=(
  exp:g1-29dof-wbt-w-object-pure-rl-policy-command-after-lift
  command:g1-29dof-wbt-w-object-generalist
  reward:g1-29dof-wbt-w-object-generalist-tracking-no-contact
  perception:camera_depth_d435i
  termination:g1-29dof-wbt-generalist-z-only
  randomization:g1_29dof_wbt_w_object_pure_rl
  logger:wandb
  --training.project=carry-any --training.name="${RUN_NAME}" --training.num-envs="${TOTAL_ENVIRONMENTS}"
  --training.seed=42 --training.multigpu=True --training.export-onnx=True
  --algo.config.distill.enabled=False --algo.config.num-learning-iterations="${TARGET_ITERATIONS}"
  --algo.config.num-steps-per-env=24 --algo.config.num-learning-epochs=7 --algo.config.num-mini-batches=4
  --algo.config.clip-param=0.2 --algo.config.gamma=0.99 --algo.config.lam=0.95 --algo.config.value-loss-coef=1.0
  --algo.config.entropy-coef=0.005 --algo.config.max-grad-norm=1.0 --algo.config.schedule=adaptive
  --algo.config.desired-kl=0.01 --algo.config.actor-learning-rate=0.001 --algo.config.critic-learning-rate=0.001
  --algo.config.min-actor-learning-rate=0.00001 --algo.config.max-actor-learning-rate=0.01
  --algo.config.min-critic-learning-rate=0.00001 --algo.config.max-critic-learning-rate=0.01
  --algo.config.init-noise-std=1.0 --algo.config.module-dict.actor.min-noise-std=0.01
  --algo.config.normalize-actor-obs=False --algo.config.normalize-critic-obs=False
  --algo.config.save-interval="${SAVE_INTERVAL}" --algo.config.reset-rollout-at-checkpoint=False
  --algo.config.module-dict.actor.type=MLP --algo.config.module-dict.actor.input-dim="${ACTOR_INPUTS}"
  --algo.config.module-dict.actor.layer-config.hidden-dims='[512,256,128]'
  --algo.config.module-dict.critic.layer-config.hidden-dims='[512,256,128]'
  --observation.groups.actor_obs_root_contact_aware.history-length=1
  --observation.groups.actor_obs_drop_button.history-length=1
  --observation.groups.actor_obs_proprio_with_actions_no_linvel.history-length=1
  --observation.groups.critic_proprio_history.history-length=1
  --command.setup-terms.motion-command.params.motion-config.motion-file="${MOTION_DIR}"
  --command.setup-terms.motion-command.params.motion-config.pure-rl-policy-command-after-lift-enabled=False
  --command.setup-terms.motion-command.params.motion-config.hybrid-stage2-enabled=False
  --command.setup-terms.motion-command.params.motion-config.hybrid-velocity-enabled=False
  --command.setup-terms.motion-command.params.motion-config.contact-aware-sparse-root-command-mode=precomputed_turn_then_forward
  --command.setup-terms.motion-command.params.motion-config.zero-root-command-when-drop-active=True
  --command.setup-terms.motion-command.params.motion-config.clip-weighting-strategy=uniform_clip
  --command.setup-terms.motion-command.params.motion-config.pair-terrain-with-motion=False
  --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler=True
  --command.setup-terms.motion-command.params.motion-config.adaptive-sampling-contact-interval-root="${CONTACT_ROOT}/clips"
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=0.2
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end=0.2
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-start-iter=1
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end-iter=1
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob=0.0
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end=0.0
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-start-iter=1
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end-iter=1
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
  --termination.terms.bad-tracking.params.bad-ref-pos-threshold=1.0
  --termination.terms.bad-tracking.params.bad-ref-ori-threshold=1.1
  --termination.terms.bad-tracking.params.bad-motion-body-pos-threshold=0.5
  --termination.terms.bad-tracking.params.bad-object-pos-threshold=0.5
  --termination.terms.bad-tracking.params.bad-object-ori-threshold=1.1
  --reward.terms.motion-joint-position-error-lower.weight=0.0
  --reward.terms.motion-joint-position-error-waist.weight=0.0
  --reward.terms.t1-precontact-motion-joint-position-lower.weight=0.15
  --reward.terms.t1-precontact-motion-joint-position-lower.params.lead-steps=50
  --reward.terms.t1-precontact-motion-joint-position-lower.params.tail-steps=10
  --reward.terms.t1-precontact-motion-joint-position-lower.params.ramp-steps=5
  --reward.terms.t1-precontact-motion-joint-position-lower.params.require-complete-contact-window=True
  --reward.terms.t1-precontact-motion-joint-position-waist.weight=0.15
  --reward.terms.t1-precontact-motion-joint-position-waist.params.lead-steps=50
  --reward.terms.t1-precontact-motion-joint-position-waist.params.tail-steps=10
  --reward.terms.t1-precontact-motion-joint-position-waist.params.ramp-steps=5
  --reward.terms.t1-precontact-motion-joint-position-waist.params.require-complete-contact-window=True
  --reward.terms.offline-contact-guidance.params.contact-export-root="${CONTACT_ROOT}"
  --reward.terms.offline-contact-guidance.params.contact-region-names='["left_wrist","right_wrist","left_elbow","right_elbow","left_wrist_roll","right_wrist_roll","left_wrist_pitch","right_wrist_pitch","torso"]'
  --reward.terms.offline-contact-guidance.params.wrist-region-names='["left_wrist","right_wrist"]'
  --reward.terms.offline-contact-guidance.weight=0.0
  --perception.camera-apply-sensor-noise=True --perception.camera-warp-edge-noise=True
  --perception.camera-warp-enable-holes=True --perception.camera-warp-hole-prob=0.2
  --perception.camera-warp-additive-noise-std=0.03 --perception.camera-warp-depth-offset-std=0.03
  --perception.object-geometry-mode=mesh --robot.object.enabled=True --robot.object.object-urdf-path="${OBJECT_SPEC_PATH}"
  --simulator.config.sim.max-episode-length-s=8.0
  --simulator.config.sim.physx.gpu-found-lost-pairs-capacity=335544320
  --simulator.config.sim.physx.gpu-found-lost-aggregate-pairs-capacity=469762048
  --simulator.config.sim.physx.gpu-total-aggregate-pairs-capacity=83886080
  --simulator.config.sim.physx.gpu-collision-stack-size=268435456
  --logger.mode="$( [[ ${MODE} == formal ]] && echo online || echo offline )"
  --logger.video.enabled=False --logger.headless-recording=False --logger.video.upload-to-wandb=False
  --logger.base-dir="${LOGGER_BASE_DIR}"
)
if [[ ${MODE} == formal ]]; then TRAIN_ARGS+=(--logger.id="${RUN_ID}" --logger.resume=must); fi

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/validate_train_cli.py" --expected-motion-end-mode episodic -- "${TRAIN_ARGS[@]}"
export HOLOSOMA_TRAINING_PROVENANCE
HOLOSOMA_TRAINING_PROVENANCE=$("${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/compute_training_provenance.py" \
  --training-regime pure_rl --motion-dir "${MOTION_DIR}" --object-map "${OBJECT_SPEC_PATH}" \
  --contact-root "${CONTACT_ROOT}" --motion-shard-manifest "${HOLOSOMA_MOTION_SHARD_MANIFEST}" \
  --contact-interval-runtime-prepend-compensation true --source-root "${SOURCE_ROOT}")
HOLOSOMA_TRAINING_PROVENANCE=$("${PYTHON_BIN}" - "${HOLOSOMA_TRAINING_PROVENANCE}" \
  "${VERIFY_ROOT}/node_${NODE_RANK}.json" <<'PY'
import json, sys

provenance = json.loads(sys.argv[1])
git_verification = json.loads(open(sys.argv[2], encoding="utf-8").read())
provenance.update(
    {
        "source_distribution": "direct_remote_git_clean_checkout",
        "git_remote_url": git_verification["remote_url"],
        "git_remote_ref": git_verification["remote_ref"],
        "git_commit_sha": git_verification["commit_sha"],
        "git_tree_sha": git_verification["tree_sha"],
        "git_fetched_ref_commit": git_verification["fetched_ref_commit"],
        "git_declared_submodules": git_verification["declared_submodules"],
        "git_checkout_tracked_diff_clean": git_verification["tracked_diff_clean"],
        "git_checkout_untracked_clean": git_verification["untracked_clean"],
        "git_legacy_unmapped_gitlinks_inactive_and_empty": git_verification[
            "legacy_unmapped_gitlinks_inactive_and_empty"
        ],
    }
)
print(json.dumps(provenance, sort_keys=True, separators=(",", ":")))
PY
)

if [[ ${PREFLIGHT_ONLY:-0} == 1 ]]; then
  echo "[INFO] worker_preflight_ok mode=${MODE} rank=${NODE_RANK} commit=${COMMIT_SHA} tree=${TREE_SHA} t1_window=-50:+10 ramps=5+5 lower_weight=0.15 waist_weight=0.15 total_weight=0.30 export_onnx=true"
  exit 0
fi
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d') ]]; then
  echo "[ERROR] GPU apps appeared after preflight" >&2; exit 2
fi
cd "${SOURCE_ROOT}"
exec "${PYTHON_BIN}" -m torch.distributed.run --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" --nproc_per_node="${NPROC}" --max_restarts=0 --master_port="${MASTER_PORT}" \
  src/holosoma/holosoma/train_agent_rank_visible.py "${TRAIN_ARGS[@]}"
