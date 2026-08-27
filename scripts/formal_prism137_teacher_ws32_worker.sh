#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 20 || ! $1 =~ ^(canary|formal)$ || ! $2 =~ ^(mlp|lstm)$ || ! $3 =~ ^[0-3]$ ]]; then
  echo "usage: $0 MODE POLICY_ARCH NODE_RANK EXPECTED_IP SOURCE_ROOT PERSIST_ROOT MASTER_ADDR MASTER_PORT RUN_ID RUN_NAME CONTRACT_PATH CONTRACT_SHA RULE90_PATH RULE90_SHA CANARY_PATH CANARY_SHA COMMIT_SHA TREE_SHA SHARD_DIGEST SHARD_MANIFEST_SHA" >&2
  exit 2
fi

readonly MODE=$1 POLICY_ARCH=$2 NODE_RANK=$3 EXPECTED_IP=$4 SOURCE_ROOT=$5 PERSIST_ROOT=$6
readonly MASTER_ADDR=$7 MASTER_PORT=$8 RUN_ID=$9 RUN_NAME=${10} CONTRACT_PATH=${11}
readonly CONTRACT_SHA=${12} RULE90_PATH=${13} RULE90_SHA=${14} CANARY_PATH=${15}
readonly CANARY_SHA=${16} COMMIT_SHA=${17} TREE_SHA=${18} SHARD_DIGEST=${19}
readonly SHARD_MANIFEST_SHA=${20}
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
readonly DATASET_ROOT=/data/zzzihanw/holosoma_training_data/prism_world_zhen137_humanscale_w50_smooth5_20260826
readonly SOURCE_MOTION_DIR=${DATASET_ROOT}/motion_bank
readonly SOURCE_OBJECT_SPEC_PATH=${SOURCE_MOTION_DIR}/_clip_object_urdf_map.json
readonly SINGLE_SLOT_SOURCE_DIGEST=1dde9c006356b5c8a6bd3ae86fb854f5c2709a6767e8be73d127417c9f819099
readonly SINGLE_SLOT_VIEW_DIGEST=3153f199be72be2905dde69f9d920f809c9c77afac37edfa0b611fbfc6ad778b
readonly MOTION_DIR=/data/holosoma_inputs/prism_world_zhen137_humanscale_w50_smooth5_20260826_single_slot/by-source/${SINGLE_SLOT_VIEW_DIGEST}
readonly OBJECT_SPEC_PATH=${MOTION_DIR}/_clip_object_urdf_map.json
readonly SHARD_ROOT=${MOTION_DIR}/_rank_shards/by-source/${SHARD_DIGEST}/ws32

[[ ${COMMIT_SHA} =~ ^[0-9a-f]{40}$ && ${TREE_SHA} =~ ^[0-9a-f]{40}$ ]] || {
  echo "[ERROR] commit and tree must be full lowercase Git SHA-1 values" >&2
  exit 2
}
[[ ${SHARD_DIGEST} =~ ^[0-9a-f]{64}$ && ${SHARD_MANIFEST_SHA} =~ ^[0-9a-f]{64}$ ]] || {
  echo "[ERROR] shard digest and manifest SHA must be lowercase SHA-256 values" >&2
  exit 2
}

if [[ ${MODE} == formal ]]; then
  readonly TARGET_ITERATIONS=40000 SAVE_INTERVAL=1000 CURRICULUM_END_ITER=39999
  for value in "${RUN_ID}" "${RUN_NAME}" "${CONTRACT_PATH}" "${CONTRACT_SHA}" \
    "${RULE90_PATH}" "${RULE90_SHA}" "${CANARY_PATH}" "${CANARY_SHA}"; do
    [[ -n ${value} && ${value} != - ]] || {
      echo "[ERROR] formal mode requires immutable W&B, contract, Rule-90, and canary inputs" >&2
      exit 2
    }
  done
else
  readonly TARGET_ITERATIONS=2 SAVE_INTERVAL=2 CURRICULUM_END_ITER=1
  readonly RUN_ID_EFFECTIVE=canary_prism137_teacher_${POLICY_ARCH}_ws32
  readonly RUN_NAME_EFFECTIVE=canary_prism137_teacher_${POLICY_ARCH}_ws32_e2048
fi

if [[ ${MODE} == formal ]]; then
  readonly RUN_ID_EFFECTIVE=${RUN_ID}
  readonly RUN_NAME_EFFECTIVE=${RUN_NAME}
fi

check_sha() {
  local expected=$1 path=$2 actual
  [[ -f ${path} && ! -L ${path} ]] || {
    echo "[ERROR] missing regular immutable input: ${path}" >&2
    exit 2
  }
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

readonly RUN_ROOT=${PERSIST_ROOT}/${MODE}_${POLICY_ARCH}_${RUN_ID_EFFECTIVE}/node_${NODE_RANK}
readonly LOGGER_BASE_DIR=${PERSIST_ROOT}/${MODE}_${POLICY_ARCH}_${RUN_ID_EFFECTIVE}/shared_training_logs
readonly VERIFY_ROOT=${PERSIST_ROOT}/${MODE}_${POLICY_ARCH}_${RUN_ID_EFFECTIVE}/git_verification
readonly SCRATCH_ROOT=/dev/shm/holosoma_prism137_teacher_${MODE}_${POLICY_ARCH}_${RUN_ID_EFFECTIVE}
mkdir -p "${RUN_ROOT}" "${LOGGER_BASE_DIR}" "${VERIFY_ROOT}" "${PERSIST_ROOT}/wandb" \
  "${SCRATCH_ROOT}/tmp" "${SCRATCH_ROOT}/xdg-cache" \
  "${SCRATCH_ROOT}/robot-usd-cache" "${SCRATCH_ROOT}/object-usd-cache" \
  "${SCRATCH_ROOT}/perception-mesh-cache" "${SCRATCH_ROOT}/derived-data-cache" \
  "${SCRATCH_ROOT}/provenance-cache"

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/verify_formal_git_checkout.py" \
  --source-root "${SOURCE_ROOT}" --remote-url "${REMOTE_URL}" --remote-ref "${REMOTE_REF}" \
  --commit "${COMMIT_SHA}" --tree "${TREE_SHA}" \
  --output "${VERIFY_ROOT}/node_${NODE_RANK}.json"
readonly GIT_MANIFEST_SHA256=$(git -C "${SOURCE_ROOT}" ls-tree -r --full-tree "${COMMIT_SHA}" | sha256sum | awk '{print $1}')
readonly SOURCE_SNAPSHOT_ID=src-${GIT_MANIFEST_SHA256}

check_sha 4c4f037aee4bf41883bf1c1b65e782ec555308b181f0d0cc188d1048ebefa42d "${DATASET_ROOT}/manifests/dataset_audit.json"
check_sha 7271c7d7e49d08b8c16d7010969d289d00bf1bbd4c7d70f3460e39248e113c6b "${DATASET_ROOT}/manifests/source_manifest.json"
check_sha ef3bb3586690cee36e12d6ccd90bc382540993ab6ba0addd0c05a96e1cf3b95a "${SOURCE_OBJECT_SPEC_PATH}"
check_sha c48b2ecd8fb09b409158fd9696fd3690e17a708406efced7f30db3a4b08c4131 "${SOURCE_MOTION_DIR}/_mesh_physics_manifest.json"
check_sha 190288351fa3a92b3608c0d0b1647fda3daee51969f526b18468bcbe9016183f "${MOTION_DIR}/manifest.json"
check_sha 867522fd61c63e6fcf37e0a041792f438e821f34ff482e26b07b47de6bfb7b59 "${OBJECT_SPEC_PATH}"
check_sha "${SHARD_MANIFEST_SHA}" "${SHARD_ROOT}/manifest.json"
check_sha "${NCCL_SHA256}" "${NCCL_ROOT}/libnccl.so.2"
[[ $(find "${SOURCE_MOTION_DIR}" -maxdepth 1 -type l -name '*.npz' | wc -l) -eq 137 ]] || {
  echo "[ERROR] source motion bank must contain exactly 137 clip links" >&2
  exit 2
}
[[ $(find "${MOTION_DIR}" -maxdepth 1 -type f ! -type l -name '*.npz' | wc -l) -eq 137 ]] || {
  echo "[ERROR] single-slot view must contain exactly 137 regular clips" >&2
  exit 2
}

if [[ ${MODE} == formal ]]; then
  check_sha "${CONTRACT_SHA}" "${CONTRACT_PATH}"
  check_sha "${RULE90_SHA}" "${RULE90_PATH}"
  check_sha "${CANARY_SHA}" "${CANARY_PATH}"
  "${PYTHON_BIN}" - "${CANARY_PATH}" "${COMMIT_SHA}" "${TREE_SHA}" \
    "${SOURCE_SNAPSHOT_ID}" "${POLICY_ARCH}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
expected = {
    "accepted": True,
    "world_size": 32,
    "environments_per_rank": 2048,
    "completed_iterations_per_rank": 2,
    "commit_sha": sys.argv[2],
    "tree_sha": sys.argv[3],
    "source_snapshot_id": sys.argv[4],
    "policy_arch": sys.argv[5],
    "dataset_clip_count": 137,
    "actor_observation_dim": 178,
    "critic_observation_dim": 310,
    "pure_ppo": True,
    "distillation_enabled": False,
    "export_onnx": True,
    "onnx_checker_passed": True,
    "onnxruntime_load_passed": True,
    "pytorch_ort_parity_passed": True,
    "finite_metrics": True,
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f"invalid canary {key}: {payload.get(key)!r} != {value!r}")
if sys.argv[5] == "lstm":
    recurrent = payload.get("recurrent_policy_contract")
    expected_recurrent = {"kind": "lstm", "num_layers": 1, "hidden_dim": 256}
    if not isinstance(recurrent, dict):
        raise SystemExit("LSTM canary is missing recurrent_policy_contract")
    for key, value in expected_recurrent.items():
        if recurrent.get(key) != value:
            raise SystemExit(f"invalid recurrent canary {key}: {recurrent.get(key)!r}")
elif payload.get("recurrent_policy_contract") is not None:
    raise SystemExit("MLP canary unexpectedly declares recurrent policy state")
PY
fi

unset BASH_ENV ENV CDPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONOPTIMIZE
unset PYTHONWARNINGS PYTHONBREAKPOINT PYTHONSAFEPATH LD_LIBRARY_PATH LD_PRELOAD
unset WANDB_RUN_ID WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS WANDB_DISABLED
unset WANDB_SKIP_UPLOAD SKIP_WANDB_UPLOAD HOLOSOMA_STEP_TIMING_PROFILE
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

"${PYTHON_BIN}" - "${SOURCE_ROOT}/scripts/prepare_as_rank_shards.py" "${MOTION_DIR}" \
  "${OBJECT_SPEC_PATH}" "${SHARD_ROOT}" "${SHARD_DIGEST}" <<'PY'
import importlib.util
import sys
from pathlib import Path

module_path, motion_dir, object_map, output_root, digest = sys.argv[1:]
spec = importlib.util.spec_from_file_location("formal_rank_shards", module_path)
if spec is None or spec.loader is None:
    raise RuntimeError(module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
manifest = module.validate_published_rank_shards(
    motion_dir=Path(motion_dir),
    object_map=Path(object_map),
    output_root=Path(output_root),
    world_size=32,
    environments_per_rank=2048,
    expected_source_digest=digest,
)
if manifest["clip_count"] != 137 or not manifest["exact_clip_partition"]:
    raise RuntimeError("Unexpected prism137 partition")
if manifest["duplicated_to_fill_empty_ranks"]:
    raise RuntimeError("Rank sharding duplicated source clips")
if set(manifest["clip_cover_counts"].values()) != {1}:
    raise RuntimeError("Every source clip must appear exactly once globally")
if not manifest["rank_clip_counts_divide_environments_per_rank"]:
    raise RuntimeError("Every rank-local clip count must divide 2048")
if sum(shard["clip_count"] for shard in manifest["shards"]) != 137:
    raise RuntimeError("Global rank-local clip count does not reconstruct prism137")
PY

mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,name,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits)
[[ ${#gpu_rows[@]} -eq 8 ]] || { echo "[ERROR] expected exactly 8 GPUs" >&2; exit 2; }
for row in "${gpu_rows[@]}"; do
  [[ ${row} == *"NVIDIA L40S"* && ${row##*, } == 0 ]] || {
    echo "[ERROR] GPU health: ${row}" >&2
    exit 2
  }
done
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d') ]]; then
  echo "[ERROR] node is not GPU-idle" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC NNODES NODE_RANK MASTER_ADDR MASTER_PORT
export TORCH_DIST_BACKEND=gloo TORCH_DIST_TIMEOUT_SEC=3600
export GLOO_SOCKET_IFNAME=enp135s0 NCCL_SOCKET_IFNAME=enp135s0 NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_RETRY_CNT=34 NCCL_SOCKET_RETRY_SLEEP_MSEC=100
export NCCL_SOCKET_NTHREADS=2 NCCL_NSOCKS_PERTHREAD=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300 TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=65536 TORCH_NCCL_PROPAGATE_ERROR=1
export TORCH_NCCL_DESYNC_DEBUG=0 TORCH_NCCL_ENABLE_TIMING=0 TORCH_NCCL_BLOCKING_WAIT=0
export HOLOSOMA_GLOO_GRAD_REDUCE=0 HOLOSOMA_GLOO_BARRIER=1 HOLOSOMA_GLOO_SMALL_COLLECTIVES=1
export HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1 HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=0
export HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER=1 HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC=300
export HOLOSOMA_RANK_VISIBLE_DEVICES=1 HOLOSOMA_RANK_LOCAL_CPU_AFFINITY=1
export HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE=1 HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE=0
export HOLOSOMA_SYNC_AFTER_OPTIMIZER_STEP=0 HOLOSOMA_SYNC_AFTER_MICROBATCH_FORWARD=0
export HOLOSOMA_CONTIGUOUS_MINIBATCHES=1
if [[ ${MODE} == canary ]]; then
  export HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY=1
else
  export HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY=0
fi

export HOLOSOMA_SOURCE_ROOT=${SOURCE_ROOT} HOLOSOMA_SOURCE_SNAPSHOT_ID=${SOURCE_SNAPSHOT_ID}
export HOLOSOMA_SOURCE_MANIFEST_SHA256=${GIT_MANIFEST_SHA256}
export HOLOSOMA_GIT_REMOTE_URL=${REMOTE_URL} HOLOSOMA_GIT_REMOTE_REF=${REMOTE_REF}
export HOLOSOMA_GIT_COMMIT_SHA=${COMMIT_SHA} HOLOSOMA_GIT_TREE_SHA=${TREE_SHA}
export HOLOSOMA_FORMAL_GIT_VERIFICATION_PATH=${VERIFY_ROOT}/node_${NODE_RANK}.json
export MOTION_DIR OBJECT_SPEC_PATH OBJECT_URDF=${OBJECT_SPEC_PATH}
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST=${SINGLE_SLOT_SOURCE_DIGEST}
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST=${SINGLE_SLOT_VIEW_DIGEST}
export HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST=${SHARD_DIGEST}
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_DIR=${MOTION_DIR} HOLOSOMA_EXTERNAL_AS_WORLD_SIZE=32
export HOLOSOMA_RANK_LOCAL_MOTION_ROOT=${SHARD_ROOT}
export HOLOSOMA_MOTION_SHARD_MANIFEST=${SHARD_ROOT}/manifest.json
export HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED=1 HOLOSOMA_REQUIRE_RANK_LOCAL_SHARD_PROVENANCE=1
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=0
export HOLOSOMA_OBJECT_SPAWN_MODE=single_slot_multi_urdf HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS=1
export HOLOSOMA_REQUIRE_OBJECT_MESH_ASSETS=1 HOLOSOMA_ALLOW_LEGACY_OBJECT_URDF_FALLBACK=0
export HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE=mesh HOLOSOMA_OBJECT_COLLIDER_TYPE=convex_decomposition
export HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=0
export HOLOSOMA_REQUIRE_CONTACT_INTERVAL_COVERAGE=0 HOLOSOMA_REQUIRE_CONTACT_TARGET_COVERAGE=0
export HOLOSOMA_MOTION_METRICS_INTERVAL=16
export HOLOSOMA_DISABLE_AUTO_RESET=0 HOLOSOMA_DISABLE_CLIP_END_RESET=0 HOLOSOMA_DISABLE_MOTION_END_RESET=0
export HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES=False HOLOSOMA_PERCEPTION_INCLUDE_ROBOT_MESH=1
export HOLOSOMA_SKIP_INITIAL_CHECKPOINT=1

if [[ ${MODE} == formal ]]; then
  unset WANDB_DISABLED
  export WANDB_MODE=online WANDB_CONSOLE=off WANDB_ENTITY=zihanw22
  export HOLOSOMA_REQUIRE_WANDB_RUN=1 WANDB_RESUME=must
else
  export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_CONSOLE=off
  export HOLOSOMA_REQUIRE_WANDB_RUN=0
fi

POLICY_ARGS=(
  --algo.config.module-dict.actor.input-dim="['actor_obs']"
  --algo.config.module-dict.actor.layer-config.hidden-dims='[512,256,128]'
  --algo.config.module-dict.critic.input-dim="['critic_obs']"
  --algo.config.module-dict.critic.layer-config.hidden-dims='[512,256,128]'
)
if [[ ${POLICY_ARCH} == lstm ]]; then
  POLICY_ARGS+=(
    --algo.config.module-dict.actor.type=LSTM
    --algo.config.module-dict.actor.layer-config.lstm-hidden-dim=256
    --algo.config.module-dict.actor.layer-config.lstm-num-layers=1
    --algo.config.module-dict.critic.type=LSTM
    --algo.config.module-dict.critic.layer-config.lstm-hidden-dim=256
    --algo.config.module-dict.critic.layer-config.lstm-num-layers=1
  )
else
  POLICY_ARGS+=(
    --algo.config.module-dict.actor.type=MLP
    --algo.config.module-dict.critic.type=MLP
  )
fi

TRAIN_ARGS=(
  exp:g1-29dof-wbt-w-object-generalist-teacher-linvel
  command:g1-29dof-wbt-w-object-generalist
  reward:g1-29dof-wbt-w-object-generalist-tracking-no-contact
  perception:camera_depth_d435i
  termination:g1-29dof-wbt-generalist
  randomization:g1-29dof-wbt-w-object-teacher-state-robust-with-camera
  logger:wandb
  --training.project=carry-any
  --training.name="${RUN_NAME_EFFECTIVE}"
  --training.num-envs="${TOTAL_ENVIRONMENTS}"
  --training.seed=42
  --training.multigpu=True
  --training.export-onnx=True
  --algo.config.distill.enabled=False
  --algo.config.num-learning-iterations="${TARGET_ITERATIONS}"
  --algo.config.num-steps-per-env=24
  --algo.config.num-learning-epochs=7
  --algo.config.num-mini-batches=4
  --algo.config.clip-param=0.2
  --algo.config.gamma=0.99
  --algo.config.lam=0.95
  --algo.config.value-loss-coef=1.0
  --algo.config.entropy-coef=0.005
  --algo.config.max-grad-norm=1.0
  --algo.config.schedule=adaptive
  --algo.config.desired-kl=0.01
  --algo.config.actor-learning-rate=0.001
  --algo.config.critic-learning-rate=0.001
  --algo.config.min-actor-learning-rate=0.00001
  --algo.config.max-actor-learning-rate=0.01
  --algo.config.min-critic-learning-rate=0.00001
  --algo.config.max-critic-learning-rate=0.01
  --algo.config.init-noise-std=1.0
  --algo.config.module-dict.actor.min-noise-std=0.01
  --algo.config.normalize-actor-obs=False
  --algo.config.normalize-critic-obs=False
  --algo.config.init-at-random-ep-len=True
  --algo.config.save-interval="${SAVE_INTERVAL}"
  --algo.config.reset-rollout-at-checkpoint=False
  "${POLICY_ARGS[@]}"
  --observation.groups.actor_obs.history-length=1
  --observation.groups.critic_obs.history-length=1
  --command.setup-terms.motion-command.params.motion-config.motion-file="${MOTION_DIR}"
  --command.setup-terms.motion-command.params.motion-config.pure-rl-policy-command-after-lift-enabled=False
  --command.setup-terms.motion-command.params.motion-config.hybrid-stage2-enabled=False
  --command.setup-terms.motion-command.params.motion-config.hybrid-velocity-enabled=False
  --command.setup-terms.motion-command.params.motion-config.contact-aware-sparse-root-command-mode=tracking_error
  --command.setup-terms.motion-command.params.motion-config.zero-root-command-when-drop-active=True
  --command.setup-terms.motion-command.params.motion-config.clip-weighting-strategy=uniform_clip
  --command.setup-terms.motion-command.params.motion-config.pair-terrain-with-motion=False
  --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler=False
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=0.2
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end=0.2
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-start-iter=0
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end-iter="${CURRICULUM_END_ITER}"
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob=0.0
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end=0.0
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-start-iter=0
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end-iter="${CURRICULUM_END_ITER}"
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale=1.0
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.dof-pos=0.20
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.dof-vel=0.35
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.root-pos='[0.08,0.08,0.025]'
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.root-rot='[0.15,0.15,0.30]'
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.root-lin-vel='[0.20,0.20,0.10]'
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.root-ang-vel='[0.25,0.25,0.35]'
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.object-pos='[0.08,0.08,0.0]'
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-prepend=True
  --command.setup-terms.motion-command.params.motion-config.default-pose-prepend-duration-s=0.2
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-append=True
  --command.setup-terms.motion-command.params.motion-config.default-pose-append-duration-s=2.0
  --command.setup-terms.motion-command.params.motion-config.contact-interval-runtime-prepend-compensation=False
  --reward.terms.offline-contact-guidance.weight=0.0
  --perception.camera-apply-sensor-noise=True
  --perception.camera-warp-edge-noise=True
  --perception.camera-warp-enable-holes=True
  --perception.camera-warp-hole-prob=0.2
  --perception.camera-warp-additive-noise-std=0.03
  --perception.camera-warp-depth-offset-std=0.03
  --perception.object-geometry-mode=mesh
  --perception.sensor-offset='[0.0576235,0.01753,0.42987]'
  --perception.camera-mount-quat='[0.0,0.40354529635239006,0.0,0.9149596678498247]'
  --perception.camera-frame-quat='[-0.5,0.5,-0.5,0.5]'
  --robot.object.enabled=True
  --robot.object.object-urdf-path="${OBJECT_SPEC_PATH}"
  --simulator.config.scene.env-spacing=5.0
  --simulator.config.sim.max-episode-length-s=10.0
  --simulator.config.sim.physx.gpu-found-lost-pairs-capacity=335544320
  --simulator.config.sim.physx.gpu-found-lost-aggregate-pairs-capacity=469762048
  --simulator.config.sim.physx.gpu-total-aggregate-pairs-capacity=83886080
  --simulator.config.sim.physx.gpu-collision-stack-size=268435456
  --logger.entity=zihanw22
  --logger.name="${RUN_NAME_EFFECTIVE}"
  --logger.mode="$( [[ ${MODE} == formal ]] && echo online || echo offline )"
  --logger.video.enabled=False
  --logger.headless-recording=False
  --logger.video.upload-to-wandb=False
  --logger.base-dir="${LOGGER_BASE_DIR}"
)
if [[ ${MODE} == formal ]]; then
  TRAIN_ARGS+=(--logger.id="${RUN_ID_EFFECTIVE}" --logger.resume=must)
fi

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/validate_train_cli.py" \
  --expected-motion-end-mode episodic -- "${TRAIN_ARGS[@]}"

export HOLOSOMA_TRAINING_PROVENANCE
HOLOSOMA_TRAINING_PROVENANCE=$("${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/compute_training_provenance.py" \
  --training-regime pure_rl --motion-dir "${MOTION_DIR}" --object-map "${OBJECT_SPEC_PATH}" \
  --motion-shard-manifest "${HOLOSOMA_MOTION_SHARD_MANIFEST}" \
  --contact-interval-runtime-prepend-compensation false --source-root "${SOURCE_ROOT}")
HOLOSOMA_TRAINING_PROVENANCE=$("${PYTHON_BIN}" - "${HOLOSOMA_TRAINING_PROVENANCE}" \
  "${VERIFY_ROOT}/node_${NODE_RANK}.json" "${POLICY_ARCH}" <<'PY'
import json
import sys

provenance = json.loads(sys.argv[1])
git_verification = json.loads(open(sys.argv[2], encoding="utf-8").read())
provenance.update({
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
    "policy_arch": sys.argv[3],
    "lstm_hidden_dim": 256 if sys.argv[3] == "lstm" else None,
    "lstm_num_layers": 1 if sys.argv[3] == "lstm" else None,
})
print(json.dumps(provenance, sort_keys=True, separators=(",", ":")))
PY
)
printf '%s\n' "${HOLOSOMA_TRAINING_PROVENANCE}" > "${RUN_ROOT}/training_provenance.json"

if [[ ${MODE} == formal && ${NODE_RANK} == 0 ]]; then
  "${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/wandb_replay_preflight.py" verify \
    --manifest "${RULE90_PATH}" --expected-manifest-sha256 "${RULE90_SHA}" \
    --required-manifest-version 1 --expected-source-snapshot-id "${SOURCE_SNAPSHOT_ID}" \
    --expected-entity zihanw22 --expected-project carry-any --expected-run-id "${RUN_ID_EFFECTIVE}" \
    --expected-run-name "${RUN_NAME_EFFECTIVE}" --expected-world-size 32
fi

if [[ ${PREFLIGHT_ONLY:-0} == 1 ]]; then
  echo "[INFO] worker_preflight_ok mode=${MODE} policy_arch=${POLICY_ARCH} node_rank=${NODE_RANK} world_size=32 envs_per_rank=2048 global_envs=65536 clips=137 actor=178 critic=310 pure_ppo=true tracking_error=true contact=false max_episode_s=10 export_onnx=true"
  exit 0
fi
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d') ]]; then
  echo "[ERROR] GPU apps appeared after preflight" >&2
  exit 2
fi

cd "${SOURCE_ROOT}"
exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${NPROC}" --max_restarts=0 --master_port="${MASTER_PORT}" \
  src/holosoma/holosoma/train_agent_rank_visible.py "${TRAIN_ARGS[@]}"
