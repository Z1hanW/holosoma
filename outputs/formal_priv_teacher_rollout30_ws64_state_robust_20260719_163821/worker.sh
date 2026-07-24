#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-7]$ ]]; then
  echo "usage: $0 NODE_RANK(0..7)" >&2
  exit 2
fi

export NODE_RANK="$1"
readonly EXPECTED_NODE_IPS=(
  10.99.0.141
  10.99.0.186
  10.99.1.154
  10.99.0.167
  10.99.0.77
  10.99.0.165
  10.99.0.227
  10.99.0.18
)
readonly EXPECTED_NODE_IP="${EXPECTED_NODE_IPS[$NODE_RANK]}"
readonly SOURCE_ID=src-a3d2416dc71f953488fa3c447fc110e9880ae1c8536d58ec4951e4f1d0d8738c
readonly SOURCE_DIGEST=a3d2416dc71f953488fa3c447fc110e9880ae1c8536d58ec4951e4f1d0d8738c
readonly SOURCE_ROOT="/home/ubuntu/FAR/holosoma_runs/${SOURCE_ID}"
readonly RUN_ROOT=/home/ubuntu/FAR/holosoma_runs/formal_priv_teacher_rollout30_ws64_state_robust_20260719_163821
readonly RUN_NAME=priv_teacher_rollout30_ws64_state_robust_u8init_noreset_20260719_163821
readonly RANDOMIZATION_CONTRACT="${RUN_ROOT}/teacher_randomization_contract.json"
readonly RANDOMIZATION_CONTRACT_SHA=6477e9657a1d635dc287350972fe2f88139c523a02b6c7e46504ced9031960f3
readonly INIT_SHA=80cb13e13590239d015ba0a29bdbae901b7785a9789d144745c7bd330059cd68
readonly POLICY_INIT=/home/ubuntu/FAR/holosoma_runs/.assets/teacher-policy-init/80cb13e13590239d015ba0a29bdbae901b7785a9789d144745c7bd330059cd68/model_05000.pt
readonly PYTHON_RUNTIME=/home/ubuntu/FAR/holosoma_runs/.runtime/python/python-runtime-v2-2dcddba9d88fd6b5a3fe701e2b74c59eae988071e52d8bdb44a2474bc16b9e1f/site-packages
readonly PYTHON_RUNTIME_MANIFEST_SHA=2dcddba9d88fd6b5a3fe701e2b74c59eae988071e52d8bdb44a2474bc16b9e1f
readonly NCCL_ROOT=/home/ubuntu/FAR/holosoma_runs/.runtime/nccl/e4a7aee9c3eecf53fac780441d2f03b578ab8db8874b71f8e391bcec7adb2899
readonly NCCL_SHA=e4a7aee9c3eecf53fac780441d2f03b578ab8db8874b71f8e391bcec7adb2899
readonly BANK_REL=data/ds_as_data/prism_debug30_convexhull_allmesh_solid_box_bin_barrel_ball
readonly BANK="${SOURCE_ROOT}/${BANK_REL}"
readonly SINGLE_BASE="${BANK}/_scientific_teacher64_single_slot"
readonly SINGLE_DIR="${SINGLE_BASE}/by-source/77738917deb60e578dc695841b3a07b10ad4f50371d3c0500474f41c78f71f90"
readonly SHARD_ROOT="${BANK}/_scientific_teacher64_rank_shards"
readonly CONTACT_ROOT="${SOURCE_ROOT}/data/ds_as_data/debug39_realmesh_rollout_u8udzw0u_model05000_retake4gpu_20260706_0205_target/contact_export_from_teacher_realmesh_rollout"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/wandb" "${RUN_ROOT}/provenance-cache/node_${NODE_RANK}"
status_file="${RUN_ROOT}/node_rank_${NODE_RANK}.exit"
rm -f "${status_file}"
trap 'rc=$?; printf "%s\t%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$NODE_RANK" > "$status_file"' EXIT

if ! hostname -I | tr ' ' '\n' | grep -Fxq "${EXPECTED_NODE_IP}"; then
  echo "[ERROR] node-rank/IP mismatch: rank=${NODE_RANK} expected=${EXPECTED_NODE_IP} actual=$(hostname -I)" >&2
  exit 2
fi
[[ -f "${SOURCE_ROOT}/.holosoma_snapshot/id" ]]
grep -Fxq "${SOURCE_ID}" "${SOURCE_ROOT}/.holosoma_snapshot/id"
grep -Eq "^[0-9a-f]{64}  " "${SOURCE_ROOT}/.holosoma_snapshot/source_manifest.sha256"
(cd "${SOURCE_ROOT}" && sha256sum --quiet -c .holosoma_snapshot/source_manifest.sha256)

check_sha() {
  local expected="$1" path="$2" actual
  [[ -f "${path}" ]] || { echo "[ERROR] missing integrity input: ${path}" >&2; exit 2; }
  actual=$(sha256sum "${path}" | awk '{print $1}')
  [[ "${actual}" == "${expected}" ]] || {
    echo "[ERROR] SHA mismatch: ${path} expected=${expected} actual=${actual}" >&2
    exit 2
  }
}
check_sha "${RANDOMIZATION_CONTRACT_SHA}" "${RANDOMIZATION_CONTRACT}"
check_sha "${INIT_SHA}" "${POLICY_INIT}"
check_sha "${PYTHON_RUNTIME_MANIFEST_SHA}" "${PYTHON_RUNTIME}/.holosoma-runtime-manifest.sha256"
check_sha "${NCCL_SHA}" "${NCCL_ROOT}/libnccl.so.2"
check_sha fa243691f3cf81f3d5f98a23708df94005e4b560944aec5d53574bf9e07bba34 "${SINGLE_DIR}/manifest.json"
check_sha 475dff39cbe73580467ff9a3eb042a4c475bcc11db6d42fe784a41fadd8ab90a "${SHARD_ROOT}/manifest.json"
[[ -d "${CONTACT_ROOT}/clips" ]]

mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,name,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits)
[[ ${#gpu_rows[@]} -eq 8 ]] || { echo "[ERROR] expected 8 GPUs, found ${#gpu_rows[@]}" >&2; exit 2; }
for row in "${gpu_rows[@]}"; do
  [[ "${row}" == *"NVIDIA L40S"* ]] || { echo "[ERROR] unexpected GPU: ${row}" >&2; exit 2; }
  [[ "${row##*, }" == 0 ]] || { echo "[ERROR] nonzero volatile UECC: ${row}" >&2; exit 2; }
done
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d') ]]; then
  echo "[ERROR] selected node is no longer GPU-idle; refusing launch" >&2
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader >&2 || true
  exit 2
fi

export HOME=/home/ubuntu
export PYTHON_BIN=/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3.11
export PATH="$(dirname "${PYTHON_BIN}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="${PYTHON_RUNTIME}:${SOURCE_ROOT}/src/holosoma:${SOURCE_ROOT}/src/holosoma_inference:${SOURCE_ROOT}/src"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export OMP_NUM_THREADS=1

export HOLOSOMA_SOURCE_ROOT="${SOURCE_ROOT}"
export HOLOSOMA_SOURCE_SNAPSHOT_ID="${SOURCE_ID}"
export HOLOSOMA_SOURCE_MANIFEST_SHA256="${SOURCE_DIGEST}"
export HOLOSOMA_PYTHON_RUNTIME_MANIFEST_SHA256="${PYTHON_RUNTIME_MANIFEST_SHA}"
export PYTHON_RUNTIME_SITEPACKAGES="${PYTHON_RUNTIME}"
export HOLOSOMA_DATA_PROVENANCE_CACHE_ROOT="${RUN_ROOT}/provenance-cache/node_${NODE_RANK}"
export HOLOSOMA_LAUNCH_TOKEN=21c32b5b7fc92c71711e485fe30ba9ce9ef18bef9322d968870981cce8f45e81
export HOLOSOMA_LAUNCH_EPOCH=1784479913

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC=8
export NNODES=8
export MASTER_ADDR=10.99.0.141
export MASTER_PORT=31651
export NUM_ENVS=65536
export PER_GPU_ENVS=1024
export TRAINING_SEED=42

export NCCL_LIB_DIR="${NCCL_ROOT}"
export NCCL_LIB_SHA256="${NCCL_SHA}"
export LD_LIBRARY_PATH="${NCCL_ROOT}"
export LD_PRELOAD="${NCCL_ROOT}/libnccl.so.2"
export NCCL_SOCKET_IFNAME=enp135s0
export GLOO_SOCKET_IFNAME=enp135s0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_RETRY_CNT=34
export NCCL_SOCKET_RETRY_SLEEP_MSEC=100
export NCCL_SOCKET_NTHREADS=2
export NCCL_NSOCKS_PERTHREAD=4
export TORCH_DIST_BACKEND=gloo
export TORCH_DIST_TIMEOUT_SEC=3600
export HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC=300
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=65536
export TORCH_NCCL_PROPAGATE_ERROR=1
export TORCH_NCCL_DESYNC_DEBUG=0
export TORCH_NCCL_ENABLE_TIMING=0
export TORCH_NCCL_BLOCKING_WAIT=0
export HOLOSOMA_GLOO_GRAD_REDUCE=0
export HOLOSOMA_GLOO_BARRIER=1
export HOLOSOMA_GLOO_SMALL_COLLECTIVES=1
export HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1
export HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=0
export HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER=1
export HOLOSOMA_RANK_VISIBLE_DEVICES=1
export HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE=1
export HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE=0
export HOLOSOMA_SYNC_AFTER_OPTIMIZER_STEP=0
export HOLOSOMA_SYNC_AFTER_MICROBATCH_FORWARD=0
export HOLOSOMA_RANK_LOCAL_CPU_AFFINITY=1
export HOLOSOMA_CONTIGUOUS_MINIBATCHES=1
export HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY=0

export AS_DATA_DIR="${BANK}"
export AS_OBJECT_MAP="${BANK}/_clip_object_urdf_map.json"
export AS_EXPECTED_TOTAL=30
export AS_SINGLE_SLOT_MOTION_BASE="${SINGLE_BASE}"
export HOLOSOMA_RANK_LOCAL_MOTION_ROOT="${SHARD_ROOT}"
export HOLOSOMA_MOTION_SHARD_MANIFEST="${SHARD_ROOT}/manifest.json"
export HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED=1
export HOLOSOMA_REQUIRE_RANK_LOCAL_SHARD_PROVENANCE=1
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=0
export CLIP_WEIGHTING_STRATEGY=uniform_clip
export USE_ADAPTIVE_TIMESTEPS_SAMPLER=False
export FREEZE_AT_TIMESTEP_ZERO_PROB=0.0
export DEFAULT_POSE_PREPEND_ENABLED=1
export DEFAULT_POSE_PREPEND_DURATION_S=0.2
export CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION=True
export CONTACT_EXPORT_ROOT="${CONTACT_ROOT}"
export ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT="${CONTACT_ROOT}/clips"
export REWARD_CONFIG=g1-29dof-wbt-w-object-generalist-offline-contact-guidance
export RANDOMIZATION_PRESET=g1_29dof_wbt_w_object_teacher_state_robust
export INIT_AT_RANDOM_EP_LEN=True
export POLICY_HISTORY_LENGTH=1
export DISABLE_ACTOR_HISTORY=True
export DISABLE_CRITIC_HISTORY=True

unset RESUME_CKPT RESUME_CHECKPOINT RESUME_MODEL_FILE RESUME_STEP
export POLICY_INIT_CKPT="${POLICY_INIT}"
export WANDB_PROJECT=carry-any
export WANDB_ENTITY=zihanw22
export WANDB_RUN_ID=aofybnlc
export WANDB_RESUME=must
export WANDB_RESUME_SAME_RUN=0
export WANDB_CONSOLE=off
export WANDB_INIT_TIMEOUT=120
export WANDB_DIR="${RUN_ROOT}/wandb"
export SEQUENCE_NAME="${RUN_NAME}"
export ACTOR_LR=1e-5
export CRITIC_LR=1e-5
export NUM_LEARNING_EPOCHS=7
export NUM_LEARNING_ITERATIONS=40000
export SAVE_INTERVAL=1000
export EXPORT_ONNX=False

cd "${SOURCE_ROOT}"
echo "[INFO] formal_teacher_worker_preflight_ok node_rank=${NODE_RANK} host=$(hostname) ip=${EXPECTED_NODE_IP} source=${SOURCE_ID} wandb=zihanw22/carry-any/aofybnlc randomization=g1_29dof_wbt_w_object_teacher_state_robust checkpoint_reset=false"
bash train_as_general.sh "${RUN_NAME}" \
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=0.2 \
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale=1.0 \
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.dof-pos=0.20 \
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.dof-vel=0.35 \
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.root-pos='[0.08,0.08,0.025]' \
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.root-rot='[0.15,0.15,0.30]' \
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.root-lin-vel='[0.20,0.20,0.10]' \
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.root-ang-vel='[0.25,0.25,0.35]' \
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.object-pos='[0.08,0.08,0.0]' \
  --algo.config.reset-rollout-at-checkpoint=False \
  --algo.config.num-steps-per-env=24 \
  --algo.config.num-mini-batches=4 \
  --algo.config.clip-param=0.2 \
  --algo.config.gamma=0.99 \
  --algo.config.lam=0.95 \
  --algo.config.value-loss-coef=1.0 \
  --algo.config.entropy-coef=0.005 \
  --algo.config.max-grad-norm=1.0 \
  --algo.config.schedule=adaptive \
  --algo.config.desired-kl=0.01 \
  --algo.config.init-noise-std=1.0 \
  --algo.config.module-dict.actor.layer-config.hidden-dims='[512,256,128]' \
  --algo.config.module-dict.critic.layer-config.hidden-dims='[512,256,128]'
