#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ! $2 =~ ^[0-3]$ ]]; then
  echo "usage: $0 {original|rollout} NODE_RANK(0..3)" >&2
  exit 2
fi

export ARM="$1"
export NODE_RANK="$2"
case "${ARM}" in
  original)
    EXPECTED_NODE_IPS=(10.99.0.141 10.99.0.186 10.99.1.154 10.99.0.167)
    MASTER_ADDR=10.99.0.141
    MASTER_PORT=31701
    RUN_NAME=teacher_scratch_linvel_original3_ws32_20260720_062837
    WANDB_RUN_ID=rukkpmdv
    LAUNCH_TOKEN=6d68c874cee86a02f15a6834a327afb3cca994257e1325f6e5ec1ae8cfebfd83
    BANK_REL=data/ds_as_data/teacher_ab3_true_original_solid_sharedphys__src_d67ffcdb2f5676d8ae59c8fd0450f5e2c7d498617113ab58e33da6871266fd2d
    SINGLE_VIEW_DIGEST=990ac19acd647d87428fb17bc982dc7978530689b8023e5b25b191180481f5e6
    SHARD_SOURCE_DIGEST=2cb39848ded7589e424073b70d52ab4e0b4573ce4548187cdf24076c50a139ba
    SOLID_MANIFEST_SHA=31ff7a044ba4ff6b2363a2e35850a097145412f17c3d7af152c327888ab97626
    SINGLE_MANIFEST_SHA=b9ad1c8c5bd7d9096c93ccf3ab8b13ab240f0705ee60170dd2f7ec944833e152
    SHARD_MANIFEST_SHA=f935854de44e003887445411538235977d7c2a23e0cf1fec33c1c3e56a8f0233
    REPLAY_MANIFEST_SHA=029a0e9f8d33a47654bd49450ce919e8d3f75352a74cb3dd1f960fe23a52b09e
    MOTION_BARREL_SHA=bb6e45d37b2bd5077a1e4e09d57659d92cfdaf4722daea963c1881dcb1cf4297
    MOTION_BALL_SHA=6797ebd9f4523e00c89209f81ac94e8c9df89cb6a37f9ccb4b33acbb1b79bb96
    MOTION_BIN_SHA=7f332ccce879c7c61e80ad84040bf1de13997dc2d464613dd7cb37b57092768e
    ;;
  rollout)
    EXPECTED_NODE_IPS=(10.99.0.77 10.99.0.165 10.99.0.227 10.99.0.18)
    MASTER_ADDR=10.99.0.77
    MASTER_PORT=31711
    RUN_NAME=teacher_scratch_linvel_rollout3_ws32_20260720_062837
    WANDB_RUN_ID=ppclmh15
    LAUNCH_TOKEN=6a4ece92ccf0a452ba7a28d8327111fb36e54f627d64e76d57123ca9c7c1dfee
    BANK_REL=data/ds_as_data/teacher_ab3_true_rollout_u8_solid_sharedphys__src_5ae3055eeff0098dad099ec26d721b76ac7675207e7d41a6ee3aaa2a15080b49
    SINGLE_VIEW_DIGEST=25d7623ec5d785ee12595322f5939afe4655ea8472ed14fc9b1af5cf2c915f83
    SHARD_SOURCE_DIGEST=65892fc13af9ef3115b3ddc671ac13cb1fd700b97b2c0fc38e4168aa785fbdaf
    SOLID_MANIFEST_SHA=8c9d666027e0f462c7414399c1555fd32b761654e292a1a3fb5e6352a4be4d6a
    SINGLE_MANIFEST_SHA=c1e0c883be5b1062e5510a6406bc10ee056642fdba2d9d2d9c974e9cc65d7a65
    SHARD_MANIFEST_SHA=5335d513dc6ffac141754df3b04f4eb3eab76a9b29afc45d528948cf2c475776
    REPLAY_MANIFEST_SHA=f100571288c69436456faa5ded6722be293a8a2e4cd6512d63abfec5d9e1c673
    MOTION_BARREL_SHA=382e0aaffc8a6e4dd4c1906eaed50c5ed3e244bdd3e769e5581f374e60f06126
    MOTION_BALL_SHA=87644e984e4af1e7b75f3e3f83d822a0ee18e2b7604968d38f16d5b80cae46bb
    MOTION_BIN_SHA=9c981f20edb97a9d598fee7beb15c42278ad4bb0bc1725812540fd243f35adb4
    ;;
  *)
    echo "[ERROR] arm must be original or rollout; got ${ARM}" >&2
    exit 2
    ;;
esac

readonly EXPECTED_NODE_IP="${EXPECTED_NODE_IPS[$NODE_RANK]}"
readonly SOURCE_ID=src-a8879814e247e78b0e686fb8bdf94b985d008b2960aa187b482ef8de7b644399
readonly SOURCE_DIGEST=a8879814e247e78b0e686fb8bdf94b985d008b2960aa187b482ef8de7b644399
readonly SOURCE_ROOT="/home/ubuntu/FAR/holosoma_runs/${SOURCE_ID}"
readonly LAUNCH_ROOT=/home/ubuntu/FAR/holosoma_runs/formal_teacher_ab3_ws32_scratch_linvel_20260720_062837
readonly RUN_ROOT="${LAUNCH_ROOT}/${ARM}"
readonly RANDOMIZATION_CONTRACT="${LAUNCH_ROOT}/teacher_randomization_contract.json"
readonly RANDOMIZATION_CONTRACT_SHA=aecbfa4551e9f64e1ca21ffa10a44498dc1f72094c8322a63e7ca582e4a96762
readonly AB_CONTRACT="${LAUNCH_ROOT}/ab_experiment_contract.json"
readonly AB_CONTRACT_SHA=3f9c91e09e3b06070f87a7d98d3994e2fe8f93524930b8a39b45b8141252ed68
readonly REPLAY_MANIFEST="${RUN_ROOT}/replay_preflight_manifest.json"
readonly PYTHON_RUNTIME=/home/ubuntu/FAR/holosoma_runs/.runtime/python/python-runtime-v2-2dcddba9d88fd6b5a3fe701e2b74c59eae988071e52d8bdb44a2474bc16b9e1f/site-packages
readonly PYTHON_RUNTIME_MANIFEST_SHA=2dcddba9d88fd6b5a3fe701e2b74c59eae988071e52d8bdb44a2474bc16b9e1f
readonly NCCL_ROOT=/home/ubuntu/FAR/holosoma_runs/.runtime/nccl/e4a7aee9c3eecf53fac780441d2f03b578ab8db8874b71f8e391bcec7adb2899
readonly NCCL_SHA=e4a7aee9c3eecf53fac780441d2f03b578ab8db8874b71f8e391bcec7adb2899
readonly BANK="${SOURCE_ROOT}/${BANK_REL}"
readonly SINGLE_BASE="${BANK}/_scientific_teacher_ab_single_slot"
readonly SINGLE_DIR="${SINGLE_BASE}/by-source/${SINGLE_VIEW_DIGEST}"
readonly SHARD_ROOT="${SINGLE_DIR}/_rank_shards/by-source/${SHARD_SOURCE_DIGEST}/ws32"
readonly CONTACT_ROOT="${BANK}/contact_export_from_teacher_realmesh_rollout"
readonly SOURCE_OBJECT_URDF_ROOT="${SOURCE_ROOT}/data/ds_as_data/teacher_ab3_original_raw_local_20260720/_single_slot_urdfs"
readonly SHARED_OBJECT_MESH_ROOT="${SOURCE_ROOT}/data/ds_as_data/prism_debug30_convexhull_allmesh/objects_convex_hull"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/wandb" "${RUN_ROOT}/provenance-cache/node_${NODE_RANK}"
status_file="${RUN_ROOT}/node_rank_${NODE_RANK}.exit"
rm -f "${status_file}"
trap 'rc=$?; printf "%s\t%s\t%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$ARM" "$NODE_RANK" > "$status_file"' EXIT

if ! hostname -I | tr ' ' '\n' | grep -Fxq "${EXPECTED_NODE_IP}"; then
  echo "[ERROR] node-rank/IP mismatch: arm=${ARM} rank=${NODE_RANK} expected=${EXPECTED_NODE_IP} actual=$(hostname -I)" >&2
  exit 2
fi

check_sha() {
  local expected="$1" path="$2" actual
  [[ -f "${path}" ]] || { echo "[ERROR] missing integrity input: ${path}" >&2; exit 2; }
  actual=$(sha256sum "${path}" | awk '{print $1}')
  [[ "${actual}" == "${expected}" ]] || {
    echo "[ERROR] SHA mismatch: ${path} expected=${expected} actual=${actual}" >&2
    exit 2
  }
}

[[ -f "${SOURCE_ROOT}/.holosoma_snapshot/id" ]]
grep -Fxq "${SOURCE_ID}" "${SOURCE_ROOT}/.holosoma_snapshot/id"
(cd "${SOURCE_ROOT}" && sha256sum --quiet -c .holosoma_snapshot/source_manifest.sha256)
check_sha "${RANDOMIZATION_CONTRACT_SHA}" "${RANDOMIZATION_CONTRACT}"
check_sha "${AB_CONTRACT_SHA}" "${AB_CONTRACT}"
check_sha "${REPLAY_MANIFEST_SHA}" "${REPLAY_MANIFEST}"
check_sha "${PYTHON_RUNTIME_MANIFEST_SHA}" "${PYTHON_RUNTIME}/.holosoma-runtime-manifest.sha256"
check_sha "${NCCL_SHA}" "${NCCL_ROOT}/libnccl.so.2"
check_sha "${SOLID_MANIFEST_SHA}" "${BANK}/manifest.json"
check_sha 02acd5041a0dc9f5a60e0f3a09b9de7014a1d6fc4032113d8ead2682ac6c5afc "${BANK}/_clip_object_urdf_map.json"
check_sha "${SINGLE_MANIFEST_SHA}" "${SINGLE_DIR}/manifest.json"
check_sha "${SHARD_MANIFEST_SHA}" "${SHARD_ROOT}/manifest.json"
check_sha "${MOTION_BARREL_SHA}" "${BANK}/scaledown__any_barrel_25.npz"
check_sha "${MOTION_BALL_SHA}" "${BANK}/unscale__any_ball_29.npz"
check_sha "${MOTION_BIN_SHA}" "${BANK}/unscale__any_bin_29.npz"
check_sha 5f52fbf224f05e9b3345bf49fabfeec4507a5507236fd002b66befca88f8d5e3 "${SOURCE_OBJECT_URDF_ROOT}/scaledown_any_barrel_25.urdf"
check_sha ae6a5e08a01e04c81dd10f4e79892cb394e1b24b4fa19f6c6b6579a9fe9d33de "${SOURCE_OBJECT_URDF_ROOT}/unscale_any_ball_29.urdf"
check_sha 5a0d2fadf8d3ce48d1d5b865c8c83164a51e6a7dc05f2dd69bdbe6de317d94ce "${SOURCE_OBJECT_URDF_ROOT}/unscale_any_bin_29.urdf"
check_sha 24d046ad6047fa8f33c63138c9be35975d5a0e078bc6df321b2068442d64f4c5 "${SHARED_OBJECT_MESH_ROOT}/object_mesh_yup_e7481f4304/object_mesh_yup_convex_hull.obj"
check_sha 9734a65b4cd1127c96fad2b499832cbe5f5c7608200c593127c45db31b92d5b9 "${SHARED_OBJECT_MESH_ROOT}/object_mesh_yup_e6603064bd/object_mesh_yup_convex_hull.obj"
check_sha daae95872696e55484f37a166978fca182303ce1bb73b26d851b0d085784890d "${SHARED_OBJECT_MESH_ROOT}/object_mesh_yup_266fbb26f5/object_mesh_yup_convex_hull.obj"
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
export HOLOSOMA_LAUNCH_TOKEN="${LAUNCH_TOKEN}"
export HOLOSOMA_LAUNCH_EPOCH=1784529558

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC=8
export NNODES=4
export MASTER_ADDR
export MASTER_PORT
export NUM_ENVS=65536
export PER_GPU_ENVS=2048
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
export AS_EXPECTED_TOTAL=3
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
export EXP=g1-29dof-wbt-w-object-generalist-teacher-linvel
export COMMAND_CONFIG=g1-29dof-wbt-w-object-generalist

unset RESUME_CKPT RESUME_CHECKPOINT RESUME_MODEL_FILE RESUME_STEP
unset POLICY_INIT_CKPT POLICY_INIT_CHECKPOINT RESUME_FROM_BOX BOX_RESUME_CKPT
unset RESUME_WANDB_ID WANDB_MODE WANDB_DISABLED
export WANDB_PROJECT=carry-any
export WANDB_ENTITY=zihanw22
export WANDB_RUN_ID
export WANDB_RESUME=must
export WANDB_RESUME_SAME_RUN=0
export WANDB_CONSOLE=off
export WANDB_INIT_TIMEOUT=120
export WANDB_DIR="${RUN_ROOT}/wandb"
export SEQUENCE_NAME="${RUN_NAME}"
export ACTOR_LR=1e-3
export CRITIC_LR=1e-3
export NUM_LEARNING_EPOCHS=7
export NUM_LEARNING_ITERATIONS=40000
export SAVE_INTERVAL=1000
export EXPORT_ONNX=False

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[INFO] formal_teacher_ab_worker_preflight_only_ok arm=${ARM} node_rank=${NODE_RANK} host=$(hostname) ip=${EXPECTED_NODE_IP}"
  exit 0
fi

cd "${SOURCE_ROOT}"
echo "[INFO] formal_teacher_ab_worker_preflight_ok arm=${ARM} node_rank=${NODE_RANK} host=$(hostname) ip=${EXPECTED_NODE_IP} source=${SOURCE_ID} wandb=zihanw22/carry-any/${WANDB_RUN_ID} motion_kind=${ARM} fresh_policy=true teacher_linvel=true checkpoint_reset=false"
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
  --algo.config.min-actor-learning-rate=0.00001 \
  --algo.config.max-actor-learning-rate=0.01 \
  --algo.config.min-critic-learning-rate=0.00001 \
  --algo.config.max-critic-learning-rate=0.01 \
  --algo.config.init-noise-std=1.0 \
  --algo.config.module-dict.actor.layer-config.hidden-dims='[512,256,128]' \
  --algo.config.module-dict.critic.layer-config.hidden-dims='[512,256,128]'
