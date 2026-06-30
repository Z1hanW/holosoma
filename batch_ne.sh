#!/usr/bin/env bash
set -euo pipefail

# Prepare and launch the 51-clip convex-hull AS solid distillation run across
# the 8xL40S nodes listed below. This script runs from one control node and
# starts one tmux session per node.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "${SCRIPT_DIR}"

# Private IPs for the requested SkyPilot clusters. Cluster names are not DNS
# names on the training nodes, so default to VPC-reachable private IPs.
DEFAULT_NODES=(
  10.99.1.60   # z1hanw
  10.99.1.122  # zzzihanw-f
  10.99.1.21   # zzzihanw-e
  10.99.0.18
  10.99.0.227
  10.99.0.116
  10.99.0.165
  10.99.0.167
)

usage() {
  cat <<'EOF'
Usage:
  bash batch_ne.sh prepare     # git pull + cp_ch.sh on every node
  bash batch_ne.sh launch      # start multi-node training in tmux
  bash batch_ne.sh all         # prepare, then launch
  bash batch_ne.sh status      # show tmux/log status on every node
  bash batch_ne.sh stop        # kill only this script's tmux session

Useful env:
  NODES="node0 node1 ..."      override node list
  REMOTE_REPO=/home/ubuntu/FAR/holosoma
  SESSION=distill_as_ch51_64gpu
  PER_GPU_ENVS=2048            1024 minimum recommended; try 4096 if stable
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  MASTER_ADDR=<node0>          default first node
  MASTER_PORT=29651
  NCCL_SOCKET_IFNAME=enp135s0  network interface for NCCL
  GLOO_SOCKET_IFNAME=enp135s0  network interface for Gloo
  NCCL_IB_DISABLE=1            force TCP socket path on these nodes
  NCCL_SOCKET_FAMILY=AF_INET   force IPv4 on the private VPC interface
  NCCL_LIB_DIR=<path>          prepend runtime NCCL library directory
  TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
  TORCH_DIST_BACKEND=gloo      use pure Gloo for torch distributed
  TORCH_NCCL_ENABLE_MONITORING=0
  TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
  TORCH_NCCL_DUMP_ON_TIMEOUT=1
  HOLOSOMA_GLOO_GRAD_REDUCE=1
  HOLOSOMA_GLOO_BARRIER=1
  HOLOSOMA_GLOO_SMALL_COLLECTIVES=1
  HOLOSOMA_SKIP_WANDB_CHECKPOINT_UPLOAD=1
  GIT_REMOTE=origin
  GIT_BRANCH=main
  CH_BANK_NAME=as_realmesh67000_finalpos_convexsurface51_convexhull
  RESUME_FROM_BOX=1            initialize student policy from box-button checkpoint
  BOX_POLICY_INIT_REF=<ckpt>   default box policy initializer for RESUME_FROM_BOX=1
  BOX_POLICY_INIT_CACHE_ROOT=~/.cache/holosoma/checkpoints
  RESUME_TRAINING_CKPT=<ckpt>  full training checkpoint resume; disables RESUME_FROM_BOX unless explicitly set
  RESUME_WANDB_RUN_ID=<id>     optional same-run W&B resume id for RESUME_TRAINING_CKPT
  WANDB_RESUME_MODE=must       W&B resume mode when RESUME_WANDB_RUN_ID is set
  STUDENT_ACTOR_HIDDEN_DIMS='[2048,1024,512,256,128]'  default for new/init runs; unset for full resume
  PPO_SCHEDULE_STEP_EPOCHS=700 new runs increase PPO by 0.1 every 700 iters, so BC drops by 0.1
  NUM_MINI_BATCHES=64
  NUM_LEARNING_EPOCHS=1
  RESTART=1                    kill existing tmux session with same name
  DRY_RUN=1                    print remote commands only
EOF
}

ACTION=${1:-all}
case "${ACTION}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

if [[ -n "${NODES:-}" ]]; then
  # shellcheck disable=SC2206
  NODE_LIST=(${NODES})
else
  NODE_LIST=("${DEFAULT_NODES[@]}")
fi

if [[ "${#NODE_LIST[@]}" -lt 1 ]]; then
  echo "[ERROR] Empty node list." >&2
  exit 2
fi

REMOTE_REPO=${REMOTE_REPO:-/home/ubuntu/FAR/holosoma}
SESSION=${SESSION:-distill_as_ch51_64gpu}
PER_GPU_ENVS=${PER_GPU_ENVS:-2048}
MIN_PER_GPU_ENVS=${MIN_PER_GPU_ENVS:-1024}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
NNODES=${NNODES:-${#NODE_LIST[@]}}
MASTER_ADDR=${MASTER_ADDR:-${NODE_LIST[0]}}
MASTER_PORT=${MASTER_PORT:-29651}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp135s0}
GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
NCCL_DEBUG=${NCCL_DEBUG:-WARN}
TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
TORCH_DIST_BACKEND=${TORCH_DIST_BACKEND:-gloo}
TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0}
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}
TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}
TORCH_NCCL_PROPAGATE_ERROR=${TORCH_NCCL_PROPAGATE_ERROR:-1}
TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG:-0}
TORCH_NCCL_ENABLE_TIMING=${TORCH_NCCL_ENABLE_TIMING:-0}
TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-0}
NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
NCCL_SOCKET_RETRY_CNT=${NCCL_SOCKET_RETRY_CNT:-120}
NCCL_SOCKET_RETRY_SLEEP_MSEC=${NCCL_SOCKET_RETRY_SLEEP_MSEC:-1000}
NCCL_SOCKET_NTHREADS=${NCCL_SOCKET_NTHREADS:-2}
NCCL_NSOCKS_PERTHREAD=${NCCL_NSOCKS_PERTHREAD:-4}
NCCL_LIB_DIR=${NCCL_LIB_DIR:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/nvidia/nccl/lib}
GIT_REMOTE=${GIT_REMOTE:-origin}
GIT_BRANCH=${GIT_BRANCH:-main}
CH_BANK_NAME=${CH_BANK_NAME:-as_realmesh67000_finalpos_convexsurface51_convexhull}
RESTART=${RESTART:-1}
DRY_RUN=${DRY_RUN:-0}
SSH_OPTS=${SSH_OPTS:-"-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
RESUME_FROM_BOX_WAS_SET=${RESUME_FROM_BOX+x}
RESUME_FROM_BOX=${RESUME_FROM_BOX:-1}
RESUME_TRAINING_CKPT=${RESUME_TRAINING_CKPT:-${RESUME_CHECKPOINT:-${RESUME_CKPT:-}}}
RESUME_WANDB_RUN_ID=${RESUME_WANDB_RUN_ID:-${WANDB_RUN_ID:-${RESUME_WANDB_ID:-}}}
WANDB_RESUME_MODE=${WANDB_RESUME_MODE:-${WANDB_RESUME:-must}}
WANDB_ENTITY=${WANDB_ENTITY:-zihanw22}
WANDB_RESUME_SAME_RUN=${WANDB_RESUME_SAME_RUN:-}
STUDENT_ACTOR_HIDDEN_DIMS_WAS_SET=${STUDENT_ACTOR_HIDDEN_DIMS+x}
NUM_LEARNING_ITERATIONS=${NUM_LEARNING_ITERATIONS:-40000}
NUM_MINI_BATCHES=${NUM_MINI_BATCHES:-64}
NUM_LEARNING_EPOCHS=${NUM_LEARNING_EPOCHS:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}
if [[ -n "${RESUME_TRAINING_CKPT}" ]]; then
  # Full training resumes should continue in the late hybrid regime by default:
  # BC/DAgger weight 0.1 and PPO/RL weight 0.9.
  PPO_START_EPOCH=${PPO_START_EPOCH:-0}
  DAGGER_END_EPOCH=${DAGGER_END_EPOCH:-1}
  PPO_START_COEFF=${PPO_START_COEFF:-0.9}
  PPO_TARGET_COEFF=${PPO_TARGET_COEFF:-0.9}
else
  # New runs start from pure DAgger/BC and drop BC by 0.1 every 700 iterations:
  # PPO coeff 0.0, 0.1, ..., 0.9 over iterations 0..6300.
  PPO_START_EPOCH=${PPO_START_EPOCH:-0}
  DAGGER_END_EPOCH=${DAGGER_END_EPOCH:-6300}
  PPO_START_COEFF=${PPO_START_COEFF:-0.0}
  PPO_TARGET_COEFF=${PPO_TARGET_COEFF:-0.9}
fi
PPO_SCHEDULE_STEP_EPOCHS=${PPO_SCHEDULE_STEP_EPOCHS:-700}
DAGGER_LOSS_COEF=${DAGGER_LOSS_COEF:-1.0}
HOLOSOMA_SKIP_INITIAL_CHECKPOINT=${HOLOSOMA_SKIP_INITIAL_CHECKPOINT:-1}
HOLOSOMA_SKIP_GRAD_FINITE_CHECK=${HOLOSOMA_SKIP_GRAD_FINITE_CHECK:-1}
HOLOSOMA_SKIP_LOSS_FINITE_CHECK=${HOLOSOMA_SKIP_LOSS_FINITE_CHECK:-1}
HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION=${HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION:-1}
HOLOSOMA_DEBUG_HEARTBEAT=${HOLOSOMA_DEBUG_HEARTBEAT:-0}
HOLOSOMA_DEBUG_ACTOR=${HOLOSOMA_DEBUG_ACTOR:-0}
HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE=${HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE:-0}
HOLOSOMA_SYNC_AFTER_OPTIMIZER_STEP=${HOLOSOMA_SYNC_AFTER_OPTIMIZER_STEP:-0}
HOLOSOMA_SYNC_AFTER_MICROBATCH_FORWARD=${HOLOSOMA_SYNC_AFTER_MICROBATCH_FORWARD:-0}
HOLOSOMA_DEBUG_MICROBATCH_ALL=${HOLOSOMA_DEBUG_MICROBATCH_ALL:-0}
HOLOSOMA_GLOO_GRAD_REDUCE=${HOLOSOMA_GLOO_GRAD_REDUCE:-1}
HOLOSOMA_GLOO_BARRIER=${HOLOSOMA_GLOO_BARRIER:-1}
HOLOSOMA_GLOO_SMALL_COLLECTIVES=${HOLOSOMA_GLOO_SMALL_COLLECTIVES:-1}
HOLOSOMA_CONTIGUOUS_MINIBATCHES=${HOLOSOMA_CONTIGUOUS_MINIBATCHES:-1}
HOLOSOMA_DAGGER_SUPERVISED_ONLY=${HOLOSOMA_DAGGER_SUPERVISED_ONLY:-1}
HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP=${HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP:-1}
HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH=${HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH:-16}
HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD=${HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD:-1}
HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC=${HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC:-1}
HOLOSOMA_SKIP_WANDB_CHECKPOINT_UPLOAD=${HOLOSOMA_SKIP_WANDB_CHECKPOINT_UPLOAD:-1}
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}
DEFAULT_BOX_RESUME_RUN=${DEFAULT_BOX_RESUME_RUN:-https://wandb.ai/zihanw22/boxer/runs/d9m3z369-recovered}
DEFAULT_BOX_RESUME_MODEL_FILE=${DEFAULT_BOX_RESUME_MODEL_FILE:-model_22000.pt}
DEFAULT_BOX_RESUME_CHECKPOINT=${DEFAULT_BOX_RESUME_CHECKPOINT:-${DEFAULT_BOX_RESUME_RUN}/files/${DEFAULT_BOX_RESUME_MODEL_FILE}}
BOX_POLICY_INIT_REF=${BOX_POLICY_INIT_REF:-${BOX_RESUME_CKPT:-${RESUME_FROM_BOX_CKPT:-${DEFAULT_BOX_RESUME_CHECKPOINT}}}}
BOX_POLICY_INIT_CACHE_ROOT=${BOX_POLICY_INIT_CACHE_ROOT:-/home/ubuntu/.cache/holosoma/checkpoints}
case "$(echo "${RESUME_FROM_BOX}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    RESUME_FROM_BOX=1
    ;;
  0|false|no|off|"")
    RESUME_FROM_BOX=0
    ;;
  *)
    echo "[ERROR] RESUME_FROM_BOX must be a boolean. Got: ${RESUME_FROM_BOX}" >&2
    exit 2
    ;;
esac
if [[ -n "${RESUME_TRAINING_CKPT}" ]]; then
  case "${RESUME_TRAINING_CKPT}" in
    wandb://*|*.pt|/*|./*|../*)
      ;;
    https://wandb.ai/*/runs/*/files/*.pt)
      ;;
    *)
      echo "[ERROR] RESUME_TRAINING_CKPT must be a .pt checkpoint path, wandb:// URI, or W&B /files/<model>.pt URL. Got: ${RESUME_TRAINING_CKPT}" >&2
      exit 2
      ;;
  esac
  if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
    if [[ -n "${RESUME_FROM_BOX_WAS_SET}" ]]; then
      echo "[ERROR] RESUME_TRAINING_CKPT is a full checkpoint resume and cannot be combined with RESUME_FROM_BOX=1." >&2
      exit 2
    fi
    RESUME_FROM_BOX=0
  fi
fi
if [[ -z "${WANDB_RESUME_SAME_RUN}" ]]; then
  if [[ -n "${RESUME_WANDB_RUN_ID}" ]]; then
    WANDB_RESUME_SAME_RUN=1
  else
    WANDB_RESUME_SAME_RUN=0
  fi
fi
if [[ -z "${STUDENT_ACTOR_HIDDEN_DIMS_WAS_SET}" ]]; then
  if [[ -n "${RESUME_TRAINING_CKPT}" ]]; then
    STUDENT_ACTOR_HIDDEN_DIMS=""
  else
    STUDENT_ACTOR_HIDDEN_DIMS="[2048,1024,512,256,128]"
  fi
fi
NFS_CH_BANK=${NFS_CH_BANK:-/nfs/zzzihanw/ds_as_data/_distill/${CH_BANK_NAME}.tar}

if ! [[ "${PER_GPU_ENVS}" =~ ^[0-9]+$ ]] || (( PER_GPU_ENVS < MIN_PER_GPU_ENVS )); then
  echo "[ERROR] PER_GPU_ENVS must be an integer >= ${MIN_PER_GPU_ENVS}. Got: ${PER_GPU_ENVS}" >&2
  exit 2
fi
if ! [[ "${NPROC}" =~ ^[0-9]+$ ]] || (( NPROC < 1 )); then
  echo "[ERROR] NPROC must be a positive integer. Got: ${NPROC}" >&2
  exit 2
fi
if ! [[ "${NNODES}" =~ ^[0-9]+$ ]] || (( NNODES != ${#NODE_LIST[@]} )); then
  echo "[ERROR] NNODES must equal node list length. Got NNODES=${NNODES}, nodes=${#NODE_LIST[@]}" >&2
  exit 2
fi

TOTAL_GPUS=$((NPROC * NNODES))
TOTAL_NUM_ENVS=$((PER_GPU_ENVS * TOTAL_GPUS))
if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  RUN_NAME=${RUN_NAME:-g1_w_object_distill_as_button_solid_ch51_${TOTAL_GPUS}gpu_init_box}
  TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_button_solid_ch51_${TOTAL_GPUS}gpu_init_box_depth}
else
  RUN_NAME=${RUN_NAME:-g1_w_object_distill_as_button_solid_ch51_${TOTAL_GPUS}gpu}
  TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_button_solid_ch51_${TOTAL_GPUS}gpu_depth}
fi
SCHEDULE_NAME=${SCHEDULE_NAME:-as_ch51_sparse_root_ppo_first_contact_drop_button_solid}
STUDENT_ACTOR_HIDDEN_DIMS_NOTE=${STUDENT_ACTOR_HIDDEN_DIMS:-checkpoint/default}
SCHEDULE_NOTES=${SCHEDULE_NOTES:-"${NNODES} nodes x ${NPROC} GPUs AS solid distillation on the 51-clip collision-convex-hull bank, PER_GPU_ENVS=${PER_GPU_ENVS}, actor hidden dims ${STUDENT_ACTOR_HIDDEN_DIMS_NOTE}. PPO/DAgger schedule: PPO ${PPO_START_COEFF}->${PPO_TARGET_COEFF}, step=${PPO_SCHEDULE_STEP_EPOCHS}, end=${DAGGER_END_EPOCH}; effective BC weight is 1-PPO. Visual meshes remain real source meshes for depth rendering; collision meshes use convex hulls. Clips are final-position successes whose retained contact points are all within 1cm of the convex hull surface."}
LOG_DIR="logs/batch_ne/${SESSION}_${RUN_STAMP}"

quote() {
  printf '%q' "$1"
}

remote_run() {
  local node="$1"
  local cmd="$2"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN] ssh %s %s\n' "${node}" "${cmd}"
    return 0
  fi
  # shellcheck disable=SC2086
  ssh ${SSH_OPTS} "${node}" "${cmd}"
}

prepare_node() {
  local node="$1"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
cd $(quote "${REMOTE_REPO}")
echo "[INFO][${node}] repo=\$(pwd)"
git fetch $(quote "${GIT_REMOTE}")
git pull --ff-only $(quote "${GIT_REMOTE}") $(quote "${GIT_BRANCH}")
chmod +x cp_ch.sh batch_ne.sh || true
PULL_CODE=0 CH_BANK_NAME=$(quote "${CH_BANK_NAME}") NFS_CH_BANK=$(quote "${NFS_CH_BANK}") bash cp_ch.sh
EOF
)
  remote_run "${node}" "${cmd}"
}

launch_node() {
  local node="$1"
  local node_rank="$2"
  local log_file="${REMOTE_REPO}/${LOG_DIR}/node_${node_rank}_${node}.log"
  local env_exports
  env_exports=$(cat <<EOF
export CUDA_VISIBLE_DEVICES=$(quote "${CUDA_VISIBLE_DEVICES}")
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
export ACCEPT_EULA=${ACCEPT_EULA:-Y}
export NPROC=$(quote "${NPROC}")
export NNODES=$(quote "${NNODES}")
export NODE_RANK=$(quote "${node_rank}")
export MASTER_ADDR=$(quote "${MASTER_ADDR}")
export MASTER_PORT=$(quote "${MASTER_PORT}")
export NCCL_SOCKET_IFNAME=$(quote "${NCCL_SOCKET_IFNAME}")
export GLOO_SOCKET_IFNAME=$(quote "${GLOO_SOCKET_IFNAME}")
export NCCL_IB_DISABLE=$(quote "${NCCL_IB_DISABLE}")
export NCCL_DEBUG=$(quote "${NCCL_DEBUG}")
export TORCH_DIST_BACKEND=$(quote "${TORCH_DIST_BACKEND}")
export TORCH_NCCL_ASYNC_ERROR_HANDLING=$(quote "${TORCH_NCCL_ASYNC_ERROR_HANDLING}")
export TORCH_NCCL_ENABLE_MONITORING=$(quote "${TORCH_NCCL_ENABLE_MONITORING}")
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=$(quote "${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}")
export TORCH_NCCL_DUMP_ON_TIMEOUT=$(quote "${TORCH_NCCL_DUMP_ON_TIMEOUT}")
export TORCH_NCCL_TRACE_BUFFER_SIZE=$(quote "${TORCH_NCCL_TRACE_BUFFER_SIZE}")
export TORCH_NCCL_PROPAGATE_ERROR=$(quote "${TORCH_NCCL_PROPAGATE_ERROR}")
export TORCH_NCCL_DESYNC_DEBUG=$(quote "${TORCH_NCCL_DESYNC_DEBUG}")
export TORCH_NCCL_ENABLE_TIMING=$(quote "${TORCH_NCCL_ENABLE_TIMING}")
export TORCH_NCCL_BLOCKING_WAIT=$(quote "${TORCH_NCCL_BLOCKING_WAIT}")
export NCCL_SOCKET_FAMILY=$(quote "${NCCL_SOCKET_FAMILY}")
export NCCL_SOCKET_RETRY_CNT=$(quote "${NCCL_SOCKET_RETRY_CNT}")
export NCCL_SOCKET_RETRY_SLEEP_MSEC=$(quote "${NCCL_SOCKET_RETRY_SLEEP_MSEC}")
export NCCL_SOCKET_NTHREADS=$(quote "${NCCL_SOCKET_NTHREADS}")
export NCCL_NSOCKS_PERTHREAD=$(quote "${NCCL_NSOCKS_PERTHREAD}")
export NCCL_LIB_DIR=$(quote "${NCCL_LIB_DIR}")
export LD_LIBRARY_PATH=$(quote "${NCCL_LIB_DIR}")\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}
export PER_GPU_ENVS=$(quote "${PER_GPU_ENVS}")
export CH_BANK_NAME=$(quote "${CH_BANK_NAME}")
export CORL_SOLID80_BANK_NAME=$(quote "${CH_BANK_NAME}")
export AS_SUCCESS133_FINAL0P5=1
export AS_RANK_LOCAL_SHARDS=1
export RESUME_FROM_BOX=$(quote "${RESUME_FROM_BOX}")
export RESUME_CKPT=$(quote "${RESUME_TRAINING_CKPT}")
export BOX_POLICY_INIT_REF=$(quote "${BOX_POLICY_INIT_REF}")
export BOX_POLICY_INIT_CACHE_ROOT=$(quote "${BOX_POLICY_INIT_CACHE_ROOT}")
export OMOMO_EXPECTED_TOTAL=51
export RESUME_FROM_BOX_EXPECTED_TOTAL=51
export STUDENT_ACTOR_HIDDEN_DIMS=$(quote "${STUDENT_ACTOR_HIDDEN_DIMS}")
export RUN_NAME=$(quote "${RUN_NAME}")
export TRAINING_NAME=$(quote "${TRAINING_NAME}")
export TRAINING_PROJECT=carry-any
export WANDB_PROJECT=carry-any
export WANDB_ENTITY=$(quote "${WANDB_ENTITY}")
export RESUME_WANDB_RUN_ID=$(quote "${RESUME_WANDB_RUN_ID}")
export WANDB_RESUME_MODE=$(quote "${WANDB_RESUME_MODE}")
export SCHEDULE_NAME=$(quote "${SCHEDULE_NAME}")
export SCHEDULE_NOTES=$(quote "${SCHEDULE_NOTES}")
export NUM_LEARNING_ITERATIONS=$(quote "${NUM_LEARNING_ITERATIONS}")
export NUM_MINI_BATCHES=$(quote "${NUM_MINI_BATCHES}")
export NUM_LEARNING_EPOCHS=$(quote "${NUM_LEARNING_EPOCHS}")
export SAVE_INTERVAL=$(quote "${SAVE_INTERVAL}")
export PPO_START_EPOCH=$(quote "${PPO_START_EPOCH}")
export DAGGER_END_EPOCH=$(quote "${DAGGER_END_EPOCH}")
export PPO_START_COEFF=$(quote "${PPO_START_COEFF}")
export PPO_TARGET_COEFF=$(quote "${PPO_TARGET_COEFF}")
export PPO_SCHEDULE_STEP_EPOCHS=$(quote "${PPO_SCHEDULE_STEP_EPOCHS}")
export DAGGER_LOSS_COEF=$(quote "${DAGGER_LOSS_COEF}")
export HOLOSOMA_SKIP_INITIAL_CHECKPOINT=$(quote "${HOLOSOMA_SKIP_INITIAL_CHECKPOINT}")
export HOLOSOMA_SKIP_GRAD_FINITE_CHECK=$(quote "${HOLOSOMA_SKIP_GRAD_FINITE_CHECK}")
export HOLOSOMA_SKIP_LOSS_FINITE_CHECK=$(quote "${HOLOSOMA_SKIP_LOSS_FINITE_CHECK}")
export HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION=$(quote "${HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION}")
export HOLOSOMA_DEBUG_HEARTBEAT=$(quote "${HOLOSOMA_DEBUG_HEARTBEAT}")
export HOLOSOMA_DEBUG_ACTOR=$(quote "${HOLOSOMA_DEBUG_ACTOR}")
export HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE=$(quote "${HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE}")
export HOLOSOMA_SYNC_AFTER_OPTIMIZER_STEP=$(quote "${HOLOSOMA_SYNC_AFTER_OPTIMIZER_STEP}")
export HOLOSOMA_SYNC_AFTER_MICROBATCH_FORWARD=$(quote "${HOLOSOMA_SYNC_AFTER_MICROBATCH_FORWARD}")
export HOLOSOMA_DEBUG_MICROBATCH_ALL=$(quote "${HOLOSOMA_DEBUG_MICROBATCH_ALL}")
export HOLOSOMA_GLOO_GRAD_REDUCE=$(quote "${HOLOSOMA_GLOO_GRAD_REDUCE}")
export HOLOSOMA_GLOO_BARRIER=$(quote "${HOLOSOMA_GLOO_BARRIER}")
export HOLOSOMA_GLOO_SMALL_COLLECTIVES=$(quote "${HOLOSOMA_GLOO_SMALL_COLLECTIVES}")
export HOLOSOMA_CONTIGUOUS_MINIBATCHES=$(quote "${HOLOSOMA_CONTIGUOUS_MINIBATCHES}")
export HOLOSOMA_DAGGER_SUPERVISED_ONLY=$(quote "${HOLOSOMA_DAGGER_SUPERVISED_ONLY}")
export HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP=$(quote "${HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP}")
export HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH=$(quote "${HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH}")
export HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD=$(quote "${HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD}")
export HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC=$(quote "${HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC}")
export HOLOSOMA_SKIP_WANDB_CHECKPOINT_UPLOAD=$(quote "${HOLOSOMA_SKIP_WANDB_CHECKPOINT_UPLOAD}")
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=$(quote "${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE}")
export EXPORT_ONNX=False
export WANDB_RESUME_SAME_RUN=$(quote "${WANDB_RESUME_SAME_RUN}")
export OBJECT_GEOMETRY_MODE=mesh
export HOLOSOMA_OBJECT_SPAWN_MODE=single_slot_multi_urdf
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=1
export TORCH_DIST_TIMEOUT_SEC=3600
export MAX_RESTARTS=0
export PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY:-301989888}
export PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY=${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY:-301989888}
export PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY:-134217728}
export PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-268435456}
EOF
)
  local train_cmd
  train_cmd=$(cat <<EOF
set -euo pipefail
cd $(quote "${REMOTE_REPO}")
mkdir -p $(quote "${REMOTE_REPO}/${LOG_DIR}")
${env_exports}
source ./scripts/gpu_launch_defaults.sh
if ! ip link show "\${NCCL_SOCKET_IFNAME}" >/dev/null 2>&1; then
  echo "[ERROR][${node}] NCCL_SOCKET_IFNAME=\${NCCL_SOCKET_IFNAME} does not exist on this node." >&2
  ip -o link show >&2 || true
  exit 2
fi
if [[ "\${RESUME_FROM_BOX}" == "1" ]]; then
  BOX_RESUME_CKPT="\$("\${PYTHON_BIN}" - "\${BOX_POLICY_INIT_REF}" "\${BOX_POLICY_INIT_CACHE_ROOT}" <<'PY'
from __future__ import annotations

import contextlib
import re
import sys
from pathlib import Path

import torch


def parse_wandb_ref(ref: str) -> tuple[str, str]:
    if ref.startswith("https://wandb.ai/"):
        clean = ref.split("?", 1)[0]
        parts = clean.removeprefix("https://wandb.ai/").split("/")
        if len(parts) < 6 or parts[2] != "runs" or parts[4] != "files":
            raise SystemExit(f"[ERROR] W&B checkpoint URL must include /runs/<id>/files/<model.pt>: {ref}")
        entity, project, run_id = parts[0], parts[1], parts[3]
        file_name = "/".join(parts[5:])
        return f"{entity}/{project}/{run_id}", file_name

    if ref.startswith("wandb://"):
        parts = ref.removeprefix("wandb://").split("/")
        if len(parts) >= 5 and parts[2] == "runs":
            entity, project, run_id = parts[0], parts[1], parts[3]
            file_name = "/".join(parts[4:])
        elif len(parts) >= 4:
            entity, project, run_id = parts[0], parts[1], parts[2]
            file_name = "/".join(parts[3:])
        else:
            raise SystemExit(f"[ERROR] W&B checkpoint URI must include a model .pt file: {ref}")
        return f"{entity}/{project}/{run_id}", file_name

    raise ValueError("not a W&B reference")


def validate_checkpoint(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"checkpoint validation failed for {path}: {exc}") from exc


ref = sys.argv[1]
cache_root = Path(sys.argv[2]).expanduser().resolve()

try:
    run_path, file_name = parse_wandb_ref(ref)
except ValueError:
    local_path = Path(ref).expanduser().resolve()
    validate_checkpoint(local_path)
    print(local_path)
    raise SystemExit(0)

if not file_name.endswith(".pt"):
    raise SystemExit(f"[ERROR] Expected a .pt checkpoint, got: {file_name}")

safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_path)
cache_dir = cache_root / safe_run
cache_dir.mkdir(parents=True, exist_ok=True)
target = cache_dir / Path(file_name).name

try:
    validate_checkpoint(target)
except Exception:
    target.unlink(missing_ok=True)
    import wandb

    api = wandb.Api(timeout=60)
    run = api.run(run_path)
    with contextlib.redirect_stdout(sys.stderr):
        downloaded = run.file(file_name).download(root=str(cache_dir), replace=True)
    downloaded_path = Path(downloaded.name)
    if not downloaded_path.is_absolute():
        downloaded_path = (Path.cwd() / downloaded_path).resolve()
    else:
        downloaded_path = downloaded_path.resolve()
    if downloaded_path != target:
        target.unlink(missing_ok=True)
        downloaded_path.replace(target)
    validate_checkpoint(target)

print(target)
PY
)"
  export BOX_RESUME_CKPT
  export RESUME_FROM_BOX_CKPT="\${BOX_RESUME_CKPT}"
  export POLICY_INIT_CKPT="\${BOX_RESUME_CKPT}"
  echo "[INFO][${node}] box_policy_init_checkpoint_local=\${BOX_RESUME_CKPT}"
fi
echo "[INFO][${node}] session=${SESSION} node_rank=${node_rank}/${NNODES} per_gpu_envs=${PER_GPU_ENVS} total_num_envs=${TOTAL_NUM_ENVS}"
echo "[INFO][${node}] master=${MASTER_ADDR}:${MASTER_PORT} log=${log_file}"
if [[ -n "\${RESUME_CKPT}" ]]; then
  echo "[INFO][${node}] resume_training_checkpoint=\${RESUME_CKPT}"
fi
if [[ -n "\${RESUME_WANDB_RUN_ID}" ]]; then
  echo "[INFO][${node}] wandb_same_run_resume=\${WANDB_ENTITY}/carry-any/\${RESUME_WANDB_RUN_ID} mode=\${WANDB_RESUME_MODE}"
fi
echo "[INFO][${node}] actor_hidden_dims=\${STUDENT_ACTOR_HIDDEN_DIMS}"
echo "[INFO][${node}] num_mini_batches=\${NUM_MINI_BATCHES}"
echo "[INFO][${node}] num_learning_epochs=\${NUM_LEARNING_EPOCHS}"
echo "[INFO][${node}] ppo_schedule=\${PPO_START_EPOCH}->\${DAGGER_END_EPOCH} start=\${PPO_START_COEFF} target=\${PPO_TARGET_COEFF} step_epochs=\${PPO_SCHEDULE_STEP_EPOCHS} dagger_loss_coef=\${DAGGER_LOSS_COEF}"
echo "[INFO][${node}] dagger_supervised_only=\${HOLOSOMA_DAGGER_SUPERVISED_ONLY}"
echo "[INFO][${node}] supervised_actor_microbatch=\${HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH}"
echo "[INFO][${node}] skip_critic_weight_sync=\${HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC}"
echo "[INFO][${node}] torch_dist_backend=\${TORCH_DIST_BACKEND}"
echo "[INFO][${node}] gloo_grad_reduce=\${HOLOSOMA_GLOO_GRAD_REDUCE}"
echo "[INFO][${node}] gloo_barrier=\${HOLOSOMA_GLOO_BARRIER}"
echo "[INFO][${node}] gloo_small_collectives=\${HOLOSOMA_GLOO_SMALL_COLLECTIVES}"
echo "[INFO][${node}] skip_wandb_checkpoint_upload=\${HOLOSOMA_SKIP_WANDB_CHECKPOINT_UPLOAD}"
echo "[INFO][${node}] nccl_if=\${NCCL_SOCKET_IFNAME} gloo_if=\${GLOO_SOCKET_IFNAME} nccl_ib_disable=\${NCCL_IB_DISABLE} socket_family=\${NCCL_SOCKET_FAMILY}"
echo "[INFO][${node}] nccl_retry_cnt=\${NCCL_SOCKET_RETRY_CNT} retry_sleep_msec=\${NCCL_SOCKET_RETRY_SLEEP_MSEC} socket_nthreads=\${NCCL_SOCKET_NTHREADS} nsocks_perthread=\${NCCL_NSOCKS_PERTHREAD}"
echo "[INFO][${node}] torch_nccl_async=\${TORCH_NCCL_ASYNC_ERROR_HANDLING} monitoring=\${TORCH_NCCL_ENABLE_MONITORING} heartbeat_sec=\${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC} dump_on_timeout=\${TORCH_NCCL_DUMP_ON_TIMEOUT} trace_buffer=\${TORCH_NCCL_TRACE_BUFFER_SIZE} propagate_error=\${TORCH_NCCL_PROPAGATE_ERROR} desync_debug=\${TORCH_NCCL_DESYNC_DEBUG} enable_timing=\${TORCH_NCCL_ENABLE_TIMING}"
echo "[INFO][${node}] nccl_lib_dir=\${NCCL_LIB_DIR}"
"\${PYTHON_BIN}" - <<'PY'
import ctypes.util
import os
import socket

import torch

try:
    import torch.cuda.nccl as nccl
    nccl_version = nccl.version()
except Exception as exc:
    nccl_version = f"unavailable:{type(exc).__name__}:{exc}"

print(
    "[INFO][nccl-preflight] "
    f"host={socket.gethostname()} "
    f"torch={torch.__version__} "
    f"torch_cuda={torch.version.cuda} "
    f"torch_nccl={nccl_version} "
    f"cuda_available={torch.cuda.is_available()} "
    f"cuda_device_count={torch.cuda.device_count()} "
    f"libnccl_find={ctypes.util.find_library('nccl')} "
    f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}"
)
PY
TRAIN_EXTRA_ARGS=()
if [[ -n "\${RESUME_WANDB_RUN_ID}" ]]; then
  if [[ -n "\${WANDB_ENTITY}" ]]; then
    TRAIN_EXTRA_ARGS+=(--logger.entity="\${WANDB_ENTITY}")
  fi
  TRAIN_EXTRA_ARGS+=(--logger.id="\${RESUME_WANDB_RUN_ID}" --logger.resume="\${WANDB_RESUME_MODE}")
fi
exec bash distill_as_button_solid.sh "\${TRAIN_EXTRA_ARGS[@]}" 2>&1 | tee $(quote "${log_file}")
EOF
)
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
cd $(quote "${REMOTE_REPO}")
mkdir -p $(quote "${REMOTE_REPO}/${LOG_DIR}")
if tmux has-session -t $(quote "${SESSION}") 2>/dev/null; then
  if [[ $(quote "${RESTART}") == "1" ]]; then
    tmux kill-session -t $(quote "${SESSION}")
  else
    echo "[ERROR][${node}] tmux session already exists: ${SESSION}" >&2
    exit 1
  fi
fi
tmux new-session -d -s $(quote "${SESSION}") $(quote "${train_cmd}")
tmux display-message -p -t $(quote "${SESSION}") "[INFO][${node}] started #{session_name}"
EOF
)
  remote_run "${node}" "${cmd}"
}

status_node() {
  local node="$1"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
cd $(quote "${REMOTE_REPO}")
echo "===== ${node} ====="
tmux has-session -t $(quote "${SESSION}") 2>/dev/null && tmux list-sessions | grep -F $(quote "${SESSION}") || echo "tmux:${SESSION}:not-running"
latest_log=\$(ls -1t logs/batch_ne/${SESSION}_*/node_*_${node}.log 2>/dev/null | head -1 || true)
if [[ -n "\${latest_log}" ]]; then
  echo "log=\${latest_log}"
  tail -40 "\${latest_log}"
fi
EOF
)
  remote_run "${node}" "${cmd}"
}

stop_node() {
  local node="$1"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
if tmux has-session -t $(quote "${SESSION}") 2>/dev/null; then
  tmux kill-session -t $(quote "${SESSION}")
  echo "[INFO][${node}] stopped ${SESSION}"
else
  echo "[INFO][${node}] ${SESSION} not running"
fi
EOF
)
  remote_run "${node}" "${cmd}"
}

run_prepare() {
  local pids=()
  local failed=0
  for node in "${NODE_LIST[@]}"; do
    echo "[INFO] Preparing ${node}"
    prepare_node "${node}" &
    pids+=("$!")
  done
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed != 0 )); then
    echo "[ERROR] One or more nodes failed during prepare." >&2
    exit 1
  fi
}

run_launch() {
  echo "[INFO] Launching ${NNODES} nodes x ${NPROC} GPUs, PER_GPU_ENVS=${PER_GPU_ENVS}, TOTAL_NUM_ENVS=${TOTAL_NUM_ENVS}"
  local rank=0
  for node in "${NODE_LIST[@]}"; do
    echo "[INFO] Launching ${node} node_rank=${rank}"
    launch_node "${node}" "${rank}"
    rank=$((rank + 1))
  done
}

run_status() {
  for node in "${NODE_LIST[@]}"; do
    status_node "${node}" || true
  done
}

run_stop() {
  for node in "${NODE_LIST[@]}"; do
    stop_node "${node}" || true
  done
}

case "${ACTION}" in
  prepare)
    run_prepare
    ;;
  launch)
    run_launch
    ;;
  all)
    run_prepare
    run_launch
    ;;
  status)
    run_status
    ;;
  stop)
    run_stop
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
