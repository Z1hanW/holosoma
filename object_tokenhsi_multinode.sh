#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${SCRIPT_DIR}"

quote() {
  printf "%q" "$1"
}

find_remote_free_port() {
  local host="$1"
  local start="${2:-29800}"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${host}" "python3 - <<PY
import socket
start = int(${start})
for port in range(start, start + 200):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('', port))
        except OSError:
            continue
        print(port)
        raise SystemExit(0)
raise SystemExit('No free port found')
PY"
}

DEFAULT_NODE_HOSTS="10.0.100.200 10.0.72.226 10.0.90.122 10.0.123.134"
NODE_HOSTS="${NODE_HOSTS:-${DEFAULT_NODE_HOSTS}}"
read -r -a NODE_HOST_ARRAY <<< "${NODE_HOSTS}"
if [[ "${#NODE_HOST_ARRAY[@]}" -lt 1 ]]; then
  echo "NODE_HOSTS is empty." >&2
  exit 1
fi

NNODES="${NNODES:-${#NODE_HOST_ARRAY[@]}}"
if [[ "${NNODES}" != "${#NODE_HOST_ARRAY[@]}" ]]; then
  echo "NNODES=${NNODES} does not match NODE_HOSTS count ${#NODE_HOST_ARRAY[@]}." >&2
  exit 1
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
ENVS_PER_GPU="${ENVS_PER_GPU:-256}"
TOTAL_GPUS="${TOTAL_GPUS:-$((NNODES * GPUS_PER_NODE))}"
TOTAL_ENVS="${TOTAL_ENVS:-$((TOTAL_GPUS * ENVS_PER_GPU))}"

TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
MASTER_ADDR="${MASTER_ADDR:-${NODE_HOST_ARRAY[0]}}"
MASTER_PORT="${MASTER_PORT:-$(find_remote_free_port "${MASTER_ADDR}" 29800)}"

WANDB_ENTITY="${WANDB_ENTITY:-zihanw22}"
WANDB_PROJECT="${WANDB_PROJECT:-holosomatest}"
WANDB_MODE="${WANDB_MODE:-online}"
EXP_NAME="${EXP_NAME:-g1-29dof-wbt-w-object-height-scan-tokenhsi}"
NUM_ITERATIONS="${NUM_ITERATIONS:-40000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
NUM_STEPS_PER_ENV="${NUM_STEPS_PER_ENV:-24}"
NUM_LEARNING_EPOCHS="${NUM_LEARNING_EPOCHS:-7}"
NUM_MINI_BATCHES="${NUM_MINI_BATCHES:-4}"
TRAINING_SEED="${TRAINING_SEED:-42}"
PHYSX_GPU_COLLISION_STACK_SIZE="${PHYSX_GPU_COLLISION_STACK_SIZE:-536870912}"

REMOTE_REPO="${REMOTE_REPO:-/home/ubuntu/FAR/holosoma_object_tokenhsi_20260704_211351}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hssim}"
SYNC_SCRIPT="${SYNC_SCRIPT:-1}"
KILL_EXISTING="${KILL_EXISTING:-0}"

MOTION_DIR="${MOTION_DIR:-data/ds_as_data/debug/_single_slot_motion_bank}"
OBJECT_URDF_MAP="${OBJECT_URDF_MAP:-${MOTION_DIR}/_clip_object_urdf_map.json}"

RUN_NAME="${RUN_NAME:-debug39_object_tokenhsi_multinode4x8_${TOTAL_GPUS}gpu_${ENVS_PER_GPU}envpergpu_${TOTAL_ENVS}env_${NUM_ITERATIONS}iter_${TIMESTAMP}}"
LOGGER_GROUP="${LOGGER_GROUP:-debug39-object-tokenhsi-multinode4x8-${TIMESTAMP}}"
SESSION="${SESSION:-object_tokenhsi_multinode_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-logs/run_commands}"
SCRIPT_BASENAME="$(basename "${BASH_SOURCE[0]}")"

if [[ "${1:-}" == "--node-run" ]]; then
  shift
  NODE_RANK="${NODE_RANK:?NODE_RANK must be set for --node-run}"

  if [[ ! -d "${MOTION_DIR}" ]]; then
    echo "Missing motion dir: ${MOTION_DIR}" >&2
    exit 1
  fi
  if [[ ! -f "${OBJECT_URDF_MAP}" ]]; then
    echo "Missing object URDF map: ${OBJECT_URDF_MAP}" >&2
    exit 1
  fi

  source /home/ubuntu/.holosoma_deps/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null \
    || source /home/ubuntu/.holosoma_deps/miniconda3/etc/profile.d/conda.sh

  conda activate "${CONDA_ENV_NAME}"
  source scripts/source_isaacsim_setup.sh

  export PYTHONPATH="${REMOTE_REPO}/src/holosoma:${PYTHONPATH:-}"
  export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"
  export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

  torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    src/holosoma/holosoma/train_agent.py \
    "exp:${EXP_NAME}" \
    logger:wandb \
    --logger.entity="${WANDB_ENTITY}" \
    --logger.project="${WANDB_PROJECT}" \
    --logger.name="${RUN_NAME}" \
    --logger.group="${LOGGER_GROUP}" \
    --logger.mode="${WANDB_MODE}" \
    --logger.video.enabled=False \
    --training.multigpu=True \
    --training.project="${WANDB_PROJECT}" \
    --training.name="${RUN_NAME}" \
    --training.seed="${TRAINING_SEED}" \
    --training.num-envs="${TOTAL_ENVS}" \
    --training.export-onnx=False \
    --command.setup_terms.motion_command.params.motion_config.motion_dir="${MOTION_DIR}" \
    --robot.object.object-urdf-path="${OBJECT_URDF_MAP}" \
    --simulator.config.sim.physx.gpu-collision-stack-size="${PHYSX_GPU_COLLISION_STACK_SIZE}" \
    --algo.config.num-learning-iterations="${NUM_ITERATIONS}" \
    --algo.config.num-steps-per-env="${NUM_STEPS_PER_ENV}" \
    --algo.config.num-learning-epochs="${NUM_LEARNING_EPOCHS}" \
    --algo.config.num-mini-batches="${NUM_MINI_BATCHES}" \
    --algo.config.save-interval="${SAVE_INTERVAL}" \
    "$@"
  exit 0
fi

mkdir -p "${LOG_DIR}"
printf "%s\n" "${RUN_NAME}" > "${LOG_DIR}/${SESSION}.run_name"
printf "%s\n" "${LOGGER_GROUP}" > "${LOG_DIR}/${SESSION}.group"
printf "%s\n" "${REMOTE_REPO}" > "${LOG_DIR}/${SESSION}.remote_repo"
printf "%s\n" "${NODE_HOST_ARRAY[@]}" > "${LOG_DIR}/${SESSION}.nodes"

echo "Launching object-tokenhsi multi-node WBT training."
echo "  session: ${SESSION}"
echo "  run_name: ${RUN_NAME}"
echo "  group: ${LOGGER_GROUP}"
echo "  exp: ${EXP_NAME}"
echo "  nodes: ${NODE_HOST_ARRAY[*]}"
echo "  master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  total_envs: ${TOTAL_ENVS} (${NNODES} nodes x ${GPUS_PER_NODE} GPUs x ${ENVS_PER_GPU} envs/GPU)"
echo "  remote_repo: ${REMOTE_REPO}"

for node_rank in "${!NODE_HOST_ARRAY[@]}"; do
  host="${NODE_HOST_ARRAY[$node_rank]}"
  remote_log="${REMOTE_REPO}/${LOG_DIR}/${SESSION}_node${node_rank}_${host//./-}.log"

  if [[ "${SYNC_SCRIPT}" == "1" ]]; then
    rsync -az "${SCRIPT_DIR}/${SCRIPT_BASENAME}" "${host}:${REMOTE_REPO}/${SCRIPT_BASENAME}"
  fi

  if ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${host}" \
    "tmux has-session -t $(quote "${SESSION}") 2>/dev/null"; then
    if [[ "${KILL_EXISTING}" == "1" ]]; then
      ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${host}" \
        "tmux kill-session -t $(quote "${SESSION}")"
    else
      echo "Remote tmux session already exists on ${host}: ${SESSION}" >&2
      exit 1
    fi
  fi

  remote_env="NODE_RANK=$(quote "${node_rank}") NODE_HOSTS=$(quote "${NODE_HOSTS}") NNODES=$(quote "${NNODES}") GPUS_PER_NODE=$(quote "${GPUS_PER_NODE}") ENVS_PER_GPU=$(quote "${ENVS_PER_GPU}") TOTAL_GPUS=$(quote "${TOTAL_GPUS}") TOTAL_ENVS=$(quote "${TOTAL_ENVS}") MASTER_ADDR=$(quote "${MASTER_ADDR}") MASTER_PORT=$(quote "${MASTER_PORT}") TIMESTAMP=$(quote "${TIMESTAMP}") WANDB_ENTITY=$(quote "${WANDB_ENTITY}") WANDB_PROJECT=$(quote "${WANDB_PROJECT}") WANDB_MODE=$(quote "${WANDB_MODE}") EXP_NAME=$(quote "${EXP_NAME}") NUM_ITERATIONS=$(quote "${NUM_ITERATIONS}") SAVE_INTERVAL=$(quote "${SAVE_INTERVAL}") NUM_STEPS_PER_ENV=$(quote "${NUM_STEPS_PER_ENV}") NUM_LEARNING_EPOCHS=$(quote "${NUM_LEARNING_EPOCHS}") NUM_MINI_BATCHES=$(quote "${NUM_MINI_BATCHES}") TRAINING_SEED=$(quote "${TRAINING_SEED}") PHYSX_GPU_COLLISION_STACK_SIZE=$(quote "${PHYSX_GPU_COLLISION_STACK_SIZE}") REMOTE_REPO=$(quote "${REMOTE_REPO}") CONDA_ENV_NAME=$(quote "${CONDA_ENV_NAME}") MOTION_DIR=$(quote "${MOTION_DIR}") OBJECT_URDF_MAP=$(quote "${OBJECT_URDF_MAP}") RUN_NAME=$(quote "${RUN_NAME}") LOGGER_GROUP=$(quote "${LOGGER_GROUP}") SESSION=$(quote "${SESSION}") LOG_DIR=$(quote "${LOG_DIR}")"

  remote_cmd="cd $(quote "${REMOTE_REPO}") && mkdir -p $(quote "${LOG_DIR}") && env ${remote_env} bash $(quote "${REMOTE_REPO}/${SCRIPT_BASENAME}") --node-run > $(quote "${remote_log}") 2>&1"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${host}" \
    "tmux new-session -d -s $(quote "${SESSION}") $(quote "${remote_cmd}")"
  echo "  node_rank ${node_rank}: ${host} -> ${remote_log}"
done
