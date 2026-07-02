#!/usr/bin/env bash
set -euo pipefail

# Multi-node depth-student distillation on remote nodes only.
# Default topology:
#   2 remote nodes x 8 GPUs x 1024 envs/GPU = 16384 total envs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${SCRIPT_DIR}"

quote() {
  printf "%q" "$1"
}

find_remote_free_port() {
  local host="$1"
  local start="${2:-29660}"
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

DEFAULT_TEACHER_CHECKPOINT="logs/holosomatest/20260629_043623-ip-10-0-73-59_g1_29dof_wbt_slope_climbing_8gpu_4096env_20260629_043601-locomotion/model_20000.pt"
DEFAULT_NODE_HOSTS="10.0.74.86 10.0.100.200"
NODE_HOSTS="${NODE_HOSTS:-${DEFAULT_NODE_HOSTS}}"
read -r -a NODE_HOST_ARRAY <<< "${NODE_HOSTS}"
if [[ "${#NODE_HOST_ARRAY[@]}" -lt 1 ]]; then
  echo "NODE_HOSTS is empty." >&2
  exit 1
fi

NNODES="${NNODES:-${#NODE_HOST_ARRAY[@]}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
ENVS_PER_GPU="${ENVS_PER_GPU:-1024}"
TOTAL_GPUS="${TOTAL_GPUS:-$((NNODES * GPUS_PER_NODE))}"
TOTAL_ENVS="${TOTAL_ENVS:-$((TOTAL_GPUS * ENVS_PER_GPU))}"

HOSTNAME_SHORT="${HOSTNAME_SHORT:-$(hostname)}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
MASTER_ADDR="${MASTER_ADDR:-${NODE_HOST_ARRAY[0]}}"
MASTER_PORT="${MASTER_PORT:-$(find_remote_free_port "${MASTER_ADDR}" 29660)}"
MASTER_LABEL="${MASTER_ADDR//./-}"

WANDB_ENTITY="${WANDB_ENTITY:-zihanw22}"
WANDB_PROJECT="${WANDB_PROJECT:-holosomatest}"
WANDB_MODE="${WANDB_MODE:-online}"
NUM_ITERATIONS="${NUM_ITERATIONS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
LOGGING_INTERVAL="${LOGGING_INTERVAL:-25}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
STUDENT_ROLLOUT_PROB="${STUDENT_ROLLOUT_PROB:-0.0}"
DEPTH_HEIGHT="${DEPTH_HEIGHT:-58}"
DEPTH_WIDTH="${DEPTH_WIDTH:-87}"
RAW_DEPTH_HEIGHT="${RAW_DEPTH_HEIGHT:-60}"
RAW_DEPTH_WIDTH="${RAW_DEPTH_WIDTH:-106}"
DEPTH_MIN_RANGE="${DEPTH_MIN_RANGE:-0.3}"
DEPTH_MAX_RANGE="${DEPTH_MAX_RANGE:-2.0}"
DEPTH_HORIZONTAL_FOV_DEG="${DEPTH_HORIZONTAL_FOV_DEG:-101.41}"
DEPTH_CAMERA_BODY_NAME="${DEPTH_CAMERA_BODY_NAME:-torso_link}"
DEPTH_CAMERA_DEBUG_VIS="${DEPTH_CAMERA_DEBUG_VIS:-0}"
DISTILL_TAG="${DISTILL_TAG:-slope}"
LOG_BASE_DIR="${LOG_BASE_DIR:-logs}"

GIT_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo nogit)"
REMOTE_REPO="${REMOTE_REPO:-/home/ubuntu/FAR/holosoma_distill_${GIT_SHA}}"
SYNC_REPO="${SYNC_REPO:-1}"
SYNC_TEACHER_CHECKPOINT="${SYNC_TEACHER_CHECKPOINT:-1}"
KILL_EXISTING="${KILL_EXISTING:-0}"

TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${1:-${DEFAULT_TEACHER_CHECKPOINT}}}"
if [[ "${TEACHER_CHECKPOINT}" == wandb://* ]]; then
  REMOTE_TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT}"
else
  if [[ ! -f "${TEACHER_CHECKPOINT}" ]]; then
    echo "Missing local teacher checkpoint: ${TEACHER_CHECKPOINT}" >&2
    exit 1
  fi
  REMOTE_TEACHER_CHECKPOINT="${REMOTE_TEACHER_CHECKPOINT:-${REMOTE_REPO}/logs/teacher_checkpoints/$(basename "${TEACHER_CHECKPOINT}")}"
fi

RUN_NAME="${RUN_NAME:-${HOSTNAME_SHORT}_g1_29dof_depth_student_distill_${DISTILL_TAG}_multinode${NNODES}x${GPUS_PER_NODE}_${TOTAL_GPUS}gpu_${ENVS_PER_GPU}env_master${MASTER_LABEL}_${TIMESTAMP}}"
SESSION="${SESSION:-csp_multinode_depth_distill_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-logs/run_commands}"
SCRIPT_BASENAME="$(basename "${BASH_SOURCE[0]}")"

if [[ "${1:-}" == "--node-run" ]]; then
  shift
  NODE_RANK="${NODE_RANK:?NODE_RANK must be set for --node-run}"

  source /home/ubuntu/.holosoma_deps/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null \
    || source /home/ubuntu/.holosoma_deps/miniconda3/etc/profile.d/conda.sh

  conda activate "${CONDA_ENV_NAME:-hssim}"
  source scripts/source_isaacsim_setup.sh

  export PYTHONPATH="${REMOTE_REPO}/src/holosoma:${PYTHONPATH:-}"
  export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

  DEPTH_CAMERA_FLAGS=()
  if [[ "${DEPTH_CAMERA_DEBUG_VIS,,}" == "true" || "${DEPTH_CAMERA_DEBUG_VIS}" == "1" ]]; then
    DEPTH_CAMERA_FLAGS+=(--depth-camera-debug-vis)
  fi

  torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    src/holosoma/holosoma/distill_depth_student.py \
    --teacher-checkpoint "${REMOTE_TEACHER_CHECKPOINT}" \
    --num-envs "${TOTAL_ENVS}" \
    --iterations "${NUM_ITERATIONS}" \
    --save-interval "${SAVE_INTERVAL}" \
    --logging-interval "${LOGGING_INTERVAL}" \
    --learning-rate "${LEARNING_RATE}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --max-grad-norm "${MAX_GRAD_NORM}" \
    --student-rollout-prob "${STUDENT_ROLLOUT_PROB}" \
    --depth-height "${DEPTH_HEIGHT}" \
    --depth-width "${DEPTH_WIDTH}" \
    --raw-depth-height "${RAW_DEPTH_HEIGHT}" \
    --raw-depth-width "${RAW_DEPTH_WIDTH}" \
    --depth-min-range "${DEPTH_MIN_RANGE}" \
    --depth-max-range "${DEPTH_MAX_RANGE}" \
    --depth-horizontal-fov-deg "${DEPTH_HORIZONTAL_FOV_DEG}" \
    --depth-camera-body-name "${DEPTH_CAMERA_BODY_NAME}" \
    --run-name "${RUN_NAME}" \
    --project "${WANDB_PROJECT}" \
    --log-base-dir "${LOG_BASE_DIR}" \
    --wandb \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-mode "${WANDB_MODE}" \
    "${DEPTH_CAMERA_FLAGS[@]}" \
    "$@"
  exit 0
fi

if [[ "${1:-}" == "${TEACHER_CHECKPOINT}" ]]; then
  shift
fi

if ! git diff-index --quiet HEAD --; then
  echo "Working tree has uncommitted changes; commit or stash before syncing remote code." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
printf "%s\n" "${RUN_NAME}" > "${LOG_DIR}/${SESSION}.run_name"
printf "%s\n" "${NODE_HOST_ARRAY[@]}" > "${LOG_DIR}/${SESSION}.nodes"
printf "%s\n" "${REMOTE_REPO}" > "${LOG_DIR}/${SESSION}.remote_repo"

echo "Launching CSP multi-node depth student distillation."
echo "  session: ${SESSION}"
echo "  run_name: ${RUN_NAME}"
echo "  nodes: ${NODE_HOST_ARRAY[*]}"
echo "  master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  total_envs: ${TOTAL_ENVS} (${NNODES} nodes x ${GPUS_PER_NODE} GPUs x ${ENVS_PER_GPU} envs/GPU)"
echo "  teacher_checkpoint: ${TEACHER_CHECKPOINT}"
echo "  remote_teacher_checkpoint: ${REMOTE_TEACHER_CHECKPOINT}"
echo "  remote_repo: ${REMOTE_REPO}"
echo "  physics_rollout: true"

for node_rank in "${!NODE_HOST_ARRAY[@]}"; do
  host="${NODE_HOST_ARRAY[$node_rank]}"
  remote_log="${REMOTE_REPO}/${LOG_DIR}/${SESSION}_node${node_rank}_${host//./-}.log"

  if [[ "${SYNC_REPO}" == "1" ]]; then
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${host}" \
      "mkdir -p $(quote "${REMOTE_REPO}")"
    git archive HEAD | ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${host}" \
      "tar -x -C $(quote "${REMOTE_REPO}")"
  fi

  if [[ "${SYNC_TEACHER_CHECKPOINT}" == "1" && "${TEACHER_CHECKPOINT}" != wandb://* ]]; then
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${host}" \
      "mkdir -p $(quote "$(dirname "${REMOTE_TEACHER_CHECKPOINT}")")"
    rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10" \
      "${TEACHER_CHECKPOINT}" "${host}:$(quote "${REMOTE_TEACHER_CHECKPOINT}")"
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

  remote_env="NODE_RANK=$(quote "${node_rank}") NNODES=$(quote "${NNODES}") GPUS_PER_NODE=$(quote "${GPUS_PER_NODE}") ENVS_PER_GPU=$(quote "${ENVS_PER_GPU}") TOTAL_GPUS=$(quote "${TOTAL_GPUS}") TOTAL_ENVS=$(quote "${TOTAL_ENVS}") MASTER_ADDR=$(quote "${MASTER_ADDR}") MASTER_PORT=$(quote "${MASTER_PORT}") TIMESTAMP=$(quote "${TIMESTAMP}") HOSTNAME_SHORT=$(quote "${HOSTNAME_SHORT}") WANDB_ENTITY=$(quote "${WANDB_ENTITY}") WANDB_PROJECT=$(quote "${WANDB_PROJECT}") WANDB_MODE=$(quote "${WANDB_MODE}") NUM_ITERATIONS=$(quote "${NUM_ITERATIONS}") SAVE_INTERVAL=$(quote "${SAVE_INTERVAL}") LOGGING_INTERVAL=$(quote "${LOGGING_INTERVAL}") LEARNING_RATE=$(quote "${LEARNING_RATE}") WEIGHT_DECAY=$(quote "${WEIGHT_DECAY}") MAX_GRAD_NORM=$(quote "${MAX_GRAD_NORM}") STUDENT_ROLLOUT_PROB=$(quote "${STUDENT_ROLLOUT_PROB}") DEPTH_HEIGHT=$(quote "${DEPTH_HEIGHT}") DEPTH_WIDTH=$(quote "${DEPTH_WIDTH}") RAW_DEPTH_HEIGHT=$(quote "${RAW_DEPTH_HEIGHT}") RAW_DEPTH_WIDTH=$(quote "${RAW_DEPTH_WIDTH}") DEPTH_MIN_RANGE=$(quote "${DEPTH_MIN_RANGE}") DEPTH_MAX_RANGE=$(quote "${DEPTH_MAX_RANGE}") DEPTH_HORIZONTAL_FOV_DEG=$(quote "${DEPTH_HORIZONTAL_FOV_DEG}") DEPTH_CAMERA_BODY_NAME=$(quote "${DEPTH_CAMERA_BODY_NAME}") DEPTH_CAMERA_DEBUG_VIS=$(quote "${DEPTH_CAMERA_DEBUG_VIS}") DISTILL_TAG=$(quote "${DISTILL_TAG}") LOG_BASE_DIR=$(quote "${LOG_BASE_DIR}") REMOTE_REPO=$(quote "${REMOTE_REPO}") TEACHER_CHECKPOINT=$(quote "${REMOTE_TEACHER_CHECKPOINT}") REMOTE_TEACHER_CHECKPOINT=$(quote "${REMOTE_TEACHER_CHECKPOINT}") RUN_NAME=$(quote "${RUN_NAME}") SESSION=$(quote "${SESSION}") LOG_DIR=$(quote "${LOG_DIR}")"

  remote_cmd="cd $(quote "${REMOTE_REPO}") && mkdir -p $(quote "${LOG_DIR}") && env ${remote_env} bash $(quote "${REMOTE_REPO}/${SCRIPT_BASENAME}") --node-run > $(quote "${remote_log}") 2>&1"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${host}" \
    "tmux new-session -d -s $(quote "${SESSION}") $(quote "${remote_cmd}")"
  echo "  node_rank ${node_rank}: ${host} -> ${remote_log}"
done
