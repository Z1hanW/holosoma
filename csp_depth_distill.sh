#!/usr/bin/env bash
set -euo pipefail

# Distill a trained PPO/FastSAC tracking teacher into a depth-based student.
# Default rollout is far-tracking-style hybrid DAgger + PPO: student actions
# step the true IsaacSim/PhysX env, and teacher actions supervise DAgger.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${SCRIPT_DIR}"

find_free_port() {
  python - "$@" <<'PY'
import socket
import sys

start = int(sys.argv[1]) if len(sys.argv) > 1 else 29605
for port in range(start, start + 200):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", port))
        except OSError:
            continue
        print(port)
        raise SystemExit(0)
raise SystemExit("No free port found")
PY
}

quote() {
  printf "%q" "$1"
}

HOSTNAME_SHORT="${HOSTNAME_SHORT:-$(hostname)}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"

WANDB_ENTITY="${WANDB_ENTITY:-zihanw22}"
WANDB_PROJECT="${WANDB_PROJECT:-holosomatest}"
NUM_GPUS="${NUM_GPUS:-8}"
ENVS_PER_GPU="${ENVS_PER_GPU:-1024}"
TOTAL_ENVS="${TOTAL_ENVS:-$((NUM_GPUS * ENVS_PER_GPU))}"
NUM_ITERATIONS="${NUM_ITERATIONS:-20000}"
TRAINING_MODE="${TRAINING_MODE:-hybrid}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
LOGGING_INTERVAL="${LOGGING_INTERVAL:-25}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
SCHEDULE="${SCHEDULE:-adaptive}"
DESIRED_KL="${DESIRED_KL:-0.01}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-1e-5}"
MAX_LEARNING_RATE="${MAX_LEARNING_RATE:-1e-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
DEPTH_WEIGHT_DECAY="${DEPTH_WEIGHT_DECAY:-1e-2}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
STUDENT_ROLLOUT_PROB="${STUDENT_ROLLOUT_PROB:-0.0}"
INIT_NOISE_STD="${INIT_NOISE_STD:-0.01}"
NUM_STEPS_PER_UPDATE="${NUM_STEPS_PER_UPDATE:-24}"
NUM_LEARNING_EPOCHS="${NUM_LEARNING_EPOCHS:-2}"
NUM_MINI_BATCHES="${NUM_MINI_BATCHES:-96}"
CLIP_PARAM="${CLIP_PARAM:-0.2}"
GAMMA="${GAMMA:-0.99}"
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
VALUE_LOSS_COEF="${VALUE_LOSS_COEF:-1.0}"
ENTROPY_COEF="${ENTROPY_COEF:-0.001}"
DAGGER_LOSS_COEF="${DAGGER_LOSS_COEF:-10.0}"
PPO_START_EPOCH="${PPO_START_EPOCH:-${PPO_START_STEP:-0}}"
DAGGER_END_EPOCH="${DAGGER_END_EPOCH:-${DAGGER_END_STEP:-10000}}"
DEPTH_HEIGHT="${DEPTH_HEIGHT:-58}"
DEPTH_WIDTH="${DEPTH_WIDTH:-87}"
RAW_DEPTH_HEIGHT="${RAW_DEPTH_HEIGHT:-60}"
RAW_DEPTH_WIDTH="${RAW_DEPTH_WIDTH:-106}"
DEPTH_MIN_RANGE="${DEPTH_MIN_RANGE:-0.3}"
DEPTH_MAX_RANGE="${DEPTH_MAX_RANGE:-2.0}"
DEPTH_HORIZONTAL_FOV_DEG="${DEPTH_HORIZONTAL_FOV_DEG:-101.41}"
DEPTH_CAMERA_BODY_NAME="${DEPTH_CAMERA_BODY_NAME:-torso_link}"
DEPTH_CAMERA_DEBUG_VIS="${DEPTH_CAMERA_DEBUG_VIS:-0}"
DISTILL_TAG="${DISTILL_TAG:-tracking}"

TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${1:-}}"
RUN_NAME="${RUN_NAME:-${HOSTNAME_SHORT}_g1_29dof_depth_student_${TRAINING_MODE}_distill_${DISTILL_TAG}_${NUM_GPUS}gpu_${ENVS_PER_GPU}env_${TIMESTAMP}}"
SESSION="${SESSION:-csp_depth_distill_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-logs/run_commands}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${SESSION}.log}"
MASTER_PORT="${MASTER_PORT:-$(find_free_port 29605)}"

if [[ -z "${TEACHER_CHECKPOINT}" ]]; then
  echo "TEACHER_CHECKPOINT is required, or pass it as the first argument." >&2
  exit 1
fi

if [[ "${1:-}" != "--run" && "${RUN_IN_TMUX:-1}" == "1" ]]; then
  mkdir -p "${LOG_DIR}"
  printf "%s\n" "${RUN_NAME}" > "${LOG_DIR}/${SESSION}.run_name"

  TMUX_ENV="RUN_IN_TMUX=0 TIMESTAMP=$(quote "${TIMESTAMP}") HOSTNAME_SHORT=$(quote "${HOSTNAME_SHORT}") WANDB_ENTITY=$(quote "${WANDB_ENTITY}") WANDB_PROJECT=$(quote "${WANDB_PROJECT}") NUM_GPUS=$(quote "${NUM_GPUS}") ENVS_PER_GPU=$(quote "${ENVS_PER_GPU}") TOTAL_ENVS=$(quote "${TOTAL_ENVS}") NUM_ITERATIONS=$(quote "${NUM_ITERATIONS}") TRAINING_MODE=$(quote "${TRAINING_MODE}") SAVE_INTERVAL=$(quote "${SAVE_INTERVAL}") LOGGING_INTERVAL=$(quote "${LOGGING_INTERVAL}") LEARNING_RATE=$(quote "${LEARNING_RATE}") SCHEDULE=$(quote "${SCHEDULE}") DESIRED_KL=$(quote "${DESIRED_KL}") MIN_LEARNING_RATE=$(quote "${MIN_LEARNING_RATE}") MAX_LEARNING_RATE=$(quote "${MAX_LEARNING_RATE}") WEIGHT_DECAY=$(quote "${WEIGHT_DECAY}") DEPTH_WEIGHT_DECAY=$(quote "${DEPTH_WEIGHT_DECAY}") MAX_GRAD_NORM=$(quote "${MAX_GRAD_NORM}") STUDENT_ROLLOUT_PROB=$(quote "${STUDENT_ROLLOUT_PROB}") INIT_NOISE_STD=$(quote "${INIT_NOISE_STD}") NUM_STEPS_PER_UPDATE=$(quote "${NUM_STEPS_PER_UPDATE}") NUM_LEARNING_EPOCHS=$(quote "${NUM_LEARNING_EPOCHS}") NUM_MINI_BATCHES=$(quote "${NUM_MINI_BATCHES}") CLIP_PARAM=$(quote "${CLIP_PARAM}") GAMMA=$(quote "${GAMMA}") GAE_LAMBDA=$(quote "${GAE_LAMBDA}") VALUE_LOSS_COEF=$(quote "${VALUE_LOSS_COEF}") ENTROPY_COEF=$(quote "${ENTROPY_COEF}") DAGGER_LOSS_COEF=$(quote "${DAGGER_LOSS_COEF}") PPO_START_EPOCH=$(quote "${PPO_START_EPOCH}") DAGGER_END_EPOCH=$(quote "${DAGGER_END_EPOCH}") DEPTH_HEIGHT=$(quote "${DEPTH_HEIGHT}") DEPTH_WIDTH=$(quote "${DEPTH_WIDTH}") RAW_DEPTH_HEIGHT=$(quote "${RAW_DEPTH_HEIGHT}") RAW_DEPTH_WIDTH=$(quote "${RAW_DEPTH_WIDTH}") DEPTH_MIN_RANGE=$(quote "${DEPTH_MIN_RANGE}") DEPTH_MAX_RANGE=$(quote "${DEPTH_MAX_RANGE}") DEPTH_HORIZONTAL_FOV_DEG=$(quote "${DEPTH_HORIZONTAL_FOV_DEG}") DEPTH_CAMERA_BODY_NAME=$(quote "${DEPTH_CAMERA_BODY_NAME}") DEPTH_CAMERA_DEBUG_VIS=$(quote "${DEPTH_CAMERA_DEBUG_VIS}") DISTILL_TAG=$(quote "${DISTILL_TAG}") TEACHER_CHECKPOINT=$(quote "${TEACHER_CHECKPOINT}") RUN_NAME=$(quote "${RUN_NAME}") SESSION=$(quote "${SESSION}") LOG_DIR=$(quote "${LOG_DIR}") LOG_FILE=$(quote "${LOG_FILE}") MASTER_PORT=$(quote "${MASTER_PORT}")"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    TMUX_ENV="CUDA_VISIBLE_DEVICES=$(quote "${CUDA_VISIBLE_DEVICES}") ${TMUX_ENV}"
  fi
  TMUX_CMD="cd $(quote "${SCRIPT_DIR}") && env ${TMUX_ENV} bash $(quote "${SCRIPT_DIR}/csp_depth_distill.sh") --run > $(quote "${LOG_FILE}") 2>&1"

  tmux new-session -d -s "${SESSION}" "${TMUX_CMD}"
  echo "Started CSP depth student distillation."
  echo "  session: ${SESSION}"
  echo "  run_name: ${RUN_NAME}"
  echo "  log: ${LOG_FILE}"
  echo "  master_port: ${MASTER_PORT}"
  echo "  total_envs: ${TOTAL_ENVS} (${NUM_GPUS} x ${ENVS_PER_GPU})"
  echo "  training_mode: ${TRAINING_MODE}"
  echo "  kl_schedule: ${SCHEDULE} desired_kl=${DESIRED_KL} lr=[${MIN_LEARNING_RATE}, ${MAX_LEARNING_RATE}]"
  echo "  teacher_checkpoint: ${TEACHER_CHECKPOINT}"
  echo "  distill_tag: ${DISTILL_TAG}"
  echo "  depth: ${DEPTH_HEIGHT}x${DEPTH_WIDTH} from raw ${RAW_DEPTH_HEIGHT}x${RAW_DEPTH_WIDTH}"
  echo "  physics_rollout: true"
  exit 0
fi

if [[ "${1:-}" == "--run" ]]; then
  shift
fi
if [[ -n "${1:-}" && "${1}" == "${TEACHER_CHECKPOINT}" ]]; then
  shift
fi

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null \
  || source /home/ubuntu/.holosoma_deps/miniconda3/etc/profile.d/conda.sh

conda activate "${CONDA_ENV_NAME:-hssim}"
source scripts/source_isaacsim_setup.sh

export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

DEPTH_CAMERA_FLAGS=()
if [[ "${DEPTH_CAMERA_DEBUG_VIS,,}" == "true" || "${DEPTH_CAMERA_DEBUG_VIS}" == "1" ]]; then
  DEPTH_CAMERA_FLAGS+=(--depth-camera-debug-vis)
fi

torchrun \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${NUM_GPUS}" \
  src/holosoma/holosoma/distill_depth_student.py \
  --teacher-checkpoint "${TEACHER_CHECKPOINT}" \
  --num-envs "${TOTAL_ENVS}" \
  --iterations "${NUM_ITERATIONS}" \
  --training-mode "${TRAINING_MODE}" \
  --save-interval "${SAVE_INTERVAL}" \
  --logging-interval "${LOGGING_INTERVAL}" \
  --learning-rate "${LEARNING_RATE}" \
  --schedule "${SCHEDULE}" \
  --desired-kl "${DESIRED_KL}" \
  --min-learning-rate "${MIN_LEARNING_RATE}" \
  --max-learning-rate "${MAX_LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --depth-weight-decay "${DEPTH_WEIGHT_DECAY}" \
  --max-grad-norm "${MAX_GRAD_NORM}" \
  --student-rollout-prob "${STUDENT_ROLLOUT_PROB}" \
  --init-noise-std "${INIT_NOISE_STD}" \
  --num-steps-per-update "${NUM_STEPS_PER_UPDATE}" \
  --num-learning-epochs "${NUM_LEARNING_EPOCHS}" \
  --num-mini-batches "${NUM_MINI_BATCHES}" \
  --clip-param "${CLIP_PARAM}" \
  --gamma "${GAMMA}" \
  --gae-lambda "${GAE_LAMBDA}" \
  --value-loss-coef "${VALUE_LOSS_COEF}" \
  --entropy-coef "${ENTROPY_COEF}" \
  --dagger-loss-coef "${DAGGER_LOSS_COEF}" \
  --ppo-start-epoch "${PPO_START_EPOCH}" \
  --dagger-end-epoch "${DAGGER_END_EPOCH}" \
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
  --wandb \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-project "${WANDB_PROJECT}" \
  "${DEPTH_CAMERA_FLAGS[@]}" \
  "$@"
