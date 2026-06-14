#!/usr/bin/env bash
set -euo pipefail

# Prepare and launch the 51-clip convex-hull AS solid distillation run across
# the six 8xL40S nodes listed below. This script runs from one control node and
# starts one tmux session per node.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "${SCRIPT_DIR}"

# Private IPs for the requested SkyPilot clusters. Cluster names are not DNS
# names on the training nodes, so default to VPC-reachable private IPs.
DEFAULT_NODES=(
  10.99.0.141  # zzzihanw-15
  10.99.0.97   # zzzihanw-17
  10.99.1.69   # zzzihanw-z
  10.99.1.60   # z1hanw
  10.99.1.122  # zzzihanw-f
  10.99.1.21   # zzzihanw-e
)

usage() {
  cat <<'EOF'
Usage:
  bash batch_ne.sh prepare     # git pull + cp_ch.sh on every node
  bash batch_ne.sh launch      # start 48-GPU training in tmux
  bash batch_ne.sh all         # prepare, then launch
  bash batch_ne.sh status      # show tmux/log status on every node
  bash batch_ne.sh stop        # kill only this script's tmux session

Useful env:
  NODES="node0 node1 ..."      override node list
  REMOTE_REPO=/home/ubuntu/FAR/holosoma
  SESSION=distill_as_ch51_48gpu
  PER_GPU_ENVS=2048            1024 minimum recommended; try 4096 if stable
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  MASTER_ADDR=<node0>          default first node
  MASTER_PORT=29651
  GIT_REMOTE=origin
  GIT_BRANCH=main
  CH_BANK_NAME=as_realmesh67000_finalpos_convexsurface51_convexhull
  RESUME_FROM_BOX=1            initialize student policy from box-button checkpoint
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
SESSION=${SESSION:-distill_as_ch51_48gpu}
PER_GPU_ENVS=${PER_GPU_ENVS:-2048}
MIN_PER_GPU_ENVS=${MIN_PER_GPU_ENVS:-1024}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
NNODES=${NNODES:-${#NODE_LIST[@]}}
MASTER_ADDR=${MASTER_ADDR:-${NODE_LIST[0]}}
MASTER_PORT=${MASTER_PORT:-29651}
GIT_REMOTE=${GIT_REMOTE:-origin}
GIT_BRANCH=${GIT_BRANCH:-main}
CH_BANK_NAME=${CH_BANK_NAME:-as_realmesh67000_finalpos_convexsurface51_convexhull}
RESTART=${RESTART:-1}
DRY_RUN=${DRY_RUN:-0}
SSH_OPTS=${SSH_OPTS:-"-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
RESUME_FROM_BOX=${RESUME_FROM_BOX:-1}
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
if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  RUN_NAME=${RUN_NAME:-g1_w_object_distill_as_button_solid_ch51_48gpu_init_box}
  TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_button_solid_ch51_48gpu_init_box_depth}
else
  RUN_NAME=${RUN_NAME:-g1_w_object_distill_as_button_solid_ch51_48gpu}
  TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_button_solid_ch51_48gpu_depth}
fi
SCHEDULE_NAME=${SCHEDULE_NAME:-as_ch51_sparse_root_ppo_first_contact_drop_button_solid}
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

TOTAL_NUM_ENVS=$((PER_GPU_ENVS * NPROC * NNODES))
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
export PER_GPU_ENVS=$(quote "${PER_GPU_ENVS}")
export CH_BANK_NAME=$(quote "${CH_BANK_NAME}")
export CORL_SOLID80_BANK_NAME=$(quote "${CH_BANK_NAME}")
export AS_SUCCESS133_FINAL0P5=1
export AS_RANK_LOCAL_SHARDS=1
export RESUME_FROM_BOX=$(quote "${RESUME_FROM_BOX}")
export OMOMO_EXPECTED_TOTAL=51
export RESUME_FROM_BOX_EXPECTED_TOTAL=51
export RUN_NAME=$(quote "${RUN_NAME}")
export TRAINING_NAME=$(quote "${TRAINING_NAME}")
export SCHEDULE_NAME=$(quote "${SCHEDULE_NAME}")
export SCHEDULE_NOTES=$(quote "48-GPU AS solid distillation on the 51-clip convex-hull bank. Clips are final-position successes whose retained contact points are all within 1cm of the convex hull surface.")
export PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY:-301989888}
export PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY=${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY:-301989888}
EOF
)
  local train_cmd
  train_cmd=$(cat <<EOF
set -euo pipefail
cd $(quote "${REMOTE_REPO}")
mkdir -p $(quote "${REMOTE_REPO}/${LOG_DIR}")
${env_exports}
echo "[INFO][${node}] session=${SESSION} node_rank=${node_rank}/${NNODES} per_gpu_envs=${PER_GPU_ENVS} total_num_envs=${TOTAL_NUM_ENVS}"
echo "[INFO][${node}] master=${MASTER_ADDR}:${MASTER_PORT} log=${log_file}"
exec bash distill_as_button_solid.sh 2>&1 | tee $(quote "${log_file}")
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
