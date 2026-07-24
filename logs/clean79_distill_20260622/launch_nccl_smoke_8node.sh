#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/ubuntu/FAR/holosoma}
cd "${REPO}"

if [[ -n "${NODES:-}" ]]; then
  # shellcheck disable=SC2206
  NODE_LIST=(${NODES})
else
  NODE_LIST=(
    10.99.1.134
    10.99.0.117
    10.99.0.18
    10.99.0.227
    10.99.0.116
    10.99.0.165
    10.99.0.239
    10.99.0.167
  )
fi

ACTION=${1:-launch}
SESSION=${SESSION:-nccl_smoke_8node}
RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}
LOG_DIR=${LOG_DIR:-logs/clean79_distill_20260622/${SESSION}_${RUN_STAMP}}
MASTER_ADDR=${MASTER_ADDR:-${NODE_LIST[0]}}
MASTER_PORT=${MASTER_PORT:-29881}
NPROC=${NPROC:-8}
NNODES=${NNODES:-${#NODE_LIST[@]}}
SSH_OPTS=${SSH_OPTS:-"-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"}
PYTHON_BIN=${PYTHON_BIN:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp135s0}
GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
NCCL_DEBUG=${NCCL_DEBUG:-WARN}
NCCL_LIB_DIR=${NCCL_LIB_DIR:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/nvidia/nccl/lib}

if (( ${#NODE_LIST[@]} != 8 )); then
  echo "[ERROR] Expected 8 nodes, got ${#NODE_LIST[@]}: ${NODE_LIST[*]}" >&2
  exit 2
fi

quote() {
  printf '%q' "$1"
}

remote_run() {
  local node="$1"
  local cmd="$2"
  # shellcheck disable=SC2086
  ssh ${SSH_OPTS} "${node}" "${cmd}"
}

sync_node() {
  local node="$1"
  remote_run "${node}" "mkdir -p $(quote "${REPO}/scripts") $(quote "${REPO}/${LOG_DIR}")"
  rsync -az scripts/distributed_barrier_smoke.py "${node}:${REPO}/scripts/"
}

launch_node() {
  local node="$1"
  local node_rank="$2"
  local run_script="${REPO}/${LOG_DIR}/run_node_${node_rank}.sh"
  local log_file="${REPO}/${LOG_DIR}/node_${node_rank}_${node}.log"
  local remote_script
  remote_script=$(cat <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(quote "${REPO}")
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=$(quote "${NCCL_SOCKET_IFNAME}")
export GLOO_SOCKET_IFNAME=$(quote "${GLOO_SOCKET_IFNAME}")
export NCCL_IB_DISABLE=$(quote "${NCCL_IB_DISABLE}")
export NCCL_DEBUG=$(quote "${NCCL_DEBUG}")
export TORCH_DIST_TIMEOUT_SEC=\${TORCH_DIST_TIMEOUT_SEC:-120}
export NCCL_LIB_DIR=$(quote "${NCCL_LIB_DIR}")
export LD_LIBRARY_PATH=$(quote "${NCCL_LIB_DIR}")\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}
exec $(quote "${PYTHON_BIN}") -m torch.distributed.run \
  --nnodes=$(quote "${NNODES}") \
  --node_rank=$(quote "${node_rank}") \
  --master_addr=$(quote "${MASTER_ADDR}") \
  --nproc_per_node=$(quote "${NPROC}") \
  --max_restarts=0 \
  --master_port=$(quote "${MASTER_PORT}") \
  scripts/distributed_barrier_smoke.py
EOF
)
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
cd $(quote "${REPO}")
mkdir -p $(quote "${REPO}/${LOG_DIR}")
cat > $(quote "${run_script}") <<'REMOTE_SCRIPT'
${remote_script}
REMOTE_SCRIPT
chmod +x $(quote "${run_script}")
if tmux has-session -t $(quote "${SESSION}") 2>/dev/null; then
  tmux kill-session -t $(quote "${SESSION}")
fi
tmux new-session -d -s $(quote "${SESSION}") "timeout 180 bash $(quote "${run_script}") 2>&1 | tee $(quote "${log_file}")"
tmux display-message -p -t $(quote "${SESSION}") "[INFO][${node}] started #{session_name}"
EOF
)
  remote_run "${node}" "${cmd}"
}

stop_node() {
  local node="$1"
  remote_run "${node}" "tmux kill-session -t $(quote "${SESSION}") 2>/dev/null || true; pkill -f 'distributed_barrier_smoke.py' 2>/dev/null || true"
}

case "${ACTION}" in
  sync)
    for node in "${NODE_LIST[@]}"; do
      echo "[INFO] sync ${node}"
      sync_node "${node}"
    done
    ;;
  launch)
    for node in "${NODE_LIST[@]}"; do
      sync_node "${node}"
    done
    for idx in "${!NODE_LIST[@]}"; do
      echo "[INFO] launch ${NODE_LIST[$idx]} rank=${idx}"
      launch_node "${NODE_LIST[$idx]}" "${idx}"
    done
    ;;
  stop)
    for node in "${NODE_LIST[@]}"; do
      stop_node "${node}"
    done
    ;;
  *)
    echo "Usage: $0 {sync|launch|stop}" >&2
    exit 2
    ;;
esac
