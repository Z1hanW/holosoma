#!/usr/bin/env bash
set -euo pipefail

# Prepare and launch AS real-mesh tracking/generalist training across:
#   zzzihanw-9, zzzihanw-8, zzzihanw, zzzihanw-1, zzzihanw-4
#
# Target: 5 nodes x 8 GPUs = 40 GPUs, exactly 4096 environments per GPU.
# train_agent.py divides --training.num-envs by WORLD_SIZE, so this launcher
# passes NUM_ENVS=4096*40=163840 as the global environment count.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "${SCRIPT_DIR}"

DEFAULT_NODES=(
  zzzihanw-9
  zzzihanw-8
  zzzihanw
  zzzihanw-1
  zzzihanw-4
)

usage() {
  cat <<'EOF'
Usage:
  bash batch_track.sh prepare   # pull code, sync local assets, run cp_as.sh on every node
  bash batch_track.sh launch    # start 40-GPU training in tmux
  bash batch_track.sh all       # prepare, then launch
  bash batch_track.sh status    # show tmux/log status on every node
  bash batch_track.sh stop      # kill only this launcher's tmux session

Useful env:
  NODES="zzzihanw-9 zzzihanw-8 zzzihanw zzzihanw-1 zzzihanw-4"
  REMOTE_REPO=/home/ubuntu/FAR/holosoma
  SESSION=as_track_40gpu_box_resume
  PER_GPU_ENVS=4096
  NPROC=8
  MASTER_ADDR=10.0.86.119
  MASTER_PORT=29640
  BOX_CHECKPOINT_SRC=/data/logs_new/boxer/.../model_30500.pt
  BOX_CHECKPOINT_REL=checkpoints/box/model_30500.pt
  POLICY_HISTORY_LENGTH=1      explicit target-policy history for box warm start
  BOX_RESUME_HISTORY_LENGTH=1  history encoded by the initializer
  PREPARE_CP_AS=1
  SYNC_LOCAL_ASSETS=1
  RESTART=1
  DRY_RUN=1
EOF
}

ACTION=${1:-all}
case "${ACTION}" in
  -h|--help|help)
    usage
    exit 0
    ;;
  launch|all)
    echo "[ERROR] batch_track.sh launch is quarantined: this legacy path performs per-node git pull and" >&2
    echo "[ERROR] cannot prove immutable source/data/checkpoint/NCCL provenance. Its historical 195-clip" >&2
    echo "[ERROR] contact bank also lacks 28 box sidecars and mixes 30 Hz retarget intervals with rollout steps." >&2
    echo "[ERROR] Use batch_ne.sh with a content-addressed snapshot and a fully validated contact bank." >&2
    exit 2
    ;;
esac

if [[ -n "${NODES:-}" ]]; then
  # shellcheck disable=SC2206
  NODE_LIST=(${NODES})
else
  NODE_LIST=("${DEFAULT_NODES[@]}")
fi

REMOTE_REPO=${REMOTE_REPO:-/home/ubuntu/FAR/holosoma}
SESSION=${SESSION:-as_track_40gpu_box_resume}
PER_GPU_ENVS=${PER_GPU_ENVS:-4096}
NPROC=${NPROC:-8}
NNODES=${NNODES:-${#NODE_LIST[@]}}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
MASTER_ADDR=${MASTER_ADDR:-10.0.86.119}
MASTER_PORT=${MASTER_PORT:-29640}
GIT_REMOTE=${GIT_REMOTE:-origin}
GIT_BRANCH=${GIT_BRANCH:-main}
RESTART=${RESTART:-1}
PREPARE_CP_AS=${PREPARE_CP_AS:-1}
SYNC_LOCAL_ASSETS=${SYNC_LOCAL_ASSETS:-1}
DRY_RUN=${DRY_RUN:-0}
SSH_OPTS=${SSH_OPTS:-"-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_DIR=${LOG_DIR:-logs/batch_track/${SESSION}_${RUN_STAMP}}

AS_KEEP_BANK=${AS_KEEP_BANK:-carryany_filter_scale_noscale_keep169_20260513}
AS_OUTPUT_BANK=${AS_OUTPUT_BANK:-${AS_KEEP_BANK}_plus_box_teacher_rollout}
AS_EXPECTED_TOTAL=${AS_EXPECTED_TOTAL:-195}
AS_KEEP_EXPECTED_TOTAL=${AS_KEEP_EXPECTED_TOTAL:-167}
LOCAL_TEACHER_BANK_REL=${LOCAL_TEACHER_BANK_REL:-outputs/teacher_box_contacts_rollout_ref_motionbank_20260415b_utc/motion_bank}
LOCAL_TEACHER_BANK_SRC=${LOCAL_TEACHER_BANK_SRC:-${SCRIPT_DIR}/${LOCAL_TEACHER_BANK_REL}}

BOX_CHECKPOINT_SRC=${BOX_CHECKPOINT_SRC:-/data/logs_new/boxer/20260610_001228-g1_29dof_wbt_w_object_distill_box_button_contact_aware_drop_button_depth-locomotion/model_30500.pt}
BOX_CHECKPOINT_REL=${BOX_CHECKPOINT_REL:-checkpoints/box/$(basename "${BOX_CHECKPOINT_SRC}")}
BOX_CHECKPOINT_REMOTE="${REMOTE_REPO}/${BOX_CHECKPOINT_REL}"
BOX_RESUME_HISTORY_LENGTH=${BOX_RESUME_HISTORY_LENGTH:-1}
POLICY_HISTORY_LENGTH=${POLICY_HISTORY_LENGTH:-${BOX_RESUME_HISTORY_LENGTH}}

RUN_NAME=${RUN_NAME:-as-track-real-mesh-cotrack-40gpu-resume-box-history${POLICY_HISTORY_LENGTH}}
WANDB_PROJECT=${WANDB_PROJECT:-carry-any}
HOLOSOMA_OBJECT_COLLIDER_TYPE=${HOLOSOMA_OBJECT_COLLIDER_TYPE:-convex_hull}
TRAINING_SEED=${TRAINING_SEED:-}
RANDOMIZATION_PRESET=${RANDOMIZATION_PRESET:-}
INIT_AT_RANDOM_EP_LEN=${INIT_AT_RANDOM_EP_LEN:-}
NUM_LEARNING_ITERATIONS=${NUM_LEARNING_ITERATIONS:-40000}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}

PHYSX_GPU_MAX_RIGID_CONTACT_COUNT=${PHYSX_GPU_MAX_RIGID_CONTACT_COUNT:-8388608}
PHYSX_GPU_MAX_RIGID_PATCH_COUNT=${PHYSX_GPU_MAX_RIGID_PATCH_COUNT:-1048576}
PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY:-536870912}
PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY=${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY:-536870912}
PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY:-134217728}
PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-33554432}
PHYSX_GPU_HEAP_CAPACITY=${PHYSX_GPU_HEAP_CAPACITY:-33554432}
PHYSX_GPU_TEMP_BUFFER_CAPACITY=${PHYSX_GPU_TEMP_BUFFER_CAPACITY:-8388608}

if [[ "${#NODE_LIST[@]}" -lt 1 ]]; then
  echo "[ERROR] Empty node list." >&2
  exit 2
fi
if ! [[ "${PER_GPU_ENVS}" =~ ^[0-9]+$ ]] || (( PER_GPU_ENVS != 4096 )); then
  echo "[ERROR] PER_GPU_ENVS must be exactly 4096 for this run. Got: ${PER_GPU_ENVS}" >&2
  exit 2
fi
if ! [[ "${NPROC}" =~ ^[0-9]+$ ]] || (( NPROC != 8 )); then
  echo "[ERROR] NPROC must be exactly 8 for these 8-GPU nodes. Got: ${NPROC}" >&2
  exit 2
fi
if ! [[ "${NNODES}" =~ ^[0-9]+$ ]] || (( NNODES != ${#NODE_LIST[@]} )); then
  echo "[ERROR] NNODES must equal node list length. Got NNODES=${NNODES}, nodes=${#NODE_LIST[@]}" >&2
  exit 2
fi
if ! [[ "${BOX_RESUME_HISTORY_LENGTH}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] BOX_RESUME_HISTORY_LENGTH must be a positive integer. Got: ${BOX_RESUME_HISTORY_LENGTH}" >&2
  exit 2
fi
if ! [[ "${POLICY_HISTORY_LENGTH}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] POLICY_HISTORY_LENGTH must be a positive integer. Got: ${POLICY_HISTORY_LENGTH}" >&2
  exit 2
fi
if [[ "${POLICY_HISTORY_LENGTH}" != "${BOX_RESUME_HISTORY_LENGTH}" ]]; then
  echo "[ERROR] Box warm start cannot silently change target-policy history." >&2
  echo "[ERROR] POLICY_HISTORY_LENGTH=${POLICY_HISTORY_LENGTH}, BOX_RESUME_HISTORY_LENGTH=${BOX_RESUME_HISTORY_LENGTH}." >&2
  echo "[ERROR] Use history ${BOX_RESUME_HISTORY_LENGTH}, or disable box initialization in a different launcher." >&2
  exit 2
fi

TOTAL_GPUS=$((NNODES * NPROC))
TOTAL_NUM_ENVS=$((TOTAL_GPUS * PER_GPU_ENVS))

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

rsync_to_node() {
  local src="$1"
  local node="$2"
  local dst="$3"
  if [[ "${SYNC_LOCAL_ASSETS}" != "1" ]]; then
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN] rsync -az %s %s:%s\n' "${src}" "${node}" "${dst}"
    return 0
  fi
  # shellcheck disable=SC2086
  rsync -az -e "ssh ${SSH_OPTS}" "${src}" "${node}:${dst}"
}

sync_local_assets() {
  local node="$1"
  if [[ "${SYNC_LOCAL_ASSETS}" != "1" ]]; then
    return 0
  fi
  if [[ ! -d "${LOCAL_TEACHER_BANK_SRC}" ]]; then
    echo "[ERROR] Missing local teacher motion bank: ${LOCAL_TEACHER_BANK_SRC}" >&2
    exit 1
  fi
  if [[ ! -f "${BOX_CHECKPOINT_SRC}" ]]; then
    echo "[ERROR] Missing local box checkpoint: ${BOX_CHECKPOINT_SRC}" >&2
    exit 1
  fi

  remote_run "${node}" "mkdir -p $(quote "${REMOTE_REPO}/${LOCAL_TEACHER_BANK_REL}") $(quote "$(dirname "${BOX_CHECKPOINT_REMOTE}")")"
  rsync_to_node "${LOCAL_TEACHER_BANK_SRC}/" "${node}" "${REMOTE_REPO}/${LOCAL_TEACHER_BANK_REL}/"
  rsync_to_node "${BOX_CHECKPOINT_SRC}" "${node}" "${BOX_CHECKPOINT_REMOTE}"
}

prepare_node() {
  local node="$1"
  sync_local_assets "${node}"

  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
cd $(quote "${REMOTE_REPO}")
echo "[INFO][${node}] repo=\$(pwd)"
git fetch $(quote "${GIT_REMOTE}") $(quote "${GIT_BRANCH}")
git pull --ff-only $(quote "${GIT_REMOTE}") $(quote "${GIT_BRANCH}")
chmod +x batch_track.sh cp_as.sh train_as_general.sh train_object_generalist_ds.sh || true
test -f $(quote "${BOX_CHECKPOINT_REMOTE}")
test -d $(quote "${REMOTE_REPO}/${LOCAL_TEACHER_BANK_REL}")
source ./scripts/gpu_launch_defaults.sh
export PATH="\$(dirname "\${PYTHON_BIN}"):\${PATH}"
if [[ $(quote "${PREPARE_CP_AS}") == "1" ]]; then
  COPY_KEEP_BANK=1 KEEP_EXPECTED_TOTAL=$(quote "${AS_KEEP_EXPECTED_TOTAL}") OUTPUT_BANK_NAME=$(quote "${AS_OUTPUT_BANK}") bash cp_as.sh
fi
"\${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

repo = Path.cwd()
mixed = repo / "data" / "ds_as_data" / "${AS_OUTPUT_BANK}"
keep = repo / "data" / "ds_as_data" / "${AS_KEEP_BANK}"
contact = keep / "contact_export_from_retarget"
expected_total = int("${AS_EXPECTED_TOTAL}")
expected_keep = int("${AS_KEEP_EXPECTED_TOTAL}")

npz_total = len(list(mixed.glob("*.npz")))
if npz_total != expected_total:
    raise SystemExit(f"[ERROR] {mixed} has {npz_total} .npz files, expected {expected_total}")

map_path = mixed / "_clip_object_urdf_map.json"
payload = json.loads(map_path.read_text(encoding="utf-8"))
clips = payload.get("clips", payload)
if len(clips) != expected_total:
    raise SystemExit(f"[ERROR] {map_path} has {len(clips)} clips, expected {expected_total}")

keep_npz = len(list(keep.glob("*.npz")))
if keep_npz != expected_keep:
    raise SystemExit(f"[ERROR] {keep} has {keep_npz} .npz files, expected {expected_keep}")
if not contact.is_dir():
    raise SystemExit(f"[ERROR] Missing contact export root: {contact}")

print(f"[INFO] AS data ready: mixed={npz_total}, keep={keep_npz}, contact={contact}")
PY
EOF
)
  remote_run "${node}" "${cmd}"
}

launch_node() {
  local node="$1"
  local node_rank="$2"
  local safe_node="${node//[^A-Za-z0-9_.-]/_}"
  local log_file="${REMOTE_REPO}/${LOG_DIR}/node_${node_rank}_${safe_node}.log"
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
export NUM_ENVS=$(quote "${TOTAL_NUM_ENVS}")
export TOTAL_NUM_ENVS=$(quote "${TOTAL_NUM_ENVS}")
export AS_EXPECTED_TOTAL=$(quote "${AS_EXPECTED_TOTAL}")
export AS_DATA_DIR=$(quote "data/ds_as_data/${AS_OUTPUT_BANK}")
export AS_OBJECT_MAP=$(quote "data/ds_as_data/${AS_OUTPUT_BANK}/_clip_object_urdf_map.json")
export CONTACT_EXPORT_ROOT=$(quote "data/ds_as_data/${AS_KEEP_BANK}/contact_export_from_retarget")
export RESUME_FROM_BOX=1
export BOX_RESUME_CKPT=$(quote "${BOX_CHECKPOINT_REMOTE}")
export BOX_RESUME_HISTORY_LENGTH=$(quote "${BOX_RESUME_HISTORY_LENGTH}")
export POLICY_HISTORY_LENGTH=$(quote "${POLICY_HISTORY_LENGTH}")
export WANDB_RESUME_SAME_RUN=0
export WANDB_PROJECT=$(quote "${WANDB_PROJECT}")
export SEQUENCE_NAME=$(quote "${RUN_NAME}")
export NUM_LEARNING_ITERATIONS=$(quote "${NUM_LEARNING_ITERATIONS}")
export SAVE_INTERVAL=$(quote "${SAVE_INTERVAL}")
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=1
export HOLOSOMA_OBJECT_SPAWN_MODE=single_slot_multi_urdf
export HOLOSOMA_OBJECT_COLLIDER_TYPE=$(quote "${HOLOSOMA_OBJECT_COLLIDER_TYPE}")
export OBJECT_GEOMETRY_MODE=mesh
export PHYSX_GPU_MAX_RIGID_CONTACT_COUNT=$(quote "${PHYSX_GPU_MAX_RIGID_CONTACT_COUNT}")
export PHYSX_GPU_MAX_RIGID_PATCH_COUNT=$(quote "${PHYSX_GPU_MAX_RIGID_PATCH_COUNT}")
export PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=$(quote "${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY}")
export PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY=$(quote "${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY}")
export PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=$(quote "${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY}")
export PHYSX_GPU_COLLISION_STACK_SIZE=$(quote "${PHYSX_GPU_COLLISION_STACK_SIZE}")
export PHYSX_GPU_HEAP_CAPACITY=$(quote "${PHYSX_GPU_HEAP_CAPACITY}")
export PHYSX_GPU_TEMP_BUFFER_CAPACITY=$(quote "${PHYSX_GPU_TEMP_BUFFER_CAPACITY}")
EOF
)
  if [[ -n "${TRAINING_SEED}" ]]; then
    env_exports+=$'\n'"export TRAINING_SEED=$(quote "${TRAINING_SEED}")"
  fi
  if [[ -n "${RANDOMIZATION_PRESET}" ]]; then
    env_exports+=$'\n'"export RANDOMIZATION_PRESET=$(quote "${RANDOMIZATION_PRESET}")"
  fi
  if [[ -n "${INIT_AT_RANDOM_EP_LEN}" ]]; then
    env_exports+=$'\n'"export INIT_AT_RANDOM_EP_LEN=$(quote "${INIT_AT_RANDOM_EP_LEN}")"
  fi

  local train_cmd
  train_cmd=$(cat <<EOF
set -euo pipefail
cd $(quote "${REMOTE_REPO}")
mkdir -p $(quote "${REMOTE_REPO}/${LOG_DIR}")
${env_exports}
echo "[INFO][${node}] session=${SESSION} node_rank=${node_rank}/${NNODES} nproc=${NPROC} total_gpus=${TOTAL_GPUS}"
echo "[INFO][${node}] per_gpu_envs=${PER_GPU_ENVS} total_num_envs=${TOTAL_NUM_ENVS}"
echo "[INFO][${node}] master=${MASTER_ADDR}:${MASTER_PORT} checkpoint=${BOX_CHECKPOINT_REMOTE}"
echo "[INFO][${node}] log=${log_file}"
exec bash train_as_general.sh 2>&1 | tee $(quote "${log_file}")
EOF
)

  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
cd $(quote "${REMOTE_REPO}")
git fetch $(quote "${GIT_REMOTE}") $(quote "${GIT_BRANCH}")
git pull --ff-only $(quote "${GIT_REMOTE}") $(quote "${GIT_BRANCH}")
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
  local safe_node="${node//[^A-Za-z0-9_.-]/_}"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
cd $(quote "${REMOTE_REPO}")
echo "===== ${node} ====="
tmux has-session -t $(quote "${SESSION}") 2>/dev/null && tmux list-sessions | grep -F $(quote "${SESSION}") || echo "tmux:${SESSION}:not-running"
latest_log=\$(ls -1t logs/batch_track/${SESSION}_*/node_*_${safe_node}.log 2>/dev/null | head -1 || true)
if [[ -n "\${latest_log}" ]]; then
  echo "log=\${latest_log}"
  tail -80 "\${latest_log}"
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
  echo "[INFO] Launching ${NNODES} nodes x ${NPROC} GPUs = ${TOTAL_GPUS} GPUs"
  echo "[INFO] PER_GPU_ENVS=${PER_GPU_ENVS} TOTAL_NUM_ENVS=${TOTAL_NUM_ENVS}"
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
