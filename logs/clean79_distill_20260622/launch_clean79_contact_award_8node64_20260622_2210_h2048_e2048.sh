#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/ubuntu/FAR/holosoma}
cd "${REPO}"

NODES_DEFAULT=(
  10.99.1.134
  10.99.0.117
  10.99.0.18
  10.99.0.227
  10.99.0.116
  10.99.0.165
  10.99.0.239
  10.99.0.167
)

if [[ -n "${NODES:-}" ]]; then
  # shellcheck disable=SC2206
  NODE_LIST=(${NODES})
else
  NODE_LIST=("${NODES_DEFAULT[@]}")
fi

ACTION=${1:-launch}
SESSION=${SESSION:-clean79_contact_award_8node64_20260622_2210_h2048_e2048}
RUN_STAMP=${RUN_STAMP:-20260622_2210}
LOG_DIR=${LOG_DIR:-logs/clean79_distill_20260622/${SESSION}_${RUN_STAMP}}
SSH_OPTS=${SSH_OPTS:-"-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"}
MASTER_ADDR=${MASTER_ADDR:-${NODE_LIST[0]}}
MASTER_PORT=${MASTER_PORT:-29688}
NPROC=${NPROC:-8}
NNODES=${NNODES:-${#NODE_LIST[@]}}
PER_GPU_ENVS=${PER_GPU_ENVS:-2048}
TOTAL_NUM_ENVS=$((NNODES * NPROC * PER_GPU_ENVS))
ENV_TAG=e${PER_GPU_ENVS}
RESTART=${RESTART:-1}
TORCH_DIST_BACKEND=${TORCH_DIST_BACKEND:-gloo}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp135s0}
GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
NCCL_DEBUG=${NCCL_DEBUG:-WARN}
TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
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
HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=${HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK:-0}
HOLOSOMA_RANK_VISIBLE_DEVICES=${HOLOSOMA_RANK_VISIBLE_DEVICES:-0}
HOLOSOMA_GLOO_GRAD_REDUCE=${HOLOSOMA_GLOO_GRAD_REDUCE:-1}
HOLOSOMA_GLOO_BARRIER=${HOLOSOMA_GLOO_BARRIER:-1}
HOLOSOMA_GLOO_SMALL_COLLECTIVES=${HOLOSOMA_GLOO_SMALL_COLLECTIVES:-1}
HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE:-0}
HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER=${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER:-0}
HOLOSOMA_SKIP_WANDB_CHECKPOINT_UPLOAD=${HOLOSOMA_SKIP_WANDB_CHECKPOINT_UPLOAD:-1}
RESUME_FROM_BOX_WAS_SET=${RESUME_FROM_BOX+x}
RESUME_FROM_BOX=${RESUME_FROM_BOX:-1}
RESUME_TRAINING_CKPT=${RESUME_TRAINING_CKPT:-${RESUME_CHECKPOINT:-${RESUME_CKPT:-}}}
RESUME_WANDB_RUN_ID=${RESUME_WANDB_RUN_ID:-${WANDB_RUN_ID:-${RESUME_WANDB_ID:-}}}
WANDB_RESUME_MODE=${WANDB_RESUME_MODE:-${WANDB_RESUME:-must}}
WANDB_ENTITY=${WANDB_ENTITY:-zihanw22}
WANDB_RESUME_SAME_RUN=${WANDB_RESUME_SAME_RUN:-}
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

BANK_NAME=carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success155_bcleb5oi58000_final0p5_solid80_clean_box_bin_barrel_ball_meshphys_v1
SOURCE_MESH_BANK_NAME=carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success155_bcleb5oi58000_final0p5_primitiveproj_solid80_clean_box_bin_barrel_ball
CONTACT_EXPORT_NAME=contact_export_from_teacher_meshphys58000_20260622_043839_4gpu_all79
BANK_ROOT="${REPO}/data/ds_as_data/${BANK_NAME}"
SOURCE_MESH_BANK_ROOT="${REPO}/data/ds_as_data/${SOURCE_MESH_BANK_NAME}"
CONTACT_ROOT="${BANK_ROOT}/${CONTACT_EXPORT_NAME}"
TEACHER_CKPT="${REPO}/.teacher_checkpoints/model_58000.pt"
INIT_CKPT=${INIT_CKPT:-/home/ubuntu/.cache/holosoma/checkpoints/as_ch51_convex_pretrain_model_22000.pt}

RUN_NAME=${RUN_NAME:-g1_w_object_distill_as_button_clean79_contact_award_allregions_m58000_64gpu_${ENV_TAG}_h2048_1024_512_256_128}
TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_button_clean79_contact_award_allregions_m58000_64gpu_${ENV_TAG}_h2048_1024_512_256_128_depth}
SCHEDULE_NAME=${SCHEDULE_NAME:-as_clean79_contact_award_allregions_drop_button_ppo_first_m58000_64gpu_${ENV_TAG}_h2048}
SCHEDULE_NOTES=${SCHEDULE_NOTES:-"clean79 contact-award all79 meshphys m58000, 8 nodes x 8 GPUs, PER_GPU_ENVS=${PER_GPU_ENVS}, actor hidden dims [2048,1024,512,256,128], mesh-surface robot-side contact regions."}

quote() {
  printf '%q' "$1"
}

remote_run() {
  local node="$1"
  local cmd="$2"
  # shellcheck disable=SC2086
  ssh ${SSH_OPTS} "${node}" "${cmd}"
}

validate_config() {
  if (( ${#NODE_LIST[@]} != 8 )); then
    echo "[ERROR] Expected exactly 8 nodes, got ${#NODE_LIST[@]}: ${NODE_LIST[*]}" >&2
    exit 2
  fi
  if (( NNODES != ${#NODE_LIST[@]} )); then
    echo "[ERROR] NNODES=${NNODES} does not match node count ${#NODE_LIST[@]}" >&2
    exit 2
  fi
  if (( NPROC != 8 )); then
    echo "[ERROR] Expected NPROC=8 per node, got ${NPROC}" >&2
    exit 2
  fi
  if (( PER_GPU_ENVS < 1 )); then
    echo "[ERROR] Expected PER_GPU_ENVS >= 1, got ${PER_GPU_ENVS}" >&2
    exit 2
  fi
}

sync_node() {
  local node="$1"
  local local_ips
  local_ips="$(hostname -I 2>/dev/null || true)"
  if [[ " ${local_ips} " == *" ${node} "* ]]; then
    for path in \
      "${BANK_ROOT}/objects" \
      "${BANK_ROOT}/_single_slot_motion_bank_teacher_export_20260520_105947"; do
      if [[ -e "${path}" && ! -L "${path}" ]]; then
        mv "${path}" "${path}.bak_sync_${RUN_STAMP}"
      fi
    done
    return 0
  fi

  remote_run "${node}" "mkdir -p $(quote "${REPO}") $(quote "${REPO}/data/ds_as_data") $(quote "${REPO}/.teacher_checkpoints") /home/ubuntu/.cache/holosoma/checkpoints"
  remote_run "${node}" "for path in $(quote "${BANK_ROOT}/objects") $(quote "${BANK_ROOT}/_single_slot_motion_bank_teacher_export_20260520_105947"); do if [[ -e \"\${path}\" && ! -L \"\${path}\" ]]; then mv \"\${path}\" \"\${path}.bak_sync_$(quote "${RUN_STAMP}")\"; fi; done"
  rsync -az \
    distill_as_button.sh distill_as_perception.sh distill_box_perception.sh distill_root_box.sh distill_torso_box.sh \
    "${node}:${REPO}/"
  rsync -az scripts/gpu_launch_defaults.sh scripts/prepare_single_slot_object_map.py scripts/prepare_as_rank_shards.py \
    "${node}:${REPO}/scripts/"
  rsync -az src/holosoma/holosoma/ "${node}:${REPO}/src/holosoma/holosoma/"
  rsync -az "${SOURCE_MESH_BANK_ROOT}/" "${node}:${SOURCE_MESH_BANK_ROOT}/"
  rsync -az "${BANK_ROOT}/" "${node}:${BANK_ROOT}/"
  rsync -az "${TEACHER_CKPT}" "${node}:${TEACHER_CKPT}"
}

launch_node() {
  local node="$1"
  local node_rank="$2"
  local log_file="${REPO}/${LOG_DIR}/node_${node_rank}_${node}.log"
  local run_script="${REPO}/${LOG_DIR}/run_node_${node_rank}.sh"
  local remote_script
  remote_script=$(cat <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(quote "${REPO}")
mkdir -p $(quote "${REPO}/${LOG_DIR}")
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PYTHON_BIN=/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3
export NPROC=8
export NNODES=8
export NODE_RANK=${node_rank}
export MASTER_ADDR=$(quote "${MASTER_ADDR}")
export MASTER_PORT=$(quote "${MASTER_PORT}")
export PER_GPU_ENVS=${PER_GPU_ENVS}
export TOTAL_NUM_ENVS=${TOTAL_NUM_ENVS}
export AS_SUCCESS133_FINAL0P5=1
export AS_SUCCESS133_BANK_NAME=$(quote "${BANK_NAME}")
export OMOMO_EXPECTED_TOTAL=79
export RESUME_FROM_BOX=$(quote "${RESUME_FROM_BOX}")
export RESUME_CKPT=$(quote "${RESUME_TRAINING_CKPT}")
export WANDB_ENTITY=$(quote "${WANDB_ENTITY}")
export RESUME_WANDB_RUN_ID=$(quote "${RESUME_WANDB_RUN_ID}")
export WANDB_RESUME_MODE=$(quote "${WANDB_RESUME_MODE}")
export WANDB_RESUME_SAME_RUN=$(quote "${WANDB_RESUME_SAME_RUN}")
if [[ $(quote "${RESUME_FROM_BOX}") == "1" ]]; then
  export BOX_RESUME_CKPT=$(quote "${INIT_CKPT}")
  export RESUME_FROM_BOX_CKPT=$(quote "${INIT_CKPT}")
  export POLICY_INIT_CKPT=$(quote "${INIT_CKPT}")
  export POLICY_INIT_CHECKPOINT=$(quote "${INIT_CKPT}")
else
  unset BOX_RESUME_CKPT
  unset RESUME_FROM_BOX_CKPT
  unset POLICY_INIT_CKPT
  unset POLICY_INIT_CHECKPOINT
fi
export AS_CONTACT_AWARE=1
export ROOT_COMMAND_MODE=contact-aware
export STUDENT_ACTOR_INPUTS="['actor_obs_root_contact_aware','actor_obs_drop_button','actor_obs_proprio_with_actions_no_linvel']"
export STUDENT_ACTOR_HIDDEN_DIMS="[2048,1024,512,256,128]"
export AS_CONTACT_EXPORT_ROOT=$(quote "${CONTACT_ROOT}")
export ENABLE_OFFLINE_CONTACT_GUIDANCE=True
export OFFLINE_CONTACT_REGION_NAMES="['left_wrist','right_wrist','left_elbow','right_elbow','left_wrist_roll','right_wrist_roll','left_wrist_pitch','right_wrist_pitch','torso']"
export OFFLINE_WRIST_REGION_NAMES="['left_wrist','right_wrist']"
export RUN_NAME=$(quote "${RUN_NAME}")
export TRAINING_NAME=$(quote "${TRAINING_NAME}")
export TRAINING_PROJECT=carry-any
export WANDB_PROJECT=carry-any
export SCHEDULE_NAME=$(quote "${SCHEDULE_NAME}")
export SCHEDULE_NOTES=$(quote "${SCHEDULE_NOTES}")
export NUM_LEARNING_ITERATIONS=40000
export NUM_MINI_BATCHES=$(quote "${NUM_MINI_BATCHES:-64}")
export NUM_LEARNING_EPOCHS=$(quote "${NUM_LEARNING_EPOCHS:-1}")
export SAVE_INTERVAL=100
export HOLOSOMA_SKIP_INITIAL_CHECKPOINT=1
export HOLOSOMA_SKIP_GRAD_FINITE_CHECK=$(quote "${HOLOSOMA_SKIP_GRAD_FINITE_CHECK:-1}")
export HOLOSOMA_SKIP_LOSS_FINITE_CHECK=$(quote "${HOLOSOMA_SKIP_LOSS_FINITE_CHECK:-1}")
export HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION=$(quote "${HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION:-1}")
export HOLOSOMA_DEBUG_HEARTBEAT=$(quote "${HOLOSOMA_DEBUG_HEARTBEAT:-0}")
export HOLOSOMA_DEBUG_ACTOR=$(quote "${HOLOSOMA_DEBUG_ACTOR:-0}")
export HOLOSOMA_DEBUG_GRAD_REDUCE=$(quote "${HOLOSOMA_DEBUG_GRAD_REDUCE:-0}")
export HOLOSOMA_RANK_VISIBLE_DEVICES=$(quote "${HOLOSOMA_RANK_VISIBLE_DEVICES}")
export HOLOSOMA_GLOO_GRAD_REDUCE=$(quote "${HOLOSOMA_GLOO_GRAD_REDUCE}")
export HOLOSOMA_GLOO_BARRIER=$(quote "${HOLOSOMA_GLOO_BARRIER}")
export HOLOSOMA_GLOO_SMALL_COLLECTIVES=$(quote "${HOLOSOMA_GLOO_SMALL_COLLECTIVES}")
export HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=$(quote "${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE}")
export HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER=$(quote "${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER}")
export HOLOSOMA_SKIP_WANDB_CHECKPOINT_UPLOAD=$(quote "${HOLOSOMA_SKIP_WANDB_CHECKPOINT_UPLOAD}")
export HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE=$(quote "${HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE:-0}")
export HOLOSOMA_SYNC_AFTER_OPTIMIZER_STEP=$(quote "${HOLOSOMA_SYNC_AFTER_OPTIMIZER_STEP:-0}")
export HOLOSOMA_SYNC_AFTER_MICROBATCH_FORWARD=$(quote "${HOLOSOMA_SYNC_AFTER_MICROBATCH_FORWARD:-0}")
export HOLOSOMA_DEBUG_MICROBATCH_ALL=$(quote "${HOLOSOMA_DEBUG_MICROBATCH_ALL:-0}")
export HOLOSOMA_CONTIGUOUS_MINIBATCHES=$(quote "${HOLOSOMA_CONTIGUOUS_MINIBATCHES:-1}")
export HOLOSOMA_DAGGER_SUPERVISED_ONLY=$(quote "${HOLOSOMA_DAGGER_SUPERVISED_ONLY:-0}")
export HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP=$(quote "${HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP:-1}")
export HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH=$(quote "${HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH:-48}")
export HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD=$(quote "${HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD:-1}")
export HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC=$(quote "${HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC:-0}")
if [[ -n $(quote "${NVIDIA_TF32_OVERRIDE:-}") ]]; then
  export NVIDIA_TF32_OVERRIDE=$(quote "${NVIDIA_TF32_OVERRIDE:-}")
else
  unset NVIDIA_TF32_OVERRIDE
fi
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=$(quote "${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}")
export EXPORT_ONNX=False
export OBJECT_GEOMETRY_MODE=mesh
export HOLOSOMA_OBJECT_SPAWN_MODE=single_slot_multi_urdf
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=$(quote "${HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK}")
export TORCH_DIST_TIMEOUT_SEC=3600
export TORCH_DIST_BACKEND=$(quote "${TORCH_DIST_BACKEND}")
export MAX_RESTARTS=0
export NCCL_SOCKET_IFNAME=$(quote "${NCCL_SOCKET_IFNAME}")
export GLOO_SOCKET_IFNAME=$(quote "${GLOO_SOCKET_IFNAME}")
export NCCL_IB_DISABLE=$(quote "${NCCL_IB_DISABLE}")
export NCCL_DEBUG=$(quote "${NCCL_DEBUG}")
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
export PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=301989888
export PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY=436207616
export PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=134217728
export PHYSX_GPU_COLLISION_STACK_SIZE=268435456
if ! ip link show "\${NCCL_SOCKET_IFNAME}" >/dev/null 2>&1; then
  echo "[ERROR][${node}] NCCL_SOCKET_IFNAME=\${NCCL_SOCKET_IFNAME} not found" >&2
  ip -br link >&2 || true
  exit 1
fi
echo "[INFO][${node}] session=${SESSION} node_rank=${node_rank}/8 nproc=8 per_gpu_envs=${PER_GPU_ENVS} total_envs=${TOTAL_NUM_ENVS}"
echo "[INFO][${node}] master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[INFO][${node}] bank=${BANK_ROOT}"
echo "[INFO][${node}] contact_root=${CONTACT_ROOT}"
echo "[INFO][${node}] resume_from_box=\${RESUME_FROM_BOX}"
echo "[INFO][${node}] resume_checkpoint=\${RESUME_CKPT:-}"
echo "[INFO][${node}] wandb_resume_run_id=\${RESUME_WANDB_RUN_ID:-} wandb_resume_mode=\${WANDB_RESUME_MODE:-}"
echo "[INFO][${node}] policy_init_checkpoint=\${POLICY_INIT_CKPT:-}"
echo "[INFO][${node}] actor_hidden_dims=\${STUDENT_ACTOR_HIDDEN_DIMS}"
echo "[INFO][${node}] num_mini_batches=\${NUM_MINI_BATCHES}"
echo "[INFO][${node}] num_learning_epochs=\${NUM_LEARNING_EPOCHS}"
echo "[INFO][${node}] skip_grad_finite_check=\${HOLOSOMA_SKIP_GRAD_FINITE_CHECK}"
echo "[INFO][${node}] skip_loss_finite_check=\${HOLOSOMA_SKIP_LOSS_FINITE_CHECK}"
echo "[INFO][${node}] skip_loss_dict_accumulation=\${HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION}"
echo "[INFO][${node}] debug_actor=\${HOLOSOMA_DEBUG_ACTOR}"
echo "[INFO][${node}] debug_grad_reduce=\${HOLOSOMA_DEBUG_GRAD_REDUCE}"
echo "[INFO][${node}] rank_visible_devices=\${HOLOSOMA_RANK_VISIBLE_DEVICES}"
echo "[INFO][${node}] gloo_grad_reduce=\${HOLOSOMA_GLOO_GRAD_REDUCE}"
echo "[INFO][${node}] gloo_barrier=\${HOLOSOMA_GLOO_BARRIER}"
echo "[INFO][${node}] gloo_small_collectives=\${HOLOSOMA_GLOO_SMALL_COLLECTIVES}"
echo "[INFO][${node}] hierarchical_grad_reduce=\${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE}"
echo "[INFO][${node}] hierarchical_grad_reduce_cpu_leader=\${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER}"
echo "[INFO][${node}] skip_wandb_checkpoint_upload=\${HOLOSOMA_SKIP_WANDB_CHECKPOINT_UPLOAD}"
echo "[INFO][${node}] sync_after_grad_allreduce=\${HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE}"
echo "[INFO][${node}] sync_after_optimizer_step=\${HOLOSOMA_SYNC_AFTER_OPTIMIZER_STEP}"
echo "[INFO][${node}] sync_after_microbatch_forward=\${HOLOSOMA_SYNC_AFTER_MICROBATCH_FORWARD}"
echo "[INFO][${node}] debug_microbatch_all=\${HOLOSOMA_DEBUG_MICROBATCH_ALL}"
echo "[INFO][${node}] dagger_supervised_only=\${HOLOSOMA_DAGGER_SUPERVISED_ONLY}"
echo "[INFO][${node}] dagger_supervised_actor_only_step=\${HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP}"
echo "[INFO][${node}] supervised_actor_microbatch=\${HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH}"
echo "[INFO][${node}] supervised_actor_stream_backward=\${HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD}"
echo "[INFO][${node}] skip_critic_weight_sync=\${HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC}"
echo "[INFO][${node}] nvidia_tf32_override=\${NVIDIA_TF32_OVERRIDE:-}"
echo "[INFO][${node}] offline_contact_region_names=\${OFFLINE_CONTACT_REGION_NAMES}"
echo "[INFO][${node}] torch_dist_backend=\${TORCH_DIST_BACKEND}"
echo "[INFO][${node}] nccl_if=\${NCCL_SOCKET_IFNAME} gloo_if=\${GLOO_SOCKET_IFNAME} nccl_ib_disable=\${NCCL_IB_DISABLE}"
echo "[INFO][${node}] nccl_lib_dir=\${NCCL_LIB_DIR}"
"\${PYTHON_BIN}" - <<'PY'
import os
import subprocess
import torch

print("[INFO] torch_version=" + str(torch.__version__))
print("[INFO] torch_cuda=" + str(torch.version.cuda))
print("[INFO] torch_nccl=" + str(torch.cuda.nccl.version()))
print("[INFO] cuda_device_count=" + str(torch.cuda.device_count()))
lib = os.environ.get("NCCL_LIB_DIR", "") + "/libnccl.so.2"
try:
    print("[INFO] " + subprocess.check_output(["sha256sum", lib], text=True).strip())
    text = subprocess.check_output(["strings", lib], text=True, errors="ignore")
    for line in text.splitlines():
        if "NCCL version" in line:
            print("[INFO] " + line)
            break
except Exception as exc:
    print("[WARN] failed to inspect NCCL runtime lib: " + repr(exc))
PY
TRAIN_EXTRA_ARGS=()
if [[ -n "\${RESUME_WANDB_RUN_ID}" ]]; then
  TRAIN_EXTRA_ARGS+=("--logger.entity=\${WANDB_ENTITY}")
  TRAIN_EXTRA_ARGS+=("--logger.id=\${RESUME_WANDB_RUN_ID}")
  TRAIN_EXTRA_ARGS+=("--logger.resume=\${WANDB_RESUME_MODE}")
fi
exec bash ./distill_as_button.sh contact-aware wandb://zihanw22/carry-any/bcleb5oi/model_58000.pt "\${TRAIN_EXTRA_ARGS[@]}"
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
  if [[ $(quote "${RESTART}") == "1" ]]; then
    tmux kill-session -t $(quote "${SESSION}")
  else
    echo "[ERROR][${node}] tmux session already exists: ${SESSION}" >&2
    exit 1
  fi
fi
tmux new-session -d -s $(quote "${SESSION}") "bash $(quote "${run_script}") 2>&1 | tee $(quote "${log_file}")"
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
cd $(quote "${REPO}")
echo "===== ${node} ====="
tmux has-session -t $(quote "${SESSION}") 2>/dev/null && tmux list-sessions | grep -F $(quote "${SESSION}") || echo "tmux:${SESSION}:not-running"
latest_log=\$(ls -1t $(quote "${LOG_DIR}")/node_*_${node}.log 2>/dev/null | head -1 || true)
if [[ -n "\${latest_log}" ]]; then
  echo "log=\${latest_log}"
  tail -60 "\${latest_log}"
fi
EOF
)
  remote_run "${node}" "${cmd}"
}

stop_node() {
  local node="$1"
  remote_run "${node}" "tmux has-session -t $(quote "${SESSION}") 2>/dev/null && tmux kill-session -t $(quote "${SESSION}") && echo stopped:${SESSION} || echo not-running:${SESSION}"
}

stop_old_single_node_runs() {
  for node in 10.99.1.134 10.99.0.116; do
    remote_run "${node}" "tmux kill-session -t clean79_no_contact_award_20260622_2100_r10_h2048_e64_noonnx 2>/dev/null || true; tmux kill-session -t clean79_contact_award_20260622_2100_r10_h2048_e64_noonnx 2>/dev/null || true"
  done
}

sync_all_nodes() {
  local pids=()
  local status=0
  for node in "${NODE_LIST[@]}"; do
    echo "[INFO] sync ${node}"
    sync_node "${node}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if (( status != 0 )); then
    echo "[ERROR] One or more node syncs failed; refusing to launch." >&2
    exit 1
  fi
}

validate_config
case "${ACTION}" in
  sync)
    sync_all_nodes
    ;;
  launch)
    stop_old_single_node_runs
    sync_all_nodes
    rank=0
    for node in "${NODE_LIST[@]}"; do
      echo "[INFO] launch ${node} rank=${rank}"
      launch_node "${node}" "${rank}"
      rank=$((rank + 1))
    done
    ;;
  status)
    for node in "${NODE_LIST[@]}"; do
      status_node "${node}" || true
    done
    ;;
  stop)
    for node in "${NODE_LIST[@]}"; do
      stop_node "${node}" || true
    done
    ;;
  *)
    echo "Usage: $0 {sync|launch|status|stop}" >&2
    exit 2
    ;;
esac
