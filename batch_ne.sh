#!/usr/bin/env bash
set -euo pipefail

# Prepare and launch the 51-clip convex-hull AS solid distillation run across
# the 8xL40S nodes listed below. This script runs from one control node and
# starts one tmux session per node.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "${SCRIPT_DIR}"

ensure_training_python() {
  # Resolve once on the controller, then forward the exact absolute
  # interpreter to every node.  Per-node fallback selection would permit a
  # heterogeneous Python/PyTorch runtime even when the source snapshot agrees.
  export HOLOSOMA_PYTHON_PROFILE=hssim
  source "${SCRIPT_DIR}/scripts/gpu_launch_defaults.sh"
}

# Bash arithmetic is signed and machine-width.  Validate every caller-provided
# integer by canonical decimal string order before it can reach (( ... )); a
# huge value such as 2^64+k must not wrap to a plausible GPU count or timeout.
canonical_uint_at_most() {
  local value="$1" maximum="$2"
  local LC_ALL=C
  [[ "${value}" =~ ^(0|[1-9][0-9]*)$ \
      && "${maximum}" =~ ^(0|[1-9][0-9]*)$ ]] || return 1
  if (( ${#value} < ${#maximum} )); then
    return 0
  fi
  if (( ${#value} > ${#maximum} )); then
    return 1
  fi
  [[ "${value}" == "${maximum}" || "${value}" < "${maximum}" ]]
}

canonical_positive_uint_at_most() {
  local value="$1" maximum="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] \
    && canonical_uint_at_most "${value}" "${maximum}"
}

normalize_bool01() {
  local name="$1"
  local value="$2"
  case "$(echo "${value}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      echo 1
      ;;
    0|false|no|off|"")
      echo 0
      ;;
    *)
      echo "[ERROR] ${name} must be a boolean. Got: ${value}" >&2
      return 2
      ;;
  esac
}

controller_epoch_now() {
  local epoch
  if ! epoch=$(date +%s) \
      || ! canonical_positive_uint_at_most "${epoch}" "${MAX_SAFE_EPOCH}"; then
    echo "[ERROR] Controller clock returned a non-canonical or unsafe epoch: ${epoch:-<empty>}" >&2
    return 2
  fi
  printf '%s\n' "${epoch}"
}

controller_monotonic_seconds() {
  local uptime_raw ignored monotonic
  if [[ ! -r /proc/uptime ]] \
      || ! IFS=' ' read -r uptime_raw ignored < /proc/uptime; then
    echo "[ERROR] Cannot read the controller monotonic clock from /proc/uptime." >&2
    return 2
  fi
  monotonic=${uptime_raw%%.*}
  if ! canonical_uint_at_most "${monotonic}" "${MAX_SAFE_EPOCH}"; then
    echo "[ERROR] Controller monotonic clock is malformed: ${uptime_raw:-<empty>}" >&2
    return 2
  fi
  printf '%s\n' "${monotonic}"
}

readonly MAX_SIGNED_32=2147483647
readonly MAX_NPROC_PER_NODE=1024
readonly MAX_TOTAL_GPUS=65536
readonly MAX_STATUS_SECONDS=31536000
readonly MAX_LAUNCH_SECONDS=604800
readonly MAX_SAFE_EPOCH=999999999999999999

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
  bash batch_ne.sh prepare     # install an isolated source snapshot + prepare data
  bash batch_ne.sh launch      # start multi-node training in tmux
  bash batch_ne.sh all         # prepare, then launch
  bash batch_ne.sh status      # show tmux/log status on every node
  bash batch_ne.sh stop        # kill only this script's tmux session

Useful env:
  NODES="node0 node1 ..."      override node list
  REMOTE_REPO=/home/ubuntu/FAR/holosoma  node-local asset repo; source is never modified
  REMOTE_RUN_ROOT=/home/ubuntu/FAR/holosoma_runs  isolated content-addressed source roots
  LOGGER_BASE_DIR=<REMOTE_RUN_ROOT>/training_logs  cross-node writable experiment/log root
  REMOTE_DATA_PACKAGE_CACHE=/home/ubuntu/FAR/holosoma_runs/.data-packages
                               node-local content-addressed fallback cache for custom bank tar files
  SOURCE_SNAPSHOT_CACHE=/tmp/holosoma-run-snapshots-$USER  local archive cache
  SOURCE_SNAPSHOT_ID=src-<sha256>  reuse a previously built local snapshot archive
  PYTHON_RUNTIME_SITEPACKAGES=<path>  immutable content-addressed Python overlay shared by every node
  PYTHON_RUNTIME_MANIFEST_SHA256=<sha256>  required exact-tree overlay manifest digest
  PYTHON_RUNTIME_ARCHIVE=<path>  controller-local sealed runtime archive (prepare/all only)
  PYTHON_RUNTIME_ARCHIVE_SHA256=<sha256>  exact archive transport digest
  HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY=1  fail unless the exact overlay is configured
  SESSION=distill_as_ch51_64gpu
  PER_GPU_ENVS=2048            1024 minimum recommended; try 4096 if stable
  TRAINING_SEED=<0..4294967295>  optional base seed; rank r uses base+r
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  MASTER_ADDR=<node0>          default first node
  MASTER_PORT=29651
  NCCL_SOCKET_IFNAME=enp135s0  network interface for NCCL
  GLOO_SOCKET_IFNAME=enp135s0  network interface for Gloo
  NCCL_IB_DISABLE=1            force TCP socket path on these nodes
  NCCL_SOCKET_FAMILY=AF_INET   force IPv4 on the private VPC interface
  NCCL_LIB_DIR=<path>          prepend runtime NCCL library directory
  NCCL_LIB_SHA256=<sha256>     required for NCCL backend or hierarchical local NCCL; verifies libnccl.so.2
  TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300
  TORCH_DIST_BACKEND=gloo      use pure Gloo for torch distributed
  TORCH_DIST_TIMEOUT_SEC=3600  process-group operation timeout
  HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC=300  hierarchical NCCL/Gloo subgroup timeout
  MAX_RESTARTS=0               torchrun worker restart limit
  TORCH_NCCL_ENABLE_MONITORING=1
  TORCH_NCCL_TRACE_BUFFER_SIZE=65536
  TORCH_NCCL_DUMP_ON_TIMEOUT=1
  NCCL_SOCKET_RETRY_CNT=34
  NCCL_SOCKET_RETRY_SLEEP_MSEC=100
  HOLOSOMA_GLOO_GRAD_REDUCE=1
  HOLOSOMA_GLOO_BARRIER=1
  HOLOSOMA_GLOO_SMALL_COLLECTIVES=1
  HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=0  two-stage intra/inter-node gradient reduction
  HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=0  use node leaders for eligible integral verdict/control tensors only; floating reductions stay flat Gloo
  HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER=0  use CPU/Gloo for inter-node leaders
  HOLOSOMA_RANK_VISIBLE_DEVICES=0  expose only LOCAL_RANK's GPU to each worker
  HOLOSOMA_RANK_LOCAL_CPU_AFFINITY=0  best-effort GPU-NUMA-local CPU binding per torchrun child
  HOLOSOMA_CARB_TASKING_THREAD_COUNT=<unset>  optional explicit Carb worker count; unset preserves the runtime default
  HOLOSOMA_OBJECT_COLLIDER_TYPE=convex_decomposition  canonical PhysX object-collider cooking mode (convex_decomposition or convex_hull)
  HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=0  preserve the validated AS object sleep/reporting contract; object-filtered sensors use robot reporters
  HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE=1  flush local CUDA streams before flat NCCL gradient reduction
  HOLOSOMA_DEBUG_HEARTBEAT_VERBOSE=1  log every rollout env.step boundary for canaries
  FIXED_BC_EVAL_LOG_INTERVAL=100   fixed teacher-labeled BC evaluation cadence
  FIXED_BC_GUARD_ENABLED=True      fail closed on sustained fixed-set student forgetting
  FIXED_BC_GUARD_REFERENCE_END_EPOCH=600  freeze the pure-BC reference window here
  FIXED_BC_GUARD_MAX_REFERENCE_RATIO=2.0  relative fixed-BC regression ceiling
  FIXED_BC_GUARD_ABSOLUTE_MAX_MU_MSE=0.160 absolute fixed-BC regression ceiling
  FIXED_BC_GUARD_START_EPOCH=<dagger-end> begin consecutive breach counting here
  FIXED_BC_GUARD_CONSECUTIVE_EVALS=3  diagnostic checkpoint + fail after this many breaches
  HOLOSOMA_MOTION_METRICS_INTERVAL=16  motion-metric refresh cadence when curriculum permits
  HOLOSOMA_COLLECTION_PROFILE_CANARY=0  enable bounded collection hot-path timing (diagnostic only)
  HOLOSOMA_COLLECTION_PROFILE_SYNC_CUDA=0  synchronize CUDA at timing boundaries (intrusive; canary only)
  HOLOSOMA_COLLECTION_PROFILE_INTERVAL=1  emit one timing summary every N learning iterations
  SAVE_INTERVAL=1000           mandatory formal-AS checkpoint save/upload interval
  SKIP_GIT_PULL=1             deprecated compatibility flag; isolated prepare never runs git
  SKIP_NODE_HEALTH_CHECK=1    skip optional repo/logger/data checks; selected-GPU idle gates remain mandatory
  STATUS_STALE_SECONDS=900    status fails if the recorded training iteration stops advancing
  LAUNCH_STARTUP_TIMEOUT_SECONDS=900  bounded wait for every node to reach the torchrun boundary
  LAUNCH_STARTUP_PROBE_TIMEOUT_SECONDS=30  per-round SSH health-probe bound
  LAUNCH_STARTUP_POLL_SECONDS=5  delay between startup-health rounds
  LAUNCH_STARTUP_STABILITY_SECONDS=10  require ready sessions to remain healthy for this long
  LAUNCH_CLEANUP_TIMEOUT_SECONDS=30  per-command controller bound for failed-launch cleanup
  LAUNCH_LOCK_TIMEOUT_SECONDS=10  remote flock acquisition bound for launch lifecycle mutations
  LAUNCH_CONTROL_TIMEOUT_SECONDS=120  short intent/reservation/tmux control SSH bound
  LAUNCH_PREFLIGHT_TIMEOUT_SECONDS=900  expensive source/runtime/data preflight SSH bound
  LEGACY_STOP_EXPECTED_SNAPSHOT_ID=src-<sha256>  required exact snapshot for a pre-atomic stop
  LEGACY_STOP_EXPECTED_TOKEN=<sha256>  required exact shared token for a pre-atomic stop
  LEGACY_STOP_EXPECTED_EPOCH=<epoch>  required exact shared launch epoch for a pre-atomic stop
  LEGACY_STOP_EXPECTED_RUN_STAMP=<stamp>  required exact run stamp for a pre-atomic stop
  LEGACY_STOP_EXPECTED_TARGET=<iteration>  required exact target for a pre-atomic stop
  CH_BANK_NAME=as_realmesh67000_finalpos_convexsurface51_convexhull
  NFS_CORL_BANK=/nfs/zzzihanw/prism-debug/<bank>  use a custom NFS bank directory or tar
  LOCAL_BANK_NAME=<name>       local data/ds_as_data bank name for custom bank
  EXPECTED_CLIP_COUNT=<n>      expected clip count for custom bank copy
  KEEP_BACKUP=0                remove existing local copied bank instead of keeping .bak
  PREPARE_COPY_SCRIPT=cp_corl  use cp_corl for custom NFS_CORL_BANK, cp_ch for ch51
  PREPARE_DATA=0               install/verify immutable source snapshot only
  CORL_SOLID80_BANK_NAME=<name>  source bank name used by distill_as_button_solid.sh
  SOLID_CLIP_LIST=<file>      optional allowlist for distill_as_button_solid.sh
  SOLID_TARGET_BANK_NAME=<name> optional generated filtered-bank name
  DISTILL_AS_ENTRYPOINT=distill_as_button_solid.sh
                               repo-local AS wrapper selected from the launcher's explicit allowlist
  DISTILL_AS_FORMAL_FRESH=0|1  dual-button opt-in: reject every training-resume/policy-init alias
  ENABLE_OFFLINE_CONTACT_GUIDANCE=False  disable offline contact guidance reward
  RESUME_FROM_BOX=0            default; set 1 only with an actor-architecture-compatible box checkpoint
  BOX_POLICY_INIT_REF=<ckpt>   box policy initializer; control-local .pt files are verified and staged per node
  BOX_POLICY_INIT_EXPECTED_SHA256=<sha256>  optional exact initializer digest; required by the terminal-source gate
  HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET=<n>  require a local authenticated terminal checkpoint for target n
  BOX_POLICY_INIT_CONTROL_CACHE_ROOT=<dir>  private 0700 controller cache root for the terminal-source gate
  BOX_POLICY_INIT_EXPECTED_WORLD_SIZE=<n>  exact source topology required by the terminal-source gate
  BOX_POLICY_INIT_EXPECTED_WANDB_RUN_PATH=<entity/project/run>  exact source run identity
  BOX_POLICY_INIT_EXPECTED_SOURCE_SNAPSHOT_ID=<src-sha256>  exact source snapshot embedded in the initializer
  BOX_POLICY_INIT_REQUIRE_FRESH_SOURCE=0|1  must be 1 when the terminal-source gate is enabled
  BOX_POLICY_INIT_CACHE_ROOT=~/.cache/holosoma/checkpoints
  RESUME_TRAINING_CKPT=<ckpt>  curriculum-correct training resume; not bitwise trajectory continuation
  TARGET_LEARNING_ITERATION=40000  absolute final PPO iteration target (not an additional count)
  ALLOW_FRESH_CURRICULUM_RESUME=0  require saved rank-local AS sampler/failure/clip aggregate state
  ALLOW_NONDETERMINISTIC_RNG_RESUME=0  require saved rank-local Python/NumPy/torch RNG streams
  ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME=0  fail closed if fixed-BC evaluation provenance is missing
  ALLOW_RUNTIME_DRIFT_ON_RESUME=0  fail closed if runtime provenance differs from the checkpoint
  TEACHER_CHECKPOINT=wandb://.../model_N.pt  exact teacher artifact shared by every node
  MOTION_GENERATOR_TEACHER_EXPECTED_SHA256=<sha256>  exact generator identity required for legacy motion banks
  REQUIRE_MOTION_GENERATOR_TEACHER_MATCH=0|1  default 1; require the distillation teacher to equal the motion generator
  RESUME_WANDB_RUN_ID=<id>     optional same-run W&B resume id for RESUME_TRAINING_CKPT
  FRESH_WANDB_RUN_ID=<id>      fresh W&B id pre-bound by the mandatory vis/replay gate
  REPLAY_PREFLIGHT_MANIFEST=<json>  local replay manifest uploaded to FRESH_WANDB_RUN_ID
  REPLAY_PREFLIGHT_MANIFEST_SHA256=<sha256>  exact replay manifest digest
  REPLAY_PREFLIGHT_REQUIRED_VERSION=1|2  optional exact schema; formal-fresh dual forces v2
  WANDB_RESUME_MODE=must       W&B resume mode when RESUME_WANDB_RUN_ID is set
  WANDB_BASE_URL=https://api.wandb.ai  fixed cloud API endpoint for this scientific launcher
  WANDB_INIT_TIMEOUT=120       bounded W&B startup timeout in seconds (1..3600)
  WANDB_CONSOLE=off            fixed; tee is the authoritative, unwrapped startup/training log
  HOLOSOMA_REQUIRE_WANDB_RUN=1 fail closed unless rank zero creates an active online W&B run
  STUDENT_ACTOR_HIDDEN_DIMS='[2048,1024,512,256,128]'  default for new runs; inferred/required for training resume
  STUDENT_POLICY_TYPE=mlp|flow  actor implementation; forwarded identically to every node
  STUDENT_FLOW_STEPS=4        flow integration steps (flow policy only)
  STUDENT_FLOW_TRAIN_NOISE_STD=1.0
  STUDENT_FLOW_TIME_EPSILON=1e-4
  STUDENT_FLOW_INFERENCE_NOISE_STD=0.0
  STUDENT_ACTOR_INPUTS="[...]" optional explicit ordered actor observation groups
  STUDENT_PROPRIO_HISTORY_LENGTH=<n> optional actor proprio history override
  STUDENT_ACTION_HISTORY_LENGTH=<n> optional actor action history override
  CRITIC_PROPRIO_HISTORY_LENGTH=<n> optional critic proprio history override
  BOX_POLICY_INIT_ACTOR_HIDDEN_DIMS='[512,256,128]'  required architecture for RESUME_FROM_BOX=1
  ACTOR_LR=1e-3                actor optimizer learning rate
  CRITIC_LR=1e-3               critic optimizer learning rate
  PPO_LR_SCHEDULE=adaptive     actor LR controller: adaptive or fixed
  PPO_DESIRED_KL=0.01          positive policy-KL target for adaptive LR
  ACTOR_MIN_LR=<min(ACTOR_LR,1e-5)>  actor LR lower bound
  ACTOR_MAX_LR=<max(ACTOR_LR,1e-2)>  actor LR upper bound
  CRITIC_MIN_LR=<min(CRITIC_LR,1e-5)>  critic LR lower bound
  CRITIC_MAX_LR=<max(CRITIC_LR,1e-2)>  critic LR upper bound
  PPO_SCHEDULE_STEP_EPOCHS=700 new runs increase PPO by 0.1 every 700 iters to 0.7, retaining BC=0.3
  DAGGER_REPLAY_ENABLED=False  bounded deterministic rank-local DAgger replay (pure BC only)
  DAGGER_REPLAY_CAPACITY=512   maximum authenticated replay rows retained per rank
  DAGGER_REPLAY_BATCH_SIZE=512 replay rows sampled per actor update
  DAGGER_REPLAY_FRACTION=0.5   replay share in the current/replay BC mixture (strictly between 0 and 1)
  DAGGER_REPLAY_SEED=0         non-negative independent rank-local replay RNG base seed
  DAGGER_MATCH_STD=False      keep student exploration noise independent from teacher std by default
  PPO_START_NOISE_STD=0.1     cap early-rollout std while PPO coefficient is <= 0.1
  START_AT_TIMESTEP_ZERO_PROB=0.2  reset-at-zero curriculum start probability
  START_AT_TIMESTEP_ZERO_PROB_END=1.0  reset-at-zero curriculum final probability
  START_AT_TIMESTEP_ZERO_PROB_START_ITER=2500  curriculum start (short runs default to 0)
  START_AT_TIMESTEP_ZERO_PROB_END_ITER=<target-1>  default reaches its end on the final fresh rollout
                               explicit fresh/policy-init end must be < target; exact full resume may preserve =target
  FREEZE_AT_TIMESTEP_ZERO_PROB=0.0  freeze-at-zero curriculum start probability
  FREEZE_AT_TIMESTEP_ZERO_PROB_END=0.0  freeze-at-zero curriculum final probability
  FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=2500  curriculum start (short runs default to 0)
  FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=<target-1>  default reaches its end on the final fresh rollout
                               explicit fresh/policy-init end must be < target; exact full resume may preserve =target
  UNIFORM_T1_WINDOW_TARGET_SAMPLE_FRAC=<0..1> optional overall reset mass for the contact T1 window
  STUDENT_MOTION_END_MODE=episodic  terminate/reset at every motion end; continuing requires explicit opt-in
  CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION=True|False  explicit override; resume otherwise infers checkpoint value
  EXPORT_ONNX=False           default off for large distributed runs; explicit True is honored
  NUM_MINI_BATCHES=64
  NUM_LEARNING_EPOCHS=1
  HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY=0  set 1 together with explicit NUM_MINI_BATCHES=16 for the algorithm-changing A/B
                               example dry-run: HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY=1 NUM_MINI_BATCHES=16 DRY_RUN=1 bash batch_ne.sh launch
  RESTART=0                    required; stop/verify first, transactional in-place restart is unsupported
  DRY_RUN=1                    print remote commands only
EOF
}

if (( $# > 1 )); then
  echo "[ERROR] batch_ne.sh accepts exactly one action and no additional positional arguments." >&2
  usage >&2
  exit 2
fi
ACTION=${1:-all}
CONTROL_ONLY_ACTION=0
case "${ACTION}" in
  -h|--help|help)
    usage
    exit 0
    ;;
  prepare|launch|all)
    ;;
  status|stop)
    CONTROL_ONLY_ACTION=1
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [[ -n "${NODES:-}" ]]; then
  # Split only on caller-provided whitespace.  Array assignment with an
  # unquoted expansion also performs pathname expansion, so NODES='*' could
  # accidentally turn every controller working-tree entry into an SSH target.
  _nodes_text=${NODES//$'\n'/ }
  read -r -a NODE_LIST <<< "${_nodes_text}"
  unset _nodes_text
else
  NODE_LIST=("${DEFAULT_NODES[@]}")
fi

if [[ "${#NODE_LIST[@]}" -lt 1 ]]; then
  echo "[ERROR] Empty node list." >&2
  exit 2
fi
declare -A _SEEN_NODES=()
for _node in "${NODE_LIST[@]}"; do
  # Node names are embedded in active/tombstone/log basenames.  Keep enough
  # NAME_MAX headroom for the two 64-hex ownership fields and incoming suffix.
  if [[ ! "${_node}" =~ ^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$ ]]; then
    echo "[ERROR] Unsafe node identifier in NODES: ${_node}" >&2
    exit 2
  fi
  if [[ -n "${_SEEN_NODES[${_node}]+x}" ]]; then
    echo "[ERROR] Duplicate node in NODES: ${_node}. Each node rank must identify a unique host." >&2
    exit 2
  fi
  _SEEN_NODES["${_node}"]=1
done
unset _node _SEEN_NODES

REMOTE_REPO=${REMOTE_REPO:-/home/ubuntu/FAR/holosoma}
REMOTE_RUN_ROOT=${REMOTE_RUN_ROOT:-/home/ubuntu/FAR/holosoma_runs}
LOGGER_BASE_DIR=${LOGGER_BASE_DIR:-${REMOTE_RUN_ROOT}/training_logs}
REMOTE_DATA_PACKAGE_CACHE=${REMOTE_DATA_PACKAGE_CACHE:-${REMOTE_RUN_ROOT}/.data-packages}
SOURCE_SNAPSHOT_CACHE=${SOURCE_SNAPSHOT_CACHE:-${TMPDIR:-/tmp}/holosoma-run-snapshots-${USER:-unknown}}
SOURCE_SNAPSHOT_ID=${SOURCE_SNAPSHOT_ID:-}
SOURCE_SNAPSHOT_ARCHIVE=${SOURCE_SNAPSHOT_ARCHIVE:-}
SOURCE_SNAPSHOT_ARCHIVE_SHA256=${SOURCE_SNAPSHOT_ARCHIVE_SHA256:-}
SOURCE_MANIFEST_SHA256=${SOURCE_MANIFEST_SHA256:-}
PYTHON_RUNTIME_SITEPACKAGES=${PYTHON_RUNTIME_SITEPACKAGES:-}
PYTHON_RUNTIME_MANIFEST_SHA256=${PYTHON_RUNTIME_MANIFEST_SHA256:-}
PYTHON_RUNTIME_ARCHIVE=${PYTHON_RUNTIME_ARCHIVE:-}
PYTHON_RUNTIME_ARCHIVE_SHA256=${PYTHON_RUNTIME_ARCHIVE_SHA256:-}
PYTHON_RUNTIME_ARCHIVE_SIZE=""
PYTHON_RUNTIME_CONTROLLER_TRANSFER_ROOT=""
PYTHON_RUNTIME_CONTROLLER_TRANSFER_ARCHIVE=""
PYTHON_RUNTIME_CONTROLLER_TRANSFER_FINGERPRINT=""
HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY=${HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY:-1}
RUN_REPO=""
DISTILL_AS_ENTRYPOINT=${DISTILL_AS_ENTRYPOINT:-distill_as_button_solid.sh}
DISTILL_AS_ENTRYPOINT_PATH=""
DISTILL_AS_ENTRYPOINT_SHA256=""
DISTILL_AS_FORMAL_FRESH=${DISTILL_AS_FORMAL_FRESH:-0}
if (( CONTROL_ONLY_ACTION == 0 )); then
  # This value becomes an archive member, a shell command, and durable launch
  # metadata.  Accept only one bare root-level filename from the deliberately
  # small training-entrypoint allowlist; paths, shell fragments, and arbitrary
  # repo scripts are never launchable through this knob.
  case "${DISTILL_AS_ENTRYPOINT}" in
    distill_as_button_solid.sh|distill_as_dual_button_solid.sh)
      ;;
    *)
      echo "[ERROR] DISTILL_AS_ENTRYPOINT must be one allowlisted bare repo-local filename: distill_as_button_solid.sh or distill_as_dual_button_solid.sh. Got: ${DISTILL_AS_ENTRYPOINT@Q}" >&2
      exit 2
      ;;
  esac
  DUAL_BUTTON_STUDENT_ACTOR_INPUTS_CONTRACT="['actor_obs_root_contact_aware','actor_obs_pickup_button','actor_obs_drop_button','actor_obs_proprio_with_actions_no_linvel']"
  if [[ "${DISTILL_AS_ENTRYPOINT}" == distill_as_dual_button_solid.sh \
        && -n "${STUDENT_ACTOR_INPUTS+x}" \
        && "${STUDENT_ACTOR_INPUTS//[[:space:]]/}" \
          != "${DUAL_BUTTON_STUDENT_ACTOR_INPUTS_CONTRACT}" ]]; then
    echo "[ERROR] DISTILL_AS_ENTRYPOINT=distill_as_dual_button_solid.sh requires the exact ordered 95D STUDENT_ACTOR_INPUTS contract: ${DUAL_BUTTON_STUDENT_ACTOR_INPUTS_CONTRACT}" >&2
    exit 2
  fi
  if ! DISTILL_AS_FORMAL_FRESH=$(normalize_bool01 \
      DISTILL_AS_FORMAL_FRESH "${DISTILL_AS_FORMAL_FRESH}"); then
    exit 2
  fi
  if [[ "${DISTILL_AS_FORMAL_FRESH}" == 1 \
        && "${DISTILL_AS_ENTRYPOINT}" != distill_as_dual_button_solid.sh ]]; then
    echo "[ERROR] DISTILL_AS_FORMAL_FRESH=1 is defined only for DISTILL_AS_ENTRYPOINT=distill_as_dual_button_solid.sh." >&2
    exit 2
  fi
  if [[ "${DISTILL_AS_FORMAL_FRESH}" == 1 ]]; then
    # Reject caller intent before aliases are canonicalized, defaults are
    # materialized, Python is sourced, a snapshot is built, or a remote action
    # is attempted.  The delegated dual wrapper repeats this gate so direct
    # invocations and sealed node controls remain independently fail-closed.
    for _formal_fresh_bool_alias in \
        RESUME_FROM_BOX RESUME_FROM_PREVIOUS WANDB_RESUME_SAME_RUN; do
      if ! _formal_fresh_bool_value=$(normalize_bool01 \
          "${_formal_fresh_bool_alias}" "${!_formal_fresh_bool_alias:-0}"); then
        exit 2
      fi
      if [[ "${_formal_fresh_bool_value}" != 0 ]]; then
        echo "[ERROR] DISTILL_AS_FORMAL_FRESH=1 requires ${_formal_fresh_bool_alias}=0." >&2
        exit 2
      fi
    done
    for _formal_fresh_checkpoint_alias in \
        RESUME_TRAINING_CKPT RESUME_CKPT RESUME_CHECKPOINT RESUME_SOURCE_REF \
        RESUME_WANDB_RUN_ID RESUME_WANDB_ID WANDB_RUN_ID \
        POLICY_INIT_CKPT POLICY_INIT_CHECKPOINT POLICY_INIT_SOURCE_REF \
        BOX_POLICY_INIT_REF BOX_RESUME_CKPT RESUME_FROM_BOX_CKPT \
        DEFAULT_BOX_RESUME_CHECKPOINT DEFAULT_BOX_RESUME_RUN \
        DEFAULT_BOX_RESUME_MODEL_FILE \
        PREVIOUS_RESUME_CKPT RESUME_FROM_PREVIOUS_CKPT PREVIOUS_RESUME_RUN \
        PREVIOUS_RESUME_MODEL_FILE DEFAULT_PREVIOUS_RESUME_RUN \
        AS_POLICY_INIT_PROFILE AS_TRAINING_RESUME_REF \
        RESUME_SOURCE_EXPECTED_SHA256 POLICY_INIT_EXPECTED_SHA256 \
        BOX_POLICY_INIT_EXPECTED_SHA256 \
        HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET \
        BOX_POLICY_INIT_EXPECTED_WORLD_SIZE BOX_POLICY_INIT_EXPECTED_WANDB_RUN_PATH \
        BOX_POLICY_INIT_EXPECTED_SOURCE_SNAPSHOT_ID \
        RESUME_MODEL_FILE WANDB_MODEL_FILE RESUME_STEP; do
      if [[ -n "${!_formal_fresh_checkpoint_alias:-}" ]]; then
        echo "[ERROR] DISTILL_AS_FORMAL_FRESH=1 requires ${_formal_fresh_checkpoint_alias} to be empty/unset." >&2
        exit 2
      fi
    done
    if [[ -n "${STUDENT_PROPRIO_HISTORY_LENGTH+x}" \
          && "${STUDENT_PROPRIO_HISTORY_LENGTH}" != 1 ]]; then
      echo "[ERROR] DISTILL_AS_FORMAL_FRESH=1 requires STUDENT_PROPRIO_HISTORY_LENGTH=1 for the exact 95D actor." >&2
      exit 2
    fi
    if [[ -n "${CONTACT_AWARE_HISTORY_LENGTH+x}" \
          && "${CONTACT_AWARE_HISTORY_LENGTH}" != 1 ]]; then
      echo "[ERROR] DISTILL_AS_FORMAL_FRESH=1 requires CONTACT_AWARE_HISTORY_LENGTH=1 when supplied; longer history changes the actor dimension." >&2
      exit 2
    fi
    for _formal_fresh_history_bool_alias in \
        CONTACT_AWARE_HISTORY AS_CONTACT_AWARE_HISTORY; do
      if [[ -n "${!_formal_fresh_history_bool_alias+x}" ]]; then
        if ! _formal_fresh_history_bool_value=$(normalize_bool01 \
            "${_formal_fresh_history_bool_alias}" \
            "${!_formal_fresh_history_bool_alias}"); then
          exit 2
        fi
        if [[ "${_formal_fresh_history_bool_value}" != 0 ]]; then
          echo "[ERROR] DISTILL_AS_FORMAL_FRESH=1 requires ${_formal_fresh_history_bool_alias}=0; contact-aware history changes the exact 95D actor." >&2
          exit 2
        fi
      fi
    done
    # Make the controller-to-wrapper value explicit so a future source default
    # cannot silently turn the manifest's history-1/95D declaration into 455D.
    STUDENT_PROPRIO_HISTORY_LENGTH=1
    unset _formal_fresh_bool_alias _formal_fresh_bool_value
    unset _formal_fresh_checkpoint_alias
    unset _formal_fresh_history_bool_alias _formal_fresh_history_bool_value
  fi
else
  # Status/stop recover the durable identity from the active control script;
  # stale training-profile values must not block emergency control actions.
  DISTILL_AS_ENTRYPOINT=distill_as_button_solid.sh
  DISTILL_AS_FORMAL_FRESH=0
fi
SESSION=${SESSION:-distill_as_ch51_64gpu}
PER_GPU_ENVS=${PER_GPU_ENVS:-2048}
MIN_PER_GPU_ENVS=${MIN_PER_GPU_ENVS:-1024}
TRAINING_SEED=${TRAINING_SEED:-${SEED:-}}
OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
ACCEPT_EULA=${ACCEPT_EULA:-Y}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
NNODES=${NNODES:-${#NODE_LIST[@]}}
MASTER_ADDR=${MASTER_ADDR:-${NODE_LIST[0]}}
MASTER_PORT=${MASTER_PORT:-29651}
HOLOSOMA_PROVENANCE_MASTER_PORT=${HOLOSOMA_PROVENANCE_MASTER_PORT:-}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp135s0}
GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
NCCL_DEBUG=${NCCL_DEBUG:-WARN}
TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
TORCH_DIST_BACKEND=${TORCH_DIST_BACKEND:-gloo}
TORCH_DIST_TIMEOUT_SEC=${TORCH_DIST_TIMEOUT_SEC:-3600}
HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC=${HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC:-300}
MAX_RESTARTS=${MAX_RESTARTS:-0}
TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-1}
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-300}
TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-65536}
TORCH_NCCL_PROPAGATE_ERROR=${TORCH_NCCL_PROPAGATE_ERROR:-1}
TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG:-0}
TORCH_NCCL_ENABLE_TIMING=${TORCH_NCCL_ENABLE_TIMING:-0}
TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-0}
NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
NCCL_SOCKET_RETRY_CNT=${NCCL_SOCKET_RETRY_CNT:-34}
NCCL_SOCKET_RETRY_SLEEP_MSEC=${NCCL_SOCKET_RETRY_SLEEP_MSEC:-100}
NCCL_SOCKET_NTHREADS=${NCCL_SOCKET_NTHREADS:-2}
NCCL_NSOCKS_PERTHREAD=${NCCL_NSOCKS_PERTHREAD:-4}
NCCL_LIB_SHA256=${NCCL_LIB_SHA256:-}
if [[ -z "${NCCL_LIB_DIR+x}" ]]; then
  if [[ -n "${NCCL_LIB_SHA256}" ]]; then
    # A digest identifies the immutable runtime itself.  Resolve its default
    # path from that digest instead of silently selecting whichever NCCL wheel
    # happens to be installed in the mutable Conda environment on each node.
    NCCL_LIB_DIR="${REMOTE_RUN_ROOT}/.runtime/nccl/${NCCL_LIB_SHA256}"
  else
    NCCL_LIB_DIR=/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/nvidia/nccl/lib
  fi
fi
SKIP_GIT_PULL=${SKIP_GIT_PULL:-0}
SKIP_NODE_HEALTH_CHECK=${SKIP_NODE_HEALTH_CHECK:-0}
PREPARE_DATA=${PREPARE_DATA:-1}
STATUS_STALE_SECONDS=${STATUS_STALE_SECONDS:-900}
STATUS_STARTUP_GRACE_SECONDS=${STATUS_STARTUP_GRACE_SECONDS:-900}
STATUS_MAX_CLOCK_SKEW_SECONDS=${STATUS_MAX_CLOCK_SKEW_SECONDS:-300}
LAUNCH_STARTUP_TIMEOUT_SECONDS=${LAUNCH_STARTUP_TIMEOUT_SECONDS:-${STATUS_STARTUP_GRACE_SECONDS}}
LAUNCH_STARTUP_PROBE_TIMEOUT_SECONDS=${LAUNCH_STARTUP_PROBE_TIMEOUT_SECONDS:-30}
LAUNCH_STARTUP_POLL_SECONDS=${LAUNCH_STARTUP_POLL_SECONDS:-5}
LAUNCH_STARTUP_STABILITY_SECONDS=${LAUNCH_STARTUP_STABILITY_SECONDS:-10}
LAUNCH_CLEANUP_TIMEOUT_SECONDS=${LAUNCH_CLEANUP_TIMEOUT_SECONDS:-30}
LAUNCH_LOCK_TIMEOUT_SECONDS=${LAUNCH_LOCK_TIMEOUT_SECONDS:-10}
LAUNCH_CONTROL_TIMEOUT_SECONDS=${LAUNCH_CONTROL_TIMEOUT_SECONDS:-120}
LAUNCH_PREFLIGHT_TIMEOUT_SECONDS=${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS:-900}
CH_BANK_NAME=${CH_BANK_NAME:-as_realmesh67000_finalpos_convexsurface51_convexhull}
CORL_SOLID80_BANK_NAME_WAS_SET=${CORL_SOLID80_BANK_NAME+x}
CORL_SOLID80_BANK_NAME=${CORL_SOLID80_BANK_NAME:-${CH_BANK_NAME}}
NFS_CORL_BANK=${NFS_CORL_BANK:-}
LOCAL_BANK_NAME=${LOCAL_BANK_NAME:-}
CONTROL_CORL_PACKAGE_PATH=""
CONTROL_CORL_PACKAGE_SHA256=""
CONTROL_CORL_PACKAGE_SIZE=""
EXPECTED_CLIP_COUNT=${EXPECTED_CLIP_COUNT:-}
KEEP_BACKUP=${KEEP_BACKUP:-1}
PREPARE_COPY_SCRIPT=${PREPARE_COPY_SCRIPT:-}
SOLID_CLIP_LIST=${SOLID_CLIP_LIST:-}
SOLID_TARGET_BANK_NAME=${SOLID_TARGET_BANK_NAME:-}
SOLID_ALLOWED_OBJECT_CATEGORIES=${SOLID_ALLOWED_OBJECT_CATEGORIES:-}
SOLID_CONTACT_EXPORT_NAME=${SOLID_CONTACT_EXPORT_NAME:-contact_export_from_teacher_success133_final0p5}
MOTION_GENERATOR_TEACHER_EXPECTED_SHA256=${MOTION_GENERATOR_TEACHER_EXPECTED_SHA256:-}
REQUIRE_MOTION_GENERATOR_TEACHER_MATCH=${REQUIRE_MOTION_GENERATOR_TEACHER_MATCH:-1}
ENABLE_OFFLINE_CONTACT_GUIDANCE=${ENABLE_OFFLINE_CONTACT_GUIDANCE:-}
OMOMO_EXPECTED_TOTAL=${OMOMO_EXPECTED_TOTAL:-51}
RESUME_FROM_BOX_EXPECTED_TOTAL=${RESUME_FROM_BOX_EXPECTED_TOTAL:-${OMOMO_EXPECTED_TOTAL}}
RESTART=${RESTART:-0}
DRY_RUN=${DRY_RUN:-0}
if ! DRY_RUN=$(normalize_bool01 DRY_RUN "${DRY_RUN}"); then
  exit 2
fi
SSH_OPTS=${SSH_OPTS:-"-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
LEGACY_STOP_EXPECTED_SNAPSHOT_ID=${LEGACY_STOP_EXPECTED_SNAPSHOT_ID:-}
LEGACY_STOP_EXPECTED_TOKEN=${LEGACY_STOP_EXPECTED_TOKEN:-}
LEGACY_STOP_EXPECTED_EPOCH=${LEGACY_STOP_EXPECTED_EPOCH:-}
LEGACY_STOP_EXPECTED_RUN_STAMP=${LEGACY_STOP_EXPECTED_RUN_STAMP:-}
LEGACY_STOP_EXPECTED_TARGET=${LEGACY_STOP_EXPECTED_TARGET:-}
if (( CONTROL_ONLY_ACTION != 0 )); then
  # The authoritative run stamp is stored in the hash-bound remote control
  # script.  A stale training environment must not block emergency status or
  # stop, and the local placeholder is never used as ownership evidence.
  RUN_STAMP=control
fi

REMOTE_RUN_ROOT_CANONICAL=$(realpath -m -- "${REMOTE_RUN_ROOT}")
if [[ ! "${REMOTE_RUN_ROOT}" =~ ^/[A-Za-z0-9_.:@+/-]+$ \
      || "${REMOTE_RUN_ROOT}" != "${REMOTE_RUN_ROOT_CANONICAL}" \
      || "${REMOTE_RUN_ROOT}" == / ]]; then
  echo "[ERROR] REMOTE_RUN_ROOT must be a canonical safe non-root absolute path." >&2
  exit 2
fi
if (( CONTROL_ONLY_ACTION == 0 )); then
  LOGGER_BASE_DIR=$(realpath -m -- "${LOGGER_BASE_DIR}")
  case "${LOGGER_BASE_DIR}" in
    "${REMOTE_RUN_ROOT_CANONICAL}"/*)
      ;;
    *)
      echo "[ERROR] LOGGER_BASE_DIR must be a dedicated child of REMOTE_RUN_ROOT." >&2
      echo "[ERROR] remote_run_root=${REMOTE_RUN_ROOT_CANONICAL} logger_base_dir=${LOGGER_BASE_DIR}" >&2
      exit 2
      ;;
  esac
else
  LOGGER_BASE_DIR="${REMOTE_RUN_ROOT_CANONICAL}/training_logs"
fi
RESUME_FROM_BOX_WAS_SET=${RESUME_FROM_BOX+x}
RESUME_FROM_BOX=${RESUME_FROM_BOX:-0}
RESUME_TRAINING_CKPT=${RESUME_TRAINING_CKPT:-${RESUME_CHECKPOINT:-${RESUME_CKPT:-}}}
RESUME_WANDB_RUN_ID=${RESUME_WANDB_RUN_ID:-${WANDB_RUN_ID:-${RESUME_WANDB_ID:-}}}
FRESH_WANDB_RUN_ID=${FRESH_WANDB_RUN_ID:-}
REPLAY_PREFLIGHT_MANIFEST=${REPLAY_PREFLIGHT_MANIFEST:-}
REPLAY_PREFLIGHT_MANIFEST_SHA256=${REPLAY_PREFLIGHT_MANIFEST_SHA256:-}
REPLAY_PREFLIGHT_REQUIRED_VERSION=${REPLAY_PREFLIGHT_REQUIRED_VERSION:-}
REPLAY_AS_MOTION_CLIP_ID=""
REPLAY_AS_MOTION_NPZ_SHA256=""
REPLAY_AS_OBJECT_MAP_SHA256=""
REPLAY_AS_OBJECT_URDF_SHA256=""
REPLAY_AS_OBJECT_MESH_SHA256=""
REPLAY_AS_SINGLE_SLOT_SOURCE_DIGEST=""
REPLAY_AS_SINGLE_SLOT_VIEW_DIGEST=""
REPLAY_AS_RANK_SHARD_SOURCE_DIGEST=""
AS_EXTERNAL_CLOSURE_RECORD=""
AS_EXTERNAL_MOTION_CLIP_ID=""
AS_EXTERNAL_SOLID_SOURCE_DIGEST=""
AS_EXTERNAL_SINGLE_SLOT_SOURCE_DIGEST=""
AS_EXTERNAL_SINGLE_SLOT_VIEW_DIGEST=""
AS_EXTERNAL_RANK_SHARD_SOURCE_DIGEST=""
AS_EXTERNAL_MOTION_NPZ_SHA256=""
AS_EXTERNAL_OBJECT_MAP_SHA256=""
AS_EXTERNAL_OBJECT_URDF_SHA256=""
AS_EXTERNAL_OBJECT_MESH_SHA256=""
AS_EXTERNAL_SINGLE_SLOT_DIR=""
AS_EXTERNAL_MOTION_GENERATOR_TEACHER_SHA256=""
if (( CONTROL_ONLY_ACTION == 0 )); then
  case "${REPLAY_PREFLIGHT_REQUIRED_VERSION}" in
    ""|1|2)
      ;;
    *)
      echo "[ERROR] REPLAY_PREFLIGHT_REQUIRED_VERSION must be empty, 1, or 2. Got: ${REPLAY_PREFLIGHT_REQUIRED_VERSION}" >&2
      exit 2
      ;;
  esac
  if [[ "${DISTILL_AS_FORMAL_FRESH}" == 1 ]]; then
    if [[ -n "${REPLAY_PREFLIGHT_REQUIRED_VERSION}" \
          && "${REPLAY_PREFLIGHT_REQUIRED_VERSION}" != 2 ]]; then
      echo "[ERROR] DISTILL_AS_FORMAL_FRESH=1 requires REPLAY_PREFLIGHT_REQUIRED_VERSION=2." >&2
      exit 2
    fi
    REPLAY_PREFLIGHT_REQUIRED_VERSION=2
  fi
fi
WANDB_RESUME_MODE=${WANDB_RESUME_MODE:-${WANDB_RESUME:-must}}
WANDB_ENTITY=${WANDB_ENTITY:-zihanw22}
WANDB_RESUME_SAME_RUN=${WANDB_RESUME_SAME_RUN:-}
WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}
WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-120}
HOLOSOMA_REQUIRE_WANDB_RUN=${HOLOSOMA_REQUIRE_WANDB_RUN:-1}
if (( CONTROL_ONLY_ACTION == 0 )); then
  if [[ "${WANDB_BASE_URL}" != "https://api.wandb.ai" ]]; then
    echo "[ERROR] WANDB_BASE_URL must be exactly https://api.wandb.ai for this scientific launcher. Got: ${WANDB_BASE_URL}" >&2
    exit 2
  fi
  if ! canonical_positive_uint_at_most "${WANDB_INIT_TIMEOUT}" 3600; then
    echo "[ERROR] WANDB_INIT_TIMEOUT must be a canonical integer in [1, 3600]. Got: ${WANDB_INIT_TIMEOUT}" >&2
    exit 2
  fi
  if ! HOLOSOMA_REQUIRE_WANDB_RUN=$(normalize_bool01 \
      HOLOSOMA_REQUIRE_WANDB_RUN "${HOLOSOMA_REQUIRE_WANDB_RUN}"); then
    exit 2
  fi
  if [[ "${HOLOSOMA_REQUIRE_WANDB_RUN}" != "1" ]]; then
    echo "[ERROR] Scientific batch launch requires HOLOSOMA_REQUIRE_WANDB_RUN=1 exactly." >&2
    exit 2
  fi
  if [[ ! "${WANDB_ENTITY}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "[ERROR] WANDB_ENTITY must be one canonical W&B URL-path segment. Got: ${WANDB_ENTITY}" >&2
    exit 2
  fi
  if [[ -n "${RESUME_WANDB_RUN_ID}" ]]; then
    if [[ ! "${RESUME_WANDB_RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
      echo "[ERROR] RESUME_WANDB_RUN_ID must be one canonical W&B URL-path segment. Got: ${RESUME_WANDB_RUN_ID}" >&2
      exit 2
    fi
    if [[ "${WANDB_RESUME_MODE}" != "must" ]]; then
      echo "[ERROR] Scientific same-run W&B resume requires WANDB_RESUME_MODE=must exactly. Got: ${WANDB_RESUME_MODE}" >&2
      exit 2
    fi
  fi
  if [[ -n "${FRESH_WANDB_RUN_ID}" ]]; then
    if [[ ! "${FRESH_WANDB_RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
      echo "[ERROR] FRESH_WANDB_RUN_ID must be one canonical W&B URL-path segment. Got: ${FRESH_WANDB_RUN_ID}" >&2
      exit 2
    fi
    if [[ "${WANDB_RESUME_MODE}" != "must" ]]; then
      echo "[ERROR] A pre-bound fresh W&B run requires WANDB_RESUME_MODE=must exactly. Got: ${WANDB_RESUME_MODE}" >&2
      exit 2
    fi
  fi
fi
STUDENT_ACTOR_HIDDEN_DIMS_WAS_SET=${STUDENT_ACTOR_HIDDEN_DIMS+x}
STUDENT_POLICY_TYPE_WAS_SET=${STUDENT_POLICY_TYPE+x}
STUDENT_ACTOR_INPUTS_WAS_SET=${STUDENT_ACTOR_INPUTS+x}
STUDENT_FLOW_STEPS_WAS_SET=${STUDENT_FLOW_STEPS+x}
STUDENT_FLOW_TRAIN_NOISE_STD_WAS_SET=${STUDENT_FLOW_TRAIN_NOISE_STD+x}
STUDENT_FLOW_TIME_EPSILON_WAS_SET=${STUDENT_FLOW_TIME_EPSILON+x}
STUDENT_FLOW_INFERENCE_NOISE_STD_WAS_SET=${STUDENT_FLOW_INFERENCE_NOISE_STD+x}
TEACHER_ACTOR_OBS_HISTORY_LENGTH_WAS_SET=${TEACHER_ACTOR_OBS_HISTORY_LENGTH+x}
STUDENT_PROPRIO_HISTORY_LENGTH_WAS_SET=${STUDENT_PROPRIO_HISTORY_LENGTH+x}
STUDENT_ACTION_HISTORY_LENGTH_WAS_SET=${STUDENT_ACTION_HISTORY_LENGTH+x}
CRITIC_PROPRIO_HISTORY_LENGTH_WAS_SET=${CRITIC_PROPRIO_HISTORY_LENGTH+x}
CONTACT_AWARE_HISTORY_WAS_SET=${CONTACT_AWARE_HISTORY+x}
CONTACT_AWARE_HISTORY_LENGTH_WAS_SET=${CONTACT_AWARE_HISTORY_LENGTH+x}
STUDENT_POLICY_TYPE_EXPLICIT=0
[[ -n "${STUDENT_POLICY_TYPE_WAS_SET}" ]] && STUDENT_POLICY_TYPE_EXPLICIT=1
STUDENT_ACTOR_INPUTS_EXPLICIT=0
[[ -n "${STUDENT_ACTOR_INPUTS_WAS_SET}" ]] && STUDENT_ACTOR_INPUTS_EXPLICIT=1
STUDENT_FLOW_STEPS_EXPLICIT=0
[[ -n "${STUDENT_FLOW_STEPS_WAS_SET}" ]] && STUDENT_FLOW_STEPS_EXPLICIT=1
STUDENT_FLOW_TRAIN_NOISE_STD_EXPLICIT=0
[[ -n "${STUDENT_FLOW_TRAIN_NOISE_STD_WAS_SET}" ]] && STUDENT_FLOW_TRAIN_NOISE_STD_EXPLICIT=1
STUDENT_FLOW_TIME_EPSILON_EXPLICIT=0
[[ -n "${STUDENT_FLOW_TIME_EPSILON_WAS_SET}" ]] && STUDENT_FLOW_TIME_EPSILON_EXPLICIT=1
STUDENT_FLOW_INFERENCE_NOISE_STD_EXPLICIT=0
[[ -n "${STUDENT_FLOW_INFERENCE_NOISE_STD_WAS_SET}" ]] && STUDENT_FLOW_INFERENCE_NOISE_STD_EXPLICIT=1
STUDENT_POLICY_TYPE=${STUDENT_POLICY_TYPE:-mlp}
STUDENT_FLOW_STEPS=${STUDENT_FLOW_STEPS:-4}
STUDENT_FLOW_TRAIN_NOISE_STD=${STUDENT_FLOW_TRAIN_NOISE_STD:-1.0}
STUDENT_FLOW_TIME_EPSILON=${STUDENT_FLOW_TIME_EPSILON:-1e-4}
STUDENT_FLOW_INFERENCE_NOISE_STD=${STUDENT_FLOW_INFERENCE_NOISE_STD:-0.0}
if (( CONTROL_ONLY_ACTION == 0 )); then
  STUDENT_POLICY_TYPE=$(echo "${STUDENT_POLICY_TYPE}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
  case "${STUDENT_POLICY_TYPE}" in
    mlp|flow) ;;
    *)
      echo "[ERROR] STUDENT_POLICY_TYPE must be mlp or flow. Got: ${STUDENT_POLICY_TYPE}" >&2
      exit 2
      ;;
  esac
  # Validate this interpreter-affecting environment variable before the first
  # controller-side Python helper.  Otherwise CPython itself aborts during
  # preinitialization and bypasses the launcher's actionable contract error.
  if [[ -n "${PYTHONHASHSEED:-}" ]] \
      && { ! [[ "${PYTHONHASHSEED}" =~ ^[0-9]+$ ]] \
           || (( ${#PYTHONHASHSEED} > 10 )) \
           || (( 10#${PYTHONHASHSEED} > 4294967295 )); }; then
    echo "[ERROR] PYTHONHASHSEED must be an integer in [0, 4294967295]. Got: ${PYTHONHASHSEED}" >&2
    exit 2
  fi
  ensure_training_python
  "${PYTHON_BIN}" - \
    "${STUDENT_POLICY_TYPE}" \
    "${STUDENT_FLOW_STEPS}" \
    "${STUDENT_FLOW_TRAIN_NOISE_STD}" \
    "${STUDENT_FLOW_TIME_EPSILON}" \
    "${STUDENT_FLOW_INFERENCE_NOISE_STD}" \
    "${STUDENT_ACTOR_INPUTS_WAS_SET}" \
    "${STUDENT_ACTOR_INPUTS:-}" \
    "${RESUME_TRAINING_CKPT:+1}" \
    "${STUDENT_POLICY_TYPE_EXPLICIT}" <<'PY'
from __future__ import annotations

import ast
import math
import sys

MAX_FLOW_INTEGRATION_STEPS = 4096
MAX_FLOW_NOISE_STD = 1.0e18

(
    policy_type,
    raw_steps,
    raw_train_noise,
    raw_epsilon,
    raw_inference_noise,
    inputs_set,
    raw_inputs,
    resume_set,
    policy_type_explicit,
) = sys.argv[1:]
try:
    steps = int(raw_steps)
except ValueError as exc:
    raise SystemExit(f"[ERROR] STUDENT_FLOW_STEPS must be an integer, got {raw_steps!r}.") from exc
if steps < 1 or steps > MAX_FLOW_INTEGRATION_STEPS or str(steps) != raw_steps.strip():
    raise SystemExit(
        "[ERROR] STUDENT_FLOW_STEPS must be a canonical positive integer in "
        f"[1, {MAX_FLOW_INTEGRATION_STEPS}], got {raw_steps!r}."
    )
for name, raw, lower, upper in (
    ("STUDENT_FLOW_TRAIN_NOISE_STD", raw_train_noise, 0.0, MAX_FLOW_NOISE_STD),
    ("STUDENT_FLOW_TIME_EPSILON", raw_epsilon, 0.0, 0.49),
    ("STUDENT_FLOW_INFERENCE_NOISE_STD", raw_inference_noise, 0.0, MAX_FLOW_NOISE_STD),
):
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {name} must be numeric, got {raw!r}.") from exc
    if not math.isfinite(value) or value < lower or (upper is not None and value > upper):
        interval = f"[{lower}, {upper}]" if upper is not None else f">= {lower}"
        raise SystemExit(f"[ERROR] {name} must be finite and {interval}, got {raw!r}.")
if inputs_set:
    try:
        inputs = ast.literal_eval(raw_inputs)
    except (SyntaxError, ValueError) as exc:
        raise SystemExit(f"[ERROR] Invalid STUDENT_ACTOR_INPUTS={raw_inputs!r}: {exc}") from exc
    if not isinstance(inputs, (list, tuple)) or not inputs or not all(isinstance(item, str) and item for item in inputs):
        raise SystemExit("[ERROR] STUDENT_ACTOR_INPUTS must be a non-empty ordered list of group names.")
    if len(set(inputs)) != len(inputs):
        raise SystemExit(f"[ERROR] STUDENT_ACTOR_INPUTS contains duplicates: {inputs!r}.")
defer_policy_type_to_resume = bool(resume_set) and policy_type_explicit != "1"
if (
    policy_type != "flow"
    and not defer_policy_type_to_resume
    and (
        steps != 4
        or float(raw_train_noise) != 1.0
        or float(raw_epsilon) != 1e-4
        or float(raw_inference_noise) != 0.0
    )
):
    raise SystemExit("[ERROR] Non-default STUDENT_FLOW_* settings require STUDENT_POLICY_TYPE=flow.")
PY
fi
if (( CONTROL_ONLY_ACTION == 0 )); then
  if [[ -n "${TARGET_LEARNING_ITERATION+x}" && -n "${NUM_LEARNING_ITERATIONS+x}" && "${TARGET_LEARNING_ITERATION}" != "${NUM_LEARNING_ITERATIONS}" ]]; then
    echo "[ERROR] TARGET_LEARNING_ITERATION and legacy NUM_LEARNING_ITERATIONS disagree." >&2
    exit 2
  fi
  if [[ -n "${ADDITIONAL_LEARNING_ITERATIONS:-}" ]]; then
    echo "[ERROR] ADDITIONAL_LEARNING_ITERATIONS is not supported by this launcher; use the absolute TARGET_LEARNING_ITERATION." >&2
    exit 2
  fi
  TARGET_LEARNING_ITERATION=${TARGET_LEARNING_ITERATION:-${NUM_LEARNING_ITERATIONS:-40000}}
  if ! canonical_positive_uint_at_most \
      "${TARGET_LEARNING_ITERATION}" "${MAX_SIGNED_32}"; then
    echo "[ERROR] TARGET_LEARNING_ITERATION must be a canonical integer in [1, ${MAX_SIGNED_32}]. Got: ${TARGET_LEARNING_ITERATION}" >&2
    exit 2
  fi
else
  # Control actions derive the authoritative target from exact v2 active
  # metadata.  Ignore stale or malformed training-only target overrides.
  TARGET_LEARNING_ITERATION=1
fi
if ! canonical_positive_uint_at_most \
    "${STATUS_STALE_SECONDS}" "${MAX_STATUS_SECONDS}"; then
  echo "[ERROR] STATUS_STALE_SECONDS must be a canonical integer in [1, ${MAX_STATUS_SECONDS}]. Got: ${STATUS_STALE_SECONDS}" >&2
  exit 2
fi
if ! canonical_uint_at_most "${STATUS_MAX_CLOCK_SKEW_SECONDS}" 300; then
  echo "[ERROR] STATUS_MAX_CLOCK_SKEW_SECONDS must be a canonical integer in [0,300]. Got: ${STATUS_MAX_CLOCK_SKEW_SECONDS}" >&2
  exit 2
fi
NUM_LEARNING_ITERATIONS=${TARGET_LEARNING_ITERATION}
if (( CONTROL_ONLY_ACTION == 0 )); then
  if [[ -n "${NUM_MINI_BATCHES+x}" ]]; then
    NUM_MINI_BATCHES_WAS_SET=1
  else
    NUM_MINI_BATCHES_WAS_SET=0
  fi
  NUM_MINI_BATCHES=${NUM_MINI_BATCHES:-64}
  NUM_LEARNING_EPOCHS=${NUM_LEARNING_EPOCHS:-1}
  HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY=$(normalize_bool01 \
    HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY \
    "${HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY:-0}")
  if [[ "${HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY}" == 1 ]]; then
    if [[ "${NUM_MINI_BATCHES_WAS_SET}" != 1 || "${NUM_MINI_BATCHES}" != 16 ]]; then
      echo "[ERROR] HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY=1 requires explicit NUM_MINI_BATCHES=16; the canary never changes the default automatically." >&2
      exit 2
    fi
    if [[ "${NUM_LEARNING_EPOCHS}" != 1 ]]; then
      echo "[ERROR] The NUM_MINI_BATCHES=16 throughput canary requires NUM_LEARNING_EPOCHS=1 so the A/B changes only optimizer batch granularity." >&2
      exit 2
    fi
  elif [[ "${NUM_MINI_BATCHES}" == 16 ]]; then
    echo "[ERROR] NUM_MINI_BATCHES=16 is an algorithm-changing throughput canary; set HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY=1 explicitly." >&2
    exit 2
  fi
else
  # Emergency status/stop derives its identity from active metadata and must
  # not be blocked by a stale training canary environment.
  NUM_MINI_BATCHES_WAS_SET=0
  NUM_MINI_BATCHES=64
  NUM_LEARNING_EPOCHS=1
  HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY=0
fi
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
FIXED_BC_EVAL_LOG_INTERVAL=${FIXED_BC_EVAL_LOG_INTERVAL:-100}
HOLOSOMA_MOTION_METRICS_INTERVAL=${HOLOSOMA_MOTION_METRICS_INTERVAL:-16}
if (( CONTROL_ONLY_ACTION == 0 )); then
  HOLOSOMA_COLLECTION_PROFILE_CANARY=$(normalize_bool01 \
    HOLOSOMA_COLLECTION_PROFILE_CANARY \
    "${HOLOSOMA_COLLECTION_PROFILE_CANARY:-0}")
  HOLOSOMA_COLLECTION_PROFILE_SYNC_CUDA=$(normalize_bool01 \
    HOLOSOMA_COLLECTION_PROFILE_SYNC_CUDA \
    "${HOLOSOMA_COLLECTION_PROFILE_SYNC_CUDA:-0}")
  HOLOSOMA_COLLECTION_PROFILE_INTERVAL=${HOLOSOMA_COLLECTION_PROFILE_INTERVAL:-1}
  if [[ "${HOLOSOMA_COLLECTION_PROFILE_CANARY}" == 1 ]]; then
    if (( 10#${TARGET_LEARNING_ITERATION} > 64 )); then
      echo "[ERROR] HOLOSOMA_COLLECTION_PROFILE_CANARY=1 is diagnostic-only and requires TARGET_LEARNING_ITERATION<=64; got ${TARGET_LEARNING_ITERATION}." >&2
      exit 2
    fi
    if [[ -n "${RESUME_TRAINING_CKPT}" || -n "${RESUME_WANDB_RUN_ID}" ]]; then
      echo "[ERROR] Collection profiling requires a fresh run; training/W&B resume identities are forbidden." >&2
      exit 2
    fi
  elif [[ "${HOLOSOMA_COLLECTION_PROFILE_SYNC_CUDA}" == 1 ]]; then
    echo "[ERROR] HOLOSOMA_COLLECTION_PROFILE_SYNC_CUDA=1 requires HOLOSOMA_COLLECTION_PROFILE_CANARY=1." >&2
    exit 2
  fi
else
  HOLOSOMA_COLLECTION_PROFILE_CANARY=0
  HOLOSOMA_COLLECTION_PROFILE_SYNC_CUDA=0
  HOLOSOMA_COLLECTION_PROFILE_INTERVAL=1
fi
ACTOR_LR=${ACTOR_LR:-1e-3}
CRITIC_LR=${CRITIC_LR:-1e-3}
PPO_LR_SCHEDULE=${PPO_LR_SCHEDULE:-adaptive}
PPO_DESIRED_KL=${PPO_DESIRED_KL:-0.01}
if (( CONTROL_ONLY_ACTION == 0 )); then
  case "${PPO_LR_SCHEDULE}" in
    adaptive|fixed)
      ;;
    *)
      echo "[ERROR] PPO_LR_SCHEDULE must be exactly adaptive or fixed. Got: ${PPO_LR_SCHEDULE}" >&2
      exit 2
      ;;
  esac

  # PPO's historical implicit bounds are min(initial, 1e-5) and
  # max(initial, 1e-2). Materialize those values on the controller so the
  # effective optimizer contract is visible, provenance-bound, and identical
  # on every node instead of being recomputed from ambient defaults.
  materialize_default_lr_bound() {
    local name="$1"
    local initial="$2"
    local bound_kind="$3"
    "${PYTHON_BIN}" - "${name}" "${initial}" "${bound_kind}" <<'PY'
from __future__ import annotations

import math
import sys

name, raw_initial, bound_kind = sys.argv[1:]
try:
    initial = float(raw_initial)
except (TypeError, ValueError, OverflowError) as exc:
    raise SystemExit(f"[ERROR] {name} must be numeric. Got: {raw_initial}") from exc
if not math.isfinite(initial):
    raise SystemExit(f"[ERROR] {name} must be finite. Got: {raw_initial}")
if initial <= 0.0:
    raise SystemExit(f"[ERROR] {name} must be finite and > 0. Got: {raw_initial}")
if bound_kind == "min":
    value = min(initial, 1.0e-5)
elif bound_kind == "max":
    value = max(initial, 1.0e-2)
else:
    raise SystemExit(f"[ERROR] Unsupported LR bound kind: {bound_kind}")
print(repr(value))
PY
  }
  ACTOR_MIN_LR=${ACTOR_MIN_LR:-$(materialize_default_lr_bound ACTOR_LR "${ACTOR_LR}" min)}
  ACTOR_MAX_LR=${ACTOR_MAX_LR:-$(materialize_default_lr_bound ACTOR_LR "${ACTOR_LR}" max)}
  CRITIC_MIN_LR=${CRITIC_MIN_LR:-$(materialize_default_lr_bound CRITIC_LR "${CRITIC_LR}" min)}
  CRITIC_MAX_LR=${CRITIC_MAX_LR:-$(materialize_default_lr_bound CRITIC_LR "${CRITIC_LR}" max)}
  unset -f materialize_default_lr_bound
else
  # Control-only actions bind to active metadata and never consume these
  # experiment knobs; avoid invoking controller Python merely to inspect/stop.
  ACTOR_MIN_LR=${ACTOR_MIN_LR:-1e-5}
  ACTOR_MAX_LR=${ACTOR_MAX_LR:-1e-2}
  CRITIC_MIN_LR=${CRITIC_MIN_LR:-1e-5}
  CRITIC_MAX_LR=${CRITIC_MAX_LR:-1e-2}
fi
# These actor-distribution settings are part of the scientific experiment,
# but the delegated box/perception wrappers also expose them as ambient shell
# defaults.  Serialize the controller's canonical values explicitly so a
# node-local login profile (or a later wrapper-default change) cannot alter the
# experiment after the controller has constructed its launch identity.
ACTOR_MIN_NOISE_STD=${ACTOR_MIN_NOISE_STD:-0.01}
INIT_NOISE_STD=${INIT_NOISE_STD:-0.01}
ENTROPY_COEF=${ENTROPY_COEF:-0.0}
# A curriculum-correct training resume must retain the saved schedule. These defaults therefore match
# new runs too; the loaded iteration recomputes the correct late-run coefficient.
PPO_START_EPOCH=${PPO_START_EPOCH:-0}
DAGGER_END_EPOCH=${DAGGER_END_EPOCH:-4900}
# A fresh AS student run starts with a real behavior-cloning phase.  With the
# default 700-iteration staircase and 0.7 target this produces exact 0.1 PPO
# increments at 700, 1400, ..., 4900.  A non-zero default here both removes the
# pure-BC phase and makes those tiers non-decimal (for example 0.108888...).
PPO_START_COEFF=${PPO_START_COEFF:-0.0}
PPO_TARGET_COEFF=${PPO_TARGET_COEFF:-0.7}
PPO_SCHEDULE_STEP_EPOCHS=${PPO_SCHEDULE_STEP_EPOCHS:-700}
DAGGER_REPLAY_ENABLED=${DAGGER_REPLAY_ENABLED:-False}
DAGGER_REPLAY_CAPACITY=${DAGGER_REPLAY_CAPACITY:-512}
DAGGER_REPLAY_BATCH_SIZE=${DAGGER_REPLAY_BATCH_SIZE:-512}
DAGGER_REPLAY_FRACTION=${DAGGER_REPLAY_FRACTION:-0.5}
DAGGER_REPLAY_SEED=${DAGGER_REPLAY_SEED:-0}
FIXED_BC_GUARD_ENABLED=${FIXED_BC_GUARD_ENABLED:-True}
FIXED_BC_GUARD_REFERENCE_END_EPOCH=${FIXED_BC_GUARD_REFERENCE_END_EPOCH:-600}
FIXED_BC_GUARD_MAX_REFERENCE_RATIO=${FIXED_BC_GUARD_MAX_REFERENCE_RATIO:-2.0}
FIXED_BC_GUARD_ABSOLUTE_MAX_MU_MSE=${FIXED_BC_GUARD_ABSOLUTE_MAX_MU_MSE:-0.160}
if [[ -z "${FIXED_BC_GUARD_START_EPOCH+x}" ]]; then
  case "$(echo "${FIXED_BC_GUARD_ENABLED}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) FIXED_BC_GUARD_START_EPOCH=${DAGGER_END_EPOCH} ;;
    0|false|no|off|"") FIXED_BC_GUARD_START_EPOCH=-1 ;;
    *) FIXED_BC_GUARD_START_EPOCH=${DAGGER_END_EPOCH} ;;
  esac
fi
FIXED_BC_GUARD_CONSECUTIVE_EVALS=${FIXED_BC_GUARD_CONSECUTIVE_EVALS:-3}
if (( CONTROL_ONLY_ACTION != 0 )); then
  # Status/stop bind to immutable active metadata, not stale caller-side
  # training knobs.  Keep newly introduced guard fields from blocking an
  # emergency read/stop operation before that metadata can be inspected.
  FIXED_BC_EVAL_LOG_INTERVAL=100
  FIXED_BC_GUARD_ENABLED=False
  FIXED_BC_GUARD_REFERENCE_END_EPOCH=600
  FIXED_BC_GUARD_MAX_REFERENCE_RATIO=2.0
  FIXED_BC_GUARD_ABSOLUTE_MAX_MU_MSE=0.160
  FIXED_BC_GUARD_START_EPOCH=-1
  FIXED_BC_GUARD_CONSECUTIVE_EVALS=3
  DAGGER_REPLAY_ENABLED=False
  DAGGER_REPLAY_CAPACITY=512
  DAGGER_REPLAY_BATCH_SIZE=512
  DAGGER_REPLAY_FRACTION=0.5
  DAGGER_REPLAY_SEED=0
fi
DAGGER_LOSS_COEF=${DAGGER_LOSS_COEF:-1.0}
DAGGER_MATCH_STD=${DAGGER_MATCH_STD:-False}
PPO_START_NOISE_STD=${PPO_START_NOISE_STD:-0.1}
PPO_START_NOISE_STD_UNTIL_COEFF=${PPO_START_NOISE_STD_UNTIL_COEFF:-0.1}
# The delegated wrapper historically starts both reset curricula at iteration
# 2500 and derives their end from the training target.  Rollout iterations are
# exactly [0, TARGET_LEARNING_ITERATION), so materialize the final reachable
# rollout (TARGET-1), not the exclusive target, as the default schedule end.
# If that last rollout is before 2500, the bounded short-run schedule begins at
# iteration zero.  At target=2501, start=end=2500 is intentionally valid and
# reaches the configured end value on the final rollout.
RESET_CURRICULUM_DEFAULT_END_ITER=$((10#${TARGET_LEARNING_ITERATION} - 1))
RESET_CURRICULUM_DEFAULT_START_ITER=2500
if (( RESET_CURRICULUM_DEFAULT_END_ITER < RESET_CURRICULUM_DEFAULT_START_ITER )); then
  RESET_CURRICULUM_DEFAULT_START_ITER=0
fi
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.2}
START_AT_TIMESTEP_ZERO_PROB_END=${START_AT_TIMESTEP_ZERO_PROB_END:-1.0}
START_AT_TIMESTEP_ZERO_PROB_START_ITER=${START_AT_TIMESTEP_ZERO_PROB_START_ITER:-${RESET_CURRICULUM_DEFAULT_START_ITER}}
START_AT_TIMESTEP_ZERO_PROB_END_ITER=${START_AT_TIMESTEP_ZERO_PROB_END_ITER:-${RESET_CURRICULUM_DEFAULT_END_ITER}}
FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}
FREEZE_AT_TIMESTEP_ZERO_PROB_END=${FREEZE_AT_TIMESTEP_ZERO_PROB_END:-0.0}
FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=${FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER:-${RESET_CURRICULUM_DEFAULT_START_ITER}}
FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=${FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER:-${RESET_CURRICULUM_DEFAULT_END_ITER}}
unset RESET_CURRICULUM_DEFAULT_START_ITER RESET_CURRICULUM_DEFAULT_END_ITER
UNIFORM_T1_WINDOW_TARGET_SAMPLE_FRAC=${UNIFORM_T1_WINDOW_TARGET_SAMPLE_FRAC:-}
STUDENT_MOTION_END_MODE=${STUDENT_MOTION_END_MODE:-episodic}
CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION_WAS_SET=${CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION+x}
CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION=${CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION:-}
ALLOW_FRESH_CURRICULUM_RESUME=${ALLOW_FRESH_CURRICULUM_RESUME:-0}
ALLOW_NONDETERMINISTIC_RNG_RESUME=${ALLOW_NONDETERMINISTIC_RNG_RESUME:-0}
ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME=${ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME:-0}
ALLOW_RUNTIME_DRIFT_ON_RESUME=${ALLOW_RUNTIME_DRIFT_ON_RESUME:-0}
EXPORT_ONNX=${EXPORT_ONNX:-False}
TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT:-wandb://zihanw22/carry-any/bcleb5oi/model_67000.pt}
HOLOSOMA_SKIP_INITIAL_CHECKPOINT=${HOLOSOMA_SKIP_INITIAL_CHECKPOINT:-1}
HOLOSOMA_SKIP_GRAD_FINITE_CHECK=${HOLOSOMA_SKIP_GRAD_FINITE_CHECK:-0}
HOLOSOMA_SKIP_LOSS_FINITE_CHECK=${HOLOSOMA_SKIP_LOSS_FINITE_CHECK:-0}
HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION=${HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION:-0}
PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY:-301989888}
PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY=${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY:-301989888}
PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY:-134217728}
PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-268435456}
HOLOSOMA_OBJECT_COLLIDER_TYPE=${HOLOSOMA_OBJECT_COLLIDER_TYPE:-convex_decomposition}
HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=${HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS:-0}
HOLOSOMA_DEBUG_HEARTBEAT=${HOLOSOMA_DEBUG_HEARTBEAT:-0}
HOLOSOMA_DEBUG_HEARTBEAT_VERBOSE=${HOLOSOMA_DEBUG_HEARTBEAT_VERBOSE:-0}
HOLOSOMA_DEBUG_ACTOR=${HOLOSOMA_DEBUG_ACTOR:-0}
HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE=${HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE:-1}
HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE=${HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE:-0}
HOLOSOMA_SYNC_AFTER_OPTIMIZER_STEP=${HOLOSOMA_SYNC_AFTER_OPTIMIZER_STEP:-0}
HOLOSOMA_SYNC_AFTER_MICROBATCH_FORWARD=${HOLOSOMA_SYNC_AFTER_MICROBATCH_FORWARD:-0}
HOLOSOMA_DEBUG_MICROBATCH_ALL=${HOLOSOMA_DEBUG_MICROBATCH_ALL:-0}
HOLOSOMA_GLOO_GRAD_REDUCE=${HOLOSOMA_GLOO_GRAD_REDUCE:-1}
HOLOSOMA_GLOO_BARRIER=${HOLOSOMA_GLOO_BARRIER:-1}
HOLOSOMA_GLOO_SMALL_COLLECTIVES=${HOLOSOMA_GLOO_SMALL_COLLECTIVES:-1}
HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE:-0}
HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=${HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES:-0}
HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER=${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER:-0}
HOLOSOMA_RANK_VISIBLE_DEVICES=${HOLOSOMA_RANK_VISIBLE_DEVICES:-0}
HOLOSOMA_RANK_LOCAL_CPU_AFFINITY=${HOLOSOMA_RANK_LOCAL_CPU_AFFINITY:-0}
HOLOSOMA_CARB_TASKING_THREAD_COUNT=${HOLOSOMA_CARB_TASKING_THREAD_COUNT:-}
HOLOSOMA_CONTIGUOUS_MINIBATCHES=${HOLOSOMA_CONTIGUOUS_MINIBATCHES:-1}
HOLOSOMA_DAGGER_SUPERVISED_ONLY=${HOLOSOMA_DAGGER_SUPERVISED_ONLY:-0}
HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP=${HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP:-${HOLOSOMA_DAGGER_SUPERVISED_ONLY}}
HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH=${HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH:-16}
HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD=${HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD:-1}
HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC=${HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC:-0}
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}
DEFAULT_BOX_RESUME_RUN=${DEFAULT_BOX_RESUME_RUN:-https://wandb.ai/zihanw22/boxer/runs/d9m3z369-recovered}
DEFAULT_BOX_RESUME_MODEL_FILE=${DEFAULT_BOX_RESUME_MODEL_FILE:-model_22000.pt}
DEFAULT_BOX_RESUME_CHECKPOINT=${DEFAULT_BOX_RESUME_CHECKPOINT:-${DEFAULT_BOX_RESUME_RUN}/files/${DEFAULT_BOX_RESUME_MODEL_FILE}}
BOX_POLICY_INIT_REF=${BOX_POLICY_INIT_REF:-${BOX_RESUME_CKPT:-${RESUME_FROM_BOX_CKPT:-${DEFAULT_BOX_RESUME_CHECKPOINT}}}}
BOX_POLICY_INIT_EXPECTED_SHA256=${BOX_POLICY_INIT_EXPECTED_SHA256:-}
BOX_POLICY_INIT_CACHE_ROOT=${BOX_POLICY_INIT_CACHE_ROOT:-/home/ubuntu/.cache/holosoma/checkpoints}
BOX_POLICY_INIT_ACTOR_HIDDEN_DIMS=${BOX_POLICY_INIT_ACTOR_HIDDEN_DIMS:-'[512,256,128]'}
HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET=${HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET:-}
BOX_POLICY_INIT_CONTROL_CACHE_ROOT=${BOX_POLICY_INIT_CONTROL_CACHE_ROOT:-}
BOX_POLICY_INIT_EXPECTED_WORLD_SIZE=${BOX_POLICY_INIT_EXPECTED_WORLD_SIZE:-}
BOX_POLICY_INIT_EXPECTED_WANDB_RUN_PATH=${BOX_POLICY_INIT_EXPECTED_WANDB_RUN_PATH:-}
BOX_POLICY_INIT_EXPECTED_SOURCE_SNAPSHOT_ID=${BOX_POLICY_INIT_EXPECTED_SOURCE_SNAPSHOT_ID:-}
BOX_POLICY_INIT_REQUIRE_FRESH_SOURCE=${BOX_POLICY_INIT_REQUIRE_FRESH_SOURCE:-0}
CONTROL_BOX_POLICY_INIT_PATH=""
CONTROL_BOX_POLICY_INIT_SHA256=""
CONTROL_RESUME_TRAINING_PATH=""
CONTROL_RESUME_TRAINING_SHA256=""
CONTROL_TEACHER_CHECKPOINT_PATH=""
CONTROL_TEACHER_CHECKPOINT_SHA256=""
if (( CONTROL_ONLY_ACTION == 0 )); then
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
if [[ -n "${HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET}" ]] \
    && ! canonical_positive_uint_at_most \
      "${HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET}" "${MAX_SIGNED_32}"; then
  echo "[ERROR] HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET must be a canonical integer in [1, ${MAX_SIGNED_32}]." >&2
  exit 2
fi
if [[ -n "${BOX_POLICY_INIT_EXPECTED_SHA256}" \
      && ! "${BOX_POLICY_INIT_EXPECTED_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "[ERROR] BOX_POLICY_INIT_EXPECTED_SHA256 must be exactly 64 lowercase hexadecimal characters." >&2
  exit 2
fi
if [[ -n "${BOX_POLICY_INIT_EXPECTED_WORLD_SIZE}" ]] \
    && ! canonical_positive_uint_at_most \
      "${BOX_POLICY_INIT_EXPECTED_WORLD_SIZE}" "${MAX_TOTAL_GPUS}"; then
  echo "[ERROR] BOX_POLICY_INIT_EXPECTED_WORLD_SIZE must be a canonical integer in [1, ${MAX_TOTAL_GPUS}]." >&2
  exit 2
fi
if [[ -n "${BOX_POLICY_INIT_EXPECTED_WANDB_RUN_PATH}" \
      && ! "${BOX_POLICY_INIT_EXPECTED_WANDB_RUN_PATH}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "[ERROR] BOX_POLICY_INIT_EXPECTED_WANDB_RUN_PATH must be exactly entity/project/run using canonical path segments." >&2
  exit 2
fi
if [[ -n "${BOX_POLICY_INIT_EXPECTED_SOURCE_SNAPSHOT_ID}" \
      && ! "${BOX_POLICY_INIT_EXPECTED_SOURCE_SNAPSHOT_ID}" =~ ^src-[0-9a-f]{64}$ ]]; then
  echo "[ERROR] BOX_POLICY_INIT_EXPECTED_SOURCE_SNAPSHOT_ID must have format src-<64 lowercase SHA256 hex>." >&2
  exit 2
fi
if [[ "${BOX_POLICY_INIT_REQUIRE_FRESH_SOURCE}" != 0 \
      && "${BOX_POLICY_INIT_REQUIRE_FRESH_SOURCE}" != 1 ]]; then
  echo "[ERROR] BOX_POLICY_INIT_REQUIRE_FRESH_SOURCE must be exactly 0 or 1." >&2
  exit 2
fi

normalize_wandb_file_ref() {
  local ref="$1"
  local clean_ref="${ref%%\?*}"
  if [[ "${clean_ref}" != https://wandb.ai/*/runs/* ]]; then
    echo "${ref}"
    return 0
  fi

  local trimmed="${clean_ref#https://wandb.ai/}"
  local entity=""
  local project=""
  local run_id=""
  local marker=""
  local file_name=""
  IFS='/' read -r entity project marker run_id marker file_name <<< "${trimmed}"
  if [[ -z "${entity}" || -z "${project}" || -z "${run_id}" || "${trimmed}" != "${entity}/${project}/runs/${run_id}/files/"* ]]; then
    echo "[ERROR] Training-resume W&B URLs must include /runs/<id>/files/<checkpoint>.pt: ${ref}" >&2
    return 2
  fi
  file_name="${trimmed#${entity}/${project}/runs/${run_id}/files/}"
  if [[ -z "${file_name}" || "${file_name}" != *.pt ]]; then
    echo "[ERROR] Training-resume W&B URL must name a .pt checkpoint: ${ref}" >&2
    return 2
  fi
  echo "wandb://${entity}/${project}/${run_id}/${file_name}"
}

ALLOW_FRESH_CURRICULUM_RESUME="$(normalize_bool01 ALLOW_FRESH_CURRICULUM_RESUME "${ALLOW_FRESH_CURRICULUM_RESUME}")"
ALLOW_NONDETERMINISTIC_RNG_RESUME="$(normalize_bool01 ALLOW_NONDETERMINISTIC_RNG_RESUME "${ALLOW_NONDETERMINISTIC_RNG_RESUME}")"
RESTART="$(normalize_bool01 RESTART "${RESTART}")"
ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME="$(normalize_bool01 ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME "${ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME}")"
ALLOW_RUNTIME_DRIFT_ON_RESUME="$(normalize_bool01 ALLOW_RUNTIME_DRIFT_ON_RESUME "${ALLOW_RUNTIME_DRIFT_ON_RESUME}")"
REQUIRE_MOTION_GENERATOR_TEACHER_MATCH="$(normalize_bool01 REQUIRE_MOTION_GENERATOR_TEACHER_MATCH "${REQUIRE_MOTION_GENERATOR_TEACHER_MATCH}")"
if [[ -n "${MOTION_GENERATOR_TEACHER_EXPECTED_SHA256}" \
      && ! "${MOTION_GENERATOR_TEACHER_EXPECTED_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "[ERROR] MOTION_GENERATOR_TEACHER_EXPECTED_SHA256 must be exactly 64 lowercase hexadecimal characters." >&2
  exit 2
fi
SKIP_NODE_HEALTH_CHECK="$(normalize_bool01 SKIP_NODE_HEALTH_CHECK "${SKIP_NODE_HEALTH_CHECK}")"
PREPARE_DATA="$(normalize_bool01 PREPARE_DATA "${PREPARE_DATA}")"
HOLOSOMA_GLOO_GRAD_REDUCE="$(normalize_bool01 HOLOSOMA_GLOO_GRAD_REDUCE "${HOLOSOMA_GLOO_GRAD_REDUCE}")"
HOLOSOMA_GLOO_BARRIER="$(normalize_bool01 HOLOSOMA_GLOO_BARRIER "${HOLOSOMA_GLOO_BARRIER}")"
HOLOSOMA_GLOO_SMALL_COLLECTIVES="$(normalize_bool01 HOLOSOMA_GLOO_SMALL_COLLECTIVES "${HOLOSOMA_GLOO_SMALL_COLLECTIVES}")"
HOLOSOMA_HIERARCHICAL_GRAD_REDUCE="$(normalize_bool01 HOLOSOMA_HIERARCHICAL_GRAD_REDUCE "${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE}")"
HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES="$(normalize_bool01 HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES "${HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES}")"
HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER="$(normalize_bool01 HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER "${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER}")"
HOLOSOMA_RANK_VISIBLE_DEVICES="$(normalize_bool01 HOLOSOMA_RANK_VISIBLE_DEVICES "${HOLOSOMA_RANK_VISIBLE_DEVICES}")"
HOLOSOMA_RANK_LOCAL_CPU_AFFINITY="$(normalize_bool01 HOLOSOMA_RANK_LOCAL_CPU_AFFINITY "${HOLOSOMA_RANK_LOCAL_CPU_AFFINITY}")"
HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS="$(normalize_bool01 HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS "${HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS}")"
case "${HOLOSOMA_OBJECT_COLLIDER_TYPE}" in
  convex_decomposition|convex_hull)
    ;;
  *)
    echo "[ERROR] HOLOSOMA_OBJECT_COLLIDER_TYPE must be exactly convex_decomposition or convex_hull. Got: ${HOLOSOMA_OBJECT_COLLIDER_TYPE}" >&2
    exit 2
    ;;
esac
if [[ -n "${HOLOSOMA_CARB_TASKING_THREAD_COUNT}" ]] \
    && ! canonical_positive_uint_at_most \
      "${HOLOSOMA_CARB_TASKING_THREAD_COUNT}" "${MAX_SIGNED_32}"; then
  echo "[ERROR] HOLOSOMA_CARB_TASKING_THREAD_COUNT must be unset or a canonical integer in [1, ${MAX_SIGNED_32}]. Got: ${HOLOSOMA_CARB_TASKING_THREAD_COUNT}" >&2
  exit 2
fi
HOLOSOMA_SKIP_GRAD_FINITE_CHECK="$(normalize_bool01 HOLOSOMA_SKIP_GRAD_FINITE_CHECK "${HOLOSOMA_SKIP_GRAD_FINITE_CHECK}")"
HOLOSOMA_SKIP_LOSS_FINITE_CHECK="$(normalize_bool01 HOLOSOMA_SKIP_LOSS_FINITE_CHECK "${HOLOSOMA_SKIP_LOSS_FINITE_CHECK}")"
HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION="$(normalize_bool01 HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION "${HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION}")"
HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC="$(normalize_bool01 HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC "${HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC}")"
HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE="$(normalize_bool01 HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE "${HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE}")"
if [[ "${HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE}" != "1" ]]; then
  echo "[ERROR] Scientific batch launch requires HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE=1 so multi-stream gradients are complete before collective packing." >&2
  exit 2
fi
# These are diagnostic skip switches, not scientific tuning knobs.  Reject an
# explicit request instead of silently changing the requested experiment.
for scientific_skip_name in \
  HOLOSOMA_SKIP_GRAD_FINITE_CHECK \
  HOLOSOMA_SKIP_LOSS_FINITE_CHECK \
  HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION \
  HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC; do
  if [[ "${!scientific_skip_name}" == "1" ]]; then
    echo "[ERROR] Scientific batch launch forbids ${scientific_skip_name}=1; integrity checks, loss accounting, and model synchronization must remain enabled." >&2
    exit 2
  fi
done
unset scientific_skip_name
HOLOSOMA_DEBUG_HEARTBEAT_VERBOSE="$(normalize_bool01 HOLOSOMA_DEBUG_HEARTBEAT_VERBOSE "${HOLOSOMA_DEBUG_HEARTBEAT_VERBOSE}")"
HOLOSOMA_DAGGER_SUPERVISED_ONLY="$(normalize_bool01 HOLOSOMA_DAGGER_SUPERVISED_ONLY "${HOLOSOMA_DAGGER_SUPERVISED_ONLY}")"
HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP="$(normalize_bool01 HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP "${HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP}")"
HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD="$(normalize_bool01 HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD "${HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD}")"
if ! canonical_uint_at_most \
    "${HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH}" "${MAX_SIGNED_32}"; then
  echo "[ERROR] HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH must be a canonical integer in [0, ${MAX_SIGNED_32}]. Got: ${HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH}" >&2
  exit 2
fi
if [[ "${HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP}" != "${HOLOSOMA_DAGGER_SUPERVISED_ONLY}" ]]; then
  echo "[ERROR] HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP must equal HOLOSOMA_DAGGER_SUPERVISED_ONLY; supervised-only action BC necessarily has no critic optimizer objective." >&2
  exit 2
fi
if [[ "${HOLOSOMA_DAGGER_SUPERVISED_ONLY}" == "1" \
      && "${HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD}" == "1" \
      && -z "${HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH//0/}" ]]; then
  echo "[ERROR] HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD=1 requires a positive HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH when supervised-only mode is enabled." >&2
  exit 2
fi
_export_onnx_bool="$(normalize_bool01 EXPORT_ONNX "${EXPORT_ONNX}")"
if [[ "${_export_onnx_bool}" == "1" ]]; then
  EXPORT_ONNX=True
else
  EXPORT_ONNX=False
fi
_dagger_match_std_bool="$(normalize_bool01 DAGGER_MATCH_STD "${DAGGER_MATCH_STD}")"
if [[ "${_dagger_match_std_bool}" == "1" ]]; then
  DAGGER_MATCH_STD=True
else
  DAGGER_MATCH_STD=False
fi
unset _dagger_match_std_bool
_dagger_replay_enabled_bool="$(normalize_bool01 DAGGER_REPLAY_ENABLED "${DAGGER_REPLAY_ENABLED}")"
if [[ "${_dagger_replay_enabled_bool}" == "1" ]]; then
  DAGGER_REPLAY_ENABLED=True
else
  DAGGER_REPLAY_ENABLED=False
fi
unset _dagger_replay_enabled_bool
_fixed_bc_guard_enabled_bool="$(normalize_bool01 FIXED_BC_GUARD_ENABLED "${FIXED_BC_GUARD_ENABLED}")"
if [[ "${_fixed_bc_guard_enabled_bool}" == "1" ]]; then
  FIXED_BC_GUARD_ENABLED=True
else
  FIXED_BC_GUARD_ENABLED=False
fi
unset _fixed_bc_guard_enabled_bool
TORCH_DIST_BACKEND=$(echo "${TORCH_DIST_BACKEND}" | tr '[:upper:]' '[:lower:]')
case "${TORCH_DIST_BACKEND}" in
  nccl|gloo)
    ;;
  *)
    echo "[ERROR] TORCH_DIST_BACKEND must be nccl or gloo. Got: ${TORCH_DIST_BACKEND}" >&2
    exit 2
    ;;
esac
for runtime_bool_name in \
  NCCL_IB_DISABLE \
  TORCH_NCCL_ENABLE_MONITORING \
  TORCH_NCCL_DUMP_ON_TIMEOUT \
  TORCH_NCCL_PROPAGATE_ERROR \
  TORCH_NCCL_DESYNC_DEBUG \
  TORCH_NCCL_ENABLE_TIMING \
  TORCH_NCCL_BLOCKING_WAIT; do
  if [[ "${!runtime_bool_name}" != 0 && "${!runtime_bool_name}" != 1 ]]; then
    echo "[ERROR] ${runtime_bool_name} must be exactly 0 or 1. Got: ${!runtime_bool_name}" >&2
    exit 2
  fi
done
unset runtime_bool_name
if ! canonical_uint_at_most "${TORCH_NCCL_ASYNC_ERROR_HANDLING}" 3; then
  echo "[ERROR] TORCH_NCCL_ASYNC_ERROR_HANDLING must be a canonical integer in [0,3]. Got: ${TORCH_NCCL_ASYNC_ERROR_HANDLING}" >&2
  exit 2
fi
if ! canonical_positive_uint_at_most \
    "${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}" "${MAX_STATUS_SECONDS}"; then
  echo "[ERROR] TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC must be in [1, ${MAX_STATUS_SECONDS}]. Got: ${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}" >&2
  exit 2
fi
if ! canonical_uint_at_most "${TORCH_NCCL_TRACE_BUFFER_SIZE}" "${MAX_SIGNED_32}"; then
  echo "[ERROR] TORCH_NCCL_TRACE_BUFFER_SIZE must be in [0, ${MAX_SIGNED_32}]. Got: ${TORCH_NCCL_TRACE_BUFFER_SIZE}" >&2
  exit 2
fi
if ! canonical_uint_at_most "${NCCL_SOCKET_RETRY_CNT}" "${MAX_SIGNED_32}" \
    || ! canonical_positive_uint_at_most \
      "${NCCL_SOCKET_RETRY_SLEEP_MSEC}" "${MAX_SIGNED_32}" \
    || ! canonical_positive_uint_at_most "${NCCL_SOCKET_NTHREADS}" 16 \
    || ! canonical_positive_uint_at_most "${NCCL_NSOCKS_PERTHREAD}" 16; then
  echo "[ERROR] NCCL socket retry/thread settings are non-canonical or outside safe bounds." >&2
  exit 2
fi
if (( 10#${NCCL_SOCKET_NTHREADS} * 10#${NCCL_NSOCKS_PERTHREAD} > 64 )); then
  echo "[ERROR] NCCL_SOCKET_NTHREADS*NCCL_NSOCKS_PERTHREAD must be <=64." >&2
  exit 2
fi
if [[ ( "${TORCH_DIST_BACKEND}" == "nccl" || "${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE}" == "1" ) \
      && -z "${NCCL_LIB_SHA256}" ]]; then
  echo "[ERROR] NCCL_LIB_SHA256 is required when TORCH_DIST_BACKEND=nccl or HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1." >&2
  exit 2
fi
if [[ -n "${NCCL_LIB_SHA256}" && ! "${NCCL_LIB_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "[ERROR] NCCL_LIB_SHA256 must be a 64-character lowercase SHA256 hex digest. Got: ${NCCL_LIB_SHA256}" >&2
  exit 2
fi
if (( CONTROL_ONLY_ACTION == 0 )); then
  if [[ "${HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY}" != 0 && "${HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY}" != 1 ]]; then
    echo "[ERROR] HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY must be exactly 0 or 1." >&2
    exit 2
  fi
  if [[ -n "${PYTHON_RUNTIME_SITEPACKAGES}" || -n "${PYTHON_RUNTIME_MANIFEST_SHA256}" ]]; then
    if [[ -z "${PYTHON_RUNTIME_SITEPACKAGES}" || -z "${PYTHON_RUNTIME_MANIFEST_SHA256}" ]]; then
      echo "[ERROR] PYTHON_RUNTIME_SITEPACKAGES and PYTHON_RUNTIME_MANIFEST_SHA256 must be set together." >&2
      exit 2
    fi
    if [[ ! "${PYTHON_RUNTIME_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "[ERROR] PYTHON_RUNTIME_MANIFEST_SHA256 must be a 64-character lowercase SHA256 hex digest." >&2
      exit 2
    fi
    _python_runtime_path=$(realpath -m -- "${PYTHON_RUNTIME_SITEPACKAGES}")
    _expected_python_runtime_path="${REMOTE_RUN_ROOT}/.runtime/python/python-runtime-v2-${PYTHON_RUNTIME_MANIFEST_SHA256}/site-packages"
    if [[ "${_python_runtime_path}" != "${_expected_python_runtime_path}" ]]; then
      echo "[ERROR] PYTHON_RUNTIME_SITEPACKAGES must exactly bind its manifest identity: ${_expected_python_runtime_path}" >&2
      exit 2
    fi
    PYTHON_RUNTIME_SITEPACKAGES="${_python_runtime_path}"
    unset _expected_python_runtime_path
  elif [[ "${HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY}" == 1 ]]; then
    echo "[ERROR] This scientific launch requires PYTHON_RUNTIME_SITEPACKAGES and PYTHON_RUNTIME_MANIFEST_SHA256." >&2
    exit 2
  fi

  if [[ -n "${PYTHON_RUNTIME_ARCHIVE}" || -n "${PYTHON_RUNTIME_ARCHIVE_SHA256}" ]]; then
    if [[ -z "${PYTHON_RUNTIME_ARCHIVE}" || -z "${PYTHON_RUNTIME_ARCHIVE_SHA256}" ]]; then
      echo "[ERROR] PYTHON_RUNTIME_ARCHIVE and PYTHON_RUNTIME_ARCHIVE_SHA256 must be set together." >&2
      exit 2
    fi
    if [[ ! "${PYTHON_RUNTIME_ARCHIVE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "[ERROR] PYTHON_RUNTIME_ARCHIVE_SHA256 must be a 64-character lowercase SHA256 hex digest." >&2
      exit 2
    fi
  fi
  if [[ ( "${ACTION}" == prepare || "${ACTION}" == all ) && -n "${PYTHON_RUNTIME_SITEPACKAGES}" && ( -z "${PYTHON_RUNTIME_ARCHIVE}" || -z "${PYTHON_RUNTIME_ARCHIVE_SHA256}" ) ]]; then
    echo "[ERROR] prepare/all requires PYTHON_RUNTIME_ARCHIVE and PYTHON_RUNTIME_ARCHIVE_SHA256 when the runtime overlay is enabled." >&2
    exit 2
  fi
fi
for positive_integer_name in \
  NUM_MINI_BATCHES \
  NUM_LEARNING_EPOCHS \
  FIXED_BC_EVAL_LOG_INTERVAL \
  FIXED_BC_GUARD_CONSECUTIVE_EVALS \
  DAGGER_REPLAY_CAPACITY \
  DAGGER_REPLAY_BATCH_SIZE \
  HOLOSOMA_MOTION_METRICS_INTERVAL \
  HOLOSOMA_COLLECTION_PROFILE_INTERVAL \
  SAVE_INTERVAL \
  PPO_SCHEDULE_STEP_EPOCHS \
  TORCH_DIST_TIMEOUT_SEC \
  HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC \
  PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY \
  PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY \
  PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY \
  PHYSX_GPU_COLLISION_STACK_SIZE; do
  positive_integer_value=${!positive_integer_name}
  if ! canonical_positive_uint_at_most \
      "${positive_integer_value}" "${MAX_SIGNED_32}"; then
    echo "[ERROR] ${positive_integer_name} must be a canonical integer in [1, ${MAX_SIGNED_32}]. Got: ${positive_integer_value}" >&2
    exit 2
  fi
done
if (( CONTROL_ONLY_ACTION == 0 )) && [[ "${SAVE_INTERVAL}" != 1000 ]]; then
  echo "[ERROR] Scientific AS launch requires SAVE_INTERVAL=1000 exactly; got ${SAVE_INTERVAL}." >&2
  exit 2
fi
for nonnegative_integer_name in \
  PPO_START_EPOCH \
  DAGGER_END_EPOCH \
  DAGGER_REPLAY_SEED \
  FIXED_BC_GUARD_REFERENCE_END_EPOCH \
  START_AT_TIMESTEP_ZERO_PROB_START_ITER \
  START_AT_TIMESTEP_ZERO_PROB_END_ITER \
  FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER \
  FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER; do
  nonnegative_integer_value=${!nonnegative_integer_name}
  if ! canonical_uint_at_most \
      "${nonnegative_integer_value}" "${MAX_SIGNED_32}"; then
    echo "[ERROR] ${nonnegative_integer_name} must be a canonical integer in [0, ${MAX_SIGNED_32}]. Got: ${nonnegative_integer_value}" >&2
    exit 2
  fi
done
unset positive_integer_name positive_integer_value
unset nonnegative_integer_name nonnegative_integer_value
if (( 10#${PPO_START_EPOCH} >= 10#${DAGGER_END_EPOCH} )); then
  echo "[ERROR] PPO_START_EPOCH must be < DAGGER_END_EPOCH; got ${PPO_START_EPOCH}>=${DAGGER_END_EPOCH}." >&2
  exit 2
fi
if (( CONTROL_ONLY_ACTION == 0 )) && [[ "${FIXED_BC_GUARD_ENABLED}" == True ]]; then
  if ! canonical_uint_at_most "${FIXED_BC_GUARD_START_EPOCH}" "${MAX_SIGNED_32}"; then
    echo "[ERROR] Enabled FIXED_BC_GUARD_START_EPOCH must be a canonical integer in [0, ${MAX_SIGNED_32}]. Got: ${FIXED_BC_GUARD_START_EPOCH}" >&2
    exit 2
  fi
  if (( 10#${FIXED_BC_GUARD_REFERENCE_END_EPOCH} >= 10#${FIXED_BC_GUARD_START_EPOCH} )); then
    echo "[ERROR] Enabled fixed-BC guard requires FIXED_BC_GUARD_REFERENCE_END_EPOCH < FIXED_BC_GUARD_START_EPOCH; got ${FIXED_BC_GUARD_REFERENCE_END_EPOCH}>=${FIXED_BC_GUARD_START_EPOCH}." >&2
    exit 2
  fi
  if (( 10#${FIXED_BC_GUARD_START_EPOCH} < 10#${DAGGER_END_EPOCH} )); then
    echo "[ERROR] Enabled fixed-BC guard requires FIXED_BC_GUARD_START_EPOCH >= DAGGER_END_EPOCH; got ${FIXED_BC_GUARD_START_EPOCH}<${DAGGER_END_EPOCH}." >&2
    exit 2
  fi
  if (( 10#${FIXED_BC_GUARD_REFERENCE_END_EPOCH} % 10#${FIXED_BC_EVAL_LOG_INTERVAL} != 0 )); then
    echo "[ERROR] Enabled fixed-BC guard requires FIXED_BC_GUARD_REFERENCE_END_EPOCH to be divisible by FIXED_BC_EVAL_LOG_INTERVAL; got ${FIXED_BC_GUARD_REFERENCE_END_EPOCH} % ${FIXED_BC_EVAL_LOG_INTERVAL}." >&2
    exit 2
  fi
  if (( 10#${FIXED_BC_GUARD_START_EPOCH} % 10#${FIXED_BC_EVAL_LOG_INTERVAL} != 0 )); then
    echo "[ERROR] Enabled fixed-BC guard requires FIXED_BC_GUARD_START_EPOCH to be divisible by FIXED_BC_EVAL_LOG_INTERVAL; got ${FIXED_BC_GUARD_START_EPOCH} % ${FIXED_BC_EVAL_LOG_INTERVAL}." >&2
    exit 2
  fi
  if (( 10#${FIXED_BC_GUARD_START_EPOCH} >= 10#${TARGET_LEARNING_ITERATION} )); then
    echo "[ERROR] Enabled fixed-BC guard must start before TARGET_LEARNING_ITERATION; got ${FIXED_BC_GUARD_START_EPOCH}>=${TARGET_LEARNING_ITERATION}." >&2
    exit 2
  fi
  fixed_bc_guard_last_required_eval=$((
    10#${FIXED_BC_GUARD_START_EPOCH}
    + (10#${FIXED_BC_GUARD_CONSECUTIVE_EVALS} - 1) * 10#${FIXED_BC_EVAL_LOG_INTERVAL}
  ))
  if (( fixed_bc_guard_last_required_eval >= 10#${TARGET_LEARNING_ITERATION} )); then
    echo "[ERROR] Enabled fixed-BC guard must have enough evaluations to reach FIXED_BC_GUARD_CONSECUTIVE_EVALS before TARGET_LEARNING_ITERATION; last required evaluation=${fixed_bc_guard_last_required_eval}, target=${TARGET_LEARNING_ITERATION}." >&2
    exit 2
  fi
  unset fixed_bc_guard_last_required_eval
  fixed_bc_reference_eval_count=$((10#${FIXED_BC_GUARD_REFERENCE_END_EPOCH} / 10#${FIXED_BC_EVAL_LOG_INTERVAL} + 1))
  if (( fixed_bc_reference_eval_count < 3 )); then
    echo "[ERROR] Enabled fixed-BC guard requires at least 3 reference evaluations; reference_end=${FIXED_BC_GUARD_REFERENCE_END_EPOCH} interval=${FIXED_BC_EVAL_LOG_INTERVAL} yields ${fixed_bc_reference_eval_count}." >&2
    exit 2
  fi
  unset fixed_bc_reference_eval_count
elif (( CONTROL_ONLY_ACTION == 0 )) && [[ "${FIXED_BC_GUARD_START_EPOCH}" != -1 ]]; then
  echo "[ERROR] Disabled fixed-BC guard requires FIXED_BC_GUARD_START_EPOCH=-1; got ${FIXED_BC_GUARD_START_EPOCH}." >&2
  exit 2
fi
for reset_curriculum_prefix in START_AT_TIMESTEP_ZERO_PROB FREEZE_AT_TIMESTEP_ZERO_PROB; do
  reset_curriculum_start_iter_name="${reset_curriculum_prefix}_START_ITER"
  reset_curriculum_end_iter_name="${reset_curriculum_prefix}_END_ITER"
  reset_curriculum_start_iter=${!reset_curriculum_start_iter_name}
  reset_curriculum_end_iter=${!reset_curriculum_end_iter_name}
  if (( 10#${reset_curriculum_start_iter} > 10#${reset_curriculum_end_iter} )); then
    echo "[ERROR] ${reset_curriculum_start_iter_name} must be <= ${reset_curriculum_end_iter_name}; got ${reset_curriculum_start_iter}>${reset_curriculum_end_iter}." >&2
    exit 2
  fi
  if (( 10#${reset_curriculum_end_iter} > 10#${TARGET_LEARNING_ITERATION} )); then
    echo "[ERROR] ${reset_curriculum_end_iter_name} must be <= TARGET_LEARNING_ITERATION; got ${reset_curriculum_end_iter}>${TARGET_LEARNING_ITERATION}." >&2
    exit 2
  fi
  # Equality cannot be reached by a fresh/actor-only rollout.  Retain it only
  # for full-resume backward compatibility: downstream checkpoint/provenance
  # validation exact-binds that persisted legacy schedule before simulation.
  if (( 10#${reset_curriculum_end_iter} == 10#${TARGET_LEARNING_ITERATION} )) \
      && [[ -z "${RESUME_TRAINING_CKPT}" ]]; then
    echo "[ERROR] Fresh/policy-init ${reset_curriculum_end_iter_name} must be < TARGET_LEARNING_ITERATION so the schedule end is reached by a rollout; got ${reset_curriculum_end_iter}==${TARGET_LEARNING_ITERATION}. Equality is accepted only for an exact-bound legacy full-training resume." >&2
    exit 2
  fi
done
unset reset_curriculum_prefix reset_curriculum_start_iter_name reset_curriculum_end_iter_name
unset reset_curriculum_start_iter reset_curriculum_end_iter

# Validate all controller-owned floating-point experiment knobs before source
# snapshot construction or any remote action.  Python's float parser handles
# scientific notation while explicit finiteness checks reject NaN/Inf.
"${PYTHON_BIN}" - \
  "${ACTOR_LR}" \
  "${CRITIC_LR}" \
  "${PPO_DESIRED_KL}" \
  "${ACTOR_MIN_LR}" \
  "${ACTOR_MAX_LR}" \
  "${CRITIC_MIN_LR}" \
  "${CRITIC_MAX_LR}" \
  "${ACTOR_MIN_NOISE_STD}" \
  "${INIT_NOISE_STD}" \
  "${ENTROPY_COEF}" \
  "${PPO_START_COEFF}" \
  "${PPO_TARGET_COEFF}" \
  "${DAGGER_LOSS_COEF}" \
  "${PPO_START_NOISE_STD}" \
  "${PPO_START_NOISE_STD_UNTIL_COEFF}" \
  "${FIXED_BC_GUARD_MAX_REFERENCE_RATIO}" \
  "${FIXED_BC_GUARD_ABSOLUTE_MAX_MU_MSE}" \
  "${START_AT_TIMESTEP_ZERO_PROB}" \
  "${START_AT_TIMESTEP_ZERO_PROB_END}" \
  "${FREEZE_AT_TIMESTEP_ZERO_PROB}" \
  "${FREEZE_AT_TIMESTEP_ZERO_PROB_END}" \
  "${DAGGER_REPLAY_FRACTION}" \
  "${FIXED_BC_GUARD_ENABLED}" \
  "${FIXED_BC_GUARD_REFERENCE_END_EPOCH}" \
  "${PPO_START_EPOCH}" \
  "${DAGGER_END_EPOCH}" \
  "${PPO_SCHEDULE_STEP_EPOCHS}" \
  "${DAGGER_REPLAY_ENABLED}" \
  "${DAGGER_MATCH_STD}" <<'PY'
from __future__ import annotations

import math
import struct
import sys

names = (
    "ACTOR_LR",
    "CRITIC_LR",
    "PPO_DESIRED_KL",
    "ACTOR_MIN_LR",
    "ACTOR_MAX_LR",
    "CRITIC_MIN_LR",
    "CRITIC_MAX_LR",
    "ACTOR_MIN_NOISE_STD",
    "INIT_NOISE_STD",
    "ENTROPY_COEF",
    "PPO_START_COEFF",
    "PPO_TARGET_COEFF",
    "DAGGER_LOSS_COEF",
    "PPO_START_NOISE_STD",
    "PPO_START_NOISE_STD_UNTIL_COEFF",
    "FIXED_BC_GUARD_MAX_REFERENCE_RATIO",
    "FIXED_BC_GUARD_ABSOLUTE_MAX_MU_MSE",
    "START_AT_TIMESTEP_ZERO_PROB",
    "START_AT_TIMESTEP_ZERO_PROB_END",
    "FREEZE_AT_TIMESTEP_ZERO_PROB",
    "FREEZE_AT_TIMESTEP_ZERO_PROB_END",
    "DAGGER_REPLAY_FRACTION",
)
float_args = sys.argv[1 : 1 + len(names)]
guard_args = sys.argv[1 + len(names) :]
raw_values = dict(zip(names, float_args, strict=True))
values: dict[str, float] = {}
for name, raw in raw_values.items():
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SystemExit(f"[ERROR] {name} must be numeric. Got: {raw}") from exc
    if not math.isfinite(value):
        raise SystemExit(f"[ERROR] {name} must be finite. Got: {raw}")
    values[name] = value

for name in (
    "ACTOR_LR",
    "CRITIC_LR",
    "PPO_DESIRED_KL",
    "ACTOR_MIN_LR",
    "ACTOR_MAX_LR",
    "CRITIC_MIN_LR",
    "CRITIC_MAX_LR",
    "ACTOR_MIN_NOISE_STD",
    "INIT_NOISE_STD",
    "DAGGER_LOSS_COEF",
    "PPO_START_NOISE_STD",
    "FIXED_BC_GUARD_MAX_REFERENCE_RATIO",
    "FIXED_BC_GUARD_ABSOLUTE_MAX_MU_MSE",
):
    if values[name] <= 0.0:
        raise SystemExit(f"[ERROR] {name} must be finite and > 0. Got: {raw_values[name]}")

for optimizer_name in ("ACTOR", "CRITIC"):
    initial_name = f"{optimizer_name}_LR"
    minimum_name = f"{optimizer_name}_MIN_LR"
    maximum_name = f"{optimizer_name}_MAX_LR"
    initial = values[initial_name]
    minimum = values[minimum_name]
    maximum = values[maximum_name]
    if minimum > initial:
        raise SystemExit(
            f"[ERROR] {minimum_name} must be <= {initial_name}; "
            f"got {raw_values[minimum_name]}>{raw_values[initial_name]}."
        )
    if initial > maximum:
        raise SystemExit(
            f"[ERROR] {initial_name} must be <= {maximum_name}; "
            f"got {raw_values[initial_name]}>{raw_values[maximum_name]}."
        )
    if minimum > maximum:
        raise SystemExit(
            f"[ERROR] {minimum_name} must be <= {maximum_name}; "
            f"got {raw_values[minimum_name]}>{raw_values[maximum_name]}."
        )
if values["ENTROPY_COEF"] < 0.0:
    raise SystemExit(
        f"[ERROR] ENTROPY_COEF must be finite and >= 0. Got: {raw_values['ENTROPY_COEF']}"
    )
if values["FIXED_BC_GUARD_MAX_REFERENCE_RATIO"] < 1.0:
    raise SystemExit(
        "[ERROR] FIXED_BC_GUARD_MAX_REFERENCE_RATIO must be finite and >= 1. "
        f"Got: {raw_values['FIXED_BC_GUARD_MAX_REFERENCE_RATIO']}"
    )
if not 0.0 < values["DAGGER_REPLAY_FRACTION"] < 1.0:
    raise SystemExit(
        "[ERROR] DAGGER_REPLAY_FRACTION must be finite and strictly between 0 and 1. "
        f"Got: {raw_values['DAGGER_REPLAY_FRACTION']}"
    )
for name in (
    "PPO_START_COEFF",
    "PPO_TARGET_COEFF",
    "PPO_START_NOISE_STD_UNTIL_COEFF",
    "START_AT_TIMESTEP_ZERO_PROB",
    "START_AT_TIMESTEP_ZERO_PROB_END",
    "FREEZE_AT_TIMESTEP_ZERO_PROB",
    "FREEZE_AT_TIMESTEP_ZERO_PROB_END",
):
    if not 0.0 <= values[name] <= 1.0:
        raise SystemExit(
            f"[ERROR] {name} must be a finite probability in [0, 1]. Got: {raw_values[name]}"
        )
if values["PPO_START_COEFF"] > values["PPO_TARGET_COEFF"]:
    raise SystemExit(
        "[ERROR] PPO_START_COEFF must be <= PPO_TARGET_COEFF; "
        f"got {raw_values['PPO_START_COEFF']}>{raw_values['PPO_TARGET_COEFF']}."
    )


def operational_float32(value: float) -> float:
    try:
        return float(struct.unpack("!f", struct.pack("!f", value))[0])
    except OverflowError:
        return math.copysign(float("inf"), value)


operational_ppo_endpoints = {
    name: operational_float32(max(0.0, min(1.0, values[name])))
    for name in ("PPO_START_COEFF", "PPO_TARGET_COEFF")
}
for name, operational_value in operational_ppo_endpoints.items():
    if values[name] > 0.0 and operational_value <= 0.0:
        raise SystemExit(
            f"[ERROR] {name} is positive as a Python scalar but rounds to zero in the "
            f"float32 PPO actor loss graph: Python={raw_values[name]}, float32={operational_value}."
        )
if values["INIT_NOISE_STD"] < values["ACTOR_MIN_NOISE_STD"]:
    raise SystemExit(
        "[ERROR] INIT_NOISE_STD must be >= ACTOR_MIN_NOISE_STD; "
        f"got {raw_values['INIT_NOISE_STD']}<{raw_values['ACTOR_MIN_NOISE_STD']}."
    )
if values["PPO_START_NOISE_STD"] < values["ACTOR_MIN_NOISE_STD"]:
    raise SystemExit(
        "[ERROR] PPO_START_NOISE_STD must be >= ACTOR_MIN_NOISE_STD; "
        f"got {raw_values['PPO_START_NOISE_STD']}<{raw_values['ACTOR_MIN_NOISE_STD']}."
    )

(
    guard_enabled,
    reference_end_raw,
    ppo_start_raw,
    dagger_end_raw,
    step_raw,
    replay_enabled,
    dagger_match_std,
) = guard_args
if replay_enabled == "True":
    if guard_enabled != "True":
        raise SystemExit(
            "[ERROR] DAGGER_REPLAY_ENABLED=True requires FIXED_BC_GUARD_ENABLED=True and "
            "the non-empty typed fixed-BC dataset."
        )
    if dagger_match_std != "False":
        raise SystemExit(
            "[ERROR] DAGGER_REPLAY_ENABLED=True requires DAGGER_MATCH_STD=False because "
            "the replay schema contains no teacher std."
        )
    nonzero_endpoints = {
        name: value for name, value in operational_ppo_endpoints.items() if value != 0.0
    }
    if nonzero_endpoints:
        raise SystemExit(
            "[ERROR] DAGGER_REPLAY_ENABLED=True requires operational float32 PPO to remain "
            f"exactly zero for the entire target; got {nonzero_endpoints}."
        )
if guard_enabled == "True":
    reference_end = int(reference_end_raw)
    ppo_start = int(ppo_start_raw)
    dagger_end = int(dagger_end_raw)
    step_epochs = int(step_raw)
    if reference_end < ppo_start:
        reference_coeff = 0.0
    elif reference_end >= dagger_end:
        reference_coeff = values["PPO_TARGET_COEFF"]
    else:
        total_epochs = max(1, dagger_end - ppo_start)
        total_steps = max(1, (total_epochs + step_epochs - 1) // step_epochs)
        completed_steps = max(0, (reference_end - ppo_start) // step_epochs)
        progress = min(float(completed_steps) / float(total_steps), 1.0)
        reference_coeff = values["PPO_START_COEFF"] + progress * (
            values["PPO_TARGET_COEFF"] - values["PPO_START_COEFF"]
        )
    if reference_coeff != 0.0:
        raise SystemExit(
            "[ERROR] Enabled fixed-BC guard reference window must remain pure BC: "
            f"PPO coefficient at reference_end={reference_end} is {reference_coeff}."
        )
PY
if [[ "${MAX_RESTARTS}" != 0 ]]; then
  echo "[ERROR] Scientific launch requires MAX_RESTARTS=0 exactly; torchrun worker restart replays from the original launch checkpoint/fresh state rather than the latest exact distributed state. Got: ${MAX_RESTARTS}." >&2
  exit 2
fi
STUDENT_MOTION_END_MODE=$(echo "${STUDENT_MOTION_END_MODE}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
case "${STUDENT_MOTION_END_MODE}" in
  episodic|continuing)
    ;;
  *)
    echo "[ERROR] STUDENT_MOTION_END_MODE must be episodic or continuing. Got: ${STUDENT_MOTION_END_MODE}" >&2
    exit 2
    ;;
esac
if [[ -n "${CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION_WAS_SET}" ]]; then
  _contact_comp_bool="$(normalize_bool01 CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION "${CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION}")"
  if [[ "${_contact_comp_bool}" == "1" ]]; then
    CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION=True
  else
    CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION=False
  fi
fi

if [[ -n "${UNIFORM_T1_WINDOW_TARGET_SAMPLE_FRAC}" ]]; then
  ensure_training_python
  if ! "${PYTHON_BIN}" - "${UNIFORM_T1_WINDOW_TARGET_SAMPLE_FRAC}" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit(1) from exc
raise SystemExit(0 if math.isfinite(value) and 0.0 <= value <= 1.0 else 1)
PY
  then
    echo "[ERROR] UNIFORM_T1_WINDOW_TARGET_SAMPLE_FRAC must be a finite probability in [0, 1]. Got: ${UNIFORM_T1_WINDOW_TARGET_SAMPLE_FRAC}" >&2
    exit 2
  fi
fi

if [[ "${TEACHER_CHECKPOINT}" == https://wandb.ai/* && "${TEACHER_CHECKPOINT%%\?*}" != https://wandb.ai/*/runs/*/files/*.pt ]]; then
  echo "[ERROR] TEACHER_CHECKPOINT must identify an exact .pt artifact; bare W&B run URLs are not reproducible. Got: ${TEACHER_CHECKPOINT}" >&2
  exit 2
fi
if [[ "${TEACHER_CHECKPOINT}" == https://wandb.ai/* ]]; then
  TEACHER_CHECKPOINT="$(normalize_wandb_file_ref "${TEACHER_CHECKPOINT}")"
fi
case "${TEACHER_CHECKPOINT}" in
  wandb://*/*/*/*.pt)
    ;;
  wandb://*)
    echo "[ERROR] TEACHER_CHECKPOINT must use wandb://<entity>/<project>/<run>/<file.pt>. Got: ${TEACHER_CHECKPOINT}" >&2
    exit 2
    ;;
  *.pt)
    if [[ ! -f "${TEACHER_CHECKPOINT}" ]]; then
      echo "[ERROR] Control-local TEACHER_CHECKPOINT does not exist: ${TEACHER_CHECKPOINT}" >&2
      exit 2
    fi
    CONTROL_TEACHER_CHECKPOINT_PATH=$(realpath "${TEACHER_CHECKPOINT}")
    CONTROL_TEACHER_CHECKPOINT_SHA256=$(sha256sum "${CONTROL_TEACHER_CHECKPOINT_PATH}" | awk '{print $1}')
    if [[ ! "${CONTROL_TEACHER_CHECKPOINT_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "[ERROR] Failed to compute TEACHER_CHECKPOINT SHA256: ${CONTROL_TEACHER_CHECKPOINT_PATH}" >&2
      exit 2
    fi
    TEACHER_CHECKPOINT="${CONTROL_TEACHER_CHECKPOINT_PATH}"
    echo "[INFO] control_teacher_checkpoint=${CONTROL_TEACHER_CHECKPOINT_PATH} sha256=${CONTROL_TEACHER_CHECKPOINT_SHA256}"
    ;;
  *)
    echo "[ERROR] TEACHER_CHECKPOINT must identify an exact .pt artifact; bare W&B run URLs are not reproducible. Got: ${TEACHER_CHECKPOINT}" >&2
    exit 2
    ;;
esac

if [[ -n "${RESUME_TRAINING_CKPT}" ]]; then
  RESUME_TRAINING_CKPT="$(normalize_wandb_file_ref "${RESUME_TRAINING_CKPT}")"
fi
if [[ -n "${RESUME_TRAINING_CKPT}" ]]; then
  case "${RESUME_TRAINING_CKPT}" in
    wandb://*/*/*/*.pt)
      ;;
    wandb://*)
      echo "[ERROR] RESUME_TRAINING_CKPT must use wandb://<entity>/<project>/<run>/<file.pt>. Got: ${RESUME_TRAINING_CKPT}" >&2
      exit 2
      ;;
    *.pt)
      if [[ ! -f "${RESUME_TRAINING_CKPT}" ]]; then
        echo "[ERROR] Control-local RESUME_TRAINING_CKPT does not exist: ${RESUME_TRAINING_CKPT}" >&2
        exit 2
      fi
      CONTROL_RESUME_TRAINING_PATH=$(realpath "${RESUME_TRAINING_CKPT}")
      CONTROL_RESUME_TRAINING_SHA256=$(sha256sum "${CONTROL_RESUME_TRAINING_PATH}" | awk '{print $1}')
      if [[ ! "${CONTROL_RESUME_TRAINING_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "[ERROR] Failed to compute RESUME_TRAINING_CKPT SHA256: ${CONTROL_RESUME_TRAINING_PATH}" >&2
        exit 2
      fi
      RESUME_TRAINING_CKPT="${CONTROL_RESUME_TRAINING_PATH}"
      echo "[INFO] control_training_resume=${CONTROL_RESUME_TRAINING_PATH} sha256=${CONTROL_RESUME_TRAINING_SHA256}"
      ;;
    *)
      echo "[ERROR] RESUME_TRAINING_CKPT must be a .pt checkpoint path or wandb:// URI. Got: ${RESUME_TRAINING_CKPT}" >&2
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
case "$(echo "${WANDB_RESUME_SAME_RUN}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    WANDB_RESUME_SAME_RUN=1
    ;;
  0|false|no|off|"")
    WANDB_RESUME_SAME_RUN=0
    ;;
  *)
    echo "[ERROR] WANDB_RESUME_SAME_RUN must be a boolean. Got: ${WANDB_RESUME_SAME_RUN}" >&2
    exit 2
    ;;
esac
if [[ -n "${RESUME_WANDB_RUN_ID}" && "${WANDB_RESUME_SAME_RUN}" != "1" ]]; then
  echo "[ERROR] RESUME_WANDB_RUN_ID requires WANDB_RESUME_SAME_RUN=1; refusing contradictory same-run identity settings." >&2
  exit 2
fi
if [[ -n "${RESUME_WANDB_RUN_ID}" && -z "${RESUME_TRAINING_CKPT}" ]]; then
  echo "[ERROR] RESUME_WANDB_RUN_ID requests same-run logging resume but RESUME_TRAINING_CKPT is empty." >&2
  exit 2
fi
if [[ "${WANDB_RESUME_SAME_RUN}" == "1" && -z "${RESUME_TRAINING_CKPT}" ]]; then
  echo "[ERROR] WANDB_RESUME_SAME_RUN=1 requires RESUME_TRAINING_CKPT." >&2
  exit 2
fi
if [[ "${WANDB_RESUME_SAME_RUN}" == "1" && -z "${RESUME_WANDB_RUN_ID}" ]]; then
  echo "[ERROR] WANDB_RESUME_SAME_RUN=1 requires RESUME_WANDB_RUN_ID." >&2
  exit 2
fi
if [[ -n "${FRESH_WANDB_RUN_ID}" && -n "${RESUME_WANDB_RUN_ID}" ]]; then
  echo "[ERROR] FRESH_WANDB_RUN_ID and RESUME_WANDB_RUN_ID are mutually exclusive." >&2
  exit 2
fi
if [[ -n "${FRESH_WANDB_RUN_ID}" && -n "${RESUME_TRAINING_CKPT}" ]]; then
  echo "[ERROR] FRESH_WANDB_RUN_ID is only valid for a fresh training trajectory; RESUME_TRAINING_CKPT must be empty." >&2
  exit 2
fi
if [[ -n "${FRESH_WANDB_RUN_ID}" && "${WANDB_RESUME_SAME_RUN}" != "0" ]]; then
  echo "[ERROR] FRESH_WANDB_RUN_ID pre-binds logging identity only; WANDB_RESUME_SAME_RUN must remain 0." >&2
  exit 2
fi
if [[ -n "${FRESH_WANDB_RUN_ID}" ]]; then
  if [[ -z "${REPLAY_PREFLIGHT_MANIFEST}" || -z "${REPLAY_PREFLIGHT_MANIFEST_SHA256}" ]]; then
    echo "[ERROR] FRESH_WANDB_RUN_ID requires REPLAY_PREFLIGHT_MANIFEST and REPLAY_PREFLIGHT_MANIFEST_SHA256." >&2
    exit 2
  fi
  if [[ ! "${REPLAY_PREFLIGHT_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[ERROR] REPLAY_PREFLIGHT_MANIFEST_SHA256 must be 64 lowercase SHA256 hex." >&2
    exit 2
  fi
  if [[ ! -f "${REPLAY_PREFLIGHT_MANIFEST}" || -L "${REPLAY_PREFLIGHT_MANIFEST}" ]]; then
    echo "[ERROR] Replay preflight manifest must be a regular non-symlink file: ${REPLAY_PREFLIGHT_MANIFEST}" >&2
    exit 2
  fi
  REPLAY_PREFLIGHT_MANIFEST=$(realpath -e -- "${REPLAY_PREFLIGHT_MANIFEST}")
elif [[ -n "${REPLAY_PREFLIGHT_MANIFEST}" || -n "${REPLAY_PREFLIGHT_MANIFEST_SHA256}" ]]; then
  echo "[ERROR] Replay preflight manifest identity was supplied without FRESH_WANDB_RUN_ID." >&2
  exit 2
fi
if [[ -n "${HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET}" \
      && "${RESUME_FROM_BOX}" != "1" ]]; then
  echo "[ERROR] HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET requires RESUME_FROM_BOX=1." >&2
  exit 2
fi
if [[ -n "${HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET}" ]]; then
  if [[ -z "${BOX_POLICY_INIT_EXPECTED_SHA256}" ]]; then
    echo "[ERROR] The required terminal policy-init gate requires BOX_POLICY_INIT_EXPECTED_SHA256." >&2
    exit 2
  fi
  if [[ -z "${BOX_POLICY_INIT_CONTROL_CACHE_ROOT}" \
        || "${BOX_POLICY_INIT_CONTROL_CACHE_ROOT}" != /* ]]; then
    echo "[ERROR] The required terminal policy-init gate requires an absolute BOX_POLICY_INIT_CONTROL_CACHE_ROOT." >&2
    exit 2
  fi
  if [[ -z "${BOX_POLICY_INIT_EXPECTED_WORLD_SIZE}" ]]; then
    echo "[ERROR] The required terminal policy-init gate requires BOX_POLICY_INIT_EXPECTED_WORLD_SIZE." >&2
    exit 2
  fi
  if ! canonical_positive_uint_at_most "${NPROC}" "${MAX_NPROC_PER_NODE}" \
      || ! canonical_positive_uint_at_most "${NNODES}" "${MAX_TOTAL_GPUS}" \
      || (( 10#${NPROC} * 10#${NNODES} > MAX_TOTAL_GPUS )); then
    echo "[ERROR] The required terminal policy-init gate cannot bind an invalid active NNODES*NPROC topology." >&2
    exit 2
  fi
  if (( 10#${BOX_POLICY_INIT_EXPECTED_WORLD_SIZE} != 10#${NPROC} * 10#${NNODES} )); then
    echo "[ERROR] BOX_POLICY_INIT_EXPECTED_WORLD_SIZE must equal the active NNODES*NPROC topology." >&2
    exit 2
  fi
  if [[ -z "${BOX_POLICY_INIT_EXPECTED_WANDB_RUN_PATH}" ]]; then
    echo "[ERROR] The required terminal policy-init gate requires BOX_POLICY_INIT_EXPECTED_WANDB_RUN_PATH." >&2
    exit 2
  fi
  if [[ "${BOX_POLICY_INIT_EXPECTED_WANDB_RUN_PATH}" != "${WANDB_ENTITY}/carry-any/"* ]]; then
    echo "[ERROR] The required terminal policy-init W&B source must belong to the active WANDB_ENTITY/carry-any project." >&2
    exit 2
  fi
  if [[ -z "${BOX_POLICY_INIT_EXPECTED_SOURCE_SNAPSHOT_ID}" ]]; then
    echo "[ERROR] The required terminal policy-init gate requires BOX_POLICY_INIT_EXPECTED_SOURCE_SNAPSHOT_ID." >&2
    exit 2
  fi
  if [[ -z "${SOURCE_SNAPSHOT_ID}" \
        || "${SOURCE_SNAPSHOT_ID}" != "${BOX_POLICY_INIT_EXPECTED_SOURCE_SNAPSHOT_ID}" ]]; then
    echo "[ERROR] The required terminal policy-init source snapshot must equal the pinned formal SOURCE_SNAPSHOT_ID." >&2
    exit 2
  fi
  if [[ "${BOX_POLICY_INIT_REQUIRE_FRESH_SOURCE}" != 1 ]]; then
    echo "[ERROR] The required terminal policy-init gate requires BOX_POLICY_INIT_REQUIRE_FRESH_SOURCE=1." >&2
    exit 2
  fi
fi
if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  case "${BOX_POLICY_INIT_REF}" in
    wandb://*|https://wandb.ai/*)
      if [[ -n "${HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET}" ]]; then
        echo "[ERROR] The required terminal policy-init gate accepts only an authenticated control-local checkpoint, not a remote artifact reference." >&2
        exit 2
      fi
      # Preserve remote artifact references; every node resolves the same named W&B file.
      ;;
    *)
      if [[ -n "${HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET}" ]]; then
        if [[ "${BOX_POLICY_INIT_REF}" != /* ]]; then
          echo "[ERROR] The required terminal policy-init gate requires an absolute control-local BOX_POLICY_INIT_REF." >&2
          exit 2
        fi
        if [[ -L "${BOX_POLICY_INIT_REF}" ]]; then
          echo "[ERROR] The required terminal policy-init gate refuses a symlink BOX_POLICY_INIT_REF." >&2
          exit 2
        fi
      fi
      if [[ "${BOX_POLICY_INIT_REF}" != *.pt ]]; then
        echo "[ERROR] A control-local BOX_POLICY_INIT_REF must name a .pt checkpoint. Got: ${BOX_POLICY_INIT_REF}" >&2
        exit 2
      fi
      if [[ ! -f "${BOX_POLICY_INIT_REF}" ]]; then
        echo "[ERROR] Control-local BOX_POLICY_INIT_REF does not exist: ${BOX_POLICY_INIT_REF}" >&2
        exit 2
      fi
      if [[ -n "${HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET}" ]]; then
        ensure_training_python
        "${PYTHON_BIN}" scripts/validate_terminal_policy_init.py \
          --checkpoint "${BOX_POLICY_INIT_REF}" \
          --cache-root "${BOX_POLICY_INIT_CONTROL_CACHE_ROOT}" \
          --expected-sha256 "${BOX_POLICY_INIT_EXPECTED_SHA256}" \
          --require-terminal-target "${HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET}" \
          --expected-world-size "${BOX_POLICY_INIT_EXPECTED_WORLD_SIZE}" \
          --expected-wandb-run-path "${BOX_POLICY_INIT_EXPECTED_WANDB_RUN_PATH}" \
          --expected-source-snapshot-id "${BOX_POLICY_INIT_EXPECTED_SOURCE_SNAPSHOT_ID}" \
          --require-fresh-source
      fi
      CONTROL_BOX_POLICY_INIT_PATH=$(realpath "${BOX_POLICY_INIT_REF}")
      CONTROL_BOX_POLICY_INIT_SHA256=$(sha256sum "${CONTROL_BOX_POLICY_INIT_PATH}" | awk '{print $1}')
      if [[ ! "${CONTROL_BOX_POLICY_INIT_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "[ERROR] Failed to compute BOX_POLICY_INIT_REF SHA256: ${CONTROL_BOX_POLICY_INIT_PATH}" >&2
        exit 2
      fi
      if [[ -n "${BOX_POLICY_INIT_EXPECTED_SHA256}" \
            && "${CONTROL_BOX_POLICY_INIT_SHA256}" != "${BOX_POLICY_INIT_EXPECTED_SHA256}" ]]; then
        echo "[ERROR] Control-local BOX_POLICY_INIT_REF does not match BOX_POLICY_INIT_EXPECTED_SHA256." >&2
        exit 2
      fi
      BOX_POLICY_INIT_EXPECTED_SHA256="${CONTROL_BOX_POLICY_INIT_SHA256}"
      BOX_POLICY_INIT_REF="${CONTROL_BOX_POLICY_INIT_PATH}"
      echo "[INFO] control_box_policy_init=${CONTROL_BOX_POLICY_INIT_PATH} sha256=${CONTROL_BOX_POLICY_INIT_SHA256}"
      ;;
  esac
fi

checkpoint_actor_hidden_dims() {
  local checkpoint="$1"
  local expected_sha256="${2:-}"
  ensure_training_python
  "${PYTHON_BIN}" - "${checkpoint}" "${expected_sha256}" <<'PY'
from __future__ import annotations

import json
import sys

from holosoma.utils.checkpoint_validation import (
    load_verified_torch_checkpoint,
    validate_student_actor_contract,
)

checkpoint, _ = load_verified_torch_checkpoint(
    sys.argv[1],
    expected_sha256=sys.argv[2] or None,
    map_location="cpu",
)
try:
    actor = checkpoint["experiment_config"]["algo"]["config"]["module_dict"]["actor"]
except (KeyError, TypeError) as exc:
    raise SystemExit(
        "[ERROR] Resume checkpoint has no actor contract metadata; "
        "set STUDENT_ACTOR_HIDDEN_DIMS explicitly."
    ) from exc
try:
    dims = validate_student_actor_contract(actor)["hidden_dims"]
except ValueError as exc:
    raise SystemExit(f"[ERROR] {exc}") from exc
print(json.dumps(list(dims), separators=(",", ":")))
PY
}

if [[ -z "${STUDENT_ACTOR_HIDDEN_DIMS_WAS_SET}" ]]; then
  if [[ -n "${RESUME_TRAINING_CKPT}" ]]; then
    if [[ "${RESUME_TRAINING_CKPT}" != wandb://* && -f "${RESUME_TRAINING_CKPT}" ]]; then
      STUDENT_ACTOR_HIDDEN_DIMS="$(checkpoint_actor_hidden_dims \
        "${RESUME_TRAINING_CKPT}" "${CONTROL_RESUME_TRAINING_SHA256}")"
      echo "[INFO] Inferred actor hidden dims from training-resume checkpoint: ${STUDENT_ACTOR_HIDDEN_DIMS}"
    else
      STUDENT_ACTOR_HIDDEN_DIMS=""
      echo "[INFO] Actor hidden dims will be inferred from the node-local staged training-resume checkpoint."
    fi
  elif [[ "${RESUME_FROM_BOX}" == "1" ]]; then
    STUDENT_ACTOR_HIDDEN_DIMS="${BOX_POLICY_INIT_ACTOR_HIDDEN_DIMS}"
  else
    STUDENT_ACTOR_HIDDEN_DIMS="[2048,1024,512,256,128]"
  fi
fi
if [[ -z "${STUDENT_ACTOR_HIDDEN_DIMS}" && -z "${RESUME_TRAINING_CKPT}" ]]; then
  echo "[ERROR] STUDENT_ACTOR_HIDDEN_DIMS cannot be empty; strict checkpoint loading requires the saved architecture." >&2
  exit 2
fi
if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  _student_dims_compact="$(echo "${STUDENT_ACTOR_HIDDEN_DIMS}" | tr -d '[:space:]')"
  _box_dims_compact="$(echo "${BOX_POLICY_INIT_ACTOR_HIDDEN_DIMS}" | tr -d '[:space:]')"
  if [[ "${_student_dims_compact}" != "${_box_dims_compact}" ]]; then
    echo "[ERROR] RESUME_FROM_BOX=1 requires actor hidden dims ${BOX_POLICY_INIT_ACTOR_HIDDEN_DIMS}; got ${STUDENT_ACTOR_HIDDEN_DIMS}." >&2
    echo "[ERROR] Set BOX_POLICY_INIT_ACTOR_HIDDEN_DIMS only when checkpoint metadata confirms another exact architecture." >&2
    exit 2
  fi
fi

if [[ -n "${LOCAL_BANK_NAME}" ]]; then
  if [[ -n "${CORL_SOLID80_BANK_NAME_WAS_SET}" && "${CORL_SOLID80_BANK_NAME}" != "${LOCAL_BANK_NAME}" ]]; then
    echo "[ERROR] LOCAL_BANK_NAME and CORL_SOLID80_BANK_NAME must identify the same installed training bank." >&2
    echo "[ERROR] Got LOCAL_BANK_NAME=${LOCAL_BANK_NAME}, CORL_SOLID80_BANK_NAME=${CORL_SOLID80_BANK_NAME}." >&2
    exit 2
  fi
  CORL_SOLID80_BANK_NAME="${LOCAL_BANK_NAME}"
fi
if [[ -n "${NFS_CORL_BANK}" && -z "${LOCAL_BANK_NAME}" ]]; then
  echo "[ERROR] NFS_CORL_BANK requires LOCAL_BANK_NAME so prepare and training use one explicit bank identity." >&2
  exit 2
fi
NFS_CH_BANK=${NFS_CH_BANK:-/nfs/zzzihanw/ds_as_data/_distill/${CH_BANK_NAME}.tar}
if [[ -z "${PREPARE_COPY_SCRIPT}" ]]; then
  if [[ -n "${NFS_CORL_BANK}" || -n "${LOCAL_BANK_NAME}" || -n "${EXPECTED_CLIP_COUNT}" ]]; then
    PREPARE_COPY_SCRIPT=cp_corl
  else
    PREPARE_COPY_SCRIPT=cp_ch
  fi
fi
case "${PREPARE_COPY_SCRIPT}" in
  cp_ch|cp_corl)
    ;;
  *)
    echo "[ERROR] PREPARE_COPY_SCRIPT must be cp_ch or cp_corl. Got: ${PREPARE_COPY_SCRIPT}" >&2
    exit 2
    ;;
esac
if [[ "${PREPARE_COPY_SCRIPT}" == "cp_ch" && ( -n "${NFS_CORL_BANK}" || -n "${LOCAL_BANK_NAME}" ) ]]; then
  echo "[ERROR] PREPARE_COPY_SCRIPT=cp_ch cannot prepare a custom NFS_CORL_BANK/LOCAL_BANK_NAME for training." >&2
  echo "[ERROR] Use PREPARE_COPY_SCRIPT=cp_corl so the installed and trained bank identities remain identical." >&2
  exit 2
fi
case "${REMOTE_DATA_PACKAGE_CACHE}" in
  /*)
    ;;
  *)
    echo "[ERROR] REMOTE_DATA_PACKAGE_CACHE must be an absolute path. Got: ${REMOTE_DATA_PACKAGE_CACHE}" >&2
    exit 2
    ;;
esac
REMOTE_DATA_PACKAGE_CACHE=$(realpath -m "${REMOTE_DATA_PACKAGE_CACHE}")
REMOTE_REPO_NORMALIZED=$(realpath -m "${REMOTE_REPO}")
case "${REMOTE_DATA_PACKAGE_CACHE}" in
  "${REMOTE_REPO_NORMALIZED}"|"${REMOTE_REPO_NORMALIZED}"/*)
    echo "[ERROR] REMOTE_DATA_PACKAGE_CACHE must be outside the mutable asset/source repository: ${REMOTE_REPO}" >&2
    exit 2
    ;;
esac
fi

if (( CONTROL_ONLY_ACTION == 0 )); then
  if ! canonical_positive_uint_at_most \
      "${MIN_PER_GPU_ENVS}" "${MAX_SIGNED_32}"; then
    echo "[ERROR] MIN_PER_GPU_ENVS must be a canonical integer in [1, ${MAX_SIGNED_32}]. Got: ${MIN_PER_GPU_ENVS}" >&2
    exit 2
  fi
  if ! canonical_positive_uint_at_most \
      "${PER_GPU_ENVS}" "${MAX_SIGNED_32}" \
      || (( 10#${PER_GPU_ENVS} < 10#${MIN_PER_GPU_ENVS} )); then
    echo "[ERROR] PER_GPU_ENVS must be an integer >= ${MIN_PER_GPU_ENVS}. Got: ${PER_GPU_ENVS}" >&2
    exit 2
  fi
fi
if ! canonical_positive_uint_at_most \
    "${NPROC}" "${MAX_NPROC_PER_NODE}"; then
  echo "[ERROR] NPROC must be a canonical integer in [1, ${MAX_NPROC_PER_NODE}]. Got: ${NPROC}" >&2
  exit 2
fi
if (( CONTROL_ONLY_ACTION == 0 )) \
    && ! [[ "${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE}" =~ ^[01]$ ]]; then
  echo "[ERROR] TORCH_ALLOW_TF32_CUBLAS_OVERRIDE must be exactly 0 or 1 because PyTorch's pre-start c10 flag parser does not accept other boolean spellings. Got: ${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE}" >&2
  exit 2
fi
if ! canonical_positive_uint_at_most "${NNODES}" "${MAX_TOTAL_GPUS}" \
    || [[ "${NNODES}" != "${#NODE_LIST[@]}" ]]; then
  echo "[ERROR] NNODES must equal node list length. Got NNODES=${NNODES}, nodes=${#NODE_LIST[@]}" >&2
  exit 2
fi
TOTAL_GPUS=$((10#${NPROC} * 10#${NNODES}))
if (( TOTAL_GPUS > MAX_TOTAL_GPUS )); then
  echo "[ERROR] NPROC*NNODES must be <= ${MAX_TOTAL_GPUS}; got ${NPROC}*${NNODES}=${TOTAL_GPUS}." >&2
  exit 2
fi
if (( CONTROL_ONLY_ACTION == 0 )); then
  if (( 10#${PER_GPU_ENVS} > MAX_SIGNED_32 / TOTAL_GPUS )); then
    echo "[ERROR] PER_GPU_ENVS*TOTAL_GPUS must be <= ${MAX_SIGNED_32}; got ${PER_GPU_ENVS}*${TOTAL_GPUS}." >&2
    exit 2
  fi
  TOTAL_NUM_ENVS=$((10#${PER_GPU_ENVS} * TOTAL_GPUS))
else
  TOTAL_NUM_ENVS=0
fi
if (( CONTROL_ONLY_ACTION == 0 )) && [[ -n "${TRAINING_SEED}" ]]; then
  if [[ ! "${TRAINING_SEED}" =~ ^[0-9]+$ ]] \
      || (( ${#TRAINING_SEED} > 10 )) \
      || (( 10#${TRAINING_SEED} > 4294967295 )); then
    echo "[ERROR] TRAINING_SEED/SEED must be an integer in [0, 4294967295]. Got: ${TRAINING_SEED}" >&2
    exit 2
  fi
  _training_world_size=${TOTAL_GPUS}
  _max_training_base_seed=$((4294967295 - _training_world_size + 1))
  if (( 10#${TRAINING_SEED} > _max_training_base_seed )); then
    echo "[ERROR] TRAINING_SEED plus rank offsets must stay <= 4294967295. Got seed=${TRAINING_SEED}, world_size=${_training_world_size}, max_base=${_max_training_base_seed}" >&2
    exit 2
  fi
  unset _training_world_size _max_training_base_seed
fi

validate_gradient_reduce_contracts() {
  if [[ "${HOLOSOMA_GLOO_GRAD_REDUCE}" == "1" \
        && "${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE}" == "1" ]]; then
    echo "[ERROR] HOLOSOMA_GLOO_GRAD_REDUCE and HOLOSOMA_HIERARCHICAL_GRAD_REDUCE are mutually exclusive." >&2
    return 2
  fi
  if [[ "${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER}" == "1" \
        && "${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE}" != "1" ]]; then
    echo "[ERROR] HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER requires HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1." >&2
    return 2
  fi
  if [[ "${HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES}" == "1" \
        && ( "${HOLOSOMA_GLOO_SMALL_COLLECTIVES}" != "1" \
             || "${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE}" != "1" ) ]]; then
    echo "[ERROR] HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=1 requires HOLOSOMA_GLOO_SMALL_COLLECTIVES=1 and HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1." >&2
    return 2
  fi
  if [[ "${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE}" == "1" ]] \
      && (( NPROC <= 1 || NNODES <= 1 )); then
    local world_size=$((NPROC * NNODES))
    echo "[ERROR] HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1 requires NPROC>1 and NNODES>1; got world_size=${world_size}, local_world_size=${NPROC}, NNODES=${NNODES}." >&2
    return 2
  fi
}

validate_restart_contract() {
  if [[ "${RESTART}" == "1" ]]; then
    echo "[ERROR] RESTART=1 is not a supported transactional operation. Run 'batch_ne.sh stop', verify every node stopped, then launch with RESTART=0." >&2
    return 2
  fi
}

if [[ -z "${MASTER_ADDR}" ]]; then
  echo "[ERROR] MASTER_ADDR cannot be empty." >&2
  exit 2
fi
if [[ ! "${MASTER_ADDR}" =~ ^[A-Za-z0-9][A-Za-z0-9_.:-]{0,254}$ ]]; then
  echo "[ERROR] MASTER_ADDR must be a safe host/IP identifier: ${MASTER_ADDR}" >&2
  exit 2
fi
if [[ "${MASTER_ADDR}" != "${NODE_LIST[0]}" ]]; then
  echo "[ERROR] MASTER_ADDR must exactly equal the rank-0 host NODES[0] so controller reservation state and rank-0 clean-completion release use the same host. Got MASTER_ADDR=${MASTER_ADDR}, NODES[0]=${NODE_LIST[0]}." >&2
  exit 2
fi
if ! [[ "${MASTER_PORT}" =~ ^[1-9][0-9]{0,4}$ ]] || (( 10#${MASTER_PORT} > 65535 )); then
  echo "[ERROR] MASTER_PORT must be an integer in [1, 65535]. Got: ${MASTER_PORT}" >&2
  exit 2
fi
if [[ -z "${HOLOSOMA_PROVENANCE_MASTER_PORT}" ]]; then
  if (( 10#${MASTER_PORT} == 65535 )); then
    echo "[ERROR] MASTER_PORT=65535 requires an explicit distinct HOLOSOMA_PROVENANCE_MASTER_PORT." >&2
    exit 2
  fi
  HOLOSOMA_PROVENANCE_MASTER_PORT=$((10#${MASTER_PORT} + 1))
fi
if ! [[ "${HOLOSOMA_PROVENANCE_MASTER_PORT}" =~ ^[1-9][0-9]{0,4}$ ]] \
    || (( 10#${HOLOSOMA_PROVENANCE_MASTER_PORT} > 65535 )); then
  echo "[ERROR] HOLOSOMA_PROVENANCE_MASTER_PORT must be an integer in [1, 65535]. Got: ${HOLOSOMA_PROVENANCE_MASTER_PORT}" >&2
  exit 2
fi
if (( 10#${HOLOSOMA_PROVENANCE_MASTER_PORT} == 10#${MASTER_PORT} )); then
  echo "[ERROR] MASTER_PORT and HOLOSOMA_PROVENANCE_MASTER_PORT must be distinct." >&2
  exit 2
fi
if ! [[ "${SESSION}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "[ERROR] SESSION must be a safe basename of 1-128 characters: ${SESSION}" >&2
  exit 2
fi
if ! [[ "${RUN_STAMP}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "[ERROR] RUN_STAMP must be a safe basename of 1-128 characters: ${RUN_STAMP}" >&2
  exit 2
fi
if (( ${#SESSION} + 1 + ${#RUN_STAMP} > 255 )); then
  echo "[ERROR] SESSION_RUN_STAMP basename exceeds the 255-byte portable filesystem limit." >&2
  exit 2
fi
_legacy_stop_identity_fields=(
  LEGACY_STOP_EXPECTED_SNAPSHOT_ID
  LEGACY_STOP_EXPECTED_TOKEN
  LEGACY_STOP_EXPECTED_EPOCH
  LEGACY_STOP_EXPECTED_RUN_STAMP
  LEGACY_STOP_EXPECTED_TARGET
)
_legacy_stop_identity_count=0
for _legacy_stop_name in "${_legacy_stop_identity_fields[@]}"; do
  [[ -n "${!_legacy_stop_name}" ]] && _legacy_stop_identity_count=$((_legacy_stop_identity_count + 1))
done
if (( _legacy_stop_identity_count != 0 )); then
  if [[ "${ACTION}" != stop || _legacy_stop_identity_count -ne ${#_legacy_stop_identity_fields[@]} ]]; then
    echo "[ERROR] A legacy stop requires all five LEGACY_STOP_EXPECTED_* fields and ACTION=stop." >&2
    exit 2
  fi
  if [[ ! "${LEGACY_STOP_EXPECTED_SNAPSHOT_ID}" =~ ^src-[0-9a-f]{64}$ \
        || ! "${LEGACY_STOP_EXPECTED_TOKEN}" =~ ^[0-9a-f]{64}$ \
        || ! "${LEGACY_STOP_EXPECTED_EPOCH}" =~ ^[1-9][0-9]*$ \
        || ! "${LEGACY_STOP_EXPECTED_RUN_STAMP}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] \
      || ! canonical_positive_uint_at_most \
        "${LEGACY_STOP_EXPECTED_EPOCH}" "${MAX_SAFE_EPOCH}" \
      || ! canonical_positive_uint_at_most \
        "${LEGACY_STOP_EXPECTED_TARGET}" "${MAX_SIGNED_32}"; then
    echo "[ERROR] LEGACY_STOP_EXPECTED_* fields are malformed or outside their scientific bounds." >&2
    exit 2
  fi
  if (( ${#SESSION} + 1 + ${#LEGACY_STOP_EXPECTED_RUN_STAMP} > 255 )); then
    echo "[ERROR] Legacy SESSION_RUN_STAMP basename exceeds the portable filesystem limit." >&2
    exit 2
  fi
fi
unset _legacy_stop_identity_fields _legacy_stop_identity_count _legacy_stop_name
if ! canonical_positive_uint_at_most \
    "${STATUS_STARTUP_GRACE_SECONDS}" "${MAX_LAUNCH_SECONDS}"; then
  echo "[ERROR] STATUS_STARTUP_GRACE_SECONDS must be a canonical integer in [1, ${MAX_LAUNCH_SECONDS}]. Got: ${STATUS_STARTUP_GRACE_SECONDS}" >&2
  exit 2
fi
for _startup_integer_name in \
  LAUNCH_STARTUP_TIMEOUT_SECONDS \
  LAUNCH_STARTUP_PROBE_TIMEOUT_SECONDS \
  LAUNCH_STARTUP_POLL_SECONDS \
  LAUNCH_STARTUP_STABILITY_SECONDS \
  LAUNCH_CLEANUP_TIMEOUT_SECONDS \
  LAUNCH_LOCK_TIMEOUT_SECONDS \
  LAUNCH_CONTROL_TIMEOUT_SECONDS \
  LAUNCH_PREFLIGHT_TIMEOUT_SECONDS; do
  _startup_integer_value=${!_startup_integer_name}
  if ! canonical_positive_uint_at_most \
      "${_startup_integer_value}" "${MAX_LAUNCH_SECONDS}"; then
    echo "[ERROR] ${_startup_integer_name} must be a canonical integer in [1, ${MAX_LAUNCH_SECONDS}]. Got: ${_startup_integer_value}" >&2
    exit 2
  fi
done
unset _startup_integer_name _startup_integer_value
if (( LAUNCH_STARTUP_TIMEOUT_SECONDS <= LAUNCH_STARTUP_STABILITY_SECONDS )); then
  echo "[ERROR] LAUNCH_STARTUP_TIMEOUT_SECONDS must be greater than LAUNCH_STARTUP_STABILITY_SECONDS." >&2
  exit 2
fi
if (( LAUNCH_CLEANUP_TIMEOUT_SECONDS <= LAUNCH_LOCK_TIMEOUT_SECONDS )); then
  echo "[ERROR] LAUNCH_CLEANUP_TIMEOUT_SECONDS must be greater than LAUNCH_LOCK_TIMEOUT_SECONDS." >&2
  exit 2
fi
if (( LAUNCH_CONTROL_TIMEOUT_SECONDS <= LAUNCH_LOCK_TIMEOUT_SECONDS )); then
  echo "[ERROR] LAUNCH_CONTROL_TIMEOUT_SECONDS must be greater than LAUNCH_LOCK_TIMEOUT_SECONDS." >&2
  exit 2
fi
if (( LAUNCH_PREFLIGHT_TIMEOUT_SECONDS <= LAUNCH_LOCK_TIMEOUT_SECONDS )); then
  echo "[ERROR] LAUNCH_PREFLIGHT_TIMEOUT_SECONDS must be greater than LAUNCH_LOCK_TIMEOUT_SECONDS." >&2
  exit 2
fi

if (( CONTROL_ONLY_ACTION == 0 )); then
  RUN_BANK_LABEL=${RUN_BANK_LABEL:-${LOCAL_BANK_NAME:-${CORL_SOLID80_BANK_NAME}}}
  RUN_BANK_LABEL=${RUN_BANK_LABEL//[^A-Za-z0-9_.-]/_}
  if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
    RUN_NAME=${RUN_NAME:-g1_w_object_distill_as_button_solid_${RUN_BANK_LABEL}_${TOTAL_GPUS}gpu_init_box}
    TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_button_solid_${RUN_BANK_LABEL}_${TOTAL_GPUS}gpu_init_box_depth}
  else
    RUN_NAME=${RUN_NAME:-g1_w_object_distill_as_button_solid_${RUN_BANK_LABEL}_${TOTAL_GPUS}gpu}
    TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_button_solid_${RUN_BANK_LABEL}_${TOTAL_GPUS}gpu_depth}
  fi
  # The completion record is a deliberately whitespace-delimited protocol
  # consumed by startup/status/clean-completion checks.  LOGGER_BASE_DIR and
  # training.name are components of the emitted checkpoint/ONNX paths, so
  # accepting whitespace or control bytes here would train successfully and
  # then make the required final artifact record unparsable.
  for _completion_path_field in LOGGER_BASE_DIR TRAINING_NAME; do
    _completion_path_value=${!_completion_path_field}
    if [[ "${_completion_path_value}" =~ [[:space:][:cntrl:]] ]]; then
      echo "[ERROR] ${_completion_path_field} cannot contain whitespace or control characters because it is part of the machine-readable completion path: ${_completion_path_value@Q}" >&2
      exit 2
    fi
  done
  unset _completion_path_field _completion_path_value
  # training.name is interpolated into the experiment directory.  Slashes or
  # dot-segments can otherwise create nested/traversing paths even though the
  # final completion line remains whitespace-parseable.
  if [[ ! "${TRAINING_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
    echo "[ERROR] TRAINING_NAME must be a safe basename of 1-128 characters: ${TRAINING_NAME@Q}" >&2
    exit 2
  fi
  SCHEDULE_NAME=${SCHEDULE_NAME:-as_ch51_sparse_root_ppo_first_contact_drop_button_solid}
  STUDENT_ACTOR_HIDDEN_DIMS_NOTE=${STUDENT_ACTOR_HIDDEN_DIMS:-checkpoint/default}
  if [[ -n "${RESUME_TRAINING_CKPT}" ]]; then
    PPO_ROLLOUT_NOTE="PPO rollout uses the coefficient recomputed for the loaded learning iteration"
  else
    PPO_ROLLOUT_NOTE="PPO rollout uses the configured iteration-0 coefficient from the first batch"
  fi
  SCHEDULE_NOTES=${SCHEDULE_NOTES:-"${NNODES} nodes x ${NPROC} GPUs AS solid distillation on ${RUN_BANK_LABEL}, PER_GPU_ENVS=${PER_GPU_ENVS}, actor hidden dims ${STUDENT_ACTOR_HIDDEN_DIMS_NOTE}. ${PPO_ROLLOUT_NOTE} and follows ${PPO_START_COEFF}->${PPO_TARGET_COEFF}, step=${PPO_SCHEDULE_STEP_EPOCHS}, end=${DAGGER_END_EPOCH}; effective BC weight is ${DAGGER_LOSS_COEF}*(1-PPO). Bounded rank-local DAgger replay is enabled=${DAGGER_REPLAY_ENABLED}, capacity=${DAGGER_REPLAY_CAPACITY}, batch=${DAGGER_REPLAY_BATCH_SIZE}, fraction=${DAGGER_REPLAY_FRACTION}, seed=${DAGGER_REPLAY_SEED}; enabled replay is restricted to operational PPO=0 and a disjoint fixed-BC gate. PPO actor-LR controller is ${PPO_LR_SCHEDULE} with desired_kl=${PPO_DESIRED_KL}, actor=${ACTOR_LR} bounds=[${ACTOR_MIN_LR},${ACTOR_MAX_LR}], critic=${CRITIC_LR} bounds=[${CRITIC_MIN_LR},${CRITIC_MAX_LR}]. The fixed-BC guard freezes its reference by ${FIXED_BC_GUARD_REFERENCE_END_EPOCH} and from ${FIXED_BC_GUARD_START_EPOCH} fails closed after ${FIXED_BC_GUARD_CONSECUTIVE_EVALS} consecutive values above min(reference*${FIXED_BC_GUARD_MAX_REFERENCE_RATIO}, ${FIXED_BC_GUARD_ABSOLUTE_MAX_MU_MSE}). Teacher std matching defaults off; early PPO std is capped at ${PPO_START_NOISE_STD} through coefficient ${PPO_START_NOISE_STD_UNTIL_COEFF}. Contact T1 windows reweight the failure-adaptive timestep density according to the configured multiplicative/target-mass mode; start-at-zero is an explicit reset mixture and AS metrics report the effective distribution. Object geometry is validated from each URDF rather than inferred from the bank name."}
  if [[ "${HOLOSOMA_COLLECTION_PROFILE_CANARY}" == 1 ]]; then
    SCHEDULE_NOTES+=" Diagnostic collection profiler enabled (sync_cuda=${HOLOSOMA_COLLECTION_PROFILE_SYNC_CUDA}, interval=${HOLOSOMA_COLLECTION_PROFILE_INTERVAL}); timings are intrusive and are not production-throughput measurements."
  fi
fi
LOG_DIR="logs/batch_ne/${SESSION}_${RUN_STAMP}"

quote() {
  printf '%q' "$1"
}

cleanup_controller_python_runtime_transfer() {
  local root="${PYTHON_RUNTIME_CONTROLLER_TRANSFER_ROOT:-}"
  local archive="${PYTHON_RUNTIME_CONTROLLER_TRANSFER_ARCHIVE:-}"
  local fingerprint="${PYTHON_RUNTIME_CONTROLLER_TRANSFER_FINGERPRINT:-}"
  [[ -n "${root}" && -n "${fingerprint}" ]] || return 0
  if [[ ! -d "${root}" || -L "${root}" \
        || "$(realpath -e -- "${root}" 2>/dev/null || true)" != "${root}" \
        || "$(stat -c '%d:%i:%u:%a' -- "${root}" 2>/dev/null || true)" != "${fingerprint}" ]]; then
    echo "[ERROR] Controller Python runtime transfer root changed; refusing cleanup." >&2
    return 1
  fi
  if [[ -n "${archive}" && "${archive%/*}" == "${root}" \
        && ( -e "${archive}" || -L "${archive}" ) ]]; then
    if [[ ! -f "${archive}" || -L "${archive}" \
          || "$(stat -c '%h:%u' -- "${archive}" 2>/dev/null || true)" != "1:$(id -u)" ]]; then
      echo "[ERROR] Controller Python runtime transfer archive changed; refusing cleanup." >&2
      return 1
    fi
    chmod 600 -- "${archive}"
    rm -f -- "${archive}"
  fi
  if ! rmdir -- "${root}"; then
    echo "[ERROR] Controller Python runtime transfer root is not empty after cleanup." >&2
    return 1
  fi
  PYTHON_RUNTIME_CONTROLLER_TRANSFER_ROOT=""
  PYTHON_RUNTIME_CONTROLLER_TRANSFER_ARCHIVE=""
  PYTHON_RUNTIME_CONTROLLER_TRANSFER_FINGERPRINT=""
}

trap cleanup_controller_python_runtime_transfer EXIT

resolve_distill_as_entrypoint_identity() {
  if [[ -n "${DISTILL_AS_ENTRYPOINT_SHA256}" ]]; then
    return 0
  fi
  if [[ -z "${SOURCE_SNAPSHOT_ARCHIVE}" \
        || ! -f "${SOURCE_SNAPSHOT_ARCHIVE}" \
        || -L "${SOURCE_SNAPSHOT_ARCHIVE}" ]]; then
    echo "[ERROR] Cannot bind DISTILL_AS_ENTRYPOINT without the authenticated source snapshot archive." >&2
    return 2
  fi

  # Resolve the executable wrapper from the exact archive that will be
  # installed, not from the mutable controller checkout.  Also require the
  # archive's signed manifest to authenticate the same bytes exactly once.
  if ! DISTILL_AS_ENTRYPOINT_SHA256=$("${PYTHON_BIN}" - \
      "${SOURCE_SNAPSHOT_ARCHIVE}" "${DISTILL_AS_ENTRYPOINT}" <<'PY'
from __future__ import annotations

import hashlib
import sys
import tarfile


archive_path, entrypoint = sys.argv[1:3]
with tarfile.open(archive_path, mode="r:gz") as archive:
    members = [
        member
        for member in archive.getmembers()
        if member.name.removeprefix("./") == entrypoint
    ]
    if len(members) != 1:
        raise SystemExit(
            "[ERROR] Source snapshot must contain the selected DISTILL_AS_ENTRYPOINT "
            f"exactly once: entrypoint={entrypoint!r} matches={len(members)}"
        )
    member = members[0]
    if not member.isfile() or member.issym() or member.islnk():
        raise SystemExit(
            "[ERROR] Selected DISTILL_AS_ENTRYPOINT is not one regular non-link "
            f"snapshot member: {entrypoint!r}"
        )
    source = archive.extractfile(member)
    if source is None:
        raise SystemExit(
            f"[ERROR] Could not read selected DISTILL_AS_ENTRYPOINT: {entrypoint!r}"
        )
    digest = hashlib.sha256(source.read()).hexdigest()

    manifest_members = [
        candidate
        for candidate in archive.getmembers()
        if candidate.name.removeprefix("./")
        == ".holosoma_snapshot/source_manifest.sha256"
    ]
    if len(manifest_members) != 1 or not manifest_members[0].isfile():
        raise SystemExit("[ERROR] Source snapshot has no unique regular source manifest.")
    manifest_source = archive.extractfile(manifest_members[0])
    if manifest_source is None:
        raise SystemExit("[ERROR] Could not read source snapshot manifest.")
    manifest = manifest_source.read().decode("utf-8")
    expected = f"{digest}  ./{entrypoint}"
    if manifest.splitlines().count(expected) != 1:
        raise SystemExit(
            "[ERROR] Selected DISTILL_AS_ENTRYPOINT bytes are not authenticated "
            f"exactly once by the source manifest: {entrypoint!r}"
        )
print(digest)
PY
  ); then
    return 2
  fi
  if [[ ! "${DISTILL_AS_ENTRYPOINT_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[ERROR] Failed to compute selected DISTILL_AS_ENTRYPOINT SHA256." >&2
    return 2
  fi
  DISTILL_AS_ENTRYPOINT_PATH="${RUN_REPO}/${DISTILL_AS_ENTRYPOINT}"
  echo "[INFO] distill_as_entrypoint=${DISTILL_AS_ENTRYPOINT} path=${DISTILL_AS_ENTRYPOINT_PATH} sha256=${DISTILL_AS_ENTRYPOINT_SHA256} formal_fresh=${DISTILL_AS_FORMAL_FRESH}"
}

ensure_local_source_snapshot() {
  if [[ -n "${RUN_REPO}" ]]; then
    resolve_distill_as_entrypoint_identity
    return 0
  fi

  mkdir -p "${SOURCE_SNAPSHOT_CACHE}"
  if [[ -z "${SOURCE_SNAPSHOT_ID}" && -z "${SOURCE_SNAPSHOT_ARCHIVE}" ]]; then
    local snapshot_record
    snapshot_record=$(bash "${SCRIPT_DIR}/scripts/build_run_snapshot.sh" \
      --repo-root "${SCRIPT_DIR}" \
      --cache-root "${SOURCE_SNAPSHOT_CACHE}")
    IFS=$'\t' read -r \
      SOURCE_SNAPSHOT_ID \
      SOURCE_SNAPSHOT_ARCHIVE \
      SOURCE_SNAPSHOT_ARCHIVE_SHA256 \
      SOURCE_MANIFEST_SHA256 <<< "${snapshot_record}"
  else
    if [[ -z "${SOURCE_SNAPSHOT_ARCHIVE}" ]]; then
      SOURCE_SNAPSHOT_ARCHIVE="${SOURCE_SNAPSHOT_CACHE}/${SOURCE_SNAPSHOT_ID}.tar.gz"
    fi
    if [[ ! -f "${SOURCE_SNAPSHOT_ARCHIVE}" ]]; then
      echo "[ERROR] Pinned source snapshot archive does not exist: ${SOURCE_SNAPSHOT_ARCHIVE}" >&2
      exit 2
    fi
    local archive_snapshot_id
    local archive_manifest_sha256
    archive_snapshot_id=$(tar -xOzf "${SOURCE_SNAPSHOT_ARCHIVE}" ./.holosoma_snapshot/id)
    archive_manifest_sha256=$(
      tar -xOzf "${SOURCE_SNAPSHOT_ARCHIVE}" ./.holosoma_snapshot/source_manifest.sha256 \
        | sha256sum \
        | awk '{print $1}'
    )
    if [[ -z "${SOURCE_SNAPSHOT_ID}" ]]; then
      SOURCE_SNAPSHOT_ID="${archive_snapshot_id}"
    elif [[ "${SOURCE_SNAPSHOT_ID}" != "${archive_snapshot_id}" ]]; then
      echo "[ERROR] SOURCE_SNAPSHOT_ID=${SOURCE_SNAPSHOT_ID} does not match archive id=${archive_snapshot_id}." >&2
      exit 2
    fi
    SOURCE_MANIFEST_SHA256="${archive_manifest_sha256}"
    local actual_archive_sha256
    actual_archive_sha256=$(sha256sum "${SOURCE_SNAPSHOT_ARCHIVE}" | awk '{print $1}')
    if [[ -n "${SOURCE_SNAPSHOT_ARCHIVE_SHA256}" && "${SOURCE_SNAPSHOT_ARCHIVE_SHA256}" != "${actual_archive_sha256}" ]]; then
      echo "[ERROR] SOURCE_SNAPSHOT_ARCHIVE_SHA256 does not match ${SOURCE_SNAPSHOT_ARCHIVE}." >&2
      exit 2
    fi
    SOURCE_SNAPSHOT_ARCHIVE_SHA256="${actual_archive_sha256}"
  fi

  if [[ ! "${SOURCE_SNAPSHOT_ID}" =~ ^src-[0-9a-f]{64}$ ]]; then
    echo "[ERROR] Invalid content-addressed SOURCE_SNAPSHOT_ID: ${SOURCE_SNAPSHOT_ID}" >&2
    exit 2
  fi
  if [[ ! "${SOURCE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ || "${SOURCE_SNAPSHOT_ID}" != "src-${SOURCE_MANIFEST_SHA256}" ]]; then
    echo "[ERROR] Snapshot id does not match the source manifest digest." >&2
    exit 2
  fi
  if [[ ! "${SOURCE_SNAPSHOT_ARCHIVE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[ERROR] Invalid source snapshot archive SHA256: ${SOURCE_SNAPSHOT_ARCHIVE_SHA256}" >&2
    exit 2
  fi
  RUN_REPO="${REMOTE_RUN_ROOT}/${SOURCE_SNAPSHOT_ID}"
  echo "[INFO] source_snapshot_id=${SOURCE_SNAPSHOT_ID} source_manifest_sha256=${SOURCE_MANIFEST_SHA256}"
  echo "[INFO] source_snapshot_archive=${SOURCE_SNAPSHOT_ARCHIVE} archive_sha256=${SOURCE_SNAPSHOT_ARCHIVE_SHA256}"
  echo "[INFO] remote_run_repo=${RUN_REPO} remote_asset_repo=${REMOTE_REPO}"
  resolve_distill_as_entrypoint_identity
}

resolve_minibatch_throughput_contract() {
  if [[ -z "${SOURCE_SNAPSHOT_ARCHIVE}" || ! -f "${SOURCE_SNAPSHOT_ARCHIVE}" ]]; then
    echo "[ERROR] Cannot resolve PPO throughput without the authenticated source snapshot archive." >&2
    return 2
  fi

  # num_steps_per_env remains owned by the PPO source configuration.  Read the
  # exact literal from the content-addressed snapshot that will run instead of
  # copying its current value into a second launcher knob that could drift.
  if ! PPO_NUM_STEPS_PER_ENV=$("${PYTHON_BIN}" - "${SOURCE_SNAPSHOT_ARCHIVE}" <<'PY'
import ast
import sys
import tarfile


archive_path = sys.argv[1]
source_path = "src/holosoma/holosoma/config_values/algo.py"


def call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


with tarfile.open(archive_path, mode="r:gz") as archive:
    matches = [
        member
        for member in archive.getmembers()
        if member.name.removeprefix("./") == source_path
    ]
    if len(matches) != 1 or not matches[0].isfile():
        raise SystemExit(
            "[ERROR] Source snapshot must contain exactly one regular "
            f"{source_path}; found={len(matches)}."
        )
    if matches[0].size > 1_048_576:
        raise SystemExit(f"[ERROR] Refusing oversized PPO config source: {matches[0].size} bytes.")
    stream = archive.extractfile(matches[0])
    if stream is None:
        raise SystemExit(f"[ERROR] Could not read {source_path} from source snapshot.")
    source = stream.read().decode("utf-8")

tree = ast.parse(source, filename=source_path)
ppo_assignments = [
    statement.value
    for statement in tree.body
    if isinstance(statement, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "ppo" for target in statement.targets)
]
if len(ppo_assignments) != 1:
    raise SystemExit(
        "[ERROR] Expected exactly one top-level ppo assignment in the snapshot PPO config; "
        f"found={len(ppo_assignments)}."
    )
ppo_call = ppo_assignments[0]
if not isinstance(ppo_call, ast.Call) or call_name(ppo_call.func) != "PPOAlgoConfig":
    raise SystemExit("[ERROR] Snapshot ppo assignment is not a PPOAlgoConfig call.")
config_values = [keyword.value for keyword in ppo_call.keywords if keyword.arg == "config"]
if len(config_values) != 1:
    raise SystemExit("[ERROR] Snapshot PPOAlgoConfig must have exactly one config keyword.")
config_call = config_values[0]
if not isinstance(config_call, ast.Call) or call_name(config_call.func) != "PPOConfig":
    raise SystemExit("[ERROR] Snapshot PPOAlgoConfig.config is not a PPOConfig call.")
horizon_values = [
    keyword.value for keyword in config_call.keywords if keyword.arg == "num_steps_per_env"
]
if len(horizon_values) != 1:
    raise SystemExit(
        "[ERROR] Snapshot PPOConfig must define num_steps_per_env exactly once; "
        f"found={len(horizon_values)}."
    )
horizon_node = horizon_values[0]
if (
    not isinstance(horizon_node, ast.Constant)
    or isinstance(horizon_node.value, bool)
    or not isinstance(horizon_node.value, int)
    or horizon_node.value <= 0
):
    raise SystemExit(
        "[ERROR] Snapshot PPOConfig.num_steps_per_env must be a positive integer literal."
    )
print(horizon_node.value)
PY
  ); then
    echo "[ERROR] Failed to resolve PPOConfig.num_steps_per_env from source snapshot ${SOURCE_SNAPSHOT_ID}." >&2
    return 2
  fi
  if ! canonical_positive_uint_at_most \
      "${PPO_NUM_STEPS_PER_ENV}" "${MAX_SIGNED_32}"; then
    echo "[ERROR] Snapshot PPOConfig.num_steps_per_env is outside [1, ${MAX_SIGNED_32}]: ${PPO_NUM_STEPS_PER_ENV}" >&2
    return 2
  fi

  PPO_RANK_LOCAL_ROLLOUT_SAMPLES=$((10#${PER_GPU_ENVS} * 10#${PPO_NUM_STEPS_PER_ENV}))
  PPO_GLOBAL_ROLLOUT_SAMPLES=$((10#${TOTAL_NUM_ENVS} * 10#${PPO_NUM_STEPS_PER_ENV}))
  if (( PPO_RANK_LOCAL_ROLLOUT_SAMPLES % 10#${NUM_MINI_BATCHES} != 0 )); then
    echo "[ERROR] Rank-local rollout samples must be divisible by NUM_MINI_BATCHES so PPO drops no samples: PER_GPU_ENVS=${PER_GPU_ENVS} * snapshot_num_steps_per_env=${PPO_NUM_STEPS_PER_ENV} = ${PPO_RANK_LOCAL_ROLLOUT_SAMPLES}, NUM_MINI_BATCHES=${NUM_MINI_BATCHES}." >&2
    return 2
  fi
  if (( PPO_GLOBAL_ROLLOUT_SAMPLES % 10#${NUM_MINI_BATCHES} != 0 )); then
    echo "[ERROR] Global rollout samples are not divisible by NUM_MINI_BATCHES: global_rollout_samples=${PPO_GLOBAL_ROLLOUT_SAMPLES}, NUM_MINI_BATCHES=${NUM_MINI_BATCHES}." >&2
    return 2
  fi
  PPO_RANK_LOCAL_SAMPLES_PER_MINIBATCH_UPDATE=$((PPO_RANK_LOCAL_ROLLOUT_SAMPLES / 10#${NUM_MINI_BATCHES}))
  PPO_GLOBAL_SAMPLES_PER_MINIBATCH_UPDATE=$((PPO_GLOBAL_ROLLOUT_SAMPLES / 10#${NUM_MINI_BATCHES}))
  PPO_MINIBATCH_UPDATE_ROUNDS_PER_ITERATION=$((10#${NUM_MINI_BATCHES} * 10#${NUM_LEARNING_EPOCHS}))

  if [[ "${HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY}" == 1 ]]; then
    echo "[WARN] minibatch_throughput_canary=enabled num_mini_batches=16 semantics=changes_Adam_and_PPO_update_trajectory math_equivalent=0"
  else
    echo "[INFO] minibatch_throughput_canary=disabled num_mini_batches=${NUM_MINI_BATCHES}"
  fi
  echo "[INFO] ppo_minibatch_throughput source=snapshot:PPOConfig.num_steps_per_env snapshot_id=${SOURCE_SNAPSHOT_ID} num_steps_per_env=${PPO_NUM_STEPS_PER_ENV} rank_local_rollout_samples=${PPO_RANK_LOCAL_ROLLOUT_SAMPLES} global_rollout_samples=${PPO_GLOBAL_ROLLOUT_SAMPLES} rank_local_samples_per_minibatch_update=${PPO_RANK_LOCAL_SAMPLES_PER_MINIBATCH_UPDATE} global_samples_per_minibatch_update=${PPO_GLOBAL_SAMPLES_PER_MINIBATCH_UPDATE} minibatch_update_rounds_per_iteration=${PPO_MINIBATCH_UPDATE_ROUNDS_PER_ITERATION} num_learning_epochs=${NUM_LEARNING_EPOCHS}"
}

load_replay_external_as_contract() {
  if [[ "${REPLAY_PREFLIGHT_REQUIRED_VERSION}" != 2 ]]; then
    return 0
  fi
  # Formal-fresh dry runs intentionally render the complete sealed launch
  # command before a W&B identity/replay artifact exists.  They never execute
  # the remote barrier or embedded revalidator, so preserve that controller
  # preview contract only when both replay identity fields are absent.  Every
  # real launch (and any dry run claiming a manifest) remains fail-closed.
  if [[ "${DRY_RUN}" == 1 \
        && -z "${REPLAY_PREFLIGHT_MANIFEST}" \
        && -z "${REPLAY_PREFLIGHT_MANIFEST_SHA256}" ]]; then
    echo "[DRY_RUN] Rule-90 v2 replay bytes will be required before a real formal-fresh launch."
    return 0
  fi
  if [[ ! -f "${REPLAY_PREFLIGHT_MANIFEST}" || -L "${REPLAY_PREFLIGHT_MANIFEST}" ]]; then
    echo "[ERROR] Rule-90 v2 external-AS preflight requires a regular non-symlink replay manifest." >&2
    return 2
  fi
  local actual_manifest_sha256
  actual_manifest_sha256=$(sha256sum -- "${REPLAY_PREFLIGHT_MANIFEST}" | awk '{print $1}')
  if [[ "${actual_manifest_sha256}" != "${REPLAY_PREFLIGHT_MANIFEST_SHA256}" ]]; then
    echo "[ERROR] Replay preflight manifest SHA256 mismatch before external-AS validation: actual=${actual_manifest_sha256} expected=${REPLAY_PREFLIGHT_MANIFEST_SHA256}" >&2
    return 2
  fi
  local contract_output
  if ! contract_output=$("${PYTHON_BIN}" - "${REPLAY_PREFLIGHT_MANIFEST}" <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
try:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"[ERROR] Could not parse Rule-90 v2 replay manifest inputs: {exc}") from exc
if not isinstance(payload, dict) or payload.get("version") != 2:
    raise SystemExit("[ERROR] External-AS replay contract requires manifest version 2")
inputs = payload.get("inputs")
if not isinstance(inputs, dict):
    raise SystemExit("[ERROR] Rule-90 v2 replay manifest has no inputs object")
fields = (
    "motion_npz_sha256",
    "object_map_sha256",
    "object_urdf_sha256",
    "object_mesh_sha256",
    "single_slot_source_digest",
    "single_slot_view_digest",
    "rank_shard_source_digest",
)
clip_id = inputs.get("motion_clip_id")
if not isinstance(clip_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}", clip_id):
    raise SystemExit("[ERROR] Rule-90 v2 inputs.motion_clip_id is not one safe clip identifier")
values = [clip_id]
for field in fields:
    value = inputs.get(field)
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SystemExit(f"[ERROR] Rule-90 v2 inputs.{field} is not a lowercase SHA256 digest")
    values.append(value)
print("\t".join(values))
PY
  ); then
    return 2
  fi
  if [[ -z "${contract_output}" || "${contract_output}" == *$'\n'* ]]; then
    echo "[ERROR] Rule-90 v2 replay input parser returned a malformed record." >&2
    return 2
  fi
  IFS=$'\t' read -r \
    REPLAY_AS_MOTION_CLIP_ID \
    REPLAY_AS_MOTION_NPZ_SHA256 \
    REPLAY_AS_OBJECT_MAP_SHA256 \
    REPLAY_AS_OBJECT_URDF_SHA256 \
    REPLAY_AS_OBJECT_MESH_SHA256 \
    REPLAY_AS_SINGLE_SLOT_SOURCE_DIGEST \
    REPLAY_AS_SINGLE_SLOT_VIEW_DIGEST \
    REPLAY_AS_RANK_SHARD_SOURCE_DIGEST <<<"${contract_output}"
  local parsed_value
  for parsed_value in \
      "${REPLAY_AS_MOTION_NPZ_SHA256}" \
      "${REPLAY_AS_OBJECT_MAP_SHA256}" \
      "${REPLAY_AS_OBJECT_URDF_SHA256}" \
      "${REPLAY_AS_OBJECT_MESH_SHA256}" \
      "${REPLAY_AS_SINGLE_SLOT_SOURCE_DIGEST}" \
      "${REPLAY_AS_SINGLE_SLOT_VIEW_DIGEST}" \
      "${REPLAY_AS_RANK_SHARD_SOURCE_DIGEST}"; do
    if [[ ! "${parsed_value}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "[ERROR] Rule-90 v2 replay input parser did not return its exact digest closure." >&2
      return 2
    fi
  done
}

# Reject an accidentally selected busy/protected node before any expensive
# runtime installation, external-AS materialization/hash walk, W&B request, or
# lifecycle write.  This probe is deliberately read-only: it uses only the
# network-interface inventory and NVML-backed nvidia-smi queries, and never
# imports torch (which could create a CUDA context on the node being checked).
preflight_selected_gpus_idle_node() {
  local node="$1"
  local probe_phase="${2:-early}"
  case "${probe_phase}" in
    early|pre-launch)
      ;;
    *)
      echo "[ERROR] Internal selected-GPU idle probe phase is invalid: ${probe_phase}" >&2
      return 2
      ;;
  esac
  local runtime_path
  runtime_path="$(dirname -- "${PYTHON_BIN}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  local body cmd
  body=$(cat <<'REMOTE'
set -euo pipefail
unset BASH_ENV ENV CDPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONOPTIMIZE PYTHONWARNINGS PYTHONBREAKPOINT PYTHONSAFEPATH
unset LD_PRELOAD
export PATH="$RUNTIME_PATH"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LC_ALL=C
ip link show "$NCCL_IFACE" >/dev/null
ip link show "$GLOO_IFACE" >/dev/null
"$PYTHON_BIN_REMOTE" -I -S - "$REQUIRED_GPUS" "$SELECTED_GPUS" "$NODE_LABEL" "$PROBE_PHASE" <<'PY'
from __future__ import annotations

import csv
import io
import subprocess
import sys


required = int(sys.argv[1])
selected_tokens = [token.strip() for token in sys.argv[2].split(",") if token.strip()]
node = sys.argv[3]
phase = sys.argv[4]
if phase not in {"early", "pre-launch"}:
    raise SystemExit(f"[ERROR][{node}] invalid selected-GPU probe phase: {phase!r}")
if required < 1:
    raise SystemExit(f"[ERROR][{node}] required GPU count must be positive: {required}")
if len(selected_tokens) != required:
    raise SystemExit(
        f"[ERROR][{node}] CUDA_VISIBLE_DEVICES must select exactly {required} GPU(s), "
        f"got {len(selected_tokens)}: {selected_tokens}"
    )


def query(*fields: str) -> str:
    return subprocess.run(
        [
            "nvidia-smi",
            f"--query-{fields[0]}={fields[1]}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout


gpu_rows = query("gpu", "index,uuid")
index_to_uuid: dict[str, str] = {}
for row in csv.reader(io.StringIO(gpu_rows)):
    if len(row) < 2:
        continue
    index = row[0].strip()
    uuid = row[1].strip()
    if not index or not uuid or index in index_to_uuid:
        raise SystemExit(f"[ERROR][{node}] nvidia-smi returned a malformed GPU inventory row: {row!r}")
    index_to_uuid[index] = uuid
if len(index_to_uuid) < required:
    raise SystemExit(
        f"[ERROR][{node}] nvidia-smi found {len(index_to_uuid)} physical GPU(s), "
        f"but the launch requires {required}"
    )

available_uuids = set(index_to_uuid.values())
selected_uuids: list[str] = []
for token in selected_tokens:
    if token in index_to_uuid:
        selected_uuids.append(index_to_uuid[token])
    elif token in available_uuids:
        selected_uuids.append(token)
    else:
        raise SystemExit(
            f"[ERROR][{node}] CUDA_VISIBLE_DEVICES selects an unknown GPU index/UUID: {token!r}"
        )
if len(set(selected_uuids)) != len(selected_uuids):
    raise SystemExit(
        f"[ERROR][{node}] CUDA_VISIBLE_DEVICES selects duplicate GPUs: {selected_tokens}"
    )

selected_uuid_set = set(selected_uuids)
busy: list[str] = []
for row in csv.reader(io.StringIO(query("compute-apps", "gpu_uuid,pid,process_name"))):
    if len(row) >= 2 and row[0].strip() in selected_uuid_set:
        busy.append(",".join(part.strip() for part in row))
if busy:
    raise SystemExit(
        f"[ERROR][{node}] selected GPU(s) are not idle: " + "; ".join(busy)
    )

print(
    f"[INFO][{node}] selected_gpu_idle_preflight_ok "
    f"phase={phase} required={required} selected={selected_tokens} uuids={selected_uuids}"
)
PY
REMOTE
)
  cmd="RUNTIME_PATH=$(quote "${runtime_path}")"$'\n'
  cmd+="PYTHON_BIN_REMOTE=$(quote "${PYTHON_BIN}")"$'\n'
  cmd+="REQUIRED_GPUS=$(quote "${NPROC}")"$'\n'
  cmd+="SELECTED_GPUS=$(quote "${CUDA_VISIBLE_DEVICES}")"$'\n'
  cmd+="NCCL_IFACE=$(quote "${NCCL_SOCKET_IFNAME}")"$'\n'
  cmd+="GLOO_IFACE=$(quote "${GLOO_SOCKET_IFNAME}")"$'\n'
  cmd+="NODE_LABEL=$(quote "${node}")"$'\n'
  cmd+="PROBE_PHASE=$(quote "${probe_phase}")"$'\n'
  cmd+="${body}"
  remote_run_bounded "${node}" "${cmd}" "${LAUNCH_CONTROL_TIMEOUT_SECONDS}"
}

preflight_selected_gpus_idle_parallel() {
  local -a pids=()
  local node pid failed=0
  for node in "${NODE_LIST[@]}"; do
    preflight_selected_gpus_idle_node "${node}" early &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed != 0 )); then
    echo "[ERROR] Refusing launch because one or more selected GPU sets are busy or unhealthy; no external-AS materialization, W&B verification, or lifecycle mutation was reached." >&2
    return 1
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[INFO] Dry-run would enforce the all-node selected-GPU idle gate before heavy preflight work."
  else
    echo "[INFO] All nodes passed the read-only selected-GPU idle gate before heavy preflight work."
  fi
}

preflight_external_as_asset_closure_node() {
  local node="$1"
  local runtime_pythonpath="${RUN_REPO}/src/holosoma:${RUN_REPO}/src/holosoma_inference:${RUN_REPO}/src"
  if [[ -n "${PYTHON_RUNTIME_SITEPACKAGES}" ]]; then
    runtime_pythonpath="${PYTHON_RUNTIME_SITEPACKAGES}:${runtime_pythonpath}"
  fi
  local runtime_path
  runtime_path="$(dirname -- "${PYTHON_BIN}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  local body cmd
  body=$(cat <<'REMOTE'
set -euo pipefail
unset BASH_ENV ENV CDPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONOPTIMIZE PYTHONWARNINGS PYTHONBREAKPOINT PYTHONSAFEPATH
unset LD_PRELOAD
export PATH="$RUNTIME_PATH"
export PYTHONPATH="$RUNTIME_PYTHONPATH"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LC_ALL=C
cd "$RUN_REPO_REMOTE"
grep -Fx -- "$SNAPSHOT_ID" .holosoma_snapshot/id >/dev/null
sha256sum --quiet -c .holosoma_snapshot/source_manifest.sha256

SOURCE_BANK="$RUN_REPO_REMOTE/data/ds_as_data/$CORL_BANK_NAME"
SOURCE_MAP="$SOURCE_BANK/_clip_object_urdf_map.json"
if [[ ! -d "$SOURCE_BANK" ]]; then
  echo "[ERROR][$NODE_LABEL] External AS source bank is missing: $SOURCE_BANK" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_MAP" ]]; then
  echo "[ERROR][$NODE_LABEL] External AS source object map is missing: $SOURCE_MAP" >&2
  exit 2
fi

NORMALIZED_ALLOWED=$("$PYTHON_BIN_REMOTE" - "$RAW_ALLOWED_CATEGORIES" <<'PY'
from __future__ import annotations
import json
import sys

aliases = {
    "boxes": "box", "cube": "box", "cubes": "box", "largebox": "box", "largeboxes": "box",
    "trash": "bin", "trashcan": "bin", "trashcans": "bin", "basket": "bin", "baskets": "bin", "bins": "bin",
    "barrels": "barrel", "sphere": "ball", "spheres": "ball", "balls": "ball",
}
allowed_universe = {"box", "bin", "barrel", "ball"}
raw_text = sys.argv[1].strip() or '["box","bin","barrel","ball"]'
try:
    raw = json.loads(raw_text)
except Exception as exc:
    raise SystemExit(f"[ERROR] SOLID_ALLOWED_OBJECT_CATEGORIES must be a JSON list: {exc}") from exc
if not isinstance(raw, list):
    raise SystemExit("[ERROR] SOLID_ALLOWED_OBJECT_CATEGORIES must be a JSON list")
normalized = []
for value in raw:
    token = str(value).strip().lower().replace("-", "_")
    category = aliases.get(token, token)
    if category not in allowed_universe:
        raise SystemExit(f"[ERROR] Unsupported solid object category: {category!r}")
    if category not in normalized:
        normalized.append(category)
if not normalized:
    raise SystemExit("[ERROR] SOLID_ALLOWED_OBJECT_CATEGORIES cannot be empty")
print(json.dumps(normalized, separators=(",", ":")))
PY
)

EFFECTIVE_CLIP_LIST="$SOLID_CLIP_LIST_RAW"
if [[ -z "$EFFECTIVE_CLIP_LIST" && -f "$SOURCE_BANK/clean80_strict_success_solid_no_falldown_clips.txt" ]]; then
  EFFECTIVE_CLIP_LIST="$SOURCE_BANK/clean80_strict_success_solid_no_falldown_clips.txt"
fi
EFFECTIVE_TARGET_NAME="$SOLID_TARGET_BANK_NAME_RAW"
if [[ -z "$EFFECTIVE_TARGET_NAME" && -n "$EFFECTIVE_CLIP_LIST" ]]; then
  EFFECTIVE_TARGET_NAME="${CORL_BANK_NAME}_solid80_clean_box_bin_barrel_ball"
fi

SOLID_PREP_ARGS=(
  --source-bank "$SOURCE_BANK"
  --source-map "$SOURCE_MAP"
  --allowed-categories-json "$NORMALIZED_ALLOWED"
  --contact-export-name "$SOLID_CONTACT_EXPORT_NAME_RAW"
)
if [[ -n "$EFFECTIVE_CLIP_LIST" ]]; then
  SOLID_PREP_ARGS+=(--clip-list "$EFFECTIVE_CLIP_LIST")
fi
if [[ -n "$EFFECTIVE_TARGET_NAME" ]]; then
  SOLID_PREP_ARGS+=(--target-bank-name "$EFFECTIVE_TARGET_NAME")
fi
SOLID_PREP_OUTPUT=$("$PYTHON_BIN_REMOTE" scripts/prepare_immutable_solid_bank.py "${SOLID_PREP_ARGS[@]}")
SOLID_BANK_DIR=""
SOLID_OBJECT_MAP=""
SOLID_SELECTED_CLIP_COUNT=""
SOLID_SOURCE_DIGEST=""
while IFS='=' read -r key value; do
  case "$key" in
    SOLID_BANK_DIR|SOLID_OBJECT_MAP|SOLID_SELECTED_CLIP_COUNT|SOLID_SOURCE_DIGEST)
      printf -v "$key" '%s' "$value"
      ;;
  esac
done <<< "$SOLID_PREP_OUTPUT"
if [[ ! "$SOLID_SOURCE_DIGEST" =~ ^[0-9a-f]{64}$ \
      || ! "$SOLID_SELECTED_CLIP_COUNT" =~ ^[1-9][0-9]*$ \
      || ! -d "$SOLID_BANK_DIR" || ! -f "$SOLID_OBJECT_MAP" ]]; then
  echo "[ERROR][$NODE_LABEL] Immutable solid-AS preparation returned an incomplete identity." >&2
  exit 2
fi
if [[ -n "$EXPECTED_SELECTED_CLIP_COUNT" \
      && "$SOLID_SELECTED_CLIP_COUNT" != "$EXPECTED_SELECTED_CLIP_COUNT" ]]; then
  echo "[ERROR][$NODE_LABEL] Effective solid-AS selection count differs from the controller contract: actual=$SOLID_SELECTED_CLIP_COUNT expected=$EXPECTED_SELECTED_CLIP_COUNT" >&2
  exit 2
fi

SINGLE_BASE="$SOLID_BANK_DIR/_single_slot_motion_bank"
SINGLE_DIR=$("$PYTHON_BIN_REMOTE" scripts/prepare_immutable_single_slot_bank.py \
  --source-motion-dir "$SOLID_BANK_DIR" \
  --source-object-map "$SOLID_OBJECT_MAP" \
  --output-base "$SINGLE_BASE")
SINGLE_DIR=$(realpath -e -- "$SINGLE_DIR")
SINGLE_MAP="$SINGLE_DIR/_clip_object_urdf_map.json"
test -f "$SINGLE_MAP"

RANK_SOURCE_DIGEST=$("$PYTHON_BIN_REMOTE" scripts/prepare_as_rank_shards.py \
  --motion-dir "$SINGLE_DIR" \
  --object-map "$SINGLE_MAP" \
  --world-size "$GLOBAL_WORLD_SIZE" \
  --source-digest-only)
if [[ ! "$RANK_SOURCE_DIGEST" =~ ^[0-9a-f]{64}$ ]]; then
  echo "[ERROR][$NODE_LABEL] Effective AS rank-shard source digest is malformed." >&2
  exit 2
fi
RANK_ROOT="$SINGLE_DIR/_rank_shards/by-source/$RANK_SOURCE_DIGEST/ws$GLOBAL_WORLD_SIZE"
PUBLISHED_RANK_ROOT=$("$PYTHON_BIN_REMOTE" scripts/prepare_as_rank_shards.py \
  --motion-dir "$SINGLE_DIR" \
  --object-map "$SINGLE_MAP" \
  --output-root "$RANK_ROOT" \
  --world-size "$GLOBAL_WORLD_SIZE" \
  --expected-source-digest "$RANK_SOURCE_DIGEST")
if [[ "$(realpath -e -- "$PUBLISHED_RANK_ROOT")" != "$(realpath -e -- "$RANK_ROOT")" ]]; then
  echo "[ERROR][$NODE_LABEL] Rank-shard publisher returned a different effective root." >&2
  exit 2
fi

"$PYTHON_BIN_REMOTE" - \
  "$SINGLE_DIR" "$SINGLE_MAP" "$RANK_ROOT" "$GLOBAL_WORLD_SIZE" \
  "$SOLID_SOURCE_DIGEST" "$SOLID_SELECTED_CLIP_COUNT" "$RANK_SOURCE_DIGEST" \
  "$EXPECTED_MOTION_CLIP_ID" "$EXPECTED_MOTION_NPZ_SHA256" \
  "$EXPECTED_OBJECT_MAP_SHA256" "$EXPECTED_OBJECT_URDF_SHA256" \
  "$EXPECTED_OBJECT_MESH_SHA256" "$EXPECTED_SINGLE_SLOT_SOURCE_DIGEST" \
  "$EXPECTED_SINGLE_SLOT_VIEW_DIGEST" "$EXPECTED_RANK_SHARD_SOURCE_DIGEST" \
  "$MOTION_GENERATOR_TEACHER_EXPECTED_SHA256" \
  "$REQUIRE_MOTION_GENERATOR_TEACHER_MATCH" <<'PY'
from __future__ import annotations

import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str((Path.cwd() / "scripts").resolve()))
from prepare_as_rank_shards import (
    compute_rank_shard_source_digest,
    validate_published_rank_shards,
)
from motion_generator_teacher import (
    MOTION_GENERATOR_TEACHER_KEY,
    motion_generator_teacher_from_solid_manifest,
    validate_motion_generator_teacher,
)

(
    single_dir_raw,
    single_map_raw,
    rank_root_raw,
    world_size_raw,
    solid_source_digest,
    selected_clip_count_raw,
    rank_source_digest,
    expected_clip_id,
    expected_motion_sha,
    expected_map_sha,
    expected_urdf_sha,
    expected_mesh_sha,
    expected_single_source_digest,
    expected_single_view_digest,
    expected_rank_source_digest,
    legacy_expected_generator_sha256,
    require_generator_match_raw,
) = sys.argv[1:]


def require_regular_readable(path: Path, *, role: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise SystemExit(f"[ERROR] {role} is missing: {path}: {exc}") from exc
    if not resolved.is_file():
        raise SystemExit(f"[ERROR] {role} is not a regular file: {resolved}")
    try:
        with resolved.open("rb") as stream:
            stream.read(1)
    except OSError as exc:
        raise SystemExit(f"[ERROR] {role} is not readable: {resolved}: {exc}") from exc
    return resolved


def sha256_file(path: Path, *, role: str) -> tuple[str, Path]:
    resolved = require_regular_readable(path, role=role)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), resolved


def resolve_local(raw: object, *, base: Path, role: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise SystemExit(f"[ERROR] {role} has no local path")
    value = raw.strip()
    if value.lower().startswith(("http://", "https://", "package://", "data:")):
        raise SystemExit(f"[ERROR] {role} uses an unsupported non-local path: {value!r}")
    if value.lower().startswith("file://"):
        value = value[7:]
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else base / candidate


single_dir = Path(single_dir_raw).resolve(strict=True)
single_map = require_regular_readable(Path(single_map_raw), role="effective single-slot object map")
rank_root = Path(rank_root_raw).resolve(strict=True)
world_size = int(world_size_raw)
selected_clip_count = int(selected_clip_count_raw)
if world_size < 1 or selected_clip_count < 1:
    raise SystemExit("[ERROR] Effective AS closure has a non-positive world/clip count")
if re.fullmatch(r"[0-9a-f]{64}", solid_source_digest) is None:
    raise SystemExit("[ERROR] Effective solid source digest is malformed")
if re.fullmatch(r"[0-9a-f]{64}", rank_source_digest) is None:
    raise SystemExit("[ERROR] Effective rank source digest is malformed")

try:
    map_payload = json.loads(single_map.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"[ERROR] Could not parse effective single-slot object map: {exc}") from exc
clips = map_payload.get("clips") if isinstance(map_payload, dict) else None
if not isinstance(clips, dict) or len(clips) != selected_clip_count:
    raise SystemExit("[ERROR] Effective single-slot map does not exactly cover selected clips")
motion_ids = sorted(path.stem for path in single_dir.glob("*.npz"))
if motion_ids != sorted(str(clip_id) for clip_id in clips):
    raise SystemExit("[ERROR] Effective single-slot motion/map clip sets differ")
clip_id = expected_clip_id or motion_ids[0]
if clip_id not in clips:
    raise SystemExit(f"[ERROR] Replay motion clip is absent from effective training bank: {clip_id}")
entry = clips[clip_id]
if not isinstance(entry, dict):
    raise SystemExit(f"[ERROR] Effective object-map entry is not a mapping: {clip_id}")

motion_sha, _ = sha256_file(single_dir / f"{clip_id}.npz", role=f"motion clip {clip_id}")
map_sha, _ = sha256_file(single_map, role="effective single-slot object map")
urdf_path = resolve_local(entry.get("object_urdf_path"), base=single_map.parent, role=f"{clip_id} object_urdf_path")
urdf_sha, urdf_path = sha256_file(urdf_path, role=f"{clip_id} object URDF")
try:
    urdf_root = ET.parse(urdf_path).getroot()
except Exception as exc:
    raise SystemExit(f"[ERROR] Invalid effective object URDF {urdf_path}: {exc}") from exc

primary_mesh_raw = entry.get("object_mesh_path")
if not isinstance(primary_mesh_raw, str) or not primary_mesh_raw.strip():
    mesh_tags = urdf_root.findall(".//mesh")
    if not mesh_tags:
        raise SystemExit(f"[ERROR] Effective object URDF has no mesh: {urdf_path}")
    primary_mesh_raw = mesh_tags[0].get("filename")
    primary_mesh_base = urdf_path.parent
else:
    primary_mesh_base = single_map.parent
mesh_path = resolve_local(primary_mesh_raw, base=primary_mesh_base, role=f"{clip_id} primary object mesh")
mesh_sha, _ = sha256_file(mesh_path, role=f"{clip_id} primary object mesh")

manifest_path = require_regular_readable(single_dir / "manifest.json", role="single-slot manifest")
try:
    single_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"[ERROR] Could not parse single-slot manifest: {exc}") from exc
if require_generator_match_raw not in {"0", "1"}:
    raise SystemExit("[ERROR] Motion-generator teacher match mode is not canonical")
if legacy_expected_generator_sha256 and re.fullmatch(
    r"[0-9a-f]{64}", legacy_expected_generator_sha256
) is None:
    raise SystemExit("[ERROR] Legacy motion-generator teacher SHA256 is malformed")
try:
    single_generator_raw = single_manifest.get(MOTION_GENERATOR_TEACHER_KEY)
    single_generator = (
        None
        if single_generator_raw is None
        else validate_motion_generator_teacher(
            single_generator_raw,
            role=f"single-slot manifest {MOTION_GENERATOR_TEACHER_KEY}",
        )
    )
    solid_manifest_path = require_regular_readable(
        single_dir.parents[2] / "manifest.json",
        role="immutable solid manifest for motion-generator lineage",
    )
    solid_manifest = json.loads(solid_manifest_path.read_text(encoding="utf-8"))
    solid_generator = motion_generator_teacher_from_solid_manifest(
        solid_manifest,
        role="immutable solid manifest",
    )
except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit(f"[ERROR] Invalid motion-generator teacher lineage: {exc}") from exc
if single_generator != solid_generator:
    raise SystemExit(
        "[ERROR] Single-slot motion-generator teacher differs from its immutable solid source"
    )
if single_generator is None:
    if not legacy_expected_generator_sha256:
        raise SystemExit(
            "[ERROR] Motion bank has no authenticated generator identity; legacy banks require "
            "MOTION_GENERATOR_TEACHER_EXPECTED_SHA256=<exact recovered checkpoint SHA256>"
        )
    motion_generator_teacher_sha256 = legacy_expected_generator_sha256
else:
    motion_generator_teacher_sha256 = str(single_generator["checkpoint_sha256"])
    if (
        legacy_expected_generator_sha256
        and legacy_expected_generator_sha256 != motion_generator_teacher_sha256
    ):
        raise SystemExit(
            "[ERROR] Explicit motion-generator teacher SHA256 conflicts with authenticated bank lineage: "
            f"manifest={motion_generator_teacher_sha256} explicit={legacy_expected_generator_sha256}"
        )
single_source_digest = single_manifest.get("source_digest")
single_view_digest = single_manifest.get("view_digest")
for role, value in (
    ("single-slot source digest", single_source_digest),
    ("single-slot view digest", single_view_digest),
):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SystemExit(f"[ERROR] {role} is malformed")

# Prove that every published rank map, clip assignment, loss weight, NPZ
# namespace/link target/content, and manifest byte was deterministically
# derived from the sealed single-slot source.  Comparing only the published
# map to its own manifest SHA would let a coherent map+manifest mutation
# self-authenticate.
validate_published_rank_shards(
    motion_dir=single_dir,
    object_map=single_map,
    output_root=rank_root,
    world_size=world_size,
    expected_source_digest=rank_source_digest,
)

expected_pairs = (
    ("motion_npz_sha256", expected_motion_sha, motion_sha),
    ("object_map_sha256", expected_map_sha, map_sha),
    ("object_urdf_sha256", expected_urdf_sha, urdf_sha),
    ("object_mesh_sha256", expected_mesh_sha, mesh_sha),
    ("single_slot_source_digest", expected_single_source_digest, single_source_digest),
    ("single_slot_view_digest", expected_single_view_digest, single_view_digest),
    ("rank_shard_source_digest", expected_rank_source_digest, rank_source_digest),
)
for role, expected, actual in expected_pairs:
    if expected and expected != actual:
        raise SystemExit(
            f"[ERROR] Effective external AS {role} differs from Rule-90 v2 replay inputs: "
            f"actual={actual} expected={expected}"
        )

print(
    "AS_EXTERNAL_ASSET_CLOSURE\t"
    + "\t".join(
        (
            clip_id,
            solid_source_digest,
            str(single_source_digest),
            str(single_view_digest),
            rank_source_digest,
            motion_sha,
            map_sha,
            urdf_sha,
            mesh_sha,
            str(single_dir),
            motion_generator_teacher_sha256,
        )
    )
)
PY
REMOTE
)
  cmd="RUNTIME_PATH=$(quote "${runtime_path}")"$'\n'
  cmd+="RUNTIME_PYTHONPATH=$(quote "${runtime_pythonpath}")"$'\n'
  cmd+="RUN_REPO_REMOTE=$(quote "${RUN_REPO}")"$'\n'
  cmd+="SNAPSHOT_ID=$(quote "${SOURCE_SNAPSHOT_ID}")"$'\n'
  cmd+="PYTHON_BIN_REMOTE=$(quote "${PYTHON_BIN}")"$'\n'
  cmd+="NODE_LABEL=$(quote "${node}")"$'\n'
  cmd+="CORL_BANK_NAME=$(quote "${CORL_SOLID80_BANK_NAME}")"$'\n'
  cmd+="RAW_ALLOWED_CATEGORIES=$(quote "${SOLID_ALLOWED_OBJECT_CATEGORIES}")"$'\n'
  cmd+="SOLID_CLIP_LIST_RAW=$(quote "${SOLID_CLIP_LIST}")"$'\n'
  cmd+="SOLID_TARGET_BANK_NAME_RAW=$(quote "${SOLID_TARGET_BANK_NAME}")"$'\n'
  cmd+="SOLID_CONTACT_EXPORT_NAME_RAW=$(quote "${SOLID_CONTACT_EXPORT_NAME}")"$'\n'
  cmd+="EXPECTED_SELECTED_CLIP_COUNT=$(quote "${OMOMO_EXPECTED_TOTAL}")"$'\n'
  cmd+="GLOBAL_WORLD_SIZE=$(quote "${TOTAL_GPUS}")"$'\n'
  cmd+="EXPECTED_MOTION_CLIP_ID=$(quote "${REPLAY_AS_MOTION_CLIP_ID}")"$'\n'
  cmd+="EXPECTED_MOTION_NPZ_SHA256=$(quote "${REPLAY_AS_MOTION_NPZ_SHA256}")"$'\n'
  cmd+="EXPECTED_OBJECT_MAP_SHA256=$(quote "${REPLAY_AS_OBJECT_MAP_SHA256}")"$'\n'
  cmd+="EXPECTED_OBJECT_URDF_SHA256=$(quote "${REPLAY_AS_OBJECT_URDF_SHA256}")"$'\n'
  cmd+="EXPECTED_OBJECT_MESH_SHA256=$(quote "${REPLAY_AS_OBJECT_MESH_SHA256}")"$'\n'
  cmd+="EXPECTED_SINGLE_SLOT_SOURCE_DIGEST=$(quote "${REPLAY_AS_SINGLE_SLOT_SOURCE_DIGEST}")"$'\n'
  cmd+="EXPECTED_SINGLE_SLOT_VIEW_DIGEST=$(quote "${REPLAY_AS_SINGLE_SLOT_VIEW_DIGEST}")"$'\n'
  cmd+="EXPECTED_RANK_SHARD_SOURCE_DIGEST=$(quote "${REPLAY_AS_RANK_SHARD_SOURCE_DIGEST}")"$'\n'
  cmd+="MOTION_GENERATOR_TEACHER_EXPECTED_SHA256=$(quote "${MOTION_GENERATOR_TEACHER_EXPECTED_SHA256}")"$'\n'
  cmd+="REQUIRE_MOTION_GENERATOR_TEACHER_MATCH=$(quote "${REQUIRE_MOTION_GENERATOR_TEACHER_MATCH}")"$'\n'
  cmd+="${body}"

  local output marker marker_count
  if ! output=$(remote_run_mutation_bounded \
      "${node}" "${cmd}" "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}"); then
    echo "[ERROR][${node}] External AS asset closure failed before W&B verification and launch intent." >&2
    return 2
  fi
  if [[ "${DRY_RUN}" == 1 ]]; then
    printf '%s\n' "${output}"
    return 0
  fi
  marker_count=$(printf '%s\n' "${output}" | grep -c '^AS_EXTERNAL_ASSET_CLOSURE' || true)
  if [[ "${marker_count}" != 1 ]]; then
    echo "[ERROR][${node}] External AS preflight did not return one exact closure record." >&2
    return 2
  fi
  marker=$(printf '%s\n' "${output}" | grep '^AS_EXTERNAL_ASSET_CLOSURE')
  if [[ "${marker}" != AS_EXTERNAL_ASSET_CLOSURE$'\t'* || "${marker}" == *$'\n'* ]]; then
    echo "[ERROR][${node}] External AS preflight returned a malformed closure record." >&2
    return 2
  fi
  echo "[INFO][${node}] external_as_asset_closure_verified ${marker#*$'\t'}" >&2
  printf '%s\n' "${marker#*$'\t'}"
}

preflight_external_as_asset_closures_parallel() {
  local result_root
  result_root=$(mktemp -d "${TMPDIR:-/tmp}/holosoma-external-as-closure.XXXXXXXX")
  chmod 700 -- "${result_root}"
  local -a pids=() result_paths=()
  local node index pid failed=0
  for index in "${!NODE_LIST[@]}"; do
    node=${NODE_LIST[${index}]}
    result_paths+=("${result_root}/${index}.record")
    preflight_external_as_asset_closure_node "${node}" \
      >"${result_paths[${index}]}" &
    pids+=("$!")
  done
  for index in "${!NODE_LIST[@]}"; do
    node=${NODE_LIST[${index}]}
    pid=${pids[${index}]}
    if ! wait "${pid}"; then
      echo "[ERROR][${node}] External AS pre-intent barrier failed." >&2
      failed=1
    fi
  done
  if (( failed != 0 )); then
    rm -rf -- "${result_root}"
    echo "[ERROR] Refusing launch because one or more nodes lack the exact external AS asset closure; W&B verification and lifecycle intent were not reached." >&2
    return 1
  fi
  if [[ "${DRY_RUN}" == 1 ]]; then
    for index in "${!result_paths[@]}"; do
      command cat -- "${result_paths[${index}]}"
    done
    rm -rf -- "${result_root}"
    echo "[INFO] Dry-run would enforce the all-node external AS asset closure barrier before W&B verification and launch intent publication."
    return 0
  fi

  local expected_record="" record
  for index in "${!NODE_LIST[@]}"; do
    node=${NODE_LIST[${index}]}
    if [[ ! -s "${result_paths[${index}]}" ]] \
        || [[ "$(awk 'END { print NR }' "${result_paths[${index}]}")" != 1 ]]; then
      echo "[ERROR][${node}] External AS closure result is missing or not one record." >&2
      failed=1
      continue
    fi
    record=$(<"${result_paths[${index}]}")
    if [[ -z "${expected_record}" ]]; then
      expected_record=${record}
    elif [[ "${record}" != "${expected_record}" ]]; then
      echo "[ERROR][${node}] External AS closure identity differs across target nodes." >&2
      failed=1
    fi
  done
  rm -rf -- "${result_root}"
  if (( failed != 0 )); then
    echo "[ERROR] Refusing launch because target nodes do not share one exact external AS byte closure." >&2
    return 1
  fi
  local -a closure_fields=()
  IFS=$'\t' read -r -a closure_fields <<<"${expected_record}"
  if (( ${#closure_fields[@]} != 11 )); then
    echo "[ERROR] Identical external AS closure record does not contain the exact sealed field set." >&2
    return 2
  fi
  AS_EXTERNAL_MOTION_CLIP_ID=${closure_fields[0]}
  AS_EXTERNAL_SOLID_SOURCE_DIGEST=${closure_fields[1]}
  AS_EXTERNAL_SINGLE_SLOT_SOURCE_DIGEST=${closure_fields[2]}
  AS_EXTERNAL_SINGLE_SLOT_VIEW_DIGEST=${closure_fields[3]}
  AS_EXTERNAL_RANK_SHARD_SOURCE_DIGEST=${closure_fields[4]}
  AS_EXTERNAL_MOTION_NPZ_SHA256=${closure_fields[5]}
  AS_EXTERNAL_OBJECT_MAP_SHA256=${closure_fields[6]}
  AS_EXTERNAL_OBJECT_URDF_SHA256=${closure_fields[7]}
  AS_EXTERNAL_OBJECT_MESH_SHA256=${closure_fields[8]}
  AS_EXTERNAL_SINGLE_SLOT_DIR=${closure_fields[9]}
  AS_EXTERNAL_MOTION_GENERATOR_TEACHER_SHA256=${closure_fields[10]}
  local closure_digest
  for closure_digest in \
      "${AS_EXTERNAL_SOLID_SOURCE_DIGEST}" \
      "${AS_EXTERNAL_SINGLE_SLOT_SOURCE_DIGEST}" \
      "${AS_EXTERNAL_SINGLE_SLOT_VIEW_DIGEST}" \
      "${AS_EXTERNAL_RANK_SHARD_SOURCE_DIGEST}" \
      "${AS_EXTERNAL_MOTION_NPZ_SHA256}" \
      "${AS_EXTERNAL_OBJECT_MAP_SHA256}" \
      "${AS_EXTERNAL_OBJECT_URDF_SHA256}" \
      "${AS_EXTERNAL_OBJECT_MESH_SHA256}" \
      "${AS_EXTERNAL_MOTION_GENERATOR_TEACHER_SHA256}"; do
    if [[ ! "${closure_digest}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "[ERROR] External AS closure record contains a malformed digest." >&2
      return 2
    fi
  done
  if [[ ! "${AS_EXTERNAL_MOTION_CLIP_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$ \
        || "${AS_EXTERNAL_SINGLE_SLOT_DIR}" \
          != "${REMOTE_REPO_NORMALIZED}/data/"*"/_single_slot_motion_bank/by-source/${AS_EXTERNAL_SINGLE_SLOT_VIEW_DIGEST}" ]]; then
    echo "[ERROR] External AS closure record does not bind one canonical repo-local single-slot view." >&2
    return 2
  fi
  AS_EXTERNAL_CLOSURE_RECORD=${expected_record}
  echo "[INFO] All nodes passed one identical external AS asset closure before W&B verification: ${expected_record}"
}

verify_fresh_wandb_replay_preflight() {
  if [[ -z "${FRESH_WANDB_RUN_ID}" ]]; then
    return 0
  fi
  local verifier="${SCRIPT_DIR}/scripts/wandb_replay_preflight.py"
  if [[ ! -f "${verifier}" || -L "${verifier}" ]]; then
    echo "[ERROR] Fresh W&B replay verifier is missing or is a symlink: ${verifier}" >&2
    return 2
  fi
  local actual_manifest_sha256
  actual_manifest_sha256=$(sha256sum -- "${REPLAY_PREFLIGHT_MANIFEST}" | awk '{print $1}')
  if [[ "${actual_manifest_sha256}" != "${REPLAY_PREFLIGHT_MANIFEST_SHA256}" ]]; then
    echo "[ERROR] Replay preflight manifest SHA256 mismatch: actual=${actual_manifest_sha256} expected=${REPLAY_PREFLIGHT_MANIFEST_SHA256}" >&2
    return 2
  fi
  local -a replay_version_args=()
  if [[ -n "${REPLAY_PREFLIGHT_REQUIRED_VERSION}" ]]; then
    replay_version_args+=(
      --required-manifest-version "${REPLAY_PREFLIGHT_REQUIRED_VERSION}"
    )
  fi
  if [[ "${REPLAY_PREFLIGHT_REQUIRED_VERSION}" == 2 ]]; then
    if [[ ! "${SOURCE_SNAPSHOT_ARCHIVE_SHA256}" =~ ^[0-9a-f]{64}$ \
          || ! "${DISTILL_AS_ENTRYPOINT_SHA256}" =~ ^[0-9a-f]{64}$ \
          || -z "${DISTILL_AS_ENTRYPOINT}" ]]; then
      echo "[ERROR] Rule-90 v2 requires authenticated source-archive and selected-entrypoint identity before replay verification." >&2
      return 2
    fi
    replay_version_args+=(
      --expected-source-archive-sha256 "${SOURCE_SNAPSHOT_ARCHIVE_SHA256}"
      --expected-entrypoint-archive-member "${DISTILL_AS_ENTRYPOINT}"
      --expected-entrypoint-sha256 "${DISTILL_AS_ENTRYPOINT_SHA256}"
    )
  fi
  "${PYTHON_BIN}" "${verifier}" verify \
    --manifest "${REPLAY_PREFLIGHT_MANIFEST}" \
    --expected-manifest-sha256 "${REPLAY_PREFLIGHT_MANIFEST_SHA256}" \
    --expected-source-snapshot-id "${SOURCE_SNAPSHOT_ID}" \
    "${replay_version_args[@]}" \
    --expected-entity "${WANDB_ENTITY}" \
    --expected-project carry-any \
    --expected-run-id "${FRESH_WANDB_RUN_ID}" \
    --expected-run-name "${RUN_NAME}" \
    --expected-world-size "${TOTAL_GPUS}"
  echo "[INFO] fresh_wandb_replay_preflight_verified=${WANDB_ENTITY}/carry-any/${FRESH_WANDB_RUN_ID} manifest_sha256=${REPLAY_PREFLIGHT_MANIFEST_SHA256}"
}

ensure_local_python_runtime_archive() {
  if [[ -z "${PYTHON_RUNTIME_SITEPACKAGES}" ]]; then
    return 0
  fi
  if [[ -z "${PYTHON_RUNTIME_ARCHIVE}" || -z "${PYTHON_RUNTIME_ARCHIVE_SHA256}" ]]; then
    echo "[ERROR] Runtime overlay preparation requires its controller archive and SHA256." >&2
    return 2
  fi
  local runtime_id="python-runtime-v2-${PYTHON_RUNTIME_MANIFEST_SHA256}"
  local expected_basename="${runtime_id}.tar.gz"
  if [[ ! -f "${PYTHON_RUNTIME_ARCHIVE}" || -L "${PYTHON_RUNTIME_ARCHIVE}" ]]; then
    echo "[ERROR] Python runtime archive is not a regular non-symlink file: ${PYTHON_RUNTIME_ARCHIVE}" >&2
    return 2
  fi
  local archive_path
  archive_path=$(realpath -e -- "${PYTHON_RUNTIME_ARCHIVE}") || return
  if [[ "$(basename -- "${archive_path}")" != "${expected_basename}" ]]; then
    echo "[ERROR] Python runtime archive basename must be ${expected_basename}." >&2
    return 2
  fi
  local current_uid
  current_uid=$(id -u)
  if [[ "$(stat -c '%h' -- "${archive_path}")" != 1 || "$(stat -c '%u' -- "${archive_path}")" != "${current_uid}" || "$(stat -c '%a' -- "${archive_path}")" != 444 ]]; then
    echo "[ERROR] Python runtime archive must be current-UID-owned, single-link, and sealed 0444: ${archive_path}" >&2
    return 2
  fi
  local archive_size
  archive_size=$(stat -c '%s' -- "${archive_path}")
  if ! canonical_positive_uint_at_most "${archive_size}" 4294967296; then
    echo "[ERROR] Python runtime archive size is empty, malformed, or exceeds 4 GiB." >&2
    return 2
  fi

  if [[ ! -d /tmp || -L /tmp \
        || "$(realpath -e -- /tmp 2>/dev/null || true)" != /tmp \
        || "$(stat -c '%u:%a' -- /tmp 2>/dev/null || true)" != "0:1777" ]]; then
    echo "[ERROR] Controller /tmp must be one real root-owned mode-1777 directory." >&2
    return 2
  fi
  if [[ -n "${PYTHON_RUNTIME_CONTROLLER_TRANSFER_ROOT}" ]]; then
    echo "[ERROR] Controller Python runtime transfer snapshot was initialized more than once." >&2
    return 2
  fi
  local transfer_root transfer_archive transfer_fingerprint
  transfer_root=$(umask 077; mktemp -d "/tmp/holosoma-python-runtime-transfer.$(id -u).XXXXXXXX")
  chmod 700 -- "${transfer_root}"
  transfer_root=$(realpath -e -- "${transfer_root}")
  transfer_fingerprint=$(stat -c '%d:%i:%u:%a' -- "${transfer_root}")
  if [[ "${transfer_fingerprint}" != *":$(id -u):700" ]]; then
    echo "[ERROR] Controller Python runtime transfer root is not private." >&2
    return 2
  fi
  transfer_archive="${transfer_root}/${expected_basename}"
  PYTHON_RUNTIME_CONTROLLER_TRANSFER_ROOT="${transfer_root}"
  PYTHON_RUNTIME_CONTROLLER_TRANSFER_ARCHIVE="${transfer_archive}"
  PYTHON_RUNTIME_CONTROLLER_TRANSFER_FINGERPRINT="${transfer_fingerprint}"

  local archive_fd path_fingerprint fd_fingerprint path_recheck
  exec {archive_fd}<"${archive_path}"
  path_fingerprint=$(stat -c '%d:%i:%f:%h:%u:%a:%s:%Y:%Z' -- "${archive_path}")
  fd_fingerprint=$(stat -Lc '%d:%i:%f:%h:%u:%a:%s:%Y:%Z' -- "/proc/$$/fd/${archive_fd}")
  path_recheck=$(stat -c '%d:%i:%f:%h:%u:%a:%s:%Y:%Z' -- "${archive_path}")
  if [[ "${path_fingerprint}" != "${fd_fingerprint}" || "${path_fingerprint}" != "${path_recheck}" ]]; then
    exec {archive_fd}<&-
    echo "[ERROR] Python runtime archive identity changed while opening: ${archive_path}" >&2
    return 2
  fi
  local actual_archive_sha256
  if ! actual_archive_sha256=$(
      (umask 077; tee -- "${transfer_archive}" <&"${archive_fd}") \
        | sha256sum \
        | awk '{print $1}'
    ); then
    exec {archive_fd}<&-
    echo "[ERROR] Could not copy the bound Python runtime archive into private controller storage." >&2
    return 2
  fi
  fd_fingerprint=$(stat -Lc '%d:%i:%f:%h:%u:%a:%s:%Y:%Z' -- "/proc/$$/fd/${archive_fd}")
  path_recheck=$(stat -c '%d:%i:%f:%h:%u:%a:%s:%Y:%Z' -- "${archive_path}")
  exec {archive_fd}<&-
  if [[ "${path_fingerprint}" != "${fd_fingerprint}" || "${path_fingerprint}" != "${path_recheck}" || "${actual_archive_sha256}" != "${PYTHON_RUNTIME_ARCHIVE_SHA256}" ]]; then
    echo "[ERROR] Python runtime archive changed or failed its exact SHA256 contract." >&2
    return 2
  fi
  chmod 400 -- "${transfer_archive}"
  if [[ ! -f "${transfer_archive}" || -L "${transfer_archive}" \
        || "$(stat -c '%h:%u:%a:%s' -- "${transfer_archive}")" != "1:${current_uid}:400:${archive_size}" ]]; then
    echo "[ERROR] Private controller Python runtime snapshot has malformed metadata." >&2
    return 2
  fi
  if ! gzip -t -- "${transfer_archive}"; then
    echo "[ERROR] Python runtime archive is not a valid gzip stream." >&2
    return 2
  fi
  local embedded_manifest_sha256
  if ! embedded_manifest_sha256=$(
      tar -xOzf "${transfer_archive}" site-packages/.holosoma-runtime-manifest.sha256 \
        | sha256sum \
        | awk '{print $1}'
    ); then
    echo "[ERROR] Python runtime archive omits its exact-tree manifest." >&2
    return 2
  fi
  if [[ "${embedded_manifest_sha256}" != "${PYTHON_RUNTIME_MANIFEST_SHA256}" ]]; then
    echo "[ERROR] Python runtime archive does not bind the requested manifest identity." >&2
    return 2
  fi
  PYTHON_RUNTIME_ARCHIVE="${transfer_archive}"
  PYTHON_RUNTIME_ARCHIVE_SIZE="${archive_size}"
  echo "[INFO] controller_python_runtime_archive_snapshot_verified=${transfer_archive} size=${archive_size}"
}

remote_run_preflight_bounded() {
  local node="$1"
  local cmd="$2"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN] ssh %s %s\n' "${node}" "${cmd}"
    return 0
  fi
  # Historical prepare callers use this helper and retain their dry-run text,
  # but real execution is always controller-bounded.  Expensive source/data
  # preparation receives the dedicated preflight budget.
  remote_run_bounded "${node}" "${cmd}" "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}"
}

remote_run_bounded() {
  local node="$1"
  local cmd="$2"
  local timeout_seconds="$3"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN] bounded-ssh timeout=%ss node=%s %s\n' "${timeout_seconds}" "${node}" "${cmd}"
    return 0
  fi
  # Non-interactive SSH inherits the node account's login umask (0002 on the
  # current fleet).  Lifecycle metadata, locks, and streamed control artifacts
  # must never silently become group-writable because of that ambient setting.
  cmd="umask 077; ${cmd}"
  # A connected SSH process can otherwise wait forever even when ConnectTimeout
  # is configured.  The launch handshake has a controller-side wall-clock
  # contract, so every probe also needs an execution bound.
  # shellcheck disable=SC2086
  timeout --foreground --signal=TERM --kill-after=5s "${timeout_seconds}s" \
    ssh ${SSH_OPTS} "${node}" "${cmd}"
}

calculate_remote_mutation_bounds() {
  local timeout_seconds="$1"
  REMOTE_MUTATION_MARGIN_SECONDS=$((timeout_seconds / 4))
  (( REMOTE_MUTATION_MARGIN_SECONDS >= 1 )) || REMOTE_MUTATION_MARGIN_SECONDS=1
  (( REMOTE_MUTATION_MARGIN_SECONDS <= 10 )) || REMOTE_MUTATION_MARGIN_SECONDS=10
  REMOTE_MUTATION_TIMEOUT_SECONDS=$((timeout_seconds - REMOTE_MUTATION_MARGIN_SECONDS))
  (( REMOTE_MUTATION_TIMEOUT_SECONDS >= 1 )) || REMOTE_MUTATION_TIMEOUT_SECONDS=1
  REMOTE_MUTATION_KILL_AFTER_SECONDS=$((REMOTE_MUTATION_MARGIN_SECONDS / 2))
  (( REMOTE_MUTATION_KILL_AFTER_SECONDS >= 1 )) || REMOTE_MUTATION_KILL_AFTER_SECONDS=1
}

remote_run_mutation_bounded() {
  local node="$1"
  local cmd="$2"
  local timeout_seconds="$3"
  local remote_timeout remote_kill_after wrapped_cmd
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN] mutation-bounded-ssh timeout=%ss node=%s %s\n' \
      "${timeout_seconds}" "${node}" "${cmd}"
    return 0
  fi
  cmd="umask 077; ${cmd}"
  # A controller-side timeout closes only its local ssh process; the remote
  # shell can otherwise keep a lifecycle lock and mutate state after the
  # controller has returned and begun a retry.  Give the remote transaction an
  # earlier TERM/KILL deadline, retaining a bounded margin for ssh to report
  # its terminal status.  TERM reaches the transaction's EXIT guard, which
  # thaws only an exact still-running legacy state; KILL then prevents any
  # unbounded remote survivor.
  calculate_remote_mutation_bounds "${timeout_seconds}"
  remote_timeout=${REMOTE_MUTATION_TIMEOUT_SECONDS}
  remote_kill_after=${REMOTE_MUTATION_KILL_AFTER_SECONDS}
  wrapped_cmd="timeout --signal=TERM --kill-after=${remote_kill_after}s ${remote_timeout}s bash -c $(quote "${cmd}")"
  # shellcheck disable=SC2086
  timeout --foreground --signal=TERM --kill-after=5s "${timeout_seconds}s" \
    ssh ${SSH_OPTS} "${node}" "${wrapped_cmd}"
}

remote_copy_to_bounded() {
  local local_path="$1"
  local node="$2"
  local remote_path="$3"
  local timeout_seconds="$4"
  # Legacy scp passes its remote path through a login shell.  Restrict this
  # generic boundary even though current OpenSSH commonly uses SFTP, so a
  # caller-controlled root can never become remote shell syntax.
  if [[ ! "${remote_path}" =~ ^/[A-Za-z0-9_.:@+/-]+$ ]]; then
    echo "[ERROR] Refusing an unsafe scp remote path." >&2
    return 2
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    # Keep the pre-existing dry-run contract: downstream audits parse this
    # exact scp-shaped record without performing any network operation.
    printf '[DRY_RUN] scp %s %s:%s\n' "${local_path}" "${node}" "${remote_path}"
    return 0
  fi
  # shellcheck disable=SC2086
  timeout --foreground --signal=TERM --kill-after=5s "${timeout_seconds}s" \
    scp ${SSH_OPTS} "${local_path}" "${node}:${remote_path}"
}

remote_run_with_stdin_bounded() {
  local node="$1"
  local cmd="$2"
  local timeout_seconds="$3"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[ERROR] remote_run_with_stdin_bounded must not be invoked directly in dry-run mode." >&2
    return 2
  fi
  # Preserve stdin for streamed, content-addressed control scripts while
  # retaining the same hard controller wall-clock bound as ordinary SSH.
  # shellcheck disable=SC2086
  timeout --foreground --signal=TERM --kill-after=5s "${timeout_seconds}s" \
    ssh ${SSH_OPTS} "${node}" "${cmd}"
}

harden_lifecycle_namespace_node() {
  local node="$1"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
RUN_ROOT=$(quote "${REMOTE_RUN_ROOT}")
CURRENT_UID=\$(id -u)
if [[ ! "\${CURRENT_UID}" =~ ^[0-9]+$ \
      || ! -d "\${RUN_ROOT}" || -L "\${RUN_ROOT}" \
      || "\$(realpath -e -- "\${RUN_ROOT}" 2>/dev/null || true)" != "\${RUN_ROOT}" \
      || "\$(stat -c %u -- "\${RUN_ROOT}")" != "\${CURRENT_UID}" ]]; then
  echo "[ERROR][${node}] REMOTE_RUN_ROOT is not one real current-UID-owned directory: \${RUN_ROOT}" >&2
  exit 2
fi
chmod 0755 "\${RUN_ROOT}"
if [[ "\$(stat -c %a -- "\${RUN_ROOT}")" != 755 \
      || "\$(stat -c %u -- "\${RUN_ROOT}")" != "\${CURRENT_UID}" ]]; then
  echo "[ERROR][${node}] REMOTE_RUN_ROOT did not reach exact owner-controlled mode 0755." >&2
  exit 2
fi
for relative in .active .rendezvous .status .active/.locks; do
  state_root="\${RUN_ROOT}/\${relative}"
  if [[ -e "\${state_root}" || -L "\${state_root}" ]]; then
    if [[ ! -d "\${state_root}" || -L "\${state_root}" ]]; then
      echo "[ERROR][${node}] Lifecycle namespace is non-directory or symlinked: \${state_root}" >&2
      exit 2
    fi
  else
    mkdir -- "\${state_root}"
  fi
  if [[ "\$(realpath -e -- "\${state_root}" 2>/dev/null || true)" != "\${state_root}" \
        || "\$(stat -c %u -- "\${state_root}")" != "\${CURRENT_UID}" ]]; then
    echo "[ERROR][${node}] Lifecycle namespace is not real/current-UID-owned: \${state_root}" >&2
    exit 2
  fi
  chmod 0700 "\${state_root}"
  if [[ "\$(stat -c %a -- "\${state_root}")" != 700 \
        || "\$(stat -c %u -- "\${state_root}")" != "\${CURRENT_UID}" ]]; then
    echo "[ERROR][${node}] Lifecycle namespace did not reach exact mode 0700: \${state_root}" >&2
    exit 2
  fi
done
reservation_lock="\${RUN_ROOT}/.rendezvous/.reservation.lock"
if [[ -e "\${reservation_lock}" || -L "\${reservation_lock}" ]]; then
  if [[ ! -f "\${reservation_lock}" || -L "\${reservation_lock}" \
        || "\$(stat -c %u -- "\${reservation_lock}")" != "\${CURRENT_UID}" \
        || "\$(stat -c %h -- "\${reservation_lock}")" != 1 ]]; then
    echo "[ERROR][${node}] Existing rendezvous lock has unsafe type/owner/link-count." >&2
    exit 2
  fi
  chmod 0600 "\${reservation_lock}"
  [[ "\$(stat -c %a -- "\${reservation_lock}")" == 600 ]] || exit 2
fi
echo "[INFO][${node}] lifecycle_namespace_hardened root=0755 private_dirs=0700"
EOF
)
  remote_run_bounded "${node}" "${cmd}" "${LAUNCH_CONTROL_TIMEOUT_SECONDS}"
}

harden_lifecycle_namespaces_parallel() {
  local -a pids=()
  local node pid index failed=0
  for node in "${NODE_LIST[@]}"; do
    harden_lifecycle_namespace_node "${node}" &
    pids+=("$!")
  done
  for index in "${!NODE_LIST[@]}"; do
    node=${NODE_LIST[${index}]}
    pid=${pids[${index}]}
    if ! wait "${pid}"; then
      echo "[ERROR][${node}] Lifecycle namespace hardening failed." >&2
      failed=1
    fi
  done
  return "${failed}"
}

ensure_local_corl_package_metadata() {
  if [[ "${PREPARE_COPY_SCRIPT}" != "cp_corl" || -z "${NFS_CORL_BANK}" ]]; then
    return 0
  fi
  if [[ ! -f "${NFS_CORL_BANK}" ]]; then
    return 0
  fi
  case "${NFS_CORL_BANK}" in
    *.tar)
      ;;
    *)
      echo "[ERROR] A file-valued NFS_CORL_BANK must be a .tar archive: ${NFS_CORL_BANK}" >&2
      exit 2
      ;;
  esac

  CONTROL_CORL_PACKAGE_PATH=$(cd "$(dirname "${NFS_CORL_BANK}")" && pwd -P)/$(basename "${NFS_CORL_BANK}")
  CONTROL_CORL_PACKAGE_SHA256=$(sha256sum "${CONTROL_CORL_PACKAGE_PATH}" | awk '{print $1}')
  CONTROL_CORL_PACKAGE_SIZE=$(stat -c %s "${CONTROL_CORL_PACKAGE_PATH}")
  if [[ ! "${CONTROL_CORL_PACKAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[ERROR] Failed to compute a valid SHA256 for control-side CoRL package: ${CONTROL_CORL_PACKAGE_PATH}" >&2
    exit 2
  fi
  if [[ ! "${CONTROL_CORL_PACKAGE_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] Control-side CoRL package is empty or has an invalid size: ${CONTROL_CORL_PACKAGE_PATH}" >&2
    exit 2
  fi
  echo "[INFO] control_corl_package=${CONTROL_CORL_PACKAGE_PATH} sha256=${CONTROL_CORL_PACKAGE_SHA256} size=${CONTROL_CORL_PACKAGE_SIZE}"
}

remote_corl_source_exists() {
  local node="$1"
  local source_path="$2"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
SOURCE_PATH=$(quote "${source_path}")
[[ -d "\${SOURCE_PATH}" || -f "\${SOURCE_PATH}" ]]
EOF
)
  # This probe is deliberately read-only. Any SSH/NFS error is treated as an
  # unavailable node-side source and must be satisfied by the verified cache.
  remote_run_bounded "${node}" "${cmd}" "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}" >/dev/null 2>&1
}

stage_corl_package_node() {
  local node="$1"
  if [[ -z "${CONTROL_CORL_PACKAGE_PATH}" || -z "${CONTROL_CORL_PACKAGE_SHA256}" ]]; then
    echo "[ERROR][${node}] Cannot stage CoRL data fallback: no control-side regular .tar file is available at ${NFS_CORL_BANK}." >&2
    return 2
  fi

  local remote_package="${REMOTE_DATA_PACKAGE_CACHE}/${CONTROL_CORL_PACKAGE_SHA256}.tar"
  local incoming_dir="${REMOTE_DATA_PACKAGE_CACHE}/.incoming"
  local remote_incoming="${incoming_dir}/${CONTROL_CORL_PACKAGE_SHA256}.${BASHPID}.tar"
  local status_cmd
  status_cmd=$(cat <<EOF
set -euo pipefail
FINAL=$(quote "${remote_package}")
EXPECTED=$(quote "${CONTROL_CORL_PACKAGE_SHA256}")
if [[ ! -e "\${FINAL}" ]]; then
  echo MISSING
  exit 0
fi
if [[ ! -f "\${FINAL}" ]]; then
  echo "[ERROR] Refusing non-file data-package cache entry: \${FINAL}" >&2
  exit 2
fi
actual=\$(sha256sum "\${FINAL}" | awk '{print \$1}')
if [[ "\${actual}" != "\${EXPECTED}" ]]; then
  echo "[ERROR] Existing data-package cache entry has wrong SHA256: \${FINAL} actual=\${actual} expected=\${EXPECTED}" >&2
  exit 2
fi
echo VALID
EOF
)

  local reserve_cmd
  reserve_cmd=$(cat <<EOF
set -euo pipefail
umask 077
mkdir -p $(quote "${REMOTE_DATA_PACKAGE_CACHE}") $(quote "${incoming_dir}")
if [[ -e $(quote "${remote_incoming}") ]]; then
  echo "[ERROR] Refusing to overwrite existing incoming data package: $(quote "${remote_incoming}")" >&2
  exit 2
fi
EOF
)

  local publish_cmd
  publish_cmd=$(cat <<EOF
set -euo pipefail
CACHE_ROOT=$(quote "${REMOTE_DATA_PACKAGE_CACHE}")
FINAL=$(quote "${remote_package}")
INCOMING=$(quote "${remote_incoming}")
EXPECTED=$(quote "${CONTROL_CORL_PACKAGE_SHA256}")
EXPECTED_SIZE=$(quote "${CONTROL_CORL_PACKAGE_SIZE}")
cleanup() { rm -f "\${INCOMING}"; }
trap cleanup EXIT
test -f "\${INCOMING}"
actual_size=\$(stat -c %s "\${INCOMING}")
if [[ "\${actual_size}" != "\${EXPECTED_SIZE}" ]]; then
  echo "[ERROR] Staged data package size mismatch: \${INCOMING} actual=\${actual_size} expected=\${EXPECTED_SIZE}" >&2
  exit 2
fi
actual=\$(sha256sum "\${INCOMING}" | awk '{print \$1}')
if [[ "\${actual}" != "\${EXPECTED}" ]]; then
  echo "[ERROR] Staged data package SHA256 mismatch: \${INCOMING} actual=\${actual} expected=\${EXPECTED}" >&2
  exit 2
fi
chmod a-w "\${INCOMING}"
exec 9>"\${CACHE_ROOT}/.publish-\${EXPECTED}.lock"
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 9; then
  echo "[ERROR] Timed out acquiring the data-package publish lock." >&2
  exit 1
fi
if [[ -e "\${FINAL}" ]]; then
  if [[ ! -f "\${FINAL}" ]]; then
    echo "[ERROR] Refusing non-file data-package cache entry: \${FINAL}" >&2
    exit 2
  fi
  final_sha=\$(sha256sum "\${FINAL}" | awk '{print \$1}')
  if [[ "\${final_sha}" != "\${EXPECTED}" ]]; then
    echo "[ERROR] Refusing to overwrite corrupt data-package cache entry: \${FINAL}" >&2
    exit 2
  fi
  rm -f "\${INCOMING}"
  echo "[INFO] reused_verified_data_package=\${FINAL}"
else
  mv -T --no-clobber "\${INCOMING}" "\${FINAL}"
  if [[ -e "\${INCOMING}" ]]; then
    if [[ ! -f "\${FINAL}" ]]; then
      echo "[ERROR] Concurrent data-package publisher created a non-file entry: \${FINAL}" >&2
      exit 2
    fi
    final_sha=\$(sha256sum "\${FINAL}" | awk '{print \$1}')
    if [[ "\${final_sha}" != "\${EXPECTED}" ]]; then
      echo "[ERROR] Concurrent data-package publisher produced the wrong SHA256: \${FINAL}" >&2
      exit 2
    fi
    rm -f "\${INCOMING}"
    echo "[INFO] reused_verified_data_package=\${FINAL}"
  else
    echo "[INFO] installed_verified_data_package=\${FINAL}"
  fi
fi
final_sha=\$(sha256sum "\${FINAL}" | awk '{print \$1}')
test "\${final_sha}" = "\${EXPECTED}"
trap - EXIT
EOF
)

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] if node ${node} cannot read ${NFS_CORL_BANK}, use verified control-side package fallback"
    remote_run_preflight_bounded "${node}" "${status_cmd}"
    remote_run_preflight_bounded "${node}" "${reserve_cmd}"
    printf '[DRY_RUN] scp %s %s:%s\n' "${CONTROL_CORL_PACKAGE_PATH}" "${node}" "${remote_incoming}"
    remote_run_preflight_bounded "${node}" "${publish_cmd}"
    RESOLVED_NODE_CORL_BANK="${remote_package}"
    return 0
  fi

  local cache_status
  if ! cache_status=$(remote_run_bounded "${node}" "${status_cmd}" "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}"); then
    echo "[ERROR][${node}] Failed to validate node-local data-package cache entry ${remote_package}." >&2
    return 2
  fi
  if [[ "${cache_status}" == "VALID" ]]; then
    echo "[INFO][${node}] reused_verified_data_package=${remote_package}"
    RESOLVED_NODE_CORL_BANK="${remote_package}"
    return 0
  fi
  if [[ "${cache_status}" != "MISSING" ]]; then
    echo "[ERROR][${node}] Unexpected data-package cache probe result: ${cache_status}" >&2
    return 2
  fi

  remote_run_preflight_bounded "${node}" "${reserve_cmd}"
  if ! remote_copy_to_bounded \
      "${CONTROL_CORL_PACKAGE_PATH}" "${node}" "${remote_incoming}" \
      "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}"; then
    remote_run_bounded "${node}" "rm -f $(quote "${remote_incoming}")" \
      "${LAUNCH_CLEANUP_TIMEOUT_SECONDS}" || true
    echo "[ERROR][${node}] Failed to stage control-side CoRL package to ${remote_incoming}." >&2
    return 2
  fi
  remote_run_preflight_bounded "${node}" "${publish_cmd}"
  RESOLVED_NODE_CORL_BANK="${remote_package}"
}

resolve_node_corl_source() {
  local node="$1"
  RESOLVED_NODE_CORL_BANK="${NFS_CORL_BANK}"
  if [[ "${PREPARE_COPY_SCRIPT}" != "cp_corl" || -z "${NFS_CORL_BANK}" ]]; then
    return 0
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ -n "${CONTROL_CORL_PACKAGE_PATH}" ]]; then
      stage_corl_package_node "${node}"
    else
      echo "[DRY_RUN] node ${node} must provide custom CoRL source ${NFS_CORL_BANK}; no control-side tar fallback is available"
    fi
    return 0
  fi

  if remote_corl_source_exists "${node}" "${NFS_CORL_BANK}"; then
    echo "[INFO][${node}] node_corl_source=${NFS_CORL_BANK}"
    return 0
  fi
  if [[ -z "${CONTROL_CORL_PACKAGE_PATH}" ]]; then
    echo "[ERROR][${node}] CoRL source is unavailable on the node (${NFS_CORL_BANK}) and no control-side regular .tar fallback exists." >&2
    return 2
  fi
  stage_corl_package_node "${node}"
}

install_source_snapshot_node() {
  local node="$1"
  ensure_local_source_snapshot
  local incoming_dir="${REMOTE_RUN_ROOT}/.incoming"
  local remote_archive="${incoming_dir}/${SOURCE_SNAPSHOT_ID}.${SOURCE_SNAPSHOT_ARCHIVE_SHA256}.tar.gz"

  remote_run_preflight_bounded "${node}" "set -euo pipefail; mkdir -p $(quote "${incoming_dir}")"
  remote_copy_to_bounded \
    "${SOURCE_SNAPSHOT_ARCHIVE}" "${node}" "${remote_archive}" \
    "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}"

  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
ASSET_REPO=$(quote "${REMOTE_REPO}")
RUN_ROOT=$(quote "${REMOTE_RUN_ROOT}")
SNAPSHOT_ID=$(quote "${SOURCE_SNAPSHOT_ID}")
EXPECTED_ARCHIVE_SHA256=$(quote "${SOURCE_SNAPSHOT_ARCHIVE_SHA256}")
EXPECTED_MANIFEST_SHA256=$(quote "${SOURCE_MANIFEST_SHA256}")
REMOTE_ARCHIVE=$(quote "${remote_archive}")
RUN_REPO="\${RUN_ROOT}/\${SNAPSHOT_ID}"
mkdir -p "\${RUN_ROOT}"
exec 9>"\${RUN_ROOT}/.snapshot-install.lock"
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 9; then
  echo "[ERROR] Timed out acquiring the source-snapshot install lock." >&2
  exit 1
fi

actual_archive_sha256=\$(sha256sum "\${REMOTE_ARCHIVE}" | awk '{print \$1}')
if [[ "\${actual_archive_sha256}" != "\${EXPECTED_ARCHIVE_SHA256}" ]]; then
  echo "[ERROR] Source archive SHA256 mismatch on $(quote "${node}"): \${actual_archive_sha256}" >&2
  exit 2
fi

source_mode_closure_matches() {
  local snapshot_root="\$1"
  (cd "\${snapshot_root}" && {
    find . -maxdepth 0 -type d -printf 'd\t%m\t%p\0'
    find . -mindepth 1 -maxdepth 1 -type f -printf 'f\t%m\t%p\0'
    for source_dir in src scripts tests submodules; do
      [[ -d "./\${source_dir}" ]] || continue
      find "./\${source_dir}" -type f -printf 'f\t%m\t%p\0'
      find "./\${source_dir}" -type d -printf 'd\t%m\t%p\0'
    done
    find ./.holosoma_snapshot -type d -printf 'd\t%m\t%p\0'
  } | sort -z | cmp -s - .holosoma_snapshot/source_modes.nul)
}

signed_source_directories_are_sealed() {
  local snapshot_root="\$1"
  ! (cd "\${snapshot_root}" && {
    find . -maxdepth 0 -type d -perm /222 -print
    for source_dir in src scripts tests submodules .holosoma_snapshot; do
      [[ -d "./\${source_dir}" ]] || continue
      find "./\${source_dir}" -type d -perm /222 -print
    done
  } | grep -q .)
}

verify_snapshot() {
  local snapshot_root="\$1"
  test -d "\${snapshot_root}"
  test -f "\${snapshot_root}/.holosoma_snapshot/source_manifest.sha256"
  test ! -L "\${snapshot_root}/.holosoma_snapshot/source_manifest.sha256"
  test -f "\${snapshot_root}/.holosoma_snapshot/id"
  test ! -L "\${snapshot_root}/.holosoma_snapshot/id"
  test "\$(<"\${snapshot_root}/.holosoma_snapshot/id")" = "\${SNAPSHOT_ID}"
  local actual_manifest_sha256
  actual_manifest_sha256=\$(sha256sum "\${snapshot_root}/.holosoma_snapshot/source_manifest.sha256" | awk '{print \$1}')
  if [[ "\${actual_manifest_sha256}" != "\${EXPECTED_MANIFEST_SHA256}" ]]; then
    echo "[ERROR] Source manifest digest mismatch in \${snapshot_root}." >&2
    return 2
  fi
  (cd "\${snapshot_root}" && sha256sum --quiet -c .holosoma_snapshot/source_manifest.sha256)
  # sha256sum -c validates listed files but does not reject an injected extra
  # module.  Recreate the signed regular-file manifest and require byte-for-
  # byte equality so the snapshot ID closes the entire executable source set.
  if ! (cd "\${snapshot_root}" && {
      find . -mindepth 1 -maxdepth 1 -type f -print0
      for source_dir in src scripts tests submodules .holosoma_snapshot; do
        [[ -d "./\${source_dir}" ]] || continue
        find "./\${source_dir}" -type f \
          ! -path './.holosoma_snapshot/source_manifest.sha256' \
          ! -path './.holosoma_snapshot/id' -print0
      done
    } | sort -z | xargs -0 -r sha256sum \
      | cmp -s - .holosoma_snapshot/source_manifest.sha256); then
    echo "[ERROR] Snapshot regular-file set/content does not exactly match its signed manifest: \${snapshot_root}" >&2
    return 2
  fi
  # Content hashes do not cover permission bits.  Recreate the NUL-delimited
  # mode closure that was itself signed into source_manifest.sha256 so a
  # cached or installed snapshot cannot silently change executable semantics.
  if ! source_mode_closure_matches "\${snapshot_root}"; then
    echo "[ERROR] Snapshot source mode closure does not match its signed manifest: \${snapshot_root}" >&2
    return 2
  fi
  # Directory write protection prevents ordinary jobs from unlinking a
  # read-only file through its writable parent.  It is an accidental-mutation
  # guard, not a security boundary against the owning Unix account; therefore
  # every reuse and launch still performs this exact closure check.
  if ! signed_source_directories_are_sealed "\${snapshot_root}"; then
    echo "[ERROR] Snapshot has a writable signed source directory: \${snapshot_root}" >&2
    return 2
  fi
  if [[ "\$(stat -c '%a' -- "\${snapshot_root}/.holosoma_snapshot")" != 555 ]]; then
    echo "[ERROR] Snapshot metadata directory is not sealed 0555: \${snapshot_root}" >&2
    return 2
  fi
  local metadata_name metadata_path
  for metadata_name in \
      asset_links.tsv source_symlinks.tsv source_modes.nul \
      source_manifest.sha256 id; do
    metadata_path="\${snapshot_root}/.holosoma_snapshot/\${metadata_name}"
    if [[ ! -f "\${metadata_path}" || -L "\${metadata_path}" \
          || "\$(stat -c '%a' -- "\${metadata_path}")" != 444 ]]; then
      echo "[ERROR] Snapshot metadata file is not a sealed regular 0444 file: \${metadata_path}" >&2
      return 2
    fi
  done
  # A new top-level package can participate in Python import resolution even
  # when every regular file under the signed source directories still matches.
  # Only staged source roots and explicitly non-executable runtime roots may
  # exist alongside the signed root files.
  if find "\${snapshot_root}" -mindepth 1 -maxdepth 1 -type d \
      ! -name src ! -name scripts ! -name tests ! -name submodules \
      ! -name .holosoma_snapshot ! -name .checkpoint_cache \
      ! -name .teacher_checkpoints ! -name .run_control ! -name logs \
      -print | grep -q .; then
    echo "[ERROR] Snapshot contains an unexpected top-level directory: \${snapshot_root}" >&2
    return 2
  fi
  # Runtime state is deliberately writable but its directory names are
  # protected by the sealed snapshot root.  Require the exact real top-level
  # containers on both fresh install and cache reuse; their contents are
  # mutable and intentionally outside the signed source closure.
  local runtime_dir runtime_path
  for runtime_dir in \
      .checkpoint_cache .teacher_checkpoints .run_control logs logs/batch_ne; do
    runtime_path="\${snapshot_root}/\${runtime_dir}"
    if [[ ! -d "\${runtime_path}" || -L "\${runtime_path}" \
          || "\$(stat -c '%a' -- "\${runtime_path}")" != 700 ]]; then
      echo "[ERROR] Snapshot runtime directory is not a real writable 0700 directory: \${runtime_path}" >&2
      return 2
    fi
  done
  while IFS=$'\t' read -r link_path link_target; do
    [[ -n "\${link_path}" ]] || continue
    test -L "\${snapshot_root}/\${link_path}"
    test "\$(readlink "\${snapshot_root}/\${link_path}")" = "\${link_target}"
  done <"\${snapshot_root}/.holosoma_snapshot/source_symlinks.tsv"
  while IFS=$'\t' read -r link_path asset_path; do
    [[ -n "\${link_path}" ]] || continue
    test -L "\${snapshot_root}/\${link_path}"
    test "\$(readlink "\${snapshot_root}/\${link_path}")" = "\${ASSET_REPO}/\${asset_path}"
    test -e "\${snapshot_root}/\${link_path}"
  done <"\${snapshot_root}/.holosoma_snapshot/asset_links.tsv"
  local expected_symlink_count actual_symlink_count
  expected_symlink_count=\$((
    \$(wc -l <"\${snapshot_root}/.holosoma_snapshot/source_symlinks.tsv")
    + \$(wc -l <"\${snapshot_root}/.holosoma_snapshot/asset_links.tsv")
  ))
  actual_symlink_count=\$(cd "\${snapshot_root}" && {
    find . -mindepth 1 -maxdepth 1 -type l -print
    for source_dir in src scripts tests submodules .holosoma_snapshot; do
      [[ -d "./\${source_dir}" ]] || continue
      find "./\${source_dir}" -type l -print
    done
  } | wc -l)
  if (( actual_symlink_count != expected_symlink_count )); then
    echo "[ERROR] Snapshot has an unexpected/missing symlink: expected=\${expected_symlink_count} actual=\${actual_symlink_count}" >&2
    return 2
  fi
  if (cd "\${snapshot_root}" && {
      find . -mindepth 1 -maxdepth 1 \
        \( -type b -o -type c -o -type p -o -type s \) -print
      for source_dir in src scripts tests submodules .holosoma_snapshot; do
        [[ -d "./\${source_dir}" ]] || continue
        find "./\${source_dir}" -xdev \
          \( -type b -o -type c -o -type p -o -type s \) -print
      done
    } | grep -q .); then
    echo "[ERROR] Snapshot contains an unsupported special filesystem entry: \${snapshot_root}" >&2
    return 2
  fi
}

if [[ -e "\${RUN_REPO}" && ! -d "\${RUN_REPO}" ]]; then
  echo "[ERROR] Refusing non-directory snapshot destination: \${RUN_REPO}" >&2
  exit 2
fi
if [[ -d "\${RUN_REPO}" ]]; then
  verify_snapshot "\${RUN_REPO}"
  rm -f "\${REMOTE_ARCHIVE}"
  echo "[INFO][${node}] reused_verified_source_snapshot=\${RUN_REPO}"
  exit 0
fi

TEMP_ROOT="\${RUN_ROOT}/.\${SNAPSHOT_ID}.tmp.\$\$"
rm -rf "\${TEMP_ROOT}"
mkdir -p "\${TEMP_ROOT}"
cleanup_snapshot_temp_root() {
  # Signed directories are read-only.  Re-open only this unpublished private
  # tree so failure cleanup can remove it.
  chmod -R u+w "\${TEMP_ROOT}" 2>/dev/null || true
  rm -rf "\${TEMP_ROOT}"
}
trap cleanup_snapshot_temp_root EXIT
# Archive modes are part of the signed source contract.  Preserve them
# exactly instead of letting a node-local login/service umask silently remove
# read or execute bits and create cross-node runtime drift.
tar -xzf "\${REMOTE_ARCHIVE}" -C "\${TEMP_ROOT}" \
  --no-same-owner --same-permissions
test "\$(<"\${TEMP_ROOT}/.holosoma_snapshot/id")" = "\${SNAPSHOT_ID}"
actual_manifest_sha256=\$(sha256sum "\${TEMP_ROOT}/.holosoma_snapshot/source_manifest.sha256" | awk '{print \$1}')
test "\${actual_manifest_sha256}" = "\${EXPECTED_MANIFEST_SHA256}"
(cd "\${TEMP_ROOT}" && sha256sum --quiet -c .holosoma_snapshot/source_manifest.sha256)
if ! source_mode_closure_matches "\${TEMP_ROOT}"; then
  echo "[ERROR] Extracted snapshot source mode closure differs from its signed manifest." >&2
  exit 2
fi
if ! signed_source_directories_are_sealed "\${TEMP_ROOT}"; then
  echo "[ERROR] Extracted snapshot has a writable signed source directory." >&2
  exit 2
fi

# Asset links and the explicit runtime-state roots are the only entries that
# must be created after extraction.  Open only their already authenticated
# real parent directories and remember the exact signed mode for restoration.
declare -A temporarily_opened_directory_modes=()
open_signed_directory_for_install() {
  local directory="\$1"
  local resolved_directory lexical_directory original_mode
  if [[ ! -d "\${directory}" || -L "\${directory}" ]]; then
    echo "[ERROR] Snapshot install parent is missing, non-directory, or symlinked: \${directory}" >&2
    return 2
  fi
  resolved_directory=\$(realpath -e -- "\${directory}")
  lexical_directory=\$(realpath -m -- "\${directory}")
  if [[ "\${resolved_directory}" != "\${lexical_directory}" ]]; then
    echo "[ERROR] Snapshot install parent traverses a symlink: \${directory}" >&2
    return 2
  fi
  if [[ -z "\${temporarily_opened_directory_modes[\${directory}]+x}" ]]; then
    original_mode=\$(stat -c '%a' -- "\${directory}")
    if [[ ! "\${original_mode}" =~ ^[0-7]{3,4}\$ ]]; then
      echo "[ERROR] Could not capture snapshot install parent mode: \${directory}" >&2
      return 2
    fi
    temporarily_opened_directory_modes["\${directory}"]="\${original_mode}"
    chmod u+w "\${directory}"
  fi
}

while IFS=$'\t' read -r link_path asset_path; do
  [[ -n "\${link_path}" ]] || continue
  case "\${link_path}" in
    /*|..|../*|*/../*|*/..) echo "[ERROR] Unsafe snapshot asset-link path: \${link_path}" >&2; exit 2 ;;
  esac
  case "\${asset_path}" in
    /*|..|../*|*/../*|*/..) echo "[ERROR] Unsafe node asset path: \${asset_path}" >&2; exit 2 ;;
  esac
  asset_target="\${ASSET_REPO}/\${asset_path}"
  if [[ ! -e "\${asset_target}" ]]; then
    echo "[ERROR] Required node-local snapshot asset is missing: \${asset_target}" >&2
    exit 2
  fi
  link_parent="\$(dirname -- "\${TEMP_ROOT}/\${link_path}")"
  open_signed_directory_for_install "\${link_parent}"
  if [[ -e "\${TEMP_ROOT}/\${link_path}" || -L "\${TEMP_ROOT}/\${link_path}" ]]; then
    echo "[ERROR] Snapshot archive unexpectedly contains asset-link destination: \${link_path}" >&2
    exit 2
  fi
  ln -s "\${asset_target}" "\${TEMP_ROOT}/\${link_path}"
done <"\${TEMP_ROOT}/.holosoma_snapshot/asset_links.tsv"

# These paths contain mutable launch control, exact checkpoint caches, and
# logs.  They are intentionally excluded from the signed source manifest, but
# are created before the 0555 root is restored so their names cannot be
# accidentally replaced through the snapshot parent.
open_signed_directory_for_install "\${TEMP_ROOT}"
mkdir -p \
  "\${TEMP_ROOT}/.checkpoint_cache" \
  "\${TEMP_ROOT}/.teacher_checkpoints" \
  "\${TEMP_ROOT}/.run_control" \
  "\${TEMP_ROOT}/logs/batch_ne"
chmod 700 \
  "\${TEMP_ROOT}/.checkpoint_cache" \
  "\${TEMP_ROOT}/.teacher_checkpoints" \
  "\${TEMP_ROOT}/.run_control" \
  "\${TEMP_ROOT}/logs" \
  "\${TEMP_ROOT}/logs/batch_ne"

for opened_directory in "\${!temporarily_opened_directory_modes[@]}"; do
  chmod "\${temporarily_opened_directory_modes[\${opened_directory}]}" \
    "\${opened_directory}"
done
unset temporarily_opened_directory_modes opened_directory link_parent

verify_snapshot "\${TEMP_ROOT}"
mv "\${TEMP_ROOT}" "\${RUN_REPO}"
trap - EXIT
rm -f "\${REMOTE_ARCHIVE}"
verify_snapshot "\${RUN_REPO}"
echo "[INFO][${node}] installed_verified_source_snapshot=\${RUN_REPO}"
EOF
)
  remote_run_preflight_bounded "${node}" "${cmd}"
}

cleanup_python_runtime_transfer_node() {
  local node="$1"
  local transfer_token="$2"
  local runtime_id="$3"
  local archive_sha256="$4"
  local runtime_root="${REMOTE_RUN_ROOT}/.runtime/python"
  local incoming_root="${runtime_root}/.incoming"
  local transfer_root="${incoming_root}/${transfer_token}"
  local remote_archive="${transfer_root}/${runtime_id}.${archive_sha256}.tar.gz"
  local cleanup_body cleanup_cmd
  cleanup_body=$(cat <<'REMOTE'
set -euo pipefail
shopt -s dotglob nullglob
current_uid=$(id -u)
if [[ ! "$TOKEN" =~ ^[0-9a-f]{64}$ \
      || "$TRANSFER_ROOT" != "$INCOMING_ROOT/$TOKEN" \
      || "${REMOTE_ARCHIVE%/*}" != "$TRANSFER_ROOT" ]]; then
  echo "[ERROR][$NODE_LABEL] Refusing an unbound Python runtime transfer cleanup." >&2
  exit 2
fi
for spec in "$RUNTIME_ROOT:700" "$INCOMING_ROOT:700"; do
  path=${spec%:*}
  expected_mode=${spec##*:}
  if [[ ! -d "$path" || -L "$path" \
        || "$(realpath -e -- "$path" 2>/dev/null || true)" != "$path" \
        || "$(stat -c '%u:%a' -- "$path" 2>/dev/null || true)" != "$current_uid:$expected_mode" ]]; then
    echo "[ERROR][$NODE_LABEL] Runtime transfer cleanup namespace is malformed: $path" >&2
    exit 2
  fi
done
if [[ ! -e "$TRANSFER_ROOT" && ! -L "$TRANSFER_ROOT" ]]; then
  exit 0
fi
if [[ ! -d "$TRANSFER_ROOT" || -L "$TRANSFER_ROOT" \
      || "$(realpath -e -- "$TRANSFER_ROOT" 2>/dev/null || true)" != "$TRANSFER_ROOT" \
      || "$(stat -c '%u:%a' -- "$TRANSFER_ROOT" 2>/dev/null || true)" != "$current_uid:700" ]]; then
  echo "[ERROR][$NODE_LABEL] Runtime transfer token directory is malformed during cleanup." >&2
  exit 2
fi
entries=("$TRANSFER_ROOT"/*)
if (( ${#entries[@]} > 1 )); then
  echo "[ERROR][$NODE_LABEL] Runtime transfer token directory contains multiple entries." >&2
  exit 2
fi
if (( ${#entries[@]} == 1 )) && [[ "${entries[0]}" != "$REMOTE_ARCHIVE" ]]; then
  echo "[ERROR][$NODE_LABEL] Runtime transfer token directory contains an unexpected entry." >&2
  exit 2
fi
if (( ${#entries[@]} == 1 )); then
  if [[ ! -f "$REMOTE_ARCHIVE" || -L "$REMOTE_ARCHIVE" \
        || "$(stat -c '%h:%u' -- "$REMOTE_ARCHIVE" 2>/dev/null || true)" != "1:$current_uid" ]]; then
    echo "[ERROR][$NODE_LABEL] Refusing to remove an aliased runtime transfer archive." >&2
    exit 2
  fi
  chmod 600 -- "$REMOTE_ARCHIVE"
  rm -f -- "$REMOTE_ARCHIVE"
fi
rmdir -- "$TRANSFER_ROOT"
REMOTE
)
  cleanup_cmd="TOKEN=$(quote "${transfer_token}")"$'\n'
  cleanup_cmd+="RUNTIME_ROOT=$(quote "${runtime_root}")"$'\n'
  cleanup_cmd+="INCOMING_ROOT=$(quote "${incoming_root}")"$'\n'
  cleanup_cmd+="TRANSFER_ROOT=$(quote "${transfer_root}")"$'\n'
  cleanup_cmd+="REMOTE_ARCHIVE=$(quote "${remote_archive}")"$'\n'
  cleanup_cmd+="NODE_LABEL=$(quote "${node}")"$'\n'
  cleanup_cmd+="${cleanup_body}"
  remote_run_mutation_bounded \
    "${node}" "${cleanup_cmd}" "${LAUNCH_CLEANUP_TIMEOUT_SECONDS}"
}

reconcile_python_runtime_publication_node() {
  local node="$1"
  local runtime_root="${REMOTE_RUN_ROOT}/.runtime/python"
  local verifier="${RUN_REPO}/scripts/verify_python_runtime_overlay.py"
  local installer="${RUN_REPO}/scripts/install_python_runtime_overlay.py"
  local runtime_path
  runtime_path="$(dirname -- "${PYTHON_BIN}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  local body cmd
  body=$(cat <<'REMOTE'
set -euo pipefail
unset BASH_ENV ENV CDPATH PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT
unset PYTHONOPTIMIZE PYTHONWARNINGS PYTHONBREAKPOINT PYTHONSAFEPATH LD_PRELOAD
export PATH="$RUNTIME_PATH"
export PYTHONDONTWRITEBYTECODE=1
export LC_ALL=C
"$PYTHON_BIN_REMOTE" -I -S "$INSTALLER" --runtime-root "$RUNTIME_ROOT" --manifest-sha256 "$MANIFEST_SHA256" --verifier "$VERIFIER" --lock-timeout-seconds "$LOCK_TIMEOUT" --probe-only
REMOTE
)
  cmd="RUNTIME_PATH=$(quote "${runtime_path}")"$'\n'
  cmd+="PYTHON_BIN_REMOTE=$(quote "${PYTHON_BIN}")"$'\n'
  cmd+="INSTALLER=$(quote "${installer}")"$'\n'
  cmd+="RUNTIME_ROOT=$(quote "${runtime_root}")"$'\n'
  cmd+="MANIFEST_SHA256=$(quote "${PYTHON_RUNTIME_MANIFEST_SHA256}")"$'\n'
  cmd+="VERIFIER=$(quote "${verifier}")"$'\n'
  cmd+="LOCK_TIMEOUT=$(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}")"$'\n'
  cmd+="${body}"
  remote_run_mutation_bounded \
    "${node}" "${cmd}" "${LAUNCH_CLEANUP_TIMEOUT_SECONDS}"
}

prepare_python_runtime_overlay_node() {
  local node="$1"
  if [[ -z "${PYTHON_RUNTIME_SITEPACKAGES}" ]]; then
    return 0
  fi
  local runtime_root="${REMOTE_RUN_ROOT}/.runtime/python"
  local runtime_id="python-runtime-v2-${PYTHON_RUNTIME_MANIFEST_SHA256}"
  local verifier="${RUN_REPO}/scripts/verify_python_runtime_overlay.py"
  local installer="${RUN_REPO}/scripts/install_python_runtime_overlay.py"
  local runtime_path
  runtime_path="$(dirname -- "${PYTHON_BIN}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

  local namespace_body namespace_cmd
  namespace_body=$(cat <<'REMOTE'
set -euo pipefail
unset BASH_ENV ENV CDPATH PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT
unset PYTHONOPTIMIZE PYTHONWARNINGS PYTHONBREAKPOINT PYTHONSAFEPATH LD_PRELOAD
CURRENT_UID=$(id -u)
if [[ ! -d "$RUN_ROOT" || -L "$RUN_ROOT" || "$(realpath -e -- "$RUN_ROOT")" != "$RUN_ROOT" || "$(stat -c '%u' -- "$RUN_ROOT")" != "$CURRENT_UID" ]]; then
  echo "[ERROR][$NODE_LABEL] Python runtime parent is not one real current-UID run root." >&2
  exit 2
fi
for spec in ".runtime:755" ".runtime/python:700" ".runtime/python/.incoming:700" ".runtime/python/.locks:700"; do
  relative=${spec%%:*}
  expected_mode=${spec##*:}
  path="$RUN_ROOT/$relative"
  if [[ -e "$path" || -L "$path" ]]; then
    if [[ ! -d "$path" || -L "$path" || "$(realpath -e -- "$path")" != "$path" || "$(stat -c '%u' -- "$path")" != "$CURRENT_UID" ]]; then
      echo "[ERROR][$NODE_LABEL] Python runtime namespace is aliased or has the wrong owner: $path" >&2
      exit 2
    fi
  else
    mkdir -m "$expected_mode" -- "$path"
  fi
  chmod "$expected_mode" -- "$path"
  if [[ "$(stat -c '%a' -- "$path")" != "$expected_mode" ]]; then
    echo "[ERROR][$NODE_LABEL] Python runtime namespace did not reach mode $expected_mode: $path" >&2
    exit 2
  fi
done
REMOTE
)
  namespace_cmd="RUN_ROOT=$(quote "${REMOTE_RUN_ROOT}")"$'\n'"NODE_LABEL=$(quote "${node}")"$'\n'"${namespace_body}"
  remote_run_preflight_bounded "${node}" "${namespace_cmd}"

  local probe_body probe_cmd
  probe_body=$(cat <<'REMOTE'
set -euo pipefail
unset BASH_ENV ENV CDPATH PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT
unset PYTHONOPTIMIZE PYTHONWARNINGS PYTHONBREAKPOINT PYTHONSAFEPATH LD_PRELOAD
export PATH="$RUNTIME_PATH"
export PYTHONDONTWRITEBYTECODE=1
export LC_ALL=C
"$PYTHON_BIN_REMOTE" -I -S "$INSTALLER" --runtime-root "$RUNTIME_ROOT" --manifest-sha256 "$MANIFEST_SHA256" --verifier "$VERIFIER" --lock-timeout-seconds "$LOCK_TIMEOUT" --probe-only
REMOTE
)
  probe_cmd="RUNTIME_PATH=$(quote "${runtime_path}")"$'\n'
  probe_cmd+="PYTHON_BIN_REMOTE=$(quote "${PYTHON_BIN}")"$'\n'
  probe_cmd+="INSTALLER=$(quote "${installer}")"$'\n'
  probe_cmd+="RUNTIME_ROOT=$(quote "${runtime_root}")"$'\n'
  probe_cmd+="MANIFEST_SHA256=$(quote "${PYTHON_RUNTIME_MANIFEST_SHA256}")"$'\n'
  probe_cmd+="VERIFIER=$(quote "${verifier}")"$'\n'
  probe_cmd+="LOCK_TIMEOUT=$(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}")"$'\n'
  probe_cmd+="${probe_body}"
  if [[ "${DRY_RUN}" == 1 ]]; then
    remote_run_preflight_bounded "${node}" "${probe_cmd}"
  elif remote_run_mutation_bounded \
      "${node}" "${probe_cmd}" "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}"; then
    echo "[INFO][${node}] python_runtime_prepare_reused=${PYTHON_RUNTIME_SITEPACKAGES}"
    return 0
  else
    local probe_status=$?
    if (( probe_status != 3 )); then
      echo "[ERROR][${node}] Existing Python runtime is malformed or could not be probed." >&2
      return "${probe_status}"
    fi
  fi

  local transfer_token
  transfer_token=$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')
  if [[ ! "${transfer_token}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[ERROR][${node}] Could not generate a private Python runtime transfer token." >&2
    return 2
  fi
  local transfer_root="${runtime_root}/.incoming/${transfer_token}"
  local remote_archive="${transfer_root}/${runtime_id}.${PYTHON_RUNTIME_ARCHIVE_SHA256}.tar.gz"
  local reserve_body reserve_cmd
  reserve_body=$(cat <<'REMOTE'
set -euo pipefail
umask 077
if [[ -e "$TRANSFER_ROOT" || -L "$TRANSFER_ROOT" ]]; then
  echo "[ERROR][$NODE_LABEL] Runtime transfer token path already exists." >&2
  exit 2
fi
mkdir -m 700 -- "$TRANSFER_ROOT"
test "$(stat -c '%u:%a' -- "$TRANSFER_ROOT")" = "$(id -u):700"
REMOTE
)
  reserve_cmd="TRANSFER_ROOT=$(quote "${transfer_root}")"$'\n'"NODE_LABEL=$(quote "${node}")"$'\n'"${reserve_body}"
  local transfer_status
  if remote_run_mutation_bounded \
      "${node}" "${reserve_cmd}" "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}"; then
    :
  else
    transfer_status=$?
    if ! cleanup_python_runtime_transfer_node \
        "${node}" "${transfer_token}" "${runtime_id}" \
        "${PYTHON_RUNTIME_ARCHIVE_SHA256}"; then
      echo "[ERROR][${node}] Ambiguous runtime transfer reservation could not be cleaned." >&2
    fi
    return "${transfer_status}"
  fi
  if remote_copy_to_bounded \
      "${PYTHON_RUNTIME_ARCHIVE}" "${node}" "${remote_archive}" \
      "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}"; then
    :
  else
    transfer_status=$?
    echo "[ERROR][${node}] Python runtime archive transfer failed." >&2
    if ! cleanup_python_runtime_transfer_node \
        "${node}" "${transfer_token}" "${runtime_id}" \
        "${PYTHON_RUNTIME_ARCHIVE_SHA256}"; then
      echo "[ERROR][${node}] Failed runtime archive transfer could not be cleaned." >&2
    fi
    return "${transfer_status}"
  fi

  local publish_body publish_cmd
  publish_body=$(cat <<'REMOTE'
set -euo pipefail
unset BASH_ENV ENV CDPATH PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT
unset PYTHONOPTIMIZE PYTHONWARNINGS PYTHONBREAKPOINT PYTHONSAFEPATH LD_PRELOAD
export PATH="$RUNTIME_PATH"
export PYTHONDONTWRITEBYTECODE=1
export LC_ALL=C
cleanup_transfer() {
  if [[ -d "$TRANSFER_ROOT" && ! -L "$TRANSFER_ROOT" && "$(stat -c '%u:%a' -- "$TRANSFER_ROOT")" == "$(id -u):700" ]]; then
    if [[ -e "$REMOTE_ARCHIVE" || -L "$REMOTE_ARCHIVE" ]]; then
      if [[ -f "$REMOTE_ARCHIVE" && ! -L "$REMOTE_ARCHIVE" && "$(stat -c '%h:%u' -- "$REMOTE_ARCHIVE")" == "1:$(id -u)" ]]; then
        chmod 600 -- "$REMOTE_ARCHIVE" || true
        rm -f -- "$REMOTE_ARCHIVE" || true
      fi
    fi
    rmdir -- "$TRANSFER_ROOT" 2>/dev/null || true
  fi
}
trap cleanup_transfer EXIT
if [[ ! -f "$REMOTE_ARCHIVE" || -L "$REMOTE_ARCHIVE" \
      || "$(stat -c '%h:%u:%s' -- "$REMOTE_ARCHIVE")" != "1:$(id -u):$ARCHIVE_SIZE" ]]; then
  echo "[ERROR][$NODE_LABEL] Transferred Python runtime archive metadata is malformed." >&2
  exit 2
fi
chmod 400 -- "$REMOTE_ARCHIVE"
if [[ "$(stat -c '%h:%u:%a:%s' -- "$REMOTE_ARCHIVE")" != "1:$(id -u):400:$ARCHIVE_SIZE" ]]; then
  echo "[ERROR][$NODE_LABEL] Transferred Python runtime archive could not be sealed 0400." >&2
  exit 2
fi
"$PYTHON_BIN_REMOTE" -I -S "$INSTALLER" --runtime-root "$RUNTIME_ROOT" --manifest-sha256 "$MANIFEST_SHA256" --verifier "$VERIFIER" --archive "$REMOTE_ARCHIVE" --archive-sha256 "$ARCHIVE_SHA256" --lock-timeout-seconds "$LOCK_TIMEOUT"
echo "[INFO][$NODE_LABEL] python_runtime_prepare_installed=$SITE_PACKAGES"
REMOTE
)
  publish_cmd="RUNTIME_PATH=$(quote "${runtime_path}")"$'\n'
  publish_cmd+="PYTHON_BIN_REMOTE=$(quote "${PYTHON_BIN}")"$'\n'
  publish_cmd+="INSTALLER=$(quote "${installer}")"$'\n'
  publish_cmd+="RUNTIME_ROOT=$(quote "${runtime_root}")"$'\n'
  publish_cmd+="MANIFEST_SHA256=$(quote "${PYTHON_RUNTIME_MANIFEST_SHA256}")"$'\n'
  publish_cmd+="VERIFIER=$(quote "${verifier}")"$'\n'
  publish_cmd+="REMOTE_ARCHIVE=$(quote "${remote_archive}")"$'\n'
  publish_cmd+="ARCHIVE_SHA256=$(quote "${PYTHON_RUNTIME_ARCHIVE_SHA256}")"$'\n'
  publish_cmd+="ARCHIVE_SIZE=$(quote "${PYTHON_RUNTIME_ARCHIVE_SIZE}")"$'\n'
  publish_cmd+="TRANSFER_ROOT=$(quote "${transfer_root}")"$'\n'
  publish_cmd+="LOCK_TIMEOUT=$(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}")"$'\n'
  publish_cmd+="NODE_LABEL=$(quote "${node}")"$'\n'
  publish_cmd+="SITE_PACKAGES=$(quote "${PYTHON_RUNTIME_SITEPACKAGES}")"$'\n'
  publish_cmd+="${publish_body}"
  if remote_run_mutation_bounded \
      "${node}" "${publish_cmd}" "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}"; then
    return 0
  else
    transfer_status=$?
  fi
  local transfer_cleanup_verified=0
  if cleanup_python_runtime_transfer_node \
      "${node}" "${transfer_token}" "${runtime_id}" \
      "${PYTHON_RUNTIME_ARCHIVE_SHA256}"; then
    transfer_cleanup_verified=1
  else
    echo "[ERROR][${node}] Failed runtime publication could not prove transfer cleanup." >&2
  fi
  local reconcile_status
  if reconcile_python_runtime_publication_node "${node}"; then
    if (( transfer_cleanup_verified == 0 )); then
      if cleanup_python_runtime_transfer_node \
          "${node}" "${transfer_token}" "${runtime_id}" \
          "${PYTHON_RUNTIME_ARCHIVE_SHA256}"; then
        transfer_cleanup_verified=1
      fi
    fi
    if (( transfer_cleanup_verified == 1 )); then
      echo "[INFO][${node}] Reconciled an ambiguous runtime publication to one strictly verified installed runtime."
      return 0
    fi
    echo "[ERROR][${node}] Runtime publication completed but transfer cleanup is unconfirmed." >&2
    return "${transfer_status}"
  else
    reconcile_status=$?
  fi
  if (( reconcile_status == 3 && transfer_cleanup_verified == 1 )); then
    echo "[ERROR][${node}] Runtime publication failed with a verified missing terminal state." >&2
  elif (( reconcile_status == 3 )); then
    echo "[ERROR][${node}] Runtime is currently missing but transfer revocation is unconfirmed." >&2
  else
    echo "[ERROR][${node}] Runtime publication terminal state remains unconfirmed after reconciliation." >&2
  fi
  return "${transfer_status}"
}

prepare_node() {
  local node="$1"
  resolve_node_corl_source "${node}"
  local node_corl_bank="${RESOLVED_NODE_CORL_BANK}"
  install_source_snapshot_node "${node}"
  prepare_python_runtime_overlay_node "${node}"
  local runtime_pythonpath="${RUN_REPO}/src/holosoma:${RUN_REPO}/src/holosoma_inference:${RUN_REPO}/src"
  if [[ -n "${PYTHON_RUNTIME_SITEPACKAGES}" ]]; then
    runtime_pythonpath="${PYTHON_RUNTIME_SITEPACKAGES}:${runtime_pythonpath}"
  fi
  local runtime_path
  runtime_path="$(dirname -- "${PYTHON_BIN}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
unset BASH_ENV ENV CDPATH
unset PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONOPTIMIZE PYTHONWARNINGS PYTHONBREAKPOINT PYTHONSAFEPATH
unset LD_PRELOAD
export PATH=$(quote "${runtime_path}")
export PYTHONPATH=$(quote "${runtime_pythonpath}")
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$(quote "${NCCL_LIB_DIR}")
cd $(quote "${RUN_REPO}")
echo "[INFO][${node}] isolated_source_repo=\$(pwd) asset_repo=$(quote "${REMOTE_REPO}")"
echo "[INFO][${node}] remote dirty repository source is never modified; SKIP_GIT_PULL=$(quote "${SKIP_GIT_PULL}") is compatibility-only"
grep -Fx -- $(quote "${SOURCE_SNAPSHOT_ID}") .holosoma_snapshot/id >/dev/null
(sha256sum --quiet -c .holosoma_snapshot/source_manifest.sha256)
# Launch/copy wrappers are always invoked through bash below.  Do not mutate
# their signed executable modes after the install verifier has accepted them.
export PYTHON_BIN=$(quote "${PYTHON_BIN}")
export PYTHONHASHSEED=$(quote "${PYTHONHASHSEED}")
export CUBLAS_WORKSPACE_CONFIG=$(quote "${CUBLAS_WORKSPACE_CONFIG}")
export HOLOSOMA_PYTHON_PROFILE=hssim
export PYTHONDONTWRITEBYTECODE=1
source ./scripts/gpu_launch_defaults.sh
PYTHON_RUNTIME_SITEPACKAGES=$(quote "${PYTHON_RUNTIME_SITEPACKAGES}")
PYTHON_RUNTIME_MANIFEST_SHA256=$(quote "${PYTHON_RUNTIME_MANIFEST_SHA256}")
if [[ -n "\${PYTHON_RUNTIME_SITEPACKAGES}" ]]; then
  "\${PYTHON_BIN}" scripts/verify_python_runtime_overlay.py \
    --site-packages "\${PYTHON_RUNTIME_SITEPACKAGES}" \
    --manifest-sha256 "\${PYTHON_RUNTIME_MANIFEST_SHA256}" \
    --require-distribution-closure \
    --require-current-runtime-binding
  echo "[INFO][${node}] python_runtime_prepare_overlay_verified=\${PYTHON_RUNTIME_SITEPACKAGES} manifest_sha256=\${PYTHON_RUNTIME_MANIFEST_SHA256}"
fi
if [[ $(quote "${PREPARE_DATA}") == "0" ]]; then
  echo "[INFO][${node}] immutable source snapshot prepared; PREPARE_DATA=0 leaves node-local data unchanged"
elif [[ $(quote "${PREPARE_COPY_SCRIPT}") == "cp_corl" ]]; then
  CORL_BANK_NAME=$(quote "${CORL_SOLID80_BANK_NAME}") \
  NFS_CORL_BANK=$(quote "${node_corl_bank}") \
  LOCAL_BANK_NAME=$(quote "${LOCAL_BANK_NAME:-${CORL_SOLID80_BANK_NAME}}") \
  EXPECTED_CLIP_COUNT=$(quote "${EXPECTED_CLIP_COUNT:-${OMOMO_EXPECTED_TOTAL}}") \
  KEEP_BACKUP=$(quote "${KEEP_BACKUP}") \
  bash cp_corl.sh
else
  PULL_CODE=0 CH_BANK_NAME=$(quote "${CH_BANK_NAME}") NFS_CH_BANK=$(quote "${NFS_CH_BANK}") bash cp_ch.sh
fi
EOF
)
  remote_run_preflight_bounded "${node}" "${cmd}"
}

rendezvous_state_path() {
  local port="$1"
  local master_key
  master_key=$(printf '%s' "${MASTER_ADDR}" | sha256sum | awk '{print $1}')
  printf '%s/.rendezvous/%s_%s.state' "${REMOTE_RUN_ROOT}" "${master_key}" "${port}"
}

reserve_rendezvous_ports() {
  local launch_token="$1"
  local main_state provenance_state
  main_state=$(rendezvous_state_path "${MASTER_PORT}")
  provenance_state=$(rendezvous_state_path "${HOLOSOMA_PROVENANCE_MASTER_PORT}")
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
STATE_ROOT=$(quote "${REMOTE_RUN_ROOT}/.rendezvous")
MAIN_STATE=$(quote "${main_state}")
PROVENANCE_STATE=$(quote "${provenance_state}")
TOKEN=$(quote "${launch_token}")
SESSION_NAME=$(quote "${SESSION}")
CANCELLATION_STATE="\${STATE_ROOT}/cancelled.\${TOKEN}.state"
mkdir -p "\${STATE_ROOT}"
$(private_lifecycle_file_validation_helpers)
open_private_lifecycle_lock "\${STATE_ROOT}/.reservation.lock" 9
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 9; then
  echo "[ERROR] Timed out acquiring the rendezvous reservation lock." >&2
  exit 1
fi
if [[ -e "\${CANCELLATION_STATE}" || -L "\${CANCELLATION_STATE}" ]]; then
  if ! validate_private_state_file_metadata "\${CANCELLATION_STATE}" 4096 0 \
      || [[ ! -f "\${CANCELLATION_STATE}" || -L "\${CANCELLATION_STATE}" \
        || "\$(awk 'END { print NR }' "\${CANCELLATION_STATE}")" != 1 \
        || "\$(awk -F '\t' 'NR == 1 { print NF }' "\${CANCELLATION_STATE}")" != 6 ]]; then
    echo "[ERROR] Rendezvous cancellation tombstone is malformed: \${CANCELLATION_STATE}" >&2
    exit 2
  fi
  cancel_version="" cancel_token="" cancel_session="" cancel_master=""
  cancel_main_port="" cancel_provenance_port=""
  IFS=\$'\t' read -r cancel_version cancel_token cancel_session cancel_master \
    cancel_main_port cancel_provenance_port < "\${CANCELLATION_STATE}" || true
  if [[ "\${cancel_version}" != 1 || "\${cancel_token}" != "\${TOKEN}" \
        || "\${cancel_session}" != "\${SESSION_NAME}" \
        || "\${cancel_master}" != $(quote "${MASTER_ADDR}") \
        || "\${cancel_main_port}" != $(quote "${MASTER_PORT}") \
        || "\${cancel_provenance_port}" != $(quote "${HOLOSOMA_PROVENANCE_MASTER_PORT}") ]]; then
    echo "[ERROR] Rendezvous cancellation tombstone identity mismatch: \${CANCELLATION_STATE}" >&2
    exit 2
  fi
  echo "[ERROR] Rendezvous reservation token was durably cancelled before commit." >&2
  exit 3
fi
incoming_main="\${MAIN_STATE}.incoming.\${TOKEN}"
incoming_provenance="\${PROVENANCE_STATE}.incoming.\${TOKEN}"
cleanup_incoming() { rm -f -- "\${incoming_main}" "\${incoming_provenance}"; }
trap cleanup_incoming EXIT
for state in "\${MAIN_STATE}" "\${PROVENANCE_STATE}"; do
  if [[ -e "\${state}" || -L "\${state}" ]]; then
    echo "[ERROR] Rendezvous port is already reserved: \${state} metadata=\$(cat "\${state}" 2>/dev/null || true)" >&2
    exit 2
  fi
done
for port in $(quote "${MASTER_PORT}") $(quote "${HOLOSOMA_PROVENANCE_MASTER_PORT}"); do
  listener_output=\$(ss -H -ltn "sport = :\${port}") || {
    echo "[ERROR] Failed to inspect rendezvous TCP port ${MASTER_ADDR}:\${port}." >&2
    exit 2
  }
  if [[ -n "\${listener_output}" ]]; then
    echo "[ERROR] Rendezvous TCP port is already listening on ${MASTER_ADDR}:\${port}." >&2
    exit 2
  fi
done
created_at=\$(date +%s)
if [[ ! "\${created_at}" =~ ^[1-9][0-9]*$ || \${#created_at} -gt 18 ]]; then
  echo "[ERROR] Rendezvous reservation clock returned a non-canonical or unsafe epoch: \${created_at}" >&2
  exit 2
fi
printf '2\t%s\t%s\t%s\t%s\n' "\${TOKEN}" "\${SESSION_NAME}" $(quote "${MASTER_PORT}") "\${created_at}" > "\${incoming_main}"
printf '2\t%s\t%s\t%s\t%s\n' "\${TOKEN}" "\${SESSION_NAME}" $(quote "${HOLOSOMA_PROVENANCE_MASTER_PORT}") "\${created_at}" > "\${incoming_provenance}"
mv -T "\${incoming_main}" "\${MAIN_STATE}"
if ! mv -T "\${incoming_provenance}" "\${PROVENANCE_STATE}"; then
  rm -f -- "\${MAIN_STATE}"
  exit 2
fi
validate_private_state_file_metadata "\${MAIN_STATE}" 4096 0
validate_private_state_file_metadata "\${PROVENANCE_STATE}" 4096 0
trap - EXIT
echo "[INFO] reserved rendezvous ports session=\${SESSION_NAME} master=${MASTER_ADDR}:${MASTER_PORT} provenance=${MASTER_ADDR}:${HOLOSOMA_PROVENANCE_MASTER_PORT} token=\${TOKEN}"
EOF
)
  remote_run_bounded "${MASTER_ADDR}" "${cmd}" "${LAUNCH_CONTROL_TIMEOUT_SECONDS}"
}

cancel_rendezvous_reservation() {
  local launch_token="$1"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
STATE_ROOT=$(quote "${REMOTE_RUN_ROOT}/.rendezvous")
TOKEN=$(quote "${launch_token}")
CANCELLATION_STATE="\${STATE_ROOT}/cancelled.\${TOKEN}.state"
mkdir -p "\${STATE_ROOT}"
$(private_lifecycle_file_validation_helpers)
open_private_lifecycle_lock "\${STATE_ROOT}/.reservation.lock" 9
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 9; then
  echo "[ERROR] Timed out acquiring the rendezvous cancellation lock." >&2
  exit 1
fi
validate_reservation_cancellation() {
  validate_private_state_file_metadata "\${CANCELLATION_STATE}" 4096 0 \
    && [[ -f "\${CANCELLATION_STATE}" && ! -L "\${CANCELLATION_STATE}" \
      && "\$(awk 'END { print NR }' "\${CANCELLATION_STATE}")" == 1 \
      && "\$(awk -F '\t' 'NR == 1 { print NF }' "\${CANCELLATION_STATE}")" == 6 ]] || return 2
  local version token session_name master_addr main_port provenance_port
  version="" token="" session_name="" master_addr="" main_port="" provenance_port=""
  IFS=\$'\t' read -r version token session_name master_addr main_port provenance_port \
    < "\${CANCELLATION_STATE}" || true
  [[ "\${version}" == 1 \
      && "\${token}" == "\${TOKEN}" \
      && "\${session_name}" == $(quote "${SESSION}") \
      && "\${master_addr}" == $(quote "${MASTER_ADDR}") \
      && "\${main_port}" == $(quote "${MASTER_PORT}") \
      && "\${provenance_port}" == $(quote "${HOLOSOMA_PROVENANCE_MASTER_PORT}") ]]
}
if [[ -e "\${CANCELLATION_STATE}" || -L "\${CANCELLATION_STATE}" ]]; then
  validate_reservation_cancellation || {
    echo "[ERROR] Existing rendezvous cancellation tombstone is malformed." >&2
    exit 2
  }
else
  incoming="\${CANCELLATION_STATE}.incoming.\$\$"
  printf '1\t%s\t%s\t%s\t%s\t%s\n' \
    "\${TOKEN}" $(quote "${SESSION}") $(quote "${MASTER_ADDR}") \
    $(quote "${MASTER_PORT}") $(quote "${HOLOSOMA_PROVENANCE_MASTER_PORT}") > "\${incoming}"
  mv -T "\${incoming}" "\${CANCELLATION_STATE}"
  validate_private_state_file_metadata "\${CANCELLATION_STATE}" 4096 0
fi
echo "[INFO] durably cancelled rendezvous reservation token=\${TOKEN}"
EOF
)
  remote_run_bounded "${MASTER_ADDR}" "${cmd}" "${LAUNCH_CLEANUP_TIMEOUT_SECONDS}"
}

rendezvous_release_validation_helpers() {
  private_lifecycle_file_validation_helpers
  cat <<'EOF'
validate_rendezvous_state_exact() {
  local state="$1" expected_token="$2" expected_session="$3" expected_port="$4"
  local version owner_token owner_session owner_port created_at field_count line_count
  validate_private_state_file_metadata \
    "${state}" 4096 "${ALLOW_LEGACY_PRIVATE_STATE_MODE:-0}" || return
  line_count=$(awk 'END { print NR }' "${state}") || return 2
  field_count=$(awk -F '\t' 'NR == 1 { print NF }' "${state}") || return 2
  if [[ "${line_count}" != 1 || "${field_count}" != 5 ]]; then
    echo "[ERROR] Rendezvous reservation must contain exactly one five-field TSV record: ${state}" >&2
    return 2
  fi
  version="" owner_token="" owner_session="" owner_port="" created_at=""
  IFS=$'\t' read -r version owner_token owner_session owner_port created_at < "${state}" || true
  if [[ "${version}" != 2 || "${owner_token}" != "${expected_token}" \
        || "${owner_session}" != "${expected_session}" \
        || "${owner_port}" != "${expected_port}" \
        || ! "${created_at}" =~ ^[1-9][0-9]*$ \
        || ${#created_at} -gt 18 ]]; then
    echo "[ERROR] Rendezvous reservation identity mismatch: ${state}" >&2
    return 2
  fi
  validated_rendezvous_created_at="${created_at}"
}

validate_and_release_rendezvous_pair() {
  local main_state="$1" provenance_state="$2" expected_token="$3"
  local expected_session="$4" expected_main_port="$5" expected_provenance_port="$6"
  local require_present="${7:-0}"
  local main_exists=0 provenance_exists=0 main_created_at provenance_created_at
  [[ -e "${main_state}" || -L "${main_state}" ]] && main_exists=1
  [[ -e "${provenance_state}" || -L "${provenance_state}" ]] && provenance_exists=1
  if (( main_exists == 0 && provenance_exists == 0 )); then
    if [[ "${require_present}" == 1 ]]; then
      echo "[ERROR] Authoritative clean-completion release requires both reservation records to be present." >&2
      return 2
    fi
    return 0
  fi
  # Two-phase release: validate the complete pair before entering the delete
  # phase. A missing/corrupt/mismatched member preserves both endpoints.
  validate_rendezvous_state_exact \
    "${main_state}" "${expected_token}" "${expected_session}" "${expected_main_port}" || return
  main_created_at=${validated_rendezvous_created_at}
  validate_rendezvous_state_exact \
    "${provenance_state}" "${expected_token}" "${expected_session}" "${expected_provenance_port}" || return
  provenance_created_at=${validated_rendezvous_created_at}
  if [[ "${main_created_at}" != "${provenance_created_at}" ]]; then
    echo "[ERROR] Rendezvous reservation pair has mismatched transaction timestamps: main=${main_created_at} provenance=${provenance_created_at}" >&2
    return 2
  fi
  local port listener_output
  for port in "${expected_main_port}" "${expected_provenance_port}"; do
    listener_output=$(ss -H -ltn "sport = :${port}") || {
      echo "[ERROR] Failed to verify listener closure for rendezvous TCP port ${port}." >&2
      return 2
    }
    if [[ -n "${listener_output}" ]]; then
      echo "[ERROR] Refusing rendezvous release while TCP port ${port} still has a listener." >&2
      return 2
    fi
  done
  rm -f -- "${main_state}" "${provenance_state}"
  echo "[INFO] released exact rendezvous reservation pair main=${main_state} provenance=${provenance_state} token=${expected_token}"
}
EOF
}

private_lifecycle_file_validation_helpers() {
  cat <<'EOF'
validate_private_state_file_metadata() {
  local path="$1" max_size="$2" allow_legacy_mode="${3:-0}"
  local current_uid owner mode links size extra
  current_uid=$(id -u) || return 2
  if [[ ! -f "${path}" || -L "${path}" ]]; then
    echo "[ERROR] Private lifecycle state is missing, non-regular, or symlinked: ${path}" >&2
    return 2
  fi
  owner="" mode="" links="" size="" extra=""
  read -r owner mode links size extra < <(stat -c '%u %a %h %s' -- "${path}") || return 2
  if [[ -n "${extra}" || "${owner}" != "${current_uid}" \
        || "${links}" != 1 || ! "${size}" =~ ^[1-9][0-9]*$ \
        || ! "${max_size}" =~ ^[1-9][0-9]*$ \
        || ${#size} -gt ${#max_size} ]]; then
    echo "[ERROR] Private lifecycle state has unsafe owner/link-count/size: ${path}" >&2
    return 2
  fi
  if (( ${#size} == ${#max_size} && 10#${size} > 10#${max_size} )); then
    echo "[ERROR] Private lifecycle state exceeds ${max_size} bytes: ${path}" >&2
    return 2
  fi
  if [[ "${mode}" != 600 \
        && ! ( "${allow_legacy_mode}" == 1 \
          && ( "${mode}" == 644 || "${mode}" == 664 ) ) ]]; then
    echo "[ERROR] Private lifecycle state mode is not 0600: ${path} mode=${mode}" >&2
    return 2
  fi
  validated_private_state_mode=${mode}
}

validate_private_lifecycle_lock() {
  local path="$1" current_uid owner mode links size dev ino extra
  current_uid=$(id -u) || return 2
  if [[ ! -f "${path}" || -L "${path}" ]]; then
    echo "[ERROR] Lifecycle lock is non-regular or symlinked: ${path}" >&2
    return 2
  fi
  owner="" mode="" links="" size="" dev="" ino="" extra=""
  read -r owner mode links size dev ino extra \
    < <(stat -c '%u %a %h %s %d %i' -- "${path}") || return 2
  if [[ -n "${extra}" || "${owner}" != "${current_uid}" \
        || "${mode}" != 600 || "${links}" != 1 || "${size}" != 0 \
        || ! "${dev}" =~ ^[0-9]+$ || ! "${ino}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] Lifecycle lock has unsafe owner/mode/link-count/size: ${path}" >&2
    return 2
  fi
  validated_lock_dev=${dev}
  validated_lock_ino=${ino}
}

open_private_lifecycle_lock() {
  local path="$1" fd_number="$2" fd_path fd_dev fd_ino
  if [[ ! -e "${path}" && ! -L "${path}" ]]; then
    if ! (umask 077; set -o noclobber; : > "${path}") 2>/dev/null; then
      echo "[ERROR] Failed to create private lifecycle lock: ${path}" >&2
      return 2
    fi
  fi
  validate_private_lifecycle_lock "${path}" || return
  case "${fd_number}" in
    8) exec 8<> "${path}" ; fd_path=/proc/self/fd/8 ;;
    9) exec 9<> "${path}" ; fd_path=/proc/self/fd/9 ;;
    *) echo "[ERROR] Unsupported lifecycle lock descriptor: ${fd_number}" >&2; return 2 ;;
  esac
  fd_dev=$(stat -Lc %d -- "${fd_path}") || return 2
  fd_ino=$(stat -Lc %i -- "${fd_path}") || return 2
  validate_private_lifecycle_lock "${path}" || return
  if [[ "${fd_dev}" != "${validated_lock_dev}" \
        || "${fd_ino}" != "${validated_lock_ino}" ]]; then
    echo "[ERROR] Lifecycle lock pathname changed while it was opened: ${path}" >&2
    return 2
  fi
}
EOF
}

active_state_validation_helpers() {
  # Every lifecycle decision consumes one exact v2 TSV record.  Checking only
  # the first line would let appended metadata be silently ignored by a
  # health check, ownership transition, session kill, or reservation release.
  private_lifecycle_file_validation_helpers
  cat <<'EOF'
canonical_positive_decimal_at_most() {
  local value="$1" maximum="$2"
  local LC_ALL=C
  [[ "${value}" =~ ^[1-9][0-9]*$ \
      && "${maximum}" =~ ^[1-9][0-9]*$ ]] || return 1
  if (( ${#value} < ${#maximum} )); then
    return 0
  fi
  if (( ${#value} > ${#maximum} )); then
    return 1
  fi
  [[ "${value}" == "${maximum}" || "${value}" < "${maximum}" ]]
}

load_active_state_v2_exact() {
  local state="$1" allow_legacy_mode="${2:-0}"
  local line_count field_count active_log_basename
  validated_private_state_mode=""
  validate_private_state_file_metadata "${state}" 4096 "${allow_legacy_mode}" || return
  active_state_legacy_mode=0
  [[ "${validated_private_state_mode}" == 600 ]] || active_state_legacy_mode=1
  line_count=$(awk 'END { print NR }' "${state}") || return 2
  field_count=$(awk -F '\t' 'NR == 1 { print NF }' "${state}") || return 2
  if [[ "${line_count}" != 1 || "${field_count}" != 8 ]]; then
    echo "[ERROR] Active lifecycle metadata must contain exactly one eight-field TSV record: ${state}" >&2
    return 2
  fi
  active_version="" active_phase="" active_snapshot="" active_log_dir=""
  active_target="" active_token="" active_command_sha="" active_epoch=""
  IFS=$'\t' read -r active_version active_phase active_snapshot active_log_dir \
    active_target active_token active_command_sha active_epoch < "${state}" || true
  if [[ "${active_version}" != 2 \
        || ! "${active_snapshot}" =~ ^src-[0-9a-f]{64}$ \
        || ! "${active_log_dir}" =~ ^logs/batch_ne/[A-Za-z0-9][A-Za-z0-9_.-]{0,256}$ \
        || ! "${active_target}" =~ ^[1-9][0-9]*$ \
        || ! "${active_token}" =~ ^[0-9a-f]{64}$ \
        || ! "${active_epoch}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] Active lifecycle metadata has malformed v2 identity fields: ${state}" >&2
    return 2
  fi
  active_log_basename=${active_log_dir#logs/batch_ne/}
  if (( ${#active_log_basename} > 255 )) \
      || ! canonical_positive_decimal_at_most "${active_target}" 2147483647; then
    echo "[ERROR] Active lifecycle metadata exceeds numeric/path bounds: ${state}" >&2
    return 2
  fi
  case "${active_phase}" in
    launching)
      [[ "${active_command_sha}" == pending ]] || return 2
      ;;
    running|stopping|stopped)
      [[ "${active_command_sha}" =~ ^[0-9a-f]{64}$ ]] || return 2
      ;;
    rolling_back|rolled_back)
      [[ "${active_command_sha}" == pending \
          || "${active_command_sha}" =~ ^[0-9a-f]{64}$ ]] || return 2
      ;;
    *)
      echo "[ERROR] Active lifecycle metadata has unsupported phase=${active_phase}: ${state}" >&2
      return 2
      ;;
  esac
}

migrate_loaded_active_state_to_private() {
  local state="$1" incoming
  if [[ "${active_state_legacy_mode:-0}" != 1 ]]; then
    return 0
  fi
  incoming="${state}.incoming.legacy-mode.${BASHPID:-$$}.${RANDOM}"
  if [[ -e "${incoming}" || -L "${incoming}" ]]; then
    echo "[ERROR] Legacy active-state migration incoming path already exists: ${incoming}" >&2
    return 2
  fi
  if ! (umask 077; set -o noclobber; printf \
      '2\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${active_phase}" "${active_snapshot}" "${active_log_dir}" \
      "${active_target}" "${active_token}" "${active_command_sha}" \
      "${active_epoch}" > "${incoming}") 2>/dev/null \
      || ! validate_private_state_file_metadata "${incoming}" 4096 0; then
    rm -f -- "${incoming}"
    echo "[ERROR] Failed to construct private legacy active-state migration." >&2
    return 2
  fi
  mv -T -- "${incoming}" "${state}"
  load_active_state_v2_exact "${state}" 0
}

# Compare canonical positive decimal epochs without shell arithmetic.  Epochs
# are persisted external metadata and may exceed the machine integer width;
# length followed by lexical order is exact for the validated representation.
positive_decimal_is_strictly_older() {
  local candidate="$1" reference="$2"
  local LC_ALL=C
  [[ "${candidate}" =~ ^[1-9][0-9]*$ \
      && "${reference}" =~ ^[1-9][0-9]*$ ]] || return 2
  if (( ${#candidate} < ${#reference} )); then
    return 0
  fi
  if (( ${#candidate} > ${#reference} )); then
    return 1
  fi
  [[ "${candidate}" < "${reference}" ]]
}

active_state_has_session_namespace() {
  local expected_session="$1"
  [[ "${active_log_dir}" == "logs/batch_ne/${expected_session}_"* ]]
}
EOF
}

tmux_session_query_helpers() {
  cat <<'EOF'
query_tmux_session_presence() {
  local session_name="$1" query_rc
  if tmux has-session -t "${session_name}" 2>/dev/null; then
    tmux_session_present=1
    return 0
  else
    query_rc=$?
  fi
  if [[ "${query_rc}" == 1 ]]; then
    tmux_session_present=0
    return 0
  fi
  echo "[ERROR] tmux session-presence query failed for ${session_name} (rc=${query_rc})." >&2
  return 2
}
EOF
}

launch_process_closure_helpers() {
  # A tmux server disappearing does not prove that torchrun/Python descendants
  # exited.  All payload descendants inherit the atomic launch identity, so a
  # bounded /proc scan can target this launch without touching unrelated jobs.
  cat <<'EOF'
process_has_launch_identity() {
  local pid="$1" expected_token="$2" expected_command_sha="$3" expected_epoch="$4"
  local entry token_fields=0 token_matches=0 command_fields=0 command_value=""
  local epoch_fields=0 epoch_matches=0
  [[ "${pid}" =~ ^[1-9][0-9]*$ && "${pid}" != "$$" && -r "/proc/${pid}/environ" ]] || return 1
  while IFS= read -r -d '' entry; do
    case "${entry}" in
      HOLOSOMA_LAUNCH_TOKEN=*)
        token_fields=$((token_fields + 1))
        [[ "${entry#*=}" == "${expected_token}" ]] \
          && token_matches=$((token_matches + 1))
        ;;
      HOLOSOMA_COMMAND_SHA256=*)
        command_fields=$((command_fields + 1))
        command_value=${entry#*=}
        ;;
      HOLOSOMA_LAUNCH_EPOCH=*)
        epoch_fields=$((epoch_fields + 1))
        [[ "${entry#*=}" == "${expected_epoch}" ]] \
          && epoch_matches=$((epoch_matches + 1))
        ;;
    esac
    # A non-matching identity field is data to validate below, not a failed
    # /proc read.  Without this sentinel, the while loop inherits a failed
    # comparison from its final entry and an appended OR-return misclassifies a
    # same-token process with a wrong epoch as unrelated.
    :
  done 2>/dev/null < "/proc/${pid}/environ" || return 1
  (( token_matches != 0 )) || return 1
  if (( token_fields != 1 || token_matches != 1 \
        || epoch_fields != 1 || epoch_matches != 1 \
        || command_fields != 1 )); then
    echo "[ERROR] Process ${pid} has conflicting/duplicate exact launch identity for token=${expected_token}." >&2
    return 2
  fi
  if [[ "${expected_command_sha}" == pending ]]; then
    if [[ ! "${command_value}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "[ERROR] Process ${pid} has malformed command identity for token=${expected_token}." >&2
      return 2
    fi
  elif [[ "${command_value}" != "${expected_command_sha}" ]]; then
    echo "[ERROR] Process ${pid} command identity conflicts with token=${expected_token}." >&2
    return 2
  fi
  return 0
}

collect_launch_identity_pids() {
  local expected_token="$1" expected_command_sha="$2" expected_epoch="$3"
  local proc pid match_rc
  launch_identity_pids=()
  for proc in /proc/[0-9]*; do
    pid=${proc##*/}
    if process_has_launch_identity \
        "${pid}" "${expected_token}" "${expected_command_sha}" "${expected_epoch}"; then
      launch_identity_pids+=("${pid}")
    else
      match_rc=$?
      (( match_rc == 1 )) || return "${match_rc}"
    fi
  done
}

verify_no_launch_identity_processes() {
  local expected_token="$1" expected_command_sha="$2" expected_epoch="$3"
  collect_launch_identity_pids \
    "${expected_token}" "${expected_command_sha}" "${expected_epoch}" || return
  if (( ${#launch_identity_pids[@]} != 0 )); then
    echo "[ERROR] Exact launch-identity processes remain: ${launch_identity_pids[*]}" >&2
    return 2
  fi
}

# Intent publication does not know the eventual command digest.  For its
# cancellation closure, bind every process by token+epoch and require the
# inherited command field itself to be a single canonical SHA256.  A malformed
# matching environment is corruption, not evidence of absence.
process_matches_launch_token_epoch() {
  local pid="$1" expected_token="$2" expected_epoch="$3"
  local entry token_fields=0 token_matches=0 command_fields=0
  local epoch_fields=0 epoch_matches=0 command_value=""
  [[ "${pid}" =~ ^[1-9][0-9]*$ && "${pid}" != "$$" \
      && -r "/proc/${pid}/environ" ]] || return 1
  while IFS= read -r -d '' entry; do
    case "${entry}" in
      HOLOSOMA_LAUNCH_TOKEN=*)
        token_fields=$((token_fields + 1))
        [[ "${entry#*=}" == "${expected_token}" ]] \
          && token_matches=$((token_matches + 1))
        ;;
      HOLOSOMA_COMMAND_SHA256=*)
        command_fields=$((command_fields + 1))
        command_value=${entry#*=}
        ;;
      HOLOSOMA_LAUNCH_EPOCH=*)
        epoch_fields=$((epoch_fields + 1))
        [[ "${entry#*=}" == "${expected_epoch}" ]] \
          && epoch_matches=$((epoch_matches + 1))
        ;;
    esac
    # Keep the loop status tied to opening/reading /proc, never to the value
    # comparison performed for the final environment entry.
    :
  done 2>/dev/null < "/proc/${pid}/environ" || return 1
  # The 256-bit token is the ownership anchor.  Once this token appears in a
  # process environment, a missing/wrong/duplicate epoch or command field is
  # corruption of this launch identity, never evidence that the process is
  # unrelated.  Return "not ours" only when the token itself is absent.
  if (( token_matches == 0 )); then
    return 1
  fi
  if (( token_fields != 1 || token_matches != 1 \
        || epoch_fields != 1 || epoch_matches != 1 \
        || command_fields != 1 )) \
      || [[ ! "${command_value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[ERROR] Process ${pid} has malformed environment for launch token=${expected_token} epoch=${expected_epoch}." >&2
    return 2
  fi
  return 0
}

collect_launch_token_epoch_pids() {
  local expected_token="$1" expected_epoch="$2" proc pid match_rc
  launch_token_epoch_pids=()
  for proc in /proc/[0-9]*; do
    pid=${proc##*/}
    if process_matches_launch_token_epoch \
        "${pid}" "${expected_token}" "${expected_epoch}"; then
      launch_token_epoch_pids+=("${pid}")
    else
      match_rc=$?
      (( match_rc == 1 )) || return "${match_rc}"
    fi
  done
}

verify_no_launch_token_epoch_processes() {
  local expected_token="$1" expected_epoch="$2"
  collect_launch_token_epoch_pids "${expected_token}" "${expected_epoch}" || return
  if (( ${#launch_token_epoch_pids[@]} != 0 )); then
    echo "[ERROR] Launch token+epoch processes remain: ${launch_token_epoch_pids[*]}" >&2
    return 2
  fi
}

terminate_launch_identity_processes_bounded() {
  local expected_token="$1" expected_command_sha="$2" expected_epoch="$3"
  local attempt
  collect_launch_identity_pids \
    "${expected_token}" "${expected_command_sha}" "${expected_epoch}" || return
  if (( ${#launch_identity_pids[@]} != 0 )); then
    kill -TERM -- "${launch_identity_pids[@]}" 2>/dev/null || true
  fi
  for ((attempt = 0; attempt < 20; attempt++)); do
    collect_launch_identity_pids \
      "${expected_token}" "${expected_command_sha}" "${expected_epoch}" || return
    (( ${#launch_identity_pids[@]} != 0 )) || return 0
    sleep 0.1
  done
  kill -KILL -- "${launch_identity_pids[@]}" 2>/dev/null || true
  for ((attempt = 0; attempt < 20; attempt++)); do
    collect_launch_identity_pids \
      "${expected_token}" "${expected_command_sha}" "${expected_epoch}" || return
    (( ${#launch_identity_pids[@]} != 0 )) || return 0
    sleep 0.1
  done
  echo "[ERROR] Exact launch-identity processes survived bounded TERM/KILL cleanup: ${launch_identity_pids[*]}" >&2
  return 2
}
EOF
}

legacy_stop_process_helpers() {
  # Launches produced before atomic tmux environment binding did not export the
  # launch identity into torchrun descendants.  Their only process ownership
  # root is the exact, option-bound tmux pane.  Keep this compatibility logic
  # separate from the modern environment-identity cleanup path.
  cat <<'EOF'
legacy_read_proc_identity() {
  local pid="$1" stat_record stat_tail ignored status_record status_key status_uid status_rest
  local uid_fields=0 verify_stat_record verify_stat_tail verify_start
  local -a stat_records=() status_records=() verify_stat_records=() verify_stat_fields=()
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 2
  if [[ ! -d "/proc/${pid}" ]]; then
    return 1
  fi
  # Buffer each dynamic procfs record before parsing it.  Besides avoiding a
  # torn record, this keeps the exact-identity recheck cheap during full /proc
  # closure scans (no fork and no byte-at-a-time shell read).
  if ! mapfile -t -n 1 stat_records 2>/dev/null < "/proc/${pid}/stat" \
      || (( ${#stat_records[@]} != 1 )); then
    [[ -d "/proc/${pid}" ]] || return 1
    echo "[ERROR] Could not read process identity for live PID ${pid}." >&2
    return 2
  fi
  stat_record=${stat_records[0]}
  # /proc/PID/stat field 2 is parenthesized and may itself contain spaces or
  # right parentheses.  Removing through the final ') ' leaves fields 3+.
  stat_tail=${stat_record##*) }
  legacy_proc_state="" legacy_proc_ppid="" legacy_proc_pgrp=""
  legacy_proc_session="" legacy_proc_start=""
  IFS=' ' read -r \
    legacy_proc_state legacy_proc_ppid legacy_proc_pgrp legacy_proc_session \
    ignored ignored ignored ignored ignored ignored ignored ignored ignored \
    ignored ignored ignored ignored ignored ignored legacy_proc_start ignored \
    <<<"${stat_tail}" || true
  if [[ ! "${legacy_proc_state}" =~ ^[A-Za-z]$ \
        || ! "${legacy_proc_ppid}" =~ ^[0-9]+$ \
        || ! "${legacy_proc_pgrp}" =~ ^[0-9]+$ \
        || ! "${legacy_proc_session}" =~ ^[0-9]+$ \
        || ! "${legacy_proc_start}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] Malformed /proc identity for PID ${pid}." >&2
    return 2
  fi
  # Keep UID verification on every captured identity without forking one
  # external stat process per PID per closure round.  On busy training nodes
  # that fork amplification can consume the entire bounded cleanup window.
  legacy_proc_uid=""
  if [[ ! -r "/proc/${pid}/status" ]]; then
    [[ -d "/proc/${pid}" ]] || return 1
    echo "[ERROR] Could not read process owner for live PID ${pid}." >&2
    return 2
  fi
  # Bash read consumes this dynamic procfs file through repeated small reads.  A
  # task state change can then shift the regenerated contents between reads and
  # tear a field boundary (observed as a split/missing Uid field even for a stable
  # kernel thread).  mapfile buffers the procfs snapshot before parsing it.
  if ! mapfile -t status_records 2>/dev/null < "/proc/${pid}/status"; then
    [[ -d "/proc/${pid}" ]] || return 1
    echo "[ERROR] Could not read process owner for live PID ${pid}." >&2
    return 2
  fi
  for status_record in "${status_records[@]}"; do
    [[ "${status_record}" == Uid:* ]] || continue
    status_key="" status_uid="" status_rest=""
    IFS=$'\t ' read -r status_key status_uid status_rest \
      <<<"${status_record}" || true
    if [[ "${status_key}" == Uid: ]]; then
      uid_fields=$((uid_fields + 1))
      legacy_proc_uid=${status_uid}
    fi
  done
  # Bind the buffered owner record back to the same PID generation.  A process
  # which exited or was reused while the two procfs records were read is simply
  # absent from this snapshot; malformed data from a stable live identity is a
  # hard error.
  if ! mapfile -t -n 1 verify_stat_records 2>/dev/null < "/proc/${pid}/stat" \
      || (( ${#verify_stat_records[@]} != 1 )); then
    [[ -d "/proc/${pid}" ]] || return 1
    echo "[ERROR] Could not re-read process identity for live PID ${pid}." >&2
    return 2
  fi
  verify_stat_record=${verify_stat_records[0]}
  verify_stat_tail=${verify_stat_record##*) }
  IFS=' ' read -r -a verify_stat_fields <<<"${verify_stat_tail}" || true
  verify_start=${verify_stat_fields[19]:-}
  if [[ ! "${verify_start}" =~ ^[1-9][0-9]*$ ]]; then
    [[ -d "/proc/${pid}" ]] || return 1
    echo "[ERROR] Malformed re-read /proc identity for PID ${pid}." >&2
    return 2
  fi
  [[ "${verify_start}" == "${legacy_proc_start}" ]] || return 1
  if (( uid_fields != 1 )) || [[ ! "${legacy_proc_uid}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Malformed process owner for live PID ${pid}." >&2
    return 2
  fi
}

legacy_pid_matches_start() {
  local pid="$1" expected_start="$2" identity_rc
  if legacy_read_proc_identity "${pid}"; then
    [[ "${legacy_proc_start}" == "${expected_start}" ]]
    return
  else
    identity_rc=$?
  fi
  (( identity_rc == 1 )) && return 1
  return "${identity_rc}"
}

legacy_validate_pane_process() {
  local pane_pid="$1" expected_control="$2" identity_rc bash_path pane_exe arg
  local current_uid
  legacy_read_proc_identity "${pane_pid}" || return
  current_uid=$(id -u) || return 2
  if [[ ! "${current_uid}" =~ ^[0-9]+$ \
        || "${legacy_proc_uid}" != "${current_uid}" ]]; then
    echo "[ERROR] Legacy tmux pane PID ${pane_pid} is not owned by the current user." >&2
    return 2
  fi
  if [[ "${legacy_proc_pgrp}" != "${pane_pid}" \
        || "${legacy_proc_session}" != "${pane_pid}" ]]; then
    echo "[ERROR] Legacy tmux pane PID ${pane_pid} is not its exact process-group/session leader." >&2
    return 2
  fi
  legacy_pane_start=${legacy_proc_start}
  legacy_pane_argv=()
  while IFS= read -r -d '' arg; do
    legacy_pane_argv+=("${arg}")
  done 2>/dev/null < "/proc/${pane_pid}/cmdline" || true
  if (( ${#legacy_pane_argv[@]} != 2 )) \
      || [[ "${legacy_pane_argv[0]##*/}" != bash \
        || "${legacy_pane_argv[1]}" != "${expected_control}" ]]; then
    echo "[ERROR] Legacy tmux pane PID ${pane_pid} is not exactly 'bash <hash-bound-control-script>'." >&2
    return 2
  fi
  bash_path=$(command -v bash 2>/dev/null || true)
  pane_exe=$(readlink -f "/proc/${pane_pid}/exe" 2>/dev/null || true)
  if [[ -z "${bash_path}" || -z "${pane_exe}" \
        || "${pane_exe}" != "$(readlink -f "${bash_path}")" ]]; then
    echo "[ERROR] Legacy tmux pane PID ${pane_pid} does not execute the expected Bash binary." >&2
    return 2
  fi
}

legacy_read_proc_cgroup_v2_exact() {
  local pid="$1" expected_start="$2" identity_rc record path
  local -a records=()
  legacy_proc_cgroup_path=""
  legacy_pid_matches_start "${pid}" "${expected_start}" || return
  if ! mapfile -t records 2>/dev/null < "/proc/${pid}/cgroup"; then
    if legacy_pid_matches_start "${pid}" "${expected_start}"; then
      echo "[ERROR] Could not read cgroup identity for live legacy PID ${pid}." >&2
      return 2
    else
      identity_rc=$?
      (( identity_rc == 1 )) && return 1
      return "${identity_rc}"
    fi
  fi
  if (( ${#records[@]} != 1 )) || [[ ! "${records[0]}" =~ ^0::/ ]]; then
    echo "[ERROR] Legacy PID ${pid} is not in one canonical unified cgroup-v2 hierarchy." >&2
    return 2
  fi
  record=${records[0]}
  path=${record#0::}
  if (( ${#path} > 1024 )) \
      || [[ ! "${path}" =~ ^/[A-Za-z0-9_.:@+/-]+$ \
        || "${path}" == / || "${path}" == */ \
        || "${path}" == *//* || "${path}" == */./* \
        || "${path}" == */../* || "${path}" == */. || "${path}" == */.. ]]; then
    echo "[ERROR] Legacy PID ${pid} has an unsafe or non-canonical cgroup-v2 path." >&2
    return 2
  fi
  # Bind the dynamic cgroup record back to the same PID generation and UID.
  legacy_pid_matches_start "${pid}" "${expected_start}" || return
  legacy_proc_cgroup_path=${path}
}

legacy_validate_leaf_cgroup_v2() {
  local cgroup_path="$1" expected_dev="${2:-}" expected_ino="${3:-}"
  local cgroup_dir resolved current_uid fs_magic child_dir file file_uid
  local cgroup_basename expected_cgroup_path cgroup_type record key value extra
  local descendant_count="" dying_descendant_count=""
  local descendant_count_seen=0 dying_descendant_count_seen=0
  local -a type_records=() stat_records=()
  current_uid=$(id -u) || return 2
  cgroup_dir="/sys/fs/cgroup${cgroup_path}"
  fs_magic=$(stat -f -c %t -- /sys/fs/cgroup 2>/dev/null || true)
  resolved=$(realpath -e -- "${cgroup_dir}" 2>/dev/null || true)
  if [[ "${fs_magic}" != 63677270 || -z "${resolved}" \
        || "${resolved}" != "${cgroup_dir}" || ! -d "${cgroup_dir}" \
        || -L "${cgroup_dir}" ]]; then
    echo "[ERROR] Legacy pane cgroup is not one exact cgroup-v2 directory: ${cgroup_path}." >&2
    return 2
  fi
  cgroup_basename=${cgroup_path##*/}
  if [[ ! "${cgroup_basename}" =~ ^tmux-spawn-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.scope$ ]]; then
    echo "[ERROR] Legacy pane is not contained by one exact tmux-spawn scope." >&2
    return 2
  fi
  expected_cgroup_path="/user.slice/user-${current_uid}.slice/user@${current_uid}.service/${cgroup_basename}"
  if [[ "${cgroup_path}" != "${expected_cgroup_path}" ]]; then
    echo "[ERROR] Legacy pane tmux-spawn scope is outside the exact current-user service boundary." >&2
    return 2
  fi
  legacy_cgroup_dev=$(stat -c %d -- "${cgroup_dir}") || return 2
  legacy_cgroup_ino=$(stat -c %i -- "${cgroup_dir}") || return 2
  legacy_cgroup_uid=$(stat -c %u -- "${cgroup_dir}") || return 2
  if [[ ! "${legacy_cgroup_dev}" =~ ^[1-9][0-9]*$ \
        || ! "${legacy_cgroup_ino}" =~ ^[1-9][0-9]*$ \
        || "${legacy_cgroup_uid}" != "${current_uid}" \
        || ( -n "${expected_dev}" && "${legacy_cgroup_dev}" != "${expected_dev}" ) \
        || ( -n "${expected_ino}" && "${legacy_cgroup_ino}" != "${expected_ino}" ) ]]; then
    echo "[ERROR] Legacy pane cgroup identity/owner changed or is unsafe: ${cgroup_path}." >&2
    return 2
  fi
  for file in cgroup.procs cgroup.events cgroup.freeze cgroup.kill cgroup.type cgroup.stat; do
    if [[ ! -f "${cgroup_dir}/${file}" || -L "${cgroup_dir}/${file}" ]]; then
      echo "[ERROR] Legacy pane cgroup lacks exact ${file} support." >&2
      return 2
    fi
    file_uid=$(stat -c %u -- "${cgroup_dir}/${file}") || return 2
    if [[ "${file_uid}" != "${current_uid}" ]]; then
      echo "[ERROR] Legacy pane cgroup ${file} is not owned by current UID ${current_uid}." >&2
      return 2
    fi
  done
  if [[ ! -r "${cgroup_dir}/cgroup.procs" \
        || ! -r "${cgroup_dir}/cgroup.events" \
        || ! -r "${cgroup_dir}/cgroup.freeze" \
        || ! -w "${cgroup_dir}/cgroup.freeze" \
        || ! -w "${cgroup_dir}/cgroup.kill" ]]; then
    echo "[ERROR] Legacy pane cgroup freezer/kill boundary is not readable and writable by the launch owner." >&2
    return 2
  fi
  if ! mapfile -t type_records < "${cgroup_dir}/cgroup.type" 2>/dev/null \
      || (( ${#type_records[@]} != 1 )) \
      || [[ "${type_records[0]}" != domain ]]; then
    echo "[ERROR] Legacy pane tmux-spawn cgroup must be one domain cgroup." >&2
    return 2
  fi
  cgroup_type=${type_records[0]}
  if ! mapfile -t stat_records < "${cgroup_dir}/cgroup.stat" 2>/dev/null; then
    echo "[ERROR] Could not read legacy pane cgroup descendant statistics." >&2
    return 2
  fi
  for record in "${stat_records[@]}"; do
    key="" value="" extra=""
    read -r key value extra <<<"${record}" || true
    case "${key}" in
      nr_descendants)
        descendant_count_seen=$((descendant_count_seen + 1))
        descendant_count=${value}
        ;;
      nr_dying_descendants)
        dying_descendant_count_seen=$((dying_descendant_count_seen + 1))
        dying_descendant_count=${value}
        ;;
    esac
  done
  if (( descendant_count_seen != 1 || dying_descendant_count_seen != 1 )) \
      || [[ "${descendant_count}" != 0 || "${dying_descendant_count}" != 0 ]]; then
    echo "[ERROR] Legacy pane cgroup must have zero live and dying descendant cgroups." >&2
    return 2
  fi
  child_dir=$(find "${cgroup_dir}" -mindepth 1 -type d -print -quit 2>/dev/null || true)
  if [[ -n "${child_dir}" ]]; then
    echo "[ERROR] Legacy pane cgroup must be a leaf; nested cgroups are outside the exact receipt boundary." >&2
    return 2
  fi
  legacy_cgroup_dir=${cgroup_dir}
  legacy_cgroup_type=${cgroup_type}
}

legacy_open_exact_cgroup_fd() {
  local cgroup_path="$1" expected_dev="${2:-}" expected_ino="${3:-}"
  local fd_path fd_dev fd_ino
  legacy_open_cgroup_fd=""
  legacy_open_cgroup_fd_path=""
  legacy_validate_leaf_cgroup_v2 \
    "${cgroup_path}" "${expected_dev}" "${expected_ino}" || return
  exec {legacy_open_cgroup_fd}< "${legacy_cgroup_dir}" || {
    echo "[ERROR] Could not open the exact legacy cgroup directory." >&2
    return 2
  }
  fd_path="/proc/self/fd/${legacy_open_cgroup_fd}"
  fd_dev=$(stat -Lc %d -- "${fd_path}") || {
    exec {legacy_open_cgroup_fd}<&-
    legacy_open_cgroup_fd=""
    return 2
  }
  fd_ino=$(stat -Lc %i -- "${fd_path}") || {
    exec {legacy_open_cgroup_fd}<&-
    legacy_open_cgroup_fd=""
    return 2
  }
  if [[ "${fd_dev}" != "${legacy_cgroup_dev}" \
        || "${fd_ino}" != "${legacy_cgroup_ino}" \
        || ( -n "${expected_dev}" && "${fd_dev}" != "${expected_dev}" ) \
        || ( -n "${expected_ino}" && "${fd_ino}" != "${expected_ino}" ) ]]; then
    exec {legacy_open_cgroup_fd}<&-
    legacy_open_cgroup_fd=""
    echo "[ERROR] Open legacy cgroup directory does not match its authenticated identity." >&2
    return 2
  fi
  legacy_open_cgroup_fd_path=${fd_path}
}

legacy_close_exact_cgroup_fd() {
  if [[ "${legacy_open_cgroup_fd:-}" =~ ^[0-9]+$ ]]; then
    exec {legacy_open_cgroup_fd}<&- || return 2
  fi
  legacy_open_cgroup_fd=""
  legacy_open_cgroup_fd_path=""
}

legacy_read_cgroup_frozen_state_fd() {
  local fd_path="${legacy_open_cgroup_fd_path:-}"
  local record key value extra frozen_count=0 populated_count=0
  local current_dev current_ino
  local -a freeze_records=() event_records=()
  if [[ -z "${fd_path}" \
        || ! "${legacy_open_cgroup_fd:-}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Legacy cgroup state read lacks an authenticated open directory." >&2
    return 2
  fi
  current_dev=$(stat -Lc %d -- "${fd_path}") || return 2
  current_ino=$(stat -Lc %i -- "${fd_path}") || return 2
  if [[ "${current_dev}" != "${legacy_cgroup_dev}" \
        || "${current_ino}" != "${legacy_cgroup_ino}" \
        || ! -f "${fd_path}/cgroup.freeze" \
        || -L "${fd_path}/cgroup.freeze" \
        || ! -f "${fd_path}/cgroup.events" \
        || -L "${fd_path}/cgroup.events" ]]; then
    echo "[ERROR] Open legacy cgroup state files changed after authentication." >&2
    return 2
  fi
  if ! mapfile -t freeze_records < "${fd_path}/cgroup.freeze" 2>/dev/null \
      || (( ${#freeze_records[@]} != 1 )) \
      || [[ ! "${freeze_records[0]}" =~ ^[01]$ ]]; then
    echo "[ERROR] Legacy pane cgroup returned a malformed freezer state." >&2
    return 2
  fi
  if ! mapfile -t event_records < "${fd_path}/cgroup.events" 2>/dev/null; then
    echo "[ERROR] Could not read legacy pane cgroup events." >&2
    return 2
  fi
  legacy_cgroup_event_frozen=""
  legacy_cgroup_populated=""
  for record in "${event_records[@]}"; do
    key="" value="" extra=""
    read -r key value extra <<<"${record}" || true
    if [[ "${key}" == frozen ]]; then
      frozen_count=$((frozen_count + 1))
      legacy_cgroup_event_frozen=${value}
    elif [[ "${key}" == populated ]]; then
      populated_count=$((populated_count + 1))
      legacy_cgroup_populated=${value}
    fi
  done
  if (( frozen_count != 1 || populated_count != 1 )) \
      || [[ ! "${legacy_cgroup_event_frozen}" =~ ^[01]$ \
        || ! "${legacy_cgroup_populated}" =~ ^[01]$ ]]; then
    echo "[ERROR] Legacy pane cgroup returned malformed effective freezer events." >&2
    return 2
  fi
  legacy_cgroup_freeze_requested=${freeze_records[0]}
  legacy_cgroup_frozen=${legacy_cgroup_event_frozen}
}

legacy_read_cgroup_frozen_state() {
  local cgroup_path="$1" expected_dev="$2" expected_ino="$3"
  local rc=0 close_rc
  legacy_open_exact_cgroup_fd \
    "${cgroup_path}" "${expected_dev}" "${expected_ino}" || return
  legacy_read_cgroup_frozen_state_fd || rc=$?
  legacy_close_exact_cgroup_fd || {
    close_rc=$?
    (( rc != 0 )) || rc=${close_rc}
  }
  return "${rc}"
}

legacy_set_cgroup_frozen_exact() {
  local cgroup_path="$1" expected_dev="$2" expected_ino="$3" desired="$4"
  local attempt rc=0
  [[ "${desired}" =~ ^[01]$ ]] || return 2
  legacy_open_exact_cgroup_fd \
    "${cgroup_path}" "${expected_dev}" "${expected_ino}" || return
  if ! printf '%s\n' "${desired}" > "${legacy_open_cgroup_fd_path}/cgroup.freeze"; then
    echo "[ERROR] Could not set exact legacy pane cgroup freezer=${desired}." >&2
    legacy_close_exact_cgroup_fd || true
    return 2
  fi
  for ((attempt = 0; attempt < 100; attempt++)); do
    if legacy_read_cgroup_frozen_state_fd; then
      :
    else
      rc=$?
      legacy_close_exact_cgroup_fd || true
      return "${rc}"
    fi
    if [[ "${legacy_cgroup_freeze_requested}" != "${desired}" ]]; then
      echo "[ERROR] Legacy pane cgroup freezer request changed during its bounded transition." >&2
      legacy_close_exact_cgroup_fd || true
      return 2
    fi
    if [[ "${legacy_cgroup_frozen}" == "${desired}" ]]; then
      legacy_close_exact_cgroup_fd
      return
    fi
    sleep 0.05
  done
  legacy_close_exact_cgroup_fd || true
  echo "[ERROR] Legacy pane cgroup did not reach freezer=${desired} within the bounded wait." >&2
  return 2
}

legacy_load_exact_pane_cgroup() {
  local pane_pid="$1" pane_start="$2" identity_rc self_start self_cgroup
  legacy_read_proc_cgroup_v2_exact "${pane_pid}" "${pane_start}" || return
  legacy_pane_cgroup_path=${legacy_proc_cgroup_path}
  legacy_validate_leaf_cgroup_v2 "${legacy_pane_cgroup_path}" || return
  legacy_pane_cgroup_dev=${legacy_cgroup_dev}
  legacy_pane_cgroup_ino=${legacy_cgroup_ino}
  legacy_read_proc_identity "$$" || return 2
  self_start=${legacy_proc_start}
  legacy_read_proc_cgroup_v2_exact "$$" "${self_start}" || return
  self_cgroup=${legacy_proc_cgroup_path}
  if [[ "${self_cgroup}" == "${legacy_pane_cgroup_path}" ]]; then
    echo "[ERROR] Refusing to freeze a legacy pane cgroup which contains the stop controller." >&2
    return 2
  fi
  legacy_read_cgroup_frozen_state \
    "${legacy_pane_cgroup_path}" "${legacy_pane_cgroup_dev}" \
    "${legacy_pane_cgroup_ino}" || return
  legacy_pane_cgroup_freeze_requested=${legacy_cgroup_freeze_requested}
  legacy_pane_cgroup_frozen=${legacy_cgroup_frozen}
  # Re-bind the pane after every cgroup capability/identity read.
  legacy_pid_matches_start "${pane_pid}" "${pane_start}" || {
    identity_rc=$?
    (( identity_rc == 1 )) \
      && echo "[ERROR] Exact legacy pane disappeared during cgroup preflight." >&2
    return 2
  }
  legacy_read_proc_cgroup_v2_exact "${pane_pid}" "${pane_start}" || return
  [[ "${legacy_proc_cgroup_path}" == "${legacy_pane_cgroup_path}" ]] || {
    echo "[ERROR] Exact legacy pane changed cgroup during preflight." >&2
    return 2
  }
  legacy_pane_cgroup_fingerprint=$(
    printf '%s\0%s\0%s\0%s' \
      "${legacy_pane_cgroup_path}" "${legacy_pane_cgroup_dev}" \
      "${legacy_pane_cgroup_ino}" "${legacy_cgroup_uid}" \
      | sha256sum | awk '{print $1}'
  ) || return 2
  [[ "${legacy_pane_cgroup_fingerprint}" =~ ^[0-9a-f]{64}$ ]] || return 2
}

legacy_load_exact_single_pane() {
  local expected_session="$1" expected_control="$2" panes line_count field_count
  local pane_session pane_pid pane_dead extra
  if ! panes=$(tmux list-panes -t "${expected_session}" -s \
      -F $'#{session_name}\t#{pane_pid}\t#{pane_dead}' 2>/dev/null); then
    echo "[ERROR] Could not enumerate every pane in legacy tmux session ${expected_session}." >&2
    return 2
  fi
  line_count=$(awk 'END { print NR }' <<<"${panes}") || return 2
  field_count=$(awk -F '\t' 'NR == 1 { print NF }' <<<"${panes}") || return 2
  if [[ "${line_count}" != 1 || "${field_count}" != 3 ]]; then
    echo "[ERROR] Legacy stop requires exactly one pane in the exact tmux session." >&2
    return 2
  fi
  pane_session="" pane_pid="" pane_dead="" extra=""
  IFS=$'\t' read -r pane_session pane_pid pane_dead extra <<<"${panes}" || true
  if [[ "${pane_session}" != "${expected_session}" \
        || ! "${pane_pid}" =~ ^[1-9][0-9]*$ \
        || "${pane_dead}" != 0 || -n "${extra}" ]]; then
    echo "[ERROR] Legacy tmux pane record is malformed or belongs to another session." >&2
    return 2
  fi
  legacy_validate_pane_process "${pane_pid}" "${expected_control}" || return
  legacy_pane_pid=${pane_pid}
  legacy_load_exact_pane_cgroup "${pane_pid}" "${legacy_pane_start}" || return
}

legacy_capture_receipt_path() {
  local active_state="$1" token="$2" command_sha="$3" epoch="$4"
  local receipt_identity_sha
  receipt_identity_sha=$(printf 'legacy-process-v2-cgroup\t%s\t%s\t%s' \
    "${token}" "${command_sha}" "${epoch}" | sha256sum | awk '{print $1}') || return 2
  [[ "${receipt_identity_sha}" =~ ^[0-9a-f]{64}$ ]] || return 2
  # Node identifiers are bounded to 80 bytes above.  Hashing the three long
  # ownership fields keeps the receipt plus its durable intent/.in residue
  # below NAME_MAX while the bodies still validate every original field.
  printf '%s.legacy-processes.%s\n' "${active_state}" "${receipt_identity_sha}"
}

legacy_load_capture_receipt() {
  local receipt="$1" expected_token="$2" expected_epoch="$3"
  local expected_command="$4" expected_snapshot="$5" expected_log_dir="$6"
  local expected_target="$7" size mode owner link_count line_count header_fields
  local expected_line_count
  local version token epoch command_sha snapshot log_dir target root_pid root_start
  local cgroup_path cgroup_dev cgroup_ino count extra current_uid
  local pid start uid ppid pgrp session previous_pid=0 actual_count=0 root_seen=0
  if [[ ! -f "${receipt}" || -L "${receipt}" ]]; then
    echo "[ERROR] Legacy process-capture receipt is missing, non-regular, or symlinked: ${receipt}" >&2
    return 2
  fi
  size=$(stat -c %s "${receipt}") || return 2
  mode=$(stat -c %a "${receipt}") || return 2
  owner=$(stat -c %u "${receipt}") || return 2
  link_count=$(stat -c %h "${receipt}") || return 2
  if [[ ! "${size}" =~ ^[1-9][0-9]*$ || ${#size} -gt 7 \
        || "${mode}" != 400 || "${owner}" != "$(id -u)" \
        || "${link_count}" != 1 ]]; then
    echo "[ERROR] Legacy process-capture receipt has unsafe size/mode/owner/link-count: ${receipt}" >&2
    return 2
  fi
  line_count=$(awk 'END { print NR }' "${receipt}") || return 2
  header_fields=$(awk -F '\t' 'NR == 1 { print NF }' "${receipt}") || return 2
  version="" token="" epoch="" command_sha="" snapshot="" log_dir=""
  target="" root_pid="" root_start="" cgroup_path="" cgroup_dev=""
  cgroup_ino="" count="" extra=""
  IFS=$'\t' read -r version token epoch command_sha snapshot log_dir target \
    root_pid root_start cgroup_path cgroup_dev cgroup_ino count extra \
    < "${receipt}" || true
  if [[ "${header_fields}" != 13 || -n "${extra}" \
        || "${version}" != 2 \
        || "${token}" != "${expected_token}" \
        || "${epoch}" != "${expected_epoch}" \
        || "${command_sha}" != "${expected_command}" \
        || "${snapshot}" != "${expected_snapshot}" \
        || "${log_dir}" != "${expected_log_dir}" \
        || "${target}" != "${expected_target}" \
        || ! "${root_pid}" =~ ^[1-9][0-9]*$ || ${#root_pid} -gt 7 \
        || ! "${root_start}" =~ ^[1-9][0-9]*$ || ${#root_start} -gt 20 \
        || ${#cgroup_path} -gt 1024 \
        || ! "${cgroup_path}" =~ ^/[A-Za-z0-9_.:@+/-]+$ \
        || "${cgroup_path}" == / || "${cgroup_path}" == */ \
        || "${cgroup_path}" == *//* || "${cgroup_path}" == */./* \
        || "${cgroup_path}" == */../* || "${cgroup_path}" == */. \
        || "${cgroup_path}" == */.. \
        || ! "${cgroup_dev}" =~ ^[1-9][0-9]*$ || ${#cgroup_dev} -gt 20 \
        || ! "${cgroup_ino}" =~ ^[1-9][0-9]*$ || ${#cgroup_ino} -gt 20 \
        || ! "${count}" =~ ^[1-9][0-9]*$ || ${#count} -gt 5 ]]; then
    echo "[ERROR] Legacy process-capture receipt header/identity is malformed: ${receipt}" >&2
    return 2
  fi
  if (( 10#${count} > 65536 )); then
    echo "[ERROR] Legacy process-capture receipt exceeds the 65536-process safety bound: ${receipt}" >&2
    return 2
  fi
  expected_line_count=$((10#${count} + 1))
  if [[ "${line_count}" != "${expected_line_count}" ]]; then
    echo "[ERROR] Legacy process-capture receipt line count is inconsistent: ${receipt}" >&2
    return 2
  fi
  legacy_receipt_pids=()
  legacy_receipt_starts=()
  legacy_receipt_uids=()
  legacy_receipt_ppids=()
  legacy_receipt_pgrps=()
  legacy_receipt_sessions=()
  current_uid=$(id -u) || return 2
  while IFS=$'\t' read -r pid start uid ppid pgrp session extra; do
    actual_count=$((actual_count + 1))
    if [[ ! "${pid}" =~ ^[1-9][0-9]*$ || ${#pid} -gt 7 \
          || ! "${start}" =~ ^[1-9][0-9]*$ || ${#start} -gt 20 \
          || ! "${uid}" =~ ^[0-9]+$ || "${uid}" != "${owner}" \
          || "${uid}" != "${current_uid}" \
          || ! "${ppid}" =~ ^[0-9]+$ || ${#ppid} -gt 7 \
          || ! "${pgrp}" =~ ^[1-9][0-9]*$ || ${#pgrp} -gt 7 \
          || ! "${session}" =~ ^[1-9][0-9]*$ || ${#session} -gt 7 \
          || -n "${extra}" ]] \
        || (( pid <= previous_pid )); then
      echo "[ERROR] Legacy process-capture receipt PID records are non-canonical: ${receipt}" >&2
      return 2
    fi
    legacy_receipt_pids+=("${pid}")
    legacy_receipt_starts+=("${start}")
    legacy_receipt_uids+=("${uid}")
    legacy_receipt_ppids+=("${ppid}")
    legacy_receipt_pgrps+=("${pgrp}")
    legacy_receipt_sessions+=("${session}")
    previous_pid=${pid}
    if [[ "${pid}" == "${root_pid}" && "${start}" == "${root_start}" ]]; then
      root_seen=1
    fi
  done < <(tail -n +2 "${receipt}")
  if (( actual_count != count || root_seen != 1 )); then
    echo "[ERROR] Legacy process-capture receipt count/root record is inconsistent: ${receipt}" >&2
    return 2
  fi
  legacy_receipt_root_pid=${root_pid}
  legacy_receipt_root_start=${root_start}
  legacy_receipt_cgroup_path=${cgroup_path}
  legacy_receipt_cgroup_dev=${cgroup_dev}
  legacy_receipt_cgroup_ino=${cgroup_ino}
  legacy_receipt_cgroup_fingerprint=$(
    printf '%s\0%s\0%s\0%s' \
      "${cgroup_path}" "${cgroup_dev}" "${cgroup_ino}" "${owner}" \
      | sha256sum | awk '{print $1}'
  ) || return 2
  [[ "${legacy_receipt_cgroup_fingerprint}" =~ ^[0-9a-f]{64}$ ]] || return 2
}

legacy_collect_frozen_cgroup_members() {
  local root_pid="$1" root_start="$2" cgroup_path="$3"
  local cgroup_dev="$4" cgroup_ino="$5" expected_control="$6"
  local current_uid pid index identity_rc member_count root_seen=0
  local cursor parent depth
  local -a raw_pids=() sorted_pids=()
  local -A seen=() starts=() uids=() ppids=() pgrps=() sessions=()
  current_uid=$(id -u) || return 2
  legacy_read_cgroup_frozen_state \
    "${cgroup_path}" "${cgroup_dev}" "${cgroup_ino}" || return
  if [[ "${legacy_cgroup_freeze_requested}" != 1 \
        || "${legacy_cgroup_frozen}" != 1 ]]; then
    echo "[ERROR] Legacy cgroup membership may be captured only after freezer=1 is effective." >&2
    return 2
  fi
  if ! mapfile -t raw_pids < "${legacy_cgroup_dir}/cgroup.procs" 2>/dev/null; then
    echo "[ERROR] Could not enumerate the frozen legacy tmux-spawn cgroup." >&2
    return 2
  fi
  member_count=${#raw_pids[@]}
  if (( member_count == 0 || member_count > 65536 )); then
    echo "[ERROR] Frozen legacy tmux-spawn cgroup has an invalid process count: ${member_count}." >&2
    return 2
  fi
  for pid in "${raw_pids[@]}"; do
    if [[ ! "${pid}" =~ ^[1-9][0-9]*$ || ${#pid} -gt 7 \
          || -n "${seen[${pid}]+x}" ]]; then
      echo "[ERROR] Frozen legacy tmux-spawn cgroup returned non-canonical process IDs." >&2
      return 2
    fi
    seen[${pid}]=1
  done
  mapfile -t sorted_pids < <(printf '%s\n' "${raw_pids[@]}" | LC_ALL=C sort -n)
  legacy_discovered_pids=()
  legacy_discovered_starts=()
  legacy_discovered_uids=()
  legacy_discovered_ppids=()
  legacy_discovered_pgrps=()
  legacy_discovered_sessions=()
  for pid in "${sorted_pids[@]}"; do
    legacy_read_proc_identity "${pid}" || {
      identity_rc=$?
      echo "[ERROR] Frozen legacy cgroup member PID ${pid} disappeared during capture." >&2
      return "${identity_rc}"
    }
    if [[ "${legacy_proc_uid}" != "${current_uid}" ]]; then
      echo "[ERROR] Frozen legacy cgroup member PID ${pid} is not owned by current UID ${current_uid}." >&2
      return 2
    fi
    legacy_read_proc_cgroup_v2_exact "${pid}" "${legacy_proc_start}" || return
    if [[ "${legacy_proc_cgroup_path}" != "${cgroup_path}" ]]; then
      echo "[ERROR] Frozen legacy cgroup member PID ${pid} changed cgroup during capture." >&2
      return 2
    fi
    legacy_read_proc_identity "${pid}" || return 2
    starts[${pid}]=${legacy_proc_start}
    uids[${pid}]=${legacy_proc_uid}
    ppids[${pid}]=${legacy_proc_ppid}
    pgrps[${pid}]=${legacy_proc_pgrp}
    sessions[${pid}]=${legacy_proc_session}
    legacy_discovered_pids+=("${pid}")
    legacy_discovered_starts+=("${legacy_proc_start}")
    legacy_discovered_uids+=("${legacy_proc_uid}")
    legacy_discovered_ppids+=("${legacy_proc_ppid}")
    legacy_discovered_pgrps+=("${legacy_proc_pgrp}")
    legacy_discovered_sessions+=("${legacy_proc_session}")
    if [[ "${pid}" == "${root_pid}" && "${legacy_proc_start}" == "${root_start}" ]]; then
      root_seen=1
    fi
  done
  (( root_seen == 1 )) || {
    echo "[ERROR] Exact legacy pane root is absent from its frozen tmux-spawn cgroup." >&2
    return 2
  }
  legacy_validate_pane_process "${root_pid}" "${expected_control}" || return
  [[ "${legacy_pane_start}" == "${root_start}" ]] || return 2

  # The r21 compatibility authorization is deliberately narrower than a
  # generic same-UID cgroup cleanup.  Prove that every frozen scope member has
  # an exact captured-parent chain back to this pane generation.  This rejects
  # both a daemon which reparented before the freezer boundary and an unrelated
  # same-UID task explicitly migrated into the scope.  Once effective freeze
  # is reached, the parent generations cannot exit/reuse while these edges are
  # checked; cgroup.kill closes concurrent migration/fork at the commit point.
  for pid in "${sorted_pids[@]}"; do
    [[ "${pid}" != "${root_pid}" ]] || continue
    cursor=${pid}
    depth=0
    while [[ "${cursor}" != "${root_pid}" ]]; do
      parent=${ppids[${cursor}]:-}
      if [[ ! "${parent}" =~ ^[1-9][0-9]*$ \
            || -z "${seen[${parent}]+x}" ]]; then
        echo "[ERROR] Frozen legacy cgroup contains PID ${pid} outside the exact pane PPID closure." >&2
        return 2
      fi
      cursor=${parent}
      depth=$((depth + 1))
      if (( depth > member_count )); then
        echo "[ERROR] Frozen legacy cgroup PPID closure is cyclic or malformed." >&2
        return 2
      fi
    done
  done

  # Re-read every tuple while the cgroup is effectively frozen so neither a
  # stale PPID edge nor PID reuse can enter the durable receipt.
  for index in "${!legacy_discovered_pids[@]}"; do
    pid=${legacy_discovered_pids[${index}]}
    legacy_read_proc_identity "${pid}" || return 2
    if [[ "${legacy_proc_start}" != "${starts[${pid}]}" \
          || "${legacy_proc_uid}" != "${uids[${pid}]}" \
          || "${legacy_proc_ppid}" != "${ppids[${pid}]}" \
          || "${legacy_proc_pgrp}" != "${pgrps[${pid}]}" \
          || "${legacy_proc_session}" != "${sessions[${pid}]}" ]]; then
      echo "[ERROR] Frozen legacy cgroup identity tuple changed during capture: PID ${pid}." >&2
      return 2
    fi
    legacy_read_proc_cgroup_v2_exact "${pid}" "${legacy_proc_start}" || return
    [[ "${legacy_proc_cgroup_path}" == "${cgroup_path}" ]] || return 2
  done
  legacy_read_cgroup_frozen_state \
    "${cgroup_path}" "${cgroup_dev}" "${cgroup_ino}" || return
  [[ "${legacy_cgroup_freeze_requested}" == 1 \
        && "${legacy_cgroup_frozen}" == 1 ]] || return 2
}

legacy_capture_intent_path() {
  printf '%s.freeze-intent\n' "$1"
}

legacy_validate_safe_publication_residue() {
  local path="$1" max_size="$2" size mode owner links current_uid
  current_uid=$(id -u) || return 2
  if [[ ! -f "${path}" || -L "${path}" ]]; then
    echo "[ERROR] Legacy publication residue is not one regular non-symlink file." >&2
    return 2
  fi
  size=$(stat -c %s -- "${path}") || return 2
  mode=$(stat -c %a -- "${path}") || return 2
  owner=$(stat -c %u -- "${path}") || return 2
  links=$(stat -c %h -- "${path}") || return 2
  if [[ ! "${size}" =~ ^[0-9]+$ || ${#size} -gt 8 \
        || ! "${max_size}" =~ ^[1-9][0-9]*$ \
        || "${owner}" != "${current_uid}" || "${links}" != 1 \
        || ( "${mode}" != 400 && "${mode}" != 600 ) ]] \
      || (( 10#${size} > 10#${max_size} )); then
    echo "[ERROR] Legacy publication residue has unsafe size/mode/owner/link-count." >&2
    return 2
  fi
}

legacy_load_capture_intent() {
  local intent="$1" expected_token="$2" expected_epoch="$3"
  local expected_command="$4" expected_snapshot="$5" expected_log_dir="$6"
  local expected_target="$7" size mode owner links fields current_uid
  local version token epoch command snapshot log_dir target root_pid root_start
  local cgroup_path cgroup_dev cgroup_ino uid extra
  if [[ ! -f "${intent}" || -L "${intent}" ]]; then
    echo "[ERROR] Legacy cgroup freeze intent is missing, non-regular, or symlinked." >&2
    return 2
  fi
  size=$(stat -c %s -- "${intent}") || return 2
  mode=$(stat -c %a -- "${intent}") || return 2
  owner=$(stat -c %u -- "${intent}") || return 2
  links=$(stat -c %h -- "${intent}") || return 2
  fields=$(awk -F '\t' 'NR == 1 { print NF } END { if (NR != 1) print "bad" }' \
    "${intent}") || return 2
  current_uid=$(id -u) || return 2
  version="" token="" epoch="" command="" snapshot="" log_dir=""
  target="" root_pid="" root_start="" cgroup_path="" cgroup_dev=""
  cgroup_ino="" uid="" extra=""
  IFS=$'\t' read -r version token epoch command snapshot log_dir target \
    root_pid root_start cgroup_path cgroup_dev cgroup_ino uid extra \
    < "${intent}" || true
  if [[ "${fields}" != 13 || -n "${extra}" \
        || ! "${size}" =~ ^[1-9][0-9]*$ || ${#size} -gt 5 \
        || "${mode}" != 400 || "${owner}" != "${current_uid}" \
        || "${links}" != 1 || "${version}" != 1 \
        || "${token}" != "${expected_token}" \
        || "${epoch}" != "${expected_epoch}" \
        || "${command}" != "${expected_command}" \
        || "${snapshot}" != "${expected_snapshot}" \
        || "${log_dir}" != "${expected_log_dir}" \
        || "${target}" != "${expected_target}" \
        || ! "${root_pid}" =~ ^[1-9][0-9]*$ || ${#root_pid} -gt 7 \
        || ! "${root_start}" =~ ^[1-9][0-9]*$ || ${#root_start} -gt 20 \
        || ${#cgroup_path} -gt 1024 \
        || ! "${cgroup_path}" =~ ^/[A-Za-z0-9_.:@+/-]+$ \
        || "${cgroup_path}" == / || "${cgroup_path}" == */ \
        || "${cgroup_path}" == *//* || "${cgroup_path}" == */./* \
        || "${cgroup_path}" == */../* || "${cgroup_path}" == */. \
        || "${cgroup_path}" == */.. \
        || ! "${cgroup_dev}" =~ ^[1-9][0-9]*$ || ${#cgroup_dev} -gt 20 \
        || ! "${cgroup_ino}" =~ ^[1-9][0-9]*$ || ${#cgroup_ino} -gt 20 \
        || "${uid}" != "${current_uid}" ]]; then
    echo "[ERROR] Legacy cgroup freeze intent is malformed or identity-mismatched." >&2
    return 2
  fi
  legacy_intent_root_pid=${root_pid}
  legacy_intent_root_start=${root_start}
  legacy_intent_cgroup_path=${cgroup_path}
  legacy_intent_cgroup_dev=${cgroup_dev}
  legacy_intent_cgroup_ino=${cgroup_ino}
  legacy_intent_uid=${uid}
  legacy_intent_cgroup_fingerprint=$(
    printf '%s\0%s\0%s\0%s' \
      "${cgroup_path}" "${cgroup_dev}" "${cgroup_ino}" "${owner}" \
      | sha256sum | awk '{print $1}'
  ) || return 2
  [[ "${legacy_intent_cgroup_fingerprint}" =~ ^[0-9a-f]{64}$ ]] || return 2
}

legacy_publish_capture_intent() {
  local intent="$1" token="$2" epoch="$3" command="$4" snapshot="$5"
  local log_dir="$6" target="$7" root_pid="$8" root_start="$9"
  local cgroup_path="${10}" cgroup_dev="${11}" cgroup_ino="${12}"
  local incoming current_uid
  current_uid=$(id -u) || return 2
  if [[ -e "${intent}" || -L "${intent}" ]]; then
    legacy_load_capture_intent \
      "${intent}" "${token}" "${epoch}" "${command}" "${snapshot}" \
      "${log_dir}" "${target}" || return
    if [[ "${legacy_intent_root_pid}" != "${root_pid}" \
          || "${legacy_intent_root_start}" != "${root_start}" \
          || "${legacy_intent_cgroup_path}" != "${cgroup_path}" \
          || "${legacy_intent_cgroup_dev}" != "${cgroup_dev}" \
          || "${legacy_intent_cgroup_ino}" != "${cgroup_ino}" ]]; then
      echo "[ERROR] Existing legacy cgroup freeze intent names a different pane boundary." >&2
      return 2
    fi
    return 0
  fi
  incoming="${intent}.in"
  if [[ -e "${incoming}" || -L "${incoming}" ]]; then
    echo "[ERROR] Legacy cgroup freeze-intent incoming path already exists." >&2
    return 2
  fi
  if ! (umask 077; set -o noclobber; printf \
      '1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${token}" "${epoch}" "${command}" "${snapshot}" "${log_dir}" \
      "${target}" "${root_pid}" "${root_start}" "${cgroup_path}" \
      "${cgroup_dev}" "${cgroup_ino}" "${current_uid}" > "${incoming}") 2>/dev/null \
      || [[ ! -f "${incoming}" || -L "${incoming}" ]] \
      || ! chmod 0400 "${incoming}" \
      || ! legacy_load_capture_intent \
        "${incoming}" "${token}" "${epoch}" "${command}" "${snapshot}" \
        "${log_dir}" "${target}"; then
    rm -f -- "${incoming}"
    echo "[ERROR] Could not construct a canonical legacy cgroup freeze intent." >&2
    return 2
  fi
  if ! mv -nT -- "${incoming}" "${intent}" \
      || [[ -e "${incoming}" || -L "${incoming}" ]]; then
    rm -f -- "${incoming}"
    echo "[ERROR] Could not atomically publish the legacy cgroup freeze intent." >&2
    return 2
  fi
  legacy_load_capture_intent \
    "${intent}" "${token}" "${epoch}" "${command}" "${snapshot}" \
    "${log_dir}" "${target}"
}

legacy_unfreeze_captured() {
  [[ -n "${legacy_capture_cgroup_path:-}" \
      && -n "${legacy_capture_cgroup_dev:-}" \
      && -n "${legacy_capture_cgroup_ino:-}" ]] || return 0
  legacy_set_cgroup_frozen_exact \
    "${legacy_capture_cgroup_path}" "${legacy_capture_cgroup_dev}" \
    "${legacy_capture_cgroup_ino}" 0
}

legacy_capture_freeze_and_publish() {
  local root_pid="$1" root_start="$2" receipt="$3" token="$4" epoch="$5"
  local command_sha="$6" snapshot="$7" log_dir="$8" target="$9"
  local expected_control="${10}" intent round index
  local closure_fingerprint previous_fingerprint="" stable_rounds=0
  local incoming count
  declare -g legacy_capture_root_pid="${root_pid}"
  declare -g legacy_capture_cgroup_path="${legacy_pane_cgroup_path:-}"
  declare -g legacy_capture_cgroup_dev="${legacy_pane_cgroup_dev:-}"
  declare -g legacy_capture_cgroup_ino="${legacy_pane_cgroup_ino:-}"

  if [[ -z "${legacy_capture_cgroup_path}" \
        || -z "${legacy_capture_cgroup_dev}" \
        || -z "${legacy_capture_cgroup_ino}" ]]; then
    echo "[ERROR] Legacy capture lacks the exact preflight tmux-spawn cgroup identity." >&2
    return 2
  fi
  legacy_validate_pane_process "${root_pid}" "${expected_control}" || return
  [[ "${legacy_pane_start}" == "${root_start}" ]] || return 2
  legacy_read_proc_cgroup_v2_exact "${root_pid}" "${root_start}" || return
  if [[ "${legacy_proc_cgroup_path}" != "${legacy_capture_cgroup_path}" ]]; then
    echo "[ERROR] Exact legacy pane changed cgroup after preflight." >&2
    return 2
  fi
  legacy_validate_leaf_cgroup_v2 \
    "${legacy_capture_cgroup_path}" "${legacy_capture_cgroup_dev}" \
    "${legacy_capture_cgroup_ino}" || return

  intent=$(legacy_capture_intent_path "${receipt}") || return 2
  legacy_publish_capture_intent \
    "${intent}" "${token}" "${epoch}" "${command_sha}" "${snapshot}" \
    "${log_dir}" "${target}" "${root_pid}" "${root_start}" \
    "${legacy_capture_cgroup_path}" "${legacy_capture_cgroup_dev}" \
    "${legacy_capture_cgroup_ino}" || return

  if ! legacy_set_cgroup_frozen_exact \
      "${legacy_capture_cgroup_path}" "${legacy_capture_cgroup_dev}" \
      "${legacy_capture_cgroup_ino}" 1; then
    if legacy_unfreeze_captured; then
      rm -f -- "${intent}"
    fi
    return 2
  fi

  for ((round = 0; round < 4; round++)); do
    if ! legacy_collect_frozen_cgroup_members \
        "${root_pid}" "${root_start}" "${legacy_capture_cgroup_path}" \
        "${legacy_capture_cgroup_dev}" "${legacy_capture_cgroup_ino}" \
        "${expected_control}"; then
      if legacy_unfreeze_captured; then
        rm -f -- "${intent}"
      fi
      return 2
    fi
    closure_fingerprint=$(
      for index in "${!legacy_discovered_pids[@]}"; do
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${legacy_discovered_pids[${index}]}" \
          "${legacy_discovered_starts[${index}]}" \
          "${legacy_discovered_uids[${index}]}" \
          "${legacy_discovered_ppids[${index}]}" \
          "${legacy_discovered_pgrps[${index}]}" \
          "${legacy_discovered_sessions[${index}]}"
      done | sha256sum | awk '{print $1}'
    ) || closure_fingerprint=""
    if [[ ! "${closure_fingerprint}" =~ ^[0-9a-f]{64}$ ]]; then
      if legacy_unfreeze_captured; then
        rm -f -- "${intent}"
      fi
      return 2
    fi
    if [[ "${closure_fingerprint}" == "${previous_fingerprint}" ]]; then
      stable_rounds=$((stable_rounds + 1))
    else
      stable_rounds=1
    fi
    previous_fingerprint=${closure_fingerprint}
    (( stable_rounds >= 2 )) && break
    sleep 0.05
  done
  if (( stable_rounds < 2 )); then
    echo "[ERROR] Frozen legacy cgroup did not produce two identical exact membership rounds." >&2
    if legacy_unfreeze_captured; then
      rm -f -- "${intent}"
    fi
    return 2
  fi

  if [[ -e "${receipt}" || -L "${receipt}" ]]; then
    echo "[ERROR] Refusing to overwrite an existing legacy cgroup receipt." >&2
    if legacy_unfreeze_captured; then
      rm -f -- "${intent}"
    fi
    return 2
  fi
  incoming="${receipt}.in"
  if [[ -e "${incoming}" || -L "${incoming}" ]]; then
    echo "[ERROR] Legacy cgroup receipt incoming path already exists." >&2
    if legacy_unfreeze_captured; then
      rm -f -- "${intent}"
    fi
    return 2
  fi
  count=${#legacy_discovered_pids[@]}
  if ! (umask 077; set -o noclobber; {
      printf '2\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${token}" "${epoch}" "${command_sha}" "${snapshot}" "${log_dir}" \
        "${target}" "${root_pid}" "${root_start}" \
        "${legacy_capture_cgroup_path}" "${legacy_capture_cgroup_dev}" \
        "${legacy_capture_cgroup_ino}" "${count}"
      for index in "${!legacy_discovered_pids[@]}"; do
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${legacy_discovered_pids[${index}]}" \
          "${legacy_discovered_starts[${index}]}" \
          "${legacy_discovered_uids[${index}]}" \
          "${legacy_discovered_ppids[${index}]}" \
          "${legacy_discovered_pgrps[${index}]}" \
          "${legacy_discovered_sessions[${index}]}"
      done
    } > "${incoming}") 2>/dev/null \
      || [[ ! -f "${incoming}" || -L "${incoming}" ]] \
      || ! chmod 0400 "${incoming}" \
      || ! legacy_load_capture_receipt \
        "${incoming}" "${token}" "${epoch}" "${command_sha}" \
        "${snapshot}" "${log_dir}" "${target}"; then
    rm -f -- "${incoming}"
    if legacy_unfreeze_captured; then
      rm -f -- "${intent}"
    fi
    echo "[ERROR] Could not construct a canonical legacy cgroup receipt." >&2
    return 2
  fi
  if ! mv -nT -- "${incoming}" "${receipt}" \
      || [[ -e "${incoming}" || -L "${incoming}" ]]; then
    rm -f -- "${incoming}"
    if legacy_unfreeze_captured; then
      rm -f -- "${intent}"
    fi
    echo "[ERROR] Could not atomically publish the legacy cgroup receipt." >&2
    return 2
  fi
  if ! legacy_load_capture_receipt \
      "${receipt}" "${token}" "${epoch}" "${command_sha}" \
      "${snapshot}" "${log_dir}" "${target}"; then
    if legacy_unfreeze_captured; then
      rm -f -- "${receipt}" "${intent}"
    fi
    return 2
  fi
  if ! rm -f -- "${intent}"; then
    if legacy_unfreeze_captured; then
      rm -f -- "${receipt}" "${intent}"
    fi
    echo "[ERROR] Could not retire the durable legacy cgroup freeze intent." >&2
    return 2
  fi
}

legacy_unfreeze_receipt_cgroup() {
  legacy_set_cgroup_frozen_exact \
    "${legacy_receipt_cgroup_path}" "${legacy_receipt_cgroup_dev}" \
    "${legacy_receipt_cgroup_ino}" 0
}

legacy_validate_frozen_receipt_cgroup_open_body() {
  local pid index
  legacy_read_cgroup_frozen_state_fd || return
  if [[ "${legacy_cgroup_freeze_requested}" != 1 \
        || "${legacy_cgroup_frozen}" != 1 ]]; then
    echo "[ERROR] Executable legacy receipt members are not held by the exact effective cgroup freeze." >&2
    return 2
  fi
  if ! mapfile -t member_pids < "${legacy_open_cgroup_fd_path}/cgroup.procs" 2>/dev/null; then
    echo "[ERROR] Could not enumerate exact frozen receipt cgroup members." >&2
    return 2
  fi
  member_count=${#member_pids[@]}
  (( member_count > 0 && member_count <= 65536 )) || return 2
  for pid in "${member_pids[@]}"; do
    if [[ ! "${pid}" =~ ^[1-9][0-9]*$ \
          || -n "${member_seen[${pid}]+x}" \
          || -z "${receipt_index[${pid}]+x}" ]]; then
      echo "[ERROR] Frozen legacy cgroup contains a process outside its immutable v2 receipt." >&2
      return 2
    fi
    member_seen[${pid}]=1
    index=${receipt_index[${pid}]}
    legacy_pid_matches_start "${pid}" "${legacy_receipt_starts[${index}]}" || return 2
    if [[ "${legacy_proc_uid}" != "${legacy_receipt_uids[${index}]}" \
          || "${legacy_proc_ppid}" != "${legacy_receipt_ppids[${index}]}" \
          || "${legacy_proc_pgrp}" != "${legacy_receipt_pgrps[${index}]}" \
          || "${legacy_proc_session}" != "${legacy_receipt_sessions[${index}]}" ]]; then
      echo "[ERROR] Frozen legacy cgroup member tuple no longer matches its immutable receipt." >&2
      return 2
    fi
  done
  for pid in "${!live_receipt[@]}"; do
    [[ -n "${member_seen[${pid}]+x}" ]] || {
      echo "[ERROR] A live legacy receipt identity is absent from its exact frozen cgroup." >&2
      return 2
    }
  done
  legacy_validate_pane_process \
    "${legacy_receipt_root_pid}" "${expected_control}" || return
  [[ "${legacy_pane_start}" == "${legacy_receipt_root_start}" ]] || return 2
  legacy_read_cgroup_frozen_state_fd || return
  [[ "${legacy_cgroup_freeze_requested}" == 1 \
        && "${legacy_cgroup_frozen}" == 1 ]] || return 2
}

legacy_validate_frozen_receipt_cgroup() {
  local expected_control="$1" index pid match_rc current_uid root_executable=0
  local executable_count=0 member_count=0 rc=0 close_rc
  local -a member_pids=()
  local -A receipt_index=() live_receipt=() member_seen=()
  current_uid=$(id -u) || return 2
  for index in "${!legacy_receipt_pids[@]}"; do
    receipt_index[${legacy_receipt_pids[${index}]}]=${index}
  done
  for index in "${!legacy_receipt_pids[@]}"; do
    pid=${legacy_receipt_pids[${index}]}
    if legacy_pid_matches_start "${pid}" "${legacy_receipt_starts[${index}]}"; then
      if [[ "${legacy_proc_uid}" != "${legacy_receipt_uids[${index}]}" \
            || "${legacy_proc_uid}" != "${current_uid}" ]]; then
        echo "[ERROR] Live legacy receipt owner drifted before cgroup cleanup: PID ${pid}." >&2
        return 2
      fi
      # A process sealed by cgroup.kill may remain briefly as a zombie while
      # tmux/systemd reaps it.  Its kernel parent can legitimately change then;
      # PID/start/UID still bind the generation, and terminal closure waits for
      # complete reap.  Only executable members authorize another kill.
      case "${legacy_proc_state}" in
        Z|X|x) continue ;;
      esac
      if [[ "${legacy_proc_ppid}" != "${legacy_receipt_ppids[${index}]}" \
            || "${legacy_proc_pgrp}" != "${legacy_receipt_pgrps[${index}]}" \
            || "${legacy_proc_session}" != "${legacy_receipt_sessions[${index}]}" ]]; then
        echo "[ERROR] Live legacy receipt tuple drifted before cgroup cleanup: PID ${pid}." >&2
        return 2
      fi
      legacy_read_proc_cgroup_v2_exact \
        "${pid}" "${legacy_receipt_starts[${index}]}" || return
      if [[ "${legacy_proc_cgroup_path}" != "${legacy_receipt_cgroup_path}" ]]; then
        echo "[ERROR] Live legacy receipt PID ${pid} moved outside its exact cgroup." >&2
        return 2
      fi
      live_receipt[${pid}]=1
      executable_count=$((executable_count + 1))
      [[ "${pid}" != "${legacy_receipt_root_pid}" ]] || root_executable=1
    else
      match_rc=$?
      (( match_rc == 1 )) || return "${match_rc}"
    fi
  done
  # Once every executable receipt identity is gone, the scope may already have
  # been collected by systemd.  No path lookup is then allowed to target a
  # later same-name scope; exact PID/start reap remains the terminal proof.
  (( executable_count != 0 )) || return 0
  if (( root_executable != 1 )); then
    echo "[ERROR] Legacy pane root is sealed/absent beside executable receipt members." >&2
    return 2
  fi
  legacy_open_exact_cgroup_fd \
    "${legacy_receipt_cgroup_path}" "${legacy_receipt_cgroup_dev}" \
    "${legacy_receipt_cgroup_ino}" || return
  legacy_validate_frozen_receipt_cgroup_open_body || rc=$?
  legacy_close_exact_cgroup_fd || {
    close_rc=$?
    (( rc != 0 )) || rc=${close_rc}
  }
  return "${rc}"
}

legacy_collect_receipt_survivors() {
  local index match_rc
  legacy_receipt_survivors=()
  for index in "${!legacy_receipt_pids[@]}"; do
    if legacy_pid_matches_start \
        "${legacy_receipt_pids[${index}]}" "${legacy_receipt_starts[${index}]}"; then
      legacy_receipt_survivors+=("${legacy_receipt_pids[${index}]}")
    else
      match_rc=$?
      (( match_rc == 1 )) || return "${match_rc}"
    fi
  done
}

legacy_collect_receipt_executable_survivors() {
  local index match_rc
  legacy_receipt_executable_survivors=()
  for index in "${!legacy_receipt_pids[@]}"; do
    if legacy_pid_matches_start \
        "${legacy_receipt_pids[${index}]}" "${legacy_receipt_starts[${index}]}"; then
      case "${legacy_proc_state}" in
        Z|X|x) ;;
        *) legacy_receipt_executable_survivors+=("${legacy_receipt_pids[${index}]}") ;;
      esac
    else
      match_rc=$?
      (( match_rc == 1 )) || return "${match_rc}"
    fi
  done
}

legacy_verify_receipt_closed() {
  legacy_collect_receipt_survivors || return
  if (( ${#legacy_receipt_survivors[@]} != 0 )); then
    echo "[ERROR] Captured legacy launch processes remain: ${legacy_receipt_survivors[*]}" >&2
    return 2
  fi
}

legacy_verify_receipt_cgroup_empty_or_gone() {
  local exact_path="/sys/fs/cgroup${legacy_receipt_cgroup_path}"
  if [[ ! -e "${exact_path}" && ! -L "${exact_path}" ]]; then
    return 0
  fi
  legacy_read_cgroup_frozen_state \
    "${legacy_receipt_cgroup_path}" "${legacy_receipt_cgroup_dev}" \
    "${legacy_receipt_cgroup_ino}" || return
  if [[ "${legacy_cgroup_populated}" != 0 ]]; then
    echo "[ERROR] Exact legacy receipt cgroup remains populated after stopped closure." >&2
    return 2
  fi
}

legacy_verify_terminal_receipt_closure() {
  legacy_verify_receipt_closed || return
  legacy_verify_receipt_cgroup_empty_or_gone
}

legacy_wait_receipt_cgroup_empty_or_gone_bounded() {
  local exact_path="/sys/fs/cgroup${legacy_receipt_cgroup_path}"
  local state_rc
  while :; do
    if [[ ! -e "${exact_path}" && ! -L "${exact_path}" ]]; then
      return 0
    fi
    state_rc=0
    legacy_read_cgroup_frozen_state \
      "${legacy_receipt_cgroup_path}" "${legacy_receipt_cgroup_dev}" \
      "${legacy_receipt_cgroup_ino}" || state_rc=$?
    if (( state_rc == 0 )) && [[ "${legacy_cgroup_populated}" == 0 ]]; then
      return 0
    fi
    # Removal between the pathname check and the authenticated open is a valid
    # terminal result.  A present replacement or malformed original remains a
    # hard identity error and must never be retried as if it were ordinary
    # systemd cleanup latency.
    if (( state_rc != 0 )); then
      if [[ ! -e "${exact_path}" && ! -L "${exact_path}" ]]; then
        return 0
      fi
      return "${state_rc}"
    fi
    (( SECONDS < LEGACY_STOP_CLEANUP_DEADLINE_SECONDS )) || break
    sleep 0.05
  done
  echo "[ERROR] Exact legacy receipt cgroup did not reach populated=0 or exact path removal within the cleanup deadline." >&2
  return 2
}

legacy_wait_receipt_closed_bounded() {
  while :; do
    legacy_collect_receipt_survivors || return
    (( ${#legacy_receipt_survivors[@]} == 0 )) && return 0
    (( SECONDS < LEGACY_STOP_CLEANUP_DEADLINE_SECONDS )) || break
    sleep 0.05
  done
  echo "[ERROR] Captured legacy identities were sealed but not reaped within the cleanup deadline: ${legacy_receipt_survivors[*]}" >&2
  return 2
}

legacy_revalidate_open_receipt_members() {
  local expected_control="$1" current_uid index pid match_rc
  local -a member_pids=()
  local -A receipt_index=() member_seen=()
  current_uid=$(id -u) || return 2
  for index in "${!legacy_receipt_pids[@]}"; do
    receipt_index[${legacy_receipt_pids[${index}]}]=${index}
  done
  legacy_read_cgroup_frozen_state_fd || return
  if [[ "${legacy_cgroup_freeze_requested}" != 1 \
        || "${legacy_cgroup_frozen}" != 1 ]]; then
    echo "[ERROR] Exact receipt cgroup lost effective freeze before cgroup.kill commit." >&2
    return 2
  fi
  if ! mapfile -t member_pids < "${legacy_open_cgroup_fd_path}/cgroup.procs" 2>/dev/null \
      || (( ${#member_pids[@]} == 0 || ${#member_pids[@]} > 65536 )); then
    echo "[ERROR] Exact receipt cgroup membership is empty/malformed before cgroup.kill commit." >&2
    return 2
  fi
  for pid in "${member_pids[@]}"; do
    if [[ ! "${pid}" =~ ^[1-9][0-9]*$ \
          || -n "${member_seen[${pid}]+x}" \
          || -z "${receipt_index[${pid}]+x}" ]]; then
      echo "[ERROR] Exact receipt cgroup gained an unauthorized member before cgroup.kill commit." >&2
      return 2
    fi
    member_seen[${pid}]=1
    index=${receipt_index[${pid}]}
    legacy_pid_matches_start "${pid}" "${legacy_receipt_starts[${index}]}" || return 2
    if [[ "${legacy_proc_uid}" != "${legacy_receipt_uids[${index}]}" \
          || "${legacy_proc_uid}" != "${current_uid}" \
          || "${legacy_proc_ppid}" != "${legacy_receipt_ppids[${index}]}" \
          || "${legacy_proc_pgrp}" != "${legacy_receipt_pgrps[${index}]}" \
          || "${legacy_proc_session}" != "${legacy_receipt_sessions[${index}]}" ]]; then
      echo "[ERROR] Exact receipt member tuple drifted before cgroup.kill commit." >&2
      return 2
    fi
  done
  for index in "${!legacy_receipt_pids[@]}"; do
    pid=${legacy_receipt_pids[${index}]}
    if legacy_pid_matches_start "${pid}" "${legacy_receipt_starts[${index}]}"; then
      case "${legacy_proc_state}" in
        Z|X|x) ;;
        *)
          [[ -n "${member_seen[${pid}]+x}" ]] || {
            echo "[ERROR] Executable receipt identity left its exact cgroup before commit." >&2
            return 2
          }
          ;;
      esac
    else
      match_rc=$?
      (( match_rc == 1 )) || return "${match_rc}"
    fi
  done
  legacy_validate_pane_process \
    "${legacy_receipt_root_pid}" "${expected_control}" || return
  [[ "${legacy_pane_start}" == "${legacy_receipt_root_start}" ]] || return 2
  legacy_read_cgroup_frozen_state_fd || return
  [[ "${legacy_cgroup_freeze_requested}" == 1 \
        && "${legacy_cgroup_frozen}" == 1 ]] || return 2
}

legacy_terminate_receipt_bounded() {
  local expected_control="$1" fd_path
  local state_rc exact_path
  legacy_validate_frozen_receipt_cgroup "${expected_control}" || return
  legacy_collect_receipt_executable_survivors || return
  if (( ${#legacy_receipt_executable_survivors[@]} == 0 )); then
    # An idempotent retry commonly enters here after cgroup.kill has removed
    # every executable receipt member but systemd is still retiring the exact
    # scope.  Wait within the same transaction deadline instead of converting
    # ordinary populated/path-removal latency into a spurious hard failure.
    legacy_wait_receipt_cgroup_empty_or_gone_bounded
    return
  fi

  # Hold the authenticated scope directory itself open across the irreversible
  # write.  Rechecking only a pathname would leave a deletion/recreation TOCTOU
  # between the final receipt comparison and cgroup.kill.
  legacy_open_exact_cgroup_fd \
    "${legacy_receipt_cgroup_path}" "${legacy_receipt_cgroup_dev}" \
    "${legacy_receipt_cgroup_ino}" || return
  fd_path=${legacy_open_cgroup_fd_path}
  exact_path="/sys/fs/cgroup${legacy_receipt_cgroup_path}"
  if [[ ! -f "${fd_path}/cgroup.kill" \
        || -L "${fd_path}/cgroup.kill" ]]; then
    legacy_close_exact_cgroup_fd || true
    echo "[ERROR] Open legacy cgroup directory does not match the immutable receipt." >&2
    return 2
  fi
  if ! legacy_revalidate_open_receipt_members "${expected_control}"; then
    legacy_close_exact_cgroup_fd || true
    return 2
  fi
  if ! printf '1\n' > "${fd_path}/cgroup.kill"; then
    legacy_close_exact_cgroup_fd || true
    echo "[ERROR] Exact legacy cgroup.kill transaction failed; scope remains quarantined." >&2
    return 2
  fi

  # cgroup.kill is the kernel primitive which closes concurrent forks and
  # migrations during the kill transaction.  Require both receipt executable
  # closure and cgroup.events:populated=0 before returning.  If systemd removes
  # this exact scope first, successful removal itself proves that the cgroup was
  # empty; the held directory FD still binds that conclusion to its dev/inode.
  while :; do
    legacy_collect_receipt_executable_survivors || {
      legacy_close_exact_cgroup_fd || true
      return 2
    }
    state_rc=0
    legacy_read_cgroup_frozen_state_fd || state_rc=$?
    if (( ${#legacy_receipt_executable_survivors[@]} == 0 )); then
      if (( state_rc == 0 )) && [[ "${legacy_cgroup_populated}" == 0 ]]; then
        legacy_close_exact_cgroup_fd
        return
      fi
      # The held FD remains bound to the authenticated dev/inode after systemd
      # unlinks its pathname.  Its cgroup.events snapshot may still report the
      # pre-unlink populated value while kernel references drain; exact path
      # removal plus zero executable receipt identities is already sufficient
      # to close the FD and proceed to tmux/reap verification.
      if [[ ! -e "${exact_path}" && ! -L "${exact_path}" ]]; then
        legacy_close_exact_cgroup_fd
        return
      fi
    fi
    if (( state_rc != 0 )) \
        && [[ -e "${exact_path}" || -L "${exact_path}" ]]; then
      legacy_close_exact_cgroup_fd || true
      return "${state_rc}"
    fi
    (( SECONDS < LEGACY_STOP_CLEANUP_DEADLINE_SECONDS )) || break
    sleep 0.05
  done
  legacy_close_exact_cgroup_fd || true
  echo "[ERROR] Exact legacy cgroup.kill did not reach receipt closure plus populated=0: ${legacy_receipt_executable_survivors[*]}" >&2
  return 2
}
EOF
}

release_rendezvous_ports() {
  local launch_token="$1"
  local main_state provenance_state
  main_state=$(rendezvous_state_path "${MASTER_PORT}")
  provenance_state=$(rendezvous_state_path "${HOLOSOMA_PROVENANCE_MASTER_PORT}")
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
STATE_ROOT=$(quote "${REMOTE_RUN_ROOT}/.rendezvous")
TOKEN=$(quote "${launch_token}")
SESSION_NAME=$(quote "${SESSION}")
ALLOW_LEGACY_PRIVATE_STATE_MODE=$(quote "${LEGACY_STOP_EXPECTED_TOKEN:+1}")
ALLOW_LEGACY_PRIVATE_STATE_MODE=\${ALLOW_LEGACY_PRIVATE_STATE_MODE:-0}
mkdir -p "\${STATE_ROOT}"
$(private_lifecycle_file_validation_helpers)
open_private_lifecycle_lock "\${STATE_ROOT}/.reservation.lock" 9
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 9; then
  echo "[ERROR] Timed out acquiring the rendezvous release lock." >&2
  exit 1
fi
$(rendezvous_release_validation_helpers)
validate_and_release_rendezvous_pair \
  $(quote "${main_state}") $(quote "${provenance_state}") \
  "\${TOKEN}" "\${SESSION_NAME}" $(quote "${MASTER_PORT}") \
  $(quote "${HOLOSOMA_PROVENANCE_MASTER_PORT}")
EOF
)
  remote_run_bounded "${MASTER_ADDR}" "${cmd}" "${LAUNCH_CLEANUP_TIMEOUT_SECONDS}"
}

publish_launch_intent_node() {
  local node="$1"
  local launch_token="$2"
  local launch_epoch="$3"
  local expected_predecessor="$4"
  if [[ "${expected_predecessor}" != ABSENT \
        && ! "${expected_predecessor}" =~ ^TERMINAL:[0-9a-f]{64}$ ]]; then
    echo "[ERROR][${node}] Invalid controller launch predecessor: ${expected_predecessor}" >&2
    return 2
  fi
  local active_state_path
  active_state_path="${REMOTE_RUN_ROOT}/.active/$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}')_${node}.state"
  local cancellation_state_path
  cancellation_state_path="${REMOTE_RUN_ROOT}/.active/cancelled.$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}').${node}.${launch_token}.state"
  local tmux_lock_path
  tmux_lock_path="${REMOTE_RUN_ROOT}/.active/.locks/holosoma-tmux-$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}').lock"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
ACTIVE_STATE=$(quote "${active_state_path}")
CANCELLATION_STATE=$(quote "${cancellation_state_path}")
EXPECTED_PREDECESSOR=$(quote "${expected_predecessor}")
mkdir -p "\$(dirname "\${ACTIVE_STATE}")"
$(private_lifecycle_file_validation_helpers)
open_private_lifecycle_lock $(quote "${tmux_lock_path}") 8
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 8; then
  echo "[ERROR][${node}] Timed out acquiring the ${SESSION} lifecycle lock while publishing launch intent." >&2
  exit 1
fi
# A controller timeout can leave this remote shell alive behind the lifecycle
# lock.  A persistent exact-token tombstone, written by controller cleanup
# under the same lock, makes a late publisher fail before its atomic mv.
if [[ -e "\${CANCELLATION_STATE}" || -L "\${CANCELLATION_STATE}" ]]; then
  if ! validate_private_state_file_metadata "\${CANCELLATION_STATE}" 4096 0 \
      || [[ ! -f "\${CANCELLATION_STATE}" || -L "\${CANCELLATION_STATE}" \
        || "\$(awk 'END { print NR }' "\${CANCELLATION_STATE}")" != 1 \
        || "\$(awk -F '\t' 'NR == 1 { print NF }' "\${CANCELLATION_STATE}")" != 8 ]]; then
    echo "[ERROR][${node}] Launch cancellation tombstone is malformed: \${CANCELLATION_STATE}" >&2
    exit 2
  fi
  cancel_version="" cancel_token="" cancel_epoch="" cancel_snapshot=""
  cancel_log_dir="" cancel_target="" cancel_session="" cancel_node=""
  IFS=\$'\t' read -r cancel_version cancel_token cancel_epoch cancel_snapshot \
    cancel_log_dir cancel_target cancel_session cancel_node < "\${CANCELLATION_STATE}" || true
  if [[ "\${cancel_version}" != 1 \
        || "\${cancel_token}" != $(quote "${launch_token}") \
        || "\${cancel_epoch}" != $(quote "${launch_epoch}") \
        || "\${cancel_snapshot}" != $(quote "${SOURCE_SNAPSHOT_ID}") \
        || "\${cancel_log_dir}" != $(quote "${LOG_DIR}") \
        || "\${cancel_target}" != $(quote "${TARGET_LEARNING_ITERATION}") \
        || "\${cancel_session}" != $(quote "${SESSION}") \
        || "\${cancel_node}" != $(quote "${node}") ]]; then
    echo "[ERROR][${node}] Launch cancellation tombstone identity is malformed: \${CANCELLATION_STATE}" >&2
    exit 2
  fi
  echo "[ERROR][${node}] Launch token was durably cancelled before intent publication." >&2
  exit 3
fi
# Preflight and publication are separate SSH calls.  A second controller can
# pass preflight, pause, and otherwise overwrite a job which starts in between;
# repeat both ownership checks while holding the same lock used by tmux launch.
$(tmux_session_query_helpers)
query_tmux_session_presence $(quote "${SESSION}") || exit
if (( tmux_session_present == 1 )); then
  echo "[ERROR][${node}] Refusing to publish launch intent while tmux session ${SESSION} exists." >&2
  exit 1
fi
$(active_state_validation_helpers)
$(launch_process_closure_helpers)
if [[ -e "\${ACTIVE_STATE}" || -L "\${ACTIVE_STATE}" ]]; then
  if ! load_active_state_v2_exact "\${ACTIVE_STATE}" \
      || ! active_state_has_session_namespace $(quote "${SESSION}"); then
    echo "[ERROR][${node}] Refusing to replace malformed active launch metadata: \${ACTIVE_STATE}" >&2
    exit 1
  fi
  version=\${active_version} phase=\${active_phase} snapshot=\${active_snapshot}
  log_dir=\${active_log_dir} target=\${active_target} token=\${active_token}
  command_sha=\${active_command_sha} epoch=\${active_epoch}
  case "\${phase}" in
    launching)
      if [[ "\${token}" != $(quote "${launch_token}") ]]; then
        echo "[ERROR][${node}] Refusing concurrent launch intent: active launching token=\${token}." >&2
        exit 1
      fi
      if [[ "\${snapshot}" == $(quote "${SOURCE_SNAPSHOT_ID}") \
            && "\${log_dir}" == $(quote "${LOG_DIR}") \
            && "\${target}" == $(quote "${TARGET_LEARNING_ITERATION}") \
            && "\${command_sha}" == pending \
            && "\${epoch}" == $(quote "${launch_epoch}") ]]; then
        verify_no_launch_token_epoch_processes "\${token}" "\${epoch}"
        echo "[INFO][${node}] exact launch intent already published for token=\${token}"
        exit 0
      fi
      echo "[ERROR][${node}] Same-token launch intent differs from the requested immutable identity." >&2
      exit 1
      ;;
    running|rolling_back|stopping)
      echo "[ERROR][${node}] Refusing to replace active phase=\${phase} token=\${token} with a new launch intent." >&2
      exit 1
      ;;
    stopped|rolled_back)
      if [[ ! "\${EXPECTED_PREDECESSOR}" =~ ^TERMINAL:[0-9a-f]{64}$ ]]; then
        echo "[ERROR][${node}] Launch predecessor CAS mismatch: expected absent metadata but found terminal phase=\${phase}." >&2
        exit 1
      fi
      verify_no_launch_token_epoch_processes "\${token}" "\${epoch}"
      current_predecessor_sha=\$(sha256sum "\${ACTIVE_STATE}" | awk '{print \$1}')
      if [[ "TERMINAL:\${current_predecessor_sha}" != "\${EXPECTED_PREDECESSOR}" ]]; then
        echo "[ERROR][${node}] Launch predecessor CAS mismatch: terminal active metadata changed after preflight." >&2
        exit 1
      fi
      ;;
    *)
      echo "[ERROR][${node}] Refusing unsupported active lifecycle phase=\${phase}." >&2
      exit 1
      ;;
  esac
elif [[ "\${EXPECTED_PREDECESSOR}" != ABSENT ]]; then
  echo "[ERROR][${node}] Launch predecessor CAS mismatch: preflight terminal metadata is now absent." >&2
  exit 1
fi
ACTIVE_INCOMING="\${ACTIVE_STATE}.incoming.${launch_token}"
rm -f -- "\${ACTIVE_INCOMING}"
printf '2\tlaunching\t%s\t%s\t%s\t%s\tpending\t%s\n' \
  $(quote "${SOURCE_SNAPSHOT_ID}") \
  $(quote "${LOG_DIR}") \
  $(quote "${TARGET_LEARNING_ITERATION}") \
  $(quote "${launch_token}") \
  $(quote "${launch_epoch}") > "\${ACTIVE_INCOMING}"
mv -T "\${ACTIVE_INCOMING}" "\${ACTIVE_STATE}"
validate_private_state_file_metadata "\${ACTIVE_STATE}" 4096 0
EOF
)
  remote_run_bounded "${node}" "${cmd}" "${LAUNCH_CONTROL_TIMEOUT_SECONDS}"
}

cancel_launch_intent_node() {
  local node="$1"
  local launch_token="$2"
  local launch_epoch="$3"
  local active_state_path
  active_state_path="${REMOTE_RUN_ROOT}/.active/$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}')_${node}.state"
  local cancellation_state_path
  cancellation_state_path="${REMOTE_RUN_ROOT}/.active/cancelled.$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}').${node}.${launch_token}.state"
  local tmux_lock_path
  tmux_lock_path="${REMOTE_RUN_ROOT}/.active/.locks/holosoma-tmux-$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}').lock"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
ACTIVE_STATE=$(quote "${active_state_path}")
CANCELLATION_STATE=$(quote "${cancellation_state_path}")
mkdir -p "\$(dirname "\${ACTIVE_STATE}")"
$(private_lifecycle_file_validation_helpers)
open_private_lifecycle_lock $(quote "${tmux_lock_path}") 8
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 8; then
  echo "[ERROR][${node}] Timed out acquiring lifecycle lock while cancelling launch intent." >&2
  exit 1
fi
validate_cancellation_tombstone() {
  validate_private_state_file_metadata "\${CANCELLATION_STATE}" 4096 0 \
    && [[ -f "\${CANCELLATION_STATE}" && ! -L "\${CANCELLATION_STATE}" \
      && "\$(awk 'END { print NR }' "\${CANCELLATION_STATE}")" == 1 \
      && "\$(awk -F '\t' 'NR == 1 { print NF }' "\${CANCELLATION_STATE}")" == 8 ]] || return 2
  local version token epoch snapshot log_dir target session_name node_name
  version="" token="" epoch="" snapshot="" log_dir="" target="" session_name="" node_name=""
  IFS=\$'\t' read -r version token epoch snapshot log_dir target session_name node_name \
    < "\${CANCELLATION_STATE}" || true
  [[ "\${version}" == 1 \
      && "\${token}" == $(quote "${launch_token}") \
      && "\${epoch}" == $(quote "${launch_epoch}") \
      && "\${snapshot}" == $(quote "${SOURCE_SNAPSHOT_ID}") \
      && "\${log_dir}" == $(quote "${LOG_DIR}") \
      && "\${target}" == $(quote "${TARGET_LEARNING_ITERATION}") \
      && "\${session_name}" == $(quote "${SESSION}") \
      && "\${node_name}" == $(quote "${node}") ]]
}
if [[ -e "\${CANCELLATION_STATE}" || -L "\${CANCELLATION_STATE}" ]]; then
  validate_cancellation_tombstone || {
    echo "[ERROR][${node}] Existing launch cancellation tombstone is malformed." >&2
    exit 2
  }
else
  cancellation_incoming="\${CANCELLATION_STATE}.incoming.\$\$"
  printf '1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    $(quote "${launch_token}") $(quote "${launch_epoch}") \
    $(quote "${SOURCE_SNAPSHOT_ID}") $(quote "${LOG_DIR}") \
    $(quote "${TARGET_LEARNING_ITERATION}") $(quote "${SESSION}") \
    $(quote "${node}") > "\${cancellation_incoming}"
  mv -T "\${cancellation_incoming}" "\${CANCELLATION_STATE}"
  validate_private_state_file_metadata "\${CANCELLATION_STATE}" 4096 0
fi
$(active_state_validation_helpers)
$(tmux_session_query_helpers)
$(tmux_ownership_helpers)
active_disposition=absent
if [[ -e "\${ACTIVE_STATE}" || -L "\${ACTIVE_STATE}" ]]; then
  if ! load_active_state_v2_exact "\${ACTIVE_STATE}" \
      || ! active_state_has_session_namespace $(quote "${SESSION}"); then
    echo "[ERROR][${node}] Cancel tombstone is durable, but active metadata is malformed." >&2
    exit 2
  fi
  if [[ "\${active_token}" == $(quote "${launch_token}") ]]; then
    if [[ "\${active_epoch}" != $(quote "${launch_epoch}") \
          || "\${active_snapshot}" != $(quote "${SOURCE_SNAPSHOT_ID}") \
          || "\${active_log_dir}" != $(quote "${LOG_DIR}") \
          || "\${active_target}" != $(quote "${TARGET_LEARNING_ITERATION}") \
          || "\${active_command_sha}" != pending ]]; then
      echo "[ERROR][${node}] Same-token active metadata differs from the cancelled intent." >&2
      exit 2
    fi
    case "\${active_phase}" in
      launching) active_disposition=old_intent ;;
      rolled_back) active_disposition=old_terminal ;;
      *)
        echo "[ERROR][${node}] Cannot cancel same-token active phase=\${active_phase}." >&2
        exit 2
        ;;
    esac
  else
    active_disposition=other
  fi
fi
query_tmux_session_presence $(quote "${SESSION}") || exit
if (( tmux_session_present == 1 )); then
  if [[ "\${active_disposition}" != other ]]; then
    echo "[ERROR][${node}] Cannot close cancelled intent while a same-name tmux session may belong to it." >&2
    exit 2
  fi
  case "\${active_phase}" in
    running|rolling_back|stopping)
      if [[ ! "\${active_command_sha}" =~ ^[0-9a-f]{64}$ ]] \
          || ! tmux_session_has_complete_new_identity \
            $(quote "${SESSION}") "\${active_token}" \
            "\${active_command_sha}" "\${active_epoch}"; then
        echo "[ERROR][${node}] Different-token active metadata was preserved, but same-name tmux is not its exact atomic identity." >&2
        exit 2
      fi
      ;;
    *)
      echo "[ERROR][${node}] Different-token phase=\${active_phase} cannot own a live same-name tmux during intent cancellation." >&2
      exit 2
      ;;
  esac
fi
if [[ "\${active_disposition}" == old_intent ]]; then
  active_incoming="\${ACTIVE_STATE}.incoming.cancelled.${launch_token}.\$\$"
  printf '2\trolled_back\t%s\t%s\t%s\t%s\tpending\t%s\n' \
    $(quote "${SOURCE_SNAPSHOT_ID}") $(quote "${LOG_DIR}") \
    $(quote "${TARGET_LEARNING_ITERATION}") $(quote "${launch_token}") \
    $(quote "${launch_epoch}") > "\${active_incoming}"
  mv -T "\${active_incoming}" "\${ACTIVE_STATE}"
  validate_private_state_file_metadata "\${ACTIVE_STATE}" 4096 0
fi
echo "[INFO][${node}] durably cancelled launch intent token=$(quote "${launch_token}") active_disposition=\${active_disposition}"
EOF
)
  remote_run_bounded "${node}" "${cmd}" "${LAUNCH_CLEANUP_TIMEOUT_SECONDS}"
}

LAST_LAUNCH_ACTIVE_PREDECESSOR=""

preflight_launch_intent_node() {
  local node="$1"
  LAST_LAUNCH_ACTIVE_PREDECESSOR=""
  local tmux_lock_path
  tmux_lock_path="${REMOTE_RUN_ROOT}/.active/.locks/holosoma-tmux-$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}').lock"
  local active_state_path
  active_state_path="${REMOTE_RUN_ROOT}/.active/$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}')_${node}.state"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
cd $(quote "${RUN_REPO}")
grep -Fx -- $(quote "${SOURCE_SNAPSHOT_ID}") .holosoma_snapshot/id >/dev/null
test "\$(sha256sum .holosoma_snapshot/source_manifest.sha256 | awk '{print \$1}')" = $(quote "${SOURCE_MANIFEST_SHA256}")
(sha256sum --quiet -c .holosoma_snapshot/source_manifest.sha256)
if ! ({
    find . -mindepth 1 -maxdepth 1 -type f -print0
    for source_dir in src scripts tests submodules .holosoma_snapshot; do
      [[ -d "./\${source_dir}" ]] || continue
      find "./\${source_dir}" -type f \
        ! -path './.holosoma_snapshot/source_manifest.sha256' \
        ! -path './.holosoma_snapshot/id' -print0
    done
  } | sort -z | xargs -0 -r sha256sum \
    | cmp -s - .holosoma_snapshot/source_manifest.sha256); then
  echo "[ERROR][${node}] Installed snapshot executable file closure changed after prepare." >&2
  exit 2
fi
if ! ({
    find . -maxdepth 0 -type d -printf 'd\t%m\t%p\0'
    find . -mindepth 1 -maxdepth 1 -type f -printf 'f\t%m\t%p\0'
    for source_dir in src scripts tests submodules; do
      [[ -d "./\${source_dir}" ]] || continue
      find "./\${source_dir}" -type f -printf 'f\t%m\t%p\0'
      find "./\${source_dir}" -type d -printf 'd\t%m\t%p\0'
    done
    find ./.holosoma_snapshot -type d -printf 'd\t%m\t%p\0'
  } | sort -z | cmp -s - .holosoma_snapshot/source_modes.nul); then
  echo "[ERROR][${node}] Installed snapshot source mode closure changed after prepare." >&2
  exit 2
fi
if ({
    find . -maxdepth 0 -type d -perm /222 -print
    for source_dir in src scripts tests submodules .holosoma_snapshot; do
      [[ -d "./\${source_dir}" ]] || continue
      find "./\${source_dir}" -type d -perm /222 -print
    done
  } | grep -q .); then
  echo "[ERROR][${node}] Installed snapshot gained a writable signed source directory after prepare." >&2
  exit 2
fi
if [[ "\$(stat -c '%a' -- .holosoma_snapshot)" != 555 ]]; then
  echo "[ERROR][${node}] Installed snapshot metadata directory is not sealed 0555 after prepare." >&2
  exit 2
fi
for metadata_name in \
    asset_links.tsv source_symlinks.tsv source_modes.nul \
    source_manifest.sha256 id; do
  metadata_path=".holosoma_snapshot/\${metadata_name}"
  if [[ ! -f "\${metadata_path}" || -L "\${metadata_path}" \
        || "\$(stat -c '%a' -- "\${metadata_path}")" != 444 ]]; then
    echo "[ERROR][${node}] Installed snapshot metadata file is not sealed 0444 after prepare: \${metadata_path}" >&2
    exit 2
  fi
done
if find . -mindepth 1 -maxdepth 1 -type d \
    ! -name src ! -name scripts ! -name tests ! -name submodules \
    ! -name .holosoma_snapshot ! -name .checkpoint_cache \
    ! -name .teacher_checkpoints ! -name .run_control ! -name logs \
    -print | grep -q .; then
  echo "[ERROR][${node}] Installed snapshot top-level directory closure changed after prepare." >&2
  exit 2
fi
for runtime_dir in \
    .checkpoint_cache .teacher_checkpoints .run_control logs logs/batch_ne; do
  if [[ ! -d "\${runtime_dir}" || -L "\${runtime_dir}" \
        || "\$(stat -c '%a' -- "\${runtime_dir}")" != 700 ]]; then
    echo "[ERROR][${node}] Installed snapshot runtime directory boundary changed after prepare: \${runtime_dir}" >&2
    exit 2
  fi
done
while IFS=\$'\t' read -r link_path link_target; do
  [[ -n "\${link_path}" ]] || continue
  [[ -L "\${link_path}" && "\$(readlink "\${link_path}")" == "\${link_target}" ]] || {
    echo "[ERROR][${node}] Installed source symlink changed after prepare: \${link_path}" >&2
    exit 2
  }
done < .holosoma_snapshot/source_symlinks.tsv
while IFS=\$'\t' read -r link_path asset_path; do
  [[ -n "\${link_path}" ]] || continue
  expected_target=$(quote "${REMOTE_REPO}")/"\${asset_path}"
  [[ -L "\${link_path}" && "\$(readlink "\${link_path}")" == "\${expected_target}" ]] || {
    echo "[ERROR][${node}] Installed asset symlink changed after prepare: \${link_path}" >&2
    exit 2
  }
done < .holosoma_snapshot/asset_links.tsv
expected_symlink_count=\$((
  \$(wc -l < .holosoma_snapshot/source_symlinks.tsv)
  + \$(wc -l < .holosoma_snapshot/asset_links.tsv)
))
actual_symlink_count=\$({
  find . -mindepth 1 -maxdepth 1 -type l -print
  for source_dir in src scripts tests submodules .holosoma_snapshot; do
    [[ -d "./\${source_dir}" ]] || continue
    find "./\${source_dir}" -type l -print
  done
} | wc -l)
if (( actual_symlink_count != expected_symlink_count )); then
  echo "[ERROR][${node}] Installed snapshot source symlink closure changed after prepare." >&2
  exit 2
fi
if ({
    find . -mindepth 1 -maxdepth 1 \
      \( -type b -o -type c -o -type p -o -type s \) -print
    for source_dir in src scripts tests submodules .holosoma_snapshot; do
      [[ -d "./\${source_dir}" ]] || continue
      find "./\${source_dir}" -xdev \
        \( -type b -o -type c -o -type p -o -type s \) -print
    done
  } | grep -q .); then
  echo "[ERROR][${node}] Installed snapshot gained an unsupported special filesystem entry after prepare." >&2
  exit 2
fi
$(private_lifecycle_file_validation_helpers)
open_private_lifecycle_lock $(quote "${tmux_lock_path}") 8
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 8; then
  echo "[ERROR][${node}] Timed out acquiring the ${SESSION} lifecycle preflight lock." >&2
  exit 1
fi
$(tmux_session_query_helpers)
query_tmux_session_presence $(quote "${SESSION}") || exit
if (( tmux_session_present == 1 )); then
  echo "[ERROR][${node}] tmux session already exists: ${SESSION}" >&2
  exit 1
fi
ACTIVE_STATE=$(quote "${active_state_path}")
if [[ ! -e "\${ACTIVE_STATE}" && ! -L "\${ACTIVE_STATE}" ]]; then
  printf 'ABSENT\n'
  exit 0
fi
$(active_state_validation_helpers)
$(launch_process_closure_helpers)
if ! load_active_state_v2_exact "\${ACTIVE_STATE}" \
    || ! active_state_has_session_namespace $(quote "${SESSION}"); then
  echo "[ERROR][${node}] Launch predecessor active metadata is malformed: \${ACTIVE_STATE}" >&2
  exit 2
fi
case "\${active_phase}" in
  stopped|rolled_back)
    ;;
  *)
    echo "[ERROR][${node}] Refusing launch over nonterminal predecessor phase=\${active_phase} token=\${active_token}." >&2
    exit 2
    ;;
esac
verify_no_launch_token_epoch_processes "\${active_token}" "\${active_epoch}"
predecessor_sha=\$(sha256sum "\${ACTIVE_STATE}" | awk '{print \$1}')
[[ "\${predecessor_sha}" =~ ^[0-9a-f]{64}$ ]] || exit 2
printf 'TERMINAL:%s\n' "\${predecessor_sha}"
EOF
)
  if [[ "${DRY_RUN}" == 1 ]]; then
    remote_run_bounded "${node}" "${cmd}" "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}"
    LAST_LAUNCH_ACTIVE_PREDECESSOR=ABSENT
    return 0
  fi
  local predecessor preflight_rc
  if predecessor=$(remote_run_bounded \
      "${node}" "${cmd}" "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}"); then
    :
  else
    preflight_rc=$?
    return "${preflight_rc}"
  fi
  if [[ "${predecessor}" != ABSENT \
        && ! "${predecessor}" =~ ^TERMINAL:[0-9a-f]{64}$ ]]; then
    echo "[ERROR][${node}] Launch preflight returned malformed predecessor identity: ${predecessor}" >&2
    return 2
  fi
  LAST_LAUNCH_ACTIVE_PREDECESSOR="${predecessor}"
}

mark_launch_state_node() {
  local node="$1"
  local launch_token="$2"
  local launch_epoch="$3"
  local new_phase="$4"
  local active_state_path
  active_state_path="${REMOTE_RUN_ROOT}/.active/$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}')_${node}.state"
  local tmux_lock_path
  tmux_lock_path="${REMOTE_RUN_ROOT}/.active/.locks/holosoma-tmux-$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}').lock"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
$(private_lifecycle_file_validation_helpers)
open_private_lifecycle_lock $(quote "${tmux_lock_path}") 8
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 8; then
  echo "[ERROR][${node}] Timed out acquiring the ${SESSION} lifecycle state lock." >&2
  exit 1
fi
ACTIVE_STATE=$(quote "${active_state_path}")
$(active_state_validation_helpers)
if ! load_active_state_v2_exact "\${ACTIVE_STATE}" \
    || ! active_state_has_session_namespace $(quote "${SESSION}"); then
  echo "[ERROR][${node}] Cannot transition missing or malformed active metadata for this launch." >&2
  exit 2
fi
if [[ "\${active_snapshot}" != $(quote "${SOURCE_SNAPSHOT_ID}") \
      || "\${active_log_dir}" != $(quote "${LOG_DIR}") \
      || "\${active_target}" != $(quote "${TARGET_LEARNING_ITERATION}") \
      || "\${active_token}" != $(quote "${launch_token}") \
      || "\${active_epoch}" != $(quote "${launch_epoch}") ]]; then
  echo "[ERROR][${node}] Refusing lifecycle transition for non-matching launch identity." >&2
  exit 2
fi
case $(quote "${new_phase}"):"\${active_phase}":"\${active_command_sha}" in
  rolling_back:launching:pending|rolling_back:running:*|rolling_back:rolling_back:*)
    ;;
  rolled_back:launching:pending|rolled_back:rolling_back:pending|rolled_back:rolled_back:pending)
    ;;
  *)
    echo "[ERROR][${node}] Refusing unsupported lifecycle transition \${active_phase}->$(quote "${new_phase}") command_sha=\${active_command_sha}." >&2
    exit 2
    ;;
esac
incoming="\${ACTIVE_STATE}.incoming.${launch_token}"
printf '2\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  $(quote "${new_phase}") "\${active_snapshot}" "\${active_log_dir}" "\${active_target}" \
  "\${active_token}" "\${active_command_sha}" "\${active_epoch}" > "\${incoming}"
mv -T "\${incoming}" "\${ACTIVE_STATE}"
validate_private_state_file_metadata "\${ACTIVE_STATE}" 4096 0
EOF
)
  remote_run_bounded "${node}" "${cmd}" "${LAUNCH_CLEANUP_TIMEOUT_SECONDS}"
}

STAGED_CHECKPOINT_REMOTE_PATH=""

stage_verified_control_checkpoint() {
  local node="$1"
  local node_rank="$2"
  local control_path="$3"
  local expected_sha256="$4"
  local cache_kind="$5"
  local label="$6"

  if [[ ! -f "${control_path}" ]]; then
    echo "[ERROR][${node}] Control-local ${label} disappeared before staging: ${control_path}" >&2
    return 2
  fi
  if [[ ! "${expected_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[ERROR][${node}] Invalid expected SHA256 for ${label}: ${expected_sha256}" >&2
    return 2
  fi
  local control_sha256
  control_sha256=$(sha256sum "${control_path}" | awk '{print $1}')
  if [[ "${control_sha256}" != "${expected_sha256}" ]]; then
    echo "[ERROR][${node}] Control-local ${label} changed after preflight: actual=${control_sha256} expected=${expected_sha256} path=${control_path}" >&2
    return 2
  fi

  local remote_dir="${RUN_REPO}/.checkpoint_cache/${cache_kind}"
  local remote_final="${remote_dir}/${expected_sha256}.pt"
  local remote_incoming="${remote_dir}/.${expected_sha256}.${node_rank}.$$.incoming"
  remote_run_bounded "${node}" "set -euo pipefail
mkdir -p $(quote "${remote_dir}")
if [[ -e $(quote "${remote_incoming}") ]]; then
  echo '[ERROR] Refusing to overwrite an existing ${label} incoming file: $(quote "${remote_incoming}")' >&2
  exit 2
fi
" "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}"
  if ! remote_copy_to_bounded \
      "${control_path}" "${node}" "${remote_incoming}" \
      "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}"; then
    remote_run_bounded "${node}" "rm -f $(quote "${remote_incoming}")" "${LAUNCH_CLEANUP_TIMEOUT_SECONDS}" || true
    echo "[ERROR][${node}] Failed to stage ${label}." >&2
    return 2
  fi
  remote_run_bounded "${node}" "set -euo pipefail
CACHE_ROOT=$(quote "${remote_dir}")
INCOMING=$(quote "${remote_incoming}")
FINAL=$(quote "${remote_final}")
EXPECTED=$(quote "${expected_sha256}")
cleanup() { rm -f -- \"\${INCOMING}\"; }
trap cleanup EXIT
test -f \"\${INCOMING}\"
actual=\$(sha256sum \"\${INCOMING}\" | awk '{print \$1}')
if [[ \"\${actual}\" != \"\${EXPECTED}\" ]]; then
  echo \"[ERROR] Staged ${label} SHA256 mismatch: \${INCOMING} actual=\${actual} expected=\${EXPECTED}\" >&2
  exit 2
fi
chmod 0444 \"\${INCOMING}\"
exec 9>\"\${CACHE_ROOT}/.publish-\${EXPECTED}.lock\"
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 9; then
  echo \"[ERROR] Timed out acquiring the ${label} publish lock.\" >&2
  exit 1
fi
if [[ -e \"\${FINAL}\" ]]; then
  if [[ ! -f \"\${FINAL}\" ]]; then
    echo \"[ERROR] Refusing non-file ${label} cache entry: \${FINAL}\" >&2
    exit 2
  fi
  final_sha=\$(sha256sum \"\${FINAL}\" | awk '{print \$1}')
  if [[ \"\${final_sha}\" != \"\${EXPECTED}\" ]]; then
    echo \"[ERROR] Refusing corrupt ${label} cache entry: \${FINAL}\" >&2
    exit 2
  fi
  rm -f -- \"\${INCOMING}\"
  echo \"[INFO][${node}] reused_verified_${cache_kind}=\${FINAL} sha256=\${final_sha}\"
else
  mv -T --no-clobber \"\${INCOMING}\" \"\${FINAL}\"
  if [[ -e \"\${INCOMING}\" ]]; then
    final_sha=\$(sha256sum \"\${FINAL}\" | awk '{print \$1}')
    if [[ \"\${final_sha}\" != \"\${EXPECTED}\" ]]; then
      echo \"[ERROR] Concurrent ${label} publisher produced the wrong SHA256: \${FINAL}\" >&2
      exit 2
    fi
    rm -f -- \"\${INCOMING}\"
    echo \"[INFO][${node}] reused_verified_${cache_kind}=\${FINAL} sha256=\${final_sha}\"
  else
    echo \"[INFO][${node}] installed_verified_${cache_kind}=\${FINAL} sha256=\${EXPECTED}\"
  fi
fi
final_sha=\$(sha256sum \"\${FINAL}\" | awk '{print \$1}')
if [[ \"\${final_sha}\" != \"\${EXPECTED}\" ]]; then
  echo \"[ERROR] Published ${label} SHA256 mismatch: \${FINAL} actual=\${final_sha} expected=\${EXPECTED}\" >&2
  exit 2
fi
trap - EXIT
" "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}"
  STAGED_CHECKPOINT_REMOTE_PATH="${remote_final}"
}

LAST_LAUNCHED_COMMAND_SHA=""

tmux_ownership_helpers() {
  # Emit one remote-shell implementation for every lifecycle consumer.  It is
  # important to distinguish an absent field from a present-but-empty field:
  # empty/removed atomic identity entries are corruption, not a legacy tmux
  # session, and an explicitly empty @option conflicts with every real launch.
  cat <<'EOF'
load_tmux_ownership_identity() {
  local identity_session="$1" environment_dump options_dump line value

  # A failed field lookup is not evidence that a field is absent: tmux may be
  # returning a transport/server/protocol error.  Require two complete,
  # session-scoped dumps first and parse absence only from those successful
  # snapshots.  Any dump failure therefore fails every ownership predicate.
  environment_dump=$(tmux show-environment -t "${identity_session}" 2>/dev/null) || return 1
  options_dump=$(tmux show-options -t "${identity_session}" 2>/dev/null) || return 1

  tmux_env_token_present=0
  tmux_env_token_valid=0
  tmux_env_token_value=""
  tmux_env_command_sha_present=0
  tmux_env_command_sha_valid=0
  tmux_env_command_sha_value=""
  tmux_env_epoch_present=0
  tmux_env_epoch_valid=0
  tmux_env_epoch_value=""
  while IFS= read -r line; do
    case "${line}" in
      HOLOSOMA_LAUNCH_TOKEN=*)
        (( tmux_env_token_present == 0 )) || return 1
        tmux_env_token_present=1
        tmux_env_token_valid=1
        tmux_env_token_value=${line#*=}
        ;;
      -HOLOSOMA_LAUNCH_TOKEN|HOLOSOMA_LAUNCH_TOKEN)
        (( tmux_env_token_present == 0 )) || return 1
        tmux_env_token_present=1
        tmux_env_token_valid=0
        ;;
      HOLOSOMA_COMMAND_SHA256=*)
        (( tmux_env_command_sha_present == 0 )) || return 1
        tmux_env_command_sha_present=1
        tmux_env_command_sha_valid=1
        tmux_env_command_sha_value=${line#*=}
        ;;
      -HOLOSOMA_COMMAND_SHA256|HOLOSOMA_COMMAND_SHA256)
        (( tmux_env_command_sha_present == 0 )) || return 1
        tmux_env_command_sha_present=1
        tmux_env_command_sha_valid=0
        ;;
      HOLOSOMA_LAUNCH_EPOCH=*)
        (( tmux_env_epoch_present == 0 )) || return 1
        tmux_env_epoch_present=1
        tmux_env_epoch_valid=1
        tmux_env_epoch_value=${line#*=}
        ;;
      -HOLOSOMA_LAUNCH_EPOCH|HOLOSOMA_LAUNCH_EPOCH)
        (( tmux_env_epoch_present == 0 )) || return 1
        tmux_env_epoch_present=1
        tmux_env_epoch_valid=0
        ;;
    esac
  done <<<"${environment_dump}"

  tmux_option_token_present=0
  tmux_option_token_value=""
  tmux_option_command_sha_present=0
  tmux_option_command_sha_value=""
  tmux_option_epoch_present=0
  tmux_option_epoch_value=""
  while IFS= read -r line; do
    case "${line}" in
      '@holosoma_launch_token '*)
        (( tmux_option_token_present == 0 )) || return 1
        tmux_option_token_present=1
        value=${line#* }
        tmux_option_token_value=${value}
        ;;
      @holosoma_launch_token)
        return 1
        ;;
      '@holosoma_command_sha256 '*)
        (( tmux_option_command_sha_present == 0 )) || return 1
        tmux_option_command_sha_present=1
        value=${line#* }
        tmux_option_command_sha_value=${value}
        ;;
      @holosoma_command_sha256)
        return 1
        ;;
      '@holosoma_launch_epoch '*)
        (( tmux_option_epoch_present == 0 )) || return 1
        tmux_option_epoch_present=1
        value=${line#* }
        tmux_option_epoch_value=${value}
        ;;
      @holosoma_launch_epoch)
        return 1
        ;;
    esac
  done <<<"${options_dump}"
}

tmux_atomic_identity_matches() {
  local expected_token="$1" expected_command_sha="$2" expected_epoch="$3"
  [[ "${tmux_env_token_present}" == 1 && "${tmux_env_token_valid}" == 1 \
        && "${tmux_env_token_value}" == "${expected_token}" \
        && "${tmux_env_command_sha_present}" == 1 && "${tmux_env_command_sha_valid}" == 1 \
        && "${tmux_env_command_sha_value}" == "${expected_command_sha}" \
        && "${tmux_env_epoch_present}" == 1 && "${tmux_env_epoch_valid}" == 1 \
        && "${tmux_env_epoch_value}" == "${expected_epoch}" ]]
}

tmux_atomic_identity_is_absent() {
  [[ "${tmux_env_token_present}" == 0 \
        && "${tmux_env_command_sha_present}" == 0 \
        && "${tmux_env_epoch_present}" == 0 ]]
}

tmux_options_do_not_conflict() {
  local expected_token="$1" expected_command_sha="$2" expected_epoch="$3"
  [[ ( "${tmux_option_token_present}" == 0 \
          || "${tmux_option_token_value}" == "${expected_token}" ) \
        && ( "${tmux_option_command_sha_present}" == 0 \
          || "${tmux_option_command_sha_value}" == "${expected_command_sha}" ) \
        && ( "${tmux_option_epoch_present}" == 0 \
          || "${tmux_option_epoch_value}" == "${expected_epoch}" ) ]]
}

tmux_options_match_exactly() {
  local expected_token="$1" expected_command_sha="$2" expected_epoch="$3"
  [[ "${tmux_option_token_present}" == 1 \
        && "${tmux_option_token_value}" == "${expected_token}" \
        && "${tmux_option_command_sha_present}" == 1 \
        && "${tmux_option_command_sha_value}" == "${expected_command_sha}" \
        && "${tmux_option_epoch_present}" == 1 \
        && "${tmux_option_epoch_value}" == "${expected_epoch}" ]]
}

tmux_session_has_new_atomic_identity() {
  local identity_session="$1" expected_token="$2" expected_command_sha="$3" expected_epoch="$4"
  load_tmux_ownership_identity "${identity_session}" \
    && tmux_atomic_identity_matches "${expected_token}" "${expected_command_sha}" "${expected_epoch}" \
    && tmux_options_do_not_conflict "${expected_token}" "${expected_command_sha}" "${expected_epoch}"
}

tmux_session_has_complete_new_identity() {
  local identity_session="$1" expected_token="$2" expected_command_sha="$3" expected_epoch="$4"
  load_tmux_ownership_identity "${identity_session}" \
    && tmux_atomic_identity_matches "${expected_token}" "${expected_command_sha}" "${expected_epoch}" \
    && tmux_options_match_exactly "${expected_token}" "${expected_command_sha}" "${expected_epoch}"
}

tmux_session_is_owned_for_cleanup() {
  local identity_session="$1" expected_token="$2" expected_command_sha="$3" expected_epoch="$4"
  load_tmux_ownership_identity "${identity_session}" || return 1
  if tmux_atomic_identity_matches "${expected_token}" "${expected_command_sha}" "${expected_epoch}"; then
    tmux_options_do_not_conflict "${expected_token}" "${expected_command_sha}" "${expected_epoch}"
    return
  fi
  # Backward compatibility is deliberately narrow: all three atomic fields
  # must be truly absent, while all three legacy options must be present and
  # exact.  Partial, removed, empty, or mismatched atomic fields fail closed.
  tmux_atomic_identity_is_absent \
    && tmux_options_match_exactly "${expected_token}" "${expected_command_sha}" "${expected_epoch}"
}
EOF
}

launch_node() {
  local node="$1"
  local node_rank="$2"
  local launch_token="$3"
  local launch_epoch="$4"
  local command_sha_result_file="$5"
  LAST_LAUNCHED_COMMAND_SHA=""
  ensure_local_source_snapshot
  local log_file="${RUN_REPO}/${LOG_DIR}/node_${node_rank}_${node}.log"
  local node_resume_ref="${RESUME_TRAINING_CKPT}"
  local node_teacher_ref="${TEACHER_CHECKPOINT}"
  local node_box_policy_init_ref=""
  if [[ "${RESUME_FROM_BOX}" == 1 ]]; then
    node_box_policy_init_ref="${BOX_POLICY_INIT_REF}"
  fi
  # The training process must not import code from an arbitrary remote login
  # environment.  Keep one exact path contract for the early health probe and
  # the content-addressed tmux payload; the optional overlay is itself bound by
  # its manifest digest below.
  local runtime_pythonpath="${RUN_REPO}/src/holosoma:${RUN_REPO}/src/holosoma_inference:${RUN_REPO}/src"
  if [[ -n "${PYTHON_RUNTIME_SITEPACKAGES}" ]]; then
    runtime_pythonpath="${PYTHON_RUNTIME_SITEPACKAGES}:${runtime_pythonpath}"
  fi
  local runtime_path
  runtime_path="$(dirname -- "${PYTHON_BIN}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  remote_run_bounded "${node}" "set -euo pipefail
unset BASH_ENV ENV CDPATH
unset PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONOPTIMIZE PYTHONWARNINGS PYTHONBREAKPOINT PYTHONSAFEPATH
unset LD_PRELOAD
export PATH=$(quote "${runtime_path}")
export PYTHONPATH=$(quote "${runtime_pythonpath}")
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$(quote "${NCCL_LIB_DIR}")
if [[ ! -d $(quote "${RUN_REPO}") ]]; then
  echo '[ERROR] Isolated source snapshot is not installed on ${node}; run batch_ne.sh prepare or all first.' >&2
  exit 2
fi
cd $(quote "${RUN_REPO}")
grep -Fx -- $(quote "${SOURCE_SNAPSHOT_ID}") .holosoma_snapshot/id >/dev/null
test \"\$(sha256sum .holosoma_snapshot/source_manifest.sha256 | awk '{print \$1}')\" = $(quote "${SOURCE_MANIFEST_SHA256}")
(sha256sum --quiet -c .holosoma_snapshot/source_manifest.sha256)
export PYTHON_BIN=$(quote "${PYTHON_BIN}")
export PYTHONHASHSEED=$(quote "${PYTHONHASHSEED}")
export CUBLAS_WORKSPACE_CONFIG=$(quote "${CUBLAS_WORKSPACE_CONFIG}")
export HOLOSOMA_PYTHON_PROFILE=hssim
export PYTHONDONTWRITEBYTECODE=1
source ./scripts/gpu_launch_defaults.sh
PYTHON_RUNTIME_SITEPACKAGES=$(quote "${PYTHON_RUNTIME_SITEPACKAGES}")
PYTHON_RUNTIME_MANIFEST_SHA256=$(quote "${PYTHON_RUNTIME_MANIFEST_SHA256}")
if [[ -n "\${PYTHON_RUNTIME_SITEPACKAGES}" ]]; then
  $(quote "${PYTHON_BIN}") scripts/verify_python_runtime_overlay.py \
    --site-packages "\${PYTHON_RUNTIME_SITEPACKAGES}" \
    --manifest-sha256 "\${PYTHON_RUNTIME_MANIFEST_SHA256}" \
    --require-distribution-closure \
    --require-current-runtime-binding
  echo \"[INFO][${node}] python_runtime_launch_overlay_verified=\${PYTHON_RUNTIME_SITEPACKAGES} manifest_sha256=\${PYTHON_RUNTIME_MANIFEST_SHA256}\"
fi
" "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}"
  if [[ "${SKIP_NODE_HEALTH_CHECK}" != "1" ]]; then
    remote_run_bounded "${node}" "set -euo pipefail
unset BASH_ENV ENV CDPATH
unset PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONOPTIMIZE PYTHONWARNINGS PYTHONBREAKPOINT PYTHONSAFEPATH
unset LD_PRELOAD
export PATH=$(quote "${runtime_path}")
export PYTHONPATH=$(quote "${runtime_pythonpath}")
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$(quote "${NCCL_LIB_DIR}")
cd $(quote "${RUN_REPO}")
grep -Fx -- $(quote "${SOURCE_SNAPSHOT_ID}") .holosoma_snapshot/id >/dev/null
(sha256sum --quiet -c .holosoma_snapshot/source_manifest.sha256)
test -f batch_ne.sh
test -f scripts/build_run_snapshot.sh
test -f scripts/resolve_exact_checkpoint.py
test -f scripts/compute_training_provenance.py
test -f src/holosoma/holosoma/utils/provenance_preflight.py
LOGGER_BASE_DIR=$(quote "${LOGGER_BASE_DIR}")
mkdir -p -- \"\${LOGGER_BASE_DIR}\"
if [[ ! -d \"\${LOGGER_BASE_DIR}\" || ! -w \"\${LOGGER_BASE_DIR}\" ]]; then
  echo \"[ERROR] LOGGER_BASE_DIR is not a writable directory: \${LOGGER_BASE_DIR}\" >&2
  exit 2
fi
LOGGER_WRITE_PROBE=\$(mktemp \"\${LOGGER_BASE_DIR}/.holosoma-health-write-probe.XXXXXX\")
rm -f -- \"\${LOGGER_WRITE_PROBE}\"
echo \"[INFO][${node}] logger_base_dir_health_verified=\${LOGGER_BASE_DIR}\"
ip link show $(quote "${NCCL_SOCKET_IFNAME}") >/dev/null
ip link show $(quote "${GLOO_SOCKET_IFNAME}") >/dev/null
export PYTHON_BIN=$(quote "${PYTHON_BIN}")
export PYTHONHASHSEED=$(quote "${PYTHONHASHSEED}")
export CUBLAS_WORKSPACE_CONFIG=$(quote "${CUBLAS_WORKSPACE_CONFIG}")
export HOLOSOMA_PYTHON_PROFILE=hssim
export PYTHONDONTWRITEBYTECODE=1
source ./scripts/gpu_launch_defaults.sh
PYTHON_RUNTIME_SITEPACKAGES=$(quote "${PYTHON_RUNTIME_SITEPACKAGES}")
PYTHON_RUNTIME_MANIFEST_SHA256=$(quote "${PYTHON_RUNTIME_MANIFEST_SHA256}")
if [[ -n \"\${PYTHON_RUNTIME_SITEPACKAGES}\" ]]; then
  \"\${PYTHON_BIN}\" scripts/verify_python_runtime_overlay.py \
    --site-packages \"\${PYTHON_RUNTIME_SITEPACKAGES}\" \
    --manifest-sha256 \"\${PYTHON_RUNTIME_MANIFEST_SHA256}\" \
    --require-distribution-closure \
    --require-current-runtime-binding
  runtime_manifest=\"\${PYTHON_RUNTIME_SITEPACKAGES}/.holosoma-runtime-manifest.sha256\"
  test -f \"\${runtime_manifest}\"
  actual_runtime_manifest_sha256=\$(sha256sum \"\${runtime_manifest}\" | awk '{print \$1}')
  if [[ \"\${actual_runtime_manifest_sha256}\" != \"\${PYTHON_RUNTIME_MANIFEST_SHA256}\" ]]; then
    echo \"[ERROR] Python runtime manifest SHA256 mismatch: actual=\${actual_runtime_manifest_sha256} expected=\${PYTHON_RUNTIME_MANIFEST_SHA256}\" >&2
    exit 2
  fi
  (cd \"\${PYTHON_RUNTIME_SITEPACKAGES}\" && sha256sum --quiet -c .holosoma-runtime-manifest.sha256)
  export HOLOSOMA_PYTHON_RUNTIME_MANIFEST_SHA256=\"\${PYTHON_RUNTIME_MANIFEST_SHA256}\"
  echo \"[INFO][${node}] python_runtime_overlay_verified=\${PYTHON_RUNTIME_SITEPACKAGES} manifest_sha256=\${PYTHON_RUNTIME_MANIFEST_SHA256}\"
fi
NCCL_RUNTIME_LIB=$(quote "${NCCL_LIB_DIR}/libnccl.so.2")
NCCL_RUNTIME_EXPECTED_SHA256=$(quote "${NCCL_LIB_SHA256}")
if [[ -n "\${NCCL_RUNTIME_EXPECTED_SHA256}" ]]; then
  if [[ ! -f "\${NCCL_RUNTIME_LIB}" ]]; then
    echo \"[ERROR] Required NCCL runtime library is not a regular file: \${NCCL_RUNTIME_LIB}\" >&2
    exit 2
  fi
  actual_nccl_lib_sha256=\$(sha256sum "\${NCCL_RUNTIME_LIB}" | awk '{print \$1}')
  if [[ "\${actual_nccl_lib_sha256}" != "\${NCCL_RUNTIME_EXPECTED_SHA256}" ]]; then
    echo \"[ERROR] NCCL runtime SHA256 mismatch: \${NCCL_RUNTIME_LIB} actual=\${actual_nccl_lib_sha256} expected=\${NCCL_RUNTIME_EXPECTED_SHA256}\" >&2
    exit 2
  fi
  echo \"[INFO][${node}] nccl_runtime_health_sha256_verified=\${actual_nccl_lib_sha256} path=\${NCCL_RUNTIME_LIB}\"
fi
if [[ $(quote "${TORCH_DIST_BACKEND}") == nccl || $(quote "${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE}") == 1 ]]; then
  export LD_PRELOAD=$(quote "${NCCL_LIB_DIR}/libnccl.so.2")
fi
SOURCE_BANK=$(quote "${RUN_REPO}/data/ds_as_data/${CORL_SOLID80_BANK_NAME}")
CONTACT_ROOT="\${SOURCE_BANK}/$(quote "${SOLID_CONTACT_EXPORT_NAME}")"
test -d "\${SOURCE_BANK}"
test -f "\${SOURCE_BANK}/_clip_object_urdf_map.json"
test -d "\${CONTACT_ROOT}"
CONTACT_VALIDATOR_RUNTIME_ARGS=()
if [[ $(quote "${CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION}") == True ]]; then
  CONTACT_VALIDATOR_RUNTIME_ARGS+=(--runtime-prepend-compensation --runtime-prepend-duration-s 0.2)
fi
"\${PYTHON_BIN}" scripts/validate_contact_sidecars.py \
  --motion-dir "\${SOURCE_BANK}" \
  --contact-root "\${CONTACT_ROOT}" \
  --motion-end-mode $(quote "${STUDENT_MOTION_END_MODE}") \
  "\${CONTACT_VALIDATOR_RUNTIME_ARGS[@]}" \
  --expected-total $(quote "${OMOMO_EXPECTED_TOTAL}") >/dev/null
echo '[INFO] node_data_sidecar_health_check_ok bank='"\${SOURCE_BANK}"' clips='$(quote "${OMOMO_EXPECTED_TOTAL}")
" "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}"
  fi
  # Mandatory even when SKIP_NODE_HEALTH_CHECK=1: a second read-only inventory
  # check closes the interval between the early fleet gate and this node's
  # actual launch payload staging.  Optional repo/logger/data probes above may
  # be skipped, but launching onto a selected busy/protected GPU may not.
  preflight_selected_gpus_idle_node "${node}" pre-launch
  if [[ -n "${CONTROL_RESUME_TRAINING_PATH}" ]]; then
    stage_verified_control_checkpoint \
      "${node}" "${node_rank}" \
      "${CONTROL_RESUME_TRAINING_PATH}" "${CONTROL_RESUME_TRAINING_SHA256}" \
      training_resume training-resume-checkpoint
    node_resume_ref="${STAGED_CHECKPOINT_REMOTE_PATH}"
  fi
  if [[ -n "${CONTROL_TEACHER_CHECKPOINT_PATH}" ]]; then
    stage_verified_control_checkpoint \
      "${node}" "${node_rank}" \
      "${CONTROL_TEACHER_CHECKPOINT_PATH}" "${CONTROL_TEACHER_CHECKPOINT_SHA256}" \
      teacher teacher-checkpoint
    node_teacher_ref="${STAGED_CHECKPOINT_REMOTE_PATH}"
  fi
  if [[ "${RESUME_FROM_BOX}" == "1" && -n "${CONTROL_BOX_POLICY_INIT_PATH}" ]]; then
    local remote_policy_init_dir="${RUN_REPO}/.checkpoint_cache/policy_init"
    local remote_policy_init_path="${remote_policy_init_dir}/${CONTROL_BOX_POLICY_INIT_SHA256}.pt"
    local remote_policy_init_incoming="${remote_policy_init_dir}/.${CONTROL_BOX_POLICY_INIT_SHA256}.${node_rank}.$$.incoming"
    remote_run_bounded "${node}" "set -euo pipefail
mkdir -p $(quote "${remote_policy_init_dir}")
if [[ -e $(quote "${remote_policy_init_incoming}") ]]; then
  echo '[ERROR] Refusing to overwrite an existing policy-init incoming file: $(quote "${remote_policy_init_incoming}")' >&2
  exit 2
fi
" "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}"
    if ! remote_copy_to_bounded \
        "${CONTROL_BOX_POLICY_INIT_PATH}" "${node}" "${remote_policy_init_incoming}" \
        "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}"; then
      remote_run_bounded "${node}" "rm -f $(quote "${remote_policy_init_incoming}")" "${LAUNCH_CLEANUP_TIMEOUT_SECONDS}" || true
      echo "[ERROR][${node}] Failed to stage policy-init checkpoint." >&2
      return 2
    fi
    remote_run_bounded "${node}" "set -euo pipefail
CACHE_ROOT=$(quote "${remote_policy_init_dir}")
INCOMING=$(quote "${remote_policy_init_incoming}")
FINAL=$(quote "${remote_policy_init_path}")
EXPECTED=$(quote "${CONTROL_BOX_POLICY_INIT_SHA256}")
cleanup() { rm -f \"\${INCOMING}\"; }
trap cleanup EXIT
test -f \"\${INCOMING}\"
actual=\$(sha256sum \"\${INCOMING}\" | awk '{print \$1}')
if [[ \"\${actual}\" != \"\${EXPECTED}\" ]]; then
  echo \"[ERROR] Staged policy-init SHA256 mismatch: \${INCOMING} actual=\${actual} expected=\${EXPECTED}\" >&2
  exit 2
fi
chmod a-w \"\${INCOMING}\"
exec 9>\"\${CACHE_ROOT}/.publish-\${EXPECTED}.lock\"
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 9; then
  echo \"[ERROR] Timed out acquiring the policy-init publish lock.\" >&2
  exit 1
fi
if [[ -e \"\${FINAL}\" ]]; then
  if [[ ! -f \"\${FINAL}\" ]]; then
    echo \"[ERROR] Refusing non-file policy-init cache entry: \${FINAL}\" >&2
    exit 2
  fi
  final_sha=\$(sha256sum \"\${FINAL}\" | awk '{print \$1}')
  if [[ \"\${final_sha}\" != \"\${EXPECTED}\" ]]; then
    echo \"[ERROR] Refusing corrupt policy-init cache entry: \${FINAL}\" >&2
    exit 2
  fi
  rm -f \"\${INCOMING}\"
  echo \"[INFO][${node}] reused_verified_policy_init=\${FINAL}\"
else
  mv -T --no-clobber \"\${INCOMING}\" \"\${FINAL}\"
  if [[ -e \"\${INCOMING}\" ]]; then
    final_sha=\$(sha256sum \"\${FINAL}\" | awk '{print \$1}')
    if [[ \"\${final_sha}\" != \"\${EXPECTED}\" ]]; then
      echo \"[ERROR] Concurrent policy-init publisher produced the wrong SHA256: \${FINAL}\" >&2
      exit 2
    fi
    rm -f \"\${INCOMING}\"
    echo \"[INFO][${node}] reused_verified_policy_init=\${FINAL}\"
  else
    echo \"[INFO][${node}] installed_verified_policy_init=\${FINAL}\"
  fi
fi
final_sha=\$(sha256sum \"\${FINAL}\" | awk '{print \$1}')
test \"\${final_sha}\" = \"\${EXPECTED}\"
trap - EXIT
" "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}"
    node_box_policy_init_ref="${remote_policy_init_path}"
  fi
  local env_exports
  env_exports=$(cat <<EOF
export CUDA_VISIBLE_DEVICES=$(quote "${CUDA_VISIBLE_DEVICES}")
# A non-interactive ssh/tmux shell can still inherit process-startup and Python
# startup hooks from the node service.  Child wrappers are separate Bash
# processes, so clear the hooks here before any delegated launcher is invoked.
unset BASH_ENV ENV CDPATH
unset PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONOPTIMIZE PYTHONWARNINGS PYTHONBREAKPOINT PYTHONSAFEPATH
# HOLOSOMA_ORIGINAL_* is worker-produced rank-visible metadata.  Never inherit
# aliases from a controller/login shell into a new torchrun topology.
unset HOLOSOMA_ORIGINAL_LOCAL_RANK
unset HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE
unset HOLOSOMA_ORIGINAL_CUDA_VISIBLE_DEVICES
# A remote login shell may retain aliases from an older run.  Downstream
# launchers intentionally support these aliases for direct/manual use, so an
# explicit empty canonical value alone would still fall back to stale state.
# Clear every resume/policy-init alias before exporting this launch's one
# canonical contract.  The verified box-policy path is re-exported below only
# after RESUME_FROM_BOX validation succeeds.
unset RESUME_TRAINING_CKPT
unset RESUME_CHECKPOINT
unset RESUME_WANDB_ID
unset WANDB_RUN_ID
unset WANDB_RESUME
unset POLICY_INIT_CKPT
unset POLICY_INIT_CHECKPOINT
unset POLICY_INIT_SOURCE_REF
unset BOX_RESUME_CKPT
unset RESUME_FROM_BOX_CKPT
unset DEFAULT_BOX_RESUME_RUN
unset DEFAULT_BOX_RESUME_MODEL_FILE
unset BOX_RESUME_MODEL_FILE
unset DEFAULT_BOX_RESUME_CHECKPOINT
unset RESUME_FROM_PREVIOUS
unset PREVIOUS_RESUME_CKPT
unset RESUME_FROM_PREVIOUS_CKPT
unset PREVIOUS_RESUME_RUN
unset PREVIOUS_RESUME_MODEL_FILE
unset DEFAULT_PREVIOUS_RESUME_RUN
unset PREVIOUS_POLICY_INIT_CACHE_ROOT
unset AS_POLICY_INIT_PROFILE
unset POLICY_INIT_EXPECTED_SHA256
unset BOX_POLICY_INIT_EXPECTED_SHA256
unset POLICY_INIT_CACHE_ROOT
unset HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET
unset BOX_POLICY_INIT_CONTROL_CACHE_ROOT
unset BOX_POLICY_INIT_EXPECTED_WORLD_SIZE
unset BOX_POLICY_INIT_EXPECTED_WANDB_RUN_PATH
unset BOX_POLICY_INIT_EXPECTED_SOURCE_SNAPSHOT_ID
unset BOX_POLICY_INIT_REQUIRE_FRESH_SOURCE
unset RESUME_MODEL_FILE
unset WANDB_MODEL_FILE
unset RESUME_STEP
unset DISTILL_AS_ENTRYPOINT
unset DISTILL_AS_ENTRYPOINT_PATH
unset DISTILL_AS_ENTRYPOINT_SHA256
unset DISTILL_AS_FORMAL_FRESH
# W&B 0.26 accepts a broad and evolving WANDB_*/_WANDB_* settings surface.
# A stale WANDB_LAUNCH or sweep/service/private setting can override even
# explicit wandb.init identity fields.  Preserve only node-local credential
# sources, erase every other public/private W&B setting, then export the
# controller's canonical scientific contract below.
for _wandb_env_name in \${!WANDB_@} \${!_WANDB_@}; do
  case "\${_wandb_env_name}" in
    WANDB_API_KEY|WANDB_IDENTITY_TOKEN_FILE|WANDB_CREDENTIALS_FILE)
      ;;
    *)
      unset "\${_wandb_env_name}"
      ;;
  esac
done
unset _wandb_env_name
unset WANDB_NAME
unset WANDB_MODE
unset WANDB_DISABLED
unset WANDB_BASE_URL
unset WANDB_DIR
unset WANDB_INIT_TIMEOUT
unset WANDB_CONSOLE
unset WANDB_RUN_GROUP
unset WANDB_TAGS
unset WANDB_SWEEP_ID
unset HOLOSOMA_REQUIRE_WANDB_RUN
unset LOGGER
# The solid-AS wrapper treats these as direct/manual data-source overrides.
# They must not leak from a login shell and supersede the controller's exact
# CORL_SOLID80_BANK_NAME/contact contract.
unset AS_SUCCESS133_BANK_NAME
unset OMOMO_DATA_DIR
unset OMOMO_OBJECT_MAP
unset AS_CONTACT_EXPORT_ROOT
unset AS_RESUME_DATA_DIR
unset AS_RESUME_OBJECT_MAP
unset AS_TRAINING_RESUME_REF
unset AS_TRAINING_RESUME_CACHE_ROOT
unset AS_SINGLE_SLOT_MOTION_BASE
unset AS_SINGLE_SLOT_MOTION_DIR
unset AS_RANK_SHARD_ROOT
unset HOLOSOMA_RANK_LOCAL_MOTION_ROOT
unset HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED
unset DEFAULT_RESUME_FROM_BOX_AS_BANK
unset RESUME_FROM_BOX_AS_DATA_DIR
unset RESUME_FROM_BOX_AS_OBJECT_MAP
unset RESUME_FROM_BOX_CONTACT_EXPORT_ROOT
# These process-wide switches alter training observations, reset/data
# distributions, or numerical execution, but are not supported overrides of
# this fixed scientific launcher.  ssh does not forward the controller's
# arbitrary environment, so inheriting a node-local profile value would make
# otherwise identical snapshots run different experiments.  Force the
# documented defaults before downstream wrappers set their own exact values.
unset HOLOSOMA_EVAL_DISABLE_ROLLOUT_REFERENCE_REWARDS
unset HOLOSOMA_EVAL_ALLOW_MISSING_CAMERA_SENSOR_NOISE_STATE
unset HOLOSOMA_EVAL_DEBUG_PATH
unset HOLOSOMA_EVAL_DEBUG_LIMIT
unset HOLOSOMA_ALLOW_LEGACY_OBJECT_URDF_FALLBACK
unset HOLOSOMA_ALLOW_LEGACY_ROLLOUT_RESTART_RESUME
unset HOLOSOMA_ALLOW_LEGACY_UNPROVENANCED_RESUME
unset HOLOSOMA_ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD
unset HOLOSOMA_ALLOW_LEGACY_UNVERIFIED_TEACHER_LOAD
unset HOLOSOMA_CLEAN_ROBOT_USD_CACHE
unset HOLOSOMA_DISABLE_HETEROGENEOUS_OBJECT_SINGLE_SLOT
unset HOLOSOMA_FORCE_HETEROGENEOUS_OBJECT_SINGLE_SLOT
unset HOLOSOMA_PROVENANCE_TIMEOUT_SEC
unset HOLOSOMA_DEBUG_ACTOR_ALL
unset HOLOSOMA_DEBUG_GRAD_REDUCE
unset HOLOSOMA_DEBUG_STATE_SYNC
unset HOLOSOMA_DEBUG_TRAINING_PHASES
unset HOLOSOMA_DEBUG_TRAINING_PHASE_DIR
unset HOLOSOMA_FORCE_RICH_LIVE_LOGGING
unset HOLOSOMA_STEP_TIMING
unset HOLOSOMA_STEP_TIMING_PROFILE
unset HOLOSOMA_STEP_TIMING_SYNC_CUDA
unset HOLOSOMA_STEP_TIMING_INTERVAL
unset HOLOSOMA_COLLECTION_PROFILE_CANARY
unset HOLOSOMA_COLLECTION_PROFILE_SYNC_CUDA
unset HOLOSOMA_COLLECTION_PROFILE_INTERVAL
unset HOLOSOMA_CAMERA_LOG_ROOT_BACK_EVERY
unset HOLOSOMA_CAMERA_WARN_ROOT_BACK_RATIO
unset HOLOSOMA_CAMERA_AUTOFIX_BACKWARD
unset HOLOSOMA_CAMERA_BACKWARD_RATIO_THRESHOLD
unset HOLOSOMA_CAMERA_DISABLE_OFFSETS
unset HOLOSOMA_CAMERA_EXTRA_YAW_DEG
unset HOLOSOMA_CAMERA_RANDOMIZE_PLACEMENT
unset HOLOSOMA_CAMERA_STRICT_WARP
unset HOLOSOMA_DEFM_FORWARD_BATCH_SIZE
unset HOLOSOMA_DEFAULT_POSE_INIT
unset HOLOSOMA_DISABLE_ACTIVE_OBS_GROUP_FILTER
unset HOLOSOMA_DISABLE_AUTO_RESET
unset HOLOSOMA_DISABLE_BAD_TRACKING_RESET
unset HOLOSOMA_DISABLE_CLIP_END_RESET
unset HOLOSOMA_DISABLE_MOTION_END_RESET
unset HOLOSOMA_DISABLE_ONLINE_CONTACT_PRIOR
unset HOLOSOMA_FAR_TRACKING_DISABLE_COMBINED_DEPTH_MESHES
unset HOLOSOMA_FORCE_ROUND_ROBIN_CLIP_ASSIGNMENT
unset HOLOSOMA_FORCE_ROUND_ROBIN_CLIP_START
unset HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE
unset HOLOSOMA_MUJOCO_RESET_NOISE
unset HOLOSOMA_ONLINE_CONTACT_PRIOR
unset HOLOSOMA_PERCEPTION_INCLUDE_ROBOT_MESH
unset HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES
unset HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE
unset HOLOSOMA_PERCEPTION_SENSOR_OFFSET_DELTA
unset HOLOSOMA_PERCEPTION_SENSOR_OFFSET_OVERRIDE
unset HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS
unset HOLOSOMA_RESET_TO_DEFAULT_POSE
unset HOLOSOMA_STRICT_PERCEPTION_CAMERA_SOURCE
unset HOLOSOMA_STRICT_PERCEPTION_OBJECT_MESHES
unset ISAAC_SCANDOTS_INCLUDE_MISSES
unset ISAAC_SCANDOTS_USE_DEPTH_MASK
unset HOLOSOMA_SYNC_EACH_ITERATION
# Delegated AS/box launchers intentionally retain environment-variable
# defaults for direct interactive use.  None of the following are supported
# node-local overrides of this fixed batch experiment.  Clear them before the
# canonical exports below; values that this controller supports explicitly are
# re-exported later from controller-validated state.
unset AS_CONTACT_AWARE
unset AS_CONTACT_AWARE_HISTORY
unset CONTACT_AWARE
unset CONTACT_AWARE_HISTORY
unset CONTACT_AWARE_HISTORY_LENGTH
unset CONTACT_AWARE_CARRY_WINDOW_MODE
unset CONTACT_AWARE_PEAK_HEIGHT_ALPHA
unset CONTACT_AWARE_PEAK_HEIGHT_SMOOTHING_STEPS
unset CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE
unset CONTACT_AWARE_SPARSE_ROOT_SEGMENT_STEPS
unset CONTACT_AWARE_SPARSE_ROOT_ZERO_YAW_THRESHOLD_DEG
unset CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION
unset STUDENT_ACTOR_INPUTS
unset STUDENT_PROPRIO_HISTORY_LENGTH
unset STUDENT_ACTION_HISTORY_LENGTH
unset CRITIC_PROPRIO_HISTORY_LENGTH
unset TEACHER_ACTOR_OBS_HISTORY_LENGTH
unset FORCE_EIGHT_GPU_CONFIG
unset CORL_128
unset DATA_MODE
unset EXP
unset ROOT_COMMAND_MODE
unset SCHEDULE_VARIANT
unset SHOO7SR1_NEAR03_DEBUG
unset SHOO7SR1_OBS_VARIANT
unset TEACHER_COMPAT_PROFILE
unset TEACHER_OBS_KEYS
unset TEACHER_PERCEPTION_PRESET
unset TEACHER_PERCEPTION_OBS_KEY
unset TRACKER_PROFILE
unset PERCEPTION_PRESET
unset CRITIC_PERCEPTION_PRESET
unset CRITIC_PERCEPTION_OBS_KEY
unset PERCEPTION_INTO_POLICY_MODULES
unset PERCEPTION_INTO_CRITIC_MODULES
unset DISTILL_MODE
unset DISTILL_LOSS_TYPE
unset DISTILL_ENABLED
unset BC_LOSS_COEF
unset ACTOR_LR
unset CRITIC_LR
unset PPO_LR_SCHEDULE
unset PPO_DESIRED_KL
unset ACTOR_MIN_LR
unset ACTOR_MAX_LR
unset CRITIC_MIN_LR
unset CRITIC_MAX_LR
unset SWITCH_TO_RL_AFTER
unset TEACHER_ACTION_MIX_RATIO
unset TEACHER_ACTION_MIX_RATIO_START
unset TEACHER_ACTION_MIX_RATIO_END
unset TEACHER_ACTION_MIX_RATIO_END_ITERATION
unset DAGGER_REPLAY_ENABLED
unset DAGGER_REPLAY_CAPACITY
unset DAGGER_REPLAY_BATCH_SIZE
unset DAGGER_REPLAY_FRACTION
unset DAGGER_REPLAY_SEED
unset CLIP_TEACHER_ACTIONS
unset CLIP_ACTIONS_THRESHOLD
unset USE_ADAPTIVE_TIMESTEPS_SAMPLER
unset START_AT_TIMESTEP_ZERO_PROB
unset START_AT_TIMESTEP_ZERO_PROB_END
unset START_AT_TIMESTEP_ZERO_PROB_START_ITER
unset START_AT_TIMESTEP_ZERO_PROB_END_ITER
unset FREEZE_AT_TIMESTEP_ZERO_PROB
unset FREEZE_AT_TIMESTEP_ZERO_PROB_END
unset FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER
unset FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER
unset UNIFORM_T1_WINDOW_SAMPLING_ENABLED
unset UNIFORM_T1_WINDOW_HALF_WIDTH_STEPS
unset UNIFORM_T1_WINDOW_DENSITY_BOOST
unset PAIR_TERRAIN_WITH_MOTION
unset DAGGER_IGNORE_EPISODE_INITIAL_STEPS
unset MAX_EPISODE_LENGTH_S
unset RESET_TO_DEFAULT_POSE
unset ENABLE_DEFAULT_POSE_PREPEND
unset DEFAULT_POSE_PREPEND_DURATION_S
unset ENABLE_DEFAULT_POSE_APPEND
unset DEFAULT_POSE_APPEND_DURATION_S
unset BAD_TRACKING_THRESHOLD_AUGMENT
unset BAD_TRACKING_THRESHOLD_MULTIPLIER
unset BAD_TRACKING_THRESHOLD_SCALE
unset IMAGE_WIDTH
unset IMAGE_HEIGHT
unset CAMERA_PITCH_DEG
unset CAMERA_FAR
unset CAMERA_MAX_DISTANCE
unset CAMERA_APPLY_SENSOR_NOISE
unset CAMERA_WARP_FREQ_RATIO
unset CAMERA_WARP_EDGE_NOISE
unset CAMERA_WARP_ENABLE_HOLES
unset CAMERA_WARP_HOLE_PROB
unset CAMERA_WARP_ADDITIVE_NOISE_STD
unset CAMERA_WARP_DEPTH_OFFSET_STD
unset PERCEPTION_WARP_PREPROCESS
unset AS_PUSH_INTERVAL_S
unset AS_MAX_PUSH_VEL
unset HOLOSOMA_OBJECT_COLLIDER_TYPE
unset HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS
unset HOLOSOMA_REQUIRE_OBJECT_MESH_ASSETS
unset HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY
unset HOLOSOMA_DEBUG_TILE_LAYOUT
unset HOLOSOMA_W_OBJECT_URDF
unset HOLOSOMA_ISAACSIM_KIT_ARGS
unset ISAACSIM_KIT_ARGS
unset CHECK_ONLY
unset DRY_RUN
unset LD_PRELOAD
# This controller owns the fixed scientific reset/perception semantics.  Bind
# both launcher aliases and the process-wide variables before AS provenance is
# computed; node-local shell profiles must not select a different experiment.
export PERCEPTION_INTO_POLICY_MODULES=True
export RESET_TO_DEFAULT_POSE=False
export HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES=True
export HOLOSOMA_RESET_TO_DEFAULT_POSE=False
export HOLOSOMA_SOURCE_SNAPSHOT_ID=$(quote "${SOURCE_SNAPSHOT_ID}")
export HOLOSOMA_SOURCE_MANIFEST_SHA256=$(quote "${SOURCE_MANIFEST_SHA256}")
export DISTILL_AS_ENTRYPOINT=$(quote "${DISTILL_AS_ENTRYPOINT}")
export DISTILL_AS_ENTRYPOINT_PATH=$(quote "${DISTILL_AS_ENTRYPOINT_PATH}")
export DISTILL_AS_ENTRYPOINT_SHA256=$(quote "${DISTILL_AS_ENTRYPOINT_SHA256}")
export DISTILL_AS_FORMAL_FRESH=$(quote "${DISTILL_AS_FORMAL_FRESH}")
export HOLOSOMA_LAUNCH_TOKEN=$(quote "${launch_token}")
export HOLOSOMA_LAUNCH_EPOCH=$(quote "${launch_epoch}")
export SESSION=$(quote "${SESSION}")
export RUN_STAMP=$(quote "${RUN_STAMP}")
export HOLOSOMA_ACTIVE_LOG_DIR=$(quote "${LOG_DIR}")
export HOLOSOMA_ASSET_REPO=$(quote "${REMOTE_REPO}")
export PYTHON_BIN=$(quote "${PYTHON_BIN}")
export PYTHONHASHSEED=$(quote "${PYTHONHASHSEED}")
export CUBLAS_WORKSPACE_CONFIG=$(quote "${CUBLAS_WORKSPACE_CONFIG}")
export HOLOSOMA_PYTHON_PROFILE=hssim
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHON_RUNTIME_SITEPACKAGES=$(quote "${PYTHON_RUNTIME_SITEPACKAGES}")
export PYTHON_RUNTIME_MANIFEST_SHA256=$(quote "${PYTHON_RUNTIME_MANIFEST_SHA256}")
export HOLOSOMA_PYTHON_RUNTIME_MANIFEST_SHA256=$(quote "${PYTHON_RUNTIME_MANIFEST_SHA256}")
export PYTHONPATH=$(quote "${runtime_pythonpath}")
export PATH=$(quote "${runtime_path}")
export LOGGER_BASE_DIR=$(quote "${LOGGER_BASE_DIR}")
export OMNI_KIT_ACCEPT_EULA=$(quote "${OMNI_KIT_ACCEPT_EULA}")
export ACCEPT_EULA=$(quote "${ACCEPT_EULA}")
export NPROC=$(quote "${NPROC}")
export NNODES=$(quote "${NNODES}")
export NODE_RANK=$(quote "${node_rank}")
export MASTER_ADDR=$(quote "${MASTER_ADDR}")
export MASTER_PORT=$(quote "${MASTER_PORT}")
export HOLOSOMA_PROVENANCE_MASTER_PORT=$(quote "${HOLOSOMA_PROVENANCE_MASTER_PORT}")
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
export NCCL_LIB_SHA256=$(quote "${NCCL_LIB_SHA256}")
export LD_LIBRARY_PATH=$(quote "${NCCL_LIB_DIR}")
export PER_GPU_ENVS=$(quote "${PER_GPU_ENVS}")
export TRAINING_SEED=$(quote "${TRAINING_SEED}")
unset SEED
export CH_BANK_NAME=$(quote "${CH_BANK_NAME}")
export CORL_SOLID80_BANK_NAME=$(quote "${CORL_SOLID80_BANK_NAME}")
export SOLID_CLIP_LIST=$(quote "${SOLID_CLIP_LIST}")
export SOLID_TARGET_BANK_NAME=$(quote "${SOLID_TARGET_BANK_NAME}")
export SOLID_ALLOWED_OBJECT_CATEGORIES=$(quote "${SOLID_ALLOWED_OBJECT_CATEGORIES}")
export SOLID_CONTACT_EXPORT_NAME=$(quote "${SOLID_CONTACT_EXPORT_NAME}")
export ENABLE_OFFLINE_CONTACT_GUIDANCE=$(quote "${ENABLE_OFFLINE_CONTACT_GUIDANCE}")
export AS_SUCCESS133_FINAL0P5=1
export AS_RANK_LOCAL_SHARDS=1
export AS_SINGLE_SLOT_MOTION_BASE=$(quote "${AS_EXTERNAL_SINGLE_SLOT_DIR%/by-source/*}")
export HOLOSOMA_EXTERNAL_AS_MOTION_CLIP_ID=$(quote "${AS_EXTERNAL_MOTION_CLIP_ID}")
export HOLOSOMA_EXTERNAL_AS_SOLID_SOURCE_DIGEST=$(quote "${AS_EXTERNAL_SOLID_SOURCE_DIGEST}")
export HOLOSOMA_EXTERNAL_AS_SELECTED_CLIP_COUNT=$(quote "${OMOMO_EXPECTED_TOTAL}")
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST=$(quote "${AS_EXTERNAL_SINGLE_SLOT_SOURCE_DIGEST}")
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST=$(quote "${AS_EXTERNAL_SINGLE_SLOT_VIEW_DIGEST}")
export HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST=$(quote "${AS_EXTERNAL_RANK_SHARD_SOURCE_DIGEST}")
export HOLOSOMA_EXTERNAL_AS_MOTION_NPZ_SHA256=$(quote "${AS_EXTERNAL_MOTION_NPZ_SHA256}")
export HOLOSOMA_EXTERNAL_AS_OBJECT_MAP_SHA256=$(quote "${AS_EXTERNAL_OBJECT_MAP_SHA256}")
export HOLOSOMA_EXTERNAL_AS_OBJECT_URDF_SHA256=$(quote "${AS_EXTERNAL_OBJECT_URDF_SHA256}")
export HOLOSOMA_EXTERNAL_AS_OBJECT_MESH_SHA256=$(quote "${AS_EXTERNAL_OBJECT_MESH_SHA256}")
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_DIR=$(quote "${AS_EXTERNAL_SINGLE_SLOT_DIR}")
export HOLOSOMA_EXTERNAL_AS_WORLD_SIZE=$(quote "${TOTAL_GPUS}")
export HOLOSOMA_EXTERNAL_AS_MOTION_GENERATOR_TEACHER_SHA256=$(quote "${AS_EXTERNAL_MOTION_GENERATOR_TEACHER_SHA256}")
export REQUIRE_MOTION_GENERATOR_TEACHER_MATCH=$(quote "${REQUIRE_MOTION_GENERATOR_TEACHER_MATCH}")
export RESUME_FROM_BOX=$(quote "${RESUME_FROM_BOX}")
export RESUME_CKPT=$(quote "${node_resume_ref}")
export RESUME_SOURCE_REF=$(quote "${node_resume_ref}")
export RESUME_SOURCE_EXPECTED_SHA256=$(quote "${CONTROL_RESUME_TRAINING_SHA256}")
export TEACHER_CHECKPOINT=$(quote "${node_teacher_ref}")
export TEACHER_CHECKPOINT_EXPECTED_SHA256=$(quote "${CONTROL_TEACHER_CHECKPOINT_SHA256}")
export DEFAULT_AS_TEACHER_CHECKPOINT=$(quote "${node_teacher_ref}")
export BOX_POLICY_INIT_REF=$(quote "${node_box_policy_init_ref}")
export BOX_POLICY_INIT_EXPECTED_SHA256=$(quote "${BOX_POLICY_INIT_EXPECTED_SHA256}")
export BOX_POLICY_INIT_CACHE_ROOT=$(quote "${BOX_POLICY_INIT_CACHE_ROOT}")
export BOX_POLICY_INIT_ACTOR_HIDDEN_DIMS=$(quote "${BOX_POLICY_INIT_ACTOR_HIDDEN_DIMS}")
export HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET=$(quote "${HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET}")
export OMOMO_EXPECTED_TOTAL=$(quote "${OMOMO_EXPECTED_TOTAL}")
export RESUME_FROM_BOX_EXPECTED_TOTAL=$(quote "${RESUME_FROM_BOX_EXPECTED_TOTAL}")
export STUDENT_ACTOR_HIDDEN_DIMS=$(quote "${STUDENT_ACTOR_HIDDEN_DIMS}")
export SHOO7SR1_NEAR03_DEBUG=0
export STUDENT_POLICY_TYPE=$(quote "${STUDENT_POLICY_TYPE}")
export STUDENT_POLICY_TYPE_EXPLICIT=$(quote "${STUDENT_POLICY_TYPE_EXPLICIT}")
export STUDENT_ACTOR_INPUTS_EXPLICIT=$(quote "${STUDENT_ACTOR_INPUTS_EXPLICIT}")
export STUDENT_FLOW_STEPS=$(quote "${STUDENT_FLOW_STEPS}")
export STUDENT_FLOW_STEPS_EXPLICIT=$(quote "${STUDENT_FLOW_STEPS_EXPLICIT}")
export STUDENT_FLOW_TRAIN_NOISE_STD=$(quote "${STUDENT_FLOW_TRAIN_NOISE_STD}")
export STUDENT_FLOW_TRAIN_NOISE_STD_EXPLICIT=$(quote "${STUDENT_FLOW_TRAIN_NOISE_STD_EXPLICIT}")
export STUDENT_FLOW_TIME_EPSILON=$(quote "${STUDENT_FLOW_TIME_EPSILON}")
export STUDENT_FLOW_TIME_EPSILON_EXPLICIT=$(quote "${STUDENT_FLOW_TIME_EPSILON_EXPLICIT}")
export STUDENT_FLOW_INFERENCE_NOISE_STD=$(quote "${STUDENT_FLOW_INFERENCE_NOISE_STD}")
export STUDENT_FLOW_INFERENCE_NOISE_STD_EXPLICIT=$(quote "${STUDENT_FLOW_INFERENCE_NOISE_STD_EXPLICIT}")
export RUN_NAME=$(quote "${RUN_NAME}")
export TRAINING_NAME=$(quote "${TRAINING_NAME}")
export TRAINING_PROJECT=carry-any
export LOGGER=logger:wandb
export WANDB_PROJECT=carry-any
export WANDB_ENTITY=$(quote "${WANDB_ENTITY}")
export WANDB_BASE_URL=$(quote "${WANDB_BASE_URL}")
export WANDB_INIT_TIMEOUT=$(quote "${WANDB_INIT_TIMEOUT}")
# W&B 0.26 resolves its default console=auto to a rank-zero-only stdout/stderr
# wrapper.  Startup evidence is a cross-rank line protocol consumed from tee's
# exact log, so keep every rank on the same unwrapped output path.  Metrics,
# summaries, files, and artifacts remain enabled; only W&B console capture is
# disabled.
export WANDB_CONSOLE=off
export HOLOSOMA_REQUIRE_WANDB_RUN=$(quote "${HOLOSOMA_REQUIRE_WANDB_RUN}")
export RESUME_WANDB_RUN_ID=$(quote "${RESUME_WANDB_RUN_ID}")
export FRESH_WANDB_RUN_ID=$(quote "${FRESH_WANDB_RUN_ID}")
export WANDB_RESUME_MODE=$(quote "${WANDB_RESUME_MODE}")
export SCHEDULE_NAME=$(quote "${SCHEDULE_NAME}")
export SCHEDULE_NOTES=$(quote "${SCHEDULE_NOTES}")
export NUM_LEARNING_ITERATIONS=$(quote "${NUM_LEARNING_ITERATIONS}")
export TARGET_LEARNING_ITERATION=$(quote "${TARGET_LEARNING_ITERATION}")
export NUM_MINI_BATCHES=$(quote "${NUM_MINI_BATCHES}")
export NUM_LEARNING_EPOCHS=$(quote "${NUM_LEARNING_EPOCHS}")
export HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY=$(quote "${HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY}")
export SAVE_INTERVAL=$(quote "${SAVE_INTERVAL}")
export FIXED_BC_EVAL_LOG_INTERVAL=$(quote "${FIXED_BC_EVAL_LOG_INTERVAL}")
export FIXED_BC_GUARD_ENABLED=$(quote "${FIXED_BC_GUARD_ENABLED}")
export FIXED_BC_GUARD_REFERENCE_END_EPOCH=$(quote "${FIXED_BC_GUARD_REFERENCE_END_EPOCH}")
export FIXED_BC_GUARD_MAX_REFERENCE_RATIO=$(quote "${FIXED_BC_GUARD_MAX_REFERENCE_RATIO}")
export FIXED_BC_GUARD_ABSOLUTE_MAX_MU_MSE=$(quote "${FIXED_BC_GUARD_ABSOLUTE_MAX_MU_MSE}")
export FIXED_BC_GUARD_START_EPOCH=$(quote "${FIXED_BC_GUARD_START_EPOCH}")
export FIXED_BC_GUARD_CONSECUTIVE_EVALS=$(quote "${FIXED_BC_GUARD_CONSECUTIVE_EVALS}")
export HOLOSOMA_MOTION_METRICS_INTERVAL=$(quote "${HOLOSOMA_MOTION_METRICS_INTERVAL}")
export HOLOSOMA_COLLECTION_PROFILE_CANARY=$(quote "${HOLOSOMA_COLLECTION_PROFILE_CANARY}")
export HOLOSOMA_COLLECTION_PROFILE_SYNC_CUDA=$(quote "${HOLOSOMA_COLLECTION_PROFILE_SYNC_CUDA}")
export HOLOSOMA_COLLECTION_PROFILE_INTERVAL=$(quote "${HOLOSOMA_COLLECTION_PROFILE_INTERVAL}")
export HOLOSOMA_STEP_TIMING=$(quote "${HOLOSOMA_COLLECTION_PROFILE_CANARY}")
export HOLOSOMA_STEP_TIMING_PROFILE=$(quote "${HOLOSOMA_COLLECTION_PROFILE_CANARY}")
export HOLOSOMA_STEP_TIMING_SYNC_CUDA=$(quote "${HOLOSOMA_COLLECTION_PROFILE_SYNC_CUDA}")
export HOLOSOMA_STEP_TIMING_INTERVAL=$(quote "${HOLOSOMA_COLLECTION_PROFILE_INTERVAL}")
export ACTOR_LR=$(quote "${ACTOR_LR}")
export CRITIC_LR=$(quote "${CRITIC_LR}")
export PPO_LR_SCHEDULE=$(quote "${PPO_LR_SCHEDULE}")
export PPO_DESIRED_KL=$(quote "${PPO_DESIRED_KL}")
export ACTOR_MIN_LR=$(quote "${ACTOR_MIN_LR}")
export ACTOR_MAX_LR=$(quote "${ACTOR_MAX_LR}")
export CRITIC_MIN_LR=$(quote "${CRITIC_MIN_LR}")
export CRITIC_MAX_LR=$(quote "${CRITIC_MAX_LR}")
export ACTOR_MIN_NOISE_STD=$(quote "${ACTOR_MIN_NOISE_STD}")
export INIT_NOISE_STD=$(quote "${INIT_NOISE_STD}")
export ENTROPY_COEF=$(quote "${ENTROPY_COEF}")
export PPO_START_EPOCH=$(quote "${PPO_START_EPOCH}")
export DAGGER_END_EPOCH=$(quote "${DAGGER_END_EPOCH}")
export PPO_START_COEFF=$(quote "${PPO_START_COEFF}")
export PPO_TARGET_COEFF=$(quote "${PPO_TARGET_COEFF}")
export PPO_SCHEDULE_STEP_EPOCHS=$(quote "${PPO_SCHEDULE_STEP_EPOCHS}")
export DAGGER_REPLAY_ENABLED=$(quote "${DAGGER_REPLAY_ENABLED}")
export DAGGER_REPLAY_CAPACITY=$(quote "${DAGGER_REPLAY_CAPACITY}")
export DAGGER_REPLAY_BATCH_SIZE=$(quote "${DAGGER_REPLAY_BATCH_SIZE}")
export DAGGER_REPLAY_FRACTION=$(quote "${DAGGER_REPLAY_FRACTION}")
export DAGGER_REPLAY_SEED=$(quote "${DAGGER_REPLAY_SEED}")
export DAGGER_LOSS_COEF=$(quote "${DAGGER_LOSS_COEF}")
export DAGGER_MATCH_STD=$(quote "${DAGGER_MATCH_STD}")
export PPO_START_NOISE_STD=$(quote "${PPO_START_NOISE_STD}")
export PPO_START_NOISE_STD_UNTIL_COEFF=$(quote "${PPO_START_NOISE_STD_UNTIL_COEFF}")
export START_AT_TIMESTEP_ZERO_PROB=$(quote "${START_AT_TIMESTEP_ZERO_PROB}")
export START_AT_TIMESTEP_ZERO_PROB_END=$(quote "${START_AT_TIMESTEP_ZERO_PROB_END}")
export START_AT_TIMESTEP_ZERO_PROB_START_ITER=$(quote "${START_AT_TIMESTEP_ZERO_PROB_START_ITER}")
export START_AT_TIMESTEP_ZERO_PROB_END_ITER=$(quote "${START_AT_TIMESTEP_ZERO_PROB_END_ITER}")
export FREEZE_AT_TIMESTEP_ZERO_PROB=$(quote "${FREEZE_AT_TIMESTEP_ZERO_PROB}")
export FREEZE_AT_TIMESTEP_ZERO_PROB_END=$(quote "${FREEZE_AT_TIMESTEP_ZERO_PROB_END}")
export FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=$(quote "${FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER}")
export FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=$(quote "${FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER}")
export UNIFORM_T1_WINDOW_TARGET_SAMPLE_FRAC=$(quote "${UNIFORM_T1_WINDOW_TARGET_SAMPLE_FRAC}")
export STUDENT_MOTION_END_MODE=$(quote "${STUDENT_MOTION_END_MODE}")
export ALLOW_FRESH_CURRICULUM_RESUME=$(quote "${ALLOW_FRESH_CURRICULUM_RESUME}")
export HOLOSOMA_ALLOW_FRESH_CURRICULUM_RESUME=$(quote "${ALLOW_FRESH_CURRICULUM_RESUME}")
export ALLOW_NONDETERMINISTIC_RNG_RESUME=$(quote "${ALLOW_NONDETERMINISTIC_RNG_RESUME}")
export HOLOSOMA_ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME=$(quote "${ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME}")
export HOLOSOMA_ALLOW_RUNTIME_DRIFT_ON_RESUME=$(quote "${ALLOW_RUNTIME_DRIFT_ON_RESUME}")
export HOLOSOMA_SKIP_INITIAL_CHECKPOINT=$(quote "${HOLOSOMA_SKIP_INITIAL_CHECKPOINT}")
export HOLOSOMA_SKIP_GRAD_FINITE_CHECK=$(quote "${HOLOSOMA_SKIP_GRAD_FINITE_CHECK}")
export HOLOSOMA_SKIP_LOSS_FINITE_CHECK=$(quote "${HOLOSOMA_SKIP_LOSS_FINITE_CHECK}")
export HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION=$(quote "${HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION}")
export HOLOSOMA_DEBUG_HEARTBEAT=$(quote "${HOLOSOMA_DEBUG_HEARTBEAT}")
export HOLOSOMA_DEBUG_HEARTBEAT_VERBOSE=$(quote "${HOLOSOMA_DEBUG_HEARTBEAT_VERBOSE}")
export HOLOSOMA_DEBUG_ACTOR=$(quote "${HOLOSOMA_DEBUG_ACTOR}")
export HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE=$(quote "${HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE}")
export HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE=$(quote "${HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE}")
export HOLOSOMA_SYNC_AFTER_OPTIMIZER_STEP=$(quote "${HOLOSOMA_SYNC_AFTER_OPTIMIZER_STEP}")
export HOLOSOMA_SYNC_AFTER_MICROBATCH_FORWARD=$(quote "${HOLOSOMA_SYNC_AFTER_MICROBATCH_FORWARD}")
export HOLOSOMA_DEBUG_MICROBATCH_ALL=$(quote "${HOLOSOMA_DEBUG_MICROBATCH_ALL}")
export HOLOSOMA_GLOO_GRAD_REDUCE=$(quote "${HOLOSOMA_GLOO_GRAD_REDUCE}")
export HOLOSOMA_GLOO_BARRIER=$(quote "${HOLOSOMA_GLOO_BARRIER}")
export HOLOSOMA_GLOO_SMALL_COLLECTIVES=$(quote "${HOLOSOMA_GLOO_SMALL_COLLECTIVES}")
export HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=$(quote "${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE}")
export HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=$(quote "${HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES}")
export HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER=$(quote "${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER}")
export HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC=$(quote "${HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC}")
export HOLOSOMA_RANK_VISIBLE_DEVICES=$(quote "${HOLOSOMA_RANK_VISIBLE_DEVICES}")
export HOLOSOMA_RANK_LOCAL_CPU_AFFINITY=$(quote "${HOLOSOMA_RANK_LOCAL_CPU_AFFINITY}")
export HOLOSOMA_CARB_TASKING_THREAD_COUNT=$(quote "${HOLOSOMA_CARB_TASKING_THREAD_COUNT}")
export HOLOSOMA_OBJECT_COLLIDER_TYPE=$(quote "${HOLOSOMA_OBJECT_COLLIDER_TYPE}")
export HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=$(quote "${HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS}")
export HOLOSOMA_CONTIGUOUS_MINIBATCHES=$(quote "${HOLOSOMA_CONTIGUOUS_MINIBATCHES}")
export HOLOSOMA_DAGGER_SUPERVISED_ONLY=$(quote "${HOLOSOMA_DAGGER_SUPERVISED_ONLY}")
export HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP=$(quote "${HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP}")
export HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH=$(quote "${HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH}")
export HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD=$(quote "${HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD}")
export HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC=$(quote "${HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC}")
unset HOLOSOMA_SKIP_WANDB_CHECKPOINT_UPLOAD
unset HOLOSOMA_SKIP_WANDB_FILE_UPLOAD
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=$(quote "${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE}")
export EXPORT_ONNX=$(quote "${EXPORT_ONNX}")
export WANDB_RESUME_SAME_RUN=$(quote "${WANDB_RESUME_SAME_RUN}")
export OBJECT_GEOMETRY_MODE=mesh
export HOLOSOMA_OBJECT_SPAWN_MODE=single_slot_multi_urdf
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=1
export TORCH_DIST_TIMEOUT_SEC=$(quote "${TORCH_DIST_TIMEOUT_SEC}")
export MAX_RESTARTS=$(quote "${MAX_RESTARTS}")
export PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=$(quote "${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY}")
export PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY=$(quote "${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY}")
export PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=$(quote "${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY}")
export PHYSX_GPU_COLLISION_STACK_SIZE=$(quote "${PHYSX_GPU_COLLISION_STACK_SIZE}")
EOF
)
  if [[ -n "${STUDENT_ACTOR_INPUTS_WAS_SET}" ]]; then
    env_exports+=$'\n'"export STUDENT_ACTOR_INPUTS=$(quote "${STUDENT_ACTOR_INPUTS}")"
  fi
  if [[ -n "${TEACHER_ACTOR_OBS_HISTORY_LENGTH_WAS_SET}" ]]; then
    env_exports+=$'\n'"export TEACHER_ACTOR_OBS_HISTORY_LENGTH=$(quote "${TEACHER_ACTOR_OBS_HISTORY_LENGTH}")"
  fi
  if [[ -n "${STUDENT_PROPRIO_HISTORY_LENGTH_WAS_SET}" ]]; then
    env_exports+=$'\n'"export STUDENT_PROPRIO_HISTORY_LENGTH=$(quote "${STUDENT_PROPRIO_HISTORY_LENGTH}")"
  fi
  if [[ -n "${STUDENT_ACTION_HISTORY_LENGTH_WAS_SET}" ]]; then
    env_exports+=$'\n'"export STUDENT_ACTION_HISTORY_LENGTH=$(quote "${STUDENT_ACTION_HISTORY_LENGTH}")"
  fi
  if [[ -n "${CRITIC_PROPRIO_HISTORY_LENGTH_WAS_SET}" ]]; then
    env_exports+=$'\n'"export CRITIC_PROPRIO_HISTORY_LENGTH=$(quote "${CRITIC_PROPRIO_HISTORY_LENGTH}")"
  fi
  if [[ -n "${CONTACT_AWARE_HISTORY_WAS_SET}" ]]; then
    env_exports+=$'\n'"export CONTACT_AWARE_HISTORY=$(quote "${CONTACT_AWARE_HISTORY}")"
  fi
  if [[ -n "${CONTACT_AWARE_HISTORY_LENGTH_WAS_SET}" ]]; then
    env_exports+=$'\n'"export CONTACT_AWARE_HISTORY_LENGTH=$(quote "${CONTACT_AWARE_HISTORY_LENGTH}")"
  fi
  if [[ -n "${CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION_WAS_SET}" ]]; then
    env_exports+=$'\n'"export CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION=$(quote "${CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION}")"
  fi
  local tmux_lock_path
  tmux_lock_path="${REMOTE_RUN_ROOT}/.active/.locks/holosoma-tmux-$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}').lock"
  local active_state_path
  active_state_path="${REMOTE_RUN_ROOT}/.active/$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}')_${node}.state"
  local train_cmd
train_cmd=$(cat <<EOF
set -euo pipefail
cd $(quote "${RUN_REPO}")
for required_log_dir in \
    $(quote "${RUN_REPO}/logs") \
    $(quote "${RUN_REPO}/logs/batch_ne") \
    $(quote "${RUN_REPO}/${LOG_DIR}"); do
  if [[ ! -d "\${required_log_dir}" || -L "\${required_log_dir}" ]]; then
    echo "[ERROR][${node}] Active training log directory is missing, non-directory, or symlinked: \${required_log_dir}" >&2
    exit 2
  fi
done
grep -Fx -- $(quote "${SOURCE_SNAPSHOT_ID}") .holosoma_snapshot/id >/dev/null
(sha256sum --quiet -c .holosoma_snapshot/source_manifest.sha256)
if ! ({
    find . -maxdepth 0 -type d -printf 'd\t%m\t%p\0'
    find . -mindepth 1 -maxdepth 1 -type f -printf 'f\t%m\t%p\0'
    for source_dir in src scripts tests submodules; do
      [[ -d "./\${source_dir}" ]] || continue
      find "./\${source_dir}" -type f -printf 'f\t%m\t%p\0'
      find "./\${source_dir}" -type d -printf 'd\t%m\t%p\0'
    done
    find ./.holosoma_snapshot -type d -printf 'd\t%m\t%p\0'
  } | sort -z | cmp -s - .holosoma_snapshot/source_modes.nul); then
  echo "[ERROR][${node}] Installed snapshot source mode closure changed before training." >&2
  exit 2
fi
if ({
    find . -maxdepth 0 -type d -perm /222 -print
    for source_dir in src scripts tests submodules .holosoma_snapshot; do
      [[ -d "./\${source_dir}" ]] || continue
      find "./\${source_dir}" -type d -perm /222 -print
    done
  } | grep -q .); then
  echo "[ERROR][${node}] Installed snapshot has a writable signed source directory before training." >&2
  exit 2
fi
${env_exports}
case "\${DISTILL_AS_ENTRYPOINT}" in
  distill_as_button_solid.sh|distill_as_dual_button_solid.sh)
    ;;
  *)
    echo "[ERROR][${node}] Sealed node control contains a non-allowlisted DISTILL_AS_ENTRYPOINT: \${DISTILL_AS_ENTRYPOINT}" >&2
    exit 2
    ;;
esac
if [[ "\${DISTILL_AS_ENTRYPOINT_PATH}" != "\${PWD}/\${DISTILL_AS_ENTRYPOINT}" \
      || ! -f "\${DISTILL_AS_ENTRYPOINT_PATH}" \
      || -L "\${DISTILL_AS_ENTRYPOINT_PATH}" ]]; then
  echo "[ERROR][${node}] Selected DISTILL_AS_ENTRYPOINT path is not the exact regular repo-local snapshot file: \${DISTILL_AS_ENTRYPOINT_PATH}" >&2
  exit 2
fi
actual_distill_as_entrypoint_sha256=\$(sha256sum -- "\${DISTILL_AS_ENTRYPOINT_PATH}" | awk '{print \$1}')
if [[ "\${actual_distill_as_entrypoint_sha256}" != "\${DISTILL_AS_ENTRYPOINT_SHA256}" \
      || "\${DISTILL_AS_ENTRYPOINT_SHA256}" != $(quote "${DISTILL_AS_ENTRYPOINT_SHA256}") ]]; then
  echo "[ERROR][${node}] Selected DISTILL_AS_ENTRYPOINT SHA256 mismatch: actual=\${actual_distill_as_entrypoint_sha256} expected=\${DISTILL_AS_ENTRYPOINT_SHA256}" >&2
  exit 2
fi
echo "[INFO][${node}] distill_as_entrypoint_verified=\${DISTILL_AS_ENTRYPOINT} path=\${DISTILL_AS_ENTRYPOINT_PATH} sha256=\${DISTILL_AS_ENTRYPOINT_SHA256} formal_fresh=\${DISTILL_AS_FORMAL_FRESH}"
unset actual_distill_as_entrypoint_sha256
if [[ $(quote "${node_rank}") == "0" ]]; then
  RENDEZVOUS_STATE_ROOT=$(quote "${REMOTE_RUN_ROOT}/.rendezvous")
  RENDEZVOUS_MAIN_STATE=$(quote "$(rendezvous_state_path "${MASTER_PORT}")")
  RENDEZVOUS_PROVENANCE_STATE=$(quote "$(rendezvous_state_path "${HOLOSOMA_PROVENANCE_MASTER_PORT}")")
  LAUNCH_TOKEN=$(quote "${launch_token}")
  $(rendezvous_release_validation_helpers)
  $(active_state_validation_helpers)
  release_owned_rendezvous_after_success() {
    mkdir -p "\${RENDEZVOUS_STATE_ROOT}"
    $(private_lifecycle_file_validation_helpers)
    open_private_lifecycle_lock "\${RENDEZVOUS_STATE_ROOT}/.reservation.lock" 9
    if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 9; then
      echo "[WARN][${node}] Timed out acquiring rendezvous release lock after authoritative successful completion; explicit stop must retry." >&2
      return 1
    fi
    validate_and_release_rendezvous_pair \
      "\${RENDEZVOUS_MAIN_STATE}" "\${RENDEZVOUS_PROVENANCE_STATE}" \
      "\${LAUNCH_TOKEN}" $(quote "${SESSION}") $(quote "${MASTER_PORT}") \
      $(quote "${HOLOSOMA_PROVENANCE_MASTER_PORT}") 1
  }
fi
mkdir -p -- "\${LOGGER_BASE_DIR}"
if [[ ! -d "\${LOGGER_BASE_DIR}" || ! -w "\${LOGGER_BASE_DIR}" ]]; then
  echo "[ERROR][${node}] LOGGER_BASE_DIR is not a writable directory: \${LOGGER_BASE_DIR}" >&2
  exit 2
fi
LOGGER_WRITE_PROBE=\$(mktemp "\${LOGGER_BASE_DIR}/.holosoma-write-probe.XXXXXX")
rm -f -- "\${LOGGER_WRITE_PROBE}"
echo "[INFO][${node}] logger_base_dir_verified=\${LOGGER_BASE_DIR}"
source ./scripts/gpu_launch_defaults.sh
if [[ -n "\${PYTHON_RUNTIME_SITEPACKAGES}" ]]; then
  "\${PYTHON_BIN}" scripts/verify_python_runtime_overlay.py \
    --site-packages "\${PYTHON_RUNTIME_SITEPACKAGES}" \
    --manifest-sha256 "\${PYTHON_RUNTIME_MANIFEST_SHA256}" \
    --require-distribution-closure \
    --require-current-runtime-binding
  runtime_manifest="\${PYTHON_RUNTIME_SITEPACKAGES}/.holosoma-runtime-manifest.sha256"
  test -f "\${runtime_manifest}"
  actual_runtime_manifest_sha256=\$(sha256sum "\${runtime_manifest}" | awk '{print \$1}')
  if [[ "\${actual_runtime_manifest_sha256}" != "\${PYTHON_RUNTIME_MANIFEST_SHA256}" ]]; then
    echo "[ERROR][${node}] Python runtime manifest SHA256 mismatch: actual=\${actual_runtime_manifest_sha256} expected=\${PYTHON_RUNTIME_MANIFEST_SHA256}" >&2
    exit 2
  fi
  (cd "\${PYTHON_RUNTIME_SITEPACKAGES}" && sha256sum --quiet -c .holosoma-runtime-manifest.sha256)
  echo "[INFO][${node}] python_runtime_train_overlay_verified=\${PYTHON_RUNTIME_SITEPACKAGES} manifest_sha256=\${PYTHON_RUNTIME_MANIFEST_SHA256}"
fi
PYTHON3_BIN="\$(command -v python3 2>/dev/null || true)"
if [[ -z "\${PYTHON3_BIN}" ]]; then
  echo "[ERROR][${node}] python3 is unavailable after pinning PYTHON_BIN=\${PYTHON_BIN}." >&2
  exit 2
fi
PYTHON3_REALPATH="\$(readlink -f -- "\${PYTHON3_BIN}")"
if [[ "\${PYTHON3_REALPATH}" != "\${PYTHON_BIN}" ]]; then
  echo "[ERROR][${node}] Python interpreter split-brain: PYTHON_BIN=\${PYTHON_BIN} python3=\${PYTHON3_REALPATH}." >&2
  exit 2
fi
"\${PYTHON_BIN}" - <<'PY'
import sys

import numpy
import torch

print(
    f"[INFO][python-preflight] executable={sys.executable} "
    f"torch={torch.__version__} numpy={numpy.__version__}"
)
PY
NCCL_RUNTIME_LIB="\${NCCL_LIB_DIR}/libnccl.so.2"
if [[ -n "\${NCCL_LIB_SHA256}" ]]; then
  if [[ ! -f "\${NCCL_RUNTIME_LIB}" ]]; then
    echo "[ERROR] Required NCCL runtime library is not a regular file: \${NCCL_RUNTIME_LIB}" >&2
    exit 2
  fi
  actual_nccl_lib_sha256=\$(sha256sum "\${NCCL_RUNTIME_LIB}" | awk '{print \$1}')
  if [[ "\${actual_nccl_lib_sha256}" != "\${NCCL_LIB_SHA256}" ]]; then
    echo "[ERROR] NCCL runtime SHA256 mismatch: \${NCCL_RUNTIME_LIB} actual=\${actual_nccl_lib_sha256} expected=\${NCCL_LIB_SHA256}" >&2
    exit 2
  fi
  echo "[INFO][${node}] nccl_runtime_train_sha256_verified=\${actual_nccl_lib_sha256} path=\${NCCL_RUNTIME_LIB}"
fi
if [[ "\${TORCH_DIST_BACKEND}" == "nccl" || "\${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE}" == "1" ]]; then
  export LD_PRELOAD="\${NCCL_RUNTIME_LIB}"
  echo "[INFO][${node}] nccl_runtime_preload=\${NCCL_RUNTIME_LIB}"
fi
if ! ip link show "\${NCCL_SOCKET_IFNAME}" >/dev/null 2>&1; then
  echo "[ERROR][${node}] NCCL_SOCKET_IFNAME=\${NCCL_SOCKET_IFNAME} does not exist on this node." >&2
  ip -o link show >&2 || true
  exit 2
fi
if [[ -n "\${RESUME_SOURCE_REF}" ]]; then
  RESUME_CKPT="\$("\${PYTHON_BIN}" scripts/resolve_exact_checkpoint.py \
    --ref "\${RESUME_SOURCE_REF}" \
    --cache-root "${RUN_REPO}/.checkpoint_cache/training_resume")"
  if [[ -n "\${RESUME_SOURCE_EXPECTED_SHA256}" ]]; then
    RESUME_SOURCE_ACTUAL_SHA256=\$(sha256sum "\${RESUME_CKPT}" | awk '{print \$1}')
    if [[ "\${RESUME_SOURCE_ACTUAL_SHA256}" != "\${RESUME_SOURCE_EXPECTED_SHA256}" ]]; then
      echo "[ERROR][${node}] Resolved training-resume SHA256 mismatch: \${RESUME_CKPT} actual=\${RESUME_SOURCE_ACTUAL_SHA256} expected=\${RESUME_SOURCE_EXPECTED_SHA256}" >&2
      exit 2
    fi
    echo "[INFO][${node}] training_resume_sha256_verified=\${RESUME_SOURCE_ACTUAL_SHA256}"
  fi
  IFS=$'\t' read -r RESUME_ACTOR_HIDDEN_DIMS RESUME_NEXT_ITER RESUME_STUDENT_POLICY_TYPE RESUME_STUDENT_ACTOR_INPUTS RESUME_STUDENT_FLOW_STEPS RESUME_STUDENT_FLOW_TRAIN_NOISE_STD RESUME_STUDENT_FLOW_TIME_EPSILON RESUME_STUDENT_FLOW_INFERENCE_NOISE_STD < <(
    "\${PYTHON_BIN}" - "\${RESUME_CKPT}" "\${TARGET_LEARNING_ITERATION}" <<'PY'
from __future__ import annotations

import json
import os
import sys

from holosoma.utils.checkpoint_validation import (
    load_verified_torch_checkpoint,
    validate_student_actor_contract,
    validate_checkpoint_iterations,
)

checkpoint_path, target_raw = sys.argv[1:3]
checkpoint, _ = load_verified_torch_checkpoint(
    checkpoint_path,
    expected_sha256=os.environ.get("RESUME_SOURCE_EXPECTED_SHA256") or None,
    map_location="cpu",
)
try:
    actor = checkpoint["experiment_config"]["algo"]["config"]["module_dict"]["actor"]
except (KeyError, TypeError) as exc:
    raise SystemExit("[ERROR] Training-resume checkpoint lacks actor contract metadata.") from exc
try:
    actor_contract = validate_student_actor_contract(actor)
except ValueError as exc:
    raise SystemExit(f"[ERROR] {exc}") from exc
dims = actor_contract["hidden_dims"]
policy_type = actor_contract["policy_type"]
actor_inputs = actor_contract["actor_inputs"]
flow_steps = actor_contract["flow_steps"]
flow_train_noise = actor_contract["flow_train_noise"]
flow_epsilon = actor_contract["flow_epsilon"]
flow_inference_noise = actor_contract["flow_inference_noise"]
if os.environ.get("STUDENT_ACTOR_INPUTS_EXPLICIT") == "1":
    import ast

    try:
        requested_actor_inputs = ast.literal_eval(os.environ["STUDENT_ACTOR_INPUTS"])
    except (KeyError, SyntaxError, ValueError) as exc:
        raise SystemExit("[ERROR] Explicit STUDENT_ACTOR_INPUTS is malformed during resume.") from exc
    if tuple(requested_actor_inputs) != actor_inputs:
        raise SystemExit(
            "[ERROR] Explicit STUDENT_ACTOR_INPUTS does not match training-resume checkpoint "
            f"inputs={list(actor_inputs)!r}."
        )
for explicit_flag, environment_name, saved_value, parser in (
    ("STUDENT_FLOW_STEPS_EXPLICIT", "STUDENT_FLOW_STEPS", flow_steps, int),
    (
        "STUDENT_FLOW_TRAIN_NOISE_STD_EXPLICIT",
        "STUDENT_FLOW_TRAIN_NOISE_STD",
        flow_train_noise,
        float,
    ),
    ("STUDENT_FLOW_TIME_EPSILON_EXPLICIT", "STUDENT_FLOW_TIME_EPSILON", flow_epsilon, float),
    (
        "STUDENT_FLOW_INFERENCE_NOISE_STD_EXPLICIT",
        "STUDENT_FLOW_INFERENCE_NOISE_STD",
        flow_inference_noise,
        float,
    ),
):
    if os.environ.get(explicit_flag) == "1":
        try:
            requested_value = parser(os.environ[environment_name])
        except (KeyError, ValueError) as exc:
            raise SystemExit(f"[ERROR] Explicit {environment_name} is malformed during resume.") from exc
        if requested_value != saved_value:
            raise SystemExit(
                f"[ERROR] Explicit {environment_name}={requested_value!r} does not match "
                f"training-resume checkpoint value={saved_value!r}."
            )
try:
    saved_iter, next_iter = validate_checkpoint_iterations(checkpoint)
except (TypeError, ValueError) as exc:
    raise SystemExit(f"[ERROR] Invalid training-resume iteration metadata: {exc}") from exc
target = int(target_raw)
if target <= next_iter:
    raise SystemExit(
        f"[ERROR] TARGET_LEARNING_ITERATION={target} must be greater than checkpoint next_iter={next_iter} "
        f"(saved iter={saved_iter})."
    )
print(
    json.dumps(list(dims), separators=(",", ":")),
    next_iter,
    policy_type,
    json.dumps(list(actor_inputs), separators=(",", ":")),
    flow_steps,
    flow_train_noise,
    flow_epsilon,
    flow_inference_noise,
    sep="\t",
)
PY
  )
  if [[ -n "\${STUDENT_ACTOR_HIDDEN_DIMS}" && "\${STUDENT_ACTOR_HIDDEN_DIMS//[[:space:]]/}" != "\${RESUME_ACTOR_HIDDEN_DIMS//[[:space:]]/}" ]]; then
    echo "[ERROR] Explicit STUDENT_ACTOR_HIDDEN_DIMS=\${STUDENT_ACTOR_HIDDEN_DIMS} does not match training-resume checkpoint dims=\${RESUME_ACTOR_HIDDEN_DIMS}." >&2
    exit 2
  fi
  export STUDENT_ACTOR_HIDDEN_DIMS="\${RESUME_ACTOR_HIDDEN_DIMS}"
  if [[ "\${STUDENT_POLICY_TYPE_EXPLICIT}" == "1" && "\${STUDENT_POLICY_TYPE}" != "\${RESUME_STUDENT_POLICY_TYPE}" ]]; then
    echo "[ERROR] Explicit STUDENT_POLICY_TYPE=\${STUDENT_POLICY_TYPE} does not match training-resume checkpoint type=\${RESUME_STUDENT_POLICY_TYPE}." >&2
    exit 2
  fi
  if [[ "\${STUDENT_POLICY_TYPE_EXPLICIT}" != "1" ]]; then
    export STUDENT_POLICY_TYPE="\${RESUME_STUDENT_POLICY_TYPE}"
  fi
  if [[ "\${STUDENT_ACTOR_INPUTS_EXPLICIT}" != "1" ]]; then
    export STUDENT_ACTOR_INPUTS="\${RESUME_STUDENT_ACTOR_INPUTS}"
  fi
  if [[ "\${STUDENT_FLOW_STEPS_EXPLICIT}" != "1" ]]; then
    export STUDENT_FLOW_STEPS="\${RESUME_STUDENT_FLOW_STEPS}"
  fi
  if [[ "\${STUDENT_FLOW_TRAIN_NOISE_STD_EXPLICIT}" != "1" ]]; then
    export STUDENT_FLOW_TRAIN_NOISE_STD="\${RESUME_STUDENT_FLOW_TRAIN_NOISE_STD}"
  fi
  if [[ "\${STUDENT_FLOW_TIME_EPSILON_EXPLICIT}" != "1" ]]; then
    export STUDENT_FLOW_TIME_EPSILON="\${RESUME_STUDENT_FLOW_TIME_EPSILON}"
  fi
  if [[ "\${STUDENT_FLOW_INFERENCE_NOISE_STD_EXPLICIT}" != "1" ]]; then
    export STUDENT_FLOW_INFERENCE_NOISE_STD="\${RESUME_STUDENT_FLOW_INFERENCE_NOISE_STD}"
  fi
  export RESUME_CKPT
  echo "[INFO][${node}] training_resume_checkpoint_local=\${RESUME_CKPT} next_iter=\${RESUME_NEXT_ITER} target_iter=\${TARGET_LEARNING_ITERATION} policy_type=\${STUDENT_POLICY_TYPE} actor_inputs=\${STUDENT_ACTOR_INPUTS} resume_mode=curriculum_correct_not_bitwise_trajectory"
fi
TEACHER_CHECKPOINT="\$("\${PYTHON_BIN}" scripts/resolve_exact_checkpoint.py \
  --ref "\${TEACHER_CHECKPOINT}" \
  --cache-root "${RUN_REPO}/.checkpoint_cache/teacher")"
TEACHER_CHECKPOINT_ACTUAL_SHA256=\$(sha256sum "\${TEACHER_CHECKPOINT}" | awk '{print \$1}')
if [[ ! "\${TEACHER_CHECKPOINT_ACTUAL_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "[ERROR][${node}] Could not compute resolved teacher-checkpoint SHA256: \${TEACHER_CHECKPOINT}" >&2
  exit 2
fi
if [[ -n "\${TEACHER_CHECKPOINT_EXPECTED_SHA256}" ]]; then
  if [[ "\${TEACHER_CHECKPOINT_ACTUAL_SHA256}" != "\${TEACHER_CHECKPOINT_EXPECTED_SHA256}" ]]; then
    echo "[ERROR][${node}] Resolved teacher-checkpoint SHA256 mismatch: \${TEACHER_CHECKPOINT} actual=\${TEACHER_CHECKPOINT_ACTUAL_SHA256} expected=\${TEACHER_CHECKPOINT_EXPECTED_SHA256}" >&2
    exit 2
  fi
fi
echo "[INFO][${node}] teacher_checkpoint_sha256_verified=\${TEACHER_CHECKPOINT_ACTUAL_SHA256}"
if [[ "\${REQUIRE_MOTION_GENERATOR_TEACHER_MATCH}" == "1" \
      && "\${TEACHER_CHECKPOINT_ACTUAL_SHA256}" != "\${HOLOSOMA_EXTERNAL_AS_MOTION_GENERATOR_TEACHER_SHA256}" ]]; then
  echo "[ERROR][${node}] Distillation-label teacher does not match the teacher that generated the input motion: label_teacher_sha256=\${TEACHER_CHECKPOINT_ACTUAL_SHA256} motion_generator_teacher_sha256=\${HOLOSOMA_EXTERNAL_AS_MOTION_GENERATOR_TEACHER_SHA256}. Select the generator checkpoint or explicitly set REQUIRE_MOTION_GENERATOR_TEACHER_MATCH=0 for a separately justified cross-teacher experiment." >&2
  exit 2
fi
echo "[INFO][${node}] motion_generator_teacher_binding_verified=\${HOLOSOMA_EXTERNAL_AS_MOTION_GENERATOR_TEACHER_SHA256} require_match=\${REQUIRE_MOTION_GENERATOR_TEACHER_MATCH}"
export TEACHER_CHECKPOINT
export DEFAULT_AS_TEACHER_CHECKPOINT="\${TEACHER_CHECKPOINT}"
echo "[INFO][${node}] teacher_checkpoint_local=\${TEACHER_CHECKPOINT}"
if [[ "\${RESUME_FROM_BOX}" == "1" ]]; then
  BOX_RESUME_CKPT="\$("\${PYTHON_BIN}" scripts/resolve_exact_checkpoint.py \
    --ref "\${BOX_POLICY_INIT_REF}" \
    --cache-root "\${BOX_POLICY_INIT_CACHE_ROOT}")"
  if [[ -n "\${BOX_POLICY_INIT_EXPECTED_SHA256}" ]]; then
    BOX_POLICY_INIT_ACTUAL_SHA256=\$(sha256sum "\${BOX_RESUME_CKPT}" | awk '{print \$1}')
    if [[ "\${BOX_POLICY_INIT_ACTUAL_SHA256}" != "\${BOX_POLICY_INIT_EXPECTED_SHA256}" ]]; then
      echo "[ERROR] Resolved policy-init SHA256 mismatch: \${BOX_RESUME_CKPT} actual=\${BOX_POLICY_INIT_ACTUAL_SHA256} expected=\${BOX_POLICY_INIT_EXPECTED_SHA256}" >&2
      exit 2
    fi
    echo "[INFO][${node}] box_policy_init_sha256_verified=\${BOX_POLICY_INIT_ACTUAL_SHA256}"
  fi
  "\${PYTHON_BIN}" - "\${BOX_RESUME_CKPT}" "\${STUDENT_ACTOR_HIDDEN_DIMS}" <<'PY'
from __future__ import annotations

import ast
import os
import sys

from holosoma.utils.checkpoint_validation import load_verified_torch_checkpoint
from holosoma.utils.policy_init_preflight import (
    required_policy_init_terminal_target_from_env,
    validate_policy_init_terminal_source_payload,
)

checkpoint_path, expected_dims_raw = sys.argv[1:3]
checkpoint, _ = load_verified_torch_checkpoint(
    checkpoint_path,
    expected_sha256=os.environ.get("BOX_POLICY_INIT_EXPECTED_SHA256") or None,
    map_location="cpu",
)
required_terminal_target = required_policy_init_terminal_target_from_env()
if required_terminal_target is not None:
    validate_policy_init_terminal_source_payload(
        checkpoint,
        required_target=required_terminal_target,
    )
try:
    actor_cfg = checkpoint["experiment_config"]["algo"]["config"]["module_dict"]["actor"]
    actor_type = actor_cfg["type"]
    actor_inputs = list(actor_cfg["input_dim"])
    actor_hidden_dims = list(actor_cfg["layer_config"]["hidden_dims"])
except (KeyError, TypeError) as exc:
    raise SystemExit(
        "[ERROR] Box policy-init checkpoint lacks architecture metadata; "
        "strict policy init cannot be preflighted safely."
    ) from exc

expected_type = "MLPPerceptionEncoder"
expected_inputs = [
    "actor_obs_root_contact_aware",
    "actor_obs_drop_button",
    "actor_obs_proprio_with_actions_no_linvel",
]
try:
    expected_dims = list(ast.literal_eval(expected_dims_raw))
except Exception as exc:
    raise SystemExit(f"[ERROR] Invalid STUDENT_ACTOR_HIDDEN_DIMS={expected_dims_raw!r}: {exc}") from exc

mismatches = []
if actor_type != expected_type:
    mismatches.append(f"actor type checkpoint={actor_type!r} expected={expected_type!r}")
if actor_inputs != expected_inputs:
    mismatches.append(f"actor inputs checkpoint={actor_inputs!r} expected={expected_inputs!r}")
if actor_hidden_dims != expected_dims:
    mismatches.append(f"hidden dims checkpoint={actor_hidden_dims!r} expected={expected_dims!r}")
if mismatches:
    raise SystemExit("[ERROR] Incompatible RESUME_FROM_BOX checkpoint:\n  - " + "\n  - ".join(mismatches))
print(
    "[INFO] box_policy_init_architecture_verified "
    f"type={actor_type} inputs={actor_inputs} hidden_dims={actor_hidden_dims}"
)
PY
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
if [[ -n "\${FRESH_WANDB_RUN_ID}" ]]; then
  echo "[INFO][${node}] wandb_fresh_prebound_replay=\${WANDB_ENTITY}/carry-any/\${FRESH_WANDB_RUN_ID} mode=\${WANDB_RESUME_MODE}"
fi
echo "[INFO][${node}] actor_hidden_dims=\${STUDENT_ACTOR_HIDDEN_DIMS}"
echo "[INFO][${node}] num_mini_batches=\${NUM_MINI_BATCHES}"
echo "[INFO][${node}] num_learning_epochs=\${NUM_LEARNING_EPOCHS}"
echo "[INFO][${node}] ppo_minibatch_throughput canary=\${HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY} source=snapshot:PPOConfig.num_steps_per_env snapshot_id=\${HOLOSOMA_SOURCE_SNAPSHOT_ID} num_steps_per_env=${PPO_NUM_STEPS_PER_ENV} rank_local_rollout_samples=${PPO_RANK_LOCAL_ROLLOUT_SAMPLES} global_rollout_samples=${PPO_GLOBAL_ROLLOUT_SAMPLES} rank_local_samples_per_minibatch_update=${PPO_RANK_LOCAL_SAMPLES_PER_MINIBATCH_UPDATE} global_samples_per_minibatch_update=${PPO_GLOBAL_SAMPLES_PER_MINIBATCH_UPDATE} minibatch_update_rounds_per_iteration=${PPO_MINIBATCH_UPDATE_ROUNDS_PER_ITERATION}"
echo "[INFO][${node}] save_interval=\${SAVE_INTERVAL} fixed_bc_eval_log_interval=\${FIXED_BC_EVAL_LOG_INTERVAL} motion_metrics_interval=\${HOLOSOMA_MOTION_METRICS_INTERVAL}"
echo "[INFO][${node}] collection_profile canary=\${HOLOSOMA_COLLECTION_PROFILE_CANARY} sync_cuda=\${HOLOSOMA_COLLECTION_PROFILE_SYNC_CUDA} interval=\${HOLOSOMA_COLLECTION_PROFILE_INTERVAL} diagnostic_only=1"
echo "[INFO][${node}] fixed_bc_guard enabled=\${FIXED_BC_GUARD_ENABLED} reference_end_epoch=\${FIXED_BC_GUARD_REFERENCE_END_EPOCH} max_reference_ratio=\${FIXED_BC_GUARD_MAX_REFERENCE_RATIO} absolute_max_mu_mse=\${FIXED_BC_GUARD_ABSOLUTE_MAX_MU_MSE} start_epoch=\${FIXED_BC_GUARD_START_EPOCH} consecutive_evals=\${FIXED_BC_GUARD_CONSECUTIVE_EVALS}"
echo "[INFO][${node}] ppo_lr_controller schedule=\${PPO_LR_SCHEDULE} desired_kl=\${PPO_DESIRED_KL} actor_lr=\${ACTOR_LR} actor_bounds=[\${ACTOR_MIN_LR},\${ACTOR_MAX_LR}] critic_lr=\${CRITIC_LR} critic_bounds=[\${CRITIC_MIN_LR},\${CRITIC_MAX_LR}]"
echo "[INFO][${node}] ppo_schedule=\${PPO_START_EPOCH}->\${DAGGER_END_EPOCH} start=\${PPO_START_COEFF} target=\${PPO_TARGET_COEFF} step_epochs=\${PPO_SCHEDULE_STEP_EPOCHS} dagger_loss_coef=\${DAGGER_LOSS_COEF}"
echo "[INFO][${node}] dagger_replay enabled=\${DAGGER_REPLAY_ENABLED} capacity_per_rank=\${DAGGER_REPLAY_CAPACITY} batch_per_update=\${DAGGER_REPLAY_BATCH_SIZE} fraction=\${DAGGER_REPLAY_FRACTION} seed=\${DAGGER_REPLAY_SEED} pure_bc_required=1 disjoint_fixed_bc_required=1"
if [[ "\${HOLOSOMA_DAGGER_SUPERVISED_ONLY}" == "1" ]]; then
  effective_supervised_actor_microbatch="\${HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH}"
  effective_supervised_actor_stream_backward="\${HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD}"
else
  effective_supervised_actor_microbatch=0
  effective_supervised_actor_stream_backward=0
fi
echo "[INFO][${node}] dagger_supervised_only=\${HOLOSOMA_DAGGER_SUPERVISED_ONLY} actor_only_step=\${HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP}"
echo "[INFO][${node}] supervised_actor_microbatch requested=\${HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH} effective=\${effective_supervised_actor_microbatch} stream_backward_requested=\${HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD} stream_backward_effective=\${effective_supervised_actor_stream_backward}"
unset effective_supervised_actor_microbatch effective_supervised_actor_stream_backward
echo "[INFO][${node}] skip_critic_weight_sync=\${HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC}"
echo "[INFO][${node}] torch_dist_backend=\${TORCH_DIST_BACKEND} timeout_sec=\${TORCH_DIST_TIMEOUT_SEC} max_restarts=\${MAX_RESTARTS}"
echo "[INFO][${node}] gloo_grad_reduce=\${HOLOSOMA_GLOO_GRAD_REDUCE}"
echo "[INFO][${node}] gloo_barrier=\${HOLOSOMA_GLOO_BARRIER}"
echo "[INFO][${node}] gloo_small_collectives=\${HOLOSOMA_GLOO_SMALL_COLLECTIVES}"
echo "[INFO][${node}] hierarchical_grad_reduce=\${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE} hierarchical_small_collectives=\${HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES} cpu_leader=\${HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER} hierarchical_pg_timeout_sec=\${HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC} hierarchical_small_scope=eligible_integral_verdict_control_only floating_reductions=flat_gloo"
echo "[INFO][${node}] rank_visible_devices=\${HOLOSOMA_RANK_VISIBLE_DEVICES}"
echo "[INFO][${node}] rank_local_cpu_affinity=\${HOLOSOMA_RANK_LOCAL_CPU_AFFINITY} carb_tasking_thread_count=\${HOLOSOMA_CARB_TASKING_THREAD_COUNT:-<runtime-default>}"
echo "[INFO][${node}] object_collider_type=\${HOLOSOMA_OBJECT_COLLIDER_TYPE}"
echo "[INFO][${node}] object_contact_reporters=\${HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS}"
echo "[INFO][${node}] wandb_checkpoint_upload=enabled"
echo "[INFO][${node}] nccl_if=\${NCCL_SOCKET_IFNAME} gloo_if=\${GLOO_SOCKET_IFNAME} nccl_ib_disable=\${NCCL_IB_DISABLE} socket_family=\${NCCL_SOCKET_FAMILY}"
echo "[INFO][${node}] nccl_retry_cnt=\${NCCL_SOCKET_RETRY_CNT} retry_sleep_msec=\${NCCL_SOCKET_RETRY_SLEEP_MSEC} socket_nthreads=\${NCCL_SOCKET_NTHREADS} nsocks_perthread=\${NCCL_NSOCKS_PERTHREAD}"
echo "[INFO][${node}] torch_nccl_async=\${TORCH_NCCL_ASYNC_ERROR_HANDLING} monitoring=\${TORCH_NCCL_ENABLE_MONITORING} heartbeat_sec=\${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC} dump_on_timeout=\${TORCH_NCCL_DUMP_ON_TIMEOUT} trace_buffer=\${TORCH_NCCL_TRACE_BUFFER_SIZE} propagate_error=\${TORCH_NCCL_PROPAGATE_ERROR} desync_debug=\${TORCH_NCCL_DESYNC_DEBUG} enable_timing=\${TORCH_NCCL_ENABLE_TIMING}"
echo "[INFO][${node}] nccl_lib_dir=\${NCCL_LIB_DIR} nccl_lib_sha256=\${NCCL_LIB_SHA256:-<not-required>}"
"\${PYTHON_BIN}" - <<'PY'
import ctypes
import os
import socket
from pathlib import Path

import torch


def mapped_nccl_paths() -> list[str]:
    paths: set[str] = set()
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6:
            continue
        raw_path = parts[5]
        if raw_path.startswith("/") and "libnccl.so" in Path(raw_path).name:
            paths.add(str(Path(raw_path).resolve()))
    return sorted(paths)


def runtime_nccl_version() -> int:
    function = ctypes.CDLL(None).ncclGetVersion
    function.argtypes = [ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int
    version = ctypes.c_int()
    result = int(function(ctypes.byref(version)))
    if result != 0:
        raise RuntimeError(f"ncclGetVersion failed with result={result}")
    return int(version.value)


backend = os.environ.get("TORCH_DIST_BACKEND", "").strip().lower()
hierarchical = os.environ.get("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", "0") == "1"
requires_nccl = backend == "nccl" or hierarchical
mapped_paths = mapped_nccl_paths()
if requires_nccl:
    expected_path = str((Path(os.environ["NCCL_LIB_DIR"]) / "libnccl.so.2").resolve())
    nccl_version = runtime_nccl_version()
    if mapped_paths != [expected_path]:
        raise RuntimeError(
            "NCCL runtime mapping mismatch after torch import: "
            f"expected_only={expected_path!r} mapped={mapped_paths!r}"
        )
else:
    expected_path = None
    try:
        nccl_version = runtime_nccl_version()
    except Exception as exc:
        nccl_version = f"unavailable:{type(exc).__name__}:{exc}"

print(
    "[INFO][nccl-preflight] "
    f"host={socket.gethostname()} "
    f"backend={backend} "
    f"hierarchical_grad_reduce={hierarchical} "
    f"torch={torch.__version__} "
    f"torch_cuda={torch.version.cuda} "
    f"runtime_nccl={nccl_version} "
    f"expected_nccl_path={expected_path} "
    f"mapped_nccl_paths={mapped_paths} "
    f"cuda_available={torch.cuda.is_available()} "
    f"cuda_device_count={torch.cuda.device_count()} "
    f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')} "
    f"LD_PRELOAD={os.environ.get('LD_PRELOAD', '')}"
)
PY
# This marker is emitted by the content-addressed tmux payload only after its
# source, interpreter, checkpoint, interface, and NCCL runtime preflights have
# completed.  It is deliberately launch-bound and written directly to the
# active log so the controller never treats tmux creation itself as readiness.
printf 'HOLOSOMA_STARTUP_READY token=%s launch_epoch=%s source_snapshot=%s phase=batch_preflight_complete\n' \
  $(quote "${launch_token}") $(quote "${launch_epoch}") $(quote "${SOURCE_SNAPSHOT_ID}") \
  >> $(quote "${log_file}")
# distill_torso_box.sh prints final_train_command only after its complete CLI
# validation and immediately before invoking torch.distributed.run.  The
# bounded controller handshake requires that boundary, every worker's
# cross-rank provenance signal, and the later launch-bound marker emitted only
# after real env/algo/checkpoint setup and a main-process-group barrier.
export PRINT_TRAIN_CMD=1
TRAIN_EXTRA_ARGS=(--logger.entity="\${WANDB_ENTITY}")
if [[ -n "\${RESUME_WANDB_RUN_ID}" ]]; then
  TRAIN_EXTRA_ARGS+=(--logger.id="\${RESUME_WANDB_RUN_ID}" --logger.resume="\${WANDB_RESUME_MODE}")
elif [[ -n "\${FRESH_WANDB_RUN_ID}" ]]; then
  TRAIN_EXTRA_ARGS+=(--logger.id="\${FRESH_WANDB_RUN_ID}" --logger.resume="\${WANDB_RESUME_MODE}")
fi
if [[ "\${DISTILL_AS_FORMAL_FRESH}" == 1 ]]; then
  for _formal_dual_arg in "\${TRAIN_EXTRA_ARGS[@]}"; do
    _formal_dual_arg_normalized="\${_formal_dual_arg//_/-}"
    case "\${_formal_dual_arg_normalized}" in
      --observation.groups.actor-obs-root-contact-aware.history-length|\
      --observation.groups.actor-obs-root-contact-aware.history-length=*|\
      --observation.groups.actor-obs-pickup-button.history-length|\
      --observation.groups.actor-obs-pickup-button.history-length=*|\
      --observation.groups.actor-obs-drop-button.history-length|\
      --observation.groups.actor-obs-drop-button.history-length=*|\
      --observation.groups.actor-obs-proprio-with-actions-no-linvel.history-length|\
      --observation.groups.actor-obs-proprio-with-actions-no-linvel.history-length=*)
        echo "[ERROR][${node}] Formal dual TRAIN_EXTRA_ARGS must not override any of the four manifest-bound actor history lengths: \${_formal_dual_arg}" >&2
        exit 2
        ;;
    esac
  done
  unset _formal_dual_arg _formal_dual_arg_normalized
fi
# Close the barrier-to-entrypoint mutation window as tightly as batch_ne can:
# the sealed tmux control reopens and rehashes the exact view immediately
# before invoking the selected wrapper.  The wrapper then independently
# recomputes its content-addressed views and training provenance before
# torchrun/simulator startup.
"\${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str((Path.cwd() / "scripts").resolve()))
from prepare_as_rank_shards import (
    compute_rank_shard_source_digest,
    validate_published_rank_shards,
)
from motion_generator_teacher import (
    MOTION_GENERATOR_TEACHER_KEY,
    motion_generator_teacher_from_solid_manifest,
    validate_motion_generator_teacher,
)


def required_env(name: str, *, digest: bool = False) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"[ERROR] Sealed train control is missing {name}")
    if digest and re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SystemExit(f"[ERROR] Sealed train control has malformed {name}")
    return value


def readable_file(path: Path, *, role: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise SystemExit(f"[ERROR] {role} is missing before entrypoint: {path}: {exc}") from exc
    if not resolved.is_file():
        raise SystemExit(f"[ERROR] {role} is not a regular file before entrypoint: {resolved}")
    try:
        with resolved.open("rb") as stream:
            stream.read(1)
    except OSError as exc:
        raise SystemExit(f"[ERROR] {role} is unreadable before entrypoint: {resolved}: {exc}") from exc
    return resolved


def file_sha(path: Path, *, role: str) -> tuple[str, Path]:
    resolved = readable_file(path, role=role)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), resolved


def local_path(raw: object, *, base: Path, role: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise SystemExit(f"[ERROR] {role} is empty before entrypoint")
    value = raw.strip()
    if value.lower().startswith(("http://", "https://", "package://", "data:")):
        raise SystemExit(f"[ERROR] {role} is non-local before entrypoint: {value!r}")
    if value.lower().startswith("file://"):
        value = value[7:]
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else base / candidate


def canonical_json_sha(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def immutable_relative_record(base: Path, record: object, *, role: str) -> str:
    if not isinstance(record, dict):
        raise SystemExit(f"[ERROR] {role} manifest record is not a mapping")
    relative_raw = record.get("path")
    expected_sha = record.get("sha256")
    expected_size = record.get("size")
    if not isinstance(relative_raw, str) or not relative_raw:
        raise SystemExit(f"[ERROR] {role} manifest record has no path")
    relative = Path(relative_raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(f"[ERROR] {role} manifest path escapes its immutable root: {relative_raw!r}")
    if not isinstance(expected_sha, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        raise SystemExit(f"[ERROR] {role} manifest record has a malformed digest")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
        raise SystemExit(f"[ERROR] {role} manifest record has an invalid size")
    candidate = base / relative
    current = base
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise SystemExit(f"[ERROR] {role} contains a symlink after the all-node barrier: {current}")
    actual_sha, resolved = file_sha(candidate, role=role)
    if resolved.stat().st_mode & 0o222:
        raise SystemExit(f"[ERROR] {role} is writable after the all-node barrier: {resolved}")
    if actual_sha != expected_sha or resolved.stat().st_size != expected_size:
        raise SystemExit(f"[ERROR] {role} differs from its immutable manifest after the all-node barrier")
    return relative.as_posix()


clip_id = required_env("HOLOSOMA_EXTERNAL_AS_MOTION_CLIP_ID")
solid_source_expected = required_env("HOLOSOMA_EXTERNAL_AS_SOLID_SOURCE_DIGEST", digest=True)
selected_clip_count_expected = int(required_env("HOLOSOMA_EXTERNAL_AS_SELECTED_CLIP_COUNT"))
single_source_expected = required_env("HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST", digest=True)
single_view_expected = required_env("HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST", digest=True)
rank_source_expected = required_env("HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST", digest=True)
motion_expected = required_env("HOLOSOMA_EXTERNAL_AS_MOTION_NPZ_SHA256", digest=True)
map_expected = required_env("HOLOSOMA_EXTERNAL_AS_OBJECT_MAP_SHA256", digest=True)
urdf_expected = required_env("HOLOSOMA_EXTERNAL_AS_OBJECT_URDF_SHA256", digest=True)
mesh_expected = required_env("HOLOSOMA_EXTERNAL_AS_OBJECT_MESH_SHA256", digest=True)
generator_teacher_expected = required_env(
    "HOLOSOMA_EXTERNAL_AS_MOTION_GENERATOR_TEACHER_SHA256",
    digest=True,
)
single_dir = Path(required_env("HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_DIR")).resolve(strict=True)
world_size = int(required_env("HOLOSOMA_EXTERNAL_AS_WORLD_SIZE"))
if world_size < 1:
    raise SystemExit("[ERROR] External AS world size is not positive before entrypoint")
if selected_clip_count_expected < 1:
    raise SystemExit("[ERROR] External AS selected clip count is not positive before entrypoint")
if single_dir.name != single_view_expected or single_dir.parent.name != "by-source":
    raise SystemExit("[ERROR] External AS single-slot path is not bound to its expected view digest")
if single_dir.parent.parent.name != "_single_slot_motion_bank":
    raise SystemExit("[ERROR] External AS single-slot path is outside its immutable solid generation")

# Re-open the immutable solid generation as well as the derived single/rank
# views.  The solid source digest commits to the source motion, source map,
# transitive object assets, contact tree, metadata, selection, and allowlist;
# the checks below prove that the published motion/map/contact snapshot still
# matches that committed manifest immediately before the real wrapper runs.
solid_dir = single_dir.parents[2]
solid_manifest_path = solid_dir / "manifest.json"
solid_marker_path = solid_dir / ".generated_by_prepare_immutable_solid_bank"
for role, raw_path in (
    ("immutable solid manifest", solid_manifest_path),
    ("immutable solid marker", solid_marker_path),
):
    if raw_path.is_symlink():
        raise SystemExit(f"[ERROR] {role} became a symlink after the all-node barrier: {raw_path}")
    resolved = readable_file(raw_path, role=role)
    if resolved.stat().st_mode & 0o222:
        raise SystemExit(f"[ERROR] {role} is writable after the all-node barrier: {resolved}")
try:
    solid_manifest = json.loads(solid_manifest_path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"[ERROR] Immutable solid manifest changed or is invalid before entrypoint: {exc}") from exc
if not isinstance(solid_manifest, dict) or solid_manifest.get("version") != 5:
    raise SystemExit("[ERROR] Immutable solid manifest has an unsupported schema before entrypoint")
source_identity = solid_manifest.get("source_identity")
if not isinstance(source_identity, dict):
    raise SystemExit("[ERROR] Immutable solid manifest has no source identity before entrypoint")
if canonical_json_sha(source_identity) != solid_source_expected:
    raise SystemExit("[ERROR] Immutable solid source identity changed after the all-node barrier")
if solid_manifest.get("source_digest") != solid_source_expected:
    raise SystemExit("[ERROR] Immutable solid source digest changed after the all-node barrier")
if solid_manifest.get("selected_clip_count") != selected_clip_count_expected:
    raise SystemExit("[ERROR] Immutable solid clip count changed after the all-node barrier")
try:
    solid_generator_teacher = motion_generator_teacher_from_solid_manifest(
        solid_manifest,
        role="immutable solid manifest before entrypoint",
    )
except ValueError as exc:
    raise SystemExit(f"[ERROR] Invalid solid motion-generator teacher lineage: {exc}") from exc
if (
    solid_generator_teacher is not None
    and solid_generator_teacher["checkpoint_sha256"] != generator_teacher_expected
):
    raise SystemExit(
        "[ERROR] Immutable solid motion-generator teacher changed after the all-node barrier"
    )
try:
    solid_output_root = Path(str(solid_manifest.get("output_root", ""))).resolve(strict=True)
except OSError as exc:
    raise SystemExit(f"[ERROR] Immutable solid output root is missing before entrypoint: {exc}") from exc
if solid_output_root != solid_dir:
    raise SystemExit("[ERROR] Immutable solid manifest points at a different output generation")

published_motion = solid_manifest.get("published_motion_files")
source_motion = source_identity.get("motion_files")
if not isinstance(published_motion, list) or len(published_motion) != selected_clip_count_expected:
    raise SystemExit("[ERROR] Immutable solid manifest has the wrong published motion count")
if not isinstance(source_motion, list) or len(source_motion) != selected_clip_count_expected:
    raise SystemExit("[ERROR] Immutable solid source identity has the wrong motion count")
published_motion_names = {
    immutable_relative_record(solid_dir, record, role="immutable solid motion")
    for record in published_motion
}
actual_motion_names = {path.name for path in solid_dir.glob("*.npz")}
if published_motion_names != actual_motion_names:
    raise SystemExit("[ERROR] Immutable solid motion namespace changed after the all-node barrier")

source_motion_identity = {}
for record in source_motion:
    if not isinstance(record, dict):
        raise SystemExit("[ERROR] Immutable solid source motion record is not a mapping")
    source_name = Path(str(record.get("path", ""))).name
    source_motion_identity[source_name] = (record.get("size"), record.get("sha256"))
published_motion_identity = {
    str(record.get("path")): (record.get("size"), record.get("sha256"))
    for record in published_motion
    if isinstance(record, dict)
}
if source_motion_identity != published_motion_identity:
    raise SystemExit("[ERROR] Immutable solid published motions differ from the committed source identity")

published_map_record = solid_manifest.get("published_object_map")
published_map_name = immutable_relative_record(
    solid_dir,
    published_map_record,
    role="immutable solid object map",
)
solid_map_path = solid_dir / published_map_name
try:
    solid_map_payload = json.loads(solid_map_path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"[ERROR] Immutable solid object map is invalid before entrypoint: {exc}") from exc
if canonical_json_sha(solid_map_payload) != source_identity.get("filtered_object_map_sha256"):
    raise SystemExit("[ERROR] Immutable solid object map differs from the committed filtered map")

contact_name = source_identity.get("contact_export_name")
if (
    not isinstance(contact_name, str)
    or not contact_name
    or Path(contact_name).name != contact_name
):
    raise SystemExit("[ERROR] Immutable solid source identity has an invalid contact snapshot name")
contact_root = solid_dir / contact_name
if contact_root.is_symlink() or not contact_root.is_dir():
    raise SystemExit("[ERROR] Immutable solid contact snapshot is missing or symlinked before entrypoint")
published_contact = solid_manifest.get("published_contact_files")
if not isinstance(published_contact, list) or not published_contact:
    raise SystemExit("[ERROR] Immutable solid manifest has no published contact snapshot records")
if published_contact != source_identity.get("contact_files"):
    raise SystemExit("[ERROR] Immutable solid contact snapshot differs from the committed source identity")
published_contact_names = {
    immutable_relative_record(contact_root, record, role="immutable solid contact payload")
    for record in published_contact
}
actual_contact_names = set()
for path in contact_root.rglob("*"):
    if path.is_symlink():
        raise SystemExit(f"[ERROR] Immutable solid contact snapshot contains a symlink: {path}")
    if path.is_file():
        actual_contact_names.add(path.relative_to(contact_root).as_posix())
    elif not path.is_dir():
        raise SystemExit(f"[ERROR] Immutable solid contact snapshot has an unsupported entry: {path}")
if actual_contact_names != published_contact_names:
    raise SystemExit("[ERROR] Immutable solid contact namespace changed after the all-node barrier")

published_metadata = solid_manifest.get("published_metadata_files")
if not isinstance(published_metadata, list):
    raise SystemExit("[ERROR] Immutable solid manifest has invalid metadata records")
published_metadata_names = {
    immutable_relative_record(solid_dir, record, role="immutable solid metadata")
    for record in published_metadata
}
source_metadata = source_identity.get("metadata_files")
if not isinstance(source_metadata, list) or len(source_metadata) != len(published_metadata):
    raise SystemExit("[ERROR] Immutable solid metadata differs from its committed source identity")
source_metadata_identity = {
    Path(str(record.get("path", ""))).name: (record.get("size"), record.get("sha256"))
    for record in source_metadata
    if isinstance(record, dict)
}
published_metadata_identity = {
    str(record.get("path")): (record.get("size"), record.get("sha256"))
    for record in published_metadata
    if isinstance(record, dict)
}
if source_metadata_identity != published_metadata_identity \
        or published_metadata_names != set(published_metadata_identity):
    raise SystemExit("[ERROR] Immutable solid published metadata differs from its committed source identity")

single_map = readable_file(single_dir / "_clip_object_urdf_map.json", role="effective object map")
manifest_path = readable_file(single_dir / "manifest.json", role="single-slot manifest")
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = json.loads(single_map.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"[ERROR] External AS JSON changed or is invalid before entrypoint: {exc}") from exc
if manifest.get("source_digest") != single_source_expected or manifest.get("view_digest") != single_view_expected:
    raise SystemExit("[ERROR] Single-slot manifest identity changed after the all-node barrier")
try:
    single_generator_raw = manifest.get(MOTION_GENERATOR_TEACHER_KEY)
    single_generator_teacher = (
        None
        if single_generator_raw is None
        else validate_motion_generator_teacher(
            single_generator_raw,
            role="single-slot motion-generator teacher before entrypoint",
        )
    )
except ValueError as exc:
    raise SystemExit(f"[ERROR] Invalid single-slot motion-generator teacher lineage: {exc}") from exc
if single_generator_teacher != solid_generator_teacher:
    raise SystemExit(
        "[ERROR] Single-slot motion-generator teacher differs from its immutable solid source"
    )
if (
    single_generator_teacher is not None
    and single_generator_teacher["checkpoint_sha256"] != generator_teacher_expected
):
    raise SystemExit(
        "[ERROR] Single-slot motion-generator teacher changed after the all-node barrier"
    )
clips = payload.get("clips") if isinstance(payload, dict) else None
if not isinstance(clips, dict) or clip_id not in clips or not isinstance(clips[clip_id], dict):
    raise SystemExit("[ERROR] Replay-bound clip disappeared from the effective object map")
entry = clips[clip_id]

motion_actual, _ = file_sha(single_dir / f"{clip_id}.npz", role="replay-bound motion")
map_actual, _ = file_sha(single_map, role="effective object map")
urdf_candidate = local_path(entry.get("object_urdf_path"), base=single_map.parent, role="object_urdf_path")
urdf_actual, urdf_path = file_sha(urdf_candidate, role="effective object URDF")
try:
    urdf_root = ET.parse(urdf_path).getroot()
except Exception as exc:
    raise SystemExit(f"[ERROR] Effective object URDF is invalid before entrypoint: {exc}") from exc
primary_mesh_raw = entry.get("object_mesh_path")
if isinstance(primary_mesh_raw, str) and primary_mesh_raw.strip():
    mesh_base = single_map.parent
else:
    mesh_tags = urdf_root.findall(".//mesh")
    if not mesh_tags:
        raise SystemExit("[ERROR] Effective object URDF has no mesh before entrypoint")
    primary_mesh_raw = mesh_tags[0].get("filename")
    mesh_base = urdf_path.parent
mesh_actual, _ = file_sha(
    local_path(primary_mesh_raw, base=mesh_base, role="primary object mesh"),
    role="primary object mesh",
)
for role, expected, actual in (
    ("motion", motion_expected, motion_actual),
    ("object map", map_expected, map_actual),
    ("object URDF", urdf_expected, urdf_actual),
    ("object mesh", mesh_expected, mesh_actual),
):
    if actual != expected:
        raise SystemExit(
            f"[ERROR] External AS {role} changed after the all-node barrier: "
            f"actual={actual} expected={expected}"
        )

rank_source_actual = compute_rank_shard_source_digest(
    motion_dir=single_dir,
    object_map=single_map,
    world_size=world_size,
)
if rank_source_actual != rank_source_expected:
    raise SystemExit(
        "[ERROR] External AS full asset closure changed after the all-node barrier: "
        f"actual={rank_source_actual} expected={rank_source_expected}"
    )
rank_root = single_dir / "_rank_shards" / "by-source" / rank_source_expected / f"ws{world_size}"
validate_published_rank_shards(
    motion_dir=single_dir,
    object_map=single_map,
    output_root=rank_root,
    world_size=world_size,
    expected_source_digest=rank_source_expected,
)
print(
    "[INFO] external_as_asset_closure_reverified_before_entrypoint "
    f"clip={clip_id} single_view={single_view_expected} rank_source={rank_source_expected}"
)
PY
bash "\${DISTILL_AS_ENTRYPOINT_PATH}" "\${TRAIN_EXTRA_ARGS[@]}" 2>&1 | tee -a $(quote "${log_file}")
if [[ $(quote "${node_rank}") == "0" ]]; then
  # Never release rendezvous ownership from a generic EXIT trap: controller
  # rollback kills rank0 before it knows whether every other node was cleaned,
  # and a rank0 crash must quarantine the endpoints rather than make them
  # reusable under surviving workers.  A zero pipeline status is only the
  # first condition: re-bind active metadata, this immutable control script,
  # the fresh node log, and the single producer-format completion record while
  # holding the lifecycle lock before releasing either endpoint.
  completion_active_state=$(quote "${active_state_path}")
  completion_log=$(quote "${log_file}")
  completion_script_sha=\$(sha256sum "\$0" | awk '{print \$1}')
  completion_env_sha=\${HOLOSOMA_COMMAND_SHA256:-}
  if [[ ! "\${completion_script_sha}" =~ ^[0-9a-f]{64}$ \
        || "\${completion_env_sha}" != "\${completion_script_sha}" ]]; then
    echo "[ERROR][${node}] Successful training control script is not bound to the atomic tmux command SHA256; preserving rendezvous reservations." >&2
    exit 2
  fi
  $(private_lifecycle_file_validation_helpers)
  open_private_lifecycle_lock $(quote "${tmux_lock_path}") 8
  if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 8; then
    echo "[ERROR][${node}] Timed out acquiring lifecycle lock for clean-completion validation; preserving rendezvous reservations." >&2
    exit 2
  fi
  if [[ ! -f "\${completion_log}" || -L "\${completion_log}" ]]; then
    echo "[ERROR][${node}] Clean-completion active metadata/log is missing, non-regular, or symlinked; preserving rendezvous reservations." >&2
    exit 2
  fi
  if ! load_active_state_v2_exact "\${completion_active_state}" \
      || ! active_state_has_session_namespace $(quote "${SESSION}") \
      || [[ "\${active_phase}" != running \
        || "\${active_snapshot}" != $(quote "${SOURCE_SNAPSHOT_ID}") \
        || "\${active_log_dir}" != $(quote "${LOG_DIR}") \
        || "\${active_target}" != $(quote "${TARGET_LEARNING_ITERATION}") \
        || "\${active_token}" != $(quote "${launch_token}") \
        || "\${active_command_sha}" != "\${completion_env_sha}" \
        || "\${active_epoch}" != $(quote "${launch_epoch}") ]]; then
    echo "[ERROR][${node}] Clean-completion active metadata drifted from this exact launch; preserving rendezvous reservations." >&2
    exit 2
  fi
  expected_binding="HOLOSOMA_LAUNCH_BINDING token=$(quote "${launch_token}") command_sha256=\${completion_env_sha} launch_epoch=$(quote "${launch_epoch}")"
  binding_count=\$(grep -Fxc -- "\${expected_binding}" "\${completion_log}" || true)
  binding_total=\$(grep -Ec '^HOLOSOMA_LAUNCH_BINDING ' "\${completion_log}" || true)
  completion_total=\$(grep -Ec '^HOLOSOMA_RUN_COMPLETE ' "\${completion_log}" || true)
  if [[ "\${binding_count}" != 1 || "\${binding_total}" != 1 \
        || "\${completion_total}" != 1 ]]; then
    echo "[ERROR][${node}] Clean-completion log lacks one exact binding or contains a non-unique completion record; preserving rendezvous reservations." >&2
    exit 2
  fi
  if [[ "\${EXPORT_ONNX}" == True ]]; then
    completion_pattern=$(quote "^HOLOSOMA_RUN_COMPLETE target_iteration=${TARGET_LEARNING_ITERATION} checkpoint=[^[:space:]]*/model_$(printf '%05d' "${TARGET_LEARNING_ITERATION}")\\.pt onnx=[^[:space:]]*/model_$(printf '%05d' "${TARGET_LEARNING_ITERATION}")\\.onnx onnx_sha256=[0-9a-f]{64}$")
  else
    completion_pattern=$(quote "^HOLOSOMA_RUN_COMPLETE target_iteration=${TARGET_LEARNING_ITERATION} checkpoint=[^[:space:]]*/model_$(printf '%05d' "${TARGET_LEARNING_ITERATION}")\\.pt$")
  fi
  completion_count=\$(grep -Ec "\${completion_pattern}" "\${completion_log}" || true)
  if [[ "\${completion_count}" != 1 ]]; then
    echo "[ERROR][${node}] Successful training pipeline lacks the unique current-target producer-format completion record; preserving rendezvous reservations." >&2
    exit 2
  fi
  release_owned_rendezvous_after_success
fi
EOF
)
  local train_cmd_sha256
  train_cmd_sha256=$(printf '%s' "${train_cmd}" | sha256sum | awk '{print $1}')
  if [[ ! "${train_cmd_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[ERROR][${node}] Failed to compute launch-script SHA256." >&2
    return 2
  fi
  # The caller records this exact digest for ownership-safe rollback.  It is
  # also independently written into active metadata, tmux and the fresh log.
  LAST_LAUNCHED_COMMAND_SHA="${train_cmd_sha256}"
  printf '%s\n' "${train_cmd_sha256}" > "${command_sha_result_file}"

  # The complete launch payload contains all fail-closed checkpoint, runtime,
  # and provenance preflights. Passing it as an ssh/tmux argv eventually hits
  # the remote command-length limit. Stream it over stdin into an immutable,
  # content-addressed control script instead, then keep the tmux argv short.
  local remote_control_dir="${RUN_REPO}/.run_control"
  local remote_train_script="${remote_control_dir}/train-${train_cmd_sha256}.sh"
  local remote_train_incoming="${remote_control_dir}/.incoming/train-${train_cmd_sha256}.${BASHPID}.sh"
  local install_train_cmd
  install_train_cmd=$(cat <<EOF
set -euo pipefail
umask 077
CONTROL_DIR=$(quote "${remote_control_dir}")
INCOMING_DIR=$(quote "${remote_control_dir}/.incoming")
INCOMING=$(quote "${remote_train_incoming}")
FINAL=$(quote "${remote_train_script}")
EXPECTED=$(quote "${train_cmd_sha256}")
ensure_real_launch_control_dir() {
  local required_dir="\$1"
  if [[ -e "\${required_dir}" || -L "\${required_dir}" ]]; then
    if [[ ! -d "\${required_dir}" || -L "\${required_dir}" ]]; then
      echo "[ERROR] Refusing non-directory or symlinked launch-control directory: \${required_dir}" >&2
      return 2
    fi
    return 0
  fi
  mkdir -- "\${required_dir}" 2>/dev/null || true
  if [[ ! -d "\${required_dir}" || -L "\${required_dir}" ]]; then
    echo "[ERROR] Failed to create a real launch-control directory: \${required_dir}" >&2
    return 2
  fi
}
ensure_real_launch_control_dir "\${CONTROL_DIR}"
ensure_real_launch_control_dir "\${INCOMING_DIR}"
if [[ -e "\${INCOMING}" || -L "\${INCOMING}" ]]; then
  echo "[ERROR] Refusing to overwrite existing launch-script incoming path: \${INCOMING}" >&2
  exit 2
fi
trap 'rm -f -- "\${INCOMING}"' EXIT
if ! (umask 077; set -o noclobber; cat > "\${INCOMING}"); then
  echo "[ERROR] Failed to create fresh launch-script incoming file: \${INCOMING}" >&2
  exit 2
fi
if [[ ! -f "\${INCOMING}" || -L "\${INCOMING}" ]]; then
  echo "[ERROR] Launch-script incoming path is non-regular or symlinked: \${INCOMING}" >&2
  exit 2
fi
actual=\$(sha256sum "\${INCOMING}" | awk '{print \$1}')
if [[ ! -f "\${INCOMING}" || -L "\${INCOMING}" \
      || "\${actual}" != "\${EXPECTED}" ]]; then
  echo "[ERROR] Streamed launch-script SHA256 mismatch: actual=\${actual} expected=\${EXPECTED}" >&2
  exit 2
fi
chmod 0500 "\${INCOMING}"
INSTALL_LOCK="\${CONTROL_DIR}/.install.lock"
if [[ -e "\${INSTALL_LOCK}" || -L "\${INSTALL_LOCK}" ]] \
    && [[ ! -f "\${INSTALL_LOCK}" || -L "\${INSTALL_LOCK}" ]]; then
  echo "[ERROR] Refusing non-regular or symlinked launch-script install lock: \${INSTALL_LOCK}" >&2
  exit 2
fi
exec 9>"\${INSTALL_LOCK}"
if [[ ! -f "\${INSTALL_LOCK}" || -L "\${INSTALL_LOCK}" ]]; then
  echo "[ERROR] Launch-script install lock is non-regular or symlinked: \${INSTALL_LOCK}" >&2
  exit 2
fi
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 9; then
  echo "[ERROR] Timed out acquiring the launch-script install lock." >&2
  exit 1
fi
if [[ -e "\${FINAL}" || -L "\${FINAL}" ]]; then
  if [[ ! -f "\${FINAL}" || -L "\${FINAL}" ]]; then
    echo "[ERROR] Refusing non-regular or symlinked content-addressed launch script: \${FINAL}" >&2
    exit 2
  fi
  existing=\$(sha256sum "\${FINAL}" | awk '{print \$1}')
  if [[ "\${existing}" != "\${EXPECTED}" ]]; then
    echo "[ERROR] Existing launch script has wrong SHA256: \${FINAL} actual=\${existing} expected=\${EXPECTED}" >&2
    exit 2
  fi
else
  mv -T --no-clobber "\${INCOMING}" "\${FINAL}"
fi
if [[ ! -f "\${FINAL}" || -L "\${FINAL}" ]]; then
  echo "[ERROR] Published launch script became non-regular or symlinked: \${FINAL}" >&2
  exit 2
fi
final_sha256=\$(sha256sum "\${FINAL}" | awk '{print \$1}')
if [[ "\${final_sha256}" != "\${EXPECTED}" ]]; then
  echo "[ERROR] Published launch-script SHA256 mismatch: actual=\${final_sha256} expected=\${EXPECTED}" >&2
  exit 2
fi
trap - EXIT
rm -f -- "\${INCOMING}"
echo "[INFO][${node}] installed_verified_launch_script=\${FINAL} sha256=\${final_sha256}"
EOF
)
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN] stream launch script over ssh stdin node=%s path=%s sha256=%s bytes=%s\n' \
      "${node}" "${remote_train_script}" "${train_cmd_sha256}" "${#train_cmd}"
    printf '%s\n' "${install_train_cmd}"
    # Preserve the expanded payload in dry-run output so launcher-contract
    # tests can continue to inspect every forwarded training option.
    printf '%s\n' "${train_cmd}"
  else
    printf '%s' "${train_cmd}" | \
      remote_run_with_stdin_bounded \
        "${node}" "${install_train_cmd}" "${LAUNCH_CONTROL_TIMEOUT_SECONDS}"
  fi

  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
cd $(quote "${RUN_REPO}")
$(private_lifecycle_file_validation_helpers)
open_private_lifecycle_lock $(quote "${tmux_lock_path}") 8
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 8; then
  echo "[ERROR][${node}] Timed out acquiring the ${SESSION} lifecycle launch lock." >&2
  exit 1
fi
CONTROL_SCRIPT=$(quote "${remote_train_script}")
EXPECTED_SCRIPT_SHA256=$(quote "${train_cmd_sha256}")
if [[ ! -f "\${CONTROL_SCRIPT}" || -L "\${CONTROL_SCRIPT}" ]]; then
  echo "[ERROR][${node}] Verified launch script is missing, non-regular, or symlinked: \${CONTROL_SCRIPT}" >&2
  exit 2
fi
actual_script_sha256=\$(sha256sum "\${CONTROL_SCRIPT}" | awk '{print \$1}')
if [[ ! -f "\${CONTROL_SCRIPT}" || -L "\${CONTROL_SCRIPT}" \
      || "\${actual_script_sha256}" != "\${EXPECTED_SCRIPT_SHA256}" ]]; then
  echo "[ERROR][${node}] Launch-script SHA256 mismatch before tmux: actual=\${actual_script_sha256} expected=\${EXPECTED_SCRIPT_SHA256}" >&2
  exit 2
fi
$(tmux_session_query_helpers)
query_tmux_session_presence $(quote "${SESSION}") || exit
if (( tmux_session_present == 1 )); then
  echo "[ERROR][${node}] tmux session already exists: ${SESSION}" >&2
  exit 1
fi
ACTIVE_STATE=$(quote "${active_state_path}")
$(active_state_validation_helpers)
if ! load_active_state_v2_exact "\${ACTIVE_STATE}" \
    || ! active_state_has_session_namespace $(quote "${SESSION}"); then
  echo "[ERROR][${node}] Launch intent is missing or malformed: \${ACTIVE_STATE}" >&2
  exit 1
fi
if [[ "\${active_version}" != "2" || "\${active_phase}" != "launching" \
      || "\${active_snapshot}" != $(quote "${SOURCE_SNAPSHOT_ID}") \
      || "\${active_log_dir}" != $(quote "${LOG_DIR}") \
      || "\${active_target}" != $(quote "${TARGET_LEARNING_ITERATION}") \
      || "\${active_token}" != $(quote "${launch_token}") \
      || "\${active_command_sha}" != "pending" \
      || "\${active_epoch}" != $(quote "${launch_epoch}") ]]; then
  echo "[ERROR][${node}] Launch intent does not match this launch token/epoch." >&2
  exit 1
fi
LOGS_ROOT=$(quote "${RUN_REPO}/logs")
BATCH_LOG_ROOT=$(quote "${RUN_REPO}/logs/batch_ne")
ACTIVE_LOG_DIR=$(quote "${RUN_REPO}/${LOG_DIR}")
LOG_STAGING_PREFIX=$(quote "${RUN_REPO}/logs/batch_ne/.incoming-log-${launch_token}-${node}")
LOG_STAGING_DIR="\${LOG_STAGING_PREFIX}.\${BASHPID}"
LOG_OWNER_BASENAME=.holosoma_launch_owner_v2
EXPECTED_LOG_OWNER=$(quote $'2\t'"${SOURCE_SNAPSHOT_ID}"$'\t'"${SESSION}"$'\t'"${LOG_DIR}"$'\t'"${TARGET_LEARNING_ITERATION}"$'\t'"${NNODES}"$'\t'"${launch_token}"$'\t'"${launch_epoch}")
for required_log_parent in "\${LOGS_ROOT}" "\${BATCH_LOG_ROOT}"; do
  if [[ ! -d "\${required_log_parent}" || -L "\${required_log_parent}" ]]; then
    echo "[ERROR][${node}] Refusing non-directory or symlinked active-log parent: \${required_log_parent}" >&2
    exit 2
  fi
done
log_dir_has_exact_launch_owner() {
  local candidate_dir="\$1"
  local owner_file="\${candidate_dir}/\${LOG_OWNER_BASENAME}"
  local owner_line_count owner_field_count owner_value
  [[ -d "\${candidate_dir}" && ! -L "\${candidate_dir}" \
        && -f "\${owner_file}" && ! -L "\${owner_file}" ]] || return 1
  owner_line_count=\$(awk 'END { print NR }' "\${owner_file}") || return 1
  owner_field_count=\$(awk -F '\t' 'NR == 1 { print NF }' "\${owner_file}") || return 1
  [[ "\${owner_line_count}" == 1 && "\${owner_field_count}" == 8 ]] || return 1
  owner_value=\$(<"\${owner_file}") || return 1
  [[ "\${owner_value}" == "\${EXPECTED_LOG_OWNER}" ]]
}
cleanup_exact_log_staging() {
  local owner_file="\${LOG_STAGING_DIR}/\${LOG_OWNER_BASENAME}"
  [[ "\${LOG_STAGING_CREATED_BY_THIS_PROCESS:-0}" == 1 ]] || return 0
  if log_dir_has_exact_launch_owner "\${LOG_STAGING_DIR}"; then
    rm -f -- "\${owner_file}" || return 1
    rmdir -- "\${LOG_STAGING_DIR}" || return 1
  elif [[ -d "\${LOG_STAGING_DIR}" && ! -L "\${LOG_STAGING_DIR}" \
          && ! -e "\${owner_file}" && ! -L "\${owner_file}" ]]; then
    rmdir -- "\${LOG_STAGING_DIR}" 2>/dev/null || true
  fi
  LOG_STAGING_CREATED_BY_THIS_PROCESS=0
}
refuse_unowned_active_log_dir() {
  echo "[ERROR][${node}] Refusing to reuse pre-existing run-specific log directory without this exact launch owner: \${ACTIVE_LOG_DIR}" >&2
  return 2
}

# RUN_REPO may be a shared filesystem.  Publish the run directory once from a
# token-unique staging directory, carrying an immutable owner record in the
# same atomic rename.  Other nodes in this exact launch may then share the
# directory, while stale/symlinked/mismatched paths still fail closed.
if [[ -e "\${ACTIVE_LOG_DIR}" || -L "\${ACTIVE_LOG_DIR}" ]]; then
  log_dir_has_exact_launch_owner "\${ACTIVE_LOG_DIR}" \
    || { refuse_unowned_active_log_dir; exit 2; }
else
  LOG_STAGING_CREATED_BY_THIS_PROCESS=0
  if ! mkdir -- "\${LOG_STAGING_DIR}" 2>/dev/null \
      || [[ ! -d "\${LOG_STAGING_DIR}" || -L "\${LOG_STAGING_DIR}" ]]; then
    echo "[ERROR][${node}] Failed to create a unique real run-log staging directory: \${LOG_STAGING_DIR}" >&2
    exit 2
  fi
  LOG_STAGING_CREATED_BY_THIS_PROCESS=1
  trap 'cleanup_exact_log_staging || true' EXIT
  LOG_STAGING_OWNER="\${LOG_STAGING_DIR}/\${LOG_OWNER_BASENAME}"
  if ! (umask 077; set -o noclobber; printf '%s\n' "\${EXPECTED_LOG_OWNER}" > "\${LOG_STAGING_OWNER}") 2>/dev/null \
      || ! chmod 0400 "\${LOG_STAGING_OWNER}" \
      || ! log_dir_has_exact_launch_owner "\${LOG_STAGING_DIR}"; then
    cleanup_exact_log_staging || true
    echo "[ERROR][${node}] Failed to create an exact atomic run-log ownership staging directory." >&2
    exit 2
  fi
  # Each publisher owns a distinct staging directory.  If another node wins
  # the rename on shared storage, only an exact owner record makes that result
  # equivalent; the losing publisher removes only its own staging directory.
  mv_rc=0
  mv -T --no-clobber "\${LOG_STAGING_DIR}" "\${ACTIVE_LOG_DIR}" 2>/dev/null || mv_rc=\$?
  if [[ -e "\${LOG_STAGING_DIR}" || -L "\${LOG_STAGING_DIR}" ]]; then
    if log_dir_has_exact_launch_owner "\${ACTIVE_LOG_DIR}"; then
      cleanup_exact_log_staging || {
        echo "[ERROR][${node}] Could not clean an exact duplicate run-log staging directory after mv rc=\${mv_rc}." >&2
        exit 2
      }
    else
      cleanup_exact_log_staging || true
      refuse_unowned_active_log_dir
      exit 2
    fi
  fi
  if ! log_dir_has_exact_launch_owner "\${ACTIVE_LOG_DIR}"; then
    cleanup_exact_log_staging || true
    refuse_unowned_active_log_dir
    exit 2
  fi
  LOG_STAGING_CREATED_BY_THIS_PROCESS=0
  trap - EXIT
fi
ACTIVE_LOG=$(quote "${log_file}")
if [[ -e "\${ACTIVE_LOG}" || -L "\${ACTIVE_LOG}" ]]; then
  echo "[ERROR][${node}] Refusing to overwrite pre-existing active log path: \${ACTIVE_LOG}" >&2
  exit 2
fi
if ! (umask 077; set -o noclobber; : > "\${ACTIVE_LOG}") 2>/dev/null \
    || [[ ! -f "\${ACTIVE_LOG}" || -L "\${ACTIVE_LOG}" ]]; then
  echo "[ERROR][${node}] Failed to create a fresh regular non-symlink active log: \${ACTIVE_LOG}" >&2
  exit 2
fi
printf 'HOLOSOMA_LAUNCH_BINDING token=%s command_sha256=%s launch_epoch=%s\n' \
  $(quote "${launch_token}") $(quote "${train_cmd_sha256}") $(quote "${launch_epoch}") >> "\${ACTIVE_LOG}"
# Keep the per-session mutation lock in this launcher shell while tmux creates
# and binds the new session, but do not let tmux (and therefore the long-lived
# training process) inherit the lock descriptor.  An inherited FD would keep
# the flock held for the entire run and make batch_ne.sh stop deadlock while
# waiting for the job that it is supposed to stop.
$(tmux_ownership_helpers)
cleanup_new_session_if_owned() {
  if tmux_session_has_new_atomic_identity \
      $(quote "${SESSION}") $(quote "${launch_token}") \
      $(quote "${train_cmd_sha256}") $(quote "${launch_epoch}"); then
    tmux kill-session -t $(quote "${SESSION}") 2>/dev/null || true
    return 0
  fi
  echo "[WARN][${node}] Refusing to remove ${SESSION}: atomic session environment is incomplete/mismatched or an existing ownership option conflicts." >&2
  return 1
}
# Bind the complete ownership triple in the same tmux server operation that
# creates the detached session.  If SSH disappears immediately after
# new-session commits, rollback can still distinguish this exact orphan from a
# same-name session owned by another launch.  The @options below remain the
# stable monitoring interface and are cross-checked before publication.
tmux new-session -d -s $(quote "${SESSION}") \
  -e $(quote "HOLOSOMA_LAUNCH_TOKEN=${launch_token}") \
  -e $(quote "HOLOSOMA_COMMAND_SHA256=${train_cmd_sha256}") \
  -e $(quote "HOLOSOMA_LAUNCH_EPOCH=${launch_epoch}") \
  $(quote "exec bash ${remote_train_script}") 8>&-
if ! tmux_session_has_new_atomic_identity \
    $(quote "${SESSION}") $(quote "${launch_token}") \
    $(quote "${train_cmd_sha256}") $(quote "${launch_epoch}"); then
  cleanup_new_session_if_owned || true
  echo "[ERROR][${node}] Newly created tmux session lacks the exact atomic ownership environment." >&2
  exit 1
fi
if ! tmux set-option -t $(quote "${SESSION}") @holosoma_launch_token $(quote "${launch_token}"); then
  cleanup_new_session_if_owned || true
  echo "[ERROR][${node}] Failed to record launch ownership token; removed the unowned new session." >&2
  exit 1
fi
if ! tmux set-option -t $(quote "${SESSION}") @holosoma_command_sha256 $(quote "${train_cmd_sha256}"); then
  cleanup_new_session_if_owned || true
  echo "[ERROR][${node}] Failed to record launch command SHA256; removed the incompletely-owned session." >&2
  exit 1
fi
if ! tmux set-option -t $(quote "${SESSION}") @holosoma_launch_epoch $(quote "${launch_epoch}"); then
  cleanup_new_session_if_owned || true
  echo "[ERROR][${node}] Failed to record launch epoch; removed the incompletely-owned session." >&2
  exit 1
fi
if ! tmux_session_has_complete_new_identity \
    $(quote "${SESSION}") $(quote "${launch_token}") \
    $(quote "${train_cmd_sha256}") $(quote "${launch_epoch}"); then
  cleanup_new_session_if_owned || true
  echo "[ERROR][${node}] tmux atomic environment/options do not agree with the exact launch identity." >&2
  exit 1
fi
if ! tmux display-message -p -t $(quote "${SESSION}") "[INFO][${node}] started #{session_name}"; then
  cleanup_new_session_if_owned || true
  echo "[ERROR][${node}] Failed to verify the newly launched tmux session." >&2
  exit 1
fi
mkdir -p "\$(dirname "\${ACTIVE_STATE}")"
ACTIVE_INCOMING="\${ACTIVE_STATE}.incoming.${launch_token}"
printf '2\trunning\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  $(quote "${SOURCE_SNAPSHOT_ID}") \
  $(quote "${LOG_DIR}") \
  $(quote "${TARGET_LEARNING_ITERATION}") \
  $(quote "${launch_token}") \
  $(quote "${train_cmd_sha256}") \
  $(quote "${launch_epoch}") > "\${ACTIVE_INCOMING}"
mv -T "\${ACTIVE_INCOMING}" "\${ACTIVE_STATE}"
validate_private_state_file_metadata "\${ACTIVE_STATE}" 4096 0
EOF
)
  remote_run_bounded "${node}" "${cmd}" "${LAUNCH_CONTROL_TIMEOUT_SECONDS}"
}

status_node() {
  local node="$1"
  local active_state_path
  active_state_path="${REMOTE_RUN_ROOT}/.active/$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}')_${node}.state"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
cd $(quote "${REMOTE_RUN_ROOT}")
echo "===== ${node} ====="
unhealthy=0
now=\$(date +%s)
expected_target=$(quote "${TARGET_LEARNING_ITERATION}")
latest_log=""
active_version="" active_phase="" active_snapshot="" active_log_dir="" active_target=""
active_token="" active_command_sha="" active_epoch=""
ACTIVE_STATE=$(quote "${active_state_path}")
$(active_state_validation_helpers)
$(launch_process_closure_helpers)
$(tmux_session_query_helpers)
if ! load_active_state_v2_exact "\${ACTIVE_STATE}" \
    || ! active_state_has_session_namespace $(quote "${SESSION}"); then
  echo "[ERROR] active launch metadata is malformed or unsupported: \${ACTIVE_STATE}"
  exit 1
fi
expected_target="\${active_target}"
if [[ ! "\${now}" =~ ^[1-9][0-9]*$ || \${#now} -gt 18 ]]; then
  echo "[ERROR] current epoch is not a canonical positive decimal: \${now}"
  exit 1
fi
launch_age=0
if [[ "\${active_epoch}" != "\${now}" ]]; then
  if positive_decimal_is_strictly_older "\${active_epoch}" "\${now}"; then
    launch_age=\$((now - active_epoch))
  else
    if (( \${#active_epoch} != \${#now} )); then
      echo "[ERROR] active launch epoch exceeds the bounded node-clock skew without safe subtraction: active=\${active_epoch} now=\${now}"
      exit 1
    fi
    future_skew=\$((active_epoch - now))
    if (( future_skew > $(quote "${STATUS_MAX_CLOCK_SKEW_SECONDS}") )); then
      echo "[ERROR] active launch epoch exceeds STATUS_MAX_CLOCK_SKEW_SECONDS=$(quote "${STATUS_MAX_CLOCK_SKEW_SECONDS}"): active=\${active_epoch} now=\${now} skew=\${future_skew}"
      exit 1
    fi
    echo "[WARN] active launch epoch is ahead of this node clock by \${future_skew}s; launch_age_seconds is clamped to zero."
  fi
fi
log_glob=$(quote "${REMOTE_RUN_ROOT}")/"\${active_snapshot}/\${active_log_dir}"/node_*_$(quote "${node}").log
matching_logs=(\${log_glob})
if (( \${#matching_logs[@]} == 1 )); then
  if [[ -f "\${matching_logs[0]}" && ! -L "\${matching_logs[0]}" ]]; then
    latest_log="\${matching_logs[0]}"
  elif [[ -e "\${matching_logs[0]}" || -L "\${matching_logs[0]}" ]]; then
    echo "[ERROR] unique active log candidate is non-regular or symlinked: \${matching_logs[0]}"
    exit 1
  fi
elif (( \${#matching_logs[@]} > 1 )); then
  echo "[ERROR] active log glob is ambiguous: \${matching_logs[*]}"
  exit 1
fi
echo "active_state=\${ACTIVE_STATE} phase=\${active_phase} snapshot=\${active_snapshot} target=\${expected_target} token=\${active_token} command_sha256=\${active_command_sha} launch_epoch=\${active_epoch}"

tmux_running=0
query_tmux_session_presence $(quote "${SESSION}") || exit
if (( tmux_session_present == 1 )); then
  tmux_running=1
  tmux display-message -p -t $(quote "${SESSION}") 'tmux_session=#{session_name}' || unhealthy=1
else
  echo "[INFO] tmux:${SESSION}:not-running"
fi
if [[ "\${active_phase}" == stopped || "\${active_phase}" == rolled_back ]]; then
  if (( tmux_running == 1 )); then
    echo "[ERROR] lifecycle phase is \${active_phase}, but the same-name tmux session is still running."
    exit 1
  fi
  if ! verify_no_launch_identity_processes \
      "\${active_token}" "\${active_command_sha}" "\${active_epoch}"; then
    echo "[ERROR] lifecycle phase is \${active_phase}, but exact launch-identity processes remain."
    exit 1
  fi
  echo "run_state=\${active_phase}"
  exit 0
fi
if [[ "\${active_phase}" == stopping || "\${active_phase}" == rolling_back ]]; then
  echo "run_state=\${active_phase}"
fi
if (( tmux_running == 1 )) && [[ "\${active_phase}" == running ]]; then
  $(tmux_ownership_helpers)
  if ! tmux_session_has_complete_new_identity \
      $(quote "${SESSION}") "\${active_token}" "\${active_command_sha}" "\${active_epoch}"; then
    echo "[ERROR] tmux atomic environment/options ownership does not match v2 active metadata."
    unhealthy=1
  fi
fi

completed=0
startup_grace=0
if [[ "\${active_phase}" == launching \
      || ( "\${active_phase}" == running && "\${tmux_running}" == 1 ) ]] \
    && (( launch_age < $(quote "${STATUS_STARTUP_GRACE_SECONDS}") )); then
  startup_grace=1
fi
if [[ -n "\${latest_log}" ]]; then
  echo "log=\${latest_log}"
  log_mtime=\$(stat -c %Y "\${latest_log}")
  if [[ ! "\${log_mtime}" =~ ^(0|[1-9][0-9]*)$ || \${#log_mtime} -gt 18 ]]; then
    echo "[ERROR] active log mtime is not a safe canonical epoch: \${log_mtime}"
    exit 1
  fi
  log_age=0
  if [[ "\${log_mtime}" == 0 ]]; then
    log_age=\${now}
  elif [[ "\${log_mtime}" != "\${now}" ]]; then
    if positive_decimal_is_strictly_older "\${log_mtime}" "\${now}"; then
      log_age=\$((now - log_mtime))
    else
      if (( \${#log_mtime} != \${#now} )); then
        echo "[ERROR] active log mtime exceeds the bounded node-clock skew without safe subtraction: mtime=\${log_mtime} now=\${now}"
        exit 1
      fi
      log_future_skew=\$((log_mtime - now))
      if (( log_future_skew > $(quote "${STATUS_MAX_CLOCK_SKEW_SECONDS}") )); then
        echo "[ERROR] active log mtime exceeds STATUS_MAX_CLOCK_SKEW_SECONDS=$(quote "${STATUS_MAX_CLOCK_SKEW_SECONDS}"): mtime=\${log_mtime} now=\${now} skew=\${log_future_skew}"
        exit 1
      fi
      echo "[WARN] active log mtime is ahead of this node clock by \${log_future_skew}s; log_age_seconds is clamped to zero."
    fi
  fi
  expected_binding="HOLOSOMA_LAUNCH_BINDING token=\${active_token} command_sha256=\${active_command_sha} launch_epoch=\${active_epoch}"
  binding_count=\$(grep -Fxc -- "\${expected_binding}" "\${latest_log}" || true)
  binding_total=\$(grep -Ec '^HOLOSOMA_LAUNCH_BINDING ' "\${latest_log}" || true)
  log_bound=0
  if [[ "\${active_phase}" != launching \
        && "\${binding_count}" == 1 && "\${binding_total}" == 1 ]]; then
    log_bound=1
  elif (( binding_total != 0 )); then
    echo "[ERROR] active log contains duplicate or launch-conflicting binding records."
    unhealthy=1
  fi
  if (( log_bound == 0 && startup_grace == 0 )); then
    echo "[ERROR] active log is not bound to the v2 launch token/command/epoch."
    unhealthy=1
  elif (( log_bound == 0 )); then
    echo "[INFO] launch is within startup grace; waiting for the active log binding."
  fi
  last_iter=\$(sed -nE \
    -e 's/.*HOLOSOMA_PROGRESS completed_iteration=([0-9]+)([^0-9].*)?$/\1/p' \
    -e 's/.*Heartbeat: iter[[:space:]]+([0-9]+)([^0-9].*)?$/\1/p' \
    -e 's/.*Entering PPO\.learn at iteration[[:space:]]+([0-9]+)\..*$/\1/p' \
    -e 's/.*Learning iteration[[:space:]]+([0-9]+)\/[0-9]+.*$/\1/p' \
    "\${latest_log}" 2>/dev/null | tail -1 || true)
  echo "progress_iter=\${last_iter:-<not-seen>} log_age_seconds=\${log_age}"

  expected_model_iteration=\$(printf '%05d' "\${expected_target}")
  completion_pattern="^HOLOSOMA_RUN_COMPLETE target_iteration=\${expected_target} checkpoint=[^[:space:]]*/model_\${expected_model_iteration}\\.pt( onnx=[^[:space:]]*/model_\${expected_model_iteration}\\.onnx onnx_sha256=[0-9a-f]{64})?\$"
  completion_total=\$(grep -Ec '^HOLOSOMA_RUN_COMPLETE ' "\${latest_log}" || true)
  completion_count=\$(grep -Ec "\${completion_pattern}" "\${latest_log}" || true)
  completion_valid=0
  if [[ "\${completion_total}" == 1 && "\${completion_count}" == 1 ]]; then
    completion_valid=1
  elif (( completion_total != 0 )); then
    echo "[ERROR] active log contains duplicate or non-canonical completion records."
    unhealthy=1
  fi
  if (( log_bound == 1 && completion_valid == 1 )); then
    # Startup grace protects slow initialization, not a teardown which has
    # already emitted its authoritative final marker. A hung finalizer must
    # become stale on the normal STATUS_STALE_SECONDS schedule.
    startup_grace=0
  fi
  if (( log_bound == 1 && tmux_running == 0 )) \
      && [[ "\${active_phase}" == running \
            && "\${completion_valid}" == 1 ]]; then
    completed=1
    echo "run_state=completed target_iteration=\${expected_target}"
  elif (( log_bound == 1 && tmux_running == 1 )) \
      && (( completion_valid == 1 )); then
    echo "run_state=finalizing target_iteration=\${expected_target}"
  elif (( tmux_running == 0 && startup_grace == 0 )) \
      && [[ "\${active_phase}" == running ]]; then
    echo "[ERROR] tmux is not running and no matching completion marker was found (expected_target=\${expected_target})."
    unhealthy=1
  fi

  if (( completed == 0 && startup_grace == 0 && log_age >= $(quote "${STATUS_STALE_SECONDS}") )) \
      && [[ "\${active_phase}" == running ]]; then
    echo "[ERROR] training log has not changed for \${log_age} seconds."
    unhealthy=1
  fi

  status_dir=.status
  status_file="\${status_dir}/$(quote "${SESSION}_${node}.state")"
  if [[ -e "\${status_dir}" || -L "\${status_dir}" ]]; then
    if [[ ! -d "\${status_dir}" || -L "\${status_dir}" ]]; then
      echo "[ERROR] status progress root is non-directory or symlinked: \${status_dir}"
      exit 1
    fi
  else
    mkdir -- "\${status_dir}"
  fi
  if [[ ! -d "\${status_dir}" || -L "\${status_dir}" ]]; then
    echo "[ERROR] status progress root is not a real directory: \${status_dir}"
    exit 1
  fi
  if [[ -e "\${status_file}" || -L "\${status_file}" ]] \
      && [[ ! -f "\${status_file}" || -L "\${status_file}" ]]; then
    echo "[ERROR] status progress state is non-regular or symlinked: \${status_file}"
    exit 1
  fi
  status_incoming=""
  cleanup_status_incoming() {
    [[ -z "\${status_incoming}" ]] || rm -f -- "\${status_incoming}"
  }
  trap cleanup_status_incoming EXIT
  write_status_progress() {
    local token="\$1" log_path="\$2" iteration="\$3" changed_at="\$4"
    status_incoming="\${status_file}.\$\$.incoming"
    if [[ -e "\${status_incoming}" || -L "\${status_incoming}" ]]; then
      echo "[ERROR] status progress incoming path already exists: \${status_incoming}"
      return 2
    fi
    if ! (umask 077; set -o noclobber; \
        printf '%s %s %s %s\n' "\${token}" "\${log_path}" \
          "\${iteration}" "\${changed_at}" > "\${status_incoming}") \
        || [[ ! -f "\${status_incoming}" || -L "\${status_incoming}" ]]; then
      echo "[ERROR] failed to create exact status progress incoming state"
      return 2
    fi
    mv -T "\${status_incoming}" "\${status_file}"
    status_incoming=""
  }
  previous_token="" previous_log="" previous_iter="" previous_change=""
  if [[ -f "\${status_file}" ]]; then
    if [[ "\$(awk 'END { print NR }' "\${status_file}")" == 1 \
          && "\$(awk 'NR == 1 { print NF }' "\${status_file}")" == 4 ]]; then
      read -r previous_token previous_log previous_iter previous_change < "\${status_file}" || true
    else
      echo "[WARN] ignoring malformed prior status progress record shape: \${status_file}"
    fi
  fi
  previous_change_invalid=0
  if [[ -n "\${previous_change}" ]]; then
    if [[ ! "\${previous_change}" =~ ^[1-9][0-9]*$ ]]; then
      previous_change_invalid=1
    elif [[ "\${previous_change}" != "\${now}" ]] \
        && ! positive_decimal_is_strictly_older "\${previous_change}" "\${now}"; then
      previous_change_invalid=1
    fi
  fi
  if [[ -n "\${previous_iter}" && ! "\${previous_iter}" =~ ^[0-9]+$ ]] \
      || (( previous_change_invalid == 1 )); then
    echo "[WARN] ignoring malformed prior status progress metadata: \${status_file}"
    previous_token=""
    previous_log=""
    previous_iter=""
    previous_change=""
  fi
  if [[ "\${previous_token}" != "\${active_token}" || "\${previous_log}" != "\${latest_log}" ]]; then
    previous_token="\${active_token}"
    previous_log="\${latest_log}"
    previous_iter=""
    previous_change="\${now}"
  fi
  if [[ -n "\${last_iter}" ]]; then
    if [[ "\${last_iter}" != "\${previous_iter}" ]]; then
      write_status_progress "\${active_token}" "\${latest_log}" "\${last_iter}" "\${now}"
      previous_change="\${now}"
    elif (( completed == 0 && startup_grace == 0 )) \
        && [[ "\${active_phase}" == running \
              && -n "\${previous_change}" \
              && \$((now - previous_change)) -ge $(quote "${STATUS_STALE_SECONDS}") ]]; then
      echo "[ERROR] training iteration has not advanced for \$((now - previous_change)) seconds (iter=\${last_iter})."
      unhealthy=1
    fi
  elif (( startup_grace == 1 || log_age < $(quote "${STATUS_STALE_SECONDS}") )); then
    echo "[WARN] no training iteration has been observed in the active log yet."
  fi

  # grep must consume all of tail's output under pipefail; grep -q can make
  # tail die with SIGPIPE and hide a real fatal match.
  if tail -1000 "\${latest_log}" | grep -Ei '(^|[^[:alnum:]_])(nan|[-+]?inf)([^[:alnum:]_]|$)|non[-_ ]finite|Traceback \(most recent call last\)|Exception occurred during training|ChildFailedError' >/dev/null; then
    echo "[ERROR] NaN/Inf/non-finite value or Python training exception detected in recent log output."
    unhealthy=1
  fi
  tail -40 "\${latest_log}"
else
  if (( startup_grace == 1 )); then
    echo "run_state=starting launch_age_seconds=\${launch_age}; no active log yet (grace=$(quote "${STATUS_STARTUP_GRACE_SECONDS}")s)."
  elif [[ "\${active_phase}" == stopped || "\${active_phase}" == rolled_back ]]; then
    echo "run_state=\${active_phase}"
  else
    echo "[ERROR] no v2-bound batch_ne log found for node ${node} after startup grace."
    unhealthy=1
  fi
fi
exit "\${unhealthy}"
EOF
)
  remote_run_bounded "${node}" "${cmd}" "${LAUNCH_STARTUP_PROBE_TIMEOUT_SECONDS}"
}

startup_health_node() {
  local node="$1"
  local node_rank="$2"
  local launch_token="$3"
  local launch_epoch="$4"
  local probe_timeout_seconds="$5"
  local active_state_path
  active_state_path="${REMOTE_RUN_ROOT}/.active/$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}')_${node}.state"
  local log_file="${RUN_REPO}/${LOG_DIR}/node_${node_rank}_${node}.log"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
ACTIVE_STATE=$(quote "${active_state_path}")
ACTIVE_LOG=$(quote "${log_file}")
EXPECTED_EXPORT_ONNX=$(quote "${EXPORT_ONNX}")
$(active_state_validation_helpers)
if ! load_active_state_v2_exact "\${ACTIVE_STATE}" \
    || ! active_state_has_session_namespace $(quote "${SESSION}") \
    || [[ "\${active_phase}" != running \
      || "\${active_snapshot}" != $(quote "${SOURCE_SNAPSHOT_ID}") \
      || "\${active_log_dir}" != $(quote "${LOG_DIR}") \
      || "\${active_target}" != $(quote "${TARGET_LEARNING_ITERATION}") \
      || "\${active_token}" != $(quote "${launch_token}") \
      || ! "\${active_command_sha}" =~ ^[0-9a-f]{64}$ \
      || "\${active_epoch}" != $(quote "${launch_epoch}") ]]; then
  echo "[ERROR][startup][${node}] active metadata is not the exact running identity for this launch: \${ACTIVE_STATE}" >&2
  exit 1
fi
version=\${active_version} phase=\${active_phase} snapshot=\${active_snapshot}
log_dir=\${active_log_dir} target=\${active_target} token=\${active_token}
command_sha=\${active_command_sha} epoch=\${active_epoch}
$(tmux_session_query_helpers)
query_tmux_session_presence $(quote "${SESSION}") || exit
if (( tmux_session_present == 0 )); then
  echo "[ERROR][startup][${node}] owned tmux exited before startup became healthy." >&2
  exit 1
fi
$(tmux_ownership_helpers)
if ! tmux_session_has_complete_new_identity \
    $(quote "${SESSION}") "\${token}" "\${command_sha}" "\${epoch}"; then
  echo "[ERROR][startup][${node}] tmux atomic environment/options do not match active metadata." >&2
  exit 1
fi
if [[ ! -f "\${ACTIVE_LOG}" || -L "\${ACTIVE_LOG}" ]]; then
  echo "[ERROR][startup][${node}] exact active log is missing or symlinked: \${ACTIVE_LOG}" >&2
  exit 1
fi
expected_binding="HOLOSOMA_LAUNCH_BINDING token=\${token} command_sha256=\${command_sha} launch_epoch=\${epoch}"
binding_count=\$(grep -Fxc -- "\${expected_binding}" "\${ACTIVE_LOG}" || true)
binding_total=\$(grep -Ec '^HOLOSOMA_LAUNCH_BINDING ' "\${ACTIVE_LOG}" || true)
if [[ "\${binding_count}" != 1 || "\${binding_total}" != 1 ]]; then
  echo "[ERROR][startup][${node}] launch binding must be one exact record: exact=\${binding_count} total=\${binding_total}." >&2
  exit 1
fi
completion_total=\$(grep -Ec '^HOLOSOMA_RUN_COMPLETE ' "\${ACTIVE_LOG}" || true)
if [[ "\${EXPECTED_EXPORT_ONNX}" == True ]]; then
  startup_completion_pattern=$(quote "^HOLOSOMA_RUN_COMPLETE target_iteration=${TARGET_LEARNING_ITERATION} checkpoint=[^[:space:]]*/model_$(printf '%05d' "${TARGET_LEARNING_ITERATION}")\\.pt onnx=[^[:space:]]*/model_$(printf '%05d' "${TARGET_LEARNING_ITERATION}")\\.onnx onnx_sha256=[0-9a-f]{64}$")
else
  startup_completion_pattern=$(quote "^HOLOSOMA_RUN_COMPLETE target_iteration=${TARGET_LEARNING_ITERATION} checkpoint=[^[:space:]]*/model_$(printf '%05d' "${TARGET_LEARNING_ITERATION}")\\.pt$")
fi
completion_count=\$(grep -Ec "\${startup_completion_pattern}" "\${ACTIVE_LOG}" || true)
if (( completion_total != 0 )) \
    && [[ "\${completion_total}" != 1 || "\${completion_count}" != 1 ]]; then
  echo "[ERROR][startup][${node}] completion evidence is duplicate, wrong-target, or non-canonical: exact=\${completion_count} total=\${completion_total}." >&2
  exit 1
fi
# This exact log was truncated immediately before the launch binding was
# written, so scanning it in full cannot pick up a previous launch.  Do not
# limit this to the recent tail: lengthy simulator preflight output could push
# an early worker traceback beyond an arbitrary line window.
if grep -Ei \
    '(^|[^[:alnum:]_])(nan|[-+]?inf)([^[:alnum:]_]|$)|non[-_ ]finite|Traceback \(most recent call last\)|Exception occurred during training|ChildFailedError' \
    "\${ACTIVE_LOG}" >/dev/null; then
  echo "[ERROR][startup][${node}] fatal/non-finite evidence appeared during startup." >&2
  tail -40 "\${ACTIVE_LOG}" >&2
  exit 1
fi
expected_ready="HOLOSOMA_STARTUP_READY token=\${token} launch_epoch=\${epoch} source_snapshot=\${snapshot} phase=batch_preflight_complete"
ready_count=\$(grep -Fxc -- "\${expected_ready}" "\${ACTIVE_LOG}" || true)
ready_total_count=\$(grep -Ec '^HOLOSOMA_STARTUP_READY ' "\${ACTIVE_LOG}" || true)
torchrun_boundary_count=\$(grep -Ec '^\[INFO\] final_train_command:' "\${ACTIVE_LOG}" || true)
distributed_ready_count=\$(grep -Ec '^\[INFO\] cross_rank_training_provenance_verified world_size=$(quote "${TOTAL_GPUS}") ' "\${ACTIVE_LOG}" || true)
distributed_ready_total_count=\$(grep -Ec '^\[INFO\] cross_rank_training_provenance_verified ' "\${ACTIVE_LOG}" || true)
worker_marker_count=\$(grep -Ec '^\[INFO\] final_worker_preflight_verified ' "\${ACTIVE_LOG}" || true)
valid_worker_count=0
duplicate_worker_count=0
for ((local_rank = 0; local_rank < $(quote "${NPROC}"); local_rank++)); do
  global_rank=\$(( $(quote "${node_rank}") * $(quote "${NPROC}") + local_rank ))
  expected_worker="[INFO] final_worker_preflight_verified global_rank=\${global_rank} local_rank=\${local_rank} world_size=$(quote "${TOTAL_GPUS}") source_snapshot=\${snapshot} launch_token=\${token} launch_epoch=\${epoch}"
  worker_rank_count=\$(grep -Fxc -- "\${expected_worker}" "\${ACTIVE_LOG}" || true)
  if [[ "\${worker_rank_count}" == 1 ]]; then
    valid_worker_count=\$((valid_worker_count + 1))
  elif (( worker_rank_count > 1 )); then
    duplicate_worker_count=\$((duplicate_worker_count + 1))
  fi
done
if (( ready_count > 1 \
      || ready_total_count > 1 \
      || torchrun_boundary_count > 1 \
      || distributed_ready_count > $(quote "${NPROC}") \
      || distributed_ready_total_count > $(quote "${NPROC}") \
      || worker_marker_count > $(quote "${NPROC}") \
      || duplicate_worker_count != 0 )); then
  echo "[ERROR][startup][${node}] duplicate or launch-mismatched startup evidence is not valid: ready=\${ready_count}/\${ready_total_count} torchrun_boundary=\${torchrun_boundary_count} distributed_provenance=\${distributed_ready_count}/\${distributed_ready_total_count} worker_markers=\${worker_marker_count} valid_unique_workers=\${valid_worker_count} duplicate_worker_ranks=\${duplicate_worker_count}." >&2
  exit 1
fi
# ACTIVE_LOG is still being appended by tee while startup is probed.  The
# aggregate prefix count above and the per-rank exact counts are intentionally
# separate reads, so a just-started/partially written marker can transiently
# make worker_marker_count differ from valid_worker_count.  That observation is
# pending, not terminal corruption.  Acceptance still requires every exact
# launch-bound rank marker below and the controller's later stability window;
# monotonic duplicate/over-limit evidence remains immediately fatal above.
if (( ready_count != 1 \
      || ready_total_count != ready_count \
      || torchrun_boundary_count != 1 \
      || distributed_ready_count != $(quote "${NPROC}") \
      || distributed_ready_total_count != distributed_ready_count \
      || valid_worker_count != $(quote "${NPROC}") \
      || worker_marker_count != valid_worker_count )); then
  echo "[INFO][startup][${node}] pending batch_preflight=\${ready_count}/1 torchrun_boundary=\${torchrun_boundary_count}/1 distributed_provenance=\${distributed_ready_count}/$(quote "${NPROC}") final_workers=\${valid_worker_count}/$(quote "${NPROC}") observed_batch_preflight=\${ready_total_count}/1 observed_distributed_provenance=\${distributed_ready_total_count}/$(quote "${NPROC}") observed_worker_markers=\${worker_marker_count}/$(quote "${NPROC}")"
  tail -12 "\${ACTIVE_LOG}"
  exit 75
fi
echo "[INFO][startup][${node}] ready token=\${token} command_sha256=\${command_sha} epoch=\${epoch} distributed_provenance=\${distributed_ready_count} final_workers=\${valid_worker_count}"
EOF
)
  remote_run_bounded "${node}" "${cmd}" "${probe_timeout_seconds}"
}

wait_for_launch_startup() {
  local launch_token="$1"
  local launch_epoch="$2"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] bounded startup-health handshake would verify all ${#NODE_LIST[@]} node(s)."
    return 0
  fi
  if ! command -v timeout >/dev/null 2>&1; then
    echo "[ERROR] GNU timeout is required for the bounded startup-health handshake." >&2
    return 1
  fi

  local started_at now deadline remaining probe_timeout sleep_seconds
  started_at=$(controller_monotonic_seconds) || return
  deadline=$((started_at + LAUNCH_STARTUP_TIMEOUT_SECONDS))
  local -a first_ready_at=()
  local round=0
  while true; do
    now=$(controller_monotonic_seconds) || return
    remaining=$((deadline - now))
    if (( remaining <= 0 )); then
      echo "[ERROR] Startup-health handshake timed out after ${LAUNCH_STARTUP_TIMEOUT_SECONDS}s." >&2
      return 1
    fi
    probe_timeout=${LAUNCH_STARTUP_PROBE_TIMEOUT_SECONDS}
    (( probe_timeout <= remaining )) || probe_timeout=${remaining}
    round=$((round + 1))
    echo "[INFO] Startup-health round=${round} remaining=${remaining}s probe_timeout=${probe_timeout}s"

    local round_dir
    if ! round_dir=$(mktemp -d "${TMPDIR:-/tmp}/holosoma-startup-health.XXXXXX"); then
      echo "[ERROR] Failed to create controller-local startup probe directory." >&2
      return 1
    fi
    local -a probe_pids=()
    local node_index node
    for node_index in "${!NODE_LIST[@]}"; do
      node=${NODE_LIST[${node_index}]}
      (
        set +e
        startup_health_node "${node}" "${node_index}" "${launch_token}" "${launch_epoch}" "${probe_timeout}" \
          >"${round_dir}/${node_index}.out" 2>&1
        printf '%s\n' "$?" >"${round_dir}/${node_index}.rc"
      ) &
      probe_pids[${node_index}]=$!
    done
    for node_index in "${!NODE_LIST[@]}"; do
      wait "${probe_pids[${node_index}]}" || true
    done

    now=$(controller_monotonic_seconds) || return
    local all_stable=1
    local fatal=0
    local rc ready_age
    for node_index in "${!NODE_LIST[@]}"; do
      node=${NODE_LIST[${node_index}]}
      if [[ -f "${round_dir}/${node_index}.out" ]]; then
        sed "s/^/[${node}] /" "${round_dir}/${node_index}.out"
      fi
      rc=""
      [[ -f "${round_dir}/${node_index}.rc" ]] && read -r rc <"${round_dir}/${node_index}.rc"
      case "${rc}" in
        0)
          if [[ -z "${first_ready_at[${node_index}]+x}" ]]; then
            first_ready_at[${node_index}]=${now}
          fi
          ready_age=$((now - first_ready_at[${node_index}]))
          if (( ready_age < LAUNCH_STARTUP_STABILITY_SECONDS )); then
            all_stable=0
            echo "[INFO][startup][${node}] stability=${ready_age}/${LAUNCH_STARTUP_STABILITY_SECONDS}s"
          fi
          ;;
        75|124)
          unset 'first_ready_at['"${node_index}"']'
          all_stable=0
          if [[ "${rc}" == 124 ]]; then
            echo "[WARN][startup][${node}] bounded SSH probe timed out; retrying within the global deadline." >&2
          fi
          ;;
        *)
          unset 'first_ready_at['"${node_index}"']'
          all_stable=0
          fatal=1
          echo "[ERROR][startup][${node}] health probe failed with rc=${rc:-missing}." >&2
          ;;
      esac
    done
    rm -rf -- "${round_dir}"
    if (( fatal != 0 )); then
      return 1
    fi
    if (( all_stable != 0 )); then
      echo "[INFO] Startup-health handshake passed for all ${#NODE_LIST[@]} node(s); ready evidence remained healthy for ${LAUNCH_STARTUP_STABILITY_SECONDS}s."
      return 0
    fi
    now=$(controller_monotonic_seconds) || return
    remaining=$((deadline - now))
    if (( remaining <= 0 )); then
      echo "[ERROR] Startup-health handshake timed out after ${LAUNCH_STARTUP_TIMEOUT_SECONDS}s." >&2
      return 1
    fi
    sleep_seconds=${LAUNCH_STARTUP_POLL_SECONDS}
    (( sleep_seconds <= remaining )) || sleep_seconds=${remaining}
    sleep "${sleep_seconds}"
  done
}

read_stop_identity_node_modern_only() {
  local node="$1"
  local tmux_lock_path
  tmux_lock_path="${REMOTE_RUN_ROOT}/.active/.locks/holosoma-tmux-$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}').lock"
  local active_state_path
  active_state_path="${REMOTE_RUN_ROOT}/.active/$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}')_${node}.state"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
$(private_lifecycle_file_validation_helpers)
open_private_lifecycle_lock $(quote "${tmux_lock_path}") 8
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 8; then
  echo "[ERROR][${node}] Timed out acquiring the ${SESSION} lifecycle stop-identity lock." >&2
  exit 1
fi
ACTIVE_STATE=$(quote "${active_state_path}")
$(active_state_validation_helpers)
if ! load_active_state_v2_exact "\${ACTIVE_STATE}" \
    || ! active_state_has_session_namespace $(quote "${SESSION}"); then
  echo "[ERROR][${node}] Refusing to stop ${SESSION}: exact v2 active metadata is unavailable." >&2
  exit 2
fi
if [[ ! "\${active_command_sha}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "[ERROR][${node}] Refusing to stop ${SESSION}: active command identity is incomplete." >&2
  exit 2
fi
case "\${active_phase}" in
  running|stopping|stopped) ;;
  *)
    echo "[ERROR][${node}] Refusing normal stop from lifecycle phase=\${active_phase}." >&2
    exit 2
    ;;
esac
CONTROL_SCRIPT=$(quote "${REMOTE_RUN_ROOT}")/"\${active_snapshot}/.run_control/train-\${active_command_sha}.sh"
if [[ ! -f "\${CONTROL_SCRIPT}" || -L "\${CONTROL_SCRIPT}" \
      || "\$(sha256sum "\${CONTROL_SCRIPT}" | awk '{print \$1}')" != "\${active_command_sha}" ]]; then
  echo "[ERROR][${node}] Refusing stop: immutable node control script is missing, symlinked, or hash-mismatched." >&2
  exit 2
fi
for topology_name in \
    SESSION RUN_STAMP HOLOSOMA_ACTIVE_LOG_DIR \
    NNODES NODE_RANK NPROC MASTER_ADDR MASTER_PORT HOLOSOMA_PROVENANCE_MASTER_PORT; do
  topology_count=\$(grep -Ec "^export \${topology_name}=" "\${CONTROL_SCRIPT}" || true)
  if [[ "\${topology_count}" != 1 ]]; then
    echo "[ERROR][${node}] Refusing stop: control script must contain one export \${topology_name}= record." >&2
    exit 2
  fi
done
embedded_session=\$(sed -nE 's/^export SESSION=([A-Za-z0-9][A-Za-z0-9_.-]{0,127})$/\1/p' "\${CONTROL_SCRIPT}")
embedded_run_stamp=\$(sed -nE 's/^export RUN_STAMP=([A-Za-z0-9][A-Za-z0-9_.-]{0,127})$/\1/p' "\${CONTROL_SCRIPT}")
# Escape the regex end anchor while constructing this unquoted heredoc;
# otherwise Bash substitutes the controller positional-argument count and
# emits a syntactically invalid remote sed expression.
embedded_log_dir=\$(sed -nE 's#^export HOLOSOMA_ACTIVE_LOG_DIR=(logs/batch_ne/[A-Za-z0-9][A-Za-z0-9_.-]{0,254})\$#\1#p' "\${CONTROL_SCRIPT}")
embedded_nnodes=\$(sed -nE 's/^export NNODES=([0-9]+)$/\1/p' "\${CONTROL_SCRIPT}")
embedded_node_rank=\$(sed -nE 's/^export NODE_RANK=([0-9]+)$/\1/p' "\${CONTROL_SCRIPT}")
embedded_nproc=\$(sed -nE 's/^export NPROC=([0-9]+)$/\1/p' "\${CONTROL_SCRIPT}")
embedded_master_addr=\$(sed -nE 's/^export MASTER_ADDR=([^[:space:]]+)$/\1/p' "\${CONTROL_SCRIPT}")
embedded_master_port=\$(sed -nE 's/^export MASTER_PORT=([0-9]+)$/\1/p' "\${CONTROL_SCRIPT}")
embedded_provenance_port=\$(sed -nE 's/^export HOLOSOMA_PROVENANCE_MASTER_PORT=([0-9]+)$/\1/p' "\${CONTROL_SCRIPT}")
if [[ "\${embedded_session}" != $(quote "${SESSION}") \
      || "\${embedded_log_dir}" != "logs/batch_ne/\${embedded_session}_\${embedded_run_stamp}" \
      || "\${embedded_log_dir}" != "\${active_log_dir}" \
      || ! "\${embedded_nnodes}" =~ ^[1-9][0-9]*$ \
      || ! "\${embedded_node_rank}" =~ ^(0|[1-9][0-9]*)$ \
      || ! "\${embedded_nproc}" =~ ^[1-9][0-9]*$ \
      || "\${embedded_nproc}" != $(quote "${NPROC}") \
      || "\${embedded_master_addr}" != $(quote "${MASTER_ADDR}") \
      || "\${embedded_master_port}" != $(quote "${MASTER_PORT}") \
      || "\${embedded_provenance_port}" != $(quote "${HOLOSOMA_PROVENANCE_MASTER_PORT}") ]]; then
  echo "[ERROR][${node}] Refusing stop: embedded session/log/topology/master/ports do not match this controller invocation." >&2
  exit 2
fi
if tmux has-session -t $(quote "${SESSION}") 2>/dev/null; then
  $(tmux_ownership_helpers)
  if ! tmux_session_is_owned_for_cleanup \
      $(quote "${SESSION}") "\${active_token}" "\${active_command_sha}" "\${active_epoch}"; then
    echo "[ERROR][${node}] Refusing to stop ${SESSION}: same-name tmux is not bound to the active token/command/epoch across atomic environment/options." >&2
    exit 2
  fi
else
  tmux_rc=\$?
  [[ "\${tmux_rc}" == 1 ]] || {
    echo "[ERROR][${node}] Could not verify tmux absence during stop-identity preflight (rc=\${tmux_rc})." >&2
    exit 2
  }
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "\${active_token}" "\${active_epoch}" "\${active_snapshot}" \
  "\${active_log_dir}" "\${active_target}" "\${active_command_sha}" \
  "\${embedded_nnodes}" "\${embedded_node_rank}"
EOF
)
  remote_run_bounded "${node}" "${cmd}" "${LAUNCH_CLEANUP_TIMEOUT_SECONDS}"
}

read_stop_identity_node() {
  local node="$1"
  local tmux_lock_path
  tmux_lock_path="${REMOTE_RUN_ROOT}/.active/.locks/holosoma-tmux-$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}').lock"
  local active_state_path
  active_state_path="${REMOTE_RUN_ROOT}/.active/$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}')_${node}.state"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
$(private_lifecycle_file_validation_helpers)
open_private_lifecycle_lock $(quote "${tmux_lock_path}") 8
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 8; then
  echo "[ERROR][${node}] Timed out acquiring the ${SESSION} lifecycle stop-identity lock." >&2
  exit 1
fi
ACTIVE_STATE=$(quote "${active_state_path}")
$(active_state_validation_helpers)
$(legacy_stop_process_helpers)
ALLOW_LEGACY_ACTIVE_MODE=$(quote "${LEGACY_STOP_EXPECTED_TOKEN:+1}")
ALLOW_LEGACY_ACTIVE_MODE=\${ALLOW_LEGACY_ACTIVE_MODE:-0}
if ! load_active_state_v2_exact "\${ACTIVE_STATE}" "\${ALLOW_LEGACY_ACTIVE_MODE}"; then
  echo "[ERROR][${node}] Refusing to stop ${SESSION}: exact v2 active metadata is unavailable." >&2
  exit 2
fi
if [[ "\${active_state_legacy_mode:-0}" == 1 ]]; then
  # The compatibility loader has parsed bounded data, but none of it becomes
  # an ownership/path decision until all five externally supplied identity
  # fields authenticate the complete legacy record.
  expected_legacy_log_dir=$(quote "logs/batch_ne/${SESSION}_${LEGACY_STOP_EXPECTED_RUN_STAMP}")
  if [[ "\${active_snapshot}" != $(quote "${LEGACY_STOP_EXPECTED_SNAPSHOT_ID}") \
        || "\${active_token}" != $(quote "${LEGACY_STOP_EXPECTED_TOKEN}") \
        || "\${active_epoch}" != $(quote "${LEGACY_STOP_EXPECTED_EPOCH}") \
        || "\${active_log_dir}" != "\${expected_legacy_log_dir}" \
        || "\${active_target}" != $(quote "${LEGACY_STOP_EXPECTED_TARGET}") ]]; then
    echo "[ERROR][${node}] Legacy active identity does not match the explicit five-field stop authorization." >&2
    exit 2
  fi
fi
if ! active_state_has_session_namespace $(quote "${SESSION}"); then
  echo "[ERROR][${node}] Refusing to stop ${SESSION}: exact v2 active metadata is unavailable." >&2
  exit 2
fi
if [[ ! "\${active_command_sha}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "[ERROR][${node}] Refusing to stop ${SESSION}: active command identity is incomplete." >&2
  exit 2
fi
case "\${active_phase}" in
  running|stopping|stopped) ;;
  *)
    echo "[ERROR][${node}] Refusing normal stop from lifecycle phase=\${active_phase}." >&2
    exit 2
    ;;
esac
CONTROL_SCRIPT=$(quote "${REMOTE_RUN_ROOT}")/"\${active_snapshot}/.run_control/train-\${active_command_sha}.sh"
if [[ ! -f "\${CONTROL_SCRIPT}" || -L "\${CONTROL_SCRIPT}" \
      || "\$(sha256sum "\${CONTROL_SCRIPT}" | awk '{print \$1}')" != "\${active_command_sha}" ]]; then
  echo "[ERROR][${node}] Refusing stop: immutable node control script is missing, symlinked, or hash-mismatched." >&2
  exit 2
fi
for topology_name in \
    NNODES NODE_RANK NPROC MASTER_ADDR MASTER_PORT HOLOSOMA_PROVENANCE_MASTER_PORT; do
  topology_count=\$(grep -Ec "^export \${topology_name}=" "\${CONTROL_SCRIPT}" || true)
  if [[ "\${topology_count}" != 1 ]]; then
    echo "[ERROR][${node}] Refusing stop: control script must contain one export \${topology_name}= record." >&2
    exit 2
  fi
done
embedded_nnodes=\$(sed -nE 's/^export NNODES=([0-9]+)$/\1/p' "\${CONTROL_SCRIPT}")
embedded_node_rank=\$(sed -nE 's/^export NODE_RANK=([0-9]+)$/\1/p' "\${CONTROL_SCRIPT}")
embedded_nproc=\$(sed -nE 's/^export NPROC=([0-9]+)$/\1/p' "\${CONTROL_SCRIPT}")
embedded_master_addr=\$(sed -nE 's/^export MASTER_ADDR=([^[:space:]]+)$/\1/p' "\${CONTROL_SCRIPT}")
embedded_master_port=\$(sed -nE 's/^export MASTER_PORT=([0-9]+)$/\1/p' "\${CONTROL_SCRIPT}")
embedded_provenance_port=\$(sed -nE 's/^export HOLOSOMA_PROVENANCE_MASTER_PORT=([0-9]+)$/\1/p' "\${CONTROL_SCRIPT}")
if [[ ! "\${embedded_nnodes}" =~ ^[1-9][0-9]*$ \
      || ! "\${embedded_node_rank}" =~ ^(0|[1-9][0-9]*)$ \
      || ! "\${embedded_nproc}" =~ ^[1-9][0-9]*$ \
      || "\${embedded_nproc}" != $(quote "${NPROC}") \
      || "\${embedded_master_addr}" != $(quote "${MASTER_ADDR}") \
      || "\${embedded_master_port}" != $(quote "${MASTER_PORT}") \
      || "\${embedded_provenance_port}" != $(quote "${HOLOSOMA_PROVENANCE_MASTER_PORT}") ]]; then
  echo "[ERROR][${node}] Refusing stop: embedded session/log/topology/master/ports do not match this controller invocation." >&2
  exit 2
fi
session_export_count=\$(grep -Ec '^export SESSION=' "\${CONTROL_SCRIPT}" || true)
run_stamp_export_count=\$(grep -Ec '^export RUN_STAMP=' "\${CONTROL_SCRIPT}" || true)
log_dir_export_count=\$(grep -Ec '^export HOLOSOMA_ACTIVE_LOG_DIR=' "\${CONTROL_SCRIPT}" || true)
stop_contract=""
if [[ "\${session_export_count}" == 1 \
      && "\${run_stamp_export_count}" == 1 \
      && "\${log_dir_export_count}" == 1 ]]; then
  stop_contract=modern
  embedded_session=\$(sed -nE 's/^export SESSION=([A-Za-z0-9][A-Za-z0-9_.-]{0,127})$/\1/p' "\${CONTROL_SCRIPT}")
  embedded_run_stamp=\$(sed -nE 's/^export RUN_STAMP=([A-Za-z0-9][A-Za-z0-9_.-]{0,127})$/\1/p' "\${CONTROL_SCRIPT}")
  embedded_log_dir=\$(sed -nE 's#^export HOLOSOMA_ACTIVE_LOG_DIR=(logs/batch_ne/[A-Za-z0-9][A-Za-z0-9_.-]{0,254})\$#\1#p' "\${CONTROL_SCRIPT}")
  if [[ "\${embedded_session}" != $(quote "${SESSION}") \
        || "\${embedded_log_dir}" != "logs/batch_ne/\${embedded_session}_\${embedded_run_stamp}" \
        || "\${embedded_log_dir}" != "\${active_log_dir}" ]]; then
    echo "[ERROR][${node}] Refusing stop: embedded session/log/topology/master/ports do not match this controller invocation." >&2
    exit 2
  fi
  entrypoint_layout_count=0
  for entrypoint_field in \
      DISTILL_AS_ENTRYPOINT DISTILL_AS_ENTRYPOINT_PATH \
      DISTILL_AS_ENTRYPOINT_SHA256 DISTILL_AS_FORMAL_FRESH; do
    entrypoint_field_count=\$(grep -Ec "^export \${entrypoint_field}=" "\${CONTROL_SCRIPT}" || true)
    if [[ "\${entrypoint_field_count}" == 1 ]]; then
      entrypoint_layout_count=\$((entrypoint_layout_count + 1))
    elif [[ "\${entrypoint_field_count}" != 0 ]]; then
      echo "[ERROR][${node}] Refusing stop: control script contains a non-unique \${entrypoint_field} identity." >&2
      exit 2
    fi
  done
  if [[ "\${entrypoint_layout_count}" == 4 ]]; then
    embedded_entrypoint=\$(sed -nE 's/^export DISTILL_AS_ENTRYPOINT=(distill_as_(button|dual_button)_solid\.sh)$/\1/p' "\${CONTROL_SCRIPT}")
    # The end anchor must be escaped while this unquoted controller heredoc is
    # expanded; otherwise the shell substitutes its positional-argument count
    # and corrupts the remote sed expression before stop preflight runs.
    embedded_entrypoint_path=\$(sed -nE 's#^export DISTILL_AS_ENTRYPOINT_PATH=([^[:space:]]+)\$#\1#p' "\${CONTROL_SCRIPT}")
    embedded_entrypoint_sha256=\$(sed -nE 's/^export DISTILL_AS_ENTRYPOINT_SHA256=([0-9a-f]{64})$/\1/p' "\${CONTROL_SCRIPT}")
    embedded_formal_fresh=\$(sed -nE 's/^export DISTILL_AS_FORMAL_FRESH=([01])$/\1/p' "\${CONTROL_SCRIPT}")
    expected_entrypoint_path=$(quote "${REMOTE_RUN_ROOT}")/"\${active_snapshot}/\${embedded_entrypoint}"
    expected_entrypoint_log=$(quote "${REMOTE_RUN_ROOT}")/"\${active_snapshot}/\${active_log_dir}/node_\${embedded_node_rank}_${node}.log"
    expected_entrypoint_pipeline='bash "\${DISTILL_AS_ENTRYPOINT_PATH}" "\${TRAIN_EXTRA_ARGS[@]}" 2>&1 | tee -a '
    expected_entrypoint_pipeline+="\${expected_entrypoint_log}"
    if [[ -z "\${embedded_entrypoint}" \
          || "\${embedded_entrypoint_path}" != "\${expected_entrypoint_path}" \
          || -z "\${embedded_entrypoint_sha256}" \
          || -z "\${embedded_formal_fresh}" \
          || ( "\${embedded_formal_fresh}" == 1 \
            && "\${embedded_entrypoint}" != distill_as_dual_button_solid.sh ) \
          || ! -f "\${embedded_entrypoint_path}" \
          || -L "\${embedded_entrypoint_path}" \
          || "\$(sha256sum -- "\${embedded_entrypoint_path}" | awk '{print \$1}')" != "\${embedded_entrypoint_sha256}" \
          || "\$(grep -Fxc -- "\${expected_entrypoint_pipeline}" "\${CONTROL_SCRIPT}" || true)" != 1 ]]; then
      echo "[ERROR][${node}] Refusing stop: selected distill entrypoint identity/pipeline is incomplete, mismatched, or unauthenticated." >&2
      exit 2
    fi
  elif [[ "\${entrypoint_layout_count}" != 0 ]]; then
    echo "[ERROR][${node}] Refusing stop: selected distill entrypoint exports form a partial layout." >&2
    exit 2
  fi
elif [[ "\${session_export_count}" == 0 \
        && "\${run_stamp_export_count}" == 0 \
        && "\${log_dir_export_count}" == 0 ]]; then
  stop_contract=legacy
  if [[ -z $(quote "${LEGACY_STOP_EXPECTED_SNAPSHOT_ID}") \
        || -z $(quote "${LEGACY_STOP_EXPECTED_TOKEN}") \
        || -z $(quote "${LEGACY_STOP_EXPECTED_EPOCH}") \
        || -z $(quote "${LEGACY_STOP_EXPECTED_RUN_STAMP}") \
        || -z $(quote "${LEGACY_STOP_EXPECTED_TARGET}") ]]; then
    echo "[ERROR][${node}] Legacy control layout requires all five explicit LEGACY_STOP_EXPECTED_* fields." >&2
    exit 2
  fi
  expected_legacy_log_dir=$(quote "logs/batch_ne/${SESSION}_${LEGACY_STOP_EXPECTED_RUN_STAMP}")
  if [[ "\${active_snapshot}" != $(quote "${LEGACY_STOP_EXPECTED_SNAPSHOT_ID}") \
        || "\${active_token}" != $(quote "${LEGACY_STOP_EXPECTED_TOKEN}") \
        || "\${active_epoch}" != $(quote "${LEGACY_STOP_EXPECTED_EPOCH}") \
        || "\${active_log_dir}" != "\${expected_legacy_log_dir}" \
        || "\${active_target}" != $(quote "${LEGACY_STOP_EXPECTED_TARGET}") ]]; then
    echo "[ERROR][${node}] Legacy active identity does not match the explicit five-field stop authorization." >&2
    exit 2
  fi
  for forbidden_export in \
      HOLOSOMA_LAUNCH_TOKEN HOLOSOMA_COMMAND_SHA256 HOLOSOMA_LAUNCH_EPOCH; do
    if [[ "\$(grep -Ec "^export \${forbidden_export}=" "\${CONTROL_SCRIPT}" || true)" != 0 ]]; then
      echo "[ERROR][${node}] Legacy control layout is partial/hybrid at export \${forbidden_export}." >&2
      exit 2
    fi
  done
  expected_run_repo=$(quote "${REMOTE_RUN_ROOT}/${LEGACY_STOP_EXPECTED_SNAPSHOT_ID}")
  expected_absolute_log_dir="\${expected_run_repo}/\${expected_legacy_log_dir}"
  expected_log_file="\${expected_absolute_log_dir}/node_\${embedded_node_rank}_${node}.log"
  expected_manifest_sha=\${active_snapshot#src-}
  legacy_target_count=\$(grep -Ec '^export TARGET_LEARNING_ITERATION=' "\${CONTROL_SCRIPT}" || true)
  embedded_legacy_target=\$(sed -nE 's/^export TARGET_LEARNING_ITERATION=([0-9]+)$/\1/p' "\${CONTROL_SCRIPT}")
  if [[ "\${legacy_target_count}" != 1 \
        || "\${embedded_legacy_target}" != "\${active_target}" \
        || "\$(grep -Fxc -- "cd \${expected_run_repo}" "\${CONTROL_SCRIPT}" || true)" != 1 \
        || "\$(grep -Fxc -- "mkdir -p \${expected_absolute_log_dir}" "\${CONTROL_SCRIPT}" || true)" != 1 \
        || "\$(grep -Fxc -- "export HOLOSOMA_SOURCE_SNAPSHOT_ID=\${active_snapshot}" "\${CONTROL_SCRIPT}" || true)" != 1 \
        || "\$(grep -Fxc -- "export HOLOSOMA_SOURCE_MANIFEST_SHA256=\${expected_manifest_sha}" "\${CONTROL_SCRIPT}" || true)" != 1 \
        || "\$(grep -Fxc -- $(quote "export RUN_NAME=${SESSION}") "\${CONTROL_SCRIPT}" || true)" != 1 \
        || "\$(grep -Fxc -- "echo \"[INFO][${node}] master=\${embedded_master_addr}:\${embedded_master_port} log=\${expected_log_file}\"" "\${CONTROL_SCRIPT}" || true)" != 1 ]]; then
    echo "[ERROR][${node}] Legacy hash-bound control script does not prove the exact source/session/log identity." >&2
    exit 2
  fi
  expected_legacy_pipeline='bash distill_as_button_solid.sh "\${TRAIN_EXTRA_ARGS[@]}" 2>&1 | tee -a '
  expected_legacy_pipeline+="\${expected_log_file}"
  tee_record=\$(grep -Fx -- "\${expected_legacy_pipeline}" "\${CONTROL_SCRIPT}" || true)
  last_nonempty_record=\$(awk 'NF { record = \$0 } END { print record }' "\${CONTROL_SCRIPT}")
  if [[ -z "\${tee_record}" || "\${tee_record}" == *$'\n'* \
        || "\${tee_record}" != "\${expected_legacy_pipeline}" \
        || "\${last_nonempty_record}" != "\${expected_legacy_pipeline}" ]]; then
    echo "[ERROR][${node}] Legacy hash-bound control must end in one exact foreground training/tee pipeline." >&2
    exit 2
  fi
else
  echo "[ERROR][${node}] Refusing stop: SESSION/RUN_STAMP/HOLOSOMA_ACTIVE_LOG_DIR exports form a partial layout." >&2
  exit 2
fi
if [[ "\${stop_contract}" != legacy \
      && "\${active_state_legacy_mode:-0}" == 1 ]]; then
  echo "[ERROR][${node}] Non-private active-state mode is supported only for an explicitly authorized legacy control layout." >&2
  exit 2
fi
tmux_running=0
legacy_exact_pane_loaded=0
if tmux has-session -t $(quote "${SESSION}") 2>/dev/null; then
  tmux_running=1
  $(tmux_ownership_helpers)
  if [[ "\${stop_contract}" == modern ]]; then
    if ! tmux_session_is_owned_for_cleanup \
        $(quote "${SESSION}") "\${active_token}" "\${active_command_sha}" "\${active_epoch}"; then
      echo "[ERROR][${node}] Refusing to stop ${SESSION}: same-name tmux is not bound to the active token/command/epoch across atomic environment/options." >&2
      exit 2
    fi
  else
    if ! load_tmux_ownership_identity $(quote "${SESSION}") \
        || ! tmux_atomic_identity_is_absent \
        || ! tmux_options_match_exactly \
          "\${active_token}" "\${active_command_sha}" "\${active_epoch}"; then
      echo "[ERROR][${node}] Refusing legacy stop: tmux options-only ownership does not match active identity." >&2
      exit 2
    fi
    # A running legacy owner must still expose its one exact live pane.  Once
    # the durable state is stopping, group cleanup may already have killed the
    # pane root; retry ownership is then the exact options + active + receipt
    # triple, and requiring a live pane would make safe closure impossible.
    if [[ "\${active_phase}" == running ]]; then
      legacy_load_exact_single_pane \
        $(quote "${SESSION}") "\${CONTROL_SCRIPT}" || {
        echo "[ERROR][${node}] Refusing legacy stop: running tmux lacks one exact control-script pane." >&2
        exit 2
      }
      legacy_exact_pane_loaded=1
    fi
  fi
else
  tmux_rc=\$?
  [[ "\${tmux_rc}" == 1 ]] || {
    echo "[ERROR][${node}] Could not verify tmux absence during stop-identity preflight (rc=\${tmux_rc})." >&2
    exit 2
  }
fi
if [[ "\${stop_contract}" == legacy ]]; then
  capture_receipt=\$(legacy_capture_receipt_path \
    "\${ACTIVE_STATE}" "\${active_token}" "\${active_command_sha}" "\${active_epoch}")
  capture_intent=\$(legacy_capture_intent_path "\${capture_receipt}")
  receipt_incoming="\${capture_receipt}.in"
  intent_incoming="\${capture_intent}.in"
  for candidate in "\${ACTIVE_STATE}".legacy-processes.*; do
    [[ -e "\${candidate}" || -L "\${candidate}" ]] || continue
    case "\${candidate}" in
      "\${capture_receipt}"|"\${capture_intent}") ;;
      "\${receipt_incoming}")
        legacy_validate_safe_publication_residue "\${candidate}" 10000000 || exit
        ;;
      "\${intent_incoming}")
        legacy_validate_safe_publication_residue "\${candidate}" 8192 || exit
        ;;
      *)
        echo "[ERROR][${node}] Refusing legacy stop beside an unknown/old process receipt or incoming path." >&2
        exit 2
        ;;
    esac
  done
  if [[ ( -e "\${capture_receipt}" || -L "\${capture_receipt}" ) \
        && ( -e "\${receipt_incoming}" || -L "\${receipt_incoming}" ) ]] \
      || [[ ( -e "\${capture_intent}" || -L "\${capture_intent}" ) \
        && ( -e "\${intent_incoming}" || -L "\${intent_incoming}" ) ]]; then
    echo "[ERROR][${node}] Canonical legacy metadata and its incoming path coexist." >&2
    exit 2
  fi
  if [[ -e "\${capture_receipt}" || -L "\${capture_receipt}" ]]; then
    legacy_load_capture_receipt \
      "\${capture_receipt}" "\${active_token}" "\${active_epoch}" \
      "\${active_command_sha}" "\${active_snapshot}" "\${active_log_dir}" \
      "\${active_target}" || exit
    if (( legacy_exact_pane_loaded == 1 )) \
        && [[ "\${legacy_receipt_root_pid}" != "\${legacy_pane_pid}" \
          || "\${legacy_receipt_root_start}" != "\${legacy_pane_start}" \
          || "\${legacy_receipt_cgroup_fingerprint}" != "\${legacy_pane_cgroup_fingerprint}" ]]; then
      echo "[ERROR][${node}] Legacy receipt root does not match the exact current pane identity." >&2
      exit 2
    fi
    if [[ -e "\${capture_intent}" || -L "\${capture_intent}" ]]; then
      legacy_load_capture_intent \
        "\${capture_intent}" "\${active_token}" "\${active_epoch}" \
        "\${active_command_sha}" "\${active_snapshot}" "\${active_log_dir}" \
        "\${active_target}" || exit
      if [[ "\${legacy_intent_root_pid}" != "\${legacy_receipt_root_pid}" \
            || "\${legacy_intent_root_start}" != "\${legacy_receipt_root_start}" \
            || "\${legacy_intent_cgroup_path}" != "\${legacy_receipt_cgroup_path}" \
            || "\${legacy_intent_cgroup_dev}" != "\${legacy_receipt_cgroup_dev}" \
            || "\${legacy_intent_cgroup_ino}" != "\${legacy_receipt_cgroup_ino}" ]]; then
        echo "[ERROR][${node}] Legacy freeze intent and process receipt name different boundaries." >&2
        exit 2
      fi
    fi
    legacy_cgroup_fingerprint=\${legacy_receipt_cgroup_fingerprint}
    if [[ "\${active_phase}" == stopped ]]; then
      (( tmux_running == 0 )) || {
        echo "[ERROR][${node}] Legacy stopped metadata still has a live tmux session." >&2
        exit 2
      }
      legacy_verify_terminal_receipt_closure || exit
    elif [[ "\${active_phase}" == stopping ]]; then
      legacy_validate_frozen_receipt_cgroup "\${CONTROL_SCRIPT}" || exit
    elif (( legacy_exact_pane_loaded == 1 )); then
      legacy_read_cgroup_frozen_state \
        "\${legacy_receipt_cgroup_path}" "\${legacy_receipt_cgroup_dev}" \
        "\${legacy_receipt_cgroup_ino}" || exit
      if [[ "\${legacy_cgroup_freeze_requested}" == 1 \
            && "\${legacy_cgroup_frozen}" == 1 ]]; then
        legacy_validate_frozen_receipt_cgroup "\${CONTROL_SCRIPT}" || exit
      elif [[ "\${legacy_cgroup_freeze_requested}" != 0 \
            || "\${legacy_cgroup_frozen}" != 0 ]]; then
        echo "[ERROR][${node}] Running legacy receipt has an unrecoverable freezer transition." >&2
        exit 2
      fi
    else
      legacy_validate_frozen_receipt_cgroup "\${CONTROL_SCRIPT}" || exit
    fi
  else
    if (( tmux_running == 0 )) || [[ "\${active_phase}" != running ]] \
        || (( legacy_exact_pane_loaded != 1 )); then
      echo "[ERROR][${node}] Legacy stop without a live exact pane requires a valid process-capture receipt." >&2
      exit 2
    fi
    legacy_cgroup_fingerprint=\${legacy_pane_cgroup_fingerprint}
    legacy_read_cgroup_frozen_state \
      "\${legacy_pane_cgroup_path}" "\${legacy_pane_cgroup_dev}" \
      "\${legacy_pane_cgroup_ino}" || exit
    legacy_pane_cgroup_freeze_requested=\${legacy_cgroup_freeze_requested}
    legacy_pane_cgroup_frozen=\${legacy_cgroup_frozen}
    if [[ -e "\${capture_intent}" || -L "\${capture_intent}" ]]; then
      legacy_load_capture_intent \
        "\${capture_intent}" "\${active_token}" "\${active_epoch}" \
        "\${active_command_sha}" "\${active_snapshot}" "\${active_log_dir}" \
        "\${active_target}" || exit
      if [[ "\${legacy_intent_root_pid}" != "\${legacy_pane_pid}" \
            || "\${legacy_intent_root_start}" != "\${legacy_pane_start}" \
            || "\${legacy_intent_cgroup_path}" != "\${legacy_pane_cgroup_path}" \
            || "\${legacy_intent_cgroup_dev}" != "\${legacy_pane_cgroup_dev}" \
            || "\${legacy_intent_cgroup_ino}" != "\${legacy_pane_cgroup_ino}" ]]; then
        echo "[ERROR][${node}] Legacy freeze intent does not match the exact current pane cgroup." >&2
        exit 2
      fi
    elif [[ -e "\${receipt_incoming}" || -L "\${receipt_incoming}" ]]; then
      echo "[ERROR][${node}] Receipt incoming residue lacks its durable exact freeze intent." >&2
      exit 2
    elif [[ -e "\${intent_incoming}" || -L "\${intent_incoming}" ]]; then
      if [[ "\${legacy_pane_cgroup_freeze_requested}" != 0 \
            || "\${legacy_pane_cgroup_frozen}" != 0 ]]; then
        echo "[ERROR][${node}] Unpublished intent residue cannot authorize a frozen cgroup." >&2
        exit 2
      fi
    elif [[ "\${legacy_pane_cgroup_freeze_requested}" != 0 \
          || "\${legacy_pane_cgroup_frozen}" != 0 ]]; then
      echo "[ERROR][${node}] Legacy pane is frozen without a durable exact intent/receipt." >&2
      exit 2
    fi
    if [[ -e "\${capture_intent}" || -L "\${capture_intent}" ]] \
        && [[ ( "\${legacy_pane_cgroup_freeze_requested}" != 0 \
              || "\${legacy_pane_cgroup_frozen}" != 0 ) \
          && ( "\${legacy_pane_cgroup_freeze_requested}" != 1 \
              || "\${legacy_pane_cgroup_frozen}" != 1 ) ]]; then
      echo "[ERROR][${node}] Durable legacy intent observes an incomplete freezer transition." >&2
      exit 2
    fi
  fi
else
  legacy_cgroup_fingerprint=-
fi
if [[ "\${stop_contract}" == legacy && "\${active_state_legacy_mode:-0}" == 1 ]]; then
  # This compatibility path is reachable only after the controller supplied
  # all five explicit legacy authorization fields and the immutable control,
  # tmux, pane/cgroup, and any receipt identity were revalidated above.
  migrate_loaded_active_state_to_private "\${ACTIVE_STATE}" || exit
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "\${active_token}" "\${active_epoch}" "\${active_snapshot}" \
  "\${active_log_dir}" "\${active_target}" "\${active_command_sha}" \
  "\${embedded_nnodes}" "\${embedded_node_rank}" "\${stop_contract}" \
  "\${legacy_cgroup_fingerprint}"
EOF
)
  remote_run_bounded "${node}" "${cmd}" "${LAUNCH_CLEANUP_TIMEOUT_SECONDS}"
}

stop_node_modern_only() {
  local node="$1"
  local expected_token="$2"
  local expected_epoch="$3"
  local expected_snapshot="$4"
  local expected_log_dir="$5"
  local expected_target="$6"
  local expected_command_sha="$7"
  local tmux_lock_path
  tmux_lock_path="${REMOTE_RUN_ROOT}/.active/.locks/holosoma-tmux-$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}').lock"
  local active_state_path
  active_state_path="${REMOTE_RUN_ROOT}/.active/$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}')_${node}.state"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
$(private_lifecycle_file_validation_helpers)
open_private_lifecycle_lock $(quote "${tmux_lock_path}") 8
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 8; then
  echo "[ERROR][${node}] Timed out acquiring the ${SESSION} lifecycle stop lock." >&2
  exit 1
fi
ACTIVE_STATE=$(quote "${active_state_path}")
$(active_state_validation_helpers)
$(launch_process_closure_helpers)
if ! load_active_state_v2_exact "\${ACTIVE_STATE}" \
    || ! active_state_has_session_namespace $(quote "${SESSION}") \
    || [[ "\${active_token}" != $(quote "${expected_token}") \
      || "\${active_epoch}" != $(quote "${expected_epoch}") \
      || "\${active_snapshot}" != $(quote "${expected_snapshot}") \
      || "\${active_log_dir}" != $(quote "${expected_log_dir}") \
      || "\${active_target}" != $(quote "${expected_target}") \
      || "\${active_command_sha}" != $(quote "${expected_command_sha}") ]]; then
  echo "[ERROR][${node}] Refusing to stop ${SESSION}: active identity changed after all-node preflight." >&2
  exit 2
fi
case "\${active_phase}" in
  running|stopping|stopped) ;;
  *)
    echo "[ERROR][${node}] Refusing normal stop from lifecycle phase=\${active_phase}." >&2
    exit 2
    ;;
esac
tmux_running=0
legacy_exact_pane_loaded=0
if tmux has-session -t $(quote "${SESSION}") 2>/dev/null; then
  tmux_running=1
  $(tmux_ownership_helpers)
  if ! tmux_session_is_owned_for_cleanup \
      $(quote "${SESSION}") "\${active_token}" "\${active_command_sha}" "\${active_epoch}"; then
    echo "[ERROR][${node}] Refusing to stop ${SESSION}: same-name tmux is not bound to the expected identity." >&2
    exit 2
  fi
else
  tmux_rc=\$?
  [[ "\${tmux_rc}" == 1 ]] || {
    echo "[ERROR][${node}] Could not inspect tmux during stop cleanup (rc=\${tmux_rc})." >&2
    exit 2
  }
fi
if [[ "\${active_phase}" == stopped && "\${tmux_running}" == 0 ]] \
    && verify_no_launch_identity_processes \
      "\${active_token}" "\${active_command_sha}" "\${active_epoch}"; then
  echo "[INFO][${node}] ${SESSION} already has verified stopped closure"
  exit 0
fi
incoming="\${ACTIVE_STATE}.incoming.stopping.\$\$"
printf '2\tstopping\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "\${active_snapshot}" "\${active_log_dir}" "\${active_target}" "\${active_token}" \
  "\${active_command_sha}" "\${active_epoch}" > "\${incoming}"
mv -T "\${incoming}" "\${ACTIVE_STATE}"
validate_private_state_file_metadata "\${ACTIVE_STATE}" 4096 0
if (( tmux_running == 1 )); then
  tmux kill-session -t $(quote "${SESSION}")
fi
terminate_launch_identity_processes_bounded \
  "\${active_token}" "\${active_command_sha}" "\${active_epoch}"
if tmux has-session -t $(quote "${SESSION}") 2>/dev/null; then
  echo "[ERROR][${node}] Same-name tmux survived stop cleanup." >&2
  exit 2
else
  tmux_rc=\$?
  [[ "\${tmux_rc}" == 1 ]] || {
    echo "[ERROR][${node}] Could not verify tmux absence after stop cleanup (rc=\${tmux_rc})." >&2
    exit 2
  }
fi
verify_no_launch_identity_processes \
  "\${active_token}" "\${active_command_sha}" "\${active_epoch}"
incoming="\${ACTIVE_STATE}.incoming.stopped.\$\$"
printf '2\tstopped\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "\${active_snapshot}" "\${active_log_dir}" "\${active_target}" "\${active_token}" \
  "\${active_command_sha}" "\${active_epoch}" > "\${incoming}"
mv -T "\${incoming}" "\${ACTIVE_STATE}"
validate_private_state_file_metadata "\${ACTIVE_STATE}" 4096 0
echo "[INFO][${node}] stopped ${SESSION} with exact process closure"
EOF
)
  remote_run_bounded "${node}" "${cmd}" "${LAUNCH_CLEANUP_TIMEOUT_SECONDS}"
}

stop_node() {
  local node="$1"
  local expected_token="$2"
  local expected_epoch="$3"
  local expected_snapshot="$4"
  local expected_log_dir="$5"
  local expected_target="$6"
  local expected_command_sha="$7"
  local expected_contract="$8"
  local expected_cgroup_fingerprint="$9"
  local stop_mode="${10:-commit}"
  local remote_stop_timeout remote_stop_deadline_budget
  calculate_remote_mutation_bounds "${LAUNCH_CLEANUP_TIMEOUT_SECONDS}"
  remote_stop_timeout=${REMOTE_MUTATION_TIMEOUT_SECONDS}
  remote_stop_deadline_budget=$((remote_stop_timeout - 1))
  (( remote_stop_deadline_budget >= 0 )) || remote_stop_deadline_budget=0
  local tmux_lock_path
  tmux_lock_path="${REMOTE_RUN_ROOT}/.active/.locks/holosoma-tmux-$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}').lock"
  local active_state_path
  active_state_path="${REMOTE_RUN_ROOT}/.active/$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}')_${node}.state"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
# Share one absolute remote-shell deadline across lock acquisition, cgroup.kill
# convergence, idempotent stopping retries, tmux cleanup, and PID reap.  It is
# one second earlier than the mutation wrapper's TERM deadline so terminal
# metadata can be published before the controller-side execution bound.
LEGACY_STOP_CLEANUP_DEADLINE_SECONDS=\$((SECONDS + $(quote "${remote_stop_deadline_budget}")))
$(private_lifecycle_file_validation_helpers)
open_private_lifecycle_lock $(quote "${tmux_lock_path}") 8
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 8; then
  echo "[ERROR][${node}] Timed out acquiring the ${SESSION} lifecycle stop lock." >&2
  exit 1
fi
ACTIVE_STATE=$(quote "${active_state_path}")
STOP_CONTRACT=$(quote "${expected_contract}")
EXPECTED_CGROUP_FINGERPRINT=$(quote "${expected_cgroup_fingerprint}")
STOP_MODE=$(quote "${stop_mode}")
$(active_state_validation_helpers)
$(launch_process_closure_helpers)
$(legacy_stop_process_helpers)
if [[ "\${STOP_CONTRACT}" != modern && "\${STOP_CONTRACT}" != legacy ]]; then
  echo "[ERROR][${node}] Refusing stop with unknown process-closure contract=\${STOP_CONTRACT}." >&2
  exit 2
fi
if [[ ( "\${STOP_CONTRACT}" == modern && "\${EXPECTED_CGROUP_FINGERPRINT}" != - ) \
      || ( "\${STOP_CONTRACT}" == legacy \
        && ! "\${EXPECTED_CGROUP_FINGERPRINT}" =~ ^[0-9a-f]{64}$ ) ]]; then
  echo "[ERROR][${node}] Refusing stop with a malformed contract/cgroup fingerprint pair." >&2
  exit 2
fi
if [[ "\${STOP_MODE}" != commit && "\${STOP_MODE}" != arm ]]; then
  echo "[ERROR][${node}] Refusing stop with unknown transaction mode=\${STOP_MODE}." >&2
  exit 2
fi
if [[ "\${STOP_MODE}" == arm && "\${STOP_CONTRACT}" != legacy ]]; then
  echo "[ERROR][${node}] The all-node arm barrier is valid only for legacy cgroup stop." >&2
  exit 2
fi
if ! load_active_state_v2_exact "\${ACTIVE_STATE}" \
    || ! active_state_has_session_namespace $(quote "${SESSION}") \
    || [[ "\${active_token}" != $(quote "${expected_token}") \
      || "\${active_epoch}" != $(quote "${expected_epoch}") \
      || "\${active_snapshot}" != $(quote "${expected_snapshot}") \
      || "\${active_log_dir}" != $(quote "${expected_log_dir}") \
      || "\${active_target}" != $(quote "${expected_target}") \
      || "\${active_command_sha}" != $(quote "${expected_command_sha}") ]]; then
  echo "[ERROR][${node}] Refusing to stop ${SESSION}: active identity changed after all-node preflight." >&2
  exit 2
fi
case "\${active_phase}" in
  running|stopping|stopped) ;;
  *)
    echo "[ERROR][${node}] Refusing normal stop from lifecycle phase=\${active_phase}." >&2
    exit 2
    ;;
esac
CONTROL_SCRIPT=$(quote "${REMOTE_RUN_ROOT}")/"\${active_snapshot}/.run_control/train-\${active_command_sha}.sh"
if [[ ! -f "\${CONTROL_SCRIPT}" || -L "\${CONTROL_SCRIPT}" \
      || "\$(sha256sum "\${CONTROL_SCRIPT}" | awk '{print \$1}')" != "\${active_command_sha}" ]]; then
  echo "[ERROR][${node}] Immutable control script changed after the all-node stop preflight." >&2
  exit 2
fi
tmux_running=0
legacy_exact_pane_loaded=0
if tmux has-session -t $(quote "${SESSION}") 2>/dev/null; then
  tmux_running=1
  $(tmux_ownership_helpers)
  if [[ "\${STOP_CONTRACT}" == modern ]]; then
    if ! tmux_session_is_owned_for_cleanup \
        $(quote "${SESSION}") "\${active_token}" "\${active_command_sha}" "\${active_epoch}"; then
      echo "[ERROR][${node}] Refusing modern stop: tmux identity changed after all-node preflight." >&2
      exit 2
    fi
  else
    if ! load_tmux_ownership_identity $(quote "${SESSION}") \
        || ! tmux_atomic_identity_is_absent \
        || ! tmux_options_match_exactly \
          "\${active_token}" "\${active_command_sha}" "\${active_epoch}"; then
      echo "[ERROR][${node}] Refusing legacy stop: exact options ownership changed after all-node preflight." >&2
      exit 2
    fi
    if [[ "\${active_phase}" == running ]]; then
      legacy_load_exact_single_pane \
        $(quote "${SESSION}") "\${CONTROL_SCRIPT}" || {
        echo "[ERROR][${node}] Refusing legacy stop: running pane ownership changed after all-node preflight." >&2
        exit 2
      }
      legacy_exact_pane_loaded=1
    fi
  fi
else
  tmux_rc=\$?
  [[ "\${tmux_rc}" == 1 ]] || {
    echo "[ERROR][${node}] Could not inspect tmux during stop cleanup (rc=\${tmux_rc})." >&2
    exit 2
  }
fi
if [[ "\${STOP_CONTRACT}" == modern ]]; then
  if [[ "\${active_phase}" == stopped && "\${tmux_running}" == 0 ]] \
      && verify_no_launch_identity_processes \
        "\${active_token}" "\${active_command_sha}" "\${active_epoch}"; then
    echo "[INFO][${node}] ${SESSION} already has verified stopped closure"
    exit 0
  fi
  incoming="\${ACTIVE_STATE}.incoming.stopping.\$\$"
  printf '2\tstopping\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "\${active_snapshot}" "\${active_log_dir}" "\${active_target}" "\${active_token}" \
    "\${active_command_sha}" "\${active_epoch}" > "\${incoming}"
  mv -T "\${incoming}" "\${ACTIVE_STATE}"
  validate_private_state_file_metadata "\${ACTIVE_STATE}" 4096 0
  if (( tmux_running == 1 )); then
    tmux kill-session -t $(quote "${SESSION}")
  fi
  terminate_launch_identity_processes_bounded \
    "\${active_token}" "\${active_command_sha}" "\${active_epoch}"
  if tmux has-session -t $(quote "${SESSION}") 2>/dev/null; then
    echo "[ERROR][${node}] Same-name tmux survived modern stop cleanup." >&2
    exit 2
  else
    tmux_rc=\$?
    [[ "\${tmux_rc}" == 1 ]] || exit 2
  fi
  verify_no_launch_identity_processes \
    "\${active_token}" "\${active_command_sha}" "\${active_epoch}"
else
  capture_receipt=\$(legacy_capture_receipt_path \
    "\${ACTIVE_STATE}" "\${active_token}" "\${active_command_sha}" "\${active_epoch}")
  capture_intent=\$(legacy_capture_intent_path "\${capture_receipt}")
  receipt_incoming="\${capture_receipt}.in"
  intent_incoming="\${capture_intent}.in"
  legacy_receipt_loaded=0
  legacy_recapture_mode=none
  legacy_bound_cgroup_fingerprint=""
  for candidate in "\${ACTIVE_STATE}".legacy-processes.*; do
    [[ -e "\${candidate}" || -L "\${candidate}" ]] || continue
    case "\${candidate}" in
      "\${capture_receipt}"|"\${capture_intent}") ;;
      "\${receipt_incoming}")
        legacy_validate_safe_publication_residue "\${candidate}" 10000000 || exit
        ;;
      "\${intent_incoming}")
        legacy_validate_safe_publication_residue "\${candidate}" 8192 || exit
        ;;
      *)
        echo "[ERROR][${node}] Refusing legacy stop beside an unknown/old receipt or incoming path." >&2
        exit 2
        ;;
    esac
  done
  if [[ ( -e "\${capture_receipt}" || -L "\${capture_receipt}" ) \
        && ( -e "\${receipt_incoming}" || -L "\${receipt_incoming}" ) ]] \
      || [[ ( -e "\${capture_intent}" || -L "\${capture_intent}" ) \
        && ( -e "\${intent_incoming}" || -L "\${intent_incoming}" ) ]]; then
    echo "[ERROR][${node}] Canonical legacy metadata and its incoming path coexist." >&2
    exit 2
  fi
  if [[ -e "\${capture_receipt}" || -L "\${capture_receipt}" ]]; then
    legacy_load_capture_receipt \
      "\${capture_receipt}" "\${active_token}" "\${active_epoch}" \
      "\${active_command_sha}" "\${active_snapshot}" "\${active_log_dir}" \
      "\${active_target}" || exit
    legacy_receipt_loaded=1
    legacy_bound_cgroup_fingerprint=\${legacy_receipt_cgroup_fingerprint}
    if (( legacy_exact_pane_loaded == 1 )) \
        && [[ "\${legacy_receipt_root_pid}" != "\${legacy_pane_pid}" \
          || "\${legacy_receipt_root_start}" != "\${legacy_pane_start}" \
          || "\${legacy_receipt_cgroup_fingerprint}" != "\${legacy_pane_cgroup_fingerprint}" ]]; then
      echo "[ERROR][${node}] Legacy receipt no longer matches the exact current pane/cgroup." >&2
      exit 2
    fi
    if [[ -e "\${capture_intent}" || -L "\${capture_intent}" ]]; then
      legacy_load_capture_intent \
        "\${capture_intent}" "\${active_token}" "\${active_epoch}" \
        "\${active_command_sha}" "\${active_snapshot}" "\${active_log_dir}" \
        "\${active_target}" || exit
      if [[ "\${legacy_intent_root_pid}" != "\${legacy_receipt_root_pid}" \
            || "\${legacy_intent_root_start}" != "\${legacy_receipt_root_start}" \
            || "\${legacy_intent_cgroup_fingerprint}" != "\${legacy_receipt_cgroup_fingerprint}" ]]; then
        echo "[ERROR][${node}] Legacy receipt and freeze intent name different boundaries." >&2
        exit 2
      fi
    fi
    if [[ "\${legacy_bound_cgroup_fingerprint}" != "\${EXPECTED_CGROUP_FINGERPRINT}" ]]; then
      echo "[ERROR][${node}] Legacy receipt cgroup changed after all-node preflight." >&2
      exit 2
    fi
    if [[ "\${active_phase}" == running ]]; then
      legacy_read_cgroup_frozen_state \
        "\${legacy_receipt_cgroup_path}" "\${legacy_receipt_cgroup_dev}" \
        "\${legacy_receipt_cgroup_ino}" || exit
      if [[ "\${legacy_cgroup_freeze_requested}" == 1 \
            && "\${legacy_cgroup_frozen}" == 1 ]]; then
        legacy_validate_frozen_receipt_cgroup "\${CONTROL_SCRIPT}" || exit
      elif [[ "\${legacy_cgroup_freeze_requested}" == 0 \
            && "\${legacy_cgroup_frozen}" == 0 \
            && "\${legacy_exact_pane_loaded}" == 1 ]]; then
        legacy_recapture_mode=thawed
      else
        echo "[ERROR][${node}] Running legacy receipt has an unrecoverable freezer state." >&2
        exit 2
      fi
    elif [[ "\${active_phase}" == stopping ]]; then
      legacy_validate_frozen_receipt_cgroup "\${CONTROL_SCRIPT}" || exit
    fi
  else
    if [[ "\${active_phase}" != running || "\${tmux_running}" != 1 \
          || "\${legacy_exact_pane_loaded}" != 1 ]]; then
      echo "[ERROR][${node}] Legacy stop without a canonical receipt requires its exact live running pane." >&2
      exit 2
    fi
    legacy_bound_cgroup_fingerprint=\${legacy_pane_cgroup_fingerprint}
    if [[ "\${legacy_bound_cgroup_fingerprint}" != "\${EXPECTED_CGROUP_FINGERPRINT}" ]]; then
      echo "[ERROR][${node}] Legacy pane cgroup changed after all-node preflight." >&2
      exit 2
    fi
    legacy_read_cgroup_frozen_state \
      "\${legacy_pane_cgroup_path}" "\${legacy_pane_cgroup_dev}" \
      "\${legacy_pane_cgroup_ino}" || exit
    legacy_pane_cgroup_freeze_requested=\${legacy_cgroup_freeze_requested}
    legacy_pane_cgroup_frozen=\${legacy_cgroup_frozen}
    if [[ -e "\${capture_intent}" || -L "\${capture_intent}" ]]; then
      legacy_load_capture_intent \
        "\${capture_intent}" "\${active_token}" "\${active_epoch}" \
        "\${active_command_sha}" "\${active_snapshot}" "\${active_log_dir}" \
        "\${active_target}" || exit
      if [[ "\${legacy_intent_root_pid}" != "\${legacy_pane_pid}" \
            || "\${legacy_intent_root_start}" != "\${legacy_pane_start}" \
            || "\${legacy_intent_cgroup_fingerprint}" != "\${legacy_pane_cgroup_fingerprint}" ]]; then
        echo "[ERROR][${node}] Legacy freeze intent does not match the exact current pane/cgroup." >&2
        exit 2
      fi
    elif [[ -e "\${receipt_incoming}" || -L "\${receipt_incoming}" ]]; then
      echo "[ERROR][${node}] Receipt incoming residue lacks its durable exact freeze intent." >&2
      exit 2
    fi
    if [[ "\${legacy_pane_cgroup_freeze_requested}" == 0 \
          && "\${legacy_pane_cgroup_frozen}" == 0 ]]; then
      legacy_recapture_mode=thawed
    elif [[ "\${legacy_pane_cgroup_freeze_requested}" == 1 \
          && "\${legacy_pane_cgroup_frozen}" == 1 \
          && -f "\${capture_intent}" && ! -L "\${capture_intent}" ]]; then
      legacy_recapture_mode=frozen
    else
      echo "[ERROR][${node}] Receipt-free legacy pane has no recoverable intent/freezer state." >&2
      exit 2
    fi
  fi
  legacy_stop_committed=0
  [[ "\${active_phase}" == stopping ]] && legacy_stop_committed=1
  legacy_precommit_cleanup() {
    (( legacy_stop_committed == 0 )) || return 0
    # Close the window between active-state mv and the in-memory commit flag.
    # Unfreeze/delete is permitted only while disk still contains this exact
    # running identity.  stopping/corrupt/unreadable state preserves the
    # durable intent/receipt and effective freeze for a fail-closed retry.
    load_active_state_v2_exact "\${ACTIVE_STATE}" >/dev/null 2>&1 || return 0
    [[ "\${active_phase}" == running \
          && "\${active_snapshot}" == $(quote "${expected_snapshot}") \
          && "\${active_log_dir}" == $(quote "${expected_log_dir}") \
          && "\${active_target}" == $(quote "${expected_target}") \
          && "\${active_token}" == $(quote "${expected_token}") \
          && "\${active_command_sha}" == $(quote "${expected_command_sha}") \
          && "\${active_epoch}" == $(quote "${expected_epoch}") \
          && "\${legacy_bound_cgroup_fingerprint}" == "\${EXPECTED_CGROUP_FINGERPRINT}" ]] \
      || return 0
    legacy_thaw_complete=0
    if [[ -f "\${capture_receipt}" && ! -L "\${capture_receipt}" ]]; then
      legacy_load_capture_receipt \
        "\${capture_receipt}" "\${active_token}" "\${active_epoch}" \
        "\${active_command_sha}" "\${active_snapshot}" "\${active_log_dir}" \
        "\${active_target}" >/dev/null 2>&1 || return 0
      [[ "\${legacy_receipt_cgroup_fingerprint}" == "\${EXPECTED_CGROUP_FINGERPRINT}" ]] \
        || return 0
      if legacy_unfreeze_receipt_cgroup >/dev/null 2>&1; then
        legacy_thaw_complete=1
      fi
    elif [[ -f "\${capture_intent}" && ! -L "\${capture_intent}" ]]; then
      legacy_load_capture_intent \
        "\${capture_intent}" "\${active_token}" "\${active_epoch}" \
        "\${active_command_sha}" "\${active_snapshot}" "\${active_log_dir}" \
        "\${active_target}" >/dev/null 2>&1 || return 0
      [[ "\${legacy_intent_cgroup_fingerprint}" == "\${EXPECTED_CGROUP_FINGERPRINT}" ]] \
        || return 0
      legacy_capture_cgroup_path=\${legacy_intent_cgroup_path}
      legacy_capture_cgroup_dev=\${legacy_intent_cgroup_dev}
      legacy_capture_cgroup_ino=\${legacy_intent_cgroup_ino}
      if legacy_unfreeze_captured >/dev/null 2>&1; then
        legacy_thaw_complete=1
      fi
    elif [[ -n "\${legacy_capture_cgroup_path:-}" \
          && -n "\${legacy_capture_cgroup_dev:-}" \
          && -n "\${legacy_capture_cgroup_ino:-}" ]]; then
      if legacy_unfreeze_captured >/dev/null 2>&1; then
        legacy_thaw_complete=1
      fi
    else
      legacy_thaw_complete=1
    fi
    # Returning to running invalidates every frozen membership snapshot; a
    # later retry must freeze the complete cgroup again.
    if (( legacy_thaw_complete == 1 )); then
      rm -f -- "\${capture_receipt}" "\${capture_intent}" \
        "\${receipt_incoming}" "\${intent_incoming}"
    fi
  }
  trap legacy_precommit_cleanup EXIT
  if [[ "\${legacy_recapture_mode}" == thawed ]]; then
    rm -f -- "\${capture_receipt}" "\${capture_intent}" \
      "\${receipt_incoming}" "\${intent_incoming}" || exit
    legacy_receipt_loaded=0
  elif [[ "\${legacy_recapture_mode}" == frozen ]]; then
    rm -f -- "\${receipt_incoming}" "\${intent_incoming}" || exit
    legacy_receipt_loaded=0
  elif [[ "\${legacy_recapture_mode}" != none ]]; then
    exit 2
  fi
  if (( legacy_receipt_loaded == 0 )); then
    legacy_capture_freeze_and_publish \
      "\${legacy_pane_pid}" "\${legacy_pane_start}" "\${capture_receipt}" \
      "\${active_token}" "\${active_epoch}" "\${active_command_sha}" \
      "\${active_snapshot}" "\${active_log_dir}" "\${active_target}" \
      "\${CONTROL_SCRIPT}" || exit
    legacy_load_capture_receipt \
      "\${capture_receipt}" "\${active_token}" "\${active_epoch}" \
      "\${active_command_sha}" "\${active_snapshot}" "\${active_log_dir}" \
      "\${active_target}" || exit
    legacy_receipt_loaded=1
  fi
  if [[ "\${legacy_receipt_cgroup_fingerprint}" != "\${EXPECTED_CGROUP_FINGERPRINT}" ]]; then
    echo "[ERROR][${node}] Captured legacy cgroup differs from all-node preflight." >&2
    exit 2
  fi
  if [[ -e "\${capture_intent}" || -L "\${capture_intent}" ]]; then
    legacy_load_capture_intent \
      "\${capture_intent}" "\${active_token}" "\${active_epoch}" \
      "\${active_command_sha}" "\${active_snapshot}" "\${active_log_dir}" \
      "\${active_target}" || exit
    if [[ "\${legacy_intent_cgroup_fingerprint}" != "\${EXPECTED_CGROUP_FINGERPRINT}" \
          || "\${legacy_intent_root_pid}" != "\${legacy_receipt_root_pid}" \
          || "\${legacy_intent_root_start}" != "\${legacy_receipt_root_start}" ]]; then
      echo "[ERROR][${node}] Refusing to retire a freeze intent for a different receipt boundary." >&2
      exit 2
    fi
    rm -f -- "\${capture_intent}" || exit
  fi
  if [[ -e "\${receipt_incoming}" || -L "\${receipt_incoming}" \
        || -e "\${intent_incoming}" || -L "\${intent_incoming}" ]]; then
    echo "[ERROR][${node}] Legacy publication residue survived canonical capture." >&2
    exit 2
  fi
  if [[ "\${active_phase}" == stopped ]]; then
    (( tmux_running == 0 )) || {
      echo "[ERROR][${node}] Legacy stopped metadata still has a live tmux session." >&2
      exit 2
    }
    legacy_verify_terminal_receipt_closure || exit
    legacy_stop_committed=1
    trap - EXIT
    echo "[INFO][${node}] ${SESSION} already has verified legacy stopped closure"
    exit 0
  fi
  # Multi-node legacy stop is explicitly two-phase.  An arm call leaves this
  # exact active identity running-but-frozen only after its canonical receipt
  # is published.  The controller begins irreversible cgroup.kill on no node
  # until every node has returned this proof.  A lost controller reply merely
  # leaves a durable frozen receipt which a later stop can re-arm idempotently.
  if [[ "\${STOP_MODE}" == arm ]]; then
    legacy_validate_frozen_receipt_cgroup "\${CONTROL_SCRIPT}" || exit
    legacy_stop_committed=1
    trap - EXIT
    echo "[INFO][${node}] armed ${SESSION} with exact frozen legacy cgroup receipt"
    exit 0
  fi
  if (( legacy_exact_pane_loaded == 1 )); then
    if [[ "\${legacy_receipt_root_pid}" != "\${legacy_pane_pid}" \
          || "\${legacy_receipt_root_start}" != "\${legacy_pane_start}" ]]; then
      echo "[ERROR][${node}] Legacy capture receipt root does not match the exact current tmux pane." >&2
      exit 2
    fi
  fi
  legacy_validate_frozen_receipt_cgroup "\${CONTROL_SCRIPT}" || exit
  incoming="\${ACTIVE_STATE}.incoming.stopping.\$\$"
  printf '2\tstopping\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "\${active_snapshot}" "\${active_log_dir}" "\${active_target}" "\${active_token}" \
    "\${active_command_sha}" "\${active_epoch}" > "\${incoming}"
  mv -T "\${incoming}" "\${ACTIVE_STATE}"
  validate_private_state_file_metadata "\${ACTIVE_STATE}" 4096 0
  legacy_stop_committed=1
  # Destroy the frozen receipt closure before asking tmux to tear down its
  # pane.  Otherwise HUP/TERM could run r21's EXIT trap and release rendezvous
  # before all nodes have proved process closure.
  legacy_terminate_receipt_bounded "\${CONTROL_SCRIPT}" || exit
  if tmux has-session -t $(quote "${SESSION}") 2>/dev/null; then
    tmux_kill_rc=0
    tmux kill-session -t $(quote "${SESSION}") || tmux_kill_rc=\$?
    if (( tmux_kill_rc != 0 )) \
        && tmux has-session -t $(quote "${SESSION}") 2>/dev/null; then
      echo "[ERROR][${node}] Exact legacy processes are closed, but tmux kill-session failed (rc=\${tmux_kill_rc})." >&2
      exit 2
    fi
  else
    tmux_rc=\$?
    [[ "\${tmux_rc}" == 1 ]] || exit 2
  fi
  if tmux has-session -t $(quote "${SESSION}") 2>/dev/null; then
    echo "[ERROR][${node}] Same-name tmux survived legacy stop cleanup." >&2
    exit 2
  else
    tmux_rc=\$?
    [[ "\${tmux_rc}" == 1 ]] || exit 2
  fi
  legacy_wait_receipt_closed_bounded || exit
  legacy_verify_terminal_receipt_closure || exit
  trap - EXIT
fi
incoming="\${ACTIVE_STATE}.incoming.stopped.\$\$"
printf '2\tstopped\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "\${active_snapshot}" "\${active_log_dir}" "\${active_target}" "\${active_token}" \
  "\${active_command_sha}" "\${active_epoch}" > "\${incoming}"
mv -T "\${incoming}" "\${ACTIVE_STATE}"
validate_private_state_file_metadata "\${ACTIVE_STATE}" 4096 0
echo "[INFO][${node}] stopped ${SESSION} with exact \${STOP_CONTRACT} process closure"
EOF
)
  remote_run_mutation_bounded "${node}" "${cmd}" "${LAUNCH_CLEANUP_TIMEOUT_SECONDS}"
}

stop_launched_node() {
  local node="$1"
  local launch_token="$2"
  local expected_command_sha="$3"
  local expected_launch_epoch="$4"
  local tmux_lock_path
  tmux_lock_path="${REMOTE_RUN_ROOT}/.active/.locks/holosoma-tmux-$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}').lock"
  local active_state_path
  active_state_path="${REMOTE_RUN_ROOT}/.active/$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}')_${node}.state"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
$(private_lifecycle_file_validation_helpers)
open_private_lifecycle_lock $(quote "${tmux_lock_path}") 8
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 8; then
  echo "[ERROR][${node}] Timed out acquiring the ${SESSION} lifecycle rollback lock; owned session cleanup is unconfirmed." >&2
  exit 1
fi
ACTIVE_STATE=$(quote "${active_state_path}")
$(active_state_validation_helpers)
$(launch_process_closure_helpers)
if ! load_active_state_v2_exact "\${ACTIVE_STATE}" \
    || ! active_state_has_session_namespace $(quote "${SESSION}") \
    || [[ "\${active_snapshot}" != $(quote "${SOURCE_SNAPSHOT_ID}") \
      || "\${active_log_dir}" != $(quote "${LOG_DIR}") \
      || "\${active_target}" != $(quote "${TARGET_LEARNING_ITERATION}") \
      || "\${active_token}" != $(quote "${launch_token}") \
      || "\${active_epoch}" != $(quote "${expected_launch_epoch}") \
      || ( "\${active_command_sha}" != pending \
        && "\${active_command_sha}" != $(quote "${expected_command_sha}") ) ]]; then
  echo "[ERROR][${node}] Exact active ownership metadata is unavailable during launch rollback." >&2
  exit 2
fi
case "\${active_phase}" in
  launching|running|rolling_back|rolled_back) ;;
  *)
    echo "[ERROR][${node}] Refusing launch rollback from lifecycle phase=\${active_phase}." >&2
    exit 2
    ;;
esac
write_owned_phase() {
  local new_phase="\$1" incoming
  incoming="\${ACTIVE_STATE}.incoming.${launch_token}.\${new_phase}.\$\$"
  printf '2\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "\${new_phase}" "\${active_snapshot}" "\${active_log_dir}" "\${active_target}" \
    "\${active_token}" $(quote "${expected_command_sha}") "\${active_epoch}" > "\${incoming}"
  mv -T "\${incoming}" "\${ACTIVE_STATE}"
  validate_private_state_file_metadata "\${ACTIVE_STATE}" 4096 0
  active_command_sha=$(quote "${expected_command_sha}")
}
tmux_running=0
if tmux has-session -t $(quote "${SESSION}") 2>/dev/null; then
  tmux_running=1
  $(tmux_ownership_helpers)
  if ! tmux_session_is_owned_for_cleanup \
      $(quote "${SESSION}") $(quote "${launch_token}") \
      $(quote "${expected_command_sha}") $(quote "${expected_launch_epoch}"); then
    echo "[WARN][${node}] Refusing to stop ${SESSION} during launch rollback: session is not owned by this launch. Exact token/command/epoch mismatch across atomic environment/options." >&2
    exit 2
  fi
else
  tmux_rc=\$?
  [[ "\${tmux_rc}" == 1 ]] || {
    echo "[ERROR][${node}] Could not inspect tmux during launch rollback (rc=\${tmux_rc})." >&2
    exit 2
  }
fi
if [[ "\${active_phase}" == rolled_back && "\${tmux_running}" == 0 ]] \
    && verify_no_launch_identity_processes \
      $(quote "${launch_token}") $(quote "${expected_command_sha}") \
      $(quote "${expected_launch_epoch}"); then
  echo "[INFO][${node}] ${SESSION} already has verified rollback closure"
  exit 0
fi
write_owned_phase rolling_back
if (( tmux_running == 1 )); then
  tmux kill-session -t $(quote "${SESSION}")
fi
terminate_launch_identity_processes_bounded \
  $(quote "${launch_token}") $(quote "${expected_command_sha}") \
  $(quote "${expected_launch_epoch}")
if tmux has-session -t $(quote "${SESSION}") 2>/dev/null; then
  echo "[ERROR][${node}] Same-name tmux survived launch rollback cleanup." >&2
  exit 2
else
  tmux_rc=\$?
  [[ "\${tmux_rc}" == 1 ]] || {
    echo "[ERROR][${node}] Could not verify tmux absence after launch rollback (rc=\${tmux_rc})." >&2
    exit 2
  }
fi
verify_no_launch_identity_processes \
  $(quote "${launch_token}") $(quote "${expected_command_sha}") \
  $(quote "${expected_launch_epoch}")
write_owned_phase rolled_back
echo "[INFO][${node}] rolled back owned session ${SESSION}"
EOF
)
  # Rollback mutates durable lifecycle metadata and may kill an owned process
  # tree. Bound the remote transaction itself as well as the local SSH client,
  # so a disconnected controller cannot leave a late writer holding the
  # lifecycle lock or racing an idempotent retry.
  remote_run_mutation_bounded "${node}" "${cmd}" "${LAUNCH_CLEANUP_TIMEOUT_SECONDS}"
}

mark_launch_states_parallel() {
  local launch_token="$1"
  local launch_epoch="$2"
  local phase="$3"
  shift 3
  local -a nodes=("$@") pids=()
  local node pid index failed=0
  for node in "${nodes[@]}"; do
    mark_launch_state_node "${node}" "${launch_token}" "${launch_epoch}" "${phase}" &
    pids+=("$!")
  done
  for index in "${!nodes[@]}"; do
    pid=${pids[${index}]}
    if ! wait "${pid}"; then
      echo "[ERROR][cleanup][${nodes[${index}]}] Failed to confirm bounded lifecycle transition to ${phase}." >&2
      failed=1
    fi
  done
  return "${failed}"
}

cancel_launch_intents_parallel() {
  local launch_token="$1"
  local launch_epoch="$2"
  shift 2
  local -a nodes=("$@") pids=()
  local node pid index failed=0
  for node in "${nodes[@]}"; do
    cancel_launch_intent_node "${node}" "${launch_token}" "${launch_epoch}" &
    pids+=("$!")
  done
  for index in "${!nodes[@]}"; do
    pid=${pids[${index}]}
    if ! wait "${pid}"; then
      echo "[ERROR][cleanup][${nodes[${index}]}] Durable launch-intent cancellation was not confirmed." >&2
      failed=1
    fi
  done
  return "${failed}"
}

verify_cancelled_intent_closure_node() {
  local node="$1"
  local launch_token="$2"
  local launch_epoch="$3"
  local expected_snapshot="$4"
  local expected_log_dir="$5"
  local expected_target="$6"
  local session_sha active_state_path cancellation_state_path tmux_lock_path
  session_sha=$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}')
  active_state_path="${REMOTE_RUN_ROOT}/.active/${session_sha}_${node}.state"
  cancellation_state_path="${REMOTE_RUN_ROOT}/.active/cancelled.${session_sha}.${node}.${launch_token}.state"
  tmux_lock_path="${REMOTE_RUN_ROOT}/.active/.locks/holosoma-tmux-${session_sha}.lock"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
ACTIVE_STATE=$(quote "${active_state_path}")
CANCELLATION_STATE=$(quote "${cancellation_state_path}")
$(private_lifecycle_file_validation_helpers)
open_private_lifecycle_lock $(quote "${tmux_lock_path}") 8
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 8; then
  echo "[ERROR][${node}] Timed out acquiring cancelled-intent closure lock." >&2
  exit 1
fi
if ! validate_private_state_file_metadata "\${CANCELLATION_STATE}" 4096 0 \
    || [[ ! -f "\${CANCELLATION_STATE}" || -L "\${CANCELLATION_STATE}" \
      || "\$(awk 'END { print NR }' "\${CANCELLATION_STATE}")" != 1 \
      || "\$(awk -F '\t' 'NR == 1 { print NF }' "\${CANCELLATION_STATE}")" != 8 ]]; then
  echo "[ERROR][${node}] Cancelled-intent closure lacks an exact regular tombstone." >&2
  exit 2
fi
cancel_version="" cancel_token="" cancel_epoch="" cancel_snapshot=""
cancel_log_dir="" cancel_target="" cancel_session="" cancel_node=""
IFS=\$'\t' read -r cancel_version cancel_token cancel_epoch cancel_snapshot \
  cancel_log_dir cancel_target cancel_session cancel_node < "\${CANCELLATION_STATE}" || true
if [[ "\${cancel_version}" != 1 \
      || "\${cancel_token}" != $(quote "${launch_token}") \
      || "\${cancel_epoch}" != $(quote "${launch_epoch}") \
      || "\${cancel_snapshot}" != $(quote "${expected_snapshot}") \
      || "\${cancel_log_dir}" != $(quote "${expected_log_dir}") \
      || "\${cancel_target}" != $(quote "${expected_target}") \
      || "\${cancel_session}" != $(quote "${SESSION}") \
      || "\${cancel_node}" != $(quote "${node}") ]]; then
  echo "[ERROR][${node}] Cancelled-intent tombstone identity mismatch." >&2
  exit 2
fi
$(active_state_validation_helpers)
$(tmux_session_query_helpers)
$(tmux_ownership_helpers)
$(launch_process_closure_helpers)
active_disposition=absent
if [[ -e "\${ACTIVE_STATE}" || -L "\${ACTIVE_STATE}" ]]; then
  if ! load_active_state_v2_exact "\${ACTIVE_STATE}" \
      || ! active_state_has_session_namespace $(quote "${SESSION}"); then
    echo "[ERROR][${node}] Cancelled-intent closure found malformed active metadata." >&2
    exit 2
  fi
  if [[ "\${active_token}" == $(quote "${launch_token}") ]]; then
    if [[ "\${active_phase}" != rolled_back \
          || "\${active_snapshot}" != $(quote "${expected_snapshot}") \
          || "\${active_log_dir}" != $(quote "${expected_log_dir}") \
          || "\${active_target}" != $(quote "${expected_target}") \
          || "\${active_command_sha}" != pending \
          || "\${active_epoch}" != $(quote "${launch_epoch}") ]]; then
      echo "[ERROR][${node}] Same-token metadata does not prove cancelled-intent terminal closure." >&2
      exit 2
    fi
    active_disposition=old_terminal
  else
    active_disposition=other
  fi
fi
query_tmux_session_presence $(quote "${SESSION}") || exit
if (( tmux_session_present == 1 )); then
  if [[ "\${active_disposition}" != other ]]; then
    echo "[ERROR][${node}] Cancelled-intent closure found a same-name tmux which may belong to the cancelled token." >&2
    exit 2
  fi
  case "\${active_phase}" in
    running|rolling_back|stopping)
      if [[ ! "\${active_command_sha}" =~ ^[0-9a-f]{64}$ ]] \
          || ! tmux_session_has_complete_new_identity \
            $(quote "${SESSION}") "\${active_token}" \
            "\${active_command_sha}" "\${active_epoch}"; then
        echo "[ERROR][${node}] Same-name tmux does not prove the preserved different-token active identity." >&2
        exit 2
      fi
      ;;
    *)
      echo "[ERROR][${node}] Preserved different-token phase=\${active_phase} cannot own a live same-name tmux." >&2
      exit 2
      ;;
  esac
fi
verify_no_launch_token_epoch_processes \
  $(quote "${launch_token}") $(quote "${launch_epoch}")
echo "[INFO][${node}] verified cancelled-intent closure active_disposition=\${active_disposition}"
EOF
)
  remote_run_bounded "${node}" "${cmd}" "${LAUNCH_CLEANUP_TIMEOUT_SECONDS}"
}

verify_cancelled_intent_closures_parallel() {
  local launch_token="$1" launch_epoch="$2" expected_snapshot="$3"
  local expected_log_dir="$4" expected_target="$5"
  shift 5
  local -a nodes=("$@") pids=()
  local node pid index failed=0
  for node in "${nodes[@]}"; do
    verify_cancelled_intent_closure_node \
      "${node}" "${launch_token}" "${launch_epoch}" \
      "${expected_snapshot}" "${expected_log_dir}" "${expected_target}" &
    pids+=("$!")
  done
  for index in "${!nodes[@]}"; do
    pid=${pids[${index}]}
    if ! wait "${pid}"; then
      echo "[ERROR][cleanup][${nodes[${index}]}] Cancelled-intent closure is unconfirmed." >&2
      failed=1
    fi
  done
  return "${failed}"
}

rollback_owned_nodes_parallel() {
  local launch_token="$1"
  local launch_epoch="$2"
  shift 2
  # Remaining arguments are safe-node=sha256 pairs.  Both components are
  # validated before launch and cannot contain '='.
  local -a owned_specs=("$@") pids=()
  local spec node command_sha pid index failed=0
  for spec in "${owned_specs[@]}"; do
    node=${spec%%=*}
    command_sha=${spec#*=}
    stop_launched_node "${node}" "${launch_token}" "${command_sha}" "${launch_epoch}" &
    pids+=("$!")
  done
  for index in "${!owned_specs[@]}"; do
    spec=${owned_specs[${index}]}
    node=${spec%%=*}
    pid=${pids[${index}]}
    if ! wait "${pid}"; then
      echo "[ERROR][cleanup][${node}] Exact owned tmux cleanup was not confirmed within the bound; manual retry is required." >&2
      failed=1
    fi
  done
  return "${failed}"
}

verify_cleanup_closure_node() {
  local node="$1"
  local launch_token="$2"
  local launch_epoch="$3"
  local expected_command_sha="$4"
  local expected_phase="$5"
  local expected_snapshot="$6"
  local expected_log_dir="$7"
  local expected_target="$8"
  local tmux_lock_path
  tmux_lock_path="${REMOTE_RUN_ROOT}/.active/.locks/holosoma-tmux-$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}').lock"
  local active_state_path
  active_state_path="${REMOTE_RUN_ROOT}/.active/$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}')_${node}.state"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
$(private_lifecycle_file_validation_helpers)
open_private_lifecycle_lock $(quote "${tmux_lock_path}") 8
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 8; then
  echo "[ERROR][${node}] Timed out acquiring final ${SESSION} cleanup-closure lock." >&2
  exit 1
fi
ACTIVE_STATE=$(quote "${active_state_path}")
$(active_state_validation_helpers)
$(launch_process_closure_helpers)
if ! load_active_state_v2_exact "\${ACTIVE_STATE}" \
    || ! active_state_has_session_namespace $(quote "${SESSION}") \
    || [[ "\${active_phase}" != $(quote "${expected_phase}") \
      || "\${active_snapshot}" != $(quote "${expected_snapshot}") \
      || "\${active_log_dir}" != $(quote "${expected_log_dir}") \
      || "\${active_target}" != $(quote "${expected_target}") \
      || "\${active_token}" != $(quote "${launch_token}") \
      || "\${active_command_sha}" != $(quote "${expected_command_sha}") \
      || "\${active_epoch}" != $(quote "${launch_epoch}") ]]; then
  echo "[ERROR][${node}] Final cleanup metadata does not prove exact $(quote "${expected_phase}") closure." >&2
  exit 2
fi
if tmux has-session -t $(quote "${SESSION}") 2>/dev/null; then
  echo "[ERROR][${node}] Final cleanup closure failed: same-name tmux still exists." >&2
  exit 2
else
  tmux_rc=\$?
  [[ "\${tmux_rc}" == 1 ]] || {
    echo "[ERROR][${node}] Final cleanup closure could not verify tmux absence (rc=\${tmux_rc})." >&2
    exit 2
  }
fi
verify_no_launch_identity_processes \
  $(quote "${launch_token}") $(quote "${expected_command_sha}") \
  $(quote "${launch_epoch}")
echo "[INFO][${node}] verified exact $(quote "${expected_phase}") process/session closure"
EOF
)
  remote_run_bounded "${node}" "${cmd}" "${LAUNCH_CLEANUP_TIMEOUT_SECONDS}"
}

verify_cleanup_closures_parallel() {
  local launch_token="$1"
  local launch_epoch="$2"
  local expected_phase="$3"
  local expected_snapshot="$4"
  local expected_log_dir="$5"
  local expected_target="$6"
  shift 6
  local -a closure_specs=("$@") pids=()
  local spec node command_sha pid index failed=0
  for spec in "${closure_specs[@]}"; do
    node=${spec%%=*}
    command_sha=${spec#*=}
    verify_cleanup_closure_node \
      "${node}" "${launch_token}" "${launch_epoch}" "${command_sha}" \
      "${expected_phase}" "${expected_snapshot}" "${expected_log_dir}" \
      "${expected_target}" &
    pids+=("$!")
  done
  for index in "${!closure_specs[@]}"; do
    spec=${closure_specs[${index}]}
    node=${spec%%=*}
    pid=${pids[${index}]}
    if ! wait "${pid}"; then
      echo "[ERROR][cleanup][${node}] Final exact ${expected_phase} closure was not proven." >&2
      failed=1
    fi
  done
  return "${failed}"
}

verify_legacy_stop_cleanup_closure_node() {
  local node="$1"
  local launch_token="$2"
  local launch_epoch="$3"
  local expected_command_sha="$4"
  local expected_phase="$5"
  local expected_snapshot="$6"
  local expected_log_dir="$7"
  local expected_target="$8"
  local expected_cgroup_fingerprint="$9"
  local tmux_lock_path
  tmux_lock_path="${REMOTE_RUN_ROOT}/.active/.locks/holosoma-tmux-$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}').lock"
  local active_state_path
  active_state_path="${REMOTE_RUN_ROOT}/.active/$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}')_${node}.state"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
$(private_lifecycle_file_validation_helpers)
open_private_lifecycle_lock $(quote "${tmux_lock_path}") 8
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 8; then
  echo "[ERROR][${node}] Timed out acquiring final legacy ${SESSION} cleanup-closure lock." >&2
  exit 1
fi
ACTIVE_STATE=$(quote "${active_state_path}")
$(active_state_validation_helpers)
$(legacy_stop_process_helpers)
if ! load_active_state_v2_exact "\${ACTIVE_STATE}" \
    || ! active_state_has_session_namespace $(quote "${SESSION}") \
    || [[ "\${active_phase}" != $(quote "${expected_phase}") \
      || "\${active_snapshot}" != $(quote "${expected_snapshot}") \
      || "\${active_log_dir}" != $(quote "${expected_log_dir}") \
      || "\${active_target}" != $(quote "${expected_target}") \
      || "\${active_token}" != $(quote "${launch_token}") \
      || "\${active_command_sha}" != $(quote "${expected_command_sha}") \
      || "\${active_epoch}" != $(quote "${launch_epoch}") ]]; then
  echo "[ERROR][${node}] Final legacy cleanup metadata does not prove exact stopped closure." >&2
  exit 2
fi
if tmux has-session -t $(quote "${SESSION}") 2>/dev/null; then
  echo "[ERROR][${node}] Final legacy cleanup closure failed: same-name tmux still exists." >&2
  exit 2
else
  tmux_rc=\$?
  [[ "\${tmux_rc}" == 1 ]] || exit 2
fi
capture_receipt=\$(legacy_capture_receipt_path \
  "\${ACTIVE_STATE}" "\${active_token}" "\${active_command_sha}" "\${active_epoch}")
for candidate in "\${ACTIVE_STATE}".legacy-processes.*; do
  [[ -e "\${candidate}" || -L "\${candidate}" ]] || continue
  if [[ "\${candidate}" != "\${capture_receipt}" ]]; then
    echo "[ERROR][${node}] Final legacy cleanup retains an intent, incoming file, or unknown receipt." >&2
    exit 2
  fi
done
legacy_load_capture_receipt \
  "\${capture_receipt}" "\${active_token}" "\${active_epoch}" \
  "\${active_command_sha}" "\${active_snapshot}" "\${active_log_dir}" \
  "\${active_target}"
if [[ "\${legacy_receipt_cgroup_fingerprint}" != $(quote "${expected_cgroup_fingerprint}") ]]; then
  echo "[ERROR][${node}] Final legacy receipt cgroup differs from the all-node stop preflight." >&2
  exit 2
fi
legacy_verify_terminal_receipt_closure
echo "[INFO][${node}] verified exact legacy stopped process/session closure"
EOF
)
  remote_run_bounded "${node}" "${cmd}" "${LAUNCH_CLEANUP_TIMEOUT_SECONDS}"
}

verify_stop_cleanup_closures_parallel() {
  local launch_token="$1"
  local launch_epoch="$2"
  local expected_phase="$3"
  local expected_snapshot="$4"
  local expected_log_dir="$5"
  local expected_target="$6"
  local expected_contract="$7"
  shift 7
  local -a closure_specs=("$@") pids=()
  local spec node identity command_sha cgroup_fingerprint pid index failed=0
  for spec in "${closure_specs[@]}"; do
    node=${spec%%=*}
    identity=${spec#*=}
    command_sha=${identity%%:*}
    cgroup_fingerprint=${identity#*:}
    if [[ "${expected_contract}" == legacy ]]; then
      verify_legacy_stop_cleanup_closure_node \
        "${node}" "${launch_token}" "${launch_epoch}" "${command_sha}" \
        "${expected_phase}" "${expected_snapshot}" "${expected_log_dir}" \
        "${expected_target}" "${cgroup_fingerprint}" &
    else
      verify_cleanup_closure_node \
        "${node}" "${launch_token}" "${launch_epoch}" "${command_sha}" \
        "${expected_phase}" "${expected_snapshot}" "${expected_log_dir}" \
        "${expected_target}" &
    fi
    pids+=("$!")
  done
  for index in "${!closure_specs[@]}"; do
    spec=${closure_specs[${index}]}
    node=${spec%%=*}
    pid=${pids[${index}]}
    if ! wait "${pid}"; then
      echo "[ERROR][cleanup][${node}] Final exact ${expected_contract} ${expected_phase} closure was not proven." >&2
      failed=1
    fi
  done
  return "${failed}"
}

recover_launch_identity_node() {
  local node="$1"
  local launch_token="$2"
  local launch_epoch="$3"
  local active_state_path
  active_state_path="${REMOTE_RUN_ROOT}/.active/$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}')_${node}.state"
  local tmux_lock_path
  tmux_lock_path="${REMOTE_RUN_ROOT}/.active/.locks/holosoma-tmux-$(printf '%s' "${SESSION}" | sha256sum | awk '{print $1}').lock"
  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
$(private_lifecycle_file_validation_helpers)
open_private_lifecycle_lock $(quote "${tmux_lock_path}") 8
if ! flock -w $(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}") -x 8; then
  echo "[ERROR][${node}] Timed out acquiring the ${SESSION} lifecycle identity-recovery lock." >&2
  exit 1
fi
ACTIVE_STATE=$(quote "${active_state_path}")
$(active_state_validation_helpers)
if ! load_active_state_v2_exact "\${ACTIVE_STATE}" \
    || ! active_state_has_session_namespace $(quote "${SESSION}") \
    || [[ "\${active_snapshot}" != $(quote "${SOURCE_SNAPSHOT_ID}") \
      || "\${active_log_dir}" != $(quote "${LOG_DIR}") \
      || "\${active_target}" != $(quote "${TARGET_LEARNING_ITERATION}") \
      || "\${active_token}" != $(quote "${launch_token}") \
      || "\${active_epoch}" != $(quote "${launch_epoch}") ]]; then
  echo "[ERROR][${node}] Active metadata cannot recover this launch's exact identity." >&2
  exit 2
fi
version=\${active_version} phase=\${active_phase} snapshot=\${active_snapshot}
log_dir=\${active_log_dir} target=\${active_target} token=\${active_token}
command_sha=\${active_command_sha} epoch=\${active_epoch}
if [[ ( "\${phase}" == running || "\${phase}" == rolling_back ) \
      && "\${command_sha}" =~ ^[0-9a-f]{64}$ ]]; then
  printf 'owned\t%s\n' "\${command_sha}"
  exit 0
fi
if [[ "\${phase}" != launching || "\${command_sha}" != pending ]]; then
  echo "[ERROR][${node}] Active metadata has no recoverable command identity or exact intent-only state." >&2
  exit 2
fi
$(tmux_session_query_helpers)
query_tmux_session_presence $(quote "${SESSION}") || exit
if (( tmux_session_present == 0 )); then
  # Cancel the intent while still holding the same lifecycle lock used by
  # tmux creation.  Returning a read-only "no session" observation would be a
  # TOCTOU: a delayed launch shell could create tmux after this SSH returned
  # but before controller rollback marked the state.
  incoming="\${ACTIVE_STATE}.incoming.${launch_token}.identity-recovery.\$\$"
  printf '2\trolling_back\t%s\t%s\t%s\t%s\tpending\t%s\n' \
    "\${snapshot}" "\${log_dir}" "\${target}" "\${token}" "\${epoch}" > "\${incoming}"
  mv -T "\${incoming}" "\${ACTIVE_STATE}"
  validate_private_state_file_metadata "\${ACTIVE_STATE}" 4096 0
  echo 'intent-only'
  exit 0
fi
# tmux may have committed atomically before SSH reported failure while active
# metadata still contains only the intent. Recover the command digest from the
# complete atomic environment, never from mutable/partial options alone.
$(tmux_ownership_helpers)
load_tmux_ownership_identity $(quote "${SESSION}") || {
  echo "[ERROR][${node}] Cannot read complete tmux identity while recovering an ambiguous launch." >&2
  exit 2
}
if [[ "\${tmux_env_token_present}" != 1 || "\${tmux_env_token_valid}" != 1 \
      || "\${tmux_env_token_value}" != $(quote "${launch_token}") \
      || "\${tmux_env_command_sha_present}" != 1 || "\${tmux_env_command_sha_valid}" != 1 \
      || ! "\${tmux_env_command_sha_value}" =~ ^[0-9a-f]{64}$ \
      || "\${tmux_env_epoch_present}" != 1 || "\${tmux_env_epoch_valid}" != 1 \
      || "\${tmux_env_epoch_value}" != $(quote "${launch_epoch}") ]] \
    || ! tmux_options_do_not_conflict \
      $(quote "${launch_token}") "\${tmux_env_command_sha_value}" $(quote "${launch_epoch}"); then
  echo "[ERROR][${node}] Ambiguous tmux session has no exact recoverable atomic launch identity." >&2
  exit 2
fi
printf 'owned\t%s\n' "\${tmux_env_command_sha_value}"
EOF
)
  remote_run_bounded "${node}" "${cmd}" "${LAUNCH_CLEANUP_TIMEOUT_SECONDS}"
}

verify_python_runtime_before_intent_node() {
  local node="$1"
  if [[ -z "${PYTHON_RUNTIME_SITEPACKAGES}" ]]; then
    return 0
  fi
  local runtime_pythonpath="${PYTHON_RUNTIME_SITEPACKAGES}:${RUN_REPO}/src/holosoma:${RUN_REPO}/src/holosoma_inference:${RUN_REPO}/src"
  local runtime_path
  runtime_path="$(dirname -- "${PYTHON_BIN}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  local body cmd
  body=$(cat <<'REMOTE'
set -euo pipefail
unset BASH_ENV ENV CDPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONOPTIMIZE PYTHONWARNINGS PYTHONBREAKPOINT PYTHONSAFEPATH
unset LD_PRELOAD
export PATH="$RUNTIME_PATH"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LC_ALL=C
cd "$RUN_REPO_REMOTE"
grep -Fx -- "$SNAPSHOT_ID" .holosoma_snapshot/id >/dev/null
sha256sum --quiet -c .holosoma_snapshot/source_manifest.sha256
"$PYTHON_BIN_REMOTE" -I -S scripts/install_python_runtime_overlay.py --runtime-root "$RUNTIME_ROOT" --manifest-sha256 "$MANIFEST_SHA256" --verifier scripts/verify_python_runtime_overlay.py --lock-timeout-seconds "$LOCK_TIMEOUT" --probe-only
export PYTHONPATH="$RUNTIME_PYTHONPATH"
"$PYTHON_BIN_REMOTE" scripts/verify_python_runtime_overlay.py --site-packages "$SITE_PACKAGES" --manifest-sha256 "$MANIFEST_SHA256" --require-distribution-closure --require-current-runtime-binding
echo "[INFO][$NODE_LABEL] python_runtime_pre_intent_verified=$SITE_PACKAGES"
REMOTE
)
  cmd="RUNTIME_PATH=$(quote "${runtime_path}")"$'\n'
  cmd+="RUN_REPO_REMOTE=$(quote "${RUN_REPO}")"$'\n'
  cmd+="SNAPSHOT_ID=$(quote "${SOURCE_SNAPSHOT_ID}")"$'\n'
  cmd+="PYTHON_BIN_REMOTE=$(quote "${PYTHON_BIN}")"$'\n'
  cmd+="RUNTIME_ROOT=$(quote "${REMOTE_RUN_ROOT}/.runtime/python")"$'\n'
  cmd+="MANIFEST_SHA256=$(quote "${PYTHON_RUNTIME_MANIFEST_SHA256}")"$'\n'
  cmd+="LOCK_TIMEOUT=$(quote "${LAUNCH_LOCK_TIMEOUT_SECONDS}")"$'\n'
  cmd+="RUNTIME_PYTHONPATH=$(quote "${runtime_pythonpath}")"$'\n'
  cmd+="SITE_PACKAGES=$(quote "${PYTHON_RUNTIME_SITEPACKAGES}")"$'\n'
  cmd+="NODE_LABEL=$(quote "${node}")"$'\n'
  cmd+="${body}"
  remote_run_mutation_bounded \
    "${node}" "${cmd}" "${LAUNCH_PREFLIGHT_TIMEOUT_SECONDS}"
}

verify_python_runtimes_before_intent_parallel() {
  if [[ -z "${PYTHON_RUNTIME_SITEPACKAGES}" ]]; then
    return 0
  fi
  local -a pids=()
  local node pid failed=0
  for node in "${NODE_LIST[@]}"; do
    verify_python_runtime_before_intent_node "${node}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed != 0 )); then
    echo "[ERROR] Refusing launch because one or more nodes failed the pre-intent Python runtime barrier." >&2
    return 1
  fi
  if [[ "${DRY_RUN}" == 1 ]]; then
    echo "[INFO] Dry-run would enforce the all-node Python runtime barrier before launch intent publication."
  else
    echo "[INFO] All nodes passed the Python runtime barrier before launch intent publication."
  fi
}

run_prepare() {
  ensure_training_python
  # `all` and explicit `prepare` must not install/hash/copy source, runtime, or
  # data on a busy/protected training node.  This mandatory read-only gate is
  # repeated by run_launch after preparation to cover that preparation window.
  preflight_selected_gpus_idle_parallel
  # Freeze one local snapshot before forking node preparations. Building inside
  # each background child could capture different source if a file changed
  # while the nodes were being prepared.
  ensure_local_source_snapshot
  # Bind the controller archive once before per-node forks. Every node receives
  # the same authenticated bytes, while publication remains node-local.
  ensure_local_python_runtime_archive
  # Hash a control-visible custom tar exactly once before the per-node forks.
  # Nodes that can read NFS continue using it directly; only missing nodes use
  # this immutable content-addressed fallback.
  ensure_local_corl_package_metadata
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
  # Reject unsupported/mutually-exclusive reduction paths on the controller,
  # before source installation, remote intent publication, or GPU startup.
  validate_gradient_reduce_contracts
  validate_restart_contract
  if [[ "${DISTILL_AS_FORMAL_FRESH}" == 1 \
        && "${DRY_RUN}" != 1 \
        && -z "${FRESH_WANDB_RUN_ID}" ]]; then
    echo "[ERROR] A non-dry-run DISTILL_AS_FORMAL_FRESH=1 launch requires a fresh W&B identity and a Rule-90 v2 replay manifest." >&2
    exit 2
  fi
  if ! command -v timeout >/dev/null 2>&1; then
    echo "[ERROR] GNU timeout is required before any transactional launch mutation." >&2
    exit 2
  fi
  ensure_training_python
  ensure_local_source_snapshot
  # Resolve the minibatch geometry from the exact source snapshot before any
  # remote runtime maintenance, lifecycle mutation, rendezvous reservation, or
  # GPU process can begin.
  resolve_minibatch_throughput_contract
  # Parse only the already SHA-bound local Rule-90 inputs here.  No live W&B
  # request is made until every node has proved the exact external AS bytes
  # that the sealed wrappers will materialize and consume.
  load_replay_external_as_contract
  # A mistakenly selected busy node must be rejected using only read-only
  # inventory queries before runtime/data hashing can contend with a protected
  # training job.  launch_node retains its later idle check to close the race
  # between this early fleet gate and actual worker startup.
  preflight_selected_gpus_idle_parallel
  # The external-AS barrier uses the selected scientific interpreter.  Prove
  # its installed byte/distribution closure before asking it to inspect data.
  verify_python_runtimes_before_intent_parallel
  preflight_external_as_asset_closures_parallel
  # A fresh formal run is first created only to hold its exact kinematic
  # vis/replay summary.  Re-verify the immutable local manifest and the live
  # W&B media record only after the all-node data closure, and still before
  # touching any remote lifecycle state or GPU.
  verify_fresh_wandb_replay_preflight
  if ! harden_lifecycle_namespaces_parallel; then
    echo "[ERROR] Refusing launch because one or more node lifecycle namespaces are unsafe." >&2
    exit 1
  fi
  echo "[INFO] Launching ${NNODES} nodes x ${NPROC} GPUs, PER_GPU_ENVS=${PER_GPU_ENVS}, TOTAL_NUM_ENVS=${TOTAL_NUM_ENVS}"
  local rank=0
  local launch_token
  if [[ ! -c /dev/urandom || -L /dev/urandom ]] \
      || ! command -v od >/dev/null 2>&1 \
      || ! command -v tr >/dev/null 2>&1; then
    echo "[ERROR] A real /dev/urandom plus od/tr is required for launch ownership tokens." >&2
    exit 2
  fi
  if ! launch_token=$(LC_ALL=C od -An -v -N32 -tx1 /dev/urandom | tr -d '[:space:]'); then
    echo "[ERROR] Failed to read a 256-bit launch ownership token from /dev/urandom." >&2
    exit 2
  fi
  if [[ ! "${launch_token}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[ERROR] Kernel CSPRNG returned a malformed launch ownership token." >&2
    exit 2
  fi
  local launch_epoch
  launch_epoch=$(controller_epoch_now) || exit
  if [[ ! "${launch_epoch}" =~ ^[1-9][0-9]*$ || ${#launch_epoch} -gt 18 ]]; then
    echo "[ERROR] Controller clock returned a non-canonical or unsafe launch epoch: ${launch_epoch}" >&2
    exit 2
  fi
  local cleanup_failed=0
  local preflight_node
  local -A launch_active_predecessors=()
  for preflight_node in "${NODE_LIST[@]}"; do
    echo "[INFO] Preflighting launch intent on ${preflight_node}"
    if ! preflight_launch_intent_node "${preflight_node}"; then
      echo "[ERROR] All-node launch-intent preflight failed before active metadata was changed." >&2
      exit 1
    fi
    launch_active_predecessors["${preflight_node}"]="${LAST_LAUNCH_ACTIVE_PREDECESSOR}"
  done
  local intent_nodes=()
  local intent_node
  local intent_predecessor
  for intent_node in "${NODE_LIST[@]}"; do
    echo "[INFO] Publishing launch intent on ${intent_node}"
    intent_predecessor=${launch_active_predecessors["${intent_node}"]}
    if ! publish_launch_intent_node \
        "${intent_node}" "${launch_token}" "${launch_epoch}" \
        "${intent_predecessor}"; then
      # SSH can lose its reply after the remote atomic mv committed this
      # node's intent.  A same-lock persistent cancellation tombstone makes a
      # delayed publisher unable to resurrect the ambiguous current node.
      echo "[ERROR] Failed to publish all-node launch intent; rolling back $((${#intent_nodes[@]} + 1)) possible published intent(s), including the ambiguous current node." >&2
      cleanup_failed=0
      local -a possible_intent_nodes=("${intent_nodes[@]}" "${intent_node}")
      if ! cancel_launch_intents_parallel \
          "${launch_token}" "${launch_epoch}" "${possible_intent_nodes[@]}"; then
        echo "[ERROR] One or more possible launch intents were not durably cancelled." >&2
        cleanup_failed=1
      fi
      if ! verify_cancelled_intent_closures_parallel \
          "${launch_token}" "${launch_epoch}" \
          "${SOURCE_SNAPSHOT_ID}" "${LOG_DIR}" "${TARGET_LEARNING_ITERATION}" \
          "${possible_intent_nodes[@]}"; then
        cleanup_failed=1
      fi
      if (( cleanup_failed == 0 )); then
        release_rendezvous_ports "${launch_token}" || \
          echo "[ERROR] Owned rendezvous release remains unconfirmed after bounded cleanup." >&2
      else
        echo "[ERROR] Preserving any owned rendezvous reservations because launch-intent closure was not proven." >&2
      fi
      exit 1
    fi
    intent_nodes+=("${intent_node}")
  done
  if ! reserve_rendezvous_ports "${launch_token}"; then
    echo "[ERROR] Failed to reserve both rendezvous ports; rolling back launch intents." >&2
    cleanup_failed=0
    if ! cancel_rendezvous_reservation "${launch_token}"; then
      echo "[ERROR] Failed to durably cancel the possibly delayed rendezvous reservation." >&2
      cleanup_failed=1
    fi
    local -a reserved_intent_specs=()
    for preflight_node in "${intent_nodes[@]}"; do
      reserved_intent_specs+=("${preflight_node}=pending")
    done
    if ! mark_launch_states_parallel \
        "${launch_token}" "${launch_epoch}" rolled_back "${intent_nodes[@]}"; then
      echo "[ERROR] One or more launch intents remain unconfirmed after bounded cleanup." >&2
      cleanup_failed=1
    fi
    if ! verify_cleanup_closures_parallel \
        "${launch_token}" "${launch_epoch}" rolled_back \
        "${SOURCE_SNAPSHOT_ID}" "${LOG_DIR}" "${TARGET_LEARNING_ITERATION}" \
        "${reserved_intent_specs[@]}"; then
      cleanup_failed=1
    fi
    if (( cleanup_failed == 0 )); then
      release_rendezvous_ports "${launch_token}" || \
        echo "[ERROR] Owned rendezvous release remains unconfirmed after bounded cleanup." >&2
    else
      echo "[ERROR] Preserving any owned rendezvous reservations because failed-reservation intent closure was not proven." >&2
    fi
    exit 1
  fi
  local launched_nodes=()
  local owned_launch_specs=()
  local -A owned_launch_node_set=()
  local -A owned_launch_command_sha=()
  local launch_result_file launch_rc launch_identity_unconfirmed recovered_identity
  for node in "${NODE_LIST[@]}"; do
    echo "[INFO] Launching ${node} node_rank=${rank}"
    LAST_LAUNCHED_COMMAND_SHA=""
    launch_rc=0
    launch_identity_unconfirmed=0
    launch_result_file=""
    if ! launch_result_file=$(mktemp "${TMPDIR:-/tmp}/holosoma-launch-result.XXXXXX"); then
      echo "[ERROR] Failed to create the controller-local launch result file for ${node}." >&2
      launch_rc=2
    else
      # A function invoked directly by an inverted if condition runs with Bash's
      # errexit disabled throughout its body.  Execute launch_node as a plain
      # command in an independent set -e subshell, then capture its status in
      # a parent shell which is temporarily non-errexit so every failed remote
      # preflight reaches the transactional rollback below.
      set +e
      (
        set -e
        launch_node \
          "${node}" "${rank}" "${launch_token}" "${launch_epoch}" \
          "${launch_result_file}"
      )
      launch_rc=$?
      set -e
      if [[ -f "${launch_result_file}" ]]; then
        LAST_LAUNCHED_COMMAND_SHA=$(<"${launch_result_file}")
      fi
      rm -f -- "${launch_result_file}" || true
    fi
    if [[ ! "${LAST_LAUNCHED_COMMAND_SHA}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "[ERROR][${node}] Launch result did not return its exact command SHA256; recovering the token-bound remote identity." >&2
      recovered_identity=""
      if recovered_identity=$(recover_launch_identity_node \
          "${node}" "${launch_token}" "${launch_epoch}"); then
        case "${recovered_identity}" in
          owned$'\t'*)
            LAST_LAUNCHED_COMMAND_SHA=${recovered_identity#*$'\t'}
            if [[ ! "${LAST_LAUNCHED_COMMAND_SHA}" =~ ^[0-9a-f]{64}$ ]]; then
              launch_identity_unconfirmed=1
            else
              echo "[INFO][${node}] Recovered exact command SHA256 from token-bound remote identity for rollback." >&2
            fi
            ;;
          intent-only)
            # This is safe only when launch_node itself failed.  A zero return
            # contradicts an intent-only remote state and remains unconfirmed.
            (( launch_rc != 0 )) || launch_identity_unconfirmed=1
            ;;
          *)
            launch_identity_unconfirmed=1
            ;;
        esac
      else
        launch_identity_unconfirmed=1
      fi
      if (( launch_identity_unconfirmed != 0 )); then
        # Without an exact command digest this node is not assumed intent-only
        # and must never be falsely marked rolled_back or have its ports made
        # reusable.
        owned_launch_node_set["${node}"]=1
      fi
      (( launch_rc != 0 )) || launch_rc=2
    fi
    if (( launch_rc != 0 )); then
      echo "[ERROR] Launch failed on ${node}; rolling back ${#launched_nodes[@]} previously started node(s)." >&2
      cleanup_failed=0
      if (( launch_identity_unconfirmed != 0 )); then
        cleanup_failed=1
      fi
      if ! mark_launch_states_parallel \
          "${launch_token}" "${launch_epoch}" rolling_back "${intent_nodes[@]}"; then
        echo "[ERROR] One or more rolling_back lifecycle transitions remain unconfirmed." >&2
        cleanup_failed=1
      fi
      # The ssh connection may fail after creating the session but before
      # reporting success. Include this node only after its immutable command
      # digest exists; rollback then requires the exact token/digest/epoch.
      if [[ "${LAST_LAUNCHED_COMMAND_SHA}" =~ ^[0-9a-f]{64}$ ]]; then
        owned_launch_specs+=("${node}=${LAST_LAUNCHED_COMMAND_SHA}")
        owned_launch_node_set["${node}"]=1
        owned_launch_command_sha["${node}"]="${LAST_LAUNCHED_COMMAND_SHA}"
      fi
      if ! rollback_owned_nodes_parallel \
          "${launch_token}" "${launch_epoch}" "${owned_launch_specs[@]}"; then
        echo "[ERROR] One or more exact owned sessions may remain after bounded rollback." >&2
        cleanup_failed=1
      fi
      local -a intent_only_nodes=()
      for intent_node in "${intent_nodes[@]}"; do
        if [[ -z "${owned_launch_node_set[${intent_node}]+x}" ]]; then
          intent_only_nodes+=("${intent_node}")
        fi
      done
      if ! mark_launch_states_parallel \
          "${launch_token}" "${launch_epoch}" rolled_back "${intent_only_nodes[@]}"; then
        echo "[ERROR] One or more intent-only rolled_back transitions remain unconfirmed." >&2
        cleanup_failed=1
      fi
      local -a rollback_closure_specs=()
      for intent_node in "${intent_nodes[@]}"; do
        if [[ -n "${owned_launch_command_sha[${intent_node}]+x}" ]]; then
          rollback_closure_specs+=("${intent_node}=${owned_launch_command_sha[${intent_node}]}")
        elif [[ -z "${owned_launch_node_set[${intent_node}]+x}" ]]; then
          rollback_closure_specs+=("${intent_node}=pending")
        else
          echo "[ERROR][cleanup][${intent_node}] No exact command identity is available for final cleanup closure." >&2
          cleanup_failed=1
        fi
      done
      if (( ${#rollback_closure_specs[@]} != ${#intent_nodes[@]} )) \
          || ! verify_cleanup_closures_parallel \
            "${launch_token}" "${launch_epoch}" rolled_back \
            "${SOURCE_SNAPSHOT_ID}" "${LOG_DIR}" "${TARGET_LEARNING_ITERATION}" \
            "${rollback_closure_specs[@]}"; then
        cleanup_failed=1
      fi
      if (( cleanup_failed == 0 )); then
        release_rendezvous_ports "${launch_token}" || \
          echo "[ERROR] Owned rendezvous release remains unconfirmed after bounded cleanup." >&2
      else
        echo "[ERROR] Preserving owned rendezvous reservations as quarantine because launch rollback was not fully confirmed." >&2
      fi
      exit 1
    fi
    launched_nodes+=("${node}")
    owned_launch_specs+=("${node}=${LAST_LAUNCHED_COMMAND_SHA}")
    owned_launch_node_set["${node}"]=1
    owned_launch_command_sha["${node}"]="${LAST_LAUNCHED_COMMAND_SHA}"
    rank=$((rank + 1))
  done
  if ! wait_for_launch_startup "${launch_token}" "${launch_epoch}"; then
    echo "[ERROR] Startup-health handshake failed; rolling back only ${#launched_nodes[@]} session(s) owned by this launch token." >&2
    cleanup_failed=0
    if ! mark_launch_states_parallel \
        "${launch_token}" "${launch_epoch}" rolling_back "${intent_nodes[@]}"; then
      echo "[ERROR] One or more rolling_back lifecycle transitions remain unconfirmed." >&2
      cleanup_failed=1
    fi
    if ! rollback_owned_nodes_parallel \
        "${launch_token}" "${launch_epoch}" "${owned_launch_specs[@]}"; then
      echo "[ERROR] One or more exact owned sessions may remain after bounded rollback." >&2
      cleanup_failed=1
    fi
    if ! verify_cleanup_closures_parallel \
        "${launch_token}" "${launch_epoch}" rolled_back \
        "${SOURCE_SNAPSHOT_ID}" "${LOG_DIR}" "${TARGET_LEARNING_ITERATION}" \
        "${owned_launch_specs[@]}"; then
      cleanup_failed=1
    fi
    if (( cleanup_failed == 0 )); then
      release_rendezvous_ports "${launch_token}" || \
        echo "[ERROR] Owned rendezvous release remains unconfirmed after bounded cleanup." >&2
    else
      echo "[ERROR] Preserving owned rendezvous reservations as quarantine because launch rollback was not fully confirmed." >&2
    fi
    exit 1
  fi
  echo "[INFO] Launch accepted only after the bounded all-node startup-health handshake."
}

run_status() {
  local failed=0
  for node in "${NODE_LIST[@]}"; do
    if ! status_node "${node}"; then
      failed=1
    fi
  done
  return "${failed}"
}

arm_legacy_stop_nodes_parallel() {
  local expected_token="$1"
  local expected_epoch="$2"
  local expected_snapshot="$3"
  local expected_log_dir="$4"
  local expected_target="$5"
  shift 5
  local -a stop_specs=("$@") pids=()
  local spec node identity command_sha cgroup_fingerprint pid index failed=0
  for spec in "${stop_specs[@]}"; do
    node=${spec%%=*}
    identity=${spec#*=}
    command_sha=${identity%%:*}
    cgroup_fingerprint=${identity#*:}
    stop_node \
      "${node}" "${expected_token}" "${expected_epoch}" \
      "${expected_snapshot}" "${expected_log_dir}" "${expected_target}" \
      "${command_sha}" legacy "${cgroup_fingerprint}" arm &
    pids+=("$!")
  done
  for index in "${!stop_specs[@]}"; do
    spec=${stop_specs[${index}]}
    node=${spec%%=*}
    pid=${pids[${index}]}
    if ! wait "${pid}"; then
      echo "[ERROR][stop][${node}] Legacy all-node arm/freeze did not complete." >&2
      failed=1
    fi
  done
  return "${failed}"
}

stop_nodes_parallel() {
  local expected_token="$1"
  local expected_epoch="$2"
  local expected_snapshot="$3"
  local expected_log_dir="$4"
  local expected_target="$5"
  local expected_contract="$6"
  shift 6
  local -a stop_specs=("$@") pids=()
  local spec node identity command_sha cgroup_fingerprint pid index failed=0
  for spec in "${stop_specs[@]}"; do
    node=${spec%%=*}
    identity=${spec#*=}
    command_sha=${identity%%:*}
    cgroup_fingerprint=${identity#*:}
    stop_node \
      "${node}" "${expected_token}" "${expected_epoch}" \
      "${expected_snapshot}" "${expected_log_dir}" "${expected_target}" \
      "${command_sha}" "${expected_contract}" "${cgroup_fingerprint}" commit &
    pids+=("$!")
  done
  for index in "${!stop_specs[@]}"; do
    spec=${stop_specs[${index}]}
    node=${spec%%=*}
    pid=${pids[${index}]}
    if ! wait "${pid}"; then
      echo "[ERROR][stop][${node}] Exact stop/process cleanup did not complete." >&2
      failed=1
    fi
  done
  return "${failed}"
}

run_stop() {
  if [[ "${DRY_RUN}" == 1 ]]; then
    echo "[DRY_RUN] stop would preflight one exact shared launch identity across all nodes, stop each per-node command identity, verify process/session closure, and release only that token's reservation pair."
    return 0
  fi
  if ! harden_lifecycle_namespaces_parallel; then
    echo "[ERROR] Refusing stop because one or more node lifecycle namespaces are unsafe." >&2
    return 1
  fi
  local expected_token="" expected_epoch="" expected_snapshot=""
  local expected_log_dir="" expected_target="" expected_contract=""
  local identity_output node node_index
  local -a identity_fields=() stop_specs=()
  for node_index in "${!NODE_LIST[@]}"; do
    node=${NODE_LIST[${node_index}]}
    identity_output=""
    if ! identity_output=$(read_stop_identity_node "${node}"); then
      echo "[ERROR][stop][${node}] Could not preflight exact active stop identity; no node was mutated." >&2
      return 1
    fi
    if [[ "${identity_output}" == *$'\n'* ]]; then
      echo "[ERROR][stop][${node}] Stop identity probe returned more than one record; no node was mutated." >&2
      return 1
    fi
    IFS=$'\t' read -r -a identity_fields <<<"${identity_output}"
    if (( ${#identity_fields[@]} != 10 )) \
        || [[ ! "${identity_fields[0]}" =~ ^[0-9a-f]{64}$ \
          || ! "${identity_fields[1]}" =~ ^[1-9][0-9]*$ \
          || ! "${identity_fields[2]}" =~ ^src-[0-9a-f]{64}$ \
          || -z "${identity_fields[3]}" \
          || ! "${identity_fields[4]}" =~ ^[1-9][0-9]*$ \
          || ! "${identity_fields[5]}" =~ ^[0-9a-f]{64}$ \
          || ! "${identity_fields[6]}" =~ ^[1-9][0-9]*$ \
          || ! "${identity_fields[7]}" =~ ^[0-9]+$ \
          || ! "${identity_fields[8]}" =~ ^(modern|legacy)$ \
          || ! "${identity_fields[9]}" =~ ^(-|[0-9a-f]{64})$ ]]; then
      echo "[ERROR][stop][${node}] Stop identity probe returned malformed fields; no node was mutated." >&2
      return 1
    fi
    if [[ ( "${identity_fields[8]}" == modern && "${identity_fields[9]}" != - ) \
          || ( "${identity_fields[8]}" == legacy \
            && ! "${identity_fields[9]}" =~ ^[0-9a-f]{64}$ ) ]]; then
      echo "[ERROR][stop][${node}] Stop contract and cgroup fingerprint disagree; no node was mutated." >&2
      return 1
    fi
    if [[ "${identity_fields[6]}" != "${#NODE_LIST[@]}" \
          || "${identity_fields[7]}" != "${node_index}" ]]; then
      echo "[ERROR][stop][${node}] Embedded topology does not prove complete ordered membership: embedded_nnodes=${identity_fields[6]} embedded_rank=${identity_fields[7]} expected_nnodes=${#NODE_LIST[@]} expected_rank=${node_index}; no node was mutated." >&2
      return 1
    fi
    if [[ -z "${expected_token}" ]]; then
      expected_token=${identity_fields[0]}
      expected_epoch=${identity_fields[1]}
      expected_snapshot=${identity_fields[2]}
      expected_log_dir=${identity_fields[3]}
      expected_target=${identity_fields[4]}
      expected_contract=${identity_fields[8]}
    elif [[ "${identity_fields[0]}" != "${expected_token}" \
          || "${identity_fields[1]}" != "${expected_epoch}" \
          || "${identity_fields[2]}" != "${expected_snapshot}" \
          || "${identity_fields[3]}" != "${expected_log_dir}" \
          || "${identity_fields[4]}" != "${expected_target}" \
          || "${identity_fields[8]}" != "${expected_contract}" ]]; then
      echo "[ERROR][stop][${node}] Nodes do not share one token/epoch/snapshot/log/target/closure-contract identity; no node was mutated." >&2
      return 1
    fi
    # Each rank's immutable control script contains node/rank-specific content,
    # so command SHA256 is intentionally stored and revalidated per node.
    stop_specs+=("${node}=${identity_fields[5]}:${identity_fields[9]}")
  done
  if [[ "${expected_contract}" == legacy ]]; then
    if ! arm_legacy_stop_nodes_parallel \
        "${expected_token}" "${expected_epoch}" "${expected_snapshot}" \
        "${expected_log_dir}" "${expected_target}" "${stop_specs[@]}"; then
      echo "[ERROR] Legacy all-node freeze barrier was incomplete; no node entered the irreversible commit phase. Any exact armed receipts remain frozen for an idempotent stop retry, and rendezvous reservations remain quarantined." >&2
      return 1
    fi
    echo "[INFO] Legacy all-node freeze barrier accepted exact receipts from every node."
  fi
  local failed=0
  if ! stop_nodes_parallel \
      "${expected_token}" "${expected_epoch}" "${expected_snapshot}" \
      "${expected_log_dir}" "${expected_target}" "${expected_contract}" \
      "${stop_specs[@]}"; then
    failed=1
  fi
  if ! verify_stop_cleanup_closures_parallel \
      "${expected_token}" "${expected_epoch}" stopped \
      "${expected_snapshot}" "${expected_log_dir}" "${expected_target}" \
      "${expected_contract}" "${stop_specs[@]}"; then
    failed=1
  fi
  if (( failed == 0 )); then
    if ! release_rendezvous_ports "${expected_token}"; then
      echo "[ERROR] Exact stopped closure was proven, but token-bound rendezvous release failed; reservations remain quarantined." >&2
      failed=1
    fi
  else
    echo "[ERROR] One or more nodes failed exact stop closure; preserving rendezvous reservations." >&2
  fi
  return "${failed}"
}

case "${ACTION}" in
  prepare)
    run_prepare
    ;;
  launch)
    run_launch
    ;;
  all)
    # Do not perform remote preparation for a job that can never launch.
    validate_gradient_reduce_contracts
    validate_restart_contract
    ensure_training_python
    ensure_local_source_snapshot
    resolve_minibatch_throughput_contract
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
