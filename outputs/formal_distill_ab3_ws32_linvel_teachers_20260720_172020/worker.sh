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
    MASTER_PORT=31721
    RUN_NAME=distill_linvel_original3_ws32_t9k_20260720_172020
    WANDB_RUN_ID=tigpqe1l
    LAUNCH_TOKEN=21bfcb989958b6e473aca241184bd245db50b458e70d3dbca270e56605afa46d
    BANK_REL=data/ds_as_data/teacher_ab3_true_original_solid_sharedphys__src_d67ffcdb2f5676d8ae59c8fd0450f5e2c7d498617113ab58e33da6871266fd2d
    SINGLE_SOURCE_DIGEST=a8af0278ea6da305fc9624a7dacacc16dbc3c5403580e99d812aea6efd2a4c3e
    SINGLE_VIEW_DIGEST=990ac19acd647d87428fb17bc982dc7978530689b8023e5b25b191180481f5e6
    SHARD_SOURCE_DIGEST=2cb39848ded7589e424073b70d52ab4e0b4573ce4548187cdf24076c50a139ba
    SOLID_MANIFEST_SHA=31ff7a044ba4ff6b2363a2e35850a097145412f17c3d7af152c327888ab97626
    SINGLE_MANIFEST_SHA=b9ad1c8c5bd7d9096c93ccf3ab8b13ab240f0705ee60170dd2f7ec944833e152
    SHARD_MANIFEST_SHA=f935854de44e003887445411538235977d7c2a23e0cf1fec33c1c3e56a8f0233
    REPLAY_MANIFEST_SHA=50e5ac347a28fcb75dac90a4c317c831a63ff85280a6626b9ae19f9c20e03ba8
    TEACHER_SHA=7a342bb21830024495dfe3f95100b2d8c50de54398ed740cd158983f76a75a07
    TEACHER_MODEL=model_09000.pt
    TEACHER_PARENT_RUN=zihanw22/carry-any/rukkpmdv
    MOTION_GENERATOR_SHA=
    MOTION_BARREL_SHA=bb6e45d37b2bd5077a1e4e09d57659d92cfdaf4722daea963c1881dcb1cf4297
    MOTION_BALL_SHA=6797ebd9f4523e00c89209f81ac94e8c9df89cb6a37f9ccb4b33acbb1b79bb96
    MOTION_BIN_SHA=7f332ccce879c7c61e80ad84040bf1de13997dc2d464613dd7cb37b57092768e
    ;;
  rollout)
    EXPECTED_NODE_IPS=(10.99.0.77 10.99.0.165 10.99.0.227 10.99.0.18)
    MASTER_ADDR=10.99.0.77
    MASTER_PORT=31731
    RUN_NAME=distill_linvel_rollout3_ws32_t10k_20260720_172020
    WANDB_RUN_ID=ugqc8xn0
    LAUNCH_TOKEN=97ac85ac4f882be35590620dbe4fafc0c85d62c3b2934906d3c934f0cee35206
    BANK_REL=data/ds_as_data/teacher_ab3_true_rollout_u8_solid_sharedphys__src_5ae3055eeff0098dad099ec26d721b76ac7675207e7d41a6ee3aaa2a15080b49
    SINGLE_SOURCE_DIGEST=18aa7972ca802ca654420bb06475ce9731cead28c46010b28680341f9bdcaf25
    SINGLE_VIEW_DIGEST=25d7623ec5d785ee12595322f5939afe4655ea8472ed14fc9b1af5cf2c915f83
    SHARD_SOURCE_DIGEST=65892fc13af9ef3115b3ddc671ac13cb1fd700b97b2c0fc38e4168aa785fbdaf
    SOLID_MANIFEST_SHA=8c9d666027e0f462c7414399c1555fd32b761654e292a1a3fb5e6352a4be4d6a
    SINGLE_MANIFEST_SHA=c1e0c883be5b1062e5510a6406bc10ee056642fdba2d9d2d9c974e9cc65d7a65
    SHARD_MANIFEST_SHA=5335d513dc6ffac141754df3b04f4eb3eab76a9b29afc45d528948cf2c475776
    REPLAY_MANIFEST_SHA=c2b4b84abd26ad75f037d88983fdc51dc32d3f2699b876790fa26f5b8f38703a
    TEACHER_SHA=3700483de167cdac73c4ed495997e7071d7d9fb2b5283ab16194c63e8b561587
    TEACHER_MODEL=model_10000.pt
    TEACHER_PARENT_RUN=zihanw22/carry-any/ppclmh15
    MOTION_GENERATOR_SHA=80cb13e13590239d015ba0a29bdbae901b7785a9789d144745c7bd330059cd68
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
readonly LAUNCH_ROOT=/home/ubuntu/FAR/holosoma_runs/formal_distill_ab3_ws32_linvel_teachers_20260720_172020
readonly RUN_ROOT="${LAUNCH_ROOT}/${ARM}"
readonly RUN_CONTRACT="${LAUNCH_ROOT}/run_contract.json"
readonly RUN_CONTRACT_SHA=ec55853de00a09cf31f1cf76f21ee5bf9c8253b119362f93e0e079cbf2def93b
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
readonly TEACHER_CHECKPOINT="/home/ubuntu/FAR/holosoma_runs/.assets/distill-teachers/${TEACHER_SHA}/${TEACHER_MODEL}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/wandb" "${RUN_ROOT}/training_logs" "${RUN_ROOT}/provenance-cache/node_${NODE_RANK}"
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
check_sha "${RUN_CONTRACT_SHA}" "${RUN_CONTRACT}"
check_sha "${REPLAY_MANIFEST_SHA}" "${REPLAY_MANIFEST}"
check_sha "${PYTHON_RUNTIME_MANIFEST_SHA}" "${PYTHON_RUNTIME}/.holosoma-runtime-manifest.sha256"
check_sha "${NCCL_SHA}" "${NCCL_ROOT}/libnccl.so.2"
check_sha "${TEACHER_SHA}" "${TEACHER_CHECKPOINT}"
check_sha "${SOLID_MANIFEST_SHA}" "${BANK}/manifest.json"
check_sha 02acd5041a0dc9f5a60e0f3a09b9de7014a1d6fc4032113d8ead2682ac6c5afc "${BANK}/_clip_object_urdf_map.json"
check_sha "${SINGLE_MANIFEST_SHA}" "${SINGLE_DIR}/manifest.json"
check_sha "${SHARD_MANIFEST_SHA}" "${SHARD_ROOT}/manifest.json"
check_sha "${MOTION_BARREL_SHA}" "${BANK}/scaledown__any_barrel_25.npz"
check_sha "${MOTION_BALL_SHA}" "${BANK}/unscale__any_ball_29.npz"
check_sha "${MOTION_BIN_SHA}" "${BANK}/unscale__any_bin_29.npz"
[[ -d "${CONTACT_ROOT}/clips" ]]

/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3.11 - "${TEACHER_CHECKPOINT}" <<'PY'
from __future__ import annotations

import sys
import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
actor = state.get("actor_model_state_dict")
if not isinstance(actor, dict):
    raise SystemExit("[ERROR] teacher checkpoint lacks actor_model_state_dict")
expected = {
    "actor_module.module.0.weight": (512, 178),
    "actor_module.module.2.weight": (256, 512),
    "actor_module.module.4.weight": (128, 256),
    "actor_module.module.6.weight": (29, 128),
}
for key, shape in expected.items():
    value = actor.get(key)
    if value is None or tuple(value.shape) != shape:
        raise SystemExit(f"[ERROR] teacher actor mismatch {key}: actual={getattr(value, 'shape', None)} expected={shape}")
config = state.get("experiment_config", {})
terms = config.get("observation", {}).get("groups", {}).get("actor_obs", {}).get("terms", {})
if list(terms)[-1:] != ["base_lin_vel"]:
    raise SystemExit("[ERROR] teacher actor_obs metadata does not end in base_lin_vel")
if state.get("next_iter") not in (9000, 10000):
    raise SystemExit(f"[ERROR] unexpected teacher next_iter={state.get('next_iter')!r}")
print(f"[INFO] teacher_contract_ok first_layer=512x178 action_dim=29 next_iter={state['next_iter']}")
PY

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
export HOLOSOMA_LAUNCH_EPOCH=1784568020

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC=8
export NNODES=4
export MASTER_ADDR
export MASTER_PORT
export PER_GPU_ENVS=64
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
export HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY=1

export AS_SUCCESS133_FINAL0P5=0
export AS_CONTACT_AWARE=1
export AS_CONTACT_AWARE_HISTORY=0
export ROOT_COMMAND_MODE=contact-aware
export OMOMO_DATA_DIR="${BANK}"
export OMOMO_OBJECT_MAP="${BANK}/_clip_object_urdf_map.json"
export OMOMO_EXPECTED_TOTAL=3
export RESUME_FROM_BOX_EXPECTED_TOTAL=3
export AS_CONTACT_EXPORT_ROOT="${CONTACT_ROOT}"
export RESUME_FROM_BOX_CONTACT_EXPORT_ROOT="${CONTACT_ROOT}"
export AS_SINGLE_SLOT_MOTION_BASE="${SINGLE_BASE}"
export AS_RANK_LOCAL_SHARDS=1
export AS_RANK_SHARD_ROOT="${SHARD_ROOT}"
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST="${SINGLE_SOURCE_DIGEST}"
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST="${SINGLE_VIEW_DIGEST}"
export HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST="${SHARD_SOURCE_DIGEST}"
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_DIR="${SINGLE_DIR}"
export HOLOSOMA_EXTERNAL_AS_WORLD_SIZE=32
export HOLOSOMA_RANK_LOCAL_MOTION_ROOT="${SHARD_ROOT}"
export HOLOSOMA_MOTION_SHARD_MANIFEST="${SHARD_ROOT}/manifest.json"
export HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED=1
export HOLOSOMA_REQUIRE_RANK_LOCAL_SHARD_PROVENANCE=1
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=0
export CLIP_WEIGHTING_STRATEGY=uniform_clip
export STUDENT_MOTION_END_MODE=episodic
export ENABLE_DEFAULT_POSE_PREPEND=True
export DEFAULT_POSE_PREPEND_DURATION_S=0.2
export ENABLE_DEFAULT_POSE_APPEND=True
export DEFAULT_POSE_APPEND_DURATION_S=2.0
export CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION=True
export ENABLE_OFFLINE_CONTACT_GUIDANCE=True
export PURE_REAL_OMOMO_PREFIXES='["scaledown__","unscale__"]'
export USE_ADAPTIVE_TIMESTEPS_SAMPLER=True
export UNIFORM_T1_WINDOW_SAMPLING_ENABLED=True
export UNIFORM_T1_WINDOW_HALF_WIDTH_STEPS=50
export UNIFORM_T1_WINDOW_DENSITY_BOOST=7.0
export UNIFORM_T1_WINDOW_TARGET_SAMPLE_FRAC=
export START_AT_TIMESTEP_ZERO_PROB=0.2
export START_AT_TIMESTEP_ZERO_PROB_END=0.2
export START_AT_TIMESTEP_ZERO_PROB_START_ITER=0
export START_AT_TIMESTEP_ZERO_PROB_END_ITER=39999
export FREEZE_AT_TIMESTEP_ZERO_PROB=0.0
export FREEZE_AT_TIMESTEP_ZERO_PROB_END=0.0
export FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=0
export FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=39999

export TEACHER_CHECKPOINT
export TEACHER_CHECKPOINT_EXPECTED_SHA256="${TEACHER_SHA}"
export TEACHER_PARENT_RUN
export MOTION_GENERATOR_TEACHER_EXPECTED_SHA256="${MOTION_GENERATOR_SHA}"
export REQUIRE_MOTION_GENERATOR_TEACHER_MATCH=0
export TEACHER_OBS_KEYS=actor_obs
export TEACHER_ACTOR_OBS_HISTORY_LENGTH=1
export TEACHER_COMPAT_PROFILE=none
export TEACHER_PERCEPTION_PRESET=none
export TEACHER_PERCEPTION_OBS_KEY=
export STRICT_TEACHER_LOAD=True
export CLIP_TEACHER_ACTIONS=True
export CLIP_ACTIONS_THRESHOLD=8.0

export EXP=g1-29dof-wbt-w-object-distill-sparse-root-cmd-teacher-linvel
export PERCEPTION_PRESET=camera_depth_d435i
export STUDENT_POLICY_TYPE=mlp
export STUDENT_ACTOR_HIDDEN_DIMS='[512,256,128]'
export STUDENT_ACTOR_INPUTS="['actor_obs_root_contact_aware','actor_obs_drop_button','actor_obs_proprio_with_actions_no_linvel']"
export CONTACT_AWARE_HISTORY=0
export TEACHER_ACTOR_OBS_HISTORY_LENGTH=1
export STUDENT_PROPRIO_HISTORY_LENGTH=1
export STUDENT_ACTION_HISTORY_LENGTH=1
export CRITIC_PROPRIO_HISTORY_LENGTH=1
export HOLOSOMA_PERCEPTION_INCLUDE_ROBOT_MESH=1
export CAMERA_APPLY_SENSOR_NOISE=True
export CAMERA_WARP_EDGE_NOISE=True
export CAMERA_WARP_ENABLE_HOLES=True
export CAMERA_WARP_HOLE_PROB=0.2
export CAMERA_WARP_ADDITIVE_NOISE_STD=0.03
export CAMERA_WARP_DEPTH_OFFSET_STD=0.03
export AS_PUSH_INTERVAL_S='[0.5,2.0]'
export AS_MAX_PUSH_VEL='[0.7,0.7,0.25,0.7,0.7,1.0]'
export HOLOSOMA_OBJECT_COLLIDER_TYPE=convex_decomposition
export HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=0

export NUM_LEARNING_ITERATIONS=40000
export SAVE_INTERVAL=1000
export EXPORT_ONNX=True
export FIXED_BC_EVAL_LOG_INTERVAL=100
export FIXED_BC_GUARD_ENABLED=True
export FIXED_BC_GUARD_REFERENCE_END_EPOCH=600
export FIXED_BC_GUARD_MAX_REFERENCE_RATIO=100.0
export FIXED_BC_GUARD_ABSOLUTE_MAX_MU_MSE=100.0
export FIXED_BC_GUARD_START_EPOCH=700
export FIXED_BC_GUARD_CONSECUTIVE_EVALS=3
export NUM_MINI_BATCHES=16
export NUM_LEARNING_EPOCHS=1
export PPO_LR_SCHEDULE=fixed
export PPO_DESIRED_KL=0.01
export ACTOR_LR=1e-3
export CRITIC_LR=1e-3
export ACTOR_MIN_LR=1e-5
export ACTOR_MAX_LR=1e-2
export CRITIC_MIN_LR=1e-5
export CRITIC_MAX_LR=1e-2
export INIT_NOISE_STD=0.1
export ACTOR_MIN_NOISE_STD=0.1
export ENTROPY_COEF=0
export DAGGER_LOSS_COEF=1
export DAGGER_MATCH_STD=False
export PPO_START_EPOCH=0
export DAGGER_END_EPOCH=700
export PPO_START_COEFF=0.0
export PPO_TARGET_COEFF=0.0
export PPO_SCHEDULE_STEP_EPOCHS=700
export DAGGER_REPLAY_ENABLED=True
export DAGGER_REPLAY_CAPACITY=512
export DAGGER_REPLAY_BATCH_SIZE=96
export DAGGER_REPLAY_FRACTION=0.5
export DAGGER_REPLAY_SEED=42
export HOLOSOMA_DAGGER_SUPERVISED_ONLY=1
export HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP=1
export HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH=0
export HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD=0
export HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC=0
export HOLOSOMA_SKIP_INITIAL_CHECKPOINT=1
export HOLOSOMA_SKIP_GRAD_FINITE_CHECK=0
export HOLOSOMA_SKIP_LOSS_FINITE_CHECK=0
export HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION=0
export HOLOSOMA_MOTION_METRICS_INTERVAL=16

unset RESUME_CKPT RESUME_CHECKPOINT RESUME_MODEL_FILE RESUME_STEP
unset POLICY_INIT_CKPT POLICY_INIT_CHECKPOINT RESUME_FROM_BOX BOX_RESUME_CKPT
unset RESUME_WANDB_ID WANDB_MODE WANDB_DISABLED
export RESUME_FROM_BOX=0
export WANDB_PROJECT=carry-any
export WANDB_ENTITY=zihanw22
export TRAINING_PROJECT=carry-any
export WANDB_RUN_ID
export WANDB_RESUME=must
export WANDB_RESUME_SAME_RUN=0
export WANDB_CONSOLE=off
export WANDB_INIT_TIMEOUT=120
export WANDB_DIR="${RUN_ROOT}/wandb"
export LOGGER_BASE_DIR="${RUN_ROOT}/training_logs"
export RUN_NAME
export TRAINING_NAME="${RUN_NAME}"
export SCHEDULE_NAME="pure_bc_replay_linvel_teacher_ab3_ws32"
export SCHEDULE_NOTES="Fresh 32-rank, three-motion perception-student distillation. The matched new privileged teacher consumes exact 178D actor_obs including base_lin_vel; the student excludes base_lin_vel and consumes contact-aware sparse root, drop button, proprio/action history, and noisy depth. Pure actor supervised DAgger with bounded rank-local replay, PPO exactly zero, fixed actor LR 1e-3, teacher mean labels clipped to +/-8, no resume or policy initialization."

if [[ "${NODE_RANK}" == 0 && "${SKIP_REPLAY_REMOTE_VERIFY:-0}" != 1 ]]; then
  "${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/wandb_replay_preflight.py" verify \
    --manifest "${REPLAY_MANIFEST}" \
    --expected-manifest-sha256 "${REPLAY_MANIFEST_SHA}" \
    --expected-source-snapshot-id "${SOURCE_ID}" \
    --expected-entity zihanw22 \
    --expected-project carry-any \
    --expected-run-id "${WANDB_RUN_ID}" \
    --expected-run-name "${RUN_NAME}" \
    --expected-world-size 32
fi

if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "[INFO] formal_distill_ab_worker_preflight_only_ok arm=${ARM} node_rank=${NODE_RANK} host=$(hostname) ip=${EXPECTED_NODE_IP}"
  exit 0
fi

cd "${SOURCE_ROOT}"
echo "[INFO] formal_distill_ab_worker_preflight_ok arm=${ARM} node_rank=${NODE_RANK} host=$(hostname) ip=${EXPECTED_NODE_IP} source=${SOURCE_ID} wandb=zihanw22/carry-any/${WANDB_RUN_ID} teacher_sha=${TEACHER_SHA} teacher_input=178 student_has_linvel=false fresh_student=true"
exec bash distill_as_button.sh "${TEACHER_CHECKPOINT}" \
  reward:g1-29dof-wbt-w-object-generalist-offline-contact-guidance \
  --algo.config.reset-rollout-at-checkpoint=False \
  --algo.config.num-steps-per-env=24
