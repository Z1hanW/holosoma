#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 16 || ! $1 =~ ^(canary|formal)$ || ! $2 =~ ^(teacher_9x40k|teacher_ch228k)$ ]]; then
  echo "usage: $0 MODE TEACHER_ARM EXPECTED_IP SOURCE_ROOT PERSIST_ROOT MASTER_PORT RUN_ID RUN_NAME CONTRACT_PATH CONTRACT_SHA RULE90_PATH RULE90_SHA CANARY_PATH CANARY_SHA COMMIT_SHA TREE_SHA" >&2
  exit 2
fi

readonly MODE=$1 TEACHER_ARM=$2 EXPECTED_IP=$3 SOURCE_ROOT=$4 PERSIST_ROOT=$5 MASTER_PORT=$6
readonly RUN_ID=$7 RUN_NAME=$8 CONTRACT_PATH=$9 CONTRACT_SHA=${10} RULE90_PATH=${11}
readonly RULE90_SHA=${12} CANARY_PATH=${13} CANARY_SHA=${14} COMMIT_SHA=${15} TREE_SHA=${16}
readonly REMOTE_URL=https://github.com/Z1hanW/holosoma
readonly REMOTE_REF=main
readonly WORLD_SIZE=8 ENVIRONMENTS_PER_RANK=2048 TOTAL_ENVIRONMENTS=16384
readonly PYTHON_BIN=/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3.11
readonly PYTHON_RUNTIME_ROOT=/data/holosoma_runs/.runtime/python/python-runtime-v2-dd7ca81fa848917c362b3a239893a7a26f4c89d42b4f85cb515d91622f1690bc
readonly PYTHON_RUNTIME=${PYTHON_RUNTIME_ROOT}/site-packages
readonly PYTHON_RUNTIME_SHA256=dd7ca81fa848917c362b3a239893a7a26f4c89d42b4f85cb515d91622f1690bc
readonly NCCL_ROOT=/home/ubuntu/FAR/holosoma_runs/.runtime/nccl/e4a7aee9c3eecf53fac780441d2f03b578ab8db8874b71f8e391bcec7adb2899
readonly NCCL_SHA256=e4a7aee9c3eecf53fac780441d2f03b578ab8db8874b71f8e391bcec7adb2899
readonly ENCODER_TYPE=far_tracking_cnn_small
readonly CONTACT_PROFILE=no_contact
readonly DISTILL_REWARD_CONFIG_VALUE=g1-29dof-wbt-w-object-generalist-tracking-no-contact
readonly ENABLE_OFFLINE_CONTACT_GUIDANCE_VALUE=False POSITIVE_CONTACT_REWARD_VALUE=false
readonly CONTACT_SUMMARY='offline contact guidance disabled (weight 0)'
readonly OBJECT_MAP_SHA256=867522fd61c63e6fcf37e0a041792f438e821f34ff482e26b07b47de6bfb7b59
readonly ASSET_ROOT=/data/holosoma_training/formal_distill_prism137_rollout_teacher_ab_ws8x2_20260830

case ${TEACHER_ARM} in
  teacher_9x40k)
    readonly PARENT_TEACHER_RUN=9xkizjec TEACHER_MODEL_NAME=model_40000
    readonly TEACHER_ROOT=${ASSET_ROOT}/teachers/9xkizjec
    readonly TEACHER_SHA256=8ea11d3b5862c075becd232aa5d557a180966289710656d1d6f4daadab2edef3
    readonly TEACHER_ONNX_SHA256=81d583b78ad721213420c1959c1607d2c9bf0a8e7fd0e6ca3c5b6fe96655c2e0
    readonly TEACHER_PAIR_SHA256=3037ef8564154d1c6069304769d277480248b1b2f7d4078ea6f5802885e6fbdb
    readonly T1_VALID_WINDOW_CLIPS=134
    readonly RAW_ROLLOUT_DIGEST=e5a49e8917624b6159342baa80e78942b585302f357b4a52281c78ed1f77119d
    readonly COMMAND_BANK_DIGEST=0bed60584acf93924aad962eea63dcf1231aa3bf7a8372ce1573ba4179c80d42
    readonly COMMAND_MANIFEST_SHA256=c891f1a321c36928ecbc97a30a6aaf0e652b07dfdf03a57a379da5ac1c17826a
    readonly SINGLE_SLOT_SOURCE_DIGEST=70d969fad7673cc33d90085c9a4f9f7efba963908e8fbe943f5c8eeeb1ebe10e
    readonly SINGLE_SLOT_VIEW_DIGEST=a618d992e1eda1ccdfad02d3b60abf2297efa76491fbff2ef2db5ac10be9c331
    readonly SINGLE_SLOT_MANIFEST_SHA256=dc54af247772cb257c47eb27a0947c13953f90ed5a2a3f77b744d536e98d0ec6
    readonly RANK_SHARD_SOURCE_DIGEST=cf674ef17e43b8d815ad53cb092a759bc2f7239c6d082206594d876472e67f7e
    readonly RANK_SHARD_MANIFEST_SHA256=f1984964315e90fb0f1b377da9c28bf209e724f2aa64dc272443efd8470947fb
    readonly CONTACT_SOURCE_DIGEST=f18d6bcb318fb228d865aff962e54ad4714cb93047539b133af56cde699ec3f5
    readonly CONTACT_MANIFEST_SHA256=e1ec91a8fb23bfac5733cf96c723786251111c8504a7a7d5711a7b29f1723aed
    readonly RAW_MOTION_DIR=/data/holosoma_inputs/9xkizjec_model40000_rollout137_20260830/by-source/${RAW_ROLLOUT_DIGEST}
    readonly SOURCE_MOTION_DIR=/data/holosoma_inputs/9xkizjec_model40000_rollout137_precomputed_turn_forward_v1/by-source/${COMMAND_BANK_DIGEST}
    readonly CONTACT_ROOT=/data/holosoma_inputs/9xkizjec_model40000_rollout137_runtime_contact_intervals/by-source/${CONTACT_SOURCE_DIGEST}
    ;;
  teacher_ch228k)
    readonly PARENT_TEACHER_RUN=ch2ckwzw TEACHER_MODEL_NAME=model_28000
    readonly TEACHER_ROOT=${ASSET_ROOT}/teachers/ch2ckwzw
    readonly TEACHER_SHA256=c5d0977a1cc02160ef08140f89d40f1b64302510216b1256597cb697d31202dc
    readonly TEACHER_ONNX_SHA256=9e44dfec9472ee18cef57e7b8ea025498086c90f967e5a3895c3e6013584b36b
    readonly TEACHER_PAIR_SHA256=01257a4e6a6c656475d7afe38a1bd407128f6112f87a24753d34d73ba51b8de9
    readonly T1_VALID_WINDOW_CLIPS=133
    readonly RAW_ROLLOUT_DIGEST=c6e5ecd09c0237be13b7623e757a50ffcbbe61b7af7e5759dcedc9ba278f12b5
    readonly COMMAND_BANK_DIGEST=78640eff4bcca7052e0febfd0b8ad9f57fac238e5fdfe4426a4b42f28fdb3306
    readonly COMMAND_MANIFEST_SHA256=665249ef7880a99cf0dd0758eb3c10c097357ebbc0c38c562a85ba9731cbf4c4
    readonly SINGLE_SLOT_SOURCE_DIGEST=08551bf9b2621c74315fcc0169255cc3c3f74cb881ba7364a21a88c390f20263
    readonly SINGLE_SLOT_VIEW_DIGEST=30a10b728c3252605db69ca23184460569aa30b6569a1fa97a00d9b2afcb1b23
    readonly SINGLE_SLOT_MANIFEST_SHA256=1811529ed4e03a860c46a0de4fb88b51cbdb93cc3fc79aa68de6646fbb96e473
    readonly RANK_SHARD_SOURCE_DIGEST=0b80793e304e4284581645a713dcadb983e548da28cd536fe518bf862647a83e
    readonly RANK_SHARD_MANIFEST_SHA256=2c502e367e38efb5770ed134615581c0d4fe41c3688c7d4182afbed8f6a9c58a
    readonly CONTACT_SOURCE_DIGEST=a4453a2d1dd0ee37da916429bbd8bb697333b7cef633e37af37f0d009212f1f0
    readonly CONTACT_MANIFEST_SHA256=e12868dc7c058126f8602a8423cdbc33965d1c75d13adbef506ed8e42aceaf94
    readonly RAW_MOTION_DIR=/data/holosoma_inputs/ch2ckwzw_model28000_rollout137_20260830/by-source/${RAW_ROLLOUT_DIGEST}
    readonly SOURCE_MOTION_DIR=/data/holosoma_inputs/ch2ckwzw_model28000_rollout137_precomputed_turn_forward_v1/by-source/${COMMAND_BANK_DIGEST}
    readonly CONTACT_ROOT=/data/holosoma_inputs/ch2ckwzw_model28000_rollout137_runtime_contact_intervals/by-source/${CONTACT_SOURCE_DIGEST}
    ;;
esac
readonly SOURCE_OBJECT_SPEC_PATH=${SOURCE_MOTION_DIR}/_clip_object_urdf_map.json
readonly SINGLE_SLOT_DIR=${SOURCE_MOTION_DIR}/_single_slot_motion_bank/by-source/${SINGLE_SLOT_VIEW_DIGEST}
readonly TEACHER=${TEACHER_ROOT}/${TEACHER_MODEL_NAME}.pt
readonly TEACHER_ONNX=${TEACHER_ROOT}/${TEACHER_MODEL_NAME}.onnx
readonly TEACHER_PAIR=${TEACHER_ROOT}/${TEACHER_MODEL_NAME}.pair.json
readonly RANK_SHARD_DIR=${SINGLE_SLOT_DIR}/_rank_shards/by-source/${RANK_SHARD_SOURCE_DIGEST}/ws8

if [[ ${MODE} == formal ]]; then
  readonly TARGET_ITERATIONS=40000 SAVE_INTERVAL=1000 CURRICULUM_END_ITER=39999
  [[ ${RUN_ID} =~ ^[a-z0-9]{8}$ && ${RUN_NAME} != - && ${CONTRACT_PATH} != - \
     && ${RULE90_PATH} != - && ${CANARY_PATH} != - ]]
else
  readonly TARGET_ITERATIONS=2 SAVE_INTERVAL=2 CURRICULUM_END_ITER=1
fi

readonly STUDENT_TERMINATION_PROFILE_VALUE=g1_29dof_wbt_generalist_z_only
readonly BAD_REF_POS=1.0 BAD_REF_ORI=1.2 BAD_BODY_POS=0.55 BAD_OBJECT_POS=0.65 BAD_OBJECT_ORI=1.2
readonly PPO_START=0.1 PPO_TARGET=0.9 DAGGER_MATCH_STD_VALUE=True
readonly PPO_START_NOISE_STD_VALUE= PPO_START_NOISE_STD_FORCE_NONE_VALUE=True
readonly START_ZERO_END=1.0
if [[ ${MODE} == formal ]]; then
  readonly START_ZERO_START_ITER=2500 START_ZERO_END_ITER=39999
else
  readonly START_ZERO_START_ITER=0 START_ZERO_END_ITER=1
fi
readonly SCHEDULE_NAME_VALUE=distill_prism137_teacher_rollout_ab_sw_schedule
readonly SCHEDULE_SUMMARY='SW BadTrackingZOnly thresholds; PPO/BC 0.1/0.9 to 0.9/0.1; timestep-zero probability 0.2 to 1.0; uniform clip weighting plus legacy uniform-T1 window boost 7x; adaptive failure sampler off'

check_sha() {
  local expected=$1 path=$2 actual
  [[ -f ${path} && ! -L ${path} ]] || { echo "[ERROR] missing regular integrity input: ${path}" >&2; exit 2; }
  actual=$(sha256sum "${path}" | awk '{print $1}')
  [[ ${actual} == "${expected}" ]] || {
    echo "[ERROR] SHA mismatch: ${path} expected=${expected} actual=${actual}" >&2
    exit 2
  }
}

hostname -I | tr ' ' '\n' | grep -Fxq "${EXPECTED_IP}" || {
  echo "[ERROR] expected node IP ${EXPECTED_IP}, got $(hostname -I)" >&2; exit 2;
}
readonly VERIFY_ROOT=${PERSIST_ROOT}/${MODE}_${TEACHER_ARM}_${CONTACT_PROFILE}_git_verification
mkdir -p "${VERIFY_ROOT}"
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/verify_formal_git_checkout.py" \
  --source-root "${SOURCE_ROOT}" --remote-url "${REMOTE_URL}" --remote-ref "${REMOTE_REF}" \
  --commit "${COMMIT_SHA}" --tree "${TREE_SHA}" --output "${VERIFY_ROOT}/node_0.json"
readonly GIT_MANIFEST_SHA256=$(git -C "${SOURCE_ROOT}" ls-tree -r --full-tree "${COMMIT_SHA}" | sha256sum | awk '{print $1}')
readonly SOURCE_SNAPSHOT_ID=src-${GIT_MANIFEST_SHA256}

check_sha "${COMMAND_MANIFEST_SHA256}" "${SOURCE_MOTION_DIR}/manifest.json"
check_sha "${SINGLE_SLOT_MANIFEST_SHA256}" "${SINGLE_SLOT_DIR}/manifest.json"
check_sha "${OBJECT_MAP_SHA256}" "${SOURCE_OBJECT_SPEC_PATH}"
check_sha "${TEACHER_SHA256}" "${TEACHER}"
check_sha "${TEACHER_ONNX_SHA256}" "${TEACHER_ONNX}"
check_sha "${TEACHER_PAIR_SHA256}" "${TEACHER_PAIR}"
check_sha "${RANK_SHARD_MANIFEST_SHA256}" "${RANK_SHARD_DIR}/manifest.json"
check_sha "${CONTACT_MANIFEST_SHA256}" "${CONTACT_ROOT}/manifest.json"
"${PYTHON_BIN}" - "${RAW_MOTION_DIR}/manifest.json" "${SOURCE_MOTION_DIR}/manifest.json" \
  "${RAW_ROLLOUT_DIGEST}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

raw_path, command_path, expected_digest = map(Path, sys.argv[1:])
expected_digest = str(expected_digest)
raw_bytes = raw_path.read_bytes()
raw = json.loads(raw_bytes)
command = json.loads(command_path.read_text())
if raw.get("source_digest") != expected_digest:
    raise SystemExit("raw rollout manifest source_digest mismatch")
if command.get("source_payload_digest") != expected_digest:
    raise SystemExit("command bank does not bind the exact raw rollout digest")
if command.get("source_manifest_sha256") != hashlib.sha256(raw_bytes).hexdigest():
    raise SystemExit("command bank does not bind the exact raw rollout manifest bytes")
PY
check_sha "${NCCL_SHA256}" "${NCCL_ROOT}/libnccl.so.2"
if [[ ${MODE} == formal ]]; then
  check_sha "${CONTRACT_SHA}" "${CONTRACT_PATH}"
  check_sha "${RULE90_SHA}" "${RULE90_PATH}"
  check_sha "${CANARY_SHA}" "${CANARY_PATH}"
fi

readonly MOTION_DIR=${SOURCE_MOTION_DIR}
readonly OBJECT_SPEC_PATH=${MOTION_DIR}/_clip_object_urdf_map.json
[[ $(find "${MOTION_DIR}" -maxdepth 1 -type f ! -type l -name '*.npz' | wc -l) -eq 137 ]]
[[ $(find "${MOTION_DIR}/_single_slot_urdfs" -maxdepth 1 -type f ! -type l -name '*.urdf' | wc -l) -eq 137 ]]
[[ $(find "${CONTACT_ROOT}/clips" -mindepth 2 -maxdepth 2 -type f -name contact_intervals.json | wc -l) -eq 137 ]]
"${PYTHON_BIN}" - "${MOTION_DIR}" <<'PY'
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
phase_counts = np.zeros((3,), dtype=np.int64)
for path in sorted(root.glob("*.npz")):
    with np.load(path, allow_pickle=False) as data:
        if not {"policy_command_xy_yaw", "policy_command_phase"} <= set(data.files):
            raise SystemExit(f"missing command fields: {path}")
        command = np.asarray(data["policy_command_xy_yaw"])
        phase = np.asarray(data["policy_command_phase"])
        if command.shape != (359, 3) or phase.shape != (359,):
            raise SystemExit(f"command timeline mismatch: {path}: {command.shape} {phase.shape}")
        if not np.all(np.isfinite(command)) or not np.all(command[:, 1] == 0.0):
            raise SystemExit(f"invalid command tensor: {path}")
        if not np.all((command[:, 0] == 0.0) | (command[:, 2] == 0.0)):
            raise SystemExit(f"forward/yaw overlap: {path}")
        phase_counts += np.bincount(phase.astype(np.int64), minlength=3)
if not np.all(phase_counts > 0):
    raise SystemExit(f"missing command phase: {phase_counts.tolist()}")
PY
PYTHONPATH="${SOURCE_ROOT}/src/holosoma:${SOURCE_ROOT}/src/holosoma_inference:${SOURCE_ROOT}/src" \
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/validate_contact_sidecars.py" \
  --motion-dir "${MOTION_DIR}" --contact-root "${CONTACT_ROOT}" \
  --expected-total 137 --motion-end-mode episodic \
  --runtime-prepend-compensation --runtime-prepend-duration-s 0.2 \
  --allow-missing-offline-contact-targets \
  --expected-valid-runtime-windows "${T1_VALID_WINDOW_CLIPS}" >/dev/null
readonly ACTUAL_T1_VALID_WINDOW_CLIPS=$("${PYTHON_BIN}" - "${CONTACT_ROOT}" <<'PY'
import json, sys
from pathlib import Path
regions = (
    "left_wrist", "right_wrist", "left_elbow", "right_elbow",
    "left_wrist_roll", "right_wrist_roll", "left_wrist_pitch",
    "right_wrist_pitch", "torso",
)
count = 0
for path in sorted((Path(sys.argv[1]) / "clips").glob("*/contact_intervals.json")):
    payload = json.loads(path.read_text())
    if any(
        isinstance(payload.get(name), list)
        and len(payload[name]) == 2
        and payload[name][0] >= 0
        and payload[name][1] > payload[name][0]
        for name in regions
    ):
        count += 1
print(count)
PY
)
[[ ${ACTUAL_T1_VALID_WINDOW_CLIPS} -eq ${T1_VALID_WINDOW_CLIPS} ]] || {
  echo "[ERROR] T1 contact-window coverage drift: actual=${ACTUAL_T1_VALID_WINDOW_CLIPS} expected=${T1_VALID_WINDOW_CLIPS}" >&2
  exit 2
}

readonly RUN_ROOT=${PERSIST_ROOT}/${MODE}_${TEACHER_ARM}_${CONTACT_PROFILE}_${RUN_ID}
readonly LOGGER_BASE_DIR=${RUN_ROOT}/training_logs
readonly SCRATCH_ROOT=/dev/shm/holosoma_distill_prism137_${MODE}_${TEACHER_ARM}_${CONTACT_PROFILE}
mkdir -p "${RUN_ROOT}" "${LOGGER_BASE_DIR}" "${PERSIST_ROOT}/wandb" \
  "${SCRATCH_ROOT}/tmp" "${SCRATCH_ROOT}/xdg-cache" "${SCRATCH_ROOT}/robot-usd-cache" \
  "${SCRATCH_ROOT}/object-usd-cache" "${SCRATCH_ROOT}/perception-mesh-cache" \
  "${SCRATCH_ROOT}/derived-data-cache" "${SCRATCH_ROOT}/provenance-cache"

unset BASH_ENV ENV CDPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONOPTIMIZE
unset PYTHONWARNINGS PYTHONBREAKPOINT PYTHONSAFEPATH LD_LIBRARY_PATH LD_PRELOAD
unset WANDB_RUN_ID WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS
[[ ${HOME:-} == /home/ubuntu ]] || {
  echo "[ERROR] unexpected service-account home: ${HOME:-<unset>}" >&2; exit 2;
}
export PATH="$(dirname "${PYTHON_BIN}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="${PYTHON_RUNTIME}:${SOURCE_ROOT}/src/holosoma:${SOURCE_ROOT}/src/holosoma_inference:${SOURCE_ROOT}/src"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export PYTHON_RUNTIME_SITEPACKAGES=${PYTHON_RUNTIME}
export PYTHON_RUNTIME_MANIFEST_SHA256=${PYTHON_RUNTIME_SHA256}
export HOLOSOMA_PYTHON_RUNTIME_MANIFEST_SHA256=${PYTHON_RUNTIME_SHA256}
export HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY=1
export NCCL_LIB_DIR=${NCCL_ROOT} NCCL_LIB_SHA256=${NCCL_SHA256}
export LD_LIBRARY_PATH=${NCCL_ROOT} LD_PRELOAD=${NCCL_ROOT}/libnccl.so.2
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y HEADLESS=1 OMP_NUM_THREADS=1
export TMPDIR=${SCRATCH_ROOT}/tmp XDG_CACHE_HOME=${SCRATCH_ROOT}/xdg-cache
export WANDB_DIR=${PERSIST_ROOT}/wandb
export HOLOSOMA_RUNTIME_SCRATCH_ROOT=${SCRATCH_ROOT}
export HOLOSOMA_ROBOT_USD_CACHE_DIR=${SCRATCH_ROOT}/robot-usd-cache
export HOLOSOMA_OBJECT_USD_CACHE_DIR=${SCRATCH_ROOT}/object-usd-cache
export HOLOSOMA_PERCEPTION_MESH_CACHE_DIR=${SCRATCH_ROOT}/perception-mesh-cache
export HOLOSOMA_ISAACSIM_KIT_ARGS="--/UJITSO/datastore/localCachePath=${SCRATCH_ROOT}/derived-data-cache --/UJITSO/datastore/localDataStore/largeChunkDiskBudgetMB=1024"
export HOLOSOMA_DATA_PROVENANCE_CACHE_ROOT=${SCRATCH_ROOT}/provenance-cache

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/verify_python_runtime_overlay.py" \
  --site-packages "${PYTHON_RUNTIME}" --manifest-sha256 "${PYTHON_RUNTIME_SHA256}" \
  --require-distribution-closure --require-current-runtime-binding
"${PYTHON_BIN}" -c 'import onnx, onnxruntime; assert onnx.__version__ == "1.21.0"; assert onnxruntime.__version__ == "1.25.1"'

mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,name,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits)
[[ ${#gpu_rows[@]} -eq 8 ]] || { echo "[ERROR] expected exactly 8 GPUs" >&2; exit 2; }
for row in "${gpu_rows[@]}"; do
  [[ ${row} == *"NVIDIA L40S"* && ${row##*, } == 0 ]] || { echo "[ERROR] GPU health: ${row}" >&2; exit 2; }
done
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d') ]]; then
  echo "[ERROR] node is not GPU-idle" >&2; exit 2
fi

if [[ ${MODE} == formal ]]; then
  "${PYTHON_BIN}" - "${CANARY_PATH}" "${COMMIT_SHA}" "${TREE_SHA}" "${SOURCE_SNAPSHOT_ID}" \
    "${ENCODER_TYPE}" "${TEACHER_ARM}" "${TEACHER_SHA256}" "${COMMAND_BANK_DIGEST}" \
    "${SINGLE_SLOT_VIEW_DIGEST}" \
    "${PPO_START}" "${PPO_TARGET}" "${POSITIVE_CONTACT_REWARD_VALUE}" \
    "${T1_VALID_WINDOW_CLIPS}" <<'PY'
import json, sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text())
positive_contact_reward = sys.argv[12].lower() == "true"
expected={"accepted":True,"world_size":8,"environments_per_rank":2048,
          "completed_iterations_per_rank":2,"commit_sha":sys.argv[2],"tree_sha":sys.argv[3],
          "source_snapshot_id":sys.argv[4],"encoder_type":sys.argv[5],
          "teacher_arm":sys.argv[6],"teacher_pt_sha256":sys.argv[7],
          "rollout_command_bank_digest":sys.argv[8],"single_slot_view_digest":sys.argv[9],
          "dataset_clip_count":137,
          "actor_scalar_observation_dim":94,"actor_total_observation_dim":126,
          "ppo_start_coefficient":float(sys.argv[10]),"ppo_target_coefficient":float(sys.argv[11]),
          "contact_profile":"no_contact","positive_contact_reward":positive_contact_reward,"export_onnx":True,
          "t1_contact_window_valid_clip_count":int(sys.argv[13]),
          "onnx_checker_passed":True,"onnxruntime_load_passed":True,
          "pytorch_ort_parity_passed":True,"finite_metrics":True,"distillation_enabled":True}
for key,value in expected.items():
    if p.get(key)!=value: raise SystemExit(f"invalid canary {key}: {p.get(key)!r} != {value!r}")
sw_expected = {
    "dagger_match_std": True,
    "ppo_start_noise_std": None,
    "termination_function": "holosoma.managers.termination.terms.wbt:BadTrackingZOnly",
    "bad_tracking_thresholds": {
        "ref_pos": 1.0, "ref_ori": 1.2, "body_pos": 0.55,
        "object_pos": 0.65, "object_ori": 1.2,
    },
    "start_at_timestep_zero_probability_start": 0.2,
    "start_at_timestep_zero_probability_end": 1.0,
    "adaptive_timestep_sampler": False,
    "clip_weighting_strategy": "uniform_clip",
    "uniform_t1_window_sampling_enabled": True,
    "uniform_t1_window_half_width_steps": 50,
    "uniform_t1_window_density_boost": 7.0,
    "command_mode": "precomputed_turn_then_forward",
    "button_window_mode": "kinematic_lift",
    "carry_window_mode": "peak_height",
}
for key, value in sw_expected.items():
    if p.get(key) != value:
        raise SystemExit(f"invalid canary {key}: {p.get(key)!r} != {value!r}")
PY
  "${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/wandb_replay_preflight.py" verify \
    --manifest "${RULE90_PATH}" --expected-manifest-sha256 "${RULE90_SHA}" \
    --required-manifest-version 1 --expected-source-snapshot-id "${SOURCE_SNAPSHOT_ID}" \
    --expected-entity zihanw22 --expected-project carry-any --expected-run-id "${RUN_ID}" \
    --expected-run-name "${RUN_NAME}" --expected-world-size 8
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT
export TORCH_DIST_BACKEND=gloo TORCH_DIST_TIMEOUT_SEC=3600 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1
export HOLOSOMA_GLOO_GRAD_REDUCE=1 HOLOSOMA_GLOO_BARRIER=1 HOLOSOMA_GLOO_SMALL_COLLECTIVES=1
export HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=0 HOLOSOMA_RANK_VISIBLE_DEVICES=1 HOLOSOMA_RANK_LOCAL_CPU_AFFINITY=1
export HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE=1 HOLOSOMA_CONTIGUOUS_MINIBATCHES=1

export HOLOSOMA_SOURCE_ROOT=${SOURCE_ROOT} HOLOSOMA_SOURCE_SNAPSHOT_ID=${SOURCE_SNAPSHOT_ID}
export HOLOSOMA_SOURCE_MANIFEST_SHA256=${GIT_MANIFEST_SHA256}
export HOLOSOMA_GIT_REMOTE_URL=${REMOTE_URL} HOLOSOMA_GIT_REMOTE_REF=${REMOTE_REF}
export HOLOSOMA_GIT_COMMIT_SHA=${COMMIT_SHA} HOLOSOMA_GIT_TREE_SHA=${TREE_SHA}
export HOLOSOMA_FORMAL_GIT_VERIFICATION_PATH=${VERIFY_ROOT}/node_0.json
export OMOMO_DATA_DIR=${MOTION_DIR} OMOMO_OBJECT_MAP=${OBJECT_SPEC_PATH}
export MOTION_DIR OBJECT_URDF=${OBJECT_SPEC_PATH} CONTACT_EXPORT_ROOT=${CONTACT_ROOT} AS_CONTACT_EXPORT_ROOT=${CONTACT_ROOT}
export AS_SINGLE_SLOT_MOTION_BASE=${MOTION_DIR}/_single_slot_motion_bank
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST=${SINGLE_SLOT_SOURCE_DIGEST}
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST=${SINGLE_SLOT_VIEW_DIGEST}
export HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST=${RANK_SHARD_SOURCE_DIGEST}
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_DIR=${SINGLE_SLOT_DIR}
export HOLOSOMA_EXTERNAL_AS_WORLD_SIZE=${WORLD_SIZE}
export HOLOSOMA_RANK_LOCAL_MOTION_ROOT=${RANK_SHARD_DIR}
export HOLOSOMA_MOTION_SHARD_MANIFEST=${RANK_SHARD_DIR}/manifest.json
export HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED=1 HOLOSOMA_REQUIRE_RANK_LOCAL_SHARD_PROVENANCE=1
export OMOMO_EXPECTED_TOTAL=137 RESUME_FROM_BOX_EXPECTED_TOTAL=137 AS_SUCCESS133_FINAL0P5=0 AS_RANK_LOCAL_SHARDS=1
export SOLID_ALLOWED_OBJECT_CATEGORIES='["box","ball","barrel","bin"]'
export CONTACT_SIDECAR_MODE=full-sidecars REQUIRE_MOTION_GENERATOR_TEACHER_MATCH=1
export ALLOW_PARTIAL_CONTACT_SIDECARS=1
export HOLOSOMA_OBJECT_SPAWN_MODE=single_slot_multi_urdf HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS=1
export HOLOSOMA_REQUIRE_OBJECT_MESH_ASSETS=1 HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE=mesh
export HOLOSOMA_OBJECT_COLLIDER_TYPE=convex_decomposition HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=0
export HOLOSOMA_REQUIRE_CONTACT_INTERVAL_COVERAGE=0 HOLOSOMA_REQUIRE_CONTACT_TARGET_COVERAGE=0
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=0 HOLOSOMA_PERCEPTION_INCLUDE_ROBOT_MESH=1
export HOLOSOMA_MOTION_METRICS_INTERVAL=16 HOLOSOMA_SKIP_INITIAL_CHECKPOINT=1
export HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES=True

export DISTILL_EXPERIMENT_CONFIG=g1-29dof-wbt-w-object-distill-sparse-root-cmd-teacher-linvel
export EXP=${DISTILL_EXPERIMENT_CONFIG}
export DISTILL_REWARD_CONFIG=${DISTILL_REWARD_CONFIG_VALUE}
export ENABLE_OFFLINE_CONTACT_GUIDANCE=${ENABLE_OFFLINE_CONTACT_GUIDANCE_VALUE}
export TEACHER_CHECKPOINT=${TEACHER} TEACHER_CHECKPOINT_EXPECTED_SHA256=${TEACHER_SHA256}
export DEFAULT_AS_TEACHER_CHECKPOINT=${TEACHER}
export MOTION_GENERATOR_TEACHER_EXPECTED_SHA256=${TEACHER_SHA256}
export EXPORT_ONNX=True
export RESUME_FROM_BOX=0 RESUME_FROM_PREVIOUS=0 WANDB_RESUME_SAME_RUN=0
unset RESUME_TRAINING_CKPT RESUME_CHECKPOINT RESUME_CKPT POLICY_INIT_CKPT POLICY_INIT_CHECKPOINT
export CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE=precomputed_turn_then_forward
export ZERO_ROOT_COMMAND_WHEN_DROP_ACTIVE=True CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION=True CAMERA_PITCH_DEG=0
export STUDENT_ACTOR_INPUTS="['actor_obs_root_contact_aware','actor_obs_drop_button','actor_obs_proprio_with_actions_no_linvel']"
export STUDENT_ACTOR_HIDDEN_DIMS='[512,256,128]' STUDENT_POLICY_TYPE=mlp
export PER_GPU_ENVS=${ENVIRONMENTS_PER_RANK} MIN_PER_GPU_ENVS=${ENVIRONMENTS_PER_RANK} TOTAL_NUM_ENVS=${TOTAL_ENVIRONMENTS}
export TRAINING_SEED=42 NUM_LEARNING_ITERATIONS=${TARGET_ITERATIONS} TARGET_LEARNING_ITERATION=${TARGET_ITERATIONS}
export SAVE_INTERVAL NUM_MINI_BATCHES=4 NUM_LEARNING_EPOCHS=7
export PPO_START_EPOCH=0 DAGGER_END_EPOCH=4000 PPO_START_COEFF=${PPO_START} PPO_TARGET_COEFF=${PPO_TARGET} PPO_SCHEDULE_STEP_EPOCHS=500
if [[ -n ${PPO_START_NOISE_STD_VALUE} ]]; then
  export PPO_START_NOISE_STD=${PPO_START_NOISE_STD_VALUE}
else
  unset PPO_START_NOISE_STD
fi
export PPO_START_NOISE_STD_FORCE_NONE=${PPO_START_NOISE_STD_FORCE_NONE_VALUE:-False}
export PPO_START_NOISE_STD_UNTIL_COEFF=0.1
export DAGGER_LOSS_COEF=1.0 DAGGER_MATCH_STD=${DAGGER_MATCH_STD_VALUE} DAGGER_REPLAY_ENABLED=False
export TEACHER_ACTION_MIX_RATIO=0 DAGGER_IGNORE_ZERO_TEACHER_ACTIONS=True
export FIXED_BC_EVAL_LOG_INTERVAL=100 FIXED_BC_GUARD_ENABLED=False
export START_AT_TIMESTEP_ZERO_PROB=0.2 START_AT_TIMESTEP_ZERO_PROB_END=${START_ZERO_END}
export START_AT_TIMESTEP_ZERO_PROB_START_ITER=${START_ZERO_START_ITER} START_AT_TIMESTEP_ZERO_PROB_END_ITER=${START_ZERO_END_ITER}
export UNIFORM_T1_WINDOW_SAMPLING_ENABLED=True UNIFORM_T1_WINDOW_HALF_WIDTH_STEPS=50
export UNIFORM_T1_WINDOW_DENSITY_BOOST=7.0
unset UNIFORM_T1_WINDOW_TARGET_SAMPLE_FRAC
export FREEZE_AT_TIMESTEP_ZERO_PROB=0.0 FREEZE_AT_TIMESTEP_ZERO_PROB_END=0.0
export FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=0 FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=${CURRICULUM_END_ITER}
export PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=402653184 PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY=402653184
export PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=134217728 PHYSX_GPU_COLLISION_STACK_SIZE=268435456
export RUN_NAME TRAINING_NAME=${RUN_NAME} TRAINING_PROJECT=carry-any LOGGER_BASE_DIR PRINT_TRAIN_CMD=1
export SCHEDULE_NAME=${SCHEDULE_NAME_VALUE}_${CONTACT_PROFILE}
export SCHEDULE_NOTES="Fresh ${PARENT_TEACHER_RUN}/${TEACHER_MODEL_NAME} frozen label-teacher distillation on its own exact137 simulator-rollout command bank; ${SCHEDULE_SUMMARY}; ${CONTACT_SUMMARY}; teacher arm=${TEACHER_ARM}; encoder=${ENCODER_TYPE}; all other fields fixed across A/B."
export STUDENT_TERMINATION_PROFILE_OVERRIDE=${STUDENT_TERMINATION_PROFILE_VALUE}
export BAD_TRACKING_REF_POS_THRESHOLD_OVERRIDE=${BAD_REF_POS}
export BAD_TRACKING_REF_ORI_THRESHOLD_OVERRIDE=${BAD_REF_ORI}
export BAD_TRACKING_BODY_POS_THRESHOLD_OVERRIDE=${BAD_BODY_POS}
export BAD_TRACKING_OBJECT_POS_THRESHOLD_OVERRIDE=${BAD_OBJECT_POS}
export BAD_TRACKING_OBJECT_ORI_THRESHOLD_OVERRIDE=${BAD_OBJECT_ORI}

if [[ ${MODE} == formal ]]; then
  unset WANDB_DISABLED
  export WANDB_MODE=online WANDB_CONSOLE=off WANDB_ENTITY=zihanw22 WANDB_RESUME_MODE=must HOLOSOMA_REQUIRE_WANDB_RUN=1
else
  export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_CONSOLE=off HOLOSOMA_REQUIRE_WANDB_RUN=0
fi

EXTRA_ARGS=(
  "reward:${DISTILL_REWARD_CONFIG}"
  randomization:g1_29dof_wbt_w_object_with_action_delay
  --training.export-onnx=True
  --algo.config.reset-rollout-at-checkpoint=False
  --algo.config.num-steps-per-env=24
  --simulator.config.sim.max-episode-length-s=10.0
  --command.setup-terms.motion-command.params.motion-config.clip-weighting-strategy=uniform_clip
  --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler=False
  --command.setup-terms.motion-command.params.motion-config.contact-aware-button-window-mode=kinematic_lift
  --command.setup-terms.motion-command.params.motion-config.contact-aware-carry-window-mode=peak_height
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-prepend=True
  --command.setup-terms.motion-command.params.motion-config.default-pose-prepend-duration-s=0.2
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-append=True
  --command.setup-terms.motion-command.params.motion-config.default-pose-append-duration-s=2.0
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale=1.0
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.dof-pos=0.20
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.dof-vel=0.35
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.root-pos='[0.08,0.08,0.025]'
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.root-rot='[0.15,0.15,0.30]'
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.root-lin-vel='[0.20,0.20,0.10]'
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.root-ang-vel='[0.25,0.25,0.35]'
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.object-pos='[0.08,0.08,0.0]'
  --randomization.setup-terms.setup-dof-pos-bias.params.dof-pos-bias-range='[-0.01,0.01]'
  --randomization.setup-terms.setup-dof-pos-bias.params.enabled=True
  --randomization.setup-terms.actuator-randomizer-state.params.kp-range='[0.9,1.1]'
  --randomization.setup-terms.actuator-randomizer-state.params.kd-range='[0.9,1.1]'
  --randomization.setup-terms.actuator-randomizer-state.params.enable-pd-gain=True
  --randomization.setup-terms.setup-action-delay-buffers.params.ctrl-delay-step-range='[0,1]'
  --randomization.setup-terms.setup-action-delay-buffers.params.enabled=True
  --randomization.setup-terms.setup-torque-rfi.params.enabled=True
  --randomization.setup-terms.setup-torque-rfi.params.rfi-lim=0.01
  --perception.sensor-offset='[0.0576235,0.01753,0.42987]'
  --perception.camera-mount-quat='[0.0,0.40354529635239006,0.0,0.9149596678498247]'
  --perception.camera-frame-quat='[-0.5,0.5,-0.5,0.5]'
  --perception.encoder-type="${ENCODER_TYPE}"
  --reward.terms.offline-contact-guidance.weight=0.0
  --reward.terms.offline-contact-guidance.params.contact-weight=10.0
  --reward.terms.offline-contact-guidance.params.wrist-weight=5.0
  --reward.terms.offline-contact-guidance.params.force-threshold=1.0
  --reward.terms.offline-contact-guidance.params.position-sigma=0.08
  --reward.terms.offline-contact-guidance.params.force-sigma=10.0
  --reward.terms.body-contact-reward-arms.weight=0.0
  --reward.terms.body-contact-reward-palms.weight=0.0
  --reward.terms.body-contact-reward-torso.weight=0.0
  --reward.terms.body-contact-reward-left-wrist-yaw.weight=0.0
  --reward.terms.body-contact-reward-right-wrist-yaw.weight=0.0
)
if [[ ${CONTACT_PROFILE} == no_contact ]]; then
  # When guidance is disabled the generic launcher only emits the zero weight;
  # preserve the exact immutable sidecar path and region metadata explicitly.
  EXTRA_ARGS+=(
    --reward.terms.offline-contact-guidance.params.contact-export-root "${CONTACT_ROOT}"
    --reward.terms.offline-contact-guidance.params.contact-region-names='["left_wrist","right_wrist","left_elbow","right_elbow","left_wrist_roll","right_wrist_roll","left_wrist_pitch","right_wrist_pitch","torso"]'
    --reward.terms.offline-contact-guidance.params.wrist-region-names='["left_wrist","right_wrist"]'
  )
fi
if [[ ${MODE} == formal ]]; then EXTRA_ARGS+=(--logger.id="${RUN_ID}" --logger.resume=must); fi

if [[ ${PREFLIGHT_ONLY:-0} == 1 ]]; then
  echo "[INFO] worker_preflight_ok mode=${MODE} teacher_arm=${TEACHER_ARM} teacher=${TEACHER_SHA256} clips=137 encoder=${ENCODER_TYPE} actor_scalar=94 actor_total=126 ppo=${PPO_START}->${PPO_TARGET} termination=${STUDENT_TERMINATION_PROFILE_VALUE} command=precomputed_turn_then_forward button=kinematic_lift sampling=uniform_clip_plus_uniform_t1_boost7_no_adaptive_failure_sampler contact_profile=${CONTACT_PROFILE} positive_contact_reward=${POSITIVE_CONTACT_REWARD_VALUE} export_onnx=true"
  exit 0
fi
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d') ]]; then
  echo "[ERROR] GPU apps appeared after preflight" >&2; exit 2
fi
cd "${SOURCE_ROOT}"
echo "[INFO] launch mode=${MODE} teacher_arm=${TEACHER_ARM} encoder=${ENCODER_TYPE} teacher_sha=${TEACHER_SHA256} ppo_schedule=${PPO_START}_to_${PPO_TARGET} termination=${STUDENT_TERMINATION_PROFILE_VALUE} command=precomputed_turn_then_forward contact_profile=${CONTACT_PROFILE} positive_contact_reward=${POSITIVE_CONTACT_REWARD_VALUE} export_onnx=true"
exec bash "${SOURCE_ROOT}/distill_as_button.sh" "${TEACHER}" "${EXTRA_ARGS[@]}"
