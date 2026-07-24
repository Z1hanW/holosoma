#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 2 || ${1:-0} != +([0-9]) || ${2:-34100} != +([0-9]) ]]; then
  echo "usage: $0 [GPU_INDEX=0] [VISER_PORT=34100]" >&2
  exit 2
fi

readonly GPU_INDEX="${1:-0}"
readonly SERVER_PORT="${2:-34100}"
if (( GPU_INDEX < 0 || GPU_INDEX > 7 || SERVER_PORT < 1024 || SERVER_PORT > 65535 )); then
  echo "[ERROR] GPU_INDEX must be 0..7 and VISER_PORT must be 1024..65535" >&2
  exit 2
fi

readonly EVAL_ROOT=/home/ubuntu/FAR/holosoma_runs/formal_teacher_rl_model2k_eval3_20260720
readonly RUN_ROOT=/home/ubuntu/FAR/holosoma_runs/formal_teacher_rl_model2k_viser_20260720
readonly SOURCE_ID=src-a8879814e247e78b0e686fb8bdf94b985d008b2960aa187b482ef8de7b644399
readonly SOURCE_DIGEST=a8879814e247e78b0e686fb8bdf94b985d008b2960aa187b482ef8de7b644399
readonly SOURCE_ROOT="${EVAL_ROOT}/src_ball"
readonly PYTHON_BIN=/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3.11
readonly PYTHON_RUNTIME=/home/ubuntu/FAR/holosoma_runs/.runtime/python/python-runtime-v2-2dcddba9d88fd6b5a3fe701e2b74c59eae988071e52d8bdb44a2474bc16b9e1f/site-packages
readonly CHECKPOINT="${EVAL_ROOT}/checkpoint_cache/zihanw22_carry-any_crjedc0u/by-sha256/4e4f8739a332abfd49c23760e07c8413d56441793eaa07167ef80a8762a8ba19.pt"
readonly CHECKPOINT_SHA256=4e4f8739a332abfd49c23760e07c8413d56441793eaa07167ef80a8762a8ba19
readonly BANK=/home/ubuntu/FAR/holosoma/data/ds_as_data/teacher_ab3_true_rollout_u8_solid_sharedphys__src_5ae3055eeff0098dad099ec26d721b76ac7675207e7d41a6ee3aaa2a15080b49/_scientific_teacher_ab_single_slot/by-source/25d7623ec5d785ee12595322f5939afe4655ea8472ed14fc9b1af5cf2c915f83
readonly OBJECT_MAP="${BANK}/_clip_object_urdf_map.json"

check_sha() {
  local expected="$1"
  local path="$2"
  [[ -f "${path}" ]] || { echo "[ERROR] missing integrity input: ${path}" >&2; exit 2; }
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || {
    echo "[ERROR] SHA mismatch: ${path}" >&2
    exit 2
  }
}

cd "${SOURCE_ROOT}"
sha256sum --quiet -c .holosoma_snapshot/source_manifest.sha256
check_sha "${CHECKPOINT_SHA256}" "${CHECKPOINT}"
check_sha 382e0aaffc8a6e4dd4c1906eaed50c5ed3e244bdd3e769e5581f374e60f06126 "${BANK}/scaledown__any_barrel_25.npz"
check_sha 87644e984e4af1e7b75f3e3f83d822a0ee18e2b7604968d38f16d5b80cae46bb "${BANK}/unscale__any_ball_29.npz"
check_sha 9c981f20edb97a9d598fee7beb15c42278ad4bb0bc1725812540fd243f35adb4 "${BANK}/unscale__any_bin_29.npz"
check_sha 4826a9c053de18475ec59e61f1378734b51f1906e1f8cd88bf1592a5b17509d1 "${OBJECT_MAP}"
check_sha 0a69875d5a2d4ed19a62040a1ee326e81f68b49c934f4ed659d21b23828843e3 "${BANK}/_single_slot_urdfs/scaledown_any_barrel_25.urdf"
check_sha 2e9d3d7c47f5915415e2aefa0b76b7294a829bfb5bb74dc637466a40dc38d556 "${BANK}/_single_slot_urdfs/unscale_any_ball_29.urdf"
check_sha c2009cb217f8157bfd581bceffe1ad422d98722b425af8341057674118e1c385 "${BANK}/_single_slot_urdfs/unscale_any_bin_29.urdf"
check_sha 24d046ad6047fa8f33c63138c9be35975d5a0e078bc6df321b2068442d64f4c5 /home/ubuntu/FAR/holosoma/data/ds_as_data/prism_debug30_convexhull_allmesh/objects_convex_hull/object_mesh_yup_e7481f4304/object_mesh_yup_convex_hull.obj
check_sha 9734a65b4cd1127c96fad2b499832cbe5f5c7608200c593127c45db31b92d5b9 /home/ubuntu/FAR/holosoma/data/ds_as_data/prism_debug30_convexhull_allmesh/objects_convex_hull/object_mesh_yup_e6603064bd/object_mesh_yup_convex_hull.obj
check_sha daae95872696e55484f37a166978fca182303ce1bb73b26d851b0d085784890d /home/ubuntu/FAR/holosoma/data/ds_as_data/prism_debug30_convexhull_allmesh/objects_convex_hull/object_mesh_yup_266fbb26f5/object_mesh_yup_convex_hull.obj
[[ "$(find "${BANK}" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq 3 ]]

if ss -ltnH | awk '{print $4}' | rg -q "(:|])${SERVER_PORT}$"; then
  echo "[ERROR] Viser port ${SERVER_PORT} is already in use" >&2
  exit 2
fi

mkdir -p \
  "${RUN_ROOT}/cache/torch_extensions" \
  "${RUN_ROOT}/cache/warp" \
  "${RUN_ROOT}/cache/ov" \
  "${RUN_ROOT}/logs" \
  "${RUN_ROOT}/metrics"

export PATH="$(dirname "${PYTHON_BIN}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHON_BIN
export PYTHONPATH="${PYTHON_RUNTIME}:${SOURCE_ROOT}/src/holosoma:${SOURCE_ROOT}/src/holosoma_inference:${SOURCE_ROOT}/src"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export HOLOSOMA_ORIGINAL_LOCAL_RANK="${GPU_INDEX}"
export OMP_NUM_THREADS=1
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export WANDB_MODE=disabled
export WANDB_SILENT=true
export TORCH_EXTENSIONS_DIR="${RUN_ROOT}/cache/torch_extensions"
export WARP_CACHE_PATH="${RUN_ROOT}/cache/warp"
export OMNI_USER_CACHE_PATH="${RUN_ROOT}/cache/ov"

export HOLOSOMA_SOURCE_ROOT="${SOURCE_ROOT}"
export HOLOSOMA_SOURCE_SNAPSHOT_ID="${SOURCE_ID}"
export HOLOSOMA_SOURCE_MANIFEST_SHA256="${SOURCE_DIGEST}"
export HOLOSOMA_EVAL_POLICY=checkpoint_actor
export HOLOSOMA_EVAL_DISABLE_ROLLOUT_REFERENCE_REWARDS=1
export HOLOSOMA_EVAL_ALLOW_MISSING_CAMERA_SENSOR_NOISE_STATE=1
export HOLOSOMA_CLEAN_ROBOT_USD_CACHE=1
export HOLOSOMA_DISABLE_AUTO_RESET=1
export HOLOSOMA_DISABLE_MOTION_END_RESET=1
export HOLOSOMA_DISABLE_CLIP_END_RESET=1
export HOLOSOMA_RESET_TO_DEFAULT_POSE=0
export HOLOSOMA_EVAL_DEBUG_PATH="${RUN_ROOT}/metrics/checkpoint_actor_policy_io.jsonl"
export HOLOSOMA_EVAL_DEBUG_LIMIT=20

export INFER_DATASET=omomo
export MOTION_DIR="${BANK}"
export OBJECT_URDF="${OBJECT_MAP}"
export NUM_ENVS=1
export HEADLESS=1
export MOCAP_PERCEPTION_PRESET=checkpoint
export DEFAULT_DISTILL_PROPRIO_HISTORY_ONLY=0
export DISABLE_RANDOMIZATION=False
export START_AT_TIMESTEP_ZERO_PROB=1.0
export FREEZE_AT_TIMESTEP_ZERO_PROB=0.0
export RESET_NOISE_SCALE=0.0
export VISER_PORT="${SERVER_PORT}"
export VISER_ENV_ID=0
export VISER_UPDATE_HZ=30
export VISER_RECENTER=True
export VISER_ENABLE_MANUAL_GUI=1
export VISER_ENABLE_CLIP_GUI=1
export VISER_ENABLE_SIMULATION_CONTROL_GUI=1
export VISER_ENABLE_RESET_OBJECT_GUI=1
export VISER_MANUAL_CONTROL_DEFAULT=1
export VISER_FORCE_MANUAL_CONTROL=0
export VISER_MANUAL_USE_HW_JOYSTICK=0
export VISER_CLIP_LOCK_DEFAULT=1
export VISER_SHOW_TARGET_KEYPOINTS=0
export VISER_START_PAUSED=1
export LOGURU_LEVEL=INFO
export PY_LOG_LEVEL=WARNING

echo "[INFO] launching Viser teacher checkpoint_actor gpu=${GPU_INDEX} port=${SERVER_PORT}"
echo "[INFO] checkpoint=wandb://zihanw22/carry-any/crjedc0u/model_02000.pt sha256=${CHECKPOINT_SHA256}"
echo "[INFO] clips=scaledown__any_barrel_25,unscale__any_ball_29,unscale__any_bin_29 randomization=disabled num_envs=1 start_paused=1"

exec bash infer_box_joystick.sh mocap "${CHECKPOINT}" \
  --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler False \
  --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-sampling-enabled False \
  randomization:disabled \
  logger:disabled \
  --logger.video.enabled False
