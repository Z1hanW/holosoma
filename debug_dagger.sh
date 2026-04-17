#!/usr/bin/env bash
set -euo pipefail

# Debug the distill_box_perception.sh DAgger path on one sequence: box_10.
#
# Pipeline:
# 1. Build a temporary one-clip motion bank with only box_10.
# 2. Export teacher rollout/contact artifacts for that clip with 2 envs.
# 3. Run the student DAgger job with 2 envs and live loss logging.
#
# Losses to inspect:
# - TensorBoard: Loss/bc_loss, Loss/distill_loss, Eval/fixed_bc_mu_mse
# - Console: the same loss keys are printed each learning iteration.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-python}
BOX_CLIP_ID=${BOX_CLIP_ID:-box_10}
NUM_ENVS=${NUM_ENVS:-2}
NPROC=${NPROC:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DEFAULT_TEACHER_CHECKPOINT=${DEFAULT_TEACHER_CHECKPOINT:-"wandb://zihanw22/boxer/u5lguxvl/model_17000.pt"}
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${DEFAULT_TEACHER_CHECKPOINT}}"
if [[ $# -gt 0 ]]; then
  case "$1" in
    wandb://*|https://wandb.ai/*|/*|./*|../*|*.pt)
      TEACHER_CHECKPOINT="$1"
      shift
      ;;
  esac
fi

HSSIM_BIN_DIR=${HSSIM_BIN_DIR:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin}
if [[ -d "${HSSIM_BIN_DIR}" ]]; then
  export PATH="${HSSIM_BIN_DIR}:${PATH}"
fi

DS_DATA_ROOT=${DS_DATA_ROOT:-"${SCRIPT_DIR}/data/ds_box_data"}
SOURCE_MOTION_DIR=${SOURCE_MOTION_DIR:-"${DS_DATA_ROOT}/train_g1_w_obj_prepared"}
DEBUG_ROOT=${DEBUG_ROOT:-"${SCRIPT_DIR}/outputs/debug_dagger/${BOX_CLIP_ID}"}
mkdir -p "${DEBUG_ROOT}"
DEBUG_ROOT=$(cd "${DEBUG_ROOT}" && pwd)
INPUT_MOTION_BANK=${INPUT_MOTION_BANK:-"${DEBUG_ROOT}/input_motion_bank"}
TEACHER_OUTPUT_DIR=${TEACHER_OUTPUT_DIR:-"${DEBUG_ROOT}/teacher_rollout"}
TEACHER_CLIPS_ROOT="${TEACHER_OUTPUT_DIR}/clips"
STUDENT_MOTION_DIR="${TEACHER_OUTPUT_DIR}/motion_bank"
TRAINING_LOG_BASE_DIR=${TRAINING_LOG_BASE_DIR:-"${DEBUG_ROOT}/logs"}

RUN_TEACHER_ROLLOUT=${RUN_TEACHER_ROLLOUT:-1}
FORCE_TEACHER_ROLLOUT=${FORCE_TEACHER_ROLLOUT:-0}
DEBUG_DAGGER_DRY_RUN=${DEBUG_DAGGER_DRY_RUN:-${DRY_RUN:-0}}

NUM_LEARNING_ITERATIONS=${NUM_LEARNING_ITERATIONS:-10}
PPO_START_EPOCH=${PPO_START_EPOCH:-$((NUM_LEARNING_ITERATIONS + 1))}
DAGGER_END_EPOCH=${DAGGER_END_EPOCH:-$((NUM_LEARNING_ITERATIONS + 2))}
PPO_TARGET_COEFF=${PPO_TARGET_COEFF:-0.0}
PPO_START_COEFF=${PPO_START_COEFF:-0.0}
PPO_SCHEDULE_STEP_EPOCHS=${PPO_SCHEDULE_STEP_EPOCHS:-0}
FIXED_BC_EVAL_NUM_SAMPLES=${FIXED_BC_EVAL_NUM_SAMPLES:-2}
FIXED_BC_EVAL_LOG_INTERVAL=${FIXED_BC_EVAL_LOG_INTERVAL:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
MAX_EPISODE_LENGTH_S=${MAX_EPISODE_LENGTH_S:-8.0}
MAX_ROLLOUT_STEPS=${MAX_ROLLOUT_STEPS:-}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.2}
RESET_NOISE_SCALE=${RESET_NOISE_SCALE:-1.0}
USE_ADAPTIVE_TIMESTEPS_SAMPLER=${USE_ADAPTIVE_TIMESTEPS_SAMPLER:-True}
ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT=${ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT:-"${TEACHER_CLIPS_ROOT}"}

LOGGER=${LOGGER:-logger:disabled}
TRAINING_PROJECT=${TRAINING_PROJECT:-boxer_debug}
RUN_NAME=${RUN_NAME:-debug_dagger_${BOX_CLIP_ID}}
TRAINING_NAME=${TRAINING_NAME:-debug_dagger_${BOX_CLIP_ID}_student_loss}

ENABLE_VISER=${ENABLE_VISER:-True}
VISER_PORT=${VISER_PORT:-18090}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_ENV_COUNT=${VISER_ENV_COUNT:-2}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-15}
VISER_DISTILL_MINIMAL_UI=${VISER_DISTILL_MINIMAL_UI:-1}
VISER_SHOW_TARGET_KEYPOINTS=${VISER_SHOW_TARGET_KEYPOINTS:-1}
VISER_SHOW_ENV_SEQUENCE_LABELS=${VISER_SHOW_ENV_SEQUENCE_LABELS:-1}

START_TENSORBOARD=${START_TENSORBOARD:-1}
TENSORBOARD_PORT=${TENSORBOARD_PORT:-6007}

is_truthy() {
  case "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

prepare_single_clip_bank() {
  local source_dir="$1"
  local target_dir="$2"
  local clip_id="$3"

  "${PYTHON_BIN}" - "${source_dir}" "${target_dir}" "${clip_id}" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

source_dir = Path(sys.argv[1]).expanduser().resolve()
target_dir = Path(sys.argv[2]).expanduser().resolve()
clip_id = sys.argv[3]

source_npz = source_dir / f"{clip_id}.npz"
if not source_npz.is_file():
    raise SystemExit(f"[ERROR] Source clip not found: {source_npz}")

target_dir.mkdir(parents=True, exist_ok=True)
for existing in target_dir.glob("*.npz"):
    existing.unlink()

target_npz = target_dir / source_npz.name
if target_npz.exists() or target_npz.is_symlink():
    target_npz.unlink()
os.symlink(source_npz, target_npz)

clip_map: dict[str, object] = {}
metadata_uses_clips_key = True
for candidate in (source_dir / "_clip_object_urdf_map.json", source_dir / "clip_object_urdf_map.json"):
    if not candidate.is_file():
        continue
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
        clip_map = payload["clips"]
        metadata_uses_clips_key = True
    elif isinstance(payload, dict):
        clip_map = payload
        metadata_uses_clips_key = False
    break

entry = clip_map.get(clip_id) if isinstance(clip_map, dict) else None
if entry is None:
    with np.load(source_npz, allow_pickle=True) as data:
        object_urdf_path = ""
        if "object_urdf_path" in data:
            raw = np.asarray(data["object_urdf_path"])
            if raw.size:
                item = raw.item() if raw.shape == () else raw.reshape(-1)[0]
                object_urdf_path = str(item).strip()
        if not object_urdf_path:
            raise SystemExit(f"[ERROR] Missing object_urdf_path for {clip_id}")
        entry = {"object_name": clip_id, "object_urdf_path": object_urdf_path}

output = {"clips": {clip_id: entry}} if metadata_uses_clips_key else {clip_id: entry}
(target_dir / "_clip_object_urdf_map.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(target_dir)
PY
}

verify_teacher_artifacts() {
  local clips_root="$1"
  local motion_bank="$2"
  local clip_id="$3"

  "${PYTHON_BIN}" - "${clips_root}" "${motion_bank}" "${clip_id}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

clips_root = Path(sys.argv[1]).expanduser().resolve()
motion_bank = Path(sys.argv[2]).expanduser().resolve()
clip_id = sys.argv[3]

if not clips_root.is_dir():
    raise SystemExit(f"[ERROR] Teacher clips root not found: {clips_root}")
clip_dirs = [p for p in clips_root.iterdir() if p.is_dir() and p.name.endswith(f"_{clip_id}")]
if not clip_dirs:
    raise SystemExit(f"[ERROR] No teacher clip dir for {clip_id} under {clips_root}")
clip_dir = sorted(clip_dirs)[0]
required = [
    clip_dir / "teacher_rollout_reference.npz",
    clip_dir / "left_wrist_contact_interval_steps.npy",
    clip_dir / "right_wrist_contact_interval_steps.npy",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("[ERROR] Missing teacher rollout artifacts:\n" + "\n".join(missing))
if not (motion_bank / f"{clip_id}.npz").is_file():
    raise SystemExit(f"[ERROR] Teacher rollout motion bank missing {clip_id}.npz: {motion_bank}")
if not (motion_bank / "_clip_object_urdf_map.json").is_file():
    raise SystemExit(f"[ERROR] Teacher rollout motion bank missing _clip_object_urdf_map.json: {motion_bank}")
print(clip_dir)
PY
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

mkdir -p "${DEBUG_ROOT}"
prepare_single_clip_bank "${SOURCE_MOTION_DIR}" "${INPUT_MOTION_BANK}" "${BOX_CLIP_ID}" >/dev/null

echo "[INFO] box_clip_id=${BOX_CLIP_ID}"
echo "[INFO] input_motion_bank=${INPUT_MOTION_BANK}"
echo "[INFO] teacher_output_dir=${TEACHER_OUTPUT_DIR}"
echo "[INFO] student_motion_dir=${STUDENT_MOTION_DIR}"
echo "[INFO] num_envs=${NUM_ENVS} nproc=${NPROC} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] teacher_checkpoint=${TEACHER_CHECKPOINT}"

teacher_cmd=(
  env
  DATA_MODE=pure-sd
  MOTION_DIR="${INPUT_MOTION_BANK}"
  OBJECT_URDF="${INPUT_MOTION_BANK}/_clip_object_urdf_map.json"
  NUM_ENVS="${NUM_ENVS}"
  HEADLESS=True
  OUTPUT_DIR="${TEACHER_OUTPUT_DIR}"
  START_AT_TIMESTEP_ZERO_PROB=1.0
  FREEZE_AT_TIMESTEP_ZERO_PROB=0.0
  RESET_NOISE_SCALE=0.0
  USE_ADAPTIVE_TIMESTEPS_SAMPLER=False
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  bash "${SCRIPT_DIR}/infer_teacher_box_contacts.sh" "${TEACHER_CHECKPOINT}"
)
if [[ -n "${MAX_ROLLOUT_STEPS}" ]]; then
  teacher_cmd+=(--max-rollout-steps "${MAX_ROLLOUT_STEPS}")
fi

if is_truthy "${DEBUG_DAGGER_DRY_RUN}"; then
  echo "[DRY_RUN] teacher rollout command:"
  print_command "${teacher_cmd[@]}"
else
  if is_truthy "${RUN_TEACHER_ROLLOUT}"; then
    if is_truthy "${FORCE_TEACHER_ROLLOUT}"; then
      case "$(cd "$(dirname "${TEACHER_OUTPUT_DIR}")" && pwd)/$(basename "${TEACHER_OUTPUT_DIR}")" in
        "${DEBUG_ROOT}"/*)
          rm -rf "${TEACHER_OUTPUT_DIR}"
          ;;
        *)
          echo "[ERROR] Refusing to delete TEACHER_OUTPUT_DIR outside DEBUG_ROOT: ${TEACHER_OUTPUT_DIR}" >&2
          exit 2
          ;;
      esac
    fi
    if [[ ! -f "${STUDENT_MOTION_DIR}/${BOX_CLIP_ID}.npz" ]]; then
      echo "[INFO] Exporting teacher rollout/contact artifacts..."
      "${teacher_cmd[@]}"
    else
      echo "[INFO] Teacher rollout motion bank already exists; set FORCE_TEACHER_ROLLOUT=1 to regenerate."
    fi
  fi
  TEACHER_CLIP_DIR="$(verify_teacher_artifacts "${TEACHER_CLIPS_ROOT}" "${STUDENT_MOTION_DIR}" "${BOX_CLIP_ID}")"
  echo "[INFO] teacher_clip_dir=${TEACHER_CLIP_DIR}"
fi

if ! is_truthy "${DEBUG_DAGGER_DRY_RUN}" && is_truthy "${START_TENSORBOARD}" && command -v tensorboard >/dev/null 2>&1; then
  mkdir -p "${TRAINING_LOG_BASE_DIR}/${TRAINING_PROJECT}"
  nohup tensorboard \
    --logdir "${TRAINING_LOG_BASE_DIR}/${TRAINING_PROJECT}" \
    --host 0.0.0.0 \
    --port "${TENSORBOARD_PORT}" \
    > "${DEBUG_ROOT}/tensorboard.log" 2>&1 &
  echo "[INFO] tensorboard_url=http://127.0.0.1:${TENSORBOARD_PORT}"
  echo "[INFO] tensorboard_log=${DEBUG_ROOT}/tensorboard.log"
elif ! is_truthy "${DEBUG_DAGGER_DRY_RUN}" && is_truthy "${START_TENSORBOARD}"; then
  echo "[WARN] tensorboard command not found; student losses will still print in the training console."
fi

rollout_ref_terms=(
  teacher_rollout_global_ref_position_error_exp
  teacher_rollout_global_ref_orientation_error_exp
  teacher_rollout_relative_body_position_error_exp
  teacher_rollout_relative_body_orientation_error_exp
  teacher_rollout_global_body_lin_vel
  teacher_rollout_global_body_ang_vel
  teacher_rollout_object_global_ref_position_error_exp
  teacher_rollout_object_global_ref_orientation_error_exp
)
reward_root_args=()
for term in "${rollout_ref_terms[@]}"; do
  reward_root_args+=("--reward.terms.${term}.params.rollout_reference_root=${TEACHER_CLIPS_ROOT}")
done
reward_root_args+=(
  "--reward.terms.offline_wrist_target_guidance.params.contact_export_root=${TEACHER_CLIPS_ROOT}"
  "--reward.terms.offline_contact_guidance.params.contact_export_root=${TEACHER_CLIPS_ROOT}"
)

student_env=(
  EXP=g1-29dof-wbt-w-object-distill-sparse-root-cmd-r2s-rollout-ref
  DATA_MODE=pure-sd
  TRACKER_PROFILE=old-tracker
  SCHEDULE_VARIANT=default
  RUN_NAME="${RUN_NAME}"
  TRAINING_NAME="${TRAINING_NAME}"
  TRAINING_PROJECT="${TRAINING_PROJECT}"
  LOGGER="${LOGGER}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  NPROC="${NPROC}"
  NUM_ENVS="${NUM_ENVS}"
  MOTION_DIR="${STUDENT_MOTION_DIR}"
  NUM_LEARNING_ITERATIONS="${NUM_LEARNING_ITERATIONS}"
  SAVE_INTERVAL="${SAVE_INTERVAL}"
  PPO_START_EPOCH="${PPO_START_EPOCH}"
  DAGGER_END_EPOCH="${DAGGER_END_EPOCH}"
  PPO_TARGET_COEFF="${PPO_TARGET_COEFF}"
  PPO_START_COEFF="${PPO_START_COEFF}"
  PPO_SCHEDULE_STEP_EPOCHS="${PPO_SCHEDULE_STEP_EPOCHS}"
  TEACHER_ACTION_MIX_RATIO=0.0
  FIXED_BC_EVAL_LOG_INTERVAL="${FIXED_BC_EVAL_LOG_INTERVAL}"
  START_AT_TIMESTEP_ZERO_PROB="${START_AT_TIMESTEP_ZERO_PROB}"
  RESET_NOISE_SCALE="${RESET_NOISE_SCALE}"
  MAX_EPISODE_LENGTH_S="${MAX_EPISODE_LENGTH_S}"
  USE_ADAPTIVE_TIMESTEPS_SAMPLER="${USE_ADAPTIVE_TIMESTEPS_SAMPLER}"
  ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT="${ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT}"
  VISER_DISTILL_MINIMAL_UI="${VISER_DISTILL_MINIMAL_UI}"
  VISER_SHOW_TARGET_KEYPOINTS="${VISER_SHOW_TARGET_KEYPOINTS}"
  VISER_SHOW_ENV_SEQUENCE_LABELS="${VISER_SHOW_ENV_SEQUENCE_LABELS}"
)

student_cmd=(
  env
  "${student_env[@]}"
  bash "${SCRIPT_DIR}/distill_box_perception.sh" "${TEACHER_CHECKPOINT}"
  "${RUN_NAME}"
  "${reward_root_args[@]}"
  "--logger.base-dir=${TRAINING_LOG_BASE_DIR}"
  "--algo.config.num-mini-batches=1"
  "--algo.config.num-learning-epochs=1"
  "--algo.config.distill.fixed-bc-eval-num-samples=${FIXED_BC_EVAL_NUM_SAMPLES}"
  "--training.enable-viser=${ENABLE_VISER}"
  "--training.viser-port=${VISER_PORT}"
  "--training.viser-env-id=${VISER_ENV_ID}"
  "--training.viser-env-count=${VISER_ENV_COUNT}"
  "--training.viser-update-hz=${VISER_UPDATE_HZ}"
  "$@"
)

echo "[INFO] Starting student DAgger. Watch console Loss/bc_loss and TensorBoard Eval/fixed_bc_mu_mse."
echo "[INFO] viser_url=http://127.0.0.1:${VISER_PORT}"
if is_truthy "${DEBUG_DAGGER_DRY_RUN}"; then
  echo "[DRY_RUN] student DAgger command:"
  print_command "${student_cmd[@]}"
  exit 0
fi

"${student_cmd[@]}"
