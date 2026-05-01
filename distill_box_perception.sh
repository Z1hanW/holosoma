#!/usr/bin/env bash
set -euo pipefail

# Distill object-carry generalist -> sparse-goal pickup student with depth perception access.
#
# Student policy observation (actor):
# - actor_obs_root: root-frame relative command [dx, dy, dyaw]
# - actor_obs_proprio (base_lin_vel, base_ang_vel, dof_pos, dof_vel) + actor_obs_actions single-step action
# - perception_obs (camera depth)
# - No actor box state is used by student actor.
#
# Teacher policy observation:
# - auto-matched to the selected teacher checkpoint, following the same teacher
#   compatibility interface pattern as distill_box_drop_mixed.sh
#
# Schedule variants / modes:
# - default: old-tracker profile keeps pure DAgger by default
# - dagger_mix: pure DAgger with teacher-action rollout mix 0.7 -> 0.0 by default
# - dag_first: pure DAgger first, then a small PPO blend
# - ppo_first: PPO+DAgger from iteration 0
# - contact-aware: ppo-first student whose sparse root command is zeroed before
#   pickup and during the release/putdown tail
# - contact-aware-history: contact-aware plus 5-frame student/critic proprio history
# - shoo7sr1-near03-debug: shoo7sr1 debug reproduction; only depth near
#   differs from the saved shoo7sr1 config (0.3 instead of 0.1)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/scripts/object_generalist_ds_paths.sh"
source "${SCRIPT_DIR}/scripts/gpu_launch_defaults.sh"

USE_LEGACY_DS=${USE_LEGACY_DS:-auto}
LEGACY_DS_ROOT=${LEGACY_DS_ROOT:-"${SCRIPT_DIR}/data/ds_box_data_legacy"}
LEGACY_PREPARE_SCRIPT="${SCRIPT_DIR}/cp_legacy.sh"

DEFAULT_TEACHER_CHECKPOINT=${DEFAULT_TEACHER_CHECKPOINT:-"wandb://zihanw22/boxer/u5lguxvl/model_17000.pt"}
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${DEFAULT_TEACHER_CHECKPOINT}}"
POSITIONAL_RUN_NAME=""
DATA_MODE=${DATA_MODE:-pure-sd}
TRACKER_PROFILE=${TRACKER_PROFILE:-old-tracker}
SCHEDULE_VARIANT=${SCHEDULE_VARIANT:-default}
ROOT_COMMAND_MODE=${ROOT_COMMAND_MODE:-default}
CONTACT_AWARE_HISTORY=${CONTACT_AWARE_HISTORY:-0}
CONTACT_AWARE_HISTORY_LENGTH=${CONTACT_AWARE_HISTORY_LENGTH:-5}
SHOO7SR1_NEAR03_DEBUG=${SHOO7SR1_NEAR03_DEBUG:-0}
PYTHON_BIN=${PYTHON_BIN:-python}

parse_wandb_run_url() {
  local ref="$1"
  local clean_ref="${ref%%\?*}"
  if [[ "${clean_ref}" != https://wandb.ai/*/runs/* ]]; then
    return 1
  fi
  local trimmed="${clean_ref#https://wandb.ai/}"
  local entity=""
  local project=""
  local run_id=""
  local explicit_file=""
  IFS='/' read -r -a parts <<< "${trimmed}"
  if [[ "${#parts[@]}" -lt 4 || "${parts[2]}" != "runs" ]]; then
    return 1
  fi
  entity="${parts[0]}"
  project="${parts[1]}"
  run_id="${parts[3]}"
  if [[ -z "${entity}" || -z "${project}" || -z "${run_id}" ]]; then
    return 1
  fi
  if [[ "${#parts[@]}" -ge 6 && "${parts[4]}" == "files" ]]; then
    explicit_file="${trimmed#${entity}/${project}/runs/${run_id}/files/}"
  fi
  printf '%s\t%s\t%s\t%s\n' "${entity}" "${project}" "${run_id}" "${explicit_file}"
}

resolve_remote_wandb_checkpoint_name() {
  local entity="$1"
  local project="$2"
  local run_id="$3"
  "${PYTHON_BIN}" - "${entity}" "${project}" "${run_id}" <<'PY' 2>/dev/null || true
import re
import sys

try:
    import wandb
except Exception:
    sys.exit(0)

entity, project, run_id = sys.argv[1:4]
api = wandb.Api(timeout=30)
run = api.run(f"{entity}/{project}/{run_id}")

def _coerce_int(value):
    try:
        return int(value)
    except Exception:
        return None

summary = getattr(run, "summary", {}) or {}
step_hint = _coerce_int(summary.get("_step"))
if step_hint is None:
    step_hint = _coerce_int(summary.get("global_step"))
if step_hint is None:
    step_hint = _coerce_int(getattr(run, "lastHistoryStep", None))
if step_hint is None:
    step_hint = 0

save_interval = None
cfg = getattr(run, "config", {}) or {}
algo_cfg = cfg.get("algo") if isinstance(cfg, dict) else None
if isinstance(algo_cfg, dict):
    algo_cfg = algo_cfg.get("config")
if isinstance(algo_cfg, dict):
    save_interval = _coerce_int(algo_cfg.get("save_interval"))
if save_interval is None or save_interval <= 0:
    save_interval = 500

pattern = re.compile(r"^model_(\d+)\.pt$")
start_step = max(step_hint - (step_hint % save_interval), 0)
best_step = -1
best_name = ""
for step in range(start_step, -1, -save_interval):
    name = f"model_{step:05d}.pt"
    try:
        file_obj = run.file(name)
    except Exception:
        continue
    size = _coerce_int(getattr(file_obj, "size", None))
    match = pattern.match(getattr(file_obj, "name", ""))
    if match is None or size is None or size <= 0:
        continue
    file_step = int(match.group(1))
    if file_step >= best_step:
        best_step = file_step
        best_name = name
        break

if best_name:
    print(best_name)
PY
}

normalize_checkpoint_ref() {
  local ref="$1"
  if [[ "${ref}" != https://wandb.ai/*/runs/* ]]; then
    echo "${ref}"
    return 0
  fi

  local parsed=""
  local entity=""
  local project=""
  local run_id=""
  local explicit_file=""
  local model_file="${WANDB_MODEL_FILE:-}"

  parsed="$(parse_wandb_run_url "${ref}" || true)"
  if [[ -z "${parsed}" ]]; then
    echo "${ref}"
    return 0
  fi

  IFS=$'\t' read -r entity project run_id explicit_file <<< "${parsed}"
  if [[ -n "${explicit_file}" ]]; then
    model_file="${explicit_file}"
  elif [[ -z "${model_file}" ]]; then
    model_file="$(resolve_remote_wandb_checkpoint_name "${entity}" "${project}" "${run_id}")"
    if [[ -n "${model_file}" ]]; then
      echo "[INFO] Resolved wandb run URL to latest remote checkpoint: ${model_file}" >&2
    fi
  fi

  if [[ -z "${model_file}" ]]; then
    echo "[ERROR] Could not determine checkpoint for W&B run URL: ${ref}" >&2
    echo "[ERROR] Pass a /files/<checkpoint>.pt URL or set WANDB_MODEL_FILE." >&2
    return 2
  fi

  echo "wandb://${entity}/${project}/${run_id}/${model_file}"
}

is_checkpoint_ref() {
  local ref="$1"
  [[ "${ref}" == wandb://* || "${ref}" == https://wandb.ai/*/runs/* || "${ref}" == /* || "${ref}" == ./* || "${ref}" == ../* || "${ref}" == *.pt ]]
}

while [[ $# -gt 0 ]]; do
  first_arg_normalized=$(echo "$1" | tr '[:upper:]' '[:lower:]')
  case "${first_arg_normalized}" in
    mix-naive)
      DATA_MODE="mix-naive"
      shift
      ;;
    pure-real|pure-omomo)
      DATA_MODE="pure-real"
      shift
      ;;
    pure-sd|pure-ds)
      DATA_MODE="pure-sd"
      shift
      ;;
    default)
      SCHEDULE_VARIANT="default"
      shift
      ;;
    dagger_mix|dagger-mix|daggermix)
      SCHEDULE_VARIANT="dagger_mix"
      shift
      ;;
    dag_first|dag-first|dagger-first)
      SCHEDULE_VARIANT="dag_first"
      shift
      ;;
    ppo_first|ppo-first)
      SCHEDULE_VARIANT="ppo_first"
      shift
      ;;
    contact-aware|contact_aware|contactaware)
      ROOT_COMMAND_MODE="contact-aware"
      shift
      ;;
    contact-aware-history|contact_aware_history|contactaware-history|contactaware_history)
      ROOT_COMMAND_MODE="contact-aware"
      CONTACT_AWARE_HISTORY=1
      shift
      ;;
    shoo7sr1-near03-debug|shoo7sr1_near03_debug|shoo7sr1-debug|shoo7sr1_debug)
      SHOO7SR1_NEAR03_DEBUG=1
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [[ "${SHOO7SR1_NEAR03_DEBUG}" == "1" ]]; then
  DATA_MODE="pure-sd"
  TRACKER_PROFILE="old-tracker"
  SCHEDULE_VARIANT="ppo_first"
  ROOT_COMMAND_MODE="default"
  EXP="g1-29dof-wbt-w-object-distill-sparse-root-cmd-r2s-rollout-ref-shoo7sr1-debug"
fi

POSITIONAL_TEACHER_CHECKPOINT_SET=0
while [[ $# -gt 0 && "$1" != -* ]]; do
  if is_checkpoint_ref "$1"; then
    if [[ "${POSITIONAL_TEACHER_CHECKPOINT_SET}" -eq 0 ]]; then
      TEACHER_CHECKPOINT="$1"
      POSITIONAL_TEACHER_CHECKPOINT_SET=1
      shift
      continue
    fi
    break
  fi

  case "$1" in
    run:*|run-name:*|run_name:*|name:*)
      POSITIONAL_RUN_NAME="${1#*:}"
      shift
      continue
      ;;
  esac

  if [[ -z "${POSITIONAL_RUN_NAME}" ]]; then
    POSITIONAL_RUN_NAME="$1"
    shift
    continue
  fi
  break
done

if [[ -z "${TEACHER_CHECKPOINT}" ]]; then
  echo "Usage: $0 [mix-naive|pure-real|pure-sd] [default|dagger-mix|dag_first|ppo-first|contact-aware|contact-aware-history|shoo7sr1-near03-debug] [run_name] [teacher_checkpoint.pt|wandb_run_url] [extra train args...]" >&2
  exit 1
fi

STUDENT_ACTOR_INPUTS_EXPLICIT=0
[[ -n "${STUDENT_ACTOR_INPUTS+x}" ]] && STUDENT_ACTOR_INPUTS_EXPLICIT=1

DS_DATA_ROOT_EXPLICIT=0
[[ -n "${DS_DATA_ROOT+x}" ]] && DS_DATA_ROOT_EXPLICIT=1
LEGACY_DS_ENABLED=0
LEGACY_DS_PREPARED=0
case "$(echo "${USE_LEGACY_DS}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    LEGACY_DS_ENABLED=1
    ;;
  0|false|no|off)
    LEGACY_DS_ENABLED=0
    ;;
  auto|"")
    if [[ "${TRACKER_PROFILE}" == "old-tracker" || "${DATA_MODE}" == "pure-sd" ]]; then
      LEGACY_DS_ENABLED=1
    fi
    ;;
  *)
    echo "[ERROR] USE_LEGACY_DS must be one of: auto|0|1|true|false|yes|no|on|off. Got: ${USE_LEGACY_DS}" >&2
    exit 2
    ;;
esac

if [[ "${LEGACY_DS_ENABLED}" == "1" ]]; then
  if [[ ! -f "${LEGACY_PREPARE_SCRIPT}" ]]; then
    echo "[ERROR] legacy prepare script not found: ${LEGACY_PREPARE_SCRIPT}" >&2
    exit 1
  fi
  bash "${LEGACY_PREPARE_SCRIPT}"
  LEGACY_DS_PREPARED=1
fi

TEACHER_OBS_KEYS_EXPLICIT=0
[[ -n "${TEACHER_OBS_KEYS+x}" ]] && TEACHER_OBS_KEYS_EXPLICIT=1
TEACHER_PERCEPTION_PRESET_EXPLICIT=0
[[ -n "${TEACHER_PERCEPTION_PRESET+x}" ]] && TEACHER_PERCEPTION_PRESET_EXPLICIT=1
TEACHER_PERCEPTION_OBS_KEY_EXPLICIT=0
[[ -n "${TEACHER_PERCEPTION_OBS_KEY+x}" ]] && TEACHER_PERCEPTION_OBS_KEY_EXPLICIT=1
PERCEPTION_INTO_CRITIC_MODULES_EXPLICIT=0
[[ -n "${PERCEPTION_INTO_CRITIC_MODULES+x}" ]] && PERCEPTION_INTO_CRITIC_MODULES_EXPLICIT=1
TEACHER_ACTOR_OBS_HISTORY_LENGTH_EXPLICIT=0
[[ -n "${TEACHER_ACTOR_OBS_HISTORY_LENGTH+x}" ]] && TEACHER_ACTOR_OBS_HISTORY_LENGTH_EXPLICIT=1
STUDENT_PROPRIO_HISTORY_LENGTH_EXPLICIT=0
[[ -n "${STUDENT_PROPRIO_HISTORY_LENGTH+x}" ]] && STUDENT_PROPRIO_HISTORY_LENGTH_EXPLICIT=1
CRITIC_PROPRIO_HISTORY_LENGTH_EXPLICIT=0
[[ -n "${CRITIC_PROPRIO_HISTORY_LENGTH+x}" ]] && CRITIC_PROPRIO_HISTORY_LENGTH_EXPLICIT=1
TEACHER_ACTION_MIX_RATIO_EXPLICIT=0
[[ -n "${TEACHER_ACTION_MIX_RATIO+x}" ]] && TEACHER_ACTION_MIX_RATIO_EXPLICIT=1
TEACHER_ACTION_MIX_RATIO_START_EXPLICIT=0
[[ -n "${TEACHER_ACTION_MIX_RATIO_START+x}" ]] && TEACHER_ACTION_MIX_RATIO_START_EXPLICIT=1
TEACHER_ACTION_MIX_RATIO_END_EXPLICIT=0
[[ -n "${TEACHER_ACTION_MIX_RATIO_END+x}" ]] && TEACHER_ACTION_MIX_RATIO_END_EXPLICIT=1
TEACHER_ACTION_MIX_RATIO_END_ITERATION_EXPLICIT=0
[[ -n "${TEACHER_ACTION_MIX_RATIO_END_ITERATION+x}" ]] && TEACHER_ACTION_MIX_RATIO_END_ITERATION_EXPLICIT=1
PPO_START_EPOCH_EXPLICIT=0
[[ -n "${PPO_START_EPOCH+x}" ]] && PPO_START_EPOCH_EXPLICIT=1
DAGGER_END_EPOCH_EXPLICIT=0
[[ -n "${DAGGER_END_EPOCH+x}" ]] && DAGGER_END_EPOCH_EXPLICIT=1
NUM_LEARNING_ITERATIONS_EXPLICIT=0
[[ -n "${NUM_LEARNING_ITERATIONS+x}" ]] && NUM_LEARNING_ITERATIONS_EXPLICIT=1
NUM_ENVS_EXPLICIT=0
[[ -n "${NUM_ENVS+x}" ]] && NUM_ENVS_EXPLICIT=1
PER_GPU_ENVS_EXPLICIT=0
[[ -n "${PER_GPU_ENVS+x}" ]] && PER_GPU_ENVS_EXPLICIT=1
TOTAL_NUM_ENVS_EXPLICIT=0
[[ -n "${TOTAL_NUM_ENVS+x}" ]] && TOTAL_NUM_ENVS_EXPLICIT=1
PPO_TARGET_COEFF_EXPLICIT=0
[[ -n "${PPO_TARGET_COEFF+x}" ]] && PPO_TARGET_COEFF_EXPLICIT=1
PPO_START_COEFF_EXPLICIT=0
[[ -n "${PPO_START_COEFF+x}" ]] && PPO_START_COEFF_EXPLICIT=1
DAGGER_LOSS_COEF_EXPLICIT=0
[[ -n "${DAGGER_LOSS_COEF+x}" ]] && DAGGER_LOSS_COEF_EXPLICIT=1
PPO_SCHEDULE_STEP_EPOCHS_EXPLICIT=0
[[ -n "${PPO_SCHEDULE_STEP_EPOCHS+x}" ]] && PPO_SCHEDULE_STEP_EPOCHS_EXPLICIT=1
USE_ADAPTIVE_TIMESTEPS_SAMPLER_EXPLICIT=0
[[ -n "${USE_ADAPTIVE_TIMESTEPS_SAMPLER+x}" ]] && USE_ADAPTIVE_TIMESTEPS_SAMPLER_EXPLICIT=1
SCHEDULE_NAME_EXPLICIT=0
[[ -n "${SCHEDULE_NAME+x}" ]] && SCHEDULE_NAME_EXPLICIT=1
SCHEDULE_NOTES_EXPLICIT=0
[[ -n "${SCHEDULE_NOTES+x}" ]] && SCHEDULE_NOTES_EXPLICIT=1
IMAGE_WIDTH_EXPLICIT=0
[[ -n "${IMAGE_WIDTH+x}" ]] && IMAGE_WIDTH_EXPLICIT=1
IMAGE_HEIGHT_EXPLICIT=0
[[ -n "${IMAGE_HEIGHT+x}" ]] && IMAGE_HEIGHT_EXPLICIT=1
CAMERA_FAR_EXPLICIT=0
[[ -n "${CAMERA_FAR+x}" ]] && CAMERA_FAR_EXPLICIT=1
CAMERA_MAX_DISTANCE_EXPLICIT=0
[[ -n "${CAMERA_MAX_DISTANCE+x}" ]] && CAMERA_MAX_DISTANCE_EXPLICIT=1
PERCEPTION_WARP_PREPROCESS_EXPLICIT=0
[[ -n "${PERCEPTION_WARP_PREPROCESS+x}" ]] && PERCEPTION_WARP_PREPROCESS_EXPLICIT=1
CAMERA_APPLY_SENSOR_NOISE_EXPLICIT=0
[[ -n "${CAMERA_APPLY_SENSOR_NOISE+x}" ]] && CAMERA_APPLY_SENSOR_NOISE_EXPLICIT=1
EXPORT_ONNX_EXPLICIT=0
[[ -n "${EXPORT_ONNX+x}" ]] && EXPORT_ONNX_EXPLICIT=1
MOTION_DIR_EXPLICIT=0
[[ -n "${MOTION_DIR+x}" ]] && MOTION_DIR_EXPLICIT=1

# Sim2real default: ppo-first uses sparse-goal pickup; other schedules keep the legacy sparse-root-cmd path.
if [[ -z "${EXP:-}" ]]; then
  if [[ "${SCHEDULE_VARIANT}" == "ppo_first" ]]; then
    EXP="g1-29dof-wbt-w-object-distill-sparse-goal-mixed-r2s-rollout-ref-pickup"
  else
    EXP="g1-29dof-wbt-w-object-distill-sparse-root-cmd-r2s-rollout-ref"
  fi
fi
if [[ -z "${RUN_NAME:-}" ]]; then
  if [[ "${EXP}" == "g1-29dof-wbt-w-object-distill-sparse-goal-mixed-r2s-rollout-ref-pickup" ]]; then
    RUN_NAME="g1_w_object_distill_box_perception_sparse_goal_pickup_r2s_rollout_ref"
  else
    RUN_NAME="g1_w_object_distill_box_perception_sparse_root_cmd_r2s_rollout_ref"
  fi
fi
if [[ -z "${TRAINING_NAME:-}" ]]; then
  if [[ "${EXP}" == "g1-29dof-wbt-w-object-distill-sparse-goal-mixed-r2s-rollout-ref-pickup" ]]; then
    TRAINING_NAME="g1_29dof_wbt_w_object_distill_box_perception_sparse_goal_pickup_r2s_rollout_ref_access_to_depth"
  else
    TRAINING_NAME="g1_29dof_wbt_w_object_distill_box_perception_sparse_root_cmd_r2s_rollout_ref_access_to_depth"
  fi
fi
TRAINING_PROJECT=${TRAINING_PROJECT:-boxer}
OLD_TRACKER_MAX_BOX_ID=${OLD_TRACKER_MAX_BOX_ID:-92}
OLD_TRACKER_DAGGER_ITERATIONS=${OLD_TRACKER_DAGGER_ITERATIONS:-40000}
if [[ -n "${POSITIONAL_RUN_NAME}" ]]; then
  RUN_NAME="${POSITIONAL_RUN_NAME}"
fi

# Keep launcher self-contained: direct `bash ./distill_box_perception.sh` works out of box.
HSSIM_BIN_DIR=${HSSIM_BIN_DIR:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin}
if [[ -d "${HSSIM_BIN_DIR}" ]]; then
  export PATH="${HSSIM_BIN_DIR}:${PATH}"
fi
CUDA_VISIBLE_DEVICES="$(default_cuda_visible_devices_all "${CUDA_VISIBLE_DEVICES:-}")"
if [[ -z "${NPROC:-}" ]]; then
  NPROC="$(count_cuda_visible_devices "${CUDA_VISIBLE_DEVICES}")"
fi
if ! [[ "${NPROC}" =~ ^[0-9]+$ ]] || (( NPROC < 1 )); then
  echo "[ERROR] NPROC must be a positive integer. Got: ${NPROC}" >&2
  exit 1
fi
DEFAULT_ENVS_PER_GPU=${DEFAULT_ENVS_PER_GPU:-4096}
# In this launcher, NUM_ENVS means envs per GPU. train_agent.py expects a global
# all-rank total and divides by WORLD_SIZE, so we multiply once just before launch.
if [[ "${TOTAL_NUM_ENVS_EXPLICIT}" -eq 1 ]]; then
  :
elif [[ "${PER_GPU_ENVS_EXPLICIT}" -eq 1 ]]; then
  NUM_ENVS="${PER_GPU_ENVS}"
elif [[ "${NUM_ENVS_EXPLICIT}" -eq 0 ]]; then
  NUM_ENVS="${DEFAULT_ENVS_PER_GPU}"
fi
if [[ "${LEGACY_DS_ENABLED}" == "1" && "${TRACKER_PROFILE}" == "old-tracker" && "${DATA_MODE}" == "pure-sd" && "${DS_DATA_ROOT_EXPLICIT}" -eq 0 ]]; then
  DS_DATA_ROOT="${LEGACY_DS_ROOT}"
else
  DS_DATA_ROOT=${DS_DATA_ROOT:-"${SCRIPT_DIR}/data/ds_box_data"}
fi
DS_DATA_ROOT="$(ogds_resolve_data_root "${DS_DATA_ROOT}")"
DEFAULT_DS_PREPARED_MOTION_DIR="$(ogds_default_motion_dir "${DS_DATA_ROOT}" pure-sd)"
DEFAULT_MIX_NAIVE_MOTION_DIR="$(ogds_default_motion_dir "${DS_DATA_ROOT}" mix-naive)"
DEFAULT_TEACHER_ROLLOUT_MOTION_DIR="${SCRIPT_DIR}/outputs/motion_bank"
DEFAULT_TEACHER_ROLLOUT_CONTACT_DIR="${SCRIPT_DIR}/outputs/clips"
TEACHER_ROLLOUT_FILTER_ENABLED=${TEACHER_ROLLOUT_FILTER_ENABLED:-True}
TEACHER_ROLLOUT_SUCCESS_CLIPS_FILE=${TEACHER_ROLLOUT_SUCCESS_CLIPS_FILE:-"${SCRIPT_DIR}/outputs_sts/success_clips.txt"}
TEACHER_ROLLOUT_FILTERED_MOTION_DIR=${TEACHER_ROLLOUT_FILTERED_MOTION_DIR:-"${SCRIPT_DIR}/outputs/motion_bank_success_box_0_92_0p3"}
OLD_TRACKER_CACHE_ROOT="${DS_DATA_ROOT}/_motion_subsets"
PURE_REAL_OMOMO_PREFIXES=${PURE_REAL_OMOMO_PREFIXES:-'["sub"]'}
USING_TEACHER_ROLLOUT_MOTION_BANK=0
TEACHER_ROLLOUT_FILTER_CLIP_COUNT=""
TEACHER_ROLLOUT_FILTER_CONTACT_COUNT=""
TEACHER_ROLLOUT_FILTER_SOURCE_DIR=""

prepare_old_tracker_motion_subset() {
  local source_dir="$1"
  local max_box_id="$2"
  local subset_dir="${OLD_TRACKER_CACHE_ROOT}/$(basename "${source_dir}")_old_tracker_box_le_${max_box_id}"

  mkdir -p "${subset_dir}"
  "${PYTHON_BIN}" - "${source_dir}" "${subset_dir}" "${max_box_id}" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

source_dir = Path(sys.argv[1]).expanduser().resolve()
subset_dir = Path(sys.argv[2]).expanduser().resolve()
max_box_id = int(sys.argv[3])

subset_dir.mkdir(parents=True, exist_ok=True)
selected: list[Path] = []
for clip_path in sorted(source_dir.glob("box_*.npz")):
    stem = clip_path.stem
    suffix = stem.split("_", 1)[1] if "_" in stem else ""
    if not suffix.isdigit():
        continue
    if int(suffix) <= max_box_id:
        selected.append(clip_path)

if not selected:
    raise SystemExit(f"No numeric box clips <= {max_box_id} found in {source_dir}")

for existing in subset_dir.glob("*.npz"):
    existing.unlink()

for clip_path in selected:
    target = subset_dir / clip_path.name
    if target.exists() or target.is_symlink():
        target.unlink()
    os.symlink(clip_path, target)

metadata_payload = {}
clip_metadata = {}
metadata_uses_clips_key = False
for candidate in (source_dir / "_clip_object_urdf_map.json", source_dir / "clip_object_urdf_map.json"):
    if candidate.is_file():
        with candidate.open("r", encoding="utf-8") as f:
            metadata_payload = json.load(f)
        if isinstance(metadata_payload, dict) and isinstance(metadata_payload.get("clips"), dict):
            clip_metadata = metadata_payload["clips"]
            metadata_uses_clips_key = True
        elif isinstance(metadata_payload, dict):
            clip_metadata = metadata_payload
        break

subset_map_path = subset_dir / "_clip_object_urdf_map.json"
if subset_map_path.exists():
    subset_map_path.unlink()

if clip_metadata:
    filtered_clips = {key: value for key, value in clip_metadata.items() if (subset_dir / f"{key}.npz").exists()}
    if filtered_clips:
        output_payload = {"clips": filtered_clips} if metadata_uses_clips_key else filtered_clips
        with subset_map_path.open("w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2, sort_keys=True)
            f.write("\n")

print(str(subset_dir))
print(len(selected))
PY
}

prepare_teacher_rollout_success_motion_subset() {
  local source_dir="$1"
  local success_clips_file="$2"
  local subset_dir="$3"

  mkdir -p "${subset_dir}"
  "${PYTHON_BIN}" - "${source_dir}" "${subset_dir}" "${success_clips_file}" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

source_dir = Path(sys.argv[1]).expanduser().resolve()
subset_dir = Path(sys.argv[2]).expanduser().resolve()
success_clips_file = Path(sys.argv[3]).expanduser().resolve()

if not success_clips_file.is_file():
    raise SystemExit(f"Teacher rollout success clip filter not found: {success_clips_file}")

clip_ids: list[str] = []
seen: set[str] = set()
for raw_line in success_clips_file.read_text(encoding="utf-8").splitlines():
    clip_id = raw_line.strip()
    if not clip_id or clip_id.startswith("#"):
        continue
    if "/" in clip_id or "\\" in clip_id:
        raise SystemExit(f"Invalid clip id in {success_clips_file}: {clip_id}")
    if clip_id not in seen:
        clip_ids.append(clip_id)
        seen.add(clip_id)

if not clip_ids:
    raise SystemExit(f"No clip ids found in teacher rollout success filter: {success_clips_file}")

missing = [clip_id for clip_id in clip_ids if not (source_dir / f"{clip_id}.npz").is_file()]
if missing:
    preview = ", ".join(missing[:20])
    raise SystemExit(f"Success-filtered teacher rollout clips missing from {source_dir}: {preview}")

subset_dir.mkdir(parents=True, exist_ok=True)
for existing in subset_dir.glob("*.npz"):
    existing.unlink()

for clip_id in clip_ids:
    source_path = (source_dir / f"{clip_id}.npz").resolve()
    target = subset_dir / source_path.name
    if target.exists() or target.is_symlink():
        target.unlink()
    os.symlink(source_path, target)

metadata_payload = {}
clip_metadata = {}
metadata_uses_clips_key = True
for candidate in (source_dir / "_clip_object_urdf_map.json", source_dir / "clip_object_urdf_map.json"):
    if candidate.is_file():
        with candidate.open("r", encoding="utf-8") as f:
            metadata_payload = json.load(f)
        if isinstance(metadata_payload, dict) and isinstance(metadata_payload.get("clips"), dict):
            clip_metadata = metadata_payload["clips"]
            metadata_uses_clips_key = True
        elif isinstance(metadata_payload, dict):
            clip_metadata = metadata_payload
            metadata_uses_clips_key = False
        break

filtered_clips = {}
missing_metadata: list[str] = []
for clip_id in clip_ids:
    entry = clip_metadata.get(clip_id) if isinstance(clip_metadata, dict) else None
    if entry is not None:
        filtered_clips[clip_id] = entry
        continue
    npz_path = source_dir / f"{clip_id}.npz"
    try:
        data = np.load(npz_path, allow_pickle=True)
        urdf = ""
        if "object_urdf_path" in data:
            arr = np.asarray(data["object_urdf_path"])
            if arr.size:
                item = arr.item() if arr.shape == () else arr.reshape(-1)[0]
                urdf = str(item).strip()
    except Exception:
        urdf = ""
    if not urdf:
        missing_metadata.append(clip_id)
    else:
        filtered_clips[clip_id] = {"object_urdf_path": urdf}

if missing_metadata:
    preview = ", ".join(missing_metadata[:20])
    raise SystemExit(f"Success-filtered teacher rollout clips missing object metadata: {preview}")

subset_map_path = subset_dir / "_clip_object_urdf_map.json"
if subset_map_path.exists() or subset_map_path.is_symlink():
    subset_map_path.unlink()

output_payload = {"clips": filtered_clips} if metadata_uses_clips_key else filtered_clips
with subset_map_path.open("w", encoding="utf-8") as f:
    json.dump(output_payload, f, indent=2, sort_keys=True)
    f.write("\n")

print(str(subset_dir))
print(len(clip_ids))
PY
}

validate_teacher_rollout_success_contact_artifacts() {
  local contact_root="$1"
  local success_clips_file="$2"

  "${PYTHON_BIN}" - "${contact_root}" "${success_clips_file}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

contact_root = Path(sys.argv[1]).expanduser().resolve()
success_clips_file = Path(sys.argv[2]).expanduser().resolve()

clip_ids = [
    line.strip()
    for line in success_clips_file.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
]
required_files = ("teacher_rollout_reference.npz", "left_wrist_contact_interval_steps.npy")

def infer_clip_id(dir_name: str) -> str:
    return dir_name.split("_", 1)[1].strip() if "_" in dir_name else dir_name.strip()

contact_dirs: dict[str, Path] = {}
for candidate in sorted(contact_root.iterdir()):
    if candidate.is_dir():
        contact_dirs.setdefault(infer_clip_id(candidate.name), candidate)

missing: list[str] = []
for clip_id in clip_ids:
    clip_dir = contact_dirs.get(clip_id)
    if clip_dir is None:
        missing.append(f"{clip_id}:contact_dir")
        continue
    for file_name in required_files:
        if not (clip_dir / file_name).is_file():
            missing.append(f"{clip_id}:{file_name}")

if missing:
    preview = ", ".join(missing[:20])
    raise SystemExit(f"Success-filtered teacher rollout clips missing contact artifacts in {contact_root}: {preview}")

print(len(clip_ids))
PY
}

case "${DATA_MODE}" in
  default|pure-sd)
    DATA_MODE="pure-sd"
    MOTION_DIR=${MOTION_DIR:-"${DEFAULT_DS_PREPARED_MOTION_DIR}"}
    ;;
  pure-real|pure-omomo)
    DATA_MODE="pure-real"
    MOTION_DIR=${MOTION_DIR:-"${DEFAULT_MIX_NAIVE_MOTION_DIR}"}
    ;;
  mix-naive)
    MOTION_DIR=${MOTION_DIR:-"${DEFAULT_MIX_NAIVE_MOTION_DIR}"}
    ;;
  *)
    echo "[ERROR] Unsupported DATA_MODE='${DATA_MODE}'. Use one of: default, mix-naive, pure-real, pure-sd" >&2
    exit 2
    ;;
esac

if [[ (
  "${EXP}" == "g1-29dof-wbt-w-object-distill-sparse-root-cmd-r2s-rollout-ref" ||
  "${EXP}" == "g1-29dof-wbt-w-object-distill-sparse-root-cmd-r2s-rollout-ref-shoo7sr1-debug" ||
  "${EXP}" == "g1-29dof-wbt-w-object-distill-sparse-goal-mixed-r2s-rollout-ref" ||
  "${EXP}" == "g1-29dof-wbt-w-object-distill-sparse-goal-mixed-r2s-rollout-ref-pickup"
) && "${MOTION_DIR_EXPLICIT}" -eq 0 ]]; then
  if compgen -G "${DEFAULT_TEACHER_ROLLOUT_MOTION_DIR}/box_*.npz" > /dev/null; then
    _teacher_rollout_filter_enabled_norm="$(echo "${TEACHER_ROLLOUT_FILTER_ENABLED}" | tr '[:upper:]' '[:lower:]')"
    case "${_teacher_rollout_filter_enabled_norm}" in
      1|true|yes|on)
        mapfile -t _teacher_rollout_subset_info < <(
          prepare_teacher_rollout_success_motion_subset \
            "${DEFAULT_TEACHER_ROLLOUT_MOTION_DIR}" \
            "${TEACHER_ROLLOUT_SUCCESS_CLIPS_FILE}" \
            "${TEACHER_ROLLOUT_FILTERED_MOTION_DIR}"
        )
        MOTION_DIR="${_teacher_rollout_subset_info[0]}"
        TEACHER_ROLLOUT_FILTER_CLIP_COUNT="${_teacher_rollout_subset_info[1]:-}"
        TEACHER_ROLLOUT_FILTER_SOURCE_DIR="${DEFAULT_TEACHER_ROLLOUT_MOTION_DIR}"
        ;;
      *)
        MOTION_DIR="${DEFAULT_TEACHER_ROLLOUT_MOTION_DIR}"
        ;;
    esac
    USING_TEACHER_ROLLOUT_MOTION_BANK=1
  else
    echo "[ERROR] rollout-ref experiment requires teacher rollout motion bank at ${DEFAULT_TEACHER_ROLLOUT_MOTION_DIR}" >&2
    exit 2
  fi
  if ! compgen -G "${DEFAULT_TEACHER_ROLLOUT_CONTACT_DIR}/*/teacher_rollout_reference.npz" > /dev/null; then
    echo "[ERROR] rollout-ref experiment requires teacher rollout references under ${DEFAULT_TEACHER_ROLLOUT_CONTACT_DIR}" >&2
    exit 2
  fi
  if ! compgen -G "${DEFAULT_TEACHER_ROLLOUT_CONTACT_DIR}/*/left_wrist_contact_interval_steps.npy" > /dev/null; then
    echo "[ERROR] rollout-ref experiment requires wrist contact intervals under ${DEFAULT_TEACHER_ROLLOUT_CONTACT_DIR}" >&2
    exit 2
  fi
  if [[ -n "${TEACHER_ROLLOUT_FILTER_CLIP_COUNT}" ]]; then
    TEACHER_ROLLOUT_FILTER_CONTACT_COUNT="$(
      validate_teacher_rollout_success_contact_artifacts \
        "${DEFAULT_TEACHER_ROLLOUT_CONTACT_DIR}" \
        "${TEACHER_ROLLOUT_SUCCESS_CLIPS_FILE}"
    )"
  fi
fi

OLD_TRACKER_CLIP_COUNT=""
if [[ "${TRACKER_PROFILE}" == "old-tracker" && "${DATA_MODE}" == "pure-sd" && "${MOTION_DIR_EXPLICIT}" -eq 0 && "${USING_TEACHER_ROLLOUT_MOTION_BANK}" -eq 0 ]]; then
  mapfile -t _old_tracker_subset_info < <(prepare_old_tracker_motion_subset "${MOTION_DIR}" "${OLD_TRACKER_MAX_BOX_ID}")
  MOTION_DIR="${_old_tracker_subset_info[0]}"
  OLD_TRACKER_CLIP_COUNT="${_old_tracker_subset_info[1]:-}"
fi

OBJECT_MAP_PATH="${MOTION_DIR}/_clip_object_urdf_map.json"
OBJECT_BANK_WANDB_ENV="${MOTION_DIR}/_object_bank_wandb.env"
rm -f "${OBJECT_BANK_WANDB_ENV}"

if [[ -f "${OBJECT_MAP_PATH}" ]]; then
  ACTIVE_OBJECT_BANK_INFO=$(
    "${PYTHON_BIN}" - "${MOTION_DIR}" "${OBJECT_MAP_PATH}" <<'PY'
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

motion_dir = Path(sys.argv[1]).expanduser().resolve()
object_map = Path(sys.argv[2]).expanduser().resolve()
repo_root = Path.cwd().resolve()
sys.path.insert(0, str(repo_root / "src" / "holosoma"))
from holosoma.utils.path import resolve_data_file_path
payload = json.loads(object_map.read_text(encoding="utf-8"))
clips = payload["clips"] if isinstance(payload, dict) and isinstance(payload.get("clips"), dict) else payload
if not isinstance(clips, dict) or not clips:
    raise SystemExit(f"[ERROR] Invalid clip-object map payload: {object_map}")

active_clip_ids = [path.stem for path in sorted(motion_dir.glob("*.npz"))]
if not active_clip_ids:
    raise SystemExit(f"[ERROR] No .npz clips found under active MOTION_DIR: {motion_dir}")

resolved_urdfs: list[str] = []
missing: list[str] = []
for clip_id in active_clip_ids:
    entry = clips.get(clip_id)
    urdf = ""
    if isinstance(entry, str):
        urdf = entry.strip()
    elif isinstance(entry, dict):
        urdf = str(entry.get("object_urdf_path", "")).strip()

    if not urdf:
        npz_path = motion_dir / f"{clip_id}.npz"
        data = np.load(npz_path, allow_pickle=True)
        if "object_urdf_path" in data:
            arr = np.asarray(data["object_urdf_path"])
            if arr.size:
                item = arr.item() if arr.shape == () else arr.reshape(-1)[0]
                urdf = str(item).strip()

    if not urdf:
        missing.append(clip_id)
        continue
    urdf_candidate = Path(urdf).expanduser()
    if not urdf_candidate.is_absolute() and not urdf.startswith("holosoma/data"):
        motion_relative = (motion_dir / urdf).resolve()
        if motion_relative.is_file():
            urdf_candidate = motion_relative
        else:
            urdf_candidate = Path(resolve_data_file_path(urdf)).expanduser().resolve()
    else:
        urdf_candidate = Path(resolve_data_file_path(urdf)).expanduser().resolve()
    resolved_urdfs.append(str(urdf_candidate))

if missing:
    preview = ", ".join(missing[:10])
    raise SystemExit(
        f"[ERROR] Active motion clips missing object_urdf_path resolution in {object_map}: {preview}"
    )

counts = Counter(resolved_urdfs)
top = ", ".join(f"{Path(path).name}:{count}" for path, count in counts.most_common(5))
box_clip_count = 0
omomo_clip_count = 0
box_urdfs: set[str] = set()
omomo_urdfs: set[str] = set()
for clip_id, urdf in zip(active_clip_ids, resolved_urdfs, strict=True):
    is_omomo = clip_id.startswith("sub") or Path(urdf).name == "objects_largebox.urdf"
    if is_omomo:
        omomo_clip_count += 1
        omomo_urdfs.add(urdf)
    else:
        box_clip_count += 1
        box_urdfs.add(urdf)

env_path = motion_dir / "_object_bank_wandb.env"
env_path.write_text(
    "\n".join(
        [
            f"HOLOSOMA_OBJECT_BANK_TOTAL_MOTION_COUNT={len(active_clip_ids)}",
            f"HOLOSOMA_OBJECT_BANK_TOTAL_UNIQUE_URDF_COUNT={len(counts)}",
            f"HOLOSOMA_OBJECT_BANK_BOX_MOTION_COUNT={box_clip_count}",
            f"HOLOSOMA_OBJECT_BANK_BOX_UNIQUE_URDF_COUNT={len(box_urdfs)}",
            f"HOLOSOMA_OBJECT_BANK_OMOMO_MOTION_COUNT={omomo_clip_count}",
            f"HOLOSOMA_OBJECT_BANK_OMOMO_UNIQUE_URDF_COUNT={len(omomo_urdfs)}",
            f"HOLOSOMA_OBJECT_BANK_MOTION_DIR={motion_dir}",
            f"HOLOSOMA_OBJECT_BANK_OBJECT_MAP={object_map}",
        ]
    )
    + "\n",
    encoding="utf-8",
)

print(
    f"{len(active_clip_ids)}|{len(counts)}|{box_clip_count}|{len(box_urdfs)}|"
    f"{omomo_clip_count}|{len(omomo_urdfs)}|{top}"
)
PY
  )
else
  ACTIVE_OBJECT_BANK_INFO=""
fi

if [[ -f "${OBJECT_BANK_WANDB_ENV}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${OBJECT_BANK_WANDB_ENV}"
  set +a
fi

TEACHER_OBS_KEYS=${TEACHER_OBS_KEYS:-actor_obs}
TEACHER_PERCEPTION_PRESET=${TEACHER_PERCEPTION_PRESET:-none}
TEACHER_PERCEPTION_OBS_KEY=${TEACHER_PERCEPTION_OBS_KEY-teacher_perception_obs}
CRITIC_PERCEPTION_PRESET=${CRITIC_PERCEPTION_PRESET:-none}
CRITIC_PERCEPTION_OBS_KEY=${CRITIC_PERCEPTION_OBS_KEY:-critic_perception_obs}
PERCEPTION_INTO_CRITIC_MODULES=${PERCEPTION_INTO_CRITIC_MODULES:-}
TEACHER_ACTOR_OBS_HISTORY_LENGTH=${TEACHER_ACTOR_OBS_HISTORY_LENGTH:-}
TEACHER_COMPAT_PROFILE=${TEACHER_COMPAT_PROFILE:-auto}
TEACHER_COMPAT_NOTES=${TEACHER_COMPAT_NOTES:-}
TEACHER_ACTION_MIX_RATIO=${TEACHER_ACTION_MIX_RATIO:-0.0}
TEACHER_ACTION_MIX_RATIO_START=${TEACHER_ACTION_MIX_RATIO_START:-}
TEACHER_ACTION_MIX_RATIO_END=${TEACHER_ACTION_MIX_RATIO_END:-}
TEACHER_ACTION_MIX_RATIO_END_ITERATION=${TEACHER_ACTION_MIX_RATIO_END_ITERATION:-}
BC_LOSS_COEF=${BC_LOSS_COEF:-1.0}
NUM_LEARNING_ITERATIONS=${NUM_LEARNING_ITERATIONS:-40000}
PPO_START_EPOCH=${PPO_START_EPOCH:-1000}
DAGGER_END_EPOCH=${DAGGER_END_EPOCH:-4500}
PPO_TARGET_COEFF=${PPO_TARGET_COEFF:-0.9}
PPO_START_COEFF=${PPO_START_COEFF:-0.0}
PPO_START_NOISE_STD=${PPO_START_NOISE_STD:-}
PPO_START_NOISE_STD_UNTIL_COEFF=${PPO_START_NOISE_STD_UNTIL_COEFF:-0.1}
PPO_SCHEDULE_STEP_EPOCHS=${PPO_SCHEDULE_STEP_EPOCHS:-500}
DAGGER_LOSS_COEF=${DAGGER_LOSS_COEF:-1.0}
DAGGER_MIX_TEACHER_ACTION_MIX_RATIO_START=${DAGGER_MIX_TEACHER_ACTION_MIX_RATIO_START:-0.7}
DAGGER_MIX_TEACHER_ACTION_MIX_RATIO_END=${DAGGER_MIX_TEACHER_ACTION_MIX_RATIO_END:-0.0}
DAGGER_MIX_TEACHER_ACTION_MIX_RATIO_END_ITERATION=${DAGGER_MIX_TEACHER_ACTION_MIX_RATIO_END_ITERATION:-3500}
FIXED_BC_EVAL_LOG_INTERVAL=${FIXED_BC_EVAL_LOG_INTERVAL:-1000}
if [[ "${SCHEDULE_NAME_EXPLICIT}" -eq 0 ]]; then
  if [[ "${EXP}" == *"sparse-goal-mixed"* ]]; then
    SCHEDULE_NAME="sparse_goal_pickup_teacher_anchor_v1_step_mix"
  else
    SCHEDULE_NAME="sparse_root_teacher_anchor_v3_step_mix"
  fi
fi
if [[ "${SCHEDULE_NOTES_EXPLICIT}" -eq 0 ]]; then
  if [[ "${EXP}" == *"sparse-goal-mixed"* ]]; then
    SCHEDULE_NOTES="Sparse-goal pickup perception distill. PPO/DAgger use the default staircase blend: PPO starts at 1000, increases by 0.1 every 500 iterations until capping at 0.9; pickup success is object latch plus current lift, not drop-to-target xy."
  else
    SCHEDULE_NOTES="No teacher rollout mix. PPO/DAgger use the default staircase blend: PPO starts at 1000, increases by 0.1 every 500 iterations until capping at 0.9; with dagger_loss_coef=1.0, the effective BC weight drops from 1.0 to 0.1 over the same schedule."
  fi
fi
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.2}
START_AT_TIMESTEP_ZERO_PROB_END=${START_AT_TIMESTEP_ZERO_PROB_END:-1.0}
START_AT_TIMESTEP_ZERO_PROB_START_ITER=${START_AT_TIMESTEP_ZERO_PROB_START_ITER:-2500}
START_AT_TIMESTEP_ZERO_PROB_END_ITER=${START_AT_TIMESTEP_ZERO_PROB_END_ITER:-}
FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}
FREEZE_AT_TIMESTEP_ZERO_PROB_END=${FREEZE_AT_TIMESTEP_ZERO_PROB_END:-0.0}
FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=${FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER:-2500}
FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=${FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER:-}
USE_ADAPTIVE_TIMESTEPS_SAMPLER=${USE_ADAPTIVE_TIMESTEPS_SAMPLER:-False}
ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT=${ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT:-"${SCRIPT_DIR}/outputs/clips"}
CONTACT_AWARE_CARRY_WINDOW_MODE=${CONTACT_AWARE_CARRY_WINDOW_MODE:-rel_z}
CONTACT_AWARE_PEAK_HEIGHT_ALPHA=${CONTACT_AWARE_PEAK_HEIGHT_ALPHA:-0.91}
CONTACT_AWARE_PEAK_HEIGHT_SMOOTHING_STEPS=${CONTACT_AWARE_PEAK_HEIGHT_SMOOTHING_STEPS:-5}
UNIFORM_T1_WINDOW_SAMPLING_ENABLED=${UNIFORM_T1_WINDOW_SAMPLING_ENABLED:-True}
UNIFORM_T1_WINDOW_HALF_WIDTH_STEPS=${UNIFORM_T1_WINDOW_HALF_WIDTH_STEPS:-50}
UNIFORM_T1_WINDOW_DENSITY_BOOST=${UNIFORM_T1_WINDOW_DENSITY_BOOST:-7.0}
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
PERCEPTION_PRESET=${PERCEPTION_PRESET:-camera_depth_d435i}
EXPORT_ONNX=${EXPORT_ONNX:-True}
STUDENT_ACTOR_INPUTS=${STUDENT_ACTOR_INPUTS:-"['actor_obs_root','actor_obs_proprio','actor_obs_actions']"}
DAGGER_MATCH_STD=${DAGGER_MATCH_STD:-True}
ENTROPY_COEF=${ENTROPY_COEF:-0.0}
DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES=${DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES:-True}
DAGGER_IGNORE_EPISODE_INITIAL_STEPS=${DAGGER_IGNORE_EPISODE_INITIAL_STEPS:-0}
MAX_EPISODE_LENGTH_S=${MAX_EPISODE_LENGTH_S:-8.0}
RESET_TO_DEFAULT_POSE=${RESET_TO_DEFAULT_POSE:-False}
ENABLE_DEFAULT_POSE_PREPEND=${ENABLE_DEFAULT_POSE_PREPEND:-False}
DEFAULT_POSE_PREPEND_DURATION_S=${DEFAULT_POSE_PREPEND_DURATION_S:-0.0}
ENABLE_DEFAULT_POSE_APPEND=${ENABLE_DEFAULT_POSE_APPEND:-False}
DEFAULT_POSE_APPEND_DURATION_S=${DEFAULT_POSE_APPEND_DURATION_S:-0.0}
VISER_DISTILL_MINIMAL_UI=${VISER_DISTILL_MINIMAL_UI:-1}
VISER_SHOW_TARGET_KEYPOINTS=${VISER_SHOW_TARGET_KEYPOINTS:-0}

if [[ "${ROOT_COMMAND_MODE}" == "contact-aware" ]]; then
  SCHEDULE_VARIANT="ppo_first"
  if [[ "${STUDENT_ACTOR_INPUTS_EXPLICIT}" -eq 0 ]]; then
    STUDENT_ACTOR_INPUTS="['actor_obs_root_contact_aware','actor_obs_proprio','actor_obs_actions']"
  fi
  if [[ "${USE_ADAPTIVE_TIMESTEPS_SAMPLER_EXPLICIT}" -eq 0 ]]; then
    USE_ADAPTIVE_TIMESTEPS_SAMPLER=True
  fi
  CONTACT_AWARE_CARRY_WINDOW_MODE=${CONTACT_AWARE_CARRY_WINDOW_MODE:-peak_height}
  if [[ "${CONTACT_AWARE_CARRY_WINDOW_MODE}" == "rel_z" ]]; then
    CONTACT_AWARE_CARRY_WINDOW_MODE=peak_height
  fi
  if [[ "${CONTACT_AWARE_HISTORY}" == "1" ]]; then
    if [[ "${STUDENT_PROPRIO_HISTORY_LENGTH_EXPLICIT}" -eq 0 ]]; then
      STUDENT_PROPRIO_HISTORY_LENGTH="${CONTACT_AWARE_HISTORY_LENGTH}"
    fi
    if [[ "${CRITIC_PROPRIO_HISTORY_LENGTH_EXPLICIT}" -eq 0 ]]; then
      CRITIC_PROPRIO_HISTORY_LENGTH="${CONTACT_AWARE_HISTORY_LENGTH}"
    fi
  fi
fi

# Keep camera intrinsics/range on preset defaults unless explicitly overridden.
# Pitch is intentionally written into the run config by default.
IMAGE_WIDTH=${IMAGE_WIDTH:-}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-}
CAMERA_PITCH_DEG=${CAMERA_PITCH_DEG-10}
CAMERA_FAR=${CAMERA_FAR:-}
CAMERA_MAX_DISTANCE=${CAMERA_MAX_DISTANCE:-}
PERCEPTION_WARP_PREPROCESS=${PERCEPTION_WARP_PREPROCESS:-}
CAMERA_APPLY_SENSOR_NOISE=${CAMERA_APPLY_SENSOR_NOISE:-True}
OBJECT_GEOMETRY_MODE=${OBJECT_GEOMETRY_MODE:-mesh}

if [[ "${SHOO7SR1_NEAR03_DEBUG}" == "1" ]]; then
  # Debug ablation: reproduce shoo7sr1, except depth near is intentionally 0.3.
  PERCEPTION_PRESET="camera_depth_d435i"
  EXPORT_ONNX=True
  STUDENT_ACTOR_INPUTS="['actor_obs_root','actor_obs_proprio_no_linvel']"
  STUDENT_ACTOR_INPUTS_EXPLICIT=1
  TEACHER_COMPAT_PROFILE="u5lguxvl_generalist"
  TEACHER_OBS_KEYS="actor_obs"
  TEACHER_OBS_KEYS_EXPLICIT=1
  TEACHER_PERCEPTION_PRESET="none"
  TEACHER_PERCEPTION_PRESET_EXPLICIT=1
  TEACHER_PERCEPTION_OBS_KEY=""
  TEACHER_PERCEPTION_OBS_KEY_EXPLICIT=1
  TEACHER_ACTOR_OBS_HISTORY_LENGTH="5"
  TEACHER_ACTOR_OBS_HISTORY_LENGTH_EXPLICIT=1
  STUDENT_PROPRIO_HISTORY_LENGTH=5
  CRITIC_PROPRIO_HISTORY_LENGTH=5
  CRITIC_PERCEPTION_PRESET="none"
  CRITIC_PERCEPTION_OBS_KEY="critic_perception_obs"
  PERCEPTION_INTO_CRITIC_MODULES="False"
  PERCEPTION_INTO_CRITIC_MODULES_EXPLICIT=1
  PPO_START_EPOCH=0
  PPO_START_EPOCH_EXPLICIT=1
  DAGGER_END_EPOCH=4000
  DAGGER_END_EPOCH_EXPLICIT=1
  PPO_START_COEFF=0.1
  PPO_START_COEFF_EXPLICIT=1
  PPO_TARGET_COEFF=0.9
  PPO_TARGET_COEFF_EXPLICIT=1
  PPO_SCHEDULE_STEP_EPOCHS=500
  PPO_SCHEDULE_STEP_EPOCHS_EXPLICIT=1
  DAGGER_LOSS_COEF=1.0
  DAGGER_LOSS_COEF_EXPLICIT=1
  PPO_START_NOISE_STD=0.1
  PPO_START_NOISE_STD_UNTIL_COEFF=0.1
  TEACHER_ACTION_MIX_RATIO=0.0
  TEACHER_ACTION_MIX_RATIO_START=""
  TEACHER_ACTION_MIX_RATIO_END=""
  TEACHER_ACTION_MIX_RATIO_END_ITERATION=""
  BC_LOSS_COEF=1.0
  USE_ADAPTIVE_TIMESTEPS_SAMPLER=True
  USE_ADAPTIVE_TIMESTEPS_SAMPLER_EXPLICIT=1
  SCHEDULE_NAME="sparse_root_teacher_anchor_v4_ppo_first_step_mix"
  SCHEDULE_NAME_EXPLICIT=1
  SCHEDULE_NOTES="PPO-first hybrid: PPO and DAgger are both active from iteration 0. PPO starts at 0.1 and increases by 0.1 every 500 iterations until 0.9 at 4000; effective DAgger weight starts at 0.9 and decreases to 0.1."
  SCHEDULE_NOTES_EXPLICIT=1
  if [[ "${NUM_LEARNING_ITERATIONS_EXPLICIT}" -eq 0 ]]; then
    NUM_LEARNING_ITERATIONS=20001
  fi
  DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES=False
  SPARSE_GOAL_ENABLED=False
  CLIP_GOAL_DELTA_MIN_STEPS=60
  CLIP_GOAL_DELTA_MAX_STEPS=180
  COMMAND_ONLY_ENV_PROB_START=0.0
  COMMAND_ONLY_ENV_PROB_END=0.0
  COMMAND_ONLY_ENV_PROB_START_ITER=None
  COMMAND_ONLY_ENV_PROB_END_ITER=None
  EVAL_COMMAND_ONLY_ENV_PROB=None
  EXTERNAL_GOAL_PROB_START=0.0
  EXTERNAL_GOAL_PROB_END=1.0
  EXTERNAL_GOAL_PROB_START_ITER=None
  EXTERNAL_GOAL_PROB_END_ITER=None
  EXTERNAL_GOAL_PROB_RAMP_RESETS=200000
  EVAL_EXTERNAL_GOAL_PROB=None
  EXTERNAL_GOAL_RANGE_RAMP_RESETS=None
  EXTERNAL_GOAL_RANGE_START_ITER=None
  EXTERNAL_GOAL_RANGE_END_ITER=None
  EXTERNAL_GOAL_SAMPLING_MODE=box
  EXTERNAL_GOAL_RADIUS_MIN_START=None
  EXTERNAL_GOAL_RADIUS_MAX_START=None
  EXTERNAL_GOAL_RADIUS_MIN=1.0
  EXTERNAL_GOAL_RADIUS_MAX=3.4
  EXTERNAL_GOAL_POS_LOCAL_MIN_START=None
  EXTERNAL_GOAL_POS_LOCAL_MAX_START=None
  EXTERNAL_GOAL_POS_LOCAL_MIN="[1.0, -0.8, 0.7]"
  EXTERNAL_GOAL_POS_LOCAL_MAX="[1.75, 0.8, 1.0]"
  START_AT_TIMESTEP_ZERO_PROB=0.2
  START_AT_TIMESTEP_ZERO_PROB_END=None
  START_AT_TIMESTEP_ZERO_PROB_START_ITER=None
  START_AT_TIMESTEP_ZERO_PROB_END_ITER=None
  FREEZE_AT_TIMESTEP_ZERO_PROB=0.0
  FREEZE_AT_TIMESTEP_ZERO_PROB_END=0.0
  FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=2500
  FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=10000
  UNIFORM_T1_WINDOW_SAMPLING_ENABLED=False
  UNIFORM_T1_WINDOW_DENSITY_BOOST=1.0
  IMAGE_WIDTH=106
  IMAGE_WIDTH_EXPLICIT=1
  IMAGE_HEIGHT=60
  IMAGE_HEIGHT_EXPLICIT=1
  CAMERA_PITCH_DEG=10
  CAMERA_FAR=3.0
  CAMERA_FAR_EXPLICIT=1
  CAMERA_MAX_DISTANCE=3.0
  CAMERA_MAX_DISTANCE_EXPLICIT=1
  PERCEPTION_WARP_PREPROCESS=True
  PERCEPTION_WARP_PREPROCESS_EXPLICIT=1
  CAMERA_APPLY_SENSOR_NOISE=False
  CAMERA_APPLY_SENSOR_NOISE_EXPLICIT=1
  OBJECT_GEOMETRY_MODE=default
  PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=134217728
  PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY=134217728
  PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=16777216
fi

OBJECT_GEOMETRY_MODE_NORM=""
HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE=""
PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE=""

if [[ -n "${OBJECT_GEOMETRY_MODE}" ]]; then
  case "$(echo "${OBJECT_GEOMETRY_MODE}" | tr '[:upper:]' '[:lower:]')" in
    default|preset|null|none)
      ;;
    1|true|yes|on|primitive|primitives|box|cuboid)
      OBJECT_GEOMETRY_MODE_NORM="primitive"
      HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE="primitive"
      PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE="primitive"
      ;;
    0|false|no|off|mesh|urdf|disable|disabled)
      OBJECT_GEOMETRY_MODE_NORM="mesh"
      HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE="urdf"
      PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE="mesh"
      ;;
    *)
      echo "[ERROR] OBJECT_GEOMETRY_MODE must be one of: default/on/off/primitive/mesh. Got: ${OBJECT_GEOMETRY_MODE}" >&2
      exit 2
      ;;
  esac
fi

if [[ "${TEACHER_ACTION_MIX_RATIO_EXPLICIT}" -eq 0 && "${TEACHER_ACTION_MIX_RATIO_START_EXPLICIT}" -eq 0 && "${TEACHER_ACTION_MIX_RATIO_END_EXPLICIT}" -eq 0 && "${TEACHER_ACTION_MIX_RATIO_END_ITERATION_EXPLICIT}" -eq 0 ]]; then
  TEACHER_ACTION_MIX_RATIO="0.0"
fi

if [[ "${TRACKER_PROFILE}" == "old-tracker" && "${SCHEDULE_VARIANT}" == "default" ]]; then
  if [[ "${PPO_START_EPOCH_EXPLICIT}" -eq 0 ]]; then
    PPO_START_EPOCH=$((NUM_LEARNING_ITERATIONS + 1))
  fi
  if [[ "${DAGGER_END_EPOCH_EXPLICIT}" -eq 0 ]]; then
    DAGGER_END_EPOCH=$((NUM_LEARNING_ITERATIONS + 2))
  fi
  if [[ "${PPO_TARGET_COEFF_EXPLICIT}" -eq 0 ]]; then
    PPO_TARGET_COEFF=0.0
  fi
  if [[ "${PPO_SCHEDULE_STEP_EPOCHS_EXPLICIT}" -eq 0 ]]; then
    PPO_SCHEDULE_STEP_EPOCHS=0
  fi
  if [[ "${SCHEDULE_NAME_EXPLICIT}" -eq 0 ]]; then
    if [[ "${EXP}" == *"sparse-goal-mixed"* ]]; then
      SCHEDULE_NAME="old_tracker_sparse_goal_pickup_pure_dagger_40k"
    else
      SCHEDULE_NAME="old_tracker_pure_dagger_40k"
    fi
  fi
  if [[ "${SCHEDULE_NOTES_EXPLICIT}" -eq 0 ]]; then
    if [[ "${EXP}" == *"sparse-goal-mixed"* ]]; then
      SCHEDULE_NOTES="Default old-tracker profile: sparse-goal pickup pure DAgger for 40000 iterations with PPO disabled by default. Motion bank is capped to numeric box clips <= 92 to match the old tracker coverage and avoid going beyond it."
    else
      SCHEDULE_NOTES="Default old-tracker profile: pure DAgger for 40000 iterations with PPO disabled by default. Motion bank is capped to numeric box clips <= 92 to match the old tracker coverage and avoid going beyond it."
    fi
  fi
fi

case "${SCHEDULE_VARIANT}" in
  default)
    ;;
  dagger_mix)
    # Keep PPO disabled while using teacher-action rollout mixing. PPO storage
    # keeps the sampled student action, while the environment may step with the
    # teacher action, so high teacher-action mix should remain a DAgger-only phase.
    if [[ "${PPO_START_EPOCH_EXPLICIT}" -eq 0 ]]; then
      PPO_START_EPOCH=$((NUM_LEARNING_ITERATIONS + 1))
    fi
    if [[ "${DAGGER_END_EPOCH_EXPLICIT}" -eq 0 ]]; then
      DAGGER_END_EPOCH=$((NUM_LEARNING_ITERATIONS + 2))
    fi
    if [[ "${PPO_TARGET_COEFF_EXPLICIT}" -eq 0 ]]; then
      PPO_TARGET_COEFF=0.0
    fi
    if [[ "${PPO_SCHEDULE_STEP_EPOCHS_EXPLICIT}" -eq 0 ]]; then
      PPO_SCHEDULE_STEP_EPOCHS=0
    fi
    if [[ "${DAGGER_LOSS_COEF_EXPLICIT}" -eq 0 ]]; then
      DAGGER_LOSS_COEF=1.0
    fi
    if [[ "${TEACHER_ACTION_MIX_RATIO_START_EXPLICIT}" -eq 0 && "${TEACHER_ACTION_MIX_RATIO_END_EXPLICIT}" -eq 0 && "${TEACHER_ACTION_MIX_RATIO_END_ITERATION_EXPLICIT}" -eq 0 && "${TEACHER_ACTION_MIX_RATIO_EXPLICIT}" -eq 0 ]]; then
      TEACHER_ACTION_MIX_RATIO_START="${DAGGER_MIX_TEACHER_ACTION_MIX_RATIO_START}"
      TEACHER_ACTION_MIX_RATIO_END="${DAGGER_MIX_TEACHER_ACTION_MIX_RATIO_END}"
      TEACHER_ACTION_MIX_RATIO_END_ITERATION="${DAGGER_MIX_TEACHER_ACTION_MIX_RATIO_END_ITERATION}"
    elif [[ "${TEACHER_ACTION_MIX_RATIO_START_EXPLICIT}" -eq 1 || "${TEACHER_ACTION_MIX_RATIO_END_EXPLICIT}" -eq 1 || "${TEACHER_ACTION_MIX_RATIO_END_ITERATION_EXPLICIT}" -eq 1 ]]; then
      if [[ "${TEACHER_ACTION_MIX_RATIO_START_EXPLICIT}" -eq 0 ]]; then
        TEACHER_ACTION_MIX_RATIO_START="${DAGGER_MIX_TEACHER_ACTION_MIX_RATIO_START}"
      fi
      if [[ "${TEACHER_ACTION_MIX_RATIO_END_EXPLICIT}" -eq 0 ]]; then
        TEACHER_ACTION_MIX_RATIO_END="${DAGGER_MIX_TEACHER_ACTION_MIX_RATIO_END}"
      fi
      if [[ "${TEACHER_ACTION_MIX_RATIO_END_ITERATION_EXPLICIT}" -eq 0 ]]; then
        TEACHER_ACTION_MIX_RATIO_END_ITERATION="${DAGGER_MIX_TEACHER_ACTION_MIX_RATIO_END_ITERATION}"
      fi
    fi
    if [[ -n "${TEACHER_ACTION_MIX_RATIO_START}" && -n "${TEACHER_ACTION_MIX_RATIO_END}" ]]; then
      TEACHER_ACTION_MIX_RATIO="${TEACHER_ACTION_MIX_RATIO_START}"
    fi
    if [[ "${SCHEDULE_NAME_EXPLICIT}" -eq 0 ]]; then
      if [[ "${EXP}" == *"sparse-goal-mixed"* ]]; then
        SCHEDULE_NAME="sparse_goal_pickup_dagger_action_mix"
      else
        SCHEDULE_NAME="old_tracker_dagger_action_mix"
      fi
    fi
    if [[ "${SCHEDULE_NOTES_EXPLICIT}" -eq 0 ]]; then
      SCHEDULE_NOTES="Pure DAgger with teacher-action rollout mixing. PPO is disabled by default. Teacher actions step the environment with probability ${TEACHER_ACTION_MIX_RATIO_START:-${TEACHER_ACTION_MIX_RATIO}} and linearly anneal to ${TEACHER_ACTION_MIX_RATIO_END:-${TEACHER_ACTION_MIX_RATIO}} by iteration ${TEACHER_ACTION_MIX_RATIO_END_ITERATION:-0}; this stabilizes early perception rollouts while still returning to student-driven DAgger data."
    fi
    ;;
  dag_first)
    if [[ "${PPO_START_EPOCH_EXPLICIT}" -eq 0 ]]; then
      PPO_START_EPOCH=2000
    fi
    if [[ "${DAGGER_END_EPOCH_EXPLICIT}" -eq 0 ]]; then
      DAGGER_END_EPOCH=3000
    fi
    if [[ "${PPO_TARGET_COEFF_EXPLICIT}" -eq 0 ]]; then
      PPO_TARGET_COEFF=0.3
    fi
    if [[ "${PPO_SCHEDULE_STEP_EPOCHS_EXPLICIT}" -eq 0 ]]; then
      PPO_SCHEDULE_STEP_EPOCHS=0
    fi
    if [[ "${DAGGER_LOSS_COEF_EXPLICIT}" -eq 0 ]]; then
      DAGGER_LOSS_COEF=1.0
    fi
    if [[ "${SCHEDULE_NAME_EXPLICIT}" -eq 0 ]]; then
      if [[ "${EXP}" == *"sparse-goal-mixed"* ]]; then
        SCHEDULE_NAME="sparse_goal_pickup_teacher_anchor_v2_dag_first"
      else
        SCHEDULE_NAME="sparse_root_teacher_anchor_v2_dag_first"
      fi
    fi
    if [[ "${SCHEDULE_NOTES_EXPLICIT}" -eq 0 ]]; then
      if [[ "${EXP}" == *"sparse-goal-mixed"* ]]; then
        SCHEDULE_NOTES="Sparse-goal pickup perception distill. 0-2000 pure DAgger with PPO disabled. 2000-3000 PPO ramps 0->0.3 while DAgger stays dominant."
      else
        SCHEDULE_NOTES="No teacher rollout mix. 0-2000 pure DAgger with PPO disabled. 2000-3000 PPO ramps 0->0.3 while DAgger stays dominant."
      fi
    fi
    ;;
  ppo_first)
    if [[ "${USE_ADAPTIVE_TIMESTEPS_SAMPLER_EXPLICIT}" -eq 0 ]]; then
      USE_ADAPTIVE_TIMESTEPS_SAMPLER=True
    fi
    if [[ "${PPO_START_EPOCH_EXPLICIT}" -eq 0 ]]; then
      PPO_START_EPOCH=0
    fi
    if [[ "${DAGGER_END_EPOCH_EXPLICIT}" -eq 0 ]]; then
      DAGGER_END_EPOCH=4000
    fi
    if [[ "${PPO_START_COEFF_EXPLICIT}" -eq 0 ]]; then
      PPO_START_COEFF=0.1
    fi
    if [[ "${PPO_TARGET_COEFF_EXPLICIT}" -eq 0 ]]; then
      PPO_TARGET_COEFF=0.9
    fi
    if [[ "${PPO_SCHEDULE_STEP_EPOCHS_EXPLICIT}" -eq 0 ]]; then
      PPO_SCHEDULE_STEP_EPOCHS=500
    fi
    if [[ "${DAGGER_LOSS_COEF_EXPLICIT}" -eq 0 ]]; then
      DAGGER_LOSS_COEF=1.0
    fi
    if [[ "${SCHEDULE_NAME_EXPLICIT}" -eq 0 ]]; then
      if [[ "${EXP}" == *"sparse-goal-mixed"* ]]; then
        SCHEDULE_NAME="sparse_goal_pickup_teacher_anchor_v4_ppo_first_step_mix"
      else
        SCHEDULE_NAME="sparse_root_teacher_anchor_v4_ppo_first_step_mix"
      fi
    fi
    if [[ "${SCHEDULE_NOTES_EXPLICIT}" -eq 0 ]]; then
      if [[ "${EXP}" == *"sparse-goal-mixed"* ]]; then
        SCHEDULE_NOTES="Sparse-goal pickup PPO-first hybrid: PPO and DAgger are both active from iteration 0. PPO starts at 0.1 and increases by 0.1 every 500 iterations until 0.9 at 4000; effective DAgger weight starts at 0.9 and decreases to 0.1."
      else
        SCHEDULE_NOTES="PPO-first hybrid: PPO and DAgger are both active from iteration 0. PPO starts at 0.1 and increases by 0.1 every 500 iterations until 0.9 at 4000; effective DAgger weight starts at 0.9 and decreases to 0.1."
      fi
    fi
    ;;
  *)
    echo "[ERROR] Unsupported SCHEDULE_VARIANT='${SCHEDULE_VARIANT}'. Use one of: default, dagger_mix, dag_first, ppo_first" >&2
    exit 2
    ;;
esac

if [[ "${ROOT_COMMAND_MODE}" == "contact-aware" ]]; then
  if [[ "${SCHEDULE_NAME_EXPLICIT}" -eq 0 ]]; then
    SCHEDULE_NAME="${SCHEDULE_NAME}_contact_aware"
  fi
  if [[ "${SCHEDULE_NOTES_EXPLICIT}" -eq 0 ]]; then
    SCHEDULE_NOTES="${SCHEDULE_NOTES} Contact-aware student sparse root command uses peak-height carry-window detection by default: command stays zero before the object is stably near peak carry height and after it stably drops below that height band."
  fi
  if [[ "${CONTACT_AWARE_HISTORY}" == "1" ]]; then
    if [[ "${SCHEDULE_NAME_EXPLICIT}" -eq 0 ]]; then
      SCHEDULE_NAME="${SCHEDULE_NAME}_history${CONTACT_AWARE_HISTORY_LENGTH}"
    fi
    if [[ "${SCHEDULE_NOTES_EXPLICIT}" -eq 0 ]]; then
      SCHEDULE_NOTES="${SCHEDULE_NOTES} Contact-aware-history additionally sets student actor proprio history and critic proprio history to ${CONTACT_AWARE_HISTORY_LENGTH} unless explicitly overridden."
    fi
  fi
fi

START_AT_TIMESTEP_ZERO_PROB_END_ITER=${START_AT_TIMESTEP_ZERO_PROB_END_ITER:-${NUM_LEARNING_ITERATIONS}}
FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=${FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER:-${NUM_LEARNING_ITERATIONS}}

SPARSE_GOAL_ENABLED=${SPARSE_GOAL_ENABLED:-False}
CLIP_GOAL_DELTA_MIN_STEPS=${CLIP_GOAL_DELTA_MIN_STEPS:-45}
CLIP_GOAL_DELTA_MAX_STEPS=${CLIP_GOAL_DELTA_MAX_STEPS:-120}
COMMAND_ONLY_ENV_PROB_START=${COMMAND_ONLY_ENV_PROB_START:-0.0}
COMMAND_ONLY_ENV_PROB_END=${COMMAND_ONLY_ENV_PROB_END:-0.5}
COMMAND_ONLY_ENV_PROB_START_ITER=${COMMAND_ONLY_ENV_PROB_START_ITER:-2500}
COMMAND_ONLY_ENV_PROB_END_ITER=${COMMAND_ONLY_ENV_PROB_END_ITER:-3500}
EVAL_COMMAND_ONLY_ENV_PROB=${EVAL_COMMAND_ONLY_ENV_PROB:-1.0}
EXTERNAL_GOAL_PROB_START=${EXTERNAL_GOAL_PROB_START:-0.0}
EXTERNAL_GOAL_PROB_END=${EXTERNAL_GOAL_PROB_END:-0.25}
EXTERNAL_GOAL_PROB_START_ITER=${EXTERNAL_GOAL_PROB_START_ITER:-2500}
EXTERNAL_GOAL_PROB_END_ITER=${EXTERNAL_GOAL_PROB_END_ITER:-${NUM_LEARNING_ITERATIONS}}
EXTERNAL_GOAL_PROB_RAMP_RESETS=${EXTERNAL_GOAL_PROB_RAMP_RESETS:-150000}
EVAL_EXTERNAL_GOAL_PROB=${EVAL_EXTERNAL_GOAL_PROB:-1.0}
EXTERNAL_GOAL_RANGE_RAMP_RESETS=${EXTERNAL_GOAL_RANGE_RAMP_RESETS:-${EXTERNAL_GOAL_PROB_RAMP_RESETS}}
EXTERNAL_GOAL_RANGE_START_ITER=${EXTERNAL_GOAL_RANGE_START_ITER:-2500}
EXTERNAL_GOAL_RANGE_END_ITER=${EXTERNAL_GOAL_RANGE_END_ITER:-${NUM_LEARNING_ITERATIONS}}
EXTERNAL_GOAL_SAMPLING_MODE=${EXTERNAL_GOAL_SAMPLING_MODE:-annulus}
EXTERNAL_GOAL_RADIUS_MIN_START=${EXTERNAL_GOAL_RADIUS_MIN_START:-1.00}
EXTERNAL_GOAL_RADIUS_MAX_START=${EXTERNAL_GOAL_RADIUS_MAX_START:-1.70}
EXTERNAL_GOAL_RADIUS_MIN=${EXTERNAL_GOAL_RADIUS_MIN:-1.00}
EXTERNAL_GOAL_RADIUS_MAX=${EXTERNAL_GOAL_RADIUS_MAX:-3.40}
EXTERNAL_GOAL_POS_LOCAL_MIN_START=${EXTERNAL_GOAL_POS_LOCAL_MIN_START:-"[1.00, -0.20, 0.185]"}
EXTERNAL_GOAL_POS_LOCAL_MAX_START=${EXTERNAL_GOAL_POS_LOCAL_MAX_START:-"[1.25, 0.20, 0.185]"}
EXTERNAL_GOAL_POS_LOCAL_MIN=${EXTERNAL_GOAL_POS_LOCAL_MIN:-"[1.00, -0.75, 0.185]"}
EXTERNAL_GOAL_POS_LOCAL_MAX=${EXTERNAL_GOAL_POS_LOCAL_MAX:-"[1.75, 0.75, 0.185]"}

TEACHER_REF_RUN_ID="5vlz6pj8"
TEACHER_REF_MODEL_FILE="model_24000.pt"
TEACHER_REF_LOCAL_CHECKPOINT="${SCRIPT_DIR}/.teacher_checkpoints/${TEACHER_REF_MODEL_FILE}"
TEACHER_REF_PERCEPTION_PRESET="heightmap"
TEACHER_U5LGUXVL_RUN_ID="u5lguxvl"
TEACHER_U5LGUXVL_MODEL_FILE="model_17000.pt"
TEACHER_U5LGUXVL_LOCAL_CHECKPOINT="${SCRIPT_DIR}/.teacher_checkpoints/${TEACHER_U5LGUXVL_MODEL_FILE}"
TEACHER_COMPAT_PROFILE_RESOLVED="${TEACHER_COMPAT_PROFILE}"
TEACHER_COMPAT_NOTES_AUTO=""

if [[ "${TEACHER_CHECKPOINT}" == https://wandb.ai/*/runs/* ]]; then
  TEACHER_CHECKPOINT="$(normalize_checkpoint_ref "${TEACHER_CHECKPOINT}")"
fi

append_teacher_compat_note() {
  local note="$1"
  if [[ -z "${note}" ]]; then
    return
  fi
  if [[ -z "${TEACHER_COMPAT_NOTES_AUTO}" ]]; then
    TEACHER_COMPAT_NOTES_AUTO="${note}"
  else
    TEACHER_COMPAT_NOTES_AUTO="${TEACHER_COMPAT_NOTES_AUTO}; ${note}"
  fi
}

if [[ "${TEACHER_COMPAT_PROFILE_RESOLVED}" == "auto" ]]; then
  if [[ "${TEACHER_CHECKPOINT}" == *"${TEACHER_REF_RUN_ID}"* || "${TEACHER_CHECKPOINT}" == "${TEACHER_REF_LOCAL_CHECKPOINT}" ]]; then
    TEACHER_COMPAT_PROFILE_RESOLVED="soft_5vlz6pj8"
  elif [[ "${TEACHER_CHECKPOINT}" == *"${TEACHER_U5LGUXVL_RUN_ID}"* || "${TEACHER_CHECKPOINT}" == "${TEACHER_U5LGUXVL_LOCAL_CHECKPOINT}" ]]; then
    TEACHER_COMPAT_PROFILE_RESOLVED="u5lguxvl_generalist"
  else
    TEACHER_COMPAT_PROFILE_RESOLVED="none"
  fi
fi

case "${TEACHER_COMPAT_PROFILE_RESOLVED}" in
  none)
    ;;
  soft_5vlz6pj8)
    if [[ "${TEACHER_OBS_KEYS_EXPLICIT}" -eq 0 ]]; then
      TEACHER_OBS_KEYS="actor_obs_teacher_compat"
    fi
    if [[ "${TEACHER_PERCEPTION_PRESET_EXPLICIT}" -eq 0 ]]; then
      TEACHER_PERCEPTION_PRESET="${TEACHER_REF_PERCEPTION_PRESET}"
    fi
    if [[ "${TEACHER_PERCEPTION_OBS_KEY_EXPLICIT}" -eq 0 ]]; then
      TEACHER_PERCEPTION_OBS_KEY="teacher_perception_obs"
    fi
    if [[ "${TEACHER_ACTOR_OBS_HISTORY_LENGTH_EXPLICIT}" -eq 0 ]]; then
      TEACHER_ACTOR_OBS_HISTORY_LENGTH=""
    fi
    append_teacher_compat_note "teacher_obs_keys defaulted to actor_obs_teacher_compat for exact legacy ordering"
    append_teacher_compat_note "teacher now consumes ${TEACHER_PERCEPTION_PRESET} via ${TEACHER_PERCEPTION_OBS_KEY}"
    ;;
  u5lguxvl_generalist)
    if [[ "${TEACHER_OBS_KEYS_EXPLICIT}" -eq 0 ]]; then
      TEACHER_OBS_KEYS="actor_obs_legacy"
      append_teacher_compat_note "teacher_obs_keys defaulted to actor_obs_legacy to match u5lguxvl legacy object-target schema"
    else
      append_teacher_compat_note "teacher_obs_keys explicitly set to ${TEACHER_OBS_KEYS} for u5lguxvl teacher compatibility"
    fi
    if [[ "${TEACHER_PERCEPTION_PRESET_EXPLICIT}" -eq 0 ]]; then
      TEACHER_PERCEPTION_PRESET="none"
    fi
    if [[ "${TEACHER_PERCEPTION_OBS_KEY_EXPLICIT}" -eq 0 ]]; then
      TEACHER_PERCEPTION_OBS_KEY=""
    fi
    if [[ "${TEACHER_ACTOR_OBS_HISTORY_LENGTH_EXPLICIT}" -eq 0 ]]; then
      TEACHER_ACTOR_OBS_HISTORY_LENGTH="5"
      append_teacher_compat_note "teacher actor observation history length set to ${TEACHER_ACTOR_OBS_HISTORY_LENGTH} for u5lguxvl teacher checkpoint compatibility"
    else
      append_teacher_compat_note "teacher actor observation history length explicitly set to ${TEACHER_ACTOR_OBS_HISTORY_LENGTH} for u5lguxvl teacher checkpoint compatibility"
    fi
    append_teacher_compat_note "teacher perception disabled to match u5lguxvl teacher"
    ;;
  *)
    echo "Unknown TEACHER_COMPAT_PROFILE: ${TEACHER_COMPAT_PROFILE_RESOLVED}" >&2
    exit 1
    ;;
esac

if [[ -n "${TEACHER_COMPAT_NOTES_AUTO}" ]]; then
  if [[ -n "${TEACHER_COMPAT_NOTES}" ]]; then
    TEACHER_COMPAT_NOTES="${TEACHER_COMPAT_NOTES}; ${TEACHER_COMPAT_NOTES_AUTO}"
  else
    TEACHER_COMPAT_NOTES="${TEACHER_COMPAT_NOTES_AUTO}"
  fi
fi
if [[ "${SHOO7SR1_NEAR03_DEBUG}" == "1" ]]; then
  TEACHER_COMPAT_NOTES="teacher_obs_keys defaulted to actor_obs to match u5lguxvl teacher; teacher perception disabled to match u5lguxvl teacher; actor_obs history length set to 5 to match teacher checkpoint"
fi

if [[ "${TOTAL_NUM_ENVS_EXPLICIT}" -eq 1 ]]; then
  if ! [[ "${TOTAL_NUM_ENVS}" =~ ^[0-9]+$ ]] || (( TOTAL_NUM_ENVS < NPROC )); then
    echo "[ERROR] TOTAL_NUM_ENVS must be an integer >= NPROC. Got TOTAL_NUM_ENVS=${TOTAL_NUM_ENVS:-<empty>} NPROC=${NPROC}" >&2
    exit 1
  fi
  if (( TOTAL_NUM_ENVS % NPROC != 0 )); then
    echo "[ERROR] TOTAL_NUM_ENVS must be divisible by NPROC so per-GPU envs are exact. Got TOTAL_NUM_ENVS=${TOTAL_NUM_ENVS} NPROC=${NPROC}" >&2
    exit 1
  fi
  PER_GPU_ENVS=$((TOTAL_NUM_ENVS / NPROC))
  NUM_ENVS="${TOTAL_NUM_ENVS}"
else
  PER_GPU_ENVS="${NUM_ENVS}"
  if ! [[ "${PER_GPU_ENVS}" =~ ^[0-9]+$ ]] || (( PER_GPU_ENVS < 1 )); then
    echo "[ERROR] NUM_ENVS/PER_GPU_ENVS must be a positive per-GPU env count. Got: ${PER_GPU_ENVS:-<empty>}" >&2
    exit 1
  fi
  NUM_ENVS=$((PER_GPU_ENVS * NPROC))
fi

echo "[INFO] teacher_checkpoint=${TEACHER_CHECKPOINT}"
echo "[INFO] teacher_compat_profile=${TEACHER_COMPAT_PROFILE_RESOLVED}"
echo "[INFO] teacher_obs_keys=${TEACHER_OBS_KEYS}"
echo "[INFO] teacher_perception_preset=${TEACHER_PERCEPTION_PRESET} teacher_perception_obs_key=${TEACHER_PERCEPTION_OBS_KEY}"
echo "[INFO] critic_perception_preset=${CRITIC_PERCEPTION_PRESET} critic_perception_obs_key=${CRITIC_PERCEPTION_OBS_KEY}"
if [[ "${PERCEPTION_INTO_CRITIC_MODULES_EXPLICIT}" -eq 1 ]]; then
  echo "[INFO] perception_into_critic_modules=${PERCEPTION_INTO_CRITIC_MODULES}"
else
  echo "[INFO] perception_into_critic_modules=<preset default>"
fi
if [[ -n "${TEACHER_ACTOR_OBS_HISTORY_LENGTH}" ]]; then
  echo "[INFO] teacher_actor_obs_history_length=${TEACHER_ACTOR_OBS_HISTORY_LENGTH}"
fi
if [[ -n "${STUDENT_PROPRIO_HISTORY_LENGTH:-}" ]]; then
  echo "[INFO] student_proprio_history_length=${STUDENT_PROPRIO_HISTORY_LENGTH}"
fi
if [[ -n "${CRITIC_PROPRIO_HISTORY_LENGTH:-}" ]]; then
  echo "[INFO] critic_proprio_history_length=${CRITIC_PROPRIO_HISTORY_LENGTH}"
fi
echo "[INFO] run_name=${RUN_NAME} training_name=${TRAINING_NAME}"
echo "[INFO] exp=${EXP} perception=${PERCEPTION_PRESET}"
echo "[INFO] export_onnx=${EXPORT_ONNX}"
echo "[INFO] shoo7sr1_near03_debug=${SHOO7SR1_NEAR03_DEBUG}"
if [[ -n "${CAMERA_PITCH_DEG}" ]]; then
  echo "[INFO] camera_pitch_deg=${CAMERA_PITCH_DEG}"
else
  echo "[INFO] camera_pitch_deg=<preset default>"
fi
if [[ -n "${CAMERA_FAR}" || -n "${CAMERA_MAX_DISTANCE}" ]]; then
  echo "[INFO] camera_far=${CAMERA_FAR:-<preset default>} camera_max_distance=${CAMERA_MAX_DISTANCE:-<preset default>}"
fi
echo "[INFO] camera_apply_sensor_noise=${CAMERA_APPLY_SENSOR_NOISE}"
if [[ -n "${OBJECT_GEOMETRY_MODE_NORM}" ]]; then
  echo "[INFO] object_geometry_mode=${OBJECT_GEOMETRY_MODE_NORM} simulator_object_spawn_mode=${HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE}"
else
  echo "[INFO] object_geometry_mode=<default>"
fi
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES} nproc=${NPROC} per_gpu_envs=${PER_GPU_ENVS} total_num_envs=${NUM_ENVS}"
echo "[INFO] data_mode=${DATA_MODE}"
echo "[INFO] tracker_profile=${TRACKER_PROFILE}"
echo "[INFO] root_command_mode=${ROOT_COMMAND_MODE}"
echo "[INFO] contact_aware_history=${CONTACT_AWARE_HISTORY} history_length=${CONTACT_AWARE_HISTORY_LENGTH}"
echo "[INFO] use_legacy_ds=${LEGACY_DS_ENABLED} prepared=${LEGACY_DS_PREPARED} ds_data_root=${DS_DATA_ROOT}"
if [[ -n "${MOTION_DIR:-}" ]]; then
  echo "[INFO] motion_dir=${MOTION_DIR}"
fi
if [[ "${USING_TEACHER_ROLLOUT_MOTION_BANK}" -eq 1 ]]; then
  echo "[INFO] using_teacher_rollout_motion_bank=1"
fi
if [[ -n "${TEACHER_ROLLOUT_FILTER_CLIP_COUNT}" ]]; then
  echo "[INFO] teacher_rollout_success_filter=${TEACHER_ROLLOUT_SUCCESS_CLIPS_FILE} kept=${TEACHER_ROLLOUT_FILTER_CLIP_COUNT} contact_clips=${TEACHER_ROLLOUT_FILTER_CONTACT_COUNT} source=${TEACHER_ROLLOUT_FILTER_SOURCE_DIR}"
fi
if [[ -n "${ACTIVE_OBJECT_BANK_INFO}" ]]; then
  IFS='|' read -r ACTIVE_OBJECT_BANK_CLIPS ACTIVE_OBJECT_BANK_URDFS ACTIVE_OBJECT_BANK_BOX_MOTION ACTIVE_OBJECT_BANK_BOX_URDFS ACTIVE_OBJECT_BANK_OMOMO_MOTION ACTIVE_OBJECT_BANK_OMOMO_URDFS ACTIVE_OBJECT_BANK_TOP <<< "${ACTIVE_OBJECT_BANK_INFO}"
  echo "[INFO] active_object_bank=${ACTIVE_OBJECT_BANK_CLIPS} clips ${ACTIVE_OBJECT_BANK_URDFS} unique_urdfs (box_motion=${ACTIVE_OBJECT_BANK_BOX_MOTION} box_urdfs=${ACTIVE_OBJECT_BANK_BOX_URDFS} omomo_motion=${ACTIVE_OBJECT_BANK_OMOMO_MOTION} omomo_urdfs=${ACTIVE_OBJECT_BANK_OMOMO_URDFS} top=${ACTIVE_OBJECT_BANK_TOP})"
fi
if [[ -n "${HOLOSOMA_OBJECT_BANK_TOTAL_MOTION_COUNT:-}" ]]; then
  echo "[INFO] object_bank_counts total_motion=${HOLOSOMA_OBJECT_BANK_TOTAL_MOTION_COUNT} total_urdfs=${HOLOSOMA_OBJECT_BANK_TOTAL_UNIQUE_URDF_COUNT} box_motion=${HOLOSOMA_OBJECT_BANK_BOX_MOTION_COUNT} box_urdfs=${HOLOSOMA_OBJECT_BANK_BOX_UNIQUE_URDF_COUNT} omomo_motion=${HOLOSOMA_OBJECT_BANK_OMOMO_MOTION_COUNT} omomo_urdfs=${HOLOSOMA_OBJECT_BANK_OMOMO_UNIQUE_URDF_COUNT}"
fi
if [[ -n "${OLD_TRACKER_CLIP_COUNT}" ]]; then
  echo "[INFO] old_tracker_numeric_box_clip_count=${OLD_TRACKER_CLIP_COUNT} max_box_id=${OLD_TRACKER_MAX_BOX_ID}"
fi
if [[ "${DATA_MODE}" == "pure-real" ]]; then
  echo "[INFO] DATA_MODE=pure-real uses the mixed bank but samples only OMOMO clips."
  echo "[INFO] OMOMO clip prefixes=${PURE_REAL_OMOMO_PREFIXES}"
fi
echo "[INFO] schedule_variant=${SCHEDULE_VARIANT}"
echo "[INFO] schedule_name=${SCHEDULE_NAME}"
echo "[INFO] schedule_notes=${SCHEDULE_NOTES}"
echo "[INFO] student_actor_inputs=${STUDENT_ACTOR_INPUTS}"
echo "[INFO] num_learning_iterations=${NUM_LEARNING_ITERATIONS}"
echo "[INFO] bc_loss_coef=${BC_LOSS_COEF} dagger_loss_coef=${DAGGER_LOSS_COEF} teacher_action_mix_ratio=${TEACHER_ACTION_MIX_RATIO}"
if [[ -n "${TEACHER_ACTION_MIX_RATIO_START}" || -n "${TEACHER_ACTION_MIX_RATIO_END}" || -n "${TEACHER_ACTION_MIX_RATIO_END_ITERATION}" ]]; then
  echo "[INFO] teacher_action_mix_schedule=${TEACHER_ACTION_MIX_RATIO_START}->${TEACHER_ACTION_MIX_RATIO_END} end_iter=${TEACHER_ACTION_MIX_RATIO_END_ITERATION}"
fi
echo "[INFO] ppo_schedule=${PPO_START_EPOCH}->${DAGGER_END_EPOCH} start=${PPO_START_COEFF} target=${PPO_TARGET_COEFF} step_epochs=${PPO_SCHEDULE_STEP_EPOCHS} dagger_loss_coef=${DAGGER_LOSS_COEF}"
if [[ -n "${PPO_START_NOISE_STD}" ]]; then
  echo "[INFO] ppo_start_noise_std=${PPO_START_NOISE_STD} until_coeff=${PPO_START_NOISE_STD_UNTIL_COEFF}"
else
  echo "[INFO] ppo_start_noise_std=<disabled>"
fi
echo "[INFO] fixed_bc_eval_log_interval=${FIXED_BC_EVAL_LOG_INTERVAL}"
echo "[INFO] use_adaptive_timesteps_sampler=${USE_ADAPTIVE_TIMESTEPS_SAMPLER}"
echo "[INFO] adaptive_sampling_contact_interval_root=${ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT}"
echo "[INFO] uniform_t1_window_sampling=${UNIFORM_T1_WINDOW_SAMPLING_ENABLED} half_width=${UNIFORM_T1_WINDOW_HALF_WIDTH_STEPS} density_boost=${UNIFORM_T1_WINDOW_DENSITY_BOOST}"
echo "[INFO] command_only_env_prob=${COMMAND_ONLY_ENV_PROB_START}->${COMMAND_ONLY_ENV_PROB_END} iter=${COMMAND_ONLY_ENV_PROB_START_ITER}->${COMMAND_ONLY_ENV_PROB_END_ITER}"
echo "[INFO] sparse_goal_enabled=${SPARSE_GOAL_ENABLED} ext_prob=${EXTERNAL_GOAL_PROB_START}->${EXTERNAL_GOAL_PROB_END}"
echo "[INFO] external_goal_prob_iter=${EXTERNAL_GOAL_PROB_START_ITER}->${EXTERNAL_GOAL_PROB_END_ITER}"
echo "[INFO] external_goal_sampling_mode=${EXTERNAL_GOAL_SAMPLING_MODE}"
echo "[INFO] external_goal_radius_start=${EXTERNAL_GOAL_RADIUS_MIN_START} -> ${EXTERNAL_GOAL_RADIUS_MAX_START}"
echo "[INFO] external_goal_radius_end=${EXTERNAL_GOAL_RADIUS_MIN} -> ${EXTERNAL_GOAL_RADIUS_MAX}"
echo "[INFO] external_goal_range_xy_start=${EXTERNAL_GOAL_POS_LOCAL_MIN_START} -> ${EXTERNAL_GOAL_POS_LOCAL_MAX_START}"
echo "[INFO] external_goal_range_xy_end=${EXTERNAL_GOAL_POS_LOCAL_MIN} -> ${EXTERNAL_GOAL_POS_LOCAL_MAX}"
echo "[INFO] external_goal_range_iter=${EXTERNAL_GOAL_RANGE_START_ITER}->${EXTERNAL_GOAL_RANGE_END_ITER}"
echo "[INFO] start_at_timestep_zero_prob=${START_AT_TIMESTEP_ZERO_PROB}->${START_AT_TIMESTEP_ZERO_PROB_END} iter=${START_AT_TIMESTEP_ZERO_PROB_START_ITER}->${START_AT_TIMESTEP_ZERO_PROB_END_ITER}"
echo "[INFO] freeze_at_timestep_zero_prob=${FREEZE_AT_TIMESTEP_ZERO_PROB}->${FREEZE_AT_TIMESTEP_ZERO_PROB_END} iter=${FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER}->${FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER}"
echo "[INFO] entropy_coef=${ENTROPY_COEF} dagger_match_std=${DAGGER_MATCH_STD}"
echo "[INFO] default_pose_prepend=${ENABLE_DEFAULT_POSE_PREPEND} duration_s=${DEFAULT_POSE_PREPEND_DURATION_S} default_pose_append=${ENABLE_DEFAULT_POSE_APPEND} append_duration_s=${DEFAULT_POSE_APPEND_DURATION_S}"
echo "[INFO] viser_distill_minimal_ui=${VISER_DISTILL_MINIMAL_UI}"
echo "[INFO] viser_show_target_keypoints=${VISER_SHOW_TARGET_KEYPOINTS}"
echo "[INFO] dagger_ignore_external_goal_samples=${DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES}"
echo "[INFO] dagger_ignore_episode_initial_steps=${DAGGER_IGNORE_EPISODE_INITIAL_STEPS}"
echo "[INFO] max_episode_length_s=${MAX_EPISODE_LENGTH_S}"
if [[ -n "${TEACHER_COMPAT_NOTES}" ]]; then
  echo "[WARN] teacher_compat_notes=${TEACHER_COMPAT_NOTES}"
fi

EXTRA_DISTILL_ARGS=()
if [[ -n "${TEACHER_ACTOR_OBS_HISTORY_LENGTH}" ]]; then
  IFS=',' read -r -a _teacher_obs_key_list <<< "${TEACHER_OBS_KEYS}"
  for _raw_teacher_obs_key in "${_teacher_obs_key_list[@]}"; do
    _teacher_obs_group="$(echo "${_raw_teacher_obs_key}" | tr -d "[]'\"[:space:]")"
    if [[ -n "${_teacher_obs_group}" && "${_teacher_obs_group}" == actor_obs* ]]; then
      EXTRA_DISTILL_ARGS+=(
        --observation.groups."${_teacher_obs_group}".history-length="${TEACHER_ACTOR_OBS_HISTORY_LENGTH}"
      )
    fi
  done
fi
if [[ -n "${STUDENT_PROPRIO_HISTORY_LENGTH:-}" ]]; then
  EXTRA_DISTILL_ARGS+=(
    --observation.groups.actor_obs_proprio.history-length="${STUDENT_PROPRIO_HISTORY_LENGTH}"
    --observation.groups.actor_obs_proprio_no_linvel.history-length="${STUDENT_PROPRIO_HISTORY_LENGTH}"
  )
fi
if [[ -n "${CRITIC_PROPRIO_HISTORY_LENGTH:-}" ]]; then
  EXTRA_DISTILL_ARGS+=(
    --observation.groups.critic_proprio_history.history-length="${CRITIC_PROPRIO_HISTORY_LENGTH}"
  )
fi
if [[ -n "${PPO_START_NOISE_STD}" ]]; then
  EXTRA_DISTILL_ARGS+=(
    --algo.config.distill.ppo-start-noise-std="${PPO_START_NOISE_STD}"
    --algo.config.distill.ppo-start-noise-std-until-coeff="${PPO_START_NOISE_STD_UNTIL_COEFF}"
  )
fi
if [[ "${EXPORT_ONNX_EXPLICIT}" -eq 1 || "${EXPORT_ONNX}" == "False" || "${EXPORT_ONNX}" == "false" ]]; then
  EXTRA_DISTILL_ARGS+=(--training.export-onnx="${EXPORT_ONNX}")
fi
if [[ "${SHOO7SR1_NEAR03_DEBUG}" == "1" ]]; then
  EXTRA_DISTILL_ARGS+=(
    --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.root-lin-vel="[0.5, 0.5, 0.2]"
    --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.root-ang-vel="[0.52, 0.52, 0.78]"
  )
fi
if [[ "${DATA_MODE}" == "pure-real" ]]; then
  EXTRA_DISTILL_ARGS+=(
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.enabled=True
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.clean-clip-name-prefixes="${PURE_REAL_OMOMO_PREFIXES}"
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.stage-start-iterations='[0]'
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.clean-group-probabilities='[1.0]'
  )
fi

PERCEPTION_OVERRIDE_ARGS=()
if [[ "${PERCEPTION_PRESET}" != "none" ]]; then
  if [[ "${IMAGE_WIDTH_EXPLICIT}" -eq 1 ]]; then
    PERCEPTION_OVERRIDE_ARGS+=(--perception.camera-width="${IMAGE_WIDTH}")
  fi
  if [[ "${IMAGE_HEIGHT_EXPLICIT}" -eq 1 ]]; then
    PERCEPTION_OVERRIDE_ARGS+=(--perception.camera-height="${IMAGE_HEIGHT}")
  fi
  if [[ -n "${CAMERA_PITCH_DEG}" ]]; then
    PERCEPTION_OVERRIDE_ARGS+=(--perception.camera-pitch-deg="${CAMERA_PITCH_DEG}")
  fi
  if [[ "${PERCEPTION_INTO_CRITIC_MODULES_EXPLICIT}" -eq 1 ]]; then
    PERCEPTION_OVERRIDE_ARGS+=(--perception.inject-into-critic-modules="${PERCEPTION_INTO_CRITIC_MODULES}")
  fi
  if [[ "${CAMERA_FAR_EXPLICIT}" -eq 1 ]]; then
    PERCEPTION_OVERRIDE_ARGS+=(--perception.camera-far="${CAMERA_FAR}")
  fi
  if [[ "${CAMERA_MAX_DISTANCE_EXPLICIT}" -eq 1 ]]; then
    PERCEPTION_OVERRIDE_ARGS+=(--perception.max-distance="${CAMERA_MAX_DISTANCE}")
  fi
  if [[ "${PERCEPTION_WARP_PREPROCESS_EXPLICIT}" -eq 1 ]]; then
    PERCEPTION_OVERRIDE_ARGS+=(--perception.camera-warp-preprocess="${PERCEPTION_WARP_PREPROCESS}")
  fi
  if [[ "${CAMERA_APPLY_SENSOR_NOISE_EXPLICIT}" -eq 1 || "${PERCEPTION_PRESET}" == camera_depth_* ]]; then
    PERCEPTION_OVERRIDE_ARGS+=(--perception.camera-apply-sensor-noise="${CAMERA_APPLY_SENSOR_NOISE}")
  fi
  if [[ -n "${PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE}" ]]; then
    PERCEPTION_OVERRIDE_ARGS+=(--perception.object-geometry-mode="${PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE}")
  fi
fi

TEACHER_PERCEPTION_ARGS=(
  --algo.config.distill.teacher-perception-preset="${TEACHER_PERCEPTION_PRESET}"
)
if [[ -n "${TEACHER_PERCEPTION_OBS_KEY}" ]]; then
  TEACHER_PERCEPTION_ARGS+=(
    --algo.config.distill.teacher-perception-obs-key="${TEACHER_PERCEPTION_OBS_KEY}"
  )
fi

CRITIC_PERCEPTION_ARGS=(
  --algo.config.distill.critic-perception-preset="${CRITIC_PERCEPTION_PRESET}"
)
if [[ -n "${CRITIC_PERCEPTION_OBS_KEY}" ]]; then
  CRITIC_PERCEPTION_ARGS+=(
    --algo.config.distill.critic-perception-obs-key="${CRITIC_PERCEPTION_OBS_KEY}"
  )
fi

OBJECT_GEOMETRY_MODE_ENV=()
if [[ -n "${HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE}" ]]; then
  OBJECT_GEOMETRY_MODE_ENV=(HOLOSOMA_OBJECT_SPAWN_MODE="${HOLOSOMA_OBJECT_SPAWN_MODE_OVERRIDE}")
fi

exec env \
  EXP="${EXP}" \
  RUN_NAME="${RUN_NAME}" \
  TRAINING_NAME="${TRAINING_NAME}" \
  TRAINING_PROJECT="${TRAINING_PROJECT}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  NPROC="${NPROC}" \
  PER_GPU_ENVS="${PER_GPU_ENVS}" \
  TOTAL_NUM_ENVS="${NUM_ENVS}" \
  MOTION_DIR="${MOTION_DIR}" \
  TEACHER_OBS_KEYS="${TEACHER_OBS_KEYS}" \
  TEACHER_ACTION_MIX_RATIO="${TEACHER_ACTION_MIX_RATIO}" \
  TEACHER_ACTION_MIX_RATIO_START="${TEACHER_ACTION_MIX_RATIO_START}" \
  TEACHER_ACTION_MIX_RATIO_END="${TEACHER_ACTION_MIX_RATIO_END}" \
  TEACHER_ACTION_MIX_RATIO_END_ITERATION="${TEACHER_ACTION_MIX_RATIO_END_ITERATION}" \
  BC_LOSS_COEF="${BC_LOSS_COEF}" \
  NUM_LEARNING_ITERATIONS="${NUM_LEARNING_ITERATIONS}" \
  PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY="${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY:-}" \
  PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY="${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY:-}" \
  PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY="${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY:-}" \
  PPO_START_EPOCH="${PPO_START_EPOCH}" \
  DAGGER_END_EPOCH="${DAGGER_END_EPOCH}" \
  PPO_TARGET_COEFF="${PPO_TARGET_COEFF}" \
  DAGGER_LOSS_COEF="${DAGGER_LOSS_COEF}" \
  DAGGER_MATCH_STD="${DAGGER_MATCH_STD}" \
  ENTROPY_COEF="${ENTROPY_COEF}" \
  START_AT_TIMESTEP_ZERO_PROB="${START_AT_TIMESTEP_ZERO_PROB}" \
  START_AT_TIMESTEP_ZERO_PROB_END="${START_AT_TIMESTEP_ZERO_PROB_END}" \
  START_AT_TIMESTEP_ZERO_PROB_START_ITER="${START_AT_TIMESTEP_ZERO_PROB_START_ITER}" \
  START_AT_TIMESTEP_ZERO_PROB_END_ITER="${START_AT_TIMESTEP_ZERO_PROB_END_ITER}" \
  UNIFORM_T1_WINDOW_SAMPLING_ENABLED="${UNIFORM_T1_WINDOW_SAMPLING_ENABLED}" \
  UNIFORM_T1_WINDOW_HALF_WIDTH_STEPS="${UNIFORM_T1_WINDOW_HALF_WIDTH_STEPS}" \
  UNIFORM_T1_WINDOW_DENSITY_BOOST="${UNIFORM_T1_WINDOW_DENSITY_BOOST}" \
  HOLOSOMA_RESET_TO_DEFAULT_POSE="${RESET_TO_DEFAULT_POSE}" \
  ENABLE_DEFAULT_POSE_PREPEND="${ENABLE_DEFAULT_POSE_PREPEND}" \
  DEFAULT_POSE_PREPEND_DURATION_S="${DEFAULT_POSE_PREPEND_DURATION_S}" \
  ENABLE_DEFAULT_POSE_APPEND="${ENABLE_DEFAULT_POSE_APPEND}" \
  DEFAULT_POSE_APPEND_DURATION_S="${DEFAULT_POSE_APPEND_DURATION_S}" \
  VISER_DISTILL_MINIMAL_UI="${VISER_DISTILL_MINIMAL_UI}" \
  VISER_SHOW_TARGET_KEYPOINTS="${VISER_SHOW_TARGET_KEYPOINTS}" \
  PAIR_TERRAIN_WITH_MOTION="${PAIR_TERRAIN_WITH_MOTION}" \
  HOLOSOMA_OBJECT_BANK_TOTAL_MOTION_COUNT="${HOLOSOMA_OBJECT_BANK_TOTAL_MOTION_COUNT:-}" \
  HOLOSOMA_OBJECT_BANK_TOTAL_UNIQUE_URDF_COUNT="${HOLOSOMA_OBJECT_BANK_TOTAL_UNIQUE_URDF_COUNT:-}" \
  HOLOSOMA_OBJECT_BANK_BOX_MOTION_COUNT="${HOLOSOMA_OBJECT_BANK_BOX_MOTION_COUNT:-}" \
  HOLOSOMA_OBJECT_BANK_BOX_UNIQUE_URDF_COUNT="${HOLOSOMA_OBJECT_BANK_BOX_UNIQUE_URDF_COUNT:-}" \
  HOLOSOMA_OBJECT_BANK_OMOMO_MOTION_COUNT="${HOLOSOMA_OBJECT_BANK_OMOMO_MOTION_COUNT:-}" \
  HOLOSOMA_OBJECT_BANK_OMOMO_UNIQUE_URDF_COUNT="${HOLOSOMA_OBJECT_BANK_OMOMO_UNIQUE_URDF_COUNT:-}" \
  HOLOSOMA_OBJECT_BANK_MOTION_DIR="${HOLOSOMA_OBJECT_BANK_MOTION_DIR:-}" \
  HOLOSOMA_OBJECT_BANK_OBJECT_MAP="${HOLOSOMA_OBJECT_BANK_OBJECT_MAP:-}" \
  "${OBJECT_GEOMETRY_MODE_ENV[@]}" \
  bash "${SCRIPT_DIR}/distill_root_box.sh" "${TEACHER_CHECKPOINT}" \
    "perception:${PERCEPTION_PRESET}" \
    --algo.config.module-dict.actor.input-dim "${STUDENT_ACTOR_INPUTS}" \
    --algo.config.distill.schedule-name="${SCHEDULE_NAME}" \
    --algo.config.distill.schedule-notes="${SCHEDULE_NOTES}" \
    --algo.config.distill.teacher-compat-profile="${TEACHER_COMPAT_PROFILE_RESOLVED}" \
    --algo.config.distill.teacher-compat-notes="${TEACHER_COMPAT_NOTES}" \
    "${TEACHER_PERCEPTION_ARGS[@]}" \
    "${CRITIC_PERCEPTION_ARGS[@]}" \
    --algo.config.distill.dagger-ignore-external-goal-samples="${DAGGER_IGNORE_EXTERNAL_GOAL_SAMPLES}" \
    --algo.config.distill.dagger-ignore-episode-initial-steps="${DAGGER_IGNORE_EPISODE_INITIAL_STEPS}" \
    --algo.config.distill.fixed-bc-eval-log-interval="${FIXED_BC_EVAL_LOG_INTERVAL}" \
    --algo.config.distill.ppo-start-coeff="${PPO_START_COEFF}" \
    --algo.config.distill.ppo-target-coeff="${PPO_TARGET_COEFF}" \
    --algo.config.distill.ppo-schedule-step-epochs="${PPO_SCHEDULE_STEP_EPOCHS}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.enabled="${SPARSE_GOAL_ENABLED}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.clip-goal-delta-min-steps="${CLIP_GOAL_DELTA_MIN_STEPS}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.clip-goal-delta-max-steps="${CLIP_GOAL_DELTA_MAX_STEPS}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.command-only-env-prob-start="${COMMAND_ONLY_ENV_PROB_START}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.command-only-env-prob-end="${COMMAND_ONLY_ENV_PROB_END}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.command-only-env-prob-start-iter="${COMMAND_ONLY_ENV_PROB_START_ITER}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.command-only-env-prob-end-iter="${COMMAND_ONLY_ENV_PROB_END_ITER}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.eval-command-only-env-prob="${EVAL_COMMAND_ONLY_ENV_PROB}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-prob-start="${EXTERNAL_GOAL_PROB_START}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-prob-end="${EXTERNAL_GOAL_PROB_END}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-prob-start-iter="${EXTERNAL_GOAL_PROB_START_ITER}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-prob-end-iter="${EXTERNAL_GOAL_PROB_END_ITER}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-prob-ramp-resets="${EXTERNAL_GOAL_PROB_RAMP_RESETS}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.eval-external-goal-prob="${EVAL_EXTERNAL_GOAL_PROB}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-range-ramp-resets="${EXTERNAL_GOAL_RANGE_RAMP_RESETS}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-range-start-iter="${EXTERNAL_GOAL_RANGE_START_ITER}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-range-end-iter="${EXTERNAL_GOAL_RANGE_END_ITER}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-sampling-mode="${EXTERNAL_GOAL_SAMPLING_MODE}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-radius-min-start="${EXTERNAL_GOAL_RADIUS_MIN_START}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-radius-max-start="${EXTERNAL_GOAL_RADIUS_MAX_START}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-radius-min="${EXTERNAL_GOAL_RADIUS_MIN}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-radius-max="${EXTERNAL_GOAL_RADIUS_MAX}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-pos-local-min-start "${EXTERNAL_GOAL_POS_LOCAL_MIN_START}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-pos-local-max-start "${EXTERNAL_GOAL_POS_LOCAL_MAX_START}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-pos-local-min "${EXTERNAL_GOAL_POS_LOCAL_MIN}" \
    --command.setup-terms.motion-command.params.motion-config.sparse-object-goal.external-goal-pos-local-max "${EXTERNAL_GOAL_POS_LOCAL_MAX}" \
    --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler="${USE_ADAPTIVE_TIMESTEPS_SAMPLER}" \
    --command.setup-terms.motion-command.params.motion-config.adaptive-sampling-contact-interval-root="${ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT}" \
    --command.setup-terms.motion-command.params.motion-config.contact-aware-carry-window-mode="${CONTACT_AWARE_CARRY_WINDOW_MODE}" \
    --command.setup-terms.motion-command.params.motion-config.contact-aware-peak-height-alpha="${CONTACT_AWARE_PEAK_HEIGHT_ALPHA}" \
    --command.setup-terms.motion-command.params.motion-config.contact-aware-peak-height-smoothing-steps="${CONTACT_AWARE_PEAK_HEIGHT_SMOOTHING_STEPS}" \
    --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-sampling-enabled="${UNIFORM_T1_WINDOW_SAMPLING_ENABLED}" \
    --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-half-width-steps="${UNIFORM_T1_WINDOW_HALF_WIDTH_STEPS}" \
    --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-density-boost="${UNIFORM_T1_WINDOW_DENSITY_BOOST}" \
    --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end="${START_AT_TIMESTEP_ZERO_PROB_END}" \
    --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-start-iter="${START_AT_TIMESTEP_ZERO_PROB_START_ITER}" \
    --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end-iter="${START_AT_TIMESTEP_ZERO_PROB_END_ITER}" \
    --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob="${FREEZE_AT_TIMESTEP_ZERO_PROB}" \
    --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end="${FREEZE_AT_TIMESTEP_ZERO_PROB_END}" \
    --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-start-iter="${FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER}" \
    --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end-iter="${FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER}" \
    --simulator.config.sim.max_episode_length_s "${MAX_EPISODE_LENGTH_S}" \
    "${EXTRA_DISTILL_ARGS[@]}" \
    "${PERCEPTION_OVERRIDE_ARGS[@]}" \
    "$@"
