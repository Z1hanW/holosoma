#!/usr/bin/env bash
set -euo pipefail

# Teacher-policy inference for box tracking on the same motion-box pairs used by
# train_object_generalist.sh, with Isaac Sim <-> Viser sync enabled.
#
# Usage:
#   bash infer_box_tracking.sh [teacher_checkpoint.pt|wandb://...] [extra tyro args...]
#
# Optional env vars:
#   TEACHER_CHECKPOINT        (default: wandb://zihanw22/boxer/a5ohxuta/model_09000.pt)
#   LEGACY_OBS                (default: 0; set 1/true to require legacy checkpoint observation layout)
#   REQUIRE_HEIGHTMAP         (default: 0; set 1/true to require checkpoint perception.enabled=True and output_mode=heightmap)
#   DEFAULT_LEGACY_TEACHER_CHECKPOINT
#                             (optional; used as default checkpoint when LEGACY_OBS=1 and no checkpoint is explicitly provided)
#   INFER_DATASET             (default: mixed; options: omomo|behave|behave_carry|behave_sq_carry|mixed)
#   MOTION_DIR                (optional override; if unset, chosen by INFER_DATASET)
#   MOTION_CLIP_NAME          (optional: pin a single clip)
#   OBJECT_URDF               (optional override; if unset, chosen by INFER_DATASET)
#   NUM_ENVS                  (default: 1)
#   HEADLESS                  (default: False; set True for headless eval)
#   VISER_PORT                (default: random)
#   VISER_ENV_ID              (default: 0)
#   VISER_UPDATE_HZ           (default: 30)
#   VISER_RECENTER            (default: True)
#   VISER_SYNC_TO_SIM         (default: True)
#   VISER_FORCE_DT            (default: True)
#   VISER_LOAD_URDF           (default: 1; URDF meshes are shown in Viser, but pose/object selection comes from Isaac Sim runtime state)
#   START_AT_TIMESTEP_ZERO_PROB
#                             (default: 0.2; matches checkpoint default)
#   FREEZE_AT_TIMESTEP_ZERO_PROB
#                             (default: 0.95; matches checkpoint default)
#   ENABLE_DEFAULT_POSE_PREPEND
#                             (default: False; disable runtime default-pose warmup for more stable interactive resets)
#   DEFAULT_POSE_PREPEND_DURATION_S
#                             (default: 0.0; only used when ENABLE_DEFAULT_POSE_PREPEND=True)
#   DISABLE_RANDOMIZATION     (default: True)
#   VIS_GPU                   (default: auto; picks least-used GPU if CUDA_VISIBLE_DEVICES is unset)

usage() {
  cat <<'EOF'
Usage:
  bash infer_box_tracking.sh [teacher_checkpoint.pt|wandb://...] [extra tyro args...]

Examples:
  bash infer_box_tracking.sh
  bash infer_box_tracking.sh /abs/path/to/model_17000.pt
  MOTION_CLIP_NAME=sub3_largebox_003_mj_w_obj bash infer_box_tracking.sh

Dataset selection examples:
  INFER_DATASET=omomo bash infer_box_tracking.sh
  INFER_DATASET=behave bash infer_box_tracking.sh
  INFER_DATASET=behave_carry bash infer_box_tracking.sh
  INFER_DATASET=mixed bash infer_box_tracking.sh
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help|help)
      usage
      exit 0
      ;;
  esac
fi

# https://wandb.ai/zihanw22/boxer/runs/a5ohxuta/files?nw=nwuserz1hanw
DEFAULT_TEACHER_CHECKPOINT=${DEFAULT_TEACHER_CHECKPOINT:-"wandb://zihanw22/boxer/a5ohxuta/model_09000.pt"}
DEFAULT_LEGACY_TEACHER_CHECKPOINT="${DEFAULT_LEGACY_TEACHER_CHECKPOINT:-}"
LEGACY_OBS=${LEGACY_OBS:-0}
legacy_obs_normalized=$(echo "${LEGACY_OBS}" | tr '[:upper:]' '[:lower:]')
if [[ "${legacy_obs_normalized}" == "1" || "${legacy_obs_normalized}" == "true" ]]; then
  LEGACY_OBS_ENABLED=1
else
  LEGACY_OBS_ENABLED=0
fi
REQUIRE_HEIGHTMAP=${REQUIRE_HEIGHTMAP:-0}
require_heightmap_normalized=$(echo "${REQUIRE_HEIGHTMAP}" | tr '[:upper:]' '[:lower:]')
if [[ "${require_heightmap_normalized}" == "1" || "${require_heightmap_normalized}" == "true" ]]; then
  HEIGHTMAP_REQUIRED=1
else
  HEIGHTMAP_REQUIRED=0
fi

TEACHER_CHECKPOINT_FROM_ENV=0
if [[ -n "${TEACHER_CHECKPOINT+x}" || -n "${CKPT+x}" ]]; then
  TEACHER_CHECKPOINT_FROM_ENV=1
fi
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${CKPT:-${DEFAULT_TEACHER_CHECKPOINT}}}"
TEACHER_CHECKPOINT_FROM_ARG=0

if [[ $# -gt 0 ]]; then
  if [[ "$1" == wandb://* || "$1" == /* || "$1" == ./* || "$1" == ../* || "$1" == *.pt ]]; then
    TEACHER_CHECKPOINT="$1"
    TEACHER_CHECKPOINT_FROM_ARG=1
    shift
  fi
fi

if [[ "${LEGACY_OBS_ENABLED}" == "1" ]]; then
  if [[ "${TEACHER_CHECKPOINT_FROM_ENV}" != "1" && "${TEACHER_CHECKPOINT_FROM_ARG}" != "1" ]]; then
    if [[ -n "${DEFAULT_LEGACY_TEACHER_CHECKPOINT}" ]]; then
      TEACHER_CHECKPOINT="${DEFAULT_LEGACY_TEACHER_CHECKPOINT}"
    else
      echo "[ERROR] LEGACY_OBS=1 requires an explicit legacy checkpoint." >&2
      echo "[ERROR] Provide TEACHER_CHECKPOINT/CKPT/positional .pt, or set DEFAULT_LEGACY_TEACHER_CHECKPOINT." >&2
      exit 2
    fi
  fi
fi

pick_first_existing_path() {
  local candidate=""
  for candidate in "$@"; do
    if [[ -e "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  if [[ $# -gt 0 ]]; then
    echo "$1"
  fi
}

INFER_DATASET=${INFER_DATASET:-${DATASET:-mixed}}
INFER_DATASET=$(echo "${INFER_DATASET}" | tr '[:upper:]' '[:lower:]' | tr -d '[][:space:]')
case "${INFER_DATASET}" in
  omomo|behave|behave_carry|behave_sq_carry|mixed) ;;
  *)
    echo "[ERROR] INFER_DATASET must be one of: omomo, behave, behave_carry, behave_sq_carry, mixed. Got: ${INFER_DATASET}" >&2
    exit 2
    ;;
esac

DEFAULT_OMOMO_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"
DEFAULT_BEHAVE_MOTION_DIR="$(pick_first_existing_path \
  "${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/behave_carry" \
  "${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/behave_sq_carry")"
DEFAULT_MIXED_MOTION_DIR="$(pick_first_existing_path \
  "${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_carry_aug_mix_ml" \
  "${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_sq_carry_aug_mix_ml")"
DEFAULT_OMOMO_URDF="${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"
DEFAULT_BEHAVE_MAP_FILE="${DEFAULT_BEHAVE_MOTION_DIR}/_clip_object_urdf_map.json"
DEFAULT_MIXED_MAP_FILE="${DEFAULT_MIXED_MOTION_DIR}/_clip_object_urdf_map.json"

MOTION_DIR_FROM_ENV=0
if [[ -n "${MOTION_DIR+x}" ]]; then
  MOTION_DIR_FROM_ENV=1
fi
OBJECT_URDF_FROM_ENV=0
if [[ -n "${OBJECT_URDF+x}" ]]; then
  OBJECT_URDF_FROM_ENV=1
fi

if [[ "${MOTION_DIR_FROM_ENV}" != "1" ]]; then
  case "${INFER_DATASET}" in
    omomo)
      MOTION_DIR="${DEFAULT_OMOMO_MOTION_DIR}"
      ;;
    behave|behave_carry|behave_sq_carry)
      MOTION_DIR="${DEFAULT_BEHAVE_MOTION_DIR}"
      ;;
    mixed)
      MOTION_DIR="${DEFAULT_MIXED_MOTION_DIR}"
      ;;
  esac
fi

MOTION_CLIP_NAME=${MOTION_CLIP_NAME:-}
if [[ "${OBJECT_URDF_FROM_ENV}" != "1" ]]; then
  case "${INFER_DATASET}" in
    omomo)
      OBJECT_URDF="${DEFAULT_OMOMO_URDF}"
      ;;
    behave|behave_carry|behave_sq_carry)
      if [[ -f "${DEFAULT_BEHAVE_MAP_FILE}" ]]; then
        OBJECT_URDF="${DEFAULT_BEHAVE_MAP_FILE}"
      else
        echo "[ERROR] BEHAVE map file not found: ${DEFAULT_BEHAVE_MAP_FILE}" >&2
        exit 2
      fi
      ;;
    mixed)
      if [[ -f "${DEFAULT_MIXED_MAP_FILE}" ]]; then
        OBJECT_URDF="${DEFAULT_MIXED_MAP_FILE}"
      else
        echo "[ERROR] Mixed map file not found: ${DEFAULT_MIXED_MAP_FILE}" >&2
        exit 2
      fi
      ;;
  esac
fi

NUM_ENVS=${NUM_ENVS:-1}
HEADLESS_RAW=${HEADLESS:-False}
HEADLESS_NORM=$(echo "${HEADLESS_RAW}" | tr '[:upper:]' '[:lower:]')
case "${HEADLESS_NORM}" in
  1|true|yes|on)
    HEADLESS_FLAG=True
    export HEADLESS=1
    ;;
  0|false|no|off|"")
    HEADLESS_FLAG=False
    export HEADLESS=0
    ;;
  *)
    echo "[ERROR] HEADLESS must be one of: 0/1/true/false/yes/no/on/off. Got: ${HEADLESS_RAW}" >&2
    exit 2
    ;;
esac
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_RECENTER=${VISER_RECENTER:-True}
VISER_SYNC_TO_SIM=${VISER_SYNC_TO_SIM:-True}
VISER_FORCE_DT=${VISER_FORCE_DT:-True}
VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS:-False}
VISER_LOAD_URDF=${VISER_LOAD_URDF:-1}

START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.2}
FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.95}
ENABLE_DEFAULT_POSE_PREPEND=${ENABLE_DEFAULT_POSE_PREPEND:-False}
DEFAULT_POSE_PREPEND_DURATION_S=${DEFAULT_POSE_PREPEND_DURATION_S:-0.0}
RESET_NOISE_SCALE=${RESET_NOISE_SCALE:-0.0}
MAX_EPISODE_LENGTH_S=${MAX_EPISODE_LENGTH_S:-1000000}
SIM_ENV_SPACING=${SIM_ENV_SPACING:-0.0}
PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-67108864}
DISABLE_RANDOMIZATION=${DISABLE_RANDOMIZATION:-True}
VIS_GPU=${VIS_GPU:-auto}

# Pick a less-loaded GPU by default for IsaacSim startup stability.
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" || -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  if [[ "${VIS_GPU}" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      AUTO_GPU="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2,2n | head -n1 | cut -d, -f1 | tr -d ' ')"
      if [[ -n "${AUTO_GPU}" ]]; then
        export CUDA_VISIBLE_DEVICES="${AUTO_GPU}"
      fi
    fi
  elif [[ "${VIS_GPU}" =~ ^[0-9]+$ ]]; then
    export CUDA_VISIBLE_DEVICES="${VIS_GPU}"
  fi
fi

# Useful defaults for interactive motion/clip inspection.
export VISER_ENABLE_CLIP_GUI=${VISER_ENABLE_CLIP_GUI:-1}
export VISER_ENABLE_MANUAL_GUI=${VISER_ENABLE_MANUAL_GUI:-0}
export VISER_SHOW_TARGET_KEYPOINTS=${VISER_SHOW_TARGET_KEYPOINTS:-1}
export VISER_START_PAUSED=${VISER_START_PAUSED:-0}
export VISER_LOAD_URDF

if [[ "${TEACHER_CHECKPOINT}" != wandb://* ]] && [[ ! -f "${TEACHER_CHECKPOINT}" ]]; then
  echo "[ERROR] teacher checkpoint not found: ${TEACHER_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -e "${MOTION_DIR}" ]]; then
  echo "[ERROR] MOTION_DIR not found: ${MOTION_DIR}" >&2
  exit 1
fi
if [[ ! -f "${OBJECT_URDF}" ]]; then
  echo "[ERROR] OBJECT_URDF not found: ${OBJECT_URDF}" >&2
  exit 1
fi

if [[ -d "${MOTION_DIR}" && -n "${MOTION_CLIP_NAME}" && ! -f "${MOTION_DIR}/${MOTION_CLIP_NAME}.npz" ]]; then
  echo "[ERROR] MOTION_CLIP_NAME not found in MOTION_DIR: ${MOTION_CLIP_NAME}.npz" >&2
  exit 2
fi

if [[ "${LEGACY_OBS_ENABLED}" == "1" || "${HEIGHTMAP_REQUIRED}" == "1" ]]; then
  python - <<'PY' "${TEACHER_CHECKPOINT}" "${LEGACY_OBS_ENABLED}" "${HEIGHTMAP_REQUIRED}" || exit 2
import sys
import tempfile
from pathlib import Path

import torch


def parse_bool(v: str) -> bool:
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _parse_wandb_reference(reference: str) -> tuple[str, str]:
    if not reference.startswith("wandb://"):
        raise ValueError("Not a wandb:// reference")
    remainder = reference[len("wandb://") :]
    parts = remainder.split("/")
    if len(parts) < 4:
        raise ValueError(
            "Invalid wandb checkpoint path. Expected wandb://<entity>/<project>/<run_id>/<checkpoint_name>"
        )
    entity, project = parts[0], parts[1]
    run_id_index = 2
    if len(parts) > 4 and parts[2] == "runs":
        run_id_index = 3
    if run_id_index >= len(parts):
        raise ValueError(
            "Invalid wandb checkpoint path. Expected wandb://<entity>/<project>/<run_id>/<checkpoint_name>"
        )
    run_id = parts[run_id_index]
    ckpt_name = "/".join(parts[run_id_index + 1 :]).strip()
    if not ckpt_name:
        raise ValueError(
            "wandb checkpoint reference must include checkpoint filename, e.g. model_12000.pt"
        )
    return f"{entity}/{project}/{run_id}", ckpt_name


def load_payload(checkpoint_ref: str):
    if checkpoint_ref.startswith("wandb://"):
        import wandb

        run_path, ckpt_name = _parse_wandb_reference(checkpoint_ref)
        run = wandb.Api().run(run_path)
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloaded = run.file(ckpt_name).download(root=tmp_dir, replace=True)
            ckpt_path = Path(downloaded.name)
            if not ckpt_path.is_absolute():
                ckpt_path = (Path.cwd() / ckpt_path).resolve()
            payload = torch.load(ckpt_path, map_location="cpu")
            return payload
    return torch.load(checkpoint_ref, map_location="cpu")


checkpoint_ref = sys.argv[1]
require_legacy = parse_bool(sys.argv[2])
require_heightmap = parse_bool(sys.argv[3])

payload = load_payload(checkpoint_ref)
cfg = payload.get("experiment_config")
if not isinstance(cfg, dict):
    raise SystemExit(f"[ERROR] checkpoint has no experiment_config dict: {checkpoint_ref}")

obs_cfg = cfg.get("observation")
groups = obs_cfg.get("groups", {}) if isinstance(obs_cfg, dict) else {}
actor_obs = groups.get("actor_obs", {}) if isinstance(groups, dict) else {}
terms = actor_obs.get("terms", {}) if isinstance(actor_obs, dict) else {}
if not isinstance(terms, dict):
    raise SystemExit(f"[ERROR] checkpoint actor_obs.terms is invalid: {checkpoint_ref}")

if require_legacy:
    legacy_forbidden = ("obj_lin_vel_b", "obj_ang_vel_b")
    present = [name for name in legacy_forbidden if name in terms]
    if present:
        raise SystemExit(
            "[ERROR] LEGACY_OBS=1 but checkpoint actor_obs is non-legacy "
            f"(contains {present}): {checkpoint_ref}"
        )

if require_heightmap:
    perception_cfg = cfg.get("perception")
    if not isinstance(perception_cfg, dict):
        raise SystemExit(
            "[ERROR] REQUIRE_HEIGHTMAP=1 but checkpoint has no perception config dict: "
            f"{checkpoint_ref}"
        )
    enabled = bool(perception_cfg.get("enabled", False))
    output_mode = str(perception_cfg.get("output_mode", "")).strip()
    if not enabled:
        raise SystemExit(
            "[ERROR] REQUIRE_HEIGHTMAP=1 but checkpoint perception.enabled is False: "
            f"{checkpoint_ref}"
        )
    if output_mode != "heightmap":
        raise SystemExit(
            "[ERROR] REQUIRE_HEIGHTMAP=1 but checkpoint perception.output_mode is "
            f"'{output_mode}' (expected 'heightmap'): {checkpoint_ref}"
        )
    if "perception_obs" not in groups:
        raise SystemExit(
            "[ERROR] REQUIRE_HEIGHTMAP=1 but observation groups has no 'perception_obs': "
            f"{checkpoint_ref}"
        )

print(
    f"[INFO] Checkpoint validation passed (legacy={require_legacy}, "
    f"heightmap={require_heightmap}): {checkpoint_ref}"
)
PY
fi

EXTRA_ARGS=("$@")

cmd=(
  python -m holosoma.visualize physics
  --checkpoint "${TEACHER_CHECKPOINT}"
  --motion-dir "${MOTION_DIR}"
  --num-envs "${NUM_ENVS}"
  --headless "${HEADLESS_FLAG}"
  --pair-terrain-with-motion "${PAIR_TERRAIN_WITH_MOTION}"
  --viser-port "${VISER_PORT}"
  --viser-env-id "${VISER_ENV_ID}"
  --viser-update-hz "${VISER_UPDATE_HZ}"
  --viser-recenter "${VISER_RECENTER}"
  --training.viser_sync_to_sim "${VISER_SYNC_TO_SIM}"
  --training.viser_force_dt "${VISER_FORCE_DT}"
  --training.viser_show_scandots "${VISER_SHOW_SCANDOTS}"
  --simulator.config.scene.env_spacing "${SIM_ENV_SPACING}"
  --simulator.config.sim.max_episode_length_s "${MAX_EPISODE_LENGTH_S}"
  --simulator.config.sim.physx.gpu_collision_stack_size "${PHYSX_GPU_COLLISION_STACK_SIZE}"
  --robot.object.enabled True
  --robot.object.object_urdf_path "${OBJECT_URDF}"
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob "${START_AT_TIMESTEP_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.freeze_at_timestep_zero_prob "${FREEZE_AT_TIMESTEP_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend "${ENABLE_DEFAULT_POSE_PREPEND}"
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s "${DEFAULT_POSE_PREPEND_DURATION_S}"
  --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.overall_noise_scale "${RESET_NOISE_SCALE}"
)

if [[ -n "${MOTION_CLIP_NAME}" ]]; then
  cmd+=(
    --command.setup_terms.motion_command.params.motion_config.motion_clip_name "${MOTION_CLIP_NAME}"
  )
fi

if [[ "${DISABLE_RANDOMIZATION}" == "True" || "${DISABLE_RANDOMIZATION}" == "true" ]]; then
  cmd+=(
    --randomization.setup_terms.push_randomizer_state.params.enabled False
    --randomization.reset_terms.randomize_push_schedule.params.enabled False
    --randomization.step_terms.apply_pushes.params.enabled False
    --randomization.setup_terms.actuator_randomizer_state.params.enable_pd_gain False
    --randomization.setup_terms.actuator_randomizer_state.params.enable_rfi_lim False
    --randomization.setup_terms.setup_action_delay_buffers.params.enabled False
    --randomization.reset_terms.randomize_action_delay.params.enabled False
    --randomization.setup_terms.randomize_robot_rigid_body_material_startup.params.enabled False
    --randomization.setup_terms.randomize_base_com_startup.params.enabled False
    --randomization.setup_terms.setup_dof_pos_bias.params.enabled False
    --randomization.reset_terms.randomize_dof_state.params.randomize_dof_pos_bias False
    --randomization.setup_terms.setup_camera_raycast_randomization.params.enabled False
    --randomization.reset_terms.randomize_camera_raycast.params.enabled False
    --randomization.setup_terms.randomize_object_rigid_body_material_startup.params.enabled False
    --randomization.setup_terms.randomize_object_rigid_body_mass_startup.params.enabled False
    --randomization.setup_terms.randomize_object_rigid_body_inertia_startup.params.enabled False
  )
fi

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] teacher_checkpoint=${TEACHER_CHECKPOINT}"
echo "[INFO] legacy_obs_enabled=${LEGACY_OBS_ENABLED}"
echo "[INFO] require_heightmap=${HEIGHTMAP_REQUIRED}"
echo "[INFO] infer_dataset=${INFER_DATASET}"
echo "[INFO] motion_dir=${MOTION_DIR}"
echo "[INFO] motion_clip_name=${MOTION_CLIP_NAME:-<auto>}"
echo "[INFO] object_urdf=${OBJECT_URDF}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] headless=${HEADLESS_FLAG} (env HEADLESS=${HEADLESS})"
echo "[INFO] viser=http://localhost:${VISER_PORT}"
echo "[INFO] viser_sync_to_sim=${VISER_SYNC_TO_SIM} viser_force_dt=${VISER_FORCE_DT}"
echo "[INFO] viser_load_urdf=${VISER_LOAD_URDF}"
echo "[INFO] enable_default_pose_prepend=${ENABLE_DEFAULT_POSE_PREPEND} duration_s=${DEFAULT_POSE_PREPEND_DURATION_S}"
echo "[INFO] disable_randomization=${DISABLE_RANDOMIZATION}"

"${cmd[@]}"
