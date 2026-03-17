#!/usr/bin/env bash
set -euo pipefail

# Teacher-policy inference for box tracking.
#
# Defaults prefer the motion/object settings serialized inside the checkpoint so
# single-motion tracking runs from train_object_base.sh replay their training
# motion/object by default. If the checkpoint does not provide them, the script
# falls back to the same single-motion defaults used by train_object_base.sh.
#
# Usage:
#   bash infer_box_tracking_single.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra tyro args...]
#
# Optional env vars:
#   TEACHER_CHECKPOINT        (default: latest checkpoint from the train_object_base.sh W&B run)
#   WANDB_MODEL_FILE          (optional; used when TEACHER_CHECKPOINT is a W&B run URL without /files/<checkpoint>)
#   LEGACY_OBS                (default: 0; set 1/true to require legacy checkpoint observation layout)
#   REQUIRE_HEIGHTMAP         (default: 0; set 1/true to require checkpoint perception.enabled=True and output_mode=heightmap)
#   DEFAULT_LEGACY_TEACHER_CHECKPOINT
#                             (optional; used as default checkpoint when LEGACY_OBS=1 and no checkpoint is explicitly provided)
#   MOTION_DIR                (optional override; if unset, prefer checkpoint motion_file, else the train_object_base.sh single-motion default)
#   MOTION_CLIP_NAME          (optional: pin a single clip)
#   OBJECT_URDF               (optional override; if unset, prefer checkpoint object_urdf_path, else the train_object_base.sh single-object default)
#   NUM_ENVS                  (default: 1)
#   HEADLESS                  (default: False; set True for headless eval)
#   VISER_PORT                (default: random)
#   VISER_ENV_ID              (default: 0)
#   VISER_UPDATE_HZ           (default: 30)
#   VISER_RECENTER            (default: True)
#   VISER_SYNC_TO_SIM         (default: True)
#   VISER_FORCE_DT            (default: True)
#   VISER_LOAD_URDF           (default: 1; URDF meshes are shown in Viser, but pose/object selection comes from Isaac Sim runtime state)
#   DISABLE_RANDOMIZATION     (default: True)
#   VIS_GPU                   (default: auto; picks least-used GPU if CUDA_VISIBLE_DEVICES is unset)

usage() {
  cat <<'EOF'
Usage:
  bash infer_box_tracking_single.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra tyro args...]

Examples:
  bash infer_box_tracking_single.sh
  bash infer_box_tracking_single.sh /abs/path/to/model_17000.pt
  bash infer_box_tracking_single.sh https://wandb.ai/zihanw22/boxer/runs/gx2wduvw
  MOTION_CLIP_NAME=sub3_largebox_003_mj_w_obj bash infer_box_tracking_single.sh
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
DEFAULT_TEACHER_CHECKPOINT=${DEFAULT_TEACHER_CHECKPOINT:-"https://wandb.ai/zihanw22/boxer/runs/opq0wbyq"}
DEFAULT_LEGACY_TEACHER_CHECKPOINT="${DEFAULT_LEGACY_TEACHER_CHECKPOINT:-}"
if [[ -n "${WANDB_MODEL_FILE+x}" && -n "${WANDB_MODEL_FILE}" ]]; then
  WANDB_MODEL_FILE_FROM_ENV=1
else
  WANDB_MODEL_FILE_FROM_ENV=0
fi

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

  python - "${entity}" "${project}" "${run_id}" <<'PY' 2>/dev/null || true
import re
import sys
from pathlib import Path

repo_root = Path.cwd().resolve()
sanitized_sys_path: list[str] = []
for path_entry in sys.path:
    if path_entry in {"", "."}:
        continue
    try:
        if Path(path_entry).resolve() == repo_root:
            continue
    except Exception:
        pass
    sanitized_sys_path.append(path_entry)
sys.path = sanitized_sys_path

try:
    import wandb
except Exception:
    sys.exit(0)

entity, project, run_id = sys.argv[1:4]
api = wandb.Api(timeout=30)
run = api.run(f"{entity}/{project}/{run_id}")
numbered_models: list[tuple[int, str]] = []
pt_candidates: list[str] = []
model_pattern = re.compile(r"^model_(\d+)\.pt$")

for file_obj in run.files():
    name = getattr(file_obj, "name", "")
    if not name.endswith(".pt"):
        continue
    pt_candidates.append(name)
    match = model_pattern.match(name)
    if match:
        numbered_models.append((int(match.group(1)), name))

if numbered_models:
    print(max(numbered_models, key=lambda item: item[0])[1])
elif len(pt_candidates) == 1:
    print(pt_candidates[0])
elif pt_candidates:
    print(sorted(pt_candidates)[-1])
PY
}

resolve_data_path() {
  local path_value="$1"
  if [[ -z "${path_value}" ]]; then
    echo ""
    return 0
  fi

  if [[ "${path_value}" == s3://* || "${path_value}" == /* ]]; then
    echo "${path_value}"
    return 0
  fi

  if [[ "${path_value}" == holosoma/data/* ]]; then
    echo "${SCRIPT_DIR}/src/holosoma/${path_value}"
    return 0
  fi

  python - "${path_value}" <<'PY'
import sys
from pathlib import Path

print(str(Path(sys.argv[1]).expanduser().resolve()))
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
  local model_file=""

  parsed="$(parse_wandb_run_url "${ref}" || true)"
  if [[ -z "${parsed}" ]]; then
    echo "${ref}"
    return 0
  fi

  IFS=$'\t' read -r entity project run_id explicit_file <<< "${parsed}"
  if [[ -n "${explicit_file}" ]]; then
    model_file="${explicit_file}"
  elif [[ "${WANDB_MODEL_FILE_FROM_ENV}" == "1" ]]; then
    model_file="${WANDB_MODEL_FILE}"
  else
    model_file="$(resolve_remote_wandb_checkpoint_name "${entity}" "${project}" "${run_id}")"
    if [[ -n "${model_file}" ]]; then
      echo "[INFO] Resolved wandb run URL to remote checkpoint: ${model_file}" >&2
    fi
  fi

  if [[ -z "${model_file}" ]]; then
    echo "[ERROR] Could not determine a .pt checkpoint for W&B run URL: ${ref}" >&2
    echo "[ERROR] Pass a /files/<checkpoint>.pt URL or set WANDB_MODEL_FILE." >&2
    return 2
  fi

  echo "wandb://${entity}/${project}/${run_id}/${model_file}"
}

extract_checkpoint_motion_defaults() {
  local checkpoint_ref="$1"

  python - <<'PY' "${checkpoint_ref}" 2>/dev/null || true
import sys
import tempfile
from pathlib import Path

import torch


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
            return torch.load(ckpt_path, map_location="cpu")
    return torch.load(checkpoint_ref, map_location="cpu")


payload = load_payload(sys.argv[1])
cfg = payload.get("experiment_config")
if not isinstance(cfg, dict):
    sys.exit(0)

command_cfg = cfg.get("command")
setup_terms = command_cfg.get("setup_terms", {}) if isinstance(command_cfg, dict) else {}
motion_command = setup_terms.get("motion_command", {}) if isinstance(setup_terms, dict) else {}
params = motion_command.get("params", {}) if isinstance(motion_command, dict) else {}
motion_cfg = params.get("motion_config", {}) if isinstance(params, dict) else {}
robot_cfg = cfg.get("robot")
object_cfg = robot_cfg.get("object", {}) if isinstance(robot_cfg, dict) else {}

motion_file = motion_cfg.get("motion_file") if isinstance(motion_cfg, dict) else None
motion_clip_name = motion_cfg.get("motion_clip_name") if isinstance(motion_cfg, dict) else None
object_urdf_path = object_cfg.get("object_urdf_path") if isinstance(object_cfg, dict) else None

for value in (motion_file, motion_clip_name, object_urdf_path):
    print("" if value is None else str(value))
PY
}

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
  if [[ "$1" == wandb://* || "$1" == https://wandb.ai/*/runs/* || "$1" == /* || "$1" == ./* || "$1" == ../* || "$1" == *.pt ]]; then
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

TEACHER_CHECKPOINT="$(normalize_checkpoint_ref "${TEACHER_CHECKPOINT}")"

DEFAULT_SINGLE_MOTION_SOURCE="${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
DEFAULT_SINGLE_OBJECT_URDF="${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"

MOTION_DIR_FROM_ENV=0
if [[ -n "${MOTION_DIR+x}" ]]; then
  MOTION_DIR_FROM_ENV=1
fi
OBJECT_URDF_FROM_ENV=0
if [[ -n "${OBJECT_URDF+x}" ]]; then
  OBJECT_URDF_FROM_ENV=1
fi
MOTION_CLIP_NAME_FROM_ENV=0
if [[ -n "${MOTION_CLIP_NAME+x}" ]]; then
  MOTION_CLIP_NAME_FROM_ENV=1
fi

CHECKPOINT_MOTION_SOURCE=""
CHECKPOINT_MOTION_CLIP_NAME=""
CHECKPOINT_OBJECT_URDF=""
mapfile -t checkpoint_defaults_lines < <(extract_checkpoint_motion_defaults "${TEACHER_CHECKPOINT}")
checkpoint_motion_source="${checkpoint_defaults_lines[0]:-}"
checkpoint_motion_clip_name="${checkpoint_defaults_lines[1]:-}"
checkpoint_object_urdf="${checkpoint_defaults_lines[2]:-}"
if [[ -n "${checkpoint_motion_source}" ]]; then
  CHECKPOINT_MOTION_SOURCE="$(resolve_data_path "${checkpoint_motion_source}")"
fi
if [[ -n "${checkpoint_motion_clip_name}" ]]; then
  CHECKPOINT_MOTION_CLIP_NAME="${checkpoint_motion_clip_name}"
fi
if [[ -n "${checkpoint_object_urdf}" ]]; then
  CHECKPOINT_OBJECT_URDF="$(resolve_data_path "${checkpoint_object_urdf}")"
fi

if [[ "${MOTION_DIR_FROM_ENV}" != "1" ]]; then
  if [[ -n "${CHECKPOINT_MOTION_SOURCE}" ]]; then
    MOTION_DIR="${CHECKPOINT_MOTION_SOURCE}"
  else
    MOTION_DIR="${DEFAULT_SINGLE_MOTION_SOURCE}"
  fi
fi

if [[ "${MOTION_CLIP_NAME_FROM_ENV}" != "1" && -n "${CHECKPOINT_MOTION_CLIP_NAME}" ]]; then
  MOTION_CLIP_NAME="${CHECKPOINT_MOTION_CLIP_NAME}"
else
  MOTION_CLIP_NAME=${MOTION_CLIP_NAME:-}
fi
if [[ "${OBJECT_URDF_FROM_ENV}" != "1" ]]; then
  if [[ -n "${CHECKPOINT_OBJECT_URDF}" ]]; then
    OBJECT_URDF="${CHECKPOINT_OBJECT_URDF}"
  else
    OBJECT_URDF="${DEFAULT_SINGLE_OBJECT_URDF}"
  fi
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

START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-1.0}
FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}
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
echo "[INFO] motion_dir=${MOTION_DIR}"
echo "[INFO] motion_clip_name=${MOTION_CLIP_NAME:-<auto>}"
echo "[INFO] object_urdf=${OBJECT_URDF}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] headless=${HEADLESS_FLAG} (env HEADLESS=${HEADLESS})"
echo "[INFO] viser=http://localhost:${VISER_PORT}"
echo "[INFO] viser_sync_to_sim=${VISER_SYNC_TO_SIM} viser_force_dt=${VISER_FORCE_DT}"
echo "[INFO] viser_load_urdf=${VISER_LOAD_URDF}"
echo "[INFO] disable_randomization=${DISABLE_RANDOMIZATION}"

"${cmd[@]}"
