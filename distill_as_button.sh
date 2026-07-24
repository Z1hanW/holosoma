#!/usr/bin/env bash
set -euo pipefail

# Distill an AS/OMOMO real-mesh teacher into a contact-aware drop-button
# depth-perception student.
#
# This wrapper keeps the AS data/URDF validation and single-slot object setup in
# distill_as_perception.sh, while matching distill_box_button.sh's student
# interface:
# - actor_obs_root_contact_aware
# - actor_obs_drop_button
# - actor_obs_proprio_with_actions_no_linvel
#
# actor_obs_drop_button is 0 before carry-end t2 and 1 from t2 to clip end.

usage() {
  cat <<'EOF'
Usage:
  bash distill_as_button.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]
  TEACHER_CHECKPOINT=<teacher_checkpoint> bash distill_as_button.sh [extra args...]
  bash distill_as_button.sh success133 [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]
  bash distill_as_button.sh resume-from-box-button [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]

Examples:
  bash cp_tao.sh success133
  bash distill_as_button.sh
  bash distill_as_button.sh corl_128
  bash distill_as_button.sh /data/logs_new/carry-any/<run>/model_01000.pt
  bash distill_as_button.sh wandb://zihanw22/carry-any/<run_id>/model_01000.pt
  bash distill_as_button.sh resume-from-box-button

Defaults:
  teacher: latest model from https://wandb.ai/zihanw22/carry-any/runs/bcleb5oi
  AS bank: success133 teacher-rollout filtered AS bank
  corl_128: curated 126-clip success155 subset under data/ds_as_data/corl_128
  student inputs: actor_obs_root_contact_aware, actor_obs_drop_button,
                  actor_obs_proprio_with_actions_no_linvel

Useful env vars:
  AS_SUCCESS133_FINAL0P5=0      do not force the success133 AS bank; provide a contact-capable AS bank yourself
  RESUME_FROM_BOX=1             initialize from an architecture-compatible box-button policy
  BOX_RESUME_CKPT=<checkpoint>  box policy initializer; actor keys/shapes must match exactly
  BAD_TRACKING_THRESHOLD_AUGMENT=1.0|1.1|1.2|1.4
                                  scale bad-tracking termination thresholds from gt/generalist strict values
  STUDENT_POLICY_TYPE=mlp|flow   student actor type; mlp is the default
  STUDENT_FLOW_STEPS=4           Euler steps for flow actor inference when STUDENT_POLICY_TYPE=flow
  STUDENT_FLOW_TRAIN_NOISE_STD=1.0
                                  base Gaussian std for flow-matching training targets
  STUDENT_FLOW_INFERENCE_NOISE_STD=0.0
                                  keep 0 for deterministic policy export/deployment
  RUN_NAME=<name>               override W&B run display name
  TRAINING_NAME=<name>          override log/checkpoint training name
  DRY_RUN=1                     forwarded to the delegated launcher if supported
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/scripts/gpu_launch_defaults.sh"
SCHEDULE_NAME_USER_SET=0
[[ -n "${SCHEDULE_NAME+x}" ]] && SCHEDULE_NAME_USER_SET=1
SCHEDULE_NOTES_USER_SET=0
[[ -n "${SCHEDULE_NOTES+x}" ]] && SCHEDULE_NOTES_USER_SET=1

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

POSITIONAL=()
CORL_128=0
while [[ $# -gt 0 ]]; do
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
    as|as-button|as_button|button|drop-button|drop_button)
      shift
      ;;
    corl_128|corl128|corl-128)
      CORL_128=1
      AS_SUCCESS133_FINAL0P5=1
      AS_SUCCESS133_BANK_NAME=corl_128
      OMOMO_EXPECTED_TOTAL=126
      RESUME_FROM_BOX_EXPECTED_TOTAL=126
      shift
      ;;
    success133|as-success133|as_success133|success133-final0p5|success133_final0p5)
      AS_SUCCESS133_FINAL0P5=1
      POSITIONAL+=("$1")
      shift
      ;;
    resume-from-box-button|resume_from_box_button|init-box-button|init_box_button)
      RESUME_FROM_BOX=1
      shift
      ;;
    resume-from-box|resume_from_box)
      RESUME_FROM_BOX=1
      POSITIONAL+=("$1")
      shift
      ;;
    default)
      SCHEDULE_VARIANT=default
      shift
      ;;
    dagger_mix|dagger-mix|daggermix)
      SCHEDULE_VARIANT=dagger_mix
      shift
      ;;
    dag_first|dag-first|dagger-first)
      SCHEDULE_VARIANT=dag_first
      shift
      ;;
    ppo_first|ppo-first)
      SCHEDULE_VARIANT=ppo_first
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${AS_SUCCESS133_FINAL0P5+x}" ]]; then
  AS_SUCCESS133_FINAL0P5=1
fi

normalize_bool() {
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
      exit 2
      ;;
  esac
}

AS_SUCCESS133_FINAL0P5="$(normalize_bool AS_SUCCESS133_FINAL0P5 "${AS_SUCCESS133_FINAL0P5:-0}")"
RESUME_FROM_BOX="$(normalize_bool RESUME_FROM_BOX "${RESUME_FROM_BOX:-0}")"
STUDENT_POLICY_TYPE="$(echo "${STUDENT_POLICY_TYPE:-mlp}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
case "${STUDENT_POLICY_TYPE}" in
  mlp|flow)
    ;;
  *)
    echo "[ERROR] STUDENT_POLICY_TYPE must be one of: mlp|flow. Got: ${STUDENT_POLICY_TYPE}" >&2
    exit 2
    ;;
esac

DROP_BUTTON_POLICY_INIT=0
if [[ "${RESUME_FROM_BOX}" == "1" || "${AS_POLICY_INIT_PROFILE:-}" == "drop_button_mlp_perception" ]]; then
  DROP_BUTTON_POLICY_INIT=1
fi

if [[ "${DROP_BUTTON_POLICY_INIT}" == "1" && "${STUDENT_POLICY_TYPE}" != "mlp" ]]; then
  echo "[ERROR] The drop-button policy-init profile cannot initialize a ${STUDENT_POLICY_TYPE} actor from an MLP checkpoint." >&2
  echo "[ERROR] Use an architecture-matched flow training checkpoint through RESUME_CKPT instead." >&2
  exit 2
fi

if [[ "${DROP_BUTTON_POLICY_INIT}" == "1" ]]; then
  BOX_POLICY_INIT_ACTOR_HIDDEN_DIMS=${BOX_POLICY_INIT_ACTOR_HIDDEN_DIMS:-'[512,256,128]'}
  STUDENT_ACTOR_HIDDEN_DIMS=${STUDENT_ACTOR_HIDDEN_DIMS:-${BOX_POLICY_INIT_ACTOR_HIDDEN_DIMS}}
  _student_dims_compact="$(echo "${STUDENT_ACTOR_HIDDEN_DIMS}" | tr -d '[:space:]')"
  _box_dims_compact="$(echo "${BOX_POLICY_INIT_ACTOR_HIDDEN_DIMS}" | tr -d '[:space:]')"
  if [[ "${_student_dims_compact}" != "${_box_dims_compact}" ]]; then
    echo "[ERROR] Drop-button policy init requires actor hidden dims ${BOX_POLICY_INIT_ACTOR_HIDDEN_DIMS}; got ${STUDENT_ACTOR_HIDDEN_DIMS}." >&2
    exit 2
  fi
  export BOX_POLICY_INIT_ACTOR_HIDDEN_DIMS
  export STUDENT_ACTOR_HIDDEN_DIMS
fi

if [[ "${DROP_BUTTON_POLICY_INIT}" == "1" ]]; then
  DEFAULT_BOX_RESUME_RUN=${DEFAULT_BOX_RESUME_RUN:-"https://wandb.ai/zihanw22/boxer/runs/d9m3z369-recovered"}
  DEFAULT_BOX_RESUME_MODEL_FILE=${DEFAULT_BOX_RESUME_MODEL_FILE:-model_22000.pt}
  BOX_RESUME_MODEL_FILE=${BOX_RESUME_MODEL_FILE:-${DEFAULT_BOX_RESUME_MODEL_FILE}}
  DEFAULT_BOX_RESUME_CHECKPOINT=${DEFAULT_BOX_RESUME_CHECKPOINT:-"${DEFAULT_BOX_RESUME_RUN}/files/${BOX_RESUME_MODEL_FILE}"}
  if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
    BOX_RESUME_CKPT=${BOX_RESUME_CKPT:-${RESUME_FROM_BOX_CKPT:-${DEFAULT_BOX_RESUME_CHECKPOINT}}}
  fi
  export DEFAULT_BOX_RESUME_RUN
  export DEFAULT_BOX_RESUME_MODEL_FILE
  export BOX_RESUME_MODEL_FILE
  export DEFAULT_BOX_RESUME_CHECKPOINT
  export BOX_RESUME_CKPT
fi

export AS_SUCCESS133_FINAL0P5
export RESUME_FROM_BOX
export STUDENT_POLICY_TYPE
if [[ "${CORL_128:-0}" == "1" ]]; then
  export AS_SUCCESS133_BANK_NAME
  export OMOMO_EXPECTED_TOTAL
  export RESUME_FROM_BOX_EXPECTED_TOTAL
fi
export AS_CONTACT_AWARE=1
export ROOT_COMMAND_MODE="${ROOT_COMMAND_MODE:-contact-aware}"
export SCHEDULE_VARIANT="${SCHEDULE_VARIANT:-ppo_first}"
export STUDENT_ACTOR_INPUTS="${STUDENT_ACTOR_INPUTS:-['actor_obs_root_contact_aware','actor_obs_drop_button','actor_obs_proprio_with_actions_no_linvel']}"
export DAGGER_MATCH_STD="${DAGGER_MATCH_STD:-False}"
export PPO_START_NOISE_STD="${PPO_START_NOISE_STD:-0.1}"
export PPO_START_NOISE_STD_UNTIL_COEFF="${PPO_START_NOISE_STD_UNTIL_COEFF:-0.1}"
if [[ "${DROP_BUTTON_POLICY_INIT}" == "1" ]]; then
  "${PYTHON_BIN}" - "${STUDENT_ACTOR_INPUTS}" <<'PY'
from __future__ import annotations

import ast
import sys

expected = [
    "actor_obs_root_contact_aware",
    "actor_obs_drop_button",
    "actor_obs_proprio_with_actions_no_linvel",
]
try:
    actual = list(ast.literal_eval(sys.argv[1]))
except Exception as exc:
    raise SystemExit(f"[ERROR] Invalid STUDENT_ACTOR_INPUTS={sys.argv[1]!r}: {exc}") from exc
if actual != expected:
    raise SystemExit(
        "[ERROR] Drop-button policy init is only architecture-compatible with actor inputs "
        f"{expected!r}; got {actual!r}."
    )
PY
  export AS_POLICY_INIT_PROFILE=drop_button_mlp_perception

  _policy_init_checkpoint="${POLICY_INIT_CKPT:-${POLICY_INIT_CHECKPOINT:-${BOX_RESUME_CKPT:-}}}"
  if [[ -z "${_policy_init_checkpoint}" ]]; then
    echo "[ERROR] AS_POLICY_INIT_PROFILE=drop_button_mlp_perception requires an exact policy-init checkpoint." >&2
    exit 2
  fi
  POLICY_INIT_CACHE_ROOT=${POLICY_INIT_CACHE_ROOT:-"${HOME}/.cache/holosoma/policy_init"}
  _policy_init_checkpoint=$("${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/resolve_exact_checkpoint.py" \
    --ref "${_policy_init_checkpoint}" \
    --cache-root "${POLICY_INIT_CACHE_ROOT}")
  export POLICY_INIT_CKPT="${_policy_init_checkpoint}"
  if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
    BOX_RESUME_CKPT="${_policy_init_checkpoint}"
    export BOX_RESUME_CKPT
  fi
  if [[ ! -f "${_policy_init_checkpoint}" ]]; then
    echo "[ERROR] Policy-init checkpoint is not a readable local file: ${_policy_init_checkpoint}" >&2
    exit 2
  fi
  "${PYTHON_BIN}" - "${_policy_init_checkpoint}" "${BOX_POLICY_INIT_ACTOR_HIDDEN_DIMS}" <<'PY'
from __future__ import annotations

import ast
import os
import sys

from holosoma.utils.checkpoint_validation import load_verified_torch_checkpoint

checkpoint_path, expected_dims_raw = sys.argv[1:3]
checkpoint, _ = load_verified_torch_checkpoint(
    checkpoint_path,
    expected_sha256=(
        os.environ.get("BOX_POLICY_INIT_EXPECTED_SHA256")
        or os.environ.get("POLICY_INIT_EXPECTED_SHA256")
        or None
    ),
    map_location="cpu",
)
try:
    actor = checkpoint["experiment_config"]["algo"]["config"]["module_dict"]["actor"]
    actor_type = actor["type"]
    actor_inputs = list(actor["input_dim"])
    layer = actor["layer_config"]
    hidden_dims = list(layer["hidden_dims"])
except (KeyError, TypeError) as exc:
    raise SystemExit("[ERROR] Policy-init checkpoint lacks strict actor architecture metadata.") from exc

expected_inputs = [
    "actor_obs_root_contact_aware",
    "actor_obs_drop_button",
    "actor_obs_proprio_with_actions_no_linvel",
]
expected_dims = list(ast.literal_eval(expected_dims_raw))
mismatches = []
if actor_type != "MLPPerceptionEncoder":
    mismatches.append(f"type checkpoint={actor_type!r} expected='MLPPerceptionEncoder'")
if actor_inputs != expected_inputs:
    mismatches.append(f"inputs checkpoint={actor_inputs!r} expected={expected_inputs!r}")
if hidden_dims != expected_dims:
    mismatches.append(f"hidden_dims checkpoint={hidden_dims!r} expected={expected_dims!r}")
if str(layer.get("perception_input_name", "")) != "perception_obs":
    mismatches.append(
        f"perception_input_name checkpoint={layer.get('perception_input_name')!r} expected='perception_obs'"
    )
if mismatches:
    raise SystemExit("[ERROR] Policy-init checkpoint profile mismatch:\n  - " + "\n  - ".join(mismatches))
print(f"[INFO] policy_init_profile_verified checkpoint={checkpoint_path}")
PY
fi
export ENABLE_DEFAULT_POSE_PREPEND="${ENABLE_DEFAULT_POSE_PREPEND:-True}"
export DEFAULT_POSE_PREPEND_DURATION_S="${DEFAULT_POSE_PREPEND_DURATION_S:-0.2}"
export HOLOSOMA_PERCEPTION_INCLUDE_ROBOT_MESH="${HOLOSOMA_PERCEPTION_INCLUDE_ROBOT_MESH:-1}"
export HOLOSOMA_MOTION_METRICS_INTERVAL="${HOLOSOMA_MOTION_METRICS_INTERVAL:-16}"
export BAD_TRACKING_THRESHOLD_AUGMENT="${BAD_TRACKING_THRESHOLD_AUGMENT:-1.0}"

if [[ "${CORL_128:-0}" == "1" && "${RESUME_FROM_BOX:-0}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_corl128_button_init_box}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_corl128_button_init_box_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_corl128_init_box_sparse_root_ppo_first_contact_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-CORL 126-clip curated AS teacher-rollout subset from success155_bcleb5oi58000_final0p5_primitiveproj, initialized from an architecture-matched box-button actor. The subset keeps the cleaned box/bin/barrel/ball clips, scale lamp, scale chair plus noscale__any_chair_85, all table clips, and the selected monitor set. Contact T1 windows reweight the failure-adaptive timestep density according to the configured multiplicative/target-mass mode; start-at-zero is an explicit reset mixture and AS metrics report the effective distribution. PPO starts at the configured iteration-0 coefficient. The drop button is 0 before t2 and 1 from t2 onward.}"
elif [[ "${CORL_128:-0}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_corl128_button}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_corl128_button_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_corl128_sparse_root_ppo_first_contact_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-CORL 126-clip curated AS teacher-rollout subset from success155_bcleb5oi58000_final0p5_primitiveproj. The subset keeps the cleaned box/bin/barrel/ball clips, scale lamp, scale chair plus noscale__any_chair_85, all table clips, and the selected monitor set. Contact T1 windows reweight the failure-adaptive timestep density according to the configured multiplicative/target-mass mode; start-at-zero is an explicit reset mixture and AS metrics report the effective distribution. PPO starts at the configured iteration-0 coefficient. The drop button is 0 before t2 and 1 from t2 onward.}"
elif [[ "${AS_SUCCESS133_FINAL0P5:-0}" == "1" && "${RESUME_FROM_BOX:-0}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_success133_final0p5_button_init_box}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_success133_final0p5_button_init_box_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_success133_final0p5_init_box_sparse_root_ppo_first_contact_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS teacher-rollout filtered 133-clip real-mesh perception distill initialized from an architecture-matched box-button actor. Clips satisfy stable_contact_success=True and final_object_position_error_m<=0.5. Contact T1 windows reweight the failure-adaptive timestep density according to the configured multiplicative/target-mass mode; start-at-zero is an explicit reset mixture and AS metrics report the effective distribution. PPO starts at the configured iteration-0 coefficient. The drop button is 0 before t2 and 1 from t2 onward.}"
elif [[ "${AS_SUCCESS133_FINAL0P5:-0}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_success133_final0p5_button}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_success133_final0p5_button_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_success133_final0p5_sparse_root_ppo_first_contact_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS teacher-rollout filtered 133-clip real-mesh perception distill with contact-aware sparse root and drop-button student input. Clips satisfy stable_contact_success=True and final_object_position_error_m<=0.5. Contact T1 windows reweight the failure-adaptive timestep density according to the configured multiplicative/target-mass mode; start-at-zero is an explicit reset mixture and AS metrics report the effective distribution. PPO starts at the configured iteration-0 coefficient. The drop button is 0 before t2 and 1 from t2 onward.}"
elif [[ "${RESUME_FROM_BOX:-0}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_keep169_button_init_box}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_keep169_button_init_box_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_keep169_init_box_sparse_root_ppo_first_contact_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS keep169 real-mesh perception distill initialized from an architecture-matched box-button actor. Contact T1 windows reweight the failure-adaptive timestep density according to the configured multiplicative/target-mass mode; start-at-zero is an explicit reset mixture and AS metrics report the effective distribution. PPO starts at the configured iteration-0 coefficient. The drop button is 0 before t2 and 1 from t2 onward.}"
else
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_button}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_button_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_sparse_root_ppo_first_contact_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS real-mesh perception distill with contact-aware sparse root and drop-button student input. The button interface adds actor_obs_drop_button, 0 before carry-end t2 and 1 from t2 through clip end; root command behavior is unchanged.}"
fi

if [[ "${SCHEDULE_VARIANT}" != "ppo_first" ]]; then
  if [[ "${SCHEDULE_NAME_USER_SET}" -eq 0 ]]; then
    SCHEDULE_NAME="${SCHEDULE_NAME//ppo_first/${SCHEDULE_VARIANT}}"
    export SCHEDULE_NAME
  fi
  if [[ "${SCHEDULE_NOTES_USER_SET}" -eq 0 ]]; then
    case "${SCHEDULE_VARIANT}" in
      dagger_mix)
        SCHEDULE_NOTES="Contact-aware AS drop-button distillation using the explicit dagger_mix contract: PPO is disabled and teacher-action rollout mixing follows the configured annealing schedule."
        ;;
      dag_first)
        SCHEDULE_NOTES="Contact-aware AS drop-button distillation using the explicit dag_first contract: pure DAgger precedes the configured PPO blend."
        ;;
      default)
        SCHEDULE_NOTES="Contact-aware AS drop-button distillation using the explicit default schedule contract selected by the delegated launcher."
        ;;
    esac
    export SCHEDULE_NOTES
  fi
fi

if [[ "${STUDENT_ACTOR_INPUTS}" == *"actor_obs_pickup_button"* ]]; then
  echo "[INFO] Launching AS/OMOMO pickup/drop-button perception distillation"
else
  echo "[INFO] Launching AS/OMOMO drop-button perception distillation"
fi
echo "[INFO] student_actor_inputs=${STUDENT_ACTOR_INPUTS}"
if [[ "${STUDENT_ACTOR_INPUTS}" == *"actor_obs_pickup_button"* ]]; then
  echo "[INFO] pickup_button_interface=1 pickup_button=1_before_t1_0_from_t1_to_end"
fi
echo "[INFO] drop_button_interface=1 drop_button=0_before_t2_1_from_t2_to_end"
echo "[INFO] student_policy_type=${STUDENT_POLICY_TYPE}"
if [[ "${STUDENT_POLICY_TYPE}" == "flow" ]]; then
  export STUDENT_FLOW_STEPS="${STUDENT_FLOW_STEPS:-4}"
  export STUDENT_FLOW_TRAIN_NOISE_STD="${STUDENT_FLOW_TRAIN_NOISE_STD:-1.0}"
  export STUDENT_FLOW_TIME_EPSILON="${STUDENT_FLOW_TIME_EPSILON:-1e-4}"
  export STUDENT_FLOW_INFERENCE_NOISE_STD="${STUDENT_FLOW_INFERENCE_NOISE_STD:-0.0}"
  echo "[INFO] student_flow steps=${STUDENT_FLOW_STEPS} train_noise_std=${STUDENT_FLOW_TRAIN_NOISE_STD} time_epsilon=${STUDENT_FLOW_TIME_EPSILON} inference_noise_std=${STUDENT_FLOW_INFERENCE_NOISE_STD}"
fi
echo "[INFO] HOLOSOMA_MOTION_METRICS_INTERVAL=${HOLOSOMA_MOTION_METRICS_INTERVAL}"

exec bash "${SCRIPT_DIR}/distill_as_perception.sh" "${POSITIONAL[@]}"
