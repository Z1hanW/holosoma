#!/usr/bin/env bash
set -euo pipefail

# AS distillation with explicit pickup and drop buttons. This is a thin wrapper
# over distill_as_button.sh: data selection, single-slot object setup,
# resume-from-box behavior, DAgger/PPO schedule, camera randomization, and all
# other training settings stay on the existing button path.
#
# Student inputs:
# - actor_obs_root_contact_aware
# - actor_obs_pickup_button
# - actor_obs_drop_button
# - actor_obs_proprio_with_actions_no_linvel
#
# actor_obs_pickup_button is 1 before carry-start t1 and 0 from t1 onward.
# actor_obs_drop_button is 0 before carry-end t2 and 1 from t2 to clip end.

usage() {
  cat <<'EOF'
Usage:
  bash distill_as_dual_button.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]
  bash distill_as_dual_button.sh corl_128 [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]
  bash distill_as_dual_button.sh --resume-from-box [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]

Behavior:
  Delegates AS data preparation and launch to distill_as_button.sh, but changes
  the actor input contract to include actor_obs_pickup_button in addition to
  actor_obs_drop_button.

Examples:
  bash cp_corl.sh
  bash distill_as_dual_button.sh corl_128

Button convention:
  pickup_button = 1 before carry-start t1, 0 from t1 through clip end
  drop_button   = 0 before carry-end t2,   1 from t2 through clip end

Useful env vars:
  RESUME_FROM_BOX=1          initialize policy weights from box-button; default d9m3z369/model_17000.pt
  BOX_RESUME_CKPT=<checkpoint>  override the box policy initializer
  RUN_NAME=<name>            override W&B run display name
  TRAINING_NAME=<name>       override log/checkpoint training name
  SCHEDULE_NAME=<name>       override schedule label
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

POSITIONAL=()
CORL_128=0
while [[ $# -gt 0 ]]; do
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
    -h|--help|help)
      usage
      exit 0
      ;;
    dual|dual-button|dual_button|pickup-drop-button|pickup_drop_button)
      shift
      ;;
    corl_128|corl128|corl-128)
      CORL_128=1
      AS_SUCCESS133_FINAL0P5=1
      POSITIONAL+=("$1")
      shift
      ;;
    success133|as-success133|as_success133|success133-final0p5|success133_final0p5)
      AS_SUCCESS133_FINAL0P5=1
      POSITIONAL+=("$1")
      shift
      ;;
    --resume-from-box|--resume_from_box|resume-from-box|resume_from_box|resume-from-box-button|resume_from_box_button|init-box-button|init_box_button)
      RESUME_FROM_BOX=1
      shift
      ;;
    --no-resume-from-box|--no_resume_from_box|no-resume-from-box|no_resume_from_box)
      RESUME_FROM_BOX=0
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${AS_SUCCESS133_FINAL0P5+x}" && -z "${RESUME_FROM_BOX+x}" ]]; then
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

export AS_SUCCESS133_FINAL0P5
export RESUME_FROM_BOX
if [[ "${CORL_128}" == "1" ]]; then
  export OMOMO_EXPECTED_TOTAL="${OMOMO_EXPECTED_TOTAL:-128}"
  export RESUME_FROM_BOX_EXPECTED_TOTAL="${RESUME_FROM_BOX_EXPECTED_TOTAL:-128}"
fi
export AS_CONTACT_AWARE=1
export ROOT_COMMAND_MODE="${ROOT_COMMAND_MODE:-contact-aware}"
export SCHEDULE_VARIANT="${SCHEDULE_VARIANT:-ppo_first}"
export STUDENT_ACTOR_INPUTS="${STUDENT_ACTOR_INPUTS:-['actor_obs_root_contact_aware','actor_obs_pickup_button','actor_obs_drop_button','actor_obs_proprio_with_actions_no_linvel']}"

if [[ "${CORL_128}" == "1" && "${RESUME_FROM_BOX}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_corl128_dual_button_init_box}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_corl128_dual_button_init_box_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_corl128_init_box_sparse_root_ppo_first_contact_pickup_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-CORL 128-clip curated AS teacher-rollout subset from success155_bcleb5oi58000_final0p5_primitiveproj, initialized from box-button actor policy parameters. It uses teacher-exported contact sidecars for offline contact guidance and adaptive contact-window sampling. The dual-button interface adds pickup_button, 1 before carry-start t1 and 0 from t1 onward, plus drop_button, 0 before carry-end t2 and 1 from t2 through clip end.}"
elif [[ "${CORL_128}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_corl128_dual_button}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_corl128_dual_button_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_corl128_sparse_root_ppo_first_contact_pickup_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-CORL 128-clip curated AS teacher-rollout subset from success155_bcleb5oi58000_final0p5_primitiveproj. It uses teacher-exported contact sidecars for offline contact guidance and adaptive contact-window sampling. The dual-button interface adds pickup_button, 1 before carry-start t1 and 0 from t1 onward, plus drop_button, 0 before carry-end t2 and 1 from t2 through clip end.}"
elif [[ "${AS_SUCCESS133_FINAL0P5}" == "1" && "${RESUME_FROM_BOX}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_success133_final0p5_dual_button_init_box}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_success133_final0p5_dual_button_init_box_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_success133_final0p5_init_box_sparse_root_ppo_first_contact_pickup_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS teacher-rollout filtered perception distill initialized from box-button policy parameters. Uses contact-aware sparse root plus pickup/drop button student inputs: pickup_button is 1 before carry-start t1 and 0 from t1 onward; drop_button is 0 before carry-end t2 and 1 from t2 through clip end. Data prep, contact sidecars, adaptive sampling, camera randomization, DAgger/PPO schedule, and training-from-iteration-0 behavior are inherited from distill_as_button.sh.}"
elif [[ "${AS_SUCCESS133_FINAL0P5}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_success133_final0p5_dual_button}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_success133_final0p5_dual_button_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_success133_final0p5_sparse_root_ppo_first_contact_pickup_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS teacher-rollout filtered perception distill with contact-aware sparse root plus pickup/drop button student inputs: pickup_button is 1 before carry-start t1 and 0 from t1 onward; drop_button is 0 before carry-end t2 and 1 from t2 through clip end. Data prep, contact sidecars, adaptive sampling, camera randomization, and DAgger/PPO schedule are inherited from distill_as_button.sh.}"
elif [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_keep169_dual_button_init_box}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_keep169_dual_button_init_box_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_keep169_init_box_sparse_root_ppo_first_contact_pickup_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS keep169 perception distill initialized from box-button policy parameters. Uses contact-aware sparse root plus pickup/drop button student inputs: pickup_button is 1 before carry-start t1 and 0 from t1 onward; drop_button is 0 before carry-end t2 and 1 from t2 through clip end. Data prep, contact sidecars, adaptive sampling, DAgger/PPO schedule, and training-from-iteration-0 behavior are inherited from distill_as_button.sh.}"
else
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_dual_button}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_dual_button_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_sparse_root_ppo_first_contact_pickup_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS perception distill with contact-aware sparse root plus pickup/drop button student inputs: pickup_button is 1 before carry-start t1 and 0 from t1 onward; drop_button is 0 before carry-end t2 and 1 from t2 through clip end. Data prep, contact sidecars, adaptive sampling, camera randomization, and DAgger/PPO schedule are inherited from distill_as_button.sh.}"
fi

echo "[INFO] Launching AS/OMOMO dual-button perception distillation"
echo "[INFO] student_actor_inputs=${STUDENT_ACTOR_INPUTS}"
echo "[INFO] pickup_button_interface=1 pickup_button=1_before_t1_0_from_t1_to_end"
echo "[INFO] drop_button_interface=1 drop_button=0_before_t2_1_from_t2_to_end"

exec bash "${SCRIPT_DIR}/distill_as_button.sh" "${POSITIONAL[@]}"
