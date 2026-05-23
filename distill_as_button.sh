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
  corl_128: curated 128-clip success155 subset under data/ds_as_data/corl_128
  student inputs: actor_obs_root_contact_aware, actor_obs_drop_button,
                  actor_obs_proprio_with_actions_no_linvel

Useful env vars:
  AS_SUCCESS133_FINAL0P5=0      do not force the success133 AS bank; provide a contact-capable AS bank yourself
  RESUME_FROM_BOX=1             initialize from a box-button policy; default d9m3z369/model_22000.pt
  BOX_RESUME_CKPT=<checkpoint>  box policy initializer
  RUN_NAME=<name>               override W&B run display name
  TRAINING_NAME=<name>          override log/checkpoint training name
  DRY_RUN=1                     forwarded to the delegated launcher if supported
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

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
      OMOMO_EXPECTED_TOTAL=128
      RESUME_FROM_BOX_EXPECTED_TOTAL=128
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

if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  DEFAULT_BOX_RESUME_RUN=${DEFAULT_BOX_RESUME_RUN:-"https://wandb.ai/zihanw22/boxer/runs/d9m3z369"}
  DEFAULT_BOX_RESUME_MODEL_FILE=${DEFAULT_BOX_RESUME_MODEL_FILE:-model_22000.pt}
  BOX_RESUME_MODEL_FILE=${BOX_RESUME_MODEL_FILE:-${DEFAULT_BOX_RESUME_MODEL_FILE}}
  export DEFAULT_BOX_RESUME_RUN
  export DEFAULT_BOX_RESUME_MODEL_FILE
  export BOX_RESUME_MODEL_FILE
fi

export AS_SUCCESS133_FINAL0P5
export RESUME_FROM_BOX
if [[ "${CORL_128:-0}" == "1" ]]; then
  export AS_SUCCESS133_BANK_NAME
  export OMOMO_EXPECTED_TOTAL
  export RESUME_FROM_BOX_EXPECTED_TOTAL
fi
export AS_CONTACT_AWARE=1
export ROOT_COMMAND_MODE="${ROOT_COMMAND_MODE:-contact-aware}"
export SCHEDULE_VARIANT="${SCHEDULE_VARIANT:-ppo_first}"
export STUDENT_ACTOR_INPUTS="${STUDENT_ACTOR_INPUTS:-['actor_obs_root_contact_aware','actor_obs_drop_button','actor_obs_proprio_with_actions_no_linvel']}"

if [[ "${CORL_128:-0}" == "1" && "${RESUME_FROM_BOX:-0}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_corl128_button_init_box}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_corl128_button_init_box_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_corl128_init_box_sparse_root_ppo_first_contact_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-CORL 128-clip curated AS teacher-rollout subset from success155_bcleb5oi58000_final0p5_primitiveproj, initialized from box-button actor policy parameters. The subset keeps all box/bin/barrel/ball clips, scale lamp, scale chair plus noscale__any_chair_85, all table clips, and the selected monitor set. It uses teacher-exported contact sidecars for offline contact guidance and adaptive contact-window sampling. The button interface adds actor_obs_drop_button, 0 before carry-end t2 and 1 from t2 through clip end.}"
elif [[ "${CORL_128:-0}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_corl128_button}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_corl128_button_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_corl128_sparse_root_ppo_first_contact_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-CORL 128-clip curated AS teacher-rollout subset from success155_bcleb5oi58000_final0p5_primitiveproj. The subset keeps all box/bin/barrel/ball clips, scale lamp, scale chair plus noscale__any_chair_85, all table clips, and the selected monitor set. It uses teacher-exported contact sidecars for offline contact guidance and adaptive contact-window sampling. The button interface adds actor_obs_drop_button, 0 before carry-end t2 and 1 from t2 through clip end.}"
elif [[ "${AS_SUCCESS133_FINAL0P5:-0}" == "1" && "${RESUME_FROM_BOX:-0}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_success133_final0p5_button_init_box}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_success133_final0p5_button_init_box_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_success133_final0p5_init_box_sparse_root_ppo_first_contact_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS teacher-rollout filtered 133-clip real-mesh perception distill initialized from box-button actor policy parameters. Clips satisfy stable_contact_success=True and final_object_position_error_m<=0.5, use teacher-exported contact sidecars for offline contact guidance and adaptive contact-window sampling, and keep the PPO+DAgger hybrid active from iteration 0. The button interface adds actor_obs_drop_button, 0 before carry-end t2 and 1 from t2 through clip end; root command behavior is unchanged.}"
elif [[ "${AS_SUCCESS133_FINAL0P5:-0}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_success133_final0p5_button}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_success133_final0p5_button_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_success133_final0p5_sparse_root_ppo_first_contact_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS teacher-rollout filtered 133-clip real-mesh perception distill with contact-aware sparse root and drop-button student input. Clips satisfy stable_contact_success=True and final_object_position_error_m<=0.5, use teacher-exported contact sidecars for offline contact guidance and adaptive contact-window sampling, and keep the PPO+DAgger hybrid active from iteration 0. The button interface adds actor_obs_drop_button, 0 before carry-end t2 and 1 from t2 through clip end; root command behavior is unchanged.}"
elif [[ "${RESUME_FROM_BOX:-0}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_keep169_button_init_box}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_keep169_button_init_box_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_keep169_init_box_sparse_root_ppo_first_contact_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS keep169 real-mesh perception distill initialized from box-button actor policy parameters. Training starts from iteration 0 with current AS data/contact/schedule, uses retarget-exported wrist contact sidecars for offline contact guidance and adaptive contact-window sampling, and keeps the PPO+DAgger hybrid active from iteration 0. The button interface adds actor_obs_drop_button, 0 before carry-end t2 and 1 from t2 through clip end; root command behavior is unchanged.}"
else
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_button}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_button_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_sparse_root_ppo_first_contact_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS real-mesh perception distill with contact-aware sparse root and drop-button student input. The button interface adds actor_obs_drop_button, 0 before carry-end t2 and 1 from t2 through clip end; root command behavior is unchanged.}"
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

exec bash "${SCRIPT_DIR}/distill_as_perception.sh" "${POSITIONAL[@]}"
