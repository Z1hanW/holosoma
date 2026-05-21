#!/usr/bin/env bash
set -euo pipefail

# AS distillation on the solid-object subset with explicit pickup and drop
# buttons. This is a thin wrapper over distill_as_button_solid.sh: data prep,
# resume-from-box behavior, DAgger/PPO schedule, camera randomization, and all
# other training settings stay on the existing solid-button path.
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
  bash distill_as_dual_button_solid.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]
  bash distill_as_dual_button_solid.sh --resume-from-box [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]
  CHECK_ONLY=1 bash distill_as_dual_button_solid.sh
  bash distill_as_dual_button_solid.sh --check-only

Behavior:
  Delegates solid-object data preparation and launch to distill_as_button_solid.sh,
  but changes the actor input contract to include actor_obs_pickup_button in
  addition to actor_obs_drop_button.

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
while [[ $# -gt 0 ]]; do
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
    -h|--help|help)
      usage
      exit 0
      ;;
    dual|dual-button|dual_button|pickup-drop-button|pickup_drop_button)
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

AS_SUCCESS133_FINAL0P5="$(normalize_bool AS_SUCCESS133_FINAL0P5 "${AS_SUCCESS133_FINAL0P5:-1}")"
RESUME_FROM_BOX="$(normalize_bool RESUME_FROM_BOX "${RESUME_FROM_BOX:-0}")"

export AS_SUCCESS133_FINAL0P5
export RESUME_FROM_BOX
export STUDENT_ACTOR_INPUTS="${STUDENT_ACTOR_INPUTS:-['actor_obs_root_contact_aware','actor_obs_pickup_button','actor_obs_drop_button','actor_obs_proprio_with_actions_no_linvel']}"

if [[ "${AS_SUCCESS133_FINAL0P5}" == "1" && "${RESUME_FROM_BOX}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_success133_final0p5_dual_button_solid_init_box}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_success133_final0p5_dual_button_solid_init_box_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_success133_final0p5_init_box_sparse_root_ppo_first_contact_pickup_drop_button_solid}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS teacher-rollout filtered solid-object perception distill initialized from box-button policy parameters. Uses contact-aware sparse root plus pickup/drop button student inputs: pickup_button is 1 before carry-start t1 and 0 from t1 onward; drop_button is 0 before carry-end t2 and 1 from t2 through clip end. Solid data prep, contact sidecars, adaptive sampling, camera randomization, DAgger/PPO schedule, and training-from-iteration-0 behavior are inherited from distill_as_button_solid.sh.}"
elif [[ "${AS_SUCCESS133_FINAL0P5}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_success133_final0p5_dual_button_solid}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_success133_final0p5_dual_button_solid_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_success133_final0p5_sparse_root_ppo_first_contact_pickup_drop_button_solid}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS teacher-rollout filtered solid-object perception distill with contact-aware sparse root plus pickup/drop button student inputs: pickup_button is 1 before carry-start t1 and 0 from t1 onward; drop_button is 0 before carry-end t2 and 1 from t2 through clip end. Solid data prep, contact sidecars, adaptive sampling, camera randomization, and DAgger/PPO schedule are inherited from distill_as_button_solid.sh.}"
elif [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_keep169_dual_button_solid_init_box}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_keep169_dual_button_solid_init_box_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_keep169_init_box_sparse_root_ppo_first_contact_pickup_drop_button_solid}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS keep169 solid-object perception distill initialized from box-button policy parameters. Uses contact-aware sparse root plus pickup/drop button student inputs: pickup_button is 1 before carry-start t1 and 0 from t1 onward; drop_button is 0 before carry-end t2 and 1 from t2 through clip end. Solid data prep, contact sidecars, adaptive sampling, camera randomization, DAgger/PPO schedule, and training-from-iteration-0 behavior are inherited from distill_as_button_solid.sh.}"
else
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_dual_button_solid}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_dual_button_solid_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_sparse_root_ppo_first_contact_pickup_drop_button_solid}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS solid-object perception distill with contact-aware sparse root plus pickup/drop button student inputs: pickup_button is 1 before carry-start t1 and 0 from t1 onward; drop_button is 0 before carry-end t2 and 1 from t2 through clip end. Solid data prep, contact sidecars, adaptive sampling, camera randomization, and DAgger/PPO schedule are inherited from distill_as_button_solid.sh.}"
fi

echo "[INFO] Launching AS/OMOMO dual-button solid perception distillation"
echo "[INFO] student_actor_inputs=${STUDENT_ACTOR_INPUTS}"
echo "[INFO] pickup_button_interface=1 pickup_button=1_before_t1_0_from_t1_to_end"
echo "[INFO] drop_button_interface=1 drop_button=0_before_t2_1_from_t2_to_end"

exec bash "${SCRIPT_DIR}/distill_as_button_solid.sh" "${POSITIONAL[@]}"
