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
# Button t1/t2 come only from the sustained source-motion object-root rel-z
# lift window. Contact unions remain adaptive-sampling data; the independently
# configured root carry window (formal default: peak_height) is unchanged.

usage() {
  cat <<'EOF'
Usage:
  bash distill_as_dual_button.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]
  bash distill_as_dual_button.sh corl_128 [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]

Behavior:
  Delegates AS data preparation and launch to distill_as_button.sh, but changes
  the actor input contract to include actor_obs_pickup_button in addition to
  actor_obs_drop_button.

Examples:
  bash cp_corl.sh
  bash distill_as_dual_button.sh corl_128
  # If data/ds_as_data/corl_128 exists and no other data source is selected,
  # this wrapper defaults to corl_128.

Button convention:
  t1/t2         = sustained kinematic-lift source window (object_z - root_z)
  pickup_button = 1 before kinematic t1, 0 from t1 through clip end
  drop_button   = 0 before kinematic t2, 1 from t2 through clip end
  Contact unions still drive adaptive sampling; root carry remains independent.

Useful env vars:
  RESUME_FROM_BOX=0          required: a single-button box actor is not shape-compatible with this dual-button actor
  RUN_NAME=<name>            override W&B run display name
  TRAINING_NAME=<name>       override log/checkpoint training name
  SCHEDULE_NAME=<name>       override schedule label
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

POSITIONAL=()
CORL_128=0
DATA_SELECTOR_SET=0
USER_SET_AS_SUCCESS133_BANK_NAME=${AS_SUCCESS133_BANK_NAME+x}
USER_SET_AS_SUCCESS133_FINAL0P5=${AS_SUCCESS133_FINAL0P5+x}
USER_SET_OMOMO_DATA_DIR=${OMOMO_DATA_DIR+x}
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
      DATA_SELECTOR_SET=1
      AS_SUCCESS133_FINAL0P5=1
      POSITIONAL+=("$1")
      shift
      ;;
    success133|as-success133|as_success133|success133-final0p5|success133_final0p5)
      DATA_SELECTOR_SET=1
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

if [[ "${CORL_128}" == "0" \
      && "${DATA_SELECTOR_SET}" == "0" \
      && -z "${USER_SET_AS_SUCCESS133_BANK_NAME}" \
      && -z "${USER_SET_AS_SUCCESS133_FINAL0P5}" \
      && -z "${USER_SET_OMOMO_DATA_DIR}" \
      && -d "${SCRIPT_DIR}/data/ds_as_data/corl_128" ]]; then
  CORL_128=1
  AS_SUCCESS133_FINAL0P5=1
  POSITIONAL=("corl_128" "${POSITIONAL[@]}")
fi

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
if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  echo "[ERROR] RESUME_FROM_BOX=1 is incompatible with the dual-button actor: adding pickup_button changes the strict first-layer shape." >&2
  echo "[ERROR] Train the dual-button actor from scratch or curriculum-resume an architecture-matched dual-button checkpoint with RESUME_CKPT (not bitwise trajectory continuation)." >&2
  exit 2
fi

readonly DUAL_BUTTON_STUDENT_ACTOR_INPUTS="['actor_obs_root_contact_aware','actor_obs_pickup_button','actor_obs_drop_button','actor_obs_proprio_with_actions_no_linvel']"
readonly -a DUAL_BUTTON_HISTORY_GROUPS=(
  actor_obs_root_contact_aware
  actor_obs_pickup_button
  actor_obs_drop_button
  actor_obs_proprio_with_actions_no_linvel
)
STUDENT_ACTOR_INPUTS="${STUDENT_ACTOR_INPUTS:-${DUAL_BUTTON_STUDENT_ACTOR_INPUTS}}"
if [[ "${STUDENT_ACTOR_INPUTS//[[:space:]]/}" \
      != "${DUAL_BUTTON_STUDENT_ACTOR_INPUTS}" ]]; then
  echo "[ERROR] Dual-button distillation requires the exact ordered 95D STUDENT_ACTOR_INPUTS contract: ${DUAL_BUTTON_STUDENT_ACTOR_INPUTS}" >&2
  echo "[ERROR] Got STUDENT_ACTOR_INPUTS=${STUDENT_ACTOR_INPUTS}" >&2
  exit 2
fi
for dual_button_arg in "${POSITIONAL[@]}"; do
  dual_button_option=${dual_button_arg%%=*}
  dual_button_option=${dual_button_option,,}
  dual_button_option=${dual_button_option//_/-}
  case "${dual_button_option}" in
    --observation.groups.actor-obs-root-contact-aware.history-length|\
    --observation.groups.actor-obs-pickup-button.history-length|\
    --observation.groups.actor-obs-drop-button.history-length|\
    --observation.groups.actor-obs-proprio-with-actions-no-linvel.history-length)
      echo "[ERROR] Dual-button actor history is launcher-owned and fixed at 1; remove forwarded override ${dual_button_arg}." >&2
      exit 2
      ;;
  esac
done
if [[ -n "${STUDENT_PROPRIO_HISTORY_LENGTH+x}" \
      && "${STUDENT_PROPRIO_HISTORY_LENGTH}" != 1 ]]; then
  echo "[ERROR] Dual-button exact 95D actor requires STUDENT_PROPRIO_HISTORY_LENGTH=1." >&2
  exit 2
fi
if [[ -n "${CONTACT_AWARE_HISTORY_LENGTH+x}" \
      && "${CONTACT_AWARE_HISTORY_LENGTH}" != 1 ]]; then
  echo "[ERROR] Dual-button exact 95D actor requires CONTACT_AWARE_HISTORY_LENGTH=1 when supplied." >&2
  exit 2
fi
for dual_button_history_bool in CONTACT_AWARE_HISTORY AS_CONTACT_AWARE_HISTORY; do
  dual_button_history_value="$(normalize_bool \
    "${dual_button_history_bool}" "${!dual_button_history_bool:-0}")"
  if [[ "${dual_button_history_value}" != 0 ]]; then
    echo "[ERROR] Dual-button exact 95D actor requires ${dual_button_history_bool}=0." >&2
    exit 2
  fi
done
STUDENT_PROPRIO_HISTORY_LENGTH=1
CONTACT_AWARE_HISTORY=0
AS_CONTACT_AWARE_HISTORY=0
CONTACT_AWARE_HISTORY_LENGTH=1
HOLOSOMA_DUAL_BUTTON_HISTORY_CLI_OWNED=1
DUAL_BUTTON_CANONICAL_HISTORY_ARGS=()
for dual_button_history_group in "${DUAL_BUTTON_HISTORY_GROUPS[@]}"; do
  DUAL_BUTTON_CANONICAL_HISTORY_ARGS+=(
    "--observation.groups.${dual_button_history_group}.history-length=1"
  )
done
unset dual_button_arg dual_button_option
unset dual_button_history_bool dual_button_history_value dual_button_history_group

export AS_SUCCESS133_FINAL0P5
export RESUME_FROM_BOX
if [[ "${CORL_128}" == "1" ]]; then
  export OMOMO_EXPECTED_TOTAL="${OMOMO_EXPECTED_TOTAL:-128}"
  export RESUME_FROM_BOX_EXPECTED_TOTAL="${RESUME_FROM_BOX_EXPECTED_TOTAL:-128}"
fi
export AS_CONTACT_AWARE=1
export ROOT_COMMAND_MODE="${ROOT_COMMAND_MODE:-contact-aware}"
export SCHEDULE_VARIANT="${SCHEDULE_VARIANT:-ppo_first}"
export STUDENT_ACTOR_INPUTS
export STUDENT_PROPRIO_HISTORY_LENGTH
export CONTACT_AWARE_HISTORY AS_CONTACT_AWARE_HISTORY CONTACT_AWARE_HISTORY_LENGTH
export HOLOSOMA_DUAL_BUTTON_HISTORY_CLI_OWNED
readonly DUAL_BUTTON_WINDOW_MODE=kinematic_lift
CONTACT_AWARE_BUTTON_WINDOW_MODE="${CONTACT_AWARE_BUTTON_WINDOW_MODE:-${DUAL_BUTTON_WINDOW_MODE}}"
if [[ "${CONTACT_AWARE_BUTTON_WINDOW_MODE}" != "${DUAL_BUTTON_WINDOW_MODE}" ]]; then
  echo "[ERROR] Dual-button distillation requires CONTACT_AWARE_BUTTON_WINDOW_MODE=${DUAL_BUTTON_WINDOW_MODE}. Got: ${CONTACT_AWARE_BUTTON_WINDOW_MODE}" >&2
  exit 2
fi
export CONTACT_AWARE_BUTTON_WINDOW_MODE

if [[ "${CORL_128}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_corl128_dual_button}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_corl128_dual_button_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_corl128_sparse_root_ppo_first_contact_pickup_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-CORL dual-button AS distill. Button t1/t2 are the sustained source-motion object_z-root_z kinematic-lift window: pickup is 1 before t1 and drop is 1 from t2. Independently, the all-region contact union reweights adaptive timestep sampling and the root carry window keeps its configured semantics (formal default peak_height). Start-at-zero is an explicit reset mixture; PPO starts at the configured iteration-0 coefficient.}"
elif [[ "${AS_SUCCESS133_FINAL0P5}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_success133_final0p5_dual_button}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_success133_final0p5_dual_button_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_success133_final0p5_sparse_root_ppo_first_contact_pickup_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS teacher-rollout filtered perception distill with contact-aware sparse root plus pickup/drop inputs. Button t1/t2 use only the sustained source-motion object_z-root_z kinematic-lift window; contact unions remain adaptive-sampling data and the root carry window remains independently configured (formal default peak_height). Data prep, camera randomization, and DAgger/PPO schedule are inherited from distill_as_button.sh.}"
else
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_dual_button}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_dual_button_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_sparse_root_ppo_first_contact_pickup_drop_button}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS perception distill with contact-aware sparse root plus pickup/drop inputs. Button t1/t2 use only the sustained source-motion object_z-root_z kinematic-lift window; contact unions remain adaptive-sampling data and the root carry window remains independently configured (formal default peak_height). Data prep, camera randomization, and DAgger/PPO schedule are inherited from distill_as_button.sh.}"
fi

echo "[INFO] Launching AS/OMOMO dual-button perception distillation"
echo "[INFO] student_actor_inputs=${STUDENT_ACTOR_INPUTS}"
echo "[INFO] contact_aware_button_window_mode=${CONTACT_AWARE_BUTTON_WINDOW_MODE}"
echo "[INFO] button_window_semantics=source_object_root_rel_z_sustained_lift contact_union_role=adaptive_sampling root_carry_role=independent"
echo "[INFO] pickup_button_interface=1 pickup_button=1_before_t1_0_from_t1_to_end"
echo "[INFO] drop_button_interface=1 drop_button=0_before_t2_1_from_t2_to_end"

exec bash "${SCRIPT_DIR}/distill_as_button.sh" \
  "${POSITIONAL[@]}" "${DUAL_BUTTON_CANONICAL_HISTORY_ARGS[@]}"
