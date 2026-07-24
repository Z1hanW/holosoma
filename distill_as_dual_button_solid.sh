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
# Button t1/t2 come only from the sustained source-motion object-root rel-z
# lift window. Contact unions remain adaptive-sampling data; the independently
# configured root carry window (formal default: peak_height) is unchanged.

usage() {
  cat <<'EOF'
Usage:
  bash distill_as_dual_button_solid.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]
  CHECK_ONLY=1 bash distill_as_dual_button_solid.sh
  bash distill_as_dual_button_solid.sh --check-only

Behavior:
  Delegates solid-object data preparation and launch to distill_as_button_solid.sh,
  but changes the actor input contract to include actor_obs_pickup_button in
  addition to actor_obs_drop_button.

Button convention:
  t1/t2         = sustained kinematic-lift source window (object_z - root_z)
  pickup_button = 1 before kinematic t1, 0 from t1 through clip end
  drop_button   = 0 before kinematic t2, 1 from t2 through clip end
  Contact unions still drive adaptive sampling; root carry remains independent.

Useful env vars:
  RESUME_FROM_BOX=0          required: a single-button box actor is not shape-compatible with this dual-button actor
  DISTILL_AS_FORMAL_FRESH=1  require a fresh trajectory and reject every resume/policy-init alias
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
RESUME_FROM_PREVIOUS="$(normalize_bool RESUME_FROM_PREVIOUS "${RESUME_FROM_PREVIOUS:-0}")"
DISTILL_AS_FORMAL_FRESH="$(normalize_bool DISTILL_AS_FORMAL_FRESH "${DISTILL_AS_FORMAL_FRESH:-0}")"

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
  echo "[ERROR] Dual-button solid distillation requires the exact ordered 95D STUDENT_ACTOR_INPUTS contract: ${DUAL_BUTTON_STUDENT_ACTOR_INPUTS}" >&2
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

readonly DUAL_BUTTON_WINDOW_MODE=kinematic_lift
CONTACT_AWARE_BUTTON_WINDOW_MODE="${CONTACT_AWARE_BUTTON_WINDOW_MODE:-${DUAL_BUTTON_WINDOW_MODE}}"
if [[ "${CONTACT_AWARE_BUTTON_WINDOW_MODE}" != "${DUAL_BUTTON_WINDOW_MODE}" ]]; then
  echo "[ERROR] Dual-button solid distillation requires CONTACT_AWARE_BUTTON_WINDOW_MODE=${DUAL_BUTTON_WINDOW_MODE}. Got: ${CONTACT_AWARE_BUTTON_WINDOW_MODE}" >&2
  exit 2
fi

if [[ "${DISTILL_AS_FORMAL_FRESH}" == 1 ]]; then
  for formal_fresh_bool_alias in \
      RESUME_FROM_BOX RESUME_FROM_PREVIOUS WANDB_RESUME_SAME_RUN; do
    formal_fresh_bool_value="$(normalize_bool \
      "${formal_fresh_bool_alias}" "${!formal_fresh_bool_alias:-0}")"
    if [[ "${formal_fresh_bool_value}" != 0 ]]; then
      echo "[ERROR] DISTILL_AS_FORMAL_FRESH=1 requires ${formal_fresh_bool_alias}=0." >&2
      exit 2
    fi
  done
  for formal_fresh_checkpoint_alias in \
      RESUME_TRAINING_CKPT RESUME_CKPT RESUME_CHECKPOINT RESUME_SOURCE_REF \
      RESUME_WANDB_RUN_ID RESUME_WANDB_ID WANDB_RUN_ID \
      POLICY_INIT_CKPT POLICY_INIT_CHECKPOINT POLICY_INIT_SOURCE_REF \
      BOX_POLICY_INIT_REF BOX_RESUME_CKPT RESUME_FROM_BOX_CKPT \
      DEFAULT_BOX_RESUME_CHECKPOINT DEFAULT_BOX_RESUME_RUN \
      DEFAULT_BOX_RESUME_MODEL_FILE \
      PREVIOUS_RESUME_CKPT RESUME_FROM_PREVIOUS_CKPT PREVIOUS_RESUME_RUN \
      PREVIOUS_RESUME_MODEL_FILE DEFAULT_PREVIOUS_RESUME_RUN \
      AS_POLICY_INIT_PROFILE AS_TRAINING_RESUME_REF \
      RESUME_SOURCE_EXPECTED_SHA256 POLICY_INIT_EXPECTED_SHA256 \
      BOX_POLICY_INIT_EXPECTED_SHA256 \
      HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET \
      BOX_POLICY_INIT_EXPECTED_WORLD_SIZE BOX_POLICY_INIT_EXPECTED_WANDB_RUN_PATH \
      BOX_POLICY_INIT_EXPECTED_SOURCE_SNAPSHOT_ID \
      RESUME_MODEL_FILE WANDB_MODEL_FILE RESUME_STEP; do
    if [[ -n "${!formal_fresh_checkpoint_alias:-}" ]]; then
      echo "[ERROR] DISTILL_AS_FORMAL_FRESH=1 requires ${formal_fresh_checkpoint_alias} to be empty/unset." >&2
      exit 2
    fi
  done
  for formal_fresh_arg in "${POSITIONAL[@]}"; do
    case "${formal_fresh_arg//_/-}" in
      --training.checkpoint|--training.checkpoint=*|\
      --training.policy-init-checkpoint|--training.policy-init-checkpoint=*)
        echo "[ERROR] DISTILL_AS_FORMAL_FRESH=1 forbids forwarded resume/policy-init CLI: ${formal_fresh_arg}" >&2
        exit 2
        ;;
    esac
  done
  unset formal_fresh_bool_alias formal_fresh_bool_value
  unset formal_fresh_checkpoint_alias formal_fresh_arg
fi

if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  echo "[ERROR] RESUME_FROM_BOX=1 is incompatible with the dual-button actor: adding pickup_button changes the strict first-layer shape." >&2
  echo "[ERROR] Train from scratch or curriculum-resume an architecture-matched dual-button checkpoint with RESUME_CKPT (not bitwise trajectory continuation)." >&2
  exit 2
fi
if [[ "${RESUME_FROM_PREVIOUS}" == "1" ]]; then
  echo "[ERROR] RESUME_FROM_PREVIOUS=1 points to the saved single-button policy and is incompatible with the dual-button actor's additional pickup_button input." >&2
  echo "[ERROR] Use RESUME_CKPT with a full architecture-matched dual-button checkpoint." >&2
  exit 2
fi

export AS_SUCCESS133_FINAL0P5
export RESUME_FROM_BOX
export RESUME_FROM_PREVIOUS
export DISTILL_AS_FORMAL_FRESH
export STUDENT_ACTOR_INPUTS
export CONTACT_AWARE_BUTTON_WINDOW_MODE
export STUDENT_PROPRIO_HISTORY_LENGTH
export CONTACT_AWARE_HISTORY AS_CONTACT_AWARE_HISTORY CONTACT_AWARE_HISTORY_LENGTH
export HOLOSOMA_DUAL_BUTTON_HISTORY_CLI_OWNED

if [[ "${AS_SUCCESS133_FINAL0P5}" == "1" ]]; then
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_success133_final0p5_dual_button_solid}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_success133_final0p5_dual_button_solid_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_success133_final0p5_sparse_root_ppo_first_contact_pickup_drop_button_solid}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS teacher-rollout filtered solid-object perception distill with contact-aware sparse root plus pickup/drop inputs. Button t1/t2 use only the sustained source-motion object_z-root_z kinematic-lift window; contact unions remain adaptive-sampling data and the root carry window remains independently configured (formal default peak_height). Solid data prep, camera randomization, and DAgger/PPO schedule are inherited from distill_as_button_solid.sh.}"
else
  export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_dual_button_solid}"
  export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_dual_button_solid_depth}"
  export SCHEDULE_NAME="${SCHEDULE_NAME:-as_sparse_root_ppo_first_contact_pickup_drop_button_solid}"
  export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS solid-object perception distill with contact-aware sparse root plus pickup/drop inputs. Button t1/t2 use only the sustained source-motion object_z-root_z kinematic-lift window; contact unions remain adaptive-sampling data and the root carry window remains independently configured (formal default peak_height). Solid data prep, camera randomization, and DAgger/PPO schedule are inherited from distill_as_button_solid.sh.}"
fi

echo "[INFO] Launching AS/OMOMO dual-button solid perception distillation"
echo "[INFO] student_actor_inputs=${STUDENT_ACTOR_INPUTS}"
echo "[INFO] contact_aware_button_window_mode=${CONTACT_AWARE_BUTTON_WINDOW_MODE}"
echo "[INFO] button_window_semantics=source_object_root_rel_z_sustained_lift contact_union_role=adaptive_sampling root_carry_role=independent"
echo "[INFO] pickup_button_interface=1 pickup_button=1_before_t1_0_from_t1_to_end"
echo "[INFO] drop_button_interface=1 drop_button=0_before_t2_1_from_t2_to_end"

exec bash "${SCRIPT_DIR}/distill_as_button_solid.sh" \
  "${POSITIONAL[@]}" "${DUAL_BUTTON_CANONICAL_HISTORY_ARGS[@]}"
