#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "${REPO_ROOT}"

TMP_DIR=$(mktemp -d)
cleanup() {
  chmod -R u+w "${TMP_DIR}" 2>/dev/null || true
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

expect_failure() {
  local output_file="$1" expected="$2"
  shift 2
  if "$@" >"${output_file}" 2>&1; then
    fail "command unexpectedly succeeded: $*"
  fi
  grep -F -- "${expected}" "${output_file}" >/dev/null || {
    sed -n '1,30p' "${output_file}" >&2
    fail "missing expected failure text: ${expected}"
  }
}

bash -n batch_ne.sh distill_as_dual_button_solid.sh

exact_inputs="['actor_obs_root_contact_aware','actor_obs_pickup_button','actor_obs_drop_button','actor_obs_proprio_with_actions_no_linvel']"

# Intercept only the final delegated `bash parent-wrapper ...` boundary.  The
# dual wrapper itself is invoked with /bin/bash, so no data, checkpoint, Python,
# simulator, or GPU work occurs in these focused contract tests.
mkdir -p "${TMP_DIR}/fake-bin"
cat >"${TMP_DIR}/fake-bin/bash" <<'EOF'
#!/bin/bash
set -euo pipefail
{
  printf 'parent=%s\n' "$1"
  printf 'formal_fresh=%s\n' "${DISTILL_AS_FORMAL_FRESH:-}"
  printf 'student_actor_inputs=%s\n' "${STUDENT_ACTOR_INPUTS:-}"
  printf 'resume_ckpt=%s\n' "${RESUME_CKPT:-}"
  printf 'args='
  printf '<%s>' "$@"
  printf '\n'
} >"${DUAL_PARENT_CAPTURE:?}"
EOF
chmod 700 "${TMP_DIR}/fake-bin/bash"

capture="${TMP_DIR}/dual-parent.capture"
env PATH="${TMP_DIR}/fake-bin:/usr/bin:/bin" \
  DUAL_PARENT_CAPTURE="${capture}" \
  DISTILL_AS_FORMAL_FRESH=1 \
  STUDENT_ACTOR_INPUTS="${exact_inputs}" \
  /bin/bash distill_as_dual_button_solid.sh --logger.entity=test >/dev/null
grep -F "parent=${REPO_ROOT}/distill_as_button_solid.sh" "${capture}" >/dev/null ||
  fail 'exact dual-button contract did not reach the solid parent wrapper'
grep -F 'formal_fresh=1' "${capture}" >/dev/null ||
  fail 'formal-fresh identity was not forwarded to the solid parent wrapper'
grep -F "student_actor_inputs=${exact_inputs}" "${capture}" >/dev/null ||
  fail 'exact ordered 95D actor inputs were not forwarded unchanged'

# The opt-in is intentionally not a behavior change for direct/manual runs.
# An architecture-matched full dual-button resume remains available when the
# formal-fresh profile is not selected.
rm -f "${capture}"
env PATH="${TMP_DIR}/fake-bin:/usr/bin:/bin" \
  DUAL_PARENT_CAPTURE="${capture}" \
  RESUME_CKPT=/tmp/manual-dual-checkpoint.pt \
  /bin/bash distill_as_dual_button_solid.sh >/dev/null
grep -F 'formal_fresh=0' "${capture}" >/dev/null ||
  fail 'direct/manual compatibility did not retain formal-fresh=0'
grep -F 'resume_ckpt=/tmp/manual-dual-checkpoint.pt' "${capture}" >/dev/null ||
  fail 'direct/manual architecture-matched resume was changed without opt-in'

for invalid_inputs in \
    "['actor_obs_root_contact_aware','actor_obs_drop_button','actor_obs_pickup_button','actor_obs_proprio_with_actions_no_linvel']" \
    "['actor_obs_root_contact_aware','actor_obs_pickup_button','actor_obs_drop_button']" \
    "['actor_obs_root_contact_aware','actor_obs_pickup_button','actor_obs_drop_button','actor_obs_proprio_with_actions_no_linvel','actor_obs_actions']"; do
  rm -f "${capture}"
  expect_failure \
    "${TMP_DIR}/invalid-input-$RANDOM.out" \
    'requires the exact ordered 95D STUDENT_ACTOR_INPUTS contract' \
    env PATH="${TMP_DIR}/fake-bin:/usr/bin:/bin" \
      DUAL_PARENT_CAPTURE="${capture}" \
      STUDENT_ACTOR_INPUTS="${invalid_inputs}" \
      /bin/bash distill_as_dual_button_solid.sh
  [[ ! -e "${capture}" ]] ||
    fail 'invalid dual-button actor inputs reached the parent wrapper'
done

formal_fresh_bool_aliases=(
  RESUME_FROM_BOX
  RESUME_FROM_PREVIOUS
  WANDB_RESUME_SAME_RUN
)
formal_fresh_checkpoint_aliases=(
  RESUME_TRAINING_CKPT RESUME_CKPT RESUME_CHECKPOINT RESUME_SOURCE_REF
  RESUME_WANDB_RUN_ID RESUME_WANDB_ID WANDB_RUN_ID
  POLICY_INIT_CKPT POLICY_INIT_CHECKPOINT POLICY_INIT_SOURCE_REF
  BOX_POLICY_INIT_REF BOX_RESUME_CKPT RESUME_FROM_BOX_CKPT
  DEFAULT_BOX_RESUME_CHECKPOINT DEFAULT_BOX_RESUME_RUN DEFAULT_BOX_RESUME_MODEL_FILE
  PREVIOUS_RESUME_CKPT RESUME_FROM_PREVIOUS_CKPT PREVIOUS_RESUME_RUN
  PREVIOUS_RESUME_MODEL_FILE DEFAULT_PREVIOUS_RESUME_RUN
  AS_POLICY_INIT_PROFILE AS_TRAINING_RESUME_REF
  RESUME_SOURCE_EXPECTED_SHA256 POLICY_INIT_EXPECTED_SHA256
  BOX_POLICY_INIT_EXPECTED_SHA256 HOLOSOMA_POLICY_INIT_REQUIRED_TERMINAL_TARGET
  BOX_POLICY_INIT_EXPECTED_WORLD_SIZE BOX_POLICY_INIT_EXPECTED_WANDB_RUN_PATH
  BOX_POLICY_INIT_EXPECTED_SOURCE_SNAPSHOT_ID
  RESUME_MODEL_FILE WANDB_MODEL_FILE RESUME_STEP
)
for alias_name in "${formal_fresh_bool_aliases[@]}"; do
  rm -f "${capture}"
  expect_failure \
    "${TMP_DIR}/formal-bool-${alias_name}.out" \
    "DISTILL_AS_FORMAL_FRESH=1 requires ${alias_name}=0" \
    env PATH="${TMP_DIR}/fake-bin:/usr/bin:/bin" \
      DUAL_PARENT_CAPTURE="${capture}" \
      DISTILL_AS_FORMAL_FRESH=1 "${alias_name}=1" \
      /bin/bash distill_as_dual_button_solid.sh
  [[ ! -e "${capture}" ]] ||
    fail "formal-fresh boolean alias ${alias_name} reached the parent wrapper"
done
for alias_name in "${formal_fresh_checkpoint_aliases[@]}"; do
  rm -f "${capture}"
  expect_failure \
    "${TMP_DIR}/formal-checkpoint-${alias_name}.out" \
    "DISTILL_AS_FORMAL_FRESH=1 requires ${alias_name} to be empty/unset" \
    env PATH="${TMP_DIR}/fake-bin:/usr/bin:/bin" \
      DUAL_PARENT_CAPTURE="${capture}" \
      DISTILL_AS_FORMAL_FRESH=1 "${alias_name}=fixture" \
      /bin/bash distill_as_dual_button_solid.sh
  [[ ! -e "${capture}" ]] ||
    fail "formal-fresh checkpoint alias ${alias_name} reached the parent wrapper"
done
unset alias_name

for forbidden_cli in \
    --training.checkpoint=/tmp/resume.pt \
    --training.policy-init-checkpoint=/tmp/init.pt; do
  rm -f "${capture}"
  expect_failure \
    "${TMP_DIR}/formal-cli-$RANDOM.out" \
    'DISTILL_AS_FORMAL_FRESH=1 forbids forwarded resume/policy-init CLI' \
    env PATH="${TMP_DIR}/fake-bin:/usr/bin:/bin" \
      DUAL_PARENT_CAPTURE="${capture}" \
      DISTILL_AS_FORMAL_FRESH=1 \
      /bin/bash distill_as_dual_button_solid.sh "${forbidden_cli}"
  [[ ! -e "${capture}" ]] ||
    fail "formal-fresh CLI ${forbidden_cli} reached the parent wrapper"
done

batch_base=(
  env
  NODES=test-node
  NNODES=1
  NPROC=1
  CUDA_VISIBLE_DEVICES=0
  PER_GPU_ENVS=1024
  DRY_RUN=1
  PREPARE_DATA=0
  SKIP_GIT_PULL=1
  SKIP_NODE_HEALTH_CHECK=1
  HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY=0
  SOURCE_SNAPSHOT_CACHE="${TMP_DIR}/snapshot-cache"
)

for invalid_entrypoint in \
    ../distill_as_dual_button_solid.sh \
    /tmp/distill_as_dual_button_solid.sh \
    distill_as_dual_button.sh \
    'distill_as_dual_button_solid.sh;true'; do
  invalid_output="${TMP_DIR}/invalid-entrypoint-$RANDOM.out"
  expect_failure \
    "${invalid_output}" \
    'DISTILL_AS_ENTRYPOINT must be one allowlisted bare repo-local filename' \
    "${batch_base[@]}" DISTILL_AS_ENTRYPOINT="${invalid_entrypoint}" \
      bash batch_ne.sh launch
  if rg -n 'source_snapshot_id=|\[DRY_RUN\].*(ssh|scp)' \
      "${invalid_output}" >/dev/null; then
    fail 'invalid entrypoint reached snapshot construction or remote actions'
  fi
done

for invalid_config_spec in \
    'DISTILL_EXPERIMENT_CONFIG=../experiment|DISTILL_EXPERIMENT_CONFIG must be empty or one safe config name' \
    'DISTILL_REWARD_CONFIG=reward:injected|DISTILL_REWARD_CONFIG must be empty or one safe config name' \
    'DISTILL_REWARD_CONFIG=reward;true|DISTILL_REWARD_CONFIG must be empty or one safe config name'; do
  invalid_config_assignment=${invalid_config_spec%%|*}
  invalid_config_error=${invalid_config_spec#*|}
  invalid_config_name=${invalid_config_assignment%%=*}
  invalid_config_value=${invalid_config_assignment#*=}
  invalid_config_output="${TMP_DIR}/invalid-config-${invalid_config_name}-$RANDOM.out"
  expect_failure \
    "${invalid_config_output}" \
    "${invalid_config_error}" \
    "${batch_base[@]}" "${invalid_config_name}=${invalid_config_value}" \
      bash batch_ne.sh launch
  if rg -n 'source_snapshot_id=|\[DRY_RUN\].*(ssh|scp)' \
      "${invalid_config_output}" >/dev/null; then
    fail "invalid ${invalid_config_name} reached snapshot construction or remote actions"
  fi
done
unset invalid_config_spec invalid_config_assignment invalid_config_error
unset invalid_config_name invalid_config_value invalid_config_output

# Emergency control actions recover entrypoint identity from the hash-bound
# active control.  A stale/poisoned training shell must not prevent status or
# stop from reaching that durable state.
"${batch_base[@]}" \
  DISTILL_AS_ENTRYPOINT='../../poisoned.sh' \
  DISTILL_AS_FORMAL_FRESH=invalid \
  RESUME_CKPT=/tmp/stale.pt \
  bash batch_ne.sh status >"${TMP_DIR}/poisoned-status.out"
grep -F '[DRY_RUN] bounded-ssh timeout=30s node=test-node ' \
  "${TMP_DIR}/poisoned-status.out" >/dev/null ||
  fail 'poisoned training entrypoint state blocked emergency status'

expect_failure \
  "${TMP_DIR}/formal-default-entrypoint.out" \
  'DISTILL_AS_FORMAL_FRESH=1 is defined only for DISTILL_AS_ENTRYPOINT=distill_as_dual_button_solid.sh' \
  "${batch_base[@]}" DISTILL_AS_FORMAL_FRESH=1 bash batch_ne.sh launch

expect_failure \
  "${TMP_DIR}/formal-v1-replay.out" \
  'DISTILL_AS_FORMAL_FRESH=1 requires REPLAY_PREFLIGHT_REQUIRED_VERSION=2' \
  "${batch_base[@]}" \
    DISTILL_AS_ENTRYPOINT=distill_as_dual_button_solid.sh \
    DISTILL_AS_FORMAL_FRESH=1 REPLAY_PREFLIGHT_REQUIRED_VERSION=1 \
    bash batch_ne.sh launch
if rg -n 'source_snapshot_id=|\[DRY_RUN\].*(ssh|scp)' \
    "${TMP_DIR}/formal-v1-replay.out" >/dev/null; then
  fail 'formal-fresh v1 replay request reached snapshot construction or remote actions'
fi

for history_spec in \
    'STUDENT_PROPRIO_HISTORY_LENGTH=5|requires STUDENT_PROPRIO_HISTORY_LENGTH=1' \
    'CONTACT_AWARE_HISTORY_LENGTH=5|requires CONTACT_AWARE_HISTORY_LENGTH=1' \
    'CONTACT_AWARE_HISTORY=1|requires CONTACT_AWARE_HISTORY=0' \
    'AS_CONTACT_AWARE_HISTORY=true|requires AS_CONTACT_AWARE_HISTORY=0'; do
  history_assignment=${history_spec%%|*}
  history_error=${history_spec#*|}
  history_name=${history_assignment%%=*}
  history_value=${history_assignment#*=}
  history_output="${TMP_DIR}/formal-history-${history_name}.out"
  expect_failure \
    "${history_output}" \
    "${history_error}" \
    "${batch_base[@]}" \
      DISTILL_AS_ENTRYPOINT=distill_as_dual_button_solid.sh \
      DISTILL_AS_FORMAL_FRESH=1 "${history_name}=${history_value}" \
      bash batch_ne.sh launch
  if rg -n 'source_snapshot_id=|\[DRY_RUN\].*(ssh|scp)' \
      "${history_output}" >/dev/null; then
    fail "formal-fresh actor-history override ${history_name} reached snapshot construction or remote actions"
  fi
done
unset history_spec history_assignment history_error history_name history_value history_output

for v2_wiring in \
    '--required-manifest-version "${REPLAY_PREFLIGHT_REQUIRED_VERSION}"' \
    '--expected-source-archive-sha256 "${SOURCE_SNAPSHOT_ARCHIVE_SHA256}"' \
    '--expected-entrypoint-archive-member "${DISTILL_AS_ENTRYPOINT}"' \
    '--expected-entrypoint-sha256 "${DISTILL_AS_ENTRYPOINT_SHA256}"' \
    'A non-dry-run DISTILL_AS_FORMAL_FRESH=1 launch requires a fresh W&B identity and a Rule-90 v2 replay manifest.'; do
  grep -F -- "${v2_wiring}" batch_ne.sh >/dev/null ||
    fail "batch formal-fresh launch omitted Rule-90 v2 wiring: ${v2_wiring}"
done
unset v2_wiring

expect_failure \
  "${TMP_DIR}/controller-formal-resume.out" \
  'DISTILL_AS_FORMAL_FRESH=1 requires RESUME_CKPT to be empty/unset' \
  "${batch_base[@]}" \
    DISTILL_AS_ENTRYPOINT=distill_as_dual_button_solid.sh \
    DISTILL_AS_FORMAL_FRESH=1 RESUME_CKPT=/tmp/resume.pt \
    bash batch_ne.sh launch
if rg -n 'source_snapshot_id=|\[DRY_RUN\].*(ssh|scp)' \
    "${TMP_DIR}/controller-formal-resume.out" >/dev/null; then
  fail 'formal-fresh resume alias reached snapshot construction or remote actions'
fi

expect_failure \
  "${TMP_DIR}/controller-wrong-dual-inputs.out" \
  'requires the exact ordered 95D STUDENT_ACTOR_INPUTS contract' \
  "${batch_base[@]}" \
    DISTILL_AS_ENTRYPOINT=distill_as_dual_button_solid.sh \
    STUDENT_ACTOR_INPUTS="['actor_obs_root_contact_aware','actor_obs_drop_button','actor_obs_pickup_button','actor_obs_proprio_with_actions_no_linvel']" \
    bash batch_ne.sh launch
if rg -n 'source_snapshot_id=|\[DRY_RUN\].*(ssh|scp)' \
    "${TMP_DIR}/controller-wrong-dual-inputs.out" >/dev/null; then
  fail 'wrong dual-button actor inputs reached snapshot construction or remote actions'
fi

dual_output="${TMP_DIR}/dual-batch.out"
"${batch_base[@]}" \
  DISTILL_AS_ENTRYPOINT=distill_as_dual_button_solid.sh \
  DISTILL_EXPERIMENT_CONFIG=g1-29dof-wbt-w-object-distill-sparse-root-cmd-teacher-linvel \
  DISTILL_REWARD_CONFIG=g1-29dof-wbt-w-object-generalist-offline-contact-guidance \
  DISTILL_AS_FORMAL_FRESH=1 \
  STUDENT_ACTOR_INPUTS="${exact_inputs}" \
  bash batch_ne.sh launch >"${dual_output}"

snapshot_id=$(sed -nE 's/^\[INFO\] source_snapshot_id=(src-[0-9a-f]{64}) .*/\1/p' \
  "${dual_output}" | head -1)
[[ "${snapshot_id}" =~ ^src-[0-9a-f]{64}$ ]] ||
  fail 'dual batch did not report one content-addressed source snapshot'
entrypoint_sha=$(sha256sum distill_as_dual_button_solid.sh | awk '{print $1}')
entrypoint_path="/home/ubuntu/FAR/holosoma_runs/${snapshot_id}/distill_as_dual_button_solid.sh"
grep -F "[INFO] distill_as_entrypoint=distill_as_dual_button_solid.sh path=${entrypoint_path} sha256=${entrypoint_sha} formal_fresh=1" \
  "${dual_output}" >/dev/null ||
  fail 'controller did not bind the selected dual entrypoint to snapshot path/SHA/profile'
for expected_control_line in \
    'export DISTILL_AS_ENTRYPOINT=distill_as_dual_button_solid.sh' \
    "export DISTILL_AS_ENTRYPOINT_PATH=${entrypoint_path}" \
    "export DISTILL_AS_ENTRYPOINT_SHA256=${entrypoint_sha}" \
    'export DISTILL_EXPERIMENT_CONFIG=g1-29dof-wbt-w-object-distill-sparse-root-cmd-teacher-linvel' \
    'export DISTILL_REWARD_CONFIG=g1-29dof-wbt-w-object-generalist-offline-contact-guidance' \
    'export EXP=g1-29dof-wbt-w-object-distill-sparse-root-cmd-teacher-linvel' \
    'export DISTILL_AS_FORMAL_FRESH=1' \
    'export HOLOSOMA_RUNTIME_SCRATCH_ROOT=' \
    'export TMPDIR=' \
    'export HOLOSOMA_OBJECT_USD_CACHE_DIR=' \
    'export HOLOSOMA_ROBOT_USD_CACHE_DIR=' \
    'Runtime scratch requires at least 4 GiB free' \
    'runtime_scratch_verified=${HOLOSOMA_RUNTIME_SCRATCH_ROOT}' \
    'export STUDENT_PROPRIO_HISTORY_LENGTH=1' \
    'TRAIN_EXTRA_ARGS+=("reward:${DISTILL_REWARD_CONFIG}")' \
    "actual_distill_as_entrypoint_sha256=\$(sha256sum -- \"\${DISTILL_AS_ENTRYPOINT_PATH}\" | awk '{print \$1}')" \
    'bash "${DISTILL_AS_ENTRYPOINT_PATH}" "${TRAIN_EXTRA_ARGS[@]}" 2>&1 | tee -a '; do
  grep -F -- "${expected_control_line}" "${dual_output}" >/dev/null ||
    fail "generated dual node control omitted entrypoint binding: ${expected_control_line}"
done
if grep -F 'TRAIN_EXTRA_ARGS+=("exp:${DISTILL_EXPERIMENT_CONFIG}")' \
    "${dual_output}" >/dev/null; then
  fail 'generated dual node control retained a second late experiment selector'
fi
for history_group in \
    actor-obs-root-contact-aware \
    actor-obs-pickup-button \
    actor-obs-drop-button \
    actor-obs-proprio-with-actions-no-linvel; do
  grep -F -- "--observation.groups.${history_group}.history-length|" \
    "${dual_output}" >/dev/null ||
    fail "generated formal dual control omitted split-form history override guard for ${history_group}"
  grep -F -- "--observation.groups.${history_group}.history-length=*" \
    "${dual_output}" >/dev/null ||
    fail "generated formal dual control omitted equals-form history override guard for ${history_group}"
done
unset history_group
if grep -F 'bash distill_as_button_solid.sh "${TRAIN_EXTRA_ARGS[@]}" 2>&1 | tee -a ' \
    "${dual_output}" >/dev/null; then
  fail 'generated dual node control retained the hardcoded single-button bypass'
fi

default_output="${TMP_DIR}/default-batch.out"
"${batch_base[@]}" bash batch_ne.sh launch >"${default_output}"
grep -F 'export DISTILL_AS_ENTRYPOINT=distill_as_button_solid.sh' \
  "${default_output}" >/dev/null ||
  fail 'default batch entrypoint compatibility changed'
grep -F 'export DISTILL_AS_FORMAL_FRESH=0' "${default_output}" >/dev/null ||
  fail 'default batch unexpectedly enabled formal-fresh mode'

# Exercise construction of the modern stop preflight without contacting a
# node.  This catches controller-heredoc expansion bugs: the sed end anchor
# must survive command generation as a literal `$`, rather than becoming the
# controller's positional-argument count before the remote command runs.
main_case_line=$(rg -n '^case "\$\{ACTION\}" in$' batch_ne.sh | tail -1 | cut -d: -f1)
[[ "${main_case_line}" =~ ^[1-9][0-9]*$ ]] ||
  fail 'could not locate the batch launcher main action dispatch'
{
  sed -e "s#^SCRIPT_DIR=.*#SCRIPT_DIR=${REPO_ROOT}#" \
      -e "${main_case_line},\$d" batch_ne.sh
  printf '%s\n' \
    'remote_run_bounded() { printf "%s\n" "$2"; }' \
    'read_stop_identity_node test-node'
} | env NODES=test-node REMOTE_RUN_ROOT="${TMP_DIR}/remote-stop-audit" \
      bash -s stop >"${TMP_DIR}/generated-modern-stop-preflight.sh"
grep -F 'DISTILL_AS_ENTRYPOINT_PATH=([^[:space:]]+)$#\1#p' \
  "${TMP_DIR}/generated-modern-stop-preflight.sh" >/dev/null ||
  fail 'modern stop preflight lost the selected-entrypoint path end anchor during heredoc expansion'
if grep -F 'DISTILL_AS_ENTRYPOINT_PATH=([^[:space:]]+)1#\1#p' \
    "${TMP_DIR}/generated-modern-stop-preflight.sh" >/dev/null; then
  fail 'modern stop preflight substituted the controller argument count into its sed expression'
fi

echo '[PASS] dual-button entrypoint/formal-fresh launcher contract'
