#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "${REPO_ROOT}"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "${TMP_DIR}"' EXIT

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
    sed -n '1,40p' "${output_file}" >&2
    fail "missing expected failure text: ${expected}"
  }
}

bash -n distill_box_perception.sh distill_as_dual_button.sh distill_as_dual_button_solid.sh

mkdir -p "${TMP_DIR}/fake-bin"
cat >"${TMP_DIR}/fake-bin/bash" <<'EOF'
#!/bin/bash
printf '%s\n' "${CONTACT_AWARE_BUTTON_WINDOW_MODE:-<unset>}" >"${BUTTON_MODE_CAPTURE:?}"
if [[ -n "${DUAL_HISTORY_ARGS_CAPTURE:-}" ]]; then
  printf '%s\n' "$@" >"${DUAL_HISTORY_ARGS_CAPTURE}"
  printf '%s\n' \
    "${STUDENT_ACTOR_INPUTS:-<unset>}" \
    "${STUDENT_PROPRIO_HISTORY_LENGTH:-<unset>}" \
    "${CONTACT_AWARE_HISTORY:-<unset>}" \
    "${AS_CONTACT_AWARE_HISTORY:-<unset>}" \
    "${CONTACT_AWARE_HISTORY_LENGTH:-<unset>}" \
    "${HOLOSOMA_DUAL_BUTTON_HISTORY_CLI_OWNED:-<unset>}" \
    >"${DUAL_HISTORY_ENV_CAPTURE:?}"
fi
EOF
chmod 700 "${TMP_DIR}/fake-bin/bash"

DUAL_ACTOR_INPUTS="['actor_obs_root_contact_aware','actor_obs_pickup_button','actor_obs_drop_button','actor_obs_proprio_with_actions_no_linvel']"
DUAL_HISTORY_GROUPS=(
  actor_obs_root_contact_aware
  actor_obs_pickup_button
  actor_obs_drop_button
  actor_obs_proprio_with_actions_no_linvel
)

for wrapper in distill_as_dual_button.sh distill_as_dual_button_solid.sh; do
  capture="${TMP_DIR}/${wrapper}.capture"
  args_capture="${TMP_DIR}/${wrapper}.args"
  env_capture="${TMP_DIR}/${wrapper}.env"
  env PATH="${TMP_DIR}/fake-bin:/usr/bin:/bin" BUTTON_MODE_CAPTURE="${capture}" \
    DUAL_HISTORY_ARGS_CAPTURE="${args_capture}" \
    DUAL_HISTORY_ENV_CAPTURE="${env_capture}" \
    TEACHER_OBS_KEYS=actor_obs_pickup_button \
    TEACHER_ACTOR_OBS_HISTORY_LENGTH=5 \
    /bin/bash "${wrapper}" >/dev/null
  [[ "$(<"${capture}")" == kinematic_lift ]] ||
    fail "${wrapper} did not force kinematic_lift"

  mapfile -t captured_args <"${args_capture}"
  (( ${#captured_args[@]} >= ${#DUAL_HISTORY_GROUPS[@]} )) ||
    fail "${wrapper} did not forward all four canonical history values"
  history_tail_start=$(( ${#captured_args[@]} - ${#DUAL_HISTORY_GROUPS[@]} ))
  for history_idx in "${!DUAL_HISTORY_GROUPS[@]}"; do
    expected_history="--observation.groups.${DUAL_HISTORY_GROUPS[history_idx]}.history-length=1"
    [[ "${captured_args[history_tail_start + history_idx]}" == "${expected_history}" ]] ||
      fail "${wrapper} canonical history tail is out of order: expected ${expected_history}"
    history_count=$(grep -Fxc -- "${expected_history}" "${args_capture}" || true)
    [[ "${history_count}" == 1 ]] ||
      fail "${wrapper} must forward ${expected_history} exactly once; got ${history_count}"
  done
  mapfile -t captured_env <"${env_capture}"
  [[ "${captured_env[0]}" == "${DUAL_ACTOR_INPUTS}" ]] ||
    fail "${wrapper} lost the exact ordered dual actor input contract"
  [[ "${captured_env[1]}" == 1 ]] || fail "${wrapper} did not pin student proprio history to 1"
  [[ "${captured_env[2]}" == 0 && "${captured_env[3]}" == 0 ]] ||
    fail "${wrapper} did not disable both contact-aware history selectors"
  [[ "${captured_env[4]}" == 1 ]] || fail "${wrapper} did not pin contact-aware history length to 1"
  [[ "${captured_env[5]}" == 1 ]] || fail "${wrapper} did not publish its internal CLI-ownership marker"

  rm -f "${capture}"
  expect_failure \
    "${TMP_DIR}/${wrapper}.invalid.out" \
    'requires CONTACT_AWARE_BUTTON_WINDOW_MODE=kinematic_lift' \
    env PATH="${TMP_DIR}/fake-bin:/usr/bin:/bin" BUTTON_MODE_CAPTURE="${capture}" \
      CONTACT_AWARE_BUTTON_WINDOW_MODE=contact_interval /bin/bash "${wrapper}"
  [[ ! -e "${capture}" ]] || fail "invalid ${wrapper} mode reached its parent"

  expect_failure \
    "${TMP_DIR}/${wrapper}.student-history.out" \
    'requires STUDENT_PROPRIO_HISTORY_LENGTH=1' \
    env PATH="${TMP_DIR}/fake-bin:/usr/bin:/bin" BUTTON_MODE_CAPTURE="${capture}" \
      STUDENT_PROPRIO_HISTORY_LENGTH=5 /bin/bash "${wrapper}"
  expect_failure \
    "${TMP_DIR}/${wrapper}.contact-history.out" \
    'requires CONTACT_AWARE_HISTORY=0' \
    env PATH="${TMP_DIR}/fake-bin:/usr/bin:/bin" BUTTON_MODE_CAPTURE="${capture}" \
      CONTACT_AWARE_HISTORY=true /bin/bash "${wrapper}"
  expect_failure \
    "${TMP_DIR}/${wrapper}.actor-inputs.out" \
    'requires the exact ordered 95D STUDENT_ACTOR_INPUTS contract' \
    env PATH="${TMP_DIR}/fake-bin:/usr/bin:/bin" BUTTON_MODE_CAPTURE="${capture}" \
      STUDENT_ACTOR_INPUTS="['actor_obs_root_contact_aware']" /bin/bash "${wrapper}"
  for history_group in "${DUAL_HISTORY_GROUPS[@]}"; do
    expect_failure \
      "${TMP_DIR}/${wrapper}.${history_group}.out" \
      'history is launcher-owned and fixed at 1' \
      env PATH="${TMP_DIR}/fake-bin:/usr/bin:/bin" BUTTON_MODE_CAPTURE="${capture}" \
        /bin/bash "${wrapper}" \
          "--observation.groups.${history_group}.history-length=5"
  done
  expect_failure \
    "${TMP_DIR}/${wrapper}.split-history.out" \
    'history is launcher-owned and fixed at 1' \
    env PATH="${TMP_DIR}/fake-bin:/usr/bin:/bin" BUTTON_MODE_CAPTURE="${capture}" \
      /bin/bash "${wrapper}" \
        --observation.groups.actor_obs_pickup_button.history_length 5
done

cat >"${TMP_DIR}/must-not-run" <<'EOF'
#!/bin/sh
exit 99
EOF
chmod 700 "${TMP_DIR}/must-not-run"

expect_failure \
  "${TMP_DIR}/invalid-env.out" \
  'CONTACT_AWARE_BUTTON_WINDOW_MODE must be exactly contact_interval or kinematic_lift' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" CONTACT_AWARE_BUTTON_WINDOW_MODE=KINEMATIC_LIFT \
    bash distill_box_perception.sh

expect_failure \
  "${TMP_DIR}/forwarded.out" \
  'is launcher-owned and cannot be overridden' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" \
    bash distill_box_perception.sh \
      --command.setup-terms.motion-command.params.motion-config.contact-aware-button-window-mode=kinematic_lift

[[ "$(grep -F -- \
  '--command.setup-terms.motion-command.params.motion-config.contact-aware-button-window-mode="${CONTACT_AWARE_BUTTON_WINDOW_MODE}"' \
  distill_box_perception.sh | wc -l)" == 1 ]] ||
  fail 'distill_box_perception.sh must emit exactly one button-window CLI value'
grep -F 'CONTACT_AWARE_BUTTON_WINDOW_MODE=${CONTACT_AWARE_BUTTON_WINDOW_MODE:-contact_interval}' \
  distill_box_perception.sh >/dev/null ||
  fail 'distill_box_perception.sh lost the legacy contact_interval default'
grep -F 'unset HOLOSOMA_DUAL_BUTTON_HISTORY_CLI_OWNED' \
  distill_box_perception.sh >/dev/null ||
  fail 'distill_box_perception.sh must not leak its launcher-internal ownership marker to training'

echo '[PASS] button-window launcher contract'
