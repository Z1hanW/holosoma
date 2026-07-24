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
  local output_file="$1"
  local expected="$2"
  shift 2
  if "$@" >"${output_file}" 2>&1; then
    fail "command unexpectedly succeeded: $*"
  fi
  grep -F "${expected}" "${output_file}" >/dev/null || {
    sed -n '1,20p' "${output_file}" >&2
    fail "missing expected failure text: ${expected}"
  }
}

assert_exact_cli_mode() {
  local output_file="$1"
  local expected_mode="$2"
  local expected_arg="--command.setup-terms.motion-command.params.motion-config.contact-aware-carry-window-mode=${expected_mode}"
  grep -F "[INFO] contact_aware_carry_window_mode=${expected_mode}" "${output_file}" >/dev/null ||
    fail "resolved carry-window mode ${expected_mode} is missing from audit output"
  [[ "$(grep -o -- "${expected_arg}" "${output_file}" | wc -l)" -eq 1 ]] ||
    fail "resolved carry-window mode ${expected_mode} must appear exactly once in the final CLI"
}

bash -n distill_box_perception.sh distill_box_button.sh train_object_generalist_ds.sh

# All seven deployment-affecting Tyro fields are launcher-owned.  Every
# launcher must reject an equals-form tail before it can source helpers or
# inspect assets; button deliberately rejects sparse controls it does not
# otherwise expose.
PROTECTED_TAIL_OPTIONS=(
  --training.export-onnx
  --command.setup-terms.motion-command.params.motion-config.contact-aware-carry-window-mode
  --command.setup-terms.motion-command.params.motion-config.contact-aware-peak-height-alpha
  --command.setup-terms.motion-command.params.motion-config.contact-aware-peak-height-smoothing-steps
  --command.setup-terms.motion-command.params.motion-config.contact-aware-sparse-root-command-mode
  --command.setup-terms.motion-command.params.motion-config.contact-aware-sparse-root-segment-steps
  --command.setup-terms.motion-command.params.motion-config.contact-aware-sparse-root-zero-yaw-threshold-deg
)
for launcher in \
  distill_box_perception.sh \
  distill_box_button.sh \
  train_object_generalist_ds.sh; do
  for protected_option in "${PROTECTED_TAIL_OPTIONS[@]}"; do
    output_name="${launcher//[^A-Za-z0-9]/_}_${protected_option##*.}"
    expect_failure \
      "${TMP_DIR}/${output_name}.out" \
      'cannot be overridden' \
      env PYTHON_BIN="${TMP_DIR}/must-not-run" \
      bash "${launcher}" "${protected_option}=adversarial"
  done
done
unset launcher protected_option output_name

# Canonical underscore aliases and split values are equally protected.  A
# repeated/double-valued field must fail on the first occurrence rather than
# depending on Tyro's later-wins behavior.
expect_failure \
  "${TMP_DIR}/perception_split_underscore_t1.out" \
  'cannot be overridden' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" bash distill_box_perception.sh \
  --command.setup_terms.motion_command.params.motion_config.contact_aware_sparse_root_command_mode \
  t1_aligned_segment
expect_failure \
  "${TMP_DIR}/button_split_underscore_t1.out" \
  'cannot be overridden' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" bash distill_box_button.sh \
  --command.setup_terms.motion_command.params.motion_config.contact_aware_sparse_root_command_mode \
  t1_aligned_segment
expect_failure \
  "${TMP_DIR}/generalist_split_underscore_carry.out" \
  'cannot be overridden' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" bash train_object_generalist_ds.sh \
  --command.setup_terms.motion_command.params.motion_config.contact_aware_carry_window_mode \
  rel_z
expect_failure \
  "${TMP_DIR}/perception_tail_export_false.out" \
  'cannot be overridden' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" bash distill_box_perception.sh \
  --training.export_onnx=False
expect_failure \
  "${TMP_DIR}/button_double_export_value.out" \
  'cannot be overridden' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" bash distill_box_button.sh \
  --training.export-onnx True --training.export_onnx=False

# A benign forwarded option must reach the launcher's next own validation and
# must not be mistaken for a protected prefix.
for launcher in \
  distill_box_perception.sh \
  distill_box_button.sh \
  train_object_generalist_ds.sh; do
  expect_failure \
    "${TMP_DIR}/${launcher//[^A-Za-z0-9]/_}_legal_extra.out" \
    'EXPORT_ONNX must be a boolean. Got: invalid' \
    env PYTHON_BIN="${TMP_DIR}/must-not-run" EXPORT_ONNX=invalid \
    bash "${launcher}" --logger.video.enabled=False
done
unset launcher

# Extremely long leading-zero integers must fail locally instead of escaping
# into awk/Python/helpers.  Ordinary leading zeros are canonicalized later in
# the real dry-run assertions below.
printf -v OVERLONG_LEADING_ZERO_INTEGER '%05000d1' 0
expect_failure \
  "${TMP_DIR}/perception_overlong_smoothing.out" \
  'CONTACT_AWARE_PEAK_HEIGHT_SMOOTHING_STEPS must be an integer in [1, 4096]' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" \
  CONTACT_AWARE_PEAK_HEIGHT_SMOOTHING_STEPS="${OVERLONG_LEADING_ZERO_INTEGER}" \
  bash distill_box_perception.sh
expect_failure \
  "${TMP_DIR}/button_overlong_smoothing.out" \
  'CONTACT_AWARE_PEAK_HEIGHT_SMOOTHING_STEPS must be an integer in [1, 4096]' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" \
  CONTACT_AWARE_PEAK_HEIGHT_SMOOTHING_STEPS="${OVERLONG_LEADING_ZERO_INTEGER}" \
  bash distill_box_button.sh
expect_failure \
  "${TMP_DIR}/generalist_overlong_segment.out" \
  'CONTACT_AWARE_SPARSE_ROOT_SEGMENT_STEPS must be an integer in [1, 1000000]' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" EXPORT_ONNX=False \
  CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE=t1_aligned_segment \
  CONTACT_AWARE_SPARSE_ROOT_SEGMENT_STEPS="${OVERLONG_LEADING_ZERO_INTEGER}" \
  bash train_object_generalist_ds.sh
unset OVERLONG_LEADING_ZERO_INTEGER

# These checks deliberately use an unusable interpreter.  The launcher-owned
# contract must fail before helpers, asset preparation, snapshots, SSH, or the
# training process can run.
expect_failure \
  "${TMP_DIR}/invalid_perception_carry_mode.out" \
  'CONTACT_AWARE_CARRY_WINDOW_MODE must be exactly rel_z or peak_height' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" CONTACT_AWARE_CARRY_WINDOW_MODE=relative_z \
  bash distill_box_perception.sh
expect_failure \
  "${TMP_DIR}/invalid_button_carry_mode.out" \
  'CONTACT_AWARE_CARRY_WINDOW_MODE must be exactly rel_z or peak_height' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" CONTACT_AWARE_CARRY_WINDOW_MODE=REL_Z \
  bash distill_box_button.sh
expect_failure \
  "${TMP_DIR}/t1_onnx_default.out" \
  'CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE=t1_aligned_segment is not implemented by inference and cannot be used with EXPORT_ONNX=True' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE=t1_aligned_segment \
  bash distill_box_perception.sh
expect_failure \
  "${TMP_DIR}/t1_onnx_alias.out" \
  'CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE=t1_aligned_segment is not implemented by inference and cannot be used with EXPORT_ONNX=True' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE=segment \
  bash distill_box_perception.sh
expect_failure \
  "${TMP_DIR}/unknown_sparse_mode.out" \
  'CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE must resolve to tracking_error or t1_aligned_segment' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE=target_delta \
  bash distill_box_perception.sh
expect_failure \
  "${TMP_DIR}/invalid_segment_steps.out" \
  'CONTACT_AWARE_SPARSE_ROOT_SEGMENT_STEPS must be an integer in [1, 1000000]' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" EXPORT_ONNX=False \
  CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE=segment CONTACT_AWARE_SPARSE_ROOT_SEGMENT_STEPS=1000001 \
  bash distill_box_perception.sh
expect_failure \
  "${TMP_DIR}/invalid_zero_yaw.out" \
  'CONTACT_AWARE_SPARSE_ROOT_ZERO_YAW_THRESHOLD_DEG must be a finite number in [0, 180]' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" EXPORT_ONNX=False \
  CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE=segment CONTACT_AWARE_SPARSE_ROOT_ZERO_YAW_THRESHOLD_DEG=nan \
  bash distill_box_perception.sh
expect_failure \
  "${TMP_DIR}/invalid_peak_alpha.out" \
  'CONTACT_AWARE_PEAK_HEIGHT_ALPHA must be a finite number in [0, 1]' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" CONTACT_AWARE_PEAK_HEIGHT_ALPHA=1.01 \
  bash distill_box_perception.sh
expect_failure \
  "${TMP_DIR}/invalid_peak_smoothing.out" \
  'CONTACT_AWARE_PEAK_HEIGHT_SMOOTHING_STEPS must be an integer in [1, 4096]' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" CONTACT_AWARE_PEAK_HEIGHT_SMOOTHING_STEPS=0 \
  bash distill_box_button.sh

# The drop-button launcher does not expose or forward the t1-aligned command
# mode, so it has no equivalent ONNX-incompatible path today.
if rg -n 'CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE' distill_box_button.sh >/dev/null; then
  fail 'button launcher unexpectedly exposes sparse-root command mode without an inference compatibility gate'
fi

MOTION_FIXTURE="outputs/motion_bank_success_box_0_92_0p3"
TEACHER_FIXTURE=".teacher_checkpoints/model_17000.pt"
if [[ ! -d "${MOTION_FIXTURE}" || ! -f "${TEACHER_FIXTURE}" ]]; then
  echo "[PASS] early box contact-mode launcher contracts (dry-run fixtures unavailable)"
  exit 0
fi

BASE_ENV=(
  env
  DRY_RUN=1
  USE_LEGACY_DS=0
  TEACHER_CHECKPOINT="${REPO_ROOT}/${TEACHER_FIXTURE}"
  TEACHER_ACTOR_OBS_HISTORY_LENGTH=5
  NPROC=1
  NNODES=1
  PER_GPU_ENVS=2
)

"${BASE_ENV[@]}" ROOT_COMMAND_MODE=contact-aware \
  bash distill_box_perception.sh >"${TMP_DIR}/perception_default.out"
assert_exact_cli_mode "${TMP_DIR}/perception_default.out" peak_height
[[ "$(grep -o -- '--training.export-onnx=True' "${TMP_DIR}/perception_default.out" | wc -l)" -eq 1 ]] ||
  fail 'perception default canonical EXPORT_ONNX=True must reach the final CLI exactly once'
grep -F 'uses peak-height carry-window detection' "${TMP_DIR}/perception_default.out" >/dev/null ||
  fail 'perception default schedule metadata does not describe peak_height semantics'

"${BASE_ENV[@]}" ROOT_COMMAND_MODE=contact-aware CONTACT_AWARE_CARRY_WINDOW_MODE=rel_z \
  bash distill_box_perception.sh >"${TMP_DIR}/perception_rel_z.out"
assert_exact_cli_mode "${TMP_DIR}/perception_rel_z.out" rel_z
grep -F 'uses object-root relative-height carry-window detection' "${TMP_DIR}/perception_rel_z.out" >/dev/null ||
  fail 'perception rel_z schedule metadata still claims peak_height semantics'

"${BASE_ENV[@]}" CONTACT_AWARE_PEAK_HEIGHT_SMOOTHING_STEPS=0005 \
  bash distill_box_button.sh >"${TMP_DIR}/button_default.out"
assert_exact_cli_mode "${TMP_DIR}/button_default.out" peak_height
[[ "$(grep -o -- '--training.export-onnx=True' "${TMP_DIR}/button_default.out" | wc -l)" -eq 1 ]] ||
  fail 'button default canonical EXPORT_ONNX=True must reach the final CLI exactly once'
[[ "$(grep -o -- '--command.setup-terms.motion-command.params.motion-config.contact-aware-peak-height-smoothing-steps=5' \
  "${TMP_DIR}/button_default.out" | wc -l)" -eq 1 ]] ||
  fail 'button smoothing steps must canonicalize leading zeros exactly once'

"${BASE_ENV[@]}" CONTACT_AWARE_CARRY_WINDOW_MODE=rel_z \
  bash distill_box_button.sh >"${TMP_DIR}/button_rel_z.out"
assert_exact_cli_mode "${TMP_DIR}/button_rel_z.out" rel_z
grep -F 'uses object-root relative-height carry-window detection' "${TMP_DIR}/button_rel_z.out" >/dev/null ||
  fail 'button rel_z schedule metadata still claims peak_height semantics'

"${BASE_ENV[@]}" ROOT_COMMAND_MODE=contact-aware EXPORT_ONNX=False \
  CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE=segment \
  CONTACT_AWARE_SPARSE_ROOT_SEGMENT_STEPS=0001000000 \
  CONTACT_AWARE_SPARSE_ROOT_ZERO_YAW_THRESHOLD_DEG=180.0 \
  bash distill_box_perception.sh >"${TMP_DIR}/t1_no_export.out"
grep -F '[INFO] export_onnx=False' "${TMP_DIR}/t1_no_export.out" >/dev/null ||
  fail 'EXPORT_ONNX=False was not preserved in audit output'
grep -F '[INFO] contact_aware_sparse_root_command_mode=t1_aligned_segment' \
  "${TMP_DIR}/t1_no_export.out" >/dev/null ||
  fail 't1-aligned training mode was not preserved with ONNX disabled'
[[ "$(grep -o -- '--training.export-onnx=False' "${TMP_DIR}/t1_no_export.out" | wc -l)" -eq 1 ]] ||
  fail 'EXPORT_ONNX=False must appear exactly once in the final training CLI'
[[ "$(grep -o -- '--command.setup-terms.motion-command.params.motion-config.contact-aware-sparse-root-command-mode=t1_aligned_segment' \
  "${TMP_DIR}/t1_no_export.out" | wc -l)" -eq 1 ]] ||
  fail 'segment alias must canonicalize to t1_aligned_segment exactly once in the final training CLI'
[[ "$(grep -o -- '--command.setup-terms.motion-command.params.motion-config.contact-aware-sparse-root-segment-steps=1000000' \
  "${TMP_DIR}/t1_no_export.out" | wc -l)" -eq 1 ]] ||
  fail 'validated segment-step upper bound must appear exactly once in the final training CLI'
[[ "$(grep -o -- '--command.setup-terms.motion-command.params.motion-config.contact-aware-sparse-root-zero-yaw-threshold-deg=180.0' \
  "${TMP_DIR}/t1_no_export.out" | wc -l)" -eq 1 ]] ||
  fail 'validated zero-yaw upper bound must appear exactly once in the final training CLI'

echo '[PASS] box contact-mode launcher contracts'
