#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "${REPO_ROOT}"

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

bash -n batch_ne.sh

grep -F 'REQUIRE_MOTION_GENERATOR_TEACHER_MATCH=${REQUIRE_MOTION_GENERATOR_TEACHER_MATCH:-1}' \
  batch_ne.sh >/dev/null || fail 'motion-generator/label-teacher equality is no longer default-on'
grep -F 'MOTION_GENERATOR_TEACHER_EXPECTED_SHA256=<exact recovered checkpoint SHA256>' \
  batch_ne.sh >/dev/null || fail 'legacy motion banks no longer fail with an exact-SHA requirement'
grep -F 'Distillation-label teacher does not match the teacher that generated the input motion' \
  batch_ne.sh >/dev/null || fail 'resolved label teacher is no longer compared with the generator identity'
grep -F 'AS_EXTERNAL_MOTION_GENERATOR_TEACHER_SHA256=${closure_fields[10]}' \
  batch_ne.sh >/dev/null || fail 'all-node external-AS closure no longer carries generator SHA'
grep -F 'MOTION_GENERATOR_TEACHER_SHA256_KEY = "motion_generator_teacher_sha256"' \
  src/holosoma/holosoma/utils/training_provenance.py >/dev/null ||
  fail 'checkpoint provenance no longer binds the motion-generator SHA'

resolve_line=$(grep -nF 'TEACHER_CHECKPOINT_ACTUAL_SHA256=\$(sha256sum "\${TEACHER_CHECKPOINT}"' \
  batch_ne.sh | head -1 | cut -d: -f1 || true)
match_line=$(grep -nF 'Distillation-label teacher does not match the teacher that generated the input motion' \
  batch_ne.sh | head -1 | cut -d: -f1 || true)
entrypoint_line=$(grep -nF 'bash "\${DISTILL_AS_ENTRYPOINT_PATH}" "\${TRAIN_EXTRA_ARGS[@]}"' \
  batch_ne.sh | head -1 | cut -d: -f1 || true)
[[ -n "${resolve_line}" && -n "${match_line}" && -n "${entrypoint_line}" ]] ||
  fail 'could not locate teacher binding/entrypoint boundaries'
(( resolve_line < match_line && match_line < entrypoint_line )) ||
  fail 'teacher identity mismatch is not rejected before the real training entrypoint'

echo '[PASS] motion-generator teacher lineage is sealed and checked before training entry'
