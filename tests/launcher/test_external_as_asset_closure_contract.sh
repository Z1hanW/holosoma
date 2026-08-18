#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "${REPO_ROOT}"
LEGACY_GENERATOR_SHA=$(printf 'a%.0s' {1..64})

TMP_DIR=$(mktemp -d)
LOCAL_GATE_ROOT=""
cleanup() {
  chmod -R u+w "${TMP_DIR}" 2>/dev/null || true
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

bash -n batch_ne.sh distill_as_button_solid.sh distill_as_perception.sh

grep -F 'Effective solid-AS source changed after the all-node barrier' \
  distill_as_button_solid.sh >/dev/null ||
  fail 'solid wrapper no longer binds its real materialization to the sealed source digest'
grep -F 'Effective AS single-slot directory changed after the all-node barrier' \
  distill_as_perception.sh >/dev/null ||
  fail 'perception wrapper no longer binds its real single-slot materialization to the sealed directory'
grep -F 'Effective AS rank-shard source changed after the all-node barrier' \
  distill_as_perception.sh >/dev/null ||
  fail 'perception wrapper no longer binds its real rank materialization to the sealed digest'
grep -F 'normalized_single_slot_dir=$(realpath -m -- "${AS_EXTERNAL_SINGLE_SLOT_DIR}")' \
  batch_ne.sh >/dev/null ||
  fail 'controller no longer requires the resolved external single-slot path to be canonical'
grep -F 'expected_single_slot_suffix="/_single_slot_motion_bank/by-source/${AS_EXTERNAL_SINGLE_SLOT_VIEW_DIGEST}"' \
  batch_ne.sh >/dev/null ||
  fail 'controller no longer binds the external single-slot path suffix to its view digest'
if grep -F '"${REMOTE_REPO_NORMALIZED}/data/"*"/_single_slot_motion_bank/by-source/' \
    batch_ne.sh >/dev/null; then
  fail 'controller still rejects signed repo assets after they resolve to an external immutable volume'
fi

# Keep the signed source-snapshot asset symlink contract intact.  The external
# AS byte closure is additive: it must not weaken the existing exact-link
# target and reachability checks.
grep -F 'done < .holosoma_snapshot/asset_links.tsv' batch_ne.sh >/dev/null ||
  fail 'launch-intent preflight no longer consumes signed asset_links.tsv'
grep -F 'expected_target=$(quote "${REMOTE_REPO}")/"\${asset_path}"' batch_ne.sh >/dev/null ||
  fail 'launch-intent preflight no longer binds asset symlinks to REMOTE_REPO'
grep -F 'test -e "\${snapshot_root}/\${link_path}"' batch_ne.sh >/dev/null ||
  fail 'snapshot installation no longer requires reachable asset symlinks'

# A read-only selected-GPU idle gate must execute before runtime installation
# and the expensive all-node external closure.  The closure must still execute
# before the live W&B verifier and launch-intent preflight/publication.
idle_line=$(grep -nF 'preflight_selected_gpus_idle_parallel' batch_ne.sh | tail -1 | cut -d: -f1)
runtime_line=$(grep -nF 'verify_python_runtimes_before_intent_parallel' batch_ne.sh | tail -1 | cut -d: -f1)
barrier_line=$(grep -nF 'preflight_external_as_asset_closures_parallel' batch_ne.sh | tail -1 | cut -d: -f1)
wandb_line=$(grep -nF 'verify_fresh_wandb_replay_preflight' batch_ne.sh | tail -1 | cut -d: -f1)
intent_line=$(grep -nF 'echo "[INFO] Preflighting launch intent on ${preflight_node}"' batch_ne.sh | cut -d: -f1)
[[ "${idle_line}" =~ ^[1-9][0-9]*$ \
    && "${runtime_line}" =~ ^[1-9][0-9]*$ \
    && "${barrier_line}" =~ ^[1-9][0-9]*$ \
    && "${wandb_line}" =~ ^[1-9][0-9]*$ \
    && "${intent_line}" =~ ^[1-9][0-9]*$ \
    && "${idle_line}" -lt "${runtime_line}" \
    && "${runtime_line}" -lt "${barrier_line}" \
    && "${barrier_line}" -lt "${wandb_line}" \
    && "${wandb_line}" -lt "${intent_line}" ]] ||
  fail 'read-only idle/runtime/external/W&B/intent gates are not safely ordered'

IDLE_PROBE_BODY="${TMP_DIR}/selected_gpu_idle_probe.sh"
awk '
  /^preflight_selected_gpus_idle_node\(\) \{/ { capture = 1 }
  capture && /^preflight_selected_gpus_idle_parallel\(\) \{/ { exit }
  capture { print }
' batch_ne.sh >"${IDLE_PROBE_BODY}"
[[ -s "${IDLE_PROBE_BODY}" ]] || fail 'could not extract selected-GPU idle probe'
grep -F 'nvidia-smi' "${IDLE_PROBE_BODY}" >/dev/null ||
  fail 'early selected-GPU idle probe no longer uses nvidia-smi inventory'
if grep -Eq 'import torch|mkdir|mktemp|tmux|harden_lifecycle|publish_launch|prepare_immutable|install_python_runtime' \
    "${IDLE_PROBE_BODY}"; then
  fail 'early selected-GPU idle probe contains a mutating/heavy operation'
fi
grep -F 'preflight_selected_gpus_idle_node "${node}" pre-launch' batch_ne.sh >/dev/null ||
  fail 'launch_node no longer performs the mandatory second selected-GPU idle check'
IDLE_PARALLEL_BODY="${TMP_DIR}/selected_gpu_idle_parallel.sh"
awk '
  /^preflight_selected_gpus_idle_parallel\(\) \{/ { capture = 1 }
  capture && /^preflight_external_as_asset_closure_node\(\) \{/ { exit }
  capture { print }
' batch_ne.sh >"${IDLE_PARALLEL_BODY}"
if grep -F 'SKIP_NODE_HEALTH_CHECK' "${IDLE_PARALLEL_BODY}" >/dev/null; then
  fail 'selected-GPU idle gate is still bypassable through SKIP_NODE_HEALTH_CHECK'
fi

# Read-only real r29 evidence: the final single-slot map semantics must match
# the Rule-90 v2 manifest fields exactly, not merely a synthetic fixture.
R29_MANIFEST="outputs/formal_single_motion_replay_20260718_r29_corrected/dual_8hdketa1/replay_preflight_manifest.json"
if [[ -f "${R29_MANIFEST}" ]]; then
  "${PYTHON_BIN:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3}" - \
      "${R29_MANIFEST}" "${REPO_ROOT}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
sys.path.insert(0, str(repo_root / "scripts"))
from prepare_as_rank_shards import compute_rank_shard_source_digest

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
inputs = manifest["inputs"]
view = inputs["single_slot_view_digest"]
candidates = list((repo_root / "data" / "ds_as_data").glob(
    f"*/_single_slot_motion_bank/by-source/{view}"
))
if len(candidates) != 1:
    raise SystemExit(f"[FAIL] expected one local r29 single-slot view, found {candidates!r}")
single_dir = candidates[0].resolve()
single_map = single_dir / "_clip_object_urdf_map.json"
single_manifest = json.loads((single_dir / "manifest.json").read_text(encoding="utf-8"))
payload = json.loads(single_map.read_text(encoding="utf-8"))
clip_id = inputs["motion_clip_id"]
entry = payload["clips"][clip_id]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def resolve(raw: str, base: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else base / path


urdf = resolve(entry["object_urdf_path"], single_map.parent)
mesh = resolve(entry["object_mesh_path"], single_map.parent)
actual = {
    "motion_npz_sha256": digest(single_dir / f"{clip_id}.npz"),
    "object_map_sha256": digest(single_map),
    "object_urdf_sha256": digest(urdf),
    "object_mesh_sha256": digest(mesh),
    "single_slot_source_digest": single_manifest["source_digest"],
    "single_slot_view_digest": single_manifest["view_digest"],
    "rank_shard_source_digest": compute_rank_shard_source_digest(
        motion_dir=single_dir,
        object_map=single_map,
        world_size=int(inputs.get("world_size", manifest["inputs"]["world_size"])),
    ),
}
for key, value in actual.items():
    if value != inputs[key]:
        raise SystemExit(
            f"[FAIL] real r29 {key} differs: actual={value} expected={inputs[key]}"
        )
print("[PASS] real r29 Rule-90 v2 AS closure matches motion/map/URDF/mesh and three digests")
PY
fi

SNAPSHOT_CACHE="${TMP_DIR}/snapshot-cache"
snapshot_record=$(bash scripts/build_run_snapshot.sh --cache-root "${SNAPSHOT_CACHE}")
IFS=$'\t' read -r SNAPSHOT_ID SNAPSHOT_ARCHIVE SNAPSHOT_ARCHIVE_SHA SNAPSHOT_MANIFEST_SHA \
  <<<"${snapshot_record}"
[[ "${SNAPSHOT_ID}" =~ ^src-[0-9a-f]{64}$ \
    && "${SNAPSHOT_ARCHIVE_SHA}" =~ ^[0-9a-f]{64}$ \
    && "${SNAPSHOT_MANIFEST_SHA}" =~ ^[0-9a-f]{64}$ ]] ||
  fail 'could not build one authenticated launcher snapshot fixture'

FAKE_BIN="${TMP_DIR}/fake-bin"
mkdir -p "${FAKE_BIN}"
cat >"${FAKE_BIN}/ssh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
node="${@: -2:1}"
cmd="${@: -1}"
if [[ "${cmd}" == *'exact launch intent already published for token='* ]]; then
  : >"${FAKE_INTENT_MARKER}"
fi
if [[ -n "${FAKE_EXTERNAL_MATERIALIZATION_MARKER:-}" \
      && "${cmd}" == *'prepare_immutable_solid_bank.py'* ]]; then
  : >"${FAKE_EXTERNAL_MATERIALIZATION_MARKER}"
fi
if [[ -n "${FAKE_EARLY_BUSY_NODE:-}" \
      && "${FAKE_EARLY_BUSY_NODE}" == "${node}" \
      && "${cmd}" == *'selected_gpu_idle_preflight_ok'* ]]; then
  : >"${FAKE_EARLY_IDLE_PROBE_MARKER}"
  echo "[ERROR][${node}] selected GPU(s) are not idle: fixture-busy-process" >&2
  exit 94
fi
if [[ "${cmd}" == *'selected_gpu_idle_preflight_ok'* ]]; then
  echo "[INFO][${node}] selected_gpu_idle_preflight_ok fixture=True"
  exit 0
fi
if [[ -n "${FAKE_PREPARE_MARKER:-}" ]]; then
  : >"${FAKE_PREPARE_MARKER}"
fi
if [[ "${cmd}" == *'REMOTE_RUN_ROOT did not reach exact owner-controlled mode'* ]]; then
  : >"${FAKE_POST_BARRIER_MARKER}"
  echo "[FAKE] stopped after complete external-AS barrier and before intent" >&2
  exit 91
fi
FAKE_SSH_NODE="${node}" bash -c "${cmd}"
EOF
chmod 0500 "${FAKE_BIN}/ssh"

make_external_repo() {
  local case_name="$1"
  local missing_role="$2"
  local external_repo="${TMP_DIR}/${case_name}/asset-repo"
  local bank="${external_repo}/data/ds_as_data/fixture_bank"
  mkdir -p "${bank}/contact_export_from_teacher_success133_final0p5/clip_ball_1"
  printf 'fixture-motion-bytes\n' >"${bank}/clip_ball_1.npz"
  printf '{"contact":true}\n' \
    >"${bank}/contact_export_from_teacher_success133_final0p5/clip_ball_1/contact.json"
  if [[ "${missing_role}" != map ]]; then
    cat >"${bank}/_clip_object_urdf_map.json" <<'JSON'
{
  "clips": {
    "clip_ball_1": {
      "object_name": "fixture_ball",
      "object_urdf_path": "object.urdf",
      "object_mesh_path": "mesh.stl",
      "object_visual_mesh_paths": ["mesh.stl"],
      "object_collision_mesh_path": "mesh.stl"
    }
  }
}
JSON
  fi
  if [[ "${missing_role}" != urdf ]]; then
    cat >"${bank}/object.urdf" <<'URDF'
<?xml version="1.0"?>
<robot name="fixture_ball">
  <link name="object">
    <visual><geometry><mesh filename="mesh.stl"/></geometry></visual>
    <collision><geometry><mesh filename="mesh.stl"/></geometry></collision>
  </link>
</robot>
URDF
  fi
  if [[ "${missing_role}" != mesh ]]; then
    printf 'solid fixture mesh bytes\n' >"${bank}/mesh.stl"
  fi
  printf '%s\n' "${external_repo}"
}

install_snapshot_fixture() {
  local case_name="$1"
  local external_repo="$2"
  local remote_root="${TMP_DIR}/${case_name}/remote-runs"
  local snapshot_root="${remote_root}/${SNAPSHOT_ID}"
  mkdir -p "${snapshot_root}"
  tar -xzf "${SNAPSHOT_ARCHIVE}" -C "${snapshot_root}" --same-permissions
  chmod -R u+w "${snapshot_root}"
  while IFS=$'\t' read -r link_path asset_path; do
    [[ -n "${link_path}" ]] || continue
    mkdir -p "${external_repo}/${asset_path}"
    mkdir -p "$(dirname "${snapshot_root}/${link_path}")"
    ln -s "${external_repo}/${asset_path}" "${snapshot_root}/${link_path}"
  done <"${snapshot_root}/.holosoma_snapshot/asset_links.tsv"
  printf '%s\n' "${remote_root}"
}

run_case() {
  local case_name="$1"
  local missing_role="$2"
  local external_repo remote_root output intent_marker post_marker rc
  external_repo=$(make_external_repo "${case_name}" "${missing_role}")
  remote_root=$(install_snapshot_fixture "${case_name}" "${external_repo}")
  output="${TMP_DIR}/${case_name}.out"
  intent_marker="${TMP_DIR}/${case_name}.intent"
  post_marker="${TMP_DIR}/${case_name}.post-barrier"
  set +e
  env \
    PATH="${FAKE_BIN}:${PATH}" \
    FAKE_INTENT_MARKER="${intent_marker}" \
    FAKE_POST_BARRIER_MARKER="${post_marker}" \
    HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY=0 \
    NODES=fixture-node NNODES=1 NPROC=1 CUDA_VISIBLE_DEVICES=0 \
    PER_GPU_ENVS=1024 DRY_RUN=0 SSH_OPTS= SKIP_GIT_PULL=1 \
    SKIP_NODE_HEALTH_CHECK=1 \
    REMOTE_REPO="${external_repo}" \
    REMOTE_RUN_ROOT="${remote_root}" \
    LOGGER_BASE_DIR="${remote_root}/training_logs" \
    SOURCE_SNAPSHOT_CACHE="${SNAPSHOT_CACHE}" \
    SOURCE_SNAPSHOT_ID="${SNAPSHOT_ID}" \
    SOURCE_SNAPSHOT_ARCHIVE="${SNAPSHOT_ARCHIVE}" \
    SOURCE_SNAPSHOT_ARCHIVE_SHA256="${SNAPSHOT_ARCHIVE_SHA}" \
    SOURCE_MANIFEST_SHA256="${SNAPSHOT_MANIFEST_SHA}" \
    CORL_SOLID80_BANK_NAME=fixture_bank \
    SOLID_ALLOWED_OBJECT_CATEGORIES='["ball"]' \
    SOLID_TARGET_BANK_NAME=fixture_selected \
    MOTION_GENERATOR_TEACHER_EXPECTED_SHA256="${LEGACY_GENERATOR_SHA}" \
    OMOMO_EXPECTED_TOTAL=1 RESUME_FROM_BOX_EXPECTED_TOTAL=1 \
    SESSION="fixture-${case_name}" RUN_STAMP="fixture-${case_name}" \
    bash batch_ne.sh launch >"${output}" 2>&1
  rc=$?
  set -e
  (( rc != 0 )) || fail "${case_name} unexpectedly reached a successful launch"
  [[ ! -e "${intent_marker}" ]] ||
    fail "${case_name} reached launch-intent publication"

  if [[ -n "${missing_role}" ]]; then
    [[ ! -e "${post_marker}" ]] ||
      fail "${case_name} passed the external barrier despite missing ${missing_role}"
    grep -F 'External AS asset closure failed before W&B verification and launch intent.' \
      "${output}" >/dev/null || {
        sed -n '1,120p' "${output}" >&2
        fail "${case_name} did not fail at the external closure boundary"
      }
  else
    [[ -e "${post_marker}" ]] || {
      sed -n '1,160p' "${output}" >&2
      fail 'complete external closure did not pass the SSH-stub barrier'
    }
    grep -F 'external_as_asset_closure_verified' "${output}" >/dev/null ||
      fail 'complete external closure emitted no exact identity'
    grep -F 'All nodes passed one identical external AS asset closure before W&B verification' \
      "${output}" >/dev/null ||
      fail 'complete external closure did not reach the all-node equality barrier'
  fi
  printf '%s\t%s\t%s\n' "${external_repo}" "${remote_root}" "${output}"
}

run_case missing-map map >/dev/null
run_case missing-urdf urdf >/dev/null
run_case missing-mesh mesh >/dev/null

# A busy selected GPU must fail before the first external bank materialization,
# before W&B verification, and before any lifecycle intent.  The fake SSH
# rejects the exact early probe command; no remote command is allowed to reach
# the later materializer marker.
BUSY_REPO=$(make_external_repo busy-node '')
BUSY_REMOTE_ROOT=$(install_snapshot_fixture busy-node "${BUSY_REPO}")
for BUSY_ACTION in launch all; do
  BUSY_OUTPUT="${TMP_DIR}/busy-node-${BUSY_ACTION}.out"
  BUSY_INTENT_MARKER="${TMP_DIR}/busy-node-${BUSY_ACTION}.intent"
  BUSY_POST_MARKER="${TMP_DIR}/busy-node-${BUSY_ACTION}.post-barrier"
  BUSY_MATERIALIZATION_MARKER="${TMP_DIR}/busy-node-${BUSY_ACTION}.materialization"
  BUSY_PREPARE_MARKER="${TMP_DIR}/busy-node-${BUSY_ACTION}.prepare"
  BUSY_IDLE_MARKER="${TMP_DIR}/busy-node-${BUSY_ACTION}.idle-probe"
  set +e
  env \
    PATH="${FAKE_BIN}:${PATH}" \
    FAKE_INTENT_MARKER="${BUSY_INTENT_MARKER}" \
    FAKE_POST_BARRIER_MARKER="${BUSY_POST_MARKER}" \
    FAKE_EXTERNAL_MATERIALIZATION_MARKER="${BUSY_MATERIALIZATION_MARKER}" \
    FAKE_PREPARE_MARKER="${BUSY_PREPARE_MARKER}" \
    FAKE_EARLY_BUSY_NODE=fixture-node \
    FAKE_EARLY_IDLE_PROBE_MARKER="${BUSY_IDLE_MARKER}" \
    HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY=0 \
    NODES=fixture-node NNODES=1 NPROC=1 CUDA_VISIBLE_DEVICES=0 \
    PER_GPU_ENVS=1024 DRY_RUN=0 SSH_OPTS= SKIP_GIT_PULL=1 \
    SKIP_NODE_HEALTH_CHECK=1 \
    REMOTE_REPO="${BUSY_REPO}" \
    REMOTE_RUN_ROOT="${BUSY_REMOTE_ROOT}" \
    LOGGER_BASE_DIR="${BUSY_REMOTE_ROOT}/training_logs" \
    SOURCE_SNAPSHOT_CACHE="${SNAPSHOT_CACHE}" \
    SOURCE_SNAPSHOT_ID="${SNAPSHOT_ID}" \
    SOURCE_SNAPSHOT_ARCHIVE="${SNAPSHOT_ARCHIVE}" \
    SOURCE_SNAPSHOT_ARCHIVE_SHA256="${SNAPSHOT_ARCHIVE_SHA}" \
    SOURCE_MANIFEST_SHA256="${SNAPSHOT_MANIFEST_SHA}" \
    CORL_SOLID80_BANK_NAME=fixture_bank \
    SOLID_ALLOWED_OBJECT_CATEGORIES='["ball"]' \
    SOLID_TARGET_BANK_NAME=fixture_selected \
    OMOMO_EXPECTED_TOTAL=1 RESUME_FROM_BOX_EXPECTED_TOTAL=1 \
    SESSION="fixture-busy-node-${BUSY_ACTION}" RUN_STAMP="fixture-busy-node-${BUSY_ACTION}" \
    bash batch_ne.sh "${BUSY_ACTION}" >"${BUSY_OUTPUT}" 2>&1
  BUSY_RC=$?
  set -e
  (( BUSY_RC != 0 )) || fail "busy selected GPU unexpectedly reached ${BUSY_ACTION}"
  [[ -e "${BUSY_IDLE_MARKER}" ]] ||
    fail "busy-node ${BUSY_ACTION} did not execute the mandatory idle probe"
  [[ ! -e "${BUSY_PREPARE_MARKER}" ]] ||
    fail "busy-node ${BUSY_ACTION} reached remote prepare work"
  [[ ! -e "${BUSY_MATERIALIZATION_MARKER}" ]] ||
    fail "busy-node ${BUSY_ACTION} reached external-AS materialization"
  [[ ! -e "${BUSY_POST_MARKER}" ]] ||
    fail "busy-node ${BUSY_ACTION} passed the external barrier"
  [[ ! -e "${BUSY_INTENT_MARKER}" ]] ||
    fail "busy-node ${BUSY_ACTION} reached launch intent"
  grep -F 'no external-AS materialization, W&B verification, or lifecycle mutation was reached' \
    "${BUSY_OUTPUT}" >/dev/null || {
      sed -n '1,140p' "${BUSY_OUTPUT}" >&2
      fail "busy-node ${BUSY_ACTION} did not fail at the mandatory read-only gate"
    }
done

IFS=$'\t' read -r COMPLETE_REPO COMPLETE_REMOTE_ROOT COMPLETE_OUTPUT \
  < <(run_case complete '')

# Extract and execute the exact Python revalidator embedded in the sealed tmux
# control.  Mutating an asset after the all-node barrier must be rejected by
# this second gate before the entrypoint/torchrun boundary.
REVALIDATOR="${TMP_DIR}/external_as_revalidate.py"
awk '
  /# Close the barrier-to-entrypoint mutation window/ { found = 1; next }
  found && /^"\\\$\{PYTHON_BIN\}" - <<'\''PY'\''$/ { capture = 1; next }
  capture && /^PY$/ { exit }
  capture { print }
' batch_ne.sh >"${REVALIDATOR}"
[[ -s "${REVALIDATOR}" ]] || fail 'could not extract sealed pre-entrypoint AS revalidator'

SINGLE_DIR=$(find "${COMPLETE_REPO}/data/ds_as_data" \
  -path '*/_single_slot_motion_bank/by-source/*' -mindepth 4 -maxdepth 6 \
  -type d | head -1)
[[ -n "${SINGLE_DIR}" && -f "${SINGLE_DIR}/manifest.json" ]] ||
  fail 'complete fixture has no generated single-slot view'
readarray -t CLOSURE_ENV < <(
  "${PYTHON_BIN:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3}" - \
      "${SINGLE_DIR}" <<'PY'
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

single = Path(sys.argv[1]).resolve()
solid = single.parents[2]
obj_map = single / "_clip_object_urdf_map.json"
manifest = json.loads((single / "manifest.json").read_text())
solid_manifest = json.loads((solid / "manifest.json").read_text())
payload = json.loads(obj_map.read_text())
clip = sorted(payload["clips"])[0]
entry = payload["clips"][clip]
def resolve(raw, base):
    path = Path(raw)
    return path if path.is_absolute() else base / path
def digest(path):
    return hashlib.sha256(path.resolve(strict=True).read_bytes()).hexdigest()
rank_root = single / "_rank_shards" / "by-source"
rank_digests = [path.name for path in rank_root.iterdir() if path.is_dir()]
if len(rank_digests) != 1:
    raise SystemExit(f"rank digest fixture is not unique: {rank_digests}")
for value in (
    clip,
    solid_manifest["source_digest"],
    solid_manifest["selected_clip_count"],
    manifest["source_digest"],
    manifest["view_digest"],
    rank_digests[0],
    digest(single / f"{clip}.npz"),
    digest(obj_map),
    digest(resolve(entry["object_urdf_path"], obj_map.parent)),
    digest(resolve(entry["object_mesh_path"], obj_map.parent)),
):
    print(value)
PY
)
(( ${#CLOSURE_ENV[@]} == 10 )) || fail 'could not derive complete fixture closure identity'

REVALIDATE_ENV=(
  env
  PYTHONPATH="${REPO_ROOT}/scripts"
  HOLOSOMA_EXTERNAL_AS_MOTION_CLIP_ID="${CLOSURE_ENV[0]}"
  HOLOSOMA_EXTERNAL_AS_SOLID_SOURCE_DIGEST="${CLOSURE_ENV[1]}"
  HOLOSOMA_EXTERNAL_AS_SELECTED_CLIP_COUNT="${CLOSURE_ENV[2]}"
  HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST="${CLOSURE_ENV[3]}"
  HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST="${CLOSURE_ENV[4]}"
  HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST="${CLOSURE_ENV[5]}"
  HOLOSOMA_EXTERNAL_AS_MOTION_NPZ_SHA256="${CLOSURE_ENV[6]}"
  HOLOSOMA_EXTERNAL_AS_OBJECT_MAP_SHA256="${CLOSURE_ENV[7]}"
  HOLOSOMA_EXTERNAL_AS_OBJECT_URDF_SHA256="${CLOSURE_ENV[8]}"
  HOLOSOMA_EXTERNAL_AS_OBJECT_MESH_SHA256="${CLOSURE_ENV[9]}"
  HOLOSOMA_EXTERNAL_AS_MOTION_GENERATOR_TEACHER_SHA256="${LEGACY_GENERATOR_SHA}"
  HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_DIR="${SINGLE_DIR}"
  HOLOSOMA_EXTERNAL_AS_WORLD_SIZE=1
  PER_GPU_ENVS=1024
)

# Model the old self-authentication bypass: change a scientific rank-map field
# and update the mutable manifest SHA to match those changed bytes.  A verifier
# that merely trusts manifest.shards[*].object_map_sha256 would accept this;
# the source-derived plan must reject it both before W&B and immediately before
# the real entrypoint.
RANK_ROOT="${SINGLE_DIR}/_rank_shards/by-source/${CLOSURE_ENV[5]}/ws1"
RANK_MAP="${RANK_ROOT}/rank_0/_clip_object_urdf_map.json"
RANK_MANIFEST="${RANK_ROOT}/manifest.json"
[[ -f "${RANK_MAP}" && -f "${RANK_MANIFEST}" ]] ||
  fail 'complete fixture has no published rank-map contract'
cp "${RANK_MAP}" "${TMP_DIR}/rank-map.original"
cp "${RANK_MANIFEST}" "${TMP_DIR}/rank-manifest.original"

tamper_rank_map_and_manifest_coherently() {
  chmod u+w "${RANK_MAP}" "${RANK_MANIFEST}"
  "${PYTHON_BIN:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3}" - \
      "${RANK_MAP}" "${RANK_MANIFEST}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

rank_map = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])


def encoded(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


map_payload = json.loads(rank_map.read_text(encoding="utf-8"))
rank_contract = map_payload["rank_local_shard"]
rank_contract["distributed_loss_weight"] = (
    float(rank_contract["distributed_loss_weight"]) + 0.125
)
map_bytes = encoded(map_payload)
rank_map.write_bytes(map_bytes)

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
rank_zero = [record for record in manifest["shards"] if record["rank"] == 0]
if len(rank_zero) != 1:
    raise SystemExit(f"expected one rank-0 manifest record, got {rank_zero!r}")
rank_zero[0]["object_map_sha256"] = hashlib.sha256(map_bytes).hexdigest()
manifest_path.write_bytes(encoded(manifest))
PY
  chmod 0444 "${RANK_MAP}" "${RANK_MANIFEST}"
}

restore_rank_map_and_manifest() {
  chmod u+w "${RANK_MAP}" "${RANK_MANIFEST}"
  cp "${TMP_DIR}/rank-map.original" "${RANK_MAP}"
  cp "${TMP_DIR}/rank-manifest.original" "${RANK_MANIFEST}"
  chmod 0444 "${RANK_MAP}" "${RANK_MANIFEST}"
}

tamper_rank_map_and_manifest_coherently
COHERENT_PREFLIGHT_OUTPUT="${TMP_DIR}/coherent-rank-preflight.out"
COHERENT_PREFLIGHT_INTENT="${TMP_DIR}/coherent-rank-preflight.intent"
COHERENT_PREFLIGHT_POST="${TMP_DIR}/coherent-rank-preflight.post-barrier"
set +e
env \
  PATH="${FAKE_BIN}:${PATH}" \
  FAKE_INTENT_MARKER="${COHERENT_PREFLIGHT_INTENT}" \
  FAKE_POST_BARRIER_MARKER="${COHERENT_PREFLIGHT_POST}" \
  HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY=0 \
  NODES=fixture-node NNODES=1 NPROC=1 CUDA_VISIBLE_DEVICES=0 \
  PER_GPU_ENVS=1024 DRY_RUN=0 SSH_OPTS= SKIP_GIT_PULL=1 \
  SKIP_NODE_HEALTH_CHECK=1 \
  REMOTE_REPO="${COMPLETE_REPO}" \
  REMOTE_RUN_ROOT="${COMPLETE_REMOTE_ROOT}" \
  LOGGER_BASE_DIR="${COMPLETE_REMOTE_ROOT}/training_logs" \
  SOURCE_SNAPSHOT_CACHE="${SNAPSHOT_CACHE}" \
  SOURCE_SNAPSHOT_ID="${SNAPSHOT_ID}" \
  SOURCE_SNAPSHOT_ARCHIVE="${SNAPSHOT_ARCHIVE}" \
  SOURCE_SNAPSHOT_ARCHIVE_SHA256="${SNAPSHOT_ARCHIVE_SHA}" \
  SOURCE_MANIFEST_SHA256="${SNAPSHOT_MANIFEST_SHA}" \
  CORL_SOLID80_BANK_NAME=fixture_bank \
  SOLID_ALLOWED_OBJECT_CATEGORIES='["ball"]' \
  SOLID_TARGET_BANK_NAME=fixture_selected \
  MOTION_GENERATOR_TEACHER_EXPECTED_SHA256="${LEGACY_GENERATOR_SHA}" \
  OMOMO_EXPECTED_TOTAL=1 RESUME_FROM_BOX_EXPECTED_TOTAL=1 \
  SESSION=fixture-coherent-rank-preflight RUN_STAMP=fixture-coherent-rank-preflight \
  bash batch_ne.sh launch >"${COHERENT_PREFLIGHT_OUTPUT}" 2>&1
COHERENT_PREFLIGHT_RC=$?
set -e
(( COHERENT_PREFLIGHT_RC != 0 )) ||
  fail 'coherently changed rank map+manifest passed the pre-W&B barrier'
[[ ! -e "${COHERENT_PREFLIGHT_POST}" ]] ||
  fail 'coherently changed rank map+manifest passed the external barrier'
[[ ! -e "${COHERENT_PREFLIGHT_INTENT}" ]] ||
  fail 'coherently changed rank map+manifest reached launch intent'
grep -F 'External AS asset closure failed before W&B verification and launch intent.' \
  "${COHERENT_PREFLIGHT_OUTPUT}" >/dev/null || {
    sed -n '1,160p' "${COHERENT_PREFLIGHT_OUTPUT}" >&2
    fail 'coherent rank-map preflight drift failed outside the external closure gate'
  }
restore_rank_map_and_manifest

"${REVALIDATE_ENV[@]}" \
  "${PYTHON_BIN:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3}" \
  "${REVALIDATOR}" >"${TMP_DIR}/revalidate-pass.out"
grep -F 'external_as_asset_closure_reverified_before_entrypoint' \
  "${TMP_DIR}/revalidate-pass.out" >/dev/null ||
  fail 'complete closure did not pass sealed pre-entrypoint revalidation'

tamper_rank_map_and_manifest_coherently
if "${REVALIDATE_ENV[@]}" \
    "${PYTHON_BIN:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3}" \
    "${REVALIDATOR}" >"${TMP_DIR}/revalidate-coherent-rank-drift.out" 2>&1; then
  fail 'post-barrier coherent rank map+manifest drift passed sealed revalidation'
fi
grep -F 'source-derived deterministic plan' \
  "${TMP_DIR}/revalidate-coherent-rank-drift.out" >/dev/null || {
    sed -n '1,100p' "${TMP_DIR}/revalidate-coherent-rank-drift.out" >&2
    fail 'post-barrier coherent rank map+manifest drift failed for an unexpected reason'
  }
restore_rank_map_and_manifest

# The sealed control also reopens the solid manifest and its copied contact
# snapshot.  A post-barrier mutation of that snapshot must fail even though the
# derived motion/map view itself is unchanged.
SOLID_DIR=$(dirname "$(dirname "$(dirname "${SINGLE_DIR}")")")
SOLID_CONTACT_FILE=$(find "${SOLID_DIR}/contact_export_from_teacher_success133_final0p5" \
  -type f | head -1)
[[ -n "${SOLID_CONTACT_FILE}" ]] || fail 'complete fixture has no immutable solid contact snapshot'
cp "${SOLID_CONTACT_FILE}" "${TMP_DIR}/solid-contact.original"
chmod u+w "${SOLID_CONTACT_FILE}"
printf 'post-barrier contact drift\n' >>"${SOLID_CONTACT_FILE}"
if "${REVALIDATE_ENV[@]}" \
    "${PYTHON_BIN:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3}" \
    "${REVALIDATOR}" >"${TMP_DIR}/revalidate-contact-drift.out" 2>&1; then
  fail 'post-barrier solid contact drift passed sealed pre-entrypoint revalidation'
fi
grep -F 'after the all-node barrier' "${TMP_DIR}/revalidate-contact-drift.out" >/dev/null || {
  sed -n '1,80p' "${TMP_DIR}/revalidate-contact-drift.out" >&2
  fail 'post-barrier solid contact drift failed for an unexpected reason'
}
cp "${TMP_DIR}/solid-contact.original" "${SOLID_CONTACT_FILE}"
chmod 0444 "${SOLID_CONTACT_FILE}"

printf 'post-barrier drift\n' >>"${COMPLETE_REPO}/data/ds_as_data/fixture_bank/mesh.stl"
if "${REVALIDATE_ENV[@]}" \
    "${PYTHON_BIN:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3}" \
    "${REVALIDATOR}" >"${TMP_DIR}/revalidate-drift.out" 2>&1; then
  fail 'post-barrier mesh drift passed sealed pre-entrypoint revalidation'
fi
grep -F 'changed after the all-node barrier' "${TMP_DIR}/revalidate-drift.out" >/dev/null || {
  sed -n '1,80p' "${TMP_DIR}/revalidate-drift.out" >&2
  fail 'post-barrier mesh drift failed for an unexpected reason'
}

# Finally execute the real outer wrapper against the source mutated after the
# barrier.  It may publish a new content-addressed solid generation, but the
# sealed digest gate must reject that generation before distill_as_button.sh.
set +e
env \
  PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3}" \
  OMOMO_DATA_DIR="${COMPLETE_REPO}/data/ds_as_data/fixture_bank" \
  OMOMO_OBJECT_MAP="${COMPLETE_REPO}/data/ds_as_data/fixture_bank/_clip_object_urdf_map.json" \
  OMOMO_EXPECTED_TOTAL=1 RESUME_FROM_BOX_EXPECTED_TOTAL=1 \
  SOLID_ALLOWED_OBJECT_CATEGORIES='["ball"]' \
  SOLID_TARGET_BANK_NAME=fixture_selected \
  HOLOSOMA_EXTERNAL_AS_SOLID_SOURCE_DIGEST="${CLOSURE_ENV[1]}" \
  HOLOSOMA_EXTERNAL_AS_SELECTED_CLIP_COUNT="${CLOSURE_ENV[2]}" \
  RESUME_FROM_BOX=0 RESUME_FROM_PREVIOUS=0 \
  bash distill_as_button_solid.sh >"${TMP_DIR}/solid-wrapper-drift.out" 2>&1
solid_wrapper_rc=$?
set -e
(( solid_wrapper_rc != 0 )) || fail 'mutated source bank passed the real solid wrapper digest gate'
grep -F 'Effective solid-AS source changed after the all-node barrier' \
  "${TMP_DIR}/solid-wrapper-drift.out" >/dev/null || {
    sed -n '1,120p' "${TMP_DIR}/solid-wrapper-drift.out" >&2
    fail 'real solid wrapper rejected mutated source for an unexpected reason'
  }

# Exercise the real perception wrapper's post-materialization gate too.  Use a
# private fixture outside repo/data, precompute its exact single view, then
# supply one wrong sealed rank digest.  The wrapper must accept only the exact
# sealed external solid/single paths, pass their source/view checks, and reject
# the rank identity before delegating to the training launcher.
LOCAL_GATE_ROOT="${TMP_DIR}/external-gate/fixture_bank"
mkdir -p "${LOCAL_GATE_ROOT}"
cp -a "${COMPLETE_REPO}/data/ds_as_data/fixture_bank/." "${LOCAL_GATE_ROOT}/"
LOCAL_SINGLE_BASE="${LOCAL_GATE_ROOT}/_single_slot_motion_bank"
LOCAL_SINGLE_DIR=$("${PYTHON_BIN:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3}" \
  scripts/prepare_immutable_single_slot_bank.py \
  --source-motion-dir "${LOCAL_GATE_ROOT}" \
  --source-object-map "${LOCAL_GATE_ROOT}/_clip_object_urdf_map.json" \
  --output-base "${LOCAL_SINGLE_BASE}")
read -r LOCAL_SINGLE_SOURCE LOCAL_SINGLE_VIEW < <(
  "${PYTHON_BIN:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3}" - \
      "${LOCAL_SINGLE_DIR}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["source_digest"], payload["view_digest"])
PY
)
LOCAL_RANK_SOURCE=$("${PYTHON_BIN:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3}" \
  scripts/prepare_as_rank_shards.py \
  --motion-dir "${LOCAL_SINGLE_DIR}" \
  --object-map "${LOCAL_SINGLE_DIR}/_clip_object_urdf_map.json" \
  --world-size 1 --environments-per-rank 1024 --source-digest-only)
[[ "${LOCAL_SINGLE_SOURCE}" =~ ^[0-9a-f]{64}$ \
    && "${LOCAL_SINGLE_VIEW}" =~ ^[0-9a-f]{64}$ \
    && "${LOCAL_RANK_SOURCE}" =~ ^[0-9a-f]{64}$ ]] ||
  fail 'could not prepare exact local single/rank identity for the perception wrapper gate'
LOCAL_TEACHER_CKPT="${TMP_DIR}/teacher.pt"
"${PYTHON_BIN:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3}" - \
    "${LOCAL_TEACHER_CKPT}" <<'PY'
import sys
import torch
torch.save({"model": {}}, sys.argv[1])
PY
WRONG_RANK_SOURCE=$(printf '0%.0s' {1..64})
[[ "${WRONG_RANK_SOURCE}" != "${LOCAL_RANK_SOURCE}" ]] ||
  WRONG_RANK_SOURCE=$(printf '1%.0s' {1..64})
set +e
env \
  PYTHONPATH="${REPO_ROOT}/src/holosoma:${PYTHONPATH:-}" \
  PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3}" \
  TEACHER_CHECKPOINT="${LOCAL_TEACHER_CKPT}" \
  AS_TEACHER_CACHE_ROOT="${TMP_DIR}/teacher-cache" \
  OMOMO_DATA_DIR="${LOCAL_GATE_ROOT}" \
  OMOMO_OBJECT_MAP="${LOCAL_GATE_ROOT}/_clip_object_urdf_map.json" \
  OMOMO_EXPECTED_TOTAL=1 AS_CONTACT_AWARE=0 AS_CONTACT_AWARE_HISTORY=0 \
  AS_SINGLE_SLOT_MOTION_BASE="${LOCAL_SINGLE_BASE}" \
  AS_RANK_LOCAL_SHARDS=1 NPROC=1 NNODES=1 CUDA_VISIBLE_DEVICES=0 PER_GPU_ENVS=1024 \
  HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST="${LOCAL_SINGLE_SOURCE}" \
  HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST="${LOCAL_SINGLE_VIEW}" \
  HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST="${WRONG_RANK_SOURCE}" \
  HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_DIR="${LOCAL_SINGLE_DIR}" \
  HOLOSOMA_EXTERNAL_AS_WORLD_SIZE=1 \
  bash distill_as_perception.sh >"${TMP_DIR}/perception-wrapper-rank-drift.out" 2>&1
perception_wrapper_rc=$?
set -e
(( perception_wrapper_rc != 0 )) || fail 'wrong sealed rank digest passed the real perception wrapper gate'
grep -F 'Effective AS rank-shard source changed after the all-node barrier' \
  "${TMP_DIR}/perception-wrapper-rank-drift.out" >/dev/null || {
    sed -n '1,160p' "${TMP_DIR}/perception-wrapper-rank-drift.out" >&2
    fail 'real perception wrapper rejected the rank contract for an unexpected reason'
  }

echo '[PASS] busy GPUs fail before heavy work; external closure fails before W&B/intent; complete closure passes; solid/contact/source/view/rank drift is rejected before training'
