#!/usr/bin/env bash
set -euo pipefail

# Install the exact CORL79 + debug30 turn-then-forward bank used by the
# 2026-08-04 formal runs. Existing targets are verified and never replaced.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PYTHON_BIN=${PYTHON_BIN:-python3}
LOCAL_DATA_ROOT=$(realpath -m "${LOCAL_DATA_ROOT:-/data/holosoma_inputs}")
INSTALL_WS32_SHARDS=${INSTALL_WS32_SHARDS:-1}

readonly DATASET=corl79_plus_debug30_decoupled_turn_forward_v1
readonly DIGEST=307e9662d498bd507b9d17ca9abf74a3654f7bf66ac6ab989c6f19c3889bddef
readonly MANIFEST_SHA256=2de9ee5ca188b70e877c32dd9f0d2975eea99d11aa077bb077cf06ea9ab897bb
readonly OBJECT_MAP_SHA256=70b466aad04837a79f6dd0f4491cb345a73c687209981acd3eb7f4a0365d8f5c
readonly ARCHIVE_SHA256=9413b7ea54d40cb57c4f5ffd6d4ed5061187faad1c0b3f17380ef3f000e71be2
readonly ARCHIVE_SIZE_BYTES=903331462
readonly DEFAULT_ARCHIVE=/nfs/zzzihanw/ds_as_data/_distill/corl79_plus_debug30_decoupled_turn_forward_v1/archives/307e9662d498bd507b9d17ca9abf74a3654f7bf66ac6ab989c6f19c3889bddef_ws64_e2048.tar.gz

readonly SHARD_DATASET=corl79_plus_debug30_decoupled_turn_forward_v1_rank_shards_ws32
readonly SHARD_DIGEST=13db668f710806bf4bc6b0541f1c99a3e2b36ad3e5179cccd31d3af8f1ab4928
readonly SHARD_MANIFEST_SHA256=19500fe84e4fef7c70cadc574b2581a41f1e85b5bf5e6cede7fa58a57ab8c858
readonly SHARD_ARCHIVE_SHA256=4d72c044a97dbd45fcbfd903afc69309bef235f9b0720df3b148945f0e9d800c
readonly SHARD_ARCHIVE_SIZE_BYTES=29022
readonly DEFAULT_SHARD_ARCHIVE=/nfs/zzzihanw/ds_as_data/_distill/corl79_plus_debug30_decoupled_turn_forward_v1_rank_shards_ws32/archives/13db668f710806bf4bc6b0541f1c99a3e2b36ad3e5179cccd31d3af8f1ab4928_ws32.tar.gz

ARCHIVE=${ARCHIVE:-${DEFAULT_ARCHIVE}}
SHARD_ARCHIVE=${SHARD_ARCHIVE:-${DEFAULT_SHARD_ARCHIVE}}

readonly REPO_DATA_ROOT=$(realpath -m "${SCRIPT_DIR}/data/ds_as_data")
readonly EXTERNAL_DATA_ROOT=$(realpath -m /data/holosoma_inputs)
if [[ "${LOCAL_DATA_ROOT}" != "${REPO_DATA_ROOT}" \
      && "${LOCAL_DATA_ROOT}" != "${EXTERNAL_DATA_ROOT}" ]]; then
  echo "[ERROR] Refusing unexpected LOCAL_DATA_ROOT: ${LOCAL_DATA_ROOT}" >&2
  exit 2
fi
case "${INSTALL_WS32_SHARDS}" in
  0|1) ;;
  *)
    echo "[ERROR] INSTALL_WS32_SHARDS must be 0 or 1." >&2
    exit 2
    ;;
esac

readonly DEST_PARENT=${LOCAL_DATA_ROOT}/${DATASET}/by-source
readonly DEST=${DEST_PARENT}/${DIGEST}
readonly SHARD_PARENT=${LOCAL_DATA_ROOT}/${SHARD_DATASET}/by-source/${SHARD_DIGEST}
readonly SHARD_DEST=${SHARD_PARENT}/ws32
readonly LOCK_ROOT=$(realpath -m "${HOLOSOMA_DATA_INSTALL_LOCK_ROOT:-/data/holosoma_sync/locks}")

verify_archive() {
  local path=$1 expected_size=$2 expected_sha=$3
  if [[ ! -f "${path}" || -L "${path}" ]]; then
    echo "[ERROR] Missing or symlinked archive: ${path}" >&2
    return 2
  fi
  [[ $(stat -c '%s' "${path}") == "${expected_size}" ]] || {
    echo "[ERROR] Archive size mismatch: ${path}" >&2
    return 2
  }
  [[ $(sha256sum "${path}" | awk '{print $1}') == "${expected_sha}" ]] || {
    echo "[ERROR] Archive SHA256 mismatch: ${path}" >&2
    return 2
  }
}

verify_bank() {
  "${PYTHON_BIN}" - "$1" "${MANIFEST_SHA256}" "${OBJECT_MAP_SHA256}" <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys

root = Path(sys.argv[1])
expected_manifest_sha = sys.argv[2]
expected_map_sha = sys.argv[3]

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

if not root.is_dir() or root.is_symlink():
    raise SystemExit(f"invalid bank root: {root}")
manifest_path = root / "manifest.json"
if digest(manifest_path) != expected_manifest_sha:
    raise SystemExit("manifest SHA256 mismatch")
manifest = json.loads(manifest_path.read_text())
expected_contract = {
    "schema_version": 1,
    "semantics": "precomputed_open_loop_heading_relative_turn_then_forward_actor_input",
    "clip_count": 109,
    "category_counts": {"box": 25, "ball": 9, "barrel": 36, "bin": 39},
    "total_phase_counts": [8888, 24110, 2668],
    "source_payload_digest": "aa4dcb12bc14df37446417d98d7179236960d2c715975d0753438d164ceafa5c",
    "derived_payload_digest": "307e9662d498bd507b9d17ca9abf74a3654f7bf66ac6ab989c6f19c3889bddef",
}
for key, expected in expected_contract.items():
    if manifest.get(key) != expected:
        raise SystemExit(f"manifest contract mismatch at {key}: {manifest.get(key)!r}")
invariants = manifest.get("invariants", {})
for key in (
    "dy_always_zero",
    "dx_nonnegative",
    "dx_and_dyaw_never_overlap",
    "pre_pickup_command_zero",
    "source_arrays_exactly_preserved",
):
    if invariants.get(key) is not True:
        raise SystemExit(f"missing required invariant: {key}")
if invariants.get("fallback_geometry_allowed") is not False:
    raise SystemExit("geometry fallback is not disabled")

records = manifest.get("generated_records")
if not isinstance(records, list) or len(records) != 1994:
    raise SystemExit("unexpected generated-record inventory")
canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
if hashlib.sha256(canonical).hexdigest() != manifest["derived_payload_digest"]:
    raise SystemExit("derived payload digest mismatch")

expected_files = {"manifest.json"}
for record in records:
    relative = PurePosixPath(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(f"unsafe manifest path: {relative}")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"missing, symlinked, or non-file payload: {relative}")
    if path.stat().st_size != int(record["size"]):
        raise SystemExit(f"payload size mismatch: {relative}")
    if digest(path) != record["sha256"]:
        raise SystemExit(f"payload SHA256 mismatch: {relative}")
    expected_files.add(relative.as_posix())

actual_files = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() or path.is_symlink()
}
if actual_files != expected_files:
    missing = sorted(expected_files - actual_files)[:5]
    extra = sorted(actual_files - expected_files)[:5]
    raise SystemExit(f"payload inventory mismatch: missing={missing} extra={extra}")
if digest(root / "_clip_object_urdf_map.json") != expected_map_sha:
    raise SystemExit("object-map SHA256 mismatch")
if any(path.is_symlink() for path in root.rglob("*")):
    raise SystemExit("base bank must not contain symlinks")
print("verified_base_files=1995 clips=109 categories=box:25,ball:9,barrel:36,bin:39")
PY
}

verify_ws32() {
  "${PYTHON_BIN}" - "$1" "${DEST}" "${SHARD_MANIFEST_SHA256}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
base = Path(sys.argv[2]).resolve(strict=True)
expected_manifest_sha = sys.argv[3]

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

if not root.is_dir() or root.is_symlink():
    raise SystemExit(f"invalid ws32 root: {root}")
manifest_path = root / "manifest.json"
if digest(manifest_path) != expected_manifest_sha:
    raise SystemExit("ws32 manifest SHA256 mismatch")
manifest = json.loads(manifest_path.read_text())
contract = {
    "version": 3,
    "world_size": 32,
    "environments_per_rank": 2048,
    "clip_count": 109,
    "source_digest": "13db668f710806bf4bc6b0541f1c99a3e2b36ad3e5179cccd31d3af8f1ab4928",
    "exact_clip_partition": True,
    "duplicated_to_fill_empty_ranks": False,
    "rank_clip_counts_divide_environments_per_rank": True,
}
for key, expected in contract.items():
    if manifest.get(key) != expected:
        raise SystemExit(f"ws32 contract mismatch at {key}: {manifest.get(key)!r}")
shards = manifest.get("shards")
if not isinstance(shards, list) or len(shards) != 32:
    raise SystemExit("ws32 must contain 32 shard records")

all_clips = []
for expected_rank, shard in enumerate(shards):
    if shard.get("rank") != expected_rank:
        raise SystemExit(f"non-canonical shard rank at index {expected_rank}")
    rank_root = root / f"rank_{expected_rank}"
    clip_ids = shard.get("clip_ids")
    if not isinstance(clip_ids, list) or len(clip_ids) != shard.get("clip_count"):
        raise SystemExit(f"invalid clip inventory for rank {expected_rank}")
    if (rank_root / "clip_ids.txt").read_text().splitlines() != clip_ids:
        raise SystemExit(f"clip_ids.txt mismatch for rank {expected_rank}")
    if digest(rank_root / "_clip_object_urdf_map.json") != shard["object_map_sha256"]:
        raise SystemExit(f"object-map mismatch for rank {expected_rank}")
    expected_names = {record["name"] for record in shard["npz_files"]}
    actual_names = {path.name for path in rank_root.glob("*.npz")}
    if actual_names != expected_names:
        raise SystemExit(f"NPZ inventory mismatch for rank {expected_rank}")
    for record in shard["npz_files"]:
        path = rank_root / record["name"]
        if not path.is_symlink():
            raise SystemExit(f"rank-local NPZ is not a symlink: {path}")
        resolved = path.resolve(strict=True)
        if resolved.parent != base or resolved.name != record["name"]:
            raise SystemExit(f"rank-local NPZ escapes exact base bank: {path}")
        if resolved.stat().st_size != int(record["size"]) or digest(resolved) != record["sha256"]:
            raise SystemExit(f"rank-local NPZ content mismatch: {path}")
    all_clips.extend(clip_ids)

if len(all_clips) != 109 or len(set(all_clips)) != 109:
    raise SystemExit("ws32 is not an exact-once 109-clip partition")
if any(path.is_symlink() and not path.exists() for path in root.rglob("*")):
    raise SystemExit("ws32 contains a broken symlink")
print("verified_ws32_ranks=32 clips=109 exact_once=1 double_sharding=0")
PY
}

install_base() {
  if [[ -e "${DEST}" || -L "${DEST}" ]]; then
    verify_bank "${DEST}"
    echo "[INFO] Reused verified immutable base bank: ${DEST}"
    return
  fi
  mkdir -p "${DEST_PARENT}" "${LOCK_ROOT}"
  exec 9>"${LOCK_ROOT}/${DIGEST}.install.lock"
  flock -w 600 -x 9
  if [[ -e "${DEST}" || -L "${DEST}" ]]; then
    verify_bank "${DEST}"
    echo "[INFO] Reused verified immutable base bank: ${DEST}"
    flock -u 9
    return
  fi

  verify_archive "${ARCHIVE}" "${ARCHIVE_SIZE_BYTES}" "${ARCHIVE_SHA256}"
  local available
  available=$(df --output=avail -B1 "${DEST_PARENT}" | tail -n 1 | tr -d ' ')
  if [[ ! "${available}" =~ ^[0-9]+$ || "${available}" -lt 5000000000 ]]; then
    echo "[ERROR] Less than 5 GB available for atomic bank installation." >&2
    return 2
  fi

  local temporary_root extracted
  temporary_root=$(mktemp -d "${DEST_PARENT}/.${DIGEST}.incoming.XXXXXX")
  extracted=${temporary_root}/${DIGEST}
  cleanup_base() {
    chmod -R u+w "${temporary_root}" 2>/dev/null || true
    rm -rf -- "${temporary_root}"
  }
  trap cleanup_base RETURN
  echo "[INFO] Extracting the authenticated 109-clip bank from ${ARCHIVE}"
  tar -xzf "${ARCHIVE}" -C "${temporary_root}" --no-same-owner \
    --strip-components=2 "${DATASET}/by-source/${DIGEST}"
  verify_bank "${extracted}"
  chmod u+w "${extracted}"
  mv "${extracted}" "${DEST}"
  chmod u-w "${DEST}"
  rmdir "${temporary_root}"
  trap - RETURN
  verify_bank "${DEST}"
  echo "[INFO] Installed verified immutable base bank: ${DEST}"
  flock -u 9
}

install_ws32() {
  [[ "${INSTALL_WS32_SHARDS}" == 1 ]] || return
  if [[ -e "${SHARD_DEST}" || -L "${SHARD_DEST}" ]]; then
    verify_ws32 "${SHARD_DEST}"
    echo "[INFO] Reused verified ws32 shard view: ${SHARD_DEST}"
    return
  fi
  mkdir -p "${SHARD_PARENT}" "${LOCK_ROOT}"
  exec 8>"${LOCK_ROOT}/${SHARD_DIGEST}.ws32.install.lock"
  flock -w 600 -x 8
  if [[ -e "${SHARD_DEST}" || -L "${SHARD_DEST}" ]]; then
    verify_ws32 "${SHARD_DEST}"
    echo "[INFO] Reused verified ws32 shard view: ${SHARD_DEST}"
    flock -u 8
    return
  fi

  verify_archive \
    "${SHARD_ARCHIVE}" "${SHARD_ARCHIVE_SIZE_BYTES}" "${SHARD_ARCHIVE_SHA256}"
  local temporary_root extracted
  temporary_root=$(mktemp -d "${SHARD_PARENT}/.ws32.incoming.XXXXXX")
  extracted=${temporary_root}/ws32
  cleanup_shard() {
    chmod -R u+w "${temporary_root}" 2>/dev/null || true
    rm -rf -- "${temporary_root}"
  }
  trap cleanup_shard RETURN
  tar -xzf "${SHARD_ARCHIVE}" -C "${temporary_root}" --no-same-owner \
    --strip-components=3 \
    "${SHARD_DATASET}/by-source/${SHARD_DIGEST}/ws32"
  [[ -d "${extracted}" && ! -L "${extracted}" ]]
  [[ $(sha256sum "${extracted}/manifest.json" | awk '{print $1}') == "${SHARD_MANIFEST_SHA256}" ]]
  chmod u+w "${extracted}"
  mv "${extracted}" "${SHARD_DEST}"
  chmod u-w "${SHARD_DEST}"
  rmdir "${temporary_root}"
  trap - RETURN
  verify_ws32 "${SHARD_DEST}"
  echo "[INFO] Installed verified ws32 shard view: ${SHARD_DEST}"
  flock -u 8
}

install_base
install_ws32
echo "[INFO] CORL79 + debug30 turn-then-forward data installation complete."
