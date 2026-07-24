#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
  echo "usage: $0 NODE [NODE ...]" >&2
  exit 2
fi

readonly SNAPSHOT_ID=src-a3d2416dc71f953488fa3c447fc110e9880ae1c8536d58ec4951e4f1d0d8738c
readonly MANIFEST_SHA=a3d2416dc71f953488fa3c447fc110e9880ae1c8536d58ec4951e4f1d0d8738c
readonly ARCHIVE_SHA=73b90f6c880ee8c7b91556168eadc13465d7f18b11eda24d570cf96b8c5c381e
readonly ARCHIVE=/home/ubuntu/FAR/holosoma_runs/.snapshots/src-a3d2416dc71f953488fa3c447fc110e9880ae1c8536d58ec4951e4f1d0d8738c.tar.gz
readonly RUNS_ROOT=/home/ubuntu/FAR/holosoma_runs
readonly ASSET_ROOT=/home/ubuntu/FAR/holosoma

[[ $(sha256sum "${ARCHIVE}" | awk '{print $1}') == "${ARCHIVE_SHA}" ]]

for node in "$@"; do
  remote_archive="${RUNS_ROOT}/.incoming/${SNAPSHOT_ID}.${ARCHIVE_SHA}.tar.gz"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${node}" \
    "mkdir -p '${RUNS_ROOT}/.incoming'"
  scp -q -o BatchMode=yes -o ConnectTimeout=10 \
    "${ARCHIVE}" "${node}:${remote_archive}"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${node}" bash -s -- \
    "${SNAPSHOT_ID}" "${MANIFEST_SHA}" "${ARCHIVE_SHA}" \
    "${remote_archive}" "${RUNS_ROOT}" "${ASSET_ROOT}" <<'REMOTE'
set -euo pipefail
snapshot_id=$1
manifest_sha=$2
archive_sha=$3
archive=$4
runs_root=$5
asset_root=$6
destination="${runs_root}/${snapshot_id}"

exec 9>"${runs_root}/.snapshot-install.lock"
flock -w 120 -x 9
[[ $(sha256sum "${archive}" | awk '{print $1}') == "${archive_sha}" ]]

verify_snapshot() {
  local root=$1 link_path link_target asset_path
  [[ -d "${root}" && ! -L "${root}" ]]
  [[ $(<"${root}/.holosoma_snapshot/id") == "${snapshot_id}" ]]
  [[ $(sha256sum "${root}/.holosoma_snapshot/source_manifest.sha256" | awk '{print $1}') == "${manifest_sha}" ]]
  (cd "${root}" && sha256sum --quiet -c .holosoma_snapshot/source_manifest.sha256)
  (cd "${root}" && {
    find . -maxdepth 0 -type d -printf 'd\t%m\t%p\0'
    find . -mindepth 1 -maxdepth 1 -type f -printf 'f\t%m\t%p\0'
    for source_dir in src scripts tests submodules; do
      [[ -d "./${source_dir}" ]] || continue
      find "./${source_dir}" -type f -printf 'f\t%m\t%p\0'
      find "./${source_dir}" -type d -printf 'd\t%m\t%p\0'
    done
    find ./.holosoma_snapshot -type d -printf 'd\t%m\t%p\0'
  } | sort -z | cmp -s - .holosoma_snapshot/source_modes.nul)
  while IFS=$'\t' read -r link_path link_target; do
    [[ -n "${link_path}" ]] || continue
    [[ -L "${root}/${link_path}" ]]
    [[ $(readlink "${root}/${link_path}") == "${link_target}" ]]
  done <"${root}/.holosoma_snapshot/source_symlinks.tsv"
  while IFS=$'\t' read -r link_path asset_path; do
    [[ -n "${link_path}" ]] || continue
    [[ -L "${root}/${link_path}" ]]
    [[ $(readlink "${root}/${link_path}") == "${asset_root}/${asset_path}" ]]
    [[ -e "${root}/${link_path}" ]]
  done <"${root}/.holosoma_snapshot/asset_links.tsv"
}

if [[ -e "${destination}" ]]; then
  verify_snapshot "${destination}"
  rm -f "${archive}"
  echo "[INFO] reused_verified_source_snapshot node=$(hostname -I | awk '{print $1}') root=${destination}"
  exit 0
fi

temporary="${runs_root}/.${snapshot_id}.tmp.$$"
cleanup() {
  chmod -R u+w "${temporary}" 2>/dev/null || true
  rm -rf "${temporary}"
}
trap cleanup EXIT
mkdir -p "${temporary}"
tar -xzf "${archive}" -C "${temporary}" --no-same-owner --same-permissions
[[ $(<"${temporary}/.holosoma_snapshot/id") == "${snapshot_id}" ]]
[[ $(sha256sum "${temporary}/.holosoma_snapshot/source_manifest.sha256" | awk '{print $1}') == "${manifest_sha}" ]]
(cd "${temporary}" && sha256sum --quiet -c .holosoma_snapshot/source_manifest.sha256)

while IFS=$'\t' read -r link_path asset_path; do
  [[ -n "${link_path}" ]] || continue
  target="${asset_root}/${asset_path}"
  [[ -e "${target}" ]]
  parent=$(dirname "${temporary}/${link_path}")
  parent_mode=$(stat -c '%a' "${parent}")
  chmod u+w "${parent}"
  ln -s "${target}" "${temporary}/${link_path}"
  chmod "${parent_mode}" "${parent}"
done <"${temporary}/.holosoma_snapshot/asset_links.tsv"

root_mode=$(stat -c '%a' "${temporary}")
chmod u+w "${temporary}"
mkdir -p \
  "${temporary}/.checkpoint_cache" \
  "${temporary}/.teacher_checkpoints" \
  "${temporary}/.run_control" \
  "${temporary}/logs/batch_ne"
chmod 700 \
  "${temporary}/.checkpoint_cache" \
  "${temporary}/.teacher_checkpoints" \
  "${temporary}/.run_control" \
  "${temporary}/logs" \
  "${temporary}/logs/batch_ne"
chmod "${root_mode}" "${temporary}"

verify_snapshot "${temporary}"
mv "${temporary}" "${destination}"
trap - EXIT
rm -f "${archive}"
verify_snapshot "${destination}"
echo "[INFO] installed_verified_source_snapshot node=$(hostname -I | awk '{print $1}') root=${destination}"
REMOTE
done
