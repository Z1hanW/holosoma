#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/augment_object_interaction_raw_bank.sh [omomo|behave|both]

Purpose:
  Reuse an existing object_interaction raw-retarget bank of *_original.npz files,
  then generate additional augmentation retarget outputs for the same clip subset
  into a separate output folder.

Default inputs:
  OMOMO source data        : /data/OMOMO_new
  OMOMO base raw bank      : src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/omomo_carry
  OMOMO augmented out dir  : src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/omomo_carry_aug_extra

  BEHAVE source data       : /data/behave/annotation_30fps_zup_carry
  BEHAVE object root       : /data/behave/objects
  BEHAVE base raw bank     : src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/behave_zup_sq_carry
  BEHAVE augmented out dir : src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/behave_zup_sq_carry_aug_extra

Important behavior:
  - Existing *_original.npz from the base raw bank are linked into the output dir.
  - Augmentations are generated into the output dir alongside those originals.
  - The original retarget is not recomputed unless the linked original is missing.

Useful env vars:
  PYTHON_BIN=/path/to/python
  CLEAN_OUT=1                      # remove output dir before running
  CLIP_LIMIT=8                     # only process first N clips for smoke tests
  MAX_WORKERS_OMOMO=4
  MAX_WORKERS_BEHAVE=2
  DROP_ORIGINAL_LINKS=1            # remove *_original.npz symlinks after success
  DRY_RUN=1                        # print commands without executing
  EXTRA_AGGRESSIVE=1               # append trans_3 + rot_2 for object_interaction
  AUGMENTATION_NAMES_CSV=trans_3,rot_2
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
RETARGET_ROOT=${RETARGET_ROOT:-"${REPO_ROOT}/src/holosoma_retargeting"}

MODE=${1:-${DATASET_MODE:-both}}
case "${MODE}" in
  omomo|behave|both) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "[ERROR] Unknown mode '${MODE}'. Use: omomo, behave, or both." >&2
    usage >&2
    exit 2
    ;;
esac

default_python="/home/ubuntu/miniconda3/envs/retgt/bin/python"
if [[ -x "${default_python}" ]]; then
  PYTHON_BIN=${PYTHON_BIN:-"${default_python}"}
else
  PYTHON_BIN=${PYTHON_BIN:-$(command -v python3 || command -v python)}
fi

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN:-<empty>}" >&2
  exit 2
fi

OMOMO_SOURCE=${OMOMO_SOURCE:-/data/OMOMO_new}
OMOMO_BASE_RAW_DIR=${OMOMO_BASE_RAW_DIR:-"${REPO_ROOT}/src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/omomo_carry"}
OMOMO_OUT_DIR=${OMOMO_OUT_DIR:-"${REPO_ROOT}/src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/omomo_carry_aug_extra"}
OMOMO_OBJECT_NAME=${OMOMO_OBJECT_NAME:-largebox}

BEHAVE_SOURCE=${BEHAVE_SOURCE:-/data/behave/annotation_30fps_zup_carry}
BEHAVE_OBJECT_ROOT=${BEHAVE_OBJECT_ROOT:-/data/behave/objects}
BEHAVE_BASE_RAW_DIR=${BEHAVE_BASE_RAW_DIR:-"${REPO_ROOT}/src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/behave_zup_sq_carry"}
BEHAVE_OUT_DIR=${BEHAVE_OUT_DIR:-"${REPO_ROOT}/src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/behave_zup_sq_carry_aug_extra"}

MAX_WORKERS_OMOMO=${MAX_WORKERS_OMOMO:-8}
MAX_WORKERS_BEHAVE=${MAX_WORKERS_BEHAVE:-4}
CLIP_LIMIT=${CLIP_LIMIT:-0}
CLEAN_OUT=${CLEAN_OUT:-0}
DROP_ORIGINAL_LINKS=${DROP_ORIGINAL_LINKS:-0}
DRY_RUN=${DRY_RUN:-0}
EXTRA_AGGRESSIVE=${EXTRA_AGGRESSIVE:-0}
AUGMENTATION_NAMES_CSV=${AUGMENTATION_NAMES_CSV:-}
TMP_ROOT=${TMP_ROOT:-/tmp/holosoma_object_aug_sources}
mkdir -p "${TMP_ROOT}"

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN]'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

prepare_out_dir() {
  local out_dir="$1"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ "${CLEAN_OUT}" == "1" && -e "${out_dir}" ]]; then
    rm -rf "${out_dir}"
  fi
  mkdir -p "${out_dir}"
}

select_count() {
  local target_dir="$1"
  local pattern="$2"
  find "${target_dir}" -maxdepth 1 -type f -name "${pattern}" | wc -l | tr -d ' '
}

build_omomo_subset_and_link_originals() {
  local subset_dir="$1"
  local out_dir="$2"
  "${PYTHON_BIN}" - "${OMOMO_BASE_RAW_DIR}" "${OMOMO_SOURCE}" "${subset_dir}" "${out_dir}" "${CLIP_LIMIT}" "${DRY_RUN}" <<'PY'
import os
import sys
from pathlib import Path

base_raw_dir = Path(sys.argv[1]).resolve()
source_dir = Path(sys.argv[2]).resolve()
subset_dir = Path(sys.argv[3]).resolve()
out_dir = Path(sys.argv[4]).resolve()
clip_limit = int(sys.argv[5])
dry_run = sys.argv[6] == "1"

if not base_raw_dir.is_dir():
    raise SystemExit(f"[ERROR] OMOMO_BASE_RAW_DIR not found: {base_raw_dir}")
if not source_dir.is_dir():
    raise SystemExit(f"[ERROR] OMOMO_SOURCE not found: {source_dir}")

subset_dir.mkdir(parents=True, exist_ok=True)
if not dry_run:
    out_dir.mkdir(parents=True, exist_ok=True)

raw_files = sorted(base_raw_dir.glob("*_original.npz"))
if clip_limit > 0:
    raw_files = raw_files[:clip_limit]
if not raw_files:
    raise SystemExit(f"[ERROR] No *_original.npz found in {base_raw_dir}")

linked_subset = 0
linked_originals = 0
missing = []
for raw_file in raw_files:
    clip_name = raw_file.stem.removesuffix("_original")
    src_pt = source_dir / f"{clip_name}.pt"
    if not src_pt.exists():
        missing.append(str(src_pt))
        continue

    subset_link = subset_dir / src_pt.name
    if subset_link.exists() or subset_link.is_symlink():
        subset_link.unlink()
    subset_link.symlink_to(src_pt)
    linked_subset += 1

    if not dry_run:
        out_link = out_dir / raw_file.name
        if out_link.exists() or out_link.is_symlink():
            out_link.unlink()
        out_link.symlink_to(raw_file.resolve())
        linked_originals += 1

if missing:
    sample = "\n".join(missing[:10])
    raise SystemExit(f"[ERROR] Missing OMOMO source .pt files:\n{sample}")

print(f"[INFO] OMOMO selected clips : {len(raw_files)}")
print(f"[INFO] OMOMO subset links   : {linked_subset}")
print(f"[INFO] OMOMO original links : {linked_originals}{' (dry-run skipped link creation)' if dry_run else ''}")
PY
}

build_behave_subset_and_link_originals() {
  local subset_dir="$1"
  local out_dir="$2"
  "${PYTHON_BIN}" - "${BEHAVE_BASE_RAW_DIR}" "${BEHAVE_SOURCE}" "${subset_dir}" "${out_dir}" "${CLIP_LIMIT}" "${DRY_RUN}" <<'PY'
import sys
from pathlib import Path

base_raw_dir = Path(sys.argv[1]).resolve()
source_dir = Path(sys.argv[2]).resolve()
subset_dir = Path(sys.argv[3]).resolve()
out_dir = Path(sys.argv[4]).resolve()
clip_limit = int(sys.argv[5])
dry_run = sys.argv[6] == "1"

if not base_raw_dir.is_dir():
    raise SystemExit(f"[ERROR] BEHAVE_BASE_RAW_DIR not found: {base_raw_dir}")
if not source_dir.is_dir():
    raise SystemExit(f"[ERROR] BEHAVE_SOURCE not found: {source_dir}")

subset_dir.mkdir(parents=True, exist_ok=True)
if not dry_run:
    out_dir.mkdir(parents=True, exist_ok=True)

raw_files = sorted(base_raw_dir.glob("*_original.npz"))
if clip_limit > 0:
    raw_files = raw_files[:clip_limit]
if not raw_files:
    raise SystemExit(f"[ERROR] No *_original.npz found in {base_raw_dir}")

linked_subset = 0
linked_originals = 0
missing = []
for raw_file in raw_files:
    clip_name = raw_file.stem.removesuffix("_original")
    seq_dir = source_dir / clip_name
    if not seq_dir.is_dir():
        missing.append(str(seq_dir))
        continue

    subset_link = subset_dir / seq_dir.name
    if subset_link.exists() or subset_link.is_symlink():
        subset_link.unlink()
    subset_link.symlink_to(seq_dir, target_is_directory=True)
    linked_subset += 1

    if not dry_run:
        out_link = out_dir / raw_file.name
        if out_link.exists() or out_link.is_symlink():
            out_link.unlink()
        out_link.symlink_to(raw_file.resolve())
        linked_originals += 1

if missing:
    sample = "\n".join(missing[:10])
    raise SystemExit(f"[ERROR] Missing BEHAVE source sequence dirs:\n{sample}")

print(f"[INFO] BEHAVE selected clips : {len(raw_files)}")
print(f"[INFO] BEHAVE subset links   : {linked_subset}")
print(f"[INFO] BEHAVE original links : {linked_originals}{' (dry-run skipped link creation)' if dry_run else ''}")
PY
}

drop_original_links() {
  local out_dir="$1"
  if [[ "${DROP_ORIGINAL_LINKS}" != "1" ]]; then
    return 0
  fi
  find "${out_dir}" -maxdepth 1 -type l -name '*_original.npz' -delete
}

run_omomo() {
  echo "[INFO] === OMOMO object_interaction augmentation ==="
  prepare_out_dir "${OMOMO_OUT_DIR}"
  local subset_dir="${TMP_ROOT}/omomo_subset_$$"
  rm -rf "${subset_dir}"
  mkdir -p "${subset_dir}"
  build_omomo_subset_and_link_originals "${subset_dir}" "${OMOMO_OUT_DIR}"

  (
    cd "${RETARGET_ROOT}" || exit 1
    cmd=(
      "${PYTHON_BIN}" examples/parallel_robot_retarget.py \
      --task-type object_interaction \
      --robot g1 \
      --data-format smplh \
      --data-dir "${subset_dir}" \
      --save-dir "${OMOMO_OUT_DIR}" \
      --max-workers "${MAX_WORKERS_OMOMO}" \
      --task-config.object-name "${OMOMO_OBJECT_NAME}" \
      --augmentation
    )
    if [[ "${EXTRA_AGGRESSIVE}" == "1" ]]; then
      cmd+=(--object-interaction-extra-aggressive)
    fi
    if [[ -n "${AUGMENTATION_NAMES_CSV}" ]]; then
      cmd+=(--augmentation-names-csv "${AUGMENTATION_NAMES_CSV}")
    fi
    run_cmd "${cmd[@]}"
  )

  if [[ "${DRY_RUN}" != "1" ]]; then
    drop_original_links "${OMOMO_OUT_DIR}"
    echo "[INFO] OMOMO originals : $(select_count "${OMOMO_OUT_DIR}" '*_original.npz')"
    echo "[INFO] OMOMO trans_*  : $(select_count "${OMOMO_OUT_DIR}" '*_trans_*.npz')"
    echo "[INFO] OMOMO rot_*    : $(select_count "${OMOMO_OUT_DIR}" '*_rot_*.npz')"
    echo "[INFO] OMOMO out dir  : ${OMOMO_OUT_DIR}"
  fi
}

run_behave() {
  echo "[INFO] === BEHAVE object_interaction augmentation ==="
  prepare_out_dir "${BEHAVE_OUT_DIR}"
  local subset_dir="${TMP_ROOT}/behave_subset_$$"
  rm -rf "${subset_dir}"
  mkdir -p "${subset_dir}"
  build_behave_subset_and_link_originals "${subset_dir}" "${BEHAVE_OUT_DIR}"

  (
    cd "${RETARGET_ROOT}" || exit 1
    cmd=(
      "${PYTHON_BIN}" examples/parallel_robot_retarget.py \
      --task-type object_interaction \
      --robot g1 \
      --data-format behave_zup \
      --data-dir "${subset_dir}" \
      --save-dir "${BEHAVE_OUT_DIR}" \
      --max-workers "${MAX_WORKERS_BEHAVE}" \
      --task-config.object-mesh-root "${BEHAVE_OBJECT_ROOT}" \
      --task-config.object-mesh-suffix "_f1000.ply" \
      --augmentation
    )
    if [[ "${EXTRA_AGGRESSIVE}" == "1" ]]; then
      cmd+=(--object-interaction-extra-aggressive)
    fi
    if [[ -n "${AUGMENTATION_NAMES_CSV}" ]]; then
      cmd+=(--augmentation-names-csv "${AUGMENTATION_NAMES_CSV}")
    fi
    run_cmd "${cmd[@]}"
  )

  if [[ "${DRY_RUN}" != "1" ]]; then
    drop_original_links "${BEHAVE_OUT_DIR}"
    echo "[INFO] BEHAVE originals : $(select_count "${BEHAVE_OUT_DIR}" '*_original.npz')"
    echo "[INFO] BEHAVE trans_*  : $(select_count "${BEHAVE_OUT_DIR}" '*_trans_*.npz')"
    echo "[INFO] BEHAVE rot_*    : $(select_count "${BEHAVE_OUT_DIR}" '*_rot_*.npz')"
    echo "[INFO] BEHAVE out dir  : ${BEHAVE_OUT_DIR}"
  fi
}

case "${MODE}" in
  omomo)
    run_omomo
    ;;
  behave)
    run_behave
    ;;
  both)
    run_omomo
    run_behave
    ;;
esac
