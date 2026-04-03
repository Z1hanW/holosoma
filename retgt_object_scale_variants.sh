#!/usr/bin/env bash
set -euo pipefail

# Only edit these lines. Then run: bash retgt_object_scale_variants.sh
DATASETS=${DATASETS:-omomo,behave}
SCALE_SPECS=${SCALE_SPECS:-"0.8 0.9 1.1 1.2"}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RETARGET_ROOT=${RETARGET_ROOT:-"${SCRIPT_DIR}/src/holosoma_retargeting"}
default_python="/home/ubuntu/miniconda3/envs/retgt/bin/python"
if [[ -x "${default_python}" ]]; then
  PYTHON_BIN=${PYTHON_BIN:-"${default_python}"}
else
  PYTHON_BIN=${PYTHON_BIN:-$(command -v python3 || command -v python)}
fi

OMOMO_SOURCE=${OMOMO_SOURCE:-/data/OMOMO_new}
OMOMO_BASE_CONVERTED=${OMOMO_BASE_CONVERTED:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"}
OMOMO_OBJECT_NAME=${OMOMO_OBJECT_NAME:-largebox}

BEHAVE_SOURCE=${BEHAVE_SOURCE:-/data/behave/annotation_30fps_zup_carry}
BEHAVE_BASE_CONVERTED=${BEHAVE_BASE_CONVERTED:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/behave_sq_carry"}
BEHAVE_OBJECT_ROOT=${BEHAVE_OBJECT_ROOT:-/data/behave/objects}
BEHAVE_OBJECT_FILTER=${BEHAVE_OBJECT_FILTER:-boxmedium,boxlarge}

RAW_OMOMO_ROOT=${RAW_OMOMO_ROOT:-"${RETARGET_ROOT}/demo_results_parallel/g1/object_interaction/omomo_carry_scale_variants"}
RAW_BEHAVE_ROOT=${RAW_BEHAVE_ROOT:-"${RETARGET_ROOT}/demo_results_parallel/g1/object_interaction/behave_zup_sq_carry_scale_variants"}
CONV_OMOMO_ROOT=${CONV_OMOMO_ROOT:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry_scale_variants"}
CONV_BEHAVE_ROOT=${CONV_BEHAVE_ROOT:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/behave_sq_carry_scale_variants"}
FINAL_OMOMO_DIR=${FINAL_OMOMO_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"}
FINAL_BEHAVE_DIR=${FINAL_BEHAVE_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/behave_sq_carry"}

MAX_WORKERS_OMOMO=${MAX_WORKERS_OMOMO:-8}
MAX_WORKERS_BEHAVE=${MAX_WORKERS_BEHAVE:-4}
TMP_ROOT=${TMP_ROOT:-/tmp/holosoma_scale_variant_sources}
LOG_DIR=${LOG_DIR:-/tmp/holosoma_scale_variant_logs}
CLEAN=${CLEAN:-0}
DRY_RUN=${DRY_RUN:-0}

usage() {
  cat <<'EOF'
Usage:
  bash retgt_object_scale_variants.sh

Purpose:
  Re-run object-interaction retargeting for OMOMO and/or BEHAVE with multiple
  object scales, convert the outputs to training npz, and flatten them into a
  directly usable scale-variant bank.

Top-level knobs:
  DATASETS="omomo,behave"
  SCALE_SPECS="0.8 0.9 1.1 1.2"

Scale format:
  - scalar:   0.8        -> uniform xyz scale (0.8, 0.8, 0.8)
  - 3-vector: 0.8x0.8x1.0

Outputs:
  OMOMO nested converted:  src/holosoma_retargeting/converted_res/object_interaction/omomo_carry_scale_variants/<tag>/
  BEHAVE nested converted: src/holosoma_retargeting/converted_res/behave_sq_carry_scale_variants/<tag>/
  OMOMO final bank:        src/holosoma_retargeting/converted_res/object_interaction/omomo_carry/
  BEHAVE final bank:       src/holosoma_retargeting/converted_res/behave_sq_carry/

Useful env vars:
  CLEAN=1
  DRY_RUN=1
  MAX_WORKERS_OMOMO=8
  MAX_WORKERS_BEHAVE=4
EOF
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN]'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] ${label} not found: ${path}" >&2
    exit 1
  fi
}

mkdir_if_needed() {
  local path="$1"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] mkdir -p ${path}"
    return 0
  fi
  mkdir -p "${path}"
}

maybe_clean_dir() {
  local path="$1"
  if [[ "${CLEAN}" != "1" ]]; then
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] rm -rf ${path}"
    return 0
  fi
  rm -rf "${path}"
}

cleanup_existing_scale_files() {
  local target_dir="$1"
  if [[ "${CLEAN}" != "1" ]]; then
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] find ${target_dir} -maxdepth 1 -type f -name '*_scale_*_mj_w_obj.npz' -delete"
    return 0
  fi
  if [[ ! -d "${target_dir}" ]]; then
    return 0
  fi
  find "${target_dir}" -maxdepth 1 -type f -name '*_scale_*_mj_w_obj.npz' -delete
}

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN:-<empty>}" >&2
  exit 1
fi
require_dir "${RETARGET_ROOT}" "RETARGET_ROOT"
mkdir_if_needed "${TMP_ROOT}"
mkdir_if_needed "${LOG_DIR}"

datasets_normalized=$(echo "${DATASETS}" | tr '[:upper:]' '[:lower:]' | tr -d '[]')
IFS=',' read -r -a dataset_tokens <<< "${datasets_normalized}"
RUN_OMOMO=0
RUN_BEHAVE=0
for token in "${dataset_tokens[@]}"; do
  dataset_key=$(echo "${token}" | tr -d '[:space:]')
  case "${dataset_key}" in
    omomo)
      RUN_OMOMO=1
      ;;
    behave)
      RUN_BEHAVE=1
      ;;
    "")
      ;;
    *)
      echo "[ERROR] Unsupported dataset '${dataset_key}' in DATASETS='${DATASETS}'." >&2
      exit 2
      ;;
  esac
done

if [[ "${RUN_OMOMO}" != "1" && "${RUN_BEHAVE}" != "1" ]]; then
  echo "[ERROR] DATASETS='${DATASETS}' selected no datasets." >&2
  exit 2
fi

IFS=' ' read -r -a SCALE_TOKENS <<< "${SCALE_SPECS}"
if [[ ${#SCALE_TOKENS[@]} -eq 0 ]]; then
  echo "[ERROR] SCALE_SPECS produced no scale tokens." >&2
  exit 2
fi

mapfile -t SCALE_ROWS < <("${PYTHON_BIN}" - "${SCALE_TOKENS[@]}" <<'PY'
import re
import sys


def format_scalar(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def parse_spec(spec: str) -> tuple[str, float, float, float]:
    spec = spec.strip()
    if not spec:
        raise ValueError("empty scale spec")
    if any(ch in spec for ch in "xX*"):
        parts = [p for p in re.split(r"[xX*]", spec) if p]
    else:
        parts = [spec]
    values = [float(p) for p in parts]
    if len(values) == 1:
        sx = sy = sz = values[0]
        tag = f"s{int(round(values[0] * 100)):03d}"
        label = format_scalar(values[0])
    elif len(values) == 3:
        sx, sy, sz = values
        tag = f"sx{int(round(sx * 100)):03d}_sy{int(round(sy * 100)):03d}_sz{int(round(sz * 100)):03d}"
        if abs(sx - sy) < 1e-9 and abs(sy - sz) < 1e-9:
            label = format_scalar(sx)
        else:
            label = "x".join(format_scalar(v) for v in (sx, sy, sz))
    else:
        raise ValueError(f"invalid scale spec: {spec}")
    return tag, label, sx, sy, sz


seen: set[str] = set()
for raw in sys.argv[1:]:
    tag, label, sx, sy, sz = parse_spec(raw)
    if tag in seen:
        raise ValueError(f"duplicate scale tag: {tag}")
    seen.add(tag)
    print(f"{tag}\t{label}\t{sx:.8g}\t{sy:.8g}\t{sz:.8g}")
PY
)

if [[ ${#SCALE_ROWS[@]} -eq 0 ]]; then
  echo "[ERROR] Failed to parse SCALE_SPECS='${SCALE_SPECS}'." >&2
  exit 2
fi

build_omomo_subset() {
  local subset_dir="$1"
  "${PYTHON_BIN}" - "${OMOMO_BASE_CONVERTED}" "${OMOMO_SOURCE}" "${subset_dir}" <<'PY'
import re
import sys
from pathlib import Path

base_converted = Path(sys.argv[1]).resolve()
source_dir = Path(sys.argv[2]).resolve()
subset_dir = Path(sys.argv[3]).resolve()
subset_dir.mkdir(parents=True, exist_ok=True)

if not base_converted.is_dir():
    raise SystemExit(f"[ERROR] OMOMO_BASE_CONVERTED not found: {base_converted}")
if not source_dir.is_dir():
    raise SystemExit(f"[ERROR] OMOMO_SOURCE not found: {source_dir}")

pattern = re.compile(r"_(?:rot|trans)_\d+_mj_w_obj$")
selected: list[str] = []
missing: list[str] = []
for npz_path in sorted(base_converted.glob("*_mj_w_obj.npz")):
    stem = npz_path.stem
    if pattern.search(stem):
        continue
    clip_name = stem.removesuffix("_mj_w_obj")
    selected.append(clip_name)
    src_pt = source_dir / f"{clip_name}.pt"
    if not src_pt.exists():
        missing.append(str(src_pt))
        continue
    dst = subset_dir / src_pt.name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src_pt)

if missing:
    sample = "\n".join(missing[:10])
    raise SystemExit(f"[ERROR] Missing OMOMO source clips:\n{sample}")

print(f"[INFO] OMOMO original clips selected: {len(selected)}")
PY
}

build_behave_subset() {
  local subset_dir="$1"
  "${PYTHON_BIN}" - "${BEHAVE_BASE_CONVERTED}" "${BEHAVE_SOURCE}" "${subset_dir}" <<'PY'
import re
import sys
from pathlib import Path

base_converted = Path(sys.argv[1]).resolve()
source_dir = Path(sys.argv[2]).resolve()
subset_dir = Path(sys.argv[3]).resolve()
subset_dir.mkdir(parents=True, exist_ok=True)

if not base_converted.is_dir():
    raise SystemExit(f"[ERROR] BEHAVE_BASE_CONVERTED not found: {base_converted}")
if not source_dir.is_dir():
    raise SystemExit(f"[ERROR] BEHAVE_SOURCE not found: {source_dir}")

pattern = re.compile(r"_(?:rot|trans)_\d+_mj_w_obj$")
selected: list[str] = []
missing: list[str] = []
for npz_path in sorted(base_converted.glob("*_mj_w_obj.npz")):
    stem = npz_path.stem
    if pattern.search(stem):
        continue
    clip_name = stem.removesuffix("_mj_w_obj")
    selected.append(clip_name)
    src_dir = source_dir / clip_name
    if not src_dir.is_dir():
        missing.append(str(src_dir))
        continue
    dst = subset_dir / src_dir.name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src_dir, target_is_directory=True)

if missing:
    sample = "\n".join(missing[:10])
    raise SystemExit(f"[ERROR] Missing BEHAVE source sequence dirs:\n{sample}")

print(f"[INFO] BEHAVE original clips selected: {len(selected)}")
PY
}

run_omomo_tag() {
  local subset_dir="$1"
  local tag="$2"
  local sx="$3"
  local sy="$4"
  local sz="$5"
  local raw_dir="${RAW_OMOMO_ROOT}/${tag}"
  local conv_dir="${CONV_OMOMO_ROOT}/${tag}"
  local log_file="${LOG_DIR}/omomo_${tag}.log"

  maybe_clean_dir "${raw_dir}"
  maybe_clean_dir "${conv_dir}"
  mkdir_if_needed "${raw_dir}"
  mkdir_if_needed "${conv_dir}"

  echo "[INFO] OMOMO scale ${tag}: (${sx}, ${sy}, ${sz})"
  run_cmd bash -lc \
    "set -euo pipefail && \
     cd $(printf '%q' "${RETARGET_ROOT}") && \
     $(printf '%q' "${PYTHON_BIN}") examples/parallel_robot_retarget.py \
       --task-type object_interaction \
       --robot g1 \
       --data-format smplh \
       --data-dir $(printf '%q' "${subset_dir}") \
       --save-dir $(printf '%q' "${raw_dir}") \
       --max-workers $(printf '%q' "${MAX_WORKERS_OMOMO}") \
       --task-config.object-name $(printf '%q' "${OMOMO_OBJECT_NAME}") \
       --task-config.object-interaction-scale-augmented $(printf '%q' "${sx}") $(printf '%q' "${sy}") $(printf '%q' "${sz}") \
       2>&1 | tee $(printf '%q' "${log_file}")"

  run_cmd env \
    INPUT_DIR="${raw_dir}" \
    OUTPUT_DIR="${conv_dir}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${SCRIPT_DIR}/retgt_post_omomo.sh"
}

run_behave_tag() {
  local subset_dir="$1"
  local tag="$2"
  local sx="$3"
  local sy="$4"
  local sz="$5"
  local raw_dir="${RAW_BEHAVE_ROOT}/${tag}"
  local conv_dir="${CONV_BEHAVE_ROOT}/${tag}"
  local log_file="${LOG_DIR}/behave_${tag}.log"

  maybe_clean_dir "${raw_dir}"
  maybe_clean_dir "${conv_dir}"
  mkdir_if_needed "${raw_dir}"
  mkdir_if_needed "${conv_dir}"

  echo "[INFO] BEHAVE scale ${tag}: (${sx}, ${sy}, ${sz})"
  run_cmd bash -lc \
    "set -euo pipefail && \
     cd $(printf '%q' "${RETARGET_ROOT}") && \
     $(printf '%q' "${PYTHON_BIN}") examples/parallel_robot_retarget.py \
       --task-type object_interaction \
       --robot g1 \
       --data-format behave_zup \
       --data-dir $(printf '%q' "${subset_dir}") \
       --save-dir $(printf '%q' "${raw_dir}") \
       --max-workers $(printf '%q' "${MAX_WORKERS_BEHAVE}") \
       --task-config.object-name $(printf '%q' "${BEHAVE_OBJECT_FILTER}") \
       --task-config.object-mesh-root $(printf '%q' "${BEHAVE_OBJECT_ROOT}") \
       --task-config.object-mesh-suffix _f1000.ply \
       --task-config.object-interaction-scale-augmented $(printf '%q' "${sx}") $(printf '%q' "${sy}") $(printf '%q' "${sz}") \
       2>&1 | tee $(printf '%q' "${log_file}")"

  run_cmd env \
    INPUT_DIR="${raw_dir}" \
    OUTPUT_DIR="${conv_dir}" \
    ROBOT="g1" \
    PYTHON_BIN="${PYTHON_BIN}" \
    DATA_FORMAT="behave_zup" \
    bash "${SCRIPT_DIR}/retgt_post_behave.sh"
}

flatten_bank() {
  local input_root="$1"
  local output_dir="$2"
  shift 2
  local tags=()
  local tag_labels=()
  local item tag label
  local cmd=()
  for item in "$@"; do
    tag="${item%%=*}"
    label="${item#*=}"
    tags+=("${tag}")
    tag_labels+=("${tag}=${label}")
  done
  cmd=("${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/flatten_nested_scale_dataset.py" \
    --input-root "${input_root}" \
    --output-dir "${output_dir}" \
    --name-style scale_suffix \
    --expected-tags "${tags[@]}")
  for item in "${tag_labels[@]}"; do
    cmd+=(--tag-label "${item}")
  done
  run_cmd "${cmd[@]}"
}

TAG_LABEL_ROWS=()
for row in "${SCALE_ROWS[@]}"; do
  IFS=$'\t' read -r tag label sx sy sz <<< "${row}"
  TAG_LABEL_ROWS+=("${tag}=${label}")
done

if [[ "${RUN_OMOMO}" == "1" ]]; then
  require_dir "${OMOMO_SOURCE}" "OMOMO_SOURCE"
  require_dir "${OMOMO_BASE_CONVERTED}" "OMOMO_BASE_CONVERTED"
  cleanup_existing_scale_files "${FINAL_OMOMO_DIR}"
  OMOMO_SUBSET_DIR="${TMP_ROOT}/omomo_subset"
  maybe_clean_dir "${OMOMO_SUBSET_DIR}"
  mkdir_if_needed "${OMOMO_SUBSET_DIR}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    build_omomo_subset "${OMOMO_SUBSET_DIR}"
  else
    echo "[DRY_RUN] build OMOMO subset at ${OMOMO_SUBSET_DIR}"
  fi
  for row in "${SCALE_ROWS[@]}"; do
    IFS=$'\t' read -r tag label sx sy sz <<< "${row}"
    run_omomo_tag "${OMOMO_SUBSET_DIR}" "${tag}" "${sx}" "${sy}" "${sz}"
  done
  flatten_bank "${CONV_OMOMO_ROOT}" "${FINAL_OMOMO_DIR}" "${TAG_LABEL_ROWS[@]}"
fi

if [[ "${RUN_BEHAVE}" == "1" ]]; then
  require_dir "${BEHAVE_SOURCE}" "BEHAVE_SOURCE"
  require_dir "${BEHAVE_BASE_CONVERTED}" "BEHAVE_BASE_CONVERTED"
  require_dir "${BEHAVE_OBJECT_ROOT}" "BEHAVE_OBJECT_ROOT"
  cleanup_existing_scale_files "${FINAL_BEHAVE_DIR}"
  BEHAVE_SUBSET_DIR="${TMP_ROOT}/behave_subset"
  maybe_clean_dir "${BEHAVE_SUBSET_DIR}"
  mkdir_if_needed "${BEHAVE_SUBSET_DIR}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    build_behave_subset "${BEHAVE_SUBSET_DIR}"
  else
    echo "[DRY_RUN] build BEHAVE subset at ${BEHAVE_SUBSET_DIR}"
  fi
  for row in "${SCALE_ROWS[@]}"; do
    IFS=$'\t' read -r tag label sx sy sz <<< "${row}"
    run_behave_tag "${BEHAVE_SUBSET_DIR}" "${tag}" "${sx}" "${sy}" "${sz}"
  done
  flatten_bank "${CONV_BEHAVE_ROOT}" "${FINAL_BEHAVE_DIR}" "${TAG_LABEL_ROWS[@]}"
fi

echo "[INFO] Requested scales: ${SCALE_SPECS}"
if [[ "${RUN_OMOMO}" == "1" ]]; then
  echo "[INFO] OMOMO final scale bank : ${FINAL_OMOMO_DIR}"
fi
if [[ "${RUN_BEHAVE}" == "1" ]]; then
  echo "[INFO] BEHAVE final scale bank: ${FINAL_BEHAVE_DIR}"
fi
