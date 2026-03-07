#!/usr/bin/env bash
set -euo pipefail

# Batch-convert all AMASS-style LAFAN npz clips into holosoma robot-only training files.
#
# Pipeline per source directory:
#   1) reorder joints into canonical G1 order via vis_amass.sh (CONVERT_ONLY=True)
#   2) convert proxy qpos/qvel into train-ready mj npz via convert_data_format_mj.py
#
# Default assumptions:
# - Input files are under ./amass (possibly nested subfolders).
# - Each source subfolder contains .npz clips in LAFAN-style layout.
#
# Usage:
#   bash retgt_amass.sh
#   AMASS_ROOT=/abs/path/to/amass bash retgt_amass.sh
#   AMASS_ROOT=/abs/path OUT_ROOT=/abs/out FORCE=1 bash retgt_amass.sh
#
# Optional env vars:
#   AMASS_ROOT          root folder to scan recursively for .npz
#   OUT_ROOT            output root for train-ready files
#   CACHE_ROOT          proxy cache root
#   ORDER_MODE          amass_csv|auto_ref|identity (default: amass_csv)
#   WRIST_POLICY        mapped|zero (default: mapped)
#   ROBOT               g1|t1 (default: g1)
#   OUTPUT_FPS          output fps in filename (default: 50)
#   MAX_CLIPS_PER_DIR   0=all, >0 limits each source folder
#   FORCE               1=rebuild existing outputs
#   PYTHON_BIN          python for vis_amass conversion stage
#   CONVERTER_PYTHON    python for convert_data_format_mj.py
#   SCENE_XML_FILE      scene xml for converter
#   INCLUDE_DIR_REGEX   only process source dirs matching regex
#   EXCLUDE_DIR_REGEX   skip source dirs matching regex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

AMASS_ROOT=${AMASS_ROOT:-"${SCRIPT_DIR}/amass"}
OUT_ROOT=${OUT_ROOT:-"${SCRIPT_DIR}/src/holosoma_retargeting_my/converted_res/robot_only/amass_all_trainready"}
CACHE_ROOT=${CACHE_ROOT:-"${SCRIPT_DIR}/.cache/amass_all_proxy"}
ORDER_MODE=${ORDER_MODE:-"amass_csv"}
WRIST_POLICY=${WRIST_POLICY:-"mapped"}
ROBOT=${ROBOT:-"g1"}
OUTPUT_FPS=${OUTPUT_FPS:-50}
MAX_CLIPS_PER_DIR=${MAX_CLIPS_PER_DIR:-0}
FORCE=${FORCE:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
CONVERTER_PYTHON=${CONVERTER_PYTHON:-"${PYTHON_BIN}"}
SCENE_XML_FILE=${SCENE_XML_FILE:-"${SCRIPT_DIR}/src/holosoma_retargeting_my/models/g1/g1_29dof.xml"}
INCLUDE_DIR_REGEX=${INCLUDE_DIR_REGEX:-""}
EXCLUDE_DIR_REGEX=${EXCLUDE_DIR_REGEX:-"(^$)"}

VIS_SCRIPT="${SCRIPT_DIR}/vis_amass.sh"
CONVERTER="${SCRIPT_DIR}/src/holosoma_retargeting_my/data_conversion/convert_data_format_mj.py"

check_python_modules() {
  local pybin="$1"
  local label="$2"
  shift 2
  local modules=("$@")

  if ! command -v "${pybin}" >/dev/null 2>&1; then
    echo "[ERROR] ${label} executable not found: ${pybin}" >&2
    return 1
  fi

  if ! "${pybin}" - "${modules[@]}" <<'PY'
import importlib
import sys

mods = sys.argv[1:]
missing = []
for name in mods:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing:
    print("MISSING:" + ",".join(missing))
    raise SystemExit(1)
PY
  then
    echo "[ERROR] ${label} missing required modules: ${modules[*]}" >&2
    echo "        set ${label} to a python env with required deps." >&2
    return 1
  fi
}

if [[ ! -d "${AMASS_ROOT}" ]]; then
  echo "[ERROR] AMASS_ROOT not found: ${AMASS_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${VIS_SCRIPT}" ]]; then
  echo "[ERROR] vis_amass.sh not found: ${VIS_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${CONVERTER}" ]]; then
  echo "[ERROR] converter not found: ${CONVERTER}" >&2
  exit 1
fi
if [[ ! -f "${SCENE_XML_FILE}" ]]; then
  echo "[ERROR] SCENE_XML_FILE not found: ${SCENE_XML_FILE}" >&2
  exit 1
fi

check_python_modules "${PYTHON_BIN}" "PYTHON_BIN" numpy
check_python_modules "${CONVERTER_PYTHON}" "CONVERTER_PYTHON" numpy torch mujoco tyro

case "${ORDER_MODE}" in
  amass_csv|auto_ref|identity) ;;
  *)
    echo "[ERROR] ORDER_MODE must be one of: amass_csv auto_ref identity" >&2
    exit 1
    ;;
esac

case "${WRIST_POLICY}" in
  mapped|zero) ;;
  *)
    echo "[ERROR] WRIST_POLICY must be one of: mapped zero" >&2
    exit 1
    ;;
esac

mkdir -p "${OUT_ROOT}" "${CACHE_ROOT}"

mapfile -t SOURCE_DIRS < <(
  find "${AMASS_ROOT}" -type f -name '*.npz' -printf '%h\n' \
    | sort -u
)

if [[ ${#SOURCE_DIRS[@]} -eq 0 ]]; then
  echo "[ERROR] No .npz files found under AMASS_ROOT: ${AMASS_ROOT}" >&2
  exit 1
fi

selected_dirs=()
for d in "${SOURCE_DIRS[@]}"; do
  rel="${d#${AMASS_ROOT}/}"
  if [[ "${d}" == "${AMASS_ROOT}" ]]; then
    rel="."
  fi
  if [[ -n "${INCLUDE_DIR_REGEX}" ]] && ! [[ "${rel}" =~ ${INCLUDE_DIR_REGEX} ]]; then
    continue
  fi
  if [[ -n "${EXCLUDE_DIR_REGEX}" ]] && [[ "${rel}" =~ ${EXCLUDE_DIR_REGEX} ]]; then
    continue
  fi
  selected_dirs+=("${d}")
done

if [[ ${#selected_dirs[@]} -eq 0 ]]; then
  echo "[ERROR] No source dirs matched filters under: ${AMASS_ROOT}" >&2
  exit 1
fi

echo "[INFO] AMASS root       : ${AMASS_ROOT}"
echo "[INFO] Output root      : ${OUT_ROOT}"
echo "[INFO] Proxy cache root : ${CACHE_ROOT}"
echo "[INFO] Source dir count : ${#selected_dirs[@]}"
echo "[INFO] ORDER_MODE       : ${ORDER_MODE}"
echo "[INFO] WRIST_POLICY     : ${WRIST_POLICY}"
echo "[INFO] MAX_CLIPS_PER_DIR: ${MAX_CLIPS_PER_DIR}"

total_dirs=0
ok_dirs=0
failed_dirs=0
total_files=0
converted_files=0
skipped_files=0
failed_files=0

for src_dir in "${selected_dirs[@]}"; do
  total_dirs=$((total_dirs + 1))
  rel="${src_dir#${AMASS_ROOT}/}"
  if [[ "${src_dir}" == "${AMASS_ROOT}" ]]; then
    rel="."
  fi
  safe_rel="${rel//\//__}"
  if [[ "${safe_rel}" == "." ]]; then
    safe_rel="_root"
  fi

  proxy_dir="${CACHE_ROOT}/${safe_rel}"
  out_dir="${OUT_ROOT}/${rel}"
  mkdir -p "${proxy_dir}" "${out_dir}"

  echo
  echo "[INFO] [${total_dirs}/${#selected_dirs[@]}] Source dir: ${src_dir}"
  echo "[INFO] Proxy dir: ${proxy_dir}"
  echo "[INFO] Output dir: ${out_dir}"

  if ! AMASS_SRC_DIR="${src_dir}" \
      CACHE_DIR="${proxy_dir}" \
      ORDER_MODE="${ORDER_MODE}" \
      WRIST_POLICY="${WRIST_POLICY}" \
      MAX_CLIPS="${MAX_CLIPS_PER_DIR}" \
      CONVERT_ONLY=True \
      PYTHON_BIN="${PYTHON_BIN}" \
      bash "${VIS_SCRIPT}"; then
    echo "[WARN] Proxy conversion failed for source dir: ${src_dir}" >&2
    failed_dirs=$((failed_dirs + 1))
    continue
  fi

  mapfile -t proxy_files < <(find "${proxy_dir}" -maxdepth 1 -type f -name '*.npz' | sort)
  if [[ ${#proxy_files[@]} -eq 0 ]]; then
    echo "[WARN] No proxy npz generated for source dir: ${src_dir}" >&2
    failed_dirs=$((failed_dirs + 1))
    continue
  fi

  dir_failed=0
  for f in "${proxy_files[@]}"; do
    total_files=$((total_files + 1))
    stem="$(basename "${f}" .npz)"
    out="${out_dir}/${stem}_mj_fps${OUTPUT_FPS}.npz"

    if [[ "${FORCE}" != "1" && -f "${out}" ]]; then
      skipped_files=$((skipped_files + 1))
      continue
    fi

    if "${CONVERTER_PYTHON}" "${CONVERTER}" \
      --input-file "${f}" \
      --robot "${ROBOT}" \
      --output-fps "${OUTPUT_FPS}" \
      --data-format lafan \
      --object-name ground \
      --scene-xml-file "${SCENE_XML_FILE}" \
      --output-name "${out}" \
      --once \
      --headless; then
      converted_files=$((converted_files + 1))
    else
      echo "[WARN] convert_data_format_mj failed: ${f}" >&2
      failed_files=$((failed_files + 1))
      dir_failed=1
    fi
  done

  if [[ "${dir_failed}" -eq 0 ]]; then
    ok_dirs=$((ok_dirs + 1))
  else
    failed_dirs=$((failed_dirs + 1))
  fi
done

echo
echo "[INFO] ===== Summary ====="
echo "[INFO] dirs total/ok/failed : ${total_dirs}/${ok_dirs}/${failed_dirs}"
echo "[INFO] files total          : ${total_files}"
echo "[INFO] files converted      : ${converted_files}"
echo "[INFO] files skipped        : ${skipped_files}"
echo "[INFO] files failed         : ${failed_files}"
echo "[INFO] output root          : ${OUT_ROOT}"

if [[ "${failed_dirs}" -gt 0 || "${failed_files}" -gt 0 ]]; then
  exit 2
fi
