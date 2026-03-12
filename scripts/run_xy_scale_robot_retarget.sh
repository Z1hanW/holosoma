#!/usr/bin/env bash
set -u

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/FAR/holosoma}
RETARGET_ROOT=${RETARGET_ROOT:-${REPO_ROOT}/src/holosoma_retargeting}
PYTHON_BIN=${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/retgt/bin/python}

OMOMO_SOURCE=${OMOMO_SOURCE:-/data/OMOMO_new}
OMOMO_FILTER_CONVERTED=${OMOMO_FILTER_CONVERTED:-/data/holosoma_moved/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry}
OMOMO_FILTERED_DIR=${OMOMO_FILTERED_DIR:-/tmp/omomo_carry_filtered_62_for_retarget}

BEHAVE_SOURCE=${BEHAVE_SOURCE:-/data/behave/annotation_30fps_zup_carry}
BEHAVE_OBJECT_ROOT=${BEHAVE_OBJECT_ROOT:-/data/behave/objects}

RAW_OMOMO_ROOT=${RAW_OMOMO_ROOT:-${RETARGET_ROOT}/demo_results_parallel/g1/object_interaction/omomo_carry_xy_0p5_1p5}
RAW_BEHAVE_ROOT=${RAW_BEHAVE_ROOT:-${RETARGET_ROOT}/demo_results_parallel/g1/object_interaction/behave_zup_sq_carry_xy_0p5_1p5}
CONV_OMOMO_ROOT=${CONV_OMOMO_ROOT:-/data/holosoma_moved/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry_xy_0p5_1p5}
CONV_BEHAVE_ROOT=${CONV_BEHAVE_ROOT:-/data/holosoma_moved/src/holosoma_retargeting/converted_res/behave_sq_carry_xy_0p5_1p5}

MAX_WORKERS_OMOMO=${MAX_WORKERS_OMOMO:-8}
MAX_WORKERS_BEHAVE=${MAX_WORKERS_BEHAVE:-4}

LOG_DIR=${LOG_DIR:-/tmp/retarget_xy_scale_logs}
SUMMARY_FILE=${SUMMARY_FILE:-${LOG_DIR}/summary.tsv}
mkdir -p "${LOG_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${RETARGET_ROOT}" ]]; then
  echo "[ERROR] RETARGET_ROOT not found: ${RETARGET_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${OMOMO_SOURCE}" ]]; then
  echo "[ERROR] OMOMO_SOURCE not found: ${OMOMO_SOURCE}" >&2
  exit 1
fi
if [[ ! -d "${OMOMO_FILTER_CONVERTED}" ]]; then
  echo "[ERROR] OMOMO_FILTER_CONVERTED not found: ${OMOMO_FILTER_CONVERTED}" >&2
  exit 1
fi
if [[ ! -d "${BEHAVE_SOURCE}" ]]; then
  echo "[ERROR] BEHAVE_SOURCE not found: ${BEHAVE_SOURCE}" >&2
  exit 1
fi
if [[ ! -d "${BEHAVE_OBJECT_ROOT}" ]]; then
  echo "[ERROR] BEHAVE_OBJECT_ROOT not found: ${BEHAVE_OBJECT_ROOT}" >&2
  exit 1
fi

# Ensure clean outputs (delete previous generated scale-augmentation data roots)
"${PYTHON_BIN}" - <<PY
import shutil
from pathlib import Path
for p in [
    Path("${RAW_OMOMO_ROOT}"),
    Path("${RAW_BEHAVE_ROOT}"),
    Path("${CONV_OMOMO_ROOT}"),
    Path("${CONV_BEHAVE_ROOT}"),
    Path("${OMOMO_FILTERED_DIR}"),
]:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
print("[INFO] cleaned roots")
PY

# Build filtered OMOMO source subset from filtered converted list.
"${PYTHON_BIN}" - <<PY
from pathlib import Path
import sys
src = Path("${OMOMO_SOURCE}")
conv = Path("${OMOMO_FILTER_CONVERTED}")
out = Path("${OMOMO_FILTERED_DIR}")
missing = []
count = 0
for npz in sorted(conv.glob("*_mj_w_obj.npz")):
    stem = npz.name.replace("_mj_w_obj.npz", "")
    pt = src / f"{stem}.pt"
    if not pt.exists():
        missing.append(stem)
        continue
    target = out / f"{stem}.pt"
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(pt)
    count += 1
print(f"[INFO] OMOMO filtered subset built: {count} clips")
if missing:
    print("[ERROR] Missing OMOMO clips:", *missing[:20], sep="\n")
    sys.exit(1)
PY

echo -e "scale\ttag\tomomo_retgt\tomomo_raw\tomomo_conv\tbehave_retgt\tbehave_raw\tbehave_conv" > "${SUMMARY_FILE}"

mapfile -t SCALE_ROWS < <("${PYTHON_BIN}" - <<'PY'
for i in range(5, 16):
    s = i / 10
    print(f"{s:.1f} xy{i*10:03d}")
PY
)

for row in "${SCALE_ROWS[@]}"; do
  scale="${row%% *}"
  tag="${row##* }"

  omomo_raw="${RAW_OMOMO_ROOT}/${tag}"
  omomo_conv="${CONV_OMOMO_ROOT}/${tag}"
  behave_raw="${RAW_BEHAVE_ROOT}/${tag}"
  behave_conv="${CONV_BEHAVE_ROOT}/${tag}"

  mkdir -p "${omomo_raw}" "${omomo_conv}" "${behave_raw}" "${behave_conv}"

  omomo_retgt_status=ok
  behave_retgt_status=ok

  echo "[INFO] === scale=${scale} tag=${tag} OMOMO retarget ==="
  if ! (
    cd "${RETARGET_ROOT}" &&
    "${PYTHON_BIN}" examples/parallel_robot_retarget.py \
      --task-type object_interaction \
      --robot g1 \
      --data-format smplh \
      --data-dir "${OMOMO_FILTERED_DIR}" \
      --save-dir "${omomo_raw}" \
      --max-workers "${MAX_WORKERS_OMOMO}" \
      --task-config.object-name largebox \
      --task-config.object-interaction-scale-augmented "${scale}" "${scale}" 1.0
  ) > "${LOG_DIR}/omomo_${tag}.log" 2>&1; then
    omomo_retgt_status=failed
  fi

  omomo_raw_count=$(find "${omomo_raw}" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')
  omomo_conv_count=0
  if [[ "${omomo_raw_count}" -gt 0 ]]; then
    echo "[INFO] === scale=${scale} tag=${tag} OMOMO convert ==="
    if (
      cd "${REPO_ROOT}" &&
      INPUT_DIR="${omomo_raw}" OUTPUT_DIR="${omomo_conv}" PYTHON_BIN="${PYTHON_BIN}" bash retgt_post_omomo.sh
    ) >> "${LOG_DIR}/omomo_${tag}.log" 2>&1; then
      omomo_conv_count=$(find "${omomo_conv}" -maxdepth 1 -type f -name '*_mj_w_obj.npz' | wc -l | tr -d ' ')
    fi
  fi

  echo "[INFO] === scale=${scale} tag=${tag} BEHAVE retarget ==="
  if ! (
    cd "${RETARGET_ROOT}" &&
    "${PYTHON_BIN}" examples/parallel_robot_retarget.py \
      --task-type object_interaction \
      --robot g1 \
      --data-format behave_zup \
      --data-dir "${BEHAVE_SOURCE}" \
      --save-dir "${behave_raw}" \
      --max-workers "${MAX_WORKERS_BEHAVE}" \
      --task-config.object-mesh-root "${BEHAVE_OBJECT_ROOT}" \
      --task-config.object-mesh-suffix "_f1000.ply" \
      --task-config.object-interaction-scale-augmented "${scale}" "${scale}" 1.0
  ) > "${LOG_DIR}/behave_${tag}.log" 2>&1; then
    behave_retgt_status=failed
  fi

  behave_raw_count=$(find "${behave_raw}" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')
  behave_conv_count=0
  if [[ "${behave_raw_count}" -gt 0 ]]; then
    echo "[INFO] === scale=${scale} tag=${tag} BEHAVE convert ==="
    if (
      cd "${REPO_ROOT}" &&
      INPUT_DIR="${behave_raw}" OUTPUT_DIR="${behave_conv}" ROBOT="g1" PYTHON_BIN="${PYTHON_BIN}" DATA_FORMAT="behave_zup" bash retgt_post_behave.sh
    ) >> "${LOG_DIR}/behave_${tag}.log" 2>&1; then
      behave_conv_count=$(find "${behave_conv}" -maxdepth 1 -type f -name '*_mj_w_obj.npz' | wc -l | tr -d ' ')
    fi
  fi

  echo -e "${scale}\t${tag}\t${omomo_retgt_status}\t${omomo_raw_count}\t${omomo_conv_count}\t${behave_retgt_status}\t${behave_raw_count}\t${behave_conv_count}" >> "${SUMMARY_FILE}"
  echo "[INFO] scale=${scale} done | OMOMO raw=${omomo_raw_count} conv=${omomo_conv_count} | BEHAVE raw=${behave_raw_count} conv=${behave_conv_count}"
done

echo "[INFO] All scales finished. Summary: ${SUMMARY_FILE}"
