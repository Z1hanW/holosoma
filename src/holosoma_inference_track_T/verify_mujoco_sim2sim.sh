#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if ! python3 -c "import mujoco" >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    echo "[verify] python env lacks mujoco; re-running in conda env 'hsmujoco'"
    exec conda run -n hsmujoco bash "$0" "$@"
  fi
  echo "[verify] mujoco python package not found and conda unavailable" >&2
  exit 1
fi

DEFAULT_CKPT="/home/ANT.AMAZON.COM/zzzihanw/FAR/ckp/singlebox_07000.onnx"
DEFAULT_MOTION_ROBOT_ONLY="${REPO_ROOT}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz"
DEFAULT_MOTION_ROBOT_W_OBJ="${REPO_ROOT}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"

CHECKPOINT=${CHECKPOINT:-"${DEFAULT_CKPT}"}
MOTION_ROBOT_ONLY=${MOTION_ROBOT_ONLY:-"${DEFAULT_MOTION_ROBOT_ONLY}"}
MOTION_ROBOT_W_OBJ=${MOTION_ROBOT_W_OBJ:-"${DEFAULT_MOTION_ROBOT_W_OBJ}"}
OBJECT_URDF=${OBJECT_URDF:-"holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"}

HEADLESS=${HEADLESS:-"True"}
REPLAY_END_FRAME=${REPLAY_END_FRAME:-400}
LOG_EVERY=${LOG_EVERY:-20}

POLICY_AUTO_LAUNCH=${POLICY_AUTO_LAUNCH:-"False"}
POLICY_TIMEOUT_SEC=${POLICY_TIMEOUT_SEC:-60}
POLICY_REQUIRE_SUCCESS_ROWS=${POLICY_REQUIRE_SUCCESS_ROWS:-1}
POLICY_REQUIRE_DIRECTIONAL_MOTION=${POLICY_REQUIRE_DIRECTIONAL_MOTION:-"True"}
POLICY_DIRECTION_MIN_STEP=${POLICY_DIRECTION_MIN_STEP:-0}
POLICY_DIRECTION_MIN_DURATION_SEC=${POLICY_DIRECTION_MIN_DURATION_SEC:-"2.0"}
POLICY_DIRECTION_MIN_COSINE=${POLICY_DIRECTION_MIN_COSINE:-"0.5"}
POLICY_DIRECTION_MIN_PAIR_COSINE=${POLICY_DIRECTION_MIN_PAIR_COSINE:-"0.3"}
POLICY_DIRECTION_MIN_ROOT_SPEED_MPS=${POLICY_DIRECTION_MIN_ROOT_SPEED_MPS:-"0.05"}
POLICY_DIRECTION_MIN_OBJECT_SPEED_MPS=${POLICY_DIRECTION_MIN_OBJECT_SPEED_MPS:-"0.05"}
POLICY_DIRECTION_MIN_ROOT_NET_DISP=${POLICY_DIRECTION_MIN_ROOT_NET_DISP:-"0.25"}
POLICY_DIRECTION_MIN_OBJECT_NET_DISP=${POLICY_DIRECTION_MIN_OBJECT_NET_DISP:-"0.25"}

TS="$(date -u +%Y%m%d_%H%M%S)"
OUT_DIR="${REPO_ROOT}/logs/replay_motion_mujoco/verify_${TS}"
mkdir -p "${OUT_DIR}"

ROBOT_ONLY_LOG="${OUT_DIR}/robot_only_pose.csv"
ROBOT_W_OBJ_LOG="${OUT_DIR}/robot_w_obj_pose.csv"

FAILURES=0

echo "[verify] repo=${REPO_ROOT}"
echo "[verify] out_dir=${OUT_DIR}"
echo "[verify] checkpoint=${CHECKPOINT}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[verify] FAIL: checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -f "${MOTION_ROBOT_ONLY}" ]]; then
  echo "[verify] FAIL: robot-only motion file not found: ${MOTION_ROBOT_ONLY}" >&2
  exit 1
fi

if [[ ! -f "${MOTION_ROBOT_W_OBJ}" ]]; then
  echo "[verify] FAIL: robot-w-obj motion file not found: ${MOTION_ROBOT_W_OBJ}" >&2
  exit 1
fi

check_pose_success() {
  local csv_path="$1"
  local label="$2"
  if [[ ! -s "${csv_path}" ]]; then
    echo "[verify][${label}] FAIL: missing csv ${csv_path}" >&2
    FAILURES=$((FAILURES + 1))
    return
  fi

  local success_count
  success_count="$(awk -F, 'NR>1 && $9==1 {c++} END {print c+0}' "${csv_path}")"
  local last_row
  last_row="$(tail -n 1 "${csv_path}")"

  if [[ "${success_count}" -gt 0 ]]; then
    echo "[verify][${label}] PASS: success frames=${success_count}, last_row=${last_row}"
  else
    echo "[verify][${label}] FAIL: no success frame found, last_row=${last_row}" >&2
    FAILURES=$((FAILURES + 1))
  fi
}

check_directional_motion() {
  local csv_path="$1"
  local label="$2"
  if ! python3 - \
    "${csv_path}" \
    "${POLICY_DIRECTION_MIN_STEP}" \
    "${POLICY_DIRECTION_MIN_DURATION_SEC}" \
    "${POLICY_DIRECTION_MIN_COSINE}" \
    "${POLICY_DIRECTION_MIN_PAIR_COSINE}" \
    "${POLICY_DIRECTION_MIN_ROOT_SPEED_MPS}" \
    "${POLICY_DIRECTION_MIN_OBJECT_SPEED_MPS}" \
    "${POLICY_DIRECTION_MIN_ROOT_NET_DISP}" \
    "${POLICY_DIRECTION_MIN_OBJECT_NET_DISP}" <<'PY'
import csv
import math
import sys

csv_path = sys.argv[1]
min_step = int(float(sys.argv[2]))
min_duration = float(sys.argv[3])
min_cos = float(sys.argv[4])
min_pair_cos = float(sys.argv[5])
min_root_speed = float(sys.argv[6])
min_object_speed = float(sys.argv[7])
min_root_disp = float(sys.argv[8])
min_object_disp = float(sys.argv[9])

rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
if not rows:
    print("FAIL empty csv")
    raise SystemExit(1)

def fval(row, key):
    raw = row.get(key)
    if raw is None or raw == "":
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")

def ival(row, key):
    raw = row.get(key)
    if raw is None or raw == "":
        return -1
    try:
        return int(float(raw))
    except ValueError:
        return -1

eval_rows = [r for r in rows if ival(r, "step") >= min_step]
if len(eval_rows) < 2:
    print(f"FAIL eval_rows={len(eval_rows)} (<2)")
    raise SystemExit(1)

root_start = (fval(eval_rows[0], "root_x"), fval(eval_rows[0], "root_y"))
root_end = (fval(eval_rows[-1], "root_x"), fval(eval_rows[-1], "root_y"))
obj_start = (fval(eval_rows[0], "object_x"), fval(eval_rows[0], "object_y"))
obj_end = (fval(eval_rows[-1], "object_x"), fval(eval_rows[-1], "object_y"))

if any(not math.isfinite(v) for v in (*root_start, *root_end)):
    print("FAIL missing root_xy")
    raise SystemExit(1)
if any(not math.isfinite(v) for v in (*obj_start, *obj_end)):
    print("FAIL missing object_xy")
    raise SystemExit(1)

root_dx = root_end[0] - root_start[0]
root_dy = root_end[1] - root_start[1]
root_disp = math.hypot(root_dx, root_dy)
obj_disp = math.hypot(obj_end[0] - obj_start[0], obj_end[1] - obj_start[1])

if root_disp < min_root_disp:
    print(f"FAIL root_disp={root_disp:.3f} (<{min_root_disp:.3f})")
    raise SystemExit(1)
if obj_disp < min_object_disp:
    print(f"FAIL object_disp={obj_disp:.3f} (<{min_object_disp:.3f})")
    raise SystemExit(1)

heading = (root_dx / (root_disp + 1e-9), root_dy / (root_disp + 1e-9))

dt_samples = []
for i in range(1, len(eval_rows)):
    dt = fval(eval_rows[i], "sim_time") - fval(eval_rows[i - 1], "sim_time")
    if dt > 0 and math.isfinite(dt):
        dt_samples.append(dt)
dt_samples.sort()
median_dt = dt_samples[len(dt_samples) // 2] if dt_samples else 0.1

best_dur = 0.0
best_rows = 0
run_dur = 0.0
run_rows = 0

for i in range(1, len(eval_rows)):
    prev = eval_rows[i - 1]
    curr = eval_rows[i]
    dt = fval(curr, "sim_time") - fval(prev, "sim_time")
    if not math.isfinite(dt) or dt <= 0:
        dt = median_dt
    drx = fval(curr, "root_x") - fval(prev, "root_x")
    dry = fval(curr, "root_y") - fval(prev, "root_y")
    dox = fval(curr, "object_x") - fval(prev, "object_x")
    doy = fval(curr, "object_y") - fval(prev, "object_y")
    if not all(math.isfinite(v) for v in (drx, dry, dox, doy)):
        good = False
    else:
        root_step_disp = math.hypot(drx, dry)
        obj_step_disp = math.hypot(dox, doy)
        root_speed = root_step_disp / max(dt, 1e-9)
        obj_speed = obj_step_disp / max(dt, 1e-9)
        root_cos = (drx * heading[0] + dry * heading[1]) / max(root_step_disp, 1e-9)
        obj_cos = (dox * heading[0] + doy * heading[1]) / max(obj_step_disp, 1e-9)
        pair_cos = (drx * dox + dry * doy) / max(root_step_disp * obj_step_disp, 1e-9)
        stable = curr.get("raw_success", "1") == "1"
        good = (
            stable
            and root_speed >= min_root_speed
            and obj_speed >= min_object_speed
            and root_cos >= min_cos
            and obj_cos >= min_cos
            and pair_cos >= min_pair_cos
        )
    if good:
        run_rows += 1
        run_dur += max(dt, 0.0)
    else:
        if run_dur > best_dur:
            best_dur = run_dur
            best_rows = run_rows
        run_rows = 0
        run_dur = 0.0

if run_dur > best_dur:
    best_dur = run_dur
    best_rows = run_rows

if best_dur < min_duration:
    print(
        f"FAIL directional_duration={best_dur:.3f}s (<{min_duration:.3f}s) "
        f"eval_rows={len(eval_rows)} root_disp={root_disp:.3f} object_disp={obj_disp:.3f}"
    )
    raise SystemExit(1)

print(
    f"PASS directional_duration={best_dur:.3f}s rows={best_rows} "
    f"eval_rows={len(eval_rows)} root_disp={root_disp:.3f} object_disp={obj_disp:.3f}"
)
PY
  then
    echo "[verify][${label}] FAIL: directional motion check failed (${csv_path})" >&2
    FAILURES=$((FAILURES + 1))
  fi
}

run_policy_check() {
  local label="$1"
  local script_path="$2"
  local policy_log="${OUT_DIR}/${label}_rollout_pose.csv"

  echo "[verify][${label}] running policy wrapper (${script_path})"
  if [[ "${POLICY_AUTO_LAUNCH,,}" == "true" || "${POLICY_AUTO_LAUNCH}" == "1" ]]; then
    set +e
    AUTO_LAUNCH=True HEADLESS="${HEADLESS}" OBJECT_URDF="${OBJECT_URDF}" \
      ROLLOUT_REQUIRE_DIRECTIONAL_MOTION="${POLICY_REQUIRE_DIRECTIONAL_MOTION}" \
      ROLLOUT_DIRECTION_MIN_STEP="${POLICY_DIRECTION_MIN_STEP}" \
      ROLLOUT_DIRECTION_MIN_DURATION_SEC="${POLICY_DIRECTION_MIN_DURATION_SEC}" \
      ROLLOUT_DIRECTION_MIN_COSINE="${POLICY_DIRECTION_MIN_COSINE}" \
      ROLLOUT_DIRECTION_MIN_PAIR_COSINE="${POLICY_DIRECTION_MIN_PAIR_COSINE}" \
      ROLLOUT_DIRECTION_MIN_ROOT_SPEED_MPS="${POLICY_DIRECTION_MIN_ROOT_SPEED_MPS}" \
      ROLLOUT_DIRECTION_MIN_OBJECT_SPEED_MPS="${POLICY_DIRECTION_MIN_OBJECT_SPEED_MPS}" \
      ROLLOUT_DIRECTION_MIN_ROOT_NET_DISP="${POLICY_DIRECTION_MIN_ROOT_NET_DISP}" \
      ROLLOUT_DIRECTION_MIN_OBJECT_NET_DISP="${POLICY_DIRECTION_MIN_OBJECT_NET_DISP}" \
      ROLLOUT_POSE_LOG="${policy_log}" timeout "${POLICY_TIMEOUT_SEC}" bash "${script_path}" "${CHECKPOINT}"
    local rc=$?
    set -e
    if [[ "${rc}" -eq 0 ]]; then
      echo "[verify][${label}] PASS: auto-launch run finished before timeout"
    elif [[ "${rc}" -eq 124 ]]; then
      echo "[verify][${label}] PASS: auto-launch reached timeout (${POLICY_TIMEOUT_SEC}s), treated as started/healthy"
    else
      echo "[verify][${label}] FAIL: policy wrapper exited with code ${rc}" >&2
      FAILURES=$((FAILURES + 1))
      return
    fi

    if [[ ! -s "${policy_log}" ]]; then
      echo "[verify][${label}] FAIL: rollout pose log missing: ${policy_log}" >&2
      FAILURES=$((FAILURES + 1))
      return
    fi
    local success_rows
    success_rows="$(awk -F, 'NR>1 && $10==1 {c++} END {print c+0}' "${policy_log}")"
    local last_row
    last_row="$(tail -n 1 "${policy_log}")"
    if [[ "${success_rows}" -ge "${POLICY_REQUIRE_SUCCESS_ROWS}" ]]; then
      echo "[verify][${label}] PASS: rollout success rows=${success_rows}, last_row=${last_row}"
    else
      echo "[verify][${label}] FAIL: rollout success rows=${success_rows}, last_row=${last_row}" >&2
      FAILURES=$((FAILURES + 1))
    fi
    if [[ "${label}" == "policy_robot_w_obj" && ( "${POLICY_REQUIRE_DIRECTIONAL_MOTION,,}" == "true" || "${POLICY_REQUIRE_DIRECTIONAL_MOTION}" == "1" ) ]]; then
      check_directional_motion "${policy_log}" "${label}"
    fi
  else
    set +e
    AUTO_LAUNCH=False HEADLESS="${HEADLESS}" OBJECT_URDF="${OBJECT_URDF}" \
      bash "${script_path}" "${CHECKPOINT}"
    local rc=$?
    set -e
    if [[ "${rc}" -eq 0 ]]; then
      echo "[verify][${label}] PASS: command generation OK (no auto-launch)"
    else
      echo "[verify][${label}] FAIL: wrapper exited with code ${rc}" >&2
      FAILURES=$((FAILURES + 1))
    fi
  fi
}

echo "[verify] replay robot-only"
MOTION_FILE="${MOTION_ROBOT_ONLY}" POSE_LOG="${ROBOT_ONLY_LOG}" HEADLESS="${HEADLESS}" LOOP=False \
  END_FRAME="${REPLAY_END_FRAME}" LOG_EVERY="${LOG_EVERY}" \
  bash "${REPO_ROOT}/src/holosoma_inference/replay_mujoco_robot_only.sh"
check_pose_success "${ROBOT_ONLY_LOG}" "replay_robot_only"

echo "[verify] replay robot-w-obj"
MOTION_FILE="${MOTION_ROBOT_W_OBJ}" OBJECT_URDF="${OBJECT_URDF}" POSE_LOG="${ROBOT_W_OBJ_LOG}" \
  HEADLESS="${HEADLESS}" LOOP=False END_FRAME="${REPLAY_END_FRAME}" LOG_EVERY="${LOG_EVERY}" \
  bash "${REPO_ROOT}/src/holosoma_inference/replay_mujoco_robot_w_obj.sh"
check_pose_success "${ROBOT_W_OBJ_LOG}" "replay_robot_w_obj"

run_policy_check "policy_robot_only" "${REPO_ROOT}/src/holosoma_inference/play_policy_mujoco_robot_only.sh"
run_policy_check "policy_robot_w_obj" "${REPO_ROOT}/src/holosoma_inference/play_policy_mujoco_robot_w_obj.sh"

if [[ "${FAILURES}" -ne 0 ]]; then
  echo "[verify] FAILED with ${FAILURES} issue(s). Artifacts: ${OUT_DIR}" >&2
  exit 1
fi

echo "[verify] SUCCESS. Artifacts: ${OUT_DIR}"
