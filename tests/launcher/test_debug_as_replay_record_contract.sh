#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "${REPO_ROOT}"

TMP_DIR=$(mktemp -d)
trap 'rm -rf -- "${TMP_DIR}"' EXIT

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

FFMPEG_BIN=$(command -v ffmpeg || true)
FFPROBE_BIN=$(command -v ffprobe || true)
[[ -n "${FFMPEG_BIN}" && -x "${FFMPEG_BIN}" ]] || fail 'ffmpeg is required for this CPU-only regression'
[[ -n "${FFPROBE_BIN}" && -x "${FFPROBE_BIN}" ]] || fail 'ffprobe is required for this CPU-only regression'

FAKE_PYTHON="${TMP_DIR}/fake-python"
cat >"${FAKE_PYTHON}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

target=$1
shift
case "${target}" in
  */prepare_as_replay_pairs.py)
    manifest=''
    work_root=''
    video_root=''
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --manifest) manifest=$2; shift 2 ;;
        --work-root) work_root=$2; shift 2 ;;
        --video-root) video_root=$2; shift 2 ;;
        *) shift ;;
      esac
    done
    [[ -n "${manifest}" && -n "${work_root}" && -n "${video_root}" ]]
    pair_dir="${work_root}/test_clip"
    video_dir="${video_root}/test_clip"
    mkdir -p "${pair_dir}" "${video_dir}" "$(dirname "${manifest}")"
    printf '{}\n' >"${pair_dir}/map.json"
    printf '<robot name="test"/>\n' >"${pair_dir}/object.urdf"
    printf 'fixture\n' >"${pair_dir}/test_clip.npz"
    printf 'test_clip\t%s\t%s\t%s\t%s\t%s\n' \
      "${pair_dir}" \
      "${pair_dir}/map.json" \
      "${pair_dir}/object.urdf" \
      "${pair_dir}/test_clip.npz" \
      "${video_dir}" >"${manifest}"
    ;;
  */replay.py)
    video_dir=''
    while [[ $# -gt 0 ]]; do
      if [[ "$1" == '--logger.video.save_dir' ]]; then
        video_dir=$2
        break
      fi
      shift
    done
    [[ -n "${video_dir}" ]]
    mkdir -p "${video_dir}"
    case "${FAKE_REPLAY_MODE:?}" in
      producer_failure)
        echo 'simulated replay producer failure' >&2
        exit 17
        ;;
      missing_mp4)
        exit 0
        ;;
      invalid_mp4)
        printf 'not an mp4\n' >"${video_dir}/episode.mp4"
        ;;
      success)
        "${FAKE_FFMPEG_BIN:?}" -v error -y \
          -f lavfi -i color=c=black:s=32x32:r=25:d=0.08 \
          -frames:v 2 -c:v libx264 -pix_fmt yuv420p \
          "${video_dir}/episode.mp4"
        ;;
      *)
        echo "unsupported fake replay mode: ${FAKE_REPLAY_MODE}" >&2
        exit 97
        ;;
    esac
    ;;
  *)
    echo "unexpected fake-python target: ${target}" >&2
    exit 98
    ;;
esac
SH
chmod +x "${FAKE_PYTHON}"

run_launcher() {
  local mode=$1
  local tee_logs=$2
  local case_root="${TMP_DIR}/${mode}"
  mkdir -p "${case_root}"
  env \
    AS_DEBUG_ROOT="${case_root}/output" \
    AS_DATA_DIR="${case_root}/input" \
    AS_OBJECT_MAP="${case_root}/input/map.json" \
    AS_EXPECTED_TOTAL=1 \
    CUDA_VISIBLE_DEVICES=0 \
    FAKE_FFMPEG_BIN="${FFMPEG_BIN}" \
    FAKE_REPLAY_MODE="${mode}" \
    FFPROBE_BIN="${FFPROBE_BIN}" \
    KEEP_GOING=1 \
    PYTHON_BIN="${FAKE_PYTHON}" \
    TEE_LOGS="${tee_logs}" \
    bash debug_as_replay_record.sh --clip test_clip
}

expect_failure() {
  local mode=$1
  local tee_logs=$2
  local expected=$3
  local output="${TMP_DIR}/${mode}.out"
  if run_launcher "${mode}" "${tee_logs}" >"${output}" 2>&1; then
    sed -n '1,120p' "${output}" >&2
    fail "${mode} unexpectedly succeeded"
  fi
  grep -F -- "${expected}" "${output}" >/dev/null || {
    sed -n '1,120p' "${output}" >&2
    fail "${mode} did not report expected failure: ${expected}"
  }
  if grep -F '[INFO] Completed 1 replay recording(s).' "${output}" >/dev/null; then
    fail "${mode} printed a false successful completion"
  fi
}

bash -n debug_as_replay_record.sh "${FAKE_PYTHON}"

expect_failure producer_failure 1 '[ERROR] Replay producer failed for test_clip'
expect_failure missing_mp4 0 \
  '[ERROR] Replay producer must create exactly one new or changed MP4 for test_clip; found=0.'
expect_failure invalid_mp4 1 '[ERROR] Replay artifact validation failed for test_clip.'

SUCCESS_OUTPUT="${TMP_DIR}/success.out"
run_launcher success 0 >"${SUCCESS_OUTPUT}" 2>&1 || {
  sed -n '1,160p' "${SUCCESS_OUTPUT}" >&2
  fail 'valid replay artifact path failed'
}
grep -F '[INFO] Replay artifact verified:' "${SUCCESS_OUTPUT}" >/dev/null \
  || fail 'successful path omitted artifact verification'
grep -F '[INFO] Completed 1 replay recording(s).' "${SUCCESS_OUTPUT}" >/dev/null \
  || fail 'successful path omitted completion'
SUCCESS_MP4="${TMP_DIR}/success/output/videos/test_clip/episode.mp4"
"${FFPROBE_BIN}" -v error -count_frames -select_streams v:0 \
  -show_entries stream=nb_read_frames -of default=noprint_wrappers=1:nokey=1 \
  "${SUCCESS_MP4}" | grep -Fx '2' >/dev/null \
  || fail 'successful fixture MP4 was not preserved as a two-frame video'

echo '[PASS] debug AS replay producer/artifact fail-closed contract'
