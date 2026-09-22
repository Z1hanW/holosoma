#!/usr/bin/env bash
# Laptop side of the FFS depth path, and by default the robot side too.
#
# Starts stereo_relay_pub.py on the Jetson over ssh (via ~/depth_relay/real_ffs_relay.sh
# there), waits until it is publishing, then runs image_server.py here: the D435i IR
# stereo pair arrives over the network, Fast-FoundationStereo runs on the local GPU,
# and the policy's depth_img_shm is written exactly as real_depth.sh would from a
# local camera. On exit (Ctrl-C, SIGTERM) the remote relay is stopped as well.
#
# Set HOLOSOMA_RELAY_HOST= (empty) to skip the remote step if you started the relay
# by hand. Start this before real_ffs_run.sh.
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export HOLOSOMA_DEPTH_PREDICTOR=ffs
export HOLOSOMA_FFS_REPO="${HOLOSOMA_FFS_REPO:-$HOME/FAR/Fast-FoundationStereo}"
export HOLOSOMA_FFS_MODEL="${HOLOSOMA_FFS_MODEL:-$HOLOSOMA_FFS_REPO/weights/c-ffs/model_best_bp2_serialize.pth}"

# ${VAR-default} (no colon) so an explicitly empty value means "don't manage the relay".
RELAY_HOST="${HOLOSOMA_RELAY_HOST-192.168.123.164}"
RELAY_PORT="${HOLOSOMA_RELAY_PORT:-5602}"
RELAY_DIR="${HOLOSOMA_RELAY_REMOTE_DIR:-~/depth_relay}"
# Space-separated systemd --user services on the robot that hold the camera (e.g.
# "lsvla-vision"). They are stopped before the relay starts and started again on
# exit if they were active. Off by default: stopping someone else's service is a
# decision, not a default.
RELAY_STOP_SERVICES="${HOLOSOMA_RELAY_STOP_SERVICES:-}"
export HOLOSOMA_REMOTE_STEREO_CONNECT="${HOLOSOMA_REMOTE_STEREO_CONNECT:-tcp://${RELAY_HOST:-192.168.123.164}:${RELAY_PORT}}"

for f in "$HOLOSOMA_FFS_MODEL" "$(dirname "$HOLOSOMA_FFS_MODEL")/cfg.yaml"; do
  if [[ ! -f "$f" ]]; then
    echo "[real_ffs_depth] ERROR: missing $f" >&2
    exit 1
  fi
done

log_dir="${ROOT_DIR}/logs/real_ffs_depth_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"

# Record what the policy actually saw. Each .npz under evidence/ holds the metric
# FFS depth, the crop/resize/clip result, the delayed 58x87 frame written to shared
# memory, and the IR stereo pair it came from. Measured at ~1.4 MB per record, so
# every 3rd frame (10 Hz, ~14 MB/s, ~50 GB/h) by default; EVERY=1 keeps every
# frame but the compressor cannot keep up at 30 Hz and starts dropping records.
# HOLOSOMA_DEPLOYMENT_AUDIT=0 turns recording off.
if [[ "${HOLOSOMA_DEPLOYMENT_AUDIT:-1}" == "1" ]]; then
  export HOLOSOMA_DEPLOYMENT_AUDIT_DIR="${log_dir}/evidence"
  export HOLOSOMA_AUDIT_DEPTH_EVERY="${HOLOSOMA_AUDIT_DEPTH_EVERY:-3}"
  export HOLOSOMA_AUDIT_DEPTH_LIMIT="${HOLOSOMA_AUDIT_DEPTH_LIMIT:-36000}"
else
  unset HOLOSOMA_DEPLOYMENT_AUDIT_DIR
fi

exec > >(tee -a "${log_dir}/depth.log") 2>&1
echo "[real_ffs_depth] log_dir=${log_dir}"
[[ -n "${HOLOSOMA_DEPLOYMENT_AUDIT_DIR:-}" ]] && echo "[real_ffs_depth] recording depth evidence to ${HOLOSOMA_DEPLOYMENT_AUDIT_DIR} (every ${HOLOSOMA_AUDIT_DEPTH_EVERY} frame(s), limit ${HOLOSOMA_AUDIT_DEPTH_LIMIT})"

SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5)

stop_remote_relay() {
  [[ -n "$RELAY_HOST" ]] || return 0
  local out
  out=$("${SSH[@]}" "$RELAY_HOST" "if [ -f ${RELAY_DIR}/relay.pid ]; then kill \"\$(cat ${RELAY_DIR}/relay.pid)\" 2>/dev/null && echo killed; rm -f ${RELAY_DIR}/relay.pid; fi
    if [ -f ${RELAY_DIR}/services.stopped ]; then for s in \$(cat ${RELAY_DIR}/services.stopped); do systemctl --user start \"\$s\" && echo \"restarted \$s\"; done; rm -f ${RELAY_DIR}/services.stopped; fi" 2>/dev/null) \
    || { echo "[real_ffs_depth] WARNING: could not reach ${RELAY_HOST} to stop the relay; check ${RELAY_DIR}/relay.pid there" >&2; return 0; }
  [[ "$out" == *killed* ]] && echo "[real_ffs_depth] remote relay stopped"
  [[ "$out" == *restarted* ]] && echo "[real_ffs_depth] remote services restored: $(printf '%s' "$out" | sed -n 's/^restarted //p' | tr '\n' ' ')"
  return 0
}

install_remote_relay() {
  # The relay is two files that live in this repo. Push them to the robot when
  # they are missing or differ, so a wiped home directory or a stale copy on the
  # Jetson can never be the reason deployment fails.
  local remote_sums
  remote_sums=$("${SSH[@]}" "$RELAY_HOST" "mkdir -p ${RELAY_DIR} && cd ${RELAY_DIR} && md5sum stereo_relay_pub.py real_ffs_relay.sh 2>/dev/null | awk '{print \$1}' | tr '\n' ' '") \
    || { echo "[real_ffs_depth] ERROR: cannot ssh to ${RELAY_HOST}" >&2; return 1; }
  local local_sums
  local_sums=$(md5sum scripts/stereo_relay_pub.py scripts/real_ffs_relay.sh | awk '{print $1}' | tr '\n' ' ')
  if [[ "$remote_sums" != "$local_sums" ]]; then
    echo "[real_ffs_depth] installing relay scripts to ${RELAY_HOST}:${RELAY_DIR}"
    scp -q -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
      scripts/stereo_relay_pub.py scripts/real_ffs_relay.sh "${RELAY_HOST}:${RELAY_DIR}/" \
      || { echo "[real_ffs_depth] ERROR: scp to ${RELAY_HOST} failed" >&2; return 1; }
    "${SSH[@]}" "$RELAY_HOST" "chmod +x ${RELAY_DIR}/real_ffs_relay.sh"
  fi
}

start_remote_relay() {
  install_remote_relay || exit 1
  echo "[real_ffs_depth] starting stereo relay on ${RELAY_HOST} (${RELAY_DIR}/real_ffs_relay.sh)"
  # Kill a previous relay via its pidfile, launch detached from this ssh session,
  # then block until it reports it is publishing so image_server never has to wait.
  "${SSH[@]}" "$RELAY_HOST" "bash -s" <<EOF
set -e
cd ${RELAY_DIR}
[ -f relay.pid ] && kill "\$(cat relay.pid)" 2>/dev/null && sleep 0.5
rm -f relay.log services.stopped
for s in ${RELAY_STOP_SERVICES}; do
  if systemctl --user is-active --quiet "\$s"; then
    systemctl --user stop "\$s" && echo "\$s" >> services.stopped && echo "  stopped \$s (will restart on exit)"
  fi
done
[ -s services.stopped ] && sleep 1.5
HOLOSOMA_RELAY_BIND='tcp://*:${RELAY_PORT}' setsid nohup ./real_ffs_relay.sh > relay.log 2>&1 < /dev/null &
echo \$! > relay.pid
for i in \$(seq 1 120); do
  grep -q publishing relay.log 2>/dev/null && { echo "  relay pid \$(cat relay.pid): \$(grep -m1 publishing relay.log)"; exit 0; }
  kill -0 "\$(cat relay.pid)" 2>/dev/null || break
  sleep 0.25
done
echo "  relay did not start within 30 s; remote log:" >&2; tail -8 relay.log >&2
echo "  camera holders now:" >&2; for v in /dev/video*; do fuser "\$v" 2>/dev/null | xargs -r -n1 ps -o pid=,cmd= -p 2>/dev/null; done | sort -u | cut -c1-120 | sed 's/^/    /' >&2
exit 1
EOF
}

image_server_pid=""
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$image_server_pid" ]] && kill -0 "$image_server_pid" 2>/dev/null; then
    kill -TERM "$image_server_pid" 2>/dev/null
    wait "$image_server_pid" 2>/dev/null || true
  fi
  stop_remote_relay
}
trap cleanup EXIT
trap 'echo "[real_ffs_depth] stop requested"; cleanup; exit 130' INT TERM

if [[ -n "$RELAY_HOST" ]]; then
  start_remote_relay
else
  echo "[real_ffs_depth] HOLOSOMA_RELAY_HOST is empty: assuming the relay is already running at ${HOLOSOMA_REMOTE_STEREO_CONNECT}"
fi

echo "[real_ffs_depth] stereo source=${HOLOSOMA_REMOTE_STEREO_CONNECT}  predictor=ffs  model=${HOLOSOMA_FFS_MODEL}"
source scripts/source_inference_setup.sh
PYTHONPATH=src/holosoma${PYTHONPATH:+:${PYTHONPATH}} \
python src/holosoma/holosoma/sensors/image_server.py remote_ffs_d435i \
  --image-saver-config.image-root-dir "${log_dir}/depth_images" &
image_server_pid=$!
wait "$image_server_pid"
