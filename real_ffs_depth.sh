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
export HOLOSOMA_REMOTE_STEREO_CONNECT="${HOLOSOMA_REMOTE_STEREO_CONNECT:-tcp://${RELAY_HOST:-192.168.123.164}:${RELAY_PORT}}"

for f in "$HOLOSOMA_FFS_MODEL" "$(dirname "$HOLOSOMA_FFS_MODEL")/cfg.yaml"; do
  if [[ ! -f "$f" ]]; then
    echo "[real_ffs_depth] ERROR: missing $f" >&2
    exit 1
  fi
done

log_dir="${ROOT_DIR}/logs/real_ffs_depth_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"
exec > >(tee -a "${log_dir}/depth.log") 2>&1
echo "[real_ffs_depth] log_dir=${log_dir}"

SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5)

stop_remote_relay() {
  [[ -n "$RELAY_HOST" ]] || return 0
  "${SSH[@]}" "$RELAY_HOST" "cd ${RELAY_DIR} 2>/dev/null && [ -f relay.pid ] && kill \"\$(cat relay.pid)\" 2>/dev/null; rm -f ${RELAY_DIR}/relay.pid" \
    && echo "[real_ffs_depth] remote relay stopped" \
    || echo "[real_ffs_depth] WARNING: could not stop remote relay on ${RELAY_HOST}; check ${RELAY_DIR}/relay.pid there" >&2
}

start_remote_relay() {
  echo "[real_ffs_depth] starting stereo relay on ${RELAY_HOST} (${RELAY_DIR}/real_ffs_relay.sh)"
  # Kill a previous relay via its pidfile, launch detached from this ssh session,
  # then block until it reports it is publishing so image_server never has to wait.
  "${SSH[@]}" "$RELAY_HOST" "bash -s" <<EOF
set -e
cd ${RELAY_DIR}
[ -f relay.pid ] && kill "\$(cat relay.pid)" 2>/dev/null && sleep 0.5
rm -f relay.log
HOLOSOMA_RELAY_BIND='tcp://*:${RELAY_PORT}' setsid nohup ./real_ffs_relay.sh > relay.log 2>&1 < /dev/null &
echo \$! > relay.pid
for i in \$(seq 1 60); do
  grep -q publishing relay.log 2>/dev/null && { echo "  relay pid \$(cat relay.pid): \$(grep -m1 publishing relay.log)"; exit 0; }
  kill -0 "\$(cat relay.pid)" 2>/dev/null || break
  sleep 0.25
done
echo "  relay did not start; remote log:" >&2; tail -8 relay.log >&2; exit 1
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
