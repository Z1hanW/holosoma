#!/usr/bin/env bash
# Robot side of the FFS depth path. Runs on the Jetson the D435i is plugged into
# and streams the IR stereo pair + calibration to the laptop, where
# real_ffs_depth.sh runs Fast-FoundationStereo on it. Start this first.
#
# Keep the IR emitter ON: with it off both the D435i's own depth and FFS depth
# get markedly noisier (measured: FFS frame-to-frame jitter 3x, D435i 4x).
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${HOLOSOMA_RELAY_PYTHON:-$HOME/.holosoma_deps/miniconda3/envs/hsinference/bin/python}"
BIND="${HOLOSOMA_RELAY_BIND:-tcp://*:5602}"
EMITTER="${HOLOSOMA_RELAY_EMITTER:-on}"

if [[ ! -x "$PY" ]]; then
  echo "[real_ffs_relay] ERROR: python not found at $PY (set HOLOSOMA_RELAY_PYTHON)" >&2
  exit 1
fi
if ! lsusb 2>/dev/null | grep -q "8086:0b3a"; then
  echo "[real_ffs_relay] ERROR: no D435i (8086:0b3a) on the USB bus. Re-seat the cable." >&2
  exit 1
fi

# The D435i can only be streamed by one process. Another RealSense client (e.g. an
# autostarted stereo server) makes our pipeline start fine and then starve with
# "Frame didn't arrive", so refuse up front and say who has it rather than fight.
holders=""
for v in /dev/video*; do
  for p in $(fuser "$v" 2>/dev/null); do
    [[ "$p" == "$$" ]] && continue
    holders+="$(ps -o pid=,cmd= -p "$p" 2>/dev/null | cut -c1-110)"$'\n'
  done
done
holders="$(printf '%s' "$holders" | sort -u | sed '/^$/d')"
if [[ -n "$holders" ]]; then
  echo "[real_ffs_relay] ERROR: the camera is already in use by another process:" >&2
  printf '%s\n' "$holders" | sed 's/^/    /' >&2
  echo "[real_ffs_relay] Stop it first (or set HOLOSOMA_RELAY_IGNORE_HOLDERS=1 to try anyway)." >&2
  [[ "${HOLOSOMA_RELAY_IGNORE_HOLDERS:-0}" == "1" ]] || exit 1
fi

echo "[real_ffs_relay] bind=${BIND} emitter=${EMITTER}"
exec "$PY" -u "${HERE}/stereo_relay_pub.py" --bind "$BIND" --emitter "$EMITTER" --stats-interval 5
