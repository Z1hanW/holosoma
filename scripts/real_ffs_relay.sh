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

echo "[real_ffs_relay] bind=${BIND} emitter=${EMITTER}"
exec "$PY" -u "${HERE}/stereo_relay_pub.py" --bind "$BIND" --emitter "$EMITTER" --stats-interval 5
