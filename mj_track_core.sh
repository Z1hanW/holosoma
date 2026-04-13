#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HOLOSOMA_MJ_TRACK_INTERNAL_CORE=1

exec "$ROOT_DIR/mj_track.sh" "$@"
