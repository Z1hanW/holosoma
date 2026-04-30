#!/usr/bin/env bash
set -euo pipefail

SRC="/nfs/zzzihanw/omomo_45"
DST="/home/ubuntu/FAR/holosoma/data/ds_as_data/omomo"

if [[ ! -d "$SRC" ]]; then
  echo "Source directory not found: $SRC" >&2
  exit 1
fi

mkdir -p "$DST"

if command -v rsync >/dev/null 2>&1; then
  rsync -a --info=progress2 "$SRC"/ "$DST"/
else
  cp -a "$SRC"/. "$DST"/
fi

echo "Copied $SRC -> $DST"
