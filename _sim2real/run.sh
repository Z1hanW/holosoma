#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_registry.sh"

usage() {
  cat <<EOF
Usage:
  UNITREE_INTERFACE=<iface> bash _sim2real/run.sh <policy> [extra run_policy args...]

Registered policies:
$(sim2real_known_policies | sed 's/^/  /')
EOF
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

POLICY_NAME="${1:-w5qostjn}"
if [[ "$POLICY_NAME" == "current" || "$POLICY_NAME" == "default" ]]; then
  POLICY_NAME="w5qostjn"
fi
if [[ $# -gt 0 ]]; then
  shift
fi

if ! sim2real_load_policy "$POLICY_NAME"; then
  echo "[ERROR] Unknown sim2real policy: $POLICY_NAME" >&2
  echo "[INFO] Registered policies:" >&2
  sim2real_known_policies | sed 's/^/  /' >&2
  exit 2
fi

exec bash "$SCRIPT_DIR/_common_run_policy.sh" "$@"
