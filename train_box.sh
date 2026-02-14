#!/usr/bin/env bash
set -euo pipefail

# Alias for box-object training workflow.
# Runs train_object_base.sh, which now does:
#   inspect kinematic motion -> close viewer -> start training with live Viser.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/train_object_base.sh" "$@"
