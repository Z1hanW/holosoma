#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "${SCRIPT_DIR}/src/holosoma_retargeting/retgt_behave.sh"
