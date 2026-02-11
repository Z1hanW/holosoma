#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"${SCRIPT_DIR}/src/holosoma_retargeting/examples/retgt_behave_zup.sh"
