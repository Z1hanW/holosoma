#!/bin/bash
# CI runs this inside holosoma docker
set -ex

source /workspace/holosoma/scripts/source_isaacgym_setup.sh
pip install -e /workspace/holosoma/src/holosoma[unitree]
pip install -e /workspace/holosoma/src/holosoma[booster]
pip install -e /workspace/holosoma/src/holosoma_inference

cd /workspace/holosoma
pytest -s --ignore=thirdparty -m "not isaacsim"
