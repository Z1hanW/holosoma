#!/bin/bash
# CI runs this inside holosoma docker
set -ex

source /workspace/holosoma/scripts/source_isaacsim_setup.sh
python -m pip install -e /workspace/holosoma/src/holosoma[unitree]
python -m pip install -e /workspace/holosoma/src/holosoma[booster]
python -m pip install -e /workspace/holosoma/src/holosoma_inference

cd /workspace/holosoma
python -m pytest -s -m "isaacsim" --ignore=holosoma/holosoma/envs/legged_base_task/tests/ --ignore=thirdparty
