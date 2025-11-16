#!/bin/bash

# Build the Docker image using the holosoma directory as context

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # holosoma/src/holosoma_inference/docker
SRC_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")" # holosoma/src

ECR_REPO="241533154612.dkr.ecr.us-east-1.amazonaws.com"

cmd="docker build "$SRC_DIR" -f "$SCRIPT_DIR/Dockerfile" -t "$ECR_REPO/humanoid-onboard""
echo $cmd
eval $cmd

rm $SCRIPT_DIR/*.whl
