#!/bin/bash

# Build the Docker image using the FAR-FALCON directory as context

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # FAR-FALCON/holosoma_inference/docker
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")" # FAR-FALCON

ECR_REPO="241533154612.dkr.ecr.us-east-1.amazonaws.com"

cmd="docker build "$ROOT_DIR" -f "$SCRIPT_DIR/Dockerfile" -t "$ECR_REPO/humanoid-onboard""
echo $cmd
eval $cmd

rm $SCRIPT_DIR/*.whl
