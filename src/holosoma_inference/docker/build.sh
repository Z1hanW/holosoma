#!/bin/bash

# Build the Docker image using the holosoma directory as context

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # holosoma/src/holosoma_inference/docker
SRC_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")" # holosoma/src

IMAGE_NAME="holosoma-onboard"

cmd="docker build $SRC_DIR -f "$SCRIPT_DIR/Dockerfile" -t $IMAGE_NAME"
echo $cmd
eval $cmd

rm "$SCRIPT_DIR"/*.whl
