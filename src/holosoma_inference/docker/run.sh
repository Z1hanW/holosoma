#!/bin/bash

# Parse command line arguments
NEW_CONTAINER=false
if [ "$1" == "--new" ]; then
    NEW_CONTAINER=true
fi

# Get the project root directory dynamically
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # FAR-FALCON/holosoma_inference/docker
PARENT_DIR="$(dirname "$SCRIPT_DIR")" # FAR-FALCON/holosoma_inference
PROJECT_ROOT="$(dirname "$PARENT_DIR")" # FAR-FALCON
CONTAINER_NAME="far-jetson-container"
IMAGE_NAME="241533154612.dkr.ecr.us-east-1.amazonaws.com/humanoid-onboard"

# Function to create a new container
create_container() {
    docker run -it \
        --privileged \
        --name ${CONTAINER_NAME} \
        --network host \
        -e DISPLAY=$DISPLAY \
        -e XAUTHORITY=/root/.Xauthority \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v ~/cyclonedds_ws/:/workspace/cyclonedds_ws \
        -v $HOME/.Xauthority:/root/.Xauthority:ro \
        -v "$PROJECT_ROOT":/workspace/FAR-FALCON \
        -w /workspace/FAR-FALCON \
        "$IMAGE_NAME"
}

# Try to set xhost permissions if display is available
if [ -n "$DISPLAY" ]; then
    xhost +local:docker 2>/dev/null || echo "Warning: Could not set xhost permissions (no display available)"
fi

# If --new flag is set, stop and remove existing container
if [ "$NEW_CONTAINER" = true ]; then
    if docker ps -a --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo "Removing existing container ${CONTAINER_NAME}..."
        docker stop ${CONTAINER_NAME} 2>/dev/null
        docker rm ${CONTAINER_NAME}
    fi
    echo "Creating new container..."
    create_container
else
    # Check if container exists
    if docker ps -a --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container ${CONTAINER_NAME} already exists."

        # Check if container is running
        if docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
            echo "Container is running. Attaching to it..."
            docker exec -it -w /workspace/FAR-FALCON ${CONTAINER_NAME} /bin/bash

        else
            echo "Container is stopped. Starting and attaching to it..."
            docker start ${CONTAINER_NAME}
            docker exec -it -w /workspace/FAR-FALCON ${CONTAINER_NAME} /bin/bash
        fi
    else
        echo "Container ${CONTAINER_NAME} does not exist. Creating new container..."
        create_container
    fi
fi