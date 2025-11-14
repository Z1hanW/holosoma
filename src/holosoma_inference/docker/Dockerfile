FROM ros:humble

ENV ROS_DISTRO humble
WORKDIR /workspace/

# Install ohmybash
RUN bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh)"

# Install dependencies + dev tools
RUN apt-get update && apt-get install -y python3-pip vim
RUN pip3 install onnxruntime

# Build far_ws workspace
COPY far_ws /workspace/FAR-FALCON/far_ws
WORKDIR /workspace/FAR-FALCON/far_ws
RUN bash build.sh

# Install Sensors: Realsense SDK
# Install librealsense dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential pkg-config libssl-dev libusb-1.0-0-dev udev \
    python3-pip python3-colcon-common-extensions python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# Build librealsense.
# We use the RSUSB (libuvc) backend, rather than kernel patches, because the script can take ~30mins to generate.
RUN git clone --depth=1 https://github.com/IntelRealSense/librealsense.git /opt/librealsense \
    && mkdir -p /opt/librealsense/build \
    && cd /opt/librealsense/build \
    && cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF -DFORCE_RSUSB_BACKEND=ON \
    && make -j"$(nproc)" && make install

# Install ros2 realsense package
RUN apt-get update && apt-get install -y --no-install-recommends ros-${ROS_DISTRO}-realsense2-camera ros-${ROS_DISTRO}-realsense2-description 

ENV LD_LIBRARY_PATH="/usr/lib/python3.10/dist-packages/lib/"

# Auto-source ROS environment on shell login
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /workspace/FAR-FALCON/far_ws/install/local_setup.bash" >> /root/.bashrc

# Install holosoma_inference (as the last step, since the cache of this layer will get invalidated frequently)
COPY holosoma_inference /workspace/FAR-FALCON/holosoma_inference/
RUN pip3 install -e "/workspace/FAR-FALCON/holosoma_inference[unitree]"
RUN pip3 install -e "/workspace/FAR-FALCON/holosoma_inference[booster]"
RUN pip3 install pin

WORKDIR /workspace/FAR-FALCON
