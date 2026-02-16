"""Default image server configurations."""

import dataclasses

from holosoma.config_types.image_server import ImageServerConfig, ImageSaverConfig, ImageVisualizerConfig

# Default for sim2sim MuJoCo image streaming.
base = ImageServerConfig()
base_saver = ImageSaverConfig()
base_visualizer = ImageVisualizerConfig()

mujoco = dataclasses.replace(
    base,
    enable_gum_depth_prediction=False,
    enable_camera_depth_prediction=True,
    depth_source="depth",
    save_images=True,
)

# MuJoCo sim2sim preset for single D435i depth camera (far-tracking distillation).
# Note: near_clip, far_clip, resized_height, resized_width, crop_*, and frame_rate
# are auto-synced from CameraProps by sim_utils._sync_image_server_config().
mujoco_d435i = dataclasses.replace(
    base,
    enable_gum_depth_prediction=False,
    enable_camera_depth_prediction=True,
    depth_source="depth",
    save_images=False,
    visualize_images=True,
    num_delay_frames=7,
)

# Debug-friendly ZED profile with visualization and both depth sources enabled.
real_verbose = dataclasses.replace(
    base,
    enable_gum_depth_prediction=True,
    enable_camera_depth_prediction=True,
    depth_source="depth",
    visualize_images=True,
    save_images=True,
)

# Default ZED inference profile.
real = dataclasses.replace(
    base,
    enable_gum_depth_prediction=False,
    enable_camera_depth_prediction=True,
    depth_source="depth",
    visualize_images=False,
)

# Add gum depth prediction, 
real_enable_gum = dataclasses.replace(real, 
    enable_gum_depth_prediction=True,
    depth_source="depth",
    visualize_images=False,
)
 
# Add gum depth prediction and use ZED depth as the source for policy.
real_depth_gum = dataclasses.replace(real, 
    enable_gum_depth_prediction=True,
    depth_source="depth_gum",
    visualize_images=False,
)


# MuJoCo sim2sim D435i with GUM depth prediction as the policy depth source.
mujoco_depth_gum_d435i = dataclasses.replace(
    mujoco_d435i,
    enable_gum_depth_prediction=True,
    depth_source="depth_gum",
    visualize_images=True,
    save_images=False,
)

# Real-robot D435i.
real_d435i = dataclasses.replace(
    base,
    enable_gum_depth_prediction=False,
    enable_camera_depth_prediction=True,
    depth_source="depth",
    near_clip=0.3,
    far_clip=3.0,
    resized_height=58,
    resized_width=87,
    frame_rate=50,
    crop_y_start=16,
    crop_x_start=32,
    crop_x_end=-32,
    save_images=False,
    visualize_images=True,
    camera_type="realsense",
)

# Debug-friendly D435i profile with visualization and both depth sources enabled.
real_verbose_d435i = dataclasses.replace(
    real_d435i,
    enable_gum_depth_prediction=True,
    visualize_images=True,
    save_images=True,
)

# D435i with GUM depth prediction enabled (policy still uses camera depth).
real_enable_gum_d435i = dataclasses.replace(
    real_d435i,
    enable_gum_depth_prediction=True,
    visualize_images=True,
)

# D435i with GUM depth prediction as the policy depth source.
real_depth_gum_d435i = dataclasses.replace(
    real_d435i,
    enable_gum_depth_prediction=True,
    depth_source="depth_gum",
    visualize_images=False,
)

DEFAULTS = {
    "mujoco": mujoco,
    "mujoco_d435i": mujoco_d435i,
    "mujoco_depth_gum_d435i": mujoco_depth_gum_d435i,
    "real_verbose": real_verbose,
    "real": real,
    "real_enable_gum": real_enable_gum,
    "real_depth_gum": real_depth_gum,
    "real_d435i": real_d435i,
    "real_verbose_d435i": real_verbose_d435i,
    "real_enable_gum_d435i": real_enable_gum_d435i,
    "real_depth_gum_d435i": real_depth_gum_d435i,
}
