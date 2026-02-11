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
    enable_zed_depth_prediction=True,
    depth_source="depth",
    save_images=True,
)

# Debug-friendly ZED profile with visualization and both depth sources enabled.
real_verbose = dataclasses.replace(
    base,
    enable_gum_depth_prediction=True,
    enable_zed_depth_prediction=True,
    depth_source="depth",
    visualize_images=True,
    save_images=True,
)

# Default ZED inference profile.
real = dataclasses.replace(
    base,
    enable_gum_depth_prediction=False,
    enable_zed_depth_prediction=True,
    depth_source="depth",
    visualize_images=False,
)

# Add gum depth prediction, 
real_enable_gum = dataclasses.replace(real, 
    enable_gum_depth_prediction=True,
    visualize_images=False,
)
 
# Add gum depth prediction and use ZED depth as the source for policy.
real_depth_gum = dataclasses.replace(real, 
    enable_gum_depth_prediction=True,
    depth_source="depth_gum",
    visualize_images=False,
)


DEFAULTS = {
    "mujoco": mujoco,
    "real_verbose": real_verbose,
    "real": real,
    "real_enable_gum": real_enable_gum,
    "real_depth_gum": real_depth_gum,
}
