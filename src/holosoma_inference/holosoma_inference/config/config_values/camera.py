"""Default camera configurations for holosoma_inference.

This module provides pre-configured camera setups for different
robot types and tasks, migrated from the original YAML configurations.
"""

from __future__ import annotations

from holosoma_inference.config.config_types.camera import CameraConfig, CameraPose, CameraProps

# =============================================================================
# Camera Configurations
# =============================================================================

# Default dual depth camera setup (migrated from camera_old.yaml)
dual_depth_cameras = CameraConfig(
    poses={
        "cam_front_depth": CameraPose(
            parent_link="robot/torso_link",
            camera_offset=(0.1, 0.0, 0.1),  # x, y, z [m]
            camera_rotation=(0.0, 75.0, 0.0),  # roll, pitch, yaw [deg]
        ),
        "cam_back_depth": CameraPose(
            parent_link="robot/torso_link",
            camera_offset=(-0.1, 0.0, 0.1),  # x, y, z [m]
            camera_rotation=(0.0, 75.0, 180.0),  # roll, pitch, yaw [deg]
        ),
    },
    props=CameraProps(
        image_type="depth",
        width=240,
        height=135,
        resized_width=48,
        resized_height=27,
        # ZED 2i - HD@720p FOV settings
        horizontal_fov=101.41,
        vertical_fov=69.00,
        # Alternative ZED 2i - HD@1080p FOV settings:
        # horizontal_fov=84.38,
        # vertical_fov=54.03,
        near_clip=0.1,
        far_clip=2.0,
        frame_rate=10,
        image_show=False,
        depth_delay=1,  # total delay = depth_delay / frame_rate [s]
    ),
)


# =============================================================================
# Default Configurations Dictionary
# =============================================================================

DEFAULTS = {
    "dual-depth": dual_depth_cameras,
}
"""Dictionary of all available camera configurations.

Keys use hyphen-case naming convention for CLI compatibility.
"""
