"""Locomotion camera presets for the G1 robot."""
from __future__ import annotations
from holosoma.config_types.camera import CameraManagerCfg, CameraPose, CameraProps, CameraTermCfg

depth_camera_props=CameraProps(
        image_type="depth",
        width=1280,
        height=720,

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
        image_show=False,

        frame_rate=10,
        depth_delay=1,  # total delay = depth_delay / frame_rate [s]
    )


dual_depth_cameras = CameraManagerCfg(terms={
    "front_depth": CameraTermCfg(
        func="holosoma.managers.camera.terms.depth:DepthCamera",
        params={
            "pose": CameraPose(
                camera_body_link="torso_link",
                camera_offset=(0.1, 0.0, 0.1),  # x, y, z [m]
                camera_rotation=(0.0, 75.0, 0.0),  # roll, pitch, yaw [deg]
            ),
            "props": depth_camera_props,
        },
    ),
    "back_depth": CameraTermCfg(
        func="holosoma.managers.camera.terms.depth:DepthCamera",
        params={
            "props": depth_camera_props,
            "pose": CameraPose(
                camera_body_link="torso_link",
                camera_offset=(-0.1, 0.0, 0.1),  # x, y, z [m]
                camera_rotation=(0.0, 75.0, 180.0),  # roll, pitch, yaw [deg]
            ),
        },
    ),
})

__all__ = ["dual_depth_cameras"]
