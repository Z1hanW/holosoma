"""Default camera manager configurations."""

from holosoma.config_types.camera import CameraManagerCfg, CameraPose, CameraProps, CameraTermCfg
from holosoma.config_values.loco.g1.camera import dual_depth_cameras

none = CameraManagerCfg()

# D435i depth camera props for far-tracking distillation
d435i_depth_props = CameraProps(
    image_type="depth",
    width=106,
    height=60,
    resized_width=87,
    resized_height=58,
    horizontal_fov=89.5,
    vertical_fov=58.6,
    near_clip=0.3,
    far_clip=3.0,
    image_show=True,
    frame_rate=30,
    depth_delay=0,
    crop_y_start=2,
    crop_x_start=4,
    crop_x_end=-4,
    latency_frame=0,
    buffer_len=3,
)

single_d435i_depth = CameraManagerCfg(terms={
    "d435i_depth": CameraTermCfg(
        func="holosoma.managers.camera.terms.depth:DepthCamera",
        params={
            "pose": CameraPose(
                camera_body_link="torso_link",
                camera_offset=(0.01, 0.01, 0.44),
                # Equivalent to far-tracking mount (1, 27, 1) plus camera_pitch_deg=10.
                camera_rotation=(1.21741572, 36.99830943, 1.11565009),
            ),
            "props": d435i_depth_props,
        },
    ),
})

# ZED 2i depth camera props for far-tracking distillation (G1FlatZed2iConfig)
# Native: 1280x720, warp raycast: 240x135, policy resize: (58, 87)
# Training uses "resize": (58, 87) in distillation_env_cfg.py WarpObservationsCfg
zed2i_depth_props = CameraProps(
    image_type="depth",
    width=1280,
    height=720,
    resized_width=87,
    resized_height=58,
    horizontal_fov=101.41,
    vertical_fov=101.41 * (135 / 240),
    near_clip=0.1,
    far_clip=2.0,
    image_show=False,
    frame_rate=10,
)

single_zed2i_depth = CameraManagerCfg(terms={
    "zed2i_depth": CameraTermCfg(
        func="holosoma.managers.camera.terms.depth:DepthCamera",
        params={
            "pose": CameraPose(
                camera_body_link="torso_link",
                camera_offset=(0.1, 0.0, 0.1),
                camera_rotation=(0.0, 75.0, 0.0),     # warp convention, same as training G1FlatZed2iConfig
            ),
            "props": zed2i_depth_props,
        },
    ),
})

DEFAULTS = {
    "none": none,
    "dual_depth_cameras": dual_depth_cameras,
    "single_d435i_depth": single_d435i_depth,
    "single_zed2i_depth": single_zed2i_depth,
}

__all__ = ["dual_depth_cameras", "single_d435i_depth", "single_zed2i_depth"]
