"""Default camera manager configurations."""

from holosoma.config_types.camera import CameraManagerCfg, CameraPose, CameraProps, CameraTermCfg
from holosoma.config_values.loco.g1.camera import dual_depth_cameras

none = CameraManagerCfg()

# D435i depth camera props for far-tracking distillation
d435i_depth_props = CameraProps(
    image_type="depth",
    width=848,
    height=480,
    resized_width=87,
    resized_height=58,
    horizontal_fov=89.5,
    vertical_fov=89.5 * (60 / 106),
    near_clip=0.3,
    far_clip=3.0,
    image_show=True,
    frame_rate=50,
    depth_delay=0,
    crop_y_start=16,
    crop_x_start=32,
    crop_x_end=-32,
    latency_frame=(7, 8),
    buffer_len=9,
)

single_d435i_depth = CameraManagerCfg(terms={
    "d435i_depth": CameraTermCfg(
        func="holosoma.managers.camera.terms.depth:DepthCamera",
        params={
            "pose": CameraPose(
                camera_body_link="torso_link",
                camera_offset=(0.01, 0.01, 0.44),
                camera_rotation=(1.0, 27.0, 1.0),
            ),
            "props": d435i_depth_props,
        },
    ),
})

DEFAULTS = {
    "none": none,
    "dual_depth_cameras": dual_depth_cameras,
    "single_d435i_depth": single_d435i_depth,
}

__all__ = ["dual_depth_cameras", "single_d435i_depth"]