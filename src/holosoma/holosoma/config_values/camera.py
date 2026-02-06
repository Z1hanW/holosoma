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
    horizontal_fov=86.0,
    vertical_fov=86.0 * (60 / 106),
    near_clip=0.15,
    far_clip=3.0,
    image_show=False,
    frame_rate=50,
    depth_delay=0,
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