"""Default camera manager configurations."""

from holosoma.config_types.camera import CameraManagerCfg
from holosoma.config_values.loco.g1.camera import dual_depth_cameras

none = CameraManagerCfg()

DEFAULTS = {
    "none": none,
    "dual_depth_cameras": dual_depth_cameras,
}

__all__ = ["dual_depth_cameras"]