"""Intel RealSense D435i camera wrapper for real-robot depth capture.

Follows the same interface as ZedCamerasWrapper so it can be used
interchangeably with ImageServer.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RealSenseCameraConfig:
    """Configuration for a single RealSense camera."""

    serial_number: str = ""
    """Camera serial number. Empty string means use the first available device."""

    img_shape: tuple[int, int] = (480, 848)
    """Image shape as (height, width)."""

    fps: int = 30
    """Capture frame rate for both depth and color streams."""

    enable_color: bool = True
    """Enable the color (RGB) stream."""

    align_depth_to_color: bool = True
    """Align depth frames to the color sensor viewport."""

    positive_inf_depth_value: float = 3.0
    """How +inf should be mapped in the depth image, in meters."""

    negative_inf_depth_value: float = 0.1
    """How -inf should be mapped in the depth image, in meters."""

    nan_depth_value: float = 0.0
    """How nan / zero-depth should be mapped in the depth image, in meters."""


class RealSenseCamera:
    """Manages a single Intel RealSense D435i camera via pyrealsense2."""

    def __init__(self, config: RealSenseCameraConfig):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError(
                "pyrealsense2 is not available. "
                "Install it with: pip install pyrealsense2"
            ) from exc

        self.rs = rs
        self.config = config
        self._init_camera()

    def _init_camera(self):
        rs = self.rs
        height, width = self.config.img_shape

        self.pipeline = rs.pipeline()
        rs_config = rs.config()

        if self.config.serial_number:
            rs_config.enable_device(self.config.serial_number)

        # Enable depth stream
        rs_config.enable_stream(
            rs.stream.depth, width, height, rs.format.z16, self.config.fps,
        )

        # Enable color stream
        if self.config.enable_color:
            rs_config.enable_stream(
                rs.stream.color, width, height, rs.format.bgr8, self.config.fps,
            )

        profile = self.pipeline.start(rs_config)

        # Depth scale: converts raw uint16 depth values to meters
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Align object (reusable across frames)
        if self.config.align_depth_to_color and self.config.enable_color:
            self.align = rs.align(rs.stream.color)
        else:
            self.align = None

        print(
            f"[RealSense] Initialized: serial={self.config.serial_number or 'auto'}, "
            f"resolution={width}x{height}@{self.config.fps}fps, "
            f"depth_scale={self.depth_scale:.6f}"
        )

    def capture(self) -> dict:
        """Capture aligned depth and color frames.

        Returns
        -------
        dict
            ``{"depth": np.ndarray (H, W) float32 in meters,
               "rgb": np.ndarray (H, W, 3) uint8 BGR or None}``
        """
        frames = self.pipeline.wait_for_frames()

        if self.align is not None:
            frames = self.align.process(frames)

        # Depth
        depth_frame = frames.get_depth_frame()
        if depth_frame:
            depth_data = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
            # Replace zeros (invalid depth) with nan_depth_value
            depth_data[depth_data == 0.0] = self.config.nan_depth_value
            depth_data = np.nan_to_num(
                depth_data,
                nan=self.config.nan_depth_value,
                posinf=self.config.positive_inf_depth_value,
                neginf=self.config.negative_inf_depth_value,
            )
        else:
            depth_data = None

        # Color
        rgb_data = None
        if self.config.enable_color:
            color_frame = frames.get_color_frame()
            if color_frame:
                rgb_data = np.asanyarray(color_frame.get_data())  # BGR uint8

        return {"depth": depth_data, "rgb": rgb_data}

    def release(self):
        """Stop the RealSense pipeline."""
        self.pipeline.stop()
        print("[RealSense] Released")


# ───────────────────────────────────────────────────────
# RealSense Cameras Wrapper
# ───────────────────────────────────────────────────────

DEFAULT_REALSENSE_CAMERA_CONFIGS: dict[str, RealSenseCameraConfig] = {
    "d435i_depth": RealSenseCameraConfig(),
}


@dataclass(frozen=True)
class RealSenseCamerasConfig:
    """Configuration for one or more RealSense cameras."""

    terms: dict[str, RealSenseCameraConfig] = field(
        default_factory=lambda: DEFAULT_REALSENSE_CAMERA_CONFIGS.copy(),
    )


class RealSenseCamerasWrapper:
    """Wraps one or more RealSense cameras with the same interface as ZedCamerasWrapper."""

    def __init__(self, config: RealSenseCamerasConfig):
        self.cameras = {
            name: RealSenseCamera(cfg) for name, cfg in config.terms.items()
        }
        self.num_cameras = len(self.cameras)

    def get_frames(self) -> dict:
        """Capture frames from all cameras.

        Returns
        -------
        dict
            ``{"depth": {name: ndarray}, "rgb": {name: ndarray}}``
        """
        depth_data: dict[str, np.ndarray] = {}
        rgb_data: dict[str, np.ndarray] = {}
        for name, camera in self.cameras.items():
            frame = camera.capture()
            depth_data[name] = frame["depth"]
            rgb_data[name] = frame["rgb"]
        return {"depth": depth_data, "rgb": rgb_data}
