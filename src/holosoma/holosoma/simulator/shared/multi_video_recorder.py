from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from loguru import logger

from holosoma.config_types.video import CartesianCameraConfig, FixedCameraConfig, SphericalCameraConfig, VideoConfig


class MultiVideoRecorder:
    """Thin wrapper that fans out video events to multiple recorders."""

    def __init__(self, recorders: list[Any]) -> None:
        self.recorders = recorders

    @property
    def enabled(self) -> bool:
        return any(bool(getattr(recorder, "enabled", False)) for recorder in self.recorders)

    @property
    def is_recording(self) -> bool:
        return any(bool(getattr(recorder, "is_recording", False)) for recorder in self.recorders)

    @property
    def current_episode(self) -> int:
        episodes = [int(getattr(recorder, "current_episode", 0)) for recorder in self.recorders]
        return max(episodes) if episodes else 0

    def setup_recording(self) -> None:
        for recorder in self.recorders:
            recorder.setup_recording()

    def capture_frame(self, env_id: int | None = None) -> None:
        for recorder in self.recorders:
            target_env_id = int(getattr(recorder.config, "record_env_id", 0))
            recorder.capture_frame(target_env_id)

    def start_recording(self, episode_id: int) -> None:
        for recorder in self.recorders:
            recorder.start_recording(episode_id)

    def stop_recording(self) -> None:
        for recorder in self.recorders:
            recorder.stop_recording()

    def on_episode_start(self, env_id: int) -> None:
        for recorder in self.recorders:
            recorder.on_episode_start(env_id)

    def on_episode_end(self, env_id: int) -> None:
        for recorder in self.recorders:
            recorder.on_episode_end(env_id)

    def cleanup(self) -> None:
        for recorder in self.recorders:
            recorder.cleanup()


def _camera_from_spec(spec: dict[str, Any], default_camera: Any) -> Any:
    camera_spec = spec.get("camera")
    if camera_spec is None:
        return default_camera
    if not isinstance(camera_spec, dict):
        raise TypeError(f"Expected 'camera' to be an object, got {type(camera_spec)!r}.")

    camera_type = str(camera_spec.get("type", getattr(default_camera, "type", "cartesian"))).lower()
    if camera_type == "cartesian":
        return CartesianCameraConfig(
            offset=list(camera_spec.get("offset", getattr(default_camera, "offset", [2.0, 2.0, 1.0]))),
            target_offset=list(camera_spec.get("target_offset", getattr(default_camera, "target_offset", [0.0, 0.0, 0.3]))),
        )
    if camera_type == "spherical":
        return SphericalCameraConfig(
            distance=float(camera_spec.get("distance", getattr(default_camera, "distance", 3.0))),
            azimuth=float(camera_spec.get("azimuth", getattr(default_camera, "azimuth", 90.0))),
            elevation=float(camera_spec.get("elevation", getattr(default_camera, "elevation", 20.0))),
        )
    if camera_type == "fixed":
        return FixedCameraConfig(
            position=list(camera_spec.get("position", getattr(default_camera, "position", [5.0, 5.0, 3.0]))),
            target=list(camera_spec.get("target", getattr(default_camera, "target", [0.0, 0.0, 1.0]))),
        )
    raise ValueError(f"Unsupported camera type in multi-view spec: {camera_type}")


def build_multi_video_recorder_from_spec(
    base_config: VideoConfig,
    simulator: Any,
    multi_view_spec_json: str | None,
) -> Any:
    """Build either a single recorder or a multi-recorder wrapper from a JSON spec."""
    from holosoma.simulator.isaacsim.video_recorder import IsaacSimVideoRecorder

    if not multi_view_spec_json:
        return IsaacSimVideoRecorder(base_config, simulator)

    try:
        payload = json.loads(multi_view_spec_json)
    except json.JSONDecodeError as exc:
        logger.warning(f"Invalid HOLOSOMA_VIDEO_MULTI_VIEWS_JSON; falling back to single recorder: {exc}")
        return IsaacSimVideoRecorder(base_config, simulator)

    if isinstance(payload, dict):
        views = payload.get("views", [])
    else:
        views = payload

    if not isinstance(views, list) or not views:
        logger.warning("HOLOSOMA_VIDEO_MULTI_VIEWS_JSON did not contain a non-empty list; falling back to single recorder.")
        return IsaacSimVideoRecorder(base_config, simulator)

    recorders = []
    for idx, spec in enumerate(views):
        if not isinstance(spec, dict):
            raise TypeError(f"Multi-view spec entry {idx} must be an object, got {type(spec)!r}.")

        record_env_id = int(spec.get("record_env_id", base_config.record_env_id))
        view_save_dir = spec.get("save_dir", base_config.save_dir)
        view_upload_to_wandb = bool(spec.get("upload_to_wandb", base_config.upload_to_wandb))
        view_keep_local_copy = bool(spec.get("keep_local_copy", base_config.keep_local_copy))
        view_show_overlay = bool(spec.get("show_command_overlay", base_config.show_command_overlay))
        view_enabled = bool(spec.get("enabled", base_config.enabled))
        view_interval = int(spec.get("interval", base_config.interval))
        view_playback_rate = float(spec.get("playback_rate", base_config.playback_rate))
        view_camera_smoothing = float(spec.get("camera_smoothing", base_config.camera_smoothing))
        view_output_format = str(spec.get("output_format", base_config.output_format))
        view_vertical_fov = float(spec.get("vertical_fov", base_config.vertical_fov))
        view_wandb_key = str(spec.get("wandb_key", f"{base_config.wandb_key}/env_{record_env_id}"))

        camera = _camera_from_spec(spec, base_config.camera)
        cfg = replace(
            base_config,
            enabled=view_enabled,
            interval=view_interval,
            playback_rate=view_playback_rate,
            camera_smoothing=view_camera_smoothing,
            output_format=view_output_format,
            save_dir=view_save_dir,
            upload_to_wandb=view_upload_to_wandb,
            keep_local_copy=view_keep_local_copy,
            wandb_key=view_wandb_key,
            show_command_overlay=view_show_overlay,
            record_env_id=record_env_id,
            camera=camera,
            vertical_fov=view_vertical_fov,
        )
        recorder = IsaacSimVideoRecorder(cfg, simulator)
        recorders.append(recorder)

        logger.info(
            "Configured multi video view {}: env_id={}, save_dir={}, camera_type={}, upload_to_wandb={}, wandb_key={}",
            idx,
            record_env_id,
            view_save_dir,
            getattr(camera, "type", type(camera).__name__),
            view_upload_to_wandb,
            view_wandb_key,
        )

    if len(recorders) == 1:
        return recorders[0]
    return MultiVideoRecorder(recorders)
