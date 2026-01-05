"""Perception configuration presets."""

from holosoma.config_types.perception import PerceptionConfig

none = PerceptionConfig(enabled=False)

heightmap = PerceptionConfig(
    enabled=True,
    output_mode="heightmap",
    grid_size=11,
    grid_interval=0.1,
    update_hz=50.0,
    encoder_output_dim=512,
)

DEFAULTS = {
    "none": none,
    "heightmap": heightmap,
}

__all__ = ["none", "heightmap", "DEFAULTS"]
