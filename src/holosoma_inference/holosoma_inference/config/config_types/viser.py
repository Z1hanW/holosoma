"""Viser visualization configuration for holosoma_inference."""

from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ViserConfig:
    """Configuration for optional Viser visualization during inference."""

    enabled: bool = False
    """Enable the Viser viewer."""

    port: int = 6060
    """Port to host the Viser server."""

    update_interval: int = 1
    """Update Viser every N policy steps."""

    show_meshes: bool = True
    """Show visual meshes in the Viser viewer."""

    add_grid: bool = True
    """Add a ground grid to the Viser scene."""

    grid_size: float = 10.0
    """Size of the ground grid."""

    urdf_path: str | None = None
    """Optional override for the robot URDF path used by Viser."""
