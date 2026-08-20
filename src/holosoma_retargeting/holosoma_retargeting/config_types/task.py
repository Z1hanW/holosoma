"""Configuration types for retargeting task settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskConfig:
    """Task-specific configuration parameters.

    These parameters control task-specific behavior like ground meshgrid generation,
    object sampling, augmentation, and scaling. Can be overridden via CLI.
    """

    # Object name
    # Auto-determined based on task_type if None: "largebox" for object_interaction,
    # "multi_boxes" for climbing, "ground" for robot_only
    object_name: str | None = None

    # Additional object-only geometry multiplier. For w-obj-scale, the effective
    # mesh/contact-point scale is smpl_scale * object_scale_multiplier; human
    # motion and the object center trajectory still use smpl_scale alone.
    object_scale_multiplier: float = 1.0

    # Ground meshgrid (robot_only task)
    ground_size: int = 15
    ground_range: tuple[float, float] = (-1.0, 1.0)

    # Climbing ground meshgrid (climbing task)
    climbing_ground_size: int = 8
    climbing_ground_range: tuple[float, float] = (-2.0, 2.0)

    # Surface weight parameters for climbing object sampling (climbing task)
    # Used in weighted_surface_sampling: points with z-coordinate > threshold get high weight
    # This biases sampling toward top surfaces (important for climbing contact points)
    surface_weight_threshold: float = 0.9  # z-coordinate threshold for high-weight points
    surface_weight_high: int = 20  # Weight for top surface points (z > threshold)
    surface_weight_low: int = 1  # Weight for other points

    # Object directory (for climbing tasks)
    # Auto-determined from data_path / task_name if None
    object_dir: Path | None = None

    # Optional fully assembled MuJoCo scene for object tasks that also contain
    # static geometry, such as CRISP terrain blocks.
    scene_xml_file: Path | None = None
