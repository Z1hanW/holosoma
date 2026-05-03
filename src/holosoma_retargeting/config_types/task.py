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
    # Optional explicit MuJoCo scene XML override (useful for robot-w-obj scenes)
    scene_xml_file: Path | None = None
    # Optional object-geometry name token used by contact filtering
    # (e.g., "obj" when scene geoms are named generically instead of per-object names)
    object_contact_name: str | None = None
    # Optional object mesh root (e.g., BEHAVE objects root)
    object_mesh_root: Path | None = None
    # Optional object mesh filename suffix (e.g., "_sq.obj" or "_f1000.ply")
    object_mesh_suffix: str | None = "_f1000.ply"
    # Retained for CLI/config compatibility. object_interaction now follows
    # holosoma_gt scale handling and keeps object assets at native scale.
    object_interaction_scale_augmented: tuple[float, float, float] | None = None
