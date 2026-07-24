"""Observation configuration types for holosoma_inference."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ObservationTermDescriptor:
    """Canonical training-time semantics for one deployed observation term.

    Deployment intentionally does not reproduce observation noise.  ``noise``
    records the distribution used while training so metadata-backed policies
    can still be authenticated against the exact training contract.
    """

    func: str
    """Fully qualified training observation function path."""

    params: dict[str, Any] = field(default_factory=dict)
    """Serialized keyword arguments passed to the training observation term."""

    noise: float = 0.0
    """Training-time noise magnitude; inference itself remains deterministic."""

    clip: tuple[float, float] | None = None
    """Per-term post-scale clip bounds used during training."""


@dataclass(frozen=True)
class ObservationConfig:
    """Observation space configuration.

    Defines which observations are used, their dimensions,
    scaling factors, history lengths, etc. for policy inference.
    """

    obs_dict: dict[str, list[str]]
    """Maps observation group names to lists of observation components.

    Each key represents an observation group (e.g., "actor_obs", "critic_obs"),
    and each value is a list of observation component names that belong to that group.

    Example:
        {"actor_obs": ["base_ang_vel", "projected_gravity", "dof_pos"]}
    """

    obs_dims: dict[str, int]
    """Dimension of each observation component.

    Maps each observation component name to its dimensionality.

    Example:
        {"base_ang_vel": 3, "dof_pos": 29, "actions": 29}
    """

    obs_scales: dict[str, float]
    """Scaling factor applied to each observation component.

    Maps each observation component name to a scaling factor that will be
    multiplied with the raw observation values during preprocessing.

    Example:
        {"base_ang_vel": 0.25, "dof_vel": 0.05, "projected_gravity": 1.0}
    """

    history_length_dict: dict[str, int]
    """Number of timesteps to keep in history for each observation group.

    Maps each observation group name to the number of historical timesteps
    to maintain. A value of 1 means only the current observation is used.

    Example:
        {"actor_obs": 1, "critic_obs": 3}
    """

    clip_observations: float = 100.0
    """Global post-scale observation clip, matching the training environment."""

    term_descriptors: dict[str, ObservationTermDescriptor] = field(default_factory=dict)
    """Canonical semantic descriptors for every deployed observation term."""

    group_concatenate: dict[str, bool] = field(default_factory=dict)
    """Expected training ``concatenate`` setting for each deployed actor group."""

    group_enable_noise: dict[str, bool] = field(default_factory=dict)
    """Expected training ``enable_noise`` setting for each deployed actor group."""
