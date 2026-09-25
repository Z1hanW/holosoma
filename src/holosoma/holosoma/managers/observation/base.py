"""Base classes and protocols for observation terms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.observation import ObsTermCfg


_ObservationCallable = TypeVar("_ObservationCallable", bound=Callable[..., torch.Tensor])
_REUSABLE_BASE_TERM_ATTRIBUTE = "__holosoma_reusable_observation_base_term__"


def reusable_observation_base_term(func: _ObservationCallable) -> _ObservationCallable:
    """Mark a function as safe for exact, same-``compute`` base-result reuse.

    The function must be deterministic for the current environment state, must
    not consume RNG, and must not mutate either the environment or returned
    tensor.  The observation manager additionally requires empty term params
    and an explicit manager-level opt-in before it uses this declaration.
    Stateful :class:`ObservationTermBase` instances are never reusable.
    """

    setattr(func, _REUSABLE_BASE_TERM_ATTRIBUTE, True)
    return func


def is_reusable_observation_base_term(func: Callable[..., torch.Tensor]) -> bool:
    """Return whether ``func`` carries the explicit exact-reuse declaration."""

    return getattr(func, _REUSABLE_BASE_TERM_ATTRIBUTE, False) is True


class ObservationTermBase(ABC):
    """Base class for stateful observation terms.

    This class provides the interface for observation terms that need to maintain
    internal state (e.g., history buffers, filters). For simple stateless observations,
    use plain functions instead.

    Note: Currently not used in basic locomotion implementation, but provided for
    future extensibility.
    """

    def __init__(self, cfg: ObsTermCfg, env: Any):
        """Initialize observation term.

        Args:
            cfg: Configuration for this observation term
            env: Environment instance (typically a ``BaseTask`` subclass)
        """
        self.cfg = cfg
        self.env = env

    @abstractmethod
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal state for specified environments.

        Args:
            env_ids: Environment IDs to reset. If None, reset all environments.
        """

    @abstractmethod
    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        """Compute observation.

        Args:
            env: Environment instance
            **kwargs: Additional parameters from config

        Returns:
            Observation tensor of shape [num_envs, obs_dim]
        """
