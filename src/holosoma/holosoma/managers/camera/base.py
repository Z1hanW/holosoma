"""Base class for camera terms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from holosoma.config_types.camera import CameraTermCfg


class CameraTermBase(ABC):
    """Base class for stateful camera terms."""

    def __init__(self, cfg: CameraTermCfg, env: Any):
        """Initialize camera term.

        Parameters
        ----------
        cfg : CameraTermCfg
            Configuration for this camera term
        env : Any
            Environment instance (typically BaseTask or subclass)
        """
        self._cfg = cfg
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device

    @abstractmethod
    def setup(self) -> None:
        """Setup hook called once during environment initialization."""

    @abstractmethod
    def capture(self) -> None:
        """Capture camera outputs for the current simulation step."""

    @abstractmethod
    def draw_debug_viz(self) -> None:
        """Draw debug visualization."""