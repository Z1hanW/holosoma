"""Camera manager coordinating camera hooks."""

from __future__ import annotations

from typing import Any

from holosoma.config_types.camera import CameraManagerCfg
from holosoma.managers.utils import resolve_callable

from .base import CameraTermBase


class CameraManager:
    """Drive camera terms at setup and capture boundaries.

    Parameters
    ----------
    cfg : CameraManagerCfg
        Camera manager configuration specifying terms and parameters.
    env : Any
        Environment instance operated on by camera terms.
    device : str
        Device identifier used by camera terms.
    """

    def __init__(self, cfg: CameraManagerCfg, env: Any, device: str):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.logger = getattr(env, "logger", None)

        self.camera_terms: dict[str, CameraTermBase] = {}
        # for term_name, term_cfg in self.cfg.terms.items():
        #     self.camera_terms[term_name] = resolve_callable(
        #         term_cfg.func,
        #         context=f"camera term '{term_name}'",
        #     )(term_cfg, self.env)

    def setup(self) -> None:
        """Run setup hooks."""

        # TODO: get the entire scene.

        for term in self.camera_terms.values():
            term.setup()
    
    def step(self) -> None:
        """Run step hooks."""
        # camera manager will be called every env step, but one should be able to control
        for term in self.camera_terms.values():
            term.step()

    def capture(self) -> None:
        """Capture camera outputs for the current step."""
        for term in self.camera_terms.values():
            term.capture()

    def draw_debug_viz(self) -> None:
        for term in self.camera_terms.values():
            term.draw_debug_viz()

    def get_state(self, term_name: str) -> CameraTermBase:
        """Retrieve a stateful camera term by name.

        Parameters
        ----------
        term_name : str
            Name of the camera term.

        Returns
        -------
        CameraTermBase
            Stateful camera term instance.
        """
        del term_name
        return self.camera_term
