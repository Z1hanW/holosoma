from .base import BasePolicy
from .depth_distillation import DepthDistillationPolicy
from .locomotion import LocomotionPolicy
from .wbt import WholeBodyTrackingPolicy

__all__ = ["BasePolicy", "DepthDistillationPolicy", "LocomotionPolicy", "WholeBodyTrackingPolicy"]
