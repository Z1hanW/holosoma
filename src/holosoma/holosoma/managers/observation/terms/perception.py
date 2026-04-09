"""Perception observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from holosoma.envs.base_task.base_task import BaseTask


def perception_obs(env: BaseTask) -> torch.Tensor:
    perception = getattr(env, "perception_manager", None)
    if perception is None:
        raise AttributeError("Environment is missing perception_manager for perception_obs.")
    return perception.get_obs()


def teacher_perception_obs(env: BaseTask) -> torch.Tensor:
    perception = getattr(env, "teacher_perception_manager", None)
    if perception is None:
        raise AttributeError("Environment is missing teacher_perception_manager for teacher_perception_obs.")
    return perception.get_obs()


def critic_perception_obs(env: BaseTask) -> torch.Tensor:
    perception = getattr(env, "critic_perception_manager", None)
    if perception is None:
        raise AttributeError("Environment is missing critic_perception_manager for critic_perception_obs.")
    return perception.get_obs()
