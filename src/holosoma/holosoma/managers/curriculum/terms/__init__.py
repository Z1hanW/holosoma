"""Curriculum term implementations."""

from .locomotion import WObjectDifficultyCurriculum, clamp_reward_penalty, configure_reward_penalty, update_reward_penalty

__all__ = ["WObjectDifficultyCurriculum", "clamp_reward_penalty", "configure_reward_penalty", "update_reward_penalty"]
