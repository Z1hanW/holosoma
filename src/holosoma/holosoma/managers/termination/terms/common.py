"""Common termination term helpers."""

from __future__ import annotations

from holosoma.utils.safe_torch_import import torch


def timeout_exceeded(env, **_) -> torch.Tensor:
    """Terminate environments that exceeded the maximum episode length."""
    # ``BaseTask`` increments ``episode_length_buf`` before checking
    # termination.  Using ``>`` here therefore executes one extra control step
    # beyond the configured horizon and shifts time-limit bootstrapping by one
    # transition.
    return env.episode_length_buf >= env.max_episode_length
