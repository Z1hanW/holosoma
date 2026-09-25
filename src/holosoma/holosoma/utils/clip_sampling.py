"""Helpers for clip-group sampling schedules."""

from __future__ import annotations

from bisect import bisect_right
from typing import Sequence

import torch


def build_prefix_mask(clip_ids: Sequence[str], prefixes: Sequence[str]) -> torch.Tensor:
    """Return a boolean mask for clip ids that match any configured prefix."""
    normalized_prefixes = tuple(prefix.strip().lower() for prefix in prefixes if prefix and prefix.strip())
    if not normalized_prefixes:
        return torch.zeros(len(clip_ids), dtype=torch.bool)
    return torch.tensor(
        [clip_id.strip().lower().startswith(normalized_prefixes) for clip_id in clip_ids],
        dtype=torch.bool,
    )


def piecewise_constant_schedule_value(
    current_iteration: int | None,
    stage_start_iterations: Sequence[int],
    stage_values: Sequence[float],
) -> float:
    """Return the value of a piecewise-constant iteration schedule."""
    if len(stage_start_iterations) != len(stage_values):
        raise ValueError(
            "stage_start_iterations and stage_values must have the same length. "
            f"Got {len(stage_start_iterations)} and {len(stage_values)}."
        )
    if not stage_start_iterations:
        raise ValueError("stage_start_iterations and stage_values must be non-empty.")

    starts = [int(value) for value in stage_start_iterations]
    if any(value < 0 for value in starts):
        raise ValueError(f"stage_start_iterations must be non-negative, got {starts}.")
    if any(curr < prev for prev, curr in zip(starts, starts[1:])):
        raise ValueError(f"stage_start_iterations must be sorted ascending, got {starts}.")

    values = [float(value) for value in stage_values]
    iteration = 0 if current_iteration is None else int(current_iteration)
    stage_idx = max(0, bisect_right(starts, iteration) - 1)
    return float(values[stage_idx])


def project_group_weights(
    base_weights: torch.Tensor,
    *,
    clean_mask: torch.Tensor,
    clean_group_probability: float,
) -> torch.Tensor:
    """Project clip weights to a target clean/noisy group split.

    Within each group, the relative weighting from ``base_weights`` is preserved.
    """
    if base_weights.ndim != 1:
        raise ValueError(f"base_weights must be 1-D, got shape {tuple(base_weights.shape)}.")
    if clean_mask.ndim != 1:
        raise ValueError(f"clean_mask must be 1-D, got shape {tuple(clean_mask.shape)}.")
    if base_weights.shape[0] != clean_mask.shape[0]:
        raise ValueError(
            "base_weights and clean_mask must have the same length. "
            f"Got {base_weights.shape[0]} and {clean_mask.shape[0]}."
        )

    clean_prob = float(min(max(clean_group_probability, 0.0), 1.0))
    total = torch.sum(base_weights)
    if not torch.isfinite(total) or total.item() <= 0.0:
        raise ValueError("base_weights must sum to a positive finite value.")

    normalized = base_weights / total
    clean_mask = clean_mask.to(device=normalized.device, dtype=torch.bool)
    noisy_mask = ~clean_mask

    if not torch.any(clean_mask) or not torch.any(noisy_mask):
        return normalized

    clean_total = torch.sum(normalized[clean_mask])
    noisy_total = torch.sum(normalized[noisy_mask])
    if clean_total.item() <= 0.0 or noisy_total.item() <= 0.0:
        return normalized

    projected = torch.zeros_like(normalized)
    projected[clean_mask] = normalized[clean_mask] / clean_total * clean_prob
    projected[noisy_mask] = normalized[noisy_mask] / noisy_total * (1.0 - clean_prob)
    return projected
