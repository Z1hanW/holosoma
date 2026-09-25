from __future__ import annotations

import math
import os

import torch
import torch.distributed as dist
from torch import nn

_GLOO_NORMALIZATION_GROUP = None


def _gloo_normalization_enabled() -> bool:
    return os.environ.get("HOLOSOMA_GLOO_SMALL_COLLECTIVES", "").lower() in ("1", "true", "yes", "on")


def _get_gloo_normalization_group():
    global _GLOO_NORMALIZATION_GROUP
    if _GLOO_NORMALIZATION_GROUP is None:
        _GLOO_NORMALIZATION_GROUP = dist.new_group(ranks=list(range(dist.get_world_size())), backend="gloo")
    return _GLOO_NORMALIZATION_GROUP


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical statistics."""

    def __init__(self, shape, device, eps: float = 1e-2, until: int | None = None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.device = device
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        # A floating count preserves fractional rank weights used when the AS
        # clip bank is duplicated to fill more distributed ranks than there
        # are unique shards.  Integer checkpoints load compatibly into this
        # buffer through ``load_state_dict``.
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64).to(device))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        center: bool = True,
        update: bool = True,
        sample_weight: float = 1.0,
    ) -> torch.Tensor:
        if x.shape[1:] != self._mean.shape[1:]:
            raise ValueError(f"Expected input of shape (*,{self._mean.shape[1:]}), got {x.shape}")

        if self.training and update:
            self.update(x, sample_weight=sample_weight)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        return x / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x, *, sample_weight: float = 1.0):
        if self.until is not None and self.count >= self.until:
            return

        sample_weight = float(sample_weight)
        if not math.isfinite(sample_weight) or sample_weight < 0.0:
            raise ValueError(
                f"EmpiricalNormalization sample_weight must be finite and non-negative, got {sample_weight!r}."
            )

        if dist.is_available() and dist.is_initialized():
            x_shifted = x - self._mean
            local_sum_shifted = torch.sum(x_shifted, dim=0, keepdim=True) * sample_weight
            local_sum_sq_shifted = torch.sum(x_shifted.pow(2), dim=0, keepdim=True) * sample_weight
            local_weighted_count = x_shifted.new_tensor([float(x.shape[0]) * sample_weight])

            feature_count = local_sum_shifted.numel()
            stats_to_sync = torch.cat(
                [
                    local_sum_shifted.reshape(-1),
                    local_sum_sq_shifted.reshape(-1),
                    local_weighted_count,
                ]
            )
            if _gloo_normalization_enabled():
                cpu_stats = stats_to_sync.detach().cpu()
                dist.all_reduce(cpu_stats, op=dist.ReduceOp.SUM, group=_get_gloo_normalization_group())
                stats_to_sync = cpu_stats.to(device=stats_to_sync.device, dtype=stats_to_sync.dtype)
            else:
                dist.all_reduce(stats_to_sync, op=dist.ReduceOp.SUM)
            global_sum_shifted = stats_to_sync[:feature_count].reshape_as(local_sum_shifted)
            global_sum_sq_shifted = stats_to_sync[feature_count : 2 * feature_count].reshape_as(
                local_sum_sq_shifted
            )
            batch_weight = stats_to_sync[-1].to(dtype=torch.float64)
            if batch_weight <= 0.0:
                raise ValueError("EmpiricalNormalization requires positive global sample weight.")

            batch_mean_shifted = global_sum_shifted / batch_weight.to(dtype=global_sum_shifted.dtype)
            batch_var = (
                global_sum_sq_shifted / batch_weight.to(dtype=global_sum_sq_shifted.dtype)
                - batch_mean_shifted.pow(2)
            ).clamp_min_(0.0)
            batch_mean = batch_mean_shifted + self._mean
        else:
            batch_weight = self.count.new_tensor(float(x.shape[0]) * sample_weight)
            if batch_weight <= 0.0:
                return
            batch_mean = torch.mean(x, dim=0, keepdim=True)
            batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)

        new_count = self.count + batch_weight

        delta = batch_mean - self._mean
        old_fraction = (self.count / new_count).to(dtype=self._mean.dtype)
        batch_fraction = (batch_weight / new_count).to(dtype=self._mean.dtype)
        self._mean.copy_(self._mean + delta * batch_fraction)

        self._var.copy_(
            self._var * old_fraction
            + batch_var * batch_fraction
            + delta.pow(2) * old_fraction * batch_fraction
        )
        self._std.copy_(torch.sqrt(self._var + self.eps))
        self.count.copy_(new_count)
