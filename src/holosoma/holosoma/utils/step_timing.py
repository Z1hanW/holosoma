from __future__ import annotations

import os
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from holosoma.utils.safe_torch_import import torch


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


def env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return default


def env_int(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class StepTiming:
    """Low-overhead opt-in wall-time accumulator for training hot paths."""

    def __init__(
        self,
        *,
        enabled: bool,
        device: str | torch.device | None = None,
        sync_cuda: bool = False,
    ) -> None:
        self.enabled = enabled
        self.device = device
        self.sync_cuda = sync_cuda
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    @classmethod
    def from_env(cls, *, device: str | torch.device | None = None) -> "StepTiming":
        profiling_enabled = env_flag("HOLOSOMA_STEP_TIMING_PROFILE", default=False)
        timing_requested = env_flag("HOLOSOMA_STEP_TIMING", default=False)
        return cls(
            enabled=profiling_enabled and timing_requested,
            device=device,
            sync_cuda=profiling_enabled and env_flag("HOLOSOMA_STEP_TIMING_SYNC_CUDA", default=False),
        )

    def reset(self) -> None:
        self._totals.clear()
        self._counts.clear()

    def add(self, name: str, elapsed_s: float) -> None:
        if not self.enabled:
            return
        self._totals[name] = self._totals.get(name, 0.0) + elapsed_s
        self._counts[name] = self._counts.get(name, 0) + 1

    @contextmanager
    def record(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync_cuda_if_needed()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync_cuda_if_needed()
            self.add(name, time.perf_counter() - start)

    def snapshot(self, *, reset: bool = False) -> dict[str, dict[str, float]]:
        snapshot: dict[str, dict[str, float]] = {}
        for name, total_s in self._totals.items():
            count = self._counts.get(name, 0)
            if count <= 0:
                continue
            total_ms = total_s * 1000.0
            snapshot[name] = {
                "sum_ms": total_ms,
                "mean_ms": total_ms / float(count),
                "count": float(count),
            }
        if reset:
            self.reset()
        return snapshot

    def _sync_cuda_if_needed(self) -> None:
        if not self.sync_cuda or not torch.cuda.is_available():
            return
        if self.device is None:
            torch.cuda.synchronize()
            return
        try:
            torch.cuda.synchronize(self.device)
        except (TypeError, ValueError):
            torch.cuda.synchronize()


def flatten_timing_snapshot(
    snapshot: dict[str, dict[str, float]],
    *,
    prefix: str,
    fields: Iterable[str] = ("sum_ms", "mean_ms"),
) -> dict[str, float]:
    flat: dict[str, float] = {}
    for name, stats in snapshot.items():
        safe_name = name.replace("/", "_")
        for field in fields:
            value = stats.get(field)
            if value is not None:
                flat[f"{prefix}_{safe_name}_{field}"] = float(value)
    return flat


def compact_timing_summary(
    snapshot: dict[str, dict[str, float]],
    preferred_order: Iterable[str],
    *,
    max_extra: int = 3,
) -> str:
    if not snapshot:
        return "<empty>"

    parts: list[str] = []
    seen: set[str] = set()
    for name in preferred_order:
        stats = snapshot.get(name)
        if stats is None:
            continue
        parts.append(_format_timing_part(name, stats))
        seen.add(name)

    remaining = [
        (name, stats)
        for name, stats in snapshot.items()
        if name not in seen
    ]
    remaining.sort(key=lambda item: item[1].get("sum_ms", 0.0), reverse=True)
    for name, stats in remaining[:max_extra]:
        parts.append(_format_timing_part(name, stats))

    return ", ".join(parts) if parts else "<empty>"


def _format_timing_part(name: str, stats: dict[str, float]) -> str:
    return (
        f"{name}:sum={stats.get('sum_ms', 0.0):.2f}ms"
        f"/mean={stats.get('mean_ms', 0.0):.2f}ms"
        f"/n={int(stats.get('count', 0.0))}"
    )
