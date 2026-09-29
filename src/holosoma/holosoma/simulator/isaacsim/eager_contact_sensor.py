"""Fixed-shape eager update helpers for IsaacLab contact sensors.

IsaacLab's generic :class:`SensorBase` supports per-environment update periods.
That generality requires a CUDA ``nonzero()`` every time a contact sensor is
updated or read.  Holosoma's training contact sensors instead have history and
sample every physics step for every environment.  This module implements that
strict, fixed-shape contract without depending on Isaac Sim, which also makes
the timestamp/history semantics independently testable.
"""

from __future__ import annotations

from typing import Any


def eager_every_step_is_compatible(*, update_period: float, physics_dt: float, history_length: int) -> bool:
    """Return whether every scene update must sample every environment.

    A positive history is required because IsaacLab already forces those
    sensors to compute eagerly.  ``update_period <= physics_dt`` means every
    physics update is due, so resolving a dynamic subset cannot change which
    rows are sampled.
    """

    return int(history_length) > 0 and float(update_period) <= float(physics_dt) + 1.0e-12


def eager_update_all(sensor: Any, dt: float) -> None:
    """Apply one all-environment sensor update with SensorBase-equivalent clocks."""

    sensor._timestamp += dt
    sensor._update_buffers_impl(sensor._eager_all_env_ids)
    sensor._timestamp_last_update.copy_(sensor._timestamp)
    sensor._is_outdated.zero_()
    sensor._eager_has_sample = True


def eager_data(sensor: Any) -> Any:
    """Return the eager cache, taking one fixed-shape initial sample if needed.

    After a reset, ContactSensor.reset() deliberately zeros selected history
    rows and marks them outdated.  Returning the cache until the next physics
    update preserves those zeros; re-querying PhysX immediately would copy the
    previous episode's contact report back into the new episode.
    """

    if not sensor._eager_has_sample:
        sensor._update_buffers_impl(sensor._eager_all_env_ids)
        sensor._timestamp_last_update.copy_(sensor._timestamp)
        sensor._is_outdated.zero_()
        sensor._eager_has_sample = True
    return sensor._data
