from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from holosoma.simulator.isaacsim.eager_contact_sensor import (
    eager_data,
    eager_every_step_is_compatible,
    eager_update_all,
)


class _FakeSensor:
    def __init__(self, num_envs: int = 3) -> None:
        self._timestamp = torch.zeros(num_envs)
        self._timestamp_last_update = torch.zeros(num_envs)
        self._is_outdated = torch.ones(num_envs, dtype=torch.bool)
        self._eager_all_env_ids = torch.arange(num_envs)
        self._eager_has_sample = False
        self._data = SimpleNamespace(history=torch.zeros(num_envs, 2))
        self.samples = 0

    def _update_buffers_impl(self, env_ids: torch.Tensor) -> None:
        assert torch.equal(env_ids, self._eager_all_env_ids)
        self.samples += 1
        self._data.history[:, 1] = self._data.history[:, 0]
        self._data.history[:, 0] = float(self.samples)


class _FakeAirTimeSensor(_FakeSensor):
    """Small ContactSensor state machine used to check timestamp semantics."""

    def __init__(self, num_envs: int = 3) -> None:
        super().__init__(num_envs=num_envs)
        self._data.current_air_time = torch.zeros(num_envs)
        self._data.last_air_time = torch.zeros(num_envs)
        self._data.current_contact_time = torch.zeros(num_envs)
        self._data.last_contact_time = torch.zeros(num_envs)
        self.is_contact = torch.zeros(num_envs, dtype=torch.bool)

    def _update_buffers_impl(self, env_ids: torch.Tensor) -> None:
        super()._update_buffers_impl(env_ids)
        elapsed_time = self._timestamp[env_ids] - self._timestamp_last_update[env_ids]
        is_contact = self.is_contact[env_ids]
        is_first_contact = (self._data.current_air_time[env_ids] > 0) & is_contact
        is_first_detached = (self._data.current_contact_time[env_ids] > 0) & ~is_contact
        self._data.last_air_time[env_ids] = torch.where(
            is_first_contact,
            self._data.current_air_time[env_ids] + elapsed_time,
            self._data.last_air_time[env_ids],
        )
        self._data.current_air_time[env_ids] = torch.where(
            ~is_contact,
            self._data.current_air_time[env_ids] + elapsed_time,
            0.0,
        )
        self._data.last_contact_time[env_ids] = torch.where(
            is_first_detached,
            self._data.current_contact_time[env_ids] + elapsed_time,
            self._data.last_contact_time[env_ids],
        )
        self._data.current_contact_time[env_ids] = torch.where(
            is_contact,
            self._data.current_contact_time[env_ids] + elapsed_time,
            0.0,
        )

    def reset_rows(self, env_ids: torch.Tensor) -> None:
        """Mirror the relevant SensorBase/ContactSensor.reset operations."""

        self._timestamp[env_ids] = 0.0
        self._timestamp_last_update[env_ids] = 0.0
        self._is_outdated[env_ids] = True
        self._data.history[env_ids] = 0.0
        self._data.current_air_time[env_ids] = 0.0
        self._data.last_air_time[env_ids] = 0.0
        self._data.current_contact_time[env_ids] = 0.0
        self._data.last_contact_time[env_ids] = 0.0


def test_eager_contract_only_accepts_due_history_sensors() -> None:
    assert eager_every_step_is_compatible(update_period=0.005, physics_dt=0.005, history_length=4)
    assert eager_every_step_is_compatible(update_period=0.0, physics_dt=0.005, history_length=1)
    assert not eager_every_step_is_compatible(update_period=0.01, physics_dt=0.005, history_length=4)
    assert not eager_every_step_is_compatible(update_period=0.005, physics_dt=0.005, history_length=0)


def test_initial_data_uses_one_fixed_shape_sample_then_reads_cache() -> None:
    sensor = _FakeSensor()

    assert eager_data(sensor) is sensor._data
    assert sensor.samples == 1
    assert not torch.any(sensor._is_outdated)
    assert eager_data(sensor) is sensor._data
    assert sensor.samples == 1


def test_update_advances_clocks_and_history_once() -> None:
    sensor = _FakeSensor()
    eager_data(sensor)

    eager_update_all(sensor, 0.005)

    assert sensor.samples == 2
    assert torch.equal(sensor._timestamp, torch.full((3,), 0.005))
    assert torch.equal(sensor._timestamp_last_update, sensor._timestamp)
    assert torch.equal(sensor._data.history[:, 0], torch.full((3,), 2.0))
    assert torch.equal(sensor._data.history[:, 1], torch.full((3,), 1.0))


def test_scene_update_before_first_data_takes_exactly_one_sample() -> None:
    sensor = _FakeSensor()

    eager_update_all(sensor, 0.005)

    assert sensor.samples == 1
    assert sensor._eager_has_sample
    assert torch.equal(sensor._timestamp, torch.full((3,), 0.005))
    assert torch.equal(sensor._timestamp_last_update, sensor._timestamp)
    assert eager_data(sensor) is sensor._data
    assert sensor.samples == 1


def test_reset_zero_cache_is_not_repopulated_before_next_physics_update() -> None:
    sensor = _FakeSensor()
    eager_data(sensor)
    sensor._data.history[1].zero_()
    sensor._is_outdated[1] = True

    # This models refresh_sim_tensors immediately after a reset write.  It must
    # observe the explicit zero, not query the prior PhysX contact report.
    data = eager_data(sensor)
    assert sensor.samples == 1
    assert torch.equal(data.history[1], torch.zeros(2))

    eager_update_all(sensor, 0.005)
    assert sensor.samples == 2
    assert torch.equal(data.history[1], torch.tensor([2.0, 0.0]))


def test_partial_reset_preserves_air_time_and_history_semantics() -> None:
    sensor = _FakeAirTimeSensor()
    sensor.is_contact[:] = torch.tensor([False, True, False])
    eager_data(sensor)
    eager_update_all(sensor, 0.005)

    assert torch.equal(sensor._data.current_air_time, torch.tensor([0.005, 0.0, 0.005]))
    assert torch.equal(sensor._data.current_contact_time, torch.tensor([0.0, 0.005, 0.0]))

    # Exercise both air->contact and contact->air transitions before resetting
    # one row.  These are the same elapsed-time equations used by ContactSensor.
    sensor.is_contact[:] = torch.tensor([True, False, False])
    eager_update_all(sensor, 0.005)
    assert torch.equal(sensor._data.last_air_time, torch.tensor([0.010, 0.0, 0.0]))
    assert torch.equal(sensor._data.last_contact_time, torch.tensor([0.0, 0.010, 0.0]))

    reset_ids = torch.tensor([1])
    sensor.reset_rows(reset_ids)
    samples_before_refresh = sensor.samples

    # The reset-state observation sees explicit zeros.  The next scene update
    # samples all rows and uses one physics dt for both reset and live rows.
    data = eager_data(sensor)
    assert sensor.samples == samples_before_refresh
    assert torch.equal(data.history[1], torch.zeros(2))
    assert sensor._is_outdated[1]

    sensor.is_contact[1] = True
    eager_update_all(sensor, 0.005)
    assert torch.equal(sensor._timestamp, torch.tensor([0.015, 0.005, 0.015]))
    assert torch.equal(sensor._timestamp_last_update, sensor._timestamp)
    assert not torch.any(sensor._is_outdated)
    assert torch.isclose(sensor._data.current_contact_time[0], torch.tensor(0.010))
    assert torch.isclose(sensor._data.current_contact_time[1], torch.tensor(0.005))
    assert torch.equal(data.history[1], torch.tensor([4.0, 0.0]))


def test_holosoma_contact_sensors_do_not_enable_unused_air_time_tracking() -> None:
    """Keep the per-substep fast path free of unconsumed duration kernels."""

    source_path = (
        Path(__file__).resolve().parents[2]
        / "src/holosoma/holosoma/simulator/isaacsim/isaacsim.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    contact_cfg_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ContactSensorCfg"
    ]

    assert len(contact_cfg_calls) == 2
    for call in contact_cfg_calls:
        track_air_time = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "track_air_time"),
            None,
        )
        assert isinstance(track_air_time, ast.Constant)
        assert track_air_time.value is False
