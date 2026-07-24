from __future__ import annotations

import torch

import holosoma.simulator.isaacsim.proxy_utils as proxy_utils
from holosoma.simulator.isaacsim.proxy_utils import RootStatesProxy
from holosoma.simulator.isaacsim.state_utils import (
    fullstate_wxyz_to_xyzw,
    fullstate_xyzw_to_wxyz,
)


def _root_states(num_envs: int = 6) -> torch.Tensor:
    # Distinct values in every field make row and quaternion-order mistakes
    # visible without relying on quaternion normalization.
    return torch.arange(num_envs * 13, dtype=torch.float32).reshape(num_envs, 13) / 17.0


def _assert_proxy_matches_reference(
    proxy: RootStatesProxy,
    expected_xyzw: torch.Tensor,
    backing_wxyz: torch.Tensor,
    *,
    backing_ptr: int,
    xyzw_ptr: int,
) -> None:
    assert proxy.tensor_wxyz is backing_wxyz
    assert proxy.tensor_wxyz.data_ptr() == backing_ptr
    assert proxy.tensor_xyzw.data_ptr() == xyzw_ptr
    assert torch.equal(proxy.tensor_xyzw, expected_xyzw)
    assert torch.equal(backing_wxyz, fullstate_xyzw_to_wxyz(expected_xyzw))


def test_partial_writes_preserve_all_pytorch_row_and_tuple_index_semantics():
    backing_wxyz = _root_states()
    expected_xyzw = fullstate_wxyz_to_xyzw(backing_wxyz)
    proxy = RootStatesProxy(backing_wxyz)
    backing_ptr = backing_wxyz.data_ptr()
    xyzw_ptr = proxy.tensor_xyzw.data_ptr()

    bool_rows = torch.tensor([True, False, True, False, False, True])
    writes = [
        (1, torch.linspace(-1.0, 1.0, 13)),
        (slice(2, 4), torch.arange(26, dtype=torch.float32).reshape(2, 13) + 100.0),
        (torch.tensor([0, 4]), torch.arange(13, dtype=torch.float32) + 200.0),
        (bool_rows, torch.arange(39, dtype=torch.float32).reshape(3, 13) + 300.0),
        (
            (torch.tensor([1, 5]), slice(3, 7)),
            torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
        ),
        (
            (torch.tensor([0, 2]), torch.tensor([0, 8])),
            torch.tensor([-8.0, -9.0]),
        ),
        (
            (torch.tensor([3, 3]), slice(0, 2)),
            torch.tensor([[31.0, 32.0], [41.0, 42.0]]),
        ),
        ((Ellipsis, slice(10, 13)), torch.tensor([9.0, 8.0, 7.0])),
    ]

    for index, value in writes:
        expected_xyzw[index] = value
        proxy[index] = value
        _assert_proxy_matches_reference(
            proxy,
            expected_xyzw,
            backing_wxyz,
            backing_ptr=backing_ptr,
            xyzw_ptr=xyzw_ptr,
        )


def test_partial_write_converts_only_affected_complete_rows(monkeypatch):
    backing_wxyz = _root_states()
    proxy = RootStatesProxy(backing_wxyz)
    original_convert = proxy_utils.fullstate_xyzw_to_wxyz
    converted_shapes: list[tuple[int, ...]] = []

    def recording_convert(states_xyzw: torch.Tensor) -> torch.Tensor:
        converted_shapes.append(tuple(states_xyzw.shape))
        return original_convert(states_xyzw)

    monkeypatch.setattr(proxy_utils, "fullstate_xyzw_to_wxyz", recording_convert)

    proxy[torch.tensor([1, 4]), :3] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert converted_shapes == [(2, 13)]

    converted_shapes.clear()
    proxy[2, 3:7] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    assert converted_shapes == [(1, 13)]


def test_multidimensional_boolean_index_updates_only_unique_affected_rows(monkeypatch):
    backing_wxyz = _root_states()
    expected_xyzw = fullstate_wxyz_to_xyzw(backing_wxyz)
    proxy = RootStatesProxy(backing_wxyz)
    original_convert = proxy_utils.fullstate_xyzw_to_wxyz
    converted_shapes: list[tuple[int, ...]] = []

    def recording_convert(states_xyzw: torch.Tensor) -> torch.Tensor:
        converted_shapes.append(tuple(states_xyzw.shape))
        return original_convert(states_xyzw)

    monkeypatch.setattr(proxy_utils, "fullstate_xyzw_to_wxyz", recording_convert)
    mask = torch.zeros_like(expected_xyzw, dtype=torch.bool)
    mask[1, [0, 3]] = True
    mask[4, [6, 12]] = True
    values = torch.tensor([-1.0, -2.0, -3.0, -4.0])

    expected_xyzw[mask] = values
    proxy[mask] = values

    assert converted_shapes == [(2, 13)]
    assert torch.equal(proxy.tensor_xyzw, expected_xyzw)
    assert torch.equal(backing_wxyz, fullstate_xyzw_to_wxyz(expected_xyzw))

    converted_shapes.clear()
    row_ids = torch.tensor([0, 5])
    columns = torch.tensor([1, 11])
    expected_xyzw[..., row_ids, columns] = torch.tensor([21.0, 22.0])
    proxy[..., row_ids, columns] = torch.tensor([21.0, 22.0])

    assert converted_shapes == [(2, 13)]
    assert torch.equal(proxy.tensor_xyzw, expected_xyzw)
    assert torch.equal(backing_wxyz, fullstate_xyzw_to_wxyz(expected_xyzw))


def test_full_assignment_updates_backing_in_place_without_rebinding(monkeypatch):
    backing_wxyz = _root_states()
    proxy = RootStatesProxy(backing_wxyz)
    backing_ptr = backing_wxyz.data_ptr()
    xyzw_ptr = proxy.tensor_xyzw.data_ptr()
    replacement_xyzw = torch.flip(proxy.tensor_xyzw, dims=(0,)).clone() + 500.0
    original_convert = proxy_utils.fullstate_xyzw_to_wxyz
    converted_shapes: list[tuple[int, ...]] = []

    def recording_convert(states_xyzw: torch.Tensor) -> torch.Tensor:
        converted_shapes.append(tuple(states_xyzw.shape))
        return original_convert(states_xyzw)

    monkeypatch.setattr(proxy_utils, "fullstate_xyzw_to_wxyz", recording_convert)
    proxy[:, :] = replacement_xyzw

    assert converted_shapes == [(backing_wxyz.shape[0], 13)]
    _assert_proxy_matches_reference(
        proxy,
        replacement_xyzw,
        backing_wxyz,
        backing_ptr=backing_ptr,
        xyzw_ptr=xyzw_ptr,
    )
