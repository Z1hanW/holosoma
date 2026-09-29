from __future__ import annotations

import torch

from holosoma.managers.termination.terms.locomotion import _apply_probability


def test_probability_gate_samples_each_environment_independently(monkeypatch) -> None:
    requested_shapes: list[torch.Size] = []

    def fake_rand(shape, *, device):
        requested_shapes.append(torch.Size(shape))
        assert str(device) == "cpu"
        return torch.tensor([0.1, 0.9, 0.2, 0.8], device=device)

    monkeypatch.setattr(torch, "rand", fake_rand)
    mask = torch.tensor([True, True, False, True])

    result = _apply_probability(mask, probability=0.5, device=torch.device("cpu"))

    assert requested_shapes == [mask.shape]
    assert torch.equal(result, torch.tensor([True, False, False, False]))
