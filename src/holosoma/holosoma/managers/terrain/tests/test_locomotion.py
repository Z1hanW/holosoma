from types import SimpleNamespace

import pytest

from holosoma.managers.terrain.terms import locomotion
from holosoma.managers.terrain.terms.locomotion import TerrainLocomotion
from holosoma.utils.safe_torch_import import torch


@pytest.mark.parametrize(
    "env_ids",
    [
        None,
        torch.tensor([3, 0, 2], dtype=torch.long),
    ],
)
def test_get_feet_heights_uses_cartesian_env_and_body_selection(
    monkeypatch: pytest.MonkeyPatch,
    env_ids,
) -> None:
    num_envs, num_bodies = 4, 5
    rigid_body_pos = torch.arange(
        num_envs * num_bodies * 3,
        dtype=torch.float32,
    ).reshape(num_envs, num_bodies, 3)
    feet_indices = torch.tensor([1, 3], dtype=torch.long)

    term = object.__new__(TerrainLocomotion)
    term.env = SimpleNamespace(
        simulator=SimpleNamespace(_rigid_body_pos=rigid_body_pos),
        feet_height_indices=feet_indices,
    )
    term._offset_pos = torch.zeros(3, dtype=torch.float32)
    term._ray_directions_feet = torch.zeros(num_envs, len(feet_indices), 3)
    term._warp_mesh = object()

    captured: dict[str, torch.Tensor] = {}

    def fake_ray_cast(ray_starts, ray_directions, mesh):
        assert mesh is term._warp_mesh
        captured["starts"] = ray_starts.clone()
        captured["directions"] = ray_directions.clone()
        hits = ray_starts.clone()
        hits[..., 2] -= 0.25
        return hits

    monkeypatch.setattr(locomotion.warp_utils, "ray_cast", fake_ray_cast)

    heights, hits = term._get_feet_heights(env_ids)

    selected_envs = rigid_body_pos if env_ids is None else rigid_body_pos[env_ids]
    expected_positions = selected_envs[:, feet_indices, :]
    assert captured["starts"].shape == (selected_envs.shape[0], len(feet_indices), 3)
    assert torch.equal(captured["starts"], expected_positions)
    assert captured["directions"].shape == captured["starts"].shape
    assert torch.allclose(heights, torch.full_like(heights, 0.25))
    assert torch.allclose(hits[..., 2], expected_positions[..., 2] - 0.25)
