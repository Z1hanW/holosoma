from __future__ import annotations

import ast
import inspect
import random

import numpy as np
import pytest

from holosoma import train_agent
from holosoma.utils import sim_utils
from holosoma.utils import torch_utils
from holosoma.utils.common import seeding, validate_numpy_seed
from holosoma.utils.safe_torch_import import torch


@pytest.mark.parametrize("seed", [0, 1, 2**32 - 1])
def test_numpy_seed_accepts_closed_mt19937_range(seed: int) -> None:
    assert validate_numpy_seed(seed) == seed


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5, "1"])
def test_numpy_seed_rejects_values_numpy_random_seed_cannot_use(seed) -> None:
    with pytest.raises(ValueError, match="Seed must be an integer"):
        validate_numpy_seed(seed)


def test_generic_simulation_validates_seed_before_app_launcher() -> None:
    source = inspect.getsource(sim_utils.setup_simulation_environment)
    seed_validation = source.index("rank_training_seed(")
    deterministic_validation = source.index("validate_deterministic_runtime(")
    simulator_imports = source.index("setup_simulator_imports(config)")
    app_launcher = source.index("setup_isaaclab_launcher(config, device)")
    seed_application = source.index("seeding(seed, torch_deterministic=")

    assert seed_validation < deterministic_validation < simulator_imports < app_launcher < seed_application


def test_sim_utils_does_not_import_warp_initializing_managers_at_module_scope() -> None:
    tree = ast.parse(inspect.getsource(sim_utils))
    top_level_imports = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "holosoma.managers.perception" not in top_level_imports
    assert "holosoma.managers.terrain.manager" not in top_level_imports


def _assert_numpy_state_equal(left, right) -> None:
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


@pytest.mark.parametrize("seed_fn", [seeding, torch_utils.set_seed])
def test_missing_cublas_determinism_prerequisite_preserves_all_rngs(
    monkeypatch: pytest.MonkeyPatch,
    seed_fn,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    cuda_calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda *_: cuda_calls.append("manual_seed"))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *_: cuda_calls.append("manual_seed_all"))

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        seed_fn(-1 if seed_fn is torch_utils.set_seed else 123, torch_deterministic=True)

    assert random.getstate() == python_state
    _assert_numpy_state_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert cuda_calls == []


def test_invalid_seed_is_rejected_before_any_rng_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    cuda_calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda *_: cuda_calls.append("manual_seed"))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *_: cuda_calls.append("manual_seed_all"))
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()

    with pytest.raises(ValueError, match="Seed must be an integer"):
        seeding(2**32, torch_deterministic=False)

    assert random.getstate() == python_state
    _assert_numpy_state_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert cuda_calls == []


def test_rank_seed_contract_covers_distributed_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "104")
    monkeypatch.setenv("RANK", "103")
    assert train_agent._current_rank_training_seed(17) == 120


@pytest.mark.parametrize("entrypoint", [train_agent.TrainingContext.__enter__, train_agent.train])
def test_direct_training_entrypoints_validate_seed_before_simulator(entrypoint) -> None:
    source = inspect.getsource(entrypoint)
    validation = source.index("_current_rank_training_seed(")
    simulator = source.index("init_sim_imports(")
    assert validation < simulator
