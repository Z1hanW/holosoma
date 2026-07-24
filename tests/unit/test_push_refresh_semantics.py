from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[2]
_METHOD_NAMES = {
    "_get_envs_to_refresh",
    "_push_robots",
    "_refresh_envs_after_reset",
    "_reset_buffers_callback",
}


def _minimal_manager(relative_path: str, class_name: str) -> type:
    """Compile only the four methods under test, without importing simulator code."""
    source_path = _ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    source_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = [
        node
        for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name in _METHOD_NAMES
    ]
    assert {method.name for method in methods} == _METHOD_NAMES
    minimal_class = ast.ClassDef(
        name=class_name,
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[minimal_class], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
    return namespace[class_name]


class _SimulatorRecorder:
    def __init__(self, root_states: torch.Tensor) -> None:
        self.robot_root_states = root_states
        self.all_root_states = root_states
        self.dof_state = object()
        self.push_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.root_calls: list[torch.Tensor] = []
        self.dof_calls: list[torch.Tensor] = []
        self.contact_calls: list[torch.Tensor] = []
        self.refresh_calls = 0

    def set_actor_root_state_tensor_robots(self, env_ids, states) -> None:
        self.push_calls.append((env_ids.clone(), states.clone()))

    def set_actor_root_state_tensor(self, env_ids, _states) -> None:
        self.root_calls.append(env_ids.clone())

    def set_dof_state_tensor(self, env_ids, _states=None) -> None:
        self.dof_calls.append(env_ids.clone())

    def clear_contact_forces_history(self, env_ids) -> None:
        self.contact_calls.append(env_ids.clone())

    def refresh_sim_tensors(self) -> None:
        self.refresh_calls += 1


@pytest.mark.parametrize(
    ("relative_path", "class_name", "velocity_width", "velocity_slice", "additive"),
    [
        (
            "src/holosoma/holosoma/envs/wbt/wbt_manager.py",
            "WholeBodyTrackingManager",
            6,
            slice(7, 13),
            True,
        ),
        (
            "src/holosoma/holosoma/envs/locomotion/locomotion_manager.py",
            "LeggedRobotLocomotionManager",
            2,
            slice(7, 9),
            False,
        ),
    ],
)
def test_write_through_push_does_not_enter_reset_refresh_path(
    relative_path: str,
    class_name: str,
    velocity_width: int,
    velocity_slice: slice,
    additive: bool,
) -> None:
    manager_class = _minimal_manager(relative_path, class_name)
    env = manager_class()
    env.device = "cpu"
    env.num_envs = 4
    env.randomization_manager = None
    env._max_push_vel = torch.linspace(0.2, 0.8, velocity_width)
    env.push_robot_vel_buf = torch.zeros((env.num_envs, velocity_width))
    env.record_push_robot_vel_buf = torch.zeros_like(env.push_robot_vel_buf)
    env.need_to_refresh_envs = torch.zeros(env.num_envs, dtype=torch.bool)
    env.episode_length_buf = torch.full((env.num_envs,), 12, dtype=torch.long)
    env.reset_buf = torch.zeros(env.num_envs, dtype=torch.long)
    env._pending_episode_update_mask = torch.zeros(env.num_envs, dtype=torch.bool)

    root_states = torch.arange(env.num_envs * 13, dtype=torch.float32).reshape(env.num_envs, 13)
    simulator = _SimulatorRecorder(root_states)
    env.simulator = simulator
    perception_calls: list[torch.Tensor] = []
    env._pre_compute_observations_callback = lambda env_ids=None: perception_calls.append(env_ids.clone())

    reset_env_ids = torch.tensor([3])
    pushed_env_ids = torch.tensor([0, 2])
    env._reset_buffers_callback(reset_env_ids)
    velocity_before = root_states[:, velocity_slice].clone()

    torch.manual_seed(7)
    env._push_robots(pushed_env_ids)

    expected = env.push_robot_vel_buf[pushed_env_ids]
    if additive:
        expected = velocity_before[pushed_env_ids] + expected
    torch.testing.assert_close(root_states[pushed_env_ids, velocity_slice], expected)
    assert len(simulator.push_calls) == 1
    push_call_ids, pushed_states = simulator.push_calls[0]
    assert torch.equal(push_call_ids, pushed_env_ids)
    assert torch.equal(pushed_states, root_states)

    # A push is already in the backend; only the real reset may clear contact
    # history or trigger a second perception refresh on the following step.
    refresh_env_ids = env._get_envs_to_refresh()
    assert torch.equal(refresh_env_ids, reset_env_ids)
    env._refresh_envs_after_reset(refresh_env_ids)

    assert len(simulator.root_calls) == 1
    assert len(simulator.dof_calls) == 1
    assert len(simulator.contact_calls) == 1
    assert torch.equal(simulator.root_calls[0], reset_env_ids)
    assert torch.equal(simulator.dof_calls[0], reset_env_ids)
    assert torch.equal(simulator.contact_calls[0], reset_env_ids)
    assert simulator.refresh_calls == 1
    assert len(perception_calls) == 1
    assert torch.equal(perception_calls[0], reset_env_ids)
    assert not torch.any(env.need_to_refresh_envs)
