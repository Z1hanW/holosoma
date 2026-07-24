from __future__ import annotations

import ast
from pathlib import Path

import pytest


_EVALUATION_ENTRYPOINTS = {
    "src/holosoma/holosoma/eval_agent.py": 2,
    "src/holosoma/holosoma/export_teacher_box_contacts.py": 1,
    "src/holosoma/holosoma/viser_physics_rollout.py": 1,
    "scripts/record_checkpoint_inference.py": 1,
    "vis_scripts/eval_agent_viser_clip.py": 1,
    "vis_scripts/eval_agent_viser.py": 1,
}


def _algo_lifecycle_events(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    events: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if not isinstance(owner, ast.Name) or owner.id != "algo":
            continue
        if node.func.attr not in {
            "attach_evaluation_metadata",
            "setup",
            "load_evaluation",
        }:
            continue
        events.append((node.lineno, node.col_offset, node.func.attr))
    return [event for _, _, event in sorted(events)]


@pytest.mark.parametrize(("relative_path", "expected_runs"), _EVALUATION_ENTRYPOINTS.items())
def test_evaluation_metadata_is_attached_before_setup_and_load(
    relative_path: str,
    expected_runs: int,
) -> None:
    events = _algo_lifecycle_events(Path(relative_path))

    assert events == [
        event
        for _ in range(expected_runs)
        for event in (
            "attach_evaluation_metadata",
            "setup",
            "load_evaluation",
        )
    ]
