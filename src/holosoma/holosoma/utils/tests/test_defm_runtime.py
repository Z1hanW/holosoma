from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from holosoma.utils.defm_runtime import (
    DEFM_MATERIALIZATION_MODE_ENV,
    set_defm_checkpoint_restore_mode,
    set_defm_materialization_mode,
)
from holosoma.utils.defm_source import (
    require_module_within_defm_root,
    resolve_defm_source_root,
)


def test_defm_materialization_mode_overrides_ambient_state(monkeypatch):
    monkeypatch.setenv(DEFM_MATERIALIZATION_MODE_ENV, "ambient-must-not-win")

    assert set_defm_checkpoint_restore_mode() == "full_resume"
    assert os.environ[DEFM_MATERIALIZATION_MODE_ENV] == "full_resume"


def test_defm_materialization_mode_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv(DEFM_MATERIALIZATION_MODE_ENV, "sentinel")

    with pytest.raises(ValueError, match="must be one of"):
        set_defm_materialization_mode("checkpoint-ish")
    assert os.environ[DEFM_MATERIALIZATION_MODE_ENV] == "sentinel"


def test_defm_source_resolution_rejects_ambiguous_checkouts(tmp_path):
    project = tmp_path / "project"
    for source_root in (project / "defm", project / "submodules" / "defm"):
        model_factory = source_root / "defm" / "model_factory.py"
        model_factory.parent.mkdir(parents=True)
        model_factory.write_text("# test\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Multiple local DeFM source trees"):
        resolve_defm_source_root(environ={}, anchor=project / "entry.py")


def test_explicit_defm_source_is_authoritative_and_import_origin_is_checked(tmp_path):
    selected = tmp_path / "selected"
    model_factory = selected / "defm" / "model_factory.py"
    model_factory.parent.mkdir(parents=True)
    model_factory.write_text("# selected\n", encoding="utf-8")
    resolved = resolve_defm_source_root(
        environ={"HOLOSOMA_DEFM_ROOT": str(selected)},
        anchor=tmp_path / "entry.py",
    )

    assert resolved == selected.resolve()
    require_module_within_defm_root(
        SimpleNamespace(__file__=str(model_factory)),
        resolved,
        name="defm.model_factory",
    )
    with pytest.raises(RuntimeError, match="outside the pinned/hashed DeFM root"):
        require_module_within_defm_root(
            SimpleNamespace(__file__=str(tmp_path / "other" / "model_factory.py")),
            resolved,
            name="defm.model_factory",
        )


@pytest.mark.parametrize(
    ("relative_path", "function_name"),
    [
        ("eval_agent.py", "run_eval_with_tyro"),
        ("viser_physics_rollout.py", "run_physics_rollout"),
        ("export_teacher_box_contacts.py", "run_export_with_tyro"),
    ],
)
def test_checkpoint_entrypoint_sets_restore_mode_before_algo_setup(
    relative_path: str,
    function_name: str,
):
    package_root = Path(__file__).resolve().parents[2]
    tree = ast.parse((package_root / relative_path).read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]

    restore_lines = [
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "set_defm_checkpoint_restore_mode"
    ]
    setup_lines = [
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "setup"
    ]

    assert len(restore_lines) == 1
    assert setup_lines
    assert restore_lines[0] < min(setup_lines)
