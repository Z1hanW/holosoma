from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "finalize_prism_combined_viewer.py"
)
SPEC = importlib.util.spec_from_file_location(
    "finalize_prism_combined_viewer",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_viewer(root: Path, sequence: str) -> None:
    (root / "viewer" / "retarget" / sequence).mkdir(parents=True)
    (root / "viewer" / "metadata" / "box_mesh" / sequence).mkdir(parents=True)
    (root / "viewer" / "retarget" / sequence / f"{sequence}_original.npz").touch()
    (root / "viewer" / "metadata" / "box_mesh" / sequence / f"{sequence}.obj").touch()
    (root / "viewer" / "adapter_manifest.json").write_text(
        json.dumps([{"sequence": sequence, "state": "accepted"}]),
        encoding="utf-8",
    )


def test_combine_viewers_links_both_sources(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    output = tmp_path / "combined"
    output.mkdir()
    make_viewer(primary, "sequence_a")
    make_viewer(secondary, "sequence_b")

    rows = MODULE.combine_viewers(
        [primary, secondary],
        output,
        expected_count=2,
    )

    assert [row["sequence"] for row in rows] == ["sequence_a", "sequence_b"]
    assert (output / "viewer" / "retarget" / "sequence_a").is_symlink()
    assert (output / "viewer" / "retarget" / "sequence_b").is_symlink()
    assert (
        output / "viewer" / "metadata" / "box_mesh" / "sequence_a"
    ).is_symlink()
    manifest = json.loads(
        (output / "viewer" / "adapter_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest) == 2
