from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "corl_numbers" / "train_as_general_realdata.sh"
MARKER_NAME = ".generated_by_train_as_general_realdata"
MANIFEST_NAME = "realdata_union_manifest.json"


def _write_bank(root: Path, clip_id: str, asset_name: str, motion: bytes) -> None:
    objects = root / "objects"
    objects.mkdir(parents=True)
    mesh = objects / f"{asset_name}.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    urdf = objects / f"{asset_name}.urdf"
    urdf.write_text(
        "<robot name='fixture'><link name='object'><visual><geometry>"
        f"<mesh filename='{mesh.name}'/>"
        "</geometry></visual></link></robot>\n",
        encoding="utf-8",
    )
    (root / f"{clip_id}.npz").write_bytes(motion)
    (root / "_clip_object_urdf_map.json").write_text(
        json.dumps(
            {
                "clips": {
                    clip_id: {
                        "object_name": asset_name,
                        "object_urdf_path": f"objects/{urdf.name}",
                        "object_mesh_path": f"objects/{mesh.name}",
                    }
                }
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def fixture_repo(tmp_path: Path) -> dict[str, Path | dict[str, str]]:
    repo = tmp_path / "repo"
    wrapper = repo / "corl_numbers" / WRAPPER.name
    wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, wrapper)

    capture = repo / "launcher-env.txt"
    (repo / "train_as_general.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        ": \"${TEST_CAPTURE:?}\"\n"
        "printf '%s\\n' \"${AS_DATA_DIR}\" \"${AS_OBJECT_MAP}\" "
        "\"${AS_EXPECTED_TOTAL}\" \"${CONTACT_EXPORT_ROOT-<unset>}\" "
        "\"${AS_SINGLE_SLOT_MOTION_BASE}\" \"${REWARD_CONFIG}\" "
        "\"${HOLOSOMA_REQUIRE_CONTACT_INTERVAL_COVERAGE}\" "
        "\"${HOLOSOMA_REQUIRE_CONTACT_TARGET_COVERAGE}\" > \"${TEST_CAPTURE}\"\n",
        encoding="utf-8",
    )

    data_root = repo / "data"
    realdata_root = data_root / "corl_numbers"
    first = realdata_root / "bank_a"
    second = realdata_root / "bank_b"
    _write_bank(first, "clip_a", "box", b"motion-a-v1")
    _write_bank(second, "clip_b", "chair", b"motion-b-v1")

    alias = realdata_root / "realdata_union"
    contact = realdata_root / "contact_exports"
    env = {
        **os.environ,
        "REALDATA_ROOT": "data/corl_numbers",
        "OMOMO_REALDATA_DIR": "data/corl_numbers/bank_a",
        "BEHAVE_REALDATA_DIR": "data/corl_numbers/bank_b",
        "REALDATA_UNION_DIR": "data/corl_numbers/realdata_union",
        "REALDATA_CONTACT_EXPORT_ROOT": "data/corl_numbers/contact_exports",
        "AS_SINGLE_SLOT_MOTION_BASE": "data/corl_numbers/single_slot_motion_bank",
        "AS_EXPECTED_TOTAL": "2",
        "TEST_CAPTURE": str(capture),
    }
    return {
        "repo": repo,
        "wrapper": wrapper,
        "first": first,
        "second": second,
        "alias": alias,
        "contact": contact,
        "capture": capture,
        "env": env,
    }


def _run(fixture: dict[str, Path | dict[str, str]]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(fixture["wrapper"])],
        cwd=fixture["repo"],
        env=fixture["env"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_union_is_self_contained_content_addressed_and_atomically_republished(
    fixture_repo: dict[str, Path | dict[str, str]],
) -> None:
    result = _run(fixture_repo)
    assert result.returncode == 0, result.stdout + result.stderr

    alias = fixture_repo["alias"]
    assert isinstance(alias, Path)
    assert alias.is_symlink()
    first_generation = alias.resolve(strict=True)
    assert first_generation.parent.name == "_realdata_union_generations"
    assert len(first_generation.name) == 64
    assert (first_generation / MARKER_NAME).is_file()
    assert (first_generation / MANIFEST_NAME).is_file()
    assert not (first_generation / "clip_a.npz").is_symlink()
    assert (first_generation / "clip_a.npz").read_bytes() == b"motion-a-v1"
    assert stat.S_IMODE((first_generation / "clip_a.npz").stat().st_mode) == 0o444

    object_map = json.loads(
        (first_generation / "_clip_object_urdf_map.json").read_text(encoding="utf-8")
    )
    assert object_map["generation"]["id"] == first_generation.name
    for entry in object_map["clips"].values():
        for key in ("object_urdf_path", "object_mesh_path"):
            copied_asset = first_generation / entry[key]
            assert copied_asset.is_file()
            assert not copied_asset.is_symlink()
            assert first_generation in copied_asset.resolve(strict=True).parents

    manifest = json.loads((first_generation / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["generation_id"] == first_generation.name
    assert manifest["clip_count"] == 2
    assert all(len(record["sha256"]) == 64 for record in manifest["source_files"])

    capture = fixture_repo["capture"]
    assert isinstance(capture, Path)
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "data/corl_numbers/realdata_union",
        "data/corl_numbers/realdata_union/_clip_object_urdf_map.json",
        "2",
        "<unset>",
        "data/corl_numbers/single_slot_motion_bank",
        "g1-29dof-wbt-w-object-generalist",
        "0",
        "0",
    ]
    contact = fixture_repo["contact"]
    assert isinstance(contact, Path) and contact.is_dir()
    assert first_generation not in contact.parents

    # Identical inputs reuse exactly the same immutable generation.
    same = _run(fixture_repo)
    assert same.returncode == 0, same.stdout + same.stderr
    assert alias.resolve(strict=True) == first_generation

    # Source drift creates a new generation.  The previously published bytes
    # remain unchanged, which would not be true for a symlink-only union.
    first_source = fixture_repo["first"]
    assert isinstance(first_source, Path)
    (first_source / "clip_a.npz").write_bytes(b"motion-a-v2")
    changed = _run(fixture_repo)
    assert changed.returncode == 0, changed.stdout + changed.stderr
    second_generation = alias.resolve(strict=True)
    assert second_generation != first_generation
    assert (second_generation / "clip_a.npz").read_bytes() == b"motion-a-v2"
    assert (first_generation / "clip_a.npz").read_bytes() == b"motion-a-v1"


def test_failed_rebuild_never_replaces_last_complete_alias(
    fixture_repo: dict[str, Path | dict[str, str]],
) -> None:
    first = _run(fixture_repo)
    assert first.returncode == 0, first.stdout + first.stderr
    alias = fixture_repo["alias"]
    second_source = fixture_repo["second"]
    assert isinstance(alias, Path) and isinstance(second_source, Path)
    published = alias.resolve(strict=True)

    (second_source / "clip_b.npz").unlink()
    (second_source / "clip_a.npz").write_bytes(b"duplicate")
    payload = json.loads(
        (second_source / "_clip_object_urdf_map.json").read_text(encoding="utf-8")
    )
    payload["clips"]["clip_a"] = payload["clips"].pop("clip_b")
    (second_source / "_clip_object_urdf_map.json").write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )

    failed = _run(fixture_repo)
    assert failed.returncode != 0
    assert "duplicate clip id" in failed.stderr.lower()
    assert alias.resolve(strict=True) == published
    assert (published / "clip_a.npz").read_bytes() == b"motion-a-v1"
    assert not list(published.parent.glob(".*.staging-*"))


def test_concurrent_publishers_serialize_on_one_complete_generation(
    fixture_repo: dict[str, Path | dict[str, str]],
) -> None:
    processes = [
        subprocess.Popen(
            ["bash", str(fixture_repo["wrapper"])],
            cwd=fixture_repo["repo"],
            env=fixture_repo["env"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(4)
    ]
    outputs = [process.communicate(timeout=30) for process in processes]
    assert all(process.returncode == 0 for process in processes), outputs
    alias = fixture_repo["alias"]
    assert isinstance(alias, Path) and alias.is_symlink()
    generation = alias.resolve(strict=True)
    assert (generation / MARKER_NAME).is_file()
    assert (generation / "clip_a.npz").read_bytes() == b"motion-a-v1"
    assert len([path for path in generation.parent.iterdir() if path.is_dir()]) == 1
    assert not list(generation.parent.glob(".*.staging-*"))
    assert not list(alias.parent.glob(f".{alias.name}.publish-*"))


def test_existing_union_alias_escape_fails_closed_before_any_publication(
    fixture_repo: dict[str, Path | dict[str, str]],
    tmp_path: Path,
) -> None:
    alias = fixture_repo["alias"]
    assert isinstance(alias, Path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    alias.symlink_to(outside, target_is_directory=True)

    result = _run(fixture_repo)
    assert result.returncode != 0
    assert "escapes its owned generation root" in result.stderr
    assert alias.is_symlink() and alias.resolve(strict=True) == outside
    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    assert not (alias.parent / "_realdata_union_generations").exists()


@pytest.mark.parametrize("escape_kind", ["source_asset", "contact_root", "single_slot_base"])
def test_repo_internal_symlink_escape_is_rejected_before_union_writes(
    fixture_repo: dict[str, Path | dict[str, str]],
    tmp_path: Path,
    escape_kind: str,
) -> None:
    outside = tmp_path / f"outside-{escape_kind}"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    if escape_kind == "source_asset":
        first_source = fixture_repo["first"]
        assert isinstance(first_source, Path)
        escaped_asset = first_source / "objects" / "box.obj"
        escaped_asset.unlink()
        escaped_asset.symlink_to(sentinel)
    elif escape_kind == "contact_root":
        contact = fixture_repo["contact"]
        assert isinstance(contact, Path)
        contact.symlink_to(outside, target_is_directory=True)
    else:
        environ = fixture_repo["env"]
        assert isinstance(environ, dict)
        escaped_base = fixture_repo["repo"] / environ["AS_SINGLE_SLOT_MOTION_BASE"]
        assert isinstance(escaped_base, Path)
        escaped_base.symlink_to(outside, target_is_directory=True)

    result = _run(fixture_repo)
    assert result.returncode != 0
    assert "symlink" in result.stderr.lower()
    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    alias = fixture_repo["alias"]
    assert isinstance(alias, Path) and not os.path.lexists(alias)
    assert not (alias.parent / "_realdata_union_generations").exists()


def test_recognized_legacy_union_is_migrated_with_atomic_exchange(
    fixture_repo: dict[str, Path | dict[str, str]],
) -> None:
    alias = fixture_repo["alias"]
    assert isinstance(alias, Path)
    alias.mkdir()
    (alias / MARKER_NAME).write_text(
        "generated by corl_numbers/train_as_general_realdata.sh\n",
        encoding="utf-8",
    )
    legacy_map = {
        "clips": {"legacy": {}},
        "notes": "Generated symlink union for corl_numbers/train_as_general_realdata.sh.",
    }
    (alias / "_clip_object_urdf_map.json").write_text(
        json.dumps(legacy_map) + "\n", encoding="utf-8"
    )
    legacy_source = fixture_repo["first"]
    assert isinstance(legacy_source, Path)
    (alias / "legacy.npz").symlink_to(legacy_source / "clip_a.npz")

    result = _run(fixture_repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert alias.is_symlink()
    assert (alias / "clip_a.npz").read_bytes() == b"motion-a-v1"
    assert not list(alias.parent.glob(f".{alias.name}.publish-*"))


def test_legacy_union_with_mutable_output_is_preserved_and_rejected(
    fixture_repo: dict[str, Path | dict[str, str]],
) -> None:
    alias = fixture_repo["alias"]
    assert isinstance(alias, Path)
    alias.mkdir()
    (alias / MARKER_NAME).write_text(
        "generated by corl_numbers/train_as_general_realdata.sh\n",
        encoding="utf-8",
    )
    (alias / "_clip_object_urdf_map.json").write_text(
        json.dumps(
            {
                "clips": {"legacy": {}},
                "notes": "Generated symlink union for corl_numbers/train_as_general_realdata.sh.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    mutable_output = alias / "contact_export_from_retarget"
    mutable_output.mkdir()
    sentinel = mutable_output / "contact.npz"
    sentinel.write_bytes(b"must survive")

    result = _run(fixture_repo)
    assert result.returncode != 0
    assert "non-view data" in result.stderr
    assert not alias.is_symlink()
    assert sentinel.read_bytes() == b"must survive"
    assert not (alias.parent / "_realdata_union_generations").exists()
