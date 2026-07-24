from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARE_SCRIPT = REPO_ROOT / "scripts" / "prepare_teacher_as_realmesh_rollout.py"
CP_TAO_SCRIPT = REPO_ROOT / "cp_tao.sh"


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_teacher_as_realmesh_rollout_safe_paths", PREPARE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_box_asset(bank: Path, *, object_name: str = "box") -> dict[str, object]:
    asset_dir = bank / "objects" / object_name
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "model.obj").write_text("o box\n", encoding="utf-8")
    (asset_dir / "model.urdf").write_text(
        '<robot name="box"><link name="object"><visual><geometry>'
        '<mesh filename="model.obj"/></geometry></visual></link></robot>\n',
        encoding="utf-8",
    )
    return {
        "object_name": object_name,
        "object_size": [1.0, 1.0, 1.0],
        "object_urdf_path": f"objects/{object_name}/model.urdf",
        "object_mesh_path": f"objects/{object_name}/model.obj",
    }


def _identity(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}


def _directory_manifest(path: Path) -> list[dict[str, object]]:
    return [
        {"path": candidate.relative_to(path).as_posix(), **_identity(candidate)}
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file() and not candidate.is_symlink()
    ]


def _write_rollout_motion(
    path: Path,
    *,
    object_name: str,
    object_urdf_path: Path,
    variant: float = 0.0,
    nan_field: str | None = None,
    short_field: str | None = None,
    body_quaternion_w: float = 1.0,
    object_quaternion_w: float = 1.0,
) -> None:
    trajectory_length = 3
    num_joints = 2
    num_bodies = 2
    payload: dict[str, np.ndarray] = {
        "fps": np.asarray([30], dtype=np.int32),
        "body_names": np.asarray(["body_a", "body_b"]),
        "joint_names": np.asarray(["joint_a", "joint_b"]),
        "joint_pos": np.full((trajectory_length, num_joints + 7), variant, dtype=np.float32),
        "joint_vel": np.zeros((trajectory_length, num_joints + 6), dtype=np.float32),
        "body_pos_w": np.zeros((trajectory_length, num_bodies, 3), dtype=np.float32),
        "body_quat_w": np.zeros((trajectory_length, num_bodies, 4), dtype=np.float32),
        "body_lin_vel_w": np.zeros((trajectory_length, num_bodies, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((trajectory_length, num_bodies, 3), dtype=np.float32),
        "object_name": np.asarray(object_name),
        "object_urdf_path": np.asarray(str(object_urdf_path)),
        "object_size": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        "object_pos_w": np.zeros((trajectory_length, 3), dtype=np.float32),
        "object_quat_w": np.zeros((trajectory_length, 4), dtype=np.float32),
        "object_lin_vel_w": np.zeros((trajectory_length, 3), dtype=np.float32),
        "object_ang_vel_w": np.zeros((trajectory_length, 3), dtype=np.float32),
    }
    payload["body_quat_w"][..., 0] = body_quaternion_w
    payload["object_quat_w"][..., 0] = object_quaternion_w
    if nan_field is not None:
        payload[nan_field].reshape(-1)[0] = np.nan
    if short_field is not None:
        payload[short_field] = payload[short_field][:-1]
    np.savez_compressed(path, **payload)


def _write_shard_output_manifest(
    *,
    shard_output: Path,
    shard_name: str,
    prepared_manifest_sha256: str,
    checkpoint_sha256: str,
    expected_clip_ids: list[str],
    object_map_payload: dict[str, object],
) -> None:
    clip_outputs: dict[str, object] = {}
    for clip_id in expected_clip_ids:
        matches = [
            candidate
            for candidate in (shard_output / "clips").iterdir()
            if candidate.name == clip_id or candidate.name.split("_", 1)[-1] == clip_id
        ]
        assert len(matches) == 1
        clip_outputs[clip_id] = {
            "directory_name": matches[0].name,
            "files": _directory_manifest(matches[0]),
        }
    exporter = REPO_ROOT / "src" / "holosoma" / "holosoma" / "export_teacher_box_contacts.py"
    payload = {
        "schema": "teacher_realmesh_rollout_shard_output_v1",
        "shard_name": shard_name,
        "prepared_manifest_sha256": prepared_manifest_sha256,
        "teacher_checkpoint_sha256": checkpoint_sha256,
        "expected_clip_ids": expected_clip_ids,
        "object_map_payload": object_map_payload,
        "exporter_code": _identity(exporter),
        "summary_csv": _identity(shard_output / "summary.csv"),
        "summary_json": _identity(shard_output / "summary.json"),
        "success_clips": _identity(shard_output / "success_clips.txt"),
        "failure_clips": _identity(shard_output / "failure_clips.txt"),
        "object_map_output": _identity(shard_output / "motion_bank" / "_clip_object_urdf_map.json"),
        "motion_outputs": {
            clip_id: _identity(shard_output / "motion_bank" / f"{clip_id}.npz")
            for clip_id in expected_clip_ids
        },
        "clip_outputs": clip_outputs,
    }
    (shard_output / "shard_output_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def prepare_module():
    return _load_prepare_module()


def test_destructive_path_requires_strict_owned_descendant(tmp_path: Path, prepare_module) -> None:
    allowed = tmp_path / "owned"
    allowed.mkdir()
    safe = allowed / "run-001"

    assert prepare_module._validate_destructive_path(safe, allowed_root=allowed, label="test target") == safe

    for unsafe in (allowed, tmp_path / "outside", Path("/"), Path(".")):
        with pytest.raises(SystemExit):
            prepare_module._validate_destructive_path(unsafe, allowed_root=allowed, label="test target")


def test_destructive_path_rejects_symlink_escape_and_alias(tmp_path: Path, prepare_module) -> None:
    allowed = tmp_path / "owned"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (allowed / "escape").symlink_to(outside, target_is_directory=True)
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="symlink"):
        prepare_module._validate_destructive_path(
            allowed / "escape" / "victim",
            allowed_root=allowed,
            label="test target",
        )

    assert sentinel.read_text(encoding="utf-8") == "keep\n"

    allowed_alias = tmp_path / "owned-alias"
    allowed_alias.symlink_to(allowed, target_is_directory=True)
    with pytest.raises(SystemExit, match="allowed root"):
        prepare_module._validate_destructive_path(
            allowed_alias / "victim",
            allowed_root=allowed_alias,
            label="test target",
        )


def test_force_cannot_remove_unmarked_directory(tmp_path: Path, prepare_module) -> None:
    allowed = tmp_path / "owned"
    target = allowed / "user-bank"
    target.mkdir(parents=True)
    sentinel = target / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="force cannot bypass"):
        prepare_module._safe_remove_generated(
            target,
            allowed_root=allowed,
            label="test bank",
            force=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_generated_cleanup_requires_marker_and_preserves_launcher_bootstrap(tmp_path: Path, prepare_module) -> None:
    allowed = tmp_path / "owned"
    generated = allowed / "generated"
    generated.mkdir(parents=True)
    (generated / prepare_module.MARKER_NAME).write_text("generated\n", encoding="utf-8")
    (generated / "old.txt").write_text("old\n", encoding="utf-8")
    (generated / "old-dir").mkdir()

    resolved = prepare_module._safe_remove_generated(
        generated,
        allowed_root=allowed,
        label="test generated bank",
        force=False,
    )
    assert resolved == generated
    assert list(generated.iterdir()) == []

    bootstrap = allowed / "bootstrap"
    bootstrap.mkdir()
    stdout_path = bootstrap / "prepare_stdout.json"
    stdout_path.write_text("", encoding="utf-8")
    prepare_module._safe_remove_generated(
        bootstrap,
        allowed_root=allowed,
        label="test shard root",
        force=False,
        preserve_names=frozenset({"prepare_stdout.json"}),
    )
    assert list(bootstrap.iterdir()) == [stdout_path]


def test_copy_target_cannot_escape_generated_root(tmp_path: Path, prepare_module) -> None:
    allowed = tmp_path / "owned"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    source = tmp_path / "source.txt"
    source.write_text("source\n", encoding="utf-8")
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        prepare_module._copy_or_symlink(
            source,
            allowed / ".." / "outside" / "sentinel.txt",
            symlink=False,
            allowed_root=allowed,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_stable_source_map_read_rejects_in_place_race(
    tmp_path: Path,
    prepare_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_map = tmp_path / "map.json"
    source_map.write_bytes(b"x" * (5 * 1024 * 1024))
    original_read = prepare_module.os.read
    mutated = False

    def racing_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, count)
        if chunk and not mutated:
            mutated = True
            with source_map.open("r+b") as stream:
                stream.seek(0)
                stream.write(b"y")
                stream.flush()
                os.fsync(stream.fileno())
        return chunk

    monkeypatch.setattr(prepare_module.os, "read", racing_read)
    with pytest.raises(SystemExit, match="changed while it was read"):
        prepare_module._stable_regular_file_bytes(source_map, label="source object map")


def test_contact_copy_rejects_source_change_after_prehash(
    tmp_path: Path,
    prepare_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    generation = tmp_path / "generation"
    source.mkdir()
    generation.mkdir()
    payload = source / "teacher_rollout_reference.npz"
    payload.write_bytes(b"before")
    expected = prepare_module._directory_content_manifest(source)
    original_copytree = prepare_module.shutil.copytree

    def racing_copytree(src: Path, dst: Path, *args, **kwargs):
        payload.write_bytes(b"after")
        return original_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(prepare_module.shutil, "copytree", racing_copytree)
    with pytest.raises(SystemExit, match="changed while copying"):
        prepare_module._copy_clip_dir_with_bank_metadata(
            source,
            destination,
            object_entry=None,
            generation_root=generation,
            motion_bank_dir=generation / "motion_bank",
            expected_input_manifest=expected,
        )


def test_existing_generation_must_match_expected_payload_and_content(tmp_path: Path, prepare_module) -> None:
    generation = tmp_path / "generation"
    generation.mkdir()
    (generation / prepare_module.MARKER_NAME).write_text("generated\n", encoding="utf-8")
    (generation / "output.bin").write_bytes(b"original")
    content_files = prepare_module._directory_content_manifest(
        generation,
        excluded_relative_paths=prepare_module.PUBLICATION_ROOT_METADATA_PATHS,
    )
    payload = {
        "schema": "fixture",
        "input": "authenticated",
        "summary_counts": {"total": 0, "success": 0, "failure": 0},
        "all_summary_rows": [],
        "successful_object_map": {},
        "final_content_manifest_sha256": prepare_module._canonical_json_sha256(content_files),
        "final_content_manifest": content_files,
    }
    publication_id = prepare_module._canonical_json_sha256(payload)
    content_addressed_generation = tmp_path / publication_id
    generation.rename(content_addressed_generation)
    generation = content_addressed_generation
    manifest = prepare_module._generation_manifest_from_publication_payload(publication_id, payload)
    (generation / "realmesh_rollout_manifest.json").write_bytes(
        prepare_module._pretty_json_bytes(manifest)
    )
    merge_manifest = {
        **manifest,
        "output_generation": str(generation),
        "target_generation": str(generation),
        "compatibility_view": "OUTPUT_ROOT/current -> TARGET_BANK -> target_generation",
    }
    (generation / "merge_manifest.json").write_bytes(
        prepare_module._pretty_json_bytes(merge_manifest)
    )
    integrity = {
        "schema": "teacher_realmesh_rollout_generation_integrity_v2",
        "publication_id": publication_id,
        "final_content_manifest_sha256": payload["final_content_manifest_sha256"],
        "files": content_files,
    }
    integrity_path = generation / prepare_module.INTEGRITY_MANIFEST_NAME
    integrity_path.write_bytes(prepare_module._pretty_json_bytes(integrity))

    prepare_module._validate_generation_integrity(
        generation,
        publication_id=publication_id,
        expected_manifest=manifest,
        expected_content_files=content_files,
    )

    expected_files = [record for record in content_files if record["path"] != "output.bin"]
    with pytest.raises(SystemExit, match="differs from this publication"):
        prepare_module._validate_generation_integrity(
            generation,
            publication_id=publication_id,
            expected_manifest=manifest,
            expected_content_files=expected_files,
        )

    # The publication ID must still reject a coordinated rewrite of both one
    # output and the independently stored integrity file.
    (generation / "output.bin").write_bytes(b"tampered")
    tampered_files = prepare_module._directory_content_manifest(
        generation,
        excluded_relative_paths=prepare_module.PUBLICATION_ROOT_METADATA_PATHS,
    )
    integrity_path.write_text(
        json.dumps(
            {
                **integrity,
                "final_content_manifest_sha256": prepare_module._canonical_json_sha256(tampered_files),
                "files": tampered_files,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="differs from its publication payload"):
        prepare_module._validate_generation_integrity(generation, publication_id=publication_id)

    (generation / "output.bin").write_bytes(b"original")
    integrity_path.write_bytes(prepare_module._pretty_json_bytes(integrity))
    tampered_manifest = {**manifest, "success_count": 999}
    (generation / "realmesh_rollout_manifest.json").write_bytes(
        prepare_module._pretty_json_bytes(tampered_manifest)
    )
    (generation / "merge_manifest.json").write_bytes(
        prepare_module._pretty_json_bytes(
            {
                **tampered_manifest,
                "output_generation": str(generation),
                "target_generation": str(generation),
                "compatibility_view": "OUTPUT_ROOT/current -> TARGET_BANK -> target_generation",
            }
        )
    )
    with pytest.raises(SystemExit, match="canonical payload-derived"):
        prepare_module._validate_generation_integrity(generation, publication_id=publication_id)

    (generation / "realmesh_rollout_manifest.json").write_bytes(
        prepare_module._pretty_json_bytes(manifest)
    )
    (generation / "merge_manifest.json").write_bytes(
        prepare_module._pretty_json_bytes(merge_manifest)
    )
    poisoned_manifest = dict(manifest)
    poisoned_manifest["publication_payload"] = {"schema": "fixture", "input": "forged"}
    (generation / "realmesh_rollout_manifest.json").write_text(
        json.dumps(poisoned_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="payload digest mismatch"):
        prepare_module._validate_generation_integrity(generation, publication_id=publication_id)


def test_prepare_shards_creates_read_only_self_contained_snapshot(
    tmp_path: Path,
    prepare_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_bank = tmp_path / "source-bank"
    source_bank.mkdir()
    (source_bank / "box_clip.npz").write_bytes(b"fixture")
    object_entry = _write_box_asset(source_bank)
    source_map = source_bank / "_clip_object_urdf_map.json"
    source_map.write_text(
        json.dumps({"clips": {"box_clip": object_entry}}) + "\n",
        encoding="utf-8",
    )
    shard_namespace = tmp_path / "data" / "_teacher_rollout_shards"
    shard_root = shard_namespace / "run-001"
    shard_namespace.mkdir(parents=True)
    monkeypatch.setattr(prepare_module, "TEACHER_ROLLOUT_SHARD_ROOT", shard_namespace)

    prepare_module.prepare_shards(
        SimpleNamespace(
            source_bank=source_bank,
            source_map=source_map,
            shard_root=shard_root,
            allowed_categories="box",
            exclude_clips="",
            expected_total=1,
            num_shards=1,
            per_gpu_envs=1,
        )
    )

    assert (shard_root / prepare_module.MARKER_NAME).is_file()
    assert not (shard_root / "shard_00" / "box_clip.npz").is_symlink()
    assert (shard_root / "shard_00" / "objects").is_symlink()
    assert (shard_root / "shard_00" / "objects").resolve() == shard_root / "_asset_snapshot" / "objects"
    assert (shard_root / "manifest.json").is_file()
    assert not (shard_root.stat().st_mode & 0o222)
    assert not ((shard_root / "shard_00" / "box_clip.npz").stat().st_mode & 0o222)


def test_prepare_shards_dry_run_never_mutates_existing_shard_root(
    tmp_path: Path,
    prepare_module,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_bank = tmp_path / "source-bank"
    source_bank.mkdir()
    (source_bank / "box_clip.npz").write_bytes(b"fixture")
    object_entry = _write_box_asset(source_bank)
    source_map = source_bank / "_clip_object_urdf_map.json"
    source_map.write_text(json.dumps({"clips": {"box_clip": object_entry}}) + "\n", encoding="utf-8")
    shard_namespace = tmp_path / "data" / "_teacher_rollout_shards"
    shard_root = shard_namespace / "run-001"
    shard_root.mkdir(parents=True)
    sentinel = shard_root / "user-sentinel.txt"
    sentinel.write_text("must remain byte-identical\n", encoding="utf-8")
    monkeypatch.setattr(prepare_module, "TEACHER_ROLLOUT_SHARD_ROOT", shard_namespace)

    prepare_module.prepare_shards(
        SimpleNamespace(
            source_bank=source_bank,
            source_map=source_map,
            shard_root=shard_root,
            allowed_categories="box",
            exclude_clips="",
            expected_total=1,
            num_shards=1,
            per_gpu_envs=1,
            dry_run=True,
        )
    )

    plan = json.loads(capsys.readouterr().out)
    assert plan["selected_clip_count"] == 1
    assert list(shard_root.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding="utf-8") == "must remain byte-identical\n"


@pytest.mark.parametrize(
    "bad_path",
    ["/tmp/external.urdf", "package://objects/model.urdf", "../outside/model.urdf", "model.urdf"],
)
def test_prepare_rejects_non_self_contained_object_paths_before_writes(
    tmp_path: Path,
    prepare_module,
    monkeypatch: pytest.MonkeyPatch,
    bad_path: str,
) -> None:
    source_bank = tmp_path / "source-bank"
    source_bank.mkdir()
    (source_bank / "box_clip.npz").write_bytes(b"fixture")
    source_map = source_bank / "_clip_object_urdf_map.json"
    source_map.write_text(
        json.dumps(
            {
                "clips": {
                    "box_clip": {
                        "object_name": "box",
                        "object_size": [1.0, 1.0, 1.0],
                        "object_urdf_path": bad_path,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    shard_namespace = tmp_path / "data" / "_teacher_rollout_shards"
    shard_root = shard_namespace / "run-001"
    shard_namespace.mkdir(parents=True)
    monkeypatch.setattr(prepare_module, "TEACHER_ROLLOUT_SHARD_ROOT", shard_namespace)

    with pytest.raises(SystemExit):
        prepare_module.prepare_shards(
            SimpleNamespace(
                source_bank=source_bank,
                source_map=source_map,
                shard_root=shard_root,
                allowed_categories="box",
                exclude_clips="",
                expected_total=1,
                num_shards=1,
                per_gpu_envs=1,
                dry_run=False,
                rollout_contract_json="{}",
            )
        )
    assert not shard_root.exists()


def test_prepare_rejects_symlinked_object_asset_before_writes(
    tmp_path: Path,
    prepare_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_bank = tmp_path / "source-bank"
    source_bank.mkdir()
    (source_bank / "box_clip.npz").write_bytes(b"fixture")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model.obj").write_text("o box\n", encoding="utf-8")
    (outside / "model.urdf").write_text(
        '<robot name="box"><link name="o"><visual><geometry><mesh filename="model.obj"/>'
        "</geometry></visual></link></robot>\n",
        encoding="utf-8",
    )
    (source_bank / "objects").symlink_to(outside, target_is_directory=True)
    source_map = source_bank / "_clip_object_urdf_map.json"
    source_map.write_text(
        json.dumps(
            {
                "clips": {
                    "box_clip": {
                        "object_name": "box",
                        "object_size": [1.0, 1.0, 1.0],
                        "object_urdf_path": "objects/model.urdf",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    shard_namespace = tmp_path / "data" / "_teacher_rollout_shards"
    shard_namespace.mkdir(parents=True)
    shard_root = shard_namespace / "run-001"
    monkeypatch.setattr(prepare_module, "TEACHER_ROLLOUT_SHARD_ROOT", shard_namespace)

    with pytest.raises(SystemExit):
        prepare_module.prepare_shards(
            SimpleNamespace(
                source_bank=source_bank,
                source_map=source_map,
                shard_root=shard_root,
                allowed_categories="box",
                exclude_clips="",
                expected_total=1,
                num_shards=1,
                per_gpu_envs=1,
                dry_run=False,
                rollout_contract_json="{}",
            )
        )
    assert not shard_root.exists()


def test_merge_replaces_only_marked_outputs_inside_owned_roots(
    tmp_path: Path,
    prepare_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data" / "ds_as_data"
    shard_namespace = data_root / "_teacher_rollout_shards"
    output_namespace = tmp_path / "outputs" / "teacher_as_contacts"
    source_bank = data_root / "source-bank"
    target_bank = data_root / "target-bank"
    output_root = output_namespace / "run-001"
    source_bank.mkdir(parents=True)
    clip_ids = ["box_clip", "box_fail"]
    for clip_id in clip_ids:
        (source_bank / f"{clip_id}.npz").write_bytes(f"source-{clip_id}".encode())
    object_entry = _write_box_asset(source_bank)
    clip_map_payload: dict[str, object] = {"clips": {clip_id: object_entry for clip_id in clip_ids}}
    clip_map = json.dumps(clip_map_payload, sort_keys=True) + "\n"
    (source_bank / "_clip_object_urdf_map.json").write_text(clip_map, encoding="utf-8")
    teacher_checkpoint = tmp_path / "teacher.pt"
    teacher_checkpoint.write_bytes(b"teacher checkpoint fixture")
    teacher_sha256 = hashlib.sha256(teacher_checkpoint.read_bytes()).hexdigest()

    monkeypatch.setattr(prepare_module, "DATA_BANK_ROOT", data_root)
    monkeypatch.setattr(prepare_module, "TEACHER_ROLLOUT_SHARD_ROOT", shard_namespace)
    monkeypatch.setattr(prepare_module, "TEACHER_ROLLOUT_OUTPUT_ROOT", output_namespace)
    prepared_root = shard_namespace / "run-001-input"
    prepare_module.prepare_shards(
        SimpleNamespace(
            source_bank=source_bank,
            source_map=source_bank / "_clip_object_urdf_map.json",
            shard_root=prepared_root,
            allowed_categories="box",
            exclude_clips="",
            expected_total=2,
            num_shards=1,
            per_gpu_envs=2,
            dry_run=False,
            rollout_contract_json="{}",
        )
    )
    prepared_manifest = prepared_root / "manifest.json"
    prepared_manifest_sha256 = hashlib.sha256(prepared_manifest.read_bytes()).hexdigest()
    shard_input = prepared_root / "shard_00"
    prepared_urdf = (shard_input / str(object_entry["object_urdf_path"])).resolve()

    shard_output = output_root / "shards" / "shard_00"
    clips_src = shard_output / "clips"
    motion_src = shard_output / "motion_bank"
    motion_src.mkdir(parents=True)
    for clip_index, clip_id in enumerate(clip_ids):
        clip_dir = clips_src / f"{clip_index:04d}_{clip_id}"
        clip_dir.mkdir(parents=True)
        (clip_dir / "teacher_rollout_reference.npz").write_bytes(f"reference-{clip_id}".encode())
        _write_rollout_motion(
            motion_src / f"{clip_id}.npz",
            object_name="box",
            object_urdf_path=prepared_urdf,
        )
    (motion_src / "_clip_object_urdf_map.json").write_text(clip_map, encoding="utf-8")
    summary_rows = [
        {
            "clip_id": "box_clip",
            "success": "true",
            "status": "ok",
            "object_name": "box",
            "object_urdf_path": str(prepared_urdf),
            "primitive_extent_x": "1.0",
            "primitive_extent_y": "1.0",
            "primitive_extent_z": "1.0",
            "num_steps": "3",
        },
        {
            "clip_id": "box_fail",
            "success": "false",
            "status": "failed",
            "object_name": "box",
            "object_urdf_path": str(prepared_urdf),
            "primitive_extent_x": "1.0",
            "primitive_extent_y": "1.0",
            "primitive_extent_z": "1.0",
            "num_steps": "3",
        },
    ]
    with (shard_output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    summary_json_payload = {
        "checkpoint": str(teacher_checkpoint),
        "source_checkpoint_sha256": teacher_sha256,
        "source_training_provenance": None,
        "saved_wandb_path": None,
        "num_clips": 2,
        "num_success": 1,
        "num_failure": 1,
        "num_envs": 2,
        "export_config": {"output_dir": str(shard_output), "min_contact_frames": 10},
    }
    (shard_output / "summary.json").write_text(json.dumps(summary_json_payload), encoding="utf-8")
    (shard_output / "success_clips.txt").write_text("box_clip\n", encoding="utf-8")
    (shard_output / "failure_clips.txt").write_text("box_fail\n", encoding="utf-8")
    _write_shard_output_manifest(
        shard_output=shard_output,
        shard_name="shard_00",
        prepared_manifest_sha256=prepared_manifest_sha256,
        checkpoint_sha256=teacher_sha256,
        expected_clip_ids=clip_ids,
        object_map_payload=clip_map_payload,
    )
    # Normalization is deterministic for both successful and failed rollout
    # trajectories and internally verifies that every non-URDF array is bitwise
    # unchanged in dtype, shape, and value.
    normalized_root_bytes: dict[str, bytes] = {}
    for clip_id in clip_ids:
        first = prepare_module._normalized_rollout_motion_npz_bytes(
            motion_src / f"{clip_id}.npz",
            published_urdf_path=str(object_entry["object_urdf_path"]),
            expected_source_identity=_identity(motion_src / f"{clip_id}.npz"),
        )
        second = prepare_module._normalized_rollout_motion_npz_bytes(
            motion_src / f"{clip_id}.npz",
            published_urdf_path=str(object_entry["object_urdf_path"]),
            expected_source_identity=_identity(motion_src / f"{clip_id}.npz"),
        )
        assert first == second
        normalized_root_bytes[clip_id] = first

    # Exercise the one-time atomic migration from the legacy generated
    # directory layout to the immutable generation alias.
    target_bank.mkdir()
    (target_bank / prepare_module.MARKER_NAME).write_text("legacy generated bank\n", encoding="utf-8")
    (target_bank / "retired.txt").write_text("old\n", encoding="utf-8")
    args = SimpleNamespace(
        output_root=output_root,
        target_bank=target_bank,
        source_bank=source_bank,
        prepared_manifest=prepared_manifest,
        prepared_manifest_sha256=prepared_manifest_sha256,
        contact_export_name="contact_export",
        expected_teacher_checkpoint_sha256=teacher_sha256,
        teacher_checkpoint_path=teacher_checkpoint,
        teacher_checkpoint_source="fixture://teacher",
        force=True,
        save_visualization=False,
        save_visualization_preview_png=False,
        save_visualization_face_heatmap_png=False,
    )

    prepare_module.merge_outputs(args)
    first_generation = target_bank.resolve()
    first_manifest = json.loads((target_bank / "realmesh_rollout_manifest.json").read_text(encoding="utf-8"))
    first_publication_id = first_manifest["publication_id"]
    # A second run exercises marker-authorized cleanup of all three generated
    # destinations without granting --force access to arbitrary directories;
    # identical inputs must reuse the exact immutable generation.
    prepare_module.merge_outputs(args)

    assert target_bank.is_symlink()
    assert target_bank.resolve() == first_generation
    assert (target_bank / "realmesh_rollout_manifest.json").is_file()
    assert (target_bank / prepare_module.INTEGRITY_MANIFEST_NAME).is_file()
    assert (output_root / "clips").is_symlink()
    assert (output_root / "motion_bank").is_symlink()
    assert os.readlink(output_root / "clips") == "current/clips"
    assert os.readlink(output_root / "motion_bank") == "current/motion_bank"
    assert (output_root / "current").resolve() == target_bank.resolve()
    assert (output_root / "clips" / prepare_module.MARKER_NAME).is_file()
    assert (output_root / "motion_bank" / "box_clip.npz").is_file()
    assert (target_bank / prepare_module.MARKER_NAME).is_file()
    assert (target_bank / "box_clip.npz").is_file()
    assert not (target_bank / "box_fail.npz").exists()
    assert (target_bank / "contact_export" / "clips" / "0000_box_clip").is_dir()
    root_map = json.loads((target_bank / "_clip_object_urdf_map.json").read_text())
    slot_map = json.loads((target_bank / "_single_slot_motion_bank" / "_clip_object_urdf_map.json").read_text())
    output_map = json.loads((target_bank / "motion_bank" / "_clip_object_urdf_map.json").read_text())
    assert root_map["clips"]["box_clip"]["object_urdf_path"] == "objects/box/model.urdf"
    assert slot_map["clips"]["box_clip"]["object_urdf_path"] == "../objects/box/model.urdf"
    assert output_map == slot_map
    with np.load(target_bank / "box_clip.npz", allow_pickle=False) as payload:
        assert str(np.asarray(payload["object_urdf_path"]).reshape(-1)[0]) == "objects/box/model.urdf"
    with np.load(target_bank / "_single_slot_motion_bank" / "box_clip.npz", allow_pickle=False) as payload:
        assert str(np.asarray(payload["object_urdf_path"]).reshape(-1)[0]) == "../objects/box/model.urdf"
    assert (target_bank / "box_clip.npz").read_bytes() == normalized_root_bytes["box_clip"]
    assert json.loads((target_bank / "all_rollout_motion_inputs.json").read_text()) == {
        clip_id: _identity(motion_src / f"{clip_id}.npz") for clip_id in clip_ids
    }

    # A changed successful motion produces a new generation and atomically
    # switches the stable alias while retaining the old content-addressed one.
    first_motion_bytes = (target_bank / "box_clip.npz").read_bytes()
    _write_rollout_motion(
        motion_src / "box_clip.npz",
        object_name="box",
        object_urdf_path=prepared_urdf,
        variant=1.0,
    )
    _write_shard_output_manifest(
        shard_output=shard_output,
        shard_name="shard_00",
        prepared_manifest_sha256=prepared_manifest_sha256,
        checkpoint_sha256=teacher_sha256,
        expected_clip_ids=clip_ids,
        object_map_payload=clip_map_payload,
    )
    prepare_module.merge_outputs(args)
    second_generation = target_bank.resolve()
    second_manifest = json.loads((target_bank / "realmesh_rollout_manifest.json").read_text(encoding="utf-8"))
    assert second_generation != first_generation
    assert second_manifest["publication_id"] != first_publication_id
    assert first_generation.is_dir()
    assert (first_generation / "box_clip.npz").read_bytes() == first_motion_bytes
    expected_second_motion = prepare_module._normalized_rollout_motion_npz_bytes(
        motion_src / "box_clip.npz",
        published_urdf_path="objects/box/model.urdf",
        expected_source_identity=_identity(motion_src / "box_clip.npz"),
    )
    assert (second_generation / "box_clip.npz").read_bytes() == expected_second_motion

    # Both successful and failed trajectories are bound by the exporter-side
    # shard manifest, even though only successful trajectories are published.
    _write_rollout_motion(
        motion_src / "box_clip.npz",
        object_name="box",
        object_urdf_path=prepared_urdf,
        variant=2.0,
    )
    with pytest.raises(SystemExit, match="changed after the shard output manifest"):
        prepare_module.merge_outputs(args)
    assert target_bank.resolve() == second_generation
    _write_rollout_motion(
        motion_src / "box_clip.npz",
        object_name="box",
        object_urdf_path=prepared_urdf,
        variant=1.0,
    )
    _write_shard_output_manifest(
        shard_output=shard_output,
        shard_name="shard_00",
        prepared_manifest_sha256=prepared_manifest_sha256,
        checkpoint_sha256=teacher_sha256,
        expected_clip_ids=clip_ids,
        object_map_payload=clip_map_payload,
    )
    _write_rollout_motion(
        motion_src / "box_fail.npz",
        object_name="box",
        object_urdf_path=prepared_urdf,
        variant=2.0,
    )
    with pytest.raises(SystemExit, match="changed after the shard output manifest"):
        prepare_module.merge_outputs(args)
    assert target_bank.resolve() == second_generation

    # A freshly committed but scientifically invalid shard is still rejected.
    _write_rollout_motion(
        motion_src / "box_fail.npz",
        object_name="box",
        object_urdf_path=prepared_urdf,
        nan_field="object_pos_w",
    )
    _write_shard_output_manifest(
        shard_output=shard_output,
        shard_name="shard_00",
        prepared_manifest_sha256=prepared_manifest_sha256,
        checkpoint_sha256=teacher_sha256,
        expected_clip_ids=clip_ids,
        object_map_payload=clip_map_payload,
    )
    with pytest.raises(SystemExit, match="NaN or infinity"):
        prepare_module.merge_outputs(args)
    _write_rollout_motion(
        motion_src / "box_fail.npz",
        object_name="box",
        object_urdf_path=prepared_urdf,
        body_quaternion_w=0.0,
    )
    _write_shard_output_manifest(
        shard_output=shard_output,
        shard_name="shard_00",
        prepared_manifest_sha256=prepared_manifest_sha256,
        checkpoint_sha256=teacher_sha256,
        expected_clip_ids=clip_ids,
        object_map_payload=clip_map_payload,
    )
    with pytest.raises(SystemExit, match="body_quat_w must contain unit quaternions"):
        prepare_module.merge_outputs(args)
    _write_rollout_motion(
        motion_src / "box_fail.npz",
        object_name="box",
        object_urdf_path=prepared_urdf,
        object_quaternion_w=1.5,
    )
    _write_shard_output_manifest(
        shard_output=shard_output,
        shard_name="shard_00",
        prepared_manifest_sha256=prepared_manifest_sha256,
        checkpoint_sha256=teacher_sha256,
        expected_clip_ids=clip_ids,
        object_map_payload=clip_map_payload,
    )
    with pytest.raises(SystemExit, match="object_quat_w must contain unit quaternions"):
        prepare_module.merge_outputs(args)
    _write_rollout_motion(
        motion_src / "box_fail.npz",
        object_name="box",
        object_urdf_path=prepared_urdf,
        short_field="joint_vel",
    )
    _write_shard_output_manifest(
        shard_output=shard_output,
        shard_name="shard_00",
        prepared_manifest_sha256=prepared_manifest_sha256,
        checkpoint_sha256=teacher_sha256,
        expected_clip_ids=clip_ids,
        object_map_payload=clip_map_payload,
    )
    with pytest.raises(SystemExit, match="joint_vel shape differs"):
        prepare_module.merge_outputs(args)
    _write_rollout_motion(
        motion_src / "box_fail.npz",
        object_name="wrong-object",
        object_urdf_path=prepared_urdf,
    )
    _write_shard_output_manifest(
        shard_output=shard_output,
        shard_name="shard_00",
        prepared_manifest_sha256=prepared_manifest_sha256,
        checkpoint_sha256=teacher_sha256,
        expected_clip_ids=clip_ids,
        object_map_payload=clip_map_payload,
    )
    with pytest.raises(SystemExit, match="object_name differs"):
        prepare_module.merge_outputs(args)
    _write_rollout_motion(
        motion_src / "box_fail.npz",
        object_name="box",
        object_urdf_path=prepared_urdf,
    )
    wrong_map_payload = json.loads(json.dumps(clip_map_payload))
    wrong_map_payload["clips"]["box_fail"]["object_name"] = "wrong-object"
    (motion_src / "_clip_object_urdf_map.json").write_text(
        json.dumps(wrong_map_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_shard_output_manifest(
        shard_output=shard_output,
        shard_name="shard_00",
        prepared_manifest_sha256=prepared_manifest_sha256,
        checkpoint_sha256=teacher_sha256,
        expected_clip_ids=clip_ids,
        object_map_payload=clip_map_payload,
    )
    with pytest.raises(SystemExit, match="not exactly equal to its prepared subset"):
        prepare_module.merge_outputs(args)
    (motion_src / "_clip_object_urdf_map.json").write_text(clip_map, encoding="utf-8")
    _write_shard_output_manifest(
        shard_output=shard_output,
        shard_name="shard_00",
        prepared_manifest_sha256=prepared_manifest_sha256,
        checkpoint_sha256=teacher_sha256,
        expected_clip_ids=clip_ids,
        object_map_payload=clip_map_payload,
    )

    bad_counts = json.loads((shard_output / "summary.json").read_text(encoding="utf-8"))
    bad_counts["num_clips"] = 3
    (shard_output / "summary.json").write_text(json.dumps(bad_counts), encoding="utf-8")
    _write_shard_output_manifest(
        shard_output=shard_output,
        shard_name="shard_00",
        prepared_manifest_sha256=prepared_manifest_sha256,
        checkpoint_sha256=teacher_sha256,
        expected_clip_ids=clip_ids,
        object_map_payload=clip_map_payload,
    )
    with pytest.raises(SystemExit, match="counts disagree"):
        prepare_module.merge_outputs(args)
    assert target_bank.resolve() == second_generation
    bad_counts["num_clips"] = 2
    (shard_output / "summary.json").write_text(json.dumps(bad_counts), encoding="utf-8")
    _write_shard_output_manifest(
        shard_output=shard_output,
        shard_name="shard_00",
        prepared_manifest_sha256=prepared_manifest_sha256,
        checkpoint_sha256=teacher_sha256,
        expected_clip_ids=clip_ids,
        object_map_payload=clip_map_payload,
    )

    # An unexpected extra shard must fail before moving the current alias.
    shard_01 = output_root / "shards" / "shard_01"
    shutil.copytree(shard_output, shard_01)
    with pytest.raises(SystemExit, match="do not exactly match"):
        prepare_module.merge_outputs(args)
    assert target_bank.resolve() == second_generation
    shutil.rmtree(shard_01)

    # A published bank must remain consumable after the prepared rollout
    # snapshot (which supplied the original absolute NPZ URDF metadata) is gone.
    prepare_module._safe_remove_generated(
        prepared_root,
        allowed_root=shard_namespace,
        label="prepared test snapshot",
        force=False,
    )
    prepared_root.rmdir()
    assert not prepared_root.exists()
    for bank_view in (
        target_bank,
        target_bank / "_single_slot_motion_bank",
        target_bank / "motion_bank",
        output_root / "current",
        output_root / "motion_bank",
    ):
        map_payload = json.loads((bank_view / "_clip_object_urdf_map.json").read_text())
        map_entry = map_payload["clips"]["box_clip"]
        mapped_urdf = (bank_view / map_entry["object_urdf_path"]).resolve()
        assert mapped_urdf.is_file()
        mapped_urdf.relative_to(second_generation)
        with np.load(bank_view / "box_clip.npz", allow_pickle=False) as payload:
            embedded_urdf = str(np.asarray(payload["object_urdf_path"]).reshape(-1)[0])
            assert embedded_urdf == map_entry["object_urdf_path"]
            assert (bank_view / embedded_urdf).resolve() == mapped_urdf
            assert np.isfinite(np.asarray(payload["joint_pos"])).all()


def _copy_cp_tao_fixture(tmp_path: Path) -> tuple[Path, Path]:
    fixture_repo = tmp_path / "repo"
    fixture_repo.mkdir()
    script = fixture_repo / "cp_tao.sh"
    shutil.copy2(CP_TAO_SCRIPT, script)
    (fixture_repo / "data" / "ds_as_data").mkdir(parents=True)
    (fixture_repo / "outputs").mkdir()
    return fixture_repo, script


def _run_cp_tao(script: Path, *, env_overrides: dict[str, str], success133: bool) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in (
        "ONLY_AS_SUCCESS133",
        "COPY_AS_SUCCESS133",
        "LOCAL_AS_ROOT",
        "RAW_EXPORT_DEST",
        "FILTERED_MOTION_BANK_NAME",
        "AS_SUCCESS133_BANK_NAME",
        "AS_SUCCESS133_CONTACT_EXPORT_NAME",
        "AS_SUCCESS133_SOURCE",
        "AS_SUCCESS133_SOURCE_CANDIDATES",
        "ROLLOUT_ASSET_ROOT",
        "ROLLOUT_ARCHIVE",
        "NFS_TAO_ROOT",
    ):
        env.pop(key, None)
    env.update(env_overrides)
    command = ["bash", str(script)]
    if success133:
        command.append("success133")
    return subprocess.run(
        command,
        cwd=script.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_cp_tao_rejects_external_local_bank_root_before_writes(tmp_path: Path) -> None:
    fixture_repo, script = _copy_cp_tao_fixture(tmp_path)
    outside = tmp_path / "outside-bank-root"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    result = _run_cp_tao(
        script,
        env_overrides={
            "LOCAL_AS_ROOT": str(outside),
            "AS_SUCCESS133_SOURCE": str(tmp_path / "missing-source"),
        },
        success133=True,
    )

    assert result.returncode != 0
    assert "LOCAL_AS_ROOT must be the repo-owned AS bank root" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    assert not any((fixture_repo / "data" / "ds_as_data").iterdir())


def test_cp_tao_rejects_explicit_empty_bank_root(tmp_path: Path) -> None:
    fixture_repo, script = _copy_cp_tao_fixture(tmp_path)
    result = _run_cp_tao(
        script,
        env_overrides={"LOCAL_AS_ROOT": ""},
        success133=True,
    )

    assert result.returncode != 0
    assert "LOCAL_AS_ROOT must be the repo-owned AS bank root" in result.stderr
    assert not any((fixture_repo / "data" / "ds_as_data").iterdir())


def test_cp_tao_rejects_raw_export_symlink_escape_before_source_checks(tmp_path: Path) -> None:
    fixture_repo, script = _copy_cp_tao_fixture(tmp_path)
    outside = tmp_path / "outside-output"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    escape = fixture_repo / "outputs" / "escape"
    escape.symlink_to(outside, target_is_directory=True)

    result = _run_cp_tao(
        script,
        env_overrides={
            "RAW_EXPORT_DEST": str(escape / "victim"),
            "ROLLOUT_ASSET_ROOT": str(tmp_path / "missing-assets"),
            "ROLLOUT_ARCHIVE": str(tmp_path / "missing-archive.tar.gz"),
        },
        success133=False,
    )

    assert result.returncode != 0
    assert "symlink" in result.stderr.lower()
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_cp_tao_accepts_exact_repo_bank_root_then_fails_closed_on_missing_source(tmp_path: Path) -> None:
    fixture_repo, script = _copy_cp_tao_fixture(tmp_path)
    result = _run_cp_tao(
        script,
        env_overrides={"AS_SUCCESS133_SOURCE": str(tmp_path / "missing-source")},
        success133=True,
    )

    assert result.returncode != 0
    assert "AS success133 source not found" in result.stderr
    assert "LOCAL_AS_ROOT" not in result.stderr
    assert not any((fixture_repo / "data" / "ds_as_data").iterdir())
