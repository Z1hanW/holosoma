from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_prepare_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "prepare_as_rank_shards.py"
    spec = importlib.util.spec_from_file_location("prepare_as_rank_shards", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_object(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    mesh = root / f"{name}.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    urdf = root / f"{name}.urdf"
    urdf.write_text(
        f"""
<robot name="{name}">
  <link name="baseLink">
    <visual>
      <geometry><mesh filename="{mesh.name}"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="{mesh.name}"/></geometry>
    </collision>
  </link>
</robot>
""".strip(),
        encoding="utf-8",
    )
    return urdf


def _write_motion_bank(tmp_path: Path, *, clip_count: int, unique_urdfs: int) -> tuple[Path, Path]:
    motion_dir = tmp_path / "motion"
    motion_dir.mkdir()
    urdf_dir = motion_dir / "_single_slot_urdfs"
    urdfs = [_write_object(urdf_dir, f"obj_{idx}") for idx in range(unique_urdfs)]

    clips = {}
    for idx in range(clip_count):
        clip_id = f"clip_{idx:03d}"
        (motion_dir / f"{clip_id}.npz").write_bytes(b"placeholder")
        urdf = urdfs[idx % unique_urdfs]
        clips[clip_id] = {
            "object_name": urdf.stem,
            "object_urdf_path": str(urdf.relative_to(motion_dir)),
            "object_mesh_path": str((urdf.with_suffix(".obj")).relative_to(motion_dir)),
        }

    object_map = motion_dir / "_clip_object_urdf_map.json"
    object_map.write_text(json.dumps({"clips": clips}, indent=2), encoding="utf-8")
    return motion_dir, object_map


def _clip_counts(output_root: Path, world_size: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rank in range(world_size):
        rank_dir = output_root / f"rank_{rank}"
        payload = json.loads((rank_dir / "_clip_object_urdf_map.json").read_text(encoding="utf-8"))
        for clip_id, entry in payload["clips"].items():
            counts[clip_id] = counts.get(clip_id, 0) + 1
            assert (rank_dir / f"{clip_id}.npz").is_symlink()
            assert (rank_dir / entry["object_urdf_path"]).resolve().is_file()
            assert (rank_dir / entry["object_mesh_path"]).resolve().is_file()
    return counts


def test_prepare_rank_shards_partitions_object_groups(tmp_path: Path) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=6, unique_urdfs=6)
    output_root = tmp_path / "rank_shards" / "ws3"

    manifest = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=3,
    )

    assert manifest["strategy"] == "object_closure"
    assert manifest["exact_clip_partition"] is True
    assert manifest["duplicated_to_fill_empty_ranks"] is False
    assert all(shard["clip_count"] > 0 for shard in manifest["shards"])
    assert _clip_counts(output_root, 3) == {f"clip_{idx:03d}": 1 for idx in range(6)}


def test_prepare_rank_shards_splits_by_clip_when_world_size_exceeds_unique_urdfs(tmp_path: Path) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=5, unique_urdfs=2)
    output_root = tmp_path / "rank_shards" / "ws4"

    manifest = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=4,
    )

    assert manifest["strategy"] == "clip"
    assert manifest["exact_clip_partition"] is True
    assert manifest["duplicated_to_fill_empty_ranks"] is False
    assert all(shard["clip_count"] > 0 for shard in manifest["shards"])
    assert _clip_counts(output_root, 4) == {f"clip_{idx:03d}": 1 for idx in range(5)}


def test_prepare_rank_shards_keeps_multi_object_clip_closure(tmp_path: Path) -> None:
    module = _load_prepare_module()
    motion_dir = tmp_path / "motion"
    motion_dir.mkdir()
    urdf_dir = motion_dir / "objects"
    urdf_a = _write_object(urdf_dir, "obj_a")
    urdf_b = _write_object(urdf_dir, "obj_b")
    urdf_c = _write_object(urdf_dir, "obj_c")
    for clip_id in ("clip_pair", "clip_single"):
        (motion_dir / f"{clip_id}.npz").write_bytes(b"placeholder")
    object_map = motion_dir / "_clip_object_urdf_map.json"
    object_map.write_text(
        json.dumps(
            {
                "clips": {
                    "clip_pair": {
                        "object_urdf_paths": [
                            str(urdf_a.relative_to(motion_dir)),
                            str(urdf_b.relative_to(motion_dir)),
                        ],
                        "object_mesh_paths": [
                            str(urdf_a.with_suffix(".obj").relative_to(motion_dir)),
                            str(urdf_b.with_suffix(".obj").relative_to(motion_dir)),
                        ],
                    },
                    "clip_single": {
                        "object_urdf_path": str(urdf_c.relative_to(motion_dir)),
                        "object_mesh_path": str(urdf_c.with_suffix(".obj").relative_to(motion_dir)),
                    },
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "rank_shards" / "ws2"

    manifest = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=2,
    )

    assert manifest["strategy"] == "object_closure"
    pair_rank = None
    for rank in range(2):
        payload = json.loads((output_root / f"rank_{rank}" / "_clip_object_urdf_map.json").read_text())
        if "clip_pair" in payload["clips"]:
            pair_rank = rank
            entry = payload["clips"]["clip_pair"]
            assert len(entry["object_urdf_paths"]) == 2
            assert len(entry["object_mesh_paths"]) == 2
            assert all((output_root / f"rank_{rank}" / path).resolve().is_file() for path in entry["object_urdf_paths"])
            assert all((output_root / f"rank_{rank}" / path).resolve().is_file() for path in entry["object_mesh_paths"])
    assert pair_rank is not None


def test_rank_local_path_resolver_uses_local_rank(monkeypatch, tmp_path: Path) -> None:
    from holosoma.utils.rank_local_shards import resolve_rank_local_motion_path, resolve_rank_local_object_map

    root = tmp_path / "rank_shards" / "ws2"
    rank_dir = root / "rank_1"
    rank_dir.mkdir(parents=True)
    object_map = rank_dir / "_clip_object_urdf_map.json"
    object_map.write_text('{"clips": {}}', encoding="utf-8")
    base_motion = tmp_path / "motion"
    base_motion.mkdir()

    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_MOTION_ROOT", str(root))
    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")

    assert resolve_rank_local_motion_path(base_motion) == str(rank_dir)
    assert resolve_rank_local_object_map(base_motion / "_clip_object_urdf_map.json") == str(object_map)


def test_rank_local_path_resolver_uses_global_rank_for_multinode(monkeypatch, tmp_path: Path) -> None:
    from holosoma.utils.rank_local_shards import resolve_rank_local_motion_path, resolve_rank_local_object_map

    root = tmp_path / "rank_shards" / "ws48"
    rank_dir = root / "rank_17"
    rank_dir.mkdir(parents=True)
    object_map = rank_dir / "_clip_object_urdf_map.json"
    object_map.write_text('{"clips": {}}', encoding="utf-8")
    base_motion = tmp_path / "motion"
    base_motion.mkdir()

    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_MOTION_ROOT", str(root))
    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED", "1")
    monkeypatch.setenv("WORLD_SIZE", "48")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "17")
    monkeypatch.setenv("LOCAL_RANK", "1")

    assert resolve_rank_local_motion_path(base_motion) == str(rank_dir)
    assert resolve_rank_local_object_map(base_motion / "_clip_object_urdf_map.json") == str(object_map)
