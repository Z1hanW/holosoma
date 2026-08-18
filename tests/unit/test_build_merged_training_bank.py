from __future__ import annotations

import json
from pathlib import Path
import sys
import tarfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_merged_training_bank import (  # noqa: E402
    SourceSpec,
    build_bank,
    thaw_and_remove,
    verify_archive,
    verify_bank,
)


def _write_motion(path: Path, clip_id: str, urdf: Path) -> None:
    frames = 3
    np.savez_compressed(
        path,
        body_ang_vel_w=np.zeros((frames, 32, 3), dtype=np.float32),
        body_lin_vel_w=np.zeros((frames, 32, 3), dtype=np.float32),
        body_names=np.asarray([f"body_{index}" for index in range(32)]),
        body_pos_w=np.zeros((frames, 32, 3), dtype=np.float32),
        body_quat_w=np.zeros((frames, 32, 4), dtype=np.float32),
        fps=np.asarray([50], dtype=np.int32),
        joint_names=np.asarray([f"joint_{index}" for index in range(29)]),
        joint_pos=np.zeros((frames, 36), dtype=np.float32),
        joint_vel=np.zeros((frames, 35), dtype=np.float32),
        object_ang_vel_w=np.zeros((frames, 3), dtype=np.float32),
        object_lin_vel_w=np.zeros((frames, 3), dtype=np.float32),
        object_name=np.asarray(clip_id),
        object_pos_w=np.zeros((frames, 3), dtype=np.float32),
        object_quat_w=np.zeros((frames, 4), dtype=np.float32),
        object_size=np.ones(3, dtype=np.float32),
        object_urdf_path=np.asarray(str(urdf)),
    )


def _write_source(root: Path, *, clip_id: str, category: str) -> SourceSpec:
    motion_dir = root / "motion"
    contact_root = root / "contacts"
    object_dir = root / "objects" / clip_id
    motion_dir.mkdir(parents=True)
    object_dir.mkdir(parents=True)
    mesh = object_dir / "mesh.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="ascii")
    urdf = object_dir / f"{clip_id}.urdf"
    urdf.write_text(
        "<?xml version='1.0'?>\n"
        f"<robot name='{clip_id}'><link name='baseLink'>"
        "<inertial><mass value='1'/><origin xyz='0 0 0' rpy='0 0 0'/>"
        "<inertia ixx='1' ixy='0' ixz='0' iyy='1' iyz='0' izz='1'/></inertial>"
        f"<visual><geometry><mesh filename='{mesh}' scale='1 1 1'/></geometry></visual>"
        f"<collision><geometry><mesh filename='{mesh}' scale='1 1 1'/></geometry></collision>"
        "</link></robot>\n",
        encoding="ascii",
    )
    _write_motion(motion_dir / f"{clip_id}.npz", clip_id, urdf)
    (motion_dir / "_clip_object_urdf_map.json").write_text(
        json.dumps(
            {
                "motion_transition_source": {
                    "version": 1,
                    "source_semantics": "global_multi_clip_runtime",
                    "source_clip_count": 1,
                },
                "clips": {
                    clip_id: {
                        "object_name": clip_id,
                        "object_urdf_path": str(urdf),
                        "object_mesh_path": str(mesh),
                        "mesh_physics_category": category,
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    contact_dir = contact_root / "clips" / f"0000_{clip_id}"
    contact_dir.mkdir(parents=True)
    for side in ("left_wrist", "right_wrist"):
        np.save(contact_dir / f"{side}_contact_points.npy", np.zeros((1, 3), dtype=np.float32))
        np.save(contact_dir / f"{side}_contact_point_counts.npy", np.ones(1, dtype=np.int32))
        np.save(contact_dir / f"{side}_contact_interval_steps.npy", np.asarray([0, 1], dtype=np.int32))
    np.savez_compressed(contact_dir / "teacher_rollout_reference.npz", clip_id=np.asarray(clip_id))
    return SourceSpec(label=root.name, motion_dir=motion_dir, contact_root=contact_root)


def test_builds_portable_immutable_union(tmp_path: Path) -> None:
    source_a = _write_source(tmp_path / "corl", clip_id="box_1", category="box")
    source_b = _write_source(tmp_path / "debug", clip_id="unscale__any_bin_1", category="bin")
    output_root: Path | None = None
    try:
        output_root, manifest_sha = build_bank(
            [source_a, source_b],
            output_base=tmp_path / "published",
            contact_export_name="contact_export_merged",
            expected_total=2,
        )
        assert verify_bank(
            output_root,
            expected_digest=output_root.name,
            expected_manifest_sha256=manifest_sha,
        ) == manifest_sha

        object_map = json.loads((output_root / "_clip_object_urdf_map.json").read_text())
        assert set(object_map["clips"]) == {"box_1", "unscale__any_bin_1"}
        for entry in object_map["clips"].values():
            assert not Path(entry["object_urdf_path"]).is_absolute()
            assert not Path(entry["object_mesh_path"]).is_absolute()

        manifest = json.loads((output_root / "manifest.json").read_text())
        assert manifest["clip_count"] == 2
        assert manifest["category_counts"] == {"bin": 1, "box": 1}
        assert manifest["geometry_contract"]["fallback_allowed"] is False
        assert "source_identity" not in manifest
        assert manifest["source_labels"] == ["corl", "debug"]

        archive = tmp_path / f"{output_root.name}.tar.gz"
        with tarfile.open(archive, mode="w:gz") as stream:
            stream.add(output_root, arcname=output_root.name)
        archive_sha, member_count = verify_archive(
            archive,
            expected_digest=output_root.name,
            expected_manifest_sha256=manifest_sha,
        )
        assert len(archive_sha) == 64
        assert member_count > len(manifest["published_files"])
    finally:
        if output_root is not None:
            thaw_and_remove(output_root)
