from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


def load_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "rebuild_prism_height_staging.py"
    )
    spec = importlib.util.spec_from_file_location("prism_height_staging", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frame0_z_extent_uses_first_frame_only() -> None:
    module = load_module()
    vertices = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]],
            [[0.0, 0.0, -2.0], [0.0, 0.0, 4.0]],
        ]
    )
    assert module.frame0_z_extent(vertices) == (0.0, 1.7, 1.7)


def test_reground_object_track_respects_new_scale() -> None:
    module = load_module()
    vertices = np.asarray(
        [
            [-0.1, -0.1, -0.2],
            [0.1, -0.1, -0.2],
            [0.1, 0.1, 0.2],
            [-0.1, 0.1, 0.2],
        ]
    )
    poses = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.05],
        ],
        dtype=np.float32,
    )
    output, report = module.reground_object_track(
        poses,
        "xyzw",
        vertices,
        0.8,
        ground_z=0.0,
        clearance_m=1e-5,
    )
    assert np.array_equal(output[:, :6], poses[:, :6])
    assert report["modified_frame_count"] >= 1
    assert min(report["source_bottom_z_after_m"]) >= 0.0
    assert min(report["predicted_retarget_bottom_z_after_m"]) >= 0.0


def test_rebuild_staging_updates_height_paths_and_contacts(
    tmp_path: Path,
) -> None:
    module = load_module()
    source_root = tmp_path / "source"
    sequence = "prism_cf_bin_m0_v1"
    sequence_root = source_root / sequence
    mesh_root = sequence_root / "object_mesh_yup_coacd"
    mesh_root.mkdir(parents=True)
    mesh_path = mesh_root / "object.obj"
    mesh_path.write_text(
        "\n".join(
            [
                "v -0.1 -0.1 -0.2",
                "v 0.1 -0.1 -0.2",
                "v 0.1 0.1 0.2",
                "v -0.1 0.1 0.2",
                "f 1 2 3",
                "f 1 3 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    poses = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.05],
        ],
        dtype=np.float32,
    )
    local_contacts = np.zeros((2, 1, 3), dtype=np.float32)
    np.savez_compressed(
        sequence_root / "input_for_retarget.npz",
        sequence=np.asarray(sequence),
        human_joints=np.zeros((2, 15, 3), dtype=np.float32),
        object_poses=poses,
        object_pose=poses.copy(),
        object_pose_quat_order=np.asarray("xyzw"),
        mesh_file=np.asarray(str(mesh_path)),
        human_height_m=np.asarray(1.59, dtype=np.float32),
        human_mesh_min_z_after_m=np.zeros(2, dtype=np.float32),
        allframe_object_grounded_ground_z_m=np.asarray(0.0, dtype=np.float32),
        allframe_object_grounded_clearance_m=np.asarray(
            1e-5, dtype=np.float32
        ),
        allframe_object_grounded_vertical_lift_m=np.zeros(
            2, dtype=np.float32
        ),
        object_contact_points_local=local_contacts,
        object_contact_points_world=local_contacts.copy(),
    )
    height_manifest = tmp_path / "heights.json"
    height_manifest.write_text(
        json.dumps(
            {
                "formula": module.HEIGHT_FORMULA,
                "rows": [
                    {
                        "sequence": sequence,
                        "human_height_m": 1.66,
                        "frame0_min_z_m": 0.0,
                        "frame0_max_z_m": 1.66,
                        "source_mesh": "/source/mesh.npz",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "output"
    report = module.rebuild_staging(
        source_root,
        height_manifest,
        output_root,
        robot_height_m=1.32,
        expected_count=1,
        reground_object=True,
    )
    assert report["status"] == "pass"
    with np.load(
        output_root / sequence / "input_for_retarget.npz",
        allow_pickle=True,
    ) as output:
        assert np.isclose(float(output["human_height_m"]), 1.66)
        assert str(output["human_height_source"]) == module.HEIGHT_SOURCE
        assert str(output["mesh_file"]).startswith(str(output_root))
        expected_world = np.asarray(output["object_poses"])[:, None, 4:7]
        np.testing.assert_allclose(
            output["object_contact_points_world"], expected_world
        )
        assert np.isclose(
            float(output["allframe_object_grounded_human_scale"]),
            1.32 / 1.66,
        )
