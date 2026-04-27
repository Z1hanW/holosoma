from __future__ import annotations

import json
from pathlib import Path

import pytest

from holosoma.utils.object_geometry import load_urdf_box_primitive_metadata


def _iter_unique_urdfs_from_clip_map(map_path: Path) -> list[Path]:
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    clips = payload["clips"] if isinstance(payload, dict) and isinstance(payload.get("clips"), dict) else payload
    if not isinstance(clips, dict):
        raise TypeError(f"Unexpected clip map payload in {map_path}")

    unique_urdfs: set[Path] = set()
    for value in clips.values():
        if isinstance(value, str):
            raw_urdf = value.strip()
        elif isinstance(value, dict):
            raw_urdf = str(value.get("object_urdf_path", "")).strip()
        else:
            raw_urdf = ""
        if raw_urdf:
            unique_urdfs.add(Path(raw_urdf).expanduser().resolve())
    return sorted(unique_urdfs)


@pytest.mark.parametrize(
    "map_path",
    [
        Path("/home/ubuntu/FAR/holosoma/data/ds_box_data/train_g1_w_obj_prepared/_clip_object_urdf_map.json"),
        Path("/home/ubuntu/FAR/holosoma/data/ds_box_data/train_g1_w_obj_prepared_plus_omomo_orig/_clip_object_urdf_map.json"),
    ],
)
def test_default_clip_object_maps_are_primitive_compatible(map_path: Path):
    if not map_path.is_file():
        pytest.skip(f"clip-object map not present on this node: {map_path}")

    missing_metadata: list[Path] = []
    unique_urdfs = _iter_unique_urdfs_from_clip_map(map_path)
    assert unique_urdfs, f"expected at least one URDF in {map_path}"

    for urdf_path in unique_urdfs:
        metadata = load_urdf_box_primitive_metadata(urdf_path)
        if metadata is None:
            missing_metadata.append(urdf_path)

    assert not missing_metadata, (
        f"primitive mode rejected {len(missing_metadata)} URDF(s) from {map_path}: "
        + ", ".join(str(path) for path in missing_metadata[:5])
    )


def test_explicit_box_urdf_resolves_stable_primitive_extents(tmp_path: Path):
    urdf_path = tmp_path / "box.urdf"
    urdf_path.write_text(
        """<?xml version="1.0"?>
<robot name="box">
  <link name="baseLink">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.4 0.2 0.1"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.4 0.2 0.1"/></geometry>
    </collision>
  </link>
</robot>
""",
        encoding="utf-8",
    )

    metadata = load_urdf_box_primitive_metadata(urdf_path)
    assert metadata is not None
    assert metadata.extents == pytest.approx((0.4, 0.2, 0.1), abs=1.0e-6)
    assert metadata.center_offset == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-6)


def test_non_cuboid_box_mesh_urdf_is_not_primitive_compatible():
    urdf_path = Path(
        "/home/ubuntu/FAR/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"
    )
    if not urdf_path.is_file():
        pytest.skip(f"URDF asset not present on this node: {urdf_path}")

    assert load_urdf_box_primitive_metadata(urdf_path) is None
