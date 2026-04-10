from __future__ import annotations

from pathlib import Path
import textwrap

from holosoma.simulator.shared.urdf_topology import extract_urdf_topology_signature
from holosoma.utils.object_geometry import load_urdf_box_primitive_extents, load_urdf_box_primitive_metadata


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_generated_box_urdfs_share_topology_signature() -> None:
    sig_a = extract_urdf_topology_signature(
        REPO_ROOT / "data/ds_box_data/train_g1_w_obj_prepared/_generated_urdfs/box_30.urdf"
    )
    sig_b = extract_urdf_topology_signature(
        REPO_ROOT / "data/ds_box_data/train_g1_w_obj_prepared/_generated_urdfs/box_31.urdf"
    )

    assert sig_a == sig_b


def test_box_urdf_topology_signature_detects_link_name_mismatch() -> None:
    generated_sig = extract_urdf_topology_signature(
        REPO_ROOT / "data/ds_box_data/train_g1_w_obj_prepared/_generated_urdfs/box_30.urdf"
    )
    behave_sig = extract_urdf_topology_signature(
        REPO_ROOT / "src/holosoma_retargeting/models/behave_objects/boxlarge/boxlarge.urdf"
    )

    assert generated_sig != behave_sig


def test_generated_box_urdf_resolves_box_primitive_extents() -> None:
    extents = load_urdf_box_primitive_extents(
        REPO_ROOT / "data/ds_box_data/train_g1_w_obj_prepared/_generated_urdfs/box_30.urdf"
    )

    assert extents is not None
    assert all(value > 0.0 for value in extents)


def test_generated_box_urdf_resolves_box_primitive_metadata() -> None:
    metadata = load_urdf_box_primitive_metadata(
        REPO_ROOT / "data/ds_box_data/train_g1_w_obj_prepared/_generated_urdfs/box_30.urdf"
    )

    assert metadata is not None
    assert metadata.mass == 0.1
    assert metadata.static_friction == 0.9
    assert metadata.dynamic_friction == 0.9
    assert metadata.compliant_contact_stiffness == 30000.0
    assert metadata.compliant_contact_damping == 1000.0
    assert metadata.visual_color == (0.7, 0.8, 0.9)


def test_box_primitive_metadata_rejects_offset_geometry_origin(tmp_path: Path) -> None:
    urdf_path = tmp_path / "offset_box.urdf"
    urdf_path.write_text(
        textwrap.dedent(
            """\
            <?xml version="1.0" ?>
            <robot name="offset_box">
              <link name="baseLink">
                <inertial>
                  <mass value="0.1"/>
                  <origin xyz="0 0 0"/>
                  <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
                </inertial>
                <collision>
                  <origin xyz="0.1 0 0"/>
                  <geometry>
                    <box size="0.4 0.4 0.8"/>
                  </geometry>
                </collision>
              </link>
            </robot>
            """
        ),
        encoding="utf-8",
    )

    assert load_urdf_box_primitive_metadata(urdf_path) is None
