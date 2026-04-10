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


@pytest.mark.parametrize(
    ("urdf_path", "expected_extents"),
    [
        (
            Path("/home/ubuntu/FAR/holosoma/data/ds_box_data/train_g1_w_obj_prepared/_generated_urdfs/box_10.urdf"),
            (0.4585591, 0.2551282, 0.22102714),
        ),
        (
            Path(
                "/home/ubuntu/FAR/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"
            ),
            (0.4711539, 0.45873022, 0.4078947),
        ),
    ],
)
def test_known_box_urdfs_resolve_stable_primitive_extents(
    urdf_path: Path,
    expected_extents: tuple[float, float, float],
):
    if not urdf_path.is_file():
        pytest.skip(f"URDF asset not present on this node: {urdf_path}")

    metadata = load_urdf_box_primitive_metadata(urdf_path)
    assert metadata is not None
    assert metadata.extents == pytest.approx(expected_extents, abs=1.0e-6)
