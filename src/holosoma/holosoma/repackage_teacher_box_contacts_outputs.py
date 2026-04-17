from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from holosoma.export_teacher_box_contacts import _EXPORT_REGION_LABELS, _save_overlay_assets


def _copy_matching(src_dir: Path, dst_dir: Path, pattern: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in src_dir.glob(pattern):
        if path.is_file():
            dst_path = dst_dir / path.name
            try:
                if path.resolve() == dst_path.resolve():
                    continue
            except Exception:
                pass
            shutil.copy2(path, dst_path)


def repackage_outputs(
    source_dir: Path,
    data_root: Path,
    vis_root: Path,
    stats_root: Path,
    *,
    flatten: bool = False,
) -> tuple[Path, Path, Path]:
    source_dir = source_dir.resolve()
    run_name = source_dir.name
    if flatten:
        data_dir = data_root.resolve()
        vis_dir = vis_root.resolve()
        stats_dir = stats_root.resolve()
    else:
        data_dir = data_root.resolve() / run_name
        vis_dir = vis_root.resolve() / run_name
        stats_dir = stats_root.resolve() / run_name

    for base_dir in (data_dir, vis_dir, stats_dir):
        base_dir.mkdir(parents=True, exist_ok=True)

    source_clips_dir = source_dir / "clips"
    for source_clip_dir in sorted(source_clips_dir.iterdir()):
        if not source_clip_dir.is_dir():
            continue

        data_clip_dir = data_dir / "clips" / source_clip_dir.name
        vis_clip_dir = vis_dir / "clips" / source_clip_dir.name
        stats_clip_dir = stats_dir / "clips" / source_clip_dir.name
        data_clip_dir.mkdir(parents=True, exist_ok=True)
        vis_clip_dir.mkdir(parents=True, exist_ok=True)
        stats_clip_dir.mkdir(parents=True, exist_ok=True)

        _copy_matching(source_clip_dir, data_clip_dir, "*.npy")
        _copy_matching(source_clip_dir, data_clip_dir, "*.npz")
        _copy_matching(source_clip_dir, stats_clip_dir, "*.csv")
        _copy_matching(source_clip_dir, stats_clip_dir, "*.json")
        _copy_matching(source_clip_dir, stats_clip_dir, "*.txt")

        metadata_path = source_clip_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        primitive_points_xyz = np.load(source_clip_dir / "primitive_contact_points.npy")
        primitive_counts = np.load(source_clip_dir / "primitive_contact_point_counts.npy")
        region_points_by_label = {}
        region_counts_by_label = {}
        for label in _EXPORT_REGION_LABELS:
            points_path = source_clip_dir / f"{label}_contact_points.npy"
            counts_path = source_clip_dir / f"{label}_contact_point_counts.npy"
            region_points_by_label[label] = (
                np.load(points_path) if points_path.exists() else np.zeros((0, 3), dtype=np.float32)
            )
            region_counts_by_label[label] = (
                np.load(counts_path) if counts_path.exists() else np.zeros((0,), dtype=np.int32)
            )
        _save_overlay_assets(
            vis_clip_dir,
            clip_id=str(metadata["clip_id"]),
            object_name=str(metadata.get("object_name", "")),
            object_urdf_path=str(metadata.get("object_urdf_path", "")),
            extents_xyz=np.asarray(metadata["primitive_extents_xyz"], dtype=np.float32),
            retained_points_xyz=primitive_points_xyz,
            retained_counts=primitive_counts,
            display_points_xyz=primitive_points_xyz,
            display_point_labels=[_EXPORT_REGION_LABELS[0]] * int(primitive_points_xyz.shape[0]),
            region_points_by_label=region_points_by_label,
            save_glb=True,
            save_preview_png=True,
            save_face_heatmap_png=True,
        )

    for pattern in ("*.csv", "*.json", "*.txt"):
        _copy_matching(source_dir, stats_dir, pattern)

    source_motion_bank_dir = source_dir / "motion_bank"
    if source_motion_bank_dir.is_dir():
        motion_bank_dir = data_dir / "motion_bank"
        motion_bank_dir.mkdir(parents=True, exist_ok=True)
        _copy_matching(source_motion_bank_dir, motion_bank_dir, "*.npz")
        _copy_matching(source_motion_bank_dir, motion_bank_dir, "*.json")

    manifest = {
        "source_dir": str(source_dir),
        "data_dir": str(data_dir),
        "vis_dir": str(vis_dir),
        "stats_dir": str(stats_dir),
    }
    (stats_dir / "repackage_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return data_dir, vis_dir, stats_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Repackage teacher box contact outputs into data/vis/stats roots.")
    parser.add_argument("--source-dir", required=True, help="Existing teacher_box_contacts run directory.")
    parser.add_argument("--data-root", default="outputs", help="Destination root for .npy data files.")
    parser.add_argument("--vis-root", default="outputs_vis", help="Destination root for visualization files.")
    parser.add_argument("--stats-root", default="outputs_sts", help="Destination root for csv/json/txt statistics.")
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Write directly into the provided roots instead of adding a run-name wrapper directory.",
    )
    args = parser.parse_args()

    data_dir, vis_dir, stats_dir = repackage_outputs(
        source_dir=Path(args.source_dir),
        data_root=Path(args.data_root),
        vis_root=Path(args.vis_root),
        stats_root=Path(args.stats_root),
        flatten=args.flatten,
    )
    print(f"data_dir={data_dir}")
    print(f"vis_dir={vis_dir}")
    print(f"stats_dir={stats_dir}")


if __name__ == "__main__":
    main()
