#!/usr/bin/env python3
"""Generate G1 wrist-yaw-frame rubberhand meshes for Warp depth raycasting."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_DIR = REPO_ROOT / "src/holosoma/holosoma/data/robots/g1"


def _origin_xyz(root: ET.Element, joint_name: str) -> np.ndarray:
    joint = root.find(f".//joint[@name='{joint_name}']")
    if joint is None:
        raise ValueError(f"joint not found: {joint_name}")
    origin = joint.find("origin")
    if origin is None:
        return np.zeros(3, dtype=float)
    rpy = [float(v) for v in str(origin.get("rpy", "0 0 0")).split()]
    if any(abs(v) > 1e-9 for v in rpy):
        raise ValueError(f"{joint_name} has nonzero rpy; update this generator before using it")
    return np.array([float(v) for v in str(origin.get("xyz", "0 0 0")).split()], dtype=float)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"invalid mesh: {path}")
    return mesh


def generate(robot_dir: Path) -> None:
    urdf_path = robot_dir / "g1_29dof.urdf"
    mesh_dir = robot_dir / "meshes"
    root = ET.parse(urdf_path).getroot()

    specs = [
        ("left", "left_hand_palm_joint"),
        ("right", "right_hand_palm_joint"),
    ]
    for side, joint_name in specs:
        wrist_mesh = _load_mesh(mesh_dir / f"{side}_wrist_yaw_link.STL")
        hand_mesh = _load_mesh(mesh_dir / f"{side}_rubber_hand.STL").copy()
        hand_mesh.apply_translation(_origin_xyz(root, joint_name))

        combined = trimesh.util.concatenate([wrist_mesh, hand_mesh])
        output_path = mesh_dir / f"combined_{side}_wrist_rubberhand.STL"
        combined.export(output_path)
        print(
            f"{output_path}: vertices={len(combined.vertices)} "
            f"faces={len(combined.faces)} bounds={combined.bounds.tolist()}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-dir", type=Path, default=DEFAULT_ROBOT_DIR)
    args = parser.parse_args()
    generate(args.robot_dir.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
