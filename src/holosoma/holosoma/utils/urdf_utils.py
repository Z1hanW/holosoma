"""URDF helpers for resolving fixed-link transforms."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.rotations import quat_apply, quat_from_euler_xyz, quat_mul
from holosoma.utils.safe_torch_import import torch


def _parse_vec3(text: str | None) -> tuple[float, float, float]:
    if not text:
        return (0.0, 0.0, 0.0)
    parts = text.split()
    parts += ["0.0"] * max(0, 3 - len(parts))
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _get_urdf_text(robot_config) -> tuple[str, str]:
    asset_root = robot_config.asset.asset_root
    if asset_root.startswith("@holosoma/"):
        asset_root = asset_root.replace("@holosoma", get_holosoma_root())
    urdf_path = os.path.join(asset_root, robot_config.asset.urdf_file)
    return urdf_path, Path(urdf_path).read_text(encoding="utf-8")


def resolve_fixed_link_offset(
    robot_config,
    link_name: str | None,
    *,
    available_links: Iterable[str] | None = None,
    device: str | torch.device | None = None,
    max_depth: int = 8,
) -> tuple[str, torch.Tensor, torch.Tensor] | None:
    """Resolve a fixed-link offset to the nearest available parent link.

    Returns (parent_link, offset_pos, offset_quat) where offset_* transforms the
    parent link into the requested link. Returns None when the link is not found
    or the fixed joint chain cannot be resolved.
    """
    if not link_name:
        return None

    _, urdf_text = _get_urdf_text(robot_config)
    root = ET.fromstring(urdf_text)

    joint_map: dict[str, tuple[str, tuple[float, float, float], tuple[float, float, float]]] = {}
    for joint in root.findall("joint"):
        if joint.get("type") != "fixed":
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_link = parent.get("link")
        child_link = child.get("link")
        if not parent_link or not child_link:
            continue
        origin = joint.find("origin")
        xyz = _parse_vec3(origin.get("xyz") if origin is not None else None)
        rpy = _parse_vec3(origin.get("rpy") if origin is not None else None)
        joint_map[child_link] = (parent_link, xyz, rpy)

    if link_name not in joint_map:
        return None

    available_set = set(available_links) if available_links is not None else None
    device = device if device is not None else torch.device("cpu")

    offset_pos = torch.zeros(3, device=device, dtype=torch.float32)
    offset_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32)

    current_link = link_name
    for _ in range(max_depth):
        entry = joint_map.get(current_link)
        if entry is None:
            break
        parent_link, xyz, rpy = entry
        step_pos = torch.tensor(xyz, device=device, dtype=torch.float32)
        step_rpy = torch.tensor(rpy, device=device, dtype=torch.float32)
        step_quat = quat_from_euler_xyz(step_rpy[0], step_rpy[1], step_rpy[2])

        offset_pos = step_pos + quat_apply(step_quat.unsqueeze(0), offset_pos.unsqueeze(0), w_last=True).squeeze(0)
        offset_quat = quat_mul(step_quat.unsqueeze(0), offset_quat.unsqueeze(0), w_last=True).squeeze(0)

        current_link = parent_link
        if available_set is None or current_link in available_set:
            return current_link, offset_pos, offset_quat

    return None
