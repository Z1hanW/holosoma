from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET


def _normalize_geometry_types(link: ET.Element, tag_name: str) -> tuple[str, ...]:
    geometry_types: list[str] = []
    for node in link.findall(tag_name):
        geometry = node.find("geometry")
        if geometry is None:
            geometry_types.append("")
            continue
        child_tags = sorted(child.tag for child in geometry if isinstance(child.tag, str))
        geometry_types.append("+".join(child_tags))
    return tuple(sorted(geometry_types))


@lru_cache(maxsize=512)
def extract_urdf_topology_signature(urdf_path: str | Path) -> tuple[tuple, tuple]:
    """Return a conservative topology signature for URDF hierarchy compatibility checks.

    The signature intentionally ignores mesh filenames, material names, masses, and inertia values.
    It keeps only the structure that affects the imported prim hierarchy for rigid-body view binding:
    link names, the count/type of visual and collision geoms per link, and the joint graph.
    """

    resolved_path = Path(urdf_path).expanduser().resolve()
    root = ET.parse(resolved_path).getroot()

    link_signatures: list[tuple] = []
    for link in root.findall("link"):
        link_signatures.append(
            (
                link.get("name", "").strip(),
                link.find("inertial") is not None,
                len(link.findall("visual")),
                _normalize_geometry_types(link, "visual"),
                len(link.findall("collision")),
                _normalize_geometry_types(link, "collision"),
            )
        )

    joint_signatures: list[tuple[str, str, str, str]] = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        joint_signatures.append(
            (
                joint.get("name", "").strip(),
                joint.get("type", "").strip(),
                parent.get("link", "").strip() if parent is not None else "",
                child.get("link", "").strip() if child is not None else "",
            )
        )

    return tuple(link_signatures), tuple(sorted(joint_signatures))
