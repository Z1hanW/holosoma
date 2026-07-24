"""Content-close simulator assets selected by the effective training config.

This module intentionally has no Isaac Sim, torch, or GPU imports.  Scientific
launches call it after Tyro and the observation/perception config overrides have
finished, but before the simulator or any checkpoint preflight starts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import mmap
import os
import re
import shlex
import struct
import xml.etree.ElementTree as ET
from collections.abc import Mapping, MutableMapping
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote

from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.defm_source import resolve_defm_source_root
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.training_provenance import (
    ENV_NAME,
    RUNTIME_ASSET_DIGEST_KEY,
    RUNTIME_ASSET_MANIFEST_KEY,
    RUNTIME_ASSET_MANIFEST_VERSION,
    RUNTIME_ASSET_PHASE_FINAL,
    RUNTIME_ASSET_PHASE_KEY,
    RUNTIME_ASSET_PHASE_PENDING,
    canonical_runtime_asset_manifest_json,
    canonical_training_provenance_json,
    embedded_runtime_asset_manifest_sha256,
    pending_runtime_asset_manifest_sha256,
    validate_semantic_environment_binding,
    validate_training_provenance,
)


RUNTIME_ASSET_MANIFEST_FILENAME = "runtime_asset_manifest.json"
_ISAACSIM_TARGET = "holosoma.simulator.isaacsim.isaacsim.IsaacSim"
_REMOTE_URI_PREFIXES = ("http://", "https://", "s3://", "omniverse://")
_MTL_TEXTURE_DIRECTIVES = {
    "bump",
    "decal",
    "disp",
    "map_bump",
    "map_d",
    "map_ka",
    "map_kd",
    "map_ke",
    "map_ks",
    "map_ns",
    "norm",
    "refl",
}
_OBJ_MTLLIB_RE = re.compile(rb"(?im)^[ \t]*mtllib(?:[ \t]+[^\r\n]*)?(?=\r?$)")

AssetIdentityRecorder = Callable[[Path], None]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_value(child)
            for key, child in vars(value).items()
            if not str(key).startswith("_")
        }
    raise ValueError(f"runtime asset semantics contain a non-JSON value: {value!r}")


def _stable_sha256_file(
    path: Path,
    *,
    role: str,
    allow_empty: bool = False,
) -> tuple[str, int]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{role} does not exist or is not a regular file: {path}")
    before = path.stat()
    if before.st_size <= 0 and not allow_empty:
        raise ValueError(f"{role} is empty: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    before_identity = _stat_identity(before)
    after_identity = _stat_identity(after)
    if before_identity != after_identity:
        raise RuntimeError(f"{role} changed while its provenance digest was being computed: {path}")
    return digest.hexdigest(), int(after.st_size)


def _stat_identity(stat_result: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
    )


def _require_file_identity(
    path: Path,
    *,
    expected: tuple[int, int, int, int],
    role: str,
) -> None:
    try:
        actual = _stat_identity(path.stat())
    except FileNotFoundError as exc:
        raise RuntimeError(f"{role} disappeared while its dependency closure was being computed: {path}") from exc
    if actual != expected:
        raise RuntimeError(f"{role} changed while its dependency closure was being computed: {path}")


def _asset_record(
    path: Path,
    *,
    role: str,
    reference: str,
    identity_recorder: AssetIdentityRecorder | None = None,
) -> dict[str, Any]:
    digest, size = _stable_sha256_file(path, role=role)
    if identity_recorder is not None:
        identity_recorder(path)
    return {
        "reference": reference,
        "size": size,
        "sha256": digest,
    }


def _resolve_asset_root(raw_root: str) -> Path:
    raw_root = str(raw_root).strip()
    if not raw_root:
        raise ValueError("robot.asset.asset_root is empty")
    if raw_root == "@holosoma":
        raw_root = get_holosoma_root()
    elif raw_root.startswith("@holosoma/"):
        raw_root = str(Path(get_holosoma_root()) / raw_root[len("@holosoma/") :])
    elif raw_root.startswith("@"):
        raise ValueError(f"unsupported robot asset-root alias: {raw_root!r}")
    if raw_root.startswith(_REMOTE_URI_PREFIXES):
        raise ValueError(f"robot asset root must be local for content closure: {raw_root!r}")
    resolved = Path(resolve_data_file_path(raw_root)).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"robot asset root does not exist: {resolved}")
    return resolved


def _resolve_local_reference(
    raw_reference: str,
    *,
    base_dir: Path,
    asset_root: Path,
    role: str,
    identity_recorder: AssetIdentityRecorder | None = None,
) -> Path:
    raw_reference = str(raw_reference).strip()
    if not raw_reference:
        raise ValueError(f"{role} has an empty filename")
    lowered = raw_reference.lower()
    if lowered.startswith(_REMOTE_URI_PREFIXES) or lowered.startswith("data:"):
        raise ValueError(f"{role} uses a non-local asset URI that cannot be content-closed: {raw_reference!r}")
    if raw_reference.startswith("package://"):
        path = asset_root / unquote(raw_reference[len("package://") :])
    elif raw_reference.startswith("file://"):
        file_reference = unquote(raw_reference[len("file://") :])
        path = Path(file_reference).expanduser()
        if not path.is_absolute():
            path = base_dir / path
    elif raw_reference.startswith("holosoma/data"):
        path = Path(resolve_data_file_path(raw_reference)).expanduser()
    else:
        path = Path(raw_reference).expanduser()
        if not path.is_absolute():
            path = base_dir / path
    if identity_recorder is not None:
        # Preserve the lexical path as well as the resolved target.  A cached
        # validation must be invalidated when a symlink is retargeted even if
        # the old target still exists and retains the same file identity.
        identity_recorder(path)
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{role} asset does not exist: {path}")
    return path


def _xml_elements(root: ET.Element, local_name: str) -> list[ET.Element]:
    return [element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == local_name]


def _parse_shell_words(line: str, *, role: str) -> list[str]:
    try:
        return shlex.split(line, comments=True, posix=True)
    except ValueError as exc:
        raise ValueError(f"invalid quoted asset reference in {role}: {line!r}") from exc


def _obj_dependency_records(
    mesh_path: Path,
    *,
    role: str,
    identity_recorder: AssetIdentityRecorder | None = None,
) -> list[dict[str, Any]]:
    dependencies: list[dict[str, Any]] = []
    material_paths: list[tuple[str, Path]] = []
    with mesh_path.open("rb") as stream:
        if mesh_path.stat().st_size:
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                for match in _OBJ_MTLLIB_RE.finditer(mapped):
                    try:
                        line = match.group(0).decode("utf-8", errors="strict")
                    except UnicodeDecodeError as exc:
                        raise ValueError(
                            f"{role} OBJ mtllib directive is not valid UTF-8: {mesh_path}"
                        ) from exc
                    words = _parse_shell_words(line, role=role)
                    if len(words) < 2:
                        raise ValueError(f"{role} OBJ has an empty mtllib directive: {mesh_path}")
                    for reference in words[1:]:
                        material_path = _resolve_local_reference(
                            reference,
                            base_dir=mesh_path.parent,
                            asset_root=mesh_path.parent,
                            role=f"{role}.material",
                            identity_recorder=identity_recorder,
                        )
                        material_paths.append((reference, material_path))

    for material_reference, material_path in sorted(material_paths, key=lambda item: item[0]):
        material_identity = _stat_identity(material_path.stat())
        material_record = _asset_record(
            material_path,
            role=f"{role}.material",
            reference=material_reference,
            identity_recorder=identity_recorder,
        )
        texture_records: list[dict[str, Any]] = []
        try:
            material_lines = material_path.read_text(encoding="utf-8", errors="strict").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{role} MTL is not valid UTF-8 and cannot be dependency-scanned: {material_path}"
            ) from exc
        for line in material_lines:
            stripped = line.lstrip()
            directive = stripped.split(maxsplit=1)[0].lower() if stripped else ""
            if not directive or (
                directive not in _MTL_TEXTURE_DIRECTIVES and not directive.startswith("map_")
            ):
                continue
            words = _parse_shell_words(line, role=f"{role}.material")
            if len(words) < 2:
                raise ValueError(f"{role} MTL has an empty texture directive: {material_path}")
            # MTL map options precede the filename.  The final shell word is
            # the texture path for all directives supported by common loaders.
            texture_reference = words[-1]
            texture_path = _resolve_local_reference(
                texture_reference,
                base_dir=material_path.parent,
                asset_root=material_path.parent,
                role=f"{role}.material_texture",
                identity_recorder=identity_recorder,
            )
            texture_records.append(
                _asset_record(
                    texture_path,
                    role=f"{role}.material_texture",
                    reference=texture_reference,
                    identity_recorder=identity_recorder,
                )
            )
        material_record["textures"] = sorted(
            texture_records,
            key=lambda record: (record["reference"], record["sha256"]),
        )
        _require_file_identity(
            material_path,
            expected=material_identity,
            role=f"{role}.material",
        )
        dependencies.append({"kind": "material", **material_record})
    return dependencies


def _gltf_dependency_records(
    mesh_path: Path,
    *,
    role: str,
    identity_recorder: AssetIdentityRecorder | None = None,
) -> list[dict[str, Any]]:
    try:
        payload = json.loads(mesh_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{role} has invalid glTF JSON at {mesh_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{role} glTF root must be a JSON object: {mesh_path}")
    return _gltf_payload_dependency_records(
        payload,
        mesh_path=mesh_path,
        role=role,
        identity_recorder=identity_recorder,
    )


def _gltf_payload_dependency_records(
    payload: dict[str, Any],
    *,
    mesh_path: Path,
    role: str,
    identity_recorder: AssetIdentityRecorder | None = None,
) -> list[dict[str, Any]]:
    for collection_name in ("buffers", "images"):
        collection = payload.get(collection_name, [])
        if not isinstance(collection, list):
            raise ValueError(f"{role} glTF {collection_name} must be a list: {mesh_path}")
        if any(not isinstance(entry, dict) for entry in collection):
            raise ValueError(f"{role} glTF {collection_name} entries must be objects: {mesh_path}")

    records: list[dict[str, Any]] = []
    for field_path, raw_uri in _gltf_uri_fields(payload):
        if not isinstance(raw_uri, str):
            raise ValueError(
                f"{role} glTF URI field {_format_json_path(field_path)} must be a string: {mesh_path}"
            )
        uri = raw_uri.strip()
        if uri.lower().startswith("data:"):
            continue
        kind = "external"
        if len(field_path) == 3 and field_path[0] == "buffers" and field_path[2] == "uri":
            kind = "buffer"
        elif len(field_path) == 3 and field_path[0] == "images" and field_path[2] == "uri":
            kind = "texture"
        dependency_role = f"{role}.gltf{_format_json_path(field_path)}"
        dependency_path = _resolve_local_reference(
            unquote(uri),
            base_dir=mesh_path.parent,
            asset_root=mesh_path.parent,
            role=dependency_role,
            identity_recorder=identity_recorder,
        )
        records.append(
            {
                "kind": kind,
                **_asset_record(
                    dependency_path,
                    role=dependency_role,
                    reference=uri,
                    identity_recorder=identity_recorder,
                ),
            }
        )
    return records


def _gltf_uri_fields(
    value: Any,
    path: tuple[str | int, ...] = (),
) -> list[tuple[tuple[str | int, ...], Any]]:
    fields: list[tuple[tuple[str | int, ...], Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, str(key))
            if str(key).lower() == "uri":
                fields.append((child_path, child))
            else:
                fields.extend(_gltf_uri_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            fields.extend(_gltf_uri_fields(child, (*path, index)))
    return fields


def _format_json_path(path: tuple[str | int, ...]) -> str:
    return "".join(f"[{item}]" if isinstance(item, int) else f".{item}" for item in path)


def _glb_json_payload(mesh_path: Path, *, role: str) -> dict[str, Any]:
    """Read and validate the JSON chunk of a glTF 2.0 binary container."""

    file_size = mesh_path.stat().st_size
    with mesh_path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise ValueError(f"{role} GLB header is truncated: {mesh_path}")
        magic, version, declared_length = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2:
            raise ValueError(
                f"{role} is not a supported glTF 2.0 GLB: magic={magic!r}, version={version}"
            )
        if declared_length != file_size:
            raise ValueError(
                f"{role} GLB declared length does not match file size: "
                f"declared={declared_length}, actual={file_size}, path={mesh_path}"
            )
        if declared_length % 4 != 0:
            raise ValueError(f"{role} GLB file length is not 4-byte aligned: {mesh_path}")

        chunk_header = stream.read(8)
        if len(chunk_header) != 8:
            raise ValueError(f"{role} GLB has no complete JSON chunk: {mesh_path}")
        json_length, json_type = struct.unpack("<II", chunk_header)
        if json_type != 0x4E4F534A or json_length <= 0:
            raise ValueError(f"{role} GLB first chunk is not a non-empty JSON chunk: {mesh_path}")
        if json_length % 4 != 0:
            raise ValueError(f"{role} GLB JSON chunk is not 4-byte aligned: {mesh_path}")
        json_bytes = stream.read(json_length)
        if len(json_bytes) != json_length:
            raise ValueError(f"{role} GLB JSON chunk is truncated: {mesh_path}")

        offset = 20 + json_length
        while offset < declared_length:
            remaining_header = stream.read(8)
            if len(remaining_header) != 8:
                raise ValueError(f"{role} GLB chunk header is truncated: {mesh_path}")
            chunk_length, _chunk_type = struct.unpack("<II", remaining_header)
            if chunk_length % 4 != 0:
                raise ValueError(f"{role} GLB chunk is not 4-byte aligned: {mesh_path}")
            offset += 8 + chunk_length
            if offset > declared_length:
                raise ValueError(f"{role} GLB chunk exceeds its declared file length: {mesh_path}")
            stream.seek(chunk_length, os.SEEK_CUR)
        if offset != declared_length or stream.tell() != declared_length:
            raise ValueError(f"{role} GLB chunk table does not close at the file boundary: {mesh_path}")

    try:
        payload = json.loads(json_bytes.rstrip(b" \t\r\n\x00").decode("utf-8", errors="strict"))
    except Exception as exc:
        raise ValueError(f"{role} GLB has invalid JSON at {mesh_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{role} GLB JSON root must be an object: {mesh_path}")
    return payload


def _glb_dependency_records(
    mesh_path: Path,
    *,
    role: str,
    identity_recorder: AssetIdentityRecorder | None = None,
) -> list[dict[str, Any]]:
    return _gltf_payload_dependency_records(
        _glb_json_payload(mesh_path, role=role),
        mesh_path=mesh_path,
        role=role,
        identity_recorder=identity_recorder,
    )


def _dae_dependency_records(
    mesh_path: Path,
    *,
    role: str,
    identity_recorder: AssetIdentityRecorder | None = None,
) -> list[dict[str, Any]]:
    try:
        root = ET.parse(mesh_path).getroot()
    except Exception as exc:
        raise ValueError(f"{role} has invalid COLLADA XML at {mesh_path}: {exc}") from exc
    _reject_external_collada_document_references(root, mesh_path=mesh_path, role=role)
    records: list[dict[str, Any]] = []
    image_index = 0
    for image in _xml_elements(root, "image"):
        init_from_elements = _xml_elements(image, "init_from")
        if not init_from_elements:
            raise ValueError(f"{role} COLLADA <image> has no <init_from>: {mesh_path}")
        for element in init_from_elements:
            reference = (element.text or "").strip()
            if not reference:
                raise ValueError(f"{role} COLLADA <image>/<init_from> is empty: {mesh_path}")
            if reference.startswith("#"):
                raise ValueError(
                    f"{role} COLLADA image uses an unsupported internal reference {reference!r}: {mesh_path}"
                )
            if reference.lower().startswith("data:"):
                # Embedded bytes are already covered by the DAE file digest.
                image_index += 1
                continue
            dependency_path = _resolve_local_reference(
                reference,
                base_dir=mesh_path.parent,
                asset_root=mesh_path.parent,
                role=f"{role}.collada_texture[{image_index}]",
                identity_recorder=identity_recorder,
            )
            records.append(
                {
                    "kind": "texture",
                    **_asset_record(
                        dependency_path,
                        role=f"{role}.collada_texture[{image_index}]",
                        reference=reference,
                        identity_recorder=identity_recorder,
                    ),
                }
            )
            image_index += 1
    return records


def _reject_external_collada_document_references(
    root: ET.Element,
    *,
    mesh_path: Path,
    role: str,
) -> None:
    """Reject cross-document COLLADA links whose recursive closure is unaudited."""

    for element in root.iter():
        element_name = element.tag.rsplit("}", 1)[-1].lower()
        for raw_name, raw_value in element.attrib.items():
            attribute_name = raw_name.rsplit("}", 1)[-1].lower()
            is_document_reference = attribute_name in {"url", "href"} or (
                attribute_name == "source" and element_name in {"skin", "morph"}
            )
            if not is_document_reference:
                continue
            reference = str(raw_value).strip()
            if reference and not reference.startswith("#"):
                raise ValueError(
                    f"{role} COLLADA uses unsupported external document reference "
                    f"{reference!r} in <{element_name}>: {mesh_path}"
                )


def _ply_dependency_records(
    mesh_path: Path,
    *,
    role: str,
    identity_recorder: AssetIdentityRecorder | None = None,
) -> list[dict[str, Any]]:
    with mesh_path.open("rb") as stream:
        header = stream.read(1024 * 1024)
    marker = b"end_header"
    marker_index = header.find(marker)
    if marker_index < 0:
        raise ValueError(f"{role} PLY header has no end_header marker: {mesh_path}")
    try:
        header_text = header[:marker_index].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{role} PLY header is not ASCII: {mesh_path}") from exc
    records: list[dict[str, Any]] = []
    for index, line in enumerate(header_text.splitlines()):
        words = line.strip().split(maxsplit=2)
        if len(words) != 3 or words[0].lower() != "comment" or words[1].lower() != "texturefile":
            continue
        reference = words[2].strip()
        dependency_path = _resolve_local_reference(
            reference,
            base_dir=mesh_path.parent,
            asset_root=mesh_path.parent,
            role=f"{role}.ply_texture[{index}]",
            identity_recorder=identity_recorder,
        )
        records.append(
            {
                "kind": "texture",
                **_asset_record(
                    dependency_path,
                    role=f"{role}.ply_texture[{index}]",
                    reference=reference,
                    identity_recorder=identity_recorder,
                ),
            }
        )
    return records


def _mesh_record(
    mesh_path: Path,
    *,
    role: str,
    reference: str,
    identity_recorder: AssetIdentityRecorder | None = None,
) -> dict[str, Any]:
    mesh_path = mesh_path.expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"{role} does not exist or is not a regular file: {mesh_path}")
    mesh_identity = _stat_identity(mesh_path.stat())
    suffix = mesh_path.suffix.lower()
    if suffix == ".obj":
        dependencies = _obj_dependency_records(
            mesh_path,
            role=role,
            identity_recorder=identity_recorder,
        )
    elif suffix == ".gltf":
        dependencies = _gltf_dependency_records(
            mesh_path,
            role=role,
            identity_recorder=identity_recorder,
        )
    elif suffix == ".glb":
        dependencies = _glb_dependency_records(
            mesh_path,
            role=role,
            identity_recorder=identity_recorder,
        )
    elif suffix == ".dae":
        dependencies = _dae_dependency_records(
            mesh_path,
            role=role,
            identity_recorder=identity_recorder,
        )
    elif suffix == ".ply":
        dependencies = _ply_dependency_records(
            mesh_path,
            role=role,
            identity_recorder=identity_recorder,
        )
    elif suffix == ".stl":
        dependencies = []
    else:
        raise ValueError(
            f"{role} uses mesh format {suffix or '<none>'!r}, whose transitive external-asset "
            "closure is not implemented; refusing unverifiable scientific startup"
        )
    mesh_record = {
        **_asset_record(
            mesh_path,
            role=role,
            reference=reference,
            identity_recorder=identity_recorder,
        ),
        "format": suffix.lstrip("."),
        "dependencies": sorted(
            dependencies,
            key=lambda record: (record["kind"], record["reference"], record["sha256"]),
        ),
    }
    _require_file_identity(mesh_path, expected=mesh_identity, role=role)
    return mesh_record


def build_urdf_asset_manifest(
    urdf_path: Path | str,
    *,
    role: str,
    asset_root: Path | str | None = None,
    reference: str | None = None,
    require_mesh: bool = False,
    identity_recorder: AssetIdentityRecorder | None = None,
) -> dict[str, Any]:
    """Content-close one local URDF and every supported transitive asset.

    This is deliberately shared by robot provenance, AS object-bank
    provenance, and the Isaac object converter cache key.  A format that has
    no audited dependency scanner is rejected instead of being treated as a
    leaf file.
    """

    unresolved_urdf = Path(urdf_path).expanduser()
    if identity_recorder is not None:
        identity_recorder(unresolved_urdf)
    resolved_urdf = unresolved_urdf.resolve()
    if resolved_urdf.suffix.lower() != ".urdf":
        raise ValueError(f"{role} is not a URDF: {resolved_urdf}")
    resolved_asset_root = (
        Path(asset_root).expanduser().resolve()
        if asset_root is not None
        else resolved_urdf.parent
    )
    urdf_identity = _stat_identity(resolved_urdf.stat())
    urdf_reference = str(reference if reference is not None else resolved_urdf.name)
    urdf_record = _asset_record(
        resolved_urdf,
        role=role,
        reference=urdf_reference,
        identity_recorder=identity_recorder,
    )
    try:
        root = ET.parse(resolved_urdf).getroot()
    except Exception as exc:
        raise ValueError(f"{role} has invalid URDF XML at {resolved_urdf}: {exc}") from exc
    if root.tag.rsplit("}", 1)[-1] != "robot":
        raise ValueError(f"{role} root element is not <robot>: {resolved_urdf}")

    mesh_assets: list[dict[str, Any]] = []
    seen_meshes: set[Path] = set()
    for index, mesh in enumerate(_xml_elements(root, "mesh")):
        mesh_reference = str(mesh.get("filename", "") or "").strip()
        mesh_path = _resolve_local_reference(
            mesh_reference,
            base_dir=resolved_urdf.parent,
            asset_root=resolved_asset_root,
            role=f"{role}.mesh[{index}]",
            identity_recorder=identity_recorder,
        )
        if mesh_path in seen_meshes:
            continue
        seen_meshes.add(mesh_path)
        mesh_assets.append(
            _mesh_record(
                mesh_path,
                role=f"{role}.mesh[{index}]",
                reference=mesh_reference,
                identity_recorder=identity_recorder,
            )
        )
    if require_mesh and not mesh_assets:
        raise ValueError(f"{role} contains no mesh assets: {resolved_urdf}")

    texture_assets: list[dict[str, Any]] = []
    seen_textures: set[Path] = set()
    for index, texture in enumerate(_xml_elements(root, "texture")):
        texture_reference = str(texture.get("filename", "") or "").strip()
        texture_path = _resolve_local_reference(
            texture_reference,
            base_dir=resolved_urdf.parent,
            asset_root=resolved_asset_root,
            role=f"{role}.texture[{index}]",
            identity_recorder=identity_recorder,
        )
        if texture_path in seen_textures:
            continue
        seen_textures.add(texture_path)
        texture_assets.append(
            _asset_record(
                texture_path,
                role=f"{role}.texture[{index}]",
                reference=texture_reference,
                identity_recorder=identity_recorder,
            )
        )

    result = {
        "urdf": urdf_record,
        "mesh_assets": sorted(
            mesh_assets,
            key=lambda record: (record["reference"], record["sha256"]),
        ),
        "texture_assets": sorted(
            texture_assets,
            key=lambda record: (record["reference"], record["sha256"]),
        ),
    }
    _require_file_identity(resolved_urdf, expected=urdf_identity, role=role)
    return result


def urdf_asset_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Return a canonical digest for a standalone URDF closure."""

    return hashlib.sha256(_canonical_json(dict(manifest)).encode("utf-8")).hexdigest()


def object_urdf_conversion_cache_key(
    urdf_path: Path | str,
    *,
    collider_type: str,
    object_scale: tuple[float, float, float] | None,
) -> str:
    """Content-address an IsaacLab object-URDF conversion request."""

    resolved_urdf = Path(urdf_path).expanduser().resolve()
    source_manifest = build_urdf_asset_manifest(
        resolved_urdf,
        role="IsaacSim object converter source",
        asset_root=resolved_urdf.parent,
        reference=resolved_urdf.name,
        require_mesh=True,
    )
    identity = {
        "version": 2,
        "source_manifest_sha256": urdf_asset_manifest_sha256(source_manifest),
        "collider_type": normalize_object_collider_type(collider_type),
        "scale": list(object_scale) if object_scale is not None else None,
        "converter_semantics": {
            "fix_base": False,
            "make_instanceable": True,
            "merge_fixed_joints": True,
            "replace_cylinders_with_capsules": True,
        },
    }
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def _robot_manifest(config: Any) -> tuple[dict[str, Any], Path, Path]:
    try:
        asset = config.robot.asset
    except AttributeError as exc:
        raise ValueError("effective ExperimentConfig has no robot.asset configuration") from exc
    if getattr(asset, "usd_file", None) is not None:
        raise ValueError(
            "scientific runtime asset closure does not support robot.asset.usd_file/USD composition; "
            "configure a URDF source or add a complete USD dependency resolver"
        )
    asset_root = _resolve_asset_root(str(getattr(asset, "asset_root", "")))
    urdf_reference = str(getattr(asset, "urdf_file", "") or "").strip()
    if not urdf_reference:
        raise ValueError("robot.asset.urdf_file is empty")
    urdf_path = Path(urdf_reference).expanduser()
    if not urdf_path.is_absolute():
        urdf_path = asset_root / urdf_path
    urdf_path = urdf_path.resolve()
    closure = build_urdf_asset_manifest(
        urdf_path,
        role="robot URDF",
        asset_root=asset_root,
        reference=urdf_reference,
    )

    return (
        {
            "source_format": "urdf",
            "robot_type": str(getattr(asset, "robot_type", "") or ""),
            **closure,
        },
        asset_root,
        urdf_path,
    )


_TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_ENV_VALUES = frozenset({"", "0", "false", "no", "off"})


def _normalized_bool_env(
    name: str,
    *,
    environ: Mapping[str, str],
    default: bool,
) -> bool:
    raw = environ.get(name)
    if raw is None:
        return bool(default)
    normalized = raw.strip().lower()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _manager_bool_override(
    name: str,
    configured: bool,
    *,
    environ: Mapping[str, str],
) -> bool:
    return _normalized_bool_env(name, environ=environ, default=configured)


def normalize_object_spawn_mode(raw_value: str | None) -> str:
    """Normalize the object spawner selector or reject an unaudited mode."""

    normalized = "" if raw_value is None else raw_value.strip().lower()
    if normalized == "":
        return "urdf"
    if normalized == "auto":
        return "auto"
    if normalized in {
        "single_slot_multi_urdf",
        "single-slot-multi-urdf",
        "single_slot",
        "single-slot",
        "heterogeneous_single_slot",
        "heterogeneous-single-slot",
    }:
        return "single_slot_multi_urdf"
    if normalized in {"urdf", "mesh", "off", "disable", "disabled"}:
        return "urdf"
    if normalized in {"primitive", "primitives", "box", "cuboid"}:
        raise ValueError(
            "HOLOSOMA_OBJECT_SPAWN_MODE primitive/box spawning is disabled for scientific runs; "
            "use mesh URDF spawning"
        )
    raise ValueError(f"unsupported HOLOSOMA_OBJECT_SPAWN_MODE={raw_value!r}")


def normalize_object_collider_type(raw_value: str | None) -> str:
    """Normalize the exact IsaacLab URDF collider conversion strategy."""

    normalized = "convex_hull" if raw_value is None else raw_value.strip().lower()
    if normalized in {"convex_decomposition", "convex_decomp", "decomposition", "vhacd"}:
        return "convex_decomposition"
    if normalized in {"convex_hull", "hull"}:
        return "convex_hull"
    raise ValueError(f"unsupported HOLOSOMA_OBJECT_COLLIDER_TYPE={raw_value!r}")


def object_loader_semantics_from_env(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve every environment control that changes object physics/topology."""

    if environ is None:
        environ = os.environ
    raw_spawn_mode = environ.get("HOLOSOMA_OBJECT_SPAWN_MODE")
    spawn_mode = normalize_object_spawn_mode(raw_spawn_mode)
    disable_single_slot = _normalized_bool_env(
        "HOLOSOMA_DISABLE_HETEROGENEOUS_OBJECT_SINGLE_SLOT",
        environ=environ,
        default=False,
    )
    force_single_slot = _normalized_bool_env(
        "HOLOSOMA_FORCE_HETEROGENEOUS_OBJECT_SINGLE_SLOT",
        environ=environ,
        default=False,
    )
    single_slot_requested = spawn_mode == "single_slot_multi_urdf"
    if disable_single_slot and (force_single_slot or single_slot_requested):
        raise ValueError(
            "conflicting heterogeneous object topology controls: single-slot spawning is both disabled and forced"
        )
    rank_local_root_active = bool(environ.get("HOLOSOMA_RANK_LOCAL_MOTION_ROOT", "").strip())
    rank_local_sharding_enabled = rank_local_root_active and _normalized_bool_env(
        "HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED",
        environ=environ,
        default=True,
    )
    legacy_rank_sharding_requested = _normalized_bool_env(
        "HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK",
        environ=environ,
        default=False,
    )
    return {
        "spawn_mode": spawn_mode,
        "spawn_mode_explicit": bool(raw_spawn_mode is not None and raw_spawn_mode.strip()),
        "single_slot_requested": single_slot_requested,
        "disable_heterogeneous_single_slot": disable_single_slot,
        "force_heterogeneous_single_slot": force_single_slot,
        "require_single_slot_objects": _normalized_bool_env(
            "HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS",
            environ=environ,
            default=False,
        ),
        "collider_type": normalize_object_collider_type(
            environ.get("HOLOSOMA_OBJECT_COLLIDER_TYPE")
        ),
        "activate_contact_sensors": _normalized_bool_env(
            "HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS",
            environ=environ,
            default=True,
        ),
        "allow_legacy_object_urdf_fallback": _normalized_bool_env(
            "HOLOSOMA_ALLOW_LEGACY_OBJECT_URDF_FALLBACK",
            environ=environ,
            default=False,
        ),
        "rank_local_sharding_enabled": rank_local_sharding_enabled,
        "legacy_rank_sharding_requested": legacy_rank_sharding_requested,
        "legacy_rank_sharding_effective": (
            legacy_rank_sharding_requested and not rank_local_sharding_enabled
        ),
        "urdf_converter": {
            "schema_version": 2,
            "fix_base": False,
            "make_instanceable": True,
            "merge_fixed_joints": True,
            "force_usd_conversion": True,
            "replace_cylinders_with_capsules": True,
        },
    }


def _object_loader_manifest(
    config: Any,
    *,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    robot = getattr(config, "robot", None)
    object_config = getattr(robot, "object", None)
    enabled = bool(object_config is not None and getattr(object_config, "enabled", False))
    object_path_present = bool(
        str(getattr(object_config, "object_urdf_path", "") or "").strip()
        if object_config is not None
        else False
    )
    active = enabled and object_path_present
    result: dict[str, Any] = {
        "active": active,
        "configured_enabled": enabled,
        "object_path_present": object_path_present,
        "configured_scale": _json_value(getattr(object_config, "scale", None)),
        "semantics": None,
    }
    if active:
        result["semantics"] = object_loader_semantics_from_env(environ)
    return result


def _scene_manifest(config: Any) -> dict[str, Any]:
    simulator = getattr(config, "simulator", None)
    simulator_config = getattr(simulator, "config", None)
    scene = getattr(simulator_config, "scene", None)
    if scene is None:
        return {
            "scene_files": [],
            "rigid_objects": [],
            "configured_semantics": None,
        }
    if isinstance(scene, Mapping):
        scene_files = scene.get("scene_files", [])
        rigid_objects = scene.get("rigid_objects", [])
    else:
        scene_files = getattr(scene, "scene_files", [])
        rigid_objects = getattr(scene, "rigid_objects", [])
    if scene_files:
        raise ValueError(
            "scientific runtime asset closure does not yet support simulator scene.scene_files; "
            "refusing to omit USD/scene composition dependencies"
        )
    if rigid_objects:
        raise ValueError(
            "scientific runtime asset closure does not yet support simulator scene.rigid_objects; "
            "refusing to omit standalone scene-object dependencies"
        )
    return {
        "scene_files": [],
        "rigid_objects": [],
        "configured_semantics": {
            "replicate_physics": bool(
                scene.get("replicate_physics", True)
                if isinstance(scene, Mapping)
                else getattr(scene, "replicate_physics", True)
            ),
            "env_spacing": float(
                scene.get("env_spacing", 20.0)
                if isinstance(scene, Mapping)
                else getattr(scene, "env_spacing", 20.0)
            ),
        },
    }


def _bundled_far_tracking_manifest() -> dict[str, Any]:
    package_root = Path(get_holosoma_root()).resolve()
    implementation_root = package_root / "third_party" / "ft_warp_sensors"
    source_paths = sorted(implementation_root.glob("*.py"), key=lambda path: path.name)
    if not source_paths:
        raise FileNotFoundError(
            f"bundled far_tracking_warp implementation is missing: {implementation_root}"
        )
    return {
        "kind": "holosoma_bundled_ft_warp_sensors",
        "selection": "fixed",
        "sources": [
            _asset_record(
                path,
                role=f"bundled far_tracking_warp source {path.name}",
                reference=f"holosoma/third_party/ft_warp_sensors/{path.name}",
            )
            for path in source_paths
        ],
    }


def _resolve_defm_source_root(environ: Mapping[str, str]) -> Path:
    return resolve_defm_source_root(environ=environ, anchor=Path(__file__))


@dataclasses.dataclass(frozen=True)
class _DeFMRuntimeConfig:
    encoder_type: str
    encoder_pretrained: bool
    encoder_pretrained_path: str | None
    encoder_pretrained_sha256: str | None


def _module_defm_runtime_config(config: Any, role: str) -> tuple[bool, _DeFMRuntimeConfig | None]:
    """Return the effective actor/critic DeFM module, if module config exists."""

    algo_wrapper = getattr(config, "algo", None)
    algo_config = getattr(algo_wrapper, "config", None) if algo_wrapper is not None else None
    module_dict = getattr(algo_config, "module_dict", None) if algo_config is not None else None
    if module_dict is None:
        return False, None
    module_cfg = getattr(module_dict, role, None)
    layer_cfg = getattr(module_cfg, "layer_config", None) if module_cfg is not None else None
    if layer_cfg is None:
        return True, None
    perception_input_name = str(getattr(layer_cfg, "perception_input_name", "") or "").strip()
    encoder_type = str(getattr(layer_cfg, "perception_encoder_type", "") or "").strip().lower()
    if not perception_input_name or not encoder_type.startswith("defm_"):
        return True, None
    pretrained = getattr(layer_cfg, "perception_pretrained", None)
    if not isinstance(pretrained, bool):
        raise ValueError(f"Active {role} {encoder_type} perception_pretrained must be a boolean.")
    return True, _DeFMRuntimeConfig(
        encoder_type=encoder_type,
        encoder_pretrained=pretrained,
        encoder_pretrained_path=getattr(layer_cfg, "perception_pretrained_path", None),
        encoder_pretrained_sha256=getattr(layer_cfg, "perception_pretrained_sha256", None),
    )


def _perception_defm_runtime_config(perception: Any) -> _DeFMRuntimeConfig | None:
    if perception is None or not bool(getattr(perception, "enabled", False)):
        return None
    encoder_type = str(getattr(perception, "encoder_type", "") or "").strip().lower()
    if not encoder_type.startswith("defm_"):
        return None
    pretrained = getattr(perception, "encoder_pretrained", None)
    if not isinstance(pretrained, bool):
        raise ValueError(f"Active {encoder_type} perception encoder_pretrained must be a boolean.")
    return _DeFMRuntimeConfig(
        encoder_type=encoder_type,
        encoder_pretrained=pretrained,
        encoder_pretrained_path=getattr(perception, "encoder_pretrained_path", None),
        encoder_pretrained_sha256=getattr(perception, "encoder_pretrained_sha256", None),
    )


def _defm_source_file_manifest(environ: Mapping[str, str]) -> list[dict[str, Any]]:
    source_root = _resolve_defm_source_root(environ)
    source_paths = sorted(
        (
            path
            for path in source_root.rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".yaml", ".yml"}
            and ".git" not in path.parts
            and "__pycache__" not in path.parts
        ),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    if not source_paths:
        raise ValueError(f"Pinned DeFM source tree contains no runtime files: {source_root}")
    return [
        _asset_record(
            path,
            role=f"DeFM source {path.relative_to(source_root).as_posix()}",
            reference=f"submodules/defm/{path.relative_to(source_root).as_posix()}",
        )
        for path in source_paths
    ]


def _distribution_content_manifest(distribution_name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise FileNotFoundError(
            f"Active DeFM runtime requires Python distribution {distribution_name!r}."
        ) from exc
    declared_files = distribution.files
    if declared_files is None:
        raise ValueError(
            f"Python distribution {distribution_name!r} exposes no auditable installed-file manifest."
        )
    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    for relative_path in sorted(declared_files, key=lambda value: str(value)):
        path = Path(distribution.locate_file(relative_path)).resolve()
        if not path.is_file():
            continue
        file_sha256, size = _stable_sha256_file(
            path,
            role=f"DeFM Python dependency {distribution_name}:{relative_path}",
            allow_empty=True,
        )
        relative_text = str(relative_path).replace(os.sep, "/")
        digest.update(relative_text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha256.encode("ascii"))
        digest.update(b"\n")
        file_count += 1
        total_size += size
    if file_count == 0:
        raise ValueError(f"Python distribution {distribution_name!r} has no regular installed files.")
    return {
        "distribution": str(distribution.metadata.get("Name") or distribution_name),
        "version": str(distribution.version),
        "file_count": file_count,
        "total_size": total_size,
        "content_manifest_sha256": digest.hexdigest(),
    }


def _defm_python_dependency_manifest() -> list[dict[str, Any]]:
    # HoloSoma constructs DeFM without upstream weight loading, then verifies
    # and strictly loads a required local checkpoint. The optional
    # huggingface_hub download branch is therefore not executable scientific
    # state and must not bind local/non-DeFM runs to a network client.
    return [
        _distribution_content_manifest(name)
        for name in ("torchvision", "omegaconf", "Pillow")
    ]


def _defm_runtime_manifest(
    perception: Any,
    *,
    environ: Mapping[str, str],
    consume_pretrained_checkpoint: bool,
    source_files: list[dict[str, Any]] | None = None,
    python_dependencies: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    encoder_type = str(getattr(perception, "encoder_type", "") or "").strip().lower()
    if source_files is None:
        source_files = _defm_source_file_manifest(environ)
    if python_dependencies is None:
        python_dependencies = _defm_python_dependency_manifest()

    pretrained = getattr(perception, "encoder_pretrained", None)
    if not isinstance(pretrained, bool):
        raise ValueError("DeFM perception encoder_pretrained must be a boolean.")
    declared_path = getattr(perception, "encoder_pretrained_path", None)
    declared_sha256 = getattr(perception, "encoder_pretrained_sha256", None)
    weight_record = None
    if pretrained:
        if not isinstance(declared_path, str) or not declared_path.strip():
            raise ValueError(
                "Active pretrained DeFM perception requires a local encoder_pretrained_path."
            )
        if (
            not isinstance(declared_sha256, str)
            or len(declared_sha256) != 64
            or any(char not in "0123456789abcdef" for char in declared_sha256)
        ):
            raise ValueError(
                "Active pretrained DeFM perception requires encoder_pretrained_sha256."
            )
        if consume_pretrained_checkpoint:
            weight_path = Path(declared_path).expanduser()
            if not weight_path.is_absolute():
                weight_path = Path.cwd() / weight_path
            weight_path = weight_path.resolve()
            if not weight_path.is_file():
                raise FileNotFoundError(f"DeFM pretrained checkpoint does not exist: {weight_path}")
            weight_record = _asset_record(
                weight_path,
                role=f"{encoder_type} pretrained checkpoint",
                reference=declared_path,
            )
            if weight_record["sha256"] != declared_sha256:
                raise ValueError(
                    f"{encoder_type} pretrained checkpoint SHA256 mismatch: "
                    f"expected={declared_sha256} actual={weight_record['sha256']}"
                )
        else:
            # Full resume (or actor-only policy init without critic injection)
            # constructs architecture only and immediately restores strict
            # checkpoint state. Record the declared lineage without reopening
            # an irrelevant external weight file.
            weight_record = {
                "reference": declared_path,
                "size": None,
                "sha256": declared_sha256,
            }
    elif declared_path is not None or declared_sha256 is not None:
        raise ValueError(
            "DeFM pretrained weights are disabled but a pretrained path/SHA is still configured."
        )

    return {
        "encoder_type": encoder_type,
        "source_files": source_files,
        "python_dependencies": python_dependencies,
        "xformers_disabled": True,
        "pretrained": pretrained,
        "pretrained_checkpoint_consumed": bool(
            pretrained and consume_pretrained_checkpoint
        ),
        "pretrained_checkpoint": weight_record,
    }


def _single_perception_manager_manifest(
    perception: Any,
    *,
    urdf_path: Path,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    enabled = bool(perception is not None and getattr(perception, "enabled", False))
    output_mode = str(getattr(perception, "output_mode", "") or "") if perception is not None else ""
    camera_source = str(getattr(perception, "camera_source", "") or "") if perception is not None else ""
    configured_include = bool(getattr(perception, "camera_include_robot_mesh", False)) if perception is not None else False
    effective_include = _manager_bool_override(
        "HOLOSOMA_PERCEPTION_INCLUDE_ROBOT_MESH",
        configured_include,
        environ=environ,
    )
    far_tracking_active = (
        enabled
        and output_mode == "camera_depth"
        and camera_source == "far_tracking_warp"
    )
    explicit_map_active = (
        far_tracking_active
        and effective_include
    )
    configured_object_geometry_mode = (
        getattr(perception, "object_geometry_mode", None) if perception is not None else None
    )
    raw_object_geometry_mode = (
        configured_object_geometry_mode
        if configured_object_geometry_mode is not None
        else environ.get("HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE", "")
    )
    normalized_object_geometry_mode = str(raw_object_geometry_mode or "").strip().lower()
    if normalized_object_geometry_mode in {"", "mesh", "urdf", "off", "false", "0", "no"}:
        normalized_object_geometry_mode = "mesh"
    else:
        raise ValueError(
            "scientific runtime asset closure supports only mesh URDF object perception geometry; "
            f"got {raw_object_geometry_mode!r}"
        )
    result: dict[str, Any] = {
        "enabled": enabled,
        "output_mode": output_mode,
        "camera_source": camera_source,
        "camera_warp_normalize": (
            bool(getattr(perception, "camera_warp_normalize", False))
            if perception is not None
            else None
        ),
        "encoder_type": (
            str(getattr(perception, "encoder_type", "") or "")
            if perception is not None
            else ""
        ),
        "far_tracking_implementation": (
            _bundled_far_tracking_manifest() if far_tracking_active else None
        ),
        "camera_include_robot_mesh": effective_include,
        # Object URDF/mesh bytes remain owned by the launcher's existing
        # motion_shard_manifest_sha256 closure.  Record only the effective
        # selection semantics here instead of duplicating a partial file claim.
        "object_geometry_mode": normalized_object_geometry_mode,
        "explicit_camera_mesh_map_active": explicit_map_active,
        "explicit_camera_meshes": [],
    }
    if not explicit_map_active:
        return result

    raw_map = getattr(perception, "camera_mesh_file_map", None)
    if not isinstance(raw_map, Mapping) or not raw_map:
        raise ValueError(
            "active far_tracking_warp robot-mesh perception requires a non-empty "
            "perception.camera_mesh_file_map"
        )
    disable_combined = _normalized_bool_env(
        "HOLOSOMA_FAR_TRACKING_DISABLE_COMBINED_DEPTH_MESHES",
        environ=environ,
        default=False,
    )
    mesh_root = urdf_path.parent / "meshes"
    explicit_records: list[dict[str, Any]] = []
    for link_name_raw, mesh_reference_raw in sorted(raw_map.items(), key=lambda item: str(item[0])):
        link_name = str(link_name_raw).strip()
        mesh_reference = str(mesh_reference_raw).strip()
        if not link_name:
            raise ValueError("perception.camera_mesh_file_map contains an empty link name")
        if not mesh_reference:
            raise ValueError(f"perception.camera_mesh_file_map[{link_name!r}] is empty")
        if mesh_reference.lower().startswith(_REMOTE_URI_PREFIXES) or mesh_reference.lower().startswith("data:"):
            raise ValueError(
                f"perception.camera_mesh_file_map[{link_name!r}] is non-local: {mesh_reference!r}"
            )
        declared_path = Path(mesh_reference).expanduser()
        if not declared_path.is_absolute():
            # This exactly matches far_tracking_warp's asset_meshes_root,
            # including relative subdirectories when explicitly configured.
            declared_path = mesh_root / declared_path
        declared_path = declared_path.resolve()
        declared_asset = _mesh_record(
            declared_path,
            role=f"perception.camera_mesh_file_map[{link_name!r}]",
            reference=mesh_reference,
        )

        if disable_combined and mesh_reference.startswith("combined_"):
            candidates = [f"{link_name}.STL", f"{link_name}.stl", mesh_reference]
        else:
            candidates = [mesh_reference, f"{link_name}.STL", f"{link_name}.stl"]
        selected_reference = next(
            (
                candidate
                for candidate in candidates
                if (mesh_root / Path(candidate).expanduser()).resolve().is_file()
            ),
            None,
        )
        if selected_reference is None:
            raise FileNotFoundError(
                f"no runtime far_tracking_warp mesh exists for link {link_name!r} under {mesh_root}"
            )
        selected_path = (mesh_root / Path(selected_reference).expanduser()).resolve()
        selected_asset = _mesh_record(
            selected_path,
            role=f"far_tracking_warp selected mesh[{link_name!r}]",
            reference=selected_reference,
        )
        explicit_records.append(
            {
                "link_name": link_name,
                "declared_asset": declared_asset,
                "selected_runtime_asset": selected_asset,
            }
        )
    result["disable_combined_depth_meshes"] = disable_combined
    result["explicit_camera_meshes"] = explicit_records
    return result


def _distill_perception_manager_config(config: Any, preset_attr: str) -> Any | None:
    algo_wrapper = getattr(config, "algo", None)
    algo_config = getattr(algo_wrapper, "config", None) if algo_wrapper is not None else None
    distill = getattr(algo_config, "distill", None) if algo_config is not None else None
    preset_name = str(getattr(distill, preset_attr, "") or "").strip() if distill is not None else ""
    if not preset_name or preset_name.lower() == "none":
        return None
    from holosoma.config_values import perception as perception_values  # noqa: PLC0415

    if preset_name not in perception_values.DEFAULTS:
        raise ValueError(f"Unknown distill.{preset_attr}: {preset_name}")
    return perception_values.DEFAULTS[preset_name]


def _perception_manifest(
    config: Any,
    *,
    urdf_path: Path,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    """Close all perception managers and policy-side external encoders by role."""

    student_perception = getattr(config, "perception", None)
    teacher_perception = _distill_perception_manager_config(
        config,
        "teacher_perception_preset",
    )
    critic_perception = _distill_perception_manager_config(
        config,
        "critic_perception_preset",
    )
    manager_roles = {
        role: _single_perception_manager_manifest(
            perception,
            urdf_path=urdf_path,
            environ=environ,
        )
        for role, perception in (
            ("student", student_perception),
            ("teacher", teacher_perception),
            ("critic", critic_perception),
        )
    }

    actor_module_configured, actor_defm = _module_defm_runtime_config(config, "actor")
    critic_module_configured, critic_defm = _module_defm_runtime_config(config, "critic")
    if not actor_module_configured and bool(
        getattr(student_perception, "inject_into_policy_modules", True)
    ):
        actor_defm = _perception_defm_runtime_config(student_perception)
    if not critic_module_configured:
        if critic_perception is not None:
            critic_defm = _perception_defm_runtime_config(critic_perception)
        elif bool(getattr(student_perception, "inject_into_critic_modules", False)):
            critic_defm = _perception_defm_runtime_config(student_perception)

    def manager_uses_defm(perception: Any) -> bool:
        return bool(
            perception is not None
            and getattr(perception, "enabled", False)
            and str(getattr(perception, "encoder_type", "") or "").strip().lower().startswith("defm_")
        )

    defm_runtime_active = bool(
        actor_defm is not None
        or critic_defm is not None
        or any(
            manager_uses_defm(perception)
            for perception in (student_perception, teacher_perception, critic_perception)
        )
    )
    defm_source_files = _defm_source_file_manifest(environ) if defm_runtime_active else None
    defm_python_dependencies = _defm_python_dependency_manifest() if defm_runtime_active else None

    training = getattr(config, "training", None)
    full_resume = bool(training is not None and getattr(training, "checkpoint", None) is not None)
    policy_init = bool(
        training is not None and getattr(training, "policy_init_checkpoint", None) is not None
    )
    policy_encoder_roles = {
        "actor": (
            _defm_runtime_manifest(
                actor_defm,
                environ=environ,
                consume_pretrained_checkpoint=not full_resume and not policy_init,
                source_files=defm_source_files,
                python_dependencies=defm_python_dependencies,
            )
            if actor_defm is not None
            else None
        ),
        "critic": (
            _defm_runtime_manifest(
                critic_defm,
                environ=environ,
                consume_pretrained_checkpoint=not full_resume,
                source_files=defm_source_files,
                python_dependencies=defm_python_dependencies,
            )
            if critic_defm is not None
            else None
        ),
    }

    # Preserve the original student-manager fields for consumers of the v2
    # manifest while making all additional managers and actual model roles
    # explicit.  ``defm`` is now an actor-policy compatibility alias rather
    # than a claim that PerceptionManager itself instantiates the encoder.
    result = dict(manager_roles["student"])
    result["manager_roles"] = manager_roles
    result["policy_encoder_roles"] = policy_encoder_roles
    result["defm"] = policy_encoder_roles["actor"]
    result["defm_runtime_source_files"] = defm_source_files
    result["defm_python_dependencies"] = defm_python_dependencies
    result["defm_xformers_disabled"] = True if defm_runtime_active else None
    result["teacher_policy_encoder_authority"] = "authenticated_teacher_checkpoint"
    return result


def _terrain_manifest(config: Any) -> dict[str, Any]:
    try:
        terrain_term = config.terrain.terrain_term
    except AttributeError as exc:
        raise ValueError("effective ExperimentConfig has no terrain.terrain_term configuration") from exc
    raw_mesh_type = getattr(terrain_term, "mesh_type", None)
    mesh_type = str(getattr(raw_mesh_type, "value", raw_mesh_type)).strip().lower()
    if mesh_type != "plane":
        raise ValueError(
            f"scientific runtime asset closure currently supports only analytic plane terrain, got {mesh_type!r}; "
            "procedural/load_obj/USD terrain must add a complete runtime dependency closure before launch"
        )
    terrain_func = str(getattr(terrain_term, "func", "") or "").strip()
    if terrain_func != "holosoma.managers.terrain.terms.locomotion:TerrainLocomotion":
        raise ValueError(
            "scientific plane-terrain closure supports only the audited TerrainLocomotion implementation, "
            f"got {terrain_func!r}"
        )
    terrain_config = getattr(terrain_term, "terrain_config", {})
    if terrain_config:
        raise ValueError("plane terrain must not configure terrain_config; the runtime expects an analytic flat plane")
    obj_file_path = str(getattr(terrain_term, "obj_file_path", "") or "").strip()
    obj_metadata_path = str(getattr(terrain_term, "obj_metadata_path", "") or "").strip()
    if obj_file_path or obj_metadata_path:
        raise ValueError(
            "plane terrain must not declare OBJ paths; they are ignored by the runtime and would create a false asset claim"
        )
    configured_semantics = _json_value(terrain_term)
    return {
        "mesh_type": "plane",
        "external_assets": [],
        "configured_semantics": configured_semantics,
        "isaacsim_collision_semantics": {
            "terrain_type": "plane",
            "friction_combine_mode": "multiply",
            "restitution_combine_mode": "multiply",
            "static_friction": float(getattr(terrain_term, "static_friction")),
            "dynamic_friction": float(getattr(terrain_term, "dynamic_friction")),
            # IsaacSim._setup_scene deliberately fixes analytic-plane
            # restitution to zero, independent of the generic config field.
            "restitution": 0.0,
        },
    }


def build_runtime_asset_manifest(
    config: Any,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a path-independent, content-addressed manifest for active assets."""

    if environ is None:
        environ = os.environ
    simulator = getattr(config, "simulator", None)
    target = str(getattr(simulator, "_target_", "") or "")
    if target != _ISAACSIM_TARGET:
        raise ValueError(
            f"scientific runtime asset closure is implemented for {_ISAACSIM_TARGET!r}, got {target!r}"
        )
    robot, _asset_root, urdf_path = _robot_manifest(config)
    manifest = {
        "version": RUNTIME_ASSET_MANIFEST_VERSION,
        "simulator": {"target": target},
        "scene": _scene_manifest(config),
        "robot": robot,
        "object_loader": _object_loader_manifest(config, environ=environ),
        "perception": _perception_manifest(
            config,
            urdf_path=urdf_path,
            environ=environ,
        ),
        "terrain": _terrain_manifest(config),
    }
    # Assert serializability and reject NaN/Infinity before hashing.
    _canonical_json(manifest)
    return manifest


def runtime_asset_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest using its canonical, path-independent JSON encoding."""

    return embedded_runtime_asset_manifest_sha256(dict(manifest))


def persist_runtime_asset_manifest(
    path: Path | str,
    provenance: Mapping[str, Any],
) -> Path:
    """Atomically persist and re-verify a finalized embedded manifest."""

    finalized = validate_training_provenance(dict(provenance), require_finalized=True)
    manifest = finalized[RUNTIME_ASSET_MANIFEST_KEY]
    canonical = canonical_runtime_asset_manifest_json(manifest)
    declared_digest = finalized[RUNTIME_ASSET_DIGEST_KEY]
    actual_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if actual_digest != declared_digest:  # defensive: validator already proves this.
        raise ValueError(
            "refusing to persist runtime asset manifest with a mismatched digest: "
            f"declared={declared_digest} actual={actual_digest}"
        )

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(canonical + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    persisted = json.loads(destination.read_text(encoding="utf-8"))
    persisted_digest = embedded_runtime_asset_manifest_sha256(persisted)
    if persisted_digest != declared_digest:
        raise RuntimeError(
            "persisted runtime asset manifest failed digest verification: "
            f"declared={declared_digest} persisted={persisted_digest} path={destination}"
        )
    return destination


def finalize_runtime_asset_provenance(
    config: Any,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Replace a launch-time pending asset sentinel with the effective digest.

    An absent provenance environment keeps ordinary, non-scientific launches
    unchanged.  A present provenance payload must be v2.  Re-finalization is
    idempotent only when the current effective assets produce the same digest.
    """

    if environ is None:
        environ = os.environ
    raw = environ.get(ENV_NAME)
    if raw is None or not raw.strip():
        return None
    try:
        provenance_raw = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {ENV_NAME} JSON: {exc}") from exc
    provenance = validate_training_provenance(provenance_raw, require_finalized=False)
    validate_semantic_environment_binding(provenance, environ=environ)
    manifest = build_runtime_asset_manifest(config, environ=environ)
    digest = runtime_asset_manifest_sha256(manifest)
    phase = provenance[RUNTIME_ASSET_PHASE_KEY]
    if phase == RUNTIME_ASSET_PHASE_PENDING:
        if provenance[RUNTIME_ASSET_DIGEST_KEY] != pending_runtime_asset_manifest_sha256():
            raise ValueError("pending runtime asset provenance uses an invalid digest sentinel")
        provenance[RUNTIME_ASSET_PHASE_KEY] = RUNTIME_ASSET_PHASE_FINAL
        provenance[RUNTIME_ASSET_DIGEST_KEY] = digest
        provenance[RUNTIME_ASSET_MANIFEST_KEY] = manifest
    elif phase == RUNTIME_ASSET_PHASE_FINAL:
        if provenance[RUNTIME_ASSET_DIGEST_KEY] != digest:
            raise ValueError(
                "already-finalized runtime asset provenance does not match the effective config/assets: "
                f"declared={provenance[RUNTIME_ASSET_DIGEST_KEY]} actual={digest}"
            )
        if provenance[RUNTIME_ASSET_MANIFEST_KEY] != manifest:
            raise ValueError(
                "already-finalized runtime asset provenance embeds a manifest that does not match "
                "the effective config/assets"
            )
    else:  # pragma: no cover - validate_training_provenance rejects this first.
        raise ValueError(f"unsupported runtime asset provenance phase: {phase!r}")
    finalized = validate_training_provenance(provenance, require_finalized=True)
    environ[ENV_NAME] = canonical_training_provenance_json(finalized)
    return finalized
