#!/usr/bin/env python3
"""Export a gsplat PLY to a NuRec USD stage via an external exporter (e.g., 3DGruT)."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_PROPS = {"x", "y", "z"}
OPTIONAL_GSPLAT_PROPS = {
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
}
USD_EXTS = (".usdz", ".usd", ".usda", ".usdc")


def _read_ply_header(path: Path) -> tuple[str, int, list[str]]:
    fmt = ""
    vertex_count = -1
    vertex_props: list[str] = []
    current_element = None

    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading PLY header.")
            text = line.decode("ascii", errors="ignore").strip()
            if text == "end_header":
                break
            if not text:
                continue
            parts = text.split()
            if parts[0] == "format":
                if len(parts) < 2:
                    raise ValueError(f"Malformed PLY format line: {text}")
                fmt = parts[1]
            elif parts[0] == "element":
                if len(parts) < 3:
                    raise ValueError(f"Malformed PLY element line: {text}")
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                if len(parts) < 3:
                    raise ValueError(f"Malformed PLY property line: {text}")
                if parts[1] == "list":
                    continue
                vertex_props.append(parts[2])

    if not fmt:
        raise ValueError("PLY header missing format line.")
    if vertex_count < 0:
        raise ValueError("PLY header missing vertex count.")
    if not vertex_props:
        raise ValueError("PLY header has no vertex properties.")

    return fmt, vertex_count, vertex_props


def _validate_ply(path: Path) -> None:
    fmt, vertex_count, props = _read_ply_header(path)
    if fmt not in {"ascii", "binary_little_endian"}:
        raise ValueError(f"Unsupported PLY format: {fmt}")
    missing = REQUIRED_PROPS.difference(props)
    if missing:
        raise ValueError(f"PLY is missing required properties: {', '.join(sorted(missing))}")
    optional_missing = OPTIONAL_GSPLAT_PROPS.difference(props)
    if optional_missing:
        print(
            "Warning: PLY is missing some gsplat properties (ok if exporter supports this): "
            + ", ".join(sorted(optional_missing)),
            file=sys.stderr,
        )
    print(f"PLY header ok: format={fmt}, vertices={vertex_count}, props={len(props)}")


def _format_command(template: str, *, input_path: Path, output_dir: Path) -> list[str]:
    command = template.format(input=str(input_path), output_dir=str(output_dir), output=str(output_dir))
    return shlex.split(command)


def _run_export(command: list[str]) -> None:
    print("Running exporter:")
    print("  " + " ".join(shlex.quote(arg) for arg in command))
    subprocess.run(command, check=True)


def _resolve_exported_usd(output_dir: Path, exported_usd: str | None) -> Path:
    if exported_usd:
        path = Path(exported_usd)
        if not path.is_absolute():
            path = output_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Exported USD file not found: {path}")
        return path

    candidates: list[Path] = []
    for ext in USD_EXTS:
        candidates.extend(output_dir.rglob(f"*{ext}"))

    if not candidates:
        raise FileNotFoundError(
            f"No USD assets found in {output_dir}. "
            "Pass --exported-usd if the exporter writes elsewhere."
        )
    if len(candidates) > 1:
        listing = "\n".join(f"- {p}" for p in sorted(candidates))
        raise ValueError(
            "Multiple USD assets found. Pass --exported-usd to pick one:\n" + listing
        )
    return candidates[0]


def _write_stage_usda(stage_path: Path, referenced_asset: Path) -> None:
    rel = os.path.relpath(referenced_asset, stage_path.parent)
    rel = Path(rel).as_posix()
    contents = f"""#usda 1.0
(
    upAxis = \"Z\"
    metersPerUnit = 1
)

def Xform \"World\" (
    references = @{rel}@
)
{{
}}
"""
    stage_path.write_text(contents, encoding="utf-8")


def _create_stage(exported_usd: Path, output_dir: Path, stage_name: str | None, overwrite: bool) -> Path:
    if stage_name:
        stage_path = output_dir / stage_name
    else:
        stage_path = output_dir / ("stage.usdz" if exported_usd.suffix == ".usdz" else "stage.usda")

    if stage_path.exists():
        if not overwrite:
            raise FileExistsError(f"Stage already exists: {stage_path} (use --overwrite to replace)")
        stage_path.unlink()

    if exported_usd.suffix == ".usdz" and stage_path.suffix == ".usdz":
        if exported_usd.resolve() != stage_path.resolve():
            shutil.copy2(exported_usd, stage_path)
        return stage_path

    if stage_path.suffix not in {".usd", ".usda"}:
        raise ValueError(
            f"Stage file {stage_path} must end with .usd/.usda when referencing non-usdz assets."
        )
    _write_stage_usda(stage_path, exported_usd)
    return stage_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrap a 3DGruT (or similar) exporter to produce a NuRec-ready USD stage."
    )
    parser.add_argument("--input-ply", required=True, help="Path to gsplat-exported PLY.")
    parser.add_argument("--output-dir", required=True, help="Directory where USD assets will be written.")
    parser.add_argument(
        "--export-command",
        default=None,
        help=(
            "Exporter command template with {input} and {output_dir} placeholders. "
            "Example: \"3dgrut export --input {input} --output {output_dir}\""
        ),
    )
    parser.add_argument(
        "--exported-usd",
        default=None,
        help="Explicit exported USD/USDZ path (relative to output-dir if not absolute).",
    )
    parser.add_argument(
        "--stage-name",
        default=None,
        help="Stage file name to create (default: stage.usdz or stage.usda depending on export).",
    )
    parser.add_argument("--skip-export", action="store_true", help="Skip running exporter step.")
    parser.add_argument("--no-validate", action="store_true", help="Skip PLY header validation.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing stage file.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_ply).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Input PLY not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_validate:
        _validate_ply(input_path)

    if not args.skip_export:
        if not args.export_command:
            raise SystemExit("--export-command is required unless --skip-export is set.")
        command = _format_command(args.export_command, input_path=input_path, output_dir=output_dir)
        _run_export(command)

    exported_usd = _resolve_exported_usd(output_dir, args.exported_usd)
    stage_path = _create_stage(exported_usd, output_dir, args.stage_name, args.overwrite)

    print("NuRec stage ready:")
    print(f"  exported asset: {exported_usd}")
    print(f"  stage file: {stage_path}")


if __name__ == "__main__":
    main()
