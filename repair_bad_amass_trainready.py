#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "src/holosoma_retargeting/converted_res/robot_only/amass_all_trainready"
DEFAULT_AMASS_ROOT = REPO_ROOT / "amass"
DEFAULT_CACHE_ROOT = REPO_ROOT / ".cache/amass_all_proxy"
DEFAULT_VIS_SCRIPT = REPO_ROOT / "vis_amass.sh"
DEFAULT_CONVERTER = REPO_ROOT / "src/holosoma_retargeting_my/data_conversion/convert_data_format_mj.py"
DEFAULT_SCENE_XML = REPO_ROOT / "src/holosoma_retargeting_my/models/g1/g1_29dof.xml"
DEFAULT_MANIFEST = REPO_ROOT / "logs/amass_all_trainready_bad_npz_latest.tsv"
REQUIRED_KEYS = {
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "joint_names",
    "body_names",
    "fps",
}


@dataclass(frozen=True)
class BrokenClip:
    rel_path: Path
    output_path: Path
    source_path: Path
    proxy_path: Path
    size_bytes: int
    reason: str


def _proxy_subdir_for_rel_dir(rel_dir: Path) -> str:
    rel_str = rel_dir.as_posix()
    if rel_str in ("", "."):
        return "_root"
    return rel_str.replace("/", "__")


def _source_rel_from_output_rel(output_rel: Path, output_fps: int) -> Path:
    suffix = f"_mj_fps{output_fps}"
    stem = output_rel.stem
    if not stem.endswith(suffix):
        raise ValueError(f"Output clip does not end with '{suffix}': {output_rel}")
    source_stem = stem[: -len(suffix)]
    return output_rel.with_name(f"{source_stem}.npz")


def _iter_npz_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.npz") if path.is_file())


def scan_broken_clips(
    *,
    output_root: Path,
    amass_root: Path,
    cache_root: Path,
    output_fps: int,
) -> list[BrokenClip]:
    broken: list[BrokenClip] = []
    for output_path in _iter_npz_files(output_root):
        rel_path = output_path.relative_to(output_root)
        source_rel = _source_rel_from_output_rel(rel_path, output_fps)
        proxy_subdir = _proxy_subdir_for_rel_dir(source_rel.parent)
        reason: str | None = None
        if not zipfile.is_zipfile(output_path):
            reason = "not a valid .npz zip archive"
        else:
            try:
                with np.load(output_path, allow_pickle=True) as data:
                    missing = sorted(REQUIRED_KEYS - set(data.files))
                if missing:
                    reason = f"missing required keys: {missing}"
            except Exception as exc:
                reason = f"failed to inspect npz: {type(exc).__name__}: {exc}"

        if reason is None:
            continue
        broken.append(
            BrokenClip(
                rel_path=rel_path,
                output_path=output_path,
                source_path=amass_root / source_rel,
                proxy_path=cache_root / proxy_subdir / source_rel.name,
                size_bytes=output_path.stat().st_size,
                reason=reason,
            )
        )
    return broken


def write_manifest(clips: list[BrokenClip], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "rel_path",
                "output_path",
                "source_path",
                "proxy_path",
                "size_bytes",
                "source_exists",
                "proxy_exists",
                "reason",
            ]
        )
        for clip in clips:
            writer.writerow(
                [
                    clip.rel_path.as_posix(),
                    str(clip.output_path),
                    str(clip.source_path),
                    str(clip.proxy_path),
                    str(clip.size_bytes),
                    "1" if clip.source_path.exists() else "0",
                    "1" if clip.proxy_path.exists() else "0",
                    clip.reason,
                ]
            )


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def ensure_proxy_dir(
    *,
    source_dir: Path,
    proxy_dir: Path,
    vis_script: Path,
    python_bin: str,
    order_mode: str,
    wrist_policy: str,
) -> None:
    proxy_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "AMASS_SRC_DIR": str(source_dir),
            "CACHE_DIR": str(proxy_dir),
            "ORDER_MODE": order_mode,
            "WRIST_POLICY": wrist_policy,
            "MAX_CLIPS": "0",
            "CONVERT_ONLY": "True",
            "PYTHON_BIN": python_bin,
        }
    )
    _run(["bash", str(vis_script)], env=env)


def repair_clips(
    *,
    clips: list[BrokenClip],
    amass_root: Path,
    cache_root: Path,
    vis_script: Path,
    converter: Path,
    scene_xml_file: Path,
    python_bin: str,
    robot: str,
    output_fps: int,
    order_mode: str,
    wrist_policy: str,
) -> tuple[int, int]:
    repaired = 0
    failed = 0
    rebuilt_proxy_dirs: set[Path] = set()

    for clip in clips:
        try:
            if not clip.source_path.exists():
                raise FileNotFoundError(f"source motion not found: {clip.source_path}")

            if not clip.proxy_path.exists():
                proxy_dir = clip.proxy_path.parent
                if proxy_dir not in rebuilt_proxy_dirs:
                    source_dir = clip.source_path.parent
                    ensure_proxy_dir(
                        source_dir=source_dir,
                        proxy_dir=proxy_dir,
                        vis_script=vis_script,
                        python_bin=python_bin,
                        order_mode=order_mode,
                        wrist_policy=wrist_policy,
                    )
                    rebuilt_proxy_dirs.add(proxy_dir)

            if not clip.proxy_path.exists():
                raise FileNotFoundError(f"proxy motion not found after rebuild: {clip.proxy_path}")

            clip.output_path.parent.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    python_bin,
                    str(converter),
                    "--input-file",
                    str(clip.proxy_path),
                    "--robot",
                    robot,
                    "--output-fps",
                    str(output_fps),
                    "--data-format",
                    "lafan",
                    "--object-name",
                    "ground",
                    "--scene-xml-file",
                    str(scene_xml_file),
                    "--output-name",
                    str(clip.output_path),
                    "--once",
                    "--headless",
                ]
            )
            if not zipfile.is_zipfile(clip.output_path):
                raise RuntimeError(f"output still invalid after regeneration: {clip.output_path}")

            repaired += 1
            print(f"[OK] repaired {clip.rel_path.as_posix()}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {clip.rel_path.as_posix()} :: {exc}", file=sys.stderr)

    return repaired, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan and optionally repair broken *_mj_fps50.npz clips in amass_all_trainready."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--amass-root", type=Path, default=DEFAULT_AMASS_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--vis-script", type=Path, default=DEFAULT_VIS_SCRIPT)
    parser.add_argument("--converter", type=Path, default=DEFAULT_CONVERTER)
    parser.add_argument("--scene-xml-file", type=Path, default=DEFAULT_SCENE_XML)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--robot", default="g1")
    parser.add_argument("--output-fps", type=int, default=50)
    parser.add_argument("--order-mode", default="amass_csv")
    parser.add_argument("--wrist-policy", default="mapped")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of broken clips to report/repair.")
    parser.add_argument("--repair", action="store_true", help="Regenerate broken clips in-place.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    amass_root = args.amass_root.resolve()
    cache_root = args.cache_root.resolve()

    if not output_root.exists():
        print(f"[ERROR] output root not found: {output_root}", file=sys.stderr)
        return 2

    broken = scan_broken_clips(
        output_root=output_root,
        amass_root=amass_root,
        cache_root=cache_root,
        output_fps=args.output_fps,
    )
    if args.limit > 0:
        broken = broken[: args.limit]

    write_manifest(broken, args.manifest)
    print(f"[INFO] wrote manifest: {args.manifest}")
    print(f"[INFO] broken clips: {len(broken)}")

    if not broken:
        return 0

    missing_source = sum(1 for clip in broken if not clip.source_path.exists())
    missing_proxy = sum(1 for clip in broken if not clip.proxy_path.exists())
    print(f"[INFO] missing source clips: {missing_source}")
    print(f"[INFO] missing proxy clips : {missing_proxy}")

    if not args.repair:
        for clip in broken[:10]:
            print(f"[BAD] {clip.rel_path.as_posix()}")
        return 0

    repaired, failed = repair_clips(
        clips=broken,
        amass_root=amass_root,
        cache_root=cache_root,
        vis_script=args.vis_script.resolve(),
        converter=args.converter.resolve(),
        scene_xml_file=args.scene_xml_file.resolve(),
        python_bin=args.python_bin,
        robot=args.robot,
        output_fps=args.output_fps,
        order_mode=args.order_mode,
        wrist_policy=args.wrist_policy,
    )
    print(f"[INFO] repaired: {repaired}")
    print(f"[INFO] failed  : {failed}")
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
