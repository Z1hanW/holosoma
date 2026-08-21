#!/usr/bin/env python3
"""Build and launch one viewer for two completed PRISM height pipelines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_continuation(path: Path):
    spec = importlib.util.spec_from_file_location("prism_height_continuation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import continuation script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_complete_pipeline(root: Path) -> dict:
    state_path = root / "pipeline_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"pipeline state is missing: {state_path}")
    state = read_json(state_path)
    if state.get("status") != "complete":
        raise RuntimeError(f"pipeline is not complete: {root}: {state}")
    return state


def link_directory(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=True)


def combine_viewers(
    source_roots: list[Path],
    output_root: Path,
    *,
    expected_count: int,
) -> list[dict]:
    viewer_root = output_root / "viewer"
    retarget_root = viewer_root / "retarget"
    metadata_root = viewer_root / "metadata"
    (metadata_root / "before").mkdir(parents=True)

    rows = []
    seen = set()
    for source_root in source_roots:
        manifest_path = source_root / "viewer" / "adapter_manifest.json"
        source_rows = read_json(manifest_path)
        for row in source_rows:
            sequence = row["sequence"]
            if sequence in seen:
                raise ValueError(f"duplicate sequence across viewers: {sequence}")
            seen.add(sequence)
            link_directory(
                source_root / "viewer" / "retarget" / sequence,
                retarget_root / sequence,
            )
            link_directory(
                source_root / "viewer" / "metadata" / "box_mesh" / sequence,
                metadata_root / "box_mesh" / sequence,
            )
            rows.append(
                {
                    **row,
                    "source_viewer_root": str(source_root / "viewer"),
                }
            )

    if len(rows) != expected_count:
        raise RuntimeError(
            f"combined viewer has {len(rows)}/{expected_count} sequences"
        )
    rows.sort(key=lambda row: row["sequence"])
    atomic_json(viewer_root / "adapter_manifest.json", rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-pipeline-root", type=Path, required=True)
    parser.add_argument("--secondary-pipeline-root", type=Path, required=True)
    parser.add_argument("--secondary-staging-root", type=Path, required=True)
    parser.add_argument("--continuation-script", type=Path, required=True)
    parser.add_argument("--object-repo", type=Path, required=True)
    parser.add_argument("--trajopt-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=135)
    parser.add_argument("--secondary-expected-count", type=int, default=67)
    parser.add_argument("--viewer-port", type=int, default=9304)
    parser.add_argument("--viewer-sequence", default="prism_cf_bin_m0_v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())

    primary_state = require_complete_pipeline(args.primary_pipeline_root)
    secondary_state = require_complete_pipeline(args.secondary_pipeline_root)
    args.output_root.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "running",
        "started_at": utc_now(),
        "primary_pipeline_root": str(args.primary_pipeline_root),
        "secondary_pipeline_root": str(args.secondary_pipeline_root),
        "primary_pipeline_state": primary_state,
        "secondary_pipeline_state": secondary_state,
    }
    atomic_json(args.output_root / "combined_state.json", state)
    try:
        continuation = load_continuation(args.continuation_script)
        secondary_summary = read_json(
            args.secondary_pipeline_root / "trajopt" / "summary.json"
        )
        secondary_viewer_args = argparse.Namespace(
            output_root=args.secondary_pipeline_root,
            staging_root=args.secondary_staging_root,
            expected_count=args.secondary_expected_count,
        )
        secondary_viewer = continuation.build_viewer(
            secondary_viewer_args,
            secondary_summary,
        )

        rows = combine_viewers(
            [args.primary_pipeline_root, args.secondary_pipeline_root],
            args.output_root,
            expected_count=args.expected_count,
        )
        viewer_args = argparse.Namespace(
            output_root=args.output_root,
            baseline_root=args.primary_pipeline_root,
            trajopt_repo=args.trajopt_repo,
            object_repo=args.object_repo,
            model=args.model,
            viewer_port=args.viewer_port,
            viewer_sequence=args.viewer_sequence,
        )
        viewer = continuation.start_viewer(viewer_args)
        state.update(
            {
                "status": "complete",
                "completed_at": utc_now(),
                "sequence_count": len(rows),
                "secondary_viewer": secondary_viewer,
                "viewer": viewer,
            }
        )
        atomic_json(args.output_root / "combined_state.json", state)
        return 0
    except Exception as exc:
        state.update(
            {
                "status": "failed",
                "failed_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_json(args.output_root / "combined_state.json", state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
