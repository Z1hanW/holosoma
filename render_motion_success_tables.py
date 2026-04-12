#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def _table_lines(rows: list[dict], title: str) -> list[str]:
    lines = [
        f"# {title}",
        "",
        "| clip_idx | clip_name | success | done_step | timed_out | incomplete | env_id |",
        "| ---: | --- | --- | ---: | --- | --- | ---: |",
    ]
    for row in rows:
        done_step = "" if row.get("done_step") is None else str(int(row["done_step"]))
        lines.append(
            "| {clip_idx} | `{clip_name}` | {success} | {done_step} | {timed_out} | {incomplete} | {env_id} |".format(
                clip_idx=int(row["clip_idx"]),
                clip_name=str(row["clip_name"]),
                success="true" if row.get("success") else "false",
                done_step=done_step,
                timed_out="true" if row.get("timed_out") else "false",
                incomplete="true" if row.get("incomplete") else "false",
                env_id=int(row["env_id"]),
            )
        )
    lines.append("")
    return lines


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: render_motion_success_tables.py <eval_output_dir>")

    out_dir = Path(sys.argv[1]).expanduser().resolve()
    summary_path = out_dir / "summary.json"
    per_motion_path = out_dir / "per_motion_results.json"
    if not summary_path.exists() or not per_motion_path.exists():
        raise SystemExit(f"missing summary.json or per_motion_results.json under {out_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = json.loads(per_motion_path.read_text(encoding="utf-8"))
    rows = sorted(rows, key=lambda row: int(row["clip_idx"]))
    failed_rows = [row for row in rows if not row.get("success")]
    success_rows = [row for row in rows if row.get("success")]

    summary_md_lines = [
        "# Motion Success Summary",
        "",
        "| checkpoint | num_envs | batch_size | max_steps | num_clips | num_success | success_rate | num_incomplete |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| `{summary['checkpoint']}` | {int(summary['num_envs'])} | {int(summary['batch_size'])} | "
            f"{int(summary['max_steps'])} | {int(summary['num_clips'])} | {int(summary['num_success'])} | "
            f"{float(summary['success_rate']):.4f} | {int(summary['num_incomplete'])} |"
        ),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_md_lines), encoding="utf-8")
    (out_dir / "per_motion_results.md").write_text(
        "\n".join(_table_lines(rows, "Per-Motion Results")),
        encoding="utf-8",
    )
    (out_dir / "failed_motions.md").write_text(
        "\n".join(_table_lines(failed_rows, "Failed Motions")),
        encoding="utf-8",
    )
    (out_dir / "successful_motions.md").write_text(
        "\n".join(_table_lines(success_rows, "Successful Motions")),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
