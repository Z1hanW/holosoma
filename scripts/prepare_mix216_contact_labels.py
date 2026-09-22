#!/usr/bin/env python3
"""Convert authenticated legacy CORL interval NPY to JSON, without new labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

import contact_sampling_factorial as recipe
from build_merged_training_bank import discover_contact_dirs, freeze_tree, published_file_records


def interval_json(directory):
    if (directory / "contact_intervals.json").exists():
        raise ValueError("Legacy conversion requires an NPY-only source, not an existing JSON")
    result = {}
    for region in ("left_wrist", "right_wrist"):
        value = np.load(directory / (region + "_contact_interval_steps.npy"), allow_pickle=False)
        if value.shape != (2,) or value.dtype.kind not in "iu":
            raise ValueError(f"Invalid legacy interval: {directory}:{region}")
        start, end = map(int, value)
        if [start, end] != [-1, -1] and not (0 <= start < end):
            raise ValueError(f"Invalid legacy interval bounds: {directory}:{region}")
        result[region] = [start, end]
    return result


def prepare(source_identity, output):
    if output.exists():
        raise ValueError("Refusing to replace dataset identity")
    data = json.loads(source_identity.read_text())
    source = Path(data["contact_root"])
    dirs = discover_contact_dirs(source)
    if len(dirs) != 216:
        raise ValueError("Expected exact 216 contact directories")
    records = published_file_records(source)
    identity = {"schema": "mixed216_contact_json_from_legacy_npy_v1", "source_records": records,
                "source_bank_manifest_sha256": data["bank_manifest_sha256"], "clip_count": 216}
    digest = recipe.json_sha(identity)
    base = Path("/data/holosoma_inputs/corl79_ch2_40k_rollout137_contact_runtime_v1/by-source")
    base.mkdir(parents=True, exist_ok=True)
    target = base / digest
    if target.exists():
        raise ValueError(f"Contact generation already exists: {target}")
    draft = Path(tempfile.mkdtemp(prefix=".contact216-", dir=base))
    conversions = []
    for clip, directory in sorted(dirs.items()):
        dest = draft / "clips" / clip
        shutil.copytree(directory, dest)
        if clip.startswith("prism_"):
            if not (dest / "contact_intervals.json").is_file():
                raise ValueError(f"Missing authenticated modern contact JSON: {clip}")
        else:
            values = interval_json(directory)
            dest.chmod(0o755)
            recipe.save(dest / "contact_intervals.json", values)
            conversions.append(clip)
            for region, pair in json.loads((dest / "contact_intervals.json").read_text()).items():
                np.testing.assert_array_equal(pair, np.load(directory / (region + "_contact_interval_steps.npy"), allow_pickle=False))
        for original in directory.rglob("*"):
            if original.is_file() and recipe.sha(original) != recipe.sha(dest / original.relative_to(directory)):
                raise ValueError(f"Original sidecar changed: {original}")
    if len(conversions) != 79:
        raise ValueError("Legacy conversion did not cover exactly CORL79")
    manifest = {**identity, "payload_digest": digest, "converted_clips": conversions,
                "interval_values_unchanged": True, "records": published_file_records(draft)}
    recipe.save(draft / "manifest.json", manifest)
    freeze_tree(draft)
    draft.rename(target)
    data.update({"contact_root": str(target), "contact_manifest_sha256": recipe.sha(target / "manifest.json"),
                 "legacy_interval_json_conversion_count": 79, "legacy_interval_values_unchanged": True})
    recipe.save(output, data)
    print(json.dumps({"contact_root": str(target), "converted": 79, "interval_values_unchanged": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_identity, args.output)
