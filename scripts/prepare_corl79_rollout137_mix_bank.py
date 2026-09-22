#!/usr/bin/env python3
"""Publish the exact CORL79 + ch2/40K rollout137 union without changing trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

import contact_sampling_factorial as recipe
from build_merged_training_bank import SourceSpec, build_bank, verify_bank
from prepare_as_rank_shards import compute_rank_shard_source_digest, prepare_rank_shards

CORL_PARENT = recipe.ROOT / "data/ds_as_data/carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success155_bcleb5oi58000_final0p5_primitiveproj_solid80_clean_box_bin_barrel_ball_cominertia_categorymass_v2"
CORL = CORL_PARENT / "_scientific_corl79_single_slot/by-source/c9e02244ac1e3c870564f70837a963b03a337430bb1b4a58dc50610868df8027"
RAW137 = Path("/data/holosoma_inputs/ch2ckwzw_model40000_rollout137_20260908/by-source/0bab4cdc7a2ff469dfe7855d817caef84ef26b834a00a9b73eb05db05e20bf47")
COUNTS = {"box": 60, "ball": 38, "barrel": 69, "bin": 49}
CONTACT_NAME = "contact_export_corl79_ch2_40k_rollout137"


def prepare(output):
    if output.exists():
        raise ValueError(f"Refusing to replace identity: {output}")
    contact_manifest = Path(recipe.CONTACT) / "manifest.json"
    if recipe.sha(contact_manifest) != recipe.CONTACT_MANIFEST_SHA:
        raise ValueError("Contact manifest changed")
    contact = json.loads(contact_manifest.read_text())
    if (recipe.sha(RAW137 / "manifest.json") != contact["raw_manifest_sha256"]
            or contact["producer_checkpoint_sha256"] != recipe.TEACHER_SHA):
        raise ValueError("Rollout/contact producer mismatch")
    for row in contact["records"]:
        if recipe.sha(Path(recipe.CONTACT) / row["path"]) != row["sha256"]:
            raise ValueError(f"Changed contact payload: {row['path']}")
    old109 = Path("/data/holosoma_inputs/corl79_plus_debug30_realmesh_categorymass_v1/by-source/aa4dcb12bc14df37446417d98d7179236960d2c715975d0753438d164ceafa5c/manifest.json")
    old_source = next(s for s in json.loads(old109.read_text())["merge_source_identity"]["sources"] if s["label"] == "corl79")
    if recipe.sha(CORL / "manifest.json") != old_source["source_manifest"]["sha256"]:
        raise ValueError("CORL source is not the prior actual79 generation")
    for clip in old_source["clips"]:
        if recipe.sha(CORL / clip["motion"]["path"]) != clip["motion"]["sha256"]:
            raise ValueError(f"Changed CORL motion: {clip['clip_id']}")
    sources = [SourceSpec("corl79", CORL, CORL_PARENT / "contact_export_from_teacher_success133_final0p5"),
               SourceSpec("ch2_40k_rollout137", RAW137, Path(recipe.CONTACT), "source_motion")]
    raw, raw_sha = build_bank(sources, output_base=Path("/data/holosoma_inputs/corl79_ch2_40k_rollout137_realmesh_v1"),
                             contact_export_name=CONTACT_NAME, expected_total=216)
    verify_bank(raw, expected_digest=raw.name, expected_manifest_sha256=raw_sha)
    base = Path("/data/holosoma_inputs/corl79_ch2_40k_rollout137_precomputed_turn_forward_v1/by-source")
    draft = base / "prepared_generation"
    recipe.run([sys.executable, recipe.ROOT / "scripts/build_decoupled_root_command_bank.py",
                "--source", raw, "--output", draft, "--expected-clip-count", "216",
                "--expected-category-counts-json", json.dumps(COUNTS), "--copy-portable-source-tree",
                "--expected-source-payload-digest", raw.name, "--expected-source-manifest-sha256", raw_sha], timeout=1800)
    manifest = json.loads((draft / "manifest.json").read_text())
    bank = base / manifest["derived_payload_digest"]
    if bank.exists():
        raise ValueError(f"Generation already exists: {bank}")
    draft.rename(bank)
    # New commands must exactly preserve the current137 command semantics as well as all trajectories.
    unchanged = 0
    no_lift = []
    for clip in manifest["clips"]:
        name = clip["clip_id"] + ".npz"
        original = Path(recipe.BANK) / name if clip["clip_id"].startswith("prism_") else CORL / name
        with np.load(original, allow_pickle=False) as a, np.load(bank / name, allow_pickle=False) as b:
            for key in a.files:
                if not np.array_equal(a[key], b[key]):
                    raise ValueError(f"Motion/command changed: {name}:{key}")
            unchanged += 1
            height_range = float(np.ptp(b["object_pos_w"][:, 2]))
            if height_range < .01:
                no_lift.append({"clip": clip["clip_id"], "object_world_z_range_m": height_range})
    if unchanged != 216 or manifest["category_counts"] != COUNTS:
        raise ValueError("Merged coverage/category mismatch")
    digest = compute_rank_shard_source_digest(motion_dir=bank, object_map=bank / "_clip_object_urdf_map.json",
                                               world_size=64, environments_per_rank=2048)
    shards = bank / "_rank_shards/by-source" / digest / "ws64"
    bank.chmod(0o755)
    try:
        shard = prepare_rank_shards(motion_dir=bank, object_map=bank / "_clip_object_urdf_map.json", output_root=shards,
                                    world_size=64, environments_per_rank=2048, expected_source_digest=digest)
    finally:
        bank.chmod(0o555)
    if shard["clip_count"] != 216 or not shard["exact_clip_partition"] or set(shard["clip_cover_counts"].values()) != {1}:
        raise ValueError("Rank shards do not cover every clip exactly once")
    identity = {"bank": str(bank), "contact_root": str(bank / CONTACT_NAME), "clip_count": 216,
                "category_counts": COUNTS, "raw_bank": str(raw), "raw_manifest_sha256": raw_sha,
                "bank_manifest_sha256": recipe.sha(bank / "manifest.json"),
                "object_map_sha256": recipe.sha(bank / "_clip_object_urdf_map.json"),
                "single_slot_source_digest": manifest["source_view_digest"], "single_slot_view_digest": bank.name,
                "shard_digest": digest, "shard_root": str(shards), "shard_manifest_sha256": recipe.sha(shards / "manifest.json"),
                "rank_clip_counts": [s["clip_count"] for s in shard["shards"]],
                "command_mode": "precomputed_turn_then_forward", "unchanged_source_clips": unchanged,
                "source_counts": {"corl79": 79, "ch2_40k_rollout137": 137}, "no_lift_clips_retained": no_lift,
                "online_teacher_sha256": recipe.TEACHER_SHA,
                "source137_contact_manifest_sha256": recipe.CONTACT_MANIFEST_SHA,
                "source79_manifest_sha256": old_source["source_manifest"]["sha256"]}
    recipe.save(output, identity)
    print(json.dumps(identity), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
