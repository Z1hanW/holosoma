"""Compare input-provenance digests across every torchrun rank before simulation."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import torch.distributed as dist

from holosoma.utils.atomic_output import emit_atomic_stdout_record
from holosoma.utils.training_provenance import parse_training_provenance


def scientific_provenance_contract(provenance: dict) -> dict:
    """Return fields that must be identical for scientific training semantics.

    Kernel/OS build strings remain recorded in every rank's provenance for
    diagnostics, but heterogeneous cloud nodes need not run the same kernel
    patch build.  Numerical/runtime package versions, source, data and all
    checkpoint digests remain strict.
    """

    contract = dict(provenance)
    environment = contract.get("environment")
    if isinstance(environment, dict):
        semantic_environment = dict(environment)
        semantic_environment.pop("platform", None)
        contract["environment"] = semantic_environment
    return contract


def _mismatch_summary(reference: dict, candidate: dict) -> list[str]:
    paths: list[str] = []

    def visit(left: object, right: object, prefix: str) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left).union(right)):
                visit(left.get(key), right.get(key), f"{prefix}.{key}" if prefix else str(key))
        elif left != right:
            paths.append(f"{prefix}: reference={left!r} candidate={right!r}")

    visit(reference, candidate, "")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--master-port", required=True, type=int)
    args = parser.parse_args()
    if args.world_size < 1:
        raise SystemExit("[ERROR] provenance world size must be positive")
    if not 1 <= args.master_port <= 65535:
        raise SystemExit("[ERROR] provenance master port must be in [1, 65535]")

    provenance = parse_training_provenance(sys.stdin.read())
    if provenance is None:
        raise SystemExit("[ERROR] missing training provenance payload")
    rank = int(os.environ.get("RANK", "0"))
    actual_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if actual_world_size != args.world_size:
        raise SystemExit(
            f"[ERROR] provenance WORLD_SIZE={actual_world_size} does not match requested {args.world_size}"
        )

    if args.world_size > 1:
        # ``torchrun`` exports TORCHELASTIC_USE_AGENT_STORE=True for its child
        # workers.  This preflight deliberately uses a separate port so it
        # must create a separate TCPStore; inheriting the elastic flag makes
        # every rank (including rank 0) a client and deadlocks until timeout.
        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
        os.environ["MASTER_PORT"] = str(args.master_port)
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            rank=rank,
            world_size=args.world_size,
            timeout=datetime.timedelta(seconds=int(os.environ.get("HOLOSOMA_PROVENANCE_TIMEOUT_SEC", "300"))),
        )
        gathered: list[dict | None] = [None] * args.world_size
        dist.all_gather_object(gathered, provenance)
        local_contract = scientific_provenance_contract(provenance)
        contracts = [scientific_provenance_contract(value) if isinstance(value, dict) else value for value in gathered]
        mismatches = [index for index, value in enumerate(contracts) if value != local_contract]
        if mismatches:
            details = {
                str(index): _mismatch_summary(local_contract, contracts[index])
                for index in mismatches
                if isinstance(contracts[index], dict)
            }
            raise RuntimeError(
                "Cross-rank training provenance mismatch; refusing to start simulation. "
                f"local_rank={rank} mismatching_ranks={mismatches} "
                f"semantic_differences={json.dumps(details, sort_keys=True)}"
            )
        platforms = {
            str(index): value.get("environment", {}).get("platform")
            for index, value in enumerate(gathered)
            if isinstance(value, dict) and isinstance(value.get("environment"), dict)
        }
        if rank == 0 and len(set(platforms.values())) > 1:
            print(
                "[WARN] cross_rank_platform_builds_differ "
                f"platform_by_rank={json.dumps(platforms, sort_keys=True)}",
                flush=True,
            )
        dist.barrier()
        dist.destroy_process_group()

    emit_atomic_stdout_record(
        "[INFO] cross_rank_training_provenance_verified "
        f"world_size={args.world_size} training_regime={provenance['training_regime']} "
        f"teacher_enabled={provenance['teacher_enabled']} "
        f"teacher_sha256={provenance['teacher_sha256']} "
        f"motion_shard_manifest_sha256={provenance['motion_shard_manifest_sha256']} "
        f"contact_sidecar_manifest_sha256={provenance['contact_sidecar_manifest_sha256']} "
        f"source_bundle_sha256={provenance['source_bundle_sha256']}"
    )


if __name__ == "__main__":
    main()
