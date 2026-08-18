#!/usr/bin/env python3
"""Derive an honest deterministic-eval perception contract from an ONNX artifact."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import onnx


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} MODEL.onnx")

    model = onnx.load(sys.argv[1], load_external_data=False)
    metadata: dict[str, object] = {}
    for prop in model.metadata_props:
        if prop.key in metadata:
            raise SystemExit(f"Duplicate ONNX metadata key: {prop.key}")
        try:
            metadata[prop.key] = json.loads(prop.value)
        except json.JSONDecodeError:
            metadata[prop.key] = prop.value

    contract = metadata.get("perception_observation_contract")
    declared_digest = metadata.get("perception_observation_contract_sha256")
    if not isinstance(contract, dict) or contract.get("version") != 2:
        raise SystemExit("ONNX artifact lacks a version-2 perception observation contract")
    if not isinstance(declared_digest, str) or len(declared_digest) != 64:
        raise SystemExit("ONNX artifact lacks a valid perception observation contract digest")
    original_digest = hashlib.sha256(_canonical_bytes(contract)).hexdigest()
    if original_digest != declared_digest.lower():
        raise SystemExit("ONNX perception observation contract digest does not match its payload")
    if contract.get("camera_reset_randomization") is None:
        raise SystemExit("ONNX perception contract already has reset randomization disabled")
    setup_randomization = contract.get("camera_setup_randomization")
    if setup_randomization is not None and (
        not isinstance(setup_randomization, dict)
        or setup_randomization.get("enabled") is not False
    ):
        raise SystemExit("Cannot derive deterministic eval contract while camera setup jitter is enabled")

    eval_contract = deepcopy(contract)
    eval_contract["camera_reset_randomization"] = None
    eval_digest = hashlib.sha256(_canonical_bytes(eval_contract)).hexdigest()
    envelope = base64.b64encode(
        _canonical_bytes({"contract": eval_contract, "sha256": eval_digest})
    ).decode("ascii")

    print(f"PERCEPTION_RANDOMIZATION_ENABLED=False")
    print("PERCEPTION_RANDOMIZATION_CONTRACT_STATUS=eval-reset-randomization-disabled")
    print(f"PERCEPTION_CONTRACT_ENVELOPE_B64={envelope}")
    print(f"HOLOSOMA_EVAL_PERCEPTION_CONTRACT_ORIGINAL_SHA256={original_digest}")
    print(f"HOLOSOMA_EVAL_PERCEPTION_CONTRACT_SHA256_OVERRIDE={eval_digest}")


if __name__ == "__main__":
    main()
