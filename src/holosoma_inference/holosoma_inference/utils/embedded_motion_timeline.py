"""Digest-bound provenance for motion tensors embedded in patched WBT ONNX files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

from holosoma_inference.utils.policy_contract import (
    PolicyContractError,
    effective_motion_transition_settings_from_metadata,
)


EMBEDDED_MOTION_TIMELINE_CONTRACT_KEY = "embedded_motion_timeline_contract"
EMBEDDED_MOTION_TIMELINE_CONTRACT_SHA256_KEY = (
    "embedded_motion_timeline_contract_sha256"
)
EMBEDDED_MOTION_TIMELINE_CONTRACT_VERSION = 1
EMBEDDED_MOTION_TENSOR_NAMES = (
    "joint_pos",
    "joint_vel",
    "ref_pos_xyz",
    "ref_quat_xyzw",
)

_MAX_TRANSITION_STEPS = 4096
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def read_stable_regular_file_bytes(path: Path, *, label: str) -> bytes:
    """Read one immutable-by-descriptor regular-file payload."""

    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file: {path}")
        payload = stream.read()
        after = os.fstat(stream.fileno())
    if not payload:
        raise ValueError(f"{label} is empty: {path}")
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or len(payload) != before.st_size
    ):
        raise RuntimeError(f"{label} changed while it was being read: {path}")
    return payload


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PolicyContractError(
            "Embedded motion timeline contract must contain strict finite JSON values."
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def embedded_motion_tensors_sha256(tensors: Mapping[str, Any]) -> tuple[str, int]:
    """Hash the exact float32 tensors patched behind the four motion Gather nodes."""

    if set(tensors) != set(EMBEDDED_MOTION_TENSOR_NAMES):
        raise PolicyContractError(
            "Embedded motion tensor set must contain exactly "
            f"{list(EMBEDDED_MOTION_TENSOR_NAMES)}."
        )

    arrays: dict[str, np.ndarray] = {}
    frame_count: int | None = None
    for name in EMBEDDED_MOTION_TENSOR_NAMES:
        raw = np.asarray(tensors[name])
        if raw.dtype != np.dtype(np.float32):
            raise PolicyContractError(
                f"Embedded motion tensor {name!r} must have dtype float32, got {raw.dtype}."
            )
        array = np.ascontiguousarray(raw.astype("<f4", copy=False))
        if array.ndim != 2:
            raise PolicyContractError(
                f"Embedded motion tensor {name!r} must be rank two, got shape={array.shape}."
            )
        expected_width = 3 if name == "ref_pos_xyz" else 4 if name == "ref_quat_xyzw" else None
        if expected_width is not None and array.shape[1] != expected_width:
            raise PolicyContractError(
                f"Embedded motion tensor {name!r} must have width {expected_width}, "
                f"got shape={array.shape}."
            )
        if name == "joint_vel" and "joint_pos" in arrays and array.shape != arrays["joint_pos"].shape:
            raise PolicyContractError(
                "Embedded joint_pos and joint_vel tensors must have identical shapes."
            )
        if array.shape[0] <= 0:
            raise PolicyContractError(f"Embedded motion tensor {name!r} must not be empty.")
        if frame_count is None:
            frame_count = int(array.shape[0])
        elif array.shape[0] != frame_count:
            raise PolicyContractError(
                "All embedded motion tensors must have the same frame count."
            )
        if not np.isfinite(array).all():
            raise PolicyContractError(
                f"Embedded motion tensor {name!r} contains non-finite values."
            )
        arrays[name] = array

    digest = hashlib.sha256()
    for name in EMBEDDED_MOTION_TENSOR_NAMES:
        array = arrays[name]
        header = json.dumps(
            {"dtype": "float32-le", "name": name, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, byteorder="big", signed=False))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    assert frame_count is not None
    return digest.hexdigest(), frame_count


def embedded_motion_tensors_from_model(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    """Extract the exact Constant tensors indexed by the patched motion Gather nodes."""

    nodes_by_output: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for output in node.output:
            nodes_by_output.setdefault(output, []).append(node)

    result: dict[str, np.ndarray] = {}
    for name in EMBEDDED_MOTION_TENSOR_NAMES:
        gather_matches = [
            node for node in nodes_by_output.get(name, ()) if node.op_type == "Gather"
        ]
        if len(gather_matches) != 1:
            raise PolicyContractError(
                f"Patched ONNX must contain exactly one Gather producing {name!r}; "
                f"found={len(gather_matches)}."
            )
        gather = gather_matches[0]
        if not gather.input:
            raise PolicyContractError(f"Motion Gather {name!r} has no data input.")
        constant_matches = [
            node
            for node in nodes_by_output.get(gather.input[0], ())
            if node.op_type == "Constant"
        ]
        if len(constant_matches) != 1:
            raise PolicyContractError(
                f"Motion Gather {name!r} must be backed by exactly one Constant; "
                f"found={len(constant_matches)}."
            )
        value_attributes = [
            attribute for attribute in constant_matches[0].attribute if attribute.name == "value"
        ]
        if len(value_attributes) != 1 or not value_attributes[0].HasField("t"):
            raise PolicyContractError(
                f"Motion Constant backing {name!r} must contain exactly one tensor value."
            )
        result[name] = np.asarray(numpy_helper.to_array(value_attributes[0].t))
    return result


def _constant_tensor_from_output(
    nodes_by_output: Mapping[str, list[onnx.NodeProto]],
    output_name: str,
    *,
    label: str,
) -> np.ndarray:
    matches = [
        node for node in nodes_by_output.get(output_name, ()) if node.op_type == "Constant"
    ]
    if len(matches) != 1:
        raise PolicyContractError(
            f"{label} must be backed by exactly one Constant; found={len(matches)}."
        )
    attributes = [attribute for attribute in matches[0].attribute if attribute.name == "value"]
    if len(attributes) != 1 or not attributes[0].HasField("t"):
        raise PolicyContractError(f"{label} Constant must contain exactly one tensor value.")
    return np.asarray(numpy_helper.to_array(attributes[0].t))


def _validate_embedded_motion_indexing(
    model: onnx.ModelProto,
    *,
    embedded_frame_count: int,
) -> None:
    """Ensure runtime indexing exposes the full authenticated tensor timeline."""

    nodes_by_output: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for output in node.output:
            nodes_by_output.setdefault(output, []).append(node)

    index_inputs: set[str] = set()
    for name in EMBEDDED_MOTION_TENSOR_NAMES:
        gathers = [
            node for node in nodes_by_output.get(name, ()) if node.op_type == "Gather"
        ]
        if len(gathers) != 1 or len(gathers[0].input) != 2:
            raise PolicyContractError(
                f"Embedded motion output {name!r} must have exactly one two-input Gather."
            )
        axis_attributes = [
            attribute for attribute in gathers[0].attribute if attribute.name == "axis"
        ]
        if len(axis_attributes) > 1 or (
            axis_attributes and int(axis_attributes[0].i) != 0
        ):
            raise PolicyContractError(
                f"Embedded motion Gather {name!r} must index tensor axis zero."
            )
        index_inputs.add(gathers[0].input[1])
    if len(index_inputs) != 1:
        raise PolicyContractError(
            "All embedded motion Gather nodes must share one authenticated timeline index."
        )

    index_output = next(iter(index_inputs))
    clips = [node for node in nodes_by_output.get(index_output, ()) if node.op_type == "Clip"]
    if len(clips) != 1 or len(clips[0].input) != 3:
        raise PolicyContractError(
            "Embedded motion timeline index must be produced by exactly one three-input Clip."
        )
    clip = clips[0]
    if not clip.input[0]:
        raise PolicyContractError("Embedded motion Clip data input must not be empty.")
    if not clip.input[1] or not clip.input[2]:
        raise PolicyContractError(
            "Embedded motion Clip must use explicit minimum and maximum inputs."
        )
    minimum = _constant_tensor_from_output(
        nodes_by_output,
        clip.input[1],
        label="Embedded motion Clip minimum",
    )
    maximum = _constant_tensor_from_output(
        nodes_by_output,
        clip.input[2],
        label="Embedded motion Clip maximum",
    )
    for value, expected, label in (
        (minimum, 0, "minimum"),
        (maximum, embedded_frame_count - 1, "maximum"),
    ):
        if value.size != 1 or value.dtype.kind not in {"i", "u"}:
            raise PolicyContractError(
                f"Embedded motion Clip {label} must be one integer scalar tensor."
            )
        if int(value.reshape(-1)[0]) != expected:
            raise PolicyContractError(
                f"Embedded motion Clip {label} contradicts authenticated frame count: "
                f"actual={int(value.reshape(-1)[0])}, expected={expected}."
            )

    _validate_embedded_motion_time_step_lineage(
        model,
        nodes_by_output=nodes_by_output,
        clip_data_input=clip.input[0],
    )


def _validate_embedded_motion_time_step_lineage(
    model: onnx.ModelProto,
    *,
    nodes_by_output: Mapping[str, list[onnx.NodeProto]],
    clip_data_input: str,
) -> None:
    """Require the Clip index to be a shape-only canonicalization of ``time_step``.

    Exported combined WBT graphs use ``time_step -> Cast(INT64) -> Squeeze``
    while small fixtures and some older exporters expose an INT64 ``time_step``
    directly.  Anything that changes values (including a Constant, initializer,
    arithmetic node, or a different graph input) can silently pin or permute the
    authenticated tensor timeline and must fail closed.
    """

    time_step_inputs = [item for item in model.graph.input if item.name == "time_step"]
    if len(time_step_inputs) != 1:
        raise PolicyContractError(
            "Embedded motion timeline requires exactly one graph input named 'time_step'."
        )
    if any(initializer.name == "time_step" for initializer in model.graph.initializer):
        raise PolicyContractError(
            "Embedded motion graph input 'time_step' must not also be an initializer."
        )
    if nodes_by_output.get("time_step"):
        raise PolicyContractError(
            "Embedded motion graph input 'time_step' must not also be produced by a node."
        )

    tensor_type = time_step_inputs[0].type.tensor_type
    source_element_type = int(tensor_type.elem_type)
    allowed_source_types = {onnx.TensorProto.FLOAT, onnx.TensorProto.INT64}
    if source_element_type not in allowed_source_types:
        raise PolicyContractError(
            "Embedded motion graph input 'time_step' must have FLOAT or INT64 tensor type."
        )

    current = clip_data_input
    visited: set[str] = set()
    cast_count = 0
    while current != "time_step":
        if not current or current in visited:
            raise PolicyContractError(
                "Embedded motion Clip data lineage is empty or cyclic before 'time_step'."
            )
        visited.add(current)
        producers = list(nodes_by_output.get(current, ()))
        if len(producers) != 1:
            raise PolicyContractError(
                "Every embedded motion Clip data value must have exactly one producer on its "
                f"lineage to 'time_step'; value={current!r}, producers={len(producers)}."
            )
        node = producers[0]
        if node.domain not in {"", "ai.onnx"}:
            raise PolicyContractError(
                "Embedded motion Clip data lineage may use only canonical ai.onnx operators."
            )
        if node.op_type == "Identity":
            if len(node.input) != 1 or list(node.attribute):
                raise PolicyContractError(
                    "Embedded motion time_step Identity must have one input and no attributes."
                )
        elif node.op_type == "Cast":
            cast_count += 1
            to_attributes = [attribute for attribute in node.attribute if attribute.name == "to"]
            if (
                len(node.input) != 1
                or len(to_attributes) != 1
                or len(node.attribute) != 1
                or int(to_attributes[0].i) != onnx.TensorProto.INT64
            ):
                raise PolicyContractError(
                    "Embedded motion time_step Cast must have one input and cast exactly to INT64."
                )
        elif node.op_type == "Squeeze":
            if len(node.input) not in {1, 2}:
                raise PolicyContractError(
                    "Embedded motion time_step Squeeze must have one data input and at most one axes input."
                )
            axes_attributes = [
                attribute for attribute in node.attribute if attribute.name == "axes"
            ]
            unexpected_attributes = [
                attribute.name for attribute in node.attribute if attribute.name != "axes"
            ]
            if unexpected_attributes or len(axes_attributes) > 1:
                raise PolicyContractError(
                    "Embedded motion time_step Squeeze contains non-canonical attributes."
                )
            if len(node.input) == 2:
                if not node.input[1] or axes_attributes:
                    raise PolicyContractError(
                        "Embedded motion time_step Squeeze axes must have one unambiguous source."
                    )
                axes = _constant_tensor_from_output(
                    nodes_by_output,
                    node.input[1],
                    label="Embedded motion time_step Squeeze axes",
                )
                if axes.ndim != 1 or axes.dtype.kind not in {"i", "u"}:
                    raise PolicyContractError(
                        "Embedded motion time_step Squeeze axes must be a rank-one integer Constant."
                    )
            elif axes_attributes:
                axes = np.asarray(axes_attributes[0].ints, dtype=np.int64)
                if axes.ndim != 1:
                    raise PolicyContractError(
                        "Embedded motion time_step Squeeze axes attribute must be rank one."
                    )
        else:
            raise PolicyContractError(
                "Embedded motion Clip data must derive from 'time_step' only through Identity, "
                f"Cast(INT64), and Squeeze; found {node.op_type!r}."
            )
        current = node.input[0]

    if cast_count > 1:
        raise PolicyContractError(
            "Embedded motion time_step lineage may contain at most one Cast(INT64)."
        )
    if source_element_type == onnx.TensorProto.FLOAT and cast_count != 1:
        raise PolicyContractError(
            "FLOAT time_step must pass through exactly one Cast(INT64) before motion indexing."
        )
    if source_element_type == onnx.TensorProto.INT64 and cast_count != 0:
        raise PolicyContractError(
            "INT64 time_step must not be redundantly recast before motion indexing."
        )


def build_embedded_motion_timeline_contract(
    *,
    source_motion_sha256: str,
    source_frame_count: int,
    embedded_tensors_sha256: str,
    embedded_frame_count: int,
    motion_transition_contract_sha256: str | None,
    effective_prepend_steps: int,
    effective_append_steps: int,
    materialization: str,
) -> tuple[dict[str, Any], str]:
    contract = {
        "version": EMBEDDED_MOTION_TIMELINE_CONTRACT_VERSION,
        "materialization": materialization,
        "source_motion_sha256": source_motion_sha256,
        "source_frame_count": source_frame_count,
        "embedded_tensors_sha256": embedded_tensors_sha256,
        "embedded_frame_count": embedded_frame_count,
        "motion_transition_contract_sha256": motion_transition_contract_sha256,
        "effective_prepend_steps": effective_prepend_steps,
        "effective_append_steps": effective_append_steps,
    }
    canonical = _validate_embedded_motion_timeline_contract_fields(contract)
    return canonical, _canonical_json_sha256(canonical)


def _validate_sha256(value: Any, *, path: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        suffix = " or null" if optional else ""
        raise PolicyContractError(
            f"{path} must be 64 lowercase hexadecimal characters{suffix}."
        )
    return value


def _validate_bounded_integer(
    value: Any,
    *,
    path: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise PolicyContractError(f"{path} must be an integer.")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        if maximum is None:
            requirement = f"at least {minimum}"
        else:
            requirement = f"in [{minimum}, {maximum}]"
        raise PolicyContractError(f"{path} must be {requirement}.")
    return result


def _validate_embedded_motion_timeline_contract_fields(
    raw_contract: Any,
) -> dict[str, Any]:
    expected_keys = {
        "version",
        "materialization",
        "source_motion_sha256",
        "source_frame_count",
        "embedded_tensors_sha256",
        "embedded_frame_count",
        "motion_transition_contract_sha256",
        "effective_prepend_steps",
        "effective_append_steps",
    }
    if not isinstance(raw_contract, Mapping) or set(raw_contract) != expected_keys:
        actual = set(raw_contract) if isinstance(raw_contract, Mapping) else set()
        raise PolicyContractError(
            "embedded_motion_timeline_contract must contain exactly "
            f"{sorted(expected_keys)}; missing={sorted(expected_keys - actual)}, "
            f"unexpected={sorted(repr(key) for key in actual - expected_keys)}."
        )
    version = raw_contract["version"]
    if isinstance(version, bool) or not isinstance(version, Integral) or int(version) != 1:
        raise PolicyContractError("embedded_motion_timeline_contract.version must equal integer 1.")
    materialization = raw_contract["materialization"]
    if not isinstance(materialization, str) or materialization not in {
        "effective_training_timeline",
        "raw_unsafe_diagnostic",
    }:
        raise PolicyContractError(
            "embedded_motion_timeline_contract.materialization must be exactly "
            "'effective_training_timeline' or 'raw_unsafe_diagnostic'."
        )
    source_sha = _validate_sha256(
        raw_contract["source_motion_sha256"],
        path="embedded_motion_timeline_contract.source_motion_sha256",
    )
    tensor_sha = _validate_sha256(
        raw_contract["embedded_tensors_sha256"],
        path="embedded_motion_timeline_contract.embedded_tensors_sha256",
    )
    transition_sha = _validate_sha256(
        raw_contract["motion_transition_contract_sha256"],
        path="embedded_motion_timeline_contract.motion_transition_contract_sha256",
        optional=True,
    )
    source_frames = _validate_bounded_integer(
        raw_contract["source_frame_count"],
        path="embedded_motion_timeline_contract.source_frame_count",
        minimum=1,
    )
    embedded_frames = _validate_bounded_integer(
        raw_contract["embedded_frame_count"],
        path="embedded_motion_timeline_contract.embedded_frame_count",
        minimum=1,
    )
    prepend_steps = _validate_bounded_integer(
        raw_contract["effective_prepend_steps"],
        path="embedded_motion_timeline_contract.effective_prepend_steps",
        minimum=0,
        maximum=_MAX_TRANSITION_STEPS,
    )
    append_steps = _validate_bounded_integer(
        raw_contract["effective_append_steps"],
        path="embedded_motion_timeline_contract.effective_append_steps",
        minimum=0,
        maximum=_MAX_TRANSITION_STEPS,
    )
    if transition_sha is None and (prepend_steps or append_steps):
        raise PolicyContractError(
            "A material transition requires motion_transition_contract_sha256 provenance."
        )
    expected_frames = source_frames
    if materialization == "effective_training_timeline":
        expected_frames += prepend_steps + append_steps
    elif not (prepend_steps or append_steps):
        raise PolicyContractError(
            "raw_unsafe_diagnostic is meaningful only when an effective transition was skipped."
        )
    if embedded_frames != expected_frames:
        raise PolicyContractError(
            "Embedded motion frame count contradicts its materialization state: "
            f"declared={embedded_frames}, expected={expected_frames}."
        )
    return {
        "version": 1,
        "materialization": materialization,
        "source_motion_sha256": source_sha,
        "source_frame_count": source_frames,
        "embedded_tensors_sha256": tensor_sha,
        "embedded_frame_count": embedded_frames,
        "motion_transition_contract_sha256": transition_sha,
        "effective_prepend_steps": prepend_steps,
        "effective_append_steps": append_steps,
    }


def embedded_motion_timeline_contract_from_metadata(
    metadata: Mapping[str, Any],
    *,
    required: bool = False,
) -> dict[str, Any] | None:
    raw_contract = metadata.get(EMBEDDED_MOTION_TIMELINE_CONTRACT_KEY)
    declared_digest = metadata.get(EMBEDDED_MOTION_TIMELINE_CONTRACT_SHA256_KEY)
    if raw_contract is None and declared_digest is None:
        if required:
            raise PolicyContractError(
                "Policy artifact is missing embedded_motion_timeline_contract provenance."
            )
        return None
    if raw_contract is None or declared_digest is None:
        raise PolicyContractError(
            "Embedded motion timeline metadata must include both its contract and SHA-256."
        )
    contract = _validate_embedded_motion_timeline_contract_fields(raw_contract)
    digest = _validate_sha256(
        declared_digest,
        path=EMBEDDED_MOTION_TIMELINE_CONTRACT_SHA256_KEY,
    )
    computed = _canonical_json_sha256(contract)
    if digest != computed:
        raise PolicyContractError(
            "Embedded motion timeline contract SHA-256 does not match its serialized contract: "
            f"declared={digest}, computed={computed}."
        )

    settings = effective_motion_transition_settings_from_metadata(metadata)
    if contract["motion_transition_contract_sha256"] != settings["contract_sha256"]:
        raise PolicyContractError(
            "Embedded motion timeline provenance is bound to a different motion transition contract."
        )
    for phase_name in ("prepend", "append"):
        if contract[f"effective_{phase_name}_steps"] != int(settings[phase_name]["steps"]):
            raise PolicyContractError(
                f"Embedded motion timeline {phase_name} steps contradict the effective training contract."
            )
    return contract


def validate_embedded_motion_timeline_model(
    model: onnx.ModelProto,
    metadata: Mapping[str, Any],
    *,
    allow_unsafe_diagnostic: bool = False,
    allow_legacy_unprovenanced_repatch_source: bool = False,
) -> dict[str, Any] | None:
    """Validate provenance against the actual constants in one parsed ONNX payload."""

    if type(allow_unsafe_diagnostic) is not bool:
        raise PolicyContractError("allow_unsafe_diagnostic must be boolean.")
    if type(allow_legacy_unprovenanced_repatch_source) is not bool:
        raise PolicyContractError(
            "allow_legacy_unprovenanced_repatch_source must be boolean."
        )
    contract = embedded_motion_timeline_contract_from_metadata(metadata)
    if contract is None:
        experiment_config = metadata.get("experiment_config")
        command = (
            experiment_config.get("command")
            if isinstance(experiment_config, Mapping)
            else None
        )
        setup_terms = command.get("setup_terms") if isinstance(command, Mapping) else None
        motion_command = (
            setup_terms.get("motion_command") if isinstance(setup_terms, Mapping) else None
        )
        params = motion_command.get("params") if isinstance(motion_command, Mapping) else None
        motion_cfg = params.get("motion_config") if isinstance(params, Mapping) else None
        motion_file = motion_cfg.get("motion_file") if isinstance(motion_cfg, Mapping) else None
        clip_id = motion_cfg.get("motion_clip_id") if isinstance(motion_cfg, Mapping) else None
        clip_name = motion_cfg.get("motion_clip_name") if isinstance(motion_cfg, Mapping) else None
        # This is the exact metadata signature written by the legacy patcher.
        # Original exporter artifacts normally retain a bank/directory source
        # and null clip selectors.  The patcher may ingest this legacy form to
        # repair it, but deployment must not guess which timeline it embedded.
        looks_like_legacy_patch = (
            isinstance(motion_file, str)
            and bool(motion_file)
            and Path(motion_file).suffix.lower() == ".npz"
            and isinstance(clip_id, int)
            and not isinstance(clip_id, bool)
            and clip_id == 0
            and isinstance(clip_name, str)
            and bool(clip_name)
            and clip_name == Path(motion_file).stem
        )
        if looks_like_legacy_patch and not allow_legacy_unprovenanced_repatch_source:
            raise PolicyContractError(
                "Legacy patched ONNX has an ambiguous embedded motion timeline but no "
                "embedded_motion_timeline_contract provenance; re-run patch_motion_onnx."
            )
        return None
    tensors = embedded_motion_tensors_from_model(model)
    actual_sha, actual_frames = embedded_motion_tensors_sha256(tensors)
    if actual_sha != contract["embedded_tensors_sha256"]:
        raise PolicyContractError(
            "Embedded motion tensors do not match their SHA-256 provenance: "
            f"declared={contract['embedded_tensors_sha256']}, actual={actual_sha}."
        )
    if actual_frames != contract["embedded_frame_count"]:
        raise PolicyContractError(
            "Embedded motion tensors do not match their declared frame count: "
            f"declared={contract['embedded_frame_count']}, actual={actual_frames}."
        )
    _validate_embedded_motion_indexing(
        model,
        embedded_frame_count=actual_frames,
    )
    if contract["materialization"] == "raw_unsafe_diagnostic" and not allow_unsafe_diagnostic:
        raise PolicyContractError(
            "ONNX embeds a raw_unsafe_diagnostic motion timeline that intentionally skips the "
            "authenticated effective training transition."
        )
    return contract
