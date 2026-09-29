"""Small, simulator-independent validators for checkpoint state.

These helpers deliberately validate values before callers mutate a live model.
They are shared by the short-lived preflight processes and the in-process PPO
load paths so direct callers cannot bypass the same fail-closed checks.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from holosoma.config_types.algo import (
    MAX_FLOW_INTEGRATION_STEPS,
    MAX_FLOW_NOISE_STD,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TERMINAL_FIXED_BC_EVAL_STATE_VERSION = 1
TERMINAL_FIXED_BC_EVAL_STATE_KEY = "terminal_fixed_bc_eval"
TERMINAL_FIXED_BC_EVAL_STATE_SHA256_KEY = "terminal_fixed_bc_eval_sha256"
FIXED_BC_EVAL_ALLOCATION_VERSION = 1
FIXED_BC_EVAL_ALLOCATION_SCHEME = "rank_quotient_remainder"


@dataclass(frozen=True)
class CheckpointFileSecurityContract:
    """Metadata requirements checked on the descriptor used for loading.

    The ordinary checkpoint loader authenticates bytes and descriptor
    stability.  Controller-owned scientific inputs additionally need a
    private, single-link pathname contract so a permissive cache entry or a
    pathname replacement cannot be silently accepted before launch.
    """

    owner_uid: int
    mode: int = 0o400
    link_count: int = 1
    minimum_size: int = 1
    bind_pathname: bool = True


def canonical_student_policy_type(actor_type: Any) -> str:
    """Map persisted actor class names to the launcher policy contract.

    Perception overrides persist the effective runtime class name in the
    experiment config, so an MLP student can legitimately be saved as
    ``MLPPerceptionEncoder`` (and its flow counterpart as
    ``FlowMLPPerceptionEncoder``).  Match a small explicit leaf-name set; a
    substring test would silently accept unrelated future actor classes.
    """

    raw_type = "" if actor_type is None else str(actor_type).strip()
    leaf_type = raw_type.rsplit(".", 1)[-1]
    normalized = leaf_type.casefold().replace("-", "").replace("_", "")
    if normalized in {"mlp", "mlpperceptionencoder"}:
        return "mlp"
    if normalized in {"flowmlp", "flowmlpperceptionencoder"}:
        return "flow"
    raise ValueError(f"Unsupported actor type in training-resume checkpoint: {actor_type!r}")


def validate_student_actor_contract(actor: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize the persisted student actor architecture.

    This is intentionally stricter than module construction: checkpoint
    metadata is an ownership/scientific-resume contract, not permissive user
    input.  In particular, booleans are not integer dimensions/step counts,
    observation groups are ordered and unique, and all flow scalars are
    finite and in the same ranges accepted by a fresh launcher.
    """

    if not isinstance(actor, Mapping):
        raise ValueError("Training-resume checkpoint actor contract must be a mapping.")
    layer = actor.get("layer_config")
    if not isinstance(layer, Mapping):
        raise ValueError("Training-resume checkpoint actor layer_config must be a mapping.")

    policy_type = canonical_student_policy_type(actor.get("type"))
    raw_dims = layer.get("hidden_dims")
    if (
        not isinstance(raw_dims, (list, tuple))
        or not raw_dims
        or any(type(value) is not int or value < 1 or value > 2_147_483_647 for value in raw_dims)
    ):
        raise ValueError(
            f"Invalid actor hidden dims in training-resume checkpoint: {raw_dims!r}"
        )
    hidden_dims = tuple(raw_dims)

    raw_inputs = actor.get("input_dim")
    if (
        not isinstance(raw_inputs, (list, tuple))
        or not raw_inputs
        or any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in raw_inputs
        )
        or len(set(raw_inputs)) != len(raw_inputs)
    ):
        raise ValueError(
            f"Invalid actor input groups in training-resume checkpoint: {raw_inputs!r}"
        )
    actor_inputs = tuple(raw_inputs)

    raw_flow_steps = layer.get("flow_integration_steps", 4)
    if (
        type(raw_flow_steps) is not int
        or not 1 <= raw_flow_steps <= MAX_FLOW_INTEGRATION_STEPS
    ):
        raise ValueError(
            "Invalid flow_integration_steps in training-resume checkpoint: "
            f"{raw_flow_steps!r}"
        )

    def finite_float(name: str, default: float, *, upper: float | None = None) -> float:
        raw_value = layer.get(name, default)
        if isinstance(raw_value, bool) or not isinstance(raw_value, numbers.Real):
            raise ValueError(
                f"Invalid {name} in training-resume checkpoint: {raw_value!r}"
            )
        try:
            value = float(raw_value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                f"Invalid {name} in training-resume checkpoint: {raw_value!r}"
            ) from exc
        if not math.isfinite(value) or value < 0.0 or (upper is not None and value > upper):
            raise ValueError(
                f"Invalid {name} in training-resume checkpoint: {raw_value!r}"
            )
        return value

    flow_train_noise = finite_float(
        "flow_train_noise_std",
        1.0,
        upper=MAX_FLOW_NOISE_STD,
    )
    flow_epsilon = finite_float("flow_time_epsilon", 1e-4, upper=0.49)
    flow_inference_noise = finite_float(
        "flow_inference_noise_std",
        0.0,
        upper=MAX_FLOW_NOISE_STD,
    )
    if policy_type != "flow" and (
        raw_flow_steps != 4
        or flow_train_noise != 1.0
        or flow_epsilon != 1e-4
        or flow_inference_noise != 0.0
    ):
        raise ValueError(
            "Non-default persisted flow settings require a FlowMLP student actor."
        )

    return {
        "policy_type": policy_type,
        "hidden_dims": hidden_dims,
        "actor_inputs": actor_inputs,
        "flow_steps": raw_flow_steps,
        "flow_train_noise": flow_train_noise,
        "flow_epsilon": flow_epsilon,
        "flow_inference_noise": flow_inference_noise,
    }


def _checkpoint_file_identity(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
    )


def _checkpoint_file_security_identity(
    stat_result: os.stat_result,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_mode),
        int(stat_result.st_nlink),
        int(stat_result.st_uid),
        int(stat_result.st_gid),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
    )


def _validate_checkpoint_file_security_contract(
    stat_result: os.stat_result,
    contract: CheckpointFileSecurityContract,
    *,
    path: Path,
) -> None:
    for name, value, minimum in (
        ("owner_uid", contract.owner_uid, 0),
        ("mode", contract.mode, 0),
        ("link_count", contract.link_count, 1),
        ("minimum_size", contract.minimum_size, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(
                f"Checkpoint file security contract {name} must be an integer >= {minimum}, "
                f"got {value!r}."
            )
    if contract.mode > 0o7777:
        raise ValueError(
            f"Checkpoint file security contract mode is invalid: {contract.mode!r}."
        )
    if type(contract.bind_pathname) is not bool:
        raise ValueError("Checkpoint file security contract bind_pathname must be boolean.")
    if not stat.S_ISREG(stat_result.st_mode):
        raise ValueError(f"Checkpoint is not a regular file: {path}.")
    actual_mode = stat.S_IMODE(stat_result.st_mode)
    if (
        int(stat_result.st_uid) != contract.owner_uid
        or actual_mode != contract.mode
        or int(stat_result.st_nlink) != contract.link_count
        or int(stat_result.st_size) < contract.minimum_size
    ):
        raise ValueError(
            "Checkpoint violates its private file metadata contract: "
            f"path={path} owner={stat_result.st_uid} mode={actual_mode:04o} "
            f"links={stat_result.st_nlink} size={stat_result.st_size}."
        )


def _sha256_open_checkpoint(stream) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def load_verified_torch_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
    map_location: Any = "cpu",
    file_security: CheckpointFileSecurityContract | None = None,
) -> tuple[Any, str]:
    """Safely load the exact stable bytes whose SHA256 was authenticated.

    A single no-follow file descriptor is used for both hashing and loading,
    closing path-replacement races.  Identity and digest are checked again
    after deserialization to reject in-place mutation.  The weights-only
    loader forbids arbitrary pickle globals before any checkpoint content is
    trusted.
    """

    path = Path(checkpoint_path).expanduser()
    if file_security is not None:
        # Preserve the lexical final component so O_NOFOLLOW can reject an
        # alias instead of resolving it before the descriptor is opened.
        path = Path(os.path.abspath(os.fspath(path)))
        if not hasattr(os, "O_NOFOLLOW"):
            raise RuntimeError(
                "Private checkpoint validation requires O_NOFOLLOW support."
            )
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str) or _SHA256_RE.fullmatch(expected_sha256) is None
    ):
        raise ValueError(
            f"Expected checkpoint SHA256 must be 64 lowercase hexadecimal characters, "
            f"got {expected_sha256!r}."
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise OSError(f"Unable to open checkpoint as a no-follow regular file: {path}: {exc}") from exc

    with os.fdopen(descriptor, "rb", closefd=True) as stream:
        initial_stat = os.fstat(stream.fileno())
        if not stat.S_ISREG(initial_stat.st_mode):
            raise ValueError(f"Checkpoint is not a regular file: {path}.")
        initial_security_identity = _checkpoint_file_security_identity(initial_stat)
        if file_security is not None:
            _validate_checkpoint_file_security_contract(
                initial_stat,
                file_security,
                path=path,
            )
            if file_security.bind_pathname:
                try:
                    initial_path_stat = os.lstat(path)
                except OSError as exc:
                    raise OSError(
                        f"Unable to bind private checkpoint pathname: {path}: {exc}"
                    ) from exc
                if (
                    _checkpoint_file_security_identity(initial_path_stat)
                    != initial_security_identity
                ):
                    raise RuntimeError(
                        "Private checkpoint pathname does not identify the opened file: "
                        f"path={path}."
                    )
        initial_identity = _checkpoint_file_identity(initial_stat)
        initial_digest = _sha256_open_checkpoint(stream)
        after_hash_identity = _checkpoint_file_identity(os.fstat(stream.fileno()))
        if after_hash_identity != initial_identity:
            raise RuntimeError(
                "Checkpoint changed while its SHA256 was being computed: "
                f"path={path} before={initial_identity} after={after_hash_identity}."
            )
        if expected_sha256 is not None and initial_digest != expected_sha256:
            raise ValueError(
                "Checkpoint SHA256 does not match the authenticated training provenance: "
                f"path={path} expected={expected_sha256} actual={initial_digest}."
            )

        stream.seek(0)
        checkpoint = torch.load(
            stream,
            map_location=map_location,
            weights_only=True,
        )
        after_load_identity = _checkpoint_file_identity(os.fstat(stream.fileno()))
        if after_load_identity != initial_identity:
            raise RuntimeError(
                "Checkpoint changed while it was being safely deserialized: "
                f"path={path} before={initial_identity} after={after_load_identity}."
            )
        final_digest = _sha256_open_checkpoint(stream)
        final_stat = os.fstat(stream.fileno())
        final_identity = _checkpoint_file_identity(final_stat)
        if final_identity != initial_identity or final_digest != initial_digest:
            raise RuntimeError(
                "Checkpoint changed across verified deserialization: "
                f"path={path} identity_before={initial_identity} identity_after={final_identity} "
                f"sha256_before={initial_digest} sha256_after={final_digest}."
            )
        if file_security is not None:
            final_security_identity = _checkpoint_file_security_identity(final_stat)
            if final_security_identity != initial_security_identity:
                raise RuntimeError(
                    "Private checkpoint metadata changed across verified deserialization: "
                    f"path={path}."
                )
            _validate_checkpoint_file_security_contract(
                final_stat,
                file_security,
                path=path,
            )
            if file_security.bind_pathname:
                try:
                    final_path_stat = os.lstat(path)
                except OSError as exc:
                    raise OSError(
                        f"Private checkpoint pathname disappeared while loading: {path}: {exc}"
                    ) from exc
                if (
                    _checkpoint_file_security_identity(final_path_stat)
                    != initial_security_identity
                ):
                    raise RuntimeError(
                        "Private checkpoint pathname changed across verified deserialization: "
                        f"path={path}."
                    )
    return checkpoint, initial_digest


def _require_integral(value: Any, *, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"Checkpoint {path} must be an integer, got {value!r}.")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"Checkpoint {path} must be >= {minimum}, got {parsed}.")
    return parsed


def validate_checkpoint_iterations(checkpoint: Mapping[str, Any]) -> tuple[int, int]:
    """Return ``(completed_iter, next_iter)`` after validating save semantics.

    New checkpoints serialize both ``iter`` and ``next_iter``.  Legacy
    checkpoints may omit ``next_iter`` and/or use ``iteration`` as the
    completed-update field; those forms remain supported without allowing an
    explicit contradictory counter to silently skip or repeat curriculum
    updates.
    """

    if "iter" in checkpoint:
        completed_iter = _require_integral(checkpoint["iter"], path="iter")
        if "iteration" in checkpoint:
            metadata_iter = _require_integral(checkpoint["iteration"], path="iteration")
            if metadata_iter != completed_iter:
                raise ValueError(
                    "Checkpoint iteration metadata is inconsistent: "
                    f"iter={completed_iter}, iteration={metadata_iter}."
                )
    elif "iteration" in checkpoint:
        completed_iter = _require_integral(checkpoint["iteration"], path="iteration")
    else:
        raise ValueError("Checkpoint is missing completed-iteration metadata ('iter' or 'iteration').")

    expected_next_iter = completed_iter + 1
    if "next_iter" not in checkpoint:
        return completed_iter, expected_next_iter

    next_iter = _require_integral(checkpoint["next_iter"], path="next_iter")
    if next_iter != expected_next_iter:
        raise ValueError(
            "Checkpoint iteration metadata is inconsistent: explicit next_iter must equal iter + 1; "
            f"iter={completed_iter}, next_iter={next_iter}, expected={expected_next_iter}."
        )
    return completed_iter, next_iter


def terminal_fixed_bc_eval_state_sha256(state: Mapping[str, Any]) -> str:
    """Return the canonical digest embedded beside a terminal BC proof."""

    return hashlib.sha256(
        json.dumps(
            dict(state),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def validate_terminal_fixed_bc_eval_state(
    state: Any,
    *,
    expected_completed_iteration: int | None = None,
) -> dict[str, Any]:
    """Validate the immutable final fixed-BC observation schema."""

    if not isinstance(state, Mapping):
        raise ValueError("terminal_fixed_bc_eval must be a mapping.")
    required = {
        "version",
        "terminal_observation",
        "completed_iteration",
        "next_iteration",
        "run_target_iteration",
        "scheduled_evaluation",
        "guard_enabled",
        "guard_applied",
        "fixed_bc_eval_log_interval",
        "fixed_bc_eval_num_samples",
        "world_size",
        "fixed_bc_global_dataset_sha256",
        "fixed_bc_guard_config_sha256",
        "fixed_bc_guard_state_sha256",
        "fixed_bc_guard_threshold_mu_mse",
        "fixed_bc_terminal_within_threshold",
        "fixed_bc_mu_mse",
        "fixed_bc_num_samples",
        "fixed_bc_weighted_num_samples",
        "fixed_bc_expected_weighted_num_samples",
        "fixed_bc_rank_strata",
    }
    if set(state) != required:
        raise ValueError(
            "terminal_fixed_bc_eval has an invalid field set: "
            f"missing={sorted(required - set(state))}, extra={sorted(set(state) - required)}."
        )
    if (
        type(state["version"]) is not int
        or state["version"] != TERMINAL_FIXED_BC_EVAL_STATE_VERSION
    ):
        raise ValueError(
            "terminal_fixed_bc_eval.version must equal "
            f"{TERMINAL_FIXED_BC_EVAL_STATE_VERSION}."
        )
    for key in (
        "terminal_observation",
        "scheduled_evaluation",
        "guard_enabled",
        "guard_applied",
    ):
        if type(state[key]) is not bool:
            raise ValueError(f"terminal_fixed_bc_eval.{key} must be boolean.")
    if state["terminal_observation"] is not True:
        raise ValueError("terminal_fixed_bc_eval.terminal_observation must be true.")
    for key, minimum in (
        ("completed_iteration", 0),
        ("next_iteration", 1),
        ("run_target_iteration", 1),
        ("fixed_bc_eval_log_interval", 1),
        ("fixed_bc_eval_num_samples", 1),
        ("world_size", 1),
        ("fixed_bc_num_samples", 1),
        ("fixed_bc_rank_strata", 1),
    ):
        value = state[key]
        if type(value) is not int or value < minimum:
            raise ValueError(
                f"terminal_fixed_bc_eval.{key} must be an integer >= {minimum}."
            )
    completed_iteration = state["completed_iteration"]
    if state["next_iteration"] != completed_iteration + 1:
        raise ValueError(
            "terminal_fixed_bc_eval next/completed iteration fields are inconsistent."
        )
    if state["run_target_iteration"] != state["next_iteration"]:
        raise ValueError(
            "terminal_fixed_bc_eval is not bound to the saved run target."
        )
    if (
        expected_completed_iteration is not None
        and completed_iteration != int(expected_completed_iteration)
    ):
        raise ValueError(
            "terminal_fixed_bc_eval is bound to the wrong checkpoint iteration: "
            f"state={completed_iteration}, expected={expected_completed_iteration}."
        )
    interval = state["fixed_bc_eval_log_interval"]
    expected_scheduled = completed_iteration % interval == 0
    if state["scheduled_evaluation"] is not expected_scheduled:
        raise ValueError(
            "terminal_fixed_bc_eval scheduled_evaluation does not match its iteration/cadence."
        )
    expected_guard_applied = bool(
        state["guard_enabled"] and state["scheduled_evaluation"]
    )
    if state["guard_applied"] is not expected_guard_applied:
        raise ValueError(
            "terminal_fixed_bc_eval guard_applied must equal "
            "guard_enabled && scheduled_evaluation."
        )
    if state["fixed_bc_num_samples"] != state["fixed_bc_eval_num_samples"]:
        raise ValueError(
            "terminal_fixed_bc_eval realized sample count differs from its configured budget."
        )
    expected_strata = min(
        state["fixed_bc_eval_num_samples"], state["world_size"]
    )
    if state["fixed_bc_rank_strata"] != expected_strata:
        raise ValueError(
            "terminal_fixed_bc_eval rank-strata count differs from its allocation contract."
        )
    for key in (
        "fixed_bc_mu_mse",
        "fixed_bc_weighted_num_samples",
        "fixed_bc_expected_weighted_num_samples",
    ):
        value = state[key]
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"terminal_fixed_bc_eval.{key} must be a real scalar.")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0.0:
            raise ValueError(
                f"terminal_fixed_bc_eval.{key} must be finite and non-negative."
            )
    if float(state["fixed_bc_weighted_num_samples"]) <= 0.0:
        raise ValueError(
            "terminal_fixed_bc_eval.fixed_bc_weighted_num_samples must be positive."
        )
    if float(state["fixed_bc_expected_weighted_num_samples"]) <= 0.0:
        raise ValueError(
            "terminal_fixed_bc_eval.fixed_bc_expected_weighted_num_samples must be positive."
        )
    if not math.isclose(
        float(state["fixed_bc_weighted_num_samples"]),
        float(state["fixed_bc_expected_weighted_num_samples"]),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "terminal_fixed_bc_eval weighted sample count differs from its "
            "rank-weighted allocation contract."
        )

    for key in (
        "fixed_bc_global_dataset_sha256",
        "fixed_bc_guard_config_sha256",
    ):
        digest = state[key]
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"terminal_fixed_bc_eval.{key} is malformed.")

    guard_state_digest = state["fixed_bc_guard_state_sha256"]
    threshold = state["fixed_bc_guard_threshold_mu_mse"]
    within_threshold = state["fixed_bc_terminal_within_threshold"]
    if state["guard_enabled"]:
        if (
            not isinstance(guard_state_digest, str)
            or _SHA256_RE.fullmatch(guard_state_digest) is None
        ):
            raise ValueError(
                "terminal_fixed_bc_eval.fixed_bc_guard_state_sha256 is malformed."
            )
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, numbers.Real)
            or not math.isfinite(float(threshold))
            or float(threshold) < 0.0
        ):
            raise ValueError(
                "terminal_fixed_bc_eval requires a finite frozen guard threshold."
            )
        if type(within_threshold) is not bool:
            raise ValueError(
                "terminal_fixed_bc_eval.fixed_bc_terminal_within_threshold "
                "must be boolean when the guard is enabled."
            )
        expected_within_threshold = float(state["fixed_bc_mu_mse"]) <= float(
            threshold
        )
        if within_threshold is not expected_within_threshold:
            raise ValueError(
                "terminal_fixed_bc_eval threshold verdict is inconsistent with its MSE."
            )
        if not within_threshold:
            raise RuntimeError(
                "Final fixed-BC observation exceeds the frozen scientific guard threshold: "
                f"mu_mse={float(state['fixed_bc_mu_mse']):.17g}, "
                f"threshold={float(threshold):.17g}."
            )
    elif (
        guard_state_digest is not None
        or threshold is not None
        or within_threshold is not None
    ):
        raise ValueError(
            "terminal_fixed_bc_eval guard state, threshold, and verdict must all "
            "be null when the guard is disabled."
        )
    return dict(state)


def validate_checkpoint_terminal_fixed_bc_eval(
    checkpoint: Mapping[str, Any],
    *,
    expected_completed_iteration: int,
) -> dict[str, Any] | None:
    """Validate the paired terminal proof and its canonical state digest."""

    state_present = TERMINAL_FIXED_BC_EVAL_STATE_KEY in checkpoint
    digest_present = TERMINAL_FIXED_BC_EVAL_STATE_SHA256_KEY in checkpoint
    if not state_present and not digest_present:
        return None
    if state_present != digest_present:
        raise ValueError(
            "Checkpoint terminal fixed-BC observation must provide both state and SHA256."
        )
    state = checkpoint.get(TERMINAL_FIXED_BC_EVAL_STATE_KEY)
    digest = checkpoint.get(TERMINAL_FIXED_BC_EVAL_STATE_SHA256_KEY)
    if state is None or digest is None:
        raise ValueError(
            "Checkpoint terminal fixed-BC observation must provide both state and SHA256."
        )
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ValueError("Checkpoint terminal_fixed_bc_eval_sha256 is malformed.")
    canonical = validate_terminal_fixed_bc_eval_state(
        state,
        expected_completed_iteration=expected_completed_iteration,
    )
    computed = terminal_fixed_bc_eval_state_sha256(canonical)
    if digest != computed:
        raise ValueError(
            "Checkpoint terminal_fixed_bc_eval_sha256 does not authenticate its state."
        )
    return canonical


def checkpoint_saved_run_target(checkpoint: Mapping[str, Any]) -> int:
    experiment_config = checkpoint.get("experiment_config")
    algo = (
        experiment_config.get("algo")
        if isinstance(experiment_config, Mapping)
        else None
    )
    algo_config = algo.get("config") if isinstance(algo, Mapping) else None
    target = (
        algo_config.get("num_learning_iterations")
        if isinstance(algo_config, Mapping)
        else None
    )
    if type(target) is not int or target < 1:
        raise ValueError(
            "A terminal fixed-BC checkpoint requires a positive saved "
            "experiment_config.algo.config.num_learning_iterations."
        )
    return target


def fixed_bc_dataset_sha256(
    dataset: Mapping[str, Any],
    *,
    expected_rows: int,
    required_tensor_keys: set[str] | frozenset[str],
    context: str,
    expected_widths: Mapping[str, int] | None = None,
    expected_dtype: torch.dtype | None = None,
) -> str:
    """Authenticate one immutable fixed-BC rank stratum.

    Widths and dtype are optional so a simulator-independent controller can
    verify the exact serialized dataset bytes.  PPO supplies its live tensor
    contract as a second, stronger semantic check before actor commit.
    """

    if not isinstance(dataset, Mapping):
        raise ValueError(f"{context} must be a tensor mapping.")
    if type(expected_rows) is not int or expected_rows < 0:
        raise ValueError(f"{context} expected_rows must be a non-negative integer.")
    required = set(required_tensor_keys)
    if (
        not required
        or any(not isinstance(key, str) or not key for key in required)
    ):
        raise ValueError(f"{context} required tensor keys are invalid.")
    if expected_rows == 0:
        unexpected = required.intersection(dataset)
        if unexpected:
            raise ValueError(
                f"{context} has tensors for a zero-row allocation: {sorted(unexpected)}."
            )
    else:
        missing = required - set(dataset)
        if missing:
            raise ValueError(f"{context} is missing tensors: {sorted(missing)}.")

    widths = dict(expected_widths or {})
    unknown_widths = set(widths) - required
    if unknown_widths:
        raise ValueError(
            f"{context} has width contracts for unknown tensors: {sorted(unknown_widths)}."
        )
    for key, width in widths.items():
        if type(width) is not int or width <= 0:
            raise ValueError(f"{context} expected width for {key!r} must be positive.")

    digest = hashlib.sha256()
    digest.update(b"holosoma-fixed-bc-dataset-v1\x00")
    digest.update(str(expected_rows).encode("ascii"))
    for key in sorted(required):
        if expected_rows == 0:
            continue
        value = dataset[key]
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise ValueError(f"{context}[{key!r}] must be a dense strided tensor.")
        if value.ndim != 2:
            raise ValueError(f"{context}[{key!r}] must be rank 2.")
        if expected_dtype is not None and value.dtype != expected_dtype:
            raise ValueError(
                f"{context}[{key!r}] dtype={value.dtype} != expected {expected_dtype}."
            )
        if int(value.shape[0]) != expected_rows:
            raise ValueError(
                f"{context}[{key!r}] rows={value.shape[0]} != expected {expected_rows}."
            )
        if int(value.shape[1]) <= 0:
            raise ValueError(f"{context}[{key!r}] must have a positive width.")
        if key in widths and int(value.shape[1]) != widths[key]:
            raise ValueError(
                f"{context}[{key!r}] width={value.shape[1]} != expected {widths[key]}."
            )
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{context}[{key!r}] contains NaN or infinity.")
        cpu_value = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(cpu_value.dtype).encode("ascii"))
        digest.update(b"\x00")
        digest.update(
            json.dumps(list(cpu_value.shape), separators=(",", ":")).encode(
                "ascii"
            )
        )
        digest.update(b"\x00")
        digest.update(cpu_value.view(torch.uint8).numpy().tobytes(order="C"))
        digest.update(b"\xff")
    return digest.hexdigest()


def fixed_bc_global_dataset_sha256(
    digest_by_rank: Mapping[str, str],
    *,
    global_sample_budget: int,
    world_size: int,
) -> str:
    payload = {
        "version": 1,
        "allocation_scheme": FIXED_BC_EVAL_ALLOCATION_SCHEME,
        "global_sample_budget": int(global_sample_budget),
        "world_size": int(world_size),
        "local_dataset_digest_by_rank": {
            str(rank): digest_by_rank[str(rank)] for rank in range(world_size)
        },
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def validate_terminal_fixed_bc_eval_artifact_payload(
    checkpoint: Mapping[str, Any],
    *,
    expected_completed_iteration: int,
    required_tensor_keys: set[str] | frozenset[str],
    expected_widths: Mapping[str, int] | None = None,
    expected_dtype: torch.dtype | None = None,
    expected_guard_config_sha256: str | None = None,
    require_terminal: bool = False,
) -> dict[str, Any] | None:
    """Validate a terminal proof, frozen dataset, and optional guard state."""

    terminal_state = validate_checkpoint_terminal_fixed_bc_eval(
        checkpoint,
        expected_completed_iteration=expected_completed_iteration,
    )
    if terminal_state is None:
        if require_terminal:
            raise ValueError(
                "Policy-init source must contain a terminal fixed-BC observation and SHA256."
            )
        return None

    saved_target = checkpoint_saved_run_target(checkpoint)
    if saved_target != terminal_state["run_target_iteration"]:
        raise ValueError(
            "Terminal fixed-BC observation is attached to a non-final checkpoint: "
            f"saved_target={saved_target}, state_target="
            f"{terminal_state['run_target_iteration']}."
        )

    states_by_rank = checkpoint.get("fixed_bc_eval_by_rank")
    world_size = int(terminal_state["world_size"])
    budget = int(terminal_state["fixed_bc_eval_num_samples"])
    expected_keys = {str(rank) for rank in range(world_size)}
    if not isinstance(states_by_rank, Mapping) or set(states_by_rank) != expected_keys:
        actual_keys = (
            sorted(str(key) for key in states_by_rank)
            if isinstance(states_by_rank, Mapping)
            else []
        )
        raise ValueError(
            "Terminal fixed-BC checkpoint dataset map does not match its saved world: "
            f"checkpoint={actual_keys}, expected={sorted(expected_keys)}."
        )
    quotient, remainder = divmod(budget, world_size)
    digest_by_rank: dict[str, str] = {}
    for rank in range(world_size):
        rank_state = states_by_rank[str(rank)]
        expected_target = quotient + int(rank < remainder)
        if not isinstance(rank_state, Mapping):
            raise ValueError(f"fixed_bc_eval_by_rank[{rank}] must be a mapping.")
        allocation_expectations = {
            "allocation_version": FIXED_BC_EVAL_ALLOCATION_VERSION,
            "allocation_scheme": FIXED_BC_EVAL_ALLOCATION_SCHEME,
            "global_sample_budget": budget,
            "world_size": world_size,
            "rank": rank,
            "local_target": expected_target,
            "ready": True,
            "size": expected_target,
        }
        mismatches = [
            f"{key}={rank_state.get(key)!r} != {expected!r}"
            for key, expected in allocation_expectations.items()
            if type(rank_state.get(key)) is not type(expected)
            or rank_state.get(key) != expected
        ]
        if mismatches:
            raise ValueError(
                f"fixed_bc_eval_by_rank[{rank}] is not the terminal frozen stratum: "
                + "; ".join(mismatches)
            )
        digest_by_rank[str(rank)] = fixed_bc_dataset_sha256(
            rank_state,
            expected_rows=expected_target,
            required_tensor_keys=required_tensor_keys,
            expected_widths=expected_widths,
            expected_dtype=expected_dtype,
            context=f"terminal checkpoint fixed BC dataset rank {rank}",
        )
    dataset_digest = fixed_bc_global_dataset_sha256(
        digest_by_rank,
        global_sample_budget=budget,
        world_size=world_size,
    )
    if dataset_digest != terminal_state["fixed_bc_global_dataset_sha256"]:
        raise ValueError(
            "Terminal fixed-BC observation does not authenticate the serialized frozen dataset."
        )

    if expected_guard_config_sha256 is not None:
        if (
            not isinstance(expected_guard_config_sha256, str)
            or _SHA256_RE.fullmatch(expected_guard_config_sha256) is None
        ):
            raise ValueError("Expected fixed-BC guard config SHA256 is malformed.")
        if (
            terminal_state["fixed_bc_guard_config_sha256"]
            != expected_guard_config_sha256
        ):
            raise ValueError(
                "Terminal fixed-BC guard configuration differs from the runtime."
            )

    guard_payload = checkpoint.get("fixed_bc_guard_state")
    if terminal_state["guard_enabled"]:
        if not isinstance(guard_payload, Mapping):
            raise ValueError(
                "Guard-enabled terminal fixed-BC observation requires serialized guard state."
            )
        if (
            guard_payload.get("config_fingerprint")
            != terminal_state["fixed_bc_guard_config_sha256"]
        ):
            raise ValueError(
                "Terminal fixed-BC observation guard fingerprint differs from its guard state."
            )
        guard_state_digest = terminal_fixed_bc_eval_state_sha256(guard_payload)
        if guard_state_digest != terminal_state["fixed_bc_guard_state_sha256"]:
            raise ValueError(
                "Terminal fixed-BC observation does not authenticate its periodic guard state."
            )
        guard_threshold = guard_payload.get("threshold_mu_mse")
        if (
            isinstance(guard_threshold, bool)
            or not isinstance(guard_threshold, numbers.Real)
            or float(guard_threshold)
            != float(terminal_state["fixed_bc_guard_threshold_mu_mse"])
        ):
            raise ValueError(
                "Terminal fixed-BC observation threshold differs from its guard state."
            )
    elif guard_payload is not None:
        raise ValueError(
            "Guard-disabled terminal fixed-BC checkpoint unexpectedly contains guard state."
        )
    return terminal_state


def validate_finite_tree(value: Any, *, path: str) -> None:
    """Reject non-finite numeric leaves in nested checkpoint state."""

    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all().item()):
            non_finite = int((~torch.isfinite(value)).sum().item())
            raise ValueError(
                f"Checkpoint {path} contains {non_finite} non-finite tensor value(s) "
                f"with shape={tuple(value.shape)} dtype={value.dtype}."
            )
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            validate_finite_tree(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            validate_finite_tree(child, path=f"{path}[{index}]")
        return
    if isinstance(value, numbers.Number) and not isinstance(value, (bool, numbers.Integral)):
        try:
            finite = math.isfinite(value)
        except TypeError:
            finite = math.isfinite(float(value.real)) and math.isfinite(float(value.imag))
        if not finite:
            raise ValueError(f"Checkpoint {path} must be finite, got {value!r}.")


def require_mapping(checkpoint: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = checkpoint.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Checkpoint {key} must be a mapping.")
    return value


def validate_module_state_compatibility(
    state: Mapping[str, Any],
    *,
    reference_state: Mapping[str, Any],
    path: str,
    allow_legacy_integral_count: bool = False,
) -> None:
    """Prove that a strict module load cannot partially copy then fail.

    ``Module.load_state_dict(strict=True)`` does not provide transactional
    semantics: it can copy compatible tensors before reporting a later shape
    mismatch.  Compare the complete key/type/shape/dtype/layout contract first
    so callers can reject an incompatible checkpoint without touching a live
    module.
    """

    expected_keys = set(reference_state)
    actual_keys = set(state)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys, key=repr)
        extra = sorted(actual_keys - expected_keys, key=repr)
        raise ValueError(
            f"Checkpoint {path} keys are incompatible with the runtime module: "
            f"missing={missing}, extra={extra}."
        )

    for key in reference_state:
        checkpoint_value = state[key]
        runtime_value = reference_state[key]
        value_path = f"{path}[{key!r}]"
        if isinstance(runtime_value, torch.nn.parameter.UninitializedParameter):
            if not isinstance(checkpoint_value, torch.Tensor):
                raise ValueError(
                    f"Checkpoint {value_path} must be a tensor for a lazy runtime parameter, got "
                    f"{type(checkpoint_value).__name__}."
                )
            continue
        if isinstance(runtime_value, torch.Tensor):
            if not isinstance(checkpoint_value, torch.Tensor):
                raise ValueError(
                    f"Checkpoint {value_path} must be a tensor, got "
                    f"{type(checkpoint_value).__name__}."
                )
            if checkpoint_value.shape != runtime_value.shape:
                raise ValueError(
                    f"Checkpoint {value_path} shape {tuple(checkpoint_value.shape)} is incompatible "
                    f"with runtime shape {tuple(runtime_value.shape)}."
                )
            legacy_count_dtype = (
                allow_legacy_integral_count
                and key == "count"
                and runtime_value.dtype == torch.float64
                and checkpoint_value.dtype
                in {
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                }
                and checkpoint_value.numel() == 1
                and abs(int(checkpoint_value.item())) <= 2**53
            )
            if checkpoint_value.dtype != runtime_value.dtype and not legacy_count_dtype:
                raise ValueError(
                    f"Checkpoint {value_path} dtype {checkpoint_value.dtype} is incompatible "
                    f"with runtime dtype {runtime_value.dtype}."
                )
            if checkpoint_value.layout != runtime_value.layout:
                raise ValueError(
                    f"Checkpoint {value_path} layout {checkpoint_value.layout} is incompatible "
                    f"with runtime layout {runtime_value.layout}."
                )
        elif isinstance(checkpoint_value, torch.Tensor) or type(checkpoint_value) is not type(runtime_value):
            raise ValueError(
                f"Checkpoint {value_path} type {type(checkpoint_value).__name__} is incompatible "
                f"with runtime type {type(runtime_value).__name__}."
            )


def _validate_optimizer_tensor_shapes(value: Any, *, parameter: torch.Tensor, path: str) -> None:
    """Reject optimizer tensor state that cannot describe its live parameter."""

    if isinstance(value, torch.Tensor):
        # Optimizers commonly store a scalar step plus parameter-shaped moment
        # tensors.  The PPO actor/critic optimizers use exactly that contract.
        if value.shape == parameter.shape:
            if value.dtype != parameter.dtype:
                raise ValueError(
                    f"Checkpoint {path} dtype {value.dtype} is incompatible with parameter "
                    f"dtype {parameter.dtype}."
                )
            if value.layout != parameter.layout:
                raise ValueError(
                    f"Checkpoint {path} layout {value.layout} is incompatible with parameter "
                    f"layout {parameter.layout}."
                )
        elif value.numel() != 1:
            raise ValueError(
                f"Checkpoint {path} shape {tuple(value.shape)} is incompatible with parameter "
                f"shape {tuple(parameter.shape)}."
            )
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_optimizer_tensor_shapes(child, parameter=parameter, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_optimizer_tensor_shapes(child, parameter=parameter, path=f"{path}[{index}]")


def _validate_parameter_state_tensor(value: Any, *, parameter: torch.Tensor, path: str) -> None:
    """Validate one optimizer moment/buffer against its exact parameter."""

    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Checkpoint {path} must be a tensor, got {type(value).__name__}.")
    if value.shape != parameter.shape:
        raise ValueError(
            f"Checkpoint {path} shape {tuple(value.shape)} is incompatible with parameter "
            f"shape {tuple(parameter.shape)}."
        )
    if value.dtype != parameter.dtype:
        raise ValueError(
            f"Checkpoint {path} dtype {value.dtype} is incompatible with parameter "
            f"dtype {parameter.dtype}."
        )
    if value.layout != parameter.layout:
        raise ValueError(
            f"Checkpoint {path} layout {value.layout} is incompatible with parameter "
            f"layout {parameter.layout}."
        )


def validate_optimizer_compatibility(
    state: Mapping[str, Any],
    *,
    optimizer: torch.optim.Optimizer,
    path: str,
) -> None:
    """Validate optimizer group/parameter topology before loading live state."""

    saved_groups = state.get("param_groups")
    if not isinstance(saved_groups, list):
        raise ValueError(f"Checkpoint {path}.param_groups must be a list.")
    runtime_groups = optimizer.param_groups
    runtime_serialized_groups = optimizer.state_dict().get("param_groups")
    if not isinstance(runtime_serialized_groups, list) or len(runtime_serialized_groups) != len(runtime_groups):
        raise ValueError(f"Runtime optimizer {path} produced an invalid parameter-group state contract.")
    if len(saved_groups) != len(runtime_groups):
        raise ValueError(
            f"Checkpoint {path} has {len(saved_groups)} parameter groups, but the runtime optimizer "
            f"has {len(runtime_groups)}."
        )

    parameter_by_saved_id: dict[int, torch.Tensor] = {}
    group_by_saved_id: dict[int, int] = {}
    for group_index, (saved_group, runtime_group) in enumerate(
        zip(saved_groups, runtime_groups, strict=True)
    ):
        group_path = f"{path}.param_groups[{group_index}]"
        if not isinstance(saved_group, Mapping):
            raise ValueError(f"Checkpoint {group_path} must be a mapping.")
        saved_parameters = saved_group.get("params")
        runtime_parameters = runtime_group.get("params")
        if not isinstance(saved_parameters, (list, tuple)):
            raise ValueError(f"Checkpoint {group_path}.params must be a list or tuple.")
        if not isinstance(runtime_parameters, (list, tuple)):
            raise ValueError(f"Runtime optimizer {group_path}.params must be a list or tuple.")
        if len(saved_parameters) != len(runtime_parameters):
            raise ValueError(
                f"Checkpoint {group_path} has {len(saved_parameters)} parameters, but the runtime "
                f"group has {len(runtime_parameters)}."
            )
        runtime_saved_parameters = runtime_serialized_groups[group_index].get("params")
        if not isinstance(runtime_saved_parameters, (list, tuple)):
            raise ValueError(f"Runtime optimizer {group_path}.params state must be a list or tuple.")
        missing_options = sorted((set(runtime_group) - {"params"}) - set(saved_group), key=repr)
        if missing_options:
            raise ValueError(
                f"Checkpoint {group_path} is missing runtime optimizer option(s) {missing_options}."
            )
        option_mismatches = []
        for option in set(runtime_group) - {"params", "lr"}:
            saved_value = saved_group[option]
            runtime_value = runtime_group[option]
            try:
                equal = type(saved_value) is type(runtime_value) and bool(saved_value == runtime_value)
            except (RuntimeError, TypeError, ValueError):
                equal = False
            if not equal:
                option_mismatches.append(
                    f"{option}: checkpoint={saved_value!r} runtime={runtime_value!r}"
                )
        if option_mismatches:
            raise ValueError(
                f"Checkpoint {group_path} optimizer options differ from the runtime configuration: "
                + "; ".join(sorted(option_mismatches))
            )
        parsed_group_ids: list[int] = []
        for parameter_index, (saved_id, runtime_parameter) in enumerate(
            zip(saved_parameters, runtime_parameters, strict=True)
        ):
            if isinstance(saved_id, bool) or not isinstance(saved_id, numbers.Integral):
                raise ValueError(
                    f"Checkpoint {group_path}.params[{parameter_index}] must be an integer id, "
                    f"got {saved_id!r}."
                )
            parsed_id = int(saved_id)
            if parsed_id in parameter_by_saved_id:
                raise ValueError(f"Checkpoint {path} repeats optimizer parameter id {parsed_id}.")
            if not isinstance(runtime_parameter, torch.Tensor):
                raise ValueError(
                    f"Runtime optimizer {group_path}.params[{parameter_index}] is not a tensor."
                )
            parameter_by_saved_id[parsed_id] = runtime_parameter
            group_by_saved_id[parsed_id] = group_index
            parsed_group_ids.append(parsed_id)
        if parsed_group_ids != list(runtime_saved_parameters):
            raise ValueError(
                f"Checkpoint {group_path}.params order/ids {parsed_group_ids!r} do not match "
                f"the runtime canonical order/ids {list(runtime_saved_parameters)!r}; optimizer moments "
                "cannot be rebound safely."
            )

    saved_parameter_state = state.get("state")
    if not isinstance(saved_parameter_state, Mapping):
        raise ValueError(f"Checkpoint {path}.state must be a mapping.")
    for saved_id, parameter_state in saved_parameter_state.items():
        if isinstance(saved_id, bool) or not isinstance(saved_id, numbers.Integral):
            raise ValueError(f"Checkpoint {path}.state key must be an integer id, got {saved_id!r}.")
        parsed_id = int(saved_id)
        parameter = parameter_by_saved_id.get(parsed_id)
        if parameter is None:
            raise ValueError(
                f"Checkpoint {path}.state contains unknown optimizer parameter id {parsed_id}."
            )
        if not isinstance(parameter_state, Mapping):
            raise ValueError(
                f"Checkpoint {path}.state[{parsed_id}] must be a mapping, got "
                f"{type(parameter_state).__name__}."
            )
        group_index = group_by_saved_id[parsed_id]
        runtime_group = runtime_groups[group_index]
        state_keys = set(parameter_state)
        if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            if state_keys:
                expected_state_keys = {"step", "exp_avg", "exp_avg_sq"}
                if bool(runtime_group.get("amsgrad", False)):
                    expected_state_keys.add("max_exp_avg_sq")
                if state_keys != expected_state_keys:
                    raise ValueError(
                        f"Checkpoint {path}.state[{parsed_id}] Adam state keys are invalid: "
                        f"missing={sorted(expected_state_keys - state_keys)}, "
                        f"extra={sorted(state_keys - expected_state_keys)}."
                    )
                raw_step = parameter_state["step"]
                if isinstance(raw_step, torch.Tensor):
                    if raw_step.numel() != 1 or raw_step.is_complex():
                        raise ValueError(
                            f"Checkpoint {path}.state[{parsed_id}].step must be one real scalar."
                        )
                    raw_step = raw_step.item()
                if isinstance(raw_step, bool) or not isinstance(raw_step, numbers.Real):
                    raise ValueError(
                        f"Checkpoint {path}.state[{parsed_id}].step must be a real scalar, got {raw_step!r}."
                    )
                parsed_step = float(raw_step)
                if not math.isfinite(parsed_step) or parsed_step < 0.0 or not parsed_step.is_integer():
                    raise ValueError(
                        f"Checkpoint {path}.state[{parsed_id}].step must be finite, integral, and "
                        f"non-negative, got {raw_step!r}."
                    )
                for moment_name in expected_state_keys - {"step"}:
                    _validate_parameter_state_tensor(
                        parameter_state[moment_name],
                        parameter=parameter,
                        path=f"{path}.state[{parsed_id}].{moment_name}",
                    )
        elif isinstance(optimizer, torch.optim.SGD):
            expected_state_keys = {"momentum_buffer"} if float(runtime_group.get("momentum", 0.0)) != 0.0 else set()
            if state_keys != expected_state_keys:
                raise ValueError(
                    f"Checkpoint {path}.state[{parsed_id}] SGD state keys are invalid: "
                    f"missing={sorted(expected_state_keys - state_keys)}, "
                    f"extra={sorted(state_keys - expected_state_keys)}."
                )
            if state_keys:
                _validate_parameter_state_tensor(
                    parameter_state["momentum_buffer"],
                    parameter=parameter,
                    path=f"{path}.state[{parsed_id}].momentum_buffer",
                )
        elif state_keys:
            raise ValueError(
                f"Checkpoint {path} uses unsupported optimizer state schema "
                f"{type(optimizer).__module__}.{type(optimizer).__qualname__}; refusing a non-empty "
                "state that cannot be validated fail-closed."
            )
        if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD)):
            _validate_optimizer_tensor_shapes(
                parameter_state,
                parameter=parameter,
                path=f"{path}.state[{parsed_id}]",
            )


def validate_optimizer_state(
    state: Mapping[str, Any],
    *,
    path: str,
    minimum_lr: float,
    maximum_lr: float,
) -> float:
    """Validate optimizer numeric state and every parameter-group LR.

    The returned value is the first group LR, which remains the PPO-level
    adaptive-learning-rate scalar.  Every group is independently checked even
    though the current actor and critic optimizers each use one group.
    """

    param_groups = state.get("param_groups")
    if not isinstance(param_groups, list) or not param_groups:
        raise ValueError(f"Checkpoint {path}.param_groups must be a non-empty list.")

    parsed_lrs: list[float] = []
    for index, group in enumerate(param_groups):
        group_path = f"{path}.param_groups[{index}]"
        if not isinstance(group, Mapping):
            raise ValueError(f"Checkpoint {group_path} must be a mapping.")
        raw_lr = group.get("lr")
        if isinstance(raw_lr, bool) or not isinstance(raw_lr, numbers.Real):
            raise ValueError(f"Checkpoint {group_path}.lr must be a real number, got {raw_lr!r}.")
        lr = float(raw_lr)
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError(f"Checkpoint {group_path}.lr must be finite and > 0, got {raw_lr!r}.")
        if not minimum_lr <= lr <= maximum_lr:
            raise ValueError(
                f"Checkpoint {group_path}.lr={lr} is outside the configured bounds "
                f"[{minimum_lr}, {maximum_lr}]."
            )
        parsed_lrs.append(lr)

    validate_finite_tree(state, path=path)
    return parsed_lrs[0]
