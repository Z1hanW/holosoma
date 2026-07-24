"""Versioned checkpoint support for process-global stochastic streams.

Training currently uses the process-global Python, NumPy, torch CPU, and
torch CUDA generators.  Capturing all four is therefore part of a scientific
training resume.  This module deliberately does not claim to checkpoint
simulator state or arbitrary user-created generator objects.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch


RNG_CHECKPOINT_VERSION = 1
ALLOW_NONDETERMINISTIC_RNG_RESUME_ENV = "ALLOW_NONDETERMINISTIC_RNG_RESUME"
_RNG_STATE_KEYS = {
    "version",
    "python_random_state",
    "numpy_random_state",
    "torch_cpu_rng_state",
    "torch_cuda_visible_device_count",
    "torch_cuda_current_device",
    "torch_cuda_rng_state",
}
_EXPECTED_DEVICE_UNSET = object()


def capture_rng_checkpoint_state() -> dict[str, Any]:
    """Return an independent CPU snapshot of every process-global RNG."""

    cuda_device_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    cuda_current_device = int(torch.cuda.current_device()) if cuda_device_count else None
    # One process/rank trains on exactly one selected CUDA device.  Capturing
    # every *visible* device would initialize contexts on peer GPUs under a
    # conventional torchrun launch and can itself exhaust memory.
    cuda_state = (
        torch.cuda.get_rng_state(cuda_current_device).detach().cpu().clone()
        if cuda_current_device is not None
        else None
    )
    numpy_state = np.random.get_state()
    return {
        "version": RNG_CHECKPOINT_VERSION,
        "python_random_state": random.getstate(),
        "numpy_random_state": (
            numpy_state[0],
            # Keep the checkpoint weights-only-loadable.  A raw NumPy array
            # requires arbitrary-pickle mode in PyTorch >=2.6 and would make
            # otherwise safe teacher/policy-init/inference consumers reject
            # every checkpoint written after RNG support was added.
            torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ),
        "torch_cpu_rng_state": torch.get_rng_state().detach().cpu().clone(),
        "torch_cuda_visible_device_count": cuda_device_count,
        "torch_cuda_current_device": cuda_current_device,
        "torch_cuda_rng_state": cuda_state,
    }


def _validate_python_state(value: Any, *, path: str) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"Checkpoint {path} must be a tuple.")
    try:
        validator = random.Random()
        validator.setstate(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Checkpoint {path} is not a valid Python random state.") from exc
    return value


def _validate_numpy_state(value: Any, *, path: str) -> tuple[Any, ...]:
    if not isinstance(value, tuple) or len(value) != 5:
        raise ValueError(f"Checkpoint {path} must be a five-element NumPy RandomState tuple.")
    algorithm, keys, position, has_gauss, cached_gaussian = value
    if algorithm != "MT19937":
        raise ValueError(f"Checkpoint {path} uses unsupported NumPy bit generator {algorithm!r}.")
    if isinstance(keys, np.ndarray):
        if keys.dtype != np.uint32 or keys.shape != (624,):
            raise ValueError(
                f"Checkpoint {path}[1] must be a uint32 NumPy array or CPU int64 tensor "
                f"with shape (624,), got {type(keys).__name__}, dtype={getattr(keys, 'dtype', None)}, "
                f"shape={getattr(keys, 'shape', None)}."
            )
        normalized_keys = torch.from_numpy(keys.astype(np.int64, copy=True))
    elif isinstance(keys, torch.Tensor):
        if keys.device.type != "cpu" or keys.dtype != torch.int64 or keys.shape != (624,):
            raise ValueError(
                f"Checkpoint {path}[1] must be a uint32 NumPy array or CPU int64 tensor "
                f"with shape (624,), got {type(keys).__name__}, device={keys.device}, "
                f"dtype={keys.dtype}, shape={tuple(keys.shape)}."
            )
        normalized_keys = keys.detach().cpu().clone()
        if bool(((normalized_keys < 0) | (normalized_keys > np.iinfo(np.uint32).max)).any().item()):
            raise ValueError(f"Checkpoint {path}[1] contains values outside the uint32 range.")
    else:
        raise ValueError(
            f"Checkpoint {path}[1] must be a uint32 NumPy array or CPU int64 tensor with shape (624,), "
            f"got {type(keys).__name__}, dtype={getattr(keys, 'dtype', None)}, "
            f"shape={getattr(keys, 'shape', None)}."
        )
    if isinstance(position, (bool, np.bool_)) or not isinstance(position, (int, np.integer)):
        raise ValueError(f"Checkpoint {path}[2] must be an integer position.")
    if not 0 <= int(position) <= 624:
        raise ValueError(f"Checkpoint {path}[2] position must be in [0, 624], got {position}.")
    if isinstance(has_gauss, (bool, np.bool_)):
        has_gauss = int(has_gauss)
    if not isinstance(has_gauss, (int, np.integer)) or int(has_gauss) not in (0, 1):
        raise ValueError(f"Checkpoint {path}[3] Gaussian-cache flag must be 0 or 1.")
    if isinstance(cached_gaussian, (bool, np.bool_)) or not isinstance(
        cached_gaussian, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"Checkpoint {path}[4] must be a finite Gaussian-cache scalar.")
    if not math.isfinite(float(cached_gaussian)):
        raise ValueError(f"Checkpoint {path}[4] Gaussian-cache scalar must be finite.")
    normalized = (
        "MT19937",
        normalized_keys,
        int(position),
        int(has_gauss),
        float(cached_gaussian),
    )
    try:
        validator = np.random.RandomState()
        validator.set_state(
            (
                normalized[0],
                normalized_keys.numpy().astype(np.uint32, copy=True),
                normalized[2],
                normalized[3],
                normalized[4],
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Checkpoint {path} is not a valid NumPy random state.") from exc
    return normalized


def _validate_torch_state(value: Any, *, path: str, device: str = "cpu") -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Checkpoint {path} must be a torch tensor.")
    if value.device.type != "cpu" or value.dtype != torch.uint8 or value.ndim != 1 or value.numel() == 0:
        raise ValueError(
            f"Checkpoint {path} must be a non-empty one-dimensional CPU uint8 tensor, "
            f"got device={value.device}, dtype={value.dtype}, shape={tuple(value.shape)}."
        )
    state = value.detach().cpu().clone()
    try:
        torch.Generator(device=device).set_state(state)
    except RuntimeError as exc:
        raise ValueError(f"Checkpoint {path} is not a valid torch RNG state for {device}.") from exc
    return state


def validate_rng_checkpoint_state(
    state: Any,
    *,
    path: str = "rng_state",
    expected_cuda_device_count: int | None = None,
    expected_cuda_device_index: int | None | object = _EXPECTED_DEVICE_UNSET,
    validate_cuda_generators: bool = True,
) -> dict[str, Any]:
    """Validate and clone one rank-local RNG snapshot without global mutation."""

    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {path} must be a mapping.")
    missing = _RNG_STATE_KEYS - set(state)
    extra = set(state) - _RNG_STATE_KEYS
    if missing or extra:
        raise ValueError(
            f"Checkpoint {path} keys are invalid: missing={sorted(missing)}, extra={sorted(extra)}."
        )
    version = state["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != RNG_CHECKPOINT_VERSION:
        raise ValueError(f"Checkpoint {path}.version must be {RNG_CHECKPOINT_VERSION}, got {version!r}.")

    device_count = state["torch_cuda_visible_device_count"]
    if isinstance(device_count, bool) or not isinstance(device_count, int) or device_count < 0:
        raise ValueError(f"Checkpoint {path}.torch_cuda_visible_device_count must be a non-negative integer.")
    if expected_cuda_device_count is not None and device_count != expected_cuda_device_count:
        raise ValueError(
            f"Checkpoint {path} CUDA visibility differs from this process: "
            f"checkpoint={device_count}, runtime={expected_cuda_device_count}."
        )
    cuda_device_index = state["torch_cuda_current_device"]
    if device_count == 0:
        if cuda_device_index is not None:
            raise ValueError(
                f"Checkpoint {path}.torch_cuda_current_device must be None when no CUDA device is visible."
            )
    elif (
        isinstance(cuda_device_index, bool)
        or not isinstance(cuda_device_index, int)
        or not 0 <= cuda_device_index < device_count
    ):
        raise ValueError(
            f"Checkpoint {path}.torch_cuda_current_device must be an integer in "
            f"[0, {device_count}), got {cuda_device_index!r}."
        )
    if expected_cuda_device_index is not _EXPECTED_DEVICE_UNSET and cuda_device_index != expected_cuda_device_index:
        raise ValueError(
            f"Checkpoint {path} current CUDA device differs from this process: "
            f"checkpoint={cuda_device_index}, runtime={expected_cuda_device_index}."
        )

    cuda_state = state["torch_cuda_rng_state"]
    if cuda_device_index is None:
        if cuda_state is not None:
            raise ValueError(
                f"Checkpoint {path}.torch_cuda_rng_state must be None when CUDA is unavailable."
            )
        normalized_cuda_state = None
    else:
        # Structural preflight runs before CUDA initialization.  A CUDA state
        # is longer than a CPU state and therefore cannot be fed to a CPU
        # Generator; perform the tensor checks directly in that mode.
        if not validate_cuda_generators:
            if (
                not isinstance(cuda_state, torch.Tensor)
                or cuda_state.device.type != "cpu"
                or cuda_state.dtype != torch.uint8
                or cuda_state.ndim != 1
                or cuda_state.numel() == 0
            ):
                raise ValueError(
                    f"Checkpoint {path}.torch_cuda_rng_state must be a non-empty "
                    "one-dimensional CPU uint8 tensor."
                )
            normalized_cuda_state = cuda_state.detach().cpu().clone()
        else:
            normalized_cuda_state = _validate_torch_state(
                cuda_state,
                path=f"{path}.torch_cuda_rng_state",
                device=f"cuda:{cuda_device_index}",
            )

    return {
        "version": RNG_CHECKPOINT_VERSION,
        "python_random_state": _validate_python_state(
            state["python_random_state"], path=f"{path}.python_random_state"
        ),
        "numpy_random_state": _validate_numpy_state(
            state["numpy_random_state"], path=f"{path}.numpy_random_state"
        ),
        "torch_cpu_rng_state": _validate_torch_state(
            state["torch_cpu_rng_state"], path=f"{path}.torch_cpu_rng_state"
        ),
        "torch_cuda_visible_device_count": device_count,
        "torch_cuda_current_device": cuda_device_index,
        "torch_cuda_rng_state": normalized_cuda_state,
    }


def restore_rng_checkpoint_state(state: Any, *, path: str = "rng_state") -> None:
    """Atomically restore all process-global RNGs from a validated snapshot."""

    runtime_cuda_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    runtime_cuda_device = int(torch.cuda.current_device()) if runtime_cuda_count else None
    normalized = validate_rng_checkpoint_state(
        state,
        path=path,
        expected_cuda_device_count=runtime_cuda_count,
        expected_cuda_device_index=runtime_cuda_device,
        validate_cuda_generators=True,
    )
    rollback = capture_rng_checkpoint_state()
    try:
        random.setstate(normalized["python_random_state"])
        numpy_state = normalized["numpy_random_state"]
        np.random.set_state(
            (
                numpy_state[0],
                numpy_state[1].numpy().astype(np.uint32, copy=True),
                numpy_state[2],
                numpy_state[3],
                numpy_state[4],
            )
        )
        torch.set_rng_state(normalized["torch_cpu_rng_state"])
        if runtime_cuda_device is not None:
            torch.cuda.set_rng_state(normalized["torch_cuda_rng_state"], device=runtime_cuda_device)
    except Exception:
        # Validation above makes this exceptional, but a driver/runtime error
        # must not leave a half-restored process-global state.
        random.setstate(rollback["python_random_state"])
        rollback_numpy_state = rollback["numpy_random_state"]
        np.random.set_state(
            (
                rollback_numpy_state[0],
                rollback_numpy_state[1].numpy().astype(np.uint32, copy=True),
                rollback_numpy_state[2],
                rollback_numpy_state[3],
                rollback_numpy_state[4],
            )
        )
        torch.set_rng_state(rollback["torch_cpu_rng_state"])
        if runtime_cuda_device is not None:
            torch.cuda.set_rng_state(rollback["torch_cuda_rng_state"], device=runtime_cuda_device)
        raise
