# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from Isaac Lab v2.0.0 (https://github.com/isaac-sim/IsaacLab)
# Contributors: https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md

"""Torch utility functions for tensor operations and RL-specific helpers.

This module contains utility functions for:
- Tensor conversion and random number generation
- RL trajectory processing (split, pad, unpad)
- Custom batched quaternion operations for [N, M, 3] shapes

Note:
    For general rotation and quaternion math, use holosoma.isaac_utils.rotations instead.
    This module only contains utilities and RL-specific helpers.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import numpy.typing as npt

from holosoma.utils.safe_torch_import import torch
from holosoma.utils.torch_jit import torch_jit_script

# ============================================================================
# Math Utilities (from maths.py)
# ============================================================================


@torch_jit_script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalize a tensor along the last dimension.

    Args:
        x: Input tensor to normalize.
        eps: Small epsilon value to prevent division by zero.

    Returns:
        Normalized tensor.
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch_jit_script
def copysign(a: float, b: torch.Tensor) -> torch.Tensor:
    """Copy the sign of tensor b to scalar a.

    Args:
        a: Scalar value.
        b: Tensor whose signs to copy.

    Returns:
        Tensor with magnitude of a and signs of b.
    """
    a_tensor = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a_tensor) * torch.sign(b)


def set_seed(seed: int, torch_deterministic: bool = False) -> int:
    """Set random seed across all modules for reproducibility.

    Args:
        seed: Random seed value. If -1, generates random seed.
        torch_deterministic: If True, enables deterministic operations.

    Returns:
        The seed that was set.
    """
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print(f"Setting seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


# ============================================================================
# Tensor Utilities
# ============================================================================


def to_torch(
    x: npt.NDArray[np.float64] | list[Any] | tuple[Any, ...],
    dtype: torch.dtype = torch.float,
    device: str | torch.device = "cuda:0",
    requires_grad: bool = False,
) -> torch.Tensor:
    """Convert input to torch tensor.

    Parameters
    ----------
    x : Union[npt.NDArray[np.float64], list[Any], tuple[Any, ...]]
        Input data to convert to tensor. Can be numpy array, list, or tuple.
    dtype : torch.dtype, optional
        Desired data type of the tensor, by default torch.float
    device : Union[str, torch.device], optional
        Device to place the tensor on, by default "cuda:0"
    requires_grad : bool, optional
        Whether to track gradients, by default False

    Returns
    -------
    torch.Tensor
        Converted tensor with specified dtype and device
    """
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


@torch_jit_script
def torch_rand_float(lower: float, upper: float, shape: tuple[int, int], device: str) -> torch.Tensor:
    """Generate random float tensor.

    Parameters
    ----------
    lower : float
        Lower bound
    upper : float
        Upper bound
    shape : tuple[int, int] | torch.Size
        Shape of output tensor. Can be a tuple or torch.Size object.
    device : str
        Device to place tensor on

    Returns
    -------
    torch.Tensor
        Random tensor of specified shape
    """
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def get_axis_params(value: float, axis_idx: int, dtype: npt.DTypeLike = np.float64, n_dims: int = 3) -> list[float]:
    """Construct arguments to `Vec` according to axis index.

    Parameters
    ----------
    value : float
        Value to set at axis_idx
    axis_idx : int
        Index of axis to set value
    dtype : npt.DTypeLike, optional
        Output dtype, by default np.float64
    n_dims : int, optional
        Number of dimensions, by default 3

    Returns
    -------
    list[float]
        list of parameters with specified values
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.0
    params = np.where(zs == 1.0, value, zs)
    return list(params.astype(dtype))


# ============================================================================
# RL Trajectory Processing
# ============================================================================


def split_and_pad_trajectories(tensor: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split trajectories at done indices and pad to longest trajectory length.

    This is used in RL training to process variable-length trajectories
    by splitting them at episode boundaries and padding for batch processing.

    Args:
        tensor: Input tensor of shape (time, num_envs, ...).
        dones: Done flags of shape (time, num_envs).

    Returns:
        A tuple containing:
            - Padded trajectories of shape (max_length, num_trajectories, ...).
            - Masks of shape (max_length, num_trajectories) indicating valid entries.

    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
               ]

        Output: [ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],    |    [True, True, False, False],
                 [b1, b2, 0, 0],    |    [True, True, False, False],
                 [b3, b4, b5, 0],   |    [True, True, True, False],
                 [b6, 0, 0, 0]      |    [True, False, False, False],
                ]                   | ]

    Note:
        Assumes input has dimension order: [time, num_envs, additional_dimensions].
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    # add at least one full length trajectory
    trajectories = trajectories + (torch.zeros(tensor.shape[0], *tensor.shape[2:], device=tensor.device),)
    # pad the trajectories to the length of the longest trajectory
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)  # type: ignore[arg-type]
    # remove the added tensor
    padded_trajectories = padded_trajectories[:, :-1]

    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Remove padding from trajectories using masks.

    This is the inverse operation of split_and_pad_trajectories, used to
    reconstruct the original trajectory structure after processing.

    Args:
        trajectories: Padded trajectories of shape (max_length, num_trajectories, ...).
        masks: Masks of shape (max_length, num_trajectories) indicating valid entries.

    Returns:
        Unpadded trajectories, of shape (length, num_trajectories, ...).

    Example:
        Input: [ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],    |    [True, True, False, False],
                 [b1, b2, 0, 0],    |    [True, True, False, False],
                 [b3, b4, b5, 0],   |    [True, True, True, False],
                 [b6, 0, 0, 0]      |    [True, False, False, False],
                ]                   | ]

        Output: [ [a1, a2, a3, a4 | a5, a6],
                  [b1, b2 | b3, b4, b5 | b6]
                ]
    """
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def batched_index_select(batch_input: torch.Tensor, dim: int, batch_index: torch.Tensor) -> torch.Tensor:
    """Select values from `batch_input` using batched `batch_index` along dimension `dim`.

    This is useful for efficiently selecting different indices per batch element.

    Args:
        batch_input: A tensor of shape [B, ..., D, ...].
        dim: The dimension along which to index.
        batch_index: A LongTensor of shape [B, ..., K] with indices to select.

    Returns:
        A tensor with selected values of shape [B, ..., K, ...].
    """
    views = [batch_input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(batch_input.shape))]
    expanse = list(batch_input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = batch_index.view(views).expand(expanse)
    return torch.gather(batch_input, dim, index)
