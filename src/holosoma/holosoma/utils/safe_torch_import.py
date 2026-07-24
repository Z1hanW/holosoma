import os


# This module is the first supported gate before IsaacGym/torch.  Establish a
# reproducible cuBLAS workspace before either backend can create a CUDA handle.
_cublas_workspace_config = os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)
if _cublas_workspace_config not in {":4096:8", ":16:8"}:
    raise RuntimeError(
        "CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 before importing CUDA backends; "
        f"got {_cublas_workspace_config!r}."
    )
del _cublas_workspace_config

# Ensure that torch is imported after isaacgym, if isaacgym is installed.
try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

__all__ = ["F", "GradScaler", "TensorDict", "TensorboardSummaryWriter", "autocast", "nn", "optim", "torch"]
