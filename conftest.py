"""Pytest configuration to ensure proper import order for isaacgym compatibility."""

# Import torch safely before any isaacgym imports during test collection
from holosoma.utils.safe_torch_import import torch  # noqa: F401
