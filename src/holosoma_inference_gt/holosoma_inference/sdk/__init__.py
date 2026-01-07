"""
Robot communication package.

This package provides unified interfaces for robot state processing and command sending
across different robot platforms (Unitree, Booster, etc.).
"""

from .command_sender import create_command_sender
from .interface_wrapper import InterfaceWrapper
from .state_processor import create_state_processor

__all__ = [
    "InterfaceWrapper",
    "create_command_sender",
    "create_state_processor",
]
