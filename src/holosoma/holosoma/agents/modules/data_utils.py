from __future__ import annotations

import os

import torch
from torch import Tensor


class RolloutStorage:
    """Simple buffer for storing rollout data during training.

    This is a lightweight storage for PPO rollout data. It stores transitions in tensors
    and provides methods for adding data and generating mini-batches.
    """

    def __init__(self, num_envs: int, num_transitions_per_env: int, device: str = "cpu"):
        """Initialize the rollout storage.

        Args:
            num_envs: Number of parallel environments
            num_transitions_per_env: Number of transitions to store per environment
            device: Device to store tensors on
        """
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0

        # Dictionary to store all data buffers
        self._buffers: dict[str, Tensor] = {}
        # Transition fields are written once per environment step through
        # ``add``.  Derived fields (for example GAE returns/advantages) are
        # populated as complete tensors after collection through ``__setitem__``.
        # Keeping those lifecycles explicit prevents a missing write from
        # silently reusing rows left by the preceding rollout.
        self._required_on_add: set[str] = set()
        self._derived_keys: set[str] = set()
        self._written_derived_keys: set[str] = set()
        self._schema_frozen = False

    @property
    def registered_keys(self) -> frozenset[str]:
        """Return the immutable set of fields in this storage schema."""

        return frozenset(self._buffers)

    @property
    def required_on_add_keys(self) -> frozenset[str]:
        """Return fields that every collected transition must provide."""

        return frozenset(self._required_on_add)

    @property
    def derived_keys(self) -> frozenset[str]:
        """Return fields written as complete tensors after collection."""

        return frozenset(self._derived_keys)

    def register(
        self,
        key: str,
        shape: tuple[int, ...] | list[int] = (),
        dtype: torch.dtype = torch.float,
        *,
        required_on_add: bool = True,
    ):
        """Register a new data key to store in the buffer.

        Args:
            key: Name of the data field (e.g., "obs", "actions", "rewards")
            shape: Shape of each data element (excluding batch dimensions)
            dtype: Data type of the tensor
            required_on_add: Whether every call to :meth:`add` must provide
                this transition field. Set this to ``False`` only for a field
                written as one complete tensor through :meth:`__setitem__`
                after rollout collection.
        """
        if self.step != 0 or self._schema_frozen:
            raise RuntimeError(
                "Rollout fields must be registered before transition collection starts: "
                f"current_step={self.step}, schema_frozen={self._schema_frozen}."
            )
        if key in self._buffers:
            raise ValueError(f"Key '{key}' already registered")

        if not isinstance(shape, (list, tuple)):
            raise ValueError("shape must be a list or tuple")

        # Create buffer with shape: [num_transitions_per_env, num_envs, *shape]
        buffer = torch.zeros((self.num_transitions_per_env, self.num_envs, *shape), dtype=dtype, device=self.device)
        self._buffers[key] = buffer
        if required_on_add:
            self._required_on_add.add(key)
        else:
            self._derived_keys.add(key)

    def add(self, **data: Tensor):
        """Add a transition to the buffer.

        Args:
            **data: Keyword arguments where keys are buffer names and values are tensors
                   of shape [num_envs, ...] to store at the current step

        Example:
            storage.add(obs=obs, actions=actions, rewards=rewards, dones=dones)
        """
        if self.step >= self.num_transitions_per_env:
            raise RuntimeError(f"Buffer overflow: step {self.step} >= {self.num_transitions_per_env}")

        provided_keys = set(data)
        unknown_keys = provided_keys.difference(self._buffers)
        if unknown_keys:
            raise KeyError(f"Cannot add unregistered rollout fields: {sorted(unknown_keys)}")
        derived_keys = provided_keys.intersection(self._derived_keys)
        if derived_keys:
            raise ValueError(
                "Derived rollout fields must be written as complete tensors after collection, not through add: "
                f"{sorted(derived_keys)}"
            )
        missing_keys = self._required_on_add.difference(provided_keys)
        if missing_keys:
            raise KeyError(
                f"Transition at step {self.step} is missing required rollout fields: {sorted(missing_keys)}"
            )

        prepared: dict[str, Tensor] = {}
        for key, value in data.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"Rollout field '{key}' must be a torch.Tensor, got {type(value).__name__}.")

            if value.requires_grad:
                raise ValueError(f"Cannot store tensor with requires_grad=True for key '{key}'")

            if hasattr(value, "is_inference") and value.is_inference():
                # Convert inference tensors to normal tensors for training-time autograd.
                with torch.inference_mode(False):
                    value = value.clone()

            destination = self._buffers[key][self.step]
            if value.shape != destination.shape:
                raise ValueError(
                    f"Rollout field '{key}' has shape {tuple(value.shape)}, expected {tuple(destination.shape)}."
                )
            if value.dtype != destination.dtype:
                raise ValueError(
                    f"Rollout field '{key}' has dtype {value.dtype}, expected {destination.dtype}."
                )
            if value.device != destination.device:
                raise ValueError(
                    f"Rollout field '{key}' is on {value.device}, expected {destination.device}."
                )
            prepared[key] = value

        # Validate the complete transition before mutating any buffer so a
        # failed add cannot leave a partially written row behind.
        for key, value in prepared.items():
            self._buffers[key][self.step].copy_(value)

        self._schema_frozen = True
        self.step += 1

    def __getitem__(self, key: str) -> Tensor:
        """Get the buffer for a specific key.

        Args:
            key: Name of the buffer

        Returns:
            Tensor of shape [num_transitions_per_env, num_envs, ...]
        """
        if key not in self._buffers:
            raise KeyError(f"Key '{key}' not registered")
        return self._buffers[key]

    def __setitem__(self, key: str, value: Tensor):
        """Set the entire buffer for a specific key.

        Args:
            key: Name of the buffer
            value: Tensor of shape [num_transitions_per_env, num_envs, ...]
        """
        if key not in self._buffers:
            raise KeyError(f"Key '{key}' not registered")

        if key in self._derived_keys and self.step != self.num_transitions_per_env:
            raise RuntimeError(
                "A derived rollout field can only be written after collection is complete: "
                f"key={key!r}, step={self.step}, capacity={self.num_transitions_per_env}."
            )
        if not isinstance(value, Tensor):
            raise TypeError(f"Rollout buffer '{key}' must be a torch.Tensor, got {type(value).__name__}.")
        if value.requires_grad:
            raise ValueError("Cannot store tensor with requires_grad=True")

        if hasattr(value, "is_inference") and value.is_inference():
            # Convert inference tensors to normal tensors for training-time autograd.
            with torch.inference_mode(False):
                value = value.clone()

        destination = self._buffers[key]
        if value.shape != destination.shape:
            raise ValueError(
                f"Rollout buffer '{key}' has shape {tuple(value.shape)}, expected {tuple(destination.shape)}."
            )
        if value.dtype != destination.dtype:
            raise ValueError(
                f"Rollout buffer '{key}' has dtype {value.dtype}, expected {destination.dtype}."
            )
        if value.device != destination.device:
            raise ValueError(
                f"Rollout buffer '{key}' is on {value.device}, expected {destination.device}."
            )
        self._buffers[key].copy_(value)
        if key in self._derived_keys:
            self._written_derived_keys.add(key)

    def clear(self):
        """Clear the buffer and reset the step counter."""
        self.step = 0
        # Derived tensors are not overwritten by ``add``. Clear their bytes as
        # well as their readiness markers so stale returns/advantages can never
        # be observed after a new rollout starts.
        for key in self._derived_keys:
            self._buffers[key].zero_()
        self._written_derived_keys.clear()

    def _validate_ready_for_minibatches(self) -> None:
        if self.step != self.num_transitions_per_env:
            raise RuntimeError(
                "Rollout storage must be completely filled before minibatch generation: "
                f"step={self.step}, capacity={self.num_transitions_per_env}."
            )
        missing_derived = self._derived_keys.difference(self._written_derived_keys)
        if missing_derived:
            raise RuntimeError(
                "Rollout-derived fields were not populated before minibatch generation: "
                f"{sorted(missing_derived)}"
            )

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8, keys: set[str] | None = None):
        """Generate randomized mini-batches for training.

        This flattens the time and environment dimensions and creates random mini-batches.

        Args:
            num_mini_batches: Number of mini-batches to create per epoch
            num_epochs: Number of times to iterate over the data

        Yields:
            Dictionary mapping buffer keys to mini-batch tensors
        """
        self._validate_ready_for_minibatches()
        batch_size = self.num_envs * self.num_transitions_per_env
        if num_mini_batches <= 0:
            raise ValueError("num_mini_batches must be positive.")
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")
        if batch_size % num_mini_batches != 0:
            raise ValueError(
                "Rollout batch size must be divisible by num_mini_batches so no samples are dropped: "
                f"batch_size={batch_size}, num_mini_batches={num_mini_batches}."
            )
        mini_batch_size = batch_size // num_mini_batches

        selected_keys = list(keys) if keys is not None else list(self._buffers)
        if not selected_keys:
            raise ValueError("At least one rollout field must be selected for minibatch generation.")
        unknown_keys = set(selected_keys).difference(self._buffers)
        if unknown_keys:
            raise KeyError(f"Requested unregistered rollout fields: {sorted(unknown_keys)}")
        use_contiguous = os.environ.get("HOLOSOMA_CONTIGUOUS_MINIBATCHES", "").lower() not in (
            "",
            "0",
            "false",
            "no",
        )

        # Flatten all buffers: [num_transitions_per_env, num_envs, ...] -> [batch_size, ...]
        flattened = {key: self._buffers[key].flatten(0, 1) for key in selected_keys}

        for _ in range(num_epochs):
            if not use_contiguous:
                # Each optimization epoch must see a fresh ordering.  Reusing
                # one permutation silently couples minibatches across epochs.
                indices = torch.randperm(batch_size, requires_grad=False, device=self.device)
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                if use_contiguous:
                    mini_batch = {key: flattened[key][start:end] for key in selected_keys}
                else:
                    batch_indices = indices[start:end]
                    mini_batch = {key: flattened[key][batch_indices] for key in selected_keys}
                yield mini_batch

    def sequence_mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 1):
        """Generate sequential mini-batches for recurrent training.

        This keeps the time dimension intact and batches across environments.

        Yields:
            Dictionary mapping buffer keys to tensors of shape [T, B, ...]
        """
        self._validate_ready_for_minibatches()
        if num_mini_batches <= 0:
            raise ValueError("num_mini_batches must be positive.")
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")
        if self.num_envs % num_mini_batches != 0:
            raise ValueError("num_envs must be divisible by num_mini_batches for sequence batching.")

        envs_per_batch = self.num_envs // num_mini_batches

        for _ in range(num_epochs):
            env_indices = torch.randperm(self.num_envs, requires_grad=False, device=self.device)
            for i in range(num_mini_batches):
                env_ids = env_indices[i * envs_per_batch : (i + 1) * envs_per_batch]
                mini_batch = {key: buf[:, env_ids] for key, buf in self._buffers.items()}
                yield mini_batch
