"""
module.py — Base class for all neural network modules.

Every layer in miniGrad.nn inherits from Module. It provides:
- Parameter collection (recursively finds all Tensors with requires_grad=True)
- Gradient zeroing
- Train/eval mode switching
- Forward pass hook via __call__
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

import numpy as np

from minigrad.tensor import Tensor


class Module:
    """
    Base class for all neural network modules.

    Subclasses must implement forward().
    Parameters are automatically collected from attributes.
    """

    training: bool = True

    def __call__(self, *args, **kwargs):
        """Delegate to forward() — enables model(x) syntax."""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Define the forward pass. Must be implemented by subclasses."""
        raise NotImplementedError(
            f"Module [{type(self).__name__}] is missing the required `forward` function"
        )

    def parameters(self) -> List[Tensor]:
        """
        Recursively collect all Tensors with requires_grad=True.
        This is what the optimizer uses to know what to update.
        """
        params: List[Tensor] = []
        for attr in vars(self).values():
            if isinstance(attr, Tensor) and attr.requires_grad:
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
        return params

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        """Yield (name, Tensor) pairs for all trainable parameters."""
        for name, attr in vars(self).items():
            if name.startswith("_"):
                continue
            key = prefix + ("." if prefix else "") + name
            if isinstance(attr, Tensor) and attr.requires_grad:
                yield key, attr
            elif isinstance(attr, Module):
                yield from attr.named_parameters(key)
            elif isinstance(attr, (list, tuple)):
                for i, item in enumerate(attr):
                    item_key = f"{key}.{i}"
                    if isinstance(item, Module):
                        yield from item.named_parameters(item_key)
                    elif isinstance(item, Tensor) and item.requires_grad:
                        yield item_key, item

    def state_dict(self) -> dict[str, np.ndarray]:
        """
        Return a copy of learnable parameters and public NumPy buffers.

        BatchNorm running statistics are included as buffers. Private arrays
        such as Dropout masks are intentionally skipped.
        """
        state: dict[str, np.ndarray] = {}

        def collect(module: Module, prefix: str = "") -> None:
            for name, attr in vars(module).items():
                if name.startswith("_"):
                    continue
                key = prefix + ("." if prefix else "") + name
                if isinstance(attr, Tensor):
                    state[key] = attr.data.copy()
                elif isinstance(attr, np.ndarray):
                    state[key] = attr.copy()
                elif isinstance(attr, Module):
                    collect(attr, key)
                elif isinstance(attr, (list, tuple)):
                    for i, item in enumerate(attr):
                        item_key = f"{key}.{i}"
                        if isinstance(item, Module):
                            collect(item, item_key)
                        elif isinstance(item, Tensor):
                            state[item_key] = item.data.copy()

        collect(self)
        return state

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        """
        Load parameters and buffers from a state dictionary.

        Args:
            state: Mapping produced by state_dict().
            strict: If True, require exact key and shape matches.
        """
        remaining = set(state)
        missing: list[str] = []

        def load(module: Module, prefix: str = "") -> None:
            for name, attr in vars(module).items():
                if name.startswith("_"):
                    continue
                key = prefix + ("." if prefix else "") + name
                if isinstance(attr, Tensor):
                    if key not in state:
                        missing.append(key)
                        continue
                    value = np.array(state[key], dtype=np.float64)
                    if strict and value.shape != attr.data.shape:
                        raise ValueError(f"shape mismatch for {key}: expected {attr.data.shape}, got {value.shape}")
                    attr.data = value.reshape(attr.data.shape)
                    attr.grad = np.zeros_like(attr.data)
                    remaining.discard(key)
                elif isinstance(attr, np.ndarray):
                    if key not in state:
                        missing.append(key)
                        continue
                    value = np.array(state[key], dtype=np.float64)
                    if strict and value.shape != attr.shape:
                        raise ValueError(f"shape mismatch for {key}: expected {attr.shape}, got {value.shape}")
                    setattr(module, name, value.reshape(attr.shape))
                    remaining.discard(key)
                elif isinstance(attr, Module):
                    load(attr, key)
                elif isinstance(attr, (list, tuple)):
                    for i, item in enumerate(attr):
                        item_key = f"{key}.{i}"
                        if isinstance(item, Module):
                            load(item, item_key)
                        elif isinstance(item, Tensor):
                            if item_key not in state:
                                missing.append(item_key)
                                continue
                            value = np.array(state[item_key], dtype=np.float64)
                            if strict and value.shape != item.data.shape:
                                raise ValueError(
                                    f"shape mismatch for {item_key}: expected {item.data.shape}, got {value.shape}"
                                )
                            item.data = value.reshape(item.data.shape)
                            item.grad = np.zeros_like(item.data)
                            remaining.discard(item_key)

        load(self)

        if strict and (missing or remaining):
            problems = []
            if missing:
                problems.append(f"missing keys: {sorted(missing)}")
            if remaining:
                problems.append(f"unexpected keys: {sorted(remaining)}")
            raise KeyError("; ".join(problems))

    def save(self, path: str | Path) -> None:
        """Save state_dict() to a compressed .npz file."""
        np.savez_compressed(path, **self.state_dict())

    def load(self, path: str | Path, strict: bool = True) -> None:
        """Load a state dictionary from a .npz file."""
        with np.load(path) as data:
            self.load_state_dict({key: data[key] for key in data.files}, strict=strict)

    def zero_grad(self) -> None:
        """Set gradients of all parameters to zero. Call before loss.backward()."""
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode: bool = True) -> Module:
        """Set training mode. Affects Dropout, BatchNorm, etc."""
        self.training = mode
        for attr in vars(self).values():
            if isinstance(attr, Module):
                attr.train(mode)
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Module):
                        item.train(mode)
        return self

    def eval(self) -> Module:
        """Set evaluation mode. Equivalent to train(False)."""
        return self.train(False)

    def named_modules(self, prefix: str = "") -> Iterator[tuple]:
        """Yield (name, module) pairs for all submodules."""
        yield prefix, self
        for name, attr in vars(self).items():
            if isinstance(attr, Module):
                submodule_prefix = prefix + ("." if prefix else "") + name
                yield from attr.named_modules(submodule_prefix)
            elif isinstance(attr, (list, tuple)):
                for i, item in enumerate(attr):
                    if isinstance(item, Module):
                        submodule_prefix = prefix + ("." if prefix else "") + f"{name}[{i}]"
                        yield from item.named_modules(submodule_prefix)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __setattr__(self, name: str, value) -> None:
        # Track submodules for train/eval propagation
        super().__setattr__(name, value)
