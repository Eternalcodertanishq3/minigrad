"""
module.py — Base class for all neural network modules.

Every layer in miniGrad.nn inherits from Module. It provides:
- Parameter collection (recursively finds all Tensors with requires_grad=True)
- Gradient zeroing
- Train/eval mode switching
- Forward pass hook via __call__
"""
from __future__ import annotations

from typing import List, Iterator

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
