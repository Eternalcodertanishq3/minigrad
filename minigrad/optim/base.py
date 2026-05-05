"""
base.py — Base optimizer class.

All optimizers inherit from Optimizer and implement step().
This base class handles parameter collection and zero_grad().
"""
from __future__ import annotations

from typing import List

from minigrad.tensor import Tensor


class Optimizer:
    """
    Base class for all optimizers.

    Args:
        params: List of parameters to optimize
    """

    def __init__(self, params: List[Tensor]) -> None:
        self.params = [p for p in params if p.requires_grad]

    def zero_grad(self) -> None:
        """Zero the gradients of all parameters."""
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        """Perform a single optimization step. Must be implemented by subclasses."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_params={len(self.params)})"
