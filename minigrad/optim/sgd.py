"""
sgd.py — Stochastic Gradient Descent optimizer.

Supports:
- Vanilla SGD:          w = w - lr * grad
- Momentum:             v = beta * v + grad; w = w - lr * v
- Nesterov momentum:    v = beta * v + grad; w = w - lr * (beta * v + grad)
- Weight decay (L2):    grad = grad + weight_decay * w

Reference: "On the momentum term in gradient descent learning algorithms"
(Qian, 1999) for momentum theory.
"""
from __future__ import annotations

import numpy as np
from typing import List

from minigrad.tensor import Tensor
from minigrad.optim.base import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Args:
        params:       Parameters to optimize
        lr:           Learning rate (default: 0.01)
        momentum:     Momentum factor (default: 0)
        weight_decay: L2 regularization coefficient (default: 0)
        nesterov:     Use Nesterov momentum (default: False)
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a positive momentum factor")

        # Velocity buffers — one per parameter
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        """Perform a single SGD update step."""
        for i, p in enumerate(self.params):
            grad = p.grad.copy()

            # L2 weight decay: grad = grad + weight_decay * w
            if self.weight_decay > 0:
                grad += self.weight_decay * p.data

            if self.momentum > 0:
                # Update velocity: v = beta * v + grad
                self.velocities[i] = self.momentum * self.velocities[i] + grad

                if self.nesterov:
                    # Nesterov: apply step using lookahead gradient
                    # w = w - lr * (beta * v + grad) where v is the updated velocity
                    p.data -= self.lr * (self.momentum * self.velocities[i] + grad)
                else:
                    # Standard momentum: w = w - lr * v
                    p.data -= self.lr * self.velocities[i]
            else:
                # Vanilla SGD
                p.data -= self.lr * grad

    def __repr__(self) -> str:
        return (
            f"SGD(lr={self.lr}, momentum={self.momentum}, "
            f"weight_decay={self.weight_decay}, nesterov={self.nesterov})"
        )
