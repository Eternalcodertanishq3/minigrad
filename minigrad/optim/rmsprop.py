"""
rmsprop.py — RMSProp optimizer.

Maintains a moving average of squared gradients per parameter.
Divides the learning rate by sqrt(v) to normalize updates.
This adapts the learning rate per-parameter — rare features get larger updates.

Reference: "Lecture 6.5 - RMSProp" (Tieleman & Hinton, 2012)
COURSERA: Neural Networks for Machine Learning
"""
from __future__ import annotations

import numpy as np
from typing import List

from minigrad.tensor import Tensor
from minigrad.optim.base import Optimizer


class RMSprop(Optimizer):
    """
    RMSProp optimizer.

    Args:
        params:       Parameters to optimize
        lr:           Learning rate (default: 0.001)
        alpha:        Smoothing constant for squared gradient average (default: 0.99)
        eps:          Term for numerical stability (default: 1e-8)
        weight_decay: L2 regularization (default: 0)
        momentum:     Momentum factor (default: 0)
        centered:     If True, compute centered RMSProp (default: False)
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.001,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered

        # Squared gradient average (velocity)
        self.v = [np.zeros_like(p.data) for p in self.params]

        # Momentum buffers
        if momentum > 0:
            self.buffers = [np.zeros_like(p.data) for p in self.params]

        # Gradient average for centered RMSProp
        if centered:
            self.grad_avg = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        """Perform a single RMSProp update step."""
        for i, p in enumerate(self.params):
            grad = p.grad.copy()

            # L2 weight decay
            if self.weight_decay > 0:
                grad += self.weight_decay * p.data

            # Update squared gradient average: v = alpha * v + (1-alpha) * grad^2
            self.v[i] = self.alpha * self.v[i] + (1.0 - self.alpha) * (grad ** 2)

            denom = np.sqrt(self.v[i] + self.eps)

            if self.centered:
                # Update gradient average
                self.grad_avg[i] = self.alpha * self.grad_avg[i] + (1.0 - self.alpha) * grad
                denom = np.sqrt(self.v[i] - self.grad_avg[i] ** 2 + self.eps)

            if self.momentum > 0:
                # Apply momentum to the update direction
                self.buffers[i] = self.momentum * self.buffers[i] + grad / denom
                p.data -= self.lr * self.buffers[i]
            else:
                # Standard update
                p.data -= self.lr * grad / denom

    def __repr__(self) -> str:
        return (
            f"RMSprop(lr={self.lr}, alpha={self.alpha}, eps={self.eps}, "
            f"weight_decay={self.weight_decay}, momentum={self.momentum}, centered={self.centered})"
        )
