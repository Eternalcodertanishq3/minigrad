"""
schedulers.py — Learning rate schedulers.

Adjust the learning rate during training to improve convergence.
All schedulers follow PyTorch's interface: call scheduler.step() after
each epoch (or step, depending on the scheduler).
"""
from __future__ import annotations

import math
from typing import Optional

from minigrad.optim.base import Optimizer


class _LRScheduler:
    """Base class for all learning rate schedulers."""

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = last_epoch
        self._step_count = 0
        # Initialize
        self.step()

    def get_lr(self) -> float:
        """Compute the learning rate. Must be implemented by subclasses."""
        raise NotImplementedError

    def step(self, epoch: Optional[int] = None) -> None:
        """Update the learning rate."""
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        self._step_count += 1
        self.optimizer.lr = self.get_lr()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(base_lr={self.base_lr})"


class StepLR(_LRScheduler):
    """
    Decays the learning rate by gamma every step_size epochs.

    Args:
        optimizer:  Wrapped optimizer
        step_size:  Period of learning rate decay (in epochs)
        gamma:      Multiplicative factor of learning rate decay (default: 0.1)
    """

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1,
                 last_epoch: int = -1) -> None:
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.base_lr * (self.gamma ** (self.last_epoch // self.step_size))


class CosineAnnealingLR(_LRScheduler):
    """
    Cosine annealing learning rate schedule.

    Decreases the learning rate following a cosine curve from base_lr to eta_min
    over T_max epochs.

    Args:
        optimizer: Wrapped optimizer
        T_max:     Maximum number of epochs
        eta_min:   Minimum learning rate (default: 0)
    """

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0,
                 last_epoch: int = -1) -> None:
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.eta_min + (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * self.last_epoch / self.T_max)
        ) / 2


class ExponentialLR(_LRScheduler):
    """
    Decays the learning rate by gamma every epoch.

    Args:
        optimizer: Wrapped optimizer
        gamma:     Multiplicative factor of learning rate decay
    """

    def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int = -1) -> None:
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.base_lr * (self.gamma ** self.last_epoch)
