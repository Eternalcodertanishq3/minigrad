"""
adam.py — Adam and AdamW optimizers.

Adam (Adaptive Moment Estimation) combines momentum and adaptive learning rates.
It maintains per-parameter estimates of the first moment (mean gradient)
and second moment (uncentered variance of gradient).

AdamW decouples weight decay from the gradient update, which improves
generalization compared to standard Adam with L2 regularization.

Reference: "Adam: A Method for Stochastic Optimization"
(Kingma & Ba, 2014) — https://arxiv.org/abs/1412.6980

"Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017)
— https://arxiv.org/abs/1711.05101 for AdamW
"""
from __future__ import annotations

import numpy as np
from typing import List

from minigrad.tensor import Tensor
from minigrad.optim.base import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer.

    Args:
        params:       Parameters to optimize
        lr:           Learning rate (default: 1e-3)
        betas:        Coefficients for running averages (default: (0.9, 0.999))
        eps:          Term for numerical stability (default: 1e-8)
        weight_decay: L2 regularization (default: 0)

    Update rule:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)   # bias correction
        v_hat = v_t / (1 - beta2^t)   # bias correction
        w = w - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0  # step counter

        # First moment estimates (momentum)
        self.m = [np.zeros_like(p.data) for p in self.params]
        # Second moment estimates (velocity / adaptive learning rate)
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        """Perform a single Adam update step."""
        self.t += 1
        b1, b2 = self.beta1, self.beta2

        for i, p in enumerate(self.params):
            grad = p.grad.copy()

            # L2 weight decay (applied in gradient, not in weight update)
            if self.weight_decay > 0:
                grad += self.weight_decay * p.data

            # Update biased first moment estimate: m = beta1 * m + (1-beta1) * grad
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * grad

            # Update biased second raw moment estimate: v = beta2 * v + (1-beta2) * grad^2
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * (grad ** 2)

            # Bias correction — critical for early training steps
            # Without this, m and v are biased toward zero initially
            m_hat = self.m[i] / (1.0 - b1 ** self.t)
            v_hat = self.v[i] / (1.0 - b2 ** self.t)

            # Parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def __repr__(self) -> str:
        return (
            f"Adam(lr={self.lr}, betas=({self.beta1}, {self.beta2}), "
            f"eps={self.eps}, weight_decay={self.weight_decay})"
        )


class AdamW(Optimizer):
    """
    AdamW optimizer.

    Adam with decoupled weight decay. Weight decay is applied directly to
    the weights rather than being added to the gradient. This provides
    better regularization and generalization than Adam with L2 penalty.

    Reference: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017)

    Args:
        params:       Parameters to optimize
        lr:           Learning rate (default: 1e-3)
        betas:        Coefficients for running averages (default: (0.9, 0.999))
        eps:          Term for numerical stability (default: 1e-8)
        weight_decay: Decoupled weight decay (default: 0)
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        """Perform a single AdamW update step."""
        self.t += 1
        b1, b2 = self.beta1, self.beta2

        for i, p in enumerate(self.params):
            grad = p.grad.copy()

            # Decoupled weight decay: apply directly to weights
            if self.weight_decay > 0:
                p.data *= (1.0 - self.lr * self.weight_decay)

            # Update moment estimates (same as Adam)
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * grad
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * (grad ** 2)

            # Bias correction
            m_hat = self.m[i] / (1.0 - b1 ** self.t)
            v_hat = self.v[i] / (1.0 - b2 ** self.t)

            # Parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def __repr__(self) -> str:
        return (
            f"AdamW(lr={self.lr}, betas=({self.beta1}, {self.beta2}), "
            f"eps={self.eps}, weight_decay={self.weight_decay})"
        )
