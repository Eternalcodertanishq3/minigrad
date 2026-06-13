"""
dropout.py — Dropout regularization layer.

Randomly zeroes some of the elements of the input tensor with probability p.
This prevents neurons from co-adapting too much (reduces overfitting).

Training:  Apply dropout mask and scale by 1/(1-p) (inverted dropout)
Evaluation: Passthrough (no dropout)

Reference: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
(Srivastava et al., 2014)
"""
from __future__ import annotations

import numpy as np

from minigrad.tensor import Tensor
from minigrad.nn.module import Module


class Dropout(Module):
    """
    Dropout layer.

    Args:
        p: Probability of an element to be zeroed (default: 0.5)

    During training, each element is kept with probability (1-p) and scaled
    by 1/(1-p). During evaluation, this is a no-op.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not (0 <= p < 1):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
        self._mask: np.ndarray | None = None

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            # Evaluation mode or no dropout — passthrough
            return x

        if self.p == 1.0:
            return Tensor(np.zeros_like(x.data), requires_grad=False)

        # Inverted dropout: scale during training so eval is a no-op
        scale = 1.0 / (1.0 - self.p)
        mask = (np.random.rand(*x.data.shape) >= self.p).astype(x.data.dtype)
        self._mask = mask

        out_data = x.data * mask * scale
        out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,), _op="dropout")

        def _backward() -> None:
            if x.requires_grad:
                x.grad += out.grad * mask * scale

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


class Dropout2D(Module):
    """
    Dropout for 4D convolutional inputs (N, C, H, W).

    Drops entire channels rather than individual elements.
    More appropriate for convolutional feature maps.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not (0 <= p < 1):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x

        if x.data.ndim != 4:
            raise ValueError(f"Dropout2D expected 4D input, got {x.data.ndim}D")

        N, C, H, W = x.data.shape
        scale = 1.0 / (1.0 - self.p)
        # Drop entire channels: (N, C, 1, 1)
        mask = (np.random.rand(N, C, 1, 1) >= self.p).astype(x.data.dtype)

        out_data = x.data * mask * scale
        out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,), _op="dropout2d")

        def _backward() -> None:
            if x.requires_grad:
                x.grad += out.grad * mask * scale

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"Dropout2D(p={self.p})"
