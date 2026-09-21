"""
linear.py — Fully connected (dense) layer.

Computes: y = x @ W^T + b
Uses Kaiming He initialization for weight stability in deep networks.
Supports arbitrary leading dimensions (e.g. (B, T, C) for sequences).
"""
from __future__ import annotations

import numpy as np

from minigrad.tensor import Tensor
from minigrad.nn.module import Module


class Linear(Module):
    """
    Linear transformation: out = input @ weight + bias

    Args:
        in_features:  Size of each input sample
        out_features: Size of each output sample
        bias:         If False, the layer will not learn an additive bias
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()

        # Kaiming He initialization — critical for training deep networks
        # std = sqrt(2 / in_features) for ReLU activations
        scale = np.sqrt(2.0 / in_features)

        self.weight = Tensor(
            np.random.randn(in_features, out_features) * scale,
            requires_grad=True,
        )

        self.bias: Tensor | None
        if bias:
            self.bias = Tensor(
                np.zeros(out_features),
                requires_grad=True,
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (..., in_features)
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Handle arbitrary leading dimensions (e.g. (B, T, C) for sequences)
        leading_shape = x.data.shape[:-1]
        in_features = x.data.shape[-1]

        if x.data.ndim > 2:
            x_2d = x.reshape(-1, in_features)
        else:
            x_2d = x

        out = x_2d @ self.weight  # (N, in) @ (in, out) -> (N, out)

        if self.bias is not None:
            out = out + self.bias

        # Restore leading dimensions
        if x.data.ndim > 2:
            out = out.reshape(*leading_shape, self.weight.data.shape[1])

        return out

    def __repr__(self) -> str:
        return (
            f"Linear(in_features={self.weight.shape[0]}, "
            f"out_features={self.weight.shape[1]}, bias={self.bias is not None})"
        )
