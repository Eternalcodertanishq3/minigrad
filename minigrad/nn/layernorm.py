"""
layernorm.py — Layer Normalization.

Unlike BatchNorm which normalizes across the batch dimension,
LayerNorm normalizes across the feature dimensions for each sample independently.
This makes it suitable for sequence models (Transformers, RNNs) where
batch statistics are unreliable.

Reference: "Layer Normalization" (Ba, Kiros & Hinton, 2016)
"""
from __future__ import annotations

import weakref

import numpy as np

from minigrad.nn.module import Module
from minigrad.tensor import Tensor


class LayerNorm(Module):
    """
    Layer Normalization over the last D dimensions.

    Args:
        normalized_shape: Input shape from the last D dimensions
                         (e.g., (C,) for features, (C, H, W) for conv)
        eps:             Small constant for numerical stability
    """

    def __init__(self, normalized_shape, eps: float = 1e-5) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        # Learnable affine parameters
        self.gamma = Tensor(np.ones(self.normalized_shape), requires_grad=True)
        self.beta = Tensor(np.zeros(self.normalized_shape), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Normalize over the last len(normalized_shape) dimensions.

        Args:
            x: Input tensor of shape (*, *normalized_shape)
        Returns:
            Normalized tensor of same shape
        """
        ndim = len(self.normalized_shape)
        # Axes to normalize over (last ndim dimensions)
        axes = tuple(range(x.data.ndim - ndim, x.data.ndim))

        mean = x.data.mean(axis=axes, keepdims=True)
        var = x.data.var(axis=axes, keepdims=True)
        x_normalized = (x.data - mean) / np.sqrt(var + self.eps)

        out_data = self.gamma.data * x_normalized + self.beta.data

        requires_grad = x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad
        result = Tensor(out_data, requires_grad=requires_grad,
                       _children=(x, self.gamma, self.beta), _op="layer_norm", _ctx=(axes, self.eps))

        if result.requires_grad:
            res_ref = weakref.ref(result)

            def _backward() -> None:
                res = res_ref()
                if res is None:
                    return
                N = 1
                for a in axes:
                    N *= x.data.shape[a]
                std_inv = 1.0 / np.sqrt(var + self.eps)

                if self.gamma.requires_grad:
                    self.gamma.grad += (res.grad * x_normalized).sum(
                        axis=tuple(range(x.data.ndim - ndim)), keepdims=False
                    ) if x.data.ndim > ndim else (res.grad * x_normalized)

                if self.beta.requires_grad:
                    self.beta.grad += res.grad.sum(
                        axis=tuple(range(x.data.ndim - ndim)), keepdims=False
                    ) if x.data.ndim > ndim else res.grad

                if x.requires_grad:
                    dx_normalized = res.grad * self.gamma.data
                    dx_var = (dx_normalized * (x.data - mean) * -0.5 * std_inv**3).sum(axis=axes, keepdims=True)
                    dx_mean = (dx_normalized * -std_inv).sum(axis=axes, keepdims=True)
                    dx_var_term = 2.0 * (x.data - mean) / N * dx_var
                    dx = dx_normalized * std_inv + dx_var_term + dx_mean / N
                    x.grad += dx

            result._backward = _backward
        return result

    def __repr__(self) -> str:
        return f"LayerNorm({self.normalized_shape})"
