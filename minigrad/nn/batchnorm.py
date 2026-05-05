"""
batchnorm.py — Batch Normalization layers.

Normalizes layer inputs to have zero mean and unit variance.
This dramatically improves training stability and allows higher learning rates.

Two modes:
- Training:   Normalize using batch statistics, update running statistics
- Evaluation: Normalize using running statistics (frozen)

Reference: "Batch Normalization: Accelerating Deep Network Training by
Reducing Internal Covariate Shift" (Ioffe & Szegedy, 2015)
"""
from __future__ import annotations

import numpy as np

from minigrad.tensor import Tensor
from minigrad.nn.module import Module


class BatchNorm1D(Module):
    """
    Batch Normalization for 2D or 3D inputs (N, C) or (N, C, L).

    Args:
        num_features: Number of features/channels
        eps:          Small constant for numerical stability
        momentum:     Momentum for running mean/variance update
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters: scale (gamma) and shift (beta)
        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)

        # Running statistics for evaluation mode
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with different behavior for train/eval mode.

        Training:   use batch mean/var, update running stats
        Evaluation: use running mean/var, don't update
        """
        if x.data.ndim == 2:
            # (N, C) — fully connected layer output
            axis = 0
            shape = (1, -1)
        elif x.data.ndim == 3:
            # (N, C, L) — 1D convolution output
            axis = (0, 2)
            shape = (1, -1, 1)
        else:
            raise ValueError(f"BatchNorm1D expected 2D or 3D input, got {x.data.ndim}D")

        if self.training:
            # Use batch statistics
            batch_mean = x.data.mean(axis=axis, keepdims=True)
            batch_var = x.data.var(axis=axis, keepdims=True)

            # Normalize
            x_normalized = (x.data - batch_mean) / np.sqrt(batch_var + self.eps)

            # Scale and shift
            out_data = self.gamma.data.reshape(shape) * x_normalized + self.beta.data.reshape(shape)

            result = Tensor(out_data, requires_grad=x.requires_grad,
                           _children=(x, self.gamma, self.beta), _op="batch_norm_1d")

            # Update running statistics (no gradient here)
            mean_val = x.data.mean(axis=axis)
            var_val = x.data.var(axis=axis)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_val
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_val

            def _backward() -> None:
                if x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad:
                    N = np.prod([x.data.shape[d] for d in (axis if isinstance(axis, tuple) else (axis,))])
                    std_inv = 1.0 / np.sqrt(batch_var + self.eps)

                    # Gradient w.r.t. gamma
                    if self.gamma.requires_grad:
                        self.gamma.grad += (result.grad * x_normalized).sum(axis=axis).flatten()

                    # Gradient w.r.t. beta
                    if self.beta.requires_grad:
                        self.beta.grad += result.grad.sum(axis=axis).flatten()

                    # Gradient w.r.t. input (complex due to mean/var dependencies)
                    if x.requires_grad:
                        dx_normalized = result.grad * self.gamma.data.reshape(shape)
                        dx_var = (dx_normalized * (x.data - batch_mean) * -0.5 * std_inv**3).sum(axis=axis, keepdims=True)
                        dx_mean = (dx_normalized * -std_inv).sum(axis=axis, keepdims=True)
                        dx_var_term = 2.0 * (x.data - batch_mean) / N * dx_var
                        dx = dx_normalized * std_inv + dx_var_term + dx_mean / N
                        x.grad += dx

            result._backward = _backward

        else:
            # Evaluation mode: use running statistics
            x_normalized = (x.data - self.running_mean.reshape(shape)) / np.sqrt(self.running_var.reshape(shape) + self.eps)
            out_data = self.gamma.data.reshape(shape) * x_normalized + self.beta.data.reshape(shape)
            result = Tensor(out_data, requires_grad=False)  # No gradients in eval

        return result

    def __repr__(self) -> str:
        return f"BatchNorm1D({self.num_features})"


class BatchNorm2D(Module):
    """
    Batch Normalization for 4D convolutional inputs (N, C, H, W).

    Normalizes each channel independently across the batch and spatial dimensions.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: Tensor) -> Tensor:
        """Input: (N, C, H, W)"""
        if x.data.ndim != 4:
            raise ValueError(f"BatchNorm2D expected 4D input, got {x.data.ndim}D")

        N, C, H, W = x.data.shape
        axis = (0, 2, 3)  # average over batch and spatial dims
        shape = (1, -1, 1, 1)

        if self.training:
            batch_mean = x.data.mean(axis=axis, keepdims=True)  # (1, C, 1, 1)
            batch_var = x.data.var(axis=axis, keepdims=True)    # (1, C, 1, 1)

            x_normalized = (x.data - batch_mean) / np.sqrt(batch_var + self.eps)
            out_data = self.gamma.data.reshape(shape) * x_normalized + self.beta.data.reshape(shape)

            result = Tensor(out_data, requires_grad=x.requires_grad,
                           _children=(x, self.gamma, self.beta), _op="batch_norm_2d")

            # Update running stats
            mean_val = x.data.mean(axis=axis)  # (C,)
            var_val = x.data.var(axis=axis)    # (C,)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_val
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_val

            def _backward() -> None:
                if x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad:
                    N_total = N * H * W  # total elements per channel
                    std_inv = 1.0 / np.sqrt(batch_var + self.eps)

                    if self.gamma.requires_grad:
                        self.gamma.grad += (result.grad * x_normalized).sum(axis=axis).flatten()

                    if self.beta.requires_grad:
                        self.beta.grad += result.grad.sum(axis=axis).flatten()

                    if x.requires_grad:
                        dx_normalized = result.grad * self.gamma.data.reshape(shape)
                        dx_var = (dx_normalized * (x.data - batch_mean) * -0.5 * std_inv**3).sum(axis=axis, keepdims=True)
                        dx_mean = (dx_normalized * -std_inv).sum(axis=axis, keepdims=True)
                        dx_var_term = 2.0 * (x.data - batch_mean) / N_total * dx_var
                        dx = dx_normalized * std_inv + dx_var_term + dx_mean / N_total
                        x.grad += dx

            result._backward = _backward

        else:
            x_normalized = (x.data - self.running_mean.reshape(shape)) / np.sqrt(self.running_var.reshape(shape) + self.eps)
            out_data = self.gamma.data.reshape(shape) * x_normalized + self.beta.data.reshape(shape)
            result = Tensor(out_data, requires_grad=False)

        return result

    def __repr__(self) -> str:
        return f"BatchNorm2D({self.num_features})"
