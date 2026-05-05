"""
activations.py — Non-linear activation layers.

Each activation is a Module that applies a pointwise non-linearity.
These introduce the non-linear capacity that makes neural networks
universal function approximators.
"""
from __future__ import annotations

from minigrad.tensor import Tensor
from minigrad.nn.module import Module


class ReLU(Module):
    """
    Rectified Linear Unit: f(x) = max(0, x)

    The most widely used activation. Solves the vanishing gradient problem
    for positive inputs. Computationally cheap.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Module):
    """
    Sigmoid: f(x) = 1 / (1 + exp(-x))

    Maps any input to (0, 1). Historically popular but suffers from
    vanishing gradients at extremes. Still used for binary classification outputs.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(Module):
    """
    Hyperbolic tangent: f(x) = tanh(x)

    Maps any input to (-1, 1). Zero-centered output helps with
    gradient flow compared to sigmoid.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()

    def __repr__(self) -> str:
        return "Tanh()"


class GELU(Module):
    """
    Gaussian Error Linear Unit.

    GELU(x) = x * Φ(x) where Φ is the CDF of the standard normal distribution.
    Smooth alternative to ReLU. Used in Transformer architectures (BERT, GPT).

    We use the tanh approximation for computational efficiency.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()

    def __repr__(self) -> str:
        return "GELU()"


class Softmax(Module):
    """
    Softmax activation: f(x_i) = exp(x_i) / sum_j(exp(x_j))

    Converts logits to probability distributions. Typically used
    as the final layer for multi-class classification.
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        from minigrad.ops import softmax
        return softmax(x, axis=self.dim)

    def __repr__(self) -> str:
        return f"Softmax(dim={self.dim})"


class LeakyReLU(Module):
    """
    Leaky ReLU: f(x) = x if x > 0, else f(x) = alpha * x

    Allows a small gradient when x < 0 to prevent dying ReLU problem.
    """

    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        # x * (x > 0) + alpha * x * (x <= 0) = x * ((x > 0) + alpha * (x <= 0))
        mask = (x.data > 0).astype(x.data.dtype) + self.negative_slope * (x.data <= 0).astype(x.data.dtype)
        out_data = x.data * mask
        out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,), _op="leaky_relu")

        def _backward() -> None:
            if x.requires_grad:
                x.grad += mask * out.grad

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"LeakyReLU(negative_slope={self.negative_slope})"


class ELU(Module):
    """
    Exponential Linear Unit.

    f(x) = x if x > 0, else alpha * (exp(x) - 1)
    Smooth negative region helps with mean activation closer to zero.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        import numpy as np
        out_data = np.where(x.data > 0, x.data, self.alpha * (np.exp(x.data) - 1))
        out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,), _op="elu")

        def _backward() -> None:
            if x.requires_grad:
                grad = np.where(x.data > 0, 1.0, self.alpha * np.exp(x.data))
                x.grad += grad * out.grad

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"ELU(alpha={self.alpha})"
