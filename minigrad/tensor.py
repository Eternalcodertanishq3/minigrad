"""
tensor.py — The heart of miniGrad.

Tensor wraps a NumPy array and tracks every operation to build a dynamic
computation graph. Calling .backward() traverses the graph in reverse
topological order and applies the chain rule at each node.

This is exactly how PyTorch's autograd works — just in pure Python/NumPy.
"""
from __future__ import annotations

import numpy as np
from typing import Set, Tuple, Callable, Union, Optional, List

# Type alias for convenience
ArrayLike = Union[np.ndarray, list, tuple, float, int]


class Tensor:
    """
    A Tensor tracks its own data, gradient, and computation history.

    Attributes:
        data: The underlying NumPy array.
        grad:  Gradient accumulated during backprop (same shape as data).
        requires_grad: Whether this tensor needs gradients computed.
        _backward: Function that computes gradients w.r.t. parents.
        _prev: Set of parent tensors in the computation graph.
        _op: String name of the operation that created this tensor.
    """

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _children: Tuple[Tensor, ...] = (),
        _op: str = "",
    ) -> None:
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Tensor] = set(_children)
        self._op: str = _op

    # ------------------------------------------------------------------
    # Helper: ensure the other operand is a Tensor
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_tensor(other: Union[Tensor, ArrayLike]) -> Tensor:
        return other if isinstance(other, Tensor) else Tensor(other)

    # ------------------------------------------------------------------
    # Core autograd operations
    # ------------------------------------------------------------------

    def __add__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        other = self._ensure_tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="add",
        )

        def _backward() -> None:
            # d(a+b)/da = 1, d(a+b)/db = 1
            grad = out.grad
            if self.requires_grad:
                self.grad += Tensor._unbroadcast(grad, self.data.shape)
            if other.requires_grad:
                other.grad += Tensor._unbroadcast(grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        return self.__add__(other)

    def __sub__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        return self.__add__(-other)

    def __rsub__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        return self._ensure_tensor(other).__add__(-self)

    def __neg__(self) -> Tensor:
        return self * (-1)

    def __mul__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        other = self._ensure_tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="mul",
        )

        def _backward() -> None:
            # d(a*b)/da = b, d(a*b)/db = a
            if self.requires_grad:
                self.grad += Tensor._unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += Tensor._unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        return self * (self._ensure_tensor(other) ** (-1))

    def __rtruediv__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        return self._ensure_tensor(other) * (self ** (-1))

    def __pow__(self, other: Union[int, float]) -> Tensor:
        out = Tensor(
            self.data ** other,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"pow^{other}",
        )

        def _backward() -> None:
            # d(x^n)/dx = n * x^(n-1)
            if self.requires_grad:
                self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other: Tensor) -> Tensor:
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="matmul",
        )

        def _backward() -> None:
            # d(AB)/dA = grad @ B.T
            # d(AB)/dB = A.T @ grad
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Activation & math operations
    # ------------------------------------------------------------------

    def relu(self) -> Tensor:
        out = Tensor(
            np.maximum(0.0, self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        # Stable sigmoid
        z = self.data
        out_data = np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sigmoid",
        )

        def _backward() -> None:
            if self.requires_grad:
                # d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
                self.grad += out.data * (1 - out.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        out_data = np.tanh(self.data)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="tanh",
        )

        def _backward() -> None:
            if self.requires_grad:
                # d(tanh)/dx = 1 - tanh^2(x)
                self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward
        return out

    def gelu(self) -> Tensor:
        """GELU activation: x * Φ(x) where Φ is the standard normal CDF."""
        x = self.data
        # Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        cdf_approx = 0.5 * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))
        out_data = x * cdf_approx
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="gelu",
        )

        def _backward() -> None:
            if self.requires_grad:
                # Derivative of GELU approximation
                tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x**3)
                tanh_val = np.tanh(tanh_arg)
                sech2 = 1.0 - tanh_val**2
                dx = 0.5 + 0.5 * tanh_val + x * 0.5 * sech2 * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x**2)
                self.grad += dx * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Tensor:
        out_data = np.exp(self.data)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="exp",
        )

        def _backward() -> None:
            if self.requires_grad:
                # d(exp(x))/dx = exp(x)
                self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        out = Tensor(
            np.log(self.data + 1e-9),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log",
        )

        def _backward() -> None:
            if self.requires_grad:
                # d(ln(x))/dx = 1/x
                self.grad += (1.0 / (self.data + 1e-9)) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Shape operations
    # ------------------------------------------------------------------

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward() -> None:
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    # Need to restore reduced dimensions for broadcasting
                    axes = (axis,) if isinstance(axis, int) else axis
                    axes_tuple = tuple(axes) if isinstance(axes, tuple) else (axes,)
                    # Build shape with 1s at reduced axes
                    shape = []
                    grad_idx = 0
                    for i in range(self.data.ndim):
                        if i in axes_tuple:
                            shape.append(1)
                        else:
                            shape.append(grad.shape[grad_idx])
                            grad_idx += 1
                    grad = grad.reshape(shape)
                self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        if axis is None:
            n = self.data.size
        elif isinstance(axis, int):
            n = self.data.shape[axis]
        else:
            n = 1
            for a in axis:
                n *= self.data.shape[a]

        out = Tensor(
            self.data.mean(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="mean",
        )

        def _backward() -> None:
            if self.requires_grad:
                grad = out.grad / n
                if axis is not None and not keepdims:
                    axes = (axis,) if isinstance(axis, int) else axis
                    axes_tuple = tuple(axes) if isinstance(axes, tuple) else (axes,)
                    shape = []
                    grad_idx = 0
                    for i in range(self.data.ndim):
                        if i in axes_tuple:
                            shape.append(1)
                        else:
                            shape.append(grad.shape[grad_idx])
                            grad_idx += 1
                    grad = grad.reshape(shape)
                self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def reshape(self, *shape: int) -> Tensor:
        original_shape = self.data.shape
        out = Tensor(
            self.data.reshape(shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad.reshape(original_shape)

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> Tensor:
        if not axes:
            axes = tuple(reversed(range(self.data.ndim)))
        out = Tensor(
            self.data.transpose(axes),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="transpose",
        )

        # Compute inverse permutation for backward
        inverse_axes = tuple(axes.index(i) for i in range(len(axes)))

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad.transpose(inverse_axes)

        out._backward = _backward
        return out

    def flatten(self) -> Tensor:
        return self.reshape(-1)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, idx) -> Tensor:
        out = Tensor(
            self.data[idx],
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="getitem",
        )

        def _backward() -> None:
            if self.requires_grad:
                np.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Broadcasting helper
    # ------------------------------------------------------------------

    @staticmethod
    def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Sum gradients along dimensions that were broadcast during forward.
        When a tensor of shape (3,) is added to (64, 3), the gradient for
        the (3,) tensor must be summed over the batch dimension.
        """
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, (g, s) in enumerate(zip(grad.shape, shape)):
            if g != s:
                grad = grad.sum(axis=i, keepdims=True)
        return grad.reshape(shape)

    # ------------------------------------------------------------------
    # Backward pass — topological sort + chain rule
    # ------------------------------------------------------------------

    def backward(self) -> None:
        """
        Backpropagate gradients through the computation graph.

        1. Topologically sort the graph (children before parents).
        2. Seed the output gradient with 1s.
        3. Visit each node in reverse topological order and call its _backward.
        """
        topo: List[Tensor] = []
        visited: Set[int] = set()

        def build_topo(node: Tensor) -> None:
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        # Seed: dL/dL = 1
        self.grad = np.ones_like(self.data)

        # Reverse topological order: apply chain rule
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def item(self) -> float:
        return float(self.data.flat[0])

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    def numpy(self) -> np.ndarray:
        """Return a detached copy of the data."""
        return self.data.copy()

    def copy(self) -> Tensor:
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    def __repr__(self) -> str:
        grad_flag = ", grad_fn" if self._op else ""
        return f"Tensor({self.data}, requires_grad={self.requires_grad}{grad_flag})"

    def __len__(self) -> int:
        return len(self.data)
