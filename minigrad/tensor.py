"""
tensor.py — The heart of miniGrad.

Tensor wraps a NumPy array and tracks every operation to build a dynamic
computation graph. Calling .backward() traverses the graph in reverse
topological order and applies the chain rule at each node.

This is exactly how PyTorch's autograd works — just in pure Python/NumPy.
"""
from __future__ import annotations

import weakref
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from minigrad.contracts import GRAPH_FREED_ERROR_MSG, GraphState

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

    __slots__ = (
        "data",
        "grad",
        "requires_grad",
        "_backward",
        "_prev",
        "_op",
        "_ctx",
        "_lifecycle",
        "_consumers",
        "__weakref__",
    )

    def __init__(
        self,
        data: ArrayLike,
        dtype: Optional[Union[np.dtype, type, str]] = None,
        requires_grad: bool = False,
        _children: Tuple[Tensor, ...] = (),
        _op: str = "",
        _ctx: Any = None,
    ) -> None:
        from minigrad.graph import is_grad_enabled

        if isinstance(data, (np.ndarray, np.generic)):
            self.data: np.ndarray = np.asarray(data) if dtype is None else np.asarray(data, dtype=dtype)
        elif dtype is not None:
            self.data = np.array(data, dtype=dtype)
        else:
            self.data = np.array(data, dtype=np.float64)

        self.grad: Any = np.zeros_like(self.data, dtype=Tensor._grad_dtype(self.data.dtype))
        if not is_grad_enabled():
            self.requires_grad = False
            self._prev: Tuple[Tensor, ...] = ()
        else:
            self.requires_grad = requires_grad
            self._prev = tuple(_children)
        self._backward: Callable[[], None] = lambda: None
        self._op: str = _op
        self._ctx: Any = _ctx
        self._lifecycle: int = GraphState.LIVE
        self._consumers: List[Any] = []
        for child in self._prev:
            child._consumers.append(weakref.ref(self))

    @staticmethod
    def _grad_dtype(dtype: Union[np.dtype, type, str]) -> np.dtype:
        """Gradients are continuous rates of change and must always be floating-point."""
        dt = np.dtype(dtype)
        if np.issubdtype(dt, np.floating):
            return dt
        return np.dtype(np.float64)

    @property
    def dtype(self) -> np.dtype:
        """The underlying NumPy dtype of the tensor."""
        return self.data.dtype

    def to(self, dtype: Union[np.dtype, type, str]) -> Tensor:
        """Cast tensor to a target dtype."""
        target_dtype = np.dtype(dtype)
        if self.dtype == target_dtype:
            return self
        out = Tensor(
            self.data.astype(target_dtype),
            dtype=target_dtype,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="to",
            _ctx=target_dtype,
        )

        def _backward() -> None:
            if self.requires_grad:
                dx = out.grad
                target_gtype = Tensor._grad_dtype(self.dtype)
                if isinstance(dx, np.ndarray) and dx.dtype != target_gtype:
                    dx = dx.astype(target_gtype)
                self.grad += dx

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Helper: ensure the other operand is a Tensor
    # ------------------------------------------------------------------
    @classmethod
    def _ensure_tensor(
        cls, other: Union[Tensor, ArrayLike], dtype: Optional[Union[np.dtype, type, str]] = None
    ) -> Tensor:
        if isinstance(other, Tensor):
            return other
        return cls(other, dtype=dtype)

    # ------------------------------------------------------------------
    # Core autograd operations
    # ------------------------------------------------------------------

    def __add__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        if isinstance(other, Tensor):
            res_dtype = np.result_type(self.data.dtype, other.data.dtype)
            other_t = other
        else:
            res_dtype = np.result_type(self.data, other)
            other_t = Tensor(other, dtype=res_dtype)

        self_data = self.data if self.data.dtype == res_dtype else self.data.astype(res_dtype)
        other_data = other_t.data if other_t.data.dtype == res_dtype else other_t.data.astype(res_dtype)

        out = Tensor(
            self_data + other_data,
            dtype=res_dtype,
            requires_grad=self.requires_grad or other_t.requires_grad,
            _children=(self, other_t),
            _op="add",
        )

        def _backward() -> None:
            # d(a+b)/da = 1, d(a+b)/db = 1
            grad = out.grad
            if self.requires_grad:
                dx = Tensor._unbroadcast(grad, self.data.shape)
                target_gtype = Tensor._grad_dtype(self.dtype)
                if isinstance(dx, np.ndarray) and dx.dtype != target_gtype:
                    dx = dx.astype(target_gtype)
                self.grad += dx
            if other_t.requires_grad:
                dy = Tensor._unbroadcast(grad, other_t.data.shape)
                target_other_gtype = Tensor._grad_dtype(other_t.dtype)
                if isinstance(dy, np.ndarray) and dy.dtype != target_other_gtype:
                    dy = dy.astype(target_other_gtype)
                other_t.grad += dy

        out._backward = _backward
        return out

    def __radd__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        return self.__add__(other)

    def __sub__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        if isinstance(other, Tensor):
            return self.__add__(-other)
        res_dtype = np.result_type(self.data, other)
        return self.__add__(Tensor(-np.asarray(other), dtype=res_dtype))

    def __rsub__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        if isinstance(other, Tensor):
            return other.__sub__(self)
        res_dtype = np.result_type(self.data, other)
        return Tensor(other, dtype=res_dtype).__add__(-self)

    def __neg__(self) -> Tensor:
        out = Tensor(
            -self.data,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="neg",
        )

        def _backward() -> None:
            if self.requires_grad:
                dx = -out.grad
                target_gtype = Tensor._grad_dtype(self.dtype)
                if isinstance(dx, np.ndarray) and dx.dtype != target_gtype:
                    dx = dx.astype(target_gtype)
                self.grad += dx

        out._backward = _backward
        return out

    def __mul__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        if isinstance(other, Tensor):
            res_dtype = np.result_type(self.data.dtype, other.data.dtype)
            other_t = other
        else:
            res_dtype = np.result_type(self.data, other)
            other_t = Tensor(other, dtype=res_dtype)

        self_data = self.data if self.data.dtype == res_dtype else self.data.astype(res_dtype)
        other_data = other_t.data if other_t.data.dtype == res_dtype else other_t.data.astype(res_dtype)

        out = Tensor(
            self_data * other_data,
            dtype=res_dtype,
            requires_grad=self.requires_grad or other_t.requires_grad,
            _children=(self, other_t),
            _op="mul",
        )

        def _backward() -> None:
            # d(a*b)/da = b, d(a*b)/db = a
            if self.requires_grad:
                dx = Tensor._unbroadcast(other_data * out.grad, self.data.shape)
                target_gtype = Tensor._grad_dtype(self.dtype)
                if isinstance(dx, np.ndarray) and dx.dtype != target_gtype:
                    dx = dx.astype(target_gtype)
                self.grad += dx
            if other_t.requires_grad:
                dy = Tensor._unbroadcast(self_data * out.grad, other_t.data.shape)
                target_other_gtype = Tensor._grad_dtype(other_t.dtype)
                if isinstance(dy, np.ndarray) and dy.dtype != target_other_gtype:
                    dy = dy.astype(target_other_gtype)
                other_t.grad += dy

        out._backward = _backward
        return out

    def __rmul__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        if isinstance(other, Tensor):
            res_dtype = np.result_type(self.data.dtype, other.data.dtype)
            other_t = other
        else:
            res_dtype = np.result_type(self.data.dtype, other)
            other_t = Tensor(other, dtype=res_dtype)

        # In Python/NumPy, true division always produces a floating-point type
        if not np.issubdtype(res_dtype, np.floating):
            res_dtype = np.dtype(np.float64)

        self_data = self.data.astype(res_dtype) if self.data.dtype != res_dtype else self.data
        other_data = other_t.data.astype(res_dtype) if other_t.data.dtype != res_dtype else other_t.data

        with np.errstate(divide="ignore", invalid="ignore"):
            out_data = self_data / other_data

        out = Tensor(
            out_data,
            dtype=res_dtype,
            requires_grad=self.requires_grad or other_t.requires_grad,
            _children=(self, other_t),
            _op="div",
        )

        def _backward() -> None:
            # d(a/b)/da = 1/b, d(a/b)/db = -a / (b^2)
            grad = out.grad
            if self.requires_grad:
                with np.errstate(divide="ignore", invalid="ignore"):
                    da = (1.0 / other_data) * grad
                dx = Tensor._unbroadcast(da, self.data.shape)
                target_gtype = Tensor._grad_dtype(self.dtype)
                if isinstance(dx, np.ndarray) and dx.dtype != target_gtype:
                    dx = dx.astype(target_gtype)
                self.grad += dx
            if other_t.requires_grad:
                with np.errstate(divide="ignore", invalid="ignore"):
                    db = (-self_data / (other_data ** 2)) * grad
                dy = Tensor._unbroadcast(db, other_t.data.shape)
                target_other_gtype = Tensor._grad_dtype(other_t.dtype)
                if isinstance(dy, np.ndarray) and dy.dtype != target_other_gtype:
                    dy = dy.astype(target_other_gtype)
                other_t.grad += dy

        out._backward = _backward
        return out

    def __rtruediv__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        if isinstance(other, Tensor):
            return other.__truediv__(self)
        res_dtype = np.result_type(other, self.data.dtype)
        if not np.issubdtype(res_dtype, np.floating):
            res_dtype = np.dtype(np.float64)
        other_t = Tensor(other, dtype=res_dtype)
        return other_t.__truediv__(self)

    def __pow__(self, other: Union[int, float, Tensor]) -> Tensor:
        other_t = self._ensure_tensor(other) if isinstance(other, Tensor) else other
        other_data = other_t.data if isinstance(other_t, Tensor) else other
        res_dtype = np.result_type(self.data.dtype, other_data)

        is_negative = (other < 0) if not isinstance(other, Tensor) else np.any(other.data < 0)
        # If exponent is negative, integer bases must be promoted to float (Python/NumPy rule)
        if is_negative and not np.issubdtype(res_dtype, np.floating):
            res_dtype = np.dtype(np.float64)

        self_data = self.data.astype(res_dtype) if self.data.dtype != res_dtype else self.data
        other_arr = np.asarray(other_data).astype(res_dtype) if np.asarray(other_data).dtype != res_dtype else other_data

        # Numerical Stability Contract for Power Operations:
        # In neural network autodiff, 0 raised to negative powers (e.g. in inverse norms/distances)
        # would yield +/-inf derivatives, permanently corrupting downstream parameters.
        # miniGrad enforces an intentional autograd stability policy: zero bases with negative
        # exponents are evaluated with a 1e-12 epsilon safeguard to maintain finite gradient flow.
        if is_negative:
            safe_base = np.where(self_data == 0, 1e-12, self_data)
            out_data = safe_base ** other_arr
        else:
            out_data = self_data ** other_arr

        out = Tensor(
            out_data,
            dtype=res_dtype,
            requires_grad=self.requires_grad or (isinstance(other_t, Tensor) and other_t.requires_grad),
            _children=(self,) if not isinstance(other_t, Tensor) else (self, other_t),
            _op=f"pow^{other}",
            _ctx=other,
        )

        def _backward() -> None:
            # d(x^n)/dx = n * x^(n-1)
            is_zero = (other == 0) if not isinstance(other, Tensor) else np.all(other.data == 0)
            if self.requires_grad:
                if not is_zero:
                    if is_negative:
                        safe_data = np.where(self_data == 0, 1e-12, self_data)
                        dx_val = (other_arr * (safe_data ** (other_arr - 1))) * out.grad
                    else:
                        dx_val = (other_arr * (self_data ** (other_arr - 1))) * out.grad
                    dx = Tensor._unbroadcast(dx_val, self.data.shape)
                    target_gtype = Tensor._grad_dtype(self.dtype)
                    if isinstance(dx, np.ndarray) and dx.dtype != target_gtype:
                        dx = dx.astype(target_gtype)
                    self.grad += dx
            if isinstance(other_t, Tensor) and other_t.requires_grad:
                if not is_zero:
                    # d(a^b)/db = a^b * ln(a) (EXACT: NO +1e-9)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        log_a = np.log(self_data)
                    dy = Tensor._unbroadcast(out.data * log_a * out.grad, other_t.data.shape)
                    target_other_gtype = Tensor._grad_dtype(other_t.dtype)
                    if isinstance(dy, np.ndarray) and dy.dtype != target_other_gtype:
                        dy = dy.astype(target_other_gtype)
                    other_t.grad += dy

        out._backward = _backward
        return out

    def __rpow__(self, other: Union[int, float, ArrayLike, Tensor]) -> Tensor:
        if isinstance(other, Tensor):
            return other.__pow__(self)
        res_dtype = np.result_type(other, self.data.dtype)
        is_self_negative = np.any(self.data < 0)
        if is_self_negative and not np.issubdtype(res_dtype, np.floating):
            res_dtype = np.dtype(np.float64)
        other_t = Tensor(other, dtype=res_dtype)
        return other_t.__pow__(self)

    def __matmul__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        if isinstance(other, Tensor):
            res_dtype = np.result_type(self.data.dtype, other.data.dtype)
            other_t = other
        else:
            res_dtype = np.result_type(self.data, other)
            other_t = Tensor(other, dtype=res_dtype)

        is_1d_lhs = (self.data.ndim == 1)
        self_2d = self.data.reshape(1, -1) if is_1d_lhs else self.data
        if self_2d.ndim != 2 or other_t.data.ndim != 2:
            raise ValueError(
                "Tensor matmul currently supports 2D tensors only; "
                f"got shapes {self.data.shape} and {other_t.data.shape}"
            )

        self_2d_p = self_2d if self_2d.dtype == res_dtype else self_2d.astype(res_dtype)
        other_p = other_t.data if other_t.data.dtype == res_dtype else other_t.data.astype(res_dtype)

        out_data = (self_2d_p @ other_p).reshape(-1) if is_1d_lhs else (self_2d_p @ other_p)
        out = Tensor(
            out_data,
            dtype=res_dtype,
            requires_grad=self.requires_grad or other_t.requires_grad,
            _children=(self, other_t),
            _op="matmul",
        )

        def _backward() -> None:
            # d(AB)/dA = grad @ B.T
            # d(AB)/dB = A.T @ grad
            grad_2d = out.grad.reshape(1, -1) if is_1d_lhs else out.grad
            if self.requires_grad:
                dx = grad_2d @ other_t.data.T
                dx = dx.reshape(self.data.shape) if is_1d_lhs else dx
                target_gtype = Tensor._grad_dtype(self.dtype)
                if isinstance(dx, np.ndarray) and dx.dtype != target_gtype:
                    dx = dx.astype(target_gtype)
                self.grad += dx
            if other_t.requires_grad:
                dy = self_2d.T @ grad_2d
                target_other_gtype = Tensor._grad_dtype(other_t.dtype)
                if isinstance(dy, np.ndarray) and dy.dtype != target_other_gtype:
                    dy = dy.astype(target_other_gtype)
                other_t.grad += dy

        out._backward = _backward
        return out

    def __rmatmul__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        return self._ensure_tensor(other).__matmul__(self)

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

    def sin(self) -> Tensor:
        out = Tensor(
            np.sin(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sin",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += np.cos(self.data) * out.grad

        out._backward = _backward
        return out

    def cos(self) -> Tensor:
        out = Tensor(
            np.cos(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="cos",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += -np.sin(self.data) * out.grad

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
        with np.errstate(divide="ignore", invalid="ignore"):
            out_data = np.log(self.data)
        out = Tensor(
            out_data,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log",
        )

        def _backward() -> None:
            if self.requires_grad:
                # d(ln(x))/dx = 1/x (EXACT: NO +1e-9)
                with np.errstate(divide="ignore", invalid="ignore"):
                    dx = (1.0 / self.data) * out.grad
                target_gtype = Tensor._grad_dtype(self.dtype)
                if isinstance(dx, np.ndarray) and dx.dtype != target_gtype:
                    dx = dx.astype(target_gtype)
                self.grad += dx

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Shape operations
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_axes(axis: Union[int, Tuple[int, ...]], ndim: int) -> Tuple[int, ...]:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        normalized = []
        for a in axes:
            if not -ndim <= a < ndim:
                raise ValueError(f"axis {a} is out of bounds for tensor of dimension {ndim}")
            normalized.append(a % ndim)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"duplicate value in 'axis': {axis}")
        return tuple(sorted(normalized))

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        axes = None if axis is None else Tensor._normalize_axes(axis, self.data.ndim)
        out = Tensor(
            self.data.sum(axis=axes, keepdims=keepdims),
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
            _ctx=(axes, keepdims, self.data.shape),
        )

        def _backward() -> None:
            if self.requires_grad:
                grad = out.grad
                if axes is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axes)
                self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        axes = None if axis is None else Tensor._normalize_axes(axis, self.data.ndim)
        if axes is None:
            n = self.data.size
        else:
            n = 1
            for a in axes:
                n *= self.data.shape[a]

        out = Tensor(
            self.data.mean(axis=axes, keepdims=keepdims),
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="mean",
            _ctx=(axes, keepdims, self.data.shape, n),
        )

        def _backward() -> None:
            if self.requires_grad:
                grad = out.grad / n
                if axes is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axes)
                self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def reshape(self, *shape: Any) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        original_shape = self.data.shape
        out = Tensor(
            self.data.reshape(shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
            _ctx=original_shape,
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad.reshape(original_shape)

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> Tensor:
        if not axes:
            axes = tuple(reversed(range(self.data.ndim)))
        # Compute inverse permutation for backward
        inverse_axes = tuple(axes.index(i) for i in range(len(axes)))
        out = Tensor(
            self.data.transpose(axes),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="transpose",
            _ctx=(axes, inverse_axes),
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad.transpose(inverse_axes)

        out._backward = _backward
        return out

    def split(self, split_size_or_sections: Union[int, Sequence[int]], axis: int = -1) -> List[Tensor]:
        from minigrad.ops import split
        return split(self, split_size_or_sections, axis=axis)

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
            _ctx=idx,
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

    def backward(self, create_graph: bool = False, retain_graph: bool = False) -> None:
        """
        Backpropagate gradients through the computation graph.

        Args:
            create_graph: If True, graph of the derivative will be constructed,
                          allowing higher-order derivative products to be computed.
            retain_graph: If False, the graph used to compute the grads will be freed.
        """
        if self._lifecycle == GraphState.FREED:
            raise RuntimeError(GRAPH_FREED_ERROR_MSG)

        from minigrad.glassbox import (
            GradientAnomalyError,
            diagnose_root_cause,
            is_anomaly_detection_enabled,
        )

        anomaly_check = is_anomaly_detection_enabled()

        topo: List[Tensor] = []
        visited: Set[int] = set()

        def build_topo(node: Tensor) -> None:
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        # Check for freed intermediate nodes along the graph
        for node in topo:
            if node is not self and node._prev and node._lifecycle == GraphState.FREED:
                raise RuntimeError(GRAPH_FREED_ERROR_MSG)

        if create_graph:
            from minigrad.autograd import grad

            leaf_nodes = [node for node in topo if not node._prev and node.requires_grad]
            if not leaf_nodes:
                return

            grads = grad(self, leaf_nodes, create_graph=True, retain_graph=True, allow_unused=True)
            for leaf, g in zip(leaf_nodes, grads):
                if isinstance(leaf.grad, Tensor):
                    leaf.grad = leaf.grad + g
                elif isinstance(leaf.grad, np.ndarray) and np.all(leaf.grad == 0):
                    leaf.grad = g
                else:
                    leaf.grad = Tensor(leaf.grad) + g
            self.grad = Tensor(np.ones_like(self.data), dtype=self.dtype, requires_grad=create_graph)
            return

        if anomaly_check:
            # Check 1: Forward pass NaN/Inf check if the loss itself is poisoned
            if np.isnan(self.data).any() or np.isinf(self.data).any():
                for node in topo:
                    if np.isnan(node.data).any() or np.isinf(node.data).any():
                        culprit_child = next(iter(node._prev), None)
                        details = {
                            "Culprit Node ID": f"#{id(node)}",
                            "Operation": f"[{node._op}]",
                            "Node Output Shape": str(node.data.shape),
                            "Node Output Has NaN": str(bool(np.isnan(node.data).any())),
                            "Node Output Has Inf": str(bool(np.isinf(node.data).any())),
                        }
                        if culprit_child is not None:
                            details["Input Tensor Shape"] = str(culprit_child.data.shape)
                            details["Input Data Range"] = (
                                f"min={np.min(culprit_child.data):.6e}, max={np.max(culprit_child.data):.6e}"
                                if culprit_child.data.size > 0 else "empty"
                            )
                            if node._op == "log":
                                non_pos = int(np.sum(culprit_child.data <= 0))
                                min_val = float(np.min(culprit_child.data)) if culprit_child.data.size > 0 else 0.0
                                reason = (
                                    f"Logarithm forward pass evaluated on non-positive input ({non_pos} elements <= 0, "
                                    f"min value: {min_val:.6e}). Produced NaN in forward pass."
                                )
                            else:
                                reason = f"Operation [{node._op}] produced NaN/Inf output during forward evaluation."
                        else:
                            reason = f"Node [{node._op}] produced NaN/Inf output during forward evaluation."

                        raise GradientAnomalyError(
                            node_id=id(node),
                            op=node._op or "forward_node",
                            reason=reason,
                            details=details,
                        )

        # Clear intermediate activation gradients from previous backward passes
        for node in topo:
            if node is not self and node._prev:
                node.grad = np.zeros_like(node.data, dtype=Tensor._grad_dtype(node.data.dtype))

        # Seed: dL/dL = 1
        self.grad = np.ones_like(self.data, dtype=Tensor._grad_dtype(self.data.dtype))

        # Reverse topological order: apply chain rule
        for node in reversed(topo):
            if anomaly_check:
                prev_grads = {
                    id(child): (
                        child.grad.copy()
                        if isinstance(child.grad, np.ndarray)
                        else (child.grad.data.copy() if child.grad is not None else None)
                    )
                    for child in node._prev
                }

                node._backward()

                for child in node._prev:
                    curr_g = child.grad if isinstance(child.grad, np.ndarray) else (child.grad.data if child.grad is not None else None)
                    if curr_g is None:
                        continue
                    old_g = prev_grads.get(id(child))
                    old_has_nan = np.isnan(old_g).any() if old_g is not None else False
                    old_has_inf = np.isinf(old_g).any() if old_g is not None else False

                    new_has_nan = np.isnan(curr_g).any()
                    new_has_inf = np.isinf(curr_g).any()

                    if (new_has_nan and not old_has_nan) or (new_has_inf and not old_has_inf):
                        reason = diagnose_root_cause(
                            node,
                            child,
                            old_g if old_g is not None else np.zeros_like(curr_g),
                        )
                        if new_has_nan and not old_has_nan:
                            nan_coords = np.argwhere(np.isnan(curr_g))
                            first_coord = tuple(int(x) for x in nan_coords[0]) if len(nan_coords) > 0 else ()
                        else:
                            inf_coords = np.argwhere(np.isinf(curr_g))
                            first_coord = tuple(int(x) for x in inf_coords[0]) if len(inf_coords) > 0 else ()

                        details = {
                            "Affected Child Node ID": f"#{id(child)}",
                            "Child Tensor Shape": str(child.data.shape),
                            "First Poisoned Coordinate": str(first_coord),
                            "Child Data Range": (
                                f"min={np.min(child.data):.6e}, max={np.max(child.data):.6e}"
                                if child.data.size > 0 else "empty"
                            ),
                            "Parent Op Node": f"[{node._op}] with context={getattr(node, '_ctx', None)}",
                        }
                        raise GradientAnomalyError(
                            node_id=id(node),
                            op=node._op or "unknown",
                            reason=reason,
                            details=details,
                        )
            else:
                node._backward()

        # If not retaining graph, transition root to FREED state and clear its backward closure
        if not retain_graph and not create_graph:
            self._lifecycle = GraphState.FREED
            self._backward = lambda: None

    def explain(self) -> str:
        """
        Generate a structured ASCII telemetry report of gradient flow across all nodes
        in the computation graph rooted at this tensor.
        """
        from minigrad.glassbox import explain_gradients
        return explain_gradients(self)

    def visualize(self, filename: Optional[Union[str, Any]] = "computational_graph.html") -> str:
        """
        Generate a zero-dependency interactive standalone HTML/SVG visualization
        of the computation graph rooted at this tensor, with live gradient health telemetry.
        """
        from minigrad.glassbox import visualize
        return visualize(self, filename=filename)

    def optimize_graph(
        self,
        max_passes: int = 5,
        enable_constant_folding: bool = True,
        enable_algebraic: bool = True,
        enable_fusion: bool = True,
        verify: bool = False,
    ) -> Tuple[Tensor, Any]:
        """
        Run symbolic optimizations and algebraic simplifications on the DAG rooted at this tensor.
        Returns a tuple of (optimized_tensor, OptimizationReport).
        """
        from minigrad.graph_opt import optimize_graph
        return optimize_graph(
            self,
            max_passes=max_passes,
            enable_constant_folding=enable_constant_folding,
            enable_algebraic=enable_algebraic,
            enable_fusion=enable_fusion,
            verify=verify,
        )

    def optimize(self, verify: bool = False) -> Tensor:
        """
        Convenience method: optimize computational graph and return the optimized root tensor.
        """
        from minigrad.graph_opt import optimize
        return optimize(self, verify=verify)

    def export_c(
        self,
        example_input: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None,
        filename: Optional[Union[str, Any]] = None,
        include_main: bool = True,
        model_name: str = "model",
        optimize: bool = True,
    ) -> str:
        """
        Compile the computation graph rooted at this tensor into standalone ANSI C
        with zero runtime dependencies and zero dynamic memory allocations (malloc/free).
        """
        from minigrad.compiler import export_c
        return export_c(
            self,
            example_input=example_input,
            filename=filename,
            include_main=include_main,
            model_name=model_name,
            optimize=optimize,
        )

    def to_c(
        self,
        example_input: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None,
        filename: Optional[Union[str, Any]] = None,
        include_main: bool = True,
        model_name: str = "model",
        optimize: bool = True,
    ) -> str:
        """Alias for export_c()."""
        return self.export_c(
            example_input=example_input,
            filename=filename,
            include_main=include_main,
            model_name=model_name,
            optimize=optimize,
        )

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

    def __float__(self) -> float:
        return float(self.data.flat[0])

    def __int__(self) -> int:
        return int(self.data.flat[0])

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data, dtype=self._grad_dtype(self.data.dtype))

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
