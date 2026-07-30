"""
ops.py — Low-level math operations for miniGrad.

Each function takes Tensors, performs the forward computation,
and wires up the backward pass. These are the building blocks
that neural network layers use internally.

All operations support broadcasting and track gradients through
the computation graph.
"""
from __future__ import annotations

import numpy as np
from typing import Any, Tuple, Optional, Union

from minigrad.tensor import Tensor


def add(a: Tensor, b: Union[Tensor, float, int, np.ndarray]) -> Tensor:
    return a + b


def mul(a: Tensor, b: Union[Tensor, float, int, np.ndarray]) -> Tensor:
    return a * b


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return a @ b


def sub(a: Tensor, b: Union[Tensor, float, int, np.ndarray]) -> Tensor:
    return a - b


def div(a: Tensor, b: Union[Tensor, float, int, np.ndarray]) -> Tensor:
    return a / b


def neg(a: Tensor) -> Tensor:
    return -a


def pow(a: Tensor, exponent: Union[int, float]) -> Tensor:
    return a ** exponent


def relu(a: Tensor) -> Tensor:
    return a.relu()


def sigmoid(a: Tensor) -> Tensor:
    return a.sigmoid()


def tanh(a: Tensor) -> Tensor:
    return a.tanh()


def gelu(a: Tensor) -> Tensor:
    return a.gelu()


def exp(a: Tensor) -> Tensor:
    return a.exp()


def log(a: Tensor) -> Tensor:
    return a.log()


def sum(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    return a.sum(axis=axis, keepdims=keepdims)


def mean(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    return a.mean(axis=axis, keepdims=keepdims)


def reshape(a: Tensor, *shape: int) -> Tensor:
    return a.reshape(*shape)


def transpose(a: Tensor, *axes: int) -> Tensor:
    return a.transpose(*axes)


def flatten(a: Tensor) -> Tensor:
    return a.flatten()


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """
    Numerically stable softmax along the specified axis.
    softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
    """
    shifted = x.data - np.max(x.data, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    probs = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    out = Tensor(probs, requires_grad=x.requires_grad, _children=(x,), _op="softmax")

    def _backward() -> None:
        if x.requires_grad:
            # Jacobian of softmax: diag(p) - p @ p.T
            # For batch efficiency: p * (grad - sum(p * grad, axis))
            p = out.data
            g = out.grad
            x.grad += p * (g - np.sum(p * g, axis=axis, keepdims=True))

    out._backward = _backward
    return out


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Log-softmax for numerical stability in cross-entropy."""
    shifted = x.data - np.max(x.data, axis=axis, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    log_probs = shifted - log_sum_exp
    out = Tensor(log_probs, requires_grad=x.requires_grad, _children=(x,), _op="log_softmax")

    def _backward() -> None:
        if x.requires_grad:
            probs = np.exp(out.data)
            x.grad += out.grad - probs * np.sum(out.grad, axis=axis, keepdims=True)

    out._backward = _backward
    return out


def max(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Max operation with gradient routing to the maximum positions."""
    out_data = np.max(x.data, axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,), _op="max")

    def _backward() -> None:
        if x.requires_grad:
            grad = out.grad
            if axis is not None and not keepdims:
                shape = list(x.data.shape)
                shape[axis] = 1
                grad = grad.reshape(shape)
            mask = x.data == np.max(x.data, axis=axis, keepdims=True)
            count = mask.sum(axis=axis, keepdims=True)
            x.grad += mask / count * grad

    out._backward = _backward
    return out


def min(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Min operation — reuses max on negated input."""
    return neg(max(neg(x), axis=axis, keepdims=keepdims))


def clip(x: Tensor, min_val: float, max_val: float) -> Tensor:
    """Clip values to [min_val, max_val]."""
    out_data = np.clip(x.data, min_val, max_val)
    out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,), _op="clip")

    def _backward() -> None:
        if x.requires_grad:
            mask = (x.data >= min_val) & (x.data <= max_val)
            x.grad += mask.astype(np.float64) * out.grad

    out._backward = _backward
    return out


def sqrt(x: Tensor) -> Tensor:
    return x ** 0.5


def square(x: Tensor) -> Tensor:
    return x ** 2


def abs(x: Tensor) -> Tensor:
    out_data = np.abs(x.data)
    out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,), _op="abs")

    def _backward() -> None:
        if x.requires_grad:
            x.grad += np.sign(x.data) * out.grad

    out._backward = _backward
    return out


def stack(tensors: list, axis: int = 0) -> Tensor:
    """Stack a list of tensors along a new axis."""
    data = np.stack([t.data for t in tensors], axis=axis)
    out = Tensor(data, requires_grad=any(t.requires_grad for t in tensors),
                 _children=tuple(tensors), _op="stack")

    def _backward() -> None:
        for i, t in enumerate(tensors):
            if t.requires_grad:
                idx: list[Any] = [slice(None)] * out.grad.ndim
                idx[axis] = i
                t.grad += out.grad[tuple(idx)]

    out._backward = _backward
    return out


def concat(tensors: list, axis: int = 0) -> Tensor:
    """Concatenate tensors along an existing axis."""
    data = np.concatenate([t.data for t in tensors], axis=axis)
    out = Tensor(data, requires_grad=any(t.requires_grad for t in tensors),
                 _children=tuple(tensors), _op="concat")

    def _backward() -> None:
        offset = 0
        for t in tensors:
            if t.requires_grad:
                slices = [slice(None)] * t.data.ndim
                slices[axis] = slice(offset, offset + t.data.shape[axis])
                t.grad += out.grad[tuple(slices)]
                offset += t.data.shape[axis]

    out._backward = _backward
    return out


def pad(x: Tensor, pad_width, constant_values: float = 0) -> Tensor:
    """Pad a tensor."""
    out_data = np.pad(x.data, pad_width, mode="constant", constant_values=constant_values)
    out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,), _op="pad")

    def _backward() -> None:
        if x.requires_grad:
            # Extract the original region from the padded gradient
            slices = []
            for p in pad_width:
                if isinstance(p, int):
                    slices.append(slice(p, -p if p > 0 else None))
                else:
                    slices.append(slice(p[0], -p[1] if p[1] > 0 else None))
            x.grad += out.grad[tuple(slices)]

    out._backward = _backward
    return out


def einsum(subscripts: str, *operands: Tensor) -> Tensor:
    """
    Einstein summation with autograd support.

    Supports the same notation as np.einsum: 'ij,jk->ik' for matmul,
    'bij,bjk->bik' for batched matmul, 'ii->' for trace, etc.

    Args:
        subscripts: Einsum subscript string (e.g., 'ij,jk->ik')
        *operands:  Input tensors

    Returns:
        Result tensor with gradients tracked
    """
    # Forward pass
    input_data = [op.data for op in operands]
    out_data = np.einsum(subscripts, *input_data)

    requires_grad = any(op.requires_grad for op in operands)
    out = Tensor(out_data, requires_grad=requires_grad,
                 _children=tuple(operands), _op="einsum")

    # Parse subscripts for backward
    if '->' in subscripts:
        input_subs, output_sub = subscripts.split('->')
    else:
        input_subs = subscripts
        output_sub = None
    input_sub_list = input_subs.split(',')

    def _backward() -> None:
        for i, op in enumerate(operands):
            if not op.requires_grad:
                continue
            target_sub = input_sub_list[i]

            # Trace / diagonal contraction special case (e.g. 'ii->')
            if len(set(target_sub)) < len(target_sub):
                if output_sub is None or output_sub == "":
                    dim = op.data.shape[0]
                    grad = out.grad * np.eye(dim)
                    op.grad += grad
                    continue

            # Standard einsum backward pass
            grad_sub = output_sub if output_sub is not None else ""
            backward_inputs = [grad_sub]
            backward_data = [out.grad]

            for j, op_j in enumerate(operands):
                if j != i:
                    backward_inputs.append(input_sub_list[j])
                    backward_data.append(op_j.data)

            backward_subscripts = ",".join(backward_inputs) + "->" + target_sub
            grad = np.einsum(backward_subscripts, *backward_data)
            op.grad += grad

    out._backward = _backward
    return out
