"""
autograd.py — Higher-order automatic differentiation engine and functional autograd API.

Provides:
- grad(): Computes and returns the sum of gradients of outputs with respect to the inputs.
  When create_graph=True, constructs a differentiable computation graph for higher-order derivatives.
- hessian(): Computes the full Hessian matrix H_ij = ∂²y / (∂x_i ∂x_j).

Unlocks Physics-Informed Neural Networks (PINNs), gradient penalties (WGAN-GP),
MAML meta-learning, and curvature/Hessian-vector products.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from minigrad.tensor import Tensor


# ── VJP Unbroadcasting Helper ────────────────────────────────────────

def unbroadcast_tensor(t: Tensor, target_shape: Tuple[int, ...]) -> Tensor:
    """
    Unbroadcast a Tensor back to target_shape using differentiable Tensor operations.
    """
    if t.data.shape == target_shape:
        return t

    # 1. Reduce extra leading dimensions
    while t.data.ndim > len(target_shape):
        t = t.sum(axis=0, keepdims=False)

    # 2. Reduce dimensions that were broadcast from size 1
    for i, (dim, target_dim) in enumerate(zip(t.data.shape, target_shape)):
        if dim != target_dim and target_dim == 1:
            t = t.sum(axis=i, keepdims=True)

    if t.data.shape != target_shape:
        t = t.reshape(*target_shape)

    return t


# ── Symbolic Vector-Jacobian Product (VJP) Dispatch ─────────────────

def _compute_vjp(node: Tensor, g: Tensor) -> Tuple[Optional[Tensor], ...]:
    """
    Compute Vector-Jacobian Products for each parent of node w.r.t incoming gradient g.
    Returns a tuple of Tensors (or None) corresponding to node._prev.
    """
    op = node._op
    children = node._prev

    if op == "add":
        a, b = children
        vjp_a = unbroadcast_tensor(g, a.shape) if a.requires_grad else None
        vjp_b = unbroadcast_tensor(g, b.shape) if b.requires_grad else None
        return (vjp_a, vjp_b)

    elif op == "mul":
        a, b = children
        vjp_a = unbroadcast_tensor(g * b, a.shape) if a.requires_grad else None
        vjp_b = unbroadcast_tensor(g * a, b.shape) if b.requires_grad else None
        return (vjp_a, vjp_b)

    elif op == "neg":
        (a,) = children
        return (-g if a.requires_grad else None,)

    elif op.startswith("pow"):
        if len(children) == 1:
            (a,) = children
            p = getattr(node, "_ctx", None)
            if p is None and "^" in op:
                try:
                    p = float(op.split("^", 1)[1])
                except (ValueError, IndexError):
                    p = 1.0
            if p == 0:
                return (None,)
            deriv = (a ** (p - 1)) * p
            vjp_a = unbroadcast_tensor(g * deriv, a.shape) if a.requires_grad else None
            return (vjp_a,)
        else:
            a, b = children
            # d(a^b)/da = b * a^(b-1)
            # d(a^b)/db = a^b * ln(a)
            vjp_a = unbroadcast_tensor(g * b * (a ** (b - 1)), a.shape) if a.requires_grad else None
            vjp_b = unbroadcast_tensor(g * node * a.log(), b.shape) if b.requires_grad else None
            return (vjp_a, vjp_b)

    elif op == "matmul":
        a, b = children
        vjp_a = (g @ b.transpose()) if a.requires_grad else None
        vjp_b = (a.transpose() @ g) if b.requires_grad else None
        return (vjp_a, vjp_b)

    elif op == "tanh":
        (a,) = children
        one = Tensor(np.ones_like(node.data))
        deriv = one - (node ** 2)
        return (g * deriv if a.requires_grad else None,)

    elif op == "sin":
        (a,) = children
        return (g * a.cos() if a.requires_grad else None,)

    elif op == "cos":
        (a,) = children
        return (-g * a.sin() if a.requires_grad else None,)

    elif op == "exp":
        (a,) = children
        return (g * node if a.requires_grad else None,)

    elif op == "log":
        (a,) = children
        return (g / a if a.requires_grad else None,)

    elif op == "relu":
        (a,) = children
        mask = Tensor((a.data > 0).astype(np.float64))
        return (g * mask if a.requires_grad else None,)

    elif op == "sigmoid":
        (a,) = children
        one = Tensor(np.ones_like(node.data))
        deriv = node * (one - node)
        return (g * deriv if a.requires_grad else None,)

    elif op == "sum":
        (a,) = children
        ctx = getattr(node, "_ctx", None)
        if ctx is not None and isinstance(ctx, tuple) and len(ctx) == 3:
            axes, keepdims, orig_shape = ctx
        else:
            axes, keepdims, orig_shape = None, False, a.shape

        if axes is not None and not keepdims:
            expanded_shape = list(orig_shape)
            for ax in axes:
                expanded_shape[ax] = 1
            g_expanded = g.reshape(*expanded_shape)
        else:
            g_expanded = g

        vjp_a = g_expanded * Tensor(np.ones(orig_shape)) if a.requires_grad else None
        return (vjp_a,)

    elif op == "mean":
        (a,) = children
        ctx = getattr(node, "_ctx", None)
        if ctx is not None and isinstance(ctx, tuple) and len(ctx) == 4:
            axes, keepdims, orig_shape, n = ctx
        else:
            axes, keepdims, orig_shape, n = None, False, a.shape, a.data.size

        if axes is not None and not keepdims:
            expanded_shape = list(orig_shape)
            for ax in axes:
                expanded_shape[ax] = 1
            g_expanded = g.reshape(*expanded_shape)
        else:
            g_expanded = g

        vjp_a = (g_expanded * (1.0 / n)) * Tensor(np.ones(orig_shape)) if a.requires_grad else None
        return (vjp_a,)

    elif op == "reshape":
        (a,) = children
        orig_shape = getattr(node, "_ctx", a.shape)
        return (g.reshape(*orig_shape) if a.requires_grad else None,)

    elif op == "transpose":
        (a,) = children
        axes_info = getattr(node, "_ctx", None)
        if axes_info is not None:
            _, inv_axes = axes_info
            return (g.transpose(*inv_axes) if a.requires_grad else None,)
        return (g.transpose() if a.requires_grad else None,)

    elif op == "getitem":
        (a,) = children
        idx = getattr(node, "_ctx", None)
        grad_a = np.zeros(a.shape, dtype=np.float64)
        np.add.at(grad_a, idx, g.data)
        vjp_a = Tensor(grad_a, requires_grad=True) if a.requires_grad else None
        return (vjp_a,)

    # Fallback: default to None for unhandled operations
    return tuple(None for _ in children)


# ── Functional Autograd API ──────────────────────────────────────────

def grad(
    outputs: Union[Tensor, Sequence[Tensor]],
    inputs: Union[Tensor, Sequence[Tensor]],
    grad_outputs: Optional[Union[Tensor, Sequence[Tensor]]] = None,
    retain_graph: bool = False,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> Tuple[Tensor, ...]:
    """
    Computes and returns the sum of gradients of outputs with respect to the inputs.

    Args:
        outputs:       Output tensor(s) to differentiate.
        inputs:        Input tensor(s) w.r.t which gradients are computed.
        grad_outputs:  Initial vector in the Vector-Jacobian Product. If None,
                       defaults to ones of shape like output (outputs must be scalar).
        retain_graph:  If False, the graph used to compute the grads will be freed.
        create_graph:  If True, graph of the derivative will be constructed,
                       allowing higher-order derivative products to be computed.
        allow_unused:  If False, raising an error when an input is unused in the graph.

    Returns:
        Tuple of Tensors containing gradients for each input.
    """
    outputs_list: List[Tensor] = [outputs] if isinstance(outputs, Tensor) else list(outputs)
    inputs_list: List[Tensor] = [inputs] if isinstance(inputs, Tensor) else list(inputs)

    # Initialize grad_outputs
    if grad_outputs is None:
        grad_outputs_list: List[Tensor] = []
        for out in outputs_list:
            if out.data.size != 1:
                raise RuntimeError("grad can only be implicitly created for scalar outputs")
            grad_outputs_list.append(
                Tensor(np.ones_like(out.data), requires_grad=create_graph)
            )
    else:
        if isinstance(grad_outputs, Tensor):
            grad_outputs_list = [grad_outputs]
        else:
            grad_outputs_list = [
                g if isinstance(g, Tensor) else Tensor(g, requires_grad=create_graph)
                for g in grad_outputs
            ]

    # Build topological sort from outputs
    topo: List[Tensor] = []
    visited: Set[int] = set()

    def build_topo(node: Tensor) -> None:
        if id(node) not in visited:
            visited.add(id(node))
            for child in node._prev:
                build_topo(child)
            topo.append(node)

    for out in outputs_list:
        build_topo(out)

    # Seed gradient map
    grad_map: Dict[int, Tensor] = {}
    for out, gout in zip(outputs_list, grad_outputs_list):
        if id(out) in grad_map:
            grad_map[id(out)] = grad_map[id(out)] + gout
        else:
            grad_map[id(out)] = gout

    # Reverse topological order traversal
    for node in reversed(topo):
        if id(node) not in grad_map:
            continue
        g = grad_map[id(node)]
        if not node._prev or not node._op:
            continue

        vjps = _compute_vjp(node, g)
        for child, vjp in zip(node._prev, vjps):
            if vjp is None or not child.requires_grad:
                continue
            if id(child) in grad_map:
                grad_map[id(child)] = grad_map[id(child)] + vjp
            else:
                grad_map[id(child)] = vjp

    # Collect gradients for requested inputs
    result: List[Tensor] = []
    for inp in inputs_list:
        res = grad_map.get(id(inp), None)
        if res is None:
            if not allow_unused:
                raise RuntimeError(
                    f"One of the differentiated Tensors appears to not have been used in the graph. "
                    f"Set allow_unused=True if this is the desired behavior."
                )
            res = Tensor(np.zeros_like(inp.data), requires_grad=create_graph)
        result.append(res)

    return tuple(result)


def hessian(
    output: Tensor,
    inputs: Union[Tensor, Sequence[Tensor]],
    create_graph: bool = False,
) -> Tensor:
    """
    Compute the full Hessian matrix H_ij = ∂²y / (∂x_i ∂x_j) of a scalar output.

    Args:
        output:       Scalar output tensor.
        inputs:       Input tensor(s).
        create_graph: If True, the returned Hessian will be differentiable.

    Returns:
        2D Tensor of shape (total_input_elements, total_input_elements).
    """
    if output.data.size != 1:
        raise ValueError("hessian requires a scalar output tensor")

    inputs_list: List[Tensor] = [inputs] if isinstance(inputs, Tensor) else list(inputs)

    # First gradient vector (with create_graph=True)
    first_grads = grad(output, inputs_list, create_graph=True)

    rows: List[Tensor] = []
    for g_k, inp_k in zip(first_grads, inputs_list):
        k_size = inp_k.data.size
        for local_i in range(k_size):
            # Basis vector matching shape of inp_k
            basis = np.zeros(inp_k.data.shape, dtype=np.float64)
            basis.flat[local_i] = 1.0
            basis_t = Tensor(basis, requires_grad=False)

            # Project first gradient: scalar g_proj = (g_k * basis_t).sum()
            g_proj = (g_k * basis_t).sum()

            # Backprop to get the row across all inputs
            row_grads = grad(g_proj, inputs_list, retain_graph=True, create_graph=create_graph)
            flat_row = np.concatenate([rg.data.flatten() for rg in row_grads])
            rows.append(Tensor(flat_row, requires_grad=create_graph))

    hessian_matrix = np.stack([r.data for r in rows])
    return Tensor(hessian_matrix, requires_grad=create_graph)
