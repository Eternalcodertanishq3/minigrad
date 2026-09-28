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

from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from minigrad.contracts import GRAPH_FREED_ERROR_MSG, GraphState
from minigrad.ops import concat, stack
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
            if p is None or p == 0:
                return (None,)
            p_val = float(p)
            deriv = (a ** (p_val - 1)) * p_val
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
        if a.data.ndim == 1 and b.data.ndim == 2:
            g_2d = g.reshape(1, -1) if g.data.ndim == 1 else g
            a_col = a.reshape(-1, 1)
            vjp_a = (g_2d @ b.transpose()).reshape(a.shape) if a.requires_grad else None
            vjp_b = (a_col @ g_2d) if b.requires_grad else None
            return (vjp_a, vjp_b)
        elif a.data.ndim == 2 and b.data.ndim == 1:
            g_2d = g.reshape(-1, 1) if g.data.ndim == 1 else g
            b_row = b.reshape(1, -1)
            vjp_a = (g_2d @ b_row) if a.requires_grad else None
            vjp_b = (a.transpose() @ g_2d).reshape(b.shape) if b.requires_grad else None
            return (vjp_a, vjp_b)
        else:
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
        mask = Tensor((a.data > 0).astype(a.data.dtype))
        return (g * mask if a.requires_grad else None,)

    elif op == "sigmoid":
        (a,) = children
        one = Tensor(np.ones_like(node.data))
        deriv = node * (one - node)
        return (g * deriv if a.requires_grad else None,)

    elif op == "gelu":
        (a,) = children
        c = np.sqrt(2.0 / np.pi)
        u = c * (a + 0.044715 * (a ** 3))
        tanh_u = u.tanh()
        sech2 = Tensor(np.ones_like(a.data, dtype=a.data.dtype)) - (tanh_u ** 2)
        du_dx = c * (Tensor(np.ones_like(a.data, dtype=a.data.dtype)) + 3.0 * 0.044715 * (a ** 2))
        deriv = 0.5 * (Tensor(np.ones_like(a.data, dtype=a.data.dtype)) + tanh_u) + 0.5 * a * sech2 * du_dx
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

        vjp_a = g_expanded * Tensor(np.ones(orig_shape, dtype=a.data.dtype)) if a.requires_grad else None
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

        vjp_a = (g_expanded * (1.0 / n)) * Tensor(np.ones(orig_shape, dtype=a.data.dtype)) if a.requires_grad else None
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

    elif op == "to":
        (a,) = children
        return (g.to(a.dtype) if a.requires_grad else None,)

    elif op == "getitem":
        (a,) = children
        idx = getattr(node, "_ctx", None)
        grad_a = np.zeros(a.shape, dtype=a.data.dtype)
        if idx is not None:
            np.add.at(grad_a, idx, g.data)
        vjp_a = Tensor(
            grad_a,
            requires_grad=a.requires_grad,
            _children=(g,) if g.requires_grad else (),
            _op="scatter",
            _ctx=(a.shape, idx),
        ) if a.requires_grad else None
        if vjp_a is not None and g.requires_grad:
            target_vjp = vjp_a
            def _backward() -> None:
                if g.requires_grad:
                    g.grad += target_vjp.grad[idx]
            vjp_a._backward = _backward
        return (vjp_a,)

    elif op == "scatter":
        (g_orig,) = children
        shape, idx = getattr(node, "_ctx", (None, None))
        return (g[idx] if g_orig.requires_grad else None,)

    elif op == "stack":
        axis = getattr(node, "_ctx", 0)
        norm_axis = axis if axis >= 0 else g.data.ndim + axis
        vjps: List[Optional[Tensor]] = []
        for i, child in enumerate(children):
            if child.requires_grad:
                idx = [slice(None)] * g.data.ndim
                idx[norm_axis] = i
                vjps.append(g[tuple(idx)])
            else:
                vjps.append(None)
        return tuple(vjps)

    elif op == "concat":
        axis = getattr(node, "_ctx", 0)
        norm_axis = axis if axis >= 0 else g.data.ndim + axis
        vjps_c: List[Optional[Tensor]] = []
        offset = 0
        for child in children:
            length = child.shape[norm_axis]
            if child.requires_grad:
                idx = [slice(None)] * g.data.ndim
                idx[norm_axis] = slice(offset, offset + length)
                vjps_c.append(g[tuple(idx)])
            else:
                vjps_c.append(None)
            offset += length
        return tuple(vjps_c)

    elif op in ("fused_linear", "fused_linear_relu"):
        x = children[0]
        weight = children[1]
        bias = children[2] if len(children) == 3 else None

        if op == "fused_linear_relu":
            z_data = getattr(node, "_ctx", None)
            if z_data is not None:
                mask = Tensor((z_data.reshape(node.shape) > 0.0).astype(g.data.dtype), dtype=g.data.dtype)
            else:
                mask = Tensor((node.data > 0.0).astype(g.data.dtype), dtype=g.data.dtype)
            g_eff = g * mask
        else:
            g_eff = g

        if x.data.ndim == 1 and weight.data.ndim == 2:
            g_2d = g_eff.reshape(1, -1) if g_eff.data.ndim == 1 else g_eff
            x_col = x.reshape(-1, 1)
            vjp_x = (g_2d @ weight.transpose()).reshape(x.shape) if x.requires_grad else None
            vjp_weight = (x_col @ g_2d) if weight.requires_grad else None
        elif x.data.ndim > 2 and weight.data.ndim == 2:
            g_2d = g_eff.reshape(-1, weight.shape[1])
            x_2d = x.reshape(-1, x.shape[-1])
            vjp_x = (g_2d @ weight.transpose()).reshape(x.shape) if x.requires_grad else None
            vjp_weight = (x_2d.transpose() @ g_2d) if weight.requires_grad else None
        else:
            vjp_x = (g_eff @ weight.transpose()) if x.requires_grad else None
            vjp_weight = (x.transpose() @ g_eff) if weight.requires_grad else None

        vjps_f: List[Optional[Tensor]] = [vjp_x, vjp_weight]
        if bias is not None:
            vjp_bias = unbroadcast_tensor(g_eff, bias.shape) if bias.requires_grad else None
            vjps_f.append(vjp_bias)
        return tuple(vjps_f)

    elif op == "abs":
        (a,) = children
        return (g * Tensor(np.sign(a.data).astype(a.data.dtype), dtype=a.data.dtype) if a.requires_grad else None,)

    elif op == "clip":
        (a,) = children
        ctx = getattr(node, "_ctx", None)
        min_val, max_val = ctx if ctx is not None else (-np.inf, np.inf)
        mask_clip = Tensor(((a.data >= min_val) & (a.data <= max_val)).astype(a.data.dtype), dtype=a.data.dtype)
        return (g * mask_clip if a.requires_grad else None,)

    elif op == "softmax":
        (a,) = children
        axis = getattr(node, "_ctx", -1)
        if axis is None:
            axis = -1
        p = node
        vjp_a = p * (g - (p * g).sum(axis=axis, keepdims=True)) if a.requires_grad else None
        return (vjp_a,)

    elif op == "log_softmax":
        (a,) = children
        axis = getattr(node, "_ctx", -1)
        if axis is None:
            axis = -1
        p = node.exp()
        vjp_a = (g - p * g.sum(axis=axis, keepdims=True)) if a.requires_grad else None
        return (vjp_a,)

    elif op == "max":
        (a,) = children
        ctx = getattr(node, "_ctx", None)
        axis, keepdims = ctx if ctx is not None else (None, False)
        mask_max = (a.data == np.max(a.data, axis=axis, keepdims=True)).astype(a.data.dtype)
        count = mask_max.sum(axis=axis, keepdims=True)
        if axis is not None and not keepdims:
            expanded_shape = list(a.shape)
            expanded_shape[axis] = 1
            g_expanded = g.reshape(*expanded_shape)
        else:
            g_expanded = g
        vjp_a = g_expanded * Tensor(mask_max / count, dtype=a.data.dtype) if a.requires_grad else None
        return (vjp_a,)

    elif op == "leaky_relu":
        (a,) = children
        negative_slope = getattr(node, "_ctx", 0.01)
        if negative_slope is None:
            negative_slope = 0.01
        mask_lr = (a.data > 0).astype(a.data.dtype) + float(negative_slope) * (a.data <= 0).astype(a.data.dtype)
        return (g * Tensor(mask_lr, dtype=a.data.dtype) if a.requires_grad else None,)

    elif op == "elu":
        (a,) = children
        alpha = getattr(node, "_ctx", 1.0)
        if alpha is None:
            alpha = 1.0
        deriv_elu = np.where(a.data > 0, 1.0, float(alpha) * np.exp(a.data)).astype(a.data.dtype)
        return (g * Tensor(deriv_elu, dtype=a.data.dtype) if a.requires_grad else None,)

    # Explicit error for unregistered operations
    raise NotImplementedError(f"No VJP registered for operation '{op}'")


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

    for out in outputs_list:
        if out._lifecycle == GraphState.FREED:
            raise RuntimeError(GRAPH_FREED_ERROR_MSG)

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

    for node in topo:
        if node not in outputs_list and node._prev and node._lifecycle == GraphState.FREED:
            raise RuntimeError(GRAPH_FREED_ERROR_MSG)

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
                    "One of the differentiated Tensors appears to not have been used in the graph. "
                    "Set allow_unused=True if this is the desired behavior."
                )
            res = Tensor(np.zeros_like(inp.data), dtype=inp.data.dtype, requires_grad=create_graph)
        elif res.dtype != inp.dtype:
            res = res.to(inp.dtype)
        result.append(res)

    if not retain_graph and not create_graph:
        for out in outputs_list:
            out._lifecycle = GraphState.FREED
            out._backward = lambda: None

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

    # First gradient vector (with create_graph=True, retain_graph=True)
    first_grads = grad(output, inputs_list, create_graph=True, retain_graph=True)

    rows: List[Tensor] = []
    for g_k, inp_k in zip(first_grads, inputs_list):
        k_size = inp_k.data.size
        for local_i in range(k_size):
            # Basis vector matching shape of inp_k
            basis = np.zeros(inp_k.data.shape, dtype=inp_k.data.dtype)
            basis.flat[local_i] = 1.0
            basis_t = Tensor(basis, dtype=inp_k.data.dtype, requires_grad=False)

            # Project first gradient: scalar g_proj = (g_k * basis_t).sum()
            g_proj = (g_k * basis_t).sum()

            # Backprop to get the row across all inputs
            row_grads = grad(
                g_proj,
                inputs_list,
                retain_graph=True,
                create_graph=create_graph,
                allow_unused=True,
            )
            if create_graph:
                flat_parts = [rg.reshape(-1) for rg in row_grads]
                if len(flat_parts) == 1:
                    flat_row = flat_parts[0]
                else:
                    flat_row = concat(flat_parts, axis=0)
                rows.append(flat_row)
            else:
                flat_row_data = np.concatenate([rg.data.flatten() for rg in row_grads])
                rows.append(Tensor(flat_row_data, dtype=flat_row_data.dtype, requires_grad=False))

    if create_graph:
        return stack(rows, axis=0)
    else:
        hessian_matrix = np.stack([r.data for r in rows], axis=0)
        return Tensor(hessian_matrix, dtype=hessian_matrix.dtype, requires_grad=False)
