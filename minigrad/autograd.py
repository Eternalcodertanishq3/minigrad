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

    elif op == "div":
        a, b = children
        vjp_a = unbroadcast_tensor(g / b, a.shape) if a.requires_grad else None
        vjp_b = unbroadcast_tensor(-g * a / (b ** 2), b.shape) if b.requires_grad else None
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
        if a.data.ndim == 1 and b.data.ndim == 1:
            vjp_a = unbroadcast_tensor(g * b, a.shape) if a.requires_grad else None
            vjp_b = unbroadcast_tensor(g * a, b.shape) if b.requires_grad else None
            return (vjp_a, vjp_b)
        elif a.data.ndim == 1 and b.data.ndim >= 2:
            g_exp = g.reshape(*g.shape[:-1], 1, g.shape[-1])
            a_exp = a.reshape(1, -1)
            vjp_a = (
                unbroadcast_tensor((g_exp @ b.swapaxes(-1, -2)).reshape(*g.shape[:-1], a.shape[0]), a.shape)
                if a.requires_grad else None
            )
            vjp_b = unbroadcast_tensor(a_exp.swapaxes(-1, -2) @ g_exp, b.shape) if b.requires_grad else None
            return (vjp_a, vjp_b)
        elif a.data.ndim >= 2 and b.data.ndim == 1:
            g_exp = g.reshape(*g.shape, 1)
            b_exp = b.reshape(1, -1)
            vjp_a = unbroadcast_tensor(g_exp @ b_exp, a.shape) if a.requires_grad else None
            vjp_b = (
                unbroadcast_tensor((a.swapaxes(-1, -2) @ g_exp).reshape(*g.shape, b.shape[0]), b.shape)
                if b.requires_grad else None
            )
            return (vjp_a, vjp_b)
        else:
            vjp_a = unbroadcast_tensor(g @ b.swapaxes(-1, -2), a.shape) if a.requires_grad else None
            vjp_b = unbroadcast_tensor(a.swapaxes(-1, -2) @ g, b.shape) if b.requires_grad else None
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
        c = float(np.sqrt(2.0 / np.pi))
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

    elif op in ("getitem", "embedding"):
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
            import weakref
            vjp_ref = weakref.ref(vjp_a)
            def _backward() -> None:
                tv = vjp_ref()
                if tv is not None and g.requires_grad:
                    g.grad += tv.grad[idx]
            vjp_a._backward = _backward
        return (vjp_a,)

    elif op == "scatter":
        (g_orig,) = children
        shape, idx = getattr(node, "_ctx", (None, None))
        return (g[idx] if g_orig.requires_grad else None,)

    elif op == "split":
        (a,) = children
        norm_axis, start, end = getattr(node, "_ctx", (0, 0, a.shape[0]))
        slices = tuple(slice(start, end) if i == norm_axis else slice(None) for i in range(a.data.ndim))
        grad_a = np.zeros(a.shape, dtype=a.data.dtype)
        grad_a[slices] = g.data
        vjp_a = Tensor(
            grad_a,
            requires_grad=a.requires_grad,
            _children=(g,) if g.requires_grad else (),
            _op="scatter",
            _ctx=(a.shape, slices),
        ) if a.requires_grad else None
        if vjp_a is not None and g.requires_grad:
            import weakref
            vjp_ref = weakref.ref(vjp_a)
            def _backward_split() -> None:
                tv = vjp_ref()
                if tv is not None and g.requires_grad:
                    g.grad += tv.grad[slices]
            vjp_a._backward = _backward_split
        return (vjp_a,)

    elif op == "pad":
        (a,) = children
        pad_width = getattr(node, "_ctx", None)
        if not a.requires_grad or pad_width is None:
            return (None,)
        slices_pad = []
        for p in pad_width:
            if isinstance(p, int):
                slices_pad.append(slice(p, -p if p > 0 else None))
            else:
                slices_pad.append(slice(p[0], -p[1] if p[1] > 0 else None))
        return (g[tuple(slices_pad)],)

    elif op == "einsum":
        from minigrad.ops import einsum as ops_einsum
        subscripts = getattr(node, "_ctx", None)
        if subscripts is None:
            raise NotImplementedError("einsum node missing subscript context for VJP")
        if "->" in subscripts:
            input_subs, output_sub = subscripts.split("->")
        else:
            input_subs = subscripts
            output_sub = ""
        input_sub_list = input_subs.split(",")
        vjps_ein: List[Optional[Tensor]] = []
        for i, op_i in enumerate(children):
            if not op_i.requires_grad:
                vjps_ein.append(None)
                continue
            target_sub = input_sub_list[i]
            if len(set(target_sub)) < len(target_sub) and output_sub == "":
                dim = op_i.data.shape[0]
                vjps_ein.append(g * Tensor(np.eye(dim, dtype=op_i.data.dtype)))
                continue
            backward_inputs = [output_sub]
            backward_tensors = [g]
            for j, op_j in enumerate(children):
                if j != i:
                    backward_inputs.append(input_sub_list[j])
                    backward_tensors.append(op_j)
            backward_subscripts = ",".join(backward_inputs) + "->" + target_sub
            vjps_ein.append(ops_einsum(backward_subscripts, *backward_tensors))
        return tuple(vjps_ein)

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

    elif op == "layer_norm":
        x, gamma, beta = children
        axes, eps = getattr(node, "_ctx", ((x.data.ndim - 1,), 1e-5))
        ndim = len(axes)
        N = 1
        for ax in axes:
            N *= x.data.shape[ax]
        mean = x.mean(axis=axes, keepdims=True)
        diff = x - mean
        var = (diff ** 2).mean(axis=axes, keepdims=True)
        std_inv = (var + eps) ** -0.5
        x_norm = diff * std_inv

        reduce_axes = tuple(range(x.data.ndim - ndim))
        vjp_gamma = (
            (g * x_norm).sum(axis=reduce_axes, keepdims=False)
            if (gamma.requires_grad and x.data.ndim > ndim)
            else ((g * x_norm) if gamma.requires_grad else None)
        )
        vjp_beta = (
            g.sum(axis=reduce_axes, keepdims=False)
            if (beta.requires_grad and x.data.ndim > ndim)
            else (g if beta.requires_grad else None)
        )
        if x.requires_grad:
            dx_norm = g * gamma
            dx_var = (dx_norm * diff * -0.5 * (std_inv ** 3)).sum(axis=axes, keepdims=True)
            dx_mean = (dx_norm * -std_inv).sum(axis=axes, keepdims=True)
            vjp_x = dx_norm * std_inv + (2.0 / N) * diff * dx_var + dx_mean * (1.0 / N)
        else:
            vjp_x = None
        return (vjp_x, vjp_gamma, vjp_beta)

    elif op == "cross_entropy":
        from minigrad.ops import softmax as ops_softmax
        (logits,) = children
        targets, reduction = getattr(node, "_ctx", (None, "mean"))
        if not logits.requires_grad or targets is None:
            return (None,)
        N = logits.data.shape[0]
        probs = ops_softmax(logits, axis=1)
        one_hot = np.zeros_like(logits.data)
        one_hot[np.arange(N), targets] = 1.0
        grad_ce = probs - Tensor(one_hot, dtype=logits.dtype)
        if reduction == "mean":
            grad_ce = grad_ce * (1.0 / N)
        return (grad_ce * g,)

    elif op == "bce":
        (pred,) = children
        t, reduction, eps = getattr(node, "_ctx", (None, "mean", 1e-7))
        if not pred.requires_grad or t is None:
            return (None,)
        from minigrad.ops import clip as ops_clip
        p = ops_clip(pred, eps, 1.0 - eps)
        t_tensor = Tensor(t, dtype=pred.dtype)
        grad_bce = -(t_tensor / p - (Tensor(np.ones_like(pred.data, dtype=pred.dtype)) - t_tensor) / (Tensor(np.ones_like(pred.data, dtype=pred.dtype)) - p))
        if reduction == "mean":
            grad_bce = grad_bce * (1.0 / pred.data.size)
        return (grad_bce * g,)

    elif op == "bce_with_logits":
        (logits,) = children
        t, reduction = getattr(node, "_ctx", (None, "mean"))
        if not logits.requires_grad or t is None:
            return (None,)
        grad_bcel = logits.sigmoid() - Tensor(t, dtype=logits.dtype)
        if reduction == "mean":
            grad_bcel = grad_bcel * (1.0 / logits.data.size)
        return (grad_bcel * g,)

    elif op == "nll":
        (log_probs,) = children
        targets, reduction = getattr(node, "_ctx", (None, "mean"))
        if not log_probs.requires_grad or targets is None:
            return (None,)
        N = log_probs.data.shape[0]
        grad_nll = np.zeros_like(log_probs.data)
        grad_nll[np.arange(N), targets] = -1.0
        if reduction == "mean":
            grad_nll = grad_nll / N
        return (Tensor(grad_nll, dtype=log_probs.dtype) * g,)

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

    elif op == "dropout":
        (a,) = children
        keep_scale = getattr(node, "_ctx", None)  # mask * 1/(1-p), recorded by Dropout.forward
        if keep_scale is None:
            raise NotImplementedError("Dropout node has no recorded mask; cannot build its double-backward rule.")
        return (g * Tensor(keep_scale, dtype=a.data.dtype) if a.requires_grad else None,)

    # Explicit, actionable error for operations without a differentiable (second-order) rule
    raise NotImplementedError(
        f"No VJP registered for operation '{op}'. "
        f"Double-backward (create_graph=True / hessian) is not implemented for it. "
        f"First-order backward() works for it, but its derivative is not itself differentiable. "
        f"Ops known to lack a second-order rule: conv2d, batch_norm_1d/2d, surrogate_spike and the custom "
        f"AVYAYA/SUTRA operators. Express the computation with differentiable primitives (matmul, "
        f"elementwise ops, layer_norm, einsum, ...) or avoid create_graph through this op."
    )


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

    # Build iterative topological sort from outputs
    topo: List[Tensor] = []
    visited: Set[int] = set()

    for out in outputs_list:
        if id(out) in visited:
            continue
        stack_dfs: List[Tuple[Tensor, bool]] = [(out, False)]
        while stack_dfs:
            node, processed = stack_dfs.pop()
            nid = id(node)
            if processed:
                topo.append(node)
                continue
            if nid in visited:
                continue
            visited.add(nid)
            stack_dfs.append((node, True))
            for child in reversed(list(node._prev)):
                if id(child) not in visited:
                    stack_dfs.append((child, False))

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
        from minigrad.tensor import _noop_backward
        for out in outputs_list:
            out._lifecycle = GraphState.FREED
            out._backward = _noop_backward

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
