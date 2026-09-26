"""
vmap.py — Pure Functional Vectorizing Map, Per-Sample Gradients & Batched Jacobians (Pillar 4).

Provides:
- vmap(): Automatically vectorizes any function over batch dimensions without manual loops.
- make_functional(): Converts any stateful miniGrad Module into a pure, stateless callable.
- per_sample_gradients(): Computes per-sample parameter gradients [B, *shape] in a single pass.
- jacrev(): Computes reverse-mode Jacobians of vector-valued functions.
- batched_jacobian(): Computes full Jacobian tensors [B, M, N] across batched inputs.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from minigrad.autograd import grad
from minigrad.nn.module import Module
from minigrad.ops import stack
from minigrad.tensor import Tensor

# ── Helper Utilities ────────────────────────────────────────────────

def _get_batch_size(arg: Any, axis: Optional[int]) -> Optional[int]:
    """Determine the batch dimension size of an argument."""
    if axis is None or arg is None:
        return None
    if isinstance(arg, Tensor):
        return arg.shape[axis]
    if isinstance(arg, np.ndarray):
        return arg.shape[axis]
    if isinstance(arg, (list, tuple)) and axis == 0:
        return len(arg)
    return None


def _slice_arg(arg: Any, axis: Optional[int], index: int) -> Any:
    """Extract slice `index` along `axis` from `arg`. If axis is None, return arg unmodified."""
    if axis is None or arg is None:
        return arg
    if isinstance(arg, Tensor):
        if axis == 0:
            return arg[index]
        idx: List[Any] = [slice(None)] * arg.ndim
        idx[axis] = index
        return arg[tuple(idx)]
    if isinstance(arg, np.ndarray):
        if axis == 0:
            return arg[index]
        idx_arr: List[Any] = [slice(None)] * arg.ndim
        idx_arr[axis] = index
        return arg[tuple(idx_arr)]
    if isinstance(arg, (list, tuple)) and axis == 0:
        return arg[index]
    return arg


def _stack_outputs(results: List[Any], axis: int = 0) -> Any:
    """Stack a list of results across the batch dimension."""
    if not results:
        return None

    first = results[0]

    # 1. Single Tensor output
    if isinstance(first, Tensor):
        return stack(results, axis=axis)

    # 2. NumPy array output
    if isinstance(first, np.ndarray):
        return np.stack(results, axis=axis)

    # 3. Numeric scalar output (float, int)
    if isinstance(first, (int, float, np.number)):
        return Tensor(np.stack(results, axis=axis))

    # 4. Tuple of Tensors / outputs
    if isinstance(first, tuple):
        num_items = len(first)
        stacked_items = []
        for i in range(num_items):
            item_list = [r[i] for r in results]
            stacked_items.append(_stack_outputs(item_list, axis=axis))
        return tuple(stacked_items)

    # 5. List of outputs
    if isinstance(first, list):
        num_items = len(first)
        stacked_items = []
        for i in range(num_items):
            item_list = [r[i] for r in results]
            stacked_items.append(_stack_outputs(item_list, axis=axis))
        return stacked_items

    # 6. Dict of Tensors (e.g., parameter dictionary gradients)
    if isinstance(first, dict):
        stacked_dict: Dict[str, Any] = {}
        for k in first.keys():
            item_list = [r[k] for r in results]
            stacked_dict[k] = _stack_outputs(item_list, axis=axis)
        return stacked_dict

    # Fallback: return as list
    return results


# ── Core Vectorizing Map Transform (vmap) ───────────────────────────

def vmap(
    func: Callable,
    in_axes: Union[int, Sequence[Optional[int]]] = 0,
    out_axes: int = 0,
) -> Callable:
    """
    Vectorizing Map (vmap) transform.

    Transforms a function `func` designed to work on a single example into a function
    that operates over batches of examples along the specified `in_axes`.

    Args:
        func:     The callable to vectorize. Signature: `(*args, **kwargs) -> out`.
        in_axes:  Integer or sequence specifying the batch axis for each positional argument.
                  Use `None` for unbatched/shared arguments (e.g. model parameters).
                  Defaults to 0 (all positional arguments are batched along dimension 0).
        out_axes: The axis along which the mapped outputs should be stacked. Defaults to 0.

    Returns:
        Vectorized callable.

    Example:
        >>> # Vectorize a vector-matrix product:
        >>> f = lambda w, x: x @ w
        >>> batched_f = vmap(f, in_axes=(None, 0)) # w is shared, x is batched
        >>> out = batched_f(w, batch_x)
    """
    @functools.wraps(func)
    def vmapped_fn(*args: Any, **kwargs: Any) -> Any:
        # 1. Normalize in_axes to match positional args
        axes_list: List[Optional[int]]
        if isinstance(in_axes, int):
            axes_list = [in_axes] * len(args)
        else:
            axes_list = list(in_axes)
            if len(axes_list) < len(args):
                # Pad remaining args with None (unbatched)
                axes_list.extend([None] * (len(args) - len(axes_list)))

        # 2. Determine batch size B
        batch_size: Optional[int] = None
        for arg, ax in zip(args, axes_list):
            bs = _get_batch_size(arg, ax)
            if bs is not None:
                if batch_size is None:
                    batch_size = bs
                elif batch_size != bs:
                    raise ValueError(
                        f"Inconsistent batch sizes in vmap: expected {batch_size}, got {bs}"
                    )

        if batch_size is None:
            # No batched arguments found; invoke func directly
            return func(*args, **kwargs)

        # 3. Vectorized execution across batch items
        outputs: List[Any] = []
        for i in range(batch_size):
            slice_args = tuple(_slice_arg(arg, ax, i) for arg, ax in zip(args, axes_list))
            res = func(*slice_args, **kwargs)
            outputs.append(res)

        # 4. Stack results along out_axes
        return _stack_outputs(outputs, axis=out_axes)

    return vmapped_fn


# ── Stateless Functional Module Conversion ──────────────────────────

def make_functional(
    module: Module,
    return_dict: bool = True,
) -> Tuple[Callable, Union[Dict[str, Tensor], Tuple[Tensor, ...]]]:
    """
    Converts a stateful miniGrad Module into a pure, stateless callable.

    Args:
        module:      The miniGrad Module to functionalize.
        return_dict: If True, returns parameters as a {name: Tensor} dictionary.
                     If False, returns parameters as a (Tensor, ...) tuple.

    Returns:
        Tuple of (functional_forward_fn, params), where:
        - functional_forward_fn(params, *inputs, **kwargs) executes forward pass statelessly.
        - params is the initial parameters dictionary or tuple.

    Example:
        >>> model = Linear(10, 2)
        >>> f_model, params = make_functional(model)
        >>> out = f_model(params, x)
    """
    param_names: List[str] = [name for name, _ in module.named_parameters()]

    def _set_nested_attr(obj: Any, name: str, value: Tensor) -> None:
        parts = name.split(".")
        curr = obj
        for part in parts[:-1]:
            if part.isdigit():
                curr = curr[int(part)]
            elif "[" in part and part.endswith("]"):
                base, idx_str = part[:-1].split("[")
                curr = getattr(curr, base)[int(idx_str)]
            else:
                curr = getattr(curr, part)
        last = parts[-1]
        if last.isdigit():
            curr[int(last)] = value
        elif "[" in last and last.endswith("]"):
            base, idx_str = last[:-1].split("[")
            getattr(curr, base)[int(idx_str)] = value
        else:
            setattr(curr, last, value)

    def functional_forward(params: Union[Dict[str, Tensor], Sequence[Tensor]], *inputs: Any, **kwargs: Any) -> Any:
        # Map params to names
        if isinstance(params, dict):
            new_params = params
        else:
            new_params = dict(zip(param_names, params))

        # Save existing module parameters
        orig_params: Dict[str, Tensor] = dict(module.named_parameters())

        try:
            # Bind new parameter tensors
            for name, p_tensor in new_params.items():
                _set_nested_attr(module, name, p_tensor)

            # Execute forward pass with new parameters
            out = module(*inputs, **kwargs)
            return out
        finally:
            # Restore original parameters to module
            for name, orig_tensor in orig_params.items():
                _set_nested_attr(module, name, orig_tensor)

    initial_params = dict(module.named_parameters()) if return_dict else tuple(module.parameters())
    return functional_forward, initial_params


# ── Functional Gradient Transform ───────────────────────────────────

def functional_grad(
    func: Callable,
    argnums: int = 0,
) -> Callable:
    """
    Functional gradient transform.
    Returns a callable computing the gradient of `func` with respect to the argument at `argnums`.
    Supports single Tensor, tuple/list of Tensors, and dictionary of Tensors.
    """
    @functools.wraps(func)
    def grad_fn(*args: Any, **kwargs: Any) -> Any:
        target = args[argnums]
        if isinstance(target, dict):
            keys = list(target.keys())
            target_tensors = [target[k] for k in keys]
            out = func(*args, **kwargs)
            grads = grad(out, target_tensors, allow_unused=True)
            return dict(zip(keys, grads))
        elif isinstance(target, (tuple, list)):
            target_tensors = list(target)
            out = func(*args, **kwargs)
            grads = grad(out, target_tensors, allow_unused=True)
            return type(target)(grads)
        elif isinstance(target, Tensor):
            out = func(*args, **kwargs)
            grads = grad(out, target, allow_unused=True)
            return grads[0]
        else:
            raise TypeError(f"Target at argnums={argnums} must be Tensor, tuple, or dict of Tensors, got {type(target)}")

    return grad_fn


# ── Per-Sample Gradient Engine ──────────────────────────────────────

def per_sample_gradients(
    func_or_module: Union[Callable, Module],
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    params: Optional[Union[Dict[str, Tensor], Tuple[Tensor, ...]]] = None,
    *batched_inputs: Tensor,
) -> Union[Dict[str, Tensor], Tuple[Tensor, ...]]:
    """
    Computes per-sample gradients [B, *param.shape] for all parameters in a single vectorized pass.

    Unlike standard backpropagation which sums gradients across the batch,
    per_sample_gradients preserves sample-level gradient vectors, enabling
    Differential Privacy (DP-SGD), per-sample clipping, and outlier detection.

    Args:
        func_or_module:  A callable f(params, x) or a miniGrad Module instance.
        loss_fn:         Criterion (pred, target) -> scalar Tensor.
        params:          Parameters (dict or tuple). If func_or_module is a Module
                         and params is None, parameters are automatically extracted.
        *batched_inputs: Batched inputs. The last argument is treated as the target labels Y,
                         and preceding arguments are inputs to func_or_module.

    Returns:
        Dictionary or tuple of per-sample gradient Tensors, each with leading dimension B.
    """
    if len(batched_inputs) < 2:
        raise ValueError("per_sample_gradients requires at least one feature input and one target label")

    if isinstance(func_or_module, Module):
        is_dict = not isinstance(params, (tuple, list))
        f_model, mod_params = make_functional(func_or_module, return_dict=is_dict)
        if params is None:
            params = mod_params
    else:
        f_model = func_or_module
        if params is None:
            raise ValueError("params must be provided when func_or_module is a callable")

    batched_x = batched_inputs[:-1]
    batched_y = batched_inputs[-1]

    def sample_loss(p: Any, *sample_inps: Tensor) -> Tensor:
        sample_features = sample_inps[:-1]
        sample_target = sample_inps[-1]
        pred = f_model(p, *sample_features)
        return loss_fn(pred, sample_target)

    grad_fn = functional_grad(sample_loss, argnums=0)
    in_axes = (None,) + tuple(0 for _ in batched_inputs)
    vmapped_grad = vmap(grad_fn, in_axes=in_axes)
    return vmapped_grad(params, *batched_x, batched_y)


# ── Reverse-Mode Jacobian & Batched Jacobian ────────────────────────

def jacrev(func: Callable, argnums: int = 0) -> Callable:
    """
    Computes the reverse-mode Jacobian of a vector-valued function f: R^N -> R^M.

    Args:
        func:    A callable with signature f(*args) -> Tensor.
        argnums: Specifies which argument to differentiate with respect to.

    Returns:
        A callable computing the Jacobian tensor of shape (*out.shape, *arg.shape).

    Example:
        >>> f = lambda x: x ** 2
        >>> J = jacrev(f)(Tensor([2.0, 3.0]))
        >>> # J has shape (2, 2) with diagonal [4.0, 6.0]
    """
    @functools.wraps(func)
    def jacobian_fn(*args: Any, **kwargs: Any) -> Tensor:
        target_arg = args[argnums]
        if not isinstance(target_arg, Tensor):
            raise TypeError(f"Target argument at index {argnums} must be a Tensor")

        # Ensure target_arg tracks gradients for Jacobian evaluation
        args_list = list(args)
        if not target_arg.requires_grad:
            target_arg = Tensor(target_arg.data.copy(), requires_grad=True)
            args_list[argnums] = target_arg

        # Forward pass
        out = func(*args_list, **kwargs)
        if not isinstance(out, Tensor):
            raise TypeError("jacrev expects function to return a Tensor")

        out_shape = out.shape
        out_size = out.data.size
        in_shape = target_arg.shape

        # Compute VJPs for each unit basis vector
        basis_rows: List[Tensor] = []
        for i in range(out_size):
            e_k = np.zeros(out_shape, dtype=out.data.dtype)
            e_k.flat[i] = 1.0
            e_tensor = Tensor(e_k)

            vjp = grad(out, target_arg, grad_outputs=e_tensor, retain_graph=(i < out_size - 1), allow_unused=True)[0]
            basis_rows.append(vjp)

        # Stack into Jacobian matrix
        stacked = stack(basis_rows, axis=0)
        return stacked.reshape(*out_shape, *in_shape)

    return jacobian_fn


def batched_jacobian(func: Callable, *batched_inputs: Tensor, argnums: int = 0) -> Tensor:
    """
    Computes batched Jacobian tensors [B, M, N] across a batch of inputs in a single call.

    Args:
        func:            Vector-valued function f(x) -> y.
        *batched_inputs: Inputs with batch dimension 0.
        argnums:         Index of argument to differentiate w.r.t.

    Returns:
        Tensor of shape [B, *out_shape, *in_shape].
    """
    j_fn = jacrev(func, argnums=argnums)
    in_axes = tuple(0 if i == argnums else 0 for i in range(len(batched_inputs)))
    vmapped_j = vmap(j_fn, in_axes=in_axes)
    return vmapped_j(*batched_inputs)
