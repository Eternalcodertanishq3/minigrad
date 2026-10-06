"""
graph.py — Computation graph utilities for miniGrad.

Provides topological sorting, graph visualization helpers, and
cycle detection for the dynamic computation graph built by Tensor operations.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from minigrad.tensor import Tensor


def topological_sort(root: Tensor) -> List[Tensor]:
    """
    Return a topologically sorted list of all tensors in the computation graph.
    Children appear before their parents (iterative post-order DFS).

    This guarantees that by the time we process a node during backprop,
    all gradients flowing into it have been fully accumulated, and graphs
    of arbitrary depth (> 1,000 layers) never hit Python recursion limits.
    """
    topo: List[Tensor] = []
    visited: Set[int] = set()
    stack: List[Tuple[Tensor, bool]] = [(root, False)]

    while stack:
        node, processed = stack.pop()
        node_id = id(node)
        if processed:
            topo.append(node)
            continue
        if node_id in visited:
            continue
        visited.add(node_id)
        stack.append((node, True))
        for child in reversed(node._prev):
            if id(child) not in visited:
                stack.append((child, False))

    return topo


def get_computation_graph(root: Tensor) -> Dict[int, Dict[str, Any]]:
    """
    Extract the full computation graph as a serializable dictionary.
    Useful for visualization and debugging.

    Returns:
        Dict mapping tensor id -> {tensor, op, shape, grad_shape, parents}
    """
    graph: Dict[int, Dict[str, Any]] = {}
    for node in topological_sort(root):
        node_id = id(node)
        graph[node_id] = {
            "tensor": node,
            "op": node._op,
            "shape": node.data.shape,
            "requires_grad": node.requires_grad,
            "parents": [id(p) for p in node._prev],
        }
    return graph


def print_graph(root: Tensor, max_depth: int = 10) -> None:
    """
    Pretty-print the computation graph starting from root.
    """
    visited: Set[int] = set()
    lines: List[str] = []

    def _print(node: Tensor, depth: int) -> None:
        if depth > max_depth:
            return
        node_id = id(node)
        prefix = "  " * depth + ("`-- " if depth > 0 else "")
        grad_info = f"  grad={node.grad.shape}" if node.requires_grad else ""
        op_info = f"  [{node._op}]" if node._op else "  [leaf]"
        lines.append(f"{prefix}Tensor{node.data.shape}{op_info}{grad_info}")

        if node_id in visited:
            lines.append("  " * (depth + 1) + "... (already shown)")
            return
        visited.add(node_id)

        for child in node._prev:
            _print(child, depth + 1)

    _print(root, 0)
    print("\n".join(lines))


def trace(root: Tensor) -> List[Tensor]:
    """
    Return all unique tensors in the computation graph in topological order.
    Alias for topological_sort with a more descriptive name.
    """
    return topological_sort(root)


def has_cycle(root: Tensor) -> bool:
    """
    Detect if the computation graph contains a cycle using iterative DFS.
    A proper autograd graph should always be a DAG (Directed Acyclic Graph).
    """
    GRAY, BLACK = 1, 2
    state: Dict[int, int] = defaultdict(int)
    stack: List[Tuple[Tensor, bool]] = [(root, False)]

    while stack:
        node, leaving = stack.pop()
        node_id = id(node)
        if leaving:
            state[node_id] = BLACK
            continue
        if state[node_id] == GRAY:
            return True
        if state[node_id] == BLACK:
            continue
        state[node_id] = GRAY
        stack.append((node, True))
        for child in reversed(node._prev):
            c_id = id(child)
            if state[c_id] == GRAY:
                return True
            if state[c_id] != BLACK:
                stack.append((child, False))

    return False


def detach(tensor: Tensor) -> Tensor:
    """
    Return a new Tensor with the same data but detached from the computation graph.
    No gradients will flow through this tensor.
    """
    return Tensor(tensor.data.copy(), requires_grad=False)


# Module-level flag
_grad_enabled = True


def is_grad_enabled() -> bool:
    return _grad_enabled


class no_grad:
    """Context manager and decorator that disables gradient computation.

    Usage:
        with no_grad():
            out = model(x)  # No graph built

        @no_grad()
        def evaluate(model, data):
            return model(data)
    """
    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self._prev

    def __call__(self, func):
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper
