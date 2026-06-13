"""
graph.py — Computation graph utilities for miniGrad.

Provides topological sorting, graph visualization helpers, and
cycle detection for the dynamic computation graph built by Tensor operations.
"""
from __future__ import annotations

from typing import List, Set, Callable, Dict, Any
from collections import defaultdict

from minigrad.tensor import Tensor


def topological_sort(root: Tensor) -> List[Tensor]:
    """
    Return a topologically sorted list of all tensors in the computation graph.
    Children appear before their parents (post-order DFS).

    This guarantees that by the time we process a node during backprop,
    all gradients flowing into it have been fully accumulated.
    """
    topo: List[Tensor] = []
    visited: Set[int] = set()

    def _visit(node: Tensor) -> None:
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)
        for child in node._prev:
            _visit(child)
        topo.append(node)

    _visit(root)
    return topo


def get_computation_graph(root: Tensor) -> Dict[int, Dict[str, Any]]:
    """
    Extract the full computation graph as a serializable dictionary.
    Useful for visualization and debugging.

    Returns:
        Dict mapping tensor id -> {tensor, op, shape, grad_shape, parents}
    """
    graph: Dict[int, Dict[str, Any]] = {}
    visited: Set[int] = set()

    def _build(node: Tensor) -> None:
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        for child in node._prev:
            _build(child)

        graph[node_id] = {
            "tensor": node,
            "op": node._op,
            "shape": node.data.shape,
            "requires_grad": node.requires_grad,
            "parents": [id(p) for p in node._prev],
        }

    _build(root)
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
    Detect if the computation graph contains a cycle.
    A proper autograd graph should always be a DAG (Directed Acyclic Graph).
    """
    GRAY, BLACK = 1, 2
    state: Dict[int, int] = defaultdict(int)

    def _dfs(node: Tensor) -> bool:
        node_id = id(node)
        if state[node_id] == GRAY:
            return True  # Back edge = cycle
        if state[node_id] == BLACK:
            return False

        state[node_id] = GRAY
        for child in node._prev:
            if _dfs(child):
                return True
        state[node_id] = BLACK
        return False

    return _dfs(root)


def detach(tensor: Tensor) -> Tensor:
    """
    Return a new Tensor with the same data but detached from the computation graph.
    No gradients will flow through this tensor.
    """
    return Tensor(tensor.data.copy(), requires_grad=False)


def no_grad(func: Callable) -> Callable:
    """
    Decorator that disables gradient computation inside a function.
    Useful for evaluation/inference code.

    Usage:
        @no_grad
        def evaluate(model, data):
            return model(data)
    """
    def wrapper(*args, **kwargs):
        # Save original requires_grad states
        tensors = [a for a in args if isinstance(a, Tensor)]
        original = [t.requires_grad for t in tensors]

        for t in tensors:
            t.requires_grad = False

        result = func(*args, **kwargs)

        for t, req in zip(tensors, original):
            t.requires_grad = req

        return result

    return wrapper
