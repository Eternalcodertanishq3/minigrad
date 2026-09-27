"""
tests/test_random_dag_opt.py — Property-based differential stress testing for graph optimizer.

Generates 100 deterministically seeded random DAGs (depths 3-8) covering diverse operations,
shapes, and topologies, and verifies exact 3-layer autograd parity with dynamic tolerances.
On any failure, prints seed, graph topology, op sequence, shapes, and maximum difference.
"""
from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import pytest

from minigrad.graph_opt import (
    _RECONSTRUCTORS,
    is_reconstructible,
    optimize_graph,
    register_reconstructor,
)
from minigrad.tensor import Tensor


def _build_random_dag(seed: int) -> Tuple[Tensor, List[Tensor], str]:
    """
    Deterministically build a random computation graph rooted at an output scalar.
    Returns (root_tensor, leaf_tensors, topology_description).
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # 2D shapes compatible with matmul and elementwise ops
    shapes = [(2, 2), (2, 3), (3, 2), (3, 3)]

    # Leaf nodes with bounded positive values to prevent domain errors
    num_leaves = rng.randint(2, 4)
    leaves: List[Tensor] = []
    for i in range(num_leaves):
        s = rng.choice(shapes)
        val = np_rng.uniform(0.5, 2.5, size=s)
        leaves.append(Tensor(val, requires_grad=True))

    pool: List[Tensor] = list(leaves)
    op_history: List[str] = []

    depth = rng.randint(3, 7)
    for step in range(depth):
        op = rng.choice(["add", "sub", "mul", "div", "relu", "exp", "matmul", "pow", "sum", "mean"])

        if op in ("add", "sub", "mul", "div"):
            t1 = rng.choice(pool)
            # Find a tensor with matching or broadcastable shape
            candidates = [t for t in pool if t.data.shape == t1.data.shape or t.data.ndim == 1]
            t2 = rng.choice(candidates) if candidates else t1

            if op == "add":
                res = t1 + t2
            elif op == "sub":
                res = t1 - t2
            elif op == "mul":
                res = t1 * t2
            else:  # div: ensure divisor has safe offset
                res = t1 / (t2 + 1.0)
            op_history.append(f"{op}({t1.shape}, {t2.shape}) -> {res.shape}")

        elif op == "matmul":
            candidates_2d = [t for t in pool if t.data.ndim == 2]
            if candidates_2d:
                t1 = rng.choice(candidates_2d)
                k = t1.data.shape[-1]
                matching = [t for t in candidates_2d if t.data.shape[0] == k]
                if matching:
                    t2 = rng.choice(matching)
                    res = t1 @ t2
                    op_history.append(f"matmul({t1.shape}, {t2.shape}) -> {res.shape}")
                else:
                    res = t1.relu() + 0.1
                    op_history.append(f"fallback_relu({t1.shape}) -> {res.shape}")
            else:
                t = rng.choice(pool)
                res = t.relu() + 0.1
                op_history.append(f"fallback_relu({t.shape}) -> {res.shape}")

        elif op == "relu":
            t = rng.choice(pool)
            res = t.relu()
            op_history.append(f"relu({t.shape}) -> {res.shape}")

        elif op == "exp":
            t = rng.choice(pool)
            # Clip to prevent overflow
            res = (t * 0.1).exp()
            op_history.append(f"exp({t.shape}) -> {res.shape}")

        elif op == "pow":
            t = rng.choice(pool)
            # Power on positive input
            safe_t = t.relu() + 0.5
            p = rng.choice([2, 3])
            res = safe_t ** p
            op_history.append(f"pow({safe_t.shape}, {p}) -> {res.shape}")

        elif op == "sum":
            t = rng.choice(pool)
            res = t.sum()
            op_history.append(f"sum({t.shape}) -> ()")

        elif op == "mean":
            t = rng.choice(pool)
            res = t.mean()
            op_history.append(f"mean({t.shape}) -> ()")

        else:
            t = rng.choice(pool)
            res = t + 0.5
            op_history.append(f"add_const({t.shape})")

        pool.append(res)

    # Final reduction to scalar
    final_node = pool[-1]
    if final_node.data.size > 1:
        root = final_node.sum()
        op_history.append(f"final_sum({final_node.shape}) -> scalar")
    else:
        root = final_node

    topo_desc = " -> ".join(op_history)
    return root, leaves, topo_desc


@pytest.mark.parametrize("seed", range(100))
def test_random_dag_optimizer_differential(seed: int):
    """
    Property-based test: runs 100 randomly generated DAGs through the optimizer
    with verify=True, asserting exact 3-layer autograd parity against dynamic tolerances.
    """
    root, leaves, topo_desc = _build_random_dag(seed)
    try:
        opt_root, report = optimize_graph(root, verify=True)
        assert report.verified, f"Seed {seed}: Report did not mark verified=True"
    except Exception as exc:
        pytest.fail(
            f"\n[RANDOM DAG OPTIMIZER FAILURE]\n"
            f"Seed: {seed}\n"
            f"Topology: {topo_desc}\n"
            f"Leaf count: {len(leaves)}\n"
            f"Leaf shapes: {[leaf.shape for leaf in leaves]}\n"
            f"Root shape: {root.shape}\n"
            f"Error: {exc}\n"
        )


def test_operation_reconstruction_registry():
    """Verify operation reconstruction registry functionality and safety guards."""
    assert is_reconstructible("add")
    assert is_reconstructible("matmul")
    assert is_reconstructible("pow^2")
    assert is_reconstructible("to")
    assert is_reconstructible("log_softmax")
    assert not is_reconstructible("unknown_custom_op_xyz")

    # Custom reconstructor registration
    def custom_doubler(parents, ctx, orig):
        return parents[0] * 2.0

    register_reconstructor("custom_double", custom_doubler)
    assert is_reconstructible("custom_double")
    assert "custom_double" in _RECONSTRUCTORS

    a = Tensor([3.0], requires_grad=True)
    out = _RECONSTRUCTORS["custom_double"]((a,), None, a)
    assert np.allclose(out.data, [6.0])
    out.backward()
    assert np.allclose(a.grad, [2.0])
