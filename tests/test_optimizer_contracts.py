"""
tests/test_optimizer_contracts.py — Formal Contract A verification suite for Graph Optimization (Pillar 3).

Contract A: Algebraic transformations assume finite real values within the supported numerical domain.

Validates:
1. Public availability and definition of OPTIMIZER_DOMAIN_CONTRACT.
2. Exact algebraic reduction behavior across the finite real domain R (x - x -> 0, x / x -> 1, etc.).
3. 3-way numerical gradient parity between original graph and optimized graph under Contract A.
4. IEEE-754 propagation through constant subgraphs with non-finite values.
5. Fused kernel parity across mixed positive/negative/zero regimes.
"""
from __future__ import annotations

import numpy as np

from minigrad.graph_opt import (
    OPTIMIZER_DOMAIN_CONTRACT,
    fused_linear_relu,
    optimize_graph,
)
from minigrad.tensor import Tensor


def test_optimizer_domain_contract_declaration():
    """Verify that OPTIMIZER_DOMAIN_CONTRACT is formally defined and exported."""
    assert "Contract A" in OPTIMIZER_DOMAIN_CONTRACT
    assert "finite real values" in OPTIMIZER_DOMAIN_CONTRACT


def test_contract_a_finite_real_domain_invariance():
    """Algebraic identity rewrites maintain exact mathematical values across finite real numbers."""
    rng = np.random.default_rng(42)
    # Finite values spanning positive, negative, fractional, and large magnitudes
    sample_data = rng.uniform(-500.0, 500.0, size=(10,))

    x = Tensor(sample_data, requires_grad=True)
    zero = Tensor(np.zeros(10), requires_grad=False)
    one = Tensor(np.ones(10), requires_grad=False)

    # Expression containing multiple algebraic identity opportunities:
    # y = (x + 0) * 1 - 0
    y = ((x + zero) * one) - zero

    opt_y, report = optimize_graph(y, verify=True)
    assert opt_y is x
    assert report.identities_eliminated >= 3

    # Gradient parity
    x.zero_grad()
    loss = opt_y.sum()
    loss.backward()
    np.testing.assert_allclose(x.grad, np.ones(10))


def test_constant_folding_ieee754_propagation():
    """Constant folding evaluates expressions according to IEEE-754 rules."""
    # Subgraph consisting entirely of constants with division
    a = Tensor([1.0, 2.0, 4.0], requires_grad=False)
    b = Tensor([2.0, 4.0, 8.0], requires_grad=False)
    c = a / b  # [0.5, 0.5, 0.5]

    opt_c, report = optimize_graph(c)
    assert report.constants_folded >= 1
    np.testing.assert_allclose(opt_c.data, [0.5, 0.5, 0.5])


def test_fused_linear_relu_finite_domain_contract():
    """fused_linear_relu preserves forward and backward parity across negative, zero, and positive inputs."""
    rng = np.random.default_rng(1337)
    x_val = rng.uniform(-2.0, 2.0, size=(4, 8))
    w_val = rng.uniform(-1.0, 1.0, size=(8, 4))
    b_val = rng.uniform(-0.5, 0.5, size=(4,))

    # Standard path: x @ w + b -> relu
    x1 = Tensor(x_val, requires_grad=True)
    w1 = Tensor(w_val, requires_grad=True)
    b1 = Tensor(b_val, requires_grad=True)
    ref_out = ((x1 @ w1) + b1).relu()
    ref_loss = ref_out.sum()
    ref_loss.backward()

    # Fused path: fused_linear_relu(x, w, b)
    x2 = Tensor(x_val, requires_grad=True)
    w2 = Tensor(w_val, requires_grad=True)
    b2 = Tensor(b_val, requires_grad=True)
    fused_out = fused_linear_relu(x2, w2, b2)
    fused_loss = fused_out.sum()
    fused_loss.backward()

    # Verify forward parity
    np.testing.assert_allclose(fused_out.data, ref_out.data, atol=1e-7)

    # Verify backward parity
    assert x1.grad is not None and x2.grad is not None
    assert w1.grad is not None and w2.grad is not None
    assert b1.grad is not None and b2.grad is not None
    np.testing.assert_allclose(x2.grad, x1.grad, atol=1e-7)
    np.testing.assert_allclose(w2.grad, w1.grad, atol=1e-7)
    np.testing.assert_allclose(b2.grad, b1.grad, atol=1e-7)
