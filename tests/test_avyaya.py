"""
test_avyaya.py — Automated Unit Tests for A.V.Y.A.Y.A.
(Adaptive Volume-preserving Yield-lossless Activation-inverting Y-reconstruction Autograd)

Tests:
1. test_exact_algebraic_reversibility (x == inverse(forward(x)) to machine precision)
2. test_gradient_parity_vs_standard_backprop (is_reversible=True vs False)
3. test_o1_activation_memory_scaling (O(1) graph size invariance across depth)
4. test_multi_layer_deep_reconstruction (Zero catastrophic drift across 30 layers)
5. test_heterogeneous_submodules (Linear, ReLU, GELU, Tanh, LayerNorm)
6. test_end_to_end_training_convergence (Training reversible deep net with Adam)
7. test_asymmetric_split_partitions (split_ratio != 0.5, e.g. 3:7)
8. test_batched_multi_dimensional_inputs (3D sequence and 4D tensor inputs)
"""
import numpy as np
import pytest

from minigrad.avyaya import AVYAYA, ReversibleBlock, ReversibleSequential, ReconstructionTelemetry
from minigrad.graph import topological_sort
from minigrad.nn import GELU, LayerNorm, Linear, MSELoss, ReLU, Sequential, Tanh
from minigrad.optim import Adam
from minigrad.tensor import Tensor


# ── 1. Exact Algebraic Reversibility ─────────────────────────────────

def test_exact_algebraic_reversibility():
    """
    Validates that a ReversibleBlock is 100% analytically invertible:
    x_reconstructed = block.inverse(block.forward(x)) == x (machine precision).
    """
    np.random.seed(42)
    f = Sequential([Linear(8, 16), Tanh(), Linear(16, 8)])
    g = Sequential([Linear(8, 16), ReLU(), Linear(16, 8)])

    block = ReversibleBlock(f, g, split_dim=-1, split_ratio=0.5)

    x_data = np.random.randn(4, 16)
    x = Tensor(x_data.copy())

    # Forward transformation: [x1, x2] -> [y1, y2]
    y = block.forward(x)

    # Inverse transformation: [y1, y2] -> [x1, x2]
    x_rec = block.inverse(y)

    # Verify bit-for-bit mathematical equivalence to floating point precision
    np.testing.assert_allclose(x_rec.numpy(), x_data, rtol=1e-14, atol=1e-14)


# ── 2. Gradient Parity vs Standard Backpropagation ───────────────────

def test_gradient_parity_vs_standard_backprop():
    """
    Compares input and parameter gradients between:
    - Standard unrolled autograd (is_reversible=False, caches all activations)
    - A.V.Y.A.Y.A. O(1) memory autograd (is_reversible=True, reconstructs on the fly)
    Asserts bit-for-bit gradient parity.
    """
    np.random.seed(123)

    def create_blocks():
        b1 = ReversibleBlock(Linear(6, 6), Linear(6, 6))
        b2 = ReversibleBlock(Linear(6, 6), Linear(6, 6))
        b3 = ReversibleBlock(Linear(6, 6), Linear(6, 6))
        return [b1, b2, b3]

    blocks_unrolled = create_blocks()
    blocks_avyaya = create_blocks()

    # Synchronize initial weights exactly
    for b_u, b_a in zip(blocks_unrolled, blocks_avyaya):
        for p_u, p_a in zip(b_u.parameters(), b_a.parameters()):
            p_a.data = p_u.data.copy()

    model_unrolled = ReversibleSequential(blocks_unrolled, is_reversible=False)
    model_avyaya = ReversibleSequential(blocks_avyaya, is_reversible=True)

    x_data = np.random.randn(3, 12)

    # 1. Unrolled forward and backward
    x_u = Tensor(x_data.copy(), requires_grad=True)
    out_u = model_unrolled(x_u)
    loss_u = (out_u ** 2).sum()
    loss_u.backward()

    # 2. A.V.Y.A.Y.A. O(1) forward and backward
    x_a = Tensor(x_data.copy(), requires_grad=True)
    out_a = model_avyaya(x_a)
    loss_a = (out_a ** 2).sum()
    loss_a.backward()

    # Verify forward values match
    np.testing.assert_allclose(out_u.numpy(), out_a.numpy(), rtol=1e-6, atol=1e-6)

    # Verify input gradients match
    np.testing.assert_allclose(x_u.grad, x_a.grad, rtol=1e-5, atol=1e-5)

    # Verify parameter gradients match across all blocks
    for p_u, p_a in zip(model_unrolled.parameters(), model_avyaya.parameters()):
        np.testing.assert_allclose(p_u.grad, p_a.grad, rtol=1e-5, atol=1e-5)


# ── 3. O(1) Activation Memory Scaling ────────────────────────────────

def test_o1_activation_memory_scaling():
    """
    Asserts that the autograd graph size is strictly O(1) constant
    independent of the number of reversible blocks L, whereas standard
    unrolled graphs scale linearly O(L).
    """
    for depth in [4, 12, 24]:
        blocks = [
            ReversibleBlock(Linear(4, 4), Linear(4, 4))
            for _ in range(depth)
        ]
        model = ReversibleSequential(blocks, is_reversible=True)

        x = Tensor(np.random.randn(2, 8), requires_grad=True)
        out = model(x)

        topo = topological_sort(out)
        # Graph contains only [x, all_params..., out]
        # Intermediate activation nodes are strictly 0!
        n_params = len(model.parameters())
        expected_nodes = 1 + n_params + 1  # x + params + out
        assert len(topo) == expected_nodes, (
            f"Expected {expected_nodes} nodes in O(1) graph, got {len(topo)}"
        )


# ── 4. Multi-Layer Deep Reconstruction Fidelity ──────────────────────

def test_multi_layer_deep_reconstruction():
    """
    Verifies that reconstruction across a 30-layer deep reversible network
    does not suffer from catastrophic floating-point drift.
    """
    np.random.seed(99)
    blocks = [
        ReversibleBlock(
            Sequential([Linear(8, 8), Tanh()]),
            Sequential([Linear(8, 8), Tanh()]),
        )
        for _ in range(30)
    ]
    model = ReversibleSequential(blocks, is_reversible=True)

    x = Tensor(np.random.randn(4, 16), requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    telem = model.telemetry
    assert telem is not None
    assert telem.n_blocks == 30
    # Floating point precision drift over 30 non-linear layers should be < 1e-10
    assert telem.max_drift < 1e-10, f"Max drift exceeded bound: {telem.max_drift}"
    assert telem.forward_memory_saved_ratio > 0.95


# ── 5. Heterogeneous Submodules ──────────────────────────────────────

def test_heterogeneous_submodules():
    """
    Tests ReversibleBlock with diverse activation functions and LayerNorm:
    Linear + GELU, Linear + ReLU, LayerNorm.
    """
    b1 = ReversibleBlock(
        Sequential([Linear(6, 12), GELU(), Linear(12, 6)]),
        Sequential([Linear(6, 12), ReLU(), Linear(12, 6)]),
    )
    b2 = ReversibleBlock(
        Sequential([LayerNorm(6), Linear(6, 6)]),
        Sequential([LayerNorm(6), Linear(6, 6)]),
    )

    seq = ReversibleSequential([b1, b2], is_reversible=True)

    x = Tensor(np.random.randn(3, 12), requires_grad=True)
    out = seq(x)
    loss = (out ** 2).mean()
    loss.backward()

    assert x.grad is not None
    assert np.all(np.isfinite(x.grad))
    for p in seq.parameters():
        assert p.grad is not None
        assert np.all(np.isfinite(p.grad))


# ── 6. End-to-End Training Convergence ───────────────────────────────

def test_end_to_end_training_convergence():
    """
    Trains an 8-block deep ReversibleSequential model with Adam
    on a non-linear regression task, validating loss reduction.
    """
    np.random.seed(42)

    blocks = [
        ReversibleBlock(
            Sequential([Linear(8, 16), Tanh(), Linear(16, 8)]),
            Sequential([Linear(8, 16), Tanh(), Linear(16, 8)]),
        )
        for _ in range(8)
    ]
    model = ReversibleSequential(blocks, is_reversible=True)
    optimizer = Adam(model.parameters(), lr=0.03)
    loss_fn = MSELoss()

    X_train = np.random.randn(16, 16)
    # Synthetic target: non-linear transformation
    Y_target = Tensor(np.sin(X_train) * 0.5)

    initial_loss = None
    final_loss = None

    for step in range(25):
        optimizer.zero_grad()
        pred = model(Tensor(X_train))
        loss = loss_fn(pred, Y_target)

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < initial_loss * 0.6, (
        f"Expected loss drop > 40%, got {initial_loss:.4f} -> {final_loss:.4f}"
    )


# ── 7. Asymmetric Split Partitions ───────────────────────────────────

def test_asymmetric_split_partitions():
    """
    Tests reversible coupling where input dimension is split asymmetrically:
    Total dim = 10, split_ratio = 0.3 ==> d1 = 3, d2 = 7.
    f: R^7 -> R^3
    g: R^3 -> R^7
    """
    f = Linear(7, 3)
    g = Linear(3, 7)

    block = ReversibleBlock(f, g, split_dim=-1, split_ratio=0.3)

    x_data = np.random.randn(2, 10)
    x = Tensor(x_data.copy(), requires_grad=True)

    y = block.forward(x)
    assert y.shape == (2, 10)

    # Invert
    x_rec = block.inverse(y)
    np.testing.assert_allclose(x_rec.numpy(), x_data, rtol=1e-14, atol=1e-14)

    # Backward step
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert f.weight.grad is not None
    assert g.weight.grad is not None


# ── 8. Batched Multi-Dimensional Sequence Inputs ──────────────────────

def test_batched_multi_dimensional_inputs():
    """
    Validates reversible blocks on 3D sequence tensors (Batch, Time, Dimension).
    """
    block = ReversibleBlock(
        Linear(8, 8),
        Linear(8, 8),
        split_dim=-1,
        split_ratio=0.5,
    )
    seq = ReversibleSequential([block, block], is_reversible=True)

    # (Batch=2, Time=5, Dim=16)
    x = Tensor(np.random.randn(2, 5, 16), requires_grad=True)
    out = seq(x)
    assert out.shape == (2, 5, 16)

    loss = out.sum()
    loss.backward()

    assert x.grad.shape == (2, 5, 16)
    assert seq.telemetry is not None
    assert seq.telemetry.max_drift < 1e-12
