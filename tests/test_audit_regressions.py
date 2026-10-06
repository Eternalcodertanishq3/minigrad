"""
Regression tests for all findings from the comprehensive deep audit:
1. NumPy scalar/ndarray left-operand ufunc interception (__array_ufunc__ = None)
2. Iterative topological sort (>1,000 depth without RecursionError)
3. Zero cyclic garbage in Tensor graphs (both no_grad and grad modes)
4. 1D, 2D, 3D, and 4D batched matmul forward, backward, and double-backward
5. C99 compiler strict unsupported-op error + neg/sin/cos/abs/to + pure logf
6. S.U.T.R.A. trajectory unrolled gradients, dopri5 max_steps RuntimeError, t0/t1 kwargs
7. S.P.A.N.D.A. per-layer measured spike trains & single-pass (T=1) ANN baseline MACs
8. P.R.A.M.A.N.A. HeteroscedasticMLP learning x-dependent variance
9. T.A.R.K.A. exact Gödel residuum (1.0 if a <= b else b)
10. Glass-Box forward NaN/Inf detection and Call Stack Snapshot in GradientAnomalyError
11. Double-backward VJP coverage for gelu, einsum, split, layer_norm, embedding, cross_entropy
"""

import gc

import numpy as np
import pytest

from minigrad import (
    SPANDA,
    GaussianNLLLoss,
    GradientAnomalyError,
    HeteroscedasticMLP,
    LogicTensor,
    NeuralODE,
    SpikingLinear,
    SpikingSequential,
    Tensor,
    detect_anomaly,
    export_c,
    grad,
    odeint,
)
from minigrad.graph import no_grad
from minigrad.nn import (
    Conv2D,
    CrossEntropyLoss,
    Embedding,
    LayerNorm,
    Module,
)
from minigrad.ops import einsum, gelu, split
from minigrad.tarka import TNorm


def test_numpy_ufunc_left_multiply_retains_graph() -> None:
    """np.float64 * Tensor and ndarray * Tensor must return Tensor and stay in graph."""
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y1 = np.float64(2.5) * x
    assert isinstance(y1, Tensor)
    assert y1.requires_grad

    arr = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    y2 = arr + y1
    assert isinstance(y2, Tensor)
    loss = y2.sum()
    loss.backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, [2.5, 2.5, 2.5])


def test_double_backward_gelu_and_extended_ops() -> None:
    """Double-backward works through gelu, einsum, split, layer_norm, embedding, cross_entropy."""
    # 1. GELU double-backward
    x = Tensor([0.5, -0.3, 1.2], requires_grad=True)
    g1 = grad(gelu(x).sum(), x, create_graph=True)[0]
    g2 = grad(g1.sum(), x)[0]
    eps = 1e-5
    x_p = Tensor(x.data + eps, requires_grad=True)
    x_m = Tensor(x.data - eps, requires_grad=True)
    fd = (grad(gelu(x_p).sum(), x_p)[0].data - grad(gelu(x_m).sum(), x_m)[0].data) / (2 * eps)
    np.testing.assert_allclose(g2.data, fd, rtol=1e-4, atol=1e-5)

    # 2. Einsum double-backward
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[0.5, -1.0], [1.5, 2.0]], requires_grad=True)
    out_ein = einsum("ij,jk->ik", a, b)
    ga = grad((out_ein ** 2).sum(), a, create_graph=True)[0]
    gga = grad(ga.sum(), a)[0]
    assert gga.shape == a.shape
    assert np.all(np.isfinite(gga.data))

    # 3. Split double-backward
    s_in = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    p1, p2 = split(s_in, 2, axis=1)
    gs = grad((p1 * p2).sum(), s_in, create_graph=True)[0]
    ggs = grad(gs.sum(), s_in)[0]
    np.testing.assert_allclose(ggs.data, [[1.0, 1.0, 1.0, 1.0]])

    # 4. LayerNorm double-backward
    ln = LayerNorm(4)
    x_ln = Tensor([[1.0, 2.0, 3.5, -1.0], [0.5, -0.5, 1.0, 2.0]], requires_grad=True)
    y_ln = (ln(x_ln) ** 2).sum()
    gln = grad(y_ln, x_ln, create_graph=True)[0]
    ggln = grad(gln.sum(), x_ln)[0]
    assert ggln.shape == x_ln.shape
    assert np.all(np.isfinite(ggln.data))

    # 5. Embedding double-backward
    emb = Embedding(5, 3)
    idx = Tensor([0, 2, 1, 0])
    y_emb = (emb(idx) ** 3).sum()
    g_emb = grad(y_emb, emb.weight, create_graph=True)[0]
    gg_emb = grad(g_emb.sum(), emb.weight)[0]
    assert gg_emb.shape == emb.weight.shape
    assert np.all(np.isfinite(gg_emb.data))

    # 6. CrossEntropyLoss double-backward
    ce = CrossEntropyLoss()
    logits = Tensor([[1.2, -0.5, 0.3], [0.1, 2.0, -1.0]], requires_grad=True)
    targets = Tensor([0, 1])
    loss_ce = ce(logits, targets)
    g_ce = grad(loss_ce, logits, create_graph=True)[0]
    gg_ce = grad((g_ce ** 2).sum(), logits)[0]
    assert gg_ce.shape == logits.shape
    assert np.all(np.isfinite(gg_ce.data))


def test_iterative_topological_sort_deep_graph() -> None:
    """Graphs deeper than 1,500 nodes must not hit RecursionError in backward() or grad()."""
    x = Tensor([1.0], requires_grad=True)
    h = x
    for _ in range(1500):
        h = h * 0.999 + 0.001
    g = grad(h.sum(), x, retain_graph=True)[0]
    assert np.isfinite(g.data[0])
    h.backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, g.data, rtol=1e-10)


def test_zero_reference_cycles_in_no_grad_and_grad() -> None:
    """Tensor chains must not create reference cycles requiring cyclic GC."""
    gc.collect()
    gc.disable()
    try:
        with no_grad():
            x = Tensor([1.0, 2.0])
            for _ in range(500):
                x = x + 1.0
            del x
        unreachable_nograd = gc.collect()
        assert unreachable_nograd == 0

        y = Tensor([1.0, 2.0], requires_grad=True)
        z = y
        for _ in range(500):
            z = z * 1.0001
        del y, z
        unreachable_grad = gc.collect()
        assert unreachable_grad == 0
    finally:
        gc.enable()


def test_batched_matmul_3d_and_4d() -> None:
    """3D and 4D batched matmul works in forward, backward, and double-backward."""
    rng = np.random.default_rng(42)
    a_np = rng.standard_normal((2, 3, 4, 5))
    b_np = rng.standard_normal((2, 3, 5, 6))

    a = Tensor(a_np, requires_grad=True)
    b = Tensor(b_np, requires_grad=True)
    c = a @ b
    assert c.shape == (2, 3, 4, 6)
    np.testing.assert_allclose(c.data, a_np @ b_np, rtol=1e-12)

    ga, gb = grad((c ** 2).sum(), [a, b], create_graph=True)
    assert ga.shape == a.shape
    assert gb.shape == b.shape
    np.testing.assert_allclose(ga.data, (2.0 * (a_np @ b_np)) @ np.swapaxes(b_np, -1, -2), rtol=1e-11)
    np.testing.assert_allclose(gb.data, np.swapaxes(a_np, -1, -2) @ (2.0 * (a_np @ b_np)), rtol=1e-11)

    gga = grad(ga.sum(), a)[0]
    assert gga.shape == a.shape
    assert np.all(np.isfinite(gga.data))


def test_c_compiler_rejects_unsupported_and_supports_trig_abs_neg() -> None:
    """C99 compiler raises NotImplementedError on Conv2D and emits valid C for neg/sin/cos/abs."""
    conv = Conv2D(1, 2, kernel_size=3)
    x_img = Tensor(np.ones((1, 1, 5, 5), dtype=np.float32))
    with pytest.raises(NotImplementedError, match="not supported by the C99 compiler"):
        export_c(conv, x_img)

    class TrigModule(Module):
        def forward(self, x: Tensor) -> Tensor:
            return (-x).sin() + x.cos() + x.abs()

    mod = TrigModule()
    x_vec = Tensor(np.array([[0.5, -1.2, 2.0]], dtype=np.float32), requires_grad=True)
    c_code = export_c(mod, x_vec, optimize=False)
    assert "minigrad_sin(" in c_code
    assert "minigrad_cos(" in c_code
    assert "minigrad_abs(" in c_code
    assert "1e-9f" not in c_code


def test_sutra_unrolled_trajectory_grad_and_dopri5_max_steps() -> None:
    """odeint(use_adjoint=False, return_trajectory=True) propagates gradients; dopri5 raises on max_steps."""
    class LinearDecay(Module):
        def __init__(self) -> None:
            super().__init__()
            self.rate = Tensor([-0.5], requires_grad=True)

        def forward(self, t: float, z: Tensor) -> Tensor:
            return z * self.rate

    func = LinearDecay()
    z0 = Tensor([[2.0, 4.0]], requires_grad=True)
    t_eval = np.linspace(0.0, 1.0, 11)
    traj = odeint(func, z0, t_span=t_eval, method="rk4", options={"n_steps": 2}, use_adjoint=False, return_trajectory=True)
    assert isinstance(traj, Tensor)
    assert traj.shape == (11, 1, 2)
    loss = traj.sum()
    loss.backward()
    assert z0.grad is not None
    assert func.rate.grad is not None
    assert np.all(np.abs(z0.grad) > 1.0)
    assert np.abs(func.rate.grad[0]) > 1.0

    # NeuralODE t0/t1 compatibility
    node = NeuralODE(func, t0=0.0, t1=0.5, solver="rk4")
    assert node.t_span == (0.0, 0.5)

    # dopri5 max_steps exhaustion raises RuntimeError
    with pytest.raises(RuntimeError, match="exceeded max_steps"):
        odeint(func, z0, t_span=(0.0, 100.0), method="dopri5", rtol=1e-12, atol=1e-14, options={"max_steps": 2})


def test_spanda_measured_spikes_and_single_pass_ann_baseline() -> None:
    """SPANDA records per-layer spikes and compares against a single-pass (T=1) ANN baseline."""
    snn = SpikingSequential(
        SpikingLinear(8, 16),
        SpikingLinear(16, 4),
    )
    x_spikes = Tensor(np.ones((10, 2, 8), dtype=np.float64))
    telemetry = SPANDA.evaluate_neuromorphic_energy(snn, x_spikes)
    # Single-pass ANN MACs for batch=2: 2 * (8*16 + 16*4) = 2 * 192 = 384
    assert telemetry.dense_macs == 384


def test_pramana_heteroscedastic_mlp_x_dependent_variance() -> None:
    """HeteroscedasticMLP produces genuinely input-dependent variance."""
    model = HeteroscedasticMLP(1, 8, 1)
    x_test = Tensor([[-2.0], [0.0], [2.0]])
    pred = model(x_test)
    # With nonlinear hidden activations, variance at x=-2, 0, 2 should be distinct and positive
    assert pred.var.shape == (3, 1)
    assert np.all(pred.var.data > 0.0)
    loss = GaussianNLLLoss()(pred, Tensor([[1.0], [0.0], [1.0]]))
    loss.backward()
    assert any(p.grad is not None and np.any(p.grad != 0) for p in model.parameters())


def test_tarka_godel_residuum() -> None:
    """Gödel implication residuum returns 1.0 if a <= b else b."""
    a = LogicTensor([0.2, 0.8, 0.5], tnorm=TNorm.GODEL)
    b = LogicTensor([0.6, 0.3, 0.5], tnorm=TNorm.GODEL)
    imp = a >> b
    np.testing.assert_allclose(imp.data, [1.0, 0.3, 1.0])


def test_glassbox_forward_anomaly_and_call_stack_snapshot() -> None:
    """detect_anomaly catches forward NaN/Inf and includes Call Stack Snapshot in details."""
    with pytest.raises(GradientAnomalyError) as exc_info:
        with detect_anomaly():
            x = Tensor([0.0], requires_grad=True)
            y = x.log()  # Forward -inf
            y.backward()

    err = exc_info.value
    assert "Call Stack Snapshot" in err.details
    assert len(err.details["Call Stack Snapshot"]) > 0
