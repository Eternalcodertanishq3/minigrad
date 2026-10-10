"""
Numerical regression tests for the v1.4.1 audit fixes.

Unlike smoke tests (shape / isfinite), every test here compares against an independent reference:
NumPy, central finite differences, closed forms, Monte Carlo, or a compiled C program run on a
*different* input than the one used at export time. None of them require PyTorch.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import zlib
from pathlib import Path
from typing import Callable, List

import numpy as np
import pytest

import minigrad.nn as nn
import minigrad.ops as ops
from minigrad import Tensor
from minigrad.autograd import grad
from minigrad.compiler import export_c
from minigrad.graph import topological_sort
from minigrad.graph_opt import optimize
from minigrad.pramana import (
    DistributionalLinear,
    DistributionalTensor,
    GaussianNLLLoss,
    HeteroscedasticMLP,
)
from minigrad.spanda import SPANDA, SpikingLinear, SpikingSequential
from minigrad.tarka import LogicTensor, TNorm

HAS_GCC = shutil.which("gcc") is not None


# ── helpers ──────────────────────────────────────────────────────────────────

def _fd_grad(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference gradient of a scalar function."""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        xp, xm = x.copy(), x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        g[idx] = (f(xp) - f(xm)) / (2 * eps)
    return g


def _first_grad(build: Callable[[Tensor], Tensor], x: np.ndarray) -> np.ndarray:
    t = Tensor(x.copy(), requires_grad=True)
    (g,) = grad(build(t), [t], create_graph=False)
    return np.asarray(g.data)


def _second_order_check(build: Callable[[Tensor], Tensor], x: np.ndarray, tol: float = 1e-5) -> None:
    """Verify d/dx of h(x)=sum(grad f(x)^2) computed by double-backward against finite differences."""
    t = Tensor(x.copy(), requires_grad=True)
    (g1,) = grad(build(t), [t], create_graph=True)
    (g2,) = grad((g1 * g1).sum(), [t], create_graph=False)
    numeric = _fd_grad(lambda a: float(np.sum(_first_grad(build, a) ** 2)), x, eps=1e-5)
    scale = max(1.0, float(np.abs(numeric).max()))
    assert np.abs(np.asarray(g2.data) - numeric).max() <= tol * scale


# ── core: batched matmul ─────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "sa,sb",
    [((2, 3, 4), (2, 4, 5)), ((2, 3, 4), (4, 5)), ((2, 2, 3, 4), (2, 2, 4, 2)), ((1, 3, 4), (3, 4, 2)), ((4,), (4, 3)), ((3, 4), (4,))],
)
def test_matmul_forward_and_gradients_match_numpy_and_finite_differences(sa, sb):
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=sa), rng.normal(size=sb)
    w = rng.normal(size=np.matmul(a, b).shape)
    ta, tb = Tensor(a.copy(), requires_grad=True), Tensor(b.copy(), requires_grad=True)
    out = ta @ tb
    np.testing.assert_allclose(out.data, np.matmul(a, b), atol=1e-12)
    (out * Tensor(w)).sum().backward()
    f_a = lambda x: float(np.sum(np.matmul(x, b) * w))  # noqa: E731
    f_b = lambda x: float(np.sum(np.matmul(a, x) * w))  # noqa: E731
    np.testing.assert_allclose(ta.grad, _fd_grad(f_a, a), atol=1e-6)
    np.testing.assert_allclose(tb.grad, _fd_grad(f_b, b), atol=1e-6)


def test_numpy_left_operand_stays_in_graph():
    x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    for expr in (np.float64(2.0) * x, np.array([1.0, 2.0, 3.0]) * x, np.float64(2.0) + x, np.float64(2.0) / x):
        assert isinstance(expr, Tensor)
    (np.float64(3.0) * x).sum().backward()
    np.testing.assert_allclose(x.grad, [3.0, 3.0, 3.0])


# ── autograd: second-order rules (finite-difference verified) ────────────────

_RNG = np.random.default_rng(7)
_X = _RNG.normal(size=(3, 4))
_LABELS = np.array([0, 3, 1])
_TARGET = (_RNG.random((3, 4)) > 0.5).astype(float)
_LN = nn.LayerNorm(4)
_LN.gamma.data[...] = _RNG.normal(size=4) if hasattr(_LN, "gamma") else 0.0


@pytest.mark.parametrize(
    "name,build",
    [
        ("gelu", lambda x: (x.gelu() ** 2).sum()),
        ("einsum", lambda x: ops.einsum("ij,ij->", x, x * x)),
        ("split", lambda x: sum((p ** 3).sum() * (i + 1) for i, p in enumerate(ops.split(x, [1, 3], axis=1)))),
        ("pad", lambda x: (ops.pad(x, ((1, 1), (0, 2))) ** 3).sum()),
        ("layer_norm", lambda x: (_LN(x) ** 3).sum()),
        ("cross_entropy", lambda x: nn.CrossEntropyLoss()(x, _LABELS)),
        ("bce", lambda x: nn.BCELoss()(x.sigmoid(), _TARGET)),
        ("bce_with_logits", lambda x: nn.BCEWithLogitsLoss()(x, _TARGET)),
        ("nll", lambda x: nn.NLLLoss()(ops.log_softmax(x), _LABELS)),
        ("mse", lambda x: nn.MSELoss()(x, Tensor(_TARGET))),
    ],
)
def test_second_order_rules_match_finite_differences(name, build):
    _second_order_check(build, _X)


def test_embedding_second_order_matches_finite_differences():
    idx = np.array([[1, 2, 2], [6, 0, 1]])
    table = np.random.default_rng(1).normal(size=(7, 3))

    def build(w: Tensor) -> Tensor:
        emb = nn.Embedding(7, 3)
        emb.weight = w
        return (emb(Tensor(idx)) ** 3).sum()

    _second_order_check(build, table)


def test_dropout_double_backward_matches_analytic_derivative():
    np.random.seed(3)
    x = Tensor(np.random.randn(4, 5), requires_grad=True)
    drop = nn.Dropout(0.4)
    drop.train()
    (g1,) = grad((drop(x) ** 2).sum(), [x], create_graph=True)
    (g2,) = grad((g1 * g1).sum(), [x])
    m = drop._mask / (1 - 0.4)  # y = sum((m x)^2) -> g1 = 2 m^2 x -> d/dx sum(g1^2) = 8 m^4 x
    np.testing.assert_allclose(g2.data, 8 * m ** 4 * x.data, atol=1e-12)


def test_unsupported_double_backward_raises_actionable_error():
    conv = nn.Conv2D(1, 2, 3)
    x = Tensor(np.random.default_rng(0).normal(size=(1, 1, 5, 5)), requires_grad=True)
    with pytest.raises(NotImplementedError, match="conv2d"):
        (g1,) = grad(conv(x).sum(), [x], create_graph=True)
        grad((g1 * g1).sum(), [x])


def test_embedding_accepts_ndarray_list_and_tensor_indices():
    np.random.seed(0)
    emb = nn.Embedding(7, 3)
    idx = [[1, 2], [6, 0]]
    ref = emb.weight.data[np.array(idx)]
    for arg in (idx, np.array(idx), Tensor(np.array(idx))):
        np.testing.assert_allclose(emb(arg).data, ref)


# ── graph optimizer ──────────────────────────────────────────────────────────

def test_optimizer_never_folds_or_eliminates_protected_inputs_even_if_zero():
    x = Tensor(np.zeros((2, 3)))  # all-zero, requires_grad=False: looks like a constant
    w = Tensor(np.ones((3, 3)), requires_grad=True)
    root = ((x @ w) + x * 2.0 + 1.0).sum()
    folded = optimize(root)  # unprotected: x is legitimately a constant
    kept = optimize(root, protected=[x])
    assert x in topological_sort(kept)
    assert len(topological_sort(kept)) >= len(topological_sort(folded))
    x.data[...] = np.arange(6.0).reshape(2, 3)  # new input value must flow through the protected graph
    expected = float(np.sum(x.data @ w.data + x.data * 2.0 + 1.0))
    # re-evaluate the optimized graph by rebuilding it on the new data
    rebuilt = optimize(((x @ w) + x * 2.0 + 1.0).sum(), protected=[x])
    assert abs(float(rebuilt.data) - expected) < 1e-9


def test_optimizer_reshape_rebuild_preserves_target_shape_and_values():
    rng = np.random.default_rng(0)
    x = Tensor(rng.normal(size=(2, 6)), requires_grad=True)
    w = Tensor(rng.normal(size=(6, 6)), requires_grad=True)
    y = ((x @ w) + 0.0).reshape(2, 2, 3)  # identity add is eliminated -> upstream changes -> reshape is rebuilt
    ref = y.data.copy()
    opt = optimize(y)
    assert opt.data.shape == (2, 2, 3)  # a rebuilt reshape must keep its TARGET shape (not the input shape)
    np.testing.assert_allclose(opt.data, ref, atol=1e-12)
    weights = rng.normal(size=(2, 2, 3))
    (opt * Tensor(weights)).sum().backward()
    np.testing.assert_allclose(x.grad, (weights.reshape(2, 6) @ w.data.T), atol=1e-9)


# ── C compiler: strict input-swap differential ───────────────────────────────

def _compile_and_run(tmp: Path, name: str, build, shape, grad_input: bool):
    """Export with input x0, replace the baked sample input by x1 in the C source, run, return (C out, py out)."""
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    x0 = Tensor(rng.normal(size=shape), requires_grad=grad_input)
    x1 = rng.normal(size=shape)
    safe = re.sub(r"\W", "_", name)
    cfile = tmp / f"{safe}.c"
    export_c(build(x0), x0, filename=cfile, model_name="m")
    src = cfile.read_text()
    vals = ", ".join(f"{v:.9e}f" for v in x1.ravel())
    new = re.sub(r"(static const float sample_input\[\d+\]\s*=\s*\{)[^}]*(\})", lambda m: m.group(1) + " " + vals + " " + m.group(2), src, count=1)
    assert new != src, "sample_input array not found in generated C"
    cfile.write_text(new)
    exe = tmp / cfile.stem
    subprocess.run(["gcc", "-O2", "-o", str(exe), str(cfile), "-lm"], check=True, capture_output=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True, cwd=tmp).stdout
    c_out = np.array([float(line.split("=")[-1]) for line in out.splitlines() if "output[" in line and "=" in line])
    return c_out, build(Tensor(x1, requires_grad=grad_input)).data.ravel()


_MLP = nn.Sequential([nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 3)])
_FROZEN = nn.Sequential([nn.Linear(5, 8), nn.Tanh(), nn.Linear(8, 3)])
for _p in _FROZEN.parameters():
    _p.requires_grad = False

_SUPPORTED = [
    ("mlp_trainable", lambda x: _MLP(x), (2, 5)),
    ("mlp_frozen", lambda x: _FROZEN(x), (2, 5)),
    ("scalar_consts", lambda x: x * 2.0 + 1.0, (3, 4)),
    ("scalar_left_sub", lambda x: 1.0 - x, (3, 4)),
    ("scalar_left_div", lambda x: 2.0 / (x * x + 1.0), (3, 4)),
    ("scalar_left_add", lambda x: 0.5 + x, (3, 4)),
    ("unary_mix", lambda x: (-x).sin() + x.cos() + ops.abs(x), (3, 4)),
    ("sum_all", lambda x: x.sum(), (3, 4)),
    ("sum_axis0", lambda x: x.sum(axis=0), (3, 4)),
    ("sum_axis1", lambda x: x.sum(axis=1), (3, 4)),
    ("mean_axis0", lambda x: x.mean(axis=0), (3, 4)),
    ("mean_last_3d", lambda x: x.mean(axis=-1), (2, 3, 4)),
    ("sum_adjacent_axes_3d", lambda x: x.sum(axis=(1, 2)), (2, 3, 4)),
    ("mean_middle_axis_3d", lambda x: x.mean(axis=1), (2, 3, 4)),
    ("softmax_last", lambda x: ops.softmax(x), (3, 4)),
    ("bias_add", lambda x: x + Tensor(np.arange(4.0)), (3, 4)),
]


@pytest.mark.skipif(not HAS_GCC, reason="gcc not available")
@pytest.mark.parametrize("grad_input", [False, True])
@pytest.mark.parametrize("name,build,shape", _SUPPORTED, ids=[c[0] for c in _SUPPORTED])
def test_compiled_c_matches_python_on_a_different_input(tmp_path, name, build, shape, grad_input):
    c_out, py_out = _compile_and_run(tmp_path, f"{name}_{grad_input}", build, shape, grad_input)
    assert c_out.shape == py_out.shape
    np.testing.assert_allclose(c_out, py_out, rtol=2e-4, atol=2e-4 * max(1.0, float(np.abs(py_out).max())))


_UNSUPPORTED = [
    ("sum_nonadjacent", lambda x: x.sum(axis=(0, 2)), (2, 3, 4)),
    ("softmax_axis0", lambda x: ops.softmax(x, axis=0), (3, 4)),
    ("broadcast_col", lambda x: x + x.sum(axis=1, keepdims=True), (3, 4)),
    ("broadcast_row_mul", lambda x: x * Tensor(np.arange(4.0)), (3, 4)),
    ("batched_matmul", lambda x: x @ Tensor(np.ones((2, 4, 5))), (2, 3, 4)),
    ("matrix_vector", lambda x: x @ Tensor(np.ones(4)), (3, 4)),
    ("transpose", lambda x: x.transpose(), (3, 4)),
    ("getitem", lambda x: x[1:, :2] * 2.0, (3, 4)),
    ("max_axis", lambda x: ops.max(x, axis=1), (3, 4)),
    ("clip", lambda x: ops.clip(x, -0.3, 0.3), (3, 4)),
]


@pytest.mark.parametrize("grad_input", [False, True])
@pytest.mark.parametrize("name,build,shape", _UNSUPPORTED, ids=[c[0] for c in _UNSUPPORTED])
def test_unsupported_graphs_fail_loudly_instead_of_miscompiling(tmp_path, name, build, shape, grad_input):
    x = Tensor(np.random.default_rng(0).normal(size=shape), requires_grad=grad_input)
    with pytest.raises(NotImplementedError):
        export_c(build(x), x, filename=tmp_path / "m.c", model_name="m")


def test_mha_export_fails_with_not_implemented_not_opaque_value_error(tmp_path):
    np.random.seed(0)
    mha = nn.MultiHeadAttention(16, 4)
    x = Tensor(np.random.randn(1, 5, 16))
    out = mha(x)
    out = out[0] if isinstance(out, tuple) else out
    with pytest.raises(NotImplementedError):
        export_c(out, x, filename=tmp_path / "m.c", model_name="m")


def test_detached_output_is_rejected_instead_of_baked_as_constant(tmp_path):
    x = Tensor(np.ones((2, 3)))
    detached = Tensor((x * 2.0).data)  # severed from the input
    with pytest.raises(ValueError, match="does not depend on example_input"):
        export_c(detached, x, filename=tmp_path / "m.c", model_name="m")


# ── P.R.A.M.A.N.A.: Monte Carlo accuracy + documented limits ─────────────────

def _mc_var_rel_err(act_dist: Callable, act_np: Callable, mu: np.ndarray, sigma: float, n: int = 200_000) -> np.ndarray:
    var = np.full_like(mu, sigma ** 2)
    pred = act_dist(DistributionalTensor(Tensor(mu), Tensor(var))).var.data
    xs = np.random.default_rng(0).normal(mu, sigma, size=(n,) + mu.shape)
    mc = act_np(xs).var(axis=0)
    return np.abs(pred - mc) / mc


_MU = np.array([[-1.0, -0.3, 0.3, 0.5, 1.0, 2.0]])


def test_delta_method_accurate_for_smooth_activations_at_small_noise():
    assert _mc_var_rel_err(lambda d: d.tanh(), np.tanh, _MU, 0.05).max() < 0.05
    assert _mc_var_rel_err(lambda d: d.sigmoid(), lambda x: 1 / (1 + np.exp(-x)), _MU, 0.05).max() < 0.03


def test_delta_method_error_grows_with_noise_as_documented():
    small = _mc_var_rel_err(lambda d: d.tanh(), np.tanh, _MU, 0.05).max()
    large = _mc_var_rel_err(lambda d: d.tanh(), np.tanh, _MU, 0.3).max()
    assert large > 10 * small and large > 0.2


def test_single_affine_layer_is_exact_against_monte_carlo():
    np.random.seed(2)
    layer = DistributionalLinear(6, 3)
    mu = np.random.default_rng(1).normal(size=(1, 6))
    var = np.full((1, 6), 0.04)
    pred = layer(DistributionalTensor(Tensor(mu), Tensor(var))).var.data.ravel()
    xs = np.random.default_rng(3).normal(mu, 0.2, size=(300_000, 6))
    mc = (xs @ layer.weight.data + layer.bias.data).var(axis=0)
    np.testing.assert_allclose(pred, mc, rtol=0.03)


def test_stacked_affine_layers_drop_covariance_as_documented():
    """Known limitation: composing layers is approximate. If this starts failing, update the docs."""
    np.random.seed(2)
    l1, l2 = DistributionalLinear(8, 16), DistributionalLinear(16, 3)
    mu = np.random.default_rng(1).normal(size=(1, 8))
    s = 0.05
    pred = l2(l1(DistributionalTensor(Tensor(mu), Tensor(np.full((1, 8), s ** 2))))).var.data.ravel()
    exact = s ** 2 * np.sum((l1.weight.data @ l2.weight.data) ** 2, axis=0)  # closed form for a linear stack
    rel = np.abs(pred - exact) / exact
    assert rel.max() > 0.10, "stacked layers unexpectedly exact: the 'single-layer-exact' docs can be relaxed"


def test_weight_variance_layer_tracks_magnitude_not_training_support():
    """Documented limitation: weight_uncertainty is not an OOD detector."""

    def trained(center: float) -> DistributionalLinear:
        np.random.seed(42)
        layer = DistributionalLinear(2, 1, weight_uncertainty=True, init_log_var=-3.5)
        x = np.random.uniform(center - 0.5, center + 0.5, size=(64, 2))
        y = 1.5 * x[:, :1] - 0.8 * x[:, 1:] + np.random.normal(0, 0.08, size=(64, 1))
        from minigrad.optim import Adam

        opt, lf = Adam(layer.parameters(), lr=0.04), GaussianNLLLoss()
        for _ in range(50):
            opt.zero_grad()
            lf(layer(DistributionalTensor(x, np.full_like(x, 0.002))), Tensor(y)).backward()
            opt.step()
        return layer

    def var(layer: DistributionalLinear, p: List[float]) -> float:
        return float(np.mean(layer(DistributionalTensor([p], [[0.002, 0.002]])).var.numpy()))

    shifted = trained(4.0)  # (3.5,-4.0) is IN-support, the origin is unseen
    assert var(shifted, [3.5, -4.0]) > 5 * var(shifted, [0.2, -0.3])


def test_heteroscedastic_mlp_learns_input_dependent_variance():
    from minigrad.optim import Adam

    np.random.seed(42)
    x = np.random.uniform(-2, 2, size=(200, 1))
    sigma = 0.2 + 0.5 * np.abs(x)
    y = 2 * x + 0.5 + np.random.normal(0, sigma)
    model = HeteroscedasticMLP(in_features=1, hidden_features=24, out_features=1)
    opt, lf = Adam(model.parameters(), lr=0.03), GaussianNLLLoss()
    xt, yt = Tensor(x), Tensor(y)
    for _ in range(150):
        opt.zero_grad()
        lf(model(DistributionalTensor(xt, Tensor(np.full_like(x, 0.001)))), yt).backward()
        opt.step()
    probe = np.array([[0.0], [2.0]])
    _, v = model(DistributionalTensor(probe, np.full_like(probe, 0.001))).numpy()
    assert v[1, 0] > 4 * v[0, 0]  # truth: 1.44 / 0.04 = 36x; require a clearly input-dependent profile
    assert 0.5 < v[1, 0] < 2.5 and v[0, 0] < 0.4


# ── T.A.R.K.A. ───────────────────────────────────────────────────────────────

def test_godel_implication_is_the_true_residuum():
    a, b = np.linspace(0, 1, 11), np.linspace(1, 0, 11)
    got = (LogicTensor(Tensor(a), tnorm=TNorm.GODEL) >> LogicTensor(Tensor(b), tnorm=TNorm.GODEL)).tensor.data
    np.testing.assert_allclose(got, np.where(a <= b, 1.0, b), atol=1e-12)
    luk = (LogicTensor(Tensor(a), tnorm=TNorm.LUKASIEWICZ) >> LogicTensor(Tensor(b), tnorm=TNorm.LUKASIEWICZ)).tensor.data
    assert not np.allclose(got, luk)


def test_forall_gradient_share_depends_on_violation_gap_over_tau():
    def share(v: float) -> float:
        p = np.full(8, 0.95)
        p[4] = v
        t = Tensor(p, requires_grad=True)
        LogicTensor(t).forall(tau=0.05).semantic_loss(method="linear").backward()
        g = np.abs(t.grad)
        return float(g[4] / g.sum())

    shares = [share(v) for v in (0.10, 0.50, 0.80, 0.90, 0.93)]  # gaps 0.85 .. 0.02
    assert shares[0] > 0.99 and shares[1] > 0.95  # large gap: concentrated
    assert shares[-1] < 0.30  # small gap: close to uniform (1/8 = 0.125)
    assert all(a > b for a, b in zip(shares, shares[1:]))  # monotone in the gap


def test_constant_relations_satisfy_transitivity_trivially():
    for c in (0.0, 1.0):
        rc = LogicTensor(Tensor(np.full((4, 4), c)))
        assert float(((rc & rc) >> rc).tensor.data.mean()) == pytest.approx(1.0)


# ── S.P.A.N.D.A.: energy accounting ──────────────────────────────────────────

def test_spanda_energy_accounting_uses_analog_layer0_macs_and_input_spike_acs():
    np.random.seed(5)
    model = SpikingSequential([SpikingLinear(8, 16, beta=0.9, v_th=0.5), SpikingLinear(16, 4, beta=0.9, v_th=0.5)])
    B, T = 6, 8
    t = SPANDA.evaluate_neuromorphic_energy(model, Tensor(np.random.randn(B, 8) * 2), num_steps=T)
    l0, l1 = model.layers
    assert t.dense_macs == B * (8 * 16 + 16 * 4)  # single-pass ANN
    assert t.snn_macs == B * 8 * 16  # layer 0 sees analog input: real MACs, computed once
    assert t.spiking_acs == int(round(float(np.sum(l0.last_spikes)) * 4))  # driven by layer-0 OUTPUT spikes
    assert t.neuron_updates == T * B * (16 + 4)
    ann_pj = t.dense_macs * 4.6
    cons = t.snn_macs * 4.6 + t.spiking_acs * 0.9 + t.neuron_updates * 4.6
    opt = t.snn_macs * 4.6 + t.spiking_acs * 0.9 + t.neuron_updates * 0.9
    assert t.energy_efficiency_gain == pytest.approx(ann_pj / cons)
    assert t.energy_efficiency_gain_optimistic == pytest.approx(ann_pj / opt)
    assert t.energy_efficiency_gain_optimistic >= t.energy_efficiency_gain


# ── numpy-scalar / graph-depth regressions (value-checked) ───────────────────

def test_deep_graph_gradient_value_is_exact():
    x = Tensor(np.array([1.0]), requires_grad=True)
    y = x
    for _ in range(3000):
        y = y * 1.0001
    y.sum().backward()
    assert x.grad[0] == pytest.approx(1.0001 ** 3000, rel=1e-9)
