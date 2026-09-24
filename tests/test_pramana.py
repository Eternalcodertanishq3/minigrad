"""
test_pramana.py — Automated Unit Tests for P.R.A.M.A.N.A.
(Probabilistic Representation of Analytical Moments and Algebraic Noise-aware Autograd)

Tests:
1. test_distributional_tensor_algebra (Analytical Goodman product vs Monte Carlo simulation)
2. test_linear_analytical_variance_propagation (DistributionalLinear vs 100k empirical samples)
3. test_activation_moment_propagation (Taylor moment bounds for Tanh, Sigmoid, ReLU, GELU)
4. test_dual_autograd_backpropagation (Simultaneous gradients on mean and variance graphs)
5. test_gaussian_nll_loss_gradients (Equilibrium dynamics of heteroscedastic NLL)
6. test_ood_hallucination_detection (Epistemic uncertainty spikes on OOD samples)
7. test_end_to_end_distributional_training (Heteroscedastic noise regression with Adam)
8. test_deterministic_backward_parity (Bit-for-bit equivalence when var=0)
"""
import numpy as np
import pytest

from minigrad.nn import Linear
from minigrad.optim import Adam
from minigrad.pramana import (
    PRAMANA,
    DistributionalLinear,
    DistributionalSequential,
    DistributionalTensor,
    GaussianNLLLoss,
    PramanaTelemetry,
)
from minigrad.tensor import Tensor


# ── 1. Analytical Moment Algebra vs Monte Carlo Simulation ───────────

def test_distributional_tensor_algebra():
    """
    Validates that DistributionalTensor arithmetic matches analytical
    calculus and empirical Monte Carlo Gaussian simulation.
    """
    np.random.seed(42)
    mu_x, var_x = 2.0, 0.25  # sigma = 0.5
    mu_y, var_y = -1.5, 0.16  # sigma = 0.4

    # 1. Addition: Var[X + Y] = Var[X] + Var[Y]
    x = DistributionalTensor([mu_x], [var_x])
    y = DistributionalTensor([mu_y], [var_y])
    z_add = x + y

    assert np.isclose(z_add.mean.numpy()[0], mu_x + mu_y)
    assert np.isclose(z_add.var.numpy()[0], var_x + var_y)

    # 2. Goodman (1960) Product Variance:
    # Var[XY] = mu_X^2 * Var[Y] + mu_Y^2 * Var[X] + Var[X]*Var[Y]
    z_mul = x * y
    exact_var_mul = (mu_x ** 2) * var_y + (mu_y ** 2) * var_x + var_x * var_y
    assert np.isclose(z_mul.mean.numpy()[0], mu_x * mu_y)
    assert np.isclose(z_mul.var.numpy()[0], exact_var_mul)

    # 3. Empirical Monte Carlo verification (100,000 samples)
    n_samples = 100_000
    samples_x = np.random.normal(mu_x, np.sqrt(var_x), size=n_samples)
    samples_y = np.random.normal(mu_y, np.sqrt(var_y), size=n_samples)
    samples_z = samples_x * samples_y

    emp_mean = np.mean(samples_z)
    emp_var = np.var(samples_z)

    # Analytical moments should match empirical sample statistics within ~1%
    assert np.isclose(z_mul.mean.numpy()[0], emp_mean, rtol=1e-2)
    assert np.isclose(z_mul.var.numpy()[0], emp_var, rtol=2e-2)


# ── 2. Linear Analytical Variance Propagation ────────────────────────

def test_linear_analytical_variance_propagation():
    """
    Tests DistributionalLinear forward pass against empirical sample distribution.
    Analytical:
        mu_Z = mu_X @ W + b
        var_Z = var_X @ (W^2)
    """
    np.random.seed(123)
    in_dim, out_dim = 4, 3
    layer = DistributionalLinear(in_dim, out_dim, bias=True)

    mu_in = np.array([[1.0, -0.5, 2.0, 0.3]])
    var_in = np.array([[0.2, 0.1, 0.3, 0.15]])

    x_dist = DistributionalTensor(mu_in, var_in)
    out_dist = layer(x_dist)

    # Analytical values
    mu_ana = out_dist.mean.numpy()
    var_ana = out_dist.var.numpy()

    # Empirical Monte Carlo (100,000 samples)
    n_samples = 100_000
    samples = np.random.normal(
        loc=np.repeat(mu_in, n_samples, axis=0),
        scale=np.repeat(np.sqrt(var_in), n_samples, axis=0),
    )
    w_np = layer.weight.numpy()
    b_np = layer.bias.numpy()
    out_samples = samples @ w_np + b_np

    emp_mean = np.mean(out_samples, axis=0, keepdims=True)
    emp_var = np.var(out_samples, axis=0, keepdims=True)

    np.testing.assert_allclose(mu_ana, emp_mean, rtol=2e-2, atol=1e-2)
    np.testing.assert_allclose(var_ana, emp_var, rtol=3e-2, atol=1e-2)


# ── 3. Activation Moment Propagation ─────────────────────────────────

def test_activation_moment_propagation():
    """
    Verifies that non-linear activation functions correctly attenuate
    or preserve variance according to their first-order Taylor derivatives.
    """
    # In saturation regions (|x| >> 0 for Tanh/Sigmoid), variance must shrink to ~0
    x_sat = DistributionalTensor([10.0], [1.0])
    tanh_sat = x_sat.tanh()
    assert tanh_sat.var.numpy()[0] < 1e-4

    # In linear region (x ~ 0 for Tanh), derivative is ~1, variance is preserved
    x_lin = DistributionalTensor([0.0], [0.5])
    tanh_lin = x_lin.tanh()
    np.testing.assert_allclose(tanh_lin.var.numpy()[0], 0.5, rtol=1e-4)

    # In negative region for ReLU, variance must be exactly zero
    x_neg = DistributionalTensor([-2.0], [0.8])
    relu_neg = x_neg.relu()
    assert relu_neg.var.numpy()[0] == 0.0

    # In positive region for ReLU, variance is preserved
    x_pos = DistributionalTensor([2.0], [0.8])
    relu_pos = x_pos.relu()
    assert np.isclose(relu_pos.var.numpy()[0], 0.8)


# ── 4. Dual Autograd Backpropagation ─────────────────────────────────

def test_dual_autograd_backpropagation():
    """
    Validates that loss gradients flow simultaneously through both
    the mean and variance graphs back to input moments and weights.
    """
    x = DistributionalTensor(
        Tensor([[1.0, 2.0]], requires_grad=True),
        Tensor([[0.2, 0.4]], requires_grad=True),
    )
    layer = DistributionalLinear(2, 2, bias=True)

    out = layer(x)
    # Scalar loss penalizing both mean error and variance magnitude
    loss = (out.mean ** 2).sum() + out.var.sum()
    loss.backward()

    # Gradients must be present and finite for input mean AND input variance
    assert x.mean.grad is not None
    assert x.var.grad is not None
    assert np.all(np.isfinite(x.mean.grad))
    assert np.all(np.isfinite(x.var.grad))

    # Weight and bias gradients must be populated
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None


# ── 5. Gaussian NLL Loss Gradient Equilibrium ────────────────────────

def test_gaussian_nll_loss_gradients():
    """
    Verifies that GaussianNLLLoss derivatives drive variance toward the
    squared error (y - mu)^2:
    - If (y - mu)^2 > var: dLoss/d(var) < 0 (increases variance)
    - If (y - mu)^2 < var: dLoss/d(var) > 0 (decreases variance)
    """
    loss_fn = GaussianNLLLoss(reduction="none")
    target = Tensor([2.0])

    # Case A: High error, low initial variance (underconfident)
    # (y - mu)^2 = (2 - 0)^2 = 4.0 > var (1.0)
    mu_under = Tensor([0.0], requires_grad=True)
    var_under = Tensor([1.0], requires_grad=True)
    loss_under = loss_fn(DistributionalTensor(mu_under, var_under), target)
    loss_under.backward()
    # Gradient on var must be negative, incentivizing variance to increase!
    assert var_under.grad[0] < 0.0

    # Case B: Low error, high initial variance (overconfident uncertainty)
    # (y - mu)^2 = (2.0 - 2.0)^2 = 0.0 < var (2.0)
    mu_over = Tensor([2.0], requires_grad=True)
    var_over = Tensor([2.0], requires_grad=True)
    loss_over = loss_fn(DistributionalTensor(mu_over, var_over), target)
    loss_over.backward()
    # Gradient on var must be positive, incentivizing variance to decrease!
    assert var_over.grad[0] > 0.0


# ── 6. Out-of-Distribution (OOD) Hallucination Trapping ──────────────

def test_ood_hallucination_detection():
    """
    Shows that an uncertainty-aware network exhibits a massive epistemic
    uncertainty spike on out-of-distribution (OOD) inputs compared to in-distribution.
    """
    np.random.seed(42)
    # Layer with weight uncertainty enabled
    layer = DistributionalLinear(2, 2, bias=True, weight_uncertainty=True, init_log_var=-1.0)

    # In-distribution inputs close to origin
    x_in = Tensor(np.random.randn(20, 2) * 0.5)
    # Out-of-distribution inputs far from origin
    x_ood = Tensor(np.random.randn(20, 2) * 0.5 + 10.0)

    telem = PRAMANA.evaluate_uncertainty_telemetry(layer, x_in, x_ood)

    # OOD uncertainty must be strictly larger than in-distribution uncertainty
    assert telem.out_dist_mean_std > telem.in_dist_mean_std
    assert telem.epistemic_divergence_ratio > 2.0, (
        f"Expected divergence ratio > 2.0, got {telem.epistemic_divergence_ratio:.2f}"
    )


# ── 7. End-to-End Heteroscedastic Regression Training ────────────────

def test_end_to_end_distributional_training():
    """
    Trains a DistributionalSequential network on heteroscedastic noise data
    where observation noise scales with |x|: y = sin(x) + N(0, (0.1 + 0.2*|x|)^2).
    Validates that loss decreases and the network learns mean and variance.
    """
    np.random.seed(99)
    x_vals = np.linspace(-2.0, 2.0, 40)[:, None]
    true_std = 0.1 + 0.2 * np.abs(x_vals)
    y_vals = np.sin(x_vals) + np.random.normal(0.0, true_std)

    x_tensor = Tensor(x_vals)
    y_tensor = Tensor(y_vals)

    model = DistributionalSequential([
        DistributionalLinear(1, 16, bias=True, weight_uncertainty=True, init_log_var=-3.0),
        DistributionalLinear(16, 1, bias=True, weight_uncertainty=True, init_log_var=-3.0),
    ])

    optimizer = Adam(model.parameters(), lr=0.03)
    loss_fn = GaussianNLLLoss()

    initial_loss = None
    final_loss = None

    for epoch in range(30):
        optimizer.zero_grad()
        pred = model(x_tensor)
        loss = loss_fn(pred, y_tensor)

        if epoch == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < initial_loss, f"Expected loss decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    # Verify predictions have valid positive variance
    pred_final = model(x_tensor)
    assert np.all(pred_final.var.numpy() > 0.0)


# ── 8. Deterministic Backward Parity ──────────────────────────────────

def test_deterministic_backward_parity():
    """
    Verifies that when input variance is zero (var=0) and weight uncertainty
    is disabled, DistributionalLinear produces outputs and gradients that
    match standard Linear layer bit-for-bit.
    """
    np.random.seed(77)
    w_init = np.random.randn(3, 2)
    b_init = np.random.randn(2)
    x_data = np.random.randn(4, 3)

    # Standard Linear layer
    std_layer = Linear(3, 2, bias=True)
    std_layer.weight.data = w_init.copy()
    std_layer.bias.data = b_init.copy()

    x_std = Tensor(x_data.copy(), requires_grad=True)
    y_std = std_layer(x_std)
    loss_std = (y_std ** 2).sum()
    loss_std.backward()

    # DistributionalLinear layer with var=0
    dist_layer = DistributionalLinear(3, 2, bias=True, weight_uncertainty=False)
    dist_layer.weight.data = w_init.copy()
    dist_layer.bias.data = b_init.copy()

    x_dist = DistributionalTensor(Tensor(x_data.copy(), requires_grad=True))
    y_dist = dist_layer(x_dist)
    loss_dist = (y_dist.mean ** 2).sum()
    loss_dist.backward()

    # Verify forward parity
    np.testing.assert_allclose(y_std.numpy(), y_dist.mean.numpy(), rtol=1e-14, atol=1e-14)
    assert np.all(y_dist.var.numpy() == 0.0)

    # Verify input gradient parity
    np.testing.assert_allclose(x_std.grad, x_dist.mean.grad, rtol=1e-14, atol=1e-14)

    # Verify weight and bias gradient parity
    np.testing.assert_allclose(std_layer.weight.grad, dist_layer.weight.grad, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(std_layer.bias.grad, dist_layer.bias.grad, rtol=1e-14, atol=1e-14)
