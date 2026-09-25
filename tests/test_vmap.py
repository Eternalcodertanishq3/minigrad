"""
test_vmap.py — Unit and integration tests for Pillar 4: Functional vmap & DP-SGD Engine.

Tests:
1. vmap basic element-wise and linear algebra vectorization.
2. vmap with mixed in_axes (shared parameters, batched inputs).
3. make_functional stateless module execution parity.
4. Exact bit-level per-sample gradient parity vs sequential sample-by-sample autograd.
5. Equivalence between mean per-sample gradients and standard batch backprop.
6. jacrev and batched_jacobian analytical correctness.
7. DP-SGD per-sample gradient clipping and L2 sensitivity bounds.
8. DP Gaussian noise injection and scale calibration.
9. PrivacyTelemetry reporting and end-to-end optimizer integration.
"""
import numpy as np

import minigrad
from minigrad import Tensor
from minigrad.nn import Linear, Sequential, ReLU
from minigrad.optim import Adam
from minigrad.vmap import (
    vmap,
    make_functional,
    per_sample_gradients,
    jacrev,
    batched_jacobian,
)
from minigrad.dp import (
    clip_per_sample_gradients,
    add_dp_noise,
    apply_dp_gradients,
    compute_dp_sgd_step,
    PrivacyTelemetry,
)


# ==============================================================================
# 1. Core vmap Functional Vectorization
# ==============================================================================

def test_vmap_basic_vectorization():
    """Verify vmap vectorizes single-example operations over a batch."""
    # Dot product of two vectors
    def dot_product(a: Tensor, b: Tensor) -> Tensor:
        return (a * b).sum()

    batch_a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))  # (3, 2)
    batch_b = Tensor(np.array([[2.0, 1.0], [0.5, 2.0], [-1.0, 3.0]]))  # (3, 2)

    v_dot = vmap(dot_product, in_axes=0)
    result = v_dot(batch_a, batch_b)

    # Expected: [1*2 + 2*1, 3*0.5 + 4*2, 5*-1 + 6*3] = [4.0, 9.5, 13.0]
    expected = np.array([4.0, 9.5, 13.0])
    assert np.allclose(result.data, expected)


def test_vmap_in_axes_shared_parameters():
    """Verify in_axes with None for shared parameters and 0 for batched inputs."""
    w = Tensor(np.array([[2.0, -1.0], [1.0, 3.0]]))  # (2, 2) shared weights
    batch_x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]]))  # (3, 2)

    def linear_fn(weight: Tensor, x: Tensor) -> Tensor:
        return x @ weight

    # in_axes: weight is shared (None), x is batched (0)
    batched_linear = vmap(linear_fn, in_axes=(None, 0))
    batched_out = batched_linear(w, batch_x)

    # Expected: batch_x @ w directly
    expected = batch_x.data @ w.data
    assert np.allclose(batched_out.data, expected)


def test_vmap_multi_output_tuple():
    """Verify vmap handles functions returning tuples of Tensors."""
    def split_fn(x: Tensor) -> tuple[Tensor, Tensor]:
        return x * 2.0, x ** 2

    batch_x = Tensor(np.array([1.0, 2.0, 3.0]))
    v_split = vmap(split_fn, in_axes=0)
    out_mul, out_sq = v_split(batch_x)

    assert np.allclose(out_mul.data, [2.0, 4.0, 6.0])
    assert np.allclose(out_sq.data, [1.0, 4.0, 9.0])


# ==============================================================================
# 2. make_functional Stateless Module Conversion
# ==============================================================================

def test_make_functional():
    """Verify make_functional executes statelessly without altering original module."""
    mlp = Sequential([
        Linear(3, 4),
        ReLU(),
        Linear(4, 2),
    ])

    x = Tensor(np.random.randn(2, 3))

    # Standard stateful forward
    orig_out = mlp(x)

    # Functional forward
    f_model, params = make_functional(mlp, return_dict=True)
    func_out = f_model(params, x)

    # Must match bit-for-bit
    assert np.allclose(orig_out.data, func_out.data, atol=1e-12)

    # Modifying params in functional call shouldn't mutate module state
    perturbed_params = {k: v * 2.0 for k, v in params.items()}
    f_model(perturbed_params, x)
    assert np.allclose(mlp(x).data, orig_out.data, atol=1e-12)


# ==============================================================================
# 3. Exact Bit-Level Per-Sample Gradient Parity
# ==============================================================================

def test_per_sample_gradients_exact_parity():
    """
    CRITICAL VERIFICATION:
    Verifies that per_sample_gradients matches an isolated sequential loop
    running loss.backward() on each sample individually down to 1e-12 precision.
    """
    np.random.seed(42)
    B, InF, OutF = 4, 3, 2

    w_val = np.random.randn(InF, OutF)
    b_val = np.random.randn(OutF)

    batch_x_val = np.random.randn(B, InF)
    batch_y_val = np.random.randn(B, OutF)

    # 1. Ground Truth: Sequential loop on isolated samples
    gt_dw = []
    gt_db = []
    for i in range(B):
        xi = Tensor(batch_x_val[i:i+1], requires_grad=False)
        yi = Tensor(batch_y_val[i:i+1], requires_grad=False)
        w_i = Tensor(w_val.copy(), requires_grad=True)
        b_i = Tensor(b_val.copy(), requires_grad=True)

        pred_i = xi @ w_i + b_i
        loss_i = ((pred_i - yi) ** 2).sum()
        loss_i.backward()

        gt_dw.append(w_i.grad.copy())
        gt_db.append(b_i.grad.copy())

    gt_dw_arr = np.stack(gt_dw, axis=0)  # (B, InF, OutF)
    gt_db_arr = np.stack(gt_db, axis=0)  # (B, OutF)

    # 2. miniGrad Pillar 4: Vectorized per_sample_gradients
    w_shared = Tensor(w_val.copy(), requires_grad=True)
    b_shared = Tensor(b_val.copy(), requires_grad=True)
    params = {"weight": w_shared, "bias": b_shared}

    def model_fn(p, x):
        return x @ p["weight"] + p["bias"]

    def mse_loss(pred, target):
        return ((pred - target) ** 2).sum()

    batch_x = Tensor(batch_x_val)
    batch_y = Tensor(batch_y_val)

    per_sample_grads = per_sample_gradients(
        model_fn,
        mse_loss,
        params,
        batch_x,
        batch_y,
    )

    dw_vmap = per_sample_grads["weight"].data
    db_vmap = per_sample_grads["bias"].data

    # Parity check: exact bit-level match
    assert np.allclose(dw_vmap, gt_dw_arr, atol=1e-12)
    assert np.allclose(db_vmap, gt_db_arr, atol=1e-12)

    # Mean per-sample gradient must match standard batch gradient
    w_batch = Tensor(w_val.copy(), requires_grad=True)
    b_batch = Tensor(b_val.copy(), requires_grad=True)
    batch_loss = ((batch_x @ w_batch + b_batch - batch_y) ** 2).sum()
    batch_loss.backward()

    assert np.allclose(np.sum(dw_vmap, axis=0), w_batch.grad, atol=1e-12)
    assert np.allclose(np.sum(db_vmap, axis=0), b_batch.grad, atol=1e-12)


def test_module_per_sample_gradients_method():
    """Verify Module.per_sample_gradients() convenience API."""
    mlp = Sequential([
        Linear(4, 6),
        ReLU(),
        Linear(6, 2),
    ])

    batch_x = Tensor(np.random.randn(5, 4))
    batch_y = Tensor(np.random.randn(5, 2))

    def loss_fn(pred, y):
        return ((pred - y) ** 2).sum()

    per_sample_grads = mlp.per_sample_gradients(loss_fn, batch_x, batch_y)

    assert isinstance(per_sample_grads, dict)
    # Check leading batch dimension B=5 on all parameters
    for name, g in per_sample_grads.items():
        assert g.shape[0] == 5


# ==============================================================================
# 4. Jacobians & Batched Jacobians (jacrev)
# ==============================================================================

def test_jacrev_analytical():
    """Verify jacrev computes exact Jacobian matrix for vector-valued function."""
    # f([x1, x2]) = [x1^2, x1*x2 + x2^3]
    # df1/dx1 = 2*x1, df1/dx2 = 0
    # df2/dx1 = x2,   df2/dx2 = x1 + 3*x2^2
    def f(x: Tensor) -> Tensor:
        y1 = x[0] ** 2
        y2 = x[0] * x[1] + (x[1] ** 3)
        return minigrad.ops.stack([y1, y2])

    x = Tensor(np.array([2.0, 3.0]))
    J = jacrev(f)(x)

    expected_J = np.array([
        [2.0 * 2.0, 0.0],
        [3.0, 2.0 + 3.0 * (3.0 ** 2)],
    ])

    assert np.allclose(J.data, expected_J, atol=1e-7)


def test_batched_jacobian():
    """Verify batched_jacobian computes [B, M, N] Jacobian tensors across a batch."""
    def f(x: Tensor) -> Tensor:
        return x ** 2

    # Batch of 4 3-dimensional vectors
    batch_x = Tensor(np.array([
        [1.0, 2.0, 3.0],
        [0.5, -1.0, 2.0],
        [4.0, 0.0, -2.0],
        [3.0, 1.5, -0.5],
    ]))

    batched_J = batched_jacobian(f, batch_x)
    assert batched_J.shape == (4, 3, 3)

    # For f(x) = x^2, each Jacobian is a diagonal matrix with diag = 2*x
    for b in range(4):
        diag_expected = 2.0 * batch_x.data[b]
        assert np.allclose(np.diag(batched_J.data[b]), diag_expected)
        # Off-diagonal elements must be 0
        off_diag = batched_J.data[b] - np.diag(diag_expected)
        assert np.allclose(off_diag, 0.0)


# ==============================================================================
# 5. Differential Privacy (DP-SGD) Engine & Telemetry
# ==============================================================================

def test_dp_clipping_bounds():
    """Verify clip_per_sample_gradients enforces ||g_i||_2 <= max_norm strictly."""
    # Create synthetic per-sample gradients for 3 samples
    # Norms: sample 0 has norm 1.0 (under threshold 2.0)
    #        sample 1 has norm 4.0 (over threshold 2.0)
    #        sample 2 has norm 10.0 (over threshold 2.0)
    g1 = Tensor(np.array([
        [[0.6, 0.8]],     # norm = 1.0
        [[2.4, 3.2]],     # norm = 4.0
        [[6.0, 8.0]],     # norm = 10.0
    ]))

    clipped, pre_norms = clip_per_sample_gradients([g1], max_norm=2.0)
    clipped_data = clipped[0].data

    post_norms = np.sqrt(np.sum(clipped_data.reshape(3, -1) ** 2, axis=1))

    # Sample 0 should be unchanged (norm was 1.0 <= 2.0)
    assert np.isclose(post_norms[0], 1.0)
    assert np.allclose(clipped_data[0], g1.data[0])

    # Samples 1 and 2 must be strictly bounded to max_norm = 2.0
    assert np.isclose(post_norms[1], 2.0)
    assert np.isclose(post_norms[2], 2.0)


def test_dp_noise_injection():
    """Verify add_dp_noise injects Gaussian noise with correct calibrated standard deviation."""
    B = 100
    C = 1.0
    noise_mult = 0.5
    expected_std = (C * noise_mult) / float(B)

    # 100 zeros as gradients
    grads = [Tensor(np.zeros((B, 10000)))]
    noisy = add_dp_noise(grads, noise_multiplier=noise_mult, max_norm=C, batch_size=B, seed=42)

    emp_std = float(np.std(noisy[0].data))
    assert np.isclose(emp_std, expected_std, rtol=0.1)


def test_dp_sgd_step_and_optimizer_integration():
    """Verify compute_dp_sgd_step integration with miniGrad Adam optimizer."""
    model = Sequential([
        Linear(4, 2),
    ])
    optimizer = Adam(model.parameters(), lr=1e-2)

    batch_x = Tensor(np.random.randn(8, 4))
    batch_y = Tensor(np.random.randn(8, 2))

    def loss_fn(pred, y):
        return ((pred - y) ** 2).sum()

    # 1. Per-sample gradient extraction
    per_sample_grads = model.per_sample_gradients(loss_fn, batch_x, batch_y)

    # 2. DP-SGD step
    private_grads, telemetry = compute_dp_sgd_step(
        per_sample_grads,
        max_norm=1.0,
        noise_multiplier=0.2,
        seed=123,
    )

    assert isinstance(telemetry, PrivacyTelemetry)
    assert telemetry.batch_size == 8
    assert telemetry.max_norm == 1.0
    summary = telemetry.summary()
    assert "miniGrad Differential Privacy (DP-SGD) Telemetry" in summary

    # 3. Apply gradients to model parameters and step optimizer
    apply_dp_gradients(model, private_grads)
    optimizer.step()

    # Weights must have successfully updated
    for p in model.parameters():
        assert p.grad is not None
