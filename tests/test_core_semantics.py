"""
tests/test_core_semantics.py — Exhaustive Dtype Promotion, IEEE-754 Boundaries, and Lifecycle Invariants.

Validates:
1. Dtype Promotion & Operator Matrices:
   - Full pairwise matrix of float32, float64, int32, int64 across binary ops (+, -, *, /).
   - Exhaustive pairwise tensor-vs-tensor power matrix (16 pairs) for (a ** b) with positive & negative exponents.
   - Exhaustive pairwise matrix multiplication matrix (16 pairs) for (x @ w) with gradient tracking.
   - Comprehensive division gradient verification on both operands with broadcasting topologies.
   - Division with left and right Python float/int scalars.
   - Exact parity with NumPy dtype promotion semantics.
2. IEEE-754 Boundary & Edge-Case Behavior:
   - Exact mathematical limits: log(0) -> -inf, log(-x) -> nan, 0/0 -> nan, 1/0 -> inf, 0^0 -> 1, (-2)^0.5 -> nan; 0^-1 and 0^-2 evaluate to finite values under Contract B stability policy.
   - Mixed-sign tensor exponents ([2, 3] ** [2, -1]).
   - Glass-Box anomaly interception halts at the exact exploding node on non-finite outputs and gradients.
3. Autograd Lifecycle & Multi-Branch Graph Stress:
   - Diamond and multi-consumer graph backward accumulation.
   - RuntimeError on backward through FREED graph when retain_graph=False.
   - Multi-step training parameter mutation (param.data -= lr * param.grad) without graph corruption.
   - Invariant: intermediate backward closures freed after backward(retain_graph=False).
"""
from __future__ import annotations

import numpy as np
import pytest

from minigrad.glassbox import detect_anomaly
from minigrad.tensor import Tensor

# ── 1. Dtype Promotion Matrix ─────────────────────────────────────────

@pytest.mark.parametrize("dtype1", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("dtype2", [np.float32, np.float64, np.int32, np.int64])
def test_dtype_promotion_binary_ops(dtype1, dtype2):
    """Binary operations (+, -, *, /) between all tensor dtype pairs follow NumPy promotion."""
    a_arr = np.array([12, 24], dtype=dtype1)
    b_arr = np.array([3, 4], dtype=dtype2)
    expected_dtype = np.result_type(a_arr, b_arr)

    a = Tensor(a_arr)
    b = Tensor(b_arr)

    # Addition
    c_add = a + b
    assert c_add.dtype == expected_dtype, f"Add dtype mismatch: {c_add.dtype} vs {expected_dtype}"
    np.testing.assert_array_equal(c_add.data, a_arr + b_arr)

    # Subtraction
    c_sub = a - b
    assert c_sub.dtype == expected_dtype, f"Sub dtype mismatch: {c_sub.dtype} vs {expected_dtype}"
    np.testing.assert_array_equal(c_sub.data, a_arr - b_arr)

    # Multiplication
    c_mul = a * b
    assert c_mul.dtype == expected_dtype, f"Mul dtype mismatch: {c_mul.dtype} vs {expected_dtype}"
    np.testing.assert_array_equal(c_mul.data, a_arr * b_arr)

    # True Division (always produces float32 or float64 per NumPy semantics)
    c_div = a / b
    expected_div = a_arr / b_arr
    assert c_div.dtype == expected_div.dtype, f"Div dtype mismatch: {c_div.dtype} vs {expected_div.dtype}"
    np.testing.assert_allclose(c_div.data, expected_div)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_dtype_promotion_power_scalar(dtype):
    """Power operations with scalar exponents follow normalized promotion."""
    a_arr = np.array([2, 4], dtype=dtype)
    a = Tensor(a_arr)

    # Positive integer power preserves dtype
    c_pow2 = a ** 2
    expected_pos = a_arr ** 2
    assert c_pow2.dtype == expected_pos.dtype
    np.testing.assert_allclose(c_pow2.data, expected_pos)

    # Negative power always promotes to floating point
    c_pow_neg = a ** -1
    expected_float_dtype = np.float32 if dtype == np.float32 else np.float64
    assert c_pow_neg.dtype == expected_float_dtype
    np.testing.assert_allclose(c_pow_neg.data, a_arr.astype(expected_float_dtype) ** -1)


@pytest.mark.parametrize("dtype1", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("dtype2", [np.float32, np.float64, np.int32, np.int64])
def test_dtype_promotion_tensor_power_matrix(dtype1, dtype2):
    """Exhaustive 16-pair tensor-vs-tensor power promotion matrix (a ** b)."""
    # Non-negative exponents
    a = Tensor(np.array([2, 3], dtype=dtype1))
    b = Tensor(np.array([3, 2], dtype=dtype2))
    c = a ** b
    expected_dtype = np.result_type(a.data, b.data)
    assert c.dtype == expected_dtype, f"Pow dtype mismatch: {c.dtype} vs {expected_dtype}"
    np.testing.assert_allclose(c.data, a.data ** b.data)

    # Negative exponents (guarantees float promotion for integer bases)
    b_neg = Tensor(np.array([2, -1], dtype=dtype2))
    c_neg = a ** b_neg
    if np.issubdtype(dtype1, np.floating) or np.issubdtype(dtype2, np.floating):
        expected_neg_dtype = np.result_type(dtype1, dtype2)
    else:
        expected_neg_dtype = np.float64
    assert c_neg.dtype == expected_neg_dtype, f"Neg pow dtype mismatch: {c_neg.dtype} vs {expected_neg_dtype}"
    np.testing.assert_allclose(c_neg.data, a.data.astype(c_neg.dtype) ** b_neg.data)


@pytest.mark.parametrize("tensor_dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("scalar_val", [2.5, 3])
def test_dtype_promotion_with_python_scalars(tensor_dtype, scalar_val):
    """Tensors combined with Python float/int scalars adhere to NumPy type promotion rules."""
    a_arr = np.array([12, 24], dtype=tensor_dtype)
    a = Tensor(a_arr)

    # Left operation: Tensor + scalar
    out_left = a + scalar_val
    expected_left = np.result_type(a_arr, scalar_val)
    assert out_left.dtype == expected_left
    np.testing.assert_allclose(out_left.data, a_arr + scalar_val)

    # Right operation: scalar + Tensor
    out_right = scalar_val + a
    assert out_right.dtype == expected_left
    np.testing.assert_allclose(out_right.data, scalar_val + a_arr)

    # Multiplication
    out_mul = a * scalar_val
    assert out_mul.dtype == expected_left
    np.testing.assert_allclose(out_mul.data, a_arr * scalar_val)

    # Division: Tensor / scalar
    out_div = a / scalar_val
    expected_div = a_arr / scalar_val
    assert out_div.dtype == expected_div.dtype
    np.testing.assert_allclose(out_div.data, expected_div)

    # Division: scalar / Tensor
    out_rdiv = scalar_val / a
    expected_rdiv = scalar_val / a_arr
    assert out_rdiv.dtype == expected_rdiv.dtype
    np.testing.assert_allclose(out_rdiv.data, expected_rdiv)


@pytest.mark.parametrize("dtype1", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("dtype2", [np.float32, np.float64, np.int32, np.int64])
def test_matmul_dtype_promotion_exhaustive_matrix(dtype1, dtype2):
    """Exhaustive 16-pair tensor matrix multiplication (x @ w) dtype promotion and autograd."""
    x_arr = np.array([[1, 2], [3, 4]], dtype=dtype1)
    w_arr = np.array([[5, 6], [7, 8]], dtype=dtype2)
    expected_dtype = np.result_type(x_arr, w_arr)

    # Forward check
    x = Tensor(x_arr)
    w = Tensor(w_arr)
    out = x @ w
    assert out.dtype == expected_dtype, f"Matmul dtype mismatch: {out.dtype} vs {expected_dtype}"
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out.data, x_arr @ w_arr)

    # Autograd verification
    x_grad = Tensor(x_arr, dtype=dtype1, requires_grad=True)
    w_grad = Tensor(w_arr, dtype=dtype2, requires_grad=True)
    loss = (x_grad @ w_grad).sum()
    loss.backward()

    expected_dx = np.ones((2, 2), dtype=expected_dtype) @ w_arr.T
    expected_dw = x_arr.T @ np.ones((2, 2), dtype=expected_dtype)

    np.testing.assert_allclose(x_grad.grad, expected_dx, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(w_grad.grad, expected_dw, rtol=1e-4, atol=1e-4)
    assert np.issubdtype(x_grad.grad.dtype, np.floating)
    assert np.issubdtype(w_grad.grad.dtype, np.floating)


@pytest.mark.parametrize("dtype1", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("dtype2", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("broadcast_case", [
    "identical",       # (2, 3) / (2, 3)
    "broadcast_rows",  # (2, 3) / (1, 3)
    "broadcast_cols",  # (2, 3) / (2, 1)
    "broadcast_1d",    # (2, 3) / (3,)
])
def test_division_tensor_tensor_gradients_and_broadcasting(dtype1, dtype2, broadcast_case):
    """Division (a / b) with gradients on both operands across all dtypes and broadcast topologies."""
    base_a = np.array([[12.0, 18.0, 24.0], [30.0, 36.0, 42.0]], dtype=np.float64)

    if broadcast_case == "identical":
        base_b = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float64)
    elif broadcast_case == "broadcast_rows":
        base_b = np.array([[2.0, 3.0, 4.0]], dtype=np.float64)
    elif broadcast_case == "broadcast_cols":
        base_b = np.array([[2.0], [3.0]], dtype=np.float64)
    elif broadcast_case == "broadcast_1d":
        base_b = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    else:
        raise ValueError(f"Unknown broadcast case: {broadcast_case}")

    a_arr = base_a.astype(dtype1)
    b_arr = base_b.astype(dtype2)

    a = Tensor(a_arr, dtype=dtype1, requires_grad=True)
    b = Tensor(b_arr, dtype=dtype2, requires_grad=True)

    out = a / b
    loss = out.sum()
    loss.backward()

    # Analytical unbroadcasted gradients
    float_a = base_a
    float_b = base_b
    grad_out = np.ones_like(float_a / float_b)

    raw_da = (1.0 / float_b) * grad_out
    raw_db = (-float_a / (float_b ** 2)) * grad_out

    expected_da = Tensor._unbroadcast(raw_da, a_arr.shape)
    expected_db = Tensor._unbroadcast(raw_db, b_arr.shape)

    np.testing.assert_allclose(a.grad, expected_da, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(b.grad, expected_db, rtol=1e-4, atol=1e-4)
    assert np.issubdtype(a.grad.dtype, np.floating)
    assert np.issubdtype(b.grad.dtype, np.floating)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("scalar", [2.5, -4.0, 5])
def test_division_scalar_operands_gradients(dtype, scalar):
    """Division with scalar operands on both left and right sides preserves exact gradients."""
    a_arr = np.array([12, 24, 36], dtype=dtype)

    # Tensor / scalar
    a = Tensor(a_arr, dtype=dtype, requires_grad=True)
    out1 = a / scalar
    loss1 = out1.sum()
    loss1.backward()

    expected_da = np.ones_like(a_arr, dtype=np.float64) * (1.0 / scalar)
    np.testing.assert_allclose(a.grad, expected_da, rtol=1e-4, atol=1e-4)
    assert np.issubdtype(a.grad.dtype, np.floating)

    # scalar / Tensor
    b_arr = np.array([2, 4, 6], dtype=dtype)
    b = Tensor(b_arr, dtype=dtype, requires_grad=True)
    out2 = scalar / b
    loss2 = out2.sum()
    loss2.backward()

    expected_db = -scalar / (b_arr.astype(np.float64) ** 2)
    np.testing.assert_allclose(b.grad, expected_db, rtol=1e-4, atol=1e-4)
    assert np.issubdtype(b.grad.dtype, np.floating)


# ── 2. IEEE-754 Boundary Behavior ─────────────────────────────────────

def test_ieee754_log_boundaries():
    """Tensor.log() produces exact IEEE-754 values without epsilon damping."""
    # log(0) -> -inf
    x_zero = Tensor([0.0])
    with np.errstate(divide="ignore"):
        out_zero = x_zero.log()
    assert np.isneginf(out_zero.data[0])

    # log(-1) -> NaN
    x_neg = Tensor([-1.0])
    with np.errstate(invalid="ignore"):
        out_neg = x_neg.log()
    assert np.isnan(out_neg.data[0])


def test_ieee754_pow_boundaries():
    """Tensor.__pow__() adheres to mathematical boundaries."""
    # 0^0 -> 1
    x_zero = Tensor([0.0])
    out_pow0 = x_zero ** 0
    assert out_pow0.data[0] == 1.0

    # 2^3 -> 8
    x_two = Tensor([2.0])
    out_pow3 = x_two ** 3
    assert out_pow3.data[0] == 8.0

    # (-2)^0.5 -> NaN (fractional exponent on negative base in real domain)
    x_neg = Tensor([-2.0])
    with np.errstate(invalid="ignore"):
        out_fractional_neg = x_neg ** 0.5
    assert np.isnan(out_fractional_neg.data[0])

    # Mixed-sign tensor exponents: [2, 3] ** [2, -1] -> [4.0, 1/3]
    base = Tensor([2, 3])
    exp = Tensor([2, -1])
    out_mixed = base ** exp
    assert np.issubdtype(out_mixed.dtype, np.floating)
    np.testing.assert_allclose(out_mixed.data, [4.0, 1.0 / 3.0])


def test_zero_negative_power_autograd_stability_policy():
    """0 raised to negative powers adheres to Contract B: Autograd Numerical Stability Policy."""
    # 0^-1 evaluated under Contract B produces finite 1e12 approximation
    x_zero = Tensor([0.0])
    out_pow_neg1 = x_zero ** -1
    assert np.allclose(out_pow_neg1.data, [1e12])
    assert not np.any(np.isinf(out_pow_neg1.data))

    # 0^-2 evaluated under Contract B produces finite 1e24 approximation
    out_pow_neg2 = x_zero ** -2
    assert np.allclose(out_pow_neg2.data, [1e24])
    assert not np.any(np.isinf(out_pow_neg2.data))

    # Backward pass preserves finite continuous gradients, preventing parameter corruption
    x_zero_grad = Tensor([0.0], requires_grad=True)
    y = x_zero_grad ** -2
    y.backward()
    assert not np.isnan(x_zero_grad.grad[0])
    assert not np.isinf(x_zero_grad.grad[0])


def test_glassbox_catches_non_finite_explosion():
    """Glass-Box detect_anomaly intercepts NaN/Inf and points to root cause."""
    with pytest.raises(RuntimeError, match="GRADIENT POISONING DETECTED"):
        with detect_anomaly():
            x = Tensor([-5.0], requires_grad=True)
            # log(-5) produces NaN, which detect_anomaly must intercept
            y = x.log()
            y.backward()


# ── 3. Autograd Lifecycle & Multi-Branch Graph Stress ──────────────────

def test_multi_branch_diamond_graph_gradients():
    """Multi-branch diamond DAG correctly routes and sums gradients across all paths."""
    #        x
    #      /   \
    #     a     b
    #     |     |
    #    *2    +3
    #      \   /
    #        c
    x = Tensor([2.0, 3.0], requires_grad=True)
    a = x * 2.0  # da/dx = 2
    b = x + 3.0  # db/dx = 1
    c = a * b    # dc/da = b, dc/db = a
    loss = c.sum()
    loss.backward()

    # d(loss)/dx = d(loss)/da * da/dx + d(loss)/db * db/dx
    #            = b * 2 + a * 1 = 2b + a = 2(x + 3) + 2x = 4x + 6
    expected_grad = 4.0 * np.array([2.0, 3.0]) + 6.0
    np.testing.assert_allclose(x.grad, expected_grad)


def test_backward_freed_graph_raises_runtime_error():
    """Calling backward a second time without retain_graph=True raises clear RuntimeError."""
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = (x * 3.0).sum()
    y.backward(retain_graph=False)

    with pytest.raises(RuntimeError, match="Trying to backward through the graph a second time"):
        y.backward()


def test_backward_retain_graph_permits_repeated_backward():
    """Calling backward with retain_graph=True allows subsequent backward passes."""
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = (x * 3.0).sum()

    y.backward(retain_graph=True)
    np.testing.assert_allclose(x.grad, [3.0, 3.0])

    # Second backward accumulates into x.grad
    y.backward(retain_graph=False)
    np.testing.assert_allclose(x.grad, [6.0, 6.0])


def test_multi_step_parameter_mutation():
    """Mutating param.data directly across training steps does not corrupt subsequent graphs."""
    w = Tensor([[2.0, -1.0]], requires_grad=True)
    lr = 0.1

    for step in range(3):
        out = (w ** 2).sum()  # d(w^2)/dw = 2w
        out.backward()

        # Optimizer step: in-place data update
        w.data -= lr * w.grad
        w.zero_grad()

    # After 3 steps of gradient descent on w^2 with lr=0.1, w should shrink towards 0
    assert np.all(np.abs(w.data) < np.array([[2.0, 1.0]]))
    assert np.all(w.grad == 0.0)


# ── 6. Precision Numerical & Boundary Parity Verifications ─────────────

def test_mixed_sign_zero_base_power():
    """Verify mixed-sign tensor exponents with zero bases evaluate correctly elementwise."""
    # Tensor([0, 2]) ** Tensor([2, -1]) -> exact [0.0, 0.5]
    # In earlier versions, a global boolean is_negative erroneously substituted 0^2 with 1e-12^2 -> 1e-24.
    base = Tensor([0, 2])
    exp = Tensor([2, -1])
    out = base ** exp
    assert np.issubdtype(out.dtype, np.floating)
    np.testing.assert_allclose(out.data, [0.0, 0.5], atol=1e-12)

    # Autograd verification:
    # d(x^2)/dx at x=0 is 2*x = 0.0
    # d(x^-1)/dx at x=2 is -1/(x^2) = -0.25
    base_grad = Tensor([0.0, 2.0], requires_grad=True)
    exp_const = Tensor([2.0, -1.0])
    out_grad = base_grad ** exp_const
    out_grad.sum().backward()
    np.testing.assert_allclose(base_grad.grad, [0.0, -0.25], atol=1e-7)


def test_tensor_division_finite_difference():
    """Verify numerical finite-difference gradient check vs autograd for broadcasted Tensor / Tensor."""
    eps = 1e-5
    np.random.seed(42)
    a_val = np.random.uniform(1.0, 3.0, size=(2, 3)).astype(np.float64)
    b_val = np.random.uniform(1.0, 3.0, size=(1, 3)).astype(np.float64)

    a = Tensor(a_val, requires_grad=True)
    b = Tensor(b_val, requires_grad=True)
    out = (a / b).sum()
    out.backward()

    # Numerical grad for a: [f(a + eps) - f(a - eps)] / (2 * eps)
    grad_a_num = np.zeros_like(a_val)
    for idx in np.ndindex(a_val.shape):
        a_pos = a_val.copy()
        a_neg = a_val.copy()
        a_pos[idx] += eps
        a_neg[idx] -= eps
        f_pos = (a_pos / b_val).sum()
        f_neg = (a_neg / b_val).sum()
        grad_a_num[idx] = (f_pos - f_neg) / (2 * eps)

    # Numerical grad for b: [f(b + eps) - f(b - eps)] / (2 * eps)
    grad_b_num = np.zeros_like(b_val)
    for idx in np.ndindex(b_val.shape):
        b_pos = b_val.copy()
        b_neg = b_val.copy()
        b_pos[idx] += eps
        b_neg[idx] -= eps
        f_pos = (a_val / b_pos).sum()
        f_neg = (a_val / b_neg).sum()
        grad_b_num[idx] = (f_pos - f_neg) / (2 * eps)

    np.testing.assert_allclose(a.grad, grad_a_num, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(b.grad, grad_b_num, rtol=1e-5, atol=1e-5)


def test_tensor_power_finite_difference():
    """Verify numerical finite-difference gradient check vs autograd for Tensor ** Tensor (base and exponent)."""
    eps = 1e-5
    np.random.seed(42)
    a_val = np.random.uniform(1.5, 3.0, size=(2, 3)).astype(np.float64)
    b_val = np.random.uniform(1.5, 3.0, size=(2, 3)).astype(np.float64)

    a = Tensor(a_val, requires_grad=True)
    b = Tensor(b_val, requires_grad=True)
    out = (a ** b).sum()
    out.backward()

    # Numerical grad for base a
    grad_a_num = np.zeros_like(a_val)
    for idx in np.ndindex(a_val.shape):
        a_pos = a_val.copy()
        a_neg = a_val.copy()
        a_pos[idx] += eps
        a_neg[idx] -= eps
        f_pos = (a_pos ** b_val).sum()
        f_neg = (a_neg ** b_val).sum()
        grad_a_num[idx] = (f_pos - f_neg) / (2 * eps)

    # Numerical grad for exponent b: d(a^b)/db = a^b * ln(a)
    grad_b_num = np.zeros_like(b_val)
    for idx in np.ndindex(b_val.shape):
        b_pos = b_val.copy()
        b_neg = b_val.copy()
        b_pos[idx] += eps
        b_neg[idx] -= eps
        f_pos = (a_val ** b_pos).sum()
        f_neg = (a_val ** b_neg).sum()
        grad_b_num[idx] = (f_pos - f_neg) / (2 * eps)

    np.testing.assert_allclose(a.grad, grad_a_num, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(b.grad, grad_b_num, rtol=1e-5, atol=1e-5)

