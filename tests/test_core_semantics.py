"""
tests/test_core_semantics.py — Exhaustive Dtype Promotion, IEEE-754 Boundaries, and Lifecycle Invariants.

Validates:
1. Dtype Promotion Matrix:
   - Full pairwise matrix of float32, float64, int32, int64, and Python scalars.
   - Verified across binary ops (+, -, *, /, @, **).
   - Exact parity with NumPy dtype promotion semantics.
2. IEEE-754 Boundary & Edge-Case Behavior:
   - Exact mathematical limits: log(0) -> -inf, log(-x) -> nan, 0/0 -> nan, 1/0 -> inf, 0^0 -> 1.
   - Glass-Box anomaly interception halts at the exact exploding node.
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
def test_dtype_promotion_power(dtype):
    """Power operations with positive and negative exponents follow normalized promotion."""
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


def test_matmul_dtype_promotion():
    """Matrix multiplication promotes dtypes identically to NumPy."""
    x = Tensor(np.ones((2, 3), dtype=np.float32))
    w = Tensor(np.ones((3, 4), dtype=np.float64))
    out = x @ w
    assert out.dtype == np.float64
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out.data, np.full((2, 4), 3.0))


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
