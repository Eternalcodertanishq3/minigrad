"""
tests/test_contracts.py — Formal mathematical, dtype, and numerical contract verification.

Verifies:
1. Exact Tensor.log() semantics (positive -> finite, 0 -> -inf, negative -> nan, grad -> 1/x).
2. Glass-Box anomaly detection catching non-finite values in log().
3. Exact Tensor.__pow__() base derivative (no +1e-9 hack).
4. Pure LogSumExp CrossEntropyLoss without clipping or loss capping on extreme error logits.
5. Dtype preservation (explicit dtypes, Tensor.dtype property, .to(dtype) casting).
6. NumPy/IEEE dtype promotion across binary operations and mixed scalar operands.
"""
from __future__ import annotations

import numpy as np
import pytest

from minigrad.contracts import (
    check_dynamic_parity,
    dynamic_tolerance,
)
from minigrad.glassbox import (
    GradientAnomalyError,
    detect_anomaly,
)
from minigrad.nn import CrossEntropyLoss
from minigrad.tensor import Tensor

# ── 1. Exact log() Semantics ──────────────────────────────────────────

def test_log_exact_positive():
    """x > 0 yields finite log value and exact 1/x gradient."""
    x_val = np.array([0.1, 0.5, 1.0, 2.0, 10.0], dtype=np.float64)
    x = Tensor(x_val, requires_grad=True)
    y = x.log()

    expected_y = np.log(x_val)
    assert np.allclose(y.data, expected_y, atol=1e-12, rtol=1e-12)

    y.sum().backward()
    expected_grad = 1.0 / x_val
    assert np.allclose(x.grad, expected_grad, atol=1e-12, rtol=1e-12)


def test_log_exact_zero_and_negative_values():
    """Exact math: log(0) == -inf, log(negative) == nan."""
    x = Tensor([0.0, -1.0, 2.0], requires_grad=False)
    y = x.log()
    assert np.isneginf(y.data[0])
    assert np.isnan(y.data[1])
    assert np.isfinite(y.data[2])


def test_log_glassbox_anomaly_detection_catches_nonpositive():
    """Glass-Box anomaly detection catches non-positive inputs in log forward."""
    x = Tensor([1.0, 0.0, 2.0], requires_grad=True)
    with detect_anomaly():
        y = x.log()
        with pytest.raises(GradientAnomalyError) as exc_info:
            y.backward()
        err = exc_info.value
        assert "Logarithm forward pass evaluated on non-positive input" in err.reason or "log" in err.op


# ── 2. Exact __pow__() Base Derivative ───────────────────────────────

def test_pow_exact_base_derivative():
    """Derivative w.r.t exponent is a^b * ln(a) without +1e-9 fudge factor."""
    a_val = 2.5
    b_val = 3.0
    a = Tensor([a_val], requires_grad=False)
    b = Tensor([b_val], requires_grad=True)

    y = a ** b
    y.backward()

    expected_db = (a_val ** b_val) * np.log(a_val)
    # Bit-for-bit exact parity with analytical derivative
    assert np.allclose(b.grad, [expected_db], atol=1e-14, rtol=1e-14)


# ── 3. CrossEntropy Pure LogSumExp & Extreme Logits ───────────────────

def test_cross_entropy_exact_logsumexp():
    """CrossEntropyLoss computes exact LogSumExp without clipping."""
    loss_fn = CrossEntropyLoss(reduction="mean")
    logits = Tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 1.0]], requires_grad=True)
    targets = np.array([0, 1])

    loss = loss_fn(logits, targets)

    # Reference manual logsumexp
    logits_data = logits.data
    max_l = np.max(logits_data, axis=1, keepdims=True)
    lse = max_l + np.log(np.sum(np.exp(logits_data - max_l), axis=1, keepdims=True))
    log_probs = logits_data - lse
    expected_loss = np.mean(-log_probs[np.arange(2), targets])

    assert np.allclose(loss.data, expected_loss, atol=1e-14, rtol=1e-14)

    loss.backward()
    expected_probs = np.exp(log_probs)
    expected_grad = expected_probs.copy()
    expected_grad[np.arange(2), targets] -= 1.0
    expected_grad /= 2.0
    assert np.allclose(logits.grad, expected_grad, atol=1e-14, rtol=1e-14)


def test_cross_entropy_extreme_error_logits_no_cap():
    """
    On extreme error logits (z = -1000 vs +1000 for wrong class),
    loss must not be capped at ~20.7 (-ln(1e-9)). It should scale linearly to ~2000.
    """
    loss_fn = CrossEntropyLoss(reduction="mean")
    # Correct class 0 has logit -1000; wrong class 1 has logit +1000
    logits = Tensor([[-1000.0, 1000.0]], requires_grad=True)
    targets = np.array([0])

    loss = loss_fn(logits, targets)
    # Loss should be 2000.0, NOT 20.7
    assert loss.data > 1900.0, f"Loss was capped at {loss.data}; expected ~2000.0"
    assert np.isclose(loss.data, 2000.0, atol=1e-3)


# ── 4. Dtype Preservation & Casting ──────────────────────────────────

def test_dtype_preservation_explicit():
    """Explicitly specified dtypes must be preserved in data and grad."""
    t32 = Tensor([1.0, 2.0], dtype=np.float32, requires_grad=True)
    assert t32.dtype == np.float32
    assert t32.data.dtype == np.float32
    assert t32.grad.dtype == np.float32

    t64 = Tensor([1.0, 2.0], dtype=np.float64, requires_grad=True)
    assert t64.dtype == np.float64
    assert t64.data.dtype == np.float64
    assert t64.grad.dtype == np.float64


def test_dtype_preservation_from_ndarray():
    """Creating a Tensor from an ndarray preserves the array's dtype."""
    arr32 = np.array([1.5, 2.5], dtype=np.float32)
    t = Tensor(arr32, requires_grad=True)
    assert t.dtype == np.float32
    assert t.data.dtype == np.float32


def test_dtype_default_is_float64():
    """Default list input preserves standard float64."""
    t = Tensor([1.0, 2.0])
    assert t.dtype == np.float64


def test_dtype_to_method():
    """Tensor.to(dtype) casts data and backpropagates gradients seamlessly."""
    x = Tensor([2.0, 3.0], dtype=np.float64, requires_grad=True)
    x32 = x.to(np.float32)
    assert x32.dtype == np.float32
    assert x32.data.dtype == np.float32

    # Idempotent cast
    x32_same = x32.to(np.float32)
    assert x32_same is x32

    y = (x32 * 2.0).sum()
    y.backward()
    assert x.grad.dtype == np.float64
    assert np.allclose(x.grad, [2.0, 2.0])


# ── 5. Dtype Promotion Semantics ─────────────────────────────────────

def test_dtype_promotion_binary_ops():
    """Binary operations promote types according to NumPy/IEEE rules."""
    a32 = Tensor([1.0, 2.0], dtype=np.float32, requires_grad=True)
    b32 = Tensor([3.0, 4.0], dtype=np.float32, requires_grad=True)
    c64 = Tensor([5.0, 6.0], dtype=np.float64, requires_grad=True)

    # float32 + float32 -> float32
    out_32 = a32 + b32
    assert out_32.dtype == np.float32

    # float32 + float64 -> float64
    out_64 = a32 + c64
    assert out_64.dtype == np.float64

    # float64 * float32 -> float64
    out_mul = c64 * a32
    assert out_mul.dtype == np.float64

    # Python float + float32 -> float32
    out_scalar = a32 + 2.5
    assert out_scalar.dtype == np.float32

    # Python int + float32 -> float32
    out_int = a32 * 3
    assert out_int.dtype == np.float32


def test_dtype_promotion_matmul():
    """Matmul promotes mixed float32 and float64 operands to float64."""
    m32 = Tensor(np.ones((2, 3), dtype=np.float32), requires_grad=True)
    m64 = Tensor(np.ones((3, 2), dtype=np.float64), requires_grad=True)
    res = m32 @ m64
    assert res.dtype == np.float64

    res.sum().backward()
    # Gradients are cast back to each operand's original dtype
    assert m32.grad.dtype == np.float32
    assert m64.grad.dtype == np.float64


def test_dtype_promotion_gradient_accumulation():
    """When a float32 tensor participates in float64 graph, its grad stays float32."""
    x = Tensor([2.0, 3.0], dtype=np.float32, requires_grad=True)
    y = Tensor([4.0, 5.0], dtype=np.float64, requires_grad=True)

    z = (x * y).sum()
    assert z.dtype == np.float64

    z.backward()
    assert x.grad.dtype == np.float32
    assert y.grad.dtype == np.float64
    assert np.allclose(x.grad, [4.0, 5.0])
    assert np.allclose(y.grad, [2.0, 3.0])


def test_dtype_promotion_division_and_rpow():
    """Verify truediv, rtruediv, and rpow preserve dtypes and gradients."""
    x = Tensor([2.0, 4.0], dtype=np.float32, requires_grad=True)

    # float32 / float -> float32
    div1 = x / 2.0
    assert div1.dtype == np.float32
    assert np.allclose(div1.data, [1.0, 2.0])

    # float / float32 -> float32
    div2 = 4.0 / x
    assert div2.dtype == np.float32
    assert np.allclose(div2.data, [2.0, 1.0])

    # float ** float32 -> float32 with autograd
    p = 3.0 ** x
    assert p.dtype == np.float32
    assert np.allclose(p.data, [9.0, 81.0])
    p.sum().backward()
    assert x.grad.dtype == np.float32
    expected_grad = (3.0 ** np.array([2.0, 4.0], dtype=np.float32)) * np.log(3.0)
    assert np.allclose(x.grad, expected_grad, rtol=1e-5, atol=1e-5)


# ── 6. Dynamic Parity Checker Contract ────────────────────────────────

def test_check_dynamic_parity_contract():
    """Verify dynamic tolerance computation and check_dynamic_parity."""
    ref = np.array([1.0, 10.0, 100.0], dtype=np.float64)
    # Expected tolerance: 1e-6 + 1e-5 * |ref|
    tol = dynamic_tolerance(ref, dtype=np.float64)
    assert np.isclose(tol[0], 1e-6 + 1e-5 * 1.0)
    assert np.isclose(tol[2], 1e-6 + 1e-5 * 100.0)

    # Within tolerance
    act = ref + tol * 0.5
    passed, max_diff, max_tol = check_dynamic_parity(act, ref, dtype=np.float64)
    assert passed
    assert max_diff <= max_tol

    # Outside tolerance
    act_bad = ref + tol * 2.0
    passed_bad, _, _ = check_dynamic_parity(act_bad, ref, dtype=np.float64)
    assert not passed_bad
