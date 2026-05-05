"""
test_grad_check.py — Numerical gradient verification.

Uses finite differences to independently verify that every operation's
backward pass computes correct gradients. This is how PyTorch validates
its own autograd engine.

Run: pytest tests/test_grad_check.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from minigrad import Tensor
from minigrad.utils import grad_check

np.random.seed(42)


def test_add_grad():
    """Gradient check for addition."""
    def f():
        return (x + y).sum()
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    y = Tensor(np.random.randn(3, 4), requires_grad=True)
    assert grad_check(f, [x, y], verbose=False)


def test_add_broadcast_grad():
    """Gradient check for broadcast addition."""
    def f():
        return (x + y).sum()
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    y = Tensor(np.random.randn(4), requires_grad=True)
    assert grad_check(f, [x, y], verbose=False)


def test_mul_grad():
    """Gradient check for element-wise multiplication."""
    def f():
        return (x * y).sum()
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    y = Tensor(np.random.randn(3, 4), requires_grad=True)
    assert grad_check(f, [x, y], verbose=False)


def test_matmul_grad():
    """Gradient check for matrix multiplication."""
    def f():
        return (x @ y).sum()
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    y = Tensor(np.random.randn(4, 5), requires_grad=True)
    assert grad_check(f, [x, y], verbose=False)


def test_pow_grad():
    """Gradient check for power."""
    def f():
        return (x ** 3).sum()
    x = Tensor(np.random.randn(3, 4) + 2.0, requires_grad=True)
    assert grad_check(f, [x], verbose=False)


def test_relu_grad():
    """Gradient check for ReLU."""
    def f():
        return x.relu().sum()
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    assert grad_check(f, [x], verbose=False)


def test_sigmoid_grad():
    """Gradient check for sigmoid."""
    def f():
        return x.sigmoid().sum()
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    assert grad_check(f, [x], verbose=False)


def test_tanh_grad():
    """Gradient check for tanh."""
    def f():
        return x.tanh().sum()
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    assert grad_check(f, [x], verbose=False)


def test_exp_grad():
    """Gradient check for exponential."""
    def f():
        return x.exp().sum()
    x = Tensor(np.random.randn(3, 4) * 0.5, requires_grad=True)
    assert grad_check(f, [x], verbose=False)


def test_log_grad():
    """Gradient check for natural log."""
    def f():
        return x.log().sum()
    x = Tensor(np.random.rand(3, 4) + 0.1, requires_grad=True)
    assert grad_check(f, [x], verbose=False)


def test_sum_grad():
    """Gradient check for sum reduction."""
    def f():
        return x.sum()
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    assert grad_check(f, [x], verbose=False)


def test_sum_axis_grad():
    """Gradient check for sum along axis."""
    def f():
        return x.sum(axis=0).sum()
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    assert grad_check(f, [x], verbose=False)


def test_mean_grad():
    """Gradient check for mean reduction."""
    def f():
        return x.mean()
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    assert grad_check(f, [x], verbose=False)


def test_reshape_grad():
    """Gradient check for reshape."""
    def f():
        return x.reshape(2, 6).sum()
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    assert grad_check(f, [x], verbose=False)


def test_transpose_grad():
    """Gradient check for transpose."""
    def f():
        return x.transpose().sum()
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    assert grad_check(f, [x], verbose=False)


def test_neg_grad():
    """Gradient check for negation."""
    def f():
        return (-x).sum()
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    assert grad_check(f, [x], verbose=False)


def test_sub_grad():
    """Gradient check for subtraction."""
    def f():
        return (x - y).sum()
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    y = Tensor(np.random.randn(3, 4), requires_grad=True)
    assert grad_check(f, [x, y], verbose=False)


def test_chain_rule_complex():
    """Gradient check for a complex chain of operations."""
    def f():
        h = (x @ w + b).relu()
        return ((h - y) ** 2).mean()

    x = Tensor(np.random.randn(4, 3), requires_grad=True)
    w = Tensor(np.random.randn(3, 5), requires_grad=True)
    b = Tensor(np.random.randn(5), requires_grad=True)
    y = np.random.randn(4, 5)

    assert grad_check(f, [x, w, b], verbose=False)


if __name__ == "__main__":
    print("Running numerical gradient checks...")
    tests = [
        test_add_grad, test_add_broadcast_grad, test_mul_grad, test_matmul_grad,
        test_pow_grad, test_relu_grad, test_sigmoid_grad, test_tanh_grad,
        test_exp_grad, test_log_grad, test_sum_grad, test_sum_axis_grad,
        test_mean_grad, test_reshape_grad, test_transpose_grad,
        test_neg_grad, test_sub_grad, test_chain_rule_complex,
    ]
    passed = 0
    for t in tests:
        try:
            result = t()
            if result is not False:
                print(f"  PASS: {t.__name__}")
                passed += 1
            else:
                print(f"  FAIL: {t.__name__}")
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} gradient checks passed")
