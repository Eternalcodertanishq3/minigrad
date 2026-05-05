"""
test_ops.py — Parity tests: miniGrad ops vs PyTorch.

Every operation is tested against its PyTorch equivalent to ensure
numerical correctness to 1e-6 precision.

Run: pytest tests/test_ops.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

# Skip all tests if PyTorch is not installed
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from minigrad import Tensor


# ---------------------------------------------------------------------------
# Helper: compare Tensor and torch.Tensor gradients
# ---------------------------------------------------------------------------

def tensor_to_torch(t: Tensor) -> torch.Tensor:
    """Convert miniGrad Tensor to PyTorch tensor with gradients."""
    return torch.tensor(t.data, dtype=torch.float64, requires_grad=t.requires_grad)


def assert_parity(mg_tensor: Tensor, torch_tensors: list, torch_result: torch.Tensor,
                  mg_fn, torch_fn, atol: float = 1e-6) -> None:
    """
    Compare miniGrad forward and backward against PyTorch.

    Args:
        mg_tensor:   miniGrad result tensor
        torch_tensors: List of torch tensors that were inputs
        torch_result: PyTorch result tensor
        mg_fn:       Function to rebuild miniGrad computation for grad_check
        torch_fn:    Function to rebuild PyTorch computation
    """
    # Forward parity
    np.testing.assert_allclose(mg_tensor.data, torch_result.detach().numpy(), atol=atol,
                               err_msg="Forward pass mismatch")

    # Backward parity
    mg_tensor.backward()
    torch_result.backward(gradient=torch.ones_like(torch_result))

    for mg_inp, torch_inp in zip(mg_tensor._prev, torch_tensors):
        if mg_inp.requires_grad and torch_inp.grad is not None:
            np.testing.assert_allclose(mg_inp.grad, torch_inp.grad.numpy(), atol=atol,
                                       err_msg=f"Gradient mismatch for {mg_inp._op}")


# ---------------------------------------------------------------------------
# Addition
# ---------------------------------------------------------------------------

def test_add():
    a = np.random.randn(4, 3)
    b = np.random.randn(4, 3)

    x_mg = Tensor(a, requires_grad=True)
    y_mg = Tensor(b, requires_grad=True)
    z_mg = x_mg + y_mg
    z_mg.backward()

    x_pt = torch.tensor(a, requires_grad=True, dtype=torch.float64)
    y_pt = torch.tensor(b, requires_grad=True, dtype=torch.float64)
    z_pt = x_pt + y_pt
    z_pt.backward(torch.ones_like(z_pt))

    np.testing.assert_allclose(z_mg.data, z_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)
    np.testing.assert_allclose(y_mg.grad, y_pt.grad.numpy(), atol=1e-6)


def test_add_broadcast():
    """Test broadcasting: (4, 3) + (3,)"""
    a = np.random.randn(4, 3)
    b = np.random.randn(3)

    x_mg = Tensor(a, requires_grad=True)
    y_mg = Tensor(b, requires_grad=True)
    z_mg = x_mg + y_mg
    z_mg.backward()

    x_pt = torch.tensor(a, requires_grad=True, dtype=torch.float64)
    y_pt = torch.tensor(b, requires_grad=True, dtype=torch.float64)
    z_pt = x_pt + y_pt
    z_pt.backward(torch.ones_like(z_pt))

    np.testing.assert_allclose(z_mg.data, z_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)
    np.testing.assert_allclose(y_mg.grad, y_pt.grad.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------

def test_mul():
    a = np.random.randn(4, 3)
    b = np.random.randn(4, 3)

    x_mg = Tensor(a, requires_grad=True)
    y_mg = Tensor(b, requires_grad=True)
    z_mg = x_mg * y_mg
    z_mg.backward()

    x_pt = torch.tensor(a, requires_grad=True, dtype=torch.float64)
    y_pt = torch.tensor(b, requires_grad=True, dtype=torch.float64)
    z_pt = x_pt * y_pt
    z_pt.backward(torch.ones_like(z_pt))

    np.testing.assert_allclose(z_mg.data, z_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)
    np.testing.assert_allclose(y_mg.grad, y_pt.grad.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# Matrix Multiplication
# ---------------------------------------------------------------------------

def test_matmul():
    a = np.random.randn(3, 4)
    b = np.random.randn(4, 5)

    x_mg = Tensor(a, requires_grad=True)
    y_mg = Tensor(b, requires_grad=True)
    z_mg = x_mg @ y_mg
    z_mg.backward()

    x_pt = torch.tensor(a, requires_grad=True, dtype=torch.float64)
    y_pt = torch.tensor(b, requires_grad=True, dtype=torch.float64)
    z_pt = x_pt @ y_pt
    z_pt.backward(torch.ones_like(z_pt))

    np.testing.assert_allclose(z_mg.data, z_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)
    np.testing.assert_allclose(y_mg.grad, y_pt.grad.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------

def test_pow():
    a = np.random.randn(4, 3) + 1.0  # avoid negative base with non-integer exponent issues

    x_mg = Tensor(a, requires_grad=True)
    z_mg = x_mg ** 3
    z_mg.backward()

    x_pt = torch.tensor(a, requires_grad=True, dtype=torch.float64)
    z_pt = x_pt ** 3
    z_pt.backward(torch.ones_like(z_pt))

    np.testing.assert_allclose(z_mg.data, z_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------------

def test_relu():
    a = np.random.randn(4, 3)

    x_mg = Tensor(a, requires_grad=True)
    z_mg = x_mg.relu()
    z_mg.backward()

    x_pt = torch.tensor(a, requires_grad=True, dtype=torch.float64)
    z_pt = torch.relu(x_pt)
    z_pt.backward(torch.ones_like(z_pt))

    np.testing.assert_allclose(z_mg.data, z_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------

def test_sigmoid():
    a = np.random.randn(4, 3)

    x_mg = Tensor(a, requires_grad=True)
    z_mg = x_mg.sigmoid()
    z_mg.backward()

    x_pt = torch.tensor(a, requires_grad=True, dtype=torch.float64)
    z_pt = torch.sigmoid(x_pt)
    z_pt.backward(torch.ones_like(z_pt))

    np.testing.assert_allclose(z_mg.data, z_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# Tanh
# ---------------------------------------------------------------------------

def test_tanh():
    a = np.random.randn(4, 3)

    x_mg = Tensor(a, requires_grad=True)
    z_mg = x_mg.tanh()
    z_mg.backward()

    x_pt = torch.tensor(a, requires_grad=True, dtype=torch.float64)
    z_pt = torch.tanh(x_pt)
    z_pt.backward(torch.ones_like(z_pt))

    np.testing.assert_allclose(z_mg.data, z_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# Sum
# ---------------------------------------------------------------------------

def test_sum():
    a = np.random.randn(4, 3)

    x_mg = Tensor(a, requires_grad=True)
    z_mg = x_mg.sum()
    z_mg.backward()

    x_pt = torch.tensor(a, requires_grad=True, dtype=torch.float64)
    z_pt = x_pt.sum()
    z_pt.backward()

    np.testing.assert_allclose(z_mg.data, z_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)


def test_sum_axis():
    a = np.random.randn(4, 3)

    x_mg = Tensor(a, requires_grad=True)
    z_mg = x_mg.sum(axis=0)
    z_mg.backward()

    x_pt = torch.tensor(a, requires_grad=True, dtype=torch.float64)
    z_pt = x_pt.sum(dim=0)
    z_pt.backward(torch.ones_like(z_pt))

    np.testing.assert_allclose(z_mg.data, z_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# Mean
# ---------------------------------------------------------------------------

def test_mean():
    a = np.random.randn(4, 3)

    x_mg = Tensor(a, requires_grad=True)
    z_mg = x_mg.mean()
    z_mg.backward()

    x_pt = torch.tensor(a, requires_grad=True, dtype=torch.float64)
    z_pt = x_pt.mean()
    z_pt.backward()

    np.testing.assert_allclose(z_mg.data, z_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# Reshape / Transpose
# ---------------------------------------------------------------------------

def test_reshape():
    a = np.random.randn(4, 3)

    x_mg = Tensor(a, requires_grad=True)
    z_mg = x_mg.reshape(2, 6)
    z_mg.backward()

    x_pt = torch.tensor(a, requires_grad=True, dtype=torch.float64)
    z_pt = x_pt.reshape(2, 6)
    z_pt.backward(torch.ones_like(z_pt))

    np.testing.assert_allclose(z_mg.data, z_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)


def test_transpose():
    a = np.random.randn(4, 3)

    x_mg = Tensor(a, requires_grad=True)
    z_mg = x_mg.transpose()
    z_mg.backward()

    x_pt = torch.tensor(a, requires_grad=True, dtype=torch.float64)
    z_pt = x_pt.t()
    z_pt.backward(torch.ones_like(z_pt))

    np.testing.assert_allclose(z_mg.data, z_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# Exponential and Log
# ---------------------------------------------------------------------------

def test_exp():
    a = np.random.randn(4, 3) * 0.5  # keep values moderate

    x_mg = Tensor(a, requires_grad=True)
    z_mg = x_mg.exp()
    z_mg.backward()

    x_pt = torch.tensor(a, requires_grad=True, dtype=torch.float64)
    z_pt = torch.exp(x_pt)
    z_pt.backward(torch.ones_like(z_pt))

    np.testing.assert_allclose(z_mg.data, z_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)


def test_log():
    a = np.random.rand(4, 3) + 0.1  # positive values

    x_mg = Tensor(a, requires_grad=True)
    z_mg = x_mg.log()
    z_mg.backward()

    x_pt = torch.tensor(a, requires_grad=True, dtype=torch.float64)
    z_pt = torch.log(x_pt)
    z_pt.backward(torch.ones_like(z_pt))

    np.testing.assert_allclose(z_mg.data, z_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

def test_softmax():
    from minigrad.ops import softmax
    a = np.random.randn(4, 3)

    x_mg = Tensor(a, requires_grad=True)
    z_mg = softmax(x_mg, axis=-1)
    z_mg.backward()

    x_pt = torch.tensor(a, requires_grad=True, dtype=torch.float64)
    z_pt = F.softmax(x_pt, dim=-1)
    z_pt.backward(torch.ones_like(z_pt))

    np.testing.assert_allclose(z_mg.data, z_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------

def test_neg():
    a = np.random.randn(4, 3)

    x_mg = Tensor(a, requires_grad=True)
    z_mg = -x_mg
    z_mg.backward()

    x_pt = torch.tensor(a, requires_grad=True, dtype=torch.float64)
    z_pt = -x_pt
    z_pt.backward(torch.ones_like(z_pt))

    np.testing.assert_allclose(z_mg.data, z_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# Subtraction
# ---------------------------------------------------------------------------

def test_sub():
    a = np.random.randn(4, 3)
    b = np.random.randn(4, 3)

    x_mg = Tensor(a, requires_grad=True)
    y_mg = Tensor(b, requires_grad=True)
    z_mg = x_mg - y_mg
    z_mg.backward()

    x_pt = torch.tensor(a, requires_grad=True, dtype=torch.float64)
    y_pt = torch.tensor(b, requires_grad=True, dtype=torch.float64)
    z_pt = x_pt - y_pt
    z_pt.backward(torch.ones_like(z_pt))

    np.testing.assert_allclose(z_mg.data, z_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)
    np.testing.assert_allclose(y_mg.grad, y_pt.grad.numpy(), atol=1e-6)


if __name__ == "__main__":
    if not HAS_TORCH:
        print("PyTorch not installed. Skipping parity tests.")
        sys.exit(0)

    print("Running op parity tests...")
    test_functions = [
        test_add, test_add_broadcast, test_mul, test_matmul,
        test_pow, test_relu, test_sigmoid, test_tanh,
        test_sum, test_sum_axis, test_mean, test_reshape,
        test_transpose, test_exp, test_log, test_softmax,
        test_neg, test_sub,
    ]

    passed = 0
    failed = 0
    for fn in test_functions:
        try:
            fn()
            print(f"  PASS: {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {fn.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(test_functions)} tests")
