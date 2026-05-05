"""
test_layers.py — Layer forward/backward parity tests vs PyTorch.

Verifies that miniGrad neural network layers produce numerically
identical outputs and gradients compared to PyTorch equivalents.

Run: pytest tests/test_layers.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from minigrad import Tensor
from minigrad.nn import Linear, Conv2D, ReLU, Sigmoid, Tanh
from minigrad.nn.loss import MSELoss, CrossEntropyLoss

np.random.seed(42)


def test_linear_layer():
    """Test Linear layer forward and backward against torch.nn.Linear."""
    batch_size, in_features, out_features = 8, 20, 10
    x_data = np.random.randn(batch_size, in_features)

    # miniGrad
    mg_linear = Linear(in_features, out_features)
    x_mg = Tensor(x_data, requires_grad=True)
    out_mg = mg_linear(x_mg)
    loss_mg = out_mg.sum()
    loss_mg.backward()

    # PyTorch
    pt_linear = nn.Linear(in_features, out_features, bias=True)
    pt_linear.weight.data = torch.tensor(mg_linear.weight.data.T, dtype=torch.float64)
    pt_linear.bias.data = torch.tensor(mg_linear.bias.data, dtype=torch.float64)
    x_pt = torch.tensor(x_data, requires_grad=True, dtype=torch.float64)
    out_pt = pt_linear(x_pt)
    loss_pt = out_pt.sum()
    loss_pt.backward()

    # Forward check
    np.testing.assert_allclose(out_mg.data, out_pt.detach().numpy(), atol=1e-5)

    # Input gradient check
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-5)

    # Weight gradient check (PyTorch stores as (out, in), we store as (in, out))
    np.testing.assert_allclose(mg_linear.weight.grad, pt_linear.weight.grad.numpy().T, atol=1e-5)

    # Bias gradient check
    np.testing.assert_allclose(mg_linear.bias.grad, pt_linear.bias.grad.numpy(), atol=1e-5)


def test_relu_layer():
    """Test ReLU layer against torch.nn.ReLU."""
    x_data = np.random.randn(8, 10)

    x_mg = Tensor(x_data, requires_grad=True)
    relu_mg = ReLU()
    out_mg = relu_mg(x_mg)
    out_mg.backward()

    x_pt = torch.tensor(x_data, requires_grad=True, dtype=torch.float64)
    relu_pt = nn.ReLU()
    out_pt = relu_pt(x_pt)
    out_pt.backward(torch.ones_like(out_pt))

    np.testing.assert_allclose(out_mg.data, out_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)


def test_sigmoid_layer():
    """Test Sigmoid layer against torch.nn.Sigmoid."""
    x_data = np.random.randn(8, 10)

    x_mg = Tensor(x_data, requires_grad=True)
    sig_mg = Sigmoid()
    out_mg = sig_mg(x_mg)
    out_mg.backward()

    x_pt = torch.tensor(x_data, requires_grad=True, dtype=torch.float64)
    sig_pt = nn.Sigmoid()
    out_pt = sig_pt(x_pt)
    out_pt.backward(torch.ones_like(out_pt))

    np.testing.assert_allclose(out_mg.data, out_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)


def test_tanh_layer():
    """Test Tanh layer against torch.nn.Tanh."""
    x_data = np.random.randn(8, 10)

    x_mg = Tensor(x_data, requires_grad=True)
    tanh_mg = Tanh()
    out_mg = tanh_mg(x_mg)
    out_mg.backward()

    x_pt = torch.tensor(x_data, requires_grad=True, dtype=torch.float64)
    tanh_pt = nn.Tanh()
    out_pt = tanh_pt(x_pt)
    out_pt.backward(torch.ones_like(out_pt))

    np.testing.assert_allclose(out_mg.data, out_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-6)


def test_mse_loss():
    """Test MSELoss against torch.nn.MSELoss."""
    pred_data = np.random.randn(8, 5)
    target_data = np.random.randn(8, 5)

    pred_mg = Tensor(pred_data, requires_grad=True)
    criterion_mg = MSELoss()
    loss_mg = criterion_mg(pred_mg, target_data)
    loss_mg.backward()

    pred_pt = torch.tensor(pred_data, requires_grad=True, dtype=torch.float64)
    criterion_pt = nn.MSELoss()
    loss_pt = criterion_pt(pred_pt, torch.tensor(target_data, dtype=torch.float64))
    loss_pt.backward()

    np.testing.assert_allclose(loss_mg.data, loss_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(pred_mg.grad, pred_pt.grad.numpy(), atol=1e-6)


def test_cross_entropy_loss():
    """Test CrossEntropyLoss against torch.nn.CrossEntropyLoss."""
    batch_size, num_classes = 16, 10
    logits_data = np.random.randn(batch_size, num_classes)
    targets = np.random.randint(0, num_classes, size=batch_size)

    logits_mg = Tensor(logits_data, requires_grad=True)
    criterion_mg = CrossEntropyLoss()
    loss_mg = criterion_mg(logits_mg, targets)
    loss_mg.backward()

    logits_pt = torch.tensor(logits_data, requires_grad=True, dtype=torch.float64)
    criterion_pt = nn.CrossEntropyLoss()
    loss_pt = criterion_pt(logits_pt, torch.tensor(targets, dtype=torch.int64))
    loss_pt.backward()

    np.testing.assert_allclose(loss_mg.data, loss_pt.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(logits_mg.grad, logits_pt.grad.numpy(), atol=1e-6)


def test_sequential():
    """Test Sequential container against equivalent PyTorch Sequential."""
    x_data = np.random.randn(4, 10)

    # miniGrad model
    from minigrad.nn import Sequential, Linear, ReLU
    mg_model = Sequential([
        Linear(10, 20),
        ReLU(),
        Linear(20, 5),
    ])
    x_mg = Tensor(x_data, requires_grad=True)
    out_mg = mg_model(x_mg)
    loss_mg = out_mg.sum()
    loss_mg.backward()

    # PyTorch model with same weights
    pt_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    # Copy weights
    pt_model[0].weight.data = torch.tensor(mg_model.layers[0].weight.data.T, dtype=torch.float64)
    pt_model[0].bias.data = torch.tensor(mg_model.layers[0].bias.data, dtype=torch.float64)
    pt_model[2].weight.data = torch.tensor(mg_model.layers[2].weight.data.T, dtype=torch.float64)
    pt_model[2].bias.data = torch.tensor(mg_model.layers[2].bias.data, dtype=torch.float64)

    x_pt = torch.tensor(x_data, requires_grad=True, dtype=torch.float64)
    out_pt = pt_model(x_pt)
    loss_pt = out_pt.sum()
    loss_pt.backward()

    np.testing.assert_allclose(out_mg.data, out_pt.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-5)


if __name__ == "__main__":
    if not HAS_TORCH:
        print("PyTorch not installed. Skipping parity tests.")
        sys.exit(0)

    print("Running layer parity tests...")
    tests = [
        test_linear_layer, test_relu_layer, test_sigmoid_layer,
        test_tanh_layer, test_mse_loss, test_cross_entropy_loss,
        test_sequential,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
