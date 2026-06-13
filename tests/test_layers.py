from __future__ import annotations
"""
test_layers.py — Layer forward/backward parity tests vs PyTorch.

Verifies that miniGrad neural network layers produce numerically
identical outputs and gradients compared to PyTorch equivalents.

Run: pytest tests/test_layers.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

import pytest
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

from minigrad import Tensor
from minigrad.nn import Linear, Conv2D, ReLU, Sigmoid, Tanh, Flatten, BatchNorm1D, BatchNorm2D, Sequential
from minigrad.nn.loss import MSELoss, CrossEntropyLoss

np.random.seed(42)


@requires_torch
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


@requires_torch
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


@requires_torch
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


@requires_torch
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


@requires_torch
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


@requires_torch
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


@requires_torch
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


@requires_torch
def test_conv2d_layer_parity():
    """Test Conv2D forward and backward against torch.nn.Conv2d."""
    x_data = np.random.randn(2, 3, 5, 5)

    mg_conv = Conv2D(3, 4, kernel_size=3, stride=1, padding=1)
    x_mg = Tensor(x_data, requires_grad=True)
    out_mg = mg_conv(x_mg)
    loss_mg = out_mg.sum()
    loss_mg.backward()

    pt_conv = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=True).double()
    pt_conv.weight.data = torch.tensor(mg_conv.weight.data, dtype=torch.float64)
    pt_conv.bias.data = torch.tensor(mg_conv.bias.data, dtype=torch.float64)
    x_pt = torch.tensor(x_data, requires_grad=True, dtype=torch.float64)
    out_pt = pt_conv(x_pt)
    loss_pt = out_pt.sum()
    loss_pt.backward()

    np.testing.assert_allclose(out_mg.data, out_pt.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(mg_conv.weight.grad, pt_conv.weight.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(mg_conv.bias.grad, pt_conv.bias.grad.numpy(), atol=1e-5)


def test_conv2d_sequential_gradients_with_non_grad_input():
    """Conv params must learn even when input batches do not require gradients."""
    model = Sequential([
        Conv2D(1, 2, kernel_size=3, padding=1),
        ReLU(),
        Flatten(),
        Linear(2 * 4 * 4, 3),
    ])
    x = Tensor(np.random.randn(5, 1, 4, 4))
    y = np.array([0, 1, 2, 1, 0])

    loss = CrossEntropyLoss()(model(x), y)
    for p in model.parameters():
        p.zero_grad()
    loss.backward()

    conv = model.layers[0]
    assert np.abs(conv.weight.grad).sum() > 0.0
    assert np.abs(conv.bias.grad).sum() > 0.0


@requires_torch
def test_batchnorm1d_train_parity():
    x_data = np.random.randn(6, 4)

    mg_bn = BatchNorm1D(4)
    mg_bn.gamma.data = np.random.randn(4)
    mg_bn.beta.data = np.random.randn(4)
    x_mg = Tensor(x_data, requires_grad=True)
    out_mg = mg_bn(x_mg)
    loss_mg = out_mg.sum()
    loss_mg.backward()

    pt_bn = nn.BatchNorm1d(4, eps=mg_bn.eps, momentum=mg_bn.momentum).double()
    pt_bn.weight.data = torch.tensor(mg_bn.gamma.data, dtype=torch.float64)
    pt_bn.bias.data = torch.tensor(mg_bn.beta.data, dtype=torch.float64)
    x_pt = torch.tensor(x_data, requires_grad=True, dtype=torch.float64)
    out_pt = pt_bn(x_pt)
    loss_pt = out_pt.sum()
    loss_pt.backward()

    np.testing.assert_allclose(out_mg.data, out_pt.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(mg_bn.gamma.grad, pt_bn.weight.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(mg_bn.beta.grad, pt_bn.bias.grad.numpy(), atol=1e-5)


@requires_torch
def test_batchnorm2d_train_parity():
    x_data = np.random.randn(3, 4, 5, 5)

    mg_bn = BatchNorm2D(4)
    mg_bn.gamma.data = np.random.randn(4)
    mg_bn.beta.data = np.random.randn(4)
    x_mg = Tensor(x_data, requires_grad=True)
    out_mg = mg_bn(x_mg)
    loss_mg = out_mg.sum()
    loss_mg.backward()

    pt_bn = nn.BatchNorm2d(4, eps=mg_bn.eps, momentum=mg_bn.momentum).double()
    pt_bn.weight.data = torch.tensor(mg_bn.gamma.data, dtype=torch.float64)
    pt_bn.bias.data = torch.tensor(mg_bn.beta.data, dtype=torch.float64)
    x_pt = torch.tensor(x_data, requires_grad=True, dtype=torch.float64)
    out_pt = pt_bn(x_pt)
    loss_pt = out_pt.sum()
    loss_pt.backward()

    np.testing.assert_allclose(out_mg.data, out_pt.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(x_mg.grad, x_pt.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(mg_bn.gamma.grad, pt_bn.weight.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(mg_bn.beta.grad, pt_bn.bias.grad.numpy(), atol=1e-5)


def test_batchnorm_eval_keeps_affine_gradients():
    x_data = np.random.randn(6, 4)
    mg_bn = BatchNorm1D(4)
    mg_bn.running_mean = np.random.randn(4)
    mg_bn.running_var = np.random.rand(4) + 0.5
    mg_bn.eval()

    x_mg = Tensor(x_data, requires_grad=True)
    out_mg = mg_bn(x_mg)
    out_mg.sum().backward()

    assert np.abs(x_mg.grad).sum() > 0.0
    assert np.abs(mg_bn.gamma.grad).sum() > 0.0
    assert np.abs(mg_bn.beta.grad).sum() > 0.0


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
