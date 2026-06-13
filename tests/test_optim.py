from __future__ import annotations
"""
test_optim.py — Optimizer step parity tests vs PyTorch.

Verifies that miniGrad optimizers update parameters identically
to PyTorch optimizers given the same gradients and hyperparameters.

Run: pytest tests/test_optim.py -v
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
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

from minigrad import Tensor
from minigrad.optim import SGD, Adam

np.random.seed(42)


def test_sgd_parity():
    """Compare miniGrad SGD with PyTorch SGD (no momentum)."""
    lr = 0.01
    weight_decay = 0.0

    # miniGrad
    w_data = np.random.randn(5, 3)
    w_mg = Tensor(w_data.copy(), requires_grad=True)
    mg_opt = SGD([w_mg], lr=lr, momentum=0.0, weight_decay=weight_decay)

    # PyTorch
    w_pt = torch.tensor(w_data.copy(), requires_grad=True, dtype=torch.float64)
    pt_opt = torch.optim.SGD([w_pt], lr=lr, momentum=0.0, weight_decay=weight_decay)

    # Simulate a gradient
    grad = np.random.randn(5, 3)
    w_mg.grad = grad.copy()
    w_pt.grad = torch.tensor(grad.copy(), dtype=torch.float64)

    # Step
    mg_opt.step()
    pt_opt.step()

    np.testing.assert_allclose(w_mg.data, w_pt.detach().numpy(), atol=1e-6)


def test_sgd_momentum_parity():
    """Compare miniGrad SGD with momentum against PyTorch."""
    lr = 0.01
    momentum = 0.9

    w_data = np.random.randn(5, 3)

    # miniGrad
    w_mg = Tensor(w_data.copy(), requires_grad=True)
    mg_opt = SGD([w_mg], lr=lr, momentum=momentum)

    # PyTorch
    w_pt = torch.tensor(w_data.copy(), requires_grad=True, dtype=torch.float64)
    pt_opt = torch.optim.SGD([w_pt], lr=lr, momentum=momentum)

    # Multiple steps to test momentum accumulation
    for _ in range(5):
        grad = np.random.randn(5, 3)
        w_mg.grad = grad.copy()
        w_pt.grad = torch.tensor(grad.copy(), dtype=torch.float64)

        mg_opt.step()
        pt_opt.step()

    np.testing.assert_allclose(w_mg.data, w_pt.detach().numpy(), atol=1e-5)


def test_sgd_weight_decay_parity():
    """Compare miniGrad SGD with weight decay against PyTorch."""
    lr = 0.01
    weight_decay = 0.01

    w_data = np.random.randn(5, 3)

    w_mg = Tensor(w_data.copy(), requires_grad=True)
    mg_opt = SGD([w_mg], lr=lr, weight_decay=weight_decay)

    w_pt = torch.tensor(w_data.copy(), requires_grad=True, dtype=torch.float64)
    pt_opt = torch.optim.SGD([w_pt], lr=lr, weight_decay=weight_decay)

    grad = np.random.randn(5, 3)
    w_mg.grad = grad.copy()
    w_pt.grad = torch.tensor(grad.copy(), dtype=torch.float64)

    mg_opt.step()
    pt_opt.step()

    np.testing.assert_allclose(w_mg.data, w_pt.detach().numpy(), atol=1e-5)


def test_adam_parity():
    """Compare miniGrad Adam with PyTorch Adam."""
    lr = 1e-3
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 0.0

    w_data = np.random.randn(5, 3)

    # miniGrad
    w_mg = Tensor(w_data.copy(), requires_grad=True)
    mg_opt = Adam([w_mg], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    # PyTorch
    w_pt = torch.tensor(w_data.copy(), requires_grad=True, dtype=torch.float64)
    pt_opt = torch.optim.Adam([w_pt], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    # Multiple steps
    for _ in range(10):
        grad = np.random.randn(5, 3)
        w_mg.grad = grad.copy()
        w_pt.grad = torch.tensor(grad.copy(), dtype=torch.float64)

        mg_opt.step()
        pt_opt.step()

    np.testing.assert_allclose(w_mg.data, w_pt.detach().numpy(), atol=1e-5)


def test_adam_weight_decay_parity():
    """Compare miniGrad Adam with weight decay against PyTorch Adam."""
    lr = 1e-3
    betas = (0.9, 0.999)
    weight_decay = 0.01

    w_data = np.random.randn(5, 3)

    w_mg = Tensor(w_data.copy(), requires_grad=True)
    mg_opt = Adam([w_mg], lr=lr, betas=betas, weight_decay=weight_decay)

    w_pt = torch.tensor(w_data.copy(), requires_grad=True, dtype=torch.float64)
    pt_opt = torch.optim.Adam([w_pt], lr=lr, betas=betas, weight_decay=weight_decay)

    for _ in range(10):
        grad = np.random.randn(5, 3)
        w_mg.grad = grad.copy()
        w_pt.grad = torch.tensor(grad.copy(), dtype=torch.float64)

        mg_opt.step()
        pt_opt.step()

    np.testing.assert_allclose(w_mg.data, w_pt.detach().numpy(), atol=1e-5)


if __name__ == "__main__":
    if not HAS_TORCH:
        print("PyTorch not installed. Skipping parity tests.")
        sys.exit(0)

    print("Running optimizer parity tests...")
    tests = [
        test_sgd_parity, test_sgd_momentum_parity,
        test_sgd_weight_decay_parity, test_adam_parity,
        test_adam_weight_decay_parity,
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
