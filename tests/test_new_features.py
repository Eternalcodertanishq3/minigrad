"""
test_new_features.py — Unit tests for new features and bug fixes.

Tests:
- Pow negative exponent zero base safety
- no_grad context manager and decorator
- Embedding layer forward & backward
- LayerNorm layer forward & backward
- Gradient clipping (clip_grad_norm_, clip_grad_value_)
- Learning rate schedulers (StepLR, CosineAnnealingLR, ExponentialLR)
- Einsum autograd operation
"""
import math
import numpy as np
import pytest

from minigrad.tensor import Tensor
from minigrad.graph import no_grad, is_grad_enabled
from minigrad.nn import Embedding, LayerNorm
from minigrad.nn.utils import clip_grad_norm_, clip_grad_value_
from minigrad.optim import SGD, StepLR, CosineAnnealingLR, ExponentialLR
from minigrad.ops import einsum


def test_pow_negative_exponent_zero_base():
    """Negative pow with 0.0 base should not produce NaN gradients."""
    x = Tensor([[0.0, 2.0]], requires_grad=True)
    y = x ** (-2)
    y.backward()
    assert not np.any(np.isnan(x.grad)), f"x.grad contains NaN: {x.grad}"
    assert not np.any(np.isinf(x.grad)), f"x.grad contains Inf: {x.grad}"


def test_no_grad_context_manager():
    """no_grad should function as both a context manager and decorator, skipping graph construction."""
    assert is_grad_enabled() is True

    x = Tensor([1.0, 2.0], requires_grad=True)

    with no_grad():
        assert is_grad_enabled() is False
        y = x * 2
        assert y.requires_grad is False
        assert len(y._prev) == 0

    assert is_grad_enabled() is True

    @no_grad()
    def fn():
        assert is_grad_enabled() is False
        z = x + 3
        assert z.requires_grad is False
        assert len(z._prev) == 0

    fn()
    assert is_grad_enabled() is True


def test_embedding_layer():
    """Test Embedding forward and backward lookup."""
    emb = Embedding(num_embeddings=10, embedding_dim=4)
    idx = Tensor(np.array([[1, 2], [3, 1]]), requires_grad=False)
    out = emb(idx)

    assert out.data.shape == (2, 2, 4)
    out.sum().backward()

    # Index 1 was retrieved twice, so its weight gradient should be 2.0
    assert np.allclose(emb.weight.grad[1], 2.0)
    assert np.allclose(emb.weight.grad[2], 1.0)
    assert np.allclose(emb.weight.grad[0], 0.0)


def test_layernorm_layer():
    """Test LayerNorm normalizes mean=0, std=1 and learns gamma/beta."""
    ln = LayerNorm(normalized_shape=(4,))
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    out = ln(x)

    assert out.data.shape == (2, 3, 4)
    means = out.data.mean(axis=-1)
    vars_ = out.data.var(axis=-1)
    assert np.allclose(means, 0.0, atol=1e-4)
    assert np.allclose(vars_, 1.0, atol=1e-3)

    out.sum().backward()
    assert not np.any(np.isnan(x.grad))
    assert not np.allclose(ln.gamma.grad, 0.0)
    assert not np.allclose(ln.beta.grad, 0.0)

    # Affine scaling verification
    ln.gamma.data = np.full((4,), 2.0)
    ln.beta.data = np.full((4,), 5.0)
    out2 = ln(x)
    assert np.allclose(out2.data.mean(axis=-1), 5.0, atol=1e-3)
    assert np.allclose(out2.data.var(axis=-1), 4.0, atol=1e-2)


def test_clip_grad_norm():
    """Test L2 norm clipping of gradients."""
    p1 = Tensor([1.0, 2.0], requires_grad=True)
    p2 = Tensor([3.0, 4.0], requires_grad=True)
    p1.grad = np.array([3.0, 4.0])  # norm = 5
    p2.grad = np.array([0.0, 0.0])

    total_norm = clip_grad_norm_([p1, p2], max_norm=2.5)
    assert math.isclose(total_norm, 5.0)
    # p1.grad should be scaled by 2.5 / 5.0 = 0.5
    assert np.allclose(p1.grad, [1.5, 2.0])


def test_clip_grad_value():
    """Test value clipping of gradients."""
    p = Tensor([1.0, 2.0], requires_grad=True)
    p.grad = np.array([-10.0, 5.0])
    clip_grad_value_([p], clip_value=2.0)
    assert np.allclose(p.grad, [-2.0, 2.0])


def test_lr_schedulers():
    """Test StepLR, CosineAnnealingLR, ExponentialLR schedules."""
    p = Tensor([1.0], requires_grad=True)
    opt = SGD([p], lr=0.1)

    # StepLR
    scheduler = StepLR(opt, step_size=2, gamma=0.5)
    assert math.isclose(opt.lr, 0.1)
    scheduler.step()
    assert math.isclose(opt.lr, 0.1)
    scheduler.step()
    assert math.isclose(opt.lr, 0.05)

    # CosineAnnealingLR
    opt.lr = 0.1
    c_scheduler = CosineAnnealingLR(opt, T_max=10, eta_min=0.0)
    assert math.isclose(opt.lr, 0.1)
    c_scheduler.step(5)  # half-way -> lr should be 0.05
    assert math.isclose(opt.lr, 0.05, abs_tol=1e-5)

    # ExponentialLR
    opt.lr = 0.1
    e_scheduler = ExponentialLR(opt, gamma=0.9)
    assert math.isclose(opt.lr, 0.1)
    e_scheduler.step()
    assert math.isclose(opt.lr, 0.09)


def test_einsum_autograd():
    """Test einsum matrix multiply and gradient computation."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)

    c = einsum("ij,jk->ik", a, b)
    expected = a.data @ b.data
    assert np.allclose(c.data, expected)

    c.sum().backward()
    # d(sum(A @ B))/dA = grad @ B.T = 1 @ B.T
    expected_a_grad = np.ones((2, 2)) @ b.data.T
    assert np.allclose(a.grad, expected_a_grad)


def test_einsum_trace_autograd():
    """Test einsum matrix trace (repeated indices) forward and backward."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    tr = einsum("ii->", a)
    assert math.isclose(float(tr.data), 5.0)

    tr.backward()
    # d(trace(A))/dA = eye(2)
    assert np.allclose(a.grad, np.eye(2))

