"""
test_double_backward.py — Unit tests for higher-order automatic differentiation and create_graph=True.

Tests:
- Scalar higher-order derivatives (x^3, x^4, sin, cos, exp, log, tanh)
- Multivariable vector & matrix derivatives
- Full Hessian matrix computation
- Tensor.backward(create_graph=True) with loss on gradient
- Physics-Informed Neural Network (PINN) PDE residual backpropagation
"""
import numpy as np
import pytest

from minigrad.autograd import grad, hessian
from minigrad.graph_opt import fused_linear, fused_linear_relu
from minigrad.nn import ELU, LeakyReLU
from minigrad.ops import clip, log_softmax, softmax
from minigrad.ops import max as tmax
from minigrad.tensor import Tensor


def test_scalar_polynomial_double_backward():
    """Verify d²(x³)/dx² = 6x and d²(x⁴)/dx² = 12x²."""
    # y = x^3 at x = 3 -> y' = 27, y'' = 18
    x = Tensor(3.0, requires_grad=True)
    y = x ** 3
    (dy_dx,) = grad(y, x, create_graph=True)
    assert np.isclose(dy_dx.data, 27.0)

    (d2y_dx2,) = grad(dy_dx, x, create_graph=False)
    assert np.isclose(d2y_dx2.data, 18.0)

    # y = x^4 at x = 2 -> y' = 32, y'' = 48
    x2 = Tensor(2.0, requires_grad=True)
    y2 = x2 ** 4
    (dy2_dx2,) = grad(y2, x2, create_graph=True)
    assert np.isclose(dy2_dx2.data, 32.0)

    (d2y2_dx2,) = grad(dy2_dx2, x2, create_graph=False)
    assert np.isclose(d2y2_dx2.data, 48.0)


def test_trigonometric_double_backward():
    """Verify d²(sin(x))/dx² = -sin(x) and d²(cos(x))/dx² = -cos(x)."""
    val = 1.2
    x = Tensor(val, requires_grad=True)
    y = x.sin()

    (dy_dx,) = grad(y, x, create_graph=True)
    assert np.isclose(dy_dx.data, np.cos(val))

    (d2y_dx2,) = grad(dy_dx, x, create_graph=False)
    assert np.isclose(d2y_dx2.data, -np.sin(val))

    # cos test
    x2 = Tensor(val, requires_grad=True)
    y2 = x2.cos()
    (dy2_dx2,) = grad(y2, x2, create_graph=True)
    assert np.isclose(dy2_dx2.data, -np.sin(val))

    (d2y2_dx2,) = grad(dy2_dx2, x2, create_graph=False)
    assert np.isclose(d2y2_dx2.data, -np.cos(val))


def test_transcendental_double_backward():
    """Verify d²(exp(x))/dx² = exp(x) and d²(ln(x))/dx² = -1/x²."""
    val = 2.0
    x = Tensor(val, requires_grad=True)
    y = x.exp()

    (dy_dx,) = grad(y, x, create_graph=True)
    assert np.isclose(dy_dx.data, np.exp(val))
    (d2y_dx2,) = grad(dy_dx, x, create_graph=False)
    assert np.isclose(d2y_dx2.data, np.exp(val))

    # log test
    x2 = Tensor(val, requires_grad=True)
    y2 = x2.log()
    (dy2_dx2,) = grad(y2, x2, create_graph=True)
    assert np.isclose(dy2_dx2.data, 1.0 / val)
    (d2y2_dx2,) = grad(dy2_dx2, x2, create_graph=False)
    assert np.isclose(d2y2_dx2.data, -1.0 / (val ** 2))


def test_tanh_double_backward():
    """Verify d²(tanh(x))/dx² = -2 tanh(x) (1 - tanh²(x))."""
    val = 0.7
    x = Tensor(val, requires_grad=True)
    y = x.tanh()

    (dy_dx,) = grad(y, x, create_graph=True)
    expected_dy = 1.0 - np.tanh(val) ** 2
    assert np.isclose(dy_dx.data, expected_dy)

    (d2y_dx2,) = grad(dy_dx, x, create_graph=False)
    expected_d2y = -2.0 * np.tanh(val) * (1.0 - np.tanh(val) ** 2)
    assert np.isclose(d2y_dx2.data, expected_d2y)


def test_hessian_quadratic_form():
    """Verify Hessian of 0.5 * x^T A x equals the symmetric matrix A."""
    A = np.array([[3.0, 1.5], [1.5, 4.0]])
    x_val = np.array([[1.0], [-2.0]])

    A_t = Tensor(A, requires_grad=False)
    x_t = Tensor(x_val, requires_grad=True)

    # f(x) = 0.5 * (x^T @ A @ x)
    f = ((x_t.transpose() @ A_t @ x_t) * 0.5).sum()

    H = hessian(f, x_t)
    assert H.data.shape == (2, 2)
    assert np.allclose(H.data, A, atol=1e-10)


def test_tensor_backward_create_graph():
    """Verify Tensor.backward(create_graph=True) populates .grad with differentiable Tensors."""
    x = Tensor(3.0, requires_grad=True)
    y = x ** 3
    y.backward(create_graph=True)

    assert isinstance(x.grad, Tensor)
    assert np.isclose(x.grad.data, 27.0)

    # Second backward through the gradient!
    # L = (x.grad)^2 = (27)^2 = 729
    # dL/dx = 2 * (x.grad) * (d(x.grad)/dx) = 2 * 27 * 18 = 972
    loss = (x.grad ** 2)
    (dloss_dx,) = grad(loss, x, create_graph=False)
    assert np.isclose(dloss_dx.data, 972.0)


def test_pinn_loss_backpropagation():
    """
    Verify Physics-Informed Neural Network (PINN) training step:
    Loss = (u_xx + u)^2
    Backpropagation through the PDE residual updates neural network weights.
    """
    np.random.seed(42)
    x = Tensor([[0.4]], requires_grad=True)
    W1 = Tensor(np.random.randn(1, 4), requires_grad=True)
    W2 = Tensor(np.random.randn(4, 1), requires_grad=True)

    # Forward model: u(x) = tanh(x @ W1) @ W2
    h = (x @ W1).tanh()
    u = h @ W2

    # 1st spatial derivative: u_x = du/dx (create_graph=True)
    (u_x,) = grad(u, x, create_graph=True)

    # 2nd spatial derivative: u_xx = d²u/dx² (create_graph=True)
    (u_xx,) = grad(u_x, x, create_graph=True)

    # Damped harmonic oscillator / Helmholtz residual: u_xx + u
    residual = u_xx + u
    pde_loss = (residual ** 2).sum()

    # Backpropagate PDE loss to network weights W1 and W2!
    grads = grad(pde_loss, [W1, W2], create_graph=False)
    grad_W1, grad_W2 = grads

    assert grad_W1 is not None and grad_W1.shape == W1.shape
    assert grad_W2 is not None and grad_W2.shape == W2.shape
    assert not np.all(grad_W1.data == 0), "W1 must receive non-zero PDE residual gradients"
    assert not np.all(grad_W2.data == 0), "W2 must receive non-zero PDE residual gradients"


def test_hessian_differentiable_third_derivative():
    """
    Verify that hessian(y, x, create_graph=True) builds a genuinely differentiable computation graph.
    y = x^3
    H = d²y/dx² = 6x
    d(H.sum())/dx = 6
    """
    x = Tensor([2.0], requires_grad=True)
    y = (x ** 3).sum()
    H = hessian(y, x, create_graph=True)
    assert H.requires_grad
    assert np.isclose(H.data[0, 0], 12.0)

    (third_deriv,) = grad(H.sum(), x)
    assert np.isclose(third_deriv.data[0], 6.0)

    # Multi-element case
    x2 = Tensor([2.0, 3.0], requires_grad=True)
    y2 = (x2 ** 3).sum()
    H2 = hessian(y2, x2, create_graph=True)
    assert H2.shape == (2, 2)
    assert np.isclose(H2.data[0, 0], 12.0)
    assert np.isclose(H2.data[1, 1], 18.0)
    assert np.isclose(H2.data[0, 1], 0.0)

    (third_deriv2,) = grad(H2.sum(), x2)
    assert np.allclose(third_deriv2.data, [6.0, 6.0])


def test_to_and_fused_linear_vjp():
    """Verify vjp for to, fused_linear, fused_linear_relu, and unregistered op error."""
    # 1. to() vjp casts grad back to input dtype
    x = Tensor([1.0, 2.0], dtype=np.float32, requires_grad=True)
    y = x.to(np.float64) * 2.0
    (gx,) = grad(y.sum(), x, create_graph=True)
    assert gx.dtype == np.float32
    assert np.allclose(gx.data, [2.0, 2.0])

    # 2. fused_linear vjp
    w = Tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
    b = Tensor([0.5, 1.5], requires_grad=True)
    x_in = Tensor([[1.0, 2.0]], requires_grad=True)
    out_lin = fused_linear(x_in, w, b)
    (gx_lin, gw_lin, gb_lin) = grad(out_lin.sum(), [x_in, w, b], create_graph=True)
    assert np.allclose(gx_lin.data, [[5.0, 9.0]])
    assert np.allclose(gb_lin.data, [1.0, 1.0])

    # 3. fused_linear_relu vjp
    out_relu = fused_linear_relu(x_in, w, b)
    (gx_r, gw_r, gb_r) = grad(out_relu.sum(), [x_in, w, b], create_graph=True)
    assert np.allclose(gx_r.data, [[5.0, 9.0]])

    # 4. Unknown op raises NotImplementedError
    dummy_in = Tensor([1.0], requires_grad=True)
    dummy_out = Tensor([2.0], requires_grad=True, _children=(dummy_in,), _op="unregistered_custom_op")
    with pytest.raises(NotImplementedError, match="No VJP registered for operation 'unregistered_custom_op'"):
        _ = grad(dummy_out, dummy_in)


def test_hessian_multi_input_heterogeneous_shapes():
    """Verify that hessian with multiple inputs of heterogeneous shapes supports create_graph and 3rd derivatives."""
    x1 = Tensor([2.0, 3.0], requires_grad=True)
    x2 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    # y = (x1^3).sum() + (x2^3).sum()
    y = (x1 ** 3).sum() + (x2 ** 3).sum()

    H = hessian(y, [x1, x2], create_graph=True)
    assert H.shape == (6, 6)
    assert H.requires_grad

    # Analytical 2nd derivatives on diagonal:
    # d²y/dx1_0² = 6 * 2 = 12
    # d²y/dx1_1² = 6 * 3 = 18
    # d²y/dx2_00² = 6 * 1 = 6
    # d²y/dx2_01² = 6 * 2 = 12
    # d²y/dx2_10² = 6 * 3 = 18
    # d²y/dx2_11² = 6 * 4 = 24
    expected_diag = [12.0, 18.0, 6.0, 12.0, 18.0, 24.0]
    assert np.allclose(np.diag(H.data), expected_diag)

    # Off-diagonals must be 0
    np.fill_diagonal(H.data, 0.0)
    assert np.allclose(H.data, 0.0)

    # 3rd derivatives: d(H.sum())/dx = 6 for each element
    g3_1, g3_2 = grad(H.sum(), [x1, x2])
    assert np.allclose(g3_1.data, [6.0, 6.0])
    assert np.allclose(g3_2.data, [[6.0, 6.0], [6.0, 6.0]])


def test_fused_linear_relu_3d_vjp_and_double_backward():
    """Verify 3D activations in fused_linear_relu work for VJP and double backward."""
    x = Tensor(np.random.randn(2, 5, 3), requires_grad=True)
    w = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.randn(4), requires_grad=True)

    out = fused_linear_relu(x, w, b)
    assert out.shape == (2, 5, 4)

    # First derivative w.r.t [x, w, b]
    gx, gw, gb = grad(out.sum(), [x, w, b], create_graph=True)
    assert gx.shape == (2, 5, 3)
    assert gw.shape == (3, 4)
    assert gb.shape == (4,)

    # Double backward: differentiate gradient sum w.r.t inputs
    g2_x, g2_w, g2_b = grad(gx.sum(), [x, w, b], allow_unused=True)
    assert g2_w.shape == (3, 4)


def test_vjp_extended_ops():
    """Verify VJPs for clip, softmax, log_softmax, max, leaky_relu, and elu."""
    # 1. clip VJP
    x_clip = Tensor([0.5, 5.0, -5.0], requires_grad=True)
    c = clip(x_clip, -1.0, 1.0)
    (g_clip,) = grad(c.sum(), x_clip, create_graph=True)
    assert np.allclose(g_clip.data, [1.0, 0.0, 0.0])

    # 2. softmax VJP
    x_sm = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    s = softmax(x_sm)
    (g_sm,) = grad(s.sum(), x_sm, create_graph=True)
    assert np.allclose(g_sm.data, [0.0, 0.0, 0.0], atol=1e-7)

    # 3. log_softmax VJP
    ls = log_softmax(x_sm)
    (g_ls,) = grad((ls ** 2).sum(), x_sm, create_graph=True)
    assert g_ls.shape == (3,)

    # 4. max VJP
    x_max = Tensor([[1.0, 5.0], [3.0, 2.0]], requires_grad=True)
    m = tmax(x_max, axis=1)
    (g_max,) = grad(m.sum(), x_max, create_graph=True)
    assert np.allclose(g_max.data, [[0.0, 1.0], [1.0, 0.0]])

    # 5. LeakyReLU VJP
    lrelu = LeakyReLU(negative_slope=0.1)
    x_lr = Tensor([-2.0, 3.0], requires_grad=True)
    (g_lr,) = grad(lrelu(x_lr).sum(), x_lr, create_graph=True)
    assert np.allclose(g_lr.data, [0.1, 1.0])

    # 6. ELU VJP
    elu = ELU(alpha=1.0)
    x_elu = Tensor([0.0, 2.0], requires_grad=True)
    (g_elu,) = grad(elu(x_elu).sum(), x_elu, create_graph=True)
    assert np.allclose(g_elu.data, [1.0, 1.0])


def test_dtype_preservation_autograd():
    """Verify that float32 inputs maintain float32 dtypes through autograd and hessian."""
    x = Tensor([2.0, 3.0], dtype=np.float32, requires_grad=True)
    assert x.sum().dtype == np.float32
    assert x.mean().dtype == np.float32

    y = (x ** 3).sum()
    (g,) = grad(y, x, create_graph=True)
    assert g.dtype == np.float32

    H = hessian(y, x, create_graph=True)
    assert H.dtype == np.float32

    (third_deriv,) = grad(H.sum(), x)
    assert third_deriv.dtype == np.float32


