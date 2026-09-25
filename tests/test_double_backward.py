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

from minigrad.tensor import Tensor
from minigrad.autograd import grad, hessian


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
