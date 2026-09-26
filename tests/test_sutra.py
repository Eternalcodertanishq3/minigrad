"""
test_sutra.py — Automated Unit Tests for S.U.T.R.A.
(State-space Unified Time-continuous Runge-Kutta Adjoint)

Tests:
1. test_linear_ode_analytical_solution (Euler, RK4, Dopri5 vs exact exponential)
2. test_adjoint_gradient_analytical_parity (Adjoint gradients vs calculus)
3. test_adjoint_vs_discrete_backprop_parity (use_adjoint=True vs False)
4. test_o1_memory_invariance (Autograd graph size O(1) vs O(N))
5. test_dopri5_adaptive_step_control (Tolerance-driven dynamic depth)
6. test_neural_ode_module_training (End-to-end dynamical learning with Adam)
7. test_neural_ode_in_sequential (Integration with Linear and Sequential)
8. test_time_dependent_vector_field (f(t, y) with explicit continuous time)
"""
import numpy as np

from minigrad.graph import topological_sort
from minigrad.nn import Linear, Module, MSELoss, Sequential, Tanh
from minigrad.optim import Adam
from minigrad.sutra import NeuralODE, odeint
from minigrad.tensor import Tensor

# ── 1. Linear ODE Analytical Solution Parity ──────────────────────────

def test_linear_ode_analytical_solution():
    """
    dy/dt = lambda * y  ==>  y(t) = y(0) * exp(lambda * t)
    For y(0) = 2.0, lambda = -0.5, t in [0, 2]:
    Exact y(2.0) = 2.0 * exp(-1.0) = 0.7357588823428847
    """
    lam = -0.5
    y0_val = 2.0
    t_end = 2.0
    exact_y_end = y0_val * np.exp(lam * t_end)

    def linear_vector_field(t, y):
        return y * lam

    # A. Euler solver
    sol_euler = odeint(
        linear_vector_field,
        Tensor([y0_val]),
        (0.0, t_end),
        method="euler",
        options={"n_steps": 200},
        use_adjoint=False,
    )
    assert np.isclose(sol_euler.numpy()[0], exact_y_end, rtol=1e-2)

    # B. RK4 solver
    sol_rk4 = odeint(
        linear_vector_field,
        Tensor([y0_val]),
        (0.0, t_end),
        method="rk4",
        options={"n_steps": 50},
        use_adjoint=False,
    )
    assert np.isclose(sol_rk4.numpy()[0], exact_y_end, rtol=1e-5)

    # C. Dopri5 (adaptive) solver
    sol_dopri5 = odeint(
        linear_vector_field,
        Tensor([y0_val]),
        (0.0, t_end),
        method="dopri5",
        rtol=1e-6,
        atol=1e-6,
        use_adjoint=False,
    )
    assert np.isclose(sol_dopri5.numpy()[0], exact_y_end, rtol=1e-5)


# ── 2. Adjoint Gradient Analytical Parity ────────────────────────────

class ParametricScalarField(Module):
    def __init__(self, init_weight: float):
        super().__init__()
        self.theta = Tensor([init_weight], requires_grad=True)

    def forward(self, t, y):
        return y * self.theta


def test_adjoint_gradient_analytical_parity():
    """
    Scalar ODE: dy/dt = theta * y, y(0) = y0, T = 1.0.
    Exact terminal state: y(1) = y0 * exp(theta).
    Loss: L = 0.5 * (y(1))^2 = 0.5 * y0^2 * exp(2 * theta).

    Exact analytical derivatives:
    dL / d(theta) = y0^2 * exp(2 * theta)
    dL / d(y0)    = y0 * exp(2 * theta)
    """
    y0_init = 1.5
    theta_init = 0.5

    exact_dL_dtheta = (y0_init ** 2) * np.exp(2.0 * theta_init)
    exact_dL_dy0 = y0_init * np.exp(2.0 * theta_init)

    # Initialize model
    field = ParametricScalarField(theta_init)
    y0 = Tensor([y0_init], requires_grad=True)

    # Forward with Pontryagin Continuous Adjoint
    y1 = odeint(
        field,
        y0,
        (0.0, 1.0),
        method="rk4",
        options={"n_steps": 100},
        use_adjoint=True,
    )

    loss = (y1 ** 2) * 0.5
    loss.backward()

    computed_dL_dy0 = y0.grad[0]
    computed_dL_dtheta = field.theta.grad[0]

    # Check match against exact analytical calculus
    assert np.isclose(computed_dL_dy0, exact_dL_dy0, rtol=1e-4)
    assert np.isclose(computed_dL_dtheta, exact_dL_dtheta, rtol=1e-4)


# ── 3. Adjoint vs Discrete Backprop Parity ───────────────────────────

def test_adjoint_vs_discrete_backprop_parity():
    """
    Compares parameter and input gradients between:
    - use_adjoint=False (discrete unrolled autograd)
    - use_adjoint=True (continuous augmented adjoint ODE)
    for a non-linear 2D MLP vector field.
    """
    np.random.seed(42)

    # Build vector field
    class VectorNet(Module):
        def __init__(self):
            super().__init__()
            self.l1 = Linear(2, 4)
            self.tanh = Tanh()
            self.l2 = Linear(4, 2)

        def forward(self, t, y):
            return self.l2(self.tanh(self.l1(y)))

    x_data = np.array([[0.6, -0.4]], dtype=np.float64)

    # Run 1: Discrete Unrolled
    model_unrolled = VectorNet()
    x_unrolled = Tensor(x_data.copy(), requires_grad=True)

    out_unrolled = odeint(
        model_unrolled,
        x_unrolled,
        (0.0, 0.5),
        method="rk4",
        options={"n_steps": 25},
        use_adjoint=False,
    )
    loss_unrolled = (out_unrolled ** 2).sum()
    loss_unrolled.backward()

    # Run 2: S.U.T.R.A. Adjoint Method with identical weights
    model_adjoint = VectorNet()
    model_adjoint.l1.weight.data = model_unrolled.l1.weight.data.copy()
    model_adjoint.l1.bias.data = model_unrolled.l1.bias.data.copy()
    model_adjoint.l2.weight.data = model_unrolled.l2.weight.data.copy()
    model_adjoint.l2.bias.data = model_unrolled.l2.bias.data.copy()

    x_adjoint = Tensor(x_data.copy(), requires_grad=True)

    out_adjoint = odeint(
        model_adjoint,
        x_adjoint,
        (0.0, 0.5),
        method="rk4",
        options={"n_steps": 25},
        use_adjoint=True,
    )
    loss_adjoint = (out_adjoint ** 2).sum()
    loss_adjoint.backward()

    # Verify forward values match
    np.testing.assert_allclose(out_unrolled.numpy(), out_adjoint.numpy(), rtol=1e-5)

    # Verify input gradient match
    np.testing.assert_allclose(x_unrolled.grad, x_adjoint.grad, rtol=1e-3, atol=1e-3)

    # Verify parameter gradients match
    for p_unrolled, p_adj in zip(model_unrolled.parameters(), model_adjoint.parameters()):
        np.testing.assert_allclose(p_unrolled.grad, p_adj.grad, rtol=1e-3, atol=1e-3)


# ── 4. O(1) Memory Invariance ────────────────────────────────────────

def test_o1_memory_invariance():
    """
    Asserts that the autograd graph size is strictly O(1) independent of
    the number of ODE steps N when using Pontryagin Adjoint (use_adjoint=True),
    unlike unrolled autograd which scales linearly O(N).
    """
    def simple_field(t, y):
        return -y

    # For use_adjoint=True, graph size is strictly 1 output node
    for n_steps in [10, 50, 100, 300]:
        x = Tensor([1.0], requires_grad=True)
        out = odeint(
            simple_field,
            x,
            (0.0, 1.0),
            method="euler",
            options={"n_steps": n_steps},
            use_adjoint=True,
        )
        topo = topological_sort(out)
        # Graph contains exactly [x, out]
        assert len(topo) == 2, f"Expected 2 nodes in O(1) adjoint graph, got {len(topo)}"

    # For use_adjoint=False, graph nodes scale linearly with n_steps
    x_10 = Tensor([1.0], requires_grad=True)
    out_10 = odeint(simple_field, x_10, (0.0, 1.0), method="euler", options={"n_steps": 10}, use_adjoint=False)
    nodes_10 = len(topological_sort(out_10))

    x_50 = Tensor([1.0], requires_grad=True)
    out_50 = odeint(simple_field, x_50, (0.0, 1.0), method="euler", options={"n_steps": 50}, use_adjoint=False)
    nodes_50 = len(topological_sort(out_50))

    assert nodes_50 > nodes_10 * 3, f"Expected graph size to grow with steps: {nodes_10} -> {nodes_50}"


# ── 5. Dopri5 Adaptive Step Control ──────────────────────────────────

def test_dopri5_adaptive_step_control():
    """
    Verifies that the adaptive Dormand-Prince solver dynamically adjusts
    depth (number of steps / evaluations) based on error tolerance.
    """
    def non_linear_oscillator(t, y):
        # y has 2 components: [position, velocity]
        # x'' = -x - (x^2 - 1)*x' (Van der Pol-like non-linearity)
        x = y[0]
        v = y[1]
        dx = v
        dv = -x - (x ** 2 - 1.0) * v
        return np.array([dx, dv])

    y0 = np.array([1.0, 0.0])

    # Loose tolerance: should take fewer steps
    _, telem_loose = odeint(
        non_linear_oscillator,
        y0,
        (0.0, 2.0),
        method="dopri5",
        rtol=1e-2,
        atol=1e-2,
        return_telemetry=True,
    )

    # Tight tolerance: should take more steps to satisfy precision
    _, telem_tight = odeint(
        non_linear_oscillator,
        y0,
        (0.0, 2.0),
        method="dopri5",
        rtol=1e-6,
        atol=1e-6,
        return_telemetry=True,
    )

    assert telem_tight.n_steps > telem_loose.n_steps
    assert telem_tight.n_evals > telem_loose.n_evals
    assert len(telem_tight.step_sizes) > 0


# ── 6. NeuralODE Module End-to-End Training ──────────────────────────

def test_neural_ode_module_training():
    """
    Trains a NeuralODE module to learn 2D spiral dynamics using Adam.
    True spiral dynamics: dx/dt = -0.1*x - y, dy/dt = x - 0.1*y
    """
    np.random.seed(123)

    # Create target batch from spiral (decay alpha=0.1, frequency omega=1.0)
    batch_x0 = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)

    # True target at t=0.5
    # Analytical matrix exponential for 2D spiral: exp(A*t) = exp(-0.1*t) * [[cos(t), -sin(t)], [sin(t), cos(t)]]
    t_target = 0.5
    exp_At = np.exp(-0.1 * t_target) * np.array([
        [np.cos(t_target), -np.sin(t_target)],
        [np.sin(t_target), np.cos(t_target)],
    ])
    target_data = batch_x0 @ exp_At.T
    target_tensor = Tensor(target_data)

    # Define NeuralODE model
    vector_field = Sequential([
        Linear(2, 16),
        Tanh(),
        Linear(16, 2),
    ])
    node = NeuralODE(vector_field, t_span=(0.0, 0.5), solver="rk4", use_adjoint=True)
    optimizer = Adam(node.parameters(), lr=0.05)
    loss_fn = MSELoss()

    initial_loss = None
    final_loss = None

    for epoch in range(25):
        optimizer.zero_grad()
        pred = node(Tensor(batch_x0))
        loss = loss_fn(pred, target_tensor)

        if epoch == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < initial_loss * 0.5, f"Expected loss drop > 50%, got {initial_loss:.4f} -> {final_loss:.4f}"


# ── 7. NeuralODE in Composite Sequential Pipeline ─────────────────────

def test_neural_ode_in_sequential():
    """
    Validates that NeuralODE integrates smoothly inside a composite
    Sequential model with standard feedforward layers.
    """
    vector_field = Sequential([
        Linear(4, 8),
        Tanh(),
        Linear(8, 4),
    ])
    node = NeuralODE(vector_field, t_span=(0.0, 0.2), solver="rk4", use_adjoint=True)

    composite = Sequential([
        Linear(3, 4),
        node,
        Linear(4, 1),
    ])

    x = Tensor(np.random.randn(2, 3), requires_grad=True)
    out = composite(x)
    assert out.shape == (2, 1)

    loss = out.sum()
    loss.backward()

    # Verify input and all parameters received gradients
    assert x.grad is not None
    assert np.linalg.norm(x.grad) > 0.0
    for p in composite.parameters():
        assert p.grad is not None
        assert np.linalg.norm(p.grad) > 0.0


# ── 8. Time-Dependent Vector Field f(t, y) ───────────────────────────

def test_time_dependent_vector_field():
    """
    dy/dt = -y + cos(t), y(0) = 0
    Exact analytical solution:
    y(t) = 0.5 * (sin(t) + cos(t) - exp(-t))
    At t = pi:
    y(pi) = 0.5 * (0 + (-1) - exp(-pi)) = 0.5 * (-1 - exp(-pi)) ≈ -0.521606
    """
    def non_autonomous_field(t, y):
        return -y + np.cos(t)

    t_eval = np.pi
    exact_y = 0.5 * (np.sin(t_eval) + np.cos(t_eval) - np.exp(-t_eval))

    y0 = Tensor([0.0], requires_grad=True)
    out = odeint(
        non_autonomous_field,
        y0,
        (0.0, t_eval),
        method="rk4",
        options={"n_steps": 100},
        use_adjoint=True,
    )

    np.testing.assert_allclose(out.numpy()[0], exact_y, rtol=1e-3)
