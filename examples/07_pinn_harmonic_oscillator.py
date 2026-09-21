"""
07_pinn_harmonic_oscillator.py — Physics-Informed Neural Network (PINN) for a Damped Oscillator.

Solves the second-order Ordinary Differential Equation (ODE):
    d²u/dt² + 2ζω₀ du/dt + ω₀² u = 0
Subject to initial conditions:
    u(0) = 1.0,  u'(0) = 0.0

A neural network u_θ(t) is trained entirely by minimizing:
1. PDE residual loss on collocation points: L_pde = ||u_tt + 2ζω₀ u_t + ω₀² u||²
2. Boundary condition loss at t=0:          L_bc  = (u(0) - 1)² + (u'(0) - 0)²

This demonstrates miniGrad's higher-order automatic differentiation (`grad` with `create_graph=True`),
solving differential equations without labeled training data.
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from minigrad.tensor import Tensor
from minigrad.nn.module import Module
from minigrad.nn.linear import Linear
from minigrad.optim.adam import Adam
from minigrad.autograd import grad


# ── Physical System Parameters ───────────────────────────────────────

ZETA = 0.1       # Damping ratio (underdamped: 0 < ζ < 1)
OMEGA_0 = 4.0     # Natural frequency
T_MAX = 3.0       # Time horizon
N_COLLOC = 40     # Number of collocation points


def exact_solution(t: np.ndarray, zeta: float = ZETA, omega_0: float = OMEGA_0) -> np.ndarray:
    """Analytical solution for the underdamped harmonic oscillator."""
    omega_d = omega_0 * np.sqrt(1.0 - zeta ** 2)
    decay = np.exp(-zeta * omega_0 * t)
    return decay * (np.cos(omega_d * t) + (zeta * omega_0 / omega_d) * np.sin(omega_d * t))


# ── PINN Architecture ────────────────────────────────────────────────

class PINN(Module):
    """Multi-Layer Perceptron with smooth Tanh activations for second-order PDE solving."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = Linear(1, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.out = Linear(hidden_dim, 1)

    def forward(self, t: Tensor) -> Tensor:
        h1 = self.fc1(t).tanh()
        h2 = self.fc2(h1).tanh()
        return self.out(h2)


# ── Training ─────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Physics-Informed Neural Network (PINN) — miniGrad")
    print("  Damped Harmonic Oscillator: u'' + 2ζω₀ u' + ω₀² u = 0")
    print("=" * 65)

    np.random.seed(42)
    model = PINN(hidden_dim=32)

    optimizer = Adam(model.parameters(), lr=1e-2)

    # Collocation points inside domain (0, T_MAX]
    t_colloc_np = np.linspace(0.01, T_MAX, N_COLLOC).reshape(-1, 1)

    # Initial condition point at t = 0
    t_0_np = np.array([[0.0]])

    print(f"\nModel Parameters: {sum(p.data.size for p in model.parameters()):,}")
    print(f"Collocation Points: {N_COLLOC} points in (0, {T_MAX}]")
    print("\nTraining PINN with double-backward PDE loss:")
    print("-" * 65)

    t_start = time.time()
    steps = 300

    for step in range(1, steps + 1):
        # 1. Physics loss at collocation points
        t = Tensor(t_colloc_np, requires_grad=True)
        u = model(t)

        # 1st spatial derivative: u_t = du/dt (create_graph=True)
        u_t = grad(u.sum(), t, create_graph=True)[0]

        # 2nd spatial derivative: u_tt = d²u/dt² (create_graph=True)
        u_tt = grad(u_t.sum(), t, create_graph=True)[0]

        # Damped oscillator PDE residual: R = u_tt + 2ζω₀ u_t + ω₀² u
        residual = u_tt + (2.0 * ZETA * OMEGA_0) * u_t + (OMEGA_0 ** 2) * u
        loss_pde = (residual ** 2).mean()

        # 2. Initial condition loss at t = 0: u(0) = 1, u'(0) = 0
        t0 = Tensor(t_0_np, requires_grad=True)
        u0 = model(t0)
        u0_t = grad(u0.sum(), t0, create_graph=True)[0]

        loss_ic = ((u0 - 1.0) ** 2).sum() + ((u0_t - 0.0) ** 2).sum()

        # Total weighted loss
        total_loss = loss_pde + 10.0 * loss_ic

        # Optimize neural network weights
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t_start
            pde_val = float(loss_pde.data)
            ic_val = float(loss_ic.data)
            tot_val = float(total_loss.data)
            print(f"  step {step:>4d}/{steps} | total: {tot_val:.4f} | "
                  f"pde: {pde_val:.4f} | ic: {ic_val:.4f} | time: {elapsed:.1f}s")

    # ── Evaluation vs Analytical Solution ────────────────────────────
    print("-" * 65)
    t_test = np.linspace(0, T_MAX, 100).reshape(-1, 1)
    u_pred = model(Tensor(t_test)).data.flatten()
    u_true = exact_solution(t_test.flatten())

    mse = float(np.mean((u_pred - u_true) ** 2))
    print(f"\nEvaluation over 100 test points:")
    print(f"  Mean Squared Error vs Analytical Truth: {mse:.6f}")
    print(f"  u(0) predicted: {u_pred[0]:.4f} (target: 1.0000)")
    print(f"  u(T/2) predicted: {u_pred[50]:.4f} (target: {u_true[50]:.4f})")
    print(f"  u(T) predicted: {u_pred[-1]:.4f} (target: {u_true[-1]:.4f})")
    print("\nPINN demonstration complete!")


if __name__ == "__main__":
    main()
