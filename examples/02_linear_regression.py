"""
02_linear_regression.py — Learn y = mx + b using gradient descent.

This example demonstrates:
1. Defining a simple model (linear function)
2. Computing loss (MSE)
3. Backpropagating gradients
4. Updating parameters via an optimizer

The model learns the true parameters from noisy data.

Run: python examples/02_linear_regression.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from minigrad import Tensor
from minigrad.optim import SGD


def generate_data(m_true: float = 2.0, b_true: float = -1.0, n: int = 100, noise: float = 0.5):
    """Generate synthetic linear data with noise."""
    np.random.seed(42)
    x = np.random.randn(n)
    y = m_true * x + b_true + noise * np.random.randn(n)
    return x, y


def train():
    print("=" * 60)
    print("Linear Regression — Learning y = mx + b")
    print("=" * 60)

    # Generate synthetic data
    x_data, y_data = generate_data(m_true=2.0, b_true=-1.0, n=100, noise=0.3)
    print(f"Generated {len(x_data)} data points")
    print(f"True parameters: m = 2.0, b = -1.0")

    # Initialize parameters (random guesses)
    np.random.seed(0)
    m = Tensor(np.random.randn(1), requires_grad=True)  # slope
    b = Tensor(np.zeros(1), requires_grad=True)         # intercept

    optimizer = SGD([m, b], lr=0.1)

    # Training loop
    print("\nTraining...")
    print("-" * 40)
    epochs = 200
    for epoch in range(epochs):
        # Forward pass: y_pred = m*x + b
        x = Tensor(x_data.reshape(-1, 1))
        y_true = y_data.reshape(-1, 1)
        y_pred = x * m + b

        # Compute MSE loss
        diff = y_pred - Tensor(y_true)
        loss = (diff ** 2).mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: loss = {loss.data.item():.6f}, "
                  f"m = {m.data.item():.4f}, b = {b.data.item():.4f}")

    print("-" * 40)
    print(f"\nFinal results:")
    print(f"  Learned:  m = {m.data.item():.4f}, b = {b.data.item():.4f}")
    print(f"  Expected: m = 2.0000, b = -1.0000")
    print(f"  MSE on true parameters: "
          f"{(m.data.item() - 2.0)**2 + (b.data.item() - (-1.0))**2:.6f}")


if __name__ == "__main__":
    train()
