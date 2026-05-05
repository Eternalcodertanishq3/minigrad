"""
01_scalar_autograd.py — Basic forward + backward with scalar values.

This example demonstrates the core autograd mechanism:
1. Create a computation graph manually
2. Perform forward pass
3. Call backward() to compute gradients via chain rule
4. Verify gradients match manual calculation

Run: python examples/01_scalar_autograd.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from minigrad import Tensor


def demo_basic_ops():
    """Show addition, multiplication, and their gradients."""
    print("=" * 60)
    print("Demo 1: Basic Operations")
    print("=" * 60)

    # Create tensors with gradient tracking
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)

    # Forward: c = a + b = 2 + 3 = 5
    c = a + b
    print(f"a = {a.data.item()}, b = {b.data.item()}")
    print(f"c = a + b = {c.data.item()}")

    # Backward: dc/da = 1, dc/db = 1
    c.backward()
    print(f"dc/da = {a.grad.item()} (should be 1.0)")
    print(f"dc/db = {b.grad.item()} (should be 1.0)")

    print()

    # Reset and try multiplication
    a.zero_grad()
    b.zero_grad()

    # Forward: d = a * b = 2 * 3 = 6
    d = a * b
    print(f"d = a * b = {d.data.item()}")

    # Backward: dd/da = b = 3, dd/db = a = 2
    d.backward()
    print(f"dd/da = {a.grad.item()} (should be 3.0)")
    print(f"dd/db = {b.grad.item()} (should be 2.0)")


def demo_chain_rule():
    """Demonstrate the chain rule with composite functions."""
    print("\n" + "=" * 60)
    print("Demo 2: Chain Rule — f(x) = (x + 2) * (x - 3)")
    print("=" * 60)

    # f(x) = (x + 2) * (x - 3) = x^2 - x - 6
    # df/dx = 2x - 1
    # At x = 4: f(4) = (6)*(1) = 6, df/dx = 2*4 - 1 = 7

    x = Tensor([4.0], requires_grad=True)
    u = x + 2        # u = x + 2 = 6
    v = x - 3        # v = x - 3 = 1
    f = u * v        # f = u * v = 6

    print(f"x = {x.data.item()}")
    print(f"u = x + 2 = {u.data.item()}")
    print(f"v = x - 3 = {v.data.item()}")
    print(f"f = u * v = {f.data.item()} (expected: 6.0)")

    f.backward()
    print(f"df/dx = {x.grad.item()} (expected: 7.0)")
    print(f"Manual check: 2*{x.data.item()} - 1 = {2 * x.data.item() - 1}")


def demo_polynomial():
    """Gradients of a polynomial: f(x) = 3x^2 + 2x + 1."""
    print("\n" + "=" * 60)
    print("Demo 3: Polynomial — f(x) = 3x^2 + 2x + 1")
    print("=" * 60)

    # f(x) = 3x^2 + 2x + 1
    # df/dx = 6x + 2
    # At x = 2: f(2) = 12 + 4 + 1 = 17, df/dx = 14

    x = Tensor([2.0], requires_grad=True)
    f = 3 * (x ** 2) + 2 * x + 1

    print(f"x = {x.data.item()}")
    print(f"f(x) = 3x^2 + 2x + 1 = {f.data.item()} (expected: 17.0)")

    f.backward()
    print(f"df/dx = {x.grad.item()} (expected: 14.0)")
    print(f"Manual check: 6*{x.data.item()} + 2 = {6 * x.data.item() + 2}")


def demo_relu():
    """Demonstrate ReLU and its gradient."""
    print("\n" + "=" * 60)
    print("Demo 4: ReLU Activation")
    print("=" * 60)

    x_pos = Tensor([3.0], requires_grad=True)
    x_neg = Tensor([-2.0], requires_grad=True)

    y_pos = x_pos.relu()
    y_neg = x_neg.relu()

    print(f"ReLU(3.0) = {y_pos.data.item()} (expected: 3.0)")
    print(f"ReLU(-2.0) = {y_neg.data.item()} (expected: 0.0)")

    y_pos.backward()
    y_neg.backward()

    print(f"dReLU(3.0)/dx = {x_pos.grad.item()} (expected: 1.0)")
    print(f"dReLU(-2.0)/dx = {x_neg.grad.item()} (expected: 0.0)")


def demo_computation_graph():
    """Visualize the computation graph."""
    print("\n" + "=" * 60)
    print("Demo 5: Computation Graph Visualization")
    print("=" * 60)

    x = Tensor([1.0], requires_grad=True)
    y = Tensor([2.0], requires_grad=True)
    z = Tensor([3.0], requires_grad=True)

    # f = (x + y) * z = (1+2)*3 = 9
    w = x + y      # w = 3
    f = w * z      # f = 9

    print("Graph:")
    from minigrad.graph import print_graph
    print_graph(f)

    f.backward()
    print(f"\nf = (x+y)*z = {f.data.item()}")
    print(f"df/dx = {x.grad.item()} (expected: 3.0)")
    print(f"df/dy = {y.grad.item()} (expected: 3.0)")
    print(f"df/dz = {z.grad.item()} (expected: 3.0)")


if __name__ == "__main__":
    print("\n")
    print("█" * 60)
    print("  miniGrad — Scalar Autograd Demonstration")
    print("█" * 60)
    print("\nThis example shows how miniGrad's autograd engine works")
    print("by computing gradients through various operations.\n")

    demo_basic_ops()
    demo_chain_rule()
    demo_polynomial()
    demo_relu()
    demo_computation_graph()

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
