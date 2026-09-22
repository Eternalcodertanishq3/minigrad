"""
08_glassbox_debugging.py — Glass-Box Autograd, First-NaN Localization & Visual DAG.

This example showcases Pillar 1 of miniGrad's innovations:
1. Gradient health telemetry and ASCII report (L2 norm, dynamic range, dead neuron %).
2. Interactive standalone HTML/SVG computational graph export (zero dependencies).
3. First-NaN / Inf root-cause localization with `detect_anomaly()` context manager.

Run: python examples/08_glassbox_debugging.py
"""
import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minigrad import (
    Tensor,
    detect_anomaly,
    explain_gradients,
    visualize,
    GradientAnomalyError,
)
from minigrad.nn import Linear, Sequential, ReLU


def demo_healthy_telemetry():
    """Demonstrate gradient health telemetry on a healthy MLP."""
    print("=" * 70)
    print("DEMO 1: Healthy MLP Gradient Health Telemetry")
    print("=" * 70)

    np.random.seed(42)
    model = Sequential([
        Linear(4, 8),
        ReLU(),
        Linear(8, 2),
    ])

    # Sample batch
    x = Tensor(np.random.randn(5, 4), requires_grad=False)
    target = Tensor(np.ones((5, 2)), requires_grad=False)

    # Forward
    pred = model(x)
    loss = ((pred - target) ** 2).mean()

    # Backward
    loss.backward()

    # Glass-Box explain
    report = loss.explain()
    print(report)


def demo_interactive_visual_dag():
    """Demonstrate standalone HTML/SVG DAG visualization export."""
    print("\n" + "=" * 70)
    print("DEMO 2: Zero-Dependency Interactive HTML/SVG Visual DAG")
    print("=" * 70)

    model = Sequential([
        Linear(2, 4),
        ReLU(),
        Linear(4, 1),
    ])

    x = Tensor([[1.0, -2.0]], requires_grad=False)
    pred = model(x)
    loss = (pred ** 2).sum()
    loss.backward()

    output_html = "computational_graph.html"
    loss.visualize(filename=output_html)

    print(f"Generated standalone computational graph: {output_html}")
    print("Open this file directly in any browser (Chrome, Edge, Firefox, Safari)!")
    print("Features included:")
    print("  * Smooth cubic Bezier curves connecting parent and child nodes")
    print("  * Color-coded gradient health (Green = Healthy, Yellow = Vanishing, Red = Exploding)")
    print("  * Interactive click inspector displaying exact tensor shapes, norms, and sample values")
    print("  * Smooth mouse drag-pan and scroll-wheel zoom controls")
    print("  * 100% pure Python & SVG - zero external binaries or CDN dependencies")


def demo_first_nan_root_cause_trapping():
    """Demonstrate trapping of gradient poisoning with root cause diagnosis."""
    print("\n" + "=" * 70)
    print("DEMO 3: Trapping First-NaN Root Cause with detect_anomaly()")
    print("=" * 70)

    # Simulate an operation that causes gradient explosion / singularity
    # x^0.5 at x=0 produces finite forward (0.0), but d/dx = 0.5 * 0^(-0.5) -> inf
    with np.errstate(divide="ignore"):
        x = Tensor([0.0, 1.0, 4.0], requires_grad=True)
        w = Tensor([2.0, 3.0, 1.0], requires_grad=True)

        y = (x ** 0.5) * w
        loss = y.sum()

        print("Executing backward pass inside `with detect_anomaly():`...")
        try:
            with detect_anomaly():
                loss.backward()
        except GradientAnomalyError as err:
            print("\nSuccessfully caught poisoned gradient!")
            print(err)


def main():
    demo_healthy_telemetry()
    demo_interactive_visual_dag()
    demo_first_nan_root_cause_trapping()


if __name__ == "__main__":
    main()
