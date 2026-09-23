"""
11_vmap_and_dp_sgd.py — Pillar 4: Pure Functional vmap & Differential Privacy Engine.

This example showcases miniGrad's Pillar 4 innovations:
1. Pure Functional Vectorization (vmap): vectorizing functions without manual loops.
2. Per-Sample Gradient Extraction: obtaining individual sample gradients [B, *shape].
3. Mislabeled / Outlier Sample Localization: identifying bad data points via gradient norms.
4. Batched Jacobians (jacrev): single-pass batched Jacobian tensor calculation [B, M, N].
5. Differential Privacy (DP-SGD): per-sample L2 clipping, calibrated noise, and privacy telemetry.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import minigrad
from minigrad import (
    Tensor,
    vmap,
    make_functional,
    per_sample_gradients,
    jacrev,
    batched_jacobian,
    compute_dp_sgd_step,
    apply_dp_gradients,
)
from minigrad.nn import Linear, Sequential, ReLU
from minigrad.optim import Adam


def main():
    print("=" * 74)
    print(" miniGrad Pillar 4: Pure Functional vmap & DP-SGD Engine Demo")
    print("=" * 74)

    # --------------------------------------------------------------------------
    # Part 1: Core vmap Vectorization
    # --------------------------------------------------------------------------
    print("\n[1] Vectorizing Single-Example Functions via vmap()...")

    # Define a simple vector function operating on single 1D vectors
    def polynomial_features(x: Tensor) -> Tensor:
        # Returns [x, x^2, x^3]
        return minigrad.ops.stack([x, x ** 2, x ** 3], axis=-1)

    batch_x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]]))  # Shape (3, 2)
    v_poly = vmap(polynomial_features, in_axes=0)
    poly_out = v_poly(batch_x)

    print(f"  * Input Batch Shape:          {batch_x.shape}")
    print(f"  * Vectorized Output Shape:    {poly_out.shape} (Added feature dimension)")
    print(f"  * Sample 0 Features:\n{poly_out.data[0]}")

    # --------------------------------------------------------------------------
    # Part 2: Per-Sample Gradient Extraction & Outlier Detection
    # --------------------------------------------------------------------------
    print("\n[2] Per-Sample Gradient Extraction & Mislabeled Data Detection...")

    mlp = Sequential([
        Linear(4, 8),
        ReLU(),
        Linear(8, 1),
    ])

    np.random.seed(42)
    N = 6
    clean_x = np.random.randn(N, 4).astype(np.float64)
    clean_y = (clean_x[:, 0:1] * 2.0 - clean_x[:, 1:2] * 1.5).astype(np.float64)

    # Inject one poison/mislabeled outlier sample at index 3
    clean_y[3] = 999.0  # Massive label corruption!

    batch_x = Tensor(clean_x)
    batch_y = Tensor(clean_y)

    def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
        return ((pred - target) ** 2).sum()

    # Extract per-sample gradients in a single vectorized pass!
    sample_grads = mlp.per_sample_gradients(mse_loss, batch_x, batch_y)

    print(f"  * Extracted Gradient Shapes:")
    for name, g in sample_grads.items():
        print(f"    - {name:20s}: {g.shape} (Leading dim B={N})")

    # Compute per-sample gradient L2 norms across all parameters
    sample_norms = np.zeros(N)
    for g in sample_grads.values():
        sample_norms += np.sum(g.data.reshape(N, -1) ** 2, axis=1)
    sample_norms = np.sqrt(sample_norms)

    print("\n  * Per-Sample Gradient L2 Norms:")
    for i, norm in enumerate(sample_norms):
        flag = " [!] POISON / OUTLIER DETECTED" if i == 3 else " [OK]"
        print(f"    Sample #{i}: norm = {norm:10.2f}{flag}")

    outlier_idx = int(np.argmax(sample_norms))
    print(f"  => Correctly identified corrupt sample index: #{outlier_idx} (norm = {sample_norms[outlier_idx]:.2f})")

    # --------------------------------------------------------------------------
    # Part 3: Batched Jacobians (jacrev)
    # --------------------------------------------------------------------------
    print("\n[3] Single-Pass Batched Jacobian Tensors (jacrev)...")

    # Vector-valued dynamic system: f(x) = [sin(x0)*x1, x0^2 + exp(x1)]
    def dynamic_system(x: Tensor) -> Tensor:
        y0 = x[0].sin() * x[1]
        y1 = (x[0] ** 2) + x[1].exp()
        return minigrad.ops.stack([y0, y1])

    batch_coords = Tensor(np.array([
        [0.0, 1.0],
        [np.pi / 2.0, 0.0],
        [1.0, -1.0],
    ]))

    batched_J = batched_jacobian(dynamic_system, batch_coords)
    print(f"  * Batched Input Coordinates Shape: {batch_coords.shape}")
    print(f"  * Computed Batched Jacobian Shape:  {batched_J.shape} [B, M, N]")
    print(f"  * Jacobian at Sample 0 ([0, 1]):\n{batched_J.data[0]}")

    # --------------------------------------------------------------------------
    # Part 4: Differential Privacy (DP-SGD) Step & Telemetry
    # --------------------------------------------------------------------------
    print("\n[4] Differential Privacy (DP-SGD) with Calibrated Gaussian Noise...")

    clean_batch_y = (clean_x[:, 0:1] * 2.0 - clean_x[:, 1:2] * 1.5).astype(np.float64)
    clean_batch_y_tensor = Tensor(clean_batch_y)

    dp_grads_per_sample = mlp.per_sample_gradients(mse_loss, batch_x, clean_batch_y_tensor)

    # Execute DP-SGD step: clip sensitivity to C=2.0, add noise with multiplier 0.5
    private_grads, telemetry = compute_dp_sgd_step(
        dp_grads_per_sample,
        max_norm=2.0,
        noise_multiplier=0.5,
        seed=1337,
    )

    # Print the structured ASCII telemetry table
    print("\n" + telemetry.summary())

    # Apply private gradients and step optimizer
    optimizer = Adam(mlp.parameters(), lr=1e-3)
    apply_dp_gradients(mlp, private_grads)
    optimizer.step()
    print("  * Private gradient step successfully applied to model parameters!")

    print("\n" + "=" * 74)
    print(" miniGrad Pillar 4 Demonstration Successfully Completed!")
    print("=" * 74)


if __name__ == "__main__":
    main()
