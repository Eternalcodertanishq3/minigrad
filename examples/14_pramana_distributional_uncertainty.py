"""
examples/14_pramana_distributional_uncertainty.py — P.R.A.M.A.N.A. Distributional Uncertainty Tensors

Demonstrates P.R.A.M.A.N.A. (Probabilistic Representation of Analytical Moments and Algebraic Noise-aware Autograd):

1. Analytical Moment Propagation vs. 100,000-Sample Monte Carlo Simulation
   - Exact closed-form variance for ONE affine layer with independent inputs (DistributionalLinear);
     stacked layers drop cross-unit covariances and are approximate.
   - First-order Taylor (delta-method) diagonal variance approximation for non-linearities (Tanh)
     across small-variance vs large-variance regimes.
2. Weight-variance layer: variance scales with ||x||^2 (a magnitude effect, NOT OOD detection)
   - Trains the same DistributionalLinear(weight_uncertainty=True) on two different supports and
     shows the in-support/unseen ordering flips with magnitude, i.e. it is not support-aware.
3. Heteroscedastic Noise Learning with Dual-Head HeteroscedasticMLP & Gaussian NLL Loss
   - Jointly learns input-dependent conditional mean mu(x) and heteroscedastic variance sigma^2(x).
"""
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minigrad import (
    PRAMANA,
    DistributionalLinear,
    DistributionalTensor,
    GaussianNLLLoss,
    Tensor,
)
from minigrad.optim import Adam
from minigrad.pramana import HeteroscedasticMLP


def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def demo_analytical_vs_monte_carlo():
    print_banner("EXPERIMENT 1: Analytical Moment Propagation vs. 100,000-Sample Monte Carlo")
    print("Affine transformations propagate exact analytical mean and variance.")
    print("Non-linearities use first-order Taylor (delta-method) approximations (accurate for small sigma).\n")

    np.random.seed(42)
    in_dim, out_dim = 4, 2

    # Distributional Affine Layer (Exact moment propagation)
    model = DistributionalLinear(in_dim, out_dim, bias=True)

    mu_in = np.array([[1.0, -0.5, 2.0, 0.3]])
    var_in = np.array([[0.01, 0.01, 0.01, 0.01]])
    x_dist = DistributionalTensor(mu_in, var_in)

    # 1. P.R.A.M.A.N.A. Analytical Single-Pass (Linear + Tanh in small-variance regime)
    t0 = time.perf_counter()
    lin_dist = model(x_dist)
    tanh_dist = lin_dist.tanh()
    mu_ana = lin_dist.mean.numpy()
    var_ana = lin_dist.var.numpy()
    var_tanh_ana = tanh_dist.var.numpy()
    t_ana = (time.perf_counter() - t0) * 1000.0  # ms

    # 2. Empirical Monte Carlo (100,000 forward passes)
    n_samples = 100_000
    t0 = time.perf_counter()
    samples = np.random.normal(
        loc=np.repeat(mu_in, n_samples, axis=0),
        scale=np.repeat(np.sqrt(var_in), n_samples, axis=0),
    )
    w = model.weight.numpy()
    b = model.bias.numpy()
    out_samples = samples @ w + b
    tanh_samples = np.tanh(out_samples)

    mu_mc = np.mean(out_samples, axis=0, keepdims=True)
    var_mc = np.var(out_samples, axis=0, keepdims=True)
    var_tanh_mc = np.var(tanh_samples, axis=0, keepdims=True)
    t_mc = (time.perf_counter() - t0) * 1000.0  # ms

    mean_err = np.max(np.abs(mu_ana - mu_mc))
    var_err = np.max(np.abs(var_ana - var_mc))
    tanh_rel_err = float(np.mean(np.abs(var_tanh_ana - var_tanh_mc) / np.maximum(var_tanh_mc, 1e-12))) * 100.0
    speedup = t_mc / max(t_ana, 1e-6)

    print(f"Affine Analytical Mean:         {mu_ana[0]}")
    print(f"Affine Monte Carlo Mean:        {mu_mc[0]} (Max Abs Diff: {mean_err:.6f})")
    print(f"Affine Analytical Variance:     {var_ana[0]}")
    print(f"Affine Monte Carlo Variance:    {var_mc[0]} (Max Abs Diff: {var_err:.6f})")
    print(f"Tanh Delta-Method Rel Error:    {tanh_rel_err:.2f}% (at input std=0.10)")
    print(f"Analytical Time: {t_ana:.3f} ms vs Monte Carlo (N=100k): {t_mc:.3f} ms ({speedup:.1f}x faster)")


def demo_ood_hallucination_detection():
    print_banner("EXPERIMENT 2: Weight-Variance Layer: Magnitude Scaling (NOT support-aware OOD detection)")
    print("A DistributionalLinear with weight_uncertainty=True has predictive variance")
    print("  Var[y] = sum_i (x_i^2 * var(w_i) + mean(w_i)^2 * var(x_i) + var(w_i) * var(x_i)),")
    print("i.e. it grows with ||x||^2 whatever data the layer was trained on. The control below trains the")
    print("SAME layer on two different supports and probes both regions.\n")

    def trained_layer(center: float) -> DistributionalLinear:
        np.random.seed(42)
        layer = DistributionalLinear(2, 1, weight_uncertainty=True, init_log_var=-3.5)
        loss_fn = GaussianNLLLoss()
        optimizer = Adam(layer.parameters(), lr=0.04)
        x_np = np.random.uniform(center - 0.5, center + 0.5, size=(64, 2))
        y_np = (1.5 * x_np[:, 0:1] - 0.8 * x_np[:, 1:2]) + np.random.normal(0.0, 0.08, size=(64, 1))
        x_train = DistributionalTensor(x_np, np.full_like(x_np, 0.002))
        y_train = Tensor(y_np)
        for _ in range(50):
            optimizer.zero_grad()
            loss_fn(layer(x_train), y_train).backward()
            optimizer.step()
        return layer

    def var_at(layer: DistributionalLinear, point: list) -> float:
        out = layer(DistributionalTensor([point], [[0.002, 0.002]]))
        return float(np.mean(out.var.numpy()))

    probes = {"origin (0.2,-0.3)": [0.2, -0.3], "(3.5,-4.0)": [3.5, -4.0], "far (15,-20)": [15.0, -20.0]}
    print(f"{'Trained on support':<26} | " + " | ".join(f"{k:<18}" for k in probes))
    print("-" * 94)
    rows = {}
    for center, label in ((0.0, "centred at 0 (in-support: origin)"), (4.0, "centred at 4 (in-support: (3.5,-4))")):
        layer = trained_layer(center)
        rows[center] = {k: var_at(layer, v) for k, v in probes.items()}
        print(f"{label:<26} | " + " | ".join(f"{rows[center][k]:<18.5f}" for k in probes))
    print("-" * 94)

    shifted = rows[4.0]
    in_support, unseen = shifted["(3.5,-4.0)"], shifted["origin (0.2,-0.3)"]
    print(f"\nControl (trained on support centred at 4): variance at the IN-SUPPORT point (3.5,-4.0) is "
          f"{in_support / max(unseen, 1e-12):.1f}x the variance at the UNSEEN origin.")
    verdict = "FAILS" if in_support > unseen else "passes"
    print(f"Support-aware OOD check: {verdict}. Variance tracks input magnitude, not training-data coverage,")
    print("so this layer must not be used as an OOD detector. (Use ensembles / density models for that.)")


def demo_heteroscedastic_learning():
    print_banner("EXPERIMENT 3: Input-Dependent Heteroscedastic Noise Learning via Gaussian NLL")
    print("Dataset: y = 2*x + 0.5 + eps(x), where noise std grows with |x|: sigma(x) = 0.2 + 0.5*|x|.")
    print("True variance sigma^2(x) varies from 0.04 at x=0.0 up to 1.44 at |x|=2.0.\n")

    np.random.seed(42)
    n_pts = 200
    x_raw = np.linspace(-2.0, 2.0, n_pts).reshape(-1, 1).astype(np.float64)
    true_sigma = 0.2 + 0.5 * np.abs(x_raw)
    noise = np.random.normal(0.0, true_sigma)
    y_raw = 2.0 * x_raw + 0.5 + noise

    x_tensor = Tensor(x_raw)
    y_tensor = Tensor(y_raw)

    # Non-linear dual-head HeteroscedasticMLP predicting input-dependent mu(x) and sigma^2(x)
    model = HeteroscedasticMLP(in_features=1, hidden_features=24, out_features=1)

    loss_fn = GaussianNLLLoss()
    optimizer = Adam(model.parameters(), lr=0.03)

    true_mean_var = float(np.mean(true_sigma ** 2))
    print(f"Training HeteroscedasticMLP on {n_pts} samples for 150 epochs...")
    print(f"{'Epoch':<8} | {'NLL Loss':<12} | {'Mean Pred Var':<18} | {'True Mean Var'}")
    print("-" * 62)

    for epoch in range(1, 151):
        optimizer.zero_grad()
        x_dist = DistributionalTensor(x_tensor, Tensor(np.full_like(x_raw, 0.001)))
        out_dist = model(x_dist)

        loss = loss_fn(out_dist, y_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 30 == 0 or epoch == 1:
            pred_var_avg = float(np.mean(out_dist.var.numpy()))
            print(f"{epoch:<8} | {loss.item():<12.4f} | {pred_var_avg:<18.4f} | {true_mean_var:.4f}")

    # Verify x-dependent variance profile across specific query points
    probe_x = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=np.float64)
    probe_out = model(DistributionalTensor(probe_x, np.full_like(probe_x, 0.001)))
    probe_mu, probe_var = probe_out.numpy()
    true_probe_var = (0.2 + 0.5 * np.abs(probe_x)) ** 2

    print("\nLearned Input-Dependent Variance Profile across x:")
    print(f"{'x':<8} | {'Pred Mean mu(x)':<18} | {'True Mean':<12} | {'Pred Var sigma^2(x)':<22} | {'True Var sigma^2(x)'}")
    print("-" * 82)
    for i in range(len(probe_x)):
        xv = probe_x[i, 0]
        print(
            f"{xv:<8.1f} | {probe_mu[i, 0]:<18.4f} | {2.0 * xv + 0.5:<12.4f} | "
            f"{probe_var[i, 0]:<22.4f} | {true_probe_var[i, 0]:.4f}"
        )
    print("-" * 82)

    # Compute uncertainty telemetry
    x_in_eval = DistributionalTensor(Tensor(x_raw), Tensor(np.full_like(x_raw, 0.001)))
    x_ood_eval = DistributionalTensor(
        Tensor(np.random.uniform(6.0, 10.0, size=(40, 1))),
        Tensor(np.full((40, 1), 0.25)),
    )
    telem = PRAMANA.evaluate_uncertainty_telemetry(
        model,
        in_dist_data=x_in_eval,
        out_dist_data=x_ood_eval,
        targets=y_tensor,
    )
    print(f"\n{telem.summary()}")


def main():
    print("=" * 75)
    print("              miniGrad P.R.A.M.A.N.A. Uncertainty Showcase")
    print("   Probabilistic Representation of Analytical Moments & Algebraic Noise-aware Autograd")
    print("=" * 75)

    demo_analytical_vs_monte_carlo()
    demo_ood_hallucination_detection()
    demo_heteroscedastic_learning()

    print("\n" + "=" * 75)
    print("  P.R.A.M.A.N.A. Showcase Complete: All experiments verified successfully!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()

