"""
tests/test_dp_accountant.py — Verification suite for Rényi Differential Privacy (RDP) Accountant.

Validates:
1. Analytical Gaussian RDP formulation at q=1: alpha / (2 * sigma^2).
2. Subsampled Gaussian mechanism bounds (Wang, Balle, Kasiviswanathan 2019).
3. Monotonicity of privacy expenditure over optimization steps T.
4. Conversion to (epsilon, delta)-DP via optimal Rényi order minimization.
5. RDPAccountant stateful step accumulation and reset behavior.
6. Integration with compute_dp_sgd_step and PrivacyTelemetry report generation.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from minigrad.dp import (
    RDPAccountant,
    compute_dp_sgd_step,
    compute_rdp_epsilon,
    compute_step_rdp,
    get_privacy_spent,
)
from minigrad.tensor import Tensor


def test_step_rdp_full_batch():
    """At q=1.0, RDP(alpha) matches the exact analytical Gaussian formula: alpha / (2 * sigma^2)."""
    sigma = 2.5
    for alpha in [2, 3, 5, 10, 20, 32, 64]:
        rdp = compute_step_rdp(sample_rate=1.0, noise_multiplier=sigma, alpha=alpha)
        expected = float(alpha) / (2.0 * (sigma ** 2))
        assert math.isclose(rdp, expected, rel_tol=1e-7), f"Mismatch at alpha={alpha}: {rdp} vs {expected}"


def test_step_rdp_subsampled_bounds():
    """Subsampled Gaussian mechanism (q < 1) yields strictly lower privacy loss than full batch."""
    sigma = 2.0
    q = 0.05
    for alpha in [2, 4, 8, 16, 32]:
        full_rdp = compute_step_rdp(sample_rate=1.0, noise_multiplier=sigma, alpha=alpha)
        sub_rdp = compute_step_rdp(sample_rate=q, noise_multiplier=sigma, alpha=alpha)
        assert 0.0 < sub_rdp < full_rdp, f"Expected 0 < sub_rdp < full_rdp at alpha={alpha}, got {sub_rdp} vs {full_rdp}"


def test_step_rdp_invalid_inputs():
    """Invalid parameter values raise appropriate ValueErrors."""
    with pytest.raises(ValueError, match="alpha must be an integer >= 2"):
        compute_step_rdp(sample_rate=0.1, noise_multiplier=1.0, alpha=1)

    with pytest.raises(ValueError, match="noise_multiplier must be strictly positive"):
        compute_step_rdp(sample_rate=0.1, noise_multiplier=-0.5, alpha=2)

    with pytest.raises(ValueError, match="target_delta must be in"):
        get_privacy_spent(orders=[2, 4], rdp=[0.1, 0.2], target_delta=0.0)

    with pytest.raises(ValueError, match="target_delta must be in"):
        get_privacy_spent(orders=[2, 4], rdp=[0.1, 0.2], target_delta=1.5)


def test_rdp_composition_monotonicity():
    """Privacy expenditure epsilon(delta) increases monotonically with optimization steps T."""
    q = 0.01
    sigma = 1.5
    delta = 1e-5
    steps_list = [10, 50, 100, 500, 1000]
    eps_list = []

    for steps in steps_list:
        eps, opt_alpha = compute_rdp_epsilon(
            steps=steps, noise_multiplier=sigma, target_delta=delta, sample_rate=q
        )
        assert eps > 0.0
        assert opt_alpha >= 2
        eps_list.append(eps)

    for i in range(len(eps_list) - 1):
        assert eps_list[i] < eps_list[i + 1], f"Expected monotonic growth: {eps_list[i]} < {eps_list[i+1]}"


def test_rdp_accountant_class():
    """RDPAccountant statefully accumulates steps and converts to (epsilon, delta)-DP."""
    accountant = RDPAccountant()
    assert accountant.steps == 0
    assert len(accountant.orders) == 63  # 2 to 64

    # Step 100 times
    accountant.step(noise_multiplier=2.0, sample_rate=0.01, num_steps=100)
    assert accountant.steps == 100
    eps_100, alpha_100 = accountant.get_epsilon(target_delta=1e-5)
    assert eps_100 > 0.0

    # Step 100 more times
    accountant.step(noise_multiplier=2.0, sample_rate=0.01, num_steps=100)
    assert accountant.steps == 200
    eps_200, alpha_200 = accountant.get_epsilon(target_delta=1e-5)
    assert eps_200 > eps_100

    # Reset
    accountant.reset()
    assert accountant.steps == 0
    assert np.all(accountant._rdp == 0.0)


def test_compute_dp_sgd_step_with_accountant():
    """compute_dp_sgd_step seamlessly integrates with RDPAccountant and records telemetry."""
    accountant = RDPAccountant()
    batch_size = 16
    per_sample = [Tensor(np.random.randn(batch_size, 4)) for _ in range(2)]

    priv_grads, telemetry = compute_dp_sgd_step(
        per_sample_grads=per_sample,
        max_norm=1.0,
        noise_multiplier=1.5,
        accountant=accountant,
        sample_rate=0.02,
        target_delta=1e-5,
    )

    assert len(priv_grads) == 2
    assert telemetry.cumulative_steps == 1
    assert telemetry.sample_rate == 0.02
    assert telemetry.cumulative_epsilon is not None
    assert telemetry.cumulative_epsilon > 0.0
    assert telemetry.target_delta == 1e-5

    summary_text = telemetry.summary()
    assert "Cumulative Privacy:" in summary_text
    assert "Cumulative Steps:" in summary_text
    assert "eps =" in summary_text
