"""
dp.py — Differential Privacy (DP-SGD) Engine & Telemetry for miniGrad (Pillar 4).

Implements the formal Gaussian Mechanism for Differential Privacy:
1. Per-Sample Gradient L2-Norm Bounding (Clipping):
   g_i <- g_i * min(1, C / ||g_i||_2)
2. Calibrated Gaussian Noise Injection:
   g_private = (1/B) * (sum(g_i) + N(0, sigma^2 * I)), where sigma = C * noise_multiplier
3. Privacy Telemetry & Diagnostic Accounting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from minigrad.nn.module import Module
from minigrad.tensor import Tensor


@dataclass
class PrivacyTelemetry:
    """Diagnostic statistics for a Differentially Private SGD step."""
    batch_size: int
    max_norm: float
    noise_multiplier: float
    noise_std: float
    mean_pre_clip_norm: float
    max_pre_clip_norm: float
    min_pre_clip_norm: float
    fraction_clipped: float
    noise_to_signal_ratio: float
    cumulative_steps: Optional[int] = None
    sample_rate: Optional[float] = None
    cumulative_epsilon: Optional[float] = None
    target_delta: Optional[float] = None
    optimal_alpha: Optional[int] = None

    def summary(self) -> str:
        """Render a clean, terminal-safe ASCII report of privacy parameters and gradient telemetry."""
        header = "miniGrad Differential Privacy (DP-SGD) Telemetry (Pillar 4)"
        lines = [
            "=" * 74,
            f"{header:^74}",
            "=" * 74,
            f"  * Batch Size (B):              {self.batch_size} samples",
            f"  * Max L2 Norm Bound (C):       {self.max_norm:.4f}",
            f"  * Noise Multiplier (sigma/C):  {self.noise_multiplier:.4f}",
            f"  * Effective Noise Scale:       {self.noise_std:.6e}",
            f"  * Pre-Clip Norm (Mean):        {self.mean_pre_clip_norm:.4f}",
            f"  * Pre-Clip Norm (Min / Max):   {self.min_pre_clip_norm:.4f} / {self.max_pre_clip_norm:.4f}",
            f"  * Fraction of Samples Clipped: {self.fraction_clipped * 100.0:.1f}%",
            f"  * Noise-to-Signal Ratio:       {self.noise_to_signal_ratio:.4f}",
            f"  * Privacy Sensitivity:         BOUNDED (||Delta||_2 <= {self.max_norm:.4f})",
        ]
        if self.cumulative_epsilon is not None and self.target_delta is not None:
            lines.append(
                f"  * Cumulative Privacy:          eps = {self.cumulative_epsilon:.4f} @ delta = {self.target_delta:.1e} (alpha={self.optimal_alpha})"
            )
            if self.cumulative_steps is not None:
                lines.append(f"  * Cumulative Steps:            {self.cumulative_steps}")
        lines.append("=" * 74)
        return "\n".join(lines)

    def __repr__(self) -> str:
        base = (
            f"PrivacyTelemetry(B={self.batch_size}, C={self.max_norm:.2f}, "
            f"noise_multiplier={self.noise_multiplier:.2f}, "
            f"clipped={self.fraction_clipped * 100.0:.1f}%)"
        )
        if self.cumulative_epsilon is not None:
            base = base[:-1] + f", eps={self.cumulative_epsilon:.2f})"
        return base


def clip_per_sample_gradients(
    per_sample_grads: Union[Dict[str, Tensor], Sequence[Tensor]],
    max_norm: float,
) -> Tuple[Union[Dict[str, Tensor], List[Tensor]], np.ndarray]:
    """
    Clips the per-sample gradients to bound the L2-norm of each sample's gradient to `max_norm`.

    Args:
        per_sample_grads: Dict {name: Tensor[B, ...]} or Sequence of Tensor[B, ...].
        max_norm:         Maximum allowed L2 norm C > 0.

    Returns:
        Tuple of (clipped_grads, pre_clip_norms).
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be strictly positive, got {max_norm}")

    is_dict = isinstance(per_sample_grads, dict)
    if isinstance(per_sample_grads, dict):
        keys = list(per_sample_grads.keys())
        grad_list = [per_sample_grads[k] for k in keys]
    else:
        keys = []
        grad_list = list(per_sample_grads)

    if not grad_list:
        return per_sample_grads, np.array([])  # type: ignore[return-value]

    batch_size = grad_list[0].shape[0]

    # Compute per-sample L2 norm squared across all parameters
    sample_norm_sq = np.zeros(batch_size, dtype=np.float64)
    for g in grad_list:
        g_data = g.data if isinstance(g, Tensor) else g
        # Flatten all dimensions except batch: [B, D]
        g_flat = g_data.reshape(batch_size, -1)
        sample_norm_sq += np.sum(g_flat ** 2, axis=1)

    sample_norms = np.sqrt(sample_norm_sq)

    # Compute per-sample clipping factors: min(1.0, C / (norm + 1e-12))
    clip_factors = np.minimum(1.0, max_norm / (sample_norms + 1e-12))

    # Apply clipping factors
    clipped_list: List[Tensor] = []
    for g in grad_list:
        g_data = g.data if isinstance(g, Tensor) else g
        # Reshape clip_factors for broadcasting: [B, 1, 1, ...]
        broadcast_shape = [batch_size] + [1] * (g_data.ndim - 1)
        scale = clip_factors.reshape(broadcast_shape)
        clipped_data = g_data * scale
        clipped_list.append(Tensor(clipped_data, requires_grad=False))

    if is_dict:
        return dict(zip(keys, clipped_list)), sample_norms
    return clipped_list, sample_norms


def add_dp_noise(
    clipped_grads: Union[Dict[str, Tensor], Sequence[Tensor]],
    noise_multiplier: float,
    max_norm: float,
    batch_size: int,
    seed: Optional[int] = None,
) -> Union[Dict[str, Tensor], List[Tensor]]:
    """
    Averages clipped gradients across the batch and adds calibrated Gaussian noise.

    Noise std: sigma = (max_norm * noise_multiplier) / batch_size.

    Args:
        clipped_grads:    Clipped per-sample gradients with leading dimension B.
        noise_multiplier: Ratio sigma / C.
        max_norm:         Clipping bound C.
        batch_size:       Number of samples in batch.
        seed:             Optional random seed for reproducibility.

    Returns:
        Private aggregated gradient tensors with shapes matching parameter tensors.
    """
    if noise_multiplier < 0:
        raise ValueError(f"noise_multiplier cannot be negative, got {noise_multiplier}")

    rng = np.random.default_rng(seed)
    noise_std = (max_norm * noise_multiplier) / float(batch_size) if noise_multiplier > 0 else 0.0

    is_dict = isinstance(clipped_grads, dict)
    if isinstance(clipped_grads, dict):
        items = list(clipped_grads.items())
    else:
        items = [(str(i), g) for i, g in enumerate(clipped_grads)]

    private_dict: Dict[str, Tensor] = {}
    for name, g in items:
        g_data = g.data if isinstance(g, Tensor) else g
        # Average over batch dimension (axis 0)
        mean_grad = np.mean(g_data, axis=0)

        # Add Gaussian noise
        if noise_std > 0:
            noise = rng.normal(loc=0.0, scale=noise_std, size=mean_grad.shape)
            priv_data = mean_grad + noise
        else:
            priv_data = mean_grad

        private_dict[name] = Tensor(priv_data, requires_grad=False)

    if is_dict:
        return private_dict
    return [private_dict[str(i)] for i in range(len(clipped_grads))]


def apply_dp_gradients(
    model: Module,
    private_grads: Union[Dict[str, Tensor], Sequence[Tensor]],
) -> None:
    """
    Assigns differentially private gradients directly to model parameter `.grad` attributes,
    enabling native compatibility with miniGrad standard optimizers (SGD, Adam, RMSprop).
    """
    if isinstance(private_grads, dict):
        for name, param in model.named_parameters():
            if name in private_grads:
                g = private_grads[name]
                param.grad = g.data.copy() if isinstance(g, Tensor) else g.copy()
    else:
        for param, g in zip(model.parameters(), private_grads):
            param.grad = g.data.copy() if isinstance(g, Tensor) else g.copy()


def compute_dp_sgd_step(
    per_sample_grads: Union[Dict[str, Tensor], Sequence[Tensor]],
    max_norm: float,
    noise_multiplier: float,
    seed: Optional[int] = None,
    accountant: Optional[RDPAccountant] = None,
    sample_rate: Optional[float] = None,
    target_delta: Optional[float] = None,
) -> Tuple[Union[Dict[str, Tensor], List[Tensor]], PrivacyTelemetry]:
    """
    One-step Differentially Private aggregation:
    1. Clips per-sample gradients to `max_norm`.
    2. Adds calibrated Gaussian noise.
    3. Emits comprehensive privacy and gradient health telemetry.
    4. Optionally tracks cumulative Rényi Differential Privacy (RDP) expenditure.

    Args:
        per_sample_grads: Per-sample gradients [B, ...].
        max_norm:         Clipping bound C.
        noise_multiplier: Noise ratio sigma / C.
        seed:             Optional random seed.
        accountant:       Optional RDPAccountant instance to accumulate privacy loss.
        sample_rate:      Subsampling ratio q = batch_size / total_samples (default: B / B = 1.0).
        target_delta:     Target delta for cumulative epsilon conversion.

    Returns:
        Tuple of (private_grads, PrivacyTelemetry).
    """
    # 1. Clip per-sample gradients
    clipped_grads, pre_clip_norms = clip_per_sample_gradients(per_sample_grads, max_norm=max_norm)

    batch_size = len(pre_clip_norms)
    num_clipped = int(np.sum(pre_clip_norms > max_norm))
    fraction_clipped = float(num_clipped / batch_size) if batch_size > 0 else 0.0

    noise_std = (max_norm * noise_multiplier) / float(batch_size) if batch_size > 0 else 0.0

    # 2. Add calibrated Gaussian noise
    private_grads = add_dp_noise(
        clipped_grads,
        noise_multiplier=noise_multiplier,
        max_norm=max_norm,
        batch_size=batch_size,
        seed=seed,
    )

    # 3. Compute signal vs noise telemetry
    signal_norm = 0.0
    if isinstance(private_grads, dict):
        for g in private_grads.values():
            signal_norm += float(np.sum((g.data if isinstance(g, Tensor) else g) ** 2))
    else:
        for g in private_grads:
            signal_norm += float(np.sum((g.data if isinstance(g, Tensor) else g) ** 2))
    signal_norm = float(np.sqrt(signal_norm))

    noise_to_signal = float(noise_std / (signal_norm + 1e-12))

    # 4. Optional RDP Accounting
    cum_eps: Optional[float] = None
    opt_alpha: Optional[int] = None
    cum_steps: Optional[int] = None
    q = sample_rate if sample_rate is not None else 1.0

    if accountant is not None:
        if noise_multiplier > 0:
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=q, num_steps=1)
        cum_steps = accountant.steps
        if target_delta is not None:
            cum_eps, opt_alpha = accountant.get_epsilon(target_delta=target_delta)

    telemetry = PrivacyTelemetry(
        batch_size=batch_size,
        max_norm=max_norm,
        noise_multiplier=noise_multiplier,
        noise_std=noise_std,
        mean_pre_clip_norm=float(np.mean(pre_clip_norms)) if batch_size > 0 else 0.0,
        max_pre_clip_norm=float(np.max(pre_clip_norms)) if batch_size > 0 else 0.0,
        min_pre_clip_norm=float(np.min(pre_clip_norms)) if batch_size > 0 else 0.0,
        fraction_clipped=fraction_clipped,
        noise_to_signal_ratio=noise_to_signal,
        cumulative_steps=cum_steps,
        sample_rate=q,
        cumulative_epsilon=cum_eps,
        target_delta=target_delta,
        optimal_alpha=opt_alpha,
    )

    return private_grads, telemetry


# ── Analytical Rényi Differential Privacy (RDP) Upper-Bound Accountant ──────────

def compute_step_rdp(sample_rate: float, noise_multiplier: float, alpha: int) -> float:
    """
    Computes the analytical Rényi Differential Privacy (RDP) upper bound at integer order `alpha`
    for a single step of a subsampled Gaussian mechanism.

    Formulation & Literature:
    - Full-batch Gaussian (q=1): RDP(alpha) = alpha / (2 * sigma^2) (Mironov 2017).
    - Subsampled Gaussian (0 < q < 1):
      S(alpha) = sum_{k=0}^alpha binom(alpha, k) * q^k * (1-q)^(alpha-k) * exp(k*(k-1) / (2*sigma^2))
      RDP(alpha) <= log(S(alpha)) / (alpha - 1) (Wang, Balle, Kasiviswanathan 2019, Theorem 11).

    Assumptions & Scope:
    - Sampling Model: Assumes Poisson subsampling (or uniform subsampling with replacement) at rate q = B / N.
    - Adjacency: Standard add/remove one example adjacency model.
    - Caller Responsibility: The caller is responsible for ensuring that training batches and dataset sampling
      adhere to the documented subsampling model.
    - Evaluated via numerically stable log-sum-exp to eliminate floating overflow.
    """
    if alpha < 2:
        raise ValueError(f"alpha must be an integer >= 2, got {alpha}")
    if noise_multiplier <= 0:
        raise ValueError(f"noise_multiplier must be strictly positive, got {noise_multiplier}")
    if sample_rate <= 0.0:
        return 0.0
    if sample_rate >= 1.0:
        return float(alpha) / (2.0 * (noise_multiplier ** 2))

    q = float(sample_rate)
    sigma2 = float(noise_multiplier ** 2)

    # Compute sum in log-space: log_term_k = log(binom) + k*log(q) + (alpha-k)*log(1-q) + k*(k-1)/(2*sigma^2)
    log_terms = []
    for k in range(alpha + 1):
        log_comb = math.log(math.comb(alpha, k))
        log_qk = k * math.log(q) if k > 0 else 0.0
        log_1_minus_q = (alpha - k) * math.log(1.0 - q) if (alpha - k) > 0 else 0.0
        log_exp_term = (k * (k - 1)) / (2.0 * sigma2) if k >= 2 else 0.0
        log_terms.append(log_comb + log_qk + log_1_minus_q + log_exp_term)

    max_log = max(log_terms)
    log_sum = max_log + math.log(sum(math.exp(lt - max_log) for lt in log_terms))
    return float(log_sum / (alpha - 1.0))


def compute_rdp(
    sample_rate: float,
    noise_multiplier: float,
    steps: int,
    orders: Sequence[int],
) -> np.ndarray:
    """
    Computes cumulative RDP across a sequence of orders for `steps` iterations.

    Under RDP composition, privacy loss is strictly additive:
    RDP_total(alpha) = steps * RDP_step(alpha).
    """
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")
    step_rdps = [compute_step_rdp(sample_rate, noise_multiplier, int(a)) for a in orders]
    return np.array(step_rdps, dtype=np.float64) * steps


def get_privacy_spent(
    orders: Sequence[int],
    rdp: Union[Sequence[float], np.ndarray],
    target_delta: float,
) -> Tuple[float, int]:
    """
    Converts accumulated RDP into an (epsilon, delta)-differential privacy guarantee
    by minimizing over all evaluated orders:
    epsilon(delta) = min_{alpha > 1} ( RDP(alpha) + log(1/delta) / (alpha - 1) )

    Args:
        orders:       Sequence of integer orders alpha >= 2.
        rdp:          Accumulated RDP values corresponding to orders.
        target_delta: Target delta in (0, 1).

    Returns:
        Tuple of (optimal_epsilon, optimal_alpha).
    """
    if not (0.0 < target_delta < 1.0):
        raise ValueError(f"target_delta must be in (0, 1), got {target_delta}")
    if len(orders) != len(rdp):
        raise ValueError(f"orders ({len(orders)}) and rdp ({len(rdp)}) lengths must match")

    eps_candidates = [
        float(rdp[i] + math.log(1.0 / target_delta) / (int(orders[i]) - 1.0))
        for i in range(len(orders))
    ]
    min_idx = int(np.argmin(eps_candidates))
    return float(eps_candidates[min_idx]), int(orders[min_idx])


def compute_rdp_epsilon(
    steps: int,
    noise_multiplier: float,
    target_delta: float,
    sample_rate: float = 1.0,
    orders: Optional[Sequence[int]] = None,
) -> Tuple[float, int]:
    """
    High-level convenience function computing optimal (epsilon, delta)-DP bound
    for DP-SGD training under a subsampled Gaussian mechanism.

    Args:
        steps:            Total optimization iterations.
        noise_multiplier: Ratio sigma / C.
        target_delta:     Target delta (e.g. 1e-5).
        sample_rate:      Batch subsampling ratio q = B / N (default: 1.0).
        orders:           Optional sequence of orders alpha >= 2 (default: 2..64).

    Returns:
        Tuple of (optimal_epsilon, optimal_alpha).
    """
    evaluated_orders = list(range(2, 65)) if orders is None else [int(a) for a in orders if a >= 2]
    rdp = compute_rdp(sample_rate, noise_multiplier, steps, evaluated_orders)
    return get_privacy_spent(evaluated_orders, rdp, target_delta)


class RDPAccountant:
    """
    Rényi Differential Privacy (RDP) Accountant for Subsampled Gaussian Mechanisms.

    Implements analytical RDP composition and conversion to (epsilon, delta)-DP based on:
    - Mironov (2017): "Rényi Differential Privacy"
    - Wang, Balle, Kasiviswanathan (2019): "Subsampled Rényi Differential Privacy
      of Gaussian Mechanism and its Application to Deep Learning"

    Tracks cumulative privacy expenditure across discrete training iterations with
    analytical bounds over integer Rényi orders alpha in [2, 64].
    """

    def __init__(self, orders: Optional[Sequence[int]] = None) -> None:
        if orders is None:
            self.orders = list(range(2, 65))
        else:
            self.orders = sorted([int(a) for a in orders if a >= 2])
            if not self.orders:
                raise ValueError("orders must contain at least one integer >= 2")
        self._rdp = np.zeros(len(self.orders), dtype=np.float64)
        self.steps = 0

    def step(self, noise_multiplier: float, sample_rate: float, num_steps: int = 1) -> None:
        """
        Record one or more training steps under a subsampled Gaussian mechanism.

        Args:
            noise_multiplier: Ratio sigma / C (must be > 0).
            sample_rate:      Subsampling ratio q = batch_size / total_samples (0 <= q <= 1).
            num_steps:        Number of steps to accumulate (default: 1).
        """
        if noise_multiplier <= 0:
            raise ValueError(f"noise_multiplier must be strictly positive, got {noise_multiplier}")
        if not (0.0 <= sample_rate <= 1.0):
            raise ValueError(f"sample_rate must be in [0, 1], got {sample_rate}")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        step_rdps = [compute_step_rdp(sample_rate, noise_multiplier, a) for a in self.orders]
        self._rdp += np.array(step_rdps, dtype=np.float64) * num_steps
        self.steps += num_steps

    def get_epsilon(self, target_delta: float) -> Tuple[float, int]:
        """
        Computes the minimum cumulative epsilon for a given target delta:
        epsilon(delta) = min_{alpha > 1} ( RDP(alpha) + log(1/delta) / (alpha - 1) )

        Args:
            target_delta: Target failure probability delta in (0, 1).

        Returns:
            Tuple of (optimal_epsilon, optimal_alpha).
        """
        return get_privacy_spent(self.orders, self._rdp, target_delta)

    def reset(self) -> None:
        """Reset the accumulated privacy budget and step count."""
        self._rdp.fill(0.0)
        self.steps = 0

