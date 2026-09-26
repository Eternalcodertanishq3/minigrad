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
            "=" * 74,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PrivacyTelemetry(B={self.batch_size}, C={self.max_norm:.2f}, "
            f"noise_multiplier={self.noise_multiplier:.2f}, "
            f"clipped={self.fraction_clipped * 100.0:.1f}%)"
        )


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
) -> Tuple[Union[Dict[str, Tensor], List[Tensor]], PrivacyTelemetry]:
    """
    One-step Differentially Private aggregation:
    1. Clips per-sample gradients to `max_norm`.
    2. Adds calibrated Gaussian noise.
    3. Emits comprehensive privacy and gradient health telemetry.

    Args:
        per_sample_grads: Per-sample gradients [B, ...].
        max_norm:         Clipping bound C.
        noise_multiplier: Noise ratio sigma / C.
        seed:             Optional random seed.

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
    )

    return private_grads, telemetry
