"""
pramana.py — P.R.A.M.A.N.A. (Probabilistic Representation of Analytical Moments and Algebraic Noise-aware Autograd)

Distributional Uncertainty Tensors with closed-form propagation of epistemic
and aleatoric confidence in pure NumPy/Tensor.

Features:
- DistributionalTensor: Dual-stream tensor tracking mean and variance moments (X ~ N(mu, sigma^2)).
- Closed-Form Uncertainty Calculus: Analytical Taylor moment propagation through linear
  transformations, products, and non-linearities (Tanh, Sigmoid, ReLU, GELU).
- DistributionalLinear: Dense layer propagating analytical variance in a single forward pass.
- DistributionalSequential: Container for uncertainty-aware neural networks.
- GaussianNLLLoss: Heteroscedastic negative log-likelihood loss for joint mean-variance optimization.
- PramanaTelemetry: Epistemic calibration metrics and out-of-distribution hallucination detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from minigrad.graph import no_grad
from minigrad.nn.module import Module
from minigrad.tensor import Tensor


# ── Telemetry Dataclass ──────────────────────────────────────────────

@dataclass
class PramanaTelemetry:
    """
    Diagnostic telemetry recording uncertainty calibration, confidence bounds,
    and out-of-distribution (OOD) hallucination detection metrics.
    """
    in_dist_mean_std: float
    out_dist_mean_std: float
    epistemic_divergence_ratio: float
    fraction_in_95_ci: float
    total_samples_evaluated: int

    def summary(self) -> str:
        ci_pct = self.fraction_in_95_ci * 100.0
        return (
            f"P.R.A.M.A.N.A. Uncertainty Telemetry:\n"
            f"  In-Distribution Mean Std (sigma):  {self.in_dist_mean_std:.4f}\n"
            f"  Out-of-Distribution Std (sigma):  {self.out_dist_mean_std:.4f}\n"
            f"  Epistemic Divergence Ratio:        {self.epistemic_divergence_ratio:.2f}x (OOD Uncertainty Spike)\n"
            f"  95% CI Calibration Coverage:       {ci_pct:.1f}%\n"
            f"  Samples Evaluated:                 {self.total_samples_evaluated}"
        )


# ── Distributional Tensor Core ───────────────────────────────────────

class DistributionalTensor:
    """
    A Probability Distribution Tensor X ~ N(mu, sigma^2).

    Simultaneously tracks and propagates the first moment (mean) and second
    central moment (variance) across algebraic operations and non-linearities.

    Attributes:
        mean: Tensor representing expectation E[X]
        var:  Tensor representing variance Var[X] >= 0
    """

    __slots__ = ("mean", "var")

    def __init__(
        self,
        mean: Union[Tensor, np.ndarray, float, int, Sequence],
        var: Optional[Union[Tensor, np.ndarray, float, int, Sequence]] = None,
        requires_grad: bool = False,
    ) -> None:
        if isinstance(mean, Tensor):
            self.mean = mean
        else:
            self.mean = Tensor(np.asarray(mean, dtype=np.float64), requires_grad=requires_grad)

        if var is None:
            self.var = Tensor(np.zeros_like(self.mean.data), requires_grad=requires_grad)
        elif isinstance(var, Tensor):
            self.var = var
        else:
            self.var = Tensor(np.asarray(var, dtype=np.float64), requires_grad=requires_grad)

    @property
    def std(self) -> Tensor:
        """Standard deviation sigma = sqrt(max(var, 0))."""
        clamped_var = self.var.relu() + 1e-12
        return clamped_var ** 0.5

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.mean.shape

    @property
    def ndim(self) -> int:
        return self.mean.ndim

    def __len__(self) -> int:
        return len(self.mean)

    def __getitem__(self, idx: Any) -> DistributionalTensor:
        return DistributionalTensor(self.mean[idx], self.var[idx])

    @property
    def requires_grad(self) -> bool:
        return self.mean.requires_grad or self.var.requires_grad

    def zero_grad(self) -> None:
        self.mean.zero_grad()
        self.var.zero_grad()

    def numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns detached copies of (mean, var)."""
        return self.mean.numpy(), self.var.numpy()

    # ── Arithmetic Operations ────────────────────────────────────────

    def __add__(self, other: Union[DistributionalTensor, Tensor, float, int, np.ndarray]) -> DistributionalTensor:
        if isinstance(other, DistributionalTensor):
            return DistributionalTensor(self.mean + other.mean, self.var + other.var)
        elif isinstance(other, Tensor):
            return DistributionalTensor(self.mean + other, self.var)
        else:
            return DistributionalTensor(self.mean + other, self.var)

    def __radd__(self, other: Union[DistributionalTensor, Tensor, float, int, np.ndarray]) -> DistributionalTensor:
        return self.__add__(other)

    def __sub__(self, other: Union[DistributionalTensor, Tensor, float, int, np.ndarray]) -> DistributionalTensor:
        if isinstance(other, DistributionalTensor):
            # Var[X - Y] = Var[X] + Var[Y] for independent variables
            return DistributionalTensor(self.mean - other.mean, self.var + other.var)
        elif isinstance(other, Tensor):
            return DistributionalTensor(self.mean - other, self.var)
        else:
            return DistributionalTensor(self.mean - other, self.var)

    def __rsub__(self, other: Union[DistributionalTensor, Tensor, float, int, np.ndarray]) -> DistributionalTensor:
        if isinstance(other, (Tensor, int, float, np.ndarray)):
            return DistributionalTensor(other - self.mean, self.var)
        return (-self) + other

    def __neg__(self) -> DistributionalTensor:
        return DistributionalTensor(-self.mean, self.var)

    def __mul__(self, other: Union[DistributionalTensor, Tensor, float, int, np.ndarray]) -> DistributionalTensor:
        if isinstance(other, DistributionalTensor):
            # Goodman (1960) Product Variance formula:
            # E[XY] = mu_X * mu_Y
            # Var[XY] = mu_X^2 * var_Y + mu_Y^2 * var_X + var_X * var_Y
            mu_out = self.mean * other.mean
            var_out = (self.mean ** 2) * other.var + (other.mean ** 2) * self.var + self.var * other.var
            return DistributionalTensor(mu_out, var_out)
        elif isinstance(other, Tensor):
            # Var[c * X] = c^2 * Var[X]
            return DistributionalTensor(self.mean * other, self.var * (other ** 2))
        else:
            c = float(other)
            return DistributionalTensor(self.mean * c, self.var * (c ** 2))

    def __rmul__(self, other: Union[DistributionalTensor, Tensor, float, int, np.ndarray]) -> DistributionalTensor:
        return self.__mul__(other)

    def __truediv__(self, other: Union[float, int, Tensor]) -> DistributionalTensor:
        if isinstance(other, (int, float)):
            c = float(other)
            return DistributionalTensor(self.mean / c, self.var / (c ** 2))
        elif isinstance(other, Tensor):
            return DistributionalTensor(self.mean / other, self.var / (other ** 2))
        raise NotImplementedError("Distributional division by random variables is not supported directly.")

    def __matmul__(self, other: Union[DistributionalTensor, Tensor]) -> DistributionalTensor:
        if isinstance(other, Tensor):
            # Deterministic weights W:
            # E[X @ W] = E[X] @ W
            # Var[X @ W] = Var[X] @ (W^2)
            mu_out = self.mean @ other
            var_out = self.var @ (other ** 2)
            return DistributionalTensor(mu_out, var_out)
        elif isinstance(other, DistributionalTensor):
            # Both X and W are distributional:
            # E[X @ W] = E[X] @ E[W]
            # Var[(X @ W)_j] = sum_i [ mu_X_i^2 * var_W_ij + mu_W_ij^2 * var_X_i + var_X_i * var_W_ij ]
            mu_out = self.mean @ other.mean
            var_out = (
                self.var @ (other.mean ** 2)
                + (self.mean ** 2) @ other.var
                + self.var @ other.var
            )
            return DistributionalTensor(mu_out, var_out)
        else:
            raise TypeError(f"Matmul requires Tensor or DistributionalTensor, got {type(other)}")

    # ── Non-Linear Activations (Taylor Moment Expansions) ─────────────

    def tanh(self) -> DistributionalTensor:
        """
        Tanh activation moment propagation:
        phi(x) = tanh(x)
        phi'(x) = 1 - tanh^2(x)
        E[Y] ~ tanh(mu)
        Var[Y] ~ [phi'(mu)]^2 * Var[X]
        """
        mu_out = self.mean.tanh()
        deriv = (mu_out * (-1.0)) * mu_out + 1.0  # 1 - tanh^2(mu)
        var_out = self.var * (deriv ** 2)
        return DistributionalTensor(mu_out, var_out)

    def sigmoid(self) -> DistributionalTensor:
        """
        Sigmoid activation moment propagation:
        phi(x) = sigma(x)
        phi'(x) = sigma(x) * (1 - sigma(x))
        E[Y] ~ sigma(mu)
        Var[Y] ~ [phi'(mu)]^2 * Var[X]
        """
        s = self.mean.sigmoid()
        deriv = s * (1.0 - s)
        var_out = self.var * (deriv ** 2)
        return DistributionalTensor(s, var_out)

    def relu(self, smooth: bool = False) -> DistributionalTensor:
        """
        ReLU activation moment propagation.
        If smooth=False (default):
            mu_out = ReLU(mu)
            var_out = var * (mu > 0)
        """
        mu_out = self.mean.relu()
        mask_np = (self.mean.data > 0).astype(np.float64)
        mask_t = Tensor(mask_np, requires_grad=False)
        var_out = self.var * mask_t
        return DistributionalTensor(mu_out, var_out)

    def gelu(self) -> DistributionalTensor:
        """
        GELU activation moment propagation:
        GELU(x) ~ x * Phi(x)
        """
        mu_out = self.mean.gelu()
        # Numerical derivative of GELU for variance propagation
        x_data = self.mean.data
        cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x_data + 0.044715 * (x_data ** 3))))
        pdf = np.exp(-0.5 * (x_data ** 2)) / np.sqrt(2.0 * np.pi)
        deriv_np = cdf + x_data * pdf
        deriv_t = Tensor(deriv_np, requires_grad=False)
        var_out = self.var * (deriv_t ** 2)
        return DistributionalTensor(mu_out, var_out)

    # ── Reductions ───────────────────────────────────────────────────

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> DistributionalTensor:
        return DistributionalTensor(
            self.mean.sum(axis=axis, keepdims=keepdims),
            self.var.sum(axis=axis, keepdims=keepdims),
        )

    def mean_reduce(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> DistributionalTensor:
        """Var[mean(X)] = Var[sum(X)] / N^2"""
        mu_out = self.mean.mean(axis=axis, keepdims=keepdims)
        if axis is None:
            n = self.mean.data.size
        elif isinstance(axis, int):
            n = self.mean.data.shape[axis]
        else:
            n = 1
            for a in axis:
                n *= self.mean.data.shape[a]
        var_out = self.var.sum(axis=axis, keepdims=keepdims) / (n ** 2)
        return DistributionalTensor(mu_out, var_out)

    # ── Epistemic Confidence Metrics ─────────────────────────────────

    def confidence_interval(self, k: float = 2.0) -> Tuple[Tensor, Tensor]:
        """
        Returns the (lower, upper) confidence interval bounds: mu +/- k * sigma.
        For k=2.0, covers ~95.4% of Gaussian probability mass.
        """
        spread = self.std * k
        return self.mean - spread, self.mean + spread

    def signal_to_noise_ratio(self) -> Tensor:
        """Computes SNR = |mu| / (sigma + eps)."""
        return (self.mean.relu() + (-self.mean).relu()) / (self.std + 1e-8)

    def epistemic_uncertainty(self) -> float:
        """Returns scalar mean standard deviation as general model doubt."""
        return float(np.mean(self.std.numpy()))

    def __repr__(self) -> str:
        return f"DistributionalTensor(mean={self.mean.data}, var={self.var.data})"


# ── Distributional Neural Network Layers ─────────────────────────────

class DistributionalLinear(Module):
    """
    Fully Connected Layer with Analytical Moment Propagation.

    Propagates input mean and variance through weights W in a single forward pass:
        mu_out  = x_mu @ W + b_mu
        var_out = x_var @ (W^2) + b_var
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_uncertainty: bool = False,
        init_log_var: float = -4.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.weight_uncertainty = weight_uncertainty

        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * scale,
            requires_grad=True,
        )

        self.bias: Optional[Tensor] = Tensor(np.zeros(out_features), requires_grad=True) if bias else None

        self.weight_log_var: Optional[Tensor] = None
        self.bias_log_var: Optional[Tensor] = None
        if weight_uncertainty:
            self.weight_log_var = Tensor(
                np.full((in_features, out_features), init_log_var),
                requires_grad=True,
            )
            if bias:
                self.bias_log_var = Tensor(
                    np.full(out_features, init_log_var),
                    requires_grad=True,
                )

    def forward(self, x: Union[DistributionalTensor, Tensor]) -> DistributionalTensor:
        if not isinstance(x, DistributionalTensor):
            x_dist = DistributionalTensor(x)
        else:
            x_dist = x

        mu_out = x_dist.mean @ self.weight
        if self.bias is not None:
            mu_out = mu_out + self.bias

        weight_sq = self.weight ** 2
        var_out = x_dist.var @ weight_sq

        if self.weight_uncertainty and self.weight_log_var is not None:
            w_var = self.weight_log_var.exp()
            var_out = var_out + (x_dist.mean ** 2) @ w_var + x_dist.var @ w_var
            if self.bias_log_var is not None:
                var_out = var_out + self.bias_log_var.exp()

        return DistributionalTensor(mu_out, var_out)


class DistributionalSequential(Module):
    """
    Sequential Container for Distributional Tensors.
    """

    def __init__(self, layers: Sequence[Module]) -> None:
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: Union[DistributionalTensor, Tensor]) -> DistributionalTensor:
        curr = x if isinstance(x, DistributionalTensor) else DistributionalTensor(x)
        for layer in self.layers:
            curr = layer(curr)
        return curr

    def parameters(self) -> List[Tensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, idx: int) -> Module:
        return self.layers[idx]


# ── Heteroscedastic Gaussian NLL Loss ────────────────────────────────

class GaussianNLLLoss(Module):
    """
    Heteroscedastic Gaussian Negative Log-Likelihood Loss.

    Loss = 0.5 * mean( (y - mu)^2 / var + log(var) )

    Drives the network to output high variance for noisy/uncertain examples
    and low variance for clean, predictable examples.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        pred: Union[DistributionalTensor, Tuple[Tensor, Tensor]],
        target: Union[Tensor, np.ndarray],
    ) -> Tensor:
        if isinstance(pred, DistributionalTensor):
            mu = pred.mean
            var = pred.var
        elif isinstance(pred, (tuple, list)) and len(pred) == 2:
            mu, var = pred
        else:
            raise TypeError(f"GaussianNLLLoss expects DistributionalTensor, got {type(pred)}")

        if not isinstance(target, Tensor):
            target_t = Tensor(target, requires_grad=False)
        else:
            target_t = target

        # Ensure variance is strictly positive
        var_pos = var.relu() + self.eps
        sq_err = (target_t - mu) ** 2
        loss_elements = (sq_err / var_pos + var_pos.log()) * 0.5

        if self.reduction == "mean":
            return loss_elements.mean()
        elif self.reduction == "sum":
            return loss_elements.sum()
        return loss_elements


# ── Unified PRAMANA Namespace ────────────────────────────────────────

class PRAMANA:
    """
    P.R.A.M.A.N.A. — Probabilistic Representation of Analytical Moments
    and Algebraic Noise-aware Autograd.
    """

    @staticmethod
    def evaluate_uncertainty_telemetry(
        model: Module,
        in_dist_data: Union[DistributionalTensor, Tensor],
        out_dist_data: Union[DistributionalTensor, Tensor],
        targets: Optional[Tensor] = None,
    ) -> PramanaTelemetry:
        """
        Computes formal epistemic uncertainty and calibration metrics across
        in-distribution and out-of-distribution inputs.
        """
        with no_grad():
            pred_in = model(in_dist_data)
            pred_ood = model(out_dist_data)

        in_std = float(np.mean(pred_in.std.numpy()))
        ood_std = float(np.mean(pred_ood.std.numpy()))
        ratio = ood_std / max(in_std, 1e-12)

        ci_coverage = 0.0
        if targets is not None:
            low, high = pred_in.confidence_interval(k=2.0)
            t_data = targets.numpy()
            in_ci = (t_data >= low.numpy()) & (t_data <= high.numpy())
            ci_coverage = float(np.mean(in_ci))

        return PramanaTelemetry(
            in_dist_mean_std=in_std,
            out_dist_mean_std=ood_std,
            epistemic_divergence_ratio=ratio,
            fraction_in_95_ci=ci_coverage,
            total_samples_evaluated=len(in_dist_data) + len(out_dist_data),
        )

    DistributionalTensor = DistributionalTensor
    DistributionalLinear = DistributionalLinear
    DistributionalSequential = DistributionalSequential
    GaussianNLLLoss = GaussianNLLLoss
    PramanaTelemetry = PramanaTelemetry
