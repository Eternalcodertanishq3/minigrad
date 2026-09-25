"""
lora.py — Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.

Instead of updating the full weight matrix W (d×k), LoRA freezes W and
learns a low-rank decomposition ΔW = B @ A where B is (d×r) and A is (r×k),
with r << min(d, k). This reduces trainable parameters by 10-100x while
preserving model quality.

Reference: "LoRA: Low-Rank Adaptation of Large Language Models"
(Hu et al., 2021) — https://arxiv.org/abs/2106.09685

Usage:
    # Replace specific Linear layers with LoRA versions
    model = MiniGPT(...)
    model.freeze()  # Freeze all base weights
    apply_lora(model, target_modules=["q_proj", "v_proj"], rank=4)
    # Only LoRA A/B matrices are trainable now
    optimizer = Adam(model.parameters(), lr=1e-4)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from minigrad.tensor import Tensor
from minigrad.nn.module import Module
from minigrad.nn.linear import Linear


class LoRALinear(Module):
    """
    A Linear layer augmented with low-rank adaptation.

    Computes: y = x @ W_frozen + (x @ A @ B) * scaling
    where W_frozen is the original (frozen) weight and A, B are the
    trainable low-rank matrices.

    Args:
        base_linear: The original Linear layer to wrap (will be frozen)
        rank:        Rank of the low-rank decomposition (default: 4)
        alpha:       LoRA scaling factor (default: 1.0). Effective scaling
                     is alpha / rank.
    """

    def __init__(
        self,
        base_linear: Linear,
        rank: int = 4,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()

        in_features = base_linear.weight.data.shape[0]
        out_features = base_linear.weight.data.shape[1]

        assert rank > 0, f"LoRA rank must be positive, got {rank}"
        assert rank <= min(in_features, out_features), (
            f"LoRA rank ({rank}) must be <= min(in_features={in_features}, "
            f"out_features={out_features})"
        )

        self.rank = rank
        self.scaling = alpha / rank

        # Freeze the base linear layer
        self.base = base_linear
        self.base.freeze()

        # LoRA matrices: A is (in_features, rank), B is (rank, out_features)
        # A: Kaiming-uniform initialization (same scale as base)
        # B: Zero initialization so LoRA starts as identity (ΔW = 0)
        self.lora_A = Tensor(
            np.random.randn(in_features, rank) * np.sqrt(2.0 / in_features),
            requires_grad=True,
        )
        self.lora_B = Tensor(
            np.zeros((rank, out_features)),
            requires_grad=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: base output + LoRA delta.

        Args:
            x: Input tensor of shape (..., in_features)
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Base (frozen) forward pass — still contributes to output
        base_out = self.base(x)

        # LoRA path: x @ A @ B * scaling
        # Handle 3D inputs the same way as Linear
        leading_shape = x.data.shape[:-1]
        in_features = x.data.shape[-1]

        if x.data.ndim > 2:
            x_2d = x.reshape(-1, in_features)
        else:
            x_2d = x

        lora_out = (x_2d @ self.lora_A) @ self.lora_B

        if x.data.ndim > 2:
            lora_out = lora_out.reshape(*leading_shape, self.lora_B.data.shape[1])

        return base_out + lora_out * self.scaling

    def merge(self) -> Linear:
        """
        Merge LoRA weights into the base Linear layer.

        Produces a single Linear with W_merged = W_base + A @ B * scaling.
        Use this after fine-tuning to eliminate the LoRA overhead at inference.

        Returns:
            A new Linear layer with merged weights.
        """
        merged_weight = (
            self.base.weight.data
            + self.lora_A.data @ self.lora_B.data * self.scaling
        )
        in_f, out_f = merged_weight.shape
        has_bias = self.base.bias is not None

        result = Linear(in_f, out_f, bias=has_bias)
        result.weight = Tensor(merged_weight, requires_grad=True)
        if self.base.bias is not None:
            result.bias = Tensor(self.base.bias.data.copy(), requires_grad=True)
        return result

    def __repr__(self) -> str:
        in_f = self.base.weight.data.shape[0]
        out_f = self.base.weight.data.shape[1]
        base_params = in_f * out_f
        lora_params = in_f * self.rank + self.rank * out_f
        return (
            f"LoRALinear(in={in_f}, out={out_f}, rank={self.rank}, "
            f"scaling={self.scaling:.2f}, "
            f"trainable={lora_params}/{base_params + lora_params} "
            f"({100 * lora_params / (base_params + lora_params):.1f}%))"
        )


def apply_lora(
    model: Module,
    target_modules: Optional[List[str]] = None,
    rank: int = 4,
    alpha: float = 1.0,
) -> int:
    """
    Replace Linear layers in a model with LoRA-augmented versions.

    Scans the model recursively and replaces any Linear layer whose
    attribute name matches one of `target_modules` with a LoRALinear.

    Args:
        model:          The model to apply LoRA to
        target_modules: List of attribute names to target (e.g. ["q_proj", "v_proj"]).
                        If None, replaces ALL Linear layers.
        rank:           LoRA rank
        alpha:          LoRA scaling factor

    Returns:
        Number of layers replaced

    Example:
        model = MiniGPT(...)
        model.freeze()
        count = apply_lora(model, target_modules=["q_proj", "v_proj"], rank=4)
        print(f"Applied LoRA to {count} layers")
        optimizer = Adam(model.parameters(), lr=1e-4)  # Only LoRA params
    """
    replaced = 0

    def _apply(module: Module) -> None:
        nonlocal replaced
        for name, attr in list(vars(module).items()):
            if isinstance(attr, Linear):
                if target_modules is None or name in target_modules:
                    lora_layer = LoRALinear(attr, rank=rank, alpha=alpha)
                    setattr(module, name, lora_layer)
                    replaced += 1
            elif isinstance(attr, Module):
                _apply(attr)
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Module):
                        _apply(item)

    _apply(model)
    return replaced
