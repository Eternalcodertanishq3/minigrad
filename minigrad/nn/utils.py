"""
utils.py — Neural network utilities.

Provides gradient clipping functions essential for training RNNs,
LSTMs, and Transformers where exploding gradients are common.
"""
from __future__ import annotations

import numpy as np
from typing import List, Union

from minigrad.tensor import Tensor


def clip_grad_norm_(parameters: List[Tensor], max_norm: float, norm_type: float = 2.0) -> float:
    """
    Clip the gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters: Iterable of Tensors with gradients
        max_norm:   Maximum allowed norm of the gradients
        norm_type:  Type of the used p-norm (default: 2.0 for L2 norm)

    Returns:
        Total norm of the parameter gradients (before clipping)
    """
    parameters = [p for p in parameters if p.requires_grad]
    if len(parameters) == 0:
        return 0.0

    if norm_type == float('inf'):
        total_norm = max(np.max(np.abs(p.grad)) for p in parameters)
    else:
        total_norm = np.sqrt(sum(np.sum(np.abs(p.grad) ** norm_type) for p in parameters)) ** (1.0 / norm_type) if norm_type != 2.0 else np.sqrt(sum(np.sum(p.grad ** 2) for p in parameters))

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            p.grad *= clip_coef

    return float(total_norm)


def clip_grad_value_(parameters: List[Tensor], clip_value: float) -> None:
    """
    Clip the gradients of an iterable of parameters to a specified value.

    Gradients are modified in-place.

    Args:
        parameters: Iterable of Tensors with gradients
        clip_value: Maximum allowed absolute value of the gradients
    """
    for p in parameters:
        if p.requires_grad:
            np.clip(p.grad, -clip_value, clip_value, out=p.grad)
