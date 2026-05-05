"""
loss.py — Loss functions for miniGrad.

Each loss function takes predictions and targets, computes a scalar loss,
and sets up the backward pass so gradients flow correctly to the model parameters.

All losses return a scalar Tensor that can be backpropagated through.
"""
from __future__ import annotations

import numpy as np

from minigrad.tensor import Tensor
from minigrad.nn.module import Module


class MSELoss(Module):
    """
    Mean Squared Error Loss.

    L = (1/N) * sum((pred - target)^2)

    Used for regression tasks.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f" reduction must be 'mean' or 'sum', got '{reduction}'")
        self.reduction = reduction

    def forward(self, pred: Tensor, target: np.ndarray) -> Tensor:
        """
        Args:
            pred:   Model predictions, shape (N, *)
            target: Ground truth values, same shape as pred
        Returns:
            Scalar loss tensor
        """
        target_t = Tensor(target) if not isinstance(target, Tensor) else target
        diff = pred - target_t
        squared = diff ** 2

        if self.reduction == "mean":
            loss = squared.mean()
        else:
            loss = squared.sum()

        return loss

    def __repr__(self) -> str:
        return f"MSELoss(reduction='{self.reduction}')"


class CrossEntropyLoss(Module):
    """
    Cross-Entropy Loss for multi-class classification.

    Combines log-softmax and negative log-likelihood for numerical stability.

    L = -mean(log(softmax(logits)[correct_class]))

    Args:
        reduction: "mean" or "sum"
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: np.ndarray) -> Tensor:
        """
        Args:
            logits: Raw model outputs, shape (N, num_classes)
            targets: Integer class indices, shape (N,)
        Returns:
            Scalar loss tensor
        """
        N = logits.data.shape[0]

        # Numerically stable softmax: subtract max before exp
        shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

        # Negative log-likelihood on correct classes
        correct_probs = probs[np.arange(N), targets]
        correct_logprobs = -np.log(correct_probs + 1e-9)

        if self.reduction == "mean":
            loss_val = correct_logprobs.mean()
        else:
            loss_val = correct_logprobs.sum()

        result = Tensor(loss_val, requires_grad=logits.requires_grad,
                       _children=(logits,), _op="cross_entropy")

        def _backward() -> None:
            if logits.requires_grad:
                # Gradient of softmax + CE combined = (probs - one_hot) / N (or 1 if sum)
                grad = probs.copy()
                grad[np.arange(N), targets] -= 1.0
                if self.reduction == "mean":
                    grad = grad / N
                logits.grad += grad * result.grad

        result._backward = _backward
        return result

    def __repr__(self) -> str:
        return f"CrossEntropyLoss(reduction='{self.reduction}')"


class BCELoss(Module):
    """
    Binary Cross-Entropy Loss.

    L = -mean(y * log(p) + (1-y) * log(1-p))

    Used for binary classification. Assumes inputs are already passed
    through sigmoid (probabilities in [0, 1]).
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-7) -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred: Tensor, target: np.ndarray) -> Tensor:
        """
        Args:
            pred:   Probabilities in [0, 1], shape (N, *)
            target: Ground truth in {0, 1}, same shape as pred
        Returns:
            Scalar loss tensor
        """
        p = np.clip(pred.data, self.eps, 1.0 - self.eps)
        t = target if isinstance(target, np.ndarray) else target.data

        loss_per_elem = -(t * np.log(p) + (1.0 - t) * np.log(1.0 - p))

        if self.reduction == "mean":
            loss_val = loss_per_elem.mean()
        else:
            loss_val = loss_per_elem.sum()

        result = Tensor(loss_val, requires_grad=pred.requires_grad,
                       _children=(pred,), _op="bce")

        def _backward() -> None:
            if pred.requires_grad:
                # dL/dp = -(t/p - (1-t)/(1-p))
                grad = -(t / p - (1.0 - t) / (1.0 - p))
                if self.reduction == "mean":
                    grad = grad / loss_per_elem.size
                pred.grad += grad * result.grad

        result._backward = _backward
        return result

    def __repr__(self) -> str:
        return f"BCELoss(reduction='{self.reduction}')"


class BCEWithLogitsLoss(Module):
    """
    Binary Cross-Entropy with built-in sigmoid for numerical stability.

    This is more numerically stable than using Sigmoid + BCE separately.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")
        self.reduction = reduction

    def forward(self, logits: Tensor, target: np.ndarray) -> Tensor:
        """
        Args:
            logits: Raw outputs (before sigmoid), shape (N, *)
            target: Ground truth in {0, 1}, same shape as logits
        """
        z = logits.data
        t = target if isinstance(target, np.ndarray) else target.data

        # Stable computation: max(z, 0) - z*t + log(1 + exp(-|z|))
        max_z = np.maximum(z, 0)
        loss_per_elem = max_z - z * t + np.log(1.0 + np.exp(-np.abs(z)))

        if self.reduction == "mean":
            loss_val = loss_per_elem.mean()
        else:
            loss_val = loss_per_elem.sum()

        result = Tensor(loss_val, requires_grad=logits.requires_grad,
                       _children=(logits,), _op="bce_with_logits")

        def _backward() -> None:
            if logits.requires_grad:
                # sigmoid(z) - t
                sigmoid_z = 1.0 / (1.0 + np.exp(-z))
                grad = sigmoid_z - t
                if self.reduction == "mean":
                    grad = grad / loss_per_elem.size
                logits.grad += grad * result.grad

        result._backward = _backward
        return result

    def __repr__(self) -> str:
        return f"BCEWithLogitsLoss(reduction='{self.reduction}')"


class NLLLoss(Module):
    """
    Negative Log-Likelihood Loss.

    Expects log-probabilities (output of log_softmax) and class indices.
    This is the inner part of CrossEntropyLoss.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")
        self.reduction = reduction

    def forward(self, log_probs: Tensor, targets: np.ndarray) -> Tensor:
        """
        Args:
            log_probs: Log-probabilities, shape (N, C)
            targets:   Class indices, shape (N,)
        """
        N = log_probs.data.shape[0]
        correct_logprobs = -log_probs.data[np.arange(N), targets]

        if self.reduction == "mean":
            loss_val = correct_logprobs.mean()
        else:
            loss_val = correct_logprobs.sum()

        result = Tensor(loss_val, requires_grad=log_probs.requires_grad,
                       _children=(log_probs,), _op="nll")

        def _backward() -> None:
            if log_probs.requires_grad:
                grad = np.zeros_like(log_probs.data)
                grad[np.arange(N), targets] = -1.0
                if self.reduction == "mean":
                    grad = grad / N
                log_probs.grad += grad * result.grad

        result._backward = _backward
        return result

    def __repr__(self) -> str:
        return f"NLLLoss(reduction='{self.reduction}')"
