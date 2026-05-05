"""
utils.py — Utility functions for miniGrad.

Provides gradient checking, plotting, seed setting, and other helpers
to make training and debugging easier.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Optional, List

from minigrad.tensor import Tensor


def grad_check(
    f: Callable,
    inputs: list[Tensor],
    eps: float = 1e-5,
    atol: float = 1e-4,
    verbose: bool = True,
) -> bool:
    """
    Numerical gradient check using finite differences.

    Compares analytical gradients (from backprop) with numerical gradients
    computed via central differences. Essential for verifying correctness
    of custom operations.

    Args:
        f:        Function that takes inputs and returns a scalar Tensor
        inputs:   List of Tensors to check gradients for
        eps:      Step size for finite differences
        atol:     Absolute tolerance for comparison
        verbose:  Print results

    Returns:
        True if gradients match, False otherwise

    Example:
        x = Tensor([1.0, 2.0], requires_grad=True)
        def f():
            return (x * x).sum()
        grad_check(f, [x])
    """
    # Zero grads before analytical pass to prevent accumulation
    for inp in inputs:
        inp.zero_grad()
    out = f()
    out.backward()

    all_passed = True

    for inp in inputs:
        if not inp.requires_grad:
            continue

        analytical_grad = inp.grad.copy()
        numerical_grad = np.zeros_like(inp.data)

        # Finite differences: (f(x+eps) - f(x-eps)) / (2*eps)
        it = np.nditer(inp.data, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            original = inp.data[idx]

            inp.data[idx] = original + eps
            f_plus = f().data.sum()

            inp.data[idx] = original - eps
            f_minus = f().data.sum()

            numerical_grad[idx] = (f_plus - f_minus) / (2.0 * eps)

            inp.data[idx] = original
            it.iternext()

        # Compare
        diff = np.abs(analytical_grad - numerical_grad)
        max_diff = diff.max()

        if max_diff > atol:
            all_passed = False
            if verbose:
                print(f"  FAIL: max diff = {max_diff:.2e} (atol={atol:.2e})")
                # Find worst mismatch
                worst_idx = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"  Analytical: {analytical_grad[worst_idx]:.8f}")
                print(f"  Numerical:  {numerical_grad[worst_idx]:.8f}")
        else:
            if verbose:
                print(f"  PASS: max diff = {max_diff:.2e}")

    return all_passed


def numerical_gradient(f: Callable, x: Tensor, eps: float = 1e-5) -> np.ndarray:
    """
    Compute numerical gradient of f with respect to x using central differences.

    Args:
        f:   Function that returns a scalar Tensor
        x:   Tensor to compute gradient for
        eps: Step size

    Returns:
        Numerical gradient array with same shape as x.data
    """
    grad = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        original = x.data[idx]

        x.data[idx] = original + eps
        f_plus = f().data.sum()

        x.data[idx] = original - eps
        f_minus = f().data.sum()

        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
        x.data[idx] = original
        it.iternext()

    return grad


def seed_everything(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across numpy.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)


def count_parameters(model) -> int:
    """
    Count total number of trainable parameters in a model.

    Args:
        model: A Module with parameters() method

    Returns:
        Total parameter count
    """
    return sum(p.data.size for p in model.parameters())


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Predicted class indices, shape (N,)
        targets:     True class indices, shape (N,)

    Returns:
        Accuracy as a float between 0 and 1
    """
    return np.mean(predictions == targets)


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoding.

    Args:
        labels:       Integer class labels, shape (N,)
        num_classes:  Total number of classes

    Returns:
        One-hot encoded array, shape (N, num_classes)
    """
    N = labels.shape[0]
    one_hot = np.zeros((N, num_classes))
    one_hot[np.arange(N), labels] = 1.0
    return one_hot


def plot_loss_curve(
    losses: List[float],
    save_path: Optional[str] = None,
    title: str = "Training Loss",
) -> None:
    """
    Plot training loss curve using matplotlib.

    Args:
        losses:    List of loss values
        save_path: If provided, save figure to this path
        title:     Plot title
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def describe_model(model) -> None:
    """
    Print a summary of model architecture and parameter counts.

    Args:
        model: A Module with named_modules() or parameters()
    """
    print("=" * 60)
    print("Model Summary")
    print("=" * 60)

    if hasattr(model, "named_modules"):
        for name, module in model.named_modules():
            params = sum(p.data.size for p in module.parameters()) if hasattr(module, "parameters") else 0
            if params > 0:
                print(f"{name:30s} {params:>10,} params  {module}")

    total = count_parameters(model)
    print("=" * 60)
    print(f"Total trainable parameters: {total:,}")
    print("=" * 60)


def save_model(model, path: str) -> None:
    """
    Save model parameters to a file.

    Args:
        model: Model to save
        path:  File path (e.g., 'model.npz')
    """
    params_dict = {}
    for i, p in enumerate(model.parameters()):
        params_dict[f"param_{i}"] = p.data
    np.savez(path, **params_dict)
    print(f"Model saved to {path}")


def load_model(model, path: str) -> None:
    """
    Load model parameters from a file.

    Args:
        model: Model to load parameters into
        path:  File path (e.g., 'model.npz')
    """
    data = np.load(path)
    params = model.parameters()
    for i, key in enumerate(sorted(data.files, key=lambda x: int(x.split("_")[-1]))):
        params[i].data = data[key]
    print(f"Model loaded from {path}")
