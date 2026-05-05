"""
transforms.py — Data preprocessing transforms.

Common transforms for normalizing and converting data before feeding to models.
"""
from __future__ import annotations

import numpy as np
from typing import List


class Transform:
    """Base class for data transforms."""

    def __call__(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ToTensor(Transform):
    """Convert data to float64 numpy array."""

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return np.array(data, dtype=np.float64)


class Normalize(Transform):
    """
    Normalize data with mean and std.

    out = (data - mean) / std
    """

    def __init__(self, mean: float | np.ndarray, std: float | np.ndarray) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std


class FlattenTransform(Transform):
    """Flatten multi-dimensional input to 1D."""

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return data.reshape(-1)


class Compose:
    """
    Compose multiple transforms into a single pipeline.

    Example:
        transform = Compose([ToTensor(), Normalize(0.5, 0.5)])
        data = transform(raw_data)
    """

    def __init__(self, transforms: List[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, data: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self) -> str:
        transforms_str = " -> ".join(str(t) for t in self.transforms)
        return f"Compose([{transforms_str}])"
