"""
contracts.py — Formal Mathematical, Lifecycle, and Dtype Contracts for miniGrad.

Defines the ground-truth invariants, state machines, and numerical tolerance
bounds enforced across the engine.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Tuple, Union

import numpy as np


class GraphState(IntEnum):
    """Lifecycle state machine for computation graph nodes."""
    LIVE = 1
    FREED = 2


# Standard error message required by the Autograd Lifecycle Contract
GRAPH_FREED_ERROR_MSG = (
    "Trying to backward through the graph a second time, "
    "but the saved intermediate values have already been freed. "
    "Specify retain_graph=True when calling backward the first time."
)


def dynamic_tolerance(
    reference: Union[np.ndarray, float],
    dtype: Union[np.dtype, type, str] = np.float64,
) -> np.ndarray:
    """
    Computes dynamic tolerance bound: atol + rtol * |reference|.
    Float64: atol=1e-6, rtol=1e-5
    Float32: atol=1e-4, rtol=1e-4
    """
    dt = np.dtype(dtype)
    if dt == np.float64:
        atol, rtol = 1e-6, 1e-5
    else:
        atol, rtol = 1e-4, 1e-4
    ref_arr = np.asarray(reference)
    return atol + rtol * np.abs(ref_arr)


def check_dynamic_parity(
    actual: Union[np.ndarray, float],
    reference: Union[np.ndarray, float],
    dtype: Union[np.dtype, type, str] = np.float64,
) -> Tuple[bool, float, float]:
    """
    Checks if |actual - reference| <= atol + rtol * |reference|.
    Returns (passed, max_diff, max_allowed_tol).
    """
    act = np.asarray(actual)
    ref = np.asarray(reference)
    tol = dynamic_tolerance(ref, dtype=dtype)
    diff = np.abs(act - ref)
    max_diff = float(np.max(diff)) if diff.size > 0 else 0.0
    max_tol = float(np.max(tol)) if tol.size > 0 else 0.0
    passed = bool(np.all(diff <= tol))
    return passed, max_diff, max_tol


def numerical_tolerance(
    reference: Union[np.ndarray, float],
    dtype: Union[np.dtype, type, str] = np.float64,
) -> np.ndarray:
    """
    Computes numerical tolerance bound for finite differences: atol + rtol * |reference|.
    Float64: atol=1e-4, rtol=1e-3
    Float32: atol=2e-3, rtol=1e-2
    """
    dt = np.dtype(dtype)
    if dt == np.float64:
        atol, rtol = 1e-4, 1e-3
    else:
        atol, rtol = 2e-3, 1e-2
    ref_arr = np.asarray(reference)
    return atol + rtol * np.abs(ref_arr)


def check_numerical_parity(
    actual: Union[np.ndarray, float],
    reference: Union[np.ndarray, float],
    dtype: Union[np.dtype, type, str] = np.float64,
) -> Tuple[bool, float, float]:
    """
    Checks if |actual - reference| <= atol + rtol * |reference| for finite difference checks.
    Returns (passed, max_diff, max_allowed_tol).
    """
    act = np.asarray(actual)
    ref = np.asarray(reference)
    tol = numerical_tolerance(ref, dtype=dtype)
    diff = np.abs(act - ref)
    max_diff = float(np.max(diff)) if diff.size > 0 else 0.0
    max_tol = float(np.max(tol)) if tol.size > 0 else 0.0
    passed = bool(np.all(diff <= tol))
    return passed, max_diff, max_tol

