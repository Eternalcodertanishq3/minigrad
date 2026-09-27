"""
tests/test_lifecycle.py — Formal autograd lifecycle state machine tests.

Validates the formal state transitions:
1. LIVE -> FREED upon backward(retain_graph=False).
2. RuntimeError raised on second backward when retain_graph=False.
3. retain_graph=True preserves LIVE state across successive backwards.
4. Shared subgraphs and multiple consumer execution:
   - a = x*2; b = a*3; b.backward(); a.backward()
   - z = f(x); g1 = z.sum(); g2 = z.mean(); g1.backward(); g2.backward()
5. Calling backward through an already-freed ancestor raises RuntimeError.
6. Functional autograd.grad() lifecycle enforcement with retain_graph.
7. Higher-order derivatives with create_graph=True.
"""
from __future__ import annotations

import numpy as np
import pytest

from minigrad.autograd import grad
from minigrad.contracts import GRAPH_FREED_ERROR_MSG, GraphState
from minigrad.tensor import Tensor

# ── 1. Basic State Machine Transitions ────────────────────────────────

def test_lifecycle_default_is_live():
    """Tensors are created in the LIVE state."""
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * 2.0
    assert x._lifecycle == GraphState.LIVE
    assert y._lifecycle == GraphState.LIVE


def test_lifecycle_backward_transitions_to_freed():
    """backward(retain_graph=False) transitions root to FREED."""
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = (x ** 2).sum()
    y.backward(retain_graph=False)
    assert y._lifecycle == GraphState.FREED


def test_lifecycle_second_backward_raises_runtime_error():
    """Calling backward() on a FREED node raises the canonical RuntimeError."""
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = (x ** 2).sum()
    y.backward(retain_graph=False)

    with pytest.raises(RuntimeError) as exc_info:
        y.backward()
    assert str(exc_info.value) == GRAPH_FREED_ERROR_MSG


def test_lifecycle_retain_graph_preserves_live_state():
    """backward(retain_graph=True) leaves root in LIVE state."""
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = (x * 3.0).sum()

    y.backward(retain_graph=True)
    assert y._lifecycle == GraphState.LIVE
    assert np.allclose(x.grad, [3.0, 3.0])

    # Second backward succeeds and accumulates
    y.backward(retain_graph=False)
    assert y._lifecycle == GraphState.FREED
    assert np.allclose(x.grad, [6.0, 6.0])

    # Third backward fails
    with pytest.raises(RuntimeError) as exc_info:
        y.backward()
    assert str(exc_info.value) == GRAPH_FREED_ERROR_MSG


# ── 2. Multiple Consumers and Shared Subgraphs ────────────────────────

def test_shared_subgraph_downstream_then_upstream():
    """
    Scenario: a = x*2; b = a*3; b.backward(); a.backward().
    Backwarding through downstream b frees b but leaves a accessible.
    """
    x = Tensor([2.0], requires_grad=True)
    a = x * 2.0
    b = (a * 3.0).sum()

    # Step 1: backward through b
    b.backward(retain_graph=False)
    assert b._lifecycle == GraphState.FREED
    assert a._lifecycle == GraphState.LIVE
    assert np.allclose(x.grad, [6.0])

    # Step 2: backward directly through a
    a_sum = a.sum()
    a_sum.backward(retain_graph=False)
    assert a_sum._lifecycle == GraphState.FREED
    assert np.allclose(x.grad, [8.0])

    # Calling b.backward() again fails
    with pytest.raises(RuntimeError) as exc_info:
        b.backward()
    assert str(exc_info.value) == GRAPH_FREED_ERROR_MSG

    # Calling a_sum.backward() again fails
    with pytest.raises(RuntimeError) as exc_info:
        a_sum.backward()
    assert str(exc_info.value) == GRAPH_FREED_ERROR_MSG


def test_shared_subgraph_sibling_consumers():
    """
    Scenario: z = f(x); g1 = z.sum(); g2 = z.mean(); g1.backward(); g2.backward().
    Backwarding g1 frees g1, leaving z accessible for g2.
    """
    x = Tensor([1.0, 3.0, 5.0], requires_grad=True)
    z = x ** 2
    g1 = z.sum()
    g2 = z.mean()

    # Step 1: backward g1
    g1.backward(retain_graph=False)
    assert g1._lifecycle == GraphState.FREED
    assert np.allclose(x.grad, [2.0, 6.0, 10.0])

    # Step 2: backward g2
    g2.backward(retain_graph=False)
    assert g2._lifecycle == GraphState.FREED
    expected_g2 = np.array([2.0, 6.0, 10.0]) / 3.0
    assert np.allclose(x.grad, [2.0, 6.0, 10.0] + expected_g2)

    # Calling g1 again fails
    with pytest.raises(RuntimeError) as exc_info:
        g1.backward()
    assert str(exc_info.value) == GRAPH_FREED_ERROR_MSG

    # Calling g2 again fails
    with pytest.raises(RuntimeError) as exc_info:
        g2.backward()
    assert str(exc_info.value) == GRAPH_FREED_ERROR_MSG


def test_backward_through_freed_ancestor_raises():
    """If an intermediate node was freed, backwarding through it fails."""
    x = Tensor([2.0], requires_grad=True)
    a = x * 2.0
    b = a * 3.0

    # Backward a first and free it
    a.backward(retain_graph=False)
    assert a._lifecycle == GraphState.FREED

    # Now trying to backward b through a raises RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        b.backward()
    assert str(exc_info.value) == GRAPH_FREED_ERROR_MSG


def test_backward_through_retained_ancestor_succeeds():
    """If intermediate node retained graph, downstream backward succeeds."""
    x = Tensor([2.0], requires_grad=True)
    a = x * 2.0
    b = a * 3.0

    # Backward a with retain_graph=True
    a.backward(retain_graph=True)
    assert a._lifecycle == GraphState.LIVE
    assert np.allclose(x.grad, [2.0])

    # Now backward b through a succeeds
    b.backward()
    assert b._lifecycle == GraphState.FREED
    assert np.allclose(x.grad, [8.0])


# ── 3. Functional autograd.grad() Lifecycle ──────────────────────────

def test_autograd_grad_lifecycle_freed():
    """grad(retain_graph=False) transitions outputs to FREED."""
    x = Tensor([2.0], requires_grad=True)
    y = x ** 3

    g1 = grad(y, x, retain_graph=False)[0]
    assert np.allclose(g1.data, [12.0])
    assert y._lifecycle == GraphState.FREED

    # Subsequent grad on y raises RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        grad(y, x)
    assert str(exc_info.value) == GRAPH_FREED_ERROR_MSG


def test_autograd_grad_lifecycle_retained():
    """grad(retain_graph=True) preserves LIVE state."""
    x = Tensor([2.0], requires_grad=True)
    y = x ** 3

    g1 = grad(y, x, retain_graph=True)[0]
    assert np.allclose(g1.data, [12.0])
    assert y._lifecycle == GraphState.LIVE

    # Second call succeeds
    g2 = grad(y, x, retain_graph=False)[0]
    assert np.allclose(g2.data, [12.0])
    assert y._lifecycle == GraphState.FREED


# ── 4. Higher-Order Derivatives & create_graph ───────────────────────

def test_create_graph_higher_order_derivative():
    """create_graph=True constructs derivative graph for 2nd order backward."""
    x = Tensor([3.0], requires_grad=True)
    # y = x^3 => dy/dx = 3x^2 = 27 => d²y/dx² = 6x = 18
    y = x ** 3
    y.backward(create_graph=True)

    assert isinstance(x.grad, Tensor)
    assert np.allclose(x.grad.data, [27.0])

    # Backprop through the first gradient to get second derivative
    first_grad = x.grad
    first_grad.backward()
    assert np.allclose(x.grad.data, [27.0 + 18.0])
