"""
tests/test_glassbox.py — Unit and integration tests for Glass-Box Autograd,
First-NaN Root-Cause Localization, and Interactive HTML/SVG Visualizer.
"""
from pathlib import Path
import numpy as np
import pytest

from minigrad import (
    Tensor,
    detect_anomaly,
    explain_gradients,
    visualize,
    collect_telemetry,
    GradientAnomalyError,
)
from minigrad.glassbox import is_anomaly_detection_enabled
from minigrad.nn import Linear, Sequential, ReLU


def test_detect_anomaly_context_manager_state():
    """Verify anomaly detection state activates and deactivates correctly."""
    assert not is_anomaly_detection_enabled()

    with detect_anomaly():
        assert is_anomaly_detection_enabled()
        # Nested context
        with detect_anomaly():
            assert is_anomaly_detection_enabled()
        assert is_anomaly_detection_enabled()

    assert not is_anomaly_detection_enabled()

    # Verify exception cleans up global state
    try:
        with detect_anomaly():
            raise ValueError("Deliberate error")
    except ValueError:
        pass
    assert not is_anomaly_detection_enabled()


def test_detect_anomaly_traps_backward_inf_singularity():
    """
    Test trapping of gradient explosion (inf) created during the backward pass.
    For x^0.5 at x=0, forward is 0.0, but d/dx = 0.5 * x^(-0.5) -> inf.
    """
    with np.errstate(divide="ignore"):
        x = Tensor([0.0, 4.0], requires_grad=True)
        y = x ** 0.5
        loss = y.sum()

        with pytest.raises(GradientAnomalyError) as exc_info:
            with detect_anomaly():
                loss.backward()

    err = exc_info.value
    assert "pow" in err.op
    assert err.node_id is not None
    assert "power" in err.reason.lower() or "zero" in err.reason.lower()
    assert "First Poisoned Coordinate" in err.details
    assert "(0,)" in err.details["First Poisoned Coordinate"] or "0" in err.details["First Poisoned Coordinate"]
    assert str(err).startswith("\n")
    assert "GRADIENT POISONING DETECTED" in str(err)


def test_detect_anomaly_traps_forward_nan_poisoning():
    """
    Test trapping of NaN created during the forward pass (e.g. log of negative number)
    when calling backward().
    """
    with np.errstate(invalid="ignore"):
        x = Tensor([-2.0, 1.0, 3.0], requires_grad=True)
        y = x.log()
        loss = y.sum()

        with pytest.raises(GradientAnomalyError) as exc_info:
            with detect_anomaly():
                loss.backward()

    err = exc_info.value
    assert err.op == "log"
    assert "non-positive" in err.reason.lower() or "logarithm" in err.reason.lower()
    assert "Culprit Node ID" in err.details
    assert err.details["Node Output Has NaN"] == "True"


def test_gradient_telemetry_health_statuses():
    """
    Test that collect_telemetry correctly assigns healthy, vanishing, exploding,
    and inactive statuses with accurate L2 norms and zero percentage.
    """
    # 1. Healthy branch
    w_healthy = Tensor(np.ones((4, 4)) * 0.5, requires_grad=True)
    x = Tensor(np.ones((1, 4)) * 1.0, requires_grad=False)
    out_healthy = x @ w_healthy

    # 2. Vanishing branch: scale by 1e-8
    w_vanishing = Tensor(np.ones((4, 4)) * 1.0, requires_grad=True)
    out_vanishing = (x @ w_vanishing) * 1e-8

    # 3. Exploding branch: scale by 1e5
    w_exploding = Tensor(np.ones((4, 4)) * 1.0, requires_grad=True)
    out_exploding = (x @ w_exploding) * 1e5

    loss = (out_healthy.sum() + out_vanishing.sum() + out_exploding.sum())
    loss.backward()

    telemetry = collect_telemetry(loss)

    assert id(loss) in telemetry
    assert id(w_healthy) in telemetry
    assert id(w_vanishing) in telemetry
    assert id(w_exploding) in telemetry
    assert id(x) in telemetry

    assert telemetry[id(x)].status == "inactive"
    assert not telemetry[id(x)].requires_grad

    assert telemetry[id(w_healthy)].status == "healthy"
    assert telemetry[id(w_vanishing)].status == "vanishing"
    assert telemetry[id(w_exploding)].status == "exploding"

    # Verify L2 norms are finite and match expectation
    assert 0.1 < telemetry[id(w_healthy)].l2_norm < 10.0
    assert telemetry[id(w_vanishing)].l2_norm < 1e-6
    assert telemetry[id(w_exploding)].l2_norm > 1000.0


def test_dead_neuron_percentage():
    """Test that zero_pct correctly computes percentage of zero gradient elements."""
    x = Tensor(np.array([[-2.0, -1.0], [3.0, 4.0]]), requires_grad=True)
    y = x.relu()
    loss = y.sum()
    loss.backward()

    telemetry = collect_telemetry(loss)
    t_x = telemetry[id(x)]

    # Exactly 2 elements out of 4 are inactive (zero gradient) -> 50.0%
    assert t_x.zero_pct == 50.0
    assert t_x.status == "healthy"


def test_explain_gradients_table_format():
    """Test explain_gradients string generation and convenience method."""
    model = Sequential([Linear(2, 4), ReLU(), Linear(4, 1)])
    inp = Tensor([[0.5, -0.5]], requires_grad=False)
    target = Tensor([[1.0]], requires_grad=False)

    pred = model(inp)
    loss = ((pred - target) ** 2).sum()
    loss.backward()

    # Test top-level function
    report = explain_gradients(loss)
    assert isinstance(report, str)
    assert "miniGrad Glass-Box Gradient Telemetry Report" in report
    assert "Node ID" in report
    assert "Operation" in report
    assert "L2 Norm" in report
    assert "Health Summary:" in report
    assert "[OK] Healthy" in report

    # Test Tensor.explain() convenience method
    tensor_report = loss.explain()
    assert tensor_report == report


def test_visualize_html_generation(tmp_path: Path):
    """Test HTML/SVG computational graph visualization output."""
    model = Sequential([Linear(3, 4), ReLU(), Linear(4, 2)])
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=False)
    pred = model(x)
    loss = pred.sum()
    loss.backward()

    out_file = tmp_path / "test_graph.html"

    # Test top-level visualize writing to file
    html_content = visualize(loss, filename=out_file)
    assert out_file.exists()
    assert len(html_content) > 1000

    # Test Tensor.visualize() convenience method (returns string without writing if None)
    string_html = loss.visualize(filename=None)
    assert "<!DOCTYPE html>" in string_html
    assert "<svg id=\"viewport\"" in string_html
    assert "<g id=\"nodes\">" in string_html
    assert "<g id=\"edges\">" in string_html
    assert "miniGrad Glass-Box DAG" in string_html
    assert "updateTransform()" in string_html
    assert "closeInspector()" in string_html
    assert "path d=\"M" in string_html


def test_poisoned_gradient_report_summary():
    """Verify that explain_gradients highlights poisoned nodes in the summary section."""
    a = Tensor([1.0], requires_grad=True)
    b = Tensor([float("nan")], requires_grad=True)
    c = a + b
    # Manually set gradient with NaN to inspect reporting
    c.grad = np.array([float("nan")])
    c._backward()

    report = explain_gradients(c)
    assert "[!] ROOT CAUSE SUMMARY:" in report
    assert "contains NaN/Inf gradients" in report
