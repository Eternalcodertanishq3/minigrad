"""
glassbox.py — Glass-Box Autograd, First-NaN Root-Cause Localization, and Interactive Visual DAG.

Provides:
- detect_anomaly: Context manager to trap the exact operation that introduces NaNs/Infs during backprop.
- explain_gradients(): Generates a structured ASCII telemetry report of gradient flow across all nodes.
- visualize(): Emits a standalone, zero-dependency interactive HTML/SVG graph with live health color-coding.
- GradientAnomalyError: Custom exception containing root-cause mathematical diagnostic details.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from minigrad.tensor import Tensor
from minigrad.graph import topological_sort


# ── Global Anomaly Detection State ───────────────────────────────────

_anomaly_detection_enabled: bool = False


class detect_anomaly:
    """
    Context manager for strict real-time gradient anomaly trapping.

    When enabled, the backward pass monitors each node execution and catches
    the FIRST transition from finite numbers to NaN or Inf, pinpointing the
    exact operation and input values that caused the numerical poisoning.

    Usage:
        with detect_anomaly():
            loss.backward()
    """

    def __init__(self, check_nans: bool = True, check_infs: bool = True) -> None:
        self.check_nans = check_nans
        self.check_infs = check_infs
        self._prev = False

    def __enter__(self) -> detect_anomaly:
        global _anomaly_detection_enabled
        self._prev = _anomaly_detection_enabled
        _anomaly_detection_enabled = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        global _anomaly_detection_enabled
        _anomaly_detection_enabled = self._prev


def is_anomaly_detection_enabled() -> bool:
    return _anomaly_detection_enabled


# ── Anomaly Diagnostics & Exceptions ────────────────────────────────

class GradientAnomalyError(RuntimeError):
    """Raised when a backward operation introduces NaN or Inf into gradients."""

    def __init__(
        self,
        node_id: int,
        op: str,
        reason: str,
        details: Dict[str, Any],
    ) -> None:
        self.node_id = node_id
        self.op = op
        self.reason = reason
        self.details = details

        header = "[miniGrad Glass-Box] GRADIENT POISONING DETECTED!"
        border = "=" * len(header)
        msg_lines = [
            "",
            border,
            header,
            border,
            f"  Originating Node ID: #{node_id}",
            f"  Mathematical Operation: [{op}]",
            f"  Root Cause Diagnosis:   {reason}",
            "",
            "  Culprit Inspection Telemetry:",
        ]
        for k, v in details.items():
            msg_lines.append(f"    * {k}: {v}")
        msg_lines.extend([
            border,
            "Tip: Inspect the inputs to this operation during the forward pass",
            "or use numerical stabilizers (e.g. clipping, epsilon additions, or log-sum-exp).",
            border,
        ])
        super().__init__("\n".join(msg_lines))


def diagnose_root_cause(node: Tensor, child: Tensor, child_grad_before: np.ndarray) -> str:
    """Diagnose the mathematical cause of a NaN/Inf transition in child."""
    op = node._op
    curr_grad = child.grad

    has_nan = np.isnan(curr_grad).any() and not np.isnan(child_grad_before).any()
    has_inf = np.isinf(curr_grad).any() and not np.isinf(child_grad_before).any()

    if op == "log":
        non_pos = np.sum(child.data <= 0)
        min_val = np.min(child.data)
        return (
            f"Logarithm evaluated on non-positive input ({non_pos} elements <= 0, "
            f"min value: {min_val:.6e}). Gradient d(ln(x))/dx = 1/x produced division by zero or NaN."
        )
    elif op.startswith("pow"):
        p = getattr(node, "_ctx", None)
        zero_bases = np.sum(child.data == 0)
        return (
            f"Power operation with exponent ({p}) evaluated on zero base ({zero_bases} zero elements). "
            f"Gradient d(x^p)/dx = p * x^(p-1) produced negative power on zero (division by zero)."
        )
    elif op == "div":
        # Check if child was denominator
        if len(node._prev) >= 2 and node._prev[1] is child:
            zero_denom = np.sum(child.data == 0)
            return f"Division by zero: denominator tensor contains {zero_denom} zero elements."
        return "Division operation produced unbounded gradient quotient."
    elif op == "exp":
        max_val = np.max(child.data)
        return (
            f"Exponential overflow: input max value is {max_val:.4f}. "
            f"exp(x) gradient exceeded IEEE-754 float64 representation limit."
        )
    elif op in ("sigmoid", "tanh", "softmax"):
        return f"Activation [{op}] gradient overflowed or encountered extreme saturation."
    elif has_nan:
        return f"Operation [{op}] produced NaN gradients from indeterminate form (0 * inf or inf - inf)."
    elif has_inf:
        return f"Operation [{op}] experienced gradient explosion (infinity encountered in backpropagation)."
    return f"Operation [{op}] produced invalid numerical gradients."


# ── Gradient Health Telemetry ────────────────────────────────────────

@dataclass
class NodeTelemetry:
    node_id: int
    op: str
    shape: Tuple[int, ...]
    requires_grad: bool
    l2_norm: float
    min_val: float
    max_val: float
    zero_pct: float
    status: str  # "healthy", "vanishing", "exploding", "poisoned", "inactive"
    reason: Optional[str] = None


def collect_telemetry(root: Tensor) -> Dict[int, NodeTelemetry]:
    """Analyze all nodes in the computation graph and compute gradient health metrics."""
    topo = topological_sort(root)
    telemetry: Dict[int, NodeTelemetry] = {}

    for node in topo:
        node_id = id(node)
        op_name = node._op if node._op else ("leaf" if not node._prev else "intermediate")
        shape = node.data.shape

        if not node.requires_grad:
            telemetry[node_id] = NodeTelemetry(
                node_id=node_id,
                op=op_name,
                shape=shape,
                requires_grad=False,
                l2_norm=0.0,
                min_val=0.0,
                max_val=0.0,
                zero_pct=100.0,
                status="inactive",
            )
            continue

        g = node.grad if isinstance(node.grad, np.ndarray) else node.grad.data

        has_nan = bool(np.isnan(g).any())
        has_inf = bool(np.isinf(g).any())

        if has_nan or has_inf:
            status = "poisoned"
            norm_val = float("nan") if has_nan else float("inf")
            min_val = float("nan")
            max_val = float("nan")
            zero_pct = 0.0
        else:
            norm_val = float(np.linalg.norm(g))
            min_val = float(np.min(g)) if g.size > 0 else 0.0
            max_val = float(np.max(g)) if g.size > 0 else 0.0
            zero_count = int(np.sum(g == 0.0))
            zero_pct = float(100.0 * zero_count / g.size) if g.size > 0 else 0.0

            if norm_val > 1000.0:
                status = "exploding"
            elif norm_val < 1e-6:
                status = "vanishing"
            else:
                status = "healthy"

        telemetry[node_id] = NodeTelemetry(
            node_id=node_id,
            op=op_name,
            shape=shape,
            requires_grad=True,
            l2_norm=norm_val,
            min_val=min_val,
            max_val=max_val,
            zero_pct=zero_pct,
            status=status,
        )

    return telemetry


def explain_gradients(root: Tensor) -> str:
    """
    Generate a clean, structured ASCII report of gradient flow across all nodes in the DAG.
    """
    telemetry = collect_telemetry(root)
    lines: List[str] = []

    header = "miniGrad Glass-Box Gradient Telemetry Report"
    lines.append("=" * 96)
    lines.append(f"{header:^96}")
    lines.append("=" * 96)
    lines.append(
        f"{'Node ID':<10} {'Operation':<16} {'Shape':<16} {'L2 Norm':<14} {'Min / Max':<22} {'Zero %':<10} {'Status':<12}"
    )
    lines.append("-" * 96)

    status_tags = {
        "healthy": "[OK] Healthy",
        "vanishing": "[!] Vanishing",
        "exploding": "[!] Exploding",
        "poisoned": "[X] Poisoned",
        "inactive": "[--] No Grad",
    }

    poisoned_nodes: List[NodeTelemetry] = []

    for item in telemetry.values():
        if not item.requires_grad:
            norm_str = "--"
            minmax_str = "--"
            zero_str = "--"
        elif math.isnan(item.l2_norm):
            norm_str = "NaN"
            minmax_str = "NaN / NaN"
            zero_str = "--"
            poisoned_nodes.append(item)
        elif math.isinf(item.l2_norm):
            norm_str = "Inf"
            minmax_str = "-Inf / +Inf"
            zero_str = "--"
            poisoned_nodes.append(item)
        else:
            norm_str = f"{item.l2_norm:.4e}"
            minmax_str = f"{item.min_val:+.2e} / {item.max_val:+.2e}"
            zero_str = f"{item.zero_pct:.1f}%"

        status_str = status_tags.get(item.status, item.status)
        shape_str = str(item.shape)
        lines.append(
            f"#{item.node_id % 100000:<9} {item.op:<16} {shape_str:<16} {norm_str:<14} {minmax_str:<22} {zero_str:<10} {status_str:<12}"
        )

    lines.append("-" * 96)

    # Diagnostic summary
    if poisoned_nodes:
        lines.append("[!] ROOT CAUSE SUMMARY:")
        for pn in poisoned_nodes:
            lines.append(f"  * Node #{pn.node_id % 100000} [{pn.op}] contains NaN/Inf gradients.")
        lines.append("=" * 96)
    else:
        healthy_count = sum(1 for t in telemetry.values() if t.status == "healthy")
        vanishing_count = sum(1 for t in telemetry.values() if t.status == "vanishing")
        exploding_count = sum(1 for t in telemetry.values() if t.status == "exploding")
        lines.append(
            f"Health Summary: {healthy_count} Healthy [OK] | {vanishing_count} Vanishing [!] | {exploding_count} Exploding [!]"
        )
        lines.append("=" * 96)

    return "\n".join(lines)


# ── Interactive Standalone HTML/SVG Graph Visualizer ────────────────

def visualize(
    root: Tensor,
    filename: Optional[Union[str, Path]] = "computational_graph.html",
) -> str:
    """
    Generate an interactive, standalone HTML/SVG computational graph.

    Zero external dependencies — runs directly in any web browser without Graphviz or internet access.
    Color-codes nodes and edges by gradient health with pan, zoom, and click-to-inspect tooltips.

    Args:
        root:     The root tensor (e.g. loss).
        filename: Optional output HTML path. If None, returns the HTML string without writing.

    Returns:
        The generated HTML content string.
    """
    topo = topological_sort(root)
    telemetry = collect_telemetry(root)

    # 1. Assign topological ranks (layers) from inputs (leaves) to root
    ranks: Dict[int, int] = {}
    for node in topo:
        node_id = id(node)
        if not node._prev:
            ranks[node_id] = 0
        else:
            ranks[node_id] = max(ranks.get(id(p), 0) for p in node._prev) + 1

    # Group nodes by rank
    max_rank = max(ranks.values()) if ranks else 0
    layer_nodes: Dict[int, List[Tensor]] = {r: [] for r in range(max_rank + 1)}
    for node in topo:
        layer_nodes[ranks[id(node)]].append(node)

    # 2. Compute 2D layout coordinates
    node_x: Dict[int, float] = {}
    node_y: Dict[int, float] = {}

    layer_spacing_y = 160.0
    node_spacing_x = 220.0

    max_layer_width = max(len(nodes) for nodes in layer_nodes.values()) if layer_nodes else 1
    canvas_width = max(1100.0, max_layer_width * node_spacing_x + 200.0)
    canvas_height = max(800.0, (max_rank + 1) * layer_spacing_y + 200.0)

    for rank, nodes in layer_nodes.items():
        y = 100.0 + rank * layer_spacing_y
        total_width = (len(nodes) - 1) * node_spacing_x
        start_x = (canvas_width - total_width) / 2.0
        for i, node in enumerate(nodes):
            node_x[id(node)] = start_x + i * node_spacing_x
            node_y[id(node)] = y

    # 3. Generate SVG elements
    status_colors = {
        "healthy": "#10B981",    # Emerald green
        "vanishing": "#F59E0B",  # Amber yellow
        "exploding": "#EF4444",  # Crimson red
        "poisoned": "#9333EA",   # Purple skull
        "inactive": "#6B7280",   # Cool gray
    }

    svg_edges: List[str] = []
    for node in topo:
        target_id = id(node)
        tx, ty = node_x[target_id], node_y[target_id]
        t_status = telemetry[target_id].status
        edge_color = status_colors.get(t_status, "#9CA3AF")

        for parent in node._prev:
            parent_id = id(parent)
            if parent_id in node_x:
                px, py = node_x[parent_id], node_y[parent_id]
                # Smooth cubic bezier curve
                ctrl_y1 = py + 70.0
                ctrl_y2 = ty - 70.0
                path_d = f"M {px} {py} C {px} {ctrl_y1}, {tx} {ctrl_y2}, {tx} {ty}"
                svg_edges.append(
                    f'<path d="{path_d}" stroke="{edge_color}" stroke-width="2.5" fill="none" opacity="0.75" />'
                )

    svg_nodes: List[str] = []
    node_metadata_json: Dict[str, Any] = {}

    for node in topo:
        nid = id(node)
        x, y = node_x[nid], node_y[nid]
        t = telemetry[nid]
        color = status_colors.get(t.status, "#9CA3AF")
        short_id = f"#{nid % 100000}"
        op_label = t.op[:14]

        # Prepare node metadata for interactive click panel
        node_metadata_json[str(nid)] = {
            "id": short_id,
            "full_id": nid,
            "op": t.op,
            "shape": list(t.shape),
            "requires_grad": t.requires_grad,
            "status": t.status,
            "norm": f"{t.l2_norm:.4e}" if not math.isnan(t.l2_norm) else "NaN",
            "min_val": f"{t.min_val:.4e}" if not math.isnan(t.min_val) else "NaN",
            "max_val": f"{t.max_val:.4e}" if not math.isnan(t.max_val) else "NaN",
            "zero_pct": f"{t.zero_pct:.1f}%",
            "data_sample": str(node.data.flatten()[:5].tolist()) if node.data.size > 0 else "[]",
            "grad_sample": str(node.grad.flatten()[:5].tolist()) if isinstance(node.grad, np.ndarray) and node.grad.size > 0 else "[]",
        }

        # Node card SVG representation
        rect_w = 170.0
        rect_h = 75.0
        rx = x - rect_w / 2.0
        ry = y - rect_h / 2.0

        svg_nodes.append(f"""
        <g class="graph-node" data-node-id="{nid}" style="cursor: pointer;" transform="translate({rx}, {ry})">
            <rect width="{rect_w}" height="{rect_h}" rx="12" fill="#1E293B" stroke="{color}" stroke-width="3" filter="drop-shadow(0 4px 6px rgba(0,0,0,0.3))" />
            <circle cx="20" cy="22" r="6" fill="{color}" />
            <text x="35" y="26" fill="#F8FAFC" font-size="13" font-family="system-ui, sans-serif" font-weight="700">{op_label}</text>
            <text x="20" y="46" fill="#94A3B8" font-size="11" font-family="monospace">shape: {t.shape}</text>
            <text x="20" y="62" fill="{color}" font-size="11" font-family="system-ui, sans-serif" font-weight="600">{t.status.upper()}</text>
            <text x="{rect_w - 15}" y="25" fill="#64748B" font-size="10" font-family="monospace" text-anchor="end">{short_id}</text>
        </g>
        """)

    # 4. Generate the standalone HTML template
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>miniGrad Glass-Box Computational Graph</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ background: #0F172A; color: #F8FAFC; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; overflow: hidden; height: 100vh; display: flex; flex-direction: column; }}
        header {{ background: #1E293B; border-bottom: 1px solid #334155; padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; z-index: 10; }}
        .title-group {{ display: flex; align-items: center; gap: 12px; }}
        h1 {{ font-size: 18px; font-weight: 700; color: #38BDF8; }}
        .badge {{ background: #0284C7; color: white; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; }}
        .legend {{ display: flex; gap: 16px; font-size: 12px; align-items: center; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
        #container {{ flex: 1; position: relative; overflow: hidden; cursor: grab; }}
        #container:active {{ cursor: grabbing; }}
        svg {{ width: 100%; height: 100%; }}
        .controls {{ position: absolute; bottom: 20px; left: 20px; display: flex; gap: 8px; z-index: 20; }}
        button {{ background: #1E293B; color: #F8FAFC; border: 1px solid #475569; padding: 8px 14px; border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 600; }}
        button:hover {{ background: #334155; }}
        #inspector {{ position: absolute; top: 70px; right: 20px; width: 340px; background: #1E293B; border: 1px solid #334155; border-radius: 12px; padding: 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); display: none; z-index: 30; }}
        #inspector h2 {{ font-size: 16px; margin-bottom: 12px; color: #38BDF8; display: flex; justify-content: space-between; align-items: center; }}
        .prop-row {{ display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #334155; font-size: 13px; }}
        .prop-label {{ color: #94A3B8; }}
        .prop-val {{ font-family: monospace; font-weight: 600; }}
        .sample-box {{ background: #0F172A; padding: 8px; border-radius: 6px; font-family: monospace; font-size: 11px; margin-top: 6px; word-break: break-all; color: #CBD5E1; }}
        .close-btn {{ cursor: pointer; color: #94A3B8; font-size: 16px; font-weight: bold; }}
    </style>
</head>
<body>
    <header>
        <div class="title-group">
            <h1>miniGrad Glass-Box DAG</h1>
            <span class="badge">{len(topo)} Nodes</span>
        </div>
        <div class="legend">
            <div class="legend-item"><span class="dot" style="background:#10B981"></span>Healthy</div>
            <div class="legend-item"><span class="dot" style="background:#F59E0B"></span>Vanishing (&lt;1e-6)</div>
            <div class="legend-item"><span class="dot" style="background:#EF4444"></span>Exploding (&gt;1e3)</div>
            <div class="legend-item"><span class="dot" style="background:#9333EA"></span>Poisoned (NaN/Inf)</div>
            <div class="legend-item"><span class="dot" style="background:#6B7280"></span>No Grad</div>
        </div>
    </header>

    <div id="container">
        <svg id="viewport" viewBox="0 0 {canvas_width} {canvas_height}">
            <g id="scene">
                <!-- Edges -->
                <g id="edges">
                    {"".join(svg_edges)}
                </g>
                <!-- Nodes -->
                <g id="nodes">
                    {"".join(svg_nodes)}
                </g>
            </g>
        </svg>

        <div class="controls">
            <button onclick="zoom(1.2)">Zoom +</button>
            <button onclick="zoom(0.8)">Zoom -</button>
            <button onclick="resetZoom()">Reset</button>
        </div>

        <div id="inspector">
            <h2><span id="inspect-title">Node Info</span> <span class="close-btn" onclick="closeInspector()">×</span></h2>
            <div class="prop-row"><span class="prop-label">Operation:</span><span class="prop-val" id="inspect-op">--</span></div>
            <div class="prop-row"><span class="prop-label">Shape:</span><span class="prop-val" id="inspect-shape">--</span></div>
            <div class="prop-row"><span class="prop-label">Requires Grad:</span><span class="prop-val" id="inspect-grad">--</span></div>
            <div class="prop-row"><span class="prop-label">Status:</span><span class="prop-val" id="inspect-status">--</span></div>
            <div class="prop-row"><span class="prop-label">L2 Norm:</span><span class="prop-val" id="inspect-norm">--</span></div>
            <div class="prop-row"><span class="prop-label">Min / Max Grad:</span><span class="prop-val" id="inspect-minmax">--</span></div>
            <div class="prop-row"><span class="prop-label">Zero Neurons:</span><span class="prop-val" id="inspect-zeros">--</span></div>
            <div style="margin-top: 10px;">
                <span class="prop-label" style="font-size: 11px;">Forward Sample (first 5):</span>
                <div class="sample-box" id="inspect-sample-data">--</div>
            </div>
            <div style="margin-top: 8px;">
                <span class="prop-label" style="font-size: 11px;">Gradient Sample (first 5):</span>
                <div class="sample-box" id="inspect-sample-grad">--</div>
            </div>
        </div>
    </div>

    <script>
        const metadata = {json.dumps(node_metadata_json)};
        let scale = 1.0;
        let panX = 0;
        let panY = 0;
        let isDragging = false;
        let startX, startY;

        const scene = document.getElementById('scene');
        const container = document.getElementById('container');
        const inspector = document.getElementById('inspector');

        function updateTransform() {{
            scene.setAttribute('transform', `translate(${{panX}}, ${{panY}}) scale(${{scale}})`);
        }}

        function zoom(factor) {{
            scale *= factor;
            updateTransform();
        }}

        function resetZoom() {{
            scale = 1.0;
            panX = 0;
            panY = 0;
            updateTransform();
        }}

        container.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const factor = e.deltaY < 0 ? 1.1 : 0.9;
            scale *= factor;
            updateTransform();
        }});

        container.addEventListener('mousedown', (e) => {{
            if (e.target.closest('#inspector') || e.target.closest('.controls')) return;
            isDragging = true;
            startX = e.clientX - panX;
            startY = e.clientY - panY;
        }});

        window.addEventListener('mousemove', (e) => {{
            if (!isDragging) return;
            panX = e.clientX - startX;
            panY = e.clientY - startY;
            updateTransform();
        }});

        window.addEventListener('mouseup', () => {{
            isDragging = false;
        }});

        // Node click inspect handler
        document.querySelectorAll('.graph-node').forEach(node => {{
            node.addEventListener('click', (e) => {{
                e.stopPropagation();
                const nid = node.getAttribute('data-node-id');
                const data = metadata[nid];
                if (!data) return;

                document.getElementById('inspect-title').textContent = `${{data.op}} (${{data.id}})`;
                document.getElementById('inspect-op').textContent = data.op;
                document.getElementById('inspect-shape').textContent = JSON.stringify(data.shape);
                document.getElementById('inspect-grad').textContent = data.requires_grad;
                document.getElementById('inspect-status').textContent = data.status.toUpperCase();
                document.getElementById('inspect-norm').textContent = data.norm;
                document.getElementById('inspect-minmax').textContent = `${{data.min_val}} / ${{data.max_val}}`;
                document.getElementById('inspect-zeros').textContent = data.zero_pct;
                document.getElementById('inspect-sample-data').textContent = data.data_sample;
                document.getElementById('inspect-sample-grad').textContent = data.grad_sample;

                inspector.style.display = 'block';
            }});
        }});

        function closeInspector() {{
            inspector.style.display = 'none';
        }}
    </script>
</body>
</html>"""

    if filename is not None:
        out_path = Path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_template)

    return html_template
