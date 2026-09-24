"""
miniGrad — A deep learning framework built from scratch.

Zero dependencies except NumPy. Provides:
- Autograd engine with dynamic computation graphs
- Neural network layers (Linear, Conv2D, BatchNorm, etc.)
- Optimizers (SGD, RMSprop, Adam)
- Loss functions (MSE, CrossEntropy, BCE)
- Data loading utilities

Usage:
    from minigrad import Tensor
    from minigrad.nn import Sequential, Linear, ReLU
    from minigrad.optim import Adam

    model = Sequential([Linear(784, 128), ReLU(), Linear(128, 10)])
    optimizer = Adam(model.parameters(), lr=1e-3)
"""

__version__ = "1.0.0"

from minigrad.tensor import Tensor
from minigrad.graph import topological_sort, trace, print_graph
from minigrad.safetensors import save_file, load_file, safe_open
from minigrad.autograd import grad, hessian
from minigrad.glassbox import (
    detect_anomaly,
    explain_gradients,
    visualize,
    collect_telemetry,
    GradientAnomalyError,
    NodeTelemetry,
)
from minigrad.compiler import export_c, to_c
from minigrad.graph_opt import optimize, optimize_graph, OptimizationReport
from minigrad.vmap import (
    vmap,
    make_functional,
    per_sample_gradients,
    jacrev,
    batched_jacobian,
)
from minigrad.dp import (
    clip_per_sample_gradients,
    add_dp_noise,
    apply_dp_gradients,
    compute_dp_sgd_step,
    PrivacyTelemetry,
)
from minigrad.sutra import (
    SUTRA,
    NeuralODE,
    odeint,
    AdaptiveStepTelemetry,
)

__all__ = [
    "Tensor",
    "topological_sort",
    "trace",
    "print_graph",
    "save_file",
    "load_file",
    "safe_open",
    "grad",
    "hessian",
    "detect_anomaly",
    "explain_gradients",
    "visualize",
    "collect_telemetry",
    "GradientAnomalyError",
    "NodeTelemetry",
    "export_c",
    "to_c",
    "optimize",
    "optimize_graph",
    "OptimizationReport",
    "vmap",
    "make_functional",
    "per_sample_gradients",
    "jacrev",
    "batched_jacobian",
    "clip_per_sample_gradients",
    "add_dp_noise",
    "apply_dp_gradients",
    "compute_dp_sgd_step",
    "PrivacyTelemetry",
    "SUTRA",
    "NeuralODE",
    "odeint",
    "AdaptiveStepTelemetry",
    "__version__",
]
