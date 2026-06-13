# miniGrad

[![CI](https://github.com/Eternalcodertanishq3/minigrad/actions/workflows/ci.yml/badge.svg)](https://github.com/Eternalcodertanishq3/minigrad/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/minigrad-framework.svg)](https://pypi.org/project/minigrad-framework/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

miniGrad is a NumPy-first deep learning framework built from scratch. It is designed to make the internals of modern autograd systems visible: tensors, dynamic computation graphs, neural-network layers, optimizers, losses, data loading, examples, and PyTorch parity tests.

The project is now positioned as a serious educational and research-grade NumPy framework. It is intentionally more complete than scalar-first projects like micrograd, more neural-network focused than general NumPy autograd tools, and structured with a roadmap toward tinygrad-style backend/runtime ideas.

## What miniGrad Provides

| Area | Included |
| --- | --- |
| Autograd | Dynamic computation graphs, reverse-mode autodiff, broadcasting-aware gradients |
| Tensor ops | Add, subtract, multiply, divide, matmul, reductions, reshape, transpose, indexing |
| Activations | ReLU, Sigmoid, Tanh, GELU, Softmax, LeakyReLU, ELU |
| Layers | Linear, Conv2D via im2col, BatchNorm1D/2D, Dropout, Dropout2D, Flatten, Sequential |
| Losses | MSE, CrossEntropy, BCE, BCEWithLogits, NLL |
| Optimizers | SGD with momentum/Nesterov, RMSProp, Adam, AdamW |
| Data | Dataset abstraction, DataLoader, MNIST loader, transforms |
| Tooling | CLI entry points, benchmarks, CI, Ruff, Mypy, PyTorch parity tests |

## Install

```bash
pip install -e ".[dev,torch]"
```

Runtime dependency:

```bash
pip install numpy
```

Optional development dependencies include `pytest`, `pytest-cov`, `matplotlib`, `ruff`, `mypy`, and `torch` for parity tests.

## Quick Start

```python
from minigrad import Tensor
from minigrad.nn import Sequential, Linear, ReLU, CrossEntropyLoss
from minigrad.optim import Adam

model = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(128, 10),
])

optimizer = Adam(model.parameters(), lr=1e-3)
criterion = CrossEntropyLoss()

x = Tensor(batch_x.reshape(batch_x.shape[0], -1))
loss = criterion(model(x), batch_y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## CLI

```bash
minigrad scalar
minigrad linear-regression
minigrad xor
minigrad bench-ops
```

Direct scripts are also available:

```bash
python examples/01_scalar_autograd.py
python examples/02_linear_regression.py
python examples/03_mlp_xor.py
python examples/04_mnist_mlp.py
python examples/05_mnist_cnn.py
```

The MNIST examples download data on first run. CI uses smaller smoke paths and unit tests so it does not depend on network downloads.

## Verification

Current local verification includes:

```bash
python -m pytest
python -m ruff check minigrad tests examples
python -m mypy minigrad --ignore-missing-imports
python examples/01_scalar_autograd.py
python examples/02_linear_regression.py
python examples/03_mlp_xor.py
```

The test suite covers core ops, numerical gradient checks, PyTorch parity for major layers and optimizers, public API exports, and regressions for previously fragile paths such as Conv2D gradients with non-gradient input batches, BatchNorm gradient propagation, and negative-axis reductions.

## Compared With Other Projects

| Project | Focus | How miniGrad Differs |
| --- | --- | --- |
| micrograd | Tiny scalar autograd engine | miniGrad uses N-dimensional tensors, neural-network layers, optimizers, data utilities, tests, and examples. |
| HIPS Autograd | General automatic differentiation for NumPy code | miniGrad is a teaching-oriented neural-network framework with explicit modules, optimizers, losses, and training loops. |
| tinygrad | Small but serious tensor framework with backend/runtime ambitions | miniGrad is currently NumPy/eager-first, with a roadmap for backend boundaries, devices, lazy tracing, and runtime work. |
| PyTorch/JAX | Production ML ecosystems | miniGrad is for learning, inspection, and small experiments, not large-scale production deployment. |

## Production Readiness

miniGrad is production-quality for educational NumPy experiments when:

- correctness is covered by tests,
- public APIs are importable from an installed package,
- CI fails on lint, type, test, and packaging errors,
- claims in docs match verified behavior.

It is not yet a production replacement for PyTorch, JAX, or tinygrad. The next major step toward tinygrad-class competitiveness is a backend/device abstraction, followed by lazy graph tracing and backend-specific kernels.

## Architecture

```text
minigrad/
  tensor.py          Tensor class and reverse-mode autograd
  ops.py             Functional math operations
  graph.py           Graph traversal and debugging helpers
  cli.py             Console entry points
  nn/
    module.py        Base Module class
    linear.py        Fully connected layer
    conv.py          Conv2D via im2col/col2im
    activations.py   Activation modules
    batchnorm.py     BatchNorm1D and BatchNorm2D
    dropout.py       Dropout and Dropout2D
    loss.py          Differentiable losses
    sequential.py    Sequential container
  optim/
    sgd.py           SGD, momentum, Nesterov, weight decay
    rmsprop.py       RMSProp
    adam.py          Adam and AdamW
  data/
    dataset.py       Dataset and MNISTDataset
    dataloader.py    Mini-batch loading
    transforms.py    Preprocessing transforms
  utils.py           Grad check, plotting, serialization helpers
```

## Roadmap

1. Expand test coverage for all public operations, layers, losses, and data utilities.
2. Add package build/install checks to CI.
3. Add model and optimizer `state_dict` APIs.
4. Introduce a NumPy backend boundary and explicit device labels.
5. Add lazy graph tracing and benchmark-driven runtime experiments.
6. Explore CPU kernel specialization before any GPU claims.

## License

MIT License. Use it, learn from it, and build on it.
