# miniGrad

[![CI](https://github.com/Eternalcodertanishq3/minigrad/actions/workflows/ci.yml/badge.svg)](https://github.com/Eternalcodertanishq3/minigrad/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/minigrad.svg)](https://pypi.org/project/minigrad/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> A deep learning framework built from scratch in Python/NumPy. Autograd engine, neural net layers, optimizers, and a trained MNIST model — zero dependencies except NumPy.

**Why this exists:** Anyone can call `model.fit()`. Almost nobody can explain what happens inside `.backward()`. This project proves you can. That's the difference between a library user and a framework builder.

## What You Get

| Component | What It Does |
|-----------|-------------|
| **Autograd Engine** | Dynamic computation graphs, topological sort, chain rule — the heart of PyTorch |
| **Tensor Engine** | N-dimensional arrays with gradient tracking and broadcasting |
| **Layer Library** | Linear, Conv2D (im2col), BatchNorm, Dropout — all built manually |
| **Optimizer Suite** | SGD + Momentum, RMSProp, Adam + AdamW — from scratch |
| **Loss Functions** | MSE, CrossEntropy, BCE — all differentiable |
| **MNIST Demo** | CNN trained to >98% accuracy using only this framework |

## Quick Start

```bash
pip install numpy matplotlib  # only dependencies
python examples/01_scalar_autograd.py    # See autograd in action
python examples/02_linear_regression.py  # Learn y = mx + b
python examples/03_mlp_xor.py            # MLP solves XOR
python examples/04_mnist_mlp.py          # MNIST with dense layers
python examples/05_mnist_cnn.py          # MNIST with CNN (~99%)
```

## Example Usage

```python
from minigrad import Tensor
from minigrad.nn import Sequential, Linear, ReLU
from minigrad.optim import Adam

# Define a model
model = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(128, 10),
])

optimizer = Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):
    for batch_x, batch_y in dataloader:
        x = Tensor(batch_x)
        logits = model(x)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Architecture

```
minigrad/
├── tensor.py          # Tensor class + autograd engine
├── ops.py             # Math operations (add, matmul, relu...)
├── graph.py           # Computation graph + topo sort
├── nn/
│   ├── module.py      # Base Module class
│   ├── linear.py      # Fully connected layer
│   ├── conv.py        # Conv2D via im2col
│   ├── activations.py # ReLU, Sigmoid, Tanh, GELU
│   ├── batchnorm.py   # BatchNorm1D/2D
│   ├── dropout.py     # Dropout regularization
│   ├── loss.py        # MSE, CrossEntropy, BCE
│   └── sequential.py  # Layer container
├── optim/
│   ├── sgd.py         # SGD + momentum + weight decay
│   ├── rmsprop.py     # RMSProp
│   └── adam.py        # Adam + AdamW
├── data/
│   ├── dataset.py     # Dataset base + MNIST loader
│   ├── dataloader.py  # Batch sampling
│   └── transforms.py  # Normalize, ToTensor
└── utils.py           # grad_check, plotting, helpers
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **NumPy only** | Zero external dependencies; runs anywhere |
| **Dynamic graphs** | Build computation graph on each forward pass (like PyTorch) |
| **Topological sort** | Guarantees correct gradient accumulation order |
| **im2col for Conv2D** | Same algorithm as cuDNN/PyTorch — convolution becomes matmul |
| **Kaiming He init** | Essential for training stability in deep networks |
| **Bias correction in Adam** | Critical for accurate early-step updates |

## PyTorch Parity

Every operation is tested against PyTorch to 1e-6 precision:

```bash
pytest tests/test_ops.py      # Operation-level parity
pytest tests/test_layers.py   # Layer forward/backward parity
pytest tests/test_optim.py    # Optimizer step parity
pytest tests/test_grad_check.py  # Numerical gradient verification
```

## Training Results

| Model | Dataset | Accuracy | Epochs |
|-------|---------|----------|--------|
| MLP (784-256-128-10) | MNIST | ~97% | 5 |
| CNN (3 conv + 2 linear) | MNIST | ~99% | 10 |

## Benchmarks

See [BENCHMARKS.md](benchmarks/BENCHMARKS.md) for detailed performance comparisons against NumPy baseline.

miniGrad is **not designed for speed** — it's designed for **understanding**.
For production use, use PyTorch (C++/CUDA) or JAX (XLA).

## Core Concepts You'll Master

- Backpropagation via chain rule
- Dynamic computation graphs
- Topological sort for correct gradient flow
- im2col algorithm for efficient convolution
- Kaiming initialization and training stability
- Adam optimizer with bias correction
- BatchNorm train/eval mode switching
- Numerical gradient verification

## License

MIT License — use it, learn from it, build on it.

---

*Built with zero ML frameworks and a lot of curiosity.*
