<div align="center">

<img src="https://raw.githubusercontent.com/Eternalcodertanishq3/minigrad/main/assets/logo.svg" width="140" height="140" alt="miniGrad Logo" />

# miniGrad (तर्क · सूत्र · स्पन्द)

[![CI](https://github.com/Eternalcodertanishq3/minigrad/actions/workflows/ci.yml/badge.svg)](https://github.com/Eternalcodertanishq3/minigrad/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/minigrad-framework.svg?color=blue)](https://pypi.org/project/minigrad-framework/)
[![Total Downloads](https://img.shields.io/pepy/dt/minigrad-framework?color=blue&label=total%20downloads)](https://pepy.tech/project/minigrad-framework)
[![Monthly Downloads](https://img.shields.io/pypi/dm/minigrad-framework.svg?color=blue&label=downloads%2Fmonth)](https://pypi.org/project/minigrad-framework/)
[![Tests](https://img.shields.io/badge/tests-583%20collected%20%7C%20545%20zero--dep-brightgreen.svg)](tests/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![NumPy only](https://img.shields.io/badge/dependencies-numpy%20only-red.svg)](pyproject.toml)

**A First-Principles Deep Learning & Autograd Framework with Reference Scientific Computing Paradigms.**

*Continuous-Depth Neural ODEs · Zero-Memory Reversible Computing · Analytical Uncertainty Tensors · Neuro-Symbolic Logic · Neuromorphic Spiking Dynamics · Embedded C Compiler*

[Quick Start](#-quick-start) • [Component Maturity](#-component-maturity--status-matrix) • [The 4 Pillars](#-the-4-foundational-pillars) • [The 5 Reference Paradigms](#-the-5-reference-paradigms) • [Specifications](docs/SPECIFICATIONS.md) • [Examples](#-runnable-examples) • [Verification](#-verification)

</div>

---

## 🌟 Overview: Beyond the PyTorch Shadow

Most autograd engines in the open-source ecosystem fall into one of two camps:
1. **Toy educational clones** (like `micrograd`) that can only handle basic scalar operations or small toy MLPs.
2. **Framework wrappers** that inherit PyTorch's fundamental assumptions: *"Everything is a dense tensor, layers must be discrete, activations must consume $\mathcal{O}(L)$ RAM, and numbers are deterministic with zero knowledge of their own doubt."*

**miniGrad is built from mathematical first principles and depends only on NumPy.** It implements full reverse-mode automatic differentiation, modern transformers (miniGPT, LoRA), higher-order Hessians (PINNs), and reference implementations of **five scientific computing paradigms** in readable, type-checked Python.

---

## 🏛️ Architecture: The 3 Layers of miniGrad

```
                                  THE MINIGRAD UNIFIED ENGINE
  
  LAYER 3: THE 5 REFERENCE PARADIGMS (Scientific Computing)
  ├── S.U.T.R.A.    : Continuous-Depth Neural ODEs (O(1) Graph-Node Pontryagin Adjoint Autograd)
  ├── A.V.Y.A.Y.A.  : Reversible Computing (Zero Forward Caching, O(1) Activation RAM for 500 Layers)
  ├── P.R.A.M.A.N.A.: Distributional Uncertainty Tensors (Single-Layer-Exact Affine + Delta-Method Moment Autograd)
  ├── T.A.R.K.A.    : Neuro-Symbolic Differentiable Logic (Continuous t-Norms & Axiomatic Semantic Loss)
  └── S.P.A.N.D.A.  : Neuromorphic Event-Driven SNNs (LIF Dynamics & Surrogate-Gradient BPTT)
  
  LAYER 2: THE 4 FOUNDATIONAL PILLARS (Modern Systems & Engineering Dominance)
  ├── Pillar 1: Glass-Box Autograd Engine (Root-Cause First-NaN Diagnosis, ASCII Visualizer, Telemetry)
  ├── Pillar 2: Embedded C99 Compiler (Zero-Runtime C Code Generation with OpenMP SIMD Parallelism)
  ├── Pillar 3: Symbolic Graph Optimization (Algebraic Rewrites, Constant Folding & Kernel Fusion)
  └── Pillar 4: Pure Functional vmap & DP-SGD (Batched Jacobians & Differentially Private Training)
  
  LAYER 1: THE CORE DEEP LEARNING FOUNDATION (Complete Framework Capabilities)
  ├── High-Precision Autograd & Double Backward (Hessians, PINN PDE Solvers, create_graph=True)
  ├── Transformer & Attention Stack (MultiHeadAttention, miniGPT Language Model, LoRA Fine-Tuning)
  ├── Production Neural Layers (Linear, Conv2D, BatchNorm1D/2D, LayerNorm, Dropout, Embedding)
  └── Complete Loss & Optimizer Suite (Adam, AdamW, RMSprop, SGD, CrossEntropy, BCE, SafeTensors)
```

---

## 🚦 Component Maturity & Status Matrix

miniGrad maintains explicit maturity boundaries between verified core subsystems and research prototypes:

| Subsystem | Module | Maturity | Verification & Test Coverage |
| :--- | :--- | :---: | :--- |
| **Core Autograd DAG** | `minigrad.tensor`, `minigrad.autograd` | **Stable** | Exact mathematical differentiation, iterative topological sort, reference-cycle-free `LIVE`/`FREED` lifecycle, 1D/2D/3D+ batched `matmul`, higher-order derivatives (`create_graph=True` across core ops, `gelu`, `einsum`, `split`, `pad`, `layer_norm`, `embedding`, `dropout`, and losses; `conv2d`, `batch_norm` and `surrogate_spike` have no second-order rule and raise a clear `NotImplementedError`). |
| **Glass-Box Telemetry** | `minigrad.glassbox` | **Stable** | Forward & backward first-NaN/Inf diagnosis with call-stack snapshot attribution, interactive ASCII DAG visualization. |
| **Neural Layers & Optimizers** | `minigrad.nn`, `minigrad.optim` | **Stable** | Linear, Conv2D, BatchNorm1D/2D, LayerNorm, Dropout, MultiHeadAttention, SGD, Adam, AdamW, RMSprop with PyTorch parity. |
| **Symbolic Graph Optimizer** | `minigrad.graph_opt` | **Hardened** | 100-DAG property-based randomized stress testing, canonical reconstructor registry, 3-layer differential validation, algebraic folding, kernel fusion. |
| **Embedded C Compiler** | `minigrad.compiler` | **Hardened** | Compiles feed-forward/elementwise DAGs (`Linear`, `MatMul`, `Add`, `Mul`, `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `Sin`, `Cos`, `Abs`, `Exp`, `Log`, `Softmax`, `LayerNorm`, `RMSNorm`) with 3-way differential validation (Python == FP32 C ≈ INT8 C) and static arena reuse. |
| **SafeTensors Serialization** | `minigrad.safetensors` | **Stable** | Option B mmap-assisted loading with owned numpy array copies, bfloat16 parsing, HuggingFace format interoperability. |
| **Functional vmap & DP-SGD** | `minigrad.vmap`, `minigrad.dp` | **Experimental** | Ergonomic slice-mapped per-sample gradient/Jacobian transforms, per-sample gradient clipping, Rényi differential privacy accounting. |
| **S.U.T.R.A. (Neural ODEs)** | `minigrad.sutra` | **Research** | Pontryagin continuous adjoint sensitivity, unrolled trajectory differentiation, adaptive step Dormand-Prince (`dopri5`) integrator with strict `max_steps` bounds. |
| **A.V.Y.A.Y.A. (Reversible Nets)** | `minigrad.avyaya` | **Research** | Bipartite additive coupling (`RevNet`), reverse reconstruction with $\mathcal{O}(1)$ live forward activations and measured memory telemetry. |
| **P.R.A.M.A.N.A. (Distributional)**| `minigrad.pramana` | **Research** | Exact single-layer affine/bilinear moment propagation (independent inputs; stacked layers are approximate), first-order Taylor (delta-method) non-linear variance approximations, and `HeteroscedasticMLP`. |
| **T.A.R.K.A. (Neuro-Symbolic)** | `minigrad.tarka` | **Research** | Continuous t-norms (`Product`, `Lukasiewicz`, `Godel` with exact Gödel residuum) and differentiable semantic loss. |
| **S.P.A.N.D.A. (Neuromorphic SNN)**| `minigrad.spanda` | **Research** | Leaky Integrate-and-Fire (`LIFCell`) temporal dynamics with surrogate gradient BPTT and single-pass ANN vs. $T$-step SNN SynOps profiling. |

---

## 📊 Head-to-Head Comparison

| Capability / Metric | `miniGrad` | `PyTorch` | `micrograd` / `tinygrad` |
| :--- | :---: | :---: | :---: |
| **Dependencies** | **Zero (Pure NumPy)** | ~2.5 GB C++/CUDA binaries | Pure Python / minimal C |
| **Full Test Suite** | **583 tests collected (545 zero-dep CI + 38 parity)** | Minutes / Hours | Few dozen tests |
| **Glass-Box Root-Cause NaN Debugger** | **Native Built-in** | `detect_anomaly` (slow) | ❌ None |
| **Zero-Runtime C Code Generator** | **Native (`export_c`)** | TorchScript / ExecuTorch | TinyGrad has C-gen |
| **Symbolic Graph Optimization & Fusion** | **Native Built-in** | TorchDynamo / Inductor | TinyGrad has fusion |
| **Pure Functional `vmap` & DP-SGD** | **Native Built-in** | `functorch` / `opacus` (separate) | ❌ None |
| **Higher-Order PINN PDE Solver** | **Native Built-in** | Yes | ❌ None |
| **Continuous-Depth Neural ODEs (S.U.T.R.A.)** | **Native ($O(1)$ Adjoint)** | Requires `torchdiffeq` (external) | ❌ None |
| **$O(1)$ Memory Reversible Layers (A.V.Y.A.Y.A.)** | **Native (500 Layers in $O(1)$ RAM)** | Custom external implementations | ❌ None |
| **Dual-Stream Analytical Uncertainty (P.R.A.M.A.N.A.)**| **Native (Single-Pass Moments)** | Requires 100x Monte Carlo passes | ❌ None |
| **Neuro-Symbolic Differentiable Logic (T.A.R.K.A.)** | **Native (Continuous t-Norms)** | Requires DeepProbLog / LTN (external) | ❌ None |
| **Neuromorphic Spiking Dynamics (S.P.A.N.D.A.)** | **Native (Surrogate BPTT)** | Requires `snnTorch` / `SpikingJelly` | ❌ None |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Eternalcodertanishq3/minigrad.git
cd minigrad

# Install in editable mode
pip install -e .

# Or install with development dependencies (pytest, ruff, mypy)
pip install -e ".[dev]"
```

Runtime requirement: `numpy >= 1.24.0` (the only runtime dependency).

---

### Basic Neural Network Training

```python
from minigrad import Tensor
from minigrad.nn import Sequential, Linear, ReLU, CrossEntropyLoss
from minigrad.optim import Adam

# Build model
model = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(128, 10),
])

optimizer = Adam(model.parameters(), lr=1e-3)
criterion = CrossEntropyLoss()

# Forward pass
logits = model(Tensor(x_batch))
loss = criterion(logits, y_batch)

# Backward pass & parameter update
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 🔬 The 5 Reference Paradigms
> For formal mathematical formulations, computational invariants, known approximations, and validation standards across all research modules, see [docs/SPECIFICATIONS.md](docs/SPECIFICATIONS.md).

### 1. S.U.T.R.A. (Continuous-Depth Neural ODEs)
*Sanskrit: सूत्र (Thread / Continuous Continuity)*  
**Symbolic-Unified Trajectory & Continuous-Time Residual Autograd with $\mathcal{O}(1)$ Pontryagin Adjoint Graph.**

Instead of stacking discrete layers ($L_1 \to L_2 \to L_3$), S.U.T.R.A. models hidden state evolution as a continuous differential equation (following Chen et al., 2018):
$$\frac{dz}{dt} = f_\theta(z(t), t)$$
By solving the continuous adjoint state $a(t) = \frac{\partial \mathcal{L}}{\partial z(t)}$ in reverse time, the Pontryagin Adjoint formulation maintains an $\mathcal{O}(1)$ computation graph node count invariant with respect to ODE integration steps (eliminating the $\mathcal{O}(N_{\text{steps}})$ autograd graph node explosion of discrete unrolling).

```python
from minigrad import SUTRA, NeuralODE, Tensor
from minigrad.nn import Sequential, Linear, Tanh

# Define continuous vector field dz/dt = f(z, t)
func = Sequential([Linear(2, 32), Tanh(), Linear(32, 2)])
ode_model = NeuralODE(func, t_span=(0.0, 1.0), solver="dopri5", rtol=1e-4, atol=1e-5)

# Forward continuous integration
z_final = ode_model(Tensor(z0))

# O(1) graph-node backward pass via Pontryagin Adjoint
loss = z_final.sum()
loss.backward()
```
*Run showcase:* `python examples/12_sutra_neural_ode.py`

---

### 2. A.V.Y.A.Y.A. (Reversible Invertible Computing)
*Sanskrit: अव्यय (Imperishable / Information-Lossless)*  
**Adaptive Volume-preserving Yield-lossless Activation-inverting Y-reconstruction Autograd.**

Standard deep networks cache every intermediate activation in RAM, causing an $\mathcal{O}(L \times B \times D)$ memory growth. A.V.Y.A.Y.A. implements bipartite additive coupling blocks (Gomez et al., 2017 / RevNet):
$$y_1 = x_1 + f(x_2), \quad y_2 = x_2 + g(y_1)$$
Which are algebraically invertible:
$$x_2 = y_2 - g(y_1), \quad x_1 = y_1 - f(x_2)$$
Single-block reconstruction achieves **machine precision ($\sim 2.8 \times 10^{-16}$ error)**, with cumulative IEEE-754 floating-point drift of $\sim 10^{-5}$ across 500 unrolled layers. Forward passes discard intermediate activations, and the backward pass dynamically reconstructs inputs on the fly—maintaining $\mathcal{O}(1)$ live activation tensors regardless of network depth.

```python
from minigrad import AVYAYA, ReversibleBlock, ReversibleSequential, Tensor
from minigrad.nn import Linear, GELU, Sequential

# Define arbitrary sub-networks
f = Sequential([Linear(32, 64), GELU(), Linear(64, 32)])
g = Sequential([Linear(32, 64), GELU(), Linear(64, 32)])

# 50-layer reversible model with 0 forward activation caching
layers = [ReversibleBlock(f, g) for _ in range(25)]
model = ReversibleSequential(layers)

out = model(Tensor(x_input))
loss = out.sum()
loss.backward()  # Dynamic backward reconstruction: O(1) live activations!
```
*Run showcase:* `python examples/13_avyaya_reversible_computing.py`

---

### 3. P.R.A.M.A.N.A. (Distributional Uncertainty Tensors)
*Sanskrit: प्रमाण (Valid Means of Genuine Knowledge)*  
**Probabilistic Representation of Analytical Moments & Algebraic Noise-aware Autograd.**

Standard neural networks output uncalibrated point estimates. P.R.A.M.A.N.A. introduces a dual-stream computational graph tracking both expectation $\mathbb{E}[X] = \mu$ and variance $\mathrm{Var}[X] = \sigma^2$:
* **Exact Single-Layer Affine & Independent Bilinear Propagation:** A single linear/affine transformation ($\mathbf{y} = \mathbf{x}\mathbf{W}^T + \mathbf{b}$) and independent elementwise products with independent inputs propagates exact analytical moments via Goodman's (1960) identity:
  $$\sigma_{XY}^2 = \mu_X^2 \sigma_Y^2 + \mu_Y^2 \sigma_X^2 + \sigma_X^2 \sigma_Y^2$$
* **First-Order Taylor (Delta-Method) Non-Linearities:** Non-linear activations (`relu`, `sigmoid`, `tanh`, `gelu`, `exp`, `log`) propagate diagonal variance via first-order Taylor expansion ($\sigma_y^2 \approx [f'(\mu_x)]^2 \sigma_x^2$), which is accurate in the small-noise regime ($\sigma \le 0.1$) and ignores off-diagonal inter-unit covariances.
* **Known accuracy limits (measured against 400k-sample Monte Carlo):** per activation the delta method is accurate for small noise (e.g. `tanh` variance error 1.6% at $\sigma=0.05$, 5.4% at $\sigma=0.1$, 37% at $\sigma=0.3$), but `relu` assigns zero variance whenever $\mu \le 0$ and is unreliable near the kink ($|\mu| \lesssim \sigma$). **Stacked layers are approximate** because inputs are assumed independent: a purely *linear* 2-layer stack was off by 3-32% and a 3-layer `tanh` MLP by up to ~170% on individual outputs even at $\sigma=0.05$.
* **`weight_uncertainty=True` is not an OOD detector:** predictive variance scales with $\|\mathbf{x}\|^2$ regardless of the training support (Example 14, Experiment 2 demonstrates the failure with a shifted-support control).
* **Heteroscedastic Noise Learning (`HeteroscedasticMLP`):** Dual-headed non-linear architecture trained with `GaussianNLLLoss` to learn input-dependent aleatoric variance $\sigma^2(x)$ alongside predictive mean $\mu(x)$.

```python
from minigrad import PRAMANA, DistributionalTensor, DistributionalLinear, HeteroscedasticMLP, GaussianNLLLoss

# 1. Single-pass analytical moment propagation through uncertain linear layer
x_dist = DistributionalTensor(mean=x_data, var=var_data)
layer = DistributionalLinear(in_features=4, out_features=1, bias=True, weight_var_init=1e-3)
out_dist = layer(x_dist)

# 2. Input-dependent heteroscedastic variance learning via Gaussian NLL
het_model = HeteroscedasticMLP(in_features=1, hidden_features=16, out_features=1)
pred_dist = het_model(x_input)
loss = GaussianNLLLoss()(pred_dist, target_y)
loss.backward()
```
*Run showcase:* `python examples/14_pramana_distributional_uncertainty.py`

---

### 4. T.A.R.K.A. (Neuro-Symbolic Differentiable Logic)
*Sanskrit: तर्क (Dialectical Inference & Reductio ad Absurdum)*  
**Tensorized Algebraic Reasoning & Knowledge-grounded Autograd.**

Standard deep learning learns purely from statistical correlations and can violate domain rules. T.A.R.K.A. embeds continuous first-order fuzzy logic (Product, Łukasiewicz, and Gödel t-norms with exact Gödel residuum) into autograd:
* Native overloaded operators: `&` (AND), `|` (OR), `~` (NOT), `>>` (IMPLIES), `^` (IFF).
* Differentiable softmin universal quantifiers ($\forall_\tau P(x)$) whose gradients concentrate on the worst instance in proportion to $\text{gap}/\tau$: with 8 instances and $\tau=0.05$ the single violator receives $\approx 100\%$ / $99\%$ / $81\%$ / $48\%$ / $23\%$ of the gradient for truth gaps of $0.85$ / $0.45$ / $0.15$ / $0.05$ / $0.02$ (uniform would be $12.5\%$).
* `TransitivityAxiom` raises *logical consistency* (axiom satisfaction) but does not by itself guarantee recovery of unlabeled implied links: Example 15 reports a seed-averaged supervision-only vs supervision+axiom ablation (the effect on implied links is embedding-dependent), and a constant relation satisfies transitivity trivially.
* Injects mathematical axioms (transitivity, symmetry, mutual exclusion) directly into training objectives:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \, \mathcal{L}_{\text{semantic}}(\Phi)$$

```python
from minigrad import TARKA, LogicTensor, NeuralRelation, SemanticLoss
from minigrad.tarka import TransitivityAxiom

# Deduce multi-hop transitive closures from 1-hop chain facts + TransitivityAxiom
relation = NeuralRelation(net)
trans_axiom = TransitivityAxiom(relation, weight=1.0)

# Backpropagate symbolic transitivity: (R(x,y) ^ R(y,z)) => R(x,z)
loss = trans_axiom.loss(entities)
loss.backward()
```
*Run showcase:* `python examples/15_tarka_neuro_symbolic_reasoning.py`

---

### 5. S.P.A.N.D.A. (Neuromorphic Event-Driven SNNs)
*Sanskrit: स्पन्द (The Primordial Pulse of Dynamic Consciousness)*  
**Spike-Propagation Asynchronous Network Dynamics & Autograd.**

S.P.A.N.D.A. implements Leaky Integrate-and-Fire (LIF) spiking neuron dynamics where layers communicate via sparse binary spike trains ($S \in \{0, 1\}$) across $T$ discrete timesteps, overcoming the non-differentiable Heaviside step barrier ($\delta(x) = 0$) using **Surrogate-Gradient Autograd** (Fast Sigmoid, ArcTan, Gaussian):
* **Per-layer spike recording:** Tracks actual emitted spike trains (`last_spikes`) across each `LIFLayer` and `SpikingLinear` layer during forward execution.
* **Single-Pass ANN vs. $T$-Step SNN Analytical Energy Model:** Compares a single-pass ($T=1$) dense ANN baseline ($B \times D_{\text{in}} \times D_{\text{out}}$ MACs at $E_{\text{MAC}} \approx 4.6\text{ pJ}$) against the SNN: layer 0 sees analog input and costs real MACs; deeper layers are event-driven, with $\text{ACs} = (\text{input spikes}) \times D_{\text{out}}$ at $E_{\text{AC}} \approx 0.9\text{ pJ}$ (45nm, Horowitz 2014); neuron updates ($T \times B \times D_{\text{out}}$) are reported as a range (charged as AC = optimistic, as MAC = conservative). On an untrained random 3-layer SNN ($T=10$, mean firing rate $24.35\%$) the model gives **$0.98\times$ (conservative) to $1.30\times$ (optimistic)**, i.e. *no meaningful saving*, and the SNN performs more operations than the ANN ($0.44\times$ SynOps ratio). Memory-access energy is not modeled; trained, sparser networks can differ.

```python
from minigrad import SPANDA, SpikingSequential, SpikingLinear, RateDecoder

# 2-Layer Temporal Spiking Neural Network
model = SpikingSequential([
    SpikingLinear(2, 16, beta=0.85, v_th=0.8, surrogate="fast_sigmoid"),
    SpikingLinear(16, 1, beta=0.85, v_th=0.8, surrogate="fast_sigmoid"),
])

# Forward pass across T=8 temporal integration steps
spikes = model(x_input, num_steps=8)  # Shape: (T=8, B, 1)
rate_pred = RateDecoder.decode(spikes) # Frequency readout

# Surrogate-gradient Backpropagation Through Time (BPTT)
loss = criterion(rate_pred, target_y)
loss.backward()
```
*Run showcase:* `python examples/16_spanda_neuromorphic_snn.py`

---

## ⚙️ The 4 Foundational Pillars

### Pillar 1: Glass-Box Autograd Engine (`minigrad/glassbox.py`)
* **First-NaN/Inf Root-Cause Debugger (`detect_anomaly`):** Inspects both forward-pass outputs and backward-pass gradients across the computation graph, halting at the exact operation that introduced non-finite values and attaching a creation call-stack snapshot (`"Call Stack Snapshot"`).
* **Interactive ASCII Visualizer:** Generates human-readable computation graphs in the console via `visualize(loss)`.
* **Natural-Language Gradient Explanations:** Diagnoses vanishing/exploding gradients and saturated activations in plain English via `explain_gradients(model)`.

### Pillar 2: Zero-Runtime Embedded C Compiler (`minigrad/compiler.py`)
* **Standalone ANSI C99 Export:** Compiles feed-forward and elementwise computational graphs (`Linear`, `MatMul`, `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Pow`, `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `Sin`, `Cos`, `Abs`, `Exp`, `Log`, `Softmax`, `LayerNorm`, `RMSNorm`) into self-contained C99 code (`export_c`, `to_c`, `compile_to_library`), raising `NotImplementedError` (never emitting silently wrong code) on unsupported operations and layouts: broadcasting beyond a trailing-dimension bias or scalar operand, batched / matrix-vector `matmul`, `softmax` off the last axis, reductions over non-adjacent axes, `conv2d`, `transpose`, indexing, etc. `sum`/`mean` over one axis (or a block of adjacent axes) are supported. The example input is treated as a runtime variable, so frozen (`requires_grad=False`) models compile correctly.
* **Binary Weight Decoupling (`model.bin`):** Supports decoupled binary weight loading via `mmap` / `fread`.
* **INT8 Post-Training Quantization:** Reduces weight storage by $75\%$ using symmetric `int8_t` weights with per-tensor scale factors.
* **Greedy Liveness-Based Interval Memory Reuse:** Re-uses intermediate activation buffers using greedy interval graph coloring on liveness intervals, reducing static RAM footprint by $35\%–90\%$ depending on DAG depth.
* **Cache Tiling & OpenMP:** Optional $32 \times 32$ loop blocking and `#pragma omp parallel for` pragmas.

### Pillar 3: Symbolic Graph Optimization (`minigrad/graph_opt.py`)
* Algebraic simplification rewrites ($x + 0 \to x$, $x \times 1 \to x$, $x - x \to 0$, $x / x \to 1$) under finite real-domain assumptions.
* Constant folding across static subgraphs.
* Kernel fusion: fuses `Linear + ReLU` into `fused_linear_relu` execution nodes with full first- and second-order VJP support.

### Pillar 4: Pure Functional `vmap` & DP-SGD (`minigrad/vmap.py`, `minigrad/dp.py`)
* **Ergonomic Functional Slice-Mapping (`vmap`):** Provides a clean functional batch-mapping interface (`vmap`, `make_functional`, `per_sample_gradients`) that maps pure functions across a specified batch axis and stacks the resulting outputs.
* **Reverse-Mode Jacobian Calculation (`jacrev` / `batched_jacobian`):** Computes exact per-sample Jacobians by sweeping unit basis vectors through reverse-mode `grad`.
* **Differential Privacy Engine & Analytical RDP Accountant (`dp.py`):** Implements per-sample $\ell_2$ gradient clipping, calibrated Gaussian noise injection, and an analytical Rényi Differential Privacy (`RDPAccountant`, `compute_rdp_epsilon`) upper-bound accountant based on Wang, Balle & Kasiviswanathan (2019).

---

## 📂 Runnable Examples

Explore the complete collection of 16 self-contained runnable showcases in `examples/`:

```bash
# Core Deep Learning & Autograd
python examples/01_scalar_autograd.py                    # Pure scalar & tensor autograd basics
python examples/02_linear_regression.py                  # Vectorized linear regression
python examples/03_mlp_xor.py                            # Non-linear XOR classification with MLP
python examples/04_mnist_mlp.py                          # MNIST digit classification (MLP)
python examples/05_mnist_cnn.py                          # MNIST Convolutional Neural Network (Conv2D)
python examples/06_mini_gpt.py                           # Character-level miniGPT causal transformer
python examples/07_pinn_harmonic_oscillator.py           # Physics-Informed Neural Network (PINN) PDE solver

# The 4 Foundational Pillars
python examples/08_glassbox_debugging.py                 # First-NaN root-cause diagnosis & ASCII DAG
python examples/09_embedded_c_export.py                  # Standalone ANSI C99 codegen & native compilation
python examples/10_graph_optimization.py                 # Algebraic rewrites, constant folding & fusion
python examples/11_vmap_and_dp_sgd.py                    # Functional per-sample gradients & private DP-SGD

# The 5 Reference Paradigms
python examples/12_sutra_neural_ode.py                   # S.U.T.R.A. Continuous-Depth Neural ODEs
python examples/13_avyaya_reversible_computing.py        # A.V.Y.A.Y.A. O(1) Memory 50-Layer Reversible Net
python examples/14_pramana_distributional_uncertainty.py # P.R.A.M.A.N.A. Moment Propagation & Heteroscedastic MLP
python examples/15_tarka_neuro_symbolic_reasoning.py     # T.A.R.K.A. Neuro-Symbolic Differentiable Logic
python examples/16_spanda_neuromorphic_snn.py            # S.P.A.N.D.A. Neuromorphic Event-Driven SNN
```

---

## 🧪 Verification & Testing

miniGrad enforces mathematical and regression testing across **583 collected test cases**, differential suites, and strict static typing:
* **583 tests collected**: **545 tests** pass natively with NumPy as the only dependency (the default CI environment), and **38 optional cross-framework parity tests** execute and pass when PyTorch and `safetensors` are installed (**495 / 495 pass**).

```bash
# Run complete test suite (583 tests across contracts, lifecycles, random DAGs, compiler benchmarks, audit regressions)
python -m pytest

# Run strict static type checking (0 errors)
python -m mypy minigrad tests

# Run linter & formatter checks (0 issues)
python -m ruff check minigrad tests
```

---

## 📄 License

MIT License. Designed, built, and verified from first principles. Feel free to use, study, extend, and build on it.
