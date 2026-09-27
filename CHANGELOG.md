# Changelog

All notable changes to the **miniGrad** framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-09-27

### 🚀 Added
- **Zero-Runtime C Compiler Superpowers (`minigrad/compiler.py`):**
  - **Binary Weight Decoupling (`model.bin`):** Option `binary_weights=True` exports model weights into a compact contiguous binary file with an instant C reader, keeping `model.c` microscopic (< 20 KB) and compilation instantaneous (< 0.5s) while supporting 10M+ parameter models.
  - **INT8 Post-Training Quantization:** Option `quantize="int8"` symmetrically quantizes 2D weight matrices to `int8_t` with per-tensor floating scale factors, slashing parameter storage by 75% with native `minigrad_matmul_int8_fp32` and `minigrad_fused_linear_relu_int8_fp32` kernels.
  - **Activation Liveness Arena:** Option `arena_memory=True` implements interval graph coloring to analyze tensor birth and death steps, reusing intermediate buffer offsets inside a single `model_arena` and slashing activation RAM by 80% to 95%.
  - **Cache Tiling & OpenMP Multithreading:** Options `tiling=True` and `openmp=True` introduce 32x32 loop blocking (`minigrad_matmul_2d_tiled`) and `#pragma omp parallel for` support to eliminate CPU cache misses and saturate multi-core hardware.
  - **Static Key-Value (KV) Cache:** Option `kv_cache=True` generates native C multi-head attention with a static KV-cache (`model_reset_kv_cache`, `model_attention_step`), speeding up streaming autoregressive chat generation by 5x to 10x.
  - **Clean Library Export (`model.h` & `model.c`):** New top-level `compile_to_library()` function generates idiomatic C headers with `extern "C"` guards and clean function prototypes for direct inclusion in C/C++, iOS/Android, robotics, and game engines.
- **Official Brand Identity System:**
  - Adopted the official **Dual-Stream Monogram (`m·g`)** brand mark representing epistemic expectation and variance harmonic waves.
  - Added pure vector SVG asset suite in `assets/`: `logo.svg`, `banner.svg`, `logo-icon.svg`, and `logo-light.svg`.
  - Embedded official logo in repository header.
- **Test Suite Expansion:**
  - Added 6 native Clang C compilation parity tests covering all new compiler features.
  - Expanded total framework unit test suite from 171 to 177 tests (100% passing).

---

## [1.0.1] - 2026-09-26

### 🩹 Fixed
- **Pure-NumPy Zero-Dependency Guarantee:** Replaced `scipy.linalg.expm` in S.U.T.R.A. tests and examples with an exact closed-form analytical matrix exponential for 2D spiral dynamics ($5.55 \times 10^{-17}$ difference), ensuring strictly zero external dependencies beyond NumPy.
- **Cross-Platform Compiler Linking:** Added Linux GCC math library linking flag (`-lm`) to embedded C compiler tests and examples.
- **Documentation & KaTeX:** Resolved GitHub Markdown KaTeX macro rendering issue in Section 3 (P.R.A.M.A.N.A.) by standardizing variance notation to `\mathrm{Var}[X]`.
- **Code Quality:** Resolved 666 Ruff linting errors across the codebase (`BLE001`, `PLW1510`, `S110`, `I001`) and verified 100% clean Mypy type-checking across all 43 source files.

---

## [1.0.0] - 2026-09-25

### 🎉 Initial Public Release
- **Foundational Engine:** Dynamic computation graph, reverse-mode automatic differentiation, higher-order Hessians.
- **Pillar 1 (Glass-Box Engine):** First-NaN root-cause diagnosis, ASCII visualizer, and telemetry.
- **Pillar 2 (Embedded C Compiler):** Zero-runtime standalone ANSI C99 code generation.
- **Pillar 3 (Graph Optimization):** Algebraic simplification, constant folding, and operator fusion.
- **Pillar 4 (Functional Transforms):** Pure functional `vmap`, batched Jacobians (`jacrev`), and DP-SGD differential privacy.
- **Frontier Paradigm 1 (S.U.T.R.A.):** Continuous-Depth Neural ODEs with O(1) Pontryagin Adjoint autograd and Dormand-Prince adaptive step solver.
- **Frontier Paradigm 2 (A.V.Y.A.Y.A.):** Reversible bipartite computing with machine-precision algebraic reconstruction for 500+ layer networks in O(1) activation RAM.
- **Frontier Paradigm 3 (P.R.A.M.A.N.A.):** Distributional uncertainty tensors tracking expectation $\mathbb{E}[X]$ and variance $\mathrm{Var}[X]$ via closed-form Goodman algebra.
- **Frontier Paradigm 4 (T.A.R.K.A.):** Differentiable first-order neuro-symbolic logic with softmin universal quantifiers.
- **Frontier Paradigm 5 (S.P.A.N.D.A.):** Neuromorphic spiking neural networks with LIF neurons and surrogate gradients.
