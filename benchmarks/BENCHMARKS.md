# miniGrad Benchmarks

Performance measurement of miniGrad operations against NumPy baselines using a controlled benchmarking methodology.

## Environment

- CPU: AMD/Intel x86_64
- OS: Windows 11
- NumPy: v2.4.6 (with OpenBLAS/MKL)
- Python: 3.11+
- Methodology: Arrays and tensors pre-allocated outside timed blocks; 3 warm-up runs, 10 timed iterations per test (`benchmarks/bench_ops.py`).

## Matrix Multiplication (miniGrad forward + backward vs NumPy forward)

| Size | miniGrad (ms) | NumPy (ms) | Overhead |
| :--- | :--- | :--- | :--- |
| **(64, 64)** | 0.15 | 0.01 | 12.3x |
| **(256, 256)** | 2.07 | 0.72 | 2.9x |
| **(512, 512)** | 9.92 | 3.33 | 3.0x |
| **(1024, 1024)** | 110.73 | 33.62 | 3.3x |

**Analysis**:
- For medium and large matrix sizes ($(256, 256)$ to $(1024, 1024)$), miniGrad autograd overhead is bounded at **$\sim 3\times$** the raw NumPy forward multiplication time.
- Since miniGrad computes both the forward pass and full backward automatic differentiation ($C = A @ B$ followed by $dA = dC @ B^T$ and $dB = A^T @ dC$), an overhead factor of $\sim 3\times$ represents near-optimal theoretical efficiency for a pure-Python graph engine (forward + 2 matrix products for backward $\approx 3\times$ compute).

## Element-Wise Operations (Forward + Backward, $1000 \times 1000$)

| Operation | Time (ms) | Notes |
| :--- | :--- | :--- |
| **add** | 13.16 ms | Tensor addition + unbroadcast gradient routing |
| **mul** | 13.64 ms | Element-wise product + dual-branch VJP |
| **relu** | 11.80 ms | Forward thresholding + boolean mask gradient |
| **sigmoid** | 45.17 ms | Transcendental activation + $y(1-y)$ derivative |
| **exp** | 14.58 ms | Element-wise exponential + self-scaled gradient |

## Reduction Operations ($1000 \times 1000$)

| Operation | Time (ms) | Notes |
| :--- | :--- | :--- |
| **sum** | 12.04 ms | Array aggregation + broadcast expansion gradient |
| **mean** | 1.26 ms | Scaled sum reduction + normalized gradient |

## Native C Execution Speedups

For systems-level deployment and embedded inference, miniGrad includes the Pillar 2 C compiler (`minigrad.compiler.export_c`). Compiled C execution with static activation arena memory bypasses the Python interpreter and garbage collector, delivering kernel-level execution latencies measured in microseconds (`tests/test_compiler_benchmarks.py`).
