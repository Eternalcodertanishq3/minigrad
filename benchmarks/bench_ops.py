"""
bench_ops.py — Benchmark miniGrad operations vs NumPy baseline.

Measures forward and backward pass times for core operations.
Results can be included in BENCHMARKS.md for the resume.

Run: python benchmarks/bench_ops.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
from minigrad import Tensor


def benchmark(name, fn, setup_fn=None, warmup=3, repeats=10):
    """Benchmark a function. Returns average time in ms."""
    # Warmup
    for _ in range(warmup):
        if setup_fn:
            args = setup_fn()
            fn(*args)
        else:
            fn()

    # Timed runs
    times = []
    for _ in range(repeats):
        if setup_fn:
            args = setup_fn()
            start = time.perf_counter()
            fn(*args)
        else:
            start = time.perf_counter()
            fn()
        times.append((time.perf_counter() - start) * 1000)  # ms

    return np.mean(times), np.std(times)


def main():
    print("=" * 70)
    print("miniGrad Operation Benchmarks")
    print("=" * 70)
    print(f"NumPy version: {np.__version__}")
    print()

    sizes = [(64, 64), (256, 256), (512, 512), (1024, 1024)]

    # --- Matrix Multiplication ---
    print("-" * 70)
    print("Matrix Multiplication (forward + backward)")
    print("-" * 70)
    print(f"{'Size':>12} {'miniGrad (ms)':>15} {'NumPy (ms)':>15} {'Overhead':>12}")

    for m, n in sizes:
        k = m

        def setup():
            a = Tensor(np.random.randn(m, k), requires_grad=True)
            b = Tensor(np.random.randn(k, n), requires_grad=True)
            return a, b

        def matmul_grad(a, b):
            c = a @ b
            c.backward()

        def numpy_only():
            a = np.random.randn(m, k)
            b = np.random.randn(k, n)
            c = a @ b

        t_mg, std_mg = benchmark(f"matmul_{m}x{n}", matmul_grad, setup)
        t_np, std_np = benchmark(f"numpy_{m}x{n}", numpy_only)

        overhead = t_mg / t_np if t_np > 0 else float('inf')
        print(f"{f'({m},{n})':>12} {t_mg:>14.2f} {t_np:>14.2f} {overhead:>11.2f}x")

    print()

    # --- Element-wise Operations ---
    print("-" * 70)
    print("Element-wise Operations (forward + backward)")
    print("-" * 70)
    print(f"{'Op':>12} {'Size':>10} {'Time (ms)':>15}")

    size = 1000
    ops = [
        ("add", lambda x, y: (x + y).backward()),
        ("mul", lambda x, y: (x * y).backward()),
        ("relu", lambda x: x.relu().backward()),
        ("sigmoid", lambda x: x.sigmoid().backward()),
        ("exp", lambda x: x.exp().backward()),
    ]

    for op_name, op_fn in ops:
        def setup():
            x = Tensor(np.random.randn(size, size), requires_grad=True)
            y = Tensor(np.random.randn(size, size), requires_grad=True) if op_name in ("add", "mul") else None
            return (x, y) if y is not None else (x,)

        def run_fn(*args):
            if len(args) == 2:
                op_fn(args[0], args[1])
            else:
                op_fn(args[0])

        t_mean, t_std = benchmark(f"{op_name}_{size}", run_fn, setup, repeats=10)
        print(f"{op_name:>12} {f'({size},{size})':>10} {t_mean:>14.2f}")

    print()

    # --- Reduction Operations ---
    print("-" * 70)
    print("Reduction Operations")
    print("-" * 70)
    print(f"{'Op':>12} {'Size':>10} {'Time (ms)':>15}")

    reductions = [
        ("sum", lambda x: x.sum().backward()),
        ("mean", lambda x: x.mean().backward()),
    ]

    for op_name, op_fn in reductions:
        def setup():
            x = Tensor(np.random.randn(size, size), requires_grad=True)
            return (x,)

        def run_fn(x):
            op_fn(x)

        t_mean, t_std = benchmark(f"{op_name}_{size}", run_fn, setup, repeats=10)
        print(f"{op_name:>12} {f'({size},{size})':>10} {t_mean:>14.2f}")

    print()
    print("=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
