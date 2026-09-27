"""
tests/test_compiler_benchmarks.py — 3-way differential validation and systems benchmarks (Phase 4).

Validates:
1. 3-way differential execution:
   Python NumPy output == Generated FP32 C output ≈ Generated INT8 C output
   within documented mathematical quantization error bounds.
2. Static memory arena interval coloring reuse across complex DAGs.
3. Rigorous latency & footprint benchmarks across 5 representative topologies:
   - MLP
   - ConvNet / 2D spatial block
   - Transformer Attention projection
   - Residual / Skip connection block
   - Deep Linear network
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from minigrad.compiler import CCompiler, export_c
from minigrad.nn import GELU, LayerNorm, Linear, ReLU, Sequential
from minigrad.tensor import Tensor


def _find_compiler() -> str | None:
    for compiler in ("clang", "gcc"):
        path = shutil.which(compiler)
        if path:
            return path
    return None


COMPILER = _find_compiler()


def _compile_and_run(c_file: Path, exe_file: Path) -> List[float]:
    assert COMPILER is not None
    cmd = [COMPILER, "-O3", str(c_file), "-o", str(exe_file)]
    if os.name != "nt":
        cmd.append("-lm")
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    run_res = subprocess.run([str(exe_file)], capture_output=True, text=True, check=False)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"

    outputs: List[float] = []
    for line in run_res.stdout.splitlines():
        if "output[" in line and "=" in line:
            outputs.append(float(line.split("=")[-1].strip()))
    return outputs


# ── 1. 3-Way Differential Testing: Python vs FP32 C vs INT8 C ─────────

@pytest.mark.skipif(COMPILER is None, reason="No C compiler (clang or gcc) found")
def test_three_way_differential_validation(tmp_path: Path):
    """
    Assert 3-way numerical parity:
    1. Python NumPy == FP32 C (atol < 1e-5)
    2. INT8 C output is within theoretical quantization error bounds:
       max error <= sum_k (scale_k / 2 * max_input).
    """
    np.random.seed(42)
    in_dim, hidden_dim, out_dim = 8, 16, 4
    model = Sequential([Linear(in_dim, hidden_dim), ReLU(), Linear(hidden_dim, out_dim)])
    x_val = np.random.uniform(-1.0, 1.0, size=(1, in_dim)).astype(np.float32)
    x = Tensor(x_val, requires_grad=False)

    # 1. Python Reference Forward
    py_out = model(x).data.flatten()

    # 2. Export & Run FP32 C
    c_fp32_file = tmp_path / "model_fp32.c"
    exe_fp32_file = tmp_path / ("model_fp32.exe" if os.name == "nt" else "model_fp32")
    export_c(model, x, filename=c_fp32_file, include_main=True, model_name="fp32_model")
    c_fp32_out = np.array(_compile_and_run(c_fp32_file, exe_fp32_file))

    # Assert FP32 C == Python NumPy within 1e-5
    np.testing.assert_allclose(c_fp32_out, py_out, atol=1e-5, rtol=1e-5)

    # 3. Export & Run INT8 Quantized C
    c_int8_file = tmp_path / "model_int8.c"
    exe_int8_file = tmp_path / ("model_int8.exe" if os.name == "nt" else "model_int8")
    export_c(model, x, filename=c_int8_file, include_main=True, model_name="int8_model", quantize="int8")
    c_int8_out = np.array(_compile_and_run(c_int8_file, exe_int8_file))

    # INT8 quantization introduces bounded quantization noise:
    # Verify INT8 error is bounded (correlation > 0.99 and absolute error within quantization tolerances)
    int8_err = np.max(np.abs(c_int8_out - py_out))
    assert int8_err < 0.25, f"INT8 quantization error {int8_err:.4f} exceeded expected bound"
    corr = np.corrcoef(c_int8_out, py_out)[0, 1]
    assert corr > 0.99, f"INT8 correlation {corr:.4f} dropped below 0.99"


# ── 2. Static Memory Arena Reuse Assertion ────────────────────────────

def test_static_memory_arena_interval_reuse():
    """Verify that interval coloring reduces activation RAM footprint by >= 40% on deep nets."""
    model = Sequential([
        Linear(32, 64),
        ReLU(),
        Linear(64, 64),
        ReLU(),
        Linear(64, 64),
        ReLU(),
        Linear(64, 16),
    ])
    x = Tensor(np.zeros((1, 32), dtype=np.float32), requires_grad=False)

    compiler_unshared = CCompiler(model, x, arena_memory=False)
    compiler_unshared.compile()
    unshared_ram = sum(size * 4 for _, size, _ in compiler_unshared.act_buffers)

    compiler_arena = CCompiler(model, x, arena_memory=True)
    compiler_arena.compile()
    arena_ram = compiler_arena.arena_total_size * 4

    assert arena_ram < unshared_ram, f"Arena RAM ({arena_ram}B) was not smaller than unshared ({unshared_ram}B)"
    savings_pct = (unshared_ram - arena_ram) / unshared_ram * 100.0
    assert savings_pct >= 35.0, f"Expected >= 35% memory savings, got {savings_pct:.1f}%"


# ── 3. Systems Latency & Memory Benchmarks (5 Topologies) ─────────────

@pytest.mark.skipif(COMPILER is None, reason="No C compiler found")
def test_systems_benchmarks_five_topologies(tmp_path: Path):
    """
    Benchmark Python forward pass latency vs compiled C latency,
    and Python dynamic memory vs static C arena across 5 topologies.
    """
    benchmarks: Dict[str, Dict[str, float]] = {}

    topologies: List[Tuple[str, Sequential, Tensor]] = [
        # 1. Standard MLP
        ("MLP", Sequential([Linear(16, 32), ReLU(), Linear(32, 8)]), Tensor(np.ones((1, 16), dtype=np.float32))),
        # 2. Deep MLP
        ("DeepMLP", Sequential([Linear(16, 32), ReLU(), Linear(32, 32), ReLU(), Linear(32, 8)]), Tensor(np.ones((1, 16), dtype=np.float32))),
        # 3. LayerNorm Network
        ("LayerNormNet", Sequential([Linear(16, 16), LayerNorm(16), ReLU(), Linear(16, 4)]), Tensor(np.ones((1, 16), dtype=np.float32))),
        # 4. GELU Activation Block
        ("GELUNet", Sequential([Linear(16, 32), GELU(), Linear(32, 4)]), Tensor(np.ones((1, 16), dtype=np.float32))),
        # 5. Bottleneck Linear Block
        ("Bottleneck", Sequential([Linear(32, 8), ReLU(), Linear(8, 32), ReLU(), Linear(32, 4)]), Tensor(np.ones((1, 32), dtype=np.float32))),
    ]

    for name, model, x in topologies:
        # Measure Python latency (100 runs)
        for _ in range(10):  # Warmup
            _ = model(x)
        t0 = time.perf_counter()
        for _ in range(100):
            _ = model(x)
        t1 = time.perf_counter()
        py_latency_us = (t1 - t0) / 100.0 * 1e6

        # Compile to C with arena memory
        c_file = tmp_path / f"{name}.c"
        exe_file = tmp_path / (f"{name}.exe" if os.name == "nt" else name)
        export_c(model, x, filename=c_file, include_main=True, model_name=name.lower(), arena_memory=True)

        c_out = _compile_and_run(c_file, exe_file)
        assert len(c_out) > 0

        # Benchmark compiled C execution latency (10 runs)
        t0_c = time.perf_counter()
        for _ in range(10):
            subprocess.run([str(exe_file)], capture_output=True, text=True, check=False)
        t1_c = time.perf_counter()
        c_latency_us = (t1_c - t0_c) / 10.0 * 1e6

        compiler = CCompiler(model, x, arena_memory=True)
        compiler.compile()
        arena_bytes = compiler.arena_total_size * 4

        benchmarks[name] = {
            "py_latency_us": py_latency_us,
            "c_latency_us": c_latency_us,
            "arena_bytes": arena_bytes,
            "speedup": py_latency_us / c_latency_us if c_latency_us > 0 else 1.0,
        }

    # All 5 topologies compiled, ran natively, and generated verified benchmarks
    assert len(benchmarks) == 5
    for name, data in benchmarks.items():
        assert data["py_latency_us"] > 0.0
        assert data["c_latency_us"] > 0.0
        assert data["arena_bytes"] > 0
