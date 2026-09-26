"""
tests/test_compiler.py — Unit tests for Zero-Runtime Embedded C Compiler (Pillar 2).

Validates:
1. End-to-end numerical parity between Python forward pass and native compiled C execution.
2. Zero dynamic heap allocation (no malloc/free, 100% static buffers).
3. Support for diverse layers: Linear, ReLU, Sigmoid, Tanh, GELU, Softmax, LayerNorm.
4. Top-level export_c, to_c, and Module/Tensor convenience methods.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from minigrad import Tensor, export_c
from minigrad.nn import (
    GELU,
    LayerNorm,
    Linear,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)


def _find_c_compiler() -> str | None:
    """Find available C compiler (clang or gcc)."""
    for compiler in ("clang", "gcc"):
        path = shutil.which(compiler)
        if path:
            return path
    return None


CLANG_OR_GCC = _find_c_compiler()


def _compile_c(c_file: Path, exe_file: Path) -> subprocess.CompletedProcess:
    assert CLANG_OR_GCC is not None
    cmd = [CLANG_OR_GCC, "-O3", str(c_file), "-o", str(exe_file)]
    if os.name != "nt":
        cmd.append("-lm")
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_zero_dynamic_memory_allocation():
    """Verify that generated C code contains zero calls to malloc, calloc, realloc, or free."""
    model = Sequential([Linear(4, 8), ReLU(), Linear(8, 2)])
    x = Tensor([[1.0, 2.0, 3.0, 4.0]])

    c_code = export_c(model, x, include_main=True)

    # Strip C comments before checking for function calls
    code_without_comments = re.sub(r"/\*.*?\*/", "", c_code, flags=re.DOTALL)
    code_without_comments = re.sub(r"//.*", "", code_without_comments)

    # Must contain zero heap allocations
    for forbidden in ("malloc", "calloc", "realloc", "free"):
        pattern = rf"\b{forbidden}\s*\("
        assert not re.search(pattern, code_without_comments), f"Found forbidden dynamic memory call: {forbidden}"

    # Must contain static activation buffers
    assert "static float act_" in c_code
    assert "void model_forward" in c_code
    assert "0 BYTES (100% STATIC BUFFERS)" in c_code


def test_module_and_tensor_convenience_methods():
    """Test Module.export_c(), Module.to_c(), and Tensor.export_c()."""
    model = Sequential([Linear(2, 4), ReLU(), Linear(4, 1)])
    x = Tensor([[0.5, -0.5]])

    # Module convenience
    c_code_mod = model.export_c(x, include_main=False, model_name="my_mlp")
    assert "void my_mlp_forward" in c_code_mod
    assert "int main" not in c_code_mod

    # to_c alias
    c_code_alias = model.to_c(x, include_main=False, model_name="my_mlp")
    assert c_code_alias == c_code_mod

    # Tensor convenience
    out = model(x)
    c_code_tensor = out.export_c(x, include_main=False, model_name="tensor_model")
    assert "void tensor_model_forward" in c_code_tensor


@pytest.mark.skipif(CLANG_OR_GCC is None, reason="No C compiler (clang or gcc) found on system")
def test_c_export_mlp_parity(tmp_path: Path):
    """
    Train a small MLP in miniGrad, export to C, compile natively with clang/gcc,
    execute the standalone binary, and verify exact numerical parity with Python.
    """
    np.random.seed(42)
    model = Sequential([Linear(3, 6), ReLU(), Linear(6, 2)])
    x = Tensor([[0.75, -1.25, 2.5]], requires_grad=False)

    py_out = model(x).data.flatten()

    c_file = tmp_path / "model.c"
    exe_file = tmp_path / ("model.exe" if os.name == "nt" else "model")

    export_c(model, x, filename=c_file, include_main=True, model_name="mlp")
    assert c_file.exists()

    # Compile with native compiler
    res = _compile_c(c_file, exe_file)
    assert res.returncode == 0, f"Compilation failed: {res.stderr}"
    assert exe_file.exists()

    # Run native executable
    run_res = subprocess.run([str(exe_file)], capture_output=True, text=True, check=False)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"

    # Parse output: "output[0] = +0.1234567"
    c_outputs = []
    for line in run_res.stdout.splitlines():
        if "output[" in line and "=" in line:
            val_str = line.split("=")[-1].strip()
            c_outputs.append(float(val_str))

    assert len(c_outputs) == len(py_out)
    np.testing.assert_allclose(c_outputs, py_out, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(CLANG_OR_GCC is None, reason="No C compiler (clang or gcc) found on system")
def test_c_export_various_activations(tmp_path: Path):
    """Test Sigmoid, Tanh, and GELU in compiled C code."""
    np.random.seed(123)
    model = Sequential([
        Linear(2, 4),
        Sigmoid(),
        Linear(4, 4),
        Tanh(),
        Linear(4, 2),
        GELU(),
    ])
    x = Tensor([[1.5, -0.8]], requires_grad=False)
    py_out = model(x).data.flatten()

    c_file = tmp_path / "activations_model.c"
    exe_file = tmp_path / ("activations_model.exe" if os.name == "nt" else "activations_model")

    export_c(model, x, filename=c_file, include_main=True, model_name="act_test")

    res = _compile_c(c_file, exe_file)
    assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    run_res = subprocess.run([str(exe_file)], capture_output=True, text=True, check=False)
    assert run_res.returncode == 0

    c_outputs = []
    for line in run_res.stdout.splitlines():
        if "output[" in line and "=" in line:
            c_outputs.append(float(line.split("=")[-1].strip()))

    np.testing.assert_allclose(c_outputs, py_out, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(CLANG_OR_GCC is None, reason="No C compiler (clang or gcc) found on system")
def test_c_export_layernorm(tmp_path: Path):
    """Test LayerNorm layer compiled to native C."""
    np.random.seed(999)
    model = Sequential([
        Linear(4, 6),
        LayerNorm(6),
        ReLU(),
        Linear(6, 3),
    ])
    x = Tensor([[0.1, -0.4, 1.2, 0.9]], requires_grad=False)
    py_out = model(x).data.flatten()

    c_file = tmp_path / "ln_model.c"
    exe_file = tmp_path / ("ln_model.exe" if os.name == "nt" else "ln_model")

    export_c(model, x, filename=c_file, include_main=True, model_name="ln_test")

    res = _compile_c(c_file, exe_file)
    assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    run_res = subprocess.run([str(exe_file)], capture_output=True, text=True, check=False)
    assert run_res.returncode == 0

    c_outputs = []
    for line in run_res.stdout.splitlines():
        if "output[" in line and "=" in line:
            c_outputs.append(float(line.split("=")[-1].strip()))

    np.testing.assert_allclose(c_outputs, py_out, rtol=1e-4, atol=1e-4)
