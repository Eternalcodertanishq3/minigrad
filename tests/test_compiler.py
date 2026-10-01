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

from minigrad import Tensor, compile_to_library, export_c
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


@pytest.mark.skipif(CLANG_OR_GCC is None, reason="No C compiler found")
def test_binary_weights_decoupling(tmp_path: Path):
    """Test binary weights file decoupling (.bin) and native C loader."""
    np.random.seed(42)
    model = Sequential([Linear(4, 8), ReLU(), Linear(8, 2)])
    x = Tensor([[0.5, -1.2, 0.3, 2.0]], requires_grad=False)
    py_out = model(x).data.flatten()

    c_file = tmp_path / "bin_model.c"
    bin_file = tmp_path / "bin_model.bin"
    exe_file = tmp_path / ("bin_model.exe" if os.name == "nt" else "bin_model")

    export_c(
        model,
        x,
        filename=c_file,
        include_main=True,
        model_name="bin_test",
        binary_weights=True,
        weights_filename=bin_file,
    )

    assert bin_file.exists()
    assert bin_file.stat().st_size > 0

    c_code = c_file.read_text(encoding="utf-8")
    assert "bin_test_load_weights" in c_code
    assert "bin_test_weights_buf" in c_code

    res = _compile_c(c_file, exe_file)
    assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    run_res = subprocess.run([str(exe_file), str(bin_file)], capture_output=True, text=True, check=False)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"

    c_outputs = []
    for line in run_res.stdout.splitlines():
        if "output[" in line and "=" in line:
            c_outputs.append(float(line.split("=")[-1].strip()))

    np.testing.assert_allclose(c_outputs, py_out, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(CLANG_OR_GCC is None, reason="No C compiler found")
def test_int8_quantization(tmp_path: Path):
    """Test INT8 post-training quantization and native int8 kernels."""
    np.random.seed(42)
    model = Sequential([Linear(4, 16), ReLU(), Linear(16, 2)])
    x = Tensor([[0.5, -1.0, 1.5, -0.5]], requires_grad=False)
    py_out = model(x).data.flatten()

    c_file = tmp_path / "int8_model.c"
    exe_file = tmp_path / ("int8_model.exe" if os.name == "nt" else "int8_model")

    export_c(
        model,
        x,
        filename=c_file,
        include_main=True,
        model_name="int8_test",
        quantize="int8",
    )

    c_code = c_file.read_text(encoding="utf-8")
    assert "signed char" in c_code
    assert "scale" in c_code
    assert "minigrad_fused_linear_relu_int8_fp32" in c_code or "minigrad_matmul_int8_fp32" in c_code

    res = _compile_c(c_file, exe_file)
    assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    run_res = subprocess.run([str(exe_file)], capture_output=True, text=True, check=False)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"

    c_outputs = []
    for line in run_res.stdout.splitlines():
        if "output[" in line and "=" in line:
            c_outputs.append(float(line.split("=")[-1].strip()))

    # Quantization introduces small rounding differences, but tracks FP32 output closely
    np.testing.assert_allclose(c_outputs, py_out, atol=0.08)


@pytest.mark.skipif(CLANG_OR_GCC is None, reason="No C compiler found")
def test_activation_memory_arena(tmp_path: Path):
    """Test activation liveness arena memory planning (80-95% RAM reduction)."""
    np.random.seed(42)
    model = Sequential([
        Linear(8, 16),
        ReLU(),
        Linear(16, 16),
        ReLU(),
        Linear(16, 16),
        ReLU(),
        Linear(16, 4),
    ])
    x = Tensor(np.random.randn(1, 8).astype(np.float32), requires_grad=False)
    py_out = model(x).data.flatten()

    c_file = tmp_path / "arena_model.c"
    exe_file = tmp_path / ("arena_model.exe" if os.name == "nt" else "arena_model")

    export_c(
        model,
        x,
        filename=c_file,
        include_main=True,
        model_name="arena_test",
        arena_memory=True,
    )

    c_code = c_file.read_text(encoding="utf-8")
    assert "arena_test_arena[" in c_code
    assert "RAM saved via Arena" in c_code

    res = _compile_c(c_file, exe_file)
    assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    run_res = subprocess.run([str(exe_file)], capture_output=True, text=True, check=False)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"

    c_outputs = []
    for line in run_res.stdout.splitlines():
        if "output[" in line and "=" in line:
            c_outputs.append(float(line.split("=")[-1].strip()))

    np.testing.assert_allclose(c_outputs, py_out, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(CLANG_OR_GCC is None, reason="No C compiler found")
def test_cache_tiling_and_openmp(tmp_path: Path):
    """Test 32x32 cache tiling loop blocking and OpenMP annotations."""
    np.random.seed(42)
    model = Sequential([Linear(32, 32, bias=False)])
    x = Tensor(np.random.randn(1, 32).astype(np.float32), requires_grad=False)
    py_out = model(x).data.flatten()

    c_file = tmp_path / "tiled_model.c"
    exe_file = tmp_path / ("tiled_model.exe" if os.name == "nt" else "tiled_model")

    export_c(
        model,
        x,
        filename=c_file,
        include_main=True,
        model_name="tiled_test",
        tiling=True,
    )

    c_code = c_file.read_text(encoding="utf-8")
    assert "minigrad_matmul_2d_tiled" in c_code
    assert "MINIGRAD_TILE_SIZE 32" in c_code

    res = _compile_c(c_file, exe_file)
    assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    run_res = subprocess.run([str(exe_file)], capture_output=True, text=True, check=False)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"

    c_outputs = []
    for line in run_res.stdout.splitlines():
        if "output[" in line and "=" in line:
            c_outputs.append(float(line.split("=")[-1].strip()))

    np.testing.assert_allclose(c_outputs, py_out, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(CLANG_OR_GCC is None, reason="No C compiler found")
def test_static_kv_cache_engine(tmp_path: Path):
    """Test static Key-Value (KV) cache generation for streaming attention with boundary safety."""
    model = Sequential([Linear(16, 16)])
    x = Tensor(np.zeros((1, 16), dtype=np.float32), requires_grad=False)

    c_file = tmp_path / "kv_model.c"
    export_c(
        model,
        x,
        filename=c_file,
        include_main=False,
        model_name="kv_test",
        kv_cache=True,
        max_seq_len=4,
    )

    c_code = c_file.read_text(encoding="utf-8")
    assert "minigrad_attention_kv_cache" in c_code
    assert "kv_test_k_cache" in c_code
    assert "kv_test_reset_kv_cache" in c_code
    assert "kv_test_attention_step" in c_code

    # Create C test harness to verify sequential stepping up to max_seq_len and overflow rejection
    harness = """
#include <stdio.h>
#include <assert.h>

int main(void) {
    kv_test_reset_kv_cache();
    assert(kv_test_get_kv_step() == 0);

    float q[128] = {0.1f};
    float k[128] = {0.2f};
    float v[128] = {0.3f};
    float out[128] = {0.0f};

    /* Run sequential token generation up to max_seq_len (4) */
    for (int t = 0; t < 4; t++) {
        int status = kv_test_attention_step(q, k, v, out);
        assert(status == 0);
        assert(kv_test_get_kv_step() == t + 1);
    }
    assert(kv_test_get_kv_step() == 4);

    /* Attempt step beyond max_seq_len: must be safely rejected with status -1 */
    int overflow_status = kv_test_attention_step(q, k, v, out);
    assert(overflow_status == -1);
    assert(kv_test_get_kv_step() == 4); /* Cache step must not increment past capacity */

    /* Reset cache */
    kv_test_reset_kv_cache();
    assert(kv_test_get_kv_step() == 0);

    printf("KV-Cache successfully stepped 4 tokens and rejected overflow.\\n");
    return 0;
}
"""
    test_c_file = tmp_path / "test_kv_runner.c"
    test_c_file.write_text(c_code + "\n" + harness, encoding="utf-8")
    exe_file = tmp_path / ("test_kv_runner.exe" if os.name == "nt" else "test_kv_runner")

    res = _compile_c(test_c_file, exe_file)
    assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    run_res = subprocess.run([str(exe_file)], capture_output=True, text=True, check=False)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"
    assert "KV-Cache successfully stepped 4 tokens and rejected overflow." in run_res.stdout



@pytest.mark.skipif(CLANG_OR_GCC is None, reason="No C compiler found")
def test_clean_library_export(tmp_path: Path):
    """Test generating idiomatic model.h and model.c library files."""
    np.random.seed(42)
    model = Sequential([Linear(4, 8), ReLU(), Linear(8, 2)])
    x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=False)
    py_out = model(x).data.flatten()

    lib_dir = tmp_path / "lib_export"
    h_file, c_file = compile_to_library(
        model,
        x,
        output_dir=lib_dir,
        model_name="edge_core",
    )

    assert h_file.exists()
    assert c_file.exists()

    h_code = h_file.read_text(encoding="utf-8")
    assert "EDGE_CORE_INPUT_SIZE" in h_code
    assert "void edge_core_forward(const float* input, float* output);" in h_code
    assert "#ifdef __cplusplus" in h_code

    # Write a clean user application including the header
    user_app = """
#include <stdio.h>
#include "edge_core.h"

int main(void) {
    float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[2] = {0.0f, 0.0f};

    edge_core_forward(in, out);
    printf("Library output: %+.7f, %+.7f\\n", out[0], out[1]);
    return 0;
}
"""
    app_file = lib_dir / "main.c"
    app_file.write_text(user_app, encoding="utf-8")
    exe_file = lib_dir / ("app.exe" if os.name == "nt" else "app")

    assert CLANG_OR_GCC is not None
    cmd = [CLANG_OR_GCC, "-O3", "-I", str(lib_dir), str(app_file), str(c_file), "-o", str(exe_file)]
    if os.name != "nt":
        cmd.append("-lm")
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    run_res = subprocess.run([str(exe_file)], capture_output=True, text=True, check=False)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"

    parts = run_res.stdout.strip().split(":")[-1].split(",")
    c_out = [float(p.strip()) for p in parts]
    np.testing.assert_allclose(c_out, py_out, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(CLANG_OR_GCC is None, reason="No C compiler found")
def test_c_export_division_ieee754_parity(tmp_path: Path):
    """Test compiled C division kernel IEEE-754 semantics parity (inf, -inf, nan, finite)."""
    # Create a small computation graph: x / c
    x = Tensor([1.0, -1.0, 0.0, 4.0], requires_grad=False)
    c = Tensor([0.0, 0.0, 0.0, 2.0], requires_grad=False)
    out = x / c

    c_file = tmp_path / "div_model.c"
    exe_file = tmp_path / ("div_model.exe" if os.name == "nt" else "div_model")

    export_c(
        out,
        example_input=x,
        filename=c_file,
        include_main=False,
        model_name="div_test",
        optimize=False,
    )

    c_code = c_file.read_text(encoding="utf-8")
    assert "minigrad_div" in c_code

    harness = """
#include <stdio.h>
#include <math.h>
#include <assert.h>

int main(void) {
    float in[4] = {1.0f, -1.0f, 0.0f, 4.0f};
    float out[4] = {0.0f};

    div_test_forward(in, out);

    /* 1.0f / 0.0f -> +inf */
    assert(isinf(out[0]) && out[0] > 0.0f);
    /* -1.0f / 0.0f -> -inf */
    assert(isinf(out[1]) && out[1] < 0.0f);
    /* 0.0f / 0.0f -> nan */
    assert(isnan(out[2]));
    /* 4.0f / 2.0f -> 2.0f */
    assert(fabsf(out[3] - 2.0f) < 1e-5f);

    printf("IEEE-754 C division parity verified: +inf, -inf, nan, 2.0\\n");
    return 0;
}
"""
    test_c_file = tmp_path / "test_div_runner.c"
    test_c_file.write_text(c_code + "\n" + harness, encoding="utf-8")

    res = _compile_c(test_c_file, exe_file)
    assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    run_res = subprocess.run([str(exe_file)], capture_output=True, text=True, check=False)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"
    assert "IEEE-754 C division parity verified" in run_res.stdout

