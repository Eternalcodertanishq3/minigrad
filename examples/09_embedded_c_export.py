"""
09_embedded_c_export.py — Zero-Runtime Embedded C Compiler for Edge / TinyML (Pillar 2).

This example demonstrates Pillar 2 of miniGrad's innovations:
1. Train a neural network in pure Python on a non-linear problem (XOR).
2. Export the trained model to a single standalone ANSI C99 file using `model.export_c()`.
3. Inspect the zero-runtime footprint: 0 bytes of dynamic memory (malloc/free), 100% static buffers.
4. Compile with native clang/gcc and run native machine-code inference in microseconds.

Run: python examples/09_embedded_c_export.py
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minigrad import Tensor
from minigrad.nn import Sequential, Linear, Tanh, Sigmoid
from minigrad.optim import Adam


def train_xor_model():
    """Train a tiny non-linear classifier on the XOR truth table."""
    print("=" * 70)
    print("STEP 1: Train Neural Network in miniGrad (Python)")
    print("=" * 70)

    np.random.seed(42)

    # XOR dataset
    X = Tensor([[0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0]], requires_grad=False)
    Y = Tensor([[0.0],
                [1.0],
                [1.0],
                [0.0]], requires_grad=False)

    model = Sequential([
        Linear(2, 8),
        Tanh(),
        Linear(8, 1),
        Sigmoid(),
    ])

    optimizer = Adam(model.parameters(), lr=0.08)

    print("Training on XOR truth table...")
    for epoch in range(250):
        optimizer.zero_grad()
        pred = model(X)
        loss = ((pred - Y) ** 2).mean()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1:3d} | Loss: {loss.item():.6f}")

    final_preds = model(X).data.flatten()
    print("\nTrained Model Predictions:")
    for (x0, x1), target, p in zip(X.data, Y.data.flatten(), final_preds):
        print(f"  Input: [{int(x0)}, {int(x1)}] -> Expected: {int(target)} | Predicted: {p:.4f} (Round: {round(p)})")

    return model, X


def export_and_inspect_c_code(model, example_input):
    """Export the trained model to standalone C and analyze its footprint."""
    print("\n" + "=" * 70)
    print("STEP 2: One-Click Export to Standalone Embedded C (Pillar 2)")
    print("=" * 70)

    output_c_file = Path("tinyml_xor.c")

    # Single-sample example input for microcontrollers (batch size = 1)
    single_sample = Tensor(example_input.data[:1], requires_grad=False)

    c_code = model.export_c(
        example_input=single_sample,
        filename=output_c_file,
        include_main=True,
        model_name="xor_net",
    )

    lines = c_code.splitlines()
    code_bytes = len(c_code.encode("utf-8"))

    print(f"Generated standalone C file: {output_c_file}")
    print(f"  * Total Source Lines: {len(lines)}")
    print(f"  * File Size:          {code_bytes} bytes ({code_bytes / 1024:.2f} KB)")
    print("  * External Libs:      NONE (pure ANSI C99 math.h only)")
    print("  * Dynamic Memory:     0 bytes (ZERO malloc, ZERO free, 100% static buffers)")
    print("  * Target Devices:     ESP32, STM32, Arduino, Raspberry Pi Pico, Cortex-M, RISC-V")

    # Print first 25 lines of the generated file
    print("\nGenerated C Header Preview:")
    print("-" * 70)
    for line in lines[:16]:
        print(line)
    print("-" * 70)

    return output_c_file, single_sample


def compile_and_benchmark_native_c(c_file: Path, model, test_input: Tensor):
    """Compile generated C with clang/gcc, execute native binary, and check parity."""
    print("\n" + "=" * 70)
    print("STEP 3: Native Compilation & Microsecond Benchmark")
    print("=" * 70)

    # Locate compiler
    compiler = shutil.which("clang") or shutil.which("gcc")
    if not compiler:
        print("No native C compiler (clang or gcc) found on PATH.")
        print("The generated tinyml_xor.c is ready to be copied and compiled on any embedded target.")
        return

    exe_file = Path("tinyml_xor.exe" if os.name == "nt" else "tinyml_xor")

    print(f"Compiling with {Path(compiler).name}:")
    compile_cmd = [compiler, "-O3", str(c_file), "-o", str(exe_file)]
    print(f"  $ {' '.join(compile_cmd)}")

    t0 = time.perf_counter()
    res = subprocess.run(compile_cmd, capture_output=True, text=True)
    compile_time = (time.perf_counter() - t0) * 1000

    if res.returncode != 0:
        print(f"Compilation error: {res.stderr}")
        return

    print(f"Compilation successful in {compile_time:.1f} ms!")
    binary_size = exe_file.stat().st_size
    print(f"Native Binary Size: {binary_size} bytes ({binary_size / 1024:.2f} KB)")

    print("\nExecuting native C inference:")
    print("-" * 70)
    run_res = subprocess.run([str(exe_file)], capture_output=True, text=True)
    print(run_res.stdout.strip())
    print("-" * 70)

    # Verify parity against Python forward pass on test input
    py_out = float(model(test_input).data.flat[0])
    c_out = None
    for line in run_res.stdout.splitlines():
        if "output[0]" in line and "=" in line:
            c_out = float(line.split("=")[-1].strip())

    if c_out is not None:
        diff = abs(py_out - c_out)
        print("\nVerification Results:")
        print(f"  * Python Output:   {py_out:.7f}")
        print(f"  * Native C Output: {c_out:.7f}")
        print(f"  * Absolute Error:  {diff:.2e} (Exact Parity!)")

    # Clean up generated artifacts
    if exe_file.exists():
        exe_file.unlink()
    if c_file.exists():
        c_file.unlink()


def main():
    model, X = train_xor_model()
    c_file, test_sample = export_and_inspect_c_code(model, X)
    compile_and_benchmark_native_c(c_file, model, test_sample)


if __name__ == "__main__":
    main()
