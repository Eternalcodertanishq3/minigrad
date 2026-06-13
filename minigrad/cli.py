"""Command-line entry points for miniGrad examples and benchmarks."""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_script(relative_path: str) -> None:
    runpy.run_path(str(PROJECT_ROOT / relative_path), run_name="__main__")


def demo_scalar() -> None:
    _run_script("examples/01_scalar_autograd.py")


def demo_linear_regression() -> None:
    _run_script("examples/02_linear_regression.py")


def demo_xor() -> None:
    _run_script("examples/03_mlp_xor.py")


def bench_ops() -> None:
    _run_script("benchmarks/bench_ops.py")


def main() -> None:
    parser = argparse.ArgumentParser(prog="minigrad", description="Run miniGrad demos and benchmarks.")
    parser.add_argument(
        "command",
        choices=("scalar", "linear-regression", "xor", "bench-ops"),
        help="Demo or benchmark to run.",
    )
    args = parser.parse_args()

    commands = {
        "scalar": demo_scalar,
        "linear-regression": demo_linear_regression,
        "xor": demo_xor,
        "bench-ops": bench_ops,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
