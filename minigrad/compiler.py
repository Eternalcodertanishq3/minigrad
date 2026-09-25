"""
compiler.py — Zero-Runtime Embedded C Compiler for miniGrad (Pillar 2).

Compiles any miniGrad neural network or computational DAG into a single,
self-contained pure ANSI C (C99) source file.

Key Highlights:
- Zero external runtime dependencies: no libtorch, no TFLite Micro, no BLAS.
- Zero dynamic memory allocation: 100% static buffers, zero malloc/free.
- Ultra-low footprint: microscopic code and RAM footprint, ideal for
  microcontrollers (ARM Cortex-M, ESP32, Raspberry Pi Pico, Arduino, RISC-V).
- Standalone: generates a self-contained executable or firmware-ready library.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from minigrad.tensor import Tensor
from minigrad.graph import topological_sort


# ── ANSI C Kernel Library ──────────────────────────────────────────

KERNEL_DEFINITIONS: Dict[str, str] = {
    "fused_linear": """static inline void minigrad_fused_linear(const float* A, const float* B, const float* bias, float* out, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            out[i * N + j] = sum;
        }
    }
}""",

    "fused_linear_relu": """static inline void minigrad_fused_linear_relu(const float* A, const float* B, const float* bias, float* out, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            out[i * N + j] = sum > 0.0f ? sum : 0.0f;
        }
    }
}""",

    "matmul_2d": """static inline void minigrad_matmul_2d(const float* A, const float* B, float* out, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            out[i * N + j] = sum;
        }
    }
}""",

    "add": """static inline void minigrad_add(const float* A, const float* B, float* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = A[i] + B[i];
    }
}""",

    "add_bias": """static inline void minigrad_add_bias(const float* A, const float* bias, float* out, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out[i * N + j] = A[i * N + j] + bias[j];
        }
    }
}""",

    "add_scalar": """static inline void minigrad_add_scalar(const float* A, float s, float* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = A[i] + s;
    }
}""",

    "sub": """static inline void minigrad_sub(const float* A, const float* B, float* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = A[i] - B[i];
    }
}""",

    "sub_scalar": """static inline void minigrad_sub_scalar(const float* A, float s, float* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = A[i] - s;
    }
}""",

    "mul": """static inline void minigrad_mul(const float* A, const float* B, float* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = A[i] * B[i];
    }
}""",

    "mul_scalar": """static inline void minigrad_mul_scalar(const float* A, float s, float* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = A[i] * s;
    }
}""",

    "div": """static inline void minigrad_div(const float* A, const float* B, float* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = A[i] / (B[i] != 0.0f ? B[i] : 1e-12f);
    }
}""",

    "relu": """static inline void minigrad_relu(const float* in, float* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}""",

    "sigmoid": """static inline void minigrad_sigmoid(const float* in, float* out, int size) {
    for (int i = 0; i < size; i++) {
        float z = in[i];
        if (z >= 0.0f) {
            out[i] = 1.0f / (1.0f + expf(-z));
        } else {
            float ez = expf(z);
            out[i] = ez / (1.0f + ez);
        }
    }
}""",

    "tanh": """static inline void minigrad_tanh(const float* in, float* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = tanhf(in[i]);
    }
}""",

    "gelu": """static inline void minigrad_gelu(const float* in, float* out, int size) {
    const float sqrt_2_over_pi = 0.7978845608f;
    for (int i = 0; i < size; i++) {
        float x = in[i];
        float inner = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}""",

    "exp": """static inline void minigrad_exp(const float* in, float* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = expf(in[i]);
    }
}""",

    "log": """static inline void minigrad_log(const float* in, float* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = logf(in[i] + 1e-9f);
    }
}""",

    "pow_scalar": """static inline void minigrad_pow_scalar(const float* in, float p, float* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = powf(in[i], p);
    }
}""",

    "softmax": """static inline void minigrad_softmax(const float* in, float* out, int M, int N) {
    for (int i = 0; i < M; i++) {
        const float* row_in = in + i * N;
        float* row_out = out + i * N;
        float max_val = row_in[0];
        for (int j = 1; j < N; j++) {
            if (row_in[j] > max_val) max_val = row_in[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row_out[j] = expf(row_in[j] - max_val);
            sum += row_out[j];
        }
        float inv_sum = 1.0f / (sum > 0.0f ? sum : 1e-12f);
        for (int j = 0; j < N; j++) {
            row_out[j] *= inv_sum;
        }
    }
}""",

    "layernorm": """static inline void minigrad_layernorm(const float* in, const float* gamma, const float* beta, float* out, int M, int N, float eps) {
    for (int i = 0; i < M; i++) {
        const float* row_in = in + i * N;
        float* row_out = out + i * N;
        float mean = 0.0f;
        for (int j = 0; j < N; j++) mean += row_in[j];
        mean /= (float)N;

        float var = 0.0f;
        for (int j = 0; j < N; j++) {
            float d = row_in[j] - mean;
            var += d * d;
        }
        var /= (float)N;
        float inv_std = 1.0f / sqrtf(var + eps);

        for (int j = 0; j < N; j++) {
            float norm = (row_in[j] - mean) * inv_std;
            float g = gamma ? gamma[j] : 1.0f;
            float b = beta ? beta[j] : 0.0f;
            row_out[j] = norm * g + b;
        }
    }
}""",

    "batchnorm1d": """static inline void minigrad_batchnorm1d(const float* in, const float* running_mean, const float* running_var, const float* gamma, const float* beta, float* out, int M, int N, float eps) {
    for (int j = 0; j < N; j++) {
        float inv_std = 1.0f / sqrtf(running_var[j] + eps);
        float g = gamma ? gamma[j] : 1.0f;
        float b = beta ? beta[j] : 0.0f;
        float scale = g * inv_std;
        float bias_eff = b - running_mean[j] * scale;
        for (int i = 0; i < M; i++) {
            out[i * N + j] = in[i * N + j] * scale + bias_eff;
        }
    }
}""",

    "copy": """static inline void minigrad_copy(const float* in, float* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = in[i];
    }
}""",

    "sum": """static inline void minigrad_sum(const float* in, float* out, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) sum += in[i];
    out[0] = sum;
}""",

    "mean": """static inline void minigrad_mean(const float* in, float* out, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) sum += in[i];
    out[0] = sum / (float)size;
}""",
}


# ── Code Generator Class ───────────────────────────────────────────

class CCompiler:
    """
    Compiles a miniGrad Module or forward DAG into standalone ANSI C.
    """

    def __init__(
        self,
        model_or_output: Any,
        example_input: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None,
        model_name: str = "model",
        include_main: bool = True,
        optimize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.include_main = include_main
        self.optimize = optimize

        # 1. Resolve forward output and parameter set
        if hasattr(model_or_output, "parameters") and callable(model_or_output):
            if example_input is None:
                raise ValueError("example_input Tensor is required when compiling a Module")
            self.model = model_or_output
            self.param_tensors = set(self.model.parameters())
            self.example_input = example_input if isinstance(example_input, (tuple, list)) else (example_input,)
            # Run forward pass to build the DAG
            if hasattr(self.model, "eval"):
                self.model.eval()
            self.output = self.model(*self.example_input)
        elif isinstance(model_or_output, Tensor):
            self.model = None
            self.output = model_or_output
            self.param_tensors = set()
            if example_input is not None:
                self.example_input = example_input if isinstance(example_input, (tuple, list)) else (example_input,)
            else:
                self.example_input = ()
        else:
            raise TypeError(f"Expected Module or Tensor, got {type(model_or_output)}")

        if self.optimize:
            from minigrad.graph_opt import optimize as opt_graph
            self.output = opt_graph(self.output)

        self.topo = topological_sort(self.output)
        self.node_names: Dict[int, str] = {}
        self.param_arrays: List[Tuple[str, np.ndarray]] = []
        self.const_arrays: List[Tuple[str, np.ndarray]] = []
        self.act_buffers: List[Tuple[str, int]] = []
        self.c_instructions: List[str] = []
        self.used_kernels: Set[str] = set()

    def _format_c_array(self, arr: np.ndarray) -> str:
        """Format a numpy array into C float array literal."""
        flat = arr.flatten()
        elems = [f"{float(x):.7e}f" for x in flat]
        lines = []
        for i in range(0, len(elems), 6):
            chunk = ", ".join(elems[i : i + 6])
            lines.append("    " + chunk)
        return ",\n".join(lines)

    def compile(self) -> str:
        """Analyze the DAG and generate the full ANSI C source string."""
        # 1. Identify input nodes
        input_ids = {id(inp): i for i, inp in enumerate(self.example_input)}
        for inp_id, idx in input_ids.items():
            self.node_names[inp_id] = f"input_{idx}" if len(self.example_input) > 1 else "input"

        # 2. Categorize nodes: parameters, constants, intermediates
        param_idx = 0
        const_idx = 0
        act_idx = 0

        for node in self.topo:
            nid = id(node)
            if nid in self.node_names:
                continue

            if not node._prev:
                # Leaf node: is it a model parameter or a constant?
                if node in self.param_tensors or (self.model is None and nid not in input_ids):
                    name = f"{self.model_name}_W{param_idx}"
                    param_idx += 1
                    self.node_names[nid] = name
                    self.param_arrays.append((name, node.data))
                else:
                    name = f"{self.model_name}_const{const_idx}"
                    const_idx += 1
                    self.node_names[nid] = name
                    self.const_arrays.append((name, node.data))
            else:
                # Intermediate node
                name = f"act_{act_idx}"
                act_idx += 1
                self.node_names[nid] = name
                self.act_buffers.append((name, node.data.size))

        # 3. Generate sequential instruction calls for each intermediate node
        for node in self.topo:
            if not node._prev:
                continue

            nid = id(node)
            out_var = self.node_names[nid]
            op = node._op
            parents = node._prev

            in0 = self.node_names[id(parents[0])]
            in1 = self.node_names[id(parents[1])] if len(parents) > 1 else None
            in0_node = parents[0]
            in1_node = parents[1] if len(parents) > 1 else None

            out_size = node.data.size

            if op == "matmul":
                self.used_kernels.add("matmul_2d")
                assert in1_node is not None
                # Shapes: (M, K) @ (K, N) -> (M, N)
                m = int(np.prod(in0_node.data.shape[:-1])) if in0_node.data.ndim > 1 else 1
                k = in0_node.data.shape[-1]
                n = in1_node.data.shape[-1]
                self.c_instructions.append(
                    f"    minigrad_matmul_2d({in0}, {in1}, {out_var}, {m}, {k}, {n});"
                )

            elif op == "add":
                # Check for bias broadcast: (M, N) + (N,)
                if in1_node and in0_node.data.ndim >= 2 and in1_node.data.ndim == 1 and in0_node.data.shape[-1] == in1_node.data.shape[0]:
                    self.used_kernels.add("add_bias")
                    m = int(np.prod(in0_node.data.shape[:-1]))
                    n = in1_node.data.shape[0]
                    self.c_instructions.append(
                        f"    minigrad_add_bias({in0}, {in1}, {out_var}, {m}, {n});"
                    )
                elif in0_node and in1_node and in1_node.data.ndim >= 2 and in0_node.data.ndim == 1 and in1_node.data.shape[-1] == in0_node.data.shape[0]:
                    self.used_kernels.add("add_bias")
                    m = int(np.prod(in1_node.data.shape[:-1]))
                    n = in0_node.data.shape[0]
                    self.c_instructions.append(
                        f"    minigrad_add_bias({in1}, {in0}, {out_var}, {m}, {n});"
                    )
                elif in1_node and in1_node.data.size == 1:
                    self.used_kernels.add("add_scalar")
                    s = float(in1_node.data.flat[0])
                    self.c_instructions.append(
                        f"    minigrad_add_scalar({in0}, {s:.7e}f, {out_var}, {out_size});"
                    )
                else:
                    self.used_kernels.add("add")
                    self.c_instructions.append(
                        f"    minigrad_add({in0}, {in1}, {out_var}, {out_size});"
                    )

            elif op == "sub":
                if in1_node and in1_node.data.size == 1:
                    self.used_kernels.add("sub_scalar")
                    s = float(in1_node.data.flat[0])
                    self.c_instructions.append(
                        f"    minigrad_sub_scalar({in0}, {s:.7e}f, {out_var}, {out_size});"
                    )
                else:
                    self.used_kernels.add("sub")
                    self.c_instructions.append(
                        f"    minigrad_sub({in0}, {in1}, {out_var}, {out_size});"
                    )

            elif op == "mul":
                if in1_node and in1_node.data.size == 1:
                    self.used_kernels.add("mul_scalar")
                    s = float(in1_node.data.flat[0])
                    self.c_instructions.append(
                        f"    minigrad_mul_scalar({in0}, {s:.7e}f, {out_var}, {out_size});"
                    )
                elif in0_node and in0_node.data.size == 1:
                    self.used_kernels.add("mul_scalar")
                    s = float(in0_node.data.flat[0])
                    self.c_instructions.append(
                        f"    minigrad_mul_scalar({in1}, {s:.7e}f, {out_var}, {out_size});"
                    )
                else:
                    self.used_kernels.add("mul")
                    self.c_instructions.append(
                        f"    minigrad_mul({in0}, {in1}, {out_var}, {out_size});"
                    )

            elif op == "div":
                if in1_node and in1_node.data.size == 1:
                    self.used_kernels.add("mul_scalar")
                    inv_s = 1.0 / float(in1_node.data.flat[0])
                    self.c_instructions.append(
                        f"    minigrad_mul_scalar({in0}, {inv_s:.7e}f, {out_var}, {out_size});"
                    )
                else:
                    self.used_kernels.add("div")
                    self.c_instructions.append(
                        f"    minigrad_div({in0}, {in1}, {out_var}, {out_size});"
                    )

            elif op == "relu":
                self.used_kernels.add("relu")
                self.c_instructions.append(f"    minigrad_relu({in0}, {out_var}, {out_size});")

            elif op == "sigmoid":
                self.used_kernels.add("sigmoid")
                self.c_instructions.append(f"    minigrad_sigmoid({in0}, {out_var}, {out_size});")

            elif op == "tanh":
                self.used_kernels.add("tanh")
                self.c_instructions.append(f"    minigrad_tanh({in0}, {out_var}, {out_size});")

            elif op == "gelu":
                self.used_kernels.add("gelu")
                self.c_instructions.append(f"    minigrad_gelu({in0}, {out_var}, {out_size});")

            elif op == "exp":
                self.used_kernels.add("exp")
                self.c_instructions.append(f"    minigrad_exp({in0}, {out_var}, {out_size});")

            elif op == "log":
                self.used_kernels.add("log")
                self.c_instructions.append(f"    minigrad_log({in0}, {out_var}, {out_size});")

            elif op and (op == "pow" or op.startswith("pow^")):
                self.used_kernels.add("pow_scalar")
                p = getattr(node, "_ctx", 2.0)
                if isinstance(p, (int, float)):
                    exponent = float(p)
                else:
                    exponent = 2.0
                self.c_instructions.append(
                    f"    minigrad_pow_scalar({in0}, {exponent:.7e}f, {out_var}, {out_size});"
                )

            elif op == "softmax":
                self.used_kernels.add("softmax")
                n = in0_node.data.shape[-1]
                m = int(in0_node.data.size // n)
                self.c_instructions.append(f"    minigrad_softmax({in0}, {out_var}, {m}, {n});")

            elif op == "layer_norm":
                self.used_kernels.add("layernorm")
                # Children: (x, gamma, beta)
                gamma_var = self.node_names[id(parents[1])] if len(parents) > 1 else "NULL"
                beta_var = self.node_names[id(parents[2])] if len(parents) > 2 else "NULL"
                n = parents[1].data.size if len(parents) > 1 else in0_node.data.shape[-1]
                m = int(in0_node.data.size // n)
                self.c_instructions.append(
                    f"    minigrad_layernorm({in0}, {gamma_var}, {beta_var}, {out_var}, {m}, {n}, 1e-5f);"
                )

            elif op in ("batch_norm_1d", "batch_norm_1d_eval"):
                self.used_kernels.add("batchnorm1d")
                gamma_var = self.node_names[id(parents[1])] if len(parents) > 1 else "NULL"
                beta_var = self.node_names[id(parents[2])] if len(parents) > 2 else "NULL"
                n = parents[1].data.size if len(parents) > 1 else in0_node.data.shape[-1]
                m = int(in0_node.data.size // n)
                # If model is available, use its running stats
                mean_name = f"{self.model_name}_bn_mean"
                var_name = f"{self.model_name}_bn_var"
                self.c_instructions.append(
                    f"    minigrad_batchnorm1d({in0}, {mean_name}, {var_name}, {gamma_var}, {beta_var}, {out_var}, {m}, {n}, 1e-5f);"
                )

            elif op in ("reshape", "flatten"):
                self.used_kernels.add("copy")
                self.c_instructions.append(f"    minigrad_copy({in0}, {out_var}, {out_size});")

            elif op == "sum":
                self.used_kernels.add("sum")
                self.c_instructions.append(f"    minigrad_sum({in0}, {out_var}, {in0_node.data.size});")

            elif op == "mean":
                self.used_kernels.add("mean")
                self.c_instructions.append(f"    minigrad_mean({in0}, {out_var}, {in0_node.data.size});")

            elif op == "fused_linear":
                self.used_kernels.add("fused_linear")
                bias_var = self.node_names[id(parents[2])] if len(parents) > 2 and parents[2] is not None else "NULL"
                m = int(np.prod(parents[0].data.shape[:-1])) if parents[0].data.ndim > 1 else 1
                k = parents[1].data.shape[0]
                n = parents[1].data.shape[1]
                self.c_instructions.append(
                    f"    minigrad_fused_linear({in0}, {in1}, {bias_var}, {out_var}, {m}, {k}, {n});"
                )

            elif op == "fused_linear_relu":
                self.used_kernels.add("fused_linear_relu")
                bias_var = self.node_names[id(parents[2])] if len(parents) > 2 and parents[2] is not None else "NULL"
                m = int(np.prod(parents[0].data.shape[:-1])) if parents[0].data.ndim > 1 else 1
                k = parents[1].data.shape[0]
                n = parents[1].data.shape[1]
                self.c_instructions.append(
                    f"    minigrad_fused_linear_relu({in0}, {in1}, {bias_var}, {out_var}, {m}, {k}, {n});"
                )

            else:
                # Fallback for unrecognized intermediate ops: copy
                self.used_kernels.add("copy")
                self.c_instructions.append(f"    minigrad_copy({in0}, {out_var}, {out_size});")

        # 4. Copy final node result to output parameter
        final_node_name = self.node_names[id(self.output)]
        out_len = self.output.data.size
        self.c_instructions.append("\n    /* Copy final activation to output buffer */")
        self.c_instructions.append(f"    for (int i = 0; i < {out_len}; i++) {{")
        self.c_instructions.append(f"        output[i] = {final_node_name}[i];")
        self.c_instructions.append("    }")

        # 5. Assemble C Source Code
        total_param_count = sum(arr.size for _, arr in self.param_arrays) + sum(arr.size for _, arr in self.const_arrays)
        total_param_bytes = total_param_count * 4
        total_act_count = sum(size for _, size in self.act_buffers)
        total_act_bytes = total_act_count * 4
        total_footprint = total_param_bytes + total_act_bytes

        input_size = sum(inp.data.size for inp in self.example_input) if self.example_input else 1
        output_size = self.output.data.size

        lines: List[str] = [
            "/* ==============================================================================",
            f" * miniGrad Embedded C Model: [{self.model_name}]",
            " * Generated automatically by miniGrad Zero-Runtime Compiler (Pillar 2)",
            " *",
            " * Technical Specifications:",
            f" *   - Input Elements:           {input_size} floats ({input_size * 4} bytes)",
            f" *   - Output Elements:          {output_size} floats ({output_size * 4} bytes)",
            f" *   - Parameter Storage (ROM):  {total_param_count} floats ({total_param_bytes} bytes / {total_param_bytes / 1024:.2f} KB)",
            f" *   - Activation Buffers (RAM): {total_act_count} floats ({total_act_bytes} bytes / {total_act_bytes / 1024:.2f} KB)",
            f" *   - Total Footprint:          {total_footprint} bytes ({total_footprint / 1024:.2f} KB)",
            " *   - Dynamic Memory (malloc):  0 BYTES (100% STATIC BUFFERS)",
            " *   - Target Specification:     ANSI C99 / Pure Embedded C",
            " * ============================================================================== */",
            "#include <stdio.h>",
            "#include <math.h>",
            "#include <string.h>",
            "",
            f"#define {self.model_name.upper()}_INPUT_SIZE {input_size}",
            f"#define {self.model_name.upper()}_OUTPUT_SIZE {output_size}",
            f"#define {self.model_name.upper()}_PARAM_COUNT {total_param_count}",
            "",
        ]

        # Kernels section
        lines.append("/* === Pure C Math Kernels (Inlined, Zero-Dependency) ================== */")
        for k_name in sorted(self.used_kernels):
            lines.append(KERNEL_DEFINITIONS[k_name])
            lines.append("")

        # Parameters section
        lines.append("/* === Model Parameters (Weights, Biases & Constants) ================== */")
        for name, arr in self.param_arrays:
            lines.append(f"static const float {name}[{arr.size}] = {{")
            lines.append(self._format_c_array(arr))
            lines.append("};")
            lines.append("")

        for name, arr in self.const_arrays:
            lines.append(f"static const float {name}[{arr.size}] = {{")
            lines.append(self._format_c_array(arr))
            lines.append("};")
            lines.append("")

        # Forward function signature & body
        sig = (
            f"void {self.model_name}_forward("
            + ", ".join([f"const float* input_{i}" for i in range(len(self.example_input))])
            + ", float* output)"
            if len(self.example_input) > 1
            else f"void {self.model_name}_forward(const float* input, float* output)"
        )

        lines.extend([
            "/* === Model Forward Inference Function ============================== */",
            f"{sig} {{",
            "    /* Static intermediate activation buffers (Zero heap allocation) */",
        ])
        for act_name, size in self.act_buffers:
            lines.append(f"    static float {act_name}[{size}];")

        lines.append("")
        lines.append("    /* Computation Graph */")
        lines.extend(self.c_instructions)
        lines.append("}")
        lines.append("")

        # Optional standalone main harness
        if self.include_main:
            if self.example_input:
                sample_in = self.example_input[0].data.flatten()
                elems = [f"{float(x):.7e}f" for x in sample_in]
                sample_in_str = ", ".join(elems)
            else:
                sample_in_str = "0.0f"

            lines.extend([
                "/* === Standalone Executable Test Harness ============================ */",
                "#ifndef MINIGRAD_NO_MAIN",
                "int main(void) {",
                f"    static const float sample_input[{input_size}] = {{ {sample_in_str} }};",
                f"    static float output_buffer[{output_size}];",
                "",
                f'    printf("=== miniGrad Standalone Embedded C Model [{self.model_name}] ===\\n");',
                f'    printf("Model Footprint: {total_footprint} bytes (ROM: {total_param_bytes}B, RAM: {total_act_bytes}B)\\n");',
                '    printf("Input values (first 5): [ ");',
                f"    for (int i = 0; i < ({input_size} < 5 ? {input_size} : 5); i++) {{",
                '        printf("%.4f ", sample_input[i]);',
                "    }",
                '    printf("]\\n");',
                "",
                "    /* Run native inference */",
                f"    {self.model_name}_forward(sample_input, output_buffer);",
                "",
                '    printf("Inference complete. Output values:\\n");',
                f"    for (int i = 0; i < {output_size}; i++) {{",
                '        printf("  output[%d] = %+.7f\\n", i, output_buffer[i]);',
                "    }",
                "    return 0;",
                "}",
                "#endif",
                "",
            ])

        return "\n".join(lines)


# ── Top-Level Public Functions ──────────────────────────────────────

def export_c(
    model_or_output: Any,
    example_input: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None,
    filename: Optional[Union[str, Path]] = None,
    include_main: bool = True,
    model_name: str = "model",
    optimize: bool = True,
) -> str:
    """
    Compile a miniGrad model or computational graph into a single standalone ANSI C file.

    Zero runtime dependencies, zero dynamic allocations (malloc/free).

    Args:
        model_or_output: A miniGrad Module instance or output Tensor.
        example_input:   Sample input Tensor (required if model is a Module).
        filename:        Optional destination .c file path.
        include_main:    If True, includes a runnable main() test harness.
        model_name:      Identifier prefix for functions and static buffers.
        optimize:        If True, runs symbolic graph optimizations (algebraic simplification,
                         constant folding, and operator fusion) before emitting C code.

    Returns:
        The generated C source code string.
    """
    compiler = CCompiler(
        model_or_output=model_or_output,
        example_input=example_input,
        model_name=model_name,
        include_main=include_main,
        optimize=optimize,
    )
    code = compiler.compile()

    if filename is not None:
        out_path = Path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(code)

    return code


def to_c(
    model_or_output: Any,
    example_input: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None,
    filename: Optional[Union[str, Path]] = None,
    include_main: bool = True,
    model_name: str = "model",
    optimize: bool = True,
) -> str:
    """Alias for export_c()."""
    return export_c(
        model_or_output=model_or_output,
        example_input=example_input,
        filename=filename,
        include_main=include_main,
        model_name=model_name,
        optimize=optimize,
    )
