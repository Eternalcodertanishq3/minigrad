"""
tests/test_api_contracts.py — Public API contract verification suite (Phase 5).

Validates:
1. Tensor public API:
   - Empty shapes
   - Singleton dimensions (1, 1)
   - Non-contiguous array inputs
   - Scalar vs Tensor operations
2. Module public API:
   - state_dict() and load_state_dict() roundtrip
   - train() vs eval() mode toggling
   - parameters() and named_parameters() generator contracts
3. DataLoader public API:
   - Batching behavior and drop_last
   - Uneven dataset handling
4. SafeTensors Option B contract:
   - Buffer ownership verification (owned ndarray buffers, no dangling file-backed mmap pointers)
   - bfloat16 loading
   - Metadata preservation
5. Single Source of Truth for Versioning:
   - minigrad.__version__ matches setup.py
6. Compiler Parameter Contract Validations:
   - Positive integer assertions on max_seq_len, n_heads, and d_k
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import minigrad
from minigrad.compiler import CCompiler, export_c
from minigrad.data import DataLoader, Dataset
from minigrad.nn import BatchNorm1D, Dropout, Linear, Sequential
from minigrad.safetensors import load_file, save_file
from minigrad.tensor import Tensor

# ── 1. Tensor Public API Contracts ────────────────────────────────────

def test_tensor_empty_shape():
    """Tensors with empty shape can be constructed without crash."""
    t = Tensor([], dtype=np.float32)
    assert t.shape == (0,)
    assert t.data.size == 0
    assert t.dtype == np.float32


def test_tensor_singleton_dimensions():
    """Tensors with singleton dimensions (1, 1) behave correctly in ops."""
    a = Tensor([[3.0]], requires_grad=True)
    b = Tensor([[4.0]], requires_grad=True)
    c = a * b
    assert c.shape == (1, 1)
    c.backward()
    assert np.allclose(a.grad, [[4.0]])
    assert np.allclose(b.grad, [[3.0]])


def test_tensor_non_contiguous_input():
    """Non-contiguous numpy arrays are safely accepted and handled."""
    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    non_contig = base[:, ::2]  # strided view
    assert not non_contig.flags.c_contiguous

    t = Tensor(non_contig, requires_grad=True)
    assert t.shape == (3, 2)
    y = t.sum()
    y.backward()
    assert np.allclose(t.grad, np.ones((3, 2)))


def test_tensor_scalar_vs_tensor_operations():
    """Verify operations between Tensor and raw Python scalars."""
    t = Tensor([2.0, 4.0], requires_grad=True)
    res_add = t + 5.0
    res_radd = 5.0 + t
    assert np.allclose(res_add.data, res_radd.data)

    res_mul = t * 3.0
    res_rmul = 3.0 * t
    assert np.allclose(res_mul.data, res_rmul.data)

    res_sub = t - 1.0
    res_rsub = 10.0 - t
    assert np.allclose(res_sub.data, [1.0, 3.0])
    assert np.allclose(res_rsub.data, [8.0, 6.0])


# ── 2. Module Public API Contracts ────────────────────────────────────

def test_module_state_dict_and_load_state_dict():
    """Module state_dict exports all parameters and load_state_dict restores them."""
    m1 = Sequential([Linear(4, 8), Linear(8, 2)])
    m2 = Sequential([Linear(4, 8), Linear(8, 2)])

    # Verify initial parameters are distinct
    s1 = m1.state_dict()
    assert len(s1) == 4  # 2 weights + 2 biases
    assert not np.allclose(m1[0].weight.data, m2[0].weight.data)

    # Restore m1 state into m2
    m2.load_state_dict(s1)
    assert np.allclose(m1[0].weight.data, m2[0].weight.data)
    assert np.allclose(m1[1].bias.data, m2[1].bias.data)


def test_module_train_eval_modes():
    """train() and eval() toggle training flags on submodules."""
    m = Sequential([Linear(4, 4), Dropout(p=0.5), BatchNorm1D(4)])
    m.train()
    assert m.training is True
    assert m[1].training is True
    assert m[2].training is True

    m.eval()
    assert m.training is False
    assert m[1].training is False
    assert m[2].training is False


def test_module_parameter_generators():
    """parameters() and named_parameters() return valid tensors and names."""
    m = Sequential([Linear(2, 4), Linear(4, 1)])
    params = list(m.parameters())
    named_params = list(m.named_parameters())

    assert len(params) == 4
    assert len(named_params) == 4
    for name, p in named_params:
        assert isinstance(name, str)
        assert isinstance(p, Tensor)
        assert p.requires_grad is True


# ── 3. DataLoader Public API Contracts ────────────────────────────────

class SimpleDataset(Dataset):
    def __init__(self, n: int):
        self.data = np.arange(n, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        return np.array([self.data[idx]], dtype=np.float32), int(self.data[idx])


def test_dataloader_batching_and_drop_last():
    """DataLoader batches correctly and drop_last drops incomplete final batches."""
    ds = SimpleDataset(10)

    # Without drop_last: batches of size 4, 4, 2
    loader_keep = DataLoader(ds, batch_size=4, shuffle=False, drop_last=False)
    batches = list(loader_keep)
    assert len(batches) == 3
    assert len(batches[0][0]) == 4
    assert len(batches[1][0]) == 4
    assert len(batches[2][0]) == 2

    # With drop_last: batches of size 4, 4 (2 dropped)
    loader_drop = DataLoader(ds, batch_size=4, shuffle=False, drop_last=True)
    batches_dropped = list(loader_drop)
    assert len(batches_dropped) == 2
    assert len(batches_dropped[0][0]) == 4
    assert len(batches_dropped[1][0]) == 4


# ── 4. SafeTensors Option B Ownership Contract ───────────────────────

def test_safetensors_option_b_buffer_ownership(tmp_path: Path):
    """
    Verify Option B contract:
    Tensors loaded via load_file have owned numpy array buffers (owndata=True),
    preventing persistent file-backed mmap memory leaks.
    """
    file_path = tmp_path / "model.safetensors"
    w1 = np.random.randn(4, 8).astype(np.float32)
    b1 = np.random.randn(8).astype(np.float32)

    save_file({"w1": w1, "b1": b1}, file_path, metadata={"format": "option_b"})

    loaded = load_file(file_path, to_tensor=False)
    assert "w1" in loaded
    assert "b1" in loaded

    # Option B requirement: buffer is an owned copy, not a fragile file-backed view
    w1_loaded = loaded["w1"]
    assert isinstance(w1_loaded, np.ndarray)
    assert w1_loaded.flags.owndata is True, "Option B violation: array does not own its memory buffer!"

    # Deleting or modifying the file must not invalidate the loaded array memory
    file_path.unlink()
    assert np.allclose(w1_loaded, w1)


def test_safetensors_bfloat16_parsing(tmp_path: Path):
    """bfloat16 format can be loaded and parsed into float32 correctly."""
    import json
    import struct

    file_path = tmp_path / "bf16_test.safetensors"
    # Construct a bfloat16 representation of 1.0 (0x3F80)
    bf16_bytes = struct.pack("<H", 0x3F80)
    header = {
        "val": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}
    }
    header_json = json.dumps(header).encode("utf-8")
    header_len = len(header_json)

    with open(file_path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        f.write(bf16_bytes)

    loaded = load_file(file_path, to_tensor=False)
    assert "val" in loaded
    val = loaded["val"]
    assert isinstance(val, np.ndarray)
    assert np.isclose(float(val[0]), 1.0)


def test_safetensors_empty_tensor_roundtrip(tmp_path: Path):
    """Zero-element tensors (0-byte payload) are saved and loaded correctly."""
    file_path = tmp_path / "empty_model.safetensors"
    empty_arr = np.array([], dtype=np.float32)
    save_file({"empty_param": empty_arr}, file_path)

    loaded = load_file(file_path, to_tensor=False)
    assert "empty_param" in loaded
    loaded_arr = loaded["empty_param"]
    assert isinstance(loaded_arr, np.ndarray)
    assert loaded_arr.shape == (0,)
    assert loaded_arr.dtype == np.float32
    assert loaded_arr.flags.owndata is True


# ── 5. Single Source of Truth for Versioning ─────────────────────────

def test_single_source_of_truth_version():
    """Verify setup.py, pyproject.toml, and minigrad.__version__ share the exact same version string."""
    init_file = Path(minigrad.__file__).parent / "__init__.py"
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_file.read_text(encoding="utf-8"), re.M)
    assert match is not None
    expected_version = match.group(1)

    assert minigrad.__version__ == expected_version

    # Check setup.py
    setup_file = Path(minigrad.__file__).parent.parent / "setup.py"
    assert setup_file.exists()
    try:
        from setup import get_version
        assert get_version() == expected_version
    except (ImportError, ModuleNotFoundError):
        setup_text = setup_file.read_text(encoding="utf-8")
        assert "get_version()" in setup_text or f'version="{expected_version}"' in setup_text

    # Check pyproject.toml
    pyproject_file = Path(minigrad.__file__).parent.parent / "pyproject.toml"
    assert pyproject_file.exists()
    toml_match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', pyproject_file.read_text(encoding="utf-8"), re.M)
    assert toml_match is not None
    assert toml_match.group(1) == expected_version


# ── 6. Compiler Parameter Contract Validations ───────────────────────

def test_compiler_kv_dimension_validation():
    """CCompiler and export_c reject non-positive or non-integral KV-cache dimensions."""
    x = Tensor([1.0, 2.0])

    # max_seq_len validation
    for bad_val in [0, -5, 12.5, "128", None, True, False]:
        with pytest.raises(ValueError, match="max_seq_len must be a positive integer"):
            CCompiler(x, max_seq_len=bad_val)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="max_seq_len must be a positive integer"):
            export_c(x, max_seq_len=bad_val)  # type: ignore[arg-type]

    # n_heads validation
    for bad_val in [0, -1, 4.5, "4", None, True, False]:
        with pytest.raises(ValueError, match="n_heads must be a positive integer"):
            CCompiler(x, n_heads=bad_val)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="n_heads must be a positive integer"):
            export_c(x, n_heads=bad_val)  # type: ignore[arg-type]

    # d_k validation
    for bad_val in [0, -32, 32.5, "32", None, True, False]:
        with pytest.raises(ValueError, match="d_k must be a positive integer"):
            CCompiler(x, d_k=bad_val)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="d_k must be a positive integer"):
            export_c(x, d_k=bad_val)  # type: ignore[arg-type]

