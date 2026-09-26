"""
test_safetensors.py — Unit tests and Hugging Face parity tests for safetensors reader/writer.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from minigrad.nn import Linear, Sequential
from minigrad.safetensors import load_file, safe_open, save_file
from minigrad.tensor import Tensor


def test_roundtrip_basic():
    """Verify saving and loading standard tensors preserves values, shapes, and dtypes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "model.safetensors"

        tensors = {
            "weight": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
            "bias": np.array([0.1, 0.2, 0.3], dtype=np.float64),
            "mask": np.array([True, False, True], dtype=np.bool_),
            "indices": np.array([10, 20, 30], dtype=np.int64),
        }

        save_file(tensors, filepath)
        assert filepath.exists()

        loaded = load_file(filepath)
        assert set(loaded.keys()) == set(tensors.keys())

        for k in tensors:
            assert np.array_equal(loaded[k], tensors[k]), f"Mismatch for key {k}"
            assert loaded[k].dtype == tensors[k].dtype, f"Dtype mismatch for key {k}"


def test_metadata_preservation():
    """Verify custom string metadata is stored and retrieved correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "model.safetensors"

        tensors = {"x": np.array([1.0, 2.0], dtype=np.float32)}
        metadata = {"format": "minigrad", "author": "antigravity", "version": "1.0"}

        save_file(tensors, filepath, metadata=metadata)

        with safe_open(filepath) as f:
            read_meta = f.metadata()
            assert read_meta == metadata
            assert f.keys() == ["x"]
            val = f.get_tensor("x")
            assert np.allclose(val, [1.0, 2.0])


def test_to_tensor_mode():
    """Verify load_file with to_tensor=True creates valid trainable Tensors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "model.safetensors"

        tensors = {"w": Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))}
        save_file(tensors, filepath)

        loaded = load_file(filepath, to_tensor=True)
        assert isinstance(loaded["w"], Tensor)
        assert loaded["w"].requires_grad is True

        # Test autograd on loaded tensor
        out = (loaded["w"] ** 2).sum()
        out.backward()
        assert np.allclose(loaded["w"].grad, 2 * loaded["w"].data)


def test_module_save_load_safetensors():
    """Verify Module.save_safetensors and Module.load_safetensors roundtrip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "linear.safetensors"

        model = Sequential([Linear(4, 8), Linear(8, 2)])
        x = Tensor(np.random.randn(2, 4))
        pred_before = model(x).data.copy()

        model.save_safetensors(filepath, metadata={"arch": "mlp"})
        assert filepath.exists()

        # Create fresh model with different weights
        fresh_model = Sequential([Linear(4, 8), Linear(8, 2)])
        fresh_pred = fresh_model(x).data
        assert not np.allclose(fresh_pred, pred_before)

        # Load weights
        fresh_model.load_safetensors(filepath)
        pred_after = fresh_model(x).data
        assert np.allclose(pred_after, pred_before, atol=1e-10)


def test_huggingface_official_safetensors_parity():
    """Cross-test with official Hugging Face safetensors library if installed."""
    try:
        import safetensors.numpy as hf_st
    except ImportError:
        pytest.skip("Official safetensors package not installed")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Save with miniGrad, load with official Hugging Face library
        minigrad_path = Path(tmpdir) / "from_minigrad.safetensors"
        data = {
            "f32": np.array([1.5, 2.5, 3.5], dtype=np.float32),
            "f64": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            "i32": np.array([1, 2, 3], dtype=np.int32),
        }
        metadata = {"creator": "minigrad_test"}

        save_file(data, minigrad_path, metadata=metadata)

        hf_loaded = hf_st.load_file(str(minigrad_path))
        for k in data:
            assert np.array_equal(hf_loaded[k], data[k]), f"HF failed to read key {k}"

        # 2. Save with official Hugging Face library, load with miniGrad
        hf_path = Path(tmpdir) / "from_hf.safetensors"
        hf_st.save_file(data, str(hf_path), metadata=metadata)

        minigrad_loaded = load_file(hf_path)
        for k in data:
            assert np.array_equal(minigrad_loaded[k], data[k]), f"miniGrad failed to read HF key {k}"

        with safe_open(hf_path) as f:
            assert f.metadata() == metadata
            assert np.array_equal(f.get_tensor("f32"), data["f32"])


def test_bfloat16_loading():
    """Verify bfloat16 bitcast conversion loads valid float32 values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "bf16.safetensors"

        # Construct a raw BF16 buffer
        # In IEEE 754:
        # float32 1.0 is 0x3F800000 -> BF16 is 0x3F80
        # float32 2.0 is 0x40000000 -> BF16 is 0x4000
        bf16_bytes = np.array([0x3F80, 0x4000], dtype=np.uint16).tobytes()

        import json
        import struct
        header = {
            "bf16_tensor": {
                "dtype": "BF16",
                "shape": [2],
                "data_offsets": [0, len(bf16_bytes)],
            }
        }
        header_bytes = json.dumps(header).encode("utf-8")
        with open(filepath, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(bf16_bytes)

        loaded = load_file(filepath)
        assert np.allclose(loaded["bf16_tensor"], [1.0, 2.0])
