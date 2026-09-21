"""
safetensors.py — Pure-Python reader and writer for the Hugging Face SafeTensors format.

SafeTensors is a fast, secure, and memory-mappable binary format for storing
and loading tensors without arbitrary code execution vulnerabilities (unlike pickle).

Format specification:
- First 8 bytes: unsigned 64-bit little-endian integer (<Q) specifying the byte length N of the JSON header.
- Next N bytes: UTF-8 encoded JSON string containing tensor metadata (dtype, shape, data_offsets)
  and optional __metadata__ dictionary. Padded with spaces for 8-byte data buffer alignment.
- Remaining bytes: Contiguous binary buffer of raw tensor bytes.

References:
- https://github.com/huggingface/safetensors
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from minigrad.tensor import Tensor


# ── Dtype Mappings ──────────────────────────────────────────────────

DTYPE_TO_STR: Dict[np.dtype, str] = {
    np.dtype("float64"): "F64",
    np.dtype("float32"): "F32",
    np.dtype("float16"): "F16",
    np.dtype("int64"): "I64",
    np.dtype("int32"): "I32",
    np.dtype("int16"): "I16",
    np.dtype("int8"): "I8",
    np.dtype("uint8"): "U8",
    np.dtype("bool"): "BOOL",
}

STR_TO_DTYPE: Dict[str, np.dtype] = {
    "F64": np.dtype("float64"),
    "F32": np.dtype("float32"),
    "F16": np.dtype("float16"),
    "I64": np.dtype("int64"),
    "I32": np.dtype("int32"),
    "I16": np.dtype("int16"),
    "I8": np.dtype("int8"),
    "U8": np.dtype("uint8"),
    "BOOL": np.dtype("bool"),
}


def _bf16_to_f32(raw_bytes: bytes) -> np.ndarray:
    """Convert bfloat16 byte buffer into float32 array using bit manipulation."""
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


# ── Serialization Functions ─────────────────────────────────────────

def save_file(
    tensors: Dict[str, Union[Tensor, np.ndarray]],
    filename: Union[str, Path],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Save a dictionary of tensors or NumPy arrays to a .safetensors file.

    Args:
        tensors:  Dictionary mapping parameter names to Tensors or np.ndarrays.
        filename: Destination file path.
        metadata: Optional dictionary of string key-value pairs for metadata.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    header: Dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    data_chunks: List[bytes] = []
    current_offset = 0

    for name, tensor in tensors.items():
        if isinstance(tensor, Tensor):
            arr = np.ascontiguousarray(tensor.data)
        elif isinstance(tensor, np.ndarray):
            arr = np.ascontiguousarray(tensor)
        else:
            raise TypeError(
                f"Expected Tensor or np.ndarray for tensor '{name}', got {type(tensor).__name__}"
            )

        dtype_str = DTYPE_TO_STR.get(arr.dtype)
        if dtype_str is None:
            # Fallback for types like int or float scalars
            arr = np.ascontiguousarray(arr, dtype=np.float64)
            dtype_str = "F64"

        raw_bytes = arr.tobytes()
        chunk_len = len(raw_bytes)

        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [current_offset, current_offset + chunk_len],
        }
        current_offset += chunk_len
        data_chunks.append(raw_bytes)

    # Serialize header to UTF-8 JSON
    header_json = json.dumps(header, separators=(",", ":"))
    header_bytes = header_json.encode("utf-8")

    # Pad header with trailing spaces so the data payload starts aligned to an 8-byte boundary
    # Format: 8 bytes (header_len) + header_bytes + data_payload
    pad_len = (8 - ((8 + len(header_bytes)) % 8)) % 8
    if pad_len > 0:
        header_bytes += b" " * pad_len

    header_len = len(header_bytes)
    header_len_bytes = struct.pack("<Q", header_len)

    with open(path, "wb") as f:
        f.write(header_len_bytes)
        f.write(header_bytes)
        for chunk in data_chunks:
            f.write(chunk)


def load_file(
    filename: Union[str, Path],
    to_tensor: bool = False,
    dtype: Optional[Any] = None,
) -> Dict[str, Union[np.ndarray, Tensor]]:
    """
    Load tensors from a .safetensors file.

    Args:
        filename:  Path to the .safetensors file.
        to_tensor: If True, wrap loaded arrays into miniGrad Tensor objects.
        dtype:     Optional target NumPy dtype to cast floating-point tensors to
                   (e.g., np.float64 for miniGrad autograd compatibility).

    Returns:
        Dictionary mapping tensor names to NumPy arrays (or miniGrad Tensors).
    """
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"SafeTensors file not found: {path}")

    with open(path, "rb") as f:
        # Read 8-byte header length (<Q)
        header_size_bytes = f.read(8)
        if len(header_size_bytes) < 8:
            raise ValueError(f"Invalid safetensors file: too short ({len(header_size_bytes)} bytes)")

        header_len = struct.unpack("<Q", header_size_bytes)[0]
        header_bytes = f.read(header_len)
        if len(header_bytes) < header_len:
            raise ValueError("Truncated header in safetensors file")

        header = json.loads(header_bytes.decode("utf-8"))
        data_payload = f.read()

    result: Dict[str, Union[np.ndarray, Tensor]] = {}

    target_dtype = np.dtype(dtype) if dtype is not None else None

    for name, info in header.items():
        if name == "__metadata__":
            continue

        start, end = info["data_offsets"]
        dtype_str = info["dtype"]
        shape = tuple(info["shape"])

        chunk = data_payload[start:end]

        if dtype_str == "BF16":
            arr = _bf16_to_f32(chunk).reshape(shape)
        elif dtype_str in STR_TO_DTYPE:
            arr_dtype = STR_TO_DTYPE[dtype_str]
            arr = np.frombuffer(chunk, dtype=arr_dtype).reshape(shape).copy()
        else:
            raise ValueError(f"Unsupported dtype '{dtype_str}' in safetensors file for tensor '{name}'")

        if target_dtype is not None and np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(target_dtype)

        if to_tensor:
            result[name] = Tensor(arr, requires_grad=True)
        else:
            result[name] = arr

    return result


class safe_open:
    """
    Context manager to inspect and selectively read from a SafeTensors file.

    Usage:
        with safe_open("model.safetensors") as f:
            print("Keys:", f.keys())
            print("Metadata:", f.metadata())
            tensor = f.get_tensor("layer.weight")
    """

    def __init__(self, filename: Union[str, Path], framework: str = "numpy") -> None:
        self.path = Path(filename)
        self.framework = framework
        self._file = None
        self._header: Dict[str, Any] = {}
        self._data_offset = 0

    def __enter__(self) -> safe_open:
        self._file = open(self.path, "rb")
        header_size_bytes = self._file.read(8)
        if len(header_size_bytes) < 8:
            raise ValueError("File is too small to be a valid safetensors file")

        header_len = struct.unpack("<Q", header_size_bytes)[0]
        header_bytes = self._file.read(header_len)
        self._header = json.loads(header_bytes.decode("utf-8"))
        self._data_offset = 8 + header_len
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file is not None:
            self._file.close()

    def keys(self) -> List[str]:
        """Return list of tensor names stored in the file."""
        return [k for k in self._header.keys() if k != "__metadata__"]

    def metadata(self) -> Dict[str, str]:
        """Return user metadata dictionary if present, else empty dict."""
        return self._header.get("__metadata__", {})

    def get_tensor(self, name: str) -> np.ndarray:
        """Read and return a single tensor by name without loading all others."""
        if name not in self._header or name == "__metadata__":
            raise KeyError(f"Tensor '{name}' not found in safetensors file")

        info = self._header[name]
        start, end = info["data_offsets"]
        dtype_str = info["dtype"]
        shape = tuple(info["shape"])

        self._file.seek(self._data_offset + start)
        raw_bytes = self._file.read(end - start)

        if dtype_str == "BF16":
            return _bf16_to_f32(raw_bytes).reshape(shape)
        elif dtype_str in STR_TO_DTYPE:
            arr_dtype = STR_TO_DTYPE[dtype_str]
            return np.frombuffer(raw_bytes, dtype=arr_dtype).reshape(shape).copy()
        else:
            raise ValueError(f"Unsupported dtype '{dtype_str}' for tensor '{name}'")
