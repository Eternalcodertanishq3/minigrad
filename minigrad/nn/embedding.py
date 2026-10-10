"""
embedding.py — Embedding layer for token/index-based lookups.

Maps integer indices to dense vector representations. This is the foundation
of NLP models — every word/token gets a learnable vector.

Reference: Standard embedding lookup used in Word2Vec, BERT, GPT, etc.
"""
from __future__ import annotations

import weakref
from typing import Sequence, Union

import numpy as np

from minigrad.nn.module import Module
from minigrad.tensor import Tensor


class Embedding(Module):
    """
    A simple lookup table that stores embeddings of a fixed dictionary and size.

    Args:
        num_embeddings: Size of the dictionary (vocabulary size)
        embedding_dim:  Size of each embedding vector
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Initialize with small random values
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim) * 0.01,
            requires_grad=True,
        )

    def forward(self, indices: Union[Tensor, np.ndarray, Sequence[int]]) -> Tensor:
        """
        Look up embeddings for the given indices.

        Args:
            indices: Integer Tensor, NumPy array, or (nested) list of indices in [0, num_embeddings)
        Returns:
            Tensor of shape (*indices.shape, embedding_dim)
        """
        # Use integer indices for lookup (accept Tensor / ndarray / list)
        raw = indices.data if isinstance(indices, Tensor) else np.asarray(indices)
        idx = np.asarray(raw).astype(np.intp)
        out_data = self.weight.data[idx]

        out = Tensor(
            out_data,
            requires_grad=self.weight.requires_grad,
            _children=(self.weight,),
            _op="embedding",
            _ctx=idx,
        )

        if out.requires_grad:
            out_ref = weakref.ref(out)

            def _backward() -> None:
                o = out_ref()
                if o is None:
                    return
                if self.weight.requires_grad:
                    # Scatter gradients back to the weight matrix
                    np.add.at(self.weight.grad, idx, o.grad)

            out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"
