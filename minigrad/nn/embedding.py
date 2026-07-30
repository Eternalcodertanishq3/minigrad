"""
embedding.py — Embedding layer for token/index-based lookups.

Maps integer indices to dense vector representations. This is the foundation
of NLP models — every word/token gets a learnable vector.

Reference: Standard embedding lookup used in Word2Vec, BERT, GPT, etc.
"""
from __future__ import annotations

import numpy as np

from minigrad.tensor import Tensor
from minigrad.nn.module import Module


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

    def forward(self, indices: Tensor) -> Tensor:
        """
        Look up embeddings for the given indices.

        Args:
            indices: Integer tensor of any shape containing indices in [0, num_embeddings)
        Returns:
            Tensor of shape (*indices.shape, embedding_dim)
        """
        # Use integer indices for lookup
        idx = indices.data.astype(np.intp)
        out_data = self.weight.data[idx]
        
        out = Tensor(
            out_data,
            requires_grad=self.weight.requires_grad,
            _children=(self.weight,),
            _op="embedding",
        )

        def _backward() -> None:
            if self.weight.requires_grad:
                # Scatter gradients back to the weight matrix
                np.add.at(self.weight.grad, idx, out.grad)

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"
