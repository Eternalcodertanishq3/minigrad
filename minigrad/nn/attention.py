"""
attention.py — Multi-Head Attention and Transformer Block.

Implements the core building blocks of the Transformer architecture:
- Scaled dot-product attention with optional causal masking
- Multi-head attention with learned Q/K/V projections
- Pre-norm Transformer block (GPT-2 style)

Reference: "Attention Is All You Need" (Vaswani et al., 2017)
"""
from __future__ import annotations

import math
import numpy as np

from minigrad.tensor import Tensor
from minigrad.nn.module import Module
from minigrad.nn.linear import Linear
from minigrad.nn.layernorm import LayerNorm
from minigrad.nn.dropout import Dropout
from minigrad.ops import softmax, einsum


class MultiHeadAttention(Module):
    """
    Multi-Head Attention mechanism.

    Splits the input into multiple attention heads, computes scaled
    dot-product attention independently for each head, then concatenates
    and projects the result.

    Args:
        embed_dim:  Total dimension of the model (must be divisible by num_heads)
        num_heads:  Number of parallel attention heads
        dropout:    Dropout probability on attention weights (default: 0.1)
        bias:       Whether to use bias in projection layers (default: True)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V projections
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)

        # Output projection
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        # Attention dropout
        self.attn_dropout = Dropout(dropout)

    def forward(self, x: Tensor, causal: bool = False) -> Tensor:
        """
        Compute multi-head self-attention.

        Args:
            x:      Input tensor of shape (B, T, C)
            causal: If True, apply causal mask (for autoregressive models)

        Returns:
            Output tensor of shape (B, T, C)
        """
        B, T, C = x.data.shape

        # Project to Q, K, V — each is (B, T, C)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into heads: (B, T, C) -> (B, T, H, D) -> (B, H, T, D)
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        # scores: (B, H, T, T)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = einsum("bhqd,bhkd->bhqk", q, k)
        scores = scores * scale

        # Causal mask: prevent attending to future positions
        if causal:
            mask = np.triu(np.ones((T, T)), k=1).astype(np.float64)
            mask = mask * (-1e9)
            scores = scores + Tensor(mask)

        # Attention weights
        attn = softmax(scores, axis=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum of values: (B, H, T, D)
        out = einsum("bhqk,bhkd->bhqd", attn, v)

        # Concatenate heads: (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)

        # Output projection
        out = self.out_proj(out)

        return out

    def __repr__(self) -> str:
        return (
            f"MultiHeadAttention(embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, head_dim={self.head_dim})"
        )


class TransformerBlock(Module):
    """
    A single Transformer block with pre-norm architecture (GPT-2 style).

    Structure:
        x = x + MultiHeadAttention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))

    Args:
        embed_dim:  Dimension of the model
        num_heads:  Number of attention heads
        ff_dim:     Hidden dimension of the feed-forward network
                    (default: 4 * embed_dim)
        dropout:    Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if ff_dim is None:
            ff_dim = 4 * embed_dim

        # Attention sublayer
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)

        # Feed-forward sublayer
        self.ln2 = LayerNorm(embed_dim)
        self.ff_in = Linear(embed_dim, ff_dim)
        self.ff_out = Linear(ff_dim, embed_dim)
        self.ff_dropout = Dropout(dropout)

    def forward(self, x: Tensor, causal: bool = False) -> Tensor:
        """
        Forward pass.

        Args:
            x:      Input tensor of shape (B, T, C)
            causal: If True, apply causal masking in attention

        Returns:
            Output tensor of shape (B, T, C)
        """
        # Pre-norm attention with residual
        x = x + self.attn(self.ln1(x), causal=causal)

        # Pre-norm feed-forward with residual
        h = self.ff_in(self.ln2(x))
        h = h.gelu()
        h = self.ff_out(h)
        h = self.ff_dropout(h)

        x = x + h

        return x

    def __repr__(self) -> str:
        return (
            f"TransformerBlock(embed_dim={self.ln1.normalized_shape[0]}, "
            f"num_heads={self.attn.num_heads})"
        )
