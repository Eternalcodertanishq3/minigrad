from __future__ import annotations
from minigrad.tensor import Tensor
from minigrad.nn.module import Module

class Flatten(Module):
    """
    Flattens the input tensor while preserving the batch dimension.
    Equivalent to x.reshape(N, -1).
    """
    def forward(self, x: Tensor) -> Tensor:
        N = x.data.shape[0]
        return x.reshape(N, -1)

    def __repr__(self) -> str:
        return "Flatten()"
