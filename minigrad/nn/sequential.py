"""
sequential.py — Sequential container for stacking layers.

Allows composing layers as a pipeline: output = layer_n(...(layer_2(layer_1(input))))
"""
from __future__ import annotations

from typing import List, Iterator

from minigrad.tensor import Tensor
from minigrad.nn.module import Module


class Sequential(Module):
    """
    A sequential container.

    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    Example:
        model = Sequential([
            Linear(784, 128),
            ReLU(),
            Linear(128, 10),
        ])
        out = model(x)  # x -> Linear -> ReLU -> Linear -> out
    """

    def __init__(self, layers: List[Module]) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def __iter__(self) -> Iterator[Module]:
        return iter(self.layers)

    def __getitem__(self, idx: int) -> Module:
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def __repr__(self) -> str:
        layers_str = "\n".join(f"  ({i}): {layer}" for i, layer in enumerate(self.layers))
        return f"Sequential(\n{layers_str}\n)"
