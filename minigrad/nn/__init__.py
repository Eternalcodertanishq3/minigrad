"""
minigrad.nn — Neural network layers and utilities.

Provides Module base class and common layers: Linear, Conv2D, activations,
BatchNorm, Dropout, loss functions, and containers.

Usage:
    from minigrad.nn import Sequential, Linear, ReLU, CrossEntropyLoss
    model = Sequential([Linear(784, 128), ReLU(), Linear(128, 10)])
"""
from minigrad.nn.module import Module
from minigrad.nn.linear import Linear
from minigrad.nn.conv import Conv2D
from minigrad.nn.activations import ReLU, Sigmoid, Tanh, GELU
from minigrad.nn.batchnorm import BatchNorm1D, BatchNorm2D
from minigrad.nn.dropout import Dropout
from minigrad.nn.loss import MSELoss, CrossEntropyLoss, BCELoss
from minigrad.nn.sequential import Sequential

__all__ = [
    "Module",
    "Linear",
    "Conv2D",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "GELU",
    "BatchNorm1D",
    "BatchNorm2D",
    "Dropout",
    "MSELoss",
    "CrossEntropyLoss",
    "BCELoss",
    "Sequential",
]
