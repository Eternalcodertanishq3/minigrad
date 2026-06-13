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
from minigrad.nn.activations import ReLU, Sigmoid, Tanh, GELU, Softmax, LeakyReLU, ELU
from minigrad.nn.batchnorm import BatchNorm1D, BatchNorm2D
from minigrad.nn.dropout import Dropout, Dropout2D
from minigrad.nn.flatten import Flatten
from minigrad.nn.loss import MSELoss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss, NLLLoss
from minigrad.nn.sequential import Sequential

__all__ = [
    "Module",
    "Linear",
    "Conv2D",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "GELU",
    "Softmax",
    "LeakyReLU",
    "ELU",
    "BatchNorm1D",
    "BatchNorm2D",
    "Dropout",
    "Dropout2D",
    "Flatten",
    "MSELoss",
    "CrossEntropyLoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "NLLLoss",
    "Sequential",
]
