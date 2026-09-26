"""
minigrad.optim — Optimizers for gradient-based parameter updates.

Each optimizer implements a different strategy for updating parameters
given their gradients. All follow the same interface: zero_grad() then step().
"""
from minigrad.optim.adam import Adam, AdamW
from minigrad.optim.base import Optimizer
from minigrad.optim.rmsprop import RMSprop
from minigrad.optim.schedulers import CosineAnnealingLR, ExponentialLR, StepLR
from minigrad.optim.sgd import SGD

__all__ = ["Optimizer", "SGD", "RMSprop", "Adam", "AdamW", "StepLR", "CosineAnnealingLR", "ExponentialLR"]
