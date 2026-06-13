"""
03_mlp_xor.py — 2-layer MLP solving the XOR problem.

The XOR problem is the classic proof that a single-layer perceptron
cannot solve non-linearly separable problems. A 2-layer MLP can.

XOR truth table:
    (0, 0) -> 0
    (0, 1) -> 1
    (1, 0) -> 1
    (1, 1) -> 0

Architecture: 2 inputs -> 4 hidden (ReLU) -> 1 output (Sigmoid)
Loss: Binary Cross Entropy

Run: python examples/03_mlp_xor.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from minigrad import Tensor
from minigrad.nn import Module, Linear, ReLU, Sigmoid
from minigrad.nn.loss import BCELoss
from minigrad.optim import SGD


class XORModel(Module):
    """2-layer MLP for XOR: 2 -> 4 -> 1"""

    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2, 4)
        self.relu = ReLU()
        self.fc2 = Linear(4, 1)
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def train():
    print("=" * 60)
    print("XOR Problem — 2-Layer MLP")
    print("=" * 60)

    # XOR dataset
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.float64)

    print("\nXOR Truth Table:")
    for i in range(4):
        print(f"  {X[i]} -> {int(y[i])}")

    # Model
    model = XORModel()
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.5)

    print("\nModel architecture:")
    print("  Input:  2 neurons")
    print("  Hidden: 4 neurons (ReLU)")
    print("  Output: 1 neuron (Sigmoid)")
    print(f"  Total parameters: {sum(p.data.size for p in model.parameters())}")

    # Training
    print("\nTraining...")
    print("-" * 40)

    epochs = 500
    for epoch in range(epochs):
        # Forward
        x = Tensor(X)
        pred = model(x)
        loss = criterion(pred, y.reshape(-1, 1))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            preds = (pred.data > 0.5).astype(int).flatten()
            acc = np.mean(preds == y)
            print(f"Epoch {epoch:3d}: loss = {loss.data.item():.6f}, accuracy = {acc:.2%}")

    print("-" * 40)

    # Final evaluation
    print("\nFinal predictions:")
    x = Tensor(X)
    pred = model(x)
    for i in range(4):
        pred_label = 1 if pred.data[i, 0] > 0.5 else 0
        print(f"  {X[i]} -> predicted: {pred_label} "
              f"(confidence: {pred.data[i, 0]:.4f})")

    preds = (pred.data > 0.5).astype(int).flatten()
    acc = np.mean(preds == y)
    print(f"\nFinal accuracy: {acc:.2%}")

    if acc == 1.0:
        print("PASS: XOR problem solved!")
    else:
        print("FAIL: XOR problem not fully solved - try more epochs or different lr")


if __name__ == "__main__":
    train()
