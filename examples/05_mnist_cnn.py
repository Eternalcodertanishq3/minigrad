"""
05_mnist_cnn.py — MNIST classification with a Convolutional Neural Network.

Architecture:
    Conv2D(1, 32, 3x3) + ReLU
    Conv2D(32, 64, 3x3, stride=2) + ReLU
    Conv2D(64, 64, 3x3) + ReLU
    Flatten -> Linear(7744, 128) + ReLU -> Linear(128, 10)

Expected accuracy: ~99% after 10 epochs

Run: python examples/05_mnist_cnn.py
Note: This takes longer than the MLP due to convolution operations.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from minigrad import Tensor
from minigrad.nn import Sequential, Conv2D, Linear, ReLU
from minigrad.nn.loss import CrossEntropyLoss
from minigrad.optim import Adam
from minigrad.data import MNISTDataset, DataLoader
from minigrad.utils import count_parameters


def train():
    print("=" * 60)
    print("MNIST CNN — Convolutional Neural Network")
    print("=" * 60)

    # Hyperparameters
    batch_size = 64
    epochs = 10
    lr = 1e-3
    weight_decay = 1e-4

    print(f"\nHyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs:     {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay:  {weight_decay}")

    # Data
    print("\nLoading MNIST dataset...")
    train_dataset = MNISTDataset(train=True, download=True)
    test_dataset = MNISTDataset(train=False, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Test samples:     {len(test_dataset):,}")

    # CNN Model
    model = Sequential([
        # Layer 1: (N, 1, 28, 28) -> (N, 32, 28, 28)
        Conv2D(1, 32, kernel_size=3, padding=1),
        ReLU(),

        # Layer 2: (N, 32, 28, 28) -> (N, 64, 13, 13)
        Conv2D(32, 64, kernel_size=3, stride=2),
        ReLU(),

        # Layer 3: (N, 64, 13, 13) -> (N, 64, 11, 11)
        Conv2D(64, 64, kernel_size=3),
        ReLU(),

        # Flatten: (N, 64, 11, 11) -> (N, 7744)
        # We'll do reshape in forward

        # Classifier
        Linear(7744, 128),
        ReLU(),
        Linear(128, 10),
    ])

    print(f"\nModel: CNN")
    print(f"  Conv2D(1, 32, 3x3) + ReLU")
    print(f"  Conv2D(32, 64, 3x3, s=2) + ReLU")
    print(f"  Conv2D(64, 64, 3x3) + ReLU")
    print(f"  Flatten -> Linear(7744, 128) -> Linear(128, 10)")
    print(f"Total parameters: {count_parameters(model):,}")

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training
    print("\nTraining...")
    print("-" * 50)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            # Reshape to (N, 1, 28, 28) for Conv2D
            x = Tensor(batch_x.reshape(-1, 1, 28, 28))
            y = batch_y

            # Forward
            logits = model(x)
            loss = criterion(logits, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        for batch_x, batch_y in test_loader:
            x = Tensor(batch_x.reshape(-1, 1, 28, 28))
            logits = model(x)
            preds = np.argmax(logits.data, axis=1)
            correct += np.sum(preds == batch_y)
            total += len(batch_y)

        acc = correct / total * 100
        print(f"Epoch {epoch+1}/{epochs}: loss = {avg_loss:.4f}, test accuracy = {acc:.2f}%")

    print("-" * 50)
    print(f"\nFinal test accuracy: {acc:.2f}%")
    print(f"Target: >= 99%")


if __name__ == "__main__":
    train()
