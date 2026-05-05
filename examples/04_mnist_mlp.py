"""
04_mnist_mlp.py — MNIST classification with a multi-layer perceptron.

Architecture:
    Input:  784 (28x28 flattened)
    Hidden: 256 (ReLU) -> 128 (ReLU)
    Output: 10 (softmax via CrossEntropyLoss)

Expected accuracy: ~97% after 5 epochs

Run: python examples/04_mnist_mlp.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from minigrad import Tensor
from minigrad.nn import Sequential, Linear, ReLU
from minigrad.nn.loss import CrossEntropyLoss
from minigrad.optim import Adam
from minigrad.data import MNISTDataset, DataLoader
from minigrad.utils import count_parameters, accuracy


def train():
    print("=" * 60)
    print("MNIST MLP — Multi-Layer Perceptron")
    print("=" * 60)

    # Hyperparameters
    batch_size = 128
    epochs = 5
    lr = 1e-3

    print(f"\nHyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs:     {epochs}")
    print(f"  Learning rate: {lr}")

    # Data
    print("\nLoading MNIST dataset...")
    train_dataset = MNISTDataset(train=True, download=True)
    test_dataset = MNISTDataset(train=False, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Test samples:     {len(test_dataset):,}")

    # Model: 784 -> 256 -> 128 -> 10
    model = Sequential([
        Linear(784, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10),
    ])
    print(f"\nModel: {model}")
    print(f"Total parameters: {count_parameters(model):,}")

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # Training
    print("\nTraining...")
    print("-" * 50)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            # Flatten images: (N, 28, 28) -> (N, 784)
            x = Tensor(batch_x.reshape(batch_x.shape[0], -1))
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
            x = Tensor(batch_x.reshape(batch_x.shape[0], -1))
            logits = model(x)
            preds = np.argmax(logits.data, axis=1)
            correct += np.sum(preds == batch_y)
            total += len(batch_y)

        acc = correct / total * 100
        print(f"Epoch {epoch+1}/{epochs}: loss = {avg_loss:.4f}, test accuracy = {acc:.2f}%")

    print("-" * 50)
    print(f"\nFinal test accuracy: {acc:.2f}%")
    print(f"Target: ~97%")


if __name__ == "__main__":
    train()
