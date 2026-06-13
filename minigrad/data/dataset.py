"""
dataset.py — Dataset base class and MNIST loader.

Provides an abstract Dataset class and a concrete MNISTDataset implementation
that downloads and loads the MNIST dataset from Yann LeCun's website.
"""
from __future__ import annotations

import os
import gzip
import struct
import urllib.request
from typing import Tuple

import numpy as np


class Dataset:
    """
    Abstract base class for datasets.

    Subclasses must implement __len__ and __getitem__.
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Return (data, label) pair at index."""
        raise NotImplementedError


class MNISTDataset(Dataset):
    """
    MNIST handwritten digits dataset.

    Automatically downloads the dataset if not found locally.
    60,000 training images and 10,000 test images.
    Each image is 28x28 grayscale.

    Args:
        root:     Root directory to store/load data (default: './data')
        train:    If True, load training set; else load test set
        download: If True, download dataset if not found
    """

    BASE_URL = "https://yann.lecun.com/exdb/mnist/"
    FILES = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images":  "t10k-images-idx3-ubyte.gz",
        "test_labels":  "t10k-labels-idx1-ubyte.gz",
    }

    def __init__(self, root: str = "./data", train: bool = True, download: bool = True) -> None:
        self.root = root
        self.train = train

        if download:
            self._download()

        self._load_data()

    def _download(self) -> None:
        """Download MNIST files if they don't exist locally."""
        os.makedirs(self.root, exist_ok=True)

        prefix = "train" if self.train else "test"
        image_file = self.FILES[f"{prefix}_images"]
        label_file = self.FILES[f"{prefix}_labels"]

        for filename in [image_file, label_file]:
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                url = self.BASE_URL + filename
                try:
                    urllib.request.urlretrieve(url, filepath)
                    print(f"Saved to {filepath}")
                except Exception:
                    # Fallback mirror
                    fallback_url = f"https://ossci-datasets.s3.amazonaws.com/mnist/{filename}"
                    print("Retrying from mirror...")
                    urllib.request.urlretrieve(fallback_url, filepath)
                    print(f"Saved to {filepath}")

    def _load_data(self) -> None:
        """Load images and labels from gzip files."""
        prefix = "train" if self.train else "test"

        image_file = os.path.join(self.root, self.FILES[f"{prefix}_images"])
        label_file = os.path.join(self.root, self.FILES[f"{prefix}_labels"])

        # Load images
        with gzip.open(image_file, "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            self.images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

        # Load labels
        with gzip.open(label_file, "rb") as f:
            magic, num = struct.unpack(">II", f.read(8))
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)

        # Normalize to [0, 1]
        self.images = self.images.astype(np.float64) / 255.0  # type: ignore[assignment]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.images[idx], int(self.labels[idx])
