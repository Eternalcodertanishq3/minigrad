"""
dataloader.py — Batched data loading with shuffling.

Provides efficient iteration over datasets in mini-batches.
Supports shuffling, custom batch size, and drop_last.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Iterator

from minigrad.data.dataset import Dataset


class DataLoader:
    """
    DataLoader for batching and shuffling datasets.

    Args:
        dataset:    Dataset to load data from
        batch_size: Number of samples per batch
        shuffle:    If True, shuffle data at the start of each epoch
        drop_last:  If True, drop the last incomplete batch
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over the dataset in batches."""
        indices = np.arange(len(self.dataset))

        if self.shuffle:
            np.random.shuffle(indices)

        num_batches = len(indices) // self.batch_size
        if not self.drop_last and len(indices) % self.batch_size != 0:
            num_batches += 1

        for i in range(num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, len(indices))
            batch_indices = indices[start:end]

            # Collect batch data
            batch_data = []
            batch_labels = []
            for idx in batch_indices:
                data, label = self.dataset[int(idx)]
                batch_data.append(data)
                batch_labels.append(label)

            yield np.stack(batch_data), np.array(batch_labels, dtype=np.int64)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
