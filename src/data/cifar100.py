"""CIFAR-100 dataset wrapper and train/val split."""

from __future__ import annotations

from typing import Any

import torch
import torch.utils.data
import torchvision.datasets


def get_cifar100_datasets(
    data_root: str,
    train_transform: Any,
    test_transform: Any,
) -> dict[str, torch.utils.data.Dataset]:
    """Return CIFAR-100 train and test datasets with given transforms."""
    return {
        "train": torchvision.datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=train_transform,
        ),
        "test": torchvision.datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=test_transform,
        ),
    }


def split_train_val(
    train_dataset: torch.utils.data.Dataset,
    val_fraction: float,
    seed: int,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """Split train dataset into train/val using random_split with seeded generator."""
    n = len(train_dataset)
    n_val = int(n * val_fraction)
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(train_dataset, [n_train, n_val], generator=gen)
