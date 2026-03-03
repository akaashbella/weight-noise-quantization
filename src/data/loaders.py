"""Build train/val/test dataloaders from config-driven dataset."""

from __future__ import annotations

import functools
import torch
import torch.utils.data

from .cifar100 import get_cifar100_datasets, split_train_val
from .transforms import build_test_transform, build_train_transform


def _worker_init_fn(worker_id: int, seed: int = 0) -> None:
    """Deterministic worker init: set base seed + worker_id. Must be module-level for pickle."""
    torch.manual_seed(seed + worker_id)


def build_dataloaders(
    dataset: str,
    data_root: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_fraction: float = 0.1,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = True,
) -> dict[str, torch.utils.data.DataLoader]:
    """Build train, val, test DataLoaders. dataset must be 'cifar100'."""
    if dataset != "cifar100":
        raise ValueError(f"Unknown dataset: {dataset!r}. Supported: cifar100")

    train_tf = build_train_transform(img_size)
    test_tf = build_test_transform(img_size)
    datasets = get_cifar100_datasets(data_root, train_tf, test_tf)
    train_full = datasets["train"]
    test_ds = datasets["test"]

    train_subset, val_subset = split_train_val(train_full, val_fraction, seed)

    gen = torch.Generator()
    gen.manual_seed(seed)

    persistent = persistent_workers and num_workers > 0
    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    if num_workers > 0:
        loader_kw["prefetch_factor"] = prefetch_factor
        loader_kw["worker_init_fn"] = functools.partial(_worker_init_fn, seed=seed)

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        generator=gen,
        **loader_kw,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kw,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kw,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
