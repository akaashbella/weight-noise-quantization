"""Smoke test: build CIFAR-100 dataloaders, one batch from each, check shapes and dtypes."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import torch
from src.data import build_dataloaders


def main() -> int:
    loaders = build_dataloaders(
        dataset="cifar100",
        data_root="data",
        img_size=224,
        batch_size=8,
        num_workers=2,
        seed=0,
        val_fraction=0.1,
    )
    for split in ("train", "val", "test"):
        x, y = next(iter(loaders[split]))
        print(f"  {split}: x {x.shape}, y {y.shape}")
        assert x.dim() == 4 and x.size(1) == 3 and x.size(2) == 224 and x.size(3) == 224, (
            f"Expected x [B, 3, 224, 224], got {x.shape}"
        )
        assert y.dim() == 1 and y.size(0) == x.size(0), (
            f"Expected y [B], got {y.shape}"
        )
        assert x.dtype == torch.float32, f"Expected float32, got {x.dtype}"
        assert y.min() >= 0 and y.max() < 100, (
            f"Labels must be in [0, 99], got min={y.min().item()}, max={y.max().item()}"
        )
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
