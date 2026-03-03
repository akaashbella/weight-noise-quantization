"""Model utilities: forward check and parameter counts."""

from __future__ import annotations

import torch
import torch.nn as nn


def assert_forward_works(
    model: nn.Module,
    device: str = "cpu",
    batch_size: int = 2,
    img_size: int = 224,
) -> None:
    """Run dummy forward and check output shape [B, num_classes]. Raises on failure."""
    model = model.to(device)
    model.eval()
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device)
    with torch.no_grad():
        logits = model(dummy)
    if not isinstance(logits, torch.Tensor):
        raise AssertionError(
            f"Expected model to return a Tensor, got {type(logits)}"
        )
    if logits.dim() != 2:
        raise AssertionError(
            f"Expected logits shape [B, num_classes], got {logits.shape}"
        )
    if logits.size(0) != batch_size:
        raise AssertionError(
            f"Expected batch size {batch_size}, got {logits.size(0)}"
        )


def count_params(model: nn.Module) -> int:
    """Return total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
