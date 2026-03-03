"""Shared evaluation loop for PTQ and eval_noise."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.utils.data


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_dtype: Optional[str] = None,
) -> dict[str, float]:
    """Run model over dataloader; return {"loss": float, "acc": float}. acc in percent. model.eval(), no_grad."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    use_autocast = device.type == "cuda" and amp_dtype in ("fp16", "bf16")
    dtype = torch.bfloat16 if amp_dtype == "bf16" else (torch.float16 if amp_dtype == "fp16" else None)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    steps = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if use_autocast:
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == y).float().sum().item()
            total_samples += y.size(0)
            steps += 1
    mean_loss = total_loss / steps if steps else 0.0
    acc = 100.0 * total_correct / total_samples if total_samples else 0.0
    return {"loss": mean_loss, "acc": acc}
