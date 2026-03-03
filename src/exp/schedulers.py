"""Per-iteration LR schedule: warmup + cosine decay."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.optim import Optimizer


def build_lr_scheduler(
    optimizer: "Optimizer",
    epochs: int,
    warmup_epochs: int,
    steps_per_epoch: int,
) -> "callable":
    """Return a step(global_step) callable that sets optimizer.param_groups[0]['lr'].
    Warmup: linear 0 -> base_lr over warmup_epochs. Then cosine decay to ~0.
    """
    base_lr = optimizer.param_groups[0]["lr"]
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    decay_steps = total_steps - warmup_steps

    def step(global_step: int) -> float:
        if global_step < warmup_steps:
            lr = base_lr * (global_step / warmup_steps) if warmup_steps > 0 else base_lr
        else:
            progress = (global_step - warmup_steps) / decay_steps if decay_steps > 0 else 1.0
            progress = min(1.0, progress)
            lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        optimizer.param_groups[0]["lr"] = lr
        return lr

    return step
