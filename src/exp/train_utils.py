"""Training helpers: seed, accuracy, train/val one epoch."""

from __future__ import annotations

import random
import time
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.utils.data

from .noise import WeightNoiseContext

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def set_seed(seed: int) -> None:
    """Set Python, torch, and optionally numpy seed."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _HAS_NUMPY:
        np.random.seed(seed)


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy as percent (0-100)."""
    if logits.dim() == 2:
        pred = logits.argmax(dim=1)
    else:
        pred = logits.argmax(dim=-1)
    correct = (pred == targets).float().sum().item()
    return 100.0 * correct / targets.size(0)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lr_step_fn: Callable[[int], float],
    global_step: int,
    amp_dtype: str,
    scaler: Optional[Any],
    alpha_train: float,
    log_every: int,
    logger: Any,
) -> tuple[dict, int]:
    """Run one training epoch. Returns (metrics_dict, next_global_step)."""
    model.train()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    use_amp = amp_dtype in ("fp16", "bf16")
    use_autocast = device.type == "cuda" and amp_dtype in ("fp16", "bf16")
    autocast_dtype = torch.float16 if amp_dtype == "fp16" else (torch.bfloat16 if amp_dtype == "bf16" else None)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    steps = 0
    t0 = time.perf_counter()

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        lr = lr_step_fn(global_step)
        optimizer.zero_grad(set_to_none=True)

        if alpha_train > 0:
            with WeightNoiseContext(model, alpha_train):
                if use_autocast:
                    with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                        logits = model(x)
                        loss = criterion(logits, y)
                    if use_amp and scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                else:
                    logits = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
        else:
            if use_autocast:
                with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                    logits = model(x)
                    loss = criterion(logits, y)
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()

        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        total_loss += loss.detach().item()
        total_correct += (logits.detach().argmax(dim=1) == y).float().sum().item()
        total_samples += y.size(0)
        steps += 1
        global_step += 1

        if log_every > 0 and (batch_idx + 1) % log_every == 0:
            logger.info(
                "  step %d loss=%.4f acc=%.2f lr=%.6f",
                global_step - 1,
                loss.item(),
                accuracy_top1(logits.detach(), y),
                lr,
            )

    elapsed = time.perf_counter() - t0
    avg_loss = total_loss / steps if steps else 0.0
    acc = 100.0 * total_correct / total_samples if total_samples else 0.0
    final_lr = lr_step_fn(global_step - 1) if global_step > 0 else 0.0
    return {
        "loss": avg_loss,
        "acc_top1": acc,
        "lr": final_lr,
        "time_sec": elapsed,
        "steps": steps,
    }, global_step


def validate_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Run validation. No noise, no AMP. Returns metrics dict."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    steps = 0
    t0 = time.perf_counter()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == y).float().sum().item()
            total_samples += y.size(0)
            steps += 1

    elapsed = time.perf_counter() - t0
    avg_loss = total_loss / steps if steps else 0.0
    acc = 100.0 * total_correct / total_samples if total_samples else 0.0
    return {
        "loss": avg_loss,
        "acc_top1": acc,
        "time_sec": elapsed,
        "steps": steps,
    }
