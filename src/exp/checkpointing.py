"""Save/load training checkpoints."""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    config: dict,
    scaler_state: Optional[dict] = None,
) -> None:
    """Write checkpoint (model, optimizer, epoch, best_metric, config, optional scaler)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
    }
    if scaler_state is not None:
        state["scaler"] = scaler_state
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
) -> dict:
    """Load checkpoint; optionally load optimizer and scaler. Return state dict."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return state
