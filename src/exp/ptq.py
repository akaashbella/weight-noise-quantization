"""Weights-only PTQ: symmetric int8, per-output-channel, maxabs scale."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn


def is_quantizable_module(module: nn.Module) -> bool:
    """True for nn.Conv2d and nn.Linear."""
    return isinstance(module, (nn.Conv2d, nn.Linear))


def quantize_weight_per_out_channel_symmetric_int8(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight to int8 per output channel (symmetric, maxabs/127). Returns (dequant_float, scale_per_out)."""
    # w: Conv2d [out, in, kh, kw], Linear [out, in] -> reshape to [out, -1]
    out = w.shape[0]
    w_flat = w.reshape(out, -1)
    m = w_flat.abs().amax(dim=1)
    s = m / 127.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    w_scaled = w / s.reshape(out, *([1] * (w.dim() - 1)))
    q = torch.clamp(torch.round(w_scaled), -127.0, 127.0)
    wq = q * s.reshape(out, *([1] * (w.dim() - 1)))
    return wq.to(w.dtype), s.detach().to(w.dtype)


def apply_weights_only_ptq_inplace(model: nn.Module) -> dict[str, Any]:
    """Replace Conv2d/Linear weights with dequantized float (W8 stored as float). Return layer_stats."""
    layer_stats: dict[str, Any] = {}
    for name, module in model.named_modules():
        if not is_quantizable_module(module):
            continue
        w = module.weight.data
        w_orig = w.clone()
        wq, scale_per_out = quantize_weight_per_out_channel_symmetric_int8(w)
        err = (w_orig - wq).abs().mean().item()
        w_norm = w_orig.norm(2).item()
        rel_l2 = (w_orig - wq).norm(2).item() / (w_norm + 1e-12)
        module.weight.data = wq.to(w.device)
        layer_stats[name] = {
            "type": "conv2d" if isinstance(module, nn.Conv2d) else "linear",
            "scale_shape": list(scale_per_out.shape),
            "mean_abs_err": err,
            "rel_l2_err": rel_l2,
        }
    return layer_stats


def clone_model_for_ptq(model: nn.Module, device: torch.device) -> nn.Module:
    """Deep copy model (on CPU) then move to device. State dict identical."""
    cloned = copy.deepcopy(model)
    cloned = cloned.to(device)
    return cloned
