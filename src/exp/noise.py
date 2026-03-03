"""Train-time Gaussian weight noise for Conv/Linear only (exclude bias, norm)."""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Generator

import torch
import torch.nn as nn


def is_norm_param(param_name: str) -> bool:
    """True if param is from BN/LN or similar (by name)."""
    name_lower = param_name.lower()
    if ".bn" in name_lower or "batchnorm" in name_lower:
        return True
    if "downsample.1" in param_name:  # BN in downsample
        return True
    if "norm" in name_lower or ".ln" in name_lower or "layernorm" in name_lower:
        return True
    # Standalone "bn" as in .bn1, .bn2
    if re.search(r"\.bn\d*\b", name_lower):
        return True
    return False


def eligible_param(name: str, param: nn.Parameter) -> bool:
    """True if param should receive noise: requires_grad, ndim>=2, not norm."""
    if not param.requires_grad:
        return False
    if param.ndim < 2:
        return False
    if is_norm_param(name):
        return False
    return True


def add_gaussian_noise_inplace(
    params: list[tuple[str, nn.Parameter]],
    alpha: float,
) -> list[torch.Tensor]:
    """Add N(0, alpha*std) noise to each param in place. Return list of eps for revert."""
    eps_list: list[torch.Tensor] = []
    with torch.no_grad():
        for _name, p in params:
            sigma = p.detach().float().std().item()
            eps = torch.randn_like(p, device=p.device, dtype=p.dtype) * (alpha * sigma)
            p.add_(eps)
            eps_list.append(eps)
    return eps_list


def remove_noise_inplace(
    params: list[tuple[str, nn.Parameter]],
    eps_list: list[torch.Tensor],
) -> None:
    """Subtract previously added noise from params."""
    with torch.no_grad():
        for (_name, p), eps in zip(params, eps_list):
            p.sub_(eps)


class WeightNoiseContext:
    """Context manager: add noise on enter, remove on exit. For use around forward+backward."""

    def __init__(self, model: nn.Module, alpha: float) -> None:
        self.model = model
        self.alpha = alpha
        self._params: list[tuple[str, nn.Parameter]] = []
        self._eps_list: list[torch.Tensor] = []

    def __enter__(self) -> WeightNoiseContext:
        self._params = [
            (name, p)
            for name, p in self.model.named_parameters()
            if eligible_param(name, p)
        ]
        if self._params and self.alpha > 0:
            self._eps_list = add_gaussian_noise_inplace(self._params, self.alpha)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._params and self._eps_list:
            remove_noise_inplace(self._params, self._eps_list)
        return None
