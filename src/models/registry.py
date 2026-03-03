"""Central registry: canonical name -> builder + metadata."""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

from . import builders

# Canonical names (lowercase snake_case) -> (builder_callable, metadata_dict)
_REGISTRY: dict[str, tuple[Callable[[int], nn.Module], dict]] = {
    "resnet50": (
        builders.build_resnet50,
        {
            "name": "resnet50",
            "family": "resnet",
            "torchvision_id": "resnet50",
            "notes": "ResNet-50, fc head",
            "supports_weights_only_ptq": True,
        },
    ),
    "mobilenetv3_large": (
        builders.build_mobilenetv3_large,
        {
            "name": "mobilenetv3_large",
            "family": "mobilenet",
            "torchvision_id": "mobilenet_v3_large",
            "notes": "MobileNetV3 Large, classifier[-1]",
            "supports_weights_only_ptq": True,
        },
    ),
    "convnext_tiny": (
        builders.build_convnext_tiny,
        {
            "name": "convnext_tiny",
            "family": "convnext",
            "torchvision_id": "convnext_tiny",
            "notes": "ConvNeXt Tiny, classifier[-1]",
            "supports_weights_only_ptq": True,
        },
    ),
    "efficientnet_b0": (
        builders.build_efficientnet_b0,
        {
            "name": "efficientnet_b0",
            "family": "efficientnet",
            "torchvision_id": "efficientnet_b0",
            "notes": "EfficientNet-B0, classifier[-1]",
            "supports_weights_only_ptq": True,
        },
    ),
}


def list_models() -> list[str]:
    """Return list of canonical model names."""
    return sorted(_REGISTRY.keys())


def get_model_metadata(model_name: str) -> dict:
    """Return metadata dict for the model. Raises ValueError if unknown."""
    if model_name not in _REGISTRY:
        available = ", ".join(list_models())
        raise ValueError(
            f"Unknown model: {model_name!r}. Available: {available}"
        )
    _, meta = _REGISTRY[model_name]
    return meta.copy()


def build_model(model_name: str, num_classes: int = 100) -> nn.Module:
    """Build model by canonical name. Raises ValueError if unknown."""
    if model_name not in _REGISTRY:
        available = ", ".join(list_models())
        raise ValueError(
            f"Unknown model: {model_name!r}. Available: {available}"
        )
    builder, _ = _REGISTRY[model_name]
    return builder(num_classes)
