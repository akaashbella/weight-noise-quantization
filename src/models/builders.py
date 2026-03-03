"""Per-model builder functions. All models from scratch (weights=None)."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tv_models


def build_resnet50(num_classes: int) -> nn.Module:
    """ResNet-50 for CIFAR-100, classifier head replaced for num_classes."""
    model = tv_models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_mobilenetv3_large(num_classes: int) -> nn.Module:
    """MobileNetV3 Large for CIFAR-100, classifier head replaced for num_classes."""
    model = tv_models.mobilenet_v3_large(weights=None)
    last_layer = model.classifier[-1]
    in_features = last_layer.in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_convnext_tiny(num_classes: int) -> nn.Module:
    """ConvNeXt Tiny for CIFAR-100, classifier head replaced for num_classes."""
    model = tv_models.convnext_tiny(weights=None)
    last_layer = model.classifier[-1]
    in_features = last_layer.in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int) -> nn.Module:
    """EfficientNet-B0 for CIFAR-100, classifier head replaced for num_classes."""
    model = tv_models.efficientnet_b0(weights=None)
    last_layer = model.classifier[-1]
    in_features = last_layer.in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
