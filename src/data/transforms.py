"""ImageNet-style transforms for CIFAR-100 @ 224."""

from __future__ import annotations

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


def imagenet_normalize() -> T.Normalize:
    """ImageNet mean/std normalizer."""
    return T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


def build_train_transform(img_size: int) -> T.Compose:
    """Train transform: RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize."""
    return T.Compose([
        T.RandomResizedCrop(
            img_size,
            scale=(0.7, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        imagenet_normalize(),
    ])


def build_test_transform(img_size: int) -> T.Compose:
    """Test transform: Resize(256-style), CenterCrop, ToTensor, Normalize."""
    resize_size = int(img_size * 256 / 224)
    return T.Compose([
        T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        imagenet_normalize(),
    ])
