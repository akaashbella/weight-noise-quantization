"""Models: torchvision backbones for CIFAR-100 classification."""

from .registry import build_model, get_model_metadata, list_models
from .utils import assert_forward_works, count_params, count_trainable_params

__all__ = [
    "build_model",
    "list_models",
    "get_model_metadata",
    "assert_forward_works",
    "count_params",
    "count_trainable_params",
]
