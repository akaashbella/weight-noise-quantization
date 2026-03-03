"""Canonical experiment constants."""

from __future__ import annotations

DATASET_CHOICES = ["cifar100"]
MODEL_CHOICES = [
    "resnet50",
    "mobilenetv3_large",
    "convnext_tiny",
    "efficientnet_b0",
]
TRAIN_REGIMES = ["clean", "noisy"]
DEFAULT_ALPHA_TRAIN_NOISY = 0.05
DEFAULT_ALPHA_TEST_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
DEFAULT_IMG_SIZE = 224
DEFAULT_NUM_CLASSES = 100
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE_PER_GPU = 128
DEFAULT_OPT = {
    "name": "sgd",
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
}
DEFAULT_SCHED = {"name": "cosine", "warmup_epochs": 5}
DEFAULT_AMP_DTYPE = "bf16"
DEFAULT_QUANT = {
    "enabled": True,
    "mode": "weights_only_w8",
    "granularity": "per_out_channel",
    "symmetric": True,
    "scale_method": "maxabs",
    "quantize_layers": ["conv", "linear"],
}
