"""Config dict build from CLI args, save/load with atomic write."""

from __future__ import annotations

import json
import os
import tempfile

from .constants import (
    DEFAULT_ALPHA_TRAIN_NOISY,
    DEFAULT_ALPHA_TEST_GRID,
    DEFAULT_OPT,
    DEFAULT_QUANT,
    DEFAULT_SCHED,
)


def build_config_from_args(args: "argparse.Namespace") -> dict:
    """Build full config dict from parsed CLI args. Reproduces the run."""
    train_regime = getattr(args, "train_regime", "clean")
    if train_regime == "clean":
        alpha_train = 0.0
    else:
        alpha_train = getattr(args, "alpha_train", None)
        if alpha_train is None:
            alpha_train = DEFAULT_ALPHA_TRAIN_NOISY

    alpha_test_grid = getattr(args, "alpha_test_grid", None)
    if alpha_test_grid is None:
        alpha_test_grid = list(DEFAULT_ALPHA_TEST_GRID)

    optimizer = {
        "name": "sgd",
        "lr": getattr(args, "lr"),
        "momentum": getattr(args, "momentum", DEFAULT_OPT["momentum"]),
        "weight_decay": getattr(args, "weight_decay", DEFAULT_OPT["weight_decay"]),
        "nesterov": getattr(args, "nesterov", DEFAULT_OPT["nesterov"]),
    }
    lr_schedule = {
        "name": getattr(args, "sched", DEFAULT_SCHED["name"]) or DEFAULT_SCHED["name"],
        "warmup_epochs": getattr(args, "warmup_epochs", DEFAULT_SCHED["warmup_epochs"]),
    }

    config = {
        "dataset": getattr(args, "dataset", "cifar100"),
        "img_size": getattr(args, "img_size", 224),
        "model": getattr(args, "model"),
        "num_classes": getattr(args, "num_classes", 100),
        "seed": getattr(args, "seed"),
        "train_regime": train_regime,
        "alpha_train": alpha_train,
        "noise_apply_to": ["conv", "linear"],
        "noise_exclude": ["bias", "norm"],
        "epochs": getattr(args, "epochs", 200),
        "batch_size_per_gpu": getattr(args, "batch_size_per_gpu", 128),
        "optimizer": optimizer,
        "lr_schedule": lr_schedule,
        "amp_dtype": getattr(args, "amp", "bf16"),
        "ddp": getattr(args, "ddp", False),
        "eval": {"alpha_test_grid": alpha_test_grid},
        "ptq": dict(DEFAULT_QUANT),
    }
    return config


def save_config(config: dict, path: str) -> None:
    """Write config JSON atomically (temp file then rename)."""
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dirpath or ".", prefix="config.", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_config(path: str) -> dict:
    """Load config dict from JSON file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
