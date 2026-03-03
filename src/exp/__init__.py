"""Experiment plumbing: config, paths, logging, CLI entrypoints."""

from .config import build_config_from_args, load_config, save_config
from .constants import (
    DATASET_CHOICES,
    DEFAULT_ALPHA_TEST_GRID,
    DEFAULT_ALPHA_TRAIN_NOISY,
    MODEL_CHOICES,
    TRAIN_REGIMES,
)
from .paths import (
    build_run_dir,
    checkpoints_dir,
    config_path,
    ensure_dirs,
    logs_dir,
    metrics_dir,
)
from .logging_utils import setup_logging, log_config

__all__ = [
    "build_config_from_args",
    "load_config",
    "save_config",
    "DATASET_CHOICES",
    "MODEL_CHOICES",
    "TRAIN_REGIMES",
    "DEFAULT_ALPHA_TRAIN_NOISY",
    "DEFAULT_ALPHA_TEST_GRID",
    "build_run_dir",
    "checkpoints_dir",
    "config_path",
    "ensure_dirs",
    "logs_dir",
    "metrics_dir",
    "setup_logging",
    "log_config",
]
