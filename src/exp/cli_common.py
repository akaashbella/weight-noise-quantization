"""Shared CLI argument definitions for experiment entrypoints."""

from __future__ import annotations

import argparse

from .constants import (
    DATASET_CHOICES,
    DEFAULT_AMP_DTYPE,
    DEFAULT_BATCH_SIZE_PER_GPU,
    DEFAULT_EPOCHS,
    DEFAULT_OPT,
    DEFAULT_SCHED,
    MODEL_CHOICES,
    TRAIN_REGIMES,
)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common experiment arguments to parser."""
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="cifar100",
        help="Dataset name",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Input image size (H, W)",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        required=True,
        help="Model architecture",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=100,
        help="Number of classes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed",
    )
    parser.add_argument(
        "--train-regime",
        choices=TRAIN_REGIMES,
        required=True,
        help="Training regime (clean or noisy)",
    )
    parser.add_argument(
        "--alpha-train",
        type=float,
        default=None,
        help="Noise alpha for training (noisy regime only; default from constants)",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs",
        help="Root directory for run directories",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Override computed run_dir",
    )
    parser.add_argument(
        "--amp",
        choices=["fp16", "bf16", "fp32"],
        default=DEFAULT_AMP_DTYPE,
        help="AMP dtype",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        default=False,
        help="Use distributed data parallel",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch-size-per-gpu",
        type=int,
        default=DEFAULT_BATCH_SIZE_PER_GPU,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=True,
        help="Learning rate",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=DEFAULT_OPT["momentum"],
        help="SGD momentum",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_OPT["weight_decay"],
        help="Weight decay",
    )
    parser.add_argument(
        "--nesterov",
        action="store_true",
        default=DEFAULT_OPT["nesterov"],
        help="Use Nesterov momentum",
    )
    parser.add_argument(
        "--sched",
        choices=["cosine"],
        default=DEFAULT_SCHED["name"],
        help="LR schedule",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=DEFAULT_SCHED["warmup_epochs"],
        help="Warmup epochs",
    )
    parser.add_argument(
        "--alpha-test-grid",
        type=float,
        nargs="+",
        default=None,
        help="Alpha values for noise eval (default from config/constants)",
    )
