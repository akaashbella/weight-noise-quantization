"""Run directory layout and path helpers."""

from __future__ import annotations

import os


def build_run_dir(
    runs_root: str,
    dataset: str,
    model: str,
    regime: str,
    seed: int,
) -> str:
    """Return run directory path: {runs_root}/{dataset}/{model}/{regime}/seed_{seed}."""
    return os.path.join(runs_root, dataset, model, regime, f"seed_{seed}")


def checkpoints_dir(run_dir: str) -> str:
    """Standard checkpoints subdirectory."""
    return os.path.join(run_dir, "checkpoints")


def logs_dir(run_dir: str) -> str:
    """Standard logs subdirectory."""
    return os.path.join(run_dir, "logs")


def metrics_dir(run_dir: str) -> str:
    """Standard metrics subdirectory."""
    return os.path.join(run_dir, "metrics")


def config_path(run_dir: str) -> str:
    """Path to config.json in run directory."""
    return os.path.join(run_dir, "config.json")


def ensure_dirs(run_dir: str) -> None:
    """Create run_dir and standard subdirectories (checkpoints, logs, metrics)."""
    for d in (run_dir, checkpoints_dir(run_dir), logs_dir(run_dir), metrics_dir(run_dir)):
        os.makedirs(d, exist_ok=True)
