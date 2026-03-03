"""Setup logging to stdout and file with timestamps."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime


def setup_logging(log_path: str, level: int = logging.INFO) -> logging.Logger:
    """Configure root logger to log to stdout and file. Returns logger."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(level)
    # Avoid duplicate handlers when called multiple times
    logger.handlers.clear()
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def log_config(logger: logging.Logger, config: dict) -> None:
    """Pretty-print config JSON (sorted keys, indent=2)."""
    text = json.dumps(config, indent=2, sort_keys=True)
    logger.info("Config:\n%s", text)
