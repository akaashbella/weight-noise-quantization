"""Eval PTQ (stub: load config, write stub metrics)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

from .config import load_config
from .logging_utils import log_config, setup_logging
from .paths import config_path, logs_dir, metrics_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval PTQ (stub).")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory")
    parser.add_argument(
        "--checkpoint",
        choices=["best", "last"],
        default="best",
        help="Checkpoint to load",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    cfg_path = config_path(run_dir)
    config = load_config(cfg_path)

    log_path = os.path.join(logs_dir(run_dir), "eval_ptq.log")
    setup_logging(log_path)
    logger = __import__("logging").getLogger()
    log_config(logger, config)

    ptq_mode = config.get("ptq", {}).get("mode", "weights_only_w8")
    out = {
        "mode": ptq_mode,
        "results": {},
        "note": "stub",
    }
    metrics_path = os.path.join(metrics_dir(run_dir), "eval_ptq.json")
    os.makedirs(metrics_dir(run_dir), exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=metrics_dir(run_dir), prefix="eval_ptq.", suffix=".json"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        os.replace(tmp, metrics_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    logger.info("Wrote stub %s", metrics_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
