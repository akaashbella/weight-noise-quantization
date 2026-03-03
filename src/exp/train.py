"""Training entrypoint (stub: config + dirs + logging only)."""

from __future__ import annotations

import argparse
import sys

from .cli_common import add_common_args
from .config import build_config_from_args, save_config
from .logging_utils import log_config, setup_logging
from .paths import (
    build_run_dir,
    config_path,
    ensure_dirs,
    logs_dir,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train model (stub).")
    add_common_args(parser)
    args = parser.parse_args()

    if args.out:
        run_dir = args.out
    else:
        run_dir = build_run_dir(
            args.runs_root,
            args.dataset,
            args.model,
            args.train_regime,
            args.seed,
        )

    ensure_dirs(run_dir)
    log_path = f"{logs_dir(run_dir)}/train.log"
    setup_logging(log_path)
    logger = __import__("logging").getLogger()

    config = build_config_from_args(args)
    config_path_str = config_path(run_dir)
    save_config(config, config_path_str)
    log_config(logger, config)

    from src.data import build_dataloaders

    loaders = build_dataloaders(
        dataset=config["dataset"],
        data_root="data",
        img_size=config["img_size"],
        batch_size=config["batch_size_per_gpu"],
        num_workers=8,
        seed=config["seed"],
    )
    x, y = next(iter(loaders["train"]))
    logger.info("Data batch: x %s, y %s", list(x.shape), list(y.shape))

    logger.info("TRAIN STUB: would train model here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
