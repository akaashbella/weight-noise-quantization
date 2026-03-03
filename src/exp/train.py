"""Training entrypoint: full loop with clean/noisy regimes, checkpoints, metrics."""

from __future__ import annotations

import argparse
import logging
import sys

import torch

from .checkpointing import load_checkpoint, save_checkpoint
from .cli_common import add_common_args
from .config import build_config_from_args, save_config
from .logging_utils import log_config, setup_logging
from .metrics import append_metrics_jsonl, write_metrics_json
from .paths import (
    build_run_dir,
    checkpoints_dir,
    config_path,
    ensure_dirs,
    logs_dir,
    metrics_dir,
)
from .schedulers import build_lr_scheduler
from .train_utils import set_seed, train_one_epoch, validate_one_epoch


def main() -> int:
    parser = argparse.ArgumentParser(description="Train model.")
    add_common_args(parser)
    parser.add_argument("--data-root", type=str, default="data", help="Data root")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Val split fraction")
    parser.add_argument("--log-every", type=int, default=50, help="Log every N steps")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    if getattr(args, "ddp", False):
        raise RuntimeError("DDP not implemented yet")

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
    logger = logging.getLogger()

    config = build_config_from_args(args)
    config_path_str = config_path(run_dir)
    save_config(config, config_path_str)
    log_config(logger, config)

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    from src.data import build_dataloaders
    from src.models import build_model

    loaders = build_dataloaders(
        dataset=config["dataset"],
        data_root=args.data_root,
        img_size=config["img_size"],
        batch_size=config["batch_size_per_gpu"],
        num_workers=args.num_workers,
        seed=config["seed"],
        val_fraction=args.val_fraction,
    )
    num_classes = config["num_classes"]
    model = build_model(config["model"], num_classes=num_classes)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    opt_cfg = config["optimizer"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt_cfg["lr"],
        momentum=opt_cfg["momentum"],
        weight_decay=opt_cfg["weight_decay"],
        nesterov=opt_cfg["nesterov"],
    )

    epochs = config["epochs"]
    warmup_epochs = config["lr_schedule"]["warmup_epochs"]
    steps_per_epoch = len(loaders["train"])
    lr_step_fn = build_lr_scheduler(optimizer, epochs, warmup_epochs, steps_per_epoch)

    amp_dtype = config["amp_dtype"]
    use_amp = amp_dtype in ("fp16", "bf16")
    scaler = None
    if amp_dtype == "fp16" and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    start_epoch = 0
    best_metric = -1.0
    global_step = 0
    if args.resume:
        ckpt_path = args.resume
        state = load_checkpoint(ckpt_path, model, optimizer, scaler)
        start_epoch = state.get("epoch", 0) + 1
        best_metric = state.get("best_metric", -1.0)
        global_step = start_epoch * steps_per_epoch
        logger.info("Resumed from %s at epoch %d, best_metric %.2f", ckpt_path, start_epoch - 1, best_metric)

    alpha_train = config["alpha_train"]
    train_metrics_path = f"{metrics_dir(run_dir)}/train_metrics.jsonl"
    val_metrics_path = f"{metrics_dir(run_dir)}/val_metrics.jsonl"
    ckpt_dir = checkpoints_dir(run_dir)

    for epoch in range(start_epoch, epochs):
        logger.info("Epoch %d/%d", epoch + 1, epochs)
        train_metrics, global_step = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            lr_step_fn,
            global_step,
            amp_dtype,
            scaler,
            alpha_train,
            args.log_every,
            logger,
        )
        append_metrics_jsonl(train_metrics_path, {
            "epoch": epoch,
            "split": "train",
            "loss": train_metrics["loss"],
            "acc_top1": train_metrics["acc_top1"],
            "lr": train_metrics["lr"],
            "time_sec": train_metrics["time_sec"],
        })
        logger.info("Train epoch %d loss=%.4f acc=%.2f lr=%.6f", epoch, train_metrics["loss"], train_metrics["acc_top1"], train_metrics["lr"])

        val_metrics = validate_one_epoch(model, loaders["val"], criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        append_metrics_jsonl(val_metrics_path, {
            "epoch": epoch,
            "split": "val",
            "loss": val_metrics["loss"],
            "acc_top1": val_metrics["acc_top1"],
            "lr": current_lr,
            "time_sec": val_metrics["time_sec"],
        })
        logger.info("Val   epoch %d loss=%.4f acc=%.2f", epoch, val_metrics["loss"], val_metrics["acc_top1"])

        save_checkpoint(
            f"{ckpt_dir}/last.pt",
            model,
            optimizer,
            epoch,
            best_metric,
            config,
            scaler.state_dict() if scaler else None,
        )
        if val_metrics["acc_top1"] > best_metric:
            best_metric = val_metrics["acc_top1"]
            save_checkpoint(
                f"{ckpt_dir}/best.pt",
                model,
                optimizer,
                epoch,
                best_metric,
                config,
                scaler.state_dict() if scaler else None,
            )
            logger.info("New best val acc %.2f -> best.pt", best_metric)

    # Final test eval (alpha=0, load best)
    best_ckpt = f"{ckpt_dir}/best.pt"
    load_checkpoint(best_ckpt, model, optimizer=None, scaler=None)
    model.eval()
    test_metrics = validate_one_epoch(model, loaders["test"], criterion, device)
    write_metrics_json(
        f"{metrics_dir(run_dir)}/test_metrics.json",
        {"loss": test_metrics["loss"], "acc_top1": test_metrics["acc_top1"]},
    )
    logger.info("Test (best.pt) loss=%.4f acc=%.2f", test_metrics["loss"], test_metrics["acc_top1"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
