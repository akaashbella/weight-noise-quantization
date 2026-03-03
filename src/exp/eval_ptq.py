"""Weights-only PTQ eval: W8A16, compute drop vs FP32."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile

import torch

from .checkpointing import load_checkpoint
from .config import load_config
from .eval_common import evaluate
from .logging_utils import setup_logging
from .paths import checkpoints_dir, config_path, logs_dir, metrics_dir
from .ptq import apply_weights_only_ptq_inplace, clone_model_for_ptq


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval weights-only PTQ (W8A16).")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory")
    parser.add_argument(
        "--checkpoint",
        choices=["best", "last"],
        default="best",
        help="Checkpoint to load",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Split to evaluate",
    )
    parser.add_argument("--data-root", type=str, default="data", help="Data root")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument(
        "--compute-fp32-baseline",
        action="store_true",
        help="Always compute FP32 baseline in this script",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    cfg_path = config_path(run_dir)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    config = load_config(cfg_path)

    log_path = os.path.join(logs_dir(run_dir), "eval_ptq.log")
    setup_logging(log_path)
    logger = logging.getLogger()

    ckpt_dir = checkpoints_dir(run_dir)
    ckpt_name = args.checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"{ckpt_name}.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    from src.data import build_dataloaders
    from src.models import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size if args.batch_size is not None else config["batch_size_per_gpu"]
    loaders = build_dataloaders(
        dataset=config["dataset"],
        data_root=args.data_root,
        img_size=config["img_size"],
        batch_size=batch_size,
        num_workers=args.num_workers,
        seed=config["seed"],
        val_fraction=0.1,
    )
    loader = loaders[args.split]

    model = build_model(config["model"], num_classes=config["num_classes"])
    load_checkpoint(ckpt_path, model, optimizer=None, scaler=None)
    model = model.to(device)

    amp_dtype = config.get("amp_dtype", "bf16")
    if device.type != "cuda":
        amp_dtype = "fp32"

    # FP32 baseline
    if args.compute_fp32_baseline:
        fp32_metrics = evaluate(model, loader, device, amp_dtype=None)
        fp32_loss = fp32_metrics["loss"]
        fp32_acc = fp32_metrics["acc"]
        logger.info("Computed FP32 baseline: loss=%.4f acc=%.2f", fp32_loss, fp32_acc)
    else:
        fp32_loss = None
        fp32_acc = None
        test_metrics_path = os.path.join(metrics_dir(run_dir), "test_metrics.json")
        if args.split == "test" and os.path.isfile(test_metrics_path):
            with open(test_metrics_path, encoding="utf-8") as f:
                test_data = json.load(f)
            fp32_loss = test_data.get("loss")
            fp32_acc = test_data.get("acc_top1") or test_data.get("acc")
        if fp32_loss is None or fp32_acc is None:
            logger.info("Computing FP32 baseline")
            fp32_metrics = evaluate(model, loader, device, amp_dtype=None)
            fp32_loss = fp32_metrics["loss"]
            fp32_acc = fp32_metrics["acc"]
        else:
            logger.info("FP32 baseline from %s: loss=%.4f acc=%.2f", test_metrics_path, fp32_loss, fp32_acc)

    # PTQ on clone
    model_ptq = clone_model_for_ptq(model, device)
    layer_stats = apply_weights_only_ptq_inplace(model_ptq)
    ptq_metrics = evaluate(model_ptq, loader, device, amp_dtype=None)
    ptq_loss = ptq_metrics["loss"]
    ptq_acc = ptq_metrics["acc"]

    drop_acc = fp32_acc - ptq_acc
    drop_loss = ptq_loss - fp32_loss

    ptq_cfg = config.get("ptq", {})
    out = {
        "split": args.split,
        "checkpoint": ckpt_name,
        "ptq": {
            "mode": ptq_cfg.get("mode", "weights_only_w8"),
            "granularity": ptq_cfg.get("granularity", "per_out_channel"),
            "symmetric": ptq_cfg.get("symmetric", True),
            "scale_method": ptq_cfg.get("scale_method", "maxabs"),
        },
        "fp32": {"loss": fp32_loss, "acc": fp32_acc},
        "ptq_w8": {"loss": ptq_loss, "acc": ptq_acc},
        "drop": {"acc": drop_acc, "loss": drop_loss},
        "layer_stats": layer_stats,
    }

    metrics_path = os.path.join(metrics_dir(run_dir), "eval_ptq.json")
    os.makedirs(metrics_dir(run_dir), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=metrics_dir(run_dir), prefix="eval_ptq.", suffix=".json")
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

    logger.info("PTQ W8: loss=%.4f acc=%.2f | drop acc=%.2f loss=%.4f | layers=%d",
                ptq_loss, ptq_acc, drop_acc, drop_loss, len(layer_stats))
    logger.info("Wrote %s", metrics_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
