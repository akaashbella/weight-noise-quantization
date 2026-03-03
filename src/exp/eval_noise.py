"""Eval under weight noise: load checkpoint, sweep alpha grid, write metrics and features."""

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
from .eval_noise_utils import compute_features
from .logging_utils import setup_logging
from .noise import WeightNoiseContext
from .paths import checkpoints_dir, config_path, logs_dir, metrics_dir


def _eval_one_run(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    amp_dtype: str,
) -> tuple[float, float]:
    """One full pass over loader; return (mean_loss, acc_percent). model already in eval mode."""
    use_autocast = device.type == "cuda" and amp_dtype in ("fp16", "bf16")
    dtype = torch.bfloat16 if amp_dtype == "bf16" else (torch.float16 if amp_dtype == "fp16" else None)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    steps = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if use_autocast:
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == y).float().sum().item()
            total_samples += y.size(0)
            steps += 1
    mean_loss = total_loss / steps if steps else 0.0
    acc = 100.0 * total_correct / total_samples if total_samples else 0.0
    return mean_loss, acc


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval under weight noise sweep.")
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
    parser.add_argument(
        "--alpha-test-grid",
        type=float,
        nargs="+",
        default=None,
        help="Override alpha grid (default from config)",
    )
    parser.add_argument("--data-root", type=str, default="data", help="Data root")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument(
        "--collapse-threshold",
        type=float,
        default=0.1,
        help="Acc fraction for collapse_alpha (0.1 = 10%%)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="Repeats per alpha for mean/std",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    cfg_path = config_path(run_dir)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    config = load_config(cfg_path)

    log_path = os.path.join(logs_dir(run_dir), "eval_noise.log")
    setup_logging(log_path)
    logger = logging.getLogger()

    # Alpha grid: ensure 0.0, sort, dedupe
    alpha_test_grid = args.alpha_test_grid
    if alpha_test_grid is None:
        alpha_test_grid = list(config.get("eval", {}).get("alpha_test_grid", [0.0]))
    alpha_test_grid = sorted(set(alpha_test_grid))
    if 0.0 not in alpha_test_grid:
        logger.warning("alpha_test_grid missing 0.0; inserting and evaluating first")
        alpha_test_grid.insert(0, 0.0)

    # Checkpoint path
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
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    amp_dtype = config.get("amp_dtype", "bf16")

    results = {}
    for alpha in alpha_test_grid:
        losses: list[float] = []
        accs: list[float] = []
        for _ in range(args.n_repeats):
            with WeightNoiseContext(model, alpha):
                loss, acc = _eval_one_run(model, loader, criterion, device, amp_dtype)
            losses.append(loss)
            accs.append(acc)
        n = len(losses)
        loss_mean = sum(losses) / n
        loss_std = (sum((x - loss_mean) ** 2 for x in losses) / n) ** 0.5 if n > 1 else 0.0
        acc_mean = sum(accs) / n
        acc_std = (sum((x - acc_mean) ** 2 for x in accs) / n) ** 0.5 if n > 1 else 0.0
        results[str(alpha)] = {
            "loss_mean": loss_mean,
            "loss_std": loss_std,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
        }

    alpha_list = alpha_test_grid
    acc_means = [results[str(a)]["acc_mean"] for a in alpha_list]
    features = compute_features(alpha_list, acc_means, args.collapse_threshold)

    out = {
        "split": args.split,
        "checkpoint": ckpt_name,
        "alpha_test_grid": alpha_list,
        "n_repeats": args.n_repeats,
        "results": results,
        "features": features,
    }

    metrics_path = os.path.join(metrics_dir(run_dir), "eval_noise.json")
    os.makedirs(metrics_dir(run_dir), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=metrics_dir(run_dir), prefix="eval_noise.", suffix=".json")
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

    # Log concise table
    logger.info("split=%s checkpoint=%s n_repeats=%d", args.split, ckpt_name, args.n_repeats)
    logger.info("alpha    loss_mean  acc_mean  acc_std")
    for a in alpha_list:
        r = results[str(a)]
        logger.info("%.4f   %.4f   %.2f   %.2f", a, r["loss_mean"], r["acc_mean"], r["acc_std"])
    logger.info("features: acc0=%.2f auc_acc=%.2f init_slope=%.2f alpha_50=%s collapse_alpha=%s",
                features["acc0"], features["auc_acc"], features["init_slope"],
                features["alpha_50"], features["collapse_alpha"])
    logger.info("Wrote %s", metrics_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
