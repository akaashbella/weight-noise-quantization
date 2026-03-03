"""Helpers for aggregation: read jsonl, summarize training."""

from __future__ import annotations

import json
import os


def read_jsonl(path: str) -> list[dict]:
    """Read one JSON object per line; return list of dicts."""
    if not os.path.isfile(path):
        return []
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def summarize_training(
    train_jsonl: list[dict],
    val_jsonl: list[dict],
) -> dict:
    """Extract best val, last train/val metrics. Keys: best_val_acc, best_val_epoch, last_train_acc, last_train_loss, last_val_acc, last_val_loss."""
    result: dict = {
        "best_val_acc": None,
        "best_val_epoch": None,
        "last_train_acc": None,
        "last_train_loss": None,
        "last_val_acc": None,
        "last_val_loss": None,
        "total_time_sec": None,
    }
    if val_jsonl:
        best = max(val_jsonl, key=lambda r: r.get("acc_top1", -1))
        result["best_val_acc"] = best.get("acc_top1")
        result["best_val_epoch"] = best.get("epoch")
        last_val = val_jsonl[-1]
        result["last_val_acc"] = last_val.get("acc_top1")
        result["last_val_loss"] = last_val.get("loss")
    if train_jsonl:
        last_train = train_jsonl[-1]
        result["last_train_acc"] = last_train.get("acc_top1")
        result["last_train_loss"] = last_train.get("loss")
        if "time_sec" in last_train:
            result["total_time_sec"] = sum(r.get("time_sec", 0) for r in train_jsonl)
    return result
