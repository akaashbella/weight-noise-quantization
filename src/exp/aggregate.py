"""Aggregate run configs into summary index (stub)."""

from __future__ import annotations

import argparse
import csv
import os
import sys

from .config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate run configs into summary.")
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs",
        help="Root of run directories",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="results",
        help="Output root for summary files",
    )
    args = parser.parse_args()

    runs_root = args.runs_root
    out_root = args.out_root
    if not os.path.isdir(runs_root):
        return 0  # No runs to aggregate

    configs: list[dict] = []
    for dirpath, _dirnames, filenames in os.walk(runs_root):
        if "config.json" not in filenames:
            continue
        path = os.path.join(dirpath, "config.json")
        try:
            cfg = load_config(path)
        except Exception:
            continue
        run_dir = dirpath
        cfg["_run_dir"] = run_dir
        configs.append(cfg)

    # results/cifar100/summary_index.csv with columns: dataset, model, train_regime, seed, run_dir
    by_dataset: dict[str, list[dict]] = {}
    for c in configs:
        ds = c.get("dataset", "cifar100")
        by_dataset.setdefault(ds, []).append(c)

    os.makedirs(out_root, exist_ok=True)
    for dataset, cfgs in by_dataset.items():
        sub = os.path.join(out_root, dataset)
        os.makedirs(sub, exist_ok=True)
        csv_path = os.path.join(sub, "summary_index.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "model", "train_regime", "seed", "run_dir"])
            for c in cfgs:
                w.writerow([
                    c.get("dataset", dataset),
                    c.get("model", ""),
                    c.get("train_regime", ""),
                    c.get("seed", ""),
                    c.get("_run_dir", ""),
                ])
        print(f"Wrote {csv_path} ({len(cfgs)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
