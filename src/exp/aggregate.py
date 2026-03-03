"""Aggregate runs: index, train/noise/ptq summaries, correlation, plots."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from .analysis import inner_join, load_csv, run_correlation_analysis
from .aggregate_utils import read_jsonl, summarize_training
from .config import load_config
from .paths import metrics_dir
from .plotting import produce_all_plots


def _discover_runs(runs_root: str, dataset: str) -> list[dict]:
    """Walk runs_root/dataset/*/*/ and find config.json. Return list of run info dicts."""
    runs: list[dict] = []
    base = os.path.join(runs_root, dataset)
    if not os.path.isdir(base):
        return runs
    for model in os.listdir(base):
        model_path = os.path.join(base, model)
        if not os.path.isdir(model_path):
            continue
        for regime in os.listdir(model_path):
            regime_path = os.path.join(model_path, regime)
            if not os.path.isdir(regime_path):
                continue
            for name in os.listdir(regime_path):
                run_dir = os.path.join(regime_path, name)
                cfg_path = os.path.join(run_dir, "config.json")
                if not os.path.isfile(cfg_path):
                    continue
                try:
                    cfg = load_config(cfg_path)
                except Exception:
                    continue
                meta_dir = metrics_dir(run_dir)
                runs.append({
                    "dataset": cfg.get("dataset", dataset),
                    "model": cfg.get("model", model),
                    "train_regime": cfg.get("train_regime", regime),
                    "seed": cfg.get("seed", ""),
                    "run_dir": run_dir,
                    "has_train": os.path.isfile(os.path.join(meta_dir, "train_metrics.jsonl")) and os.path.isfile(os.path.join(meta_dir, "val_metrics.jsonl")),
                    "has_eval_noise": os.path.isfile(os.path.join(meta_dir, "eval_noise.json")),
                    "has_eval_ptq": os.path.isfile(os.path.join(meta_dir, "eval_ptq.json")),
                    "has_test": os.path.isfile(os.path.join(meta_dir, "test_metrics.json")),
                })
    return runs


def _write_summary_index(runs: list[dict], out_path: str) -> None:
    cols = ["dataset", "model", "train_regime", "seed", "run_dir", "has_train", "has_eval_noise", "has_eval_ptq", "has_test"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in runs:
            row = {c: r.get(c, "") for c in cols}
            row["has_train"] = "1" if r.get("has_train") else "0"
            row["has_eval_noise"] = "1" if r.get("has_eval_noise") else "0"
            row["has_eval_ptq"] = "1" if r.get("has_eval_ptq") else "0"
            row["has_test"] = "1" if r.get("has_test") else "0"
            w.writerow(row)


def _write_summary_train(runs: list[dict], out_path: str) -> None:
    cols = ["dataset", "model", "train_regime", "seed", "best_val_acc", "best_val_epoch", "last_val_acc", "last_val_loss", "run_dir"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            if not r.get("has_train"):
                continue
            meta = metrics_dir(r["run_dir"])
            train_rows = read_jsonl(os.path.join(meta, "train_metrics.jsonl"))
            val_rows = read_jsonl(os.path.join(meta, "val_metrics.jsonl"))
            summary = summarize_training(train_rows, val_rows)
            row = {
                "dataset": r["dataset"],
                "model": r["model"],
                "train_regime": r["train_regime"],
                "seed": r["seed"],
                "best_val_acc": summary.get("best_val_acc"),
                "best_val_epoch": summary.get("best_val_epoch"),
                "last_val_acc": summary.get("last_val_acc"),
                "last_val_loss": summary.get("last_val_loss"),
                "run_dir": r["run_dir"],
            }
            w.writerow(row)


def _alpha_col(alpha: float) -> str:
    """Column name for per-alpha acc: acc_0_0, acc_0_01, etc."""
    s = str(alpha)
    return "acc_" + s.replace(".", "_")


def _write_summary_eval_noise(runs: list[dict], out_path: str) -> None:
    rows: list[dict] = []
    for r in runs:
        if not r.get("has_eval_noise"):
            continue
        p = os.path.join(metrics_dir(r["run_dir"]), "eval_noise.json")
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        features = data.get("features", {})
        results = data.get("results", {})
        row = {
            "dataset": r["dataset"],
            "model": r["model"],
            "train_regime": r["train_regime"],
            "seed": r["seed"],
            "acc0": features.get("acc0"),
            "auc_acc": features.get("auc_acc"),
            "init_slope": features.get("init_slope"),
            "alpha_50": features.get("alpha_50"),
            "collapse_alpha": features.get("collapse_alpha"),
            "run_dir": r["run_dir"],
        }
        for alpha_str, vals in results.items():
            try:
                a = float(alpha_str)
                row[_alpha_col(a)] = vals.get("acc_mean")
            except (ValueError, TypeError):
                pass
        rows.append(row)

    if not rows:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            f.write("dataset,model,train_regime,seed,acc0,auc_acc,init_slope,alpha_50,collapse_alpha,run_dir\n")
        return

    all_alpha_cols = set()
    for row in rows:
        for k in row:
            if k.startswith("acc_") and k != "acc0":
                all_alpha_cols.add(k)
    alpha_cols_sorted = sorted(all_alpha_cols, key=lambda c: float(c.replace("acc_", "").replace("_", ".")))
    cols = ["dataset", "model", "train_regime", "seed", "acc0", "auc_acc", "init_slope", "alpha_50", "collapse_alpha"] + alpha_cols_sorted + ["run_dir"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_summary_eval_ptq(runs: list[dict], out_path: str) -> None:
    cols = ["dataset", "model", "train_regime", "seed", "fp32_acc", "ptq_acc", "qdrop_acc", "fp32_loss", "ptq_loss", "qdrop_loss", "run_dir"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in runs:
            if not r.get("has_eval_ptq"):
                continue
            p = os.path.join(metrics_dir(r["run_dir"]), "eval_ptq.json")
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            fp32 = data.get("fp32", {})
            ptq = data.get("ptq_w8", {})
            drop = data.get("drop", {})
            w.writerow({
                "dataset": r["dataset"],
                "model": r["model"],
                "train_regime": r["train_regime"],
                "seed": r["seed"],
                "fp32_acc": fp32.get("acc"),
                "ptq_acc": ptq.get("acc"),
                "qdrop_acc": drop.get("acc"),
                "fp32_loss": fp32.get("loss"),
                "ptq_loss": ptq.get("loss"),
                "qdrop_loss": drop.get("loss"),
                "run_dir": r["run_dir"],
            })


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate runs: CSVs, correlation, plots.")
    parser.add_argument("--runs-root", type=str, default="runs", help="Root of run directories")
    parser.add_argument("--out-root", type=str, default="results", help="Output root")
    parser.add_argument("--dataset", type=str, default="cifar100", help="Dataset name")
    args = parser.parse_args()

    runs = _discover_runs(args.runs_root, args.dataset)
    out_sub = os.path.join(args.out_root, args.dataset)
    os.makedirs(out_sub, exist_ok=True)
    plots_dir = os.path.join(out_sub, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1) summary_index.csv
    index_path = os.path.join(out_sub, "summary_index.csv")
    _write_summary_index(runs, index_path)
    print(f"Index: {len(runs)} runs -> {index_path}")

    # 2) summary_train.csv
    train_path = os.path.join(out_sub, "summary_train.csv")
    _write_summary_train(runs, train_path)
    n_train = sum(1 for r in runs if r.get("has_train"))
    print(f"Train: {n_train} rows -> {train_path}")

    # 3) summary_eval_noise.csv
    noise_path = os.path.join(out_sub, "summary_eval_noise.csv")
    _write_summary_eval_noise(runs, noise_path)
    n_noise = sum(1 for r in runs if r.get("has_eval_noise"))
    print(f"Eval noise: {n_noise} rows -> {noise_path}")

    # 4) summary_eval_ptq.csv
    ptq_path = os.path.join(out_sub, "summary_eval_ptq.csv")
    _write_summary_eval_ptq(runs, ptq_path)
    n_ptq = sum(1 for r in runs if r.get("has_eval_ptq"))
    print(f"Eval PTQ: {n_ptq} rows -> {ptq_path}")

    # 5) Join + correlation
    noise_rows = load_csv(noise_path)
    ptq_rows = load_csv(ptq_path)
    joined = inner_join(noise_rows, ptq_rows, ["dataset", "model", "train_regime", "seed"])
    corr_rows = run_correlation_analysis(joined, qdrop_key="qdrop_acc")
    corr_path = os.path.join(out_sub, "correlation_analysis.csv")
    with open(corr_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "feature", "n", "pearson_r", "spearman_rho"])
        w.writeheader()
        w.writerows(corr_rows)
    print(f"Correlation: {len(corr_rows)} rows -> {corr_path}")

    # 6) Plots
    produce_all_plots(noise_path, joined, plots_dir)
    print(f"Plots: {plots_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
