"""Matplotlib plots for aggregation results."""

from __future__ import annotations

import csv
import os
from typing import Any


def _load_csv(path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _get_alpha_columns(rows: list[dict]) -> tuple[list[float], list[str]]:
    """Return (alphas, col_names) for curve: acc_* columns sorted by alpha; use acc0 for alpha=0 if no acc_0_0."""
    if not rows:
        return [], []
    acc_underscore = [k for k in rows[0].keys() if k.startswith("acc_")]
    acc_underscore.sort(key=lambda c: float(c.replace("acc_", "").replace("_", ".")))
    parsed = [float(c.replace("acc_", "").replace("_", ".")) for c in acc_underscore]
    if "acc0" in rows[0].keys() and 0.0 not in parsed:
        alphas = [0.0] + parsed
        col_names = ["acc0"] + acc_underscore
    else:
        alphas = parsed
        col_names = acc_underscore
    return alphas, col_names


def plot_noise_curves_by_model_regime(
    summary_eval_noise_path: str,
    out_path: str,
) -> None:
    """Plot mean acc vs alpha per (model, regime), one line per (model, regime)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_csv(summary_eval_noise_path)
    if not rows:
        plt.figure(figsize=(8, 5))
        plt.title("Noise curves (no data)")
        plt.savefig(out_path)
        plt.close()
        return

    alphas, col_names = _get_alpha_columns(rows)
    if not col_names:
        alphas = [0.0]
        col_names = ["acc0"] if "acc0" in rows[0] else []
    if not col_names:
        plt.figure(figsize=(8, 5))
        plt.title("Noise curves (no alpha columns)")
        plt.savefig(out_path)
        plt.close()
        return

    by_key: dict[tuple[str, str], list[list[float]]] = {}
    for row in rows:
        model = row.get("model", "")
        regime = row.get("train_regime", "")
        key = (model, regime)
        vals = []
        for col in col_names:
            v = row.get(col)
            try:
                vals.append(float(v) if v not in (None, "") else None)
            except (TypeError, ValueError):
                vals.append(None)
        # filter to numeric only for this row
        numeric_vals = [v for v in vals if v is not None]
        if not numeric_vals:
            continue
        if key not in by_key:
            by_key[key] = []
        by_key[key].append(vals)

    plt.figure(figsize=(10, 6))
    for (model, regime), list_of_vals in sorted(by_key.items()):
        # list_of_vals is list of rows; average per column
        n_cols = len(col_names)
        means = []
        for j in range(n_cols):
            col_vals = [r[j] for r in list_of_vals if r[j] is not None]
            means.append(sum(col_vals) / len(col_vals)) if col_vals else means.append(0.0)
        plt.plot(alphas, means, label=f"{model} {regime}", marker="o", markersize=3)
    plt.xlabel("Alpha")
    plt.ylabel("Accuracy (%)")
    plt.title("Noise curves by model and regime (mean over seeds)")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _scatter_plot(
    joined_rows: list[dict],
    x_key: str,
    y_key: str,
    out_path: str,
    title: str,
    skip_null_x: bool = False,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_vals, y_vals = [], []
    regimes = []
    for row in joined_rows:
        xv = row.get(x_key)
        yv = row.get(y_key)
        if skip_null_x and (xv is None or xv == ""):
            continue
        try:
            x_vals.append(float(xv))
            y_vals.append(float(yv))
            regimes.append(row.get("train_regime", ""))
        except (TypeError, ValueError):
            continue

    if not x_vals:
        plt.figure(figsize=(6, 5))
        plt.title(title + " (no data)")
        plt.savefig(out_path)
        plt.close()
        return

    plt.figure(figsize=(6, 5))
    clean_x = [x_vals[i] for i in range(len(x_vals)) if regimes[i] == "clean"]
    clean_y = [y_vals[i] for i in range(len(y_vals)) if regimes[i] == "clean"]
    noisy_x = [x_vals[i] for i in range(len(x_vals)) if regimes[i] == "noisy"]
    noisy_y = [y_vals[i] for i in range(len(y_vals)) if regimes[i] == "noisy"]
    if clean_x:
        plt.scatter(clean_x, clean_y, label="clean", alpha=0.7)
    if noisy_x:
        plt.scatter(noisy_x, noisy_y, label="noisy", alpha=0.7, marker="s")
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_scatter_auc_vs_qdrop(joined_rows: list[dict], out_path: str) -> None:
    _scatter_plot(joined_rows, "auc_acc", "qdrop_acc", out_path, "AUC (acc vs alpha) vs PTQ acc drop", skip_null_x=False)


def plot_scatter_slope_vs_qdrop(joined_rows: list[dict], out_path: str) -> None:
    _scatter_plot(joined_rows, "init_slope", "qdrop_acc", out_path, "Init slope vs PTQ acc drop", skip_null_x=False)


def plot_scatter_alpha50_vs_qdrop(joined_rows: list[dict], out_path: str) -> None:
    _scatter_plot(joined_rows, "alpha_50", "qdrop_acc", out_path, "Alpha_50 vs PTQ acc drop", skip_null_x=True)


def produce_all_plots(
    summary_eval_noise_path: str,
    joined_rows: list[dict],
    plots_dir: str,
) -> None:
    """Generate all four plots."""
    plot_noise_curves_by_model_regime(
        summary_eval_noise_path,
        os.path.join(plots_dir, "noise_curves_by_model_regime.png"),
    )
    plot_scatter_auc_vs_qdrop(joined_rows, os.path.join(plots_dir, "scatter_auc_vs_qdrop.png"))
    plot_scatter_slope_vs_qdrop(joined_rows, os.path.join(plots_dir, "scatter_slope_vs_qdrop.png"))
    plot_scatter_alpha50_vs_qdrop(joined_rows, os.path.join(plots_dir, "scatter_alpha50_vs_qdrop.png"))
