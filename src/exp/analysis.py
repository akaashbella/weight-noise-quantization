"""Correlation analysis: load CSV, join, Pearson/Spearman (no scipy)."""

from __future__ import annotations

import csv
import os
from typing import Any, Callable, Optional


def load_csv(path: str) -> list[dict[str, Any]]:
    """Load CSV as list of dicts (one per row)."""
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pearson(x: list[float], y: list[float]) -> Optional[float]:
    """Pearson r. Returns None if n<2 or zero variance."""
    n = len(x)
    if n != len(y) or n < 2:
        return None
    xm = sum(x) / n
    ym = sum(y) / n
    sx = sum((xi - xm) ** 2 for xi in x) ** 0.5
    sy = sum((yi - ym) ** 2 for yi in y) ** 0.5
    if sx == 0 or sy == 0:
        return None
    cov = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, y))
    return cov / (sx * sy)


def _rank(vals: list[float]) -> list[float]:
    """Assign average rank for ties. Returns list of ranks (1-based)."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(x: list[float], y: list[float]) -> Optional[float]:
    """Spearman rho = Pearson on ranks. Returns None if n<2 or constant."""
    if len(x) != len(y) or len(x) < 2:
        return None
    rx = _rank(x)
    ry = _rank(y)
    return _pearson(rx, ry)


def inner_join(
    left: list[dict],
    right: list[dict],
    keys: list[str],
) -> list[dict]:
    """Join on keys; merge dicts (right overwrites on conflict)."""
    def key_tuple(row: dict) -> tuple:
        return tuple(str(row.get(k, "")) for k in keys)

    right_by_key: dict[tuple, dict] = {}
    for row in right:
        right_by_key[key_tuple(row)] = row

    out: list[dict] = []
    for row in left:
        k = key_tuple(row)
        if k not in right_by_key:
            continue
        merged = dict(row)
        merged.update(right_by_key[k])
        out.append(merged)
    return out


def _extract_pairs(
    joined: list[dict],
    feat_key: str,
    qdrop_key: str,
    skip_null: bool = False,
) -> tuple[list[float], list[float]]:
    """Get (feat_vals, qdrop_vals) for rows where both present. If skip_null, drop rows where feat is None/empty."""
    x, y = [], []
    for row in joined:
        qd = row.get(qdrop_key)
        f = row.get(feat_key)
        if qd is None or qd == "":
            continue
        if skip_null and (f is None or f == ""):
            continue
        try:
            xv = float(f) if f is not None and f != "" else None
            yv = float(qd)
        except (TypeError, ValueError):
            continue
        if xv is None:
            continue
        x.append(xv)
        y.append(yv)
    return x, y


def run_correlation_analysis(
    joined: list[dict],
    qdrop_key: str = "qdrop_acc",
    features: Optional[list[tuple[str, bool]]] = None,
) -> list[dict]:
    """Compute Pearson and Spearman for each feature vs qdrop. features = [(key, skip_null), ...]. Returns rows for correlation_analysis.csv."""
    if features is None:
        features = [
            ("auc_acc", False),
            ("init_slope", False),
            ("alpha_50", True),
            ("collapse_alpha", True),
        ]
    rows: list[dict] = []
    for scope_name, subset in [
        ("pooled", joined),
        ("clean", [r for r in joined if r.get("train_regime") == "clean"]),
        ("noisy", [r for r in joined if r.get("train_regime") == "noisy"]),
    ]:
        for feat_key, skip_null in features:
            x, y = _extract_pairs(subset, feat_key, qdrop_key, skip_null=skip_null)
            n = len(x)
            if n < 2:
                rows.append({"scope": scope_name, "feature": feat_key, "n": n, "pearson_r": "", "spearman_rho": ""})
                continue
            pr = _pearson(x, y)
            sr = _spearman(x, y)
            rows.append({
                "scope": scope_name,
                "feature": feat_key,
                "n": n,
                "pearson_r": round(pr, 4) if pr is not None else "",
                "spearman_rho": round(sr, 4) if sr is not None else "",
            })
    return rows
