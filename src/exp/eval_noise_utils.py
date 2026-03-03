"""Feature extraction for accuracy vs alpha curves (no numpy)."""

from __future__ import annotations


def compute_auc(x: list[float], y: list[float]) -> float:
    """Trapezoidal rule AUC; x and y must be same length, x ascending."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    total = 0.0
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        total += 0.5 * (y[i] + y[i + 1]) * dx
    return total


def compute_features(
    alpha_grid: list[float],
    acc_means: list[float],
    collapse_threshold: float = 0.1,
) -> dict:
    """Compute scalar curve features. acc_means in percent (0-100). collapse_threshold as fraction (0.1 = 10%)."""
    if len(alpha_grid) != len(acc_means):
        raise ValueError("alpha_grid and acc_means length mismatch")
    if not alpha_grid:
        raise ValueError("alpha_grid must not be empty")
    # acc at alpha=0 must exist
    idx0 = None
    for i, a in enumerate(alpha_grid):
        if a == 0.0:
            idx0 = i
            break
    if idx0 is None:
        raise ValueError("alpha_grid must contain 0.0 for acc0 and alpha_50")
    acc0 = acc_means[idx0]

    # AUC of accuracy vs alpha (over sorted grid)
    auc_acc = compute_auc(alpha_grid, acc_means)

    # init_slope: (acc(alpha2) - acc(alpha1)) / (alpha2 - alpha1), alpha1=0, alpha2=0.01 or smallest nonzero
    alpha_nonzero = [a for a in alpha_grid if a > 0]
    if not alpha_nonzero:
        init_slope = 0.0
    else:
        alpha2 = min(alpha_nonzero)
        idx2 = alpha_grid.index(alpha2)
        acc2 = acc_means[idx2]
        init_slope = (acc2 - acc0) / alpha2 if alpha2 > 0 else 0.0

    # alpha_50: smallest alpha where acc <= 0.5 * acc0
    half_acc = 0.5 * acc0
    alpha_50 = None
    for i, a in enumerate(alpha_grid):
        if acc_means[i] <= half_acc:
            alpha_50 = a
            break

    # collapse_alpha: first alpha where acc <= collapse_threshold (fraction -> percent)
    acc_collapse = collapse_threshold * 100.0
    collapse_alpha = None
    for i, a in enumerate(alpha_grid):
        if acc_means[i] <= acc_collapse:
            collapse_alpha = a
            break

    return {
        "acc0": acc0,
        "auc_acc": auc_acc,
        "init_slope": init_slope,
        "alpha_50": alpha_50,
        "collapse_alpha": collapse_alpha,
        "collapse_threshold": collapse_threshold,
    }
