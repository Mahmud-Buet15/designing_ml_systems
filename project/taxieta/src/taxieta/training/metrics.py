"""Metrics that mean something to the business (Milestones 1 and 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae_min(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / 60)


def pct_within(y_true, y_pred, tol: float = 0.20) -> float:
    """% of trips whose prediction is within +/- tol of the actual duration."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_pred - y_true) <= tol * y_true) * 100)


def report(y_true, y_pred) -> dict[str, float]:
    return {"mae_min": mae_min(y_true, y_pred), "pct_within_20": pct_within(y_true, y_pred)}


def slice_report(df: pd.DataFrame, by: str, y_true: str = "duration_s", y_pred: str = "pred_s",
                 min_n: int = 30) -> pd.DataFrame:
    """Slice-based evaluation (Ch. 6). Sorted worst-first."""
    rows = []
    for key, g in df.groupby(by, observed=True):
        if len(g) < min_n:
            continue
        rows.append({by: key, "n": len(g), **report(g[y_true], g[y_pred])})
    return pd.DataFrame(rows).sort_values("mae_min", ascending=False).reset_index(drop=True)
