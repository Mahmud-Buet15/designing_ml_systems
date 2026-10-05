"""Time-based splits (Milestone 4). Random splits leak the future for time-correlated data."""

from __future__ import annotations

import pandas as pd


def time_split(df: pd.DataFrame, valid_start: str, test_start: str, test_end: str | None = None,
               ts_col: str = "pickup_ts") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts = df[ts_col]
    train = df[ts < valid_start]
    valid = df[(ts >= valid_start) & (ts < test_start)]
    test = df[ts >= test_start] if test_end is None else df[(ts >= test_start) & (ts < test_end)]
    return train, valid, test


def random_split(df: pd.DataFrame, frac_valid: float = 0.15, frac_test: float = 0.15, seed: int = 0):
    """ONLY for the leakage lab (Milestone 4, experiment #2). Don't use it for real."""
    shuffled = df.sample(frac=1.0, random_state=seed)
    n = len(df)
    n_test, n_valid = int(n * frac_test), int(n * frac_valid)
    return shuffled.iloc[n_test + n_valid:], shuffled.iloc[n_test:n_test + n_valid], shuffled.iloc[:n_test]
