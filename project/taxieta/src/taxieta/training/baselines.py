"""Baselines (Milestone 1 heuristic, Milestone 5 baselines table, Milestone 10 fallback).

Every model must beat these by enough to justify its complexity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from taxieta.features.build import HistoricalStats


class ZeroRule:
    """Always predict the training median."""

    def fit(self, y: pd.Series) -> ZeroRule:
        self.value = float(np.median(y))
        return self

    def predict(self, n: int) -> np.ndarray:
        return np.full(n, self.value)


class RandomBaseline:
    """Sample predictions from the training label distribution."""

    def fit(self, y: pd.Series, seed: int = 0) -> RandomBaseline:
        self.values = np.asarray(y)
        self.rng = np.random.default_rng(seed)
        return self

    def predict(self, n: int) -> np.ndarray:
        return self.rng.choice(self.values, size=n)


class HeuristicETA:
    """Phase 1 (no ML): median duration for (pickup zone, dropoff zone, hour-of-week),
    backing off to (pickup, dropoff), then the global median. Also the serving fallback."""

    def fit(self, trips: pd.DataFrame) -> HeuristicETA:
        self.stats = HistoricalStats(min_count=3).fit(trips)
        return self

    def predict(self, req: pd.DataFrame) -> np.ndarray:
        req = req.assign(pickup_ts=pd.to_datetime(req["pickup_ts"]))
        return self.stats.transform(req)["hist_median_s"].to_numpy()
