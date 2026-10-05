"""Batch prediction: precompute ETAs for popular (zone pair x hour-of-week) (Milestone 6).

    taxieta batch-predict --top-pairs 1000 --reference-month 2019-04

TODO(M6): make the API serve from this table on a hit and fall back to online prediction on a
miss (the hybrid pattern), then measure how stale batch ETAs get during a traffic spike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from taxieta import config, model_store
from taxieta.features.build import build_features
from taxieta.ingest.lake import read_lake


def batch_predict(top_pairs: int = 1000, reference_month: str = "2019-04") -> pd.DataFrame:
    champ = model_store.load_champion()
    if champ is None:
        raise RuntimeError("no champion model")
    y, m = map(int, reference_month.split("-"))
    trips = read_lake([(y, m)], columns=["pu_zone", "do_zone"])
    pairs = trips.value_counts(["pu_zone", "do_zone"]).head(top_pairs).reset_index()[["pu_zone", "do_zone"]]
    # A representative Monday-starting week; hour-of-week 0..167
    week = pd.Timestamp("2019-05-06") + pd.to_timedelta(np.arange(168), "h")
    grid = pairs.merge(pd.DataFrame({"pickup_ts": week}), how="cross")
    X = build_features(grid, champ["hist"], zone_speed=None)   # no live signal in batch mode
    grid["eta_s"] = np.expm1(champ["model"].predict(X, num_iteration=champ["model"].best_iteration))
    grid["hour_of_week"] = grid["pickup_ts"].dt.dayofweek * 24 + grid["pickup_ts"].dt.hour
    grid["model_version"] = champ["version"]
    out = config.FEATURES_DIR / "batch_eta.parquet"
    grid.drop(columns=["pickup_ts"]).to_parquet(out, index=False)
    print(f"wrote {len(grid):,} precomputed ETAs -> {out}")
    return grid
