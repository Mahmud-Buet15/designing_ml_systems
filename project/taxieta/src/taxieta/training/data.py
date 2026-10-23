"""Leakage-safe train/valid/test feature matrices, shared by all solution experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from taxieta.features.build import HistoricalStats, build_features, point_in_time_zone_speed
from taxieta.ingest.download import parse_months
from taxieta.ingest.lake import read_lake


@dataclass
class Splits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    hist: HistoricalStats

    def y(self, split: str) -> np.ndarray:
        return getattr(self, split)["duration_s"].to_numpy()


def _months(df: pd.DataFrame, months: list[tuple[int, int]]) -> pd.DataFrame:
    """Filters the trips data to include only the specified months.
    
    Args:
        df: DataFrame containing the trips data
        months: List of months to filter by
    
    Returns:
        DataFrame containing the trips data for the specified months
    """
    key = df["pickup_ts"].dt.year * 100 + df["pickup_ts"].dt.month # creates a key for each month by combining year and month
    return df[key.isin([y * 100 + m for y, m in months])].reset_index(drop=True) # filters the trips data to include only the specified months


def prepare_splits(train_spec: str, valid_spec: str, test_spec: str,
                   trips: pd.DataFrame | None = None, feature_fn=build_features) -> Splits:
    tr_m, va_m, te_m = parse_months(train_spec), parse_months(valid_spec), parse_months(test_spec)
    if trips is None:   # checks if trips data is already loaded, if not, then load it from the lake
        trips = read_lake(sorted(set(tr_m + va_m + te_m)))
    if "zone_speed_15m" not in trips: # checks if zone speed data which is derived from trips is already available, if not, then calculate it
        trips = trips.assign(zone_speed_15m=point_in_time_zone_speed(trips))    # gets average speed of trips in the same zone and 15 minute interval
    train, valid, test = (_months(trips, m) for m in (tr_m, va_m, te_m)) # splitting the trips data into train, valid and test sets based on months
    hist = HistoricalStats(cutoff=valid["pickup_ts"].min()).fit(train) # calculating historical stats based on train set
    feats = [feature_fn(d, hist, zone_speed=d["zone_speed_15m"]) for d in (train, valid, test)] # calculating features for train, valid and test sets
    return Splits(train, valid, test, *feats, hist=hist)
