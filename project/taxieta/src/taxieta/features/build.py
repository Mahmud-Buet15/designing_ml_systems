"""THE feature definition -- imported by training AND serving (Milestone 4, Ch. 5 & 7).

Contract: features may only use what is known at pickup:
    pickup_ts, pu_zone, do_zone   (+ history computed strictly before the cutoff,
                                   + streaming state as of pickup time)
Anything measured at dropoff (dropoff_ts, trip_distance, fare, tip...) is a LABEL LEAK.
tests/test_features.py enforces this contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from taxieta import config

REQUEST_COLUMNS = ["pickup_ts", "pu_zone", "do_zone"]
LEAKY_COLUMNS = {"dropoff_ts", "duration_s", "trip_distance", "fare_amount", "tip_amount", "total_amount"}

CATEGORICAL = ["pu_zone", "do_zone", "pu_borough", "do_borough"]
FEATURE_COLUMNS = [
    "hour", "dow", "is_weekend", "is_rush_hour", "hour_sin", "hour_cos",
    "pu_zone", "do_zone", "pu_borough", "do_borough", "same_zone", "pu_airport", "do_airport",
    "hist_median_s", "hist_n",
    "zone_speed_15m", "zone_speed_missing",
]


@lru_cache(maxsize=1)
def _borough_map() -> dict[int, str]:
    zones = pd.read_csv(config.ZONE_LOOKUP)
    return dict(zip(zones["LocationID"], zones["Borough"], strict=True))


def time_features(ts: pd.Series) -> pd.DataFrame:
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    return pd.DataFrame({
        "hour": hour,
        "dow": dow,
        "is_weekend": (dow >= 5).astype(int),
        "is_rush_hour": hour.isin([7, 8, 16, 17]).astype(int),  # 7-9am, 4-6pm (end-exclusive)
        # Cyclical encoding: 23:00 and 00:00 are neighbours (Ch. 5, fixed positional embeddings)
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
    }, index=ts.index)


def location_features(pu: pd.Series, do: pd.Series) -> pd.DataFrame:
    bmap = _borough_map()
    return pd.DataFrame({
        "pu_zone": pu.astype(int),
        "do_zone": do.astype(int),
        "pu_borough": pu.map(bmap).fillna("Unknown"),
        "do_borough": do.map(bmap).fillna("Unknown"),
        "same_zone": (pu == do).astype(int),
        "pu_airport": pu.isin(config.AIRPORT_ZONES).astype(int),
        "do_airport": do.isin(config.AIRPORT_ZONES).astype(int),
    }, index=pu.index)
    # TODO(M4): add straight-line distance between zone centroids (TLC publishes zone shapes),
    # and a hashed (pu_zone x do_zone) feature cross -- sklearn.feature_extraction.FeatureHasher.


@dataclass
class HistoricalStats:
    """Median duration per (pu, do, hour-of-week) with back-off to (pu, do) and global.

    LEAKAGE WARNING (Milestone 4, leakage lab #4): fit ONLY on data strictly before the
    period you evaluate on. Fitting on data that includes the target period leaks labels.
    """
    min_count: int = 5
    cutoff: pd.Timestamp | None = None
    _by_pair_how: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    _by_pair: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    _global: float = float("nan")

    def fit(self, trips: pd.DataFrame) -> HistoricalStats:
        if self.cutoff is not None:
            trips = trips[trips["pickup_ts"] < self.cutoff]
        t = trips.assign(how=trips["pickup_ts"].dt.dayofweek * 24 + trips["pickup_ts"].dt.hour)
        g1 = t.groupby(["pu_zone", "do_zone", "how"])["duration_s"].agg(["median", "size"])
        self._by_pair_how = g1[g1["size"] >= self.min_count]
        g2 = t.groupby(["pu_zone", "do_zone"])["duration_s"].agg(["median", "size"])
        self._by_pair = g2[g2["size"] >= self.min_count]
        self._global = float(t["duration_s"].median())
        return self

    def transform(self, req: pd.DataFrame) -> pd.DataFrame:
        how = req["pickup_ts"].dt.dayofweek * 24 + req["pickup_ts"].dt.hour
        k3 = pd.MultiIndex.from_arrays([req["pu_zone"], req["do_zone"], how])
        k2 = pd.MultiIndex.from_arrays([req["pu_zone"], req["do_zone"]])
        m3 = self._by_pair_how.reindex(k3)
        m2 = self._by_pair.reindex(k2)
        med = m3["median"].to_numpy()
        n = m3["size"].to_numpy()
        fallback = np.isnan(med)
        med = np.where(fallback, m2["median"].to_numpy(), med)
        n = np.where(fallback, m2["size"].to_numpy(), n)
        med = np.where(np.isnan(med), self._global, med)
        return pd.DataFrame({"hist_median_s": med, "hist_n": np.nan_to_num(n)}, index=req.index)


ZoneSpeedLookup = Callable[[int, pd.Timestamp], float | None]


def build_features(
    req: pd.DataFrame,
    hist: HistoricalStats,
    zone_speed: pd.Series | ZoneSpeedLookup | None = None,
) -> pd.DataFrame:
    """req must contain REQUEST_COLUMNS. `zone_speed` is either a precomputed Series aligned
    with req (training: computed point-in-time) or a lookup function (serving: online store)."""
    missing = set(REQUEST_COLUMNS) - set(req.columns)
    if missing:
        raise ValueError(f"missing request columns: {missing}")
    req = req[REQUEST_COLUMNS].copy()   # drop everything else: the leak firewall
    req["pickup_ts"] = pd.to_datetime(req["pickup_ts"])

    if zone_speed is None:
        zs = pd.Series(np.nan, index=req.index)
    elif callable(zone_speed):
        zs = pd.Series([zone_speed(int(z), t) for z, t in zip(req["pu_zone"], req["pickup_ts"], strict=True)],
                       index=req.index, dtype=float)
    else:
        zs = zone_speed.reindex(req.index).astype(float)

    X = pd.concat([
        time_features(req["pickup_ts"]),
        location_features(req["pu_zone"], req["do_zone"]),
        hist.transform(req),
        pd.DataFrame({"zone_speed_15m": zs, "zone_speed_missing": zs.isna().astype(int)}),
    ], axis=1)
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X[FEATURE_COLUMNS]


def point_in_time_zone_speed(trips: pd.DataFrame, window: str = "15min") -> pd.Series:
    """For each trip, the median speed of trips from the same pickup zone that COMPLETED in
    the `window` before this trip's pickup. Mirrors stream/consumer.py for training.

    TODO(M4): verify this matches the streaming consumer exactly on one day (M2 task C).
    """
    done = trips[["pu_zone", "dropoff_ts", "trip_distance", "duration_s"]].copy()
    done = done[(done["duration_s"] > 60) & (done["trip_distance"] > 0)]
    done["mph"] = done["trip_distance"] / (done["duration_s"] / 3600)
    out = pd.Series(np.nan, index=trips.index)
    w = pd.Timedelta(window)
    for zone, grp in trips.groupby("pu_zone"):
        d = done[done["pu_zone"] == zone].sort_values("dropoff_ts")
        if d.empty:
            continue
        ends = d["dropoff_ts"].to_numpy()
        mph = d["mph"].to_numpy()
        starts = grp["pickup_ts"].to_numpy()
        hi = np.searchsorted(ends, starts, side="left")          # completed strictly before pickup
        lo = np.searchsorted(ends, starts - w, side="left")
        vals = [np.median(mph[a:b]) if b > a else np.nan for a, b in zip(lo, hi, strict=True)]
        out.loc[grp.index] = vals
    return out
