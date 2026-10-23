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
ZONE_IDS = list(range(1, 266))
BOROUGHS = ["Bronx", "Brooklyn", "EWR", "Manhattan", "Queens", "Staten Island", "Unknown", "N/A"]
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
        """Fit the historical stats on the given trips."""
        if self.cutoff is not None:
            trips = trips[trips["pickup_ts"] < self.cutoff]
        t = trips.assign(how=trips["pickup_ts"].dt.dayofweek * 24 + trips["pickup_ts"].dt.hour) # combines day-of-week (0–6) and hour (0–23) into a single integer feature in the range 0–167, representing each of the 168 hours in a week. This makes the feature suitable for use in machine learning models that expect numerical inputs
        g1 = t.groupby(["pu_zone", "do_zone", "how"])["duration_s"].agg(["median", "size"])     # getting stats: median and count
        self._by_pair_how = g1[g1["size"] >= self.min_count]    # exact match: same pickup zone, same dropoff zone, same hour-of-week. If fewer than min_count observations exist, this group is excluded.
        g2 = t.groupby(["pu_zone", "do_zone"])["duration_s"].agg(["median", "size"])     # fallback  stats
        self._by_pair = g2[g2["size"] >= self.min_count]
        self._global = float(t["duration_s"].median())  # global median for the entire dataset
        return self

    def transform(self, req: pd.DataFrame) -> pd.DataFrame:
        """Transform the request DataFrame by adding historical features."""
        how = req["pickup_ts"].dt.dayofweek * 24 + req["pickup_ts"].dt.hour
        k3 = pd.MultiIndex.from_arrays([req["pu_zone"], req["do_zone"], how])   # creating a multi-index of (pickup_zone, dropoff_zone, hour_of_week)
        k2 = pd.MultiIndex.from_arrays([req["pu_zone"], req["do_zone"]])      # creating a multi-index of (pickup_zone, dropoff_zone)
        m3 = self._by_pair_how.reindex(k3)   # reindexing the multi-index of (pickup_zone, dropoff_zone, hour_of_week)
        m2 = self._by_pair.reindex(k2)     # reindexing the multi-index of (pickup_zone, dropoff_zone)
        med = m3["median"].to_numpy()      # getting the median of the reindexed multi-index
        n = m3["size"].to_numpy()         # getting the count of the reindexed multi-index
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

    if zone_speed is None:  # if no zone speed data is provided, then set it to NaN
        zs = pd.Series(np.nan, index=req.index)
    elif callable(zone_speed):  # if zone speed is a function, then calculate it
        zs = pd.Series([zone_speed(int(z), t) for z, t in zip(req["pu_zone"], req["pickup_ts"], strict=True)],
                       index=req.index, dtype=float)
    else:  # if zone speed is a series, then reindex it to the request index
        zs = zone_speed.reindex(req.index).astype(float)

    X = pd.concat([
        time_features(req["pickup_ts"]),
        location_features(req["pu_zone"], req["do_zone"]),
        hist.transform(req),
        pd.DataFrame({"zone_speed_15m": zs, "zone_speed_missing": zs.isna().astype(int)}),
    ], axis=1)
    # SOLVED(M3/M4): a FIXED category vocabulary. Per-frame `.astype("category")` gives different
    # integer codes in different frames (train vs valid vs a single serving request), which is a
    # silent train/serve skew bug for any model that consumes codes.
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=ZONE_IDS if c.endswith("zone") else BOROUGHS)
    return X[FEATURE_COLUMNS]


def point_in_time_zone_speed(trips: pd.DataFrame, window: str = "15min") -> pd.Series:
    """For each trip, the median speed of trips from the same pickup zone that COMPLETED in
    the `window` before this trip's pickup. Mirrors stream/consumer.py for training.

    TODO(M4): verify this matches the streaming consumer exactly on one day (M2 task C).
    """
    # Step 1: Prepare the completed trips
    done = trips[["pu_zone", "dropoff_ts", "trip_distance", "duration_s"]].copy()   # taking columns for done/completed trips
    done = done[(done["duration_s"] > 60) & (done["trip_distance"] > 0)]      # filtering out trips with duration less than or equal to 60 seconds and distance less than or equal to 0. Otherwise it will impact the stats
    done["mph"] = done["trip_distance"] / (done["duration_s"] / 3600)     # calculating the speed in mph

    # Step 2: Set up the output
    out = pd.Series(np.nan, index=trips.index)       # out has one slot per trip, all starting as NaN. Trips with no eligible recent trips stay NaN
    w = pd.Timedelta(window)    # pd.Timedelta("15min") turns the string into a time duration, so it can be subtracted from timestamps.

    # Step 3: Process each zone
    for zone, grp in trips.groupby("pu_zone"):     # For each pickup zone: grp holds the trips in that zone that need the feature.
        d = done[done["pu_zone"] == zone].sort_values("dropoff_ts")     # d holds that zone's completed trips, sorted by drop-off time. The sort is required for the binary search below.
        if d.empty:         # If the zone has no valid completed trips, it's skipped and its trips keep NaN.
            continue

        # Step 4: Find each trip's window with binary search
        ends = d["dropoff_ts"].to_numpy()   
        mph = d["mph"].to_numpy()
        starts = grp["pickup_ts"].to_numpy()        # getting the pickup times
        hi = np.searchsorted(ends, starts, side="left")          # returns indexs of high for all trips in the group. completed strictly before pickup
        lo = np.searchsorted(ends, starts - w, side="left")     # returns index of low for all trips in the group. completed within the window before pickup

        # Step 5: Compute the medians
        vals = [np.median(mph[a:b]) if b > a else np.nan for a, b in zip(lo, hi, strict=True)] # For each trip, take the speeds in its window slice and compute the median
        out.loc[grp.index] = vals       # writing the results back to the correct rows. Check explanations.md
    return out

# Note: Check docs/explanations.md for detailed explanations