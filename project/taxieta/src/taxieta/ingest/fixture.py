"""Synthetic TLC-shaped data for tests, CI, and offline development.

The real data lives at the TLC CloudFront bucket. When you can't download it (CI, a plane,
a locked-down network), `taxieta fixture` writes small files with the *same schema* as the
real ones into data/raw/, so every downstream command works unchanged.

It is deliberately imperfect: it injects the same kinds of bad rows the real data has
(negative durations, unknown zones, out-of-month timestamps), and `shift=True` mimics a
spring-2020-style distribution shift (fewer trips, faster traffic, more outer-borough trips).

Never draw modelling conclusions from fixture data -- it's a plumbing test, not a dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from taxieta import config

# A small zone table in the real lookup's format. IDs match the real ones for airports.
_ZONES = [
    (1, "EWR", "Newark Airport", "EWR"),
    (4, "Manhattan", "Alphabet City", "Yellow Zone"),
    (13, "Manhattan", "Battery Park City", "Yellow Zone"),
    (43, "Manhattan", "Central Park", "Yellow Zone"),
    (48, "Manhattan", "Clinton East", "Yellow Zone"),
    (68, "Manhattan", "East Chelsea", "Yellow Zone"),
    (79, "Manhattan", "East Village", "Yellow Zone"),
    (87, "Manhattan", "Financial District North", "Yellow Zone"),
    (100, "Manhattan", "Garment District", "Yellow Zone"),
    (107, "Manhattan", "Gramercy", "Yellow Zone"),
    (132, "Queens", "JFK Airport", "Airports"),
    (138, "Queens", "LaGuardia Airport", "Airports"),
    (142, "Manhattan", "Lincoln Square East", "Yellow Zone"),
    (161, "Manhattan", "Midtown Center", "Yellow Zone"),
    (162, "Manhattan", "Midtown East", "Yellow Zone"),
    (170, "Manhattan", "Murray Hill", "Yellow Zone"),
    (186, "Manhattan", "Penn Station/Madison Sq West", "Yellow Zone"),
    (230, "Manhattan", "Times Sq/Theatre District", "Yellow Zone"),
    (236, "Manhattan", "Upper East Side North", "Yellow Zone"),
    (237, "Manhattan", "Upper East Side South", "Yellow Zone"),
    (7, "Queens", "Astoria", "Boro Zone"),
    (129, "Queens", "Jackson Heights", "Boro Zone"),
    (181, "Brooklyn", "Park Slope", "Boro Zone"),
    (255, "Brooklyn", "Williamsburg (North Side)", "Boro Zone"),
    (61, "Brooklyn", "Crown Heights North", "Boro Zone"),
    (74, "Manhattan", "East Harlem North", "Boro Zone"),
    (168, "Bronx", "Mott Haven/Port Morris", "Boro Zone"),
    (69, "Bronx", "East Concourse/Concourse Village", "Boro Zone"),
    (5, "Staten Island", "Arden Heights", "Boro Zone"),
    (264, "Unknown", "NV", "N/A"),
    (265, "N/A", "Outside of NYC", "N/A"),
]


def zone_lookup() -> pd.DataFrame:
    return pd.DataFrame(_ZONES, columns=["LocationID", "Borough", "Zone", "service_zone"])


def _zone_coords(rng: np.random.Generator) -> dict[int, tuple[float, float]]:
    centers = {"Manhattan": (0, 0), "Queens": (8, 2), "Brooklyn": (3, -6), "Bronx": (2, 8),
               "Staten Island": (-6, -12), "EWR": (-12, -4), "Unknown": (0, 0), "N/A": (0, 0)}
    coords = {}
    for zid, boro, _, _ in _ZONES:
        cx, cy = centers[boro]
        coords[zid] = (cx + rng.normal(0, 1.5), cy + rng.normal(0, 1.5))
    coords[132] = (14, -2)   # JFK far out
    coords[138] = (7, 4)     # LGA
    return coords


def make_month(year: int, month: int, n: int = 20_000, shift: bool = False, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed + year * 100 + month)
    coords = _zone_coords(np.random.default_rng(7))  # fixed geography across months
    real = [z for z in _ZONES if z[0] not in config.UNKNOWN_ZONES]
    ids = np.array([z[0] for z in real])
    boro = np.array([z[1] for z in real])
    w = np.where(boro == "Manhattan", 10.0, 0.6)
    w = np.where(np.isin(ids, list(config.AIRPORT_ZONES)), 1.0, w)
    if shift:
        n = max(n // 8, 500)
        w = np.where(boro == "Manhattan", 3.0, w * 1.8)      # covariate shift
    w = w / w.sum()

    start = pd.Timestamp(year=year, month=month, day=1)
    days = (start + pd.offsets.MonthBegin(1) - start).days
    # Pickup times with a daily profile (busier 7-10 and 16-20)
    hours = rng.choice(24, size=n, p=_hour_profile())
    pickup = start + pd.to_timedelta(rng.integers(0, days, n), "D") + pd.to_timedelta(hours, "h") \
        + pd.to_timedelta(rng.integers(0, 3600, n), "s")
    pu = rng.choice(ids, size=n, p=w)
    do = rng.choice(ids, size=n, p=w)
    dist = np.array([np.hypot(*(np.subtract(coords[a], coords[b]))) for a, b in zip(pu, do, strict=True)]) + 0.5
    rush = np.isin(hours, [7, 8, 9, 16, 17, 18]).astype(float)
    # Longer trips use highways -> faster average speed; rush hour is slower.
    speed_mph = (9 - 3 * rush) * (1 + dist / 6) * (1.6 if shift else 1.0) * rng.lognormal(0, 0.25, n)  # concept drift
    duration_s = dist / speed_mph * 3600 + rng.normal(90, 45, n)
    jam = rng.random(n) < 0.03                      # rare heavy delays -> realistic long tail
    duration_s = np.clip(np.where(jam, duration_s * rng.uniform(2, 4.5, n), duration_s), 60, 5 * 3600)
    dropoff = pickup + pd.to_timedelta(np.round(duration_s), "s")
    metered = dist * rng.lognormal(0.05, 0.1, n)
    fare = 3 + 2.5 * metered + 0.5 * duration_s / 60

    df = pd.DataFrame({
        "VendorID": rng.choice([1, 2], n),
        "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": dropoff,
        "passenger_count": rng.choice([1.0, 1.0, 1.0, 2.0, 3.0, 0.0, np.nan], n),
        "trip_distance": metered.round(2),
        "RatecodeID": 1.0,
        "store_and_fwd_flag": "N",
        "PULocationID": pu.astype("int64"),
        "DOLocationID": do.astype("int64"),
        "payment_type": rng.choice([1, 2], n),
        "fare_amount": fare.round(2),
        "extra": 0.5,
        "mta_tax": 0.5,
        "tip_amount": (fare * rng.choice([0, 0.15, 0.2], n)).round(2),
        "tolls_amount": 0.0,
        "improvement_surcharge": 0.3,
        "total_amount": (fare * 1.2).round(2),
        "congestion_surcharge": 2.5,
    })
    # Inject realistic garbage (~1.5%)
    k = max(n // 200, 1)
    idx = rng.choice(n, 3 * k, replace=False)
    df.loc[idx[:k], "tpep_dropoff_datetime"] = df.loc[idx[:k], "tpep_pickup_datetime"] - pd.Timedelta("5min")
    df.loc[idx[k:2 * k], "PULocationID"] = 264
    df.loc[idx[2 * k:], "tpep_pickup_datetime"] = start - pd.Timedelta("2D")
    return df


def _hour_profile() -> np.ndarray:
    p = np.array([2, 1.5, 1, 0.7, 0.6, 0.8, 2, 4, 5, 5, 4.5, 4.5, 4.8, 4.8, 5, 5.2, 5.5, 6, 6.2, 5.8, 5, 4.5, 4, 3])
    return p / p.sum()


def write_fixture(months: list[tuple[int, int]], n: int = 20_000) -> None:
    config.ensure_dirs()
    zone_lookup().to_csv(config.ZONE_LOOKUP, index=False)
    for y, m in months:
        shift = y >= 2020 and m >= 3
        path = config.RAW_DIR / f"yellow_tripdata_{y}-{m:02d}.parquet"
        make_month(y, m, n=n, shift=shift).to_parquet(path, index=False)
        print(f"fixture {path.name} ({'shifted' if shift else 'baseline'})")
