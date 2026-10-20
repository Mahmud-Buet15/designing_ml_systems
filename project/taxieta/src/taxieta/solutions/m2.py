"""Milestone 2 solution: formats, iteration order, missingness, batch vs streaming.

    taxieta solve m2 [--month 2019-01] [--day 2019-01-15]

A. CSV vs Parquet: size, full read, 3-column read (row-major vs column-major, text vs binary)
B. pandas iteration by row vs by column vs NumPy (pandas is column-major)
C. Missingness of passenger_count by vendor and month (MCAR vs MAR)
D. The zone-speed feature computed in BATCH vs replayed through the STREAMING logic, one day
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

import numpy as np
import pandas as pd

from taxieta import config
from taxieta.solutions.common import Report
from taxieta.stream.consumer import batch_zone_speed, replay_zone_speed


def _formats(raw_path, rep: Report) -> None:
    """
    A. CSV vs Parquet: size, full read, 3-column read (row-major vs column-major, text vs binary)
    """
    raw = pd.read_parquet(raw_path)  # read parquet file
    tmp = tempfile.mkdtemp()  # create a temporary directory
    csv_p, pq_p = os.path.join(tmp, "m.csv"), os.path.join(tmp, "m.parquet")  # specify the file path to read from the data lake
    raw.to_csv(csv_p, index=False)      # dump as csv
    raw.to_parquet(pq_p, index=False)   # dump as parquet
    cols = ["tpep_pickup_datetime", "PULocationID", "DOLocationID"]     # specific 3 columns to read

    # getting time needed for different operations
    t = {}
    for name, fn in {
        "csv_all": lambda: pd.read_csv(csv_p),
        "parquet_all": lambda: pd.read_parquet(pq_p),
        "csv_3cols": lambda: pd.read_csv(csv_p, usecols=cols),
        "parquet_3cols": lambda: pd.read_parquet(pq_p, columns=cols),
    }.items():
        t0 = time.perf_counter()
        fn()  # read the file
        t[name] = time.perf_counter() - t0   # calculate the time taken
    
    # generating a dataframe with stats
    df = pd.DataFrame([
        {"format": "CSV (text, row-major)", "size_mb": os.path.getsize(csv_p) / 1e6,     # getting size of csv file in MB
         "read_all_s": t["csv_all"], "read_3_cols_s": t["csv_3cols"]},  # times needed
        {"format": "Parquet (binary, column-major)", "size_mb": os.path.getsize(pq_p) / 1e6,  # getting size of parquet file in MB
         "read_all_s": t["parquet_all"], "read_3_cols_s": t["parquet_3cols"]},
    ])
    rep.h(f"A. Formats ({len(raw):,} rows)")
    rep.table(df, key="formats")
    rep.p("CSV still has to parse every byte of every row to extract 3 columns. Parquet reads only "
          "those column chunks, and stores compressed, typed binary.")
    # Precision loss in CSV round trips (the book's footnote)
    x = pd.DataFrame({"v": [0.12345678901232323]})
    x.to_csv(csv_p, index=False, float_format="%.10g")
    rep.p(f"CSV float round trip with a formatter: {float(pd.read_csv(csv_p)['v'][0])!r} vs 0.12345678901232323")


def _iteration(trips: pd.DataFrame, rep: Report, n: int = 20_000) -> None:
    """B. Iterate over n rows to compare different iteration methods."""
    df = trips[["duration_s", "pu_zone", "do_zone", "pickup_hour"]].head(n).astype(float)
    t = {}
    t0 = time.perf_counter()
    for _, row in df.iterrows():    # iterating dataframe over rows
        _ = row["duration_s"]   # dummy operation
    t["pandas_by_row (iterrows)"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for col in df.columns:      # iterating dataframe over columns
        _ = df[col].sum()      # dummy operation
    t["pandas_by_column"] = time.perf_counter() - t0

    arr = df.to_numpy()      # Converting to numpy. C-order = row-major
    t0 = time.perf_counter()
    for row in arr:
        _ = row[0]
    t["numpy_by_row"] = time.perf_counter() - t0

    # generating a dataframe with stats
    rep.h(f"B. Iteration over {n:,} rows")
    rep.table(pd.DataFrame([{"method": k, "seconds": v} for k, v in t.items()]), key="iteration")
    rep.p("iterrows builds a Series per row from column-major storage. Use vectorized column ops, "
          "or convert to NumPy when you truly need rows.")


def _missingness(raw_path, rep: Report) -> None:
    """C. Passenger count missingness by vendor and month (MCAR vs MAR)."""
    raw = pd.read_parquet(raw_path, columns=["VendorID", "passenger_count", "tpep_pickup_datetime",
                                             "trip_distance"])
    raw["pc_missing_or_zero"] = raw["passenger_count"].isna() | (raw["passenger_count"] == 0)   # checking if the passenger count is missing or zero
    by_vendor = raw.groupby("VendorID")["pc_missing_or_zero"].mean().mul(100).rename("pct").reset_index()  # getting the percentage of missing or zero passenger count by vendor
    raw["hour"] = pd.to_datetime(raw["tpep_pickup_datetime"]).dt.hour
    by_hour = raw.groupby("hour")["pc_missing_or_zero"].mean().mul(100)  # getting the percentage of missing or zero passenger count by hour

    # generating a dataframe with stats
    rep.h("C. passenger_count missing-or-zero rate")
    rep.table(by_vendor, key="missing_by_vendor")
    rep.p(f"by pickup hour: min {by_hour.min():.2f}% / max {by_hour.max():.2f}%")
    dist_missing = raw.loc[raw["pc_missing_or_zero"], "trip_distance"].median()
    dist_present = raw.loc[~raw["pc_missing_or_zero"], "trip_distance"].median()
    rep.p(f"median trip_distance when missing {dist_missing:.2f} vs present {dist_present:.2f}")
    rep.p("Reading it: a rate that differs a lot by vendor/month means MAR (explained by another "
          "observed variable). Flat rates everywhere means MCAR. If missing trips look different "
          "(distance, fare), deleting them would bias the data.")


def _batch_vs_stream(trips: pd.DataFrame, day: str, rep: Report) -> None:
    """D. Batch vs streaming zone speed on a given day."""
    d0 = pd.Timestamp(day)
    # include the previous 15 min of completions so the window is warm at midnight
    one = trips[(trips["pickup_ts"] >= d0 - pd.Timedelta("2h")) & (trips["pickup_ts"] < d0 + pd.Timedelta("1D"))]   # taking a window of 2 hours before the given day and 1 day after the given day
    one = one.reset_index(drop=True)    # resetting the index
    batch = batch_zone_speed(one)       # computing batch zone speed
    stream = replay_zone_speed(one)     # computing streaming zone speed
    mask = one["pickup_ts"] >= d0       # checking if the pickup time is greater than or equal to the given day
    b, s = batch[mask], stream[mask]    # getting the batch and streaming zone speed for the given day
    both_nan = b.isna() & s.isna()      # check if both batch and streaming zone speed are NaN
    equal = both_nan | np.isclose(b.fillna(-1), s.fillna(-1), rtol=1e-9, atol=1e-9) # check if the batch and streaming zone speed are equal
    res = {"trips": int(mask.sum()), "match_rate_pct": float(equal.mean() * 100), # calculating the match rate percentage
           "missing_rate_pct": float(b.isna().mean() * 100)} # calculating the missing rate percentage
    rep.h(f"D. Batch vs streaming zone speed on {day}")
    rep.table(pd.DataFrame([res]), key="batch_vs_stream")

    # if batch and streaming zone speed are not equal, then show the differences
    if res["match_rate_pct"] < 100:
        diff = one[mask][~equal].head(5).assign(batch=b[~equal].head(5), stream=s[~equal].head(5))
        rep.table(diff[["trip_id", "pickup_ts", "pu_zone", "batch", "stream"]])
    rep.p("They match because both use ONE definition: window [t-15min, t), same junk filter, "
          "event-time ordering (a completion at exactly the pickup second is not yet visible). "
          "Change any one of those in only one path and the match rate drops: that's train/serve skew.")


def main(argv: list[str] | None = None) -> dict:
    # creating parser for sub command: solve m2
    ap = argparse.ArgumentParser(prog="taxieta solve m2")   # creating an object of ArgumentParser with the program name as "taxieta solve m2"
    ap.add_argument("--month", default="2019-01")  # optional argument
    ap.add_argument("--day", default=None, help="day for the batch-vs-stream check (default: 15th)")  # optional argument
    
    # parse arguments
    a = ap.parse_args(argv)
    y, m = map(int, a.month.split("-")) #split month string and convert to int
    raw_path = config.RAW_DIR / f"yellow_tripdata_{y}-{m:02d}.parquet"
    from taxieta.ingest.lake import read_lake

    trips = read_lake([(y, m)]) # get the trips data as dataframe
    rep = Report("m2", "Milestone 2 -- data engineering")
    _formats(raw_path, rep)  # A. CSV vs Parquet: size, full read, 3-column read (row-major vs column-major, text vs binary)
    _iteration(trips, rep)   # B. pandas iteration by row vs by column vs NumPy (pandas is column-major)
    _missingness(raw_path, rep)  # C. Missingness of passenger_count by vendor and month (MCAR vs MAR)
    _batch_vs_stream(trips, a.day or f"{y}-{m:02d}-15", rep)   # D. The zone-speed feature computed in BATCH vs replayed through the STREAMING logic, one day
    rep.save()  # save the report
    return rep.data   # return the report data
