"""Transform + Load: raw monthly Parquet -> validated, partitioned data lake (Milestone 2).

Layout:  data/lake/trips/year=2019/month=1/part-0.parquet
Query it with DuckDB (see taxieta.lake_queries) or pandas.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from taxieta import config
from taxieta.ingest.validate import CLEAN_SCHEMA, reject_bad_rows, standardize


def load_zones(path: Path | None = None) -> pd.DataFrame:
    zones = pd.read_csv(path or config.ZONE_LOOKUP)
    return zones.rename(
        columns={"LocationID": "zone_id", "Borough": "borough", "Zone": "zone", "service_zone": "service_zone"}
    )


def transform(clean: pd.DataFrame, zones: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    df = clean.sort_values("pickup_ts").reset_index(drop=True)
    # Stable, unique id: needed to join trip_started with trip_completed events (natural labels).
    df["trip_id"] = [f"{year}{month:02d}-{i:08d}" for i in range(len(df))]
    df["duration_s"] = (df["dropoff_ts"] - df["pickup_ts"]).dt.total_seconds().astype(float)
    df["pickup_hour"] = df["pickup_ts"].dt.hour.astype("int32")
    df["pickup_dow"] = df["pickup_ts"].dt.dayofweek.astype("int32")
    zmap = zones.set_index("zone_id")["borough"]
    df["pu_borough"] = df["pu_zone"].map(zmap).astype(object)
    df["do_borough"] = df["do_zone"].map(zmap).astype(object)
    # Lineage columns (Ch. 4 "data lineage")
    df["source_file"] = f"yellow_tripdata_{year}-{month:02d}.parquet"
    df["year"] = year
    df["month"] = month
    return df


def ingest_month(
    year: int,
    month: int,
    sample_frac: float | None = None,
    seed: int = 42,
    raw_path: Path | None = None,
) -> dict:
    """Validate + transform + write one month to the lake. Returns the rejection report."""
    config.ensure_dirs()
    raw_path = raw_path or config.RAW_DIR / f"yellow_tripdata_{year}-{month:02d}.parquet"

    # data loading and transforming
    raw = pd.read_parquet(raw_path)
    df = standardize(raw)

    if sample_frac:
        # Simple random sampling. TODO(M3): compare with stratified (by borough) and
        # convenience (first N days) samples -- which one is biased, and how?
        df = df.sample(frac=sample_frac, random_state=seed)

    clean, report = reject_bad_rows(df, year, month)
    out = transform(clean, load_zones(), year, month)
    CLEAN_SCHEMA.validate(out, lazy=True)

    # ingesting data to data lake
    part_dir = config.LAKE_DIR / f"year={year}" / f"month={month}"
    part_dir.mkdir(parents=True, exist_ok=True)
    out.drop(columns=["year", "month"]).to_parquet(part_dir / "part-0.parquet", index=False)

    # making report
    report.update({"year": year, "month": month, "sample_frac": sample_frac})
    rep_dir = config.REPORTS_DIR / "ingest"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / f"{year}-{month:02d}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return report


def read_lake(months: list[tuple[int, int]] | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    """Read lake partitions into pandas. months=None reads everything."""
    import duckdb

    glob = str(config.LAKE_DIR / "*" / "*" / "*.parquet")
    cols = ", ".join(columns) if columns else "*"
    where = ""
    if months:
        conds = " OR ".join(f"(year={y} AND month={m})" for y, m in months)
        where = f"WHERE {conds}"
    q = f"SELECT {cols} FROM read_parquet('{glob}', hive_partitioning=1) {where} ORDER BY pickup_ts"
    return duckdb.sql(q).df()
