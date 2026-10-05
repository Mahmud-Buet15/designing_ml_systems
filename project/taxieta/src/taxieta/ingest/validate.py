"""Extract-step validation (Milestone 2, Ch. 3 "ETL") and data tests (Ch. 8 "feature validation").

Two layers:
1. `reject_bad_rows` -- row-level rules. Bad rows are dropped *and counted by reason*, so a
   spike in any reason becomes a monitoring signal later (Milestone 7, "raw inputs").
2. `CLEAN_SCHEMA` -- a pandera schema the cleaned data must satisfy ("unit tests for data").
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from taxieta import config

RAW_COLUMNS = {
    "tpep_pickup_datetime": "pickup_ts",
    "tpep_dropoff_datetime": "dropoff_ts",
    "PULocationID": "pu_zone",
    "DOLocationID": "do_zone",
    "passenger_count": "passenger_count",
    "trip_distance": "trip_distance",
    "fare_amount": "fare_amount",
    "tip_amount": "tip_amount",
    "total_amount": "total_amount",
    "payment_type": "payment_type",
    "VendorID": "vendor_id",
}


def standardize(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename TLC columns to short snake_case names; keep only what we use."""
    df = raw.rename(columns=RAW_COLUMNS)[list(RAW_COLUMNS.values())].copy()
    # Normalize to naive datetime64[ns] (Parquet may carry us precision).
    df["pickup_ts"] = pd.to_datetime(df["pickup_ts"]).astype("datetime64[ns]")
    df["dropoff_ts"] = pd.to_datetime(df["dropoff_ts"]).astype("datetime64[ns]")
    df["pu_zone"] = df["pu_zone"].astype("int32")
    df["do_zone"] = df["do_zone"].astype("int32")
    return df


def reject_bad_rows(
    df: pd.DataFrame, year: int, month: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return (clean rows, {reason: rejected_count}). A row is counted under its first failing rule."""
    duration = (df["dropoff_ts"] - df["pickup_ts"]).dt.total_seconds()
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_end = month_start + pd.offsets.MonthBegin(1)

    rules: dict[str, pd.Series] = {
        "pickup_outside_month": (df["pickup_ts"] < month_start) | (df["pickup_ts"] >= month_end),
        "unknown_zone": df["pu_zone"].isin(config.UNKNOWN_ZONES) | df["do_zone"].isin(config.UNKNOWN_ZONES),
        "dropoff_before_pickup": duration < 0,
        "zero_duration": duration == 0,
        "duration_over_6h": duration > config.MAX_DURATION_S,
        # TODO(M2): is dropping passenger_count == 0 / null the right call? Classify the
        # missingness (MCAR/MAR/MNAR, Ch. 5) before deciding. For now we only count it.
    }
    report: dict[str, int] = {}
    bad = pd.Series(False, index=df.index)
    for reason, mask in rules.items():
        newly_bad = mask & ~bad
        report[reason] = int(newly_bad.sum())
        bad |= mask
    report["passenger_count_missing_or_zero(kept)"] = int(
        (df["passenger_count"].isna() | (df["passenger_count"] == 0)).sum()
    )
    report["rows_in"] = len(df)
    report["rows_out"] = int((~bad).sum())
    return df.loc[~bad].copy(), report


CLEAN_SCHEMA = pa.DataFrameSchema(
    {
        "trip_id": pa.Column(str, unique=True),
        "pickup_ts": pa.Column("datetime64[ns]"),
        "dropoff_ts": pa.Column("datetime64[ns]"),
        "pu_zone": pa.Column("int32", pa.Check.in_range(1, 263)),
        "do_zone": pa.Column("int32", pa.Check.in_range(1, 263)),
        "duration_s": pa.Column(float, pa.Check.in_range(1, config.MAX_DURATION_S)),
        "pickup_hour": pa.Column("int32", pa.Check.in_range(0, 23)),
        "pickup_dow": pa.Column("int32", pa.Check.in_range(0, 6)),
        "pu_borough": pa.Column(str, nullable=True),
        "do_borough": pa.Column(str, nullable=True),
    },
    checks=[pa.Check(lambda d: (d["dropoff_ts"] > d["pickup_ts"]).all(), error="dropoff<=pickup")],
    strict=False,
    coerce=False,
)
