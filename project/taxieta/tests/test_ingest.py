import pandas as pd

from taxieta.ingest.download import parse_months, trip_url
from taxieta.ingest.lake import read_lake
from taxieta.ingest.validate import CLEAN_SCHEMA


def test_parse_months_ranges_across_year():
    assert parse_months("2019-11:2020-02,2021-05") == [(2019, 11), (2019, 12), (2020, 1), (2020, 2), (2021, 5)]


def test_trip_url_pattern():
    assert trip_url(2019, 1).endswith("/trip-data/yellow_tripdata_2019-01.parquet")


def test_bad_rows_are_rejected_and_counted(lake):
    rep = lake["2019-01"]
    assert rep["rows_out"] < rep["rows_in"]
    assert rep["pickup_outside_month"] > 0 and rep["unknown_zone"] > 0 and rep["dropoff_before_pickup"] > 0


def test_lake_is_clean_and_partitioned(lake):
    df = read_lake([(2019, 1)])
    assert len(df) == lake["2019-01"]["rows_out"]
    assert (df["pickup_ts"].dt.month == 1).all()
    CLEAN_SCHEMA.validate(df.assign(pickup_ts=df["pickup_ts"].astype("datetime64[ns]"),
                                    dropoff_ts=df["dropoff_ts"].astype("datetime64[ns]")))
    assert df["trip_id"].is_unique
    assert pd.api.types.is_float_dtype(df["duration_s"])
