"""The leak firewall: features must depend only on what is known at pickup."""

import pandas as pd

from taxieta.features.build import (
    FEATURE_COLUMNS,
    LEAKY_COLUMNS,
    HistoricalStats,
    build_features,
    point_in_time_zone_speed,
)
from taxieta.ingest.lake import read_lake


def test_no_leaky_feature_names():
    assert not set(FEATURE_COLUMNS) & LEAKY_COLUMNS


def test_features_ignore_after_the_fact_columns(lake):
    trips = read_lake([(2019, 2)]).head(500)
    hist = HistoricalStats().fit(read_lake([(2019, 1)]))
    a = build_features(trips, hist)
    tampered = trips.assign(trip_distance=0.0, fare_amount=999.0, duration_s=1.0,
                            dropoff_ts=trips["pickup_ts"])
    b = build_features(tampered, hist)
    pd.testing.assert_frame_equal(a, b)


def test_historical_stats_respect_cutoff(lake):
    trips = read_lake([(2019, 1), (2019, 2)])
    cutoff = pd.Timestamp("2019-02-01")
    h_cut = HistoricalStats(cutoff=cutoff).fit(trips)
    h_before = HistoricalStats().fit(trips[trips["pickup_ts"] < cutoff])
    req = read_lake([(2019, 3)]).head(300)
    pd.testing.assert_frame_equal(h_cut.transform(req), h_before.transform(req))


def test_point_in_time_zone_speed_never_sees_the_future(lake):
    trips = read_lake([(2019, 1)]).head(3000)
    zs = point_in_time_zone_speed(trips)
    # The very first trip of each zone has no completed trips before it.
    first = trips.sort_values("pickup_ts").groupby("pu_zone").head(1).index
    assert zs.loc[first].isna().all()
