from collections import Counter

import pandas as pd

from taxieta.continual.bandit import ThompsonSampler
from taxieta.ingest.lake import read_lake
from taxieta.labeling.sampling import reservoir, stratified
from taxieta.stream.consumer import ZoneSpeedWindow
from taxieta.stream.events import TripCompleted, TripStarted, event_stream
from taxieta.training.metrics import pct_within


def test_events_are_time_ordered_and_labels_come_after_starts(lake):
    trips = read_lake([(2019, 1)]).head(200)
    seen_start = set()
    last_ts = None
    for ts, ev in event_stream(trips):
        assert last_ts is None or ts >= last_ts
        last_ts = ts
        if isinstance(ev, TripStarted):
            assert not hasattr(ev, "duration_s")     # no label at pickup time
            seen_start.add(ev.trip_id)
        else:
            assert isinstance(ev, TripCompleted) and ev.trip_id in seen_start


def test_zone_speed_window_expires_old_events():
    w = ZoneSpeedWindow(pd.Timedelta("15min"))
    t0 = pd.Timestamp("2019-01-01 08:00")
    w.add(1, t0, 2.0, 600)                        # 12 mph
    w.add(1, t0 + pd.Timedelta("10min"), 1.0, 600)  # 6 mph
    assert w.median_speed(1, t0 + pd.Timedelta("11min")) == (9.0, 2)
    assert w.median_speed(1, t0 + pd.Timedelta("20min")) == (6.0, 1)
    assert w.median_speed(2, t0) == (None, 0)


def test_reservoir_is_roughly_uniform():
    counts = Counter()
    for seed in range(2000):
        counts.update(reservoir(range(10), k=3, seed=seed))
    freqs = [counts[i] / 2000 for i in range(10)]
    assert all(abs(f - 0.3) < 0.05 for f in freqs)


def test_stratified_keeps_rare_strata(lake):
    trips = read_lake([(2019, 1)])
    s = stratified(trips, "pu_borough", frac=0.05, seed=1)
    assert set(s["pu_borough"]) == set(trips["pu_borough"])


def test_pct_within():
    assert pct_within([100, 100, 100, 100], [100, 119, 121, 50]) == 50.0


def test_thompson_sampler_finds_best_arm():
    import numpy as np
    rng = np.random.default_rng(0)
    ts = ThompsonSampler(["a", "b", "c"], seed=0)
    p = {"a": 0.4, "b": 0.6, "c": 0.5}
    picks = Counter()
    for _ in range(3000):
        arm = ts.choose()
        picks[arm] += 1
        ts.update(arm, int(rng.random() < p[arm]))
    assert picks.most_common(1)[0][0] == "b"
