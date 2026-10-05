"""Streaming feature: median speed of trips completed in each pickup zone over the last
15 minutes of *event time* (Milestones 2 and 6).

    taxieta consume            # reads trip_completed from Kafka, writes the online store

The window logic (`ZoneSpeedWindow`) is a pure class so it can be unit-tested and reused.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict, deque

import pandas as pd

from taxieta import config
from taxieta.stream.store import OnlineStore

DEFAULT_WINDOW = pd.Timedelta("15min")


class ZoneSpeedWindow:
    def __init__(self, window: pd.Timedelta = DEFAULT_WINDOW):
        self.window = window
        self._events: dict[int, deque[tuple[pd.Timestamp, float]]] = defaultdict(deque)

    def add(self, zone: int, dropoff_ts: pd.Timestamp, distance_mi: float, duration_s: float) -> None:
        if duration_s <= 60 or distance_mi <= 0:
            return  # ignore junk; TODO(M2): count what you ignore -- it's a monitoring signal
        self._events[zone].append((dropoff_ts, distance_mi / (duration_s / 3600)))

    def median_speed(self, zone: int, now: pd.Timestamp) -> tuple[float | None, int]:
        q = self._events[zone]
        while q and q[0][0] < now - self.window:
            q.popleft()
        if not q:
            return None, 0
        return statistics.median(s for _, s in q), len(q)


# TODO(M2): implement the BATCH version of the same feature over the lake, e.g.
#   def batch_zone_speed(trips: pd.DataFrame, zone: int, at: pd.Timestamp) -> float | None
# then compare it against ZoneSpeedWindow for one day. Every mismatch is train/serve skew.


def run_consumer(group_id: str = "zone-speed") -> None:  # pragma: no cover - needs Kafka
    from confluent_kafka import Consumer

    store = OnlineStore()
    win = ZoneSpeedWindow()
    c = Consumer({"bootstrap.servers": config.KAFKA_BOOTSTRAP, "group.id": group_id,
                  "auto.offset.reset": "earliest"})
    c.subscribe([config.TOPIC_COMPLETED])
    print("consuming trip_completed ... Ctrl-C to stop")
    try:
        while True:
            msg = c.poll(1.0)
            if msg is None or msg.error():
                continue
            ev = json.loads(msg.value())
            ts = pd.Timestamp(ev["dropoff_ts"])
            win.add(ev["pu_zone"], ts, ev["trip_distance"], ev["duration_s"])
            mph, n = win.median_speed(ev["pu_zone"], ts)
            store.put_zone_speed(ev["pu_zone"], mph, n, ts.isoformat())
    finally:
        c.close()
