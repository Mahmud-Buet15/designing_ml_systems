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
        self.ignored = 0   # SOLVED(M2): junk events are counted -- a spike is a monitoring signal
        self.accepted = 0

    def add(self, zone: int, dropoff_ts: pd.Timestamp, distance_mi: float, duration_s: float) -> None:
        """Junk-filtering: ignore trips that are too short or have no distance.
        
        These are excluded from the median calculation and counted in `self.ignored`. This
        ensures that the median speed reflects only valid, meaningful trips.
        """
        if duration_s <= 60 or distance_mi <= 0:
            self.ignored += 1
            return
        self.accepted += 1
        self._events[zone].append((dropoff_ts, distance_mi / (duration_s / 3600)))

    def median_speed(self, zone: int, now: pd.Timestamp) -> tuple[float | None, int]:
        """Calculates the median speed for a given zone at a specific time.
        
        Removes expired events (outside the time window) and computes the median speed
        of the remaining events.
        
        Args:
            zone: The pickup zone ID.
            now: The current timestamp.
            
        Returns:
            A tuple of (median_speed, event_count).
        """
        q = self._events[zone]  # get the deque of events for the given zone
        while q and q[0][0] < now - self.window:  # remove expired events (outside the time window)
            q.popleft()
        if not q:  # if no events are left, return None, 0
            return None, 0
        return statistics.median(s for _, s in q), len(q)


def batch_zone_speed(trips: pd.DataFrame, window: str = "15min") -> pd.Series:
    """SOLVED(M2): the BATCH version of the same feature, for every trip's pickup moment.

    Same definition as ZoneSpeedWindow: completions from the same pickup zone whose dropoff is in
    [pickup - window, pickup) with duration > 60 s and distance > 0. Implemented once in
    taxieta.features.build.point_in_time_zone_speed and reused here, so training and the
    streaming consumer can't drift apart (compare them with `taxieta solve m2`).
    """
    from taxieta.features.build import point_in_time_zone_speed

    return point_in_time_zone_speed(trips, window)


def replay_zone_speed(trips: pd.DataFrame, window: pd.Timedelta = DEFAULT_WINDOW) -> pd.Series:
    """Run the STREAMING logic over historical events (no Kafka) and record, at every
    trip_started event, what the online store would hold for its pickup zone."""
    from taxieta.stream.events import TripCompleted, event_stream

    win = ZoneSpeedWindow(window)
    out: dict[str, float | None] = {}
    for ts, ev in event_stream(trips): # for every trip event...
        if isinstance(ev, TripCompleted): # if trip is completed then add it to the window
            win.add(ev.pu_zone, ts, ev.trip_distance, ev.duration_s)
        else:                             # if trip is started then get the median speed from the window for that trip's pickup zone
            out[ev.trip_id] = win.median_speed(ev.pu_zone, ts)[0]
    return trips["trip_id"].map(out).astype(float)  # using it to merge the result with the original dataframe  

# Note: Check docs/explanations.md for detailed explanations

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
