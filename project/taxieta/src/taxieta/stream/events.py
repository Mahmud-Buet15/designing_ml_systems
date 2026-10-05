"""Turn lake rows into the two event types the live system sees (Milestone 2).

trip_started   -- emitted at pickup time, carries ONLY what is known at pickup.
trip_completed -- emitted at dropoff time, carries the natural label (duration) and
                  after-the-fact facts (distance, fare). Never use these as features!
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class TripStarted:
    trip_id: str
    pickup_ts: str
    pu_zone: int
    do_zone: int

    topic = "trip_started"


@dataclass
class TripCompleted:
    trip_id: str
    pickup_ts: str
    dropoff_ts: str
    pu_zone: int
    do_zone: int
    duration_s: float
    trip_distance: float
    fare_amount: float

    topic = "trip_completed"


def to_dict(ev: TripStarted | TripCompleted) -> dict:
    return asdict(ev)


def event_stream(trips: pd.DataFrame) -> Iterator[tuple[pd.Timestamp, TripStarted | TripCompleted]]:
    """Yield (event_time, event) for all trips, merged in event-time order."""
    starts = trips[["trip_id", "pickup_ts", "pu_zone", "do_zone"]].assign(kind=0, ts=trips["pickup_ts"])
    ends = trips[["trip_id", "pickup_ts", "dropoff_ts", "pu_zone", "do_zone", "duration_s",
                  "trip_distance", "fare_amount"]].assign(kind=1, ts=trips["dropoff_ts"])
    merged = pd.concat([starts, ends], ignore_index=True).sort_values(["ts", "kind"], kind="stable")
    for r in merged.itertuples(index=False):
        if r.kind == 0:
            yield r.ts, TripStarted(r.trip_id, r.pickup_ts.isoformat(), int(r.pu_zone), int(r.do_zone))
        else:
            yield r.ts, TripCompleted(
                r.trip_id, r.pickup_ts.isoformat(), r.dropoff_ts.isoformat(), int(r.pu_zone),
                int(r.do_zone), float(r.duration_s), float(r.trip_distance), float(r.fare_amount),
            )
