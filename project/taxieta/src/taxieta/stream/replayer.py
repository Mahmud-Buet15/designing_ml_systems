"""Replay a historical month as a live event stream (Milestone 2).

    taxieta replay --month 2019-05 --speedup 600            # 1 real second = 10 simulated minutes
    taxieta replay --month 2020-04 --sink print --limit 20  # no Kafka needed

Sinks:
  kafka  -- publish JSON to Redpanda/Kafka topics trip_started / trip_completed
  print  -- print events (debugging)
  http   -- POST trip_started events straight to the prediction API (Milestone 6)
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from taxieta import config
from taxieta.ingest.lake import read_lake
from taxieta.stream.events import TripCompleted, TripStarted, event_stream, to_dict

Event = TripStarted | TripCompleted


def kafka_sink() -> tuple[Callable[[Event], None], Callable[[], None]]:
    from confluent_kafka import Producer

    producer = Producer({"bootstrap.servers": config.KAFKA_BOOTSTRAP, "linger.ms": 20})

    def send(ev: Event) -> None:
        producer.produce(ev.topic, key=ev.trip_id, value=json.dumps(to_dict(ev)))
        producer.poll(0)

    return send, lambda: producer.flush(10)


def print_sink() -> tuple[Callable[[Event], None], Callable[[], None]]:
    return (lambda ev: print(ev.topic, json.dumps(to_dict(ev)))), (lambda: None)


def http_sink(url: str = "http://localhost:8000/predict") -> tuple[Callable[[Event], None], Callable[[], None]]:
    import httpx

    client = httpx.Client(timeout=2.0)

    def send(ev: Event) -> None:
        if isinstance(ev, TripStarted):
            client.post(url, json={"trip_id": ev.trip_id, "pickup_ts": ev.pickup_ts,
                                   "pu_zone": ev.pu_zone, "do_zone": ev.do_zone})
        else:
            # Labels arrive later -> append to the label log so the joiner can match them.
            from taxieta.labeling.join_labels import record_completion
            record_completion(to_dict(ev))

    return send, client.close


SINKS = {"kafka": kafka_sink, "print": print_sink, "http": http_sink}


def replay(month: str, speedup: float = 600.0, sink: str = "kafka", limit: int | None = None,
           no_sleep: bool = False) -> int:
    y, m = map(int, month.split("-"))
    trips = read_lake([(y, m)])
    if limit:
        trips = trips.head(limit)
    send, close = SINKS[sink]()
    t0_sim = None
    t0_wall = time.monotonic()
    n = 0
    try:
        for ts, ev in event_stream(trips):
            if t0_sim is None:
                t0_sim = ts
            if not no_sleep:
                target = (ts - t0_sim).total_seconds() / speedup
                delay = target - (time.monotonic() - t0_wall)
                if delay > 0:
                    time.sleep(delay)
            send(ev)
            n += 1
    finally:
        close()
    print(f"replayed {n} events from {month}")
    return n
