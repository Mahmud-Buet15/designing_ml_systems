"""Prediction log: every prediction with its features and model version (Milestone 6).

Monitoring (M7), label joining (M3/M8) and 'log and wait' retraining (M8) all read this.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pandas as pd

from taxieta import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id TEXT PRIMARY KEY,
    trip_id TEXT,
    logged_at TEXT,
    pickup_ts TEXT,
    pu_zone INTEGER,
    do_zone INTEGER,
    model_version TEXT,
    arm TEXT,             -- champion / challenger / shadow / fallback (M8)
    pred_s REAL,
    latency_ms REAL,
    fallback INTEGER,
    features TEXT         -- JSON: the exact feature vector used
)"""


class PredictionLog:
    def __init__(self, path: Path | None = None):
        path = path or config.PREDICTION_LOG
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(SCHEMA)

    def write(self, row: dict) -> None:
        row = {**row, "features": json.dumps(row.get("features", {}), default=str)}
        cols = ",".join(row)
        qs = ",".join("?" * len(row))
        with self._lock:
            self.conn.execute(f"INSERT OR REPLACE INTO predictions ({cols}) VALUES ({qs})", list(row.values()))
            self.conn.commit()

    def read(self, since: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM predictions" + (" WHERE logged_at >= ?" if since else "")
        return pd.read_sql_query(q, self.conn, params=(since,) if since else None)
