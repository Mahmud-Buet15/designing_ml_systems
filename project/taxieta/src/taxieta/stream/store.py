"""A tiny SQLite 'online store' for streaming features (Milestone 2, replaced by Feast in M9).

Real systems use Redis/DynamoDB/etc. SQLite keeps the concepts visible with zero setup.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from taxieta import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS zone_speed (
    zone_id INTEGER PRIMARY KEY,
    median_mph REAL,
    n INTEGER,
    as_of TEXT
)"""


class OnlineStore:
    def __init__(self, path: Path | None = None):
        path = path or config.ONLINE_FEATURES
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(SCHEMA)

    def put_zone_speed(self, zone_id: int, median_mph: float | None, n: int, as_of: str) -> None:
        self.conn.execute(
            "INSERT INTO zone_speed VALUES (?,?,?,?) ON CONFLICT(zone_id) DO UPDATE SET "
            "median_mph=excluded.median_mph, n=excluded.n, as_of=excluded.as_of",
            (zone_id, median_mph, n, as_of),
        )
        self.conn.commit()

    def get_zone_speed(self, zone_id: int) -> tuple[float | None, str | None]:
        row = self.conn.execute(
            "SELECT median_mph, as_of FROM zone_speed WHERE zone_id=?", (zone_id,)
        ).fetchone()
        return (row[0], row[1]) if row else (None, None)
