"""Label computation: join predictions (made at pickup) with completions (natural labels)
that arrive later (Milestones 3, 7, 8; Ch. 4 "natural labels", Ch. 9 "label computation").
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from taxieta import config

LABELS_DB = config.ONLINE_DIR / "labels.sqlite"


def _conn(path: Path = LABELS_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE IF NOT EXISTS completions (trip_id TEXT PRIMARY KEY, payload TEXT, "
              "dropoff_ts TEXT)")
    return c


def record_completion(event: dict) -> None:
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO completions VALUES (?,?,?)",
                  (event["trip_id"], json.dumps(event), event["dropoff_ts"]))


def load_completions() -> pd.DataFrame:
    with _conn() as c:
        rows = c.execute("SELECT payload FROM completions").fetchall()
    return pd.DataFrame([json.loads(r[0]) for r in rows])


def join_predictions_with_labels(
    predictions: pd.DataFrame, completions: pd.DataFrame, label_window: pd.Timedelta | None = None
) -> pd.DataFrame:
    """Inner-join on trip_id. With `label_window`, labels that arrive later than
    pickup + window are treated as 'not yet available' -- which trips does that drop?"""
    if predictions.empty or completions.empty:
        return pd.DataFrame()
    df = predictions.merge(completions[["trip_id", "dropoff_ts", "duration_s"]], on="trip_id", how="inner")
    df["pickup_ts"] = pd.to_datetime(df["pickup_ts"])
    df["dropoff_ts"] = pd.to_datetime(df["dropoff_ts"])
    df["feedback_delay_s"] = (df["dropoff_ts"] - df["pickup_ts"]).dt.total_seconds()
    if label_window is not None:
        df = df[df["feedback_delay_s"] <= label_window.total_seconds()]
    return df


def feedback_loop_report(joined: pd.DataFrame) -> dict:
    d = joined["feedback_delay_s"] / 60
    return {"n": int(len(d)), "median_min": float(d.median()), "p90_min": float(d.quantile(0.9)),
            "p99_min": float(d.quantile(0.99))}
