"""Shared bits for the TaxiETA DAGs (Airflow 3, Milestone 9).

Design choice worth noticing (Ch. 10, "conflicting dependencies"): Airflow and TaxiETA live
in DIFFERENT Python environments. The DAGs never `import taxieta`; each task shells out to
the `taxieta` CLI installed in its own venv (TAXIETA_BIN). Airflow's pinned dependencies and
your ML stack can't break each other, and every task is runnable by hand for debugging.
"""

from __future__ import annotations

import os
from datetime import date

from airflow.sdk import Asset

TAXIETA_BIN = os.environ.get("TAXIETA_BIN", "taxieta")
API_URL = os.environ.get("TAXIETA_API_URL", "http://api:8000")

# Assets = data-aware scheduling. Producers declare outlets; consumers schedule on them.
LAKE = Asset("taxieta://lake/trips")                      # new month landed in the lake
DRIFT = Asset("taxieta://monitoring/drift_detected")      # monitoring saw a real shift
CHAMPION = Asset("taxieta://models/champion")             # a new champion was promoted

DEFAULT_ARGS = {"owner": "you", "retries": 1}


def shift_month(month: str, delta: int) -> str:
    y, m = map(int, month.split("-"))
    d = date(y, m, 1)
    idx = d.year * 12 + (d.month - 1) + delta
    return f"{idx // 12}-{idx % 12 + 1:02d}"
