"""Central configuration. Everything overridable with environment variables so the same
code runs on your laptop, in Docker Compose, and inside Airflow tasks."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("TAXIETA_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("TAXIETA_DATA", ROOT / "data"))

RAW_DIR = DATA / "raw"
LAKE_DIR = DATA / "lake" / "trips"          # Parquet, partitioned year=/month=
FEATURES_DIR = DATA / "features"
ONLINE_DIR = DATA / "online"                # SQLite online stores + prediction log
MODELS_DIR = DATA / "models"                # local model artifacts (MLflow optional)
REPORTS_DIR = Path(os.environ.get("TAXIETA_REPORTS", ROOT / "reports"))

ZONE_LOOKUP = RAW_DIR / "taxi_zone_lookup.csv"
PREDICTION_LOG = ONLINE_DIR / "predictions.sqlite"
ONLINE_FEATURES = ONLINE_DIR / "online_features.sqlite"

TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net"

# Kafka / Redpanda
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:19092")
TOPIC_STARTED = "trip_started"
TOPIC_COMPLETED = "trip_completed"

# MLflow (optional; falls back to local joblib files if unset)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
MODEL_NAME = "taxieta-eta"

# Domain constants
UNKNOWN_ZONES = {264, 265}                  # "Unknown" / "Outside of NYC" in the zone lookup
AIRPORT_ZONES = {1, 132, 138}               # Newark (EWR), JFK, LaGuardia
MAX_DURATION_S = 6 * 3600
LONG_TRIP_S = 60 * 60                        # LongTrip side task threshold (Milestone 3)

# Serving
FALLBACK_TIMEOUT_MS = int(os.environ.get("FALLBACK_TIMEOUT_MS", "100"))


def ensure_dirs() -> None:
    for d in (RAW_DIR, LAKE_DIR, FEATURES_DIR, ONLINE_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
