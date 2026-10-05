"""Online prediction service (Milestones 6, 8, 10).

    uvicorn taxieta.serving.app:app --port 8000
    curl -X POST localhost:8000/predict -H 'content-type: application/json' \
         -d '{"pickup_ts": "2019-05-06T08:15:00", "pu_zone": 161, "do_zone": 132}'

Done for you: shared feature code, online-store lookup with staleness check, a hard timeout
that falls back to the heuristic (smooth failing, Ch. 11), and prediction logging.
Left for you: TODO(M5) intervals, TODO(M8) shadow/canary/A-B routing, TODO(M6) micro-batching.
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from taxieta import config, model_store
from taxieta.features.build import build_features
from taxieta.serving.predlog import PredictionLog
from taxieta.stream.store import OnlineStore

MAX_FEATURE_AGE = pd.Timedelta("30min")
_pool = ThreadPoolExecutor(max_workers=4)
STATE: dict = {}


class PredictRequest(BaseModel):
    pickup_ts: datetime
    pu_zone: int = Field(ge=1, le=265)
    do_zone: int = Field(ge=1, le=265)
    trip_id: str | None = None


class PredictResponse(BaseModel):
    request_id: str
    eta_seconds: float
    model_version: str
    fallback: bool
    interval: tuple[float, float] | None = None


def load_state() -> None:
    STATE["champion"] = model_store.load_champion()
    STATE["store"] = OnlineStore()
    STATE["log"] = PredictionLog()


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_state()
    yield


app = FastAPI(title="TaxiETA", lifespan=lifespan)


def _zone_speed_lookup(zone: int, at: pd.Timestamp) -> float | None:
    mph, as_of = STATE["store"].get_zone_speed(zone)
    if mph is None or as_of is None:
        return None
    age = at - pd.Timestamp(as_of)
    if age > MAX_FEATURE_AGE or age < -MAX_FEATURE_AGE:
        return None   # stale (or from the future) -> treat as missing, never silently reuse
    return mph


def _model_predict(champ: dict, X: pd.DataFrame) -> float:
    booster = champ["model"]
    return float(np.expm1(booster.predict(X, num_iteration=booster.best_iteration))[0])


@app.get("/health")
def health() -> dict:
    champ = STATE.get("champion")
    return {"status": "ok" if champ else "no_model", "model_version": champ["version"] if champ else None}


@app.post("/reload")
def reload() -> dict:
    """Pick up a newly promoted champion without restarting (called by the Airflow DAG)."""
    load_state()
    return health()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    champ = STATE.get("champion")
    if champ is None:
        raise HTTPException(503, "no champion model; run `taxieta train` first")
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())
    frame = pd.DataFrame([{"pickup_ts": pd.Timestamp(req.pickup_ts).tz_localize(None),
                           "pu_zone": req.pu_zone, "do_zone": req.do_zone}])
    X = build_features(frame, champ["hist"], zone_speed=_zone_speed_lookup)

    fallback = False
    try:
        eta = _pool.submit(_model_predict, champ, X).result(timeout=config.FALLBACK_TIMEOUT_MS / 1000)
    except (FutTimeout, Exception):
        # Smooth failing (Ch. 11): slower/broken main model -> fast, worse, but always-available heuristic
        eta = float(champ["heuristic"].predict(frame)[0])
        fallback = True

    # TODO(M8): shadow deployment -- also score the challenger here (async), log it with arm="shadow".
    # TODO(M8): canary / A-B -- route a hash(trip_id)-based fraction of traffic to the challenger.
    # TODO(M5/M10): return an interval from quantile models when uncertainty is high.

    latency_ms = (time.perf_counter() - t0) * 1000
    STATE["log"].write({
        "request_id": request_id, "trip_id": req.trip_id,
        "logged_at": datetime.now(UTC).isoformat(),
        "pickup_ts": frame["pickup_ts"].iloc[0].isoformat(),
        "pu_zone": req.pu_zone, "do_zone": req.do_zone,
        "model_version": champ["version"], "arm": "fallback" if fallback else "champion",
        "pred_s": eta, "latency_ms": latency_ms, "fallback": int(fallback),
        "features": X.iloc[0].astype(object).where(X.iloc[0].notna(), None).to_dict(),
    })
    return PredictResponse(request_id=request_id, eta_seconds=eta,
                           model_version=champ["version"], fallback=fallback)
