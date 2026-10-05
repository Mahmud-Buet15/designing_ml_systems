"""Drift detection (Milestone 7, Ch. 8).

    taxieta drift-check --reference 2019-04 --current 2020-04

Compares a reference window with a current window on:
  * numeric features   -> two-sample Kolmogorov-Smirnov test
  * categorical ones   -> Population Stability Index (PSI)
  * the label          -> KS on duration (label shift), when labels are available
Writes reports/drift/<current>.json with a top-level "drift_detected" the Airflow DAG branches on.

TODO(M7): hourly/daily windows, same-hour-of-week references (to kill seasonality false
alarms), sliding vs cumulative metrics, multivariate drift (alibi-detect MMD / Evidently).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from taxieta import config, model_store
from taxieta.features.build import build_features, point_in_time_zone_speed
from taxieta.ingest.lake import read_lake

NUMERIC = ["hour", "hist_median_s", "zone_speed_15m"]
CATEGORICAL = ["pu_borough", "do_borough", "pu_airport", "do_airport", "is_weekend"]


def psi(ref: pd.Series, cur: pd.Series, eps: float = 1e-4) -> float:
    r = ref.astype(str).value_counts(normalize=True)
    c = cur.astype(str).value_counts(normalize=True)
    idx = r.index.union(c.index)
    r, c = r.reindex(idx, fill_value=0) + eps, c.reindex(idx, fill_value=0) + eps
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, numeric=NUMERIC, categorical=CATEGORICAL,
                 alpha: float = 0.01, psi_threshold: float = 0.2, sample: int = 20_000,
                 seed: int = 0) -> dict:
    # KS p-values shrink with sample size: with millions of rows everything is "significant".
    # Sampling a fixed n is a crude fix -- TODO(M7): think about effect size vs significance.
    ref = ref.sample(min(sample, len(ref)), random_state=seed)
    cur = cur.sample(min(sample, len(cur)), random_state=seed)
    out: dict = {"n_ref": len(ref), "n_cur": len(cur), "features": {}}
    for col in numeric:
        a, b = ref[col].dropna(), cur[col].dropna()
        if len(a) < 30 or len(b) < 30:
            continue
        stat, p = ks_2samp(a, b)
        out["features"][col] = {"test": "ks", "stat": float(stat), "p_value": float(p), "drift": bool(p < alpha)}
    for col in categorical:
        v = psi(ref[col], cur[col])
        out["features"][col] = {"test": "psi", "psi": v, "drift": bool(v > psi_threshold)}
    out["drift_detected"] = any(f["drift"] for f in out["features"].values())
    return out


def check(reference: str, current: str) -> dict:
    champ = model_store.load_champion()
    if champ is None:
        raise RuntimeError("no champion model")
    frames = {}
    for name, spec in (("ref", reference), ("cur", current)):
        y, m = map(int, spec.split("-"))
        trips = read_lake([(y, m)])
        trips["zone_speed_15m"] = point_in_time_zone_speed(trips)
        X = build_features(trips, champ["hist"], zone_speed=trips["zone_speed_15m"])
        X["duration_s"] = trips["duration_s"].to_numpy()
        frames[name] = X
    rep = drift_report(frames["ref"], frames["cur"], numeric=NUMERIC + ["duration_s"])
    rep.update({"reference": reference, "current": current, "model_version": champ["version"]})
    d = config.REPORTS_DIR / "drift"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{current}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "features"}, indent=2))
    for f, v in rep["features"].items():
        print(f"  {f:18s} {'DRIFT' if v['drift'] else 'ok':5s} {v}")
    return rep
