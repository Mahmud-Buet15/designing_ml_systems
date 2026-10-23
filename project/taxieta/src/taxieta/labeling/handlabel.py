"""Hand labels for evaluating weak supervision (Milestone 3).

    1. export:  taxieta solve m3 --export-for-labeling   -> data/labels/to_label.csv
    2. fill the `label` column yourself: 1 = INVALID trip (meter/data glitch), 0 = VALID
       (write your guideline down first -- see docs/solutions/M3.md)
    3. save as data/labels/hand_labels.csv (and optionally a second pass, hand_labels_2.csv)

On fixture data we can use the generator's ground truth instead (fixture_truth_*.parquet).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from taxieta import config

LABELS_DIR = config.DATA / "labels"
KEY = ["pickup_ts", "pu_zone", "do_zone"]


def export_for_labeling(trips: pd.DataFrame, L: np.ndarray, n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Stratify by LF outcome so rare 'invalid' candidates are well represented: half from rows
    any LF flags INVALID, half from the rest."""
    flagged = (L == 1).any(axis=1)
    rng = np.random.default_rng(seed)
    a = trips[flagged].sample(min(n // 2, int(flagged.sum())), random_state=seed)
    b = trips[~flagged].sample(n - len(a), random_state=int(rng.integers(1e9)))
    out = pd.concat([a, b]).sample(frac=1, random_state=seed)
    cols = ["trip_id", "pickup_ts", "dropoff_ts", "pu_zone", "do_zone", "pu_borough", "do_borough",
            "duration_s", "trip_distance", "fare_amount"]
    out = out[cols].assign(label="")
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(LABELS_DIR / "to_label.csv", index=False)
    return out


def load_hand_labels(name: str = "hand_labels.csv") -> pd.DataFrame | None:
    p = LABELS_DIR / name
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df.dropna(subset=["label"]).astype({"label": int})


def fixture_truth(trips: pd.DataFrame) -> pd.Series | None:
    """1/0 ground truth for fixture data, aligned with `trips`; None if not fixture data."""
    files = sorted(config.RAW_DIR.glob("fixture_truth_*.parquet"))
    if not files:
        return None
    t = pd.concat(pd.read_parquet(f) for f in files).rename(columns={
        "tpep_pickup_datetime": "pickup_ts", "PULocationID": "pu_zone", "DOLocationID": "do_zone"})
    t["pickup_ts"] = pd.to_datetime(t["pickup_ts"]).astype("datetime64[ns]")
    t = t.astype({"pu_zone": "int32", "do_zone": "int32"}).drop_duplicates(KEY)
    k = trips[KEY].astype({"pu_zone": "int32", "do_zone": "int32"})
    merged = k.merge(t, on=KEY, how="left")
    return pd.Series(merged["invalid"].fillna(0).astype(int).to_numpy(), index=trips.index)


def cohens_kappa(a: pd.Series, b: pd.Series) -> float:
    """Agreement between two labelers beyond chance (label multiplicity, Ch. 4)."""
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in np.union1d(a, b))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0
