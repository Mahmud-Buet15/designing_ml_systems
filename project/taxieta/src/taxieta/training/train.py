"""Train an ETA model and register it as a challenger (Milestones 5 and 8).

    taxieta train --train 2019-01:2019-03 --valid 2019-04 --test 2019-05

What's done for you: time-based split, leakage-safe historical features, point-in-time
streaming features, LightGBM on log-duration, baselines, a slice report, the model store,
and optional MLflow logging (set MLFLOW_TRACKING_URI).

What's left for you: see the TODO(M5)/TODO(M8) markers.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from taxieta import config, model_store
from taxieta.features.build import HistoricalStats, build_features, point_in_time_zone_speed
from taxieta.ingest.download import parse_months
from taxieta.ingest.lake import read_lake
from taxieta.training import metrics
from taxieta.training.baselines import HeuristicETA, ZeroRule

DEFAULT_PARAMS = {
    "objective": "regression_l1",   # MAE on log-duration; TODO(M5): compare huber / l2
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 42,
}


def _months_frame(df: pd.DataFrame, months: list[tuple[int, int]]) -> pd.DataFrame:
    key = df["pickup_ts"].dt.year * 100 + df["pickup_ts"].dt.month
    return df[key.isin([y * 100 + m for y, m in months])]


def train(train_spec: str, valid_spec: str, test_spec: str, params: dict | None = None,
          num_boost_round: int = 2000, parent: str | None = None) -> dict:
    params = {**DEFAULT_PARAMS, **(params or {})}
    tr_m, va_m, te_m = parse_months(train_spec), parse_months(valid_spec), parse_months(test_spec)
    trips = read_lake(tr_m + va_m + te_m)
    trips["zone_speed_15m"] = point_in_time_zone_speed(trips)

    train_df, valid_df, test_df = (_months_frame(trips, m) for m in (tr_m, va_m, te_m))
    cutoff = valid_df["pickup_ts"].min()
    # Leakage-safe: history is fitted on TRAIN ONLY (strictly before the validation period).
    hist = HistoricalStats(cutoff=cutoff).fit(train_df)

    def xy(df):
        X = build_features(df, hist, zone_speed=df["zone_speed_15m"])
        return X, np.log1p(df["duration_s"].to_numpy())

    X_tr, y_tr = xy(train_df)
    X_va, y_va = xy(valid_df)
    X_te, _ = xy(test_df)

    dtrain = lgb.Dataset(X_tr, y_tr)
    booster = lgb.train(
        params, dtrain, num_boost_round=num_boost_round,
        # reference=dtrain keeps category codes/bins consistent between train and valid
        valid_sets=[lgb.Dataset(X_va, y_va, reference=dtrain)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    # TODO(M8 stage 3, stateful training): accept an `init_model=<parent booster>` and continue
    # training on only the newest data instead of retraining from scratch. Compare compute + MAE.

    heuristic = HeuristicETA().fit(train_df)
    zero = ZeroRule().fit(train_df["duration_s"])

    results = {}
    for name, df, X in (("valid", valid_df, X_va), ("test", test_df, X_te)):
        y = df["duration_s"].to_numpy()
        results[name] = {
            "model": metrics.report(y, np.expm1(booster.predict(X, num_iteration=booster.best_iteration))),
            "heuristic": metrics.report(y, heuristic.predict(df)),
            "zero_rule": metrics.report(y, zero.predict(len(df))),
        }

    test_scored = test_df.assign(pred_s=np.expm1(booster.predict(X_te, num_iteration=booster.best_iteration)))
    slices = metrics.slice_report(test_scored, "pu_borough")
    # TODO(M5): add slices by airport/non-airport, hour bucket, trip length; check Simpson's paradox.
    # TODO(M5): perturbation, invariance and directional tests; calibration for the LongTrip task.

    meta = {
        "parent": parent,
        "data_window": {"train": train_spec, "valid": valid_spec, "test": test_spec,
                        "rows": {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)}},
        "features": list(X_tr.columns),
        "params": params,
        "best_iteration": booster.best_iteration,
        "metrics": results,
        "slices_test_pu_borough": slices.to_dict(orient="records"),
    }
    version = model_store.save(booster, hist, heuristic, meta)
    _maybe_log_mlflow(version, params, results, meta)

    out = {"version": version, **results}
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / f"train_{version}.json").write_text(json.dumps(meta | {"version": version},
                                                                          indent=2, default=str))
    print(json.dumps(out, indent=2))
    print(slices.to_string(index=False))
    if model_store.champion_version() is None:
        model_store.set_champion(version)
        print(f"no champion yet -> {version} is now champion")
    return out


def _maybe_log_mlflow(version: str, params: dict, results: dict, meta: dict) -> None:
    if not config.MLFLOW_TRACKING_URI:
        return
    import mlflow

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("taxieta")
    with mlflow.start_run(run_name=version):
        mlflow.log_params({k: v for k, v in params.items() if k != "verbose"})
        mlflow.log_dict(meta, "meta.json")
        for split, by_model in results.items():
            for model_name, ms in by_model.items():
                for k, v in ms.items():
                    mlflow.log_metric(f"{split}.{model_name}.{k}", v)
        mlflow.set_tags({"version": version, "git_commit": str(meta.get("git_commit"))})
        # TODO(M5/M9): log the model with mlflow.lightgbm.log_model and register it under
        # config.MODEL_NAME so the MLflow Model Registry becomes your model store.
