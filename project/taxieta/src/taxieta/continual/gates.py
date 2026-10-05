"""Evaluation gates before promoting a challenger (Milestone 8, Ch. 9).

    taxieta gate --challenger v0002 --backtest 2019-05

Writes reports/gates/<challenger>.json with {"promote": bool, "gates": [...]}.
The Airflow retrain DAG branches on "promote".

Gates implemented: (1) backtest MAE no worse than champion (+tolerance), (2) no borough
slice more than `slice_tol` worse. TODO(M8): add the static trusted test set gate, the
directional/invariance tests from M5, and a cap on how far one day of data can move the model.
"""

from __future__ import annotations

import json

import numpy as np

from taxieta import config, model_store
from taxieta.features.build import build_features, point_in_time_zone_speed
from taxieta.ingest.lake import read_lake
from taxieta.training import metrics


def _score(bundle: dict, trips) -> np.ndarray:
    X = build_features(trips, bundle["hist"], zone_speed=trips["zone_speed_15m"])
    return np.expm1(bundle["model"].predict(X, num_iteration=bundle["model"].best_iteration))


def run_gates(challenger: str, backtest: str, tol: float = 0.0, slice_tol: float = 0.10) -> dict:
    champ = model_store.load_champion()
    chall = model_store.load(challenger)
    y, m = map(int, backtest.split("-"))
    trips = read_lake([(y, m)])
    trips["zone_speed_15m"] = point_in_time_zone_speed(trips)
    y_true = trips["duration_s"].to_numpy()

    gates = []
    if champ is None or champ["version"] == challenger:
        gates.append({"name": "no_incumbent", "passed": True})
    else:
        p_champ, p_chall = _score(champ, trips), _score(chall, trips)
        m_champ, m_chall = metrics.mae_min(y_true, p_champ), metrics.mae_min(y_true, p_chall)
        gates.append({"name": "backtest_mae", "champion": m_champ, "challenger": m_chall,
                      "passed": m_chall <= m_champ * (1 + tol)})
        s_champ = metrics.slice_report(trips.assign(pred_s=p_champ), "pu_borough").set_index("pu_borough")
        s_chall = metrics.slice_report(trips.assign(pred_s=p_chall), "pu_borough").set_index("pu_borough")
        ratio = (s_chall["mae_min"] / s_champ["mae_min"]).dropna()
        worst = ratio.idxmax() if len(ratio) else None
        gates.append({"name": "slice_regression", "worst_slice": worst,
                      "worst_ratio": float(ratio.max()) if len(ratio) else None,
                      "passed": bool((ratio <= 1 + slice_tol).all())})

    decision = {"challenger": challenger, "champion": champ["version"] if champ else None,
                "backtest": backtest, "gates": gates, "promote": all(g["passed"] for g in gates)}
    d = config.REPORTS_DIR / "gates"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{challenger}.json").write_text(json.dumps(decision, indent=2, default=str))
    print(json.dumps(decision, indent=2, default=str))
    return decision


def promote(challenger: str) -> None:
    # TODO(M8): replace instant promotion with a canary rollout (5% -> 25% -> 50% -> 100%).
    model_store.set_champion(challenger)
    print(f"promoted {challenger} to champion")
