"""DAG 3 -- retrain, gate, promote (Milestones 8 and 9; Ch. 9 continual learning).

    (DRIFT asset or manual) -> plan_windows -> train_challenger -> gate
        -> branch -> promote -> reload_api (CHAMPION asset)
                  -> keep_champion (+ alert)

The branch is the book's conditional dependency: "deploy A if A is better; otherwise B".
"""

from __future__ import annotations

import json
import urllib.request

import pendulum
from airflow.sdk import Param, dag, task
from taxieta_common import API_URL, CHAMPION, DEFAULT_ARGS, DRIFT, TAXIETA_BIN, shift_month


@dag(
    dag_id="taxieta_retrain",
    schedule=[DRIFT],                    # TODO(M8): try a cron schedule and compare
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    params={"month": Param("2019-06", type="string",
                           description="used on manual runs; drift-triggered runs use the event's month")},
    tags=["taxieta", "training"],
)
def taxieta_retrain():
    @task
    def plan_windows(triggering_asset_events=None, params=None) -> dict:
        months = [ev.extra.get("month") for evs in (triggering_asset_events or {}).values()
                  for ev in evs if ev.extra.get("month")]
        m = sorted(months)[-1] if months else params["month"]
        # TODO(M8): is this the right window? Measure the value of data freshness and decide.
        plan = {"train": f"{shift_month(m, -3)}:{shift_month(m, -2)}",
                "valid": shift_month(m, -1), "test": m, "backtest": m}
        print(json.dumps(plan))
        return plan

    @task.bash
    def train_challenger(plan: dict) -> str:
        # TODO(M8 stage 3): pass the champion as a warm start for stateful training.
        return (f"{TAXIETA_BIN} train --train {plan['train']} --valid {plan['valid']} "
                f"--test {plan['test']}")

    @task.bash
    def gate(version: str, plan: dict) -> str:
        return f"{TAXIETA_BIN} gate --challenger {version.strip()} --backtest {plan['backtest']}"

    @task.branch
    def decide(promote: str) -> str:
        return "promote" if promote.strip().lower() == "true" else "keep_champion"

    @task.bash
    def promote(version: str) -> str:
        # TODO(M8): replace with a canary rollout (5% -> 25% -> 50% -> 100%) with auto-rollback.
        return f"{TAXIETA_BIN} promote --challenger {version.strip()}"

    @task(outlets=[CHAMPION])
    def reload_api():
        try:
            req = urllib.request.Request(f"{API_URL}/reload", method="POST")
            with urllib.request.urlopen(req, timeout=10) as r:   # noqa: S310
                print(r.read().decode())
        except Exception as e:  # the API may not be running in every setup
            print(f"could not reach API at {API_URL}: {e}")

    @task
    def keep_champion(version: str):
        # TODO(M7/M8): send an alert with the gate report and a runbook link.
        print(f"challenger {version.strip()} failed its gates; keeping the champion")

    plan = plan_windows()
    version = train_challenger(plan)
    branch = decide(gate(version, plan))
    p = promote(version)
    branch >> [p, keep_champion(version)]
    p >> reload_api()


taxieta_retrain()
