"""DAG 2 -- drift monitoring, triggered by new data in the lake (Milestones 7 and 9).

    (LAKE asset) -> drift_check -> branch -> signal_drift (DRIFT asset) | no_drift

This is the book's Stage 4 "drift-based trigger" (Ch. 9): retraining is scheduled on the
DRIFT asset, not on the clock. TODO(M8): compare against time/performance/volume triggers.
"""

from __future__ import annotations

import pendulum
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import Metadata, Param, dag, task
from taxieta_common import DEFAULT_ARGS, DRIFT, LAKE, TAXIETA_BIN


@dag(
    dag_id="taxieta_monitoring",
    schedule=[LAKE],
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    params={
        "reference": Param("2019-04", type="string",
                           description="reference month (e.g. the champion's validation month)"),
        "month": Param("2019-05", type="string",
                       description="used on manual runs; asset-triggered runs use the event's month"),
    },
    tags=["taxieta", "monitoring"],
)
def taxieta_monitoring():
    @task
    def which_month(triggering_asset_events=None, params=None) -> str:
        """Read the month from the LAKE asset event metadata (set by taxieta_ingest)."""
        months = [ev.extra.get("month") for evs in (triggering_asset_events or {}).values()
                  for ev in evs if ev.extra.get("month")]
        month = sorted(months)[-1] if months else params["month"]
        print(f"checking month {month}")
        return month

    @task.bash
    def drift_check(month: str, params=None) -> str:
        # last stdout line ("true"/"false") becomes this task's XCom return value
        return f"{TAXIETA_BIN} drift-check --reference {params['reference']} --current {month}"

    @task.branch
    def decide(drift: str) -> str:
        return "signal_drift" if drift.strip().lower() == "true" else "no_drift"

    @task(outlets=[DRIFT])
    def signal_drift(month: str):
        print(f"drift detected in {month}; emitting DRIFT asset event")
        yield Metadata(DRIFT, extra={"month": month})

    # TODO(M7): alert with a policy + channel + runbook (Slack webhook / email), and count alerts
    # per day to watch for alert fatigue.
    no_drift = EmptyOperator(task_id="no_drift")

    month = which_month()
    branch = decide(drift_check(month))
    branch >> [signal_drift(month), no_drift]


taxieta_monitoring()
