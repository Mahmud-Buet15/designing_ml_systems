"""DAG 1 -- ingest a month into the lake (Milestones 2 and 9).

Trigger manually with a month param (simulated time), e.g. {"month": "2020-04"}.
On success it updates the LAKE asset (with the month as metadata), which triggers monitoring.

    download -> ingest -> data_quality_gate -> (LAKE asset event)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pendulum
from airflow.sdk import Metadata, Param, dag, task
from taxieta_common import DEFAULT_ARGS, LAKE, TAXIETA_BIN

REPORTS = Path(os.environ.get("TAXIETA_REPORTS", "/opt/taxieta/reports"))


@dag(
    dag_id="taxieta_ingest",
    schedule=None,                       # TODO(M9): try "@monthly" with real dates instead
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    params={
        "month": Param("2019-05", type="string", pattern=r"^\d{4}-\d{2}$"),
        "source": Param("download", enum=["download", "fixture"],
                        description="fixture = synthetic data when you're offline"),
        "sample_frac": Param(0.1, type="number", minimum=0.001, maximum=1.0),
    },
    tags=["taxieta", "data"],
)
def taxieta_ingest():
    @task.bash
    def fetch(params=None) -> str:
        cmd = "download" if params["source"] == "download" else "fixture"
        return f"{TAXIETA_BIN} {cmd} --months {params['month']}"

    @task.bash
    def ingest(params=None) -> str:
        return f"{TAXIETA_BIN} ingest --months {params['month']} --sample-frac {params['sample_frac']}"

    @task(outlets=[LAKE])
    def data_quality_gate(params=None, max_reject_pct: float = 5.0):
        """Fail loudly if too many rows were rejected -- the 'raw inputs' monitor (Ch. 8)."""
        rep = json.loads((REPORTS / "ingest" / f"{params['month']}.json").read_text())
        reject_pct = 100 * (1 - rep["rows_out"] / max(rep["rows_in"], 1))
        print(f"rejected {reject_pct:.2f}% of rows: {rep}")
        if reject_pct > max_reject_pct:
            raise ValueError(f"reject rate {reject_pct:.1f}% > {max_reject_pct}%: check upstream data")
        # Attach the month to the asset event so downstream DAGs know what changed.
        yield Metadata(LAKE, extra={"month": params["month"]})

    fetch() >> ingest() >> data_quality_gate()


taxieta_ingest()
