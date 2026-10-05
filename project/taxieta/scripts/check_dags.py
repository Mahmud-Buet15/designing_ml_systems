"""Fail if any DAG has import errors (run with Airflow's Python, not the taxieta venv)."""

import sys

from airflow.dag_processing.dagbag import DagBag

EXPECTED = {"taxieta_ingest", "taxieta_monitoring", "taxieta_retrain"}

folder = sys.argv[1] if len(sys.argv) > 1 else "dags"
sys.path.insert(0, folder)   # Airflow puts DAGS_FOLDER on sys.path; do the same here
bag = DagBag(folder)
if bag.import_errors:
    for path, err in bag.import_errors.items():
        print(f"IMPORT ERROR {path}:\n{err}")
    sys.exit(1)
missing = EXPECTED - set(bag.dag_ids)
if missing:
    sys.exit(f"missing DAGs: {missing}")
for dag_id in sorted(bag.dag_ids):
    print(f"ok  {dag_id}: {[t.task_id for t in bag.dags[dag_id].tasks]}")
