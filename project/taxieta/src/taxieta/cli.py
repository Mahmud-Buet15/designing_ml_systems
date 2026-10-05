"""`taxieta <command>` -- one entry point for your laptop, Docker and Airflow tasks alike.

Run `taxieta -h` or `taxieta <command> -h` for options.
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="taxieta", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("fixture", help="write synthetic TLC-shaped data (offline/CI)")
    s.add_argument("--months", default="2019-01:2019-06,2020-01:2020-06")
    s.add_argument("--rows", type=int, default=20_000)

    s = sub.add_parser("download", help="download real TLC Parquet files + zone lookup")
    s.add_argument("--months", default="2019-01:2019-06,2020-01:2020-06")
    s.add_argument("--force", action="store_true")

    s = sub.add_parser("ingest", help="validate + transform + write the lake")
    s.add_argument("--months", default="2019-01:2019-06,2020-01:2020-06")
    s.add_argument("--sample-frac", type=float, default=None,
                   help="simple random down-sample while developing, e.g. 0.05")

    s = sub.add_parser("query", help="run the DuckDB OLAP queries over the lake")
    s.add_argument("name", nargs="?")

    s = sub.add_parser("replay", help="stream a month of trips as live events")
    s.add_argument("--month", required=True)
    s.add_argument("--speedup", type=float, default=600.0)
    s.add_argument("--sink", choices=["kafka", "print", "http"], default="kafka")
    s.add_argument("--limit", type=int)
    s.add_argument("--no-sleep", action="store_true")

    sub.add_parser("consume", help="streaming zone-speed feature consumer (Kafka)")

    s = sub.add_parser("train", help="train + register a model version")
    s.add_argument("--train", default="2019-01:2019-03")
    s.add_argument("--valid", default="2019-04")
    s.add_argument("--test", default="2019-05")
    s.add_argument("--parent")

    s = sub.add_parser("batch-predict", help="precompute ETAs for popular zone pairs")
    s.add_argument("--top-pairs", type=int, default=1000)
    s.add_argument("--reference-month", default="2019-04")

    s = sub.add_parser("drift-check", help="compare feature/label distributions")
    s.add_argument("--reference", required=True)
    s.add_argument("--current", required=True)

    s = sub.add_parser("gate", help="evaluation gates for a challenger")
    s.add_argument("--challenger", required=True)
    s.add_argument("--backtest", required=True)

    s = sub.add_parser("promote", help="make a version the champion")
    s.add_argument("--challenger", required=True)

    sub.add_parser("champion", help="print the current champion's metadata")

    s = sub.add_parser("serve", help="run the prediction API")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)

    a = p.parse_args(argv)

    if a.cmd == "fixture":
        from taxieta.ingest.download import parse_months
        from taxieta.ingest.fixture import write_fixture
        write_fixture(parse_months(a.months), n=a.rows)
    elif a.cmd == "download":
        from taxieta.ingest.download import download_months, parse_months
        download_months(parse_months(a.months), force=a.force)
    elif a.cmd == "ingest":
        from taxieta.ingest.download import parse_months
        from taxieta.ingest.lake import ingest_month
        for y, m in parse_months(a.months):
            ingest_month(y, m, sample_frac=a.sample_frac)
    elif a.cmd == "query":
        from taxieta.lake_queries import run
        run(a.name)
    elif a.cmd == "replay":
        from taxieta.stream.replayer import replay
        replay(a.month, a.speedup, a.sink, a.limit, a.no_sleep)
    elif a.cmd == "consume":
        from taxieta.stream.consumer import run_consumer
        run_consumer()
    elif a.cmd == "train":
        from taxieta.training.train import train
        out = train(a.train, a.valid, a.test, parent=a.parent)
        print(out["version"])            # last line = machine-readable (Airflow XCom)
    elif a.cmd == "batch-predict":
        from taxieta.serving.batch import batch_predict
        batch_predict(a.top_pairs, a.reference_month)
    elif a.cmd == "drift-check":
        from taxieta.monitoring.drift import check
        rep = check(a.reference, a.current)
        print(str(rep["drift_detected"]).lower())
    elif a.cmd == "gate":
        from taxieta.continual.gates import run_gates
        decision = run_gates(a.challenger, a.backtest)
        print(str(decision["promote"]).lower())
    elif a.cmd == "promote":
        from taxieta.continual.gates import promote
        promote(a.challenger)
    elif a.cmd == "champion":
        from taxieta import model_store
        ch = model_store.load_champion()
        print(json.dumps(ch["meta"] if ch else None, indent=2, default=str))
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("taxieta.serving.app:app", host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
