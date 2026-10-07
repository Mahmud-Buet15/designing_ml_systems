"""Milestone 1 solution: the numbers the design doc needs.

    taxieta solve m1 [--history 2019-01:2019-04] [--eval 2019-05]

* Heuristic ETA (Phase 1, no ML) MAE and % within +/-20% on the evaluation month -- the bar
  every later model must clear.
* Peak load estimate (requests/s) from the busiest minute, scaled back up if you ingested a
  sample -- feeds the scalability requirement and the load test in M6.
* Label facts that shape the framing: duration percentiles and the LongTrip positive rate.
"""

from __future__ import annotations

import argparse

import pandas as pd

from taxieta import config
from taxieta.ingest.download import parse_months
from taxieta.ingest.lake import read_lake
from taxieta.solutions.common import Report, sample_frac_used
from taxieta.training import metrics
from taxieta.training.baselines import HeuristicETA, ZeroRule


def main(argv: list[str] | None = None) -> dict:
    # creating parser for sub command: solve m1
    ap = argparse.ArgumentParser(prog="taxieta solve m1")
    ap.add_argument("--history", default="2019-01:2019-04")  # optional argument
    ap.add_argument("--eval", default="2019-05")  # optional argument

    # parse arguments
    a = ap.parse_args(argv)         # This parses the list argv and returns a Namespace object. You read the values as attributes: a.history and a.eval.

    # parse months and read lake
    hist_m, eval_m = parse_months(a.history), parse_months(a.eval)  
    history = read_lake(hist_m)
    ev = read_lake(eval_m)

    # creating report
    rep = Report("m1", "Milestone 1 -- framing numbers")  # report instance

    # 1. Baselines on the evaluation month
    heur = HeuristicETA().fit(history)  # heuristic model
    zero = ZeroRule().fit(history["duration_s"])  # zero rule model
    y = ev["duration_s"].to_numpy()  # numpy array of duration
    rows = [
        {"model": "zero rule (median)", **metrics.report(y, zero.predict(len(ev)))},
        {"model": "heuristic (zone pair x hour-of-week)", **metrics.report(y, heur.predict(ev))},
    ]
    rep.h(f"Baselines on {a.eval} (history {a.history})")
    rep.table(pd.DataFrame(rows), key="baselines")

    # 2. Peak load -> requests per second (one prediction per trip start)
    y0, m0 = eval_m[0]
    frac = sample_frac_used(y0, m0)
    per_min = ev.set_index("pickup_ts").resample("1min").size()    # number of trips starting in each minute
    per_hour = ev.set_index("pickup_ts").resample("1h").size()   # number of trips starting in each hour
    load = {
        "sample_frac": frac,
        "busiest_minute": str(per_min.idxmax()),  # the minute with the most trips
        "peak_rps_minute": float(per_min.max() / frac / 60),  # peak requests per second
        "peak_rps_hour_avg": float(per_hour.max() / frac / 3600),
        "mean_rps": float(len(ev) / frac / ((ev["pickup_ts"].max() - ev["pickup_ts"].min()).total_seconds())),
    }
    rep.h("Load estimate (scaled to 100% of trips)")
    rep.table(pd.DataFrame([load]), key="load")
    rep.p("Size the service for the peak minute plus headroom (2-3x), not the average.")

    # 3. Label facts that shape framing
    d = ev["duration_s"] / 60
    facts = {
        "p50_min": d.quantile(0.5), "p90_min": d.quantile(0.9), "p99_min": d.quantile(0.99),
        "skew": float(d.skew()),
        "longtrip_rate_pct": float((ev["duration_s"] > config.LONG_TRIP_S).mean() * 100),
    }
    rep.h("Label distribution")
    rep.table(pd.DataFrame([facts]), key="label_facts")
    rep.p("Right-skewed durations -> predict log(duration) or use MAE/Huber; LongTrip is a rare "
          "class -> accuracy is useless there (M3).")
    rep.save()
    return rep.data
