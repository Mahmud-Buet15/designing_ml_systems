"""Milestone 3 solution: sampling, labels, weak supervision, class imbalance, active learning.

    taxieta solve m3 [--train 2019-01:2019-03 --valid 2019-04 --test 2019-05]
    taxieta solve m3 --export-for-labeling      # writes data/labels/to_label.csv (200 rows)
"""

from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from taxieta.ingest.download import parse_months
from taxieta.ingest.lake import read_lake
from taxieta.labeling import handlabel, lfs
from taxieta.labeling.join_labels import feedback_loop_report, join_predictions_with_labels
from taxieta.labeling.sampling import (
    convenience,
    recency_weights,
    reservoir,
    simple_random,
    stratified,
    weighted,
)
from taxieta.solutions.common import Report, out_dir
from taxieta.training import longtrip
from taxieta.training.data import prepare_splits


def tv_distance(a: pd.Series, b: pd.Series) -> float:
    """Total variation distance between two discrete distributions (Ch. 3)."""
    # 1. Compute empirical probability distributions
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)

    # 2. Get the unified set of all unique categories
    idx = pa.index.union(pb.index)

    # 3. Align indices, fill missing categories with 0, compute TVD formula
    return float(0.5 * (pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).abs().sum())

# Note: check docs/explanations.md for detailed explanation of this function


def part_a_sampling(full: pd.DataFrame, rep: Report) -> None:
    rep.h("A. Sampling (5% samples vs the full population)")
    samples = {
        "simple random": simple_random(full, 0.05, seed=1),
        "stratified by pu_borough": stratified(full, "pu_borough", 0.05, seed=1),
        "convenience (first 5 days/month)": convenience(full, 5),
    }
    rows = []
    for name, s in samples.items():
        rows.append({
            "sample": name, "rows": len(s),
            "borough_tv_distance": tv_distance(full["pu_borough"], s["pu_borough"]), # getting tv distance for categorical variables
            "staten_island_rows": int((s["pu_borough"] == "Staten Island").sum()), # getting number of rows in staten island
            "duration_ks_stat": float(ks_2samp(full["duration_s"], s["duration_s"]).statistic), # getting distribution similarity for continuous variables
            "weekend_share_pct": float((s["pickup_ts"].dt.dayofweek >= 5).mean() * 100), # getting weekend share percentage
        })
    rows.append({"sample": "FULL", "rows": len(full), "borough_tv_distance": 0.0,
                 "staten_island_rows": int((full["pu_borough"] == "Staten Island").sum()),
                 "duration_ks_stat": 0.0, "weekend_share_pct": float((full["pickup_ts"].dt.dayofweek >= 5).mean() * 100)})
    rep.table(pd.DataFrame(rows), key="sampling")
    rep.p("Convenience sampling skews calendar composition (weekday/weekend mix, holidays, "
          "month-start effects): selection bias you never chose. Stratified guarantees the rare "
          "strata; simple random is unbiased but can nearly miss them.")

    w = recency_weights(full["pickup_ts"])
    ws = weighted(full, w, n=len(full) // 20, seed=1)   # getting weighted samples  
    recent = full["pickup_ts"] >= full["pickup_ts"].max() - pd.Timedelta(weeks=2)   # checking recent weeks data in full dataset
    rep.p(f"Weighted (recent 2 weeks x2): recent share {recent.mean():.1%} in population -> "
          f"{(ws['pickup_ts'] >= full['pickup_ts'].max() - pd.Timedelta(weeks=2)).mean():.1%} in sample.")  # checking recent weeks data in weighted sample dataset

    stream = full.sort_values("pickup_ts")["pickup_hour"].tolist()
    k = min(10_000, len(stream) // 3) # getting number of samples to be taken from the stream
    res = pd.Series(reservoir(stream, k=k, seed=3))
    rand = full["pickup_hour"].sample(k, random_state=3) # getting random samples from the full dataset
    rep.p(f"Reservoir (k={k}) hour-of-day TV distance to population: {tv_distance(full['pickup_hour'], res):.4f} "
          f"(a simple random sample of the same size: {tv_distance(full['pickup_hour'], rand):.4f}). "
          "Both are ~0: uniform without knowing the stream length.")


def part_b_labels(may: pd.DataFrame, rep: Report) -> None:
    rep.h("B1. Natural labels: label computation and feedback delay")
    preds = may[["trip_id", "pickup_ts", "pu_zone", "do_zone"]].assign(pickup_ts=lambda d: d["pickup_ts"].astype(str))
    comps = may[["trip_id", "dropoff_ts", "duration_s"]].assign(dropoff_ts=lambda d: d["dropoff_ts"].astype(str))
    joined = join_predictions_with_labels(preds, comps)
    rep.table(pd.DataFrame([feedback_loop_report(joined)]), key="feedback_delay")
    rows = []
    for w in ["15min", "30min", "60min", "3h"]:
        j = join_predictions_with_labels(preds, comps, label_window=pd.Timedelta(w))
        rows.append({"label_window": w, "labeled_pct": 100 * len(j) / len(joined),
                     "mean_duration_min_labeled": j["duration_s"].mean() / 60,
                     "mean_duration_min_all": joined["duration_s"].mean() / 60})
    rep.table(pd.DataFrame(rows), key="label_window")
    rep.p("Short windows give fast labels but silently drop the LONG trips: the labeled set's mean "
          "duration is biased low, and so would be a model retrained on it.")


def part_b_weak_supervision(trips: pd.DataFrame, rep: Report, export: bool) -> None:
    rep.h("B2. Weak supervision: 'invalid trip' labels with Snorkel")
    p_invalid, summary, L = lfs.apply_and_summarize(trips)
    if export:
        handlabel.export_for_labeling(trips, L)
        rep.p(f"exported rows to label -> {handlabel.LABELS_DIR / 'to_label.csv'}")
    hand = handlabel.load_hand_labels()
    if hand is not None:
        idx = trips.index[trips["trip_id"].isin(hand["trip_id"])]
        y = trips.loc[idx, "trip_id"].map(hand.set_index("trip_id")["label"]).to_numpy()
        source = f"your hand labels (n={len(y)})"
    else:
        truth = handlabel.fixture_truth(trips)
        if truth is None:
            rep.p("No hand labels yet: run with --export-for-labeling, label ~200 rows, save "
                  "data/labels/hand_labels.csv, rerun. Showing coverage/conflicts only.")
            rep.table(summary.reset_index(names="lf"), key="lf_summary")
            return
        idx, y = trips.index, truth.to_numpy()
        source = "fixture ground truth (stand-in for hand labels)"
    from snorkel.labeling import LFAnalysis

    pos = np.searchsorted(trips.index, idx)
    L_eval = L[pos]
    summary_y = LFAnalysis(L=L_eval, lfs=lfs.LFS).lf_summary(Y=y)
    rep.p(f"Evaluated against {source}; invalid rate {y.mean():.2%}")
    rep.table(summary_y.reset_index(names="lf"), key="lf_summary")

    mv = lfs.majority_vote(L_eval)
    rows = []
    for name, pred in (("majority vote", np.where(mv == -1, 0, mv)),
                       ("LabelModel (p>0.5)", (p_invalid[pos] > 0.5).astype(int))):
        tp = int(((pred == 1) & (y == 1)).sum())
        rows.append({"combiner": name, "precision": tp / max((pred == 1).sum(), 1),
                     "recall": tp / max((y == 1).sum(), 1), "flagged": int((pred == 1).sum())})
    rep.table(pd.DataFrame(rows), key="ws_combiners")

    h2 = handlabel.load_hand_labels("hand_labels_2.csv")
    if hand is not None and h2 is not None:
        m = hand.merge(h2, on="trip_id", suffixes=("_1", "_2"))
        rep.p(f"Label multiplicity: {len(m)} rows labeled twice, raw agreement "
              f"{(m['label_1'] == m['label_2']).mean():.1%}, Cohen's kappa {handlabel.cohens_kappa(m['label_1'], m['label_2']):.2f}")


def part_c_imbalance(splits, rep: Report) -> dict:
    rep.h("C. LongTrip (> 60 min): class imbalance")
    y_tr, y_va, y_te = (longtrip.labels(getattr(splits, s)) for s in ("train", "valid", "test"))   # getting the labels for the training, validation, and test sets
    rep.p(f"positive rate train {y_tr.mean():.2%}, test {y_te.mean():.2%}; zero-rule accuracy on test "
          f"{1 - y_te.mean():.2%} with recall 0 -- accuracy is useless here.")
    rows, probs = [], {}
    for v in longtrip.VARIANTS:    # iterating over the different variants of the model
        f = longtrip.train_variant(v, splits.X_train, y_tr, splits.X_valid, y_va)   # training the model with the current variant
        probs[v] = (f(splits.X_valid), f(splits.X_test))    # getting the probabilities for the validation and test sets
        rows.append({"variant": v, **longtrip.evaluate(y_te, probs[v][1])})   # evaluating the model with the current variant
    res = pd.DataFrame(rows)    # creating a dataframe with the results
    rep.table(res, key="imbalance")
    rep.p("Compare PR-AUC (threshold-free, minority-focused) first; precision/recall at 0.5 depend on "
          "calibration. mean_pred_prob >> base_rate for 'undersampled' = probabilities inflated.")

    best = res.sort_values("pr_auc", ascending=False)["variant"].iloc[0]
    _plot_curves(y_te, probs[best][1], best)
    thr, c_va = longtrip.best_threshold_by_cost(y_va, probs[best][0], cost_fn=10, cost_fp=1)
    p_te = probs[best][1]
    c_te = (10 * ((p_te < thr) & (y_te == 1)).sum() + ((p_te >= thr) & (y_te == 0)).sum()) / len(y_te)
    c_05 = (10 * ((p_te < 0.5) & (y_te == 1)).sum() + ((p_te >= 0.5) & (y_te == 0)).sum()) / len(y_te)
    rep.p(f"Cost-sensitive threshold for '{best}' (FN costs 10x FP), chosen on VALIDATION: {thr:.3f}. "
          f"Test cost/trip {c_te:.4f} vs {c_05:.4f} at 0.5.")
    return {"best": best}


def _plot_curves(y, p, name) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(y, p)
    pr, rc, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].plot(fpr, tpr)
    ax[0].plot([0, 1], [0, 1], "--", color="grey")
    ax[0].set(title=f"ROC ({name})", xlabel="FPR", ylabel="TPR")
    ax[1].plot(rc, pr)
    ax[1].axhline(y.mean(), ls="--", color="grey", label="random = base rate")
    ax[1].set(title=f"Precision-Recall ({name})", xlabel="recall", ylabel="precision")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir() / "m3_longtrip_curves.png", dpi=110)
    plt.close(fig)


def part_d_active_learning(splits, rep: Report, rounds: int = 5, start: int = 1000, step: int = 500) -> None:
    rep.h("D. (Stretch) Active learning: uncertainty vs random")
    X, y = splits.X_train, longtrip.labels(splits.train)
    Xt, yt = splits.X_test, longtrip.labels(splits.test)
    rng = np.random.default_rng(0)
    init = rng.choice(len(y), start, replace=False)
    rows = []
    for strategy in ("random", "uncertainty"):
        labeled = set(init.tolist())
        for r in range(rounds + 1):
            idx = np.array(sorted(labeled))
            b = lgb.train({**longtrip.BASE_PARAMS, "min_data_in_leaf": 5}, lgb.Dataset(X.iloc[idx], y[idx]), 150)
            ap = longtrip.evaluate(yt, b.predict(Xt))["pr_auc"]
            rows.append({"strategy": strategy, "round": r, "labels": len(idx), "test_pr_auc": ap})
            pool = np.setdiff1d(np.arange(len(y)), idx)
            if strategy == "random":
                pick = rng.choice(pool, step, replace=False)
            else:
                p = b.predict(X.iloc[pool])
                ent = -(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12))
                pick = pool[np.argsort(-ent)[:step]]
            labeled.update(pick.tolist())
    df = pd.DataFrame(rows)
    rep.table(df.pivot(index="labels", columns="strategy", values="test_pr_auc").reset_index(), key="active_learning")
    rep.p("Uncertainty sampling usually climbs faster on rare classes: it spends labels near the "
          "decision boundary instead of on thousands of obvious short trips.")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(prog="taxieta solve m3")
    ap.add_argument("--train", default="2019-01:2019-03")
    ap.add_argument("--valid", default="2019-04")
    ap.add_argument("--test", default="2019-05")
    ap.add_argument("--export-for-labeling", action="store_true")
    ap.add_argument("--skip-active-learning", action="store_true")
    a = ap.parse_args(argv)
    import warnings
    warnings.filterwarnings("ignore")

    rep = Report("m3", "Milestone 3 -- training data")
    full = read_lake(parse_months(a.train) + parse_months(a.valid))  # loads all data for training and validation
    part_a_sampling(full, rep)    # performs sampling on the training data
    part_b_labels(read_lake(parse_months(a.test)), rep)  # loads all data for testing and performs labeling
    part_b_weak_supervision(full.reset_index(drop=True), rep, a.export_for_labeling)  # performs weak supervision on the training data
    splits = prepare_splits(a.train, a.valid, a.test)  # prepares the splits for the training data
    part_c_imbalance(splits, rep)  # performs class imbalance analysis on the training data
    if not a.skip_active_learning:
        part_d_active_learning(splits, rep)  # performs active learning on the training data
    rep.save()
    return rep.data
