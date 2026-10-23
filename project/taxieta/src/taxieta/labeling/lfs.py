"""Weak supervision for 'invalid trip' labels with Snorkel (Milestone 3, Ch. 4).

Install:  uv pip install -e ".[weak]"

Two labeling functions (LFs) are written for you; write 4-6 more, then combine them with
Snorkel's LabelModel and compare against ~200 trips you hand-label yourself.
Labeling functions see after-the-fact columns (distance, fare) -- that's fine for LABELING
data quality; it would be leakage if these were model FEATURES.
"""

from __future__ import annotations

ABSTAIN, VALID, INVALID = -1, 0, 1


def _lf(fn):
    try:
        from snorkel.labeling import labeling_function
        return labeling_function()(fn)
    except ImportError:  # keep the module importable without snorkel
        return fn


@_lf
def lf_implied_speed_too_high(x) -> int:
    if x.duration_s > 0 and x.trip_distance / (x.duration_s / 3600) > 80:
        return INVALID
    return ABSTAIN


@_lf
def lf_zero_distance_high_fare(x) -> int:
    if x.trip_distance == 0 and x.fare_amount > 20:
        return INVALID
    return ABSTAIN


# SOLVED(M3): more labeling functions. Each encodes one heuristic; they overlap and conflict on
# purpose -- the LabelModel learns how much to trust each one.
@_lf
def lf_same_zone_very_long(x) -> int:
    """Meter left running: starts and ends in the same zone but lasts over 2 hours."""
    if x.pu_zone == x.do_zone and x.duration_s > 2 * 3600:
        return INVALID
    return ABSTAIN


@_lf
def lf_fare_far_below_minimum(x) -> int:
    """NYC metered fares have a flag drop plus per-minute charges; pennies for a real trip are a glitch."""
    if x.duration_s > 5 * 60 and x.fare_amount < 2.5:
        return INVALID
    return ABSTAIN


@_lf
def lf_long_distance_tiny_duration(x) -> int:
    if x.trip_distance > 20 and x.duration_s < 10 * 60:
        return INVALID
    return ABSTAIN


@_lf
def lf_plausible_manhattan(x) -> int:
    """A positive (VALID) LF: ordinary in-Manhattan trip with plausible speed and fare."""
    if x.duration_s <= 0:
        return ABSTAIN
    mph = x.trip_distance / (x.duration_s / 3600)
    fare_per_min = x.fare_amount / max(x.duration_s / 60, 1)
    if getattr(x, "pu_borough", None) == "Manhattan" and 2 <= mph <= 35 and 0.3 <= fare_per_min <= 5:
        return VALID
    return ABSTAIN


@_lf
def lf_plausible_speed_and_fare(x) -> int:
    if x.duration_s <= 0 or x.trip_distance <= 0:
        return ABSTAIN
    mph = x.trip_distance / (x.duration_s / 3600)
    per_mile = x.fare_amount / x.trip_distance
    if 3 <= mph <= 50 and 1.5 <= per_mile <= 15:
        return VALID
    return ABSTAIN


@_lf
def lf_fare_per_mile_extreme(x) -> int:
    if x.trip_distance > 0.5 and x.fare_amount / x.trip_distance > 40:
        return INVALID
    return ABSTAIN


LFS = [
    lf_implied_speed_too_high, lf_zero_distance_high_fare, lf_same_zone_very_long,
    lf_fare_far_below_minimum, lf_long_distance_tiny_duration, lf_fare_per_mile_extreme,
    lf_plausible_manhattan, lf_plausible_speed_and_fare,
]


def apply_lfs(df):
    """Label matrix L (n_samples x n_LFs) with values in {-1, 0, 1}."""
    from snorkel.labeling import PandasLFApplier

    return PandasLFApplier(lfs=LFS).apply(df, progress_bar=False)


def apply_and_summarize(df, y_true=None):
    """Returns (LabelModel P(INVALID), LF summary). With y_true, the summary includes each LF's
    empirical accuracy (Snorkel's LFAnalysis)."""
    from snorkel.labeling import LFAnalysis
    from snorkel.labeling.model import LabelModel

    L = apply_lfs(df)
    summary = LFAnalysis(L=L, lfs=LFS).lf_summary(Y=y_true)
    lm = LabelModel(cardinality=2, verbose=False)
    lm.fit(L_train=L, n_epochs=500, seed=0, log_freq=500, progress_bar=False)
    return lm.predict_proba(L)[:, INVALID], summary, L


def majority_vote(L):
    from snorkel.labeling.model import MajorityLabelVoter

    return MajorityLabelVoter(cardinality=2).predict(L=L, tie_break_policy="abstain")
