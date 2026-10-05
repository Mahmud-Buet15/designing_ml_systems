"""Weak supervision for 'invalid trip' labels with Snorkel (Milestone 3, Ch. 4).

Install:  uv pip install -e ".[weak]"

Two labeling functions are written for you; write 4-6 more, then combine them with
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


# TODO(M3): add more LFs, e.g.
#   - same pickup/dropoff zone but duration > 2 h            -> INVALID
#   - fare far below the minimum for its duration            -> INVALID
#   - ordinary Manhattan trip with plausible speed and fare  -> VALID
LFS = [lf_implied_speed_too_high, lf_zero_distance_high_fare]


def apply_and_summarize(df):  # pragma: no cover - requires snorkel
    """Returns (LabelModel probabilities, LF summary)."""
    from snorkel.labeling import LFAnalysis, PandasLFApplier
    from snorkel.labeling.model import LabelModel

    L = PandasLFApplier(lfs=LFS).apply(df)
    summary = LFAnalysis(L=L, lfs=LFS).lf_summary()
    lm = LabelModel(cardinality=2, verbose=False)
    lm.fit(L_train=L, n_epochs=200, seed=0)
    return lm.predict_proba(L), summary
