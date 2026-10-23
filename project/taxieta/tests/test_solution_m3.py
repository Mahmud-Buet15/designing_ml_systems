"""M3 solution: imbalance utilities behave as documented."""

import lightgbm as lgb
import numpy as np
import pandas as pd

from taxieta.labeling.handlabel import cohens_kappa
from taxieta.training import longtrip


def test_undersample_keeps_all_positives():
    X = pd.DataFrame({"a": np.arange(1000)})
    y = (np.arange(1000) % 50 == 0).astype(int)
    Xu, yu = longtrip.undersample(X, y, ratio=2)
    assert yu.sum() == y.sum() and len(yu) == 3 * y.sum()


def test_focal_gradient_matches_direction_of_log_loss():
    obj = longtrip.focal_objective(alpha=0.5, gamma=0.0)   # gamma=0, alpha=.5 -> half log loss
    y = np.array([1.0, 0.0])
    x = np.array([0.3, 0.3])
    d = lgb.Dataset(np.zeros((2, 1)), y).construct()
    grad, hess = obj(x, d)
    p = longtrip.sigmoid(x)
    assert np.allclose(grad, 0.5 * (p - y), atol=1e-5) and (hess > 0).all()


def test_cost_threshold_prefers_recall_when_fn_expensive():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(int)
    p = np.clip(0.05 + 0.3 * y + rng.normal(0, 0.1, 5000), 0, 1)
    t_cheap, _ = longtrip.best_threshold_by_cost(y, p, cost_fn=1, cost_fp=1)
    t_costly, _ = longtrip.best_threshold_by_cost(y, p, cost_fn=20, cost_fp=1)
    assert t_costly < t_cheap


def test_kappa():
    assert cohens_kappa(pd.Series([1, 0, 1, 0]), pd.Series([1, 0, 1, 0])) == 1.0
    assert abs(cohens_kappa(pd.Series([1, 1, 0, 0]), pd.Series([1, 0, 1, 0]))) < 1e-9
