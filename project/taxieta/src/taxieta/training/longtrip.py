"""LongTrip side task: P(duration > 60 min) -- class imbalance (M3) and calibration (M5)."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from taxieta import config

BASE_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 30,
               "feature_fraction": 0.9, "verbose": -1, "seed": 7}


def labels(df: pd.DataFrame) -> np.ndarray:
    """Generates binary labels for long trips."""  
    return (df["duration_s"].to_numpy() > config.LONG_TRIP_S).astype(int)


def sigmoid(x):
    from scipy.special import expit

    return expit(x)


def focal_objective(alpha: float = 0.25, gamma: float = 2.0, h: float = 1e-4):
    """Binary focal loss (Lin et al.) as a LightGBM custom objective.

    FL(p) = -[y a (1-p)^g log p + (1-y)(1-a) p^g log(1-p)],  p = sigmoid(raw score).
    Gradient/Hessian w.r.t. the raw score by central finite differences -- exact enough in float64
    and much easier to read than the closed form."""

    def loss(x, y):
        p = np.clip(sigmoid(x), 1e-12, 1 - 1e-12)
        return -(y * alpha * (1 - p) ** gamma * np.log(p) + (1 - y) * (1 - alpha) * p ** gamma * np.log(1 - p))

    def obj(preds, data):
        y = data.get_label()
        lp, l0, lm = loss(preds + h, y), loss(preds, y), loss(preds - h, y)
        grad = (lp - lm) / (2 * h)
        hess = np.maximum((lp - 2 * l0 + lm) / h ** 2, 1e-6)
        return grad, hess

    return obj


def undersample(X: pd.DataFrame, y: np.ndarray, ratio: float = 1.0, seed: int = 0):
    """Keep all positives and `ratio` x as many negatives. ONLY ever applied to the training split."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    keep = np.concatenate([pos, rng.choice(neg, min(len(neg), int(len(pos) * ratio)), replace=False)])
    keep.sort()
    return X.iloc[keep], y[keep]


def train_variant(name: str, X_tr, y_tr, X_va, y_va, rounds: int = 400):
    """Returns a predict_proba(X) function for one imbalance strategy."""
    es = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    if name == "plain":
        # training with early stopping and logging
        b = lgb.train(BASE_PARAMS, lgb.Dataset(X_tr, y_tr), rounds,
                      valid_sets=[lgb.Dataset(X_va, y_va)], callbacks=es)
        return lambda X: b.predict(X, num_iteration=b.best_iteration)   # Returns a lightweight anonymous function (lambda) instead of the raw Booster object b

    if name == "undersampled":
        Xu, yu = undersample(X_tr, y_tr)
        b = lgb.train(BASE_PARAMS, lgb.Dataset(Xu, yu), rounds,
                      valid_sets=[lgb.Dataset(X_va, y_va)], callbacks=es)
        return lambda X: b.predict(X, num_iteration=b.best_iteration)

    if name == "class_weights":
        # calculate scale_pos_weight by dividing the number of negative samples by the number of positive samples
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)         # max(..., 1): Safeguards against division by zero in case y_tr contains no positive samples.
        b = lgb.train({**BASE_PARAMS, "scale_pos_weight": spw}, lgb.Dataset(X_tr, y_tr), rounds,    # Injecting scale_pos_weight into Training. giving more weight to the minority class. Note: if spw=9.0,LightGBM will scale the gradients of the positive class by a factor of 9.0, penalizing false negatives much more heavily than false positives.
                      valid_sets=[lgb.Dataset(X_va, y_va)], callbacks=es)
        return lambda X: b.predict(X, num_iteration=b.best_iteration)
        
    if name == "focal": # trains a LightGBM model with a custom Focal Loss objective and initializes the training using the prior log-odds (base margin) of the training labels.
        # Calculating Prior Log-Odds
        base = float(np.log(max(y_tr.mean(), 1e-6) / max(1 - y_tr.mean(), 1e-6))) # Initializes the model with the log-odds of the training data's base rate
        params = {**BASE_PARAMS, "objective": focal_objective()} # Injecting focal loss into Training. 
        dtr = lgb.Dataset(X_tr, y_tr, init_score=np.full(len(y_tr), base)) # Creating LightGBM Dataset for training with initial scores
        dva = lgb.Dataset(X_va, y_va, init_score=np.full(len(y_va), base)) # Creating LightGBM Dataset for validation with initial scores
        b = lgb.train(params, dtr, rounds, valid_sets=[dva], callbacks=[lgb.log_evaluation(0)]) # Training the model with early stopping and logging
        return lambda X: sigmoid(b.predict(X, raw_score=True) + base)   # Returning Predictor with Base Margin Re-added

    if name == "two_phase": # two-phase transfer learning (pre-train then fine-tune) strategy to combat class imbalance with LightGBM.
        # phase 1(Pre-training phase): training a balanced model on a fraction of the data
        Xu, yu = undersample(X_tr, y_tr)
        b1 = lgb.train(BASE_PARAMS, lgb.Dataset(Xu, yu), 150)
        # phase 2: fine-tune on the ORIGINAL distribution, starting from the balanced model
        b2 = lgb.train(BASE_PARAMS, lgb.Dataset(X_tr, y_tr), 150, init_model=b1,    # init_model=b1: Continues training by taking the 150 trees from b1 as the starting foundation and boosting additional trees on top of them.
                       valid_sets=[lgb.Dataset(X_va, y_va)], callbacks=es)
        return lambda X: b2.predict(X, num_iteration=b2.best_iteration)
    raise ValueError(name)

# check docs/explanations_training_strategy.md for further details


VARIANTS = ["plain", "undersampled", "class_weights", "focal", "two_phase"]


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    yhat = (p >= threshold).astype(int)
    return {
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "pr_auc": average_precision_score(y, p) if y.any() else float("nan"),
        "roc_auc": roc_auc_score(y, p) if 0 < y.mean() < 1 else float("nan"),
        "mean_pred_prob": float(p.mean()),
        "base_rate": float(y.mean()),
    }


def best_threshold_by_cost(y: np.ndarray, p: np.ndarray, cost_fn: float, cost_fp: float) -> tuple[float, float]:
    """Cost-sensitive thresholding (Ch. 4): minimize cost_fn*FN + cost_fp*FP on VALIDATION."""
    grid = np.unique(np.quantile(p, np.linspace(0, 1, 201)))
    costs = [(cost_fn * ((p < t) & (y == 1)).sum() + cost_fp * ((p >= t) & (y == 0)).sum()) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i] / len(y))
