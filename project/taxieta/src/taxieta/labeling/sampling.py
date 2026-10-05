"""Sampling methods (Milestone 3, Ch. 4)."""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import TypeVar

import pandas as pd

T = TypeVar("T")


def simple_random(df: pd.DataFrame, frac: float, seed: int = 0) -> pd.DataFrame:
    return df.sample(frac=frac, random_state=seed)


def stratified(df: pd.DataFrame, by: str, frac: float, seed: int = 0) -> pd.DataFrame:
    """Sample `frac` of EACH stratum so rare groups (Staten Island, airports) survive."""
    return df.groupby(by, group_keys=False, dropna=False).sample(frac=frac, random_state=seed)


def convenience(df: pd.DataFrame, first_n_days: int = 5) -> pd.DataFrame:
    """A deliberately biased sample: 'whatever loaded first'. Compare it with the others."""
    day = df["pickup_ts"].dt.day
    return df[day <= first_n_days]


def reservoir(stream: Iterable[T], k: int, seed: int | None = None) -> list[T]:
    """Uniform sample of k items from a stream of unknown length; each kept with prob k/n."""
    rng = random.Random(seed)
    sample: list[T] = []
    for n, item in enumerate(stream, start=1):
        if n <= k:
            sample.append(item)
        else:
            i = rng.randint(1, n)
            if i <= k:
                sample[i - 1] = item
    return sample


def weighted(df: pd.DataFrame, weights: pd.Series, n: int, seed: int = 0) -> pd.DataFrame:
    """TODO(M3): weighted sampling, e.g. recent weeks 2x. Justify the weights in your log,
    and test the 'recent data is more valuable' claim properly in Milestone 8."""
    return df.sample(n=n, weights=weights, random_state=seed)
