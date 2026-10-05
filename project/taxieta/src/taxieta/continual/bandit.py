"""Thompson sampling over models (Milestone 8, Ch. 9 "Bandits").

Reward = 1 if a prediction lands within +/-20% of the actual duration. Rewards arrive only
when trip_completed does (a short, but real, feedback loop).

TODO(M8): wire this into serving (choose the arm per request, log it in the prediction log's
`arm` column) and into the label joiner (call `update` when labels arrive). Compare trips
needed to find the best model vs. your A/B test.
"""

from __future__ import annotations

import numpy as np


class ThompsonSampler:
    def __init__(self, arms: list[str], seed: int | None = None):
        self.arms = list(arms)
        self.alpha = np.ones(len(arms))   # successes + 1
        self.beta = np.ones(len(arms))    # failures + 1
        self.rng = np.random.default_rng(seed)

    def choose(self) -> str:
        return self.arms[int(np.argmax(self.rng.beta(self.alpha, self.beta)))]

    def update(self, arm: str, reward: int) -> None:
        i = self.arms.index(arm)
        self.alpha[i] += reward
        self.beta[i] += 1 - reward

    def means(self) -> dict[str, float]:
        return dict(zip(self.arms, (self.alpha / (self.alpha + self.beta)).tolist(), strict=True))
