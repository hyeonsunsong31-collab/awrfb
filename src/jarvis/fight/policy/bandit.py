"""Discounted Thompson sampling over tactics.

Each tactic keeps a Normal-posterior estimate of its reward (damage dealt minus damage taken per
period). Old evidence is discounted so the choice tracks THIS opponent as they adapt. Opponent-model
priors seed the arms at the start of a match.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Arm:
    n: float = 0.0  # discounted count
    s: float = 0.0  # discounted reward sum
    ss: float = 0.0  # discounted sum of squares

    @property
    def mean(self) -> float:
        return self.s / self.n if self.n > 0 else 0.0

    @property
    def var(self) -> float:
        if self.n < 2:
            return 0.05
        return max(1e-4, self.ss / self.n - self.mean**2)


class DiscountedThompson:
    def __init__(self, arms: list[str], discount: float = 0.97, prior_sd: float = 0.1, seed: int | None = None) -> None:
        self.arms = {a: Arm() for a in arms}
        self.discount = discount
        self.prior_sd = prior_sd
        self.prior_mean: dict[str, float] = {a: 0.0 for a in arms}
        self.rng = random.Random(seed)

    def set_priors(self, priors: dict[str, float], weight: float = 2.0) -> None:
        """Seed arms with pseudo-observations (from the opponent model / past matches)."""
        for a, m in priors.items():
            if a in self.arms:
                self.prior_mean[a] = m
                arm = self.arms[a]
                arm.n += weight
                arm.s += m * weight
                arm.ss += (m * m + 0.01) * weight

    def choose(self) -> str:
        best, best_v = None, -math.inf
        for a, arm in self.arms.items():
            if arm.n <= 0:
                mu, sd = self.prior_mean[a], self.prior_sd
            else:
                mu = arm.mean
                sd = math.sqrt(arm.var / arm.n + (self.prior_sd**2) / (1 + arm.n))
            v = self.rng.gauss(mu, sd)
            if v > best_v:
                best, best_v = a, v
        assert best is not None
        return best

    def update(self, arm: str, reward: float) -> None:
        for a in self.arms.values():
            a.n *= self.discount
            a.s *= self.discount
            a.ss *= self.discount
        x = self.arms[arm]
        x.n += 1
        x.s += reward
        x.ss += reward * reward

    def summary(self) -> dict[str, dict[str, float]]:
        return {a: {"n": round(x.n, 2), "mean": round(x.mean, 4)} for a, x in self.arms.items()}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({a: asdict(x) for a, x in self.arms.items()}), encoding="utf-8")

    def load(self, path: Path, weight: float = 0.3) -> None:
        """Carry over (down-weighted) knowledge from previous matches."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for a, d in data.items():
            if a in self.arms:
                self.arms[a] = Arm(d["n"] * weight, d["s"] * weight, d["ss"] * weight)
