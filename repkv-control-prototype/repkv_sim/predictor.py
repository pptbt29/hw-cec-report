"""Class-conditional return prediction.

The controller is given the think-time distribution of a session class, never
the realized draw. From that it answers two questions about a session that has
been idle for some time without returning: how likely is a return inside the
next window, and how much longer is the wait likely to be.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _lognormal_cdf(x: float, median: float, sigma: float) -> float:
    if x <= 0 or median <= 0 or sigma <= 0:
        return 0.0
    return 0.5 * (1.0 + math.erf(math.log(x / median) / (sigma * math.sqrt(2.0))))


@dataclass(frozen=True)
class ReturnModel:
    """Think-time distribution plus the chance the session returns at all.

    `continue_probability` carries the mass at infinity: a session that will
    never send another turn is a right-censored sample, so waiting longer
    without a return is evidence both that the return is close and that it may
    never happen. Both effects appear in `window_probability`.
    """

    median_gap_s: float
    sigma: float
    continue_probability: float
    floor_s: float
    ceiling_s: float

    def think_cdf(self, x: float) -> float:
        """CDF of the think time, conditional on the session returning."""
        if x < self.floor_s:
            return 0.0
        if x >= self.ceiling_s:
            return 1.0
        return _lognormal_cdf(x, self.median_gap_s, self.sigma)

    def window_probability(self, waited_s: float, window_s: float) -> float:
        """q(t, delta) for a session idle for `waited_s` without returning."""
        if window_s <= 0:
            return 0.0
        waited = max(0.0, waited_s)
        survived = 1.0 - self.continue_probability * self.think_cdf(waited)
        if survived <= 1e-12:
            return 0.0
        gained = self.continue_probability * (
            self.think_cdf(waited + window_s) - self.think_cdf(waited)
        )
        return min(1.0, max(0.0, gained / survived))

    def median_residual_s(self, waited_s: float) -> float:
        """Median remaining wait, conditional on the session returning.

        Used as the single time at which node queues and TTFT are evaluated.
        It is always non-negative, so a session that has already waited longer
        than its median gap is treated as returning soon rather than as having
        missed a deadline.
        """
        waited = max(0.0, waited_s)
        base = self.think_cdf(waited)
        if base >= 1.0:
            return 0.0
        target = base + 0.5 * (1.0 - base)
        low, high = waited, self.ceiling_s
        if self.think_cdf(high) < target:
            return max(0.0, high - waited)
        for _ in range(40):
            middle = 0.5 * (low + high)
            if self.think_cdf(middle) < target:
                low = middle
            else:
                high = middle
        return max(0.0, 0.5 * (low + high) - waited)


@dataclass(frozen=True)
class SessionClass:
    """Workload-side description of one session class."""

    name: str
    median_gap_s: float
    base_return: float

    def model(self, completed_turns: int, sigma: float, floor_s: float, ceiling_s: float) -> ReturnModel:
        return ReturnModel(
            median_gap_s=self.median_gap_s,
            sigma=sigma,
            continue_probability=max(0.18, self.base_return - 0.045 * completed_turns),
            floor_s=floor_s,
            ceiling_s=ceiling_s,
        )
