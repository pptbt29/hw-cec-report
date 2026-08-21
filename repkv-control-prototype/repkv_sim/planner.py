"""Pure KV-prefix recovery and preparation planning logic.

The portable question answered here is: given only causally available prefix
locations, what is the smallest HBM prefix that makes one node TTFT-SLO
feasible, and what contiguous transfer/restore/recompute work can produce it
before the predicted return time?
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass


METHODS = ("restore", "transfer", "recompute")


@dataclass(frozen=True)
class WorkSegment:
    method: str
    start: int
    end: int

    @property
    def blocks(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class PrefixAvailability:
    local_hbm: int
    local_host: int
    remote_hbm: int


@dataclass(frozen=True)
class RecoveryPlan:
    segments: tuple[WorkSegment, ...]
    seconds: float

    def blocks(self, method: str) -> int:
        return sum(segment.blocks for segment in self.segments if segment.method == method)


@dataclass(frozen=True)
class PreparationPlan:
    target_prefix: int
    segments: tuple[WorkSegment, ...]
    seconds: float
    normalized_cost: float

    def blocks(self, method: str) -> int:
        return sum(segment.blocks for segment in self.segments if segment.method == method)


def _regions(start: int, end: int, availability: PrefixAvailability) -> list[tuple[int, int, tuple[str, ...]]]:
    if not 0 <= start <= end:
        raise ValueError(f"invalid prefix interval [{start}, {end})")
    boundaries = {start, end}
    for boundary in (availability.local_host, availability.remote_hbm):
        if start < boundary < end:
            boundaries.add(boundary)
    ordered = sorted(boundaries)
    regions: list[tuple[int, int, tuple[str, ...]]] = []
    for left, right in zip(ordered, ordered[1:]):
        methods = ["recompute"]
        if right <= availability.local_host:
            methods.append("restore")
        if right <= availability.remote_hbm:
            methods.append("transfer")
        regions.append((left, right, tuple(methods)))
    return regions


def _compress(segments: list[WorkSegment]) -> tuple[WorkSegment, ...]:
    compressed: list[WorkSegment] = []
    for segment in segments:
        if segment.blocks <= 0:
            continue
        if compressed and compressed[-1].method == segment.method and compressed[-1].end == segment.start:
            previous = compressed[-1]
            compressed[-1] = WorkSegment(previous.method, previous.start, segment.end)
        else:
            compressed.append(segment)
    return tuple(compressed)


def fastest_recovery(
    start_prefix: int,
    context_blocks: int,
    availability: PrefixAvailability,
    foreground_rates: dict[str, float],
) -> RecoveryPlan:
    """Return the fastest strict-source recovery of [start_prefix, context)."""
    segments: list[WorkSegment] = []
    seconds = 0.0
    for left, right, methods in _regions(start_prefix, context_blocks, availability):
        method = max(methods, key=lambda name: foreground_rates[name])
        segments.append(WorkSegment(method, left, right))
        seconds += (right - left) / foreground_rates[method]
    return RecoveryPlan(_compress(segments), seconds)


def cheapest_preparation(
    start_prefix: int,
    target_prefix: int,
    availability: PrefixAvailability,
    time_limit_s: float,
    background_rates: dict[str, float],
    method_costs: dict[str, float],
) -> PreparationPlan | None:
    """Enumerate region-level hybrid plans and choose the cheapest timely one.

    Source coverage is checked per contiguous region. Recompute is always
    available because the simulator retains source tokens.
    """
    if target_prefix <= start_prefix:
        return PreparationPlan(start_prefix, (), 0.0, 0.0)
    if time_limit_s <= 0:
        return None
    regions = _regions(start_prefix, target_prefix, availability)
    best: tuple[float, float, tuple[WorkSegment, ...]] | None = None
    for choices in itertools.product(*(methods for _, _, methods in regions)):
        segments = tuple(
            WorkSegment(method, left, right)
            for (left, right, _), method in zip(regions, choices)
        )
        seconds = sum(segment.blocks / background_rates[segment.method] for segment in segments)
        if seconds > time_limit_s + 1e-9:
            continue
        cost = sum(segment.blocks * method_costs[segment.method] for segment in segments)
        candidate = (cost, seconds, _compress(list(segments)))
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        return None
    return PreparationPlan(target_prefix, best[2], best[1], best[0])


def minimum_slo_preparation(
    context_blocks: int,
    availability: PrefixAvailability,
    queue_s: float,
    prompt_s: float,
    slo_s: float,
    time_limit_s: float,
    foreground_rates: dict[str, float],
    background_rates: dict[str, float],
    method_costs: dict[str, float],
) -> PreparationPlan | None:
    """Find the minimum prepared prefix that crosses the hard TTFT boundary."""
    current = fastest_recovery(availability.local_hbm, context_blocks, availability, foreground_rates)
    if queue_s + prompt_s + current.seconds <= slo_s + 1e-9:
        return None
    if queue_s + prompt_s > slo_s + 1e-9:
        return None
    for target in range(availability.local_hbm + 1, context_blocks + 1):
        hypothetical = PrefixAvailability(target, max(target, availability.local_host), availability.remote_hbm)
        residual = fastest_recovery(target, context_blocks, hypothetical, foreground_rates)
        if queue_s + prompt_s + residual.seconds > slo_s + 1e-9:
            continue
        plan = cheapest_preparation(
            availability.local_hbm,
            target,
            availability,
            time_limit_s,
            background_rates,
            method_costs,
        )
        if plan is not None:
            return plan
    return None


def success_probability(ttft_s: float, slo_s: float, uncertainty_s: float) -> float:
    z = (ttft_s - slo_s) / max(uncertainty_s, 1e-9)
    if z >= 35:
        return 0.0
    if z <= -35:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def at_least_one_success(probabilities: list[float]) -> float:
    failure = 1.0
    for probability in probabilities:
        failure *= 1.0 - min(1.0, max(0.0, probability))
    return 1.0 - failure


# Conditional means of the five equal-probability strata of a standard normal.
SHARED_STRATA: tuple[tuple[float, float], ...] = (
    (-1.3998, 0.2),
    (-0.5244, 0.2),
    (0.0, 0.2),
    (0.5244, 0.2),
    (1.3998, 0.2),
)


def routable_probability(
    ttfts: list[float],
    slo_s: float,
    shared_uncertainty_s: float,
    node_uncertainty_s: float,
) -> float:
    """Probability that at least one node meets the SLO under correlated error.

    Errors in the predicted return time and the predicted prompt size shift
    every node's TTFT by the same amount, so they are integrated as a shared
    term outside the per-node product. Only queue error is treated as
    independent. With a single shared term this collapses towards single-node
    success instead of rewarding redundancy that does not exist.
    """
    if shared_uncertainty_s <= 0:
        return at_least_one_success([success_probability(ttft, slo_s, node_uncertainty_s) for ttft in ttfts])
    total = 0.0
    for offset, weight in SHARED_STRATA:
        shift = offset * shared_uncertainty_s
        total += weight * at_least_one_success(
            [success_probability(ttft + shift, slo_s, node_uncertainty_s) for ttft in ttfts]
        )
    return total
