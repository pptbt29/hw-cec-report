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


def _methods_for(right: int, availability: PrefixAvailability) -> tuple[str, ...]:
    if right <= availability.local_host:
        return ("recompute", "restore", "transfer") if right <= availability.remote_hbm else ("recompute", "restore")
    return ("recompute", "transfer") if right <= availability.remote_hbm else ("recompute",)


def _regions(start: int, end: int, availability: PrefixAvailability) -> list[tuple[int, int, tuple[str, ...]]]:
    if not 0 <= start <= end:
        raise ValueError(f"invalid prefix interval [{start}, {end})")
    host, remote = availability.local_host, availability.remote_hbm
    inner_host = start < host < end
    inner_remote = start < remote < end
    if not inner_host and not inner_remote:
        if start == end:
            return []
        return [(start, end, _methods_for(end, availability))]
    boundaries = {start, end}
    if inner_host:
        boundaries.add(host)
    if inner_remote:
        boundaries.add(remote)
    ordered = sorted(boundaries)
    return [
        (left, right, _methods_for(right, availability))
        for left, right in zip(ordered, ordered[1:])
    ]


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
    regions = _regions(start_prefix, context_blocks, availability)
    if not regions:
        return RecoveryPlan((), 0.0)
    if len(regions) == 1:
        left, right, methods = regions[0]
        method = max(methods, key=foreground_rates.__getitem__)
        return RecoveryPlan((WorkSegment(method, left, right),), (right - left) / foreground_rates[method])
    segments: list[WorkSegment] = []
    seconds = 0.0
    for left, right, methods in regions:
        method = max(methods, key=foreground_rates.__getitem__)
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
    """Find the minimum prepared prefix that crosses the hard TTFT boundary.

    Residual recovery time is non-increasing in the target prefix: raising the
    target both shortens the interval left to recover and can only widen the
    set of sources available for what remains. Feasibility is therefore
    monotone in the target and the smallest feasible one is found by bisection.
    The cheapest timely plan is likewise non-decreasing in duration, so if the
    smallest feasible target cannot be built within the time limit no larger
    target can either.
    """
    current = fastest_recovery(availability.local_hbm, context_blocks, availability, foreground_rates)
    budget = slo_s + 1e-9 - queue_s - prompt_s
    if current.seconds <= budget:
        return None
    if budget < 0:
        return None
    low, high = availability.local_hbm + 1, context_blocks
    if low > high:
        return None

    def residual_seconds(target: int) -> float:
        hypothetical = PrefixAvailability(target, max(target, availability.local_host), availability.remote_hbm)
        return fastest_recovery(target, context_blocks, hypothetical, foreground_rates).seconds

    if residual_seconds(high) > budget:
        return None
    while low < high:
        middle = (low + high) // 2
        if residual_seconds(middle) <= budget:
            high = middle
        else:
            low = middle + 1
    return cheapest_preparation(
        availability.local_hbm,
        low,
        availability,
        time_limit_s,
        background_rates,
        method_costs,
    )


def success_probability(ttft_s: float, slo_s: float, uncertainty_s: float) -> float:
    z = (ttft_s - slo_s) / (uncertainty_s if uncertainty_s > 1e-9 else 1e-9)
    if z >= 35:
        return 0.0
    if z <= -35:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def at_least_one_success(probabilities: list[float]) -> float:
    failure = 1.0
    for probability in probabilities:
        if probability <= 0.0:
            continue
        if probability >= 1.0:
            return 1.0
        failure *= 1.0 - probability
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
    node_uncertainties_s: list[float],
) -> float:
    """Probability that at least one node meets the SLO under correlated error.

    Errors in the predicted return time and the predicted prompt size shift
    every node's TTFT by the same amount, so they are integrated as a shared
    term outside the per-node product. Only queue error is treated as
    independent. Treating every node as independent lets a cluster of nodes
    that all sit exactly on the SLO boundary look collectively safe, which
    removes the value of preparing any of them.

    Node uncertainty is passed per node because queue error is not uniform
    across the cluster: a node with a long projected backlog carries more
    absolute error than an idle one.
    """
    if len(ttfts) != len(node_uncertainties_s):
        raise ValueError("ttfts and node_uncertainties_s must have equal length")
    if shared_uncertainty_s <= 0:
        return at_least_one_success(
            [success_probability(ttft, slo_s, sigma) for ttft, sigma in zip(ttfts, node_uncertainties_s)]
        )
    pairs = list(zip(ttfts, node_uncertainties_s))
    total = 0.0
    for offset, weight in SHARED_STRATA:
        shift = offset * shared_uncertainty_s
        failure = 1.0
        for ttft, sigma in pairs:
            probability = success_probability(ttft + shift, slo_s, sigma)
            if probability <= 0.0:
                continue
            if probability >= 1.0:
                failure = 0.0
                break
            failure *= 1.0 - probability
        total += weight * (1.0 - failure)
    return total
