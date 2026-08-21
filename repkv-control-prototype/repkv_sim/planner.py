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
    compute_seconds: float = 0.0

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
        return RecoveryPlan((), 0.0, 0.0)
    if len(regions) == 1:
        left, right, methods = regions[0]
        method = max(methods, key=foreground_rates.__getitem__)
        seconds = (right - left) / foreground_rates[method]
        compute = seconds if method == "recompute" else 0.0
        return RecoveryPlan((WorkSegment(method, left, right),), seconds, compute)
    segments: list[WorkSegment] = []
    seconds = 0.0
    compute = 0.0
    for left, right, methods in regions:
        method = max(methods, key=foreground_rates.__getitem__)
        segments.append(WorkSegment(method, left, right))
        span = (right - left) / foreground_rates[method]
        seconds += span
        if method == "recompute":
            compute += span
    return RecoveryPlan(_compress(segments), seconds, compute)


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


def pinned_availability(local_hbm: int) -> PrefixAvailability:
    """Sources that are still there if DRAM and peer copies vanish.

    Only local HBM is reserved by this node. A host restore or a remote
    transfer may disappear before the request arrives, so a durable TTFT
    path cannot count on them.
    """
    return PrefixAvailability(local_hbm, local_hbm, local_hbm)


def layout_success_probability(
    volatile_s: float,
    durable_s: float,
    budget_s: float,
    copy_persist: float,
) -> float:
    """Chance the current layout still meets the recovery budget at arrival.

    Volatile recovery uses host and remote copies; durable recovery uses only
    local HBM and recomputes the rest. `copy_persist` is the probability those
    volatile copies are still reachable when the request lands.
    """
    persist = min(1.0, max(0.0, copy_persist))
    volatile_ok = 1.0 if volatile_s <= budget_s else 0.0
    durable_ok = 1.0 if durable_s <= budget_s else 0.0
    return persist * volatile_ok + (1.0 - persist) * durable_ok


def decide_minimum_slo_preparation(
    context_blocks: int,
    availability: PrefixAvailability,
    queue_s: float,
    prompt_s: float,
    slo_s: float,
    time_limit_s: float,
    foreground_rates: dict[str, float],
    background_rates: dict[str, float],
    method_costs: dict[str, float],
    copy_persist: float = 1.0,
    already_feasible_probability: float = 1.0,
) -> tuple[PreparationPlan | None, str]:
    """Return the minimum SLO plan and why a plan is absent.

    Residual recovery time is non-increasing in the target prefix: raising the
    target both shortens the interval left to recover and can only widen the
    set of sources available for what remains. Feasibility is therefore
    monotone in the target and the smallest feasible one is found by bisection.
    The cheapest timely plan is likewise non-decreasing in duration, so if the
    smallest feasible target cannot be built within the time limit no larger
    target can either.

    Two vetoes refuse work before bisection. `queue_exceeds_budget` means the
    predicted device queue already overruns the deadline, so a local prefix
    would not help. `already_feasible` means the current layout is likely
    enough to recover in time that preparing has no SLO work left. Likelihood
    mixes a volatile path (host and remote copies) with a durable path (local
    HBM only) using `copy_persist`. Full-prefix preparation does not apply
    either veto.
    """
    # Recovery overlaps with the device queue rather than adding to it: a
    # restore or a transfer proceeds while the device serves other requests.
    # The two therefore have to fit under the deadline separately.
    budget = slo_s + 1e-9 - prompt_s
    if queue_s > budget:
        return None, "queue_exceeds_budget"

    def volatile_seconds(target: int) -> float:
        hypothetical = PrefixAvailability(target, availability.local_host, availability.remote_hbm)
        return fastest_recovery(target, context_blocks, hypothetical, foreground_rates).seconds

    def durable_seconds(target: int) -> float:
        return fastest_recovery(
            target, context_blocks, pinned_availability(target), foreground_rates
        ).seconds

    def success_at(target: int) -> float:
        return layout_success_probability(
            volatile_seconds(target),
            durable_seconds(target),
            budget,
            copy_persist,
        )

    current = availability.local_hbm
    if success_at(current) >= already_feasible_probability - 1e-12:
        return None, "already_feasible"
    low, high = current + 1, context_blocks
    if low > high:
        return None, "already_feasible"
    if success_at(high) < already_feasible_probability - 1e-12:
        return None, "cannot_cross_slo"
    while low < high:
        middle = (low + high) // 2
        if success_at(middle) >= already_feasible_probability - 1e-12:
            high = middle
        else:
            low = middle + 1
    plan = cheapest_preparation(
        availability.local_hbm,
        low,
        availability,
        time_limit_s,
        background_rates,
        method_costs,
    )
    if plan is None or not plan.segments:
        return None, "no_timely_plan"
    return plan, "ok"


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
    copy_persist: float = 1.0,
    already_feasible_probability: float = 1.0,
) -> PreparationPlan | None:
    """Find the minimum prepared prefix that crosses the hard TTFT boundary."""
    plan, _reason = decide_minimum_slo_preparation(
        context_blocks,
        availability,
        queue_s,
        prompt_s,
        slo_s,
        time_limit_s,
        foreground_rates,
        background_rates,
        method_costs,
        copy_persist,
        already_feasible_probability,
    )
    return plan


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
    durable_ttfts: list[float] | None = None,
    persists: list[float] | None = None,
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

    When `durable_ttfts` and `persists` are set, each node mixes a volatile
    path (current host and remote copies) with a durable path (local HBM
    only). `persists[n]` is the probability the volatile copies are still
    reachable at arrival.
    """
    if len(ttfts) != len(node_uncertainties_s):
        raise ValueError("ttfts and node_uncertainties_s must have equal length")
    if durable_ttfts is None:
        durable_ttfts = ttfts
    if persists is None:
        persists = [1.0] * len(ttfts)
    if len(durable_ttfts) != len(ttfts) or len(persists) != len(ttfts):
        raise ValueError("durable_ttfts and persists must match ttfts")

    def node_probability(volatile: float, durable: float, persist: float, sigma: float, shift: float) -> float:
        held = min(1.0, max(0.0, persist))
        volatile_p = success_probability(volatile + shift, slo_s, sigma)
        if held >= 1.0:
            return volatile_p
        durable_p = success_probability(durable + shift, slo_s, sigma)
        return held * volatile_p + (1.0 - held) * durable_p

    triples = list(zip(ttfts, durable_ttfts, persists, node_uncertainties_s))
    if shared_uncertainty_s <= 0:
        return at_least_one_success(
            [node_probability(volatile, durable, persist, sigma, 0.0) for volatile, durable, persist, sigma in triples]
        )
    total = 0.0
    for offset, weight in SHARED_STRATA:
        shift = offset * shared_uncertainty_s
        failure = 1.0
        for volatile, durable, persist, sigma in triples:
            probability = node_probability(volatile, durable, persist, sigma, shift)
            if probability <= 0.0:
                continue
            if probability >= 1.0:
                failure = 0.0
                break
            failure *= 1.0 - probability
        total += weight * (1.0 - failure)
    return total
