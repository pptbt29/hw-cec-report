from __future__ import annotations

import csv
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .planner import (
    PrefixAvailability,
    PreparationPlan,
    WorkSegment,
    at_least_one_success,
    cheapest_preparation,
    fastest_recovery,
    minimum_slo_preparation,
    routable_probability,
    success_probability,
)


POLICIES = ("on_demand", "eager_full", "repkv")


@dataclass(frozen=True)
class Config:
    nodes: int = 4
    sessions: int = 48
    horizon_s: float = 240.0
    step_s: float = 0.5
    hbm_blocks: int = 180
    ttft_slo_s: float = 2.0
    prefill_blocks_s: float = 24.0
    decode_blocks_s: float = 10.0
    transfer_blocks_s: float = 20.0
    host_restore_blocks_s: float = 28.0
    recompute_blocks_s: float = 12.0
    background_transfer_fraction: float = 0.45
    background_restore_fraction: float = 0.45
    background_compute_fraction: float = 0.30
    preparation_horizon_s: float = 18.0
    queue_uncertainty_s: float = 0.25
    shared_uncertainty_s: float = 0.45
    block_batch: int = 4
    high_watermark: float = 0.92
    low_watermark: float = 0.82
    max_prepared_nodes_per_session: int = 2
    resource_penalty: float = 0.22
    arrival_forecast: bool = True
    assumed_output_blocks: float = 4.0
    output_estimate_weight: float = 0.2

    @property
    def foreground_rates(self) -> dict[str, float]:
        return {
            "transfer": self.transfer_blocks_s,
            "restore": self.host_restore_blocks_s,
            "recompute": self.recompute_blocks_s,
        }

    @property
    def background_rates(self) -> dict[str, float]:
        return {
            "transfer": self.transfer_blocks_s * self.background_transfer_fraction,
            "restore": self.host_restore_blocks_s * self.background_restore_fraction,
            "recompute": self.recompute_blocks_s * self.background_compute_fraction,
        }

    @property
    def method_costs(self) -> dict[str, float]:
        return {"restore": 0.65, "transfer": 1.0, "recompute": 1.35}


@dataclass(frozen=True)
class Turn:
    sid: int
    index: int
    arrival: float
    prompt_blocks: int
    output_blocks: int
    history_blocks: int
    predicted_arrival: float
    predicted_prompt_blocks: int
    return_probability: float


@dataclass
class KVReplica:
    hbm_prefix: int = 0
    host_prefix: int = 0
    prepared_blocks: int = 0
    last_used: float = 0.0


@dataclass
class Node:
    nid: int
    capacity: int
    busy_until: float = 0.0
    replicas: dict[int, KVReplica] = field(default_factory=dict)
    active_reservations: dict[int, int] = field(default_factory=dict)
    background_reserved: int = 0

    @property
    def hbm_used(self) -> int:
        return sum(replica.hbm_prefix for replica in self.replicas.values())

    @property
    def active_extra(self) -> int:
        return sum(
            max(0, target - self.replicas.get(sid, KVReplica()).hbm_prefix)
            for sid, target in self.active_reservations.items()
        )

    @property
    def committed(self) -> int:
        return self.hbm_used + self.active_extra + self.background_reserved


@dataclass
class SessionState:
    sid: int
    completed_turn: int = -1
    context_blocks: int = 0
    active_until: float = 0.0
    active_node: int | None = None


@dataclass(frozen=True)
class InFlightBatch:
    sid: int
    nid: int
    method: str
    start: int
    end: int
    complete_at: float

    @property
    def blocks(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class PlanCandidate:
    sid: int
    nid: int
    predicted_arrival: float
    return_probability: float
    plan: PreparationPlan
    expected_gain: float
    normalized_cost: float
    net_value: float
    score: float
    slack_s: float


@dataclass
class Metrics:
    requests: int = 0
    successes: int = 0
    ttfts: list[float] = field(default_factory=list)
    background: dict[str, int] = field(default_factory=lambda: {name: 0 for name in ("restore", "transfer", "recompute")})
    demand: dict[str, int] = field(default_factory=lambda: {name: 0 for name in ("restore", "transfer", "recompute")})
    hbm_block_seconds: float = 0.0
    prepared_blocks: int = 0
    used_prepared_blocks: int = 0
    demoted_prepared_blocks: int = 0
    demoted_blocks: int = 0
    cancelled_batches: int = 0
    overlapping_turns: int = 0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def generate_workload(cfg: Config, seed: int) -> list[Turn]:
    """Generate calibrated predictions without revealing the realized next gap.

    A session class determines both a declared return probability and a gap
    distribution. The realized return and the predicted gap are independent
    draws conditioned on that class; the controller never receives actual
    future time plus noise.
    """
    rng = random.Random(seed)
    classes = (
        ("fast", 12.0, 0.88),
        ("normal", 22.0, 0.78),
        ("slow", 40.0, 0.66),
    )
    turns: list[Turn] = []
    for sid in range(cfg.sessions):
        _, median_gap, base_return = rng.choice(classes)
        arrival = rng.expovariate(1 / 28.0)
        history = 0
        max_turns = rng.randint(3, 8)
        for index in range(max_turns):
            prompt = rng.randint(8, 16) if index == 0 else rng.randint(3, 9)
            output = rng.randint(2, 7)
            return_probability = max(0.18, base_return - 0.045 * index)
            predicted_gap = median_gap * math.exp(rng.gauss(0.0, 0.28))
            predicted_arrival = arrival + predicted_gap
            predicted_prompt = max(1, round((5.5 if index else 6.0) * math.exp(rng.gauss(0.0, 0.22))))
            turns.append(
                Turn(
                    sid=sid,
                    index=index,
                    arrival=round(arrival / cfg.step_s) * cfg.step_s,
                    prompt_blocks=prompt,
                    output_blocks=output,
                    history_blocks=history,
                    predicted_arrival=predicted_arrival,
                    predicted_prompt_blocks=predicted_prompt,
                    return_probability=return_probability,
                )
            )
            history += prompt + output
            if index + 1 >= max_turns or rng.random() > return_probability:
                break
            actual_gap = median_gap * math.exp(rng.gauss(0.0, 0.55))
            arrival += max(6.0, min(actual_gap, 75.0))
            if arrival > cfg.horizon_s:
                break
    return sorted((turn for turn in turns if turn.arrival <= cfg.horizon_s), key=lambda item: (item.arrival, item.sid))


class Simulator:
    def __init__(self, cfg: Config, turns: list[Turn], policy: str, seed: int = 0):
        if policy not in POLICIES:
            raise ValueError(policy)
        self.cfg = cfg
        self.turns = turns
        self.policy = policy
        self.seed = seed
        self.nodes = [Node(nid, cfg.hbm_blocks) for nid in range(cfg.nodes)]
        self.sessions = {sid: SessionState(sid) for sid in range(cfg.sessions)}
        self.metrics = Metrics()
        self.request_completions: list[tuple[float, Turn, int]] = []
        self.inflight: list[InFlightBatch] = []
        self.request_rows: list[dict[str, float | int | str]] = []
        self.turn_lookup = {(turn.sid, turn.index): turn for turn in turns}
        self.arrivals: dict[float, list[Turn]] = {}
        for turn in turns:
            self.arrivals.setdefault(turn.arrival, []).append(turn)
        self.now = -cfg.step_s
        self.last_candidates: list[PlanCandidate] = []
        self.last_events: list[str] = []
        self.output_estimate = cfg.assumed_output_blocks
        self._forecast_cache: dict[int, dict[int, float]] | None = None

    def run(self) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
        steps = int(self.cfg.horizon_s / self.cfg.step_s) + 1
        for _ in range(steps):
            self.step()
        self._complete_inflight(float("inf"))
        self._complete_requests(float("inf"))
        success = max(self.metrics.successes, 1)
        remaining_prepared = sum(replica.prepared_blocks for node in self.nodes for replica in node.replicas.values())
        unused = remaining_prepared + self.metrics.demoted_prepared_blocks
        summary: dict[str, float | int | str] = {
            "policy": self.policy,
            "requests": self.metrics.requests,
            "slo_successes": self.metrics.successes,
            "slo_goodput_rps": self.metrics.successes / self.cfg.horizon_s,
            "slo_attainment": self.metrics.successes / max(1, self.metrics.requests),
            "p99_ttft_s": percentile(self.metrics.ttfts, 0.99),
            "transfer_blocks_per_success": (self.metrics.background["transfer"] + self.metrics.demand["transfer"]) / success,
            "restore_blocks_per_success": (self.metrics.background["restore"] + self.metrics.demand["restore"]) / success,
            "recompute_blocks_per_success": (self.metrics.background["recompute"] + self.metrics.demand["recompute"]) / success,
            "hbm_block_seconds_per_success": self.metrics.hbm_block_seconds / success,
            "unused_preparation_ratio": unused / max(1, self.metrics.prepared_blocks),
            "demoted_blocks": self.metrics.demoted_blocks,
            "cancelled_batches": self.metrics.cancelled_batches,
            "overlapping_turns": self.metrics.overlapping_turns,
        }
        return summary, self.request_rows

    def step(self) -> list[str]:
        self.now = round(self.now + self.cfg.step_s, 9)
        self.last_events = []
        self._forecast_cache = None
        self._complete_inflight(self.now)
        self._complete_requests(self.now)
        for turn in self.arrivals.get(self.now, []):
            self._arrive(turn)
        if self.policy != "on_demand":
            self._prepare()
        self.metrics.hbm_block_seconds += sum(node.hbm_used for node in self.nodes) * self.cfg.step_s
        self._validate()
        return list(self.last_events)

    def _replica(self, nid: int, sid: int) -> KVReplica:
        return self.nodes[nid].replicas.get(sid, KVReplica())

    def _availability(self, sid: int, nid: int, local_hbm_override: int | None = None) -> PrefixAvailability:
        local = self._replica(nid, sid)
        remote = max((self._replica(node.nid, sid).hbm_prefix for node in self.nodes if node.nid != nid), default=0)
        return PrefixAvailability(
            local.hbm_prefix if local_hbm_override is None else local_hbm_override,
            local.host_prefix,
            remote,
        )

    def _pending_returns(self) -> list[tuple[float, int, Turn, SessionState]]:
        pending: list[tuple[float, int, Turn, SessionState]] = []
        for state in self.sessions.values():
            if state.completed_turn < 0 or state.active_until > self.now:
                continue
            previous = self.turn_lookup.get((state.sid, state.completed_turn))
            if previous is None:
                continue
            pending.append((max(self.now, previous.predicted_arrival), state.sid, previous, state))
        return sorted(pending, key=lambda item: (item[0], item[1]))

    def _forecast(self) -> dict[int, dict[int, float]]:
        """Project per-node backlog at each session's predicted return time.

        Sessions are inserted in predicted arrival order, so the view recorded
        for a session excludes its own load but includes every other session
        predicted to return earlier. Node choice mirrors the router, and the
        added occupancy is discounted by the session return probability.

        A session whose predicted arrival has already passed contributes no
        load: the point predictor no longer states when it will return, and
        charging its full service demand to the current instant would inflate
        every queue estimate. This makes the forecast a known under-estimate.
        """
        if self._forecast_cache is not None:
            return self._forecast_cache
        forecast: dict[int, dict[int, float]] = {}
        if not self.cfg.arrival_forecast:
            self._forecast_cache = forecast
            return forecast
        projected = {node.nid: node.busy_until for node in self.nodes}
        decode_s = self.output_estimate / self.cfg.decode_blocks_s
        for arrival, sid, previous, state in self._pending_returns():
            forecast[sid] = dict(projected)
            if previous.predicted_arrival <= self.now:
                continue
            prompt = previous.predicted_prompt_blocks / self.cfg.prefill_blocks_s
            best_nid, best_ttft, best_service = 0, math.inf, 0.0
            for node in self.nodes:
                availability = self._availability(sid, node.nid)
                recovery = fastest_recovery(
                    availability.local_hbm,
                    state.context_blocks,
                    availability,
                    self.cfg.foreground_rates,
                )
                ttft = max(0.0, projected[node.nid] - arrival) + recovery.seconds + prompt
                if ttft < best_ttft:
                    best_nid, best_ttft = node.nid, ttft
                    best_service = recovery.seconds + prompt + decode_s
            projected[best_nid] = max(projected[best_nid], arrival) + previous.return_probability * best_service
        self._forecast_cache = forecast
        return forecast

    def _projected_busy_until(self, sid: int, nid: int) -> float:
        view = self._forecast().get(sid)
        if view is None:
            return self.nodes[nid].busy_until
        return view[nid]

    def _predicted_ttft(self, state: SessionState, previous: Turn, node: Node, hbm_override: int | None = None) -> float:
        predicted_time = max(self.now, previous.predicted_arrival)
        availability = self._availability(state.sid, node.nid, hbm_override)
        recovery = fastest_recovery(
            availability.local_hbm,
            state.context_blocks,
            availability,
            self.cfg.foreground_rates,
        )
        queue = max(0.0, self._projected_busy_until(state.sid, node.nid) - predicted_time)
        prompt = previous.predicted_prompt_blocks / self.cfg.prefill_blocks_s
        return queue + recovery.seconds + prompt

    def _cluster_probability(self, state: SessionState, previous: Turn, override: tuple[int, int] | None = None) -> float:
        ttfts = []
        for node in self.nodes:
            hbm_override = override[1] if override and override[0] == node.nid else None
            ttfts.append(self._predicted_ttft(state, previous, node, hbm_override))
        return routable_probability(
            ttfts,
            self.cfg.ttft_slo_s,
            self.cfg.shared_uncertainty_s,
            self.cfg.queue_uncertainty_s,
        )

    def _complete_inflight(self, now: float) -> None:
        ready = [batch for batch in self.inflight if batch.complete_at <= now]
        self.inflight = [batch for batch in self.inflight if batch.complete_at > now]
        for batch in sorted(ready, key=lambda item: (item.complete_at, item.sid, item.nid)):
            node = self.nodes[batch.nid]
            node.background_reserved -= batch.blocks
            replica = node.replicas.setdefault(batch.sid, KVReplica(last_used=batch.complete_at))
            source_ok = batch.method == "recompute"
            if batch.method == "restore":
                source_ok = replica.host_prefix >= batch.end
            elif batch.method == "transfer":
                source_ok = any(
                    other.nid != batch.nid and self._replica(other.nid, batch.sid).hbm_prefix >= batch.end
                    for other in self.nodes
                )
            if replica.hbm_prefix != batch.start or not source_ok:
                self.metrics.cancelled_batches += 1
                self.last_events.append(f"cancel sid={batch.sid} n={batch.nid} {batch.method}[{batch.start},{batch.end})")
                continue
            replica.hbm_prefix = batch.end
            replica.host_prefix = max(replica.host_prefix, batch.end)
            replica.prepared_blocks += batch.blocks
            replica.last_used = batch.complete_at
            self.metrics.prepared_blocks += batch.blocks
            self.metrics.background[batch.method] += batch.blocks
            self.last_events.append(f"complete sid={batch.sid} n={batch.nid} {batch.method}[{batch.start},{batch.end})")

    def _complete_requests(self, now: float) -> None:
        ready = [item for item in self.request_completions if item[0] <= now]
        self.request_completions = [item for item in self.request_completions if item[0] > now]
        for completion, turn, nid in sorted(ready, key=lambda item: (item[0], item[1].sid, item[1].index)):
            self._forecast_cache = None
            node = self.nodes[nid]
            state = self.sessions[turn.sid]
            weight = self.cfg.output_estimate_weight
            self.output_estimate += weight * (turn.output_blocks - self.output_estimate)
            context = turn.history_blocks + turn.prompt_blocks + turn.output_blocks
            node.active_reservations.pop(turn.sid, None)
            replica = node.replicas.setdefault(turn.sid, KVReplica())
            replica.hbm_prefix = context
            replica.host_prefix = max(replica.host_prefix, context)
            replica.prepared_blocks = 0
            replica.last_used = completion
            state.completed_turn = turn.index
            state.context_blocks = context
            state.active_until = 0.0
            state.active_node = None
            self.last_events.append(f"finish sid={turn.sid} turn={turn.index} n={nid} context={context}")

    def _arrive(self, turn: Turn) -> None:
        self._forecast_cache = None
        state = self.sessions[turn.sid]
        if state.active_until > self.now:
            self.metrics.overlapping_turns += 1
            self.last_events.append(f"overlap sid={turn.sid} turn={turn.index}")
        final_context = turn.history_blocks + turn.prompt_blocks + turn.output_blocks
        choices: list[tuple[float, int, tuple[WorkSegment, ...], int]] = []
        for node in self.nodes:
            availability = self._availability(turn.sid, node.nid)
            recovery = fastest_recovery(
                availability.local_hbm,
                turn.history_blocks,
                availability,
                self.cfg.foreground_rates,
            )
            queue = max(0.0, node.busy_until - self.now)
            ttft = queue + recovery.seconds + turn.prompt_blocks / self.cfg.prefill_blocks_s
            extra = max(0, final_context - self._replica(node.nid, turn.sid).hbm_prefix)
            if self._available_capacity(node.nid, turn.sid) >= extra:
                choices.append((ttft, node.nid, recovery.segments, extra))
        if not choices:
            raise RuntimeError(f"no node can reserve active KV for sid={turn.sid}, context={final_context}")
        ttft, nid, recovery_segments, extra = min(choices, key=lambda item: (item[0], item[1]))
        if not self._ensure_capacity(nid, extra, turn.sid):
            raise RuntimeError("capacity precheck and eviction disagree")
        node = self.nodes[nid]
        node.active_reservations[turn.sid] = final_context
        replica = node.replicas.get(turn.sid)
        if replica:
            used = min(replica.prepared_blocks, turn.history_blocks)
            replica.prepared_blocks -= used
            replica.last_used = self.now
            self.metrics.used_prepared_blocks += used
        for segment in recovery_segments:
            self.metrics.demand[segment.method] += segment.blocks
        start = max(self.now, node.busy_until)
        service_without_queue = ttft - max(0.0, node.busy_until - self.now)
        completion = start + service_without_queue + turn.output_blocks / self.cfg.decode_blocks_s
        node.busy_until = completion
        state.active_until = completion
        state.active_node = nid
        self.request_completions.append((completion, turn, nid))
        success = ttft <= self.cfg.ttft_slo_s
        self.metrics.requests += 1
        self.metrics.successes += int(success)
        self.metrics.ttfts.append(ttft)
        methods = "+".join(segment.method for segment in recovery_segments) or "hit"
        self.request_rows.append(
            {
                "policy": self.policy,
                "sid": turn.sid,
                "turn": turn.index,
                "arrival_s": self.now,
                "node": nid,
                "history_blocks": turn.history_blocks,
                "local_hbm_prefix": self._replica(nid, turn.sid).hbm_prefix,
                "recovery": methods,
                "ttft_s": ttft,
                "slo_success": int(success),
            }
        )
        self.last_events.append(f"arrive sid={turn.sid} turn={turn.index} -> n={nid} ttft={ttft:.3f} success={int(success)}")

    def _collect_candidates(self) -> list[PlanCandidate]:
        raw: list[tuple[SessionState, Turn, Node, PreparationPlan, float, float]] = []
        for state in self.sessions.values():
            if state.completed_turn < 0 or state.active_until > self.now:
                continue
            previous = self.turn_lookup.get((state.sid, state.completed_turn))
            if previous is None:
                continue
            time_left = previous.predicted_arrival - self.now
            if time_left <= 0 or time_left > self.cfg.preparation_horizon_s:
                continue
            before = self._cluster_probability(state, previous)
            for node in self.nodes:
                availability = self._availability(state.sid, node.nid)
                if availability.local_hbm >= state.context_blocks:
                    continue
                queue = max(0.0, self._projected_busy_until(state.sid, node.nid) - previous.predicted_arrival)
                prompt = previous.predicted_prompt_blocks / self.cfg.prefill_blocks_s
                if self.policy == "eager_full":
                    plan = cheapest_preparation(
                        availability.local_hbm,
                        state.context_blocks,
                        availability,
                        time_left,
                        self.cfg.background_rates,
                        self.cfg.method_costs,
                    )
                else:
                    plan = minimum_slo_preparation(
                        state.context_blocks,
                        availability,
                        queue,
                        prompt,
                        self.cfg.ttft_slo_s,
                        time_left,
                        self.cfg.foreground_rates,
                        self.cfg.background_rates,
                        self.cfg.method_costs,
                    )
                if plan is None or not plan.segments:
                    continue
                after = self._cluster_probability(state, previous, (node.nid, plan.target_prefix))
                gain = previous.return_probability * max(0.0, after - before)
                hbm_cost = (plan.target_prefix - availability.local_hbm) * time_left / max(
                    1.0, node.capacity * self.cfg.preparation_horizon_s
                )
                raw.append((state, previous, node, plan, gain, hbm_cost))
        if not raw:
            return []
        demand = {
            method: sum(plan.blocks(method) / max(1.0, self.cfg.background_rates[method] * (previous.predicted_arrival - self.now))
                        for _, previous, _, plan, _, _ in raw)
            for method in self.cfg.background_rates
        }
        hbm_pressure = sum(node.committed for node in self.nodes) / max(1, sum(node.capacity for node in self.nodes))
        prices = {method: 1.0 + max(0.0, demand[method] - 1.0) ** 2 for method in demand}
        candidates: list[PlanCandidate] = []
        for state, previous, node, plan, gain, hbm_cost in raw:
            resource_cost = hbm_cost * (1.0 + 3.0 * max(0.0, hbm_pressure - self.cfg.low_watermark))
            for method in self.cfg.background_rates:
                denominator = self.cfg.background_rates[method] * self.cfg.preparation_horizon_s
                resource_cost += prices[method] * plan.blocks(method) / max(1.0, denominator)
            if self.policy == "eager_full":
                net = 1.0 - 0.01 * resource_cost
                score = 1.0 / max(resource_cost, 1e-9)
            else:
                net = gain - self.cfg.resource_penalty * resource_cost
                score = net / max(resource_cost, 1e-9)
            slack = previous.predicted_arrival - self.now - plan.seconds
            if net > 0:
                candidates.append(
                    PlanCandidate(
                        state.sid,
                        node.nid,
                        previous.predicted_arrival,
                        previous.return_probability,
                        plan,
                        gain,
                        resource_cost,
                        net,
                        score,
                        slack,
                    )
                )
        return candidates

    def _prepare(self) -> None:
        self._forecast_cache = None
        self.last_candidates = self._collect_candidates()
        budgets = {
            method: max(0, math.floor(rate * self.cfg.step_s + 1e-9))
            for method, rate in self.cfg.background_rates.items()
        }
        selected_sessions: set[int] = set()
        inflight_targets = {(batch.sid, batch.nid) for batch in self.inflight}
        ordered = sorted(self.last_candidates, key=lambda candidate: (candidate.slack_s, -candidate.score, candidate.sid, candidate.nid))
        for candidate in ordered:
            if candidate.sid in selected_sessions or (candidate.sid, candidate.nid) in inflight_targets:
                continue
            prepared_nodes = sum(
                1 for node in self.nodes if self._replica(node.nid, candidate.sid).prepared_blocks > 0
            )
            prepared_node_limit = 1 if self.policy == "eager_full" else self.cfg.max_prepared_nodes_per_session
            if prepared_nodes >= prepared_node_limit and self._replica(candidate.nid, candidate.sid).prepared_blocks == 0:
                continue
            segment = candidate.plan.segments[0]
            current = self._replica(candidate.nid, candidate.sid).hbm_prefix
            if segment.start != current:
                continue
            amount = min(segment.blocks, budgets[segment.method], self.cfg.block_batch)
            if amount <= 0:
                continue
            end = current + amount
            if segment.method == "restore" and self._replica(candidate.nid, candidate.sid).host_prefix < end:
                continue
            if segment.method == "transfer" and not any(
                node.nid != candidate.nid and self._replica(node.nid, candidate.sid).hbm_prefix >= end
                for node in self.nodes
            ):
                continue
            if not self._ensure_capacity(candidate.nid, amount, candidate.sid):
                continue
            node = self.nodes[candidate.nid]
            node.background_reserved += amount
            duration = amount / self.cfg.background_rates[segment.method]
            batch = InFlightBatch(candidate.sid, candidate.nid, segment.method, current, end, self.now + duration)
            self.inflight.append(batch)
            budgets[segment.method] -= amount
            selected_sessions.add(candidate.sid)
            self.last_events.append(
                f"dispatch sid={candidate.sid} n={candidate.nid} {segment.method}[{current},{end}) "
                f"target={candidate.plan.target_prefix} net={candidate.net_value:.4f}"
            )

    def _available_capacity(self, nid: int, protected_sid: int) -> int:
        node = self.nodes[nid]
        evictable = sum(
            replica.hbm_prefix
            for sid, replica in node.replicas.items()
            if sid != protected_sid and self.sessions[sid].active_node != nid
        )
        return node.capacity - node.committed + evictable

    def _ensure_capacity(self, nid: int, extra: int, protected_sid: int) -> bool:
        node = self.nodes[nid]
        if self._available_capacity(nid, protected_sid) < extra:
            return False
        target = node.capacity
        if node.committed + extra > int(node.capacity * self.cfg.high_watermark):
            target = max(extra, int(node.capacity * self.cfg.low_watermark))
        while node.committed + extra > target:
            candidates = [
                sid for sid, replica in node.replicas.items()
                if sid != protected_sid and replica.hbm_prefix > 0 and self.sessions[sid].active_node != nid
            ]
            if not candidates:
                target = node.capacity
                if node.committed + extra <= target:
                    break
                return False
            if self.policy == "repkv":
                victim = min(candidates, key=lambda sid: self._eviction_score(sid, nid))
            else:
                victim = min(candidates, key=lambda sid: node.replicas[sid].last_used)
            replica = node.replicas[victim]
            amount = min(self.cfg.block_batch, replica.hbm_prefix, node.committed + extra - target)
            old_prefix = replica.hbm_prefix
            replica.host_prefix = max(replica.host_prefix, old_prefix)
            replica.hbm_prefix -= amount
            prepared = min(amount, replica.prepared_blocks)
            replica.prepared_blocks -= prepared
            self.metrics.demoted_prepared_blocks += prepared
            self.metrics.demoted_blocks += amount
            self.last_events.append(f"demote sid={victim} n={nid} [{replica.hbm_prefix},{old_prefix})")
        return node.committed + extra <= node.capacity

    def _eviction_score(self, sid: int, nid: int) -> float:
        state = self.sessions[sid]
        previous = self.turn_lookup.get((sid, state.completed_turn))
        replica = self._replica(nid, sid)
        amount = min(self.cfg.block_batch, replica.hbm_prefix)
        if previous is None or previous.predicted_arrival <= self.now:
            return 0.0001
        before = self._cluster_probability(state, previous)
        after = self._cluster_probability(state, previous, (nid, replica.hbm_prefix - amount))
        expected_loss = previous.return_probability * max(0.0, before - after)
        demotion_cost = 0.002 * amount
        return (expected_loss + demotion_cost) / amount

    def readiness_snapshot(self) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for state in self.sessions.values():
            if state.completed_turn < 0:
                continue
            previous = self.turn_lookup.get((state.sid, state.completed_turn))
            for node in self.nodes:
                availability = self._availability(state.sid, node.nid)
                recovery = fastest_recovery(
                    availability.local_hbm,
                    state.context_blocks,
                    availability,
                    self.cfg.foreground_rates,
                )
                predicted_ttft = self._predicted_ttft(state, previous, node) if previous else recovery.seconds
                rows.append(
                    {
                        "sid": state.sid,
                        "node": node.nid,
                        "valid_hbm_prefix": availability.local_hbm,
                        "host_prefix": availability.local_host,
                        "remaining_recovery_s": recovery.seconds,
                        "predicted_ttft_s": predicted_ttft,
                        "slo_feasible": int(predicted_ttft <= self.cfg.ttft_slo_s),
                    }
                )
        return rows

    def _validate(self) -> None:
        for node in self.nodes:
            if node.committed > node.capacity:
                raise AssertionError(f"node {node.nid} capacity exceeded: {node.committed}>{node.capacity}")
            if node.background_reserved < 0:
                raise AssertionError("negative background reservation")
            for sid, replica in node.replicas.items():
                if not 0 <= replica.hbm_prefix <= replica.host_prefix:
                    raise AssertionError(f"invalid prefixes sid={sid} n={node.nid}: {replica}")
                if not 0 <= replica.prepared_blocks <= replica.hbm_prefix:
                    raise AssertionError(f"invalid prepared count sid={sid} n={node.nid}: {replica}")
                if replica.hbm_prefix > self.sessions[sid].context_blocks and self.sessions[sid].active_node != node.nid:
                    raise AssertionError(f"replica exceeds completed context sid={sid} n={node.nid}")


def run_experiment(cfg: Config, seeds: list[int], output_dir: Path) -> list[dict[str, float | int | str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, float | int | str]] = []
    requests: list[dict[str, float | int | str]] = []
    for seed in seeds:
        workload = generate_workload(cfg, seed)
        for policy in POLICIES:
            summary, rows = Simulator(cfg, workload, policy, seed).run()
            summary["seed"] = seed
            summaries.append(summary)
            for row in rows:
                row["seed"] = seed
            requests.extend(rows)
    _write_csv(output_dir / "summary.csv", summaries)
    _write_csv(output_dir / "requests.csv", requests)
    return summaries


def aggregate(summaries: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    fields = (
        "slo_goodput_rps",
        "slo_attainment",
        "p99_ttft_s",
        "transfer_blocks_per_success",
        "restore_blocks_per_success",
        "recompute_blocks_per_success",
        "hbm_block_seconds_per_success",
        "unused_preparation_ratio",
    )
    rows: list[dict[str, float | str]] = []
    for policy in POLICIES:
        subset = [row for row in summaries if row["policy"] == policy]
        result: dict[str, float | str] = {"policy": policy}
        for field_name in fields:
            result[field_name] = statistics.mean(float(row[field_name]) for row in subset)
        rows.append(result)
    return rows


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
