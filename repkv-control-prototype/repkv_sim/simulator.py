from __future__ import annotations

import csv
import math
import random
import statistics
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from .planner import (
    PrefixAvailability,
    PreparationPlan,
    RecoveryPlan,
    WorkSegment,
    at_least_one_success,
    cheapest_preparation,
    decide_minimum_slo_preparation,
    fastest_recovery,
    routable_probability,
    success_probability,
)
from .predictor import ReturnModel, SessionClass
from .resources import ResourceModel


POLICIES = ("on_demand", "eager_full", "repkv")


RESOURCES = ("compute", "link", "host")


@dataclass(frozen=True)
class Config:
    nodes: int = 4
    sessions: int = 48
    horizon_s: float = 900.0
    step_s: float = 1.0
    ttft_slo_s: float = 3.0
    resources: ResourceModel = ResourceModel()
    hbm_blocks_override: int = 0
    preparation_horizon_s: float = 60.0
    queue_uncertainty_s: float = 0.25
    shared_uncertainty_s: float = 0.45
    queue_uncertainty_scale: float = 0.8
    shared_uncertainty_scale: float = 0.5
    residual_alpha: float = 0.05
    ttft_residual_z: float = 1.0
    already_feasible_probability: float = 0.9
    unique_copy_persist: float = 0.5
    min_copy_persist: float = 0.15
    reprepare_cost_weight: float = 1.0
    spare_replica_factor: float = 0.01
    displacement_cost_weight: float = 1.0
    value_based_eviction: bool = True
    block_batch: int = 128
    high_watermark: float = 0.92
    low_watermark: float = 0.82
    max_prepared_nodes_per_session: int = 2
    resource_penalty: float = 0.22
    arrival_forecast: bool = True
    output_estimate_weight: float = 0.2
    first_prompt_tokens: int = 4000
    follow_prompt_tokens: int = 3000
    output_tokens: int = 1500
    size_sigma: float = 0.30
    think_sigma: float = 0.55
    think_scale: float = 1.0
    think_floor_s: float = 15.0
    think_ceiling_s: float = 900.0
    predictor_sigma_scale: float = 1.0
    min_return_probability: float = 0.05
    concurrent_sessions: int = 0

    @property
    def hbm_blocks(self) -> int:
        return self.hbm_blocks_override or self.resources.hbm_blocks

    @property
    def block_tokens(self) -> int:
        return self.resources.model.block_tokens

    @property
    def prefill_blocks_s(self) -> float:
        return self.resources.prefill_blocks_s

    @property
    def decode_blocks_s(self) -> float:
        return self.resources.decode_blocks_s

    @property
    def decode_slots(self) -> int:
        return self.resources.decode_slots

    @property
    def host_blocks(self) -> int:
        return self.resources.host_blocks

    @cached_property
    def assumed_output_blocks(self) -> float:
        return max(1.0, self.output_tokens / self.block_tokens)

    # These are read on nearly every recovery computation, so they are built
    # once per Config rather than on each access.
    @cached_property
    def foreground_rates(self) -> dict[str, float]:
        return self.resources.rates

    @cached_property
    def background_rates(self) -> dict[str, float]:
        """Background work runs at the same physical rate as foreground work.

        There is no fixed background fraction. A background batch is only
        dispatched onto a resource that still has idle time in the current
        control period, and once dispatched it occupies that resource exactly
        as foreground work does, so the sharing is decided by contention rather
        than by a constant.
        """
        return self.resources.rates

    @cached_property
    def method_costs(self) -> dict[str, float]:
        """Seconds of a serial resource consumed per block.

        Using the inverse rate makes the planner minimise resource-seconds,
        which is the quantity foreground work competes for, instead of an
        arbitrary preference ordering between methods.
        """
        return {method: 1.0 / rate for method, rate in self.resources.rates.items()}

    @cached_property
    def resource_of_method(self) -> dict[str, str]:
        return self.resources.resource_of_method


@dataclass(frozen=True)
class Turn:
    """One turn of a session.

    `think_time_s` is the delay from the previous turn's completion, not from
    its arrival: a user cannot send the next question before receiving the
    previous answer. Turn 0 uses the session start time instead.
    """

    sid: int
    index: int
    think_time_s: float
    prompt_blocks: int
    output_blocks: int
    history_blocks: int
    predicted_prompt_blocks: int


@dataclass(frozen=True)
class Workload:
    """A closed-loop script: sizes and think times are fixed, arrivals are not.

    Every policy replays the same script, so prompt sizes, output sizes, think
    times and continuation decisions are identical. Arrival times differ
    because they follow each policy's own completion times.
    """

    turns: tuple[Turn, ...]
    session_start: dict[int, float]
    classes: dict[int, SessionClass]

    def script(self, sid: int, index: int) -> Turn | None:
        return self._index.get((sid, index))

    @property
    def max_context_blocks(self) -> int:
        return max(
            (turn.history_blocks + turn.prompt_blocks + turn.output_blocks for turn in self.turns),
            default=0,
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", {(turn.sid, turn.index): turn for turn in self.turns})

    def model(self, sid: int, completed_turns: int, cfg: Config) -> ReturnModel:
        session_class = self.classes[sid]
        scaled = SessionClass(
            session_class.name,
            session_class.median_gap_s * cfg.think_scale,
            session_class.base_return,
        )
        return scaled.model(
            completed_turns,
            cfg.think_sigma * cfg.predictor_sigma_scale,
            cfg.think_floor_s * cfg.think_scale,
            cfg.think_ceiling_s * cfg.think_scale,
        )


@dataclass(frozen=True)
class Prediction:
    """What the controller believes about a session's next turn."""

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
    """One serving node with three serial resources.

    `until` holds, per resource, the time it next becomes free. Device compute
    carries prefill and recompute; the network link carries remote transfers;
    the host link carries restores from DRAM. Foreground and background work
    occupy the same timelines, so preparing KV in the background delays
    whatever wants that resource next.

    Decode is not on any of the three. Under continuous batching a request that
    has been prefilled joins the running batch and every sequence in the batch
    advances one token per step, so decode does not serialise behind other
    requests; it occupies one of a fixed number of slots. `decode_finish` holds
    the completion time of each occupied slot, so a request arriving when the
    batch is full waits for the earliest slot to free.

    Recovery that only needs the link or the host path can proceed while the
    device prefills another request, which is why the three are tracked
    separately rather than folded into one busy time.
    """

    nid: int
    capacity: int
    slots: int
    host_capacity: int = 1 << 30
    until: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in RESOURCES})
    decode_finish: list[float] = field(default_factory=list)
    replicas: dict[int, KVReplica] = field(default_factory=dict)
    active_reservations: dict[int, int] = field(default_factory=dict)
    background_reserved: int = 0

    @property
    def busy_until(self) -> float:
        """When the device can start a newly queued request's prefill."""
        return self.until["compute"]

    def slot_free_at(self, now: float) -> float:
        """When a decode slot becomes available."""
        pending = [finish for finish in self.decode_finish if finish > now]
        if len(pending) < self.slots:
            return now
        pending.sort()
        return pending[len(pending) - self.slots]

    @property
    def hbm_used(self) -> int:
        return sum(replica.hbm_prefix for replica in self.replicas.values())

    @property
    def host_used(self) -> int:
        return sum(replica.host_prefix for replica in self.replicas.values())

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
    idle_since: float = 0.0


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
    displacement_cost: float
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
    exhausted_replenishments: int = 0
    decode_seconds: float = 0.0
    host_evicted_blocks: int = 0
    # A first turn has no prior KV, so nothing can be placed for it and its
    # TTFT is set by prefilling the whole prompt. Continuation turns are the
    # ones a placement policy can act on, and are reported separately.
    continuation_requests: int = 0
    continuation_successes: int = 0
    deferred_admissions: int = 0
    background_seconds: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in RESOURCES}
    )
    foreground_seconds: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in RESOURCES}
    )
    prep_inspects: int = 0
    prep_accepted: int = 0
    prep_skip_already_feasible: int = 0
    prep_skip_queue: int = 0
    prep_skip_no_plan: int = 0
    prep_skip_nonpositive: int = 0
    prep_target_blocks: int = 0
    prep_context_blocks: int = 0
    ttft_residual_sum: float = 0.0
    ttft_residual_abs_sum: float = 0.0
    ttft_residual_n: int = 0
    queue_residual_sum: float = 0.0
    queue_residual_n: int = 0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


@dataclass
class ResidualTracker:
    """Online mean and spread of (actual - predicted) seconds.

    The fluid queue is a lower bound in expectation, so the mean residual is
    typically positive. Control decisions add this mean, and optionally a
    multiple of the spread, to the fluid point estimate. The tracker is
    updated from realized arrivals and never reads the future.
    """

    alpha: float = 0.05
    prior_std: float = 0.25
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, predicted: float, actual: float) -> None:
        err = actual - predicted
        self.count += 1
        if self.count == 1:
            self.mean = err
            self.m2 = 0.0
            return
        delta = err - self.mean
        self.mean += self.alpha * delta
        self.m2 = (1.0 - self.alpha) * (self.m2 + self.alpha * delta * delta)

    def std(self) -> float:
        if self.count < 2:
            return self.prior_std
        return math.sqrt(max(0.0, self.m2))

    def adjust(self, predicted: float, z: float = 0.0) -> float:
        if self.count < 2:
            return max(0.0, predicted)
        # The fluid point estimate is a lower bound in expectation. A negative
        # residual usually means the request arrived earlier than the median
        # wait used to evaluate the queue, not that the fluid model is high.
        # Only underestimation is written back into the point estimate.
        bias = self.mean if self.mean > 0.0 else 0.0
        return max(0.0, predicted + bias + z * self.std())


SESSION_CLASSES = (
    SessionClass("fast", 45.0, 0.88),
    SessionClass("normal", 120.0, 0.78),
    SessionClass("slow", 300.0, 0.66),
)


def generate_workload(cfg: Config, seed: int) -> Workload:
    """Generate a closed-loop session script.

    Think times are drawn once per turn and measured from the previous turn's
    completion, so the script fixes how long a user waits after reading an
    answer but not when the next request lands. Turn count is bounded by both a
    per-session cap and a Bernoulli continuation draw, so some sessions stop
    early and are right-censored samples for the predictor.

    Sizes are drawn in tokens and converted to whole KV blocks, so the workload
    is stated in the same units as the model and the hardware sheet. The first
    turn carries a long document or system prompt; later turns are short
    questions over the accumulated history.

    The controller receives the class parameters, never these draws.
    """
    rng = random.Random(seed)
    turns: list[Turn] = []
    session_start: dict[int, float] = {}
    classes: dict[int, SessionClass] = {}
    block_tokens = cfg.block_tokens
    sigma = cfg.size_sigma

    def blocks(tokens: float) -> int:
        return max(1, int(math.ceil(tokens / block_tokens)))

    for sid in range(cfg.sessions):
        session_class = rng.choice(SESSION_CLASSES)
        classes[sid] = session_class
        session_start[sid] = rng.expovariate(1 / 28.0)
        history = 0
        think = 0.0
        max_turns = rng.randint(3, 8)
        for index in range(max_turns):
            mean_prompt = cfg.first_prompt_tokens if index == 0 else cfg.follow_prompt_tokens
            prompt = blocks(mean_prompt * math.exp(rng.gauss(0.0, sigma)))
            output = blocks(cfg.output_tokens * math.exp(rng.gauss(0.0, sigma)))
            predicted_prompt = blocks(mean_prompt * math.exp(rng.gauss(0.0, 0.22)))
            turns.append(
                Turn(
                    sid=sid,
                    index=index,
                    think_time_s=think,
                    prompt_blocks=prompt,
                    output_blocks=output,
                    history_blocks=history,
                    predicted_prompt_blocks=predicted_prompt,
                )
            )
            history += prompt + output
            continue_probability = max(0.18, session_class.base_return - 0.045 * index)
            if index + 1 >= max_turns or rng.random() > continue_probability:
                break
            gap = session_class.median_gap_s * cfg.think_scale * math.exp(rng.gauss(0.0, cfg.think_sigma))
            think = max(cfg.think_floor_s * cfg.think_scale, min(gap, cfg.think_ceiling_s * cfg.think_scale))
    return Workload(tuple(turns), session_start, classes)


class Simulator:
    def __init__(self, cfg: Config, workload: Workload, policy: str, seed: int = 0):
        if policy not in POLICIES:
            raise ValueError(policy)
        largest = workload.max_context_blocks
        if largest > cfg.hbm_blocks:
            raise ValueError(
                f"hbm_blocks={cfg.hbm_blocks} cannot hold the largest session context "
                f"({largest} blocks); no node could ever serve that turn"
            )
        self.cfg = cfg
        self.workload = workload
        self.policy = policy
        self.seed = seed
        self.nodes = [
            Node(nid, cfg.hbm_blocks, cfg.decode_slots, cfg.host_blocks)
            for nid in range(cfg.nodes)
        ]
        self.sessions = {sid: SessionState(sid) for sid in range(cfg.sessions)}
        self.metrics = Metrics()
        self.request_completions: list[tuple[float, Turn, int]] = []
        self.inflight: list[InFlightBatch] = []
        self.request_rows: list[dict[str, float | int | str]] = []
        self.arrivals: dict[float, list[Turn]] = {}
        self.first_arrival: dict[tuple[int, int], float] = {}
        self.now = -cfg.step_s
        self.last_candidates: list[PlanCandidate] = []
        self.last_events: list[str] = []
        self.output_estimate = cfg.assumed_output_blocks
        self._forecast_cache: dict[int, dict[int, float]] | None = None
        self._prediction_cache: dict[int, Prediction] | None = None
        self._price_cache: dict[int, float] = {}
        self._probability_cache: dict[tuple, tuple[float, tuple[float, ...]]] = {}
        self._recovery_cache: dict[tuple[int, int, int, int], RecoveryPlan] = {}
        self._queue_residual = ResidualTracker(alpha=cfg.residual_alpha, prior_std=cfg.queue_uncertainty_s)
        self._ttft_residual = ResidualTracker(alpha=cfg.residual_alpha, prior_std=cfg.shared_uncertainty_s)
        self._pool = sorted(workload.session_start)
        live = cfg.concurrent_sessions if cfg.concurrent_sessions > 0 else len(self._pool)
        self._admitted = 0
        for sid in self._pool[:live]:
            self._admit(sid, workload.session_start[sid])

    def _admit(self, sid: int, at: float) -> None:
        first = self.workload.script(sid, 0)
        self._admitted = max(self._admitted, self._pool.index(sid) + 1)
        if first is not None:
            self._schedule(first, at)

    def _replenish(self, at: float) -> None:
        """Start the next unused session script so live concurrency is held.

        Without this the workload contains a fixed total number of turns, so
        shortening think times only front-loads the same work instead of
        sustaining load.

        Once the script pool runs out, live concurrency decays for the rest of
        the horizon and the run no longer measures the requested load. That is
        counted rather than silently tolerated, because it is easy to hit by
        raising `concurrent_sessions` without also enlarging `sessions`.
        """
        if self.cfg.concurrent_sessions <= 0:
            return
        if self._admitted >= len(self._pool):
            self.metrics.exhausted_replenishments += 1
            return
        self._admit(self._pool[self._admitted], at)

    def _schedule(self, turn: Turn, at: float) -> None:
        step = self.cfg.step_s
        slot = round(math.ceil(max(at, 0.0) / step - 1e-9) * step, 9)
        if slot > self.cfg.horizon_s:
            return
        self.arrivals.setdefault(slot, []).append(turn)

    def _invalidate(self) -> None:
        self._forecast_cache = None
        self._prediction_cache = None
        self._price_cache = {}
        self._probability_cache = {}

    def _predictions(self) -> dict[int, Prediction]:
        """Recompute every idle session's next-turn prediction for this tick.

        The predicted arrival is the conditional median remaining wait, so it
        never falls into the past. A session that has already waited longer
        than its median gap is treated as returning soon, which is what the
        conditional distribution says, rather than as having missed a deadline.
        """
        if self._prediction_cache is not None:
            return self._prediction_cache
        predictions: dict[int, Prediction] = {}
        for state in self.sessions.values():
            if state.completed_turn < 0:
                continue
            script = self.workload.script(state.sid, state.completed_turn)
            if script is None:
                continue
            model = self.workload.model(state.sid, state.completed_turn + 1, self.cfg)
            waited = max(0.0, self.now - state.idle_since)
            predictions[state.sid] = Prediction(
                predicted_arrival=self.now + model.median_residual_s(waited),
                predicted_prompt_blocks=script.predicted_prompt_blocks,
                return_probability=model.window_probability(waited, self.cfg.preparation_horizon_s),
            )
        self._prediction_cache = predictions
        return predictions

    def _prediction(self, sid: int) -> Prediction | None:
        return self._predictions().get(sid)

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
            "continuation_attainment": self.metrics.continuation_successes
            / max(1, self.metrics.continuation_requests),
            "p99_ttft_s": percentile(self.metrics.ttfts, 0.99),
            "transfer_blocks_per_success": (self.metrics.background["transfer"] + self.metrics.demand["transfer"]) / success,
            "restore_blocks_per_success": (self.metrics.background["restore"] + self.metrics.demand["restore"]) / success,
            "recompute_blocks_per_success": (self.metrics.background["recompute"] + self.metrics.demand["recompute"]) / success,
            "hbm_block_seconds_per_success": self.metrics.hbm_block_seconds / success,
            "unused_preparation_ratio": unused / max(1, self.metrics.prepared_blocks),
            "prepared_blocks": self.metrics.prepared_blocks,
            "used_prepared_blocks": self.metrics.used_prepared_blocks,
            "foreground_transfer_blocks": self.metrics.demand["transfer"],
            "prep_accept_rate": self.metrics.prep_accepted / max(1, self.metrics.prep_inspects),
            "prep_skip_already_feasible_rate": self.metrics.prep_skip_already_feasible
            / max(1, self.metrics.prep_inspects),
            "prep_skip_queue_rate": self.metrics.prep_skip_queue / max(1, self.metrics.prep_inspects),
            "prep_skip_nonpositive_rate": self.metrics.prep_skip_nonpositive / max(1, self.metrics.prep_inspects),
            "prep_target_fraction": self.metrics.prep_target_blocks
            / max(1, self.metrics.prep_context_blocks),
            "demoted_blocks": self.metrics.demoted_blocks,
            "cancelled_batches": self.metrics.cancelled_batches,
            "overlapping_turns": self.metrics.overlapping_turns,
            "exhausted_replenishments": self.metrics.exhausted_replenishments,
            "deferred_admissions": self.metrics.deferred_admissions,
            "ttft_residual_mean_s": self.metrics.ttft_residual_sum / max(1, self.metrics.ttft_residual_n),
            "ttft_residual_mae_s": self.metrics.ttft_residual_abs_sum / max(1, self.metrics.ttft_residual_n),
            "queue_residual_mean_s": self.metrics.queue_residual_sum / max(1, self.metrics.queue_residual_n),
            "queue_residual_std_s": self._queue_residual.std(),
            "ttft_residual_std_s": self._ttft_residual.std(),
        }
        return summary, self.request_rows

    def step(self) -> list[str]:
        self.now = round(self.now + self.cfg.step_s, 9)
        self.last_events = []
        self._invalidate()
        self._complete_inflight(self.now)
        self._complete_requests(self.now)
        for turn in self.arrivals.get(self.now, []):
            self._arrive(turn)
        if self.policy != "on_demand":
            self._prepare()
        self.metrics.hbm_block_seconds += sum(node.hbm_used for node in self.nodes) * self.cfg.step_s
        self._validate()
        return list(self.last_events)

    def _recovery(self, start: int, context: int, host: int, remote: int) -> RecoveryPlan:
        """Memoised `fastest_recovery`.

        Foreground rates are fixed for a run, so the plan is a pure function of
        the four prefix bounds. The same bounds recur constantly across nodes,
        sessions and ticks, and the recovery planner is the single hottest
        function in the control loop.
        """
        key = (start, context, host, remote)
        plan = self._recovery_cache.get(key)
        if plan is None:
            plan = fastest_recovery(
                start, context, PrefixAvailability(start, host, remote), self.cfg.foreground_rates
            )
            if len(self._recovery_cache) > 200_000:
                self._recovery_cache.clear()
            self._recovery_cache[key] = plan
        return plan

    def _replica(self, nid: int, sid: int) -> KVReplica:
        return self.nodes[nid].replicas.get(sid, KVReplica())

    def _availability(self, sid: int, nid: int, local_hbm_override: int | None = None) -> PrefixAvailability:
        """Prefix bounds of the three sources this node can pull from.

        The remote bound covers a peer's DRAM as well as its HBM. Both paths
        leave the peer, cross the network and land in local HBM, and the network
        is the narrower of the two links, so they take the same time. Excluding
        a peer's DRAM would make recomputation look necessary in cases where a
        real cluster would simply fetch the KV.
        """
        local = self._replica(nid, sid)
        remote = 0
        for node in self.nodes:
            if node.nid == nid:
                continue
            replica = node.replicas.get(sid)
            if replica is None:
                continue
            reachable = replica.hbm_prefix if replica.hbm_prefix > replica.host_prefix else replica.host_prefix
            if reachable > remote:
                remote = reachable
        return PrefixAvailability(
            local.hbm_prefix if local_hbm_override is None else local_hbm_override,
            local.host_prefix,
            remote,
        )

    def _pending_returns(self) -> list[tuple[float, int, Prediction, SessionState]]:
        pending: list[tuple[float, int, Prediction, SessionState]] = []
        for state in self.sessions.values():
            if state.active_until > self.now:
                continue
            prediction = self._prediction(state.sid)
            if prediction is None:
                continue
            pending.append((prediction.predicted_arrival, state.sid, prediction, state))
        return sorted(pending, key=lambda item: (item[0], item[1]))

    def _forecast(self) -> dict[int, dict[int, float]]:
        """Project per-node backlog at each session's predicted return time.

        `busy_until` only covers work a node has already admitted, while a
        preparation decision is about a moment several seconds ahead. Sessions
        are inserted in predicted arrival order, so the view recorded for a
        session covers every other session predicted to return earlier but not
        its own load. Node choice mirrors the router and the added occupancy is
        discounted by the probability that the session returns in the window.
        """
        if self._forecast_cache is not None:
            return self._forecast_cache
        forecast: dict[int, dict[int, float]] = {}
        if not self.cfg.arrival_forecast:
            self._forecast_cache = forecast
            return forecast
        count = len(self.nodes)
        slot_count = self.cfg.decode_slots
        projected = {node.nid: node.busy_until for node in self.nodes}
        slots = {
            node.nid: [finish for finish in node.decode_finish if finish > self.now]
            for node in self.nodes
        }
        decode_estimate = self.output_estimate / self.cfg.decode_blocks_s

        def slot_wait(nid: int, at: float) -> float:
            busy = sorted(finish for finish in slots[nid] if finish > at)
            if len(busy) < slot_count:
                return at
            return busy[len(busy) - slot_count]

        for arrival, sid, prediction, state in self._pending_returns():
            waits = {nid: slot_wait(nid, arrival) for nid in range(count)}
            view = {
                nid: projected[nid] if projected[nid] > waits[nid] else waits[nid]
                for nid in range(count)
            }
            forecast[sid] = view
            prompt = prediction.predicted_prompt_blocks / self.cfg.prefill_blocks_s
            best_nid, best_ttft, best_service = 0, math.inf, 0.0
            hbm, host = self._prefix_signature(sid)
            for nid in range(count):
                remote = self._remote_prefix(nid, hbm, host)
                recovery = self._recovery(hbm[nid], state.context_blocks, host[nid], remote)
                queue = max(0.0, view[nid] - arrival)
                ttft = max(queue, recovery.seconds) + prompt
                if ttft < best_ttft:
                    best_nid, best_ttft = nid, ttft
                    # Decode runs in a batch slot, not on the device timeline,
                    # so only prefill and recompute occupy the device.
                    best_service = recovery.compute_seconds + prompt
            projected[best_nid] = max(projected[best_nid], arrival) + prediction.return_probability * best_service
            slots[best_nid].append(waits[best_nid] + prediction.return_probability * decode_estimate)
        self._forecast_cache = forecast
        return forecast

    def _projected_busy_until(self, sid: int, nid: int) -> float:
        view = self._forecast().get(sid)
        if view is None:
            return self.nodes[nid].busy_until
        return view[nid]

    def _fluid_queue(self, sid: int, predicted_arrival: float, nid: int) -> float:
        return max(0.0, self._projected_busy_until(sid, nid) - predicted_arrival)

    def _predicted_queue(
        self,
        state: SessionState,
        prediction: Prediction,
        node: Node,
        z: float | None = None,
    ) -> float:
        fluid = self._fluid_queue(state.sid, prediction.predicted_arrival, node.nid)
        margin = self.cfg.ttft_residual_z if z is None else z
        return self._queue_residual.adjust(fluid, margin)

    def _predicted_ttft(
        self,
        state: SessionState,
        prediction: Prediction,
        node: Node,
        hbm_override: int | None = None,
    ) -> float:
        availability = self._availability(state.sid, node.nid, hbm_override)
        recovery = fastest_recovery(
            availability.local_hbm,
            state.context_blocks,
            availability,
            self.cfg.foreground_rates,
        )
        prompt = prediction.predicted_prompt_blocks / self.cfg.prefill_blocks_s
        queue = self._predicted_queue(state, prediction, node)
        return (queue if queue > recovery.seconds else recovery.seconds) + prompt

    def _fill_persist(self, fill: float) -> float:
        """Probability a copy survives until arrival, from current occupancy.

        Below the low watermark there is no eviction pressure, so volatile
        copies are treated as certain. Above it, persistence falls toward
        `min_copy_persist` as the tier fills.
        """
        low = self.cfg.low_watermark
        floor = self.cfg.min_copy_persist
        if fill <= low:
            return 1.0
        t = (fill - low) / max(1e-9, 1.0 - low)
        return max(floor, 1.0 - t * (1.0 - floor))

    def _peer_persist(self, nid: int, hbm: tuple[int, ...], host: tuple[int, ...]) -> float:
        """Persistence of the longest peer copy this node would transfer from."""
        best = 0
        persist = 1.0
        for node in self.nodes:
            if node.nid == nid:
                continue
            reachable = hbm[node.nid] if hbm[node.nid] > host[node.nid] else host[node.nid]
            if reachable < best:
                continue
            if hbm[node.nid] >= host[node.nid]:
                fill = node.committed / max(1, node.capacity)
            else:
                fill = node.host_used / max(1, node.host_capacity)
            peer = self._fill_persist(fill)
            if reachable > best or peer < persist:
                best = reachable
                persist = peer
        return persist

    def _source_persist(
        self,
        nid: int,
        hbm: tuple[int, ...],
        host: tuple[int, ...],
        context: int,
    ) -> float:
        """Probability host/remote copies used by this node are still there.

        Local HBM is reserved on this node, so a fully pinned prefix is
        certain. Host and remote copies are not reserved; occupancy and
        uniqueness decide how much the controller may rely on them.
        """
        local_hbm = hbm[nid]
        if local_hbm >= context:
            return 1.0
        persist = 1.0
        volatile = False
        if host[nid] > local_hbm:
            volatile = True
            node = self.nodes[nid]
            persist = min(persist, self._fill_persist(node.host_used / max(1, node.host_capacity)))
        remote = self._remote_prefix(nid, hbm, host)
        if remote > (host[nid] if host[nid] > local_hbm else local_hbm):
            volatile = True
            persist = min(persist, self._peer_persist(nid, hbm, host))
        covering = sum(1 for index in range(len(hbm)) if (hbm[index] if hbm[index] > host[index] else host[index]) >= context)
        if volatile and covering <= 1:
            persist = min(persist, self.cfg.unique_copy_persist)
        return max(self.cfg.min_copy_persist, persist)

    def _apply_prefix_override(
        self,
        hbm: tuple[int, ...],
        host: tuple[int, ...],
        override: tuple[int, ...] | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Apply a hypothetical prefix change to every node's view of the copies.

        Eviction and preparation both change one node's HBM. Other nodes can
        transfer from that HBM, so the remote bound has to move with it;
        otherwise a node would keep treating a prefix as fetchable after the
        copy that actually holds it had been dropped. A three-tuple also
        updates that node's DRAM, matching a demotion that writes the prefix
        to the host tier before shrinking HBM.
        """
        if override is None:
            return hbm, host
        nid, new_hbm, *rest = override
        changed_hbm = list(hbm)
        changed_hbm[nid] = new_hbm
        if not rest:
            return tuple(changed_hbm), host
        changed_host = list(host)
        changed_host[nid] = rest[0]
        return tuple(changed_hbm), tuple(changed_host)

    def _cluster_evaluation(
        self,
        state: SessionState,
        prediction: Prediction,
        override: tuple[int, ...] | None = None,
    ) -> tuple[float, tuple[float, ...]]:
        """Cluster success probability and per-node predicted TTFTs.

        Queue error is not a fixed number of seconds. The projected backlog of
        §8.1 is a fluid quantity, and both the amount of load that materialises
        and the node it lands on are uncertain in proportion to that backlog.
        The cluster-wide part is folded into the shared term, scaled by the mean
        projected queue; the node-specific part scales with each node's own
        projected queue. With constant uncertainty the estimate stays near one
        whenever the fluid queue is small, which is precisely the regime where
        the fluid queue is least trustworthy.

        Fluid queues are then shifted by the online residual of (actual minus
        predicted) seconds. Each node mixes a volatile TTFT (host and remote
        copies) with a durable TTFT (local HBM only) using `_source_persist`.
        """
        sid = state.sid
        hbm, host = self._apply_prefix_override(*self._prefix_signature(sid), override)
        fills = tuple((node.committed, node.host_used) for node in self.nodes)
        residual = (round(self._queue_residual.mean, 4), round(self._queue_residual.std(), 4))
        key = (sid, state.context_blocks, hbm, host, fills, residual)
        cached = self._probability_cache.get(key)
        if cached is not None:
            return cached
        context = state.context_blocks
        arrival = prediction.predicted_arrival
        prompt = prediction.predicted_prompt_blocks / self.cfg.prefill_blocks_s
        view = self._forecast().get(sid)
        count = len(self.nodes)
        volatile_ttfts: list[float] = []
        durable_ttfts: list[float] = []
        mixed_ttfts: list[float] = []
        persists: list[float] = []
        queues: list[float] = []
        z = self.cfg.ttft_residual_z
        for nid in range(count):
            remote = self._remote_prefix(nid, hbm, host)
            recovery = self._recovery(hbm[nid], context, host[nid], remote)
            durable = self._recovery(hbm[nid], context, hbm[nid], hbm[nid])
            busy = self.nodes[nid].busy_until if view is None else view[nid]
            fluid = busy - arrival
            if fluid < 0.0:
                fluid = 0.0
            queue = self._queue_residual.adjust(fluid, z)
            queues.append(queue)
            # Recovery that uses the link or the host path runs while the
            # device serves other requests, so the two overlap rather than add.
            volatile_ttft = (queue if queue > recovery.seconds else recovery.seconds) + prompt
            durable_ttft = (queue if queue > durable.seconds else durable.seconds) + prompt
            persist = self._source_persist(nid, hbm, host, context)
            volatile_ttfts.append(volatile_ttft)
            durable_ttfts.append(durable_ttft)
            persists.append(persist)
            mixed_ttfts.append(persist * volatile_ttft + (1.0 - persist) * durable_ttft)
        mean_queue = sum(queues) / count
        measured = self._ttft_residual.std() if self._ttft_residual.count >= 2 else 0.0
        shared = self.cfg.shared_uncertainty_s + self.cfg.shared_uncertainty_scale * mean_queue
        shared = max(shared, measured)
        node_uncertainties = [
            max(self.cfg.queue_uncertainty_s + self.cfg.queue_uncertainty_scale * queue, measured)
            for queue in queues
        ]
        value = routable_probability(
            volatile_ttfts,
            self.cfg.ttft_slo_s,
            shared,
            node_uncertainties,
            durable_ttfts=durable_ttfts,
            persists=persists,
        )
        result = (value, tuple(mixed_ttfts))
        self._probability_cache[key] = result
        return result

    def _cluster_probability(
        self,
        state: SessionState,
        prediction: Prediction,
        override: tuple[int, ...] | None = None,
    ) -> float:
        """Probability that the cluster serves this session's next turn in time."""
        return self._cluster_evaluation(state, prediction, override)[0]

    def _remote_prefix(
        self, nid: int, hbm: tuple[int, ...], host: tuple[int, ...]
    ) -> int:
        """Longest peer prefix reachable by transfer, including a peer's DRAM.

        A demotion writes the prefix to the host tier, so after HBM is shortened
        the remaining copy is often only on DRAM. The router already treats that
        copy as a transfer source; the forecast and the cluster probability have
        to see the same bound, otherwise they would treat a fetchable prefix as
        missing and overstate both the value of retaining HBM and the need to
        prepare a local copy.
        """
        remote = 0
        for other, prefix in enumerate(hbm):
            if other == nid:
                continue
            reachable = prefix if prefix > host[other] else host[other]
            if reachable > remote:
                remote = reachable
        return remote

    def _prefix_signature(self, sid: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The part of cluster state this session's TTFT prediction depends on.

        Only this session's own replicas enter its predicted TTFT, so using
        them in the cache key makes the memo exact across the demotions that
        happen inside a single reclamation round. The queue forecast is already
        fixed for the tick.
        """
        hbm: list[int] = []
        host: list[int] = []
        for node in self.nodes:
            replica = node.replicas.get(sid)
            if replica is None:
                hbm.append(0)
                host.append(0)
            else:
                hbm.append(replica.hbm_prefix)
                host.append(replica.host_prefix)
        return tuple(hbm), tuple(host)

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
                    other.nid != batch.nid
                    and max(
                        self._replica(other.nid, batch.sid).hbm_prefix,
                        self._replica(other.nid, batch.sid).host_prefix,
                    )
                    >= batch.end
                    for other in self.nodes
                )
            if replica.hbm_prefix != batch.start or not source_ok:
                self.metrics.cancelled_batches += 1
                self.last_events.append(f"cancel sid={batch.sid} n={batch.nid} {batch.method}[{batch.start},{batch.end})")
                continue
            replica.hbm_prefix = batch.end
            replica.prepared_blocks += batch.blocks
            replica.last_used = batch.complete_at
            if batch.method == "restore":
                # Already on this node's DRAM; a restore does not add to it.
                replica.host_prefix = max(replica.host_prefix, batch.end)
            self.metrics.prepared_blocks += batch.blocks
            self.metrics.background[batch.method] += batch.blocks
            self.last_events.append(f"complete sid={batch.sid} n={batch.nid} {batch.method}[{batch.start},{batch.end})")

    def _complete_requests(self, now: float) -> None:
        ready = [item for item in self.request_completions if item[0] <= now]
        self.request_completions = [item for item in self.request_completions if item[0] > now]
        for completion, turn, nid in sorted(ready, key=lambda item: (item[0], item[1].sid, item[1].index)):
            self._invalidate()
            node = self.nodes[nid]
            state = self.sessions[turn.sid]
            weight = self.cfg.output_estimate_weight
            self.output_estimate += weight * (turn.output_blocks - self.output_estimate)
            context = turn.history_blocks + turn.prompt_blocks + turn.output_blocks
            node.active_reservations.pop(turn.sid, None)
            replica = node.replicas.setdefault(turn.sid, KVReplica())
            replica.hbm_prefix = context
            replica.prepared_blocks = 0
            replica.last_used = completion
            self._write_host(nid, turn.sid, context, completion)
            state.completed_turn = turn.index
            state.context_blocks = context
            state.active_until = 0.0
            state.active_node = None
            state.idle_since = completion
            self.last_events.append(f"finish sid={turn.sid} turn={turn.index} n={nid} context={context}")
            following = self.workload.script(turn.sid, turn.index + 1)
            if following is not None:
                self._schedule(following, completion + following.think_time_s)
            else:
                self._replenish(completion)

    def _service_ttft(
        self,
        node: Node,
        segments: tuple[WorkSegment, ...],
        prompt_s: float,
        commit: bool,
    ) -> float:
        """Time to first token when this request is served by this node.

        Recovery segments run in order, each queued behind whatever already
        holds its resource, so a restore waits for the host link but not for
        the device. Prefill then needs the device and a free decode slot. With
        `commit` the resource timelines are advanced; without it the same
        arithmetic is used to score the node without side effects.
        """
        rates = self.cfg.foreground_rates
        resource_of = self.cfg.resource_of_method
        held = dict(node.until)
        ready = self.now
        for segment in segments:
            resource = resource_of[segment.method]
            start = held[resource] if held[resource] > ready else ready
            ready = start + segment.blocks / rates[segment.method]
            held[resource] = ready
            if commit:
                self.metrics.foreground_seconds[resource] += ready - start
        slot = node.slot_free_at(self.now)
        if slot > ready:
            ready = slot
        start = held["compute"] if held["compute"] > ready else ready
        first_token = start + prompt_s
        held["compute"] = first_token
        if commit:
            self.metrics.foreground_seconds["compute"] += prompt_s
            node.until.update(held)
        return first_token - self.now

    def _arrive(self, turn: Turn) -> None:
        self._invalidate()
        state = self.sessions[turn.sid]
        arrived = self.first_arrival.setdefault((turn.sid, turn.index), self.now)
        if state.active_until > self.now:
            self.metrics.overlapping_turns += 1
            self.last_events.append(f"overlap sid={turn.sid} turn={turn.index}")
        final_context = turn.history_blocks + turn.prompt_blocks + turn.output_blocks
        prompt_s = turn.prompt_blocks / self.cfg.prefill_blocks_s
        choices: list[tuple[float, int, tuple[WorkSegment, ...], int]] = []
        for node in self.nodes:
            availability = self._availability(turn.sid, node.nid)
            recovery = fastest_recovery(
                availability.local_hbm,
                turn.history_blocks,
                availability,
                self.cfg.foreground_rates,
            )
            ttft = self._service_ttft(node, recovery.segments, prompt_s, commit=False)
            extra = max(0, final_context - self._replica(node.nid, turn.sid).hbm_prefix)
            if self._available_capacity(node.nid, turn.sid) >= extra:
                choices.append((ttft, node.nid, recovery.segments, extra))
        if not choices:
            # No node can hold this context even after reclaiming every
            # evictable prefix. A real serving stack queues the request rather
            # than failing it, so the turn is retried on the next control
            # period and the wait is charged to its TTFT.
            self.metrics.deferred_admissions += 1
            self.last_events.append(f"defer sid={turn.sid} turn={turn.index} context={final_context}")
            self._schedule(turn, self.now + self.cfg.step_s)
            return
        ttft, nid, recovery_segments, extra = min(choices, key=lambda item: (item[0], item[1]))
        if not self._ensure_capacity(nid, extra, turn.sid):
            raise RuntimeError("capacity precheck and eviction disagree")
        node = self.nodes[nid]
        raw_ttft: float | None = None
        if turn.index > 0:
            prediction = self._prediction(turn.sid)
            if prediction is not None:
                fluid_queue = self._fluid_queue(turn.sid, prediction.predicted_arrival, nid)
                actual_queue = max(
                    0.0,
                    node.slot_free_at(self.now) - self.now,
                    node.until["compute"] - self.now,
                )
                err_q = actual_queue - fluid_queue
                close = abs(self.now - prediction.predicted_arrival) <= max(5.0, 5.0 * self.cfg.step_s)
                if close:
                    self._queue_residual.update(fluid_queue, actual_queue)
                    self.metrics.queue_residual_sum += err_q
                    self.metrics.queue_residual_n += 1
                recovery_s = sum(
                    segment.blocks / self.cfg.foreground_rates[segment.method]
                    for segment in recovery_segments
                )
                prompt_hat = prediction.predicted_prompt_blocks / self.cfg.prefill_blocks_s
                raw_ttft = (fluid_queue if fluid_queue > recovery_s else recovery_s) + prompt_hat
                if not close:
                    raw_ttft = None
        node.active_reservations[turn.sid] = final_context
        replica = node.replicas.get(turn.sid)
        if replica:
            used = min(replica.prepared_blocks, turn.history_blocks)
            replica.prepared_blocks -= used
            replica.last_used = self.now
            self.metrics.used_prepared_blocks += used
        for segment in recovery_segments:
            self.metrics.demand[segment.method] += segment.blocks
        service_ttft = self._service_ttft(node, recovery_segments, prompt_s, commit=True)
        if raw_ttft is not None:
            self._ttft_residual.update(raw_ttft, service_ttft)
            err = service_ttft - raw_ttft
            self.metrics.ttft_residual_sum += err
            self.metrics.ttft_residual_abs_sum += abs(err)
            self.metrics.ttft_residual_n += 1
        ttft = service_ttft + (self.now - arrived)
        decode_s = turn.output_blocks / self.cfg.decode_blocks_s
        completion = self.now + service_ttft + decode_s
        node.decode_finish = [finish for finish in node.decode_finish if finish > self.now]
        node.decode_finish.append(completion)
        self.metrics.decode_seconds += decode_s
        state.active_until = completion
        state.active_node = nid
        self.request_completions.append((completion, turn, nid))
        success = ttft <= self.cfg.ttft_slo_s
        self.metrics.requests += 1
        self.metrics.successes += int(success)
        if turn.index > 0:
            self.metrics.continuation_requests += 1
            self.metrics.continuation_successes += int(success)
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
        raw: list[tuple[SessionState, Prediction, Node, PreparationPlan, float, float]] = []
        for state in self.sessions.values():
            if state.active_until > self.now:
                continue
            prediction = self._prediction(state.sid)
            if prediction is None:
                continue
            if prediction.return_probability < self.cfg.min_return_probability:
                continue
            time_left = max(self.cfg.step_s, prediction.predicted_arrival - self.now)
            before = self._cluster_probability(state, prediction)
            for node in self.nodes:
                availability = self._availability(state.sid, node.nid)
                if availability.local_hbm >= state.context_blocks:
                    continue
                queue = self._predicted_queue(state, prediction, node, z=0.0)
                prompt = prediction.predicted_prompt_blocks / self.cfg.prefill_blocks_s
                self.metrics.prep_inspects += 1
                if self.policy == "eager_full":
                    plan = cheapest_preparation(
                        availability.local_hbm,
                        state.context_blocks,
                        availability,
                        time_left,
                        self.cfg.background_rates,
                        self.cfg.method_costs,
                    )
                    reason = "ok" if plan is not None and plan.segments else "no_timely_plan"
                else:
                    hbm, host = self._prefix_signature(state.sid)
                    persist = self._source_persist(node.nid, hbm, host, state.context_blocks)
                    plan, reason = decide_minimum_slo_preparation(
                        state.context_blocks,
                        availability,
                        queue,
                        prompt,
                        self.cfg.ttft_slo_s,
                        time_left,
                        self.cfg.foreground_rates,
                        self.cfg.background_rates,
                        self.cfg.method_costs,
                        persist,
                        self.cfg.already_feasible_probability,
                    )
                if plan is None or not plan.segments:
                    if reason == "already_feasible":
                        self.metrics.prep_skip_already_feasible += 1
                    elif reason == "queue_exceeds_budget":
                        self.metrics.prep_skip_queue += 1
                    else:
                        self.metrics.prep_skip_no_plan += 1
                    continue
                after = self._cluster_probability(state, prediction, (node.nid, plan.target_prefix))
                added = plan.target_prefix - availability.local_hbm
                gain = prediction.return_probability * max(0.0, after - before)
                displacement = self.cfg.displacement_cost_weight * self._displacement_cost(node.nid, added)
                hbm_cost = added * time_left / max(
                    1.0, node.capacity * self.cfg.preparation_horizon_s
                )
                raw.append((state, prediction, node, plan, gain, displacement, hbm_cost))
        if not raw:
            return []
        demand = {
            method: sum(
                plan.blocks(method)
                / max(1.0, self.cfg.background_rates[method] * max(self.cfg.step_s, prediction.predicted_arrival - self.now))
                for _, prediction, _, plan, _, _, _ in raw
            )
            for method in self.cfg.background_rates
        }
        hbm_pressure = sum(node.committed for node in self.nodes) / max(1, sum(node.capacity for node in self.nodes))
        prices = {method: 1.0 + max(0.0, demand[method] - 1.0) ** 2 for method in demand}
        candidates: list[PlanCandidate] = []
        for state, prediction, node, plan, gain, displacement, hbm_cost in raw:
            resource_cost = hbm_cost * (1.0 + 3.0 * max(0.0, hbm_pressure - self.cfg.low_watermark))
            for method in self.cfg.background_rates:
                denominator = self.cfg.background_rates[method] * self.cfg.preparation_horizon_s
                resource_cost += prices[method] * plan.blocks(method) / max(1.0, denominator)
            if self.policy == "eager_full":
                # Always-prepare probe: not a dominance-safe controller.
                # No-op has value 0; this surrogate is almost always positive.
                net = 1.0 - 0.01 * resource_cost
                score = 1.0 / max(resource_cost, 1e-9)
            else:
                net = gain - displacement - self.cfg.resource_penalty * resource_cost
                score = net / max(resource_cost + displacement, 1e-9)
            slack = prediction.predicted_arrival - self.now - plan.seconds
            if net > 0:
                self.metrics.prep_accepted += 1
                self.metrics.prep_target_blocks += plan.target_prefix
                self.metrics.prep_context_blocks += state.context_blocks
                candidates.append(
                    PlanCandidate(
                        state.sid,
                        node.nid,
                        prediction.predicted_arrival,
                        prediction.return_probability,
                        plan,
                        gain,
                        displacement,
                        resource_cost,
                        net,
                        score,
                        slack,
                    )
                )
            else:
                self.metrics.prep_skip_nonpositive += 1
        return candidates

    def _prepare(self) -> None:
        self._invalidate()
        self.last_candidates = self._collect_candidates()
        horizon = self.now + self.cfg.step_s
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
            node = self.nodes[candidate.nid]
            resource = self.cfg.resource_of_method[segment.method]
            # Background work is opportunistic: it may only claim a resource
            # that still has idle time left in this control period, and it then
            # occupies that resource exactly as foreground work does.
            free_at = node.until[resource]
            if free_at >= horizon:
                continue
            amount = min(segment.blocks, self.cfg.block_batch)
            if amount <= 0:
                continue
            end = current + amount
            if segment.method == "restore" and self._replica(candidate.nid, candidate.sid).host_prefix < end:
                continue
            if segment.method == "transfer" and not any(
                other.nid != candidate.nid
                and max(
                    self._replica(other.nid, candidate.sid).hbm_prefix,
                    self._replica(other.nid, candidate.sid).host_prefix,
                )
                >= end
                for other in self.nodes
            ):
                continue
            if not self._ensure_capacity(candidate.nid, amount, candidate.sid):
                continue
            node.background_reserved += amount
            start = free_at if free_at > self.now else self.now
            complete_at = start + amount / self.cfg.background_rates[segment.method]
            node.until[resource] = complete_at
            self.metrics.background_seconds[resource] += complete_at - start
            batch = InFlightBatch(candidate.sid, candidate.nid, segment.method, current, end, complete_at)
            self.inflight.append(batch)
            selected_sessions.add(candidate.sid)
            self.last_events.append(
                f"dispatch sid={candidate.sid} n={candidate.nid} {segment.method}[{current},{end}) "
                f"target={candidate.plan.target_prefix} net={candidate.net_value:.4f}"
            )

    def _displacement_price(self, nid: int) -> float:
        """Marginal expected cost per HBM block of freeing space on this node.

        A preparation that does not fit in free capacity is paid for by
        demoting some other session's prefix, and `_ensure_capacity` will pick
        the cheapest victim available at that moment. Charging the preparation
        the victim's own eviction score is what makes the two sides one
        decision: without it the candidate sees what it gains and not what it
        displaces, so preparation performs an unpriced transfer of HBM.

        The price is the cheapest victim's score, evaluated once per tick per
        node. It ignores that the requesting session is itself protected from
        eviction; taking the minimum over the remaining sessions makes that
        omission immaterial in all but degenerate cases.
        """
        cached = self._price_cache.get(nid)
        if cached is not None:
            return cached
        node = self.nodes[nid]
        victims = [
            sid for sid, replica in node.replicas.items()
            if replica.hbm_prefix > 0 and self.sessions[sid].active_node != nid
        ]
        price = min((self._eviction_score(sid, nid) for sid in victims), default=0.0)
        self._price_cache[nid] = price
        return price

    def _displacement_cost(self, nid: int, blocks: int) -> float:
        """Expected cost of the demotions this preparation would force.

        Reclamation is triggered at the high watermark, not at full capacity,
        so the headroom that matters is the distance to that watermark. Only
        the blocks that must be freed to keep this preparation under the
        trigger are charged; the extra depth of the drop to the low watermark
        is hysteresis and would occur on the next trigger regardless of which
        action pulled it.
        """
        node = self.nodes[nid]
        trigger = int(node.capacity * self.cfg.high_watermark)
        deficit = node.committed + blocks - trigger
        if deficit <= 0:
            return 0.0
        return self._displacement_price(nid) * deficit

    def _write_host(self, nid: int, sid: int, prefix: int, at: float) -> None:
        """Record a prefix on the node's DRAM tier, evicting to make room.

        The host tier is finite, so a session's KV does not stay recoverable
        forever. Once the last copy of a prefix is gone from every node's HBM
        and DRAM, the only way back is recomputation. Host eviction is
        least-recently-used: the value ordering used for HBM is not applied
        here, and whether it should be is a separate question.
        """
        node = self.nodes[nid]
        replica = node.replicas.setdefault(sid, KVReplica(last_used=at))
        if prefix <= replica.host_prefix:
            return
        replica.host_prefix = prefix
        used = node.host_used
        if used <= node.host_capacity:
            return
        victims = sorted(
            (other for other, other_replica in node.replicas.items()
             if other != sid and other_replica.host_prefix > 0),
            key=lambda other: node.replicas[other].last_used,
        )
        for victim in victims:
            if used <= node.host_capacity:
                break
            other = node.replicas[victim]
            dropped = other.host_prefix
            other.host_prefix = 0
            used -= dropped
            self.metrics.host_evicted_blocks += dropped
            self.last_events.append(f"host-evict sid={victim} n={nid} {dropped} blocks")
        if used > node.host_capacity:
            # Nothing else to release: the write itself has to be truncated.
            keep = max(0, replica.host_prefix - (used - node.host_capacity))
            self.metrics.host_evicted_blocks += replica.host_prefix - keep
            replica.host_prefix = keep

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
            if self.policy == "repkv" and self.cfg.value_based_eviction:
                victim = min(candidates, key=lambda sid: self._eviction_score(sid, nid))
            else:
                victim = min(candidates, key=lambda sid: node.replicas[sid].last_used)
            replica = node.replicas[victim]
            amount = min(self.cfg.block_batch, replica.hbm_prefix, node.committed + extra - target)
            old_prefix = replica.hbm_prefix
            self._write_host(nid, victim, old_prefix, self.now)
            replica.hbm_prefix -= amount
            prepared = min(amount, replica.prepared_blocks)
            replica.prepared_blocks -= prepared
            self.metrics.demoted_prepared_blocks += prepared
            self.metrics.demoted_blocks += amount
            self.last_events.append(f"demote sid={victim} n={nid} [{replica.hbm_prefix},{old_prefix})")
        return node.committed + extra <= node.capacity

    def _eviction_score(self, sid: int, nid: int) -> float:
        """Expected cost per HBM block freed, in the units used by preparation.

        The first term is the loss of cluster SLO success, not the loss of
        success on this node. A prefix that another node can already serve
        inside the deadline therefore has little routing value here, even if
        this node would itself miss the deadline after the demotion. The
        hypothetical after-state writes the current HBM prefix to DRAM and
        then shortens HBM, matching `_ensure_capacity`, and other nodes see
        that HBM go away as a transfer source.

        The second term is the demotion work itself. The third is the
        background restore needed to put the interval back before the
        predicted return. That rebuild is charged only when no other node
        remains predicted-SLO-feasible without this HBM: a spare replica
        does not have to be rebuilt on the node that dropped it. Without the
        rebuild term the controller can pay for a preparation and then
        reclaim a unique copy in the same horizon at no recorded cost.

        Only a session with no remaining turn in its script has no predicted
        return and is therefore free to reclaim.
        """
        state = self.sessions[sid]
        replica = self._replica(nid, sid)
        amount = min(self.cfg.block_batch, replica.hbm_prefix)
        if amount <= 0:
            return 0.0
        prediction = self._prediction(sid)
        if prediction is None:
            return 0.0
        new_hbm = replica.hbm_prefix - amount
        new_host = replica.host_prefix if replica.host_prefix > replica.hbm_prefix else replica.hbm_prefix
        before, _ = self._cluster_evaluation(state, prediction)
        after, ttfts_after = self._cluster_evaluation(state, prediction, (nid, new_hbm, new_host))
        expected_loss = prediction.return_probability * max(0.0, before - after)
        demotion_cost = 0.002 * amount
        other_covers = any(
            ttft <= self.cfg.ttft_slo_s for other, ttft in enumerate(ttfts_after) if other != nid
        )
        if other_covers:
            # Another node still predicted-SLO-feasible without this HBM.
            # The copy is spare: do not charge a local rebuild, and shrink
            # the demotion floor so it ranks below any copy the cluster still
            # needs. The remaining expected_loss is only the diversification
            # value of a second feasible node.
            demotion_cost *= self.cfg.spare_replica_factor
            reprepare_cost = 0.0
        else:
            rebuild_work = amount / max(
                1.0, self.cfg.background_rates["restore"] * self.cfg.preparation_horizon_s
            )
            reprepare_cost = (
                self.cfg.reprepare_cost_weight
                * prediction.return_probability
                * self.cfg.resource_penalty
                * rebuild_work
            )
        return (expected_loss + demotion_cost + reprepare_cost) / amount

    def readiness_snapshot(self) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for state in self.sessions.values():
            if state.completed_turn < 0:
                continue
            prediction = self._prediction(state.sid)
            for node in self.nodes:
                availability = self._availability(state.sid, node.nid)
                recovery = fastest_recovery(
                    availability.local_hbm,
                    state.context_blocks,
                    availability,
                    self.cfg.foreground_rates,
                )
                predicted_ttft = (
                    self._predicted_ttft(state, prediction, node) if prediction else recovery.seconds
                )
                rows.append(
                    {
                        "sid": state.sid,
                        "node": node.nid,
                        "valid_hbm_prefix": availability.local_hbm,
                        "host_prefix": availability.local_host,
                        "remaining_recovery_s": recovery.seconds,
                        "return_probability": prediction.return_probability if prediction else 0.0,
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
                if replica.hbm_prefix < 0 or replica.host_prefix < 0:
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
        "requests",
        "slo_goodput_rps",
        "slo_attainment",
        "continuation_attainment",
        "p99_ttft_s",
        "transfer_blocks_per_success",
        "restore_blocks_per_success",
        "recompute_blocks_per_success",
        "hbm_block_seconds_per_success",
        "unused_preparation_ratio",
        "prepared_blocks",
        "used_prepared_blocks",
        "foreground_transfer_blocks",
        "prep_accept_rate",
        "prep_skip_already_feasible_rate",
        "prep_skip_queue_rate",
        "prep_skip_nonpositive_rate",
        "prep_target_fraction",
        "deferred_admissions",
        "exhausted_replenishments",
        "ttft_residual_mean_s",
        "ttft_residual_mae_s",
        "queue_residual_mean_s",
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
