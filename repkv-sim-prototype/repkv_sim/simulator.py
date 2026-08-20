from __future__ import annotations

import csv
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path


POLICIES = ("on_demand", "eager_full", "repkv")


@dataclass(frozen=True)
class Config:
    nodes: int = 4
    sessions: int = 40
    horizon_s: float = 220.0
    step_s: float = 0.5
    hbm_blocks: int = 180
    ttft_slo_s: float = 2.0
    prefill_blocks_s: float = 24.0
    decode_blocks_s: float = 10.0
    transfer_blocks_s: float = 20.0
    recompute_blocks_s: float = 12.0
    background_transfer_fraction: float = 0.45
    background_compute_fraction: float = 0.30
    preparation_horizon_s: float = 18.0
    prediction_error_s: float = 5.0
    prompt_prediction_error: float = 0.25
    queue_uncertainty_s: float = 0.55
    block_batch: int = 4


@dataclass(frozen=True)
class Turn:
    sid: int
    index: int
    arrival: float
    prompt_blocks: int
    output_blocks: int
    history_blocks: int
    predicted_arrival: float | None
    predicted_prompt_blocks: int
    return_probability: float


@dataclass
class CacheEntry:
    blocks: int = 0
    prepared_blocks: int = 0
    last_used: float = 0.0


@dataclass
class Node:
    nid: int
    capacity: int
    busy_until: float = 0.0
    cache: dict[int, CacheEntry] = field(default_factory=dict)

    @property
    def used(self) -> int:
        return sum(entry.blocks for entry in self.cache.values())


@dataclass
class SessionState:
    sid: int
    completed_turn: int = -1
    context_blocks: int = 0
    active_until: float = 0.0
    active_node: int | None = None


@dataclass
class Metrics:
    requests: int = 0
    successes: int = 0
    ttfts: list[float] = field(default_factory=list)
    bg_transfer_blocks: int = 0
    bg_recompute_blocks: int = 0
    demand_transfer_blocks: int = 0
    demand_recompute_blocks: int = 0
    hbm_block_seconds: float = 0.0
    prepared_blocks: int = 0
    used_prepared_blocks: int = 0
    evicted_prepared_blocks: int = 0
    overlapping_turns: int = 0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def generate_workload(cfg: Config, seed: int) -> list[Turn]:
    rng = random.Random(seed)
    turns: list[Turn] = []
    for sid in range(cfg.sessions):
        turn_count = rng.randint(3, 7)
        arrival = rng.expovariate(1 / 20.0)
        history = 0
        raw: list[tuple[float, int, int, int]] = []
        for index in range(turn_count):
            if index:
                # Correlated returns create bursts without fixing a node by hand.
                gap = rng.lognormvariate(math.log(16.0), 0.65)
                arrival += max(10.0, min(gap, 60.0))
            prompt = rng.randint(3, 9) if index else rng.randint(8, 16)
            output = rng.randint(2, 7)
            raw.append((arrival, prompt, output, history))
            history += prompt + output

        for index, (arrival, prompt, output, history) in enumerate(raw):
            if index + 1 < len(raw):
                next_arrival, next_prompt, _, _ = raw[index + 1]
                error = rng.gauss(0.0, cfg.prediction_error_s)
                predicted_arrival = max(arrival + cfg.step_s, next_arrival + error)
                prompt_error = rng.gauss(0.0, cfg.prompt_prediction_error)
                predicted_prompt = max(1, round(next_prompt * (1 + prompt_error)))
                probability = max(0.15, min(0.98, 0.90 - abs(error) / 35.0))
            else:
                # Some completed sessions look as if they may return but never do.
                if rng.random() < 0.35:
                    predicted_arrival = arrival + rng.lognormvariate(math.log(15.0), 0.6)
                    predicted_prompt = rng.randint(3, 9)
                    probability = rng.uniform(0.20, 0.50)
                else:
                    predicted_arrival = None
                    predicted_prompt = 0
                    probability = 0.0
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
                    return_probability=probability,
                )
            )
    return sorted((turn for turn in turns if turn.arrival <= cfg.horizon_s), key=lambda x: (x.arrival, x.sid))


class Simulator:
    def __init__(self, cfg: Config, turns: list[Turn], policy: str, seed: int):
        if policy not in POLICIES:
            raise ValueError(policy)
        self.cfg = cfg
        self.turns = turns
        self.policy = policy
        self.rng = random.Random(seed + 1000)
        self.nodes = [Node(i, cfg.hbm_blocks) for i in range(cfg.nodes)]
        self.sessions = {sid: SessionState(sid) for sid in range(cfg.sessions)}
        self.metrics = Metrics()
        self.completions: list[tuple[float, Turn, int]] = []
        self.request_rows: list[dict[str, float | int | str]] = []
        self.turn_lookup = {(t.sid, t.index): t for t in turns}

    def run(self) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
        arrivals: dict[float, list[Turn]] = {}
        for turn in self.turns:
            arrivals.setdefault(turn.arrival, []).append(turn)

        steps = int(self.cfg.horizon_s / self.cfg.step_s) + 1
        for step in range(steps):
            now = round(step * self.cfg.step_s, 9)
            self._complete_requests(now)
            if self.policy != "on_demand":
                self._prepare(now)
            for turn in arrivals.get(now, []):
                self._arrive(now, turn)
            self.metrics.hbm_block_seconds += sum(n.used for n in self.nodes) * self.cfg.step_s

        self._complete_requests(float("inf"))
        remaining_prepared = sum(e.prepared_blocks for n in self.nodes for e in n.cache.values())
        unused = self.metrics.evicted_prepared_blocks + remaining_prepared
        p99 = percentile(self.metrics.ttfts, 0.99)
        success = max(self.metrics.successes, 1)
        summary: dict[str, float | int | str] = {
            "policy": self.policy,
            "requests": self.metrics.requests,
            "slo_successes": self.metrics.successes,
            "slo_goodput_rps": self.metrics.successes / self.cfg.horizon_s,
            "slo_attainment": self.metrics.successes / max(self.metrics.requests, 1),
            "p99_ttft_s": p99,
            "bg_transfer_blocks": self.metrics.bg_transfer_blocks,
            "bg_recompute_blocks": self.metrics.bg_recompute_blocks,
            "demand_transfer_blocks": self.metrics.demand_transfer_blocks,
            "demand_recompute_blocks": self.metrics.demand_recompute_blocks,
            "hbm_block_seconds": self.metrics.hbm_block_seconds,
            "transfer_blocks_per_success": (
                self.metrics.bg_transfer_blocks + self.metrics.demand_transfer_blocks
            ) / success,
            "recompute_blocks_per_success": (
                self.metrics.bg_recompute_blocks + self.metrics.demand_recompute_blocks
            ) / success,
            "hbm_block_seconds_per_success": self.metrics.hbm_block_seconds / success,
            "prepared_blocks": self.metrics.prepared_blocks,
            "unused_preparation_ratio": unused / max(self.metrics.prepared_blocks, 1),
            "overlapping_turns": self.metrics.overlapping_turns,
        }
        return summary, self.request_rows

    def _complete_requests(self, now: float) -> None:
        ready = [item for item in self.completions if item[0] <= now]
        self.completions = [item for item in self.completions if item[0] > now]
        for completion, turn, nid in sorted(ready, key=lambda item: (item[0], item[1].sid, item[1].index)):
            state = self.sessions[turn.sid]
            state.completed_turn = turn.index
            state.context_blocks = turn.history_blocks + turn.prompt_blocks + turn.output_blocks
            state.active_until = 0.0
            state.active_node = None
            self._ensure_capacity(nid, state.context_blocks, completion, protected_sid=turn.sid)
            entry = self.nodes[nid].cache.setdefault(turn.sid, CacheEntry())
            entry.blocks = state.context_blocks
            entry.prepared_blocks = 0
            entry.last_used = completion

    def _arrive(self, now: float, turn: Turn) -> None:
        if self.sessions[turn.sid].active_until > now:
            self.metrics.overlapping_turns += 1
        best: tuple[float, int, str, int] | None = None
        for node in self.nodes:
            entry = node.cache.get(turn.sid, CacheEntry())
            missing = max(0, turn.history_blocks - entry.blocks)
            method, recovery = self._fastest_recovery(turn.sid, missing)
            queue = max(0.0, node.busy_until - now)
            ttft = queue + recovery + turn.prompt_blocks / self.cfg.prefill_blocks_s
            candidate = (ttft, node.nid, method, missing)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        ttft, nid, method, missing = best
        node = self.nodes[nid]
        entry = node.cache.get(turn.sid)
        if entry:
            used = min(entry.prepared_blocks, turn.history_blocks)
            self.metrics.used_prepared_blocks += used
            entry.prepared_blocks -= used
            entry.last_used = now
        if missing:
            if method == "transfer":
                self.metrics.demand_transfer_blocks += missing
            else:
                self.metrics.demand_recompute_blocks += missing

        start = max(now, node.busy_until)
        completion = start + (ttft - max(0.0, node.busy_until - now)) + turn.output_blocks / self.cfg.decode_blocks_s
        node.busy_until = completion
        state = self.sessions[turn.sid]
        state.active_until = completion
        state.active_node = nid
        self.completions.append((completion, turn, nid))

        success = ttft <= self.cfg.ttft_slo_s
        self.metrics.requests += 1
        self.metrics.successes += int(success)
        self.metrics.ttfts.append(ttft)
        self.request_rows.append(
            {
                "policy": self.policy,
                "sid": turn.sid,
                "turn": turn.index,
                "arrival_s": now,
                "node": nid,
                "history_blocks": turn.history_blocks,
                "missing_blocks": missing,
                "recovery": method,
                "ttft_s": ttft,
                "slo_success": int(success),
            }
        )

    def _fastest_recovery(self, sid: int, missing: int) -> tuple[str, float]:
        if not missing:
            return "hit", 0.0
        source_exists = any(n.cache.get(sid, CacheEntry()).blocks > 0 for n in self.nodes)
        transfer = missing / self.cfg.transfer_blocks_s if source_exists else float("inf")
        recompute = missing / self.cfg.recompute_blocks_s
        return ("transfer", transfer) if transfer <= recompute else ("recompute", recompute)

    def _prepare(self, now: float) -> None:
        transfer_budget = int(self.cfg.transfer_blocks_s * self.cfg.background_transfer_fraction * self.cfg.step_s)
        compute_budget = int(self.cfg.recompute_blocks_s * self.cfg.background_compute_fraction * self.cfg.step_s)
        plans: list[tuple[float, int, int, str, int]] = []

        for state in self.sessions.values():
            if state.completed_turn < 0 or state.active_until > now:
                continue
            previous = self.turn_lookup.get((state.sid, state.completed_turn))
            if previous is None or previous.predicted_arrival is None:
                continue
            time_to_return = previous.predicted_arrival - now
            if time_to_return > self.cfg.preparation_horizon_s or time_to_return < -self.cfg.step_s:
                continue
            if self.policy == "eager_full":
                plan = self._eager_plan(state, previous, now)
            else:
                plan = self._repkv_plan(state, previous, now)
            if plan:
                plans.append(plan)

        for _, sid, nid, method, needed in sorted(plans, reverse=True):
            if needed <= 0:
                continue
            budget = transfer_budget if method == "transfer" else compute_budget
            amount = min(needed, budget, self.cfg.block_batch)
            if amount <= 0:
                continue
            if not self._ensure_capacity(nid, amount, now, protected_sid=sid):
                continue
            entry = self.nodes[nid].cache.setdefault(sid, CacheEntry(last_used=now))
            entry.blocks += amount
            entry.prepared_blocks += amount
            self.metrics.prepared_blocks += amount
            if method == "transfer":
                transfer_budget -= amount
                self.metrics.bg_transfer_blocks += amount
            else:
                compute_budget -= amount
                self.metrics.bg_recompute_blocks += amount

    def _eager_plan(self, state: SessionState, previous: Turn, now: float) -> tuple[float, int, int, str, int] | None:
        choices = []
        holders = {n.nid for n in self.nodes if n.cache.get(state.sid, CacheEntry()).blocks >= state.context_blocks}
        for node in self.nodes:
            if node.nid in holders:
                continue
            missing = max(0, state.context_blocks - node.cache.get(state.sid, CacheEntry()).blocks)
            if not missing:
                continue
            method, _ = self._fastest_recovery(state.sid, missing)
            queue = max(0.0, node.busy_until - max(now, previous.predicted_arrival or now))
            choices.append((-queue, state.sid, node.nid, method, missing))
        return max(choices) if choices else None

    def _repkv_plan(self, state: SessionState, previous: Turn, now: float) -> tuple[float, int, int, str, int] | None:
        predicted_time = max(now, previous.predicted_arrival or now)
        current_best = min(self._predicted_ttft(state.sid, state.context_blocks, previous, n, predicted_time) for n in self.nodes)
        current_success = self._success_probability(current_best)
        plans = []
        for node in self.nodes:
            entry = node.cache.get(state.sid, CacheEntry())
            missing = max(0, state.context_blocks - entry.blocks)
            if missing <= 0:
                continue
            queue = max(0.0, node.busy_until - predicted_time)
            prompt_time = previous.predicted_prompt_blocks / self.cfg.prefill_blocks_s
            method, _ = self._fastest_recovery(state.sid, missing)
            rate = self.cfg.transfer_blocks_s if method == "transfer" else self.cfg.recompute_blocks_s
            remaining_allowed = max(0, math.floor((self.cfg.ttft_slo_s - queue - prompt_time) * rate))
            needed = max(0, missing - remaining_allowed)
            if needed <= 0:
                continue
            after_missing = missing - needed
            after = queue + after_missing / rate + prompt_time
            new_success = self._success_probability(min(current_best, after))
            value = previous.return_probability * max(0.0, new_success - current_success)
            if value <= 0:
                continue
            time_left = max(self.cfg.step_s, predicted_time - now)
            if needed / rate > time_left:
                continue
            transfer_weight = 1.0 if method == "transfer" else 1.35
            memory_weight = 0.015 * max(time_left, 1.0)
            cost = needed * (transfer_weight + memory_weight)
            score = value / max(cost, 1e-9)
            plans.append((score, state.sid, node.nid, method, needed))
        return max(plans) if plans else None

    def _predicted_ttft(self, sid: int, context: int, previous: Turn, node: Node, predicted_time: float) -> float:
        local = node.cache.get(sid, CacheEntry()).blocks
        missing = max(0, context - local)
        _, recovery = self._fastest_recovery(sid, missing)
        queue = max(0.0, node.busy_until - predicted_time)
        return queue + recovery + previous.predicted_prompt_blocks / self.cfg.prefill_blocks_s

    def _success_probability(self, ttft: float) -> float:
        z = (ttft - self.cfg.ttft_slo_s) / max(self.cfg.queue_uncertainty_s, 1e-6)
        if z > 30:
            return 0.0
        if z < -30:
            return 1.0
        return 1.0 / (1.0 + math.exp(z))

    def _ensure_capacity(self, nid: int, extra: int, now: float, protected_sid: int) -> bool:
        node = self.nodes[nid]
        while node.used + extra > node.capacity:
            candidates = [sid for sid, entry in node.cache.items() if sid != protected_sid and entry.blocks > 0 and self.sessions[sid].active_node != nid]
            if not candidates:
                return False
            if self.policy == "repkv":
                victim = min(candidates, key=lambda sid: self._retention_value(sid, node, now))
            else:
                victim = min(candidates, key=lambda sid: node.cache[sid].last_used)
            entry = node.cache[victim]
            amount = min(self.cfg.block_batch, entry.blocks, node.used + extra - node.capacity)
            prepared_evicted = min(amount, entry.prepared_blocks)
            self.metrics.evicted_prepared_blocks += prepared_evicted
            entry.prepared_blocks -= prepared_evicted
            entry.blocks -= amount
            if entry.blocks == 0:
                del node.cache[victim]
        return True

    def _retention_value(self, sid: int, node: Node, now: float) -> float:
        state = self.sessions[sid]
        previous = self.turn_lookup.get((sid, state.completed_turn))
        if previous is None or previous.predicted_arrival is None:
            return 0.0
        time_to_return = max(0.0, previous.predicted_arrival - now)
        entry = node.cache[sid]
        # Higher probability, nearer return, and scarcer copies make KV harder to remove.
        copies = sum(1 for n in self.nodes if n.cache.get(sid, CacheEntry()).blocks >= entry.blocks)
        return previous.return_probability / (1.0 + time_to_return) / max(copies, 1)


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
    fields = [
        "slo_goodput_rps",
        "slo_attainment",
        "p99_ttft_s",
        "transfer_blocks_per_success",
        "recompute_blocks_per_success",
        "hbm_block_seconds_per_success",
        "unused_preparation_ratio",
    ]
    rows = []
    for policy in POLICIES:
        subset = [row for row in summaries if row["policy"] == policy]
        result: dict[str, float | str] = {"policy": policy}
        for field_name in fields:
            values = [float(row[field_name]) for row in subset]
            result[field_name] = statistics.mean(values)
        rows.append(result)
    return rows


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
