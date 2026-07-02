"""Workload trace generator for the offloading simulation.

Produces a reproducible list of requests organised as sessions, supporting
LLM / VLM / VLA model roles, preset (sampled) generation lengths, prefix reuse
and post-midpoint user-mobility entry switching.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from .large_model import (
    LengthDistributionSpec,
    ModelType,
    ModelSpec,
    get_model,
)


@dataclass
class LengthDistribution:
    kind: str = "lognormal"  # fixed | normal | lognormal
    mean: float = 128.0
    std: float = 64.0
    minimum: int = 1
    maximum: int = 4096

    @classmethod
    def from_spec(cls, spec: LengthDistributionSpec) -> "LengthDistribution":
        return cls(
            kind=spec.kind,
            mean=spec.mean,
            std=spec.std,
            minimum=spec.minimum,
            maximum=spec.maximum,
        )

    def sample(self, rng: random.Random) -> int:
        if self.kind == "fixed":
            value = self.mean
        elif self.kind == "normal":
            value = rng.gauss(self.mean, self.std)
        elif self.kind == "lognormal":
            mean = max(self.mean, 1e-6)
            std = max(self.std, 1e-6)
            sigma2 = math.log(1.0 + (std * std) / (mean * mean))
            sigma = math.sqrt(sigma2)
            mu = math.log(mean) - 0.5 * sigma2
            value = math.exp(rng.gauss(mu, sigma))
        else:
            raise ValueError(f"unknown distribution kind {self.kind!r}")
        return int(max(self.minimum, min(self.maximum, round(value))))


@dataclass
class Request:
    request_id: int
    session_id: str
    model_name: str
    model_type: str
    arrival_ms: float
    entry_node: int
    priority: str
    group_name: str
    sla_ms: float

    prompt_text_tokens: int
    visual_tokens: int
    state_tokens: int
    input_tokens: int
    output_len: int

    prefix_id: str
    prefix_tokens: int
    is_session_first: bool
    turn_index: int
    expected_session_turns: float

    home_node: int = 0
    mobility_switched: bool = False
    mobility_transitioned: bool = False
    mobility_granularity: str = "request"
    mobility_ratio: float = 0.0
    mobility_residency_turns: int = 0
    mobility_active: bool = False
    mobility_residency_age: int = 0
    mobility_start_ms: float = 0.0
    expected_interarrival_ms: float = 0.0


@dataclass
class Session:
    session_id: str
    model_name: str
    num_turns: int
    home_node: int
    prefix_id: str
    prefix_tokens: int
    mobility_switched: bool = False


@dataclass
class WorkloadGroup:
    model_name: str
    name: str = "default"
    entry_mode: str = "counts"  # counts | ratios
    priority: str = "normal"
    concurrency: int = 24
    entry_concurrency: Optional[List[int]] = None
    entry_ratios: Optional[List[float]] = None
    sla_ms: Optional[float] = None
    arrival_rate: Optional[float] = None  # requests/sec; derived if None
    prompt_dist: LengthDistribution = field(
        default_factory=lambda: LengthDistribution("lognormal", 512, 384, 16, 4096)
    )
    output_dist: Optional[LengthDistribution] = None
    turns_mean: float = 4.0
    turns_min: int = 1
    turns_max: int = 12
    image_size: Tuple[int, int] = (0, 0)
    num_frames: int = 1
    shared_prefix_tokens: int = 0
    history_growth: float = 0.6  # fraction of prior turn carried into prefix


@dataclass
class WorkloadConfig:
    groups: List[WorkloadGroup]
    num_nodes: int = 3
    duration_ms: float = 60000.0
    session_start_spread_frac: float = 0.8
    mobility_start_frac: float = 0.5
    mobility_ratio: float = 0.2
    mobility_granularity: str = "request"  # request | session | markov
    mobility_residency_turns: int = 2
    seed: int = 0

    @staticmethod
    def default_experiment() -> "WorkloadConfig":
        code_prompt = LengthDistribution("lognormal", 800, 600, 32, 4096)
        vlm_prompt = LengthDistribution("lognormal", 128, 96, 8, 1024)
        vla_prompt = LengthDistribution("fixed", 32, 0, 32, 32)
        vla_output = LengthDistribution("fixed", 7, 0, 7, 7)
        return WorkloadConfig(
            groups=[
                WorkloadGroup(
                    model_name="CodeLlama34B",
                    name="high",
                    priority="high",
                    concurrency=72,
                    entry_concurrency=[16, 24, 32],
                    sla_ms=150.0,
                    prompt_dist=code_prompt,
                    turns_mean=3.0,
                    shared_prefix_tokens=256,
                ),
                WorkloadGroup(
                    model_name="CodeLlama34B",
                    name="normal",
                    priority="normal",
                    concurrency=288,
                    entry_concurrency=[64, 96, 128],
                    sla_ms=500.0,
                    prompt_dist=code_prompt,
                    turns_mean=5.0,
                    shared_prefix_tokens=256,
                ),
                WorkloadGroup(
                    model_name="Qwen2-VL-7B-Instruct",
                    name="default",
                    priority="normal",
                    concurrency=72,
                    entry_concurrency=[16, 24, 32],
                    sla_ms=500.0,
                    prompt_dist=vlm_prompt,
                    turns_mean=3.0,
                    image_size=(1024, 768),
                    shared_prefix_tokens=64,
                ),
                WorkloadGroup(
                    model_name="OpenVLA-7B",
                    name="default",
                    priority="normal",
                    concurrency=24,
                    entry_concurrency=[8, 8, 8],
                    sla_ms=200.0,
                    prompt_dist=vla_prompt,
                    output_dist=vla_output,
                    turns_mean=3.0,
                    image_size=(224, 224),
                    shared_prefix_tokens=0,
                    history_growth=0.0,
                ),
            ],
            num_nodes=3,
            duration_ms=60000.0,
            mobility_start_frac=0.5,
            mobility_ratio=0.2,
            mobility_residency_turns=2,
            seed=0,
        )


class DataGenerator:
    def __init__(self, config: WorkloadConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self._model_cache: Dict[str, ModelSpec] = {}

    def _model(self, name: str) -> ModelSpec:
        if name not in self._model_cache:
            self._model_cache[name] = get_model(name)
        return self._model_cache[name]

    def _sample_turns(self, group: WorkloadGroup) -> int:
        value = self.rng.gauss(group.turns_mean, max(group.turns_mean * 0.4, 1.0))
        return int(max(group.turns_min, min(group.turns_max, round(value))))

    def _arrival_rate(self, group: WorkloadGroup) -> float:
        if group.arrival_rate is not None:
            return group.arrival_rate
        # derive a steady arrival rate from concurrency over the run duration
        seconds = max(self.config.duration_ms / 1000.0, 1e-6)
        return group.concurrency / seconds

    def _entry_session_counts(self, group: WorkloadGroup, node_count: int) -> List[int]:
        if group.entry_mode == "ratios" or group.entry_ratios:
            total = max(
                node_count,
                min(256 * node_count, int(round(group.concurrency))),
            )
            ratios = [
                max(float(v), 0.0)
                for v in (group.entry_ratios or [])[:node_count]
            ]
            ratios += [0.0] * (node_count - len(ratios))
            if sum(ratios) <= 0:
                ratios = [1.0] * node_count
            weight_sum = sum(ratios)
            remaining = total - node_count
            raw = [remaining * value / weight_sum for value in ratios]
            counts = [1 + min(255, int(math.floor(value))) for value in raw]
            left = total - sum(counts)
            order = sorted(
                range(node_count),
                key=lambda i: raw[i] - math.floor(raw[i]),
                reverse=True,
            )
            while left > 0:
                changed = False
                for i in order:
                    if counts[i] >= 256:
                        continue
                    counts[i] += 1
                    left -= 1
                    changed = True
                    if left == 0:
                        break
                if not changed:
                    break
            return counts
        if group.entry_concurrency:
            counts = [
                max(1, min(256, int(round(v))))
                for v in group.entry_concurrency[:node_count]
            ]
            if len(counts) < node_count:
                counts += [1] * (node_count - len(counts))
            return counts
        total = max(int(round(group.concurrency)), 1)
        base = total // node_count
        rem = total % node_count
        return [max(1, min(256, base + (1 if i < rem else 0))) for i in range(node_count)]

    def generate(self) -> List[Request]:
        requests: List[Request] = []
        node_count = max(self.config.num_nodes, 1)
        session_counter = 0

        for gi, group in enumerate(self.config.groups):
            model = self._model(group.model_name)
            sla = group.sla_ms if group.sla_ms is not None else model.default_sla_ms
            out_dist = group.output_dist or LengthDistribution.from_spec(
                model.default_output_dist
            )
            entry_counts = self._entry_session_counts(group, node_count)
            group.concurrency = sum(entry_counts)
            rate = self._arrival_rate(group)

            for home_node, n_sessions in enumerate(entry_counts):
                for _ in range(n_sessions):
                    session_counter += 1
                    sid = f"g{gi}-s{session_counter}"
                    num_turns = self._sample_turns(group)
                    prefix_id = f"{group.model_name}:{sid}"

                    # session start time spread across the run
                    session_start = self.rng.uniform(
                        0.0,
                        self.config.duration_ms
                        * max(0.0, min(self.config.session_start_spread_frac, 1.0)),
                    )
                    clock = session_start
                    carried_tokens = group.shared_prefix_tokens

                    for turn in range(num_turns):
                        if model.model_type == ModelType.VLA:
                            prompt_text = int(round(group.prompt_dist.mean))
                        else:
                            prompt_text = group.prompt_dist.sample(self.rng)
                        vis_tokens = model.visual_tokens(
                            group.image_size[0],
                            group.image_size[1],
                            group.num_frames,
                        )
                        state_tokens = 8 if model.model_type == ModelType.VLA else 0
                        input_tokens = model.input_tokens(
                            prompt_text,
                            image_width=group.image_size[0],
                            image_height=group.image_size[1],
                            num_frames=group.num_frames,
                            state_tokens=state_tokens,
                        )
                        if model.model_type == ModelType.VLA:
                            output_len = int(round(out_dist.mean))
                        else:
                            output_len = out_dist.sample(self.rng)

                        prefix_tokens = int(carried_tokens)

                        if turn == 0:
                            inter_arrival = 0.0
                        else:
                            inter_arrival = self.rng.expovariate(
                                max(rate, 1e-6)
                            ) * 1000.0
                        clock += inter_arrival
                        if clock > self.config.duration_ms:
                            break

                        requests.append(
                            Request(
                                request_id=-1,
                                session_id=sid,
                                model_name=group.model_name,
                                model_type=model.model_type.value,
                                arrival_ms=clock,
                                entry_node=home_node,
                                priority=group.name,
                                group_name=group.name,
                                sla_ms=sla,
                                prompt_text_tokens=prompt_text,
                                visual_tokens=vis_tokens,
                                state_tokens=state_tokens,
                                input_tokens=input_tokens,
                                output_len=output_len,
                                prefix_id=prefix_id,
                                prefix_tokens=prefix_tokens,
                                is_session_first=(turn == 0),
                                turn_index=turn,
                                expected_session_turns=group.turns_mean,
                                home_node=home_node,
                                mobility_granularity=self.config.mobility_granularity,
                                mobility_ratio=self.config.mobility_ratio,
                                mobility_residency_turns=(
                                    self.config.mobility_residency_turns
                                ),
                                mobility_active=(clock >= (
                                    self.config.duration_ms
                                    * self.config.mobility_start_frac
                                )),
                                mobility_start_ms=(
                                    self.config.duration_ms
                                    * self.config.mobility_start_frac
                                ),
                                expected_interarrival_ms=(
                                    1000.0 / max(rate, 1e-6)
                                ),
                            )
                        )

                        # next turn reuses this turn's context as growing prefix
                        carried_tokens = (
                            prefix_tokens
                            + int(group.history_growth * (input_tokens + output_len))
                        )

        self._apply_mobility(requests, node_count)
        requests.sort(key=lambda r: r.arrival_ms)
        for idx, req in enumerate(requests):
            req.request_id = idx
        return requests

    def _apply_mobility(self, requests: List[Request], node_count: int) -> None:
        if node_count <= 1 or self.config.mobility_ratio <= 0:
            return
        switch_after = self.config.duration_ms * self.config.mobility_start_frac

        if self.config.mobility_granularity == "session":
            sessions = sorted({r.session_id for r in requests})
            home_by_session = {r.session_id: r.home_node for r in requests}
            switched = [
                s for s in sessions if self.rng.random() < self.config.mobility_ratio
            ]
            switched_set = set(switched)
            session_entry = {
                s: self._pick_new_entry(home_by_session[s], node_count)
                for s in switched
            }
            transitioned = set()
            for r in requests:
                if r.arrival_ms >= switch_after and r.session_id in switched_set:
                    r.entry_node = session_entry[r.session_id]
                    r.mobility_switched = True
                    if r.session_id not in transitioned:
                        r.mobility_transitioned = True
                        transitioned.add(r.session_id)
        elif self.config.mobility_granularity == "markov":
            by_session: Dict[str, List[Request]] = {}
            for r in requests:
                by_session.setdefault(r.session_id, []).append(r)
            residency = max(int(self.config.mobility_residency_turns), 0)
            for rows in by_session.values():
                rows.sort(key=lambda r: r.turn_index)
                current = rows[0].home_node
                turns_at_current = 0
                for r in rows:
                    if r.arrival_ms < switch_after:
                        r.entry_node = current
                        turns_at_current += 1
                        r.mobility_residency_age = turns_at_current
                        continue
                    # Draw on every eligible-time request so changing residency
                    # does not shift the random stream for later sessions.
                    move_roll = self.rng.random()
                    proposed_entry = self._pick_new_entry(current, node_count)
                    if (
                        turns_at_current >= residency
                        and move_roll < self.config.mobility_ratio
                    ):
                        current = proposed_entry
                        turns_at_current = 0
                        r.mobility_transitioned = True
                    if current != r.home_node:
                        r.mobility_switched = True
                    r.entry_node = current
                    turns_at_current += 1
                    r.mobility_residency_age = turns_at_current
        else:
            for r in requests:
                if (
                    r.arrival_ms >= switch_after
                    and self.rng.random() < self.config.mobility_ratio
                ):
                    self._switch_entry(r, node_count)

    def _pick_new_entry(self, home_node: int, node_count: int) -> int:
        candidates = [n for n in range(node_count) if n != home_node]
        if not candidates:
            return home_node
        return self.rng.choice(candidates)

    def _switch_entry(
        self, req: Request, node_count: int, new_entry: Optional[int] = None
    ) -> None:
        candidates = [n for n in range(node_count) if n != req.home_node]
        if not candidates:
            return
        req.entry_node = new_entry if new_entry is not None else self.rng.choice(candidates)
        req.mobility_switched = True
        req.mobility_transitioned = True

    def summary(self, requests: List[Request]) -> dict:
        if not requests:
            return {"num_requests": 0}

        def pct(values: List[float], p: float) -> float:
            if not values:
                return 0.0
            s = sorted(values)
            k = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
            return s[k]

        by_model: Dict[str, int] = {}
        by_priority: Dict[str, int] = {}
        prompts = [r.input_tokens for r in requests]
        outputs = [r.output_len for r in requests]
        moved = sum(1 for r in requests if r.mobility_switched)
        sessions = {r.session_id for r in requests}
        kv_tokens = sum(r.input_tokens + r.output_len for r in requests)

        for r in requests:
            by_model[r.model_name] = by_model.get(r.model_name, 0) + 1
            by_priority[r.priority] = by_priority.get(r.priority, 0) + 1

        return {
            "num_requests": len(requests),
            "num_sessions": len(sessions),
            "by_model": by_model,
            "by_priority": by_priority,
            "input_tokens_avg": sum(prompts) / len(prompts),
            "input_tokens_p95": pct(prompts, 95),
            "output_len_avg": sum(outputs) / len(outputs),
            "output_len_p95": pct(outputs, 95),
            "mobility_ratio": moved / len(requests),
            "total_kv_tokens": kv_tokens,
            "duration_ms": self.config.duration_ms,
        }

    def to_jsonl(self, requests: List[Request], path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for r in requests:
                fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    gen = DataGenerator(WorkloadConfig.default_experiment())
    reqs = gen.generate()
    summary = gen.summary(reqs)
    print("workload summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("\nfirst 3 requests:")
    for r in reqs[:3]:
        print(
            f"  #{r.request_id} {r.model_name}/{r.priority} entry={r.entry_node} "
            f"in={r.input_tokens} out={r.output_len} sla={r.sla_ms}ms "
            f"prefix={r.prefix_tokens} moved={r.mobility_switched}"
        )
