"""Paired experiment for quantifying the effect of the KV manager.

The routing rule is fixed to request-level exhaustive minimum-E2E selection.
For every random seed, one generated request trace is replayed twice:

* ``greedy``: passive, whole-state recovery without proactive placement;
* ``greedy_kv``: proactive placement plus block-level residual recovery.

This isolates the total system effect of the KV manager while keeping the
router, workload, model, hardware, and network identical within each pair.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .config import DEFAULT_CONFIG_PATH, load_config
from .data_generator import DataGenerator, LengthDistribution
from .large_model import get_model
from .router import Policy, simulate_trace


MODEL_NAME = "Qwen2-VL-7B-Instruct"
POLICIES = (Policy.NEAREST, Policy.GREEDY, Policy.GREEDY_KV)
QUEUE_GAP_BINS = (0, 50, 100, 150, 200, 250, 300, 400, 600, math.inf)


def percentile(values: Iterable[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, round((len(ordered) - 1) * p / 100))
    return float(ordered[index])


def mean_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "ci95_half": 0.0}
    mean = statistics.fmean(values)
    half = 0.0
    if len(values) > 1:
        # Student-t critical values for the experiment's common seed counts.
        critical = {3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
        half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return {
        "mean": mean,
        "min": min(values),
        "max": max(values),
        "ci95_half": half,
    }


def build_config(concurrency: int, seed: int):
    cfg = copy.deepcopy(load_config(DEFAULT_CONFIG_PATH))
    cfg.models = [model for model in cfg.models if model.name == MODEL_NAME]
    cfg.workload.groups = [
        group for group in cfg.workload.groups if group.model_name == MODEL_NAME
    ]
    group = cfg.workload.groups[0]
    group.entry_mode = "ratios"
    group.entry_concurrency = None
    group.entry_ratios = [1.0, 1.0, 2.0]
    group.concurrency = concurrency
    group.inter_turn_mean_ms = 5000.0
    group.turns_mean = 40.0
    group.turns_min = 32
    group.turns_max = 64
    group.turns_dist = LengthDistribution("lognormal", 40, 10, 32, 64)
    group.num_frames = 4

    cfg.workload.duration_ms = 120000.0
    cfg.workload.session_start_spread_frac = 0.8
    cfg.workload.seed = seed
    cfg.workload.mobility_start_frac = 0.1
    cfg.workload.mobility_ratio = 0.2
    cfg.workload.mobility_granularity = "markov"
    cfg.workload.mobility_residency_turns = 8
    cfg.router.gamma = 1.0
    cfg.apply()
    return cfg


def _simulate(policy: Policy, concurrency: int, seed: int) -> Dict:
    cfg = build_config(concurrency, seed)
    requests = DataGenerator(cfg.workload).generate()
    cluster = cfg.cluster
    return simulate_trace(
        policy,
        requests,
        get_model(MODEL_NAME),
        cfg.hardware,
        cfg.new_network(),
        num_nodes=cluster.num_nodes,
        staleness_ms=cluster.staleness_ms,
        gamma=cfg.router.gamma,
        decode_batch_size=cfg.router.decode_batch_size,
        sla_margin_ms=cfg.router.sla_margin_ms,
        reject_intrinsically_infeasible=cfg.router.reject_intrinsically_infeasible,
        collect_records=True,
        kv_capacity_bytes=cluster.kv_capacity_bytes,
        activation_reserve_bytes=cluster.activation_reserve_bytes,
        prefill_batch_size=cluster.prefill_batch_size,
        sla_slack_scheduling=cluster.sla_slack_scheduling,
        token_id_bytes=cfg.router.token_id_bytes,
        request_overhead_bytes=cfg.router.request_overhead_bytes,
        response_overhead_bytes=cfg.router.response_overhead_bytes,
        visual_bytes_per_token=cfg.router.visual_bytes_per_token,
        kv_manager_config=cfg.kv_manager,
        long_term_block_level_kv=cfg.router.long_term_block_level_kv,
        long_term_kv_block_level_kv=cfg.router.long_term_kv_block_level_kv,
    )


def _recovery_events(result: Dict) -> List[Dict[str, float]]:
    events: List[Dict[str, float]] = []
    for record in result["records"]:
        if record["mode"] not in ("migrate", "recompute"):
            continue
        if record["complete_kv_owner_e2e_ms"] <= 0:
            continue
        queue_gap = (
            float(record["complete_kv_owner_queue_ms"])
            - float(record["t_queue"])
        )
        recovery = float(record["t_state"])
        total_advantage = (
            float(record["complete_kv_owner_e2e_ms"])
            - float(record["e2e"])
        )
        events.append({
            "queue_gap_ms": queue_gap,
            "recovery_ms": recovery,
            "queue_minus_recovery_ms": queue_gap - recovery,
            "total_advantage_ms": total_advantage,
            "other_cost_advantage_ms": total_advantage - (queue_gap - recovery),
        })
    return events


def _compact(result: Dict, concurrency: int, seed: int, policy: Policy) -> Dict:
    records = result["records"]
    return {
        "concurrency": concurrency,
        "seed": seed,
        "policy": policy.value,
        "requests": int(result["num_requests"]),
        "avg_e2e_ms": float(result["avg_e2e_ms"]),
        "avg_ttft_ms": statistics.fmean(float(row["ttft"]) for row in records),
        "avg_queue_ms": float(result["avg_queue_ms"]),
        "avg_state_ms": float(result["avg_state_ms"]),
        "sla_violation_ratio": float(result["sla_violation_ratio"]),
        "cross_node_ratio": float(result["cross_node_ratio"]),
        "migrate_count": int(result["migrate_count"]),
        "recompute_count": int(result["recompute_count"]),
        "placement_completed_bytes_mb": float(
            result["placement_completed_bytes_mb"]
        ),
        "placement_completed_tasks": int(result["placement_completed_tasks"]),
        "recovery_events": _recovery_events(result),
    }


def worker(job: Tuple[int, int, str]) -> Dict:
    concurrency, seed, policy_value = job
    policy = Policy(policy_value)
    return _compact(_simulate(policy, concurrency, seed), concurrency, seed, policy)


def _event_summary(events: List[Dict[str, float]]) -> Dict:
    output: Dict = {"count": len(events), "queue_gap_bins": []}
    if not events:
        return output
    for key in events[0]:
        values = [event[key] for event in events]
        output[key] = {
            "mean": statistics.fmean(values),
            "p25": percentile(values, 25),
            "p50": percentile(values, 50),
            "p75": percentile(values, 75),
            "min": min(values),
            "max": max(values),
        }
    output["within_20ms_of_boundary_ratio"] = statistics.fmean(
        event["total_advantage_ms"] <= 20.0 for event in events
    )
    for lower, upper in zip(QUEUE_GAP_BINS, QUEUE_GAP_BINS[1:]):
        selected = [
            event for event in events
            if lower <= event["queue_gap_ms"] < upper
        ]
        if not selected:
            continue
        recovery = [event["recovery_ms"] for event in selected]
        output["queue_gap_bins"].append({
            "lower_ms": lower,
            "upper_ms": None if math.isinf(upper) else upper,
            "count": len(selected),
            "share": len(selected) / len(events),
            "mean_queue_gap_ms": statistics.fmean(
                event["queue_gap_ms"] for event in selected
            ),
            "mean_recovery_ms": statistics.fmean(recovery),
            "recovery_p25_ms": percentile(recovery, 25),
            "recovery_p75_ms": percentile(recovery, 75),
        })
    return output


def aggregate(runs: List[Dict]) -> Dict:
    grouped: Dict[Tuple[int, str], List[Dict]] = {}
    for run in runs:
        grouped.setdefault((run["concurrency"], run["policy"]), []).append(run)

    output: Dict = {"by_concurrency": {}, "paired_effect": {}}
    metrics = (
        "requests", "avg_e2e_ms", "avg_ttft_ms", "avg_queue_ms",
        "avg_state_ms", "sla_violation_ratio", "cross_node_ratio", "migrate_count",
        "recompute_count", "placement_completed_bytes_mb",
        "placement_completed_tasks",
    )
    for (concurrency, policy), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["seed"])
        concurrency_out = output["by_concurrency"].setdefault(
            str(concurrency), {}
        )
        policy_out = {
            metric: mean_summary([float(row[metric]) for row in rows])
            for metric in metrics
        }
        events = [event for row in rows for event in row["recovery_events"]]
        policy_out["selected_recovery_events"] = _event_summary(events)
        concurrency_out[policy] = policy_out

    for concurrency in sorted({run["concurrency"] for run in runs}):
        passive = {
            run["seed"]: run for run in runs
            if run["concurrency"] == concurrency and run["policy"] == "greedy"
        }
        active = {
            run["seed"]: run for run in runs
            if run["concurrency"] == concurrency and run["policy"] == "greedy_kv"
        }
        seeds = sorted(set(passive) & set(active))
        paired = {}
        for metric in ("avg_e2e_ms", "avg_ttft_ms", "avg_queue_ms", "avg_state_ms"):
            absolute = [passive[s][metric] - active[s][metric] for s in seeds]
            relative = [
                1.0 - active[s][metric] / passive[s][metric]
                if passive[s][metric] else 0.0
                for s in seeds
            ]
            paired[metric] = {
                "absolute_reduction": mean_summary(absolute),
                "relative_reduction": mean_summary(relative),
            }
        output["paired_effect"][str(concurrency)] = paired
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", default="8,16,24,36,48")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--output",
        default="outputs/kv-manager-effect/summary.json",
    )
    args = parser.parse_args()
    concurrencies = [int(value) for value in args.concurrency.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    jobs = [
        (concurrency, seed, policy.value)
        for concurrency in concurrencies
        for seed in seeds
        for policy in POLICIES
    ]

    runs: List[Dict] = []
    if args.workers <= 1:
        for index, job in enumerate(jobs, 1):
            runs.append(worker(job))
            print(f"progress {index}/{len(jobs)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(worker, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), 1):
                runs.append(future.result())
                print(f"progress {index}/{len(futures)}", flush=True)

    output = {
        "experiment": {
            "model": MODEL_NAME,
            "duration_ms": 120000.0,
            "seeds": seeds,
            "concurrencies": concurrencies,
            "entry_ratios": [1.0, 1.0, 2.0],
            "mobility_start_fraction": 0.1,
            "mobility_probability": 0.2,
            "minimum_residency_turns": 8,
            "session_turn_distribution": {
                "kind": "lognormal", "mean": 40, "std": 10,
                "minimum": 32, "maximum": 64,
            },
            "comparison": {
                "nearest": "nearest-entry execution reference",
                "greedy": "request-level exhaustive routing; KV manager off",
                "greedy_kv": (
                    "same request-level exhaustive routing; proactive placement "
                    "and block-level residual recovery on"
                ),
            },
        },
        "runs": sorted(
            runs,
            key=lambda row: (row["concurrency"], row["seed"], row["policy"]),
        ),
        "aggregate": aggregate(runs),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT_JSON={path.resolve()}")


if __name__ == "__main__":
    main()
