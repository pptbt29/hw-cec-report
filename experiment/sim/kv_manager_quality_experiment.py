"""Component ablation and capability sweep for the KV manager.

All variants use the same request-level exhaustive Router and replay identical
request traces within each ``(concurrency, seed)`` pair.  The experiment
separates proactive prefix placement from block-level residual recovery, then
increases placement aggressiveness and finally adds a zero-residual upper
bound.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .data_generator import DataGenerator
from .kv_manager_effect_experiment import (
    MODEL_NAME,
    build_config,
    mean_summary,
    percentile,
)
from .large_model import get_model
from .router import Policy, simulate_trace


@dataclass(frozen=True)
class ManagerLevel:
    name: str
    proactive: bool
    block_level: bool
    base_replication_factor: float
    max_replication_fraction: float
    background_bandwidth_fraction: float
    state_recovery_scale: float = 1.0
    description: str = ""


LEVELS: Tuple[ManagerLevel, ...] = (
    ManagerLevel(
        "off", False, False, 0.0, 0.0, 0.0,
        description="no proactive placement; whole-state passive recovery",
    ),
    ManagerLevel(
        "block_only", False, True, 0.0, 0.0, 0.0,
        description="block-level residual recovery without proactive placement",
    ),
    ManagerLevel(
        "proactive_only", True, False, 2.0, 0.50, 0.20,
        description="current proactive placement with whole-state recovery",
    ),
    ManagerLevel(
        "weak", True, True, 1.0, 0.25, 0.05,
        description="block recovery plus conservative proactive placement",
    ),
    ManagerLevel(
        "current", True, True, 2.0, 0.50, 0.20,
        description="current block recovery and proactive placement settings",
    ),
    ManagerLevel(
        "strong", True, True, 4.0, 1.00, 0.50,
        description="aggressive placement with up to a full missing prefix",
    ),
    ManagerLevel(
        "ideal_upper", True, True, 4.0, 1.00, 0.50, 0.0,
        description="strong placement with zero residual recovery latency",
    ),
)
LEVEL_BY_NAME = {level.name: level for level in LEVELS}


def _recovery_summary(records: List[Dict]) -> Dict[str, float]:
    values = [
        float(record["t_state"])
        for record in records
        if record["mode"] in ("migrate", "recompute")
        and record["complete_kv_owner_e2e_ms"] > 0
    ]
    if not values:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
    }


def simulate_level(concurrency: int, seed: int, level: ManagerLevel) -> Dict:
    cfg = build_config(concurrency, seed)
    requests = DataGenerator(cfg.workload).generate()
    cluster = cfg.cluster
    manager = replace(
        cfg.kv_manager,
        enabled=level.proactive,
        base_replication_factor=level.base_replication_factor,
        max_replication_fraction=level.max_replication_fraction,
        background_bandwidth_fraction=level.background_bandwidth_fraction,
    )
    # GREEDY_KV is used as the experimental carrier whenever either manager
    # capability is enabled.  The two capabilities remain independently
    # controlled by ``manager.enabled`` and ``long_term_kv_block_level_kv``.
    policy = Policy.GREEDY_KV if (level.proactive or level.block_level) else Policy.GREEDY
    result = simulate_trace(
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
        kv_manager_config=manager,
        long_term_block_level_kv=False,
        long_term_kv_block_level_kv=level.block_level,
        state_recovery_scale=level.state_recovery_scale,
    )
    records = result["records"]
    return {
        "concurrency": concurrency,
        "seed": seed,
        "level": level.name,
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
        "selected_recovery": _recovery_summary(records),
    }


def worker(job: Tuple[int, int, str]) -> Dict:
    concurrency, seed, level_name = job
    return simulate_level(concurrency, seed, LEVEL_BY_NAME[level_name])


def aggregate(runs: List[Dict]) -> Dict:
    metrics = (
        "requests", "avg_e2e_ms", "avg_ttft_ms", "avg_queue_ms",
        "avg_state_ms", "sla_violation_ratio", "cross_node_ratio",
        "migrate_count", "recompute_count", "placement_completed_bytes_mb",
        "placement_completed_tasks",
    )
    grouped: Dict[Tuple[int, str], List[Dict]] = {}
    for run in runs:
        grouped.setdefault((run["concurrency"], run["level"]), []).append(run)

    by_concurrency: Dict[str, Dict] = {}
    for (concurrency, level), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["seed"])
        output = {
            metric: mean_summary([float(row[metric]) for row in rows])
            for metric in metrics
        }
        recovery_counts = [row["selected_recovery"]["count"] for row in rows]
        output["selected_recovery_count"] = mean_summary(recovery_counts)
        # Event-weighted descriptions are mechanism diagnostics, not seed CIs.
        total_events = sum(recovery_counts)
        output["selected_recovery_event_weighted"] = {
            "count": total_events,
            "mean_ms": (
                sum(
                    row["selected_recovery"]["mean_ms"]
                    * row["selected_recovery"]["count"]
                    for row in rows
                ) / total_events
                if total_events else 0.0
            ),
        }
        by_concurrency.setdefault(str(concurrency), {})[level] = output

    paired_vs_off: Dict[str, Dict] = {}
    for concurrency in sorted({run["concurrency"] for run in runs}):
        rows = [run for run in runs if run["concurrency"] == concurrency]
        indexed = {(row["seed"], row["level"]): row for row in rows}
        seeds = sorted({row["seed"] for row in rows})
        concurrency_out = {}
        for level in sorted({row["level"] for row in rows}):
            if level == "off":
                continue
            level_out = {}
            for metric in (
                "avg_e2e_ms", "avg_ttft_ms", "avg_queue_ms",
                "avg_state_ms", "sla_violation_ratio",
            ):
                absolute = [
                    indexed[(seed, "off")][metric]
                    - indexed[(seed, level)][metric]
                    for seed in seeds
                ]
                relative = [
                    1.0 - indexed[(seed, level)][metric]
                    / indexed[(seed, "off")][metric]
                    if indexed[(seed, "off")][metric] else 0.0
                    for seed in seeds
                ]
                level_out[metric] = {
                    "absolute_reduction": mean_summary(absolute),
                    "relative_reduction": mean_summary(relative),
                }
            concurrency_out[level] = level_out
        paired_vs_off[str(concurrency)] = concurrency_out
    return {"by_concurrency": by_concurrency, "paired_vs_off": paired_vs_off}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", default="16,24,36")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--levels",
        default=",".join(level.name for level in LEVELS),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output", default="outputs/kv-manager-quality/summary.json"
    )
    args = parser.parse_args()
    concurrencies = [int(value) for value in args.concurrency.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    levels = [value for value in args.levels.split(",") if value]
    unknown = sorted(set(levels) - set(LEVEL_BY_NAME))
    if unknown:
        raise ValueError(f"unknown KV manager levels: {unknown}")
    jobs = [
        (concurrency, seed, level)
        for concurrency in concurrencies
        for seed in seeds
        for level in levels
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
            "concurrencies": concurrencies,
            "seeds": seeds,
            "levels": {
                level.name: {
                    "proactive": level.proactive,
                    "block_level": level.block_level,
                    "base_replication_factor": level.base_replication_factor,
                    "max_replication_fraction": level.max_replication_fraction,
                    "background_bandwidth_fraction": (
                        level.background_bandwidth_fraction
                    ),
                    "state_recovery_scale": level.state_recovery_scale,
                    "description": level.description,
                }
                for level in LEVELS if level.name in levels
            },
        },
        "runs": sorted(
            runs,
            key=lambda row: (row["concurrency"], row["seed"], row["level"]),
        ),
        "aggregate": aggregate(runs),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT_JSON={path.resolve()}")


if __name__ == "__main__":
    main()
